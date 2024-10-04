import Mathlib

namespace circle_with_all_three_colors_l144_144645

-- Define color type using an inductive type with three colors
inductive Color
| red
| green
| blue

-- Define a function that assigns a color to each point in the plane
def color_function (point : ℝ × ℝ) : Color := sorry

-- Define the main theorem stating that for any coloring, there exists a circle that contains points of all three colors
theorem circle_with_all_three_colors (color_func : ℝ × ℝ → Color) (exists_red : ∃ p : ℝ × ℝ, color_func p = Color.red)
                                      (exists_green : ∃ p : ℝ × ℝ, color_func p = Color.green) 
                                      (exists_blue : ∃ p : ℝ × ℝ, color_func p = Color.blue) :
    ∃ (c : ℝ × ℝ) (r : ℝ), ∃ p1 p2 p3 : ℝ × ℝ, 
             color_func p1 = Color.red ∧ color_func p2 = Color.green ∧ color_func p3 = Color.blue ∧ 
             (dist p1 c = r) ∧ (dist p2 c = r) ∧ (dist p3 c = r) :=
by 
  sorry

end circle_with_all_three_colors_l144_144645


namespace sum_of_remainders_3_digit_numbers_l144_144144

theorem sum_of_remainders_3_digit_numbers :
  let a_1 := 102
  let a_n := 998
  let d := 3
  let n := (a_n - a_1) / d + 1
  let S_n := n / 2 * (a_1 + a_n)
  S_n = 164450 :=
by
  -- Definitions based on conditions
  let a_1 := 102
  let a_n := 998
  let d := 3
  let n := (a_n - a_1) / d + 1
  let S_n := n / 2 * (a_1 + a_n)
  -- Skip the proof
  sorry

end sum_of_remainders_3_digit_numbers_l144_144144


namespace arithmetic_sqrt_of_9_l144_144902

theorem arithmetic_sqrt_of_9 : (∃ (sqrt : ℝ), sqrt = 3 ∧ ∀ x, x*x = 9 → x = sqrt) :=
by
  existsi (3 : ℝ)
  split
  exact rfl
  intros x hx
  exact sqrt_unique hx 3

end arithmetic_sqrt_of_9_l144_144902


namespace strawberry_jelly_sales_l144_144011

def jelly_sales (grape strawberry raspberry plum : ℕ) : Prop :=
  grape = 2 * strawberry ∧
  raspberry = 2 * plum ∧
  raspberry = grape / 3 ∧
  plum = 6

theorem strawberry_jelly_sales {grape strawberry raspberry plum : ℕ}
    (h : jelly_sales grape strawberry raspberry plum) : 
    strawberry = 18 :=
by
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2.1
  have h4 := h.2.2.2
  sorry

end strawberry_jelly_sales_l144_144011


namespace power_expression_l144_144678

theorem power_expression (a b : ℕ) (h1 : a = 12) (h2 : b = 18) : (3^a * 3^b) = (243^6) :=
by
  let c := 3
  have h3 : a + b = 30 := by simp [h1, h2]
  have h4 : 3^(a + b) = 3^30 := by rw [h3]
  have h5 : 3^30 = 243^6 := by norm_num
  sorry  -- skip other detailed steps

end power_expression_l144_144678


namespace triangle_area_is_17_point_5_l144_144587

-- Define the points A, B, and C as tuples of coordinates
def A : (ℝ × ℝ) := (2, 2)
def B : (ℝ × ℝ) := (7, 2)
def C : (ℝ × ℝ) := (4, 9)

-- Function to calculate the area of a triangle given its vertices
noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1 / 2) * abs ((x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2)))

-- The theorem statement asserting the area of the triangle is 17.5 square units
theorem triangle_area_is_17_point_5 :
  area_of_triangle A B C = 17.5 :=
by
  sorry -- Proof is omitted

end triangle_area_is_17_point_5_l144_144587


namespace find_discriminant_l144_144366

variables {a b c : ℝ}
variables (P : ℝ → ℝ)
def is_quadratic_polynomial (P : ℝ → ℝ) : Prop := ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, P x = a * x^2 + b * x + c)

theorem find_discriminant (h1 : is_quadratic_polynomial P)
  (h2 : ∃ x, P x = x - 2)
  (h3 : ∃ y, P y = 1 - y / 2)
  : ∃ D, D = -1/2 := 
sorry

end find_discriminant_l144_144366


namespace find_natural_numbers_l144_144341

open Nat

theorem find_natural_numbers (n : ℕ) (h : ∃ m : ℤ, 2^n + 33 = m^2) : n = 4 ∨ n = 8 :=
sorry

end find_natural_numbers_l144_144341


namespace squirrel_count_l144_144425

theorem squirrel_count (n m : ℕ) (h1 : n = 12) (h2 : m = 12 + 12 / 3) : n + m = 28 := by
  sorry

end squirrel_count_l144_144425


namespace determine_120_percent_of_y_l144_144400

def x := 0.80 * 350
def y := 0.60 * x
def result := 1.20 * y

theorem determine_120_percent_of_y : result = 201.6 := by
  sorry

end determine_120_percent_of_y_l144_144400


namespace arithmetic_square_root_of_nine_l144_144927

theorem arithmetic_square_root_of_nine :
  real.sqrt 9 = 3 :=
by
  sorry

end arithmetic_square_root_of_nine_l144_144927


namespace papi_calot_plants_l144_144880

theorem papi_calot_plants : 
  let rows := 7
  let plants_per_row := 18
  let additional_plants := 15
  let initial_plants := rows * plants_per_row
in initial_plants + additional_plants = 141 := by
  sorry

end papi_calot_plants_l144_144880


namespace group_size_l144_144955

def total_blocks : ℕ := 820
def num_groups : ℕ := 82

theorem group_size :
  total_blocks / num_groups = 10 := 
by 
  sorry

end group_size_l144_144955


namespace triangle_inequality_l144_144050

theorem triangle_inequality (a b c : ℝ) (h : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a / (b + c) + b / (c + a) + c / (a + b) ≤ 2 :=
sorry

end triangle_inequality_l144_144050


namespace parabolas_intersect_l144_144335

theorem parabolas_intersect :
  let eq1 (x : ℝ) := 3 * x^2 - 4 * x + 2
  let eq2 (x : ℝ) := -x^2 + 6 * x + 8
  (∃ x y : ℝ, y = eq1 x ∧ y = eq2 x ∧ x = -0.5 ∧ y = 4.75) ∧
  (∃ x y : ℝ, y = eq1 x ∧ y = eq2 x ∧ x = 3 ∧ y = 17) :=
by sorry

end parabolas_intersect_l144_144335


namespace blue_marbles_count_l144_144713

theorem blue_marbles_count
  (total_marbles : ℕ)
  (yellow_marbles : ℕ)
  (red_marbles : ℕ)
  (blue_marbles : ℕ)
  (yellow_probability : ℚ)
  (total_marbles_eq : yellow_marbles = 6)
  (yellow_probability_eq : yellow_probability = 1 / 4)
  (red_marbles_eq : red_marbles = 11)
  (total_marbles_def : total_marbles = yellow_marbles * 4)
  (blue_marbles_def : blue_marbles = total_marbles - red_marbles - yellow_marbles) :
  blue_marbles = 7 :=
sorry

end blue_marbles_count_l144_144713


namespace extra_people_needed_l144_144174

theorem extra_people_needed 
  (initial_people : ℕ) 
  (initial_time : ℕ) 
  (final_time : ℕ) 
  (work_done : ℕ) 
  (all_paint_same_rate : initial_people * initial_time = work_done) :
  initial_people = 8 →
  initial_time = 3 →
  final_time = 2 →
  work_done = 24 →
  ∃ extra_people : ℕ, extra_people = 4 :=
by
  sorry

end extra_people_needed_l144_144174


namespace c_share_l144_144461

theorem c_share (A B C : ℕ) 
    (h1 : A = 1/2 * B) 
    (h2 : B = 1/2 * C) 
    (h3 : A + B + C = 406) : 
    C = 232 := by 
    sorry

end c_share_l144_144461


namespace newer_model_distance_l144_144786

-- Given conditions
def older_model_distance : ℕ := 160
def newer_model_factor : ℝ := 1.25

-- The statement to be proved
theorem newer_model_distance :
  newer_model_factor * (older_model_distance : ℝ) = 200 := by
  sorry

end newer_model_distance_l144_144786


namespace lower_limit_tip_percentage_l144_144601

namespace meal_tip

def meal_cost : ℝ := 35.50
def total_paid : ℝ := 40.825
def tip_limit : ℝ := 15

-- Define the lower limit tip percentage as the solution to the given conditions.
theorem lower_limit_tip_percentage :
  ∃ x : ℝ, x > 0 ∧ x < 25 ∧ (meal_cost + (x / 100) * meal_cost = total_paid) → 
  x = tip_limit :=
sorry

end meal_tip

end lower_limit_tip_percentage_l144_144601


namespace no_tiling_triminos_l144_144480

theorem no_tiling_triminos (board_size : ℕ) (trimino_size : ℕ) (remaining_squares : ℕ) 
  (H_board : board_size = 8) (H_trimino : trimino_size = 3) (H_remaining : remaining_squares = 63) : 
  ¬ ∃ (triminos : ℕ), triminos * trimino_size = remaining_squares :=
by {
  sorry
}

end no_tiling_triminos_l144_144480


namespace exists_pythagorean_number_in_range_l144_144850

def is_pythagorean_area (a : ℕ) : Prop :=
  ∃ (x y z : ℕ), x^2 + y^2 = z^2 ∧ a = (x * y) / 2

theorem exists_pythagorean_number_in_range (n : ℕ) (hn : n > 12) : 
  ∃ (m : ℕ), is_pythagorean_area m ∧ n < m ∧ m < 2 * n :=
sorry

end exists_pythagorean_number_in_range_l144_144850


namespace initial_deadline_in_days_l144_144553

theorem initial_deadline_in_days
  (men_initial : ℕ)
  (days_initial : ℕ)
  (hours_per_day_initial : ℕ)
  (fraction_work_initial : ℚ)
  (additional_men : ℕ)
  (hours_per_day_additional : ℕ)
  (fraction_work_additional : ℚ)
  (total_work : ℚ := men_initial * days_initial * hours_per_day_initial)
  (remaining_days : ℚ := (men_initial * days_initial * hours_per_day_initial) / (additional_men * hours_per_day_additional * fraction_work_additional))
  (total_days : ℚ := days_initial + remaining_days) :
  men_initial = 100 →
  days_initial = 25 →
  hours_per_day_initial = 8 →
  fraction_work_initial = 1 / 3 →
  additional_men = 160 →
  hours_per_day_additional = 10 →
  fraction_work_additional = 2 / 3 →
  total_days = 37.5 :=
by
  intros
  sorry

end initial_deadline_in_days_l144_144553


namespace Cameron_task_completion_l144_144531

theorem Cameron_task_completion (C : ℝ) (h1 : ∃ x, x = 9 / C) (h2 : ∃ y, y = 1 / 2) (total_work : ∃ z, z = 1):
  9 - 9 / C + 1/2 = 1 -> C = 18 := by
  sorry

end Cameron_task_completion_l144_144531


namespace candy_total_l144_144761

theorem candy_total (r b : ℕ) (hr : r = 145) (hb : b = 3264) : r + b = 3409 := by
  -- We can use Lean's rewrite tactic to handle the equalities, but since proof is skipped,
  -- it's not necessary to write out detailed tactics here.
  sorry

end candy_total_l144_144761


namespace projection_matrix_3_4_l144_144184

theorem projection_matrix_3_4 :
  let v := λ α : Type, @vector α 2 := ![3, 4]
  let proj := λ x : vector ℝ 2, (v ℝ ⬝ x) / (v ℝ ⬝ v ℝ) • v ℝ
  proj = (λ x : vector ℝ 2, matrix.mul_vec ![
     ![9 / 25, 12 / 25],
     ![12 / 25, 16 / 25]
  ] x) :=
by sorry

end projection_matrix_3_4_l144_144184


namespace prob_four_children_at_least_one_boy_one_girl_l144_144612

-- Define the probability of a single birth being a boy or a girl
def prob_boy_or_girl : ℚ := 1/2

-- Calculate the probability of all children being boys or all girls
def prob_all_boys : ℚ := (prob_boy_or_girl)^4
def prob_all_girls : ℚ := (prob_boy_or_girl)^4

-- Calculate the probability of having neither all boys nor all girls
def prob_at_least_one_boy_one_girl : ℚ := 1 - (prob_all_boys + prob_all_girls)

-- The theorem to prove
theorem prob_four_children_at_least_one_boy_one_girl : 
  prob_at_least_one_boy_one_girl = 7/8 := 
by 
  sorry

end prob_four_children_at_least_one_boy_one_girl_l144_144612


namespace arithmetic_square_root_of_nine_l144_144896

theorem arithmetic_square_root_of_nine : Real.sqrt 9 = 3 :=
sorry

end arithmetic_square_root_of_nine_l144_144896


namespace mother_hen_heavier_l144_144116

-- Define the weights in kilograms
def weight_mother_hen : ℝ := 2.3
def weight_baby_chick : ℝ := 0.4

-- State the theorem with the final correct answer
theorem mother_hen_heavier :
  weight_mother_hen - weight_baby_chick = 1.9 :=
by
  sorry

end mother_hen_heavier_l144_144116


namespace largest_sphere_surface_area_in_cone_l144_144206

theorem largest_sphere_surface_area_in_cone :
  (∀ (r : ℝ), (∃ (r : ℝ), r > 0 ∧ (1^2 + (3^2 - r^2) = 3^2)) →
    4 * π * r^2 ≤ 2 * π) :=
by
  sorry

end largest_sphere_surface_area_in_cone_l144_144206


namespace calculate_total_cost_l144_144795

def sandwich_cost : ℕ := 4
def soda_cost : ℕ := 3
def discount_threshold : ℕ := 10
def discount_rate : ℝ := 0.10
def num_sandwiches : ℕ := 7
def num_sodas : ℕ := 5

theorem calculate_total_cost :
  let total_items := num_sandwiches + num_sodas
  let cost_before_discount := num_sandwiches * sandwich_cost + num_sodas * soda_cost
  let discount := if total_items > discount_threshold then cost_before_discount * discount_rate else 0
  let final_cost := cost_before_discount - discount
  final_cost = 38.7 :=
by
  sorry

end calculate_total_cost_l144_144795


namespace max_M_value_l144_144109

noncomputable def M (x y z w : ℝ) : ℝ :=
  x * w + 2 * y * w + 3 * x * y + 3 * z * w + 4 * x * z + 5 * y * z

theorem max_M_value (x y z w : ℝ) (h : x + y + z + w = 1) :
  (M x y z w) ≤ 3 / 2 :=
sorry

end max_M_value_l144_144109


namespace find_discriminant_l144_144364

variables {a b c : ℝ}
variables (P : ℝ → ℝ)
def is_quadratic_polynomial (P : ℝ → ℝ) : Prop := ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, P x = a * x^2 + b * x + c)

theorem find_discriminant (h1 : is_quadratic_polynomial P)
  (h2 : ∃ x, P x = x - 2)
  (h3 : ∃ y, P y = 1 - y / 2)
  : ∃ D, D = -1/2 := 
sorry

end find_discriminant_l144_144364


namespace element_in_set_l144_144456

theorem element_in_set : 1 ∈ ({0, 1} : Set ℕ) := 
by 
  -- Proof goes here
  sorry

end element_in_set_l144_144456


namespace exponent_product_to_sixth_power_l144_144690

theorem exponent_product_to_sixth_power :
  ∃ n : ℤ, 3^(12) * 3^(18) = n^6 ∧ n = 243 :=
by
  use 243
  sorry

end exponent_product_to_sixth_power_l144_144690


namespace percentage_of_males_l144_144856

theorem percentage_of_males (P : ℝ) (total_employees : ℝ) (below_50_male_count : ℝ) :
  total_employees = 2800 →
  0.70 * (P / 100 * total_employees) = below_50_male_count →
  below_50_male_count = 490 →
  P = 25 :=
by
  intros h_total h_eq h_below_50
  sorry

end percentage_of_males_l144_144856


namespace lowest_discount_l144_144748

theorem lowest_discount (c m : ℝ) (p : ℝ) (h_c : c = 100) (h_m : m = 150) (h_p : p = 0.05) :
  ∃ (x : ℝ), m * (x / 100) = c * (1 + p) ∧ x = 70 :=
by
  use 70
  sorry

end lowest_discount_l144_144748


namespace five_star_three_l144_144170

def star (a b : ℤ) : ℤ := a^2 - 2 * a * b + b^2

theorem five_star_three : star 5 3 = 4 := by
  sorry

end five_star_three_l144_144170


namespace find_F_of_circle_l144_144513

def circle_equation (x y F : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y + F = 0

def is_circle_with_radius (x y F r : ℝ) : Prop := 
  ∃ k h, (x - k)^2 + (y + h)^2 = r

theorem find_F_of_circle {F : ℝ} :
  (∀ x y : ℝ, circle_equation x y F) ∧ 
  is_circle_with_radius 1 1 F 4 → F = -2 := 
by
  sorry

end find_F_of_circle_l144_144513


namespace expenses_denoted_as_negative_l144_144438

theorem expenses_denoted_as_negative (income_yuan expenses_yuan : Int) (h : income_yuan = 6) : 
  expenses_yuan = -4 :=
by
  sorry

end expenses_denoted_as_negative_l144_144438


namespace avg_annual_growth_rate_profit_exceeds_340_l144_144476

variable (P2018 P2020 : ℝ)
variable (r : ℝ)

theorem avg_annual_growth_rate :
    P2018 = 200 → P2020 = 288 →
    (1 + r)^2 = P2020 / P2018 →
    r = 0.2 :=
by
  intros hP2018 hP2020 hGrowth
  sorry

theorem profit_exceeds_340 (P2020 : ℝ) (r : ℝ) :
    P2020 = 288 → r = 0.2 →
    P2020 * (1 + r) > 340 :=
by
  intros hP2020 hr
  sorry

end avg_annual_growth_rate_profit_exceeds_340_l144_144476


namespace g_sum_even_function_l144_144254

def g (a b c d x : ℝ) : ℝ := a * x ^ 8 + b * x ^ 6 - c * x ^ 4 + d * x ^ 2 + 5

theorem g_sum_even_function 
  (a b c d : ℝ) 
  (h : g a b c d 2 = 4)
  : g a b c d 2 + g a b c d (-2) = 8 :=
by
  sorry

end g_sum_even_function_l144_144254


namespace arithmetic_sqrt_9_l144_144931

theorem arithmetic_sqrt_9 : real.sqrt 9 = 3 :=
by
  sorry

end arithmetic_sqrt_9_l144_144931


namespace product_of_numbers_l144_144577

theorem product_of_numbers (x y : ℤ) (h1 : x + y = 37) (h2 : x - y = 5) : x * y = 336 := by
  sorry

end product_of_numbers_l144_144577


namespace remaining_distance_l144_144121

-- Definitions of the given conditions
def D : ℕ := 500
def daily_alpha : ℕ := 30
def daily_beta : ℕ := 50
def effective_beta : ℕ := daily_beta / 2

-- Proving the theorem with given conditions
theorem remaining_distance (n : ℕ) (h : n = 25) :
  D - daily_alpha * n = 2 * (D - effective_beta * n) :=
by
  sorry

end remaining_distance_l144_144121


namespace perimeter_with_new_tiles_l144_144260

theorem perimeter_with_new_tiles (p_original : ℕ) (num_original_tiles : ℕ) (num_new_tiles : ℕ)
  (h1 : p_original = 16)
  (h2 : num_original_tiles = 9)
  (h3 : num_new_tiles = 3) :
  ∃ p_new : ℕ, p_new = 17 :=
by
  sorry

end perimeter_with_new_tiles_l144_144260


namespace Mike_given_total_cookies_l144_144451

-- All given conditions
variables (total Tim fridge Mike Anna : Nat)
axiom h1 : total = 256
axiom h2 : Tim = 15
axiom h3 : fridge = 188
axiom h4 : Anna = 2 * Tim
axiom h5 : total = Tim + Anna + fridge + Mike

-- The goal of the proof
theorem Mike_given_total_cookies : Mike = 23 :=
by
  sorry

end Mike_given_total_cookies_l144_144451


namespace largest_digit_div_by_6_l144_144132

/-- M is the largest digit such that 3190M is divisible by 6 -/
theorem largest_digit_div_by_6 (M : ℕ) : (M ≤ 9) → (3190 * 10 + M) % 2 = 0 → (3190 * 10 + M) % 3 = 0 → M = 8 := 
by
  intro hM9 hDiv2 hDiv3
  sorry

end largest_digit_div_by_6_l144_144132


namespace books_sold_online_l144_144466

theorem books_sold_online (X : ℤ) 
  (h1: 743 = 502 + (37 + X) + (74 + X + 34) - 160) : 
  X = 128 := 
by sorry

end books_sold_online_l144_144466


namespace average_growth_rate_equation_l144_144151

-- Define the current and target processing capacities
def current_capacity : ℝ := 1000
def target_capacity : ℝ := 1200

-- Define the time period in months
def months : ℕ := 2

-- Define the monthly average growth rate
variable (x : ℝ)

-- The statement to be proven: current capacity increased by the growth rate over 2 months equals the target capacity 
theorem average_growth_rate_equation :
  current_capacity * (1 + x) ^ months = target_capacity :=
sorry

end average_growth_rate_equation_l144_144151


namespace ratio_problem_l144_144845

variable (a b c d : ℝ)

theorem ratio_problem (h1 : a / b = 3) (h2 : b / c = 1 / 4) (h3 : c / d = 5) : d / a = 4 / 15 := 
sorry

end ratio_problem_l144_144845


namespace beta_cannot_be_determined_l144_144227

variables (α β : ℝ)
def consecutive_interior_angles (α β : ℝ) : Prop := -- define what it means for angles to be consecutive interior angles
  α + β = 180  -- this is true for interior angles, for illustrative purposes.

theorem beta_cannot_be_determined
  (h1 : consecutive_interior_angles α β)
  (h2 : α = 55) :
  ¬(∃ β, β = α) :=
by
  sorry

end beta_cannot_be_determined_l144_144227


namespace tip_percentage_l144_144598

/--
A family paid $30 for food, the sales tax rate is 9.5%, and the total amount paid was $35.75. Prove that the tip percentage is 9.67%.
-/
theorem tip_percentage (food_cost : ℝ) (sales_tax_rate : ℝ) (total_paid : ℝ)
  (h1 : food_cost = 30)
  (h2 : sales_tax_rate = 0.095)
  (h3 : total_paid = 35.75) :
  ((total_paid - (food_cost * (1 + sales_tax_rate))) / food_cost) * 100 = 9.67 :=
by
  sorry

end tip_percentage_l144_144598


namespace abs_gt_two_l144_144700

theorem abs_gt_two (x : ℝ) : |x| > 2 → x > 2 ∨ x < -2 :=
by
  intros
  sorry

end abs_gt_two_l144_144700


namespace expenses_denoted_as_negative_l144_144439

theorem expenses_denoted_as_negative (income_yuan expenses_yuan : Int) (h : income_yuan = 6) : 
  expenses_yuan = -4 :=
by
  sorry

end expenses_denoted_as_negative_l144_144439


namespace geometric_sequence_sum_is_9_l144_144711

theorem geometric_sequence_sum_is_9 {a : ℕ → ℝ} (q : ℝ) 
  (h3a7 : a 3 * a 7 = 8) 
  (h4a6 : a 4 + a 6 = 6)
  (h_geom : ∀ n, a (n + 1) = a n * q) : a 2 + a 8 = 9 :=
sorry

end geometric_sequence_sum_is_9_l144_144711


namespace arithmetic_square_root_of_9_l144_144913

theorem arithmetic_square_root_of_9 : sqrt 9 = 3 := by
  sorry

end arithmetic_square_root_of_9_l144_144913


namespace class_size_count_l144_144729

theorem class_size_count : 
  ∃ (n : ℕ), 
  n = 6 ∧ 
  (∀ (b g : ℕ), (2 < b ∧ b < 10) → (14 < g ∧ g < 23) → b + g > 25 → 
    ∃ (sizes : Finset ℕ), sizes.card = n ∧ 
    ∀ (s : ℕ), s ∈ sizes → (∃ (b' g' : ℕ), s = b' + g' ∧ s > 25)) :=
sorry

end class_size_count_l144_144729


namespace prime_numbers_satisfying_equation_l144_144813

theorem prime_numbers_satisfying_equation :
  ∀ p : ℕ, Nat.Prime p →
    (∃ x y : ℕ, 1 ≤ x ∧ 1 ≤ y ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p) →
    p = 2 ∨ p = 3 ∨ p = 7 := 
by 
  intro p hpprime h
  sorry

end prime_numbers_satisfying_equation_l144_144813


namespace bottle_t_capsules_l144_144629

theorem bottle_t_capsules 
  (num_capsules_r : ℕ)
  (cost_r : ℝ)
  (cost_t : ℝ)
  (cost_per_capsule_difference : ℝ)
  (h1 : num_capsules_r = 250)
  (h2 : cost_r = 6.25)
  (h3 : cost_t = 3.00)
  (h4 : cost_per_capsule_difference = 0.005) :
  ∃ (num_capsules_t : ℕ), num_capsules_t = 150 := 
by
  sorry

end bottle_t_capsules_l144_144629


namespace ellipse_properties_l144_144510

noncomputable def standard_equation_of_ellipse (x y : ℝ) : Prop :=
  (x^2) / 4 + y^2 = 1

noncomputable def trajectory_equation_midpoint (x y : ℝ) : Prop :=
  ((2 * x - 1)^2) / 4 + (2 * y - 1 / 2)^2 = 1

theorem ellipse_properties :
  (∀ x y : ℝ, standard_equation_of_ellipse x y) ∧
  (∀ x y : ℝ, trajectory_equation_midpoint x y) :=
by
  sorry

end ellipse_properties_l144_144510


namespace probability_reach_origin_from_3_3_l144_144318

noncomputable def P : ℕ → ℕ → ℚ
| 0, 0 => 1
| x+1, 0 => 0
| 0, y+1 => 0
| x+1, y+1 => (1/3) * P x (y+1) + (1/3) * P (x+1) y + (1/3) * P x y

theorem probability_reach_origin_from_3_3 : P 3 3 = 1 / 27 := by
  sorry

end probability_reach_origin_from_3_3_l144_144318


namespace cubic_yard_to_cubic_feet_l144_144668

theorem cubic_yard_to_cubic_feet (h : 1 = 3) : 1 = 27 := 
by
  sorry

end cubic_yard_to_cubic_feet_l144_144668


namespace problem1_problem2_l144_144114

-- Problem 1: Prove that the solutions of x^2 + 6x - 7 = 0 are x = -7 and x = 1
theorem problem1 (x : ℝ) : x^2 + 6*x - 7 = 0 ↔ (x = -7 ∨ x = 1) := by
  -- Proof omitted
  sorry

-- Problem 2: Prove that the solutions of 4x(2x+1) = 3(2x+1) are x = -1/2 and x = 3/4
theorem problem2 (x : ℝ) : 4*x*(2*x + 1) = 3*(2*x + 1) ↔ (x = -1/2 ∨ x = 3/4) := by
  -- Proof omitted
  sorry

end problem1_problem2_l144_144114


namespace maximum_value_of_f_l144_144274

noncomputable def f (x : ℝ) : ℝ := x + 2 * Real.cos x

theorem maximum_value_of_f :
  ∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = (Real.pi / 6) + Real.sqrt 3 ∧ 
  ∀ y ∈ Set.Icc 0 (Real.pi / 2), f y ≤ f (Real.pi / 6) :=
by
  sorry

end maximum_value_of_f_l144_144274


namespace arithmetic_seq_a6_l144_144708

open Real

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, (a (n + m) = a n + a m - a 0)

-- Given conditions
def condition_1 (a : ℕ → ℝ) : Prop :=
  a 2 = 4

def condition_2 (a : ℕ → ℝ) : Prop :=
  a 4 = 2

-- Mathematical statement
theorem arithmetic_seq_a6 
  (a : ℕ → ℝ)
  (h_seq: arithmetic_sequence a)
  (h_cond1 : condition_1 a)
  (h_cond2 : condition_2 a) : 
  a 6 = 0 := 
sorry

end arithmetic_seq_a6_l144_144708


namespace tan_diff_sum_angles_l144_144842

open Real

variables (α β : ℝ)

def is_acute (x : ℝ) : Prop := 0 < x ∧ x < π / 2

-- Conditions
axiom cos_alpha : cos α = 2 * sqrt 5 / 5
axiom cos_beta : cos β = 3 * sqrt 10 / 10
axiom acute_alpha : is_acute α
axiom acute_beta : is_acute β

-- Question (I): Prove tan(α - β) = 1/7
theorem tan_diff : tan (α - β) = 1 / 7 :=
sorry

-- Question (II): Prove α + β = π/4
theorem sum_angles : α + β = π / 4 :=
sorry

end tan_diff_sum_angles_l144_144842


namespace square_field_area_l144_144567

theorem square_field_area (s : ℕ) (area cost_per_meter total_cost gate_width : ℕ):
  area = s^2 →
  cost_per_meter = 2 →
  total_cost = 1332 →
  gate_width = 1 →
  (4 * s - 2 * gate_width) * cost_per_meter = total_cost →
  area = 27889 :=
by
  intros h_area h_cost_per_meter h_total_cost h_gate_width h_equation
  sorry

end square_field_area_l144_144567


namespace expenses_notation_l144_144436

theorem expenses_notation (income expense : ℤ) (h_income : income = 6) (h_expense : -expense = income) : expense = -4 := 
by
  sorry

end expenses_notation_l144_144436


namespace find_natural_numbers_l144_144039

theorem find_natural_numbers (a b : ℕ) (p : ℕ) (hp : Nat.Prime p)
  (h : a^3 - b^3 = 633 * p) : a = 16 ∧ b = 13 :=
by
  sorry

end find_natural_numbers_l144_144039


namespace find_number_l144_144340

theorem find_number (x : ℝ) (h : 0.6667 * x + 1 = 0.75 * x) : x = 12 :=
sorry

end find_number_l144_144340


namespace even_composite_sum_consecutive_odd_numbers_l144_144967

theorem even_composite_sum_consecutive_odd_numbers (a k : ℤ) : ∃ (n m : ℤ), n = 2 * k ∧ m = n * (2 * a + n) ∧ m % 4 = 0 :=
by
  sorry

end even_composite_sum_consecutive_odd_numbers_l144_144967


namespace volleyball_club_girls_l144_144995

theorem volleyball_club_girls (B G : ℕ) (h1 : B + G = 32) (h2 : (1 / 3 : ℝ) * G + ↑B = 20) : G = 18 := 
by
  sorry

end volleyball_club_girls_l144_144995


namespace washing_machine_heavy_washes_l144_144789

theorem washing_machine_heavy_washes
  (H : ℕ)                                  -- The number of heavy washes
  (heavy_wash_gallons : ℕ := 20)            -- Gallons of water for a heavy wash
  (regular_wash_gallons : ℕ := 10)          -- Gallons of water for a regular wash
  (light_wash_gallons : ℕ := 2)             -- Gallons of water for a light wash
  (num_regular_washes : ℕ := 3)             -- Number of regular washes
  (num_light_washes : ℕ := 1)               -- Number of light washes
  (num_bleach_rinses : ℕ := 2)              -- Number of bleach rinses (extra light washes)
  (total_water_needed : ℕ := 76)            -- Total gallons of water needed
  (h_regular_wash_water : num_regular_washes * regular_wash_gallons = 30)
  (h_light_wash_water : num_light_washes * light_wash_gallons = 2)
  (h_bleach_rinse_water : num_bleach_rinses * light_wash_gallons = 4) :
  20 * H + 30 + 2 + 4 = 76 → H = 2 :=
by
  intros
  sorry

end washing_machine_heavy_washes_l144_144789


namespace alpha_inverse_proportional_beta_l144_144432

theorem alpha_inverse_proportional_beta (α β : ℝ) (k : ℝ) :
  (∀ β1 α1, α1 * β1 = k) → (4 * 2 = k) → (β = -3) → (α = -8/3) :=
by
  sorry

end alpha_inverse_proportional_beta_l144_144432


namespace arithmetic_square_root_of_9_l144_144917

theorem arithmetic_square_root_of_9 : Real.sqrt 9 = 3 := by
  sorry

end arithmetic_square_root_of_9_l144_144917


namespace tires_should_be_swapped_l144_144750

-- Define the conditions
def front_wear_out_distance : ℝ := 25000
def rear_wear_out_distance : ℝ := 15000

-- Define the distance to swap tires
def swap_distance : ℝ := 9375

-- Theorem statement
theorem tires_should_be_swapped :
  -- The distance for both tires to wear out should be the same
  swap_distance + (front_wear_out_distance - swap_distance) * (rear_wear_out_distance / front_wear_out_distance) = rear_wear_out_distance :=
sorry

end tires_should_be_swapped_l144_144750


namespace three_pow_mul_l144_144694

theorem three_pow_mul (a b : ℕ) (h_a : a = 12) (h_b : b = 18) :
  3^a * 3^b = 243^6 := by
  rw [h_a, h_b]
  calc
    3^12 * 3^18
      = 3^(12 + 18) : by rw [pow_add]
  ... = 3^30 : by norm_num
  ... = (3^5)^6 : by rw [pow_mul, ← mul_comm]
  ... = 243^6 : by norm_num

end three_pow_mul_l144_144694


namespace combined_squirrel_count_is_28_l144_144424

def squirrel_count_combined(first_student_count : ℕ, second_student_addition_fraction : ℚ) : ℕ :=
  let second_student_count := first_student_count + (second_student_addition_fraction * first_student_count).natAbs
  in first_student_count + second_student_count

theorem combined_squirrel_count_is_28 (h1 : first_student_count = 12) (h2 : second_student_addition_fraction = 1/3) :
  squirrel_count_combined first_student_count second_student_addition_fraction = 28 := by
  sorry

end combined_squirrel_count_is_28_l144_144424


namespace integer_to_sixth_power_l144_144685

theorem integer_to_sixth_power (a b : ℕ) (h : 3^a * 3^b = 3^(a + b)) (ha : a = 12) (hb : b = 18) : 
  ∃ x : ℕ, x = 243 ∧ x^6 = 3^(a + b) :=
by
  sorry

end integer_to_sixth_power_l144_144685


namespace work_done_in_give_days_l144_144459

theorem work_done_in_give_days (a b c : ℚ) :
  (a + b + c) = (1 / 4) ∧ a = (1 / 12) ∧ b = (1 / 9) → c = (1 / 18) :=
by
  intro h
  sorry

end work_done_in_give_days_l144_144459


namespace arithmetic_square_root_of_9_l144_144911

theorem arithmetic_square_root_of_9 : sqrt 9 = 3 := by
  sorry

end arithmetic_square_root_of_9_l144_144911


namespace slope_of_parallel_lines_l144_144744

theorem slope_of_parallel_lines (m : ℝ)
  (y1 y2 y3 : ℝ)
  (h1 : y1 = 2) 
  (h2 : y2 = 3) 
  (h3 : y3 = 4)
  (sum_of_x_intercepts : (-2 / m) + (-3 / m) + (-4 / m) = 36) :
  m = -1 / 4 := by
  sorry

end slope_of_parallel_lines_l144_144744


namespace calculate_AE_l144_144582

variable {k : ℝ} (A B C D E : Type*)

namespace Geometry

def shared_angle (A B C : Type*) : Prop := sorry -- assumes triangles share angle A

def prop_constant_proportion (AB AC AD AE : ℝ) (k : ℝ) : Prop :=
  AB * AC = k * AD * AE

theorem calculate_AE
  (A B C D E : Type*) 
  (AB AC AD AE : ℝ)
  (h_shared : shared_angle A B C)
  (h_AB : AB = 5)
  (h_AC : AC = 7)
  (h_AD : AD = 2)
  (h_proportion : prop_constant_proportion AB AC AD AE k)
  (h_k : k = 1) :
  AE = 17.5 := 
sorry

end Geometry

end calculate_AE_l144_144582


namespace range_of_k_for_distinct_real_roots_l144_144048

theorem range_of_k_for_distinct_real_roots (k : ℝ) :
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (k - 1) * x1^2 - 2 * x1 + 1 = 0 ∧ (k - 1) * x2^2 - 2 * x2 + 1 = 0) →
    k < 2 ∧ k ≠ 1 :=
by
  sorry

end range_of_k_for_distinct_real_roots_l144_144048


namespace polynomial_discriminant_l144_144373

theorem polynomial_discriminant (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
by
  sorry

end polynomial_discriminant_l144_144373


namespace smallest_class_size_l144_144526

theorem smallest_class_size
  (x : ℕ)
  (h1 : ∀ y : ℕ, y = x + 2)
  (total_students : 5 * x + 2 > 40) :
  ∃ (n : ℕ), n = 5 * x + 2 ∧ n = 42 :=
by
  sorry

end smallest_class_size_l144_144526


namespace arithmetic_square_root_of_nine_l144_144906

theorem arithmetic_square_root_of_nine : ∃ (x : ℝ), (x * x = 9) ∧ (x ≥ 0) ∧ (x = 3) :=
by
  sorry

end arithmetic_square_root_of_nine_l144_144906


namespace smallest_three_digit_in_pascals_triangle_l144_144136

theorem smallest_three_digit_in_pascals_triangle : ∃ k n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ ∀ m, ((m <= n) ∧ (m >= 100)) → m ≥ n :=
by
  sorry

end smallest_three_digit_in_pascals_triangle_l144_144136


namespace flyers_left_to_hand_out_l144_144238

-- Definitions for given conditions
def total_flyers : Nat := 1236
def jack_handout : Nat := 120
def rose_handout : Nat := 320

-- Statement of the problem
theorem flyers_left_to_hand_out : total_flyers - (jack_handout + rose_handout) = 796 :=
by
  -- proof goes here
  sorry

end flyers_left_to_hand_out_l144_144238


namespace daphne_visits_l144_144638

theorem daphne_visits (n : ℕ) (h1 : n = 400) (h2: ∀ k, (k % 3 = 0 ∨ k % 6 = 0 ∨ k % 5 = 0) ↔ (k = 3 ∨ k = 6 ∨ k = 5)) :
    (exactly_two_visits : ℕ) :=
sorry

end daphne_visits_l144_144638


namespace necessary_condition_not_sufficient_condition_l144_144056

noncomputable def zero_point (a : ℝ) : Prop :=
  ∃ x : ℝ, 3^x + a - 1 = 0

noncomputable def decreasing_log (a : ℝ) : Prop :=
  0 < a ∧ a < 1

theorem necessary_condition (a : ℝ) (h : zero_point a) : 0 < a ∧ a < 1 := sorry

theorem not_sufficient_condition (a : ℝ) (h : 0 < a ∧ a < 1) : ¬(zero_point a) := sorry

end necessary_condition_not_sufficient_condition_l144_144056


namespace seungho_more_marbles_l144_144426

variable (S H : ℕ)

-- Seungho gave 273 marbles to Hyukjin
def given_marbles : ℕ := 273

-- After giving 273 marbles, Seungho has 477 more marbles than Hyukjin
axiom marbles_condition : S - given_marbles = (H + given_marbles) + 477

theorem seungho_more_marbles (S H : ℕ) (marbles_condition : S - 273 = (H + 273) + 477) : S = H + 1023 :=
by
  sorry

end seungho_more_marbles_l144_144426


namespace son_working_alone_l144_144294

theorem son_working_alone (M S : ℝ) (h1: M = 1 / 5) (h2: M + S = 1 / 3) : 1 / S = 7.5 :=
  by
  sorry

end son_working_alone_l144_144294


namespace quadratic_polynomial_discriminant_l144_144370

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₁ : a ≠ 0)
  (h₂ : ∃ x : ℝ, a * x^2 + b * x + c = x - 2 ∧ (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h₃ : ∃ x : ℝ, a * x^2 + b * x + c = 1 - x / 2 ∧ (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l144_144370


namespace arithmetic_sqrt_of_9_l144_144938

def arithmetic_sqrt (n : ℕ) : ℕ :=
  Nat.sqrt n

theorem arithmetic_sqrt_of_9 : arithmetic_sqrt 9 = 3 :=
by
  sorry

end arithmetic_sqrt_of_9_l144_144938


namespace smallest_solution_is_39_over_8_l144_144196

noncomputable def smallest_solution (x : ℝ) : Prop :=
  (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ (x ≠ 0) ∧ (x ≠ 3)

theorem smallest_solution_is_39_over_8 : ∃ x > 0, smallest_solution x ∧ x = 39 / 8 :=
by
  sorry

end smallest_solution_is_39_over_8_l144_144196


namespace integer_to_sixth_power_l144_144684

theorem integer_to_sixth_power (a b : ℕ) (h : 3^a * 3^b = 3^(a + b)) (ha : a = 12) (hb : b = 18) : 
  ∃ x : ℕ, x = 243 ∧ x^6 = 3^(a + b) :=
by
  sorry

end integer_to_sixth_power_l144_144684


namespace simplification_at_negative_two_l144_144887

noncomputable def simplify_expression (x : ℚ) : ℚ :=
  ((x^2 - 4*x + 4) / (x^2 - 1)) / ((x^2 - 2*x) / (x + 1)) + (1 / (x - 1))

theorem simplification_at_negative_two :
  ∀ x : ℚ, -2 ≤ x ∧ x ≤ 2 ∧ x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 → simplify_expression (-2) = -1 :=
by simp [simplify_expression]; sorry

end simplification_at_negative_two_l144_144887


namespace prob_neq_zero_l144_144288

noncomputable def probability_no_one (a b c d : ℕ) : ℚ :=
  if 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ 1 ≤ d ∧ d ≤ 6 
  then (5/6)^4 
  else 0

theorem prob_neq_zero (a b c d : ℕ) :
  (1 ≤ a) ∧ (a ≤ 6) ∧ (1 ≤ b) ∧ (b ≤ 6) ∧ (1 ≤ c) ∧ (c ≤ 6) ∧ (1 ≤ d) ∧ (d ≤ 6) →
  (a - 1) * (b - 1) * (c - 1) * (d - 1) ≠ 0 ↔ 
  probability_no_one a b c d = 625/1296 :=
by
  sorry

end prob_neq_zero_l144_144288


namespace quadratic_discriminant_l144_144359

variable {a b c : ℝ}
variable (h₁ : a ≠ 0)
variable (h₂ : (b - 1)^2 - 4 * a * (c + 2) = 0)
variable (h₃ : (b + 1/2)^2 - 4 * a * (c - 1) = 0)

theorem quadratic_discriminant : b^2 - 4 * a * c = -1 / 2 := 
by
  have h₁' : (b - 1)^2 - 4 * a * (c + 2) = 0 := h₂
  have h₂' : (b + 1/2)^2 - 4 * a * (c - 1) = 0 := h₃
  sorry

end quadratic_discriminant_l144_144359


namespace geometric_sequence_sum_a_l144_144524

theorem geometric_sequence_sum_a (a : ℤ) (S : ℕ → ℤ) (a_n : ℕ → ℤ) 
  (h1 : ∀ n : ℕ, S n = 2^n + a)
  (h2 : ∀ n : ℕ, a_n n = if n = 1 then S 1 else S n - S (n - 1)) :
  a = -1 :=
by
  sorry

end geometric_sequence_sum_a_l144_144524


namespace arithmetic_square_root_of_nine_l144_144894

theorem arithmetic_square_root_of_nine : Real.sqrt 9 = 3 :=
sorry

end arithmetic_square_root_of_nine_l144_144894


namespace intersection_M_N_l144_144839

def M : Set ℝ := { x | (x - 2) / (x - 3) < 0 }
def N : Set ℝ := { x | Real.log (x - 2) / Real.log (1 / 2) ≥ 1 }

theorem intersection_M_N : M ∩ N = { x | 2 < x ∧ x ≤ 5 / 2 } :=
by
  sorry

end intersection_M_N_l144_144839


namespace find_number_l144_144067

theorem find_number (x : ℤ) (h : 5 * x - 28 = 232) : x = 52 :=
by
  sorry

end find_number_l144_144067


namespace sally_gave_joan_5_balloons_l144_144865

theorem sally_gave_joan_5_balloons (x : ℕ) (h1 : 9 + x - 2 = 12) : x = 5 :=
by
  -- Proof is skipped
  sorry

end sally_gave_joan_5_balloons_l144_144865


namespace rain_probability_tel_aviv_l144_144297

open scoped Classical

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k) * (p^k) * ((1 - p)^(n - k))

theorem rain_probability_tel_aviv :
  binomial_probability 6 4 0.5 = 0.234375 :=
by 
  sorry

end rain_probability_tel_aviv_l144_144297


namespace solution_set_inequality_l144_144950

theorem solution_set_inequality (x : ℝ) (h1 : 2 < 1 / (x - 1)) (h2 : 1 / (x - 1) < 3) (h3 : x - 1 > 0) :
  4 / 3 < x ∧ x < 3 / 2 :=
sorry

end solution_set_inequality_l144_144950


namespace Tanya_accompanied_two_l144_144826

-- Define the number of songs sung by each girl
def Anya_songs : ℕ := 8
def Tanya_songs : ℕ := 6
def Olya_songs : ℕ := 3
def Katya_songs : ℕ := 7

-- Assume each song is sung by three girls
def total_songs : ℕ := (Anya_songs + Tanya_songs + Olya_songs + Katya_songs) / 3

-- Define the number of times Tanya accompanied
def Tanya_accompanied : ℕ := total_songs - Tanya_songs

-- Prove that Tanya accompanied 2 times
theorem Tanya_accompanied_two : Tanya_accompanied = 2 :=
by sorry

end Tanya_accompanied_two_l144_144826


namespace circle_condition_l144_144751

theorem circle_condition (m : ℝ) :
    (4 * m) ^ 2 + 4 - 4 * 5 * m > 0 ↔ (m < 1 / 4 ∨ m > 1) := sorry

end circle_condition_l144_144751


namespace smallest_possible_n_l144_144140

theorem smallest_possible_n (n : ℕ) (h : lcm 60 n / gcd 60 n = 60) : n = 16 :=
sorry

end smallest_possible_n_l144_144140


namespace soccer_tournament_games_l144_144876

-- Define the single-elimination tournament problem
def single_elimination_games (teams : ℕ) : ℕ :=
  teams - 1

-- Define the specific problem instance
def teams := 20

-- State the theorem
theorem soccer_tournament_games : single_elimination_games teams = 19 :=
  sorry

end soccer_tournament_games_l144_144876


namespace min_value_of_x2_plus_y2_l144_144506

-- Define the problem statement
theorem min_value_of_x2_plus_y2 (x y : ℝ) (h : 3 * x + y = 10) : x^2 + y^2 ≥ 10 :=
sorry

end min_value_of_x2_plus_y2_l144_144506


namespace minimum_value_of_f_l144_144091

noncomputable def f (x : ℝ) : ℝ := |2*x - 1| + |3*x - 2| + |4*x - 3| + |5*x - 4|

theorem minimum_value_of_f : ∃ x : ℝ, (∀ y : ℝ, f y ≥ 1) ∧ f x = 1 :=
by
  sorry

end minimum_value_of_f_l144_144091


namespace value_of_a_b_l144_144808

theorem value_of_a_b:
  ∃ (a b : ℕ), a = 3 ∧ b = 2 ∧ (a + 6 * 10^3 + 7 * 10^2 + 9 * 10 + b) % 72 = 0 :=
by
  sorry

end value_of_a_b_l144_144808


namespace find_divisor_l144_144521

theorem find_divisor (h : 2994 / 14.5 = 171) : 29.94 / 1.75 = 17.1 :=
by
  sorry

end find_divisor_l144_144521


namespace fixed_point_range_l144_144045

theorem fixed_point_range (a : ℝ) : (∃ x : ℝ, x = x^2 + x + a) → a ≤ 0 :=
sorry

end fixed_point_range_l144_144045


namespace rakesh_salary_l144_144266

variable (S : ℝ) -- The salary S is a real number
variable (h : 0.595 * S = 2380) -- Condition derived from the problem

theorem rakesh_salary : S = 4000 :=
by
  sorry

end rakesh_salary_l144_144266


namespace solve_quadratic_equation_l144_144113

theorem solve_quadratic_equation (x : ℝ) :
  (6 * x^2 - 3 * x - 1 = 2 * x - 2) ↔ (x = 1 / 3 ∨ x = 1 / 2) :=
by sorry

end solve_quadratic_equation_l144_144113


namespace find_y_l144_144068

variables (x y : ℝ)

theorem find_y (h1 : x = 103) (h2 : x^3 * y - 4 * x^2 * y + 4 * x * y = 515400) : y = 1 / 2 :=
sorry

end find_y_l144_144068


namespace range_of_m_l144_144662

theorem range_of_m (m : ℝ) :
  (1 - 2 * m > 0) ∧ (m + 1 > 0) → -1 < m ∧ m < 1/2 :=
by
  sorry

end range_of_m_l144_144662


namespace fraction_calculation_l144_144332

theorem fraction_calculation : (3/10 : ℚ) + (5/100 : ℚ) - (2/1000 : ℚ) = 348/1000 := 
by 
  sorry

end fraction_calculation_l144_144332


namespace total_shaded_area_l144_144012

theorem total_shaded_area (S T : ℕ) 
  (h1 : 12 / S = 4)
  (h2 : S / T = 3) :
  (S * S) + 8 * (T * T) = 17 :=
by
  sorry

end total_shaded_area_l144_144012


namespace symmetric_line_eq_l144_144968

-- Define the original line equation
def original_line (x: ℝ) : ℝ := -2 * x - 3

-- Define the symmetric line with respect to y-axis
def symmetric_line (x: ℝ) : ℝ := 2 * x - 3

-- The theorem stating the symmetric line with respect to the y-axis
theorem symmetric_line_eq : (∀ x: ℝ, original_line (-x) = symmetric_line x) :=
by
  -- Proof goes here
  sorry

end symmetric_line_eq_l144_144968


namespace remainder_addition_l144_144872

theorem remainder_addition (k m : ℤ) (x y : ℤ) (h₁ : x = 124 * k + 13) (h₂ : y = 186 * m + 17) :
  ((x + y + 19) % 62) = 49 :=
by {
  sorry
}

end remainder_addition_l144_144872


namespace amount_paid_per_person_is_correct_l144_144465

noncomputable def amount_each_person_paid (total_bill : ℝ) (tip_rate : ℝ) (tax_rate : ℝ) (num_people : ℕ) : ℝ := 
  let tip_amount := tip_rate * total_bill
  let tax_amount := tax_rate * total_bill
  let total_amount := total_bill + tip_amount + tax_amount
  total_amount / num_people

theorem amount_paid_per_person_is_correct :
  amount_each_person_paid 425 0.18 0.08 15 = 35.7 :=
by
  sorry

end amount_paid_per_person_is_correct_l144_144465


namespace sphere_radius_equals_4_l144_144782

noncomputable def radius_of_sphere
  (sun_parallel : true)
  (meter_stick_height : ℝ)
  (meter_stick_shadow : ℝ)
  (sphere_shadow_distance : ℝ) : ℝ :=
if h : meter_stick_height / meter_stick_shadow = sphere_shadow_distance / 16 then
  4
else
  sorry

theorem sphere_radius_equals_4 
  (sun_parallel : true = true)
  (meter_stick_height : ℝ := 1)
  (meter_stick_shadow : ℝ := 4)
  (sphere_shadow_distance : ℝ := 16) : 
  radius_of_sphere sun_parallel meter_stick_height meter_stick_shadow sphere_shadow_distance = 4 :=
by
  simp [radius_of_sphere]
  sorry

end sphere_radius_equals_4_l144_144782


namespace exists_trinomial_with_exponents_three_l144_144141

theorem exists_trinomial_with_exponents_three (x y : ℝ) :
  ∃ (a b c : ℝ) (t1 t2 t3 : ℕ × ℕ), 
  t1.1 + t1.2 = 3 ∧ t2.1 + t2.2 = 3 ∧ t3.1 + t3.2 = 3 ∧
  (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧
  (a * x ^ t1.1 * y ^ t1.2 + b * x ^ t2.1 * y ^ t2.2 + c * x ^ t3.1 * y ^ t3.2 ≠ 0) := sorry

end exists_trinomial_with_exponents_three_l144_144141


namespace projection_onto_3_4_matrix_l144_144188

def projection_matrix := λ (u : ℝ) (v : ℝ), (3 * u + 4 * v) / 25

theorem projection_onto_3_4_matrix :
  ∀ (x y : ℝ),
  (λ (u v : ℝ), (3 * x + 4 * y) / 25) = (λ (u v : ℝ), (\(u * 9 / 25) + (v * 12 / 25), (u * 12 / 25) + (v * 16 / 25))) :=
by
  sorry

end projection_onto_3_4_matrix_l144_144188


namespace proof_a_eq_neg2x_or_3x_l144_144848

theorem proof_a_eq_neg2x_or_3x (a b x : ℝ) (h1 : a - b = x) (h2 : a^3 - b^3 = 19 * x^3) (h3 : x ≠ 0) : 
  a = -2 * x ∨ a = 3 * x :=
  sorry

end proof_a_eq_neg2x_or_3x_l144_144848


namespace arithmetic_sqrt_9_l144_144930

theorem arithmetic_sqrt_9 : real.sqrt 9 = 3 :=
by
  sorry

end arithmetic_sqrt_9_l144_144930


namespace arithmetic_sqrt_9_l144_144932

theorem arithmetic_sqrt_9 : real.sqrt 9 = 3 :=
by
  sorry

end arithmetic_sqrt_9_l144_144932


namespace smallest_integer_greater_than_sqrt5_plus_sqrt3_pow6_l144_144961

theorem smallest_integer_greater_than_sqrt5_plus_sqrt3_pow6 :
  ∃ n : ℤ, n = 3323 ∧ n > (Real.sqrt 5 + Real.sqrt 3)^6 ∧ ∀ m : ℤ, m > (Real.sqrt 5 + Real.sqrt 3)^6 → n ≤ m :=
by
  sorry

end smallest_integer_greater_than_sqrt5_plus_sqrt3_pow6_l144_144961


namespace cupcakes_leftover_l144_144100

theorem cupcakes_leftover {total_cupcakes nutty_cupcakes gluten_free_cupcakes children children_no_nuts child_only_gf leftover_nutty leftover_regular : Nat} :
  total_cupcakes = 84 →
  children = 7 →
  nutty_cupcakes = 18 →
  gluten_free_cupcakes = 25 →
  children_no_nuts = 2 →
  child_only_gf = 1 →
  leftover_nutty = 3 →
  leftover_regular = 2 →
  leftover_nutty + leftover_regular = 5 :=
by
  sorry

end cupcakes_leftover_l144_144100


namespace flyers_left_l144_144245

theorem flyers_left (total_flyers : ℕ) (jack_flyers : ℕ) (rose_flyers : ℕ) (h1 : total_flyers = 1236) (h2 : jack_flyers = 120) (h3 : rose_flyers = 320) : (total_flyers - (jack_flyers + rose_flyers) = 796) := 
by
  sorry

end flyers_left_l144_144245


namespace child_support_owed_l144_144549

noncomputable def income_first_3_years : ℕ := 3 * 30000
noncomputable def raise_per_year : ℕ := 30000 * 20 / 100
noncomputable def new_salary : ℕ := 30000 + raise_per_year
noncomputable def income_next_4_years : ℕ := 4 * new_salary
noncomputable def total_income : ℕ := income_first_3_years + income_next_4_years
noncomputable def total_child_support : ℕ := total_income * 30 / 100
noncomputable def amount_paid : ℕ := 1200
noncomputable def amount_owed : ℕ := total_child_support - amount_paid

theorem child_support_owed : amount_owed = 69000 := by
  sorry

end child_support_owed_l144_144549


namespace polynomial_diff_l144_144512

theorem polynomial_diff (m n : ℤ) (h1 : 2 * m + 2 = 0) (h2 : n - 4 = 0) :
  (4 * m^2 * n - 3 * m * n^2) - 2 * (m^2 * n + m * n^2) = -72 := 
by {
  -- This is where the proof would go, so we put sorry for now
  sorry
}

end polynomial_diff_l144_144512


namespace composite_product_division_l144_144642

noncomputable def firstFiveCompositeProduct : ℕ := 4 * 6 * 8 * 9 * 10
noncomputable def nextFiveCompositeProduct : ℕ := 12 * 14 * 15 * 16 * 18

theorem composite_product_division : firstFiveCompositeProduct / nextFiveCompositeProduct = 1 / 42 := by
  sorry

end composite_product_division_l144_144642


namespace quadratic_discriminant_l144_144361

variable {a b c : ℝ}
variable (h₁ : a ≠ 0)
variable (h₂ : (b - 1)^2 - 4 * a * (c + 2) = 0)
variable (h₃ : (b + 1/2)^2 - 4 * a * (c - 1) = 0)

theorem quadratic_discriminant : b^2 - 4 * a * c = -1 / 2 := 
by
  have h₁' : (b - 1)^2 - 4 * a * (c + 2) = 0 := h₂
  have h₂' : (b + 1/2)^2 - 4 * a * (c - 1) = 0 := h₃
  sorry

end quadratic_discriminant_l144_144361


namespace locus_of_midpoint_l144_144001

theorem locus_of_midpoint
  (x y : ℝ)
  (h : ∃ (A : ℝ × ℝ), A = (2*x, 2*y) ∧ (A.1)^2 + (A.2)^2 - 8*A.1 = 0) :
  x^2 + y^2 - 4*x = 0 :=
by
  sorry

end locus_of_midpoint_l144_144001


namespace geometric_progression_exists_l144_144042

theorem geometric_progression_exists :
  ∃ (b_1 b_2 b_3 b_4 q : ℚ), 
    b_1 - b_2 = 35 ∧ 
    b_3 - b_4 = 560 ∧ 
    b_2 = b_1 * q ∧ 
    b_3 = b_1 * q^2 ∧ 
    b_4 = b_1 * q^3 ∧ 
    ((b_1 = 7 ∧ q = -4 ∧ b_2 = -28 ∧ b_3 = 112 ∧ b_4 = -448) ∨ 
    (b_1 = -35/3 ∧ q = 4 ∧ b_2 = -140/3 ∧ b_3 = -560/3 ∧ b_4 = -2240/3)) :=
by
  sorry

end geometric_progression_exists_l144_144042


namespace john_weekly_loss_is_525000_l144_144866

-- Define the constants given in the problem
def daily_production : ℕ := 1000
def production_cost_per_tire : ℝ := 250
def selling_price_factor : ℝ := 1.5
def potential_daily_sales : ℕ := 1200
def days_in_week : ℕ := 7

-- Define the selling price per tire
def selling_price_per_tire : ℝ := production_cost_per_tire * selling_price_factor

-- Define John's current daily earnings from selling 1000 tires
def current_daily_earnings : ℝ := daily_production * selling_price_per_tire

-- Define John's potential daily earnings from selling 1200 tires
def potential_daily_earnings : ℝ := potential_daily_sales * selling_price_per_tire

-- Define the daily loss by not being able to produce all the tires
def daily_loss : ℝ := potential_daily_earnings - current_daily_earnings

-- Define the weekly loss
def weekly_loss : ℝ := daily_loss * days_in_week

-- Statement: Prove that John's weekly financial loss is $525,000
theorem john_weekly_loss_is_525000 : weekly_loss = 525000 :=
by
  sorry

end john_weekly_loss_is_525000_l144_144866


namespace arithmetic_sequence_a6_l144_144706

theorem arithmetic_sequence_a6 (a : ℕ → ℝ)
  (h4_8 : ∃ a4 a8, (a 4 = a4) ∧ (a 8 = a8) ∧ a4^2 - 6*a4 + 5 = 0 ∧ a8^2 - 6*a8 + 5 = 0) :
  a 6 = 3 := by 
  sorry

end arithmetic_sequence_a6_l144_144706


namespace population_percentage_l144_144778

-- Definitions based on the given conditions
def percentage (part : ℕ) (whole : ℕ) : ℕ := (part * 100) / whole

-- Conditions from the problem statement
def part_population : ℕ := 23040
def total_population : ℕ := 25600

-- The theorem stating that the percentage is 90
theorem population_percentage : percentage part_population total_population = 90 :=
  by
    -- Proof steps would go here, we only need to state the theorem
    sorry

end population_percentage_l144_144778


namespace power_expression_l144_144681

theorem power_expression (a b : ℕ) (h1 : a = 12) (h2 : b = 18) : (3^a * 3^b) = (243^6) :=
by
  let c := 3
  have h3 : a + b = 30 := by simp [h1, h2]
  have h4 : 3^(a + b) = 3^30 := by rw [h3]
  have h5 : 3^30 = 243^6 := by norm_num
  sorry  -- skip other detailed steps

end power_expression_l144_144681


namespace solve_equation_l144_144200

noncomputable def smallest_solution : Rat :=
  (8 - Real.sqrt 145) / 3

theorem solve_equation : 
  ∃ x : ℝ, (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ x = smallest_solution := sorry

end solve_equation_l144_144200


namespace cost_per_rug_proof_l144_144420

noncomputable def cost_per_rug (price_sold : ℝ) (number_rugs : ℕ) (profit : ℝ) : ℝ :=
  let total_revenue := number_rugs * price_sold
  let total_cost := total_revenue - profit
  total_cost / number_rugs

theorem cost_per_rug_proof : cost_per_rug 60 20 400 = 40 :=
by
  -- Lean will need the proof steps here, which are skipped
  -- The solution steps illustrate how Lean would derive this in a proof
  sorry

end cost_per_rug_proof_l144_144420


namespace cogs_produced_after_speed_increase_l144_144162

-- Define the initial conditions of the problem
def initial_cogs := 60
def initial_rate := 15
def increased_rate := 60
def average_output := 24

-- Variables to represent the number of cogs produced after the speed increase and the total time taken for each phase
variable (x : ℕ)

-- Assuming the equations representing the conditions
def initial_time := initial_cogs / initial_rate
def increased_time := x / increased_rate

def total_cogs := initial_cogs + x
def total_time := initial_time + increased_time

-- Define the overall average output equation
def average_eq := average_output * total_time = total_cogs

-- The proposition we want to prove
theorem cogs_produced_after_speed_increase : x = 60 :=
by
  -- Using the equation from the conditions
  have h1 : average_eq := sorry
  sorry

end cogs_produced_after_speed_increase_l144_144162


namespace total_pennies_l144_144482

variable (C J : ℕ)

def cassandra_pennies : ℕ := 5000
def james_pennies (C : ℕ) : ℕ := C - 276

theorem total_pennies (hC : C = cassandra_pennies) (hJ : J = james_pennies C) :
  C + J = 9724 :=
by
  sorry

end total_pennies_l144_144482


namespace arithmetic_sqrt_of_9_l144_144903

theorem arithmetic_sqrt_of_9 : (∃ (sqrt : ℝ), sqrt = 3 ∧ ∀ x, x*x = 9 → x = sqrt) :=
by
  existsi (3 : ℝ)
  split
  exact rfl
  intros x hx
  exact sqrt_unique hx 3

end arithmetic_sqrt_of_9_l144_144903


namespace problem_1_system_solution_problem_2_system_solution_l144_144739

theorem problem_1_system_solution (x y : ℝ)
  (h1 : x - 2 * y = 1)
  (h2 : 4 * x + 3 * y = 26) :
  x = 5 ∧ y = 2 :=
sorry

theorem problem_2_system_solution (x y : ℝ)
  (h1 : 2 * x + 3 * y = 3)
  (h2 : 5 * x - 3 * y = 18) :
  x = 3 ∧ y = -1 :=
sorry

end problem_1_system_solution_problem_2_system_solution_l144_144739


namespace tan_405_eq_1_l144_144168

theorem tan_405_eq_1 : Real.tan (405 * Real.pi / 180) = 1 :=
by
  have h1 : (405 * Real.pi / 180) = (45 * Real.pi / 180 + 2 * Real.pi), by norm_num
  rw [h1, Real.tan_add_two_pi]
  exact Real.tan_pi_div_four.symm

end tan_405_eq_1_l144_144168


namespace arithmetic_square_root_of_nine_l144_144908

theorem arithmetic_square_root_of_nine : ∃ (x : ℝ), (x * x = 9) ∧ (x ≥ 0) ∧ (x = 3) :=
by
  sorry

end arithmetic_square_root_of_nine_l144_144908


namespace exposed_sides_correct_l144_144628

-- Define the number of sides of each polygon
def sides_triangle := 3
def sides_square := 4
def sides_pentagon := 5
def sides_hexagon := 6
def sides_heptagon := 7

-- Total sides from all polygons
def total_sides := sides_triangle + sides_square + sides_pentagon + sides_hexagon + sides_heptagon

-- Number of shared sides
def shared_sides := 4

-- Final number of exposed sides
def exposed_sides := total_sides - shared_sides

-- Statement to prove
theorem exposed_sides_correct : exposed_sides = 21 :=
by {
  -- This part will contain the proof which we do not need. Replace with 'sorry' for now.
  sorry
}

end exposed_sides_correct_l144_144628


namespace student_history_score_l144_144787

theorem student_history_score 
  (math : ℕ) 
  (third : ℕ) 
  (average : ℕ) 
  (H : ℕ) 
  (h_math : math = 74)
  (h_third : third = 67)
  (h_avg : average = 75)
  (h_overall_avg : (math + third + H) / 3 = average) : 
  H = 84 :=
by
  sorry

end student_history_score_l144_144787


namespace find_4a_add_c_find_2a_sub_2b_sub_c_l144_144057

variables {R : Type*} [CommRing R]

theorem find_4a_add_c (a b c : ℝ) (h : ∀ x : ℝ, (x^3 + a * x^2 + b * x + c) = (x^2 + 3 * x - 4) * (x + (a - 3) - b + 4 - c)) :
  4 * a + c = 12 :=
sorry

theorem find_2a_sub_2b_sub_c (a b c : ℝ) (h : ∀ x : ℝ, (x^3 + a * x^2 + b * x + c) = (x^2 + 3 * x - 4) * (x + (a - 3) - b + 4 - c)) :
  2 * a - 2 * b - c = 14 :=
sorry

end find_4a_add_c_find_2a_sub_2b_sub_c_l144_144057


namespace max_f_value_l144_144821

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 + Real.sqrt 3 * Real.cos x - 3 / 4

theorem max_f_value : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 1 ∧ ∃ x₀ ∈ Set.Icc 0 (Real.pi / 2), f x₀ = 1 :=
by
  sorry

end max_f_value_l144_144821


namespace print_shop_x_charges_l144_144046

theorem print_shop_x_charges (x : ℝ) (h1 : ∀ y : ℝ, y = 1.70) (h2 : 40 * x + 20 = 40 * 1.70) : x = 1.20 :=
by
  sorry

end print_shop_x_charges_l144_144046


namespace inequality_solution_l144_144890

theorem inequality_solution (x y : ℝ) : 
  (x^2 - 4 * x * y + 4 * x^2 < x^2) ↔ (x < y ∧ y < 3 * x ∧ x > 0) := 
sorry

end inequality_solution_l144_144890


namespace books_per_shelf_l144_144879

theorem books_per_shelf (total_books : ℕ) (books_taken : ℕ) (shelves : ℕ) (books_left : ℕ) (books_per_shelf : ℕ) :
  total_books = 46 →
  books_taken = 10 →
  shelves = 9 →
  books_left = total_books - books_taken →
  books_per_shelf = books_left / shelves →
  books_per_shelf = 4 :=
by
  sorry

end books_per_shelf_l144_144879


namespace smallest_base_for_101_l144_144962

theorem smallest_base_for_101 : ∃ b : ℕ, b = 10 ∧ b ≤ 101 ∧ 101 < b^2 :=
by
  -- We state the simplest form of the theorem,
  -- then use the answer from the solution step.
  use 10
  sorry

end smallest_base_for_101_l144_144962


namespace y_intercept_of_line_l144_144835

def line_equation (x y : ℝ) : Prop := x - 2 * y + 4 = 0

theorem y_intercept_of_line : ∀ y : ℝ, line_equation 0 y → y = 2 :=
by 
  intro y h
  unfold line_equation at h
  sorry

end y_intercept_of_line_l144_144835


namespace total_pennies_l144_144481

variable (C J : ℕ)

def cassandra_pennies : ℕ := 5000
def james_pennies (C : ℕ) : ℕ := C - 276

theorem total_pennies (hC : C = cassandra_pennies) (hJ : J = james_pennies C) :
  C + J = 9724 :=
by
  sorry

end total_pennies_l144_144481


namespace find_a_find_k_max_l144_144834

-- Problem 1
theorem find_a (f : ℝ → ℝ) (a : ℝ) 
  (hf : ∀ x, f x = x * (a + Real.log x))
  (hmin : ∃ x, f x = -Real.exp (-2) ∧ ∀ y, f y ≥ f x) : a = 1 := 
sorry

-- Problem 2
theorem find_k_max {k : ℤ} : 
  (∀ x > 1, k < (x * (1 + Real.log x)) / (x - 1)) → k ≤ 3 :=
sorry

end find_a_find_k_max_l144_144834


namespace hexagon_area_l144_144314

theorem hexagon_area (s t_height : ℕ) (tri_area rect_area : ℕ) :
    s = 2 →
    t_height = 4 →
    tri_area = 1 / 2 * s * t_height →
    rect_area = (s + s + s) * (t_height + t_height) →
    rect_area - 4 * tri_area = 32 :=
by
  sorry

end hexagon_area_l144_144314


namespace solution_set_of_3x2_minus_7x_gt_6_l144_144337

theorem solution_set_of_3x2_minus_7x_gt_6 (x : ℝ) :
  3 * x^2 - 7 * x > 6 ↔ (x < -2 / 3 ∨ x > 3) := 
by
  sorry

end solution_set_of_3x2_minus_7x_gt_6_l144_144337


namespace sum_of_possible_coefficient_values_l144_144874

theorem sum_of_possible_coefficient_values :
  let pairs := [(1, 48), (2, 24), (3, 16), (4, 12), (6, 8)]
  let values := pairs.map (fun (r, s) => r + s)
  values.sum = 124 :=
by
  sorry

end sum_of_possible_coefficient_values_l144_144874


namespace jill_arrives_before_jack_l144_144714

theorem jill_arrives_before_jack
  (distance : ℝ)
  (jill_speed : ℝ)
  (jack_speed : ℝ)
  (jill_time_minutes : ℝ)
  (jack_time_minutes : ℝ) :
  distance = 2 →
  jill_speed = 15 →
  jack_speed = 6 →
  jill_time_minutes = (distance / jill_speed) * 60 →
  jack_time_minutes = (distance / jack_speed) * 60 →
  jack_time_minutes - jill_time_minutes = 12 :=
by
  sorry

end jill_arrives_before_jack_l144_144714


namespace geometric_series_common_ratio_l144_144817

theorem geometric_series_common_ratio (r : ℚ) : 
  (∃ (a : ℚ), a = 4 / 7 ∧ a * r = 16 / 21) → r = 4 / 3 :=
by
  sorry

end geometric_series_common_ratio_l144_144817


namespace arithmetic_square_root_of_nine_l144_144904

theorem arithmetic_square_root_of_nine : ∃ (x : ℝ), (x * x = 9) ∧ (x ≥ 0) ∧ (x = 3) :=
by
  sorry

end arithmetic_square_root_of_nine_l144_144904


namespace total_resistance_l144_144171

theorem total_resistance (R₀ : ℝ) (h : R₀ = 10) : 
  let R₃ := R₀; let R₄ := R₀; let R₃₄ := R₃ + R₄;
  let R₂ := R₀; let R₅ := R₀; let R₂₃₄ := 1 / (1 / R₂ + 1 / R₃₄ + 1 / R₅);
  let R₁ := R₀; let R₆ := R₀; let R₁₂₃₄ := R₁ + R₂₃₄ + R₆;
  R₁₂₃₄ = 13.33 :=
by 
  sorry

end total_resistance_l144_144171


namespace apples_per_pie_l144_144419

-- Definitions of the conditions
def number_of_pies : ℕ := 10
def harvested_apples : ℕ := 50
def to_buy_apples : ℕ := 30
def total_apples_needed : ℕ := harvested_apples + to_buy_apples

-- The theorem to prove
theorem apples_per_pie :
  (total_apples_needed / number_of_pies) = 8 := 
sorry

end apples_per_pie_l144_144419


namespace Linda_total_distance_is_25_l144_144727

theorem Linda_total_distance_is_25 : 
  ∃ (x : ℤ), x > 0 ∧ 
  (60/x + 60/(x+5) + 60/(x+10) + 60/(x+15) = 25) :=
by 
  sorry

end Linda_total_distance_is_25_l144_144727


namespace john_ultramarathon_distance_l144_144077

theorem john_ultramarathon_distance :
  let initial_time := 8
  let time_increase_percentage := 0.75
  let speed_increase := 4
  let initial_speed := 8
  initial_time * (1 + time_increase_percentage) * (initial_speed + speed_increase) = 168 :=
by
  let initial_time := 8
  let time_increase_percentage := 0.75
  let speed_increase := 4
  let initial_speed := 8
  sorry

end john_ultramarathon_distance_l144_144077


namespace min_waiting_time_max_waiting_time_expected_waiting_time_l144_144301

open Nat

noncomputable def C : ℕ → ℕ → ℕ
| n, 0     => 1
| 0, k     => 0
| n+1, k+1 => C n k + C n (k+1)

def a := 1
def b := 5
def n := 5
def m := 3

def T_min := a * C (n - 1) 2 + m * n * a + b * C m 2
def T_max := a * C n 2 + b * m * n + b * C m 2
def E_T := C (n + m) 2 * (b * m + a * n) / (m + n)

theorem min_waiting_time : T_min = 40 := by
  sorry

theorem max_waiting_time : T_max = 100 := by
  sorry

theorem expected_waiting_time : E_T = 70 := by
  sorry

end min_waiting_time_max_waiting_time_expected_waiting_time_l144_144301


namespace elaine_earnings_increase_l144_144535

variable (E P : ℝ)

theorem elaine_earnings_increase :
  (0.25 * (E * (1 + P / 100)) = 1.4375 * 0.20 * E) → P = 15 :=
by
  intro h
  -- Start an intermediate transformation here
  sorry

end elaine_earnings_increase_l144_144535


namespace calculate_perimeter_of_staircase_region_l144_144074

-- Define the properties and dimensions of the staircase-shaped region
def is_right_angle (angle : ℝ) : Prop := angle = 90

def congruent_side_length : ℝ := 1

def bottom_base_length : ℝ := 12

def total_area : ℝ := 78

def perimeter_region : ℝ := 34.5

theorem calculate_perimeter_of_staircase_region
  (is_right_angle : ∀ angle, is_right_angle angle)
  (congruent_sides_count : ℕ := 12)
  (total_congruent_side_length : ℝ := congruent_sides_count * congruent_side_length)
  (bottom_base_length : ℝ)
  (total_area : ℝ)
  : bottom_base_length = 12 ∧ total_area = 78 → 
    ∃ perimeter : ℝ, perimeter = 34.5 :=
by
  admit -- Proof goes here

end calculate_perimeter_of_staircase_region_l144_144074


namespace smallest_solution_to_equation_l144_144194

noncomputable def smallest_solution := (11 - Real.sqrt 445) / 6

theorem smallest_solution_to_equation:
  ∃ x : ℝ, (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ (x = smallest_solution) :=
sorry

end smallest_solution_to_equation_l144_144194


namespace arithmetic_expression_evaluation_l144_144031

theorem arithmetic_expression_evaluation :
  (1 / 6 * -6 / (-1 / 6) * 6) = 36 :=
by {
  sorry
}

end arithmetic_expression_evaluation_l144_144031


namespace joan_dozen_of_eggs_l144_144863

def number_of_eggs : ℕ := 72
def dozen : ℕ := 12

theorem joan_dozen_of_eggs : (number_of_eggs / dozen) = 6 := by
  sorry

end joan_dozen_of_eggs_l144_144863


namespace flyers_left_l144_144240

theorem flyers_left (initial_flyers : ℕ) (jack_flyers : ℕ) (rose_flyers : ℕ) (left_flyers : ℕ) :
  initial_flyers = 1236 →
  jack_flyers = 120 →
  rose_flyers = 320 →
  left_flyers = 796 →
  initial_flyers - (jack_flyers + rose_flyers) = left_flyers := 
by
  intros h_initial h_jack h_rose h_left
  rw [h_initial, h_jack, h_rose, h_left]
  simp
  sorry

end flyers_left_l144_144240


namespace least_element_in_valid_set_l144_144537

noncomputable def is_valid_set (T : Set ℕ) : Prop :=
  ∀ c d ∈ T, c < d → ¬ (d % c = 0)

noncomputable def is_prime_or_gt_15 (n : ℕ) : Prop :=
  Prime n ∨ (15 < n ∧ n ≤ 18)

theorem least_element_in_valid_set :
  ∃ (T : Set ℕ), T ⊆ {n | 1 ≤ n ∧ n ≤ 18} ∧ is_valid_set T ∧ (∀ t ∈ T, is_prime_or_gt_15 t) ∧ (T.card = 8) ∧ (∀ t ∈ T, 2 ∈ T ∧ (∀ x ∈ T, 2 ≤ x)) :=
sorry

end least_element_in_valid_set_l144_144537


namespace average_branches_per_foot_correct_l144_144635

def height_tree_1 : ℕ := 50
def branches_tree_1 : ℕ := 200
def height_tree_2 : ℕ := 40
def branches_tree_2 : ℕ := 180
def height_tree_3 : ℕ := 60
def branches_tree_3 : ℕ := 180
def height_tree_4 : ℕ := 34
def branches_tree_4 : ℕ := 153

def total_height := height_tree_1 + height_tree_2 + height_tree_3 + height_tree_4
def total_branches := branches_tree_1 + branches_tree_2 + branches_tree_3 + branches_tree_4
def average_branches_per_foot := total_branches / total_height

theorem average_branches_per_foot_correct : average_branches_per_foot = 713 / 184 := 
  by
    -- Proof omitted, directly state the result
    sorry

end average_branches_per_foot_correct_l144_144635


namespace noemi_initial_money_l144_144422

variable (money_lost_roulette : ℕ := 400)
variable (money_lost_blackjack : ℕ := 500)
variable (money_left : ℕ)
variable (money_started : ℕ)

axiom money_left_condition : money_left > 0
axiom total_loss_condition : money_lost_roulette + money_lost_blackjack = 900

theorem noemi_initial_money (h1 : money_lost_roulette = 400) (h2 : money_lost_blackjack = 500)
    (h3 : money_started - 900 = money_left) (h4 : money_left > 0) :
    money_started > 900 := by
  sorry

end noemi_initial_money_l144_144422


namespace sum_of_digits_second_smallest_multiple_l144_144082

theorem sum_of_digits_second_smallest_multiple :
  sum_of_digits (2 * nat.lcm (list.range 9).tail) = 15 :=
by
  sorry

end sum_of_digits_second_smallest_multiple_l144_144082


namespace greatest_integer_y_l144_144960

theorem greatest_integer_y (y : ℤ) : abs (3 * y - 4) ≤ 21 → y ≤ 8 :=
by
  sorry

end greatest_integer_y_l144_144960


namespace neg_p_sufficient_not_necessary_q_l144_144085

theorem neg_p_sufficient_not_necessary_q (p q : Prop) 
  (h₁ : p → ¬q) 
  (h₂ : ¬(¬q → p)) : (q → ¬p) ∧ ¬(¬p → q) :=
sorry

end neg_p_sufficient_not_necessary_q_l144_144085


namespace largest_of_seven_consecutive_integers_l144_144123

-- Define the main conditions as hypotheses
theorem largest_of_seven_consecutive_integers (n : ℕ) (h_sum : 7 * n + 21 = 2401) : 
  n + 6 = 346 :=
by
  -- Conditions from the problem are utilized here
  sorry

end largest_of_seven_consecutive_integers_l144_144123


namespace average_branches_per_foot_l144_144637

theorem average_branches_per_foot :
  let b1 := 200
  let h1 := 50
  let b2 := 180
  let h2 := 40
  let b3 := 180
  let h3 := 60
  let b4 := 153
  let h4 := 34
  (b1 / h1 + b2 / h2 + b3 / h3 + b4 / h4) / 4 = 4 := by
  sorry

end average_branches_per_foot_l144_144637


namespace lcm_36_98_is_1764_l144_144653

theorem lcm_36_98_is_1764 : Nat.lcm 36 98 = 1764 := by
  sorry

end lcm_36_98_is_1764_l144_144653


namespace number_of_cars_in_trains_l144_144280

theorem number_of_cars_in_trains
  (s1 s2 s3 : ℕ)
  (h1 : s1 = 462)
  (h2 : s2 = 546)
  (h3 : s3 = 630)
  (g : ℕ := Nat.gcd (Nat.gcd s1 s2) s3)
  (h_g : g = 42) :
  (s1 / g = 11) ∧ (s2 / g = 13) ∧ (s3 / g = 15) :=
by
  rw [h1, h2, h3, h_g]
  norm_num
  exact dec_trivial

end number_of_cars_in_trains_l144_144280


namespace distance_from_P_to_x_axis_l144_144857

-- Define the point P with coordinates (4, -3)
def P : ℝ × ℝ := (4, -3)

-- Define the distance from a point to the x-axis as the absolute value of the y-coordinate
def distance_to_x_axis (point : ℝ × ℝ) : ℝ :=
  abs point.snd

-- State the theorem to be proved
theorem distance_from_P_to_x_axis : distance_to_x_axis P = 3 :=
by
  -- The proof is not required; we can use sorry to skip it
  sorry

end distance_from_P_to_x_axis_l144_144857


namespace total_questions_needed_l144_144099

def m_total : ℕ := 35
def p_total : ℕ := 15
def t_total : ℕ := 20

def m_written : ℕ := (3 * m_total) / 7
def p_written : ℕ := p_total / 5
def t_written : ℕ := t_total / 4

def m_remaining : ℕ := m_total - m_written
def p_remaining : ℕ := p_total - p_written
def t_remaining : ℕ := t_total - t_written

def total_remaining : ℕ := m_remaining + p_remaining + t_remaining

theorem total_questions_needed : total_remaining = 47 := by
  sorry

end total_questions_needed_l144_144099


namespace symmetric_line_eq_l144_144969

theorem symmetric_line_eq (x y : ℝ) : 
  (∀ (x y : ℝ), y = -2 * x - 3 → y = 2 * (-x) - 3) :=
by 
  assume x y h,
  sorry

end symmetric_line_eq_l144_144969


namespace mass_percentage_of_calcium_in_calcium_oxide_l144_144499

theorem mass_percentage_of_calcium_in_calcium_oxide
  (Ca_molar_mass : ℝ)
  (O_molar_mass : ℝ)
  (Ca_mass : Ca_molar_mass = 40.08)
  (O_mass : O_molar_mass = 16.00) :
  ((Ca_molar_mass / (Ca_molar_mass + O_molar_mass)) * 100) = 71.45 :=
by
  sorry

end mass_percentage_of_calcium_in_calcium_oxide_l144_144499


namespace total_visit_plans_l144_144892

def exhibitions : List String := ["Opera Culture Exhibition", "Ming Dynasty Imperial Cellar Porcelain Exhibition", "Historical Green Landscape Painting Exhibition", "Zhao Mengfu Calligraphy and Painting Exhibition"]

def painting_exhibitions : List String := ["Historical Green Landscape Painting Exhibition", "Zhao Mengfu Calligraphy and Painting Exhibition"]

def non_painting_exhibitions : List String := ["Opera Culture Exhibition", "Ming Dynasty Imperial Cellar Porcelain Exhibition"]

def num_visit_plans (exhibit_list : List String) (paintings : List String) (non_paintings : List String) : Nat :=
  let case1 := paintings.length * non_paintings.length * 2
  let case2 := if paintings.length >= 2 then 2 else 0
  case1 + case2

theorem total_visit_plans : num_visit_plans exhibitions painting_exhibitions non_painting_exhibitions = 10 :=
  sorry

end total_visit_plans_l144_144892


namespace arithmetic_sqrt_of_nine_l144_144923

-- Define the arithmetic square root function which only considers non-negative values
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ :=
  if hx : x ≥ 0 then Real.sqrt x else 0

-- The theorem to prove: The arithmetic square root of 9 is 3.
theorem arithmetic_sqrt_of_nine : arithmetic_sqrt 9 = 3 :=
by
  sorry

end arithmetic_sqrt_of_nine_l144_144923


namespace crayons_given_proof_l144_144103

def initial_crayons : ℕ := 110
def total_lost_crayons : ℕ := 412
def more_lost_than_given : ℕ := 322

def G : ℕ := 45 -- This is the given correct answer to prove.

theorem crayons_given_proof :
  ∃ G : ℕ, (G + (G + more_lost_than_given)) = total_lost_crayons ∧ G = 45 :=
by
  sorry

end crayons_given_proof_l144_144103


namespace arithmetic_square_root_of_nine_l144_144907

theorem arithmetic_square_root_of_nine : ∃ (x : ℝ), (x * x = 9) ∧ (x ≥ 0) ∧ (x = 3) :=
by
  sorry

end arithmetic_square_root_of_nine_l144_144907


namespace quadratic_discriminant_l144_144360

variable {a b c : ℝ}
variable (h₁ : a ≠ 0)
variable (h₂ : (b - 1)^2 - 4 * a * (c + 2) = 0)
variable (h₃ : (b + 1/2)^2 - 4 * a * (c - 1) = 0)

theorem quadratic_discriminant : b^2 - 4 * a * c = -1 / 2 := 
by
  have h₁' : (b - 1)^2 - 4 * a * (c + 2) = 0 := h₂
  have h₂' : (b + 1/2)^2 - 4 * a * (c - 1) = 0 := h₃
  sorry

end quadratic_discriminant_l144_144360


namespace lcm_ac_least_value_l144_144943

theorem lcm_ac_least_value (a b c : ℕ) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 24) : 
  Nat.lcm a c = 30 :=
sorry

end lcm_ac_least_value_l144_144943


namespace trajectory_eqn_of_point_Q_l144_144284

theorem trajectory_eqn_of_point_Q 
  (P : ℝ × ℝ)
  (Q : ℝ × ℝ)
  (A : ℝ × ℝ := (-2, 0))
  (B : ℝ × ℝ := (2, 0))
  (l : ℝ := 10 / 3) 
  (hP_on_l : P.1 = l)
  (hQ_on_AP : (Q.2 * -4) = Q.1 * (P.2 - 0) - (P.2 * -4))
  (hBP_perp_BQ : (Q.2 * 4) = -Q.1 * ((3 * P.2) / 4 - 2))
: (Q.1^2 / 4) + Q.2^2 = 1 :=
sorry

end trajectory_eqn_of_point_Q_l144_144284


namespace range_of_m_l144_144122

def f (x : ℝ) : ℝ := -x^3 - 2*x^2 + 4*x

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≥ m^2 - 14 * m) ↔ 3 ≤ m ∧ m ≤ 11 :=
by
  sorry

end range_of_m_l144_144122


namespace total_plants_to_buy_l144_144882

theorem total_plants_to_buy (rows plants_per_row additional_plants : ℕ) 
  (h1 : rows = 7) (h2 : plants_per_row = 18) (h3 : additional_plants = 15) : 
  rows * plants_per_row + additional_plants = 141 :=
by
  -- Definitions from conditions
  rw [h1, h2, h3]
  -- Simplify the expression
  sorry

end total_plants_to_buy_l144_144882


namespace smallest_solution_to_equation_l144_144192

noncomputable def smallest_solution := (11 - Real.sqrt 445) / 6

theorem smallest_solution_to_equation:
  ∃ x : ℝ, (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ (x = smallest_solution) :=
sorry

end smallest_solution_to_equation_l144_144192


namespace flyers_left_l144_144244

theorem flyers_left (total_flyers : ℕ) (jack_flyers : ℕ) (rose_flyers : ℕ) (h1 : total_flyers = 1236) (h2 : jack_flyers = 120) (h3 : rose_flyers = 320) : (total_flyers - (jack_flyers + rose_flyers) = 796) := 
by
  sorry

end flyers_left_l144_144244


namespace find_a_l144_144212

theorem find_a (a : ℝ) (h : ∃ x, x = -1 ∧ 4 * x^3 + 2 * a * x = 8) : a = -6 :=
sorry

end find_a_l144_144212


namespace range_of_x2_plus_y2_l144_144573

theorem range_of_x2_plus_y2 (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f x = -f (-x))
  (h_increasing : ∀ x y : ℝ, x < y → f x < f y)
  (x y : ℝ)
  (h_inequality : f (x^2 - 6 * x) + f (y^2 - 8 * y + 24) < 0) :
  16 < x^2 + y^2 ∧ x^2 + y^2 < 36 :=
sorry

end range_of_x2_plus_y2_l144_144573


namespace isosceles_triangle_height_l144_144163

theorem isosceles_triangle_height (s h : ℝ) (eq_areas : (2 * s * s) = (1/2 * s * h)) : h = 4 * s :=
by
  sorry

end isosceles_triangle_height_l144_144163


namespace polynomial_discriminant_l144_144374

theorem polynomial_discriminant (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
by
  sorry

end polynomial_discriminant_l144_144374


namespace arithmetic_sqrt_of_9_l144_144901

theorem arithmetic_sqrt_of_9 : (∃ (sqrt : ℝ), sqrt = 3 ∧ ∀ x, x*x = 9 → x = sqrt) :=
by
  existsi (3 : ℝ)
  split
  exact rfl
  intros x hx
  exact sqrt_unique hx 3

end arithmetic_sqrt_of_9_l144_144901


namespace interest_percentage_correct_l144_144464

noncomputable def encyclopedia_cost : ℝ := 1200
noncomputable def down_payment : ℝ := 500
noncomputable def monthly_payment : ℝ := 70
noncomputable def final_payment : ℝ := 45
noncomputable def num_monthly_payments : ℕ := 12
noncomputable def total_installment_payments : ℝ := (num_monthly_payments * monthly_payment) + final_payment
noncomputable def total_cost_paid : ℝ := total_installment_payments + down_payment
noncomputable def amount_borrowed : ℝ := encyclopedia_cost - down_payment
noncomputable def interest_paid : ℝ := total_cost_paid - encyclopedia_cost
noncomputable def interest_percentage : ℝ := (interest_paid / amount_borrowed) * 100

theorem interest_percentage_correct : interest_percentage = 26.43 := by
  sorry

end interest_percentage_correct_l144_144464


namespace common_ratio_of_geometric_series_l144_144815

theorem common_ratio_of_geometric_series (a b : ℚ) (h1 : a = 4 / 7) (h2 : b = 16 / 21) :
  b / a = 4 / 3 :=
by
  sorry

end common_ratio_of_geometric_series_l144_144815


namespace Razorback_tshirt_problem_l144_144565

theorem Razorback_tshirt_problem
  (A T : ℕ)
  (h1 : A + T = 186)
  (h2 : 78 * T = 1092) :
  A = 172 := by
  sorry

end Razorback_tshirt_problem_l144_144565


namespace solve_for_r_l144_144112

theorem solve_for_r (r s : ℚ) (h : (2 * (r - 45)) / 3 = (3 * s - 2 * r) / 4) (s_val : s = 20) :
  r = 270 / 7 :=
by
  sorry

end solve_for_r_l144_144112


namespace fibonacci_arith_sequence_a_eq_665_l144_144891

theorem fibonacci_arith_sequence_a_eq_665 (F : ℕ → ℕ) (a b c : ℕ) :
  (F 1 = 1) →
  (F 2 = 1) →
  (∀ n, n ≥ 3 → F n = F (n - 1) + F (n - 2)) →
  (a + b + c = 2000) →
  (F a < F b ∧ F b < F c ∧ F b - F a = F c - F b) →
  a = 665 :=
by
  sorry

end fibonacci_arith_sequence_a_eq_665_l144_144891


namespace candidate_knows_Excel_and_willing_nights_l144_144978

variable (PExcel PXNight : ℝ)
variable (H1 : PExcel = 0.20) (H2 : PXNight = 0.30)

theorem candidate_knows_Excel_and_willing_nights : (PExcel * PXNight) = 0.06 :=
by
  rw [H1, H2]
  norm_num

end candidate_knows_Excel_and_willing_nights_l144_144978


namespace cylinder_lateral_area_cylinder_volume_cylinder_surface_area_cone_volume_l144_144028

-- Problem 1
theorem cylinder_lateral_area (C H : ℝ) (hC : C = 1.8) (hH : H = 1.5) :
  C * H = 2.7 := by sorry 

-- Problem 2
theorem cylinder_volume (D H : ℝ) (hD : D = 3) (hH : H = 8) :
  (3.14 * ((D * 10 / 2) ^ 2) * H) = 5652 :=
by sorry

-- Problem 3
theorem cylinder_surface_area (r h : ℝ) (hr : r = 6) (hh : h = 5) :
    (3.14 * r * 2 * h + 3.14 * r ^ 2 * 2) = 414.48 :=
by sorry

-- Problem 4
theorem cone_volume (B H : ℝ) (hB : B = 18.84) (hH : H = 6) :
  (1 / 3 * B * H) = 37.68 :=
by sorry

end cylinder_lateral_area_cylinder_volume_cylinder_surface_area_cone_volume_l144_144028


namespace strawberry_jelly_sales_l144_144010

def jelly_sales (grape strawberry raspberry plum : ℕ) : Prop :=
  grape = 2 * strawberry ∧
  raspberry = 2 * plum ∧
  raspberry = grape / 3 ∧
  plum = 6

theorem strawberry_jelly_sales {grape strawberry raspberry plum : ℕ}
    (h : jelly_sales grape strawberry raspberry plum) : 
    strawberry = 18 :=
by
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2.1
  have h4 := h.2.2.2
  sorry

end strawberry_jelly_sales_l144_144010


namespace prob_four_children_at_least_one_boy_one_girl_l144_144611

-- Define the probability of a single birth being a boy or a girl
def prob_boy_or_girl : ℚ := 1/2

-- Calculate the probability of all children being boys or all girls
def prob_all_boys : ℚ := (prob_boy_or_girl)^4
def prob_all_girls : ℚ := (prob_boy_or_girl)^4

-- Calculate the probability of having neither all boys nor all girls
def prob_at_least_one_boy_one_girl : ℚ := 1 - (prob_all_boys + prob_all_girls)

-- The theorem to prove
theorem prob_four_children_at_least_one_boy_one_girl : 
  prob_at_least_one_boy_one_girl = 7/8 := 
by 
  sorry

end prob_four_children_at_least_one_boy_one_girl_l144_144611


namespace probability_at_least_one_boy_and_one_girl_l144_144623

theorem probability_at_least_one_boy_and_one_girl :
  (∀ (n : ℕ), (ℙ(birth_is_boy) = ℙ(birth_is_girl)) ∧ n = 4) →
  (∃ p : ℚ, p = 7 / 8 ∧
    p = 1 - (ℙ(all_boys) + ℙ(all_girls))) :=
by
  sorry

-- Definitions to be used
def birth_is_boy := sorry -- Placeholder for an event where a birth is a boy
def birth_is_girl := sorry -- Placeholder for an event where a birth is a girl
def all_boys := sorry -- Placeholder for an event where all four children are boys
def all_girls := sorry -- Placeholder for an event where all four children are girls

end probability_at_least_one_boy_and_one_girl_l144_144623


namespace quadratic_polynomial_discriminant_l144_144381

def P (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₀ : a ≠ 0)
  (h₁ : ∃ x : ℝ, P a b c x = x - 2 ∧ (discriminant a (b - 1) (c + 2) = 0))
  (h₂ : ∃ x : ℝ, P a b c x = 1 - x / 2 ∧ (discriminant a (b + 1 / 2) (c - 1) = 0)) :
  discriminant a b c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l144_144381


namespace binary_to_decimal_conversion_l144_144568

theorem binary_to_decimal_conversion : (1 * 2^2 + 1 * 2^1 + 0 * 2^0 = 6) := by
  sorry

end binary_to_decimal_conversion_l144_144568


namespace geometric_sequence_problem_l144_144709

-- Definitions
def is_geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop := ∀ n, a (n + 1) = q * a n

-- Problem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ)
    (h_geom : is_geom_seq a q)
    (h1 : a 3 * a 7 = 8)
    (h2 : a 4 + a 6 = 6) :
    a 2 + a 8 = 9 :=
sorry

end geometric_sequence_problem_l144_144709


namespace three_pow_mul_l144_144692

theorem three_pow_mul (a b : ℕ) (h_a : a = 12) (h_b : b = 18) :
  3^a * 3^b = 243^6 := by
  rw [h_a, h_b]
  calc
    3^12 * 3^18
      = 3^(12 + 18) : by rw [pow_add]
  ... = 3^30 : by norm_num
  ... = (3^5)^6 : by rw [pow_mul, ← mul_comm]
  ... = 243^6 : by norm_num

end three_pow_mul_l144_144692


namespace total_min_waiting_time_total_max_waiting_time_total_expected_waiting_time_l144_144300

variables (a b: ℕ) (n m: ℕ)

def C (x y : ℕ) : ℕ := x.choose y

def T_min (a n m : ℕ) : ℕ :=
  a * C n 2 + a * m * n + b * C m 2

def T_max (a n m : ℕ) : ℕ :=
  a * C n 2 + b * m * n + b * C m 2

def E_T (a b n m : ℕ) : ℕ :=
  C (n + m) 2 * ((b * m + a * n) / (m + n))

theorem total_min_waiting_time (a b : ℕ) : T_min 1 5 3 = 40 :=
  by sorry

theorem total_max_waiting_time (a b : ℕ) : T_max 1 5 3 = 100 :=
  by sorry

theorem total_expected_waiting_time (a b : ℕ) : E_T 1 5 5 3 = 70 :=
  by sorry

end total_min_waiting_time_total_max_waiting_time_total_expected_waiting_time_l144_144300


namespace simplify_expression_l144_144738

theorem simplify_expression (x : ℝ) : 7 * x + 8 - 3 * x + 14 = 4 * x + 22 :=
by
  sorry

end simplify_expression_l144_144738


namespace find_solution_l144_144040

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def sum_first_p_squares (p : ℕ) : ℕ := p * (p + 1) * (2 * p + 1) / 6

theorem find_solution : ∃ (n p : ℕ), p.Prime ∧ sum_first_n n = 3 * sum_first_p_squares p ∧ (n, p) = (5, 2) := 
by
  sorry

end find_solution_l144_144040


namespace child_support_calculation_l144_144550

noncomputable def owed_child_support (yearly_salary : ℕ) (raise_pct: ℝ) 
(raise_years_additional_salary: ℕ) (payment_percentage: ℝ) 
(payment_years_salary_before_raise: ℕ) (already_paid : ℝ) : ℝ :=
  let initial_salary := yearly_salary * payment_years_salary_before_raise
  let increase_amount := yearly_salary * raise_pct
  let new_salary := yearly_salary + increase_amount
  let salary_after_raise := new_salary * raise_years_additional_salary
  let total_income := initial_salary + salary_after_raise
  let total_support_due := total_income * payment_percentage
  total_support_due - already_paid

theorem child_support_calculation:
  owed_child_support 30000 0.2 4 0.3 3 1200 = 69000 :=
by
  sorry

end child_support_calculation_l144_144550


namespace probability_of_three_correct_deliveries_l144_144044

-- Define a combination function
def combination (n k : ℕ) : ℕ :=
  n.choose k

-- Define the factorial function
def factorial : ℕ → ℕ
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

-- Define the problem with conditions and derive the required probability
theorem probability_of_three_correct_deliveries :
  (combination 5 3) / (factorial 5) = 1 / 12 := by
  sorry

end probability_of_three_correct_deliveries_l144_144044


namespace ship_distances_l144_144993

-- Define the conditions based on the initial problem statement
variables (f : ℕ → ℝ)
def distances_at_known_times : Prop :=
  f 0 = 49 ∧ f 2 = 25 ∧ f 3 = 121

-- Define the questions to prove the distances at unknown times
def distance_at_time_1 : Prop :=
  f 1 = 1

def distance_at_time_4 : Prop :=
  f 4 = 289

-- The proof problem
theorem ship_distances
  (f : ℕ → ℝ)
  (hf : ∀ t, ∃ a b c, f t = a*t^2 + b*t + c)
  (h_known : distances_at_known_times f) :
  distance_at_time_1 f ∧ distance_at_time_4 f :=
by
  sorry

end ship_distances_l144_144993


namespace volume_correct_l144_144823

open Set Real

-- Define the conditions: the inequality and the constraints on x, y, z
def region (x y z : ℝ) : Prop :=
  abs (z + x + y) + abs (z + x - y) ≤ 10 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0

-- Define the volume calculation
def volume_of_region : ℝ :=
  62.5

-- State the theorem
theorem volume_correct : ∀ (x y z : ℝ), region x y z → volume_of_region = 62.5 :=
by
  intro x y z h
  sorry

end volume_correct_l144_144823


namespace probability_of_at_least_one_boy_and_one_girl_l144_144617

noncomputable def probability_at_least_one_boy_and_one_girl: ℚ :=
  7 / 8

axiom equally_likely_birth : ∀ i : ℕ, (i = 0 ∨ i = 1) → (0.5 : ℝ)

theorem probability_of_at_least_one_boy_and_one_girl (n : ℕ) (condition : n = 4) : 
  probability_at_least_one_boy_and_one_girl = 7 / 8 :=
sorry

end probability_of_at_least_one_boy_and_one_girl_l144_144617


namespace substitution_modulo_l144_144776

-- Definitions based on conditions
def total_players := 15
def starting_lineup := 10
def substitutes := 5
def max_substitutions := 2

-- Define the number of substitutions ways for the cases 0, 1, and 2 substitutions
def a_0 := 1
def a_1 := starting_lineup * substitutes
def a_2 := starting_lineup * substitutes * (starting_lineup - 1) * (substitutes - 1)

-- Summing the total number of substitution scenarios
def total_substitution_scenarios := a_0 + a_1 + a_2

-- Theorem statement to verify the result modulo 500
theorem substitution_modulo : total_substitution_scenarios % 500 = 351 := by
  sorry

end substitution_modulo_l144_144776


namespace total_pennies_donated_l144_144489

def cassandra_pennies : ℕ := 5000
def james_pennies : ℕ := cassandra_pennies - 276
def total_pennies : ℕ := cassandra_pennies + james_pennies

theorem total_pennies_donated : total_pennies = 9724 := by
  sorry

end total_pennies_donated_l144_144489


namespace beads_problem_l144_144327

theorem beads_problem :
  ∃ b : ℕ, (b % 6 = 5) ∧ (b % 8 = 3) ∧ (b % 9 = 7) ∧ (b = 179) :=
by
  sorry

end beads_problem_l144_144327


namespace quiz_show_prob_l144_144604

-- Definitions extracted from the problem conditions
def n : ℕ := 4 -- Number of questions
def p_correct : ℚ := 1 / 4 -- Probability of guessing a question correctly
def p_incorrect : ℚ := 3 / 4 -- Probability of guessing a question incorrectly

-- We need to prove that the probability of answering at least 3 out of 4 questions correctly 
-- by guessing randomly is 13/256.
theorem quiz_show_prob :
  (Nat.choose n 3 * (p_correct ^ 3) * (p_incorrect ^ 1) +
   Nat.choose n 4 * (p_correct ^ 4)) = 13 / 256 :=
by sorry

end quiz_show_prob_l144_144604


namespace unique_position_of_chess_piece_l144_144804

theorem unique_position_of_chess_piece (x y : ℕ) (h : x^2 + x * y - 2 * y^2 = 13) : (x = 5) ∧ (y = 4) :=
sorry

end unique_position_of_chess_piece_l144_144804


namespace polynomial_equivalence_l144_144699

theorem polynomial_equivalence (x y : ℝ) (h : y = x + 1/x) :
  (x^2 * (y^2 + 2*y - 5) = 0) ↔ (x^4 + 2*x^3 - 3*x^2 + 2*x + 1 = 0) :=
by
  sorry

end polynomial_equivalence_l144_144699


namespace bus_time_l144_144093

variable (t1 t2 t3 t4 : ℕ)

theorem bus_time
  (h1 : t1 = 25)
  (h2 : t2 = 40)
  (h3 : t3 = 15)
  (h4 : t4 = 10) :
  t1 + t2 + t3 + t4 = 90 := by
  sorry

end bus_time_l144_144093


namespace part1_part2_part3_l144_144724

noncomputable def setA := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
noncomputable def setB (m : ℝ) := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem part1 (m : ℝ) : setB m ⊆ setA ↔ m ≤ 3 :=
sorry

noncomputable def finite_setA := {-2, -1, 0, 1, 2, 3, 4, 5 : ℤ}

theorem part2 : 2 ^ 8 - 2 = 254 :=
sorry

theorem part3 (m : ℝ) : (∀ x, x ∈ setA → x ∉ setB m) ↔ (m < 2 ∨ m > 4) :=
sorry

end part1_part2_part3_l144_144724


namespace probability_of_fair_die_given_roll_of_three_l144_144970

variables (P_F P_U P_R3_if_F P_R3_if_U : ℝ)
variables (P_F_eq : P_F = 0.25) (P_U_eq : P_U = 0.75)
variables (P_R3_if_F_eq : P_R3_if_F = 1/6) (P_R3_if_U_eq : P_R3_if_U = 1/3)

noncomputable def P_R3 : ℝ :=
  P_R3_if_F * P_F + P_R3_if_U * P_U

theorem probability_of_fair_die_given_roll_of_three :
  P_R3_if_F * P_F / P_R3 = 1 / 7 :=
by
  have P_R3_eq : P_R3 = (1 / 6) * 0.25 + (1 / 3) * 0.75 := sorry
  have P_R3_calculated : P_R3 = 7 / 24 := sorry
  show (1 / 6) * 0.25 / (7 / 24) = 1 / 7 from sorry

end probability_of_fair_die_given_roll_of_three_l144_144970


namespace brandy_used_0_17_pounds_of_chocolate_chips_l144_144478

def weight_of_peanuts : ℝ := 0.17
def weight_of_raisins : ℝ := 0.08
def total_weight_of_trail_mix : ℝ := 0.42

theorem brandy_used_0_17_pounds_of_chocolate_chips :
  total_weight_of_trail_mix - (weight_of_peanuts + weight_of_raisins) = 0.17 :=
by
  sorry

end brandy_used_0_17_pounds_of_chocolate_chips_l144_144478


namespace greatest_three_digit_divisible_by_3_5_6_l144_144452

theorem greatest_three_digit_divisible_by_3_5_6 : 
    ∃ n : ℕ, 
        (100 ≤ n ∧ n ≤ 999) ∧ 
        (∃ k₃ : ℕ, n = 3 * k₃) ∧ 
        (∃ k₅ : ℕ, n = 5 * k₅) ∧ 
        (∃ k₆ : ℕ, n = 6 * k₆) ∧ 
        (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999) ∧ (∃ k₃ : ℕ, m = 3 * k₃) ∧ (∃ k₅ : ℕ, m = 5 * k₅) ∧ (∃ k₆ : ℕ, m = 6 * k₆) → m ≤ 990) := by
  sorry

end greatest_three_digit_divisible_by_3_5_6_l144_144452


namespace projection_matrix_correct_l144_144185

variables {R : Type*} [field R] [decidable_eq R]
variables (x y : R)
def vector_v : matrix (fin 2) (fin 1) R := ![![3], ![4]]
def vector_u : matrix (fin 2) (fin 1) R := ![![x], ![y]]
def projection_matrix : matrix (fin 2) (fin 2) R := ![![9/25, 12/25], ![12/25, 16/25]]

theorem projection_matrix_correct :
  (projection_matrix R) ⬝ (vector_u x y) = (25 : R)⁻¹ • (transpose (vector_v 3 4) ⬝ (vector_u x y)) ⬝ (vector_v 3 4) := 
sorry

end projection_matrix_correct_l144_144185


namespace functional_equation_zero_solution_l144_144038

theorem functional_equation_zero_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (f x + x + y) = f (x + y) + y * f y) :
  ∀ x : ℝ, f x = 0 :=
sorry

end functional_equation_zero_solution_l144_144038


namespace average_infection_rate_l144_144014

theorem average_infection_rate (x : ℕ) : 
  1 + x + x * (1 + x) = 81 :=
sorry

end average_infection_rate_l144_144014


namespace sin_double_angle_pi_six_l144_144350

theorem sin_double_angle_pi_six (α : ℝ)
  (h : 2 * Real.sin α = 1 + 2 * Real.sqrt 3 * Real.cos α) :
  Real.sin (2 * α - Real.pi / 6) = 7 / 8 :=
sorry

end sin_double_angle_pi_six_l144_144350


namespace courses_choice_count_l144_144852

-- Define the conditions and prove the required statement.
theorem courses_choice_count (courses : Finset ℕ) (h : courses.card = 6) :
  (∃ A B : Finset ℕ, A.card = 3 ∧ B.card = 3 ∧ ∃ common_course : ℕ, common_course ∈ A ∧ common_course ∈ B ∧
    (A \ {common_course}).card = 2 ∧ (B \ {common_course}).card = 2 ∧ (A \ {common_course}) ∩ (B \ {common_course}) = ∅ ∧
    (∑ A' B', ((A ∩ B).card = 1)) = 180) := 
sorry

end courses_choice_count_l144_144852


namespace ruth_gave_janet_53_stickers_l144_144404

-- Definitions: Janet initially has 3 stickers, after receiving more from Ruth, she has 56 stickers in total.
def janet_initial : ℕ := 3
def janet_total : ℕ := 56

-- The statement to prove: Ruth gave Janet 53 stickers.
def stickers_from_ruth (initial: ℕ) (total: ℕ) : ℕ :=
  total - initial

theorem ruth_gave_janet_53_stickers : stickers_from_ruth janet_initial janet_total = 53 :=
by sorry

end ruth_gave_janet_53_stickers_l144_144404


namespace arithmetic_sequence_ratio_l144_144527

theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 d : ℝ)
  (h1 : ∀ n, a n = a1 + (n - 1) * d) (h2 : ∀ n, S n = n * (2 * a1 + (n - 1) * d) / 2)
  (h_nonzero: ∀ n, a n ≠ 0):
  (S 5) / (a 3) = 5 :=
by
  sorry

end arithmetic_sequence_ratio_l144_144527


namespace arithmetic_sequence_a3_l144_144529

theorem arithmetic_sequence_a3 (a : ℕ → ℕ) (h1 : a 6 = 6) (h2 : a 9 = 9) : a 3 = 3 :=
by
  -- proof goes here
  sorry

end arithmetic_sequence_a3_l144_144529


namespace additional_people_needed_l144_144176

theorem additional_people_needed
  (initial_people : ℕ) (initial_time : ℕ) (new_time : ℕ)
  (h_initial : initial_people * initial_time = 24)
  (h_time : new_time = 2)
  (h_initial_people : initial_people = 8)
  (h_initial_time : initial_time = 3) :
  (24 / new_time) - initial_people = 4 :=
by
  sorry

end additional_people_needed_l144_144176


namespace value_2_stddevs_less_than_mean_l144_144120

-- Definitions based on the conditions
def mean : ℝ := 10.5
def stddev : ℝ := 1
def value := mean - 2 * stddev

-- Theorem we aim to prove
theorem value_2_stddevs_less_than_mean : value = 8.5 := by
  -- proof will go here
  sorry

end value_2_stddevs_less_than_mean_l144_144120


namespace matrix_vector_computation_l144_144411

-- Setup vectors and their corresponding matrix multiplication results
variables {R : Type*} [Field R]
variables {M : Matrix (Fin 2) (Fin 2) R} {u z : Fin 2 → R}

-- Conditions given in (a)
def condition1 : M.mulVec u = ![3, -4] :=
  sorry

def condition2 : M.mulVec z = ![-1, 6] :=
  sorry

-- Statement equivalent to the proof problem given in (c)
theorem matrix_vector_computation :
  M.mulVec (3 • u - 2 • z) = ![11, -24] :=
by
  -- Use the conditions to prove the theorem
  sorry

end matrix_vector_computation_l144_144411


namespace grades_calculation_l144_144446

-- Defining the conditions
def total_students : ℕ := 22800
def students_per_grade : ℕ := 75

-- Stating the theorem to be proved
theorem grades_calculation : total_students / students_per_grade = 304 := sorry

end grades_calculation_l144_144446


namespace max_a_no_lattice_point_l144_144783

theorem max_a_no_lattice_point (a : ℚ) : a = 35 / 51 ↔ 
  (∀ (m : ℚ), (2 / 3 < m ∧ m < a) → 
    (∀ (x : ℤ), (0 < x ∧ x ≤ 50) → 
      ¬ ∃ (y : ℤ), y = m * x + 5)) :=
sorry

end max_a_no_lattice_point_l144_144783


namespace binomial_coefficient_max_term_l144_144811

theorem binomial_coefficient_max_term (n k : ℕ) (x y : ℕ) :
  let term (n k : ℕ) (x y : ℕ) := (Nat.choose n k) * x^(n - k) * y^k in
  (term 213 147 1 (√5)) = (Nat.choose 213 147) * (√5)^147 :=
sorry

end binomial_coefficient_max_term_l144_144811


namespace max_ratio_xy_l144_144087

def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem max_ratio_xy (x y : ℕ) (hx : two_digit x) (hy : two_digit y) (hmean : (x + y) / 2 = 60) : x / y ≤ 33 / 7 :=
by
  sorry

end max_ratio_xy_l144_144087


namespace sum_of_remainders_11111k_43210_eq_141_l144_144725

theorem sum_of_remainders_11111k_43210_eq_141 :
  (List.sum (List.map (fun k => (11111 * k + 43210) % 31) [0, 1, 2, 3, 4, 5])) = 141 :=
by
  -- Proof is omitted: sorry
  sorry

end sum_of_remainders_11111k_43210_eq_141_l144_144725


namespace child_support_owed_l144_144547

noncomputable def income_first_3_years : ℕ := 3 * 30000
noncomputable def raise_per_year : ℕ := 30000 * 20 / 100
noncomputable def new_salary : ℕ := 30000 + raise_per_year
noncomputable def income_next_4_years : ℕ := 4 * new_salary
noncomputable def total_income : ℕ := income_first_3_years + income_next_4_years
noncomputable def total_child_support : ℕ := total_income * 30 / 100
noncomputable def amount_paid : ℕ := 1200
noncomputable def amount_owed : ℕ := total_child_support - amount_paid

theorem child_support_owed : amount_owed = 69000 := by
  sorry

end child_support_owed_l144_144547


namespace correct_statements_l144_144304

def problem_statements :=
  [ "The negation of the statement 'There exists an x ∈ ℝ such that x^2 - 3x + 3 = 0' is true.",
    "The statement '-1/2 < x < 0' is a necessary but not sufficient condition for '2x^2 - 5x - 3 < 0'.",
    "The negation of the statement 'If xy = 0, then at least one of x or y is equal to 0' is true.",
    "The curves x^2/25 + y^2/9 = 1 and x^2/(25 − k) + y^2/(9 − k) = 1 (9 < k < 25) share the same foci.",
    "There exists a unique line that passes through the point (1,3) and is tangent to the parabola y^2 = 4x."
  ]

theorem correct_statements :
  (∀ x : ℝ, ¬(x^2 - 3 * x + 3 = 0)) ∧ 
  ¬ (¬-1/2 < x ∧ x < 0 → 2 * x^2 - 5*x - 3 < 0) ∧ 
  (∀ x y : ℝ, xy ≠ 0 → x ≠ 0 ∧ y ≠ 0) ∧ 
  (∀ k : ℝ, 9 < k ∧ k < 25 → ∀ x y : ℝ, (x^2 / (25 - k) + y^2 / (9 - k) = 1) → (x^2 / 25 + y^2 / 9 = 1) → (x ≠ 0 ∨ y ≠ 0)) ∧ 
  ¬ (∃ l : ℝ, ∀ pt : ℝ × ℝ, pt = (1, 3) → ∀ y : ℝ, y^2 = 4 * pt.1 → y = 2 * pt.2)
:= 
  sorry

end correct_statements_l144_144304


namespace tin_can_allocation_l144_144449

-- Define the total number of sheets of tinplate available
def total_sheets := 108

-- Define the number of sheets used for can bodies
variable (x : ℕ)

-- Define the number of can bodies a single sheet makes
def can_bodies_per_sheet := 15

-- Define the number of can bottoms a single sheet makes
def can_bottoms_per_sheet := 42

-- Define the equation to be proven
theorem tin_can_allocation :
  2 * can_bodies_per_sheet * x = can_bottoms_per_sheet * (total_sheets - x) :=
  sorry

end tin_can_allocation_l144_144449


namespace ratio_of_amount_lost_l144_144532

noncomputable def amount_lost (initial_amount spent_motorcycle spent_concert after_loss : ℕ) : ℕ :=
  let remaining_after_motorcycle := initial_amount - spent_motorcycle
  let remaining_after_concert := remaining_after_motorcycle / 2
  remaining_after_concert - after_loss

noncomputable def ratio (a b : ℕ) : ℕ × ℕ :=
  let g := Nat.gcd a b
  (a / g, b / g)

theorem ratio_of_amount_lost 
  (initial_amount spent_motorcycle spent_concert after_loss : ℕ)
  (h1 : initial_amount = 5000)
  (h2 : spent_motorcycle = 2800)
  (h3 : spent_concert = (initial_amount - spent_motorcycle) / 2)
  (h4 : after_loss = 825) :
  ratio (amount_lost initial_amount spent_motorcycle spent_concert after_loss)
        spent_concert = (1, 4) := by
  sorry

end ratio_of_amount_lost_l144_144532


namespace intersection_of_sets_l144_144391

def set_a : Set ℝ := { x | -x^2 + 2 * x ≥ 0 }
def set_b : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def set_intersection : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem intersection_of_sets : (set_a ∩ set_b) = set_intersection := by 
  sorry

end intersection_of_sets_l144_144391


namespace probability_blue_then_green_l144_144118

-- Definitions based on the conditions
def faces := 12
def red_faces := 5
def blue_faces := 4
def yellow_faces := 2
def green_faces := 1

-- Probabilities based on the problem setup
def probability_blue := blue_faces / faces
def probability_green := green_faces / faces

-- Proof statement
theorem probability_blue_then_green :
  (probability_blue * probability_green) = (1 / 36) :=
by
  sorry

end probability_blue_then_green_l144_144118


namespace find_denominator_l144_144313

theorem find_denominator (x : ℕ) (dec_form_of_frac_4128 : ℝ) (h1: 4128 / x = dec_form_of_frac_4128) 
    : x = 4387 :=
by
  have h: dec_form_of_frac_4128 = 0.9411764705882353 := sorry
  sorry

end find_denominator_l144_144313


namespace min_value_of_2a_plus_3b_l144_144664

theorem min_value_of_2a_plus_3b
  (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_perpendicular : (x - (2 * b - 3) * y + 6 = 0) ∧ (2 * b * x + a * y - 5 = 0)) :
  2 * a + 3 * b = 25 / 2 :=
sorry

end min_value_of_2a_plus_3b_l144_144664


namespace sales_tax_is_5_percent_l144_144157

theorem sales_tax_is_5_percent :
  let cost_tshirt := 8
  let cost_sweater := 18
  let cost_jacket := 80
  let discount := 0.10
  let num_tshirts := 6
  let num_sweaters := 4
  let num_jackets := 5
  let total_cost_with_tax := 504
  let total_cost_before_discount := (num_jackets * cost_jacket)
  let discount_amount := discount * total_cost_before_discount
  let discounted_cost_jackets := total_cost_before_discount - discount_amount
  let total_cost_before_tax := (num_tshirts * cost_tshirt) + (num_sweaters * cost_sweater) + discounted_cost_jackets
  let sales_tax := (total_cost_with_tax - total_cost_before_tax)
  let sales_tax_percentage := (sales_tax / total_cost_before_tax) * 100
  sales_tax_percentage = 5 := by
  sorry

end sales_tax_is_5_percent_l144_144157


namespace sin_600_eq_l144_144641

theorem sin_600_eq : Real.sin (600 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_600_eq_l144_144641


namespace quadratic_polynomial_discriminant_l144_144379

def P (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₀ : a ≠ 0)
  (h₁ : ∃ x : ℝ, P a b c x = x - 2 ∧ (discriminant a (b - 1) (c + 2) = 0))
  (h₂ : ∃ x : ℝ, P a b c x = 1 - x / 2 ∧ (discriminant a (b + 1 / 2) (c - 1) = 0)) :
  discriminant a b c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l144_144379


namespace polynomial_discriminant_l144_144376

theorem polynomial_discriminant (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
by
  sorry

end polynomial_discriminant_l144_144376


namespace usual_time_to_school_l144_144974

variables (R T : ℝ)

theorem usual_time_to_school :
  (3 / 2) * R * (T - 4) = R * T -> T = 12 :=
by sorry

end usual_time_to_school_l144_144974


namespace point_not_on_line_pq_neg_l144_144523

theorem point_not_on_line_pq_neg (p q : ℝ) (h : p * q < 0) : ¬ (21 * p + q = -101) := 
by sorry

end point_not_on_line_pq_neg_l144_144523


namespace triangle_is_isosceles_l144_144052

variable (a b c : ℝ)
variable (h : a^2 - b * c = a * (b - c))

theorem triangle_is_isosceles (a b c : ℝ) (h : a^2 - b * c = a * (b - c)) : a = b ∨ b = c ∨ c = a := by
  sorry

end triangle_is_isosceles_l144_144052


namespace investment_initial_amount_l144_144793

theorem investment_initial_amount (P : ℝ) (h1 : ∀ (x : ℝ), 0 < x → (1 + 0.10) * x = 1.10 * x) (h2 : 1.21 * P = 363) : P = 300 :=
sorry

end investment_initial_amount_l144_144793


namespace find_a6_l144_144237

-- Define the geometric sequence and the given terms
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

variables {a : ℕ → ℝ} (r : ℝ)

-- Given conditions
axiom a_2 : a 2 = 2
axiom a_10 : a 10 = 8
axiom geo_seq : geometric_sequence a

-- Statement to prove
theorem find_a6 : a 6 = 4 :=
sorry

end find_a6_l144_144237


namespace minimum_breaks_l144_144780

-- Definitions based on conditions given in the problem statement
def longitudinal_grooves : ℕ := 2
def transverse_grooves : ℕ := 3

-- The problem statement to be proved
theorem minimum_breaks (l t : ℕ) (hl : l = longitudinal_grooves) (ht : t = transverse_grooves) :
  l + t = 4 :=
by
  sorry

end minimum_breaks_l144_144780


namespace prove_sum_l144_144232

variables {a : ℕ → ℝ} {r : ℝ}
variable (pos : ∀ n, 0 < a n)

-- Defining the conditions
def geom_seq (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = a n * r

def condition1 (a : ℕ → ℝ) (r : ℝ) : Prop := a 0 + a 1 + a 2 = 2
def condition2 (a : ℕ → ℝ) (r : ℝ) : Prop := a 2 + a 3 + a 4 = 8

-- The main theorem statement
theorem prove_sum (a : ℕ → ℝ) (r : ℝ) (pos : ∀ n, 0 < a n)
  (geom : geom_seq a r) (h1 : condition1 a r) (h2 : condition2 a r) :
  a 3 + a 4 + a 5 = 16 :=
sorry

end prove_sum_l144_144232


namespace exponent_product_to_sixth_power_l144_144689

theorem exponent_product_to_sixth_power :
  ∃ n : ℤ, 3^(12) * 3^(18) = n^6 ∧ n = 243 :=
by
  use 243
  sorry

end exponent_product_to_sixth_power_l144_144689


namespace max_value_of_sum_l144_144517

theorem max_value_of_sum (a c d : ℤ) (b : ℕ) (h1 : a + b = c) (h2 : b + c = d) (h3 : c + d = a) :
  a + b + c + d ≤ -5 := 
sorry

end max_value_of_sum_l144_144517


namespace arithmetic_sqrt_of_9_l144_144899

theorem arithmetic_sqrt_of_9 : (∃ (sqrt : ℝ), sqrt = 3 ∧ ∀ x, x*x = 9 → x = sqrt) :=
by
  existsi (3 : ℝ)
  split
  exact rfl
  intros x hx
  exact sqrt_unique hx 3

end arithmetic_sqrt_of_9_l144_144899


namespace power_zero_equals_one_specific_case_l144_144586

theorem power_zero_equals_one 
    (a b : ℤ) 
    (h : a ≠ 0)
    (h2 : b ≠ 0) : 
    (a / b : ℚ) ^ 0 = 1 := 
by {
  sorry
}

-- Specific case
theorem specific_case : 
  ( ( (-123456789 : ℤ) / (9876543210 : ℤ) : ℚ ) ^ 0 = 1 ) := 
by {
  apply power_zero_equals_one;
  norm_num;
  sorry
}

end power_zero_equals_one_specific_case_l144_144586


namespace imaginary_part_of_z_l144_144214

-- Define the imaginary unit i where i^2 = -1
def imaginary_unit : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := (2 + imaginary_unit) * (1 - imaginary_unit)

-- State the theorem to prove the imaginary part of z
theorem imaginary_part_of_z : Complex.im z = -1 := by
  sorry

end imaginary_part_of_z_l144_144214


namespace simplify_expr_l144_144047

variable {x y : ℝ}

theorem simplify_expr (hx : x ≠ 0) (hy : y ≠ 0) :
  ((x^3 + 1) / x) * ((y^3 + 1) / y) - ((x^3 - 1) / y) * ((y^3 - 1) / x) = 2 * x^2 + 2 * y^2 :=
by sorry

end simplify_expr_l144_144047


namespace time_to_return_l144_144323

-- Given conditions
def distance : ℝ := 1000
def return_speed : ℝ := 142.85714285714286

-- Goal to prove
theorem time_to_return : distance / return_speed = 7 := 
by
  sorry

end time_to_return_l144_144323


namespace dilation_complex_l144_144749

theorem dilation_complex :
  let c := (1 : ℂ) - (2 : ℂ) * I
  let k := 3
  let z := -1 + I
  (k * (z - c) + c = -5 + 7 * I) :=
by
  sorry

end dilation_complex_l144_144749


namespace least_number_of_cans_l144_144293

theorem least_number_of_cans (maaza : ℕ) (pepsi : ℕ) (sprite : ℕ) (gcd_val : ℕ) (total_cans : ℕ)
  (h1 : maaza = 50) (h2 : pepsi = 144) (h3 : sprite = 368) (h_gcd : gcd maaza (gcd pepsi sprite) = gcd_val)
  (h_total_cans : total_cans = maaza / gcd_val + pepsi / gcd_val + sprite / gcd_val) :
  total_cans = 281 :=
sorry

end least_number_of_cans_l144_144293


namespace circles_ACD_and_BCD_orthogonal_l144_144089

-- Define mathematical objects and conditions
variables (A B C D : Point) -- Points in general position on the plane
variables (circle : Point → Point → Point → Circle)

-- Circles intersect orthogonally property
def orthogonal_intersection (c1 c2 : Circle) : Prop :=
  -- Definition of orthogonal intersection of circles goes here (omitted for brevity)
  sorry

-- Given conditions
def circles_ABC_and_ABD_orthogonal : Prop :=
  orthogonal_intersection (circle A B C) (circle A B D)

-- Theorem statement
theorem circles_ACD_and_BCD_orthogonal (h : circles_ABC_and_ABD_orthogonal A B C D circle) :
  orthogonal_intersection (circle A C D) (circle B C D) :=
sorry

end circles_ACD_and_BCD_orthogonal_l144_144089


namespace profit_benny_wants_to_make_l144_144477

noncomputable def pumpkin_pies : ℕ := 10
noncomputable def cherry_pies : ℕ := 12
noncomputable def cost_pumpkin_pie : ℝ := 3
noncomputable def cost_cherry_pie : ℝ := 5
noncomputable def price_per_pie : ℝ := 5

theorem profit_benny_wants_to_make : 5 * (pumpkin_pies + cherry_pies) - (pumpkin_pies * cost_pumpkin_pie + cherry_pies * cost_cherry_pie) = 20 :=
by
  sorry

end profit_benny_wants_to_make_l144_144477


namespace range_of_a_l144_144525

theorem range_of_a
  (x0 : ℝ) (a : ℝ)
  (hx0 : x0 > 1)
  (hineq : (x0 + 1) * Real.log x0 < a * (x0 - 1)) :
  a > 2 :=
sorry

end range_of_a_l144_144525


namespace find_term_number_l144_144148

-- Define the arithmetic sequence
def arithmetic_seq (a d : Int) (n : Int) := a + (n - 1) * d

-- Define the condition: first term and common difference
def a1 := 4
def d := 3

-- Prove that the 672nd term is 2017
theorem find_term_number (n : Int) (h : arithmetic_seq a1 d n = 2017) : n = 672 := by
  sorry

end find_term_number_l144_144148


namespace sum_remainder_l144_144666

theorem sum_remainder (a b c : ℕ) 
  (h1 : a % 15 = 11) 
  (h2 : b % 15 = 13) 
  (h3 : c % 15 = 9) :
  (a + b + c) % 15 = 3 := 
by
  sorry

end sum_remainder_l144_144666


namespace gcf_lcm_60_72_l144_144131

def gcf_lcm_problem (a b : ℕ) : Prop :=
  gcd a b = 12 ∧ lcm a b = 360

theorem gcf_lcm_60_72 : gcf_lcm_problem 60 72 :=
by {
  sorry
}

end gcf_lcm_60_72_l144_144131


namespace logarithm_identity_l144_144338

noncomputable section

open Real

theorem logarithm_identity : 
  log 10 = (log (sqrt 5) / log 10 + (1 / 2) * log 20) :=
sorry

end logarithm_identity_l144_144338


namespace question1_question2_question3_l144_144806

variables {a x1 x2 : ℝ}

-- Definition of the quadratic equation
def quadratic_eq (a x : ℝ) : ℝ := a * x^2 + x + 1

-- Conditions
axiom a_positive : a > 0
axiom roots_exist : quadratic_eq a x1 = 0 ∧ quadratic_eq a x2 = 0
axiom roots_real : x1 + x2 = -1 / a ∧ x1 * x2 = 1 / a

-- Question 1
theorem question1 : (1 + x1) * (1 + x2) = 1 :=
sorry

-- Question 2
theorem question2 : x1 < -1 ∧ x2 < -1 :=
sorry

-- Additional condition for question 3
axiom ratio_in_range : x1 / x2 ∈ Set.Icc (1 / 10 : ℝ) 10

-- Question 3
theorem question3 : a <= 1 / 4 :=
sorry

end question1_question2_question3_l144_144806


namespace random_event_is_option_D_l144_144291

-- Definitions based on conditions
def rains_without_clouds : Prop := false
def like_charges_repel : Prop := true
def seeds_germinate_without_moisture : Prop := false
def draw_card_get_1 : Prop := true

-- Proof statement
theorem random_event_is_option_D : 
  (¬ rains_without_clouds ∧ like_charges_repel ∧ ¬ seeds_germinate_without_moisture ∧ draw_card_get_1) →
  (draw_card_get_1 = true) :=
by sorry

end random_event_is_option_D_l144_144291


namespace five_card_draw_probability_l144_144701

noncomputable def probability_at_least_one_card_from_each_suit : ℚ := 3 / 32

theorem five_card_draw_probability :
  let deck_size := 52
  let suits := 4
  let cards_drawn := 5
  (1 : ℚ) * (3 / 4) * (1 / 2) * (1 / 4) = probability_at_least_one_card_from_each_suit := by
  sorry

end five_card_draw_probability_l144_144701


namespace log_value_l144_144522

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_value (x : ℝ) (h : log_base 3 (5 * x) = 3) : log_base x 125 = 3 / 2 :=
  by
  sorry

end log_value_l144_144522


namespace car_maintenance_expense_l144_144247

-- Define constants and conditions
def miles_per_year : ℕ := 12000
def oil_change_interval : ℕ := 3000
def oil_change_price (quarter : ℕ) : ℕ := 
  if quarter = 1 then 55 
  else if quarter = 2 then 45 
  else if quarter = 3 then 50 
  else 40
def free_oil_changes_per_year : ℕ := 1

def tire_rotation_interval : ℕ := 6000
def tire_rotation_cost : ℕ := 40
def tire_rotation_discount : ℕ := 10 -- In percent

def brake_pad_interval : ℕ := 24000
def brake_pad_cost : ℕ := 200
def brake_pad_discount : ℕ := 20 -- In percent
def brake_pad_membership_cost : ℕ := 60
def membership_duration : ℕ := 2 -- In years

def total_annual_expense : ℕ :=
  let oil_changes := (miles_per_year / oil_change_interval) - free_oil_changes_per_year
  let oil_cost := (oil_change_price 2 + oil_change_price 3 + oil_change_price 4) -- Free oil change in Q1
  let tire_rotations := miles_per_year / tire_rotation_interval
  let tire_cost := (tire_rotation_cost * (100 - tire_rotation_discount) / 100) * tire_rotations
  let brake_pad_cost_per_year := (brake_pad_cost * (100 - brake_pad_discount) / 100) / membership_duration
  let membership_cost_per_year := brake_pad_membership_cost / membership_duration
  oil_cost + tire_cost + (brake_pad_cost_per_year + membership_cost_per_year)

-- Assert the proof problem
theorem car_maintenance_expense : total_annual_expense = 317 := by
  sorry

end car_maintenance_expense_l144_144247


namespace month_days_l144_144276

theorem month_days (letters_per_day packages_per_day total_mail six_months : ℕ) (h1 : letters_per_day = 60) (h2 : packages_per_day = 20) (h3 : total_mail = 14400) (h4 : six_months = 6) : 
  total_mail / (letters_per_day + packages_per_day) / six_months = 30 :=
by sorry

end month_days_l144_144276


namespace inequality_positive_real_l144_144556

theorem inequality_positive_real (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 1) :
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 :=
sorry

end inequality_positive_real_l144_144556


namespace largest_smallest_difference_l144_144820

theorem largest_smallest_difference (a b c d : ℚ) (h₁ : a = 2.5) (h₂ : b = 22/13) (h₃ : c = 0.7) (h₄ : d = 32/33) :
  max (max a b) (max c d) - min (min a b) (min c d) = 1.8 := by
  sorry

end largest_smallest_difference_l144_144820


namespace possible_values_of_reciprocal_sum_l144_144722

theorem possible_values_of_reciprocal_sum (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 2) (h4 : x * y = 1) : 
  1/x + 1/y = 2 := 
sorry

end possible_values_of_reciprocal_sum_l144_144722


namespace find_tan_angle_F2_F1_B_l144_144536

-- Definitions for the points and chord lengths
def F1 : Type := ℝ × ℝ
def F2 : Type := ℝ × ℝ
def A : Type := ℝ × ℝ
def B : Type := ℝ × ℝ

-- Given distances
def F1A : ℝ := 3
def AB : ℝ := 4
def BF1 : ℝ := 5

-- The angle we want to find the tangent of
def angle_F2_F1_B (F1 F2 A B : Type) : ℝ := sorry -- Placeholder for angle calculation

-- The main theorem to prove
theorem find_tan_angle_F2_F1_B (F1 F2 A B : Type) (F1A_dist : F1A = 3) (AB_dist : AB = 4) (BF1_dist : BF1 = 5) :
  angle_F2_F1_B F1 F2 A B = 1 / 7 :=
sorry

end find_tan_angle_F2_F1_B_l144_144536


namespace tan_a2_a12_l144_144387

noncomputable def arithmetic_term (a d : ℝ) (n : ℕ) : ℝ := a + d * (n - 1)

theorem tan_a2_a12 (a d : ℝ) (h : a + (a + 6 * d) + (a + 12 * d) = 4 * Real.pi) :
  Real.tan (arithmetic_term a d 2 + arithmetic_term a d 12) = - Real.sqrt 3 :=
by
  sorry

end tan_a2_a12_l144_144387


namespace fraction_eq_l144_144306

theorem fraction_eq (x : ℝ) (h1 : x * 180 = 24) (h2 : x < 20 / 100) : x = 2 / 15 :=
sorry

end fraction_eq_l144_144306


namespace prob_four_children_at_least_one_boy_one_girl_l144_144613

-- Define the probability of a single birth being a boy or a girl
def prob_boy_or_girl : ℚ := 1/2

-- Calculate the probability of all children being boys or all girls
def prob_all_boys : ℚ := (prob_boy_or_girl)^4
def prob_all_girls : ℚ := (prob_boy_or_girl)^4

-- Calculate the probability of having neither all boys nor all girls
def prob_at_least_one_boy_one_girl : ℚ := 1 - (prob_all_boys + prob_all_girls)

-- The theorem to prove
theorem prob_four_children_at_least_one_boy_one_girl : 
  prob_at_least_one_boy_one_girl = 7/8 := 
by 
  sorry

end prob_four_children_at_least_one_boy_one_girl_l144_144613


namespace bakery_rolls_combinations_l144_144305

theorem bakery_rolls_combinations : 
  ∃ n : ℕ, n = 10 ∧ 
  (∃ k : ℕ, ∃ t : ℕ, (k = 4) ∧ (t = 2) ∧ (n = k * t + (n - k * t)) 
    ∧ (nat.choose (n - k * t + k - 1) (k - 1)) = 10) := by
  sorry

end bakery_rolls_combinations_l144_144305


namespace cube_vertex_numbering_impossible_l144_144861

-- Definition of the cube problem
def vertex_numbering_possible : Prop :=
  ∃ (v : Fin 8 → ℕ), (∀ i, 1 ≤ v i ∧ v i ≤ 8) ∧
    (∀ (e1 e2 : (Fin 8 × Fin 8)), e1 ≠ e2 → (v e1.1 + v e1.2 ≠ v e2.1 + v e2.2))

theorem cube_vertex_numbering_impossible : ¬ vertex_numbering_possible :=
sorry

end cube_vertex_numbering_impossible_l144_144861


namespace graph_remains_connected_after_deletions_l144_144310

theorem graph_remains_connected_after_deletions
  (G : SimpleGraph (Fin 1998))
  (h_connected : G.IsConnected)
  (h_degree : ∀ v : Fin 1998, G.degree v = 3)
  (vertex_set : Finset (Fin 1998))
  (h_card_vertex_set : vertex_set.card = 200)
  (h_no_adjacent : ∀ v w ∈ vertex_set, ¬G.Adj v w) :
  G.deleteVertices vertex_set).IsConnected :=
begin
  sorry 
end

end graph_remains_connected_after_deletions_l144_144310


namespace find_f_l144_144252

theorem find_f (f : ℝ → ℝ) (h₀ : f 0 = 1) (h₁ : ∀ x y, f (x * y) = f ((x^2 + y^2) / 2) + (x - y)^2) : 
  ∀ x, f x = 1 - 2 * x :=
by
  sorry  -- Proof not required

end find_f_l144_144252


namespace three_exp_product_sixth_power_l144_144674

theorem three_exp_product_sixth_power :
  ∃ n : ℤ, 3^12 * 3^18 = n^6 ∧ n = 243 :=
by
  existsi 243
  split
  · sorry
  · refl

end three_exp_product_sixth_power_l144_144674


namespace lines_intersect_at_common_point_iff_l144_144256

theorem lines_intersect_at_common_point_iff (a b : ℝ) :
  (∃ x y : ℝ, a * x + 2 * b * y + 3 * (a + b + 1) = 0 ∧ 
               b * x + 2 * (a + b + 1) * y + 3 * a = 0 ∧ 
               (a + b + 1) * x + 2 * a * y + 3 * b = 0) ↔ 
  a + b = -1/2 :=
by
  sorry

end lines_intersect_at_common_point_iff_l144_144256


namespace calculate_expression_l144_144630

theorem calculate_expression : (1000^2) / (252^2 - 248^2) = 500 := sorry

end calculate_expression_l144_144630


namespace probability_at_least_one_boy_and_girl_l144_144615

section
variable (n : ℕ) (p : ℚ)
-- Condition: Birth of a boy is equally likely as a girl (p = 1/2)
def equally_likely : ℚ := 1 / 2

-- Function to calculate the probability of all boys or all girls
def same_gender_probability (n : ℕ) (p : ℚ) : ℚ :=
p ^ n

/-- Theorem: The probability that among four children, there is at least one boy and one girl is 7/8. -/
theorem probability_at_least_one_boy_and_girl :
  same_gender_probability 4 equally_likely + same_gender_probability 4 equally_likely = (1 / 8) →
  1 - 1 / 8 = 7 / 8 :=
by
  sorry

end

end probability_at_least_one_boy_and_girl_l144_144615


namespace min_value_a_2b_3c_l144_144208

theorem min_value_a_2b_3c (a b c : ℝ)
  (h : ∀ x y : ℝ, x + 2 * y - 3 ≤ a * x + b * y + c ∧ a * x + b * y + c ≤ x + 2 * y + 3) :
  a + 2 * b - 3 * c ≥ -2 :=
sorry

end min_value_a_2b_3c_l144_144208


namespace sum_of_common_ratios_l144_144414

theorem sum_of_common_ratios (k p r : ℝ) (h : k ≠ 0) (h1 : k * p ≠ k * r)
  (h2 : k * p ^ 2 - k * r ^ 2 = 3 * (k * p - k * r)) : p + r = 3 :=
by
  sorry

end sum_of_common_ratios_l144_144414


namespace find_n_l144_144944

theorem find_n (e n : ℕ) (h1 : Nat.lcm e n = 690)
  (h2 : 100 ≤ n ∧ n < 1000)
  (h3 : ¬ (3 ∣ n))
  (h4 : ¬ (2 ∣ e)) :
  n = 230 :=
by
  sorry

end find_n_l144_144944


namespace total_pennies_donated_l144_144488

def cassandra_pennies : ℕ := 5000
def james_pennies : ℕ := cassandra_pennies - 276
def total_pennies : ℕ := cassandra_pennies + james_pennies

theorem total_pennies_donated : total_pennies = 9724 := by
  sorry

end total_pennies_donated_l144_144488


namespace loan_balance_formula_l144_144541

variable (c V : ℝ) (t n : ℝ)

theorem loan_balance_formula :
  V = c / (1 + t)^(3 * n) →
  n = (Real.log (c / V)) / (3 * Real.log (1 + t)) :=
by sorry

end loan_balance_formula_l144_144541


namespace unique_solution_l144_144342

theorem unique_solution (m n : ℕ) (h1 : n^4 ∣ 2 * m^5 - 1) (h2 : m^4 ∣ 2 * n^5 + 1) : m = 1 ∧ n = 1 :=
by
  sorry

end unique_solution_l144_144342


namespace total_cases_of_candy_correct_l144_144745

-- Define the number of cases of chocolate bars and lollipops
def cases_of_chocolate_bars : ℕ := 25
def cases_of_lollipops : ℕ := 55

-- Define the total number of cases of candy
def total_cases_of_candy : ℕ := cases_of_chocolate_bars + cases_of_lollipops

-- Prove that the total number of cases of candy is 80
theorem total_cases_of_candy_correct : total_cases_of_candy = 80 := by
  sorry

end total_cases_of_candy_correct_l144_144745


namespace tangent_line_at_x0_minimum_value_on_interval_minimum_value_on_interval_high_minimum_value_on_interval_mid_l144_144665

noncomputable def f (x a : ℝ) : ℝ := (x - a) * Real.exp x

theorem tangent_line_at_x0 (a : ℝ) (h : a = 2) : 
    (∃ m b : ℝ, (∀ x : ℝ, f x a = m * x + b) ∧ m = -1 ∧ b = -2) :=
by 
    sorry

theorem minimum_value_on_interval (a : ℝ) :
    (1 ≤ a) → (a ≤ 2) → f 1 a = (1 - a) * Real.exp 1 :=
by 
    sorry

theorem minimum_value_on_interval_high (a : ℝ) :
    (a ≥ 3) → f 2 a = (2 - a) * Real.exp 2 :=
by 
    sorry

theorem minimum_value_on_interval_mid (a : ℝ) :
    (2 < a) → (a < 3) → f (a - 1) a = -(Real.exp (a - 1)) :=
by 
    sorry

end tangent_line_at_x0_minimum_value_on_interval_minimum_value_on_interval_high_minimum_value_on_interval_mid_l144_144665


namespace Jake_initial_balloons_l144_144791

theorem Jake_initial_balloons (J : ℕ) 
  (h1 : 6 = (J + 3) + 1) : 
  J = 2 :=
by
  sorry

end Jake_initial_balloons_l144_144791


namespace arithmetic_sqrt_of_nine_l144_144919

-- Define the arithmetic square root function which only considers non-negative values
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ :=
  if hx : x ≥ 0 then Real.sqrt x else 0

-- The theorem to prove: The arithmetic square root of 9 is 3.
theorem arithmetic_sqrt_of_nine : arithmetic_sqrt 9 = 3 :=
by
  sorry

end arithmetic_sqrt_of_nine_l144_144919


namespace arithmetic_sqrt_of_13_l144_144893

theorem arithmetic_sqrt_of_13 : Real.sqrt 13 = Real.sqrt 13 := by
  sorry

end arithmetic_sqrt_of_13_l144_144893


namespace coordinates_of_points_l144_144460

theorem coordinates_of_points
  (R : ℝ) (a b : ℝ)
  (hR : R = 10)
  (h_area : 1/2 * a * b = 600)
  (h_a_gt_b : a > b) :
  (a, 0) = (40, 0) ∧ (0, b) = (0, 30) ∧ (16, 18) = (16, 18) :=
  sorry

end coordinates_of_points_l144_144460


namespace small_beaker_salt_fraction_l144_144322

theorem small_beaker_salt_fraction
  (S L : ℝ) 
  (h1 : L = 5 * S)
  (h2 : L * (1 / 5) = S)
  (h3 : L * 0.3 = S * 1.5)
  : (S * 0.5) / S = 0.5 :=
by 
  sorry

end small_beaker_salt_fraction_l144_144322


namespace product_of_decimal_numbers_l144_144296

theorem product_of_decimal_numbers 
  (h : 213 * 16 = 3408) : 
  1.6 * 21.3 = 34.08 :=
by
  sorry

end product_of_decimal_numbers_l144_144296


namespace common_ratio_of_geometric_sequence_l144_144663

theorem common_ratio_of_geometric_sequence 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_seq : ∀ n, a (n + 1) = a n * q) 
  (h_inc : ∀ n, a n < a (n + 1)) 
  (h_a2 : a 2 = 2) 
  (h_diff : a 4 - a 3 = 4) : 
  q = 2 := 
sorry

end common_ratio_of_geometric_sequence_l144_144663


namespace annie_overtakes_bonnie_l144_144996

-- Define the conditions
def track_circumference : ℝ := 300
def bonnie_speed (v : ℝ) : ℝ := v
def annie_speed (v : ℝ) : ℝ := 1.5 * v

-- Define the statement for proving the number of laps completed by Annie when she first overtakes Bonnie
theorem annie_overtakes_bonnie (v t : ℝ) : 
  bonnie_speed v * t = track_circumference * 2 → 
  annie_speed v * t = track_circumference * 3 :=
by
  sorry

end annie_overtakes_bonnie_l144_144996


namespace flyers_left_l144_144242

theorem flyers_left (total_flyers : ℕ) (jack_flyers : ℕ) (rose_flyers : ℕ) :
  total_flyers = 1236 → jack_flyers = 120 → rose_flyers = 320 → total_flyers - (jack_flyers + rose_flyers) = 796 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact eq.refl _

end flyers_left_l144_144242


namespace temp_difference_l144_144102

theorem temp_difference
  (temp_beijing : ℤ) 
  (temp_hangzhou : ℤ) 
  (h_beijing : temp_beijing = -10) 
  (h_hangzhou : temp_hangzhou = -1) : 
  temp_beijing - temp_hangzhou = -9 := 
by 
  rw [h_beijing, h_hangzhou] 
  sorry

end temp_difference_l144_144102


namespace solve_for_x_l144_144562

theorem solve_for_x (x : ℤ) (h : 3 * x - 7 = 11) : x = 6 :=
by
  sorry

end solve_for_x_l144_144562


namespace frogs_need_new_pond_l144_144005

theorem frogs_need_new_pond
  (num_frogs : ℕ) 
  (num_tadpoles : ℕ) 
  (num_survivor_tadpoles : ℕ) 
  (pond_capacity : ℕ) 
  (hc1 : num_frogs = 5)
  (hc2 : num_tadpoles = 3 * num_frogs)
  (hc3 : num_survivor_tadpoles = (2 * num_tadpoles) / 3)
  (hc4 : pond_capacity = 8):
  ((num_frogs + num_survivor_tadpoles) - pond_capacity) = 7 :=
by sorry

end frogs_need_new_pond_l144_144005


namespace womenInBusinessClass_l144_144267

-- Given conditions
def totalPassengers : ℕ := 300
def percentageWomen : ℚ := 70 / 100
def percentageWomenBusinessClass : ℚ := 15 / 100

def numberOfWomen (totalPassengers : ℕ) (percentageWomen : ℚ) : ℚ := 
  totalPassengers * percentageWomen

def numberOfWomenBusinessClass (numberOfWomen : ℚ) (percentageWomenBusinessClass : ℚ) : ℚ := 
  numberOfWomen * percentageWomenBusinessClass

-- Theorem to prove
theorem womenInBusinessClass (totalPassengers : ℕ) (percentageWomen : ℚ) (percentageWomenBusinessClass : ℚ) :
  numberOfWomenBusinessClass (numberOfWomen totalPassengers percentageWomen) percentageWomenBusinessClass = 32 := 
by 
  -- The proof steps would go here
  sorry

end womenInBusinessClass_l144_144267


namespace find_x_l144_144454

theorem find_x :
  ∃ x : ℝ, ((x * 0.85) / 2.5) - (8 * 2.25) = 5.5 ∧
  x = 69.11764705882353 :=
by
  sorry

end find_x_l144_144454


namespace probability_at_least_one_boy_and_one_girl_l144_144624

theorem probability_at_least_one_boy_and_one_girl :
  (∀ (n : ℕ), (ℙ(birth_is_boy) = ℙ(birth_is_girl)) ∧ n = 4) →
  (∃ p : ℚ, p = 7 / 8 ∧
    p = 1 - (ℙ(all_boys) + ℙ(all_girls))) :=
by
  sorry

-- Definitions to be used
def birth_is_boy := sorry -- Placeholder for an event where a birth is a boy
def birth_is_girl := sorry -- Placeholder for an event where a birth is a girl
def all_boys := sorry -- Placeholder for an event where all four children are boys
def all_girls := sorry -- Placeholder for an event where all four children are girls

end probability_at_least_one_boy_and_one_girl_l144_144624


namespace total_min_waiting_time_total_max_waiting_time_total_expected_waiting_time_l144_144299

variables (a b: ℕ) (n m: ℕ)

def C (x y : ℕ) : ℕ := x.choose y

def T_min (a n m : ℕ) : ℕ :=
  a * C n 2 + a * m * n + b * C m 2

def T_max (a n m : ℕ) : ℕ :=
  a * C n 2 + b * m * n + b * C m 2

def E_T (a b n m : ℕ) : ℕ :=
  C (n + m) 2 * ((b * m + a * n) / (m + n))

theorem total_min_waiting_time (a b : ℕ) : T_min 1 5 3 = 40 :=
  by sorry

theorem total_max_waiting_time (a b : ℕ) : T_max 1 5 3 = 100 :=
  by sorry

theorem total_expected_waiting_time (a b : ℕ) : E_T 1 5 5 3 = 70 :=
  by sorry

end total_min_waiting_time_total_max_waiting_time_total_expected_waiting_time_l144_144299


namespace intersection_of_lines_l144_144652

theorem intersection_of_lines :
  ∃ x y : ℚ, (8 * x - 3 * y = 9) ∧ (6 * x + 2 * y = 20) ∧ (x = 39 / 17) ∧ (y = 53 / 17) :=
by
  sorry

end intersection_of_lines_l144_144652


namespace number_of_mixed_vegetable_plates_l144_144159

def cost_of_chapati := 6
def cost_of_rice := 45
def cost_of_mixed_vegetable := 70
def chapatis_ordered := 16
def rice_ordered := 5
def ice_cream_cups := 6 -- though not used, included for completeness
def total_amount_paid := 1111

def total_cost_of_known_items := (chapatis_ordered * cost_of_chapati) + (rice_ordered * cost_of_rice)
def amount_spent_on_mixed_vegetable := total_amount_paid - total_cost_of_known_items

theorem number_of_mixed_vegetable_plates : 
  amount_spent_on_mixed_vegetable / cost_of_mixed_vegetable = 11 := 
by sorry

end number_of_mixed_vegetable_plates_l144_144159


namespace num_valid_pairs_l144_144320

theorem num_valid_pairs (a b : ℕ) (hb : b > a) (h_unpainted_area : ab = 3 * (a - 4) * (b - 4)) :
  (∃ (a b : ℕ), b > a ∧ ab = 3 * (a-4) * (b-4) ∧ (a-6) * (b-6) = 12 ∧ ((a, b) = (7, 18) ∨ (a, b) = (8, 12))) ∧
  (2 = 2) :=
by sorry

end num_valid_pairs_l144_144320


namespace fraction_of_selected_color_films_equals_five_twenty_sixths_l144_144152

noncomputable def fraction_of_selected_color_films (x y : ℕ) : ℚ :=
  let bw_films := 40 * x
  let color_films := 10 * y
  let selected_bw_films := (y / x * 1 / 100) * bw_films
  let selected_color_films := color_films
  let total_selected_films := selected_bw_films + selected_color_films
  selected_color_films / total_selected_films

theorem fraction_of_selected_color_films_equals_five_twenty_sixths (x y : ℕ) (h1 : x > 0) (h2 : y > 0) :
  fraction_of_selected_color_films x y = 5 / 26 := by
  sorry

end fraction_of_selected_color_films_equals_five_twenty_sixths_l144_144152


namespace correct_option_is_C_l144_144766

variable (a b : ℝ)

def option_A : Prop := (a - b) ^ 2 = a ^ 2 - b ^ 2
def option_B : Prop := a ^ 2 + a ^ 2 = a ^ 4
def option_C : Prop := (a ^ 2) ^ 3 = a ^ 6
def option_D : Prop := a ^ 2 * a ^ 2 = a ^ 6

theorem correct_option_is_C : option_C a :=
by
  sorry

end correct_option_is_C_l144_144766


namespace volume_frustum_l144_144609

noncomputable def volume_of_frustum (base_edge_original : ℝ) (altitude_original : ℝ) 
(base_edge_smaller : ℝ) (altitude_smaller : ℝ) : ℝ :=
let volume_original := (1 / 3) * (base_edge_original ^ 2) * altitude_original
let volume_smaller := (1 / 3) * (base_edge_smaller ^ 2) * altitude_smaller
(volume_original - volume_smaller)

theorem volume_frustum
  (base_edge_original : ℝ) (altitude_original : ℝ) 
  (base_edge_smaller : ℝ) (altitude_smaller : ℝ)
  (h_base_edge_original : base_edge_original = 10)
  (h_altitude_original : altitude_original = 10)
  (h_base_edge_smaller : base_edge_smaller = 5)
  (h_altitude_smaller : altitude_smaller = 5) :
  volume_of_frustum base_edge_original altitude_original base_edge_smaller altitude_smaller = (875 / 3) :=
by
  rw [h_base_edge_original, h_altitude_original, h_base_edge_smaller, h_altitude_smaller]
  simp [volume_of_frustum]
  sorry

end volume_frustum_l144_144609


namespace find_x_of_series_eq_15_l144_144172

noncomputable def infinite_series (x : ℝ) : ℝ :=
  5 + (5 + x) / 3 + (5 + 2 * x) / 3^2 + (5 + 3 * x) / 3^3 + ∑' n, (5 + (n + 1) * x) / 3 ^ (n + 1)

theorem find_x_of_series_eq_15 (x : ℝ) (h : infinite_series x = 15) : x = 10 :=
sorry

end find_x_of_series_eq_15_l144_144172


namespace max_value_of_f_l144_144203

def f (x : ℝ) : ℝ := 9 * x - 4 * x^2

theorem max_value_of_f :
  (∀ x : ℝ, f x ≤ 5.0625) ∧ (∃ x : ℝ, f x = 5.0625) :=
by
  sorry

end max_value_of_f_l144_144203


namespace find_pair_l144_144755

noncomputable def x_n (n : ℕ) : ℝ := n / (n + 2016)

theorem find_pair :
  ∃ (m n : ℕ), x_n 2016 = (x_n m) * (x_n n) ∧ (m = 6048 ∧ n = 4032) :=
by {
  sorry
}

end find_pair_l144_144755


namespace find_f_minus_3_l144_144084

def rational_function (f : ℚ → ℚ) : Prop :=
  ∀ x : ℚ, x ≠ 0 → 4 * f (1 / x) + (3 * f x / x) = 2 * x^2

theorem find_f_minus_3 (f : ℚ → ℚ) (h : rational_function f) : 
  f (-3) = 494 / 117 :=
by
  sorry

end find_f_minus_3_l144_144084


namespace cone_sections_equal_surface_area_l144_144308

theorem cone_sections_equal_surface_area {m r : ℝ} (h_r_pos : r > 0) (h_m_pos : m > 0) :
  ∃ (m1 m2 : ℝ), 
  (m1 = m / Real.sqrt 3) ∧ 
  (m2 = m / 3 * Real.sqrt 6) :=
sorry

end cone_sections_equal_surface_area_l144_144308


namespace arithmetic_square_root_of_nine_l144_144898

theorem arithmetic_square_root_of_nine : Real.sqrt 9 = 3 :=
sorry

end arithmetic_square_root_of_nine_l144_144898


namespace height_of_right_triangle_l144_144264

theorem height_of_right_triangle (a b c : ℝ) (h : ℝ) (h_right : a^2 + b^2 = c^2) (h_area : h = (a * b) / c) : h = (a * b) / c := 
by
  sorry

end height_of_right_triangle_l144_144264


namespace flyers_left_l144_144241

theorem flyers_left (initial_flyers : ℕ) (jack_flyers : ℕ) (rose_flyers : ℕ) (left_flyers : ℕ) :
  initial_flyers = 1236 →
  jack_flyers = 120 →
  rose_flyers = 320 →
  left_flyers = 796 →
  initial_flyers - (jack_flyers + rose_flyers) = left_flyers := 
by
  intros h_initial h_jack h_rose h_left
  rw [h_initial, h_jack, h_rose, h_left]
  simp
  sorry

end flyers_left_l144_144241


namespace set_representation_listing_method_l144_144756

def is_in_set (a : ℤ) : Prop := 0 < 2 * a - 1 ∧ 2 * a - 1 ≤ 5

def M : Set ℤ := {a | is_in_set a}

theorem set_representation_listing_method :
  M = {1, 2, 3} :=
sorry

end set_representation_listing_method_l144_144756


namespace ratio_of_elements_l144_144032

theorem ratio_of_elements (total_weight : ℕ) (element_B_weight : ℕ) 
  (h_total : total_weight = 324) (h_B : element_B_weight = 270) :
  (total_weight - element_B_weight) / element_B_weight = 1 / 5 :=
by
  sorry

end ratio_of_elements_l144_144032


namespace min_cubes_l144_144473

theorem min_cubes (a b c : ℕ) (h₁ : (a - 1) * (b - 1) * (c - 1) = 240) : a * b * c = 385 :=
  sorry

end min_cubes_l144_144473


namespace vendor_throws_away_8_percent_l144_144608

theorem vendor_throws_away_8_percent (total_apples: ℕ) (h₁ : total_apples > 0) :
    let apples_after_first_day := total_apples * 40 / 100
    let thrown_away_first_day := apples_after_first_day * 10 / 100
    let apples_after_second_day := (apples_after_first_day - thrown_away_first_day) * 30 / 100
    let thrown_away_second_day := apples_after_second_day * 20 / 100
    let apples_after_third_day := (apples_after_second_day - thrown_away_second_day) * 60 / 100
    let thrown_away_third_day := apples_after_third_day * 30 / 100
    total_apples > 0 → (8 : ℕ) * total_apples = (thrown_away_first_day + thrown_away_second_day + thrown_away_third_day) * 100 := 
by
    -- Placeholder proof
    sorry

end vendor_throws_away_8_percent_l144_144608


namespace find_a_l144_144215

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^x / (1 + a * 2^x)

theorem find_a (a : ℝ) (f : ℝ → ℝ) (h_f_def : ∀ x, f x = 2^x / (1 + a * 2^x))
  (h_symm : ∀ x, f x + f (-x) = 1) : a = 1 :=
sorry

end find_a_l144_144215


namespace wire_lengths_l144_144324

variables (total_length first second third fourth : ℝ)

def wire_conditions : Prop :=
  total_length = 72 ∧
  first = second + 3 ∧
  third = 2 * second - 2 ∧
  fourth = 0.5 * (first + second + third) ∧
  second + first + third + fourth = total_length

theorem wire_lengths 
  (h : wire_conditions total_length first second third fourth) :
  second = 11.75 ∧ first = 14.75 ∧ third = 21.5 ∧ fourth = 24 :=
sorry

end wire_lengths_l144_144324


namespace childSupportOwed_l144_144544

def annualIncomeBeforeRaise : ℕ := 30000
def yearsBeforeRaise : ℕ := 3
def raisePercentage : ℕ := 20
def annualIncomeAfterRaise (incomeBeforeRaise raisePercentage : ℕ) : ℕ :=
  incomeBeforeRaise + (incomeBeforeRaise * raisePercentage / 100)
def yearsAfterRaise : ℕ := 4
def childSupportPercentage : ℕ := 30
def amountPaid : ℕ := 1200

def calculateChildSupport (incomeYears : ℕ → ℕ → ℕ) (supportPercentage : ℕ) (years : ℕ) : ℕ :=
  (incomeYears years supportPercentage) * supportPercentage / 100 * years

def totalChildSupportOwed : ℕ :=
  (calculateChildSupport (λ _ _ => annualIncomeBeforeRaise) childSupportPercentage yearsBeforeRaise) +
  (calculateChildSupport (λ _ _ => annualIncomeAfterRaise annualIncomeBeforeRaise raisePercentage) childSupportPercentage yearsAfterRaise)

theorem childSupportOwed : totalChildSupportOwed - amountPaid = 69000 :=
by trivial

end childSupportOwed_l144_144544


namespace cost_price_per_meter_l144_144020

def selling_price_for_85_meters : ℝ := 8925
def profit_per_meter : ℝ := 25
def number_of_meters : ℝ := 85

theorem cost_price_per_meter : (selling_price_for_85_meters - profit_per_meter * number_of_meters) / number_of_meters = 80 := by
  sorry

end cost_price_per_meter_l144_144020


namespace projection_matrix_is_correct_l144_144189

noncomputable def projectionMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  let v : Fin 2 → ℝ := ![3, 4]
  (1 / (v 0 ^ 2 + v 1 ^ 2)) • (λ i j, v i * v j)

theorem projection_matrix_is_correct :
  projectionMatrix = ![![9/25, 12/25], ![12/25, 16/25]] :=
by
  sorry

end projection_matrix_is_correct_l144_144189


namespace fraction_irreducible_l144_144559
-- Import necessary libraries

-- Define the problem to prove
theorem fraction_irreducible (n: ℕ) (h: n > 0) : gcd (21 * n + 4) (14 * n + 3) = 1 := 
  sorry

end fraction_irreducible_l144_144559


namespace Cagney_and_Lacey_Cupcakes_l144_144479

-- Conditions
def CagneyRate := 1 / 25 -- cupcakes per second
def LaceyRate := 1 / 35 -- cupcakes per second
def TotalTimeInSeconds := 10 * 60 -- total time in seconds
def LaceyPrepTimeInSeconds := 1 * 60 -- Lacey's preparation time in seconds
def EffectiveWorkTimeInSeconds := TotalTimeInSeconds - LaceyPrepTimeInSeconds -- effective working time

-- Calculate combined rate
def CombinedRate := 1 / (1 / CagneyRate + 1 / LaceyRate) -- combined rate in cupcakes per second

-- Calculate the total number of cupcakes frosted
def TotalCupcakesFrosted := EffectiveWorkTimeInSeconds * CombinedRate -- total cupcakes frosted

-- We state the theorem that corresponds to our proof problem
theorem Cagney_and_Lacey_Cupcakes : TotalCupcakesFrosted = 37 := by
  sorry

end Cagney_and_Lacey_Cupcakes_l144_144479


namespace polygon_perimeter_exposure_l144_144633

theorem polygon_perimeter_exposure:
  let triangle_sides := 3
  let square_sides := 4
  let pentagon_sides := 5
  let hexagon_sides := 6
  let heptagon_sides := 7
  let octagon_sides := 8
  let nonagon_sides := 9
  let exposure_triangle_nonagon := triangle_sides + nonagon_sides - 2
  let other_polygons_adjacency := 2 * 5
  let exposure_other_polygons := square_sides + pentagon_sides + hexagon_sides + heptagon_sides + octagon_sides - other_polygons_adjacency
  exposure_triangle_nonagon + exposure_other_polygons = 30 :=
by sorry

end polygon_perimeter_exposure_l144_144633


namespace avg_last_four_is_63_75_l144_144939

noncomputable def average_of_list (l : List ℝ) : ℝ :=
  l.sum / l.length

variable (l : List ℝ)
variable (h_lenl : l.length = 7)
variable (h_avg7 : average_of_list l = 60)
variable (h_l3 : List ℝ := l.take 3)
variable (h_l4 : List ℝ := l.drop 3)
variable (h_avg3 : average_of_list h_l3 = 55)

theorem avg_last_four_is_63_75 : average_of_list h_l4 = 63.75 :=
by
  sorry

end avg_last_four_is_63_75_l144_144939


namespace quadratic_polynomial_discriminant_l144_144369

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₁ : a ≠ 0)
  (h₂ : ∃ x : ℝ, a * x^2 + b * x + c = x - 2 ∧ (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h₃ : ∃ x : ℝ, a * x^2 + b * x + c = 1 - x / 2 ∧ (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l144_144369


namespace relationship_between_y1_y2_l144_144774

theorem relationship_between_y1_y2 (y1 y2 : ℝ)
  (h1 : y1 = -2 * (-2) + 3)
  (h2 : y2 = -2 * 3 + 3) :
  y1 > y2 := by
  sorry

end relationship_between_y1_y2_l144_144774


namespace elements_representable_as_sum_l144_144869

open Finset
open Fintype

variables (p : ℕ) [fact (Nat.Prime p)]
variables (a : Fin (p - 1) → ℤ)

theorem elements_representable_as_sum (h : ∀ i, a i ≠ 0) :
  ∀ x : Fin p, ∃ S : Finset (Fin (p - 1)), x = ∑ i in S, a i % p :=
sorry

end elements_representable_as_sum_l144_144869


namespace range_of_expression_l144_144661

theorem range_of_expression (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  - π / 6 < 2 * α - β / 2 ∧ 2 * α - β / 2 < π :=
sorry

end range_of_expression_l144_144661


namespace conference_duration_excluding_breaks_l144_144992

-- Definitions based on the conditions
def total_hours : Nat := 14
def additional_minutes : Nat := 20
def break_minutes : Nat := 15

-- Total time including breaks
def total_time_minutes : Nat := total_hours * 60 + additional_minutes
-- Number of breaks
def number_of_breaks : Nat := total_hours
-- Total break time
def total_break_minutes : Nat := number_of_breaks * break_minutes

-- Proof statement
theorem conference_duration_excluding_breaks :
  total_time_minutes - total_break_minutes = 650 := by
  sorry

end conference_duration_excluding_breaks_l144_144992


namespace shopkeeper_profit_percent_l144_144605

theorem shopkeeper_profit_percent
  (initial_value : ℝ)
  (percent_lost_theft : ℝ)
  (percent_total_loss : ℝ)
  (remaining_value : ℝ)
  (total_loss_value : ℝ)
  (selling_price : ℝ)
  (profit : ℝ)
  (profit_percent : ℝ)
  (h_initial_value : initial_value = 100)
  (h_percent_lost_theft : percent_lost_theft = 20)
  (h_percent_total_loss : percent_total_loss = 12)
  (h_remaining_value : remaining_value = initial_value - (percent_lost_theft / 100) * initial_value)
  (h_total_loss_value : total_loss_value = (percent_total_loss / 100) * initial_value)
  (h_selling_price : selling_price = initial_value - total_loss_value)
  (h_profit : profit = selling_price - remaining_value)
  (h_profit_percent : profit_percent = (profit / remaining_value) * 100) :
  profit_percent = 10 := by
  sorry

end shopkeeper_profit_percent_l144_144605


namespace original_sandbox_capacity_l144_144150

theorem original_sandbox_capacity :
  ∃ (L W H : ℝ), 8 * (L * W * H) = 80 → L * W * H = 10 :=
by
  sorry

end original_sandbox_capacity_l144_144150


namespace fencing_rate_correct_l144_144344

noncomputable def rate_per_meter (d : ℝ) (cost : ℝ) : ℝ :=
  cost / (Real.pi * d)

theorem fencing_rate_correct : rate_per_meter 26 122.52211349000194 = 1.5 := by
  sorry

end fencing_rate_correct_l144_144344


namespace arithmetic_square_root_of_9_l144_144914

theorem arithmetic_square_root_of_9 : Real.sqrt 9 = 3 := by
  sorry

end arithmetic_square_root_of_9_l144_144914


namespace diff_of_squares_div_l144_144589

-- Definitions from the conditions
def a : ℕ := 125
def b : ℕ := 105

-- The main statement to be proved
theorem diff_of_squares_div {a b : ℕ} (h1 : a = 125) (h2 : b = 105) : (a^2 - b^2) / 20 = 230 := by
  sorry

end diff_of_squares_div_l144_144589


namespace arith_seq_sum_first_110_l144_144054

variable {α : Type*} [OrderedRing α]

theorem arith_seq_sum_first_110 (a₁ d : α) :
  (10 * a₁ + 45 * d = 100) →
  (100 * a₁ + 4950 * d = 10) →
  (110 * a₁ + 5995 * d = -110) :=
by
  intros h1 h2
  sorry

end arith_seq_sum_first_110_l144_144054


namespace chess_game_probabilities_l144_144105

theorem chess_game_probabilities :
  let p_draw := 1 / 2
  let p_b_win := 1 / 3
  let p_sum := 1
  let p_a_win := p_sum - p_draw - p_b_win
  let p_a_not_lose := p_draw + p_a_win
  let p_b_not_lose := p_draw + p_b_win
  A := p_a_win = 1 / 6
  B := p_a_not_lose = 1 / 2
  C := p_a_win = 2 / 3
  D := p_b_not_lose = 1 / 2
  in ¬ (p_a_win = 1 / 6 ∧ p_a_not_lose ≠ 1 / 2 ∧ p_a_win ≠ 2 / 3 ∧ p_b_not_lose ≠ 1 / 2)
:=
sorry

end chess_game_probabilities_l144_144105


namespace imaginary_part_of_exp_neg_pi_div_6_eq_neg_one_half_l144_144269

theorem imaginary_part_of_exp_neg_pi_div_6_eq_neg_one_half :
  (Complex.exp (-Complex.I * Real.pi / 6)).im = -1/2 := by
sorry

end imaginary_part_of_exp_neg_pi_div_6_eq_neg_one_half_l144_144269


namespace sum_prob_less_one_l144_144539

theorem sum_prob_less_one (x y z : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) (hz : 0 < z ∧ z < 1) :
  x * (1 - y) * (1 - z) + (1 - x) * y * (1 - z) + (1 - x) * (1 - y) * z < 1 :=
by
  sorry

end sum_prob_less_one_l144_144539


namespace recycle_cans_l144_144502

theorem recycle_cans (initial_cans : ℕ) (recycle_rate : ℕ) (n1 n2 n3 : ℕ)
  (h1 : initial_cans = 450)
  (h2 : recycle_rate = 5)
  (h3 : n1 = initial_cans / recycle_rate)
  (h4 : n2 = n1 / recycle_rate)
  (h5 : n3 = n2 / recycle_rate)
  (h6 : n3 / recycle_rate = 0) : 
  n1 + n2 + n3 = 111 :=
by
  sorry

end recycle_cans_l144_144502


namespace number_of_white_cats_l144_144401

theorem number_of_white_cats (total_cats : ℕ) (percent_black : ℤ) (grey_cats : ℕ) : 
  total_cats = 16 → 
  percent_black = 25 →
  grey_cats = 10 → 
  (total_cats - (total_cats * percent_black / 100 + grey_cats)) = 2 :=
by
  intros
  sorry

end number_of_white_cats_l144_144401


namespace solve_equation_l144_144199

noncomputable def smallest_solution : Rat :=
  (8 - Real.sqrt 145) / 3

theorem solve_equation : 
  ∃ x : ℝ, (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ x = smallest_solution := sorry

end solve_equation_l144_144199


namespace min_value_frac_sum_l144_144540

theorem min_value_frac_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (3 * z / (x + 2 * y) + 5 * x / (2 * y + 3 * z) + 2 * y / (3 * x + z)) ≥ 3 / 4 :=
by
  sorry

end min_value_frac_sum_l144_144540


namespace preceding_integer_l144_144226

def bin_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc bit => 2 * acc + if bit then 1 else 0) 0

theorem preceding_integer : bin_to_nat [true, true, false, false, false] - 1 = bin_to_nat [true, false, true, true, true] := by
  sorry

end preceding_integer_l144_144226


namespace job_candidates_excel_nights_l144_144980

theorem job_candidates_excel_nights (hasExcel : ℝ) (dayShift : ℝ) 
    (h1 : hasExcel = 0.2) (h2 : dayShift = 0.7) : 
    (1 - dayShift) * hasExcel = 0.06 :=
by
  sorry

end job_candidates_excel_nights_l144_144980


namespace not_m_gt_132_l144_144720

theorem not_m_gt_132 (m : ℕ) (hm : 0 < m)
  (H : ∃ (k : ℕ), 1 / 2 + 1 / 3 + 1 / 11 + 1 / (m:ℚ) = k) :
  m ≤ 132 :=
sorry

end not_m_gt_132_l144_144720


namespace count_right_triangles_with_given_conditions_l144_144207

-- Define the type of our points as a pair of integers
def Point := (ℤ × ℤ)

-- Define the orthocenter being a specific point
def isOrthocenter (P : Point) := P = (-1, 7)

-- Define that a given triangle has a right angle at the origin
def rightAngledAtOrigin (O A B : Point) :=
  O = (0, 0) ∧
  (A.fst = 0 ∨ A.snd = 0) ∧
  (B.fst = 0 ∨ B.snd = 0) ∧
  (A.fst ≠ 0 ∨ A.snd ≠ 0) ∧
  (B.fst ≠ 0 ∨ B.snd ≠ 0)

-- Define that the points are lattice points
def areLatticePoints (O A B : Point) :=
  ∃ t k : ℤ, (A = (3 * t, 4 * t) ∧ B = (-4 * k, 3 * k)) ∨
            (B = (3 * t, 4 * t) ∧ A = (-4 * k, 3 * k))

-- Define the number of right triangles given the constraints
def numberOfRightTriangles : ℕ := 2

-- Statement of the problem
theorem count_right_triangles_with_given_conditions :
  ∃ (O A B : Point),
    rightAngledAtOrigin O A B ∧
    isOrthocenter (-1, 7) ∧
    areLatticePoints O A B ∧
    numberOfRightTriangles = 2 :=
  sorry

end count_right_triangles_with_given_conditions_l144_144207


namespace kyoko_bought_three_balls_l144_144851

theorem kyoko_bought_three_balls
  (cost_per_ball : ℝ)
  (total_paid : ℝ)
  (number_of_balls : ℝ)
  (h_cost_per_ball : cost_per_ball = 1.54)
  (h_total_paid : total_paid = 4.62)
  (h_number_of_balls : number_of_balls = total_paid / cost_per_ball) :
  number_of_balls = 3 := by
  sorry

end kyoko_bought_three_balls_l144_144851


namespace max_remainder_when_divided_by_8_l144_144984

-- Define the problem: greatest possible remainder when apples divided by 8.
theorem max_remainder_when_divided_by_8 (n : ℕ) : ∃ r : ℕ, r < 8 ∧ r = 7 ∧ n % 8 = r := 
sorry

end max_remainder_when_divided_by_8_l144_144984


namespace discriminant_of_P_l144_144354

theorem discriminant_of_P (a b c : ℚ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1/2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1/2 := 
begin
  -- Proof omitted for brevity
  sorry
end

end discriminant_of_P_l144_144354


namespace correct_operation_l144_144317

theorem correct_operation (x : ℝ) (f : ℝ → ℝ) (h : ∀ x, (x / 10) = 0.01 * f x) : 
  f x = 10 * x :=
by
  sorry

end correct_operation_l144_144317


namespace arithmetic_sqrt_of_9_l144_144934

def arithmetic_sqrt (n : ℕ) : ℕ :=
  Nat.sqrt n

theorem arithmetic_sqrt_of_9 : arithmetic_sqrt 9 = 3 :=
by
  sorry

end arithmetic_sqrt_of_9_l144_144934


namespace range_of_a1_l144_144403

noncomputable def infinite_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * q ^ n

lemma geometric_series_sum (a1 q : ℝ) (h : |q| < 1) :
  ∑' (n : ℕ), infinite_geometric_sequence a1 q n = a1 / (1 - q) := sorry

theorem range_of_a1 (a1 : ℝ) (q : ℝ) 
  (h : |q| < 1) 
  (h_sum : ∑' (n : ℕ), infinite_geometric_sequence a1 q n = 1 / 2) : 
  a1 ∈ (set.Ioo 0 (1/2) ∪ set.Ioo (1/2) 1) :=
begin
  sorry
end

end range_of_a1_l144_144403


namespace hamsters_count_l144_144754

-- Define the conditions as parameters
variables (ratio_rabbit_hamster : ℕ × ℕ)
variables (rabbits : ℕ)
variables (hamsters : ℕ)

-- Given conditions
def ratio_condition : ratio_rabbit_hamster = (4, 5) := sorry
def rabbits_condition : rabbits = 20 := sorry

-- The theorem to be proven
theorem hamsters_count : ratio_rabbit_hamster = (4, 5) -> rabbits = 20 -> hamsters = 25 :=
by
  intro h1 h2
  sorry

end hamsters_count_l144_144754


namespace problem_1_problem_2_l144_144862

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| + |3 * x - 2|

theorem problem_1 {a b : ℝ} (h : ∀ x, f x ≤ 5 → -4 * a / 5 ≤ x ∧ x ≤ 3 * b / 5) : 
  a = 1 ∧ b = 2 :=
sorry

theorem problem_2 {a b m : ℝ} (h1 : a = 1) (h2 : b = 2) (h3 : ∀ x, |x - a| + |x + b| ≥ m^2 - 3 * m + 5) :
  ∃ m, m = 2 :=
sorry

end problem_1_problem_2_l144_144862


namespace train_distance_difference_l144_144583

theorem train_distance_difference:
  ∀ (D1 D2 : ℕ) (t : ℕ), 
    (D1 = 20 * t) →            -- Slower train's distance
    (D2 = 25 * t) →           -- Faster train's distance
    (D1 + D2 = 450) →         -- Total distance between stations
    (D2 - D1 = 50) := 
by
  intros D1 D2 t h1 h2 h3
  sorry

end train_distance_difference_l144_144583


namespace sin_4A_plus_sin_4B_plus_sin_4C_eq_neg_4_sin_2A_sin_2B_sin_2C_l144_144977

theorem sin_4A_plus_sin_4B_plus_sin_4C_eq_neg_4_sin_2A_sin_2B_sin_2C
  {A B C : ℝ}
  (h : A + B + C = π) :
  Real.sin (4 * A) + Real.sin (4 * B) + Real.sin (4 * C) = -4 * Real.sin (2 * A) * Real.sin (2 * B) * Real.sin (2 * C) :=
sorry

end sin_4A_plus_sin_4B_plus_sin_4C_eq_neg_4_sin_2A_sin_2B_sin_2C_l144_144977


namespace total_donation_l144_144484

-- Definitions
def cassandra_pennies : ℕ := 5000
def james_deficit : ℕ := 276
def james_pennies : ℕ := cassandra_pennies - james_deficit

-- Theorem to prove the total donation
theorem total_donation : cassandra_pennies + james_pennies = 9724 :=
by
  -- Proof is omitted
  sorry

end total_donation_l144_144484


namespace find_b_l144_144578

theorem find_b (a b : ℝ) (k : ℝ) (h1 : a * b = k) (h2 : a + b = 40) (h3 : a - 2 * b = 10) (ha : a = 4) : b = 75 :=
  sorry

end find_b_l144_144578


namespace nigel_gave_away_l144_144101

theorem nigel_gave_away :
  ∀ (original : ℕ) (gift_from_mother : ℕ) (final : ℕ) (money_given_away : ℕ),
    original = 45 →
    gift_from_mother = 80 →
    final = 2 * original + 10 →
    final = original - money_given_away + gift_from_mother →
    money_given_away = 25 :=
by
  intros original gift_from_mother final money_given_away
  sorry

end nigel_gave_away_l144_144101


namespace least_adjacent_probability_l144_144129

theorem least_adjacent_probability (n : ℕ) 
    (h₀ : 0 < n)
    (h₁ : (∀ m : ℕ, 0 < m ∧ m < n → (4 * m^2 - 4 * m + 8) / (m^2 * (m^2 - 1)) ≥ 1 / 2015)) : 
    (4 * n^2 - 4 * n + 8) / (n^2 * (n^2 - 1)) < 1 / 2015 := by
  sorry

end least_adjacent_probability_l144_144129


namespace find_integer_tuples_l144_144812

theorem find_integer_tuples (a b c x y z : ℤ) :
  a + b + c = x * y * z →
  x + y + z = a * b * c →
  a ≥ b → b ≥ c → c ≥ 1 →
  x ≥ y → y ≥ z → z ≥ 1 →
  (a, b, c, x, y, z) = (2, 2, 2, 6, 1, 1) ∨
  (a, b, c, x, y, z) = (5, 2, 1, 8, 1, 1) ∨
  (a, b, c, x, y, z) = (3, 3, 1, 7, 1, 1) ∨
  (a, b, c, x, y, z) = (3, 2, 1, 6, 2, 1) :=
by
  sorry

end find_integer_tuples_l144_144812


namespace q_minus_r_max_value_l144_144143

theorem q_minus_r_max_value :
  ∃ (q r : ℕ), q > 99 ∧ q < 1000 ∧ r > 99 ∧ r < 1000 ∧ 
    q = 100 * (q / 100) + 10 * ((q / 10) % 10) + (q % 10) ∧ 
    r = 100 * (q % 10) + 10 * ((q / 10) % 10) + (q / 100) ∧ 
    q - r = 297 :=
by sorry

end q_minus_r_max_value_l144_144143


namespace cost_of_milk_l144_144279

-- Given conditions
def total_cost_of_groceries : ℕ := 42
def cost_of_bananas : ℕ := 12
def cost_of_bread : ℕ := 9
def cost_of_apples : ℕ := 14

-- Prove that the cost of milk is $7
theorem cost_of_milk : total_cost_of_groceries - (cost_of_bananas + cost_of_bread + cost_of_apples) = 7 := 
by 
  sorry

end cost_of_milk_l144_144279


namespace Ganesh_avg_speed_l144_144771

theorem Ganesh_avg_speed (D : ℝ) : 
  (∃ (V : ℝ), (39.6 = (2 * D) / ((D / 44) + (D / V))) ∧ V = 36) :=
by
  sorry

end Ganesh_avg_speed_l144_144771


namespace find_m_n_l144_144108

theorem find_m_n (m n : ℤ) (h : m^2 - 2 * m * n + 2 * n^2 - 8 * n + 16 = 0) : m = 4 ∧ n = 4 := 
by {
  sorry
}

end find_m_n_l144_144108


namespace simplify_evaluate_expression_l144_144268

theorem simplify_evaluate_expression (a b : ℤ) (h1 : a = -2) (h2 : b = 4) : 
  (-(3 * a)^2 + 6 * a * b - (a^2 + 3 * (a - 2 * a * b))) = 14 :=
by
  rw [h1, h2]
  sorry

end simplify_evaluate_expression_l144_144268


namespace arithmetic_square_root_of_9_l144_144910

theorem arithmetic_square_root_of_9 : sqrt 9 = 3 := by
  sorry

end arithmetic_square_root_of_9_l144_144910


namespace correct_samples_for_senior_l144_144779

-- Define the total number of students in each section
def junior_students : ℕ := 400
def senior_students : ℕ := 200
def total_students : ℕ := junior_students + senior_students

-- Define the total number of samples to be drawn
def total_samples : ℕ := 60

-- Calculate the number of samples to be drawn from each section
def junior_samples : ℕ := total_samples * junior_students / total_students
def senior_samples : ℕ := total_samples - junior_samples

-- The theorem to prove
theorem correct_samples_for_senior :
  senior_samples = 20 :=
by
  sorry

end correct_samples_for_senior_l144_144779


namespace cosine_sine_inequality_theorem_l144_144213

theorem cosine_sine_inequality_theorem (θ : ℝ) :
  (∀ x : ℝ, 0 ≤ x → x ≤ 1 → 
    x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) ↔
    (π / 12 < θ ∧ θ < 5 * π / 12) :=
by
  sorry

end cosine_sine_inequality_theorem_l144_144213


namespace slope_of_tangent_at_A_l144_144948

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem slope_of_tangent_at_A :
  (deriv f 0) = 1 :=
by
  sorry

end slope_of_tangent_at_A_l144_144948


namespace recurring_decimal_sum_l144_144649

noncomputable def x : ℚ := 1 / 3

noncomputable def y : ℚ := 14 / 999

noncomputable def z : ℚ := 5 / 9999

theorem recurring_decimal_sum :
  x + y + z = 3478 / 9999 := by
  sorry

end recurring_decimal_sum_l144_144649


namespace arithmetic_sqrt_of_9_l144_144937

def arithmetic_sqrt (n : ℕ) : ℕ :=
  Nat.sqrt n

theorem arithmetic_sqrt_of_9 : arithmetic_sqrt 9 = 3 :=
by
  sorry

end arithmetic_sqrt_of_9_l144_144937


namespace marissa_tied_boxes_l144_144095

def Total_ribbon : ℝ := 4.5
def Leftover_ribbon : ℝ := 1
def Ribbon_per_box : ℝ := 0.7

theorem marissa_tied_boxes : (Total_ribbon - Leftover_ribbon) / Ribbon_per_box = 5 := by
  sorry

end marissa_tied_boxes_l144_144095


namespace triangle_angle_tangent_ratio_triangle_tan_A_minus_B_maximum_l144_144726

theorem triangle_angle_tangent_ratio (A B C : ℝ) (a b c : ℝ) (h1 : a * Real.cos B - b * Real.cos A = 3 / 5 * c) :
  Real.tan A / Real.tan B = 4 := sorry

theorem triangle_tan_A_minus_B_maximum (A B C : ℝ) (a b c : ℝ) (h1 : a * Real.cos B - b * Real.cos A = 3 / 5 * c)
  (h2 : Real.tan A / Real.tan B = 4) : Real.tan (A - B) ≤ 3 / 4 := sorry

end triangle_angle_tangent_ratio_triangle_tan_A_minus_B_maximum_l144_144726


namespace xy_value_l144_144519

namespace ProofProblem

variables {x y : ℤ}

theorem xy_value (h1 : x * (x + y) = x^2 + 12) (h2 : x - y = 3) : x * y = 12 :=
by
  -- The proof is not required here
  sorry

end ProofProblem

end xy_value_l144_144519


namespace totalCandies_l144_144075

def bobCandies : Nat := 10
def maryCandies : Nat := 5
def sueCandies : Nat := 20
def johnCandies : Nat := 5
def samCandies : Nat := 10

theorem totalCandies : bobCandies + maryCandies + sueCandies + johnCandies + samCandies = 50 := 
by
  sorry

end totalCandies_l144_144075


namespace inclination_angle_of_vertical_line_l144_144571

theorem inclination_angle_of_vertical_line :
  ∀ x : ℝ, x = Real.tan (60 * Real.pi / 180) → ∃ θ : ℝ, θ = 90 := by
  sorry

end inclination_angle_of_vertical_line_l144_144571


namespace arithmetic_sqrt_of_nine_l144_144921

-- Define the arithmetic square root function which only considers non-negative values
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ :=
  if hx : x ≥ 0 then Real.sqrt x else 0

-- The theorem to prove: The arithmetic square root of 9 is 3.
theorem arithmetic_sqrt_of_nine : arithmetic_sqrt 9 = 3 :=
by
  sorry

end arithmetic_sqrt_of_nine_l144_144921


namespace combined_ppf_two_females_l144_144958

open Real

/-- 
Proof that the combined PPF (Production Possibility Frontier) 
of two females, given their individual PPFs, is 
M = 80 - 2K with K ≤ 40 
-/

theorem combined_ppf_two_females (M K : ℝ) (h1 : M = 40 - 2 * K) (h2 : K ≤ 20) :
  M ≤ 80 - 2 * K :=

-- Given that the individual PPF for each of the two females is \( M = 40 - 2K \)
have h3 : M = 40 - 2 * K, by exact h1
-- The combined PPF of the two females is \( M = 80 - 2K \)

-- Given \( K \leq 20 \), the combined maximum \( K \leq 40 \)
have h4 : K ≤ 40, by linarith

show M ≤ 80 - 2 * K, by linarith

end combined_ppf_two_females_l144_144958


namespace total_people_ball_l144_144796

theorem total_people_ball (n m : ℕ) (h1 : n + m < 50) (h2 : 3 * n = 20 * m) : n + m = 41 := 
sorry

end total_people_ball_l144_144796


namespace verify_statements_l144_144596

def true_statements : List Nat := 
  [1, 3, 4]

theorem verify_statements :
  (¬ ∃ x : ℝ, x^2 - 3 * x + 3 = 0) ∧
  (¬ ∀ x : ℝ, -1/2 < x ∧ x < 0 → 2 * x^2 - 5 * x - 3 < 0) ∧
  (¬ (∀ x y : ℝ, x * y = 0 → x = 0 ∨ y = 0)) ∧
  (∀ k : ℝ, 9 < k ∧ k < 25 → 
      (∀ x y : ℝ, (x^2 / (25 - k) + y^2 / (9 - k) = 1 → x^2 / 25 + y^2 / 9 = 1))) ∧
  (¬ ∃ l1 l2 : ℝ → ℝ, 
      ((l1 (1) = 3) ∧ (l1^2 = 4*x)) ∧ ((l2 (1) = 3) ∧ (l2^2 = 4*x))) :=
by
  unfold true_statements
  sorry

end verify_statements_l144_144596


namespace probability_ge_sqrt2_l144_144221

noncomputable def probability_length_chord_ge_sqrt2
  (a : ℝ)
  (h : a ≠ 0)
  (intersect_cond : ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ (x - a)^2 + (y - a)^2 = 1)
  : ℝ :=
  if -1 ≤ a ∧ a ≤ 1 then (1 / Real.sqrt (1^2 + 1^2)) else 0

theorem probability_ge_sqrt2 
  (a : ℝ) 
  (h : a ≠ 0) 
  (intersect_cond : ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ (x - a)^2 + (y - a)^2 = 1)
  (length_cond : (Real.sqrt (4 - 2*a^2) ≥ Real.sqrt 2)) : 
  probability_length_chord_ge_sqrt2 a h intersect_cond = (Real.sqrt 2 / 2) :=
by
  sorry

end probability_ge_sqrt2_l144_144221


namespace find_value_l144_144659

variables (a b c d : ℝ)

theorem find_value
  (h1 : a - b = 3)
  (h2 : c + d = 2) :
  (a + c) - (b - d) = 5 :=
by sorry

end find_value_l144_144659


namespace john_ultramarathon_distance_l144_144076

theorem john_ultramarathon_distance :
  let initial_time := 8
  let time_increase_percentage := 0.75
  let speed_increase := 4
  let initial_speed := 8
  initial_time * (1 + time_increase_percentage) * (initial_speed + speed_increase) = 168 :=
by
  let initial_time := 8
  let time_increase_percentage := 0.75
  let speed_increase := 4
  let initial_speed := 8
  sorry

end john_ultramarathon_distance_l144_144076


namespace total_donation_l144_144486

-- Definitions
def cassandra_pennies : ℕ := 5000
def james_deficit : ℕ := 276
def james_pennies : ℕ := cassandra_pennies - james_deficit

-- Theorem to prove the total donation
theorem total_donation : cassandra_pennies + james_pennies = 9724 :=
by
  -- Proof is omitted
  sorry

end total_donation_l144_144486


namespace min_possible_value_l144_144255

theorem min_possible_value
  (a b c d e f g h : Int)
  (h_distinct : List.Nodup [a, b, c, d, e, f, g, h])
  (h_set_a : a ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_b : b ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_c : c ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_d : d ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_e : e ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_f : f ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_g : g ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_h : h ∈ [-9, -6, -3, 0, 1, 3, 6, 10]) :
  ∃ a b c d e f g h : Int,
  ((a + b + c + d)^2 + (e + f + g + h)^2) = 2
  :=
  sorry

end min_possible_value_l144_144255


namespace chairs_built_in_10_days_l144_144799

-- Define the conditions as variables
def hours_per_day : ℕ := 8
def days_worked : ℕ := 10
def hours_per_chair : ℕ := 5

-- State the problem as a conjecture or theorem
theorem chairs_built_in_10_days : (hours_per_day * days_worked) / hours_per_chair = 16 := by
    sorry

end chairs_built_in_10_days_l144_144799


namespace frogs_need_new_pond_l144_144004

theorem frogs_need_new_pond
  (num_frogs : ℕ) 
  (num_tadpoles : ℕ) 
  (num_survivor_tadpoles : ℕ) 
  (pond_capacity : ℕ) 
  (hc1 : num_frogs = 5)
  (hc2 : num_tadpoles = 3 * num_frogs)
  (hc3 : num_survivor_tadpoles = (2 * num_tadpoles) / 3)
  (hc4 : pond_capacity = 8):
  ((num_frogs + num_survivor_tadpoles) - pond_capacity) = 7 :=
by sorry

end frogs_need_new_pond_l144_144004


namespace probability_of_at_least_one_boy_and_one_girl_l144_144619

noncomputable def probability_at_least_one_boy_and_one_girl: ℚ :=
  7 / 8

axiom equally_likely_birth : ∀ i : ℕ, (i = 0 ∨ i = 1) → (0.5 : ℝ)

theorem probability_of_at_least_one_boy_and_one_girl (n : ℕ) (condition : n = 4) : 
  probability_at_least_one_boy_and_one_girl = 7 / 8 :=
sorry

end probability_of_at_least_one_boy_and_one_girl_l144_144619


namespace money_last_weeks_l144_144146

-- Define the amounts of money earned and spent per week
def money_mowing : ℕ := 5
def money_weed_eating : ℕ := 58
def weekly_spending : ℕ := 7

-- Define the total money earned
def total_money : ℕ := money_mowing + money_weed_eating

-- Define the number of weeks the money will last
def weeks_last (total : ℕ) (weekly : ℕ) : ℕ := total / weekly

-- Theorem stating the number of weeks the money will last
theorem money_last_weeks : weeks_last total_money weekly_spending = 9 := by
  sorry

end money_last_weeks_l144_144146


namespace calculate_molecular_weight_CaBr2_l144_144331

def atomic_weight_Ca : ℝ := 40.08                 -- The atomic weight of calcium (Ca)
def atomic_weight_Br : ℝ := 79.904                -- The atomic weight of bromine (Br)
def molecular_weight_CaBr2 : ℝ := atomic_weight_Ca + 2 * atomic_weight_Br  -- Definition of molecular weight of CaBr₂

theorem calculate_molecular_weight_CaBr2 : molecular_weight_CaBr2 = 199.888 := by
  sorry

end calculate_molecular_weight_CaBr2_l144_144331


namespace find_deleted_files_l144_144336

def original_files : Nat := 21
def remaining_files : Nat := 7
def deleted_files : Nat := 14

theorem find_deleted_files : original_files - remaining_files = deleted_files := by
  sorry

end find_deleted_files_l144_144336


namespace division_of_converted_values_l144_144832

theorem division_of_converted_values 
  (h : 144 * 177 = 25488) : 
  254.88 / 0.177 = 1440 := by
  sorry

end division_of_converted_values_l144_144832


namespace find_b_squared_l144_144940

theorem find_b_squared :
  let ellipse_eq := ∀ x y : ℝ, x^2 / 25 + y^2 / b^2 = 1
  let hyperbola_eq := ∀ x y : ℝ, x^2 / 225 - y^2 / 144 = 1 / 36
  let coinciding_foci := 
    let c_ellipse := Real.sqrt (25 - b^2)
    let c_hyperbola := Real.sqrt ((225 / 36) + (144 / 36))
    c_ellipse = c_hyperbola
  ellipse_eq ∧ hyperbola_eq ∧ coinciding_foci → b^2 = 14.75
:= by sorry

end find_b_squared_l144_144940


namespace ratio_of_a_to_b_and_c_l144_144777

theorem ratio_of_a_to_b_and_c (A B C : ℝ) (h1 : A = 160) (h2 : A + B + C = 400) (h3 : B = (2/3) * (A + C)) :
  A / (B + C) = 2 / 3 :=
by
  sorry

end ratio_of_a_to_b_and_c_l144_144777


namespace david_marks_in_english_l144_144493

theorem david_marks_in_english
  (math phys chem bio : ℕ)
  (avg subs : ℕ) 
  (h_math : math = 95) 
  (h_phys : phys = 82) 
  (h_chem : chem = 97) 
  (h_bio : bio = 95) 
  (h_avg : avg = 93)
  (h_subs : subs = 5) :
  ∃ E : ℕ, (avg * subs = E + math + phys + chem + bio) ∧ E = 96 :=
by
  sorry

end david_marks_in_english_l144_144493


namespace system_of_equations_solution_l144_144115

theorem system_of_equations_solution (x y z u v : ℤ) 
  (h1 : x + y + z + u = 5)
  (h2 : y + z + u + v = 1)
  (h3 : z + u + v + x = 2)
  (h4 : u + v + x + y = 0)
  (h5 : v + x + y + z = 4) :
  v = -2 ∧ x = 2 ∧ y = 1 ∧ z = 3 ∧ u = -1 := 
by 
  sorry

end system_of_equations_solution_l144_144115


namespace maximum_value_of_x2y3z_l144_144830

theorem maximum_value_of_x2y3z (x y z : ℝ) (h : x^2 + y^2 + z^2 = 5) : 
  x + 2 * y + 3 * z ≤ Real.sqrt 70 :=
by 
  sorry

end maximum_value_of_x2y3z_l144_144830


namespace emma_deposit_withdraw_ratio_l144_144810

theorem emma_deposit_withdraw_ratio (initial_balance withdrawn new_balance : ℤ) 
  (h1 : initial_balance = 230) 
  (h2 : withdrawn = 60) 
  (h3 : new_balance = 290) 
  (deposited : ℤ) 
  (h_deposit : new_balance = initial_balance - withdrawn + deposited) :
  (deposited / withdrawn = 2) := 
sorry

end emma_deposit_withdraw_ratio_l144_144810


namespace length_of_bridge_correct_l144_144298

noncomputable def length_of_bridge (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_seconds : ℝ) : ℝ :=
  let train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600
  let total_distance : ℝ := train_speed_ms * crossing_time_seconds
  total_distance - train_length

theorem length_of_bridge_correct :
  length_of_bridge 500 42 60 = 200.2 :=
by
  sorry -- Proof of the theorem

end length_of_bridge_correct_l144_144298


namespace total_pennies_l144_144483

variable (C J : ℕ)

def cassandra_pennies : ℕ := 5000
def james_pennies (C : ℕ) : ℕ := C - 276

theorem total_pennies (hC : C = cassandra_pennies) (hJ : J = james_pennies C) :
  C + J = 9724 :=
by
  sorry

end total_pennies_l144_144483


namespace multiplication_correct_l144_144453

theorem multiplication_correct :
  72514 * 99999 = 7250675486 :=
by
  sorry

end multiplication_correct_l144_144453


namespace triangle_max_area_in_quarter_ellipse_l144_144584

theorem triangle_max_area_in_quarter_ellipse (a b c : ℝ) (h : c^2 = a^2 - b^2) :
  ∃ (T_max : ℝ), T_max = b / 2 :=
by sorry

end triangle_max_area_in_quarter_ellipse_l144_144584


namespace area_between_tangent_circles_l144_144283

theorem area_between_tangent_circles (r : ℝ) (h_r : r > 0) :
  let area_trapezoid := 4 * r^2 * Real.sqrt 3
  let area_sector1 := π * r^2 / 3
  let area_sector2 := 3 * π * r^2 / 2
  area_trapezoid - (area_sector1 + area_sector2) = r^2 * (24 * Real.sqrt 3 - 11 * π) / 6 := by
  sorry

end area_between_tangent_circles_l144_144283


namespace cone_height_l144_144002

theorem cone_height (V : ℝ) (π : ℝ) (r h : ℝ) (sqrt2 : ℝ) :
  V = 9720 * π →
  sqrt2 = Real.sqrt 2 →
  h = r * sqrt2 →
  V = (1/3) * π * r^2 * h →
  h = 38.7 :=
by
  intros
  sorry

end cone_height_l144_144002


namespace probability_bernardo_less_than_silvia_l144_144998

open Finset

def bernardo_choices : Finset (Finset (Fin 9)) := (powerset (range 9)).filter (λ s => s.card = 3)
def silvia_choices : Finset (Finset (Fin 10)) := (powerset (range 10)).filter (λ s => s.card = 3)

-- Define the event that Bernardo's number is less than Silvia's number
def event_b_less_s (b s : Finset ℕ) : Prop := b.to_list.sorted_lt s.to_list

-- Calculate the probability that Bernardo's number is less than Silvia's
def probability_b_less_s : ℚ := 
  (bernardo_choices.card : ℚ) / (silvia_choices.card : ℚ) * 
  ((1 / 2) + (choose (9 : ℚ) 2 / choose (10 : ℚ) 3))

theorem probability_bernardo_less_than_silvia : probability_b_less_s = 14 / 25 := sorry

end probability_bernardo_less_than_silvia_l144_144998


namespace arithmetic_sequence_sum_l144_144235

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ)
  (h_seq : ∀ n, a (n + 1) = a n + d)
  (h_a7 : a 7 = 12) :
  a 3 + a 11 = 24 :=
by
  sorry

end arithmetic_sequence_sum_l144_144235


namespace rate_of_interest_is_8_l144_144986

def principal_B : ℕ := 5000
def time_B : ℕ := 2
def principal_C : ℕ := 3000
def time_C : ℕ := 4
def total_interest : ℕ := 1760

theorem rate_of_interest_is_8 :
  ∃ (R : ℝ), ((principal_B * R * time_B) / 100 + (principal_C * R * time_C) / 100 = total_interest) → R = 8 := 
by
  sorry

end rate_of_interest_is_8_l144_144986


namespace cost_price_correct_l144_144275

open Real

-- Define the cost price of the table
def cost_price (C : ℝ) : ℝ := C

-- Define the marked price
def marked_price (C : ℝ) : ℝ := 1.30 * C

-- Define the discounted price
def discounted_price (C : ℝ) : ℝ := 0.85 * (marked_price C)

-- Define the final price after sales tax
def final_price (C : ℝ) : ℝ := 1.12 * (discounted_price C)

-- Given that the final price is 9522.84
axiom final_price_value : final_price 9522.84 = 1.2376 * 7695

-- Main theorem stating the problem to prove
theorem cost_price_correct (C : ℝ) : final_price C = 9522.84 -> C = 7695 := by
  sorry

end cost_price_correct_l144_144275


namespace percent_increase_l144_144591

theorem percent_increase (original value new_value : ℕ) (h1 : original_value = 20) (h2 : new_value = 25) :
  ((new_value - original_value) / original_value) * 100 = 25 :=
by
  -- Proof omitted
  sorry

end percent_increase_l144_144591


namespace inradius_length_l144_144412

noncomputable def inradius (BC AB AC IC : ℝ) (r : ℝ) : Prop :=
  ∀ (r : ℝ), ((BC = 40) ∧ (AB = AC) ∧ (IC = 24)) →
    r = 4 * Real.sqrt 11

theorem inradius_length (BC AB AC IC : ℝ) (r : ℝ) :
  (BC = 40) ∧ (AB = AC) ∧ (IC = 24) →
  r = 4 * Real.sqrt 11 := 
by
  sorry

end inradius_length_l144_144412


namespace joe_paint_usage_l144_144142

theorem joe_paint_usage :
  ∀ (total_paint initial_remaining_paint final_remaining_paint paint_first_week paint_second_week total_used : ℕ),
  total_paint = 360 →
  initial_remaining_paint = total_paint - paint_first_week →
  final_remaining_paint = initial_remaining_paint - paint_second_week →
  paint_first_week = (2 * total_paint) / 3 →
  paint_second_week = (1 * initial_remaining_paint) / 5 →
  total_used = paint_first_week + paint_second_week →
  total_used = 264 :=
by
  sorry

end joe_paint_usage_l144_144142


namespace t_shirts_per_package_l144_144875

theorem t_shirts_per_package (total_tshirts : ℕ) (packages : ℕ) (tshirts_per_package : ℕ) :
  total_tshirts = 70 → packages = 14 → tshirts_per_package = total_tshirts / packages → tshirts_per_package = 5 :=
by
  sorry

end t_shirts_per_package_l144_144875


namespace find_a_and_b_l144_144205

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b

theorem find_a_and_b (a b : ℝ) (h_a : a < 0) (h_max : a + b = 3) (h_min : -a + b = -1) : a = -2 ∧ b = 1 :=
by
  sorry

end find_a_and_b_l144_144205


namespace arithmetic_sqrt_9_l144_144929

theorem arithmetic_sqrt_9 : real.sqrt 9 = 3 :=
by
  sorry

end arithmetic_sqrt_9_l144_144929


namespace abs_linear_combination_l144_144180

theorem abs_linear_combination (a b : ℝ) :
  (∀ x y : ℝ, |a * x + b * y| + |b * x + a * y| = |x| + |y|) →
  (a = 1 ∧ b = 0) ∨ (a = 0 ∧ b = 1) ∨ (a = 0 ∧ b = -1) ∨ (a = -1 ∧ b = 0) :=
by {
  sorry
}

end abs_linear_combination_l144_144180


namespace additional_people_needed_l144_144175

theorem additional_people_needed
  (initial_people : ℕ) (initial_time : ℕ) (new_time : ℕ)
  (h_initial : initial_people * initial_time = 24)
  (h_time : new_time = 2)
  (h_initial_people : initial_people = 8)
  (h_initial_time : initial_time = 3) :
  (24 / new_time) - initial_people = 4 :=
by
  sorry

end additional_people_needed_l144_144175


namespace total_trees_l144_144024

-- Definitions based on the conditions
def ava_trees : ℕ := 9
def lily_trees : ℕ := ava_trees - 3

-- Theorem stating the total number of apple trees planted by Ava and Lily
theorem total_trees : ava_trees + lily_trees = 15 := by
  -- We skip the proof for now
  sorry

end total_trees_l144_144024


namespace three_digit_number_count_l144_144224

def total_three_digit_numbers : ℕ := 900

def count_ABA : ℕ := 9 * 9  -- 81

def count_ABC : ℕ := 9 * 9 * 8  -- 648

def valid_three_digit_numbers : ℕ := total_three_digit_numbers - (count_ABA + count_ABC)

theorem three_digit_number_count :
  valid_three_digit_numbers = 171 := by
  sorry

end three_digit_number_count_l144_144224


namespace zach_needs_more_tickets_l144_144458

theorem zach_needs_more_tickets {ferris_wheel_tickets roller_coaster_tickets log_ride_tickets zach_tickets : ℕ} :
  ferris_wheel_tickets = 2 ∧
  roller_coaster_tickets = 7 ∧
  log_ride_tickets = 1 ∧
  zach_tickets = 1 →
  (ferris_wheel_tickets + roller_coaster_tickets + log_ride_tickets - zach_tickets = 9) :=
by
  intro h
  sorry

end zach_needs_more_tickets_l144_144458


namespace find_width_l144_144952

namespace RectangleProblem

variables {w l : ℝ}

-- Conditions
def length_is_three_times_width (w l : ℝ) : Prop := l = 3 * w
def sum_of_length_and_width_equals_three_times_area (w l : ℝ) : Prop := l + w = 3 * (l * w)

-- Theorem statement
theorem find_width (w l : ℝ) (h1 : length_is_three_times_width w l) (h2 : sum_of_length_and_width_equals_three_times_area w l) :
  w = 4 / 9 :=
sorry

end RectangleProblem

end find_width_l144_144952


namespace find_discriminant_l144_144363

variables {a b c : ℝ}
variables (P : ℝ → ℝ)
def is_quadratic_polynomial (P : ℝ → ℝ) : Prop := ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, P x = a * x^2 + b * x + c)

theorem find_discriminant (h1 : is_quadratic_polynomial P)
  (h2 : ∃ x, P x = x - 2)
  (h3 : ∃ y, P y = 1 - y / 2)
  : ∃ D, D = -1/2 := 
sorry

end find_discriminant_l144_144363


namespace roots_equal_condition_l144_144639

theorem roots_equal_condition (a c : ℝ) (h : a ≠ 0) :
    (∀ x1 x2, (a * x1 * x1 + 4 * a * x1 + c = 0) ∧ (a * x2 * x2 + 4 * a * x2 + c = 0) → x1 = x2) ↔ c = 4 * a := 
by
  sorry

end roots_equal_condition_l144_144639


namespace largest_divisor_poly_l144_144133

-- Define the polynomial and the required properties
def poly (n : ℕ) : ℕ := (n+1) * (n+3) * (n+5) * (n+7) * (n+11)

-- Define the conditions and the proof statement
theorem largest_divisor_poly (n : ℕ) (h_even : n % 2 = 0) : ∃ d, d = 15 ∧ ∀ m, m ∣ poly n → m ≤ d :=
by
  sorry

end largest_divisor_poly_l144_144133


namespace marissa_tied_boxes_l144_144096

theorem marissa_tied_boxes 
  (r_total : ℝ) (r_per_box : ℝ) (r_left : ℝ) (h_total : r_total = 4.5)
  (h_per_box : r_per_box = 0.7) (h_left : r_left = 1) :
  (r_total - r_left) / r_per_box = 5 :=
by
  sorry

end marissa_tied_boxes_l144_144096


namespace complement_of_supplement_of_35_degree_l144_144130

def angle : ℝ := 35
def supplement (x : ℝ) : ℝ := 180 - x
def complement (x : ℝ) : ℝ := 90 - x

theorem complement_of_supplement_of_35_degree :
  complement (supplement angle) = -55 := by
  sorry

end complement_of_supplement_of_35_degree_l144_144130


namespace problem_1_system_solution_problem_2_system_solution_l144_144740

theorem problem_1_system_solution (x y : ℝ)
  (h1 : x - 2 * y = 1)
  (h2 : 4 * x + 3 * y = 26) :
  x = 5 ∧ y = 2 :=
sorry

theorem problem_2_system_solution (x y : ℝ)
  (h1 : 2 * x + 3 * y = 3)
  (h2 : 5 * x - 3 * y = 18) :
  x = 3 ∧ y = -1 :=
sorry

end problem_1_system_solution_problem_2_system_solution_l144_144740


namespace cube_surface_area_proof_l144_144472

-- Conditions
def prism_volume : ℕ := 10 * 5 * 20
def cube_volume : ℕ := 1000
def edge_length_of_cube : ℕ := 10
def cube_surface_area (s : ℕ) : ℕ := 6 * s * s

-- Theorem Statement
theorem cube_surface_area_proof : cube_volume = prism_volume → cube_surface_area edge_length_of_cube = 600 := 
by
  intros h
  -- Proof goes here
  sorry

end cube_surface_area_proof_l144_144472


namespace find_a_minus_b_l144_144520

theorem find_a_minus_b (a b : ℚ)
  (h1 : 2 = a + b / 2)
  (h2 : 7 = a - b / 2)
  : a - b = 19 / 2 := 
  sorry

end find_a_minus_b_l144_144520


namespace sum_even_integers_202_to_300_is_12550_l144_144759

-- Definitions
def sum_first_n_even_integers (n : ℕ) : ℕ :=
  n * (n + 1)

def sum_even_integers_in_range (start end_ : ℕ) : ℕ :=
  let n := (end_ - start) / 2 + 1 in
  n * (start + end_) / 2

-- Theorem Statement
theorem sum_even_integers_202_to_300_is_12550 :
  sum_even_integers_in_range 202 300 = 12550 := by
sorry

end sum_even_integers_202_to_300_is_12550_l144_144759


namespace camp_boys_count_l144_144233

/-- The ratio of boys to girls and total number of individuals in the camp including teachers
is given, we prove the number of boys is 26. -/
theorem camp_boys_count 
  (b g t : ℕ) -- b = number of boys, g = number of girls, t = number of teachers
  (h1 : b = 3 * (t - 5))  -- boys count related to some integer "t" minus teachers
  (h2 : g = 4 * (t - 5))  -- girls count related to some integer "t" minus teachers
  (total_individuals : t = 65) : 
  b = 26 :=
by
  have h : 3 * (t - 5) + 4 * (t - 5) + 5 = 65 := sorry
  sorry

end camp_boys_count_l144_144233


namespace m_value_l144_144166

theorem m_value (m : ℝ) (h : (243:ℝ) ^ (1/3) = 3 ^ m) : m = 5 / 3 :=
sorry

end m_value_l144_144166


namespace total_limes_l144_144160

-- Define the number of limes picked by Alyssa, Mike, and Tom's plums
def alyssa_limes : ℕ := 25
def mike_limes : ℕ := 32
def tom_plums : ℕ := 12

theorem total_limes : alyssa_limes + mike_limes = 57 := by
  -- The proof is omitted as per the instruction
  sorry

end total_limes_l144_144160


namespace integral_of_2x_minus_1_over_x_sq_l144_144802

theorem integral_of_2x_minus_1_over_x_sq:
  ∫ x in (1 : ℝ)..3, (2 * x - (1 / x^2)) = 26 / 3 := by
  sorry

end integral_of_2x_minus_1_over_x_sq_l144_144802


namespace simplify_expression_l144_144560

noncomputable def expr1 := (Real.sqrt 462) / (Real.sqrt 330)
noncomputable def expr2 := (Real.sqrt 245) / (Real.sqrt 175)
noncomputable def expr_simplified := (12 * Real.sqrt 35) / 25

theorem simplify_expression :
  expr1 + expr2 = expr_simplified :=
sorry

end simplify_expression_l144_144560


namespace find_number_l144_144989

theorem find_number (x : ℝ) (h : 2 * x - 2.6 * 4 = 10) : x = 10.2 :=
sorry

end find_number_l144_144989


namespace quadratic_polynomial_discriminant_l144_144367

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₁ : a ≠ 0)
  (h₂ : ∃ x : ℝ, a * x^2 + b * x + c = x - 2 ∧ (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h₃ : ∃ x : ℝ, a * x^2 + b * x + c = 1 - x / 2 ∧ (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l144_144367


namespace f_value_at_3_l144_144508

def f (x : ℝ) := 2 * (x + 1) + 1

theorem f_value_at_3 : f 3 = 9 :=
by sorry

end f_value_at_3_l144_144508


namespace digit_in_base_l144_144640

theorem digit_in_base (t : ℕ) (h1 : t ≤ 9) (h2 : 5 * 7 + t = t * 9 + 3) : t = 4 := by
  sorry

end digit_in_base_l144_144640


namespace integer_to_sixth_power_l144_144683

theorem integer_to_sixth_power (a b : ℕ) (h : 3^a * 3^b = 3^(a + b)) (ha : a = 12) (hb : b = 18) : 
  ∃ x : ℕ, x = 243 ∧ x^6 = 3^(a + b) :=
by
  sorry

end integer_to_sixth_power_l144_144683


namespace power_function_at_100_l144_144217

-- Given a power function f(x) = x^α that passes through the point (9, 3),
-- show that f(100) = 10.

theorem power_function_at_100 (α : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x ^ α)
  (h2 : f 9 = 3) : f 100 = 10 :=
sorry

end power_function_at_100_l144_144217


namespace pq_inequality_l144_144651

theorem pq_inequality (p : ℝ) (q : ℝ) (hp : 0 ≤ p) (hp2 : p < 2) (hq : q > 0) :
  4 * (p * q^2 + 2 * p^2 * q + 4 * q^2 + 5 * p * q) / (p + q) > 3 * p^2 * q :=
by {
  sorry
}

end pq_inequality_l144_144651


namespace series_sum_eq_l144_144805

noncomputable def sum_series : ℝ :=
∑' n : ℕ, if h : n > 0 then (4 * n + 3) / ((4 * n)^2 * (4 * n + 4)^2) else 0

theorem series_sum_eq :
  sum_series = 1 / 256 := by
  sorry

end series_sum_eq_l144_144805


namespace find_y_l144_144070

variable {L B y : ℝ}

theorem find_y (h1 : 2 * ((L + y) + (B + y)) - 2 * (L + B) = 16) : y = 4 :=
by
  sorry

end find_y_l144_144070


namespace pentadecagon_diagonals_l144_144190

def number_of_diagonals (n : Nat) : Nat :=
  (n * (n - 3)) / 2

theorem pentadecagon_diagonals :
  number_of_diagonals 15 = 90 := 
by
  sorry

end pentadecagon_diagonals_l144_144190


namespace average_branches_per_foot_l144_144636

theorem average_branches_per_foot :
  let b1 := 200
  let h1 := 50
  let b2 := 180
  let h2 := 40
  let b3 := 180
  let h3 := 60
  let b4 := 153
  let h4 := 34
  (b1 / h1 + b2 / h2 + b3 / h3 + b4 / h4) / 4 = 4 := by
  sorry

end average_branches_per_foot_l144_144636


namespace find_expression_value_l144_144319

def g (p q r s : ℝ) (x : ℝ) : ℝ := p * x^3 + q * x^2 + r * x + s

theorem find_expression_value (p q r s : ℝ) (h1 : g p q r s (-1) = 2) (h2 : g p q r s (-2) = -1) (h3 : g p q r s (1) = -2) :
  9 * p - 3 * q + 3 * r - s = -2 :=
by
  sorry

end find_expression_value_l144_144319


namespace person_income_l144_144941

theorem person_income 
    (income expenditure savings : ℕ) 
    (h1 : income = 3 * (income / 3)) 
    (h2 : expenditure = 2 * (income / 3)) 
    (h3 : savings = 7000) 
    (h4 : income = expenditure + savings) : 
    income = 21000 := 
by 
  sorry

end person_income_l144_144941


namespace smallest_number_of_ducks_l144_144258

theorem smallest_number_of_ducks (n_ducks n_cranes : ℕ) (h1 : n_ducks = n_cranes) : 
  ∃ n, n_ducks = n ∧ n_cranes = n ∧ n = Nat.lcm 13 17 := by
  use 221
  sorry

end smallest_number_of_ducks_l144_144258


namespace infinitely_many_m_l144_144595

theorem infinitely_many_m (k l : ℕ) (hk : 0 < k) (hl : 0 < l) :
  ∃ᶠ m in Filter.atTop, m ≥ k ∧ Nat.gcd (Nat.choose m k) l = 1 :=
sorry

end infinitely_many_m_l144_144595


namespace joan_took_marbles_l144_144111

-- Each condition is used as a definition.
def original_marbles : ℕ := 86
def remaining_marbles : ℕ := 61

-- The theorem states that the number of marbles Joan took equals 25.
theorem joan_took_marbles : (original_marbles - remaining_marbles) = 25 := by
  sorry    -- Add sorry to skip the proof.

end joan_took_marbles_l144_144111


namespace projection_matrix_l144_144187

theorem projection_matrix
  (x y : ℝ) :
  let v := ![3, 4]
  let proj_v := (v ⬝ ![x, y]) / (v ⬝ v) • v
  let proj_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
    ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]] in
  proj_v = proj_matrix.mul_vec ![x, y] :=
by
  let v := ![3, 4]
  let proj_v := (v ⬝ ![x, y]) / (v ⬝ v) • v
  let proj_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]]
  sorry

end projection_matrix_l144_144187


namespace intersection_M_N_l144_144069

def M : Set ℝ := {x | (x - 1) * (x - 4) = 0}
def N : Set ℝ := {x | (x + 1) * (x - 3) < 0}

theorem intersection_M_N :
  M ∩ N = {1} :=
sorry

end intersection_M_N_l144_144069


namespace perfect_square_pairs_l144_144041

-- Definition of a perfect square
def is_perfect_square (k : ℕ) : Prop :=
∃ (n : ℕ), n * n = k

-- Main theorem statement
theorem perfect_square_pairs (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  is_perfect_square ((2^m - 1) * (2^n - 1)) ↔ (m = n) ∨ (m = 3 ∧ n = 6) ∨ (m = 6 ∧ n = 3) :=
sorry

end perfect_square_pairs_l144_144041


namespace total_reading_materials_l144_144773

theorem total_reading_materials (magazines newspapers : ℕ) (h1 : magazines = 425) (h2 : newspapers = 275) : 
  magazines + newspapers = 700 :=
by 
  sorry

end total_reading_materials_l144_144773


namespace sin_alpha_second_quadrant_l144_144210

theorem sin_alpha_second_quadrant (α : ℝ) (h_α_quad_2 : π / 2 < α ∧ α < π) (h_cos_α : Real.cos α = -1 / 3) : Real.sin α = 2 * Real.sqrt 2 / 3 := 
sorry

end sin_alpha_second_quadrant_l144_144210


namespace total_donation_l144_144485

-- Definitions
def cassandra_pennies : ℕ := 5000
def james_deficit : ℕ := 276
def james_pennies : ℕ := cassandra_pennies - james_deficit

-- Theorem to prove the total donation
theorem total_donation : cassandra_pennies + james_pennies = 9724 :=
by
  -- Proof is omitted
  sorry

end total_donation_l144_144485


namespace exists_irrationals_pow_rational_l144_144886

-- Conditions: 
def sqrt2_irrational : Prop := irrational (real.sqrt 2)

def sqrt2_pow_rational_or_irrational : Prop :=
  (rational ((real.sqrt 2) ^ (real.sqrt 2)) ∨ irrational ((real.sqrt 2) ^ (real.sqrt 2)))

-- Theorem Statement: 
theorem exists_irrationals_pow_rational : sqrt2_irrational ∧ sqrt2_pow_rational_or_irrational →
  ∃ a b : ℝ, irrational a ∧ irrational b ∧ rational (a ^ b) :=
sorry

end exists_irrationals_pow_rational_l144_144886


namespace ratio_odd_even_divisors_l144_144250

def sum_of_divisors (n : ℕ) : ℕ := sorry -- This should be implemented as a function that calculates sum of divisors

def sum_of_odd_divisors (n : ℕ) : ℕ := sorry -- This should be implemented as a function that calculates sum of odd divisors

def sum_of_even_divisors (n : ℕ) : ℕ := sorry -- This should be implemented as a function that calculates sum of even divisors

theorem ratio_odd_even_divisors (M : ℕ) (h : M = 36 * 36 * 98 * 210) :
  sum_of_odd_divisors M / sum_of_even_divisors M = 1 / 60 :=
by {
  sorry
}

end ratio_odd_even_divisors_l144_144250


namespace B_power_identity_l144_144718

open Matrix

variables {R : Type*} [CommRing R] {n : Type*} [Fintype n] [DecidableEq n]

def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 2], ![3, -1]]

theorem B_power_identity :
  B^4 = 0 • B + 49 • (1 : Matrix (Fin 2) (Fin 2) ℤ) :=
by
  sorry -- Proof goes here

end B_power_identity_l144_144718


namespace sum_exponents_binary_3400_l144_144395

theorem sum_exponents_binary_3400 : 
  ∃ (a b c d e : ℕ), 
    3400 = 2^a + 2^b + 2^c + 2^d + 2^e ∧ 
    a > b ∧ b > c ∧ c > d ∧ d > e ∧ 
    a + b + c + d + e = 38 :=
sorry

end sum_exponents_binary_3400_l144_144395


namespace John_distance_proof_l144_144080

def initial_running_time : ℝ := 8
def increase_percentage : ℝ := 0.75
def initial_speed : ℝ := 8
def speed_increase : ℝ := 4

theorem John_distance_proof : 
  (initial_running_time + initial_running_time * increase_percentage) * (initial_speed + speed_increase) = 168 := 
by
  -- Proof can be completed here
  sorry

end John_distance_proof_l144_144080


namespace power_expression_l144_144679

theorem power_expression (a b : ℕ) (h1 : a = 12) (h2 : b = 18) : (3^a * 3^b) = (243^6) :=
by
  let c := 3
  have h3 : a + b = 30 := by simp [h1, h2]
  have h4 : 3^(a + b) = 3^30 := by rw [h3]
  have h5 : 3^30 = 243^6 := by norm_num
  sorry  -- skip other detailed steps

end power_expression_l144_144679


namespace card_collection_problem_l144_144644

theorem card_collection_problem 
  (m : ℕ) 
  (h : (2 * m + 1) / 3 = 56) : 
  m = 84 :=
sorry

end card_collection_problem_l144_144644


namespace range_of_m_l144_144441

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x * Real.log x - m * x^2

def has_two_extreme_points (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f m x₁ = f m x₂ ∧ (∀ x, x = x₁ ∨ x = x₂ ∨ f m x ≤ f m x₁ ∨ f m x ≤ f m x₂)

theorem range_of_m :
  ∀ m : ℝ, has_two_extreme_points (m) ↔ 0 < m ∧ m < 1 / 2 := 
by
  sorry

end range_of_m_l144_144441


namespace diane_age_proof_l144_144765

noncomputable def diane_age (A Al D : ℕ) : Prop :=
  ((A + (30 - D) = 60) ∧ (Al + (30 - D) = 15) ∧ (A + Al = 47)) → (D = 16)

theorem diane_age_proof : ∃ (D : ℕ), ∃ (A Al : ℕ), diane_age A Al D :=
by {
  sorry
}

end diane_age_proof_l144_144765


namespace sections_capacity_l144_144447

theorem sections_capacity (total_people sections : ℕ) 
  (h1 : total_people = 984) 
  (h2 : sections = 4) : 
  total_people / sections = 246 := 
by
  sorry

end sections_capacity_l144_144447


namespace angle_A_is_120_max_sin_B_plus_sin_C_l144_144858

-- Define the measures in degrees using real numbers
variable (a b c R : Real)
variable (A B C : ℝ) (sin cos : ℝ → ℝ)

-- Question 1: Prove A = 120 degrees given the initial condition
theorem angle_A_is_120
  (H1 : 2 * a * (sin A) = (2 * b + c) * (sin B) + (2 * c + b) * (sin C)) :
  A = 120 :=
by
  sorry

-- Question 2: Given the angles sum to 180 degrees and A = 120 degrees, prove the max value of sin B + sin C is 1
theorem max_sin_B_plus_sin_C
  (H2 : A + B + C = 180)
  (H3 : A = 120) :
  (sin B) + (sin C) ≤ 1 :=
by
  sorry

end angle_A_is_120_max_sin_B_plus_sin_C_l144_144858


namespace total_distance_in_land_miles_l144_144330

-- Definitions based on conditions
def speed_one_sail : ℕ := 25
def time_one_sail : ℕ := 4
def distance_one_sail := speed_one_sail * time_one_sail

def speed_two_sails : ℕ := 50
def time_two_sails : ℕ := 4
def distance_two_sails := speed_two_sails * time_two_sails

def conversion_factor : ℕ := 115  -- Note: 1.15 * 100 for simplicity with integers

-- Theorem to prove the total distance in land miles
theorem total_distance_in_land_miles : (distance_one_sail + distance_two_sails) * conversion_factor / 100 = 345 := by
  sorry

end total_distance_in_land_miles_l144_144330


namespace sin_double_angle_of_tangent_l144_144697

theorem sin_double_angle_of_tangent (α : ℝ) (h : Real.tan (π + α) = 2) : Real.sin (2 * α) = 4 / 5 := by
  sorry

end sin_double_angle_of_tangent_l144_144697


namespace problems_on_each_worksheet_l144_144475

-- Define the conditions
def worksheets_total : Nat := 9
def worksheets_graded : Nat := 5
def problems_left : Nat := 16

-- Define the number of remaining worksheets and the problems per worksheet
def remaining_worksheets : Nat := worksheets_total - worksheets_graded
def problems_per_worksheet : Nat := problems_left / remaining_worksheets

-- Prove the number of problems on each worksheet
theorem problems_on_each_worksheet : problems_per_worksheet = 4 :=
by
  sorry

end problems_on_each_worksheet_l144_144475


namespace flyers_left_to_hand_out_l144_144239

-- Definitions for given conditions
def total_flyers : Nat := 1236
def jack_handout : Nat := 120
def rose_handout : Nat := 320

-- Statement of the problem
theorem flyers_left_to_hand_out : total_flyers - (jack_handout + rose_handout) = 796 :=
by
  -- proof goes here
  sorry

end flyers_left_to_hand_out_l144_144239


namespace childSupportOwed_l144_144545

def annualIncomeBeforeRaise : ℕ := 30000
def yearsBeforeRaise : ℕ := 3
def raisePercentage : ℕ := 20
def annualIncomeAfterRaise (incomeBeforeRaise raisePercentage : ℕ) : ℕ :=
  incomeBeforeRaise + (incomeBeforeRaise * raisePercentage / 100)
def yearsAfterRaise : ℕ := 4
def childSupportPercentage : ℕ := 30
def amountPaid : ℕ := 1200

def calculateChildSupport (incomeYears : ℕ → ℕ → ℕ) (supportPercentage : ℕ) (years : ℕ) : ℕ :=
  (incomeYears years supportPercentage) * supportPercentage / 100 * years

def totalChildSupportOwed : ℕ :=
  (calculateChildSupport (λ _ _ => annualIncomeBeforeRaise) childSupportPercentage yearsBeforeRaise) +
  (calculateChildSupport (λ _ _ => annualIncomeAfterRaise annualIncomeBeforeRaise raisePercentage) childSupportPercentage yearsAfterRaise)

theorem childSupportOwed : totalChildSupportOwed - amountPaid = 69000 :=
by trivial

end childSupportOwed_l144_144545


namespace age_difference_64_l144_144746

variables (Patrick Michael Monica : ℕ)
axiom age_ratio_1 : ∃ (x : ℕ), Patrick = 3 * x ∧ Michael = 5 * x
axiom age_ratio_2 : ∃ (y : ℕ), Michael = 3 * y ∧ Monica = 5 * y
axiom age_sum : Patrick + Michael + Monica = 196

theorem age_difference_64 : Monica - Patrick = 64 :=
by {
  sorry
}

end age_difference_64_l144_144746


namespace problem_l144_144219

noncomputable def p : Prop :=
  ∀ x : ℝ, (0 < x) → Real.exp x > 1 + x

def q (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (-x) + 2 = -(f x + 2)) → ∀ x : ℝ, f (-x) = f x - 4

theorem problem (f : ℝ → ℝ) : p ∨ q f :=
  sorry

end problem_l144_144219


namespace binomial_distrib_not_equiv_binom_expansion_l144_144434

theorem binomial_distrib_not_equiv_binom_expansion (a b : ℝ) (n : ℕ) (p : ℝ) (h1: a = p) (h2: b = 1 - p):
    ¬ (∃ k : ℕ, p ^ k * (1 - p) ^ (n - k) = (a + b) ^ n) := sorry

end binomial_distrib_not_equiv_binom_expansion_l144_144434


namespace probability_at_least_one_boy_one_girl_l144_144621

noncomputable def probability_one_boy_one_girl : ℚ :=
  1 - (1 / 16) - (1 / 16)

theorem probability_at_least_one_boy_one_girl :
  probability_one_boy_one_girl = 7 / 8 := by
  sorry

end probability_at_least_one_boy_one_girl_l144_144621


namespace fruit_costs_l144_144600

theorem fruit_costs (
    A O B : ℝ
) (h1 : O = A + 0.28)
  (h2 : B = A - 0.15)
  (h3 : 3 * A + 7 * O + 5 * B = 7.84) :
  A = 0.442 ∧ O = 0.722 ∧ B = 0.292 :=
by
  -- The proof is omitted here; replacing with sorry for now
  sorry

end fruit_costs_l144_144600


namespace jack_initial_checked_plates_l144_144715

-- Define Jack's initial and resultant plate counts
variable (C : Nat)
variable (initial_flower_plates : Nat := 4)
variable (broken_flower_plates : Nat := 1)
variable (polka_dotted_plates := 2 * C)
variable (total_plates : Nat := 27)

-- Statement of the problem
theorem jack_initial_checked_plates (h_eq : 3 + C + 2 * C = total_plates) : C = 8 :=
by
  sorry

end jack_initial_checked_plates_l144_144715


namespace david_english_marks_l144_144494

def david_marks (math physics chemistry biology avg : ℕ) : ℕ :=
  avg * 5 - (math + physics + chemistry + biology)

theorem david_english_marks :
  let math := 95
  let physics := 82
  let chemistry := 97
  let biology := 95
  let avg := 93
  david_marks math physics chemistry biology avg = 96 :=
by
  -- Proof is skipped
  sorry

end david_english_marks_l144_144494


namespace discriminant_of_P_l144_144352

theorem discriminant_of_P (a b c : ℚ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1/2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1/2 := 
begin
  -- Proof omitted for brevity
  sorry
end

end discriminant_of_P_l144_144352


namespace john_run_distance_l144_144079

theorem john_run_distance :
  ∀ (initial_hours : ℝ) (increase_time_percent : ℝ) (initial_speed : ℝ) (increase_speed : ℝ),
  initial_hours = 8 → increase_time_percent = 0.75 → initial_speed = 8 → increase_speed = 4 →
  let increased_hours := initial_hours * increase_time_percent,
      total_hours := initial_hours + increased_hours,
      new_speed := initial_speed + increase_speed,
      distance := total_hours * new_speed in
  distance = 168 := 
by
  intros initial_hours increase_time_percent initial_speed increase_speed h_hours h_time h_speed h_increase
  let increased_hours := initial_hours * increase_time_percent
  let total_hours := initial_hours + increased_hours
  let new_speed := initial_speed + increase_speed
  let distance := total_hours * new_speed
  sorry

end john_run_distance_l144_144079


namespace exponent_product_to_sixth_power_l144_144687

theorem exponent_product_to_sixth_power :
  ∃ n : ℤ, 3^(12) * 3^(18) = n^6 ∧ n = 243 :=
by
  use 243
  sorry

end exponent_product_to_sixth_power_l144_144687


namespace geometric_series_common_ratio_l144_144819

theorem geometric_series_common_ratio (r : ℚ) : 
  (∃ (a : ℚ), a = 4 / 7 ∧ a * r = 16 / 21) → r = 4 / 3 :=
by
  sorry

end geometric_series_common_ratio_l144_144819


namespace find_B_plus_C_l144_144516

theorem find_B_plus_C 
(A B C : ℕ)
(h1 : A ≠ B)
(h2 : B ≠ C)
(h3 : C ≠ A)
(h4 : A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0)
(h5 : A < 5 ∧ B < 5 ∧ C < 5)
(h6 : 25 * A + 5 * B + C + 25 * B + 5 * C + A + 25 * C + 5 * A + B = 125 * A + 25 * A + 5 * A) : 
B + C = 4 * A := by
  sorry

end find_B_plus_C_l144_144516


namespace no_such_function_exists_l144_144107

open Classical

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (f 0 > 0) ∧ (∀ (x y : ℝ), f (x + y) ≥ f x + y * f (f x)) :=
sorry

end no_such_function_exists_l144_144107


namespace square_difference_division_l144_144803

theorem square_difference_division (a b : ℕ) (h₁ : a = 121) (h₂ : b = 112) :
  (a^2 - b^2) / 9 = 233 :=
by
  sorry

end square_difference_division_l144_144803


namespace pencil_case_cost_l144_144877

-- Defining given conditions
def initial_amount : ℕ := 10
def toy_truck_cost : ℕ := 3
def remaining_amount : ℕ := 5
def total_spent : ℕ := initial_amount - remaining_amount

-- Proof statement
theorem pencil_case_cost : total_spent - toy_truck_cost = 2 :=
by
  sorry

end pencil_case_cost_l144_144877


namespace solve_for_y_l144_144888

theorem solve_for_y (y : ℝ) (h : y ≠ 2) :
  (7 * y / (y - 2) - 4 / (y - 2) = 3 / (y - 2)) → y = 1 :=
by
  intro h_eq
  sorry

end solve_for_y_l144_144888


namespace average_marks_combined_l144_144474

theorem average_marks_combined (P C M B E : ℕ) (h : P + C + M + B + E = P + 280) : 
  (C + M + B + E) / 4 = 70 :=
by 
  sorry

end average_marks_combined_l144_144474


namespace gcd_of_factorials_l144_144182

theorem gcd_of_factorials (n m : ℕ) (h1 : n = 7) (h2 : m = 8) :
  Nat.gcd (n.factorial) (m.factorial) = 5040 := by
  have fact7 : 7.factorial = 5040 := by
    norm_num
  rw [h1, h2]
  rw [Nat.factorial_succ]
  rw [<-mul_comm 8 7.factorial, fact7]
  exact Nat.gcd_mul_left 8 5040 1

end gcd_of_factorials_l144_144182


namespace many_people_sharing_car_l144_144325

theorem many_people_sharing_car (x y : ℤ) 
  (h1 : 3 * (y - 2) = x) 
  (h2 : 2 * y + 9 = x) : 
  3 * (y - 2) = 2 * y + 9 := 
by
  -- by assumption h1 and h2, we already have the setup, refute/validate consistency
  sorry

end many_people_sharing_car_l144_144325


namespace job_candidates_excel_nights_l144_144981

theorem job_candidates_excel_nights (hasExcel : ℝ) (dayShift : ℝ) 
    (h1 : hasExcel = 0.2) (h2 : dayShift = 0.7) : 
    (1 - dayShift) * hasExcel = 0.06 :=
by
  sorry

end job_candidates_excel_nights_l144_144981


namespace books_sold_l144_144257

theorem books_sold (original_books : ℕ) (remaining_books : ℕ) (sold_books : ℕ) 
  (h1 : original_books = 51) 
  (h2 : remaining_books = 6) 
  (h3 : sold_books = original_books - remaining_books) : 
  sold_books = 45 :=
by 
  sorry

end books_sold_l144_144257


namespace find_square_value_l144_144470

theorem find_square_value (y : ℝ) (h : 4 * y^2 + 3 = 7 * y + 12) : (8 * y - 4)^2 = 202 := 
by
  sorry

end find_square_value_l144_144470


namespace saplings_problem_l144_144737

theorem saplings_problem (x : ℕ) :
  (∃ n : ℕ, 5 * x + 3 = n ∧ 6 * x - 4 = n) ↔ 5 * x + 3 = 6 * x - 4 :=
by
  sorry

end saplings_problem_l144_144737


namespace elise_initial_money_l144_144037

theorem elise_initial_money :
  ∃ (X : ℤ), X + 13 - 2 - 18 = 1 ∧ X = 8 :=
by
  sorry

end elise_initial_money_l144_144037


namespace integer_to_sixth_power_l144_144682

theorem integer_to_sixth_power (a b : ℕ) (h : 3^a * 3^b = 3^(a + b)) (ha : a = 12) (hb : b = 18) : 
  ∃ x : ℕ, x = 243 ∧ x^6 = 3^(a + b) :=
by
  sorry

end integer_to_sixth_power_l144_144682


namespace solve_log_sin_eq_l144_144563

noncomputable def log_base (b : ℝ) (a : ℝ) : ℝ :=
  Real.log a / Real.log b

theorem solve_log_sin_eq :
  ∀ x : ℝ, 
  (0 < Real.sin x ∧ Real.sin x < 1) →
  log_base (Real.sin x) 4 * log_base (Real.sin x ^ 2) 2 = 4 →
  ∃ k : ℤ, x = (-1)^k * (Real.pi / 4) + Real.pi * k := 
by
  sorry

end solve_log_sin_eq_l144_144563


namespace proof_1_proof_2_l144_144889

noncomputable def problem_1 (x : ℝ) : Prop :=
  (3 * x - 2) / (x - 1) > 1 → x > 1

noncomputable def problem_2 (x a : ℝ) : Prop :=
  if a = 0 then False
  else if a > 0 then -a < x ∧ x < 2 * a
  else if a < 0 then 2 * a < x ∧ x < -a
  else False

-- Sorry to skip the proofs
theorem proof_1 (x : ℝ) (h : problem_1 x) : x > 1 :=
  sorry

theorem proof_2 (x a : ℝ) (h : x * x - a * x - 2 * a * a < 0) : problem_2 x a :=
  sorry

end proof_1_proof_2_l144_144889


namespace prove_total_number_of_apples_l144_144167

def avg_price (light_price heavy_price : ℝ) (light_proportion heavy_proportion : ℝ) : ℝ :=
  light_proportion * light_price + heavy_proportion * heavy_price

def weighted_avg_price (prices proportions : List ℝ) : ℝ :=
  (List.map (λ ⟨p, prop⟩ => p * prop) (List.zip prices proportions)).sum

noncomputable def total_num_apples (total_earnings weighted_price : ℝ) : ℝ :=
  total_earnings / weighted_price

theorem prove_total_number_of_apples : 
  let light_proportion := 0.6
  let heavy_proportion := 0.4
  let prices := [avg_price 0.4 0.6 light_proportion heavy_proportion, 
                 avg_price 0.1 0.15 light_proportion heavy_proportion,
                 avg_price 0.25 0.35 light_proportion heavy_proportion,
                 avg_price 0.15 0.25 light_proportion heavy_proportion,
                 avg_price 0.2 0.3 light_proportion heavy_proportion,
                 avg_price 0.05 0.1 light_proportion heavy_proportion]
  let proportions := [0.4, 0.2, 0.15, 0.1, 0.1, 0.05]
  let weighted_avg := weighted_avg_price prices proportions
  total_num_apples 120 weighted_avg = 392 :=
by
  sorry

end prove_total_number_of_apples_l144_144167


namespace lcm_36_125_l144_144043

-- Define the prime factorizations
def factorization_36 : List (ℕ × ℕ) := [(2, 2), (3, 2)]
def factorization_125 : List (ℕ × ℕ) := [(5, 3)]

-- Least common multiple definition
noncomputable def my_lcm (a b : ℕ) : ℕ :=
  a * b / (Nat.gcd a b)

-- Theorem to prove
theorem lcm_36_125 : my_lcm 36 125 = 4500 :=
by
  sorry

end lcm_36_125_l144_144043


namespace player_holds_seven_black_cards_l144_144603

theorem player_holds_seven_black_cards
    (total_cards : ℕ := 13)
    (num_red_cards : ℕ := 6)
    (S D H C : ℕ)
    (h1 : D = 2 * S)
    (h2 : H = 2 * D)
    (h3 : C = 6)
    (h4 : S + D + H + C = total_cards) :
    S + C = 7 := 
by
  sorry

end player_holds_seven_black_cards_l144_144603


namespace smallest_six_digit_number_divisible_by_25_35_45_15_l144_144295

theorem smallest_six_digit_number_divisible_by_25_35_45_15 :
  ∃ n : ℕ, 100000 ≤ n ∧ n < 1000000 ∧ 
           (25 ∣ n) ∧ 
           (35 ∣ n) ∧ 
           (45 ∣ n) ∧ 
           (15 ∣ n) ∧ 
           (∀ m : ℕ, 100000 ≤ m ∧ m < 1000000 ∧ 
                     (25 ∣ m) ∧ 
                     (35 ∣ m) ∧ 
                     (45 ∣ m) ∧ 
                     (15 ∣ m) → n ≤ m) :=
by
  use 100800
  sorry

end smallest_six_digit_number_divisible_by_25_35_45_15_l144_144295


namespace jelly_sold_l144_144009

theorem jelly_sold (G S R P : ℕ) (h1 : G = 2 * S) (h2 : R = 2 * P) (h3 : R = G / 3) (h4 : P = 6) : S = 18 := by
  sorry

end jelly_sold_l144_144009


namespace polynomial_discriminant_l144_144372

theorem polynomial_discriminant (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
by
  sorry

end polynomial_discriminant_l144_144372


namespace matrix_equation_l144_144870

def M : Matrix (Fin 2) (Fin 2) ℤ := ![![4, 5], ![-6, -2]]
def p : ℤ := 2
def q : ℤ := -18

theorem matrix_equation :
  M * M = p • M + q • (1 : Matrix (Fin 2) (Fin 2) ℤ) :=
by
  sorry

end matrix_equation_l144_144870


namespace ned_weekly_revenue_l144_144731

-- Conditions
def normal_mouse_cost : ℕ := 120
def percentage_increase : ℕ := 30
def mice_sold_per_day : ℕ := 25
def days_store_is_open_per_week : ℕ := 4

-- Calculate cost of a left-handed mouse
def left_handed_mouse_cost : ℕ := normal_mouse_cost + (normal_mouse_cost * percentage_increase / 100)

-- Calculate daily revenue
def daily_revenue : ℕ := mice_sold_per_day * left_handed_mouse_cost

-- Calculate weekly revenue
def weekly_revenue : ℕ := daily_revenue * days_store_is_open_per_week

-- Theorem to prove
theorem ned_weekly_revenue : weekly_revenue = 15600 := 
by 
  sorry

end ned_weekly_revenue_l144_144731


namespace union_of_S_and_T_l144_144838

-- Definitions of the sets S and T
def S : Set ℝ := { y | ∃ x : ℝ, y = Real.exp x - 2 }
def T : Set ℝ := { x | -4 ≤ x ∧ x ≤ 1 }

-- Lean proof problem statement
theorem union_of_S_and_T : (S ∪ T) = { y | -4 ≤ y } :=
by
  sorry

end union_of_S_and_T_l144_144838


namespace coefficient_x2_in_expansion_l144_144402

theorem coefficient_x2_in_expansion (a b : ℤ) (n k : ℕ) (h_a : a = 2) (h_b : b = x) (h_n : n = 4) (h_k : k = 2) :
  (Nat.choose n k * a^(n - k)) = 24 := by
  sorry

end coefficient_x2_in_expansion_l144_144402


namespace move_line_up_l144_144399

theorem move_line_up (x : ℝ) :
  let y_initial := 4 * x - 1
  let y_moved := y_initial + 2
  y_moved = 4 * x + 1 :=
by
  let y_initial := 4 * x - 1
  let y_moved := y_initial + 2
  show y_moved = 4 * x + 1
  sorry

end move_line_up_l144_144399


namespace initial_sand_in_bucket_A_l144_144027

theorem initial_sand_in_bucket_A (C : ℝ) : 
  let bucketB_capacity := C / 2
  let sand_in_B := (3 / 8) * bucketB_capacity
  let after_pour := (7 / 16) * C
  let x := after_pour - sand_in_B
  x / C = 1 / 4 := by
  let bucketB_capacity := C / 2
  let sand_in_B := (3 / 8) * bucketB_capacity
  let after_pour := (7 / 16) * C
  let x := after_pour - sand_in_B
  show x / C = 1 / 4
  sorry

end initial_sand_in_bucket_A_l144_144027


namespace james_muffins_correct_l144_144328

-- Arthur baked 115 muffins
def arthur_muffins : ℕ := 115

-- James baked 12 times as many muffins as Arthur
def james_multiplier : ℕ := 12

-- The number of muffins James baked
def james_muffins : ℕ := arthur_muffins * james_multiplier

-- The expected result
def expected_james_muffins : ℕ := 1380

-- The statement we want to prove
theorem james_muffins_correct : james_muffins = expected_james_muffins := by
  sorry

end james_muffins_correct_l144_144328


namespace quadratic_discriminant_l144_144358

variable {a b c : ℝ}
variable (h₁ : a ≠ 0)
variable (h₂ : (b - 1)^2 - 4 * a * (c + 2) = 0)
variable (h₃ : (b + 1/2)^2 - 4 * a * (c - 1) = 0)

theorem quadratic_discriminant : b^2 - 4 * a * c = -1 / 2 := 
by
  have h₁' : (b - 1)^2 - 4 * a * (c + 2) = 0 := h₂
  have h₂' : (b + 1/2)^2 - 4 * a * (c - 1) = 0 := h₃
  sorry

end quadratic_discriminant_l144_144358


namespace frogs_needing_new_pond_l144_144007

theorem frogs_needing_new_pond : 
    (initial_frogs tadpoles_survival_fraction pond_capacity : ℕ) 
    (h1 : initial_frogs = 5) 
    (h2 : tadpoles_survival_fraction = 2 / 3) 
    (h3 : pond_capacity = 8) 
    : (initial_frogs + tadpoles_survival_fraction * 3 * initial_frogs - pond_capacity = 7) :=
by
    sorry

end frogs_needing_new_pond_l144_144007


namespace climbing_stairs_l144_144013

noncomputable def total_methods_to_climb_stairs : ℕ :=
  (Nat.choose 8 5) + (Nat.choose 8 6) + (Nat.choose 8 7) + 1

theorem climbing_stairs (n : ℕ := 9) (min_steps : ℕ := 6) (max_steps : ℕ := 9)
  (H1 : min_steps ≤ n)
  (H2 : n ≤ max_steps)
  : total_methods_to_climb_stairs = 93 := by
  sorry

end climbing_stairs_l144_144013


namespace negation_of_exists_l144_144946

theorem negation_of_exists :
  ¬ (∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ ∀ x : ℝ, x^2 - 2*x + 1 ≥ 0 :=
by
  sorry

end negation_of_exists_l144_144946


namespace sum_of_squares_l144_144229

theorem sum_of_squares :
  ∃ p q r s t u : ℤ, (∀ x : ℤ, 729 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) ∧ 
    (p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 8210) :=
sorry

end sum_of_squares_l144_144229


namespace common_ratio_of_geometric_series_l144_144814

theorem common_ratio_of_geometric_series (a b : ℚ) (h1 : a = 4 / 7) (h2 : b = 16 / 21) :
  b / a = 4 / 3 :=
by
  sorry

end common_ratio_of_geometric_series_l144_144814


namespace extra_people_needed_l144_144173

theorem extra_people_needed 
  (initial_people : ℕ) 
  (initial_time : ℕ) 
  (final_time : ℕ) 
  (work_done : ℕ) 
  (all_paint_same_rate : initial_people * initial_time = work_done) :
  initial_people = 8 →
  initial_time = 3 →
  final_time = 2 →
  work_done = 24 →
  ∃ extra_people : ℕ, extra_people = 4 :=
by
  sorry

end extra_people_needed_l144_144173


namespace arithmetic_square_root_of_nine_l144_144926

theorem arithmetic_square_root_of_nine :
  real.sqrt 9 = 3 :=
by
  sorry

end arithmetic_square_root_of_nine_l144_144926


namespace candy_remaining_l144_144956

theorem candy_remaining
  (initial_candies : ℕ)
  (talitha_took : ℕ)
  (solomon_took : ℕ)
  (h_initial : initial_candies = 349)
  (h_talitha : talitha_took = 108)
  (h_solomon : solomon_took = 153) :
  initial_candies - (talitha_took + solomon_took) = 88 :=
by
  sorry

end candy_remaining_l144_144956


namespace three_pow_mul_l144_144695

theorem three_pow_mul (a b : ℕ) (h_a : a = 12) (h_b : b = 18) :
  3^a * 3^b = 243^6 := by
  rw [h_a, h_b]
  calc
    3^12 * 3^18
      = 3^(12 + 18) : by rw [pow_add]
  ... = 3^30 : by norm_num
  ... = (3^5)^6 : by rw [pow_mul, ← mul_comm]
  ... = 243^6 : by norm_num

end three_pow_mul_l144_144695


namespace diorama_time_subtraction_l144_144326

theorem diorama_time_subtraction (P B X : ℕ) (h1 : B = 3 * P - X) (h2 : B = 49) (h3 : P + B = 67) : X = 5 :=
by
  sorry

end diorama_time_subtraction_l144_144326


namespace problem_1_problem_2_l144_144389

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a|

theorem problem_1 (x : ℝ) : (∀ x, f x 4 < 8 - |x - 1|) → x ∈ Set.Ioo (-1 : ℝ) (13 / 3) :=
by sorry

theorem problem_2 (a : ℝ) : (∃ x, f x a > 8 + |2 * x - 1|) → a > 9 ∨ a < -7 :=
by sorry

end problem_1_problem_2_l144_144389


namespace polynomial_remainder_l144_144501

theorem polynomial_remainder :
  ∀ (x : ℂ), (x^1010 % (x^4 - 1)) = x^2 :=
sorry

end polynomial_remainder_l144_144501


namespace z_in_third_quadrant_l144_144058

def i := Complex.I

def z := i + 2 * (i^2) + 3 * (i^3)

theorem z_in_third_quadrant : 
    let z_real := Complex.re z
    let z_imag := Complex.im z
    z_real < 0 ∧ z_imag < 0 :=
by
  sorry

end z_in_third_quadrant_l144_144058


namespace factorization_of_1386_l144_144225

-- We start by defining the number and the requirements.
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def factors_mult (a b : ℕ) : Prop := a * b = 1386
def factorization_count (count : ℕ) : Prop :=
  ∃ (a b : ℕ), is_two_digit a ∧ is_two_digit b ∧ factors_mult a b ∧ 
  (∀ c d, is_two_digit c ∧ is_two_digit d ∧ factors_mult c d → 
  (c = a ∧ d = b ∨ c = b ∧ d = a) → c = a ∧ d = b ∨ c = b ∧ d = a) ∧
  count = 4

-- Now, we state the theorem.
theorem factorization_of_1386 : factorization_count 4 :=
sorry

end factorization_of_1386_l144_144225


namespace product_of_decimals_l144_144138

def x : ℝ := 0.8
def y : ℝ := 0.12

theorem product_of_decimals : x * y = 0.096 :=
by
  sorry

end product_of_decimals_l144_144138


namespace child_support_owed_l144_144548

noncomputable def income_first_3_years : ℕ := 3 * 30000
noncomputable def raise_per_year : ℕ := 30000 * 20 / 100
noncomputable def new_salary : ℕ := 30000 + raise_per_year
noncomputable def income_next_4_years : ℕ := 4 * new_salary
noncomputable def total_income : ℕ := income_first_3_years + income_next_4_years
noncomputable def total_child_support : ℕ := total_income * 30 / 100
noncomputable def amount_paid : ℕ := 1200
noncomputable def amount_owed : ℕ := total_child_support - amount_paid

theorem child_support_owed : amount_owed = 69000 := by
  sorry

end child_support_owed_l144_144548


namespace no_such_function_l144_144265

theorem no_such_function :
  ¬ ∃ f : ℝ → ℝ, (∀ y x : ℝ, 0 < x → x < y → f y > (y - x) * (f x)^2) :=
by
  sorry

end no_such_function_l144_144265


namespace set_in_proportion_l144_144457

theorem set_in_proportion : 
  let a1 := 3
  let a2 := 9
  let b1 := 10
  let b2 := 30
  (a1 * b2 = a2 * b1) := 
by {
  sorry
}

end set_in_proportion_l144_144457


namespace correct_calculation_l144_144290

theorem correct_calculation (a : ℝ) : (3 * a^3)^2 = 9 * a^6 :=
by sorry

end correct_calculation_l144_144290


namespace range_of_a_l144_144735

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2 * a * x + 4 > 0

def q (a : ℝ) : Prop :=
  ∃ x y : ℝ, (x > 0 ∧ y > 0 ∨ x < 0 ∧ y < 0) ∧ y + (a - 1) * x + 2 * a - 1 = 0

def valid_a (a : ℝ) : Prop :=
  (p a ∨ q a) ∧ ¬(p a ∧ q a)

theorem range_of_a (a : ℝ) :
  valid_a a →
  (a ≤ -2 ∨ (1 ≤ a ∧ a < 2)) :=
sorry

end range_of_a_l144_144735


namespace candidate_knows_Excel_and_willing_nights_l144_144979

variable (PExcel PXNight : ℝ)
variable (H1 : PExcel = 0.20) (H2 : PXNight = 0.30)

theorem candidate_knows_Excel_and_willing_nights : (PExcel * PXNight) = 0.06 :=
by
  rw [H1, H2]
  norm_num

end candidate_knows_Excel_and_willing_nights_l144_144979


namespace solve_for_t_l144_144733

theorem solve_for_t (t : ℚ) :
  (t+2) * (4*t-4) = (4*t-6) * (t+3) + 3 → t = 7/2 :=
by {
  sorry
}

end solve_for_t_l144_144733


namespace basement_pump_time_l144_144983

/-- A basement has a 30-foot by 36-foot rectangular floor, flooded to a depth of 24 inches.
Using three pumps, each pumping 10 gallons per minute, and knowing that a cubic foot of water
contains 7.5 gallons, this theorem asserts it will take 540 minutes to pump out all the water. -/
theorem basement_pump_time :
  let length := 30 -- in feet
  let width := 36 -- in feet
  let depth_inch := 24 -- in inches
  let depth := depth_inch / 12 -- converting depth to feet
  let volume_ft3 := length * width * depth -- volume in cubic feet
  let gallons_per_ft3 := 7.5 -- gallons per cubic foot
  let total_gallons := volume_ft3 * gallons_per_ft3 -- total volume in gallons
  let pump_capacity_gpm := 10 -- gallons per minute per pump
  let total_pumps := 3 -- number of pumps
  let total_pump_gpm := pump_capacity_gpm * total_pumps -- total gallons per minute for all pumps
  let pump_time := total_gallons / total_pump_gpm -- time in minutes to pump all the water
  pump_time = 540 := sorry

end basement_pump_time_l144_144983


namespace hearts_total_shaded_area_l144_144569

theorem hearts_total_shaded_area (A B C D : ℕ) (hA : A = 1) (hB : B = 4) (hC : C = 9) (hD : D = 16) :
  (D - C) + (B - A) = 10 := 
by 
  sorry

end hearts_total_shaded_area_l144_144569


namespace min_count_to_ensure_multiple_of_5_l144_144349

theorem min_count_to_ensure_multiple_of_5 (n : ℕ) (S : Finset ℕ) (hS : S = Finset.range 31) :
  25 ≤ S.card ∧ (∀ (T : Finset ℕ), T ⊆ S → T.card = 24 → ↑(∃ x ∈ T, x % 5 = 0)) :=
by sorry

end min_count_to_ensure_multiple_of_5_l144_144349


namespace discriminant_of_P_l144_144356

theorem discriminant_of_P (a b c : ℚ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1/2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1/2 := 
begin
  -- Proof omitted for brevity
  sorry
end

end discriminant_of_P_l144_144356


namespace abc_zero_l144_144976

-- Define the given conditions as hypotheses
theorem abc_zero (a b c : ℚ) 
  (h1 : (a^2 + 1)^3 = b + 1)
  (h2 : (b^2 + 1)^3 = c + 1)
  (h3 : (c^2 + 1)^3 = a + 1) : 
  a = 0 ∧ b = 0 ∧ c = 0 := 
sorry

end abc_zero_l144_144976


namespace quadratic_polynomial_discriminant_l144_144371

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₁ : a ≠ 0)
  (h₂ : ∃ x : ℝ, a * x^2 + b * x + c = x - 2 ∧ (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h₃ : ∃ x : ℝ, a * x^2 + b * x + c = 1 - x / 2 ∧ (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l144_144371


namespace length_of_AB_l144_144987

-- Definitions of the given entities
def is_on_parabola (A : ℝ × ℝ) : Prop := A.2^2 = 4 * A.1
def focus : ℝ × ℝ := (1, 0)
def line_through_focus (l : ℝ × ℝ → Prop) : Prop := l focus

-- The theorem we need to prove
theorem length_of_AB (A B : ℝ × ℝ) (l : ℝ × ℝ → Prop)
  (h1 : is_on_parabola A)
  (h2 : is_on_parabola B)
  (h3 : line_through_focus l)
  (h4 : l A)
  (h5 : l B)
  (h6 : A.1 + B.1 = 10 / 3) :
  dist A B = 16 / 3 :=
sorry

end length_of_AB_l144_144987


namespace num_integers_div_10_or_12_l144_144654

-- Define the problem in Lean
theorem num_integers_div_10_or_12 (N : ℕ) : (1 ≤ N ∧ N ≤ 2007) ∧ (N % 10 = 0 ∨ N % 12 = 0) ↔ ∃ k, k = 334 := by
  sorry

end num_integers_div_10_or_12_l144_144654


namespace baking_time_correct_l144_144418

/-- Mark lets the bread rise for 120 minutes twice. -/
def rising_time : ℕ := 120 * 2

/-- Mark spends 10 minutes kneading the bread. -/
def kneading_time : ℕ := 10

/-- Total time taken to finish making the bread. -/
def total_time : ℕ := 280

/-- Calculate the baking time based on the given conditions. -/
def baking_time (rising kneading total : ℕ) : ℕ := total - (rising + kneading)

theorem baking_time_correct :
  baking_time rising_time kneading_time total_time = 30 := 
by 
  -- Proof is omitted
  sorry

end baking_time_correct_l144_144418


namespace frogs_needing_new_pond_l144_144006

theorem frogs_needing_new_pond : 
    (initial_frogs tadpoles_survival_fraction pond_capacity : ℕ) 
    (h1 : initial_frogs = 5) 
    (h2 : tadpoles_survival_fraction = 2 / 3) 
    (h3 : pond_capacity = 8) 
    : (initial_frogs + tadpoles_survival_fraction * 3 * initial_frogs - pond_capacity = 7) :=
by
    sorry

end frogs_needing_new_pond_l144_144006


namespace key_lime_yield_l144_144165

def audrey_key_lime_juice_yield (cup_to_key_lime_juice_ratio: ℚ) (lime_juice_doubling_factor: ℚ) (tablespoons_per_cup: ℕ) (num_key_limes: ℕ) : ℚ :=
  let total_lime_juice_cups := cup_to_key_lime_juice_ratio * lime_juice_doubling_factor
  let total_lime_juice_tablespoons := total_lime_juice_cups * tablespoons_per_cup
  total_lime_juice_tablespoons / num_key_limes

-- Statement of the problem
theorem key_lime_yield :
  audrey_key_lime_juice_yield (1/4) 2 16 8 = 1 := 
by 
  sorry

end key_lime_yield_l144_144165


namespace chairs_built_in_10_days_l144_144800

-- Define the conditions as variables
def hours_per_day : ℕ := 8
def days_worked : ℕ := 10
def hours_per_chair : ℕ := 5

-- State the problem as a conjecture or theorem
theorem chairs_built_in_10_days : (hours_per_day * days_worked) / hours_per_chair = 16 := by
    sorry

end chairs_built_in_10_days_l144_144800


namespace number_of_girls_in_school_l144_144809

-- Variables representing the population and the sample.
variables (total_students sample_size boys_sample girls_sample : ℕ)

-- Initial conditions.
def initial_conditions := 
  total_students = 1600 ∧ 
  sample_size = 200 ∧
  girls_sample = 90 ∧
  boys_sample = 110 ∧
  (girls_sample + 20 = boys_sample)

-- Statement to prove.
theorem number_of_girls_in_school (x: ℕ) 
  (h : initial_conditions total_students sample_size boys_sample girls_sample) :
  x = 720 :=
by {
  -- Obligatory proof omitted.
  sorry
}

end number_of_girls_in_school_l144_144809


namespace money_distribution_problem_l144_144528

theorem money_distribution_problem :
  ∃ n : ℕ, (3 * n + n * (n - 1) / 2 = 100 * n) ∧ n = 195 :=
by {
  use 195,
  sorry
}

end money_distribution_problem_l144_144528


namespace soldier_rearrangement_20x20_soldier_rearrangement_21x21_l144_144770

theorem soldier_rearrangement_20x20 (d : ℝ) : d ≤ 10 * Real.sqrt 2 :=
by
  -- Problem (a) setup and conditions
  sorry

theorem soldier_rearrangement_21x21 (d : ℝ) : d ≤ 10 * Real.sqrt 2 :=
by
  -- Problem (b) setup and conditions
  sorry

end soldier_rearrangement_20x20_soldier_rearrangement_21x21_l144_144770


namespace continuous_function_fixed_point_l144_144440

variable (f : ℝ → ℝ)
variable (h_cont : Continuous f)
variable (h_comp : ∀ x : ℝ, ∃ n : ℕ, n > 0 ∧ (f^[n] x = 1))

theorem continuous_function_fixed_point : f 1 = 1 := 
by
  sorry

end continuous_function_fixed_point_l144_144440


namespace smallest_solution_is_39_over_8_l144_144195

noncomputable def smallest_solution (x : ℝ) : Prop :=
  (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ (x ≠ 0) ∧ (x ≠ 3)

theorem smallest_solution_is_39_over_8 : ∃ x > 0, smallest_solution x ∧ x = 39 / 8 :=
by
  sorry

end smallest_solution_is_39_over_8_l144_144195


namespace sum_powers_l144_144083

open Complex

theorem sum_powers (ω : ℂ) (h₁ : ω^5 = 1) (h₂ : ω ≠ 1) : 
  ω^10 + ω^12 + ω^14 + ω^16 + ω^18 + ω^20 + ω^22 + ω^24 + ω^26 + ω^28 + ω^30 = 1 := sorry

end sum_powers_l144_144083


namespace solve_quadratic_solution_l144_144575

theorem solve_quadratic_solution (x : ℝ) : (3 * x^2 - 6 * x = 0) ↔ (x = 0 ∨ x = 2) :=
sorry

end solve_quadratic_solution_l144_144575


namespace find_x_l144_144497

theorem find_x (x : ℝ) : abs (2 * x - 1) = 3 * x + 6 ∧ x + 2 > 0 ↔ x = -1 := 
by
  sorry

end find_x_l144_144497


namespace largest_consecutive_positive_elements_l144_144503

theorem largest_consecutive_positive_elements (a : ℕ → ℝ)
  (h₁ : ∀ n ≥ 2, a n = a (n-1) + a (n+2)) :
  ∃ m, m = 5 ∧ ∀ k < m, a k > 0 :=
sorry

end largest_consecutive_positive_elements_l144_144503


namespace solve_equation_l144_144198

noncomputable def smallest_solution : Rat :=
  (8 - Real.sqrt 145) / 3

theorem solve_equation : 
  ∃ x : ℝ, (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ x = smallest_solution := sorry

end solve_equation_l144_144198


namespace percentage_of_ducks_among_non_heron_l144_144853

def birds_percentage (geese swans herons ducks total_birds : ℕ) : ℕ :=
  let non_heron_birds := total_birds - herons
  let duck_percentage := (ducks * 100) / non_heron_birds
  duck_percentage

theorem percentage_of_ducks_among_non_heron : 
  birds_percentage 28 20 15 32 100 = 37 :=   /- 37 approximates 37.6 -/
sorry

end percentage_of_ducks_among_non_heron_l144_144853


namespace problem_I_problem_II_problem_III_l144_144514

noncomputable def f (a x : ℝ) := a * x * Real.exp x
noncomputable def f' (a x : ℝ) := a * (1 + x) * Real.exp x

theorem problem_I (a : ℝ) (h : a ≠ 0) :
  (if a > 0 then ∀ x, (f' a x > 0 ↔ x > -1) ∧ (f' a x < 0 ↔ x < -1)
  else ∀ x, (f' a x > 0 ↔ x < -1) ∧ (f' a x < 0 ↔ x > -1)) :=
sorry

theorem problem_II (h : ∃ a : ℝ, a = 1) :
  ∃ (x : ℝ) (y : ℝ), x = -1 ∧ f 1 (-1) = -1 / Real.exp 1 ∧ ¬ ∃ y, ∀ x, y = f 1 x ∧ (f' 1 x) < 0 :=
sorry

theorem problem_III (h : ∃ m : ℝ, f 1 m = e * m * Real.exp m ∧ f' 1 m = e * (1 + m) * Real.exp m) :
  ∃ a : ℝ, a = 1 / 2 :=
sorry

end problem_I_problem_II_problem_III_l144_144514


namespace arithmetic_square_root_of_nine_l144_144895

theorem arithmetic_square_root_of_nine : Real.sqrt 9 = 3 :=
sorry

end arithmetic_square_root_of_nine_l144_144895


namespace line_y2_does_not_pass_through_fourth_quadrant_l144_144316

theorem line_y2_does_not_pass_through_fourth_quadrant (k b : ℝ) (h1 : k < 0) (h2 : b > 0) : 
  ¬(∃ x y : ℝ, (y = b * x - k ∧ x > 0 ∧ y < 0)) := 
by 
  sorry

end line_y2_does_not_pass_through_fourth_quadrant_l144_144316


namespace sector_angle_l144_144384

theorem sector_angle (r l : ℝ) (h₁ : 2 * r + l = 4) (h₂ : 1/2 * l * r = 1) : l / r = 2 :=
by
  sorry

end sector_angle_l144_144384


namespace sum_and_product_l144_144572

theorem sum_and_product (c d : ℝ) (h1 : 2 * c = -8) (h2 : c^2 - d = 4) : c + d = 8 := by
  sorry

end sum_and_product_l144_144572


namespace brendan_total_wins_l144_144999

-- Define the number of matches won in each round
def matches_won_first_round : ℕ := 6
def matches_won_second_round : ℕ := 4
def matches_won_third_round : ℕ := 3
def matches_won_final_round : ℕ := 5

-- Define the total number of matches won
def total_matches_won : ℕ := 
  matches_won_first_round + matches_won_second_round + matches_won_third_round + matches_won_final_round

-- State the theorem that needs to be proven
theorem brendan_total_wins : total_matches_won = 18 := by
  sorry

end brendan_total_wins_l144_144999


namespace more_geese_than_ducks_l144_144626

def mallard_start := 25
def wood_start := 15
def geese_start := 2 * mallard_start - 10
def swan_start := 3 * wood_start + 8

def mallard_after_morning := mallard_start + 4
def wood_after_morning := wood_start + 8
def geese_after_morning := geese_start + 7
def swan_after_morning := swan_start

def mallard_after_noon := mallard_after_morning
def wood_after_noon := wood_after_morning - 6
def geese_after_noon := geese_after_morning - 5
def swan_after_noon := swan_after_morning - 9

def mallard_after_later := mallard_after_noon + 8
def wood_after_later := wood_after_noon + 10
def geese_after_later := geese_after_noon
def swan_after_later := swan_after_noon + 4

def mallard_after_evening := mallard_after_later + 5
def wood_after_evening := wood_after_later + 3
def geese_after_evening := geese_after_later + 15
def swan_after_evening := swan_after_later + 11

def mallard_final := 0
def wood_final := wood_after_evening - (3 / 4 : ℚ) * wood_after_evening
def geese_final := geese_after_evening - (1 / 5 : ℚ) * geese_after_evening
def swan_final := swan_after_evening - (1 / 2 : ℚ) * swan_after_evening

theorem more_geese_than_ducks :
  (geese_final - (mallard_final + wood_final)) = 38 :=
by sorry

end more_geese_than_ducks_l144_144626


namespace Jack_heavier_than_Sam_l144_144953

def total_weight := 96 -- total weight of Jack and Sam in pounds
def jack_weight := 52 -- Jack's weight in pounds

def sam_weight := total_weight - jack_weight

theorem Jack_heavier_than_Sam : jack_weight - sam_weight = 8 := by
  -- Here we would provide a proof, but we leave it as sorry for now.
  sorry

end Jack_heavier_than_Sam_l144_144953


namespace arithmetic_geometric_sequences_l144_144871

variable {S T : ℕ → ℝ}
variable {a b : ℕ → ℝ}

theorem arithmetic_geometric_sequences (h1 : a 3 = b 3)
  (h2 : a 4 = b 4)
  (h3 : (S 5 - S 3) / (T 4 - T 2) = 5) :
  (a 5 + a 3) / (b 5 + b 3) = - (3 / 5) := by
  sorry

end arithmetic_geometric_sequences_l144_144871


namespace pieces_not_chewed_l144_144542

theorem pieces_not_chewed : 
  (8 * 7 - 54) = 2 := 
by 
  sorry

end pieces_not_chewed_l144_144542


namespace acute_triangle_altitude_inequality_l144_144234

theorem acute_triangle_altitude_inequality (a b c d e f : ℝ) 
  (A B C : ℝ) 
  (acute_triangle : (d = b * Real.sin C) ∧ (d = c * Real.sin B) ∧
                    (e = a * Real.sin C) ∧ (f = a * Real.sin B))
  (projections : (de = b * Real.cos B) ∧ (df = c * Real.cos C))
  : (de + df ≤ a) := 
sorry

end acute_triangle_altitude_inequality_l144_144234


namespace find_value_of_m_l144_144504

theorem find_value_of_m : ∃ m : ℤ, 2^4 - 3 = 5^2 + m ∧ m = -12 :=
by
  use -12
  sorry

end find_value_of_m_l144_144504


namespace arithmetic_square_root_of_nine_l144_144925

theorem arithmetic_square_root_of_nine :
  real.sqrt 9 = 3 :=
by
  sorry

end arithmetic_square_root_of_nine_l144_144925


namespace max_possible_score_l144_144530

theorem max_possible_score (s : ℝ) (h : 80 = s * 2) : s * 5 ≥ 100 :=
by
  -- sorry placeholder for the proof
  sorry

end max_possible_score_l144_144530


namespace find_rainy_days_l144_144734

theorem find_rainy_days 
  (n d T H P R : ℤ) 
  (h1 : R + (d - R) = d)
  (h2 : 3 * (d - R) = T)
  (h3 : n * R = H)
  (h4 : T = H + P)
  (hd : 1 ≤ d ∧ d ≤ 31)
  (hR_range : 0 ≤ R ∧ R ≤ d) :
  R = (3 * d - P) / (n + 3) :=
sorry

end find_rainy_days_l144_144734


namespace average_branches_per_foot_correct_l144_144634

def height_tree_1 : ℕ := 50
def branches_tree_1 : ℕ := 200
def height_tree_2 : ℕ := 40
def branches_tree_2 : ℕ := 180
def height_tree_3 : ℕ := 60
def branches_tree_3 : ℕ := 180
def height_tree_4 : ℕ := 34
def branches_tree_4 : ℕ := 153

def total_height := height_tree_1 + height_tree_2 + height_tree_3 + height_tree_4
def total_branches := branches_tree_1 + branches_tree_2 + branches_tree_3 + branches_tree_4
def average_branches_per_foot := total_branches / total_height

theorem average_branches_per_foot_correct : average_branches_per_foot = 713 / 184 := 
  by
    -- Proof omitted, directly state the result
    sorry

end average_branches_per_foot_correct_l144_144634


namespace actual_diameter_of_tissue_l144_144145

variable (magnified_diameter : ℝ) (magnification_factor : ℝ)

theorem actual_diameter_of_tissue 
    (h1 : magnified_diameter = 0.2) 
    (h2 : magnification_factor = 1000) : 
    magnified_diameter / magnification_factor = 0.0002 := 
  by
    sorry

end actual_diameter_of_tissue_l144_144145


namespace expenses_notation_l144_144437

theorem expenses_notation (income expense : ℤ) (h_income : income = 6) (h_expense : -expense = income) : expense = -4 := 
by
  sorry

end expenses_notation_l144_144437


namespace fraction_proof_l144_144351

variables (m n p q : ℚ)

theorem fraction_proof
  (h1 : m / n = 18)
  (h2 : p / n = 9)
  (h3 : p / q = 1 / 15) :
  m / q = 2 / 15 :=
by sorry

end fraction_proof_l144_144351


namespace Papi_Calot_has_to_buy_141_plants_l144_144881

noncomputable def calc_number_of_plants : Nat :=
  let initial_plants := 7 * 18
  let additional_plants := 15
  initial_plants + additional_plants

theorem Papi_Calot_has_to_buy_141_plants :
  calc_number_of_plants = 141 :=
by
  sorry

end Papi_Calot_has_to_buy_141_plants_l144_144881


namespace sum_of_squares_l144_144106

theorem sum_of_squares (n : ℕ) : ∃ k : ℤ, (∃ a b : ℤ, k = a^2 + b^2) ∧ (∃ d : ℕ, d ≥ n) :=
by
  sorry

end sum_of_squares_l144_144106


namespace integer_part_of_shortest_distance_l144_144985

def cone_slant_height := 21
def cone_radius := 14
def ant_position := cone_slant_height / 2
def angle_opposite := 240
def cos_angle_opposite := -1 / 2

noncomputable def shortest_distance := 
  Real.sqrt ((ant_position ^ 2) + (ant_position ^ 2) + (2 * ant_position ^ 2 * cos_angle_opposite))

theorem integer_part_of_shortest_distance : Int.floor shortest_distance = 18 :=
by
  /- Proof steps go here -/
  sorry

end integer_part_of_shortest_distance_l144_144985


namespace solution_set_of_abs_2x_minus_1_ge_3_l144_144201

theorem solution_set_of_abs_2x_minus_1_ge_3 :
  { x : ℝ | |2 * x - 1| ≥ 3 } = { x : ℝ | x ≤ -1 } ∪ { x : ℝ | x ≥ 2 } := 
sorry

end solution_set_of_abs_2x_minus_1_ge_3_l144_144201


namespace total_cost_correct_l144_144581

-- Condition C1: There are 13 hearts in a deck of 52 playing cards. 
def hearts_in_deck : ℕ := 13

-- Condition C2: The number of cows is twice the number of hearts.
def cows_in_Devonshire : ℕ := 2 * hearts_in_deck

-- Condition C3: Each cow is sold at $200.
def cost_per_cow : ℕ := 200

-- Question Q1: Calculate the total cost of the cows.
def total_cost_of_cows : ℕ := cows_in_Devonshire * cost_per_cow

-- Final statement we need to prove
theorem total_cost_correct : total_cost_of_cows = 5200 := by
  -- This will be proven in the proof body
  sorry

end total_cost_correct_l144_144581


namespace expand_product_l144_144178

theorem expand_product (x : ℤ) : 
  (3 * x + 4) * (2 * x - 6) = 6 * x^2 - 10 * x - 24 :=
by
  sorry

end expand_product_l144_144178


namespace smallest_solution_to_equation_l144_144193

noncomputable def smallest_solution := (11 - Real.sqrt 445) / 6

theorem smallest_solution_to_equation:
  ∃ x : ℝ, (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ (x = smallest_solution) :=
sorry

end smallest_solution_to_equation_l144_144193


namespace small_bottles_sold_percentage_l144_144017

theorem small_bottles_sold_percentage
  (small_bottles : ℕ) (big_bottles : ℕ) (percent_sold_big_bottles : ℝ)
  (remaining_bottles : ℕ) (percent_sold_small_bottles : ℝ) :
  small_bottles = 6000 ∧
  big_bottles = 14000 ∧
  percent_sold_big_bottles = 0.23 ∧
  remaining_bottles = 15580 ∧ 
  percent_sold_small_bottles / 100 * 6000 + 0.23 * 14000 + remaining_bottles = small_bottles + big_bottles →
  percent_sold_small_bottles = 37 := 
by
  intros
  exact sorry

end small_bottles_sold_percentage_l144_144017


namespace smallest_product_l144_144758

theorem smallest_product (S : Set ℤ) (hS : S = { -8, -3, -2, 2, 4 }) :
  ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ a * b = -32 ∧ ∀ (x y : ℤ), x ∈ S → y ∈ S → x * y ≥ -32 :=
by
  sorry

end smallest_product_l144_144758


namespace initial_riding_time_l144_144543

theorem initial_riding_time (t : ℝ) (h1 : t * 60 + 90 + 30 + 120 = 270) : t * 60 = 30 :=
by sorry

end initial_riding_time_l144_144543


namespace smallest_x_multiple_of_53_l144_144496

theorem smallest_x_multiple_of_53 : ∃ (x : Nat), (x > 0) ∧ ( ∀ (n : Nat), (n > 0) ∧ ((3 * n + 43) % 53 = 0) → x ≤ n ) ∧ ((3 * x + 43) % 53 = 0) :=
sorry

end smallest_x_multiple_of_53_l144_144496


namespace power_expression_l144_144677

theorem power_expression (a b : ℕ) (h1 : a = 12) (h2 : b = 18) : (3^a * 3^b) = (243^6) :=
by
  let c := 3
  have h3 : a + b = 30 := by simp [h1, h2]
  have h4 : 3^(a + b) = 3^30 := by rw [h3]
  have h5 : 3^30 = 243^6 := by norm_num
  sorry  -- skip other detailed steps

end power_expression_l144_144677


namespace veggies_minus_fruits_l144_144760

-- Definitions of quantities as given in the conditions
def cucumbers : ℕ := 6
def tomatoes : ℕ := 8
def apples : ℕ := 2
def bananas : ℕ := 4

-- Problem Statement
theorem veggies_minus_fruits : (cucumbers + tomatoes) - (apples + bananas) = 8 :=
by 
  -- insert proof here
  sorry

end veggies_minus_fruits_l144_144760


namespace sweater_cost_l144_144717

theorem sweater_cost (S : ℚ) (M : ℚ) (C : ℚ) (h1 : S = 80) (h2 : M = 3 / 4 * 80) (h3 : C = S - M) : C = 20 := by
  sorry

end sweater_cost_l144_144717


namespace original_number_j_l144_144248

noncomputable def solution (n : ℚ) : ℚ := (3 * (n + 3) - 5) / 3

theorem original_number_j { n : ℚ } (h : solution n = 10) : n = 26 / 3 :=
by
  sorry

end original_number_j_l144_144248


namespace rectangle_area_l144_144592

theorem rectangle_area (l w : ℝ) (h1 : 2 * l + 2 * w = 14) (h2 : l^2 + w^2 = 25) : l * w = 12 :=
by
  sorry

end rectangle_area_l144_144592


namespace intersection_point_of_lines_l144_144942

theorem intersection_point_of_lines :
  ∃ x y : ℝ, 
    2 * x + y - 7 = 0 ∧ 
    x + 2 * y - 5 = 0 ∧ 
    x = 3 ∧ 
    y = 1 := 
by {
  sorry
}

end intersection_point_of_lines_l144_144942


namespace smallest_c_in_range_l144_144655

-- Define the quadratic function g(x)
def g (x c : ℝ) : ℝ := 2 * x ^ 2 - 4 * x + c

-- Define the condition for c
def in_range_5 (c : ℝ) : Prop :=
  ∃ x : ℝ, g x c = 5

-- The theorem stating that the smallest value of c for which 5 is in the range of g is 7
theorem smallest_c_in_range : ∃ c : ℝ, c = 7 ∧ ∀ c' : ℝ, (in_range_5 c' → 7 ≤ c') :=
sorry

end smallest_c_in_range_l144_144655


namespace right_triangle_median_square_l144_144975

theorem right_triangle_median_square (a b c k_a k_b : ℝ) :
  c = Real.sqrt (a^2 + b^2) → -- c is the hypotenuse
  k_a = Real.sqrt ((2 * b^2 + 2 * (a^2 + b^2) - a^2) / 4) → -- k_a is the median to side a
  k_b = Real.sqrt ((2 * a^2 + 2 * (a^2 + b^2) - b^2) / 4) → -- k_b is the median to side b
  c^2 = (4 / 5) * (k_a^2 + k_b^2) :=
by
  intros h_c h_ka h_kb
  sorry

end right_triangle_median_square_l144_144975


namespace number_of_new_galleries_l144_144790

-- Definitions based on conditions
def number_of_pictures_first_gallery := 9
def number_of_pictures_per_new_gallery := 2
def pencils_per_picture := 4
def pencils_per_exhibition_signature := 2
def total_pencils_used := 88

-- Theorem statement according to the correct answer
theorem number_of_new_galleries 
  (number_of_pictures_first_gallery : ℕ)
  (number_of_pictures_per_new_gallery : ℕ)
  (pencils_per_picture : ℕ)
  (pencils_per_exhibition_signature : ℕ)
  (total_pencils_used : ℕ)
  (drawing_pencils_first_gallery := number_of_pictures_first_gallery * pencils_per_picture)
  (signing_pencils_first_gallery := pencils_per_exhibition_signature)
  (total_pencils_first_gallery := drawing_pencils_first_gallery + signing_pencils_first_gallery)
  (pencils_for_new_galleries := total_pencils_used - total_pencils_first_gallery)
  (pencils_per_new_gallery := (number_of_pictures_per_new_gallery * pencils_per_picture) + pencils_per_exhibition_signature) :
  pencils_per_new_gallery > 0 → pencils_for_new_galleries / pencils_per_new_gallery = 5 :=
sorry

end number_of_new_galleries_l144_144790


namespace total_apple_trees_l144_144026

-- Definitions and conditions
def ava_trees : ℕ := 9
def lily_trees : ℕ := ava_trees - 3
def total_trees : ℕ := ava_trees + lily_trees

-- Statement to be proved
theorem total_apple_trees :
  total_trees = 15 := by
  sorry

end total_apple_trees_l144_144026


namespace triangle_side_length_l144_144509

theorem triangle_side_length (a : ℝ) (h1 : 4 < a) (h2 : a < 8) : a = 6 :=
sorry

end triangle_side_length_l144_144509


namespace cakes_served_at_lunch_today_l144_144991

variable (L : ℕ)
variable (dinnerCakes : ℕ) (yesterdayCakes : ℕ) (totalCakes : ℕ)

theorem cakes_served_at_lunch_today :
  (dinnerCakes = 6) → (yesterdayCakes = 3) → (totalCakes = 14) → (L + dinnerCakes + yesterdayCakes = totalCakes) → L = 5 :=
by
  intros h_dinner h_yesterday h_total h_eq
  sorry

end cakes_served_at_lunch_today_l144_144991


namespace convert_base4_to_base10_l144_144807

-- Define a function to convert a base 4 number to base 10
def base4_to_base10 (n : Nat) : Nat :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  let d3 := (n / 1000) % 10
  d3 * 4^3 + d2 * 4^2 + d1 * 4^1 + d0 * 4^0

-- Assert the proof problem
theorem convert_base4_to_base10 : base4_to_base10 3201 = 225 :=
by
  -- The proof script goes here; for now, we use 'sorry' as a placeholder
  sorry

end convert_base4_to_base10_l144_144807


namespace chess_probability_l144_144104

theorem chess_probability (P_draw P_B_win : ℚ) (h_draw : P_draw = 1/2) (h_B_win : P_B_win = 1/3) :
  (1 - P_draw - P_B_win = 1/6) ∧ -- Statement A is correct
  (P_draw + (1 - P_draw - P_B_win) ≠ 1/2) ∧ -- Statement B is incorrect as it's not 1/2
  (1 - P_draw - P_B_win ≠ 2/3) ∧ -- Statement C is incorrect as it's not 2/3
  (P_draw + P_B_win ≠ 1/2) := -- Statement D is incorrect as it's not 1/2
by
  -- Insert proof here
  sorry

end chess_probability_l144_144104


namespace solve_system_of_equations_l144_144885

theorem solve_system_of_equations :
  ∃ (x y: ℝ), (x - y - 1 = 0) ∧ (4 * (x - y) - y = 0) ∧ (x = 5) ∧ (y = 4) :=
by
  sorry

end solve_system_of_equations_l144_144885


namespace white_tshirts_l144_144292

theorem white_tshirts (packages shirts_per_package : ℕ) (h1 : packages = 71) (h2 : shirts_per_package = 6) : packages * shirts_per_package = 426 := 
by 
  sorry

end white_tshirts_l144_144292


namespace binomial_odd_sum_l144_144124

theorem binomial_odd_sum (n : ℕ) (h : (2:ℕ)^(n - 1) = 64) : n = 7 :=
by
  sorry

end binomial_odd_sum_l144_144124


namespace part1_part2_l144_144090

-- Definitions for problem conditions and questions

/-- 
Let p and q be two distinct prime numbers greater than 5. 
Show that if p divides 5^q - 2^q then q divides p - 1.
-/
theorem part1 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (hp_gt_5 : 5 < p) (hq_gt_5 : 5 < q) (h_distinct : p ≠ q) 
  (h_div : p ∣ 5^q - 2^q) : q ∣ p - 1 :=
by sorry

/-- 
Let p and q be two distinct prime numbers greater than 5.
Deduce that pq does not divide (5^p - 2^p)(5^q - 2^q).
-/
theorem part2 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (hp_gt_5 : 5 < p) (hq_gt_5 : 5 < q) (h_distinct : p ≠ q) 
  (h_div_q_p1 : q ∣ p - 1)
  (h_div_p_q1 : p ∣ q - 1) : ¬(pq : ℕ) ∣ (5^p - 2^p) * (5^q - 2^q) :=
by sorry

end part1_part2_l144_144090


namespace renaldo_distance_l144_144557

theorem renaldo_distance (R : ℕ) (h : R + (1/3 : ℝ) * R + 7 = 27) : R = 15 :=
by sorry

end renaldo_distance_l144_144557


namespace probability_of_at_least_one_boy_and_one_girl_l144_144618

noncomputable def probability_at_least_one_boy_and_one_girl: ℚ :=
  7 / 8

axiom equally_likely_birth : ∀ i : ℕ, (i = 0 ∨ i = 1) → (0.5 : ℝ)

theorem probability_of_at_least_one_boy_and_one_girl (n : ℕ) (condition : n = 4) : 
  probability_at_least_one_boy_and_one_girl = 7 / 8 :=
sorry

end probability_of_at_least_one_boy_and_one_girl_l144_144618


namespace tan_sum_product_l144_144988

theorem tan_sum_product (tan : ℝ → ℝ) : 
  (1 + tan 23) * (1 + tan 22) = 2 + tan 23 * tan 22 := by sorry

end tan_sum_product_l144_144988


namespace cow_cost_calculation_l144_144580

constant hearts_per_card : ℕ := 4
constant cards_in_deck : ℕ := 52
constant cost_per_cow : ℕ := 200

def total_hearts : ℕ := hearts_per_card * cards_in_deck
def number_of_cows : ℕ := 2 * total_hearts
def total_cost_of_cows : ℕ := number_of_cows * cost_per_cow

theorem cow_cost_calculation :
  total_cost_of_cows = 83200 := by
  -- Placeholder proof
  sorry

end cow_cost_calculation_l144_144580


namespace cost_of_each_candy_bar_l144_144450

-- Definitions of the conditions
def initial_amount : ℕ := 20
def final_amount : ℕ := 12
def number_of_candy_bars : ℕ := 4

-- Statement of the proof problem: prove the cost of each candy bar
theorem cost_of_each_candy_bar :
  (initial_amount - final_amount) / number_of_candy_bars = 2 := by
  sorry

end cost_of_each_candy_bar_l144_144450


namespace arithmetic_sequence_common_difference_l144_144855

theorem arithmetic_sequence_common_difference (a : ℕ → ℤ) (d : ℤ) (h1 : a 3 = 7) (h2 : a 7 = -5)
  (h3 : ∀ n, a (n + 1) = a n + d) : 
  d = -3 :=
sorry

end arithmetic_sequence_common_difference_l144_144855


namespace problem_div_expansion_l144_144631

theorem problem_div_expansion (m : ℝ) : ((2 * m^2 - m)^2) / (-m^2) = -4 * m^2 + 4 * m - 1 := 
by sorry

end problem_div_expansion_l144_144631


namespace max_abs_asin_b_l144_144049

theorem max_abs_asin_b (a b c : ℝ) (h : ∀ x : ℝ, |a * (Real.cos x)^2 + b * Real.sin x + c| ≤ 1) :
  ∃ M : ℝ, (∀ x : ℝ, |a * Real.sin x + b| ≤ M) ∧ M = 2 :=
sorry

end max_abs_asin_b_l144_144049


namespace min_value_x_l144_144394

open Real 

variable (x : ℝ)

theorem min_value_x (hx_pos : 0 < x) 
    (ineq : log x ≥ 2 * log 3 + (1 / 3) * log x + 1) : 
    x ≥ 27 * exp (3 / 2) :=
by 
  sorry

end min_value_x_l144_144394


namespace sum_seq_equals_2_pow_n_minus_1_l144_144491

-- Define the sequences a_n and b_n with given conditions
def a (n : ℕ) : ℕ := if n = 0 then 2 else if n = 1 then 4 else sorry
def b (n : ℕ) : ℕ := if n = 0 then 2 else if n = 1 then 4 else sorry

-- Relation for a_n: 2a_{n+1} = a_n + a_{n+2}
axiom a_relation (n : ℕ) : 2 * a (n + 1) = a n + a (n + 2)

-- Inequalities for b_n
axiom b_inequality_1 (n : ℕ) : b (n + 1) - b n < 2^n + 1 / 2
axiom b_inequality_2 (n : ℕ) : b (n + 2) - b n > 3 * 2^n - 1

-- Note that b_n ∈ ℤ is implied by the definition being in ℕ

-- Prove that the sum of the first n terms of the sequence { n * b_n / a_n }
theorem sum_seq_equals_2_pow_n_minus_1 (n : ℕ) : 
  (Finset.range n).sum (λ k => k * b k / a k) = 2^n - 1 := 
sorry

end sum_seq_equals_2_pow_n_minus_1_l144_144491


namespace arithmetic_square_root_of_9_l144_144912

theorem arithmetic_square_root_of_9 : sqrt 9 = 3 := by
  sorry

end arithmetic_square_root_of_9_l144_144912


namespace blue_stripe_area_l144_144003

def cylinder_diameter : ℝ := 20
def cylinder_height : ℝ := 60
def stripe_width : ℝ := 4
def stripe_revolutions : ℕ := 3

theorem blue_stripe_area : 
  let circumference := Real.pi * cylinder_diameter
  let stripe_length := stripe_revolutions * circumference
  let expected_area := stripe_width * stripe_length
  expected_area = 240 * Real.pi :=
by
  sorry

end blue_stripe_area_l144_144003


namespace chewing_gums_count_l144_144098

-- Given conditions
def num_chocolate_bars : ℕ := 55
def num_candies : ℕ := 40
def total_treats : ℕ := 155

-- Definition to be proven
def num_chewing_gums : ℕ := total_treats - (num_chocolate_bars + num_candies)

-- Theorem statement
theorem chewing_gums_count : num_chewing_gums = 60 :=
by 
  -- here would be the proof steps, but it's omitted as per the instruction
  sorry

end chewing_gums_count_l144_144098


namespace solve_inequality_l144_144827

variable (f : ℝ → ℝ)

-- Define the conditions
axiom f_decreasing : ∀ x y : ℝ, x ≤ y → f y ≤ f x

-- Prove the main statement
theorem solve_inequality (h : ∀ x : ℝ, f (f x) = x) : ∀ x : ℝ, f (f x) = x := 
by
  sorry

end solve_inequality_l144_144827


namespace math_problem_l144_144410

noncomputable def problem_statement (f : ℚ → ℝ) : Prop :=
  (∀ r s : ℚ, ∃ n : ℤ, f (r + s) = f r + f s + n) →
  ∃ (q : ℕ) (p : ℤ), abs (f (1 / q) - p) ≤ 1 / 2012

-- To state this problem as a theorem in Lean 4
theorem math_problem (f : ℚ → ℝ) :
  problem_statement f :=
sorry

end math_problem_l144_144410


namespace problem_inequality_l144_144204

variable {x y : ℝ}

theorem problem_inequality (hx : 2 < x) (hy : 2 < y) : 
  (x^2 - x) / (y^2 + y) + (y^2 - y) / (x^2 + x) > 2 / 3 := 
  sorry

end problem_inequality_l144_144204


namespace exists_linear_function_l144_144868

-- Define the properties of the function f
def is_contraction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, |f x - f y| ≤ |x - y|

-- Define the property of an arithmetic progression
def is_arith_seq (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, ∃ d : ℝ, ∀ n : ℕ, (f^[n] x) = x + n * d

-- Main theorem to prove
theorem exists_linear_function (f : ℝ → ℝ) (h1 : is_contraction f) (h2 : is_arith_seq f) : ∃ a : ℝ, ∀ x : ℝ, f x = x + a :=
sorry

end exists_linear_function_l144_144868


namespace gcd_228_2008_l144_144442

theorem gcd_228_2008 : Int.gcd 228 2008 = 4 := by
  sorry

end gcd_228_2008_l144_144442


namespace problem1_problem2_l144_144431

-- Problem 1
theorem problem1 (x : ℝ) (h1 : 2 * x > 1 - x) (h2 : x + 2 < 4 * x - 1) : x > 1 := 
by
  sorry

-- Problem 2
theorem problem2 (x : ℝ)
  (h1 : (2 / 3) * x + 5 > 1 - x)
  (h2 : x - 1 ≤ (3 / 4) * x - 1 / 8) :
  -12 / 5 < x ∧ x ≤ 7 / 2 := 
by
  sorry

end problem1_problem2_l144_144431


namespace intersection_A_B_l144_144209

-- Define set A
def A : Set Int := { x | x^2 - x - 2 ≤ 0 }

-- Define set B
def B : Set Int := { x | x < 1 }

-- Define the intersection set
def intersection_AB : Set Int := { -1, 0 }

-- Formalize the proof statement
theorem intersection_A_B : (A ∩ B) = intersection_AB :=
by sorry

end intersection_A_B_l144_144209


namespace measure_of_y_l144_144134

theorem measure_of_y (y : ℕ) (h₁ : 40 + 2 * y + y = 180) : y = 140 / 3 :=
by
  sorry

end measure_of_y_l144_144134


namespace range_of_a_l144_144390

theorem range_of_a (a : ℝ) : 
  (∀ n : ℕ, n ≥ 8 → (a * (n^2) + n + 5) > (a * ((n + 1)^2) + (n + 1) + 5)) → 
  (a * (1^2) + 1 + 5 < a * (2^2) + 2 + 5) →
  (a * (2^2) + 2 + 5 < a * (3^2) + 3 + 5) →
  (a * (3^2) + 3 + 5 < a * (4^2) + 4 + 5) →
  (- (1 / 7) < a ∧ a < - (1 / 17)) :=
by
  sorry

end range_of_a_l144_144390


namespace max_value_xyz_l144_144253

theorem max_value_xyz (x y z : ℝ) (h : x + y + 2 * z = 5) : 
  (∃ x y z : ℝ, x + y + 2 * z = 5 ∧ xy + xz + yz = 25/6) :=
sorry

end max_value_xyz_l144_144253


namespace rectangular_block_height_l144_144657

theorem rectangular_block_height (l w h : ℕ) 
  (volume_eq : l * w * h = 42) 
  (perimeter_eq : 2 * l + 2 * w = 18) : 
  h = 3 :=
by
  sorry

end rectangular_block_height_l144_144657


namespace common_ratio_of_geometric_series_l144_144816

theorem common_ratio_of_geometric_series (a b : ℚ) (h1 : a = 4 / 7) (h2 : b = 16 / 21) :
  b / a = 4 / 3 :=
by
  sorry

end common_ratio_of_geometric_series_l144_144816


namespace revenue_fraction_l144_144000

variable (N D J : ℝ)
variable (h1 : J = 1 / 5 * N)
variable (h2 : D = 4.166666666666666 * (N + J) / 2)

theorem revenue_fraction (h1 : J = 1 / 5 * N) (h2 : D = 4.166666666666666 * (N + J) / 2) : N / D = 2 / 5 :=
by
  sorry

end revenue_fraction_l144_144000


namespace dice_probability_ne_zero_l144_144289

theorem dice_probability_ne_zero :
  let outcomes := {[1, 2, 3, 4, 5, 6]} in
  ∃ (a b c d : ℕ) (h1 : a ∈ outcomes) (h2 : b ∈ outcomes) (h3 : c ∈ outcomes) (h4 : d ∈ outcomes),
  ((a - 1) * (b - 1) * (c - 1) * (d - 1) ≠ 0) →
  (prob_of_event := (5/6)^4) →
  prob_of_event = 625 / 1296 := 
sorry

end dice_probability_ne_zero_l144_144289


namespace no_triangle_with_perfect_square_sides_l144_144339

theorem no_triangle_with_perfect_square_sides :
  ∃ (a b : ℕ), a > 1000 ∧ b > 1000 ∧
    ∀ (c : ℕ), (∃ d : ℕ, c = d^2) → 
    ¬ (a + b > c ∧ b + c > a ∧ a + c > b) :=
sorry

end no_triangle_with_perfect_square_sides_l144_144339


namespace total_trees_l144_144023

-- Definitions based on the conditions
def ava_trees : ℕ := 9
def lily_trees : ℕ := ava_trees - 3

-- Theorem stating the total number of apple trees planted by Ava and Lily
theorem total_trees : ava_trees + lily_trees = 15 := by
  -- We skip the proof for now
  sorry

end total_trees_l144_144023


namespace evaluate_expression_l144_144202

theorem evaluate_expression (x : ℝ) :
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3 * x^3 - 5 * x^2 + 12 * x + 2 :=
by
  sorry

end evaluate_expression_l144_144202


namespace equilateral_triangle_side_length_l144_144443

theorem equilateral_triangle_side_length (total_length : ℕ) (h1 : total_length = 78) : (total_length / 3) = 26 :=
by
  sorry

end equilateral_triangle_side_length_l144_144443


namespace number_of_freshmen_to_sample_l144_144315

-- Define parameters
def total_students : ℕ := 900
def sample_size : ℕ := 45
def freshmen_count : ℕ := 400
def sophomores_count : ℕ := 300
def juniors_count : ℕ := 200

-- Define the stratified sampling calculation
def stratified_sampling_calculation (group_size : ℕ) (total_size : ℕ) (sample_size : ℕ) : ℕ :=
  (group_size * sample_size) / total_size

-- Theorem stating that the number of freshmen to be sampled is 20
theorem number_of_freshmen_to_sample : stratified_sampling_calculation freshmen_count total_students sample_size = 20 := by
  sorry

end number_of_freshmen_to_sample_l144_144315


namespace combined_weight_l144_144702

-- Given constants
def JakeWeight : ℕ := 198
def WeightLost : ℕ := 8
def KendraWeight := (JakeWeight - WeightLost) / 2

-- Prove the combined weight of Jake and Kendra
theorem combined_weight : JakeWeight + KendraWeight = 293 := by
  sorry

end combined_weight_l144_144702


namespace tangent_line_eq_extreme_values_range_of_a_l144_144216

noncomputable def f (x : ℝ) (a: ℝ) : ℝ := x^2 - a * Real.log x

-- (I) Proving the tangent line equation is y = x for a = 1 at x = 1.
theorem tangent_line_eq (h : ∀ x, f x 1 = x^2 - Real.log x) :
  ∃ y : (ℝ → ℝ), y = id ∧ y 1 = x :=
sorry

-- (II) Proving extreme values of the function f(x).
theorem extreme_values (a: ℝ) :
  (∃ x_min : ℝ, f x_min a = (a/2) - (a/2) * Real.log (a/2)) ∧ 
  (∀ x, ¬∃ x_max : ℝ, f x_max a > f x a) :=
sorry

-- (III) Proving the range of values for a.
theorem range_of_a :
  (∀ x, 2*x - (a/x) ≥ 0 → 2 < x) → a ≤ 8 :=
sorry

end tangent_line_eq_extreme_values_range_of_a_l144_144216


namespace polynomial_discriminant_l144_144375

theorem polynomial_discriminant (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
by
  sorry

end polynomial_discriminant_l144_144375


namespace rectangle_perimeter_eq_30sqrt10_l144_144016

theorem rectangle_perimeter_eq_30sqrt10 (A : ℝ) (l : ℝ) (w : ℝ) 
  (hA : A = 500) (hlw : l = 2 * w) (hArea : A = l * w) : 
  2 * (l + w) = 30 * Real.sqrt 10 :=
by
  sorry

end rectangle_perimeter_eq_30sqrt10_l144_144016


namespace find_surface_area_of_ball_l144_144382

noncomputable def surface_area_of_ball : ℝ :=
  let tetrahedron_edge := 4
  let tetrahedron_volume := (1 / 3) * (Real.sqrt 3 / 4) * tetrahedron_edge ^ 2 * (Real.sqrt 16 - 16 / 3)
  let water_volume := (7 / 8) * tetrahedron_volume
  let remaining_volume := (1 / 8) * tetrahedron_volume
  let remaining_edge := 2
  let ball_radius := Real.sqrt 6 / 6
  let surface_area := 4 * Real.pi * ball_radius ^ 2
  surface_area

theorem find_surface_area_of_ball :
  let tetrahedron_edge := 4
  let tetrahedron_volume := (1 / 3) * (Real.sqrt 3 / 4) * tetrahedron_edge ^ 2 * (Real.sqrt 16 - 16 / 3)
  let water_volume := (7 / 8) * tetrahedron_volume
  let remaining_volume := (1 / 8) * tetrahedron_volume
  let remaining_edge := 2
  let ball_radius := Real.sqrt 6 / 6
  let surface_area := 4 * Real.pi * ball_radius ^ 2
  surface_area = (2 / 3) * Real.pi :=
by
  let tetrahedron_edge := 4
  let tetrahedron_volume := (1 / 3) * (Real.sqrt 3 / 4) * tetrahedron_edge ^ 2 * (Real.sqrt 16 - 16 / 3)
  let water_volume := (7 / 8) * tetrahedron_volume
  let remaining_volume := (1 / 8) * tetrahedron_volume
  let remaining_edge := 2
  let ball_radius := Real.sqrt 6 / 6
  let surface_area := 4 * Real.pi * ball_radius ^ 2
  sorry

end find_surface_area_of_ball_l144_144382


namespace elmer_saves_21_875_percent_l144_144647

noncomputable def old_car_efficiency (x : ℝ) := x
noncomputable def new_car_efficiency (x : ℝ) := 1.6 * x

noncomputable def gasoline_cost (c : ℝ) := c
noncomputable def diesel_cost (c : ℝ) := 1.25 * c

noncomputable def trip_distance := 1000

noncomputable def old_car_fuel_consumption (x : ℝ) := trip_distance / x
noncomputable def new_car_fuel_consumption (x : ℝ) := trip_distance / (new_car_efficiency x)

noncomputable def old_car_trip_cost (x c : ℝ) := (trip_distance / x) * c
noncomputable def new_car_trip_cost (x c : ℝ) := (trip_distance / (new_car_efficiency x)) * (diesel_cost c)

noncomputable def savings (x c : ℝ) := old_car_trip_cost x c - new_car_trip_cost x c
noncomputable def percentage_savings (x c : ℝ) := (savings x c) / (old_car_trip_cost x c) * 100

theorem elmer_saves_21_875_percent (x c : ℝ) : percentage_savings x c = 21.875 := 
sorry

end elmer_saves_21_875_percent_l144_144647


namespace total_charge_for_2_hours_l144_144597

theorem total_charge_for_2_hours (F A : ℕ) 
  (h1 : F = A + 40) 
  (h2 : F + 4 * A = 375) : 
  F + A = 174 :=
by 
  sorry

end total_charge_for_2_hours_l144_144597


namespace smallest_number_divisible_l144_144593

theorem smallest_number_divisible (x : ℕ) :
  (∃ n : ℕ, x = n * 5 + 24) ∧
  (∃ n : ℕ, x = n * 10 + 24) ∧
  (∃ n : ℕ, x = n * 15 + 24) ∧
  (∃ n : ℕ, x = n * 20 + 24) →
  x = 84 :=
by
  sorry

end smallest_number_divisible_l144_144593


namespace price_of_other_frisbees_l144_144018

theorem price_of_other_frisbees 
  (P : ℝ) 
  (x : ℝ)
  (h1 : x + (64 - x) = 64)
  (h2 : P * x + 4 * (64 - x) = 196)
  (h3 : 64 - x ≥ 4) 
  : P = 3 :=
sorry

end price_of_other_frisbees_l144_144018


namespace find_f_1789_l144_144769

-- Given conditions as definitions

def f : ℕ → ℕ := sorry
axiom f_a (n : ℕ) (hn : n > 0) : f(f(n)) = 4 * n + 9
axiom f_b (k : ℕ) : f(2^k) = 2^(k+1) + 3

-- The theorem to prove f(1789) = 3581 given the conditions
theorem find_f_1789 : f(1789) = 3581 := sorry

end find_f_1789_l144_144769


namespace calculate_f_g2_l144_144065

def f (x : ℝ) : ℝ := x^2 + 1
def g (x : ℝ) : ℝ := 2 * x^3 - 1

theorem calculate_f_g2 : f (g 2) = 226 := by
  sorry

end calculate_f_g2_l144_144065


namespace three_g_two_plus_two_g_neg_four_l144_144490

def g (x : ℝ) : ℝ := 2 * x ^ 2 - 2 * x + 11

theorem three_g_two_plus_two_g_neg_four : 3 * g 2 + 2 * g (-4) = 147 := by
  sorry

end three_g_two_plus_two_g_neg_four_l144_144490


namespace coeff_a_zero_l144_144230

-- Define the problem in Lean 4

theorem coeff_a_zero (a b c : ℝ) (h : ∀ p : ℝ, 0 < p → ∀ x, a * x^2 + b * x + c + p = 0 → 0 < x) :
  a = 0 :=
sorry

end coeff_a_zero_l144_144230


namespace minimize_sum_pos_maximize_product_pos_l144_144579

def N : ℕ := 10^1001 - 1

noncomputable def find_min_sum_position : ℕ := 996

noncomputable def find_max_product_position : ℕ := 995

theorem minimize_sum_pos :
  ∀ m : ℕ, (m ≠ find_min_sum_position) → 
      (2 * 10^m + 10^(1001-m) - 10) ≥ (2 * 10^find_min_sum_position + 10^(1001-find_min_sum_position) - 10) := 
sorry

theorem maximize_product_pos :
  ∀ m : ℕ, (m ≠ find_max_product_position) → 
      ((2 * 10^m - 1) * (10^(1001 - m) - 9)) ≤ ((2 * 10^find_max_product_position - 1) * (10^(1001 - find_max_product_position) - 9)) :=
sorry

end minimize_sum_pos_maximize_product_pos_l144_144579


namespace kyle_car_payment_l144_144408

theorem kyle_car_payment (income rent utilities retirement groceries insurance miscellaneous gas x : ℕ)
  (h_income : income = 3200)
  (h_rent : rent = 1250)
  (h_utilities : utilities = 150)
  (h_retirement : retirement = 400)
  (h_groceries : groceries = 300)
  (h_insurance : insurance = 200)
  (h_miscellaneous : miscellaneous = 200)
  (h_gas : gas = 350)
  (h_expenses : rent + utilities + retirement + groceries + insurance + miscellaneous + gas + x = income) :
  x = 350 :=
by sorry

end kyle_car_payment_l144_144408


namespace range_c_of_sets_l144_144220

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem range_c_of_sets (c : ℝ) (h₀ : c > 0)
  (A := { x : ℝ | log2 x < 1 })
  (B := { x : ℝ | 0 < x ∧ x < c })
  (hA_union_B_eq_B : A ∪ B = B) :
  c ≥ 2 :=
by
  -- Minimum outline is provided, the proof part is replaced with "sorry" to indicate the point to be proved
  sorry

end range_c_of_sets_l144_144220


namespace John_distance_proof_l144_144081

def initial_running_time : ℝ := 8
def increase_percentage : ℝ := 0.75
def initial_speed : ℝ := 8
def speed_increase : ℝ := 4

theorem John_distance_proof : 
  (initial_running_time + initial_running_time * increase_percentage) * (initial_speed + speed_increase) = 168 := 
by
  -- Proof can be completed here
  sorry

end John_distance_proof_l144_144081


namespace max_and_min_A_l144_144602

noncomputable def B := {B : ℕ // B > 22222222 ∧ gcd B 18 = 1}
noncomputable def A (B : B) : ℕ := 10^8 * ((B.val % 10)) + (B.val / 10)

noncomputable def A_max := 999999998
noncomputable def A_min := 122222224

theorem max_and_min_A : 
  (∃ B : B, A B = A_max) ∧ (∃ B : B, A B = A_min) := sorry

end max_and_min_A_l144_144602


namespace remainder_3_302_plus_302_div_by_3_151_plus_3_101_plus_1_l144_144764

-- Definitions from the conditions
def a : ℕ := 3^302
def b : ℕ := 3^151 + 3^101 + 1

-- Theorem: Prove that the remainder when a + 302 is divided by b is 302.
theorem remainder_3_302_plus_302_div_by_3_151_plus_3_101_plus_1 :
  (a + 302) % b = 302 :=
by {
  sorry
}

end remainder_3_302_plus_302_div_by_3_151_plus_3_101_plus_1_l144_144764


namespace percent_decrease_of_y_l144_144433

theorem percent_decrease_of_y (k x y q : ℝ) (h_inv_prop : x * y = k) (h_pos : 0 < x ∧ 0 < y) (h_q : 0 < q) :
  let x' := x * (1 + q / 100)
  let y' := y * 100 / (100 + q)
  (y - y') / y * 100 = (100 * q) / (100 + q) :=
by
  sorry

end percent_decrease_of_y_l144_144433


namespace cones_slant_height_angle_l144_144285

theorem cones_slant_height_angle :
  ∀ (α: ℝ),
  α = 2 * Real.arccos (Real.sqrt (2 / (2 + Real.sqrt 2))) :=
by
  sorry

end cones_slant_height_angle_l144_144285


namespace no_real_roots_iff_k_gt_1_div_4_l144_144396

theorem no_real_roots_iff_k_gt_1_div_4 (k : ℝ) :
  (∀ x : ℝ, ¬ (x^2 - x + k = 0)) ↔ k > 1 / 4 :=
by
  sorry

end no_real_roots_iff_k_gt_1_div_4_l144_144396


namespace power_zero_equals_one_specific_case_l144_144585

theorem power_zero_equals_one 
    (a b : ℤ) 
    (h : a ≠ 0)
    (h2 : b ≠ 0) : 
    (a / b : ℚ) ^ 0 = 1 := 
by {
  sorry
}

-- Specific case
theorem specific_case : 
  ( ( (-123456789 : ℤ) / (9876543210 : ℤ) : ℚ ) ^ 0 = 1 ) := 
by {
  apply power_zero_equals_one;
  norm_num;
  sorry
}

end power_zero_equals_one_specific_case_l144_144585


namespace population_increase_l144_144505

theorem population_increase (k l m : ℝ) : 
  (1 + k/100) * (1 + l/100) * (1 + m/100) = 
  1 + (k + l + m)/100 + (k*l + k*m + l*m)/10000 + k*l*m/1000000 :=
by sorry

end population_increase_l144_144505


namespace probability_at_least_one_boy_and_girl_l144_144616

section
variable (n : ℕ) (p : ℚ)
-- Condition: Birth of a boy is equally likely as a girl (p = 1/2)
def equally_likely : ℚ := 1 / 2

-- Function to calculate the probability of all boys or all girls
def same_gender_probability (n : ℕ) (p : ℚ) : ℚ :=
p ^ n

/-- Theorem: The probability that among four children, there is at least one boy and one girl is 7/8. -/
theorem probability_at_least_one_boy_and_girl :
  same_gender_probability 4 equally_likely + same_gender_probability 4 equally_likely = (1 / 8) →
  1 - 1 / 8 = 7 / 8 :=
by
  sorry

end

end probability_at_least_one_boy_and_girl_l144_144616


namespace Ruby_apples_remaining_l144_144110

def Ruby_original_apples : ℕ := 6357912
def Emily_takes_apples : ℕ := 2581435
def Ruby_remaining_apples (R E : ℕ) : ℕ := R - E

theorem Ruby_apples_remaining : Ruby_remaining_apples Ruby_original_apples Emily_takes_apples = 3776477 := by
  sorry

end Ruby_apples_remaining_l144_144110


namespace find_value_of_M_l144_144671

theorem find_value_of_M (M : ℝ) (h : 0.2 * M = 0.6 * 1230) : M = 3690 :=
by {
  sorry
}

end find_value_of_M_l144_144671


namespace child_support_calculation_l144_144551

noncomputable def owed_child_support (yearly_salary : ℕ) (raise_pct: ℝ) 
(raise_years_additional_salary: ℕ) (payment_percentage: ℝ) 
(payment_years_salary_before_raise: ℕ) (already_paid : ℝ) : ℝ :=
  let initial_salary := yearly_salary * payment_years_salary_before_raise
  let increase_amount := yearly_salary * raise_pct
  let new_salary := yearly_salary + increase_amount
  let salary_after_raise := new_salary * raise_years_additional_salary
  let total_income := initial_salary + salary_after_raise
  let total_support_due := total_income * payment_percentage
  total_support_due - already_paid

theorem child_support_calculation:
  owed_child_support 30000 0.2 4 0.3 3 1200 = 69000 :=
by
  sorry

end child_support_calculation_l144_144551


namespace roots_operation_zero_l144_144169

def operation (a b : ℝ) : ℝ := a * b - a - b

theorem roots_operation_zero {x1 x2 : ℝ}
  (h1 : x1 + x2 = -1)
  (h2 : x1 * x2 = -1) :
  operation x1 x2 = 0 :=
by
  sorry

end roots_operation_zero_l144_144169


namespace probability_at_least_one_boy_and_girl_l144_144614

section
variable (n : ℕ) (p : ℚ)
-- Condition: Birth of a boy is equally likely as a girl (p = 1/2)
def equally_likely : ℚ := 1 / 2

-- Function to calculate the probability of all boys or all girls
def same_gender_probability (n : ℕ) (p : ℚ) : ℚ :=
p ^ n

/-- Theorem: The probability that among four children, there is at least one boy and one girl is 7/8. -/
theorem probability_at_least_one_boy_and_girl :
  same_gender_probability 4 equally_likely + same_gender_probability 4 equally_likely = (1 / 8) →
  1 - 1 / 8 = 7 / 8 :=
by
  sorry

end

end probability_at_least_one_boy_and_girl_l144_144614


namespace expand_product_l144_144648

theorem expand_product (x : ℝ) : (x + 5) * (x + 9) = x^2 + 14 * x + 45 :=
by
  sorry

end expand_product_l144_144648


namespace find_function_expression_l144_144051

theorem find_function_expression (f : ℝ → ℝ) : 
  (∀ x : ℝ, f (x - 1) = x^2 - 3 * x) → 
  (∀ x : ℝ, f x = x^2 - x - 2) :=
by
  sorry

end find_function_expression_l144_144051


namespace quadratic_discriminant_l144_144357

variable {a b c : ℝ}
variable (h₁ : a ≠ 0)
variable (h₂ : (b - 1)^2 - 4 * a * (c + 2) = 0)
variable (h₃ : (b + 1/2)^2 - 4 * a * (c - 1) = 0)

theorem quadratic_discriminant : b^2 - 4 * a * c = -1 / 2 := 
by
  have h₁' : (b - 1)^2 - 4 * a * (c + 2) = 0 := h₂
  have h₂' : (b + 1/2)^2 - 4 * a * (c - 1) = 0 := h₃
  sorry

end quadratic_discriminant_l144_144357


namespace total_apples_picked_l144_144997

-- Define the number of apples picked by Benny
def applesBenny : Nat := 2

-- Define the number of apples picked by Dan
def applesDan : Nat := 9

-- The theorem we want to prove
theorem total_apples_picked : applesBenny + applesDan = 11 := 
by 
  sorry

end total_apples_picked_l144_144997


namespace median_mode_25_l144_144860

def points : List ℕ := [23, 25, 25, 23, 30, 27, 25]

def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (· ≤ ·)
  sorted.nth (sorted.length / 2) |>.getD 0

def mode (l : List ℕ) : ℕ :=
  l.foldl (λmap n => map.insert n (map.findD n 0 + 1)) (RBMap .toNat (·≤·)) 
    |> .max |>.fst

theorem median_mode_25 : (median points = 25) ∧ (mode points = 25) := by
  sorry

end median_mode_25_l144_144860


namespace fence_poles_placement_l144_144153

def total_bridges_length (bridges : List ℕ) : ℕ :=
  bridges.sum

def effective_path_length (path_length : ℕ) (bridges_length : ℕ) : ℕ :=
  path_length - bridges_length

def poles_on_one_side (effective_length : ℕ) (interval : ℕ) : ℕ :=
  effective_length / interval

def total_poles (path_length : ℕ) (interval : ℕ) (bridges : List ℕ) : ℕ :=
  let bridges_length := total_bridges_length bridges
  let effective_length := effective_path_length path_length bridges_length
  let poles_one_side := poles_on_one_side effective_length interval
  2 * poles_one_side + 2

theorem fence_poles_placement :
  total_poles 2300 8 [48, 58, 62] = 534 := by
  sorry

end fence_poles_placement_l144_144153


namespace cost_effective_plan1_l144_144610

/-- 
Plan 1 involves purchasing a 80 yuan card and a subsequent fee of 10 yuan per session.
Plan 2 involves a fee of 20 yuan per session without purchasing the card.
We want to prove that Plan 1 is more cost-effective than Plan 2 for any number of sessions x > 8.
-/
theorem cost_effective_plan1 (x : ℕ) (h : x > 8) : 
  10 * x + 80 < 20 * x :=
sorry

end cost_effective_plan1_l144_144610


namespace evaluate_f_at_2_l144_144964

def f (x : ℝ) : ℝ := x^2 - 3 * x

theorem evaluate_f_at_2 :
  f 2 = -2 :=
by
  sorry

end evaluate_f_at_2_l144_144964


namespace value_of_f_at_2_l144_144966

def f (x : ℝ) : ℝ := x^2 - 3 * x

theorem value_of_f_at_2 : f 2 = -2 := 
by 
  sorry

end value_of_f_at_2_l144_144966


namespace triangle_construction_feasible_l144_144222

theorem triangle_construction_feasible (a b s : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (a - b) / 2 < s) (h4 : s < (a + b) / 2) :
  ∃ c, (a + b > c ∧ b + c > a ∧ c + a > b) :=
sorry

end triangle_construction_feasible_l144_144222


namespace perimeter_of_square_l144_144119

-- Given conditions
variables {x y : ℝ} (h1 : x - y = 5) (h2 : x * y > 0)

theorem perimeter_of_square (h : (∃ s : ℝ, s^2 = 5 * (x * y))) : 
  ∃ p : ℝ, p = 4 * Real.sqrt (5 * x * y) :=
by
  obtain ⟨s, hs⟩ := h
  use 4 * s
  rw hs
  congr
  field_simp [Real.sqrt_mul (by norm_num : (5 : ℝ)) (x * y)]
  sorry

end perimeter_of_square_l144_144119


namespace famous_figures_mathematicians_l144_144021

-- List of figures encoded as integers for simplicity
def Bill_Gates := 1
def Gauss := 2
def Liu_Xiang := 3
def Nobel := 4
def Chen_Jingrun := 5
def Chen_Xingshen := 6
def Gorky := 7
def Einstein := 8

-- Set of mathematicians encoded as a set of integers
def mathematicians : Set ℕ := {2, 5, 6}

-- Correct answer set
def correct_answer_set : Set ℕ := {2, 5, 6}

-- The statement to prove
theorem famous_figures_mathematicians:
  mathematicians = correct_answer_set :=
by sorry

end famous_figures_mathematicians_l144_144021


namespace arithmetic_square_root_of_nine_l144_144928

theorem arithmetic_square_root_of_nine :
  real.sqrt 9 = 3 :=
by
  sorry

end arithmetic_square_root_of_nine_l144_144928


namespace quadratic_polynomial_discriminant_l144_144380

def P (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₀ : a ≠ 0)
  (h₁ : ∃ x : ℝ, P a b c x = x - 2 ∧ (discriminant a (b - 1) (c + 2) = 0))
  (h₂ : ∃ x : ℝ, P a b c x = 1 - x / 2 ∧ (discriminant a (b + 1 / 2) (c - 1) = 0)) :
  discriminant a b c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l144_144380


namespace sugar_solution_sweeter_l144_144607

theorem sugar_solution_sweeter (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) : 
    (b + m) / (a + m) > b / a :=
sorry

end sugar_solution_sweeter_l144_144607


namespace three_pow_mul_l144_144696

theorem three_pow_mul (a b : ℕ) (h_a : a = 12) (h_b : b = 18) :
  3^a * 3^b = 243^6 := by
  rw [h_a, h_b]
  calc
    3^12 * 3^18
      = 3^(12 + 18) : by rw [pow_add]
  ... = 3^30 : by norm_num
  ... = (3^5)^6 : by rw [pow_mul, ← mul_comm]
  ... = 243^6 : by norm_num

end three_pow_mul_l144_144696


namespace parallel_lines_slope_l144_144398

theorem parallel_lines_slope (a : ℝ) : 
  (∀ x y : ℝ, ax + 2 * y + 1 = 0 → ∀ x y : ℝ, x + y - 2 = 0 → True) → 
  a = 2 :=
by
  sorry

end parallel_lines_slope_l144_144398


namespace arithmetic_sqrt_of_nine_l144_144920

-- Define the arithmetic square root function which only considers non-negative values
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ :=
  if hx : x ≥ 0 then Real.sqrt x else 0

-- The theorem to prove: The arithmetic square root of 9 is 3.
theorem arithmetic_sqrt_of_nine : arithmetic_sqrt 9 = 3 :=
by
  sorry

end arithmetic_sqrt_of_nine_l144_144920


namespace shoes_total_price_l144_144246

-- Define the variables involved
variables (S J : ℝ)

-- Define the conditions
def condition1 : Prop := J = (1 / 4) * S
def condition2 : Prop := 6 * S + 4 * J = 560

-- Define the total price calculation
def total_price : ℝ := 6 * S

-- State the theorem and proof goal
theorem shoes_total_price (h1 : condition1 S J) (h2 : condition2 S J) : total_price S = 480 := 
sorry

end shoes_total_price_l144_144246


namespace sum_of_numbers_l144_144071

theorem sum_of_numbers (a b c : ℝ) (h1 : 2 * a + b = 46) (h2 : b + 2 * c = 53) (h3 : 2 * c + a = 29) :
  a + b + c = 48.8333 :=
by
  sorry

end sum_of_numbers_l144_144071


namespace intersection_when_a_eq_4_range_for_A_subset_B_l144_144064

-- Define the conditions
def setA : Set ℝ := { x | (1 - x) / (x - 7) > 0 }
def setB (a : ℝ) : Set ℝ := { x | x^2 - 2 * x - a^2 - 2 * a < 0 }

-- First proof goal: When a = 4, find A ∩ B
theorem intersection_when_a_eq_4 :
  setA ∩ (setB 4) = { x : ℝ | 1 < x ∧ x < 6 } :=
sorry

-- Second proof goal: Find the range for a such that A ⊆ B
theorem range_for_A_subset_B :
  { a : ℝ | setA ⊆ setB a } = { a : ℝ | a ≤ -7 ∨ a ≥ 5 } :=
sorry

end intersection_when_a_eq_4_range_for_A_subset_B_l144_144064


namespace person_A_misses_at_least_once_in_4_shots_person_B_stops_after_5_shots_due_to_2_consecutive_misses_l144_144555

-- Define the probability of hitting the target for Person A and Person B
def p_hit_A : ℚ := 2 / 3
def p_hit_B : ℚ := 3 / 4

-- Define the complementary probabilities (missing the target)
def p_miss_A := 1 - p_hit_A
def p_miss_B := 1 - p_hit_B

-- Prove the probability that Person A, shooting 4 times, misses the target at least once
theorem person_A_misses_at_least_once_in_4_shots :
  (1 - (p_hit_A ^ 4)) = 65 / 81 :=
by 
  sorry

-- Prove the probability that Person B stops shooting exactly after 5 shots
-- due to missing the target consecutively 2 times
theorem person_B_stops_after_5_shots_due_to_2_consecutive_misses :
  (p_hit_B * p_hit_B * p_miss_B * (p_miss_B * p_miss_B)) = 45 / 1024 :=
by
  sorry

end person_A_misses_at_least_once_in_4_shots_person_B_stops_after_5_shots_due_to_2_consecutive_misses_l144_144555


namespace unpaintedRegionArea_l144_144128

def boardWidth1 : ℝ := 5
def boardWidth2 : ℝ := 7
def angle : ℝ := 45

theorem unpaintedRegionArea
  (bw1 bw2 angle : ℝ)
  (h1 : bw1 = boardWidth1)
  (h2 : bw2 = boardWidth2)
  (h3 : angle = 45) :
  let base := bw2 * Real.sqrt 2
  let height := bw1
  let area := base * height
  area = 35 * Real.sqrt 2 :=
by
  sorry

end unpaintedRegionArea_l144_144128


namespace ants_no_collision_probability_l144_144036

-- Definitions
def cube_vertices : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7}

def adjacent (v : ℕ) : Finset ℕ :=
  match v with
  | 0 => {1, 3, 4}
  | 1 => {0, 2, 5}
  | 2 => {1, 3, 6}
  | 3 => {0, 2, 7}
  | 4 => {0, 5, 7}
  | 5 => {1, 4, 6}
  | 6 => {2, 5, 7}
  | 7 => {3, 4, 6}
  | _ => ∅

-- Hypothesis: Each ant moves independently to one of the three adjacent vertices.

-- Result to prove
def X : ℕ := sorry  -- The number of valid ways ants can move without collisions

theorem ants_no_collision_probability : 
  ∃ X, (X / (3 : ℕ)^8 = X / 6561) :=
  by
    sorry

end ants_no_collision_probability_l144_144036


namespace angle_D_in_triangle_DEF_l144_144859

theorem angle_D_in_triangle_DEF 
  (E F D : ℝ) 
  (hEF : F = 3 * E) 
  (hE : E = 15) 
  (h_sum_angles : D + E + F = 180) : D = 120 :=
by
  -- Proof goes here
  sorry

end angle_D_in_triangle_DEF_l144_144859


namespace part1_part2_part3_l144_144388

noncomputable def f (x a : ℝ) : ℝ := x^2 + (x - 1) * |x - a|

-- Part 1
theorem part1 (a : ℝ) (x : ℝ) (h : a = -1) : 
  (f x a = 1) ↔ (x ≤ -1 ∨ x = 1) :=
sorry

-- Part 2
theorem part2 (a : ℝ) : 
  (∀ x y : ℝ, x < y → f x a < f y a) ↔ (a ≥ 1 / 3) :=
sorry

-- Part 3
theorem part3 (a : ℝ) (h1 : a < 1) (h2 : ∀ x : ℝ, f x a ≥ 2 * x - 3) : 
  -3 ≤ a ∧ a < 1 :=
sorry

end part1_part2_part3_l144_144388


namespace last_two_digits_of_7_pow_2016_l144_144732

theorem last_two_digits_of_7_pow_2016 : (7^2016 : ℕ) % 100 = 1 := 
by {
  sorry
}

end last_two_digits_of_7_pow_2016_l144_144732


namespace remainder_eval_at_4_l144_144471

def p : ℚ → ℚ := sorry

def r (x : ℚ) : ℚ := sorry

theorem remainder_eval_at_4 :
  (p 1 = 2) →
  (p 3 = 5) →
  (p (-2) = -2) →
  (∀ x, ∃ q : ℚ → ℚ, p x = (x - 1) * (x - 3) * (x + 2) * q x + r x) →
  r 4 = 38 / 7 :=
sorry

end remainder_eval_at_4_l144_144471


namespace three_exp_product_sixth_power_l144_144673

theorem three_exp_product_sixth_power :
  ∃ n : ℤ, 3^12 * 3^18 = n^6 ∧ n = 243 :=
by
  existsi 243
  split
  · sorry
  · refl

end three_exp_product_sixth_power_l144_144673


namespace ben_chairs_in_10_days_l144_144797

noncomputable def chairs_built_per_day (hours_per_shift : ℕ) (hours_per_chair : ℕ) : ℕ :=
  hours_per_shift / hours_per_chair

theorem ben_chairs_in_10_days 
  (hours_per_shift : ℕ)
  (hours_per_chair : ℕ)
  (days: ℕ)
  (h_shift: hours_per_shift = 8)
  (h_chair: hours_per_chair = 5)
  (h_days: days = 10) : 
  chairs_built_per_day hours_per_shift hours_per_chair * days = 10 :=
by 
  -- We insert a placeholder 'sorry' to be replaced by an actual proof.
  sorry

end ben_chairs_in_10_days_l144_144797


namespace exponent_product_to_sixth_power_l144_144688

theorem exponent_product_to_sixth_power :
  ∃ n : ℤ, 3^(12) * 3^(18) = n^6 ∧ n = 243 :=
by
  use 243
  sorry

end exponent_product_to_sixth_power_l144_144688


namespace abs_x_plus_one_ge_one_l144_144949

theorem abs_x_plus_one_ge_one {x : ℝ} : |x + 1| ≥ 1 ↔ x ≤ -2 ∨ x ≥ 0 :=
by
  sorry

end abs_x_plus_one_ge_one_l144_144949


namespace square_perimeter_l144_144588

def perimeter_of_square (side_length : ℝ) : ℝ :=
  4 * side_length

theorem square_perimeter (side_length : ℝ) (h : side_length = 5) : perimeter_of_square side_length = 20 := by
  sorry

end square_perimeter_l144_144588


namespace sum_of_a_and_b_l144_144698

variables {a b m : ℝ}

theorem sum_of_a_and_b (h1 : a^2 + a * b = 16 + m) (h2 : b^2 + a * b = 9 - m) : a + b = 5 ∨ a + b = -5 :=
by sorry

end sum_of_a_and_b_l144_144698


namespace quadratic_solution_l144_144383

theorem quadratic_solution (a : ℝ) (h : 2^2 - 3 * 2 + a = 0) : 2 * a - 1 = 3 :=
by {
  sorry
}

end quadratic_solution_l144_144383


namespace min_value_expression_l144_144632

variable {a b c : ℝ}

theorem min_value_expression (h1 : a < b) (h2 : a > 0) (h3 : b^2 - 4 * a * c ≤ 0) : 
  ∃ m : ℝ, m = 3 ∧ (∀ x : ℝ, ((a + b + c) / (b - a)) ≥ m) := 
sorry

end min_value_expression_l144_144632


namespace largest_sum_valid_set_l144_144092

-- Define the conditions for the set S
def valid_set (S : Finset ℕ) : Prop :=
  (∀ x ∈ S, 0 < x ∧ x ≤ 15) ∧
  ∀ (A B : Finset ℕ), A ⊆ S → B ⊆ S → A ≠ B → A ∩ B = ∅ → A.sum id ≠ B.sum id

-- The theorem stating the largest sum of such a set
theorem largest_sum_valid_set : ∃ (S : Finset ℕ), valid_set S ∧ S.sum id = 61 :=
sorry

end largest_sum_valid_set_l144_144092


namespace total_apple_trees_l144_144025

-- Definitions and conditions
def ava_trees : ℕ := 9
def lily_trees : ℕ := ava_trees - 3
def total_trees : ℕ := ava_trees + lily_trees

-- Statement to be proved
theorem total_apple_trees :
  total_trees = 15 := by
  sorry

end total_apple_trees_l144_144025


namespace arithmetic_square_root_of_9_l144_144918

theorem arithmetic_square_root_of_9 : Real.sqrt 9 = 3 := by
  sorry

end arithmetic_square_root_of_9_l144_144918


namespace cirrus_clouds_count_l144_144757

theorem cirrus_clouds_count 
  (cirrus_clouds cumulus_clouds cumulonimbus_clouds : ℕ)
  (h1 : cirrus_clouds = 4 * cumulus_clouds)
  (h2 : cumulus_clouds = 12 * cumulonimbus_clouds)
  (h3 : cumulonimbus_clouds = 3) : 
  cirrus_clouds = 144 :=
by sorry

end cirrus_clouds_count_l144_144757


namespace simplify_product_of_fractions_l144_144428

theorem simplify_product_of_fractions :
  (25 / 24) * (18 / 35) * (56 / 45) = (50 / 3) :=
by sorry

end simplify_product_of_fractions_l144_144428


namespace arithmetic_sequence_value_l144_144062

theorem arithmetic_sequence_value :
  ∀ (a_n : ℕ → ℤ) (d : ℤ),
    (∀ n : ℕ, a_n n = a_n 0 + ↑n * d) →
    a_n 2 = 4 →
    a_n 4 = 8 →
    a_n 10 = 20 :=
by
  intros a_n d h_arith h_a3 h_a5
  --
  sorry

end arithmetic_sequence_value_l144_144062


namespace arithmetic_sqrt_of_9_l144_144936

def arithmetic_sqrt (n : ℕ) : ℕ :=
  Nat.sqrt n

theorem arithmetic_sqrt_of_9 : arithmetic_sqrt 9 = 3 :=
by
  sorry

end arithmetic_sqrt_of_9_l144_144936


namespace not_divisible_by_n_only_prime_3_l144_144149

-- Problem 1: Prove that for any natural number \( n \) greater than 1, \( 2^n - 1 \) is not divisible by \( n \)
theorem not_divisible_by_n (n : ℕ) (h1 : 1 < n) : ¬ (n ∣ (2^n - 1)) :=
sorry

-- Problem 2: Prove that the only prime number \( n \) such that \( 2^n + 1 \) is divisible by \( n^2 \) is \( n = 3 \)
theorem only_prime_3 (n : ℕ) (hn : Nat.Prime n) (hdiv : n^2 ∣ (2^n + 1)) : n = 3 :=
sorry

end not_divisible_by_n_only_prime_3_l144_144149


namespace f_decreasing_on_neg_infty_2_l144_144272

def f (x : ℝ) := x^2 - 4 * x + 3

theorem f_decreasing_on_neg_infty_2 :
  ∀ x y : ℝ, x < y → y ≤ 2 → f y < f x :=
by
  sorry

end f_decreasing_on_neg_infty_2_l144_144272


namespace frobenius_two_vars_l144_144409

theorem frobenius_two_vars (a b n : ℤ) (ha : 0 < a) (hb : 0 < b) (hgcd : Int.gcd a b = 1) (hn : n > a * b - a - b) :
  ∃ x y : ℕ, n = a * x + b * y :=
by
  sorry

end frobenius_two_vars_l144_144409


namespace p_squared_plus_41_composite_for_all_primes_l144_144741

theorem p_squared_plus_41_composite_for_all_primes (p : ℕ) (hp : Prime p) : 
  ∃ d : ℕ, d > 1 ∧ d < p^2 + 41 ∧ d ∣ (p^2 + 41) :=
by
  sorry

end p_squared_plus_41_composite_for_all_primes_l144_144741


namespace arithmetic_expression_evaluation_l144_144030

theorem arithmetic_expression_evaluation :
  (1 / 6 * -6 / (-1 / 6) * 6) = 36 :=
by {
  sorry
}

end arithmetic_expression_evaluation_l144_144030


namespace equal_distribution_arithmetic_sequence_l144_144147

theorem equal_distribution_arithmetic_sequence :
  ∃ a d : ℚ, (a - 2 * d) + (a - d) = (a + (a + d) + (a + 2 * d)) ∧
  5 * a = 5 ∧
  a + 2 * d = 2 / 3 :=
by
  sorry

end equal_distribution_arithmetic_sequence_l144_144147


namespace problem_statement_l144_144841

noncomputable def universal_set : Set ℤ := {x : ℤ | x^2 - 5*x - 6 < 0 }

def A : Set ℤ := {x : ℤ | -1 < x ∧ x ≤ 2 }

def B : Set ℤ := {2, 3, 5}

def complement_U_A : Set ℤ := {x : ℤ | x ∈ universal_set ∧ ¬(x ∈ A)}

theorem problem_statement : 
  (complement_U_A ∩ B) = {3, 5} :=
by 
  sorry

end problem_statement_l144_144841


namespace attendance_calculation_l144_144223

theorem attendance_calculation (total_students : ℕ) (attendance_rate : ℚ)
  (h1 : total_students = 120)
  (h2 : attendance_rate = 0.95) :
  total_students * attendance_rate = 114 := 
  sorry

end attendance_calculation_l144_144223


namespace hours_per_day_initial_l144_144982

-- Definition of the problem and conditions
def initial_men : ℕ := 75
def depth1 : ℕ := 50
def additional_men : ℕ := 65
def total_men : ℕ := initial_men + additional_men
def depth2 : ℕ := 70
def hours_per_day2 : ℕ := 6
def work1 (H : ℝ) := initial_men * H * depth1
def work2 := total_men * hours_per_day2 * depth2

-- Statement to prove
theorem hours_per_day_initial (H : ℝ) (h1 : work1 H = work2) : H = 15.68 :=
by
  sorry

end hours_per_day_initial_l144_144982


namespace geometric_progression_first_term_l144_144278

theorem geometric_progression_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 8) 
  (h2 : a + a * r = 5) : 
  a = 2 * (4 - Real.sqrt 6) ∨ a = 2 * (4 + Real.sqrt 6) := 
by sorry

end geometric_progression_first_term_l144_144278


namespace green_eyed_snack_min_l144_144703

variable {total_count green_eyes_count snack_bringers_count : ℕ}

def least_green_eyed_snack_bringers (total_count green_eyes_count snack_bringers_count : ℕ) : ℕ :=
  green_eyes_count - (total_count - snack_bringers_count)

theorem green_eyed_snack_min 
  (h_total : total_count = 35)
  (h_green_eyes : green_eyes_count = 18)
  (h_snack_bringers : snack_bringers_count = 24)
  : least_green_eyed_snack_bringers total_count green_eyes_count snack_bringers_count = 7 :=
by
  rw [h_total, h_green_eyes, h_snack_bringers]
  unfold least_green_eyed_snack_bringers
  norm_num

end green_eyed_snack_min_l144_144703


namespace fraction_product_value_l144_144954

theorem fraction_product_value : 
  (4 / 5 : ℚ) * (5 / 6) * (6 / 7) * (7 / 8) * (8 / 9) = 4 / 9 :=
by
  sorry

end fraction_product_value_l144_144954


namespace combined_PPF_two_females_combined_PPF_two_males_combined_PPF_male_female_l144_144959

-- Definition: Combined PPF for two females
theorem combined_PPF_two_females (K : ℝ) (h : K ≤ 40) :
  (∀ K₁ K₂, K = K₁ + K₂ →  40 - 2 * K₁ + 40 - 2 * K₂ = 80 - 2 * K) := sorry

-- Definition: Combined PPF for two males
theorem combined_PPF_two_males (K : ℝ) (h : K ≤ 16) :
  (∀ K₁ K₂, K₁ = 0.5 * K → K₂ = 0.5 * K → 64 - K₁^2 + 64 - K₂^2 = 128 - 0.5 * K^2) := sorry

-- Definition: Combined PPF for one male and one female (piecewise)
theorem combined_PPF_male_female (K : ℝ) :
  (K ≤ 1 → (∀ K₁ K₂, K₁ = K → K₂ = 0 → 64 - K₁^2 + 40 - 2 * K₂ = 104 - K^2)) ∧
  (1 < K ∧ K ≤ 21 → (∀ K₁ K₂, K₁ = 1 → K₂ = K - 1 → 64 - K₁^2 + 40 - 2 * K₂ = 105 - 2 * K)) ∧
  (21 < K ∧ K ≤ 28 → (∀ K₁ K₂, K₁ = K - 20 → K₂ = 20 → 64 - K₁^2 + 40 - 2 * K₂ = 40 * K - K^2 - 336)) := sorry

end combined_PPF_two_females_combined_PPF_two_males_combined_PPF_male_female_l144_144959


namespace height_drawn_to_hypotenuse_l144_144566

-- Definitions for the given problem
variables {A B C D : Type}
variables {area : ℝ}
variables {angle_ratio : ℝ}
variables {h : ℝ}

-- Given conditions
def is_right_triangle (A B C : Type) : Prop := -- definition for the right triangle
sorry

def area_of_triangle (A B C : Type) (area: ℝ) : Prop := 
area = ↑(2 : ℝ) * Real.sqrt 3  -- area given as 2√3 cm²

def angle_bisector_ratios (A B C D : Type) (ratio: ℝ) : Prop :=
ratio = 1 / 2  -- given ratio 1:2

-- Question statement
theorem height_drawn_to_hypotenuse (A B C D : Type) 
  (right_triangle : is_right_triangle A B C)
  (area_cond : area_of_triangle A B C area)
  (angle_ratio_cond : angle_bisector_ratios A B C D angle_ratio):
  h = Real.sqrt 3 :=
sorry

end height_drawn_to_hypotenuse_l144_144566


namespace population_hypothetical_town_l144_144444

theorem population_hypothetical_town :
  ∃ (a b c : ℕ), a^2 + 150 = b^2 + 1 ∧ b^2 + 1 + 150 = c^2 ∧ a^2 = 5476 :=
by {
  sorry
}

end population_hypothetical_town_l144_144444


namespace tangential_quadrilateral_difference_l144_144990

-- Definitions of the conditions given in the problem
def is_cyclic_quadrilateral (a b c d : ℝ) : Prop := sorry -- In real setting, it means the quadrilateral vertices lie on a circle
def is_tangential_quadrilateral (a b c d : ℝ) : Prop := sorry -- In real setting, it means the sides are tangent to a common incircle
def point_tangency (a b c : ℝ) : Prop := sorry

-- Main theorem
theorem tangential_quadrilateral_difference (AB BC CD DA : ℝ) (x y : ℝ) 
  (h1 : is_cyclic_quadrilateral AB BC CD DA)
  (h2 : is_tangential_quadrilateral AB BC CD DA)
  (h3 : AB = 80) (h4 : BC = 140) (h5 : CD = 120) (h6 : DA = 100)
  (h7 : point_tangency x y CD)
  (h8 : x + y = 120) :
  |x - y| = 80 := 
sorry

end tangential_quadrilateral_difference_l144_144990


namespace factor_of_quadratic_implies_m_value_l144_144462

theorem factor_of_quadratic_implies_m_value (m : ℤ) : (∀ x : ℤ, (x + 6) ∣ (x^2 - m * x - 42)) → m = 1 := by
  sorry

end factor_of_quadratic_implies_m_value_l144_144462


namespace train_speed_l144_144994

theorem train_speed (length : ℝ) (time : ℝ) (h_length : length = 700) (h_time : time = 40) : length / time = 17.5 :=
by
  -- length / time represents the speed of the train
  -- given length = 700 meters and time = 40 seconds
  -- we have to prove that 700 / 40 = 17.5
  sorry

end train_speed_l144_144994


namespace simplify_expression_correct_l144_144429

noncomputable def simplify_expression (m n : ℝ) : ℝ :=
  ( (2 - n) / (n - 1) + 4 * ((m - 1) / (m - 2)) ) /
  ( n^2 * ((m - 1) / (n - 1)) + m^2 * ((2 - n) / (m - 2)) )

theorem simplify_expression_correct :
  simplify_expression (Real.rpow 400 (1/4)) (Real.sqrt 5) = (Real.sqrt 5) / 5 := 
sorry

end simplify_expression_correct_l144_144429


namespace marissa_tied_boxes_l144_144094

def Total_ribbon : ℝ := 4.5
def Leftover_ribbon : ℝ := 1
def Ribbon_per_box : ℝ := 0.7

theorem marissa_tied_boxes : (Total_ribbon - Leftover_ribbon) / Ribbon_per_box = 5 := by
  sorry

end marissa_tied_boxes_l144_144094


namespace arithmetic_square_root_of_9_l144_144915

theorem arithmetic_square_root_of_9 : Real.sqrt 9 = 3 := by
  sorry

end arithmetic_square_root_of_9_l144_144915


namespace roses_in_each_bouquet_l144_144312

theorem roses_in_each_bouquet (R : ℕ)
(roses_bouquets daisies_bouquets total_bouquets total_flowers daisies_per_bouquet total_daisies : ℕ)
(h1 : total_bouquets = 20)
(h2 : roses_bouquets = 10)
(h3 : daisies_bouquets = 10)
(h4 : total_flowers = 190)
(h5 : daisies_per_bouquet = 7)
(h6 : total_daisies = daisies_bouquets * daisies_per_bouquet)
(h7 : total_flowers - total_daisies = roses_bouquets * R) :
R = 12 :=
by
  sorry

end roses_in_each_bouquet_l144_144312


namespace quadratic_polynomial_discriminant_l144_144368

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₁ : a ≠ 0)
  (h₂ : ∃ x : ℝ, a * x^2 + b * x + c = x - 2 ∧ (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h₃ : ∃ x : ℝ, a * x^2 + b * x + c = 1 - x / 2 ∧ (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l144_144368


namespace kendalls_nickels_l144_144249

theorem kendalls_nickels :
  ∀ (n_quarters n_dimes n_nickels : ℕ),
  (n_quarters = 10) →
  (n_dimes = 12) →
  ((n_quarters * 25) + (n_dimes * 10) + (n_nickels * 5) = 400) →
  n_nickels = 6 :=
by
  intros n_quarters n_dimes n_nickels hq hd heq
  sorry

end kendalls_nickels_l144_144249


namespace max_area_triangle_AM_l144_144385

noncomputable def CircleE (a b : ℝ) : (ℝ → ℝ → Prop) :=
  λ x y, (x^2 / a^2) + (y^2 / b^2) = 1

def isEquilateralTriangle (A B C : ℝ × ℝ) (side_length : ℝ) : Prop :=
  dist A B = side_length ∧ dist B C = side_length ∧ dist C A = side_length

theorem max_area_triangle_AM :
  ∃ M A B : ℝ × ℝ, 
  (∀ a b : ℝ, b > 0 → CircleE a b M.1 M.2) →
  isEquilateralTriangle (0, √3) M A 2 →
  isEquilateralTriangle M A B 2 →
  (∃ AM BM AB : ℝ, AM ≠ 0 ∧ BM ≠ 0 ∧ AB ≠ 0 ∧ AM * BM < 0) →
  ∃ S : ℝ, S = sqrt 3 / 2 := sorry

end max_area_triangle_AM_l144_144385


namespace minimum_a_condition_l144_144055

theorem minimum_a_condition (a : ℝ) (h₀ : 0 < a) 
  (h₁ : ∀ x : ℝ, 1 < x → x + a / (x - 1) ≥ 5) :
  4 ≤ a :=
sorry

end minimum_a_condition_l144_144055


namespace range_of_a_l144_144515

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, |x - a| + |x + 1| ≤ 2) ↔ (a < -3 ∨ a > 1) :=
    sorry

end range_of_a_l144_144515


namespace evaluate_expression_l144_144029

theorem evaluate_expression : (-7)^3 / 7^2 - 4^4 + 5^2 = -238 := by
  sorry

end evaluate_expression_l144_144029


namespace original_number_is_correct_l144_144156

noncomputable def original_number : ℝ :=
  let x := 11.26666666666667
  let y := 30.333333333333332
  x + y

theorem original_number_is_correct (x y : ℝ) (h₁ : 10 * x + 22 * y = 780) (h₂ : y = 30.333333333333332) : 
  original_number = 41.6 :=
by
  sorry

end original_number_is_correct_l144_144156


namespace total_marks_math_physics_l144_144788

variables (M P C : ℕ)
axiom condition1 : C = P + 20
axiom condition2 : (M + C) / 2 = 45

theorem total_marks_math_physics : M + P = 70 :=
by sorry

end total_marks_math_physics_l144_144788


namespace megan_carrots_second_day_l144_144728

theorem megan_carrots_second_day : 
  ∀ (initial : ℕ) (thrown : ℕ) (total : ℕ) (second_day : ℕ),
  initial = 19 →
  thrown = 4 →
  total = 61 →
  second_day = (total - (initial - thrown)) →
  second_day = 46 :=
by
  intros initial thrown total second_day h_initial h_thrown h_total h_second_day
  rw [h_initial, h_thrown, h_total] at h_second_day
  sorry

end megan_carrots_second_day_l144_144728


namespace arithmetic_sqrt_of_nine_l144_144922

-- Define the arithmetic square root function which only considers non-negative values
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ :=
  if hx : x ≥ 0 then Real.sqrt x else 0

-- The theorem to prove: The arithmetic square root of 9 is 3.
theorem arithmetic_sqrt_of_nine : arithmetic_sqrt 9 = 3 :=
by
  sorry

end arithmetic_sqrt_of_nine_l144_144922


namespace proposition_equivalence_l144_144590

open Classical

theorem proposition_equivalence
  (p q : Prop) :
  ¬(p ∨ q) ↔ (¬p ∧ ¬q) :=
by sorry

end proposition_equivalence_l144_144590


namespace problem_solution_l144_144060

variable (f : ℝ → ℝ)

noncomputable def solution_set (x : ℝ) : Prop :=
  (0 < x ∧ x < 1/2) ∨ (2 < x)

theorem problem_solution
  (hf_even : ∀ x, f x = f (-x))
  (hf_decreasing : ∀ x y, x < y ∧ y ≤ 0 → f x > f y)
  (hf_at_1 : f 1 = 2) :
  ∀ x, f (Real.log x / Real.log 2) > 2 ↔ solution_set x :=
by
  sorry

end problem_solution_l144_144060


namespace minimum_waste_l144_144767

/-- Zenobia's cookout problem setup -/
def LCM_hot_dogs_buns : Nat := Nat.lcm 10 12

def hot_dog_packages : Nat := LCM_hot_dogs_buns / 10
def bun_packages : Nat := LCM_hot_dogs_buns / 12

def waste_hot_dog_packages : ℝ := hot_dog_packages * 0.4
def waste_bun_packages : ℝ := bun_packages * 0.3
def total_waste : ℝ := waste_hot_dog_packages + waste_bun_packages

theorem minimum_waste :
  hot_dog_packages = 6 ∧ bun_packages = 5 ∧ total_waste = 3.9 :=
by
  sorry

end minimum_waste_l144_144767


namespace percent_of_motorists_receive_speeding_tickets_l144_144423

theorem percent_of_motorists_receive_speeding_tickets
    (p_exceed : ℝ)
    (p_no_ticket : ℝ)
    (h1 : p_exceed = 0.125)
    (h2 : p_no_ticket = 0.20) : 
    (0.8 * p_exceed) * 100 = 10 :=
by
  sorry

end percent_of_motorists_receive_speeding_tickets_l144_144423


namespace difference_of_numbers_l144_144576

theorem difference_of_numbers (a b : ℕ) (h1 : a + b = 20460) (h2 : b % 12 = 0) (h3 : b / 10 = a) : b - a = 17314 :=
by
  sorry

end difference_of_numbers_l144_144576


namespace power_expression_l144_144680

theorem power_expression (a b : ℕ) (h1 : a = 12) (h2 : b = 18) : (3^a * 3^b) = (243^6) :=
by
  let c := 3
  have h3 : a + b = 30 := by simp [h1, h2]
  have h4 : 3^(a + b) = 3^30 := by rw [h3]
  have h5 : 3^30 = 243^6 := by norm_num
  sorry  -- skip other detailed steps

end power_expression_l144_144680


namespace david_english_marks_l144_144495

def david_marks (math physics chemistry biology avg : ℕ) : ℕ :=
  avg * 5 - (math + physics + chemistry + biology)

theorem david_english_marks :
  let math := 95
  let physics := 82
  let chemistry := 97
  let biology := 95
  let avg := 93
  david_marks math physics chemistry biology avg = 96 :=
by
  -- Proof is skipped
  sorry

end david_english_marks_l144_144495


namespace exponent_product_to_sixth_power_l144_144691

theorem exponent_product_to_sixth_power :
  ∃ n : ℤ, 3^(12) * 3^(18) = n^6 ∧ n = 243 :=
by
  use 243
  sorry

end exponent_product_to_sixth_power_l144_144691


namespace no_integer_solutions_l144_144427

theorem no_integer_solutions (a b c : ℤ) : ¬ (a^2 + b^2 = 8 * c + 6) :=
sorry

end no_integer_solutions_l144_144427


namespace wxyz_sum_l144_144086

noncomputable def wxyz (w x y z : ℕ) := 2^w * 3^x * 5^y * 7^z

theorem wxyz_sum (w x y z : ℕ) (h : wxyz w x y z = 1260) : w + 2 * x + 3 * y + 4 * z = 13 :=
sorry

end wxyz_sum_l144_144086


namespace neither_sufficient_nor_necessary_condition_l144_144393

theorem neither_sufficient_nor_necessary_condition
  (a1 b1 c1 a2 b2 c2 : ℝ) (h1 : a1 ≠ 0) (h2 : b1 ≠ 0) (h3 : c1 ≠ 0)
  (h4 : a2 ≠ 0) (h5 : b2 ≠ 0) (h6 : c2 ≠ 0) :
  (a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2) ↔
  ¬(∀ x, a1 * x^2 + b1 * x + c1 > 0 ↔ a2 * x^2 + b2 * x + c2 > 0) :=
sorry

end neither_sufficient_nor_necessary_condition_l144_144393


namespace rectangle_area_inscribed_circle_l144_144307

theorem rectangle_area_inscribed_circle (r l w : ℝ) (h_r : r = 7)
(h_ratio : l / w = 2) (h_w : w = 2 * r) :
  l * w = 392 :=
by sorry

end rectangle_area_inscribed_circle_l144_144307


namespace harold_millicent_books_l144_144667

theorem harold_millicent_books (H M : ℚ) 
  (h1 : H / 3 + M / 2 = 5 * M / 6) : H = M :=
by
  calc H = M : sorry

end harold_millicent_books_l144_144667


namespace absolute_difference_distance_l144_144736

/-- Renaldo drove 15 kilometers, Ernesto drove 7 kilometers more than one-third of Renaldo's distance, 
Marcos drove -5 kilometers. Prove that the absolute difference between the total distances driven by 
Renaldo and Ernesto combined, and the distance driven by Marcos is 22 kilometers. -/
theorem absolute_difference_distance :
  let renaldo_distance := 15
  let ernesto_distance := 7 + (1 / 3) * renaldo_distance
  let marcos_distance := -5
  abs ((renaldo_distance + ernesto_distance) - marcos_distance) = 22 := by
  sorry

end absolute_difference_distance_l144_144736


namespace sum_of_roots_of_quadratic_l144_144137

theorem sum_of_roots_of_quadratic (a b c : ℝ) (h_eq : a = 3 ∧ b = 6 ∧ c = -9) :
  (-b / a) = -2 :=
by
  rcases h_eq with ⟨ha, hb, hc⟩
  -- Proof goes here, but we can use sorry to skip it
  sorry

end sum_of_roots_of_quadratic_l144_144137


namespace set_properties_l144_144416

/-- Given sets A, B and C, prove the intersections and range conditions -/
theorem set_properties 
  (A : Set ℝ) (hA : A = {x | 2 < x ∧ x ≤ 3})
  (B : Set ℝ) (hB : B = {x | 1 < x ∧ x < 3})
  (C : ℝ → Set ℝ) (m : ℝ) (hC : C m = {x | m ≤ x}) :
  
  -- Part 1: Intersection of the complement of A with B.
  (Set.compl A ∩ B) = {x | 1 < x ∧ x ≤ 2} ∧  
  -- Part 2: Range condition for non-empty intersection of (A ∪ B) and C(m)
  (Set.Union A B ∩ C m).Nonempty ↔ m ≤ 3 :=
  
  begin
    sorry
  end

end set_properties_l144_144416


namespace problem_part_a_problem_part_b_l144_144785

def is_two_squared (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^2 ∧ a ≠ 0 ∧ b ≠ 0

def is_three_squared (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = a^2 + b^2 + c^2 ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

def is_four_squared (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), n = a^2 + b^2 + c^2 + d^2 ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

def satisfies_prime_conditions (e : ℕ) : Prop :=
  Nat.Prime (e - 2) ∧ Nat.Prime e ∧ Nat.Prime (e + 4)

def satisfies_square_sum_conditions (a b c d e : ℕ) : Prop :=
  a^2 + b^2 + c^2 + d^2 + e^2 = 2020 ∧ a < b ∧ b < c ∧ c < d ∧ d < e

theorem problem_part_a : is_two_squared 2020 ∧ is_three_squared 2020 ∧ is_four_squared 2020 := sorry

theorem problem_part_b : ∃ a b c d e : ℕ, satisfies_prime_conditions e ∧ satisfies_square_sum_conditions a b c d e :=
  sorry

end problem_part_a_problem_part_b_l144_144785


namespace smallest_solution_is_39_over_8_l144_144197

noncomputable def smallest_solution (x : ℝ) : Prop :=
  (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ (x ≠ 0) ∧ (x ≠ 3)

theorem smallest_solution_is_39_over_8 : ∃ x > 0, smallest_solution x ∧ x = 39 / 8 :=
by
  sorry

end smallest_solution_is_39_over_8_l144_144197


namespace probability_of_orange_face_l144_144742

theorem probability_of_orange_face :
  ∃ (G O P : ℕ) (total_faces : ℕ), total_faces = 10 ∧ G = 5 ∧ O = 3 ∧ P = 2 ∧
  (O / total_faces : ℚ) = 3 / 10 := by 
  sorry

end probability_of_orange_face_l144_144742


namespace product_of_numbers_l144_144125

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 26) (h2 : x - y = 8) : x * y = 153 :=
sorry

end product_of_numbers_l144_144125


namespace probability_at_least_one_boy_one_girl_l144_144620

noncomputable def probability_one_boy_one_girl : ℚ :=
  1 - (1 / 16) - (1 / 16)

theorem probability_at_least_one_boy_one_girl :
  probability_one_boy_one_girl = 7 / 8 := by
  sorry

end probability_at_least_one_boy_one_girl_l144_144620


namespace necessary_but_not_sufficient_condition_l144_144469

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (0 < a ∧ a ≤ 1) → (∀ x : ℝ, x^2 - 2*a*x + a > 0) :=
by
  sorry

end necessary_but_not_sufficient_condition_l144_144469


namespace machine_working_time_l144_144792

theorem machine_working_time (total_shirts_made : ℕ) (shirts_per_minute : ℕ)
  (h1 : total_shirts_made = 196) (h2 : shirts_per_minute = 7) :
  (total_shirts_made / shirts_per_minute = 28) :=
by
  sorry

end machine_working_time_l144_144792


namespace parabola_directrix_l144_144270

theorem parabola_directrix (y x : ℝ) (h : y = x^2) : 4 * y + 1 = 0 :=
sorry

end parabola_directrix_l144_144270


namespace complex_multiplication_l144_144518

def imaginary_unit := Complex.I

theorem complex_multiplication (h : imaginary_unit^2 = -1) : (3 + 2 * imaginary_unit) * imaginary_unit = -2 + 3 * imaginary_unit :=
by
  sorry

end complex_multiplication_l144_144518


namespace three_exp_product_sixth_power_l144_144675

theorem three_exp_product_sixth_power :
  ∃ n : ℤ, 3^12 * 3^18 = n^6 ∧ n = 243 :=
by
  existsi 243
  split
  · sorry
  · refl

end three_exp_product_sixth_power_l144_144675


namespace min_waiting_time_max_waiting_time_expected_waiting_time_l144_144302

open Nat

noncomputable def C : ℕ → ℕ → ℕ
| n, 0     => 1
| 0, k     => 0
| n+1, k+1 => C n k + C n (k+1)

def a := 1
def b := 5
def n := 5
def m := 3

def T_min := a * C (n - 1) 2 + m * n * a + b * C m 2
def T_max := a * C n 2 + b * m * n + b * C m 2
def E_T := C (n + m) 2 * (b * m + a * n) / (m + n)

theorem min_waiting_time : T_min = 40 := by
  sorry

theorem max_waiting_time : T_max = 100 := by
  sorry

theorem expected_waiting_time : E_T = 70 := by
  sorry

end min_waiting_time_max_waiting_time_expected_waiting_time_l144_144302


namespace planes_parallel_if_perpendicular_to_same_line_l144_144594

variables {Point : Type} {Line : Type} {Plane : Type} 

-- Definitions and conditions
noncomputable def is_parallel (α β : Plane) : Prop := sorry
noncomputable def is_perpendicular (l : Line) (α : Plane) : Prop := sorry

variables (l1 : Line) (α β : Plane)

theorem planes_parallel_if_perpendicular_to_same_line
  (h1 : is_perpendicular l1 α)
  (h2 : is_perpendicular l1 β) : is_parallel α β := 
sorry

end planes_parallel_if_perpendicular_to_same_line_l144_144594


namespace seq_a10_eq_90_l144_144828

noncomputable def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 0 ∧ ∀ n, a (n + 1) = a n + 2 * n

theorem seq_a10_eq_90 {a : ℕ → ℕ} (h : seq a) : a 10 = 90 :=
  sorry

end seq_a10_eq_90_l144_144828


namespace problem_l144_144752

-- Step 1: Define the transformation functions
def rotate_90_counterclockwise (h k x y : ℝ) : ℝ × ℝ :=
  (h - (y - k), k + (x - h))

def reflect_y_eq_x (x y : ℝ) : ℝ × ℝ :=
  (y, x)

-- Step 2: Define the given problem condition
theorem problem (a b : ℝ) :
  rotate_90_counterclockwise 2 3 (reflect_y_eq_x 5 1).fst (reflect_y_eq_x 5 1).snd = (a, b) →
  b - a = 0 :=
by
  intro h
  sorry

end problem_l144_144752


namespace solve_ineq_l144_144343

noncomputable def f (x : ℝ) : ℝ := (2 / (x + 2)) + (4 / (x + 8)) - (7 / 3)

theorem solve_ineq (x : ℝ) : 
  (f x ≤ 0) ↔ (x ∈ Set.Ioc (-8) 4) := 
sorry

end solve_ineq_l144_144343


namespace partial_fractions_sum_zero_l144_144430

noncomputable def sum_of_coefficients (A B C D E : ℝ) : Prop :=
  (A + B + C + D + E = 0)

theorem partial_fractions_sum_zero :
  ∀ (A B C D E : ℝ),
    (∀ x : ℝ, 1 = A*(x+1)*(x+2)*(x+3)*(x+5) + B*x*(x+2)*(x+3)*(x+5) + 
              C*x*(x+1)*(x+3)*(x+5) + D*x*(x+1)*(x+2)*(x+5) + 
              E*x*(x+1)*(x+2)*(x+3)) →
    sum_of_coefficients A B C D E :=
by sorry

end partial_fractions_sum_zero_l144_144430


namespace find_a8_in_arithmetic_sequence_l144_144236

variable {a : ℕ → ℕ} -- Define a as a function from natural numbers to natural numbers

-- Assume a is an arithmetic sequence
axiom arithmetic_sequence (a : ℕ → ℕ) : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_a8_in_arithmetic_sequence (h : a 4 + a 6 + a 8 + a 10 + a 12 = 120) : a 8 = 24 :=
by
  sorry  -- Proof to be filled in separately

end find_a8_in_arithmetic_sequence_l144_144236


namespace sum_of_three_pairwise_rel_prime_integers_l144_144282

theorem sum_of_three_pairwise_rel_prime_integers (a b c : ℕ)
  (h1: 1 < a) (h2: 1 < b) (h3: 1 < c)
  (prod: a * b * c = 216000)
  (rel_prime_ab : Nat.gcd a b = 1)
  (rel_prime_ac : Nat.gcd a c = 1)
  (rel_prime_bc : Nat.gcd b c = 1) : 
  a + b + c = 184 := 
sorry

end sum_of_three_pairwise_rel_prime_integers_l144_144282


namespace child_support_calculation_l144_144552

noncomputable def owed_child_support (yearly_salary : ℕ) (raise_pct: ℝ) 
(raise_years_additional_salary: ℕ) (payment_percentage: ℝ) 
(payment_years_salary_before_raise: ℕ) (already_paid : ℝ) : ℝ :=
  let initial_salary := yearly_salary * payment_years_salary_before_raise
  let increase_amount := yearly_salary * raise_pct
  let new_salary := yearly_salary + increase_amount
  let salary_after_raise := new_salary * raise_years_additional_salary
  let total_income := initial_salary + salary_after_raise
  let total_support_due := total_income * payment_percentage
  total_support_due - already_paid

theorem child_support_calculation:
  owed_child_support 30000 0.2 4 0.3 3 1200 = 69000 :=
by
  sorry

end child_support_calculation_l144_144552


namespace projection_onto_vector_l144_144183

noncomputable def projection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  ![\[9 / 25, 12 / 25\], \[12 / 25, 16 / 25\]]

theorem projection_onto_vector:
    ∀ (x y : ℚ), (Matrix.mul_vec projection_matrix ![\x, \y]) = ![(9 * x + 12 * y) / 25, (12 * x + 16 * y) / 25] := by
  sorry

end projection_onto_vector_l144_144183


namespace de_morgan_union_de_morgan_inter_l144_144262

open Set

variable {α : Type*} (A B : Set α)

theorem de_morgan_union : ∀ (A B : Set α), 
  compl (A ∪ B) = compl A ∩ compl B := 
by 
  intro A B
  sorry

theorem de_morgan_inter : ∀ (A B : Set α), 
  compl (A ∩ B) = compl A ∪ compl B := 
by 
  intro A B
  sorry

end de_morgan_union_de_morgan_inter_l144_144262


namespace range_of_f_l144_144500

open Real

noncomputable def f (x : ℝ) : ℝ :=
  arcsin x + arccos x + arcsec x

theorem range_of_f :
  Set.range f = {π / 2, 3 * π / 2} :=
by
  sorry

end range_of_f_l144_144500


namespace problem_statement_l144_144772

theorem problem_statement 
  (p q r x y z a b c : ℝ)
  (h1 : p / x = q / y ∧ q / y = r / z)
  (h2 : x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 1) :
  p^2 / a^2 + q^2 / b^2 + r^2 / c^2 = (p^2 + q^2 + r^2) / (x^2 + y^2 + z^2) :=
sorry  -- Proof omitted

end problem_statement_l144_144772


namespace find_discriminant_l144_144365

variables {a b c : ℝ}
variables (P : ℝ → ℝ)
def is_quadratic_polynomial (P : ℝ → ℝ) : Prop := ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, P x = a * x^2 + b * x + c)

theorem find_discriminant (h1 : is_quadratic_polynomial P)
  (h2 : ∃ x, P x = x - 2)
  (h3 : ∃ y, P y = 1 - y / 2)
  : ∃ D, D = -1/2 := 
sorry

end find_discriminant_l144_144365


namespace tangent_line_to_parabola_l144_144656

theorem tangent_line_to_parabola (k : ℝ) :
  (∃ (x y : ℝ), 4 * x + 7 * y + k = 0 ∧ y^2 = 16 * x) →
  (28 ^ 2 - 4 * 1 * (4 * k) = 0) → k = 49 :=
by
  intro h
  intro h_discriminant
  have discriminant_eq_zero : 28 ^ 2 - 4 * 1 * (4 * k) = 0 := h_discriminant
  sorry

end tangent_line_to_parabola_l144_144656


namespace evaluate_f_at_2_l144_144963

def f (x : ℝ) : ℝ := x^2 - 3 * x

theorem evaluate_f_at_2 :
  f 2 = -2 :=
by
  sorry

end evaluate_f_at_2_l144_144963


namespace age_ratio_l144_144348

-- Definitions as per the conditions
variable (j e x : ℕ)

-- Conditions from the problem
def condition1 : Prop := j - 4 = 2 * (e - 4)
def condition2 : Prop := j - 10 = 3 * (e - 10)

-- The statement we need to prove
theorem age_ratio (j e x : ℕ) (h1 : condition1 j e)
(h2 : condition2 j e) :
(j + x) * 2 = (e + x) * 3 ↔ x = 8 :=
sorry

end age_ratio_l144_144348


namespace phone_answered_within_two_rings_l144_144445

def probability_of_first_ring : ℝ := 0.5
def probability_of_second_ring : ℝ := 0.3
def probability_of_within_two_rings : ℝ := 0.8

theorem phone_answered_within_two_rings :
  probability_of_first_ring + probability_of_second_ring = probability_of_within_two_rings :=
by
  sorry

end phone_answered_within_two_rings_l144_144445


namespace value_of_a5_l144_144836

variable (a : ℕ → ℕ)

-- The initial condition
axiom initial_condition : a 1 = 2

-- The recurrence relation
axiom recurrence_relation : ∀ n : ℕ, n ≠ 0 → n * a (n+1) = 2 * (n + 1) * a n

theorem value_of_a5 : a 5 = 160 := 
sorry

end value_of_a5_l144_144836


namespace inclination_angle_of_line_l144_144286

theorem inclination_angle_of_line (m : ℝ) (b : ℝ) (h : b = -3) (h_line : ∀ x : ℝ, x - 3 = m * x + b) : 
  (Real.arctan m * 180 / Real.pi) = 45 := 
by sorry

end inclination_angle_of_line_l144_144286


namespace three_exp_product_sixth_power_l144_144672

theorem three_exp_product_sixth_power :
  ∃ n : ℤ, 3^12 * 3^18 = n^6 ∧ n = 243 :=
by
  existsi 243
  split
  · sorry
  · refl

end three_exp_product_sixth_power_l144_144672


namespace general_admission_tickets_l144_144117

variable (x y : ℕ)

theorem general_admission_tickets (h1 : x + y = 525) (h2 : 4 * x + 6 * y = 2876) : y = 388 := by
  sorry

end general_admission_tickets_l144_144117


namespace find_three_digit_integers_mod_l144_144670

theorem find_three_digit_integers_mod (n : ℕ) :
  (n % 7 = 3) ∧ (n % 8 = 6) ∧ (n % 5 = 2) ∧ (100 ≤ n) ∧ (n < 1000) :=
sorry

end find_three_digit_integers_mod_l144_144670


namespace theta_in_third_quadrant_l144_144059

theorem theta_in_third_quadrant (θ : ℝ) (h1 : Real.tan θ > 0) (h2 : Real.sin θ < 0) : 
  ∃ q : ℕ, q = 3 := 
sorry

end theta_in_third_quadrant_l144_144059


namespace david_marks_in_english_l144_144492

theorem david_marks_in_english
  (math phys chem bio : ℕ)
  (avg subs : ℕ) 
  (h_math : math = 95) 
  (h_phys : phys = 82) 
  (h_chem : chem = 97) 
  (h_bio : bio = 95) 
  (h_avg : avg = 93)
  (h_subs : subs = 5) :
  ∃ E : ℕ, (avg * subs = E + math + phys + chem + bio) ∧ E = 96 :=
by
  sorry

end david_marks_in_english_l144_144492


namespace minimum_expr_value_l144_144507

noncomputable def expr_min_value (a : ℝ) (h : a > 1) : ℝ :=
  a + 2 / (a - 1)

theorem minimum_expr_value (a : ℝ) (h : a > 1) :
  expr_min_value a h = 1 + 2 * Real.sqrt 2 :=
sorry

end minimum_expr_value_l144_144507


namespace arithmetic_sqrt_9_l144_144933

theorem arithmetic_sqrt_9 : real.sqrt 9 = 3 :=
by
  sorry

end arithmetic_sqrt_9_l144_144933


namespace discriminant_of_P_l144_144355

theorem discriminant_of_P (a b c : ℚ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1/2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1/2 := 
begin
  -- Proof omitted for brevity
  sorry
end

end discriminant_of_P_l144_144355


namespace arithmetic_sequence_general_term_l144_144063

theorem arithmetic_sequence_general_term (a : ℕ → ℤ) (d : ℤ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h_increasing : d > 0)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = a 2 ^ 2 - 4) :
  ∀ n, a n = 2 * n - 1 :=
by
  sorry

end arithmetic_sequence_general_term_l144_144063


namespace probability_at_least_one_boy_and_one_girl_l144_144625

theorem probability_at_least_one_boy_and_one_girl :
  (∀ (n : ℕ), (ℙ(birth_is_boy) = ℙ(birth_is_girl)) ∧ n = 4) →
  (∃ p : ℚ, p = 7 / 8 ∧
    p = 1 - (ℙ(all_boys) + ℙ(all_girls))) :=
by
  sorry

-- Definitions to be used
def birth_is_boy := sorry -- Placeholder for an event where a birth is a boy
def birth_is_girl := sorry -- Placeholder for an event where a birth is a girl
def all_boys := sorry -- Placeholder for an event where all four children are boys
def all_girls := sorry -- Placeholder for an event where all four children are girls

end probability_at_least_one_boy_and_one_girl_l144_144625


namespace travel_speed_l144_144015

theorem travel_speed (distance : ℕ) (time : ℕ) (h_distance : distance = 160) (h_time : time = 8) :
  ∃ speed : ℕ, speed = distance / time ∧ speed = 20 :=
by
  sorry

end travel_speed_l144_144015


namespace fraction_sum_of_roots_l144_144088

theorem fraction_sum_of_roots (x1 x2 : ℝ) (h1 : 5 * x1^2 - 3 * x1 - 2 = 0) (h2 : 5 * x2^2 - 3 * x2 - 2 = 0) (hx : x1 ≠ x2) :
  (1 / x1 + 1 / x2 = -3 / 2) :=
by
  sorry

end fraction_sum_of_roots_l144_144088


namespace no_such_quadratics_l144_144643

theorem no_such_quadratics :
  ¬ ∃ (a b c : ℤ), ∃ (x1 x2 x3 x4 : ℤ),
    (a * x1 * x2 = c ∧ a * (x1 + x2) = -b) ∧
    ((a + 1) * x3 * x4 = c + 1 ∧ (a + 1) * (x3 + x4) = -(b + 1)) :=
sorry

end no_such_quadratics_l144_144643


namespace quotient_of_integers_l144_144139

variable {x y : ℤ}

theorem quotient_of_integers (h : 1996 * x + y / 96 = x + y) : 
  (x / y = 1 / 2016) ∨ (y / x = 2016) := by
  sorry

end quotient_of_integers_l144_144139


namespace sin_neg_p_l144_144218

theorem sin_neg_p (a : ℝ) : (¬ ∃ x : ℝ, Real.sin x > a) → (a ≥ 1) := 
by
  sorry

end sin_neg_p_l144_144218


namespace school_sports_event_l144_144743

theorem school_sports_event (x y z : ℤ) (hx : x > y) (hy : y > z) (hz : z > 0)
  (points_A points_B points_E : ℤ) (ha : points_A = 22) (hb : points_B = 9) 
  (he : points_E = 9) (vault_winner_B : True) :
  ∃ n : ℕ, n = 5 ∧ second_place_grenade_throwing_team = 8^B :=
by
  sorry

end school_sports_event_l144_144743


namespace smallest_positive_integer_x_l144_144135

theorem smallest_positive_integer_x (x : ℕ) (h900 : ∃ a b c : ℕ, 900 = (2^a) * (3^b) * (5^c) ∧ a = 2 ∧ b = 2 ∧ c = 2) (h1152 : ∃ a b : ℕ, 1152 = (2^a) * (3^b) ∧ a = 7 ∧ b = 2) : x = 32 :=
by
  sorry

end smallest_positive_integer_x_l144_144135


namespace brigade_delegation_ways_l144_144854

theorem brigade_delegation_ways :
  let men := 10
  let women := 8
  let choose_men := Nat.choose men 3
  let choose_women := Nat.choose women 2
  choose_men * choose_women = 3360 :=
by
  let men := 10
  let women := 8
  let choose_men := Nat.choose men 3
  let choose_women := Nat.choose women 2
  calc
    choose_men * choose_women
    = Nat.choose men 3 * Nat.choose women 2 : by rfl
    = 120 * 28 : by sorry
    = 3360 : by sorry

end brigade_delegation_ways_l144_144854


namespace num_terminating_decimals_l144_144825

-- Define the problem conditions and statement
def is_terminating_decimal (n : ℕ) : Prop :=
  n % 3 = 0

theorem num_terminating_decimals : 
  ∃ (k : ℕ), k = 220 ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 660 → is_terminating_decimal n ↔ n % 3 = 0) := 
by
  sorry

end num_terminating_decimals_l144_144825


namespace num_monic_quadratic_trinomials_l144_144822

noncomputable def count_monic_quadratic_trinomials : ℕ :=
  4489

theorem num_monic_quadratic_trinomials :
  count_monic_quadratic_trinomials = 4489 :=
by
  sorry

end num_monic_quadratic_trinomials_l144_144822


namespace veranda_width_l144_144570

def area_of_veranda (w : ℝ) : ℝ :=
  let room_area := 19 * 12
  let total_area := room_area + 140
  let total_length := 19 + 2 * w
  let total_width := 12 + 2 * w
  total_length * total_width - room_area

theorem veranda_width:
  ∃ w : ℝ, area_of_veranda w = 140 := by
  sorry

end veranda_width_l144_144570


namespace origin_inside_ellipse_l144_144061

theorem origin_inside_ellipse (k : ℝ) (h : k^2 * 0^2 + 0^2 - 4*k*0 + 2*k*0 + k^2 - 1 < 0) : 0 < |k| ∧ |k| < 1 :=
by
  sorry

end origin_inside_ellipse_l144_144061


namespace pure_ghee_percentage_l144_144073

theorem pure_ghee_percentage (Q : ℝ) (P : ℝ) (H1 : Q = 10) (H2 : (P / 100) * Q + 10 = 0.80 * (Q + 10)) :
  P = 60 :=
sorry

end pure_ghee_percentage_l144_144073


namespace chocolates_initial_count_l144_144035

theorem chocolates_initial_count (remaining_chocolates: ℕ) 
    (daily_percentage: ℝ) (days: ℕ) 
    (final_chocolates: ℝ) 
    (remaining_fraction_proof: remaining_fraction = 0.7) 
    (days_proof: days = 3) 
    (final_chocolates_proof: final_chocolates = 28): 
    (remaining_fraction^days * (initial_chocolates:ℝ) = final_chocolates) → 
    (initial_chocolates = 82) := 
by 
  sorry

end chocolates_initial_count_l144_144035


namespace y_plus_inv_l144_144831

theorem y_plus_inv (y : ℝ) (h : y^3 + 1/y^3 = 110) : y + 1/y = 5 := 
by 
sorry

end y_plus_inv_l144_144831


namespace childSupportOwed_l144_144546

def annualIncomeBeforeRaise : ℕ := 30000
def yearsBeforeRaise : ℕ := 3
def raisePercentage : ℕ := 20
def annualIncomeAfterRaise (incomeBeforeRaise raisePercentage : ℕ) : ℕ :=
  incomeBeforeRaise + (incomeBeforeRaise * raisePercentage / 100)
def yearsAfterRaise : ℕ := 4
def childSupportPercentage : ℕ := 30
def amountPaid : ℕ := 1200

def calculateChildSupport (incomeYears : ℕ → ℕ → ℕ) (supportPercentage : ℕ) (years : ℕ) : ℕ :=
  (incomeYears years supportPercentage) * supportPercentage / 100 * years

def totalChildSupportOwed : ℕ :=
  (calculateChildSupport (λ _ _ => annualIncomeBeforeRaise) childSupportPercentage yearsBeforeRaise) +
  (calculateChildSupport (λ _ _ => annualIncomeAfterRaise annualIncomeBeforeRaise raisePercentage) childSupportPercentage yearsAfterRaise)

theorem childSupportOwed : totalChildSupportOwed - amountPaid = 69000 :=
by trivial

end childSupportOwed_l144_144546


namespace correct_algebraic_expression_l144_144455

theorem correct_algebraic_expression
  (A : String := "1 1/2 a")
  (B : String := "a × b")
  (C : String := "a ÷ b")
  (D : String := "2a") :
  D = "2a" :=
by {
  -- Explanation based on the conditions provided
  -- A: "1 1/2 a" is not properly formatted. Correct format involves improper fraction for multiplication.
  -- B: "a × b" should avoid using the multiplication sign explicitly.
  -- C: "a ÷ b" should be written as a fraction a/b.
  -- D: "2a" is correctly formatted.
  sorry
}

end correct_algebraic_expression_l144_144455


namespace parity_of_f_minimum_value_of_f_l144_144251

noncomputable def f (x a : ℝ) : ℝ := x^2 + |x - a| - 1

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f (x)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

theorem parity_of_f (a : ℝ) :
  (a = 0 → is_even_function (f a)) ∧
  (a ≠ 0 → ¬is_even_function (f a) ∧ ¬is_odd_function (f a)) := 
by sorry

theorem minimum_value_of_f (a : ℝ) :
  (a ≤ -1/2 → ∀ x : ℝ, f x a ≥ -a - 5 / 4) ∧
  (-1/2 < a ∧ a ≤ 1/2 → ∀ x : ℝ, f x a ≥ a^2 - 1) ∧
  (a > 1/2 → ∀ x : ℝ, f x a ≥ a - 5 / 4) :=
by sorry

end parity_of_f_minimum_value_of_f_l144_144251


namespace arable_land_decrease_max_l144_144303

theorem arable_land_decrease_max
  (A₀ : ℕ := 100000)
  (grain_yield_increase : ℝ := 1.22)
  (per_capita_increase : ℝ := 1.10)
  (pop_growth_rate : ℝ := 0.01)
  (years : ℕ := 10) :
  ∃ (max_decrease : ℕ), max_decrease = 4 := sorry

end arable_land_decrease_max_l144_144303


namespace arithmetic_sqrt_of_9_l144_144935

def arithmetic_sqrt (n : ℕ) : ℕ :=
  Nat.sqrt n

theorem arithmetic_sqrt_of_9 : arithmetic_sqrt 9 = 3 :=
by
  sorry

end arithmetic_sqrt_of_9_l144_144935


namespace cos_arith_prog_impossible_l144_144971

noncomputable def sin_arith_prog (x y z : ℝ) : Prop :=
  (2 * Real.sin y = Real.sin x + Real.sin z) ∧ (Real.sin x < Real.sin y) ∧ (Real.sin y < Real.sin z)

theorem cos_arith_prog_impossible (x y z : ℝ) (h : sin_arith_prog x y z) : 
  ¬(2 * Real.cos y = Real.cos x + Real.cos z) := 
by 
  sorry

end cos_arith_prog_impossible_l144_144971


namespace convex_quadrilateral_inequality_l144_144263

variable (a b c d : ℝ) -- lengths of sides of quadrilateral
variable (S : ℝ) -- Area of the quadrilateral

-- Given condition: a, b, c, d are lengths of the sides of a convex quadrilateral
def is_convex_quadrilateral (a b c d : ℝ) (S : ℝ) : Prop :=
  S ≤ (a^2 + b^2 + c^2 + d^2) / 4

theorem convex_quadrilateral_inequality (a b c d : ℝ) (S : ℝ) 
  (h : is_convex_quadrilateral a b c d S) : 
  S ≤ (a^2 + b^2 + c^2 + d^2) / 4 := 
by
  sorry

end convex_quadrilateral_inequality_l144_144263


namespace integer_to_sixth_power_l144_144686

theorem integer_to_sixth_power (a b : ℕ) (h : 3^a * 3^b = 3^(a + b)) (ha : a = 12) (hb : b = 18) : 
  ∃ x : ℕ, x = 243 ∧ x^6 = 3^(a + b) :=
by
  sorry

end integer_to_sixth_power_l144_144686


namespace sum_of_squares_of_consecutive_integers_divisible_by_5_l144_144033

theorem sum_of_squares_of_consecutive_integers_divisible_by_5 (n : ℤ) :
  (n^2 + (n+1)^2 + (n+2)^2 + (n+3)^2) % 5 = 0 :=
by
  sorry

end sum_of_squares_of_consecutive_integers_divisible_by_5_l144_144033


namespace probability_of_green_ball_l144_144707

def total_balls : ℕ := 3 + 3 + 6
def green_balls : ℕ := 3

theorem probability_of_green_ball : (green_balls : ℚ) / total_balls = 1 / 4 :=
by
  sorry

end probability_of_green_ball_l144_144707


namespace triangle_perimeter_l144_144945

theorem triangle_perimeter (x : ℕ) (h_odd : x % 2 = 1) (h_range : 1 < x ∧ x < 5) : 2 + 3 + x = 8 :=
by
  sorry

end triangle_perimeter_l144_144945


namespace bookstore_discount_l144_144843

noncomputable def discount_percentage (total_spent : ℝ) (over_22 : List ℝ) (under_20 : List ℝ) : ℝ :=
  let disc_over_22 := over_22.map (fun p => p * (1 - 0.30))
  let total_over_22 := disc_over_22.sum
  let total_with_under_20 := total_over_22 + 21
  let total_under_20 := under_20.sum
  let discount_received := total_spent - total_with_under_20
  let discount_percentage := (total_under_20 - discount_received) / total_under_20 * 100
  discount_percentage

theorem bookstore_discount :
  discount_percentage 95 [25.00, 35.00] [18.00, 12.00, 10.00] = 20 := by
  sorry

end bookstore_discount_l144_144843


namespace limit_tanxy_over_y_l144_144498

theorem limit_tanxy_over_y (f : ℝ×ℝ → ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x y, abs (x - 3) < δ ∧ abs y < δ → abs (f (x, y) - 3) < ε) :=
sorry

end limit_tanxy_over_y_l144_144498


namespace sum_divisible_by_seventeen_l144_144191

theorem sum_divisible_by_seventeen :
  (90 + 91 + 92 + 93 + 94 + 95 + 96 + 97) % 17 = 0 := 
by 
  sorry

end sum_divisible_by_seventeen_l144_144191


namespace find_f_at_8_l144_144599

theorem find_f_at_8 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (3 * x - 1) = x^2 + 2 * x + 4) :
  f 8 = 19 :=
sorry

end find_f_at_8_l144_144599


namespace cheese_fries_cost_l144_144407

def jim_money : ℝ := 20
def cousin_money : ℝ := 10
def combined_money : ℝ := jim_money + cousin_money
def expenditure : ℝ := 0.80 * combined_money
def cheeseburger_cost : ℝ := 3
def milkshake_cost : ℝ := 5
def cheeseburgers_cost : ℝ := 2 * cheeseburger_cost
def milkshakes_cost : ℝ := 2 * milkshake_cost
def meal_cost : ℝ := cheeseburgers_cost + milkshakes_cost

theorem cheese_fries_cost :
  let cheese_fries_cost := expenditure - meal_cost 
  cheese_fries_cost = 8 := 
by
  sorry

end cheese_fries_cost_l144_144407


namespace triangular_difference_l144_144801

/-- Definition of triangular numbers -/
def triangular (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Main theorem: the difference between the 30th and 29th triangular numbers is 30 -/
theorem triangular_difference : triangular 30 - triangular 29 = 30 :=
by
  sorry

end triangular_difference_l144_144801


namespace geometric_series_common_ratio_l144_144818

theorem geometric_series_common_ratio (r : ℚ) : 
  (∃ (a : ℚ), a = 4 / 7 ∧ a * r = 16 / 21) → r = 4 / 3 :=
by
  sorry

end geometric_series_common_ratio_l144_144818


namespace john_run_distance_l144_144078

theorem john_run_distance :
  ∀ (initial_hours : ℝ) (increase_time_percent : ℝ) (initial_speed : ℝ) (increase_speed : ℝ),
  initial_hours = 8 → increase_time_percent = 0.75 → initial_speed = 8 → increase_speed = 4 →
  let increased_hours := initial_hours * increase_time_percent,
      total_hours := initial_hours + increased_hours,
      new_speed := initial_speed + increase_speed,
      distance := total_hours * new_speed in
  distance = 168 := 
by
  intros initial_hours increase_time_percent initial_speed increase_speed h_hours h_time h_speed h_increase
  let increased_hours := initial_hours * increase_time_percent
  let total_hours := initial_hours + increased_hours
  let new_speed := initial_speed + increase_speed
  let distance := total_hours * new_speed
  sorry

end john_run_distance_l144_144078


namespace inspection_time_l144_144154

theorem inspection_time 
  (num_digits : ℕ) (num_letters : ℕ) 
  (letter_opts : ℕ) (start_digits : ℕ) 
  (inspection_time_three_hours : ℕ) 
  (probability : ℝ) 
  (num_vehicles : ℕ) 
  (vehicles_inspected : ℕ)
  (cond1 : num_digits = 4)
  (cond2 : num_letters = 2)
  (cond3 : letter_opts = 3)
  (cond4 : start_digits = 2)
  (cond5 : inspection_time_three_hours = 180) 
  (cond6 : probability = 0.02)
  (cond7 : num_vehicles = 900)
  (cond8 : vehicles_inspected = num_vehicles * probability) :
  vehicles_inspected = (inspection_time_three_hours / 10) :=
  sorry

end inspection_time_l144_144154


namespace cost_price_of_toy_l144_144784

-- Define the conditions
def sold_toys := 18
def selling_price := 23100
def gain_toys := 3

-- Define the cost price of one toy 
noncomputable def C := 1100

-- Lean 4 statement to prove the cost price
theorem cost_price_of_toy (C : ℝ) (sold_toys selling_price gain_toys : ℕ) (h1 : selling_price = (sold_toys + gain_toys) * C) : 
  C = 1100 := 
by
  sorry


end cost_price_of_toy_l144_144784


namespace number_of_ways_to_choose_lineup_l144_144554

/--
   We have a basketball team of 16 players with a set of 3 twins and a set of 4 quadruplets.
   We want to find the number of ways to choose 6 starters such that exactly 2 of the quadruplets are included.
--/

theorem number_of_ways_to_choose_lineup : 
  ∑ (n : ℕ) in (∅ : Finset ℕ), (real.to_nat (nat.choose 4 2) * real.to_nat (nat.choose 14 4)) = 6006 :=
by
  sorry

end number_of_ways_to_choose_lineup_l144_144554


namespace arithmetic_square_root_of_nine_l144_144905

theorem arithmetic_square_root_of_nine : ∃ (x : ℝ), (x * x = 9) ∧ (x ≥ 0) ∧ (x = 3) :=
by
  sorry

end arithmetic_square_root_of_nine_l144_144905


namespace depak_bank_account_l144_144730

theorem depak_bank_account :
  ∃ (n : ℕ), (x + 1 = 6 * n) ∧ n = 1 → x = 5 := 
sorry

end depak_bank_account_l144_144730


namespace geometric_sequence_sum_of_first_four_terms_l144_144704

theorem geometric_sequence_sum_of_first_four_terms 
  (a q : ℝ)
  (h1 : a * (1 + q) = 7)
  (h2 : a * (q^6 - 1) / (q - 1) = 91) :
  a * (1 + q + q^2 + q^3) = 28 := by
  sorry

end geometric_sequence_sum_of_first_four_terms_l144_144704


namespace cows_C_grazed_l144_144775

/-- Define the conditions for each milkman’s cow-months. -/
def A_cow_months := 24 * 3
def B_cow_months := 10 * 5
def D_cow_months := 21 * 3
def C_cow_months (x : ℕ) := x * 4

/-- Define the cost per cow-month based on A's share. -/
def cost_per_cow_month := 720 / A_cow_months

/-- Define the total rent. -/
def total_rent := 3250

/-- Define the total cow-months including C's cow-months as a variable. -/
def total_cow_months (x : ℕ) := A_cow_months + B_cow_months + C_cow_months x + D_cow_months

/-- Lean 4 statement to prove the number of cows C grazed. -/
theorem cows_C_grazed (x : ℕ) :
  total_rent = total_cow_months x * cost_per_cow_month → x = 35 := by {
  sorry
}

end cows_C_grazed_l144_144775


namespace bushes_needed_for_octagon_perimeter_l144_144022

theorem bushes_needed_for_octagon_perimeter
  (side_length : ℝ) (spacing : ℝ)
  (octagonal : ∀ (s : ℝ), s = 8 → 8 * s = 64)
  (spacing_condition : ∀ (p : ℝ), p = 64 → p / spacing = 32) :
  spacing = 2 → side_length = 8 → (64 / 2 = 32) := 
by
  sorry

end bushes_needed_for_octagon_perimeter_l144_144022


namespace interval_intersection_l144_144345

theorem interval_intersection (x : ℝ) :
  (4 * x > 2 ∧ 4 * x < 3) ∧ (5 * x > 2 ∧ 5 * x < 3) ↔ (x > 1/2 ∧ x < 3/5) :=
by
  sorry

end interval_intersection_l144_144345


namespace jerry_task_duration_l144_144533

def earnings_per_task : ℕ := 40
def hours_per_day : ℕ := 10
def days_per_week : ℕ := 7
def total_earnings : ℕ := 1400

theorem jerry_task_duration :
  (10 * 7 = 70) →
  (1400 / 40 = 35) →
  (70 / 35 = 2) →
  (total_earnings / earnings_per_task = (hours_per_day * days_per_week) / h) →
  h = 2 :=
by
  intros h1 h2 h3 h4
  -- proof steps (omitted)
  sorry

end jerry_task_duration_l144_144533


namespace phone_price_increase_is_40_percent_l144_144329

-- Definitions based on the conditions
def initial_price_tv := 500
def increased_fraction_tv := 2 / 5
def initial_price_phone := 400
def total_amount_received := 1260

-- The price increase of the TV
def final_price_tv := initial_price_tv * (1 + increased_fraction_tv)

-- The final price of the phone
def final_price_phone := total_amount_received - final_price_tv

-- The percentage increase in the phone's price
def percentage_increase_phone := ((final_price_phone - initial_price_phone) / initial_price_phone) * 100

-- The theorem to prove
theorem phone_price_increase_is_40_percent :
  percentage_increase_phone = 40 := by
  sorry

end phone_price_increase_is_40_percent_l144_144329


namespace discriminant_of_P_l144_144353

theorem discriminant_of_P (a b c : ℚ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1/2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1/2 := 
begin
  -- Proof omitted for brevity
  sorry
end

end discriminant_of_P_l144_144353


namespace Jason_has_22_5_toys_l144_144716

noncomputable def RachelToys : ℝ := 1
noncomputable def JohnToys : ℝ := RachelToys + 6.5
noncomputable def JasonToys : ℝ := 3 * JohnToys

theorem Jason_has_22_5_toys : JasonToys = 22.5 := sorry

end Jason_has_22_5_toys_l144_144716


namespace part1_part2_l144_144837

open Set

def A : Set ℤ := { x | ∃ (m n : ℤ), x = m^2 - n^2 }

theorem part1 : 3 ∈ A := 
by sorry

theorem part2 (k : ℤ) : 4 * k - 2 ∉ A := 
by sorry

end part1_part2_l144_144837


namespace total_pennies_donated_l144_144487

def cassandra_pennies : ℕ := 5000
def james_pennies : ℕ := cassandra_pennies - 276
def total_pennies : ℕ := cassandra_pennies + james_pennies

theorem total_pennies_donated : total_pennies = 9724 := by
  sorry

end total_pennies_donated_l144_144487


namespace quadratic_polynomial_discriminant_l144_144377

def P (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₀ : a ≠ 0)
  (h₁ : ∃ x : ℝ, P a b c x = x - 2 ∧ (discriminant a (b - 1) (c + 2) = 0))
  (h₂ : ∃ x : ℝ, P a b c x = 1 - x / 2 ∧ (discriminant a (b + 1 / 2) (c - 1) = 0)) :
  discriminant a b c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l144_144377


namespace max_positive_integers_l144_144849

theorem max_positive_integers (a b c d e f : ℤ) (h : (a * b + c * d * e * f) < 0) :
  ∃ n, n ≤ 5 ∧ (∀x ∈ [a, b, c, d, e, f], 0 < x → x ≤ 5) :=
by
  sorry

end max_positive_integers_l144_144849


namespace sum_of_fourth_powers_eq_82_l144_144392

theorem sum_of_fourth_powers_eq_82 (x y : ℝ) (hx : x + y = -2) (hy : x * y = -3) :
  x^4 + y^4 = 82 :=
by
  sorry

end sum_of_fourth_powers_eq_82_l144_144392


namespace attendees_proportion_l144_144705

def attendees (t k : ℕ) := k / t

theorem attendees_proportion (n t new_t : ℕ) (h1 : n * t = 15000) (h2 : t = 50) (h3 : new_t = 75) : attendees new_t 15000 = 200 :=
by
  -- Proof omitted, main goal is to assert equivalency
  sorry

end attendees_proportion_l144_144705


namespace combined_meows_l144_144126

theorem combined_meows (first_cat_freq second_cat_freq third_cat_freq : ℕ) 
  (time : ℕ) 
  (h1 : first_cat_freq = 3)
  (h2 : second_cat_freq = 2 * first_cat_freq)
  (h3 : third_cat_freq = second_cat_freq / 3)
  (h4 : time = 5) : 
  first_cat_freq * time + second_cat_freq * time + third_cat_freq * time = 55 := 
by
  sorry

end combined_meows_l144_144126


namespace heartsuit_xx_false_l144_144034

def heartsuit (x y : ℝ) : ℝ := |x - y|

theorem heartsuit_xx_false (x : ℝ) : heartsuit x x ≠ x :=
by sorry

end heartsuit_xx_false_l144_144034


namespace jeff_bought_6_pairs_l144_144405

theorem jeff_bought_6_pairs (price_of_shoes : ℝ) (num_of_shoes : ℕ) (price_of_jersey : ℝ)
  (h1 : price_of_jersey = (1 / 4) * price_of_shoes)
  (h2 : num_of_shoes * price_of_shoes = 480)
  (h3 : num_of_shoes * price_of_shoes + 4 * price_of_jersey = 560) :
  num_of_shoes = 6 :=
sorry

end jeff_bought_6_pairs_l144_144405


namespace div_power_sub_one_l144_144261

theorem div_power_sub_one : 11 * 31 * 61 ∣ 20^15 - 1 := 
by
  sorry

end div_power_sub_one_l144_144261


namespace sniper_B_has_greater_chance_of_winning_l144_144072

def pA (n : ℕ) : ℝ :=
  if n = 1 then 0.4 else if n = 2 then 0.1 else if n = 3 then 0.5 else 0

def pB (n : ℕ) : ℝ :=
  if n = 1 then 0.1 else if n = 2 then 0.6 else if n = 3 then 0.3 else 0

noncomputable def expected_score (p : ℕ → ℝ) : ℝ :=
  (1 * p 1) + (2 * p 2) + (3 * p 3)

theorem sniper_B_has_greater_chance_of_winning :
  expected_score pB > expected_score pA :=
by
  sorry

end sniper_B_has_greater_chance_of_winning_l144_144072


namespace sum_of_squares_l144_144228

theorem sum_of_squares :
  ∃ p q r s t u : ℤ, (∀ x : ℤ, 729 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) ∧ 
    (p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 8210) :=
sorry

end sum_of_squares_l144_144228


namespace arithmetic_square_root_of_nine_l144_144897

theorem arithmetic_square_root_of_nine : Real.sqrt 9 = 3 :=
sorry

end arithmetic_square_root_of_nine_l144_144897


namespace cone_shape_in_spherical_coordinates_l144_144347

-- Define the conditions as given in the problem
def spherical_coordinates (rho theta phi c : ℝ) : Prop := 
  rho = c * Real.sin phi

-- Define the main statement to prove
theorem cone_shape_in_spherical_coordinates (rho theta phi c : ℝ) (hpos : 0 < c) :
  spherical_coordinates rho theta phi c → 
  ∃ cone : Prop, cone :=
sorry

end cone_shape_in_spherical_coordinates_l144_144347


namespace part1_complement_intersection_part2_range_m_l144_144413

open Set

-- Define set A
def A : Set ℝ := { x | -1 ≤ x ∧ x < 4 }

-- Define set B parameterized by m
def B (m : ℝ) : Set ℝ := { x | m ≤ x ∧ x ≤ m + 2 }

-- Part (1): Prove the complement of the intersection for m = 3
theorem part1_complement_intersection :
  ∀ x : ℝ, x ∉ (A ∩ B 3) ↔ x < 3 ∨ x ≥ 4 :=
by
  sorry

-- Part (2): Prove the range of m for A ∩ B = ∅
theorem part2_range_m (m : ℝ) :
  (A ∩ B m = ∅) ↔ m < -3 ∨ m ≥ 4 :=
by
  sorry

end part1_complement_intersection_part2_range_m_l144_144413


namespace librarian_donated_200_books_this_year_l144_144346

noncomputable def total_books_five_years_ago : ℕ := 500
noncomputable def books_bought_two_years_ago : ℕ := 300
noncomputable def books_bought_last_year : ℕ := books_bought_two_years_ago + 100
noncomputable def total_books_current : ℕ := 1000

-- The Lean statement to prove the librarian donated 200 old books this year
theorem librarian_donated_200_books_this_year :
  total_books_five_years_ago + books_bought_two_years_ago + books_bought_last_year - total_books_current = 200 :=
by sorry

end librarian_donated_200_books_this_year_l144_144346


namespace log_mul_l144_144211

theorem log_mul (a M N : ℝ) (ha_pos : 0 < a) (hM_pos : 0 < M) (hN_pos : 0 < N) (ha_ne_one : a ≠ 1) :
    Real.log (M * N) / Real.log a = Real.log M / Real.log a + Real.log N / Real.log a := by
  sorry

end log_mul_l144_144211


namespace flyers_left_l144_144243

theorem flyers_left (total_flyers : ℕ) (jack_flyers : ℕ) (rose_flyers : ℕ) :
  total_flyers = 1236 → jack_flyers = 120 → rose_flyers = 320 → total_flyers - (jack_flyers + rose_flyers) = 796 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact eq.refl _

end flyers_left_l144_144243


namespace joan_dozen_of_eggs_l144_144864

def number_of_eggs : ℕ := 72
def dozen : ℕ := 12

theorem joan_dozen_of_eggs : (number_of_eggs / dozen) = 6 := by
  sorry

end joan_dozen_of_eggs_l144_144864


namespace company_total_payment_correct_l144_144231

def totalEmployees : Nat := 450
def firstGroup : Nat := 150
def secondGroup : Nat := 200
def thirdGroup : Nat := 100

def firstBaseSalary : Nat := 2000
def secondBaseSalary : Nat := 2500
def thirdBaseSalary : Nat := 3000

def firstInitialBonus : Nat := 500
def secondInitialBenefit : Nat := 400
def thirdInitialBenefit : Nat := 600

def firstLayoffRound1 : Nat := (20 * firstGroup) / 100
def secondLayoffRound1 : Nat := (25 * secondGroup) / 100
def thirdLayoffRound1 : Nat := (15 * thirdGroup) / 100

def remainingFirstGroupRound1 : Nat := firstGroup - firstLayoffRound1
def remainingSecondGroupRound1 : Nat := secondGroup - secondLayoffRound1
def remainingThirdGroupRound1 : Nat := thirdGroup - thirdLayoffRound1

def firstAdjustedBonusRound1 : Nat := 400
def secondAdjustedBenefitRound1 : Nat := 300

def firstLayoffRound2 : Nat := (10 * remainingFirstGroupRound1) / 100
def secondLayoffRound2 : Nat := (15 * remainingSecondGroupRound1) / 100
def thirdLayoffRound2 : Nat := (5 * remainingThirdGroupRound1) / 100

def remainingFirstGroupRound2 : Nat := remainingFirstGroupRound1 - firstLayoffRound2
def remainingSecondGroupRound2 : Nat := remainingSecondGroupRound1 - secondLayoffRound2
def remainingThirdGroupRound2 : Nat := remainingThirdGroupRound1 - thirdLayoffRound2

def thirdAdjustedBenefitRound2 : Nat := (80 * thirdInitialBenefit) / 100

def totalBaseSalary : Nat :=
  (remainingFirstGroupRound2 * firstBaseSalary)
  + (remainingSecondGroupRound2 * secondBaseSalary)
  + (remainingThirdGroupRound2 * thirdBaseSalary)

def totalBonusesAndBenefits : Nat :=
  (remainingFirstGroupRound2 * firstAdjustedBonusRound1)
  + (remainingSecondGroupRound2 * secondAdjustedBenefitRound1)
  + (remainingThirdGroupRound2 * thirdAdjustedBenefitRound2)

def totalPayment : Nat :=
  totalBaseSalary + totalBonusesAndBenefits

theorem company_total_payment_correct :
  totalPayment = 893200 :=
by
  -- proof steps
  sorry

end company_total_payment_correct_l144_144231


namespace find_c_of_parabola_l144_144435

theorem find_c_of_parabola (a b c : ℚ) (h_vertex : (5 : ℚ) = a * (3 : ℚ)^2 + b * (3 : ℚ) + c)
    (h_point : (7 : ℚ) = a * (1 : ℚ)^2 + b * (1 : ℚ) + c) :
  c = 19 / 2 :=
by
  sorry

end find_c_of_parabola_l144_144435


namespace transformed_data_properties_l144_144321

-- Definitions of the initial mean and variance
def initial_mean : ℝ := 2.8
def initial_variance : ℝ := 3.6

-- Definitions of transformation constants
def multiplier : ℝ := 2
def increment : ℝ := 60

-- New mean after transformation
def new_mean : ℝ := multiplier * initial_mean + increment

-- New variance after transformation
def new_variance : ℝ := (multiplier ^ 2) * initial_variance

-- Theorem statement
theorem transformed_data_properties :
  new_mean = 65.6 ∧ new_variance = 14.4 :=
by
  sorry

end transformed_data_properties_l144_144321


namespace collinear_points_solves_a_l144_144847

theorem collinear_points_solves_a : 
  ∀ (a : ℝ),
  let A := (1, 3)
  let B := (5, 8)
  let C := (29, a)
  (8 - 3) / (5 - 1) = (a - 8) / (29 - 5) → a = 38 :=
by 
  intro a
  let A := (1, 3)
  let B := (5, 8)
  let C := (29, a)
  intro h
  sorry

end collinear_points_solves_a_l144_144847


namespace evaluate_expression_l144_144723

variable (x : ℝ)
variable (hx : x^3 - 3 * x = 6)

theorem evaluate_expression : x^7 - 27 * x^2 = 9 * (x + 1) * (x + 6) :=
by
  sorry

end evaluate_expression_l144_144723


namespace arithmetic_square_root_of_nine_l144_144924

theorem arithmetic_square_root_of_nine :
  real.sqrt 9 = 3 :=
by
  sorry

end arithmetic_square_root_of_nine_l144_144924


namespace distance_train_A_when_meeting_l144_144763

noncomputable def distance_traveled_by_train_A : ℝ :=
  let distance := 375
  let time_A := 36
  let time_B := 24
  let speed_A := distance / time_A
  let speed_B := distance / time_B
  let relative_speed := speed_A + speed_B
  let time_meeting := distance / relative_speed
  speed_A * time_meeting

theorem distance_train_A_when_meeting :
  distance_traveled_by_train_A = 150 := by
  sorry

end distance_train_A_when_meeting_l144_144763


namespace tank_cost_minimization_l144_144311

def volume := 4800
def depth := 3
def cost_per_sqm_bottom := 150
def cost_per_sqm_walls := 120

theorem tank_cost_minimization (x : ℝ) 
  (S1 : ℝ := volume / depth)
  (S2 : ℝ := 6 * (x + (S1 / x)))
  (cost := cost_per_sqm_bottom * S1 + cost_per_sqm_walls * S2) :
  (x = 40) → cost = 297600 :=
sorry

end tank_cost_minimization_l144_144311


namespace total_distance_walked_l144_144878

-- Condition 1: Distance in feet
def distance_feet : ℝ := 30

-- Condition 2: Conversion factor from feet to meters
def feet_to_meters : ℝ := 0.3048

-- Condition 3: Number of trips
def trips : ℝ := 4

-- Question: Total distance walked in meters
theorem total_distance_walked :
  distance_feet * feet_to_meters * trips = 36.576 :=
sorry

end total_distance_walked_l144_144878


namespace sum_of_plane_angles_l144_144884

theorem sum_of_plane_angles (v f p : ℕ) (h : v = p) :
    (2 * π * (v - f) = 2 * π * (p - 2)) :=
by sorry

end sum_of_plane_angles_l144_144884


namespace cost_percentage_l144_144973

variable (t b : ℝ)

def C := t * b ^ 4
def R := t * (2 * b) ^ 4

theorem cost_percentage : R = 16 * C := by
  sorry

end cost_percentage_l144_144973


namespace sugar_ratio_l144_144177

theorem sugar_ratio (r : ℝ) (H1 : 24 * r^3 = 3) : (24 * r / 24 = 1 / 2) :=
by
  sorry

end sugar_ratio_l144_144177


namespace johnny_marbles_l144_144534

noncomputable def choose_at_least_one_red : ℕ :=
  let total_marbles := 8
  let red_marbles := 1
  let other_marbles := 7
  let choose_4_out_of_8 := Nat.choose total_marbles 4
  let choose_3_out_of_7 := Nat.choose other_marbles 3
  let choose_4_with_at_least_1_red := choose_3_out_of_7
  choose_4_with_at_least_1_red

theorem johnny_marbles : choose_at_least_one_red = 35 :=
by
  -- Sorry, proof is omitted
  sorry

end johnny_marbles_l144_144534


namespace sum_an_eq_543_l144_144824

def a_n (n : ℕ) : ℕ :=
  if n % 15 = 0 ∧ n % 16 = 0 then 15
  else if n % 16 = 0 ∧ n % 17 = 0 then 16
  else if n % 15 = 0 ∧ n % 17 = 0 then 17
  else 0

def sum_an : ℕ :=
  (Finset.range 3000).sum a_n

theorem sum_an_eq_543 : sum_an = 543 := by
  sorry

end sum_an_eq_543_l144_144824


namespace find_k_l144_144271

theorem find_k 
  (A B X Y : ℝ × ℝ)
  (hA : A = (-3, 0))
  (hB : B = (0, -3))
  (hX : X = (0, 9))
  (Yx : Y.1 = 15)
  (hXY_parallel : (Y.2 - X.2) / (Y.1 - X.1) = (B.2 - A.2) / (B.1 - A.1)) :
  Y.2 = -6 := by
  -- proofs are omitted as per the requirements
  sorry

end find_k_l144_144271


namespace olympiad_problem_l144_144019

variable (a b c d : ℕ)
variable (N : ℕ := a + b + c + d)

theorem olympiad_problem
  (h1 : (a + d) / (N:ℚ) = 0.5)
  (h2 : (b + d) / (N:ℚ) = 0.6)
  (h3 : (c + d) / (N:ℚ) = 0.7)
  : (d : ℚ) / N * 100 = 40 := by
  sorry

end olympiad_problem_l144_144019


namespace three_pow_mul_l144_144693

theorem three_pow_mul (a b : ℕ) (h_a : a = 12) (h_b : b = 18) :
  3^a * 3^b = 243^6 := by
  rw [h_a, h_b]
  calc
    3^12 * 3^18
      = 3^(12 + 18) : by rw [pow_add]
  ... = 3^30 : by norm_num
  ... = (3^5)^6 : by rw [pow_mul, ← mul_comm]
  ... = 243^6 : by norm_num

end three_pow_mul_l144_144693


namespace order_of_magnitude_l144_144066

theorem order_of_magnitude (a b : ℝ) (h1 : a > 0) (h2 : b < 0) (h3 : |a| < |b|) :
  -b > a ∧ a > -a ∧ -a > b := by
  sorry

end order_of_magnitude_l144_144066


namespace conference_session_time_l144_144309

def conference_duration_hours : ℕ := 8
def conference_duration_minutes : ℕ := 45
def break_time : ℕ := 30

theorem conference_session_time :
  (conference_duration_hours * 60 + conference_duration_minutes) - break_time = 495 :=
by sorry

end conference_session_time_l144_144309


namespace club_members_l144_144781

theorem club_members (M W : ℕ) (h1 : M + W = 30) (h2 : M + 1/3 * (W : ℝ) = 18) : M = 12 :=
by
  -- proof step
  sorry

end club_members_l144_144781


namespace fred_blue_marbles_l144_144658

theorem fred_blue_marbles (tim_marbles : ℕ) (fred_marbles : ℕ) (h1 : tim_marbles = 5) (h2 : fred_marbles = 22 * tim_marbles) : fred_marbles = 110 :=
by
  sorry

end fred_blue_marbles_l144_144658


namespace problem_1_problem_2_l144_144660

noncomputable def a : ℝ := sorry
def m : ℝ := sorry
def n : ℝ := sorry
def k : ℝ := sorry

theorem problem_1 (h1 : a^m = 2) (h2 : a^n = 4) (h3 : a^k = 32) (h4 : a ≠ 0) : 
  a^(3*m + 2*n - k) = 4 := 
sorry

theorem problem_2 (h1 : a^m = 2) (h2 : a^n = 4) (h3 : a^k = 32) (h4 : a ≠ 0) : 
  k - 3*m - n = 0 := 
sorry

end problem_1_problem_2_l144_144660


namespace fraction_ratio_l144_144181

theorem fraction_ratio (x : ℚ) : 
  (x : ℚ) / (2/6) = (3/4) / (1/2) -> (x = 1/2) :=
by {
  sorry
}

end fraction_ratio_l144_144181


namespace jelly_sold_l144_144008

theorem jelly_sold (G S R P : ℕ) (h1 : G = 2 * S) (h2 : R = 2 * P) (h3 : R = G / 3) (h4 : P = 6) : S = 18 := by
  sorry

end jelly_sold_l144_144008


namespace square_no_remainder_5_mod_9_l144_144951

theorem square_no_remainder_5_mod_9 (n : ℤ) : (n^2 % 9 ≠ 5) :=
by sorry

end square_no_remainder_5_mod_9_l144_144951


namespace contrapositive_example_l144_144747

theorem contrapositive_example (α : ℝ) : (α = Real.pi / 3 → Real.cos α = 1 / 2) → (Real.cos α ≠ 1 / 2 → α ≠ Real.pi / 3) :=
by
  sorry

end contrapositive_example_l144_144747


namespace average_score_is_7_stddev_is_2_l144_144158

-- Define the scores list
def scores : List ℝ := [7, 8, 7, 9, 5, 4, 9, 10, 7, 4]

-- Proof statement for average score
theorem average_score_is_7 : (scores.sum / scores.length) = 7 :=
by
  simp [scores]
  sorry

-- Proof statement for standard deviation
theorem stddev_is_2 : Real.sqrt ((scores.map (λ x => (x - (scores.sum / scores.length))^2)).sum / scores.length) = 2 :=
by
  simp [scores]
  sorry

end average_score_is_7_stddev_is_2_l144_144158


namespace expand_product_l144_144179

theorem expand_product (x : ℤ) : 
  (3 * x + 4) * (2 * x - 6) = 6 * x^2 - 10 * x - 24 :=
by
  sorry

end expand_product_l144_144179


namespace distribution_X_Y_l144_144762

noncomputable theory

variables {Ω : Type*} [MeasurableSpace Ω]
variables (P : MeasureTheory.ProbabilityMeasure Ω)
variables (hit_first: MeasureTheory.MeasurableSet (set.univ : set Ω))
variables (hit_second: MeasureTheory.MeasurableSet (set.univ : set Ω))
variables (miss_first: MeasureTheory.MeasurableSet (set.univ : set Ω))
variables (miss_second: MeasureTheory.MeasurableSet (set.univ : set Ω))

-- Given conditions
def prob_hit_first : ℝ := 0.3
def prob_hit_second : ℝ := 0.7

-- Definitions of the distribution laws for X and Y
def dist_X (k : ℕ) : ℝ := if k = 0 then 0 else 0.79 * 0.21^(k-1)
def dist_Y (k : ℕ) : ℝ := 0.3 * if k = 0 then 1 else 0.553 * 0.21^(k-1)

-- Theorem stating the required distributions
theorem distribution_X_Y :
  (∀ k : ℕ, MeasureTheory.ProbabilityMassFunction.probability P (dist_X k)) ∧ 
  (∀ k : ℕ, MeasureTheory.ProbabilityMassFunction.probability P (dist_Y k)) :=
by { sorry }

end distribution_X_Y_l144_144762


namespace total_profit_is_50_l144_144468

-- Define the initial conditions
def initial_milk : ℕ := 80
def initial_water : ℕ := 20
def milk_cost_per_liter : ℕ := 22
def first_mixture_milk : ℕ := 40
def first_mixture_water : ℕ := 5
def first_mixture_price : ℕ := 19
def second_mixture_milk : ℕ := 25
def second_mixture_water : ℕ := 10
def second_mixture_price : ℕ := 18
def third_mixture_milk : ℕ := initial_milk - (first_mixture_milk + second_mixture_milk)
def third_mixture_water : ℕ := 5
def third_mixture_price : ℕ := 21

-- Define variables for revenue calculations
def first_mixture_revenue : ℕ := (first_mixture_milk + first_mixture_water) * first_mixture_price
def second_mixture_revenue : ℕ := (second_mixture_milk + second_mixture_water) * second_mixture_price
def third_mixture_revenue : ℕ := (third_mixture_milk + third_mixture_water) * third_mixture_price
def total_revenue : ℕ := first_mixture_revenue + second_mixture_revenue + third_mixture_revenue

-- Define the total milk cost
def total_milk_used : ℕ := first_mixture_milk + second_mixture_milk + third_mixture_milk
def total_cost : ℕ := total_milk_used * milk_cost_per_liter

-- Define the profit as the difference between total revenue and total cost
def profit : ℕ := total_revenue - total_cost

-- Prove that the total profit is Rs. 50
theorem total_profit_is_50 : profit = 50 := by
  sorry

end total_profit_is_50_l144_144468


namespace fraction_inhabitable_l144_144127

-- Define the constants based on the given conditions
def fraction_water : ℚ := 3 / 5
def fraction_inhabitable_land : ℚ := 3 / 4

-- Define the theorem to prove that the fraction of Earth's surface that is inhabitable is 3/10
theorem fraction_inhabitable (w h : ℚ) (hw : w = fraction_water) (hh : h = fraction_inhabitable_land) : 
  (1 - w) * h = 3 / 10 :=
by
  sorry

end fraction_inhabitable_l144_144127


namespace percentage_salt_solution_l144_144669

-- Definitions
def P : ℝ := 60
def ounces_added := 40
def initial_solution_ounces := 40
def initial_solution_percentage := 0.20
def final_solution_percentage := 0.40
def final_solution_ounces := 80

-- Lean Statement
theorem percentage_salt_solution (P : ℝ) :
  (8 + 0.01 * P * ounces_added) = 0.40 * final_solution_ounces → P = 60 := 
by
  sorry

end percentage_salt_solution_l144_144669


namespace range_of_m_l144_144833

theorem range_of_m (m : ℝ) : (∃ (x y : ℝ), x^2 + y^2 - 2*x - 4*y + m = 0) → m < 5 :=
by
  sorry

end range_of_m_l144_144833


namespace evaluation_expression_l144_144867

theorem evaluation_expression (a b c d : ℝ) 
  (h1 : a = Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6)
  (h2 : b = -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6)
  (h3 : c = Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6)
  (h4 : d = -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6) :
  (1/a + 1/b + 1/c + 1/d)^2 = (16 * (11 + 2 * Real.sqrt 30)) / ((11 + 2 * Real.sqrt 30 - 3 * Real.sqrt 6)^2) :=
sorry

end evaluation_expression_l144_144867


namespace find_discriminant_l144_144362

variables {a b c : ℝ}
variables (P : ℝ → ℝ)
def is_quadratic_polynomial (P : ℝ → ℝ) : Prop := ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, P x = a * x^2 + b * x + c)

theorem find_discriminant (h1 : is_quadratic_polynomial P)
  (h2 : ∃ x, P x = x - 2)
  (h3 : ∃ y, P y = 1 - y / 2)
  : ∃ D, D = -1/2 := 
sorry

end find_discriminant_l144_144362


namespace value_of_f_at_2_l144_144965

def f (x : ℝ) : ℝ := x^2 - 3 * x

theorem value_of_f_at_2 : f 2 = -2 := 
by 
  sorry

end value_of_f_at_2_l144_144965


namespace probability_a_2b_3c_gt_5_l144_144448

def isInUnitCube (a b c : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1

theorem probability_a_2b_3c_gt_5 (a b c : ℝ) :
  isInUnitCube a b c → ¬(a + 2 * b + 3 * c > 5) :=
by
  intro h
  -- The proof goes here, currently using sorry as placeholder
  sorry

end probability_a_2b_3c_gt_5_l144_144448


namespace arithmetic_square_root_of_9_l144_144909

theorem arithmetic_square_root_of_9 : sqrt 9 = 3 := by
  sorry

end arithmetic_square_root_of_9_l144_144909


namespace parallelogram_side_length_sum_l144_144574

theorem parallelogram_side_length_sum (x y z : ℚ) 
  (h1 : 3 * x - 1 = 12)
  (h2 : 4 * z + 2 = 7 * y + 3) :
  x + y + z = 121 / 21 :=
by
  sorry

end parallelogram_side_length_sum_l144_144574


namespace length_of_other_side_l144_144406

-- Defining the conditions
def roofs := 3
def sides_per_roof := 2
def length_of_one_side := 40 -- measured in feet
def shingles_per_square_foot := 8
def total_shingles := 38400

-- The proof statement
theorem length_of_other_side : 
    ∃ (L : ℕ), (total_shingles / shingles_per_square_foot / roofs / sides_per_roof = 40 * L) ∧ L = 20 :=
by
  sorry

end length_of_other_side_l144_144406


namespace three_exp_product_sixth_power_l144_144676

theorem three_exp_product_sixth_power :
  ∃ n : ℤ, 3^12 * 3^18 = n^6 ∧ n = 243 :=
by
  existsi 243
  split
  · sorry
  · refl

end three_exp_product_sixth_power_l144_144676


namespace arithmetic_sqrt_of_9_l144_144900

theorem arithmetic_sqrt_of_9 : (∃ (sqrt : ℝ), sqrt = 3 ∧ ∀ x, x*x = 9 → x = sqrt) :=
by
  existsi (3 : ℝ)
  split
  exact rfl
  intros x hx
  exact sqrt_unique hx 3

end arithmetic_sqrt_of_9_l144_144900


namespace amelia_money_left_l144_144161

theorem amelia_money_left :
  let first_course := 15
  let second_course := first_course + 5
  let dessert := 0.25 * second_course
  let total_first_three_courses := first_course + second_course + dessert
  let drink := 0.20 * total_first_three_courses
  let pre_tip_total := total_first_three_courses + drink
  let tip := 0.15 * pre_tip_total
  let total_bill := pre_tip_total + tip
  let initial_money := 60
  let money_left := initial_money - total_bill
  money_left = 4.8 :=
by
  sorry

end amelia_money_left_l144_144161


namespace sequence_a_100_l144_144277

theorem sequence_a_100 (a : ℕ → ℤ) (h₁ : a 1 = 3) (h₂ : ∀ n : ℕ, a (n + 1) = a n - 2) : a 100 = -195 :=
by
  sorry

end sequence_a_100_l144_144277


namespace find_h_l144_144273

theorem find_h (h j k : ℤ) (y_intercept1 : 3 * h ^ 2 + j = 2013) 
  (y_intercept2 : 2 * h ^ 2 + k = 2014)
  (x_intercepts1 : ∃ (y : ℤ), j = -3 * y ^ 2)
  (x_intercepts2 : ∃ (x : ℤ), k = -2 * x ^ 2) :
  h = 36 :=
by sorry

end find_h_l144_144273


namespace find_custom_operation_value_l144_144415

noncomputable def custom_operation (a b : ℤ) : ℚ := (1 : ℚ)/a + (1 : ℚ)/b

theorem find_custom_operation_value (a b : ℤ) (h1 : a + b = 12) (h2 : a * b = 32) :
  custom_operation a b = 3 / 8 := by
  sorry

end find_custom_operation_value_l144_144415


namespace ben_chairs_in_10_days_l144_144798

noncomputable def chairs_built_per_day (hours_per_shift : ℕ) (hours_per_chair : ℕ) : ℕ :=
  hours_per_shift / hours_per_chair

theorem ben_chairs_in_10_days 
  (hours_per_shift : ℕ)
  (hours_per_chair : ℕ)
  (days: ℕ)
  (h_shift: hours_per_shift = 8)
  (h_chair: hours_per_chair = 5)
  (h_days: days = 10) : 
  chairs_built_per_day hours_per_shift hours_per_chair * days = 10 :=
by 
  -- We insert a placeholder 'sorry' to be replaced by an actual proof.
  sorry

end ben_chairs_in_10_days_l144_144798


namespace distance_between_A_and_B_l144_144883

-- Definitions and conditions
variables {A B C : Type}    -- Locations
variables {v1 v2 : ℕ}       -- Speeds of person A and person B
variables {distanceAB : ℕ}  -- Distance we want to find

noncomputable def first_meet_condition (v1 v2 : ℕ) : Prop :=
  ∃ t : ℕ, (v1 * t - 108 = v2 * t - 100)

noncomputable def second_meet_condition (v1 v2 distanceAB : ℕ) : Prop :=
  distanceAB = 3750

-- Theorem statement
theorem distance_between_A_and_B (v1 v2 distanceAB : ℕ) :
  first_meet_condition v1 v2 → second_meet_condition v1 v2 distanceAB →
  distanceAB = 3750 :=
by
  intros _ _ 
  sorry

end distance_between_A_and_B_l144_144883


namespace sum_of_three_integers_l144_144281

theorem sum_of_three_integers (a b c : ℕ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c)
  (h4 : a * b * c = 216000) (h5 : Nat.coprime a b) (h6 : Nat.coprime a c) (h7 : Nat.coprime b c) :
  a + b + c = 184 :=
sorry

end sum_of_three_integers_l144_144281


namespace find_unknown_rate_of_blankets_l144_144467

theorem find_unknown_rate_of_blankets (x : ℕ) 
  (h1 : 3 * 100 = 300) 
  (h2 : 5 * 150 = 750)
  (h3 : 3 + 5 + 2 = 10) 
  (h4 : 10 * 160 = 1600) 
  (h5 : 300 + 750 + 2 * x = 1600) : 
  x = 275 := 
sorry

end find_unknown_rate_of_blankets_l144_144467


namespace THIS_code_is_2345_l144_144840

def letterToDigit (c : Char) : Option Nat :=
  match c with
  | 'M' => some 0
  | 'A' => some 1
  | 'T' => some 2
  | 'H' => some 3
  | 'I' => some 4
  | 'S' => some 5
  | 'F' => some 6
  | 'U' => some 7
  | 'N' => some 8
  | _   => none

def codeToNumber (code : String) : Option String :=
  code.toList.mapM letterToDigit >>= fun digits => some (digits.foldl (fun acc d => acc ++ toString d) "")

theorem THIS_code_is_2345 :
  codeToNumber "THIS" = some "2345" :=
by
  sorry

end THIS_code_is_2345_l144_144840


namespace arithmetic_sequence_general_term_l144_144053

theorem arithmetic_sequence_general_term
  (d : ℕ) (a : ℕ → ℕ)
  (ha4 : a 4 = 14)
  (hd : d = 3) :
  ∃ a₁, ∀ n, a n = a₁ + (n - 1) * d := by
  sorry

end arithmetic_sequence_general_term_l144_144053


namespace tennis_preference_combined_percentage_l144_144947

theorem tennis_preference_combined_percentage :
  let total_north_students := 1500
  let total_south_students := 1800
  let north_tennis_percentage := 0.30
  let south_tennis_percentage := 0.35
  let north_tennis_students := total_north_students * north_tennis_percentage
  let south_tennis_students := total_south_students * south_tennis_percentage
  let total_tennis_students := north_tennis_students + south_tennis_students
  let total_students := total_north_students + total_south_students
  let combined_percentage := (total_tennis_students / total_students) * 100
  combined_percentage = 33 := 
by
  sorry

end tennis_preference_combined_percentage_l144_144947


namespace arithmetic_square_root_of_9_l144_144916

theorem arithmetic_square_root_of_9 : Real.sqrt 9 = 3 := by
  sorry

end arithmetic_square_root_of_9_l144_144916


namespace total_pamphlets_correct_l144_144873

def mike_initial_speed := 600
def mike_initial_hours := 9
def mike_break_hours := 2
def leo_relative_hours := 1 / 3
def leo_relative_speed := 2

def total_pamphlets (mike_initial_speed mike_initial_hours mike_break_hours leo_relative_hours leo_relative_speed : ℕ) : ℕ :=
  let mike_pamphlets_before_break := mike_initial_speed * mike_initial_hours
  let mike_speed_after_break := mike_initial_speed / 3
  let mike_pamphlets_after_break := mike_speed_after_break * mike_break_hours
  let total_mike_pamphlets := mike_pamphlets_before_break + mike_pamphlets_after_break

  let leo_hours := mike_initial_hours * leo_relative_hours
  let leo_speed := mike_initial_speed * leo_relative_speed
  let leo_pamphlets := leo_hours * leo_speed

  total_mike_pamphlets + leo_pamphlets

theorem total_pamphlets_correct : total_pamphlets 600 9 2 (1 / 3 : ℕ) 2 = 9400 := 
by 
  sorry

end total_pamphlets_correct_l144_144873


namespace boxes_division_l144_144463

theorem boxes_division (total_eggs : ℚ) (eggs_per_box : ℚ) (number_of_boxes : ℚ) :
  total_eggs = 3 ∧ eggs_per_box = 1.5 -> number_of_boxes = 2 :=
begin
  intro h,
  cases h with ht hp,
  rw [ht, hp],
  norm_num,
end

end boxes_division_l144_144463


namespace roots_cubic_l144_144846

theorem roots_cubic (a b c d r s t : ℂ) 
    (h1 : a ≠ 0)
    (h2 : r + s + t = -b / a)
    (h3 : r * s + r * t + s * t = c / a)
    (h4 : r * s * t = -d / a) :
    (1 / r^2) + (1 / s^2) + (1 / t^2) = (b^2 - 2 * a * c) / (d^2) :=
by
    sorry

end roots_cubic_l144_144846


namespace question1_question2_l144_144417

namespace MathProofs

variable (U : Set ℝ) (A : Set ℝ) (B : Set ℝ)

-- Definitions based on conditions
def isA := ∀ x, A x ↔ (-3 < x ∧ x < 2)
def isB := ∀ x, B x ↔ (Real.exp (x - 1) ≥ 1)
def isCuA := ∀ x, (U \ A) x ↔ (x ≤ -3 ∨ x ≥ 2)

-- Proof of Question 1
theorem question1 : (∀ x, (A ∪ B) x ↔ (x > -3)) := by
  sorry

-- Proof of Question 2
theorem question2 : (∀ x, ((U \ A) ∩ B) x ↔ (x ≥ 2)) := by
  sorry

end MathProofs

end question1_question2_l144_144417


namespace scooter_gain_percent_l144_144558

theorem scooter_gain_percent 
  (purchase_price : ℝ) (repair_costs : ℝ) (selling_price : ℝ) 
  (h1 : purchase_price = 800) (h2 : repair_costs = 200) (h3 : selling_price = 1200) : 
  ((selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs)) * 100 = 20 :=
by
  sorry

end scooter_gain_percent_l144_144558


namespace vertex_angle_of_obtuse_isosceles_triangle_l144_144753

theorem vertex_angle_of_obtuse_isosceles_triangle 
  (a b h : ℝ)
  (θ φ : ℝ)
  (a_nonzero : a ≠ 0)
  (isosceles_triangle : a^2 = 3 * b * h)
  (b_def: b = 2 * a * Real.cos θ)
  (h_def : h = a * Real.sin θ)
  (φ_def : φ = 180 - 2 * θ)
  (obtuse : φ > 90) :
  φ = 160.53 :=
by
  sorry

end vertex_angle_of_obtuse_isosceles_triangle_l144_144753


namespace prove_ln10_order_l144_144511

def ln10_order_proof : Prop :=
  let a := Real.log 10
  let b := Real.log 100
  let c := (Real.log 10) ^ 2
  c > b ∧ b > a

theorem prove_ln10_order : ln10_order_proof := 
sorry

end prove_ln10_order_l144_144511


namespace smallest_n_for_y_n_integer_l144_144333

noncomputable def y (n : ℕ) : ℝ :=
  if n = 0 then 0 else
  if n = 1 then (5 : ℝ)^(1/3) else
  if n = 2 then ((5 : ℝ)^(1/3))^((5 : ℝ)^(1/3)) else
  y (n-1)^((5 : ℝ)^(1/3))

theorem smallest_n_for_y_n_integer : ∃ n : ℕ, y n = 5 ∧ ∀ m < n, y m ≠ ((⌊y m⌋:ℝ)) :=
by
  sorry

end smallest_n_for_y_n_integer_l144_144333


namespace square_inscribed_in_hexagon_has_side_length_l144_144564

-- Definitions for the conditions given
noncomputable def side_length_square (AB EF : ℝ) : ℝ :=
  if AB = 30 ∧ EF = 19 * (Real.sqrt 3 - 1) then 10 * Real.sqrt 3 else 0

-- The theorem stating the specified equality
theorem square_inscribed_in_hexagon_has_side_length (AB EF : ℝ)
  (hAB : AB = 30) (hEF : EF = 19 * (Real.sqrt 3 - 1)) :
  side_length_square AB EF = 10 * Real.sqrt 3 := 
by 
  -- This is the proof placeholder
  sorry

end square_inscribed_in_hexagon_has_side_length_l144_144564


namespace distribute_candies_l144_144646

-- Definitions based on conditions
def num_ways_distribute_candies : ℕ :=
  ∑ r in finset.range(8), ∑ b in finset.range(8 - r), if r >= 2 ∧ b >= 2 ∧ r + b <= 8 then
    (Nat.choose 8 r) * (Nat.choose (8 - r) b) * 2 ^ (8 - r - b)
  else
    0

-- The proof statement, no proof body required
theorem distribute_candies : num_ways_distribute_candies = 2048 := 
by sorry

end distribute_candies_l144_144646


namespace determine_f_1789_l144_144768

theorem determine_f_1789
  (f : ℕ → ℕ)
  (h1 : ∀ n : ℕ, 0 < n → f (f n) = 4 * n + 9)
  (h2 : ∀ k : ℕ, f (2^k) = 2^(k+1) + 3) :
  f 1789 = 3581 :=
sorry

end determine_f_1789_l144_144768


namespace elizabeth_revenue_per_investment_l144_144259

theorem elizabeth_revenue_per_investment :
  ∀ (revenue_per_investment_banks revenue_difference total_investments_banks total_investments_elizabeth : ℕ),
    revenue_per_investment_banks = 500 →
    total_investments_banks = 8 →
    total_investments_elizabeth = 5 →
    revenue_difference = 500 →
    ((revenue_per_investment_banks * total_investments_banks) + revenue_difference) / total_investments_elizabeth = 900 :=
by
  intros revenue_per_investment_banks revenue_difference total_investments_banks total_investments_elizabeth
  intros h_banks_revenue h_banks_investments h_elizabeth_investments h_revenue_difference
  sorry

end elizabeth_revenue_per_investment_l144_144259


namespace quadratic_polynomial_discriminant_l144_144378

def P (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₀ : a ≠ 0)
  (h₁ : ∃ x : ℝ, P a b c x = x - 2 ∧ (discriminant a (b - 1) (c + 2) = 0))
  (h₂ : ∃ x : ℝ, P a b c x = 1 - x / 2 ∧ (discriminant a (b + 1 / 2) (c - 1) = 0)) :
  discriminant a b c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l144_144378


namespace intersection_A_B_l144_144334
-- Lean 4 code statement

def set_A : Set ℝ := {x | |x - 1| > 2}
def set_B : Set ℝ := {x | x * (x - 5) < 0}
def set_intersection : Set ℝ := {x | 3 < x ∧ x < 5}

theorem intersection_A_B :
  (set_A ∩ set_B) = set_intersection := by
  sorry

end intersection_A_B_l144_144334


namespace find_k_l144_144397

theorem find_k (k : ℤ) :
  (∃ a b c : ℤ, a = 49 + k ∧ b = 441 + k ∧ c = 961 + k ∧
  (∃ r : ℚ, b = r * a ∧ c = r * r * a)) ↔ k = 1152 := by
  sorry

end find_k_l144_144397


namespace daily_expenditure_l144_144627

theorem daily_expenditure (total_spent : ℕ) (days_in_june : ℕ) (equal_consumption : Prop) :
  total_spent = 372 ∧ days_in_june = 30 ∧ equal_consumption → (372 / 30) = 12.40 := by
  sorry

end daily_expenditure_l144_144627


namespace geometric_sequence_problem_l144_144710

-- Definitions
def is_geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop := ∀ n, a (n + 1) = q * a n

-- Problem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ)
    (h_geom : is_geom_seq a q)
    (h1 : a 3 * a 7 = 8)
    (h2 : a 4 + a 6 = 6) :
    a 2 + a 8 = 9 :=
sorry

end geometric_sequence_problem_l144_144710


namespace jane_albert_same_committee_l144_144957

def probability_same_committee (total_MBAs : ℕ) (committee_size : ℕ) (num_committees : ℕ) (favorable_cases : ℕ) (total_cases : ℕ) : ℚ :=
  favorable_cases / total_cases

theorem jane_albert_same_committee :
  probability_same_committee 9 4 3 105 630 = 1 / 6 :=
by
  sorry

end jane_albert_same_committee_l144_144957


namespace drive_time_is_eleven_hours_l144_144164

-- Define the distances and speed as constants
def distance_salt_lake_to_vegas : ℕ := 420
def distance_vegas_to_los_angeles : ℕ := 273
def average_speed : ℕ := 63

-- Calculate the total distance
def total_distance : ℕ := distance_salt_lake_to_vegas + distance_vegas_to_los_angeles

-- Calculate the total time required
def total_time : ℕ := total_distance / average_speed

-- Theorem stating Andy wants to complete the drive in 11 hours
theorem drive_time_is_eleven_hours : total_time = 11 := sorry

end drive_time_is_eleven_hours_l144_144164


namespace inequality_holds_l144_144719

theorem inequality_holds (a b : ℝ) (h1 : a > 1) (h2 : 1 > b) (h3 : b > -1) : a > b^2 := 
sorry

end inequality_holds_l144_144719


namespace simple_interest_correct_l144_144287

def principal : ℝ := 400
def rate : ℝ := 0.20
def time : ℝ := 2

def simple_interest (P R T : ℝ) : ℝ := P * R * T

theorem simple_interest_correct :
  simple_interest principal rate time = 160 :=
by
  sorry

end simple_interest_correct_l144_144287


namespace projection_onto_vector_is_expected_l144_144186

def projection_matrix (u: ℝ × ℝ) : matrix (fin 2) (fin 2) ℝ :=
  let ⟨x, y⟩ := u in 
  (1 / (x^2 + y^2)) • (matrix.col_vec u ⬝ (matrix.transpose (matrix.row_vec u)))

def expected_matrix : matrix (fin 2) (fin 2) ℝ :=
  ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]]

theorem projection_onto_vector_is_expected :
  projection_matrix (3, 4) = expected_matrix := by
  sorry

end projection_onto_vector_is_expected_l144_144186


namespace range_of_a_l144_144386

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, |x - a| + |x + 1| ≤ 2) ↔ a ∈ (Set.Iio (-3) ∪ Set.Ioi 1) :=
by sorry

end range_of_a_l144_144386


namespace complex_sum_l144_144538

noncomputable def omega : ℂ := sorry
axiom h1 : omega^11 = 1
axiom h2 : omega ≠ 1

theorem complex_sum 
: omega^10 + omega^14 + omega^18 + omega^22 + omega^26 + omega^30 + omega^34 + omega^38 + omega^42 + omega^46 + omega^50 + omega^54 + omega^58 
= -omega^10 :=
sorry

end complex_sum_l144_144538


namespace probability_at_least_one_boy_one_girl_l144_144622

noncomputable def probability_one_boy_one_girl : ℚ :=
  1 - (1 / 16) - (1 / 16)

theorem probability_at_least_one_boy_one_girl :
  probability_one_boy_one_girl = 7 / 8 := by
  sorry

end probability_at_least_one_boy_one_girl_l144_144622


namespace marissa_tied_boxes_l144_144097

theorem marissa_tied_boxes 
  (r_total : ℝ) (r_per_box : ℝ) (r_left : ℝ) (h_total : r_total = 4.5)
  (h_per_box : r_per_box = 0.7) (h_left : r_left = 1) :
  (r_total - r_left) / r_per_box = 5 :=
by
  sorry

end marissa_tied_boxes_l144_144097


namespace original_square_perimeter_l144_144606

-- Define the problem statement
theorem original_square_perimeter (P_perimeter : ℕ) (hP : P_perimeter = 56) : 
  ∃ sq_perimeter : ℕ, sq_perimeter = 32 := 
by 
  sorry

end original_square_perimeter_l144_144606


namespace silver_coin_value_l144_144561

--- Definitions from the conditions
def total_value_hoard (value_silver : ℕ) := 100 * 3 * value_silver + 60 * value_silver + 33

--- Statement of the theorem to prove
theorem silver_coin_value (x : ℕ) (h : total_value_hoard x = 2913) : x = 8 :=
by {
  sorry
}

end silver_coin_value_l144_144561


namespace factor_correct_l144_144650

def factor_expression (x : ℝ) : Prop :=
  x * (x - 3) - 5 * (x - 3) = (x - 5) * (x - 3)

theorem factor_correct (x : ℝ) : factor_expression x :=
  by sorry

end factor_correct_l144_144650


namespace geometric_sequence_sum_is_9_l144_144712

theorem geometric_sequence_sum_is_9 {a : ℕ → ℝ} (q : ℝ) 
  (h3a7 : a 3 * a 7 = 8) 
  (h4a6 : a 4 + a 6 = 6)
  (h_geom : ∀ n, a (n + 1) = a n * q) : a 2 + a 8 = 9 :=
sorry

end geometric_sequence_sum_is_9_l144_144712


namespace fraction_saved_l144_144972

-- Definitions and given conditions
variables {P : ℝ} {f : ℝ}

-- Worker saves the same fraction each month, the same take-home pay each month
-- Total annual savings = 12fP and total annual savings = 2 * (amount not saved monthly)
theorem fraction_saved (h : 12 * f * P = 2 * (1 - f) * P) (P_ne_zero : P ≠ 0) : f = 1 / 7 :=
by
  -- The proof of the theorem goes here
  sorry

end fraction_saved_l144_144972


namespace man_age_twice_son_age_in_2_years_l144_144155

variable (currentAgeSon : ℕ)
variable (currentAgeMan : ℕ)
variable (Y : ℕ)

-- Given conditions
def sonCurrentAge : Prop := currentAgeSon = 23
def manCurrentAge : Prop := currentAgeMan = currentAgeSon + 25
def manAgeTwiceSonAgeInYYears : Prop := currentAgeMan + Y = 2 * (currentAgeSon + Y)

-- Theorem to prove
theorem man_age_twice_son_age_in_2_years :
  sonCurrentAge currentAgeSon →
  manCurrentAge currentAgeSon currentAgeMan →
  manAgeTwiceSonAgeInYYears currentAgeSon currentAgeMan Y →
  Y = 2 :=
by
  intros h_son_age h_man_age h_age_relation
  sorry

end man_age_twice_son_age_in_2_years_l144_144155


namespace arithmetic_sequence_problem_l144_144829

noncomputable def a_n (n : ℕ) : ℝ := sorry  -- Define the arithmetic sequence

theorem arithmetic_sequence_problem
  (a_4 : ℝ) (a_9 : ℝ)
  (h_a4 : a_4 = 5)
  (h_a9 : a_9 = 17)
  (h_arithmetic : ∀ n : ℕ, a_n (n + 1) = a_n n + (a_n 2 - a_n 1)) :
  a_n 14 = 29 :=
by
  -- the proof will utilize the property of arithmetic sequence and substitutions
  sorry

end arithmetic_sequence_problem_l144_144829


namespace arman_is_6_times_older_than_sister_l144_144794

def sisterWasTwoYearsOldFourYearsAgo := 2
def yearsAgo := 4
def armansAgeInFourYears := 40

def currentAgeOfSister := sisterWasTwoYearsOldFourYearsAgo + yearsAgo
def currentAgeOfArman := armansAgeInFourYears - yearsAgo

theorem arman_is_6_times_older_than_sister :
  currentAgeOfArman = 6 * currentAgeOfSister :=
by
  sorry

end arman_is_6_times_older_than_sister_l144_144794


namespace statement1_statement2_statement3_statement4_statement5_statement6_l144_144844

/-
Correct syntax statements in pseudo code
-/

def correct_assignment1 (A B : ℤ) : Prop :=
  B = A ∧ A = 50

def correct_assignment2 (x y z : ℕ) : Prop :=
  x = 1 ∧ y = 2 ∧ z = 3

def correct_input1 (s : String) (x : ℕ) : Prop :=
  s = "How old are you?" ∧ x ≥ 0

def correct_input2 (x : ℕ) : Prop :=
  x ≥ 0

def correct_print1 (s1 : String) (C : ℤ) : Prop :=
  s1 = "A+B=" ∧ C < 100  -- additional arbitrary condition for C

def correct_print2 (s2 : String) : Prop :=
  s2 = "Good-bye!"

theorem statement1 (A : ℤ) : ∃ B, correct_assignment1 A B :=
sorry

theorem statement2 : ∃ (x y z : ℕ), correct_assignment2 x y z :=
sorry

theorem statement3 (x : ℕ) : ∃ s, correct_input1 s x :=
sorry

theorem statement4 (x : ℕ) : correct_input2 x :=
sorry

theorem statement5 (C : ℤ) : ∃ s1, correct_print1 s1 C :=
sorry

theorem statement6 : ∃ s2, correct_print2 s2 :=
sorry

end statement1_statement2_statement3_statement4_statement5_statement6_l144_144844


namespace problem_statement_l144_144721

theorem problem_statement 
  (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) : 
  (p - r) * (q - s) / ((p - q) * (r - s)) = -3 / 2 := 
    sorry

end problem_statement_l144_144721


namespace nathan_tokens_l144_144421

theorem nathan_tokens
  (hockey_games : Nat := 5)
  (hockey_cost : Nat := 4)
  (basketball_games : Nat := 7)
  (basketball_cost : Nat := 5)
  (skee_ball_games : Nat := 3)
  (skee_ball_cost : Nat := 3)
  : hockey_games * hockey_cost + basketball_games * basketball_cost + skee_ball_games * skee_ball_cost = 64 := 
by
  sorry

end nathan_tokens_l144_144421
