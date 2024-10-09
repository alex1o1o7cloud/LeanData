import Mathlib

namespace initial_beavers_l1422_142219

theorem initial_beavers (B C : ℕ) (h1 : C = 40) (h2 : B + C + 2 * B + (C - 10) = 130) : B = 20 :=
by
  sorry

end initial_beavers_l1422_142219


namespace worker_y_defective_rate_l1422_142282

noncomputable def y_f : ℚ := 0.1666666666666668
noncomputable def d_x : ℚ := 0.005 -- converting percentage to decimal
noncomputable def d_total : ℚ := 0.0055 -- converting percentage to decimal

theorem worker_y_defective_rate :
  ∃ d_y : ℚ, d_y = 0.008 ∧ d_total = ((1 - y_f) * d_x + y_f * d_y) :=
by
  sorry

end worker_y_defective_rate_l1422_142282


namespace muffin_expense_l1422_142222

theorem muffin_expense (B D : ℝ) 
    (h1 : D = 0.90 * B) 
    (h2 : B = D + 15) : 
    B + D = 285 := 
    sorry

end muffin_expense_l1422_142222


namespace youseff_blocks_l1422_142218

-- Definition of the conditions
def time_to_walk (x : ℕ) : ℕ := x
def time_to_ride (x : ℕ) : ℕ := (20 * x) / 60
def extra_time (x : ℕ) : ℕ := time_to_walk x - time_to_ride x

-- Statement of the problem in Lean
theorem youseff_blocks : ∃ x : ℕ, extra_time x = 6 ∧ x = 9 :=
by {
  sorry
}

end youseff_blocks_l1422_142218


namespace number_of_boys_in_second_grade_l1422_142238

-- conditions definition
variables (B : ℕ) (G2 : ℕ := 11) (G3 : ℕ := 2 * (B + G2)) (total : ℕ := B + G2 + G3)

-- mathematical statement to be proved
theorem number_of_boys_in_second_grade : total = 93 → B = 20 :=
by
  -- omitting the proof
  intro h_total
  sorry

end number_of_boys_in_second_grade_l1422_142238


namespace tan_sum_l1422_142233

theorem tan_sum (α : ℝ) (h : Real.cos (π / 2 + α) = 2 * Real.cos α) : 
  Real.tan α + Real.tan (2 * α) = -2 / 3 :=
by
  sorry

end tan_sum_l1422_142233


namespace nth_equation_l1422_142209

theorem nth_equation (n : ℕ) : (2 * n + 2) ^ 2 - (2 * n) ^ 2 = 4 * (2 * n + 1) :=
by
  sorry

end nth_equation_l1422_142209


namespace doubling_profit_condition_l1422_142216

-- Definitions
def purchase_price : ℝ := 210
def initial_selling_price : ℝ := 270
def initial_items_sold : ℝ := 30
def profit_per_item (selling_price : ℝ) : ℝ := selling_price - purchase_price
def daily_profit (selling_price : ℝ) (items_sold : ℝ) : ℝ := profit_per_item selling_price * items_sold
def increase_in_items_sold_per_yuan (reduction : ℝ) : ℝ := 3 * reduction

-- Condition: Initial daily profit
def initial_daily_profit : ℝ := daily_profit initial_selling_price initial_items_sold

-- Proof problem
theorem doubling_profit_condition (reduction : ℝ) :
  daily_profit (initial_selling_price - reduction) (initial_items_sold + increase_in_items_sold_per_yuan reduction) = 2 * initial_daily_profit :=
sorry

end doubling_profit_condition_l1422_142216


namespace trapezoid_PQRS_perimeter_l1422_142207

noncomputable def trapezoid_perimeter (PQ RS : ℝ) (height : ℝ) (PS QR : ℝ) : ℝ :=
  PQ + RS + PS + QR

theorem trapezoid_PQRS_perimeter :
  ∀ (PQ RS : ℝ) (height : ℝ)
  (PS QR : ℝ),
  PQ = 6 →
  RS = 10 →
  height = 5 →
  PS = Real.sqrt (5^2 + 4^2) →
  QR = Real.sqrt (5^2 + 4^2) →
  trapezoid_perimeter PQ RS height PS QR = 16 + 2 * Real.sqrt 41 :=
by
  intros
  sorry

end trapezoid_PQRS_perimeter_l1422_142207


namespace mouse_cannot_eat_entire_cheese_l1422_142278

-- Defining the conditions of the problem
structure Cheese :=
  (size : ℕ := 3)  -- The cube size is 3x3x3
  (central_cube_removed : Bool := true)  -- The central cube is removed

inductive CubeColor
| black
| white

structure Mouse :=
  (can_eat : CubeColor -> CubeColor -> Bool)
  (adjacency : Nat -> Nat -> Bool)

def cheese_problem (c : Cheese) (m : Mouse) : Bool := sorry

-- The main theorem: It is impossible for the mouse to eat the entire piece of cheese.
theorem mouse_cannot_eat_entire_cheese : ∀ (c : Cheese) (m : Mouse),
  cheese_problem c m = false := sorry

end mouse_cannot_eat_entire_cheese_l1422_142278


namespace original_selling_price_l1422_142224

theorem original_selling_price (P SP1 SP2 : ℝ) (h1 : SP1 = 1.10 * P)
    (h2 : SP2 = 1.17 * P) (h3 : SP2 - SP1 = 35) : SP1 = 550 :=
by
  sorry

end original_selling_price_l1422_142224


namespace number_of_boys_in_school_l1422_142235

theorem number_of_boys_in_school (x g : ℕ) (h1 : x + g = 400) (h2 : g = (x * 400) / 100) : x = 80 :=
by
  sorry

end number_of_boys_in_school_l1422_142235


namespace find_g_of_2_l1422_142241

theorem find_g_of_2 {g : ℝ → ℝ} (h : ∀ x : ℝ, g (3 * x - 4) = 4 * x + 6) : g 2 = 14 :=
sorry

end find_g_of_2_l1422_142241


namespace quadratic_ineq_solution_range_of_b_for_any_a_l1422_142290

variable {α : Type*} [LinearOrderedField α]

noncomputable def f (a b x : α) : α := -3 * x^2 + a * (5 - a) * x + b

theorem quadratic_ineq_solution (a b : α) : 
  (∀ x ∈ Set.Ioo (-1 : α) 3, f a b x > 0) →
  ((a = 2 ∧ b = 9) ∨ (a = 3 ∧ b = 9)) := 
  sorry

theorem range_of_b_for_any_a (a b : α) :
  (∀ a : α, f a b 2 < 0) → 
  b < -1 / 2 := 
  sorry

end quadratic_ineq_solution_range_of_b_for_any_a_l1422_142290


namespace average_temperature_week_l1422_142276

theorem average_temperature_week :
  let sunday := 99.1
  let monday := 98.2
  let tuesday := 98.7
  let wednesday := 99.3
  let thursday := 99.8
  let friday := 99.0
  let saturday := 98.9
  (sunday + monday + tuesday + wednesday + thursday + friday + saturday) / 7 = 99.0 :=
by
  sorry

end average_temperature_week_l1422_142276


namespace cone_tangent_min_lateral_area_l1422_142268

/-- 
Given a cone with volume π / 6, prove that when the lateral area of the cone is minimized,
the tangent of the angle between the slant height and the base is sqrt(2).
-/
theorem cone_tangent_min_lateral_area :
  ∀ (r h l : ℝ), (π / 6 = (1 / 3) * π * r^2 * h) →
    (h = 1 / (2 * r^2)) →
    (l = Real.sqrt (r^2 + h^2)) →
    ((π * r * l) ≥ (3 / 4 * π)) →
    (r = Real.sqrt (2) / 2) →
    (h / r = Real.sqrt (2)) :=
by
  intro r h l V_cond h_cond l_def min_lateral_area r_val
  -- Proof steps go here (omitted as per the instruction)
  sorry

end cone_tangent_min_lateral_area_l1422_142268


namespace width_of_field_l1422_142262

theorem width_of_field (W L : ℝ) (h1 : L = (7 / 5) * W) (h2 : 2 * L + 2 * W = 360) : W = 75 :=
sorry

end width_of_field_l1422_142262


namespace pure_imaginary_solution_l1422_142260

theorem pure_imaginary_solution (a : ℝ) (i : ℂ) (h : i*i = -1) : (∀ z : ℂ, z = 1 + a * i → (z ^ 2).re = 0) → (a = 1 ∨ a = -1) := by
  sorry

end pure_imaginary_solution_l1422_142260


namespace infinitely_many_perfect_squares_of_form_l1422_142245

theorem infinitely_many_perfect_squares_of_form (k : ℕ) (h : k > 0) : 
  ∃ (n : ℕ), ∃ m : ℕ, n * 2^k - 7 = m^2 :=
by
  sorry

end infinitely_many_perfect_squares_of_form_l1422_142245


namespace calc_f_7_2_l1422_142255

variable {f : ℝ → ℝ}

axiom f_odd : ∀ x, f (-x) = -f x
axiom f_periodic : ∀ x, f (x + 2) = f x
axiom f_sqrt_on_interval : ∀ x, 0 < x ∧ x ≤ 1 → f x = Real.sqrt x

theorem calc_f_7_2 : f (7 / 2) = -Real.sqrt 2 / 2 := by
  sorry

end calc_f_7_2_l1422_142255


namespace a_power_2018_plus_b_power_2018_eq_2_l1422_142210

noncomputable def f (x a b : ℝ) : ℝ := (x + a) / (x + b)

theorem a_power_2018_plus_b_power_2018_eq_2 (a b : ℝ) :
  (∀ x : ℝ, f x a b + f (1 / x) a b = 0) → a^2018 + b^2018 = 2 :=
by 
  sorry

end a_power_2018_plus_b_power_2018_eq_2_l1422_142210


namespace parking_space_area_l1422_142204

theorem parking_space_area (L W : ℕ) (h1 : L = 9) (h2 : 2 * W + L = 37) : L * W = 126 :=
by
  -- Proof omitted.
  sorry

end parking_space_area_l1422_142204


namespace greatest_common_multiple_of_9_and_15_less_than_120_l1422_142226

-- Definition of LCM.
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- The main theorem to be proved.
theorem greatest_common_multiple_of_9_and_15_less_than_120 : ∃ x, x = 90 ∧ x < 120 ∧ x % 9 = 0 ∧ x % 15 = 0 :=
by
  -- Proof goes here.
  sorry

end greatest_common_multiple_of_9_and_15_less_than_120_l1422_142226


namespace total_pears_l1422_142228

noncomputable def Jason_pears : ℝ := 46
noncomputable def Keith_pears : ℝ := 47
noncomputable def Mike_pears : ℝ := 12
noncomputable def Sarah_pears : ℝ := 32.5
noncomputable def Emma_pears : ℝ := (2 / 3) * Mike_pears
noncomputable def James_pears : ℝ := (2 * Sarah_pears) - 3

theorem total_pears :
  Jason_pears + Keith_pears + Mike_pears + Sarah_pears + Emma_pears + James_pears = 207.5 :=
by
  sorry

end total_pears_l1422_142228


namespace number_of_sampled_medium_stores_is_five_l1422_142271

-- Definitions based on the conditions
def total_stores : ℕ := 300
def large_stores : ℕ := 30
def medium_stores : ℕ := 75
def small_stores : ℕ := 195
def sample_size : ℕ := 20

-- Proportion calculation function
def medium_store_proportion := (medium_stores : ℚ) / (total_stores : ℚ)

-- Sampled medium stores calculation
def sampled_medium_stores := medium_store_proportion * (sample_size : ℚ)

-- Theorem stating the number of medium stores drawn using stratified sampling
theorem number_of_sampled_medium_stores_is_five :
  sampled_medium_stores = 5 := 
by 
  sorry

end number_of_sampled_medium_stores_is_five_l1422_142271


namespace triangle_height_and_segments_l1422_142252

-- Define the sides of the triangle
noncomputable def a : ℝ := 13
noncomputable def b : ℝ := 14
noncomputable def c : ℝ := 15

-- Define the height h and the segments m and 15 - m
noncomputable def m : ℝ := 6.6
noncomputable def h : ℝ := 11.2
noncomputable def base_segment_left : ℝ := m
noncomputable def base_segment_right : ℝ := c - m

-- The height and segments calculation theorem
theorem triangle_height_and_segments :
  h = 11.2 ∧ m = 6.6 ∧ (c - m) = 8.4 :=
by {
  sorry
}

end triangle_height_and_segments_l1422_142252


namespace problem_a_b_c_ge_neg2_l1422_142280

theorem problem_a_b_c_ge_neg2 {a b c : ℝ} (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  (a + 1 / b > -2) ∨ (b + 1 / c > -2) ∨ (c + 1 / a > -2) → False :=
by
  sorry

end problem_a_b_c_ge_neg2_l1422_142280


namespace find_ab_l1422_142293

noncomputable def perpendicular_condition (a b : ℝ) :=
  a * (a - 1) - b = 0

noncomputable def point_on_l1_condition (a b : ℝ) :=
  -3 * a + b + 4 = 0

noncomputable def parallel_condition (a b : ℝ) :=
  a + b * (a - 1) = 0

noncomputable def distance_condition (a : ℝ) :=
  4 = abs ((-a) / (a - 1))

theorem find_ab (a b : ℝ) :
  (perpendicular_condition a b ∧ point_on_l1_condition a b ∧
   parallel_condition a b ∧ distance_condition a) →
  ((a = 2 ∧ b = 2) ∨ (a = 2 ∧ b = -2) ∨ (a = -2 ∧ b = 2)) :=
by
  sorry

end find_ab_l1422_142293


namespace complete_the_square_l1422_142246

theorem complete_the_square :
  ∀ (x : ℝ), (x^2 + 14 * x + 24 = 0) → (∃ c d : ℝ, (x + c)^2 = d ∧ d = 25) :=
by
  intro x h
  sorry

end complete_the_square_l1422_142246


namespace functional_expression_value_at_x_equals_zero_l1422_142227

-- Define the basic properties
def y_inversely_proportional_to_x_plus_2 (y x : ℝ) : Prop :=
  ∃ k : ℝ, y = k / (x + 2)

-- Given condition: y = 3 when x = -1
def condition (y x : ℝ) : Prop :=
  y = 3 ∧ x = -1

-- Theorems to prove
theorem functional_expression (y x : ℝ) :
  y_inversely_proportional_to_x_plus_2 y x ∧ condition y x → y = 3 / (x + 2) :=
by
  sorry

theorem value_at_x_equals_zero (y x : ℝ) :
  y_inversely_proportional_to_x_plus_2 y x ∧ condition y x → (y = 3 / (x + 2) ∧ x = 0 → y = 3 / 2) :=
by
  sorry

end functional_expression_value_at_x_equals_zero_l1422_142227


namespace size_of_coffee_cup_l1422_142264

-- Define the conditions and the final proof statement
variable (C : ℝ) (h1 : (1/4) * C) (h2 : (1/2) * C) (remaining_after_cold : (1/4) * C - 1 = 2)

theorem size_of_coffee_cup : C = 6 := by
  -- Here the proof would go, but we omit it with sorry
  sorry

end size_of_coffee_cup_l1422_142264


namespace mary_income_percentage_more_than_tim_l1422_142243

variables (J T M : ℝ)
-- Define the conditions
def condition1 := T = 0.5 * J -- Tim's income is 50% less than Juan's
def condition2 := M = 0.8 * J -- Mary's income is 80% of Juan's

-- Define the theorem stating the question and the correct answer
theorem mary_income_percentage_more_than_tim (J T M : ℝ) 
  (h1 : T = 0.5 * J) 
  (h2 : M = 0.8 * J) : 
  (M - T) / T * 100 = 60 := 
  by sorry

end mary_income_percentage_more_than_tim_l1422_142243


namespace perfect_squares_of_diophantine_l1422_142259

theorem perfect_squares_of_diophantine (a b : ℤ) (h : 2 * a^2 + a = 3 * b^2 + b) :
  ∃ k m : ℤ, (a - b) = k^2 ∧ (2 * a + 2 * b + 1) = m^2 := by
  sorry

end perfect_squares_of_diophantine_l1422_142259


namespace present_condition_l1422_142234

variable {α : Type} [Finite α]

-- We will represent children as members of a type α and assume there are precisely 3n children.
variable (n : ℕ) (h_odd : odd n) [h : Fintype α] (card_3n : Fintype.card α = 3 * n)

noncomputable def makes_present_to (A B : α) : α := sorry -- Create a function that maps pairs of children to exactly one child.

theorem present_condition : ∀ (A B C : α), makes_present_to A B = C → makes_present_to A C = B :=
sorry

end present_condition_l1422_142234


namespace solve_inequalities_l1422_142213

-- Define the interval [-1, 1]
def interval := {x : ℝ | -1 ≤ x ∧ x ≤ 1}

-- State the problem
theorem solve_inequalities :
  {x : ℝ | 3 * x^2 + 2 * x - 9 ≤ 0 ∧ x ≥ -1} = interval := 
sorry

end solve_inequalities_l1422_142213


namespace find_six_quotients_l1422_142208

def is_5twos_3ones (n: ℕ) : Prop :=
  n.digits 10 = [2, 2, 2, 2, 2, 1, 1, 1]

def divides_by_7 (n: ℕ) : Prop :=
  n % 7 = 0

theorem find_six_quotients:
  ∃ n₁ n₂ n₃ n₄ n₅: ℕ, 
    n₁ ≠ n₂ ∧ n₁ ≠ n₃ ∧ n₂ ≠ n₃ ∧ n₁ ≠ n₄ ∧ n₂ ≠ n₄ ∧ n₃ ≠ n₄ ∧ n₁ ≠ n₅ ∧ n₂ ≠ n₅ ∧ n₃ ≠ n₅ ∧ n₄ ≠ n₅ ∧
    is_5twos_3ones n₁ ∧ is_5twos_3ones n₂ ∧ is_5twos_3ones n₃ ∧ is_5twos_3ones n₄ ∧ is_5twos_3ones n₅ ∧
    divides_by_7 n₁ ∧ divides_by_7 n₂ ∧ divides_by_7 n₃ ∧ divides_by_7 n₄ ∧ divides_by_7 n₅ ∧
    n₁ / 7 = 1744603 ∧ n₂ / 7 = 3031603 ∧ n₃ / 7 = 3160303 ∧ n₄ / 7 = 3017446 ∧ n₅ / 7 = 3030316 :=
sorry

end find_six_quotients_l1422_142208


namespace monomial_2024_l1422_142240

def monomial (n : ℕ) : ℤ × ℕ := ((-1)^(n + 1) * (2 * n - 1), n)

theorem monomial_2024 :
  monomial 2024 = (-4047, 2024) :=
sorry

end monomial_2024_l1422_142240


namespace initial_volume_of_mixture_l1422_142265

theorem initial_volume_of_mixture
  (x : ℕ)
  (h1 : 3 * x / (2 * x + 1) = 4 / 3)
  (h2 : x = 4) :
  5 * x = 20 :=
by
  sorry

end initial_volume_of_mixture_l1422_142265


namespace bananas_per_chimp_per_day_l1422_142201

theorem bananas_per_chimp_per_day (total_chimps total_bananas : ℝ) (h_chimps : total_chimps = 45) (h_bananas : total_bananas = 72) :
  total_bananas / total_chimps = 1.6 :=
by
  rw [h_chimps, h_bananas]
  norm_num

end bananas_per_chimp_per_day_l1422_142201


namespace no_integer_coeff_trinomials_with_integer_roots_l1422_142247

theorem no_integer_coeff_trinomials_with_integer_roots :
  ¬ ∃ (a b c : ℤ),
    (∀ x : ℤ, a * x^2 + b * x + c = 0 → (∃ x1 x2 : ℤ, a = 0 ∧ x = x1 ∨ a ≠ 0 ∧ x = x1 ∨ x = x2 ∨ x = x1 ∧ x = x2)) ∧
    (∀ x : ℤ, (a + 1) * x^2 + (b + 1) * x + (c + 1) = 0 → (∃ x1 x2 : ℤ, (a + 1) = 0 ∧ x = x1 ∨ (a + 1) ≠ 0 ∧ x = x1 ∨ x = x2 ∨ x = x1 ∧ x = x2)) :=
by
  sorry

end no_integer_coeff_trinomials_with_integer_roots_l1422_142247


namespace solution_set_of_inequality_l1422_142273

variable {R : Type*} [LinearOrderedField R]

def odd_function (f : R → R) : Prop :=
  ∀ x : R, f (-x) = -f x

theorem solution_set_of_inequality
  (f : R → R)
  (odd_f : odd_function f)
  (h1 : f (-2) = 0)
  (h2 : ∀ (x1 x2 : R), x1 ≠ x2 ∧ 0 < x1 ∧ 0 < x2 → (x2 * f x1 - x1 * f x2) / (x1 - x2) < 0) :
  { x : R | (f x) / x < 0 } = { x : R | x < -2 } ∪ { x : R | x > 2 } := 
sorry

end solution_set_of_inequality_l1422_142273


namespace domain_of_function_l1422_142232

def valid_domain (x : ℝ) : Prop :=
  (2 - x ≥ 0) ∧ (x > 0) ∧ (x ≠ 2)

theorem domain_of_function :
  {x : ℝ | ∃ (y : ℝ), y = x ∧ valid_domain x} = {x | 0 < x ∧ x < 2} :=
by
  sorry

end domain_of_function_l1422_142232


namespace movie_duration_l1422_142287

theorem movie_duration :
  let start_time := (13, 30)
  let end_time := (14, 50)
  let hours := end_time.1 - start_time.1
  let minutes := end_time.2 - start_time.2
  (if minutes < 0 then (hours - 1, minutes + 60) else (hours, minutes)) = (1, 20) := by
    sorry

end movie_duration_l1422_142287


namespace square_area_in_right_triangle_l1422_142286

theorem square_area_in_right_triangle (XY ZC : ℝ) (hXY : XY = 40) (hZC : ZC = 70) : 
  ∃ s : ℝ, s^2 = 2800 ∧ s = (40 * 70) / (XY + ZC) := 
by
  sorry

end square_area_in_right_triangle_l1422_142286


namespace range_of_a_l1422_142217

theorem range_of_a (a : ℝ) (x : ℝ) : (x > a ∧ x > 1) → (x > 1) → (a ≤ 1) :=
by 
  intros hsol hx
  sorry

end range_of_a_l1422_142217


namespace score_order_l1422_142215

theorem score_order (a b c d : ℕ) 
  (h1 : b + d = a + c)
  (h2 : a + b > c + d)
  (h3 : d > b + c) :
  a > d ∧ d > b ∧ b > c := 
by
  sorry

end score_order_l1422_142215


namespace tangent_line_equation_l1422_142297

theorem tangent_line_equation (x y : ℝ) :
  (y = Real.exp x + 2) →
  (x = 0) →
  (y = 3) →
  (Real.exp x = 1) →
  (x - y + 3 = 0) :=
by
  intros h_eq h_x h_y h_slope
  -- The following proof will use the conditions to show the tangent line equation.
  sorry

end tangent_line_equation_l1422_142297


namespace product_of_roots_l1422_142296

noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

noncomputable def f_prime (a b c : ℝ) (x : ℝ) : ℝ :=
  3 * a * x^2 + 2 * b * x + c

theorem product_of_roots (a b c d x₁ x₂ : ℝ) 
  (h1 : f a b c d 0 = 0)
  (h2 : f a b c d x₁ = 0)
  (h3 : f a b c d x₂ = 0)
  (h_ext1 : f_prime a b c 1 = 0)
  (h_ext2 : f_prime a b c 2 = 0) :
  x₁ * x₂ = 6 :=
sorry

end product_of_roots_l1422_142296


namespace equal_roots_m_eq_minus_half_l1422_142291

theorem equal_roots_m_eq_minus_half (x m : ℝ) 
  (h_eq: ∀ x, ( (x * (x - 1) - (m + 1)) / ((x - 1) * (m - 1)) = x / m )) :
  m = -1/2 := by 
  sorry

end equal_roots_m_eq_minus_half_l1422_142291


namespace prove_correct_statement_l1422_142272

-- Define the conditions; we use the negation of incorrect statements
def condition1 (a b : ℝ) : Prop := a ≠ b → ¬((a - b > 0) → (a > 0 ∧ b > 0))
def condition2 (x : ℝ) : Prop := ¬(|x| > 0)
def condition4 (x : ℝ) : Prop := x ≠ 0 → (¬(∃ y, y = 1 / x))

-- Define the statement we want to prove as the correct one
def correct_statement (q : ℚ) : Prop := 0 - q = -q

-- The main theorem that combines conditions and proves the correct statement
theorem prove_correct_statement (a b : ℝ) (q : ℚ) :
  condition1 a b →
  condition2 a →
  condition4 a →
  correct_statement q :=
  by
  intros h1 h2 h4
  unfold correct_statement
  -- Proof goes here
  sorry

end prove_correct_statement_l1422_142272


namespace vertex_and_segment_condition_g_monotonically_increasing_g_minimum_value_l1422_142212

def f (x : ℝ) : ℝ := -x^2 + 2 * x + 15
def g (x a : ℝ) : ℝ := (2 - 2 * a) * x - f x

theorem vertex_and_segment_condition : 
  (f 1 = 16) ∧ ∃ x1 x2 : ℝ, (f x1 = 0) ∧ (f x2 = 0) ∧ (x2 - x1 = 8) := 
sorry

theorem g_monotonically_increasing (a : ℝ) :
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 2 → g x1 a ≤ g x2 a) ↔ a ≤ 0 :=
sorry

theorem g_minimum_value (a : ℝ) :
  (0 < a ∧ g 2 a = -4 * a - 11) ∨ (a < 0 ∧ g 0 a = -15) ∨ (0 ≤ a ∧ a ≤ 2 ∧ g a a = -a^2 - 15) :=
sorry

end vertex_and_segment_condition_g_monotonically_increasing_g_minimum_value_l1422_142212


namespace find_f_of_7_l1422_142229

variable {f : ℝ → ℝ}

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem find_f_of_7 (h1 : is_odd_function f)
                    (h2 : is_periodic_function f 4)
                    (h3 : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2) :
  f 7 = -2 := 
by
  sorry

end find_f_of_7_l1422_142229


namespace incorrect_variance_l1422_142253

noncomputable def normal_pdf (x : ℝ) : ℝ :=
  (1 / Real.sqrt (2 * Real.pi)) * Real.exp (- (x - 1)^2 / 2)

theorem incorrect_variance :
  (∫ x, normal_pdf x * x^2) - (∫ x, normal_pdf x * x)^2 ≠ 2 := 
sorry

end incorrect_variance_l1422_142253


namespace person_birth_year_and_age_l1422_142214

theorem person_birth_year_and_age (x y: ℕ) (h1: x ≤ 9) (h2: y ≤ 9) (hy: y = (88 - 10 * x) / (x + 1)):
  1988 - (1900 + 10 * x + y) = x * y → 1900 + 10 * x + y = 1964 ∧ 1988 - (1900 + 10 * x + y) = 24 :=
by
  sorry

end person_birth_year_and_age_l1422_142214


namespace loom_weaving_rate_l1422_142249

theorem loom_weaving_rate (total_cloth : ℝ) (total_time : ℝ) (rate : ℝ) 
  (h1 : total_cloth = 26) (h2 : total_time = 203.125) : rate = total_cloth / total_time := by
  sorry

#check loom_weaving_rate

end loom_weaving_rate_l1422_142249


namespace intersection_empty_implies_range_l1422_142298

-- Define the sets A and B
def setA := {x : ℝ | x ≤ 1} ∪ {x : ℝ | x ≥ 3}
def setB (a : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ a + 1}

-- Prove that if A ∩ B = ∅, then 1 < a < 2
theorem intersection_empty_implies_range (a : ℝ) (h : setA ∩ setB a = ∅) : 1 < a ∧ a < 2 :=
by
  sorry

end intersection_empty_implies_range_l1422_142298


namespace general_term_a_sum_of_bn_l1422_142200

-- Define sequences a_n and b_n
noncomputable def a (n : ℕ) : ℕ := 2 * n + 1
noncomputable def b (n : ℕ) : ℚ := 1 / ((2 * n + 1) * (2 * n + 3))

-- Conditions
lemma condition_1 (n : ℕ) : a n > 0 := by sorry
lemma condition_2 (n : ℕ) : (a n)^2 + 2 * (a n) = 4 * (n * (n + 1)) + 3 := 
  by sorry

-- Theorem for question 1
theorem general_term_a (n : ℕ) : a n = 2 * n + 1 := by sorry

-- Theorem for question 2
theorem sum_of_bn (n : ℕ) : 
  (Finset.range n).sum b = (n : ℚ) / (6 * n + 9) := by sorry

end general_term_a_sum_of_bn_l1422_142200


namespace ducks_arrival_quantity_l1422_142294

variable {initial_ducks : ℕ} (arrival_ducks : ℕ)

def initial_geese (initial_ducks : ℕ) := 2 * initial_ducks - 10

def remaining_geese (initial_ducks : ℕ) := initial_geese initial_ducks - 10

def remaining_ducks (initial_ducks arrival_ducks : ℕ) := initial_ducks + arrival_ducks

theorem ducks_arrival_quantity :
  initial_ducks = 25 →
  remaining_geese initial_ducks = 30 →
  remaining_geese initial_ducks = remaining_ducks initial_ducks arrival_ducks + 1 →
  arrival_ducks = 4 :=
by
sorry

end ducks_arrival_quantity_l1422_142294


namespace range_of_a_monotonically_decreasing_l1422_142244

noncomputable def f (x a : ℝ) := x^3 - a * x^2 + 1

theorem range_of_a_monotonically_decreasing (a : ℝ) :
  (∀ x y : ℝ, (0 < x ∧ x < 2) ∧ (0 < y ∧ y < 2) → x < y → f x a ≥ f y a) → (a ≥ 3) :=
by
  sorry

end range_of_a_monotonically_decreasing_l1422_142244


namespace percentage_increase_l1422_142203

variable (presentIncome : ℝ) (newIncome : ℝ)

theorem percentage_increase (h1 : presentIncome = 12000) (h2 : newIncome = 12240) :
  ((newIncome - presentIncome) / presentIncome) * 100 = 2 := by
  sorry

end percentage_increase_l1422_142203


namespace polynomial_root_range_l1422_142270

variable (a : ℝ)

theorem polynomial_root_range (h : ∀ x : ℂ, (2 * x^4 + a * x^3 + 9 * x^2 + a * x + 2 = 0) →
  ((x.re^2 + x.im^2 ≠ 1) ∧ x.im ≠ 0)) : (-2 * Real.sqrt 10 < a ∧ a < 2 * Real.sqrt 10) :=
sorry

end polynomial_root_range_l1422_142270


namespace income_ratio_l1422_142288

theorem income_ratio (I1 I2 E1 E2 : ℝ) (h1 : I1 = 5500) (h2 : E1 = I1 - 2200) (h3 : E2 = I2 - 2200) (h4 : E1 / E2 = 3 / 2) : I1 / I2 = 5 / 4 := by
  -- This is where the proof would go, but it's omitted for brevity.
  sorry

end income_ratio_l1422_142288


namespace algebraic_expression_value_l1422_142256

theorem algebraic_expression_value (a : ℝ) (h : 2 * a^2 + 3 * a - 5 = 0) : 6 * a^2 + 9 * a - 5 = 10 :=
by
  sorry

end algebraic_expression_value_l1422_142256


namespace most_economical_speed_and_cost_l1422_142242

open Real

theorem most_economical_speed_and_cost :
  ∀ (x : ℝ),
  (120:ℝ) / x * 36 + (120:ℝ) / x * 6 * (4 + x^2 / 360) = ((7200:ℝ) / x) + 2 * x → 
  50 ≤ x ∧ x ≤ 100 → 
  (∀ v : ℝ, (50 ≤ v ∧ v ≤ 100) → 
  (120 / v * 36 + 120 / v * 6 * (4 + v^2 / 360) ≤ 120 / x * 36 + 120 / x * 6 * (4 + x^2 / 360)) ) → 
  x = 60 → 
  (120 / x * 36 + 120 / x * 6 * (4 + x^2 / 360) = 240) :=
by
  intros x hx bounds min_cost opt_speed
  sorry

end most_economical_speed_and_cost_l1422_142242


namespace weekly_milk_production_l1422_142274

-- Conditions
def number_of_cows : ℕ := 52
def milk_per_cow_per_day : ℕ := 1000
def days_in_week : ℕ := 7

-- Statement to prove
theorem weekly_milk_production : (number_of_cows * milk_per_cow_per_day * days_in_week) = 364000 := by
  sorry

end weekly_milk_production_l1422_142274


namespace simplify_expression_l1422_142205

theorem simplify_expression (x : ℝ) (h1 : x ≠ 0) (h2 : 1 - x ≠ 0) :
  (1 - x) / x / ((1 - x) / x^2) = x := 
by 
  sorry

end simplify_expression_l1422_142205


namespace george_speed_to_school_l1422_142284

theorem george_speed_to_school :
  ∀ (D S_1 S_2 D_1 S_x : ℝ),
  D = 1.5 ∧ S_1 = 3 ∧ S_2 = 2 ∧ D_1 = 0.75 →
  S_x = (D - D_1) / ((D / S_1) - (D_1 / S_2)) →
  S_x = 6 :=
by
  intros D S_1 S_2 D_1 S_x h1 h2
  rw [h1.1, h1.2.1, h1.2.2.1, h1.2.2.2] at *
  sorry

end george_speed_to_school_l1422_142284


namespace smallest_digit_never_in_units_place_of_odd_number_l1422_142295

theorem smallest_digit_never_in_units_place_of_odd_number :
  ∀ d : ℕ, (d < 10 ∧ (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0) :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l1422_142295


namespace other_endpoint_product_l1422_142299

theorem other_endpoint_product :
  ∀ (x y : ℤ), 
    (3 = (x + 7) / 2) → 
    (-5 = (y - 1) / 2) → 
    x * y = 9 :=
by
  intro x y h1 h2
  sorry

end other_endpoint_product_l1422_142299


namespace dinner_time_correct_l1422_142254

-- Definitions based on the conditions in the problem
def pounds_per_turkey : Nat := 16
def roasting_time_per_pound : Nat := 15  -- minutes
def num_turkeys : Nat := 2
def minutes_per_hour : Nat := 60
def latest_start_time_hours : Nat := 10

-- The total roasting time in hours
def total_roasting_time_hours : Nat := 
  (roasting_time_per_pound * pounds_per_turkey * num_turkeys) / minutes_per_hour

-- The expected dinner time
def expected_dinner_time_hours : Nat := latest_start_time_hours + total_roasting_time_hours

-- The proof problem
theorem dinner_time_correct : expected_dinner_time_hours = 18 := 
by
  -- Proof goes here
  sorry

end dinner_time_correct_l1422_142254


namespace find_two_digit_number_l1422_142231

def product_of_digits (n : ℕ) : ℕ := 
-- Implementation that calculates the product of the digits of n
sorry

def sum_of_digits (n : ℕ) : ℕ := 
-- Implementation that calculates the sum of the digits of n
sorry

theorem find_two_digit_number (M : ℕ) (h1 : 10 ≤ M ∧ M < 100) (h2 : M = product_of_digits M + sum_of_digits M + 1) : M = 18 :=
by
  sorry

end find_two_digit_number_l1422_142231


namespace consecutive_integers_average_and_product_l1422_142220

theorem consecutive_integers_average_and_product (n m : ℤ) (hnm : n ≤ m) 
  (h1 : (n + m) / 2 = 20) 
  (h2 : n * m = 391) :  m - n + 1 = 7 :=
  sorry

end consecutive_integers_average_and_product_l1422_142220


namespace power_difference_divisible_by_10000_l1422_142236

theorem power_difference_divisible_by_10000 (a b : ℤ) (m : ℤ) (h : a - b = 100 * m) : ∃ k : ℤ, a^100 - b^100 = 10000 * k := by
  sorry

end power_difference_divisible_by_10000_l1422_142236


namespace quadratic_roots_l1422_142251

theorem quadratic_roots (m x1 x2 : ℝ) 
  (h1 : 2*x1^2 + 4*m*x1 + m = 0)
  (h2 : 2*x2^2 + 4*m*x2 + m = 0)
  (h3 : x1 ≠ x2)
  (h4 : x1^2 + x2^2 = 3/16) : 
  m = -1/8 := 
sorry

end quadratic_roots_l1422_142251


namespace system_of_equations_solution_l1422_142267

theorem system_of_equations_solution (x1 x2 x3 x4 x5 : ℝ) (h1 : x1 + x2 = x3^2) (h2 : x2 + x3 = x4^2)
  (h3 : x3 + x4 = x5^2) (h4 : x4 + x5 = x1^2) (h5 : x5 + x1 = x2^2) :
  x1 = 2 ∧ x2 = 2 ∧ x3 = 2 ∧ x4 = 2 ∧ x5 = 2 := 
sorry

end system_of_equations_solution_l1422_142267


namespace department_store_earnings_l1422_142202

theorem department_store_earnings :
  let original_price : ℝ := 1000000
  let discount_rate : ℝ := 0.1
  let prizes := [ (5, 1000), (10, 500), (20, 200), (40, 100), (5000, 10) ]
  let A_earnings := original_price * (1 - discount_rate)
  let total_prizes := prizes.foldl (fun sum (count, amount) => sum + count * amount) 0
  let B_earnings := original_price - total_prizes
  (B_earnings - A_earnings) >= 32000 := by
  sorry

end department_store_earnings_l1422_142202


namespace robot_min_steps_l1422_142283

theorem robot_min_steps {a b : ℕ} (ha : 0 < a) (hb : 0 < b) : ∃ n, n = a + b - Nat.gcd a b :=
by
  sorry

end robot_min_steps_l1422_142283


namespace time_taken_by_A_l1422_142266

-- Definitions for the problem conditions
def race_distance : ℕ := 1000  -- in meters
def A_beats_B_by_distance : ℕ := 48  -- in meters
def A_beats_B_by_time : ℕ := 12  -- in seconds

-- The formal statement to prove in Lean
theorem time_taken_by_A :
  ∃ T_a : ℕ, (1000 * (T_a + 12) = 952 * T_a) ∧ T_a = 250 :=
by
  sorry

end time_taken_by_A_l1422_142266


namespace steve_paid_18_l1422_142257

-- Define the conditions
def mike_price : ℝ := 5
def steve_multiplier : ℝ := 2
def shipping_rate : ℝ := 0.8

-- Define Steve's cost calculation
def steve_total_cost : ℝ :=
  let steve_dvd_price := steve_multiplier * mike_price
  let shipping_cost := shipping_rate * steve_dvd_price
  steve_dvd_price + shipping_cost

-- Prove that Steve's total payment is 18.
theorem steve_paid_18 : steve_total_cost = 18 := by
  -- Provide a placeholder for the proof
  sorry

end steve_paid_18_l1422_142257


namespace wax_he_has_l1422_142237

def total_wax : ℕ := 353
def additional_wax : ℕ := 22

theorem wax_he_has : total_wax - additional_wax = 331 := by
  sorry

end wax_he_has_l1422_142237


namespace therapy_sessions_l1422_142289

theorem therapy_sessions (F A n : ℕ) 
  (h1 : F = A + 25)
  (h2 : F + A = 115)
  (h3 : F + (n - 1) * A = 250) : 
  n = 5 := 
by sorry

end therapy_sessions_l1422_142289


namespace ball_bounce_height_l1422_142275

theorem ball_bounce_height :
  ∃ (k : ℕ), 10 * (1 / 2) ^ k < 1 ∧ (∀ m < k, 10 * (1 / 2) ^ m ≥ 1) :=
sorry

end ball_bounce_height_l1422_142275


namespace geometric_common_ratio_of_arithmetic_seq_l1422_142250

theorem geometric_common_ratio_of_arithmetic_seq 
  (a : ℕ → ℝ) (d q : ℝ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_a1 : a 1 = 2)
  (h_nonzero_diff : d ≠ 0)
  (h_geo_seq : a 1 = 2 ∧ a 3 = 2 * q ∧ a 11 = 2 * q^2) : 
  q = 4 := 
by
  sorry

end geometric_common_ratio_of_arithmetic_seq_l1422_142250


namespace quadratic_inequality_empty_solution_set_l1422_142239

theorem quadratic_inequality_empty_solution_set
  (a b c : ℝ)
  (h₁ : a > 0)
  (h₂ : ¬ ∃ x : ℝ, a * x^2 + b * x + c = 0) :
  {x : ℝ | a * x^2 + b * x + c < 0} = ∅ := 
by sorry

end quadratic_inequality_empty_solution_set_l1422_142239


namespace memorable_numbers_count_l1422_142292

def is_memorable_number (d : Fin 10 → Fin 8 → ℕ) : Prop :=
  d 0 0 = d 1 0 ∧ d 0 1 = d 1 1 ∧ d 0 2 = d 1 2 ∧ d 0 3 = d 1 3

theorem memorable_numbers_count : 
  ∃ n : ℕ, n = 10000 ∧ ∀ (d : Fin 10 → Fin 8 → ℕ), is_memorable_number d → n = 10000 :=
sorry

end memorable_numbers_count_l1422_142292


namespace smallest_whole_number_greater_than_sum_l1422_142261

theorem smallest_whole_number_greater_than_sum : 
  (3 + (1 / 3) + 4 + (1 / 4) + 6 + (1 / 6) + 7 + (1 / 7)) < 21 :=
sorry

end smallest_whole_number_greater_than_sum_l1422_142261


namespace smallest_n_produces_terminating_decimal_l1422_142248

noncomputable def smallest_n := 12

theorem smallest_n_produces_terminating_decimal (n : ℕ) (h_pos: 0 < n) : 
    (∀ m : ℕ, m > 113 → (n = m - 113 → (∃ k : ℕ, 1 ≤ k ∧ (m = 2^k ∨ m = 5^k)))) :=
by
  sorry

end smallest_n_produces_terminating_decimal_l1422_142248


namespace power_equiv_l1422_142230

theorem power_equiv (x_0 : ℝ) (h : x_0 ^ 11 + x_0 ^ 7 + x_0 ^ 3 = 1) : x_0 ^ 4 + x_0 ^ 3 - 1 = x_0 ^ 15 :=
by
  -- the proof goes here
  sorry

end power_equiv_l1422_142230


namespace solution_set_correct_l1422_142225

theorem solution_set_correct (a b : ℝ) :
  (∀ x : ℝ, - 1 / 2 < x ∧ x < 1 / 3 → ax^2 + bx + 2 > 0) →
  (a - b = -10) :=
by
  sorry

end solution_set_correct_l1422_142225


namespace work_time_A_and_C_together_l1422_142285

theorem work_time_A_and_C_together
  (A_work B_work C_work : ℝ)
  (hA : A_work = 1/3)
  (hB : B_work = 1/6)
  (hBC : B_work + C_work = 1/3) :
  1 / (A_work + C_work) = 2 := by
  sorry

end work_time_A_and_C_together_l1422_142285


namespace forty_percent_of_number_l1422_142258

theorem forty_percent_of_number (N : ℝ) (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 15) :
  0.40 * N = 180 :=
by
  sorry

end forty_percent_of_number_l1422_142258


namespace ones_digit_of_prime_sequence_l1422_142269

theorem ones_digit_of_prime_sequence (p q r s : ℕ) (h1 : p > 5) 
    (h2 : p < q ∧ q < r ∧ r < s) (h3 : q - p = 8 ∧ r - q = 8 ∧ s - r = 8) 
    (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs : Nat.Prime s) : 
    p % 10 = 3 :=
by
  sorry

end ones_digit_of_prime_sequence_l1422_142269


namespace valid_sequences_l1422_142206

-- Define the transformation function for a ten-digit number
noncomputable def transform (n : ℕ) : ℕ := sorry

-- Given sequences
def seq1 := 1101111111
def seq2 := 1201201020
def seq3 := 1021021020
def seq4 := 0112102011

-- The proof problem statement
theorem valid_sequences :
  (transform 1101111111 = seq1) ∧
  (transform 1021021020 = seq3) ∧
  (transform 0112102011 = seq4) :=
sorry

end valid_sequences_l1422_142206


namespace net_percentage_change_l1422_142279

theorem net_percentage_change (k m : ℝ) : 
  let scale_factor_1 := 1 - k / 100
  let scale_factor_2 := 1 + m / 100
  let overall_scale_factor := scale_factor_1 * scale_factor_2
  let percentage_change := (overall_scale_factor - 1) * 100
  percentage_change = m - k - k * m / 100 := 
by 
  sorry

end net_percentage_change_l1422_142279


namespace largest_of_three_consecutive_integers_sum_90_is_31_l1422_142223

theorem largest_of_three_consecutive_integers_sum_90_is_31 :
  ∃ (a b c : ℤ), (a + b + c = 90) ∧ (b = a + 1) ∧ (c = b + 1) ∧ (c = 31) :=
by
  sorry

end largest_of_three_consecutive_integers_sum_90_is_31_l1422_142223


namespace problem_2_l1422_142277

noncomputable def f (x a : ℝ) : ℝ := (1 / 2) * x ^ 2 + a * Real.log (1 - x)

theorem problem_2 (a : ℝ) (x₁ x₂ : ℝ) (h₀ : 0 < a) (h₁ : a < 1/4) (h₂ : f x₂ a = 0) 
  (h₃ : f x₁ a = 0) (hx₁ : 0 < x₁) (hx₂ : x₁ < 1/2) (h₄ : x₁ < x₂) :
  f x₂ a - x₁ > - (3 + Real.log 4) / 8 := sorry

end problem_2_l1422_142277


namespace evaluate_at_2_l1422_142211

def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem evaluate_at_2 : f 2 = 62 := 
by
  sorry

end evaluate_at_2_l1422_142211


namespace plane_equation_l1422_142281

theorem plane_equation 
  (P Q : ℝ×ℝ×ℝ) (A B : ℝ×ℝ×ℝ)
  (hp : P = (-1, 2, 5))
  (hq : Q = (3, -4, 1))
  (ha : A = (0, -2, -1))
  (hb : B = (3, 2, -1)) :
  ∃ (a b c d : ℝ), (a = 3 ∧ b = 4 ∧ c = 0 ∧ d = 1) ∧ (∀ x y z : ℝ, a * (x - 1) + b * (y + 1) + c * (z - 3) = d) :=
by
  sorry

end plane_equation_l1422_142281


namespace percent_of_dollar_in_pocket_l1422_142221

def value_of_penny : ℕ := 1  -- value of one penny in cents
def value_of_nickel : ℕ := 5  -- value of one nickel in cents
def value_of_half_dollar : ℕ := 50 -- value of one half-dollar in cents

def pennies : ℕ := 3  -- number of pennies
def nickels : ℕ := 2  -- number of nickels
def half_dollars : ℕ := 1  -- number of half-dollars

def total_value_in_cents : ℕ :=
  (pennies * value_of_penny) + (nickels * value_of_nickel) + (half_dollars * value_of_half_dollar)

def value_of_dollar_in_cents : ℕ := 100

def percent_of_dollar (value : ℕ) (total : ℕ) : ℚ := (value / total) * 100

theorem percent_of_dollar_in_pocket : percent_of_dollar total_value_in_cents value_of_dollar_in_cents = 63 :=
by
  sorry

end percent_of_dollar_in_pocket_l1422_142221


namespace cannot_form_isosceles_triangle_l1422_142263

theorem cannot_form_isosceles_triangle :
  ¬ ∃ (sticks : Finset ℕ) (a b c : ℕ), a ∈ sticks ∧ b ∈ sticks ∧ c ∈ sticks ∧
  a + b > c ∧ a + c > b ∧ b + c > a ∧ -- Triangle inequality
  (a = b ∨ b = c ∨ a = c) ∧ -- Isosceles condition
  sticks ⊆ {1, 2, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^9} := sorry

end cannot_form_isosceles_triangle_l1422_142263
