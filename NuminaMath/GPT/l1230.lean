import Mathlib

namespace NUMINAMATH_GPT_value_of_x_l1230_123073

theorem value_of_x (x y : ℝ) (h1 : x / y = 9 / 5) (h2 : y = 25) : x = 45 := by
  sorry

end NUMINAMATH_GPT_value_of_x_l1230_123073


namespace NUMINAMATH_GPT_probability_diagonals_intersect_l1230_123051

theorem probability_diagonals_intersect {n : ℕ} :
  (2 * n + 1 > 2) → 
  ∀ (total_diagonals : ℕ) (total_combinations : ℕ) (intersecting_pairs : ℕ),
    total_diagonals = 2 * n^2 - n - 1 →
    total_combinations = (total_diagonals * (total_diagonals - 1)) / 2 →
    intersecting_pairs = ((2 * n + 1) * n * (2 * n - 1) * (n - 1)) / 6 →
    (intersecting_pairs : ℚ) / (total_combinations : ℚ) = n * (2 * n - 1) / (3 * (2 * n^2 - n - 2)) := sorry

end NUMINAMATH_GPT_probability_diagonals_intersect_l1230_123051


namespace NUMINAMATH_GPT_cubic_feet_per_bag_l1230_123030

-- Definitions
def length_bed := 8 -- in feet
def width_bed := 4 -- in feet
def height_bed := 1 -- in feet
def number_of_beds := 2
def number_of_bags := 16

-- Theorem statement
theorem cubic_feet_per_bag : 
  (length_bed * width_bed * height_bed * number_of_beds) / number_of_bags = 4 :=
by
  sorry

end NUMINAMATH_GPT_cubic_feet_per_bag_l1230_123030


namespace NUMINAMATH_GPT_find_x_l1230_123000

theorem find_x (x : ℝ) : abs (2 * x - 1) = 3 * x + 6 ∧ x + 2 > 0 ↔ x = -1 := 
by
  sorry

end NUMINAMATH_GPT_find_x_l1230_123000


namespace NUMINAMATH_GPT_max_three_digit_divisible_by_4_sequence_l1230_123010

theorem max_three_digit_divisible_by_4_sequence (a : ℕ → ℕ) (n : ℕ) (h1 : ∀ k ≤ n - 2, a (k + 2) = 3 * a (k + 1) - 2 * a k - 2)
(h2 : ∀ k1 k2, k1 < k2 → a k1 < a k2) (ha2022 : ∃ k, a k = 2022) (hn : n ≥ 3) :
  ∃ m : ℕ, ∀ k, 100 ≤ a k ∧ a k ≤ 999 → a k % 4 = 0 → m ≤ 225 := by
  sorry

end NUMINAMATH_GPT_max_three_digit_divisible_by_4_sequence_l1230_123010


namespace NUMINAMATH_GPT_center_square_side_length_l1230_123080

theorem center_square_side_length (s : ℝ) :
    let total_area := 120 * 120
    let l_shape_area := (5 / 24) * total_area
    let l_shape_total_area := 4 * l_shape_area
    let center_square_area := total_area - l_shape_total_area
    s^2 = center_square_area → s = 49 :=
by
  intro total_area l_shape_area l_shape_total_area center_square_area h
  sorry

end NUMINAMATH_GPT_center_square_side_length_l1230_123080


namespace NUMINAMATH_GPT_tangent_integer_values_l1230_123081

/-- From point P outside a circle with circumference 12π units, a tangent and a secant are drawn.
      The secant divides the circle into arcs with lengths m and n. Given that the length of the
      tangent t is the geometric mean between m and n, and that m is three times n, there are zero
      possible integer values for t. -/
theorem tangent_integer_values
  (circumference : ℝ) (m n t : ℝ)
  (h_circumference : circumference = 12 * Real.pi)
  (h_sum : m + n = 12 * Real.pi)
  (h_ratio : m = 3 * n)
  (h_tangent : t = Real.sqrt (m * n)) :
  ¬(∃ k : ℤ, t = k) := 
sorry

end NUMINAMATH_GPT_tangent_integer_values_l1230_123081


namespace NUMINAMATH_GPT_depth_of_water_is_60_l1230_123057

def dean_height : ℕ := 6
def depth_multiplier : ℕ := 10
def water_depth : ℕ := depth_multiplier * dean_height

theorem depth_of_water_is_60 : water_depth = 60 := by
  -- mathematical equivalent proof problem
  sorry

end NUMINAMATH_GPT_depth_of_water_is_60_l1230_123057


namespace NUMINAMATH_GPT_hyperbola_condition_l1230_123041

theorem hyperbola_condition (a : ℝ) (h : a > 0)
  (e : ℝ) (h_e : e = Real.sqrt (1 + 4 / (a^2))) :
  (e > Real.sqrt 2) ↔ (0 < a ∧ a < 1) := 
sorry

end NUMINAMATH_GPT_hyperbola_condition_l1230_123041


namespace NUMINAMATH_GPT_remainder_div_30_l1230_123086

-- Define the conditions as Lean definitions
variables (x y z p q : ℕ)

-- Hypotheses based on the conditions
def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

-- assuming the conditions
axiom x_div_by_4 : is_divisible_by x 4
axiom y_div_by_5 : is_divisible_by y 5
axiom z_div_by_6 : is_divisible_by z 6
axiom p_div_by_7 : is_divisible_by p 7
axiom q_div_by_3 : is_divisible_by q 3

-- Statement to be proved
theorem remainder_div_30 : ((x^3) * (y^2) * (z * p * q + (x + y)^3) - 10) % 30 = 20 :=
by {
  sorry -- the proof will go here
}

end NUMINAMATH_GPT_remainder_div_30_l1230_123086


namespace NUMINAMATH_GPT_find_larger_number_l1230_123049

variables (x y : ℝ)

def sum_cond : Prop := x + y = 17
def diff_cond : Prop := x - y = 7

theorem find_larger_number (h1 : sum_cond x y) (h2 : diff_cond x y) : x = 12 :=
sorry

end NUMINAMATH_GPT_find_larger_number_l1230_123049


namespace NUMINAMATH_GPT_largest_n_crates_same_number_oranges_l1230_123031

theorem largest_n_crates_same_number_oranges (total_crates : ℕ) 
  (crate_min_oranges : ℕ) (crate_max_oranges : ℕ) 
  (h1 : total_crates = 200) (h2 : crate_min_oranges = 100) (h3 : crate_max_oranges = 130) 
  : ∃ n : ℕ, n = 7 ∧ ∀ orange_count, crate_min_oranges ≤ orange_count ∧ orange_count ≤ crate_max_oranges → ∃ k, k = n ∧ ∃ t, t ≤ total_crates ∧ t ≥ k := 
sorry

end NUMINAMATH_GPT_largest_n_crates_same_number_oranges_l1230_123031


namespace NUMINAMATH_GPT_volume_parallelepiped_l1230_123071

noncomputable def volume_of_parallelepiped (m n p d : ℝ) : ℝ :=
  if h : m > 0 ∧ n > 0 ∧ p > 0 ∧ d > 0 then
    m * n * p * (d^3) / (m^2 + n^2 + p^2)^(3/2)
  else 0

theorem volume_parallelepiped (m n p d : ℝ) (hm : m > 0) (hn : n > 0) (hp : p > 0) (hd : d > 0) :
  volume_of_parallelepiped m n p d = m * n * p * (d^3) / (m^2 + n^2 + p^2)^(3/2) := by
  sorry

end NUMINAMATH_GPT_volume_parallelepiped_l1230_123071


namespace NUMINAMATH_GPT_sum_of_100_and_98_consecutive_diff_digits_l1230_123037

def S100 (n : ℕ) : ℕ := 50 * (2 * n + 99)
def S98 (n : ℕ) : ℕ := 49 * (2 * n + 297)

theorem sum_of_100_and_98_consecutive_diff_digits (n : ℕ) :
  ¬ (S100 n % 10 = S98 n % 10) :=
sorry

end NUMINAMATH_GPT_sum_of_100_and_98_consecutive_diff_digits_l1230_123037


namespace NUMINAMATH_GPT_sum_of_three_consecutive_integers_product_336_l1230_123093

theorem sum_of_three_consecutive_integers_product_336 :
  ∃ (n : ℕ), (n - 1) * n * (n + 1) = 336 ∧ (n - 1) + n + (n + 1) = 21 :=
sorry

end NUMINAMATH_GPT_sum_of_three_consecutive_integers_product_336_l1230_123093


namespace NUMINAMATH_GPT_monthly_income_calculation_l1230_123083

variable (deposit : ℝ)
variable (percentage : ℝ)
variable (monthly_income : ℝ)

theorem monthly_income_calculation 
    (h1 : deposit = 3800) 
    (h2 : percentage = 0.32) 
    (h3 : deposit = percentage * monthly_income) : 
    monthly_income = 11875 :=
by
  sorry

end NUMINAMATH_GPT_monthly_income_calculation_l1230_123083


namespace NUMINAMATH_GPT_a_18_value_l1230_123002

-- Define the concept of an "Equally Summed Sequence"
def equallySummedSequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  ∀ n, a n + a (n + 1) = c

-- Define the specific conditions for a_1 and the common sum
def specific_sequence (a : ℕ → ℝ) : Prop :=
  equallySummedSequence a 5 ∧ a 1 = 2

-- The theorem we want to prove
theorem a_18_value (a : ℕ → ℝ) (h : specific_sequence a) : a 18 = 3 :=
sorry

end NUMINAMATH_GPT_a_18_value_l1230_123002


namespace NUMINAMATH_GPT_Berengere_contribution_l1230_123075

theorem Berengere_contribution (cake_cost_in_euros : ℝ) (emily_dollars : ℝ) (exchange_rate : ℝ)
  (h1 : cake_cost_in_euros = 6)
  (h2 : emily_dollars = 5)
  (h3 : exchange_rate = 1.25) :
  cake_cost_in_euros - emily_dollars * (1 / exchange_rate) = 2 := by
  sorry

end NUMINAMATH_GPT_Berengere_contribution_l1230_123075


namespace NUMINAMATH_GPT_striped_jerseys_count_l1230_123062

noncomputable def totalSpent : ℕ := 80
noncomputable def longSleevedJerseyCost : ℕ := 15
noncomputable def stripedJerseyCost : ℕ := 10
noncomputable def numberOfLongSleevedJerseys : ℕ := 4

theorem striped_jerseys_count :
  (totalSpent - numberOfLongSleevedJerseys * longSleevedJerseyCost) / stripedJerseyCost = 2 := by
  sorry

end NUMINAMATH_GPT_striped_jerseys_count_l1230_123062


namespace NUMINAMATH_GPT_sandwich_not_condiment_percentage_l1230_123076

theorem sandwich_not_condiment_percentage :
  (total_weight : ℝ) → (condiment_weight : ℝ) →
  total_weight = 150 → condiment_weight = 45 →
  ((total_weight - condiment_weight) / total_weight) * 100 = 70 :=
by
  intros total_weight condiment_weight h_total h_condiment
  sorry

end NUMINAMATH_GPT_sandwich_not_condiment_percentage_l1230_123076


namespace NUMINAMATH_GPT_ratio_of_x_to_y_l1230_123052

-- Given condition: The percentage that y is less than x is 83.33333333333334%.
def percentage_less_than (x y : ℝ) : Prop := (x - y) / x = 0.8333333333333334

-- Prove: The ratio R = x / y is 1/6.
theorem ratio_of_x_to_y (x y : ℝ) (h : percentage_less_than x y) : x / y = 6 := 
by sorry

end NUMINAMATH_GPT_ratio_of_x_to_y_l1230_123052


namespace NUMINAMATH_GPT_Cody_money_final_l1230_123027

-- Define the initial amount of money Cody had
def Cody_initial : ℝ := 45.0

-- Define the birthday gift amount
def birthday_gift : ℝ := 9.0

-- Define the amount spent on the game
def game_expense : ℝ := 19.0

-- Define the percentage of remaining money spent on clothes as a fraction
def clothes_spending_fraction : ℝ := 0.40

-- Define the late birthday gift received
def late_birthday_gift : ℝ := 4.5

-- Define the final amount of money Cody has
def Cody_final : ℝ :=
  let after_birthday := Cody_initial + birthday_gift
  let after_game := after_birthday - game_expense
  let spent_on_clothes := clothes_spending_fraction * after_game
  let after_clothes := after_game - spent_on_clothes
  after_clothes + late_birthday_gift

theorem Cody_money_final : Cody_final = 25.5 := by
  sorry

end NUMINAMATH_GPT_Cody_money_final_l1230_123027


namespace NUMINAMATH_GPT_solution_to_equation_l1230_123078

theorem solution_to_equation (x y : ℕ → ℕ) (h1 : x 1 = 2) (h2 : y 1 = 3)
  (h3 : ∀ k, x (k + 1) = 3 * x k + 2 * y k)
  (h4 : ∀ k, y (k + 1) = 4 * x k + 3 * y k) :
  ∀ n, 2 * (x n)^2 + 1 = (y n)^2 := 
by
  sorry

end NUMINAMATH_GPT_solution_to_equation_l1230_123078


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1230_123068

theorem sufficient_but_not_necessary (x : ℝ) :
  (x < -1 → x^2 - 1 > 0) ∧ (∃ x, x^2 - 1 > 0 ∧ ¬(x < -1)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1230_123068


namespace NUMINAMATH_GPT_sum_of_numbers_l1230_123066

theorem sum_of_numbers : 72.52 + 12.23 + 5.21 = 89.96 :=
by sorry

end NUMINAMATH_GPT_sum_of_numbers_l1230_123066


namespace NUMINAMATH_GPT_find_n_l1230_123003

theorem find_n (n : ℝ) (h1 : ∀ m : ℝ, m = 4 → m^(m/2) = 4) : 
  n^(n/2) = 8 ↔ n = 2^Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l1230_123003


namespace NUMINAMATH_GPT_range_of_a_l1230_123005

-- Function definition
def f (x a : ℝ) : ℝ := -x^3 + 3 * a^2 * x - 4 * a

-- Main theorem statement
theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ x, f x a = 0) ↔ (a ∈ Set.Ioi (Real.sqrt 2)) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1230_123005


namespace NUMINAMATH_GPT_shaded_area_10x12_floor_l1230_123082

theorem shaded_area_10x12_floor :
  let tile_white_area := π + 1
  let tile_total_area := 4
  let tile_shaded_area := tile_total_area - tile_white_area
  let num_tiles := (10 / 2) * (12 / 2)
  let total_shaded_area := num_tiles * tile_shaded_area
  total_shaded_area = 90 - 30 * π :=
by
  let tile_white_area := π + 1
  let tile_total_area := 4
  let tile_shaded_area := tile_total_area - tile_white_area
  let num_tiles := (10 / 2) * (12 / 2)
  let total_shaded_area := num_tiles * tile_shaded_area
  show total_shaded_area = 90 - 30 * π
  sorry

end NUMINAMATH_GPT_shaded_area_10x12_floor_l1230_123082


namespace NUMINAMATH_GPT_sixth_graders_more_than_seventh_l1230_123091

def total_payment_seventh_graders : ℕ := 143
def total_payment_sixth_graders : ℕ := 195
def cost_per_pencil : ℕ := 13

theorem sixth_graders_more_than_seventh :
  (total_payment_sixth_graders / cost_per_pencil) - (total_payment_seventh_graders / cost_per_pencil) = 4 :=
  by
  sorry

end NUMINAMATH_GPT_sixth_graders_more_than_seventh_l1230_123091


namespace NUMINAMATH_GPT_number_of_younger_siblings_l1230_123094

-- Definitions based on the problem conditions
def Nicole_cards : ℕ := 400
def Cindy_cards : ℕ := 2 * Nicole_cards
def Combined_cards : ℕ := Nicole_cards + Cindy_cards
def Rex_cards : ℕ := Combined_cards / 2
def Rex_remaining_cards : ℕ := 150
def Total_shares : ℕ := Rex_cards / Rex_remaining_cards
def Rex_share : ℕ := 1

-- The theorem to prove how many younger siblings Rex has
theorem number_of_younger_siblings :
  Total_shares - Rex_share = 3 :=
  by
    sorry

end NUMINAMATH_GPT_number_of_younger_siblings_l1230_123094


namespace NUMINAMATH_GPT_tasks_to_shower_l1230_123069

-- Definitions of the conditions
def tasks_to_clean_house : Nat := 7
def tasks_to_make_dinner : Nat := 4
def minutes_per_task : Nat := 10
def total_minutes : Nat := 2 * 60

-- The theorem we want to prove
theorem tasks_to_shower (x : Nat) :
  total_minutes = (tasks_to_clean_house + tasks_to_make_dinner + x) * minutes_per_task →
  x = 1 := by
  sorry

end NUMINAMATH_GPT_tasks_to_shower_l1230_123069


namespace NUMINAMATH_GPT_expression_value_l1230_123096

theorem expression_value (x y : ℝ) (h1 : x + y = 17) (h2 : x * y = 17) :
  (x^2 - 17*x) * (y + 17/y) = -289 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l1230_123096


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1230_123061

theorem sufficient_but_not_necessary (x : ℝ) : 
  (x = 1 → x^2 = 1) ∧ ¬(x^2 = 1 → x = 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1230_123061


namespace NUMINAMATH_GPT_integer_pairs_satisfying_equation_and_nonnegative_product_l1230_123038

theorem integer_pairs_satisfying_equation_and_nonnegative_product :
  ∃ (pairs : List (ℤ × ℤ)), 
    (∀ p ∈ pairs, p.1 * p.2 ≥ 0 ∧ p.1^3 + p.2^3 + 99 * p.1 * p.2 = 33^3) ∧ 
    pairs.length = 35 :=
by sorry

end NUMINAMATH_GPT_integer_pairs_satisfying_equation_and_nonnegative_product_l1230_123038


namespace NUMINAMATH_GPT_simplify_complex_expression_l1230_123077

theorem simplify_complex_expression : 
  ∀ (i : ℂ), i^2 = -1 → 3 * (4 - 2 * i) + 2 * i * (3 + 2 * i) = 8 := 
by
  intros
  sorry

end NUMINAMATH_GPT_simplify_complex_expression_l1230_123077


namespace NUMINAMATH_GPT_candy_given_l1230_123054

theorem candy_given (A R G : ℕ) (h1 : A = 15) (h2 : R = 9) : G = 6 :=
by
  sorry

end NUMINAMATH_GPT_candy_given_l1230_123054


namespace NUMINAMATH_GPT_find_price_of_fourth_variety_theorem_l1230_123004

-- Define the variables and conditions
variables (P1 P2 P3 P4 : ℝ) (Q1 Q2 Q3 Q4 : ℝ) (P_avg : ℝ)

-- Given conditions
def price_of_fourth_variety : Prop :=
  P1 = 126 ∧
  P2 = 135 ∧
  P3 = 156 ∧
  P_avg = 165 ∧
  Q1 / Q2 = 2 / 3 ∧
  Q1 / Q3 = 2 / 4 ∧
  Q1 / Q4 = 2 / 5 ∧
  (P1 * Q1 + P2 * Q2 + P3 * Q3 + P4 * Q4) / (Q1 + Q2 + Q3 + Q4) = P_avg

-- Prove that the price of the fourth variety of tea is Rs. 205.8 per kg
theorem find_price_of_fourth_variety_theorem : price_of_fourth_variety P1 P2 P3 P4 Q1 Q2 Q3 Q4 P_avg → P4 = 205.8 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_price_of_fourth_variety_theorem_l1230_123004


namespace NUMINAMATH_GPT_determine_OQ_l1230_123019

theorem determine_OQ (l m n p O A B C D Q : ℝ) (h0 : O = 0)
  (hA : A = l) (hB : B = m) (hC : C = n) (hD : D = p)
  (hQ : l ≤ Q ∧ Q ≤ m)
  (h_ratio : (|C - Q| / |Q - D|) = (|B - Q| / |Q - A|)) :
  Q = (l + m) / 2 :=
sorry

end NUMINAMATH_GPT_determine_OQ_l1230_123019


namespace NUMINAMATH_GPT_algebraic_expression_eval_l1230_123018

theorem algebraic_expression_eval (a b : ℝ) 
  (h_eq : ∀ (x : ℝ), ¬(x ≠ 0 ∧ x ≠ 1 ∧ (x / (x - 1) + (x - 1) / x = (a + b * x) / (x^2 - x)))) :
  8 * a + 4 * b - 5 = 27 := 
sorry

end NUMINAMATH_GPT_algebraic_expression_eval_l1230_123018


namespace NUMINAMATH_GPT_workers_time_l1230_123059

variables (x y: ℝ)

theorem workers_time (h1 : (x > 0) ∧ (y > 0)) 
                     (h2 : (3/x + 2/y = 11/20)) 
                     (h3 : (1/x + 1/y = 1/2)) :
                     (x = 10 ∧ y = 8) := 
by
  sorry

end NUMINAMATH_GPT_workers_time_l1230_123059


namespace NUMINAMATH_GPT_isosceles_triangle_side_length_l1230_123026

theorem isosceles_triangle_side_length :
  let a := 1
  let b := Real.sqrt 3
  let right_triangle_area := (1 / 2) * a * b
  let isosceles_triangle_area := right_triangle_area / 3
  ∃ s, s = Real.sqrt 109 / 6 ∧ 
    (∀ (base height : ℝ), 
      (base = a / 3 ∨ base = b / 3) ∧
      height = (2 * isosceles_triangle_area) / base → 
      1 / 2 * base * height = isosceles_triangle_area) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_side_length_l1230_123026


namespace NUMINAMATH_GPT_infinitude_of_composite_z_l1230_123034

theorem infinitude_of_composite_z (a : ℕ) (h : ∃ k : ℕ, k > 1 ∧ a = 4 * k^4) : 
  ∀ n : ℕ, ¬ Prime (n^4 + a) :=
by sorry

end NUMINAMATH_GPT_infinitude_of_composite_z_l1230_123034


namespace NUMINAMATH_GPT_inequality_solution_l1230_123065

theorem inequality_solution (x : ℝ) : 
  (x^2 + 4 * x + 13 > 0) -> ((x - 4) / (x^2 + 4 * x + 13) ≥ 0 ↔ x ≥ 4) :=
by
  intro h_pos
  sorry

end NUMINAMATH_GPT_inequality_solution_l1230_123065


namespace NUMINAMATH_GPT_rectangle_area_288_l1230_123040

/-- A rectangle contains eight circles arranged in a 2x4 grid. Each circle has a radius of 3 inches.
    We are asked to prove that the area of the rectangle is 288 square inches. --/
noncomputable def circle_radius : ℝ := 3
noncomputable def circles_per_width : ℕ := 2
noncomputable def circles_per_length : ℕ := 4
noncomputable def circle_diameter : ℝ := 2 * circle_radius
noncomputable def rectangle_width : ℝ := circles_per_width * circle_diameter
noncomputable def rectangle_length : ℝ := circles_per_length * circle_diameter
noncomputable def rectangle_area : ℝ := rectangle_length * rectangle_width

theorem rectangle_area_288 :
  rectangle_area = 288 :=
by
  -- Proof of the area will be filled in here.
  sorry

end NUMINAMATH_GPT_rectangle_area_288_l1230_123040


namespace NUMINAMATH_GPT_garden_roller_diameter_l1230_123070

theorem garden_roller_diameter
  (l : ℝ) (A : ℝ) (r : ℕ) (pi : ℝ)
  (h_l : l = 2)
  (h_A : A = 44)
  (h_r : r = 5)
  (h_pi : pi = 22 / 7) :
  ∃ d : ℝ, d = 1.4 :=
by {
  sorry
}

end NUMINAMATH_GPT_garden_roller_diameter_l1230_123070


namespace NUMINAMATH_GPT_expression_for_f_l1230_123097

theorem expression_for_f {f : ℤ → ℤ} (h : ∀ x, f (x + 1) = 3 * x + 4) : ∀ x, f x = 3 * x + 1 :=
by
  sorry

end NUMINAMATH_GPT_expression_for_f_l1230_123097


namespace NUMINAMATH_GPT_min_total_rope_cut_l1230_123043

theorem min_total_rope_cut (len1 len2 len3 p1 p2 p3 p4: ℕ) (hl1 : len1 = 52) (hl2 : len2 = 37)
  (hl3 : len3 = 25) (hp1 : p1 = 7) (hp2 : p2 = 3) (hp3 : p3 = 1) 
  (hp4 : ∃ x y z : ℕ, x * p1 + y * p2 + z * p3 = len1 + len2 - len3 ∧ x + y + z ≤ 25) :
  p4 = 82 := 
sorry

end NUMINAMATH_GPT_min_total_rope_cut_l1230_123043


namespace NUMINAMATH_GPT_maximum_height_when_isosceles_l1230_123046

variable (c : ℝ) (c1 c2 : ℝ)

def right_angled_triangle (c1 c2 c : ℝ) : Prop :=
  c1 * c1 + c2 * c2 = c * c

def isosceles_right_triangle (c1 c2 : ℝ) : Prop :=
  c1 = c2

noncomputable def height_relative_to_hypotenuse (c : ℝ) : ℝ :=
  c / 2

theorem maximum_height_when_isosceles 
  (c1 c2 c : ℝ) 
  (h_right : right_angled_triangle c1 c2 c) 
  (h_iso : isosceles_right_triangle c1 c2) :
  height_relative_to_hypotenuse c = c / 2 :=
  sorry

end NUMINAMATH_GPT_maximum_height_when_isosceles_l1230_123046


namespace NUMINAMATH_GPT_rectangular_prism_sum_l1230_123088

theorem rectangular_prism_sum :
  let edges := 12
  let corners := 8
  let faces := 6
  edges + corners + faces = 26 := by
  sorry

end NUMINAMATH_GPT_rectangular_prism_sum_l1230_123088


namespace NUMINAMATH_GPT_Jake_weight_loss_l1230_123089

variable (J S: ℕ) (x : ℕ)

theorem Jake_weight_loss:
  J = 93 -> J + S = 132 -> J - x = 2 * S -> x = 15 :=
by
  intros hJ hJS hCondition
  sorry

end NUMINAMATH_GPT_Jake_weight_loss_l1230_123089


namespace NUMINAMATH_GPT_probability_different_color_and_label_sum_more_than_3_l1230_123055

-- Definitions for the conditions:
structure Coin :=
  (color : Bool) -- True for Yellow, False for Green
  (label : Nat)

def coins : List Coin := [
  Coin.mk true 1,
  Coin.mk true 2,
  Coin.mk false 1,
  Coin.mk false 2,
  Coin.mk false 3
]

def outcomes : List (Coin × Coin) :=
  [(coins[0], coins[1]), (coins[0], coins[2]), (coins[0], coins[3]), (coins[0], coins[4]),
   (coins[1], coins[2]), (coins[1], coins[3]), (coins[1], coins[4]),
   (coins[2], coins[3]), (coins[2], coins[4]), (coins[3], coins[4])]

def different_color_and_label_sum_more_than_3 (c1 c2 : Coin) : Bool :=
  c1.color ≠ c2.color ∧ (c1.label + c2.label > 3)

def valid_outcomes : List (Coin × Coin) :=
  outcomes.filter (λ p => different_color_and_label_sum_more_than_3 p.fst p.snd)

-- Proof statement:
theorem probability_different_color_and_label_sum_more_than_3 :
  (valid_outcomes.length : ℚ) / (outcomes.length : ℚ) = 3 / 10 :=
by
  sorry

end NUMINAMATH_GPT_probability_different_color_and_label_sum_more_than_3_l1230_123055


namespace NUMINAMATH_GPT_value_of_expression_l1230_123024

theorem value_of_expression (m : ℝ) (h : 1 / (m - 2) = 1) : (2 / (m - 2)) - m + 2 = 1 :=
sorry

end NUMINAMATH_GPT_value_of_expression_l1230_123024


namespace NUMINAMATH_GPT_percentage_not_even_integers_l1230_123029

variable (T : ℝ) (E : ℝ)
variables (h1 : 0.36 * T = E * 0.60) -- Condition 1 translated: 36% of T are even multiples of 3.
variables (h2 : E * 0.40)            -- Condition 2 translated: 40% of E are not multiples of 3.

theorem percentage_not_even_integers : 0.40 * T = T - E :=
by
  sorry

end NUMINAMATH_GPT_percentage_not_even_integers_l1230_123029


namespace NUMINAMATH_GPT_analogy_reasoning_conducts_electricity_l1230_123092

theorem analogy_reasoning_conducts_electricity (Gold Silver Copper Iron : Prop) (conducts : Prop)
  (h1 : Gold) (h2 : Silver) (h3 : Copper) (h4 : Iron) :
  (Gold ∧ Silver ∧ Copper ∧ Iron → conducts) → (conducts → !CompleteInductive ∧ !Inductive ∧ !Deductive ∧ Analogical) :=
by
  sorry

end NUMINAMATH_GPT_analogy_reasoning_conducts_electricity_l1230_123092


namespace NUMINAMATH_GPT_equal_x_l1230_123079

theorem equal_x (x y : ℝ) (h : x / (x + 2) = (y^2 + 3 * y - 2) / (y^2 + 3 * y + 1)) :
  x = (2 * y^2 + 6 * y - 4) / 3 :=
sorry

end NUMINAMATH_GPT_equal_x_l1230_123079


namespace NUMINAMATH_GPT_minimum_discount_l1230_123016

theorem minimum_discount (x : ℝ) (hx : x ≤ 10) : 
  let cost_price := 400 
  let selling_price := 500
  let discount_price := selling_price - (selling_price * (x / 100))
  let gross_profit := discount_price - cost_price 
  gross_profit ≥ cost_price * 0.125 :=
sorry

end NUMINAMATH_GPT_minimum_discount_l1230_123016


namespace NUMINAMATH_GPT_cycle_selling_price_l1230_123074

theorem cycle_selling_price
  (cost_price : ℝ)
  (selling_price : ℝ)
  (percentage_gain : ℝ)
  (h_cost_price : cost_price = 1000)
  (h_percentage_gain : percentage_gain = 8) :
  selling_price = cost_price + (percentage_gain / 100) * cost_price :=
by
  sorry

end NUMINAMATH_GPT_cycle_selling_price_l1230_123074


namespace NUMINAMATH_GPT_inscribed_circle_radius_l1230_123023

variable (AB AC BC : ℝ) (r : ℝ)

theorem inscribed_circle_radius 
  (h1 : AB = 9) 
  (h2 : AC = 9) 
  (h3 : BC = 8) : r = (4 * Real.sqrt 65) / 13 := 
sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l1230_123023


namespace NUMINAMATH_GPT_geometric_progression_solution_l1230_123047

theorem geometric_progression_solution (x : ℝ) :
  (2 * x + 10) ^ 2 = x * (5 * x + 10) → x = 15 + 5 * Real.sqrt 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_geometric_progression_solution_l1230_123047


namespace NUMINAMATH_GPT_range_of_m_intersection_l1230_123067

theorem range_of_m_intersection (m : ℝ) :
  (∀ k : ℝ, ∃ x y : ℝ, y - k * x - 1 = 0 ∧ (x^2 / 4) + (y^2 / m) = 1) ↔ (m ∈ Set.Ico 1 4 ∪ Set.Ioi 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_intersection_l1230_123067


namespace NUMINAMATH_GPT_range_of_quadratic_function_l1230_123085

theorem range_of_quadratic_function : 
  ∀ x : ℝ, ∃ y : ℝ, y = x^2 - 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_quadratic_function_l1230_123085


namespace NUMINAMATH_GPT_coordinates_of_B_l1230_123090

-- Define the initial conditions
def A : ℝ × ℝ := (-2, 1)
def jump_units : ℝ := 4

-- Define the function to compute the new coordinates after the jump
def new_coordinates (start : ℝ × ℝ) (jump : ℝ) : ℝ × ℝ :=
  let (x, y) := start
  (x + jump, y)

-- State the theorem to be proved
theorem coordinates_of_B
  (A : ℝ × ℝ) (jump_units : ℝ)
  (hA : A = (-2, 1))
  (h_jump : jump_units = 4) :
  new_coordinates A jump_units = (2, 1) := 
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_coordinates_of_B_l1230_123090


namespace NUMINAMATH_GPT_intersection_M_N_l1230_123098

def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

def N : Set (ℝ × ℝ) := {p | p.1 = 1}

theorem intersection_M_N : M ∩ N = { (1, 0) } := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1230_123098


namespace NUMINAMATH_GPT_local_value_of_4_in_564823_l1230_123044

def face_value (d : ℕ) : ℕ := d
def place_value_of_thousands : ℕ := 1000
def local_value (d : ℕ) (p : ℕ) : ℕ := d * p

theorem local_value_of_4_in_564823 :
  local_value (face_value 4) place_value_of_thousands = 4000 :=
by 
  sorry

end NUMINAMATH_GPT_local_value_of_4_in_564823_l1230_123044


namespace NUMINAMATH_GPT_longer_side_length_l1230_123045

-- Define the relevant entities: radius, area of the circle, and rectangle conditions.
noncomputable def radius : ℝ := 6
noncomputable def area_circle : ℝ := Real.pi * radius^2
noncomputable def area_rectangle : ℝ := 3 * area_circle
noncomputable def shorter_side : ℝ := 2 * radius

-- Prove that the length of the longer side of the rectangle is 9π cm.
theorem longer_side_length : ∃ (l : ℝ), (area_rectangle = l * shorter_side) → (l = 9 * Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_longer_side_length_l1230_123045


namespace NUMINAMATH_GPT_tan_45_degrees_l1230_123033

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end NUMINAMATH_GPT_tan_45_degrees_l1230_123033


namespace NUMINAMATH_GPT_length_of_side_d_l1230_123025

variable (a b c d : ℕ)
variable (h_ratio1 : a / c = 3 / 4)
variable (h_ratio2 : b / d = 3 / 4)
variable (h_a : a = 3)
variable (h_b : b = 6)

theorem length_of_side_d (a b c d : ℕ)
  (h_ratio1 : a / c = 3 / 4)
  (h_ratio2 : b / d = 3 / 4)
  (h_a : a = 3)
  (h_b : b = 6) : d = 8 := 
sorry

end NUMINAMATH_GPT_length_of_side_d_l1230_123025


namespace NUMINAMATH_GPT_frances_card_value_l1230_123095

theorem frances_card_value (x : ℝ) (hx : 90 < x ∧ x < 180) :
  (∃ f : ℝ → ℝ, f = sin ∨ f = cos ∨ f = tan ∧
    f x = -1 ∧
    (∃ y : ℝ, y ≠ x ∧ (sin y ≠ -1 ∧ cos y ≠ -1 ∧ tan y ≠ -1))) :=
sorry

end NUMINAMATH_GPT_frances_card_value_l1230_123095


namespace NUMINAMATH_GPT_largest_sum_of_digits_l1230_123053

theorem largest_sum_of_digits (a b c : ℕ) (y : ℕ) (h1: a < 10) (h2: b < 10) (h3: c < 10) (h4: 0 < y ∧ y ≤ 12) (h5: 1000 * y = abc) :
  a + b + c = 8 := by
  sorry

end NUMINAMATH_GPT_largest_sum_of_digits_l1230_123053


namespace NUMINAMATH_GPT_commuting_time_equation_l1230_123012

-- Definitions based on the conditions
def distance_to_cemetery : ℝ := 15
def cyclists_speed (x : ℝ) : ℝ := x
def car_speed (x : ℝ) : ℝ := 2 * x
def cyclists_start_time_earlier : ℝ := 0.5

-- The statement we need to prove
theorem commuting_time_equation (x : ℝ) (h : x > 0) :
  distance_to_cemetery / cyclists_speed x =
  (distance_to_cemetery / car_speed x) + cyclists_start_time_earlier :=
by
  sorry

end NUMINAMATH_GPT_commuting_time_equation_l1230_123012


namespace NUMINAMATH_GPT_roots_in_intervals_l1230_123008

theorem roots_in_intervals {a b c : ℝ} (h₁ : a < b) (h₂ : b < c) :
  let f (x : ℝ) := (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)
  -- statement that the roots are in the intervals (a, b) and (b, c)
  ∃ r₁ r₂, (a < r₁ ∧ r₁ < b) ∧ (b < r₂ ∧ r₂ < c) ∧ f r₁ = 0 ∧ f r₂ = 0 := 
sorry

end NUMINAMATH_GPT_roots_in_intervals_l1230_123008


namespace NUMINAMATH_GPT_tangent_line_eq_l1230_123060

theorem tangent_line_eq (x y : ℝ) (h : y = e^(-5 * x) + 2) :
  ∀ (t : ℝ), t = 0 → y = 3 → y = -5 * x + 3 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_eq_l1230_123060


namespace NUMINAMATH_GPT_problem_quadratic_radicals_l1230_123056

theorem problem_quadratic_radicals (x y : ℝ) (h : 3 * y = x + 2 * y + 2) : x - y = -2 :=
sorry

end NUMINAMATH_GPT_problem_quadratic_radicals_l1230_123056


namespace NUMINAMATH_GPT_units_digit_expression_mod_10_l1230_123007

theorem units_digit_expression_mod_10 : ((2 ^ 2023) * (5 ^ 2024) * (11 ^ 2025)) % 10 = 0 := 
by 
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_units_digit_expression_mod_10_l1230_123007


namespace NUMINAMATH_GPT_sin_cos_plus_one_l1230_123009

theorem sin_cos_plus_one (x : ℝ) (h : Real.tan x = 1 / 3) : Real.sin x * Real.cos x + 1 = 13 / 10 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_plus_one_l1230_123009


namespace NUMINAMATH_GPT_place_two_in_front_l1230_123006

-- Define the conditions: the original number has hundreds digit h, tens digit t, and units digit u.
variables (h t u : ℕ)

-- Define the function representing the placement of the digit 2 before the three-digit number.
def new_number (h t u : ℕ) : ℕ :=
  2000 + 100 * h + 10 * t + u

-- State the theorem that proves the new number formed is as stated.
theorem place_two_in_front : new_number h t u = 2000 + 100 * h + 10 * t + u :=
by sorry

end NUMINAMATH_GPT_place_two_in_front_l1230_123006


namespace NUMINAMATH_GPT_intersection_A_B_l1230_123099

open Set

def U := ℝ
def A := { x : ℝ | (2 * x + 3) / (x - 2) > 0 }
def B := { x : ℝ | abs (x - 1) < 2 }

theorem intersection_A_B : (A ∩ B) = { x : ℝ | 2 < x ∧ x < 3 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1230_123099


namespace NUMINAMATH_GPT_roger_first_bag_correct_l1230_123072

noncomputable def sandra_total_pieces : ℕ := 2 * 6
noncomputable def roger_total_pieces : ℕ := sandra_total_pieces + 2
noncomputable def roger_known_bag_pieces : ℕ := 3
noncomputable def roger_first_bag_pieces : ℕ := 11

theorem roger_first_bag_correct :
  roger_total_pieces - roger_known_bag_pieces = roger_first_bag_pieces := 
  by sorry

end NUMINAMATH_GPT_roger_first_bag_correct_l1230_123072


namespace NUMINAMATH_GPT_exam_students_count_l1230_123021

theorem exam_students_count (failed_students : ℕ) (failed_percentage : ℝ) (total_students : ℕ) 
    (h1 : failed_students = 260) 
    (h2 : failed_percentage = 0.65) 
    (h3 : (failed_percentage * total_students : ℝ) = (failed_students : ℝ)) : 
    total_students = 400 := 
by 
    sorry

end NUMINAMATH_GPT_exam_students_count_l1230_123021


namespace NUMINAMATH_GPT_sqrt_seven_l1230_123013

theorem sqrt_seven (x : ℝ) : x^2 = 7 ↔ x = Real.sqrt 7 ∨ x = -Real.sqrt 7 := by
  sorry

end NUMINAMATH_GPT_sqrt_seven_l1230_123013


namespace NUMINAMATH_GPT_measure_of_angle_S_l1230_123011

-- Define the angles in the pentagon PQRST
variables (P Q R S T : ℝ)
-- Assume the conditions from the problem
variables (h1 : P = Q)
variables (h2 : Q = R)
variables (h3 : S = T)
variables (h4 : P = S - 30)
-- Assume the sum of angles in a pentagon is 540 degrees
variables (h5 : P + Q + R + S + T = 540)

theorem measure_of_angle_S :
  S = 126 := by
  -- placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_measure_of_angle_S_l1230_123011


namespace NUMINAMATH_GPT_geometric_sequence_a3_l1230_123001

theorem geometric_sequence_a3 :
  ∀ (a : ℕ → ℝ), a 1 = 2 → a 5 = 8 → (a 3 = 4 ∨ a 3 = -4) :=
by
  intros a h₁ h₅
  sorry

end NUMINAMATH_GPT_geometric_sequence_a3_l1230_123001


namespace NUMINAMATH_GPT_divisible_by_11_and_smallest_n_implies_77_l1230_123028

theorem divisible_by_11_and_smallest_n_implies_77 (n : ℕ) (h₁ : n = 7) : ∃ m : ℕ, m = 11 * n := 
sorry

end NUMINAMATH_GPT_divisible_by_11_and_smallest_n_implies_77_l1230_123028


namespace NUMINAMATH_GPT_part1_beef_noodles_mix_sauce_purchased_l1230_123087

theorem part1_beef_noodles_mix_sauce_purchased (x y : ℕ) (h1 : x + y = 170) (h2 : 15 * x + 20 * y = 3000) :
  x = 80 ∧ y = 90 :=
sorry

end NUMINAMATH_GPT_part1_beef_noodles_mix_sauce_purchased_l1230_123087


namespace NUMINAMATH_GPT_monday_to_sunday_ratio_l1230_123017

-- Define the number of pints Alice bought on Sunday
def sunday_pints : ℕ := 4

-- Define the number of pints Alice bought on Monday as a multiple of Sunday
def monday_pints (k : ℕ) : ℕ := 4 * k

-- Define the number of pints Alice bought on Tuesday
def tuesday_pints (k : ℕ) : ℚ := (4 * k) / 3

-- Define the number of pints Alice returned on Wednesday
def wednesday_return (k : ℕ) : ℚ := (2 * k) / 3

-- Define the total number of pints Alice had on Wednesday before returning the expired ones
def total_pre_return (k : ℕ) : ℚ := 18 + (2 * k) / 3

-- Define the total number of pints purchased from Sunday to Tuesday
def total_pints (k : ℕ) : ℚ := 4 + 4 * k + (4 * k) / 3

-- The statement to be proven
theorem monday_to_sunday_ratio : ∃ k : ℕ, 
  (4 * k + (4 * k) / 3 + 4 = 18 + (2 * k) / 3) ∧
  (4 * k) / 4 = 3 :=
by 
  sorry

end NUMINAMATH_GPT_monday_to_sunday_ratio_l1230_123017


namespace NUMINAMATH_GPT_range_of_m_l1230_123084

-- Define the proposition
def P : Prop := ∀ x : ℝ, ∃ m : ℝ, 4^x - 2^(x + 1) + m = 0

-- Given that the negation of P is false
axiom neg_P_false : ¬¬P

-- Prove the range of m
theorem range_of_m : ∀ m : ℝ, (∀ x : ℝ, 4^x - 2^(x + 1) + m = 0) → m ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1230_123084


namespace NUMINAMATH_GPT_minimum_value_of_expression_l1230_123020

theorem minimum_value_of_expression (a b : ℝ) (h1 : a > 0) (h2 : a > b) (h3 : ab = 1) :
  (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l1230_123020


namespace NUMINAMATH_GPT_fatima_donates_75_sq_inches_l1230_123050

/-- Fatima starts with 100 square inches of cloth and cuts it in half twice.
    The total amount of cloth she donates should be 75 square inches. -/
theorem fatima_donates_75_sq_inches:
  ∀ (cloth_initial cloth_after_first_cut cloth_after_second_cut cloth_donated_first cloth_donated_second: ℕ),
  cloth_initial = 100 → 
  cloth_after_first_cut = cloth_initial / 2 →
  cloth_donated_first = cloth_initial / 2 →
  cloth_after_second_cut = cloth_after_first_cut / 2 →
  cloth_donated_second = cloth_after_first_cut / 2 →
  cloth_donated_first + cloth_donated_second = 75 := 
by
  intros cloth_initial cloth_after_first_cut cloth_after_second_cut cloth_donated_first cloth_donated_second
  intros h_initial h_after_first h_donated_first h_after_second h_donated_second
  sorry

end NUMINAMATH_GPT_fatima_donates_75_sq_inches_l1230_123050


namespace NUMINAMATH_GPT_xiao_zhang_complete_task_l1230_123058

open Nat

def xiaoZhangCharacters (n : ℕ) : ℕ :=
match n with
| 0 => 0
| (n+1) => 2 * (xiaoZhangCharacters n)

theorem xiao_zhang_complete_task :
  ∀ (total_chars : ℕ), (total_chars > 0) → 
  (xiaoZhangCharacters 5 = (total_chars / 3)) →
  (xiaoZhangCharacters 6 = total_chars) :=
by
  sorry

end NUMINAMATH_GPT_xiao_zhang_complete_task_l1230_123058


namespace NUMINAMATH_GPT_find_coordinates_of_C_l1230_123032

def Point := (ℝ × ℝ)

def A : Point := (-2, -1)
def B : Point := (4, 7)

/-- A custom definition to express that point C divides the segment AB in the ratio 2:1 from point B. -/
def is_point_C (C : Point) : Prop :=
  ∃ k : ℝ, k = 2 / 3 ∧
  C = (k * A.1 + (1 - k) * B.1, k * A.2 + (1 - k) * B.2)

theorem find_coordinates_of_C (C : Point) (h : is_point_C C) : 
  C = (2, 13 / 3) :=
sorry

end NUMINAMATH_GPT_find_coordinates_of_C_l1230_123032


namespace NUMINAMATH_GPT_tickets_not_went_to_concert_l1230_123048

theorem tickets_not_went_to_concert :
  let total_tickets := 900
  let before_start := total_tickets * 3 / 4
  let remaining_after_start := total_tickets - before_start
  let after_first_song := remaining_after_start * 5 / 9
  let during_middle := 80
  remaining_after_start - (after_first_song + during_middle) = 20 := 
by
  let total_tickets := 900
  let before_start := total_tickets * 3 / 4
  let remaining_after_start := total_tickets - before_start
  let after_first_song := remaining_after_start * 5 / 9
  let during_middle := 80
  show remaining_after_start - (after_first_song + during_middle) = 20
  sorry

end NUMINAMATH_GPT_tickets_not_went_to_concert_l1230_123048


namespace NUMINAMATH_GPT_probability_one_letter_from_each_l1230_123039

theorem probability_one_letter_from_each
  (total_cards : ℕ)
  (adam_cards : ℕ)
  (brian_cards : ℕ)
  (h1 : total_cards = 12)
  (h2 : adam_cards = 4)
  (h3 : brian_cards = 6)
  : (4/12 * 6/11) + (6/12 * 4/11) = 4/11 := by
  sorry

end NUMINAMATH_GPT_probability_one_letter_from_each_l1230_123039


namespace NUMINAMATH_GPT_find_n_l1230_123064

def digit_sum (n : ℕ) : ℕ :=
-- This function needs a proper definition for the digit sum, we leave it as sorry for this example.
sorry

def num_sevens (n : ℕ) : ℕ :=
7 * (10^n - 1) / 9

def product (n : ℕ) : ℕ :=
8 * num_sevens n

theorem find_n (n : ℕ) : digit_sum (product n) = 800 ↔ n = 788 :=
sorry

end NUMINAMATH_GPT_find_n_l1230_123064


namespace NUMINAMATH_GPT_evaluation_expression_l1230_123042

theorem evaluation_expression : 
  20 * (10 - 10.5 / (5.2 * 14.6 - (9.2 * 5.2 + 5.4 * 3.7 - 4.6 * 1.5))) = 192.6 := 
by
  sorry

end NUMINAMATH_GPT_evaluation_expression_l1230_123042


namespace NUMINAMATH_GPT_smallest_four_digit_divisible_by_53_l1230_123063

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007 :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_divisible_by_53_l1230_123063


namespace NUMINAMATH_GPT_ciphertext_to_plaintext_l1230_123022

theorem ciphertext_to_plaintext :
  ∃ (a b c d : ℕ), (a + 2 * b = 14) ∧ (2 * b + c = 9) ∧ (2 * c + 3 * d = 23) ∧ (4 * d = 28) ∧ a = 6 ∧ b = 4 ∧ c = 1 ∧ d = 7 :=
by 
  sorry

end NUMINAMATH_GPT_ciphertext_to_plaintext_l1230_123022


namespace NUMINAMATH_GPT_min_value_perpendicular_vectors_l1230_123014

theorem min_value_perpendicular_vectors (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (hperp : x + 3 * y = 1) : (1 / x + 1 / (3 * y)) = 4 :=
by sorry

end NUMINAMATH_GPT_min_value_perpendicular_vectors_l1230_123014


namespace NUMINAMATH_GPT_count_zeros_in_decimal_rep_l1230_123036

theorem count_zeros_in_decimal_rep (n : ℕ) (h : n = 2^3 * 5^7) : 
  ∀ (a b : ℕ), (∃ (a : ℕ) (b : ℕ), n = 10^b ∧ a < 10^b) → 
  6 = b - 1 := by
  sorry

end NUMINAMATH_GPT_count_zeros_in_decimal_rep_l1230_123036


namespace NUMINAMATH_GPT_problem_solution_l1230_123035

theorem problem_solution :
  2 ^ 2000 - 3 * 2 ^ 1999 + 2 ^ 1998 - 2 ^ 1997 + 2 ^ 1996 = -5 * 2 ^ 1996 :=
by  -- initiate the proof script
  sorry  -- means "proof is omitted"

end NUMINAMATH_GPT_problem_solution_l1230_123035


namespace NUMINAMATH_GPT_negation_proposition_l1230_123015

theorem negation_proposition (m : ℤ) :
  ¬(∃ x : ℤ, x^2 + 2*x + m < 0) ↔ ∀ x : ℤ, x^2 + 2*x + m ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_l1230_123015
