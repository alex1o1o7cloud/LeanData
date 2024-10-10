import Mathlib

namespace symmetric_line_and_distance_theorem_l3721_372140

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2 * x + y + 3 = 0
def l₂ (x y : ℝ) : Prop := x - 2 * y = 0

-- Define the symmetric line l₃
def l₃ (x y : ℝ) : Prop := 2 * x - y + 3 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-2, -1)

-- Define the line m
def m (x y : ℝ) : Prop := 3 * x + 4 * y + 10 = 0 ∨ x = -2

-- Theorem statement
theorem symmetric_line_and_distance_theorem :
  (∀ x y : ℝ, l₃ x y ↔ l₁ x (-y)) ∧
  (l₂ P.1 P.2 ∧ l₃ P.1 P.2) ∧
  (m P.1 P.2 ∧ 
   ∀ x y : ℝ, m x y → 
     (x * x + y * y = 4 ∨ 
      (3 * x + 4 * y + 10)^2 / (3 * 3 + 4 * 4) = 4)) :=
sorry

end symmetric_line_and_distance_theorem_l3721_372140


namespace thief_speed_l3721_372125

/-- Proves that the thief's speed is 8 km/hr given the problem conditions -/
theorem thief_speed (initial_distance : ℝ) (policeman_speed : ℝ) (thief_distance : ℝ)
  (h1 : initial_distance = 100) -- Initial distance in meters
  (h2 : policeman_speed = 10) -- Policeman's speed in km/hr
  (h3 : thief_distance = 400) -- Distance thief runs before being overtaken in meters
  : ∃ (thief_speed : ℝ), thief_speed = 8 := by
  sorry

end thief_speed_l3721_372125


namespace propositions_p_and_not_q_l3721_372172

theorem propositions_p_and_not_q :
  (∃ x₀ : ℝ, Real.log x₀ ≥ x₀ - 1) ∧
  ¬(∀ θ : ℝ, Real.sin θ + Real.cos θ < 1) :=
by sorry

end propositions_p_and_not_q_l3721_372172


namespace base5_multiplication_l3721_372187

/-- Converts a base 5 number to base 10 --/
def baseConvert5To10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 5 --/
def baseConvert10To5 (n : ℕ) : ℕ := sorry

/-- Multiplies two base 5 numbers --/
def multiplyBase5 (a b : ℕ) : ℕ :=
  baseConvert10To5 (baseConvert5To10 a * baseConvert5To10 b)

theorem base5_multiplication :
  multiplyBase5 132 22 = 4004 := by sorry

end base5_multiplication_l3721_372187


namespace expression_equals_one_l3721_372189

theorem expression_equals_one : 
  (2001 * 2021 + 100) * (1991 * 2031 + 400) / (2011^4) = 1 := by
  sorry

end expression_equals_one_l3721_372189


namespace inscribed_cube_volume_l3721_372170

theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 9) :
  let sphere_diameter := outer_cube_edge
  let inscribed_cube_space_diagonal := sphere_diameter
  let inscribed_cube_edge := inscribed_cube_space_diagonal / Real.sqrt 3
  let inscribed_cube_volume := inscribed_cube_edge ^ 3
  inscribed_cube_volume = 81 * Real.sqrt 3 := by
  sorry

end inscribed_cube_volume_l3721_372170


namespace expand_expression_l3721_372166

theorem expand_expression (x y : ℝ) : (x + 7) * (3 * y + 8) = 3 * x * y + 8 * x + 21 * y + 56 := by
  sorry

end expand_expression_l3721_372166


namespace powers_of_i_sum_l3721_372152

def i : ℂ := Complex.I

theorem powers_of_i_sum (h1 : i^2 = -1) (h2 : i^4 = 1) :
  i^14 + i^19 + i^24 + i^29 + i^34 + i^39 = -1 - i :=
by sorry

end powers_of_i_sum_l3721_372152


namespace swimmer_distance_proof_l3721_372188

/-- Calculates the distance swam against a current given the swimmer's speed in still water,
    the current's speed, and the time spent swimming. -/
def distance_swam_against_current (swimmer_speed : ℝ) (current_speed : ℝ) (time : ℝ) : ℝ :=
  (swimmer_speed - current_speed) * time

/-- Proves that a swimmer with a given speed in still water, swimming against a specific current
    for a certain amount of time, covers the expected distance. -/
theorem swimmer_distance_proof (swimmer_speed : ℝ) (current_speed : ℝ) (time : ℝ)
    (h1 : swimmer_speed = 3)
    (h2 : current_speed = 1.7)
    (h3 : time = 2.3076923076923075)
    : distance_swam_against_current swimmer_speed current_speed time = 3 := by
  sorry

#eval distance_swam_against_current 3 1.7 2.3076923076923075

end swimmer_distance_proof_l3721_372188


namespace rectangle_circle_area_ratio_l3721_372192

theorem rectangle_circle_area_ratio (w l r : ℝ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 2 * Real.pi * r) :
  (l * w) / (Real.pi * r^2) = 2 * Real.pi / 9 := by
  sorry

end rectangle_circle_area_ratio_l3721_372192


namespace sequence_sum_formula_l3721_372117

def sequence_sum (n : ℕ) : ℚ :=
  if n = 0 then 3 / 3
  else if n = 1 then 4 + 1/3 * (sequence_sum 0)
  else (2003 - n + 1) + 1/3 * (sequence_sum (n-1))

theorem sequence_sum_formula : 
  sequence_sum 2000 = 3004.5 - 1 / (2 * 3^1999) := by sorry

end sequence_sum_formula_l3721_372117


namespace original_sum_is_600_l3721_372110

/-- Simple interest calculation function -/
def simpleInterest (principal rate time : ℝ) : ℝ :=
  principal * (1 + rate * time)

theorem original_sum_is_600 (P R : ℝ) 
  (h1 : simpleInterest P R 2 = 720)
  (h2 : simpleInterest P R 7 = 1020) : 
  P = 600 := by
  sorry

end original_sum_is_600_l3721_372110


namespace infinitely_many_primes_3_mod_4_l3721_372131

theorem infinitely_many_primes_3_mod_4 : 
  Set.Infinite {p : ℕ | Nat.Prime p ∧ p % 4 = 3} := by sorry

end infinitely_many_primes_3_mod_4_l3721_372131


namespace conference_handshakes_l3721_372195

/-- Represents a conference with two groups of people -/
structure Conference :=
  (total : ℕ)
  (group1 : ℕ)
  (group2 : ℕ)
  (h_total : total = group1 + group2)

/-- Calculates the number of handshakes in the conference -/
def handshakes (conf : Conference) : ℕ :=
  conf.group2 * (conf.group1 + conf.group2 - 1)

theorem conference_handshakes :
  ∃ (conf : Conference),
    conf.total = 40 ∧
    conf.group1 = 25 ∧
    conf.group2 = 15 ∧
    handshakes conf = 480 := by
  sorry

end conference_handshakes_l3721_372195


namespace determinant_problem_l3721_372154

theorem determinant_problem (p q r s : ℝ) :
  Matrix.det !![p, q; r, s] = -3 →
  Matrix.det !![p, 5*p + 4*q; r, 5*r + 4*s] = -12 := by
  sorry

end determinant_problem_l3721_372154


namespace wheel_radius_increase_wheel_radius_increase_approx_011_l3721_372145

/-- Calculates the increase in wheel radius given the original and new measurements -/
theorem wheel_radius_increase (original_radius : ℝ) (original_distance : ℝ) 
  (new_odometer_distance : ℝ) (new_actual_distance : ℝ) : ℝ :=
  let original_circumference := 2 * Real.pi * original_radius
  let original_rotations := original_distance * 63360 / original_circumference
  let new_radius := new_actual_distance * 63360 / (2 * Real.pi * original_rotations)
  new_radius - original_radius

/-- The increase in wheel radius is approximately 0.11 inches -/
theorem wheel_radius_increase_approx_011 :
  abs (wheel_radius_increase 12 300 310 315 - 0.11) < 0.005 := by
  sorry

end wheel_radius_increase_wheel_radius_increase_approx_011_l3721_372145


namespace tea_mixture_theorem_l3721_372151

/-- Calculates the price of a tea mixture per kg -/
def tea_mixture_price (price1 price2 price3 : ℚ) (ratio1 ratio2 ratio3 : ℚ) : ℚ :=
  let total_cost := price1 * ratio1 + price2 * ratio2 + price3 * ratio3
  let total_quantity := ratio1 + ratio2 + ratio3
  total_cost / total_quantity

/-- Theorem stating the price of the specific tea mixture -/
theorem tea_mixture_theorem : 
  tea_mixture_price 126 135 177.5 1 1 2 = 154 := by
  sorry

#eval tea_mixture_price 126 135 177.5 1 1 2

end tea_mixture_theorem_l3721_372151


namespace inequality_solution_set_l3721_372171

theorem inequality_solution_set (x : ℝ) :
  ((x + 1) / (3 - x) < 0) ↔ (x < -1 ∨ x > 3) := by
  sorry

end inequality_solution_set_l3721_372171


namespace root_transformation_l3721_372133

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 3*r₁^2 + 8 = 0) ∧ 
  (r₂^3 - 3*r₂^2 + 8 = 0) ∧ 
  (r₃^3 - 3*r₃^2 + 8 = 0) → 
  ((3*r₁)^3 - 9*(3*r₁)^2 + 216 = 0) ∧
  ((3*r₂)^3 - 9*(3*r₂)^2 + 216 = 0) ∧
  ((3*r₃)^3 - 9*(3*r₃)^2 + 216 = 0) := by
sorry

end root_transformation_l3721_372133


namespace valid_fractions_characterization_l3721_372196

def is_valid_fraction (a b : ℕ) : Prop :=
  10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧
  (a % 10 = b / 10) ∧
  (a : ℚ) / b = (a / 10 : ℚ) / (b % 10)

def valid_fractions : Set (ℕ × ℕ) :=
  {(a, b) | is_valid_fraction a b}

theorem valid_fractions_characterization :
  valid_fractions = {(19, 95), (49, 98), (11, 11), (22, 22), (33, 33),
                     (44, 44), (55, 55), (66, 66), (77, 77), (88, 88),
                     (99, 99), (16, 64), (26, 65)} :=
by sorry

end valid_fractions_characterization_l3721_372196


namespace f_minimum_at_neg_nine_halves_l3721_372112

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 9*x + 7

-- State the theorem
theorem f_minimum_at_neg_nine_halves :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = -9/2 :=
sorry

end f_minimum_at_neg_nine_halves_l3721_372112


namespace fourth_day_income_l3721_372159

def cab_driver_income (day1 day2 day3 day4 day5 : ℝ) : Prop :=
  day1 = 200 ∧ day2 = 150 ∧ day3 = 750 ∧ day5 = 500 ∧
  (day1 + day2 + day3 + day4 + day5) / 5 = 400

theorem fourth_day_income (day1 day2 day3 day4 day5 : ℝ) :
  cab_driver_income day1 day2 day3 day4 day5 → day4 = 400 := by
  sorry

end fourth_day_income_l3721_372159


namespace rogers_cookie_price_l3721_372148

/-- Represents a cookie shape -/
inductive CookieShape
| Trapezoid
| Rectangle

/-- Represents a baker's cookie production -/
structure Baker where
  name : String
  shape : CookieShape
  numCookies : ℕ
  pricePerCookie : ℕ

/-- Calculates the total earnings for a baker -/
def totalEarnings (baker : Baker) : ℕ :=
  baker.numCookies * baker.pricePerCookie

/-- Theorem: Roger's cookie price for equal earnings -/
theorem rogers_cookie_price 
  (art roger : Baker)
  (h1 : art.shape = CookieShape.Trapezoid)
  (h2 : roger.shape = CookieShape.Rectangle)
  (h3 : art.numCookies = 12)
  (h4 : art.pricePerCookie = 60)
  (h5 : totalEarnings art = totalEarnings roger) :
  roger.pricePerCookie = 40 :=
sorry

end rogers_cookie_price_l3721_372148


namespace range_of_a_l3721_372122

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → -(5-2*a)^x > -(5-2*a)^y

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ≤ -2 :=
sorry

end range_of_a_l3721_372122


namespace sticker_distribution_jeremy_sticker_problem_l3721_372199

/-- The number of ways to distribute n identical objects among k distinct groups -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute n identical objects among k distinct groups,
    with each group receiving at least one object -/
def distributeAtLeastOne (n k : ℕ) : ℕ := distribute (n - k) k

theorem sticker_distribution (n k : ℕ) (hn : n ≥ k) (hk : k > 0) :
  distributeAtLeastOne n k = distribute (n - k) k :=
by sorry

theorem jeremy_sticker_problem :
  distributeAtLeastOne 10 3 = 36 :=
by sorry

end sticker_distribution_jeremy_sticker_problem_l3721_372199


namespace periodic_function_value_l3721_372193

/-- Given a function f(x) = a*sin(πx + α) + b*cos(πx + β), if f(3) = 3, then f(2016) = -3 -/
theorem periodic_function_value (a b α β : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β)
  f 3 = 3 → f 2016 = -3 := by
  sorry

end periodic_function_value_l3721_372193


namespace abs_equality_implies_geq_one_l3721_372102

theorem abs_equality_implies_geq_one (m : ℝ) : |m - 1| = m - 1 → m ≥ 1 := by
  sorry

end abs_equality_implies_geq_one_l3721_372102


namespace rectangle_width_is_fifteen_l3721_372100

/-- Represents a rectangle with length and width in centimeters. -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem: For a rectangle where the width is 3 cm longer than the length
    and the perimeter is 54 cm, the width is 15 cm. -/
theorem rectangle_width_is_fifteen (r : Rectangle)
    (h1 : r.width = r.length + 3)
    (h2 : perimeter r = 54) :
    r.width = 15 := by
  sorry

end rectangle_width_is_fifteen_l3721_372100


namespace milk_bottles_count_l3721_372139

theorem milk_bottles_count (bread : ℕ) (total : ℕ) (h1 : bread = 37) (h2 : total = 52) :
  total - bread = 15 := by
  sorry

end milk_bottles_count_l3721_372139


namespace union_M_N_l3721_372162

-- Define the universe set U
def U : Set ℝ := {x | -3 ≤ x ∧ x < 2}

-- Define set M
def M : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define the complement of N with respect to U
def complement_N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define set N
def N : Set ℝ := U \ complement_N

-- Theorem statement
theorem union_M_N : M ∪ N = {x : ℝ | -3 ≤ x ∧ x < 1} := by
  sorry

end union_M_N_l3721_372162


namespace louies_previous_goals_correct_l3721_372128

/-- Calculates the number of goals Louie scored in previous matches before the last match -/
def louies_previous_goals (
  louies_last_match_goals : ℕ
  ) (
  brother_seasons : ℕ
  ) (
  games_per_season : ℕ
  ) (
  total_goals : ℕ
  ) : ℕ := by
  sorry

theorem louies_previous_goals_correct :
  louies_previous_goals 4 3 50 1244 = 40 := by
  sorry

end louies_previous_goals_correct_l3721_372128


namespace davids_chemistry_marks_l3721_372198

theorem davids_chemistry_marks
  (english : ℕ) (mathematics : ℕ) (physics : ℕ) (biology : ℕ) (average : ℕ)
  (h_english : english = 61)
  (h_mathematics : mathematics = 65)
  (h_physics : physics = 82)
  (h_biology : biology = 85)
  (h_average : average = 72)
  : ∃ chemistry : ℕ,
    chemistry = 67 ∧
    (english + mathematics + physics + chemistry + biology) / 5 = average :=
by sorry

end davids_chemistry_marks_l3721_372198


namespace sequence_periodicity_l3721_372121

def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) = |a (n + 1)| - a n

theorem sequence_periodicity (a : ℕ → ℝ) (h : sequence_property a) :
  ∃ m₀ : ℕ, ∀ m ≥ m₀, a (m + 9) = a m :=
sorry

end sequence_periodicity_l3721_372121


namespace sum_of_angles_in_three_triangles_l3721_372184

theorem sum_of_angles_in_three_triangles :
  ∀ (angle1 angle2 angle3 angle4 angle5 angle6 angle7 angle8 angle9 : ℝ),
    angle1 > 0 → angle2 > 0 → angle3 > 0 → angle4 > 0 → angle5 > 0 →
    angle6 > 0 → angle7 > 0 → angle8 > 0 → angle9 > 0 →
    angle1 + angle2 + angle3 = 180 →
    angle4 + angle5 + angle6 = 180 →
    angle7 + angle8 + angle9 = 180 →
    angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + angle7 + angle8 + angle9 = 540 :=
by sorry

end sum_of_angles_in_three_triangles_l3721_372184


namespace paths_equal_choose_l3721_372163

/-- The number of paths on a 3x3 grid from top-left to bottom-right -/
def num_paths : ℕ := sorry

/-- The number of ways to choose 3 items from a set of 6 items -/
def choose_3_from_6 : ℕ := Nat.choose 6 3

/-- Theorem stating that the number of paths is equal to choosing 3 from 6 -/
theorem paths_equal_choose :
  num_paths = choose_3_from_6 := by sorry

end paths_equal_choose_l3721_372163


namespace impossibility_of_option_d_l3721_372178

-- Define the basic rhombus shape
structure Rhombus :=
  (color : Bool)  -- True for white, False for gray

-- Define the operation of rotation
def rotate (r : Rhombus) : Rhombus := r

-- Define a larger shape as a collection of rhombuses
def LargerShape := List Rhombus

-- Define the four options
def option_a : LargerShape := sorry
def option_b : LargerShape := sorry
def option_c : LargerShape := sorry
def option_d : LargerShape := sorry

-- Define a function to check if a larger shape can be constructed
def can_construct (shape : LargerShape) : Prop := sorry

-- State the theorem
theorem impossibility_of_option_d :
  can_construct option_a ∧
  can_construct option_b ∧
  can_construct option_c ∧
  ¬ can_construct option_d :=
sorry

end impossibility_of_option_d_l3721_372178


namespace parabola_solution_l3721_372143

/-- Parabola intersecting x-axis at two points -/
structure Parabola where
  a : ℝ
  intersectionA : ℝ × ℝ
  intersectionB : ℝ × ℝ

/-- The parabola y = a(x+1)^2 + 2 intersects the x-axis at A(-3, 0) and B -/
def parabola_problem (p : Parabola) : Prop :=
  p.intersectionA = (-3, 0) ∧
  p.intersectionA.2 = p.a * (p.intersectionA.1 + 1)^2 + 2 ∧
  p.intersectionB.2 = p.a * (p.intersectionB.1 + 1)^2 + 2 ∧
  p.intersectionA.2 = 0 ∧
  p.intersectionB.2 = 0

theorem parabola_solution (p : Parabola) (h : parabola_problem p) :
  p.a = -1/2 ∧ p.intersectionB = (1, 0) := by
  sorry

end parabola_solution_l3721_372143


namespace valid_systematic_sample_l3721_372177

/-- Represents a systematic sample -/
structure SystematicSample where
  population : ℕ
  sampleSize : ℕ
  startPoint : ℕ
  interval : ℕ

/-- Checks if a given list is a valid systematic sample -/
def isValidSystematicSample (sample : List ℕ) (s : SystematicSample) : Prop :=
  sample.length = s.sampleSize ∧
  sample.all (·≤ s.population) ∧
  sample.all (·> 0) ∧
  ∀ i, i < sample.length - 1 → sample[i + 1]! - sample[i]! = s.interval

theorem valid_systematic_sample :
  let sample := [3, 13, 23, 33, 43]
  let s : SystematicSample := {
    population := 50,
    sampleSize := 5,
    startPoint := 3,
    interval := 10
  }
  isValidSystematicSample sample s := by
  sorry

end valid_systematic_sample_l3721_372177


namespace largest_A_k_l3721_372160

-- Define A_k
def A (k : ℕ) : ℝ := (Nat.choose 1000 k) * (0.2 ^ k)

-- State the theorem
theorem largest_A_k : ∃ (k : ℕ), k = 166 ∧ ∀ (j : ℕ), j ≠ k → j ≤ 1000 → A k ≥ A j := by
  sorry

end largest_A_k_l3721_372160


namespace horner_method_proof_l3721_372141

def horner_polynomial (x : ℝ) : ℝ := 
  ((((2 * x - 5) * x - 4) * x + 3) * x - 6) * x + 7

theorem horner_method_proof : horner_polynomial 5 = 2677 := by
  sorry

end horner_method_proof_l3721_372141


namespace intersection_of_A_and_B_l3721_372130

def A : Set ℝ := {x | x^2 - 3*x - 4 > 0}
def B : Set ℝ := {x | -2 < x ∧ x < 5}

theorem intersection_of_A_and_B :
  A ∩ B = {x | (-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5)} := by sorry

end intersection_of_A_and_B_l3721_372130


namespace correct_taobao_shopping_order_l3721_372113

-- Define the type for shopping steps
inductive ShoppingStep
| select_products
| buy_and_pay
| transfer_payment
| receive_and_confirm
| ship_goods

-- Define the shopping process
def shopping_process : List ShoppingStep :=
  [ShoppingStep.select_products, ShoppingStep.buy_and_pay, ShoppingStep.ship_goods, 
   ShoppingStep.receive_and_confirm, ShoppingStep.transfer_payment]

-- Define a function to check if the order is correct
def is_correct_order (order : List ShoppingStep) : Prop :=
  order = shopping_process

-- Theorem stating the correct order
theorem correct_taobao_shopping_order :
  is_correct_order [ShoppingStep.select_products, ShoppingStep.buy_and_pay, 
                    ShoppingStep.ship_goods, ShoppingStep.receive_and_confirm, 
                    ShoppingStep.transfer_payment] :=
by
  sorry

#check correct_taobao_shopping_order

end correct_taobao_shopping_order_l3721_372113


namespace magazine_sale_gain_l3721_372127

/-- Calculates the total gain from selling magazines -/
def total_gain (cost_price selling_price : ℝ) (num_magazines : ℕ) : ℝ :=
  (selling_price - cost_price) * num_magazines

/-- Proves that the total gain from selling 10 magazines at $3.50 each, 
    bought at $3 each, is $5 -/
theorem magazine_sale_gain : 
  total_gain 3 3.5 10 = 5 := by
  sorry

end magazine_sale_gain_l3721_372127


namespace evaluate_expression_l3721_372167

theorem evaluate_expression : 
  Real.sqrt (9 / 4) - Real.sqrt (4 / 9) + 1 / 3 = 7 / 6 := by
  sorry

end evaluate_expression_l3721_372167


namespace rectangle_circle_ratio_l3721_372104

theorem rectangle_circle_ratio (r : ℝ) (h : r > 0) : 
  ∃ (x y : ℝ), 
    x > 0 ∧ y > 0 ∧ 
    (x + 2*y)^2 = 16 * π * r^2 ∧ 
    y = r * Real.sqrt π ∧
    x / y = 2 :=
by sorry

end rectangle_circle_ratio_l3721_372104


namespace triangle_side_length_l3721_372111

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a = 3 →
  C = 2 * π / 3 →
  S = (15 * Real.sqrt 3) / 4 →
  S = (1 / 2) * a * b * Real.sin C →
  c^2 = a^2 + b^2 - 2 * a * b * Real.cos C →
  c = 7 := by
  sorry

end triangle_side_length_l3721_372111


namespace four_roots_iff_t_in_range_l3721_372155

-- Define the function f(x) = |xe^x|
noncomputable def f (x : ℝ) : ℝ := |x * Real.exp x|

-- Define the equation f^2(x) + tf(x) + 2 = 0
def has_four_distinct_roots (t : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    (f x₁)^2 + t * f x₁ + 2 = 0 ∧
    (f x₂)^2 + t * f x₂ + 2 = 0 ∧
    (f x₃)^2 + t * f x₃ + 2 = 0 ∧
    (f x₄)^2 + t * f x₄ + 2 = 0

-- The theorem to be proved
theorem four_roots_iff_t_in_range :
  ∀ t : ℝ, has_four_distinct_roots t ↔ t < -(2 * Real.exp 2 + 1) / Real.exp 1 :=
sorry

end four_roots_iff_t_in_range_l3721_372155


namespace curve_equivalence_l3721_372161

-- Define the set of points satisfying the original equation
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + p.2 - 1) * Real.sqrt (p.1^2 + p.2^2 - 4) = 0}

-- Define the set of points on the line outside or on the circle
def L : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 1 ∧ p.1^2 + p.2^2 ≥ 4}

-- Define the set of points on the circle
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

-- Theorem stating the equivalence of the sets
theorem curve_equivalence : S = L ∪ C := by sorry

end curve_equivalence_l3721_372161


namespace square_area_error_l3721_372174

/-- Given a square with a side measurement error of 38% in excess,
    the percentage of error in the calculated area is 90.44%. -/
theorem square_area_error (S : ℝ) (S_pos : S > 0) :
  let measured_side := S * (1 + 0.38)
  let actual_area := S^2
  let calculated_area := measured_side^2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.9044 := by sorry

end square_area_error_l3721_372174


namespace hyperbola_asymptotes_l3721_372153

/-- Given a hyperbola with equation x²/4 - y²/9 = -1, its asymptotes are y = ±(3/2)x -/
theorem hyperbola_asymptotes (x y : ℝ) :
  x^2 / 4 - y^2 / 9 = -1 →
  ∃ (k : ℝ), k = 3/2 ∧ (y = k*x ∨ y = -k*x) :=
sorry

end hyperbola_asymptotes_l3721_372153


namespace man_speed_man_speed_proof_l3721_372173

/-- Calculates the speed of a man moving opposite to a bullet train -/
theorem man_speed (train_length : ℝ) (train_speed : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed := train_length / passing_time * 3.6
  relative_speed - train_speed

/-- Proves that the man's speed is 4 kmph given the specific conditions -/
theorem man_speed_proof :
  man_speed 120 50 8 = 4 := by
  sorry

end man_speed_man_speed_proof_l3721_372173


namespace sqrt_two_sin_twenty_equals_cos_minus_sin_theta_l3721_372142

theorem sqrt_two_sin_twenty_equals_cos_minus_sin_theta (θ : Real) :
  θ > 0 ∧ θ < Real.pi / 2 →
  Real.sqrt 2 * Real.sin (20 * Real.pi / 180) = Real.cos θ - Real.sin θ →
  θ = 25 * Real.pi / 180 := by
sorry

end sqrt_two_sin_twenty_equals_cos_minus_sin_theta_l3721_372142


namespace max_abs_sum_of_quadratic_coeffs_l3721_372132

/-- Given a quadratic polynomial ax^2 + bx + c where |ax^2 + bx + c| ≤ 1 for all x in [-1,1],
    the maximum value of |a| + |b| + |c| is 3. -/
theorem max_abs_sum_of_quadratic_coeffs (a b c : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |a * x^2 + b * x + c| ≤ 1) →
  |a| + |b| + |c| ≤ 3 ∧ ∃ a' b' c' : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |a' * x^2 + b' * x + c'| ≤ 1) ∧ |a'| + |b'| + |c'| = 3 :=
by sorry

end max_abs_sum_of_quadratic_coeffs_l3721_372132


namespace even_function_m_value_l3721_372150

/-- A function f is even if f(x) = f(-x) for all x in its domain --/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The given function f(x) = x^2 + (m+2)x + 3 --/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + (m+2)*x + 3

theorem even_function_m_value :
  ∀ m : ℝ, IsEven (f m) → m = -2 := by
sorry

end even_function_m_value_l3721_372150


namespace repeating_decimal_as_fraction_l3721_372144

def repeating_decimal : ℚ := 7 + 2/10 + 34/99/100

theorem repeating_decimal_as_fraction : 
  repeating_decimal = 36357 / 4950 := by sorry

end repeating_decimal_as_fraction_l3721_372144


namespace inradius_plus_circumradius_le_height_l3721_372116

/-- An acute-angled triangle is a triangle where all angles are less than 90 degrees. -/
structure AcuteTriangle where
  /-- The greatest height of the triangle -/
  height : ℝ
  /-- The inradius of the triangle -/
  inradius : ℝ
  /-- The circumradius of the triangle -/
  circumradius : ℝ
  /-- All angles are less than 90 degrees -/
  acute : height > 0 ∧ inradius > 0 ∧ circumradius > 0

/-- For any acute-angled triangle, the sum of its inradius and circumradius
    is less than or equal to its greatest height. -/
theorem inradius_plus_circumradius_le_height (t : AcuteTriangle) :
  t.inradius + t.circumradius ≤ t.height := by
  sorry

end inradius_plus_circumradius_le_height_l3721_372116


namespace simplify_radical_product_l3721_372157

theorem simplify_radical_product (q : ℝ) (h : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (14 * q) = 14 * q * Real.sqrt (39 * q) :=
by sorry

end simplify_radical_product_l3721_372157


namespace spent_sixty_four_l3721_372191

/-- The total amount spent by Victor and his friend on trick decks -/
def total_spent (deck_price : ℕ) (victor_decks : ℕ) (friend_decks : ℕ) : ℕ :=
  deck_price * (victor_decks + friend_decks)

/-- Theorem: Victor and his friend spent $64 on trick decks -/
theorem spent_sixty_four :
  total_spent 8 6 2 = 64 := by
  sorry

end spent_sixty_four_l3721_372191


namespace boys_percentage_in_class_l3721_372101

theorem boys_percentage_in_class (total_students : ℕ) (boys_ratio girls_ratio : ℕ) 
  (h1 : total_students = 42)
  (h2 : boys_ratio = 3)
  (h3 : girls_ratio = 4) :
  (boys_ratio * total_students : ℚ) / ((boys_ratio + girls_ratio) * total_students) * 100 = 42857 / 1000 := by
  sorry

end boys_percentage_in_class_l3721_372101


namespace complex_equation_solution_l3721_372175

theorem complex_equation_solution (a : ℝ) :
  (2 + a * Complex.I) / (1 + Complex.I) = -2 * Complex.I → a = -2 := by
  sorry

end complex_equation_solution_l3721_372175


namespace consecutive_integers_cube_sum_l3721_372176

theorem consecutive_integers_cube_sum : 
  ∀ x : ℕ, x > 0 → 
  (x - 1) * x * (x + 1) = 12 * (3 * x) →
  (x - 1)^3 + x^3 + (x + 1)^3 = 3 * (37 : ℝ).sqrt^3 + 6 * (37 : ℝ).sqrt :=
by sorry

end consecutive_integers_cube_sum_l3721_372176


namespace b_range_l3721_372135

def P : Set ℝ := {x | x^2 - 5*x + 4 ≤ 0}

def Q (b : ℝ) : Set ℝ := {x | x^2 - (b+2)*x + 2*b ≤ 0}

theorem b_range (b : ℝ) : P ⊇ Q b ↔ b ∈ Set.Icc 1 4 := by
  sorry

end b_range_l3721_372135


namespace max_value_sine_cosine_l3721_372109

/-- Given a function f(x) = a*sin(x) + 3*cos(x) where its maximum value is 5, 
    prove that a = ±4 -/
theorem max_value_sine_cosine (a : ℝ) :
  (∀ x, a * Real.sin x + 3 * Real.cos x ≤ 5) ∧ 
  (∃ x, a * Real.sin x + 3 * Real.cos x = 5) →
  a = 4 ∨ a = -4 := by
sorry

end max_value_sine_cosine_l3721_372109


namespace complete_square_with_integer_l3721_372124

theorem complete_square_with_integer (y : ℝ) :
  ∃ (k : ℤ) (b : ℝ), y^2 + 12*y + 44 = (y + b)^2 + k := by
  sorry

end complete_square_with_integer_l3721_372124


namespace cubic_sum_theorem_l3721_372147

theorem cubic_sum_theorem (a b c d : ℕ) 
  (h : (a + b + c + d) * (a^2 + b^2 + c^2 + d^2)^2 = 2023) : 
  a^3 + b^3 + c^3 + d^3 = 43 := by
sorry

end cubic_sum_theorem_l3721_372147


namespace parabolic_triangle_area_l3721_372137

theorem parabolic_triangle_area (n : ℕ) : 
  ∃ (a b : ℤ) (m : ℕ), 
    Odd m ∧ 
    (a * (b^2 - a^2) : ℤ) = (2^n * m)^2 := by
  sorry

end parabolic_triangle_area_l3721_372137


namespace solution_equation1_solution_equation2_l3721_372182

-- Define the equations
def equation1 (x : ℝ) : Prop := 2 * (x - 1) = 2 - 5 * (x + 2)
def equation2 (x : ℝ) : Prop := (5 * x + 1) / 2 - (6 * x + 2) / 4 = 1

-- Theorem for the first equation
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = -6/7 := by sorry

-- Theorem for the second equation
theorem solution_equation2 : ∃ x : ℝ, equation2 x ∧ x = 1 := by sorry

end solution_equation1_solution_equation2_l3721_372182


namespace circle_area_ratio_l3721_372165

/-- A square with side length 2 -/
structure Square :=
  (side_length : ℝ)
  (is_two : side_length = 2)

/-- A circle outside the square -/
structure Circle :=
  (radius : ℝ)
  (center : ℝ × ℝ)

/-- The configuration of the two circles and the square -/
structure Configuration :=
  (square : Square)
  (circle1 : Circle)
  (circle2 : Circle)
  (tangent_to_PQ : circle1.center.2 = square.side_length / 2)
  (tangent_to_RS : circle2.center.2 = -square.side_length / 2)
  (tangent_to_QR_extension : circle1.center.1 + circle1.radius = circle2.center.1 - circle2.radius)

/-- The theorem stating the ratio of the areas -/
theorem circle_area_ratio (config : Configuration) : 
  (π * config.circle2.radius^2) / (π * config.circle1.radius^2) = 4 := by
  sorry

end circle_area_ratio_l3721_372165


namespace same_city_probability_l3721_372105

/-- The probability that two specific students are assigned to the same city
    given the total number of students and the number of spots in each city. -/
theorem same_city_probability
  (total_students : ℕ)
  (spots_moscow : ℕ)
  (spots_tula : ℕ)
  (spots_voronezh : ℕ)
  (h1 : total_students = 30)
  (h2 : spots_moscow = 15)
  (h3 : spots_tula = 8)
  (h4 : spots_voronezh = 7)
  (h5 : total_students = spots_moscow + spots_tula + spots_voronezh) :
  (spots_moscow.choose 2 + spots_tula.choose 2 + spots_voronezh.choose 2) / total_students.choose 2 = 154 / 435 :=
by sorry

end same_city_probability_l3721_372105


namespace rotation_to_second_quadrant_l3721_372107

/-- Given a complex number z = (-1+3i)/i, prove that rotating the point A 
    corresponding to z counterclockwise by 2π/3 radians results in a point B 
    in the second quadrant. -/
theorem rotation_to_second_quadrant (z : ℂ) : 
  z = (-1 + 3*Complex.I) / Complex.I → 
  let A := z
  let θ := 2 * Real.pi / 3
  let B := Complex.exp (Complex.I * θ) * A
  (B.re < 0 ∧ B.im > 0) := by sorry

end rotation_to_second_quadrant_l3721_372107


namespace associated_equation_k_range_l3721_372197

/-- Definition of an associated equation -/
def is_associated_equation (eq_sol : ℝ) (ineq_set : Set ℝ) : Prop :=
  eq_sol ∈ ineq_set

/-- The system of inequalities -/
def inequality_system (x : ℝ) : Prop :=
  (x - 3) / 2 ≥ x ∧ (2 * x + 5) / 2 > x / 2

/-- The solution set of the system of inequalities -/
def solution_set : Set ℝ :=
  {x | inequality_system x}

/-- The equation 2x - k = 6 -/
def equation (k : ℝ) (x : ℝ) : Prop :=
  2 * x - k = 6

theorem associated_equation_k_range :
  ∀ k : ℝ, (∃ x : ℝ, equation k x ∧ is_associated_equation x solution_set) →
    -16 < k ∧ k ≤ -12 :=
by sorry

end associated_equation_k_range_l3721_372197


namespace stream_speed_l3721_372136

/-- Given that a canoe rows upstream at 9 km/hr and downstream at 12 km/hr,
    the speed of the stream is 1.5 km/hr. -/
theorem stream_speed (upstream_speed downstream_speed : ℝ)
  (h_upstream : upstream_speed = 9)
  (h_downstream : downstream_speed = 12) :
  ∃ (canoe_speed stream_speed : ℝ),
    canoe_speed - stream_speed = upstream_speed ∧
    canoe_speed + stream_speed = downstream_speed ∧
    stream_speed = 1.5 := by
  sorry

end stream_speed_l3721_372136


namespace inequality_product_sum_l3721_372156

theorem inequality_product_sum (a₁ a₂ b₁ b₂ : ℝ) 
  (h1 : a₁ < a₂) (h2 : b₁ < b₂) : 
  a₁ * b₁ + a₂ * b₂ > a₁ * b₂ + a₂ * b₁ := by
  sorry

end inequality_product_sum_l3721_372156


namespace binary_equals_octal_l3721_372146

-- Define the binary number
def binary_num : List Bool := [true, true, false, true, false, true]

-- Define the octal number
def octal_num : Nat := 65

-- Function to convert binary to decimal
def binary_to_decimal (bin : List Bool) : Nat :=
  bin.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

-- Function to convert decimal to octal
def decimal_to_octal (n : Nat) : Nat :=
  if n < 8 then n
  else 10 * (decimal_to_octal (n / 8)) + (n % 8)

-- Theorem stating that the binary number is equal to the octal number when converted
theorem binary_equals_octal :
  decimal_to_octal (binary_to_decimal binary_num) = octal_num := by
  sorry


end binary_equals_octal_l3721_372146


namespace product_of_fractions_equals_21_8_l3721_372185

theorem product_of_fractions_equals_21_8 : 
  let f (n : ℕ) := (n^3 - 1) * (n - 2) / (n^3 + 1)
  f 3 * f 5 * f 7 * f 9 * f 11 = 21 / 8 := by
  sorry

end product_of_fractions_equals_21_8_l3721_372185


namespace bamboo_problem_l3721_372169

def arithmetic_sequence (a : ℚ → ℚ) (n : ℕ) : Prop :=
  ∃ (a₁ d : ℚ), ∀ k, a k = a₁ + (k - 1) * d

theorem bamboo_problem (a : ℚ → ℚ) :
  arithmetic_sequence a 9 →
  (a 1 + a 2 + a 3 + a 4 = 3) →
  (a 7 + a 8 + a 9 = 4) →
  a 5 = 67 / 66 := by
sorry

end bamboo_problem_l3721_372169


namespace dodo_is_sane_l3721_372103

-- Define the characters
inductive Character : Type
| Dodo : Character
| Lori : Character
| Eagle : Character

-- Define the "thinks" relation
def thinks (x y : Character) (p : Prop) : Prop := sorry

-- Define sanity
def is_sane (x : Character) : Prop := sorry

-- State the theorem
theorem dodo_is_sane :
  (thinks Dodo Lori (¬ is_sane Eagle)) →
  (thinks Lori Dodo (¬ is_sane Dodo)) →
  (thinks Eagle Dodo (is_sane Dodo)) →
  is_sane Dodo := by sorry

end dodo_is_sane_l3721_372103


namespace complex_number_quadrant_l3721_372138

theorem complex_number_quadrant : ∃ (z : ℂ), 
  z / (1 - z) = Complex.I * 2 ∧ 
  Complex.re z > 0 ∧ Complex.im z > 0 :=
sorry

end complex_number_quadrant_l3721_372138


namespace circle_C_radius_range_l3721_372158

-- Define the triangle vertices
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)
def C : ℝ × ℝ := (3, 2)

-- Define the circumcircle H
def H : ℝ × ℝ := (0, 3)

-- Define the line BH
def lineBH (x y : ℝ) : Prop := 3 * x + y - 3 = 0

-- Define a point P on line segment BH
def P (m : ℝ) : ℝ × ℝ := (m, 3 - 3 * m)

-- Define the circle with center C
def circleC (r : ℝ) (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = r^2

-- Define the theorem
theorem circle_C_radius_range :
  ∀ (r : ℝ), 
  (∀ (m : ℝ), 0 ≤ m ∧ m ≤ 1 →
    ∃ (x y : ℝ), 
      circleC r x y ∧
      circleC r ((x + m) / 2) ((y + (3 - 3 * m)) / 2) ∧
      x ≠ m ∧ y ≠ (3 - 3 * m)) →
  (∀ (m : ℝ), 0 ≤ m ∧ m ≤ 1 →
    (m - 3)^2 + (1 - 3 * m)^2 > r^2) →
  Real.sqrt 10 / 3 ≤ r ∧ r < 4 * Real.sqrt 10 / 5 :=
sorry


end circle_C_radius_range_l3721_372158


namespace sine_cosine_inequality_l3721_372119

theorem sine_cosine_inequality (a b c : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ c < 0 := by
  sorry

end sine_cosine_inequality_l3721_372119


namespace total_short_trees_after_planting_l3721_372179

/-- Represents the types of trees in the park -/
inductive TreeType
  | Oak
  | Maple
  | Pine

/-- Represents the current state of trees in the park -/
structure ParkTrees where
  shortOak : ℕ
  shortMaple : ℕ
  shortPine : ℕ
  tallOak : ℕ
  tallMaple : ℕ
  tallPine : ℕ

/-- Calculates the new number of short trees after planting -/
def newShortTrees (park : ParkTrees) : ℕ :=
  let newOak := park.shortOak + 57
  let newMaple := park.shortMaple + (park.shortMaple * 3 / 10)  -- 30% increase
  let newPine := park.shortPine + (park.shortPine / 3)  -- 1/3 increase
  newOak + newMaple + newPine

/-- Theorem stating that the total number of short trees after planting is 153 -/
theorem total_short_trees_after_planting (park : ParkTrees) 
  (h1 : park.shortOak = 41)
  (h2 : park.shortMaple = 18)
  (h3 : park.shortPine = 24)
  (h4 : park.tallOak = 44)
  (h5 : park.tallMaple = 37)
  (h6 : park.tallPine = 17) :
  newShortTrees park = 153 := by
  sorry

end total_short_trees_after_planting_l3721_372179


namespace f_inequality_l3721_372106

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem f_inequality (a b c : ℝ) (h1 : a > 0) (h2 : ∀ x, f a b c (1 - x) = f a b c (1 + x)) :
  ∀ x, f a b c (2^x) > f a b c (3^x) :=
by sorry

end f_inequality_l3721_372106


namespace cubic_root_inequality_l3721_372181

theorem cubic_root_inequality (R : ℚ) (h : R ≥ 0) : 
  let a : ℤ := 1
  let b : ℤ := 1
  let c : ℤ := 2
  let d : ℤ := 1
  let e : ℤ := 1
  let f : ℤ := 1
  |((a * R^2 + b * R + c) / (d * R^2 + e * R + f) : ℚ) - (2 : ℚ)^(1/3)| < |R - (2 : ℚ)^(1/3)| :=
by
  sorry

#check cubic_root_inequality

end cubic_root_inequality_l3721_372181


namespace abs_neg_2023_l3721_372114

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end abs_neg_2023_l3721_372114


namespace triangle_pairs_lower_bound_l3721_372183

/-- Given n points in a plane and l line segments, this theorem proves a lower bound
    for the number of triangle pairs formed. -/
theorem triangle_pairs_lower_bound
  (n : ℕ) (l : ℕ) (h_n : n ≥ 4) (h_l : l ≥ n^2 / 4 + 1)
  (no_three_collinear : sorry) -- Hypothesis for no three points being collinear
  (T : ℕ) (h_T : T = sorry) -- Definition of T as the number of triangle pairs
  : T ≥ (l * (4 * l - n^2) * (4 * l - n^2 - n)) / (2 * n^2) := by
  sorry

end triangle_pairs_lower_bound_l3721_372183


namespace sum_of_squares_l3721_372123

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 21) (h2 : x * y = 43) : x^2 + y^2 = 355 := by
  sorry

end sum_of_squares_l3721_372123


namespace worst_player_is_son_or_sister_l3721_372149

-- Define the family members
inductive FamilyMember
  | Woman
  | Brother
  | Son
  | Daughter
  | Sister

-- Define the chess skill level
def ChessSkill := Nat

structure Family where
  members : List FamilyMember
  skills : FamilyMember → ChessSkill
  worst_player : FamilyMember
  best_player : FamilyMember
  twin : FamilyMember

def is_opposite_sex (a b : FamilyMember) : Prop :=
  (a = FamilyMember.Woman ∨ a = FamilyMember.Daughter ∨ a = FamilyMember.Sister) ∧
  (b = FamilyMember.Brother ∨ b = FamilyMember.Son) ∨
  (b = FamilyMember.Woman ∨ b = FamilyMember.Daughter ∨ b = FamilyMember.Sister) ∧
  (a = FamilyMember.Brother ∨ a = FamilyMember.Son)

def are_siblings (a b : FamilyMember) : Prop :=
  (a = FamilyMember.Brother ∧ b = FamilyMember.Sister) ∨
  (a = FamilyMember.Sister ∧ b = FamilyMember.Brother)

def is_between_woman_and_sister (a : FamilyMember) : Prop :=
  a = FamilyMember.Brother ∨ a = FamilyMember.Son ∨ a = FamilyMember.Daughter

theorem worst_player_is_son_or_sister (f : Family) :
  (f.members = [FamilyMember.Woman, FamilyMember.Brother, FamilyMember.Son, FamilyMember.Daughter, FamilyMember.Sister]) →
  (is_opposite_sex f.twin f.best_player) →
  (f.skills f.worst_player = f.skills f.best_player) →
  (are_siblings f.twin f.worst_player ∨ is_between_woman_and_sister f.twin) →
  (f.worst_player = FamilyMember.Son ∨ f.worst_player = FamilyMember.Sister) :=
by sorry

end worst_player_is_son_or_sister_l3721_372149


namespace area_of_three_arc_region_l3721_372168

/-- The area of a region bounded by three identical circular arcs -/
theorem area_of_three_arc_region :
  let r : ℝ := 5 -- radius of each arc
  let θ : ℝ := Real.pi / 2 -- central angle in radians (90 degrees)
  let segment_area : ℝ := r^2 * (θ - Real.sin θ) / 2 -- area of one circular segment
  let total_area : ℝ := 3 * segment_area -- area of the entire region
  total_area = (75 * Real.pi - 150) / 4 :=
by sorry

end area_of_three_arc_region_l3721_372168


namespace sequence_end_point_sequence_end_point_proof_l3721_372190

theorem sequence_end_point : ℕ → Prop :=
  fun n =>
    (∃ k : ℕ, 9 * k ≥ 10 ∧ 9 * (k + 11109) = n) →
    n = 99999

-- The proof is omitted
theorem sequence_end_point_proof : sequence_end_point 99999 := by
  sorry

end sequence_end_point_sequence_end_point_proof_l3721_372190


namespace max_slope_no_lattice_points_l3721_372180

-- Define a lattice point
def is_lattice_point (x y : ℤ) : Prop := True

-- Define the line equation
def on_line (m : ℚ) (x y : ℤ) : Prop := y = m * x + 2

-- Define the condition for no lattice points
def no_lattice_points (m : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x ∧ x ≤ 100 → is_lattice_point x y → ¬(on_line m x y)

-- State the theorem
theorem max_slope_no_lattice_points :
  (∀ m : ℚ, 1/2 < m → m < 50/99 → no_lattice_points m) ∧
  ¬(∀ m : ℚ, 1/2 < m → m < 50/99 + ε → no_lattice_points m) :=
sorry

end max_slope_no_lattice_points_l3721_372180


namespace work_completion_time_l3721_372194

theorem work_completion_time (a_time b_time joint_time b_remaining_time : ℝ) 
  (ha : a_time = 45)
  (hjoint : joint_time = 9)
  (hb_remaining : b_remaining_time = 23)
  (h_work_rate : (joint_time * (1 / a_time + 1 / b_time)) + 
                 (b_remaining_time * (1 / b_time)) = 1) :
  b_time = 40 := by
sorry

end work_completion_time_l3721_372194


namespace least_positive_integer_with_remainders_l3721_372186

theorem least_positive_integer_with_remainders : ∃ (n : ℕ), 
  n > 0 ∧
  n % 3 = 2 ∧
  n % 4 = 3 ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  n % 7 = 6 ∧
  (∀ m : ℕ, m > 0 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 5 = 4 ∧ m % 6 = 5 ∧ m % 7 = 6 → m ≥ n) :=
by
  use 2519
  sorry

#eval 2519 % 3  -- Should output 2
#eval 2519 % 4  -- Should output 3
#eval 2519 % 5  -- Should output 4
#eval 2519 % 6  -- Should output 5
#eval 2519 % 7  -- Should output 6

end least_positive_integer_with_remainders_l3721_372186


namespace tech_students_formula_l3721_372164

/-- The number of students in technology elective courses -/
def tech_students (m : ℕ) : ℚ :=
  (1 / 3 : ℚ) * (m : ℚ) + 8

/-- The number of students in subject elective courses -/
def subject_students (m : ℕ) : ℕ := m

/-- The number of students in physical education and arts elective courses -/
def pe_arts_students (m : ℕ) : ℕ := m + 9

theorem tech_students_formula (m : ℕ) :
  tech_students m = (1 / 3 : ℚ) * (pe_arts_students m : ℚ) + 5 :=
by sorry

end tech_students_formula_l3721_372164


namespace equal_fractions_k_value_l3721_372129

theorem equal_fractions_k_value 
  (x y z k : ℝ) 
  (h : (8 : ℝ) / (x + y + 1) = k / (x + z + 2) ∧ 
       k / (x + z + 2) = (12 : ℝ) / (z - y + 3)) : 
  k = 20 := by sorry

end equal_fractions_k_value_l3721_372129


namespace both_teasers_count_l3721_372115

/-- The number of brainiacs who like both rebus teasers and math teasers -/
def both_teasers (total : ℕ) (rebus : ℕ) (math : ℕ) (neither : ℕ) (math_only : ℕ) : ℕ :=
  total - rebus - math + (rebus + math - (total - neither))

theorem both_teasers_count :
  both_teasers 100 (2 * 50) 50 4 20 = 18 := by
  sorry

end both_teasers_count_l3721_372115


namespace fraction_simplification_l3721_372134

theorem fraction_simplification :
  (5 : ℝ) / (Real.sqrt 75 + 3 * Real.sqrt 48 + Real.sqrt 27) = Real.sqrt 3 / 12 := by
  sorry

end fraction_simplification_l3721_372134


namespace complex_product_real_imag_equal_l3721_372108

theorem complex_product_real_imag_equal (a : ℝ) : 
  (Complex.re ((1 + 2*Complex.I) * (a + Complex.I)) = Complex.im ((1 + 2*Complex.I) * (a + Complex.I))) → 
  a = -3 := by
  sorry

end complex_product_real_imag_equal_l3721_372108


namespace candidate_b_votes_l3721_372120

/-- Proves that candidate B received 4560 valid votes given the election conditions -/
theorem candidate_b_votes (total_eligible : Nat) (abstention_rate : Real) (invalid_vote_rate : Real) 
  (c_vote_percentage : Real) (a_vote_reduction : Real) 
  (h1 : total_eligible = 12000)
  (h2 : abstention_rate = 0.1)
  (h3 : invalid_vote_rate = 0.2)
  (h4 : c_vote_percentage = 0.05)
  (h5 : a_vote_reduction = 0.2) : 
  ∃ (b_votes : Nat), b_votes = 4560 := by
  sorry

end candidate_b_votes_l3721_372120


namespace expand_product_l3721_372126

theorem expand_product (x : ℝ) : (x + 3) * (x - 6) * (x + 2) = x^3 - x^2 - 24*x - 36 := by
  sorry

end expand_product_l3721_372126


namespace tangent_line_and_lower_bound_l3721_372118

noncomputable def f (x : ℝ) := Real.exp x - 3 * x^2 + 4 * x

theorem tangent_line_and_lower_bound :
  (∃ (b : ℝ), ∀ x, (Real.exp 1 - 2) * x + b = f 1 + (Real.exp 1 - 6 + 4) * (x - 1)) ∧
  (∀ x ≥ 1, f x > 3) ∧
  (∃ x₀ ≥ 1, f x₀ < 4) :=
sorry

end tangent_line_and_lower_bound_l3721_372118
