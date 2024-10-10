import Mathlib

namespace triangle_inequality_l2695_269520

theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a * b + b * c + c * a ≤ a^2 + b^2 + c^2 ∧ a^2 + b^2 + c^2 < 2 * (a * b + b * c + c * a) := by
sorry

end triangle_inequality_l2695_269520


namespace factory_earnings_l2695_269546

-- Define the parameters
def hours_machines_123 : ℕ := 23
def hours_machine_4 : ℕ := 12
def production_rate_12 : ℕ := 2
def production_rate_34 : ℕ := 3
def price_13 : ℕ := 50
def price_24 : ℕ := 60

-- Define the earnings calculation function
def calculate_earnings (hours : ℕ) (rate : ℕ) (price : ℕ) : ℕ :=
  hours * rate * price

-- Theorem statement
theorem factory_earnings :
  calculate_earnings hours_machines_123 production_rate_12 price_13 +
  calculate_earnings hours_machines_123 production_rate_12 price_24 +
  calculate_earnings hours_machines_123 production_rate_34 price_13 +
  calculate_earnings hours_machine_4 production_rate_34 price_24 = 10670 := by
  sorry


end factory_earnings_l2695_269546


namespace three_primes_sum_47_product_1705_l2695_269543

theorem three_primes_sum_47_product_1705 : ∃ p q r : ℕ, 
  Prime p ∧ Prime q ∧ Prime r ∧ 
  p + q + r = 47 ∧ 
  p * q * r = 1705 := by
sorry

end three_primes_sum_47_product_1705_l2695_269543


namespace swimmer_speed_proof_l2695_269572

def swimmer_problem (distance : ℝ) (current_speed : ℝ) (time : ℝ) : Prop :=
  let still_water_speed := (distance / time) + current_speed
  still_water_speed = 3

theorem swimmer_speed_proof :
  swimmer_problem 8 1.4 5 :=
sorry

end swimmer_speed_proof_l2695_269572


namespace intersection_of_three_lines_l2695_269570

/-- Given three lines that intersect at the same point, prove the value of k. -/
theorem intersection_of_three_lines (x y : ℝ) (k : ℝ) : 
  y = -4 * x + 2 ∧ 
  y = 3 * x - 18 ∧ 
  y = 7 * x + k 
  → k = -206 / 7 := by
  sorry

end intersection_of_three_lines_l2695_269570


namespace hyperbola_equation_l2695_269574

/-- Given a hyperbola with a = 5 and c = 7, prove its standard equation. -/
theorem hyperbola_equation (a c : ℝ) (ha : a = 5) (hc : c = 7) :
  ∃ (x y : ℝ → ℝ), 
    (∀ t, x t ^ 2 / 25 - y t ^ 2 / 24 = 1) ∨
    (∀ t, y t ^ 2 / 25 - x t ^ 2 / 24 = 1) :=
by sorry

end hyperbola_equation_l2695_269574


namespace quadratic_inequality_condition_l2695_269516

-- Define the quadratic function
def f (b : ℝ) (x : ℝ) : ℝ := -x^2 + b*x + 1

-- Define the condition for the inequality
def condition (b : ℝ) : Prop :=
  ∀ x, f b x < 0 ↔ (x < 2 ∨ x > 6)

-- Theorem statement
theorem quadratic_inequality_condition (b : ℝ) :
  condition b → b = 8 := by sorry

end quadratic_inequality_condition_l2695_269516


namespace apple_tree_problem_l2695_269578

/-- The number of apples Rachel picked from the tree -/
def apples_picked : ℝ := 7.5

/-- The number of new apples that grew on the tree after Rachel picked -/
def new_apples : ℝ := 2.3

/-- The number of apples currently on the tree -/
def current_apples : ℝ := 6.2

/-- The original number of apples on the tree -/
def original_apples : ℝ := apples_picked + current_apples - new_apples

theorem apple_tree_problem :
  original_apples = 11.4 := by sorry

end apple_tree_problem_l2695_269578


namespace sin_product_equals_one_eighth_l2695_269566

theorem sin_product_equals_one_eighth :
  Real.sin (12 * π / 180) * Real.sin (36 * π / 180) * Real.sin (54 * π / 180) * Real.sin (84 * π / 180) = 1/8 :=
by sorry

end sin_product_equals_one_eighth_l2695_269566


namespace smallest_angle_solution_l2695_269533

theorem smallest_angle_solution (x : Real) : 
  (8 * Real.sin x ^ 2 * Real.cos x ^ 4 - 8 * Real.sin x ^ 4 * Real.cos x ^ 2 = 1) →
  (x ≥ 0) →
  (∀ y, y > 0 ∧ y < x → 8 * Real.sin y ^ 2 * Real.cos y ^ 4 - 8 * Real.sin y ^ 4 * Real.cos y ^ 2 ≠ 1) →
  x = 10 * π / 180 :=
by sorry

end smallest_angle_solution_l2695_269533


namespace certain_number_is_30_l2695_269539

theorem certain_number_is_30 (x : ℝ) : 0.5 * x = 0.1667 * x + 10 → x = 30 := by
  sorry

end certain_number_is_30_l2695_269539


namespace max_discount_percentage_l2695_269562

/-- The maximum discount percentage that can be applied to a product while maintaining a minimum profit margin. -/
theorem max_discount_percentage
  (cost : ℝ)              -- Cost price in yuan
  (price : ℝ)             -- Selling price in yuan
  (min_margin : ℝ)        -- Minimum profit margin as a decimal
  (h_cost : cost = 100)   -- Cost is 100 yuan
  (h_price : price = 150) -- Price is 150 yuan
  (h_margin : min_margin = 0.2) -- Minimum margin is 20%
  : ∃ (max_discount : ℝ),
    max_discount = 20 ∧
    ∀ (discount : ℝ),
      0 ≤ discount ∧ discount ≤ max_discount →
      (price * (1 - discount / 100) - cost) / cost ≥ min_margin :=
by sorry

end max_discount_percentage_l2695_269562


namespace stating_weaver_production_increase_l2695_269590

/-- Represents the daily increase in fabric production -/
def daily_increase : ℚ := 16 / 29

/-- Represents the initial daily production -/
def initial_production : ℕ := 5

/-- Represents the number of days -/
def days : ℕ := 30

/-- Represents the total production over the given period -/
def total_production : ℕ := 390

/-- 
Theorem stating that given the initial production and total production over a period,
the daily increase in production is as calculated.
-/
theorem weaver_production_increase : 
  initial_production * days + (days * (days - 1) / 2) * daily_increase = total_production := by
  sorry


end stating_weaver_production_increase_l2695_269590


namespace square_root_of_16_l2695_269579

theorem square_root_of_16 : {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end square_root_of_16_l2695_269579


namespace decreasing_linear_function_iff_negative_slope_l2695_269506

/-- A linear function f(x) = ax + b -/
def linear_function (a b : ℝ) : ℝ → ℝ := λ x ↦ a * x + b

/-- A function is decreasing if f(x1) > f(x2) whenever x1 < x2 -/
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2

theorem decreasing_linear_function_iff_negative_slope (m : ℝ) :
  is_decreasing (linear_function (m + 3) (-2)) ↔ m < -3 :=
sorry

end decreasing_linear_function_iff_negative_slope_l2695_269506


namespace nested_bracket_equals_two_l2695_269564

def bracket (x y z : ℚ) : ℚ := (x + y) / z

theorem nested_bracket_equals_two :
  bracket (bracket 45 15 60) (bracket 3 3 6) (bracket 24 6 30) = 2 := by
  sorry

end nested_bracket_equals_two_l2695_269564


namespace new_average_weight_l2695_269544

theorem new_average_weight (initial_students : Nat) (initial_avg_weight : ℝ) (new_student_weight : ℝ) :
  initial_students = 19 →
  initial_avg_weight = 15 →
  new_student_weight = 7 →
  let total_weight := initial_students * initial_avg_weight
  let new_total_weight := total_weight + new_student_weight
  let new_avg_weight := new_total_weight / (initial_students + 1)
  new_avg_weight = 14.6 := by
  sorry

end new_average_weight_l2695_269544


namespace min_value_quadratic_l2695_269548

theorem min_value_quadratic (k : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 - 8 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 6 * y + 9 ≥ 0) ∧
  (∃ x y : ℝ, 3 * x^2 - 8 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 6 * y + 9 = 0) →
  k = 3/2 ∨ k = -3/2 := by
sorry

end min_value_quadratic_l2695_269548


namespace unique_g_two_l2695_269553

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → g x * g y = g (x * y) + 2010 * (1 / x^2 + 1 / y^2 + 2009)

theorem unique_g_two (g : ℝ → ℝ) (h : FunctionalEquation g) :
    ∃! v, g 2 = v ∧ v = 8041 / 4 := by
  sorry

end unique_g_two_l2695_269553


namespace probability_two_dice_rolls_l2695_269530

-- Define the number of sides on each die
def sides : ℕ := 8

-- Define the favorable outcomes for the first die (numbers less than 4)
def favorable_first : ℕ := 3

-- Define the favorable outcomes for the second die (numbers greater than 5)
def favorable_second : ℕ := 3

-- State the theorem
theorem probability_two_dice_rolls : 
  (favorable_first / sides) * (favorable_second / sides) = 9 / 64 := by
  sorry

end probability_two_dice_rolls_l2695_269530


namespace max_children_in_class_l2695_269569

theorem max_children_in_class (x : ℕ) : 
  (∃ (chocolates_per_box : ℕ),
    -- Original plan with 6 boxes
    6 * chocolates_per_box = 10 * x + 40 ∧
    -- New plan with 4 boxes
    4 * chocolates_per_box ≥ 8 * (x - 1) + 4 ∧
    4 * chocolates_per_box < 8 * (x - 1) + 8) →
  x ≤ 23 :=
sorry

end max_children_in_class_l2695_269569


namespace triangle_angle_sum_and_type_l2695_269580

/-- A triangle with angles a, b, and c is right if its largest angle is 90 degrees --/
def is_right_triangle (a b c : ℝ) : Prop :=
  max a (max b c) = 90

theorem triangle_angle_sum_and_type 
  (a b : ℝ) 
  (ha : a = 56)
  (hb : b = 34) :
  let c := 180 - a - b
  ∃ (x : ℝ), x = c ∧ x = 90 ∧ is_right_triangle a b c :=
by
  sorry

end triangle_angle_sum_and_type_l2695_269580


namespace remaining_crayons_l2695_269563

def initial_crayons : ℕ := 440
def crayons_given_away : ℕ := 111
def crayons_lost : ℕ := 106

theorem remaining_crayons :
  initial_crayons - crayons_given_away - crayons_lost = 223 := by
  sorry

end remaining_crayons_l2695_269563


namespace confectioner_customers_l2695_269525

/-- The number of regular customers for a confectioner -/
def regular_customers : ℕ := 28

/-- The total number of pastries -/
def total_pastries : ℕ := 392

/-- The number of customers in the alternative scenario -/
def alternative_customers : ℕ := 49

/-- The difference in pastries per customer between regular and alternative scenarios -/
def pastry_difference : ℕ := 6

theorem confectioner_customers :
  regular_customers = 28 ∧
  total_pastries = 392 ∧
  alternative_customers = 49 ∧
  pastry_difference = 6 ∧
  (total_pastries / regular_customers : ℚ) = 
    (total_pastries / alternative_customers : ℚ) + pastry_difference := by
  sorry

end confectioner_customers_l2695_269525


namespace betty_oranges_l2695_269591

theorem betty_oranges (boxes : ℝ) (oranges_per_box : ℕ) :
  boxes = 3.0 → oranges_per_box = 24 → boxes * oranges_per_box = 72 := by
  sorry

end betty_oranges_l2695_269591


namespace fruit_pricing_problem_l2695_269523

theorem fruit_pricing_problem (x y : ℚ) : 
  x + y = 1000 →
  (11/9) * x + (4/7) * y = 999 →
  (9 * (11/9) = 11 ∧ 7 * (4/7) = 4) :=
by sorry

end fruit_pricing_problem_l2695_269523


namespace x15x_divisible_by_18_l2695_269513

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ := 
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def x15x (x : ℕ) : ℕ := x * 1000 + 100 + 50 + x

theorem x15x_divisible_by_18 :
  ∃! x : ℕ, x < 10 ∧ is_four_digit (x15x x) ∧ (x15x x) % 18 = 0 ∧ x = 6 := by
sorry

end x15x_divisible_by_18_l2695_269513


namespace total_distance_walked_l2695_269529

theorem total_distance_walked (first_part second_part : Real) 
  (h1 : first_part = 0.75)
  (h2 : second_part = 0.25) : 
  first_part + second_part = 1 := by
sorry

end total_distance_walked_l2695_269529


namespace painting_time_equation_l2695_269518

theorem painting_time_equation (doug_time dave_time lunch_break : ℝ) 
  (h_doug : doug_time = 6)
  (h_dave : dave_time = 8)
  (h_lunch : lunch_break = 2)
  (t : ℝ) :
  (1 / doug_time + 1 / dave_time) * (t - lunch_break) = 1 :=
by sorry

end painting_time_equation_l2695_269518


namespace x_lt_5_necessary_not_sufficient_for_x_lt_2_l2695_269557

theorem x_lt_5_necessary_not_sufficient_for_x_lt_2 :
  (∀ x : ℝ, x < 2 → x < 5) ∧ (∃ x : ℝ, x < 5 ∧ ¬(x < 2)) :=
by sorry

end x_lt_5_necessary_not_sufficient_for_x_lt_2_l2695_269557


namespace fraction_subtraction_equality_l2695_269537

theorem fraction_subtraction_equality : 
  (4 + 6 + 8 + 10) / (3 + 5 + 7) - (3 + 5 + 7 + 9) / (4 + 6 + 8) = 8 / 15 := by
  sorry

end fraction_subtraction_equality_l2695_269537


namespace weight_measurement_l2695_269586

theorem weight_measurement (n : ℕ) (h : 1 ≤ n ∧ n ≤ 63) :
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ : Bool),
    n = (if a₀ then 1 else 0) +
        (if a₁ then 2 else 0) +
        (if a₂ then 4 else 0) +
        (if a₃ then 8 else 0) +
        (if a₄ then 16 else 0) +
        (if a₅ then 32 else 0) :=
by sorry

end weight_measurement_l2695_269586


namespace pq_length_l2695_269502

/-- Given two lines and a point R that is the midpoint of a line segment PQ, 
    where P is on one line and Q is on the other, prove that the length of PQ 
    is √56512 / 33. -/
theorem pq_length (P Q R : ℝ × ℝ) : 
  R = (10, 8) →
  (∃ x, P = (x, 2*x)) →
  (∃ y, Q = (y, 4*y/11)) →
  R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) →
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = Real.sqrt 56512 / 33 := by
  sorry

#check pq_length

end pq_length_l2695_269502


namespace tv_show_payment_ratio_l2695_269521

/-- The ratio of payments to major and minor characters in a TV show -/
theorem tv_show_payment_ratio :
  let num_main_characters : ℕ := 5
  let num_minor_characters : ℕ := 4
  let minor_character_payment : ℕ := 15000
  let total_payment : ℕ := 285000
  let minor_characters_total : ℕ := num_minor_characters * minor_character_payment
  let major_characters_total : ℕ := total_payment - minor_characters_total
  (major_characters_total : ℚ) / minor_characters_total = 15 / 4 := by
  sorry

end tv_show_payment_ratio_l2695_269521


namespace largest_integer_satisfying_inequality_l2695_269545

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, (7 - 5*x > 22) → x ≤ -4 ∧ (7 - 5*(-4) > 22) :=
by sorry

end largest_integer_satisfying_inequality_l2695_269545


namespace rectangular_solid_width_l2695_269527

/-- The surface area of a rectangular solid given its length, width, and height. -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The width of a rectangular solid with length 5, depth 1, and surface area 58 is 4. -/
theorem rectangular_solid_width :
  ∃ w : ℝ, w = 4 ∧ surface_area 5 w 1 = 58 := by
  sorry

end rectangular_solid_width_l2695_269527


namespace altitude_and_angle_bisector_equations_l2695_269575

/-- Triangle ABC with vertices A(1,-1), B(-1,3), C(3,0) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- The given triangle -/
def ABC : Triangle :=
  { A := (1, -1),
    B := (-1, 3),
    C := (3, 0) }

/-- Altitude from A to BC -/
def altitude (t : Triangle) : ℝ × ℝ → Prop :=
  λ p => 4 * p.1 - 3 * p.2 - 7 = 0

/-- Angle bisector of ∠BAC -/
def angle_bisector (t : Triangle) : ℝ × ℝ → Prop :=
  λ p => p.1 - p.2 - 4 = 0

/-- Main theorem -/
theorem altitude_and_angle_bisector_equations :
  (∀ p, altitude ABC p ↔ 4 * p.1 - 3 * p.2 - 7 = 0) ∧
  (∀ p, angle_bisector ABC p ↔ p.1 - p.2 - 4 = 0) := by
  sorry

end altitude_and_angle_bisector_equations_l2695_269575


namespace dream_team_strategy_l2695_269511

/-- Represents the probabilities of correct answers for each team member and category -/
structure TeamProbabilities where
  a_category_a : ℝ
  a_category_b : ℝ
  b_category_a : ℝ
  b_category_b : ℝ

/-- Calculates the probability of entering the final round when answering a specific category first -/
def probability_enter_final (probs : TeamProbabilities) (start_with_a : Bool) : ℝ :=
  if start_with_a then
    let p3 := probs.a_category_a * probs.b_category_a * probs.a_category_b * (1 - probs.b_category_b) +
              probs.a_category_a * probs.b_category_a * (1 - probs.a_category_b) * probs.b_category_b
    let p4 := probs.a_category_a * probs.b_category_a * probs.a_category_b * probs.b_category_b
    p3 + p4
  else
    let p3 := probs.a_category_b * probs.b_category_b * probs.a_category_a * (1 - probs.b_category_a) +
              probs.a_category_b * probs.b_category_b * (1 - probs.a_category_a) * probs.b_category_a
    let p4 := probs.a_category_b * probs.b_category_b * probs.a_category_a * probs.b_category_a
    p3 + p4

/-- The main theorem to be proved -/
theorem dream_team_strategy (probs : TeamProbabilities)
  (h1 : probs.a_category_a = 0.7)
  (h2 : probs.a_category_b = 0.5)
  (h3 : probs.b_category_a = 0.4)
  (h4 : probs.b_category_b = 0.8) :
  probability_enter_final probs false > probability_enter_final probs true :=
by sorry

end dream_team_strategy_l2695_269511


namespace eel_length_ratio_l2695_269585

theorem eel_length_ratio (total_length : ℝ) (jenna_length : ℝ) :
  total_length = 64 →
  jenna_length = 16 →
  (jenna_length / (total_length - jenna_length) = 1 / 3) :=
by
  sorry

end eel_length_ratio_l2695_269585


namespace last_three_digits_of_8_to_104_l2695_269504

theorem last_three_digits_of_8_to_104 : 8^104 ≡ 984 [ZMOD 1000] := by
  sorry

end last_three_digits_of_8_to_104_l2695_269504


namespace max_sum_of_squares_l2695_269547

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 18 →
  a * b + c + d = 85 →
  a * d + b * c = 187 →
  c * d = 105 →
  a^2 + b^2 + c^2 + d^2 ≤ 1486 := by
  sorry


end max_sum_of_squares_l2695_269547


namespace time_for_b_alone_l2695_269596

/-- The time it takes for person B to complete the work alone, given the conditions of the problem. -/
theorem time_for_b_alone (a b c : ℝ) : 
  a = 1/3 →  -- A can do the work in 3 hours
  b + c = 1/3 →  -- B and C together can do it in 3 hours
  a + c = 1/2 →  -- A and C together can do it in 2 hours
  1/b = 6 :=  -- B alone takes 6 hours
by sorry


end time_for_b_alone_l2695_269596


namespace largest_three_digit_divisible_by_sum_and_11_l2695_269552

/-- Represents a three-digit integer -/
structure ThreeDigitInteger where
  value : ℕ
  is_three_digit : 100 ≤ value ∧ value ≤ 999

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem largest_three_digit_divisible_by_sum_and_11 :
  ∃ (n : ThreeDigitInteger),
    (n.value % sum_of_digits n.value = 0) ∧
    (sum_of_digits n.value % 11 = 0) ∧
    (∀ (m : ThreeDigitInteger),
      (m.value % sum_of_digits m.value = 0) ∧
      (sum_of_digits m.value % 11 = 0) →
      m.value ≤ n.value) ∧
    n.value = 990 :=
  sorry


end largest_three_digit_divisible_by_sum_and_11_l2695_269552


namespace perfect_square_quadratic_l2695_269509

theorem perfect_square_quadratic (m : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 - (m+1)*x + 1 = k^2) → (m = 1 ∨ m = -3) := by
  sorry

end perfect_square_quadratic_l2695_269509


namespace leahs_outfits_l2695_269595

/-- Calculate the number of possible outfits given the number of options for each clothing item -/
def number_of_outfits (trousers shirts jackets shoes : ℕ) : ℕ :=
  trousers * shirts * jackets * shoes

/-- Theorem: The number of outfits for Leah's wardrobe is 840 -/
theorem leahs_outfits :
  number_of_outfits 5 6 4 7 = 840 := by
  sorry

end leahs_outfits_l2695_269595


namespace substance_mass_l2695_269583

/-- Given a substance where 1 gram occupies 5 cubic centimeters, 
    the mass of 1 cubic meter of this substance is 200 kilograms. -/
theorem substance_mass (substance_density : ℝ) : 
  substance_density = 1 / 5 → -- 1 gram occupies 5 cubic centimeters
  (1 : ℝ) * substance_density * 1000000 / 1000 = 200 := by
  sorry

end substance_mass_l2695_269583


namespace parking_lot_motorcycles_l2695_269524

/-- The number of cars in the parking lot -/
def num_cars : ℕ := 19

/-- The number of wheels per car -/
def wheels_per_car : ℕ := 5

/-- The number of wheels per motorcycle -/
def wheels_per_motorcycle : ℕ := 2

/-- The total number of wheels for all vehicles -/
def total_wheels : ℕ := 117

/-- The number of motorcycles in the parking lot -/
def num_motorcycles : ℕ := (total_wheels - num_cars * wheels_per_car) / wheels_per_motorcycle

theorem parking_lot_motorcycles : num_motorcycles = 11 := by
  sorry

end parking_lot_motorcycles_l2695_269524


namespace line_segment_ratio_l2695_269587

theorem line_segment_ratio (a b c d : ℝ) :
  let O := 0
  let A := a
  let B := b
  let C := c
  let D := d
  ∀ P : ℝ, B < P ∧ P < C →
  (P - A) / (D - P) = (P - B) / (C - P) →
  P = (a * c - b * d) / (a - b + c - d) :=
by sorry

end line_segment_ratio_l2695_269587


namespace absolute_value_problem_l2695_269599

theorem absolute_value_problem (x y : ℝ) 
  (h1 : x^2 + y^2 = 1) 
  (h2 : 20 * x^3 - 15 * x = 3) : 
  |20 * y^3 - 15 * y| = 4 := by
  sorry

end absolute_value_problem_l2695_269599


namespace cube_root_simplification_l2695_269577

theorem cube_root_simplification (N : ℝ) (h : N > 1) :
  (N^2 * (N^3 * N^(2/3))^(1/3))^(1/3) = N^(29/27) := by
  sorry

end cube_root_simplification_l2695_269577


namespace sin_double_angle_proof_l2695_269515

theorem sin_double_angle_proof (α : Real) (h : Real.cos (π/4 - α) = 3/5) : 
  Real.sin (2 * α) = -7/25 := by
  sorry

end sin_double_angle_proof_l2695_269515


namespace sqrt_3_simplest_l2695_269526

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℝ) : Prop := ∃ m : ℤ, n = m^2

-- Define what it means for a quadratic radical to be in its simplest form
def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  x > 0 ∧ ¬(is_perfect_square x) ∧ ∀ y z : ℝ, (y > 1 ∧ z > 1 ∧ x = y * z) → ¬(is_perfect_square y)

-- State the theorem
theorem sqrt_3_simplest :
  is_simplest_quadratic_radical 3 ∧
  ¬(is_simplest_quadratic_radical (1/2)) ∧
  ¬(is_simplest_quadratic_radical 8) ∧
  ¬(is_simplest_quadratic_radical 4) :=
sorry

end sqrt_3_simplest_l2695_269526


namespace polynomial_root_product_l2695_269576

theorem polynomial_root_product (y₁ y₂ y₃ : ℂ) : 
  (y₁^3 - 3*y₁ + 1 = 0) → 
  (y₂^3 - 3*y₂ + 1 = 0) → 
  (y₃^3 - 3*y₃ + 1 = 0) → 
  (y₁^3 + 2) * (y₂^3 + 2) * (y₃^3 + 2) = -26 := by
sorry

end polynomial_root_product_l2695_269576


namespace unit_square_quadrilateral_bounds_l2695_269542

/-- A quadrilateral formed by selecting one point on each side of a unit square -/
structure UnitSquareQuadrilateral where
  a : ℝ  -- Length of side a
  b : ℝ  -- Length of side b
  c : ℝ  -- Length of side c
  d : ℝ  -- Length of side d
  ha : 0 ≤ a ∧ a ≤ 1  -- a is between 0 and 1
  hb : 0 ≤ b ∧ b ≤ 1  -- b is between 0 and 1
  hc : 0 ≤ c ∧ c ≤ 1  -- c is between 0 and 1
  hd : 0 ≤ d ∧ d ≤ 1  -- d is between 0 and 1

theorem unit_square_quadrilateral_bounds (q : UnitSquareQuadrilateral) :
  2 ≤ q.a^2 + q.b^2 + q.c^2 + q.d^2 ∧ q.a^2 + q.b^2 + q.c^2 + q.d^2 ≤ 4 ∧
  2 * Real.sqrt 2 ≤ q.a + q.b + q.c + q.d ∧ q.a + q.b + q.c + q.d ≤ 4 := by
  sorry

end unit_square_quadrilateral_bounds_l2695_269542


namespace sin_90_degrees_l2695_269551

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end sin_90_degrees_l2695_269551


namespace circles_intersect_l2695_269501

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 9

-- Define the centers and radii
def center1 : ℝ × ℝ := (-2, 0)
def center2 : ℝ × ℝ := (2, 1)
def radius1 : ℝ := 2
def radius2 : ℝ := 3

-- Theorem stating that the circles intersect
theorem circles_intersect :
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y :=
sorry

end circles_intersect_l2695_269501


namespace max_answered_A_l2695_269597

/-- Represents the number of people who answered each combination of questions correctly. -/
structure Answers :=
  (a : ℕ)  -- Only A
  (b : ℕ)  -- Only B
  (c : ℕ)  -- Only C
  (ab : ℕ) -- A and B
  (ac : ℕ) -- A and C
  (bc : ℕ) -- B and C
  (abc : ℕ) -- All three

/-- The conditions of the math competition problem. -/
def ValidAnswers (ans : Answers) : Prop :=
  -- Total participants
  ans.a + ans.b + ans.c + ans.ab + ans.ac + ans.bc + ans.abc = 39 ∧
  -- Condition about A answers
  ans.a = ans.ab + ans.ac + ans.abc + 5 ∧
  -- Condition about B and C answers (not A)
  ans.b + ans.bc = 2 * (ans.c + ans.bc) ∧
  -- Condition about only A, B, and C answers
  ans.a = ans.b + ans.c

/-- The number of people who answered A correctly. -/
def AnsweredA (ans : Answers) : ℕ :=
  ans.a + ans.ab + ans.ac + ans.abc

/-- The theorem stating the maximum number of people who answered A correctly. -/
theorem max_answered_A :
  ∀ ans : Answers, ValidAnswers ans → AnsweredA ans ≤ 23 :=
sorry

end max_answered_A_l2695_269597


namespace monotonic_increasing_range_always_positive_range_l2695_269528

def f (k : ℝ) (x : ℝ) : ℝ := x^2 + 2*k*x + 4

-- Part 1
theorem monotonic_increasing_range (k : ℝ) :
  (∀ x ∈ Set.Icc 1 4, Monotone (f k)) ↔ k ≥ -1 :=
sorry

-- Part 2
theorem always_positive_range (k : ℝ) :
  (∀ x : ℝ, f k x > 0) ↔ -2 < k ∧ k < 2 :=
sorry

end monotonic_increasing_range_always_positive_range_l2695_269528


namespace school_bus_seats_l2695_269507

/-- Given a school with students and buses, calculate the number of seats per bus. -/
def seats_per_bus (total_students : ℕ) (num_buses : ℕ) : ℕ :=
  total_students / num_buses

/-- Theorem stating that for a school with 11210 students and 95 buses, each bus has 118 seats. -/
theorem school_bus_seats :
  seats_per_bus 11210 95 = 118 := by
  sorry

end school_bus_seats_l2695_269507


namespace fixed_point_of_exponential_function_l2695_269558

/-- Given a real number a, prove that the function f(x) = a^(x-1) + 3 passes through the point (1, 4) -/
theorem fixed_point_of_exponential_function (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 3
  f 1 = 4 := by
  sorry

end fixed_point_of_exponential_function_l2695_269558


namespace arithmetic_sequence_formula_l2695_269593

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧
    (∀ n : ℕ, a (n + 1) = a n + d) ∧
    a 3 = 5 ∧
    ∃ r : ℝ, (a 2 = a 1 * r) ∧ (a 5 = a 2 * r)

/-- The main theorem -/
theorem arithmetic_sequence_formula (a : ℕ → ℝ) (h : ArithmeticSequence a) :
    ∀ n : ℕ, a n = 2 * n - 1 := by
  sorry

end arithmetic_sequence_formula_l2695_269593


namespace four_digit_number_satisfies_condition_l2695_269550

/-- Represents a four-digit number -/
def FourDigitNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

/-- Splits a four-digit number into two two-digit numbers -/
def SplitNumber (n : ℕ) : ℕ × ℕ :=
  (n / 100, n % 100)

/-- Checks if a number satisfies the given condition -/
def SatisfiesCondition (n : ℕ) : Prop :=
  let (a, b) := SplitNumber n
  (10 * a + b / 10) * (b % 10 + 10 * (b / 10)) + 10 * a = n

theorem four_digit_number_satisfies_condition :
  FourDigitNumber 1995 ∧
  (SplitNumber 1995).2 % 10 = 5 ∧
  SatisfiesCondition 1995 :=
by sorry

end four_digit_number_satisfies_condition_l2695_269550


namespace find_missing_number_l2695_269531

theorem find_missing_number (x : ℕ) : 
  (55 + 48 + x + 2 + 684 + 42) / 6 = 223 → x = 507 := by
  sorry

end find_missing_number_l2695_269531


namespace no_x_axis_intersection_l2695_269560

-- Define the quadratic function
def f (x : ℝ) : ℝ := -(x + 2)^2 - 1

-- Theorem stating that the function does not intersect the x-axis
theorem no_x_axis_intersection :
  ∀ x : ℝ, f x ≠ 0 := by
  sorry

end no_x_axis_intersection_l2695_269560


namespace quiche_volume_l2695_269588

/-- Calculate the total volume of a vegetable quiche --/
theorem quiche_volume 
  (spinach_initial : ℝ) 
  (mushrooms_initial : ℝ) 
  (onions_initial : ℝ)
  (spinach_reduction : ℝ) 
  (mushrooms_reduction : ℝ) 
  (onions_reduction : ℝ)
  (cream_cheese : ℝ)
  (eggs : ℝ)
  (h1 : spinach_initial = 40)
  (h2 : mushrooms_initial = 25)
  (h3 : onions_initial = 15)
  (h4 : spinach_reduction = 0.20)
  (h5 : mushrooms_reduction = 0.65)
  (h6 : onions_reduction = 0.50)
  (h7 : cream_cheese = 6)
  (h8 : eggs = 4) :
  spinach_initial * spinach_reduction + 
  mushrooms_initial * mushrooms_reduction + 
  onions_initial * onions_reduction + 
  cream_cheese + eggs = 41.75 := by
sorry

end quiche_volume_l2695_269588


namespace smallest_leading_coeff_quadratic_roots_existence_quadratic_roots_five_smallest_leading_coeff_is_five_l2695_269584

theorem smallest_leading_coeff_quadratic_roots (a : ℕ) : 
  (∃ (b c : ℤ) (x₁ x₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    0 < x₁ ∧ x₁ < 1 ∧ 
    0 < x₂ ∧ x₂ < 1 ∧ 
    ∀ (x : ℝ), (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ) = 0 ↔ x = x₁ ∨ x = x₂) →
  a ≥ 5 :=
by sorry

theorem existence_quadratic_roots_five :
  ∃ (b c : ℤ) (x₁ x₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    0 < x₁ ∧ x₁ < 1 ∧ 
    0 < x₂ ∧ x₂ < 1 ∧ 
    ∀ (x : ℝ), (5 : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ) = 0 ↔ x = x₁ ∨ x = x₂ :=
by sorry

theorem smallest_leading_coeff_is_five : 
  ∀ (a : ℕ), 
    (∃ (b c : ℤ) (x₁ x₂ : ℝ), 
      x₁ ≠ x₂ ∧ 
      0 < x₁ ∧ x₁ < 1 ∧ 
      0 < x₂ ∧ x₂ < 1 ∧ 
      ∀ (x : ℝ), (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ) = 0 ↔ x = x₁ ∨ x = x₂) →
    a ≥ 5 :=
by sorry

end smallest_leading_coeff_quadratic_roots_existence_quadratic_roots_five_smallest_leading_coeff_is_five_l2695_269584


namespace train_length_proof_l2695_269522

/-- Proves that the length of a train is equal to the total length of the train and bridge,
    given the train's speed, time to cross the bridge, and total length of train and bridge. -/
theorem train_length_proof (train_speed : ℝ) (crossing_time : ℝ) (total_length : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  total_length = 245 →
  total_length = train_speed * crossing_time - total_length + total_length :=
by
  sorry

end train_length_proof_l2695_269522


namespace field_width_l2695_269519

/-- Given a rectangular field with length 75 meters, where running around it 3 times
    covers a distance of 540 meters, prove that the width of the field is 15 meters. -/
theorem field_width (length : ℝ) (width : ℝ) (perimeter : ℝ) :
  length = 75 →
  3 * perimeter = 540 →
  perimeter = 2 * (length + width) →
  width = 15 := by
sorry

end field_width_l2695_269519


namespace imaginary_part_proof_l2695_269589

def i : ℂ := Complex.I

def z : ℂ := 1 - i

theorem imaginary_part_proof : Complex.im ((2 / z) + i ^ 2) = 1 := by
  sorry

end imaginary_part_proof_l2695_269589


namespace add_zero_or_nine_divisible_by_nine_l2695_269555

/-- Represents a ten-digit number with different digits -/
def TenDigitNumber := {n : Fin 10 → Fin 10 // Function.Injective n}

/-- The sum of digits in a ten-digit number -/
def digitSum (n : TenDigitNumber) : ℕ :=
  (Finset.univ.sum fun i => (n.val i).val)

/-- The theorem stating that adding 0 or 9 to a ten-digit number with different digits 
    results in a number divisible by 9 -/
theorem add_zero_or_nine_divisible_by_nine (n : TenDigitNumber) :
  (∃ x : Fin 10, x = 0 ∨ x = 9) ∧ 
  (∃ m : ℕ, (digitSum n + x) = 9 * m) := by
  sorry


end add_zero_or_nine_divisible_by_nine_l2695_269555


namespace joses_swimming_pool_charge_l2695_269512

/-- Proves that the daily charge for kids in Jose's swimming pool is $3 -/
theorem joses_swimming_pool_charge (kid_charge : ℚ) (adult_charge : ℚ) 
  (h1 : adult_charge = 2 * kid_charge) 
  (h2 : 8 * kid_charge + 10 * adult_charge = 588 / 7) : 
  kid_charge = 3 := by
  sorry

end joses_swimming_pool_charge_l2695_269512


namespace coconut_flavored_jelly_beans_l2695_269500

theorem coconut_flavored_jelly_beans (total : ℕ) (red_fraction : ℚ) (coconut_fraction : ℚ) :
  total = 4000 →
  red_fraction = 3 / 4 →
  coconut_fraction = 1 / 4 →
  (total * red_fraction * coconut_fraction : ℚ) = 750 := by
  sorry

end coconut_flavored_jelly_beans_l2695_269500


namespace expression_evaluation_l2695_269565

theorem expression_evaluation : 
  let x : ℤ := -2
  let expr := (x^2 - 4*x + 4) / (x^2 - 1) / ((x^2 - 2*x) / (x + 1)) + 1 / (x - 1)
  expr = -1 := by sorry

end expression_evaluation_l2695_269565


namespace average_of_ABC_l2695_269567

theorem average_of_ABC (A B C : ℚ) 
  (eq1 : 2023 * C - 4046 * A = 8092)
  (eq2 : 2023 * B - 6069 * A = 10115) :
  (A + B + C) / 3 = 2 * A + 3 := by
  sorry

end average_of_ABC_l2695_269567


namespace level_passing_game_l2695_269508

/-- A fair six-sided die -/
def Die := Finset.range 6

/-- The number of times the die is rolled at level n -/
def rolls (n : ℕ) : ℕ := n

/-- The condition for passing a level -/
def pass_level (n : ℕ) (sum : ℕ) : Prop := sum > 2^n

/-- The maximum number of levels that can be passed -/
def max_levels : ℕ := 4

/-- The probability of passing the first three levels consecutively -/
def prob_pass_three : ℚ := 100 / 243

theorem level_passing_game :
  (∀ n : ℕ, n > max_levels → ¬∃ sum : ℕ, sum ≤ 6 * rolls n ∧ pass_level n sum) ∧
  (∃ sum : ℕ, sum ≤ 6 * rolls max_levels ∧ pass_level max_levels sum) ∧
  prob_pass_three = (2/3) * (5/6) * (20/27) :=
sorry

end level_passing_game_l2695_269508


namespace toy_cost_correct_l2695_269538

/-- The cost of the assortment box of toys for Julia's new puppy -/
def toy_cost : ℝ := 40

/-- The adoption fee for the puppy -/
def adoption_fee : ℝ := 20

/-- The cost of dog food -/
def dog_food_cost : ℝ := 20

/-- The cost of one bag of treats -/
def treat_cost : ℝ := 2.5

/-- The number of treat bags purchased -/
def treat_bags : ℕ := 2

/-- The cost of the crate -/
def crate_cost : ℝ := 20

/-- The cost of the bed -/
def bed_cost : ℝ := 20

/-- The cost of the collar/leash combo -/
def collar_leash_cost : ℝ := 15

/-- The discount percentage as a decimal -/
def discount_rate : ℝ := 0.2

/-- The total amount Julia spent on the puppy -/
def total_spent : ℝ := 96

theorem toy_cost_correct : 
  (1 - discount_rate) * (adoption_fee + dog_food_cost + treat_cost * treat_bags + 
  crate_cost + bed_cost + collar_leash_cost + toy_cost) = total_spent := by
  sorry

end toy_cost_correct_l2695_269538


namespace total_campers_rowing_l2695_269573

theorem total_campers_rowing (morning_campers afternoon_campers : ℕ) 
  (h1 : morning_campers = 53)
  (h2 : afternoon_campers = 7) :
  morning_campers + afternoon_campers = 60 := by
  sorry

end total_campers_rowing_l2695_269573


namespace red_shirt_pairs_l2695_269559

theorem red_shirt_pairs (total_students : ℕ) (blue_students : ℕ) (red_students : ℕ)
  (total_pairs : ℕ) (blue_blue_pairs : ℕ) :
  total_students = 144 →
  blue_students = 63 →
  red_students = 81 →
  total_pairs = 72 →
  blue_blue_pairs = 21 →
  total_students = blue_students + red_students →
  ∃ (red_red_pairs : ℕ), red_red_pairs = 30 ∧
    red_red_pairs + blue_blue_pairs + (blue_students - 2 * blue_blue_pairs) = total_pairs :=
by sorry

end red_shirt_pairs_l2695_269559


namespace fraction_power_simplification_l2695_269535

theorem fraction_power_simplification :
  (66666 : ℕ) = 3 * 22222 →
  (66666 : ℚ)^4 / (22222 : ℚ)^4 = 81 :=
by
  sorry

end fraction_power_simplification_l2695_269535


namespace alloy_mixture_specific_alloy_mixture_l2695_269541

/-- Given two alloys with different chromium percentages, prove the amount of the second alloy
    needed to create a new alloy with a specific chromium percentage. -/
theorem alloy_mixture (first_alloy_chromium_percent : ℝ) 
                      (second_alloy_chromium_percent : ℝ)
                      (new_alloy_chromium_percent : ℝ)
                      (first_alloy_amount : ℝ) : ℝ :=
  let second_alloy_amount := 
    (new_alloy_chromium_percent * first_alloy_amount - first_alloy_chromium_percent * first_alloy_amount) /
    (second_alloy_chromium_percent - new_alloy_chromium_percent)
  second_alloy_amount

/-- Prove that 35 kg of the second alloy is needed to create the new alloy with 8.6% chromium. -/
theorem specific_alloy_mixture : 
  alloy_mixture 0.10 0.08 0.086 15 = 35 := by
  sorry

end alloy_mixture_specific_alloy_mixture_l2695_269541


namespace product_103_97_l2695_269571

theorem product_103_97 : 103 * 97 = 9991 := by
  sorry

end product_103_97_l2695_269571


namespace quadratic_polynomial_proof_l2695_269594

theorem quadratic_polynomial_proof (b c x₁ x₂ : ℝ) : 
  (∃ (a : ℝ), a ≠ 0 ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₁ ≠ x₂) →
  (b + c + x₁ + x₂ = -3) →
  (b * c * x₁ * x₂ = 36) →
  (b = 4 ∧ c = -3) :=
by sorry

end quadratic_polynomial_proof_l2695_269594


namespace inscribed_octahedron_side_length_l2695_269514

-- Define the rectangular prism
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

-- Define the octahedron
structure Octahedron where
  sideLength : ℝ

-- Define the function to calculate the side length of the inscribed octahedron
def inscribedOctahedronSideLength (prism : RectangularPrism) : ℝ :=
  sorry

-- Theorem statement
theorem inscribed_octahedron_side_length 
  (prism : RectangularPrism) 
  (h1 : prism.length = 2) 
  (h2 : prism.width = 3) 
  (h3 : prism.height = 1) :
  inscribedOctahedronSideLength prism = Real.sqrt 14 / 2 :=
by sorry

end inscribed_octahedron_side_length_l2695_269514


namespace math_physics_majors_consecutive_probability_l2695_269568

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def choose (n k : ℕ) : ℕ := 
  factorial n / (factorial k * factorial (n - k))

theorem math_physics_majors_consecutive_probability :
  let total_people : ℕ := 12
  let math_majors : ℕ := 5
  let physics_majors : ℕ := 4
  let biology_majors : ℕ := 3
  let favorable_outcomes : ℕ := choose total_people math_majors * factorial (math_majors - 1) * 
                                 choose (total_people - math_majors) physics_majors * 
                                 factorial (physics_majors - 1) * factorial biology_majors
  let total_outcomes : ℕ := factorial (total_people - 1)
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 10 := by
  sorry

end math_physics_majors_consecutive_probability_l2695_269568


namespace ellipse_condition_l2695_269592

/-- An ellipse equation with parameter k -/
def is_ellipse (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (9 - k) + y^2 / (k - 4) = 1

/-- The condition 4 < k < 9 -/
def condition (k : ℝ) : Prop := 4 < k ∧ k < 9

/-- The statement to be proven -/
theorem ellipse_condition :
  (∀ k, is_ellipse k → condition k) ∧
  ¬(∀ k, condition k → is_ellipse k) :=
sorry

end ellipse_condition_l2695_269592


namespace evaluate_expression_l2695_269581

theorem evaluate_expression : -(16 / 4 * 8 - 70 + 4^2 * 7) = -74 := by
  sorry

end evaluate_expression_l2695_269581


namespace arithmetic_mean_x_y_l2695_269532

/-- Given two real numbers x and y satisfying certain conditions, 
    prove that their arithmetic mean is 3/4 -/
theorem arithmetic_mean_x_y (x y : ℝ) 
  (h1 : x * y > 0)
  (h2 : 2 * x * (1/2) + 1 * (-1/(2*y)) = 0)  -- Perpendicularity condition
  (h3 : y / x = 2 / y)  -- Geometric sequence condition
  : (x + y) / 2 = 3/4 := by
  sorry

end arithmetic_mean_x_y_l2695_269532


namespace owen_profit_l2695_269554

/-- Calculates the profit from selling face masks given the following conditions:
  * Number of boxes bought
  * Cost per box
  * Number of masks per box
  * Number of boxes repacked
  * Number of large packets sold
  * Price of large packets
  * Number of masks in large packets
  * Price of small baggies
  * Number of masks in small baggies
-/
def calculate_profit (
  boxes_bought : ℕ
  ) (cost_per_box : ℚ
  ) (masks_per_box : ℕ
  ) (boxes_repacked : ℕ
  ) (large_packets_sold : ℕ
  ) (large_packet_price : ℚ
  ) (masks_per_large_packet : ℕ
  ) (small_baggie_price : ℚ
  ) (masks_per_small_baggie : ℕ
  ) : ℚ :=
  let total_cost := boxes_bought * cost_per_box
  let total_masks := boxes_bought * masks_per_box
  let repacked_masks := boxes_repacked * masks_per_box
  let large_packet_revenue := large_packets_sold * large_packet_price
  let remaining_masks := total_masks - (large_packets_sold * masks_per_large_packet)
  let small_baggies := remaining_masks / masks_per_small_baggie
  let small_baggie_revenue := small_baggies * small_baggie_price
  let total_revenue := large_packet_revenue + small_baggie_revenue
  total_revenue - total_cost

theorem owen_profit :
  calculate_profit 12 9 50 6 3 12 100 3 10 = 18 := by
  sorry

end owen_profit_l2695_269554


namespace expression_equals_one_l2695_269556

theorem expression_equals_one (x : ℝ) : 
  ((((x + 1)^2 * (x^2 - x + 1)^2) / (x^3 + 1)^2)^2) * 
  ((((x - 1)^2 * (x^2 + x + 1)^2) / (x^3 - 1)^2)^2) = 1 :=
by sorry

end expression_equals_one_l2695_269556


namespace long_jump_records_correct_l2695_269561

/-- Represents a long jump record -/
structure LongJumpRecord where
  height : Real
  record : Real

/-- Checks if a long jump record is correctly calculated and recorded -/
def is_correct_record (standard : Real) (jump : LongJumpRecord) : Prop :=
  jump.record = jump.height - standard

/-- The problem statement -/
theorem long_jump_records_correct (standard : Real) (xiao_ming : LongJumpRecord) (xiao_liang : LongJumpRecord)
  (h1 : standard = 1.5)
  (h2 : xiao_ming.height = 1.95)
  (h3 : xiao_ming.record = 0.45)
  (h4 : xiao_liang.height = 1.23)
  (h5 : xiao_liang.record = -0.23) :
  ¬(is_correct_record standard xiao_ming ∧ is_correct_record standard xiao_liang) :=
sorry

end long_jump_records_correct_l2695_269561


namespace quadratic_inequality_range_l2695_269517

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := (a - 2) * x^2 - 2 * (a - 2) * x - 4

-- State the theorem
theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, f a x < 0) ↔ a ∈ Set.Ioc (-2) 2 :=
sorry

end quadratic_inequality_range_l2695_269517


namespace sticker_distribution_l2695_269536

/-- The number of ways to distribute n identical objects into k distinct containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- There are 8 identical stickers -/
def num_stickers : ℕ := 8

/-- There are 4 sheets of paper -/
def num_sheets : ℕ := 4

theorem sticker_distribution :
  distribute num_stickers num_sheets = 15 :=
sorry

end sticker_distribution_l2695_269536


namespace vector_sum_in_R2_l2695_269505

/-- Given two vectors in R², prove their sum is correct -/
theorem vector_sum_in_R2 (a b : Fin 2 → ℝ) (ha : a = ![5, 2]) (hb : b = ![1, 6]) :
  a + b = ![6, 8] := by
  sorry

end vector_sum_in_R2_l2695_269505


namespace min_modulus_complex_l2695_269582

theorem min_modulus_complex (z : ℂ) : 
  (∃ x : ℝ, x^2 - 2*z*x + (3/4 : ℂ) + Complex.I = 0) → Complex.abs z ≥ 1 ∧ ∃ z₀ : ℂ, Complex.abs z₀ = 1 ∧ 
  (∃ x : ℝ, x^2 - 2*z₀*x + (3/4 : ℂ) + Complex.I = 0) := by
sorry

end min_modulus_complex_l2695_269582


namespace man_rowing_speed_l2695_269503

/-- 
Given a man's rowing speed against the stream and his speed in still water,
calculate his speed with the stream.
-/
theorem man_rowing_speed 
  (speed_against_stream : ℝ) 
  (speed_still_water : ℝ) 
  (h1 : speed_against_stream = 4) 
  (h2 : speed_still_water = 6) : 
  speed_still_water + (speed_still_water - speed_against_stream) = 8 := by
  sorry

#check man_rowing_speed

end man_rowing_speed_l2695_269503


namespace specific_female_selection_probability_l2695_269510

def total_students : ℕ := 50
def male_students : ℕ := 30
def selected_students : ℕ := 5

theorem specific_female_selection_probability :
  (selected_students : ℚ) / total_students = 1 / 10 :=
by sorry

end specific_female_selection_probability_l2695_269510


namespace money_distribution_l2695_269540

theorem money_distribution (P Q R S : ℕ) : 
  P = 2 * Q →  -- P gets twice as that of Q
  S = 4 * R →  -- S gets 4 times as that of R
  Q = R →      -- Q and R are to receive equal amounts
  S - P = 250 →  -- The difference between S and P is 250
  P + Q + R + S = 1000 :=  -- Total amount to be distributed
by sorry

end money_distribution_l2695_269540


namespace boys_who_left_l2695_269598

theorem boys_who_left (initial_boys : ℕ) (initial_girls : ℕ) (additional_girls : ℕ) (final_total : ℕ) : 
  initial_boys = 5 →
  initial_girls = 4 →
  additional_girls = 2 →
  final_total = 8 →
  initial_boys - (final_total - (initial_girls + additional_girls)) = 3 :=
by sorry

end boys_who_left_l2695_269598


namespace max_cross_section_area_correct_l2695_269549

noncomputable def max_cross_section_area (k : ℝ) (α : ℝ) : ℝ :=
  if Real.tan α < 2 then
    (1/2) * k^2 * (1 + 3 * Real.cos α ^ 2)
  else
    2 * k^2 * Real.sin (2 * α)

theorem max_cross_section_area_correct (k : ℝ) (α : ℝ) (h1 : k > 0) (h2 : 0 < α ∧ α < π/2) :
  ∀ A : ℝ, A ≤ max_cross_section_area k α := by
  sorry

end max_cross_section_area_correct_l2695_269549


namespace max_food_per_guest_l2695_269534

theorem max_food_per_guest (total_food : ℕ) (min_guests : ℕ) (max_food : ℕ) : 
  total_food = 406 → 
  min_guests = 163 → 
  max_food = 2 → 
  (total_food : ℚ) / min_guests ≤ max_food :=
by sorry

end max_food_per_guest_l2695_269534
