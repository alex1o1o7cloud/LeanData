import Mathlib

namespace visitors_equal_cats_l213_213215

-- Definition for conditions
def visitors_pets_cats (V C : ℕ) : Prop :=
  (∃ P : ℕ, P = 3 * V ∧ P = 3 * C)

-- Statement of the proof problem
theorem visitors_equal_cats {V C : ℕ}
  (h : visitors_pets_cats V C) : V = C :=
by sorry

end visitors_equal_cats_l213_213215


namespace carpenter_needs_80_woodblocks_l213_213486

-- Define the number of logs the carpenter currently has
def existing_logs : ℕ := 8

-- Define the number of woodblocks each log can produce
def woodblocks_per_log : ℕ := 5

-- Define the number of additional logs needed
def additional_logs : ℕ := 8

-- Calculate the total number of woodblocks needed
def total_woodblocks_needed : ℕ := 
  (existing_logs * woodblocks_per_log) + (additional_logs * woodblocks_per_log)

-- Prove that the total number of woodblocks needed is 80
theorem carpenter_needs_80_woodblocks : total_woodblocks_needed = 80 := by
  sorry

end carpenter_needs_80_woodblocks_l213_213486


namespace students_difference_l213_213979

theorem students_difference 
  (C : ℕ → ℕ) 
  (hC1 : C 1 = 24) 
  (hC2 : ∀ n, C n.succ = C n - d)
  (h_total : C 1 + C 2 + C 3 + C 4 + C 5 = 100) :
  d = 2 :=
by sorry

end students_difference_l213_213979


namespace weave_mats_l213_213048

theorem weave_mats (m n p q : ℕ) (h1 : m * n = p * q) (h2 : ∀ k, k = n → n * 2 = k * 2) :
  (8 * 2 = 16) :=
by
  -- This is where we would traditionally include the proof steps.
  sorry

end weave_mats_l213_213048


namespace triangle_interior_angle_ge_60_l213_213584

theorem triangle_interior_angle_ge_60 (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A < 60) (h3 : B < 60) (h4 : C < 60) : false := 
by
  sorry

end triangle_interior_angle_ge_60_l213_213584


namespace circle_outside_hexagon_area_l213_213050

theorem circle_outside_hexagon_area :
  let r := (Real.sqrt 2) / 2
  let s := 1
  let area_circle := π * r^2
  let area_hexagon := 3 * Real.sqrt 3 / 2 * s^2
  area_circle - area_hexagon = (π / 2) - (3 * Real.sqrt 3 / 2) :=
by
  sorry

end circle_outside_hexagon_area_l213_213050


namespace decreasing_interval_monotonic_find_minimum_a_l213_213102

open Real

noncomputable def f (x : ℝ) : ℝ := log x - x^2 + x

theorem decreasing_interval_monotonic (x : ℝ) (hx : 0 < x) :
  (derivative f x < 0) → (1 ≤ x) := 
sorry

theorem find_minimum_a (a : ℝ) : (∀ x, 0 < x → f(x) ≤ (a / 2 - 1) * x^2 + a * x - 1) → (2 ≤ a) :=
sorry

end decreasing_interval_monotonic_find_minimum_a_l213_213102


namespace find_r_l213_213575

variable {x y r k : ℝ}

theorem find_r (h1 : y^2 + 4 * y + 4 + Real.sqrt (x + y + k) = 0)
               (h2 : r = |x * y|) :
    r = 2 :=
by
  sorry

end find_r_l213_213575


namespace matrix_solution_l213_213836

variable {x : ℝ}

theorem matrix_solution (x: ℝ) :
  let M := (3*x) * (2*x + 1) - (1) * (2*x)
  M = 5 → (x = 5/6) ∨ (x = -1) :=
by
  sorry

end matrix_solution_l213_213836


namespace heptagon_diagonals_l213_213113

-- Define the number of sides of the polygon
def heptagon_sides : ℕ := 7

-- Define the formula for the number of diagonals of an n-gon
def diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- State the theorem we want to prove, i.e., the number of diagonals in a convex heptagon is 14
theorem heptagon_diagonals : diagonals heptagon_sides = 14 := by
  sorry

end heptagon_diagonals_l213_213113


namespace alternate_interior_angles_equal_l213_213454

-- Defining the parallel lines and the third intersecting line
def Line : Type := sorry  -- placeholder type for a line

-- Predicate to check if lines are parallel
def parallel (l1 l2 : Line) : Prop := sorry

-- Predicate to represent a line intersecting another
def intersects (l1 l2 : Line) : Prop := sorry

-- Function to get interior alternate angles formed by the intersection
def alternate_interior_angles (l1 l2 : Line) (l3 : Line) : Prop := sorry

-- Theorem statement
theorem alternate_interior_angles_equal
  (l1 l2 l3 : Line)
  (h1 : parallel l1 l2)
  (h2 : intersects l3 l1)
  (h3 : intersects l3 l2) :
  alternate_interior_angles l1 l2 l3 :=
sorry

end alternate_interior_angles_equal_l213_213454


namespace round_to_hundredth_l213_213027

theorem round_to_hundredth:
  (∀ (x : Float), x ∈ {34.561, 34.558, 34.5601, 34.56444} → Float.round (x * 100) / 100 = 34.56) ∧ 
  (Float.round (34.5539999 * 100) / 100 ≠ 34.56) :=
by
  sorry

end round_to_hundredth_l213_213027


namespace jordon_machine_number_l213_213289

theorem jordon_machine_number : 
  ∃ x : ℝ, (2 * x + 3 = 27) ∧ x = 12 :=
by
  sorry

end jordon_machine_number_l213_213289


namespace train_pass_bridge_time_l213_213899

/-- A train is 460 meters long and runs at a speed of 45 km/h. The bridge is 140 meters long. 
Prove that the time it takes for the train to pass the bridge is 48 seconds. -/
theorem train_pass_bridge_time (train_length : ℝ) (bridge_length : ℝ) (speed_kmh : ℝ) 
  (h_train_length : train_length = 460) 
  (h_bridge_length : bridge_length = 140)
  (h_speed_kmh : speed_kmh = 45)
  : (train_length + bridge_length) / (speed_kmh * 1000 / 3600) = 48 := 
by
  sorry

end train_pass_bridge_time_l213_213899


namespace corresponding_angles_equal_l213_213590

-- Definition: Corresponding angles and their equality
def corresponding_angles (α β : ℝ) : Prop :=
  -- assuming definition of corresponding angles can be defined
  sorry

theorem corresponding_angles_equal {α β : ℝ} (h : corresponding_angles α β) : α = β :=
by
  -- the proof is provided in the problem statement
  sorry

end corresponding_angles_equal_l213_213590


namespace negation1_converse1_negation2_converse2_negation3_converse3_l213_213191

-- Definitions
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
def is_prime (p : ℕ) : Prop := nat.prime p

-- Proof statements
theorem negation1 : ¬ (∀ x y : ℤ, is_odd x → is_odd y → is_even (x + y)) ↔ true := sorry
theorem converse1 : ¬ (∀ x y : ℤ, ¬ (is_odd x ∧ is_odd y) → ¬ is_even (x + y)) ↔ true := sorry

theorem negation2 : (∀ x y : ℤ, x * y = 0 → ¬ (x = 0) ∧ ¬ (y = 0)) ↔ false := sorry
theorem converse2 : (∀ x y : ℤ, x * y ≠ 0 → x ≠ 0 ∧ y ≠ 0) ↔ true := sorry

theorem negation3 : (∀ p : ℕ, is_prime p → ¬ (is_odd p)) ↔ false := sorry
theorem converse3 : (∀ p : ℕ, ¬ is_prime p → ¬ (is_odd p)) ↔ false := sorry

end negation1_converse1_negation2_converse2_negation3_converse3_l213_213191


namespace triangle_area_l213_213428

theorem triangle_area (A B C : ℝ) (AB BC CA : ℝ) (sinA sinB sinC : ℝ)
    (h1 : sinA * sinB * sinC = 1 / 1000) 
    (h2 : AB * BC * CA = 1000) : 
    (AB * BC * CA / (4 * 50)) = 5 :=
by
  -- Proof is omitted
  sorry

end triangle_area_l213_213428


namespace number_of_distinct_intersections_of_curves_l213_213524

theorem number_of_distinct_intersections_of_curves (x y : ℝ) :
  (∀ x y, x^2 - 4*y^2 = 4) ∧ (∀ x y, 4*x^2 + y^2 = 16) → 
  ∃ (x1 x2 : ℝ) (y1 y2 : ℝ), 
    ((x1, y1) ≠ (x2, y2)) ∧
    ((x1^2 - 4*y1^2 = 4) ∧ (4*x1^2 + y1^2 = 16)) ∧
    ((x2^2 - 4*y2^2 = 4) ∧ (4*x2^2 + y2^2 = 16)) ∧
    ∀ (x' y' : ℝ), 
      ((x'^2 - 4*y'^2 = 4) ∧ (4*x'^2 + y'^2 = 16)) → 
      ((x', y') = (x1, y1) ∨ (x', y') = (x2, y2)) := 
sorry

end number_of_distinct_intersections_of_curves_l213_213524


namespace national_currency_depreciation_bond_annual_coupon_income_dividend_yield_tax_deduction_l213_213188

-- Question 5
theorem national_currency_depreciation (term : String) : term = "Devaluation" := 
sorry

-- Question 6
theorem bond_annual_coupon_income 
  (purchase_price face_value annual_yield annual_coupon : ℝ) 
  (h_price : purchase_price = 900)
  (h_face : face_value = 1000)
  (h_yield : annual_yield = 0.15) 
  (h_coupon : annual_coupon = 135) : 
  annual_coupon = annual_yield * purchase_price := 
sorry

-- Question 7
theorem dividend_yield 
  (num_shares price_per_share total_dividends dividend_yield : ℝ)
  (h_shares : num_shares = 1000000)
  (h_price : price_per_share = 400)
  (h_dividends : total_dividends = 60000000)
  (h_yield : dividend_yield = 15) : 
  dividend_yield = (total_dividends / num_shares / price_per_share) * 100 :=
sorry

-- Question 8
theorem tax_deduction 
  (insurance_premium annual_salary tax_return : ℝ)
  (h_premium : insurance_premium = 120000)
  (h_salary : annual_salary = 110000)
  (h_return : tax_return = 14300) : 
  tax_return = 0.13 * min insurance_premium annual_salary := 
sorry

end national_currency_depreciation_bond_annual_coupon_income_dividend_yield_tax_deduction_l213_213188


namespace grazing_months_l213_213900

theorem grazing_months :
  ∀ (m : ℕ),
  let a_oxen := 10
  let a_months := 7
  let b_oxen := 12
  let b_months := 5
  let c_oxen := 15
  let rent := 210
  let c_share := 54
  let a_ox_months := a_oxen * a_months
  let b_ox_months := b_oxen * b_months
  let c_ox_months := c_oxen * m
  let total_ox_months := a_ox_months + b_ox_months + c_ox_months
  let c_part := (c_ox_months : ℝ) / (total_ox_months : ℝ) * rent
  (c_part = c_share) → m = 3 :=
by { sorry }

end grazing_months_l213_213900


namespace plants_given_away_l213_213855

-- Define the conditions as constants
def initial_plants : ℕ := 3
def final_plants : ℕ := 20
def months : ℕ := 3

-- Function to calculate the number of plants after n months
def plants_after_months (initial: ℕ) (months: ℕ) : ℕ := initial * (2 ^ months)

-- The proof problem statement
theorem plants_given_away : (plants_after_months initial_plants months - final_plants) = 4 :=
by
  sorry

end plants_given_away_l213_213855


namespace milk_left_in_storage_l213_213887

-- Define initial and rate conditions
def initialMilk : ℕ := 30000
def pumpedRate : ℕ := 2880
def pumpedHours : ℕ := 4
def addedRate : ℕ := 1500
def addedHours : ℕ := 7

-- The proof problem: Prove the final amount in storage tank == 28980 gallons
theorem milk_left_in_storage : 
  initialMilk - (pumpedRate * pumpedHours) + (addedRate * addedHours) = 28980 := 
sorry

end milk_left_in_storage_l213_213887


namespace find_custom_operator_result_l213_213438

def custom_operator (a b : ℝ) : ℝ := 4 * a + 3 * b

theorem find_custom_operator_result :
  custom_operator 2 5 = 23 :=
by
  sorry

end find_custom_operator_result_l213_213438


namespace max_value_l213_213556

-- Definitions for the given conditions
def point_A := (3, 1)
def line_equation (m n : ℝ) := 3 * m + n + 1 = 0
def positive_product (m n : ℝ) := m * n > 0

-- The main statement to be proved
theorem max_value (m n : ℝ) (h1 : line_equation m n) (h2 : positive_product m n) : 
  (3 / m + 1 / n) ≤ -16 :=
sorry

end max_value_l213_213556


namespace brian_stones_l213_213504

variable (W B : ℕ)
variable (total_stones : ℕ := 100)
variable (G : ℕ := 40)
variable (Gr : ℕ := 60)

theorem brian_stones :
  (W > B) →
  ((W + B = total_stones) ∧ (G + Gr = total_stones) ∧ (W = 60)) :=
by
  sorry

end brian_stones_l213_213504


namespace sufficient_but_not_necessary_condition_l213_213805

theorem sufficient_but_not_necessary_condition (a b : ℝ) : (b ≥ 0 → a^2 + b ≥ 0) ∧ ¬(∀ a b, a^2 + b ≥ 0 → b ≥ 0) := by
  sorry

end sufficient_but_not_necessary_condition_l213_213805


namespace land_area_decreases_l213_213301

theorem land_area_decreases (a : ℕ) (h : a > 4) : (a * a) > ((a + 4) * (a - 4)) :=
by
  sorry

end land_area_decreases_l213_213301


namespace largest_number_with_digits_product_120_is_85311_l213_213795

-- Define the five-digit number
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

-- Define the product of digits
def digits_product (n : ℕ) : ℕ :=
  (n / 10000) * ((n / 1000) % 10) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

-- Define the condition that the product of the digits should be 120
def product_is_120 (n : ℕ) : Prop :=
  digits_product n = 120

-- Define the condition that n is the largest such number
def largest_such_number (n : ℕ) : Prop :=
  ∀ m : ℕ, is_five_digit m → product_is_120 m → m ≤ n

-- The theorem stating that the largest five-digit number whose digits' product equals 120 is 85311
theorem largest_number_with_digits_product_120_is_85311 : ∃ n : ℕ, is_five_digit n ∧ product_is_120 n ∧ largest_such_number n ∧ n = 85311 :=
by
  use 85311
  split
  -- Prove 85311 is a five-digit number
  sorry
  split
  -- Prove the product of the digits of 85311 is 120
  sorry
  split
  -- Prove that 85311 is the largest such number
  sorry
  -- Prove n = 85311
  sorry

end largest_number_with_digits_product_120_is_85311_l213_213795


namespace arithmetic_sequence_unique_a_l213_213824

theorem arithmetic_sequence_unique_a (a : ℝ) (b : ℕ → ℝ) (a_seq : ℕ → ℝ)
  (h1 : a_seq 1 = a) (h2 : a > 0)
  (h3 : b 1 - a_seq 1 = 1) (h4 : b 2 - a_seq 2 = 2)
  (h5 : b 3 - a_seq 3 = 3)
  (unique_a : ∀ (a' : ℝ), (a_seq 1 = a' ∧ a' > 0 ∧ b 1 - a' = 1 ∧ b 2 - a_seq 2 = 2 ∧ b 3 - a_seq 3 = 3) → a' = a) :
  a = 1 / 3 :=
by
  sorry

end arithmetic_sequence_unique_a_l213_213824


namespace relationship_y1_y2_y3_l213_213630

noncomputable def parabola_value (x m : ℝ) : ℝ := -x^2 - 4 * x + m

variable (m y1 y2 y3 : ℝ)

def point_A_on_parabola : Prop := y1 = parabola_value (-3) m
def point_B_on_parabola : Prop := y2 = parabola_value (-2) m
def point_C_on_parabola : Prop := y3 = parabola_value 1 m


theorem relationship_y1_y2_y3 (hA : point_A_on_parabola y1 m)
                              (hB : point_B_on_parabola y2 m)
                              (hC : point_C_on_parabola y3 m) :
  y2 > y1 ∧ y1 > y3 := 
  sorry

end relationship_y1_y2_y3_l213_213630


namespace trigonometric_identity_l213_213193

theorem trigonometric_identity (α : ℝ) :
  (2 * (Real.cos (2 * α))^2 + Real.sqrt 3 * Real.sin (4 * α) - 1) / 
  (2 * (Real.sin (2 * α))^2 + Real.sqrt 3 * Real.sin (4 * α) - 1) = 
  Real.sin (4 * α + Real.pi / 6) / Real.sin (4 * α - Real.pi / 6) :=
sorry

end trigonometric_identity_l213_213193


namespace molly_total_swim_l213_213579

variable (meters_saturday : ℕ) (meters_sunday : ℕ)

theorem molly_total_swim (h1 : meters_saturday = 45) (h2 : meters_sunday = 28) : meters_saturday + meters_sunday = 73 := by
  sorry

end molly_total_swim_l213_213579


namespace fg_of_3_is_83_l213_213554

def g (x : ℕ) : ℕ := x ^ 3
def f (x : ℕ) : ℕ := 3 * x + 2
theorem fg_of_3_is_83 : f (g 3) = 83 := by
  sorry

end fg_of_3_is_83_l213_213554


namespace sector_area_l213_213400

theorem sector_area (arc_length : ℝ) (central_angle : ℝ) (r : ℝ) (area : ℝ) : 
  arc_length = π / 3 ∧ central_angle = π / 6 → arc_length = central_angle * r → area = 1 / 2 * central_angle * r^2 → area = π / 3 :=
by
  sorry

end sector_area_l213_213400


namespace z_in_fourth_quadrant_l213_213097

noncomputable def z : ℂ := (3 * Complex.I - 2) / (Complex.I - 1) * Complex.I

theorem z_in_fourth_quadrant : z.re < 0 ∧ z.im > 0 := by
  sorry

end z_in_fourth_quadrant_l213_213097


namespace line_through_point_area_T_l213_213353

variable (a T : ℝ)

def equation_of_line (x y : ℝ) : Prop := 2 * T * x - a^2 * y + 2 * a * T = 0

theorem line_through_point_area_T :
  ∃ (x y : ℝ), equation_of_line a T x y ∧ x = -a ∧ y = (2 * T) / a :=
by
  sorry

end line_through_point_area_T_l213_213353


namespace minimum_value_expression_l213_213845

theorem minimum_value_expression {a b c : ℝ} (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) : 
  a^2 + 4 * a * b + 9 * b^2 + 3 * b * c + c^2 ≥ 18 :=
by
  sorry

end minimum_value_expression_l213_213845


namespace expression_value_l213_213253

theorem expression_value (m n a b x : ℤ) (h1 : m = -n) (h2 : a * b = 1) (h3 : |x| = 3) :
  x = 3 ∨ x = -3 → (x = 3 → x^3 - (1 + m + n - a * b) * x^2010 + (m + n) * x^2007 + (-a * b)^2009 = 26) ∧
                  (x = -3 → x^3 - (1 + m + n - a * b) * x^2010 + (m + n) * x^2007 + (-a * b)^2009 = -28) := by
  sorry

end expression_value_l213_213253


namespace probability_sum_is_five_l213_213474

theorem probability_sum_is_five (m n : ℕ) (h_m : 1 ≤ m ∧ m ≤ 6) (h_n : 1 ≤ n ∧ n ≤ 6)
  (h_total_outcomes : ∃(total_outcomes : ℕ), total_outcomes = 36)
  (h_favorable_outcomes : ∃(favorable_outcomes : ℕ), favorable_outcomes = 4) :
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 9 :=
sorry

end probability_sum_is_five_l213_213474


namespace bowls_per_minute_l213_213660

def ounces_per_bowl : ℕ := 10
def gallons_of_soup : ℕ := 6
def serving_time_minutes : ℕ := 15
def ounces_per_gallon : ℕ := 128

theorem bowls_per_minute :
  (gallons_of_soup * ounces_per_gallon / servings_time_minutes) / ounces_per_bowl = 5 :=
by
  sorry

end bowls_per_minute_l213_213660


namespace triangle_angles_correct_l213_213177

noncomputable def triangle_angles (a b c : ℝ) : (ℝ × ℝ × ℝ) :=
by sorry

theorem triangle_angles_correct :
  triangle_angles 3 (Real.sqrt 8) (2 + Real.sqrt 2) =
    (67.5, 22.5, 90) :=
by sorry

end triangle_angles_correct_l213_213177


namespace part1_part2_l213_213950

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem part1 (h : 1 - a = -1) : a = 2 ∧ 
                                  (∀ x : ℝ, x < Real.log 2 → (Real.exp x - 2) < 0) ∧ 
                                  (∀ x : ℝ, x > Real.log 2 → (Real.exp x - 2) > 0) :=
by
  sorry

theorem part2 (h1 : x1 < Real.log 2) (h2 : x2 > Real.log 2) (h3 : f 2 x1 = f 2 x2) : 
  x1 + x2 < 2 * Real.log 2 :=
by
  sorry

end part1_part2_l213_213950


namespace rationalize_sqrt_5_div_18_l213_213867

theorem rationalize_sqrt_5_div_18 :
  (Real.sqrt (5 / 18) = Real.sqrt 10 / 6) :=
sorry

end rationalize_sqrt_5_div_18_l213_213867


namespace cube_face_product_l213_213989

open Finset

theorem cube_face_product (numbers : Finset ℕ) (hs : numbers = range (12 + 1)) :
  ∃ top_face bottom_face : Finset ℕ,
    top_face.card = 4 ∧
    bottom_face.card = 4 ∧
    (numbers \ (top_face ∪ bottom_face)).card = 4 ∧
    (∏ x in top_face, x) = (∏ x in bottom_face, x) :=
by
  use {2, 4, 9, 10}
  use {3, 5, 6, 8}
  repeat { split };
  -- Check cardinality conditions
  sorry

end cube_face_product_l213_213989


namespace jane_started_babysitting_at_age_18_l213_213707

-- Define the age Jane started babysitting
def jane_starting_age := 18

-- State Jane's current age
def jane_current_age : ℕ := 34

-- State the years since Jane stopped babysitting
def years_since_jane_stopped := 12

-- Calculate Jane's age when she stopped babysitting
def jane_age_when_stopped : ℕ := jane_current_age - years_since_jane_stopped

-- State the current age of the oldest person she could have babysat
def current_oldest_child_age : ℕ := 25

-- Calculate the age of the oldest child when Jane stopped babysitting
def age_oldest_child_when_stopped : ℕ := current_oldest_child_age - years_since_jane_stopped

-- State the condition that the child was no more than half her age at the time
def child_age_condition (jane_age : ℕ) (child_age : ℕ) : Prop := child_age ≤ jane_age / 2

-- The theorem to prove the age Jane started babysitting
theorem jane_started_babysitting_at_age_18
  (jane_current : jane_current_age = 34)
  (years_stopped : years_since_jane_stopped = 12)
  (current_oldest : current_oldest_child_age = 25)
  (age_when_stopped : jane_age_when_stopped = 22)
  (child_when_stopped : age_oldest_child_when_stopped = 13)
  (child_condition : ∀ {j : ℕ}, child_age_condition j age_oldest_child_when_stopped → False) :
  jane_starting_age = 18 :=
sorry

end jane_started_babysitting_at_age_18_l213_213707


namespace product_is_58_l213_213017

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def p := 2
def q := 29

-- Conditions based on the problem
axiom prime_p : is_prime p
axiom prime_q : is_prime q
axiom sum_eq_31 : p + q = 31

-- Theorem to be proven
theorem product_is_58 : p * q = 58 :=
by
  sorry

end product_is_58_l213_213017


namespace sum_of_two_relatively_prime_integers_l213_213606

theorem sum_of_two_relatively_prime_integers (x y : ℕ) : 0 < x ∧ x < 30 ∧ 0 < y ∧ y < 30 ∧
  gcd x y = 1 ∧ x * y + x + y = 119 ∧ x + y = 20 :=
by
  sorry

end sum_of_two_relatively_prime_integers_l213_213606


namespace day_of_week_299th_day_2004_l213_213834

noncomputable def day_of_week (day: ℕ): ℕ := day % 7

theorem day_of_week_299th_day_2004 : 
  ∀ (d: ℕ), day_of_week d = 3 → d = 45 → day_of_week 299 = 5 :=
by
  sorry

end day_of_week_299th_day_2004_l213_213834


namespace obtuse_triangle_range_a_l213_213948

noncomputable def is_obtuse_triangle (a b c : ℝ) : Prop :=
  ∃ (θ : ℝ), θ > 90 ∧ θ ≤ 120 ∧ c^2 > a^2 + b^2

theorem obtuse_triangle_range_a (a : ℝ) :
  (a + (a + 1) > a + 2) →
  is_obtuse_triangle a (a + 1) (a + 2) →
  (1.5 ≤ a ∧ a < 3) :=
by
  sorry

end obtuse_triangle_range_a_l213_213948


namespace initial_volume_solution_l213_213351

variable (V : ℝ)

theorem initial_volume_solution
  (h1 : 0.35 * V + 1.8 = 0.50 * (V + 1.8)) :
  V = 6 :=
by
  sorry

end initial_volume_solution_l213_213351


namespace seed_germination_probability_l213_213002

-- Define necessary values and variables
def n : ℕ := 3
def p : ℚ := 0.7
def k : ℕ := 2

-- Define the binomial probability formula
def binomial_probability (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

-- State the proof problem
theorem seed_germination_probability :
  binomial_probability n k p = 0.441 := 
sorry

end seed_germination_probability_l213_213002


namespace circle_shaded_region_perimeter_l213_213981

theorem circle_shaded_region_perimeter
  (O P Q : Type) [MetricSpace O]
  (r : ℝ) (OP OQ : ℝ) (arc_PQ : ℝ)
  (hOP : OP = 8)
  (hOQ : OQ = 8)
  (h_arc_PQ : arc_PQ = 8 * Real.pi) :
  (OP + OQ + arc_PQ = 16 + 8 * Real.pi) :=
by
  sorry

end circle_shaded_region_perimeter_l213_213981


namespace carriages_per_train_l213_213181

variable (c : ℕ)

theorem carriages_per_train :
  (∃ c : ℕ, (25 + 10) * c * 3 = 420) → c = 4 :=
by
  sorry

end carriages_per_train_l213_213181


namespace roots_are_positive_integers_implies_r_values_l213_213699

theorem roots_are_positive_integers_implies_r_values (r x : ℕ) (h : (r * x^2 - (2 * r + 7) * x + (r + 7) = 0) ∧ (x > 0)) :
  r = 7 ∨ r = 0 ∨ r = 1 :=
by
  sorry

end roots_are_positive_integers_implies_r_values_l213_213699


namespace contrapositive_lemma_l213_213314

theorem contrapositive_lemma (a : ℝ) (h : a^2 ≤ 9) : a < 4 := 
sorry

end contrapositive_lemma_l213_213314


namespace recess_breaks_l213_213472

theorem recess_breaks (total_outside_time : ℕ) (lunch_break : ℕ) (extra_recess : ℕ) (recess_duration : ℕ) 
  (h1 : total_outside_time = 80)
  (h2 : lunch_break = 30)
  (h3 : extra_recess = 20)
  (h4 : recess_duration = 15) : 
  (total_outside_time - (lunch_break + extra_recess)) / recess_duration = 2 := 
by {
  -- proof starts here
  sorry
}

end recess_breaks_l213_213472


namespace time_to_finish_typing_l213_213142

-- Definitions
def words_per_minute : ℕ := 38
def total_words : ℕ := 4560

-- Theorem to prove
theorem time_to_finish_typing : (total_words / words_per_minute) / 60 = 2 := by
  sorry

end time_to_finish_typing_l213_213142


namespace trapezoid_ratio_of_bases_l213_213166

theorem trapezoid_ratio_of_bases (a b : ℝ) (h : a > b)
    (H : (1 / 4) * (1 / 2) * (a - b) * h = (1 / 8) * (a + b) * h) : 
    a / b = 3 := 
sorry

end trapezoid_ratio_of_bases_l213_213166


namespace number_of_students_in_Diligence_before_transfer_l213_213275

-- Define the total number of students and the transfer information
def total_students : ℕ := 50
def transferred_students : ℕ := 2

-- Define the number of students in Diligence before the transfer
def students_in_Diligence_before : ℕ := 23

-- Let's prove that the number of students in Diligence before the transfer is 23
theorem number_of_students_in_Diligence_before_transfer :
  (total_students / 2) - transferred_students = students_in_Diligence_before :=
by {
  -- The proof is omitted as instructed
  sorry
}

end number_of_students_in_Diligence_before_transfer_l213_213275


namespace sum_of_two_numbers_l213_213155

theorem sum_of_two_numbers (x y : ℕ) (h : x = 11) (h1 : y = 3 * x + 11) : x + y = 55 := by
  sorry

end sum_of_two_numbers_l213_213155


namespace max_hours_wednesday_l213_213857

theorem max_hours_wednesday (x : ℕ) 
    (h1 : ∀ (d w : ℕ), w = x → d = x → d + w + (x + 3) = 3 * 3) 
    (h2 : ∀ (a b c : ℕ), a = b → b = c → (a + b + (c + 3))/3 = 3) :
  x = 2 := 
by
  sorry

end max_hours_wednesday_l213_213857


namespace find_f_2012_l213_213322

variable (f : ℕ → ℝ)

axiom f_one : f 1 = 3997
axiom recurrence : ∀ x, f x - f (x + 1) = 1

theorem find_f_2012 : f 2012 = 1986 :=
by
  -- Skipping proof
  sorry

end find_f_2012_l213_213322


namespace proof_problem_l213_213941

open Real

noncomputable def p : Prop := ∃ x : ℝ, x - 2 > log x / log 10
noncomputable def q : Prop := ∀ x : ℝ, x^2 > 0

theorem proof_problem :
  (p ∧ ¬q) := by
  sorry

end proof_problem_l213_213941


namespace a_put_his_oxen_for_grazing_for_7_months_l213_213194

theorem a_put_his_oxen_for_grazing_for_7_months
  (x : ℕ)
  (a_oxen : ℕ := 10)
  (b_oxen : ℕ := 12)
  (b_months : ℕ := 5)
  (c_oxen : ℕ := 15)
  (c_months : ℕ := 3)
  (total_rent : ℝ := 105)
  (c_share : ℝ := 27) :
  (c_share / total_rent = (c_oxen * c_months) / ((a_oxen * x) + (b_oxen * b_months) + (c_oxen * c_months))) → (x = 7) :=
by
  sorry

end a_put_his_oxen_for_grazing_for_7_months_l213_213194


namespace y_explicit_and_range_l213_213091

theorem y_explicit_and_range (m : ℝ) (x1 x2 : ℝ) (h1 : x1^2 - 2*(m-1)*x1 + m + 1 = 0) (h2 : x2^2 - 2*(m-1)*x2 + m + 1 = 0) :
  x1 + x2 = 2*(m-1) ∧ x1 * x2 = m + 1 ∧ (x1^2 + x2^2 = 4*m^2 - 10*m + 2) 
  ∧ ∀ (y : ℝ), (∃ m, y = 4*m^2 - 10*m + 2) → y ≥ 6 :=
by
  sorry

end y_explicit_and_range_l213_213091


namespace estimate_sqrt_expression_l213_213073

theorem estimate_sqrt_expression :
  5 < 3 * Real.sqrt 5 - 1 ∧ 3 * Real.sqrt 5 - 1 < 6 :=
by
  sorry

end estimate_sqrt_expression_l213_213073


namespace value_of_f_at_1_l213_213818

def f (x : ℝ) : ℝ := x^2 + |x - 2|

theorem value_of_f_at_1 : f 1 = 2 :=
by sorry

end value_of_f_at_1_l213_213818


namespace system1_solution_system2_solution_l213_213307

theorem system1_solution :
  ∃ x y : ℝ, 3 * x + 4 * y = 16 ∧ 5 * x - 8 * y = 34 ∧ x = 6 ∧ y = -1/2 :=
by
  sorry

theorem system2_solution :
  ∃ x y : ℝ, (x - 1) / 2 + (y + 1) / 3 = 1 ∧ x + y = 4 ∧ x = -1 ∧ y = 5 :=
by
  sorry

end system1_solution_system2_solution_l213_213307


namespace floor_diff_l213_213236

theorem floor_diff {x : ℝ} (h : x = 12.7) : 
  (⌊x^2⌋ : ℤ) - (⌊x⌋ : ℤ) * (⌊x⌋ : ℤ) = 17 :=
by
  have h1 : x = 12.7 := h
  have hx2 : x^2 = 161.29 := by sorry
  have hfloor : ⌊x⌋ = 12 := by sorry
  have hfloor2 : ⌊x^2⌋ = 161 := by sorry
  sorry

end floor_diff_l213_213236


namespace fifth_term_arithmetic_seq_l213_213467

theorem fifth_term_arithmetic_seq (a d : ℤ) 
  (h10th : a + 9 * d = 23) 
  (h11th : a + 10 * d = 26) 
  : a + 4 * d = 8 :=
sorry

end fifth_term_arithmetic_seq_l213_213467


namespace charlie_extra_charge_l213_213039

-- Define the data plan and cost structure
def data_plan_limit : ℕ := 8  -- GB
def extra_cost_per_gb : ℕ := 10  -- $ per GB

-- Define Charlie's data usage over each week
def usage_week_1 : ℕ := 2  -- GB
def usage_week_2 : ℕ := 3  -- GB
def usage_week_3 : ℕ := 5  -- GB
def usage_week_4 : ℕ := 10  -- GB

-- Calculate the total data usage and the extra data used
def total_usage : ℕ := usage_week_1 + usage_week_2 + usage_week_3 + usage_week_4
def extra_usage : ℕ := if total_usage > data_plan_limit then total_usage - data_plan_limit else 0
def extra_charge : ℕ := extra_usage * extra_cost_per_gb

-- Theorem to prove the extra charge
theorem charlie_extra_charge : extra_charge = 120 := by
  -- Skipping the proof
  sorry

end charlie_extra_charge_l213_213039


namespace hotdogs_needed_l213_213723

theorem hotdogs_needed 
  (ella_hotdogs : ℕ) (emma_hotdogs : ℕ)
  (luke_multiple : ℕ) (hunter_multiple : ℚ)
  (h_ella : ella_hotdogs = 2)
  (h_emma : emma_hotdogs = 2)
  (h_luke : luke_multiple = 2)
  (h_hunter : hunter_multiple = (3/2)) :
  ella_hotdogs + emma_hotdogs + luke_multiple * (ella_hotdogs + emma_hotdogs) + hunter_multiple * (ella_hotdogs + emma_hotdogs) = 18 := by
    sorry

end hotdogs_needed_l213_213723


namespace project_scientists_total_l213_213581

def total_scientists (S : ℕ) : Prop :=
  S / 2 + S / 5 + 21 = S

theorem project_scientists_total : ∃ S, total_scientists S ∧ S = 70 :=
by
  existsi 70
  unfold total_scientists
  sorry

end project_scientists_total_l213_213581


namespace intersection_M_N_l213_213954

open Set

-- Definitions from conditions
def M : Set ℤ := {-1, 1, 2}
def N : Set ℤ := {x | x < 1}

-- Proof statement
theorem intersection_M_N : M ∩ N = {-1} := 
by sorry

end intersection_M_N_l213_213954


namespace find_nickels_l213_213130

noncomputable def num_quarters1 := 25
noncomputable def num_dimes := 15
noncomputable def num_quarters2 := 15
noncomputable def value_quarter := 25
noncomputable def value_dime := 10
noncomputable def value_nickel := 5

theorem find_nickels (n : ℕ) :
  value_quarter * num_quarters1 + value_dime * num_dimes = value_quarter * num_quarters2 + value_nickel * n → 
  n = 80 :=
by
  sorry

end find_nickels_l213_213130


namespace binary_operation_correct_l213_213368

theorem binary_operation_correct :
  let b1 := 0b11011
  let b2 := 0b1011
  let b3 := 0b11100
  let b4 := 0b10101
  let b5 := 0b1001
  b1 + b2 - b3 + b4 - b5 = 0b11110 := by
  sorry

end binary_operation_correct_l213_213368


namespace find_m_n_l213_213343

theorem find_m_n (m n : ℕ) (h : (1/5 : ℝ)^m * (1/4 : ℝ)^n = 1 / (10 : ℝ)^4) : m = 4 ∧ n = 2 :=
sorry

end find_m_n_l213_213343


namespace find_expression_for_a_n_l213_213497

noncomputable def a_sequence (a : ℕ → ℕ) :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = 2^n

theorem find_expression_for_a_n (a : ℕ → ℕ) (h : a_sequence a) (initial : a 1 = 1) :
  ∀ n : ℕ, a (n + 1) = 2^(n + 1) - 1 :=
by
  sorry

end find_expression_for_a_n_l213_213497


namespace xyz_inequality_l213_213992

theorem xyz_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z + x * y + y * z + z * x = 4) : x + y + z ≥ 3 := 
by
  sorry

end xyz_inequality_l213_213992


namespace negation_P_l213_213687

variable (P : Prop) (P_def : ∀ x : ℝ, Real.sin x ≤ 1)

theorem negation_P : ¬P ↔ ∃ x : ℝ, Real.sin x > 1 := by
  sorry

end negation_P_l213_213687


namespace change_in_f_when_x_increased_by_2_change_in_f_when_x_decreased_by_2_l213_213101

-- Given f(x) = x^2 - 5x
def f (x : ℝ) : ℝ := x^2 - 5 * x

-- Prove the change in f(x) when x is increased by 2 is 4x - 6
theorem change_in_f_when_x_increased_by_2 (x : ℝ) : f (x + 2) - f x = 4 * x - 6 := by
  sorry

-- Prove the change in f(x) when x is decreased by 2 is -4x + 14
theorem change_in_f_when_x_decreased_by_2 (x : ℝ) : f (x - 2) - f x = -4 * x + 14 := by
  sorry

end change_in_f_when_x_increased_by_2_change_in_f_when_x_decreased_by_2_l213_213101


namespace infinite_solutions_no_solutions_l213_213808

-- Define the geometric sequence with first term a1 = 1 and common ratio q
def a1 : ℝ := 1
def a2 (q : ℝ) : ℝ := a1 * q
def a3 (q : ℝ) : ℝ := a1 * q^2
def a4 (q : ℝ) : ℝ := a1 * q^3

-- Define the system of linear equations
def system_of_eqns (x y q : ℝ) : Prop :=
  a1 * x + a3 q * y = 3 ∧ a2 q * x + a4 q * y = -2

-- Conditions for infinitely many solutions
theorem infinite_solutions (q x y : ℝ) :
  q = -2 / 3 → ∃ x y, system_of_eqns x y q :=
by
  sorry

-- Conditions for no solutions
theorem no_solutions (q : ℝ) :
  q ≠ -2 / 3 → ¬∃ x y, system_of_eqns x y q :=
by
  sorry

end infinite_solutions_no_solutions_l213_213808


namespace find_z_l213_213411

theorem find_z (x y z : ℚ) (hx : x = 11) (hy : y = -8) (h : 2 * x - 3 * z = 5 * y) :
  z = 62 / 3 :=
by
  sorry

end find_z_l213_213411


namespace candies_per_packet_l213_213218

-- Define the given conditions
def monday_to_friday_candies_per_day := 2
def weekend_candies_per_day := 1
def weekdays := 5
def weekends := 2
def weeks := 3
def packets := 2

-- Calculate the number of candies Bobby eats in a week
def candies_per_week := (monday_to_friday_candies_per_day * weekdays) + (weekend_candies_per_day * weekends)

-- Calculate the total number of candies Bobby eats in the given 3 weeks
def total_candies_in_3_weeks := candies_per_week * weeks

-- Divide the total number of candies by the number of packets to find the candies per packet
theorem candies_per_packet : total_candies_in_3_weeks / packets = 18 := 
by
  -- Adding the proof placeholder
  sorry

end candies_per_packet_l213_213218


namespace unique_n_divides_2_pow_n_minus_1_l213_213664

theorem unique_n_divides_2_pow_n_minus_1 (n : ℕ) (h : n ∣ 2^n - 1) : n = 1 :=
sorry

end unique_n_divides_2_pow_n_minus_1_l213_213664


namespace rihanna_money_left_l213_213157

-- Definitions of the item costs
def cost_of_mangoes : ℝ := 6 * 3
def cost_of_apple_juice : ℝ := 4 * 3.50
def cost_of_potato_chips : ℝ := 2 * 2.25
def cost_of_chocolate_bars : ℝ := 3 * 1.75

-- Total cost computation
def total_cost : ℝ := cost_of_mangoes + cost_of_apple_juice + cost_of_potato_chips + cost_of_chocolate_bars

-- Initial amount of money Rihanna has
def initial_money : ℝ := 50

-- Remaining money after the purchases
def remaining_money : ℝ := initial_money - total_cost

-- The theorem stating that the remaining money is $8.25
theorem rihanna_money_left : remaining_money = 8.25 := by
  -- Lean will require the proof here.
  sorry

end rihanna_money_left_l213_213157


namespace expression_for_B_A_greater_than_B_l213_213690

-- Define the polynomials A and B
def A (x : ℝ) := 3 * x^2 - 2 * x + 1
def B (x : ℝ) := 2 * x^2 - x - 3

-- Prove that the given expression for B validates the equation A + B = 5x^2 - 4x - 2.
theorem expression_for_B (x : ℝ) : A x + 2 * x^2 - x - 3 = 5 * x^2 - 4 * x - 2 :=
by {
  sorry
}

-- Prove that A is always greater than B for all values of x.
theorem A_greater_than_B (x : ℝ) : A x > B x :=
by {
  sorry
}

end expression_for_B_A_greater_than_B_l213_213690


namespace trig_identity_l213_213510

theorem trig_identity :
  ∀ (θ : ℝ),
    θ = 70 * (π / 180) →
    (1 / Real.cos θ - (Real.sqrt 3) / Real.sin θ) = Real.sec (20 * (π / 180)) ^ 2 :=
by
  sorry

end trig_identity_l213_213510


namespace sin_75_eq_sqrt6_add_sqrt2_div4_l213_213179

theorem sin_75_eq_sqrt6_add_sqrt2_div4 :
  Real.sin (75 * Real.pi / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
sorry

end sin_75_eq_sqrt6_add_sqrt2_div4_l213_213179


namespace right_angle_triangle_iff_arithmetic_progression_l213_213728

noncomputable def exists_right_angle_triangle_with_rational_sides_and_area (d : ℤ) : Prop :=
  ∃ (a b c : ℚ), (a^2 + b^2 = c^2) ∧ (a * b = 2 * d)

noncomputable def rational_squares_in_arithmetic_progression (x y z : ℚ) : Prop :=
  2 * y^2 = x^2 + z^2

theorem right_angle_triangle_iff_arithmetic_progression (d : ℤ) :
  (∃ (a b c : ℚ), (a^2 + b^2 = c^2) ∧ (a * b = 2 * d)) ↔ ∃ (x y z : ℚ), rational_squares_in_arithmetic_progression x y z :=
sorry

end right_angle_triangle_iff_arithmetic_progression_l213_213728


namespace candies_per_packet_l213_213217

-- Define the given conditions
def monday_to_friday_candies_per_day := 2
def weekend_candies_per_day := 1
def weekdays := 5
def weekends := 2
def weeks := 3
def packets := 2

-- Calculate the number of candies Bobby eats in a week
def candies_per_week := (monday_to_friday_candies_per_day * weekdays) + (weekend_candies_per_day * weekends)

-- Calculate the total number of candies Bobby eats in the given 3 weeks
def total_candies_in_3_weeks := candies_per_week * weeks

-- Divide the total number of candies by the number of packets to find the candies per packet
theorem candies_per_packet : total_candies_in_3_weeks / packets = 18 := 
by
  -- Adding the proof placeholder
  sorry

end candies_per_packet_l213_213217


namespace rent_percentage_l213_213432

-- Define Elaine's earnings last year
def E : ℝ := sorry

-- Define last year's rent expenditure
def rentLastYear : ℝ := 0.20 * E

-- Define this year's earnings
def earningsThisYear : ℝ := 1.35 * E

-- Define this year's rent expenditure
def rentThisYear : ℝ := 0.30 * earningsThisYear

-- Prove the required percentage
theorem rent_percentage : ((rentThisYear / rentLastYear) * 100) = 202.5 := by
  sorry

end rent_percentage_l213_213432


namespace expression_evaluation_l213_213077

theorem expression_evaluation (a : ℝ) (h : a = 9) : ( (a ^ (1 / 3)) / (a ^ (1 / 5)) ) = a^(2 / 15) :=
by
  sorry

end expression_evaluation_l213_213077


namespace cyclic_sum_inequality_l213_213087

open Real

theorem cyclic_sum_inequality
  (a b c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / c + b^2 / a + c^2 / b) + (b^2 / c + c^2 / a + a^2 / b) + (c^2 / a + a^2 / b + b^2 / c) + 
  7 * (a + b + c) 
  ≥ ((a + b + c)^3) / (a * b + b * c + c * a) + (2 * (a * b + b * c + c * a)^2) / (a * b * c) := 
sorry

end cyclic_sum_inequality_l213_213087


namespace Bryce_grapes_l213_213408

theorem Bryce_grapes : 
  ∃ x : ℝ, (∀ y : ℝ, y = (1/3) * x → y = x - 7) → x = 21 / 2 :=
by
  sorry

end Bryce_grapes_l213_213408


namespace percentage_increase_l213_213287

theorem percentage_increase (original new : ℝ) (h_original : original = 50) (h_new : new = 75) : 
  (new - original) / original * 100 = 50 :=
by
  sorry

end percentage_increase_l213_213287


namespace bottles_produced_by_10_machines_in_4_minutes_l213_213303

variable (rate_per_machine : ℕ)
variable (total_bottles_per_minute_six_machines : ℕ := 240)
variable (number_of_machines : ℕ := 6)
variable (new_number_of_machines : ℕ := 10)
variable (time_in_minutes : ℕ := 4)

theorem bottles_produced_by_10_machines_in_4_minutes :
  rate_per_machine = total_bottles_per_minute_six_machines / number_of_machines →
  (new_number_of_machines * rate_per_machine * time_in_minutes) = 1600 := 
sorry

end bottles_produced_by_10_machines_in_4_minutes_l213_213303


namespace number_of_exchanges_l213_213031

theorem number_of_exchanges (n : ℕ) (hz_initial : ℕ) (hl_initial : ℕ) 
  (hz_decrease : ℕ) (hl_decrease : ℕ) (k : ℕ) :
  hz_initial = 200 →
  hl_initial = 20 →
  hz_decrease = 6 →
  hl_decrease = 1 →
  k = 11 →
  (hz_initial - n * hz_decrease) = k * (hl_initial - n * hl_decrease) →
  n = 4 := 
sorry

end number_of_exchanges_l213_213031


namespace acetone_mass_percentage_O_l213_213668

-- Definition of atomic masses
def atomic_mass_C := 12.01
def atomic_mass_H := 1.008
def atomic_mass_O := 16.00

-- Definition of the molar mass of acetone
def molar_mass_acetone := (3 * atomic_mass_C) + (6 * atomic_mass_H) + atomic_mass_O

-- Definition of mass percentage of oxygen in acetone
def mass_percentage_O_acetone := (atomic_mass_O / molar_mass_acetone) * 100

theorem acetone_mass_percentage_O : mass_percentage_O_acetone = 27.55 := by sorry

end acetone_mass_percentage_O_l213_213668


namespace students_in_diligence_before_transfer_l213_213273

theorem students_in_diligence_before_transfer (D I : ℕ) 
  (h1 : D + 2 = I - 2) 
  (h2 : D + I = 50) : 
  D = 23 := 
by
  sorry

end students_in_diligence_before_transfer_l213_213273


namespace find_particular_number_l213_213500

variable (x : ℝ)

theorem find_particular_number (h : 0.46 + x = 0.72) : x = 0.26 :=
sorry

end find_particular_number_l213_213500


namespace smallest_k_for_no_real_roots_l213_213325

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem smallest_k_for_no_real_roots :
  ∃ (k : ℤ), (∀ (x : ℝ), (x * x + 6 * x + 2 * k : ℝ) ≠ 0 ∧ k ≥ 5) :=
by
  sorry

end smallest_k_for_no_real_roots_l213_213325


namespace solve_abc_l213_213248

theorem solve_abc (a b c : ℕ) (h1 : a > b ∧ b > c) 
  (h2 : 34 - 6 * (a + b + c) + (a * b + b * c + c * a) = 0) 
  (h3 : 79 - 9 * (a + b + c) + (a * b + b * c + c * a) = 0) : 
  a = 10 ∧ b = 3 ∧ c = 2 := 
by
  sorry

end solve_abc_l213_213248


namespace parallel_lines_solution_l213_213825

theorem parallel_lines_solution (a : ℝ) :
  (∀ x y : ℝ, a = 0 → (x + 2 * a * y - 1 = 0 ∧ (2 * a - 1) * x - a * y - 1 = 0) → (x = y)) ∨ 
  (∀ x y : ℝ, a = 1/4 → (x + 2 * a * y - 1 = 0 ∧ (2 * a - 1) * x - a * y - 1 = 0) → (x = y)) :=
sorry

end parallel_lines_solution_l213_213825


namespace find_radius_of_C4_l213_213367

open Real

theorem find_radius_of_C4 (R1 R2 R3 R4 : ℝ) : 
  R1 = 360 → 
  R2 = 360 → 
  R3 = 90 → 
  ∃ R4, R4 = 40 :=
by intros h1 h2 h3
   use 40
   sorry

end find_radius_of_C4_l213_213367


namespace green_more_than_blue_l213_213375

-- Define the conditions
variables (B Y G n : ℕ)
def ratio_condition := 3 * n = B ∧ 7 * n = Y ∧ 8 * n = G
def total_disks_condition := B + Y + G = 72

-- State the theorem
theorem green_more_than_blue (B Y G n : ℕ) 
  (h_ratio : ratio_condition B Y G n) 
  (h_total : total_disks_condition B Y G) 
  : G - B = 20 := 
sorry

end green_more_than_blue_l213_213375


namespace count_divisors_36_l213_213115

def is_divisor (n d : Int) : Prop := d ≠ 0 ∧ ∃ k : Int, n = d * k

theorem count_divisors_36 : 
  (Finset.filter (λ d, is_divisor 36 d) (Finset.range 37)).card 
    + (Finset.filter (λ d, is_divisor 36 (-d)) (Finset.range 37)).card
  = 18 :=
sorry

end count_divisors_36_l213_213115


namespace custom_mul_expansion_l213_213801

variable {a b x y : ℝ}

def custom_mul (a b : ℝ) : ℝ := (a - b)^2

theorem custom_mul_expansion (x y : ℝ) : custom_mul (x^2) (y^2) = (x + y)^2 * (x - y)^2 := by
  sorry

end custom_mul_expansion_l213_213801


namespace sqrt_x_minus_3_undefined_l213_213944

theorem sqrt_x_minus_3_undefined (x : ℕ) (h_pos : x > 0) : 
  (x = 1 ∨ x = 2) ↔ real.sqrt (x - 3) = 0 := sorry

end sqrt_x_minus_3_undefined_l213_213944


namespace doubled_base_and_exponent_l213_213720

theorem doubled_base_and_exponent (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0)
  (h : (2 * a) ^ (2 * b) = a ^ b * x ^ 3) : 
  x = (4 ^ b * a ^ b) ^ (1 / 3) :=
by
  sorry

end doubled_base_and_exponent_l213_213720


namespace number_of_good_colorings_l213_213238

theorem number_of_good_colorings (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) : 
  ∃ (good_colorings : ℕ), good_colorings = 6 * (2^n - 4 + 4 * 2^(m-2)) :=
sorry

end number_of_good_colorings_l213_213238


namespace brenda_age_l213_213056

variables (A B J : ℝ)

-- Conditions
def condition1 : Prop := A = 4 * B
def condition2 : Prop := J = B + 7
def condition3 : Prop := A = J

-- Target to prove
theorem brenda_age (h1 : condition1 A B) (h2 : condition2 B J) (h3 : condition3 A J) : B = 7 / 3 :=
by
  sorry

end brenda_age_l213_213056


namespace exists_root_f_between_0_and_1_l213_213371

noncomputable def f (x : ℝ) : ℝ := 4 - 4 * x - Real.exp x

theorem exists_root_f_between_0_and_1 :
  ∃ x ∈ Set.Ioo 0 1, f x = 0 :=
sorry

end exists_root_f_between_0_and_1_l213_213371


namespace oliver_baths_per_week_l213_213725

-- Define all the conditions given in the problem
def bucket_capacity : ℕ := 120
def num_buckets_to_fill_tub : ℕ := 14
def num_buckets_removed : ℕ := 3
def weekly_water_usage : ℕ := 9240

-- Calculate total water to fill bathtub, water removed, water used per bath, and baths per week
def total_tub_capacity : ℕ := num_buckets_to_fill_tub * bucket_capacity
def water_removed : ℕ := num_buckets_removed * bucket_capacity
def water_per_bath : ℕ := total_tub_capacity - water_removed
def baths_per_week : ℕ := weekly_water_usage / water_per_bath

theorem oliver_baths_per_week : baths_per_week = 7 := by
  sorry

end oliver_baths_per_week_l213_213725


namespace gcd_polynomials_l213_213943

theorem gcd_polynomials (b : ℤ) (h : b % 8213 = 0 ∧ b % 2 = 1) :
  Int.gcd (8 * b^2 + 63 * b + 144) (2 * b + 15) = 9 :=
sorry

end gcd_polynomials_l213_213943


namespace trip_total_time_trip_average_speed_l213_213204

structure Segment where
  distance : ℝ -- in kilometers
  speed : ℝ -- average speed in km/hr
  break_time : ℝ -- in minutes

def seg1 := Segment.mk 12 13 15
def seg2 := Segment.mk 18 16 30
def seg3 := Segment.mk 25 20 45
def seg4 := Segment.mk 35 25 60
def seg5 := Segment.mk 50 22 0

noncomputable def total_time_minutes (segs : List Segment) : ℝ :=
  segs.foldl (λ acc s => acc + (s.distance / s.speed) * 60 + s.break_time) 0

noncomputable def total_distance (segs : List Segment) : ℝ :=
  segs.foldl (λ acc s => acc + s.distance) 0

noncomputable def overall_average_speed (segs : List Segment) : ℝ :=
  total_distance segs / (total_time_minutes segs / 60)

def segments := [seg1, seg2, seg3, seg4, seg5]

theorem trip_total_time : total_time_minutes segments = 568.24 := by sorry
theorem trip_average_speed : overall_average_speed segments = 14.78 := by sorry

end trip_total_time_trip_average_speed_l213_213204


namespace quadratic_inequality_solution_l213_213934

-- Given a quadratic inequality, prove the solution set in interval notation.
theorem quadratic_inequality_solution (x : ℝ) : 
  3 * x ^ 2 + 9 * x + 6 ≤ 0 ↔ -2 ≤ x ∧ x ≤ -1 :=
sorry

end quadratic_inequality_solution_l213_213934


namespace time_for_completion_l213_213626

noncomputable def efficiency_b : ℕ := 100

noncomputable def efficiency_a := 130

noncomputable def total_work := efficiency_a * 23

noncomputable def combined_efficiency := efficiency_a + efficiency_b

noncomputable def time_taken := total_work / combined_efficiency

theorem time_for_completion (h1 : efficiency_a = 130)
                           (h2 : efficiency_b = 100)
                           (h3 : total_work = 2990)
                           (h4 : combined_efficiency = 230) :
  time_taken = 13 := by
  sorry

end time_for_completion_l213_213626


namespace even_function_zeros_l213_213151

noncomputable def f (x m : ℝ) : ℝ := (x - 1) * (x + m)

theorem even_function_zeros (m : ℝ) (h : ∀ x : ℝ, f x m = f (-x) m ) : 
  m = 1 ∧ (∀ x : ℝ, f x m = 0 → (x = 1 ∨ x = -1)) := by
  sorry

end even_function_zeros_l213_213151


namespace trig_expression_simplification_l213_213873

theorem trig_expression_simplification :
  ∃ a b : ℕ, 
  0 < b ∧ b < 90 ∧ 
  (1000 * Real.sin (10 * Real.pi / 180) * Real.cos (20 * Real.pi / 180) * Real.cos (30 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) = ↑a * Real.sin (b * Real.pi / 180)) ∧ 
  (100 * a + b = 12560) :=
sorry

end trig_expression_simplification_l213_213873


namespace arithmetic_progression_25th_term_l213_213384

theorem arithmetic_progression_25th_term (a1 d : ℤ) (n : ℕ) (h_a1 : a1 = 5) (h_d : d = 7) (h_n : n = 25) :
  a1 + (n - 1) * d = 173 :=
by
  sorry

end arithmetic_progression_25th_term_l213_213384


namespace round_nearest_hundredth_problem_l213_213026

noncomputable def round_nearest_hundredth (x : ℚ) : ℚ :=
  let shifted := x * 100
  let rounded := if (shifted - shifted.floor) < 0.5 then shifted.floor else shifted.ceil
  rounded / 100

theorem round_nearest_hundredth_problem :
  let A := 34.561
  let B := 34.558
  let C := 34.5539999
  let D := 34.5601
  let E := 34.56444
  round_nearest_hundredth A = 34.56 ∧
  round_nearest_hundredth B = 34.56 ∧
  round_nearest_hundredth C ≠ 34.56 ∧
  round_nearest_hundredth D = 34.56 ∧
  round_nearest_hundredth E = 34.56 :=
sorry

end round_nearest_hundredth_problem_l213_213026


namespace greatest_y_value_l213_213734

theorem greatest_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -6) : y ≤ 3 :=
sorry

end greatest_y_value_l213_213734


namespace exists_multiple_of_prime_with_all_nines_digits_l213_213865

theorem exists_multiple_of_prime_with_all_nines_digits (p : ℕ) (hp_prime : Nat.Prime p) (h2 : p ≠ 2) (h5 : p ≠ 5) :
  ∃ n : ℕ, (∀ d ∈ (n.digits 10), d = 9) ∧ p ∣ n :=
by
  sorry

end exists_multiple_of_prime_with_all_nines_digits_l213_213865


namespace fans_per_bleacher_l213_213299

theorem fans_per_bleacher 
  (total_fans : ℕ) 
  (sets_of_bleachers : ℕ) 
  (h_total : total_fans = 2436) 
  (h_sets : sets_of_bleachers = 3) : 
  total_fans / sets_of_bleachers = 812 := 
by 
  sorry

end fans_per_bleacher_l213_213299


namespace trigonometric_identity_l213_213514

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180))
  = (4 * Real.sin (10 * Real.pi / 180) / Real.sin (40 * Real.pi / 180)) :=
by sorry

end trigonometric_identity_l213_213514


namespace find_weight_B_l213_213198

-- Define the weights of A, B, and C
variables (A B C : ℝ)

-- Conditions
def avg_weight_ABC := A + B + C = 135
def avg_weight_AB := A + B = 80
def avg_weight_BC := B + C = 86

-- The statement to be proved
theorem find_weight_B (h1: avg_weight_ABC A B C) (h2: avg_weight_AB A B) (h3: avg_weight_BC B C) : B = 31 :=
sorry

end find_weight_B_l213_213198


namespace candies_per_packet_l213_213216

-- Define the given conditions
def monday_to_friday_candies_per_day := 2
def weekend_candies_per_day := 1
def weekdays := 5
def weekends := 2
def weeks := 3
def packets := 2

-- Calculate the number of candies Bobby eats in a week
def candies_per_week := (monday_to_friday_candies_per_day * weekdays) + (weekend_candies_per_day * weekends)

-- Calculate the total number of candies Bobby eats in the given 3 weeks
def total_candies_in_3_weeks := candies_per_week * weeks

-- Divide the total number of candies by the number of packets to find the candies per packet
theorem candies_per_packet : total_candies_in_3_weeks / packets = 18 := 
by
  -- Adding the proof placeholder
  sorry

end candies_per_packet_l213_213216


namespace candy_per_packet_l213_213219

-- Define the conditions as hypotheses
def bobby_weekly_candies (mon_to_fri_candies : ℕ) (sat_sun_candies : ℕ) : ℕ :=
  mon_to_fri_candies + sat_sun_candies

def total_candies_in_n_weeks (weekly_candies : ℕ) (n : ℕ) : ℕ :=
  weekly_candies * n

theorem candy_per_packet
  (mon_to_fri_candies_per_day : ℕ)
  (sat_sun_candies_per_day : ℕ)
  (days_mon_to_fri : ℕ)
  (days_weekend : ℕ)
  (num_weeks : ℕ)
  (total_packets : ℕ)
  (candies_per_packet : ℕ)
  (h1 : mon_to_fri_candies_per_day = 2)
  (h2 : sat_sun_candies_per_day = 1)
  (h3 : days_mon_to_fri = 5)
  (h4 : days_weekend = 2)
  (h5 : num_weeks = 3)
  (h6 : total_packets = 2)
  (h7 : candies_per_packet = (total_candies_in_n_weeks (bobby_weekly_candies (mon_to_fri_candies_per_day * days_mon_to_fri) (sat_sun_candies_per_day * days_weekend)) num_weeks) / total_packets) :
  candies_per_packet = 18 :=
sorry

end candy_per_packet_l213_213219


namespace no_integer_a_exists_l213_213926

theorem no_integer_a_exists (a x : ℤ)
  (h : x^3 - a * x^2 - 6 * a * x + a^2 - 3 = 0)
  (unique_sol : ∀ y : ℤ, (y^3 - a * y^2 - 6 * a * y + a^2 - 3 = 0 → y = x)) :
  false :=
by 
  sorry

end no_integer_a_exists_l213_213926


namespace find_fraction_l213_213523

variable {N : ℕ}
variable {f : ℚ}

theorem find_fraction (h1 : N = 150) (h2 : N - f * N = 60) : f = 3/5 := by
  sorry

end find_fraction_l213_213523


namespace noah_total_bill_l213_213862

def call_duration := 30 -- in minutes
def charge_per_minute := 0.05 -- in dollars per minute
def calls_per_week := 1 -- calls per week
def weeks_per_year := 52 -- weeks per year

theorem noah_total_bill:
  (calls_per_week * weeks_per_year * call_duration * charge_per_minute) = 78 :=
by
  sorry

end noah_total_bill_l213_213862


namespace value_of_d_l213_213413

theorem value_of_d (d : ℝ) (h : ∀ y : ℝ, 3 * (5 + d * y) = 15 * y + 15) : d = 5 :=
by
  sorry

end value_of_d_l213_213413


namespace cube_inequality_l213_213844

theorem cube_inequality (a b : ℝ) : a > b ↔ a^3 > b^3 :=
sorry

end cube_inequality_l213_213844


namespace knocks_to_knicks_l213_213691

-- Define the conversion relationships as conditions
variable (knicks knacks knocks : ℝ)

-- 8 knicks = 3 knacks
axiom h1 : 8 * knicks = 3 * knacks

-- 5 knacks = 6 knocks
axiom h2 : 5 * knacks = 6 * knocks

-- Proving the equivalence of 30 knocks to 66.67 knicks
theorem knocks_to_knicks : 30 * knocks = 66.67 * knicks :=
by
  sorry

end knocks_to_knicks_l213_213691


namespace find_lisa_speed_l213_213152

theorem find_lisa_speed (Distance : ℕ) (Time : ℕ) (h1 : Distance = 256) (h2 : Time = 8) : Distance / Time = 32 := 
by {
  sorry
}

end find_lisa_speed_l213_213152


namespace complement_setP_in_U_l213_213405

def setU : Set ℝ := {x | -1 < x ∧ x < 3}
def setP : Set ℝ := {x | -1 < x ∧ x ≤ 2}

theorem complement_setP_in_U : (setU \ setP) = {x | 2 < x ∧ x < 3} :=
by
  sorry

end complement_setP_in_U_l213_213405


namespace inverse_variation_y_at_x_l213_213410

variable (k x y : ℝ)

theorem inverse_variation_y_at_x :
  (∀ x y k, y = k / x → y = 6 → x = 3 → k = 18) → 
  k = 18 →
  x = 12 →
  y = 18 / 12 →
  y = 3 / 2 := by
  intros h1 h2 h3 h4
  sorry

end inverse_variation_y_at_x_l213_213410


namespace no_solution_for_vectors_l213_213659

theorem no_solution_for_vectors {t s k : ℝ} :
  (∃ t s : ℝ, (1 + 6 * t = -1 + 3 * s) ∧ (3 + 1 * t = 4 + k * s)) ↔ k ≠ 0.5 :=
sorry

end no_solution_for_vectors_l213_213659


namespace solve_x_l213_213646

def otimes (a b : ℝ) : ℝ := a - 3 * b

theorem solve_x : ∃ x : ℝ, otimes x 1 + otimes 2 x = 1 ∧ x = -1 :=
by
  use -1
  rw [otimes, otimes]
  sorry

end solve_x_l213_213646


namespace approximation_accuracy_l213_213042

noncomputable def radius (k : Circle) : ℝ := sorry
def BG_equals_radius (BG : ℝ) (r : ℝ) := BG = r
def DB_equals_radius_sqrt3 (DB DG r : ℝ) := DB = DG ∧ DG = r * Real.sqrt 3
def cos_beta (cos_beta : ℝ) := cos_beta = 1 / (2 * Real.sqrt 3)
def sin_beta (sin_beta : ℝ) := sin_beta = Real.sqrt 11 / (2 * Real.sqrt 3)
def angle_BCH (angle_BCH : ℝ) (beta : ℝ) := angle_BCH = 120 - beta
def side_nonagon (a_9 r : ℝ) := a_9 = 2 * r * Real.sin 20
def bounds_sin_20 (sin_20 : ℝ) := 0.34195 < sin_20 ∧ sin_20 < 0.34205
def error_margin_low (BH_low a_9 r : ℝ) := 0.6839 * r < a_9
def error_margin_high (BH_high a_9 r : ℝ) := a_9 < 0.6841 * r

theorem approximation_accuracy
  (r : ℝ) (BG DB DG : ℝ) (beta : ℝ) (a_9 BH_low BH_high : ℝ)
  (h1 : BG_equals_radius BG r)
  (h2 : DB_equals_radius_sqrt3 DB DG r)
  (h3 : cos_beta (1 / (2 * Real.sqrt 3)))
  (h4 : sin_beta (Real.sqrt 11 / (2 * Real.sqrt 3)))
  (h5 : angle_BCH (120 - beta) beta)
  (h6 : side_nonagon a_9 r)
  (h7 : bounds_sin_20 (Real.sin 20))
  (h8 : error_margin_low BH_low a_9 r)
  (h9 : error_margin_high BH_high a_9 r) : 
  0.6861 * r < BH_high ∧ BH_low < 0.6864 * r :=
sorry

end approximation_accuracy_l213_213042


namespace ratio_is_one_to_five_l213_213752

def ratio_of_minutes_to_hour (twelve_minutes : ℕ) (one_hour : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd twelve_minutes one_hour
  (twelve_minutes / gcd, one_hour / gcd)

theorem ratio_is_one_to_five : ratio_of_minutes_to_hour 12 60 = (1, 5) := 
by 
  sorry

end ratio_is_one_to_five_l213_213752


namespace reciprocal_of_neg_two_l213_213464

theorem reciprocal_of_neg_two :
  (∃ x : ℝ, x = -2 ∧ 1 / x = -1 / 2) :=
by
  use -2
  split
  · rfl
  · norm_num

end reciprocal_of_neg_two_l213_213464


namespace continuous_sum_m_l213_213577

noncomputable def g : ℝ → ℝ → ℝ
| x, m => if x < m then x^2 + 4 else 3 * x + 6

theorem continuous_sum_m :
  ∀ m1 m2 : ℝ, (∀ m : ℝ, (g m m1 = g m m2) → g m (m1 + m2) = g m m1 + g m m2) →
  m1 + m2 = 3 :=
sorry

end continuous_sum_m_l213_213577


namespace power_mod_zero_problem_solution_l213_213347

theorem power_mod_zero (n : ℕ) (h : n ≥ 2) : 2 ^ n % 4 = 0 :=
  sorry

theorem problem_solution : 2 ^ 300 % 4 = 0 :=
  power_mod_zero 300 (by norm_num)

end power_mod_zero_problem_solution_l213_213347


namespace jack_initial_money_l213_213563

-- Define the cost of one pair of socks
def cost_pair_socks : ℝ := 9.50

-- Define the cost of soccer shoes
def cost_soccer_shoes : ℝ := 92

-- Define the additional money Jack needs
def additional_money_needed : ℝ := 71

-- Define the total cost of two pairs of socks and one pair of soccer shoes
def total_cost : ℝ := 2 * cost_pair_socks + cost_soccer_shoes

-- Theorem to prove Jack's initial money
theorem jack_initial_money : ∃ m : ℝ, total_cost - additional_money_needed = 40 :=
by
  sorry

end jack_initial_money_l213_213563


namespace incorrect_statement_C_l213_213672

/-- 
  Prove that the function y = -1/2 * x + 3 does not intersect the y-axis at (6,0).
-/
theorem incorrect_statement_C 
: ∀ (x y : ℝ), y = -1/2 * x + 3 → (x, y) ≠ (6, 0) :=
by
  intros x y h
  sorry

end incorrect_statement_C_l213_213672


namespace remainder_problem_l213_213022

theorem remainder_problem :
  ((98 * 103 + 7) % 12) = 1 :=
by
  sorry

end remainder_problem_l213_213022


namespace value_of_expression_l213_213549

theorem value_of_expression {x y z w : ℝ} (h1 : 4 * x * z + y * w = 3) (h2 : x * w + y * z = 6) :
  (2 * x + y) * (2 * z + w) = 15 :=
by
  sorry

end value_of_expression_l213_213549


namespace tickets_won_whack_a_mole_l213_213339

variable (t : ℕ)

def tickets_from_skee_ball : ℕ := 9
def cost_per_candy : ℕ := 6
def number_of_candies : ℕ := 7
def total_tickets_needed : ℕ := cost_per_candy * number_of_candies

theorem tickets_won_whack_a_mole : t + tickets_from_skee_ball = total_tickets_needed → t = 33 :=
by
  intro h
  have h1 : total_tickets_needed = 42 := by sorry
  have h2 : tickets_from_skee_ball = 9 := by rfl
  rw [h2, h1] at h
  sorry

end tickets_won_whack_a_mole_l213_213339


namespace farmer_brown_additional_cost_l213_213508

-- Definitions for the conditions
def originalQuantity : ℕ := 10
def originalPricePerBale : ℕ := 15
def newPricePerBale : ℕ := 18
def newQuantity : ℕ := 2 * originalQuantity

-- Definition for the target equation (additional cost)
def additionalCost : ℕ := (newQuantity * newPricePerBale) - (originalQuantity * originalPricePerBale)

-- Theorem stating the problem voiced in Lean 4
theorem farmer_brown_additional_cost : additionalCost = 210 :=
by {
  sorry
}

end farmer_brown_additional_cost_l213_213508


namespace intersection_point_l213_213372

variable (x y : ℚ)

theorem intersection_point :
  (8 * x - 5 * y = 40) ∧ (10 * x + 2 * y = 14) → 
  (x = 25 / 11) ∧ (y = 48 / 11) :=
by
  sorry

end intersection_point_l213_213372


namespace infinite_squares_of_form_l213_213846

theorem infinite_squares_of_form (k : ℕ) (hk : k > 0) : ∃ᶠ n in at_top, ∃ m : ℕ, n * 2^k - 7 = m^2 := sorry

end infinite_squares_of_form_l213_213846


namespace solve_fractional_eq_l213_213306

theorem solve_fractional_eq (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ -3) : (1 / x = 6 / (x + 3)) → (x = 0.6) :=
by
  sorry

end solve_fractional_eq_l213_213306


namespace ceil_neg_3_7_l213_213074

theorem ceil_neg_3_7 : Real.ceil (-3.7) = -3 := sorry

end ceil_neg_3_7_l213_213074


namespace first_reduction_percentage_l213_213324

theorem first_reduction_percentage (P : ℝ) (x : ℝ) (h : 0.30 * (1 - x / 100) * P = 0.225 * P) : x = 25 :=
by
  sorry

end first_reduction_percentage_l213_213324


namespace keaton_annual_profit_l213_213990

theorem keaton_annual_profit :
  let orange_harvests_per_year := 12 / 2
  let apple_harvests_per_year := 12 / 3
  let peach_harvests_per_year := 12 / 4
  let blackberry_harvests_per_year := 12 / 6

  let orange_profit_per_harvest := 50 - 20
  let apple_profit_per_harvest := 30 - 15
  let peach_profit_per_harvest := 45 - 25
  let blackberry_profit_per_harvest := 70 - 30

  let total_orange_profit := orange_harvests_per_year * orange_profit_per_harvest
  let total_apple_profit := apple_harvests_per_year * apple_profit_per_harvest
  let total_peach_profit := peach_harvests_per_year * peach_profit_per_harvest
  let total_blackberry_profit := blackberry_harvests_per_year * blackberry_profit_per_harvest

  let total_annual_profit := total_orange_profit + total_apple_profit + total_peach_profit + total_blackberry_profit

  total_annual_profit = 380
:= by
  sorry

end keaton_annual_profit_l213_213990


namespace no_prime_divisible_by_56_l213_213121

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define what it means for a number to be divisible by another number
def divisible_by (a b : ℕ) : Prop :=
  b ≠ 0 ∧ ∃ k : ℕ, a = b * k

-- The main theorem stating the problem
theorem no_prime_divisible_by_56 : ¬ ∃ p : ℕ, is_prime p ∧ divisible_by p 56 :=
  sorry

end no_prime_divisible_by_56_l213_213121


namespace unique_injective_f_solution_l213_213663

noncomputable def unique_injective_function (f : ℝ → ℝ) : Prop := 
  (∀ x y : ℝ, x ≠ y → f ((x + y) / (x - y)) = (f x + f y) / (f x - f y))

theorem unique_injective_f_solution :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, x ≠ y → f ((x + y) / (x - y)) = (f x + f y) / (f x - f y))
  → (∀ x y : ℝ, f x = f y → x = y) -- injectivity condition
  → ∀ x : ℝ, f x = x :=
sorry

end unique_injective_f_solution_l213_213663


namespace perpendicular_lines_l213_213684

theorem perpendicular_lines :
  ∃ y x : ℝ, (4 * y - 3 * x = 15) ∧ (3 * y + 4 * x = 12) :=
by
  sorry

end perpendicular_lines_l213_213684


namespace initial_number_divisible_by_15_l213_213797

theorem initial_number_divisible_by_15 (N : ℕ) (h : (N - 7) % 15 = 0) : N = 22 := 
by
  sorry

end initial_number_divisible_by_15_l213_213797


namespace sum_first_five_terms_l213_213092

-- Define the arithmetic sequence {a_n}
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ a1 d : ℝ, ∀ n : ℕ, a n = a1 + (n - 1) * d

-- Define the specific condition a_5 + a_8 - a_10 = 2
def specific_condition (a : ℕ → ℝ) : Prop :=
  a 5 + a 8 - a 10 = 2

-- Define the sum of the first five terms S₅
def S5 (a : ℕ → ℝ) : ℝ :=
  a 1 + a 2 + a 3 + a 4 + a 5 

-- The statement to be proved
theorem sum_first_five_terms (a : ℕ → ℝ) (h₁ : arithmetic_sequence a) (h₂ : specific_condition a) : 
  S5 a = 10 :=
sorry

end sum_first_five_terms_l213_213092


namespace count_valid_three_digit_integers_is_24_l213_213683

-- Define the set of digits available
def digits : Finset ℕ := {2, 4, 7, 9}

-- Define a function to count the number of valid three-digit integers
noncomputable def count_three_digit_integers : Nat := 
  Finset.card (Finset.filter (λ n : ℕ, 
    ∃ h₁ h₂ h₃, n = h₁ * 100 + h₂ * 10 + h₃ ∧ 
      h₁ ∈ digits ∧ h₂ ∈ digits ∧ h₃ ∈ digits ∧ 
      h₁ ≠ h₂ ∧ h₁ ≠ h₃ ∧ h₂ ≠ h₃
  ) (Finset.range 1000) )

-- The theorem stating the total number of different three-digit integers
theorem count_valid_three_digit_integers_is_24 : count_three_digit_integers = 24 :=
by
  sorry

end count_valid_three_digit_integers_is_24_l213_213683


namespace negation_of_universal_proposition_l213_213671

theorem negation_of_universal_proposition (x : ℝ) :
  ¬ (∀ m : ℝ, 0 ≤ m ∧ m ≤ 1 → x + 1 / x ≥ 2^m) ↔ ∃ m : ℝ, (0 ≤ m ∧ m ≤ 1) ∧ (x + 1 / x < 2^m) := by
  sorry

end negation_of_universal_proposition_l213_213671


namespace inverse_fourier_transform_l213_213666

noncomputable def F (p : ℝ) : ℂ :=
if 0 < p ∧ p < 1 then 1 else 0

noncomputable def f (x : ℝ) : ℂ :=
(1 / Real.sqrt (2 * Real.pi)) * ((1 - Complex.exp (-Complex.I * x)) / (Complex.I * x))

theorem inverse_fourier_transform :
  ∀ x, (f x) = (1 / (Real.sqrt (2 * Real.pi))) * ((1 - Complex.exp (-Complex.I * x)) / (Complex.I * x)) := by
  intros
  sorry

end inverse_fourier_transform_l213_213666


namespace geometric_sequence_sum_5_is_75_l213_213422

noncomputable def geometric_sequence_sum_5 (a r : ℝ) : ℝ :=
  a * (1 + r + r^2 + r^3 + r^4)

theorem geometric_sequence_sum_5_is_75 (a r : ℝ)
  (h1 : a * (1 + r + r^2) = 13)
  (h2 : a * (1 - r^7) / (1 - r) = 183) :
  geometric_sequence_sum_5 a r = 75 :=
sorry

end geometric_sequence_sum_5_is_75_l213_213422


namespace corresponding_angles_equal_l213_213591

-- Definition: Corresponding angles and their equality
def corresponding_angles (α β : ℝ) : Prop :=
  -- assuming definition of corresponding angles can be defined
  sorry

theorem corresponding_angles_equal {α β : ℝ} (h : corresponding_angles α β) : α = β :=
by
  -- the proof is provided in the problem statement
  sorry

end corresponding_angles_equal_l213_213591


namespace functional_eq_1996_l213_213444

def f (x : ℝ) : ℝ := sorry

theorem functional_eq_1996 (f : ℝ → ℝ)
    (h : ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * ((f x)^2 - (f x) * (f y) + (f y)^2)) :
    ∀ x : ℝ, f (1996 * x) = 1996 * f x := 
sorry

end functional_eq_1996_l213_213444


namespace evaluate_expression_l213_213993

theorem evaluate_expression {x y : ℕ} (h₁ : 144 = 2^x * 3^y) (hx : x = 4) (hy : y = 2) : (1 / 7) ^ (y - x) = 49 := 
by
  sorry

end evaluate_expression_l213_213993


namespace swimming_speed_in_still_water_l213_213495

variable (v : ℝ) -- the person's swimming speed in still water

-- Conditions
variable (water_speed : ℝ := 4) -- speed of the water
variable (time : ℝ := 2) -- time taken to swim 12 km against the current
variable (distance : ℝ := 12) -- distance swam against the current

theorem swimming_speed_in_still_water :
  (v - water_speed) = distance / time → v = 10 :=
by
  sorry

end swimming_speed_in_still_water_l213_213495


namespace minimum_value_of_reciprocal_product_l213_213249

theorem minimum_value_of_reciprocal_product (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + a * b + 2 * b = 30) : 
  ∃ m : ℝ, m = 1 / (a * b) ∧ m = 1 / 18 :=
sorry

end minimum_value_of_reciprocal_product_l213_213249


namespace mean_home_runs_correct_l213_213875

-- Define the total home runs in April
def total_home_runs_April : ℕ := 5 * 4 + 6 * 4 + 8 * 2 + 10

-- Define the total home runs in May
def total_home_runs_May : ℕ := 5 * 2 + 6 * 2 + 8 * 3 + 10 * 2 + 11

-- Define the total number of top hitters/players
def total_players : ℕ := 12

-- Define the total home runs over two months
def total_home_runs : ℕ := total_home_runs_April + total_home_runs_May

-- Calculate the mean number of home runs
def mean_home_runs : ℚ := total_home_runs / total_players

-- Prove that the calculated mean is equal to the expected result
theorem mean_home_runs_correct : mean_home_runs = 12.08 := by
  sorry

end mean_home_runs_correct_l213_213875


namespace sequence_formula_l213_213983

theorem sequence_formula (a : ℕ → ℝ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = a n / (1 + 2 * a n)) :
  ∀ n, a n = 1 / (2 * n - 1) :=
by
  sorry

end sequence_formula_l213_213983


namespace range_of_a_l213_213104

theorem range_of_a (a : ℝ) (e : ℝ) (x : ℝ) (ln : ℝ → ℝ) :
  (∀ x, (1 / e) ≤ x ∧ x ≤ e → (a - x^2 = -2 * ln x)) →
  (1 ≤ a ∧ a ≤ (e^2 - 2)) :=
by
  sorry

end range_of_a_l213_213104


namespace exists_digit_combination_l213_213585

theorem exists_digit_combination (d1 d2 d3 d4 : ℕ) (H1 : 42 * (d1 * 10 + 8) = 2 * 1000 + d2 * 100 + d3 * 10 + d4) (H2: ∃ n: ℕ, n = 2 + d2 + d3 + d4 ∧ n % 2 = 1):
  d1 = 4 ∧ 42 * 48 = 2016 ∨ d1 = 6 ∧ 42 * 68 = 2856 :=
sorry

end exists_digit_combination_l213_213585


namespace find_number_l213_213477

theorem find_number (N : ℝ) (h : (5/4 : ℝ) * N = (4/5 : ℝ) * N + 27) : N = 60 :=
by
  sorry

end find_number_l213_213477


namespace probability_of_special_number_l213_213494

open Set

/-- Define the set of numbers from 40 to 999 -/
def S : Set ℕ := {n | 40 ≤ n ∧ n ≤ 999}

/-- Define what it means for a number to be either less than 60 or a multiple of 10 -/
def is_special (n : ℕ) : Prop := n < 60 ∨ n % 10 = 0

/-- Define the event of selecting a special number from the set S -/
def special_event : Set ℕ := {n ∈ S | is_special n}

/-- Define the total number of elements in S -/
def total_elements : ℕ := S.toFinset.card

/-- Define the number of special elements in S -/
def special_elements : ℕ := special_event.toFinset.card

/-- Define the probability as a rational number -/
def probability : ℚ := special_elements / total_elements

/-- The probability of selecting a special number from set S is 19/160 -/
theorem probability_of_special_number : probability = 19 / 160 := 
by sorry

end probability_of_special_number_l213_213494


namespace red_balls_estimation_l213_213980

noncomputable def numberOfRedBalls (x : ℕ) : ℝ := x / (x + 3)

theorem red_balls_estimation {x : ℕ} (h : numberOfRedBalls x = 0.85) : x = 17 :=
by
  sorry

end red_balls_estimation_l213_213980


namespace prime_power_lcm_condition_l213_213150

open Nat

theorem prime_power_lcm_condition (n : ℕ) (h : n ≥ 2) (d : ℕ → Prop) :
  (∃ p k : ℕ, prime p ∧ k ≥ 1 ∧ n = p^k) ↔ lcm (filter (λ m, m < n) (range (n + 1))) ≠ n :=
by
  sorry

end prime_power_lcm_condition_l213_213150


namespace cyclic_sum_inequality_l213_213443

open Real

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  1 / (1 + a + b) + 1 / (1 + b + c) + 1 / (1 + c + a) ≤ 1 := 
  sorry

end cyclic_sum_inequality_l213_213443


namespace prob_eventA_and_eventB_l213_213183

def boxA := {1, 2, ..., 25}
def boxB := {1, 2, ..., 30}

def eventA := {n ∈ boxA | n < 20}
def eventB := {n ∈ boxB | Nat.Prime n ∨ n > 28}

def probA : ℚ := 19 / 25
def probB : ℚ := 11 / 30

theorem prob_eventA_and_eventB : 
  probA * probB = 209 / 750 :=
by 
  sorry

end prob_eventA_and_eventB_l213_213183


namespace number_of_candies_in_a_packet_l213_213223

theorem number_of_candies_in_a_packet 
  (two_packets : ℕ)
  (candies_per_day_weekday : ℕ)
  (candies_per_day_weekend : ℕ)
  (weeks : ℕ)
  (total_candies : ℕ)
  (packet_size : ℕ)
  (H1 : two_packets = 2)
  (H2 : candies_per_day_weekday = 2)
  (H3 : candies_per_day_weekend = 1)
  (H4 : weeks = 3)
  (H5 : packet_size > 0)
  (H6 : total_candies = packets * packet_size)
  (H7 : total_candies = 3 * (5 * candies_per_day_weekday + 2 * candies_per_day_weekend))
  : packet_size = 18 :=
by
  sorry

end number_of_candies_in_a_packet_l213_213223


namespace problem_statement_l213_213387

theorem problem_statement (p x : ℝ) (h : 0 ≤ p ∧ p ≤ 4) :
  (x^2 + p*x > 4*x + p - 3) ↔ (x > 3 ∨ x < -1) := by
sorry

end problem_statement_l213_213387


namespace nina_money_l213_213580

-- Definitions based on the problem's conditions
def original_widgets := 15
def reduced_widgets := 25
def price_reduction := 5

-- The statement
theorem nina_money : 
  ∃ (W : ℝ), 15 * W = 25 * (W - 5) ∧ 15 * W = 187.5 :=
by
  sorry

end nina_money_l213_213580


namespace max_profit_l213_213640

variables (x y : ℕ)

def steel_constraint := 10 * x + 70 * y ≤ 700
def non_ferrous_constraint := 23 * x + 40 * y ≤ 642
def non_negativity := x ≥ 0 ∧ y ≥ 0
def profit := 80 * x + 100 * y

theorem max_profit (h₁ : steel_constraint x y)
                   (h₂ : non_ferrous_constraint x y)
                   (h₃ : non_negativity x y):
  profit x y = 2180 := 
sorry

end max_profit_l213_213640


namespace arithmetic_sequence_propositions_l213_213090

theorem arithmetic_sequence_propositions (a_n : ℕ → ℤ) (S : ℕ → ℤ)
  (h_S_def : ∀ n, S n = n * (a_n 1 + (a_n (n - 1))) / 2)
  (h_cond : S 6 > S 7 ∧ S 7 > S 5) :
  (∃ d, d < 0 ∧ S 11 > 0) :=
by
  sorry

end arithmetic_sequence_propositions_l213_213090


namespace speed_in_still_water_l213_213038

def upstream_speed : ℝ := 25
def downstream_speed : ℝ := 35

theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 30 :=
by
  sorry

end speed_in_still_water_l213_213038


namespace product_identity_l213_213228

theorem product_identity : 
  (7^3 - 1) / (7^3 + 1) * 
  (8^3 - 1) / (8^3 + 1) * 
  (9^3 - 1) / (9^3 + 1) * 
  (10^3 - 1) / (10^3 + 1) * 
  (11^3 - 1) / (11^3 + 1) = 
  133 / 946 := 
by
  sorry

end product_identity_l213_213228


namespace find_monotonic_intervals_max_min_on_interval_l213_213817

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

noncomputable def f' (x : ℝ) : ℝ := (Real.cos x - Real.sin x) * Real.exp x - 1

theorem find_monotonic_intervals (k : ℤ) : 
  ((2 * k * Real.pi - Real.pi < x ∧ x < 2 * k * Real.pi) → 0 < (f' x)) ∧
  ((2 * k * Real.pi < x ∧ x < 2 * k * Real.pi + Real.pi) → (f' x) < 0) :=
sorry

theorem max_min_on_interval : 
  (∀ x, 0 ≤ x ∧ x ≤ (2 * Real.pi / 3) → f 0 = 1 ∧ f (2 * Real.pi / 3) =  -((1/2) * Real.exp (2/3 * Real.pi)) - (2 * Real.pi / 3)) :=
sorry

end find_monotonic_intervals_max_min_on_interval_l213_213817


namespace series_sum_half_l213_213649

theorem series_sum_half :
  ∑' n : ℕ, (2^n / (3^(2^n) + 1)) = 1/2 := 
sorry

end series_sum_half_l213_213649


namespace largest_possible_value_of_n_l213_213004

open Nat

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ m, m ∣ p → m = 1 ∨ m = p

def largest_product : ℕ :=
  705

theorem largest_possible_value_of_n :
  ∃ (x y : ℕ), x < 10 ∧ y < 10 ∧
  is_prime x ∧ is_prime y ∧
  is_prime (10 * y - x) ∧
  largest_product = x * y * (10 * y - x) :=
by
  sorry

end largest_possible_value_of_n_l213_213004


namespace remainder_when_divided_by_2_l213_213903

theorem remainder_when_divided_by_2 (n : ℕ) (h₁ : n > 0) (h₂ : (n + 1) % 6 = 4) : n % 2 = 1 :=
by sorry

end remainder_when_divided_by_2_l213_213903


namespace find_d_l213_213576

theorem find_d (d x y : ℝ) (H1 : x - 2 * y = 5) (H2 : d * x + y = 6) (H3 : x > 0) (H4 : y > 0) :
  -1 / 2 < d ∧ d < 6 / 5 :=
by
  sorry

end find_d_l213_213576


namespace work_completion_days_l213_213197

theorem work_completion_days
  (E_q : ℝ) -- Efficiency of q
  (E_p : ℝ) -- Efficiency of p
  (E_r : ℝ) -- Efficiency of r
  (W : ℝ)  -- Total work
  (H1 : E_p = 1.5 * E_q) -- Condition 1
  (H2 : W = E_p * 25) -- Condition 2
  (H3 : E_r = 0.8 * E_q) -- Condition 3
  : (W / (E_p + E_q + E_r)) = 11.36 := -- Prove the days_needed is 11.36
by
  sorry

end work_completion_days_l213_213197


namespace first_term_of_geometric_series_l213_213365

theorem first_term_of_geometric_series (r : ℚ) (S : ℚ) (a : ℚ) (h1 : r = 1/5) (h2 : S = 100) (h3 : S = a / (1 - r)) : a = 80 := 
by
  sorry

end first_term_of_geometric_series_l213_213365


namespace gain_percent_l213_213271

variable (C S : ℝ)

theorem gain_percent (h : 50 * C = 28 * S) : ((S - C) / C) * 100 = 78.57 := by
  sorry

end gain_percent_l213_213271


namespace problem1_problem2_l213_213105

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + 1
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (x * f a x - x) / Real.exp x
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := 1 - (f a x - 1) / Real.exp x

theorem problem1 (x : ℝ) (h₁ : x ≥ 5) : g 1 x < 1 :=
sorry

theorem problem2 (a : ℝ) (h₂ : a > Real.exp 2 / 4) : 
∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 0 < x1 ∧ 0 < x2 ∧ h a x1 = 0 ∧ h a x2 = 0 :=
sorry

end problem1_problem2_l213_213105


namespace translated_function_is_even_and_in_range_l213_213741

open Real

noncomputable def f (x : ℝ) := cos (2 * x - π / 3) ^ 2

noncomputable def g (x : ℝ) := f (x + π / 6)

theorem translated_function_is_even_and_in_range :
  ∀ x : ℝ, g x = (1 + cos (4 * x)) / 2 ∧ g x = g (-x) ∧ 0 ≤ g x ∧ g x ≤ 1 :=
by
  sorry

end translated_function_is_even_and_in_range_l213_213741


namespace number_of_pairs_l213_213433

open Finset

theorem number_of_pairs (P : Finset ℕ) (hP : P = ({1, 2, 3, 4, 5, 6} : Finset ℕ)) :
  ∃ (A B : Finset ℕ), A ⊆ P ∧ B ⊆ P ∧ A.nonempty ∧ B.nonempty ∧ (A.max' (by simp [hP, P.nonempty])) < (B.min' (by simp [hP, P.nonempty])) ∧
  P.pair_count (λ A B, (A.max' (by simp [hP, A.nonempty])) < (B.min' (by simp [hP, B.nonempty]))) = 129 :=
by
  sorry

end number_of_pairs_l213_213433


namespace evaluate_expression_l213_213373

theorem evaluate_expression : -(16 / 4 * 7 - 50 + 5 * 7) = -13 :=
by
  sorry

end evaluate_expression_l213_213373


namespace max_sphere_volume_in_prism_l213_213133

theorem max_sphere_volume_in_prism
  (AB BC : ℝ) (AA₁ : ℝ)
  (h_AB_BC : AB = 6) (h_BC_AB : BC = 8) (h_AA₁ : AA₁ = 3) :
  ∃ V : ℝ, V = (9 * real.pi) / 2 :=
by {
  sorry
}

end max_sphere_volume_in_prism_l213_213133


namespace original_flow_rate_l213_213751

theorem original_flow_rate :
  ∃ F : ℚ, 
  (F * 0.75 * 0.4 * 0.6 - 1 = 2) ∧
  (F = 50/3) :=
by
  sorry

end original_flow_rate_l213_213751


namespace non_trivial_solution_exists_l213_213851

theorem non_trivial_solution_exists (a b c : ℤ) (p : ℕ) [Fact (Nat.Prime p)] :
  ∃ x y z : ℤ, (a * x^2 + b * y^2 + c * z^2) % p = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) :=
sorry

end non_trivial_solution_exists_l213_213851


namespace milk_needed_6_cookies_3_3_pints_l213_213015

def gallon_to_quarts (g : ℚ) : ℚ := g * 4
def quarts_to_pints (q : ℚ) : ℚ := q * 2
def cookies_to_pints (p : ℚ) (c : ℚ) (n : ℚ) : ℚ := (p / c) * n
def measurement_error (p : ℚ) : ℚ := p * 1.1

theorem milk_needed_6_cookies_3_3_pints :
  (measurement_error (cookies_to_pints (quarts_to_pints (gallon_to_quarts 1.5)) 24 6) = 3.3) :=
by
  sorry

end milk_needed_6_cookies_3_3_pints_l213_213015


namespace renovation_project_cement_loads_l213_213359

theorem renovation_project_cement_loads
  (s : ℚ) (d : ℚ) (t : ℚ)
  (hs : s = 0.16666666666666666) 
  (hd : d = 0.3333333333333333)
  (ht : t = 0.6666666666666666) :
  t - (s + d) = 0.1666666666666666 := by
  sorry

end renovation_project_cement_loads_l213_213359


namespace scientific_notation_2150000_l213_213790

theorem scientific_notation_2150000 : 2150000 = 2.15 * 10^6 :=
  by
  sorry

end scientific_notation_2150000_l213_213790


namespace no_nonnegative_integral_solutions_l213_213657

theorem no_nonnegative_integral_solutions :
  ¬ ∃ (x y : ℕ), (x^4 * y^4 - 14 * x^2 * y^2 + 49 = 0) ∧ (x + y = 10) :=
by
  sorry

end no_nonnegative_integral_solutions_l213_213657


namespace jack_additional_sweets_is_correct_l213_213886

/-- Initial number of sweets --/
def initial_sweets : ℕ := 22

/-- Sweets taken by Paul --/
def sweets_taken_by_paul : ℕ := 7

/-- Jack's total sweets taken --/
def jack_total_sweets_taken : ℕ := initial_sweets - sweets_taken_by_paul

/-- Half of initial sweets --/
def half_initial_sweets : ℕ := initial_sweets / 2

/-- Additional sweets taken by Jack --/
def additional_sweets_taken_by_jack : ℕ := jack_total_sweets_taken - half_initial_sweets

theorem jack_additional_sweets_is_correct : additional_sweets_taken_by_jack = 4 := by
  sorry

end jack_additional_sweets_is_correct_l213_213886


namespace dislikes_TV_and_books_l213_213279

-- The problem conditions
def total_people : ℕ := 800
def percent_dislikes_TV : ℚ := 25 / 100
def percent_dislikes_both : ℚ := 15 / 100

-- The expected answer
def expected_dislikes_TV_and_books : ℕ := 30

-- The proof problem statement
theorem dislikes_TV_and_books : 
  (total_people * percent_dislikes_TV) * percent_dislikes_both = expected_dislikes_TV_and_books := by 
  sorry

end dislikes_TV_and_books_l213_213279


namespace polynomial_A_l213_213272

theorem polynomial_A (A a : ℝ) (h : A * (a + 1) = a^2 - 1) : A = a - 1 :=
sorry

end polynomial_A_l213_213272


namespace part1_part2_l213_213406

-- Define set A
def A : Set ℝ := {x | 3 < x ∧ x < 6}

-- Define set B
def B : Set ℝ := {x | 2 < x ∧ x < 9}

-- Define set complement in ℝ
def CR (S : Set ℝ) : Set ℝ := {x | ¬ (x ∈ S)}

-- First part of the problem
theorem part1 :
  (A ∩ B = {x | 3 < x ∧ x < 6}) ∧
  (CR A ∪ CR B = {x | x ≤ 3 ∨ x ≥ 6}) :=
sorry

-- Define set C depending on a
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2 * a - 1}

-- Second part of the problem
theorem part2 (a : ℝ) (h : B ∪ C a = B) :
  a ≤ 1 ∨ (2 ≤ a ∧ a ≤ 5) :=
sorry

end part1_part2_l213_213406


namespace locus_is_hyperbola_l213_213949

theorem locus_is_hyperbola
  (x y a θ₁ θ₂ c : ℝ)
  (h1 : (x - a) * Real.cos θ₁ + y * Real.sin θ₁ = a)
  (h2 : (x - a) * Real.cos θ₂ + y * Real.sin θ₂ = a)
  (h3 : Real.tan (θ₁ / 2) - Real.tan (θ₂ / 2) = 2 * c)
  (hc : c > 1) 
  : ∃ k l m : ℝ, k * (x ^ 2) + l * x * y + m * (y ^ 2) = 1 := sorry

end locus_is_hyperbola_l213_213949


namespace range_of_a_l213_213645

/--
Let f be a function defined on the interval [-1, 1] that is increasing and odd.
If f(-a+1) + f(4a-5) > 0, then the range of the real number a is (4/3, 3/2].
-/
theorem range_of_a
  (f : ℝ → ℝ)
  (h_dom : ∀ x, -1 ≤ x ∧ x ≤ 1 → f x = f x)  -- domain condition
  (h_incr : ∀ x y, x < y → f x < f y)          -- increasing condition
  (h_odd : ∀ x, f (-x) = - f x)                -- odd function condition
  (a : ℝ)
  (h_ineq : f (-a + 1) + f (4 * a - 5) > 0) :
  4 / 3 < a ∧ a ≤ 3 / 2 :=
sorry

end range_of_a_l213_213645


namespace count_whole_numbers_between_4_and_18_l213_213673

theorem count_whole_numbers_between_4_and_18 :
  ∀ (x : ℕ), 4 < x ∧ x < 18 ↔ ∃ n : ℕ, n = 13 :=
by sorry

end count_whole_numbers_between_4_and_18_l213_213673


namespace solve_quadratic_eq_l213_213085

theorem solve_quadratic_eq (b c : ℝ) :
  (∀ x : ℝ, |x - 3| = 4 ↔ x = 7 ∨ x = -1) →
  (∀ x : ℝ, x^2 + b * x + c = 0 ↔ x = 7 ∨ x = -1) →
  b = -6 ∧ c = -7 :=
by
  intros h_abs_val_eq h_quad_eq
  sorry

end solve_quadratic_eq_l213_213085


namespace part1_monotonicity_when_a_eq_1_part2_range_of_a_l213_213819

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x - 1) * (Real.log (x - 2)) - a * (x - 3)

theorem part1_monotonicity_when_a_eq_1 :
  ∀ x, 2 < x → ∀ x1, (2 < x1 → f x 1 ≤ f x1 1) := by
  sorry

theorem part2_range_of_a :
  ∀ a, (∀ x, 3 < x → f x a > 0) → a ≤ 2 := by
  sorry

end part1_monotonicity_when_a_eq_1_part2_range_of_a_l213_213819


namespace intersecting_lines_fixed_point_l213_213258

variable (p a b : ℝ)
variable (h1 : a ≠ 0)
variable (h2 : b ≠ 0)
variable (h3 : b^2 ≠ 2 * p * a)

def parabola (M : ℝ × ℝ) : Prop := M.2^2 = 2 * p * M.1

def fixed_points (A B : ℝ × ℝ) : Prop :=
  A = (a, b) ∧ B = (-a, 0)

def intersect_parabola (M1 M2 M : ℝ × ℝ) : Prop :=
  parabola p M ∧ parabola p M1 ∧ parabola p M2 ∧ M ≠ M1 ∧ M ≠ M2

theorem intersecting_lines_fixed_point (M M1 M2 : ℝ × ℝ)
  (hP : parabola p M) 
  (hA : (a, b) ≠ M) 
  (hB : (-a, 0) ≠ M) 
  (h_intersect : intersect_parabola p M1 M2 M) :
  ∃ C : ℝ × ℝ, C = (a, 2 * p * a / b) :=
sorry

end intersecting_lines_fixed_point_l213_213258


namespace probability_of_sum_5_l213_213592

namespace DieProblem

-- Define the sample space for a single die roll
def roll_outcomes : Finset (ℕ × ℕ) := (Finset.finRange 6).product (Finset.finRange 6)

-- Define event A: the sum of the rolls is 5
def event_A (outcome : ℕ × ℕ) : Prop := outcome.fst + outcome.snd = 5

-- Finset of outcomes where event_A holds
def outcomes_A : Finset (ℕ × ℕ) := roll_outcomes.filter event_A

-- Probability of event_A
def probability_A : ℚ := (outcomes_A.card : ℚ) / (roll_outcomes.card : ℚ)

theorem probability_of_sum_5 :
  probability_A = 1 / 9 := by
  -- Proof goes here
  sorry

end DieProblem

end probability_of_sum_5_l213_213592


namespace S2016_value_l213_213426

theorem S2016_value (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h1 : a 1 = -2016)
  (h2 : ∀ n, S (n+1) = S n + a (n+1))
  (h3 : ∀ n, a (n+1) = a n + d)
  (h4 : (S 2015) / 2015 - (S 2012) / 2012 = 3) : S 2016 = -2016 := 
sorry

end S2016_value_l213_213426


namespace triangle_ABC_properties_l213_213417

theorem triangle_ABC_properties
  (a b c : ℝ)
  (A B C : ℝ)
  (area_ABC : Real.sqrt 15 * 3 = 1/2 * b * c * Real.sin A)
  (cos_A : Real.cos A = -1/4)
  (b_minus_c : b - c = 2) :
  (a = 8 ∧ Real.sin C = Real.sqrt 15 / 8) ∧
  (Real.cos (2 * A + Real.pi / 6) = (Real.sqrt 15 - 7 * Real.sqrt 3) / 16) := by
  sorry

end triangle_ABC_properties_l213_213417


namespace cube_volume_and_surface_area_l213_213326

theorem cube_volume_and_surface_area (e : ℕ) (h : 12 * e = 72) :
  (e^3 = 216) ∧ (6 * e^2 = 216) := by
  sorry

end cube_volume_and_surface_area_l213_213326


namespace angle_at_intersection_l213_213385

noncomputable def angle_between_curves (y₁ y₂ : ℝ → ℝ) (x₀ y₀ : ℝ) : ℝ :=
  let k₁ := deriv y₁ x₀
  let k₂ := deriv y₂ x₀
  let tan_phi := abs ((k₂ - k₁) / (1 + k₁ * k₂))
  arctan tan_phi

theorem angle_at_intersection :
  let y₁ : ℝ → ℝ := λ x => 2 * x^2
  let y₂ : ℝ → ℝ := λ x => x^3 + 2 * x^2 - 1
  let x₀ : ℝ := 1
  let y₀ : ℝ := 2
  angle_between_curves y₁ y₂ x₀ y₀ = arctan (3 / 29) :=
by sorry

end angle_at_intersection_l213_213385


namespace sara_frosting_total_l213_213158

def cakes_baked_each_day : List Nat := [7, 12, 8, 10, 15]
def cakes_eaten_by_Carol : List Nat := [4, 6, 3, 2, 3]
def cans_per_cake_each_day : List Nat := [2, 3, 4, 3, 2]

def total_frosting_cans_needed : Nat :=
  let remaining_cakes := List.zipWith (· - ·) cakes_baked_each_day cakes_eaten_by_Carol
  let required_cans := List.zipWith (· * ·) remaining_cakes cans_per_cake_each_day
  required_cans.foldl (· + ·) 0

theorem sara_frosting_total : total_frosting_cans_needed = 92 := by
  sorry

end sara_frosting_total_l213_213158


namespace color_3x3_grid_l213_213369

theorem color_3x3_grid :
  let grid := { (i, j) : Fin 3 × Fin 3 // true } in
  let no_shared_edge (cells : Finset (Fin 3 × Fin 3)) := ∀ x y ∈ cells, x ≠ y → 
    (|x.1 - y.1|, |x.2 - y.2|) ≠ (1, 0) ∧ (|x.1 - y.1|, |x.2 - y.2|) ≠ (0, 1) in
  Finset.card {cells : Finset (Fin 3 × Fin 3) // cells.card = 3 ∧ no_shared_edge cells} = 22 :=
begin
  sorry -- Proof to be implemented
end

end color_3x3_grid_l213_213369


namespace first_car_distance_l213_213334

-- Definitions for conditions
variable (x : ℝ) -- distance the first car ran before taking the right turn
def distance_apart_initial := 150 -- initial distance between the cars
def distance_first_car_main_road := 2 * x -- total distance first car ran on the main road
def distance_second_car := 62 -- distance the second car ran due to breakdown
def distance_between_cars := 38 -- distance between the cars after running 

-- Proof (statement only, no solution steps)
theorem first_car_distance (hx : distance_apart_initial = distance_first_car_main_road + distance_second_car + distance_between_cars) : 
  x = 25 :=
by
  unfold distance_apart_initial distance_first_car_main_road distance_second_car distance_between_cars at hx
  -- Implementation placeholder
  sorry

end first_car_distance_l213_213334


namespace relationship_of_y1_y2_l213_213833

theorem relationship_of_y1_y2 (y1 y2 : ℝ) : 
  (∃ y1 y2, (y1 = 2 / -2) ∧ (y2 = 2 / -1)) → (y1 > y2) :=
by
  sorry

end relationship_of_y1_y2_l213_213833


namespace theta_value_l213_213024

theorem theta_value (Theta : ℕ) (h_digit : Θ < 10) (h_eq : 252 / Θ = 30 + 2 * Θ) : Θ = 6 := 
by
  sorry

end theta_value_l213_213024


namespace fractions_lcm_l213_213782

noncomputable def lcm_of_fractions_lcm (numerators : List ℕ) (denominators : List ℕ) : ℕ :=
  let lcm_nums := numerators.foldr Nat.lcm 1
  let gcd_denom := denominators.foldr Nat.gcd (denominators.headD 1)
  lcm_nums / gcd_denom

theorem fractions_lcm (hnum : List ℕ := [4, 5, 7, 9, 13, 16, 19])
                      (hdenom : List ℕ := [9, 7, 15, 13, 21, 35, 45]) :
  lcm_of_fractions_lcm hnum hdenom = 1244880 :=
by
  sorry

end fractions_lcm_l213_213782


namespace min_pairs_of_friends_l213_213354

variables (n : ℕ) (invites_per_person : ℕ) (pairs_of_friends : ℕ)
variable (people_invited : ℕ)

theorem min_pairs_of_friends :
  n = 2000 ∧ invites_per_person = 1000 ∧ 
  (∀ i j, i ≠ j → a_ij = (invited_by i j + invited_by j i ∈ {0, 1, 2}) ) → 
  pairs_of_friends ≥ 1000 :=
begin
  sorry
end

end min_pairs_of_friends_l213_213354


namespace find_other_denomination_l213_213020

theorem find_other_denomination
  (total_spent : ℕ)
  (twenty_bill_value : ℕ) (other_denomination_value : ℕ)
  (twenty_bill_count : ℕ) (other_bill_count : ℕ)
  (h1 : total_spent = 80)
  (h2 : twenty_bill_value = 20)
  (h3 : other_bill_count = 2)
  (h4 : twenty_bill_count = other_bill_count + 1)
  (h5 : total_spent = twenty_bill_value * twenty_bill_count + other_denomination_value * other_bill_count) : 
  other_denomination_value = 10 :=
by
  sorry

end find_other_denomination_l213_213020


namespace problem1_problem2_l213_213226

-- Definition for the first proof problem
theorem problem1 (a b : ℝ) (h : a ≠ b) :
  (a^2 / (a - b) - b^2 / (a - b)) = a + b :=
by
  sorry

-- Definition for the second proof problem
theorem problem2 (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 0) :
  ((x^2 - 1) / ((x^2 + 2 * x + 1)) / (x^2 - x) / (x + 1)) = 1 / x :=
by
  sorry

end problem1_problem2_l213_213226


namespace smallest_of_three_consecutive_odd_numbers_l213_213327

theorem smallest_of_three_consecutive_odd_numbers (x : ℤ) (h : x + (x + 2) + (x + 4) = 69) : x = 21 :=
sorry

end smallest_of_three_consecutive_odd_numbers_l213_213327


namespace width_of_wall_is_two_l213_213485

noncomputable def volume_of_brick : ℝ := 20 * 10 * 7.5 / 10^6 -- Volume in cubic meters
def number_of_bricks : ℕ := 27000
noncomputable def volume_of_wall (width : ℝ) : ℝ := 27 * width * 0.75

theorem width_of_wall_is_two :
  ∃ (W : ℝ), volume_of_wall W = number_of_bricks * volume_of_brick ∧ W = 2 :=
by
  sorry

end width_of_wall_is_two_l213_213485


namespace correct_probability_l213_213800

-- Five people taking a true-or-false test with five questions
def random_guess (n : ℕ) : ℕ := 2^n

-- Majority correct condition 
def majority_correct (people questions : ℕ) := ∀ q : ℕ, q < questions → (3 ≤ people) ∧ (people ≤ 5)

-- Probability that every person answers exactly three questions correctly
def all_answer_three_correctly (people questions : ℕ) : ℕ := nat.choose questions 3

-- Calculation bases 
def p (a b : ℕ) : ℚ := (a : ℚ) / 2^b

-- Final lean theorem
theorem correct_probability :
  (p 255 17 = 255 / 2^17) ∧ (100 * 255 + 17 = 25517) :=
begin
  sorry
end

end correct_probability_l213_213800


namespace quotient_remainder_increase_l213_213625

theorem quotient_remainder_increase (a b q r q' r' : ℕ) (hb : b ≠ 0) 
    (h1 : a = b * q + r) (h2 : 0 ≤ r) (h3 : r < b) (h4 : 3 * a = 3 * b * q' + r') 
    (h5 : 0 ≤ r') (h6 : r' < 3 * b) :
    q' = q ∧ r' = 3 * r := by
  sorry

end quotient_remainder_increase_l213_213625


namespace customer_initial_amount_l213_213489

theorem customer_initial_amount (d c : ℕ) (h1 : c = 100 * d) (h2 : c = 2 * d) : d = 0 ∧ c = 0 := by
  sorry

end customer_initial_amount_l213_213489


namespace flag_height_l213_213333

-- Definitions based on conditions
def flag_width : ℝ := 5
def paint_cost_per_quart : ℝ := 2
def sqft_per_quart : ℝ := 4
def total_spent : ℝ := 20

-- The theorem to prove the height h of the flag
theorem flag_height (h : ℝ) (paint_needed : ℝ -> ℝ) :
  paint_needed h = 4 := sorry

end flag_height_l213_213333


namespace sin_tan_identity_of_cos_eq_tan_identity_l213_213547

open Real

variable (α : ℝ)
variable (hα : α ∈ Ioo 0 π)   -- α is in the interval (0, π)
variable (hcos : cos (2 * α) = 2 * cos (α + π / 4))

theorem sin_tan_identity_of_cos_eq_tan_identity : 
  sin (2 * α) = 1 ∧ tan α = 1 :=
by
  sorry

end sin_tan_identity_of_cos_eq_tan_identity_l213_213547


namespace intersection_sets_l213_213407

theorem intersection_sets :
  let A := { x : ℝ | x^2 - 1 ≥ 0 }
  let B := { x : ℝ | 1 ≤ x ∧ x < 3 }
  A ∩ B = { x : ℝ | 1 ≤ x ∧ x < 3 } :=
by
  sorry

end intersection_sets_l213_213407


namespace general_term_of_sequence_l213_213810

variable (a : ℕ → ℕ)
variable (h1 : ∀ m : ℕ, a (m^2) = a m ^ 2)
variable (h2 : ∀ m k : ℕ, a (m^2 + k^2) = a m * a k)

theorem general_term_of_sequence : ∀ n : ℕ, n > 0 → a n = 1 :=
by
  intros n hn
  sorry

end general_term_of_sequence_l213_213810


namespace base_b_cube_l213_213323

theorem base_b_cube (b : ℕ) : (b > 4) → (∃ n : ℕ, (b^2 + 4 * b + 4 = n^3)) ↔ (b = 5 ∨ b = 6) :=
by
  sorry

end base_b_cube_l213_213323


namespace find_cd_l213_213190

def g (c d x : ℝ) : ℝ := c * x^3 - 4 * x^2 + d * x - 7

theorem find_cd :
  let c := -1 / 3
  let d := 28 / 3
  g c d 2 = -7 ∧ g c d (-1) = -20 :=
by sorry

end find_cd_l213_213190


namespace divisor_between_40_and_50_l213_213078

theorem divisor_between_40_and_50 (n : ℕ) (h1 : 40 ≤ n) (h2 : n ≤ 50) (h3 : n ∣ (2^36 - 1)) : n = 49 :=
sorry

end divisor_between_40_and_50_l213_213078


namespace num_men_in_first_group_l213_213309

-- Define conditions
def work_rate_first_group (x : ℕ) : ℚ := (1 : ℚ) / (20 * x)
def work_rate_second_group : ℚ := (1 : ℚ) / (15 * 24)

-- State the theorem
theorem num_men_in_first_group : ∀ x : ℕ, work_rate_first_group x = work_rate_second_group → x = 18 :=
by
  intro x h
  -- You can leave the proof for the implementation
  sorry

end num_men_in_first_group_l213_213309


namespace expected_area_swept_by_rotation_l213_213292

theorem expected_area_swept_by_rotation :
  let Ω := circle 1
  let A : Point := some_point_on_Ω_circumference Ω
  let θ := Uniform 0 180
  expected_area_swept Ω A θ = 2 * π :=
sorry

end expected_area_swept_by_rotation_l213_213292


namespace constant_in_quadratic_eq_l213_213742

theorem constant_in_quadratic_eq (C : ℝ) (x₁ x₂ : ℝ) 
  (h1 : 2 * x₁ * x₁ + 5 * x₁ - C = 0) 
  (h2 : 2 * x₂ * x₂ + 5 * x₂ - C = 0) 
  (h3 : x₁ - x₂ = 5.5) : C = 12 := 
sorry

end constant_in_quadratic_eq_l213_213742


namespace base10_to_base8_440_l213_213230

theorem base10_to_base8_440 :
  ∃ k1 k2 k3,
    k1 = 6 ∧
    k2 = 7 ∧
    k3 = 0 ∧
    (440 = k1 * 64 + k2 * 8 + k3) ∧
    (64 = 8^2) ∧
    (8^3 > 440) :=
sorry

end base10_to_base8_440_l213_213230


namespace magic_8_ball_probability_l213_213839

theorem magic_8_ball_probability :
  let p_pos := 1 / 3
  let p_neg := 2 / 3
  let n := 6
  let k := 3
  (Nat.choose n k * (p_pos ^ k) * (p_neg ^ (n - k)) = 160 / 729) :=
by
  sorry

end magic_8_ball_probability_l213_213839


namespace ellipse_foci_coordinates_l213_213402

theorem ellipse_foci_coordinates :
  ∃ x y : Real, (3 * x^2 + 4 * y^2 = 12) ∧ ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0)) :=
by
  sorry

end ellipse_foci_coordinates_l213_213402


namespace total_computers_needed_l213_213340

theorem total_computers_needed
    (initial_students : ℕ)
    (students_per_computer : ℕ)
    (additional_students : ℕ)
    (initial_computers : ℕ := initial_students / students_per_computer)
    (total_computers : ℕ := initial_computers + (additional_students / students_per_computer))
    (h1 : initial_students = 82)
    (h2 : students_per_computer = 2)
    (h3 : additional_students = 16) :
    total_computers = 49 :=
by
  -- The proof would normally go here
  sorry

end total_computers_needed_l213_213340


namespace max_discount_l213_213049

-- Definitions:
def cost_price : ℝ := 400
def sale_price : ℝ := 600
def desired_profit_margin : ℝ := 0.05

-- Statement:
theorem max_discount 
  (x : ℝ) 
  (hx : sale_price * (1 - x / 100) ≥ cost_price * (1 + desired_profit_margin)) :
  x ≤ 90 := 
sorry

end max_discount_l213_213049


namespace units_digit_G_1000_l213_213154

def modified_fermat_number (n : ℕ) : ℕ := 5^(5^n) + 6

theorem units_digit_G_1000 : (modified_fermat_number 1000) % 10 = 1 :=
by
  -- The proof goes here
  sorry

end units_digit_G_1000_l213_213154


namespace percentage_slump_in_business_l213_213321

theorem percentage_slump_in_business (X Y : ℝ) (h1 : 0.04 * X = 0.05 * Y) : 
  (1 - Y / X) * 100 = 20 :=
by
  sorry

end percentage_slump_in_business_l213_213321


namespace correct_statements_count_l213_213389

-- Definitions
def proper_fraction (x : ℚ) : Prop := (0 < x) ∧ (x < 1)
def improper_fraction (x : ℚ) : Prop := (x ≥ 1)

-- Statements as conditions
def statement1 (a b : ℚ) : Prop := proper_fraction a ∧ proper_fraction b → proper_fraction (a + b)
def statement2 (a b : ℚ) : Prop := proper_fraction a ∧ proper_fraction b → proper_fraction (a * b)
def statement3 (a b : ℚ) : Prop := proper_fraction a ∧ improper_fraction b → improper_fraction (a + b)
def statement4 (a b : ℚ) : Prop := proper_fraction a ∧ improper_fraction b → improper_fraction (a * b)

-- The main theorem stating the correct answer
theorem correct_statements_count : 
  (¬ (∀ a b, statement1 a b)) ∧ 
  (∀ a b, statement2 a b) ∧ 
  (∀ a b, statement3 a b) ∧ 
  (¬ (∀ a b, statement4 a b)) → 
  (2 = 2)
:= by sorry

end correct_statements_count_l213_213389


namespace trig_order_l213_213266

theorem trig_order (θ : ℝ) (h1 : -Real.pi / 8 < θ) (h2 : θ < 0) : Real.tan θ < Real.sin θ ∧ Real.sin θ < Real.cos θ := 
sorry

end trig_order_l213_213266


namespace distance_to_directrix_l213_213681

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 64 + y^2 / 28 = 1

noncomputable def left_focus : ℝ × ℝ := (-6, 0)

noncomputable def right_focus : ℝ × ℝ := (6, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem distance_to_directrix (P : ℝ × ℝ) (hP : ellipse P.1 P.2) (hPF1 : distance P left_focus = 4) :
  distance P right_focus * 4 / 3 = 16 :=
sorry

end distance_to_directrix_l213_213681


namespace problem1_l213_213098

theorem problem1 (a b : ℝ) (ha : a > 2) (hb : b > 2) :
  (a - 2) * (b - 2) = 2 :=
sorry

end problem1_l213_213098


namespace simplify_fraction_l213_213731

-- Define the numbers involved and state their GCD
def num1 := 90
def num2 := 8100

-- State the GCD condition using a Lean 4 statement
def gcd_condition (a b : ℕ) := Nat.gcd a b = 90

-- Define the original fraction and the simplified fraction
def original_fraction := num1 / num2
def simplified_fraction := 1 / 90

-- State the proof problem that the original fraction simplifies to the simplified fraction
theorem simplify_fraction : gcd_condition num1 num2 → original_fraction = simplified_fraction := 
by
  sorry

end simplify_fraction_l213_213731


namespace triangle_right_angle_AB_solution_l213_213561

theorem triangle_right_angle_AB_solution (AC BC AB : ℝ) (hAC : AC = 6) (hBC : BC = 8) :
  (AC^2 + BC^2 = AB^2 ∨ AB^2 + AC^2 = BC^2) ↔ (AB = 10 ∨ AB = 2 * Real.sqrt 7) :=
by
  sorry

end triangle_right_angle_AB_solution_l213_213561


namespace expected_people_with_condition_l213_213450

noncomputable def proportion_of_condition := 1 / 3
def total_population := 450

theorem expected_people_with_condition :
  (proportion_of_condition * total_population) = 150 := by
  sorry

end expected_people_with_condition_l213_213450


namespace birthday_pizza_problem_l213_213526

theorem birthday_pizza_problem (m : ℕ) (h1 : m > 11) (h2 : 55 % m = 0) : 10 + 55 / m = 13 := by
  sorry

end birthday_pizza_problem_l213_213526


namespace max_neg_p_l213_213629

theorem max_neg_p (p : ℤ) (h1 : p < 0) (h2 : ∃ k : ℤ, 2001 + p = k^2) : p ≤ -65 :=
by
  sorry

end max_neg_p_l213_213629


namespace y_minus_x_value_l213_213006

theorem y_minus_x_value (x y : ℝ) (h1 : x + y = 500) (h2 : x / y = 0.8) : y - x = 55.56 :=
sorry

end y_minus_x_value_l213_213006


namespace chips_recoloring_impossible_l213_213330

theorem chips_recoloring_impossible :
  (∀ a b c : ℕ, a = 2008 ∧ b = 2009 ∧ c = 2010 →
   ¬(∃ k : ℕ, a + b + c = k ∧ (a = k ∨ b = k ∨ c = k))) :=
by sorry

end chips_recoloring_impossible_l213_213330


namespace value_of_a_l213_213936

-- Define the sets A and B and the intersection condition
def A (a : ℝ) : Set ℝ := {a ^ 2, a + 1, -3}
def B (a : ℝ) : Set ℝ := {a - 3, 2 * a - 1, a ^ 2 + 1}

theorem value_of_a (a : ℝ) (h : A a ∩ B a = {-3}) : a = -1 :=
by {
  -- Insert proof here when ready, using h to show a = -1
  sorry
}

end value_of_a_l213_213936


namespace pentagon_area_greater_than_square_third_l213_213789

theorem pentagon_area_greater_than_square_third (a b : ℝ) :
  a^2 + (a * b) / 4 + (Real.sqrt 3 / 4) * b^2 > ((a + b)^2) / 3 :=
by
  sorry

end pentagon_area_greater_than_square_third_l213_213789


namespace evaluate_expression_l213_213374

theorem evaluate_expression : -(16 / 4 * 7 - 50 + 5 * 7) = -13 :=
by
  sorry

end evaluate_expression_l213_213374


namespace solution_set_inequality_l213_213745

theorem solution_set_inequality (x : ℝ) : x * (x - 1) > 0 ↔ x < 0 ∨ x > 1 :=
by sorry

end solution_set_inequality_l213_213745


namespace newspaper_pages_l213_213545

theorem newspaper_pages (p : ℕ) (h₁ : p >= 21) (h₂ : 8•2 - 1 ≤ p) (h₃ : p ≤ 8•3) : p = 28 :=
sorry

end newspaper_pages_l213_213545


namespace total_legs_l213_213455

-- Define the number of octopuses
def num_octopuses : ℕ := 5

-- Define the number of legs per octopus
def legs_per_octopus : ℕ := 8

-- The total number of legs should be num_octopuses * legs_per_octopus
theorem total_legs : num_octopuses * legs_per_octopus = 40 :=
by
  -- The proof is omitted
  sorry

end total_legs_l213_213455


namespace fraction_nonnegative_for_all_reals_l213_213732

theorem fraction_nonnegative_for_all_reals (x : ℝ) : 
  (x^2 + 2 * x + 1) / (x^2 + 4 * x + 8) ≥ 0 :=
by
  sorry

end fraction_nonnegative_for_all_reals_l213_213732


namespace quadratic_roots_eq_k_quadratic_inequality_k_range_l213_213244

theorem quadratic_roots_eq_k (k : ℝ) (h1 : k > 0)
  (h2 : ∀ x : ℝ, 2 < x → x < 3 → k*x^2 - 2*x + 6*k < 0)
  (h3: (2 + 3) = (2/k)) : k = 2/5 :=
by sorry

theorem quadratic_inequality_k_range (k : ℝ) 
  (h1 : k > 0)
  (h2 : ∀ x : ℝ, 2 < x → x < 3 → k*x^2 - 2*x + 6*k < 0) 
: 0 < k ∧ k <= 2/5 :=
by sorry

end quadratic_roots_eq_k_quadratic_inequality_k_range_l213_213244


namespace surface_area_of_box_l213_213205

def cube_edge_length : ℕ := 1
def cubes_required : ℕ := 12

theorem surface_area_of_box (l w h : ℕ) (h1 : l * w * h = cubes_required / cube_edge_length ^ 3) :
  (2 * (l * w + w * h + h * l) = 32 ∨ 2 * (l * w + w * h + h * l) = 38 ∨ 2 * (l * w + w * h + h * l) = 40) :=
  sorry

end surface_area_of_box_l213_213205


namespace arithmetic_sequence_problem_l213_213574

variables (a_n b_n : ℕ → ℚ)
variables (S_n T_n : ℕ → ℚ)
variable (n : ℕ)

axiom sum_a_terms : ∀ n : ℕ, S_n n = n / 2 * (a_n 1 + a_n n)
axiom sum_b_terms : ∀ n : ℕ, T_n n = n / 2 * (b_n 1 + b_n n)
axiom given_fraction : ∀ n : ℕ, n > 0 → S_n n / T_n n = (2 * n + 1) / (4 * n - 2)

theorem arithmetic_sequence_problem : 
  (a_n 10) / (b_n 3 + b_n 18) + (a_n 11) / (b_n 6 + b_n 15) = 41 / 78 :=
sorry

end arithmetic_sequence_problem_l213_213574


namespace ceiling_of_neg_3_7_l213_213075

theorem ceiling_of_neg_3_7 : Int.ceil (-3.7) = -3 := by
  sorry

end ceiling_of_neg_3_7_l213_213075


namespace new_energy_vehicle_sales_growth_l213_213028

theorem new_energy_vehicle_sales_growth (x : ℝ) :
  let sales_jan := 64
  let sales_feb := 64 * (1 + x)
  let sales_mar := 64 * (1 + x)^2
  (sales_jan + sales_feb + sales_mar = 244) :=
sorry

end new_energy_vehicle_sales_growth_l213_213028


namespace car_dealership_sales_l213_213063

theorem car_dealership_sales (x : ℕ)
  (h1 : 5 * x = 30 * 8)
  (h2 : 30 + x = 78) : 
  x = 48 :=
sorry

end car_dealership_sales_l213_213063


namespace percentage_of_x_l213_213831

variable {x y : ℝ}
variable {P : ℝ}

theorem percentage_of_x (h1 : (P / 100) * x = (20 / 100) * y) (h2 : x / y = 2) : P = 10 := by
  sorry

end percentage_of_x_l213_213831


namespace number_of_divisors_of_36_l213_213116

/-- The number of integers (positive and negative) that are divisors of 36 is 18. -/
theorem number_of_divisors_of_36 : 
  let positive_divisors := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  in 2 * positive_divisors.card = 18 :=
by
  let positive_divisors := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  have h : positive_divisors.card = 9 := sorry
  show 2 * positive_divisors.card = 18
  by rw [h]; norm_num
  sorry

end number_of_divisors_of_36_l213_213116


namespace mixed_number_division_l213_213919

theorem mixed_number_division :
  (5 + 1 / 2 - (2 + 2 / 3)) / (1 + 1 / 5 + 3 + 1 / 4) = 0 + 170 / 267 := 
by
  sorry

end mixed_number_division_l213_213919


namespace t_f_3_equals_sqrt_44_l213_213997

noncomputable def t (x : ℝ) : ℝ := Real.sqrt (4 * x + 4)
noncomputable def f (x : ℝ) : ℝ := 6 + t x

theorem t_f_3_equals_sqrt_44 : t (f 3) = Real.sqrt 44 := by
  sorry

end t_f_3_equals_sqrt_44_l213_213997


namespace total_volume_l213_213051

open Real

noncomputable def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h
noncomputable def volume_cone (r h : ℝ) : ℝ := (1/3) * π * r^2 * h

theorem total_volume {d_cylinder d_cone_top d_cone_bottom h_cylinder h_cone : ℝ}
  (h1 : d_cylinder = 2) (h2 : d_cone_top = 2) (h3 : d_cone_bottom = 1)
  (h4 : h_cylinder = 14) (h5 : h_cone = 4) :
  volume_cylinder (d_cylinder / 2) h_cylinder +
  volume_cone (d_cone_top / 2) h_cone =
  (46 / 3) * π :=
by
  sorry

end total_volume_l213_213051


namespace find_ABC_l213_213173

noncomputable def g (x : ℝ) (A B C : ℝ) : ℝ := x^2 / (A * x^2 + B * x + C)

theorem find_ABC : 
  ∀ (A B C : ℝ),
  (∀ (x : ℝ), x > 2 → g x A B C > 0.3) →
  (∃ (A : ℤ), A = 4) →
  (∃ (B : ℤ), ∃ (C : ℤ), A = 4 ∧ B = 8 ∧ C = -12) →
  A + B + C = 0 :=
by
  intros A B C h1 h2 h3
  rcases h2 with ⟨intA, h2'⟩
  rcases h3 with ⟨intB, ⟨intC, h3'⟩⟩
  simp [h2', h3']
  sorry -- proof skipped

end find_ABC_l213_213173


namespace complement_intersection_eq_l213_213823

-- Definitions of sets
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {2, 5, 8}
def B : Set ℕ := {1, 3, 5, 7}

-- Definition of complement of A in U
def complement_U_A : Set ℕ := U \ A

-- The main statement to prove
theorem complement_intersection_eq :
  (complement_U_A ∩ B) = {1, 3, 7} :=
by sorry

end complement_intersection_eq_l213_213823


namespace initial_rope_length_l213_213209

theorem initial_rope_length : 
  ∀ (π : ℝ), 
  ∀ (additional_area : ℝ) (new_rope_length : ℝ), 
  additional_area = 933.4285714285714 →
  new_rope_length = 21 →
  ∃ (initial_rope_length : ℝ), 
  additional_area = π * (new_rope_length^2 - initial_rope_length^2) ∧
  initial_rope_length = 12 :=
by
  sorry

end initial_rope_length_l213_213209


namespace ammonium_nitrate_formed_l213_213082

-- Definitions based on conditions in the problem
def NH3_moles : ℕ := 3
def HNO3_moles (NH3 : ℕ) : ℕ := NH3 -- 1:1 molar ratio with NH3 for HNO3

-- Definition of the outcome
def NH4NO3_moles (NH3 NH4NO3 : ℕ) : Prop :=
  NH4NO3 = NH3

-- The theorem to prove that 3 moles of NH3 combined with sufficient HNO3 produces 3 moles of NH4NO3
theorem ammonium_nitrate_formed (NH3 NH4NO3 : ℕ) (h : NH3 = 3) :
  NH4NO3_moles NH3 NH4NO3 → NH4NO3 = 3 :=
by
  intro hn
  rw [h] at hn
  exact hn

end ammonium_nitrate_formed_l213_213082


namespace exp_base_lt_imp_cube_l213_213396

theorem exp_base_lt_imp_cube (a x y : ℝ) (h_a : 0 < a) (h_a1 : a < 1) (h_exp : a^x > a^y) : x^3 < y^3 :=
by
  sorry

end exp_base_lt_imp_cube_l213_213396


namespace decks_left_is_3_l213_213053

-- Given conditions
def price_per_deck := 2
def total_decks_start := 5
def money_earned := 4

-- The number of decks sold
def decks_sold := money_earned / price_per_deck

-- The number of decks left
def decks_left := total_decks_start - decks_sold

-- The theorem to prove 
theorem decks_left_is_3 : decks_left = 3 :=
by
  -- Here we put the steps to prove
  sorry

end decks_left_is_3_l213_213053


namespace arccos_cos_7_l213_213517

noncomputable def arccos_cos_7_eq_7_minus_2pi : Prop :=
  ∃ x : ℝ, x = 7 - 2 * Real.pi ∧ Real.arccos (Real.cos 7) = x

theorem arccos_cos_7 :
  arccos_cos_7_eq_7_minus_2pi :=
by
  sorry

end arccos_cos_7_l213_213517


namespace dilation_image_l213_213172

open Complex

noncomputable def dilation_center := (1 : ℂ) + (3 : ℂ) * I
noncomputable def scale_factor := -3
noncomputable def initial_point := -I
noncomputable def target_point := (4 : ℂ) + (15 : ℂ) * I

theorem dilation_image :
  let c := dilation_center
  let k := scale_factor
  let z := initial_point
  let z_prime := target_point
  z_prime = c + k * (z - c) := 
  by
    sorry

end dilation_image_l213_213172


namespace days_in_week_l213_213070

theorem days_in_week {F D : ℕ} (h1 : F = 3 + 11) (h2 : F = 2 * D) : D = 7 :=
by
  sorry

end days_in_week_l213_213070


namespace hot_dog_cost_l213_213916

variable {Real : Type} [LinearOrderedField Real]

-- Define the cost of a hamburger and a hot dog
variables (h d : Real)

-- Arthur's buying conditions
def condition1 := 3 * h + 4 * d = 10
def condition2 := 2 * h + 3 * d = 7

-- Problem statement: Proving that the cost of a hot dog is 1 dollar
theorem hot_dog_cost
    (h d : Real)
    (hc1 : condition1 h d)
    (hc2 : condition2 h d) : 
    d = 1 :=
sorry

end hot_dog_cost_l213_213916


namespace max_servings_l213_213358

-- Definitions based on the conditions
def servings_recipe := 3
def bananas_per_serving := 2 / servings_recipe
def strawberries_per_serving := 1 / servings_recipe
def yogurt_per_serving := 2 / servings_recipe

def emily_bananas := 4
def emily_strawberries := 3
def emily_yogurt := 6

-- Prove that Emily can make at most 6 servings while keeping the proportions the same
theorem max_servings :
  min (emily_bananas / bananas_per_serving) 
      (min (emily_strawberries / strawberries_per_serving) 
           (emily_yogurt / yogurt_per_serving)) = 6 := sorry

end max_servings_l213_213358


namespace number_of_groups_of_bananas_l213_213468

theorem number_of_groups_of_bananas (total_bananas : ℕ) (bananas_per_group : ℕ) (H_total_bananas : total_bananas = 290) (H_bananas_per_group : bananas_per_group = 145) :
    (total_bananas / bananas_per_group) = 2 :=
by {
  sorry
}

end number_of_groups_of_bananas_l213_213468


namespace find_root_of_equation_l213_213043

theorem find_root_of_equation (a b c d x : ℕ) (h_ad : a + d = 2016) (h_bc : b + c = 2016) (h_ac : a ≠ c) :
  (x - a) * (x - b) = (x - c) * (x - d) → x = 1008 :=
by
  sorry

end find_root_of_equation_l213_213043


namespace solve_base_r_l213_213448

theorem solve_base_r (r : ℕ) (hr : Even r) (x : ℕ) (hx : x = 9999) 
                     (palindrome_condition : ∃ (a b c d : ℕ), 
                      b + c = 24 ∧ 
                      ∀ (r_repr : List ℕ), 
                      r_repr.length = 8 ∧
                      r_repr = [a, b, c, d, d, c, b, a] ∧ 
                      ∃ x_squared_repr, x^2 = x_squared_repr) : r = 26 :=
by
  sorry

end solve_base_r_l213_213448


namespace ratio_closest_to_one_l213_213601

-- Define the entrance fee for adults and children.
def adult_fee : ℕ := 20
def child_fee : ℕ := 15

-- Define the total collected amount.
def total_collected : ℕ := 2400

-- Define the number of adults and children.
variables (a c : ℕ)

-- The main theorem to prove:
theorem ratio_closest_to_one 
  (h1 : a > 0) -- at least one adult
  (h2 : c > 0) -- at least one child
  (h3 : adult_fee * a + child_fee * c = total_collected) : 
  a / (c : ℚ) = 69 / 68 := 
sorry

end ratio_closest_to_one_l213_213601


namespace number_of_boys_l213_213595

theorem number_of_boys 
    (B : ℕ) 
    (total_boys_sticks : ℕ := 15 * B)
    (total_girls_sticks : ℕ := 12 * 12)
    (sticks_relation : total_girls_sticks = total_boys_sticks - 6) : 
    B = 10 :=
by
    sorry

end number_of_boys_l213_213595


namespace three_digit_numbers_with_4_and_5_correct_l213_213125

def count_three_digit_numbers_with_4_and_5 : ℕ :=
  48

theorem three_digit_numbers_with_4_and_5_correct :
  count_three_digit_numbers_with_4_and_5 = 48 :=
by
  sorry -- proof goes here

end three_digit_numbers_with_4_and_5_correct_l213_213125


namespace leg_ratio_of_right_triangle_l213_213892

theorem leg_ratio_of_right_triangle (a b c m : ℝ) (h1 : a ≤ b)
  (h2 : a * b = c * m) (h3 : c^2 = a^2 + b^2) (h4 : a^2 + m^2 = b^2) :
  (a / b) = Real.sqrt ((-1 + Real.sqrt 5) / 2) :=
sorry

end leg_ratio_of_right_triangle_l213_213892


namespace binary_to_decimal_110011_l213_213922

theorem binary_to_decimal_110011 :
  1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 = 51 :=
by
  sorry

end binary_to_decimal_110011_l213_213922


namespace train_length_l213_213757

theorem train_length
  (speed_kmph : ℕ)
  (platform_length : ℕ)
  (time_seconds : ℕ)
  (speed_m_per_s : speed_kmph * 1000 / 3600 = 20)
  (distance_covered : 20 * time_seconds = 520)
  (platform_eq : platform_length = 280)
  (time_eq : time_seconds = 26) :
  ∃ L : ℕ, L = 240 := by
  sorry

end train_length_l213_213757


namespace fraction_of_menu_l213_213724

def total_dishes (total : ℕ) : Prop := 
  6 = (1/4:ℚ) * total

def vegan_dishes (vegan : ℕ) (soy_free : ℕ) : Prop :=
  vegan = 6 ∧ soy_free = vegan - 5

theorem fraction_of_menu (total vegan soy_free : ℕ) (h1 : total_dishes total)
  (h2 : vegan_dishes vegan soy_free) : (soy_free:ℚ) / total = 1 / 24 := 
by sorry

end fraction_of_menu_l213_213724


namespace jelly_bean_match_probability_l213_213499

theorem jelly_bean_match_probability :
  let abe_total := 4 
  let bob_total := 8
  let abe_green := 2 
  let abe_red := 1 
  let abe_blue := 1
  let bob_green := 3 
  let bob_yellow := 2 
  let bob_blue := 1 
  let bob_red := 2
  let prob_green := (abe_green / abe_total) * (bob_green / bob_total)
  let prob_blue := (abe_blue / abe_total) * (bob_blue / bob_total)
  let prob_red := (abe_red / abe_total) * (bob_red / bob_total) in
  prob_green + prob_blue + prob_red = 9 / 32 :=
by
  sorry

end jelly_bean_match_probability_l213_213499


namespace seismic_activity_mismatch_percentage_l213_213420

theorem seismic_activity_mismatch_percentage
  (total_days : ℕ)
  (quiet_days_percentage : ℝ)
  (prediction_accuracy : ℝ)
  (predicted_quiet_days_percentage : ℝ)
  (quiet_prediction_correctness : ℝ)
  (active_days_percentage : ℝ)
  (incorrect_quiet_predictions : ℝ) :
  quiet_days_percentage = 0.8 →
  predicted_quiet_days_percentage = 0.64 →
  quiet_prediction_correctness = 0.7 →
  active_days_percentage = 0.2 →
  incorrect_quiet_predictions = predicted_quiet_days_percentage - (quiet_prediction_correctness * quiet_days_percentage) →
  (incorrect_quiet_predictions / active_days_percentage) * 100 = 40 := by
  sorry

end seismic_activity_mismatch_percentage_l213_213420


namespace equal_areas_of_shapes_l213_213762

noncomputable def hexagon_area (s : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * s^2

noncomputable def semicircle_area (r : ℝ) : ℝ :=
  (Real.pi * r^2) / 2

noncomputable def sector_area (theta : ℝ) (r : ℝ) : ℝ :=
  (theta / (2 * Real.pi)) * Real.pi * r^2

noncomputable def shape1_area (s : ℝ) : ℝ :=
  hexagon_area s + 3 * semicircle_area (s / 4) - 6 * sector_area (Real.pi / 3) (s / 4)

noncomputable def shape2_area (s : ℝ) : ℝ :=
  hexagon_area s + 6 * sector_area (2 * Real.pi / 3) (s / 4) - 3 * semicircle_area (s / 4)

theorem equal_areas_of_shapes (s : ℝ) : shape1_area s = shape2_area s :=
by {
  sorry
}

end equal_areas_of_shapes_l213_213762


namespace prob_at_least_one_hit_correct_prob_exactly_two_hits_correct_l213_213312

-- Define the probabilities that A, B, and C hit the target
def prob_A := 0.7
def prob_B := 0.6
def prob_C := 0.5

-- Define the probabilities that A, B, and C miss the target
def miss_A := 1 - prob_A
def miss_B := 1 - prob_B
def miss_C := 1 - prob_C

-- Probability that no one hits the target
def prob_no_hits := miss_A * miss_B * miss_C

-- Probability that at least one person hits the target
def prob_at_least_one_hit := 1 - prob_no_hits

-- Probabilities for the cases where exactly two people hit the target:
def prob_A_B_hits := prob_A * prob_B * miss_C
def prob_A_C_hits := prob_A * miss_B * prob_C
def prob_B_C_hits := miss_A * prob_B * prob_C

-- Probability that exactly two people hit the target
def prob_exactly_two_hits := prob_A_B_hits + prob_A_C_hits + prob_B_C_hits

-- Theorem statement to prove the probabilities match given conditions
theorem prob_at_least_one_hit_correct : prob_at_least_one_hit = 0.94 := by
  sorry

theorem prob_exactly_two_hits_correct : prob_exactly_two_hits = 0.44 := by
  sorry

end prob_at_least_one_hit_correct_prob_exactly_two_hits_correct_l213_213312


namespace section_Diligence_students_before_transfer_l213_213278

-- Define the variables
variables (D_after I_after D_before : ℕ)

-- Problem Statement
theorem section_Diligence_students_before_transfer :
  ∀ (D_after I_after: ℕ),
    2 + D_after = I_after
    ∧ D_after + I_after = 50 →
    ∃ D_before, D_before = D_after - 2 ∧ D_before = 23 :=
by
sorrry

end section_Diligence_students_before_transfer_l213_213278


namespace values_of_a_l213_213261

noncomputable def M : Set ℝ := {x | x^2 = 1}

noncomputable def N (a : ℝ) : Set ℝ := 
  if a = 0 then ∅ else {x | a * x = 1}

theorem values_of_a (a : ℝ) : (N a ⊆ M) ↔ (a = -1 ∨ a = 0 ∨ a = 1) := by
  sorry

end values_of_a_l213_213261


namespace problem_solution_l213_213072

noncomputable def question (x y z : ℝ) : Prop := 
  (x ≠ y ∧ y ≠ z ∧ z ≠ x) → 
  ((x + y)/(x^2 + x*y + y^2) = (y + z)/(y^2 + y*z + z^2) ∨ 
   (y + z)/(y^2 + y*z + z^2) = (z + x)/(z^2 + z*x + x^2) ∨ 
   (z + x)/(z^2 + z*x + x^2) = (x + y)/(x^2 + x*y + y^2)) → 
  ( (x + y)/(x^2 + x*y + y^2) = (y + z)/(y^2 + y*z + z^2) ∧ 
    (y + z)/(y^2 + y*z + z^2) = (z + x)/(z^2 + z*x + x^2) )

theorem problem_solution (x y z : ℝ) (h : x ≠ y ∧ y ≠ z ∧ z ≠ x) :
  ((x + y)/(x^2 + x*y + y^2) = (y + z)/(y^2 + y*z + z^2) ∨ 
   (y + z)/(y^2 + y*z + z^2) = (z + x)/(z^2 + z*x + x^2) ∨ 
   (z + x)/(z^2 + z*x + x^2) = (x + y)/(x^2 + x*y + y^2)) →
  ( (x + y)/(x^2 + x*y + y^2) = (y + z)/(y^2 + y*z + z^2) ∧ 
    (y + z)/(y^2 + y*z + z^2) = (z + x)/(z^2 + z*x + x^2) ) :=
sorry

end problem_solution_l213_213072


namespace tan_expression_value_l213_213537

noncomputable def sequence_properties (a b : ℕ → ℝ) :=
  (a 0 * a 5 * a 10 = -3 * Real.sqrt 3) ∧
  (b 0 + b 5 + b 10 = 7 * Real.pi) ∧
  (∀ n, a (n + 1) = a n * a 1) ∧
  (∀ n, b (n + 1) = b n + (b 1 - b 0))

theorem tan_expression_value (a b : ℕ → ℝ) (h : sequence_properties a b) :
  Real.tan (b 2 + b 8) / (1 - a 3 * a 7) = -Real.sqrt 3 :=
sorry

end tan_expression_value_l213_213537


namespace unit_digit_product_l213_213894

theorem unit_digit_product (n1 n2 n3 : ℕ) (a : ℕ) (b : ℕ) (c : ℕ) :
  (n1 = 68) ∧ (n2 = 59) ∧ (n3 = 71) ∧ (a = 3) ∧ (b = 6) ∧ (c = 7) →
  (a ^ n1 * b ^ n2 * c ^ n3) % 10 = 8 := by
  sorry

end unit_digit_product_l213_213894


namespace smallest_nonneg_integer_divisible_by_4_l213_213621

theorem smallest_nonneg_integer_divisible_by_4 :
  ∃ n : ℕ, (7 * (n - 3)^5 - n^2 + 16 * n - 30) % 4 = 0 ∧ ∀ m : ℕ, m < n -> (7 * (m - 3)^5 - m^2 + 16 * m - 30) % 4 ≠ 0 :=
by
  use 1
  sorry

end smallest_nonneg_integer_divisible_by_4_l213_213621


namespace students_paid_half_l213_213131

theorem students_paid_half (F H : ℕ) 
  (h1 : F + H = 25)
  (h2 : 50 * F + 25 * H = 1150) : 
  H = 4 := by
  sorry

end students_paid_half_l213_213131


namespace noah_yearly_bill_l213_213861

-- Define the length of each call in minutes
def call_duration : ℕ := 30

-- Define the cost per minute in dollars
def cost_per_minute : ℝ := 0.05

-- Define the number of weeks in a year
def weeks_in_year : ℕ := 52

-- Define the cost per call in dollars
def cost_per_call : ℝ := call_duration * cost_per_minute

-- Define the total cost for a year in dollars
def yearly_cost : ℝ := cost_per_call * weeks_in_year

-- State the theorem
theorem noah_yearly_bill : yearly_cost = 78 := by
  -- Proof follows here
  sorry

end noah_yearly_bill_l213_213861


namespace determine_k_l213_213635

theorem determine_k (k : ℚ) (h_collinear : ∃ (f : ℚ → ℚ), 
  f 0 = 3 ∧ f 7 = k ∧ f 21 = 2) : k = 8 / 3 :=
by
  sorry

end determine_k_l213_213635


namespace meal_cost_l213_213842

theorem meal_cost (total_paid change tip_rate : ℝ)
  (h_total_paid : total_paid = 20 - change)
  (h_change : change = 5)
  (h_tip_rate : tip_rate = 0.2) :
  ∃ x, x + tip_rate * x = total_paid ∧ x = 12.5 := 
by
  sorry

end meal_cost_l213_213842


namespace inequalities_not_hold_l213_213525

theorem inequalities_not_hold (x y z a b c : ℝ) (h1 : x < a) (h2 : y < b) (h3 : z < c) : 
  ¬ (x * y + y * z + z * x < a * b + b * c + c * a) ∧ 
  ¬ (x^2 + y^2 + z^2 < a^2 + b^2 + c^2) ∧ 
  ¬ (x * y * z < a * b * c) := 
sorry

end inequalities_not_hold_l213_213525


namespace compute_division_l213_213227

theorem compute_division : 0.182 / 0.0021 = 86 + 14 / 21 :=
by
  sorry

end compute_division_l213_213227


namespace find_coordinates_of_P0_find_equation_of_l_l213_213256

noncomputable def curve (x : ℝ) : ℝ := x^3 + x - 2

def tangent_slope (x : ℝ) : ℝ := 3 * x^2 + 1

def is_in_third_quadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0

def line_eq (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

/-- Problem statement 1: Find the coordinates of P₀ --/
theorem find_coordinates_of_P0 (p0 : ℝ × ℝ)
    (h_tangent_parallel : tangent_slope p0.1 = 4)
    (h_third_quadrant : is_in_third_quadrant p0) :
    p0 = (-1, -4) :=
sorry

/-- Problem statement 2: Find the equation of line l --/
theorem find_equation_of_l (P0 : ℝ × ℝ)
    (h_P0_coordinates: P0 = (-1, -4))
    (h_perpendicular : ∀ (l1_slope : ℝ), l1_slope = 4 → ∃ l_slope : ℝ, l_slope = (-1) / 4)
    (x y : ℝ) : 
    line_eq 1 4 17 x y :=
sorry

end find_coordinates_of_P0_find_equation_of_l_l213_213256


namespace selected_number_in_first_group_is_7_l213_213493

def N : ℕ := 800
def k : ℕ := 50
def interval : ℕ := N / k
def selected_number : ℕ := 39
def second_group_start : ℕ := 33
def second_group_end : ℕ := 48

theorem selected_number_in_first_group_is_7 
  (h1 : interval = 16)
  (h2 : selected_number ≥ second_group_start ∧ selected_number ≤ second_group_end)
  (h3 : ∃ n, selected_number = second_group_start + interval * n - 1) :
  selected_number % interval = 7 :=
sorry

end selected_number_in_first_group_is_7_l213_213493


namespace compute_trig_expr_l213_213516

theorem compute_trig_expr :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 2 * Real.sec (10 * Real.pi / 180) :=
by
  sorry

end compute_trig_expr_l213_213516


namespace ratio_of_other_triangle_l213_213014

noncomputable def ratioAreaOtherTriangle (m : ℝ) : ℝ := 1 / (4 * m)

theorem ratio_of_other_triangle (m : ℝ) (h : m > 0) : ratioAreaOtherTriangle m = 1 / (4 * m) :=
by
  -- Proof will be provided here
  sorry

end ratio_of_other_triangle_l213_213014


namespace sufficient_condition_l213_213392

theorem sufficient_condition (A B : Set α) (h : A ⊆ B) (x : α) : x ∈ A → x ∈ B :=
  by
    intro h1
    apply h
    exact h1

end sufficient_condition_l213_213392


namespace students_remaining_after_fifth_stop_l213_213962

theorem students_remaining_after_fifth_stop (initial_students : ℕ) (stops : ℕ) :
  initial_students = 60 →
  stops = 5 →
  (∀ n, (n < stops → ∃ k, n = 3 * k + 1) → ∀ x, x = initial_students * ((2 : ℚ) / 3)^stops) →
  initial_students * ((2 : ℚ) / 3)^stops = (640 / 81 : ℚ) :=
by
  intros h_initial h_stops h_formula
  sorry

end students_remaining_after_fifth_stop_l213_213962


namespace range_of_a_l213_213250

open Set

def A (a : ℝ) : Set ℝ := { x | a - 1 ≤ x ∧ x ≤ 2 * a + 1 }
def B : Set ℝ := { x | -2 ≤ x ∧ x ≤ 4 }

theorem range_of_a (a : ℝ) (h : A a ∪ B = B) : a ∈ Iio (-2) ∪ Icc (-1) (3 / 2) :=
by
  sorry

end range_of_a_l213_213250


namespace cost_to_consume_desired_calories_l213_213451

-- conditions
def calories_per_chip : ℕ := 10
def chips_per_bag : ℕ := 24
def cost_per_bag : ℕ := 2
def desired_calories : ℕ := 480

-- proof statement
theorem cost_to_consume_desired_calories :
  let total_calories_per_bag := chips_per_bag * calories_per_chip in
  let bags_needed := desired_calories / total_calories_per_bag in
  let total_cost := bags_needed * cost_per_bag in
  total_cost = 4 :=
by
  sorry

end cost_to_consume_desired_calories_l213_213451


namespace rectangular_prism_volume_increase_l213_213003

theorem rectangular_prism_volume_increase (L B H : ℝ) :
  let V_original := L * B * H
  let L_new := L * 1.07
  let B_new := B * 1.18
  let H_new := H * 1.25
  let V_new := L_new * B_new * H_new
  let increase_in_volume := (V_new - V_original) / V_original * 100
  increase_in_volume = 56.415 :=
by
  sorry

end rectangular_prism_volume_increase_l213_213003


namespace profit_growth_equation_l213_213618

noncomputable def profitApril : ℝ := 250000
noncomputable def profitJune : ℝ := 360000
noncomputable def averageMonthlyGrowth (x : ℝ) : ℝ := 25 * (1 + x) * (1 + x)

theorem profit_growth_equation (x : ℝ) :
  averageMonthlyGrowth x = 36 * 10000 ↔ 25 * (1 + x)^2 = 36 :=
by
  sorry

end profit_growth_equation_l213_213618


namespace trapezoid_base_ratio_l213_213165

theorem trapezoid_base_ratio 
  (a b h : ℝ) 
  (a_gt_b : a > b) 
  (quad_area_cond : (h * (a - b)) / 4 = (h * (a + b)) / 8) : 
  a = 3 * b := 
sorry

end trapezoid_base_ratio_l213_213165


namespace slope_product_constant_l213_213706

noncomputable def parabola_equation (p : ℝ) (hp : p > 0) : Prop :=
  ∀ x y : ℝ, (x ^ 2 = 2 * y ↔ x ^ 2 = 2 * p * y)

theorem slope_product_constant :
  ∀ (x1 y1 x2 y2 k1 k2 : ℝ) (P A B : ℝ × ℝ),
  P = (2, 2) →
  A = (x1, y1) →
  B = (x2, y2) →
  (∀ k: ℝ, y1 = k * (x1 + 2) + 4 ∧ y2 = k * (x2 + 2) + 4) →
  k1 = (y1 - 2) / (x1 - 2) →
  k2 = (y2 - 2) / (x2 - 2) →
  (x1 + x2 = 2 * k) →
  (x1 * x2 = -4 * k - 8) →
  k1 * k2 = -1 := 
  sorry

end slope_product_constant_l213_213706


namespace possible_to_place_12_numbers_on_cube_edges_l213_213985

-- Define a list of numbers from 1 to 12
def nums := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Define the faces of the cube in terms of the indices of nums list
def top_face := [1, 2, 9, 10]
def bottom_face := [3, 5, 6, 8]

-- Define the product of the numbers on the faces of the cube
def product_face (face : List Nat) : Nat := face.foldr (*) 1

-- The lean statement proving the problem
theorem possible_to_place_12_numbers_on_cube_edges :
  product_face (top_face.map (λ i => nums.get! (i - 1))) =
  product_face (bottom_face.map (λ i => nums.get! (i - 1))) :=
by
  sorry

end possible_to_place_12_numbers_on_cube_edges_l213_213985


namespace remainder_5_pow_100_div_18_l213_213348

theorem remainder_5_pow_100_div_18 : (5 ^ 100) % 18 = 13 := 
  sorry

end remainder_5_pow_100_div_18_l213_213348


namespace ordered_pairs_count_l213_213390

theorem ordered_pairs_count : 
  (∀ (b c : ℕ), b > 0 ∧ b ≤ 6 ∧ c > 0 ∧ c ≤ 6 ∧ b^2 - 4 * c < 0 ∧ c^2 - 4 * b < 0 → 
  ((b = 1 ∧ (c = 2 ∨ c = 3 ∨ c = 4 ∨ c = 5 ∨ c = 6)) ∨ 
  (b = 2 ∧ (c = 3 ∨ c = 4 ∨ c = 5 ∨ c = 6)) ∨ 
  (b = 3 ∧ (c = 3 ∨ c = 4 ∨ c = 5 ∨ c = 6)) ∨ 
  (b = 4 ∧ (c = 5 ∨ c = 6)))) ∧
  (∃ (n : ℕ), n = 15) := sorry

end ordered_pairs_count_l213_213390


namespace integral_rational_term_expansion_l213_213140

theorem integral_rational_term_expansion :
  ∫ x in 0.0..1.0, x ^ (1/6 : ℝ) = 6/7 := by
  sorry

end integral_rational_term_expansion_l213_213140


namespace hamburger_count_l213_213109

-- Define the number of condiments and their possible combinations
def condiment_combinations : ℕ := 2 ^ 10

-- Define the number of choices for meat patties
def meat_patties_choices : ℕ := 4

-- Define the total count of different hamburgers
def total_hamburgers : ℕ := condiment_combinations * meat_patties_choices

-- The theorem statement proving the total number of different hamburgers
theorem hamburger_count : total_hamburgers = 4096 := by
  sorry

end hamburger_count_l213_213109


namespace functional_expression_and_range_l213_213946

-- We define the main problem conditions and prove the required statements based on those conditions
theorem functional_expression_and_range (x y : ℝ) (h1 : ∃ k : ℝ, (y + 2) = k * (4 - x) ∧ k ≠ 0)
                                        (h2 : x = 3 → y = 1) :
                                        (y = -3 * x + 10) ∧ ( -2 < y ∧ y < 1 → 3 < x ∧ x < 4) :=
by
  sorry

end functional_expression_and_range_l213_213946


namespace value_of_k_l213_213879

noncomputable def roots_in_ratio_equation {k : ℝ} (h : k ≠ 0) : Prop :=
  ∃ (r₁ r₂ : ℝ), r₁ ≠ 0 ∧ r₂ ≠ 0 ∧ 
  (r₁ / r₂ = 3) ∧ 
  (r₁ + r₂ = -8) ∧ 
  (r₁ * r₂ = k)

theorem value_of_k (k : ℝ) (h : k ≠ 0) (hr : roots_in_ratio_equation h) : k = 12 :=
sorry

end value_of_k_l213_213879


namespace well_depth_is_2000_l213_213638

-- Given conditions
def total_time : ℝ := 10
def stone_law (t₁ : ℝ) : ℝ := 20 * t₁^2
def sound_velocity : ℝ := 1120

-- Statement to be proven
theorem well_depth_is_2000 :
  ∃ (d t₁ t₂ : ℝ), 
    d = stone_law t₁ ∧ t₂ = d / sound_velocity ∧ t₁ + t₂ = total_time :=
sorry

end well_depth_is_2000_l213_213638


namespace karen_total_cost_l213_213145

noncomputable def calculate_total_cost (burger_price sandwich_price smoothie_price : ℝ) (num_smoothies : ℕ)
  (discount_rate tax_rate : ℝ) (order_time : ℕ) : ℝ :=
  let total_cost_before_discount := burger_price + sandwich_price + (num_smoothies * smoothie_price)
  let discount := if total_cost_before_discount > 15 ∧ order_time ≥ 1400 ∧ order_time ≤ 1600 then total_cost_before_discount * discount_rate else 0
  let reduced_price := total_cost_before_discount - discount
  let tax := reduced_price * tax_rate
  reduced_price + tax

theorem karen_total_cost :
  calculate_total_cost 5.75 4.50 4.25 2 0.20 0.12 1545 = 16.80 :=
by
  sorry

end karen_total_cost_l213_213145


namespace heptagon_diagonals_l213_213112

-- Define the number of sides of the polygon
def heptagon_sides : ℕ := 7

-- Define the formula for the number of diagonals of an n-gon
def diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- State the theorem we want to prove, i.e., the number of diagonals in a convex heptagon is 14
theorem heptagon_diagonals : diagonals heptagon_sides = 14 := by
  sorry

end heptagon_diagonals_l213_213112


namespace balance_pitcher_with_saucers_l213_213148

-- Define the weights of the cup (C), pitcher (P), and saucer (S)
variables (C P S : ℝ)

-- Conditions provided in the problem
axiom cond1 : 2 * C + 2 * P = 14 * S
axiom cond2 : P = C + S

-- The statement to prove
theorem balance_pitcher_with_saucers : P = 4 * S :=
by
  sorry

end balance_pitcher_with_saucers_l213_213148


namespace probability_blue_given_glass_l213_213748

-- Defining the various conditions given in the problem
def total_red_balls : ℕ := 5
def total_blue_balls : ℕ := 11
def red_glass_balls : ℕ := 2
def red_wooden_balls : ℕ := 3
def blue_glass_balls : ℕ := 4
def blue_wooden_balls : ℕ := 7
def total_balls : ℕ := total_red_balls + total_blue_balls
def total_glass_balls : ℕ := red_glass_balls + blue_glass_balls

-- The mathematically equivalent proof problem statement.
theorem probability_blue_given_glass :
  (blue_glass_balls : ℚ) / (total_glass_balls : ℚ) = 2 / 3 := by
sorry

end probability_blue_given_glass_l213_213748


namespace total_pages_of_book_l213_213906

-- Definitions for the conditions
def firstChapterPages : Nat := 66
def secondChapterPages : Nat := 35
def thirdChapterPages : Nat := 24

-- Theorem stating the main question and answer
theorem total_pages_of_book : firstChapterPages + secondChapterPages + thirdChapterPages = 125 := by
  -- Proof will be provided here
  sorry

end total_pages_of_book_l213_213906


namespace cube_root_simplification_l213_213336

theorem cube_root_simplification {a b : ℕ} (h : (a * b^(1/3) : ℝ) = (2450 : ℝ)^(1/3)) 
  (a_pos : 0 < a) (b_pos : 0 < b) (h_smallest : ∀ b', 0 < b' → (∃ a', (a' * b'^(1/3) : ℝ) = (2450 : ℝ)^(1/3) → b ≤ b')) :
  a + b = 37 := 
sorry

end cube_root_simplification_l213_213336


namespace pirates_share_l213_213480

def initial_coins (N : ℕ) := N ≥ 3000 ∧ N ≤ 4000

def first_pirate (N : ℕ) := N - (2 + (N - 2) / 4)
def second_pirate (remaining : ℕ) := remaining - (2 + (remaining - 2) / 4)
def third_pirate (remaining : ℕ) := remaining - (2 + (remaining - 2) / 4)
def fourth_pirate (remaining : ℕ) := remaining - (2 + (remaining - 2) / 4)

def final_remaining (N : ℕ) :=
  let step1 := first_pirate N
  let step2 := second_pirate step1
  let step3 := third_pirate step2
  let step4 := fourth_pirate step3
  step4

theorem pirates_share (N : ℕ) (h : initial_coins N) :
  final_remaining N / 4 = 660 :=
by
  sorry

end pirates_share_l213_213480


namespace find_X_l213_213747

theorem find_X (k : ℝ) (R1 R2 X1 X2 Y1 Y2 : ℝ) (h1 : R1 = k * (X1 / Y1)) (h2 : R1 = 10) (h3 : X1 = 2) (h4 : Y1 = 4) (h5 : R2 = 8) (h6 : Y2 = 5) : X2 = 2 :=
sorry

end find_X_l213_213747


namespace cannot_be_20182017_l213_213611

theorem cannot_be_20182017 (a b : ℤ) (h : a * b * (a + b) = 20182017) : False :=
by
  sorry

end cannot_be_20182017_l213_213611


namespace point_slope_intersection_lines_l213_213932

theorem point_slope_intersection_lines : 
  ∀ s : ℝ, ∃ x y : ℝ, 2*x - 3*y = 8*s + 6 ∧ x + 2*y = 3*s - 1 ∧ y = -((2*x)/25 + 182/175) := 
sorry

end point_slope_intersection_lines_l213_213932


namespace cells_at_end_of_12th_day_l213_213634

def initial_organisms : ℕ := 8
def initial_cells_per_organism : ℕ := 4
def total_initial_cells : ℕ := initial_organisms * initial_cells_per_organism
def division_period_days : ℕ := 3
def total_duration_days : ℕ := 12
def complete_periods : ℕ := total_duration_days / division_period_days
def common_ratio : ℕ := 3

theorem cells_at_end_of_12th_day :
  total_initial_cells * common_ratio^(complete_periods - 1) = 864 := by
  sorry

end cells_at_end_of_12th_day_l213_213634


namespace Joan_pays_139_20_l213_213564

noncomputable def JKL : Type := ℝ × ℝ × ℝ

def conditions (J K L : ℝ) : Prop :=
  J + K + L = 600 ∧
  2 * J = K + 74 ∧
  L = K + 52

theorem Joan_pays_139_20 (J K L : ℝ) (h : conditions J K L) : J = 139.20 :=
by
  sorry

end Joan_pays_139_20_l213_213564


namespace sum_first_49_odd_numbers_l213_213622

theorem sum_first_49_odd_numbers : (49^2 = 2401) :=
by
  sorry

end sum_first_49_odd_numbers_l213_213622


namespace find_b_l213_213453

-- Given conditions
def varies_inversely (a b : ℝ) := ∃ K : ℝ, K = a * b
def constant_a (a : ℝ) := a = 1500
def constant_b (b : ℝ) := b = 0.25

-- The theorem to prove
theorem find_b (a : ℝ) (b : ℝ) (h_inv: varies_inversely a b)
  (h_a: constant_a a) (h_b: constant_b b): b = 0.125 := 
sorry

end find_b_l213_213453


namespace alex_buys_17_1_pounds_of_corn_l213_213518

-- Definitions based on conditions
def corn_cost_per_pound : ℝ := 1.20
def bean_cost_per_pound : ℝ := 0.50
def total_pounds : ℝ := 30
def total_cost : ℝ := 27.00

-- Define the variables
variables (c b : ℝ)

-- Theorem statement to prove the number of pounds of corn Alex buys
theorem alex_buys_17_1_pounds_of_corn (h1 : b + c = total_pounds) (h2 : bean_cost_per_pound * b + corn_cost_per_pound * c = total_cost) :
  c = 17.1 :=
sorry

end alex_buys_17_1_pounds_of_corn_l213_213518


namespace problem_statement_l213_213927

open Real

theorem problem_statement (x : ℝ) (h₀ : 0 ≤ x ∧ x ≤ 2 * π)
  (h₁ : 2 * cos x ≤ sqrt (1 + sin (2 * x)) - sqrt (1 - sin (2 * x))
  ∧ sqrt (1 + sin (2 * x)) - sqrt (1 - sin (2 * x)) ≤ sqrt 2) :
  π / 4 ≤ x ∧ x ≤ 7 * π / 4 := sorry

end problem_statement_l213_213927


namespace radius_excircle_ABC_l213_213296

variables (A B C P Q : Point)
variables (r_ABP r_APQ r_AQC : ℝ) (re_ABP re_APQ re_AQC : ℝ)
variable (r_ABC : ℝ)

-- Conditions
-- Radii of the incircles of triangles ABP, APQ, and AQC are all equal to 1
axiom incircle_ABP : r_ABP = 1
axiom incircle_APQ : r_APQ = 1
axiom incircle_AQC : r_AQC = 1

-- Radii of the corresponding excircles opposite A for ABP, APQ, and AQC are 3, 6, and 5 respectively
axiom excircle_ABP : re_ABP = 3
axiom excircle_APQ : re_APQ = 6
axiom excircle_AQC : re_AQC = 5

-- Radius of the incircle of triangle ABC is 3/2
axiom incircle_ABC : r_ABC = 3 / 2

-- Theorem stating the radius of the excircle of triangle ABC opposite A is 135
theorem radius_excircle_ABC (r_ABC : ℝ) : r_ABC = 3 / 2 → ∀ (re_ABC : ℝ), re_ABC = 135 := 
by
  intros 
  sorry

end radius_excircle_ABC_l213_213296


namespace sqrt_40000_eq_200_l213_213456

theorem sqrt_40000_eq_200 : Real.sqrt 40000 = 200 := 
sorry

end sqrt_40000_eq_200_l213_213456


namespace scooter_cost_l213_213108

variable (saved needed total_cost : ℕ)

-- The conditions given in the problem
def greg_saved_57 : saved = 57 := sorry
def greg_needs_33_more : needed = 33 := sorry

-- The proof goal
theorem scooter_cost (h1 : saved = 57) (h2 : needed = 33) :
  total_cost = saved + needed → total_cost = 90 := by
  sorry

end scooter_cost_l213_213108


namespace chromium_percentage_l213_213560

theorem chromium_percentage (c1 c2 : ℝ) (w1 w2 : ℝ) (percentage1 percentage2 : ℝ) : 
  percentage1 = 0.1 → 
  percentage2 = 0.08 → 
  w1 = 15 → 
  w2 = 35 → 
  (c1 = percentage1 * w1) → 
  (c2 = percentage2 * w2) → 
  (c1 + c2 = 4.3) → 
  ((w1 + w2) = 50) →
  ((c1 + c2) / (w1 + w2) * 100 = 8.6) := 
by 
  sorry

end chromium_percentage_l213_213560


namespace new_circle_equation_l213_213739

-- Define the initial conditions
def initial_circle_equation (x y : ℝ) : Prop := x^2 + 2*x + y^2 = 0
def radius_of_new_circle : ℝ := 2

-- Define the target equation of the circle
def target_circle_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 4

-- The theorem statement
theorem new_circle_equation (x y : ℝ) :
  initial_circle_equation x y → target_circle_equation x y :=
sorry

end new_circle_equation_l213_213739


namespace six_digit_number_divisible_by_eleven_l213_213583

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def reverse_digits (a b c : ℕ) : ℕ :=
  100 * c + 10 * b + a

def concatenate_reverse (a b c : ℕ) : ℕ :=
  100000 * a + 10000 * b + 1000 * c + 100 * c + 10 * b + a

theorem six_digit_number_divisible_by_eleven (a b c : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ 9)
  (h₃ : 0 ≤ b) (h₄ : b ≤ 9) (h₅ : 0 ≤ c) (h₆ : c ≤ 9) :
  11 ∣ concatenate_reverse a b c :=
by
  sorry

end six_digit_number_divisible_by_eleven_l213_213583


namespace solvable_system_l213_213044

theorem solvable_system (p : ℝ) : 
  (∃ (y : ℝ), ∃ (x : ℝ), ([⌊x⌋] : ℝ) = x /\
    2 * x + y = 3 / 2 ∧ 
    3 * x - 2 * y = p) ↔ 
  (∃ (k : ℤ), p = 7 * k - 3) := 
begin
  sorry
end

end solvable_system_l213_213044


namespace find_x_l213_213883

theorem find_x :
  ∀ (x : ℝ), 4.7 * x + 4.7 * 9.43 + 4.7 * 77.31 = 470 → x = 13.26 :=
by
  intro x
  intro h
  sorry

end find_x_l213_213883


namespace mother_daughter_age_l213_213212

theorem mother_daughter_age (x : ℕ) :
  let mother_age := 42
  let daughter_age := 8
  (mother_age + x = 3 * (daughter_age + x)) → x = 9 :=
by
  let mother_age := 42
  let daughter_age := 8
  intro h
  sorry

end mother_daughter_age_l213_213212


namespace triangle_altitude_l213_213479

theorem triangle_altitude (base side : ℝ) (h : ℝ) : 
  side = 6 → base = 6 → 
  (base * h) / 2 = side ^ 2 → 
  h = 12 :=
by
  intros
  sorry

end triangle_altitude_l213_213479


namespace find_triplets_l213_213665

theorem find_triplets (a k m : ℕ) (hpos_a : 0 < a) (hpos_k : 0 < k) (hpos_m : 0 < m) (h_eq : k + a^k = m + 2 * a^m) :
  ∃ t : ℕ, 0 < t ∧ (a = 1 ∧ k = t + 1 ∧ m = t) :=
by
  sorry

end find_triplets_l213_213665


namespace complex_magnitude_product_l213_213920

theorem complex_magnitude_product:
  let z1 := (4 - 3 * Complex.i)
  let z2 := (4 + 3 * Complex.i)
  |z1| * |z2| = 25 :=
by
  let z1 := (4 - 3 * Complex.i)
  let z2 := (4 + 3 * Complex.i)
  show |z1| * |z2| = 25
  sorry

end complex_magnitude_product_l213_213920


namespace find_shift_b_l213_213229

-- Define the periodic function f
variable (f : ℝ → ℝ)
-- Define the condition on f
axiom f_periodic : ∀ x, f (x - 30) = f x

-- The theorem we want to prove
theorem find_shift_b : ∃ b > 0, (∀ x, f ((x - b) / 3) = f (x / 3)) ∧ b = 90 := 
by
  sorry

end find_shift_b_l213_213229


namespace arrange_numbers_divisible_l213_213507

open List

theorem arrange_numbers_divisible :
  ∃ (σ : List ℕ), σ = [7, 1, 8, 4, 10, 6, 9, 3, 2, 5] ∧
  ∀ (i : ℕ) (h : i < length σ), σ.nthLe i h ∣ (sum (take i σ)) :=
by
  use [7, 1, 8, 4, 10, 6, 9, 3, 2, 5]
  intro i h
  cases i
  case nat.zero => simp [nat.dvd_refl]
  case nat.succ i =>
    simp
    sorry

end arrange_numbers_divisible_l213_213507


namespace numberOfBoys_playground_boys_count_l213_213470

-- Definitions and conditions
def numberOfGirls : ℕ := 28
def totalNumberOfChildren : ℕ := 63

-- Theorem statement
theorem numberOfBoys (numberOfGirls : ℕ) (totalNumberOfChildren : ℕ) : ℕ :=
  totalNumberOfChildren - numberOfGirls

-- Proof statement
theorem playground_boys_count (numberOfGirls : ℕ) (totalNumberOfChildren : ℕ) (boysOnPlayground : ℕ) : 
  numberOfGirls = 28 → 
  totalNumberOfChildren = 63 → 
  boysOnPlayground = totalNumberOfChildren - numberOfGirls →
  boysOnPlayground = 35 :=
by
  intros
  -- since no proof is required, we use sorry here
  exact sorry

end numberOfBoys_playground_boys_count_l213_213470


namespace largest_lcm_l213_213187

theorem largest_lcm :
  max (max (max (max (Nat.lcm 18 4) (Nat.lcm 18 6)) (Nat.lcm 18 9)) (Nat.lcm 18 14)) (Nat.lcm 18 18) = 126 :=
by
  sorry

end largest_lcm_l213_213187


namespace min_M_inequality_l213_213669

noncomputable def M_min : ℝ := 9 * Real.sqrt 2 / 32

theorem min_M_inequality :
  ∀ (a b c : ℝ),
    abs (a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2))
    ≤ M_min * (a^2 + b^2 + c^2)^2 :=
by
  sorry

end min_M_inequality_l213_213669


namespace binary_div_remainder_l213_213754

theorem binary_div_remainder (n : ℕ) (h : n = 0b101011100101) : n % 8 = 5 :=
by sorry

end binary_div_remainder_l213_213754


namespace math_problem_proof_l213_213674

-- Definitions from the problem conditions
def original_set : Finset ℕ := Finset.range 500
def draw_a := Finset.nth_le (Finset.insert 0 original_set) 0 sorry  -- placeholder for drawing
def remaining_set := original_set.erase draw_a
def draw_b := Finset.nth_le (Finset.insert 0 remaining_set) 0 sorry  -- placeholder for drawing

def hyperbrick_dimensions := {a1 : ℕ // a1 ∈ original_set}
def hyperbox_dimensions := {b1 : ℕ // b1 ∈ remaining_set}

-- The main theorem statement where we need to prove the sum of numerator and denominator.
theorem math_problem_proof : let p := (16 : ℚ) / 70 in (p.num + p.den) = 43 :=
by
  sorry  -- Proof to be filled in later.

end math_problem_proof_l213_213674


namespace percent_voters_for_A_l213_213558

-- Definitions from conditions
def total_voters : ℕ := 100
def percent_democrats : ℝ := 0.70
def percent_republicans : ℝ := 0.30
def percent_dems_for_A : ℝ := 0.80
def percent_reps_for_A : ℝ := 0.30

-- Calculations based on definitions
def num_democrats := total_voters * percent_democrats
def num_republicans := total_voters * percent_republicans
def dems_for_A := num_democrats * percent_dems_for_A
def reps_for_A := num_republicans * percent_reps_for_A
def total_for_A := dems_for_A + reps_for_A

-- Proof problem statement
theorem percent_voters_for_A : (total_for_A / total_voters) * 100 = 65 :=
by
  sorry

end percent_voters_for_A_l213_213558


namespace part1_part2_part3_l213_213404

-- Define the necessary constants and functions as per conditions
variable (a : ℝ) (f : ℝ → ℝ)
variable (hpos : a > 0) (hfa : f a = 1)

-- Conditions based on the problem statement
variable (hodd : ∀ x, f (-x) = -f x)
variable (hfe : ∀ x1 x2, f (x1 - x2) = (f x1 * f x2 + 1) / (f x2 - f x1))

-- 1. Prove that f(2a) = 0
theorem part1  : f (2 * a) = 0 := sorry

-- 2. Prove that there exists a constant T > 0 such that f(x + T) = f(x)
theorem part2 : ∃ T > 0, ∀ x, f (x + 4 * a) = f x := sorry

-- 3. Prove f(x) is decreasing on (0, 4a) given x ∈ (0, 2a) implies f(x) > 0
theorem part3 (hx_correct : ∀ x, 0 < x ∧ x < 2 * a → 0 < f x) :
  ∀ x1 x2, 0 < x2 ∧ x2  < x1 ∧ x1 < 4 * a → f x2 > f x1 := sorry

end part1_part2_part3_l213_213404


namespace yard_length_l213_213280

-- Definition of the problem conditions
def num_trees : Nat := 11
def distance_between_trees : Nat := 15

-- Length of the yard is given by the product of (num_trees - 1) and distance_between_trees
theorem yard_length :
  (num_trees - 1) * distance_between_trees = 150 :=
by
  sorry

end yard_length_l213_213280


namespace D_score_l213_213242

noncomputable def score_A : ℕ := 94

variables (A B C D E : ℕ)

-- Conditions
def A_scored : A = score_A := sorry
def B_highest : B > A := sorry
def C_average_AD : (C * 2) = A + D := sorry
def D_average_five : (D * 5) = A + B + C + D + E := sorry
def E_score_C2 : E = C + 2 := sorry

-- Question
theorem D_score : D = 96 :=
by {
  sorry
}

end D_score_l213_213242


namespace find_k_intersecting_lines_l213_213084

theorem find_k_intersecting_lines : 
  ∃ (k : ℚ), (∃ (x y : ℚ), y = 6 * x + 4 ∧ y = -3 * x - 30 ∧ y = 4 * x + k) ∧ k = -32 / 9 :=
by
  sorry

end find_k_intersecting_lines_l213_213084


namespace corresponding_angles_equal_l213_213589

variable {α : Type}
variables (A B : α) [angle : has_measure α (angle_measure α)]
variables (h : A.is_corresponding_with B)

theorem corresponding_angles_equal (A B : α) [angle A] [angle B] :
  A.is_corresponding_with B → A = B :=
by
  sorry

end corresponding_angles_equal_l213_213589


namespace Jermaine_more_than_Terrence_l213_213285

theorem Jermaine_more_than_Terrence :
  ∀ (total_earnings Terrence_earnings Emilee_earnings : ℕ),
    total_earnings = 90 →
    Terrence_earnings = 30 →
    Emilee_earnings = 25 →
    (total_earnings - Terrence_earnings - Emilee_earnings) - Terrence_earnings = 5 := by
  sorry

end Jermaine_more_than_Terrence_l213_213285


namespace p_has_49_l213_213627

theorem p_has_49 (P : ℝ) (h : P = (2/7) * P + 35) : P = 49 :=
by
  sorry

end p_has_49_l213_213627


namespace max_rock_value_l213_213624

def rock_value (weight_5 : Nat) (weight_4 : Nat) (weight_1 : Nat) : Nat :=
  14 * weight_5 + 11 * weight_4 + 2 * weight_1

def total_weight (weight_5 : Nat) (weight_4 : Nat) (weight_1 : Nat) : Nat :=
  5 * weight_5 + 4 * weight_4 + 1 * weight_1

theorem max_rock_value : ∃ (weight_5 weight_4 weight_1 : Nat), 
  total_weight weight_5 weight_4 weight_1 ≤ 18 ∧ 
  rock_value weight_5 weight_4 weight_1 = 50 :=
by
  -- We need to find suitable weight_5, weight_4, and weight_1.
  use 2, 2, 0 -- Example values
  apply And.intro
  -- Prove the total weight condition
  show total_weight 2 2 0 ≤ 18
  sorry
  -- Prove the value condition
  show rock_value 2 2 0 = 50
  sorry

end max_rock_value_l213_213624


namespace daily_wage_c_l213_213758

theorem daily_wage_c (a_days b_days c_days total_earnings : ℕ)
  (ratio_a_b ratio_b_c : ℚ)
  (a_wage b_wage c_wage : ℚ) :
  a_days = 6 →
  b_days = 9 →
  c_days = 4 →
  total_earnings = 1480 →
  ratio_a_b = 3 / 4 →
  ratio_b_c = 4 / 5 →
  b_wage = ratio_a_b * a_wage → 
  c_wage = ratio_b_c * b_wage → 
  a_days * a_wage + b_days * b_wage + c_days * c_wage = total_earnings →
  c_wage = 100 / 3 :=
by
  intros
  sorry

end daily_wage_c_l213_213758


namespace an_plus_an_minus_1_eq_two_pow_n_l213_213243

def a_n (n : ℕ) : ℕ := sorry -- Placeholder for the actual function a_n

theorem an_plus_an_minus_1_eq_two_pow_n (n : ℕ) (h : n ≥ 4) : a_n (n - 1) + a_n n = 2^n := 
by
  sorry

end an_plus_an_minus_1_eq_two_pow_n_l213_213243


namespace sum_of_numbers_l213_213965

theorem sum_of_numbers (a : ℝ) (n : ℕ) (h : a = 5.3) (hn : n = 8) : (a * n) = 42.4 :=
sorry

end sum_of_numbers_l213_213965


namespace complete_square_l213_213317

theorem complete_square (x m : ℝ) : x^2 + 2 * x - 2 = 0 → (x + m)^2 = 3 → m = 1 := sorry

end complete_square_l213_213317


namespace log_sum_reciprocals_of_logs_l213_213905

-- Problem (1)
theorem log_sum (log_two : Real.log 2 ≠ 0) :
    Real.log 4 / Real.log 10 + Real.log 50 / Real.log 10 - Real.log 2 / Real.log 10 = 2 := by
  sorry

-- Problem (2)
theorem reciprocals_of_logs (a b : Real) (h : 1 + Real.log a / Real.log 2 = 2 + Real.log b / Real.log 3 ∧ (1 + Real.log a / Real.log 2) = Real.log (a + b) / Real.log 6) : 
    1 / a + 1 / b = 6 := by
  sorry

end log_sum_reciprocals_of_logs_l213_213905


namespace total_cost_kept_l213_213826

def prices_all : List ℕ := [15, 18, 20, 15, 25, 30, 20, 17, 22, 23, 29]
def prices_returned : List ℕ := [20, 25, 30, 22, 23, 29]

def total_cost (prices : List ℕ) : ℕ :=
  prices.foldl (· + ·) 0

theorem total_cost_kept :
  total_cost prices_all - total_cost prices_returned = 85 :=
by
  -- The proof steps go here
  sorry

end total_cost_kept_l213_213826


namespace American_carmakers_produce_l213_213363

theorem American_carmakers_produce :
  let first := 1000000
  let second := first + 500000
  let third := first + second
  let fourth := 325000
  let fifth := 325000
  let total := first + second + third + fourth + fifth
  total = 5650000 :=
by
  let first := 1000000
  let second := first + 500000
  let third := first + second
  let fourth := 325000
  let fifth := 325000
  let total := first + second + third + fourth + fifth
  show total = 5650000
  sorry

end American_carmakers_produce_l213_213363


namespace increasing_ω_l213_213813

noncomputable def f (ω x : ℝ) : ℝ := (1 / 2) * (Real.sin ((ω * x) / 2)) * (Real.cos ((ω * x) / 2))

theorem increasing_ω (ω : ℝ) (hω : 0 < ω) :
  (∀ x y, - (Real.pi / 3) ≤ x → x ≤ y → y ≤ (Real.pi / 4) → f ω x ≤ f ω y)
  ↔ 0 < ω ∧ ω ≤ (3 / 2) :=
sorry

end increasing_ω_l213_213813


namespace probability_AC_less_than_10_cm_l213_213777

-- Definition of the problem
def distance_between_points (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Theorem statement
theorem probability_AC_less_than_10_cm :
  ∃ (α : ℝ) (h : 0 < α ∧ α < real.pi), 
  let B := (0, 0),
      A := (0, -12 : ℝ),
      C := (8 * real.cos α, 8 * real.sin α : ℝ) in
  ∃ (θ : ℝ), 
  ∃ (prob : ℝ), 
  θ =  real.atan2 (- √ (92 / 9)) (-22/3) - real.atan2 (√ (92 / 9)) (-22 / 3) ∧
  prob = θ / real.pi ∧
  distance_between_points A C < 10 ∧
  prob = 1 / 3 :=
  sorry

end probability_AC_less_than_10_cm_l213_213777


namespace kenneth_earnings_l213_213568

theorem kenneth_earnings (E : ℝ) (h1 : E - 0.1 * E = 405) : E = 450 :=
sorry

end kenneth_earnings_l213_213568


namespace initial_books_count_l213_213639

-- Definitions of the given conditions
def shelves : ℕ := 9
def books_per_shelf : ℕ := 9
def books_remaining : ℕ := shelves * books_per_shelf
def books_sold : ℕ := 39

-- Statement of the proof problem
theorem initial_books_count : books_remaining + books_sold = 120 := 
by {
  sorry
}

end initial_books_count_l213_213639


namespace knocks_to_knicks_l213_213692

-- Define the conversion relationships as conditions
variable (knicks knacks knocks : ℝ)

-- 8 knicks = 3 knacks
axiom h1 : 8 * knicks = 3 * knacks

-- 5 knacks = 6 knocks
axiom h2 : 5 * knacks = 6 * knocks

-- Proving the equivalence of 30 knocks to 66.67 knicks
theorem knocks_to_knicks : 30 * knocks = 66.67 * knicks :=
by
  sorry

end knocks_to_knicks_l213_213692


namespace no_prime_divisible_by_56_l213_213124

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def divisible_by_56 (n : ℕ) : Prop := 56 ∣ n

theorem no_prime_divisible_by_56 : ¬ ∃ p, is_prime p ∧ divisible_by_56 p := 
  sorry

end no_prime_divisible_by_56_l213_213124


namespace relay_race_athlete_orders_l213_213199

def athlete_count : ℕ := 4
def cannot_run_first_leg (athlete : ℕ) : Prop := athlete = 1
def cannot_run_fourth_leg (athlete : ℕ) : Prop := athlete = 2

theorem relay_race_athlete_orders : 
  ∃ (number_of_orders : ℕ), number_of_orders = 14 := 
by 
  -- Proof is omitted because it’s not required as per instructions.
  sorry

end relay_race_athlete_orders_l213_213199


namespace percentage_deficit_for_second_side_l213_213837

theorem percentage_deficit_for_second_side
  (L W : ℝ) 
  (measured_first_side : ℝ := 1.12 * L) 
  (error_in_area : ℝ := 1.064) : 
  (∃ x : ℝ, (1.12 * L) * ((1 - 0.01 * x) * W) = 1.064 * (L * W) → x = 5) :=
by
  sorry

end percentage_deficit_for_second_side_l213_213837


namespace employee_gross_pay_l213_213775

theorem employee_gross_pay
  (pay_rate_regular : ℝ) (pay_rate_overtime : ℝ) (regular_hours : ℝ) (overtime_hours : ℝ)
  (h1 : pay_rate_regular = 11.25)
  (h2 : pay_rate_overtime = 16)
  (h3 : regular_hours = 40)
  (h4 : overtime_hours = 10.75) :
  (pay_rate_regular * regular_hours + pay_rate_overtime * overtime_hours = 622) :=
by
  sorry

end employee_gross_pay_l213_213775


namespace cos_alpha_plus_20_eq_neg_alpha_l213_213814

variable (α : ℝ)

theorem cos_alpha_plus_20_eq_neg_alpha (h : Real.sin (α - 70 * Real.pi / 180) = α) :
    Real.cos (α + 20 * Real.pi / 180) = -α :=
by
  sorry

end cos_alpha_plus_20_eq_neg_alpha_l213_213814


namespace sum_of_eight_numbers_l213_213969

theorem sum_of_eight_numbers (avg : ℝ) (S : ℝ) (h1 : avg = 5.3) (h2 : avg = S / 8) : S = 42.4 :=
by
  sorry

end sum_of_eight_numbers_l213_213969


namespace range_of_x_l213_213529

theorem range_of_x {f : ℝ → ℝ} (h_even : ∀ x, f x = f (-x)) 
  (h_mono_dec : ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x)
  (h_f2 : f 2 = 0)
  (h_pos : ∀ x, f (x - 1) > 0) : 
  ∀ x, -1 < x ∧ x < 3 ↔ f (x - 1) > 0 :=
sorry

end range_of_x_l213_213529


namespace Gandalf_reachability_l213_213009

theorem Gandalf_reachability (n : ℕ) (h : n ≥ 1) :
  ∃ (m : ℕ), m = 1 :=
sorry

end Gandalf_reachability_l213_213009


namespace solve_for_A_l213_213550

def diamond (A B : ℝ) := 4 * A + 3 * B + 7

theorem solve_for_A : diamond A 5 = 71 → A = 12.25 := by
  intro h
  unfold diamond at h
  sorry

end solve_for_A_l213_213550


namespace x4_plus_y4_l213_213689
noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry

theorem x4_plus_y4 :
  (x^2 + (1 / x^2) = 7) →
  (x * y = 1) →
  (x^4 + y^4 = 47) :=
by
  intros h1 h2
  -- The proof will go here.
  sorry

end x4_plus_y4_l213_213689


namespace john_draw_on_back_l213_213567

theorem john_draw_on_back (total_pictures front_pictures : ℕ) (h1 : total_pictures = 15) (h2 : front_pictures = 6) : total_pictures - front_pictures = 9 :=
  by
  sorry

end john_draw_on_back_l213_213567


namespace parallel_lines_l213_213536

theorem parallel_lines (a : ℝ) : (∀ x y : ℝ, 2 * x + a * y + 1 = 0 ↔ x - 4 * y - 1 = 0) → a = -8 :=
by
  intro h -- Introduce the hypothesis that lines are parallel
  sorry -- Skip the proof

end parallel_lines_l213_213536


namespace expression_value_l213_213175

theorem expression_value (x y : ℝ) (h1 : x + y = 17) (h2 : x * y = 17) :
  (x^2 - 17*x) * (y + 17/y) = -289 :=
by
  sorry

end expression_value_l213_213175


namespace exchanges_count_l213_213034

theorem exchanges_count (n : ℕ) :
  ∀ (initial_pencils_XZ initial_pens_XL : ℕ) 
    (pencils_per_exchange pens_per_exchange : ℕ)
    (final_pencils_multiplier : ℕ)
    (pz : initial_pencils_XZ = 200) 
    (pl : initial_pens_XL = 20)
    (pe : pencils_per_exchange = 6)
    (se : pens_per_exchange = 1)
    (fm : final_pencils_multiplier = 11),
    (initial_pencils_XZ - pencils_per_exchange * n = final_pencils_multiplier * (initial_pens_XL - pens_per_exchange * n)) ↔ n = 4 :=
by
  intros initial_pencils_XZ initial_pens_XL pencils_per_exchange pens_per_exchange final_pencils_multiplier pz pl pe se fm
  sorry

end exchanges_count_l213_213034


namespace sum_first_40_terms_l213_213688

-- Defining the sequence a_n following the given conditions
noncomputable def a : ℕ → ℝ
| 0 => 1
| 1 => 3
| n + 2 => a (n + 1) * a (n - 1)

-- Defining the sum of the first 40 terms of the sequence
noncomputable def S40 := (Finset.range 40).sum a

-- The theorem stating the desired property
theorem sum_first_40_terms : S40 = 60 :=
sorry

end sum_first_40_terms_l213_213688


namespace field_area_l213_213769

-- Define a rectangular field
structure RectangularField where
  length : ℕ
  width : ℕ
  fencing : ℕ := 2 * width + length
  
-- Given conditions
def field_conditions (L W F : ℕ) : Prop :=
  L = 30 ∧ 2 * W + L = F

-- Theorem stating the required proof
theorem field_area : ∀ (L W F : ℕ), field_conditions L W F → F = 84 → (L * W) = 810 :=
by
  intros L W F h1 h2
  sorry

end field_area_l213_213769


namespace find_x_l213_213295

variable {a b x : ℝ}
variable (h₀ : b ≠ 0)
variable (h₁ : (3 * a)^(2 * b) = a^b * x^b)

theorem find_x (h₀ : b ≠ 0) (h₁ : (3 * a)^(2 * b) = a^b * x^b) : x = 9 * a :=
sorry

end find_x_l213_213295


namespace exactly_two_overlap_l213_213750

-- Define the concept of rectangles
structure Rectangle :=
  (width : ℕ)
  (height : ℕ)

-- Define the given rectangles
def rect1 : Rectangle := ⟨4, 6⟩
def rect2 : Rectangle := ⟨4, 6⟩
def rect3 : Rectangle := ⟨4, 6⟩

-- Hypothesis defining the overlapping areas
def overlap1_2 : ℕ := 4 * 2 -- first and second rectangles overlap in 8 cells
def overlap2_3 : ℕ := 2 * 6 -- second and third rectangles overlap in 12 cells
def overlap1_3 : ℕ := 0    -- first and third rectangles do not directly overlap

-- Total overlap calculation
def total_exactly_two_overlap : ℕ := (overlap1_2 + overlap2_3)

-- The theorem we need to prove
theorem exactly_two_overlap (rect1 rect2 rect3 : Rectangle) : total_exactly_two_overlap = 14 := sorry

end exactly_two_overlap_l213_213750


namespace fraction_identity_l213_213416

theorem fraction_identity
  (x w y z : ℝ)
  (hxw_pos : x * w > 0)
  (hyz_pos : y * z > 0)
  (hxw_inv_sum : 1 / x + 1 / w = 20)
  (hyz_inv_sum : 1 / y + 1 / z = 25)
  (hxw_inv : 1 / (x * w) = 6)
  (hyz_inv : 1 / (y * z) = 8) :
  (x + y) / (z + w) = 155 / 7 :=
by
  -- proof omitted
  sorry

end fraction_identity_l213_213416


namespace total_amount_is_47_69_l213_213447

noncomputable def Mell_order_cost : ℝ :=
  2 * 4 + 7

noncomputable def friend_order_cost : ℝ :=
  2 * 4 + 7 + 3

noncomputable def total_cost_before_discount : ℝ :=
  Mell_order_cost + 2 * friend_order_cost

noncomputable def discount : ℝ :=
  0.15 * total_cost_before_discount

noncomputable def total_after_discount : ℝ :=
  total_cost_before_discount - discount

noncomputable def sales_tax : ℝ :=
  0.10 * total_after_discount

noncomputable def total_to_pay : ℝ :=
  total_after_discount + sales_tax

theorem total_amount_is_47_69 : total_to_pay = 47.69 :=
by
  sorry

end total_amount_is_47_69_l213_213447


namespace evaluate_expression_l213_213237

theorem evaluate_expression (c d : ℝ) (h_c : c = 3) (h_d : d = 2) : 
  (c^2 + d + 1)^2 - (c^2 - d - 1)^2 = 80 := by 
  sorry

end evaluate_expression_l213_213237


namespace each_group_has_two_bananas_l213_213599

theorem each_group_has_two_bananas (G T : ℕ) (hG : G = 196) (hT : T = 392) : T / G = 2 :=
by
  sorry

end each_group_has_two_bananas_l213_213599


namespace image_of_element_2_l213_213544

-- Define the mapping f and conditions
def f (x : ℕ) : ℕ := 2 * x + 1

-- Define the element and its image using f
def element_in_set_A : ℕ := 2
def image_in_set_B : ℕ := f element_in_set_A

-- The theorem to prove
theorem image_of_element_2 : image_in_set_B = 5 :=
by
  -- This is where the proof would go, but we omit it with sorry
  sorry

end image_of_element_2_l213_213544


namespace find_base_r_l213_213449

noncomputable def x: ℕ := 9999

theorem find_base_r (r: ℕ) (hr_even: Even r) (hr_gt_9: r > 9) 
    (h_palindrome: ∃ a b c d: ℕ, b + c = 24 ∧ 
                   ((81 * ((r^6 * (r^6 + 2 * r^5 + 3 * r^4 + 4 * r^3 + 3 * r^2 + 2 * r + 1 + r^2)) = 
                     a * r^7 + b * r^6 + c * r^5 + d * r^4 + d * r^3 + c * r^2 + b * r + a)))):
    r = 26 :=
by
  sorry

end find_base_r_l213_213449


namespace problem1_problem2_problem3_l213_213046

-- 1. Given: ∃ x ∈ ℤ, x^2 - 2x - 3 = 0
--    Show: ∀ x ∈ ℤ, x^2 - 2x - 3 ≠ 0
theorem problem1 : (∃ x : ℤ, x^2 - 2 * x - 3 = 0) ↔ (∀ x : ℤ, x^2 - 2 * x - 3 ≠ 0) := sorry

-- 2. Given: ∀ x ∈ ℝ, x^2 + 3 ≥ 2x
--    Show: ∃ x ∈ ℝ, x^2 + 3 < 2x
theorem problem2 : (∀ x : ℝ, x^2 + 3 ≥ 2 * x) ↔ (∃ x : ℝ, x^2 + 3 < 2 * x) := sorry

-- 3. Given: If x > 1 and y > 1, then x + y > 2
--    Show: If x ≤ 1 or y ≤ 1, then x + y ≤ 2
theorem problem3 : (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2) ↔ (∀ x y : ℝ, x ≤ 1 ∨ y ≤ 1 → x + y ≤ 2) := sorry

end problem1_problem2_problem3_l213_213046


namespace trigonometric_identity_l213_213511

theorem trigonometric_identity : (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 1 - Real.sqrt 3 :=
by
  sorry

end trigonometric_identity_l213_213511


namespace distance_of_points_in_polar_coordinates_l213_213571

theorem distance_of_points_in_polar_coordinates
  (A : Real × Real) (B : Real × Real) (θ1 θ2 : Real)
  (hA : A = (5, θ1)) (hB : B = (12, θ2))
  (hθ : θ1 - θ2 = Real.pi / 2) : 
  dist (5 * Real.cos θ1, 5 * Real.sin θ1) (12 * Real.cos θ2, 12 * Real.sin θ2) = 13 := 
by sorry

end distance_of_points_in_polar_coordinates_l213_213571


namespace find_cos_minus_sin_l213_213678

-- Definitions from the conditions
variable (α : ℝ)
variable (h1 : 0 < α ∧ α < π)  -- Second quadrant
variable (h2 : Real.sin (2 * α) = -24 / 25)  -- Given sin 2α

-- Lean statement of the problem
theorem find_cos_minus_sin (α : ℝ) (h1 : 0 < α ∧ α < π) (h2 : Real.sin (2 * α) = -24 / 25) :
  Real.cos α - Real.sin α = -7 / 5 := 
sorry

end find_cos_minus_sin_l213_213678


namespace complex_number_solution_l213_213127

open Complex

noncomputable def solve_z (a : ℝ) (h : a ≥ 0) : ℂ :=
(a - real.sqrt (a ^ 2 + 4)) / 2 * I

theorem complex_number_solution (a : ℝ) (h : a ≥ 0) (z : ℂ) (hz : z * abs z + a * z + I = 0) :
  z = solve_z a h :=
begin
  sorry
end

end complex_number_solution_l213_213127


namespace sum_pairs_mod_eq_l213_213719

open Nat

def permutation_of (l : list ℕ) (n : ℕ) := 
  ∀ x, x ∈ l ↔ x ∈ list.range (n + 1)

theorem sum_pairs_mod_eq (A B : list ℕ) (n : ℕ)
  (hA : permutation_of A n) (hB : permutation_of B n) (h_even : even n) :
  ∃ i j, i ≠ j ∧ (A.nth i).getOrElse 0 + (B.nth i).getOrElse 0 % n = (A.nth j).getOrElse 0 + (B.nth j).getOrElse 0 % n :=
by
  sorry

end sum_pairs_mod_eq_l213_213719


namespace number_exceeds_80_by_120_l213_213054

theorem number_exceeds_80_by_120 : ∃ x : ℝ, x = 0.80 * x + 120 ∧ x = 600 :=
by sorry

end number_exceeds_80_by_120_l213_213054


namespace area_of_union_of_triangles_l213_213089

-- Define the vertices of the original triangle
def A : ℝ × ℝ := (3, 4)
def B : ℝ × ℝ := (5, -2)
def C : ℝ × ℝ := (7, 3)

-- Define the reflection function across the line x=5
def reflect_x5 (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (10 - x, y)

-- Define the vertices of the reflected triangle
def A' : ℝ × ℝ := reflect_x5 A
def B' : ℝ × ℝ := reflect_x5 B
def C' : ℝ × ℝ := reflect_x5 C

-- Function to calculate the area of a triangle given its vertices
def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  let (x1, y1) := P
  let (x2, y2) := Q
  let (x3, y3) := R
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- Prove that the area of the union of both triangles is 22
theorem area_of_union_of_triangles : triangle_area A B C + triangle_area A' B' C' = 22 := by
  sorry

end area_of_union_of_triangles_l213_213089


namespace f_2015_2016_l213_213680

noncomputable def f : ℤ → ℤ := sorry

theorem f_2015_2016 (h1 : ∀ x, f (-x) = -f x) (h2 : ∀ x, f (x + 2) = -f x) (h3 : f 1 = 2) :
  f 2015 + f 2016 = -2 :=
sorry

end f_2015_2016_l213_213680


namespace angle_terminal_side_equiv_l213_213427

def angle_equiv_terminal_side (θ₁ θ₂ : ℝ) : Prop :=
  ∃ k : ℤ, θ₂ = θ₁ + 2 * k * Real.pi

theorem angle_terminal_side_equiv : angle_equiv_terminal_side (-Real.pi / 3) (5 * Real.pi / 3) :=
by
  sorry

end angle_terminal_side_equiv_l213_213427


namespace trigonometric_identity_l213_213513

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180))
  = (4 * Real.sin (10 * Real.pi / 180) / Real.sin (40 * Real.pi / 180)) :=
by sorry

end trigonometric_identity_l213_213513


namespace angle_GDA_is_135_l213_213425

-- Definitions for the geometric entities and conditions mentioned
structure Triangle :=
  (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ)

structure Square :=
  (angle : ℝ := 90)

def BCD : Triangle :=
  { angle_A := 45, angle_B := 45, angle_C := 90 }

def ABCD : Square :=
  {}

def DEFG : Square :=
  {}

-- The proof problem stated in Lean 4
theorem angle_GDA_is_135 :
  ∃ θ : ℝ, θ = 135 ∧ 
  (∀ (BCD : Triangle), BCD.angle_C = 90 ∧ BCD.angle_A = 45 ∧ BCD.angle_B = 45) ∧ 
  (∀ (Square : Square), Square.angle = 90) → 
  θ = 135 :=
by
  sorry

end angle_GDA_is_135_l213_213425


namespace verify_expressions_l213_213959

variable (x y : ℝ)
variable (h : x / y = 5 / 3)

theorem verify_expressions :
  (2 * x + y) / y = 13 / 3 ∧
  y / (y - 2 * x) = 3 / -7 ∧
  (x + y) / x = 8 / 5 ∧
  x / (3 * y) = 5 / 9 ∧
  (x - 2 * y) / y = -1 / 3 := by
sorry

end verify_expressions_l213_213959


namespace encore_songs_l213_213881

-- Definitions corresponding to the conditions
def repertoire_size : ℕ := 30
def first_set_songs : ℕ := 5
def second_set_songs : ℕ := 7
def average_songs_per_set_3_and_4 : ℕ := 8

-- The statement to prove
theorem encore_songs : (repertoire_size - (first_set_songs + second_set_songs)) - (2 * average_songs_per_set_3_and_4) = 2 := by
  sorry

end encore_songs_l213_213881


namespace chord_length_l213_213978

theorem chord_length (r : ℝ) (h : r = 15) :
  ∃ (cd : ℝ), cd = 13 * Real.sqrt 3 :=
by
  sorry

end chord_length_l213_213978


namespace max_value_ineq_l213_213721

theorem max_value_ineq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (x + y)^2 / (x^2 + y^2 + xy) ≤ 4 / 3 :=
sorry

end max_value_ineq_l213_213721


namespace maryann_work_time_l213_213856

variables (C A R : ℕ)

theorem maryann_work_time
  (h1 : A = 2 * C)
  (h2 : R = 6 * C)
  (h3 : C + A + R = 1440) :
  C = 160 ∧ A = 320 ∧ R = 960 :=
by
  sorry

end maryann_work_time_l213_213856


namespace corresponding_angles_equal_l213_213587

-- Define what it means for two angles to be corresponding angles
def corresponding_angles (a b : ℝ) : Prop :=
  -- Hypothetical definition
  sorry

-- Lean 4 statement of the problem
theorem corresponding_angles_equal (a b : ℝ) (h : corresponding_angles a b) : a = b :=
by
  sorry

end corresponding_angles_equal_l213_213587


namespace gcf_45_75_90_l213_213021

-- Definitions as conditions
def number1 : Nat := 45
def number2 : Nat := 75
def number3 : Nat := 90

def factors_45 : Nat × Nat := (3, 2) -- represents 3^2 * 5^1 {prime factor 3, prime factor 5}
def factors_75 : Nat × Nat := (5, 1) -- represents 3^1 * 5^2 {prime factor 3, prime factor 5}
def factors_90 : Nat × Nat := (3, 2) -- represents 2^1 * 3^2 * 5^1 {prime factor 3, prime factor 5}

-- Theorems to be proved
theorem gcf_45_75_90 : Nat.gcd (Nat.gcd number1 number2) number3 = 15 :=
by {
  -- This is here as placeholder for actual proof
  sorry
}

end gcf_45_75_90_l213_213021


namespace johns_out_of_pocket_expense_l213_213565

-- Define the conditions given in the problem
def old_system_cost : ℤ := 250
def old_system_trade_in_value : ℤ := (80 * old_system_cost) / 100
def new_system_initial_cost : ℤ := 600
def new_system_discount : ℤ := (25 * new_system_initial_cost) / 100
def new_system_final_cost : ℤ := new_system_initial_cost - new_system_discount

-- Define the amount of money that came out of John's pocket
def out_of_pocket_expense : ℤ := new_system_final_cost - old_system_trade_in_value

-- State the theorem that needs to be proven
theorem johns_out_of_pocket_expense : out_of_pocket_expense = 250 := by
  sorry

end johns_out_of_pocket_expense_l213_213565


namespace total_interest_is_350_l213_213766

-- Define the principal amounts, rates, and time
def principal1 : ℝ := 1000
def rate1 : ℝ := 0.03
def principal2 : ℝ := 1200
def rate2 : ℝ := 0.05
def time : ℝ := 3.888888888888889

-- Calculate the interest for one year for each loan
def interest_per_year1 : ℝ := principal1 * rate1
def interest_per_year2 : ℝ := principal2 * rate2

-- Calculate the total interest for the time period for each loan
def total_interest1 : ℝ := interest_per_year1 * time
def total_interest2 : ℝ := interest_per_year2 * time

-- Finally, calculate the total interest amount
def total_interest_amount : ℝ := total_interest1 + total_interest2

-- The proof problem: Prove that total_interest_amount == 350 Rs
theorem total_interest_is_350 : total_interest_amount = 350 := by
  sorry

end total_interest_is_350_l213_213766


namespace sticker_ratio_l213_213858

theorem sticker_ratio (gold : ℕ) (silver : ℕ) (bronze : ℕ)
  (students : ℕ) (stickers_per_student : ℕ)
  (h1 : gold = 50)
  (h2 : bronze = silver - 20)
  (h3 : students = 5)
  (h4 : stickers_per_student = 46)
  (h5 : gold + silver + bronze = students * stickers_per_student) :
  silver / gold = 2 / 1 :=
by
  sorry

end sticker_ratio_l213_213858


namespace remainder_ab_cd_l213_213995

theorem remainder_ab_cd (n : ℕ) (hn: n > 0) (a b c d : ℤ) 
  (hac : a * c ≡ 1 [ZMOD n]) (hbd : b * d ≡ 1 [ZMOD n]) : 
  (a * b + c * d) % n = 2 :=
by
  sorry

end remainder_ab_cd_l213_213995


namespace initial_amount_l213_213208

theorem initial_amount (X : ℚ) (F : ℚ) :
  (∀ (X F : ℚ), F = X * (3/4)^3 → F = 37 → X = 37 * 64 / 27) :=
by
  sorry

end initial_amount_l213_213208


namespace sum_of_numbers_l213_213964

theorem sum_of_numbers (a : ℝ) (n : ℕ) (h : a = 5.3) (hn : n = 8) : (a * n) = 42.4 :=
sorry

end sum_of_numbers_l213_213964


namespace relation_between_y_l213_213582

/-- Definition of the points on the parabola y = -(x-3)^2 - 4 --/
def pointA (y₁ : ℝ) : Prop := y₁ = -(1/4 - 3)^2 - 4
def pointB (y₂ : ℝ) : Prop := y₂ = -(1 - 3)^2 - 4
def pointC (y₃ : ℝ) : Prop := y₃ = -(4 - 3)^2 - 4 

/-- Relationship between y₁, y₂, y₃ for given points on the quadratic function --/
theorem relation_between_y (y₁ y₂ y₃ : ℝ) 
  (hA : pointA y₁)
  (hB : pointB y₂)
  (hC : pointC y₃) : 
  y₁ < y₂ ∧ y₂ < y₃ := by
  sorry

end relation_between_y_l213_213582


namespace calc_expression_l213_213623

theorem calc_expression : 2 * 0 * 1 + 1 = 1 :=
by
  sorry

end calc_expression_l213_213623


namespace necessary_sufficient_condition_l213_213521

theorem necessary_sufficient_condition (n : ℕ) (hn : n > 0) :
  (∃ x : ℚ, x^n + (2 + x)^n + (2 - x)^n = 0) ↔ n = 1 :=
sorry

end necessary_sufficient_condition_l213_213521


namespace megan_roles_other_than_lead_l213_213446

def total_projects : ℕ := 800

def theater_percentage : ℚ := 50 / 100
def films_percentage : ℚ := 30 / 100
def television_percentage : ℚ := 20 / 100

def theater_lead_percentage : ℚ := 55 / 100
def theater_support_percentage : ℚ := 30 / 100
def theater_ensemble_percentage : ℚ := 10 / 100
def theater_cameo_percentage : ℚ := 5 / 100

def films_lead_percentage : ℚ := 70 / 100
def films_support_percentage : ℚ := 20 / 100
def films_minor_percentage : ℚ := 7 / 100
def films_cameo_percentage : ℚ := 3 / 100

def television_lead_percentage : ℚ := 60 / 100
def television_support_percentage : ℚ := 25 / 100
def television_recurring_percentage : ℚ := 10 / 100
def television_guest_percentage : ℚ := 5 / 100

theorem megan_roles_other_than_lead :
  let theater_projects := total_projects * theater_percentage
  let films_projects := total_projects * films_percentage
  let television_projects := total_projects * television_percentage

  let theater_other_roles := (theater_projects * theater_support_percentage) + 
                             (theater_projects * theater_ensemble_percentage) + 
                             (theater_projects * theater_cameo_percentage)

  let films_other_roles := (films_projects * films_support_percentage) + 
                           (films_projects * films_minor_percentage) + 
                           (films_projects * films_cameo_percentage)

  let television_other_roles := (television_projects * television_support_percentage) + 
                                (television_projects * television_recurring_percentage) + 
                                (television_projects * television_guest_percentage)
  
  theater_other_roles + films_other_roles + television_other_roles = 316 :=
by
  sorry

end megan_roles_other_than_lead_l213_213446


namespace solve_for_x_l213_213924

theorem solve_for_x {x : ℝ} (h : -3 * x - 10 = 4 * x + 5) : x = -15 / 7 :=
  sorry

end solve_for_x_l213_213924


namespace find_kn_l213_213080

section
variables (k n : ℝ)

def system_infinite_solutions (k n : ℝ) :=
  ∃ (y : ℝ → ℝ) (x : ℝ → ℝ),
  (∀ y, k * y + x y + n = 0) ∧
  (∀ y, |y - 2| + |y + 1| + |1 - y| + |y + 2| + x y = 0)

theorem find_kn :
  { (k, n) | system_infinite_solutions k n } = {(4, 0), (-4, 0), (2, 4), (-2, 4), (0, 6)} :=
sorry
end

end find_kn_l213_213080


namespace extra_yellow_balls_dispatched_eq_49_l213_213771

-- Define the given conditions
def ordered_balls : ℕ := 114
def white_balls : ℕ := ordered_balls / 2
def yellow_balls := ordered_balls / 2

-- Define the additional yellow balls dispatched and the ratio condition
def dispatch_error_ratio : ℚ := 8 / 15

-- The statement to prove the number of extra yellow balls dispatched
theorem extra_yellow_balls_dispatched_eq_49
  (ordered_balls_rounded : ordered_balls = 114)
  (white_balls_57 : white_balls = 57)
  (yellow_balls_57 : yellow_balls = 57)
  (ratio_condition : white_balls / (yellow_balls + x) = dispatch_error_ratio) :
  x = 49 :=
  sorry

end extra_yellow_balls_dispatched_eq_49_l213_213771


namespace units_digit_F500_is_7_l213_213300

def F (n : ℕ) : ℕ := 2 ^ (2 ^ (2 * n)) + 1

theorem units_digit_F500_is_7 : (F 500) % 10 = 7 := 
  sorry

end units_digit_F500_is_7_l213_213300


namespace opposite_directions_l213_213955

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem opposite_directions (a b : V) (h : a + 4 • b = 0) : a = -4 • b := sorry

end opposite_directions_l213_213955


namespace sum_of_eight_numbers_l213_213968

theorem sum_of_eight_numbers (avg : ℝ) (S : ℝ) (h1 : avg = 5.3) (h2 : avg = S / 8) : S = 42.4 :=
by
  sorry

end sum_of_eight_numbers_l213_213968


namespace ratio_of_middle_angle_l213_213746

theorem ratio_of_middle_angle (A B C : ℝ) 
  (h1 : A + B + C = 180)
  (h2 : C = 5 * A)
  (h3 : A = 20) :
  B / A = 3 :=
by
  sorry

end ratio_of_middle_angle_l213_213746


namespace find_r_l213_213572

noncomputable def r_value (a b : ℝ) (h : a * b = 3) : ℝ :=
  let r := (a^2 + 1 / b^2) * (b^2 + 1 / a^2)
  r

theorem find_r (a b : ℝ) (h : a * b = 3) : r_value a b h = 100 / 9 := by
  sorry

end find_r_l213_213572


namespace placement_possible_l213_213986

def can_place_numbers : Prop :=
  ∃ (top bottom : Fin 4 → ℕ), 
    (∀ i, (top i) ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) ∧
    (∀ i, (bottom i) ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) ∧
    (List.product (List.ofFn top) = List.product (List.ofFn bottom))

theorem placement_possible : can_place_numbers :=
sorry

end placement_possible_l213_213986


namespace valid_numbers_l213_213753

noncomputable def is_valid_number (a : ℕ) : Prop :=
  ∃ b c d x y : ℕ, 
    a = b * c + d ∧
    a = 10 * x + y ∧
    x > 0 ∧ x ≤ 9 ∧ y ≤ 9 ∧
    10 * x + y = 4 * x + 4 * y

theorem valid_numbers : 
  ∃ a : ℕ, (a = 12 ∨ a = 24 ∨ a = 36 ∨ a = 48) ∧ is_valid_number a :=
by
  sorry

end valid_numbers_l213_213753


namespace sin_half_angle_product_lt_quarter_l213_213498

theorem sin_half_angle_product_lt_quarter (A B C : ℝ) (h : A + B + C = 180) :
    Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) < 1 / 4 := 
    sorry

end sin_half_angle_product_lt_quarter_l213_213498


namespace proposition_D_l213_213506

theorem proposition_D (a b c d : ℝ) (h1 : a < b) (h2 : c < d) : a + c < b + d :=
sorry

end proposition_D_l213_213506


namespace students_speaking_both_languages_l213_213195

theorem students_speaking_both_languages:
  ∀ (total E T N B : ℕ),
    total = 150 →
    E = 55 →
    T = 85 →
    N = 30 →
    (total - N) = 120 →
    (E + T - B) = 120 → B = 20 :=
by
  intros total E T N B h_total h_E h_T h_N h_langs h_equiv
  sorry

end students_speaking_both_languages_l213_213195


namespace ceil_neg_3_7_l213_213076

def x : ℝ := -3.7

def ceil_function (x : ℝ) : ℤ := Int.ceil x

theorem ceil_neg_3_7 : ceil_function x = -3 := 
by 
  -- Translate the conditions into Lean 4 conditions, 
  -- equivalently define x and the function ceil_function
  sorry

end ceil_neg_3_7_l213_213076


namespace complement_A_eq_B_subset_complement_A_l213_213147

-- Definitions of sets A and B
def A : Set ℝ := {x | x^2 + 4 * x > 0 }
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1 }

-- The universal set U is the set of all real numbers
def U : Set ℝ := Set.univ

-- Complement of A in U
def complement_U_A : Set ℝ := {x | -4 ≤ x ∧ x ≤ 0}

-- Proof statement for part (1)
theorem complement_A_eq : complement_U_A = {x | -4 ≤ x ∧ x ≤ 0} :=
  sorry 

-- Proof statement for part (2)
theorem B_subset_complement_A (a : ℝ) : B a ⊆ complement_U_A ↔ -3 ≤ a ∧ a ≤ -1 :=
  sorry 

end complement_A_eq_B_subset_complement_A_l213_213147


namespace evaluate_expression_l213_213798

theorem evaluate_expression : 3 + 2 * (8 - 3) = 13 := by
  sorry

end evaluate_expression_l213_213798


namespace consecutive_integers_eq_l213_213311

theorem consecutive_integers_eq (a b c d e: ℕ) (h1: b = a + 1) (h2: c = a + 2) (h3: d = a + 3) (h4: e = a + 4) (h5: a^2 + b^2 + c^2 = d^2 + e^2) : a = 10 :=
by
  sorry

end consecutive_integers_eq_l213_213311


namespace min_value_x_plus_y_l213_213677

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 19 / x + 98 / y = 1) : 
  x + y ≥ 117 + 14 * Real.sqrt 38 :=
  sorry

end min_value_x_plus_y_l213_213677


namespace sum_of_eight_numbers_l213_213974

theorem sum_of_eight_numbers (average : ℝ) (count : ℕ) (h_avg : average = 5.3) (h_count : count = 8) : (average * count = 42.4) := sorry

end sum_of_eight_numbers_l213_213974


namespace relationship_between_sets_l213_213822

def M : Set ℤ := {x | ∃ k : ℤ, x = 5 * k - 2}
def P : Set ℤ := {x | ∃ n : ℤ, x = 5 * n + 3}
def S : Set ℤ := {x | ∃ m : ℤ, x = 10 * m + 3}

theorem relationship_between_sets : S ⊆ P ∧ P = M := by
  sorry

end relationship_between_sets_l213_213822


namespace solve_integers_l213_213870

theorem solve_integers (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  x^(2 * y) + (x + 1)^(2 * y) = (x + 2)^(2 * y) → (x = 3 ∧ y = 1) :=
by
  sorry

end solve_integers_l213_213870


namespace range_of_m_l213_213874

def one_root_condition (m : ℝ) : Prop :=
  (4 - 4 * m) * (2 * m + 4) ≤ 0 ∧ m ≠ 0

theorem range_of_m : {m : ℝ | (4 - 4 * m) * (2 * m + 4) ≤ 0 ∧ m ≠ 0} = {m | m ≤ -2 ∨ m ≥ 1} :=
by
  sorry

end range_of_m_l213_213874


namespace solve_eq1_solve_eq2_l213_213871

theorem solve_eq1 (x : ℤ) : x - 2 * (5 + x) = -4 → x = -6 := by
  sorry

theorem solve_eq2 (x : ℤ) : (2 * x - 1) / 2 = 1 - (3 - x) / 4 → x = 1 := by
  sorry

end solve_eq1_solve_eq2_l213_213871


namespace part1_part2_l213_213685

noncomputable def f (x a : ℝ) : ℝ := x^2 + a * x + 6

-- Part (I)
theorem part1 (a : ℝ) (h : a = 5) : ∀ x : ℝ, f x 5 < 0 ↔ -3 < x ∧ x < -2 := by
  sorry

-- Part (II)
theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > 0) ↔ -2 * Real.sqrt 6 < a ∧ a < 2 * Real.sqrt 6 := by
  sorry

end part1_part2_l213_213685


namespace number_of_candies_in_a_packet_l213_213222

theorem number_of_candies_in_a_packet 
  (two_packets : ℕ)
  (candies_per_day_weekday : ℕ)
  (candies_per_day_weekend : ℕ)
  (weeks : ℕ)
  (total_candies : ℕ)
  (packet_size : ℕ)
  (H1 : two_packets = 2)
  (H2 : candies_per_day_weekday = 2)
  (H3 : candies_per_day_weekend = 1)
  (H4 : weeks = 3)
  (H5 : packet_size > 0)
  (H6 : total_candies = packets * packet_size)
  (H7 : total_candies = 3 * (5 * candies_per_day_weekday + 2 * candies_per_day_weekend))
  : packet_size = 18 :=
by
  sorry

end number_of_candies_in_a_packet_l213_213222


namespace max_value_of_n_l213_213528

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (S_sum : ∀ n, S n = n * (a 1 + a n) / 2)
variable (S_2015_pos : S 2015 > 0)
variable (S_2016_neg : S 2016 < 0)

theorem max_value_of_n (a : ℕ → ℝ) (S : ℕ → ℝ)
  (S_sum : ∀ n, S n = n * (a 1 + a n) / 2)
  (S_2015_pos : S 2015 > 0)
  (S_2016_neg : S 2016 < 0) : 
  ∃ n, n = 1008 ∧ ∀ m, S m < S n := 
sorry

end max_value_of_n_l213_213528


namespace product_of_possible_x_values_l213_213830

theorem product_of_possible_x_values : 
  (∃ x1 x2 : ℚ, 
    (|15 / x1 + 4| = 3 ∧ |15 / x2 + 4| = 3) ∧
    -15 * -(15 / 7) = (225 / 7)) :=
sorry

end product_of_possible_x_values_l213_213830


namespace Alyssa_spent_in_total_l213_213213

def amount_paid_for_grapes : ℝ := 12.08
def refund_for_cherries : ℝ := 9.85
def total_spent : ℝ := amount_paid_for_grapes - refund_for_cherries

theorem Alyssa_spent_in_total : total_spent = 2.23 := by
  sorry

end Alyssa_spent_in_total_l213_213213


namespace ratio_of_radii_l213_213421

open Real

theorem ratio_of_radii (a b : ℝ) (h : π * b^2 - π * a^2 = 5 * π * a^2) : a / b = 1 / sqrt 6 :=
by
  sorry

end ratio_of_radii_l213_213421


namespace fraction_value_l213_213929

variable (x y : ℚ)

theorem fraction_value (h₁ : x = 4 / 6) (h₂ : y = 8 / 12) : 
  (6 * x + 8 * y) / (48 * x * y) = 7 / 16 :=
by
  sorry

end fraction_value_l213_213929


namespace merchant_gross_profit_l213_213036

noncomputable def grossProfit (purchase_price : ℝ) (selling_price : ℝ) (discount : ℝ) : ℝ :=
  (selling_price - discount * selling_price) - purchase_price

theorem merchant_gross_profit :
  let P := 56
  let S := (P / 0.70 : ℝ)
  let discount := 0.20
  grossProfit P S discount = 8 := 
by
  let P := 56
  let S := (P / 0.70 : ℝ)
  let discount := 0.20
  unfold grossProfit
  sorry

end merchant_gross_profit_l213_213036


namespace bus_trip_cost_l213_213864

-- Problem Statement Definitions
def distance_AB : ℕ := 4500
def cost_per_kilometer_bus : ℚ := 0.20

-- Theorem Statement
theorem bus_trip_cost : distance_AB * cost_per_kilometer_bus = 900 := by
  sorry

end bus_trip_cost_l213_213864


namespace sufficient_conditions_for_x_squared_lt_one_l213_213364

variable (x : ℝ)

theorem sufficient_conditions_for_x_squared_lt_one :
  (∀ x, (0 < x ∧ x < 1) → (x^2 < 1)) ∧
  (∀ x, (-1 < x ∧ x < 0) → (x^2 < 1)) ∧
  (∀ x, (-1 < x ∧ x < 1) → (x^2 < 1)) :=
by
  sorry

end sufficient_conditions_for_x_squared_lt_one_l213_213364


namespace xyz_value_l213_213942

theorem xyz_value (x y z : ℝ) (h1 : (x + y + z) * (x * y + x * z + y * z) = 30) 
    (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9) : x * y * z = 7 :=
by
  sorry

end xyz_value_l213_213942


namespace num_divisors_of_36_l213_213120

theorem num_divisors_of_36 : (∃ (S : Finset ℤ), (∀ x, x ∈ S ↔ x ∣ 36) ∧ S.card = 18) :=
sorry

end num_divisors_of_36_l213_213120


namespace composite_2011_2014_composite_2012_2015_l213_213704

theorem composite_2011_2014 :
  let N := 2011 * 2012 * 2013 * 2014 + 1
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ N = a * b := 
by
  let N := 2011 * 2012 * 2013 * 2014 + 1
  sorry
  
theorem composite_2012_2015 :
  let N := 2012 * 2013 * 2014 * 2015 + 1
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ N = a * b := 
by
  let N := 2012 * 2013 * 2014 * 2015 + 1
  sorry

end composite_2011_2014_composite_2012_2015_l213_213704


namespace find_unique_p_l213_213662

theorem find_unique_p (p : ℝ) (h1 : p ≠ 0) : (∀ x : ℝ, p * x^2 - 10 * x + 2 = 0 → p = 12.5) :=
by sorry

end find_unique_p_l213_213662


namespace series_sum_half_l213_213650

theorem series_sum_half :
  ∑' n : ℕ, (2^n / (3^(2^n) + 1)) = 1/2 := 
sorry

end series_sum_half_l213_213650


namespace two_n_plus_m_is_36_l213_213174

theorem two_n_plus_m_is_36 (n m : ℤ) (h1 : 3 * n - m < 5) (h2 : n + m > 26) (h3 : 3 * m - 2 * n < 46) : 
  2 * n + m = 36 :=
sorry

end two_n_plus_m_is_36_l213_213174


namespace no_viable_schedule_l213_213770

theorem no_viable_schedule :
  ∀ (studentsA studentsB : ℕ), 
    studentsA = 29 → 
    studentsB = 32 → 
    ¬ ∃ (a b : ℕ),
      (a = 29 ∧ b = 32 ∧
      (a * b = studentsA * studentsB) ∧
      (∀ (x : ℕ), x < studentsA * studentsB →
        ∃ (iA iB : ℕ), 
          iA < studentsA ∧ 
          iB < studentsB ∧ 
          -- The condition that each pair is unique within this period
          ((iA + iB) % (studentsA * studentsB) = x))) := by
  sorry

end no_viable_schedule_l213_213770


namespace roof_collapse_days_l213_213065

def leaves_per_pound : ℕ := 1000
def pounds_limit_of_roof : ℕ := 500
def leaves_per_day : ℕ := 100

theorem roof_collapse_days : (pounds_limit_of_roof * leaves_per_pound) / leaves_per_day = 5000 := by
  sorry

end roof_collapse_days_l213_213065


namespace total_computers_needed_l213_213341

theorem total_computers_needed (initial_students : ℕ) (students_per_computer : ℕ) (additional_students : ℕ) :
  initial_students = 82 →
  students_per_computer = 2 →
  additional_students = 16 →
  (initial_students + additional_students) / students_per_computer = 49 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end total_computers_needed_l213_213341


namespace at_least_one_not_less_than_six_l213_213533

-- Definitions for the conditions.
variables {a b c : ℝ}
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- The proof statement.
theorem at_least_one_not_less_than_six :
  (a + 4 / b) < 6 ∧ (b + 9 / c) < 6 ∧ (c + 16 / a) < 6 → false :=
by
  sorry

end at_least_one_not_less_than_six_l213_213533


namespace series_convergence_l213_213652

theorem series_convergence :
  ∑ n : ℕ, (2^n) / (3^(2^n) + 1) = 1 / 2 :=
by
  sorry

end series_convergence_l213_213652


namespace difference_of_numbers_l213_213605

theorem difference_of_numbers (x y : ℝ) (h1 : x * y = 23) (h2 : x + y = 24) : |x - y| = 22 :=
sorry

end difference_of_numbers_l213_213605


namespace first_instance_height_35_l213_213001
noncomputable def projectile_height (t : ℝ) : ℝ := -5 * t^2 + 30 * t

theorem first_instance_height_35 {t : ℝ} (h : projectile_height t = 35) :
  t = 3 - Real.sqrt 2 :=
sorry

end first_instance_height_35_l213_213001


namespace half_angle_second_quadrant_l213_213696

theorem half_angle_second_quadrant (k : ℤ) (α : ℝ) (hα : 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) :
    ∃ j : ℤ, (j * π + π / 4 < α / 2 ∧ α / 2 < j * π + π / 2) ∨ (j * π + 5 * π / 4 < α / 2 ∧ α / 2 < (j + 1) * π / 2) :=
sorry

end half_angle_second_quadrant_l213_213696


namespace jacob_younger_than_michael_l213_213429

-- Definitions based on the conditions.
def jacob_current_age : ℕ := 9
def michael_current_age : ℕ := 2 * (jacob_current_age + 3) - 3

-- Theorem to prove that Jacob is 12 years younger than Michael.
theorem jacob_younger_than_michael : michael_current_age - jacob_current_age = 12 :=
by
  -- Placeholder for proof
  sorry

end jacob_younger_than_michael_l213_213429


namespace find_adult_buffet_price_l213_213153

variable {A : ℝ} -- Let A be the price for the adult buffet
variable (children_cost : ℝ := 45) -- Total cost for the children's buffet
variable (senior_discount : ℝ := 0.9) -- Discount for senior citizens
variable (total_cost : ℝ := 159) -- Total amount spent by Mr. Smith
variable (num_adults : ℕ := 2) -- Number of adults (Mr. Smith and his wife)
variable (num_seniors : ℕ := 2) -- Number of senior citizens

theorem find_adult_buffet_price (h1 : children_cost = 45)
    (h2 : total_cost = 159)
    (h3 : ∀ x, num_adults * x + num_seniors * (senior_discount * x) + children_cost = total_cost)
    : A = 30 :=
by
  sorry

end find_adult_buffet_price_l213_213153


namespace oprah_winfrey_band_weights_l213_213138

theorem oprah_winfrey_band_weights :
  let weight_trombone := 10
  let weight_tuba := 20
  let weight_drum := 15
  let num_trumpets := 6
  let num_clarinets := 9
  let num_trombones := 8
  let num_tubas := 3
  let num_drummers := 2
  let total_weight := 245

  15 * x = total_weight - (num_trombones * weight_trombone + num_tubas * weight_tuba + num_drummers * weight_drum) 
  → x = 5 := by
  sorry

end oprah_winfrey_band_weights_l213_213138


namespace find_x_plus_y_l213_213268

theorem find_x_plus_y (x y : ℚ) (h1 : |x| + x + y = 10) (h2 : x + |y| - y = 12) : x + y = 18 / 5 := 
by
  sorry

end find_x_plus_y_l213_213268


namespace num_of_nickels_l213_213907

theorem num_of_nickels (n : ℕ) (h1 : n = 17) (h2 : (17 * n) - 1 = 18 * (n - 1)) : n = 17 → 17 * n = 289 → ∃ k, k = 2 :=
by 
  intros hn hv
  sorry

end num_of_nickels_l213_213907


namespace sum_of_roots_eq_three_l213_213083

-- Definitions of the polynomials
def poly1 (x : ℝ) : ℝ := 3 * x^3 + 3 * x^2 - 9 * x + 27
def poly2 (x : ℝ) : ℝ := 4 * x^3 - 16 * x^2 + 5

-- Theorem stating the sum of the roots of the given equation is 3
theorem sum_of_roots_eq_three : 
  (∀ a b c d e f g h i : ℝ, 
    (poly1 a = 0) → (poly1 b = 0) → (poly1 c = 0) → 
    (poly2 d = 0) → (poly2 e = 0) → (poly2 f = 0) →
    a + b + c + d + e + f = 3) := 
by
  sorry

end sum_of_roots_eq_three_l213_213083


namespace tangent_point_at_slope_one_l213_213099

-- Define the curve
def curve (x : ℝ) : ℝ := x^2 - 3 * x

-- Define the derivative of the curve
def derivative (x : ℝ) : ℝ := 2 * x - 3

-- State the theorem proof problem
theorem tangent_point_at_slope_one : ∃ x : ℝ, derivative x = 1 ∧ x = 2 :=
by
  sorry

end tangent_point_at_slope_one_l213_213099


namespace order_of_numbers_l213_213473

theorem order_of_numbers :
  2^30 < 10^10 ∧ 10^10 < 5^15 :=
by
  sorry

end order_of_numbers_l213_213473


namespace weight_of_square_piece_l213_213909

open Real

theorem weight_of_square_piece 
  (uniform_density : Prop)
  (side_length_triangle side_length_square : ℝ)
  (weight_triangle : ℝ)
  (ht : side_length_triangle = 6)
  (hs : side_length_square = 6)
  (wt : weight_triangle = 48) :
  ∃ weight_square : ℝ, weight_square = 27.7 :=
by
  sorry

end weight_of_square_piece_l213_213909


namespace calc_abc_squares_l213_213643

theorem calc_abc_squares :
  ∀ (a b c : ℝ),
  a^2 + 3 * b = 14 →
  b^2 + 5 * c = -13 →
  c^2 + 7 * a = -26 →
  a^2 + b^2 + c^2 = 20.75 :=
by
  intros a b c h1 h2 h3
  -- The proof is omitted; reasoning is provided in the solution.
  sorry

end calc_abc_squares_l213_213643


namespace brendas_age_l213_213059

theorem brendas_age (A B J : ℝ) 
  (h1 : A = 4 * B)
  (h2 : J = B + 7)
  (h3 : A = J) : 
  B = 7 / 3 := 
by 
  sorry

end brendas_age_l213_213059


namespace base_area_of_cuboid_l213_213202

theorem base_area_of_cuboid (V h : ℝ) (hv : V = 144) (hh : h = 8) : ∃ A : ℝ, A = 18 := by
  sorry

end base_area_of_cuboid_l213_213202


namespace minimal_t_l213_213717

noncomputable def f (n : ℕ) : ℕ := (Real.log n / Real.log 2).floor + 1

def S (n : ℕ) : Finset ℕ := Finset.range (n + 1) \ {0}

def F (S : Finset ℕ) (t : ℕ) : Type :=
  {F : Finset (Finset ℕ) // F.card = t ∧ 
                           (∀ x y ∈ S, x ≠ y → ∃ A ∈ F, (A ∩ {x, y}).card = 1) ∧ 
                           (∀ A ∈ F, A ⊆ S) ∧ 
                           (S ⊆ Finset.bUnion F id)}

theorem minimal_t (n : ℕ) (h : 2 ≤ n) : 
  ∃ t, ∀ (F, ht), @F (S n) t → t = f n := 
sorry

end minimal_t_l213_213717


namespace most_stable_yield_l213_213890

theorem most_stable_yield (S_A S_B S_C S_D : ℝ)
  (h₁ : S_A = 3.6)
  (h₂ : S_B = 2.89)
  (h₃ : S_C = 13.4)
  (h₄ : S_D = 20.14) : 
  S_B < S_A ∧ S_B < S_C ∧ S_B < S_D :=
by {
  sorry -- Proof skipped as per instructions
}

end most_stable_yield_l213_213890


namespace value_of_f_nine_halves_l213_213815

noncomputable def f : ℝ → ℝ := sorry  -- Define f with noncomputable since it's not explicitly given

axiom even_function (x : ℝ) : f x = f (-x)  -- Define the even function property
axiom not_identically_zero : ∃ x : ℝ, f x ≠ 0 -- Define the property that f is not identically zero
axiom functional_equation (x : ℝ) : x * f (x + 1) = (x + 1) * f x -- Define the given functional equation

theorem value_of_f_nine_halves : f (9 / 2) = 0 := by
  sorry

end value_of_f_nine_halves_l213_213815


namespace rationalize_denominator_l213_213866

-- Lean 4 statement
theorem rationalize_denominator : sqrt (5 / 18) = sqrt 10 / 6 := by
  sorry

end rationalize_denominator_l213_213866


namespace students_count_rental_cost_l213_213908

theorem students_count (k m : ℕ) (n : ℕ) 
  (h1 : n = 35 * k)
  (h2 : n = 55 * (m - 1) + 45) : 
  n = 175 := 
by {
  sorry
}

theorem rental_cost (x y : ℕ) 
  (total_buses : x + y = 4)
  (cost_limit : 35 * x + 55 * y ≤ 1500) : 
  320 * x + 400 * y = 1440 := 
by {
  sorry 
}

end students_count_rental_cost_l213_213908


namespace initial_average_is_correct_l213_213245

def initial_average_daily_production (n : ℕ) (today_production : ℕ) (new_average : ℕ) (initial_average : ℕ) :=
  let total_initial_production := initial_average * n
  let total_new_production := total_initial_production + today_production
  let total_days := n + 1
  total_new_production = new_average * total_days

theorem initial_average_is_correct :
  ∀ (A n today_production new_average : ℕ),
    n = 19 →
    today_production = 90 →
    new_average = 52 →
    initial_average_daily_production n today_production new_average A →
    A = 50 := by
    intros A n today_production new_average hn htoday hnew havg
    sorry

end initial_average_is_correct_l213_213245


namespace sum_of_numbers_l213_213967

theorem sum_of_numbers (a : ℝ) (n : ℕ) (h : a = 5.3) (hn : n = 8) : (a * n) = 42.4 :=
sorry

end sum_of_numbers_l213_213967


namespace coordinate_relationship_l213_213917

theorem coordinate_relationship (x y : ℝ) (h : |x| - |y| = 0) : (|x| - |y| = 0) :=
by
    sorry

end coordinate_relationship_l213_213917


namespace find_C_l213_213011

theorem find_C (A B C : ℕ) (h1 : 3 * A - A = 10) (h2 : B + A = 12) (h3 : C - B = 6) : C = 13 :=
by
  sorry

end find_C_l213_213011


namespace visitors_not_enjoyed_not_understood_l213_213329

theorem visitors_not_enjoyed_not_understood (V E U : ℕ) (hv_v : V = 520)
  (hu_e : E = U) (he : E = 3 * V / 4) : (V / 4) = 130 :=
by
  rw [hv_v] at he
  sorry

end visitors_not_enjoyed_not_understood_l213_213329


namespace youth_gathering_l213_213064

theorem youth_gathering (x : ℕ) (h1 : ∃ x, 9 * (2 * x + 12) = 20 * x) : 
  2 * x + 12 = 120 :=
by sorry

end youth_gathering_l213_213064


namespace price_per_gaming_chair_l213_213570

theorem price_per_gaming_chair 
  (P : ℝ)
  (price_per_organizer : ℝ := 78)
  (num_organizers : ℕ := 3)
  (num_chairs : ℕ := 2)
  (total_paid : ℝ := 420)
  (delivery_fee_rate : ℝ := 0.05) 
  (cost_organizers : ℝ := num_organizers * price_per_organizer)
  (cost_gaming_chairs : ℝ := num_chairs * P)
  (total_sales : ℝ := cost_organizers + cost_gaming_chairs)
  (delivery_fee : ℝ := delivery_fee_rate * total_sales) :
  total_paid = total_sales + delivery_fee → P = 83 := 
sorry

end price_per_gaming_chair_l213_213570


namespace sum_of_eight_numbers_l213_213973

theorem sum_of_eight_numbers (average : ℝ) (count : ℕ) (h_avg : average = 5.3) (h_count : count = 8) : (average * count = 42.4) := sorry

end sum_of_eight_numbers_l213_213973


namespace find_x_l213_213755

theorem find_x :
  ∃ x : ℝ, 3 * x = (26 - x) + 14 ∧ x = 10 :=
by sorry

end find_x_l213_213755


namespace find_k_l213_213829

theorem find_k (k : ℝ) (h : (-3 : ℝ)^2 + (-3 : ℝ) - k = 0) : k = 6 :=
by
  sorry

end find_k_l213_213829


namespace y_minus_x_value_l213_213005

theorem y_minus_x_value (x y : ℝ) (h1 : x + y = 500) (h2 : x / y = 0.8) : y - x = 55.56 :=
sorry

end y_minus_x_value_l213_213005


namespace correct_tile_for_b_l213_213520

structure Tile where
  top : ℕ
  right : ℕ
  bottom : ℕ
  left : ℕ

def TileI : Tile := {top := 5, right := 3, bottom := 1, left := 6}
def TileII : Tile := {top := 2, right := 6, bottom := 3, left := 5}
def TileIII : Tile := {top := 6, right := 1, bottom := 4, left := 2}
def TileIV : Tile := {top := 4, right := 5, bottom := 2, left := 1}

def RectangleBTile := TileIII

theorem correct_tile_for_b : RectangleBTile = TileIII :=
  sorry

end correct_tile_for_b_l213_213520


namespace find_custom_operator_result_l213_213440

def custom_operator (a b : ℝ) : ℝ := 4 * a + 3 * b

theorem find_custom_operator_result :
  custom_operator 2 5 = 23 :=
by
  sorry

end find_custom_operator_result_l213_213440


namespace trigonometric_identity_solution_l213_213803

theorem trigonometric_identity_solution 
  (alpha beta : ℝ)
  (h1 : π / 4 < alpha)
  (h2 : alpha < 3 * π / 4)
  (h3 : 0 < beta)
  (h4 : beta < π / 4)
  (h5 : Real.cos (π / 4 + alpha) = -4 / 5)
  (h6 : Real.sin (3 * π / 4 + beta) = 12 / 13) :
  (Real.sin (alpha + beta) = 63 / 65) ∧
  (Real.cos (alpha - beta) = -33 / 65) :=
by
  sorry

end trigonometric_identity_solution_l213_213803


namespace binary_arithmetic_l213_213067

theorem binary_arithmetic :
  let a := 0b1101
  let b := 0b0110
  let c := 0b1011
  let d := 0b1001
  a + b - c + d = 0b10001 := by
sorry

end binary_arithmetic_l213_213067


namespace measure_of_angle_ABC_l213_213126

-- Define the angles involved and their respective measures
def angle_CBD : ℝ := 90 -- Given that angle CBD is a right angle
def angle_sum : ℝ := 160 -- Sum of the angles around point B
def angle_ABD : ℝ := 50 -- Given angle ABD

-- Define angle ABC to be determined
def angle_ABC : ℝ := angle_sum - (angle_ABD + angle_CBD)

-- Define the statement
theorem measure_of_angle_ABC :
  angle_ABC = 20 :=
by 
  -- Calculations omitted
  sorry

end measure_of_angle_ABC_l213_213126


namespace necessary_and_sufficient_condition_l213_213915

theorem necessary_and_sufficient_condition (a b : ℝ) : a > b ↔ a^3 > b^3 :=
by {
  sorry
}

end necessary_and_sufficient_condition_l213_213915


namespace contrapositive_proposition_contrapositive_version_l213_213086

variable {a b : ℝ}

theorem contrapositive_proposition (h : a + b = 1) : a^2 + b^2 ≥ 1/2 :=
sorry

theorem contrapositive_version : a^2 + b^2 < 1/2 → a + b ≠ 1 :=
by
  intros h
  intro hab
  apply not_le.mpr h
  exact contrapositive_proposition hab

end contrapositive_proposition_contrapositive_version_l213_213086


namespace number_of_extremum_points_of_f_l213_213254

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then (x + 1)^3 * Real.exp (x + 1) else (-(x + 1))^3 * Real.exp (-(x + 1))

theorem number_of_extremum_points_of_f :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    ((f (x1 - epsilon) < f x1 ∧ f x1 > f (x1 + epsilon)) ∨ (f (x1 - epsilon) > f x1 ∧ f x1 < f (x1 + epsilon))) ∧
    ((f (x2 - epsilon) < f x2 ∧ f x2 > f (x2 + epsilon)) ∨ (f (x2 - epsilon) > f x2 ∧ f x2 < f (x2 + epsilon))) ∧
    ((f (x3 - epsilon) < f x3 ∧ f x3 > f (x3 + epsilon)) ∨ (f (x3 - epsilon) > f x3 ∧ f x3 < f (x3 + epsilon)))) :=
sorry

end number_of_extremum_points_of_f_l213_213254


namespace bc_over_ad_l213_213716

-- Define the rectangular prism
structure RectangularPrism :=
(length width height : ℝ)

-- Define the problem parameters
def B : RectangularPrism := ⟨2, 4, 5⟩

-- Define the volume form of S(r)
def volume (a b c d : ℝ) (r : ℝ) : ℝ := a * r^3 + b * r^2 + c * r + d

-- Prove that the relationship holds
theorem bc_over_ad (a b c d : ℝ) (r : ℝ) (h_a : a = (4 * π) / 3) (h_b : b = 11 * π) (h_c : c = 76) (h_d : d = 40) :
  (b * c) / (a * d) = 15.67 := by
  sorry

end bc_over_ad_l213_213716


namespace solve_system_a_l213_213308

theorem solve_system_a (x y : ℝ) (h1 : x^2 - 3 * x * y - 4 * y^2 = 0) (h2 : x^3 + y^3 = 65) : 
    x = 4 ∧ y = 1 :=
sorry

end solve_system_a_l213_213308


namespace geometric_sequence_sum_S6_l213_213096

theorem geometric_sequence_sum_S6 (S : ℕ → ℝ) (S_2_eq_4 : S 2 = 4) (S_4_eq_16 : S 4 = 16) :
  S 6 = 52 :=
sorry

end geometric_sequence_sum_S6_l213_213096


namespace solution_set_l213_213539

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 1 else -1

theorem solution_set :
  {x : ℝ | x + (x + 2) * f (x + 2) ≤ 5} = {x : ℝ | x ≤ 3 / 2} :=
by {
  sorry
}

end solution_set_l213_213539


namespace series_sum_l213_213647

theorem series_sum :
  ∑' n : ℕ, (2 ^ n) / (3 ^ (2 ^ n) + 1) = 1 / 2 :=
sorry

end series_sum_l213_213647


namespace Cindy_coins_l213_213066

theorem Cindy_coins (n : ℕ) (h1 : ∃ X Y : ℕ, n = X * Y ∧ Y > 1 ∧ Y < n) (h2 : ∀ Y, Y > 1 ∧ Y < n → ¬Y ∣ n → False) : n = 65536 :=
by
  sorry

end Cindy_coins_l213_213066


namespace min_value_f_at_0_l213_213806

noncomputable def f (a x : ℝ) : ℝ :=
if x ≤ 0 then (x - a)^2 else x + 1/x + a

theorem min_value_f_at_0 (a : ℝ) : (∀ x : ℝ, f a 0 ≤ f a x) ↔ 0 ≤ a ∧ a ≤ 2 :=
by
  sorry

end min_value_f_at_0_l213_213806


namespace tom_fruit_bowl_l213_213616

def initial_lemons (oranges lemons removed remaining : ℕ) : ℕ :=
  lemons

theorem tom_fruit_bowl (oranges removed remaining : ℕ) (L : ℕ) 
  (h_oranges : oranges = 3)
  (h_removed : removed = 3)
  (h_remaining : remaining = 6)
  (h_initial : oranges + L - removed = remaining) : 
  initial_lemons oranges L removed remaining = 6 :=
by
  -- Implement the proof here
  sorry

end tom_fruit_bowl_l213_213616


namespace double_bed_heavier_than_single_bed_l213_213884

theorem double_bed_heavier_than_single_bed 
  (S D : ℝ) 
  (h1 : 5 * S = 50) 
  (h2 : 2 * S + 4 * D = 100) 
  : D - S = 10 :=
sorry

end double_bed_heavier_than_single_bed_l213_213884


namespace equality_am_bn_l213_213738

theorem equality_am_bn (m n : ℝ) (x : ℝ) (a b : ℝ) (hmn : m ≠ n) (hm : m ≠ 0) (hn : n ≠ 0) :
  ((x + m) ^ 2 - (x + n) ^ 2 = (m - n) ^ 2) → (x = am + bn) → (a = 0 ∧ b = -1) :=
by
  intro h1 h2
  sorry

end equality_am_bn_l213_213738


namespace find_x_add_inv_l213_213255

theorem find_x_add_inv (x : ℝ) (h : x^3 + 1 / x^3 = 110) : x + 1 / x = 5 :=
sorry

end find_x_add_inv_l213_213255


namespace south_movement_notation_l213_213832

/-- If moving north 8m is denoted as +8m, then moving south 5m is denoted as -5m. -/
theorem south_movement_notation (north south : ℤ) (h1 : north = 8) (h2 : south = -north) : south = -5 :=
by
  sorry

end south_movement_notation_l213_213832


namespace sum_of_number_and_reverse_l213_213315

def digit_representation (n m : ℕ) (a b : ℕ) :=
  n = 10 * a + b ∧
  m = 10 * b + a ∧
  n - m = 9 * (a * b) + 3

theorem sum_of_number_and_reverse :
  ∃ a b n m : ℕ, digit_representation n m a b ∧ n + m = 22 :=
by
  sorry

end sum_of_number_and_reverse_l213_213315


namespace new_percentage_of_girls_is_5_l213_213838

theorem new_percentage_of_girls_is_5
  (initial_children : ℕ)
  (percentage_boys : ℕ)
  (added_boys : ℕ)
  (initial_total_boys : ℕ)
  (initial_total_girls : ℕ)
  (new_total_boys : ℕ)
  (new_total_children : ℕ)
  (new_percentage_girls : ℕ)
  (h1 : initial_children = 60)
  (h2 : percentage_boys = 90)
  (h3 : added_boys = 60)
  (h4 : initial_total_boys = (percentage_boys * initial_children / 100))
  (h5 : initial_total_girls = initial_children - initial_total_boys)
  (h6 : new_total_boys = initial_total_boys + added_boys)
  (h7 : new_total_children = initial_children + added_boys)
  (h8 : new_percentage_girls = (initial_total_girls * 100 / new_total_children)) :
  new_percentage_girls = 5 :=
by sorry

end new_percentage_of_girls_is_5_l213_213838


namespace preimage_of_mapping_l213_213686

def f (a b : ℝ) : ℝ × ℝ := (a + 2 * b, 2 * a - b)

theorem preimage_of_mapping : ∃ (a b : ℝ), f a b = (3, 1) ∧ (a, b) = (1, 1) :=
by
  sorry

end preimage_of_mapping_l213_213686


namespace factorize_expression_l213_213239

-- Variables used in the expression
variables (m n : ℤ)

-- The expression to be factored
def expr := 4 * m^3 * n - 16 * m * n^3

-- The desired factorized form of the expression
def factored := 4 * m * n * (m + 2 * n) * (m - 2 * n)

-- The proof problem statement
theorem factorize_expression : expr m n = factored m n :=
by sorry

end factorize_expression_l213_213239


namespace value_of_a_plus_d_l213_213759

variable (a b c d : ℝ)

theorem value_of_a_plus_d 
  (h1 : a + b = 4) 
  (h2 : b + c = 5) 
  (h3 : c + d = 3) : 
  a + d = 1 := 
by 
  sorry

end value_of_a_plus_d_l213_213759


namespace fg_of_3_eq_83_l213_213552

def g (x : ℝ) : ℝ := x^3
def f (x : ℝ) : ℝ := 3 * x + 2

theorem fg_of_3_eq_83 : f (g 3) = 83 := by
  sorry

end fg_of_3_eq_83_l213_213552


namespace a_plus_b_l213_213294

-- Definitions and conditions
def f (x : ℝ) (a b : ℝ) := a * x + b
def g (x : ℝ) := 3 * x - 7

theorem a_plus_b (a b : ℝ) (h : ∀ x : ℝ, g (f x a b) = 4 * x + 5) : a + b = 16 / 3 :=
by
  sorry

end a_plus_b_l213_213294


namespace probability_of_drawing_red_ball_l213_213180

theorem probability_of_drawing_red_ball : 
  let box1_initial_white := 2
  let box1_initial_red := 4
  let box2_initial_white := 5
  let box2_initial_red := 3

  let p_white_from_box1 := (box1_initial_white : ℚ) / (box1_initial_white + box1_initial_red)
  let p_red_from_box1 := (box1_initial_red : ℚ) / (box1_initial_white + box1_initial_red)

  let box2_after_white_in_white := box2_initial_white + 1
  let box2_after_white_in_red := box2_initial_red
  let box2_after_red_in_white := box2_initial_white
  let box2_after_red_in_red := box2_initial_red + 1

  let p_red_from_box2_after_white := (box2_after_white_in_red : ℚ) / (box2_after_white_in_white + box2_after_white_in_red)
  let p_red_from_box2_after_red := (box2_after_red_in_red : ℚ) / (box2_after_red_in_white + box2_after_red_in_red)

  let probability := p_white_from_box1 * p_red_from_box2_after_white + p_red_from_box1 * p_red_from_box2_after_red

  probability = 11 / 27 := by
  -- Proof: The detailed steps are omitted
  sorry

end probability_of_drawing_red_ball_l213_213180


namespace oven_capacity_correct_l213_213141

-- Definitions for the conditions
def dough_time := 30 -- minutes
def bake_time := 30 -- minutes
def pizzas_per_batch := 3
def total_time := 5 * 60 -- minutes (5 hours)
def total_pizzas := 12

-- Calculation of the number of batches
def batches_needed := total_pizzas / pizzas_per_batch

-- Calculation of the time for making dough
def dough_preparation_time := batches_needed * dough_time

-- Calculation of the remaining time for baking
def remaining_baking_time := total_time - dough_preparation_time

-- Calculation of the number of 30-minute baking intervals
def baking_intervals := remaining_baking_time / bake_time

-- Calculation of the capacity of the oven
def oven_capacity := total_pizzas / baking_intervals

theorem oven_capacity_correct : oven_capacity = 2 := by
  sorry

end oven_capacity_correct_l213_213141


namespace sin_double_angle_l213_213532

open Real 

theorem sin_double_angle (α β : ℝ) 
  (h1 : π / 2 < β) 
  (h2 : β < α) 
  (h3 : α < 3 * π / 4) 
  (h4 : cos (α - β) = 12 / 13) 
  (h5 : sin (α + β) = -3 / 5) : 
  sin (2 * α) = -56 / 65 := 
by 
  sorry

end sin_double_angle_l213_213532


namespace Daniela_is_12_years_old_l213_213641

noncomputable def auntClaraAge : Nat := 60

noncomputable def evelinaAge : Nat := auntClaraAge / 3

noncomputable def fidelAge : Nat := evelinaAge - 6

noncomputable def caitlinAge : Nat := fidelAge / 2

noncomputable def danielaAge : Nat := evelinaAge - 8

theorem Daniela_is_12_years_old (h_auntClaraAge : auntClaraAge = 60)
                                (h_evelinaAge : evelinaAge = 60 / 3)
                                (h_fidelAge : fidelAge = (60 / 3) - 6)
                                (h_caitlinAge : caitlinAge = ((60 / 3) - 6) / 2)
                                (h_danielaAge : danielaAge = (60 / 3) - 8) :
  danielaAge = 12 := 
  sorry

end Daniela_is_12_years_old_l213_213641


namespace abs_condition_iff_range_l213_213170

theorem abs_condition_iff_range (x : ℝ) : 
  (|x-1| + |x+2| ≤ 5) ↔ (-3 ≤ x ∧ x ≤ 2) := 
sorry

end abs_condition_iff_range_l213_213170


namespace table_tennis_basketball_teams_l213_213302

theorem table_tennis_basketball_teams (X Y : ℕ)
  (h1 : X + Y = 50) 
  (h2 : 7 * Y = 3 * X)
  (h3 : 2 * (X - 8) = 3 * (Y + 8)) :
  X = 35 ∧ Y = 15 :=
by
  sorry

end table_tennis_basketball_teams_l213_213302


namespace number_of_students_l213_213144

-- Define the conditions
variable (n : ℕ) (jayden_rank_best jayden_rank_worst : ℕ)
variable (h1 : jayden_rank_best = 100)
variable (h2 : jayden_rank_worst = 100)

-- Define the question
theorem number_of_students (h1 : jayden_rank_best = 100) (h2 : jayden_rank_worst = 100) : n = 199 := 
  sorry

end number_of_students_l213_213144


namespace alexis_dresses_l213_213705

-- Definitions based on the conditions
def isabella_total : ℕ := 13
def alexis_total : ℕ := 3 * isabella_total
def alexis_pants : ℕ := 21

-- Theorem statement
theorem alexis_dresses : alexis_total - alexis_pants = 18 := by
  sorry

end alexis_dresses_l213_213705


namespace find_value_of_a2_b2_c2_l213_213714

variable {a b c : ℝ}

theorem find_value_of_a2_b2_c2
  (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h2 : a + b + c = 0)
  (h3 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6 / 5 := 
sorry

end find_value_of_a2_b2_c2_l213_213714


namespace largest_4_digit_divisible_by_88_l213_213040

-- Define a predicate for a 4-digit number
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

-- Define a constant representing the number 88
def eighty_eight : ℕ := 88

-- State the main theorem
theorem largest_4_digit_divisible_by_88 : ∃ n : ℕ, is_four_digit n ∧ eighty_eight ∣ n ∧ ∀ m : ℕ, is_four_digit m ∧ eighty_eight ∣ m → m ≤ n :=
begin
  -- We assert that 9944 is the largest 4-digit number divisible by 88
  use 9944,
  split,
  -- Prove that 9944 is a four-digit number
  { split,
    { norm_num },
    { norm_num } },
  -- Prove that 9944 is divisible by 88
  { use 113,
    norm_num },
  -- Prove that 9944 is the largest such number
  { intros m Hm,
    cases Hm with Hm₁ Hm₂,
    have Hm₃ : m ≤ 9999 := by exact Hm₁.2,
    have Hm₄ : ∃ k, m = eighty_eight * k,
    { exact Hm₂ },
    cases Hm₄ with k Hk,
    have key : k ≤ 113 := by sorry,
    rw Hk,
    exact mul_le_mul_right' key eighty_eight }
end

end largest_4_digit_divisible_by_88_l213_213040


namespace base_conversion_l213_213169

theorem base_conversion (b : ℕ) (h : 1 * 6^2 + 4 * 6 + 2 = 2 * b^2 + b + 5) : b = 5 :=
by
  sorry

end base_conversion_l213_213169


namespace heptagon_diagonals_l213_213110

theorem heptagon_diagonals : (7 * (7 - 3)) / 2 = 14 := 
by
  rfl

end heptagon_diagonals_l213_213110


namespace trigonometric_identity_l213_213512

theorem trigonometric_identity : (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 1 - Real.sqrt 3 :=
by
  sorry

end trigonometric_identity_l213_213512


namespace problem1_l213_213045

theorem problem1 
  (α β : Real)
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - Real.pi / 4) = 1/4) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3/22 := sorry

end problem1_l213_213045


namespace summation_series_equals_half_l213_213653

theorem summation_series_equals_half :
  (\sum_{n=0}^{∞} 2^n / (3^(2^n) + 1)) = 1 / 2 :=
by
  sorry

end summation_series_equals_half_l213_213653


namespace spinner_win_sector_area_l213_213764

open Real

theorem spinner_win_sector_area (r : ℝ) (P : ℝ)
  (h_r : r = 8) (h_P : P = 3 / 7) : 
  ∃ A : ℝ, A = 192 * π / 7 :=
by
  sorry

end spinner_win_sector_area_l213_213764


namespace range_of_m_l213_213578

def prop_p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 + y^2 - 2*x - 4*y + m = 0
def prop_q (m : ℝ) : Prop := ∃ (x y : ℝ), (x^2) / (m-6) - (y^2) / (m+3) = 1

theorem range_of_m (m : ℝ) : ¬ (prop_p m ∧ prop_q m) → m ≥ -3 :=
sorry

end range_of_m_l213_213578


namespace angle_at_7_20_is_100_degrees_l213_213918

def angle_between_hands_at_7_20 : ℝ := 100

theorem angle_at_7_20_is_100_degrees
    (hour_hand_pos : ℝ := 210) -- 7 * 30 degrees
    (minute_hand_pos : ℝ := 120) -- 4 * 30 degrees
    (hour_hand_move_per_minute : ℝ := 0.5) -- 0.5 degrees per minute
    (time_past_7_clock : ℝ := 20) -- 20 minutes
    (adjacent_angle : ℝ := 30) -- angle between adjacent numbers
    : angle_between_hands_at_7_20 = 
      (hour_hand_pos - (minute_hand_pos - hour_hand_move_per_minute * time_past_7_clock)) :=
sorry

end angle_at_7_20_is_100_degrees_l213_213918


namespace video_files_initial_l213_213774

theorem video_files_initial (V : ℕ) (h1 : 4 + V - 23 = 2) : V = 21 :=
by 
  sorry

end video_files_initial_l213_213774


namespace doubling_n_constant_C_l213_213068

theorem doubling_n_constant_C (e n R r : ℝ) (h_pos_e : 0 < e) (h_pos_n : 0 < n) (h_pos_R : 0 < R) (h_pos_r : 0 < r)
  (C : ℝ) (hC : C = e^2 * n / (R + n * r^2)) :
  C = (2 * e^2 * n) / (R + 2 * n * r^2) := 
sorry

end doubling_n_constant_C_l213_213068


namespace four_digit_number_l213_213619

def digit_constraint (A B C D : ℕ) : Prop :=
  A = B / 3 ∧ C = A + B ∧ D = 3 * B

theorem four_digit_number 
  (A B C D : ℕ) 
  (h₁ : A = B / 3) 
  (h₂ : C = A + B) 
  (h₃ : D = 3 * B)
  (hA_digit : A < 10) 
  (hB_digit : B < 10)
  (hC_digit : C < 10)
  (hD_digit : D < 10) :
  1000 * A + 100 * B + 10 * C + D = 1349 := 
sorry

end four_digit_number_l213_213619


namespace arithmetic_sequence_value_l213_213940

theorem arithmetic_sequence_value :
  ∀ (a : ℕ → ℤ), 
  a 1 = 1 → 
  a 3 = -5 → 
  (a 1 - a 2 - a 3 - a 4 = 16) :=
by
  intros a h1 h3
  sorry

end arithmetic_sequence_value_l213_213940


namespace general_solution_linear_diophantine_l213_213240

theorem general_solution_linear_diophantine (a b c : ℤ) (h_coprime : Int.gcd a b = 1)
    (x1 y1 : ℤ) (h_particular_solution : a * x1 + b * y1 = c) :
    ∃ (t : ℤ), (∃ (x y : ℤ), x = x1 + b * t ∧ y = y1 - a * t ∧ a * x + b * y = c) ∧
               (∃ (x' y' : ℤ), x' = x1 - b * t ∧ y' = y1 + a * t ∧ a * x' + b * y' = c) :=
by
  sorry

end general_solution_linear_diophantine_l213_213240


namespace mass_of_substance_l213_213876

-- The conditions
def substance_density (mass_cubic_meter_kg : ℝ) (volume_cubic_meter_cm3 : ℝ) : Prop :=
  mass_cubic_meter_kg = 100 ∧ volume_cubic_meter_cm3 = 1*1000000

def specific_amount_volume_cm3 (volume_cm3 : ℝ) : Prop :=
  volume_cm3 = 10

-- The Proof Statement
theorem mass_of_substance (mass_cubic_meter_kg : ℝ) (volume_cubic_meter_cm3 : ℝ) (volume_cm3 : ℝ) (mass_grams : ℝ) :
  substance_density mass_cubic_meter_kg volume_cubic_meter_cm3 →
  specific_amount_volume_cm3 volume_cm3 →
  mass_grams = 10 :=
by
  intros hDensity hVolume
  sorry

end mass_of_substance_l213_213876


namespace fg_of_3_eq_83_l213_213551

def g (x : ℝ) : ℝ := x^3
def f (x : ℝ) : ℝ := 3 * x + 2

theorem fg_of_3_eq_83 : f (g 3) = 83 := by
  sorry

end fg_of_3_eq_83_l213_213551


namespace functional_equation_zero_l213_213791

open Function

theorem functional_equation_zero (f : ℕ+ → ℝ) 
  (h : ∀ (m n : ℕ+), n ≥ m → f (n + m) + f (n - m) = f (3 * n)) :
  ∀ n : ℕ+, f n = 0 := sorry

end functional_equation_zero_l213_213791


namespace infinite_squares_of_form_l213_213847

theorem infinite_squares_of_form (k : ℕ) (hk : k > 0) : ∃ᶠ n in at_top, ∃ m : ℕ, n * 2^k - 7 = m^2 := sorry

end infinite_squares_of_form_l213_213847


namespace origin_inside_circle_range_l213_213534

theorem origin_inside_circle_range (m : ℝ) :
  ((0 - m)^2 + (0 + m)^2 < 8) → (-2 < m ∧ m < 2) :=
by
  intros h
  sorry

end origin_inside_circle_range_l213_213534


namespace birthday_gift_l213_213931

-- Define the conditions
def friends : Nat := 8
def dollars_per_friend : Nat := 15

-- Formulate the statement to prove
theorem birthday_gift : friends * dollars_per_friend = 120 := by
  -- Proof is skipped using 'sorry'
  sorry

end birthday_gift_l213_213931


namespace J_of_given_values_l213_213933

def J (a b c : ℚ) : ℚ := a / b + b / c + c / a

theorem J_of_given_values : J 3 (-15) 10 = 49 / 30 := 
by 
  sorry

end J_of_given_values_l213_213933


namespace find_integer_for_combination_of_square_l213_213318

theorem find_integer_for_combination_of_square (y : ℝ) :
  ∃ (k : ℝ), (y^2 + 14*y + 60) = (y + 7)^2 + k ∧ k = 11 :=
by
  use 11
  sorry

end find_integer_for_combination_of_square_l213_213318


namespace ratio_smaller_triangle_to_trapezoid_area_l213_213052

theorem ratio_smaller_triangle_to_trapezoid_area (a b : ℕ) (sqrt_three : ℝ) 
  (h_a : a = 10) (h_b : b = 2) (h_sqrt_three : sqrt_three = Real.sqrt 3) :
  ( ( (sqrt_three / 4 * (b ^ 2)) / 
      ( (sqrt_three / 4 * (a ^ 2)) - 
         (sqrt_three / 4 * (b ^ 2)))) = 1 / 24 ) := 
by
  -- conditions from the problem
  have h1: a = 10 := by exact h_a
  have h2: b = 2 := by exact h_b
  have h3: sqrt_three = Real.sqrt 3 := by exact h_sqrt_three
  sorry

end ratio_smaller_triangle_to_trapezoid_area_l213_213052


namespace bike_owners_without_car_l213_213700

variable (T B C : ℕ) (H1 : T = 500) (H2 : B = 450) (H3 : C = 200)

theorem bike_owners_without_car (total bike_owners car_owners : ℕ) 
  (h_total : total = 500) (h_bike_owners : bike_owners = 450) (h_car_owners : car_owners = 200) : 
  (bike_owners - (bike_owners + car_owners - total)) = 300 := by
  sorry

end bike_owners_without_car_l213_213700


namespace ratio_of_squares_l213_213442

theorem ratio_of_squares (x y z : ℝ) (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) (h_sum_zero : x + 2 * y + 3 * z = 0) :
    (x^2 + y^2 + z^2) / (x * y + y * z + z * x) = -4 := by
  sorry

end ratio_of_squares_l213_213442


namespace factorize_expression_l213_213376

variable (a b : ℝ)

theorem factorize_expression : ab^2 - a = a * (b + 1) * (b - 1) :=
sorry

end factorize_expression_l213_213376


namespace youseff_lives_6_blocks_from_office_l213_213192

-- Definitions
def blocks_youseff_lives_from_office (x : ℕ) : Prop :=
  ∃ t_walk t_bike : ℕ,
    t_walk = x ∧
    t_bike = (20 * x) / 60 ∧
    t_walk = t_bike + 4

-- Main theorem
theorem youseff_lives_6_blocks_from_office (x : ℕ) (h : blocks_youseff_lives_from_office x) : x = 6 :=
  sorry

end youseff_lives_6_blocks_from_office_l213_213192


namespace if_2_3_4_then_1_if_1_3_4_then_2_l213_213628

variables {Plane Line : Type} 
variables (α β : Plane) (m n : Line)

-- assuming the perpendicular relationships as predicates
variable (perp : Plane → Plane → Prop) -- perpendicularity between planes
variable (perp' : Line → Line → Prop) -- perpendicularity between lines
variable (perp'' : Line → Plane → Prop) -- perpendicularity between line and plane

theorem if_2_3_4_then_1 :
  perp α β → perp'' m β → perp'' n α → perp' m n :=
by
  sorry

theorem if_1_3_4_then_2 :
  perp' m n → perp'' m β → perp'' n α → perp α β :=
by
  sorry

end if_2_3_4_then_1_if_1_3_4_then_2_l213_213628


namespace centroid_value_l213_213471

-- Define the points P, Q, R
def P : ℝ × ℝ := (4, 3)
def Q : ℝ × ℝ := (-1, 6)
def R : ℝ × ℝ := (7, -2)

-- Define the coordinates of the centroid S
noncomputable def S : ℝ × ℝ := 
  ( (4 + (-1) + 7) / 3, (3 + 6 + (-2)) / 3 )

-- Statement to prove
theorem centroid_value : 
  let x := (4 + (-1) + 7) / 3
  let y := (3 + 6 + (-2)) / 3
  8 * x + 3 * y = 101 / 3 :=
by
  let x := (4 + (-1) + 7) / 3
  let y := (3 + 6 + (-2)) / 3
  have h: 8 * x + 3 * y = 101 / 3 := sorry
  exact h

end centroid_value_l213_213471


namespace color_guard_team_row_length_l213_213600

theorem color_guard_team_row_length (n : ℕ) (p d : ℝ)
  (h_n : n = 40)
  (h_p : p = 0.4)
  (h_d : d = 0.5) :
  (n - 1) * d + n * p = 35.5 :=
by
  sorry

end color_guard_team_row_length_l213_213600


namespace cube_root_110592_l213_213391

theorem cube_root_110592 :
  (∃ x : ℕ, x^3 = 110592) ∧ 
  10^3 = 1000 ∧ 11^3 = 1331 ∧ 12^3 = 1728 ∧ 13^3 = 2197 ∧ 14^3 = 2744 ∧ 
  15^3 = 3375 ∧ 20^3 = 8000 ∧ 21^3 = 9261 ∧ 22^3 = 10648 ∧ 23^3 = 12167 ∧ 
  24^3 = 13824 ∧ 25^3 = 15625 → 48^3 = 110592 :=
by
  sorry

end cube_root_110592_l213_213391


namespace probability_all_operating_probability_shutdown_l213_213445

-- Define the events and their probabilities
def P_A : ℝ := 0.9
def P_B : ℝ := 0.8
def P_C : ℝ := 0.85

-- Prove that the probability of all three machines operating without supervision is 0.612
theorem probability_all_operating : P_A * P_B * P_C = 0.612 := 
by sorry

-- Prove that the probability of a shutdown is 0.059
theorem probability_shutdown :
    P_A * (1 - P_B) * (1 - P_C) +
    (1 - P_A) * P_B * (1 - P_C) +
    (1 - P_A) * (1 - P_B) * P_C +
    (1 - P_A) * (1 - P_B) * (1 - P_C) = 0.059 :=
by sorry

end probability_all_operating_probability_shutdown_l213_213445


namespace largest_five_digit_number_with_product_120_l213_213794

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def prod_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldr (· * ·) 1

def max_five_digit_prod_120 : ℕ := 85311

theorem largest_five_digit_number_with_product_120 :
  is_five_digit max_five_digit_prod_120 ∧ prod_of_digits max_five_digit_prod_120 = 120 :=
by
  sorry

end largest_five_digit_number_with_product_120_l213_213794


namespace min_num_edges_chromatic_l213_213136

-- Definition of chromatic number.
def chromatic_number (G : SimpleGraph V) : ℕ := sorry

-- Definition of the number of edges in a graph as a function.
def num_edges (G : SimpleGraph V) : ℕ := sorry

-- Statement of the theorem.
theorem min_num_edges_chromatic (G : SimpleGraph V) (n : ℕ) 
  (chrom_num_G : chromatic_number G = n) : 
  num_edges G ≥ n * (n - 1) / 2 :=
sorry

end min_num_edges_chromatic_l213_213136


namespace cannot_be_20182017_l213_213610

theorem cannot_be_20182017 (a b : ℤ) (h : a * b * (a + b) = 20182017) : False :=
by
  sorry

end cannot_be_20182017_l213_213610


namespace reciprocal_of_neg_two_l213_213462

theorem reciprocal_of_neg_two : ∀ x : ℝ, x = -2 → (1 / x) = -1 / 2 :=
by
  intro x h
  rw [h]
  norm_num

end reciprocal_of_neg_two_l213_213462


namespace largest_divisor_of_product_of_three_consecutive_odd_integers_l213_213293

theorem largest_divisor_of_product_of_three_consecutive_odd_integers :
  ∀ n : ℕ, n > 0 → ∃ d : ℕ, d = 3 ∧ ∀ m : ℕ, m ∣ ((2*n-1)*(2*n+1)*(2*n+3)) → m ≤ d :=
by
  sorry

end largest_divisor_of_product_of_three_consecutive_odd_integers_l213_213293


namespace cube_volume_is_8_l213_213726

theorem cube_volume_is_8 (a : ℕ) 
  (h_cond : (a+2) * (a-2) * a = a^3 - 8) : 
  a^3 = 8 := 
by
  sorry

end cube_volume_is_8_l213_213726


namespace value_of_expression_l213_213994

theorem value_of_expression {a b c : ℝ} (h_eqn : a + b + c = 15)
  (h_ab_bc_ca : ab + bc + ca = 13) (h_abc : abc = 8)
  (h_roots : Polynomial.roots (Polynomial.X^3 - 15 * Polynomial.X^2 + 13 * Polynomial.X - 8) = {a, b, c}) :
  (a / (1/a + b*c)) + (b / (1/b + c*a)) + (c / (1/c + a*b)) = 199/9 :=
by sorry

end value_of_expression_l213_213994


namespace chip_cost_l213_213452

theorem chip_cost 
  (calories_per_chip : ℕ)
  (chips_per_bag : ℕ)
  (cost_per_bag : ℕ)
  (desired_calories : ℕ)
  (h1 : calories_per_chip = 10)
  (h2 : chips_per_bag = 24)
  (h3 : cost_per_bag = 2)
  (h4 : desired_calories = 480) : 
  cost_per_bag * (desired_calories / (calories_per_chip * chips_per_bag)) = 4 := 
by 
  sorry

end chip_cost_l213_213452


namespace circle_radius_increase_l213_213737

variable (r n : ℝ) -- declare variables r and n as real numbers

theorem circle_radius_increase (h : 2 * π * (r + n) = 2 * (2 * π * r)) : r = n :=
by
  sorry

end circle_radius_increase_l213_213737


namespace relation_between_p_and_q_l213_213414

theorem relation_between_p_and_q (p q : ℝ) (α : ℝ) 
  (h1 : α + 2 * α = -p) 
  (h2 : α * (2 * α) = q) : 
  2 * p^2 = 9 * q := 
by 
  -- simplifying the provided conditions
  sorry

end relation_between_p_and_q_l213_213414


namespace range_of_a_l213_213952

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, (1 ≤ x ∧ x ≤ 2) ∧ (4 ≤ y ∧ y ≤ 5) → x * y ≤ a * x^2 + 2 * y^2) ↔ a ≥ -6 :=
by
  sorry

end range_of_a_l213_213952


namespace product_of_roots_eq_neg7_l213_213921

open Polynomial

theorem product_of_roots_eq_neg7 :
  let p := (2 : ℝ) * X^3 - (3 : ℝ) * X^2 - 10 * X + 14 in
  (p.roots.map ((id : ℝ → ℝ) ^ (-1)).prod = -7) :=
by
  sorry

end product_of_roots_eq_neg7_l213_213921


namespace initial_money_l213_213566

theorem initial_money (spent allowance total initial : ℕ) 
  (h1 : spent = 2) 
  (h2 : allowance = 26) 
  (h3 : total = 29) 
  (h4 : initial - spent + allowance = total) : 
  initial = 5 := 
by 
  sorry

end initial_money_l213_213566


namespace minimum_cost_of_candies_l213_213913

variable (Orange Apple Grape Strawberry : ℕ)

-- Conditions
def CandyRelation1 := Apple = 2 * Orange
def CandyRelation2 := Strawberry = 2 * Grape
def CandyRelation3 := Apple = 2 * Strawberry
def TotalCandies := Orange + Apple + Grape + Strawberry = 90
def CandyCost := 0.1

-- Question
theorem minimum_cost_of_candies :
  CandyRelation1 Orange Apple → 
  CandyRelation2 Grape Strawberry → 
  CandyRelation3 Apple Strawberry → 
  TotalCandies Orange Apple Grape Strawberry → 
  Orange ≥ 3 ∧ Apple ≥ 3 ∧ Grape ≥ 3 ∧ Strawberry ≥ 3 →
  (5 * CandyCost + 3 * CandyCost + 3 * CandyCost + 3 * CandyCost = 1.4) :=
sorry

end minimum_cost_of_candies_l213_213913


namespace sum_of_eight_numbers_l213_213976

theorem sum_of_eight_numbers (average : ℝ) (count : ℕ) (h_avg : average = 5.3) (h_count : count = 8) : (average * count = 42.4) := sorry

end sum_of_eight_numbers_l213_213976


namespace john_must_solve_at_least_17_correct_l213_213701

theorem john_must_solve_at_least_17_correct :
  ∀ (x : ℕ), 25 = 20 + 5 → 7 * x - (20 - x) + 2 * 5 ≥ 120 → x ≥ 17 :=
by
  intros x h1 h2
  -- Remaining steps will be included in the proof
  sorry

end john_must_solve_at_least_17_correct_l213_213701


namespace total_unique_plants_l213_213935

noncomputable def bed_A : ℕ := 600
noncomputable def bed_B : ℕ := 550
noncomputable def bed_C : ℕ := 400
noncomputable def bed_D : ℕ := 300

noncomputable def intersection_A_B : ℕ := 75
noncomputable def intersection_A_C : ℕ := 125
noncomputable def intersection_B_D : ℕ := 50
noncomputable def intersection_A_B_C : ℕ := 25

theorem total_unique_plants : 
  bed_A + bed_B + bed_C + bed_D - intersection_A_B - intersection_A_C - intersection_B_D + intersection_A_B_C = 1625 := 
by
  sorry

end total_unique_plants_l213_213935


namespace trig_identity_l213_213509

theorem trig_identity :
  ∀ (θ : ℝ),
    θ = 70 * (π / 180) →
    (1 / Real.cos θ - (Real.sqrt 3) / Real.sin θ) = Real.sec (20 * (π / 180)) ^ 2 :=
by
  sorry

end trig_identity_l213_213509


namespace journey_total_distance_l213_213546

theorem journey_total_distance :
  let speed1 := 40 -- in kmph
  let time1 := 3 -- in hours
  let speed2 := 60 -- in kmph
  let totalTime := 5 -- in hours
  let distance1 := speed1 * time1
  let time2 := totalTime - time1
  let distance2 := speed2 * time2
  let totalDistance := distance1 + distance2
  totalDistance = 240 := 
by
  sorry

end journey_total_distance_l213_213546


namespace initial_pencils_count_l213_213885

theorem initial_pencils_count (pencils_taken : ℕ) (pencils_left : ℕ) (h1 : pencils_taken = 4) (h2 : pencils_left = 75) : 
  pencils_left + pencils_taken = 79 :=
by
  sorry

end initial_pencils_count_l213_213885


namespace factory_days_worked_l213_213765

-- Define the number of refrigerators produced per hour
def refrigerators_per_hour : ℕ := 90

-- Define the number of coolers produced per hour
def coolers_per_hour : ℕ := refrigerators_per_hour + 70

-- Define the number of working hours per day
def working_hours_per_day : ℕ := 9

-- Define the total products produced per hour
def products_per_hour : ℕ := refrigerators_per_hour + coolers_per_hour

-- Define the total products produced in a day
def products_per_day : ℕ := products_per_hour * working_hours_per_day

-- Define the total number of products produced in given days
def total_products : ℕ := 11250

-- Define the number of days worked
def days_worked : ℕ := total_products / products_per_day

-- Prove that the number of days worked equals 5
theorem factory_days_worked : days_worked = 5 :=
by
  sorry

end factory_days_worked_l213_213765


namespace minimum_groups_l213_213352

theorem minimum_groups (total_players : ℕ) (max_per_group : ℕ)
  (h_total : total_players = 30)
  (h_max : max_per_group = 12) :
  ∃ x y, y ∣ total_players ∧ y ≤ max_per_group ∧ total_players / y = x ∧ x = 3 :=
by {
  sorry
}

end minimum_groups_l213_213352


namespace exchanges_count_l213_213033

theorem exchanges_count (n : ℕ) :
  ∀ (initial_pencils_XZ initial_pens_XL : ℕ) 
    (pencils_per_exchange pens_per_exchange : ℕ)
    (final_pencils_multiplier : ℕ)
    (pz : initial_pencils_XZ = 200) 
    (pl : initial_pens_XL = 20)
    (pe : pencils_per_exchange = 6)
    (se : pens_per_exchange = 1)
    (fm : final_pencils_multiplier = 11),
    (initial_pencils_XZ - pencils_per_exchange * n = final_pencils_multiplier * (initial_pens_XL - pens_per_exchange * n)) ↔ n = 4 :=
by
  intros initial_pencils_XZ initial_pens_XL pencils_per_exchange pens_per_exchange final_pencils_multiplier pz pl pe se fm
  sorry

end exchanges_count_l213_213033


namespace benny_kids_l213_213503

theorem benny_kids (total_money : ℕ) (cost_per_apple : ℕ) (apples_per_kid : ℕ) (total_apples : ℕ) (kids : ℕ) :
  total_money = 360 →
  cost_per_apple = 4 →
  apples_per_kid = 5 →
  total_apples = total_money / cost_per_apple →
  kids = total_apples / apples_per_kid →
  kids = 18 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end benny_kids_l213_213503


namespace factorization_of_x4_plus_16_l213_213923

theorem factorization_of_x4_plus_16 :
  (x : ℝ) → x^4 + 16 = (x^2 + 2 * x + 2) * (x^2 - 2 * x + 2) :=
by
  intro x
  -- Placeholder for the proof
  sorry

end factorization_of_x4_plus_16_l213_213923


namespace no_four_primes_exist_l213_213727

theorem no_four_primes_exist (a b c d : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b)
  (hc : Nat.Prime c) (hd : Nat.Prime d) (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (h4 : (1 / a : ℚ) + (1 / d) = (1 / b) + (1 / c)) : False := sorry

end no_four_primes_exist_l213_213727


namespace geo_seq_product_l213_213559

theorem geo_seq_product (a : ℕ → ℝ) (r : ℝ) (h_pos : ∀ n, 0 < a n) 
  (h_geo : ∀ n, a (n + 1) = a n * r) 
  (h_roots : a 1 ^ 2 - 10 * a 1 + 16 = 0) 
  (h_root19 : a 19 ^ 2 - 10 * a 19 + 16 = 0) : 
  a 8 * a 10 * a 12 = 64 :=
by
  sorry

end geo_seq_product_l213_213559


namespace prove_a2_b2_c2_zero_l213_213708

theorem prove_a2_b2_c2_zero (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) (h5 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) : a^2 + b^2 + c^2 = 0 := 
by 
  sorry

end prove_a2_b2_c2_zero_l213_213708


namespace Mrs_Amaro_roses_l213_213859

theorem Mrs_Amaro_roses :
  ∀ (total_roses red_roses yellow_roses pink_roses remaining_roses white_and_purple white_roses purple_roses : ℕ),
    total_roses = 500 →
    5 * total_roses % 8 = 0 →
    red_roses = total_roses * 5 / 8 →
    yellow_roses = (total_roses - red_roses) * 1 / 8 →
    pink_roses = (total_roses - red_roses) * 2 / 8 →
    remaining_roses = total_roses - red_roses - yellow_roses - pink_roses →
    remaining_roses % 2 = 0 →
    white_roses = remaining_roses / 2 →
    purple_roses = remaining_roses / 2 →
    red_roses + white_roses + purple_roses = 430 :=
by
  intros total_roses red_roses yellow_roses pink_roses remaining_roses white_and_purple white_roses purple_roses
  intro total_roses_eq
  intro red_roses_divisible
  intro red_roses_def
  intro yellow_roses_def
  intro pink_roses_def
  intro remaining_roses_def
  intro remaining_roses_even
  intro white_roses_def
  intro purple_roses_def
  sorry

end Mrs_Amaro_roses_l213_213859


namespace not_exists_odd_product_sum_l213_213609

theorem not_exists_odd_product_sum (a b : ℤ) : ¬ (a * b * (a + b) = 20182017) :=
sorry

end not_exists_odd_product_sum_l213_213609


namespace curve_is_line_l213_213793

theorem curve_is_line (r θ : ℝ) (h : r = 1 / (2 * real.sin θ - real.cos θ)) :
  ∃ a b c : ℝ, a ≠ 0 ∨ b ≠ 0 ∧ ∀ x y : ℝ, (∃ (r θ : ℝ), x = r * real.cos θ ∧ y = r * real.sin θ ∧ r = 1 / (2 * real.sin θ - real.cos θ)) → a * x + b * y + c = 0 :=
sorry

end curve_is_line_l213_213793


namespace hourly_wage_l213_213637

theorem hourly_wage (reps : ℕ) (hours_per_day : ℕ) (days : ℕ) (total_payment : ℕ) :
  reps = 50 →
  hours_per_day = 8 →
  days = 5 →
  total_payment = 28000 →
  (total_payment / (reps * hours_per_day * days) : ℕ) = 14 :=
by
  intros h_reps h_hours_per_day h_days h_total_payment
  -- Now the proof steps can be added here
  sorry

end hourly_wage_l213_213637


namespace find_P_coordinates_l213_213811

-- Given points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 2)

-- The area of triangle PAB is 5
def areaPAB (P : ℝ × ℝ) : ℝ :=
  0.5 * abs (P.1 * (A.2 - B.2) + A.1 * (B.2 - P.2) + B.1 * (P.2 - A.2))

-- Point P lies on the x-axis
def on_x_axis (P : ℝ × ℝ) : Prop := P.2 = 0

theorem find_P_coordinates (P : ℝ × ℝ) :
  on_x_axis P → areaPAB P = 5 → (P = (-4, 0) ∨ P = (6, 0)) :=
by
  sorry

end find_P_coordinates_l213_213811


namespace seq_integer_l213_213259

theorem seq_integer (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : a 2 = 1) (h3 : a 3 = 249)
(h_rec : ∀ n, a (n + 3) = (1991 + a (n + 2) * a (n + 1)) / a n) :
∀ n, ∃ b : ℤ, a n = b :=
by
  sorry

end seq_integer_l213_213259


namespace hiring_probabilities_l213_213615

-- Define the candidates and their abilities
inductive Candidate : Type
| Strong
| Moderate
| Weak

open Candidate

-- Define the ordering rule and hiring rule
def interviewOrders : List (Candidate × Candidate × Candidate) :=
  [(Strong, Moderate, Weak), (Strong, Weak, Moderate), 
   (Moderate, Strong, Weak), (Moderate, Weak, Strong),
   (Weak, Strong, Moderate), (Weak, Moderate, Strong)]

def hiresStrong (order : Candidate × Candidate × Candidate) : Bool :=
  match order with
  | (Moderate, Strong, Weak) => true
  | (Moderate, Weak, Strong) => true
  | (Weak, Strong, Moderate) => true
  | _ => false

def hiresModerate (order : Candidate × Candidate × Candidate) : Bool :=
  match order with
  | (Strong, Weak, Moderate) => true
  | (Weak, Moderate, Strong) => true
  | _ => false

-- The main theorem to be proved
theorem hiring_probabilities :
  let orders := interviewOrders
  let p := (orders.filter hiresStrong).length / orders.length
  let q := (orders.filter hiresModerate).length / orders.length
  p = 1 / 2 ∧ q = 1 / 3 := by
  sorry

end hiring_probabilities_l213_213615


namespace smallest_sum_of_five_consecutive_primes_divisible_by_five_l213_213386

theorem smallest_sum_of_five_consecutive_primes_divisible_by_five :
  ∃ (p1 p2 p3 p4 p5 : ℕ), (Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧ Nat.Prime p5) ∧
  ((p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧ p4 < p5 ∧ p5 ≤ p1 + 10)) ∧
  (p1 + p2 + p3 + p4 + p5 = 119) :=
by
  sorry

end smallest_sum_of_five_consecutive_primes_divisible_by_five_l213_213386


namespace complex_division_l213_213399

theorem complex_division (i : ℂ) (h_i : i * i = -1) : (3 - 4 * i) / i = 4 - 3 * i :=
by
  sorry

end complex_division_l213_213399


namespace find_number_l213_213466

theorem find_number (x : ℕ) (h : x + 1015 = 3016) : x = 2001 :=
sorry

end find_number_l213_213466


namespace sum_of_values_l213_213697

theorem sum_of_values (N : ℝ) (R : ℝ) (hN : N ≠ 0) (h_eq : N - 3 / N = R) :
  let N1 := (-R + Real.sqrt (R^2 + 12)) / 2
  let N2 := (-R - Real.sqrt (R^2 + 12)) / 2
  N1 + N2 = R :=
by
  sorry

end sum_of_values_l213_213697


namespace suraj_innings_count_l213_213597

theorem suraj_innings_count
  (A : ℕ := 24)  -- average before the last innings
  (new_average : ℕ := 28)  -- Suraj’s average after the last innings
  (last_score : ℕ := 92)  -- Suraj’s score in the last innings
  (avg_increase : ℕ := 4)  -- the increase in average after the last innings
  (n : ℕ)  -- number of innings before the last one
  (h_avg : A + avg_increase = new_average)  -- A + 4 = 28
  (h_eqn : n * A + last_score = (n + 1) * new_average) :  -- n * 24 + 92 = (n + 1) * 28
  n = 16 :=
by {
  sorry
}

end suraj_innings_count_l213_213597


namespace updated_mean_l213_213743

theorem updated_mean
  (n : ℕ) (obs_mean : ℝ) (decrement : ℝ)
  (h1 : n = 50) (h2 : obs_mean = 200) (h3 : decrement = 47) :
  (obs_mean - decrement) = 153 := by
  sorry

end updated_mean_l213_213743


namespace no_prime_divisible_by_56_l213_213122

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define what it means for a number to be divisible by another number
def divisible_by (a b : ℕ) : Prop :=
  b ≠ 0 ∧ ∃ k : ℕ, a = b * k

-- The main theorem stating the problem
theorem no_prime_divisible_by_56 : ¬ ∃ p : ℕ, is_prime p ∧ divisible_by p 56 :=
  sorry

end no_prime_divisible_by_56_l213_213122


namespace abs_diff_of_two_numbers_l213_213328

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x * y = 320) : |x - y| = 4 := 
by
  sorry

end abs_diff_of_two_numbers_l213_213328


namespace range_of_f3_l213_213951

def f (a c x : ℝ) : ℝ := a * x^2 - c

theorem range_of_f3 (a c : ℝ)
  (h1 : -4 ≤ f a c 1 ∧ f a c 1 ≤ -1)
  (h2 : -1 ≤ f a c 2 ∧ f a c 2 ≤ 5) :
  -1 ≤ f a c 3 ∧ f a c 3 ≤ 20 := 
sorry

end range_of_f3_l213_213951


namespace circle_radius_l213_213744

theorem circle_radius :
  ∃ c : ℝ × ℝ, 
    c.2 = 0 ∧
    (dist c (2, 3)) = (dist c (3, 7)) ∧
    (dist c (2, 3)) = (Real.sqrt 1717) / 2 :=
by
  sorry

end circle_radius_l213_213744


namespace last_three_digits_of_2_pow_15000_l213_213756

-- We need to define the given condition as a hypothesis and then state the goal.
theorem last_three_digits_of_2_pow_15000 :
  (2 ^ 500 ≡ 1 [MOD 1250]) → (2 ^ 15000 ≡ 1 [MOD 1000]) := by
  sorry

end last_three_digits_of_2_pow_15000_l213_213756


namespace fish_population_estimation_l213_213345

theorem fish_population_estimation (N : ℕ) (h1 : 80 ≤ N)
  (h_tagged_returned : true)
  (h_second_catch : 80 ≤ N)
  (h_tagged_in_second_catch : 2 = 80 * 80 / N) :
  N = 3200 :=
by
  sorry

end fish_population_estimation_l213_213345


namespace number_of_students_in_Diligence_before_transfer_l213_213276

-- Define the total number of students and the transfer information
def total_students : ℕ := 50
def transferred_students : ℕ := 2

-- Define the number of students in Diligence before the transfer
def students_in_Diligence_before : ℕ := 23

-- Let's prove that the number of students in Diligence before the transfer is 23
theorem number_of_students_in_Diligence_before_transfer :
  (total_students / 2) - transferred_students = students_in_Diligence_before :=
by {
  -- The proof is omitted as instructed
  sorry
}

end number_of_students_in_Diligence_before_transfer_l213_213276


namespace brenda_age_l213_213057

variables (A B J : ℝ)

-- Conditions
def condition1 : Prop := A = 4 * B
def condition2 : Prop := J = B + 7
def condition3 : Prop := A = J

-- Target to prove
theorem brenda_age (h1 : condition1 A B) (h2 : condition2 B J) (h3 : condition3 A J) : B = 7 / 3 :=
by
  sorry

end brenda_age_l213_213057


namespace find_value_of_a2_b2_c2_l213_213712

variable {a b c : ℝ}

theorem find_value_of_a2_b2_c2
  (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h2 : a + b + c = 0)
  (h3 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6 / 5 := 
sorry

end find_value_of_a2_b2_c2_l213_213712


namespace binomial_expansion_coeff_l213_213129

theorem binomial_expansion_coeff (n : ℕ) (h : 4 * Nat.choose n 2 = 14 * n) : n = 8 :=
by
  sorry

end binomial_expansion_coeff_l213_213129


namespace at_least_one_failure_probability_l213_213930

noncomputable def probability_at_least_one_failure : ℝ :=
  1 - (1 - 0.2)^5

theorem at_least_one_failure_probability :
  probability_at_least_one_failure ≈ 0.67232 :=
by sorry

end at_least_one_failure_probability_l213_213930


namespace find_square_l213_213679

-- Define the conditions as hypotheses
theorem find_square (p : ℕ) (sq : ℕ)
  (h1 : sq + p = 75)
  (h2 : (sq + p) + p = 142) :
  sq = 8 := by
  sorry

end find_square_l213_213679


namespace square_diagonal_length_l213_213189

theorem square_diagonal_length (rect_length rect_width : ℝ) 
  (h1 : rect_length = 45) 
  (h2 : rect_width = 40) 
  (rect_area := rect_length * rect_width) 
  (square_area := rect_area) 
  (side_length := Real.sqrt square_area) 
  (diagonal := side_length * Real.sqrt 2) :
  diagonal = 60 :=
by
  -- Proof goes here
  sorry

end square_diagonal_length_l213_213189


namespace otimes_2_5_l213_213435

def otimes (a b : ℝ) : ℝ := 4 * a + 3 * b

theorem otimes_2_5 : otimes 2 5 = 23 :=
by
  sorry

end otimes_2_5_l213_213435


namespace hawks_points_l213_213904

def touchdowns : ℕ := 3
def points_per_touchdown : ℕ := 7
def total_points (t : ℕ) (p : ℕ) : ℕ := t * p

theorem hawks_points : total_points touchdowns points_per_touchdown = 21 :=
by
  -- Proof will go here
  sorry

end hawks_points_l213_213904


namespace sum_of_numbers_l213_213963

theorem sum_of_numbers (a : ℝ) (n : ℕ) (h : a = 5.3) (hn : n = 8) : (a * n) = 42.4 :=
sorry

end sum_of_numbers_l213_213963


namespace integral_half_circle_area_l213_213604

open Real

-- Define the integrand function
def integrand (a : ℝ) (x : ℝ) : ℝ := sqrt (a ^ 2 - x ^ 2)

noncomputable def integral_of_half_circle (a : ℝ) : ℝ :=
  ∫ x in -a..a, integrand a x

-- The theorem we want to prove
theorem integral_half_circle_area (a : ℝ) (ha : 0 < a) :
  integral_of_half_circle a = (1 / 2) * π * a ^ 2 := by
  sorry

end integral_half_circle_area_l213_213604


namespace value_of_x_that_makes_sqrt_undefined_l213_213945

theorem value_of_x_that_makes_sqrt_undefined (x : ℕ) (hpos : 0 < x) : (x = 1) ∨ (x = 2) ↔ (x - 3 < 0) := by
  sorry

end value_of_x_that_makes_sqrt_undefined_l213_213945


namespace simplify_expression_l213_213594

-- Define the variables and the polynomials
variables (y : ℤ)

-- Define the expressions
def expr1 := (2 * y - 1) * (5 * y^12 - 3 * y^11 + y^9 - 4 * y^8)
def expr2 := 10 * y^13 - 11 * y^12 + 3 * y^11 + y^10 - 9 * y^9 + 4 * y^8

-- State the theorem
theorem simplify_expression : expr1 = expr2 := by
  sorry

end simplify_expression_l213_213594


namespace max_c_value_l213_213530

variable {a b c : ℝ}

theorem max_c_value (h1 : 2 * (a + b) = a * b) (h2 : a + b + c = a * b * c) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  c ≤ 8 / 15 :=
sorry

end max_c_value_l213_213530


namespace alex_correct_percentage_l213_213702

theorem alex_correct_percentage (y : ℝ) (hy_pos : y > 0) : 
  (5 / 7) * 100 = 71.43 := 
by
  sorry

end alex_correct_percentage_l213_213702


namespace sphere_surface_area_from_volume_l213_213614

theorem sphere_surface_area_from_volume 
  (V : ℝ) (h : V = 72 * Real.pi) :
  ∃ (A : ℝ), A = 36 * Real.pi * 2^(2/3) :=
by
  sorry

end sphere_surface_area_from_volume_l213_213614


namespace quadratic_rewrite_l213_213786

theorem quadratic_rewrite  (a b c x : ℤ) (h : 25 * x^2 + 30 * x - 35 = 0) (hp : 25 * x^2 + 30 * x + 9 = (5 * x + 3) ^ 2)
(hc : c = 44) : a = 5 → b = 3 → a + b + c = 52 := 
by
  intro ha hb
  sorry

end quadratic_rewrite_l213_213786


namespace find_f_of_odd_function_periodic_l213_213093

noncomputable def arctan (x : ℝ) : ℝ := sorry

theorem find_f_of_odd_function_periodic (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_periodic : ∀ x k : ℤ, f x = f (x + 3 * k))
    (α : ℝ) (h_tan : Real.tan α = 3) :
  f (2015 * Real.sin (2 * (arctan 3))) = 0 :=
sorry

end find_f_of_odd_function_periodic_l213_213093


namespace section_Diligence_students_before_transfer_l213_213277

-- Define the variables
variables (D_after I_after D_before : ℕ)

-- Problem Statement
theorem section_Diligence_students_before_transfer :
  ∀ (D_after I_after: ℕ),
    2 + D_after = I_after
    ∧ D_after + I_after = 50 →
    ∃ D_before, D_before = D_after - 2 ∧ D_before = 23 :=
by
sorrry

end section_Diligence_students_before_transfer_l213_213277


namespace ellen_smoothie_ingredients_l213_213235

theorem ellen_smoothie_ingredients :
  let strawberries := 0.2
  let yogurt := 0.1
  let orange_juice := 0.2
  strawberries + yogurt + orange_juice = 0.5 :=
by
  sorry

end ellen_smoothie_ingredients_l213_213235


namespace rectangle_perimeter_l213_213768

theorem rectangle_perimeter {w l : ℝ} 
  (h_area : l * w = 450)
  (h_length : l = 2 * w) :
  2 * (l + w) = 90 :=
by sorry

end rectangle_perimeter_l213_213768


namespace center_of_hyperbola_l213_213522

-- Define the given equation of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop :=
  ((3 * y - 6)^2 / 8^2) - ((4 * x - 5)^2 / 3^2) = 1

-- Prove that the center of the hyperbola is (5 / 4, 2)
theorem center_of_hyperbola :
  (∃ h k : ℝ, h = 5 / 4 ∧ k = 2 ∧ ∀ x y : ℝ, hyperbola_eq x y ↔ ((y - k)^2 / (8 / 3)^2 - (x - h)^2 / (3 / 4)^2 = 1)) :=
sorry

end center_of_hyperbola_l213_213522


namespace find_k_intersection_l213_213854

theorem find_k_intersection :
  ∃ (k : ℝ), 
  (∀ (x y : ℝ), y = 2 * x + 3 → y = k * x + 1 → (x = 1 ∧ y = 5) → k = 4) :=
sorry

end find_k_intersection_l213_213854


namespace Reeya_fifth_subject_score_l213_213729

theorem Reeya_fifth_subject_score 
  (a1 a2 a3 a4 : ℕ) (avg : ℕ) (subjects : ℕ) (a1_eq : a1 = 55) (a2_eq : a2 = 67) (a3_eq : a3 = 76) 
  (a4_eq : a4 = 82) (avg_eq : avg = 73) (subjects_eq : subjects = 5) :
  ∃ a5 : ℕ, (a1 + a2 + a3 + a4 + a5) / subjects = avg ∧ a5 = 85 :=
by
  sorry

end Reeya_fifth_subject_score_l213_213729


namespace factorize_expression_l213_213377

variable (a b : ℝ)

theorem factorize_expression : ab^2 - a = a * (b + 1) * (b - 1) :=
sorry

end factorize_expression_l213_213377


namespace solution_of_equation_l213_213383

noncomputable def integer_part (x : ℝ) : ℤ := Int.floor x
noncomputable def fractional_part (x : ℝ) : ℝ := x - integer_part x

theorem solution_of_equation (k : ℤ) (h : -1 ≤ k ∧ k ≤ 5) :
  ∃ x : ℝ, 4 * ↑(integer_part x) = 25 * fractional_part x - 4.5 ∧
           x = k + (8 * ↑k + 9) / 50 := 
sorry

end solution_of_equation_l213_213383


namespace perpendicular_lines_slope_l213_213543

theorem perpendicular_lines_slope (m : ℝ) : 
  ((m ≠ -3) ∧ (m ≠ -5) ∧ 
  (- (m + 3) / 4 * - (2 / (m + 5)) = -1)) ↔ m = -13 / 3 := 
sorry

end perpendicular_lines_slope_l213_213543


namespace factorize_expression_l213_213381

-- Lean 4 statement for the proof problem
theorem factorize_expression (a b : ℝ) : ab^2 - a = a * (b + 1) * (b - 1) :=
sorry

end factorize_expression_l213_213381


namespace cosine_of_angle_between_vectors_l213_213264

-- Define the vectors a and b
def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (5, 12)

-- Function to calculate the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Function to calculate the norm of a vector
def norm (u : ℝ × ℝ) : ℝ :=
  Real.sqrt (u.1^2 + u.2^2)

-- The desired statement to prove
theorem cosine_of_angle_between_vectors :
  (dot_product a b) / (norm a * norm b) = 63 / 65 :=
by
  sorry

end cosine_of_angle_between_vectors_l213_213264


namespace jeffreys_total_steps_l213_213337

-- Define the conditions
def effective_steps_per_pattern : ℕ := 1
def total_effective_distance : ℕ := 66
def steps_per_pattern : ℕ := 5

-- Define the proof problem
theorem jeffreys_total_steps : ∀ (N : ℕ), 
  N = (total_effective_distance * steps_per_pattern) := 
sorry

end jeffreys_total_steps_l213_213337


namespace sum_of_squares_l213_213960

theorem sum_of_squares (a b c d e f : ℤ)
  (h : ∀ x : ℤ, 512 * x ^ 3 + 125 = (a * x ^ 2 + b * x + c) * (d * x ^ 2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 6410 := 
sorry

end sum_of_squares_l213_213960


namespace find_r_l213_213382

-- Lean statement
theorem find_r (r : ℚ) (log_eq : Real.logb 81 (2 * r - 1) = -1 / 2) : r = 5 / 9 :=
by {
    sorry -- proof steps should not be included according to the requirements
}

end find_r_l213_213382


namespace intersection_M_N_l213_213481

def M : Set ℝ := { x | x > 1 }
def N : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }

theorem intersection_M_N :
  M ∩ N = { x | 1 < x ∧ x ≤ 2 } := 
sorry

end intersection_M_N_l213_213481


namespace min_value_exprB_four_min_value_exprC_four_l213_213025

noncomputable def exprB (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def exprC (x : ℝ) : ℝ := 1 / (Real.sin x)^2 + 1 / (Real.cos x)^2

theorem min_value_exprB_four : ∃ x : ℝ, exprB x = 4 := sorry

theorem min_value_exprC_four : ∃ x : ℝ, exprC x = 4 := sorry

end min_value_exprB_four_min_value_exprC_four_l213_213025


namespace brendas_age_l213_213060

theorem brendas_age (A B J : ℝ) 
  (h1 : A = 4 * B)
  (h2 : J = B + 7)
  (h3 : A = J) : 
  B = 7 / 3 := 
by 
  sorry

end brendas_age_l213_213060


namespace average_price_of_goat_l213_213631

theorem average_price_of_goat (total_cost_goats_hens : ℕ) (num_goats num_hens : ℕ) (avg_price_hen : ℕ)
  (h1 : total_cost_goats_hens = 2500) (h2 : num_hens = 10) (h3 : avg_price_hen = 50) (h4 : num_goats = 5) :
  (total_cost_goats_hens - num_hens * avg_price_hen) / num_goats = 400 :=
sorry

end average_price_of_goat_l213_213631


namespace find_charge_federal_return_l213_213735

-- Definitions based on conditions
def charge_federal_return (F : ℝ) : ℝ := F
def charge_state_return : ℝ := 30
def charge_quarterly_return : ℝ := 80
def sold_federal_returns : ℝ := 60
def sold_state_returns : ℝ := 20
def sold_quarterly_returns : ℝ := 10
def total_revenue : ℝ := 4400

-- Lean proof statement to verify the value of F
theorem find_charge_federal_return (F : ℝ) (h : sold_federal_returns * charge_federal_return F + sold_state_returns * charge_state_return + sold_quarterly_returns * charge_quarterly_return = total_revenue) : 
  F = 50 :=
by
  sorry

end find_charge_federal_return_l213_213735


namespace min_value_abc_l213_213852

open Real

theorem min_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : 1/a + 1/b + 1/c = 9) :
  a^4 * b^3 * c^2 ≥ 1/10368 :=
sorry

end min_value_abc_l213_213852


namespace brendas_age_l213_213058

theorem brendas_age (A B J : ℝ) 
  (h1 : A = 4 * B)
  (h2 : J = B + 7)
  (h3 : A = J) : 
  B = 7 / 3 := 
by 
  sorry

end brendas_age_l213_213058


namespace compute_trig_expr_l213_213515

theorem compute_trig_expr :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 2 * Real.sec (10 * Real.pi / 180) :=
by
  sorry

end compute_trig_expr_l213_213515


namespace number_of_exchanges_l213_213032

theorem number_of_exchanges (n : ℕ) (hz_initial : ℕ) (hl_initial : ℕ) 
  (hz_decrease : ℕ) (hl_decrease : ℕ) (k : ℕ) :
  hz_initial = 200 →
  hl_initial = 20 →
  hz_decrease = 6 →
  hl_decrease = 1 →
  k = 11 →
  (hz_initial - n * hz_decrease) = k * (hl_initial - n * hl_decrease) →
  n = 4 := 
sorry

end number_of_exchanges_l213_213032


namespace jerry_current_average_l213_213430

-- Definitions for Jerry's first 3 tests average and conditions
variable (A : ℝ)

-- Condition details
def total_score_of_first_3_tests := 3 * A
def new_desired_average := A + 2
def total_score_needed := (A + 2) * 4
def score_on_fourth_test := 93

theorem jerry_current_average :
  (total_score_needed A = total_score_of_first_3_tests A + score_on_fourth_test) → A = 85 :=
by
  sorry

end jerry_current_average_l213_213430


namespace original_number_is_9_l213_213767

theorem original_number_is_9 (x : ℤ) (h : 10 * x = x + 81) : x = 9 :=
sorry

end original_number_is_9_l213_213767


namespace table_covered_with_three_layers_l213_213013

theorem table_covered_with_three_layers (A T table_area two_layers : ℕ)
    (hA : A = 204)
    (htable : table_area = 175)
    (hcover : 140 = 80 * table_area / 100)
    (htwo_layers : two_layers = 24) :
    3 * T + 2 * two_layers + (140 - two_layers - T) = 204 → T = 20 := by
  sorry

end table_covered_with_three_layers_l213_213013


namespace cos_pi_over_4_minus_alpha_l213_213804

theorem cos_pi_over_4_minus_alpha (α : ℝ) (h : Real.sin (Real.pi / 4 + α) = 2 / 3) :
  Real.cos (Real.pi / 4 - α) = 2 / 3 := 
by
  sorry

end cos_pi_over_4_minus_alpha_l213_213804


namespace solution_set_for_a1_find_a_if_min_value_is_4_l213_213403

noncomputable def f (a x : ℝ) : ℝ := |2 * x - 1| + |a * x - 5|

theorem solution_set_for_a1 : 
  { x : ℝ | f 1 x ≥ 9 } = { x : ℝ | x ≤ -1 ∨ x > 5 } :=
sorry

theorem find_a_if_min_value_is_4 :
  ∃ a : ℝ, (0 < a ∧ a < 5) ∧ (∀ x : ℝ, f a x ≥ 4) ∧ (∃ x : ℝ, f a x = 4) ∧ a = 2 :=
sorry

end solution_set_for_a1_find_a_if_min_value_is_4_l213_213403


namespace marina_max_socks_l213_213331

theorem marina_max_socks (white black : ℕ) (hw : white = 8) (hb : black = 15) :
  ∃ n, n = 17 ∧ ∀ w b, w + b = n → 0 ≤ w ∧ 0 ≤ b ∧ w ≤ black ∧ b ≤ black ∧ w ≤ white ∧ b ≤ black → b > w :=
sorry

end marina_max_socks_l213_213331


namespace max_value_f_on_interval_l213_213394

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  (x^2 - 4) * (x - a)

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ :=
  3 * x^2 - 2 * a * x - 4

theorem max_value_f_on_interval :
  f' (-1) (1 / 2) = 0 →
  ∃ max_f, max_f = 42 ∧ ∀ x ∈ Set.Icc (-2 : ℝ) 4, f x (1 / 2) ≤ max_f :=
by
  sorry

end max_value_f_on_interval_l213_213394


namespace solution_proof_l213_213763

noncomputable def problem_statement : Prop :=
  let a : ℝ := 0.10
  let b : ℝ := 0.50
  let c : ℝ := 500
  a * (b * c) = 25

theorem solution_proof : problem_statement := by
  sorry

end solution_proof_l213_213763


namespace sqrt_product_simplified_l213_213781

theorem sqrt_product_simplified (q : ℝ) (hq : 0 < q) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q) = 21 * q * Real.sqrt (2 * q) :=
by
  sorry

end sqrt_product_simplified_l213_213781


namespace even_odd_function_value_l213_213535

theorem even_odd_function_value 
  (f g : ℝ → ℝ) 
  (h_even : ∀ x, f (-x) = f x)
  (h_odd : ∀ x, g (-x) = - g x)
  (h_eqn : ∀ x, f x + g x = x^3 + x^2 + 1) :
  f 1 - g 1 = 1 := 
by {
  sorry
}

end even_odd_function_value_l213_213535


namespace cost_of_patent_is_correct_l213_213232

-- Defining the conditions
def c_parts : ℕ := 3600
def p : ℕ := 180
def n : ℕ := 45

-- Calculation of total revenue
def total_revenue : ℕ := n * p

-- Calculation of cost of patent
def cost_of_patent (total_revenue c_parts : ℕ) : ℕ := total_revenue - c_parts

-- The theorem to be proved
theorem cost_of_patent_is_correct (R : ℕ) (H : R = total_revenue) : cost_of_patent R c_parts = 4500 :=
by
  -- this is where your proof will go
  sorry

end cost_of_patent_is_correct_l213_213232


namespace YooSeung_has_108_marbles_l213_213035

def YoungSoo_marble_count : ℕ := 21
def HanSol_marble_count : ℕ := YoungSoo_marble_count + 15
def YooSeung_marble_count : ℕ := 3 * HanSol_marble_count
def total_marble_count : ℕ := YoungSoo_marble_count + HanSol_marble_count + YooSeung_marble_count

theorem YooSeung_has_108_marbles 
  (h1 : YooSeung_marble_count = 3 * (YoungSoo_marble_count + 15))
  (h2 : HanSol_marble_count = YoungSoo_marble_count + 15)
  (h3 : total_marble_count = 165) :
  YooSeung_marble_count = 108 :=
by sorry

end YooSeung_has_108_marbles_l213_213035


namespace total_surface_area_space_l213_213496

theorem total_surface_area_space (h r1 : ℝ) (h_cond : h = 8) (r1_cond : r1 = 3) : 
  (2 * π * (r1 + 1) * h - 2 * π * r1 * h) = 16 * π := 
by
  sorry

end total_surface_area_space_l213_213496


namespace find_a9_l213_213548

noncomputable def polynomial_coefficients : Prop :=
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℤ),
  ∀ (x : ℤ),
    x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 + x^8 + x^9 + x^10 =
    a₀ + a₁ * (1 + x) + a₂ * (1 + x)^2 + a₃ * (1 + x)^3 + a₄ * (1 + x)^4 + 
    a₅ * (1 + x)^5 + a₆ * (1 + x)^6 + a₇ * (1 + x)^7 + a₈ * (1 + x)^8 + 
    a₉ * (1 + x)^9 + a₁₀ * (1 + x)^10

theorem find_a9 (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℤ) (h : polynomial_coefficients) : a₉ = -9 := by
  sorry

end find_a9_l213_213548


namespace intersecting_circles_l213_213401

theorem intersecting_circles (m c : ℝ)
  (h1 : ∃ (x1 y1 x2 y2 : ℝ), x1 = 1 ∧ y1 = 3 ∧ x2 = m ∧ y2 = 1 ∧ x1 ≠ x2 ∧ y1 ≠ y2)
  (h2 : ∀ (x y : ℝ), (x - y + (c / 2) = 0) → (x = 1 ∨ y = 3)) :
  m + c = 3 :=
sorry

end intersecting_circles_l213_213401


namespace A_equals_4_of_rounded_to_tens_9430_l213_213023

variable (A B : ℕ)

theorem A_equals_4_of_rounded_to_tens_9430
  (h1 : 9430 = 9000 + 100 * A + 10 * 3 + B)
  (h2 : B < 5)
  (h3 : 0 ≤ A ∧ A ≤ 9)
  (h4 : 0 ≤ B ∧ B ≤ 9) :
  A = 4 :=
by
  sorry

end A_equals_4_of_rounded_to_tens_9430_l213_213023


namespace total_seats_in_theater_l213_213772

def theater_charges_adults : ℝ := 3.0
def theater_charges_children : ℝ := 1.5
def total_income : ℝ := 510
def number_of_children : ℕ := 60

theorem total_seats_in_theater :
  ∃ (A C : ℕ), C = number_of_children ∧ theater_charges_adults * A + theater_charges_children * C = total_income ∧ A + C = 200 :=
by
  sorry

end total_seats_in_theater_l213_213772


namespace original_agreed_amount_l213_213492

theorem original_agreed_amount (months: ℕ) (cash: ℚ) (uniform_price: ℚ) (received_total: ℚ) (full_year: ℚ) :
  months = 9 →
  cash = 300 →
  uniform_price = 300 →
  received_total = 600 →
  full_year = (12: ℚ) →
  ((months / full_year) * (cash + uniform_price) = received_total) →
  cash + uniform_price = 800 := 
by
  intros h_months h_cash h_uniform h_received h_year h_proportion
  sorry

end original_agreed_amount_l213_213492


namespace point_in_second_quadrant_l213_213137

/-- Define the quadrants in the Cartesian coordinate system -/
def quadrant (x y : ℤ) : String :=
  if x > 0 ∧ y > 0 then "First quadrant"
  else if x < 0 ∧ y > 0 then "Second quadrant"
  else if x < 0 ∧ y < 0 then "Third quadrant"
  else if x > 0 ∧ y < 0 then "Fourth quadrant"
  else "On the axis"

theorem point_in_second_quadrant :
  quadrant (-3) 2005 = "Second quadrant" :=
by
  sorry

end point_in_second_quadrant_l213_213137


namespace polynomial_has_real_root_l213_213557

theorem polynomial_has_real_root
  (a b c d e : ℝ)
  (h : ∃ r : ℝ, ax^2 + (c - b)x + (e - d) = 0 ∧ r > 1) :
  ∃ x : ℝ, ax^4 + bx^3 + cx^2 + dx + e = 0 :=
by
  sorry

end polynomial_has_real_root_l213_213557


namespace degrees_to_radians_l213_213231

theorem degrees_to_radians (π_radians : ℝ) : 150 * π_radians / 180 = 5 * π_radians / 6 :=
by sorry

end degrees_to_radians_l213_213231


namespace find_a_b_l213_213146

theorem find_a_b (a b : ℕ) (h1 : (a^3 - a^2 + 1) * (b^3 - b^2 + 2) = 2020) : 10 * a + b = 53 :=
by {
  -- Proof to be completed
  sorry
}

end find_a_b_l213_213146


namespace find_sum_lent_l213_213476

variable (P : ℝ)

/-- Given that the annual interest rate is 4%, and the interest earned in 8 years
amounts to Rs 340 less than the sum lent, prove that the sum lent is Rs 500. -/
theorem find_sum_lent
  (h1 : ∀ I, I = P - 340 → I = (P * 4 * 8) / 100) : 
  P = 500 :=
by
  sorry

end find_sum_lent_l213_213476


namespace intersection_A_B_l213_213398

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x < 2}
def B : Set ℝ := {-3, -2, -1, 0, 1, 2}

-- Define the intersection we need to prove
def A_cap_B_target : Set ℝ := {-2, -1, 0, 1}

-- Prove the intersection of A and B equals the target set
theorem intersection_A_B :
  A ∩ B = A_cap_B_target := 
sorry

end intersection_A_B_l213_213398


namespace calculate_result_l213_213783

theorem calculate_result (x : ℝ) : (-x^3)^3 = -x^9 :=
by {
  sorry  -- Proof not required per instructions
}

end calculate_result_l213_213783


namespace distinct_ways_to_distribute_l213_213265

theorem distinct_ways_to_distribute :
  ∃ (n : ℕ), n = 7 ∧ ∀ (balls : ℕ) (boxes : ℕ)
    (indistinguishable_balls : Prop := true) 
    (indistinguishable_boxes : Prop := true), 
    balls = 6 → boxes = 3 → 
    indistinguishable_balls → 
    indistinguishable_boxes → 
    n = 7 :=
by
  sorry

end distinct_ways_to_distribute_l213_213265


namespace num_distinct_five_digit_integers_with_product_of_digits_18_l213_213233

theorem num_distinct_five_digit_integers_with_product_of_digits_18 :
  ∃ (n : ℕ), n = 70 ∧ ∀ (a b c d e : ℕ),
    a * b * c * d * e = 18 ∧ 
    1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9 ∧ 1 ≤ e ∧ e ≤ 9 → 
    (∃ (s : Finset (Fin 100000)), s.card = n) :=
  sorry

end num_distinct_five_digit_integers_with_product_of_digits_18_l213_213233


namespace painting_colors_area_l213_213914

theorem painting_colors_area
  (B G Y : ℕ)
  (h_total_blue : B + (1 / 3 : ℝ) * G = 38)
  (h_total_yellow : Y + (2 / 3 : ℝ) * G = 38)
  (h_grass_sky_relation : G = B + 6) :
  B = 27 ∧ G = 33 ∧ Y = 16 :=
by
  sorry

end painting_colors_area_l213_213914


namespace sqrt_nested_expr_l213_213234

theorem sqrt_nested_expr (x : ℝ) (hx : 0 ≤ x) : 
  (x * (x * (x * x)^(1 / 2))^(1 / 2))^(1 / 2) = (x^7)^(1 / 4) :=
sorry

end sqrt_nested_expr_l213_213234


namespace length_of_plot_l213_213196

theorem length_of_plot (breadth length : ℕ) 
                       (h1 : length = breadth + 26)
                       (fencing_cost total_cost : ℝ)
                       (h2 : fencing_cost = 26.50)
                       (h3 : total_cost = 5300)
                       (perimeter : ℝ) 
                       (h4 : perimeter = 2 * (breadth + length)) 
                       (h5 : total_cost = perimeter * fencing_cost) :
                       length = 63 :=
by
  sorry

end length_of_plot_l213_213196


namespace least_n_value_l213_213531

open Nat

theorem least_n_value (n : ℕ) (h : 1 / (n * (n + 1)) < 1 / 15) : n = 4 :=
sorry

end least_n_value_l213_213531


namespace both_questions_correct_l213_213961

-- Define variables as constants
def nA : ℝ := 0.85  -- 85%
def nB : ℝ := 0.70  -- 70%
def nAB : ℝ := 0.60 -- 60%

theorem both_questions_correct:
  nAB = 0.60 := by
  sorry

end both_questions_correct_l213_213961


namespace length_of_second_square_l213_213225

-- Define conditions as variables
def Area_flag := 135
def Area_square1 := 40
def Area_square3 := 25

-- Define the length variable for the second square
variable (L : ℕ)

-- Define the area of the second square in terms of L
def Area_square2 : ℕ := 7 * L

-- Lean statement to be proved
theorem length_of_second_square :
  Area_square1 + Area_square2 L + Area_square3 = Area_flag → L = 10 :=
by sorry

end length_of_second_square_l213_213225


namespace largest_five_digit_product_120_l213_213796

theorem largest_five_digit_product_120 : 
  ∃ n : ℕ, n = 85311 ∧ (nat.digits 10 n).product = 120 ∧ 10000 ≤ n ∧ n < 100000 :=
by
  sorry

end largest_five_digit_product_120_l213_213796


namespace pull_ups_of_fourth_student_l213_213210

theorem pull_ups_of_fourth_student 
  (avg_pullups : ℕ) 
  (num_students : ℕ) 
  (pullups_first : ℕ) 
  (pullups_second : ℕ) 
  (pullups_third : ℕ) 
  (pullups_fifth : ℕ) 
  (H_avg : avg_pullups = 10) 
  (H_students : num_students = 5) 
  (H_first : pullups_first = 9) 
  (H_second : pullups_second = 12) 
  (H_third : pullups_third = 9) 
  (H_fifth : pullups_fifth = 8) : 
  ∃ (pullups_fourth : ℕ), pullups_fourth = 12 := by
  sorry

end pull_ups_of_fourth_student_l213_213210


namespace no_sum_of_cubes_eq_2002_l213_213282

theorem no_sum_of_cubes_eq_2002 :
  ¬ ∃ (a b c : ℕ), (a ^ 3 + b ^ 3 + c ^ 3 = 2002) :=
sorry

end no_sum_of_cubes_eq_2002_l213_213282


namespace white_surface_area_fraction_l213_213633

theorem white_surface_area_fraction :
  let larger_cube_edge := 4
  let smaller_cube_edge := 1
  let total_smaller_cubes := 64
  let white_cubes := 48
  let black_cubes := 16
  let total_faces := 6
  let black_cubes_per_face := 2
  let surface_area := total_faces * larger_cube_edge^2
  let black_faces_exposed := total_faces * black_cubes_per_face
  let white_faces_exposed := surface_area - black_faces_exposed
  (white_faces_exposed / surface_area) = (7 / 8) :=
by
  let larger_cube_edge := 4
  let smaller_cube_edge := 1
  let total_smaller_cubes := 64
  let white_cubes := 48
  let black_cubes := 16
  let total_faces := 6
  let black_cubes_per_face := 2
  let surface_area := total_faces * larger_cube_edge^2
  let black_faces_exposed := total_faces * black_cubes_per_face
  let white_faces_exposed := surface_area - black_faces_exposed
  have h_white_fraction : (white_faces_exposed / surface_area) = (7 / 8) := sorry
  exact h_white_fraction

end white_surface_area_fraction_l213_213633


namespace math_equivalence_l213_213267

theorem math_equivalence (m n : ℤ) (h : |m - 2023| + (n + 2024)^2 = 0) : (m + n) ^ 2023 = -1 := 
by
  sorry

end math_equivalence_l213_213267


namespace largest_positive_integer_n_exists_l213_213787

theorem largest_positive_integer_n_exists (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ n, 
    0 < n ∧ 
    (n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 9) ∧ 
    ∀ m, 0 < m → 
      (m^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 9) → 
      m ≤ n :=
  sorry

end largest_positive_integer_n_exists_l213_213787


namespace series_sum_l213_213648

theorem series_sum :
  ∑' n : ℕ, (2 ^ n) / (3 ^ (2 ^ n) + 1) = 1 / 2 :=
sorry

end series_sum_l213_213648


namespace arithmetic_seq_sum_l213_213134

theorem arithmetic_seq_sum (a : ℕ → ℤ) (a1 a7 a3 a5 : ℤ) (S7 : ℤ)
  (h1 : a1 = a 1) (h7 : a7 = a 7) (h3 : a3 = a 3) (h5 : a5 = a 5)
  (h_arith : ∀ n m, a (n + m) = a n + a m - a 0)
  (h_S7 : (7 * (a1 + a7)) / 2 = 14) :
  a3 + a5 = 4 :=
sorry

end arithmetic_seq_sum_l213_213134


namespace genuine_coins_probability_l213_213200

-- Statement of the problem
theorem genuine_coins_probability:
  let total_coins := 12
  let genuine_coins := 9
  let counterfeit_coins := 3
  let first_pair_genuine := (genuine_coins / total_coins) * ((genuine_coins - 1) / (total_coins - 1))
  let second_pair_genuine := ((genuine_coins - 2) / (total_coins - 2)) * ((genuine_coins - 3) / (total_coins - 3))
  let prob_all_genuine := first_pair_genuine * second_pair_genuine
  (prob_all_genuine / (prob_all_genuine + _)) = 14 / 55 := sorry

end genuine_coins_probability_l213_213200


namespace total_oranges_in_stack_l213_213206

-- Definitions based on the given conditions
def base_layer_oranges : Nat := 5 * 8
def second_layer_oranges : Nat := 4 * 7
def third_layer_oranges : Nat := 3 * 6
def fourth_layer_oranges : Nat := 2 * 5
def fifth_layer_oranges : Nat := 1 * 4

-- Theorem statement equivalent to the math problem
theorem total_oranges_in_stack : base_layer_oranges + second_layer_oranges + third_layer_oranges + fourth_layer_oranges + fifth_layer_oranges = 100 :=
by
  sorry

end total_oranges_in_stack_l213_213206


namespace hyperbola_asymptotes_l213_213542

noncomputable def eccentricity_asymptotes (a b : ℝ) (h₁ : a > 0) (h₂ : b = Real.sqrt 15 * a) : Prop :=
  ∀ (x y : ℝ), (y = (Real.sqrt 15) * x) ∨ (y = -(Real.sqrt 15) * x)

theorem hyperbola_asymptotes (a : ℝ) (h₁ : a > 0) :
  eccentricity_asymptotes a (Real.sqrt 15 * a) h₁ (by simp) :=
sorry

end hyperbola_asymptotes_l213_213542


namespace problem_B_problem_D_l213_213938

noncomputable def z : ℂ := (2 * complex.I) / (real.sqrt 3 + complex.I)
noncomputable def z_conjugate : ℂ := conj z

theorem problem_B :
  complex.abs z_conjugate = 1 :=
sorry

theorem problem_D : 
  z_conjugate.re > 0 ∧ z_conjugate.im < 0 :=
sorry

end problem_B_problem_D_l213_213938


namespace problem1_problem2_l213_213953

variable (t : ℝ)

-- Problem 1
theorem problem1 (h : (4:ℝ) - 8 * t + 16 < 0) : t > 5 / 2 :=
sorry

-- Problem 2
theorem problem2 (hp: 4 - t > t - 2) (hq : t - 2 > 0) (hdisjoint : (∃ (p : Prop) (q : Prop), (p ∨ q) ∧ ¬(p ∧ q))):
  (2 < t ∧ t ≤ 5 / 2) ∨ (t ≥ 3) :=
sorry


end problem1_problem2_l213_213953


namespace FerrisWheelCostIsTwo_l213_213342

noncomputable def costFerrisWheel (rollerCoasterCost multipleRideDiscount coupon totalTicketsBought : ℝ) : ℝ :=
  totalTicketsBought + multipleRideDiscount + coupon - rollerCoasterCost

theorem FerrisWheelCostIsTwo :
  let rollerCoasterCost := 7.0
  let multipleRideDiscount := 1.0
  let coupon := 1.0
  let totalTicketsBought := 7.0
  costFerrisWheel rollerCoasterCost multipleRideDiscount coupon totalTicketsBought = 2.0 :=
by
  sorry

end FerrisWheelCostIsTwo_l213_213342


namespace reflection_matrix_values_l213_213925

theorem reflection_matrix_values (a b : ℝ) (I : Matrix (Fin 2) (Fin 2) ℝ) :
  let R : Matrix (Fin 2) (Fin 2) ℝ := ![![a, 9/26], ![b, 17/26]]
  (R * R = I) → a = -17/26 ∧ b = 0 :=
by
  sorry

end reflection_matrix_values_l213_213925


namespace doubled_marks_new_average_l213_213760

theorem doubled_marks_new_average (avg_marks : ℝ) (num_students : ℕ) (h_avg : avg_marks = 36) (h_num : num_students = 12) : 2 * avg_marks = 72 :=
by
  sorry

end doubled_marks_new_average_l213_213760


namespace total_crayons_l213_213779
-- Import the whole Mathlib to ensure all necessary components are available

-- Definitions of the number of crayons each person has
def Billy_crayons : ℕ := 62
def Jane_crayons : ℕ := 52
def Mike_crayons : ℕ := 78
def Sue_crayons : ℕ := 97

-- Theorem stating the total number of crayons is 289
theorem total_crayons : (Billy_crayons + Jane_crayons + Mike_crayons + Sue_crayons) = 289 := by
  sorry

end total_crayons_l213_213779


namespace probability_green_then_blue_l213_213484

theorem probability_green_then_blue :
  let total_marbles := 10
  let green_marbles := 6
  let blue_marbles := 4
  let prob_first_green := green_marbles / total_marbles
  let prob_second_blue := blue_marbles / (total_marbles - 1)
  prob_first_green * prob_second_blue = 4 / 15 :=
sorry

end probability_green_then_blue_l213_213484


namespace brenda_age_l213_213055

variables (A B J : ℝ)

-- Conditions
def condition1 : Prop := A = 4 * B
def condition2 : Prop := J = B + 7
def condition3 : Prop := A = J

-- Target to prove
theorem brenda_age (h1 : condition1 A B) (h2 : condition2 B J) (h3 : condition3 A J) : B = 7 / 3 :=
by
  sorry

end brenda_age_l213_213055


namespace distance_metric_l213_213355

noncomputable def d (x y : ℝ) : ℝ :=
  (|x - y|) / (Real.sqrt (1 + x^2) * Real.sqrt (1 + y^2))

theorem distance_metric (x y z : ℝ) :
  (d x x = 0) ∧
  (d x y = d y x) ∧
  (d x y + d y z ≥ d x z) := by
  sorry

end distance_metric_l213_213355


namespace money_given_to_cashier_l213_213488

theorem money_given_to_cashier (regular_ticket_cost : ℕ) (discount : ℕ) 
  (age1 : ℕ) (age2 : ℕ) (change : ℕ) 
  (h1 : regular_ticket_cost = 109)
  (h2 : discount = 5)
  (h3 : age1 = 6)
  (h4 : age2 = 10)
  (h5 : change = 74)
  (h6 : age1 < 12)
  (h7 : age2 < 12) :
  regular_ticket_cost + regular_ticket_cost + (regular_ticket_cost - discount) + (regular_ticket_cost - discount) + change = 500 :=
by
  sorry

end money_given_to_cashier_l213_213488


namespace exists_sequence_a_l213_213670

def c (n : ℕ) : ℕ := 2017 ^ n

axiom f : ℕ → ℝ

axiom condition_1 : ∀ m n : ℕ, f (m + n) ≤ 2017 * f m * f (n + 325)

axiom condition_2 : ∀ n : ℕ, 0 < f (c (n + 1)) ∧ f (c (n + 1)) < (f (c n)) ^ 2017

theorem exists_sequence_a :
  ∃ (a : ℕ → ℕ), ∀ n k : ℕ, a k < n → f n ^ c k < f (c k) ^ n := sorry

end exists_sequence_a_l213_213670


namespace johns_weekly_allowance_l213_213957

theorem johns_weekly_allowance (A : ℝ) (h1: A - (3/5) * A = (2/5) * A)
  (h2: (2/5) * A - (1/3) * (2/5) * A = (4/15) * A)
  (h3: (4/15) * A = 0.92) : A = 3.45 :=
by {
  sorry
}

end johns_weekly_allowance_l213_213957


namespace polynomial_root_sum_nonnegative_l213_213176

noncomputable def f (x : ℝ) (b c : ℝ) : ℝ := x^2 + b * x + c
noncomputable def g (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q

theorem polynomial_root_sum_nonnegative 
  (m1 m2 k1 k2 b c p q : ℝ)
  (h1 : f m1 b c = 0) (h2 : f m2 b c = 0)
  (h3 : g k1 p q = 0) (h4 : g k2 p q = 0) :
  f k1 b c + f k2 b c + g m1 p q + g m2 p q ≥ 0 := 
by
  sorry  -- Proof placeholders

end polynomial_root_sum_nonnegative_l213_213176


namespace simplify_fraction_l213_213305

theorem simplify_fraction (a b c d k : ℕ) (h₁ : a = 123) (h₂ : b = 9999) (h₃ : k = 41)
                           (h₄ : c = a / 3) (h₅ : d = b / 3)
                           (h₆ : c = k) (h₇ : d = 3333) :
  (a * k) / b = (k^2) / d :=
by
  sorry

end simplify_fraction_l213_213305


namespace discount_per_coupon_l213_213869

-- Definitions and conditions from the problem
def num_cans : ℕ := 9
def cost_per_can : ℕ := 175 -- in cents
def num_coupons : ℕ := 5
def total_payment : ℕ := 2000 -- $20 in cents
def change_received : ℕ := 550 -- $5.50 in cents
def amount_paid := total_payment - change_received

-- Mathematical proof problem
theorem discount_per_coupon :
  let total_cost_without_coupons := num_cans * cost_per_can 
  let total_discount := total_cost_without_coupons - amount_paid
  let discount_per_coupon := total_discount / num_coupons
  discount_per_coupon = 25 :=
by
  sorry

end discount_per_coupon_l213_213869


namespace math_club_problem_l213_213636

theorem math_club_problem :
  ((7^2 - 5^2) - 2)^3 = 10648 := by
  sorry

end math_club_problem_l213_213636


namespace log3_of_7_eq_ab_l213_213676

noncomputable def log3_of_2_eq_a (a : ℝ) : Prop := Real.log 2 / Real.log 3 = a
noncomputable def log2_of_7_eq_b (b : ℝ) : Prop := Real.log 7 / Real.log 2 = b

theorem log3_of_7_eq_ab (a b : ℝ) (h1 : log3_of_2_eq_a a) (h2 : log2_of_7_eq_b b) :
  Real.log 7 / Real.log 3 = a * b :=
sorry

end log3_of_7_eq_ab_l213_213676


namespace insufficient_info_for_pumpkins_l213_213283

variable (jason_watermelons : ℕ) (sandy_watermelons : ℕ) (total_watermelons : ℕ)

theorem insufficient_info_for_pumpkins (h1 : jason_watermelons = 37)
  (h2 : sandy_watermelons = 11)
  (h3 : jason_watermelons + sandy_watermelons = total_watermelons)
  (h4 : total_watermelons = 48) : 
  ¬∃ (jason_pumpkins : ℕ), true
:= by
  sorry

end insufficient_info_for_pumpkins_l213_213283


namespace johns_total_money_l213_213431

-- Defining the given conditions
def initial_amount : ℕ := 5
def amount_spent : ℕ := 2
def allowance : ℕ := 26

-- Constructing the proof statement
theorem johns_total_money : initial_amount - amount_spent + allowance = 29 :=
by
  sorry

end johns_total_money_l213_213431


namespace sum_of_squares_is_42_l213_213888

variables (D T H : ℕ)

theorem sum_of_squares_is_42
  (h1 : 3 * D + T = 2 * H)
  (h2 : 2 * H^3 = 3 * D^3 + T^3)
  (coprime : Nat.gcd (Nat.gcd D T) H = 1) :
  (T^2 + D^2 + H^2 = 42) :=
sorry

end sum_of_squares_is_42_l213_213888


namespace central_angle_of_sector_l213_213247

theorem central_angle_of_sector (l S : ℝ) (r : ℝ) (θ : ℝ) 
  (h1 : l = 5) 
  (h2 : S = 5) 
  (h3 : S = (1 / 2) * l * r) 
  (h4 : l = θ * r): θ = 2.5 := by
  sorry

end central_angle_of_sector_l213_213247


namespace num_divisors_36_l213_213119

theorem num_divisors_36 : ∃ n : ℕ, n = 18 ∧ ∀ d : ℤ, (d ≠ 0 → 36 % d = 0) → nat_abs d ∣ 36 :=
by
  sorry

end num_divisors_36_l213_213119


namespace divisors_of_36_l213_213118

def is_divisor (n : Int) (d : Int) : Prop := d ≠ 0 ∧ n % d = 0

def positive_divisors (n : Int) : List Int := 
  List.filter (λ d, d > 0 ∧ is_divisor n d) (List.range (Int.toNat n + 1))

def total_divisors (n : Int) : List Int :=
  positive_divisors n ++ List.map (λ d, -d) (positive_divisors n)

theorem divisors_of_36 : ∃ d, d = 36 ∧ (total_divisors d).length = 18 := by
  sorry

end divisors_of_36_l213_213118


namespace picnic_weather_condition_l213_213107

variables (P Q : Prop)

theorem picnic_weather_condition (h : ¬P → ¬Q) : Q → P := 
by sorry

end picnic_weather_condition_l213_213107


namespace sum_a6_a7_a8_l213_213939

-- Sequence definition and sum of the first n terms
def S (n : ℕ) : ℕ := n^2 + 3 * n

theorem sum_a6_a7_a8 : S 8 - S 5 = 48 :=
by
  -- Definition and proof details are skipped
  sorry

end sum_a6_a7_a8_l213_213939


namespace false_implies_exists_nonpositive_l213_213441

variable (f : ℝ → ℝ)

theorem false_implies_exists_nonpositive (h : ¬ ∀ x > 0, f x > 0) : ∃ x > 0, f x ≤ 0 :=
by sorry

end false_implies_exists_nonpositive_l213_213441


namespace remaining_sessions_l213_213356

theorem remaining_sessions (total_sessions : ℕ) (p1_sessions : ℕ) (p2_sessions_more : ℕ) (remaining_sessions : ℕ) :
  total_sessions = 25 →
  p1_sessions = 6 →
  p2_sessions_more = 5 →
  remaining_sessions = total_sessions - (p1_sessions + (p1_sessions + p2_sessions_more)) →
  remaining_sessions = 8 :=
by
  intros
  sorry

end remaining_sessions_l213_213356


namespace difference_of_numbers_l213_213612

theorem difference_of_numbers :
  ∃ (a b : ℕ), a + b = 36400 ∧ b = 100 * a ∧ b - a = 35640 :=
by
  sorry

end difference_of_numbers_l213_213612


namespace min_value_of_expression_l213_213541

theorem min_value_of_expression (a : ℝ) (h₀ : a > 0)
  (x₁ x₂ : ℝ)
  (h₁ : x₁ + x₂ = 4 * a)
  (h₂ : x₁ * x₂ = a * a) :
  x₁ + x₂ + a / (x₁ * x₂) = 4 :=
sorry

end min_value_of_expression_l213_213541


namespace num_divisors_of_36_l213_213117

theorem num_divisors_of_36 : 
  let positive_divisors : List Int := [1, 2, 3, 4, 6, 9, 12, 18, 36] in
  let total_divisors := 2 * List.length positive_divisors in
  total_divisors = 18 :=
by
  let positive_divisors : List Int := [1, 2, 3, 4, 6, 9, 12, 18, 36]
  let total_divisors := 2 * List.length positive_divisors
  show total_divisors = 18
  sorry

end num_divisors_of_36_l213_213117


namespace family_reunion_attendance_l213_213501

-- Define the conditions
def male_adults : ℕ := 100
def female_adults : ℕ := male_adults + 50
def total_adults : ℕ := male_adults + female_adults
def children : ℕ := 2 * total_adults

-- State the theorem to be proven
theorem family_reunion_attendance : 
  let total_people := total_adults + children in
  total_people = 750 :=
by 
  sorry

end family_reunion_attendance_l213_213501


namespace product_sum_of_roots_l213_213999

theorem product_sum_of_roots (p q r : ℂ)
  (h_eq : ∀ x : ℂ, (2 : ℂ) * x^3 + (1 : ℂ) * x^2 + (-7 : ℂ) * x + (2 : ℂ) = 0 → (x = p ∨ x = q ∨ x = r)) 
  : p * q + q * r + r * p = -7 / 2 := 
sorry

end product_sum_of_roots_l213_213999


namespace find_a_l213_213103

noncomputable def f (x a : ℝ) : ℝ := x / (x^2 + a)

theorem find_a (a : ℝ) (h_positive : a > 0) (h_max : ∀ x, x ∈ Set.Ici 1 → f x a ≤ f 1 a) :
  a = Real.sqrt 3 - 1 := by
  sorry

end find_a_l213_213103


namespace infinite_squares_form_l213_213848

theorem infinite_squares_form (k : ℕ) (hk : 0 < k) : ∃ f : ℕ → ℕ, ∀ n, ∃ a, a^2 = f n * 2^k - 7 :=
by
  sorry

end infinite_squares_form_l213_213848


namespace find_a_l213_213596

theorem find_a (f g : ℝ → ℝ) (a : ℝ) 
  (h₁ : ∀ x, f x = 2 * x / 3 + 4) 
  (h₂ : ∀ x, g x = 5 - 2 * x) 
  (h₃ : f (g a) = 7) : 
  a = 1 / 4 := 
sorry

end find_a_l213_213596


namespace current_height_of_tree_l213_213912

-- Definitions of conditions
def growth_per_year : ℝ := 0.5
def years : ℕ := 240
def final_height : ℝ := 720

-- The goal is to prove that the current height of the tree is 600 inches
theorem current_height_of_tree :
  final_height - (growth_per_year * years) = 600 := 
sorry

end current_height_of_tree_l213_213912


namespace find_k_l213_213878

-- Define the conditions
def equation : polynomial ℝ := polynomial.X^2 + 8 * polynomial.X + k
def ratio (r s : ℝ) := r = 3 * s

-- The main theorem
theorem find_k (k r s : ℝ) (h : polynomial.roots equation = {r, s}) (hratio: ratio r s) (hnonzero : r ≠ 0 ∧ s ≠ 0) : 
  k = 12 :=
sorry

end find_k_l213_213878


namespace three_digit_integers_count_l213_213682

theorem three_digit_integers_count : 
  ∃ (n : ℕ), n = 24 ∧
  (∃ (digits : Finset ℕ), digits = {2, 4, 7, 9} ∧
  (∀ a b c : ℕ, a ∈ digits → b ∈ digits → c ∈ digits → a ≠ b → b ≠ c → a ≠ c → 
  100 * a + 10 * b + c ∈ {y | 100 ≤ y ∧ y < 1000} → 4 * 3 * 2 = 24)) :=
by
  sorry

end three_digit_integers_count_l213_213682


namespace find_a_l213_213538

theorem find_a (f : ℤ → ℤ) (h1 : ∀ (x : ℤ), f (2 * x + 1) = 3 * x + 2) (h2 : f a = 2) : a = 1 := by
sorry

end find_a_l213_213538


namespace four_digit_number_divisibility_l213_213159

theorem four_digit_number_divisibility 
  (E V I L : ℕ) 
  (hE : 0 ≤ E ∧ E < 10) 
  (hV : 0 ≤ V ∧ V < 10) 
  (hI : 0 ≤ I ∧ I < 10) 
  (hL : 0 ≤ L ∧ L < 10)
  (h1 : (1000 * E + 100 * V + 10 * I + L) % 73 = 0) 
  (h2 : (1000 * V + 100 * I + 10 * L + E) % 74 = 0)
  : 1000 * L + 100 * I + 10 * V + E = 5499 := 
  sorry

end four_digit_number_divisibility_l213_213159


namespace probability_of_AB_not_selected_l213_213802

-- The definition for the probability of not selecting both A and B 
def probability_not_selected : ℚ :=
  let total_ways := Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial (4 - 2))
  let favorable_ways := 1 -- Only the selection of C and D
  favorable_ways / total_ways

-- The theorem stating the desired probability
theorem probability_of_AB_not_selected : probability_not_selected = 1 / 6 :=
by
  sorry

end probability_of_AB_not_selected_l213_213802


namespace work_combined_days_l213_213349

theorem work_combined_days (A B C : ℝ) (hA : A = 1 / 4) (hB : B = 1 / 12) (hC : C = 1 / 6) :
  1 / (A + B + C) = 2 :=
by
  sorry

end work_combined_days_l213_213349


namespace angle_bisector_slope_l213_213598

theorem angle_bisector_slope
  (m₁ m₂ : ℝ) (h₁ : m₁ = 2) (h₂ : m₂ = -1) (k : ℝ)
  (h_k : k = (m₁ + m₂ + Real.sqrt ((m₁ - m₂)^2 + 4)) / 2) :
  k = (1 + Real.sqrt 13) / 2 :=
by
  rw [h₁, h₂] at h_k
  sorry

end angle_bisector_slope_l213_213598


namespace trapezoid_base_ratio_l213_213163

-- Define variables and conditions as per the problem
variables {a b h: ℝ}
def is_trapezoid (a b: ℝ) := a > b
def area_trapezoid (a b h: ℝ) := (1 / 2) * (a + b) * h
def area_quadrilateral (a b h: ℝ) := (1 / 2) * ((a - b) / 2) * (h / 2)

-- State the problem statement
theorem trapezoid_base_ratio (a b h: ℝ) (ha: a > b) (ht: area_quadrilateral a b h = (1 / 4) * area_trapezoid a b h) :
  a / b = 3 :=
sorry

end trapezoid_base_ratio_l213_213163


namespace mary_spent_total_amount_l213_213298

def shirt_cost : ℝ := 13.04
def jacket_cost : ℝ := 12.27
def total_cost : ℝ := 25.31

theorem mary_spent_total_amount :
  shirt_cost + jacket_cost = total_cost := sorry

end mary_spent_total_amount_l213_213298


namespace union_sets_l213_213821

-- Define the sets A and B based on their conditions
def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 5 }
def B : Set ℝ := { x | 3 < x ∧ x < 9 }

-- Statement of the proof problem
theorem union_sets (x : ℝ) : (x ∈ A ∪ B) ↔ (x ∈ { x | -1 ≤ x ∧ x < 9 }) := sorry

end union_sets_l213_213821


namespace find_values_of_a_and_b_l213_213527

theorem find_values_of_a_and_b (a b : ℚ) (h1 : 4 * a + 2 * b = 92) (h2 : 6 * a - 4 * b = 60) : 
  a = 122 / 7 ∧ b = 78 / 7 :=
by {
  sorry
}

end find_values_of_a_and_b_l213_213527


namespace smallest_AAB_value_l213_213362

theorem smallest_AAB_value :
  ∃ (A B : ℕ), 
  A ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  B ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  A ≠ B ∧ 
  110 * A + B = 7 * (10 * A + B) ∧ 
  (∀ (A' B' : ℕ), 
    A' ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    B' ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    A' ≠ B' ∧ 
    110 * A' + B' = 7 * (10 * A' + B') → 
    110 * A + B ≤ 110 * A' + B') :=
by
  sorry

end smallest_AAB_value_l213_213362


namespace c_plus_2d_eq_neg59_l213_213602

theorem c_plus_2d_eq_neg59 (c d : ℤ) (h1 : (5 * (X : ℤ[X]) + c) * (5 * X + d) = 25 * X^2 - 135 * X - 150) : c + 2 * d = -59 :=
sorry

end c_plus_2d_eq_neg59_l213_213602


namespace find_total_amount_l213_213412

-- Definitions according to the conditions
def is_proportion (a b c : ℚ) (p q r : ℚ) : Prop :=
  (a * q = b * p) ∧ (a * r = c * p) ∧ (b * r = c * q)

def total_amount (second_part : ℚ) (prop_total : ℚ) : ℚ :=
  second_part / (1/3) * prop_total

-- Main statement to be proved
theorem find_total_amount (second_part : ℚ) (p1 p2 p3 : ℚ)
  (h : is_proportion p1 p2 p3 (1/2 : ℚ) (1/3 : ℚ) (3/4 : ℚ))
  : second_part = 164.6315789473684 → total_amount second_part (19/12 : ℚ) = 65.16 :=
by
  sorry

end find_total_amount_l213_213412


namespace find_ab_l213_213478

theorem find_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) :
  a * b = 10 :=
by
  sorry

end find_ab_l213_213478


namespace hyperbola_eq_from_conditions_l213_213491

-- Conditions of the problem
def hyperbola_center : Prop := ∃ (h : ℝ → ℝ → Prop), h 0 0
def hyperbola_eccentricity : Prop := ∃ e : ℝ, e = 2
def parabola_focus : Prop := ∃ p : ℝ × ℝ, p = (4, 0)
def parabola_equation : Prop := ∀ x y : ℝ, y^2 = 8 * x

-- Hyperbola equation to be proved
def hyperbola_equation : Prop := ∀ x y : ℝ, (x^2 / 4) - (y^2 / 12) = 1

-- Lean 4 theorem statement
theorem hyperbola_eq_from_conditions 
  (h_center : hyperbola_center) 
  (h_eccentricity : hyperbola_eccentricity) 
  (p_focus : parabola_focus) 
  (p_eq : parabola_equation) 
  : hyperbola_equation :=
by
  sorry

end hyperbola_eq_from_conditions_l213_213491


namespace quarter_pounder_cost_l213_213069

theorem quarter_pounder_cost :
  let fries_cost := 2 * 1.90
  let milkshakes_cost := 2 * 2.40
  let min_purchase := 18
  let current_total := fries_cost + milkshakes_cost
  let amount_needed := min_purchase - current_total
  let additional_spending := 3
  let total_cost := amount_needed + additional_spending
  total_cost = 12.40 :=
by
  sorry

end quarter_pounder_cost_l213_213069


namespace part1_part2_l213_213260

section

variable (a x : ℝ)

def A : Set ℝ := { x | x ≤ -1 } ∪ { x | x ≥ 5 }
def B (a : ℝ) : Set ℝ := { x | 2 * a ≤ x ∧ x ≤ a + 2 }

-- Part 1
theorem part1 (h : a = -1) :
  B a = { x | -2 ≤ x ∧ x ≤ 1 } ∧
  (A ∩ B a) = { x | -2 ≤ x ∧ x ≤ -1 } ∧
  (A ∪ B a) = { x | x ≤ 1 ∨ x ≥ 5 } := 
sorry

-- Part 2
theorem part2 (h : A ∩ B a = B a) :
  a ≤ -3 ∨ a > 2 := 
sorry

end

end part1_part2_l213_213260


namespace angle_BC₁_plane_BBD₁D_l213_213139

-- Define all the necessary components of the cube and its geometry
variables {A B C D A₁ B₁ C₁ D₁ : ℝ} -- placeholders for points, represented by real coordinates

def is_cube (A B C D A₁ B₁ C₁ D₁ : ℝ) : Prop := sorry -- Define the cube property (this would need a proper definition)

def space_diagonal (B C₁ : ℝ) : Prop := sorry -- Define the property of being a space diagonal

def plane (B B₁ D₁ D : ℝ) : Prop := sorry -- Define a plane through these points (again needs a definition)

-- Define the angle between a line and a plane
def angle_between_line_and_plane (BC₁ B B₁ D₁ D : ℝ) : ℝ := sorry -- Define angle calculation (requires more context)

-- The proof statement, which is currently not proven (contains 'sorry')
theorem angle_BC₁_plane_BBD₁D (s : ℝ):
  is_cube A B C D A₁ B₁ C₁ D₁ →
  space_diagonal B C₁ →
  plane B B₁ D₁ D →
  angle_between_line_and_plane B C₁ B₁ D₁ D = π / 6 :=
sorry

end angle_BC₁_plane_BBD₁D_l213_213139


namespace circumradius_of_triangle_l213_213424

theorem circumradius_of_triangle (a b S : ℝ) (A : a = 2) (B : b = 3) (Area : S = 3 * Real.sqrt 15 / 4)
  (median_cond : ∃ c m, m = (a^2 + b^2 - c^2) / (2*a*b) ∧ m < c / 2) :
  ∃ R, R = 8 / Real.sqrt 15 :=
by
  sorry

end circumradius_of_triangle_l213_213424


namespace fiona_frog_probability_l213_213799

/--
  Fiona the frog starts on lily pad 0 in a row of lily pads numbered from 0 to 15. 
  There are predators on lily pads 4, 7, and 11. 
  A morsel of food is placed on lily pad 14. 
  Fiona can either hop to the next pad, jump two pads, or jump three pads forward, each with equal probability of 1/3.
  Prove that the probability that Fiona reaches pad 14 without landing on any of the predator pads is 10/6561.
-/
theorem fiona_frog_probability : 
  let probability_to_reach_14 : ℚ := 10 / 6561
  in fiona_reaches_14_safely (pads : finset ℕ) (start : ℕ) (predators : finset ℕ) (food : ℕ) :=
  start = 0 ∧ pads = (0 : ℕ) ... 15 ∧ 
  predators = {4, 7, 11} ∧ food = 14 ∧ 
  ∀ (k : ℕ), (k > 0 ∧ k < 16) → (hop : ℕ), (hop = 1 ∨ hop = 2 ∨ hop = 3) → (prob (k, hop) = 1 / 3) 
  → (fiona_reaches_food_without_predators = probability_to_reach_14)
sorry

end fiona_frog_probability_l213_213799


namespace scientific_notation_correct_l213_213897

theorem scientific_notation_correct : 1630000 = 1.63 * 10^6 :=
by sorry

end scientific_notation_correct_l213_213897


namespace min_value_sum_pos_int_l213_213947

theorem min_value_sum_pos_int 
  (a b c : ℕ)
  (h_pos: a > 0 ∧ b > 0 ∧ c > 0)
  (h_roots: ∃ (A B : ℝ), A < 0 ∧ A > -1 ∧ B > 0 ∧ B < 1 ∧ (∀ x : ℝ, x^2*x*a + x*b + c = 0 → x = A ∨ x = B))
  : a + b + c = 11 :=
sorry

end min_value_sum_pos_int_l213_213947


namespace valid_passwords_count_l213_213214

-- Define the total number of unrestricted passwords (each digit can be 0-9)
def total_passwords := 10^5

-- Define the number of restricted passwords (those starting with the sequence 8,3,2)
def restricted_passwords := 10^2

-- State the main theorem to be proved
theorem valid_passwords_count : total_passwords - restricted_passwords = 99900 := by
  sorry

end valid_passwords_count_l213_213214


namespace divisors_of_36_count_l213_213114

theorem divisors_of_36_count : 
  {n : ℤ | n ∣ 36}.to_finset.card = 18 := 
sorry

end divisors_of_36_count_l213_213114


namespace veranda_width_l213_213603

-- Defining the conditions as given in the problem
def room_length : ℝ := 21
def room_width : ℝ := 12
def veranda_area : ℝ := 148

-- The main statement to prove
theorem veranda_width :
  ∃ (w : ℝ), (21 + 2*w) * (12 + 2*w) - 21 * 12 = 148 ∧ w = 2 :=
by
  sorry

end veranda_width_l213_213603


namespace integer_solution_l213_213338

theorem integer_solution (x : ℕ) (h : (4 * x)^2 - 2 * x = 3178) : x = 226 :=
by
  sorry

end integer_solution_l213_213338


namespace vector_addition_scalar_multiplication_l213_213661

def u : ℝ × ℝ × ℝ := (3, -2, 5)
def v : ℝ × ℝ × ℝ := (-1, 6, -3)
def result : ℝ × ℝ × ℝ := (4, 8, 4)

theorem vector_addition_scalar_multiplication :
  2 • (u + v) = result :=
by
  sorry

end vector_addition_scalar_multiplication_l213_213661


namespace not_B_l213_213644

def op (x y : ℝ) := (x - y) ^ 2

theorem not_B (x y : ℝ) : 2 * (op x y) ≠ op (2 * x) (2 * y) :=
by
  sorry

end not_B_l213_213644


namespace bag_cost_is_10_l213_213889

def timothy_initial_money : ℝ := 50
def tshirt_cost : ℝ := 8
def keychain_cost : ℝ := 2
def keychains_per_set : ℝ := 3
def number_of_tshirts : ℝ := 2
def number_of_bags : ℝ := 2
def number_of_keychains : ℝ := 21

noncomputable def cost_of_each_bag : ℝ :=
  let cost_of_tshirts := number_of_tshirts * tshirt_cost
  let remaining_money_after_tshirts := timothy_initial_money - cost_of_tshirts
  let cost_of_keychains := (number_of_keychains / keychains_per_set) * keychain_cost
  let remaining_money_after_keychains := remaining_money_after_tshirts - cost_of_keychains
  remaining_money_after_keychains / number_of_bags

theorem bag_cost_is_10 :
  cost_of_each_bag = 10 := by
  sorry

end bag_cost_is_10_l213_213889


namespace prove_a2_b2_c2_zero_l213_213710

theorem prove_a2_b2_c2_zero (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) (h5 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) : a^2 + b^2 + c^2 = 0 := 
by 
  sorry

end prove_a2_b2_c2_zero_l213_213710


namespace sum_of_eight_numbers_l213_213975

theorem sum_of_eight_numbers (average : ℝ) (count : ℕ) (h_avg : average = 5.3) (h_count : count = 8) : (average * count = 42.4) := sorry

end sum_of_eight_numbers_l213_213975


namespace students_in_diligence_before_transfer_l213_213274

theorem students_in_diligence_before_transfer (D I : ℕ) 
  (h1 : D + 2 = I - 2) 
  (h2 : D + I = 50) : 
  D = 23 := 
by
  sorry

end students_in_diligence_before_transfer_l213_213274


namespace curve_is_line_l213_213792

theorem curve_is_line (r θ : ℝ) (h : r = 1 / (2 * Real.sin θ - Real.cos θ)) :
  ∃ (a b c : ℝ), a * (r * Real.cos θ) + b * (r * Real.sin θ) + c = 0 ∧
  (a, b, c) = (-1, 2, -1) := sorry

end curve_is_line_l213_213792


namespace rita_total_hours_l213_213730

def h_backstroke : ℕ := 50
def h_breaststroke : ℕ := 9
def h_butterfly : ℕ := 121
def h_freestyle_sidestroke_per_month : ℕ := 220
def months : ℕ := 6

def h_total : ℕ := h_backstroke + h_breaststroke + h_butterfly + (h_freestyle_sidestroke_per_month * months)

theorem rita_total_hours :
  h_total = 1500 :=
by
  sorry

end rita_total_hours_l213_213730


namespace summation_series_equals_half_l213_213654

theorem summation_series_equals_half :
  (\sum_{n=0}^{∞} 2^n / (3^(2^n) + 1)) = 1 / 2 :=
by
  sorry

end summation_series_equals_half_l213_213654


namespace quadratic_coefficients_l213_213081

theorem quadratic_coefficients (x1 x2 p q : ℝ)
  (h1 : x1 - x2 = 5)
  (h2 : x1 ^ 3 - x2 ^ 3 = 35) :
  (x1 + x2 = -p ∧ x1 * x2 = q ∧ (p = 1 ∧ q = -6) ∨ 
   x1 + x2 = p ∧ x1 * x2 = q ∧ (p = -1 ∧ q = -6)) :=
by
  sorry

end quadratic_coefficients_l213_213081


namespace part1_optimal_strategy_part2_optimal_strategy_l213_213482

noncomputable def R (x1 x2 : ℝ) : ℝ := -2 * x1^2 - x2^2 + 13 * x1 + 11 * x2 - 28

theorem part1_optimal_strategy :
  ∃ x1 x2 : ℝ, x1 + x2 = 5 ∧ x1 = 2 ∧ x2 = 3 ∧
    ∀ y1 y2, y1 + y2 = 5 → (R y1 y2 - (y1 + y2) ≤ R x1 x2 - (x1 + x2)) := 
by
  sorry

theorem part2_optimal_strategy :
  ∃ x1 x2 : ℝ, x1 = 3 ∧ x2 = 5 ∧
    ∀ y1 y2, (R y1 y2 - (y1 + y2) ≤ R x1 x2 - (x1 + x2)) := 
by
  sorry

end part1_optimal_strategy_part2_optimal_strategy_l213_213482


namespace mean_home_runs_l213_213740

theorem mean_home_runs :
  let players_with_5 := 3
  let players_with_6 := 4
  let players_with_8 := 2
  let players_with_9 := 1
  let players_with_11 := 1
  let total_home_runs := (5 * players_with_5) + (6 * players_with_6) + (8 * players_with_8) + (9 * players_with_9) + (11 * players_with_11)
  let total_players := players_with_5 + players_with_6 + players_with_8 + players_with_9 + players_with_11
  (total_home_runs / total_players : ℚ) = 75 / 11 :=
by
  sorry

end mean_home_runs_l213_213740


namespace quadratic_root_condition_l213_213316

theorem quadratic_root_condition (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 > 1 ∧ x2 < 1 ∧ x1^2 + 2*a*x1 + 1 = 0 ∧ x2^2 + 2*a*x2 + 1 = 0) →
  a < -1 :=
by
  sorry

end quadratic_root_condition_l213_213316


namespace goods_train_length_is_470_l213_213490

noncomputable section

def speed_kmph := 72
def platform_length := 250
def crossing_time := 36

def speed_mps := speed_kmph * 5 / 18
def distance_covered := speed_mps * crossing_time

def length_of_train := distance_covered - platform_length

theorem goods_train_length_is_470 :
  length_of_train = 470 :=
by
  sorry

end goods_train_length_is_470_l213_213490


namespace quadratic_has_distinct_real_roots_l213_213882

-- Definitions for the quadratic equation coefficients
def a : ℝ := 3
def b : ℝ := -4
def c : ℝ := 1

-- Definition of the discriminant
def Δ : ℝ := b^2 - 4 * a * c

-- Statement of the problem: Prove that the quadratic equation has two distinct real roots
theorem quadratic_has_distinct_real_roots (hΔ : Δ = 4) : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) :=
by
  sorry

end quadratic_has_distinct_real_roots_l213_213882


namespace initial_number_of_men_l213_213168

theorem initial_number_of_men
  (M : ℕ) (A : ℕ)
  (h1 : ∀ A_new : ℕ, A_new = A + 4)
  (h2 : ∀ total_age_increase : ℕ, total_age_increase = (2 * 52) - (36 + 32))
  (h3 : ∀ sum_age_men : ℕ, sum_age_men = M * A)
  (h4 : ∀ new_sum_age_men : ℕ, new_sum_age_men = sum_age_men + ((2 * 52) - (36 + 32))) :
  M = 9 := 
by
  -- Proof skipped
  sorry

end initial_number_of_men_l213_213168


namespace isosceles_triangle_perimeter_correct_l213_213270

noncomputable def isosceles_triangle_perimeter (x y : ℝ) : ℝ :=
  if x = y then 2 * x + y else if (2 * x > y ∧ y > 2 * x - y) ∨ (2 * y > x ∧ x > 2 * y - x) then 2 * y + x else 0

theorem isosceles_triangle_perimeter_correct (x y : ℝ) (h : |x - 5| + (y - 8)^2 = 0) :
  isosceles_triangle_perimeter x y = 18 ∨ isosceles_triangle_perimeter x y = 21 := by
sorry

end isosceles_triangle_perimeter_correct_l213_213270


namespace trapezoid_base_ratio_l213_213167

theorem trapezoid_base_ratio
  (a b h : ℝ)  -- lengths of the bases and height
  (ha_gt_hb : a > b)
  (trapezoid_area : ℝ) 
  (quad_area : ℝ) 
  (h1 : trapezoid_area = (1/2) * (a + b) * h) 
  (h2 : quad_area = (1/2) * (a - b) * h / 4)
  (h3 : quad_area = trapezoid_area / 4) :
  a / b = 3 :=
by {
  sorry,
}

end trapezoid_base_ratio_l213_213167


namespace fish_tagged_initially_l213_213419

theorem fish_tagged_initially (N T : ℕ) (hN : N = 1500) 
  (h_ratio : 2 / 50 = (T:ℕ) / N) : T = 60 :=
by
  -- The proof is omitted
  sorry

end fish_tagged_initially_l213_213419


namespace rayden_spent_more_l213_213868

-- Define the conditions
def lily_ducks := 20
def lily_geese := 10
def lily_chickens := 5
def lily_pigeons := 30

def rayden_ducks := 3 * lily_ducks
def rayden_geese := 4 * lily_geese
def rayden_chickens := 5 * lily_chickens
def rayden_pigeons := lily_pigeons / 2

def duck_price := 15
def geese_price := 20
def chicken_price := 10
def pigeon_price := 5

def lily_total := lily_ducks * duck_price +
                  lily_geese * geese_price +
                  lily_chickens * chicken_price +
                  lily_pigeons * pigeon_price

def rayden_total := rayden_ducks * duck_price +
                    rayden_geese * geese_price +
                    rayden_chickens * chicken_price +
                    rayden_pigeons * pigeon_price

def spending_difference := rayden_total - lily_total

theorem rayden_spent_more : spending_difference = 1325 := 
by 
  unfold spending_difference rayden_total lily_total -- to simplify the definitions
  sorry -- Proof is omitted

end rayden_spent_more_l213_213868


namespace problem_solution_l213_213008

theorem problem_solution (x y : ℝ) (h1 : x + y = 500) (h2 : x / y = 0.8) : y - x = 500 / 9 :=
by
  sorry

end problem_solution_l213_213008


namespace line_through_two_points_l213_213030

theorem line_through_two_points (x_1 y_1 x_2 y_2 x y : ℝ) :
  (x - x_1) * (y_2 - y_1) = (y - y_1) * (x_2 - x_1) :=
sorry

end line_through_two_points_l213_213030


namespace n_minus_m_l213_213835

theorem n_minus_m (m n : ℤ) (h_m : m - 2 = 3) (h_n : n + 1 = 2) : n - m = -4 := sorry

end n_minus_m_l213_213835


namespace range_of_x_l213_213540

noncomputable def f (x : ℝ) : ℝ := x^3 + x

theorem range_of_x (x m : ℝ) (hx : x > -2 ∧ x < 2/3) (hm : m ≥ -2 ∧ m ≤ 2) :
    f (m * x - 2) + f x < 0 := sorry

end range_of_x_l213_213540


namespace otimes_2_5_l213_213437

def otimes (a b : ℝ) : ℝ := 4 * a + 3 * b

theorem otimes_2_5 : otimes 2 5 = 23 :=
by
  sorry

end otimes_2_5_l213_213437


namespace reciprocal_of_neg_two_l213_213463

theorem reciprocal_of_neg_two :
  (∃ x : ℝ, x = -2 ∧ 1 / x = -1 / 2) :=
by
  use -2
  split
  · rfl
  · norm_num

end reciprocal_of_neg_two_l213_213463


namespace smallest_n_terminating_decimal_l213_213893

theorem smallest_n_terminating_decimal :
  ∃ n : ℕ, (n > 0) ∧
           (∃ (k: ℕ), (n + 150) = 2^k ∧ k < 150) ∨ 
           (∃ (k m: ℕ), (n + 150) = 2^k * 5^m ∧ m < 150) ∧ 
           ∀ m : ℕ, ((m > 0 ∧ (∃ (j: ℕ), (m + 150) = 2^j ∧ j < 150) ∨ 
           (∃ (j l: ℕ), (m + 150) = 2^j * 5^l ∧ l < 150)) → m ≥ n)
:= ⟨10, by {
  sorry
}⟩

end smallest_n_terminating_decimal_l213_213893


namespace joe_purchased_360_gallons_l213_213286

def joe_initial_paint (P : ℝ) : Prop :=
  let first_week_paint := (1/4) * P
  let remaining_paint := (3/4) * P
  let second_week_paint := (1/2) * remaining_paint
  let total_used_paint := first_week_paint + second_week_paint
  total_used_paint = 225

theorem joe_purchased_360_gallons : ∃ P : ℝ, joe_initial_paint P ∧ P = 360 :=
by
  sorry

end joe_purchased_360_gallons_l213_213286


namespace perpendicular_lines_b_l213_213788

theorem perpendicular_lines_b (b : ℝ) : 
  (∃ (k m: ℝ), k = 3 ∧ 2 * m + b * k = 14 ∧ (k * m = -1)) ↔ b = 2 / 3 :=
sorry

end perpendicular_lines_b_l213_213788


namespace solve_system_of_equations_l213_213458

theorem solve_system_of_equations :
  ∃ (x y : ℝ), x * y * (x + y) = 30 ∧ x^3 + y^3 = 35 ∧ ((x = 3 ∧ y = 2) ∨ (x = 2 ∧ y = 3)) :=
sorry

end solve_system_of_equations_l213_213458


namespace problem1_part1_problem1_part2_problem2_l213_213675

open Set

-- Definitions for sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 2}
def U : Set ℝ := univ

-- Part (1) of the problem
theorem problem1_part1 : A ∩ B = {x | 2 < x ∧ x ≤ 3} :=
sorry

theorem problem1_part2 : A ∪ (U \ B) = {x | x ≤ 3} :=
sorry

-- Definitions for set C
def C (a : ℝ) : Set ℝ := {x | 1 < x ∧ x < a}

-- Part (2) of the problem
theorem problem2 (a : ℝ) (h : C a ⊆ A) : 1 < a ∧ a ≤ 3 :=
sorry

end problem1_part1_problem1_part2_problem2_l213_213675


namespace sufficient_condition_for_inequality_l213_213094

theorem sufficient_condition_for_inequality (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) → a ≥ 3 :=
by
  sorry

end sufficient_condition_for_inequality_l213_213094


namespace expression_parity_l213_213850

theorem expression_parity (p m : ℤ) (hp : Odd p) : (Odd (p^3 + m * p)) ↔ Even m := by
  sorry

end expression_parity_l213_213850


namespace fraction_in_jug_x_after_pouring_water_l213_213761

-- Define capacities and initial fractions
def initial_fraction_x := 1 / 4
def initial_fraction_y := 2 / 3
def fill_needed_y := 1 - initial_fraction_y -- 1/3

-- Define capacity of original jugs
variable (C : ℚ) -- We can assume capacities are rational for simplicity

-- Define initial water amounts in jugs x and y
def initial_water_x := initial_fraction_x * C
def initial_water_y := initial_fraction_y * C

-- Define the water needed to fill jug y
def additional_water_needed_y := fill_needed_y * C

-- Define the final fraction of water in jug x
def final_fraction_x := initial_fraction_x / 2 -- since half of the initial water is poured out

theorem fraction_in_jug_x_after_pouring_water :
  final_fraction_x = 1 / 8 := by
  sorry

end fraction_in_jug_x_after_pouring_water_l213_213761


namespace factorize_expression_l213_213379

-- Lean 4 statement for the proof problem
theorem factorize_expression (a b : ℝ) : ab^2 - a = a * (b + 1) * (b - 1) :=
sorry

end factorize_expression_l213_213379


namespace trigonometric_identity_l213_213251

theorem trigonometric_identity 
  (x : ℝ) 
  (h : Real.sin (x + Real.pi / 3) = 1 / 3) :
  Real.sin (5 * Real.pi / 3 - x) - Real.cos (2 * x - Real.pi / 3) = 4 / 9 :=
by
  sorry

end trigonometric_identity_l213_213251


namespace brian_stones_l213_213505

variable (W B : ℕ)
variable (total_stones : ℕ := 100)
variable (G : ℕ := 40)
variable (Gr : ℕ := 60)

theorem brian_stones :
  (W > B) →
  ((W + B = total_stones) ∧ (G + Gr = total_stones) ∧ (W = 60)) :=
by
  sorry

end brian_stones_l213_213505


namespace equal_distances_sum_of_distances_moving_distances_equal_l213_213263

-- Define the points A, B, origin O, and moving point P
def A : ℝ := -1
def B : ℝ := 3
def O : ℝ := 0

-- Define the moving point P
def P (x : ℝ) : ℝ := x

-- Define the velocities of each point
def vP : ℝ := -1
def vA : ℝ := -5
def vB : ℝ := -20

-- Proof statement ①: Distance from P to A and B are equal implies x = 1
theorem equal_distances (x : ℝ) (h : abs (x + 1) = abs (x - 3)) : x = 1 :=
sorry

-- Proof statement ②: Sum of distances from P to A and B is 5 implies x = -3/2 or 7/2
theorem sum_of_distances (x : ℝ) (h : abs (x + 1) + abs (x - 3) = 5) : x = -3/2 ∨ x = 7/2 :=
sorry

-- Proof statement ③: Moving distances equal at times t = 4/15 or 2/23
theorem moving_distances_equal (t : ℝ) (h : abs (4 * t + 1) = abs (19 * t - 3)) : t = 4/15 ∨ t = 2/23 :=
sorry

end equal_distances_sum_of_distances_moving_distances_equal_l213_213263


namespace find_x_l213_213171

theorem find_x (x : ℕ) (hcf lcm : ℕ):
  (hcf = Nat.gcd x 18) → 
  (lcm = Nat.lcm x 18) → 
  (lcm - hcf = 120) → 
  x = 42 := 
by
  sorry

end find_x_l213_213171


namespace prove_a2_b2_c2_zero_l213_213711

theorem prove_a2_b2_c2_zero (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) (h5 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) : a^2 + b^2 + c^2 = 0 := 
by 
  sorry

end prove_a2_b2_c2_zero_l213_213711


namespace parallelogram_side_lengths_l213_213178

theorem parallelogram_side_lengths (x y : ℝ) (h1 : 3 * x + 6 = 12) (h2 : 5 * y - 2 = 10) : x + y = 22 / 5 :=
by
  sorry

end parallelogram_side_lengths_l213_213178


namespace not_exists_odd_product_sum_l213_213608

theorem not_exists_odd_product_sum (a b : ℤ) : ¬ (a * b * (a + b) = 20182017) :=
sorry

end not_exists_odd_product_sum_l213_213608


namespace fg_of_3_is_83_l213_213553

def g (x : ℕ) : ℕ := x ^ 3
def f (x : ℕ) : ℕ := 3 * x + 2
theorem fg_of_3_is_83 : f (g 3) = 83 := by
  sorry

end fg_of_3_is_83_l213_213553


namespace inheritance_amount_l213_213569

-- Definitions based on conditions given
def inheritance (x : ℝ) : Prop :=
  let federal_tax := 0.25 * x
  let remaining_after_federal := x - federal_tax
  let state_tax := 0.15 * remaining_after_federal
  let total_tax := federal_tax + state_tax
  total_tax = 15000

-- The statement to be proven
theorem inheritance_amount (x : ℝ) (hx : inheritance x) : x = 41379 :=
by
  -- Proof goes here
  sorry

end inheritance_amount_l213_213569


namespace find_divisor_l213_213667

theorem find_divisor (d : ℕ) (n : ℕ) (least : ℕ)
  (h1 : least = 2)
  (h2 : n = 433124)
  (h3 : ∀ d : ℕ, (d ∣ (n + least)) → d = 2) :
  d = 2 := 
sorry

end find_divisor_l213_213667


namespace base5_div_l213_213135

-- Definitions for base 5 numbers
def n1 : ℕ := (2 * 125) + (4 * 25) + (3 * 5) + 4  -- 2434_5 in base 10 is 369
def n2 : ℕ := (1 * 25) + (3 * 5) + 2              -- 132_5 in base 10 is 42
def d  : ℕ := (2 * 5) + 1                          -- 21_5 in base 10 is 11

theorem base5_div (res : ℕ) : res = (122 : ℕ) → (n1 + n2) / d = res :=
by sorry

end base5_div_l213_213135


namespace tips_fraction_l213_213211

theorem tips_fraction (S T : ℝ) (h : T / (S + T) = 0.6363636363636364) : T / S = 1.75 :=
sorry

end tips_fraction_l213_213211


namespace total_cats_and_kittens_received_l213_213991

theorem total_cats_and_kittens_received 
  (adult_cats : ℕ) 
  (perc_female : ℕ) 
  (frac_litters : ℚ) 
  (kittens_per_litter : ℕ)
  (rescued_cats : ℕ) 
  (total_received : ℕ)
  (h1 : adult_cats = 120)
  (h2 : perc_female = 60)
  (h3 : frac_litters = 2/3)
  (h4 : kittens_per_litter = 3)
  (h5 : rescued_cats = 30)
  (h6 : total_received = 294) :
  adult_cats + rescued_cats + (frac_litters * (perc_female * adult_cats / 100) * kittens_per_litter) = total_received := 
sorry

end total_cats_and_kittens_received_l213_213991


namespace length_of_larger_cuboid_l213_213203

theorem length_of_larger_cuboid
  (n : ℕ)
  (l_small : ℝ) (w_small : ℝ) (h_small : ℝ)
  (w_large : ℝ) (h_large : ℝ)
  (V_large : ℝ)
  (n_eq : n = 56)
  (dim_small : l_small = 5 ∧ w_small = 3 ∧ h_small = 2)
  (dim_large : w_large = 14 ∧ h_large = 10)
  (V_large_eq : V_large = n * (l_small * w_small * h_small)) :
  ∃ l_large : ℝ, l_large = V_large / (w_large * h_large) ∧ l_large = 12 := by
  sorry

end length_of_larger_cuboid_l213_213203


namespace math_problem_l213_213012

theorem math_problem (A B C : ℕ) (h_pos : A > 0 ∧ B > 0 ∧ C > 0) (h_gcd : Nat.gcd (Nat.gcd A B) C = 1) (h_eq : A * Real.log 5 / Real.log 200 + B * Real.log 2 / Real.log 200 = C) : A + B + C = 6 :=
sorry

end math_problem_l213_213012


namespace quadratic_residue_iff_l213_213996

open Nat

theorem quadratic_residue_iff (p : ℕ) [Fact (Nat.Prime p)] (hp : p % 2 = 1) (n : ℤ) (hn : n % p ≠ 0) :
  (∃ a : ℤ, (a^2) % p = n % p) ↔ (n ^ ((p - 1) / 2)) % p = 1 :=
sorry

end quadratic_residue_iff_l213_213996


namespace sum_of_eight_numbers_l213_213972

theorem sum_of_eight_numbers (avg : ℝ) (S : ℝ) (h1 : avg = 5.3) (h2 : avg = S / 8) : S = 42.4 :=
by
  sorry

end sum_of_eight_numbers_l213_213972


namespace length_of_platform_l213_213632

variable (L : ℕ)

theorem length_of_platform
  (train_length : ℕ)
  (time_cross_post : ℕ)
  (time_cross_platform : ℕ)
  (train_length_eq : train_length = 300)
  (time_cross_post_eq : time_cross_post = 18)
  (time_cross_platform_eq : time_cross_platform = 39)
  : L = 350 := sorry

end length_of_platform_l213_213632


namespace polynomial_sum_of_squares_is_23456_l213_213128

theorem polynomial_sum_of_squares_is_23456 (p q r s t u : ℤ) :
  (∀ x, 1728 * x ^ 3 + 64 = (p * x ^ 2 + q * x + r) * (s * x ^ 2 + t * x + u)) →
  p ^ 2 + q ^ 2 + r ^ 2 + s ^ 2 + t ^ 2 + u ^ 2 = 23456 :=
by
  sorry

end polynomial_sum_of_squares_is_23456_l213_213128


namespace find_square_sum_l213_213880

theorem find_square_sum :
  ∃ a b c : ℕ, a = 2494651 ∧ b = 1385287 ∧ c = 9406087 ∧ (a + b + c = 3645^2) :=
by
  have h1 : 2494651 + 1385287 + 9406087 = 13286025 := by norm_num
  have h2 : 3645^2 = 13286025 := by norm_num
  exact ⟨2494651, 1385287, 9406087, rfl, rfl, rfl, h2⟩

end find_square_sum_l213_213880


namespace problem_solution_l213_213843

open Set

def A : Set ℕ := {x | x > 0 ∧ x < 9}
def B : Set ℕ := {1, 2, 3}
def C : Set ℕ := {3, 4, 5, 6}

theorem problem_solution :
  A ∩ B = {1, 2, 3} ∧
  A ∩ C = {3, 4, 5, 6} ∧
  A ∩ (B ∪ C) = {1, 2, 3, 4, 5, 6} ∧
  A ∪ (B ∩ C) = {1, 2, 3, 4, 5, 6, 7, 8} :=
by
  sorry

end problem_solution_l213_213843


namespace series_convergence_l213_213651

theorem series_convergence :
  ∑ n : ℕ, (2^n) / (3^(2^n) + 1) = 1 / 2 :=
by
  sorry

end series_convergence_l213_213651


namespace log_identity_l213_213642

noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

theorem log_identity : log 2 5 * log 3 2 * log 5 3 = 1 :=
by sorry

end log_identity_l213_213642


namespace player_A_wins_after_4_rounds_geometric_sequence_differences_find_P_2_l213_213891

-- Define probabilities of shots
def shooting_probability_A : ℝ := 0.5
def shooting_probability_B : ℝ := 0.6

-- Define initial points for questions
def initial_points_question_1 : ℝ := 0
def initial_points_question_2 : ℝ := 2

-- Given initial probabilities
def P_0 : ℝ := 0
def P_4 : ℝ := 1

-- Probability that player A wins after exactly 4 rounds
def probability_A_wins_after_4_rounds : ℝ :=
  let P_A := shooting_probability_A * (1 - shooting_probability_B)
  let P_B := shooting_probability_B * (1 - shooting_probability_A)
  let P_C := 1 - P_A - P_B
  P_A * P_C^2 * P_A + P_A * P_B * P_A^2

-- Define the probabilities P(i) for i=0..4
def P (i : ℕ) : ℝ := sorry -- Placeholder for the function

-- Define the proof problem
theorem player_A_wins_after_4_rounds : probability_A_wins_after_4_rounds = 0.0348 :=
sorry

theorem geometric_sequence_differences :
  ∀ i : ℕ, i < 4 → (P (i + 1) - P i) / (P (i + 2) - P (i + 1)) = 2/3 :=
sorry

theorem find_P_2 : P 2 = 4/13 :=
sorry

end player_A_wins_after_4_rounds_geometric_sequence_differences_find_P_2_l213_213891


namespace heptagon_diagonals_l213_213111

theorem heptagon_diagonals : (7 * (7 - 3)) / 2 = 14 := 
by
  rfl

end heptagon_diagonals_l213_213111


namespace number_of_candies_in_a_packet_l213_213224

theorem number_of_candies_in_a_packet 
  (two_packets : ℕ)
  (candies_per_day_weekday : ℕ)
  (candies_per_day_weekend : ℕ)
  (weeks : ℕ)
  (total_candies : ℕ)
  (packet_size : ℕ)
  (H1 : two_packets = 2)
  (H2 : candies_per_day_weekday = 2)
  (H3 : candies_per_day_weekend = 1)
  (H4 : weeks = 3)
  (H5 : packet_size > 0)
  (H6 : total_candies = packets * packet_size)
  (H7 : total_candies = 3 * (5 * candies_per_day_weekday + 2 * candies_per_day_weekend))
  : packet_size = 18 :=
by
  sorry

end number_of_candies_in_a_packet_l213_213224


namespace prove_a2_b2_c2_zero_l213_213709

theorem prove_a2_b2_c2_zero (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) (h5 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) : a^2 + b^2 + c^2 = 0 := 
by 
  sorry

end prove_a2_b2_c2_zero_l213_213709


namespace candy_bars_per_bag_l213_213161

/-
Define the total number of candy bars and the number of bags
-/
def totalCandyBars : ℕ := 75
def numberOfBags : ℚ := 15.0

/-
Prove that the number of candy bars per bag is 5
-/
theorem candy_bars_per_bag : totalCandyBars / numberOfBags = 5 := by
  sorry

end candy_bars_per_bag_l213_213161


namespace almond_butter_servings_l213_213487

def convert_mixed_to_fraction (a b : ℤ) (n : ℕ) : ℚ :=
  (a * n + b) / n

def servings (total servings_fraction : ℚ) : ℚ :=
  total / servings_fraction

theorem almond_butter_servings :
  servings (convert_mixed_to_fraction 35 2 3) (convert_mixed_to_fraction 2 1 2) = 14 + 4 / 15 :=
by
  sorry

end almond_butter_servings_l213_213487


namespace mowing_difference_l213_213291

-- Define the number of times mowed in spring and summer
def mowedSpring : ℕ := 8
def mowedSummer : ℕ := 5

-- Prove the difference between spring and summer mowing is 3
theorem mowing_difference : mowedSpring - mowedSummer = 3 := by
  sorry

end mowing_difference_l213_213291


namespace isosceles_obtuse_triangle_smallest_angle_l213_213776

theorem isosceles_obtuse_triangle_smallest_angle :
  ∀ (α β γ : ℝ), α = 1.8 * 90 ∧ β = γ ∧ α + β + γ = 180 → β = 9 :=
by
  intros α β γ h
  sorry

end isosceles_obtuse_triangle_smallest_angle_l213_213776


namespace reciprocal_of_neg_two_l213_213461

theorem reciprocal_of_neg_two : ∀ x : ℝ, x = -2 → (1 / x) = -1 / 2 :=
by
  intro x h
  rw [h]
  norm_num

end reciprocal_of_neg_two_l213_213461


namespace identify_value_of_expression_l213_213397

theorem identify_value_of_expression (x y z : ℝ)
  (h1 : y / (x - y) = x / (y + z))
  (h2 : z^2 = x * (y + z) - y * (x - y)) :
  (y^2 + z^2 - x^2) / (2 * y * z) = 1 / 2 := 
sorry

end identify_value_of_expression_l213_213397


namespace arc_length_of_curve_l213_213041

noncomputable def f (x : ℝ) : ℝ :=
  sqrt (x - x^2) - Real.arccos (sqrt x) + 5

theorem arc_length_of_curve :
  ∫ x in (1 / 9 : ℝ)..1, sqrt (1 + (Real.sqrt ((1 - x) / x)) ^ 2) = (4 : ℝ) / 3 :=
by sorry

end arc_length_of_curve_l213_213041


namespace initial_salty_cookies_l213_213863

theorem initial_salty_cookies (sweet_init sweet_eaten sweet_left salty_eaten : ℕ) 
  (h1 : sweet_init = 34)
  (h2 : sweet_eaten = 15)
  (h3 : sweet_left = 19)
  (h4 : salty_eaten = 56) :
  sweet_left + sweet_eaten = sweet_init → 
  sweet_init - sweet_eaten = sweet_left →
  ∃ salty_init, salty_init = salty_eaten :=
by
  sorry

end initial_salty_cookies_l213_213863


namespace time_between_peanuts_l213_213018

def peanuts_per_bag : ℕ := 30
def number_of_bags : ℕ := 4
def flight_time_hours : ℕ := 2

theorem time_between_peanuts (peanuts_per_bag number_of_bags flight_time_hours : ℕ) (h1 : peanuts_per_bag = 30) (h2 : number_of_bags = 4) (h3 : flight_time_hours = 2) :
  (flight_time_hours * 60) / (peanuts_per_bag * number_of_bags) = 1 := by
  sorry

end time_between_peanuts_l213_213018


namespace sum_of_eight_numbers_l213_213971

theorem sum_of_eight_numbers (avg : ℝ) (S : ℝ) (h1 : avg = 5.3) (h2 : avg = S / 8) : S = 42.4 :=
by
  sorry

end sum_of_eight_numbers_l213_213971


namespace total_exercise_time_l213_213840

-- Definition of constants and speeds for each day
def monday_speed := 2 -- miles per hour
def wednesday_speed := 3 -- miles per hour
def friday_speed := 6 -- miles per hour
def distance := 6 -- miles

-- Function to calculate time given distance and speed
def time (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

-- Prove the total time spent in a week
theorem total_exercise_time :
  time distance monday_speed + time distance wednesday_speed + time distance friday_speed = 6 :=
by
  -- Insert detailed proof steps here
  sorry

end total_exercise_time_l213_213840


namespace remaining_sessions_l213_213357

theorem remaining_sessions (total_sessions : ℕ) (p1_sessions : ℕ) (p2_sessions_more : ℕ) (remaining_sessions : ℕ) :
  total_sessions = 25 →
  p1_sessions = 6 →
  p2_sessions_more = 5 →
  remaining_sessions = total_sessions - (p1_sessions + (p1_sessions + p2_sessions_more)) →
  remaining_sessions = 8 :=
by
  intros
  sorry

end remaining_sessions_l213_213357


namespace annual_income_is_correct_l213_213207

noncomputable def total_investment : ℝ := 4455
noncomputable def price_per_share : ℝ := 8.25
noncomputable def dividend_rate : ℝ := 12 / 100
noncomputable def face_value : ℝ := 10

noncomputable def number_of_shares : ℝ := total_investment / price_per_share
noncomputable def dividend_per_share : ℝ := dividend_rate * face_value
noncomputable def annual_income : ℝ := dividend_per_share * number_of_shares

theorem annual_income_is_correct : annual_income = 648 := by
  sorry

end annual_income_is_correct_l213_213207


namespace radius_of_circle_l213_213607

theorem radius_of_circle (x y : ℝ) : (x^2 + y^2 - 8*x = 0) → (∃ r, r = 4) :=
by
  intro h
  sorry

end radius_of_circle_l213_213607


namespace horner_method_complexity_l213_213019

variable {α : Type*} [Field α]

/-- Evaluating a polynomial of degree n using Horner's method requires exactly n multiplications
    and n additions, and 0 exponentiations.  -/
theorem horner_method_complexity (n : ℕ) (a : Fin (n + 1) → α) (x₀ : α) :
  ∃ (muls adds exps : ℕ), 
    (muls = n) ∧ (adds = n) ∧ (exps = 0) :=
by
  sorry

end horner_method_complexity_l213_213019


namespace balls_into_boxes_l213_213828

theorem balls_into_boxes :
  (number_of_ways_to_distribute 7 4) = 128 := sorry

end balls_into_boxes_l213_213828


namespace calc_fraction_cube_l213_213784

theorem calc_fraction_cube : (88888 ^ 3 / 22222 ^ 3) = 64 := by 
    sorry

end calc_fraction_cube_l213_213784


namespace max_minute_hands_l213_213475

theorem max_minute_hands (m n : ℕ) (h1 : m * n = 27) : m + n ≤ 28 :=
by sorry

end max_minute_hands_l213_213475


namespace megatek_manufacturing_percentage_l213_213162

theorem megatek_manufacturing_percentage (total_degrees sector_degrees : ℝ)
    (h_circle: total_degrees = 360)
    (h_sector: sector_degrees = 252) :
    (sector_degrees / total_degrees) * 100 = 70 :=
by
  sorry

end megatek_manufacturing_percentage_l213_213162


namespace diophantine_solution_range_l213_213722

theorem diophantine_solution_range {a b c n : ℤ} (coprime_ab : Int.gcd a b = 1) :
  (∃ (x y : ℕ), a * x + b * y = c ∧ ∀ k : ℤ, k ≥ 1 → ∃ (x y : ℕ), a * (x + k * b) + b * (y - k * a) = c) → 
  ((n - 1) * a * b + a + b ≤ c ∧ c ≤ (n + 1) * a * b) :=
sorry

end diophantine_solution_range_l213_213722


namespace teal_more_blue_l213_213047

def numSurveyed : ℕ := 150
def numGreen : ℕ := 90
def numBlue : ℕ := 50
def numBoth : ℕ := 40
def numNeither : ℕ := 20

theorem teal_more_blue : 40 + (numSurveyed - (numBoth + (numGreen - numBoth) + numNeither)) = 80 :=
by
  -- Here we simplify numerically until we get the required answer
  -- start with calculating the total accounted and remaining
  sorry

end teal_more_blue_l213_213047


namespace khali_total_snow_volume_l213_213841

def length1 : ℝ := 25
def width1 : ℝ := 3
def depth1 : ℝ := 0.75

def length2 : ℝ := 15
def width2 : ℝ := 3
def depth2 : ℝ := 1

def volume1 : ℝ := length1 * width1 * depth1
def volume2 : ℝ := length2 * width2 * depth2
def total_volume : ℝ := volume1 + volume2

theorem khali_total_snow_volume : total_volume = 101.25 := by
  sorry

end khali_total_snow_volume_l213_213841


namespace smallest_number_is_3_l213_213182

theorem smallest_number_is_3 (a b c : ℝ) (h1 : (a + b + c) / 3 = 7) (h2 : a = 9 ∨ b = 9 ∨ c = 9) : min (min a b) c = 3 := 
sorry

end smallest_number_is_3_l213_213182


namespace interval1_increase_decrease_interval2_increasing_interval3_increase_decrease_l213_213071

section
open Real

noncomputable def interval1 (x : ℝ) : Real := log (1 - x ^ 2)
noncomputable def interval2 (x : ℝ) : Real := x * (1 + 2 * sqrt x)
noncomputable def interval3 (x : ℝ) : Real := log (abs x)

-- Function 1: p = ln(1 - x^2)
theorem interval1_increase_decrease :
  (∀ x : ℝ, -1 < x ∧ x < 0 → deriv interval1 x > 0) ∧
  (∀ x : ℝ, 0 < x ∧ x < 1 → deriv interval1 x < 0) := by
  sorry

-- Function 2: z = x(1 + 2√x)
theorem interval2_increasing :
  ∀ x : ℝ, x ≥ 0 → deriv interval2 x > 0 := by
  sorry

-- Function 3: y = ln|x|
theorem interval3_increase_decrease :
  (∀ x : ℝ, x < 0 → deriv interval3 x < 0) ∧
  (∀ x : ℝ, x > 0 → deriv interval3 x > 0) := by
  sorry

end

end interval1_increase_decrease_interval2_increasing_interval3_increase_decrease_l213_213071


namespace value_of_b_l213_213185

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 1) : b = 1 :=
sorry

end value_of_b_l213_213185


namespace alvin_earns_14_dollars_l213_213062

noncomputable def total_earnings (total_marbles : ℕ) (percent_white percent_black : ℚ)
  (price_white price_black price_colored : ℚ) : ℚ :=
  let white_marbles := percent_white * total_marbles
  let black_marbles := percent_black * total_marbles
  let colored_marbles := total_marbles - white_marbles - black_marbles
  (white_marbles * price_white) + (black_marbles * price_black) + (colored_marbles * price_colored)

theorem alvin_earns_14_dollars :
  total_earnings 100 (20/100) (30/100) 0.05 0.10 0.20 = 14 := by
  sorry

end alvin_earns_14_dollars_l213_213062


namespace initial_numbers_conditions_l213_213749

theorem initial_numbers_conditions (a b c : ℤ)
    (h : ∀ (x y z : ℤ), (x, y, z) = (17, 1967, 1983) → 
      x = y + z - 1 ∨ y = x + z - 1 ∨ z = x + y - 1) :
  (a = 2 ∧ b = 2 ∧ c = 2) → false ∧ 
  (a = 3 ∧ b = 3 ∧ c = 3) → true := 
sorry

end initial_numbers_conditions_l213_213749


namespace cube_edge_numbers_possible_l213_213988

theorem cube_edge_numbers_possible :
  ∃ (top bottom : Finset ℕ), top.card = 4 ∧ bottom.card = 4 ∧ 
  (top ∪ bottom).card = 8 ∧ 
  ∀ (n : ℕ), n ∈ top ∪ bottom → n ∈ (Finset.range 12).map (+1) ∧ 
  (∏ x in top, x) = (∏ y in bottom, y) :=
by {
  sorry,
}

end cube_edge_numbers_possible_l213_213988


namespace trapezoid_base_ratio_l213_213164

variable {A B C D K M P Q L N : Type}
variable [LinearOrderedField A]
variable [LinearOrderedField B]
variable [LinearOrderedField C]
variable [LinearOrderedField D]

def is_trapezoid (A B C D : Type) (a b : ℝ) : Prop :=
  is_parallel (AD BC) ∧ AD.length = a ∧ BC.length = b ∧ a > b

def area (A B C D : Type) (a b : ℝ) : ℝ :=
  (1 / 2) * (a + b) * h

def quadrilateral_area (K L M N : Type) (a b h : ℝ) : ℝ :=
  (1 / 4) * (a - b) * h

theorem trapezoid_base_ratio (A B C D K M P Q L N : Type) (a b : ℝ) (h : ℝ) :
  is_trapezoid A B C D a b →
  quadrilateral_area K L M N a b h = (1 / 4) * area A B C D a b ↔ a / b = 3 :=
by 
  sorry

end trapezoid_base_ratio_l213_213164


namespace find_value_of_a2_b2_c2_l213_213713

variable {a b c : ℝ}

theorem find_value_of_a2_b2_c2
  (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h2 : a + b + c = 0)
  (h3 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6 / 5 := 
sorry

end find_value_of_a2_b2_c2_l213_213713


namespace factorize_expression_l213_213380

-- Lean 4 statement for the proof problem
theorem factorize_expression (a b : ℝ) : ab^2 - a = a * (b + 1) * (b - 1) :=
sorry

end factorize_expression_l213_213380


namespace femaleRainbowTroutCount_l213_213132

noncomputable def numFemaleRainbowTrout : ℕ := 
  let numSpeckledTrout := 645
  let numFemaleSpeckled := 200
  let numMaleSpeckled := 445
  let numMaleRainbow := 150
  let totalTrout := 1000
  let numRainbowTrout := totalTrout - numSpeckledTrout
  numRainbowTrout - numMaleRainbow

theorem femaleRainbowTroutCount : numFemaleRainbowTrout = 205 := by
  -- Conditions
  let numSpeckledTrout : ℕ := 645
  let numMaleSpeckled := 2 * 200 + 45
  let totalTrout := 645 + 355
  let numRainbowTrout := totalTrout - numSpeckledTrout
  let numFemaleRainbow := numRainbowTrout - 150
  
  -- The proof would proceed here
  sorry

end femaleRainbowTroutCount_l213_213132


namespace allocation_methods_count_l213_213010

theorem allocation_methods_count :
  let doctors := 3
  let nurses := 6
  let schools := 3
  (doctors = 3 ∧ nurses = 6 ∧ schools = 3) →
  (number_of_ways : ℕ) →
  (number_of_ways = (3 * (nurses.choose 2) * 2 * ((nurses - 2).choose 2) * 1 * ((nurses - 4).choose 2))) →
  number_of_ways = 540 :=
by
  intros doctors nurses schools h number_of_ways h_ways
  have h_doctors : doctors = 3 := by tauto
  have h_nurses : nurses = 6 := by tauto
  have h_schools : schools = 3 := by tauto
  rw [h_doctors, h_nurses, h_schools] at h_ways
  sorry

end allocation_methods_count_l213_213010


namespace simplest_square_root_l213_213898

noncomputable def sqrt8 : ℝ := Real.sqrt 8
noncomputable def inv_sqrt2 : ℝ := 1 / Real.sqrt 2
noncomputable def sqrt2 : ℝ := Real.sqrt 2
noncomputable def sqrt_inv2 : ℝ := Real.sqrt (1 / 2)

theorem simplest_square_root : sqrt2 = Real.sqrt 2 := 
  sorry

end simplest_square_root_l213_213898


namespace solve_system_l213_213457

theorem solve_system (x y z w : ℝ) :
  x - y + z - w = 2 ∧
  x^2 - y^2 + z^2 - w^2 = 6 ∧
  x^3 - y^3 + z^3 - w^3 = 20 ∧
  x^4 - y^4 + z^4 - w^4 = 60 ↔
  (x = 1 ∧ y = 2 ∧ z = 3 ∧ w = 0) ∨
  (x = 1 ∧ y = 0 ∧ z = 3 ∧ w = 2) ∨
  (x = 3 ∧ y = 2 ∧ z = 1 ∧ w = 0) ∨
  (x = 3 ∧ y = 0 ∧ z = 1 ∧ w = 2) :=
sorry

end solve_system_l213_213457


namespace point_between_circles_l213_213257

theorem point_between_circles 
  (a b c x1 x2 : ℝ)
  (ellipse_eq : ∀ x y, (x^2 / a^2) + (y^2 / b^2) = 1)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c > 0)
  (quad_eq : a * x1^2 + b * x1 - c = 0)
  (quad_eq2 : a * x2^2 + b * x2 - c = 0)
  (sum_roots : x1 + x2 = -b / a)
  (prod_roots : x1 * x2 = -c / a) :
  1 < x1^2 + x2^2 ∧ x1^2 + x2^2 < 2 :=
sorry

end point_between_circles_l213_213257


namespace find_positive_A_l213_213434

theorem find_positive_A (A : ℕ) : (A^2 + 7^2 = 130) → A = 9 :=
by
  intro h
  sorry

end find_positive_A_l213_213434


namespace candy_per_packet_l213_213221

-- Define the conditions as hypotheses
def bobby_weekly_candies (mon_to_fri_candies : ℕ) (sat_sun_candies : ℕ) : ℕ :=
  mon_to_fri_candies + sat_sun_candies

def total_candies_in_n_weeks (weekly_candies : ℕ) (n : ℕ) : ℕ :=
  weekly_candies * n

theorem candy_per_packet
  (mon_to_fri_candies_per_day : ℕ)
  (sat_sun_candies_per_day : ℕ)
  (days_mon_to_fri : ℕ)
  (days_weekend : ℕ)
  (num_weeks : ℕ)
  (total_packets : ℕ)
  (candies_per_packet : ℕ)
  (h1 : mon_to_fri_candies_per_day = 2)
  (h2 : sat_sun_candies_per_day = 1)
  (h3 : days_mon_to_fri = 5)
  (h4 : days_weekend = 2)
  (h5 : num_weeks = 3)
  (h6 : total_packets = 2)
  (h7 : candies_per_packet = (total_candies_in_n_weeks (bobby_weekly_candies (mon_to_fri_candies_per_day * days_mon_to_fri) (sat_sun_candies_per_day * days_weekend)) num_weeks) / total_packets) :
  candies_per_packet = 18 :=
sorry

end candy_per_packet_l213_213221


namespace series_sum_l213_213655

theorem series_sum : 
  (∑ n : ℕ, (2^n) / (3^(2^n) + 1)) = 1 / 2 :=
by
  sorry

end series_sum_l213_213655


namespace find_z_value_l213_213100

-- We will define the variables and the given condition
variables {x y z : ℝ}

-- Translate the given condition into Lean
def given_condition (x y z : ℝ) : Prop := (1 / x^2 - 1 / y^2) = (1 / z)

-- State the theorem to prove
theorem find_z_value (x y z : ℝ) (h : given_condition x y z) : 
  z = (x^2 * y^2) / (y^2 - x^2) :=
sorry

end find_z_value_l213_213100


namespace ratio_of_amounts_l213_213156

theorem ratio_of_amounts
    (initial_cents : ℕ)
    (given_to_peter_cents : ℕ)
    (remaining_nickels : ℕ)
    (nickel_value : ℕ := 5)
    (nickels_initial := initial_cents / nickel_value)
    (nickels_to_peter := given_to_peter_cents / nickel_value)
    (nickels_remaining := nickels_initial - nickels_to_peter)
    (nickels_given_to_randi := nickels_remaining - remaining_nickels)
    (cents_to_randi := nickels_given_to_randi * nickel_value)
    (cents_initial : initial_cents = 95)
    (cents_peter : given_to_peter_cents = 25)
    (nickels_left : remaining_nickels = 4)
    :
    (cents_to_randi / given_to_peter_cents) = 2 :=
by
  sorry

end ratio_of_amounts_l213_213156


namespace fg_value_l213_213269

def g (x : ℤ) : ℤ := 4 * x - 3
def f (x : ℤ) : ℤ := 6 * x + 2

theorem fg_value : f (g 5) = 104 := by
  sorry

end fg_value_l213_213269


namespace integer_solutions_to_cube_sum_eq_2_pow_30_l213_213928

theorem integer_solutions_to_cube_sum_eq_2_pow_30 (x y : ℤ) :
  x^3 + y^3 = 2^30 → (x = 0 ∧ y = 2^10) ∨ (x = 2^10 ∧ y = 0) :=
by
  sorry

end integer_solutions_to_cube_sum_eq_2_pow_30_l213_213928


namespace evaluate_expression_l213_213812

theorem evaluate_expression (a b c : ℝ)
  (h : a / (25 - a) + b / (65 - b) + c / (60 - c) = 7) :
  5 / (25 - a) + 13 / (65 - b) + 12 / (60 - c) = 2 := 
sorry

end evaluate_expression_l213_213812


namespace fraction_lost_down_sewer_l213_213366

-- Definitions of the conditions derived from the problem
def initial_marbles := 100
def street_loss_percent := 60 / 100
def sewer_loss := 40 - 20
def remaining_marbles_after_street := initial_marbles - (initial_marbles * street_loss_percent)
def marbles_left := 20

-- The theorem statement proving the fraction of remaining marbles lost down the sewer
theorem fraction_lost_down_sewer :
  (sewer_loss / remaining_marbles_after_street) = 1 / 2 :=
by
  sorry

end fraction_lost_down_sewer_l213_213366


namespace inequality_solution_l213_213658

theorem inequality_solution {x : ℝ} : 5 * x^2 + 7 * x > 3 ↔ x < -1 ∨ x > 3/5 := by
  sorry

end inequality_solution_l213_213658


namespace number_of_classes_l213_213350

variable (s : ℕ) (h_s : s > 0)
-- Define the conditions
def student_books_year : ℕ := 4 * 12
def total_books_read : ℕ := 48
def class_books_year (s : ℕ) : ℕ := s * student_books_year
def total_classes (c s : ℕ) (h_s : s > 0) : ℕ := 1

-- Define the main theorem
theorem number_of_classes (h : total_books_read = 48) (h_s : s > 0)
  (h1 : c * class_books_year s = 48) : c = 1 := by
  sorry

end number_of_classes_l213_213350


namespace square_of_volume_of_rect_box_l213_213816

theorem square_of_volume_of_rect_box (x y z : ℝ) 
  (h1 : x * y = 15) 
  (h2 : y * z = 18) 
  (h3 : z * x = 10) : (x * y * z) ^ 2 = 2700 :=
sorry

end square_of_volume_of_rect_box_l213_213816


namespace corresponding_angles_equal_l213_213588

variable {α : Type}
variables (A B : α) [angle : has_measure α (angle_measure α)]
variables (h : A.is_corresponding_with B)

theorem corresponding_angles_equal (A B : α) [angle A] [angle B] :
  A.is_corresponding_with B → A = B :=
by
  sorry

end corresponding_angles_equal_l213_213588


namespace circle_radius_l213_213344

theorem circle_radius (D : ℝ) (h : D = 14) : (D / 2) = 7 :=
by
  sorry

end circle_radius_l213_213344


namespace arrangement_count_l213_213773

/-- Among 7 workers, 5 can do typesetting and 4 can do printing.
    Prove that the number of ways to arrange 2 people for typesetting and 2 for printing is 37. -/
theorem arrangement_count : 
  let T := 5 -- number of workers who can do typesetting
  let P := 4 -- number of workers who can do printing
  let N := 7 -- total number of workers
  ( T + P - N = 2 /\ ∑ i in Finset.range 3, (Nat.choose 3 i) * (Nat.choose 4 (2 - i)) = 37) 
  :=
by
  sorry

end arrangement_count_l213_213773


namespace ab_value_l213_213820

noncomputable def func (x : ℝ) (a b : ℝ) : ℝ := 4 * x ^ 3 - a * x ^ 2 - 2 * b * x + 2

theorem ab_value 
  (a b : ℝ)
  (h_max : func 1 a b = -3)
  (h_deriv : (12 - 2 * a - 2 * b) = 0) :
  a * b = 9 :=
by
  sorry

end ab_value_l213_213820


namespace find_f1_plus_g1_l213_213395

variable (f g : ℝ → ℝ)

-- Conditions
def even_function (h : ℝ → ℝ) := ∀ x : ℝ, h x = h (-x)
def odd_function (h : ℝ → ℝ) := ∀ x : ℝ, h x = -h (-x)
def function_relation := ∀ x : ℝ, f x - g x = x^3 + x^2 + 1

-- Mathematically equivalent proof problem
theorem find_f1_plus_g1
  (hf_even : even_function f)
  (hg_odd : odd_function g)
  (h_relation : function_relation f g) :
  f 1 + g 1 = 1 := by
  sorry

end find_f1_plus_g1_l213_213395


namespace football_outcomes_l213_213465

theorem football_outcomes : 
  ∃ (W D L : ℕ), (3 * W + D = 19) ∧ (W + D + L = 14) ∧ 
  ((W = 3 ∧ D = 10 ∧ L = 1) ∨ 
   (W = 4 ∧ D = 7 ∧ L = 3) ∨ 
   (W = 5 ∧ D = 4 ∧ L = 5) ∨ 
   (W = 6 ∧ D = 1 ∧ L = 7)) ∧
  (∀ W' D' L' : ℕ, (3 * W' + D' = 19) → (W' + D' + L' = 14) → 
    (W' = 3 ∧ D' = 10 ∧ L' = 1) ∨ 
    (W' = 4 ∧ D' = 7 ∧ L' = 3) ∨ 
    (W' = 5 ∧ D' = 4 ∧ L' = 5) ∨ 
    (W' = 6 ∧ D' = 1 ∧ L' = 7)) := 
sorry

end football_outcomes_l213_213465


namespace quotient_of_division_is_123_l213_213241

theorem quotient_of_division_is_123 :
  let d := 62976
  let v := 512
  d / v = 123 := by
  sorry

end quotient_of_division_is_123_l213_213241


namespace spending_difference_l213_213370

-- Define the cost of the candy bar
def candy_bar_cost : ℕ := 6

-- Define the cost of the chocolate
def chocolate_cost : ℕ := 3

-- Prove the difference between candy_bar_cost and chocolate_cost
theorem spending_difference : candy_bar_cost - chocolate_cost = 3 :=
by
    sorry

end spending_difference_l213_213370


namespace noah_yearly_call_cost_l213_213860

structure CallBilling (minutes_per_call : ℕ) (charge_per_minute : ℝ) (calls_per_week : ℕ) (weeks_in_year : ℕ) :=
  (total_minutes : ℕ := weeks_in_year * calls_per_week * minutes_per_call)
  (total_cost : ℝ := total_minutes * charge_per_minute)

theorem noah_yearly_call_cost :
  CallBilling 30 0.05 1 52 .total_cost = 78 := by
  sorry

end noah_yearly_call_cost_l213_213860


namespace cost_price_of_article_l213_213910

-- Define the conditions
variable (C : ℝ) -- Cost price of the article
variable (SP : ℝ) -- Selling price of the article

-- Conditions according to the problem
def condition1 : Prop := SP = 0.75 * C
def condition2 : Prop := SP + 500 = 1.15 * C

-- The theorem to prove the cost price
theorem cost_price_of_article (h₁ : condition1 C SP) (h₂ : condition2 C SP) : C = 1250 :=
by
  sorry

end cost_price_of_article_l213_213910


namespace power_of_two_l213_213896

theorem power_of_two (n : ℕ) (h : 2^n = 32 * (1 / 2) ^ 2) : n = 3 :=
by {
  sorry
}

end power_of_two_l213_213896


namespace problem_solution_l213_213007

theorem problem_solution (x y : ℝ) (h1 : x + y = 500) (h2 : x / y = 0.8) : y - x = 500 / 9 :=
by
  sorry

end problem_solution_l213_213007


namespace number_of_pairs_sold_l213_213037

-- Define the conditions
def total_amount_made : ℝ := 588
def average_price_per_pair : ℝ := 9.8

-- The theorem we want to prove
theorem number_of_pairs_sold : total_amount_made / average_price_per_pair = 60 := 
by sorry

end number_of_pairs_sold_l213_213037


namespace no_prime_divisible_by_56_l213_213123

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def divisible_by_56 (n : ℕ) : Prop := 56 ∣ n

theorem no_prime_divisible_by_56 : ¬ ∃ p, is_prime p ∧ divisible_by_56 p := 
  sorry

end no_prime_divisible_by_56_l213_213123


namespace average_speed_of_train_l213_213911

theorem average_speed_of_train (x : ℝ) (h1 : 0 < x) : 
  let Time1 := x / 40
  let Time2 := x / 10
  let TotalDistance := 3 * x
  let TotalTime := x / 8
  (TotalDistance / TotalTime = 24) :=
by
  sorry

end average_speed_of_train_l213_213911


namespace div_n_by_8_eq_2_8089_l213_213998

theorem div_n_by_8_eq_2_8089
  (n : ℕ)
  (h : n = 16^2023) :
  n / 8 = 2^8089 := by
  sorry

end div_n_by_8_eq_2_8089_l213_213998


namespace sin_theta_in_terms_of_x_l213_213297

-- Defining the conditions in Lean 4

variables {x θ : ℝ}
-- Assume θ is an acute angle
variable (hθ : 0 < θ ∧ θ < π / 2)
-- Assume the given cosine half-angle expression
variable (hcos_half : cos (θ / 2) = real.sqrt ((x + 1) / (2 * x)))

-- The proposition to prove
theorem sin_theta_in_terms_of_x :
  sin θ = real.sqrt (x^2 - 1) / x :=
sorry

end sin_theta_in_terms_of_x_l213_213297


namespace projection_not_convex_pentagon_l213_213984

noncomputable def set_of_points : set (ℝ × ℝ × ℝ) :=
  { p | ∃ (x y z : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ 
                       0 ≤ y ∧ y ≤ 1 ∧ 
                       0 ≤ z ∧ z ≤ 1 ∧ 
                       p = (x, y, z) }

def is_projection_convex_pentagon (proj : set (ℝ × ℝ)) : Prop := 
  ∃ (vertices : list (ℝ × ℝ)), vertices.length = 5 ∧ 
    (∀ (v1 v2 : ℝ × ℝ), v1 ∈ vertices ∧ v2 ∈ vertices ∧ v1 ≠ v2 → 
                        ∀ (λ : ℝ), 0 < λ ∧ λ < 1 → 
                        λ • v1 + (1 - λ) • v2 ∉ vertices) ∧
    (∀ (v : ℝ × ℝ), v ∈ proj → ∃ (λi : ℝ) (i : Fin 5), 0 < λi ∧ 
                                                    λi < 1 ∧ 
                                                    v = λi • (vertices.nth_le i (by sorry)) + 
                                                        (1 - λi) • (vertices.nth_le ((i + 1) % 5) (by sorry)))

theorem projection_not_convex_pentagon (proj : set (ℝ × ℝ)) :
  (∃ (P : set_of_points → set (ℝ × ℝ)), ∀ (p : ℝ × ℝ × ℝ) (_ : p ∈ set_of_points), P p = proj) →
  ¬ is_projection_convex_pentagon proj :=
by
  sorry

end projection_not_convex_pentagon_l213_213984


namespace sum_of_numbers_l213_213966

theorem sum_of_numbers (a : ℝ) (n : ℕ) (h : a = 5.3) (hn : n = 8) : (a * n) = 42.4 :=
sorry

end sum_of_numbers_l213_213966


namespace dinosaur_dolls_distribution_l213_213332

-- Defining the conditions
def num_dolls : ℕ := 5
def num_friends : ℕ := 2

-- Lean theorem statement
theorem dinosaur_dolls_distribution :
  (num_dolls * (num_dolls - 1) = 20) :=
by
  -- Sorry placeholder for the proof
  sorry

end dinosaur_dolls_distribution_l213_213332


namespace line_canonical_form_l213_213029

theorem line_canonical_form :
  ∃ (x y z : ℝ),
  x + y + z - 2 = 0 ∧
  x - y - 2 * z + 2 = 0 →
  ∃ (k : ℝ),
  x / k = -1 ∧
  (y - 2) / (3 * k) = 1 ∧
  z / (-2 * k) = 1 :=
sorry

end line_canonical_form_l213_213029


namespace admission_given_written_test_passed_l213_213872

variable {Ω : Type*} [MeasurableSpace Ω] {P : MeasureTheory.ProbabilityMeasure Ω}
variables (A B : Set Ω) (hA : P A = 0.2) (hC : P (A ∩ B) = 0.04)

theorem admission_given_written_test_passed :
  P[B | A] = 0.2 :=
by
  have h_cond : P[A ∩ B] = P[A] * P[B | A], from MeasureTheory.probability_def.cond_prob,
  have h_mul : 0.04 = 0.2 * P[B | A], from by rw [hC, hA, h_cond],
  linarith

end admission_given_written_test_passed_l213_213872


namespace weight_of_second_square_l213_213360

noncomputable def weight_of_square (side_length : ℝ) (density : ℝ) : ℝ :=
  side_length^2 * density

theorem weight_of_second_square :
  let s1 := 4
  let m1 := 20
  let s2 := 7
  let density := m1 / (s1 ^ 2)
  ∃ (m2 : ℝ), m2 = 61.25 :=
by
  have s1 := 4
  have m1 := 20
  have s2 := 7
  let density := m1 / (s1 ^ 2)
  have m2 := weight_of_square s2 density
  use m2
  sorry

end weight_of_second_square_l213_213360


namespace rectangle_area_comparison_l213_213778

theorem rectangle_area_comparison 
  {A A' B B' C C' D D': ℝ} 
  (h_A: A ≤ A') 
  (h_B: B ≤ B') 
  (h_C: C ≤ C') 
  (h_D: D ≤ B') : 
  A + B + C + D ≤ A' + B' + C' + D' := 
by 
  sorry

end rectangle_area_comparison_l213_213778


namespace lily_pads_half_lake_l213_213346

noncomputable def size (n : ℕ) : ℝ := sorry

theorem lily_pads_half_lake {n : ℕ} (h : size 48 = size 0 * 2^48) : size 47 = (size 48) / 2 :=
by 
  sorry

end lily_pads_half_lake_l213_213346


namespace triangle_ratio_l213_213418

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ)
  (hA : A = 2 * Real.pi / 3)
  (h_a : a = Real.sqrt 3 * c)
  (h_angle_sum : A + B + C = Real.pi)
  (h_law_of_sines : a / Real.sin A = c / Real.sin C) :
  b / c = 1 :=
sorry

end triangle_ratio_l213_213418


namespace knicks_equal_knocks_l213_213693

theorem knicks_equal_knocks :
  (8 : ℝ) / (3 : ℝ) * (5 : ℝ) / (6 : ℝ) * (30 : ℝ) = 66 + 2 / 3 :=
by
  sorry

end knicks_equal_knocks_l213_213693


namespace intersection_infinite_l213_213000

-- Define the equations of the curves
def curve1 (x y : ℝ) : Prop := 2 * x^2 - x * y - y^2 - x - 2 * y - 1 = 0
def curve2 (x y : ℝ) : Prop := 3 * x^2 - 4 * x * y + y^2 - 3 * x + y = 0

-- Theorem statement
theorem intersection_infinite : ∃ (f : ℝ → ℝ), ∀ x, curve1 x (f x) ∧ curve2 x (f x) :=
sorry

end intersection_infinite_l213_213000


namespace cos_alpha_given_tan_alpha_and_quadrant_l213_213393

theorem cos_alpha_given_tan_alpha_and_quadrant 
  (α : ℝ) 
  (h1 : Real.tan α = -1/3)
  (h2 : π/2 < α ∧ α < π) : 
  Real.cos α = -3*Real.sqrt 10 / 10 :=
by
  sorry

end cos_alpha_given_tan_alpha_and_quadrant_l213_213393


namespace squares_in_50th_ring_l213_213785

-- Define the problem using the given conditions
def centered_square_3x3 : ℕ := 3 -- Represent the 3x3 centered square

-- Define the function that computes the number of unit squares in the nth ring
def unit_squares_in_nth_ring (n : ℕ) : ℕ :=
  if n = 1 then 16
  else 24 + 8 * (n - 2)

-- Define the accumulation of unit squares up to the 50th ring
def total_squares_in_50th_ring : ℕ :=
  33 + 24 * 49

theorem squares_in_50th_ring : unit_squares_in_nth_ring 50 = 1209 :=
by
  -- Ensure that the correct value for the 50th ring can be verified
  sorry

end squares_in_50th_ring_l213_213785


namespace largest_divisor_of_462_and_231_l213_213186

def is_factor (a b : ℕ) : Prop := a ∣ b

def largest_common_divisor (a b c : ℕ) : Prop :=
  is_factor c a ∧ is_factor c b ∧ (∀ d, (is_factor d a ∧ is_factor d b) → d ≤ c)

theorem largest_divisor_of_462_and_231 :
  largest_common_divisor 462 231 231 :=
by
  sorry

end largest_divisor_of_462_and_231_l213_213186


namespace arithmetic_seq_problem_l213_213281

noncomputable def a_n (n : ℕ) (a1 : ℕ) (d : ℕ) : ℕ :=
  a1 + (n - 1) * d

theorem arithmetic_seq_problem :
  ∃ d : ℕ, a_n 1 2 d = 2 ∧ a_n 2 2 d + a_n 3 2 d = 13 ∧ (a_n 4 2 d + a_n 5 2 d + a_n 6 2 d = 42) :=
by
  sorry

end arithmetic_seq_problem_l213_213281


namespace Jeff_total_laps_l213_213284

theorem Jeff_total_laps (laps_saturday : ℕ) (laps_sunday_morning : ℕ) (laps_remaining : ℕ)
  (h1 : laps_saturday = 27) (h2 : laps_sunday_morning = 15) (h3 : laps_remaining = 56) :
  (laps_saturday + laps_sunday_morning + laps_remaining) = 98 := 
by
  sorry

end Jeff_total_laps_l213_213284


namespace tangent_circles_locus_l213_213459

theorem tangent_circles_locus :
  ∃ (a b : ℝ), ∀ (C1_center : ℝ × ℝ) (C2_center : ℝ × ℝ) (C1_radius : ℝ) (C2_radius : ℝ),
    C1_center = (0, 0) ∧ C2_center = (2, 0) ∧ C1_radius = 1 ∧ C2_radius = 3 ∧
    (∀ (r : ℝ), (a - 0)^2 + (b - 0)^2 = (r + C1_radius)^2 ∧ (a - 2)^2 + (b - 0)^2 = (C2_radius - r)^2) →
    84 * a^2 + 100 * b^2 - 64 * a - 64 = 0 := sorry

end tangent_circles_locus_l213_213459


namespace find_fraction_l213_213695

variable (x y z : ℝ)

theorem find_fraction (h : (x - y) / (z - y) = -10) : (x - z) / (y - z) = 11 := 
by
  sorry

end find_fraction_l213_213695


namespace series_sum_l213_213656

theorem series_sum : 
  (∑ n : ℕ, (2^n) / (3^(2^n) + 1)) = 1 / 2 :=
by
  sorry

end series_sum_l213_213656


namespace otimes_2_5_l213_213436

def otimes (a b : ℝ) : ℝ := 4 * a + 3 * b

theorem otimes_2_5 : otimes 2 5 = 23 :=
by
  sorry

end otimes_2_5_l213_213436


namespace probability_standard_weight_l213_213201

noncomputable def total_students : ℕ := 500
noncomputable def standard_students : ℕ := 350

theorem probability_standard_weight : (standard_students : ℚ) / (total_students : ℚ) = 7 / 10 :=
by {
  sorry
}

end probability_standard_weight_l213_213201


namespace geometric_sequence_fourth_term_l213_213319

theorem geometric_sequence_fourth_term (a : ℝ) (r : ℝ) (h : a = 512) (h1 : a * r^5 = 125) :
  a * r^3 = 1536 :=
by
  sorry

end geometric_sequence_fourth_term_l213_213319


namespace lives_after_bonus_l213_213617

variable (X Y Z : ℕ)

theorem lives_after_bonus (X Y Z : ℕ) : (X - Y + 3 * Z) = (X - Y + 3 * Z) :=
sorry

end lives_after_bonus_l213_213617


namespace haley_tickets_l213_213956

-- Conditions
def cost_per_ticket : ℕ := 4
def extra_tickets : ℕ := 5
def total_spent : ℕ := 32
def cost_extra_tickets : ℕ := extra_tickets * cost_per_ticket

-- Main proof problem
theorem haley_tickets (T : ℕ) (h : 4 * T + cost_extra_tickets = total_spent) :
  T = 3 := sorry

end haley_tickets_l213_213956


namespace common_ratio_geometric_sequence_l213_213095

theorem common_ratio_geometric_sequence
  (a_1 : ℝ) (n : ℕ) (S : ℕ → ℝ)
  (geom_sum : ∀ n q, q ≠ 1 → S n = a_1 * (1 - q^n) / (1 - q))
  (h_arithmetic : 2 * S 4 = S 5 + S 6)
  : (∃ q : ℝ, ∀ n : ℕ, q ≠ 1 → S n = a_1 * (1 - q^n) / (1 - q)) → q = -2 :=
by
  sorry

end common_ratio_geometric_sequence_l213_213095


namespace MF1_dot_MF2_range_proof_l213_213262

noncomputable def MF1_dot_MF2_range : Set ℝ :=
  Set.Icc (24 - 16 * Real.sqrt 3) (24 + 16 * Real.sqrt 3)

theorem MF1_dot_MF2_range_proof :
  ∀ (M : ℝ × ℝ), (Prod.snd M + 4) ^ 2 + (Prod.fst M) ^ 2 = 12 →
    (Prod.fst M) ^ 2 + (Prod.snd M) ^ 2 - 4 ∈ MF1_dot_MF2_range :=
by
  sorry

end MF1_dot_MF2_range_proof_l213_213262


namespace diana_wins_probability_l213_213061

theorem diana_wins_probability :
  let a := (1 / 32 : ℝ)
  let r := (1 / 32 : ℝ)
  let geom_series_sum := a / (1 - r) in
  geom_series_sum = 1 / 31 :=
by
  let a := (1 / 32 : ℝ)
  let r := (1 / 32 : ℝ)
  have sum_geom_series : geom_series_sum = a / (1 - r),
  { unfold geom_series_sum a r,
    simp [div_eq_mul_inv, mul_comm],
    sorry }

#exit

end diana_wins_probability_l213_213061


namespace sara_golf_balls_total_l213_213593

-- Define the conditions
def dozens := 16
def dozen_to_balls := 12

-- The final proof statement
theorem sara_golf_balls_total : dozens * dozen_to_balls = 192 :=
by
  sorry

end sara_golf_balls_total_l213_213593


namespace problem_l213_213088

open Real

noncomputable def f (ω a x : ℝ) := (1 / 2) * (sin (ω * x) + a * cos (ω * x))

theorem problem (a : ℝ) 
  (hω_range : 0 < ω ∧ ω ≤ 1)
  (h_f_sym1 : ∀ x, f ω a x = f ω a (π/3 - x))
  (h_f_sym2 : ∀ x, f ω a (x - π) = f ω a (x + π))
  (x1 x2 : ℝ) 
  (h_x_in_interval1 : -π/3 < x1 ∧ x1 < 5*π/3)
  (h_x_in_interval2 : -π/3 < x2 ∧ x2 < 5*π/3)
  (h_distinct : x1 ≠ x2)
  (h_f_neg_half1 : f ω a x1 = -1/2)
  (h_f_neg_half2 : f ω a x2 = -1/2) :
  (f 1 (sqrt 3) x = sin (x + π/3)) ∧ (x1 + x2 = 7*π/3) :=
by
  sorry

end problem_l213_213088


namespace geometric_sequence_a6_l213_213982

theorem geometric_sequence_a6 (a : ℕ → ℕ) (r : ℕ)
  (h₁ : a 1 = 1)
  (h₄ : a 4 = 8)
  (h_geometric : ∀ n, a n = a 1 * r^(n-1)) : 
  a 6 = 32 :=
by
  sorry

end geometric_sequence_a6_l213_213982


namespace riddles_ratio_l213_213562

theorem riddles_ratio (Josh_riddles : ℕ) (Ivory_riddles : ℕ) (Taso_riddles : ℕ) 
  (h1 : Josh_riddles = 8) 
  (h2 : Ivory_riddles = Josh_riddles + 4) 
  (h3 : Taso_riddles = 24) : 
  Taso_riddles / Ivory_riddles = 2 := 
by sorry

end riddles_ratio_l213_213562


namespace intersect_lines_l213_213460

theorem intersect_lines (k : ℝ) :
  (∃ y : ℝ, 3 * 5 - y = k ∧ -5 - y = -10) → k = 10 :=
by
  sorry

end intersect_lines_l213_213460


namespace sum_of_numbers_facing_up_is_4_probability_l213_213555

-- Definition of a uniform dice with faces numbered 1 to 6
def dice_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Definition of the sample space when the dice is thrown twice
def sample_space : Finset (ℕ × ℕ) := Finset.product dice_faces dice_faces

-- Definition of the event where the sum of the numbers is 4
def event_sum_4 : Finset (ℕ × ℕ) := sample_space.filter (fun pair => pair.1 + pair.2 = 4)

-- The number of favorable outcomes
def favorable_outcomes : ℕ := event_sum_4.card

-- The total number of possible outcomes
def total_outcomes : ℕ := sample_space.card

-- The probability of the event
def probability_event_sum_4 : ℚ := favorable_outcomes / total_outcomes

theorem sum_of_numbers_facing_up_is_4_probability :
  probability_event_sum_4 = 1 / 12 :=
by
  sorry

end sum_of_numbers_facing_up_is_4_probability_l213_213555


namespace sum_of_eight_numbers_l213_213970

theorem sum_of_eight_numbers (avg : ℝ) (S : ℝ) (h1 : avg = 5.3) (h2 : avg = S / 8) : S = 42.4 :=
by
  sorry

end sum_of_eight_numbers_l213_213970


namespace xy_sum_is_2_l213_213718

theorem xy_sum_is_2 (x y : ℝ) 
  (h1 : (x - 1) ^ 3 + 1997 * (x - 1) = -1)
  (h2 : (y - 1) ^ 3 + 1997 * (y - 1) = 1) : 
  x + y = 2 := 
  sorry

end xy_sum_is_2_l213_213718


namespace subtraction_result_l213_213807

open Matrix

namespace Vector

def a : (Fin 3 → ℝ) :=
  ![5, -3, 2]

def b : (Fin 3 → ℝ) :=
  ![-2, 4, 1]

theorem subtraction_result : a - (2 • b) = ![9, -11, 0] :=
by
  -- Skipping the proof
  sorry

end Vector

end subtraction_result_l213_213807


namespace joy_tape_deficit_l213_213290

noncomputable def tape_needed_field (width length : ℕ) : ℕ :=
2 * (length + width)

noncomputable def tape_needed_trees (num_trees circumference : ℕ) : ℕ :=
num_trees * circumference

def tape_total_needed (tape_field tape_trees : ℕ) : ℕ :=
tape_field + tape_trees

theorem joy_tape_deficit (tape_has : ℕ) (tape_field tape_trees: ℕ) : ℤ :=
tape_has - (tape_field + tape_trees)

example : joy_tape_deficit 180 (tape_needed_field 35 80) (tape_needed_trees 3 5) = -65 := by
sorry

end joy_tape_deficit_l213_213290


namespace find_x_l213_213613

theorem find_x : 
  ∃ x : ℝ, 4.7 * 13.26 + 4.7 * 9.43 + 4.7 * x = 470 ∧ x = 77.31 :=
by
  sorry

end find_x_l213_213613


namespace jordan_running_time_l213_213288

-- Define the conditions given in the problem
variables (time_steve : ℕ) (distance_steve distance_jordan_1 distance_jordan_2 distance_jordan_3 : ℕ)

-- Assign the known values
axiom time_steve_def : time_steve = 24
axiom distance_steve_def : distance_steve = 3
axiom distance_jordan_1_def : distance_jordan_1 = 2
axiom distance_jordan_2_def : distance_jordan_2 = 1
axiom distance_jordan_3_def : distance_jordan_3 = 5

axiom half_time_condition : ∀ t_2, t_2 = time_steve / 2

-- The proof problem
theorem jordan_running_time : ∀ t_j1 t_j2 t_j3, 
  (t_j1 = time_steve / 2 ∧ 
   t_j2 = t_j1 / 2 ∧ 
   t_j3 = t_j2 * 5) →
  t_j3 = 30 := 
by
  intros t_j1 t_j2 t_j3 h
  sorry

end jordan_running_time_l213_213288


namespace max_sub_min_value_l213_213853

variable {x y : ℝ}

noncomputable def expression (x y : ℝ) : ℝ :=
  (abs (x + y))^2 / ((abs x)^2 + (abs y)^2)

theorem max_sub_min_value :
  ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → 
  (expression x y ≤ 2 ∧ 0 ≤ expression x y) → 
  (∃ m M, m = 0 ∧ M = 2 ∧ M - m = 2) :=
by
  sorry

end max_sub_min_value_l213_213853


namespace sum_of_eight_numbers_l213_213977

theorem sum_of_eight_numbers (average : ℝ) (count : ℕ) (h_avg : average = 5.3) (h_count : count = 8) : (average * count = 42.4) := sorry

end sum_of_eight_numbers_l213_213977


namespace corresponding_angles_equal_l213_213586

-- Define what it means for two angles to be corresponding angles
def corresponding_angles (a b : ℝ) : Prop :=
  -- Hypothetical definition
  sorry

-- Lean 4 statement of the problem
theorem corresponding_angles_equal (a b : ℝ) (h : corresponding_angles a b) : a = b :=
by
  sorry

end corresponding_angles_equal_l213_213586


namespace trig_identity_proof_l213_213252

theorem trig_identity_proof (x : ℝ) (h : sin (x + π / 3) = 1 / 3) :
  sin (5 * π / 3 - x) - cos (2 * x - π / 3) = 4 / 9 :=
by {
  sorry
}

end trig_identity_proof_l213_213252


namespace length_YW_l213_213703

/-- Given conditions -/
def is_isosceles_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  A = B ∧ a = c

def XY : ℝ := 3
def XZ : ℝ := 5
def YZ : ℝ := 5
def ZW : ℝ := 2
def XW : ℝ := XZ - ZW

/-- Proof goal -/
theorem length_YW (a b c : ℝ) (theta : ℝ) (h_isosceles : is_isosceles_triangle a b c theta theta θ) (ZW : ℝ) :
  (Z - W = 2) →
  sqrt(9 + 9 - 36 * 41 / 50) = 4.30 :=
begin
  sorry
end

end length_YW_l213_213703


namespace find_value_of_a2_b2_c2_l213_213715

variable {a b c : ℝ}

theorem find_value_of_a2_b2_c2
  (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h2 : a + b + c = 0)
  (h3 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6 / 5 := 
sorry

end find_value_of_a2_b2_c2_l213_213715


namespace jaydee_typing_time_l213_213143

theorem jaydee_typing_time : 
  (∀ (wpm total_words : ℕ) (minutes_per_hour : ℕ),
    wpm = 38 ∧ total_words = 4560 ∧ minutes_per_hour = 60 → 
      (total_words / wpm) / minutes_per_hour = 2) :=
begin
  intros wpm total_words minutes_per_hour h,
  cases h with hwpm hwords_hours,
  cases hwords_hours with hwords hhours,
  rw [hwpm, hwords, hhours],
  norm_num,
end

end jaydee_typing_time_l213_213143


namespace rectangle_area_l213_213246

theorem rectangle_area (p q : ℝ) (x : ℝ) (h1 : x^2 + (2 * x)^2 = (p + q)^2) : 
    2 * x^2 = (2 * (p + q)^2) / 5 := 
sorry

end rectangle_area_l213_213246


namespace evaluate_operation_l213_213388

def operation (x : ℝ) : ℝ := 9 - x

theorem evaluate_operation : operation (operation 15) = 15 :=
by
  -- Proof would go here
  sorry

end evaluate_operation_l213_213388


namespace smallest_possible_AAB_l213_213361

-- Definitions of the digits A and B
def is_valid_digit (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 9

-- Definition of the condition AB equals 1/7 of AAB
def condition (A B : ℕ) : Prop := 10 * A + B = (1 / 7) * (110 * A + B)

theorem smallest_possible_AAB (A B : ℕ) : is_valid_digit A ∧ is_valid_digit B ∧ condition A B → 110 * A + B = 664 := sorry

end smallest_possible_AAB_l213_213361


namespace initial_students_l213_213736

variable (n : ℝ) (W : ℝ)

theorem initial_students 
  (h1 : W = n * 15)
  (h2 : W + 11 = (n + 1) * 14.8)
  (h3 : 15 * n + 11 = 14.8 * n + 14.8)
  (h4 : 0.2 * n = 3.8) :
  n = 19 :=
sorry

end initial_students_l213_213736


namespace transport_capacity_l213_213469

-- Declare x and y as the amount of goods large and small trucks can transport respectively
variables (x y : ℝ)

-- Given conditions
def condition1 : Prop := 2 * x + 3 * y = 15.5
def condition2 : Prop := 5 * x + 6 * y = 35

-- The goal to prove
def goal : Prop := 3 * x + 5 * y = 24.5

-- Main theorem stating that given the conditions, the goal follows
theorem transport_capacity (h1 : condition1 x y) (h2 : condition2 x y) : goal x y :=
by sorry

end transport_capacity_l213_213469


namespace ratio_of_third_layer_to_second_l213_213780

theorem ratio_of_third_layer_to_second (s1 s2 s3 : ℕ) (h1 : s1 = 2) (h2 : s2 = 2 * s1) (h3 : s3 = 12) : s3 / s2 = 3 := 
by
  sorry

end ratio_of_third_layer_to_second_l213_213780


namespace diagonals_from_vertex_of_regular_polygon_l213_213423

-- Definitions for the conditions in part a)
def exterior_angle (n : ℕ) : ℚ := 360 / n

-- Proof problem statement
theorem diagonals_from_vertex_of_regular_polygon
  (n : ℕ)
  (h1 : exterior_angle n = 36)
  : n - 3 = 7 :=
by sorry

end diagonals_from_vertex_of_regular_polygon_l213_213423


namespace no_partition_exists_l213_213184

noncomputable section

open Set

def partition_N (A B C : Set ℕ) : Prop := 
  A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅ ∧  -- Non-empty sets
  A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ C ∩ A = ∅ ∧  -- Disjoint sets
  A ∪ B ∪ C = univ ∧  -- Covers the whole ℕ
  (∀ a ∈ A, ∀ b ∈ B, a + b + 2008 ∈ C) ∧
  (∀ b ∈ B, ∀ c ∈ C, b + c + 2008 ∈ A) ∧
  (∀ c ∈ C, ∀ a ∈ A, c + a + 2008 ∈ B)

theorem no_partition_exists : ¬ ∃ (A B C : Set ℕ), partition_N A B C :=
by
  sorry

end no_partition_exists_l213_213184


namespace a_sufficient_not_necessary_l213_213937

theorem a_sufficient_not_necessary (a : ℝ) : (a > 1 → 1 / a < 1) ∧ (¬(1 / a < 1 → a > 1)) :=
by
  sorry

end a_sufficient_not_necessary_l213_213937


namespace units_digit_product_l213_213895

theorem units_digit_product (k l : ℕ) (h1 : ∀ n : ℕ, (5^n % 10) = 5) (h2 : ∀ m < 4, (6^m % 10) = 6) :
  ((5^k * 6^l) % 10) = 0 :=
by
  have h5 : (5^k % 10) = 5 := h1 k
  have h6 : (6^4 % 10) = 6 := h2 4 (by sorry)
  have h_product : (5^k * 6^l % 10) = ((5 % 10) * (6 % 10) % 10) := sorry
  norm_num at h_product
  exact h_product

end units_digit_product_l213_213895


namespace trains_cross_time_l213_213335

theorem trains_cross_time
  (length : ℝ)
  (time1 time2 : ℝ)
  (h_length : length = 120)
  (h_time1 : time1 = 5)
  (h_time2 : time2 = 15)
  : (2 * length) / ((length / time1) + (length / time2)) = 7.5 := by
  sorry

end trains_cross_time_l213_213335


namespace parabola_coefficients_l213_213160

theorem parabola_coefficients 
  (a b c : ℝ) 
  (h_vertex : ∀ x : ℝ, (2 - (-2))^2 * a + (-2 * 2 * a + b) * (2 - (-2)) + (c - 5) = 0)
  (h_point : 9 = a * (2:ℝ)^2 + b * (2:ℝ) + c) : 
  a = 1 / 4 ∧ b = 1 ∧ c = 6 := 
by 
  sorry

end parabola_coefficients_l213_213160


namespace dividend_is_5336_l213_213902

theorem dividend_is_5336 (D Q R : ℕ) (h1 : D = 10 * Q) (h2 : D = 5 * R) (h3 : R = 46) : 
  D * Q + R = 5336 := 
by sorry

end dividend_is_5336_l213_213902


namespace female_with_advanced_degrees_l213_213901

theorem female_with_advanced_degrees
  (total_employees : ℕ)
  (total_females : ℕ)
  (total_employees_with_advanced_degrees : ℕ)
  (total_employees_with_college_degree_only : ℕ)
  (total_males_with_college_degree_only : ℕ)
  (h1 : total_employees = 180)
  (h2 : total_females = 110)
  (h3 : total_employees_with_advanced_degrees = 90)
  (h4 : total_employees_with_college_degree_only = 90)
  (h5 : total_males_with_college_degree_only = 35) :
  ∃ (female_with_advanced_degrees : ℕ), female_with_advanced_degrees = 55 :=
by
  -- the proof goes here
  sorry

end female_with_advanced_degrees_l213_213901


namespace percentage_slump_in_business_l213_213320

theorem percentage_slump_in_business (X : ℝ) (Y : ℝ) :
  0.05 * Y = 0.04 * X → Y = 0.8 * X → 20 := 
by
  sorry

end percentage_slump_in_business_l213_213320


namespace knicks_equal_knocks_l213_213694

theorem knicks_equal_knocks :
  (8 : ℝ) / (3 : ℝ) * (5 : ℝ) / (6 : ℝ) * (30 : ℝ) = 66 + 2 / 3 :=
by
  sorry

end knicks_equal_knocks_l213_213694


namespace sum_of_shaded_cells_l213_213310

theorem sum_of_shaded_cells (a b c d e f : ℕ) 
  (h1: (a = 1 ∨ a = 2 ∨ a = 3) ∧ (b = 1 ∨ b = 2 ∨ b = 3) ∧ (c = 1 ∨ c = 2 ∨ c = 3) ∧ 
       (d = 1 ∨ d = 2 ∨ d = 3) ∧ (e = 1 ∨ e = 2 ∨ e = 3) ∧ (f = 1 ∨ f = 2 ∨ f = 3))
  (h2: (a ≠ b ∧ a ≠ c ∧ b ≠ c) ∧ 
       (d ≠ e ∧ d ≠ f ∧ e ≠ f) ∧ 
       (a ≠ d ∧ a ≠ f ∧ d ≠ f ∧ 
        b ≠ e ∧ b ≠ f ∧ c ≠ e ∧ c ≠ f))
  (h3: c = 3 ∧ d = 3 ∧ b = 2 ∧ e = 2)
  : b + e = 4 := 
sorry

end sum_of_shaded_cells_l213_213310


namespace unique_number_l213_213079

theorem unique_number (a : ℕ) (h1 : 1 < a) 
  (h2 : ∀ p : ℕ, Prime p → p ∣ a^6 - 1 → p ∣ a^3 - 1 ∨ p ∣ a^2 - 1) : a = 2 :=
by
  sorry

end unique_number_l213_213079


namespace max_permutations_Q_l213_213149

open Fintype Finset

def valid_permutations (Q : Finset (Equiv.Perm (Fin 100))) : Prop :=
  ∀ a b : Fin 100, a ≠ b → 
  (Finset.univ.filter (λ σ : Equiv.Perm (Fin 100), σ (a) + 1 = σ (b))).card ≤ 1

theorem max_permutations_Q : 
  ∃ (Q : Finset (Equiv.Perm (Fin 100))), 
  valid_permutations Q ∧ Q.card = 100 :=
sorry

end max_permutations_Q_l213_213149


namespace shelby_gold_stars_l213_213304

theorem shelby_gold_stars (stars_yesterday stars_today : ℕ) (h1 : stars_yesterday = 4) (h2 : stars_today = 3) :
  stars_yesterday + stars_today = 7 := 
by
  sorry

end shelby_gold_stars_l213_213304


namespace infinite_squares_form_l213_213849

theorem infinite_squares_form (k : ℕ) (hk : 0 < k) : ∃ f : ℕ → ℕ, ∀ n, ∃ a, a^2 = f n * 2^k - 7 :=
by
  sorry

end infinite_squares_form_l213_213849


namespace angle_at_3_15_l213_213519

-- Define the measurements and conditions
def hour_hand_position (hour min : ℕ) : ℝ := 
  30 * hour + 0.5 * min

def minute_hand_position (min : ℕ) : ℝ := 
  6 * min

def angle_between_hands (hour min : ℕ) : ℝ := 
  abs (minute_hand_position min - hour_hand_position hour min)

-- Theorem statement in Lean 4
theorem angle_at_3_15 : angle_between_hands 3 15 = 7.5 :=
by sorry

end angle_at_3_15_l213_213519


namespace balls_in_boxes_l213_213827

theorem balls_in_boxes : ∃ (ways : ℕ), ways = 104 ∧
  let balls := 7;
      boxes := 4 in
  -- Here should be the formal definition of the number of ways to distribute the balls into the boxes,
  -- but we state it as an existential statement acknowledging the result.
  ways = (∑ p in (finset.powerset len_le_boxes (univ (finset.range balls + 1))),
               if (∑ x in p, x) = balls then multinomial p else 0) := sorry

end balls_in_boxes_l213_213827


namespace average_of_two_integers_l213_213313

theorem average_of_two_integers {A B C D : ℕ} (h1 : A + B + C + D = 200) (h2 : C ≤ 130) : (A + B) / 2 = 35 :=
by
  sorry

end average_of_two_integers_l213_213313


namespace factorize_expression_l213_213378

variable (a b : ℝ)

theorem factorize_expression : ab^2 - a = a * (b + 1) * (b - 1) :=
sorry

end factorize_expression_l213_213378


namespace pure_imaginary_number_l213_213415

theorem pure_imaginary_number (m : ℝ) (h_real : m^2 - 5 * m + 6 = 0) (h_imag : m^2 - 3 * m ≠ 0) : m = 2 :=
sorry

end pure_imaginary_number_l213_213415


namespace candy_per_packet_l213_213220

-- Define the conditions as hypotheses
def bobby_weekly_candies (mon_to_fri_candies : ℕ) (sat_sun_candies : ℕ) : ℕ :=
  mon_to_fri_candies + sat_sun_candies

def total_candies_in_n_weeks (weekly_candies : ℕ) (n : ℕ) : ℕ :=
  weekly_candies * n

theorem candy_per_packet
  (mon_to_fri_candies_per_day : ℕ)
  (sat_sun_candies_per_day : ℕ)
  (days_mon_to_fri : ℕ)
  (days_weekend : ℕ)
  (num_weeks : ℕ)
  (total_packets : ℕ)
  (candies_per_packet : ℕ)
  (h1 : mon_to_fri_candies_per_day = 2)
  (h2 : sat_sun_candies_per_day = 1)
  (h3 : days_mon_to_fri = 5)
  (h4 : days_weekend = 2)
  (h5 : num_weeks = 3)
  (h6 : total_packets = 2)
  (h7 : candies_per_packet = (total_candies_in_n_weeks (bobby_weekly_candies (mon_to_fri_candies_per_day * days_mon_to_fri) (sat_sun_candies_per_day * days_weekend)) num_weeks) / total_packets) :
  candies_per_packet = 18 :=
sorry

end candy_per_packet_l213_213220


namespace valid_subsets_count_l213_213409

open Finset

noncomputable def count_valid_subsets (n : ℕ) : ℕ :=
  ∑ k in range 10, (choose (n - k - 1) k)

theorem valid_subsets_count :
  count_valid_subsets 20 = 17699 :=
by
  unfold count_valid_subsets
  rw [sum]
  sorry -- Steps of the proof would go here

end valid_subsets_count_l213_213409


namespace smallest_value_n_l213_213733

def factorial_factors_of_five (n : ℕ) : ℕ :=
  n / 5 + n / 25 + n / 125 + n / 625 + n / 3125

theorem smallest_value_n
  (a b c m n : ℕ)
  (h1 : a + b + c = 2003)
  (h2 : a = 2 * b)
  (h3 : a.factorial * b.factorial * c.factorial = m * 10 ^ n)
  (h4 : ¬ (10 ∣ m)) :
  n = 400 :=
by
  sorry

end smallest_value_n_l213_213733


namespace negation_of_no_honors_students_attend_school_l213_213877

-- Definitions (conditions and question)
def honors_student (x : Type) : Prop := sorry -- The condition defining an honors student
def attends_school (x : Type) : Prop := sorry -- The condition defining a student attending the school

-- The theorem statement
theorem negation_of_no_honors_students_attend_school :
  (¬ ∃ x : Type, honors_student x ∧ attends_school x) ↔ (∃ x : Type, honors_student x ∧ attends_school x) :=
sorry

end negation_of_no_honors_students_attend_school_l213_213877


namespace find_custom_operator_result_l213_213439

def custom_operator (a b : ℝ) : ℝ := 4 * a + 3 * b

theorem find_custom_operator_result :
  custom_operator 2 5 = 23 :=
by
  sorry

end find_custom_operator_result_l213_213439


namespace minimum_common_perimeter_l213_213016

noncomputable def is_integer (x: ℝ) : Prop := ∃ (n: ℤ), x = n

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ a = c) ∧ is_triangle a b c

theorem minimum_common_perimeter :
  ∃ (a b : ℝ),
    is_integer a ∧ is_integer b ∧
    4 * a = 5 * b - 18 ∧
    is_isosceles_triangle a a (2 * a - 12) ∧
    is_isosceles_triangle b b (3 * b - 30) ∧
    (2 * a + (2 * a - 12) = 2 * b + (3 * b - 30)) ∧
    (2 * a + (2 * a - 12) = 228) := sorry

end minimum_common_perimeter_l213_213016


namespace distance_traveled_is_correct_l213_213698

noncomputable def speed_in_mph : ℝ := 23.863636363636363
noncomputable def seconds : ℝ := 2

-- constants for conversion
def miles_to_feet : ℝ := 5280
def hours_to_seconds : ℝ := 3600

-- speed in feet per second
noncomputable def speed_in_fps : ℝ := speed_in_mph * miles_to_feet / hours_to_seconds

-- distance traveled
noncomputable def distance : ℝ := speed_in_fps * seconds

theorem distance_traveled_is_correct : distance = 69.68 := by
  sorry

end distance_traveled_is_correct_l213_213698


namespace rhombus_fourth_vertex_l213_213809

theorem rhombus_fourth_vertex (a b : ℝ) :
  ∃ x y : ℝ, (x, y) = (a - b, a + b) ∧ dist (a, b) (x, y) = dist (-b, a) (x, y) ∧ dist (-b, a) (x, y) = dist (0, 0) (x, y) :=
by
  use (a - b)
  use (a + b)
  sorry

end rhombus_fourth_vertex_l213_213809


namespace find_r_value_l213_213573

theorem find_r_value (n : ℕ) (r s : ℕ) (h_s : s = 2^n - 1) (h_r : r = 3^s - s) (h_n : n = 3) : r = 2180 :=
by
  sorry

end find_r_value_l213_213573


namespace radius_is_independent_variable_l213_213502

theorem radius_is_independent_variable 
  (r C : ℝ)
  (h : C = 2 * Real.pi * r) : 
  ∃ r_independent, r_independent = r := 
by
  sorry

end radius_is_independent_variable_l213_213502


namespace three_digit_numbers_count_correct_l213_213958

def digits : List ℕ := [2, 3, 4, 5, 5, 5, 6, 6]

def three_digit_numbers_count (d : List ℕ) : ℕ := 
  -- To be defined: Full implementation for counting matching three-digit numbers
  sorry

theorem three_digit_numbers_count_correct :
  three_digit_numbers_count digits = 85 :=
sorry

end three_digit_numbers_count_correct_l213_213958


namespace juice_problem_l213_213483

theorem juice_problem 
  (p_a p_y : ℚ)
  (v_a v_y : ℚ)
  (p_total v_total : ℚ)
  (ratio_a : p_a / v_a = 4)
  (ratio_y : p_y / v_y = 1 / 5)
  (p_a_val : p_a = 20)
  (p_total_val : p_total = 24)
  (v_total_eq : v_total = v_a + v_y)
  (p_y_def : p_y = p_total - p_a) :
  v_total = 25 :=
by
  sorry

end juice_problem_l213_213483


namespace cube_edge_numbers_equal_top_bottom_l213_213987

theorem cube_edge_numbers_equal_top_bottom (
  numbers : List ℕ,
  h : numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) :
  ∃ (top bottom : List ℕ),
    (∀ x, x ∈ top → x ∈ numbers) ∧
    (∀ x, x ∈ bottom → x ∈ numbers) ∧
    (top ≠ bottom) ∧
    (top.length = 4) ∧ 
    (bottom.length = 4) ∧ 
    (top.product = bottom.product) :=
begin
  sorry
end

end cube_edge_numbers_equal_top_bottom_l213_213987


namespace slope_of_line_eq_neg_four_thirds_l213_213620

variable {x y : ℝ}
variable (p₁ p₂ : ℝ × ℝ) (h₁ : 3 / p₁.1 + 4 / p₁.2 = 0) (h₂ : 3 / p₂.1 + 4 / p₂.2 = 0)

theorem slope_of_line_eq_neg_four_thirds 
  (hneq : p₁.1 ≠ p₂.1):
  (p₂.2 - p₁.2) / (p₂.1 - p₁.1) = -4 / 3 := 
sorry

end slope_of_line_eq_neg_four_thirds_l213_213620


namespace find_a_l213_213106

theorem find_a (a : ℝ) :
  (∃! x : ℝ, (a^2 - 1) * x^2 + (a + 1) * x + 1 = 0) ↔ a = 1 ∨ a = 5/3 :=
by
  sorry

end find_a_l213_213106
