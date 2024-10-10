import Mathlib

namespace find_M_l1681_168162

theorem find_M : ∃ M : ℕ, 995 + 997 + 999 + 1001 + 1003 = 5100 - M ∧ M = 104 := by
  sorry

end find_M_l1681_168162


namespace circle_equation_l1681_168131

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : Point2D
  radius : ℝ

/-- Function to check if a point is on a circle -/
def isOnCircle (p : Point2D) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Function to check if two points are symmetric with respect to y = x -/
def isSymmetricYEqX (p1 p2 : Point2D) : Prop :=
  p1.x = p2.y ∧ p1.y = p2.x

/-- Theorem: Given a circle C with radius 1 and center symmetric to (1, 0) 
    with respect to the line y = x, its standard equation is x^2 + (y - 1)^2 = 1 -/
theorem circle_equation (C : Circle) 
    (h1 : C.radius = 1)
    (h2 : isSymmetricYEqX C.center ⟨1, 0⟩) : 
    ∀ (p : Point2D), isOnCircle p C ↔ p.x^2 + (p.y - 1)^2 = 1 := by
  sorry

end circle_equation_l1681_168131


namespace find_divisor_find_divisor_proof_l1681_168149

theorem find_divisor (original : ℕ) (divisible : ℕ) (divisor : ℕ) : Prop :=
  (original = 859622) →
  (divisible = 859560) →
  (divisor = 62) →
  (original - divisible = divisor) ∧
  (divisible % divisor = 0)

/-- The proof of the theorem --/
theorem find_divisor_proof : ∃ (d : ℕ), find_divisor 859622 859560 d :=
  sorry

end find_divisor_find_divisor_proof_l1681_168149


namespace right_quadrilateral_area_area_is_twelve_l1681_168102

/-- A quadrilateral with right angles at B and D, diagonal AC of length 5, and sides AB and AD of lengths 3 and 4 respectively. -/
structure RightQuadrilateral where
  AC : ℝ
  AB : ℝ
  AD : ℝ
  ac_eq : AC = 5
  ab_eq : AB = 3
  ad_eq : AD = 4

/-- The area of the RightQuadrilateral is 12. -/
theorem right_quadrilateral_area (q : RightQuadrilateral) : ℝ :=
  12

/-- The area of a RightQuadrilateral is equal to 12. -/
theorem area_is_twelve (q : RightQuadrilateral) : right_quadrilateral_area q = 12 := by
  sorry

end right_quadrilateral_area_area_is_twelve_l1681_168102


namespace mod_23_equivalence_l1681_168124

theorem mod_23_equivalence :
  ∃! n : ℕ, 0 ≤ n ∧ n < 23 ∧ 123456 % 23 = n :=
by
  -- The proof goes here
  sorry

end mod_23_equivalence_l1681_168124


namespace coopers_age_l1681_168167

theorem coopers_age (cooper dante maria : ℕ) : 
  cooper + dante + maria = 31 →
  dante = 2 * cooper →
  maria = dante + 1 →
  cooper = 6 := by
sorry

end coopers_age_l1681_168167


namespace arithmetic_sequence_common_difference_l1681_168140

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 1 + a 7 = -2)
  (h_a3 : a 3 = 2) :
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = -3 :=
sorry

end arithmetic_sequence_common_difference_l1681_168140


namespace notP_set_equals_interval_l1681_168105

-- Define the proposition P
def P (x : ℝ) : Prop := 1 / (x^2 - x - 2) > 0

-- Define the set of x satisfying ¬P
def notP_set : Set ℝ := {x : ℝ | ¬(P x)}

-- Theorem stating that notP_set is equal to the closed interval [-1, 2]
theorem notP_set_equals_interval :
  notP_set = Set.Icc (-1 : ℝ) 2 := by sorry

end notP_set_equals_interval_l1681_168105


namespace probability_of_event_l1681_168145

noncomputable def x : ℝ := sorry

-- x is uniformly distributed between 200 and 300
axiom x_range : 200 ≤ x ∧ x < 300

-- Floor of square root of 2x is 25
axiom floor_sqrt_2x : ⌊Real.sqrt (2 * x)⌋ = 25

-- Define the event that floor of square root of x is 17
def event : Prop := ⌊Real.sqrt x⌋ = 17

-- Define the probability measure
noncomputable def P : Set ℝ → ℝ := sorry

-- Theorem statement
theorem probability_of_event :
  P {y : ℝ | 200 ≤ y ∧ y < 300 ∧ ⌊Real.sqrt (2 * y)⌋ = 25 ∧ ⌊Real.sqrt y⌋ = 17} / 
  P {y : ℝ | 200 ≤ y ∧ y < 300} = 23 / 200 := by sorry

end probability_of_event_l1681_168145


namespace trigonometric_expression_equality_l1681_168173

theorem trigonometric_expression_equality : 
  4 * Real.sin (60 * π / 180) - Real.sqrt 12 + (-3)^2 - (1 / (2 - Real.sqrt 3)) = 7 - Real.sqrt 3 := by
  sorry

end trigonometric_expression_equality_l1681_168173


namespace problem_solution_l1681_168122

theorem problem_solution (a : ℝ) (h : a^2 - 2*a = -1) : 3*a^2 - 6*a + 2027 = 2024 := by
  sorry

end problem_solution_l1681_168122


namespace product_b_sample_size_l1681_168111

/-- Calculates the number of items drawn from a specific product
    using stratified sampling method. -/
def stratifiedSample (totalItems : ℕ) (ratio : List ℕ) (sampleSize : ℕ) (productIndex : ℕ) : ℕ :=
  (sampleSize * (ratio.get! productIndex)) / (ratio.sum)

/-- Theorem: Given 1200 total items with ratio 3:4:5 for products A, B, and C,
    when drawing 60 items using stratified sampling,
    the number of items drawn from product B is 20. -/
theorem product_b_sample_size :
  let totalItems : ℕ := 1200
  let ratio : List ℕ := [3, 4, 5]
  let sampleSize : ℕ := 60
  let productBIndex : ℕ := 1
  stratifiedSample totalItems ratio sampleSize productBIndex = 20 := by
  sorry

end product_b_sample_size_l1681_168111


namespace faye_earnings_proof_l1681_168138

/-- The number of bead necklaces Faye sold -/
def bead_necklaces : ℕ := 3

/-- The number of gem stone necklaces Faye sold -/
def gem_necklaces : ℕ := 7

/-- The cost of each necklace in dollars -/
def necklace_cost : ℕ := 7

/-- Faye's total earnings from selling necklaces -/
def faye_earnings : ℕ := (bead_necklaces + gem_necklaces) * necklace_cost

theorem faye_earnings_proof : faye_earnings = 70 := by
  sorry

end faye_earnings_proof_l1681_168138


namespace beads_per_necklace_l1681_168143

theorem beads_per_necklace (total_beads : ℕ) (num_necklaces : ℕ) 
  (h1 : total_beads = 308) (h2 : num_necklaces = 11) :
  total_beads / num_necklaces = 28 := by
  sorry

end beads_per_necklace_l1681_168143


namespace patches_in_unit_l1681_168116

/-- The number of patches in a unit given cost price, selling price, and net profit -/
theorem patches_in_unit (cost_price selling_price net_profit : ℚ) : 
  cost_price = 1.25 → 
  selling_price = 12 → 
  net_profit = 1075 → 
  (net_profit / (selling_price - cost_price) : ℚ) = 100 := by
  sorry

end patches_in_unit_l1681_168116


namespace smallest_number_of_eggs_smallest_number_of_eggs_is_162_l1681_168153

theorem smallest_number_of_eggs (total_containers : ℕ) (eggs_per_full_container : ℕ) 
  (underfilled_containers : ℕ) (eggs_per_underfilled : ℕ) : ℕ :=
  let total_eggs := (total_containers - underfilled_containers) * eggs_per_full_container + 
                    underfilled_containers * eggs_per_underfilled
  have h1 : eggs_per_full_container = 15 := by sorry
  have h2 : underfilled_containers = 3 := by sorry
  have h3 : eggs_per_underfilled = 14 := by sorry
  have h4 : total_eggs > 150 := by sorry
  have h5 : ∀ n : ℕ, n < total_containers → 
            n * eggs_per_full_container - underfilled_containers ≤ 150 := by sorry
  total_eggs

theorem smallest_number_of_eggs_is_162 : 
  smallest_number_of_eggs 11 15 3 14 = 162 := by sorry

end smallest_number_of_eggs_smallest_number_of_eggs_is_162_l1681_168153


namespace cubic_with_repeated_root_l1681_168127

/-- Given a cubic polynomial 2x^3 + 8x^2 - 120x + k = 0 with a repeated root and positive k,
    prove that k = 6400/27 -/
theorem cubic_with_repeated_root (k : ℝ) : 
  (∃ x y : ℝ, (2 * x^3 + 8 * x^2 - 120 * x + k = 0) ∧ 
               (2 * y^3 + 8 * y^2 - 120 * y + k = 0) ∧ 
               (x ≠ y)) ∧
  (∃ z : ℝ, (2 * z^3 + 8 * z^2 - 120 * z + k = 0) ∧ 
            (∀ w : ℝ, 2 * w^3 + 8 * w^2 - 120 * w + k = 0 → w = z ∨ w = x ∨ w = y)) ∧
  (k > 0) →
  k = 6400 / 27 := by
sorry

end cubic_with_repeated_root_l1681_168127


namespace total_pencils_is_twelve_l1681_168161

/-- The number of pencils each child has -/
def pencils_per_child : ℕ := 6

/-- The number of children -/
def number_of_children : ℕ := 2

/-- The total number of pencils -/
def total_pencils : ℕ := pencils_per_child * number_of_children

theorem total_pencils_is_twelve : total_pencils = 12 := by
  sorry

end total_pencils_is_twelve_l1681_168161


namespace sum_of_consecutive_naturals_with_lcm_168_l1681_168148

def consecutive_naturals (n : ℕ) : Fin 3 → ℕ := λ i => n + i.val

theorem sum_of_consecutive_naturals_with_lcm_168 :
  ∃ n : ℕ, (Nat.lcm (consecutive_naturals n 0) (Nat.lcm (consecutive_naturals n 1) (consecutive_naturals n 2)) = 168) ∧
  (consecutive_naturals n 0 + consecutive_naturals n 1 + consecutive_naturals n 2 = 21) := by
  sorry

end sum_of_consecutive_naturals_with_lcm_168_l1681_168148


namespace triangle_base_value_l1681_168130

theorem triangle_base_value (L R B : ℝ) : 
  L + R + B = 50 →
  R = L + 2 →
  L = 12 →
  B = 24 := by
sorry

end triangle_base_value_l1681_168130


namespace clock_angle_at_2_30_l1681_168175

/-- The number of hours on a standard analog clock -/
def clock_hours : ℕ := 12

/-- The number of degrees in a full rotation -/
def full_rotation : ℕ := 360

/-- The time in hours (including fractional part) -/
def time : ℚ := 2.5

/-- Calculates the angle of the hour hand from the 12 o'clock position -/
def hour_hand_angle (t : ℚ) : ℚ := (t * full_rotation) / clock_hours

/-- Calculates the angle of the minute hand from the 12 o'clock position -/
def minute_hand_angle (t : ℚ) : ℚ := ((t - t.floor) * full_rotation)

/-- Calculates the absolute difference between two angles -/
def angle_difference (a b : ℚ) : ℚ := min (abs (a - b)) (full_rotation - abs (a - b))

theorem clock_angle_at_2_30 :
  angle_difference (hour_hand_angle time) (minute_hand_angle time) = 105 := by
  sorry

end clock_angle_at_2_30_l1681_168175


namespace unique_prime_product_perfect_power_l1681_168176

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The product of the first k prime numbers -/
def primeProduct (k : ℕ) : ℕ := sorry

/-- A number is a perfect power if it can be expressed as a^n where a > 1 and n > 1 -/
def isPerfectPower (m : ℕ) : Prop := sorry

theorem unique_prime_product_perfect_power :
  ∀ k : ℕ, (k ≠ 0 ∧ isPerfectPower (primeProduct k - 1)) ↔ k = 1 := by sorry

end unique_prime_product_perfect_power_l1681_168176


namespace fraction_problem_l1681_168186

theorem fraction_problem (N : ℚ) : (5 / 6 : ℚ) * N = (5 / 16 : ℚ) * N + 250 → N = 480 := by
  sorry

end fraction_problem_l1681_168186


namespace inequalities_not_equivalent_l1681_168141

theorem inequalities_not_equivalent : 
  ¬(∀ x : ℝ, (Real.sqrt (x - 1) < Real.sqrt (2 - x)) ↔ (x - 1 < 2 - x)) := by
sorry

end inequalities_not_equivalent_l1681_168141


namespace exact_blue_marbles_probability_l1681_168155

def total_marbles : ℕ := 20
def blue_marbles : ℕ := 12
def red_marbles : ℕ := 8
def num_draws : ℕ := 8
def num_blue_draws : ℕ := 5

def prob_blue : ℚ := blue_marbles / total_marbles
def prob_red : ℚ := red_marbles / total_marbles

theorem exact_blue_marbles_probability :
  (Nat.choose num_draws num_blue_draws : ℚ) *
  (prob_blue ^ num_blue_draws) *
  (prob_red ^ (num_draws - num_blue_draws)) =
  108864 / 390625 :=
by sorry

end exact_blue_marbles_probability_l1681_168155


namespace simplify_expression_l1681_168144

theorem simplify_expression (y : ℝ) : 5*y + 7*y - 3*y = 9*y := by
  sorry

end simplify_expression_l1681_168144


namespace tip_percentage_is_15_percent_l1681_168191

def lunch_cost : ℝ := 50.50
def total_spent : ℝ := 58.075

theorem tip_percentage_is_15_percent :
  (total_spent - lunch_cost) / lunch_cost * 100 = 15 := by sorry

end tip_percentage_is_15_percent_l1681_168191


namespace tangent_line_parallel_l1681_168199

/-- Given a curve y = 3x^2 + 2x with a tangent line at (1, 5) parallel to 2ax - y - 6 = 0, 
    the value of a is 4. -/
theorem tangent_line_parallel (a : ℝ) : 
  (∃ (f : ℝ → ℝ) (line : ℝ → ℝ), 
    (∀ x, f x = 3 * x^2 + 2 * x) ∧ 
    (∀ x, line x = 2 * a * x - 6) ∧
    (∃ (tangent : ℝ → ℝ), 
      (tangent 1 = f 1) ∧
      (∀ h : ℝ, h ≠ 0 → (tangent (1 + h) - tangent 1) / h = (line (1 + h) - line 1) / h))) →
  a = 4 := by
sorry

end tangent_line_parallel_l1681_168199


namespace expression_evaluation_l1681_168108

theorem expression_evaluation :
  let a : ℚ := 1/2
  let b : ℚ := -1
  (2*a - b)^2 + (a - b)*(a + b) - 5*a*(a - 2*b) = -3 :=
by sorry

end expression_evaluation_l1681_168108


namespace expand_expression_l1681_168139

theorem expand_expression (x : ℝ) : (17*x + 18 - 3*x^2) * (4*x) = -12*x^3 + 68*x^2 + 72*x := by
  sorry

end expand_expression_l1681_168139


namespace sequence_fifth_term_is_fifteen_l1681_168180

theorem sequence_fifth_term_is_fifteen (a : ℕ → ℝ) :
  (∀ n : ℕ, n ≠ 0 → a n / n = n - 2) →
  a 5 = 15 := by
sorry

end sequence_fifth_term_is_fifteen_l1681_168180


namespace meeting_probability_l1681_168195

/-- Xiaocong's arrival time at Wuhan Station -/
def xiaocong_arrival : ℝ := 13.5

/-- Duration of Xiaocong's rest at Wuhan Station -/
def xiaocong_rest : ℝ := 1

/-- Earliest possible arrival time for Xiaoming -/
def xiaoming_earliest : ℝ := 14

/-- Latest possible arrival time for Xiaoming -/
def xiaoming_latest : ℝ := 15

/-- Xiaoming's train departure time -/
def xiaoming_departure : ℝ := 15.5

/-- The probability of Xiaocong and Xiaoming meeting at Wuhan Station -/
theorem meeting_probability : ℝ := by sorry

end meeting_probability_l1681_168195


namespace spring_membership_decrease_l1681_168110

theorem spring_membership_decrease
  (fall_increase : Real)
  (total_decrease : Real)
  (h1 : fall_increase = 0.06)
  (h2 : total_decrease = 0.1414) :
  let fall_membership := 1 + fall_increase
  let spring_membership := 1 - total_decrease
  (fall_membership - spring_membership) / fall_membership = 0.19 := by
sorry

end spring_membership_decrease_l1681_168110


namespace binomial_coefficient_10_3_l1681_168196

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l1681_168196


namespace school_supplies_purchase_l1681_168106

-- Define the cost of one unit of type A and one unit of type B
def cost_A : ℝ := 15
def cost_B : ℝ := 25

-- Define the total number of units to be purchased
def total_units : ℕ := 100

-- Define the maximum total cost
def max_total_cost : ℝ := 2000

-- Theorem to prove
theorem school_supplies_purchase :
  -- Condition 1: The sum of costs of one unit of each type is $40
  cost_A + cost_B = 40 →
  -- Condition 2: The number of units of type A that can be purchased with $90 
  -- is the same as the number of units of type B that can be purchased with $150
  90 / cost_A = 150 / cost_B →
  -- Condition 3: The total cost should not exceed $2000
  ∀ y : ℕ, y ≤ total_units → cost_A * y + cost_B * (total_units - y) ≤ max_total_cost →
  -- Conclusion: The minimum number of units of type A to be purchased is 50
  (∀ z : ℕ, z < 50 → cost_A * z + cost_B * (total_units - z) > max_total_cost) ∧
  cost_A * 50 + cost_B * (total_units - 50) ≤ max_total_cost :=
by sorry


end school_supplies_purchase_l1681_168106


namespace cube_volume_from_face_perimeter_l1681_168126

/-- Given a cube with a face perimeter of 24 cm, prove its volume is 216 cubic cm. -/
theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (h : face_perimeter = 24) :
  let side_length := face_perimeter / 4
  side_length ^ 3 = 216 := by
  sorry

end cube_volume_from_face_perimeter_l1681_168126


namespace product_of_decimals_l1681_168187

theorem product_of_decimals : (0.5 : ℝ) * 0.3 = 0.15 := by
  sorry

end product_of_decimals_l1681_168187


namespace mother_age_twice_lisa_l1681_168189

/-- Lisa's birth year -/
def lisa_birth_year : ℕ := 1994

/-- The year of Lisa's 10th birthday -/
def reference_year : ℕ := 2004

/-- Lisa's age in the reference year -/
def lisa_age_reference : ℕ := 10

/-- Lisa's mother's age multiplier in the reference year -/
def mother_age_multiplier_reference : ℕ := 5

/-- The year when Lisa's mother's age is twice Lisa's age -/
def target_year : ℕ := 2034

theorem mother_age_twice_lisa (y : ℕ) :
  (y - lisa_birth_year) * 2 = (y - lisa_birth_year + mother_age_multiplier_reference * lisa_age_reference - lisa_age_reference) →
  y = target_year :=
by sorry

end mother_age_twice_lisa_l1681_168189


namespace hot_water_bottle_price_is_six_l1681_168137

/-- The price of a hot-water bottle given the conditions of the problem -/
def hot_water_bottle_price (thermometer_price : ℚ) (total_sales : ℚ) 
  (thermometer_to_bottle_ratio : ℕ) (bottles_sold : ℕ) : ℚ :=
  (total_sales - thermometer_price * (thermometer_to_bottle_ratio * bottles_sold)) / bottles_sold

theorem hot_water_bottle_price_is_six :
  hot_water_bottle_price 2 1200 7 60 = 6 := by
  sorry

end hot_water_bottle_price_is_six_l1681_168137


namespace race_result_l1681_168152

/-- The race between John and Steve --/
theorem race_result (initial_distance : ℝ) (john_speed steve_speed : ℝ) (time : ℝ) :
  initial_distance = 14 →
  john_speed = 4.2 →
  steve_speed = 3.7 →
  time = 32 →
  (john_speed * time + initial_distance) - (steve_speed * time) = 30 := by
  sorry

end race_result_l1681_168152


namespace minimum_red_chips_l1681_168114

theorem minimum_red_chips 
  (w b r : ℕ) 
  (blue_white : b ≥ w / 4)
  (blue_red : b ≤ r / 6)
  (white_blue_total : w + b ≥ 75) :
  r ≥ 90 ∧ ∀ r', (∃ w' b', 
    b' ≥ w' / 4 ∧ 
    b' ≤ r' / 6 ∧ 
    w' + b' ≥ 75 ∧ 
    r' < 90) → False :=
sorry

end minimum_red_chips_l1681_168114


namespace fruit_purchase_total_l1681_168174

/-- Calculates the total amount paid for fruits after discounts --/
def total_amount_paid (peach_price apple_price orange_price : ℚ)
                      (peach_count apple_count orange_count : ℕ)
                      (peach_discount apple_discount orange_discount : ℚ)
                      (peach_discount_threshold apple_discount_threshold orange_discount_threshold : ℚ) : ℚ :=
  let peach_total := peach_price * peach_count
  let apple_total := apple_price * apple_count
  let orange_total := orange_price * orange_count
  let peach_discount_applied := (peach_total / peach_discount_threshold).floor * peach_discount
  let apple_discount_applied := (apple_total / apple_discount_threshold).floor * apple_discount
  let orange_discount_applied := (orange_total / orange_discount_threshold).floor * orange_discount
  peach_total + apple_total + orange_total - peach_discount_applied - apple_discount_applied - orange_discount_applied

/-- Theorem stating the total amount paid for fruits after discounts --/
theorem fruit_purchase_total :
  total_amount_paid (40/100) (60/100) (50/100) 400 150 200 2 3 (3/2) 10 15 7 = 279 := by
  sorry

end fruit_purchase_total_l1681_168174


namespace smallest_n_for_special_function_l1681_168171

theorem smallest_n_for_special_function : ∃ (n : ℕ) (f : ℤ → Fin n),
  (∀ (A B : ℤ), |A - B| ∈ ({5, 7, 12} : Set ℤ) → f A ≠ f B) ∧
  (∀ (m : ℕ), m < n → ¬∃ (g : ℤ → Fin m), ∀ (A B : ℤ), |A - B| ∈ ({5, 7, 12} : Set ℤ) → g A ≠ g B) ∧
  n = 4 := by
  sorry

end smallest_n_for_special_function_l1681_168171


namespace min_savings_theorem_l1681_168197

/-- Represents Kathleen's savings and spending -/
structure KathleenFinances where
  june_savings : ℕ
  july_savings : ℕ
  august_savings : ℕ
  school_supplies_cost : ℕ
  clothes_cost : ℕ
  amount_left : ℕ

/-- The minimum amount Kathleen needs to save to get $25 from her aunt -/
def min_savings_for_bonus (k : KathleenFinances) : ℕ :=
  k.amount_left

theorem min_savings_theorem (k : KathleenFinances) 
  (h1 : k.june_savings = 21)
  (h2 : k.july_savings = 46)
  (h3 : k.august_savings = 45)
  (h4 : k.school_supplies_cost = 12)
  (h5 : k.clothes_cost = 54)
  (h6 : k.amount_left = 46)
  (h7 : k.june_savings + k.july_savings + k.august_savings - k.school_supplies_cost - k.clothes_cost = k.amount_left) :
  min_savings_for_bonus k = k.amount_left :=
by sorry

end min_savings_theorem_l1681_168197


namespace problem_solution_l1681_168119

-- Define the set D
def D : Set ℝ := {x | x < -4 ∨ x > 0}

-- Define proposition p
def p (a : ℝ) : Prop := a ∈ D

-- Define proposition q
def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 - a*x₀ - a ≤ -3

-- Theorem statement
theorem problem_solution :
  (∀ a : ℝ, q a → p a) ∧ (∃ a : ℝ, p a ∧ ¬q a) →
  D = {x : ℝ | x < -4 ∨ x > 0} :=
sorry

end problem_solution_l1681_168119


namespace union_of_M_and_N_l1681_168103

def M : Set Int := {m | -3 < m ∧ m < 2}
def N : Set Int := {n | -1 ≤ n ∧ n ≤ 3}

theorem union_of_M_and_N : M ∪ N = {-2, -1, 0, 1, 2, 3} := by sorry

end union_of_M_and_N_l1681_168103


namespace probability_next_queen_after_first_l1681_168179

/-- Represents a standard deck of 54 playing cards -/
def StandardDeck : ℕ := 54

/-- Number of queens in a standard deck -/
def QueenCount : ℕ := 4

/-- Probability of drawing a queen after the first queen -/
def ProbabilityNextQueenAfterFirst : ℚ := 2 / 27

/-- Theorem stating the probability of drawing a queen after the first queen -/
theorem probability_next_queen_after_first :
  ProbabilityNextQueenAfterFirst = QueenCount / StandardDeck :=
by
  sorry


end probability_next_queen_after_first_l1681_168179


namespace arithmetic_geometric_mean_inequality_l1681_168112

theorem arithmetic_geometric_mean_inequality 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a + b + c) / 3)^2 ≥ (a*b + b*c + c*a) / 3 := by
  sorry

end arithmetic_geometric_mean_inequality_l1681_168112


namespace train_length_train_length_proof_l1681_168157

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * time_s

/-- Proof that a train with speed 30 km/h crossing a pole in 9 seconds has a length of approximately 75 meters -/
theorem train_length_proof (ε : ℝ) (h_ε : ε > 0) :
  ∃ (l : ℝ), abs (l - train_length 30 9) < ε ∧ l = 75 := by
  sorry

end train_length_train_length_proof_l1681_168157


namespace problem_solution_l1681_168113

theorem problem_solution (m n : ℝ) 
  (h1 : m = (Real.sqrt (n^2 - 4) + Real.sqrt (4 - n^2) + 4) / (n - 2))
  (h2 : n^2 - 4 ≥ 0)
  (h3 : 4 - n^2 ≥ 0)
  (h4 : n ≠ 2) :
  |m - 2*n| + Real.sqrt (8*m*n) = 7 := by
sorry

end problem_solution_l1681_168113


namespace farmer_randy_planting_l1681_168117

/-- Calculates the number of acres each tractor needs to plant per day -/
def acres_per_tractor_per_day (total_acres : ℕ) (total_days : ℕ) 
  (tractors_first_period : ℕ) (days_first_period : ℕ)
  (tractors_second_period : ℕ) (days_second_period : ℕ) : ℚ :=
  total_acres / (tractors_first_period * days_first_period + 
                 tractors_second_period * days_second_period)

theorem farmer_randy_planting (total_acres : ℕ) (total_days : ℕ) 
  (tractors_first_period : ℕ) (days_first_period : ℕ)
  (tractors_second_period : ℕ) (days_second_period : ℕ) 
  (h1 : total_acres = 1700)
  (h2 : total_days = 5)
  (h3 : tractors_first_period = 2)
  (h4 : days_first_period = 2)
  (h5 : tractors_second_period = 7)
  (h6 : days_second_period = 3)
  (h7 : total_days = days_first_period + days_second_period) :
  acres_per_tractor_per_day total_acres total_days 
    tractors_first_period days_first_period
    tractors_second_period days_second_period = 68 := by
  sorry

#eval acres_per_tractor_per_day 1700 5 2 2 7 3

end farmer_randy_planting_l1681_168117


namespace counterexample_exists_l1681_168133

theorem counterexample_exists : ∃ a : ℝ, a > -2 ∧ ¬(a^2 > 4) :=
  ⟨0, by
    constructor
    · -- Prove 0 > -2
      sorry
    · -- Prove ¬(0^2 > 4)
      sorry⟩

#check counterexample_exists

end counterexample_exists_l1681_168133


namespace mixture_volume_proof_l1681_168158

/-- The initial volume of the mixture -/
def initial_volume : ℝ := 150

/-- The percentage of water in the initial mixture -/
def initial_water_percentage : ℝ := 0.15

/-- The volume of water added to the mixture -/
def added_water : ℝ := 20

/-- The percentage of water in the new mixture after adding water -/
def new_water_percentage : ℝ := 0.25

theorem mixture_volume_proof :
  initial_volume = 150 ∧
  initial_water_percentage * initial_volume + added_water = new_water_percentage * (initial_volume + added_water) :=
by sorry

end mixture_volume_proof_l1681_168158


namespace function_transformation_l1681_168121

theorem function_transformation (f : ℝ → ℝ) (h : f 1 = 3) : f (-(-1)) + 1 = 4 := by
  sorry

end function_transformation_l1681_168121


namespace function_satisfying_conditions_l1681_168142

theorem function_satisfying_conditions (f : ℝ → ℝ) 
  (h1 : ∀ x, f (f x * f (1 - x)) = f x) 
  (h2 : ∀ x, f (f x) = 1 - f x) : 
  ∀ x, f x = (1 : ℝ) / 2 := by
  sorry

end function_satisfying_conditions_l1681_168142


namespace right_triangle_max_ratio_l1681_168178

theorem right_triangle_max_ratio (k l a b c : ℝ) (hk : k > 0) (hl : l > 0) : 
  (k * a)^2 + (l * b)^2 = c^2 → (k * a + l * b) / c ≤ Real.sqrt 2 := by
  sorry

end right_triangle_max_ratio_l1681_168178


namespace composition_f_equals_one_over_e_l1681_168160

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.exp x else Real.log x

-- State the theorem
theorem composition_f_equals_one_over_e :
  f (f (1 / Real.exp 1)) = 1 / Real.exp 1 := by
  sorry

end composition_f_equals_one_over_e_l1681_168160


namespace fraction_cube_two_thirds_l1681_168136

theorem fraction_cube_two_thirds : (2 / 3 : ℚ) ^ 3 = 8 / 27 := by
  sorry

end fraction_cube_two_thirds_l1681_168136


namespace max_a_for_decreasing_cosine_minus_sine_l1681_168156

theorem max_a_for_decreasing_cosine_minus_sine :
  let f : ℝ → ℝ := λ x ↦ Real.cos x - Real.sin x
  ∀ a : ℝ, (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ a → f y < f x) →
    a ≤ 3 * Real.pi / 4 ∧ 
    ∃ b : ℝ, b > 3 * Real.pi / 4 ∧ ¬(∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ b → f y < f x) :=
by sorry

end max_a_for_decreasing_cosine_minus_sine_l1681_168156


namespace vector_orthogonality_l1681_168109

def a : ℝ × ℝ := (3, 1)
def b (x : ℝ) : ℝ × ℝ := (x, -2)
def c : ℝ × ℝ := (0, 2)

theorem vector_orthogonality (x : ℝ) :
  a • (b x - c) = 0 → x = 4/3 := by sorry

end vector_orthogonality_l1681_168109


namespace hiker_problem_l1681_168128

/-- A hiker's walking problem -/
theorem hiker_problem (h : ℕ) : 
  (3 * h) + (4 * (h - 1)) + 15 = 53 → 3 * h = 18 := by
  sorry

end hiker_problem_l1681_168128


namespace inequality_proof_l1681_168170

theorem inequality_proof (a b : ℝ) (h : a + b > 0) :
  a / b^2 + b / a^2 ≥ 1 / a + 1 / b :=
by sorry

end inequality_proof_l1681_168170


namespace factorization_equivalence_l1681_168154

variable (a x y : ℝ)

theorem factorization_equivalence : 
  (2*a*x^2 - 8*a*x*y + 8*a*y^2 = 2*a*(x - 2*y)^2) ∧ 
  (6*x*y^2 - 9*x^2*y - y^3 = -y*(3*x - y)^2) := by sorry

end factorization_equivalence_l1681_168154


namespace inequality_equivalence_l1681_168172

theorem inequality_equivalence (x y z : ℝ) :
  x + 3 * y + 2 * z = 6 →
  (x^2 + 9 * y^2 - 2 * x - 6 * y + 4 * z ≤ 8 ↔
   z = 3 - 1/2 * x - 3/2 * y ∧ (x - 2)^2 + (3 * y - 2)^2 ≤ 4 ∧ 0 ≤ x ∧ x ≤ 4) :=
by sorry

end inequality_equivalence_l1681_168172


namespace bowling_team_weight_l1681_168134

theorem bowling_team_weight (original_avg : ℝ) : 
  (7 * original_avg + 110 + 60) / 9 = 92 → original_avg = 94 := by
  sorry

end bowling_team_weight_l1681_168134


namespace sum_of_binary_digits_300_l1681_168151

/-- The sum of the digits in the binary representation of a natural number -/
def sum_of_binary_digits (n : ℕ) : ℕ :=
  (n.digits 2).sum

/-- Theorem: The sum of the digits in the binary representation of 300 is 4 -/
theorem sum_of_binary_digits_300 : sum_of_binary_digits 300 = 4 := by
  sorry

end sum_of_binary_digits_300_l1681_168151


namespace simplify_trig_expression_l1681_168181

theorem simplify_trig_expression (x : ℝ) :
  (1 + Real.sin x + Real.cos x) / (1 - Real.sin x + Real.cos x) = Real.tan (π / 4 + x / 2) :=
by sorry

end simplify_trig_expression_l1681_168181


namespace total_pencils_distributed_l1681_168120

/-- 
Given a teacher who distributes pencils equally among students, 
this theorem proves that the total number of pencils distributed 
is equal to the product of the number of students and the number 
of pencils each student receives.
-/
theorem total_pencils_distributed (num_students : ℕ) (pencils_per_student : ℕ) 
  (h1 : num_students = 12) 
  (h2 : pencils_per_student = 3) : 
  num_students * pencils_per_student = 36 := by
  sorry

#check total_pencils_distributed

end total_pencils_distributed_l1681_168120


namespace emily_unanswered_questions_l1681_168193

def total_questions : ℕ := 50
def new_score : ℕ := 120
def old_score : ℕ := 95

def scoring_systems (c w u : ℕ) : Prop :=
  (6 * c + u = new_score) ∧
  (50 + 3 * c - 2 * w = old_score) ∧
  (c + w + u = total_questions)

theorem emily_unanswered_questions :
  ∃ (c w u : ℕ), scoring_systems c w u ∧ u = 37 :=
by sorry

end emily_unanswered_questions_l1681_168193


namespace recipe_sugar_amount_l1681_168150

/-- The amount of sugar Katie has already put in the recipe -/
def sugar_already_added : ℝ := 0.5

/-- The amount of sugar Katie still needs to add to the recipe -/
def sugar_to_add : ℝ := 2.5

/-- The total amount of sugar required by the recipe -/
def total_sugar_needed : ℝ := sugar_already_added + sugar_to_add

theorem recipe_sugar_amount : total_sugar_needed = 3 := by
  sorry

end recipe_sugar_amount_l1681_168150


namespace two_distinct_roots_range_l1681_168132

theorem two_distinct_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - (m+2)*x - m + 1 = 0 ∧ y^2 - (m+2)*y - m + 1 = 0) ↔
  m < -8 ∨ m > 0 := by
sorry

end two_distinct_roots_range_l1681_168132


namespace mei_oranges_l1681_168184

theorem mei_oranges (peaches pears oranges baskets : ℕ) : 
  peaches = 9 →
  pears = 18 →
  baskets > 0 →
  peaches % baskets = 0 →
  pears % baskets = 0 →
  oranges % baskets = 0 →
  baskets = 3 →
  oranges = 9 :=
by sorry

end mei_oranges_l1681_168184


namespace fair_coin_probability_difference_l1681_168166

/-- The probability of getting exactly k heads in n flips of a fair coin -/
def binomialProbability (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ k * (1 / 2) ^ (n - k)

/-- The statement to prove -/
theorem fair_coin_probability_difference :
  (binomialProbability 3 2) - (binomialProbability 3 3) = 1 / 4 := by
  sorry

end fair_coin_probability_difference_l1681_168166


namespace gas_tank_fill_level_l1681_168100

theorem gas_tank_fill_level (tank_capacity : ℚ) (initial_fill_fraction : ℚ) (added_amount : ℚ) : 
  tank_capacity = 42 → 
  initial_fill_fraction = 3/4 → 
  added_amount = 7 → 
  (initial_fill_fraction * tank_capacity + added_amount) / tank_capacity = 833/909 := by
  sorry

end gas_tank_fill_level_l1681_168100


namespace parallel_vectors_condition_l1681_168192

/-- Given two vectors a and b in R², prove that if they are parallel and a = (-1, 3) and b = (1, t), then t = -3. -/
theorem parallel_vectors_condition (a b : ℝ × ℝ) (t : ℝ) : 
  a = (-1, 3) → b = (1, t) → (∃ (k : ℝ), a = k • b) → t = -3 := by
  sorry

end parallel_vectors_condition_l1681_168192


namespace two_digit_number_representation_l1681_168123

/-- Represents a two-digit number -/
def two_digit_number (x y : ℕ) : ℕ := 10 * x + y

/-- The tens digit of a two-digit number -/
def tens_digit (n : ℕ) : ℕ := n / 10

/-- The units digit of a two-digit number -/
def units_digit (n : ℕ) : ℕ := n % 10

theorem two_digit_number_representation (x y : ℕ) (h1 : x < 10) (h2 : y < 10) :
  two_digit_number x y = 10 * x + y :=
by sorry

end two_digit_number_representation_l1681_168123


namespace division_theorem_specific_case_l1681_168147

theorem division_theorem_specific_case :
  ∀ (D d Q R : ℕ),
    D = d * Q + R →
    d * Q = 135 →
    R = 2 * d →
    R < d →
    Q > 0 →
    D = 165 ∧ d = 15 ∧ Q = 9 ∧ R = 30 :=
by sorry

end division_theorem_specific_case_l1681_168147


namespace triangular_pyramid_angle_l1681_168168

/-- Represents a triangular pyramid with specific properties -/
structure TriangularPyramid where
  -- The length of the hypotenuse of the base triangle
  c : ℝ
  -- The volume of the pyramid
  V : ℝ
  -- All lateral edges form the same angle with the base plane
  lateral_angle_uniform : True
  -- This angle is equal to one of the acute angles of the right triangle in the base
  angle_matches_base : True
  -- Ensure c and V are positive
  c_pos : c > 0
  V_pos : V > 0

/-- 
Theorem: In a triangular pyramid where all lateral edges form the same angle α 
with the base plane, and this angle is equal to one of the acute angles of the 
right triangle in the base, if the hypotenuse of the base triangle is c and the 
volume of the pyramid is V, then α = arcsin(√(12V/c³)).
-/
theorem triangular_pyramid_angle (p : TriangularPyramid) : 
  ∃ α : ℝ, α = Real.arcsin (Real.sqrt (12 * p.V / p.c^3)) := by
  sorry

end triangular_pyramid_angle_l1681_168168


namespace last_match_wickets_specific_case_l1681_168125

/-- Represents a bowler's statistics -/
structure BowlerStats where
  initialAverage : ℝ
  initialWickets : ℕ
  lastMatchRuns : ℕ
  averageDecrease : ℝ

/-- Calculates the number of wickets taken in the last match -/
def lastMatchWickets (stats : BowlerStats) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, the number of wickets in the last match is 8 -/
theorem last_match_wickets_specific_case :
  let stats : BowlerStats := {
    initialAverage := 12.4,
    initialWickets := 175,
    lastMatchRuns := 26,
    averageDecrease := 0.4
  }
  lastMatchWickets stats = 8 := by sorry

end last_match_wickets_specific_case_l1681_168125


namespace comparison_theorem_l1681_168135

theorem comparison_theorem :
  (-3 / 4 : ℚ) < -2 / 3 ∧ (3 : ℤ) > -|4| := by
  sorry

end comparison_theorem_l1681_168135


namespace sports_club_membership_l1681_168182

theorem sports_club_membership (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ) :
  total = 42 →
  badminton = 20 →
  tennis = 23 →
  both = 7 →
  total - (badminton + tennis - both) = 6 :=
by sorry

end sports_club_membership_l1681_168182


namespace reciprocal_of_negative_seven_l1681_168185

theorem reciprocal_of_negative_seven :
  (1 : ℚ) / (-7 : ℚ) = -1/7 := by sorry

end reciprocal_of_negative_seven_l1681_168185


namespace parabola_intersection_sum_of_squares_l1681_168107

theorem parabola_intersection_sum_of_squares (k : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + k*x + 4 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  x₁^2 + x₂^2 > 8 :=
by sorry

end parabola_intersection_sum_of_squares_l1681_168107


namespace cos_72_minus_cos_144_l1681_168101

theorem cos_72_minus_cos_144 : Real.cos (72 * π / 180) - Real.cos (144 * π / 180) = 1.117962 := by
  sorry

end cos_72_minus_cos_144_l1681_168101


namespace min_buses_is_eleven_l1681_168165

/-- The maximum number of students a bus can hold -/
def bus_capacity : ℕ := 38

/-- The total number of students to be transported -/
def total_students : ℕ := 411

/-- The minimum number of buses needed is the ceiling of the division of total students by bus capacity -/
def min_buses : ℕ := (total_students + bus_capacity - 1) / bus_capacity

/-- Theorem stating that the minimum number of buses needed is 11 -/
theorem min_buses_is_eleven : min_buses = 11 := by sorry

end min_buses_is_eleven_l1681_168165


namespace fireworks_cost_and_remaining_l1681_168129

def small_firework_cost : ℕ := 12
def large_firework_cost : ℕ := 25

def henry_small : ℕ := 3
def henry_large : ℕ := 2
def friend_small : ℕ := 4
def friend_large : ℕ := 1

def saved_fireworks : ℕ := 6
def used_saved_fireworks : ℕ := 3

theorem fireworks_cost_and_remaining :
  (let total_cost := (henry_small + friend_small) * small_firework_cost +
                     (henry_large + friend_large) * large_firework_cost
   let remaining_fireworks := henry_small + henry_large + friend_small + friend_large +
                              (saved_fireworks - used_saved_fireworks)
   (total_cost = 159) ∧ (remaining_fireworks = 13)) := by
  sorry

end fireworks_cost_and_remaining_l1681_168129


namespace floor_times_self_equals_45_l1681_168198

theorem floor_times_self_equals_45 (y : ℝ) (h1 : y > 0) (h2 : ⌊y⌋ * y = 45) : y = 7.5 := by
  sorry

end floor_times_self_equals_45_l1681_168198


namespace alloy_mixture_l1681_168115

/-- Given two alloys with metal ratios m:n and p:q respectively, 
    this theorem proves the amounts of each alloy needed to create 1 kg 
    of a new alloy with equal parts of both metals. -/
theorem alloy_mixture (m n p q : ℝ) (hm : m > 0) (hn : n > 0) (hp : p > 0) (hq : q > 0) :
  let x := (1 : ℝ) / 2 + (m * p - n * q) / (2 * (n * p - m * q))
  x * (n / (m + n)) + (1 - x) * (p / (p + q)) = 
  x * (m / (m + n)) + (1 - x) * (q / (p + q)) :=
by sorry

end alloy_mixture_l1681_168115


namespace collinear_points_sum_l1681_168163

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (p q r : ℝ × ℝ × ℝ) : Prop := sorry

/-- The theorem states that if the given points are collinear, then a + b = 6. -/
theorem collinear_points_sum (a b : ℝ) : 
  collinear (2, a, b) (a, 3, b) (a, b, 4) → a + b = 6 := by
  sorry

end collinear_points_sum_l1681_168163


namespace base_twelve_equality_l1681_168183

/-- Given a base b, this function converts a number in base b to decimal --/
def toDecimal (digits : List Nat) (b : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The proposition states that in base b, 35₍ᵦ₎² equals 1331₍ᵦ₎, and b equals 12 --/
theorem base_twelve_equality : ∃ b : Nat, 
  b > 1 ∧ 
  (toDecimal [3, 5] b)^2 = toDecimal [1, 3, 3, 1] b ∧ 
  b = 12 := by
  sorry

end base_twelve_equality_l1681_168183


namespace problem_statement_l1681_168194

theorem problem_statement (x y : ℝ) (h : |x - 1/2| + Real.sqrt (y^2 - 1) = 0) : 
  |x| + |y| = 3/2 := by
sorry

end problem_statement_l1681_168194


namespace specific_solid_volume_l1681_168188

/-- A solid with a square base and specific edge lengths -/
structure Solid where
  s : ℝ
  base_side_length : s > 0
  upper_edge_length : ℝ
  upper_edge_parallel : upper_edge_length = 3 * s
  other_edges_length : ℝ
  other_edges_equal_s : other_edges_length = s

/-- The volume of the solid -/
noncomputable def volume (solid : Solid) : ℝ := sorry

/-- Theorem stating the volume of the specific solid -/
theorem specific_solid_volume :
  ∀ (solid : Solid),
    solid.s = 4 * Real.sqrt 2 →
    volume solid = 144 * Real.sqrt 2 := by
  sorry

end specific_solid_volume_l1681_168188


namespace min_f_1998_l1681_168190

theorem min_f_1998 (f : ℕ → ℕ) 
  (h : ∀ s t : ℕ, f (t^2 * f s) = s * (f t)^2) : 
  f 1998 ≥ 1998 := by
  sorry

end min_f_1998_l1681_168190


namespace popcorn_shrimp_orders_l1681_168159

theorem popcorn_shrimp_orders (catfish_price popcorn_price : ℚ)
  (total_orders : ℕ) (total_amount : ℚ)
  (h1 : catfish_price = 6)
  (h2 : popcorn_price = (7/2))
  (h3 : total_orders = 26)
  (h4 : total_amount = (267/2)) :
  ∃ (catfish_orders popcorn_orders : ℕ),
    catfish_orders + popcorn_orders = total_orders ∧
    catfish_price * catfish_orders + popcorn_price * popcorn_orders = total_amount ∧
    popcorn_orders = 9 := by
  sorry

end popcorn_shrimp_orders_l1681_168159


namespace drums_per_day_l1681_168104

/-- Given that 2916 drums of grapes are filled in 9 days, 
    prove that 324 drums of grapes are filled per day. -/
theorem drums_per_day (total_drums : ℕ) (total_days : ℕ) 
  (h1 : total_drums = 2916) (h2 : total_days = 9) :
  total_drums / total_days = 324 := by
  sorry

end drums_per_day_l1681_168104


namespace constant_term_of_expansion_l1681_168169

/-- The constant term in the expansion of (9x + 2/(3x))^8 -/
def constant_term : ℕ := 90720

/-- The binomial coefficient (8 choose 4) -/
def binomial_8_4 : ℕ := 70

theorem constant_term_of_expansion :
  constant_term = binomial_8_4 * 9^4 * 2^4 / 3^4 :=
by sorry

end constant_term_of_expansion_l1681_168169


namespace max_sum_abc_l1681_168118

/-- Definition of An as an n-digit number with all digits equal to a -/
def An (a : ℕ) (n : ℕ) : ℕ := a * (10^n - 1) / 9

/-- Definition of Bn as an n-digit number with all digits equal to b -/
def Bn (b : ℕ) (n : ℕ) : ℕ := b * (10^n - 1) / 9

/-- Definition of Cn as a 2n-digit number with all digits equal to c -/
def Cn (c : ℕ) (n : ℕ) : ℕ := c * (10^(2*n) - 1) / 9

/-- The main theorem stating that the maximum value of a + b + c is 18 -/
theorem max_sum_abc :
  ∃ (a b c : ℕ),
    (0 < a ∧ a ≤ 9) ∧
    (0 < b ∧ b ≤ 9) ∧
    (0 < c ∧ c ≤ 9) ∧
    (∃ (n₁ n₂ : ℕ), n₁ ≠ n₂ ∧ Cn c n₁ - Bn b n₁ = (An a n₁)^2 ∧ Cn c n₂ - Bn b n₂ = (An a n₂)^2) ∧
    a + b + c = 18 ∧
    ∀ (a' b' c' : ℕ),
      (0 < a' ∧ a' ≤ 9) →
      (0 < b' ∧ b' ≤ 9) →
      (0 < c' ∧ c' ≤ 9) →
      (∃ (n₁ n₂ : ℕ), n₁ ≠ n₂ ∧ Cn c' n₁ - Bn b' n₁ = (An a' n₁)^2 ∧ Cn c' n₂ - Bn b' n₂ = (An a' n₂)^2) →
      a' + b' + c' ≤ 18 :=
by sorry

end max_sum_abc_l1681_168118


namespace fraction_product_simplification_l1681_168164

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end fraction_product_simplification_l1681_168164


namespace bookshop_inventory_l1681_168177

/-- Calculates the final number of books in a bookshop after weekend sales and a new shipment --/
theorem bookshop_inventory (initial_inventory : ℕ) (saturday_in_store : ℕ) (saturday_online : ℕ) (sunday_in_store_multiplier : ℕ) (sunday_online_increase : ℕ) (new_shipment : ℕ) : 
  initial_inventory = 743 →
  saturday_in_store = 37 →
  saturday_online = 128 →
  sunday_in_store_multiplier = 2 →
  sunday_online_increase = 34 →
  new_shipment = 160 →
  initial_inventory - 
    (saturday_in_store + saturday_online + 
     sunday_in_store_multiplier * saturday_in_store + 
     (saturday_online + sunday_online_increase)) + 
  new_shipment = 502 := by
sorry

end bookshop_inventory_l1681_168177


namespace repeating_block_length_l1681_168146

/-- The number of digits in the smallest repeating block of the decimal expansion of 4/7 -/
def smallest_repeating_block_length : ℕ := 6

/-- The fraction we're considering -/
def fraction : ℚ := 4/7

theorem repeating_block_length :
  smallest_repeating_block_length = 6 ∧ 
  ∃ (n : ℕ) (d : ℕ+), fraction = n / d ∧ 
  smallest_repeating_block_length ≤ d - 1 :=
sorry

end repeating_block_length_l1681_168146
