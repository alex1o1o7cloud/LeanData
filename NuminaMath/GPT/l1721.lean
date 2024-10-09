import Mathlib

namespace probability_of_yellow_face_l1721_172153

theorem probability_of_yellow_face :
  let total_faces : ℕ := 10
  let yellow_faces : ℕ := 4
  (yellow_faces : ℚ) / (total_faces : ℚ) = 2 / 5 :=
by
  sorry

end probability_of_yellow_face_l1721_172153


namespace inequality_correctness_l1721_172119

theorem inequality_correctness (a b : ℝ) (h : a < b) (h₀ : b < 0) : - (1 / a) < - (1 / b) :=
sorry

end inequality_correctness_l1721_172119


namespace sin_alpha_value_l1721_172114

-- Given conditions
variables (α : ℝ) (h1 : Real.tan α = -5 / 12) (h2 : π / 2 < α ∧ α < π)

-- Assertion to prove
theorem sin_alpha_value : Real.sin α = 5 / 13 :=
by
  -- Proof goes here
  sorry

end sin_alpha_value_l1721_172114


namespace find_number_l1721_172161

def sum : ℕ := 2468 + 1375
def diff : ℕ := 2468 - 1375
def first_quotient : ℕ := 3 * diff
def second_quotient : ℕ := 5 * diff
def remainder : ℕ := 150

theorem find_number (N : ℕ) (h1 : sum = 3843) (h2 : diff = 1093) 
                    (h3 : first_quotient = 3279) (h4 : second_quotient = 5465)
                    (h5 : remainder = 150) (h6 : N = sum * first_quotient + remainder)
                    (h7 : N = sum * second_quotient + remainder) :
  N = 12609027 := 
by 
  sorry

end find_number_l1721_172161


namespace five_digit_palindromes_count_l1721_172170

theorem five_digit_palindromes_count : 
  ∃ (a b c : Fin 10), (a ≠ 0) ∧ (∃ (count : Nat), count = 9 * 10 * 10 ∧ count = 900) :=
by
  sorry

end five_digit_palindromes_count_l1721_172170


namespace correct_proposition_D_l1721_172121

theorem correct_proposition_D (a b : ℝ) (h1 : a < 0) (h2 : b < 0) : 
  (b / a) + (a / b) ≥ 2 := 
sorry

end correct_proposition_D_l1721_172121


namespace annular_region_area_l1721_172103

noncomputable def area_annulus (r1 r2 : ℝ) : ℝ :=
  (Real.pi * r2 ^ 2) - (Real.pi * r1 ^ 2)

theorem annular_region_area :
  area_annulus 4 7 = 33 * Real.pi :=
by 
  sorry

end annular_region_area_l1721_172103


namespace possibleValues_set_l1721_172197

noncomputable def possibleValues (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_sum : a + b = 3) : Set ℝ :=
  {x | x = 1/a + 1/b}

theorem possibleValues_set :
  ∀ a b : ℝ, (0 < a ∧ 0 < b) → (a + b = 3) → possibleValues a b (by sorry) (by sorry) = {x | ∃ y, y ≥ 4/3 ∧ x = y} :=
by
  sorry

end possibleValues_set_l1721_172197


namespace find_wsquared_l1721_172128

theorem find_wsquared : 
  (2 * w + 10) ^ 2 = (5 * w + 15) * (w + 6) →
  w ^ 2 = (90 + 10 * Real.sqrt 65) / 4 := 
by 
  intro h₀
  sorry

end find_wsquared_l1721_172128


namespace perfect_square_trinomial_l1721_172160

theorem perfect_square_trinomial (y : ℝ) (m : ℝ) : 
  (∃ b : ℝ, y^2 - m*y + 9 = (y + b)^2) → (m = 6 ∨ m = -6) :=
by
  intro h
  sorry

end perfect_square_trinomial_l1721_172160


namespace johns_age_l1721_172190

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
sorry

end johns_age_l1721_172190


namespace square_area_with_circles_l1721_172162

theorem square_area_with_circles
  (radius : ℝ) 
  (side_length : ℝ)
  (area : ℝ)
  (h_radius : radius = 7) 
  (h_side_length : side_length = 2 * (2 * radius)) 
  (h_area : area = side_length ^ 2) : 
  area = 784 := by
  sorry

end square_area_with_circles_l1721_172162


namespace hall_length_width_difference_l1721_172131

theorem hall_length_width_difference (L W : ℝ) 
(h1 : W = 1 / 2 * L) 
(h2 : L * W = 200) : L - W = 10 := 
by 
  sorry

end hall_length_width_difference_l1721_172131


namespace find_y_l1721_172173

variable (x y z : ℚ)

theorem find_y
  (h1 : x + y + z = 150)
  (h2 : x + 7 = y - 12)
  (h3 : x + 7 = 4 * z) :
  y = 688 / 9 :=
sorry

end find_y_l1721_172173


namespace merchant_mixture_solution_l1721_172133

variable (P C : ℝ)

def P_price : ℝ := 2.40
def C_price : ℝ := 6.00
def total_weight : ℝ := 60
def total_price_per_pound : ℝ := 3.00
def total_price : ℝ := total_price_per_pound * total_weight

theorem merchant_mixture_solution (h1 : P + C = total_weight)
                                  (h2 : P_price * P + C_price * C = total_price) :
  C = 10 := 
sorry

end merchant_mixture_solution_l1721_172133


namespace max_souls_guaranteed_l1721_172184

def initial_nuts : ℕ := 1001

def valid_N (N : ℕ) : Prop :=
  1 ≤ N ∧ N ≤ 1001

def nuts_transferred (N : ℕ) (T : ℕ) : Prop :=
  valid_N N ∧ T ≤ 71

theorem max_souls_guaranteed : (∀ N, valid_N N → ∃ T, nuts_transferred N T) :=
sorry

end max_souls_guaranteed_l1721_172184


namespace max_distance_proof_l1721_172100

-- Definitions for fuel consumption rates per 100 km
def fuel_consumption_U : Nat := 20 -- liters per 100 km
def fuel_consumption_V : Nat := 25 -- liters per 100 km
def fuel_consumption_W : Nat := 5  -- liters per 100 km
def fuel_consumption_X : Nat := 10 -- liters per 100 km

-- Definitions for total available fuel
def total_fuel : Nat := 50 -- liters

-- Distance calculation
def distance (fuel_consumption : Nat) (fuel : Nat) : Nat :=
  (fuel * 100) / fuel_consumption

-- Distances
def distance_U := distance fuel_consumption_U total_fuel
def distance_V := distance fuel_consumption_V total_fuel
def distance_W := distance fuel_consumption_W total_fuel
def distance_X := distance fuel_consumption_X total_fuel

-- Maximum total distance calculation
def maximum_total_distance : Nat :=
  distance_U + distance_V + distance_W + distance_X

-- The statement to be proved
theorem max_distance_proof :
  maximum_total_distance = 1950 := by
  sorry

end max_distance_proof_l1721_172100


namespace total_apple_trees_is_800_l1721_172181

variable (T P A : ℕ) -- Total number of trees, peach trees, and apple trees respectively
variable (samples_peach samples_apple : ℕ) -- Sampled peach trees and apple trees respectively
variable (sampled_percentage : ℕ) -- Percentage of total trees sampled

-- Given conditions
axiom H1 : sampled_percentage = 10
axiom H2 : samples_peach = 50
axiom H3 : samples_apple = 80

-- Theorem to prove the number of apple trees
theorem total_apple_trees_is_800 : A = 800 :=
by sorry

end total_apple_trees_is_800_l1721_172181


namespace drug_price_reduction_l1721_172148

theorem drug_price_reduction (x : ℝ) :
    36 * (1 - x)^2 = 25 :=
sorry

end drug_price_reduction_l1721_172148


namespace vacuum_tube_pins_and_holes_l1721_172116

theorem vacuum_tube_pins_and_holes :
  ∀ (pins holes : Finset ℕ), 
  pins = {1, 2, 3, 4, 5, 6, 7} →
  holes = {1, 2, 3, 4, 5, 6, 7} →
  (∃ (a : ℕ), ∀ k ∈ pins, ∃ b ∈ holes, (2 * k) % 7 = b) := by
  sorry

end vacuum_tube_pins_and_holes_l1721_172116


namespace rationalize_denominator_correct_l1721_172191

noncomputable def rationalize_denominator : ℚ := 
  let A := 5
  let B := 49
  let C := 21
  -- Form is (5 * ∛49) / 21
  A + B + C

theorem rationalize_denominator_correct : rationalize_denominator = 75 :=
  by 
    -- The proof steps are omitted, as they are not required for this task
    sorry

end rationalize_denominator_correct_l1721_172191


namespace no_nontrivial_solutions_l1721_172124

theorem no_nontrivial_solutions :
  ∀ (x y z t : ℤ), (¬(x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0)) → ¬(x^2 = 2 * y^2 ∧ x^4 + 3 * y^4 + 27 * z^4 = 9 * t^4) :=
by
  intros x y z t h_nontrivial h_eqs
  sorry

end no_nontrivial_solutions_l1721_172124


namespace fraction_of_work_left_correct_l1721_172138

-- Define the conditions for p, q, and r
def p_one_day_work : ℚ := 1 / 15
def q_one_day_work : ℚ := 1 / 20
def r_one_day_work : ℚ := 1 / 30

-- Define the total work done in one day by p, q, and r
def total_one_day_work : ℚ := p_one_day_work + q_one_day_work + r_one_day_work

-- Define the work done in 4 days
def work_done_in_4_days : ℚ := total_one_day_work * 4

-- Define the fraction of work left after 4 days
def fraction_of_work_left : ℚ := 1 - work_done_in_4_days

-- Statement to prove
theorem fraction_of_work_left_correct : fraction_of_work_left = 2 / 5 := by
  sorry

end fraction_of_work_left_correct_l1721_172138


namespace determinant_zero_l1721_172174

noncomputable def matrix_A (θ φ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 2 * Real.sin θ, -Real.cos θ],
    ![-2 * Real.sin θ, 0, Real.sin φ],
    ![Real.cos θ, -Real.sin φ, 0]]

theorem determinant_zero (θ φ : ℝ) : Matrix.det (matrix_A θ φ) = 0 := by
  sorry

end determinant_zero_l1721_172174


namespace number_of_blue_candles_l1721_172144

def total_candles : ℕ := 79
def yellow_candles : ℕ := 27
def red_candles : ℕ := 14
def blue_candles : ℕ := total_candles - (yellow_candles + red_candles)

theorem number_of_blue_candles : blue_candles = 38 :=
by
  unfold blue_candles
  unfold total_candles yellow_candles red_candles
  sorry

end number_of_blue_candles_l1721_172144


namespace mary_paid_amount_l1721_172123

-- Definitions for the conditions:
def is_adult (person : String) : Prop := person = "Mary"
def children_count (n : ℕ) : Prop := n = 3
def ticket_cost_adult : ℕ := 2  -- $2 for adults
def ticket_cost_child : ℕ := 1  -- $1 for children
def change_received : ℕ := 15   -- $15 change

-- Mathematical proof to find the amount Mary paid given the conditions
theorem mary_paid_amount (person : String) (n : ℕ) 
  (h1 : is_adult person) (h2 : children_count n) :
  ticket_cost_adult + ticket_cost_child * n + change_received = 20 := 
by 
  -- Sorry as the proof is not required
  sorry

end mary_paid_amount_l1721_172123


namespace ambulance_ride_cost_is_correct_l1721_172168

-- Define all the constants and conditions
def daily_bed_cost : ℝ := 900
def bed_days : ℕ := 3
def specialist_rate_per_hour : ℝ := 250
def specialist_minutes_per_day : ℕ := 15
def specialists_count : ℕ := 2
def total_bill : ℝ := 4625

noncomputable def ambulance_cost : ℝ :=
  total_bill - ((daily_bed_cost * bed_days) + (specialist_rate_per_hour * (specialist_minutes_per_day / 60) * specialists_count))

-- The proof statement
theorem ambulance_ride_cost_is_correct : ambulance_cost = 1675 := by
  sorry

end ambulance_ride_cost_is_correct_l1721_172168


namespace money_difference_l1721_172145

-- Given conditions
def packs_per_hour_peak : Nat := 6
def packs_per_hour_low : Nat := 4
def price_per_pack : Nat := 60
def hours_per_day : Nat := 15

-- Calculate total sales in peak and low seasons
def total_sales_peak : Nat :=
  packs_per_hour_peak * price_per_pack * hours_per_day

def total_sales_low : Nat :=
  packs_per_hour_low * price_per_pack * hours_per_day

-- The Lean statement proving the correct answer
theorem money_difference :
  total_sales_peak - total_sales_low = 1800 :=
by
  sorry

end money_difference_l1721_172145


namespace num_baskets_l1721_172126

axiom num_apples_each_basket : ℕ
axiom total_apples : ℕ

theorem num_baskets (h1 : num_apples_each_basket = 17) (h2 : total_apples = 629) : total_apples / num_apples_each_basket = 37 :=
  sorry

end num_baskets_l1721_172126


namespace inscribable_quadrilateral_l1721_172182

theorem inscribable_quadrilateral
  (a b c d : ℝ)
  (A : ℝ)
  (circumscribable : Prop)
  (area_condition : A = Real.sqrt (a * b * c * d))
  (A := Real.sqrt (a * b * c * d)) : 
  circumscribable → ∃ B D : ℝ, B + D = 180 :=
sorry

end inscribable_quadrilateral_l1721_172182


namespace largest_x_value_l1721_172163

noncomputable def quadratic_eq (x : ℝ) : Prop :=
  3 * (9 * x^2 + 15 * x + 20) = x * (9 * x - 60)

theorem largest_x_value (x : ℝ) :
  quadratic_eq x → x = - ((35 - Real.sqrt 745) / 12) ∨
  x = - ((35 + Real.sqrt 745) / 12) :=
by
  intro h
  sorry

end largest_x_value_l1721_172163


namespace parenthesis_removal_correctness_l1721_172125

theorem parenthesis_removal_correctness (x y z : ℝ) : 
  (x^2 - (x - y + 2 * z) ≠ x^2 - x + y - 2 * z) ∧
  (x - (-2 * x + 3 * y - 1) ≠ x + 2 * x - 3 * y + 1) ∧
  (3 * x + 2 * (x - 2 * y + 1) ≠ 3 * x + 2 * x - 4 * y + 2) ∧
  (-(x - 2) - 2 * (x^2 + 2) = -x + 2 - 2 * x^2 - 4) :=
by
  sorry

end parenthesis_removal_correctness_l1721_172125


namespace remainder_theorem_example_l1721_172117

def polynomial (x : ℝ) : ℝ := x^15 + 3

theorem remainder_theorem_example :
  polynomial (-2) = -32765 :=
by
  -- Substitute x = -2 in the polynomial and show the remainder is -32765
  sorry

end remainder_theorem_example_l1721_172117


namespace remainder_div_14_l1721_172112

variables (x k : ℕ)

theorem remainder_div_14 (h : x = 142 * k + 110) : x % 14 = 12 := by 
  sorry

end remainder_div_14_l1721_172112


namespace largest_even_among_consecutives_l1721_172107

theorem largest_even_among_consecutives (x : ℤ) (h : (x + (x + 2) + (x + 4) = x + 18)) : x + 4 = 10 :=
by
  sorry

end largest_even_among_consecutives_l1721_172107


namespace man_climbing_out_of_well_l1721_172175

theorem man_climbing_out_of_well (depth climb slip : ℕ) (h1 : depth = 30) (h2 : climb = 4) (h3 : slip = 3) : 
  let effective_climb_per_day := climb - slip
  let total_days := if depth % effective_climb_per_day = 0 then depth / effective_climb_per_day else depth / effective_climb_per_day + 1
  total_days = 30 :=
by
  sorry

end man_climbing_out_of_well_l1721_172175


namespace geometric_sequence_min_value_l1721_172151

theorem geometric_sequence_min_value
  (a : ℕ → ℝ)
  (h1 : ∀ n, 0 < a n) 
  (h2 : a 9 = 9 * a 7)
  (exists_m_n : ∃ m n, a m * a n = 9 * (a 1)^2):
  ∀ m n, (m + n = 4) → (1 / m + 9 / n) ≥ 4 :=
by
  intros m n h
  sorry

end geometric_sequence_min_value_l1721_172151


namespace find_a_l1721_172142

noncomputable def A (a : ℝ) : Set ℝ := { x | a * x - 1 = 0 }
def B : Set ℝ := { x | x^2 - 3 * x + 2 = 0 }

theorem find_a (a : ℝ) :
  A a ∪ B = B ↔ a = 0 ∨ a = 1 ∨ a = 1 / 2 :=
sorry

end find_a_l1721_172142


namespace base_five_product_l1721_172152

theorem base_five_product (n1 n2 : ℕ) (h1 : n1 = 1 * 5^2 + 3 * 5^1 + 1 * 5^0) 
                          (h2 : n2 = 1 * 5^1 + 2 * 5^0) :
  let product_dec := (n1 * n2 : ℕ)
  let product_base5 := 2 * 125 + 1 * 25 + 2 * 5 + 2 * 1
  product_dec = 287 ∧ product_base5 = 2122 := by
                                -- calculations to verify statement omitted
                                sorry

end base_five_product_l1721_172152


namespace relationship_abc_l1721_172137

noncomputable def a : ℝ := 4 / 5
noncomputable def b : ℝ := Real.sin (2 / 3)
noncomputable def c : ℝ := Real.cos (1 / 3)

theorem relationship_abc : b < a ∧ a < c := by
  sorry

end relationship_abc_l1721_172137


namespace sequence_formula_l1721_172165

theorem sequence_formula (a : ℕ → ℚ) (n : ℕ) (h1 : a 1 = 1) (h2 : a 2 = -1/2) (h3 : a 3 = 1/3) (h4 : a 4 = -1/4) :
  a n = (-1)^(n+1) * (1/n) :=
sorry

end sequence_formula_l1721_172165


namespace red_light_adds_3_minutes_l1721_172143

-- Definitions (conditions)
def first_route_time_if_all_green := 10
def second_route_time := 14
def additional_time_if_all_red := 5

-- Given that the first route is 5 minutes longer when all stoplights are red
def first_route_time_if_all_red := second_route_time + additional_time_if_all_red

-- Define red_light_time as the time each stoplight adds if it is red
def red_light_time := (first_route_time_if_all_red - first_route_time_if_all_green) / 3

-- Theorem (question == answer)
theorem red_light_adds_3_minutes :
  red_light_time = 3 :=
by
  -- proof goes here
  sorry

end red_light_adds_3_minutes_l1721_172143


namespace isosceles_triangle_angles_sum_l1721_172159

theorem isosceles_triangle_angles_sum (x : ℝ) 
  (h_triangle_sum : ∀ a b c : ℝ, a + b + c = 180)
  (h_isosceles : ∃ a b : ℝ, (a = 50 ∧ b = x) ∨ (a = x ∧ b = 50)) :
  50 + x + (180 - 50 * 2) + 65 + 80 = 195 :=
by
  sorry

end isosceles_triangle_angles_sum_l1721_172159


namespace number_of_rectangles_l1721_172177

theorem number_of_rectangles (a b : ℝ) (ha_lt_b : a < b) :
  ∃! (x y : ℝ), (x < b ∧ y < b ∧ 2 * (x + y) = a + b ∧ x * y = (a * b) / 4) := 
sorry

end number_of_rectangles_l1721_172177


namespace positive_integer_solutions_l1721_172178

theorem positive_integer_solutions (n x y z t : ℕ) (h_n : n > 0) (h_n_neq_1 : n ≠ 1) (h_x : x > 0) (h_y : y > 0) (h_z : z > 0) (h_t : t > 0) :
  (n ^ x ∣ n ^ y + n ^ z ↔ n ^ x = n ^ t) →
  ((n = 2 ∧ y = x ∧ z = x + 1 ∧ t = x + 2) ∨ (n = 3 ∧ y = x ∧ z = x ∧ t = x + 1)) :=
by
  sorry

end positive_integer_solutions_l1721_172178


namespace smallest_scalene_triangle_perimeter_l1721_172172

-- Define what it means for a number to be a prime number greater than 3
def prime_gt_3 (n : ℕ) : Prop := Prime n ∧ 3 < n

-- Define the main theorem
theorem smallest_scalene_triangle_perimeter : ∃ (a b c : ℕ), 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  prime_gt_3 a ∧ prime_gt_3 b ∧ prime_gt_3 c ∧
  Prime (a + b + c) ∧ 
  (∀ (x y z : ℕ), 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    prime_gt_3 x ∧ prime_gt_3 y ∧ prime_gt_3 z ∧
    Prime (x + y + z) → (a + b + c) ≤ (x + y + z)) ∧
  a + b + c = 23 := by
    sorry

end smallest_scalene_triangle_perimeter_l1721_172172


namespace fully_loaded_truck_weight_l1721_172105

def empty_truck : ℕ := 12000
def weight_soda_crate : ℕ := 50
def num_soda_crates : ℕ := 20
def weight_dryer : ℕ := 3000
def num_dryers : ℕ := 3
def weight_fresh_produce : ℕ := 2 * (weight_soda_crate * num_soda_crates)

def total_loaded_truck_weight : ℕ :=
  empty_truck + (weight_soda_crate * num_soda_crates) + weight_fresh_produce + (weight_dryer * num_dryers)

theorem fully_loaded_truck_weight : total_loaded_truck_weight = 24000 := by
  sorry

end fully_loaded_truck_weight_l1721_172105


namespace percentage_increase_mario_salary_is_zero_l1721_172167

variable (M : ℝ) -- Mario's salary last year
variable (P : ℝ) -- Percentage increase in Mario's salary

-- Condition 1: Mario's salary increased to $4000 this year
def mario_salary_increase (M P : ℝ) : Prop :=
  M * (1 + P / 100) = 4000 

-- Condition 2: Bob's salary last year was 3 times Mario's salary this year
def bob_salary_last_year (M : ℝ) : Prop :=
  3 * 4000 = 12000 

-- Condition 3: Bob's current salary is 20% more than his salary last year
def bob_current_salary : Prop :=
  12000 * 1.2 = 14400

-- Theorem : The percentage increase in Mario's salary is 0%
theorem percentage_increase_mario_salary_is_zero
  (h1 : mario_salary_increase M P)
  (h2 : bob_salary_last_year M)
  (h3 : bob_current_salary) : 
  P = 0 := 
sorry

end percentage_increase_mario_salary_is_zero_l1721_172167


namespace smallest_n_div_75_eq_432_l1721_172194

theorem smallest_n_div_75_eq_432 :
  ∃ n k : ℕ, (n ∣ 75 ∧ (∃ (d : ℕ), d ∣ n → d ≠ 1 → d ≠ n → n = 75 * k ∧ ∀ x: ℕ, (x ∣ n) → (x ≠ 1 ∧ x ≠ n) → False)) → ( k =  432 ) :=
by
  sorry

end smallest_n_div_75_eq_432_l1721_172194


namespace equilateral_triangle_condition_l1721_172146

-- We define points in a plane and vectors between these points
structure Point where
  x : ℝ
  y : ℝ

-- Vector subtraction
def vector (p q : Point) : Point :=
  { x := q.x - p.x, y := q.y - p.y }

-- The equation required to hold for certain type of triangles
def bisector_eq_zero (A B C A1 B1 C1 : Point) : Prop :=
  let AA1 := vector A A1
  let BB1 := vector B B1
  let CC1 := vector C C1
  AA1.x + BB1.x + CC1.x = 0 ∧ AA1.y + BB1.y + CC1.y = 0

-- Property of equilateral triangle
def is_equilateral (A B C : Point) : Prop :=
  let AB := vector A B
  let BC := vector B C
  let CA := vector C A
  (AB.x^2 + AB.y^2 = BC.x^2 + BC.y^2 ∧ BC.x^2 + BC.y^2 = CA.x^2 + CA.y^2)

-- Main theorem statement
theorem equilateral_triangle_condition (A B C A1 B1 C1 : Point)
  (h : bisector_eq_zero A B C A1 B1 C1) :
  is_equilateral A B C :=
sorry

end equilateral_triangle_condition_l1721_172146


namespace integer_value_of_expression_l1721_172102

theorem integer_value_of_expression (m n p : ℕ) (h1 : 2 ≤ m) (h2 : m ≤ 9)
  (h3 : 2 ≤ n) (h4 : n ≤ 9) (h5 : 2 ≤ p) (h6 : p ≤ 9)
  (h7 : m ≠ n ∧ n ≠ p ∧ m ≠ p) :
  (m + n + p) / (m + n) = 1 :=
sorry

end integer_value_of_expression_l1721_172102


namespace five_term_geometric_sequence_value_of_b_l1721_172110

theorem five_term_geometric_sequence_value_of_b (a b c : ℝ) (h₁ : b ^ 2 = 81) (h₂ : a ^ 2 = b) (h₃ : 1 * a = a) (h₄ : c * c = c) :
  b = 9 :=
by 
  sorry

end five_term_geometric_sequence_value_of_b_l1721_172110


namespace original_price_is_100_l1721_172176

variable (P : ℝ) -- Declare the original price P as a real number
variable (h : 0.10 * P = 10) -- The condition given in the problem

theorem original_price_is_100 (P : ℝ) (h : 0.10 * P = 10) : P = 100 := by
  sorry

end original_price_is_100_l1721_172176


namespace cost_of_song_book_l1721_172135

-- Definitions of the constants:
def cost_of_flute : ℝ := 142.46
def cost_of_music_stand : ℝ := 8.89
def total_spent : ℝ := 158.35

-- Definition of the combined cost of the flute and music stand:
def combined_cost := cost_of_flute + cost_of_music_stand

-- The final theorem to prove that the cost of the song book is $7.00:
theorem cost_of_song_book : total_spent - combined_cost = 7.00 := by
  sorry

end cost_of_song_book_l1721_172135


namespace second_rectangle_area_l1721_172118

theorem second_rectangle_area (b h x : ℝ) (hb : 0 < b) (hh : 0 < h) (hx : 0 < x) (hbx : x < h):
  2 * b * x * (h - 3 * x) / h = (2 * b * x * (h - 3 * x))/h := 
sorry

end second_rectangle_area_l1721_172118


namespace right_triangle_cosine_l1721_172192

theorem right_triangle_cosine (XY XZ YZ : ℝ) (hXY_pos : XY > 0) (hXZ_pos : XZ > 0) (hYZ_pos : YZ > 0)
  (angle_XYZ : angle_1 = 90) (tan_Z : XY / XZ = 5 / 12) : (XZ / YZ = 12 / 13) :=
by
  sorry

end right_triangle_cosine_l1721_172192


namespace solution_of_fractional_inequality_l1721_172198

noncomputable def solution_set_of_inequality : Set ℝ :=
  {x : ℝ | -3 < x ∨ x > 1/2 }

theorem solution_of_fractional_inequality :
  {x : ℝ | (2 * x - 1) / (x + 3) > 0} = solution_set_of_inequality :=
by
  sorry

end solution_of_fractional_inequality_l1721_172198


namespace probability_of_triangle_or_circle_l1721_172186

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

end probability_of_triangle_or_circle_l1721_172186


namespace triangle_area_given_conditions_l1721_172188

theorem triangle_area_given_conditions (a b c A B S : ℝ) (h₁ : (2 * c - b) * Real.cos A = a * Real.cos B) (h₂ : b = 1) (h₃ : c = 2) :
  S = (1 / 2) * b * c * Real.sin A → S = Real.sqrt 3 / 2 := 
by
  intros
  sorry

end triangle_area_given_conditions_l1721_172188


namespace tan_ratio_l1721_172120

theorem tan_ratio (x y : ℝ) (h1 : Real.sin (x + y) = 5 / 8) (h2 : Real.sin (x - y) = 1 / 4) : 
  (Real.tan x / Real.tan y) = 7 / 3 :=
sorry

end tan_ratio_l1721_172120


namespace transition_algebraic_expression_l1721_172134

theorem transition_algebraic_expression (k : ℕ) (hk : k > 0) :
  (k + 1 + k) * (k + 1 + k + 1) / (k + 1) = 4 * k + 2 :=
sorry

end transition_algebraic_expression_l1721_172134


namespace sqrt_defined_value_l1721_172147

theorem sqrt_defined_value (x : ℝ) (h : x ≥ 4) : x = 5 → true := 
by 
  intro hx
  sorry

end sqrt_defined_value_l1721_172147


namespace problem_statement_l1721_172150

variable {x y z : ℝ}

-- Lean 4 statement of the problem
theorem problem_statement (h₀ : 0 ≤ x) (h₁ : x ≤ 1) (h₂ : 0 ≤ y) (h₃ : y ≤ 1) (h₄ : 0 ≤ z) (h₅ : z ≤ 1) :
  (x / (y + z + 1) + y / (z + x + 1) + z / (x + y + 1)) ≤ 1 - (1 - x) * (1 - y) * (1 - z) := by
  sorry

end problem_statement_l1721_172150


namespace six_digit_number_l1721_172185

/-- 
Find a six-digit number that starts with the digit 1 and such that if this digit is moved to the end, the resulting number is three times the original number.
-/
theorem six_digit_number (N : ℕ) (h₁ : 100000 ≤ N ∧ N < 1000000) (h₂ : ∃ x : ℕ, N = 1 * 10^5 + x ∧ 10 * x + 1 = 3 * N) : N = 142857 :=
by sorry

end six_digit_number_l1721_172185


namespace probability_even_sum_l1721_172136

open Nat

def balls : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def even_sum_probability : ℚ :=
  let total_outcomes := 12 * 11
  let even_balls := balls.filter (λ n => n % 2 = 0)
  let odd_balls := balls.filter (λ n => n % 2 = 1)
  let even_outcomes := even_balls.length * (even_balls.length - 1)
  let odd_outcomes := odd_balls.length * (odd_balls.length - 1)
  let favorable_outcomes := even_outcomes + odd_outcomes
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

theorem probability_even_sum :
  even_sum_probability = 5 / 11 := by
  sorry

end probability_even_sum_l1721_172136


namespace marbles_total_is_260_l1721_172108

/-- Define the number of marbles in each jar. -/
def jar1 : ℕ := 80
def jar2 : ℕ := 2 * jar1
def jar3 : ℕ := jar1 / 4

/-- The total number of marbles Courtney has. -/
def total_marbles : ℕ := jar1 + jar2 + jar3

/-- Proving that the total number of marbles is 260. -/
theorem marbles_total_is_260 : total_marbles = 260 := 
by
  sorry

end marbles_total_is_260_l1721_172108


namespace ellipse_equation_l1721_172169

-- Definitions of the tangents given as conditions
def tangent1 (x y : ℝ) : Prop := 4 * x + 5 * y = 25
def tangent2 (x y : ℝ) : Prop := 9 * x + 20 * y = 75

-- The statement we need to prove
theorem ellipse_equation :
  (∀ (x y : ℝ), tangent1 x y → tangent2 x y → 
  (∃ (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0), a = 5 ∧ b = 3 ∧ 
  (x^2 / a^2 + y^2 / b^2 = 1))) :=
sorry

end ellipse_equation_l1721_172169


namespace largest_divisor_of_product_of_five_consecutive_integers_l1721_172127

theorem largest_divisor_of_product_of_five_consecutive_integers
  (a b c d e : ℤ) 
  (h: a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e):
  ∃ (n : ℤ), n = 60 ∧ n ∣ (a * b * c * d * e) :=
sorry

end largest_divisor_of_product_of_five_consecutive_integers_l1721_172127


namespace sum_ninth_power_l1721_172140

theorem sum_ninth_power (a b : ℝ) (h1 : a + b = 1) (h2 : a^2 + b^2 = 3) 
                        (h3 : a^3 + b^3 = 4) (h4 : a^4 + b^4 = 7)
                        (h5 : a^5 + b^5 = 11)
                        (h_ind : ∀ n, n ≥ 3 → a^n + b^n = a^(n-1) + b^(n-1) + a^(n-2) + b^(n-2)) :
  a^9 + b^9 = 76 :=
by
  sorry

end sum_ninth_power_l1721_172140


namespace find_numbers_l1721_172189

theorem find_numbers (a b c : ℕ) (h : a + b = 2015) (h' : a = 10 * b + c) (hc : 0 ≤ c ∧ c ≤ 9) :
  (a = 1832 ∧ b = 183) :=
sorry

end find_numbers_l1721_172189


namespace dave_bought_packs_l1721_172187

def packs_of_white_shirts (bought_total : ℕ) (white_per_pack : ℕ) (blue_packs : ℕ) (blue_per_pack : ℕ) : ℕ :=
  (bought_total - blue_packs * blue_per_pack) / white_per_pack

theorem dave_bought_packs : packs_of_white_shirts 26 6 2 4 = 3 :=
by
  sorry

end dave_bought_packs_l1721_172187


namespace boxes_of_nerds_l1721_172130

def totalCandies (kitKatBars hersheyKisses lollipops babyRuths reeseCups nerds : Nat) : Nat := 
  kitKatBars + hersheyKisses + lollipops + babyRuths + reeseCups + nerds

def adjustForGivenLollipops (total lollipopsGiven : Nat) : Nat :=
  total - lollipopsGiven

theorem boxes_of_nerds :
  ∀ (kitKatBars hersheyKisses lollipops babyRuths reeseCups lollipopsGiven totalAfterGiving nerds : Nat),
  kitKatBars = 5 →
  hersheyKisses = 3 * kitKatBars →
  lollipops = 11 →
  babyRuths = 10 →
  reeseCups = babyRuths / 2 →
  lollipopsGiven = 5 →
  totalAfterGiving = 49 →
  totalCandies kitKatBars hersheyKisses lollipops babyRuths reeseCups 0 - lollipopsGiven + nerds = totalAfterGiving →
  nerds = 8 :=
by
  intros
  sorry

end boxes_of_nerds_l1721_172130


namespace crossed_out_digit_l1721_172179

theorem crossed_out_digit (N S S' x : ℕ) (hN : N % 9 = 3) (hS : S % 9 = 3) (hS' : S' % 9 = 7)
  (hS'_eq : S' = S - x) : x = 5 :=
by
  sorry

end crossed_out_digit_l1721_172179


namespace find_M_l1721_172158

theorem find_M :
  (∃ (M : ℕ), (5 + 7 + 9) / 3 = (4020 + 4021 + 4022) / M) → M = 1723 :=
  by
  sorry

end find_M_l1721_172158


namespace intersection_of_sets_l1721_172149

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | 0 ≤ x ∧ x < 5 / 2}
def C : Set ℤ := {0, 1, 2}

theorem intersection_of_sets :
  C = A ∩ B :=
sorry

end intersection_of_sets_l1721_172149


namespace division_of_monomials_l1721_172157

variable (x : ℝ) -- ensure x is defined as a variable, here assuming x is a real number

theorem division_of_monomials (x : ℝ) : (2 * x^3 / x^2) = 2 * x := 
by 
  sorry

end division_of_monomials_l1721_172157


namespace equivalent_statements_l1721_172199

variable (P Q : Prop)

theorem equivalent_statements (h : P → Q) :
  (¬Q → ¬P) ∧ (¬P ∨ Q) :=
by 
  sorry

end equivalent_statements_l1721_172199


namespace tens_digit_of_square_ending_in_six_odd_l1721_172164

theorem tens_digit_of_square_ending_in_six_odd 
   (N : ℤ) 
   (a : ℤ) 
   (b : ℕ) 
   (hle : 0 ≤ b) 
   (hge : b < 10) 
   (hexp : N = 10 * a + b) 
   (hsqr : (N^2) % 10 = 6) : 
   ∃ k : ℕ, (N^2 / 10) % 10 = 2 * k + 1 :=
sorry -- Proof goes here

end tens_digit_of_square_ending_in_six_odd_l1721_172164


namespace ratio_area_II_to_III_l1721_172196

-- Define the properties of the squares as given in the conditions
def perimeter_region_I : ℕ := 16
def perimeter_region_II : ℕ := 32
def side_length_region_I := perimeter_region_I / 4
def side_length_region_II := perimeter_region_II / 4
def side_length_region_III := 2 * side_length_region_II
def area_region_II := side_length_region_II ^ 2
def area_region_III := side_length_region_III ^ 2

-- Prove that the ratio of the area of region II to the area of region III is 1/4
theorem ratio_area_II_to_III : (area_region_II : ℚ) / (area_region_III : ℚ) = 1 / 4 := 
by sorry

end ratio_area_II_to_III_l1721_172196


namespace largest_of_four_l1721_172113

theorem largest_of_four : 
  let a := 1 
  let b := 0 
  let c := |(-2)| 
  let d := -3 
  max (max (max a b) c) d = c := by
  sorry

end largest_of_four_l1721_172113


namespace find_x_l1721_172166

theorem find_x (x y : ℚ) (h1 : 3 * x - y = 9) (h2 : x + 4 * y = 11) : x = 47 / 13 :=
sorry

end find_x_l1721_172166


namespace no_four_digit_with_five_units_divisible_by_ten_l1721_172154

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def units_place_is_five (n : ℕ) : Prop :=
  n % 10 = 5

def divisible_by_ten (n : ℕ) : Prop :=
  n % 10 = 0

theorem no_four_digit_with_five_units_divisible_by_ten : ∀ n : ℕ, 
  is_four_digit n → units_place_is_five n → ¬ divisible_by_ten n :=
by
  intro n h1 h2
  rw [units_place_is_five] at h2
  rw [divisible_by_ten, h2]
  sorry

end no_four_digit_with_five_units_divisible_by_ten_l1721_172154


namespace total_fundamental_particles_l1721_172139

def protons := 9
def neutrons := 19 - protons
def electrons := protons
def total_particles := protons + neutrons + electrons

theorem total_fundamental_particles : total_particles = 28 := by
  sorry

end total_fundamental_particles_l1721_172139


namespace cost_price_of_radio_l1721_172155

theorem cost_price_of_radio (SP : ℝ) (L_p : ℝ) (C : ℝ) (h₁ : SP = 3200) (h₂ : L_p = 0.28888888888888886) 
  (h₃ : SP = C - (C * L_p)) : C = 4500 :=
by
  sorry

end cost_price_of_radio_l1721_172155


namespace questionnaires_drawn_l1721_172171

theorem questionnaires_drawn
  (units : ℕ → ℕ)
  (h_arithmetic : ∀ n, units (n + 1) - units n = units 1 - units 0)
  (h_total : units 0 + units 1 + units 2 + units 3 = 100)
  (h_unitB : units 1 = 20) :
  units 3 = 40 :=
by
  -- Proof would go here
  -- Establish that the arithmetic sequence difference is 10, then compute unit D (units 3)
  sorry

end questionnaires_drawn_l1721_172171


namespace greatest_value_y_l1721_172122

theorem greatest_value_y (y : ℝ) (hy : 11 = y^2 + 1/y^2) : y + 1/y ≤ Real.sqrt 13 :=
sorry

end greatest_value_y_l1721_172122


namespace actual_height_is_191_l1721_172195

theorem actual_height_is_191 :
  ∀ (n incorrect_avg correct_avg incorrect_height x : ℝ),
  n = 20 ∧ incorrect_avg = 175 ∧ correct_avg = 173 ∧ incorrect_height = 151 ∧
  (n * incorrect_avg - n * correct_avg = x - incorrect_height) →
  x = 191 :=
by
  intros n incorrect_avg correct_avg incorrect_height x h
  -- skip the proof part
  sorry

end actual_height_is_191_l1721_172195


namespace total_weekly_allowance_l1721_172156

theorem total_weekly_allowance
  (total_students : ℕ)
  (students_6dollar : ℕ)
  (students_4dollar : ℕ)
  (students_7dollar : ℕ)
  (allowance_6dollar : ℕ)
  (allowance_4dollar : ℕ)
  (allowance_7dollar : ℕ)
  (days_in_week : ℕ) :
  total_students = 100 →
  students_6dollar = 60 →
  students_4dollar = 25 →
  students_7dollar = 15 →
  allowance_6dollar = 6 →
  allowance_4dollar = 4 →
  allowance_7dollar = 7 →
  days_in_week = 7 →
  (students_6dollar * allowance_6dollar + students_4dollar * allowance_4dollar + students_7dollar * allowance_7dollar) * days_in_week = 3955 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end total_weekly_allowance_l1721_172156


namespace probability_same_color_l1721_172180

-- Definitions with conditions
def totalPlates := 11
def redPlates := 6
def bluePlates := 5

-- Calculate combinations
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The main statement
theorem probability_same_color (totalPlates redPlates bluePlates : ℕ) (h1 : totalPlates = 11) 
(h2 : redPlates = 6) (h3 : bluePlates = 5) : 
  (2 / 11 : ℚ) = ((choose redPlates 3 + choose bluePlates 3) : ℚ) / (choose totalPlates 3) :=
by
  -- Proof steps will be inserted here
  sorry

end probability_same_color_l1721_172180


namespace side_length_of_square_l1721_172183

theorem side_length_of_square (d : ℝ) (h : d = 4) : ∃ (s : ℝ), s = 2 * Real.sqrt 2 :=
by
  sorry

end side_length_of_square_l1721_172183


namespace swimming_speed_solution_l1721_172111

-- Definition of the conditions
def speed_of_water : ℝ := 2
def distance_against_current : ℝ := 10
def time_against_current : ℝ := 5

-- Definition of the person's swimming speed in still water
def swimming_speed_in_still_water (v : ℝ) :=
  distance_against_current = (v - speed_of_water) * time_against_current

-- Main theorem we want to prove
theorem swimming_speed_solution : 
  ∃ v : ℝ, swimming_speed_in_still_water v ∧ v = 4 :=
by
  sorry

end swimming_speed_solution_l1721_172111


namespace find_interesting_numbers_l1721_172141

def is_interesting (A B : ℕ) : Prop :=
  A > B ∧ (∃ p : ℕ, Nat.Prime p ∧ A - B = p) ∧ ∃ n : ℕ, A * B = n ^ 2

theorem find_interesting_numbers :
  {A | (∃ B : ℕ, is_interesting A B) ∧ 200 < A ∧ A < 400} = {225, 256, 361} :=
by
  sorry

end find_interesting_numbers_l1721_172141


namespace volume_correct_l1721_172129

-- Define the structure and conditions
structure Point where
  x : ℝ
  y : ℝ

def is_on_circle (C : Point) (P : Point) : Prop :=
  (P.x - C.x)^2 + (P.y - C.y)^2 = 25

def volume_of_solid_of_revolution (P A B : Point) : ℝ := sorry

noncomputable def main : ℝ :=
  volume_of_solid_of_revolution {x := 2, y := -8} {x := 4.58, y := -1.98} {x := -3.14, y := -3.91}

theorem volume_correct :
  main = 672.1 := by
  -- Proof skipped
  sorry

end volume_correct_l1721_172129


namespace servings_of_popcorn_l1721_172101

theorem servings_of_popcorn (popcorn_per_serving : ℕ) (jared_consumption : ℕ)
    (friend_consumption : ℕ) (num_friends : ℕ) :
    popcorn_per_serving = 30 →
    jared_consumption = 90 →
    friend_consumption = 60 →
    num_friends = 3 →
    (jared_consumption + num_friends * friend_consumption) / popcorn_per_serving = 9 := 
by
  intros h1 h2 h3 h4
  sorry

end servings_of_popcorn_l1721_172101


namespace fewest_students_possible_l1721_172132

theorem fewest_students_possible : 
  ∃ n : ℕ, n % 3 = 1 ∧ n % 6 = 4 ∧ n % 8 = 5 ∧ ∀ m, m % 3 = 1 ∧ m % 6 = 4 ∧ m % 8 = 5 → n ≤ m := 
by
  sorry

end fewest_students_possible_l1721_172132


namespace decreasing_y_as_x_increases_l1721_172104

theorem decreasing_y_as_x_increases :
  (∀ x1 x2, x1 < x2 → (-2 * x1 + 1) > (-2 * x2 + 1)) ∧
  ¬ (∀ x1 x2, x1 < x2 → (x1^2 + 1) > (x2^2 + 1)) ∧
  ¬ (∀ x1 x2, x1 < x2 → (-x1^2 + 1) > (-x2^2 + 1)) ∧
  ¬ (∀ x1 x2, x1 < x2 → (2 * x1 + 1) > (2 * x2 + 1)) :=
by
  sorry

end decreasing_y_as_x_increases_l1721_172104


namespace calculate_years_l1721_172193

variable {P R T SI : ℕ}

-- Conditions translations
def simple_interest_one_fifth (P SI : ℕ) : Prop :=
  SI = P / 5

def rate_of_interest (R : ℕ) : Prop :=
  R = 4

-- Proof of the number of years T
theorem calculate_years (h1 : simple_interest_one_fifth P SI)
                        (h2 : rate_of_interest R)
                        (h3 : SI = (P * R * T) / 100) : T = 5 :=
by
  sorry

end calculate_years_l1721_172193


namespace sum_three_numbers_l1721_172106

theorem sum_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 267) 
  (h2 : a * b + b * c + c * a = 131) : 
  a + b + c = 23 := by
  sorry

end sum_three_numbers_l1721_172106


namespace area_of_sector_l1721_172109

def radius : ℝ := 5
def central_angle : ℝ := 2

theorem area_of_sector : (1 / 2) * radius^2 * central_angle = 25 := by
  sorry

end area_of_sector_l1721_172109


namespace triangle_side_relation_l1721_172115

-- Definitions for the conditions
variable {A B C a b c : ℝ}
variable (acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2)
variable (sides_rel : a = (B * (1 + 2 * C)).sin)
variable (trig_eq : (B.sin * (1 + 2 * C.cos)) = (2 * A.sin * C.cos + A.cos * C.sin))

-- The statement to be proven
theorem triangle_side_relation (acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2)
  (sides_rel : a = (B * (1 + 2 * C)).sin)
  (trig_eq : (B.sin * (1 + 2 * C.cos)) = (2 * A.sin * C.cos + A.cos * C.sin)) :
  a = 2 * b := 
sorry

end triangle_side_relation_l1721_172115
