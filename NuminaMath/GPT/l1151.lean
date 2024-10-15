import Mathlib

namespace NUMINAMATH_GPT_tan_225_eq_1_l1151_115180

theorem tan_225_eq_1 : Real.tan (225 * Real.pi / 180) = 1 := by
  sorry

end NUMINAMATH_GPT_tan_225_eq_1_l1151_115180


namespace NUMINAMATH_GPT_final_price_of_bicycle_l1151_115173

def original_price : ℝ := 200
def first_discount_rate : ℝ := 0.40
def second_discount_rate : ℝ := 0.25

theorem final_price_of_bicycle :
  let first_sale_price := original_price - (first_discount_rate * original_price)
  let final_sale_price := first_sale_price - (second_discount_rate * first_sale_price)
  final_sale_price = 90 := by
  sorry

end NUMINAMATH_GPT_final_price_of_bicycle_l1151_115173


namespace NUMINAMATH_GPT_problem_l1151_115188

theorem problem
: 15 * (1 / 17) * 34 = 30 := by
  sorry

end NUMINAMATH_GPT_problem_l1151_115188


namespace NUMINAMATH_GPT_rectangle_perimeter_l1151_115117

open Real

def triangle_DEF_sides : ℝ × ℝ × ℝ := (9, 12, 15) -- sides of the triangle DEF

def rectangle_width : ℝ := 6 -- width of the rectangle

theorem rectangle_perimeter (a b c width : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : width = 6) :
  2 * (54 / width + width) = 30 :=
by
  sorry -- Proof is omitted as required

end NUMINAMATH_GPT_rectangle_perimeter_l1151_115117


namespace NUMINAMATH_GPT_curve_is_line_l1151_115171

def curve := {p : ℝ × ℝ | ∃ (θ : ℝ), (p.1 = (1 / (Real.sin θ + Real.cos θ)) * Real.cos θ
                                        ∧ p.2 = (1 / (Real.sin θ + Real.cos θ)) * Real.sin θ)}

-- Problem: Prove that the curve defined by the polar equation is a line.
theorem curve_is_line : ∀ (p : ℝ × ℝ), p ∈ curve → p.1 + p.2 = 1 :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_curve_is_line_l1151_115171


namespace NUMINAMATH_GPT_parallelogram_side_lengths_l1151_115166

theorem parallelogram_side_lengths (x y : ℝ) (h₁ : 3 * x + 6 = 12) (h₂ : 10 * y - 3 = 15) : x + y = 3.8 :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_side_lengths_l1151_115166


namespace NUMINAMATH_GPT_nancy_pots_created_on_Wednesday_l1151_115109

def nancy_pots_conditions (pots_Monday pots_Tuesday total_pots : ℕ) : Prop :=
  pots_Monday = 12 ∧ pots_Tuesday = 2 * pots_Monday ∧ total_pots = 50

theorem nancy_pots_created_on_Wednesday :
  ∀ pots_Monday pots_Tuesday total_pots,
  nancy_pots_conditions pots_Monday pots_Tuesday total_pots →
  (total_pots - (pots_Monday + pots_Tuesday) = 14) := by
  intros pots_Monday pots_Tuesday total_pots h
  -- proof would go here
  sorry

end NUMINAMATH_GPT_nancy_pots_created_on_Wednesday_l1151_115109


namespace NUMINAMATH_GPT_probability_triangle_or_hexagon_l1151_115181

theorem probability_triangle_or_hexagon 
  (total_shapes : ℕ) 
  (num_triangles : ℕ) 
  (num_squares : ℕ) 
  (num_circles : ℕ) 
  (num_hexagons : ℕ)
  (htotal : total_shapes = 10)
  (htriangles : num_triangles = 3)
  (hsquares : num_squares = 4)
  (hcircles : num_circles = 2)
  (hhexagons : num_hexagons = 1):
  (num_triangles + num_hexagons) / total_shapes = 2 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_probability_triangle_or_hexagon_l1151_115181


namespace NUMINAMATH_GPT_connections_required_l1151_115128

theorem connections_required (n : ℕ) (k : ℕ) (h_n : n = 30) (h_k : k = 4) :
  (n * k) / 2 = 60 := by
  sorry

end NUMINAMATH_GPT_connections_required_l1151_115128


namespace NUMINAMATH_GPT_smallest_number_of_students_in_debate_club_l1151_115162

-- Define conditions
def ratio_8th_to_6th (x₈ x₆ : ℕ) : Prop := 7 * x₆ = 4 * x₈
def ratio_8th_to_7th (x₈ x₇ : ℕ) : Prop := 6 * x₇ = 5 * x₈
def ratio_8th_to_9th (x₈ x₉ : ℕ) : Prop := 9 * x₉ = 2 * x₈

-- Problem statement
theorem smallest_number_of_students_in_debate_club 
  (x₈ x₆ x₇ x₉ : ℕ) 
  (h₁ : ratio_8th_to_6th x₈ x₆) 
  (h₂ : ratio_8th_to_7th x₈ x₇) 
  (h₃ : ratio_8th_to_9th x₈ x₉) : 
  x₈ + x₆ + x₇ + x₉ = 331 := 
sorry

end NUMINAMATH_GPT_smallest_number_of_students_in_debate_club_l1151_115162


namespace NUMINAMATH_GPT_find_phi_l1151_115132

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)

theorem find_phi (phi : ℝ) (h_shift : ∀ x : ℝ, f (x + phi) = f (-x - phi)) : 
  phi = Real.pi / 8 :=
  sorry

end NUMINAMATH_GPT_find_phi_l1151_115132


namespace NUMINAMATH_GPT_soldiers_arrival_time_l1151_115175

open Function

theorem soldiers_arrival_time
    (num_soldiers : ℕ) (distance : ℝ) (car_speed : ℝ) (car_capacity : ℕ) (walk_speed : ℝ) (start_time : ℝ) :
    num_soldiers = 12 →
    distance = 20 →
    car_speed = 20 →
    car_capacity = 4 →
    walk_speed = 4 →
    start_time = 0 →
    ∃ arrival_time, arrival_time = 2 + 36/60 :=
by
  intros
  sorry

end NUMINAMATH_GPT_soldiers_arrival_time_l1151_115175


namespace NUMINAMATH_GPT_diff_of_squares_l1151_115172

variable (a : ℝ)

theorem diff_of_squares (a : ℝ) : (a + 3) * (a - 3) = a^2 - 9 := by
  sorry

end NUMINAMATH_GPT_diff_of_squares_l1151_115172


namespace NUMINAMATH_GPT_gcd_irreducible_fraction_l1151_115168

theorem gcd_irreducible_fraction (n : ℕ) (hn: 0 < n) : gcd (3*n + 1) (5*n + 2) = 1 :=
  sorry

end NUMINAMATH_GPT_gcd_irreducible_fraction_l1151_115168


namespace NUMINAMATH_GPT_possible_values_of_a_l1151_115116

theorem possible_values_of_a (x y a : ℝ) (h1 : x + y = a) (h2 : x^3 + y^3 = a) (h3 : x^5 + y^5 = a) : 
  a = -2 ∨ a = -1 ∨ a = 0 ∨ a = 1 ∨ a = 2 :=
by sorry

end NUMINAMATH_GPT_possible_values_of_a_l1151_115116


namespace NUMINAMATH_GPT_distance_between_red_lights_in_feet_l1151_115169

theorem distance_between_red_lights_in_feet :
  let inches_between_lights := 6
  let pattern := [2, 3]
  let foot_in_inches := 12
  let pos_3rd_red := 6
  let pos_21st_red := 51
  let number_of_gaps := pos_21st_red - pos_3rd_red
  let total_distance_in_inches := number_of_gaps * inches_between_lights
  let total_distance_in_feet := total_distance_in_inches / foot_in_inches
  total_distance_in_feet = 22 := by
  sorry

end NUMINAMATH_GPT_distance_between_red_lights_in_feet_l1151_115169


namespace NUMINAMATH_GPT_glasses_total_l1151_115102

theorem glasses_total :
  ∃ (S L e : ℕ), 
    (L = S + 16) ∧ 
    (12 * S + 16 * L) / (S + L) = 15 ∧ 
    (e = 12 * S + 16 * L) ∧ 
    e = 480 :=
by
  sorry

end NUMINAMATH_GPT_glasses_total_l1151_115102


namespace NUMINAMATH_GPT_average_a_b_l1151_115179

theorem average_a_b (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (B + C) / 2 = 41)
  (h3 : B = 27) : (A + B) / 2 = 40 := 
by
  sorry

end NUMINAMATH_GPT_average_a_b_l1151_115179


namespace NUMINAMATH_GPT_shaded_area_is_28_l1151_115134

theorem shaded_area_is_28 (A B : ℕ) (h1 : A = 64) (h2 : B = 28) : B = 28 := by
  sorry

end NUMINAMATH_GPT_shaded_area_is_28_l1151_115134


namespace NUMINAMATH_GPT_daniel_total_spent_l1151_115194

/-
Daniel buys various items with given prices, receives a 10% coupon discount,
a store credit of $1.50, a 5% student discount, and faces a 6.5% sales tax.
Prove that the total amount he spends is $8.23.
-/
def total_spent (prices : List ℝ) (coupon_discount store_credit student_discount sales_tax : ℝ) : ℝ :=
  let initial_total := prices.sum
  let after_coupon := initial_total * (1 - coupon_discount)
  let after_student := after_coupon * (1 - student_discount)
  let after_store_credit := after_student - store_credit
  let final_total := after_store_credit * (1 + sales_tax)
  final_total

theorem daniel_total_spent :
  total_spent 
    [0.85, 0.50, 1.25, 3.75, 2.99, 1.45] -- prices of items
    0.10 -- 10% coupon discount
    1.50 -- $1.50 store credit
    0.05 -- 5% student discount
    0.065 -- 6.5% sales tax
  = 8.23 :=
by
  sorry

end NUMINAMATH_GPT_daniel_total_spent_l1151_115194


namespace NUMINAMATH_GPT_total_whales_seen_is_178_l1151_115164

/-
Ishmael's monitoring of whales yields the following:
- On the first trip, he counts 28 male whales and twice as many female whales.
- On the second trip, he sees 8 baby whales, each traveling with their parents.
- On the third trip, he counts half as many male whales as the first trip and the same number of female whales as on the first trip.
-/

def number_of_whales_first_trip : ℕ := 28
def number_of_female_whales_first_trip : ℕ := 2 * number_of_whales_first_trip
def total_whales_first_trip : ℕ := number_of_whales_first_trip + number_of_female_whales_first_trip

def number_of_baby_whales_second_trip : ℕ := 8
def total_whales_second_trip : ℕ := number_of_baby_whales_second_trip * 3

def number_of_male_whales_third_trip : ℕ := number_of_whales_first_trip / 2
def number_of_female_whales_third_trip : ℕ := number_of_female_whales_first_trip
def total_whales_third_trip : ℕ := number_of_male_whales_third_trip + number_of_female_whales_third_trip

def total_whales_seen : ℕ := total_whales_first_trip + total_whales_second_trip + total_whales_third_trip

theorem total_whales_seen_is_178 : total_whales_seen = 178 :=
by
  -- skip the actual proof
  sorry

end NUMINAMATH_GPT_total_whales_seen_is_178_l1151_115164


namespace NUMINAMATH_GPT_required_speed_l1151_115135

noncomputable def distance_travelled_late (d: ℝ) (t: ℝ) : ℝ :=
  50 * (t + 1/12)

noncomputable def distance_travelled_early (d: ℝ) (t: ℝ) : ℝ :=
  70 * (t - 1/12)

theorem required_speed :
  ∃ (s: ℝ), s = 58 ∧ 
  (∀ (d t: ℝ), distance_travelled_late d t = d ∧ distance_travelled_early d t = d → 
  d / t = s) :=
by
  sorry

end NUMINAMATH_GPT_required_speed_l1151_115135


namespace NUMINAMATH_GPT_percentage_rotten_apples_l1151_115127

theorem percentage_rotten_apples
  (total_apples : ℕ)
  (smell_pct : ℚ)
  (non_smelling_rotten_apples : ℕ)
  (R : ℚ) :
  total_apples = 200 →
  smell_pct = 0.70 →
  non_smelling_rotten_apples = 24 →
  0.30 * (R / 100 * total_apples) = non_smelling_rotten_apples →
  R = 40 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_percentage_rotten_apples_l1151_115127


namespace NUMINAMATH_GPT_max_profit_at_9_l1151_115129

noncomputable def R (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 10 then 10.8 - (1 / 30) * x^2
else if h : x > 10 then 108 / x - 1000 / (3 * x^2)
else 0

noncomputable def W (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 10 then 8.1 * x - x^3 / 30 - 10
else if h : x > 10 then 98 - 1000 / (3 * x) - 2.7 * x
else 0

theorem max_profit_at_9 : W 9 = 38.6 :=
sorry

end NUMINAMATH_GPT_max_profit_at_9_l1151_115129


namespace NUMINAMATH_GPT_print_shop_X_charge_l1151_115174

-- Define the given conditions
def cost_per_copy_X (x : ℝ) : Prop := x > 0
def cost_per_copy_Y : ℝ := 2.75
def total_copies : ℕ := 40
def extra_cost_Y : ℝ := 60

-- Define the main problem
theorem print_shop_X_charge (x : ℝ) (h : cost_per_copy_X x) :
  total_copies * cost_per_copy_Y = total_copies * x + extra_cost_Y → x = 1.25 :=
by
  sorry

end NUMINAMATH_GPT_print_shop_X_charge_l1151_115174


namespace NUMINAMATH_GPT_quadratic_equation_solution_diff_l1151_115155

theorem quadratic_equation_solution_diff :
  let a := 1
  let b := -6
  let c := -40
  let discriminant := b^2 - 4 * a * c
  let root1 := (-b + Real.sqrt discriminant) / (2 * a)
  let root2 := (-b - Real.sqrt discriminant) / (2 * a)
  abs (root1 - root2) = 14 := by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_quadratic_equation_solution_diff_l1151_115155


namespace NUMINAMATH_GPT_average_age_of_9_students_l1151_115121

theorem average_age_of_9_students
  (avg_20_students : ℝ)
  (n_20_students : ℕ)
  (avg_10_students : ℝ)
  (n_10_students : ℕ)
  (age_20th_student : ℝ)
  (total_age_20_students : ℝ := avg_20_students * n_20_students)
  (total_age_10_students : ℝ := avg_10_students * n_10_students)
  (total_age_9_students : ℝ := total_age_20_students - total_age_10_students - age_20th_student)
  (n_9_students : ℕ)
  (expected_avg_9_students : ℝ := total_age_9_students / n_9_students)
  (H1 : avg_20_students = 20)
  (H2 : n_20_students = 20)
  (H3 : avg_10_students = 24)
  (H4 : n_10_students = 10)
  (H5 : age_20th_student = 61)
  (H6 : n_9_students = 9) :
  expected_avg_9_students = 11 :=
sorry

end NUMINAMATH_GPT_average_age_of_9_students_l1151_115121


namespace NUMINAMATH_GPT_planes_parallel_if_line_perpendicular_to_both_l1151_115142

variables {Line Plane : Type}
variables (l : Line) (α β : Plane)

-- Assume we have a function parallel that checks if a line is parallel to a plane
-- and a function perpendicular that checks if a line is perpendicular to a plane. 
-- Also, we assume a function parallel_planes that checks if two planes are parallel.
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

theorem planes_parallel_if_line_perpendicular_to_both
  (h1 : perpendicular l α) (h2 : perpendicular l β) : parallel_planes α β :=
sorry

end NUMINAMATH_GPT_planes_parallel_if_line_perpendicular_to_both_l1151_115142


namespace NUMINAMATH_GPT_a_4_eq_15_l1151_115106

noncomputable def a : ℕ → ℕ
| 0 => 1
| (n + 1) => 2 * a n + 1

theorem a_4_eq_15 : a 3 = 15 :=
by
  sorry

end NUMINAMATH_GPT_a_4_eq_15_l1151_115106


namespace NUMINAMATH_GPT_number_of_men_in_larger_group_l1151_115143

-- Define the constants and conditions
def men1 := 36         -- men in the first group
def days1 := 18        -- days taken by the first group
def men2 := 108       -- men in the larger group (what we want to prove)
def days2 := 6         -- days taken by the second group

-- Given conditions as lean definitions
def total_work (men : Nat) (days : Nat) := men * days
def condition1 := (total_work men1 days1 = 648)
def condition2 := (total_work men2 days2 = 648)

-- Problem statement 
-- proving that men2 is 108
theorem number_of_men_in_larger_group : condition1 → condition2 → men2 = 108 :=
by
  intros
  sorry

end NUMINAMATH_GPT_number_of_men_in_larger_group_l1151_115143


namespace NUMINAMATH_GPT_loraine_wax_usage_l1151_115152

/-
Loraine makes wax sculptures of animals. Large animals take eight sticks of wax, medium animals take five sticks, and small animals take three sticks.
She made twice as many small animals as large animals, and four times as many medium animals as large animals. She used 36 sticks of wax for small animals.
Prove that Loraine used 204 sticks of wax to make all the animals.
-/

theorem loraine_wax_usage :
  ∃ (L M S : ℕ), (S = 2 * L) ∧ (M = 4 * L) ∧ (3 * S = 36) ∧ (8 * L + 5 * M + 3 * S = 204) :=
by {
  sorry
}

end NUMINAMATH_GPT_loraine_wax_usage_l1151_115152


namespace NUMINAMATH_GPT_abs_eq_four_l1151_115111

theorem abs_eq_four (x : ℝ) (h : |x| = 4) : x = 4 ∨ x = -4 :=
by
  sorry

end NUMINAMATH_GPT_abs_eq_four_l1151_115111


namespace NUMINAMATH_GPT_expression_simplifies_to_one_l1151_115193

-- Define x in terms of the given condition
def x : ℚ := (1 / 2) ^ (-1 : ℤ) + (-3) ^ (0 : ℤ)

-- Define the given expression
def expr (x : ℚ) : ℚ := (((x^2 - 1) / (x^2 - 2 * x + 1)) - (1 / (x - 1))) / (3 / (x - 1))

-- Define the theorem stating the equivalence
theorem expression_simplifies_to_one : expr x = 1 := by
  sorry

end NUMINAMATH_GPT_expression_simplifies_to_one_l1151_115193


namespace NUMINAMATH_GPT_points_on_decreasing_line_y1_gt_y2_l1151_115191
-- Import the necessary library

-- Necessary conditions and definitions
variable {x y : ℝ}

-- Given points P(3, y1) and Q(4, y2)
def y1 : ℝ := -2*3 + 4
def y2 : ℝ := -2*4 + 4

-- Lean statement to prove y1 > y2
theorem points_on_decreasing_line_y1_gt_y2 (h1 : y1 = -2 * 3 +4) (h2 : y2 = -2 * 4 + 4) : 
  y1 > y2 :=
sorry  -- Proof steps go here

end NUMINAMATH_GPT_points_on_decreasing_line_y1_gt_y2_l1151_115191


namespace NUMINAMATH_GPT_base_b_square_l1151_115147

theorem base_b_square (b : ℕ) (h : b > 4) : ∃ k : ℕ, k^2 = b^2 + 4 * b + 4 := 
by 
  sorry

end NUMINAMATH_GPT_base_b_square_l1151_115147


namespace NUMINAMATH_GPT_sequence_sum_after_operations_l1151_115126

-- Define the initial sequence length
def initial_sequence := [1, 9, 8, 8]

-- Define the sum of initial sequence
def initial_sum := initial_sequence.sum

-- Define the number of operations
def ops := 100

-- Define the increase per operation
def increase_per_op := 7

-- Define the final sum after operations
def final_sum := initial_sum + (increase_per_op * ops)

-- Prove the final sum is 726 after 100 operations
theorem sequence_sum_after_operations : final_sum = 726 := by
  -- Proof omitted as per instructions
  sorry

end NUMINAMATH_GPT_sequence_sum_after_operations_l1151_115126


namespace NUMINAMATH_GPT_problem_solution_l1151_115160

theorem problem_solution :
  ((8 * 2.25 - 5 * 0.85) / 2.5 + (3 / 5 * 1.5 - 7 / 8 * 0.35) / 1.25) = 5.975 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1151_115160


namespace NUMINAMATH_GPT_total_cleaning_time_is_100_l1151_115186

def outsideCleaningTime : ℕ := 80
def insideCleaningTime : ℕ := outsideCleaningTime / 4
def totalCleaningTime : ℕ := outsideCleaningTime + insideCleaningTime

theorem total_cleaning_time_is_100 : totalCleaningTime = 100 := by
  sorry

end NUMINAMATH_GPT_total_cleaning_time_is_100_l1151_115186


namespace NUMINAMATH_GPT_neg_i_pow_four_l1151_115182

-- Define i as the imaginary unit satisfying i^2 = -1
def i : ℂ := Complex.I

-- The proof problem: Prove (-i)^4 = 1 given i^2 = -1
theorem neg_i_pow_four : (-i)^4 = 1 :=
by
  -- sorry is used to skip proof
  sorry

end NUMINAMATH_GPT_neg_i_pow_four_l1151_115182


namespace NUMINAMATH_GPT_complex_quadrant_example_l1151_115153

open Complex

def in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem complex_quadrant_example (z : ℂ) (h : (1 - I) * z = (1 + I) ^ 2) : in_second_quadrant z :=
by
  sorry

end NUMINAMATH_GPT_complex_quadrant_example_l1151_115153


namespace NUMINAMATH_GPT_at_least_one_ge_one_l1151_115156

theorem at_least_one_ge_one (x1 x2 x3 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (hx3 : 0 < x3) :
  let a := x1 / x2
  let b := x2 / x3
  let c := x3 / x1
  a + b + c ≥ 3 → (a ≥ 1 ∨ b ≥ 1 ∨ c ≥ 1) :=
by
  intros
  sorry

end NUMINAMATH_GPT_at_least_one_ge_one_l1151_115156


namespace NUMINAMATH_GPT_percentage_y_of_x_l1151_115161

variable {x y : ℝ}

theorem percentage_y_of_x 
  (h : 0.15 * x = 0.20 * y) : y = 0.75 * x := 
sorry

end NUMINAMATH_GPT_percentage_y_of_x_l1151_115161


namespace NUMINAMATH_GPT_quadratic_function_range_l1151_115167

def range_of_quadratic_function : Set ℝ :=
  {y : ℝ | y ≥ 2}

theorem quadratic_function_range :
  ∀ x : ℝ, (∃ y : ℝ, y = x^2 - 4*x + 6 ∧ y ∈ range_of_quadratic_function) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_range_l1151_115167


namespace NUMINAMATH_GPT_vegetable_price_l1151_115154

theorem vegetable_price (v : ℝ) 
  (beef_cost : ∀ (b : ℝ), b = 3 * v)
  (total_cost : 4 * (3 * v) + 6 * v = 36) : 
  v = 2 :=
by {
  -- The proof would go here.
  sorry
}

end NUMINAMATH_GPT_vegetable_price_l1151_115154


namespace NUMINAMATH_GPT_reflected_light_ray_equation_l1151_115190

-- Definitions for the points and line
structure Point := (x : ℝ) (y : ℝ)

-- Given points M and N
def M : Point := ⟨2, 6⟩
def N : Point := ⟨-3, 4⟩

-- Given line l
def l (p : Point) : Prop := p.x - p.y + 3 = 0

-- The target equation of the reflected light ray
def target_equation (p : Point) : Prop := p.x - 6 * p.y + 27 = 0

-- Statement to prove
theorem reflected_light_ray_equation :
  (∃ K : Point, (M.x = 2 ∧ M.y = 6) ∧ l (⟨K.x + (K.x - M.x), K.y + (K.y - M.y)⟩)
     ∧ (N.x = -3 ∧ N.y = 4)) →
  (∀ P : Point, target_equation P ↔ (P.x - 6 * P.y + 27 = 0)) := by
sorry

end NUMINAMATH_GPT_reflected_light_ray_equation_l1151_115190


namespace NUMINAMATH_GPT_find_areas_after_shortening_l1151_115124

-- Define initial dimensions
def initial_length : ℤ := 5
def initial_width : ℤ := 7
def shortened_by : ℤ := 2

-- Define initial area condition
def initial_area_condition : Prop := 
  initial_length * (initial_width - shortened_by) = 15 ∨ (initial_length - shortened_by) * initial_width = 15

-- Define the resulting areas for shortening each dimension
def area_shortening_length : ℤ := (initial_length - shortened_by) * initial_width
def area_shortening_width : ℤ := initial_length * (initial_width - shortened_by)

-- Statement for proof
theorem find_areas_after_shortening
  (h : initial_area_condition) :
  area_shortening_length = 21 ∧ area_shortening_width = 25 :=
sorry

end NUMINAMATH_GPT_find_areas_after_shortening_l1151_115124


namespace NUMINAMATH_GPT_find_g_l1151_115159

noncomputable def g (x : ℝ) := -4 * x ^ 4 + x ^ 3 - 6 * x ^ 2 + x - 1

theorem find_g (x : ℝ) :
  4 * x ^ 4 + 2 * x ^ 2 - x + 7 + g x = x ^ 3 - 4 * x ^ 2 + 6 :=
by
  sorry

end NUMINAMATH_GPT_find_g_l1151_115159


namespace NUMINAMATH_GPT_problem_statement_l1151_115189

variable {a b c : ℝ}

theorem problem_statement (h : a < b) (hc : c < 0) : ¬ (a * c < b * c) :=
by sorry

end NUMINAMATH_GPT_problem_statement_l1151_115189


namespace NUMINAMATH_GPT_urn_gold_coins_percentage_l1151_115113

noncomputable def percentage_gold_coins_in_urn
  (total_objects : ℕ)
  (beads_percentage : ℝ)
  (rings_percentage : ℝ)
  (coins_percentage : ℝ)
  (silver_coins_percentage : ℝ)
  : ℝ := 
  let gold_coins_percentage := 100 - silver_coins_percentage
  let coins_total_percentage := total_objects * coins_percentage / 100
  coins_total_percentage * gold_coins_percentage / 100

theorem urn_gold_coins_percentage 
  (total_objects : ℕ)
  (beads_percentage rings_percentage : ℝ)
  (silver_coins_percentage : ℝ)
  (h1 : beads_percentage = 15)
  (h2 : rings_percentage = 15)
  (h3 : beads_percentage + rings_percentage = 30)
  (h4 : coins_percentage = 100 - 30)
  (h5 : silver_coins_percentage = 35)
  : percentage_gold_coins_in_urn total_objects beads_percentage rings_percentage (100 - 30) 35 = 45.5 :=
sorry

end NUMINAMATH_GPT_urn_gold_coins_percentage_l1151_115113


namespace NUMINAMATH_GPT_sum_powers_divisible_by_10_l1151_115195

theorem sum_powers_divisible_by_10 (n : ℕ) (hn : n % 4 ≠ 0) : 
  ∃ k : ℕ, 1^n + 2^n + 3^n + 4^n = 10 * k :=
  sorry

end NUMINAMATH_GPT_sum_powers_divisible_by_10_l1151_115195


namespace NUMINAMATH_GPT_minimize_at_five_halves_five_sixths_l1151_115122

noncomputable def minimize_expression (x y : ℝ) : ℝ :=
  (y - 1)^2 + (x + y - 3)^2 + (2 * x + y - 6)^2

theorem minimize_at_five_halves_five_sixths (x y : ℝ) :
  minimize_expression x y = 1 / 6 ↔ (x = 5 / 2 ∧ y = 5 / 6) :=
sorry

end NUMINAMATH_GPT_minimize_at_five_halves_five_sixths_l1151_115122


namespace NUMINAMATH_GPT_molecular_weight_CaOH2_correct_l1151_115138

/-- Molecular weight of Calcium hydroxide -/
def molecular_weight_CaOH2 (Ca O H : ℝ) : ℝ :=
  Ca + 2 * (O + H)

theorem molecular_weight_CaOH2_correct :
  molecular_weight_CaOH2 40.08 16.00 1.01 = 74.10 :=
by 
  -- This statement requires a proof that would likely involve arithmetic on real numbers
  sorry

end NUMINAMATH_GPT_molecular_weight_CaOH2_correct_l1151_115138


namespace NUMINAMATH_GPT_min_value_x2_y2_l1151_115185

theorem min_value_x2_y2 (x y : ℝ) (h : x^3 + y^3 + 3 * x * y = 1) : x^2 + y^2 ≥ 1 / 2 :=
by
  -- We are required to prove the minimum value of x^2 + y^2 given the condition is 1/2
  sorry

end NUMINAMATH_GPT_min_value_x2_y2_l1151_115185


namespace NUMINAMATH_GPT_hyperbola_equation_l1151_115177

theorem hyperbola_equation 
  {a b : ℝ} (ha : a > 0) (hb : b > 0) 
  (h_gt : a > b)
  (parallel_asymptote : ∃ k : ℝ, k = 2)
  (focus_on_line : ∃ cₓ : ℝ, ∃ c : ℝ, c = 5 ∧ cₓ = -5 ∧ (y = -2 * cₓ - 10)) :
  ∃ (a b : ℝ), (a^2 = 5) ∧ (b^2 = 20) ∧ (a^2 > b^2) ∧ c = 5 ∧ (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → (x^2 / 5 - y^2 / 20 = 1)) :=
sorry

end NUMINAMATH_GPT_hyperbola_equation_l1151_115177


namespace NUMINAMATH_GPT_SomeAthletesNotHonorSociety_l1151_115115

variable (Athletes HonorSociety : Type)
variable (Discipline : Athletes → Prop)
variable (isMember : Athletes → HonorSociety → Prop)

-- Some athletes are not disciplined
axiom AthletesNotDisciplined : ∃ a : Athletes, ¬Discipline a

-- All members of the honor society are disciplined
axiom AllHonorSocietyDisciplined : ∀ h : HonorSociety, ∀ a : Athletes, isMember a h → Discipline a

-- The theorem to be proved
theorem SomeAthletesNotHonorSociety : ∃ a : Athletes, ∀ h : HonorSociety, ¬isMember a h :=
  sorry

end NUMINAMATH_GPT_SomeAthletesNotHonorSociety_l1151_115115


namespace NUMINAMATH_GPT_range_of_a_minus_abs_b_l1151_115178

theorem range_of_a_minus_abs_b (a b : ℝ) (h1: 1 < a) (h2: a < 3) (h3: -4 < b) (h4: b < 2) : 
  -3 < a - |b| ∧ a - |b| < 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_minus_abs_b_l1151_115178


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1151_115133

-- Problem 1
theorem problem1 : (-1 : ℤ) ^ 2023 + (π - 3.14) ^ 0 - ((-1 / 2 : ℚ) ^ (-2 : ℤ)) = -4 := by
  sorry

-- Problem 2
theorem problem2 (x : ℚ) : 
  ((1 / 4 * x^4 + 2 * x^3 - 4 * x^2) / (-(2 * x))^2) = (1 / 16 * x^2 + 1 / 2 * x - 1) := by
  sorry

-- Problem 3
theorem problem3 (x y : ℚ) : 
  (2 * x + y + 1) * (2 * x + y - 1) = 4 * x^2 + 4 * x * y + y^2 - 1 := by
  sorry

-- Problem 4
theorem problem4 (x : ℚ) : 
  (2 * x + 3) * (2 * x - 3) - (2 * x - 1)^2 = 4 * x - 10 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1151_115133


namespace NUMINAMATH_GPT_relationship_between_y_coordinates_l1151_115170

theorem relationship_between_y_coordinates (b y1 y2 y3 : ℝ)
  (h1 : y1 = 3 * (-3) - b)
  (h2 : y2 = 3 * 1 - b)
  (h3 : y3 = 3 * (-1) - b) :
  y1 < y3 ∧ y3 < y2 := 
sorry

end NUMINAMATH_GPT_relationship_between_y_coordinates_l1151_115170


namespace NUMINAMATH_GPT_perpendicular_tangents_l1151_115198

theorem perpendicular_tangents (a b : ℝ) (h1 : ∀ (x y : ℝ), y = x^3 → y = (3 * x^2) * (x - 1) + 1 → y = 3 * (x - 1) + 1) (h2 : (a : ℝ) * 1 - (b : ℝ) * 1 = 2) 
 (h3 : (a : ℝ)/(b : ℝ) * 3 = -1) : a / b = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_tangents_l1151_115198


namespace NUMINAMATH_GPT_find_sale_in_fourth_month_l1151_115141

variable (sale1 sale2 sale3 sale5 sale6 : ℕ)
variable (TotalSales : ℕ)
variable (AverageSales : ℕ)

theorem find_sale_in_fourth_month (h1 : sale1 = 6335)
                                   (h2 : sale2 = 6927)
                                   (h3 : sale3 = 6855)
                                   (h4 : sale5 = 6562)
                                   (h5 : sale6 = 5091)
                                   (h6 : AverageSales = 6500)
                                   (h7 : TotalSales = AverageSales * 6) :
  ∃ sale4, TotalSales = sale1 + sale2 + sale3 + sale4 + sale5 + sale6 ∧ sale4 = 7230 :=
by
  sorry

end NUMINAMATH_GPT_find_sale_in_fourth_month_l1151_115141


namespace NUMINAMATH_GPT_intersection_union_complement_l1151_115104

open Set

variable (U : Set ℝ)
variable (A B : Set ℝ)

def universal_set := U = univ
def set_A := A = {x : ℝ | -1 ≤ x ∧ x < 2}
def set_B := B = {x : ℝ | 1 < x ∧ x ≤ 3}

theorem intersection (hU : U = univ) (hA : A = {x : ℝ | -1 ≤ x ∧ x < 2}) (hB : B = {x : ℝ | 1 < x ∧ x ≤ 3}) :
  A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := sorry

theorem union (hU : U = univ) (hA : A = {x : ℝ | -1 ≤ x ∧ x < 2}) (hB : B = {x : ℝ | 1 < x ∧ x ≤ 3}) :
  A ∪ B = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := sorry

theorem complement (hU : U = univ) (hA : A = {x : ℝ | -1 ≤ x ∧ x < 2}) :
  U \ A = {x : ℝ | x < -1 ∨ 2 ≤ x} := sorry

end NUMINAMATH_GPT_intersection_union_complement_l1151_115104


namespace NUMINAMATH_GPT_transform_expression_l1151_115199

theorem transform_expression (y Q : ℝ) (h : 5 * (3 * y + 7 * Real.pi) = Q) : 
  10 * (6 * y + 14 * Real.pi + 3) = 4 * Q + 30 := 
by 
  sorry

end NUMINAMATH_GPT_transform_expression_l1151_115199


namespace NUMINAMATH_GPT_solution_interval_l1151_115112

theorem solution_interval (x : ℝ) : 2 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 5 ↔ (5 / 2 : ℝ) < x ∧ x ≤ (14 / 5 : ℝ) := 
by
  sorry

end NUMINAMATH_GPT_solution_interval_l1151_115112


namespace NUMINAMATH_GPT_turtles_still_on_sand_l1151_115101

-- Define the total number of baby sea turtles
def total_turtles := 42

-- Define the function for calculating the number of swept turtles
def swept_turtles (total : Nat) : Nat := total / 3

-- Define the function for calculating the number of turtles still on the sand
def turtles_on_sand (total : Nat) (swept : Nat) : Nat := total - swept

-- Set parameters for the proof
def swept := swept_turtles total_turtles
def on_sand := turtles_on_sand total_turtles swept

-- Prove the statement
theorem turtles_still_on_sand : on_sand = 28 :=
by
  -- proof steps to be added here
  sorry

end NUMINAMATH_GPT_turtles_still_on_sand_l1151_115101


namespace NUMINAMATH_GPT_special_collection_books_l1151_115146

theorem special_collection_books (loaned_books : ℕ) (returned_percentage : ℝ) (end_of_month_books : ℕ)
    (H1 : loaned_books = 160)
    (H2 : returned_percentage = 0.65)
    (H3 : end_of_month_books = 244) :
    let books_returned := returned_percentage * loaned_books
    let books_not_returned := loaned_books - books_returned
    let original_books := end_of_month_books + books_not_returned
    original_books = 300 :=
by
  sorry

end NUMINAMATH_GPT_special_collection_books_l1151_115146


namespace NUMINAMATH_GPT_printer_Z_time_l1151_115110

theorem printer_Z_time (T_Z : ℝ) (h1 : (1.0 / 15.0 : ℝ) = (15.0 * ((1.0 / 12.0) + (1.0 / T_Z))) / 2.0833333333333335) : 
  T_Z = 18.0 :=
sorry

end NUMINAMATH_GPT_printer_Z_time_l1151_115110


namespace NUMINAMATH_GPT_stamps_total_l1151_115148

theorem stamps_total (x : ℕ) (a_initial : ℕ := 5 * x) (b_initial : ℕ := 4 * x)
                     (a_after : ℕ := a_initial - 5) (b_after : ℕ := b_initial + 5)
                     (h_ratio_initial : a_initial / b_initial = 5 / 4)
                     (h_ratio_final : a_after / b_after = 4 / 5) :
                     a_initial + b_initial = 45 :=
by
  sorry

end NUMINAMATH_GPT_stamps_total_l1151_115148


namespace NUMINAMATH_GPT_moles_of_CH4_needed_l1151_115158

theorem moles_of_CH4_needed
  (moles_C6H6_needed : ℕ)
  (reaction_balance : ∀ (C6H6 CH4 C6H5CH3 H2 : ℕ), 
    C6H6 + CH4 = C6H5CH3 + H2 → C6H6 = 1 ∧ CH4 = 1 ∧ C6H5CH3 = 1 ∧ H2 = 1)
  (H : moles_C6H6_needed = 3) :
  (3 : ℕ) = 3 :=
by 
  -- The actual proof would go here
  sorry

end NUMINAMATH_GPT_moles_of_CH4_needed_l1151_115158


namespace NUMINAMATH_GPT_log21_requires_additional_information_l1151_115130

noncomputable def log3 : ℝ := 0.4771
noncomputable def log5 : ℝ := 0.6990

theorem log21_requires_additional_information
  (log3_given : log3 = 0.4771)
  (log5_given : log5 = 0.6990) :
  ¬ (∃ c₁ c₂ : ℝ, log21 = c₁ * log3 + c₂ * log5) :=
sorry

end NUMINAMATH_GPT_log21_requires_additional_information_l1151_115130


namespace NUMINAMATH_GPT_theater_seat_count_l1151_115123

theorem theater_seat_count (number_of_people : ℕ) (empty_seats : ℕ) (total_seats : ℕ) 
  (h1 : number_of_people = 532) 
  (h2 : empty_seats = 218) 
  (h3 : total_seats = number_of_people + empty_seats) : 
  total_seats = 750 := 
by 
  sorry

end NUMINAMATH_GPT_theater_seat_count_l1151_115123


namespace NUMINAMATH_GPT_long_show_episodes_correct_l1151_115131

variable {short_show_episodes : ℕ} {short_show_duration : ℕ} {total_watched_time : ℕ} {long_show_episode_duration : ℕ}

def episodes_long_show (short_episodes_duration total_duration long_episode_duration : ℕ) : ℕ :=
  (total_duration - short_episodes_duration) / long_episode_duration

theorem long_show_episodes_correct :
  ∀ (short_show_episodes short_show_duration total_watched_time long_show_episode_duration : ℕ),
  short_show_episodes = 24 →
  short_show_duration = 1 / 2 →
  total_watched_time = 24 →
  long_show_episode_duration = 1 →
  episodes_long_show (short_show_episodes * short_show_duration) total_watched_time long_show_episode_duration = 12 := by
  intros
  sorry

end NUMINAMATH_GPT_long_show_episodes_correct_l1151_115131


namespace NUMINAMATH_GPT_cuboid_height_l1151_115137

theorem cuboid_height (l b A : ℝ) (hl : l = 10) (hb : b = 8) (hA : A = 480) :
  ∃ h : ℝ, A = 2 * (l * b + b * h + l * h) ∧ h = 320 / 36 := by
  sorry

end NUMINAMATH_GPT_cuboid_height_l1151_115137


namespace NUMINAMATH_GPT_complex_number_z_satisfies_l1151_115120

theorem complex_number_z_satisfies (z : ℂ) : 
  (z * (1 + I) + (-I) * (1 - I) = 0) → z = -1 := 
by {
  sorry
}

end NUMINAMATH_GPT_complex_number_z_satisfies_l1151_115120


namespace NUMINAMATH_GPT_a_5_is_31_l1151_115165

/-- Define the sequence a_n recursively -/
def a : Nat → Nat
| 0        => 1
| (n + 1)  => 2 * a n + 1

/-- Prove that the 5th term in the sequence is 31 -/
theorem a_5_is_31 : a 5 = 31 := 
sorry

end NUMINAMATH_GPT_a_5_is_31_l1151_115165


namespace NUMINAMATH_GPT_reduced_price_per_kg_l1151_115108

variable (P R Q : ℝ)

theorem reduced_price_per_kg :
  R = 0.75 * P →
  1200 = (Q + 5) * R →
  Q * P = 1200 →
  R = 60 :=
by
  intro h₁ h₂ h₃
  sorry

end NUMINAMATH_GPT_reduced_price_per_kg_l1151_115108


namespace NUMINAMATH_GPT_longer_side_length_l1151_115144

-- Define the conditions as parameters
variables (W : ℕ) (poles : ℕ) (distance : ℕ) (P : ℕ)

-- Assume the fixed conditions given in the problem
axiom shorter_side : W = 10
axiom number_of_poles : poles = 24
axiom distance_between_poles : distance = 5

-- Define the total perimeter based on the number of segments formed by the poles
noncomputable def perimeter (poles : ℕ) (distance : ℕ) : ℕ :=
  (poles - 4) * distance

-- The total perimeter of the rectangle
axiom total_perimeter : P = perimeter poles distance

-- Definition of the perimeter of the rectangle in terms of its sides
axiom rectangle_perimeter : ∀ (L W : ℕ), P = 2 * L + 2 * W

-- The theorem we need to prove
theorem longer_side_length (L : ℕ) : L = 40 :=
by
  -- Sorry is used to skip the actual proof for now
  sorry

end NUMINAMATH_GPT_longer_side_length_l1151_115144


namespace NUMINAMATH_GPT_Peter_initially_had_33_marbles_l1151_115150

-- Definitions based on conditions
def lostMarbles : Nat := 15
def currentMarbles : Nat := 18

-- Definition for the initial marbles calculation
def initialMarbles (lostMarbles : Nat) (currentMarbles : Nat) : Nat :=
  lostMarbles + currentMarbles

-- Theorem statement
theorem Peter_initially_had_33_marbles : initialMarbles lostMarbles currentMarbles = 33 := by
  sorry

end NUMINAMATH_GPT_Peter_initially_had_33_marbles_l1151_115150


namespace NUMINAMATH_GPT_distance_ran_each_morning_l1151_115100

-- Definitions based on conditions
def days_ran : ℕ := 3
def total_distance : ℕ := 2700

-- The goal is to prove the distance ran each morning
theorem distance_ran_each_morning : total_distance / days_ran = 900 :=
by
  sorry

end NUMINAMATH_GPT_distance_ran_each_morning_l1151_115100


namespace NUMINAMATH_GPT_factor_expression_l1151_115125

theorem factor_expression (x : ℝ) : 
  (9 * x^5 + 25 * x^3 - 4) - (x^5 - 3 * x^3 - 4) = 4 * x^3 * (2 * x^2 + 7) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1151_115125


namespace NUMINAMATH_GPT_original_balance_l1151_115118

variable (x : ℝ)
variable (y : ℝ)
variable (z : ℝ)

theorem original_balance (decrease_percentage : ℝ) (current_balance : ℝ) (original_balance : ℝ) :
  decrease_percentage = 0.10 → current_balance = 90000 → 
  current_balance = (1 - decrease_percentage) * original_balance → 
  original_balance = 100000 := by
  sorry

end NUMINAMATH_GPT_original_balance_l1151_115118


namespace NUMINAMATH_GPT_polynomial_roots_bounds_l1151_115119

theorem polynomial_roots_bounds (p : ℝ) :
  (∃ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2 ∧ (x1^4 + 3*p*x1^3 + x1^2 + 3*p*x1 + 1 = 0) ∧ (x2^4 + 3*p*x2^3 + x2^2 + 3*p*x2 + 1 = 0)) ↔ p ∈ Set.Iio (1 / 4) := by
sorry

end NUMINAMATH_GPT_polynomial_roots_bounds_l1151_115119


namespace NUMINAMATH_GPT_rhombus_area_l1151_115139

theorem rhombus_area 
  (a b : ℝ)
  (side_length : ℝ)
  (diff_diag : ℝ)
  (h_side_len : side_length = Real.sqrt 89)
  (h_diff_diag : diff_diag = 6)
  (h_diag : a - b = diff_diag ∨ b - a = diff_diag)
  (h_side_eq : side_length = Real.sqrt (a^2 + b^2)) :
  (1 / 2 * a * b) * 4 = 80 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_area_l1151_115139


namespace NUMINAMATH_GPT_count_three_digit_distinct_under_800_l1151_115157

-- Definitions
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 800
def distinct_digits (n : ℕ) : Prop := (n / 100 ≠ (n / 10) % 10) ∧ (n / 100 ≠ n % 10) ∧ ((n / 10) % 10 ≠ n % 10) 

-- Theorem
theorem count_three_digit_distinct_under_800 : ∃ k : ℕ, k = 504 ∧ ∀ n : ℕ, is_three_digit n → distinct_digits n → n < 800 :=
by 
  exists 504
  sorry

end NUMINAMATH_GPT_count_three_digit_distinct_under_800_l1151_115157


namespace NUMINAMATH_GPT_value_of_expression_l1151_115183

theorem value_of_expression (x : ℝ) (h : x^2 - 5 * x + 6 < 0) : x^2 - 5 * x + 10 = 4 :=
sorry

end NUMINAMATH_GPT_value_of_expression_l1151_115183


namespace NUMINAMATH_GPT_coords_of_A_l1151_115163

theorem coords_of_A :
  ∃ (x y : ℝ), y = Real.exp x ∧ (Real.exp x = 1) ∧ y = 1 :=
by
  use 0, 1
  have hx : Real.exp 0 = 1 := Real.exp_zero
  have hy : 1 = Real.exp 0 := hx.symm
  exact ⟨hy, hx, rfl⟩

end NUMINAMATH_GPT_coords_of_A_l1151_115163


namespace NUMINAMATH_GPT_gasohol_problem_l1151_115197

noncomputable def initial_gasohol_volume (x : ℝ) : Prop :=
  let ethanol_in_initial_mix := 0.05 * x
  let ethanol_to_add := 2
  let total_ethanol := ethanol_in_initial_mix + ethanol_to_add
  let total_volume := x + 2
  0.1 * total_volume = total_ethanol

theorem gasohol_problem (x : ℝ) : initial_gasohol_volume x → x = 36 := by
  intro h
  sorry

end NUMINAMATH_GPT_gasohol_problem_l1151_115197


namespace NUMINAMATH_GPT_combine_terms_implies_mn_l1151_115136

theorem combine_terms_implies_mn {m n : ℕ} (h1 : m = 2) (h2 : n = 3) : m ^ n = 8 :=
by
  -- We will skip the proof here
  sorry

end NUMINAMATH_GPT_combine_terms_implies_mn_l1151_115136


namespace NUMINAMATH_GPT_derivative_quadrant_l1151_115176

theorem derivative_quadrant (b c : ℝ) (H_b : b = -4) : ¬ ∃ x y : ℝ, x < 0 ∧ y > 0 ∧ 2*x + b = y := by
  sorry

end NUMINAMATH_GPT_derivative_quadrant_l1151_115176


namespace NUMINAMATH_GPT_minimum_trucks_needed_l1151_115103

theorem minimum_trucks_needed (total_weight : ℝ) (box_weight : ℕ → ℝ) 
  (n : ℕ) (H_total_weight : total_weight = 10) 
  (H_box_weight : ∀ i, box_weight i ≤ 1) 
  (truck_capacity : ℝ) 
  (H_truck_capacity : truck_capacity = 3) : 
  n = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_minimum_trucks_needed_l1151_115103


namespace NUMINAMATH_GPT_inequality_and_equality_condition_l1151_115192

theorem inequality_and_equality_condition (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ab : 1 ≤ a * b) :
  (1 / (1 + a) + 1 / (1 + b) ≤ 1) ∧ (1 / (1 + a) + 1 / (1 + b) = 1 ↔ a * b = 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_and_equality_condition_l1151_115192


namespace NUMINAMATH_GPT_cheetahs_pandas_ratio_l1151_115107

-- Let C denote the number of cheetahs 5 years ago.
-- Let P denote the number of pandas 5 years ago.
-- The conditions given are:
-- 1. The ratio of cheetahs to pandas 5 years ago was the same as it is now.
-- 2. The number of cheetahs has increased by 2.
-- 3. The number of pandas has increased by 6.
-- We need to prove that the current ratio of cheetahs to pandas is C / P.

theorem cheetahs_pandas_ratio
  (C P : ℕ)
  (h1 : C / P = (C + 2) / (P + 6)) :
  (C + 2) / (P + 6) = C / P :=
by sorry

end NUMINAMATH_GPT_cheetahs_pandas_ratio_l1151_115107


namespace NUMINAMATH_GPT_john_total_expenses_l1151_115196

theorem john_total_expenses :
  (let epiPenCost := 500
   let yearlyMedicalExpenses := 2000
   let firstEpiPenInsuranceCoverage := 0.75
   let secondEpiPenInsuranceCoverage := 0.60
   let medicalExpensesCoverage := 0.80
   let firstEpiPenCost := epiPenCost * (1 - firstEpiPenInsuranceCoverage)
   let secondEpiPenCost := epiPenCost * (1 - secondEpiPenInsuranceCoverage)
   let totalEpiPenCost := firstEpiPenCost + secondEpiPenCost
   let yearlyMedicalExpensesCost := yearlyMedicalExpenses * (1 - medicalExpensesCoverage)
   let totalCost := totalEpiPenCost + yearlyMedicalExpensesCost
   totalCost) = 725 := sorry

end NUMINAMATH_GPT_john_total_expenses_l1151_115196


namespace NUMINAMATH_GPT_equilateral_triangle_perimeter_l1151_115149

theorem equilateral_triangle_perimeter (a P : ℕ) 
  (h1 : 2 * a + 10 = 40)  -- Condition: perimeter of isosceles triangle is 40
  (h2 : P = 3 * a) :      -- Definition of perimeter of equilateral triangle
  P = 45 :=               -- Expected result
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_perimeter_l1151_115149


namespace NUMINAMATH_GPT_Patrick_hours_less_than_twice_Greg_l1151_115151

def J := 18
def G := J - 6
def total_hours := 50
def P : ℕ := sorry -- To be defined, we need to establish the proof later with the condition J + G + P = 50
def X : ℕ := sorry -- To be defined, we need to establish the proof later with the condition P = 2 * G - X

theorem Patrick_hours_less_than_twice_Greg : X = 4 := by
  -- Placeholder definitions for P and X based on the given conditions
  let P := total_hours - (J + G)
  let X := 2 * G - P
  sorry -- Proof details to be filled in

end NUMINAMATH_GPT_Patrick_hours_less_than_twice_Greg_l1151_115151


namespace NUMINAMATH_GPT_base_6_digit_divisibility_l1151_115184

theorem base_6_digit_divisibility (d : ℕ) (h1 : d < 6) : ∃ t : ℤ, (655 + 42 * d) = 13 * t :=
by sorry

end NUMINAMATH_GPT_base_6_digit_divisibility_l1151_115184


namespace NUMINAMATH_GPT_smallest_white_erasers_l1151_115105

def total_erasers (n : ℕ) (pink : ℕ) (orange : ℕ) (purple : ℕ) (white : ℕ) : Prop :=
  pink = n / 5 ∧ orange = n / 6 ∧ purple = 10 ∧ white = n - (pink + orange + purple)

theorem smallest_white_erasers : ∃ n : ℕ, ∃ pink : ℕ, ∃ orange : ℕ, ∃ purple : ℕ, ∃ white : ℕ,
  total_erasers n pink orange purple white ∧ white = 9 := sorry

end NUMINAMATH_GPT_smallest_white_erasers_l1151_115105


namespace NUMINAMATH_GPT_each_boy_brought_nine_cups_l1151_115145

/--
There are 30 students in Ms. Leech's class. Twice as many girls as boys are in the class.
There are 10 boys in the class and the total number of cups brought by the students 
in the class is 90. Prove that each boy brought 9 cups.
-/
theorem each_boy_brought_nine_cups (students girls boys cups : ℕ) 
  (h1 : students = 30) 
  (h2 : girls = 2 * boys) 
  (h3 : boys = 10) 
  (h4 : cups = 90) 
  : cups / boys = 9 := 
sorry

end NUMINAMATH_GPT_each_boy_brought_nine_cups_l1151_115145


namespace NUMINAMATH_GPT_rachel_math_homework_l1151_115140

theorem rachel_math_homework (reading_hw math_hw : ℕ) 
  (h1 : reading_hw = 4) 
  (h2 : math_hw = reading_hw + 3) : 
  math_hw = 7 := by
  sorry

end NUMINAMATH_GPT_rachel_math_homework_l1151_115140


namespace NUMINAMATH_GPT_players_started_first_half_l1151_115114

variable (total_players : Nat)
variable (first_half_substitutions : Nat)
variable (second_half_substitutions : Nat)
variable (players_not_playing : Nat)

theorem players_started_first_half :
  total_players = 24 →
  first_half_substitutions = 2 →
  second_half_substitutions = 2 * first_half_substitutions →
  players_not_playing = 7 →
  let total_substitutions := first_half_substitutions + second_half_substitutions 
  let players_played := total_players - players_not_playing
  ∃ S, S + total_substitutions = players_played ∧ S = 11 := 
by
  sorry

end NUMINAMATH_GPT_players_started_first_half_l1151_115114


namespace NUMINAMATH_GPT_surface_area_increase_l1151_115187

structure RectangularSolid (length : ℝ) (width : ℝ) (height : ℝ) where
  surface_area : ℝ := 2 * (length * width + length * height + width * height)

def cube_surface_contributions (side : ℝ) : ℝ := side ^ 2 * 3

theorem surface_area_increase
  (original : RectangularSolid 4 3 5)
  (cube_side : ℝ := 1) :
  let new_cube_contribution := cube_surface_contributions cube_side
  let removed_face : ℝ := cube_side ^ 2
  let original_surface_area := original.surface_area
  original_surface_area + new_cube_contribution - removed_face = original_surface_area + 2 :=
by
  sorry

end NUMINAMATH_GPT_surface_area_increase_l1151_115187
