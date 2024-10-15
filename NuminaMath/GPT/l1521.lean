import Mathlib

namespace NUMINAMATH_GPT_Roger_first_bag_candies_is_11_l1521_152157

-- Define the conditions
def Sandra_bags : ℕ := 2
def Sandra_candies_per_bag : ℕ := 6
def Roger_bags : ℕ := 2
def Roger_second_bag_candies : ℕ := 3
def Extra_candies_Roger_has_than_Sandra : ℕ := 2

-- Define the total candy for Sandra
def Sandra_total_candies : ℕ := Sandra_bags * Sandra_candies_per_bag

-- Using the conditions, we define the total candy for Roger
def Roger_total_candies : ℕ := Sandra_total_candies + Extra_candies_Roger_has_than_Sandra

-- Define the candy in Roger's first bag
def Roger_first_bag_candies : ℕ := Roger_total_candies - Roger_second_bag_candies

-- The proof statement we need to prove
theorem Roger_first_bag_candies_is_11 : Roger_first_bag_candies = 11 := by
  sorry

end NUMINAMATH_GPT_Roger_first_bag_candies_is_11_l1521_152157


namespace NUMINAMATH_GPT_positional_relationship_l1521_152145

theorem positional_relationship (r PO QO : ℝ) (h_r : r = 6) (h_PO : PO = 4) (h_QO : QO = 6) :
  (PO < r) ∧ (QO = r) :=
by
  sorry

end NUMINAMATH_GPT_positional_relationship_l1521_152145


namespace NUMINAMATH_GPT_proportion1_proportion2_l1521_152139

theorem proportion1 (x : ℚ) : (x / (5 / 9) = (1 / 20) / (1 / 3)) → x = 1 / 12 :=
sorry

theorem proportion2 (x : ℚ) : (x / 0.25 = 0.5 / 0.1) → x = 1.25 :=
sorry

end NUMINAMATH_GPT_proportion1_proportion2_l1521_152139


namespace NUMINAMATH_GPT_lambs_total_l1521_152185

/-
Each of farmer Cunningham's lambs is either black or white.
There are 193 white lambs, and 5855 black lambs.
Prove that the total number of lambs is 6048.
-/

theorem lambs_total (white_lambs : ℕ) (black_lambs : ℕ) (h1 : white_lambs = 193) (h2 : black_lambs = 5855) :
  white_lambs + black_lambs = 6048 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_lambs_total_l1521_152185


namespace NUMINAMATH_GPT_ab_zero_l1521_152126

theorem ab_zero (a b : ℝ)
  (h1 : a + b = 5)
  (h2 : a^3 + b^3 = 125) : 
  a * b = 0 :=
by
  sorry

end NUMINAMATH_GPT_ab_zero_l1521_152126


namespace NUMINAMATH_GPT_temperature_at_midnight_l1521_152131

def morning_temp : ℝ := 30
def afternoon_increase : ℝ := 1
def midnight_decrease : ℝ := 7

theorem temperature_at_midnight : morning_temp + afternoon_increase - midnight_decrease = 24 := by
  sorry

end NUMINAMATH_GPT_temperature_at_midnight_l1521_152131


namespace NUMINAMATH_GPT_remainder_415_pow_420_div_16_l1521_152182

theorem remainder_415_pow_420_div_16 : 415^420 % 16 = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_415_pow_420_div_16_l1521_152182


namespace NUMINAMATH_GPT_sets_are_equal_l1521_152178

def setA : Set ℤ := {a | ∃ m n l : ℤ, a = 12 * m + 8 * n + 4 * l}
def setB : Set ℤ := {b | ∃ p q r : ℤ, b = 20 * p + 16 * q + 12 * r}

theorem sets_are_equal : setA = setB := sorry

end NUMINAMATH_GPT_sets_are_equal_l1521_152178


namespace NUMINAMATH_GPT_remaining_volume_correct_l1521_152168

noncomputable def diameter_sphere : ℝ := 24
noncomputable def radius_sphere : ℝ := diameter_sphere / 2
noncomputable def height_hole1 : ℝ := 10
noncomputable def diameter_hole1 : ℝ := 3
noncomputable def radius_hole1 : ℝ := diameter_hole1 / 2
noncomputable def height_hole2 : ℝ := 10
noncomputable def diameter_hole2 : ℝ := 3
noncomputable def radius_hole2 : ℝ := diameter_hole2 / 2
noncomputable def height_hole3 : ℝ := 5
noncomputable def diameter_hole3 : ℝ := 4
noncomputable def radius_hole3 : ℝ := diameter_hole3 / 2

noncomputable def volume_sphere : ℝ := (4 / 3) * Real.pi * (radius_sphere ^ 3)
noncomputable def volume_hole1 : ℝ := Real.pi * (radius_hole1 ^ 2) * height_hole1
noncomputable def volume_hole2 : ℝ := Real.pi * (radius_hole2 ^ 2) * height_hole2
noncomputable def volume_hole3 : ℝ := Real.pi * (radius_hole3 ^ 2) * height_hole3

noncomputable def remaining_volume : ℝ := 
  volume_sphere - (2 * volume_hole1 + volume_hole3)

theorem remaining_volume_correct : remaining_volume = 2239 * Real.pi := by
  sorry

end NUMINAMATH_GPT_remaining_volume_correct_l1521_152168


namespace NUMINAMATH_GPT_combined_swim_time_l1521_152169

theorem combined_swim_time 
    (freestyle_time: ℕ)
    (backstroke_without_factors: ℕ)
    (backstroke_with_factors: ℕ)
    (butterfly_without_factors: ℕ)
    (butterfly_with_factors: ℕ)
    (breaststroke_without_factors: ℕ)
    (breaststroke_with_factors: ℕ) :
    freestyle_time = 48 ∧
    backstroke_without_factors = freestyle_time + 4 ∧
    backstroke_with_factors = backstroke_without_factors + 2 ∧
    butterfly_without_factors = backstroke_without_factors + 3 ∧
    butterfly_with_factors = butterfly_without_factors + 3 ∧
    breaststroke_without_factors = butterfly_without_factors + 2 ∧
    breaststroke_with_factors = breaststroke_without_factors - 1 →
    freestyle_time + backstroke_with_factors + butterfly_with_factors + breaststroke_with_factors = 216 :=
by
  sorry

end NUMINAMATH_GPT_combined_swim_time_l1521_152169


namespace NUMINAMATH_GPT_triangle_area_proof_l1521_152170

def vector2 := ℝ × ℝ

def a : vector2 := (6, 3)
def b : vector2 := (-4, 5)

noncomputable def det (u v : vector2) : ℝ := u.1 * v.2 - u.2 * v.1

noncomputable def parallelogram_area (u v : vector2) : ℝ := |det u v|

noncomputable def triangle_area (u v : vector2) : ℝ := parallelogram_area u v / 2

theorem triangle_area_proof : triangle_area a b = 21 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_area_proof_l1521_152170


namespace NUMINAMATH_GPT_area_of_square_field_l1521_152151

theorem area_of_square_field (d : ℝ) (s : ℝ) (A : ℝ) (h_d : d = 28) (h_relation : d = s * Real.sqrt 2) (h_area : A = s^2) :
  A = 391.922 :=
by sorry

end NUMINAMATH_GPT_area_of_square_field_l1521_152151


namespace NUMINAMATH_GPT_no_four_consecutive_perf_square_l1521_152194

theorem no_four_consecutive_perf_square :
  ¬ ∃ (x : ℕ), x > 0 ∧ ∃ (k : ℕ), x * (x + 1) * (x + 2) * (x + 3) = k^2 :=
by
  sorry

end NUMINAMATH_GPT_no_four_consecutive_perf_square_l1521_152194


namespace NUMINAMATH_GPT_f_20_plus_f_neg20_l1521_152186

noncomputable def f (a b : ℝ) (x : ℝ) := a * x^4 + b * x^2 + 5

theorem f_20_plus_f_neg20 (a b : ℝ) (h : f a b 20 = 3) : f a b 20 + f a b (-20) = 6 := by
  sorry

end NUMINAMATH_GPT_f_20_plus_f_neg20_l1521_152186


namespace NUMINAMATH_GPT_total_pebbles_l1521_152191

theorem total_pebbles (white_pebbles : ℕ) (red_pebbles : ℕ)
  (h1 : white_pebbles = 20)
  (h2 : red_pebbles = white_pebbles / 2) :
  white_pebbles + red_pebbles = 30 := by
  sorry

end NUMINAMATH_GPT_total_pebbles_l1521_152191


namespace NUMINAMATH_GPT_quadratic_real_solutions_l1521_152104

theorem quadratic_real_solutions (x y : ℝ) :
  (∃ z : ℝ, 16 * z^2 + 4 * x * y * z + (y^2 - 3) = 0) ↔ (x ≤ -2 ∨ x ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_solutions_l1521_152104


namespace NUMINAMATH_GPT_equivalent_sum_of_exponents_l1521_152161

theorem equivalent_sum_of_exponents : 3^3 + 3^3 + 3^3 = 3^4 :=
by
  sorry

end NUMINAMATH_GPT_equivalent_sum_of_exponents_l1521_152161


namespace NUMINAMATH_GPT_total_people_on_playground_l1521_152158

open Nat

-- Conditions
def num_girls := 28
def num_boys := 35
def num_3rd_grade_girls := 15
def num_3rd_grade_boys := 18
def num_teachers := 4

-- Derived values (from conditions)
def num_4th_grade_girls := num_girls - num_3rd_grade_girls
def num_4th_grade_boys := num_boys - num_3rd_grade_boys
def num_3rd_graders := num_3rd_grade_girls + num_3rd_grade_boys
def num_4th_graders := num_4th_grade_girls + num_4th_grade_boys

-- Total number of people
def total_people := num_3rd_graders + num_4th_graders + num_teachers

-- Proof statement
theorem total_people_on_playground : total_people = 67 :=
  by
     -- This is where the proof would go
     sorry

end NUMINAMATH_GPT_total_people_on_playground_l1521_152158


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1521_152156

theorem quadratic_inequality_solution : 
  ∀ x : ℝ, (2 * x ^ 2 + 7 * x + 3 > 0) ↔ (x < -3 ∨ x > -0.5) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1521_152156


namespace NUMINAMATH_GPT_chess_mixed_games_l1521_152173

theorem chess_mixed_games (W M : ℕ) (hW : W * (W - 1) / 2 = 45) (hM : M * (M - 1) / 2 = 190) : M * W = 200 :=
by
  sorry

end NUMINAMATH_GPT_chess_mixed_games_l1521_152173


namespace NUMINAMATH_GPT_sets_of_consecutive_integers_summing_to_20_l1521_152192

def sum_of_consecutive_integers (a n : ℕ) : ℕ := n * a + (n * (n - 1)) / 2

theorem sets_of_consecutive_integers_summing_to_20 : 
  (∃ (a n : ℕ), n ≥ 2 ∧ sum_of_consecutive_integers a n = 20) ∧ 
  (∀ (a1 n1 a2 n2 : ℕ), 
    (n1 ≥ 2 ∧ sum_of_consecutive_integers a1 n1 = 20 ∧ 
    n2 ≥ 2 ∧ sum_of_consecutive_integers a2 n2 = 20) → 
    (a1 = a2 ∧ n1 = n2)) :=
sorry

end NUMINAMATH_GPT_sets_of_consecutive_integers_summing_to_20_l1521_152192


namespace NUMINAMATH_GPT_maximum_area_of_rectangle_with_fixed_perimeter_l1521_152119

theorem maximum_area_of_rectangle_with_fixed_perimeter (x y : ℝ) 
  (h₁ : 2 * (x + y) = 40) 
  (h₂ : x = y) :
  x * y = 100 :=
by
  sorry

end NUMINAMATH_GPT_maximum_area_of_rectangle_with_fixed_perimeter_l1521_152119


namespace NUMINAMATH_GPT_calculate_expression_l1521_152117

theorem calculate_expression :
  (1/4 * 6.16^2) - (4 * 1.04^2) = 5.16 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1521_152117


namespace NUMINAMATH_GPT_sundae_cost_l1521_152121

theorem sundae_cost (ice_cream_cost toppings_cost : ℕ) (num_toppings : ℕ) :
  ice_cream_cost = 200  →
  toppings_cost = 50 →
  num_toppings = 10 →
  ice_cream_cost + num_toppings * toppings_cost = 700 := by
  sorry

end NUMINAMATH_GPT_sundae_cost_l1521_152121


namespace NUMINAMATH_GPT_calculation_correct_l1521_152147

theorem calculation_correct (x y : ℝ) (hx0 : x ≠ 0) (hy0 : y ≠ 0) (hxy : x = 2 * y) : 
  (x - 2 / x) * (y + 2 / y) = 1 / 2 * (x^2 - 2 * x + 8 - 16 / x) := 
by 
  sorry

end NUMINAMATH_GPT_calculation_correct_l1521_152147


namespace NUMINAMATH_GPT_circles_disjoint_l1521_152155

-- Definitions of the circles
def circleM (x y : ℝ) : Prop := x^2 + y^2 = 1
def circleN (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

-- Prove that the circles are disjoint
theorem circles_disjoint : 
  (¬ ∃ (x y : ℝ), circleM x y ∧ circleN x y) :=
by sorry

end NUMINAMATH_GPT_circles_disjoint_l1521_152155


namespace NUMINAMATH_GPT_factor_expression_l1521_152150

theorem factor_expression (a b c d : ℝ) : 
  a * (b - c)^3 + b * (c - d)^3 + c * (d - a)^3 + d * (a - b)^3 
        = ((a - b) * (b - c) * (c - d) * (d - a)) * (a + b + c + d) := 
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1521_152150


namespace NUMINAMATH_GPT_sin_pi_minus_2alpha_l1521_152189

theorem sin_pi_minus_2alpha (α : ℝ) (h1 : Real.sin (π / 2 + α) = -3 / 5) (h2 : π / 2 < α ∧ α < π) : 
  Real.sin (π - 2 * α) = -24 / 25 := by
  sorry

end NUMINAMATH_GPT_sin_pi_minus_2alpha_l1521_152189


namespace NUMINAMATH_GPT_roots_of_polynomial_l1521_152184

def P (x : ℝ) : ℝ := x^3 - 3 * x^2 - x + 3

theorem roots_of_polynomial :
  {x : ℝ | P x = 0} = {1, -1, 3} := 
sorry

end NUMINAMATH_GPT_roots_of_polynomial_l1521_152184


namespace NUMINAMATH_GPT_total_cost_correct_l1521_152180

-- Define the costs for each repair
def engine_labor_cost := 75 * 16
def engine_part_cost := 1200
def brake_labor_cost := 85 * 10
def brake_part_cost := 800
def tire_labor_cost := 50 * 4
def tire_part_cost := 600

-- Calculate the total costs
def engine_total_cost := engine_labor_cost + engine_part_cost
def brake_total_cost := brake_labor_cost + brake_part_cost
def tire_total_cost := tire_labor_cost + tire_part_cost

-- Calculate the total combined cost
def total_combined_cost := engine_total_cost + brake_total_cost + tire_total_cost

-- The theorem to prove
theorem total_cost_correct : total_combined_cost = 4850 := by
  sorry

end NUMINAMATH_GPT_total_cost_correct_l1521_152180


namespace NUMINAMATH_GPT_surface_area_of_cube_l1521_152107

-- Define the condition: volume of the cube is 1728 cubic centimeters
def volume_cube (s : ℝ) : ℝ := s^3
def given_volume : ℝ := 1728

-- Define the question: surface area of the cube
def surface_area_cube (s : ℝ) : ℝ := 6 * s^2

-- The statement that needs to be proved
theorem surface_area_of_cube :
  ∃ s : ℝ, volume_cube s = given_volume → surface_area_cube s = 864 :=
by
  sorry

end NUMINAMATH_GPT_surface_area_of_cube_l1521_152107


namespace NUMINAMATH_GPT_compute_value_l1521_152112

variable (p q : ℚ)
variable (h : ∀ x, 3 * x^2 - 7 * x - 6 = 0 → x = p ∨ x = q)

theorem compute_value (h_pq : p ≠ q) : (5 * p^3 - 5 * q^3) * (p - q)⁻¹ = 335 / 9 := by
  -- We assume p and q are the roots of the polynomial and p ≠ q.
  have sum_roots : p + q = 7 / 3 := sorry
  have prod_roots : p * q = -2 := sorry
  -- Additional steps to derive the required result (proof) are ignored here.
  sorry

end NUMINAMATH_GPT_compute_value_l1521_152112


namespace NUMINAMATH_GPT_apple_tree_production_l1521_152128

def first_year_production : ℕ := 40
def second_year_production (first_year_production : ℕ) : ℕ := 2 * first_year_production + 8
def third_year_production (second_year_production : ℕ) : ℕ := second_year_production - (second_year_production / 4)
def total_production (first_year_production second_year_production third_year_production : ℕ) : ℕ :=
    first_year_production + second_year_production + third_year_production

-- Proof statement
theorem apple_tree_production : total_production 40 88 66 = 194 := by
  sorry

end NUMINAMATH_GPT_apple_tree_production_l1521_152128


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1521_152130

-- Define the equations of the lines
def line1 (a : ℝ) (x y : ℝ) : ℝ := 2 * x + a * y + 1
def line2 (a : ℝ) (x y : ℝ) : ℝ := (a - 1) * x + 3 * y - 2

-- Define the condition for parallel lines by comparing their slopes
def parallel_condition (a : ℝ) : Prop :=  (2 * 3 = a * (a - 1))

theorem sufficient_but_not_necessary (a : ℝ) : 3 ≤ a :=
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1521_152130


namespace NUMINAMATH_GPT_sum_of_first_11_terms_l1521_152166

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Condition: the sequence is an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Conditions given in the problem
axiom h1 : a 1 + a 5 + a 9 = 39
axiom h2 : a 3 + a 7 + a 11 = 27
axiom h3 : is_arithmetic_sequence a d

-- Proof statement
theorem sum_of_first_11_terms : (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11) = 121 := 
sorry

end NUMINAMATH_GPT_sum_of_first_11_terms_l1521_152166


namespace NUMINAMATH_GPT_find_x_squared_plus_y_squared_l1521_152160

theorem find_x_squared_plus_y_squared (x y : ℝ) (h₁ : x * y = -8) (h₂ : x^2 * y + x * y^2 + 3 * x + 3 * y = 100) : x^2 + y^2 = 416 :=
sorry

end NUMINAMATH_GPT_find_x_squared_plus_y_squared_l1521_152160


namespace NUMINAMATH_GPT_counterexample_to_proposition_l1521_152142

theorem counterexample_to_proposition : ∃ (a : ℝ), a^2 > 0 ∧ a ≤ 0 :=
  sorry

end NUMINAMATH_GPT_counterexample_to_proposition_l1521_152142


namespace NUMINAMATH_GPT_michelle_phone_bill_l1521_152152

def base_cost : ℝ := 20
def text_cost_per_message : ℝ := 0.05
def minute_cost_over_20h : ℝ := 0.20
def messages_sent : ℝ := 150
def hours_talked : ℝ := 22
def allowed_hours : ℝ := 20

theorem michelle_phone_bill :
  base_cost + (messages_sent * text_cost_per_message) +
  ((hours_talked - allowed_hours) * 60 * minute_cost_over_20h) = 51.50 := by
  sorry

end NUMINAMATH_GPT_michelle_phone_bill_l1521_152152


namespace NUMINAMATH_GPT_initial_students_count_l1521_152137

theorem initial_students_count (n : ℕ) (T T' : ℚ)
    (h1 : T = n * 61.5)
    (h2 : T' = T - 24)
    (h3 : T' = (n - 1) * 64) :
  n = 16 :=
by
  sorry

end NUMINAMATH_GPT_initial_students_count_l1521_152137


namespace NUMINAMATH_GPT_prime_square_mod_12_l1521_152143

theorem prime_square_mod_12 (p : ℕ) (h_prime : Nat.Prime p) (h_ne2 : p ≠ 2) (h_ne3 : p ≠ 3) :
    (∃ n : ℤ, p = 6 * n + 1 ∨ p = 6 * n + 5) → (p^2 % 12 = 1) := by
  sorry

end NUMINAMATH_GPT_prime_square_mod_12_l1521_152143


namespace NUMINAMATH_GPT_q_value_l1521_152140

-- Define the problem conditions
def prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_multiple_of (a b : ℕ) : Prop := ∃ k, a = k * b

-- Statement of the problem
theorem q_value (p q : ℕ) (hp : prime p) (hq : prime q) (h1 : q = 13 * p + 2) (h2 : is_multiple_of (q - 1) 3) : q = 67 :=
sorry

end NUMINAMATH_GPT_q_value_l1521_152140


namespace NUMINAMATH_GPT_largest_n_with_triangle_property_l1521_152108

/-- Triangle property: For any subset {a, b, c} with a ≤ b ≤ c, a + b > c -/
def triangle_property (s : Finset ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ s → b ∈ s → c ∈ s → a ≤ b → b ≤ c → a + b > c

/-- Definition of the set {3, 4, ..., n} -/
def consecutive_set (n : ℕ) : Finset ℕ :=
  Finset.range (n + 1) \ Finset.range 3

/-- The problem statement: The largest possible value of n where all eleven-element
 subsets of {3, 4, ..., n} have the triangle property -/
theorem largest_n_with_triangle_property : ∃ n, (∀ s ⊆ consecutive_set n, s.card = 11 → triangle_property s) ∧ n = 321 := sorry

end NUMINAMATH_GPT_largest_n_with_triangle_property_l1521_152108


namespace NUMINAMATH_GPT_area_excluding_holes_l1521_152113

theorem area_excluding_holes (x : ℝ) :
  let A_large : ℝ := (x + 8) * (x + 6)
  let A_hole : ℝ := (2 * x - 4) * (x - 3)
  A_large - 2 * A_hole = -3 * x^2 + 34 * x + 24 := by
  sorry

end NUMINAMATH_GPT_area_excluding_holes_l1521_152113


namespace NUMINAMATH_GPT_compute_v_l1521_152162

variable (a b c : ℝ)

theorem compute_v (H1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -8)
                  (H2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 12)
                  (H3 : a * b * c = 1) :
  (b / (a + b) + c / (b + c) + a / (c + a)) = -8.5 :=
sorry

end NUMINAMATH_GPT_compute_v_l1521_152162


namespace NUMINAMATH_GPT_last_8_digits_of_product_l1521_152149

theorem last_8_digits_of_product :
  let p := 11 * 101 * 1001 * 10001 * 1000001 * 111
  (p % 100000000) = 87654321 :=
by
  let p := 11 * 101 * 1001 * 10001 * 1000001 * 111
  have : p % 100000000 = 87654321 := sorry
  exact this

end NUMINAMATH_GPT_last_8_digits_of_product_l1521_152149


namespace NUMINAMATH_GPT_pure_imaginary_a_zero_l1521_152123

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem pure_imaginary_a_zero (a : ℝ) (h : is_pure_imaginary (i / (1 + a * i))) : a = 0 :=
sorry

end NUMINAMATH_GPT_pure_imaginary_a_zero_l1521_152123


namespace NUMINAMATH_GPT_incorrect_statement_l1521_152148

def consecutive_interior_angles_are_supplementary (l1 l2 : ℝ) : Prop :=
  ∀ (θ₁ θ₂ : ℝ), θ₁ + θ₂ = 180 → l1 = l2

def alternate_interior_angles_are_equal (l1 l2 : ℝ) : Prop :=
  ∀ (θ₁ θ₂ : ℝ), θ₁ = θ₂ → l1 = l2

def corresponding_angles_are_equal (l1 l2 : ℝ) : Prop :=
  ∀ (θ₁ θ₂ : ℝ), θ₁ = θ₂ → l1 = l2

def complementary_angles (θ₁ θ₂ : ℝ) : Prop :=
  θ₁ + θ₂ = 90

def supplementary_angles (θ₁ θ₂ : ℝ) : Prop :=
  θ₁ + θ₂ = 180

theorem incorrect_statement :
  ¬ (∀ (θ₁ θ₂ : ℝ), θ₁ = θ₂ → l1 = l2) →
    consecutive_interior_angles_are_supplementary l1 l2 →
    alternate_interior_angles_are_equal l1 l2 →
    corresponding_angles_are_equal l1 l2 →
    (∀ (θ₁ θ₂ : ℝ), supplementary_angles θ₁ θ₂) →
    (∀ (θ₁ θ₂ : ℝ), complementary_angles θ₁ θ₂) :=
sorry

end NUMINAMATH_GPT_incorrect_statement_l1521_152148


namespace NUMINAMATH_GPT_bridge_length_is_correct_l1521_152163

noncomputable def speed_km_per_hour_to_m_per_s (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600)

noncomputable def total_distance_covered (speed_m_per_s time_s : ℝ) : ℝ :=
  speed_m_per_s * time_s

def bridge_length (total_distance train_length : ℝ) : ℝ :=
  total_distance - train_length

theorem bridge_length_is_correct : 
  let train_length := 110 
  let speed_kmph := 72
  let time_s := 12.099
  let speed_m_per_s := speed_km_per_hour_to_m_per_s speed_kmph
  let total_distance := total_distance_covered speed_m_per_s time_s
  bridge_length total_distance train_length = 131.98 := 
by
  sorry

end NUMINAMATH_GPT_bridge_length_is_correct_l1521_152163


namespace NUMINAMATH_GPT_trapezoid_area_l1521_152134

def trapezoid_diagonals_and_height (AC BD h : ℕ) :=
  (AC = 17) ∧ (BD = 113) ∧ (h = 15)

theorem trapezoid_area (AC BD h : ℕ) (area1 area2 : ℕ) 
  (H : trapezoid_diagonals_and_height AC BD h) :
  (area1 = 900 ∨ area2 = 780) :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_area_l1521_152134


namespace NUMINAMATH_GPT_cost_price_per_metre_l1521_152146

theorem cost_price_per_metre (total_selling_price : ℕ) (total_metres : ℕ) (loss_per_metre : ℕ)
  (h1 : total_selling_price = 9000)
  (h2 : total_metres = 300)
  (h3 : loss_per_metre = 6) :
  (total_selling_price + (loss_per_metre * total_metres)) / total_metres = 36 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_per_metre_l1521_152146


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1521_152110

theorem solution_set_of_inequality (x : ℝ) : 
  (|x - 1| + |x - 2| ≥ 5) ↔ (x ≤ -1 ∨ x ≥ 4) :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1521_152110


namespace NUMINAMATH_GPT_problem_l1521_152141

-- Define the matrix
def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![2, 5, 0], ![0, 2, 3], ![3, 0, 2]]

-- Define the condition that there exists a nonzero vector v such that A * v = k * v
def exists_eigenvector (k : ℝ) : Prop :=
  ∃ (v : Fin 3 → ℝ), v ≠ 0 ∧ A.mulVec v = k • v

theorem problem : ∀ (k : ℝ), exists_eigenvector k ↔ (k = 2 + (45)^(1/3)) :=
sorry

end NUMINAMATH_GPT_problem_l1521_152141


namespace NUMINAMATH_GPT_jace_gave_to_neighbor_l1521_152144

theorem jace_gave_to_neighbor
  (earnings : ℕ) (debt : ℕ) (remaining : ℕ) (cents_per_dollar : ℕ) :
  earnings = 1000 →
  debt = 358 →
  remaining = 642 →
  cents_per_dollar = 100 →
  earnings - debt - remaining = 0
:= by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_jace_gave_to_neighbor_l1521_152144


namespace NUMINAMATH_GPT_find_first_term_l1521_152135

variable {a : ℕ → ℕ}

-- Given conditions
axiom arithmetic_seq : ∀ n : ℕ, a (n + 1) + a n = 4 * n

-- Question to prove
theorem find_first_term : a 0 = 1 :=
sorry

end NUMINAMATH_GPT_find_first_term_l1521_152135


namespace NUMINAMATH_GPT_geom_sequence_sum_l1521_152103

theorem geom_sequence_sum (S : ℕ → ℤ) (a : ℕ → ℤ) (r : ℤ) 
    (h1 : ∀ n : ℕ, n ≥ 1 → S n = 3^n + r) 
    (h2 : ∀ n : ℕ, n ≥ 2 → a n = S n - S (n - 1)) 
    (h3 : a 1 = S 1) :
  r = -1 := 
sorry

end NUMINAMATH_GPT_geom_sequence_sum_l1521_152103


namespace NUMINAMATH_GPT_exterior_angle_DEG_l1521_152138

-- Define the degree measures of angles in a square and a pentagon.
def square_interior_angle := 90
def pentagon_interior_angle := 108

-- Define the sum of the adjacent interior angles at D
def adjacent_interior_sum := square_interior_angle + pentagon_interior_angle

-- Statement to prove the exterior angle DEG
theorem exterior_angle_DEG :
  360 - adjacent_interior_sum = 162 := by
  sorry

end NUMINAMATH_GPT_exterior_angle_DEG_l1521_152138


namespace NUMINAMATH_GPT_geese_initial_formation_l1521_152171

theorem geese_initial_formation (G : ℕ) 
  (h1 : G / 2 + 4 = 12) : G = 16 := 
sorry

end NUMINAMATH_GPT_geese_initial_formation_l1521_152171


namespace NUMINAMATH_GPT_sahil_selling_price_l1521_152176

-- Define the conditions
def purchased_price := 9000
def repair_cost := 5000
def transportation_charges := 1000
def profit_percentage := 50 / 100

-- Calculate the total cost
def total_cost := purchased_price + repair_cost + transportation_charges

-- Calculate the selling price
def selling_price := total_cost + (profit_percentage * total_cost)

-- The theorem to prove the selling price
theorem sahil_selling_price : selling_price = 22500 :=
by
  -- This is where the proof would go, but we skip it with sorry.
  sorry

end NUMINAMATH_GPT_sahil_selling_price_l1521_152176


namespace NUMINAMATH_GPT_Trent_traveled_distance_l1521_152118

variable (blocks_length : ℕ := 50)
variables (walking_blocks : ℕ := 4) (bus_blocks : ℕ := 7) (bicycle_blocks : ℕ := 5)
variables (walking_round_trip : ℕ := 2 * walking_blocks * blocks_length)
variables (bus_round_trip : ℕ := 2 * bus_blocks * blocks_length)
variables (bicycle_round_trip : ℕ := 2 * bicycle_blocks * blocks_length)

def total_distance_traveleed : ℕ :=
  walking_round_trip + bus_round_trip + bicycle_round_trip

theorem Trent_traveled_distance :
  total_distance_traveleed = 1600 := by
    sorry

end NUMINAMATH_GPT_Trent_traveled_distance_l1521_152118


namespace NUMINAMATH_GPT_mixed_solution_concentration_l1521_152132

-- Defining the conditions as given in the question
def weight1 : ℕ := 200
def concentration1 : ℕ := 25
def saltInFirstSolution : ℕ := (concentration1 * weight1) / 100

def weight2 : ℕ := 300
def saltInSecondSolution : ℕ := 60

def totalSalt : ℕ := saltInFirstSolution + saltInSecondSolution
def totalWeight : ℕ := weight1 + weight2

-- Statement of the proof
theorem mixed_solution_concentration :
  ((totalSalt : ℚ) / (totalWeight : ℚ)) * 100 = 22 :=
by
  sorry

end NUMINAMATH_GPT_mixed_solution_concentration_l1521_152132


namespace NUMINAMATH_GPT_probability_P_is_1_over_3_l1521_152159

-- Definitions and conditions
def A := 0
def B := 3
def C := 1
def D := 2
def length_AB := B - A
def length_CD := D - C

-- Problem statement to prove
theorem probability_P_is_1_over_3 : (length_CD / length_AB) = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_probability_P_is_1_over_3_l1521_152159


namespace NUMINAMATH_GPT_items_left_in_store_l1521_152175

theorem items_left_in_store: (4458 - 1561) + 575 = 3472 :=
by 
  sorry

end NUMINAMATH_GPT_items_left_in_store_l1521_152175


namespace NUMINAMATH_GPT_find_m_range_l1521_152129

noncomputable def proposition_p (x : ℝ) : Prop := (-2 : ℝ) ≤ x ∧ x ≤ 10
noncomputable def proposition_q (x : ℝ) (m : ℝ) : Prop := (1 - m ≤ x ∧ x ≤ 1 + m)

theorem find_m_range (m : ℝ) (h : m > 0) : (¬ ∃ x : ℝ, proposition_p x) → (¬ ∃ x : ℝ, proposition_q x m) → (¬ (¬ (¬ ∃ x : ℝ, proposition_q x m)) → ¬ (¬ ∃ x : ℝ, proposition_p x)) → m ≥ 9 := 
sorry

end NUMINAMATH_GPT_find_m_range_l1521_152129


namespace NUMINAMATH_GPT_molly_ate_11_suckers_l1521_152188

/-- 
Sienna gave Bailey half of her suckers.
Jen ate 11 suckers and gave the rest to Molly.
Molly ate some suckers and gave the rest to Harmony.
Harmony kept 3 suckers and passed the remainder to Taylor.
Taylor ate one and gave the last 5 suckers to Callie.
How many suckers did Molly eat?
-/
theorem molly_ate_11_suckers
  (sienna_bailey_suckers : ℕ)
  (jen_ate : ℕ)
  (jens_remainder_to_molly : ℕ)
  (molly_remainder_to_harmony : ℕ) 
  (harmony_kept : ℕ) 
  (harmony_remainder_to_taylor : ℕ)
  (taylor_ate : ℕ)
  (taylor_remainder_to_callie : ℕ)
  (jen_condition : jen_ate = 11)
  (harmony_condition : harmony_kept = 3)
  (taylor_condition : taylor_ate = 1)
  (taylor_final_suckers : taylor_remainder_to_callie = 5) :
  molly_ate = 11 :=
by sorry

end NUMINAMATH_GPT_molly_ate_11_suckers_l1521_152188


namespace NUMINAMATH_GPT_least_n_divisible_by_some_not_all_l1521_152165

theorem least_n_divisible_by_some_not_all (n : ℕ) (h : 1 ≤ n):
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 10 ∧ k ∣ (n^2 - n)) ∧ ¬ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ (n^2 - n)) ↔ n = 3 :=
by
  sorry

end NUMINAMATH_GPT_least_n_divisible_by_some_not_all_l1521_152165


namespace NUMINAMATH_GPT_sum_of_diagonal_elements_l1521_152197

/-- Odd numbers from 1 to 49 arranged in a 5x5 grid. -/
def table : ℕ → ℕ → ℕ
| 0, 0 => 1
| 0, 1 => 3
| 0, 2 => 5
| 0, 3 => 7
| 0, 4 => 9
| 1, 0 => 11
| 1, 1 => 13
| 1, 2 => 15
| 1, 3 => 17
| 1, 4 => 19
| 2, 0 => 21
| 2, 1 => 23
| 2, 2 => 25
| 2, 3 => 27
| 2, 4 => 29
| 3, 0 => 31
| 3, 1 => 33
| 3, 2 => 35
| 3, 3 => 37
| 3, 4 => 39
| 4, 0 => 41
| 4, 1 => 43
| 4, 2 => 45
| 4, 3 => 47
| 4, 4 => 49
| _, _ => 0

/-- Proof that the sum of five numbers chosen from the table such that no two of them are in the same row or column equals 125. -/
theorem sum_of_diagonal_elements : 
  (table 0 0 + table 1 1 + table 2 2 + table 3 3 + table 4 4) = 125 := by
  sorry

end NUMINAMATH_GPT_sum_of_diagonal_elements_l1521_152197


namespace NUMINAMATH_GPT_constant_S13_l1521_152111

theorem constant_S13 (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h_seq : ∀ n, a n = a 1 + (n - 1) * d) 
(h_sum : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2)
(h_constant : ∀ a1 d, (a 2 + a 8 + a 11 = 3 * a1 + 18 * d)) : (S 13 = 91 * d) :=
by
  sorry

end NUMINAMATH_GPT_constant_S13_l1521_152111


namespace NUMINAMATH_GPT_bottle_caps_remaining_l1521_152136

-- Define the problem using the conditions and the desired proof.
theorem bottle_caps_remaining (original_count removed_count remaining_count : ℕ) 
    (h_original : original_count = 87) 
    (h_removed : removed_count = 47)
    (h_remaining : remaining_count = original_count - removed_count) :
    remaining_count = 40 :=
by 
  rw [h_original, h_removed] at h_remaining 
  exact h_remaining

end NUMINAMATH_GPT_bottle_caps_remaining_l1521_152136


namespace NUMINAMATH_GPT_find_value_of_expression_l1521_152105

theorem find_value_of_expression (a : ℝ) (h : a^2 + 3 * a - 1 = 0) : 2 * a^2 + 6 * a + 2021 = 2023 := 
by
  sorry

end NUMINAMATH_GPT_find_value_of_expression_l1521_152105


namespace NUMINAMATH_GPT_al_initial_amount_l1521_152187

theorem al_initial_amount
  (a b c : ℕ)
  (h₁ : a + b + c = 2000)
  (h₂ : 3 * a + 2 * b + 2 * c = 3500) :
  a = 500 :=
sorry

end NUMINAMATH_GPT_al_initial_amount_l1521_152187


namespace NUMINAMATH_GPT_down_payment_calculation_l1521_152196

noncomputable def tablet_price : ℝ := 450
noncomputable def installment_1 : ℝ := 4 * 40
noncomputable def installment_2 : ℝ := 4 * 35
noncomputable def installment_3 : ℝ := 4 * 30
noncomputable def total_savings : ℝ := 70
noncomputable def total_installments := tablet_price + total_savings
noncomputable def installment_payments := installment_1 + installment_2 + installment_3
noncomputable def down_payment := total_installments - installment_payments

theorem down_payment_calculation : down_payment = 100 := by
  unfold down_payment
  unfold total_installments
  unfold installment_payments
  unfold tablet_price
  unfold total_savings
  unfold installment_1
  unfold installment_2
  unfold installment_3
  sorry

end NUMINAMATH_GPT_down_payment_calculation_l1521_152196


namespace NUMINAMATH_GPT_future_value_option_B_correct_l1521_152122

noncomputable def future_value_option_B (p q : ℝ) : ℝ :=
  150 * (1 + p / 100 / 2) ^ 6 * (1 + q / 100 / 4) ^ 12

theorem future_value_option_B_correct (p q A₂ : ℝ) :
  A₂ = 150 * (1 + p / 100 / 2) ^ 6 * (1 + q / 100 / 4) ^ 12 →
  ∃ A₂, A₂ = future_value_option_B p q :=
by
  intro h
  use A₂
  exact h

end NUMINAMATH_GPT_future_value_option_B_correct_l1521_152122


namespace NUMINAMATH_GPT_train_speed_l1521_152106

-- Define the conditions given in the problem
def train_length : ℝ := 160
def time_to_cross_man : ℝ := 4

-- Define the statement to be proved
theorem train_speed (H1 : train_length = 160) (H2 : time_to_cross_man = 4) : train_length / time_to_cross_man = 40 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l1521_152106


namespace NUMINAMATH_GPT_correct_operations_l1521_152114

theorem correct_operations : 
  (∀ x y : ℝ, x^2 + x^4 ≠ x^6) ∧
  (∀ x y : ℝ, 2*x + 4*y ≠ 6*x*y) ∧
  (∀ x : ℝ, x^6 / x^3 = x^3) ∧
  (∀ x : ℝ, (x^3)^2 = x^6) :=
by 
  sorry

end NUMINAMATH_GPT_correct_operations_l1521_152114


namespace NUMINAMATH_GPT_find_n_l1521_152199

/-- Given a natural number n such that LCM(n, 12) = 48 and GCF(n, 12) = 8, prove that n = 32. -/
theorem find_n (n : ℕ) (h1 : Nat.lcm n 12 = 48) (h2 : Nat.gcd n 12 = 8) : n = 32 :=
sorry

end NUMINAMATH_GPT_find_n_l1521_152199


namespace NUMINAMATH_GPT_find_a_l1521_152116

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^x / (1 + a * 2^x)

theorem find_a (a : ℝ) (f : ℝ → ℝ) (h_f_def : ∀ x, f x = 2^x / (1 + a * 2^x))
  (h_symm : ∀ x, f x + f (-x) = 1) : a = 1 :=
sorry

end NUMINAMATH_GPT_find_a_l1521_152116


namespace NUMINAMATH_GPT_range_of_a_l1521_152193

def tensor (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → tensor (x - a) (x + a) < 2) → -1 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1521_152193


namespace NUMINAMATH_GPT_statement_a_correct_statement_b_correct_l1521_152154

open Real

theorem statement_a_correct (a b c : ℝ) (ha : a > b) (hc : c < 0) : a + c > b + c := by
  sorry

theorem statement_b_correct (a b : ℝ) (ha : a > b) (hb : b > 0) : (a + b) / 2 > Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_GPT_statement_a_correct_statement_b_correct_l1521_152154


namespace NUMINAMATH_GPT_maximum_b_value_l1521_152125

noncomputable def f (x : ℝ) := Real.exp x - x - 1
def g (x : ℝ) := -x^2 + 4 * x - 3

theorem maximum_b_value (a b : ℝ) (h1 : 0 ≤ a) (h2 : f a = g b) : b ≤ 3 := by
  sorry

end NUMINAMATH_GPT_maximum_b_value_l1521_152125


namespace NUMINAMATH_GPT_find_r_cubed_and_reciprocal_cubed_l1521_152195

variable (r : ℝ)
variable (h : (r + 1 / r) ^ 2 = 5)

theorem find_r_cubed_and_reciprocal_cubed (r : ℝ) (h : (r + 1 / r) ^ 2 = 5) : r ^ 3 + 1 / r ^ 3 = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_GPT_find_r_cubed_and_reciprocal_cubed_l1521_152195


namespace NUMINAMATH_GPT_det_2x2_matrix_l1521_152179

open Matrix

theorem det_2x2_matrix : 
  det ![![7, -2], ![-3, 5]] = 29 := by
  sorry

end NUMINAMATH_GPT_det_2x2_matrix_l1521_152179


namespace NUMINAMATH_GPT_arccos_range_l1521_152167

theorem arccos_range (a : ℝ) (x : ℝ) (h1 : x = Real.sin a) (h2 : a ∈ Set.Icc (-Real.pi / 4) (3 * Real.pi / 4)) :
  Set.Icc 0 (3 * Real.pi / 4) = Set.image Real.arccos (Set.Icc (-Real.sqrt 2 / 2) 1) :=
by
  sorry

end NUMINAMATH_GPT_arccos_range_l1521_152167


namespace NUMINAMATH_GPT_parabola_symmetric_points_l1521_152190

-- Define the parabola and the symmetry condition
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1

def symmetric_points (P Q : ℝ × ℝ) : Prop :=
  P.1 + P.2 = 0 ∧ Q.1 + Q.2 = 0 ∧ P.1 = -Q.1 ∧ P.2 = -Q.2

-- Problem definition: Prove that if there exist symmetric points on the parabola, then a > 3/4
theorem parabola_symmetric_points (a : ℝ) :
  (∃ P Q : ℝ × ℝ, symmetric_points P Q ∧ parabola a P.1 = P.2 ∧ parabola a Q.1 = Q.2) → a > 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_parabola_symmetric_points_l1521_152190


namespace NUMINAMATH_GPT_subtracted_amount_l1521_152133

theorem subtracted_amount (N A : ℝ) (h1 : 0.30 * N - A = 20) (h2 : N = 300) : A = 70 :=
by
  sorry

end NUMINAMATH_GPT_subtracted_amount_l1521_152133


namespace NUMINAMATH_GPT_abs_neg_three_l1521_152198

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end NUMINAMATH_GPT_abs_neg_three_l1521_152198


namespace NUMINAMATH_GPT_valid_shirt_tie_combinations_l1521_152115

theorem valid_shirt_tie_combinations
  (num_shirts : ℕ)
  (num_ties : ℕ)
  (restricted_shirts : ℕ)
  (restricted_ties : ℕ)
  (h_shirts : num_shirts = 8)
  (h_ties : num_ties = 7)
  (h_restricted_shirts : restricted_shirts = 3)
  (h_restricted_ties : restricted_ties = 2) :
  num_shirts * num_ties - restricted_shirts * restricted_ties = 50 := by
  sorry

end NUMINAMATH_GPT_valid_shirt_tie_combinations_l1521_152115


namespace NUMINAMATH_GPT_set_different_l1521_152120

-- Definitions of the sets ①, ②, ③, and ④
def set1 : Set ℤ := {x | x = 1}
def set2 : Set ℤ := {y | (y - 1)^2 = 0}
def set3 : Set ℤ := {x | x = 1}
def set4 : Set ℤ := {1}

-- Lean statement to prove that set3 is different from the others
theorem set_different : set3 ≠ set1 ∧ set3 ≠ set2 ∧ set3 ≠ set4 :=
by
  -- Skipping the proof with sorry
  sorry

end NUMINAMATH_GPT_set_different_l1521_152120


namespace NUMINAMATH_GPT_solve_for_a_l1521_152183

theorem solve_for_a (a : ℝ) 
  (h : (2 * a + 16 + (3 * a - 8)) / 2 = 89) : 
  a = 34 := 
sorry

end NUMINAMATH_GPT_solve_for_a_l1521_152183


namespace NUMINAMATH_GPT_new_perimeter_of_rectangle_l1521_152181

theorem new_perimeter_of_rectangle (w : ℝ) (A : ℝ) (new_area_factor : ℝ) (L : ℝ) (L' : ℝ) (P' : ℝ) 
  (h_w : w = 10) (h_A : A = 150) (h_new_area_factor: new_area_factor = 4 / 3)
  (h_orig_length : L = A / w) (h_new_area: A' = new_area_factor * A) (h_A' : A' = 200)
  (h_new_length : L' = A' / w) (h_perimeter : P' = 2 * (L' + w)) 
  : P' = 60 :=
sorry

end NUMINAMATH_GPT_new_perimeter_of_rectangle_l1521_152181


namespace NUMINAMATH_GPT_sum_of_all_ks_l1521_152102

theorem sum_of_all_ks (k : ℤ) :
  (∃ r s : ℤ, r ≠ s ∧ (3 * x^2 - k * x + 12 = 0) ∧ (r + s = k / 3) ∧ (r * s = 4)) → (k = 15 ∨ k = -15) → k + k = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_all_ks_l1521_152102


namespace NUMINAMATH_GPT_frac_ab_eq_five_thirds_l1521_152174

theorem frac_ab_eq_five_thirds (a b : ℝ) (hb : b ≠ 0) (h : (a - b) / b = 2 / 3) : a / b = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_frac_ab_eq_five_thirds_l1521_152174


namespace NUMINAMATH_GPT_part_a_1_part_a_2_l1521_152127

noncomputable def P (x k : ℝ) := x^3 - k*x + 2

theorem part_a_1 (k : ℝ) (h : k = 5) : P 2 k = 0 :=
sorry

theorem part_a_2 {x : ℝ} : P x 5 = (x - 2) * (x^2 + 2*x - 1) :=
sorry

end NUMINAMATH_GPT_part_a_1_part_a_2_l1521_152127


namespace NUMINAMATH_GPT_paint_cost_l1521_152109

theorem paint_cost (l : ℝ) (b : ℝ) (rate : ℝ) (area : ℝ) (cost : ℝ) 
  (h1 : l = 3 * b) 
  (h2 : l = 18.9999683334125) 
  (h3 : rate = 3.00001) 
  (h4 : area = l * b) 
  (h5 : cost = area * rate) : 
  cost = 361.00 :=
by
  sorry

end NUMINAMATH_GPT_paint_cost_l1521_152109


namespace NUMINAMATH_GPT_find_a_and_b_l1521_152124

variable {x : ℝ}

/-- The problem statement: Given the function y = b + a * sin x (with a < 0), and the maximum value is -1, and the minimum value is -5,
    find the values of a and b. --/
theorem find_a_and_b (a b : ℝ) (h : a < 0) 
  (h1 : ∀ x, b + a * Real.sin x ≤ -1)
  (h2 : ∀ x, b + a * Real.sin x ≥ -5) : 
  a = -2 ∧ b = -3 := sorry

end NUMINAMATH_GPT_find_a_and_b_l1521_152124


namespace NUMINAMATH_GPT_smallest_number_of_groups_l1521_152164

theorem smallest_number_of_groups
  (participants : ℕ)
  (max_group_size : ℕ)
  (h1 : participants = 36)
  (h2 : max_group_size = 12) :
  participants / max_group_size = 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_groups_l1521_152164


namespace NUMINAMATH_GPT_side_length_of_S2_l1521_152153

theorem side_length_of_S2 (r s : ℝ) 
  (h1 : 2 * r + s = 2025) 
  (h2 : 2 * r + 3 * s = 3320) :
  s = 647.5 :=
by {
  -- proof omitted
  sorry
}

end NUMINAMATH_GPT_side_length_of_S2_l1521_152153


namespace NUMINAMATH_GPT_sale_in_fifth_month_l1521_152101

theorem sale_in_fifth_month (Sale1 Sale2 Sale3 Sale4 Sale6 AvgSale : ℤ) 
(h1 : Sale1 = 6435) (h2 : Sale2 = 6927) (h3 : Sale3 = 6855) (h4 : Sale4 = 7230) 
(h5 : Sale6 = 4991) (h6 : AvgSale = 6500) : (39000 - (Sale1 + Sale2 + Sale3 + Sale4 + Sale6)) = 6562 :=
by
  sorry

end NUMINAMATH_GPT_sale_in_fifth_month_l1521_152101


namespace NUMINAMATH_GPT_loaf_bread_cost_correct_l1521_152177

-- Given conditions
def total : ℕ := 32
def candy_bar : ℕ := 2
def final_remaining : ℕ := 18

-- Intermediate calculations as definitions
def remaining_after_candy_bar : ℕ := total - candy_bar
def turkey_cost : ℕ := remaining_after_candy_bar / 3
def remaining_after_turkey : ℕ := remaining_after_candy_bar - turkey_cost
def loaf_bread_cost : ℕ := remaining_after_turkey - final_remaining

-- Theorem stating the problem question and expected answer
theorem loaf_bread_cost_correct : loaf_bread_cost = 2 :=
sorry

end NUMINAMATH_GPT_loaf_bread_cost_correct_l1521_152177


namespace NUMINAMATH_GPT_product_of_possible_values_of_N_l1521_152172

theorem product_of_possible_values_of_N (N B D : ℤ) 
  (h1 : B = D - N) 
  (h2 : B + 10 - (D - 4) = 1 ∨ B + 10 - (D - 4) = -1) :
  N = 13 ∨ N = 15 → (13 * 15) = 195 :=
by sorry

end NUMINAMATH_GPT_product_of_possible_values_of_N_l1521_152172


namespace NUMINAMATH_GPT_Diane_age_when_conditions_met_l1521_152100

variable (Diane_current : ℕ) (Alex_current : ℕ) (Allison_current : ℕ)
variable (D : ℕ)

axiom Diane_current_age : Diane_current = 16
axiom Alex_Allison_sum : Alex_current + Allison_current = 47
axiom Diane_half_Alex : D = (Alex_current + (D - 16)) / 2
axiom Diane_twice_Allison : D = 2 * (Allison_current + (D - 16))

theorem Diane_age_when_conditions_met : D = 78 :=
by
  sorry

end NUMINAMATH_GPT_Diane_age_when_conditions_met_l1521_152100
