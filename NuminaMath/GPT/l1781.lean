import Mathlib

namespace principal_amount_is_approx_1200_l1781_178189

noncomputable def find_principal_amount : Real :=
  let R := 0.10
  let n := 2
  let T := 1
  let SI (P : Real) := P * R * T / 100
  let CI (P : Real) := P * ((1 + R / n) ^ (n * T)) - P
  let diff (P : Real) := CI P - SI P
  let target_diff := 2.999999999999936
  let P := target_diff / (0.1025 - 0.10)
  P

theorem principal_amount_is_approx_1200 : abs (find_principal_amount - 1200) < 0.0001 := 
by
  sorry

end principal_amount_is_approx_1200_l1781_178189


namespace probability_angle_AMB_acute_l1781_178196

theorem probability_angle_AMB_acute :
  let side_length := 4
  let square_area := side_length * side_length
  let semicircle_area := (1 / 2) * Real.pi * (side_length / 2) ^ 2
  let probability := 1 - semicircle_area / square_area
  probability = 1 - (Real.pi / 8) :=
sorry

end probability_angle_AMB_acute_l1781_178196


namespace correct_proportion_expression_l1781_178152

def is_fraction_correctly_expressed (numerator denominator : ℕ) (expression : String) : Prop :=
  -- Define the property of a correctly expressed fraction in English
  expression = "three-fifths"

theorem correct_proportion_expression : 
  is_fraction_correctly_expressed 3 5 "three-fifths" :=
by
  sorry

end correct_proportion_expression_l1781_178152


namespace tetrahedron_colorings_l1781_178178

-- Define the problem conditions
def tetrahedron_faces : ℕ := 4
def colors : List String := ["red", "white", "blue", "yellow"]

-- The theorem statement
theorem tetrahedron_colorings :
  ∃ n : ℕ, n = 35 ∧ ∀ (c : List String), c.length = tetrahedron_faces → c ⊆ colors →
  (true) := -- Placeholder (you can replace this condition with the appropriate condition)
by
  -- The proof is omitted with 'sorry' as instructed
  sorry

end tetrahedron_colorings_l1781_178178


namespace total_men_wages_l1781_178126

-- Define our variables and parameters
variable (M W B : ℝ)
variable (W_women : ℝ)

-- Conditions from the problem:
-- 1. 12M = WW (where WW is W_women)
-- 2. WW = 20B
-- 3. 12M + WW + 20B = 450
axiom eq_12M_WW : 12 * M = W_women
axiom eq_WW_20B : W_women = 20 * B
axiom eq_total_earnings : 12 * M + W_women + 20 * B = 450

-- Prove total wages of the men is Rs. 150
theorem total_men_wages : 12 * M = 150 := by
  sorry

end total_men_wages_l1781_178126


namespace compute_special_op_l1781_178172

-- Define the operation ※
def special_op (m n : ℚ) := (3 * m + n) * (3 * m - n) + n

-- Hypothesis for specific m and n
def m := (1 : ℚ) / 6
def n := (-1 : ℚ)

-- Proof goal
theorem compute_special_op : special_op m n = -7 / 4 := by
  sorry

end compute_special_op_l1781_178172


namespace quadratic_expression_factorization_l1781_178112

theorem quadratic_expression_factorization :
  ∃ c d : ℕ, (c > d) ∧ (x^2 - 18*x + 72 = (x - c) * (x - d)) ∧ (4*d - c = 12) := 
by
  sorry

end quadratic_expression_factorization_l1781_178112


namespace a_range_l1781_178141

variables {x a : ℝ}

def p (x : ℝ) : Prop := (4 * x - 3) ^ 2 ≤ 1
def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem a_range (h : ∀ x, ¬p x → ¬q x a ∧ (∃ x, q x a ∧ ¬p x)) :
  0 ≤ a ∧ a ≤ 1/2 :=
sorry

end a_range_l1781_178141


namespace y_is_multiple_of_16_y_is_multiple_of_8_y_is_multiple_of_4_y_is_multiple_of_2_l1781_178158

def y : ℕ := 112 + 160 + 272 + 432 + 1040 + 1264 + 4256

theorem y_is_multiple_of_16 : y % 16 = 0 :=
sorry

theorem y_is_multiple_of_8 : y % 8 = 0 :=
sorry

theorem y_is_multiple_of_4 : y % 4 = 0 :=
sorry

theorem y_is_multiple_of_2 : y % 2 = 0 :=
sorry

end y_is_multiple_of_16_y_is_multiple_of_8_y_is_multiple_of_4_y_is_multiple_of_2_l1781_178158


namespace relationship_ab_c_l1781_178170
open Real

noncomputable def a : ℝ := (1 / 3) ^ (log 3 / log (1 / 3))
noncomputable def b : ℝ := (1 / 3) ^ (log 4 / log (1 / 3))
noncomputable def c : ℝ := 3 ^ log 3

theorem relationship_ab_c : c > b ∧ b > a := by
  sorry

end relationship_ab_c_l1781_178170


namespace clock_tick_intervals_l1781_178146

theorem clock_tick_intervals (intervals_6: ℕ) (intervals_12: ℕ) (total_time_12: ℕ) (interval_time: ℕ):
  intervals_6 = 5 →
  intervals_12 = 11 →
  total_time_12 = 88 →
  interval_time = total_time_12 / intervals_12 →
  intervals_6 * interval_time = 40 :=
by
  intros h1 h2 h3 h4
  -- will continue proof here
  sorry

end clock_tick_intervals_l1781_178146


namespace nonneg_int_solutions_eq_l1781_178128

theorem nonneg_int_solutions_eq (a b : ℕ) : a^2 + b^2 = 841 * (a * b + 1) ↔ (a = 0 ∧ b = 29) ∨ (a = 29 ∧ b = 0) :=
by {
  sorry -- Proof omitted
}

end nonneg_int_solutions_eq_l1781_178128


namespace problem_3_at_7_hash_4_l1781_178121

def oper_at (a b : ℕ) : ℚ := (a * b) / (a + b)
def oper_hash (c d : ℚ) : ℚ := c + d

theorem problem_3_at_7_hash_4 :
  oper_hash (oper_at 3 7) 4 = 61 / 10 := by
  sorry

end problem_3_at_7_hash_4_l1781_178121


namespace find_q_of_polynomial_l1781_178191

noncomputable def Q (x : ℝ) (p q d : ℝ) : ℝ := x^3 + p * x^2 + q * x + d

theorem find_q_of_polynomial (p q d : ℝ) (mean_zeros twice_product sum_coeffs : ℝ)
  (h1 : mean_zeros = -p / 3)
  (h2 : twice_product = -2 * d)
  (h3 : sum_coeffs = 1 + p + q + d)
  (h4 : d = 4)
  (h5 : mean_zeros = twice_product)
  (h6 : sum_coeffs = twice_product) :
  q = -37 :=
sorry

end find_q_of_polynomial_l1781_178191


namespace find_k_l1781_178175

noncomputable def y (k x : ℝ) : ℝ := k / x

theorem find_k (k : ℝ) (h₁ : k ≠ 0) (h₂ : 1 ≤ 3) 
  (h₃ : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → x = 1 ∨ x = 3) 
  (h₄ : |y k 1 - y k 3| = 4) : k = 6 ∨ k = -6 :=
  sorry

end find_k_l1781_178175


namespace pure_ghee_added_l1781_178176

theorem pure_ghee_added
  (Q : ℕ) (hQ : Q = 30)
  (P : ℕ)
  (original_pure_ghee : ℕ := (Q / 2))
  (original_vanaspati : ℕ := (Q / 2))
  (new_total_ghee : ℕ := Q + P)
  (new_vanaspati_fraction : ℝ := 0.3) :
  original_vanaspati = (new_vanaspati_fraction * ↑new_total_ghee : ℝ) → P = 20 := by
  sorry

end pure_ghee_added_l1781_178176


namespace complex_div_eq_l1781_178154

def complex_z : ℂ := ⟨1, -2⟩
def imaginary_unit : ℂ := ⟨0, 1⟩

theorem complex_div_eq :
  (complex_z + 2) / (complex_z - 1) = 1 + (3 / 2 : ℂ) * imaginary_unit :=
by
  sorry

end complex_div_eq_l1781_178154


namespace sum_of_f_is_negative_l1781_178160

noncomputable def f (x : ℝ) : ℝ := x + x^3 + x^5

theorem sum_of_f_is_negative (x₁ x₂ x₃ : ℝ)
  (h1: x₁ + x₂ < 0)
  (h2: x₂ + x₃ < 0) 
  (h3: x₃ + x₁ < 0) :
  f x₁ + f x₂ + f x₃ < 0 := 
sorry

end sum_of_f_is_negative_l1781_178160


namespace age_difference_of_declans_sons_l1781_178108

theorem age_difference_of_declans_sons 
  (current_age_elder_son : ℕ) 
  (future_age_younger_son : ℕ) 
  (years_until_future : ℕ) 
  (current_age_elder_son_eq : current_age_elder_son = 40) 
  (future_age_younger_son_eq : future_age_younger_son = 60) 
  (years_until_future_eq : years_until_future = 30) :
  (current_age_elder_son - (future_age_younger_son - years_until_future)) = 10 := by
  sorry

end age_difference_of_declans_sons_l1781_178108


namespace total_buyers_l1781_178109

-- Definitions based on conditions
def C : ℕ := 50
def M : ℕ := 40
def B : ℕ := 19
def pN : ℝ := 0.29  -- Probability that a random buyer purchases neither

-- The theorem statement
theorem total_buyers :
  ∃ T : ℝ, (T = (C + M - B) + pN * T) ∧ T = 100 :=
by
  sorry

end total_buyers_l1781_178109


namespace exists_xyz_t_l1781_178120

theorem exists_xyz_t (x y z t : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : t > 0) (h5 : x + y + z + t = 15) : ∃ y, y = 12 :=
by
  sorry

end exists_xyz_t_l1781_178120


namespace sequence_terminates_final_value_l1781_178180

-- Define the function Lisa uses to update the number
def f (x : ℕ) : ℕ :=
  let a := x / 10
  let b := x % 10
  a + 4 * b

-- Prove that for any initial value x0, the sequence eventually becomes periodic and ends.
theorem sequence_terminates (x0 : ℕ) : ∃ N : ℕ, ∃ j : ℕ, N ≠ j ∧ (Nat.iterate f N x0) = (Nat.iterate f j x0) :=
  by sorry

-- Given the starting value, show the sequence stabilizes at 39
theorem final_value (x0 : ℕ) (h : x0 = 53^2022 - 1) : ∃ N : ℕ, Nat.iterate f N x0 = 39 :=
  by sorry

end sequence_terminates_final_value_l1781_178180


namespace union_is_real_l1781_178114

-- Definitions of sets A and B
def setA : Set ℝ := {x | x^2 - x - 2 ≥ 0}
def setB : Set ℝ := {x | x > -1}

-- Theorem to prove
theorem union_is_real :
  setA ∪ setB = Set.univ :=
by
  sorry

end union_is_real_l1781_178114


namespace birdhouse_flight_distance_l1781_178122

variable (car_distance : ℕ)
variable (lawn_chair_distance : ℕ)
variable (birdhouse_distance : ℕ)

def problem_condition1 := car_distance = 200
def problem_condition2 := lawn_chair_distance = 2 * car_distance
def problem_condition3 := birdhouse_distance = 3 * lawn_chair_distance

theorem birdhouse_flight_distance
  (h1 : car_distance = 200)
  (h2 : lawn_chair_distance = 2 * car_distance)
  (h3 : birdhouse_distance = 3 * lawn_chair_distance) :
  birdhouse_distance = 1200 := by
  sorry

end birdhouse_flight_distance_l1781_178122


namespace rectangle_width_decrease_l1781_178194

theorem rectangle_width_decrease (A L W : ℝ) (h1 : A = L * W) (h2 : 1.5 * L * W' = A) : 
  (W' = (2/3) * W) -> by exact (W - W') / W = 1 / 3 :=
by
  sorry

end rectangle_width_decrease_l1781_178194


namespace hyperbola_condition_l1781_178182

variables (a b : ℝ)
def e1 : (ℝ × ℝ) := (2, 1)
def e2 : (ℝ × ℝ) := (2, -1)

theorem hyperbola_condition (h1 : e1 = (2, 1)) (h2 : e2 = (2, -1)) (p : ℝ × ℝ)
  (h3 : p = (2 * a + 2 * b, a - b)) :
  4 * a * b = 1 :=
sorry

end hyperbola_condition_l1781_178182


namespace max_books_borrowed_l1781_178113

theorem max_books_borrowed (students : ℕ) (no_books : ℕ) (one_book : ℕ) (two_books : ℕ) (more_books : ℕ)
  (h_students : students = 30)
  (h_no_books : no_books = 5)
  (h_one_book : one_book = 12)
  (h_two_books : two_books = 8)
  (h_more_books : more_books = students - no_books - one_book - two_books)
  (avg_books : ℕ)
  (h_avg_books : avg_books = 2) :
  ∃ max_books : ℕ, max_books = 20 := 
by 
  sorry

end max_books_borrowed_l1781_178113


namespace evaluate_expression_l1781_178150

variable (x y z : ℝ)

theorem evaluate_expression (h : x / (30 - x) + y / (75 - y) + z / (50 - z) = 9) :
  6 / (30 - x) + 15 / (75 - y) + 10 / (50 - z) = 2.4 := 
sorry

end evaluate_expression_l1781_178150


namespace total_raised_is_420_l1781_178197

def pancake_cost : ℝ := 4.00
def bacon_cost : ℝ := 2.00
def stacks_sold : ℕ := 60
def slices_sold : ℕ := 90

theorem total_raised_is_420 : (pancake_cost * stacks_sold + bacon_cost * slices_sold) = 420.00 :=
by
  -- Proof goes here
  sorry

end total_raised_is_420_l1781_178197


namespace find_width_of_room_l1781_178186

section RoomWidth

variable (l C P A W : ℝ)
variable (h1 : l = 5.5)
variable (h2 : C = 16500)
variable (h3 : P = 750)
variable (h4 : A = C / P)
variable (h5 : A = l * W)

theorem find_width_of_room : W = 4 := by
  sorry

end RoomWidth

end find_width_of_room_l1781_178186


namespace unique_solution_n_l1781_178165

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem unique_solution_n (h : ∀ n : ℕ, (n > 0) → n^3 = 8 * (sum_digits n)^3 + 6 * (sum_digits n) * n + 1 → n = 17) : 
  n = 17 := 
by
  sorry

end unique_solution_n_l1781_178165


namespace eliminate_xy_l1781_178118

variable {R : Type*} [Ring R]

theorem eliminate_xy
  (x y a b c : R)
  (h1 : a = x + y)
  (h2 : b = x^3 + y^3)
  (h3 : c = x^5 + y^5) :
  5 * b * (a^3 + b) = a * (a^5 + 9 * c) :=
sorry

end eliminate_xy_l1781_178118


namespace problem_solution_l1781_178133

theorem problem_solution (m : ℤ) (x : ℤ) (h : 4 * x + 2 * m = 14) : x = 2 → m = 3 :=
by sorry

end problem_solution_l1781_178133


namespace inverse_sum_l1781_178142

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else 3 * x - x^2

theorem inverse_sum :
  (∃ x₁, g x₁ = -2 ∧ x₁ ≠ 5) ∨ (∃ x₂, g x₂ = 0 ∧ x₂ = 3) ∨ (∃ x₃, g x₃ = 4 ∧ x₃ = -1) → 
  g⁻¹ (-2) + g⁻¹ (0) + g⁻¹ (4) = 6 :=
by
  sorry

end inverse_sum_l1781_178142


namespace equivalent_mod_l1781_178171

theorem equivalent_mod (h : 5^300 ≡ 1 [MOD 1250]) : 5^9000 ≡ 1 [MOD 1000] :=
by 
  sorry

end equivalent_mod_l1781_178171


namespace smallest_value_of_x_l1781_178136

theorem smallest_value_of_x :
  ∀ x : ℚ, ( ( (5 * x - 20) / (4 * x - 5) ) ^ 3
           + ( (5 * x - 20) / (4 * x - 5) ) ^ 2
           - ( (5 * x - 20) / (4 * x - 5) )
           - 15 = 0 ) → x = 10 / 3 :=
by
  sorry

end smallest_value_of_x_l1781_178136


namespace average_test_score_before_dropping_l1781_178135

theorem average_test_score_before_dropping (A B C : ℝ) :
  (A + B + C) / 3 = 40 → (A + B + C + 20) / 4 = 35 :=
by
  intros h
  sorry

end average_test_score_before_dropping_l1781_178135


namespace factorize_expression_l1781_178151

theorem factorize_expression (x : ℝ) : 2 * x^2 - 18 = 2 * (x + 3) * (x - 3) :=
by sorry

end factorize_expression_l1781_178151


namespace find_n_times_s_l1781_178140

noncomputable def g (x : ℝ) : ℝ :=
  if x = 1 then 2011
  else if x = 2 then (1 / 2 + 2010)
  else 0 /- For purposes of the problem -/

theorem find_n_times_s :
  (∀ x y : ℝ, x > 0 → y > 0 → g x * g y = g (x * y) + 2010 * (1 / x + 1 / y + 2010)) →
  ∃ n s : ℝ, n = 1 ∧ s = (4021 / 2) ∧ n * s = 4021 / 2 :=
by
  sorry

end find_n_times_s_l1781_178140


namespace box_height_at_least_2_sqrt_15_l1781_178169

def box_height (x : ℝ) : ℝ := 2 * x
def surface_area (x : ℝ) : ℝ := 10 * x ^ 2

theorem box_height_at_least_2_sqrt_15 (x : ℝ) (h : ℝ) :
  h = box_height x →
  surface_area x ≥ 150 →
  h ≥ 2 * Real.sqrt 15 :=
by
  intros h_eq sa_ge_150
  sorry

end box_height_at_least_2_sqrt_15_l1781_178169


namespace employed_females_part_time_percentage_l1781_178102

theorem employed_females_part_time_percentage (P : ℕ) (hP1 : 0 < P)
  (h1 : ∀ x : ℕ, x = P * 6 / 10) -- 60% of P are employed
  (h2 : ∀ e : ℕ, e = P * 6 / 10) -- e is the number of employed individuals
  (h3 : ∀ f : ℕ, f = e * 4 / 10) -- 40% of employed are females
  (h4 : ∀ pt : ℕ, pt = f * 6 / 10) -- 60% of employed females are part-time
  (h5 : ∀ m : ℕ, m = P * 48 / 100) -- 48% of P are employed males
  (h6 : e = f + m) -- Employed individuals are either males or females
  : f * 6 / f * 10 = 60 := sorry

end employed_females_part_time_percentage_l1781_178102


namespace simplify_expression_l1781_178162

theorem simplify_expression : (Real.sin (15 * Real.pi / 180) + Real.sin (45 * Real.pi / 180)) / (Real.cos (15 * Real.pi / 180) + Real.cos (45 * Real.pi / 180)) = Real.tan (30 * Real.pi / 180) :=
by
  sorry

end simplify_expression_l1781_178162


namespace area_of_black_region_l1781_178187

-- Definitions for the side lengths of the smaller and larger squares
def s₁ : ℕ := 4
def s₂ : ℕ := 8

-- The mathematical problem statement in Lean 4
theorem area_of_black_region : (s₂ * s₂) - (s₁ * s₁) = 48 := by
  sorry

end area_of_black_region_l1781_178187


namespace camp_problem_l1781_178134

variable (x : ℕ) -- number of girls
variable (y : ℕ) -- number of boys
variable (total_children : ℕ) -- total number of children
variable (girls_cannot_swim : ℕ) -- number of girls who cannot swim
variable (boys_cannot_swim : ℕ) -- number of boys who cannot swim
variable (children_can_swim : ℕ) -- total number of children who can swim
variable (children_cannot_swim : ℕ) -- total number of children who cannot swim
variable (o_six_girls : ℕ) -- one-sixth of the total number of girls
variable (o_eight_boys : ℕ) -- one-eighth of the total number of boys

theorem camp_problem 
    (hc1 : total_children = 50)
    (hc2 : girls_cannot_swim = x / 6)
    (hc3 : boys_cannot_swim = y / 8)
    (hc4 : children_can_swim = 43)
    (hc5 : children_cannot_swim = total_children - children_can_swim)
    (h_total : x + y = total_children)
    (h_swim : children_cannot_swim = girls_cannot_swim + boys_cannot_swim) :
    x = 18 :=
  by
    have hc6 : children_cannot_swim = 7 := by sorry -- from hc4 and hc5
    have h_eq : x / 6 + (50 - x) / 8 = 7 := by sorry -- from hc2, hc3, hc6
    -- solving for x
    sorry

end camp_problem_l1781_178134


namespace diet_soda_count_l1781_178123

theorem diet_soda_count (D : ℕ) (h1 : 81 = D + 21) : D = 60 := by
  sorry

end diet_soda_count_l1781_178123


namespace find_f1_l1781_178149

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def condition_on_function (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, x ≤ 0 → f x = 2^x - 3 * x + 2 * m

theorem find_f1 (f : ℝ → ℝ) (m : ℝ)
  (h_odd : is_odd_function f)
  (h_condition : condition_on_function f m) :
  f 1 = -(5 / 2) :=
by
  sorry

end find_f1_l1781_178149


namespace question1_question2_l1781_178107

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - a - Real.log x

theorem question1 (a : ℝ) (h : ∀ x > 0, f x a ≥ 0) : a ≤ 1 := sorry

theorem question2 (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) : 
  x1 * Real.log x1 - x1 * Real.log x2 > x1 - x2 := sorry

end question1_question2_l1781_178107


namespace greatest_perimeter_isosceles_triangle_l1781_178148

theorem greatest_perimeter_isosceles_triangle :
  let base := 12
  let height := 15
  let segments := 6
  let max_perimeter := 32.97
  -- Assuming division such that each of the 6 pieces is of equal area,
  -- the greatest perimeter among these pieces to the nearest hundredth is:
  (∀ (base height segments : ℝ), base = 12 ∧ height = 15 ∧ segments = 6 → 
   max_perimeter = 32.97) :=
by
  sorry

end greatest_perimeter_isosceles_triangle_l1781_178148


namespace taxi_ride_distance_l1781_178167

theorem taxi_ride_distance (initial_fare additional_fare total_fare : ℝ) 
  (initial_distance : ℝ) (additional_distance increment_distance : ℝ) :
  initial_fare = 1.0 →
  additional_fare = 0.45 →
  initial_distance = 1/5 →
  increment_distance = 1/5 →
  total_fare = 7.3 →
  additional_distance = (total_fare - initial_fare) / additional_fare →
  (initial_distance + additional_distance * increment_distance) = 3 := 
by sorry

end taxi_ride_distance_l1781_178167


namespace triangle_perimeter_l1781_178145

theorem triangle_perimeter (P₁ P₂ P₃ : ℝ) (hP₁ : P₁ = 12) (hP₂ : P₂ = 14) (hP₃ : P₃ = 16) : 
  P₁ + P₂ + P₃ = 42 := by
  sorry

end triangle_perimeter_l1781_178145


namespace num_multiples_6_not_12_lt_300_l1781_178110

theorem num_multiples_6_not_12_lt_300 : 
  ∃ n : ℕ, n = 25 ∧ ∀ k : ℕ, k < 300 ∧ k % 6 = 0 ∧ k % 12 ≠ 0 → ∃ m : ℕ, k = 6 * (2 * m - 1) ∧ 1 ≤ m ∧ m ≤ 25 := 
by
  sorry

end num_multiples_6_not_12_lt_300_l1781_178110


namespace arithmetic_expression_l1781_178159

theorem arithmetic_expression : (56^2 + 56^2) / 28^2 = 8 := by
  sorry

end arithmetic_expression_l1781_178159


namespace centroid_traces_ellipse_l1781_178157

noncomputable def fixed_base_triangle (A B : ℝ × ℝ) (d : ℝ) : Prop :=
(A.1 = 0 ∧ A.2 = 0) ∧ (B.1 = d ∧ B.2 = 0)

noncomputable def vertex_moving_on_semicircle (A B C : ℝ × ℝ) : Prop :=
(C.1 - (A.1 + B.1) / 2)^2 + C.2^2 = ((B.1 - A.1) / 2)^2 ∧ C.2 ≥ 0

noncomputable def is_centroid (A B C G : ℝ × ℝ) : Prop :=
G.1 = (A.1 + B.1 + C.1) / 3 ∧ G.2 = (A.2 + B.2 + C.2) / 3

theorem centroid_traces_ellipse
  (A B C G : ℝ × ℝ) (d : ℝ) 
  (h1 : fixed_base_triangle A B d) 
  (h2 : vertex_moving_on_semicircle A B C)
  (h3 : is_centroid A B C G) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (G.1^2 / a^2 + G.2^2 / b^2 = 1) := 
sorry

end centroid_traces_ellipse_l1781_178157


namespace sum_of_squares_ge_sum_of_products_l1781_178104

theorem sum_of_squares_ge_sum_of_products (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := by
  sorry

end sum_of_squares_ge_sum_of_products_l1781_178104


namespace negative_solution_range_l1781_178164

theorem negative_solution_range (m : ℝ) : (∃ x : ℝ, 2 * x + 4 = m - x ∧ x < 0) → m < 4 := by
  sorry

end negative_solution_range_l1781_178164


namespace problem_l1781_178168

noncomputable def f : ℝ → ℝ := sorry

theorem problem (x : ℝ) :
  (f (x + 2) + f x = 0) →
  (∀ x, f (-(x - 1)) = -f (x - 1)) →
  (
    (∀ e, ¬(e > 0 ∧ ∀ x, f (x + e) = f x)) ∧
    (∀ x, f (x + 1) = f (-x + 1)) ∧
    (¬(∀ x, f x = f (-x)))
  ) :=
by
  sorry

end problem_l1781_178168


namespace parallelogram_height_l1781_178129

theorem parallelogram_height (b A : ℝ) (h : ℝ) (h_base : b = 28) (h_area : A = 896) : h = A / b := by
  simp [h_base, h_area]
  norm_num
  sorry

end parallelogram_height_l1781_178129


namespace Enid_made_8_sweaters_l1781_178131

def scarves : ℕ := 10
def sweaters_Aaron : ℕ := 5
def wool_per_scarf : ℕ := 3
def wool_per_sweater : ℕ := 4
def total_wool_used : ℕ := 82
def Enid_sweaters : ℕ := 8

theorem Enid_made_8_sweaters
  (scarves : ℕ)
  (sweaters_Aaron : ℕ)
  (wool_per_scarf : ℕ)
  (wool_per_sweater : ℕ)
  (total_wool_used : ℕ)
  (Enid_sweaters : ℕ)
  : Enid_sweaters = 8 :=
by
  sorry

end Enid_made_8_sweaters_l1781_178131


namespace carlos_picks_24_integers_l1781_178143

def is_divisor (n m : ℕ) : Prop := m % n = 0

theorem carlos_picks_24_integers :
  ∃ (s : Finset ℕ), s.card = 24 ∧ ∀ n ∈ s, is_divisor n 4500 ∧ 1 ≤ n ∧ n ≤ 4500 ∧ n % 3 = 0 :=
by
  sorry

end carlos_picks_24_integers_l1781_178143


namespace initial_action_figures_l1781_178132

theorem initial_action_figures (x : ℕ) (h : x + 2 - 7 = 10) : x = 15 :=
by
  sorry

end initial_action_figures_l1781_178132


namespace boat_stream_speed_l1781_178137

theorem boat_stream_speed (v : ℝ) (h : (60 / (15 - v)) - (60 / (15 + v)) = 2) : v = 3.5 := 
by 
  sorry
 
end boat_stream_speed_l1781_178137


namespace value_of_m_l1781_178138

theorem value_of_m (m x : ℝ) (h : x = 3) (h_eq : 3 * m - 2 * x = 6) : m = 4 := by
  -- Given x = 3
  subst h
  -- Now we have to show m = 4
  sorry

end value_of_m_l1781_178138


namespace volume_of_truncated_triangular_pyramid_l1781_178177

variable {a b H α : ℝ} (h1 : H = Real.sqrt (a * b))

theorem volume_of_truncated_triangular_pyramid
  (h2 : H = Real.sqrt (a * b))
  (h3 : 0 < a)
  (h4 : 0 < b)
  (h5 : 0 < H)
  (h6 : 0 < α) :
  (volume : ℝ) = H^3 * Real.sqrt 3 / (4 * (Real.sin α)^2) := sorry

end volume_of_truncated_triangular_pyramid_l1781_178177


namespace solve_for_A_plus_B_l1781_178116

-- Definition of the problem conditions
def T := 7 -- The common total sum for rows and columns

-- Summing the rows and columns in the partially filled table
variable (A B : ℕ)
def table_condition :=
  4 + 1 + 2 = T ∧
  2 + A + B = T ∧
  4 + 2 + B = T ∧
  1 + A + B = T

-- Statement to prove
theorem solve_for_A_plus_B (A B : ℕ) (h : table_condition A B) : A + B = 5 :=
by
  sorry

end solve_for_A_plus_B_l1781_178116


namespace number_of_hard_drives_sold_l1781_178195

theorem number_of_hard_drives_sold 
    (H : ℕ)
    (price_per_graphics_card : ℕ := 600)
    (price_per_hard_drive : ℕ := 80)
    (price_per_cpu : ℕ := 200)
    (price_per_ram_pair : ℕ := 60)
    (graphics_cards_sold : ℕ := 10)
    (cpus_sold : ℕ := 8)
    (ram_pairs_sold : ℕ := 4)
    (total_earnings : ℕ := 8960)
    (earnings_from_graphics_cards : graphics_cards_sold * price_per_graphics_card = 6000)
    (earnings_from_cpus : cpus_sold * price_per_cpu = 1600)
    (earnings_from_ram : ram_pairs_sold * price_per_ram_pair = 240)
    (earnings_from_hard_drives : H * price_per_hard_drive = 80 * H) :
  graphics_cards_sold * price_per_graphics_card +
  cpus_sold * price_per_cpu +
  ram_pairs_sold * price_per_ram_pair +
  H * price_per_hard_drive = total_earnings → H = 14 :=
by
  intros h
  sorry

end number_of_hard_drives_sold_l1781_178195


namespace remainder_of_x_squared_div_20_l1781_178105

theorem remainder_of_x_squared_div_20
  (x : ℤ)
  (h1 : 5 * x ≡ 10 [ZMOD 20])
  (h2 : 4 * x ≡ 12 [ZMOD 20]) :
  (x * x) % 20 = 4 :=
sorry

end remainder_of_x_squared_div_20_l1781_178105


namespace squares_of_natural_numbers_l1781_178115

theorem squares_of_natural_numbers (x y z : ℕ) (h : x^2 + y^2 + z^2 = 2 * (x * y + y * z + z * x)) : ∃ a b c : ℕ, x = a^2 ∧ y = b^2 ∧ z = c^2 := 
by
  sorry

end squares_of_natural_numbers_l1781_178115


namespace max_value_of_f_l1781_178161

noncomputable def f (x a : ℝ) : ℝ := - (1/3) * x ^ 3 + (1/2) * x ^ 2 + 2 * a * x

theorem max_value_of_f (a : ℝ) (h0 : 0 < a) (h1 : a < 2)
  (h2 : ∀ x, 1 ≤ x → x ≤ 4 → f x a ≥ f 4 a)
  (h3 : f 4 a = -16 / 3) :
  f 2 a = 10 / 3 :=
sorry

end max_value_of_f_l1781_178161


namespace find_smallest_c_l1781_178130

/-- Let a₀, a₁, ... and b₀, b₁, ... be geometric sequences with common ratios rₐ and r_b, 
respectively, such that ∑ i=0 ∞ aᵢ = ∑ i=0 ∞ bᵢ = 1 and 
(∑ i=0 ∞ aᵢ²)(∑ i=0 ∞ bᵢ²) = ∑ i=0 ∞ aᵢbᵢ. Prove that a₀ < 4/3 -/
theorem find_smallest_c (r_a r_b : ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h1 : ∑' n, a n = 1)
  (h2 : ∑' n, b n = 1)
  (h3 : (∑' n, (a n)^2) * (∑' n, (b n)^2) = ∑' n, (a n) * (b n)) :
  a 0 < 4 / 3 := by
  sorry

end find_smallest_c_l1781_178130


namespace simplify_fraction_l1781_178103

theorem simplify_fraction (a b c d k : ℕ) (h₁ : a = 123) (h₂ : b = 9999) (h₃ : k = 41)
                           (h₄ : c = a / 3) (h₅ : d = b / 3)
                           (h₆ : c = k) (h₇ : d = 3333) :
  (a * k) / b = (k^2) / d :=
by
  sorry

end simplify_fraction_l1781_178103


namespace coefficient_of_x5_in_expansion_l1781_178153

-- Define the polynomial expansion of (x-1)(x+1)^8
def polynomial_expansion (x : ℚ) : ℚ :=
  (x - 1) * (x + 1) ^ 8

-- Define the binomial coefficient function
def binom_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Theorem: The coefficient of x^5 in the expansion of (x-1)(x+1)^8 is 14
theorem coefficient_of_x5_in_expansion :
  binom_coeff 8 4 - binom_coeff 8 5 = 14 :=
sorry

end coefficient_of_x5_in_expansion_l1781_178153


namespace finger_cycle_2004th_l1781_178119

def finger_sequence : List String :=
  ["Little finger", "Ring finger", "Middle finger", "Index finger", "Thumb", "Index finger", "Middle finger", "Ring finger"]

theorem finger_cycle_2004th : 
  finger_sequence.get! ((2004 - 1) % finger_sequence.length) = "Index finger" :=
by
  -- The proof is not required, so we use sorry
  sorry

end finger_cycle_2004th_l1781_178119


namespace min_value_expression_l1781_178166

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ m : ℝ, (m = 4 + 6 * Real.sqrt 2) ∧ 
  ∀ a b : ℝ, (0 < a) → (0 < b) → m ≤ (Real.sqrt ((a^2 + b^2) * (2*a^2 + 4*b^2))) / (a * b) :=
by sorry

end min_value_expression_l1781_178166


namespace smallest_percentage_increase_l1781_178101

theorem smallest_percentage_increase :
  let n2005 := 75
  let n2006 := 85
  let n2007 := 88
  let n2008 := 94
  let n2009 := 96
  let n2010 := 102
  let perc_increase (a b : ℕ) := ((b - a) : ℚ) / a * 100
  perc_increase n2008 n2009 < perc_increase n2006 n2007 ∧
  perc_increase n2008 n2009 < perc_increase n2007 n2008 ∧
  perc_increase n2008 n2009 < perc_increase n2009 n2010 ∧
  perc_increase n2008 n2009 < perc_increase n2005 n2006
:= sorry

end smallest_percentage_increase_l1781_178101


namespace student_weight_l1781_178155

-- Definitions based on conditions
variables (S R : ℝ)

-- Conditions as assertions
def condition1 : Prop := S - 5 = 2 * R
def condition2 : Prop := S + R = 104

-- The statement we want to prove
theorem student_weight (h1 : condition1 S R) (h2 : condition2 S R) : S = 71 :=
by
  sorry

end student_weight_l1781_178155


namespace lengths_of_legs_l1781_178183

def is_right_triangle (a b c : ℕ) := a^2 + b^2 = c^2

theorem lengths_of_legs (a b : ℕ) 
  (h1 : is_right_triangle a b 60)
  (h2 : a + b = 84) 
  : (a = 48 ∧ b = 36) ∨ (a = 36 ∧ b = 48) :=
  sorry

end lengths_of_legs_l1781_178183


namespace polynomial_no_linear_term_l1781_178198

theorem polynomial_no_linear_term (m n : ℝ) :
  (∀ x : ℝ, (x - m) * (x - n) = x^2 + mn → n + m = 0) :=
sorry

end polynomial_no_linear_term_l1781_178198


namespace initial_orchid_bushes_l1781_178106

def final_orchid_bushes : ℕ := 35
def orchid_bushes_to_be_planted : ℕ := 13

theorem initial_orchid_bushes :
  final_orchid_bushes - orchid_bushes_to_be_planted = 22 :=
by
  sorry

end initial_orchid_bushes_l1781_178106


namespace bridge_length_correct_l1781_178192

noncomputable def train_length : ℝ := 120
noncomputable def train_speed_kmph : ℝ := 45
noncomputable def crossing_time_seconds : ℝ := 30

noncomputable def train_speed_mps : ℝ := (train_speed_kmph * 1000) / 3600
noncomputable def total_distance : ℝ := train_speed_mps * crossing_time_seconds
noncomputable def bridge_length : ℝ := total_distance - train_length

theorem bridge_length_correct : bridge_length = 255 := by
  sorry

end bridge_length_correct_l1781_178192


namespace sockPairsCount_l1781_178185

noncomputable def countSockPairs : ℕ :=
  let whitePairs := Nat.choose 6 2 -- 15
  let brownPairs := Nat.choose 7 2 -- 21
  let bluePairs := Nat.choose 3 2 -- 3
  let oneRedOneWhite := 4 * 6 -- 24
  let oneRedOneBrown := 4 * 7 -- 28
  let oneRedOneBlue := 4 * 3 -- 12
  let bothRed := Nat.choose 4 2 -- 6
  whitePairs + brownPairs + bluePairs + oneRedOneWhite + oneRedOneBrown + oneRedOneBlue + bothRed

theorem sockPairsCount : countSockPairs = 109 := by
  sorry

end sockPairsCount_l1781_178185


namespace inequality_subtraction_l1781_178199

theorem inequality_subtraction (a b c : ℝ) (h : a > b) : a - c > b - c :=
sorry

end inequality_subtraction_l1781_178199


namespace binary_rep_of_21_l1781_178111

theorem binary_rep_of_21 : 
  (Nat.digits 2 21) = [1, 0, 1, 0, 1] := 
by 
  sorry

end binary_rep_of_21_l1781_178111


namespace find_number_l1781_178139

theorem find_number (x : ℝ) (h : 0.36 * x = 129.6) : x = 360 :=
by sorry

end find_number_l1781_178139


namespace statement_2_statement_3_l1781_178179

variable {α : Type*} [LinearOrderedField α]

-- Given a quadratic function
def quadratic (a b c x : α) : α :=
  a * x^2 + b * x + c

-- Statement 2
theorem statement_2 (a b c p q : α) (hpq : p ≠ q) :
  quadratic a b c p = quadratic a b c q → quadratic a b c (p + q) = c :=
sorry

-- Statement 3
theorem statement_3 (a b c p q : α) (hpq : p ≠ q) :
  quadratic a b c (p + q) = c → (p + q = 0 ∨ quadratic a b c p = quadratic a b c q) :=
sorry

end statement_2_statement_3_l1781_178179


namespace terrell_total_distance_l1781_178163

theorem terrell_total_distance (saturday_distance sunday_distance : ℝ) (h_saturday : saturday_distance = 8.2) (h_sunday : sunday_distance = 1.6) :
  saturday_distance + sunday_distance = 9.8 :=
by
  rw [h_saturday, h_sunday]
  -- sorry
  norm_num

end terrell_total_distance_l1781_178163


namespace yan_ratio_distance_l1781_178184

theorem yan_ratio_distance (w x y : ℕ) (h : w > 0) (h_eq_time : (y / w) = (x / w) + ((x + y) / (6 * w))) :
  x / y = 5 / 7 :=
by
  sorry

end yan_ratio_distance_l1781_178184


namespace find_angle_A_l1781_178173

theorem find_angle_A (A B C : ℝ)
  (h1 : C = 2 * B)
  (h2 : B = A / 3)
  (h3 : A + B + C = 180) : A = 90 :=
by
  sorry

end find_angle_A_l1781_178173


namespace total_clothes_washed_l1781_178144

theorem total_clothes_washed (cally_white_shirts : ℕ) (cally_colored_shirts : ℕ) (cally_shorts : ℕ) (cally_pants : ℕ) 
                             (danny_white_shirts : ℕ) (danny_colored_shirts : ℕ) (danny_shorts : ℕ) (danny_pants : ℕ) 
                             (total_clothes : ℕ)
                             (hcally : cally_white_shirts = 10 ∧ cally_colored_shirts = 5 ∧ cally_shorts = 7 ∧ cally_pants = 6)
                             (hdanny : danny_white_shirts = 6 ∧ danny_colored_shirts = 8 ∧ danny_shorts = 10 ∧ danny_pants = 6)
                             (htotal : total_clothes = 58) : 
  cally_white_shirts + cally_colored_shirts + cally_shorts + cally_pants + 
  danny_white_shirts + danny_colored_shirts + danny_shorts + danny_pants = total_clothes := 
by {
  sorry
}

end total_clothes_washed_l1781_178144


namespace solve_system_of_equations_l1781_178124

theorem solve_system_of_equations (x y : ℝ) (h1 : x + y = 5) (h2 : 2 * x - y = 1) : x = 2 ∧ y = 3 := 
sorry

end solve_system_of_equations_l1781_178124


namespace working_together_time_l1781_178174

/-- A is 30% more efficient than B,
and A alone can complete the job in 23 days.
Prove that A and B working together take approximately 13 days to complete the job. -/
theorem working_together_time (Ea Eb : ℝ) (T : ℝ) (h1 : Ea = 1.30 * Eb) 
  (h2 : 1 / 23 = Ea) : T = 13 :=
sorry

end working_together_time_l1781_178174


namespace quadratic_has_distinct_real_roots_l1781_178100

-- Definitions of the coefficients
def a : ℝ := 1
def b : ℝ := -1
def c : ℝ := -2

-- Definition of the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The theorem stating the quadratic equation has two distinct real roots
theorem quadratic_has_distinct_real_roots :
  discriminant a b c > 0 :=
by
  -- Coefficients specific to the problem
  unfold a b c
  -- Calculate the discriminant
  unfold discriminant
  -- Substitute the values and compute
  sorry -- Skipping the actual proof as per instructions

end quadratic_has_distinct_real_roots_l1781_178100


namespace xy_value_is_one_l1781_178147

open Complex

theorem xy_value_is_one (x y : ℝ) (h : (1 + I) * x + (1 - I) * y = 2) : x * y = 1 :=
by
  sorry

end xy_value_is_one_l1781_178147


namespace value_of_expression_l1781_178125

variables (x y z : ℝ)

axiom eq1 : 3 * x - 4 * y - 2 * z = 0
axiom eq2 : 2 * x + 6 * y - 21 * z = 0
axiom z_ne_zero : z ≠ 0

theorem value_of_expression : (x^2 + 4 * x * y) / (y^2 + z^2) = 7 :=
sorry

end value_of_expression_l1781_178125


namespace initial_investment_B_l1781_178181
-- Import necessary Lean library

-- Define the necessary conditions and theorems
theorem initial_investment_B (x : ℝ) (profit_A : ℝ) (profit_total : ℝ)
  (initial_A : ℝ) (initial_A_after_8_months : ℝ) (profit_B : ℝ) 
  (initial_A_months : ℕ) (initial_A_after_8_months_months : ℕ) 
  (initial_B_months : ℕ) (initial_B_after_8_months_months : ℕ) : 
  initial_A = 3000 ∧ initial_A_after_8_months = 2000 ∧
  profit_A = 240 ∧ profit_total = 630 ∧ 
  profit_B = profit_total - profit_A ∧
  (initial_A * initial_A_months + initial_A_after_8_months * initial_A_after_8_months_months) /
  ((initial_B_months * x + initial_B_after_8_months_months * (x + 1000))) = 
  profit_A / profit_B →
  x = 4000 :=
by
  sorry

end initial_investment_B_l1781_178181


namespace problem_I_inequality_solution_problem_II_condition_on_b_l1781_178117

-- Define the function f(x).
def f (x : ℝ) : ℝ := |x - 2|

-- Problem (I): Proving the solution set to the given inequality.
theorem problem_I_inequality_solution (x : ℝ) : 
  f x + f (x + 1) ≥ 5 ↔ (x ≥ 4 ∨ x ≤ -1) :=
sorry

-- Problem (II): Proving the condition on |b|.
theorem problem_II_condition_on_b (a b : ℝ) (ha : |a| > 1) (h : f (a * b) > |a| * f (b / a)) :
  |b| > 2 :=
sorry

end problem_I_inequality_solution_problem_II_condition_on_b_l1781_178117


namespace difference_of_students_l1781_178188

variable (G1 G2 G5 : ℕ)

theorem difference_of_students (h1 : G1 + G2 > G2 + G5) (h2 : G5 = G1 - 30) : 
  (G1 + G2) - (G2 + G5) = 30 :=
by
  sorry

end difference_of_students_l1781_178188


namespace total_revenue_correct_l1781_178127

def price_per_book : ℝ := 25
def revenue_monday : ℝ := 60 * ((price_per_book * 0.9) * 1.05)
def revenue_tuesday : ℝ := 10 * (price_per_book * 1.03)
def revenue_wednesday : ℝ := 20 * ((price_per_book * 0.95) * 1.02)
def revenue_thursday : ℝ := 44 * ((price_per_book * 0.85) * 1.04)
def revenue_friday : ℝ := 66 * (price_per_book * 0.8)

def total_revenue : ℝ :=
  revenue_monday + revenue_tuesday + revenue_wednesday +
  revenue_thursday + revenue_friday

theorem total_revenue_correct :
  total_revenue = 4452.4 :=
by
  rw [total_revenue, revenue_monday, revenue_tuesday, revenue_wednesday, 
      revenue_thursday, revenue_friday]
  -- Verification steps would continue by calculating each term.
  sorry

end total_revenue_correct_l1781_178127


namespace total_students_course_l1781_178193

theorem total_students_course 
  (T : ℕ)
  (H1 : (1 / 5 : ℚ) * T = (1 / 5) * T)
  (H2 : (1 / 4 : ℚ) * T = (1 / 4) * T)
  (H3 : (1 / 2 : ℚ) * T = (1 / 2) * T)
  (H4 : T = (1 / 5 : ℚ) * T + (1 / 4 : ℚ) * T + (1 / 2 : ℚ) * T + 30) : 
  T = 600 :=
sorry

end total_students_course_l1781_178193


namespace frog_eyes_in_pond_l1781_178156

-- Definitions based on conditions
def num_frogs : ℕ := 6
def eyes_per_frog : ℕ := 2

-- The property to be proved
theorem frog_eyes_in_pond : num_frogs * eyes_per_frog = 12 :=
by
  sorry

end frog_eyes_in_pond_l1781_178156


namespace walking_speed_l1781_178190

noncomputable def bridge_length : ℝ := 2500  -- length of the bridge in meters
noncomputable def crossing_time_minutes : ℝ := 15  -- time to cross the bridge in minutes
noncomputable def conversion_factor_time : ℝ := 1 / 60  -- factor to convert minutes to hours
noncomputable def conversion_factor_distance : ℝ := 1 / 1000  -- factor to convert meters to kilometers

theorem walking_speed (bridge_length crossing_time_minutes conversion_factor_time conversion_factor_distance : ℝ) : 
  bridge_length = 2500 → 
  crossing_time_minutes = 15 → 
  conversion_factor_time = 1 / 60 → 
  conversion_factor_distance = 1 / 1000 → 
  (bridge_length * conversion_factor_distance) / (crossing_time_minutes * conversion_factor_time) = 10 := 
by
  sorry

end walking_speed_l1781_178190
