import Mathlib

namespace NUMINAMATH_GPT_virginia_sweettarts_l42_4243

theorem virginia_sweettarts (total_sweettarts : ℕ) (sweettarts_per_person : ℕ) (friends : ℕ) (sweettarts_left : ℕ) 
  (h1 : total_sweettarts = 13) 
  (h2 : sweettarts_per_person = 3) 
  (h3 : total_sweettarts = sweettarts_per_person * (friends + 1) + sweettarts_left) 
  (h4 : sweettarts_left < sweettarts_per_person) :
  friends = 3 :=
by
  sorry

end NUMINAMATH_GPT_virginia_sweettarts_l42_4243


namespace NUMINAMATH_GPT_number_of_connections_l42_4271

theorem number_of_connections (n k : ℕ) (h1 : n = 30) (h2 : k = 4) :
  (n * k) / 2 = 60 :=
by
  sorry

end NUMINAMATH_GPT_number_of_connections_l42_4271


namespace NUMINAMATH_GPT_james_total_cost_l42_4241

def subscription_cost (base_cost : ℕ) (free_hours : ℕ) (extra_hour_cost : ℕ) (movie_rental_cost : ℝ) (streamed_hours : ℕ) (rented_movies : ℕ) : ℝ :=
  let extra_hours := max (streamed_hours - free_hours) 0
  base_cost + extra_hours * extra_hour_cost + rented_movies * movie_rental_cost

theorem james_total_cost 
  (base_cost : ℕ)
  (free_hours : ℕ)
  (extra_hour_cost : ℕ)
  (movie_rental_cost : ℝ)
  (streamed_hours : ℕ)
  (rented_movies : ℕ)
  (h_base_cost : base_cost = 15)
  (h_free_hours : free_hours = 50)
  (h_extra_hour_cost : extra_hour_cost = 2)
  (h_movie_rental_cost : movie_rental_cost = 0.10)
  (h_streamed_hours : streamed_hours = 53)
  (h_rented_movies : rented_movies = 30) :
  subscription_cost base_cost free_hours extra_hour_cost movie_rental_cost streamed_hours rented_movies = 24 := 
by {
  sorry
}

end NUMINAMATH_GPT_james_total_cost_l42_4241


namespace NUMINAMATH_GPT_bill_project_days_l42_4233

theorem bill_project_days (naps: ℕ) (hours_per_nap: ℕ) (working_hours: ℕ) : 
  (naps = 6) → (hours_per_nap = 7) → (working_hours = 54) → 
  (naps * hours_per_nap + working_hours) / 24 = 4 := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_bill_project_days_l42_4233


namespace NUMINAMATH_GPT_average_speed_of_train_l42_4283

theorem average_speed_of_train (distance time : ℝ) (h1 : distance = 80) (h2 : time = 8) :
  distance / time = 10 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_of_train_l42_4283


namespace NUMINAMATH_GPT_negation_of_universal_statement_l42_4231

theorem negation_of_universal_statement :
  (¬ (∀ x : ℝ, x > Real.sin x)) ↔ (∃ x : ℝ, x ≤ Real.sin x) :=
by sorry

end NUMINAMATH_GPT_negation_of_universal_statement_l42_4231


namespace NUMINAMATH_GPT_num_common_points_l42_4296

noncomputable def curve (x : ℝ) : ℝ := 3 * x ^ 4 - 2 * x ^ 3 - 9 * x ^ 2 + 4

noncomputable def tangent_line (x : ℝ) : ℝ :=
  -12 * (x - 1) - 4

theorem num_common_points :
  ∃ (x1 x2 x3 : ℝ), curve x1 = tangent_line x1 ∧
                    curve x2 = tangent_line x2 ∧
                    curve x3 = tangent_line x3 ∧
                    (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) :=
sorry

end NUMINAMATH_GPT_num_common_points_l42_4296


namespace NUMINAMATH_GPT_value_of_m_l42_4230

-- Problem Statement
theorem value_of_m (m : ℝ) : (∃ x : ℝ, (m-2)*x^(|m|-1) + 16 = 0 ∧ |m| - 1 = 1) → m = -2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_l42_4230


namespace NUMINAMATH_GPT_train_departure_time_l42_4239

-- Conditions
def arrival_time : ℕ := 1000  -- Representing 10:00 as 1000 (in minutes since midnight)
def travel_time : ℕ := 15  -- 15 minutes

-- Definition of time subtraction
def time_sub (arrival : ℕ) (travel : ℕ) : ℕ :=
arrival - travel

-- Proof that the train left at 9:45
theorem train_departure_time : time_sub arrival_time travel_time = 945 := by
  sorry

end NUMINAMATH_GPT_train_departure_time_l42_4239


namespace NUMINAMATH_GPT_problem_statement_l42_4258

-- Definitions of A and B based on the given conditions
def A : ℤ := -5 * -3
def B : ℤ := 2 - 2

-- The theorem stating that A + B = 15
theorem problem_statement : A + B = 15 := 
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l42_4258


namespace NUMINAMATH_GPT_chloe_probability_l42_4217

theorem chloe_probability :
  let total_numbers := 60
  let multiples_of_4 := 15
  let non_multiples_of_4_prob := 3 / 4
  let neither_multiple_of_4_prob := (non_multiples_of_4_prob) ^ 2
  let at_least_one_multiple_of_4_prob := 1 - neither_multiple_of_4_prob
  at_least_one_multiple_of_4_prob = 7 / 16 := by
  sorry

end NUMINAMATH_GPT_chloe_probability_l42_4217


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l42_4245

-- Proof Problem 1: $A$ and $B$ are not standing together
theorem problem1 : 
  ∃ (n : ℕ), n = 480 ∧ 
  ∀ (students : Fin 6 → String),
    students 0 ≠ "A" ∨ students 1 ≠ "B" :=
sorry

-- Proof Problem 2: $C$ and $D$ must stand together
theorem problem2 : 
  ∃ (n : ℕ), n = 240 ∧ 
  ∀ (students : Fin 6 → String),
    (students 0 = "C" ∧ students 1 = "D") ∨ 
    (students 1 = "C" ∧ students 2 = "D") :=
sorry

-- Proof Problem 3: $E$ is not at the beginning and $F$ is not at the end
theorem problem3 : 
  ∃ (n : ℕ), n = 504 ∧ 
  ∀ (students : Fin 6 → String),
    students 0 ≠ "E" ∧ students 5 ≠ "F" :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l42_4245


namespace NUMINAMATH_GPT_A_inter_B_eq_A_l42_4281

def A := {x : ℝ | 0 < x ∧ x ≤ 2}
def B := {x : ℝ | x ≤ 3}

theorem A_inter_B_eq_A : A ∩ B = A := 
by 
  sorry 

end NUMINAMATH_GPT_A_inter_B_eq_A_l42_4281


namespace NUMINAMATH_GPT_count_integer_values_l42_4270

theorem count_integer_values (x : ℤ) (h1 : 4 < Real.sqrt (3 * x + 1)) (h2 : Real.sqrt (3 * x + 1) < 5) : 
  (5 < x ∧ x < 8 ∧ ∃ (N : ℕ), N = 2) :=
by sorry

end NUMINAMATH_GPT_count_integer_values_l42_4270


namespace NUMINAMATH_GPT_solve_equation_l42_4236

theorem solve_equation (x : ℝ) (h₀ : x ≠ -3) (h₁ : (2 / (x + 3)) + (3 * x / (x + 3)) - (5 / (x + 3)) = 2) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l42_4236


namespace NUMINAMATH_GPT_percentage_of_students_wearing_red_shirts_l42_4277

/-- In a school of 700 students:
    - 45% of students wear blue shirts.
    - 15% of students wear green shirts.
    - 119 students wear shirts of other colors.
    We are proving that the percentage of students wearing red shirts is 23%. --/
theorem percentage_of_students_wearing_red_shirts:
  let total_students := 700
  let blue_shirt_percentage := 45 / 100
  let green_shirt_percentage := 15 / 100
  let other_colors_students := 119
  let students_with_blue_shirts := blue_shirt_percentage * total_students
  let students_with_green_shirts := green_shirt_percentage * total_students
  let students_with_other_colors := other_colors_students
  let students_with_blue_green_or_red_shirts := total_students - students_with_other_colors
  let students_with_red_shirts := students_with_blue_green_or_red_shirts - students_with_blue_shirts - students_with_green_shirts
  (students_with_red_shirts / total_students) * 100 = 23 := by
  sorry

end NUMINAMATH_GPT_percentage_of_students_wearing_red_shirts_l42_4277


namespace NUMINAMATH_GPT_find_c_l42_4262

theorem find_c {A B C : ℝ} (a b c : ℝ) (h1 : a = 3) (h2 : b = 2) 
(h3 : a * Real.sin A + b * Real.sin B - c * Real.sin C = (6 * Real.sqrt 7 / 7) * a * Real.sin B * Real.sin C) :
  c = 2 :=
sorry

end NUMINAMATH_GPT_find_c_l42_4262


namespace NUMINAMATH_GPT_price_of_20_percent_stock_l42_4246

theorem price_of_20_percent_stock (annual_income : ℝ) (investment : ℝ) (dividend_rate : ℝ) (price_of_stock : ℝ) :
  annual_income = 1000 →
  investment = 6800 →
  dividend_rate = 20 →
  price_of_stock = 136 :=
by
  intros h_income h_investment h_dividend_rate
  sorry

end NUMINAMATH_GPT_price_of_20_percent_stock_l42_4246


namespace NUMINAMATH_GPT_solve_for_x_l42_4221

theorem solve_for_x : ∃ x : ℚ, 5 * (2 * x - 3) = 3 * (3 - 4 * x) + 15 ∧ x = (39 : ℚ) / 22 :=
by
  use (39 : ℚ) / 22
  sorry

end NUMINAMATH_GPT_solve_for_x_l42_4221


namespace NUMINAMATH_GPT_probability_correct_l42_4238

-- Definitions of given conditions
def P_AB := 2 / 3
def P_BC := 1 / 2

-- Probability that at least one road is at least 5 miles long
def probability_at_least_one_road_is_5_miles_long : ℚ :=
  1 - (1 - P_AB) * (1 - P_BC)

theorem probability_correct :
  probability_at_least_one_road_is_5_miles_long = 5 / 6 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_probability_correct_l42_4238


namespace NUMINAMATH_GPT_total_growing_space_l42_4269

noncomputable def garden_area : ℕ :=
  let area_3x3 := 3 * 3
  let total_area_3x3 := 2 * area_3x3
  let area_4x3 := 4 * 3
  let total_area_4x3 := 2 * area_4x3
  total_area_3x3 + total_area_4x3

theorem total_growing_space : garden_area = 42 :=
by
  sorry

end NUMINAMATH_GPT_total_growing_space_l42_4269


namespace NUMINAMATH_GPT_butterfat_mixture_l42_4219

/-
  Given:
  - 8 gallons of milk with 40% butterfat
  - x gallons of milk with 10% butterfat
  - Resulting mixture with 20% butterfat

  Prove:
  - x = 16 gallons
-/

theorem butterfat_mixture (x : ℝ) : 
  (0.40 * 8 + 0.10 * x) / (8 + x) = 0.20 → x = 16 := 
by
  sorry

end NUMINAMATH_GPT_butterfat_mixture_l42_4219


namespace NUMINAMATH_GPT_total_clouds_count_l42_4232

-- Definitions based on the conditions
def carson_clouds := 12
def little_brother_clouds := 5 * carson_clouds
def older_sister_clouds := carson_clouds / 2

-- The theorem statement that needs to be proved
theorem total_clouds_count : carson_clouds + little_brother_clouds + older_sister_clouds = 78 := by
  -- Definitions
  have h1 : carson_clouds = 12 := rfl
  have h2 : little_brother_clouds = 5 * 12 := rfl
  have h3 : older_sister_clouds = 12 / 2 := rfl
  sorry

end NUMINAMATH_GPT_total_clouds_count_l42_4232


namespace NUMINAMATH_GPT_range_of_a_l42_4299

noncomputable def quadratic (a x : ℝ) : ℝ := a * x^2 - 2 * x + a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, quadratic a x > 0) ↔ 0 < a ∧ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l42_4299


namespace NUMINAMATH_GPT_problem_statement_l42_4220

def p (x y : ℝ) : Prop :=
  (x^2 + y^2 ≠ 0) → ¬ (x = 0 ∧ y = 0)

def q (m : ℝ) : Prop :=
  (m > -2) → ∃ x : ℝ, x^2 + 2*x - m = 0

theorem problem_statement : ∀ (x y m : ℝ), p x y ∨ q m :=
sorry

end NUMINAMATH_GPT_problem_statement_l42_4220


namespace NUMINAMATH_GPT_find_values_l42_4284

theorem find_values (a b c : ℕ) 
    (h1 : a + b + c = 1024) 
    (h2 : c = b - 88) 
    (h3 : a = b + c) : 
    a = 712 ∧ b = 400 ∧ c = 312 :=
by {
    sorry
}

end NUMINAMATH_GPT_find_values_l42_4284


namespace NUMINAMATH_GPT_rectangle_color_invariance_l42_4255

/-- A theorem stating that in any 3x7 rectangle with some cells colored black at random, there necessarily exist four cells of the same color, whose centers are the vertices of a rectangle with sides parallel to the sides of the original rectangle. -/
theorem rectangle_color_invariance :
  ∀ (color : Fin 3 × Fin 7 → Bool), 
  ∃ i1 i2 j1 j2 : Fin 3, i1 < i2 ∧ j1 < j2 ∧ 
  color ⟨i1, j1⟩ = color ⟨i1, j2⟩ ∧ 
  color ⟨i1, j1⟩ = color ⟨i2, j1⟩ ∧ 
  color ⟨i1, j1⟩ = color ⟨i2, j2⟩ :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_rectangle_color_invariance_l42_4255


namespace NUMINAMATH_GPT_dk_is_odd_l42_4222

def NTypePermutations (k : ℕ) (x : Fin (3 * k + 1) → ℕ) : Prop :=
  (∀ i j : Fin (k + 1), i < j → x i < x j) ∧
  (∀ i j : Fin (k + 1), i < j → x (k + 1 + i) > x (k + 1 + j)) ∧
  (∀ i j : Fin (k + 1), i < j → x (2 * k + 1 + i) < x (2 * k + 1 + j))

def countNTypePermutations (k : ℕ) : ℕ :=
  sorry -- This would be the count of all N-type permutations, use advanced combinatorics or algorithms

theorem dk_is_odd (k : ℕ) (h : 0 < k) : ∃ d : ℕ, countNTypePermutations k = 2 * d + 1 :=
  sorry

end NUMINAMATH_GPT_dk_is_odd_l42_4222


namespace NUMINAMATH_GPT_minimum_negative_factors_l42_4292

theorem minimum_negative_factors (a b c d : ℝ) (h1 : a * b * c * d < 0) (h2 : a + b = 0) (h3 : c * d > 0) : 
    (∃ x ∈ [a, b, c, d], x < 0) :=
by
  sorry

end NUMINAMATH_GPT_minimum_negative_factors_l42_4292


namespace NUMINAMATH_GPT_k_gonal_number_proof_l42_4209

-- Definitions for specific k-gonal numbers based on given conditions.
def triangular_number (n : ℕ) := (1/2 : ℚ) * n^2 + (1/2 : ℚ) * n
def square_number (n : ℕ) := n^2
def pentagonal_number (n : ℕ) := (3/2 : ℚ) * n^2 - (1/2 : ℚ) * n
def hexagonal_number (n : ℕ) := 2 * n^2 - n

-- General definition for the k-gonal number
def k_gonal_number (n k : ℕ) : ℚ := ((k - 2) / 2) * n^2 + ((4 - k) / 2) * n

-- Corresponding Lean statement for the proof problem
theorem k_gonal_number_proof (n k : ℕ) (hk : k ≥ 3) :
    (k = 3 -> triangular_number n = k_gonal_number n k) ∧
    (k = 4 -> square_number n = k_gonal_number n k) ∧
    (k = 5 -> pentagonal_number n = k_gonal_number n k) ∧
    (k = 6 -> hexagonal_number n = k_gonal_number n k) ∧
    (n = 10 ∧ k = 24 -> k_gonal_number n k = 1000) :=
by
  intros
  sorry

end NUMINAMATH_GPT_k_gonal_number_proof_l42_4209


namespace NUMINAMATH_GPT_find_a2_l42_4293

open Classical

variable {a_n : ℕ → ℝ} {q : ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n m : ℕ, a (n + m) = a n * q ^ m

theorem find_a2 (h1 : geometric_sequence a_n q)
                (h2 : a_n 7 = 1 / 4)
                (h3 : a_n 3 * a_n 5 = 4 * (a_n 4 - 1)) :
  a_n 2 = 8 :=
sorry

end NUMINAMATH_GPT_find_a2_l42_4293


namespace NUMINAMATH_GPT_length_of_train_l42_4290

theorem length_of_train (speed_kmh : ℝ) (time_min : ℝ) (tunnel_length_m : ℝ) (train_length_m : ℝ) :
  speed_kmh = 78 → time_min = 1 → tunnel_length_m = 500 → train_length_m = 800.2 :=
by
  sorry

end NUMINAMATH_GPT_length_of_train_l42_4290


namespace NUMINAMATH_GPT_james_total_earnings_l42_4200

def january_earnings : ℕ := 4000
def february_earnings : ℕ := january_earnings + (50 * january_earnings / 100)
def march_earnings : ℕ := february_earnings - (20 * february_earnings / 100)
def total_earnings : ℕ := january_earnings + february_earnings + march_earnings

theorem james_total_earnings :
  total_earnings = 14800 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_james_total_earnings_l42_4200


namespace NUMINAMATH_GPT_four_digit_palindrome_divisible_by_11_probability_zero_l42_4229

theorem four_digit_palindrome_divisible_by_11_probability_zero :
  (∃ a b : ℕ, 2 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ (1001 * a + 110 * b) % 11 = 0) = false :=
by sorry

end NUMINAMATH_GPT_four_digit_palindrome_divisible_by_11_probability_zero_l42_4229


namespace NUMINAMATH_GPT_rectangle_sides_l42_4210

theorem rectangle_sides (x y : ℝ) (h1 : 4 * x = 3 * y) (h2 : x * y = 2 * (x + y)) :
  (x = 7 / 2 ∧ y = 14 / 3) ∨ (x = 14 / 3 ∧ y = 7 / 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_rectangle_sides_l42_4210


namespace NUMINAMATH_GPT_max_distinct_rectangles_l42_4240

theorem max_distinct_rectangles : 
  ∃ (rectangles : Finset ℕ), (∀ n ∈ rectangles, n > 0) ∧ rectangles.sum id = 100 ∧ rectangles.card = 14 :=
by 
  sorry

end NUMINAMATH_GPT_max_distinct_rectangles_l42_4240


namespace NUMINAMATH_GPT_cheaper_to_buy_more_books_l42_4280

def C (n : ℕ) : ℕ :=
  if n < 1 then 0
  else if n ≤ 20 then 15 * n
  else if n ≤ 40 then 14 * n - 5
  else 13 * n

noncomputable def apply_discount (n : ℕ) (cost : ℕ) : ℕ :=
  cost - 10 * (n / 10)

theorem cheaper_to_buy_more_books : 
  ∃ (n_vals : Finset ℕ), n_vals.card = 5 ∧ ∀ n ∈ n_vals, apply_discount (n + 1) (C (n + 1)) < apply_discount n (C n) :=
sorry

end NUMINAMATH_GPT_cheaper_to_buy_more_books_l42_4280


namespace NUMINAMATH_GPT_math_problem_l42_4213

noncomputable def f (x : ℝ) := |Real.exp x - 1|

theorem math_problem (x1 x2 : ℝ) (h1 : x1 < 0) (h2 : x2 > 0)
  (h3 : - Real.exp x1 * Real.exp x2 = -1) :
  (x1 + x2 = 0) ∧
  (0 < (Real.exp x2 + Real.exp x1 - 2) / (x2 - x1)) ∧
  (0 < Real.exp x1 ∧ Real.exp x1 < 1) :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l42_4213


namespace NUMINAMATH_GPT_boxes_needed_l42_4261

theorem boxes_needed (total_muffins available_boxes muffins_per_box : ℕ) (h1 : total_muffins = 95) 
  (h2 : available_boxes = 10) (h3 : muffins_per_box = 5) : 
  ((total_muffins - (available_boxes * muffins_per_box)) / muffins_per_box) = 9 := 
by
  -- the proof will be constructed here
  sorry

end NUMINAMATH_GPT_boxes_needed_l42_4261


namespace NUMINAMATH_GPT_solution_condition1_solution_condition2_solution_condition3_solution_condition4_l42_4206

-- Define the conditions
def Condition1 : Prop :=
  ∃ (total_population box1 box2 sampled : Nat),
  total_population = 30 ∧ box1 = 21 ∧ box2 = 9 ∧ sampled = 10

def Condition2 : Prop :=
  ∃ (total_population produced_by_A produced_by_B sampled : Nat),
  total_population = 30 ∧ produced_by_A = 21 ∧ produced_by_B = 9 ∧ sampled = 10

def Condition3 : Prop :=
  ∃ (total_population sampled : Nat),
  total_population = 300 ∧ sampled = 10

def Condition4 : Prop :=
  ∃ (total_population sampled : Nat),
  total_population = 300 ∧ sampled = 50

-- Define the appropriate methods
def LotteryMethod : Prop := ∃ method : String, method = "Lottery method"
def StratifiedSampling : Prop := ∃ method : String, method = "Stratified sampling"
def RandomNumberMethod : Prop := ∃ method : String, method = "Random number method"
def SystematicSampling : Prop := ∃ method : String, method = "Systematic sampling"

-- Statements to prove the appropriate methods for each condition
theorem solution_condition1 : Condition1 → LotteryMethod := by sorry
theorem solution_condition2 : Condition2 → StratifiedSampling := by sorry
theorem solution_condition3 : Condition3 → RandomNumberMethod := by sorry
theorem solution_condition4 : Condition4 → SystematicSampling := by sorry

end NUMINAMATH_GPT_solution_condition1_solution_condition2_solution_condition3_solution_condition4_l42_4206


namespace NUMINAMATH_GPT_probability_one_left_one_right_l42_4251

/-- Define the conditions: 12 left-handed gloves, 10 right-handed gloves. -/
def num_left_handed_gloves : ℕ := 12

def num_right_handed_gloves : ℕ := 10

/-- Total number of gloves is 22. -/
def total_gloves : ℕ := num_left_handed_gloves + num_right_handed_gloves

/-- Total number of ways to pick any two gloves from 22 gloves. -/
def total_pick_two_ways : ℕ := (total_gloves * (total_gloves - 1)) / 2

/-- Number of favorable outcomes picking one left-handed and one right-handed glove. -/
def favorable_outcomes : ℕ := num_left_handed_gloves * num_right_handed_gloves

/-- Define the probability as favorable outcomes divided by total outcomes. 
 It should yield 40/77. -/
theorem probability_one_left_one_right : 
  (favorable_outcomes : ℚ) / total_pick_two_ways = 40 / 77 :=
by
  -- Skip the proof.
  sorry

end NUMINAMATH_GPT_probability_one_left_one_right_l42_4251


namespace NUMINAMATH_GPT_sin_cos_double_angle_identity_l42_4253

theorem sin_cos_double_angle_identity (α : ℝ) 
  (h1 : Real.sin α = 1/3) 
  (h2 : α ∈ Set.Ioc (π/2) π) : 
  Real.sin (2*α) + Real.cos (2*α) = (7 - 4 * Real.sqrt 2) / 9 := 
by
  sorry

end NUMINAMATH_GPT_sin_cos_double_angle_identity_l42_4253


namespace NUMINAMATH_GPT_oldest_bride_age_l42_4260

theorem oldest_bride_age (B G : ℕ) (h1 : B = G + 19) (h2 : B + G = 185) :
  B = 102 :=
by
  sorry

end NUMINAMATH_GPT_oldest_bride_age_l42_4260


namespace NUMINAMATH_GPT_rectangular_solid_surface_area_l42_4286

theorem rectangular_solid_surface_area (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (hvol : a * b * c = 455) : 
  let surface_area := 2 * (a * b + b * c + c * a)
  surface_area = 382 := by
-- proof
sorry

end NUMINAMATH_GPT_rectangular_solid_surface_area_l42_4286


namespace NUMINAMATH_GPT_binom_8_3_eq_56_l42_4212

def binom (n k : ℕ) : ℕ :=
(n.factorial) / (k.factorial * (n - k).factorial)

theorem binom_8_3_eq_56 : binom 8 3 = 56 := by
  sorry

end NUMINAMATH_GPT_binom_8_3_eq_56_l42_4212


namespace NUMINAMATH_GPT_one_fifth_of_ten_x_plus_three_l42_4274

theorem one_fifth_of_ten_x_plus_three (x : ℝ) : 
  (1 / 5) * (10 * x + 3) = 2 * x + 3 / 5 := 
  sorry

end NUMINAMATH_GPT_one_fifth_of_ten_x_plus_three_l42_4274


namespace NUMINAMATH_GPT_addition_neg3_plus_2_multiplication_neg3_times_2_l42_4249

theorem addition_neg3_plus_2 : -3 + 2 = -1 :=
  by
    sorry

theorem multiplication_neg3_times_2 : (-3) * 2 = -6 :=
  by
    sorry

end NUMINAMATH_GPT_addition_neg3_plus_2_multiplication_neg3_times_2_l42_4249


namespace NUMINAMATH_GPT_polygon_edges_l42_4278

theorem polygon_edges (n : ℕ) (h1 : (n - 2) * 180 = 4 * 360 + 180) : n = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_polygon_edges_l42_4278


namespace NUMINAMATH_GPT_ace_first_king_second_prob_l42_4218

def cards : Type := { x : ℕ // x < 52 }

def ace (c : cards) : Prop := 
  c.1 = 0 ∨ c.1 = 1 ∨ c.1 = 2 ∨ c.1 = 3

def king (c : cards) : Prop := 
  c.1 = 4 ∨ c.1 = 5 ∨ c.1 = 6 ∨ c.1 = 7

def prob_ace_first_king_second : ℚ := 4 / 52 * 4 / 51

theorem ace_first_king_second_prob :
  prob_ace_first_king_second = 4 / 663 := by
  sorry

end NUMINAMATH_GPT_ace_first_king_second_prob_l42_4218


namespace NUMINAMATH_GPT_solve_for_x_l42_4215

theorem solve_for_x : 
  ∃ x : ℝ, (x^2 + 6 * x + 8 = -(x + 2) * (x + 6)) ∧ (x = -2 ∨ x = -5) :=
sorry

end NUMINAMATH_GPT_solve_for_x_l42_4215


namespace NUMINAMATH_GPT_num_divisors_m2_less_than_m_not_divide_m_l42_4204

namespace MathProof

def m : ℕ := 2^20 * 3^15 * 5^6

theorem num_divisors_m2_less_than_m_not_divide_m :
  let m2 := m ^ 2
  let total_divisors_m2 := 41 * 31 * 13
  let total_divisors_m := 21 * 16 * 7
  let divisors_m2_less_than_m := (total_divisors_m2 - 1) / 2
  divisors_m2_less_than_m - total_divisors_m = 5924 :=
by sorry

end MathProof

end NUMINAMATH_GPT_num_divisors_m2_less_than_m_not_divide_m_l42_4204


namespace NUMINAMATH_GPT_min_increase_air_quality_days_l42_4256

theorem min_increase_air_quality_days {days_in_year : ℕ} (last_year_ratio next_year_ratio : ℝ) (good_air_days : ℕ) :
  days_in_year = 365 → last_year_ratio = 0.6 → next_year_ratio > 0.7 →
  (good_air_days / days_in_year < last_year_ratio → ∀ n: ℕ, good_air_days + n ≥ 37) :=
by
  intros hdays_in_year hlast_year_ratio hnext_year_ratio h_good_air_days
  sorry

end NUMINAMATH_GPT_min_increase_air_quality_days_l42_4256


namespace NUMINAMATH_GPT_probability_is_two_thirds_l42_4202

noncomputable def probabilityOfEvent : ℚ :=
  let Ω := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 6 }
  let A := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 6 ∧ 2 * p.1 - p.2 + 2 ≥ 0 }
  let area_Ω := (2 - 0) * (6 - 0)
  let area_A := area_Ω - (1 / 2) * 2 * 4
  (area_A / area_Ω : ℚ)

theorem probability_is_two_thirds : probabilityOfEvent = (2 / 3 : ℚ) :=
  sorry

end NUMINAMATH_GPT_probability_is_two_thirds_l42_4202


namespace NUMINAMATH_GPT_find_ding_score_l42_4227

noncomputable def jia_yi_bing_avg_score : ℕ := 89
noncomputable def four_avg_score := jia_yi_bing_avg_score + 2
noncomputable def four_total_score := 4 * four_avg_score
noncomputable def jia_yi_bing_total_score := 3 * jia_yi_bing_avg_score
noncomputable def ding_score := four_total_score - jia_yi_bing_total_score

theorem find_ding_score : ding_score = 97 := 
by
  sorry

end NUMINAMATH_GPT_find_ding_score_l42_4227


namespace NUMINAMATH_GPT_constant_in_quadratic_eq_l42_4289

theorem constant_in_quadratic_eq (C : ℝ) (x₁ x₂ : ℝ) 
  (h1 : 2 * x₁ * x₁ + 5 * x₁ - C = 0) 
  (h2 : 2 * x₂ * x₂ + 5 * x₂ - C = 0) 
  (h3 : x₁ - x₂ = 5.5) : C = 12 := 
sorry

end NUMINAMATH_GPT_constant_in_quadratic_eq_l42_4289


namespace NUMINAMATH_GPT_solve_system_of_equations_l42_4275

theorem solve_system_of_equations :
  ∃ (x1 x2 x3 x4 x5 x6 x7 x8 : ℤ), 
    x1 + x2 + x3 = 6 ∧
    x2 + x3 + x4 = 9 ∧
    x3 + x4 + x5 = 3 ∧
    x4 + x5 + x6 = -3 ∧
    x5 + x6 + x7 = -9 ∧
    x6 + x7 + x8 = -6 ∧
    x7 + x8 + x1 = -2 ∧
    x8 + x1 + x2 = 2 ∧
    (x1, x2, x3, x4, x5, x6, x7, x8) = (1, 2, 3, 4, -4, -3, -2, -1) :=
by
  -- solution will be here
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l42_4275


namespace NUMINAMATH_GPT_fraction_of_value_l42_4242

def value_this_year : ℝ := 16000
def value_last_year : ℝ := 20000

theorem fraction_of_value : (value_this_year / value_last_year) = 4 / 5 := by
  sorry

end NUMINAMATH_GPT_fraction_of_value_l42_4242


namespace NUMINAMATH_GPT_determine_house_height_l42_4285

-- Definitions for the conditions
def house_shadow : ℚ := 75
def tree_height : ℚ := 15
def tree_shadow : ℚ := 20

-- Desired Height of Lily's house
def house_height : ℚ := 56

-- Theorem stating the height of the house
theorem determine_house_height :
  (house_shadow / tree_shadow = house_height / tree_height) -> house_height = 56 :=
  by
  unfold house_shadow tree_height tree_shadow house_height
  sorry

end NUMINAMATH_GPT_determine_house_height_l42_4285


namespace NUMINAMATH_GPT_solve_linear_system_l42_4272

theorem solve_linear_system (x y : ℝ) (h1 : 2 * x + 3 * y = 5) (h2 : 3 * x + 2 * y = 10) : x + y = 3 := 
by
  sorry

end NUMINAMATH_GPT_solve_linear_system_l42_4272


namespace NUMINAMATH_GPT_combined_PP_curve_l42_4211

-- Definitions based on the given conditions
def M1 (K : ℝ) : ℝ := 40 - 2 * K
def M2 (K : ℝ) : ℝ := 64 - K ^ 2
def combinedPPC (K1 K2 : ℝ) : ℝ := 128 - 0.5 * K1^2 + 40 - 2 * K2

theorem combined_PP_curve (K : ℝ) :
  (K ≤ 2 → combinedPPC K 0 = 168 - 0.5 * K^2) ∧
  (2 < K ∧ K ≤ 22 → combinedPPC 2 (K - 2) = 170 - 2 * K) ∧
  (22 < K ∧ K ≤ 36 → combinedPPC (K - 20) 20 = 20 * K - 0.5 * K^2 - 72) :=
by
  sorry

end NUMINAMATH_GPT_combined_PP_curve_l42_4211


namespace NUMINAMATH_GPT_find_x_l42_4266

theorem find_x (x : ℝ) (h : 3550 - (x / 20.04) = 3500) : x = 1002 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l42_4266


namespace NUMINAMATH_GPT_monotonicity_and_range_l42_4294

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_and_range (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, 0 < x ∧ x < 1/a → f a x < f a (1/a)) ∧ 
  (∀ x : ℝ, x > 1/a → f a x > f a (1/a)) ∧ 
  (∀ x : ℝ, f a x ≠ 0 → a > 1/Real.exp 1) :=
by
  sorry

end NUMINAMATH_GPT_monotonicity_and_range_l42_4294


namespace NUMINAMATH_GPT_find_p_l42_4279

theorem find_p (p : ℝ) :
  (∀ x : ℝ, x^2 + p * x + p - 1 = 0) →
  ((exists x1 x2 : ℝ, x^2 + p * x + p - 1 = 0 ∧ x1^2 + x1^3 = - (x2^2 + x2^3) ) → (p = 1 ∨ p = 2)) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_p_l42_4279


namespace NUMINAMATH_GPT_sin_double_angle_l42_4297

-- Define the point P
def P : ℝ × ℝ := (1, -2)

-- Define the angle α such that its terminal side passes through point P
noncomputable def α : ℝ := sorry -- The exact definition of α is not needed for this statement

-- Define r as the distance from the origin to the point P
noncomputable def r : ℝ := Real.sqrt ((P.1 ^ 2) + (P.2 ^ 2))

-- Define sin(α) and cos(α)
noncomputable def sin_α : ℝ := P.2 / r
noncomputable def cos_α : ℝ := P.1 / r

-- The proof statement
theorem sin_double_angle : 2 * sin_α * cos_α = -4 / 5 := by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l42_4297


namespace NUMINAMATH_GPT_mike_initial_cards_l42_4234

-- Define the conditions
def initial_cards (x : ℕ) := x + 13 = 100

-- Define the proof statement
theorem mike_initial_cards : initial_cards 87 :=
by
  sorry

end NUMINAMATH_GPT_mike_initial_cards_l42_4234


namespace NUMINAMATH_GPT_angle_A_area_triangle_l42_4264

-- The first problem: Proving angle A
theorem angle_A (a b c : ℝ) (A C : ℝ) 
  (h1 : (2 * b - c) * Real.cos A = a * Real.cos C) : 
  A = Real.pi / 3 :=
by sorry

-- The second problem: Finding the area of triangle ABC
theorem area_triangle (a b c : ℝ) (A : ℝ)
  (h1 : a = 3)
  (h2 : b = 2 * c)
  (h3 : A = Real.pi / 3) :
  0.5 * b * c * Real.sin A = 3 * Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_GPT_angle_A_area_triangle_l42_4264


namespace NUMINAMATH_GPT_correct_average_l42_4254

theorem correct_average (n : ℕ) (average incorrect correct : ℕ) (h1 : n = 10) (h2 : average = 15) 
(h3 : incorrect = 26) (h4 : correct = 36) :
  (n * average - incorrect + correct) / n = 16 :=
  sorry

end NUMINAMATH_GPT_correct_average_l42_4254


namespace NUMINAMATH_GPT_equation1_solution_equation2_no_solution_l42_4259

theorem equation1_solution (x: ℝ) (h: x ≠ -1/2 ∧ x ≠ 1):
  (1 / (x - 1) = 5 / (2 * x + 1)) ↔ (x = 2) :=
sorry

theorem equation2_no_solution (x: ℝ) (h: x ≠ 1 ∧ x ≠ -1):
  ¬ ( (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1 ) :=
sorry

end NUMINAMATH_GPT_equation1_solution_equation2_no_solution_l42_4259


namespace NUMINAMATH_GPT_total_price_of_property_l42_4295

theorem total_price_of_property (price_per_sq_ft: ℝ) (house_size barn_size: ℝ) (house_price barn_price total_price: ℝ) :
  price_per_sq_ft = 98 ∧ house_size = 2400 ∧ barn_size = 1000 → 
  house_price = price_per_sq_ft * house_size ∧
  barn_price = price_per_sq_ft * barn_size ∧
  total_price = house_price + barn_price →
  total_price = 333200 :=
by
  sorry

end NUMINAMATH_GPT_total_price_of_property_l42_4295


namespace NUMINAMATH_GPT_dan_total_purchase_cost_l42_4267

noncomputable def snake_toy_cost : ℝ := 11.76
noncomputable def cage_cost : ℝ := 14.54
noncomputable def heat_lamp_cost : ℝ := 6.25
noncomputable def cage_discount_rate : ℝ := 0.10
noncomputable def sales_tax_rate : ℝ := 0.08
noncomputable def found_dollar : ℝ := 1.00

noncomputable def total_cost : ℝ :=
  let cage_discount := cage_discount_rate * cage_cost
  let discounted_cage := cage_cost - cage_discount
  let subtotal_before_tax := snake_toy_cost + discounted_cage + heat_lamp_cost
  let sales_tax := sales_tax_rate * subtotal_before_tax
  let total_after_tax := subtotal_before_tax + sales_tax
  total_after_tax - found_dollar

theorem dan_total_purchase_cost : total_cost = 32.58 :=
  by 
    -- Placeholder for the proof
    sorry

end NUMINAMATH_GPT_dan_total_purchase_cost_l42_4267


namespace NUMINAMATH_GPT_alan_glasses_drank_l42_4205

-- Definition for the rate of drinking water
def glass_per_minutes := 1 / 20

-- Definition for the total time in minutes
def total_minutes := 5 * 60

-- Theorem stating the number of glasses Alan will drink in the given time
theorem alan_glasses_drank : (glass_per_minutes * total_minutes) = 15 :=
by 
  sorry

end NUMINAMATH_GPT_alan_glasses_drank_l42_4205


namespace NUMINAMATH_GPT_product_of_cubes_l42_4287

theorem product_of_cubes :
  ( (2^3 - 1) / (2^3 + 1) * (3^3 - 1) / (3^3 + 1) * (4^3 - 1) / (4^3 + 1) * 
    (5^3 - 1) / (5^3 + 1) * (6^3 - 1) / (6^3 + 1) * (7^3 - 1) / (7^3 + 1) 
  ) = 57 / 72 := 
by
  sorry

end NUMINAMATH_GPT_product_of_cubes_l42_4287


namespace NUMINAMATH_GPT_math_problem_l42_4265

-- Definitions of the conditions
variable (x y : ℝ)
axiom h1 : x + y = 5
axiom h2 : x * y = 3

-- Prove the desired equality
theorem math_problem : x + (x^4 / y^3) + (y^4 / x^3) + y = 1021 := 
by 
sorry

end NUMINAMATH_GPT_math_problem_l42_4265


namespace NUMINAMATH_GPT_not_sixth_power_of_integer_l42_4216

theorem not_sixth_power_of_integer (n : ℕ) : ¬ ∃ k : ℤ, 6 * n^3 + 3 = k^6 :=
by
  sorry

end NUMINAMATH_GPT_not_sixth_power_of_integer_l42_4216


namespace NUMINAMATH_GPT_initial_cherry_sweets_30_l42_4282

/-!
# Problem Statement
A packet of candy sweets has some cherry-flavored sweets (C), 40 strawberry-flavored sweets, 
and 50 pineapple-flavored sweets. Aaron eats half of each type of sweet and then gives away 
5 cherry-flavored sweets to his friend. There are still 55 sweets in the packet of candy.
Prove that the initial number of cherry-flavored sweets was 30.
-/

noncomputable def initial_cherry_sweets (C : ℕ) : Prop :=
  let remaining_cherry_sweets := C / 2 - 5
  let remaining_strawberry_sweets := 40 / 2
  let remaining_pineapple_sweets := 50 / 2
  remaining_cherry_sweets + remaining_strawberry_sweets + remaining_pineapple_sweets = 55

theorem initial_cherry_sweets_30 : initial_cherry_sweets 30 :=
  sorry

end NUMINAMATH_GPT_initial_cherry_sweets_30_l42_4282


namespace NUMINAMATH_GPT_star_k_l42_4257

def star (x y : ℤ) : ℤ := x^2 - 2 * y + 1

theorem star_k (k : ℤ) : star k (star k k) = -k^2 + 4 * k - 1 :=
by 
  sorry

end NUMINAMATH_GPT_star_k_l42_4257


namespace NUMINAMATH_GPT_solve_quadratics_l42_4250

theorem solve_quadratics :
  ∃ x y : ℝ, (9 * x^2 - 36 * x - 81 = 0) ∧ (y^2 + 6 * y + 9 = 0) ∧ (x + y = -1 + Real.sqrt 13 ∨ x + y = -1 - Real.sqrt 13) := 
by 
  sorry

end NUMINAMATH_GPT_solve_quadratics_l42_4250


namespace NUMINAMATH_GPT_green_paint_mixture_l42_4208

theorem green_paint_mixture :
  ∀ (x : ℝ), 
    let light_green_paint := 5
    let darker_green_paint := x
    let final_paint := light_green_paint + darker_green_paint
    1 + 0.4 * darker_green_paint = 0.25 * final_paint -> x = 5 / 3 := 
by 
  intros x
  let light_green_paint := 5
  let darker_green_paint := x
  let final_paint := light_green_paint + darker_green_paint
  sorry

end NUMINAMATH_GPT_green_paint_mixture_l42_4208


namespace NUMINAMATH_GPT_find_a_l42_4248

noncomputable def calculation (a : ℝ) (x : ℝ) (y : ℝ) (b : ℝ) (c : ℝ) : Prop :=
  (x * y) / (a * b * c) = 840

theorem find_a : calculation 50 0.0048 3.5 0.1 0.004 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l42_4248


namespace NUMINAMATH_GPT_domain_of_f_l42_4228

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 12))

theorem domain_of_f :
  ∀ x : ℝ, f x ≠ 0 ↔ (x ≠ 15 / 2) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l42_4228


namespace NUMINAMATH_GPT_range_of_a_l42_4214

noncomputable def A (a : ℝ) : Set ℝ := { x | 3 + a ≤ x ∧ x ≤ 4 + 3 * a }
noncomputable def B : Set ℝ := { x | -4 ≤ x ∧ x < 5 }

theorem range_of_a (a : ℝ) : A a ⊆ B ↔ -1/2 ≤ a ∧ a < 1/3 :=
  sorry

end NUMINAMATH_GPT_range_of_a_l42_4214


namespace NUMINAMATH_GPT_simplify_fractional_exponents_l42_4268

theorem simplify_fractional_exponents :
  (5 ^ (1/6) * 5 ^ (1/2)) / 5 ^ (1/3) = 5 ^ (1/6) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fractional_exponents_l42_4268


namespace NUMINAMATH_GPT_conclusion1_conclusion2_l42_4252

theorem conclusion1 (x y a b : ℝ) (h1 : 4^x = a) (h2 : 8^y = b) : 2^(2*x - 3*y) = a / b :=
sorry

theorem conclusion2 (x a : ℝ) (h1 : (x-1)*(x^2 + a*x + 1) - x^2 = x^3 - (a-1)*x^2 - (1-a)*x - 1) : a = 1 :=
sorry

end NUMINAMATH_GPT_conclusion1_conclusion2_l42_4252


namespace NUMINAMATH_GPT_exterior_angle_of_regular_octagon_l42_4273

def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)
def interior_angle (s : ℕ) (n : ℕ) : ℕ := sum_of_interior_angles n / s
def exterior_angle (ia : ℕ) : ℕ := 180 - ia

theorem exterior_angle_of_regular_octagon : 
    exterior_angle (interior_angle 8 8) = 45 := 
by 
  sorry

end NUMINAMATH_GPT_exterior_angle_of_regular_octagon_l42_4273


namespace NUMINAMATH_GPT_unique_two_digit_perfect_square_divisible_by_5_l42_4288

-- Define the conditions
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

-- The statement to prove: there is exactly 1 two-digit perfect square that is divisible by 5
theorem unique_two_digit_perfect_square_divisible_by_5 :
  ∃! n : ℕ, is_perfect_square n ∧ two_digit n ∧ divisible_by_5 n :=
sorry

end NUMINAMATH_GPT_unique_two_digit_perfect_square_divisible_by_5_l42_4288


namespace NUMINAMATH_GPT_find_coefficients_l42_4225

theorem find_coefficients (k b : ℝ) :
    (∀ x y : ℝ, (y = k * x) → ((x-2)^2 + y^2 = 1) → (2*x + y + b = 0)) →
    ((k = 1/2) ∧ (b = -4)) :=
by
  sorry

end NUMINAMATH_GPT_find_coefficients_l42_4225


namespace NUMINAMATH_GPT_triangle_right_angle_l42_4207

theorem triangle_right_angle (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : A = B - C) : B = 90 :=
by sorry

end NUMINAMATH_GPT_triangle_right_angle_l42_4207


namespace NUMINAMATH_GPT_regular_12gon_symmetry_and_angle_l42_4237

theorem regular_12gon_symmetry_and_angle :
  ∀ (L R : ℕ), 
  (L = 12) ∧ (R = 30) → 
  (L + R = 42) :=
by
  -- placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_regular_12gon_symmetry_and_angle_l42_4237


namespace NUMINAMATH_GPT_quadrant_conditions_l42_4291

-- Formalizing function and conditions in Lean specifics
variable {a b : ℝ}

theorem quadrant_conditions 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : 0 < a ∧ a < 1)
  (h4 : ∀ x < 0, a^x + b - 1 > 0)
  (h5 : ∀ x > 0, a^x + b - 1 > 0) :
  0 < b ∧ b < 1 := 
sorry

end NUMINAMATH_GPT_quadrant_conditions_l42_4291


namespace NUMINAMATH_GPT_soda_consumption_l42_4224

theorem soda_consumption 
    (dozens : ℕ)
    (people_per_dozen : ℕ)
    (cost_per_box : ℕ)
    (cans_per_box : ℕ)
    (family_members : ℕ)
    (payment_per_member : ℕ)
    (dozens_eq : dozens = 5)
    (people_per_dozen_eq : people_per_dozen = 12)
    (cost_per_box_eq : cost_per_box = 2)
    (cans_per_box_eq : cans_per_box = 10)
    (family_members_eq : family_members = 6)
    (payment_per_member_eq : payment_per_member = 4) :
  (60 * (cans_per_box)) / 60 = 2 :=
by
  -- proof would go here eventually
  sorry

end NUMINAMATH_GPT_soda_consumption_l42_4224


namespace NUMINAMATH_GPT_adult_ticket_cost_is_16_l42_4235

-- Define the problem
def group_size := 6 + 10 -- Total number of people
def child_tickets := 6 -- Number of children
def adult_tickets := 10 -- Number of adults
def child_ticket_cost := 10 -- Cost per child ticket
def total_ticket_cost := 220 -- Total cost for all tickets

-- Prove the cost of an adult ticket
theorem adult_ticket_cost_is_16 : 
  (total_ticket_cost - (child_tickets * child_ticket_cost)) / adult_tickets = 16 := by
  sorry

end NUMINAMATH_GPT_adult_ticket_cost_is_16_l42_4235


namespace NUMINAMATH_GPT_fraction_of_money_left_l42_4201

theorem fraction_of_money_left 
  (m c : ℝ) 
  (h1 : (1/4 : ℝ) * m = (1/2) * c) : 
  (m - c) / m = (1/2 : ℝ) :=
by
  -- the proof will be written here
  sorry

end NUMINAMATH_GPT_fraction_of_money_left_l42_4201


namespace NUMINAMATH_GPT_ab_range_l42_4203

theorem ab_range (a b : ℝ) (h : a * b = a + b + 3) : a * b ≤ 1 ∨ a * b ≥ 9 := by
  sorry

end NUMINAMATH_GPT_ab_range_l42_4203


namespace NUMINAMATH_GPT_side_length_of_square_l42_4226

theorem side_length_of_square (m : ℕ) (a : ℕ) (hm : m = 100) (ha : a^2 = m) : a = 10 :=
by 
  sorry

end NUMINAMATH_GPT_side_length_of_square_l42_4226


namespace NUMINAMATH_GPT_closest_point_l42_4263

theorem closest_point 
  (x y z : ℝ) 
  (h_plane : 3 * x - 4 * y + 5 * z = 30)
  (A : ℝ × ℝ × ℝ := (1, 2, 3)) 
  (P : ℝ × ℝ × ℝ := (x, y, z)) :
  P = (11 / 5, 2 / 5, 5) := 
sorry

end NUMINAMATH_GPT_closest_point_l42_4263


namespace NUMINAMATH_GPT_polygon_interior_sum_sum_of_exterior_angles_l42_4247

theorem polygon_interior_sum (n : ℕ) (h : (n - 2) * 180 = 1080) : n = 8 :=
by
  sorry

theorem sum_of_exterior_angles (n : ℕ) : 360 = 360 :=
by
  sorry

end NUMINAMATH_GPT_polygon_interior_sum_sum_of_exterior_angles_l42_4247


namespace NUMINAMATH_GPT_students_on_bus_l42_4276

theorem students_on_bus
    (initial_students : ℝ) (first_get_on : ℝ) (first_get_off : ℝ)
    (second_get_on : ℝ) (second_get_off : ℝ)
    (third_get_on : ℝ) (third_get_off : ℝ) :
  initial_students = 21 →
  first_get_on = 7.5 → first_get_off = 2 → 
  second_get_on = 1.2 → second_get_off = 5.3 →
  third_get_on = 11 → third_get_off = 4.8 →
  (initial_students + (first_get_on - first_get_off) +
   (second_get_on - second_get_off) +
   (third_get_on - third_get_off)) = 28.6 := by
  intros
  sorry

end NUMINAMATH_GPT_students_on_bus_l42_4276


namespace NUMINAMATH_GPT_find_z_when_y_is_6_l42_4298

variable {y z : ℚ}

/-- Condition: y^4 varies inversely with √[4]{z}. -/
def inverse_variation (k : ℚ) (y z : ℚ) : Prop :=
  y^4 * z^(1/4) = k

/-- Given constant k based on y = 3 and z = 16. -/
def k_value : ℚ := 162

theorem find_z_when_y_is_6
  (h_inv : inverse_variation k_value 3 16)
  (h_y : y = 6) :
  z = 1 / 4096 := 
sorry

end NUMINAMATH_GPT_find_z_when_y_is_6_l42_4298


namespace NUMINAMATH_GPT_intersection_is_correct_l42_4223

noncomputable def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
noncomputable def B : Set ℝ := {x : ℝ | x < 4}

theorem intersection_is_correct : A ∩ B = {x : ℝ | -2 ≤ x ∧ x < 1} :=
sorry

end NUMINAMATH_GPT_intersection_is_correct_l42_4223


namespace NUMINAMATH_GPT_problem1_l42_4244

theorem problem1 (α : ℝ) (h : Real.tan (π/4 + α) = 2) : Real.sin (2 * α) + Real.cos α ^ 2 = 3 / 2 := 
sorry

end NUMINAMATH_GPT_problem1_l42_4244
