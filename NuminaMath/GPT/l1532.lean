import Mathlib

namespace NUMINAMATH_GPT_line_through_parabola_intersects_vertex_l1532_153218

theorem line_through_parabola_intersects_vertex (y x k : ℝ) :
  (y = 6 * x ∨ 6 * x - 5 * y - 24 = 0) ∧ 
  (∃ P Q : ℝ × ℝ, (P.1)^2 = 4 * P.2 ∧ (Q.1)^2 = 4 * Q.2 ∧ 
   (P = (0, 0) ∨ Q = (0, 0)) ∧ 
   (y = 6 * x ∨ 6 * x - 5 * y - 24 = 0)) := sorry

end NUMINAMATH_GPT_line_through_parabola_intersects_vertex_l1532_153218


namespace NUMINAMATH_GPT_table_relationship_l1532_153241

theorem table_relationship (x y : ℕ) (h : (x, y) ∈ [(1, 1), (2, 8), (3, 27), (4, 64), (5, 125)]) : y = x^3 :=
sorry

end NUMINAMATH_GPT_table_relationship_l1532_153241


namespace NUMINAMATH_GPT_Amy_bought_tomato_soup_l1532_153234

-- Conditions
variables (chicken_soup_cans total_soups : ℕ)
variable (Amy_bought_soups : total_soups = 9)
variable (Amy_bought_chicken_soup : chicken_soup_cans = 6)

-- Question: How many cans of tomato soup did she buy?
def cans_of_tomato_soup (chicken_soup_cans total_soups : ℕ) : ℕ :=
  total_soups - chicken_soup_cans

-- Theorem: Prove that the number of cans of tomato soup Amy bought is 3
theorem Amy_bought_tomato_soup : 
  cans_of_tomato_soup chicken_soup_cans total_soups = 3 :=
by
  rw [Amy_bought_soups, Amy_bought_chicken_soup]
  -- The steps for the proof would follow here
  sorry

end NUMINAMATH_GPT_Amy_bought_tomato_soup_l1532_153234


namespace NUMINAMATH_GPT_eval_f_nested_l1532_153230

noncomputable def f (x : ℝ) : ℝ :=
if h : x < 0 then x + 1 else x ^ 2

theorem eval_f_nested : f (f (-2)) = 0 := by
  sorry

end NUMINAMATH_GPT_eval_f_nested_l1532_153230


namespace NUMINAMATH_GPT_product_profit_equation_l1532_153235

theorem product_profit_equation (purchase_price selling_price : ℝ) 
                                (initial_units units_decrease_per_dollar_increase : ℝ)
                                (profit : ℝ)
                                (hx : purchase_price = 35)
                                (hy : selling_price = 40)
                                (hz : initial_units = 200)
                                (hs : units_decrease_per_dollar_increase = 5)
                                (hp : profit = 1870) :
  ∃ x : ℝ, (x + (selling_price - purchase_price)) * (initial_units - units_decrease_per_dollar_increase * x) = profit :=
by { sorry }

end NUMINAMATH_GPT_product_profit_equation_l1532_153235


namespace NUMINAMATH_GPT_smallest_angle_of_quadrilateral_l1532_153285

theorem smallest_angle_of_quadrilateral 
  (x : ℝ) 
  (h1 : x + 2 * x + 3 * x + 4 * x = 360) : 
  x = 36 :=
by
  sorry

end NUMINAMATH_GPT_smallest_angle_of_quadrilateral_l1532_153285


namespace NUMINAMATH_GPT_diameter_of_circle_with_inscribed_right_triangle_l1532_153262

theorem diameter_of_circle_with_inscribed_right_triangle (a b c : ℕ) (h1 : a = 6) (h2 : b = 8) (right_triangle : a^2 + b^2 = c^2) : c = 10 :=
by
  subst h1
  subst h2
  simp at right_triangle
  sorry

end NUMINAMATH_GPT_diameter_of_circle_with_inscribed_right_triangle_l1532_153262


namespace NUMINAMATH_GPT_no_other_distinct_prime_products_l1532_153258

theorem no_other_distinct_prime_products :
  ∀ (q1 q2 q3 : Nat), 
  Prime q1 ∧ Prime q2 ∧ Prime q3 ∧ q1 ≠ q2 ∧ q2 ≠ q3 ∧ q1 ≠ q3 ∧ q1 * q2 * q3 ≠ 17 * 11 * 23 → 
  q1 + q2 + q3 ≠ 51 :=
by
  intros q1 q2 q3 h
  sorry

end NUMINAMATH_GPT_no_other_distinct_prime_products_l1532_153258


namespace NUMINAMATH_GPT_solution_l1532_153274

noncomputable def f (x : ℝ) := 
  10 / (Real.sqrt (x - 5) - 10) + 
  2 / (Real.sqrt (x - 5) - 5) + 
  9 / (Real.sqrt (x - 5) + 5) + 
  18 / (Real.sqrt (x - 5) + 10)

theorem solution : 
  f (1230 / 121) = 0 := sorry

end NUMINAMATH_GPT_solution_l1532_153274


namespace NUMINAMATH_GPT_find_x_value_l1532_153243

noncomputable def x_value (x y z : ℝ) : Prop :=
  (26 = (z + x) / 2) ∧
  (z = 52 - x) ∧
  (52 - x = (26 + y) / 2) ∧
  (y = 78 - 2 * x) ∧
  (78 - 2 * x = (8 + (52 - x)) / 2) ∧
  (x = 32)

theorem find_x_value : ∃ x y z : ℝ, x_value x y z :=
by
  use 32  -- x
  use 14  -- y derived from 78 - 2x where x = 32 leads to y = 14
  use 20  -- z derived from 52 - x where x = 32 leads to z = 20
  unfold x_value
  simp
  sorry

end NUMINAMATH_GPT_find_x_value_l1532_153243


namespace NUMINAMATH_GPT_bertha_daughters_and_granddaughters_have_no_daughters_l1532_153282

def total_daughters_and_granddaughters (daughters granddaughters : Nat) : Nat :=
daughters + granddaughters

def no_daughters (bertha_daughters bertha_granddaughters : Nat) : Nat :=
bertha_daughters + bertha_granddaughters

theorem bertha_daughters_and_granddaughters_have_no_daughters :
  (bertha_daughters : Nat) →
  (daughters_with_6_daughters : Nat) →
  (granddaughters : Nat) →
  (total_daughters_and_granddaughters bertha_daughters granddaughters = 30) →
  bertha_daughters = 6 →
  granddaughters = 6 * daughters_with_6_daughters →
  no_daughters (bertha_daughters - daughters_with_6_daughters) granddaughters = 26 :=
by
  intros bertha_daughters daughters_with_6_daughters granddaughters h_total h_bertha h_granddaughters
  sorry

end NUMINAMATH_GPT_bertha_daughters_and_granddaughters_have_no_daughters_l1532_153282


namespace NUMINAMATH_GPT_intersection_complement_eq_l1532_153286

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 3}
def B : Set ℝ := {x | x < 2}

-- Define the complement of B in ℝ
def complement_B : Set ℝ := {x | x ≥ 2}

-- Define the intersection of A and complement of B
def intersection : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

-- The theorem to be proved
theorem intersection_complement_eq : (A ∩ complement_B) = intersection :=
sorry

end NUMINAMATH_GPT_intersection_complement_eq_l1532_153286


namespace NUMINAMATH_GPT_count_distinct_four_digit_numbers_divisible_by_3_ending_in_45_l1532_153231

/--
Prove that the total number of distinct four-digit numbers that end with 45 and 
are divisible by 3 is 27.
-/
theorem count_distinct_four_digit_numbers_divisible_by_3_ending_in_45 :
  ∃ n : ℕ, n = 27 ∧ 
  ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) → (0 ≤ b ∧ b ≤ 9) → 
  (∃ k : ℕ, a + b + 9 = 3 * k) → 
  (10 * (10 * a + b) + 45) = 1000 * a + 100 * b + 45 → 
  1000 * a + 100 * b + 45 = n := sorry

end NUMINAMATH_GPT_count_distinct_four_digit_numbers_divisible_by_3_ending_in_45_l1532_153231


namespace NUMINAMATH_GPT_combined_distance_all_birds_two_seasons_l1532_153224

-- Definition of the given conditions
def number_of_birds : Nat := 20
def distance_jim_to_disney : Nat := 50
def distance_disney_to_london : Nat := 60

-- The conclusion we need to prove
theorem combined_distance_all_birds_two_seasons :
  (distance_jim_to_disney + distance_disney_to_london) * number_of_birds = 2200 :=
by
  sorry

end NUMINAMATH_GPT_combined_distance_all_birds_two_seasons_l1532_153224


namespace NUMINAMATH_GPT_MariaTotalPaid_l1532_153272

-- Define a structure to hold the conditions
structure DiscountProblem where
  discount_rate : ℝ
  discount_amount : ℝ

-- Define the given discount problem specific to Maria
def MariaDiscountProblem : DiscountProblem :=
  { discount_rate := 0.25, discount_amount := 40 }

-- Define our goal: proving the total amount paid by Maria
theorem MariaTotalPaid (p : DiscountProblem) (h₀ : p = MariaDiscountProblem) :
  let original_price := p.discount_amount / p.discount_rate
  let total_paid := original_price - p.discount_amount
  total_paid = 120 :=
by
  sorry

end NUMINAMATH_GPT_MariaTotalPaid_l1532_153272


namespace NUMINAMATH_GPT_largest_n_unique_k_l1532_153223

-- Defining the main theorem statement
theorem largest_n_unique_k :
  ∃ (n : ℕ), (n = 63) ∧ (∃! (k : ℤ), (9 / 17 : ℚ) < (n : ℚ) / ((n + k) : ℚ) ∧ (n : ℚ) / ((n + k) : ℚ) < (8 / 15 : ℚ)) :=
sorry

end NUMINAMATH_GPT_largest_n_unique_k_l1532_153223


namespace NUMINAMATH_GPT_least_xy_l1532_153256

theorem least_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 9) : xy = 108 := by
  sorry

end NUMINAMATH_GPT_least_xy_l1532_153256


namespace NUMINAMATH_GPT_total_distance_traveled_l1532_153293

theorem total_distance_traveled (x : ℕ) (d_1 d_2 d_3 d_4 d_5 d_6 : ℕ) 
  (h1 : d_1 = 60 / x) 
  (h2 : d_2 = 60 / (x + 3)) 
  (h3 : d_3 = 60 / (x + 6)) 
  (h4 : d_4 = 60 / (x + 9)) 
  (h5 : d_5 = 60 / (x + 12)) 
  (h6 : d_6 = 60 / (x + 15)) 
  (hx1 : x ∣ 60) 
  (hx2 : (x + 3) ∣ 60) 
  (hx3 : (x + 6) ∣ 60) 
  (hx4 : (x + 9) ∣ 60) 
  (hx5 : (x + 12) ∣ 60) 
  (hx6 : (x + 15) ∣ 60) :
  d_1 + d_2 + d_3 + d_4 + d_5 + d_6 = 39 := 
sorry

end NUMINAMATH_GPT_total_distance_traveled_l1532_153293


namespace NUMINAMATH_GPT_infinite_polynomial_pairs_l1532_153237

open Polynomial

theorem infinite_polynomial_pairs :
  ∀ n : ℕ, ∃ (fn gn : ℤ[X]), fn^2 - (X^4 - 2 * X) * gn^2 = 1 :=
sorry

end NUMINAMATH_GPT_infinite_polynomial_pairs_l1532_153237


namespace NUMINAMATH_GPT_find_m_l1532_153297

theorem find_m (a : ℕ → ℝ) (m : ℝ)
  (h1 : (∀ (x : ℝ), x^2 + m * x - 8 = 0 → x = a 2 ∨ x = a 8))
  (h2 : a 4 + a 6 = a 5 ^ 2 + 1) :
  m = -2 :=
sorry

end NUMINAMATH_GPT_find_m_l1532_153297


namespace NUMINAMATH_GPT_sum_of_primes_between_20_and_30_l1532_153200

/-- Define what it means to be a prime number -/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the predicate for numbers being between 20 and 30 -/
def between_20_and_30 (n : ℕ) : Prop :=
  20 < n ∧ n < 30

/-- List of prime numbers between 20 and 30 -/
def prime_list : List ℕ := [23, 29]

/-- The sum of elements in the prime list -/
def prime_sum : ℕ := prime_list.sum

/-- Prove that the sum of prime numbers between 20 and 30 is 52 -/
theorem sum_of_primes_between_20_and_30 :
  prime_sum = 52 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_primes_between_20_and_30_l1532_153200


namespace NUMINAMATH_GPT_arithmetic_sequence_terms_l1532_153201

theorem arithmetic_sequence_terms (a : ℕ → ℝ) (n : ℕ) (S : ℝ) 
  (h1 : a 0 + a 1 + a 2 = 4) 
  (h2 : a (n-3) + a (n-2) + a (n-1) = 7) 
  (h3 : (n * (a 0 + a (n-1)) / 2) = 22) : 
  n = 12 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_terms_l1532_153201


namespace NUMINAMATH_GPT_constant_function_of_functional_equation_l1532_153213

theorem constant_function_of_functional_equation {f : ℝ → ℝ} (h : ∀ x y : ℝ, 0 < x → 0 < y → f (x + y) = f (x^2 + y^2)) : ∃ c : ℝ, ∀ x : ℝ, 0 < x → f x = c := 
sorry

end NUMINAMATH_GPT_constant_function_of_functional_equation_l1532_153213


namespace NUMINAMATH_GPT_louie_mistakes_l1532_153211

theorem louie_mistakes (total_items : ℕ) (percentage_correct : ℕ) 
  (h1 : total_items = 25) 
  (h2 : percentage_correct = 80) : 
  total_items - ((percentage_correct / 100) * total_items) = 5 := 
by
  sorry

end NUMINAMATH_GPT_louie_mistakes_l1532_153211


namespace NUMINAMATH_GPT_simplify_sqrt_450_l1532_153284

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end NUMINAMATH_GPT_simplify_sqrt_450_l1532_153284


namespace NUMINAMATH_GPT_arithmetic_to_geometric_seq_l1532_153212

theorem arithmetic_to_geometric_seq
  (d a : ℕ) 
  (h1 : d ≠ 0) 
  (a_n : ℕ → ℕ)
  (h2 : ∀ n, a_n n = a + (n - 1) * d)
  (h3 : (a + 2 * d) * (a + 2 * d) = a * (a + 8 * d))
  : (a_n 2 + a_n 4 + a_n 10) / (a_n 1 + a_n 3 + a_n 9) = 16 / 13 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_to_geometric_seq_l1532_153212


namespace NUMINAMATH_GPT_max_volume_of_box_l1532_153267

theorem max_volume_of_box (sheetside : ℝ) (cutside : ℝ) (volume : ℝ) 
  (h1 : sheetside = 6) 
  (h2 : ∀ (x : ℝ), 0 < x ∧ x < (sheetside / 2) → volume = x * (sheetside - 2 * x)^2) : 
  cutside = 1 :=
by
  sorry

end NUMINAMATH_GPT_max_volume_of_box_l1532_153267


namespace NUMINAMATH_GPT_least_perimeter_of_triangle_l1532_153236

theorem least_perimeter_of_triangle (c : ℕ) (h1 : 24 + 51 > c) (h2 : c > 27) : 24 + 51 + c = 103 :=
by
  sorry

end NUMINAMATH_GPT_least_perimeter_of_triangle_l1532_153236


namespace NUMINAMATH_GPT_intersection_A_B_l1532_153248

-- Define sets A and B according to the conditions provided
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | 0 < x ∧ x < 3}

-- Define the theorem stating the intersection of A and B
theorem intersection_A_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1532_153248


namespace NUMINAMATH_GPT_harrys_morning_routine_time_l1532_153222

theorem harrys_morning_routine_time :
  (15 + 20 + 25 + 2 * 15 = 90) :=
by
  sorry

end NUMINAMATH_GPT_harrys_morning_routine_time_l1532_153222


namespace NUMINAMATH_GPT_evaluate_expression_l1532_153246

theorem evaluate_expression :
  -2 ^ 2005 + (-2) ^ 2006 + 2 ^ 2007 - 2 ^ 2008 = 2 ^ 2005 :=
by
  -- The following proof is left as an exercise.
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1532_153246


namespace NUMINAMATH_GPT_find_f_zero_l1532_153232

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_zero (a : ℝ) (h1 : ∀ x : ℝ, f (x - a) = x^3 + 1)
  (h2 : ∀ x : ℝ, f x + f (2 - x) = 2) : 
  f 0 = 0 :=
sorry

end NUMINAMATH_GPT_find_f_zero_l1532_153232


namespace NUMINAMATH_GPT_function_inverse_necessary_not_sufficient_l1532_153288

theorem function_inverse_necessary_not_sufficient (f : ℝ → ℝ) :
  (∃ g : ℝ → ℝ, ∀ x : ℝ, g (f x) = x ∧ f (g x) = x) →
  ¬ (∀ (x y : ℝ), x < y → f x < f y) :=
by
  sorry

end NUMINAMATH_GPT_function_inverse_necessary_not_sufficient_l1532_153288


namespace NUMINAMATH_GPT_part1_part2_part3_l1532_153278

variable {a b c : ℝ}

-- Part (1)
theorem part1 (a b c : ℝ) : a * (b - c) ^ 2 + b * (c - a) ^ 2 + c * (a - b) ^ 2 + 4 * a * b * c > a ^ 3 + b ^ 3 + c ^ 3 :=
sorry

-- Part (2)
theorem part2 (a b c : ℝ) : 2 * a ^ 2 * b ^ 2 + 2 * b ^ 2 * c ^ 2 + 2 * c ^ 2 * a ^ 2 > a ^ 4 + b ^ 4 + c ^ 4 :=
sorry

-- Part (3)
theorem part3 (a b c : ℝ) : 2 * a * b + 2 * b * c + 2 * c * a > a ^ 2 + b ^ 2 + c ^ 2 :=
sorry

end NUMINAMATH_GPT_part1_part2_part3_l1532_153278


namespace NUMINAMATH_GPT_number_of_voters_in_election_l1532_153265

theorem number_of_voters_in_election
  (total_membership : ℕ)
  (votes_cast : ℕ)
  (winning_percentage_cast : ℚ)
  (percentage_of_total : ℚ)
  (h_total : total_membership = 1600)
  (h_winning_percentage : winning_percentage_cast = 0.60)
  (h_percentage_of_total : percentage_of_total = 0.196875)
  (h_votes : winning_percentage_cast * votes_cast = percentage_of_total * total_membership) :
  votes_cast = 525 :=
by
  sorry

end NUMINAMATH_GPT_number_of_voters_in_election_l1532_153265


namespace NUMINAMATH_GPT_turnip_bag_weight_l1532_153208

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]
def total_weight : ℕ := 106
def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

theorem turnip_bag_weight :
  ∃ T : ℕ, T ∈ bag_weights ∧ (is_divisible_by_three (total_weight - T)) := sorry

end NUMINAMATH_GPT_turnip_bag_weight_l1532_153208


namespace NUMINAMATH_GPT_range_of_k_l1532_153219

theorem range_of_k (x y k : ℝ) (h1 : 2 * x - 3 * y = 5) (h2 : 2 * x - y = k) (h3 : x > y) : k > -5 :=
sorry

end NUMINAMATH_GPT_range_of_k_l1532_153219


namespace NUMINAMATH_GPT_measure_of_angle_ABC_l1532_153207

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

end NUMINAMATH_GPT_measure_of_angle_ABC_l1532_153207


namespace NUMINAMATH_GPT_part_a_l1532_153244

theorem part_a (a b : ℝ) (h1 : a^2 + b^2 = 1) (h2 : a + b = 1) : a * b = 0 := 
by 
  sorry

end NUMINAMATH_GPT_part_a_l1532_153244


namespace NUMINAMATH_GPT_ian_says_1306_l1532_153226

noncomputable def number_i_say := 4 * (4 * (4 * (4 * (4 * (4 * (4 * (4 * 1 - 2) - 2) - 2) - 2) - 2) - 2) - 2) - 2

theorem ian_says_1306 (n : ℕ) : 1 ≤ n ∧ n ≤ 2000 → n = 1306 :=
by sorry

end NUMINAMATH_GPT_ian_says_1306_l1532_153226


namespace NUMINAMATH_GPT_find_number_l1532_153280

-- Define the number x and the condition as a theorem to be proven.
theorem find_number (x : ℝ) (h : (1/3) * x - 5 = 10) : x = 45 :=
sorry

end NUMINAMATH_GPT_find_number_l1532_153280


namespace NUMINAMATH_GPT_prove_healthy_diet_multiple_l1532_153266

variable (rum_on_pancakes rum_earlier rum_after_pancakes : ℝ)
variable (healthy_multiple : ℝ)

-- Definitions from conditions
def Sally_gave_rum_on_pancakes : Prop := rum_on_pancakes = 10
def Don_had_rum_earlier : Prop := rum_earlier = 12
def Don_can_have_rum_after_pancakes : Prop := rum_after_pancakes = 8

-- Concluding multiple for healthy diet
def healthy_diet_multiple : Prop := healthy_multiple = (rum_on_pancakes + rum_after_pancakes - rum_earlier) / rum_on_pancakes

theorem prove_healthy_diet_multiple :
  Sally_gave_rum_on_pancakes rum_on_pancakes →
  Don_had_rum_earlier rum_earlier →
  Don_can_have_rum_after_pancakes rum_after_pancakes →
  healthy_diet_multiple rum_on_pancakes rum_earlier rum_after_pancakes healthy_multiple →
  healthy_multiple = 0.8 := 
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_prove_healthy_diet_multiple_l1532_153266


namespace NUMINAMATH_GPT_train_speed_is_28_l1532_153279

-- Define the given conditions
def train_length : ℕ := 1200
def overbridge_length : ℕ := 200
def crossing_time : ℕ := 50

-- Define the total distance
def total_distance := train_length + overbridge_length

-- Define the speed calculation function
def speed (distance time : ℕ) : ℕ := 
  distance / time

-- State the theorem to be proven
theorem train_speed_is_28 : speed total_distance crossing_time = 28 := 
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_train_speed_is_28_l1532_153279


namespace NUMINAMATH_GPT_original_number_l1532_153253

theorem original_number (x : ℝ) (h : 1.35 * x = 680) : x = 503.70 :=
sorry

end NUMINAMATH_GPT_original_number_l1532_153253


namespace NUMINAMATH_GPT_shauna_lowest_score_l1532_153216

theorem shauna_lowest_score :
  ∀ (scores : List ℕ) (score1 score2 score3 : ℕ), 
    scores = [score1, score2, score3] → 
    score1 = 82 →
    score2 = 88 →
    score3 = 93 →
    (∃ (s4 s5 : ℕ), s4 + s5 = 162 ∧ s4 ≤ 100 ∧ s5 ≤ 100) ∧
    score1 + score2 + score3 + s4 + s5 = 425 →
    min s4 s5 = 62 := 
by 
  sorry

end NUMINAMATH_GPT_shauna_lowest_score_l1532_153216


namespace NUMINAMATH_GPT_tan_alpha_value_cos2_minus_sin2_l1532_153251

variable (α : Real) 

axiom is_internal_angle (angle : Real) : angle ∈ Set.Ico 0 Real.pi 

axiom sin_cos_sum (α : Real) : α ∈ Set.Ico 0 Real.pi → Real.sin α + Real.cos α = 1 / 5

theorem tan_alpha_value (h : α ∈ Set.Ico 0 Real.pi) : Real.tan α = -4 / 3 := by 
  sorry

theorem cos2_minus_sin2 (h : Real.tan α = -4 / 3) : 1 / (Real.cos α^2 - Real.sin α^2) = -25 / 7 := by 
  sorry

end NUMINAMATH_GPT_tan_alpha_value_cos2_minus_sin2_l1532_153251


namespace NUMINAMATH_GPT_temperature_decrease_l1532_153275

-- Define the conditions
def temperature_rise (temp_increase: ℤ) : ℤ := temp_increase

-- Define the claim to be proved
theorem temperature_decrease (temp_decrease: ℤ) : temperature_rise 3 = 3 → temperature_rise (-6) = -6 :=
by
  sorry

end NUMINAMATH_GPT_temperature_decrease_l1532_153275


namespace NUMINAMATH_GPT_avg_first_3_is_6_l1532_153205

theorem avg_first_3_is_6 (A B C D : ℝ) (X : ℝ)
  (h1 : (A + B + C) / 3 = X)
  (h2 : (B + C + D) / 3 = 5)
  (h3 : A + D = 11)
  (h4 : D = 4) :
  X = 6 := 
by
  sorry

end NUMINAMATH_GPT_avg_first_3_is_6_l1532_153205


namespace NUMINAMATH_GPT_butterfly_black_dots_l1532_153273

theorem butterfly_black_dots (b f : ℕ) (total_butterflies : b = 397) (total_black_dots : f = 4764) : f / b = 12 :=
by
  sorry

end NUMINAMATH_GPT_butterfly_black_dots_l1532_153273


namespace NUMINAMATH_GPT_problem_l1532_153209

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 10 / Real.log 5
noncomputable def c : ℝ := Real.log 12 / Real.log 6

theorem problem : a > b ∧ b > c := by
  sorry

end NUMINAMATH_GPT_problem_l1532_153209


namespace NUMINAMATH_GPT_parallel_lines_not_coincident_l1532_153291

theorem parallel_lines_not_coincident (a : ℝ) :
  (∀ x y : ℝ, ax + 2 * y + 6 = 0) ∧
  (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0) →
  ¬ ∃ (b : ℝ), ∀ x y : ℝ, ax + 2 * y + b = 0 ∧ x + (a - 1) * y + b = 0 →
  a = -1 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_not_coincident_l1532_153291


namespace NUMINAMATH_GPT_older_brother_allowance_l1532_153263

theorem older_brother_allowance 
  (sum_allowance : ℕ)
  (difference : ℕ)
  (total_sum : sum_allowance = 12000)
  (additional_amount : difference = 1000) :
  ∃ (older_brother_allowance younger_brother_allowance : ℕ), 
    older_brother_allowance = younger_brother_allowance + difference ∧
    younger_brother_allowance + older_brother_allowance = sum_allowance ∧
    older_brother_allowance = 6500 :=
by {
  sorry
}

end NUMINAMATH_GPT_older_brother_allowance_l1532_153263


namespace NUMINAMATH_GPT_radar_placement_and_coverage_area_l1532_153299

theorem radar_placement_and_coverage_area (r : ℝ) (w : ℝ) (n : ℕ) (h_radars : n = 5) (h_radius : r = 13) (h_width : w = 10) :
  let max_dist := 12 / Real.sin (Real.pi / 5)
  let area_ring := (240 * Real.pi) / Real.tan (Real.pi / 5)
  max_dist = 12 / Real.sin (Real.pi / 5) ∧ area_ring = (240 * Real.pi) / Real.tan (Real.pi / 5) :=
by
  sorry

end NUMINAMATH_GPT_radar_placement_and_coverage_area_l1532_153299


namespace NUMINAMATH_GPT_range_of_a_l1532_153225

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

theorem range_of_a (a : ℝ) (h : ∀ x y : ℝ, x ≤ y → f x a ≥ f y a) : a ≤ -3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1532_153225


namespace NUMINAMATH_GPT_special_number_exists_l1532_153221

theorem special_number_exists (a b c d e : ℕ) (h1 : a < b ∧ b < c ∧ c < d ∧ d < e)
    (h2 : a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e) 
    (h_num : a * 10 + b = 13 ∧ c = 4 ∧ d * 10 + e = 52) :
    (10 * a + b) * c = 10 * d + e :=
by
  sorry

end NUMINAMATH_GPT_special_number_exists_l1532_153221


namespace NUMINAMATH_GPT_average_eq_y_value_l1532_153264

theorem average_eq_y_value :
  (y : ℤ) → (h : (15 + 25 + y) / 3 = 20) → y = 20 :=
by
  intro y h
  sorry

end NUMINAMATH_GPT_average_eq_y_value_l1532_153264


namespace NUMINAMATH_GPT_total_exercise_time_l1532_153238

theorem total_exercise_time :
  let javier_minutes_per_day := 50
  let javier_days := 7
  let sanda_minutes_per_day := 90
  let sanda_days := 3
  (javier_minutes_per_day * javier_days + sanda_minutes_per_day * sanda_days) = 620 :=
by
  sorry

end NUMINAMATH_GPT_total_exercise_time_l1532_153238


namespace NUMINAMATH_GPT_min_value_sin_cos_expr_l1532_153210

open Real

theorem min_value_sin_cos_expr (α : ℝ) (hα : 0 < α ∧ α < π / 2) :
  ∃ min_val : ℝ, min_val = 3 * sqrt 2 ∧ ∀ β, (0 < β ∧ β < π / 2) → 
    sin β + cos β + (2 * sqrt 2) / sin (β + π / 4) ≥ min_val :=
by
  sorry

end NUMINAMATH_GPT_min_value_sin_cos_expr_l1532_153210


namespace NUMINAMATH_GPT_translation_correctness_l1532_153202

-- Define the original function
def original_function (x : ℝ) : ℝ := 3 * x + 5

-- Define the translated function
def translated_function (x : ℝ) : ℝ := 3 * x

-- Define the condition for passing through the origin
def passes_through_origin (f : ℝ → ℝ) : Prop := f 0 = 0

-- The theorem to prove the correct translation
theorem translation_correctness : passes_through_origin translated_function := by
  sorry

end NUMINAMATH_GPT_translation_correctness_l1532_153202


namespace NUMINAMATH_GPT_no_solution_system_l1532_153203

theorem no_solution_system :
  ¬ ∃ (x y z : ℝ), (3 * x - 4 * y + z = 10) ∧ (6 * x - 8 * y + 2 * z = 5) ∧ (2 * x - y - z = 4) :=
by {
  sorry
}

end NUMINAMATH_GPT_no_solution_system_l1532_153203


namespace NUMINAMATH_GPT_parabola_focus_l1532_153229

theorem parabola_focus (f : ℝ) :
  (∀ x : ℝ, 2*x^2 = x^2 + (2*x^2 - f)^2 - (2*x^2 - -f)^2) →
  f = -1/8 :=
by sorry

end NUMINAMATH_GPT_parabola_focus_l1532_153229


namespace NUMINAMATH_GPT_lewis_total_earnings_l1532_153298

def Weekly_earnings : ℕ := 92
def Number_of_weeks : ℕ := 5

theorem lewis_total_earnings : Weekly_earnings * Number_of_weeks = 460 := by
  sorry

end NUMINAMATH_GPT_lewis_total_earnings_l1532_153298


namespace NUMINAMATH_GPT_tan_theta_correct_l1532_153289

noncomputable def tan_theta : Real :=
  let θ : Real := sorry
  if h_θ : θ ∈ Set.Ioc 0 (Real.pi / 4) then
    if h : Real.sin θ + Real.cos θ = 17 / 13 then
      Real.tan θ
    else
      0
  else
    0

theorem tan_theta_correct {θ : Real} (h_θ : θ ∈ Set.Ioc 0 (Real.pi / 4))
  (h : Real.sin θ + Real.cos θ = 17 / 13) :
  Real.tan θ = 5 / 12 := sorry

end NUMINAMATH_GPT_tan_theta_correct_l1532_153289


namespace NUMINAMATH_GPT_dot_product_result_l1532_153276

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (1, 1)

theorem dot_product_result : (a.1 * b.1 + a.2 * b.2) = 1 := by
  sorry

end NUMINAMATH_GPT_dot_product_result_l1532_153276


namespace NUMINAMATH_GPT_nina_total_miles_l1532_153270

noncomputable def totalDistance (warmUp firstHillUp firstHillDown firstRecovery 
                                 tempoRun secondHillUp secondHillDown secondRecovery 
                                 fartlek sprintsYards jogsBetweenSprints coolDown : ℝ) 
                                 (mileInYards : ℝ) : ℝ :=
  warmUp + 
  (firstHillUp + firstHillDown + firstRecovery) + 
  tempoRun + 
  (secondHillUp + secondHillDown + secondRecovery) + 
  fartlek + 
  (sprintsYards / mileInYards) + 
  jogsBetweenSprints + 
  coolDown

theorem nina_total_miles : 
  totalDistance 0.25 0.15 0.25 0.15 1.5 0.2 0.35 0.1 1.8 (8 * 50) (8 * 0.2) 0.3 1760 = 5.877 :=
by
  sorry

end NUMINAMATH_GPT_nina_total_miles_l1532_153270


namespace NUMINAMATH_GPT_value_of_expression_l1532_153239

theorem value_of_expression : 2 - (-5) = 7 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1532_153239


namespace NUMINAMATH_GPT_total_distance_covered_l1532_153271

-- Define the basic conditions
def num_marathons : Nat := 15
def miles_per_marathon : Nat := 26
def yards_per_marathon : Nat := 385
def yards_per_mile : Nat := 1760

-- Define the total miles and total yards covered
def total_miles : Nat := num_marathons * miles_per_marathon
def total_yards : Nat := num_marathons * yards_per_marathon

-- Convert excess yards into miles and calculate the remaining yards
def extra_miles : Nat := total_yards / yards_per_mile
def remaining_yards : Nat := total_yards % yards_per_mile

-- Compute the final total distance
def total_distance_miles : Nat := total_miles + extra_miles
def total_distance_yards : Nat := remaining_yards

-- The theorem that needs to be proven
theorem total_distance_covered :
  total_distance_miles = 393 ∧ total_distance_yards = 495 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_covered_l1532_153271


namespace NUMINAMATH_GPT_fencing_required_l1532_153215

theorem fencing_required (L W : ℕ) (hL : L = 10) (hA : L * W = 600) : L + 2 * W = 130 :=
by
  sorry

end NUMINAMATH_GPT_fencing_required_l1532_153215


namespace NUMINAMATH_GPT_fencing_required_l1532_153259

theorem fencing_required (L W : ℕ) (A : ℕ) 
  (hL : L = 20) 
  (hA : A = 680) 
  (hArea : A = L * W) : 
  2 * W + L = 88 := 
by 
  sorry

end NUMINAMATH_GPT_fencing_required_l1532_153259


namespace NUMINAMATH_GPT_find_third_test_score_l1532_153287

-- Definitions of the given conditions
def test_score_1 := 80
def test_score_2 := 70
variable (x : ℕ) -- the unknown third score
def test_score_4 := 100
def average_score (s1 s2 s3 s4 : ℕ) : ℕ := (s1 + s2 + s3 + s4) / 4

-- Theorem stating that given the conditions, the third test score must be 90
theorem find_third_test_score (h : average_score test_score_1 test_score_2 x test_score_4 = 85) : x = 90 :=
by
  sorry

end NUMINAMATH_GPT_find_third_test_score_l1532_153287


namespace NUMINAMATH_GPT_total_customers_l1532_153290

def initial_customers : ℝ := 29.0    -- 29.0 initial customers
def lunch_rush_customers : ℝ := 20.0 -- Adds 20.0 customers during lunch rush
def additional_customers : ℝ := 34.0 -- Adds 34.0 more customers

theorem total_customers : (initial_customers + lunch_rush_customers + additional_customers) = 83.0 :=
by
  sorry

end NUMINAMATH_GPT_total_customers_l1532_153290


namespace NUMINAMATH_GPT_percentage_saved_l1532_153292

noncomputable def calculateSavedPercentage : ℚ :=
  let first_tier_free_tickets := 1
  let second_tier_free_tickets_per_ticket := 2
  let number_of_tickets_purchased := 10
  let total_free_tickets :=
    first_tier_free_tickets +
    (number_of_tickets_purchased - 5) * second_tier_free_tickets_per_ticket
  let total_tickets_received := number_of_tickets_purchased + total_free_tickets
  let free_tickets := total_tickets_received - number_of_tickets_purchased
  (free_tickets / total_tickets_received) * 100

theorem percentage_saved : calculateSavedPercentage = 52.38 :=
by
  sorry

end NUMINAMATH_GPT_percentage_saved_l1532_153292


namespace NUMINAMATH_GPT_square_87_l1532_153269

theorem square_87 : 87^2 = 7569 :=
by
  sorry

end NUMINAMATH_GPT_square_87_l1532_153269


namespace NUMINAMATH_GPT_final_result_l1532_153240

-- Define the number of letters in each name
def letters_in_elida : ℕ := 5
def letters_in_adrianna : ℕ := 2 * letters_in_elida - 2

-- Define the alphabetical positions and their sums for each name
def sum_positions_elida : ℕ := 5 + 12 + 9 + 4 + 1
def sum_positions_adrianna : ℕ := 1 + 4 + 18 + 9 + 1 + 14 + 14 + 1
def sum_positions_belinda : ℕ := 2 + 5 + 12 + 9 + 14 + 4 + 1

-- Define the total sum of alphabetical positions
def total_sum_positions : ℕ := sum_positions_elida + sum_positions_adrianna + sum_positions_belinda

-- Define the average of the total sum
def average_sum_positions : ℕ := total_sum_positions / 3

-- Prove the final result
theorem final_result : (average_sum_positions * 3 - sum_positions_elida) = 109 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_final_result_l1532_153240


namespace NUMINAMATH_GPT_olivia_possible_amount_l1532_153249

theorem olivia_possible_amount (k : ℕ) :
  ∃ k : ℕ, 1 + 79 * k = 1984 :=
by
  -- Prove that there exists a non-negative integer k such that the equation holds
  sorry

end NUMINAMATH_GPT_olivia_possible_amount_l1532_153249


namespace NUMINAMATH_GPT_tom_fruit_bowl_l1532_153228

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

end NUMINAMATH_GPT_tom_fruit_bowl_l1532_153228


namespace NUMINAMATH_GPT_find_a_l1532_153255

noncomputable def A (a : ℝ) : Set ℝ := {-1, a^2 + 1, a^2 - 3}
noncomputable def B (a : ℝ) : Set ℝ := {-4, a - 1, a + 1}

theorem find_a (a : ℝ) (h : A a ∩ B a = {-2}) : a = -1 :=
sorry

end NUMINAMATH_GPT_find_a_l1532_153255


namespace NUMINAMATH_GPT_cone_inscribed_spheres_distance_l1532_153294

noncomputable def distance_between_sphere_centers (R α : ℝ) : ℝ :=
  R * (Real.sqrt 2) * Real.sin (α / 2) / (2 * Real.cos (α / 8) * Real.cos ((45 : ℝ) - α / 8))

theorem cone_inscribed_spheres_distance (R α : ℝ) (h1 : R > 0) (h2 : α > 0) :
  distance_between_sphere_centers R α = R * (Real.sqrt 2) * Real.sin (α / 2) / (2 * Real.cos (α / 8) * Real.cos ((45 : ℝ) - α / 8)) :=
by 
  sorry

end NUMINAMATH_GPT_cone_inscribed_spheres_distance_l1532_153294


namespace NUMINAMATH_GPT_solitaire_game_end_with_one_piece_l1532_153296

theorem solitaire_game_end_with_one_piece (n : ℕ) : 
  ∃ (remaining_pieces : ℕ), 
  remaining_pieces = 1 ↔ n % 3 ≠ 0 :=
sorry

end NUMINAMATH_GPT_solitaire_game_end_with_one_piece_l1532_153296


namespace NUMINAMATH_GPT_find_integer_roots_l1532_153254

open Int Polynomial

def P (x : ℤ) : ℤ := x^3 - 3 * x^2 - 13 * x + 15

theorem find_integer_roots : {x : ℤ | P x = 0} = {-3, 1, 5} := by
  sorry

end NUMINAMATH_GPT_find_integer_roots_l1532_153254


namespace NUMINAMATH_GPT_vasya_numbers_l1532_153204

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 := 
by 
  sorry

end NUMINAMATH_GPT_vasya_numbers_l1532_153204


namespace NUMINAMATH_GPT_max_non_attacking_mammonths_is_20_l1532_153217

def mamonth_attacking_diagonal_count (b: board) (m: mamonth): ℕ := 
    sorry -- define the function to count attacking diagonals of a given mammoth on the board

def max_non_attacking_mamonths_board (b: board) : ℕ :=
    sorry -- function to calculate max non-attacking mammonths given a board setup

theorem max_non_attacking_mammonths_is_20 : 
  ∀ (b : board), (max_non_attacking_mamonths_board b) ≤ 20 :=
by
  sorry

end NUMINAMATH_GPT_max_non_attacking_mammonths_is_20_l1532_153217


namespace NUMINAMATH_GPT_value_of_X_l1532_153257

noncomputable def M : ℕ := 3009 / 3
noncomputable def N : ℕ := (2 * M) / 3
noncomputable def X : ℕ := M - N

theorem value_of_X : X = 335 := by
  sorry

end NUMINAMATH_GPT_value_of_X_l1532_153257


namespace NUMINAMATH_GPT_train_crossing_time_l1532_153214

def train_length : ℕ := 1000
def train_speed_km_per_h : ℕ := 18
def train_speed_m_per_s := train_speed_km_per_h * 1000 / 3600

theorem train_crossing_time :
  train_length / train_speed_m_per_s = 200 := by
sorry

end NUMINAMATH_GPT_train_crossing_time_l1532_153214


namespace NUMINAMATH_GPT_count_divisible_by_five_l1532_153252

theorem count_divisible_by_five : 
  ∃ n : ℕ, (∀ x, 1 ≤ x ∧ x ≤ 1000 → (x % 5 = 0 → (n = 200))) :=
by
  sorry

end NUMINAMATH_GPT_count_divisible_by_five_l1532_153252


namespace NUMINAMATH_GPT_greatest_three_digit_number_l1532_153281

theorem greatest_three_digit_number
  (n : ℕ) (h_3digit : 100 ≤ n ∧ n < 1000) (h_mod7 : n % 7 = 2) (h_mod4 : n % 4 = 1) :
  n = 989 :=
sorry

end NUMINAMATH_GPT_greatest_three_digit_number_l1532_153281


namespace NUMINAMATH_GPT_total_subjects_is_41_l1532_153206

-- Define the number of subjects taken by Monica, Marius, and Millie
def subjects_monica := 10
def subjects_marius := subjects_monica + 4
def subjects_millie := subjects_marius + 3

-- Define the total number of subjects taken by all three
def total_subjects := subjects_monica + subjects_marius + subjects_millie

theorem total_subjects_is_41 : total_subjects = 41 := by
  -- This is where the proof would be, but we only need the statement
  sorry

end NUMINAMATH_GPT_total_subjects_is_41_l1532_153206


namespace NUMINAMATH_GPT_alice_probability_same_color_l1532_153277

def total_ways_to_draw : ℕ := 
  Nat.choose 9 3 * Nat.choose 6 3 * Nat.choose 3 3

def favorable_outcomes_for_alice : ℕ := 
  3 * Nat.choose 6 3 * Nat.choose 3 3

def probability_alice_same_color : ℚ := 
  favorable_outcomes_for_alice / total_ways_to_draw

theorem alice_probability_same_color : probability_alice_same_color = 1 / 28 := 
by
  -- Proof is omitted as per instructions
  sorry

end NUMINAMATH_GPT_alice_probability_same_color_l1532_153277


namespace NUMINAMATH_GPT_Owen_final_turtle_count_l1532_153220

variable (Owen_turtles : ℕ) (Johanna_turtles : ℕ)

def final_turtles (Owen_turtles Johanna_turtles : ℕ) : ℕ :=
  let initial_Owen_turtles := Owen_turtles
  let initial_Johanna_turtles := Owen_turtles - 5
  let Owen_after_month := initial_Owen_turtles * 2
  let Johanna_after_losing_half := initial_Johanna_turtles / 2
  let Owen_after_donation := Owen_after_month + Johanna_after_losing_half
  Owen_after_donation

theorem Owen_final_turtle_count : final_turtles 21 (21 - 5) = 50 :=
by
  sorry

end NUMINAMATH_GPT_Owen_final_turtle_count_l1532_153220


namespace NUMINAMATH_GPT_power_function_value_at_9_l1532_153227

noncomputable def f (x : ℝ) : ℝ := x ^ (1 / 2)

theorem power_function_value_at_9 (h : f 2 = Real.sqrt 2) : f 9 = 3 :=
by sorry

end NUMINAMATH_GPT_power_function_value_at_9_l1532_153227


namespace NUMINAMATH_GPT_trajectory_midpoint_l1532_153250

/-- Let A and B be two moving points on the circle x^2 + y^2 = 4, and AB = 2. 
    The equation of the trajectory of the midpoint M of the line segment AB is x^2 + y^2 = 3. -/
theorem trajectory_midpoint (A B : ℝ × ℝ) (M : ℝ × ℝ)
    (hA : A.1^2 + A.2^2 = 4)
    (hB : B.1^2 + B.2^2 = 4)
    (hAB : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4)
    (hM : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
    M.1^2 + M.2^2 = 3 :=
sorry

end NUMINAMATH_GPT_trajectory_midpoint_l1532_153250


namespace NUMINAMATH_GPT_security_deposit_percentage_l1532_153247

theorem security_deposit_percentage
    (daily_rate : ℝ) (pet_fee : ℝ) (service_fee_rate : ℝ) (days : ℝ) (security_deposit : ℝ)
    (total_cost : ℝ) (expected_percentage : ℝ) :
    daily_rate = 125.0 →
    pet_fee = 100.0 →
    service_fee_rate = 0.20 →
    days = 14 →
    security_deposit = 1110 →
    total_cost = daily_rate * days + pet_fee + (daily_rate * days + pet_fee) * service_fee_rate →
    expected_percentage = (security_deposit / total_cost) * 100 →
    expected_percentage = 50 :=
by
  intros
  sorry

end NUMINAMATH_GPT_security_deposit_percentage_l1532_153247


namespace NUMINAMATH_GPT_solution_set_intersection_l1532_153242

theorem solution_set_intersection (a b : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x - 3 < 0 ↔ -1 < x ∧ x < 3) →
  (∀ x : ℝ, x^2 + x - 6 < 0 ↔ -3 < x ∧ x < 2) →
  (∀ x : ℝ, x^2 + a * x + b < 0 ↔ (-1 < x ∧ x < 2)) →
  a + b = -3 :=
by 
  sorry

end NUMINAMATH_GPT_solution_set_intersection_l1532_153242


namespace NUMINAMATH_GPT_part_a_part_b_l1532_153261

def can_cut_into_equal_dominoes (n : ℕ) : Prop :=
  ∃ horiz_vert_dominoes : ℕ × ℕ,
    n % 2 = 1 ∧
    (n * n - 1) / 2 = horiz_vert_dominoes.1 + horiz_vert_dominoes.2 ∧
    horiz_vert_dominoes.1 = horiz_vert_dominoes.2

theorem part_a : can_cut_into_equal_dominoes 101 :=
by {
  sorry
}

theorem part_b : ¬can_cut_into_equal_dominoes 99 :=
by {
  sorry
}

end NUMINAMATH_GPT_part_a_part_b_l1532_153261


namespace NUMINAMATH_GPT_senate_subcommittee_l1532_153233

/-- 
Proof of the number of ways to form a Senate subcommittee consisting of 7 Republicans
and 2 Democrats from the available 12 Republicans and 6 Democrats.
-/
theorem senate_subcommittee (R D : ℕ) (choose_R choose_D : ℕ) (hR : R = 12) (hD : D = 6) 
  (h_choose_R : choose_R = 7) (h_choose_D : choose_D = 2) : 
  (Nat.choose R choose_R) * (Nat.choose D choose_D) = 11880 := by
  sorry

end NUMINAMATH_GPT_senate_subcommittee_l1532_153233


namespace NUMINAMATH_GPT_reena_loan_l1532_153260

/-- 
  Problem setup:
  Reena took a loan of $1200 at simple interest for a period equal to the rate of interest years. 
  She paid $192 as interest at the end of the loan period.
  We aim to prove that the rate of interest is 4%. 
-/
theorem reena_loan (P : ℝ) (SI : ℝ) (R : ℝ) (N : ℝ) 
  (hP : P = 1200) 
  (hSI : SI = 192) 
  (hN : N = R) 
  (hSI_formula : SI = P * R * N / 100) : 
  R = 4 := 
by 
  sorry

end NUMINAMATH_GPT_reena_loan_l1532_153260


namespace NUMINAMATH_GPT_annie_total_miles_l1532_153283

theorem annie_total_miles (initial_gallons : ℕ) (miles_per_gallon : ℕ)
  (initial_trip_miles : ℕ) (purchased_gallons : ℕ) (final_gallons : ℕ)
  (total_miles : ℕ) :
  initial_gallons = 12 →
  miles_per_gallon = 28 →
  initial_trip_miles = 280 →
  purchased_gallons = 6 →
  final_gallons = 5 →
  total_miles = 364 := by
  sorry

end NUMINAMATH_GPT_annie_total_miles_l1532_153283


namespace NUMINAMATH_GPT_calculate_drift_l1532_153295

def width_of_river : ℕ := 400
def speed_of_boat : ℕ := 10
def time_to_cross : ℕ := 50
def actual_distance_traveled := speed_of_boat * time_to_cross

theorem calculate_drift : actual_distance_traveled - width_of_river = 100 :=
by
  -- width_of_river = 400
  -- speed_of_boat = 10
  -- time_to_cross = 50
  -- actual_distance_traveled = 10 * 50 = 500
  -- expected drift = 500 - 400 = 100
  sorry

end NUMINAMATH_GPT_calculate_drift_l1532_153295


namespace NUMINAMATH_GPT_range_of_u_l1532_153268

variable (a b u : ℝ)

theorem range_of_u (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  (∀ x : ℝ, x > 0 → a^2 + b^2 ≥ x ↔ x ≤ 16) :=
sorry

end NUMINAMATH_GPT_range_of_u_l1532_153268


namespace NUMINAMATH_GPT_complex_number_property_l1532_153245

theorem complex_number_property (i : ℂ) (h : i^2 = -1) : (1 + i)^(20) - (1 - i)^(20) = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_complex_number_property_l1532_153245
