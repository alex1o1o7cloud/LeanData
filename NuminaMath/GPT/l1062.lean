import Mathlib

namespace max_value_f_l1062_106221

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x + 2) / (2*x - 2)

theorem max_value_f (x : ℝ) (h : -4 < x ∧ x < 1) : ∃ y, f y = -1 ∧ (∀ z, f z ≤ f y) :=
by 
  sorry

end max_value_f_l1062_106221


namespace maximum_distance_product_l1062_106258

theorem maximum_distance_product (α : ℝ) (hα : 0 < α ∧ α < π / 2) :
  let ρ1 := 4 * Real.cos α
  let ρ2 := 2 * Real.sin α
  |ρ1 * ρ2| ≤ 4 :=
by
  -- The proof would go here
  sorry

end maximum_distance_product_l1062_106258


namespace find_g_75_l1062_106261

variable (g : ℝ → ℝ)

def prop_1 := ∀ x y : ℝ, x > 0 → y > 0 → g (x * y) = g x / y
def prop_2 := g 50 = 30

theorem find_g_75 (h1 : prop_1 g) (h2 : prop_2 g) : g 75 = 20 :=
by
  sorry

end find_g_75_l1062_106261


namespace sum_of_opposites_is_zero_l1062_106213

theorem sum_of_opposites_is_zero (a b : ℚ) (h : a = -b) : a + b = 0 := 
by sorry

end sum_of_opposites_is_zero_l1062_106213


namespace number_of_Sunzi_books_l1062_106242

theorem number_of_Sunzi_books
    (num_books : ℕ) (total_cost : ℕ)
    (price_Zhuangzi price_Kongzi price_Mengzi price_Laozi price_Sunzi : ℕ)
    (num_Zhuangzi num_Kongzi num_Mengzi num_Laozi num_Sunzi : ℕ) :
  num_books = 300 →
  total_cost = 4500 →
  price_Zhuangzi = 10 →
  price_Kongzi = 20 →
  price_Mengzi = 15 →
  price_Laozi = 30 →
  price_Sunzi = 12 →
  num_Zhuangzi = num_Kongzi →
  num_Sunzi = 4 * num_Laozi + 15 →
  num_Zhuangzi + num_Kongzi + num_Mengzi + num_Laozi + num_Sunzi = num_books →
  price_Zhuangzi * num_Zhuangzi +
  price_Kongzi * num_Kongzi +
  price_Mengzi * num_Mengzi +
  price_Laozi * num_Laozi +
  price_Sunzi * num_Sunzi = total_cost →
  num_Sunzi = 75 :=
by
  intros h_nb h_tc h_pZ h_pK h_pM h_pL h_pS h_nZ h_nS h_books h_cost
  sorry

end number_of_Sunzi_books_l1062_106242


namespace interest_rate_A_l1062_106284

-- Given conditions
variables (Principal : ℝ := 4000)
variables (interestRate_C : ℝ := 11.5 / 100)
variables (gain_B : ℝ := 180)
variables (time : ℝ := 3)
variables (interest_from_C : ℝ := Principal * interestRate_C * time)
variables (interest_to_A : ℝ := interest_from_C - gain_B)

-- The proof goal
theorem interest_rate_A (R : ℝ) : 
  1200 = Principal * (R / 100) * time → 
  R = 10 :=
by
  sorry

end interest_rate_A_l1062_106284


namespace total_trucks_l1062_106287

theorem total_trucks {t : ℕ} (h1 : 2 * t + t = 300) : t = 100 := 
by sorry

end total_trucks_l1062_106287


namespace distance_between_foci_of_ellipse_l1062_106232

theorem distance_between_foci_of_ellipse :
  let c := (5, 2)
  let a := 5
  let b := 2
  2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 21 :=
by
  let c := (5, 2)
  let a := 5
  let b := 2
  show 2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 21
  sorry

end distance_between_foci_of_ellipse_l1062_106232


namespace Peter_vacation_l1062_106247

theorem Peter_vacation
  (A : ℕ) (S : ℕ) (M : ℕ) (T : ℕ)
  (hA : A = 5000)
  (hS : S = 2900)
  (hM : M = 700)
  (hT : T = (A - S) / M) : T = 3 :=
sorry

end Peter_vacation_l1062_106247


namespace option_b_is_incorrect_l1062_106274

theorem option_b_is_incorrect : ¬ (3 + 2 * Real.sqrt 2 = 5 * Real.sqrt 2) := by
  sorry

end option_b_is_incorrect_l1062_106274


namespace monotonicity_and_extrema_l1062_106289

noncomputable def f (x : ℝ) := (2 * x) / (x + 1)

theorem monotonicity_and_extrema :
  (∀ x1 x2 : ℝ, 3 ≤ x1 → x1 < x2 → x2 ≤ 5 → f x1 < f x2) ∧
  (f 3 = 5 / 4) ∧
  (f 5 = 3 / 2) :=
by
  sorry

end monotonicity_and_extrema_l1062_106289


namespace average_time_correct_l1062_106294

-- Define the times for each runner
def y_time : ℕ := 58
def z_time : ℕ := 26
def w_time : ℕ := 2 * z_time

-- Define the number of runners
def num_runners : ℕ := 3

-- Calculate the summed time of all runners
def total_time : ℕ := y_time + z_time + w_time

-- Calculate the average time
def average_time : ℚ := total_time / num_runners

-- Statement of the proof problem
theorem average_time_correct : average_time = 45.33 := by
  -- The proof would go here
  sorry

end average_time_correct_l1062_106294


namespace arrival_time_difference_l1062_106212

-- Define the times in minutes, with 600 representing 10:00 AM.
def my_watch_time_planned := 600
def my_watch_fast := 5
def my_watch_slow := 10

def friend_watch_time_planned := 600
def friend_watch_fast := 5

-- Calculate actual arrival times.
def my_actual_arrival_time := my_watch_time_planned - my_watch_fast + my_watch_slow
def friend_actual_arrival_time := friend_watch_time_planned - friend_watch_fast

-- Prove the arrival times and difference.
theorem arrival_time_difference :
  friend_actual_arrival_time < my_actual_arrival_time ∧
  my_actual_arrival_time - friend_actual_arrival_time = 20 :=
by
  -- Proof terms can be filled in later.
  sorry

end arrival_time_difference_l1062_106212


namespace arithmetic_geometric_condition_l1062_106276

-- Define the arithmetic sequence
noncomputable def arithmetic_seq (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n-1) * d

-- Define the sum of the first n terms of the arithmetic sequence
noncomputable def sum_arith_seq (a₁ d n : ℕ) : ℕ := n * a₁ + (n * (n-1) / 2) * d

-- Given conditions and required proofs
theorem arithmetic_geometric_condition {d a₁ : ℕ} (h : d ≠ 0) (S₃ : sum_arith_seq a₁ d 3 = 9)
  (geometric_seq : (arithmetic_seq a₁ d 5)^2 = (arithmetic_seq a₁ d 3) * (arithmetic_seq a₁ d 8)) :
  d = 1 ∧ ∀ n, sum_arith_seq 2 1 n = (n^2 + 3 * n) / 2 :=
by
  sorry

end arithmetic_geometric_condition_l1062_106276


namespace unique_solution_for_log_problem_l1062_106224

noncomputable def log_problem (x : ℝ) :=
  let a := Real.log (x / 2 - 1) / Real.log (x - 11 / 4).sqrt
  let b := 2 * Real.log (x - 11 / 4) / Real.log (x / 2 - 1 / 4)
  let c := Real.log (x / 2 - 1 / 4) / (2 * Real.log (x / 2 - 1))
  a * b * c = 2 ∧ (a = b ∧ c = a + 1)

theorem unique_solution_for_log_problem :
  ∃! x, log_problem x = true := sorry

end unique_solution_for_log_problem_l1062_106224


namespace integer_solutions_l1062_106270

theorem integer_solutions (a b c : ℤ) (h₁ : 1 < a) 
    (h₂ : a < b) (h₃ : b < c) 
    (h₄ : (a-1) * (b-1) * (c-1) ∣ a * b * c - 1) :
    (a = 3 ∧ b = 5 ∧ c = 15) 
    ∨ (a = 2 ∧ b = 4 ∧ c = 8) :=
by sorry

end integer_solutions_l1062_106270


namespace factorial_div_eq_l1062_106299

-- Define the factorial function.
def fact (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * fact (n - 1)

-- State the theorem for the given mathematical problem.
theorem factorial_div_eq : (fact 10) / ((fact 7) * (fact 3)) = 120 := by
  sorry

end factorial_div_eq_l1062_106299


namespace interest_rate_l1062_106211

-- Define the sum of money
def P : ℝ := 1800

-- Define the time period in years
def T : ℝ := 2

-- Define the difference in interests
def interest_difference : ℝ := 18

-- Define the relationship between simple interest, compound interest, and the interest rate
theorem interest_rate (R : ℝ) 
  (h1 : SI = P * R * T / 100)
  (h2 : CI = P * (1 + R/100)^2 - P)
  (h3 : CI - SI = interest_difference) :
  R = 10 :=
by
  sorry

end interest_rate_l1062_106211


namespace determine_ω_and_φ_l1062_106216

noncomputable def f (x : ℝ) (ω φ : ℝ) := 2 * Real.sin (ω * x + φ)
def smallest_positive_period (f : ℝ → ℝ) (T : ℝ) := (∀ x, f (x + T) = f x) ∧ (∀ ε > 0, ε < T → ∃ d > 0, d < T ∧ ∀ m n : ℤ, m ≠ n → f (m * d) ≠ f (n * d))

theorem determine_ω_and_φ :
  ∃ ω φ : ℝ,
    (0 < ω) ∧
    (|φ| < Real.pi / 2) ∧
    (smallest_positive_period (f ω φ) Real.pi) ∧
    (f 0 ω φ = Real.sqrt 3) ∧
    (ω = 2 ∧ φ = Real.pi / 3) :=
by
  sorry

end determine_ω_and_φ_l1062_106216


namespace total_distance_hiked_l1062_106259

theorem total_distance_hiked
  (a b c d e : ℕ)
  (h1 : a + b + c = 34)
  (h2 : b + c = 24)
  (h3 : c + d + e = 40)
  (h4 : a + c + e = 38)
  (h5 : d = 14) :
  a + b + c + d + e = 48 :=
by
  sorry

end total_distance_hiked_l1062_106259


namespace sum_of_consecutive_integers_product_l1062_106273

noncomputable def consecutive_integers_sum (n m k : ℤ) : ℤ :=
  n + m + k

theorem sum_of_consecutive_integers_product (n m k : ℤ)
  (h1 : n = m - 1)
  (h2 : k = m + 1)
  (h3 : n * m * k = 990) :
  consecutive_integers_sum n m k = 30 :=
by
  sorry

end sum_of_consecutive_integers_product_l1062_106273


namespace gcm_less_than_90_l1062_106278

theorem gcm_less_than_90 (a b : ℕ) (h1 : a = 8) (h2 : b = 12) : 
  ∃ x : ℕ, x < 90 ∧ ∀ y : ℕ, y < 90 → (a ∣ y) ∧ (b ∣ y) → y ≤ x → x = 72 :=
sorry

end gcm_less_than_90_l1062_106278


namespace total_time_assignment_l1062_106233

-- Define the time taken for each part
def time_first_part : ℕ := 25
def time_second_part : ℕ := 2 * time_first_part
def time_third_part : ℕ := 45

-- Define the total time taken for the assignment
def total_time : ℕ := time_first_part + time_second_part + time_third_part

-- The theorem stating that the total time is 120 minutes
theorem total_time_assignment : total_time = 120 := by
  sorry

end total_time_assignment_l1062_106233


namespace simplify_expression_l1062_106246

theorem simplify_expression : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by {
  sorry
}

end simplify_expression_l1062_106246


namespace mod_exp_l1062_106255

theorem mod_exp (n : ℕ) : (5^303) % 11 = 4 :=
  by sorry

end mod_exp_l1062_106255


namespace range_of_m_l1062_106268

theorem range_of_m (m : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + m * x + 2 * m - 3 < 0) ↔ 2 ≤ m ∧ m ≤ 6 := 
by
  sorry

end range_of_m_l1062_106268


namespace equal_segments_l1062_106237

-- Given a triangle ABC and D as the foot of the bisector from B
variables (A B C D E F : Point) (ABC : Triangle A B C) (Dfoot : BisectorFoot B A C D) 

-- Given that the circumcircles of triangles ABD and BCD intersect sides AB and BC at E and F respectively
variables (circABD : Circumcircle A B D) (circBCD : Circumcircle B C D)
variables (intersectAB : Intersect circABD A B E) (intersectBC : Intersect circBCD B C F)

-- The theorem to prove that AE = CF
theorem equal_segments : AE = CF :=
by
  sorry

end equal_segments_l1062_106237


namespace harry_change_l1062_106249

theorem harry_change (a : ℕ) :
  (∃ k : ℕ, a = 50 * k + 2 ∧ a < 100) ∧ (∃ m : ℕ, a = 5 * m + 4 ∧ a < 100) →
  a = 52 :=
by sorry

end harry_change_l1062_106249


namespace quadratic_has_real_roots_range_l1062_106260

noncomputable def has_real_roots (k : ℝ) : Prop :=
  let a := k
  let b := 2
  let c := -1
  b^2 - 4 * a * c ≥ 0

theorem quadratic_has_real_roots_range (k : ℝ) :
  has_real_roots k ↔ k ≥ -1 ∧ k ≠ 0 := by
sorry

end quadratic_has_real_roots_range_l1062_106260


namespace highest_elevation_l1062_106250

-- Define the function for elevation as per the conditions
def elevation (t : ℝ) : ℝ := 200 * t - 20 * t^2

-- Prove that the highest elevation reached is 500 meters
theorem highest_elevation : (exists t : ℝ, elevation t = 500) ∧ (∀ t : ℝ, elevation t ≤ 500) := sorry

end highest_elevation_l1062_106250


namespace standard_equation_of_circle_l1062_106254

-- Definitions based on problem conditions
def center : ℝ × ℝ := (-1, 2)
def radius : ℝ := 2

-- Lean statement of the problem
theorem standard_equation_of_circle :
  ∀ x y : ℝ, (x - (-1))^2 + (y - 2)^2 = radius ^ 2 ↔ (x + 1)^2 + (y - 2)^2 = 4 :=
by sorry

end standard_equation_of_circle_l1062_106254


namespace perpendicular_condition_sufficient_but_not_necessary_l1062_106228

theorem perpendicular_condition_sufficient_but_not_necessary (m : ℝ) (h : m = -1) :
  (∀ x y : ℝ, mx + (2 * m - 1) * y + 1 = 0 ∧ 3 * x + m * y + 2 = 0) → (m = 0 ∨ m = -1) → (m = 0 ∨ m = -1) :=
by
  intro h1 h2
  sorry

end perpendicular_condition_sufficient_but_not_necessary_l1062_106228


namespace range_of_a_for_quadratic_eq_l1062_106225

theorem range_of_a_for_quadratic_eq (a : ℝ) (h : ∀ x : ℝ, ax^2 = (x+1)*(x-1)) : a ≠ 1 :=
by
  sorry

end range_of_a_for_quadratic_eq_l1062_106225


namespace complex_point_quadrant_l1062_106217

theorem complex_point_quadrant 
  (i : Complex) 
  (h_i_unit : i = Complex.I) : 
  (Complex.re ((i - 3) / (1 + i)) < 0) ∧ (Complex.im ((i - 3) / (1 + i)) > 0) :=
by {
  sorry
}

end complex_point_quadrant_l1062_106217


namespace parallelogram_area_leq_half_triangle_area_l1062_106248

-- Definition of a triangle and a parallelogram inside it.
structure Triangle (α : Type) [LinearOrderedField α] :=
(A B C : α × α)

structure Parallelogram (α : Type) [LinearOrderedField α] :=
(P Q R S : α × α)

-- Function to calculate the area of a triangle
def triangle_area {α : Type} [LinearOrderedField α] (T : Triangle α) : α :=
-- Placeholder for the actual area calculation formula
sorry

-- Function to calculate the area of a parallelogram
def parallelogram_area {α : Type} [LinearOrderedField α] (P : Parallelogram α) : α :=
-- Placeholder for the actual area calculation formula
sorry

-- Statement of the problem
theorem parallelogram_area_leq_half_triangle_area {α : Type} [LinearOrderedField α]
(T : Triangle α) (P : Parallelogram α) (inside : P.P.1 < T.A.1 ∧ P.P.2 < T.C.1) : 
  parallelogram_area P ≤ 1 / 2 * triangle_area T :=
sorry

end parallelogram_area_leq_half_triangle_area_l1062_106248


namespace find_ab_l1062_106271

theorem find_ab (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_area_9 : (1/2) * (12 / a) * (12 / b) = 9) : 
  a * b = 8 := 
by 
  sorry

end find_ab_l1062_106271


namespace p_n_divisible_by_5_l1062_106227

noncomputable def p_n (n : ℕ) : ℕ := 1^n + 2^n + 3^n + 4^n

theorem p_n_divisible_by_5 (n : ℕ) (h : n ≠ 0) : p_n n % 5 = 0 ↔ n % 4 ≠ 0 := by
  sorry

end p_n_divisible_by_5_l1062_106227


namespace floor_eq_l1062_106200

theorem floor_eq (r : ℝ) (h : ⌊r⌋ + r = 12.4) : r = 6.4 := by
  sorry

end floor_eq_l1062_106200


namespace square_of_distance_is_82_l1062_106219

noncomputable def square_distance_from_B_to_center (a b : ℝ) : ℝ := a^2 + b^2

theorem square_of_distance_is_82
  (a b : ℝ)
  (r : ℝ := 11)
  (ha : a^2 + (b + 7)^2 = r^2)
  (hc : (a + 3)^2 + b^2 = r^2) :
  square_distance_from_B_to_center a b = 82 := by
  -- Proof steps omitted
  sorry

end square_of_distance_is_82_l1062_106219


namespace greatest_product_sum_300_l1062_106256

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l1062_106256


namespace purely_imaginary_complex_number_l1062_106295

theorem purely_imaginary_complex_number (a : ℝ) :
  (∃ b : ℝ, (a^2 - 3 * a + 2) = 0 ∧ a ≠ 1) → a = 2 :=
by
  sorry

end purely_imaginary_complex_number_l1062_106295


namespace common_difference_is_3_l1062_106240

variable {a : ℕ → ℝ}
variable {d : ℝ}

-- Conditions
def is_arithmetic (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def condition1 (a : ℕ → ℝ) : Prop := 
  a 3 + a 11 = 24

def condition2 (a : ℕ → ℝ) : Prop := 
  a 4 = 3

theorem common_difference_is_3 (h_arith : is_arithmetic a d) (h1 : condition1 a) (h2 : condition2 a) : 
  d = 3 := 
sorry

end common_difference_is_3_l1062_106240


namespace hide_and_seek_l1062_106244

variables (A B V G D : Prop)

-- Conditions
def condition1 : Prop := A → (B ∧ ¬V)
def condition2 : Prop := B → (G ∨ D)
def condition3 : Prop := ¬V → (¬B ∧ ¬D)
def condition4 : Prop := ¬A → (B ∧ ¬G)

-- Problem statement:
theorem hide_and_seek :
  condition1 A B V →
  condition2 B G D →
  condition3 V B D →
  condition4 A B G →
  (B ∧ V ∧ D) :=
by
  intros h1 h2 h3 h4
  -- Proof would normally go here
  sorry

end hide_and_seek_l1062_106244


namespace distribution_scheme_count_l1062_106215

noncomputable def NumberOfDistributionSchemes : Nat :=
  let plumbers := 5
  let residences := 4
  Nat.choose plumbers (residences - 1) * Nat.factorial residences

theorem distribution_scheme_count :
  NumberOfDistributionSchemes = 240 :=
by
  sorry

end distribution_scheme_count_l1062_106215


namespace product_consecutive_even_div_48_l1062_106272

theorem product_consecutive_even_div_48 (k : ℤ) : 
  (2 * k) * (2 * k + 2) * (2 * k + 4) % 48 = 0 :=
by
  sorry

end product_consecutive_even_div_48_l1062_106272


namespace symmetric_point_y_axis_l1062_106201

theorem symmetric_point_y_axis (A B : ℝ × ℝ) (hA : A = (2, 5)) (h_symm : B = (-A.1, A.2)) :
  B = (-2, 5) :=
sorry

end symmetric_point_y_axis_l1062_106201


namespace range_of_a_if_monotonic_l1062_106209

theorem range_of_a_if_monotonic :
  (∀ x : ℝ, 1 < x ∧ x < 2 → 3 * a * x^2 - 2 * x + 1 ≥ 0) → a > 1 / 3 :=
by
  sorry

end range_of_a_if_monotonic_l1062_106209


namespace max_colors_404_max_colors_406_l1062_106285

theorem max_colors_404 (n k : ℕ) (h1 : n = 404) 
  (h2 : ∃ (houses : ℕ → ℕ), (∀ c : ℕ, ∃ i : ℕ, (∀ j : ℕ, j < 100 → houses (i + j) = c) 
  ∧ ∀ c' : ℕ, c' ≠ c → (∃ j : ℕ, j < 100 → houses (i + j) ≠ c'))) : 
  k ≤ 202 :=
sorry

theorem max_colors_406 (n k : ℕ) (h1 : n = 406) 
  (h2 : ∃ (houses : ℕ → ℕ), (∀ c : ℕ, ∃ i : ℕ, (∀ j : ℕ, j < 100 → houses (i + j) = c) 
  ∧ ∀ c' : ℕ, c' ≠ c → (∃ j : ℕ, j < 100 → houses (i + j) ≠ c'))) : 
  k ≤ 202 :=
sorry

end max_colors_404_max_colors_406_l1062_106285


namespace maximize_pasture_area_l1062_106230

theorem maximize_pasture_area
  (barn_length fence_cost budget : ℕ)
  (barn_length_eq : barn_length = 400)
  (fence_cost_eq : fence_cost = 5)
  (budget_eq : budget = 1500) :
  ∃ x y : ℕ, y = 150 ∧
  x > 0 ∧
  2 * x + y = budget / fence_cost ∧
  y = barn_length - 2 * x ∧
  (x * y) = (75 * 150) :=
by
  sorry

end maximize_pasture_area_l1062_106230


namespace determine_b_l1062_106279

noncomputable def f (x b : ℝ) : ℝ := 1 / (3 * x + b)

noncomputable def f_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem determine_b (b : ℝ) :
  (∀ x : ℝ, f (f_inv x) b = x) -> b = 3 :=
by
  intro h
  sorry

end determine_b_l1062_106279


namespace division_of_fractions_l1062_106241

theorem division_of_fractions : (1 / 6) / (1 / 3) = 1 / 2 :=
by
  sorry

end division_of_fractions_l1062_106241


namespace carl_teaches_periods_l1062_106290

theorem carl_teaches_periods (cards_per_student : ℕ) (students_per_class : ℕ) (pack_cost : ℕ) (amount_spent : ℕ) (cards_per_pack : ℕ) :
  cards_per_student = 10 →
  students_per_class = 30 →
  pack_cost = 3 →
  amount_spent = 108 →
  cards_per_pack = 50 →
  (amount_spent / pack_cost) * cards_per_pack / (cards_per_student * students_per_class) = 6 :=
by
  intros hc hs hp ha hpkg
  /- proof steps would go here -/
  sorry

end carl_teaches_periods_l1062_106290


namespace intersection_A_B_l1062_106292

def A : Set ℝ := {y | ∃ x : ℝ, y = x ^ (1 / 3)}
def B : Set ℝ := {x | x > 1}

theorem intersection_A_B :
  A ∩ B = {x | x > 1} :=
sorry

end intersection_A_B_l1062_106292


namespace hyperbola_product_slopes_constant_l1062_106243

theorem hyperbola_product_slopes_constant (a b x0 y0 : ℝ) (h_a : a > 0) (h_b : b > 0) (hP : (x0 / a) ^ 2 - (y0 / b) ^ 2 = 1) (h_diff_a1_a2 : x0 ≠ a ∧ x0 ≠ -a) :
  (y0 / (x0 + a)) * (y0 / (x0 - a)) = b^2 / a^2 :=
by sorry

end hyperbola_product_slopes_constant_l1062_106243


namespace valentines_given_l1062_106281

theorem valentines_given (original current given : ℕ) (h1 : original = 58) (h2 : current = 16) (h3 : given = original - current) : given = 42 := by
  sorry

end valentines_given_l1062_106281


namespace sum_of_reciprocals_l1062_106296

theorem sum_of_reciprocals (x y : ℝ) (h₁ : x + y = 16) (h₂ : x * y = 55) :
  (1 / x + 1 / y) = 16 / 55 :=
by
  sorry

end sum_of_reciprocals_l1062_106296


namespace none_of_these_true_l1062_106267

variable (s r p q : ℝ)
variable (hs : s > 0) (hr : r > 0) (hpq : p * q ≠ 0) (h : s * (p * r) > s * (q * r))

theorem none_of_these_true : ¬ (-p > -q) ∧ ¬ (-p > q) ∧ ¬ (1 > -q / p) ∧ ¬ (1 < q / p) :=
by
  -- The hypothetical theorem to be proven would continue here
  sorry

end none_of_these_true_l1062_106267


namespace alpha_value_l1062_106226

-- Define the conditions in Lean
variables (α β γ k : ℝ)

-- Mathematically equivalent problem statements translated to Lean
theorem alpha_value :
  (∀ β γ, α = (k * γ) / β) → -- proportionality condition
  (α = 4) →
  (β = 27) →
  (γ = 3) →
  (∀ β γ, β = -81 → γ = 9 → α = -4) :=
by
  sorry

end alpha_value_l1062_106226


namespace range_of_b_div_c_l1062_106239

theorem range_of_b_div_c (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_condition : b^2 = c^2 + a * c) :
  1 < b / c ∧ b / c < 2 := 
sorry

end range_of_b_div_c_l1062_106239


namespace monotonicity_of_f_sum_of_squares_of_roots_l1062_106288

noncomputable def f (x a : Real) : Real := Real.log x - a * x^2

theorem monotonicity_of_f (a : Real) :
  (a ≤ 0 → ∀ x y : Real, 0 < x → x < y → f x a < f y a) ∧
  (a > 0 → ∀ x y : Real, 0 < x → x < Real.sqrt (1/(2 * a)) → Real.sqrt (1/(2 * a)) < y → f x a < f (Real.sqrt (1/(2 * a))) a ∧ f (Real.sqrt (1/(2 * a))) a > f y a) :=
by sorry

theorem sum_of_squares_of_roots (a x1 x2 : Real) (h1 : f x1 a = 0) (h2 : f x2 a = 0) (h3 : x1 ≠ x2) :
  x1^2 + x2^2 > 2 * Real.exp 1 :=
by sorry

end monotonicity_of_f_sum_of_squares_of_roots_l1062_106288


namespace video_game_map_width_l1062_106210

theorem video_game_map_width (volume length height : ℝ) (h1 : volume = 50)
                            (h2 : length = 5) (h3 : height = 2) :
  ∃ width : ℝ, volume = length * width * height ∧ width = 5 :=
by
  sorry

end video_game_map_width_l1062_106210


namespace min_cubes_needed_proof_l1062_106214

noncomputable def min_cubes_needed_to_form_30_digit_number : ℕ :=
  sorry

theorem min_cubes_needed_proof : min_cubes_needed_to_form_30_digit_number = 50 :=
  sorry

end min_cubes_needed_proof_l1062_106214


namespace bridge_length_proof_l1062_106206

noncomputable def train_length : ℝ := 100
noncomputable def time_to_cross_bridge : ℝ := 49.9960003199744
noncomputable def train_speed_kmph : ℝ := 18
noncomputable def conversion_factor : ℝ := 1000 / 3600
noncomputable def train_speed_mps : ℝ := train_speed_kmph * conversion_factor
noncomputable def total_distance : ℝ := train_speed_mps * time_to_cross_bridge
noncomputable def bridge_length : ℝ := total_distance - train_length

theorem bridge_length_proof : bridge_length = 149.980001599872 := 
by 
  sorry

end bridge_length_proof_l1062_106206


namespace expected_winnings_is_minus_half_l1062_106286

-- Define the given condition in Lean
noncomputable def prob_win_side_1 : ℚ := 1 / 4
noncomputable def prob_win_side_2 : ℚ := 1 / 4
noncomputable def prob_lose_side_3 : ℚ := 1 / 3
noncomputable def prob_no_change_side_4 : ℚ := 1 / 6

noncomputable def win_amount_side_1 : ℚ := 2
noncomputable def win_amount_side_2 : ℚ := 4
noncomputable def lose_amount_side_3 : ℚ := -6
noncomputable def no_change_amount_side_4 : ℚ := 0

-- Define the expected value function
noncomputable def expected_winnings : ℚ :=
  (prob_win_side_1 * win_amount_side_1) +
  (prob_win_side_2 * win_amount_side_2) +
  (prob_lose_side_3 * lose_amount_side_3) +
  (prob_no_change_side_4 * no_change_amount_side_4)

-- Statement to prove
theorem expected_winnings_is_minus_half : expected_winnings = -1 / 2 := 
by
  sorry

end expected_winnings_is_minus_half_l1062_106286


namespace nina_money_l1062_106263

variable (C : ℝ)

theorem nina_money (h1: 6 * C = 8 * (C - 1.15)) : 6 * C = 27.6 := by
  have h2: C = 4.6 := sorry
  rw [h2]
  norm_num
  done

end nina_money_l1062_106263


namespace algebraic_sum_of_coefficients_l1062_106257

open Nat

theorem algebraic_sum_of_coefficients
  (u : ℕ → ℤ)
  (h1 : u 1 = 5)
  (hrec : ∀ n : ℕ, n > 0 → u (n + 1) - u n = 3 + 4 * (n - 1)) :
  (∃ P : ℕ → ℤ, (∀ n, u n = P n) ∧ (P 1 + P 0 = 5)) :=
sorry

end algebraic_sum_of_coefficients_l1062_106257


namespace integer_equality_condition_l1062_106291

theorem integer_equality_condition
  (x y z : ℤ)
  (h : x * (x - y) + y * (y - z) + z * (z - x) = 0) :
  x = y ∧ y = z :=
sorry

end integer_equality_condition_l1062_106291


namespace coefficients_verification_l1062_106298

theorem coefficients_verification :
  let a0 := -3
  let a1 := -13 -- Not required as part of the proof but shown for completeness
  let a2 := 6
  let a3 := 0 -- Filler value to ensure there is a6 value
  let a4 := 0 -- Filler value to ensure there is a6 value
  let a5 := 0 -- Filler value to ensure there is a6 value
  let a6 := 0 -- Filler value to ensure there is a6 value
  (1 + 2*x) * (x - 2)^5 = a0 + a1 * (1 - x) + a2 * (1 - x)^2 + a3 * (1 - x)^3 + a4 * (1 - x)^4 + a5 * (1 - x)^5 + a6 * (1 - x)^6 ->
  a0 = -3 ∧
  a0 + a1 + a2 + a3 + a4 + a5 + a6 = -32 :=
by
  intro a0 a1 a2 a3 a4 a5 a6 h
  exact ⟨rfl, sorry⟩

end coefficients_verification_l1062_106298


namespace four_digit_multiples_of_13_and_7_l1062_106262

theorem four_digit_multiples_of_13_and_7 : 
  (∃ n : ℕ, 
    (∀ k : ℕ, 1000 ≤ k ∧ k < 10000 ∧ k % 91 = 0 → k = 1001 + 91 * (n - 11)) 
    ∧ n - 11 + 1 = 99) :=
by
  sorry

end four_digit_multiples_of_13_and_7_l1062_106262


namespace jerry_painting_hours_l1062_106245

-- Define the variables and conditions
def time_painting (P : ℕ) : ℕ := P
def time_counter (P : ℕ) : ℕ := 3 * P
def time_lawn : ℕ := 6
def hourly_rate : ℕ := 15
def total_paid : ℕ := 570

-- Hypothesize that the total hours spent leads to the total payment
def total_hours (P : ℕ) : ℕ := time_painting P + time_counter P + time_lawn

-- Prove that the solution for P matches the conditions
theorem jerry_painting_hours (P : ℕ) 
  (h1 : hourly_rate * total_hours P = total_paid) : 
  P = 8 :=
by
  sorry

end jerry_painting_hours_l1062_106245


namespace find_number_of_two_dollar_pairs_l1062_106220

noncomputable def pairs_of_two_dollars (x y z : ℕ) : Prop :=
  x + y + z = 15 ∧ 2 * x + 4 * y + 5 * z = 38 ∧ x >= 1 ∧ y >= 1 ∧ z >= 1

theorem find_number_of_two_dollar_pairs (x y z : ℕ) 
  (h1 : x + y + z = 15) 
  (h2 : 2 * x + 4 * y + 5 * z = 38) 
  (hx : x >= 1) 
  (hy : y >= 1) 
  (hz : z >= 1) :
  pairs_of_two_dollars x y z → x = 12 :=
by
  intros
  sorry

end find_number_of_two_dollar_pairs_l1062_106220


namespace dog_has_fewer_lives_than_cat_l1062_106282

noncomputable def cat_lives : ℕ := 9
noncomputable def mouse_lives : ℕ := 13
noncomputable def dog_lives : ℕ := mouse_lives - 7
noncomputable def dog_less_lives : ℕ := cat_lives - dog_lives

theorem dog_has_fewer_lives_than_cat : dog_less_lives = 3 := by
  sorry

end dog_has_fewer_lives_than_cat_l1062_106282


namespace sum_of_squares_not_square_l1062_106265

theorem sum_of_squares_not_square (a : ℕ) : 
  ¬ ∃ b : ℕ, (a - 1)^2 + a^2 + (a + 1)^2 = b^2 := 
by {
  sorry
}

end sum_of_squares_not_square_l1062_106265


namespace wendy_percentage_accounting_related_jobs_l1062_106264

noncomputable def wendy_accountant_years : ℝ := 25.5
noncomputable def wendy_accounting_manager_years : ℝ := 15.5 -- Including 6 months as 0.5 years
noncomputable def wendy_financial_consultant_years : ℝ := 10.25 -- Including 3 months as 0.25 years
noncomputable def wendy_tax_advisor_years : ℝ := 4
noncomputable def wendy_lifespan : ℝ := 80

theorem wendy_percentage_accounting_related_jobs :
  ((wendy_accountant_years + wendy_accounting_manager_years + wendy_financial_consultant_years + wendy_tax_advisor_years) / wendy_lifespan) * 100 = 69.0625 :=
by
  sorry

end wendy_percentage_accounting_related_jobs_l1062_106264


namespace num_people_watched_last_week_l1062_106253

variable (s f t : ℕ)
variable (h1 : s = 80)
variable (h2 : f = s - 20)
variable (h3 : t = s + 15)
variable (total_last_week total_this_week : ℕ)
variable (h4 : total_this_week = f + s + t)
variable (h5 : total_this_week = total_last_week + 35)

theorem num_people_watched_last_week :
  total_last_week = 200 := sorry

end num_people_watched_last_week_l1062_106253


namespace simon_fraction_of_alvin_l1062_106203

theorem simon_fraction_of_alvin (alvin_age simon_age : ℕ) (h_alvin : alvin_age = 30)
  (h_simon : simon_age = 10) (h_fraction : ∃ f : ℚ, simon_age + 5 = f * (alvin_age + 5)) :
  ∃ f : ℚ, f = 3 / 7 := by
  sorry

end simon_fraction_of_alvin_l1062_106203


namespace common_roots_cubic_polynomials_l1062_106205

theorem common_roots_cubic_polynomials (a b : ℝ) :
  (∃ r s : ℝ, r ≠ s ∧ (r^3 + a * r^2 + 17 * r + 10 = 0) ∧ (s^3 + a * s^2 + 17 * s + 10 = 0) ∧ 
               (r^3 + b * r^2 + 20 * r + 12 = 0) ∧ (s^3 + b * s^2 + 20 * s + 12 = 0)) →
  (a, b) = (-6, -7) :=
by sorry

end common_roots_cubic_polynomials_l1062_106205


namespace prism_faces_l1062_106251

-- Define the conditions of the problem
def prism (E : ℕ) : Prop :=
  ∃ (L : ℕ), 3 * L = E

-- Define the main proof statement
theorem prism_faces (E : ℕ) (hE : prism E) : E = 27 → 2 + E / 3 = 11 :=
by
  sorry -- Proof is not required

end prism_faces_l1062_106251


namespace ellipse_foci_distance_l1062_106218

noncomputable def center : ℝ×ℝ := (6, 3)
noncomputable def semi_major_axis_length : ℝ := 6
noncomputable def semi_minor_axis_length : ℝ := 3
noncomputable def distance_between_foci : ℝ :=
  let a := semi_major_axis_length
  let b := semi_minor_axis_length
  let c := Real.sqrt (a^2 - b^2)
  2 * c

theorem ellipse_foci_distance :
  distance_between_foci = 6 * Real.sqrt 3 := by
  sorry

end ellipse_foci_distance_l1062_106218


namespace sin_C_of_right_triangle_l1062_106229

theorem sin_C_of_right_triangle (A B C: ℝ) (sinA: ℝ) (sinB: ℝ) (sinC: ℝ) :
  (sinA = 8/17) →
  (sinB = 1) →
  (A + B + C = π) →
  (B = π / 2) →
  (sinC = 15/17) :=
  
by
  intro h_sinA h_sinB h_triangle h_B
  sorry -- Proof is not required

end sin_C_of_right_triangle_l1062_106229


namespace depth_of_well_l1062_106204

theorem depth_of_well (d : ℝ) (t1 t2 : ℝ)
  (h1 : d = 15 * t1^2)
  (h2 : t2 = d / 1100)
  (h3 : t1 + t2 = 9.5) :
  d = 870.25 := 
sorry

end depth_of_well_l1062_106204


namespace mixed_number_division_l1062_106222

theorem mixed_number_division :
  (4 + 2 / 3 + 5 + 1 / 4) / (3 + 1 / 2 - 2 + 3 / 5) = 11 + 1 / 54 :=
by
  sorry

end mixed_number_division_l1062_106222


namespace binomial_coeff_sum_l1062_106269

theorem binomial_coeff_sum 
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℝ)
  (h1 : (1 - 2 * 0 : ℝ)^(7) = a_0 + a_1 * 0 + a_2 * 0^2 + a_3 * 0^3 + a_4 * 0^4 + a_5 * 0^5 + a_6 * 0^6 + a_7 * 0^7)
  (h2 : (1 - 2 * 1 : ℝ)^(7) = a_0 + a_1 * 1 + a_2 * 1^2 + a_3 * 1^3 + a_4 * 1^4 + a_5 * 1^5 + a_6 * 1^6 + a_7 * 1^7) :
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = -2 := 
sorry

end binomial_coeff_sum_l1062_106269


namespace prob_one_AB_stuck_prob_at_least_two_stuck_l1062_106252

-- Define the events and their probabilities.
def prob_traffic_I := 1 / 10
def prob_no_traffic_I := 9 / 10
def prob_traffic_II := 3 / 5
def prob_no_traffic_II := 2 / 5

-- Define the events
def event_A := prob_traffic_I
def not_event_A := prob_no_traffic_I
def event_B := prob_traffic_I
def not_event_B := prob_no_traffic_I
def event_C := prob_traffic_II
def not_event_C := prob_no_traffic_II

-- Define the probabilities as required in the problem
def prob_exactly_one_of_A_B_in_traffic :=
  event_A * not_event_B + not_event_A * event_B

def prob_at_least_two_in_traffic :=
  event_A * event_B * not_event_C +
  event_A * not_event_B * event_C +
  not_event_A * event_B * event_C +
  event_A * event_B * event_C

-- Proofs (statements only)
theorem prob_one_AB_stuck :
  prob_exactly_one_of_A_B_in_traffic = 9 / 50 := sorry

theorem prob_at_least_two_stuck :
  prob_at_least_two_in_traffic = 59 / 500 := sorry

end prob_one_AB_stuck_prob_at_least_two_stuck_l1062_106252


namespace white_marbles_multiple_of_8_l1062_106236

-- Definitions based on conditions
def blue_marbles : ℕ := 16
def num_groups : ℕ := 8

-- Stating the problem
theorem white_marbles_multiple_of_8 (white_marbles : ℕ) :
  (blue_marbles + white_marbles) % num_groups = 0 → white_marbles % num_groups = 0 :=
by
  sorry

end white_marbles_multiple_of_8_l1062_106236


namespace tile_D_is_IV_l1062_106234

structure Tile :=
  (top : ℕ) (right : ℕ) (bottom : ℕ) (left : ℕ)

def Tile_I : Tile := ⟨3, 1, 4, 2⟩
def Tile_II : Tile := ⟨2, 3, 1, 5⟩
def Tile_III : Tile := ⟨4, 0, 3, 1⟩
def Tile_IV : Tile := ⟨5, 4, 2, 0⟩

def is_tile_D (t : Tile) : Prop :=
  t.left = 0 ∧ t.top = 5

theorem tile_D_is_IV : is_tile_D Tile_IV :=
  by
    -- skip proof here
    sorry

end tile_D_is_IV_l1062_106234


namespace cube_volume_correct_l1062_106283

-- Define the height and base dimensions of the pyramid
def pyramid_height := 15
def pyramid_base_length := 12
def pyramid_base_width := 8

-- Define the side length of the cube-shaped box
def cube_side_length := max pyramid_height pyramid_base_length

-- Define the volume of the cube-shaped box
def cube_volume := cube_side_length ^ 3

-- Theorem statement: the volume of the smallest cube-shaped box that can fit the pyramid is 3375 cubic inches
theorem cube_volume_correct : cube_volume = 3375 := by
  sorry

end cube_volume_correct_l1062_106283


namespace solution_y_amount_l1062_106277

theorem solution_y_amount :
  ∀ (y : ℝ) (volume_x volume_y : ℝ),
    volume_x = 200 ∧
    volume_y = y ∧
    10 / 100 * volume_x = 20 ∧
    30 / 100 * volume_y = 0.3 * y ∧
    (20 + 0.3 * y) / (volume_x + y) = 0.25 →
    y = 600 :=
by 
  intros y volume_x volume_y
  intros H
  sorry

end solution_y_amount_l1062_106277


namespace triangle_number_placement_l1062_106238

theorem triangle_number_placement
  (A B C D E F : ℕ)
  (h1 : A + B + C = 6)
  (h2 : D = 5)
  (h3 : E = 6)
  (h4 : D + E + F = 14)
  (h5 : B = 3) : 
  (A = 1 ∧ B = 3 ∧ C = 2 ∧ D = 5 ∧ E = 6 ∧ F = 4) :=
by {
  sorry
}

end triangle_number_placement_l1062_106238


namespace interval_a_b_l1062_106293

noncomputable def f (x : ℝ) : ℝ := |Real.log (x - 1)|

theorem interval_a_b (a b : ℝ) (x1 x2 : ℝ) (h1 : 1 < x1) (h2 : x1 < x2) (h3 : x2 < b) (h4 : f x1 > f x2) :
  a < 2 := 
sorry

end interval_a_b_l1062_106293


namespace pawns_left_l1062_106202

-- Definitions of the initial conditions
def initial_pawns : ℕ := 8
def kennedy_lost_pawns : ℕ := 4
def riley_lost_pawns : ℕ := 1

-- Definition of the total pawns left function
def total_pawns_left (initial_pawns kennedy_lost_pawns riley_lost_pawns : ℕ) : ℕ :=
  (initial_pawns - kennedy_lost_pawns) + (initial_pawns - riley_lost_pawns)

-- The statement to prove
theorem pawns_left : total_pawns_left initial_pawns kennedy_lost_pawns riley_lost_pawns = 11 := by
  -- Proof omitted
  sorry

end pawns_left_l1062_106202


namespace john_needs_to_add_empty_cans_l1062_106207

theorem john_needs_to_add_empty_cans :
  ∀ (num_full_cans : ℕ) (weight_per_full_can total_weight weight_per_empty_can required_weight : ℕ),
  num_full_cans = 6 →
  weight_per_full_can = 14 →
  total_weight = 88 →
  weight_per_empty_can = 2 →
  required_weight = total_weight - (num_full_cans * weight_per_full_can) →
  required_weight / weight_per_empty_can = 2 :=
by
  intros
  sorry

end john_needs_to_add_empty_cans_l1062_106207


namespace exists_non_degenerate_triangle_l1062_106231

theorem exists_non_degenerate_triangle
  (l : Fin 7 → ℝ)
  (h_ordered : ∀ i j, i ≤ j → l i ≤ l j)
  (h_bounds : ∀ i, 1 ≤ l i ∧ l i ≤ 12) :
  ∃ i j k : Fin 7, i < j ∧ j < k ∧ l i + l j > l k ∧ l j + l k > l i ∧ l k + l i > l j := 
sorry

end exists_non_degenerate_triangle_l1062_106231


namespace percentage_of_180_equation_l1062_106266

theorem percentage_of_180_equation (P : ℝ) (h : (P / 100) * 180 - (1 / 3) * ((P / 100) * 180) = 36) : P = 30 :=
sorry

end percentage_of_180_equation_l1062_106266


namespace perfect_rectangle_squares_l1062_106235

theorem perfect_rectangle_squares (squares : Finset ℕ) 
  (h₁ : 9 ∈ squares) 
  (h₂ : 2 ∈ squares) 
  (h₃ : squares.card = 9) 
  (h₄ : ∀ x ∈ squares, ∃ y ∈ squares, x ≠ y ∧ (gcd x y = 1)) :
  squares = {2, 5, 7, 9, 16, 25, 28, 33, 36} := 
sorry

end perfect_rectangle_squares_l1062_106235


namespace how_much_money_per_tshirt_l1062_106275

def money_made_per_tshirt 
  (total_money_tshirts : ℕ) 
  (number_tshirts : ℕ) : Prop :=
  total_money_tshirts / number_tshirts = 62

theorem how_much_money_per_tshirt 
  (total_money_tshirts : ℕ) 
  (number_tshirts : ℕ) 
  (h1 : total_money_tshirts = 11346) 
  (h2 : number_tshirts = 183) : 
  money_made_per_tshirt total_money_tshirts number_tshirts := 
by 
  sorry

end how_much_money_per_tshirt_l1062_106275


namespace determine_OP_l1062_106280

variable (a b c d : ℝ)
variable (O A B C D P : ℝ)
variable (p : ℝ)

def OnLine (O A B C D P : ℝ) : Prop := O < A ∧ A < B ∧ B < C ∧ C < D ∧ B < P ∧ P < C

theorem determine_OP (h : OnLine O A B C D P) 
(hAP : P - A = p - a) 
(hPD : D - P = d - p) 
(hBP : P - B = p - b) 
(hPC : C - P = c - p) 
(hAP_PD_BP_PC : (p - a) / (d - p) = (p - b) / (c - p)) :
  p = (a * c - b * d) / (a - b + c - d) :=
sorry

end determine_OP_l1062_106280


namespace range_cos_A_l1062_106223

theorem range_cos_A {A B C : ℚ} (h : 1 / (Real.tan B) + 1 / (Real.tan C) = 1 / (Real.tan A))
  (h_non_neg_A: 0 ≤ A) (h_less_pi_A: A ≤ π): 
  (Real.cos A ∈ Set.Ico (2 / 3) 1) :=
sorry

end range_cos_A_l1062_106223


namespace determine_p_range_l1062_106208

theorem determine_p_range :
  ∀ (p : ℝ), (∃ f : ℝ → ℝ, ∀ x : ℝ, f x = (x + 9 / 8) * (x + 9 / 8) ∧ (f x) = (8*x^2 + 18*x + 4*p)/8 ) →
  2.5 < p ∧ p < 2.6 :=
by
  sorry

end determine_p_range_l1062_106208


namespace David_squats_l1062_106297

theorem David_squats (h1: ∀ d z: ℕ, d = 3 * 58) : d = 174 :=
by
  sorry

end David_squats_l1062_106297
