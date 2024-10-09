import Mathlib

namespace f_odd_solve_inequality_l293_29347

noncomputable def f : ℝ → ℝ := sorry

axiom f_additive : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_positive : ∀ x : ℝ, x > 0 → f x > 0

theorem f_odd : ∀ x : ℝ, f (-x) = -f x := 
by
  sorry

theorem solve_inequality : {a : ℝ | f (a-4) + f (2*a+1) < 0} = {a | a < 1} := 
by
  sorry

end f_odd_solve_inequality_l293_29347


namespace largest_divisor_n4_minus_n2_l293_29355

theorem largest_divisor_n4_minus_n2 (n : ℤ) : 12 ∣ (n^4 - n^2) :=
by
  sorry

end largest_divisor_n4_minus_n2_l293_29355


namespace problem_equivalent_l293_29372

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem problem_equivalent (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -10 :=
by
  sorry

end problem_equivalent_l293_29372


namespace average_score_l293_29323

theorem average_score (s1 s2 s3 : ℕ) (n : ℕ) (h1 : s1 = 115) (h2 : s2 = 118) (h3 : s3 = 115) (h4 : n = 3) :
    (s1 + s2 + s3) / n = 116 :=
by
    sorry

end average_score_l293_29323


namespace max_a_b_l293_29389

theorem max_a_b (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_eq : a^2 + b^2 = 1) : a + b ≤ Real.sqrt 2 := sorry

end max_a_b_l293_29389


namespace a6_equals_8_l293_29359

-- Defining Sn as given in the condition
def S (n : ℕ) : ℤ :=
  if n = 0 then 0
  else n^2 - 3*n

-- Defining a_n in terms of the differences stated in the solution
def a (n : ℕ) : ℤ := S n - S (n-1)

-- The problem statement to prove
theorem a6_equals_8 : a 6 = 8 :=
by
  sorry

end a6_equals_8_l293_29359


namespace lines_parallel_if_perpendicular_to_plane_l293_29307

axiom line : Type
axiom plane : Type

-- Definitions of perpendicular and parallel
axiom perp : line → plane → Prop
axiom parallel : line → line → Prop

variables (a b : line) (α : plane)

theorem lines_parallel_if_perpendicular_to_plane (h1 : perp a α) (h2 : perp b α) : parallel a b :=
sorry

end lines_parallel_if_perpendicular_to_plane_l293_29307


namespace find_largest_natural_number_l293_29354

theorem find_largest_natural_number :
  ∃ n : ℕ, (forall m : ℕ, m > n -> m ^ 300 ≥ 3 ^ 500) ∧ n ^ 300 < 3 ^ 500 :=
  sorry

end find_largest_natural_number_l293_29354


namespace mod_exp_sub_l293_29349

theorem mod_exp_sub (a b k : ℕ) (h₁ : a ≡ 6 [MOD 7]) (h₂ : b ≡ 4 [MOD 7]) :
  (a ^ k - b ^ k) % 7 = 2 :=
sorry

end mod_exp_sub_l293_29349


namespace power_mod_eq_five_l293_29311

theorem power_mod_eq_five
  (m : ℕ)
  (h₀ : 0 ≤ m)
  (h₁ : m < 8)
  (h₂ : 13^5 % 8 = m) : m = 5 :=
by 
  sorry

end power_mod_eq_five_l293_29311


namespace boarders_joined_l293_29397

theorem boarders_joined (initial_boarders : ℕ) (initial_day_scholars : ℕ) (final_ratio_num : ℕ) (final_ratio_denom : ℕ) (new_boarders : ℕ)
  (initial_ratio_boarders_to_day_scholars : initial_boarders * 16 = 7 * initial_day_scholars)
  (initial_boarders_eq : initial_boarders = 560)
  (final_ratio : (initial_boarders + new_boarders) * 2 = final_day_scholars)
  (day_scholars_eq : initial_day_scholars = 1280) : 
  new_boarders = 80 := by
  sorry

end boarders_joined_l293_29397


namespace value_of_x_l293_29368

theorem value_of_x (x : ℝ) (h : x = -x) : x = 0 := 
by 
  sorry

end value_of_x_l293_29368


namespace ounces_of_wax_for_car_l293_29318

noncomputable def ounces_wax_for_SUV : ℕ := 4
noncomputable def initial_wax_amount : ℕ := 11
noncomputable def wax_spilled : ℕ := 2
noncomputable def wax_left_after_detailing : ℕ := 2
noncomputable def total_wax_used : ℕ := initial_wax_amount - wax_spilled - wax_left_after_detailing

theorem ounces_of_wax_for_car :
  (initial_wax_amount - wax_spilled - wax_left_after_detailing) - ounces_wax_for_SUV = 3 :=
by
  sorry

end ounces_of_wax_for_car_l293_29318


namespace gcd_x_y_not_8_l293_29303

theorem gcd_x_y_not_8 (x y : ℕ) (hx : x > 0) (hy : y = x^2 + 8) : ¬ ∃ d, d = 8 ∧ d ∣ x ∧ d ∣ y :=
by
  sorry

end gcd_x_y_not_8_l293_29303


namespace smallest_n_gt_15_l293_29380

theorem smallest_n_gt_15 (n : ℕ) : n ≡ 4 [MOD 6] → n ≡ 3 [MOD 7] → n > 15 → n = 52 :=
by
  sorry

end smallest_n_gt_15_l293_29380


namespace largest_divisor_consecutive_odd_l293_29353

theorem largest_divisor_consecutive_odd (m n : ℤ) (h : ∃ k : ℤ, m = 2 * k + 1 ∧ n = 2 * k - 1) :
  ∃ d : ℤ, d = 8 ∧ ∀ m n : ℤ, (∃ k : ℤ, m = 2 * k + 1 ∧ n = 2 * k - 1) → d ∣ (m^2 - n^2) :=
by
  sorry

end largest_divisor_consecutive_odd_l293_29353


namespace max_green_socks_l293_29361

theorem max_green_socks (g y : ℕ) (h_t : g + y ≤ 2000) (h_prob : (g * (g - 1) + y * (y - 1) = (g + y) * (g + y - 1) / 3)) :
  g ≤ 19 := by
  sorry

end max_green_socks_l293_29361


namespace parallelogram_diagonal_length_l293_29381

-- Define a structure to represent a parallelogram
structure Parallelogram :=
  (side_length : ℝ) 
  (diagonal_length : ℝ)
  (perpendicular : Bool)

-- State the theorem about the relationship between the diagonals in a parallelogram
theorem parallelogram_diagonal_length (a b : ℝ) (P : Parallelogram) (h₀ : P.side_length = a) (h₁ : P.diagonal_length = b) (h₂ : P.perpendicular = true) : 
  ∃ (AC : ℝ), AC = Real.sqrt (4 * a^2 + b^2) :=
by
  sorry

end parallelogram_diagonal_length_l293_29381


namespace find_d_l293_29376

noncomputable def f (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem find_d (a b c d : ℝ) (roots_negative_integers : ∀ x, f x a b c d = 0 → x < 0) (sum_is_2023 : a + b + c + d = 2023) :
  d = 17020 :=
sorry

end find_d_l293_29376


namespace find_n_pos_int_l293_29320

theorem find_n_pos_int (n : ℕ) (h1 : n ^ 3 + 2 * n ^ 2 + 9 * n + 8 = k ^ 3) : n = 7 := 
sorry

end find_n_pos_int_l293_29320


namespace luke_points_per_round_l293_29326

-- Define the total number of points scored 
def totalPoints : ℕ := 8142

-- Define the number of rounds played
def rounds : ℕ := 177

-- Define the points gained per round which we need to prove
def pointsPerRound : ℕ := 46

-- Now, we can state: if Luke played 177 rounds and scored a total of 8142 points, then he gained 46 points per round
theorem luke_points_per_round :
  (totalPoints = 8142) → (rounds = 177) → (totalPoints / rounds = pointsPerRound) := by
  sorry

end luke_points_per_round_l293_29326


namespace common_difference_of_arithmetic_sequence_l293_29339

theorem common_difference_of_arithmetic_sequence (a_n : ℕ → ℤ) (S_n : ℕ → ℤ) (n d a_2 S_3 a_4 : ℤ) 
  (h1 : a_2 + S_3 = -4) (h2 : a_4 = 3)
  (h3 : ∀ n, S_n = n * (a_n + (a_n + (n - 1) * d)) / 2)
  : d = 2 := by
  sorry

end common_difference_of_arithmetic_sequence_l293_29339


namespace incentive_given_to_john_l293_29324

-- Conditions (definitions)
def commission_held : ℕ := 25000
def advance_fees : ℕ := 8280
def amount_given_to_john : ℕ := 18500

-- Problem statement
theorem incentive_given_to_john : (amount_given_to_john - (commission_held - advance_fees)) = 1780 := 
by
  sorry

end incentive_given_to_john_l293_29324


namespace math_problem_l293_29335

theorem math_problem 
  (x y : ℝ) 
  (h1 : x + y = -5) 
  (h2 : x * y = 3) :
  x * Real.sqrt (y / x) + y * Real.sqrt (x / y) = -2 * Real.sqrt 3 := 
sorry

end math_problem_l293_29335


namespace pq_true_l293_29327

-- Proposition p: a^2 + b^2 < 0 is false
def p_false (a b : ℝ) : Prop := ¬ (a^2 + b^2 < 0)

-- Proposition q: (a-2)^2 + |b-3| ≥ 0 is true
def q_true (a b : ℝ) : Prop := (a - 2)^2 + |b - 3| ≥ 0

-- Theorem stating that "p ∨ q" is true
theorem pq_true (a b : ℝ) (h1 : p_false a b) (h2 : q_true a b) : (a^2 + b^2 < 0 ∨ (a - 2)^2 + |b - 3| ≥ 0) :=
by {
  sorry
}

end pq_true_l293_29327


namespace max_marks_is_400_l293_29362

-- Given conditions
def passing_mark (M : ℝ) : ℝ := 0.30 * M
def student_marks : ℝ := 80
def marks_failed_by : ℝ := 40
def pass_marks : ℝ := student_marks + marks_failed_by

-- Statement to prove
theorem max_marks_is_400 (M : ℝ) (h : passing_mark M = pass_marks) : M = 400 :=
by sorry

end max_marks_is_400_l293_29362


namespace grunters_win_4_out_of_6_l293_29387

/-- The Grunters have a probability of winning any given game as 60% --/
def p : ℚ := 3 / 5

/-- The Grunters have a probability of losing any given game as 40% --/
def q : ℚ := 1 - p

/-- The binomial coefficient for choosing exactly 4 wins out of 6 games --/
def binomial_6_4 : ℚ := Nat.choose 6 4

/-- The probability that the Grunters win exactly 4 out of the 6 games --/
def prob_4_wins : ℚ := binomial_6_4 * (p ^ 4) * (q ^ 2)

/--
The probability that the Grunters win exactly 4 out of the 6 games is
exactly $\frac{4860}{15625}$.
--/
theorem grunters_win_4_out_of_6 : prob_4_wins = 4860 / 15625 := by
  sorry

end grunters_win_4_out_of_6_l293_29387


namespace find_integer_in_range_divisible_by_18_l293_29321

theorem find_integer_in_range_divisible_by_18 
  (n : ℕ) (h1 : 900 ≤ n) (h2 : n ≤ 912) (h3 : n % 18 = 0) : n = 900 :=
sorry

end find_integer_in_range_divisible_by_18_l293_29321


namespace john_splits_profit_correctly_l293_29377

-- Conditions
def total_cookies : ℕ := 6 * 12
def revenue_per_cookie : ℝ := 1.5
def cost_per_cookie : ℝ := 0.25
def amount_per_charity : ℝ := 45

-- Computations based on conditions
def total_revenue : ℝ := total_cookies * revenue_per_cookie
def total_cost : ℝ := total_cookies * cost_per_cookie
def total_profit : ℝ := total_revenue - total_cost

-- Proof statement
theorem john_splits_profit_correctly : total_profit / amount_per_charity = 2 := by
  sorry

end john_splits_profit_correctly_l293_29377


namespace fraction_comparison_l293_29316

theorem fraction_comparison :
  let d := 0.33333333
  let f := (1 : ℚ) / 3
  f > d ∧ f - d = 1 / (3 * (10^8 : ℚ)) :=
by
  sorry

end fraction_comparison_l293_29316


namespace product_of_five_consecutive_integers_divisible_by_120_l293_29342

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) : 
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by 
  sorry

end product_of_five_consecutive_integers_divisible_by_120_l293_29342


namespace sheetrock_width_l293_29357

theorem sheetrock_width (l A w : ℕ) (h_length : l = 6) (h_area : A = 30) (h_formula : A = l * w) : w = 5 :=
by
  -- Placeholder for the proof
  sorry

end sheetrock_width_l293_29357


namespace max_side_length_triangle_l293_29352

def triangle_with_max_side_length (a b c : ℕ) (ha : a ≠ b ∧ b ≠ c ∧ c ≠ a) (hper : a + b + c = 30) : Prop :=
  a > b ∧ a > c ∧ a = 14

theorem max_side_length_triangle : ∃ a b c : ℕ, 
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a + b + c = 30 ∧ a > b ∧ a > c ∧ a = 14 :=
sorry

end max_side_length_triangle_l293_29352


namespace valid_permutations_l293_29391

theorem valid_permutations (a : Fin 101 → ℕ) :
  (∀ k, a k ≥ 2 ∧ a k ≤ 102 ∧ (∃ j, a j = k + 2)) →
  (∀ k, a (k + 1) % (k + 1) = 0) →
  (∃ cycles : List (List ℕ), cycles = [[1, 102], [1, 2, 102], [1, 3, 102], [1, 6, 102], [1, 17, 102], [1, 34, 102], 
                                       [1, 51, 102], [1, 2, 6, 102], [1, 2, 34, 102], [1, 3, 6, 102], [1, 3, 51, 102], 
                                       [1, 17, 34, 102], [1, 17, 51, 102]]) :=
sorry

end valid_permutations_l293_29391


namespace min_value_expression_l293_29308

noncomputable def sinSquare (θ : ℝ) : ℝ :=
  Real.sin (θ) ^ 2

theorem min_value_expression (θ₁ θ₂ θ₃ θ₄ : ℝ) 
  (h₁ : θ₁ > 0) (h₂ : θ₂ > 0) (h₃ : θ₃ > 0) (h₄ : θ₄ > 0)
  (sum_eq_pi : θ₁ + θ₂ + θ₃ + θ₄ = Real.pi) :
  (2 * sinSquare θ₁ + 1 / sinSquare θ₁) *
  (2 * sinSquare θ₂ + 1 / sinSquare θ₂) *
  (2 * sinSquare θ₃ + 1 / sinSquare θ₃) *
  (2 * sinSquare θ₄ + 1 / sinSquare θ₁) ≥ 81 := 
by
  sorry

end min_value_expression_l293_29308


namespace roger_cookie_price_l293_29309

noncomputable def price_per_roger_cookie (A_cookies: ℕ) (A_price_per_cookie: ℕ) (A_area_per_cookie: ℕ) (R_cookies: ℕ) (R_area_per_cookie: ℕ): ℕ :=
  by
  let A_total_earnings := A_cookies * A_price_per_cookie
  let R_total_area := A_cookies * A_area_per_cookie
  let price_per_R_cookie := A_total_earnings / R_cookies
  exact price_per_R_cookie
  
theorem roger_cookie_price {A_cookies A_price_per_cookie A_area_per_cookie R_cookies R_area_per_cookie : ℕ}
  (h1 : A_cookies = 12)
  (h2 : A_price_per_cookie = 60)
  (h3 : A_area_per_cookie = 12)
  (h4 : R_cookies = 18) -- assumed based on area calculation 144 / 8 (we need this input to match solution context)
  (h5 : R_area_per_cookie = 8) :
  price_per_roger_cookie A_cookies A_price_per_cookie A_area_per_cookie R_cookies R_area_per_cookie = 40 :=
  by
  sorry

end roger_cookie_price_l293_29309


namespace false_props_l293_29394

-- Definitions for conditions
def prop1 :=
  ∀ (a b c d : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ (a * d = b * c) → 
  (a / b = b / c ∧ b / c = c / d)

def prop2 :=
  ∀ (a : ℕ), (∃ k : ℕ, a = 2 * k) → (a % 2 = 0)

def prop3 :=
  ∀ (A : ℝ), (A > 30) → (Real.sin (A * Real.pi / 180) > 1 / 2)

-- Theorem statement
theorem false_props : (¬ prop1) ∧ (¬ prop3) :=
by sorry

end false_props_l293_29394


namespace evaluate_fraction_l293_29314

theorem evaluate_fraction : (25 * 5 + 5^2) / (5^2 - 15) = 15 := 
by
  sorry

end evaluate_fraction_l293_29314


namespace maximize_expression_l293_29341

theorem maximize_expression :
  ∀ (a b c d e : ℕ),
    a ≠ b → a ≠ c → a ≠ d → a ≠ e → b ≠ c → b ≠ d → b ≠ e → c ≠ d → c ≠ e → d ≠ e →
    (a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 4 ∨ a = 6) → 
    (b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4 ∨ b = 6) →
    (c = 1 ∨ c = 2 ∨ c = 3 ∨ c = 4 ∨ c = 6) →
    (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6) →
    (e = 1 ∨ e = 2 ∨ e = 3 ∨ e = 4 ∨ e = 6) →
    ((a : ℚ) / 2 + (d : ℚ) / e * (c / b)) ≤ 9 :=
by
  sorry

end maximize_expression_l293_29341


namespace total_hours_worked_l293_29313

theorem total_hours_worked (Amber_hours : ℕ) (h_Amber : Amber_hours = 12) 
  (Armand_hours : ℕ) (h_Armand : Armand_hours = Amber_hours / 3)
  (Ella_hours : ℕ) (h_Ella : Ella_hours = Amber_hours * 2) : 
  Amber_hours + Armand_hours + Ella_hours = 40 :=
by
  rw [h_Amber, h_Armand, h_Ella]
  norm_num
  sorry

end total_hours_worked_l293_29313


namespace max_tulips_l293_29392

theorem max_tulips (r y n : ℕ) (hr : r = y + 1) (hc : 50 * y + 31 * r ≤ 600) :
  r + y = 2 * y + 1 → n = 2 * y + 1 → n = 15 :=
by
  -- Given the constraints, we need to identify the maximum n given the costs and relationships.
  sorry

end max_tulips_l293_29392


namespace ratio_of_length_to_width_l293_29325

variable (P W L : ℕ)
variable (ratio : ℕ × ℕ)

theorem ratio_of_length_to_width (h1 : P = 336) (h2 : W = 70) (h3 : 2 * L + 2 * W = P) : ratio = (7, 5) :=
by
  sorry

end ratio_of_length_to_width_l293_29325


namespace unique_positive_real_solution_l293_29310

theorem unique_positive_real_solution :
  ∃! x : ℝ, 0 < x ∧ (x^8 + 5 * x^7 + 10 * x^6 + 2023 * x^5 - 2021 * x^4 = 0) := sorry

end unique_positive_real_solution_l293_29310


namespace calculate_triangle_area_l293_29365

-- Define the side lengths of the triangle.
def side1 : ℕ := 13
def side2 : ℕ := 13
def side3 : ℕ := 24

-- Define the area calculation.
noncomputable def triangle_area : ℕ := 60

-- Statement of the theorem we wish to prove.
theorem calculate_triangle_area :
  ∃ (a b c : ℕ) (area : ℕ), a = side1 ∧ b = side2 ∧ c = side3 ∧ area = triangle_area :=
sorry

end calculate_triangle_area_l293_29365


namespace number_of_dials_l293_29300

theorem number_of_dials (k : ℕ) (aligned_sums : ℕ → ℕ) :
  (∀ i j : ℕ, i ≠ j → aligned_sums i % 12 = aligned_sums j % 12) ↔ k = 12 :=
by
  sorry

end number_of_dials_l293_29300


namespace f_at_five_l293_29338

-- Define the function f with the property given in the condition
axiom f : ℝ → ℝ
axiom f_prop : ∀ x : ℝ, f (3 * x - 1) = x^2 + x + 1

-- Prove that f(5) = 7 given the properties above
theorem f_at_five : f 5 = 7 :=
by
  sorry

end f_at_five_l293_29338


namespace baker_made_cakes_l293_29374

-- Conditions
def cakes_sold : ℕ := 145
def cakes_left : ℕ := 72

-- Question and required proof
theorem baker_made_cakes : (cakes_sold + cakes_left = 217) :=
by
  sorry

end baker_made_cakes_l293_29374


namespace lindsey_savings_in_october_l293_29337

-- Definitions based on conditions
def savings_september := 50
def savings_november := 11
def spending_video_game := 87
def final_amount_left := 36
def mom_gift := 25

-- The theorem statement
theorem lindsey_savings_in_october (X : ℕ) 
  (h1 : savings_september + X + savings_november > 75) 
  (total_savings := savings_september + X + savings_november + mom_gift) 
  (final_condition : total_savings - spending_video_game = final_amount_left) : 
  X = 37 :=
by
  sorry

end lindsey_savings_in_october_l293_29337


namespace bailey_dog_treats_l293_29385

-- Definitions based on conditions
def total_charges_per_card : Nat := 5
def number_of_cards : Nat := 4
def chew_toys : Nat := 2
def rawhide_bones : Nat := 10

-- Total number of items bought
def total_items : Nat := total_charges_per_card * number_of_cards

-- Definition of the number of dog treats
def dog_treats : Nat := total_items - (chew_toys + rawhide_bones)

-- Theorem to prove the number of dog treats
theorem bailey_dog_treats : dog_treats = 8 := by
  -- Proof is skipped with sorry
  sorry

end bailey_dog_treats_l293_29385


namespace julie_initial_savings_l293_29330

theorem julie_initial_savings (S r : ℝ) 
  (h1 : (S / 2) * r * 2 = 120) 
  (h2 : (S / 2) * ((1 + r)^2 - 1) = 124) : 
  S = 1800 := 
sorry

end julie_initial_savings_l293_29330


namespace find_x_l293_29340

def bin_op (p1 p2 : ℤ × ℤ) : ℤ × ℤ :=
  (p1.1 - 2 * p2.1, p1.2 + 2 * p2.2)

theorem find_x :
  ∃ x y : ℤ, 
  bin_op (2, -4) (1, -3) = bin_op (x, y) (2, 1) ∧ x = 4 :=
by
  sorry

end find_x_l293_29340


namespace triangle_inequality_l293_29363

theorem triangle_inequality (a : ℝ) :
  (3/2 < a) ∧ (a < 5) ↔ ((4 * a + 1 - (3 * a - 1) < 12 - a) ∧ (4 * a + 1 + (3 * a - 1) > 12 - a)) := 
by 
  sorry

end triangle_inequality_l293_29363


namespace average_monthly_growth_rate_l293_29370

variable (x : ℝ)

-- Conditions
def turnover_January : ℝ := 36
def turnover_March : ℝ := 48

-- Theorem statement that corresponds to the problem's conditions and question
theorem average_monthly_growth_rate :
  turnover_January * (1 + x)^2 = turnover_March :=
sorry

end average_monthly_growth_rate_l293_29370


namespace mean_value_of_quadrilateral_angles_l293_29388

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l293_29388


namespace determine_value_of_y_l293_29360

variable (s y : ℕ)
variable (h_pos : s > 30)
variable (h_eq : s * s = (s - 15) * (s + y))

theorem determine_value_of_y (h_pos : s > 30) (h_eq : s * s = (s - 15) * (s + y)) : 
  y = 15 * s / (s + 15) :=
by
  sorry

end determine_value_of_y_l293_29360


namespace volume_of_polyhedron_l293_29319

theorem volume_of_polyhedron (s : ℝ) : 
  let base_area := (3 * Real.sqrt 3 / 2) * s^2
  let height := s
  let volume := (1 / 3) * base_area * height
  volume = (Real.sqrt 3 / 2) * s^3 :=
by
  let base_area := (3 * Real.sqrt 3 / 2) * s^2
  let height := s
  let volume := (1 / 3) * base_area * height
  show volume = (Real.sqrt 3 / 2) * s^3
  sorry

end volume_of_polyhedron_l293_29319


namespace bucket_full_weight_l293_29301

theorem bucket_full_weight (p q : ℝ) (x y : ℝ) 
  (h1 : x + (1 / 3) * y = p) 
  (h2 : x + (3 / 4) * y = q) : 
  x + y = (8 * q - 3 * p) / 5 := 
  by
    sorry

end bucket_full_weight_l293_29301


namespace downstream_distance_correct_l293_29334

-- Definitions based on the conditions
def still_water_speed : ℝ := 22
def stream_speed : ℝ := 5
def travel_time : ℝ := 3

-- The effective speed downstream is the sum of the still water speed and the stream speed
def effective_speed_downstream : ℝ := still_water_speed + stream_speed

-- The distance covered downstream is the product of effective speed and travel time
def downstream_distance : ℝ := effective_speed_downstream * travel_time

-- The theorem to be proven
theorem downstream_distance_correct : downstream_distance = 81 := by
  sorry

end downstream_distance_correct_l293_29334


namespace find_second_number_l293_29366

theorem find_second_number (a : ℕ) (c : ℕ) (x : ℕ) : 
  3 * a + 3 * x + 3 * c + 11 = 170 → a = 16 → c = 20 → x = 17 := 
by
  intros h1 h2 h3
  rw [h2, h3] at h1
  simp at h1
  sorry

end find_second_number_l293_29366


namespace tangent_line_at_M_l293_29315

noncomputable def isOnCircle (x y : ℝ) : Prop := x^2 + y^2 = 1

noncomputable def M : ℝ × ℝ := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)

theorem tangent_line_at_M (hM : isOnCircle (M.1) (M.2)) : (∀ x y, M.1 = x ∨ M.2 = y → x + y = Real.sqrt 2) :=
by
  sorry

end tangent_line_at_M_l293_29315


namespace vector_sum_solve_for_m_n_l293_29350

-- Define the vectors
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)

-- Problem 1: Vector sum
theorem vector_sum : 3 • a + b - 2 • c = (0, 6) :=
by sorry

-- Problem 2: Solving for m and n
theorem solve_for_m_n (m n : ℝ) (hm : a = m • b + n • c) :
  m = 5 / 9 ∧ n = 8 / 9 :=
by sorry

end vector_sum_solve_for_m_n_l293_29350


namespace mr_yadav_expenses_l293_29306

theorem mr_yadav_expenses (S : ℝ) 
  (h1 : S > 0) 
  (h2 : 0.6 * S > 0) 
  (h3 : (12 * 0.2 * S) = 48456) : 
  0.2 * S = 4038 :=
by
  sorry

end mr_yadav_expenses_l293_29306


namespace total_mail_l293_29336

def monday_mail := 65
def tuesday_mail := monday_mail + 10
def wednesday_mail := tuesday_mail - 5
def thursday_mail := wednesday_mail + 15

theorem total_mail : 
  monday_mail + tuesday_mail + wednesday_mail + thursday_mail = 295 := by
  sorry

end total_mail_l293_29336


namespace arithmetic_sequence_identity_l293_29312

theorem arithmetic_sequence_identity (a : ℕ → ℝ) (d : ℝ)
    (h_arith : ∀ n, a (n + 1) = a 1 + n * d)
    (h_sum : a 4 + a 7 + a 10 = 30) :
    a 1 - a 3 - a 6 - a 8 - a 11 + a 13 = -20 :=
sorry

end arithmetic_sequence_identity_l293_29312


namespace sqrt_equality_l293_29331

theorem sqrt_equality (n : ℤ) (h : Real.sqrt (8 + n) = 9) : n = 73 :=
by
  sorry

end sqrt_equality_l293_29331


namespace gcd_polynomial_l293_29345

theorem gcd_polynomial (b : ℕ) (h : 570 ∣ b) : Nat.gcd (5 * b^3 + 2 * b^2 + 5 * b + 95) b = 95 :=
by
  sorry

end gcd_polynomial_l293_29345


namespace minimum_value_of_abs_phi_l293_29386

theorem minimum_value_of_abs_phi (φ : ℝ) :
  (∃ k : ℤ, φ = k * π - (13 * π) / 6) → 
  ∃ φ_min : ℝ, 0 ≤ φ_min ∧ φ_min = abs φ ∧ φ_min = π / 6 :=
by
  sorry

end minimum_value_of_abs_phi_l293_29386


namespace find_x_l293_29356

variable (c d : ℝ)

theorem find_x (x : ℝ) (h : x^2 + 4 * c^2 = (3 * d - x)^2) : 
  x = (9 * d^2 - 4 * c^2) / (6 * d) :=
sorry

end find_x_l293_29356


namespace find_triangles_l293_29304

/-- In a triangle, if the side lengths a, b, c (a ≤ b ≤ c) are integers, form a geometric progression (i.e., b² = ac),
    and at least one of a or c is equal to 100, then the possible values for the triple (a, b, c) are:
    (49, 70, 100), (64, 80, 100), (81, 90, 100), 
    (100, 100, 100), (100, 110, 121), (100, 120, 144),
    (100, 130, 169), (100, 140, 196), (100, 150, 225), (100, 160, 256). 
-/
theorem find_triangles (a b c : ℕ) (h1 : a ≤ b ∧ b ≤ c) 
(h2 : b * b = a * c)
(h3 : a = 100 ∨ c = 100) : 
  (a = 49 ∧ b = 70 ∧ c = 100) ∨ 
  (a = 64 ∧ b = 80 ∧ c = 100) ∨ 
  (a = 81 ∧ b = 90 ∧ c = 100) ∨ 
  (a = 100 ∧ b = 100 ∧ c = 100) ∨ 
  (a = 100 ∧ b = 110 ∧ c = 121) ∨ 
  (a = 100 ∧ b = 120 ∧ c = 144) ∨ 
  (a = 100 ∧ b = 130 ∧ c = 169) ∨ 
  (a = 100 ∧ b = 140 ∧ c = 196) ∨ 
  (a = 100 ∧ b = 150 ∧ c = 225) ∨ 
  (a = 100 ∧ b = 160 ∧ c = 256) := sorry

end find_triangles_l293_29304


namespace otimes_calculation_l293_29344

def otimes (x y : ℝ) : ℝ := x^2 + y^2

theorem otimes_calculation (x : ℝ) : otimes x (otimes x x) = x^2 + 4 * x^4 :=
by
  sorry

end otimes_calculation_l293_29344


namespace g_value_l293_29317

theorem g_value (g : ℝ → ℝ)
  (h0 : g 0 = 0)
  (h_mono : ∀ ⦃x y : ℝ⦄, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y)
  (h_symm : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x)
  (h_prop : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3) :
  g (2 / 5) = 1 / 2 :=
sorry

end g_value_l293_29317


namespace tips_multiple_l293_29367

variable (A T : ℝ) (x : ℝ)
variable (h1 : T = 7 * A)
variable (h2 : T / 4 = x * A)

theorem tips_multiple (A T : ℝ) (x : ℝ) (h1 : T = 7 * A) (h2 : T / 4 = x * A) : x = 1.75 := by
  sorry

end tips_multiple_l293_29367


namespace find_value_of_r_l293_29333

theorem find_value_of_r (a r : ℝ) (h1 : a / (1 - r) = 20) (h2 : a * r / (1 - r^2) = 8) : r = 2 / 3 :=
by
  sorry

end find_value_of_r_l293_29333


namespace fraction_inequality_l293_29358

variable (a b c : ℝ)

theorem fraction_inequality (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : c > a) (h5 : a > b) :
  (a / (c - a)) > (b / (c - b)) := 
sorry

end fraction_inequality_l293_29358


namespace total_students_in_class_l293_29375

theorem total_students_in_class (female_students : ℕ) (male_students : ℕ) (total_students : ℕ) 
  (h1 : female_students = 13) 
  (h2 : male_students = 3 * female_students) 
  (h3 : total_students = female_students + male_students) : 
    total_students = 52 := 
by
  sorry

end total_students_in_class_l293_29375


namespace incident_reflected_eqs_l293_29384

theorem incident_reflected_eqs {x y : ℝ} :
  (∃ (A B : ℝ × ℝ), A = (2, 3) ∧ B = (1, 1) ∧ 
   (∀ (P : ℝ × ℝ), (P = A ∨ P = B → (P.1 + P.2 + 1 = 0) → false)) ∧
   (∃ (line_inc line_ref : ℝ × ℝ × ℝ),
     line_inc = (5, -4, 2) ∧
     line_ref = (4, -5, 1))) :=
sorry

end incident_reflected_eqs_l293_29384


namespace correct_equation_l293_29399

theorem correct_equation:
  (∀ x y : ℝ, -5 * (x - y) = -5 * x + 5 * y) ∧ 
  (∀ a c : ℝ, ¬ (-2 * (-a + c) = -2 * a - 2 * c)) ∧ 
  (∀ x y z : ℝ, ¬ (3 - (x + y + z) = -x + y - z)) ∧ 
  (∀ a b : ℝ, ¬ (3 * (a + 2 * b) = 3 * a + 2 * b)) :=
by
  sorry

end correct_equation_l293_29399


namespace three_digit_numbers_count_l293_29378

theorem three_digit_numbers_count : 
  ∃ (count : ℕ), count = 3 ∧ 
  ∀ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ 
             (n / 100 = 9) ∧ 
             (∃ a b c, n = 100 * a + 10 * b + c ∧ a + b + c = 27) ∧ 
             (n % 2 = 0) → count = 3 :=
sorry

end three_digit_numbers_count_l293_29378


namespace always_composite_l293_29348

theorem always_composite (p : ℕ) (hp : Nat.Prime p) : ¬Nat.Prime (p^2 + 35) ∧ ¬Nat.Prime (p^2 + 55) :=
by
  sorry

end always_composite_l293_29348


namespace find_a_l293_29305

theorem find_a (a x y : ℤ) (h_x : x = 1) (h_y : y = -3) (h_eq : a * x - y = 1) : a = -2 := by
  -- Proof skipped
  sorry

end find_a_l293_29305


namespace rainy_days_l293_29373

theorem rainy_days (n R NR : ℤ) 
  (h1 : n * R + 4 * NR = 26)
  (h2 : 4 * NR - n * R = 14)
  (h3 : R + NR = 7) : 
  R = 2 := 
sorry

end rainy_days_l293_29373


namespace strawberries_weight_l293_29382

theorem strawberries_weight (marco_weight dad_increase : ℕ) (h_marco: marco_weight = 30) (h_diff: marco_weight = dad_increase + 13) : marco_weight + (marco_weight - 13) = 47 :=
by
  sorry

end strawberries_weight_l293_29382


namespace problem_a_problem_b_l293_29329

-- Definition for real roots condition in problem A
def has_real_roots (k : ℝ) : Prop :=
  let a := 1
  let b := -3
  let c := k
  b^2 - 4 * a * c ≥ 0

-- Problem A: Proving the range of k
theorem problem_a (k : ℝ) : has_real_roots k ↔ k ≤ 9 / 4 :=
by
  sorry

-- Definition for a quadratic equation having a given root
def has_root (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

-- Problem B: Proving the value of m given a common root condition
theorem problem_b (m : ℝ) : 
  (has_root 1 (-3) 2 1 ∧ has_root (m-1) 1 (m-3) 1) ↔ m = 3 / 2 :=
by
  sorry

end problem_a_problem_b_l293_29329


namespace circle_radius_increase_l293_29393

variable (r n : ℝ) -- declare variables r and n as real numbers

theorem circle_radius_increase (h : 2 * π * (r + n) = 2 * (2 * π * r)) : r = n :=
by
  sorry

end circle_radius_increase_l293_29393


namespace profit_margin_in_terms_of_retail_price_l293_29379

theorem profit_margin_in_terms_of_retail_price
  (k c P_R : ℝ) (h1 : ∀ C, P = k * C) (h2 : ∀ C, P_R = c * (P + C)) :
  P = (k / (c * (k + 1))) * P_R :=
by sorry

end profit_margin_in_terms_of_retail_price_l293_29379


namespace range_of_m_l293_29371

open Set

def setM (m : ℝ) : Set ℝ := { x | x ≤ m }
def setP : Set ℝ := { x | x ≥ -1 }

theorem range_of_m (m : ℝ) (h : setM m ∩ setP = ∅) : m < -1 := sorry

end range_of_m_l293_29371


namespace express_in_scientific_notation_l293_29332

theorem express_in_scientific_notation :
  (10.58 * 10^9) = 1.058 * 10^10 :=
by
  sorry

end express_in_scientific_notation_l293_29332


namespace divisor_is_13_l293_29369

theorem divisor_is_13 (N D : ℕ) (h1 : N = 32) (h2 : (N - 6) / D = 2) : D = 13 := by
  sorry

end divisor_is_13_l293_29369


namespace cleaning_time_l293_29302

def lara_rate := 1 / 4
def chris_rate := 1 / 6
def combined_rate := lara_rate + chris_rate

theorem cleaning_time (t : ℝ) : 
  (combined_rate * (t - 2) = 1) ↔ (t = 22 / 5) :=
by
  sorry

end cleaning_time_l293_29302


namespace probability_point_inside_circle_l293_29328

theorem probability_point_inside_circle :
  (∃ (m n : ℕ), 1 ≤ m ∧ m ≤ 6 ∧ 1 ≤ n ∧ n ≤ 6) →
  (∃ (P : ℚ), P = 2/9) :=
by
  sorry

end probability_point_inside_circle_l293_29328


namespace intersection_with_unit_circle_l293_29351

theorem intersection_with_unit_circle (α : ℝ) : 
    let x := Real.cos (α - Real.pi / 2)
    let y := Real.sin (α - Real.pi / 2)
    (x, y) = (Real.sin α, -Real.cos α) :=
by
  sorry

end intersection_with_unit_circle_l293_29351


namespace container_volume_ratio_l293_29346

variable (A B C : ℝ)

theorem container_volume_ratio (h1 : (4 / 5) * A = (3 / 5) * B) (h2 : (3 / 5) * B = (3 / 4) * C) :
  A / C = 15 / 16 :=
sorry

end container_volume_ratio_l293_29346


namespace overall_gain_percentage_correct_l293_29398

structure Transaction :=
  (buy_prices : List ℕ)
  (sell_prices : List ℕ)

def overallGainPercentage (trans : Transaction) : ℚ :=
  let total_cost := (trans.buy_prices.foldl (· + ·) 0 : ℚ)
  let total_sell := (trans.sell_prices.foldl (· + ·) 0 : ℚ)
  (total_sell - total_cost) / total_cost * 100

theorem overall_gain_percentage_correct
  (trans : Transaction)
  (h_buy_prices : trans.buy_prices = [675, 850, 920])
  (h_sell_prices : trans.sell_prices = [1080, 1100, 1000]) :
  overallGainPercentage trans = 30.06 := by
  sorry

end overall_gain_percentage_correct_l293_29398


namespace time_to_cross_first_platform_l293_29390

noncomputable def train_length : ℝ := 30
noncomputable def first_platform_length : ℝ := 180
noncomputable def second_platform_length : ℝ := 250
noncomputable def time_second_platform : ℝ := 20

noncomputable def train_speed : ℝ :=
(train_length + second_platform_length) / time_second_platform

noncomputable def time_first_platform : ℝ :=
(train_length + first_platform_length) / train_speed

theorem time_to_cross_first_platform :
  time_first_platform = 15 :=
by
  sorry

end time_to_cross_first_platform_l293_29390


namespace solve_equation_l293_29364

theorem solve_equation (x : ℝ) : 
  3 * x * (x - 1) = 2 * x - 2 ↔ x = 1 ∨ x = 2 / 3 :=
by
  sorry

end solve_equation_l293_29364


namespace chelsea_sugar_problem_l293_29322

variable (initial_sugar : ℕ)
variable (num_bags : ℕ)
variable (sugar_lost_fraction : ℕ)

def remaining_sugar (initial_sugar : ℕ) (num_bags : ℕ) (sugar_lost_fraction : ℕ) : ℕ :=
  let sugar_per_bag := initial_sugar / num_bags
  let sugar_lost := sugar_per_bag / sugar_lost_fraction
  let remaining_bags_sugar := (num_bags - 1) * sugar_per_bag
  remaining_bags_sugar + (sugar_per_bag - sugar_lost)

theorem chelsea_sugar_problem : 
  remaining_sugar 24 4 2 = 21 :=
by
  sorry

end chelsea_sugar_problem_l293_29322


namespace how_many_years_younger_is_C_compared_to_A_l293_29396

variables (a b c d : ℕ)

def condition1 : Prop := a + b = b + c + 13
def condition2 : Prop := b + d = c + d + 7
def condition3 : Prop := a + d = 2 * c - 12

theorem how_many_years_younger_is_C_compared_to_A
  (h1 : condition1 a b c)
  (h2 : condition2 b c d)
  (h3 : condition3 a c d) : a = c + 13 :=
sorry

end how_many_years_younger_is_C_compared_to_A_l293_29396


namespace difference_of_squares_650_550_l293_29395

theorem difference_of_squares_650_550 : 650^2 - 550^2 = 120000 :=
by sorry

end difference_of_squares_650_550_l293_29395


namespace num_common_elements_1000_multiples_5_9_l293_29343

def multiples_up_to (n k : ℕ) : ℕ := n / k

def num_common_elements_in_sets (k m n : ℕ) : ℕ :=
  multiples_up_to n (Nat.lcm k m)

theorem num_common_elements_1000_multiples_5_9 :
  num_common_elements_in_sets 5 9 5000 = 111 :=
by
  -- The proof is omitted as per instructions
  sorry

end num_common_elements_1000_multiples_5_9_l293_29343


namespace cannot_be_20182017_l293_29383

theorem cannot_be_20182017 (a b : ℤ) (h : a * b * (a + b) = 20182017) : False :=
by
  sorry

end cannot_be_20182017_l293_29383
