import Mathlib

namespace NUMINAMATH_GPT_determinant_in_terms_of_roots_l2348_234891

theorem determinant_in_terms_of_roots 
  (r s t a b c : ℝ)
  (h1 : a^3 - r*a^2 + s*a - t = 0)
  (h2 : b^3 - r*b^2 + s*b - t = 0)
  (h3 : c^3 - r*c^2 + s*c - t = 0) :
  (2 + a) * ((2 + b) * (2 + c) - 4) - 2 * (2 * (2 + c) - 4) + 2 * (2 * 2 - (2 + b) * 2) = t - 2 * s :=
by
  sorry

end NUMINAMATH_GPT_determinant_in_terms_of_roots_l2348_234891


namespace NUMINAMATH_GPT_kim_total_water_intake_l2348_234842

def quarts_to_ounces (q : ℝ) : ℝ := q * 32

theorem kim_total_water_intake :
  (quarts_to_ounces 1.5) + 12 = 60 := 
by
  -- proof step 
  sorry

end NUMINAMATH_GPT_kim_total_water_intake_l2348_234842


namespace NUMINAMATH_GPT_roots_formula_l2348_234819

theorem roots_formula (x₁ x₂ p : ℝ)
  (h₁ : x₁ + x₂ = 6 * p)
  (h₂ : x₁ * x₂ = p^2)
  (h₃ : ∀ x, x ^ 2 - 6 * p * x + p ^ 2 = 0 → x = x₁ ∨ x = x₂) :
  (1 / (x₁ + p) + 1 / (x₂ + p) = 1 / p) :=
by
  sorry

end NUMINAMATH_GPT_roots_formula_l2348_234819


namespace NUMINAMATH_GPT_least_pos_int_with_12_pos_factors_is_72_l2348_234896

def least_positive_integer_with_12_factors (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 0 ∧ m ≠ n ∧ (∀ d : ℕ, d ∣ m → (d = n ∨ d = 1)) ∧
  ((∀ d : ℕ, d ∣ m → ∃ e : ℕ, e = 1) → n = 72)

theorem least_pos_int_with_12_pos_factors_is_72 (n : ℕ) :
  least_positive_integer_with_12_factors n → n = 72 := by
  sorry

end NUMINAMATH_GPT_least_pos_int_with_12_pos_factors_is_72_l2348_234896


namespace NUMINAMATH_GPT_contrapositive_of_proposition_is_false_l2348_234855

theorem contrapositive_of_proposition_is_false (x y : ℝ) 
  (h₀ : (x + y > 0) → (x > 0 ∧ y > 0)) : 
  ¬ ((x ≤ 0 ∨ y ≤ 0) → (x + y ≤ 0)) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_of_proposition_is_false_l2348_234855


namespace NUMINAMATH_GPT_jake_last_10_shots_l2348_234833

-- conditions
variable (total_shots_initially : ℕ) (shots_made_initially : ℕ) (percentage_initial : ℝ)
variable (total_shots_finally : ℕ) (shots_made_finally : ℕ) (percentage_final : ℝ)

axiom initial_conditions : shots_made_initially = percentage_initial * total_shots_initially
axiom final_conditions : shots_made_finally = percentage_final * total_shots_finally
axiom shots_difference : total_shots_finally - total_shots_initially = 10

-- prove that Jake made 7 out of the last 10 shots
theorem jake_last_10_shots : total_shots_initially = 30 → 
                             percentage_initial = 0.60 →
                             total_shots_finally = 40 → 
                             percentage_final = 0.62 →
                             shots_made_finally - shots_made_initially = 7 :=
by
  -- proofs to be filled in
  sorry

end NUMINAMATH_GPT_jake_last_10_shots_l2348_234833


namespace NUMINAMATH_GPT_problem_remainder_3_l2348_234877

theorem problem_remainder_3 :
  88 % 5 = 3 :=
by
  sorry

end NUMINAMATH_GPT_problem_remainder_3_l2348_234877


namespace NUMINAMATH_GPT_number_of_palindromes_divisible_by_6_l2348_234822

theorem number_of_palindromes_divisible_by_6 :
  let is_palindrome (n : ℕ) : Prop := n / 1000 = n % 10 ∧ (n / 100 % 10) = (n / 10 % 10)
  let valid_digits (n : ℕ) : Prop := n >= 1000 ∧ n < 10000
  let divisible_6 (n : ℕ) : Prop := n % 6 = 0
  (Finset.filter (λ n => is_palindrome n ∧ valid_digits n ∧ divisible_6 n) (Finset.range 10000)).card = 13 :=
by
  -- We define what it means to be a palindrome between 1000 and 10000
  let is_palindrome (n : ℕ) : Prop := n / 1000 = n % 10 ∧ n / 100 % 10 = n / 10 % 10
  
  -- We define a valid number between 1000 and 10000
  let valid_digits (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
  
  -- We define what it means to be divisible by 6
  let divisible_6 (n : ℕ) : Prop := n % 6 = 0

  -- Filtering the range 10000 within valid four-digit palindromes and checking for multiples of 6
  exact sorry

end NUMINAMATH_GPT_number_of_palindromes_divisible_by_6_l2348_234822


namespace NUMINAMATH_GPT_incoming_class_student_count_l2348_234876

theorem incoming_class_student_count (n : ℕ) :
  n < 1000 ∧ n % 25 = 18 ∧ n % 28 = 26 → n = 418 :=
by
  sorry

end NUMINAMATH_GPT_incoming_class_student_count_l2348_234876


namespace NUMINAMATH_GPT_math_problem_real_solution_l2348_234846

theorem math_problem_real_solution (x y : ℝ) (h : x^2 * y^2 - x * y - x / y - y / x = 4) : 
  (x - 2) * (y - 2) = 3 - 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_math_problem_real_solution_l2348_234846


namespace NUMINAMATH_GPT_water_bill_august_32m_cubed_water_usage_october_59_8_yuan_l2348_234898

noncomputable def tiered_water_bill (usage : ℕ) : ℝ :=
  if usage <= 20 then
    2.3 * usage
  else if usage <= 30 then
    2.3 * 20 + 3.45 * (usage - 20)
  else
    2.3 * 20 + 3.45 * 10 + 4.6 * (usage - 30)

-- (1) Prove that if Xiao Ming's family used 32 cubic meters of water in August, 
-- their water bill is 89.7 yuan.
theorem water_bill_august_32m_cubed : tiered_water_bill 32 = 89.7 := by
  sorry

-- (2) Prove that if Xiao Ming's family paid 59.8 yuan for their water bill in October, 
-- they used 24 cubic meters of water.
theorem water_usage_october_59_8_yuan : ∃ x : ℕ, tiered_water_bill x = 59.8 ∧ x = 24 := by
  use 24
  sorry

end NUMINAMATH_GPT_water_bill_august_32m_cubed_water_usage_october_59_8_yuan_l2348_234898


namespace NUMINAMATH_GPT_peter_erasers_l2348_234893

theorem peter_erasers (initial_erasers : ℕ) (extra_erasers : ℕ) (final_erasers : ℕ)
  (h1 : initial_erasers = 8) (h2 : extra_erasers = 3) : final_erasers = 11 :=
by
  sorry

end NUMINAMATH_GPT_peter_erasers_l2348_234893


namespace NUMINAMATH_GPT_permutations_PERCEPTION_l2348_234884

-- Define the word "PERCEPTION" and its letter frequencies
def word : String := "PERCEPTION"

def freq_P : Nat := 2
def freq_E : Nat := 2
def freq_R : Nat := 1
def freq_C : Nat := 1
def freq_T : Nat := 1
def freq_I : Nat := 1
def freq_O : Nat := 1
def freq_N : Nat := 1

-- Define the total number of letters in the word
def total_letters : Nat := 10

-- Calculate the number of permutations for the multiset
def permutations : Nat :=
  total_letters.factorial / (freq_P.factorial * freq_E.factorial)

-- Proof problem
theorem permutations_PERCEPTION :
  permutations = 907200 :=
by
  sorry

end NUMINAMATH_GPT_permutations_PERCEPTION_l2348_234884


namespace NUMINAMATH_GPT_drink_all_tea_l2348_234853

theorem drink_all_tea (cups : Fin 30 → Prop) (red blue : Fin 30 → Prop)
  (h₀ : ∀ n, cups n ↔ (red n ↔ ¬ blue n))
  (h₁ : ∃ a b, a ≠ b ∧ red a ∧ blue b)
  (h₂ : ∀ n, red n → red (n + 2))
  (h₃ : ∀ n, blue n → blue (n + 2)) :
  ∃ sequence : ℕ → Fin 30, (∀ n, cups (sequence n)) ∧ (sequence 0 ≠ sequence 1) 
  ∧ (∀ n, cups (sequence (n+1))) :=
by
  sorry

end NUMINAMATH_GPT_drink_all_tea_l2348_234853


namespace NUMINAMATH_GPT_factorize_x_cube_minus_9x_l2348_234837

-- Lean statement: Prove that x^3 - 9x = x(x+3)(x-3)
theorem factorize_x_cube_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
by
  sorry

end NUMINAMATH_GPT_factorize_x_cube_minus_9x_l2348_234837


namespace NUMINAMATH_GPT_train_speed_l2348_234852

theorem train_speed (length : ℝ) (time : ℝ) (speed : ℝ) (h_length : length = 975) (h_time : time = 48) (h_speed : speed = length / time * 3.6) : 
  speed = 73.125 := 
by 
  sorry

end NUMINAMATH_GPT_train_speed_l2348_234852


namespace NUMINAMATH_GPT_inverse_function_correct_l2348_234818

noncomputable def f (x : ℝ) : ℝ := 3 - 7 * x

noncomputable def g (x : ℝ) : ℝ := (3 - x) / 7

theorem inverse_function_correct : ∀ x : ℝ, f (g x) = x ∧ g (f x) = x :=
by
  intro x
  sorry

end NUMINAMATH_GPT_inverse_function_correct_l2348_234818


namespace NUMINAMATH_GPT_center_number_is_4_l2348_234831

-- Define the numbers and the 3x3 grid
inductive Square
| center | top_middle | left_middle | right_middle | bottom_middle

-- Define the properties of the problem
def isConsecutiveAdjacent (a b : ℕ) : Prop := 
  (a + 1 = b ∨ a = b + 1)

-- The condition to check the sum of edge squares
def sum_edge_squares (grid : Square → ℕ) : Prop := 
  grid Square.top_middle + grid Square.left_middle + grid Square.right_middle + grid Square.bottom_middle = 28

-- The condition that the center square number is even
def even_center (grid : Square → ℕ) : Prop := 
  grid Square.center % 2 = 0

-- The main theorem statement
theorem center_number_is_4 (grid : Square → ℕ) :
  (∀ i j : Square, i ≠ j → isConsecutiveAdjacent (grid i) (grid j)) → 
  (grid Square.top_middle + grid Square.left_middle + grid Square.right_middle + grid Square.bottom_middle = 28) →
  (grid Square.center % 2 = 0) →
  grid Square.center = 4 :=
by sorry

end NUMINAMATH_GPT_center_number_is_4_l2348_234831


namespace NUMINAMATH_GPT_evaluate_expression_l2348_234813

noncomputable def expression (a : ℚ) : ℚ := 
  (a / (a - 1)) / ((a + 1) / (a^2 - 1)) - (1 - 2 * a)

theorem evaluate_expression (a : ℚ) (ha : a = -1/3) : expression a = -2 :=
by 
  rw [expression, ha]
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2348_234813


namespace NUMINAMATH_GPT_two_roots_iff_a_greater_than_neg1_l2348_234879

theorem two_roots_iff_a_greater_than_neg1 (a : ℝ) :
  (∃! x : ℝ, x^2 + 2*x + 2*|x + 1| = a) ↔ a > -1 :=
sorry

end NUMINAMATH_GPT_two_roots_iff_a_greater_than_neg1_l2348_234879


namespace NUMINAMATH_GPT_not_divisible_by_pow_two_l2348_234883

theorem not_divisible_by_pow_two (n : ℕ) (h : n > 1) : ¬ (2^n ∣ (3^n + 1)) :=
by
  sorry

end NUMINAMATH_GPT_not_divisible_by_pow_two_l2348_234883


namespace NUMINAMATH_GPT_range_of_a_l2348_234868

variable (a : ℝ)

def p : Prop := ∃ x₀ : ℝ, x₀^2 - a * x₀ + 1 = 0

def q : Prop := ∀ x : ℝ, x ≥ 0 → x^2 - 2 * a * x + a^2 + 1 ≥ 1

theorem range_of_a : ¬(p a ∨ q a) → -2 < a ∧ a < 0 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l2348_234868


namespace NUMINAMATH_GPT_time_to_meet_in_minutes_l2348_234872

def distance_between_projectiles : ℕ := 1998
def speed_projectile_1 : ℕ := 444
def speed_projectile_2 : ℕ := 555

theorem time_to_meet_in_minutes : 
  (distance_between_projectiles / (speed_projectile_1 + speed_projectile_2)) * 60 = 120 := 
by
  sorry

end NUMINAMATH_GPT_time_to_meet_in_minutes_l2348_234872


namespace NUMINAMATH_GPT_value_of_expression_l2348_234820

theorem value_of_expression (x y : ℝ) (h₁ : x = 3) (h₂ : y = 4) :
  (x^3 + 3*y^3) / 9 = 24.33 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2348_234820


namespace NUMINAMATH_GPT_jeep_initial_distance_l2348_234864

theorem jeep_initial_distance (D : ℝ) (h1 : ∀ t : ℝ, t = 4 → D / t = 103.33 * (3 / 8)) :
  D = 275.55 :=
sorry

end NUMINAMATH_GPT_jeep_initial_distance_l2348_234864


namespace NUMINAMATH_GPT_length_of_bridge_l2348_234835

/-- What is the length of a bridge (in meters), which a train 156 meters long and travelling at 45 km/h can cross in 40 seconds? -/
theorem length_of_bridge (train_length: ℕ) (train_speed_kmh: ℕ) (time_seconds: ℕ) (bridge_length: ℕ) :
  train_length = 156 →
  train_speed_kmh = 45 →
  time_seconds = 40 →
  bridge_length = 344 :=
by {
  sorry
}

end NUMINAMATH_GPT_length_of_bridge_l2348_234835


namespace NUMINAMATH_GPT_solution_one_solution_two_solution_three_l2348_234857

open Real

noncomputable def problem_one (a b : ℝ) (cosA : ℝ) : ℝ :=
if a = sqrt 6 ∧ b = 2 * 1 ∧ cosA = -1/4 then 1 else 0

theorem solution_one (a b : ℝ) (cosA : ℝ) :
  a = sqrt 6 → b = 2 * 1 → cosA = -1/4 → problem_one a b cosA = 1 := by
  intros ha hb hcos
  unfold problem_one
  simp [ha, hb, hcos]

noncomputable def problem_two (a b : ℝ) (cosA sinB : ℝ) : ℝ :=
if a = sqrt 6 ∧ b = 2 * 1 ∧ cosA = -1/4 ∧ sinB = sqrt 10 / 4 then sqrt 10 / 4 else 0

theorem solution_two (a b : ℝ) (cosA sinB : ℝ) :
  a = sqrt 6 → b = 2 * 1 → cosA = -1/4 → sinB = sqrt 10 / 4 → problem_two a b cosA sinB = sqrt 10 / 4 := by
  intros ha hb hcos hsinB
  unfold problem_two
  simp [ha, hb, hcos, hsinB]

noncomputable def problem_three (a b : ℝ) (cosA sinB sin2AminusB : ℝ) : ℝ :=
if a = sqrt 6 ∧ b = 2 * 1 ∧ cosA = -1/4 ∧ sinB = sqrt 10 / 4 ∧ sin2AminusB = sqrt 10 / 8 then sqrt 10 / 8 else 0

theorem solution_three (a b : ℝ) (cosA sinB sin2AminusB : ℝ) :
  a = sqrt 6 → b = 2 * 1 → cosA = -1/4 → sinB = sqrt 10 / 4 → sin2AminusB = sqrt 10 / 8 → problem_three a b cosA sinB sin2AminusB = sqrt 10 / 8 := by
  intros ha hb hcos hsinB hsin2AminusB
  unfold problem_three
  simp [ha, hb, hcos, hsinB, hsin2AminusB]

end NUMINAMATH_GPT_solution_one_solution_two_solution_three_l2348_234857


namespace NUMINAMATH_GPT_plant_lamp_arrangements_l2348_234841

/-- Rachel has two identical basil plants and an aloe plant.
Additionally, she has two identical white lamps, two identical red lamps, and 
two identical blue lamps she can put each plant under 
(she can put more than one plant under a lamp, but each plant is under exactly one lamp). 
-/
theorem plant_lamp_arrangements : 
  let plants := ["basil", "basil", "aloe"]
  let lamps := ["white", "white", "red", "red", "blue", "blue"]
  ∃ n, n = 27 := by
  sorry

end NUMINAMATH_GPT_plant_lamp_arrangements_l2348_234841


namespace NUMINAMATH_GPT_range_of_m_l2348_234897

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x ≤ -1 → ((m^2 - m) * 4^x - 2^x < 0)) → (-1 < m ∧ m < 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2348_234897


namespace NUMINAMATH_GPT_julia_more_kids_on_Monday_l2348_234888

def kids_played_on_Tuesday : Nat := 14
def kids_played_on_Monday : Nat := 22

theorem julia_more_kids_on_Monday : kids_played_on_Monday - kids_played_on_Tuesday = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_julia_more_kids_on_Monday_l2348_234888


namespace NUMINAMATH_GPT_olivia_grocery_cost_l2348_234870

theorem olivia_grocery_cost :
  let cost_bananas := 12
  let cost_bread := 9
  let cost_milk := 7
  let cost_apples := 14
  cost_bananas + cost_bread + cost_milk + cost_apples = 42 :=
by
  rfl

end NUMINAMATH_GPT_olivia_grocery_cost_l2348_234870


namespace NUMINAMATH_GPT_greater_segment_difference_l2348_234825

theorem greater_segment_difference :
  ∀ (L1 L2 : ℝ), L1 = 7 ∧ L1^2 - L2^2 = 32 → L1 - L2 = 7 - Real.sqrt 17 :=
by
  intros L1 L2 h
  sorry

end NUMINAMATH_GPT_greater_segment_difference_l2348_234825


namespace NUMINAMATH_GPT_larger_cube_volume_is_512_l2348_234880

def original_cube_volume := 64 -- volume in cubic feet
def scale_factor := 2 -- the factor by which the dimensions are scaled

def side_length (volume : ℕ) : ℕ := volume^(1/3) -- Assuming we have a function to compute cube root

def larger_cube_volume (original_volume : ℕ) (scale_factor : ℕ) : ℕ :=
  let original_side_length := side_length original_volume
  let larger_side_length := scale_factor * original_side_length
  larger_side_length ^ 3

theorem larger_cube_volume_is_512 :
  larger_cube_volume original_cube_volume scale_factor = 512 :=
sorry

end NUMINAMATH_GPT_larger_cube_volume_is_512_l2348_234880


namespace NUMINAMATH_GPT_number_of_tangent_small_circles_l2348_234800

-- Definitions from the conditions
def central_radius : ℝ := 2
def small_radius : ℝ := 1

-- The proof problem statement
theorem number_of_tangent_small_circles : 
  ∃ n : ℕ, (∀ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ k ≠ i → 
    dist (3 * central_radius) (3 * small_radius) = 3) ∧ n = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_tangent_small_circles_l2348_234800


namespace NUMINAMATH_GPT_myopia_relation_l2348_234869

def myopia_data := 
  [(1.00, 100), (0.50, 200), (0.25, 400), (0.20, 500), (0.10, 1000)]

noncomputable def myopia_function (x : ℝ) : ℝ :=
  100 / x

theorem myopia_relation (h₁ : 100 = (1.00 : ℝ) * 100)
    (h₂ : 100 = (0.50 : ℝ) * 200)
    (h₃ : 100 = (0.25 : ℝ) * 400)
    (h₄ : 100 = (0.20 : ℝ) * 500)
    (h₅ : 100 = (0.10 : ℝ) * 1000) :
  (∀ x > 0, myopia_function x = 100 / x) ∧ (myopia_function 250 = 0.4) :=
by
  sorry

end NUMINAMATH_GPT_myopia_relation_l2348_234869


namespace NUMINAMATH_GPT_rectangle_ratio_l2348_234832

theorem rectangle_ratio (t a b : ℝ) (h₀ : b = 2 * a) (h₁ : (t + 2 * a) ^ 2 = 3 * t ^ 2) : b / a = 2 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_ratio_l2348_234832


namespace NUMINAMATH_GPT_series_2023_power_of_3_squared_20_equals_653_l2348_234854

def series (A : ℕ → ℕ) : Prop :=
  A 0 = 1 ∧ 
  ∀ n > 0, 
  A n = A (n / 2023) + A (n / 2023^2) + A (n / 2023^3)

theorem series_2023_power_of_3_squared_20_equals_653 (A : ℕ → ℕ) (h : series A) : A (2023 ^ (3^2) + 20) = 653 :=
by
  -- placeholder for proof
  sorry

end NUMINAMATH_GPT_series_2023_power_of_3_squared_20_equals_653_l2348_234854


namespace NUMINAMATH_GPT_obtuse_angle_between_line_and_plane_l2348_234802

-- Define the problem conditions
def is_obtuse_angle (θ : ℝ) : Prop := θ > 90 ∧ θ < 180

-- Define what we are proving
theorem obtuse_angle_between_line_and_plane (θ : ℝ) (h1 : θ = angle_between_line_and_plane) :
  is_obtuse_angle θ :=
sorry

end NUMINAMATH_GPT_obtuse_angle_between_line_and_plane_l2348_234802


namespace NUMINAMATH_GPT_binomial_square_coefficients_l2348_234856

noncomputable def a : ℝ := 13.5
noncomputable def b : ℝ := 18

theorem binomial_square_coefficients (c d : ℝ) :
  (∀ x : ℝ, 6 * x ^ 2 + 18 * x + a = (c * x + d) ^ 2) ∧ 
  (∀ x : ℝ, 3 * x ^ 2 + b * x + 4 = (c * x + d) ^ 2)  → 
  a = 13.5 ∧ b = 18 := sorry

end NUMINAMATH_GPT_binomial_square_coefficients_l2348_234856


namespace NUMINAMATH_GPT_cat_weight_l2348_234892

theorem cat_weight 
  (weight1 weight2 : ℕ)
  (total_weight : ℕ)
  (h1 : weight1 = 2)
  (h2 : weight2 = 7)
  (h3 : total_weight = 13) : 
  ∃ weight3 : ℕ, weight3 = 4 := 
by
  sorry

end NUMINAMATH_GPT_cat_weight_l2348_234892


namespace NUMINAMATH_GPT_degree_of_product_l2348_234894

-- Definitions for the conditions
def isDegree (p : Polynomial ℝ) (n : ℕ) : Prop :=
  p.degree = n

variable {h j : Polynomial ℝ}

-- Given conditions
axiom h_deg : isDegree h 3
axiom j_deg : isDegree j 6

-- The theorem to prove
theorem degree_of_product : h.degree = 3 → j.degree = 6 → (Polynomial.degree (Polynomial.comp h (Polynomial.X ^ 4) * Polynomial.comp j (Polynomial.X ^ 3)) = 30) :=
by
  intros h3 j6
  sorry

end NUMINAMATH_GPT_degree_of_product_l2348_234894


namespace NUMINAMATH_GPT_find_m_l2348_234878

theorem find_m (m : ℝ) : 
  (m^2 + 3 * m + 3 ≠ 0) ∧ (m^2 + 2 * m - 3 ≠ 0) ∧ 
  (m^2 + 3 * m + 3 = 1) → m = -2 := 
by
  sorry

end NUMINAMATH_GPT_find_m_l2348_234878


namespace NUMINAMATH_GPT_find_y_l2348_234812

theorem find_y : ∃ y : ℚ, y + 2/3 = 1/4 - (2/5) * 2 ∧ y = -511/420 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l2348_234812


namespace NUMINAMATH_GPT_carrie_spent_l2348_234850

-- Define the cost of one t-shirt
def cost_per_tshirt : ℝ := 9.65

-- Define the number of t-shirts bought
def num_tshirts : ℝ := 12

-- Define the total cost function
def total_cost (cost_per_tshirt : ℝ) (num_tshirts : ℝ) : ℝ := cost_per_tshirt * num_tshirts

-- State the theorem which we need to prove
theorem carrie_spent :
  total_cost cost_per_tshirt num_tshirts = 115.80 :=
by
  sorry

end NUMINAMATH_GPT_carrie_spent_l2348_234850


namespace NUMINAMATH_GPT_cubic_sum_l2348_234885

theorem cubic_sum (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b + a * c + b * c = 1) (h3 : a * b * c = -2) : 
  a^3 + b^3 + c^3 = -6 :=
sorry

end NUMINAMATH_GPT_cubic_sum_l2348_234885


namespace NUMINAMATH_GPT_max_value_of_expression_l2348_234806

theorem max_value_of_expression
  (x y z : ℝ)
  (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
  11 * x + 3 * y + 8 * z ≤ 3.1925 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l2348_234806


namespace NUMINAMATH_GPT_combined_weight_of_candles_l2348_234838

theorem combined_weight_of_candles :
  (∀ (candles: ℕ), ∀ (weight_per_candle: ℕ),
    (weight_per_candle = 8 + 1) →
    (candles = 10 - 3) →
    (candles * weight_per_candle = 63)) :=
by
  intros
  sorry

end NUMINAMATH_GPT_combined_weight_of_candles_l2348_234838


namespace NUMINAMATH_GPT_complement_of_angle_l2348_234873

theorem complement_of_angle (α : ℝ) (h : α = 23 + 36 / 60) : 180 - α = 156.4 := 
by
  sorry

end NUMINAMATH_GPT_complement_of_angle_l2348_234873


namespace NUMINAMATH_GPT_infinite_slips_have_repeated_numbers_l2348_234851

theorem infinite_slips_have_repeated_numbers
  (slips : Set ℕ) (h_inf_slips : slips.Infinite)
  (h_sub_infinite_imp_repeats : ∀ s : Set ℕ, s.Infinite → ∃ x ∈ s, ∃ y ∈ s, x ≠ y ∧ x = y) :
  ∃ n : ℕ, {x ∈ slips | x = n}.Infinite :=
by sorry

end NUMINAMATH_GPT_infinite_slips_have_repeated_numbers_l2348_234851


namespace NUMINAMATH_GPT_min_blocks_for_wall_l2348_234828

-- Definitions based on conditions
def length_of_wall := 120
def height_of_wall := 6
def block_height := 1
def block_lengths := [1, 3]
def blocks_third_row := 3

-- Function to calculate the total blocks given the constraints from the conditions
noncomputable def min_blocks_needed : Nat := 164 + 80

-- Theorem assertion that the minimum number of blocks required is 244
theorem min_blocks_for_wall : min_blocks_needed = 244 := by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_min_blocks_for_wall_l2348_234828


namespace NUMINAMATH_GPT_Morse_code_distinct_symbols_l2348_234860

-- Morse code sequences conditions
def MorseCodeSequence (n : ℕ) := n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4

-- Total number of distinct symbols calculation
def total_distinct_symbols : ℕ :=
  2 + 4 + 8 + 16

-- The theorem to prove
theorem Morse_code_distinct_symbols : total_distinct_symbols = 30 := by
  sorry

end NUMINAMATH_GPT_Morse_code_distinct_symbols_l2348_234860


namespace NUMINAMATH_GPT_average_of_remaining_two_numbers_l2348_234844

theorem average_of_remaining_two_numbers
  (a b c d e f : ℝ)
  (h1 : (a + b + c + d + e + f) / 6 = 3.9)
  (h2 : (a + b) / 2 = 3.4)
  (h3 : (c + d) / 2 = 3.85) :
  (e + f) / 2 = 4.45 :=
sorry

end NUMINAMATH_GPT_average_of_remaining_two_numbers_l2348_234844


namespace NUMINAMATH_GPT_tan_ratio_l2348_234809

theorem tan_ratio (x : ℝ) (h : Real.tan (x + Real.pi / 4) = 2) : 
  Real.tan x / Real.tan (2 * x) = 4 / 9 :=
sorry

end NUMINAMATH_GPT_tan_ratio_l2348_234809


namespace NUMINAMATH_GPT_geometric_series_common_ratio_l2348_234875

theorem geometric_series_common_ratio (a r S : ℝ) (hS : S = a / (1 - r)) (hRemove : (ar^4) / (1 - r) = S / 81) :
  r = 1/3 :=
sorry

end NUMINAMATH_GPT_geometric_series_common_ratio_l2348_234875


namespace NUMINAMATH_GPT_biology_physics_ratio_l2348_234874

theorem biology_physics_ratio (boys_bio : ℕ) (girls_bio : ℕ) (total_bio : ℕ) (total_phys : ℕ) 
  (h1 : boys_bio = 25) 
  (h2 : girls_bio = 3 * boys_bio) 
  (h3 : total_bio = boys_bio + girls_bio) 
  (h4 : total_phys = 200) : 
  total_bio / total_phys = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_biology_physics_ratio_l2348_234874


namespace NUMINAMATH_GPT_balazs_missed_number_l2348_234849

theorem balazs_missed_number (n k : ℕ) 
  (h1 : n * (n + 1) / 2 = 3000 + k)
  (h2 : 1 ≤ k)
  (h3 : k < n) : k = 3 := by
  sorry

end NUMINAMATH_GPT_balazs_missed_number_l2348_234849


namespace NUMINAMATH_GPT_divisible_by_91_l2348_234805

theorem divisible_by_91 (n : ℕ) : 91 ∣ (5^n * (5^n + 1) - 6^n * (3^n + 2^n)) := 
by 
  sorry

end NUMINAMATH_GPT_divisible_by_91_l2348_234805


namespace NUMINAMATH_GPT_sum_of_first_six_terms_l2348_234847

theorem sum_of_first_six_terms 
  {S : ℕ → ℝ} 
  (h_arith_seq : ∀ n, S n = n * (-2) + (n * (n - 1) * 3 ))
  (S_2_eq_2 : S 2 = 2)
  (S_4_eq_10 : S 4 = 10) : S 6 = 18 := 
  sorry

end NUMINAMATH_GPT_sum_of_first_six_terms_l2348_234847


namespace NUMINAMATH_GPT_correct_calculation_l2348_234810

theorem correct_calculation (x : ℝ) : (-2 * x^2)^3 = -8 * x^6 :=
by sorry

end NUMINAMATH_GPT_correct_calculation_l2348_234810


namespace NUMINAMATH_GPT_odd_function_periodic_value_l2348_234834

noncomputable def f : ℝ → ℝ := sorry  -- Define f

theorem odd_function_periodic_value:
  (∀ x, f (-x) = - f x) →  -- f is odd
  (∀ x, f (x + 3) = f x) → -- f has period 3
  f 1 = 2014 →            -- given f(1) = 2014
  f 2013 + f 2014 + f 2015 = 0 := by
  intros h_odd h_period h_f1
  sorry

end NUMINAMATH_GPT_odd_function_periodic_value_l2348_234834


namespace NUMINAMATH_GPT_number_of_men_l2348_234839

theorem number_of_men (M : ℕ) (h : M * 40 = 20 * 68) : M = 34 :=
by
  sorry

end NUMINAMATH_GPT_number_of_men_l2348_234839


namespace NUMINAMATH_GPT_profit_calculation_l2348_234858

def actors_cost : ℕ := 1200
def people_count : ℕ := 50
def cost_per_person : ℕ := 3
def food_cost : ℕ := people_count * cost_per_person
def total_cost_actors_food : ℕ := actors_cost + food_cost
def equipment_rental_cost : ℕ := 2 * total_cost_actors_food
def total_movie_cost : ℕ := total_cost_actors_food + equipment_rental_cost
def movie_sale_price : ℕ := 10000
def profit : ℕ := movie_sale_price - total_movie_cost

theorem profit_calculation : profit = 5950 := by
  sorry

end NUMINAMATH_GPT_profit_calculation_l2348_234858


namespace NUMINAMATH_GPT_trapezoid_perimeter_l2348_234827

theorem trapezoid_perimeter
  (AB CD AD BC : ℝ)
  (h1 : AB = 5)
  (h2 : CD = 5)
  (h3 : AD = 16)
  (h4 : BC = 8) :
  AB + BC + CD + AD = 34 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_perimeter_l2348_234827


namespace NUMINAMATH_GPT_complete_set_of_events_l2348_234816

-- Define the range of numbers on a die
def die_range := {n : ℕ | 1 ≤ n ∧ n ≤ 6}

-- Define what an outcome is
def outcome := { p : ℕ × ℕ | p.1 ∈ die_range ∧ p.2 ∈ die_range }

-- The theorem stating the complete set of outcomes
theorem complete_set_of_events : outcome = { p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 6 ∧ 1 ≤ p.2 ∧ p.2 ≤ 6 } :=
by sorry

end NUMINAMATH_GPT_complete_set_of_events_l2348_234816


namespace NUMINAMATH_GPT_tangent_line_at_0_maximum_integer_value_of_a_l2348_234861

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (x + 1) - a*x + 2

-- Part (1)
-- Prove that the equation of the tangent line to f(x) at x = 0 is x + y - 2 = 0 when a = 2
theorem tangent_line_at_0 {a : ℝ} (h : a = 2) : ∀ x y : ℝ, (y = f x a) → (x = 0) → (y = 2 - x) :=
by 
  sorry

-- Part (2)
-- Prove that if f(x) + 2x + x log(x+1) ≥ 0 holds for all x ≥ 0, then the maximum integer value of a is 4
theorem maximum_integer_value_of_a 
  (h : ∀ x : ℝ, x ≥ 0 → f x a + 2 * x + x * Real.log (x + 1) ≥ 0) : a ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_0_maximum_integer_value_of_a_l2348_234861


namespace NUMINAMATH_GPT_five_cds_cost_with_discount_l2348_234867

theorem five_cds_cost_with_discount
  (price_2_cds : ℝ)
  (discount_rate : ℝ)
  (num_cds : ℕ)
  (total_cost : ℝ) 
  (h1 : price_2_cds = 40)
  (h2 : discount_rate = 0.10)
  (h3 : num_cds = 5)
  : total_cost = 90 :=
by
  sorry

end NUMINAMATH_GPT_five_cds_cost_with_discount_l2348_234867


namespace NUMINAMATH_GPT_seashells_in_six_weeks_l2348_234826

def jar_weekly_update (week : Nat) (jarA : Nat) (jarB : Nat) : Nat × Nat :=
  if week % 3 = 0 then (jarA / 2, jarB / 2)
  else (jarA + 20, jarB * 2)

def total_seashells_after_weeks (initialA : Nat) (initialB : Nat) (weeks : Nat) : Nat :=
  let rec update (w : Nat) (jA : Nat) (jB : Nat) :=
    match w with
    | 0 => jA + jB
    | n + 1 =>
      let (newA, newB) := jar_weekly_update n jA jB
      update n newA newB
  update weeks initialA initialB

theorem seashells_in_six_weeks :
  total_seashells_after_weeks 50 30 6 = 97 :=
sorry

end NUMINAMATH_GPT_seashells_in_six_weeks_l2348_234826


namespace NUMINAMATH_GPT_frac_eval_eq_l2348_234887

theorem frac_eval_eq :
  let a := 19
  let b := 8
  let c := 35
  let d := 19 * 8 / 35
  ( (⌈a / b - ⌈c / d⌉⌉) / ⌈c / b + ⌈d⌉⌉) = (1 / 10) := by
  sorry

end NUMINAMATH_GPT_frac_eval_eq_l2348_234887


namespace NUMINAMATH_GPT_solve_fiftieth_term_l2348_234845

variable (a₇ a₂₁ : ℤ) (d : ℚ)

-- The conditions stated in the problem
def seventh_term : a₇ = 10 := by sorry
def twenty_first_term : a₂₁ = 34 := by sorry

-- The fifty term calculation assuming the common difference d
def fiftieth_term_is_fraction (d : ℚ) : ℚ := 10 + 43 * d

-- Translate the condition a₂₁ = a₇ + 14 * d
theorem solve_fiftieth_term : a₂₁ = a₇ + 14 * d → 
                              fiftieth_term_is_fraction d = 682 / 7 := by sorry


end NUMINAMATH_GPT_solve_fiftieth_term_l2348_234845


namespace NUMINAMATH_GPT_evaluate_expression_l2348_234859

theorem evaluate_expression :
  let a := 1
  let b := 10
  let c := 100
  let d := 1000
  (a + b + c - d) + (a + b - c + d) + (a - b + c + d) + (-a + b + c + d) = 2222 :=
by
  let a := 1
  let b := 10
  let c := 100
  let d := 1000
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2348_234859


namespace NUMINAMATH_GPT_problem1_problem2_prob_dist_problem2_expectation_l2348_234824

noncomputable def probability_A_wins_match_B_wins_once (pA pB : ℚ) : ℚ :=
  (pB * pA * pA) + (pA * pB * pA * pA)

theorem problem1 : probability_A_wins_match_B_wins_once (2/3) (1/3) = 20/81 :=
  by sorry

noncomputable def P_X (x : ℕ) (pA pB : ℚ) : ℚ :=
  match x with
  | 2 => pA^2 + pB^2
  | 3 => pB * pA^2 + pA * pB^2
  | 4 => (pA * pB * pA * pA) + (pB * pA * pB * pB)
  | 5 => (pB * pA * pB * pA) + (pA * pB * pA * pB)
  | _ => 0

theorem problem2_prob_dist : 
  P_X 2 (2/3) (1/3) = 5/9 ∧
  P_X 3 (2/3) (1/3) = 2/9 ∧
  P_X 4 (2/3) (1/3) = 10/81 ∧
  P_X 5 (2/3) (1/3) = 8/81 :=
  by sorry

noncomputable def E_X (pA pB : ℚ) : ℚ :=
  2 * (P_X 2 pA pB) + 3 * (P_X 3 pA pB) + 
  4 * (P_X 4 pA pB) + 5 * (P_X 5 pA pB)

theorem problem2_expectation : E_X (2/3) (1/3) = 224/81 :=
  by sorry

end NUMINAMATH_GPT_problem1_problem2_prob_dist_problem2_expectation_l2348_234824


namespace NUMINAMATH_GPT_daniel_sales_tax_l2348_234863

theorem daniel_sales_tax :
  let total_cost := 25
  let tax_rate := 0.05
  let tax_free_cost := 18.7
  let tax_paid := 0.3
  exists (taxable_cost : ℝ), 
    18.7 + taxable_cost + 0.05 * taxable_cost = total_cost ∧
    taxable_cost * tax_rate = tax_paid :=
by
  sorry

end NUMINAMATH_GPT_daniel_sales_tax_l2348_234863


namespace NUMINAMATH_GPT_no_common_multiples_of_3_l2348_234886

-- Define the sets X and Y
def SetX : Set ℤ := {n | 1 ≤ n ∧ n ≤ 24 ∧ n % 2 = 1}
def SetY : Set ℤ := {n | 0 ≤ n ∧ n ≤ 40 ∧ n % 2 = 0}

-- Define the condition for being a multiple of 3
def isMultipleOf3 (n : ℤ) : Prop := n % 3 = 0

-- Define the intersection of SetX and SetY that are multiples of 3
def intersectionMultipleOf3 : Set ℤ := {n | n ∈ SetX ∧ n ∈ SetY ∧ isMultipleOf3 n}

-- Prove that the set is empty
theorem no_common_multiples_of_3 : intersectionMultipleOf3 = ∅ := by
  sorry

end NUMINAMATH_GPT_no_common_multiples_of_3_l2348_234886


namespace NUMINAMATH_GPT_circle_area_from_equation_l2348_234881

theorem circle_area_from_equation :
  (∀ x y : ℝ, x^2 + y^2 - 4 * x + 6 * y = -9) →
  ∃ (r : ℝ), (r = 2) ∧
    (∃ (A : ℝ), A = π * r^2 ∧ A = 4 * π) :=
by {
  -- Conditions included as hypothesis
  sorry -- Proof to be provided here
}

end NUMINAMATH_GPT_circle_area_from_equation_l2348_234881


namespace NUMINAMATH_GPT_bag_of_food_costs_two_dollars_l2348_234895

theorem bag_of_food_costs_two_dollars
  (cost_puppy : ℕ)
  (total_cost : ℕ)
  (daily_food : ℚ)
  (bag_food_quantity : ℚ)
  (weeks : ℕ)
  (h1 : cost_puppy = 10)
  (h2 : total_cost = 14)
  (h3 : daily_food = 1/3)
  (h4 : bag_food_quantity = 3.5)
  (h5 : weeks = 3) :
  (total_cost - cost_puppy) / (21 * daily_food / bag_food_quantity) = 2 := 
  by sorry

end NUMINAMATH_GPT_bag_of_food_costs_two_dollars_l2348_234895


namespace NUMINAMATH_GPT_sum_of_multiples_of_4_between_34_and_135_l2348_234815

theorem sum_of_multiples_of_4_between_34_and_135 :
  let first := 36
  let last := 132
  let n := (last - first) / 4 + 1
  let sum := n * (first + last) / 2
  sum = 2100 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_multiples_of_4_between_34_and_135_l2348_234815


namespace NUMINAMATH_GPT_binomial_p_value_l2348_234865

noncomputable def binomial_expected_value (n : ℕ) (p : ℝ) : ℝ := n * p

theorem binomial_p_value (p : ℝ) : (binomial_expected_value 18 p = 9) → p = 1/2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_binomial_p_value_l2348_234865


namespace NUMINAMATH_GPT_wrapping_paper_area_correct_l2348_234882

variable (w h : ℝ) -- Define the base length and height of the box.

-- Lean statement for the problem asserting that the area of the wrapping paper is \(2(w+h)^2\).
def wrapping_paper_area (w h : ℝ) : ℝ := 2 * (w + h) ^ 2

-- Theorem stating that the derived formula for the area of the wrapping paper is correct.
theorem wrapping_paper_area_correct (w h : ℝ) : wrapping_paper_area w h = 2 * (w + h) ^ 2 :=
by
  sorry -- Proof is omitted

end NUMINAMATH_GPT_wrapping_paper_area_correct_l2348_234882


namespace NUMINAMATH_GPT_value_of_x_minus_y_squared_l2348_234808

theorem value_of_x_minus_y_squared (x y : ℝ) (hx : x^2 = 4) (hy : y^2 = 9) : 
  ((x - y)^2 = 1) ∨ ((x - y)^2 = 25) :=
sorry

end NUMINAMATH_GPT_value_of_x_minus_y_squared_l2348_234808


namespace NUMINAMATH_GPT_prism_edges_l2348_234814

theorem prism_edges (V F E n : ℕ) (h1 : V + F + E = 44) (h2 : V = 2 * n) (h3 : F = n + 2) (h4 : E = 3 * n) : E = 21 := by
  sorry

end NUMINAMATH_GPT_prism_edges_l2348_234814


namespace NUMINAMATH_GPT_number_of_students_l2348_234848

theorem number_of_students (S G : ℕ) (h1 : G = 2 * S / 3) (h2 : 8 = 2 * G / 5) : S = 30 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_l2348_234848


namespace NUMINAMATH_GPT_range_of_t_l2348_234821

noncomputable def f (t : ℝ) (x : ℝ) : ℝ :=
if x ≤ 0 then x^2 + 2 * t * x + t^2 else x + 1 / x + t

theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f t 0 ≤ f t x) ↔ (0 ≤ t ∧ t ≤ 2) :=
by sorry

end NUMINAMATH_GPT_range_of_t_l2348_234821


namespace NUMINAMATH_GPT_math_books_count_l2348_234817

theorem math_books_count (M H : ℕ) :
  M + H = 90 →
  4 * M + 5 * H = 396 →
  H = 90 - M →
  M = 54 :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_math_books_count_l2348_234817


namespace NUMINAMATH_GPT_sum_inverse_terms_l2348_234843

theorem sum_inverse_terms : 
  (∑' n : ℕ, if n = 0 then (0 : ℝ) else (1 / (n * (n + 3) : ℝ))) = 11 / 18 :=
by {
  -- proof to be filled in
  sorry
}

end NUMINAMATH_GPT_sum_inverse_terms_l2348_234843


namespace NUMINAMATH_GPT_factor_poly_PQ_sum_l2348_234829

theorem factor_poly_PQ_sum (P Q : ℝ) (h : (∀ x : ℝ, (x^2 + 3 * x + 4) * (x^2 + -3 * x + 4) = x^4 + P * x^2 + Q)) : P + Q = 15 :=
by
  sorry

end NUMINAMATH_GPT_factor_poly_PQ_sum_l2348_234829


namespace NUMINAMATH_GPT_minNumberOfGloves_l2348_234866

-- Define the number of participants
def numParticipants : ℕ := 43

-- Define the number of gloves needed per participant
def glovesPerParticipant : ℕ := 2

-- Define the total number of gloves
def totalGloves (participants glovesPerParticipant : ℕ) : ℕ := 
  participants * glovesPerParticipant

-- Theorem proving the minimum number of gloves required
theorem minNumberOfGloves : totalGloves numParticipants glovesPerParticipant = 86 :=
by
  sorry

end NUMINAMATH_GPT_minNumberOfGloves_l2348_234866


namespace NUMINAMATH_GPT_find_number_l2348_234899

theorem find_number (x : ℝ) (h : x - x / 3 = x - 24) : x = 72 := 
by 
  sorry

end NUMINAMATH_GPT_find_number_l2348_234899


namespace NUMINAMATH_GPT_Shara_shells_total_l2348_234801

def initial_shells : ℕ := 20
def shells_per_day : ℕ := 5
def days : ℕ := 3
def shells_fourth_day : ℕ := 6

theorem Shara_shells_total : (initial_shells + (shells_per_day * days) + shells_fourth_day) = 41 := by
  sorry

end NUMINAMATH_GPT_Shara_shells_total_l2348_234801


namespace NUMINAMATH_GPT_tangent_points_l2348_234871

theorem tangent_points (x y : ℝ) (h : y = x^3 - 3 * x) (slope_zero : 3 * x^2 - 3 = 0) :
  (x = -1 ∧ y = 2) ∨ (x = 1 ∧ y = -2) :=
sorry

end NUMINAMATH_GPT_tangent_points_l2348_234871


namespace NUMINAMATH_GPT_blocks_difference_l2348_234889

def blocks_house := 89
def blocks_tower := 63

theorem blocks_difference : (blocks_house - blocks_tower = 26) :=
by sorry

end NUMINAMATH_GPT_blocks_difference_l2348_234889


namespace NUMINAMATH_GPT_exsphere_identity_l2348_234811

-- Given definitions for heights and radii
variables {h1 h2 h3 h4 r1 r2 r3 r4 : ℝ}

-- Definition of the relationship that needs to be proven
theorem exsphere_identity 
  (h1 h2 h3 h4 r1 r2 r3 r4 : ℝ) :
  2 * (1 / h1 + 1 / h2 + 1 / h3 + 1 / h4) = 1 / r1 + 1 / r2 + 1 / r3 + 1 / r4 := 
sorry

end NUMINAMATH_GPT_exsphere_identity_l2348_234811


namespace NUMINAMATH_GPT_broadcasting_methods_count_l2348_234823

-- Defining the given conditions
def num_commercials : ℕ := 4 -- number of different commercial advertisements
def num_psa : ℕ := 2 -- number of different public service advertisements
def total_slots : ℕ := 6 -- total number of slots for commercials

-- The assertion we want to prove
theorem broadcasting_methods_count : 
  (num_psa * (total_slots - num_commercials - 1) * (num_commercials.factorial)) = 48 :=
by sorry

end NUMINAMATH_GPT_broadcasting_methods_count_l2348_234823


namespace NUMINAMATH_GPT_find_dividend_l2348_234804

noncomputable def divisor := (-14 : ℚ) / 3
noncomputable def quotient := (-286 : ℚ) / 5
noncomputable def remainder := (19 : ℚ) / 9
noncomputable def dividend := 269 + (2 / 45 : ℚ)

theorem find_dividend :
  dividend = (divisor * quotient) + remainder := by
  sorry

end NUMINAMATH_GPT_find_dividend_l2348_234804


namespace NUMINAMATH_GPT_Ruby_apples_remaining_l2348_234862

def Ruby_original_apples : ℕ := 6357912
def Emily_takes_apples : ℕ := 2581435
def Ruby_remaining_apples (R E : ℕ) : ℕ := R - E

theorem Ruby_apples_remaining : Ruby_remaining_apples Ruby_original_apples Emily_takes_apples = 3776477 := by
  sorry

end NUMINAMATH_GPT_Ruby_apples_remaining_l2348_234862


namespace NUMINAMATH_GPT_fran_threw_away_80_pct_l2348_234803

-- Definitions based on the conditions
def initial_votes_game_of_thrones := 10
def initial_votes_twilight := 12
def initial_votes_art_of_deal := 20
def altered_votes_twilight := initial_votes_twilight / 2
def new_total_votes := 2 * initial_votes_game_of_thrones

-- Theorem we are proving
theorem fran_threw_away_80_pct :
  ∃ x, x = 80 ∧
    new_total_votes = initial_votes_game_of_thrones + altered_votes_twilight + (initial_votes_art_of_deal * (1 - x / 100)) := by
  sorry

end NUMINAMATH_GPT_fran_threw_away_80_pct_l2348_234803


namespace NUMINAMATH_GPT_betty_age_l2348_234807

theorem betty_age (A M B : ℕ) (h1 : A = 2 * M) (h2 : A = 4 * B) (h3 : M = A - 22) : B = 11 :=
by
  sorry

end NUMINAMATH_GPT_betty_age_l2348_234807


namespace NUMINAMATH_GPT_chocolates_bought_in_a_month_l2348_234890

theorem chocolates_bought_in_a_month :
  ∀ (chocolates_for_her: ℕ)
    (chocolates_for_sister: ℕ)
    (chocolates_for_charlie: ℕ)
    (weeks_in_a_month: ℕ), 
  weeks_in_a_month = 4 →
  chocolates_for_her = 2 →
  chocolates_for_sister = 1 →
  chocolates_for_charlie = 10 →
  (chocolates_for_her * weeks_in_a_month + chocolates_for_sister * weeks_in_a_month + chocolates_for_charlie) = 22 :=
by
  intros chocolates_for_her chocolates_for_sister chocolates_for_charlie weeks_in_a_month
  intros h_weeks h_her h_sister h_charlie
  sorry

end NUMINAMATH_GPT_chocolates_bought_in_a_month_l2348_234890


namespace NUMINAMATH_GPT_total_clothing_l2348_234840

def num_boxes : ℕ := 4
def scarves_per_box : ℕ := 2
def mittens_per_box : ℕ := 6

theorem total_clothing :
  num_boxes * scarves_per_box + num_boxes * mittens_per_box = 32 :=
by
  sorry

end NUMINAMATH_GPT_total_clothing_l2348_234840


namespace NUMINAMATH_GPT_chloe_boxes_of_clothing_l2348_234836

theorem chloe_boxes_of_clothing (total_clothing pieces_per_box : ℕ) (h1 : total_clothing = 32) (h2 : pieces_per_box = 2 + 6) :
  ∃ B : ℕ, B = total_clothing / pieces_per_box ∧ B = 4 :=
by
  -- Proof can be filled in here
   sorry

end NUMINAMATH_GPT_chloe_boxes_of_clothing_l2348_234836


namespace NUMINAMATH_GPT_pages_read_first_day_l2348_234830

-- Alexa is reading a Nancy Drew mystery with 95 pages.
def total_pages : ℕ := 95

-- She read 58 pages the next day.
def pages_read_second_day : ℕ := 58

-- She has 19 pages left to read.
def pages_left_to_read : ℕ := 19

-- How many pages did she read on the first day?
theorem pages_read_first_day : total_pages - pages_read_second_day - pages_left_to_read = 18 := by
  -- Proof is omitted as instructed
  sorry

end NUMINAMATH_GPT_pages_read_first_day_l2348_234830
