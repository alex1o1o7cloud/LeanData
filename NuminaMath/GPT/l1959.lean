import Mathlib

namespace candies_indeterminable_l1959_195902

theorem candies_indeterminable
  (num_bags : ℕ) (cookies_per_bag : ℕ) (total_cookies : ℕ) (known_candies : ℕ) :
  num_bags = 26 →
  cookies_per_bag = 2 →
  total_cookies = 52 →
  num_bags * cookies_per_bag = total_cookies →
  ∀ (candies : ℕ), candies = known_candies → false :=
by
  intros
  sorry

end candies_indeterminable_l1959_195902


namespace price_per_rose_is_2_l1959_195922

-- Definitions from conditions
def has_amount (total_dollars : ℕ) : Prop := total_dollars = 300
def total_roses (R : ℕ) : Prop := ∃ (j : ℕ) (i : ℕ), R / 3 = j ∧ R / 2 = i ∧ j + i = 125

-- Theorem stating the price per rose
theorem price_per_rose_is_2 (R : ℕ) : 
  has_amount 300 → total_roses R → 300 / R = 2 :=
sorry

end price_per_rose_is_2_l1959_195922


namespace dot_product_zero_l1959_195965

-- Define vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (4, 3)

-- Define the dot product operation for two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the scalar multiplication and vector subtraction for 2D vectors
def scalar_mul_vec (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (c * v.1, c * v.2)

def vec_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

-- Now we state the theorem we want to prove
theorem dot_product_zero : dot_product a (vec_sub (scalar_mul_vec 2 a) b) = 0 := 
by
  sorry

end dot_product_zero_l1959_195965


namespace right_triangle_perimeter_l1959_195981

theorem right_triangle_perimeter 
  (a b c : ℕ) (h : a = 11) (h1 : a * a + b * b = c * c) (h2 : a < c) : a + b + c = 132 :=
  sorry

end right_triangle_perimeter_l1959_195981


namespace heaviest_object_can_be_weighed_is_40_number_of_different_weights_is_40_l1959_195995

def weights : List ℕ := [1, 3, 9, 27]

theorem heaviest_object_can_be_weighed_is_40 : 
  List.sum weights = 40 :=
by
  sorry

theorem number_of_different_weights_is_40 :
  List.range (List.sum weights) = List.range 40 :=
by
  sorry

end heaviest_object_can_be_weighed_is_40_number_of_different_weights_is_40_l1959_195995


namespace cos_double_angle_l1959_195953

theorem cos_double_angle (α : ℝ) (h : Real.sin α = 3 / 5) : Real.cos (2 * α) = 7 / 25 :=
sorry

end cos_double_angle_l1959_195953


namespace calculate_constants_l1959_195960

noncomputable def parabola_tangent_to_line (a b : ℝ) : Prop :=
  let discriminant := (b - 2) ^ 2 + 28 * a
  discriminant = 0

theorem calculate_constants
  (a b : ℝ)
  (h_tangent : parabola_tangent_to_line a b) :
  a = -((b - 2) ^ 2) / 28 ∧ b ≠ 2 :=
by
  sorry

end calculate_constants_l1959_195960


namespace permutations_sum_divisible_by_37_l1959_195973

theorem permutations_sum_divisible_by_37 (a b c : ℕ) (ha : 0 ≤ a ∧ a ≤ 9)
  (hb : 0 ≤ b ∧ b ≤ 9) (hc : 0 ≤ c ∧ c ≤ 9) :
    ∃ k, (100 * a + 10 * b + c) + (100 * a + 10 * c + b) + (100 * b + 10 * a + c) + (100 * b + 10 * c + a) + (100 * c + 10 * a + b) + (100 * c + 10 * b + a) = 37 * k := 
by
  sorry

end permutations_sum_divisible_by_37_l1959_195973


namespace ball_hits_ground_at_2_72_l1959_195925

-- Define the initial conditions
def initial_velocity (v₀ : ℝ) := v₀ = 30
def initial_height (h₀ : ℝ) := h₀ = 200
def ball_height (t : ℝ) : ℝ := -16 * t^2 - 30 * t + 200

-- Prove that the ball hits the ground at t = 2.72 seconds
theorem ball_hits_ground_at_2_72 (t : ℝ) (h : ℝ) 
  (v₀ : ℝ) (h₀ : ℝ) 
  (hv₀ : initial_velocity v₀) 
  (hh₀ : initial_height h₀)
  (h_eq: ball_height t = h) 
  (h₀_eq: ball_height 0 = h₀) : 
  h = 0 -> t = 2.72 :=
by
  sorry

end ball_hits_ground_at_2_72_l1959_195925


namespace scientific_notation_35_million_l1959_195913

theorem scientific_notation_35_million :
  35000000 = 3.5 * (10 : Float) ^ 7 := 
by
  sorry

end scientific_notation_35_million_l1959_195913


namespace students_playing_both_football_and_cricket_l1959_195967

theorem students_playing_both_football_and_cricket
  (total_students : ℕ)
  (students_playing_football : ℕ)
  (students_playing_cricket : ℕ)
  (students_neither_football_nor_cricket : ℕ) :
  total_students = 250 →
  students_playing_football = 160 →
  students_playing_cricket = 90 →
  students_neither_football_nor_cricket = 50 →
  (students_playing_football + students_playing_cricket - (total_students - students_neither_football_nor_cricket)) = 50 :=
by
  intros h_total h_football h_cricket h_neither
  sorry

end students_playing_both_football_and_cricket_l1959_195967


namespace range_of_function_l1959_195929

noncomputable def function_y (x : ℝ) : ℝ := -x^2 - 2 * x + 3

theorem range_of_function : 
  ∃ (a b : ℝ), a = -12 ∧ b = 4 ∧ 
  (∀ y, (∃ x, -5 ≤ x ∧ x ≤ 0 ∧ y = function_y x) ↔ a ≤ y ∧ y ≤ b) :=
sorry

end range_of_function_l1959_195929


namespace smallest_integer_k_l1959_195999

theorem smallest_integer_k (k : ℕ) : 
  (k > 1 ∧ 
   k % 13 = 1 ∧ 
   k % 7 = 1 ∧ 
   k % 5 = 1 ∧ 
   k % 3 = 1) ↔ k = 1366 := 
sorry

end smallest_integer_k_l1959_195999


namespace A_inter_B_is_empty_l1959_195972

def A : Set (ℤ × ℤ) := {p | ∃ x : ℤ, p = (x, x + 1)}
def B : Set ℤ := {y | ∃ x : ℤ, y = 2 * x}

theorem A_inter_B_is_empty : A ∩ (fun p => p.2 ∈ B) = ∅ :=
by {
  sorry
}

end A_inter_B_is_empty_l1959_195972


namespace sum_of_remainders_l1959_195962

theorem sum_of_remainders (a b c d e : ℕ)
  (h1 : a % 13 = 3)
  (h2 : b % 13 = 5)
  (h3 : c % 13 = 7)
  (h4 : d % 13 = 9)
  (h5 : e % 13 = 11) :
  (a + b + c + d + e) % 13 = 9 :=
by {
  sorry
}

end sum_of_remainders_l1959_195962


namespace cubic_eq_one_real_root_l1959_195923

-- Given a, b, c forming a geometric sequence
variables {a b c : ℝ}

-- Definition of a geometric sequence
def geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

-- Equation ax^3 + bx^2 + cx = 0
def cubic_eq (a b c x : ℝ) : Prop :=
  a * x^3 + b * x^2 + c * x = 0

-- Prove the number of real roots
theorem cubic_eq_one_real_root (h : geometric_sequence a b c) :
  ∃ x : ℝ, cubic_eq a b c x ∧ ¬∃ y ≠ x, cubic_eq a b c y :=
sorry

end cubic_eq_one_real_root_l1959_195923


namespace number_of_hens_l1959_195916

-- Let H be the number of hens and C be the number of cows
def hens_and_cows (H C : Nat) : Prop :=
  H + C = 50 ∧ 2 * H + 4 * C = 144

theorem number_of_hens : ∃ H C : Nat, hens_and_cows H C ∧ H = 28 :=
by
  -- The proof is omitted
  sorry

end number_of_hens_l1959_195916


namespace q_at_4_l1959_195978

def q (x : ℝ) : ℝ := |x - 3|^(1/3) + 3 * |x - 3|^(1/5) + 2 

theorem q_at_4 : q 4 = 6 := by
  sorry

end q_at_4_l1959_195978


namespace domain_of_function_l1959_195932

def domain_of_f (x : ℝ) : Prop :=
  (x ≤ 2) ∧ (x ≠ 1)

theorem domain_of_function :
  ∀ x : ℝ, x ∈ { x | (x ≤ 2) ∧ (x ≠ 1) } ↔ domain_of_f x :=
by
  sorry

end domain_of_function_l1959_195932


namespace intersection_of_circle_and_line_in_polar_coordinates_l1959_195997

noncomputable section

def circle_polar_eq (ρ θ : ℝ) : Prop := ρ = Real.cos θ + Real.sin θ
def line_polar_eq (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - Real.pi / 4) = Real.sqrt 2 / 2

theorem intersection_of_circle_and_line_in_polar_coordinates :
  ∀ θ ρ, (0 < θ ∧ θ < Real.pi) →
  circle_polar_eq ρ θ →
  line_polar_eq ρ θ →
  ρ = 1 ∧ θ = Real.pi / 2 :=
by
  sorry

end intersection_of_circle_and_line_in_polar_coordinates_l1959_195997


namespace simplify_and_calculate_expression_l1959_195931

variable (a b : ℤ)

theorem simplify_and_calculate_expression (h_a : a = -3) (h_b : b = -2) :
  (a + b) * (b - a) + (2 * a^2 * b - a^3) / (-a) = -8 :=
by
  -- We include the proof steps here to achieve the final result.
  sorry

end simplify_and_calculate_expression_l1959_195931


namespace find_first_5digits_of_M_l1959_195954

def last6digits (n : ℕ) : ℕ := n % 1000000

def first5digits (n : ℕ) : ℕ := n / 10

theorem find_first_5digits_of_M (M : ℕ) (h1 : last6digits M = last6digits (M^2)) (h2 : M > 999999) : first5digits M = 60937 := 
by sorry

end find_first_5digits_of_M_l1959_195954


namespace intersection_complement_l1959_195900

open Set

variable (U : Set ℕ) (A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5}) (hA : A = {2, 3, 4}) (hB : B = {1, 2})

theorem intersection_complement :
  A ∩ (U \ B) = {3, 4} :=
by
  rw [hU, hA, hB]
  sorry

end intersection_complement_l1959_195900


namespace general_formula_sum_first_n_terms_l1959_195982

-- Definitions for arithmetic sequence, geometric aspects and sum conditions 
variable {a_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}
variable {d : ℕ}
variable {b_n : ℕ → ℕ}
variable {T_n : ℕ → ℕ}

-- Given conditions
axiom sum_condition (S3 S5 : ℕ) : S3 + S5 = 50
axiom common_difference : d ≠ 0
axiom first_term (a1 : ℕ) : a_n 1 = a1
axiom geometric_conditions (a1 a4 a13 : ℕ)
  (h1 : a_n 1 = a1) (h4 : a_n 4 = a4) (h13 : a_n 13 = a13) :
  a4 = a1 + 3 * d ∧ a13 = a1 + 12 * d ∧ (a4 ^ 2 = a1 * a13)

-- Proving the general formula for a_n
theorem general_formula (a_n : ℕ → ℕ)
  (h : ∀ (n : ℕ), a_n n = 2 * n + 1) : 
  a_n n = 2 * n + 1 := 
sorry

-- Proving the sum of the first n terms of sequence {b_n}
theorem sum_first_n_terms (a_n b_n : ℕ → ℕ) (T_n : ℕ → ℕ)
  (h_bn : ∀ (n : ℕ), b_n n = (2 * n + 1) * 2 ^ (n - 1))
  (h_Tn: ∀ (n : ℕ), T_n n = 1 + (2 * n - 1) * 2^n) :
  T_n n = 1 + (2 * n - 1) * 2^n :=
sorry

end general_formula_sum_first_n_terms_l1959_195982


namespace population_of_males_l1959_195974

theorem population_of_males (total_population : ℕ) (num_parts : ℕ) (part_population : ℕ) 
  (male_population : ℕ) (female_population : ℕ) (children_population : ℕ) :
  total_population = 600 →
  num_parts = 4 →
  part_population = total_population / num_parts →
  children_population = 2 * male_population →
  male_population = part_population →
  male_population = 150 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end population_of_males_l1959_195974


namespace april_revenue_l1959_195937

def revenue_after_tax (initial_roses : ℕ) (initial_tulips : ℕ) (initial_daisies : ℕ)
                      (final_roses : ℕ) (final_tulips : ℕ) (final_daisies : ℕ)
                      (price_rose : ℝ) (price_tulip : ℝ) (price_daisy : ℝ) (tax_rate : ℝ) : ℝ :=
(price_rose * (initial_roses - final_roses) + price_tulip * (initial_tulips - final_tulips) + price_daisy * (initial_daisies - final_daisies)) * (1 + tax_rate)

theorem april_revenue :
  revenue_after_tax 13 10 8 4 3 1 4 3 2 0.10 = 78.10 := by
  sorry

end april_revenue_l1959_195937


namespace total_albums_l1959_195909

-- Defining the initial conditions
def albumsAdele : ℕ := 30
def albumsBridget : ℕ := albumsAdele - 15
def albumsKatrina : ℕ := 6 * albumsBridget
def albumsMiriam : ℕ := 7 * albumsKatrina
def albumsCarlos : ℕ := 3 * albumsMiriam
def albumsDiane : ℕ := 2 * albumsKatrina

-- Proving the total number of albums
theorem total_albums :
  albumsAdele + albumsBridget + albumsKatrina + albumsMiriam + albumsCarlos + albumsDiane = 2835 :=
by
  sorry

end total_albums_l1959_195909


namespace girls_picked_more_l1959_195950

variable (N I A V : ℕ)

theorem girls_picked_more (h1 : N > A) (h2 : N > V) (h3 : N > I)
                         (h4 : I ≥ A) (h5 : I ≥ V) (h6 : A > V) :
  N + I > A + V := by
  sorry

end girls_picked_more_l1959_195950


namespace compute_expression_l1959_195983

theorem compute_expression (p q : ℝ) (h1 : p + q = 6) (h2 : p * q = 10) : 
  p^3 + p^4 * q^2 + p^2 * q^4 + p * q^3 + p^5 * q^3 = 38676 := by
  -- Proof goes here
  sorry

end compute_expression_l1959_195983


namespace guests_equal_cost_l1959_195901

-- Rental costs and meal costs
def rental_caesars_palace : ℕ := 800
def deluxe_meal_cost : ℕ := 30
def premium_meal_cost : ℕ := 40
def rental_venus_hall : ℕ := 500
def venus_special_cost : ℕ := 35
def venus_platter_cost : ℕ := 45

-- Meal distribution percentages
def deluxe_meal_percentage : ℚ := 0.60
def premium_meal_percentage : ℚ := 0.40
def venus_special_percentage : ℚ := 0.60
def venus_platter_percentage : ℚ := 0.40

-- Total costs calculation
noncomputable def total_cost_caesars (G : ℕ) : ℚ :=
  rental_caesars_palace + deluxe_meal_cost * deluxe_meal_percentage * G + premium_meal_cost * premium_meal_percentage * G

noncomputable def total_cost_venus (G : ℕ) : ℚ :=
  rental_venus_hall + venus_special_cost * venus_special_percentage * G + venus_platter_cost * venus_platter_percentage * G

-- Statement to show the equivalence of guest count
theorem guests_equal_cost (G : ℕ) : total_cost_caesars G = total_cost_venus G → G = 60 :=
by
  sorry

end guests_equal_cost_l1959_195901


namespace max_value_2ab_3bc_lemma_l1959_195919

noncomputable def max_value_2ab_3bc (a b c : ℝ) : ℝ :=
  2 * a * b + 3 * b * c

theorem max_value_2ab_3bc_lemma
  (a b c : ℝ)
  (ha : 0 ≤ a)
  (hb : 0 ≤ b)
  (hc : 0 ≤ c)
  (h : a^2 + b^2 + c^2 = 2) :
  max_value_2ab_3bc a b c ≤ 3 :=
sorry

end max_value_2ab_3bc_lemma_l1959_195919


namespace sandy_savings_l1959_195955

theorem sandy_savings (S : ℝ) :
  let last_year_savings := 0.10 * S
  let this_year_salary := 1.10 * S
  let this_year_savings := 1.65 * last_year_savings
  let P := this_year_savings / this_year_salary
  P * 100 = 15 :=
by
  let last_year_savings := 0.10 * S
  let this_year_salary := 1.10 * S
  let this_year_savings := 1.65 * last_year_savings
  let P := this_year_savings / this_year_salary
  have hP : P = 0.165 / 1.10 := by sorry
  have hP_percent : P * 100 = 15 := by sorry
  exact hP_percent

end sandy_savings_l1959_195955


namespace snow_at_least_once_l1959_195968

noncomputable def prob_snow_at_least_once (p1 p2 p3: ℚ) : ℚ :=
  1 - (1 - p1) * (1 - p2) * (1 - p3)

theorem snow_at_least_once : 
  prob_snow_at_least_once (1/2) (2/3) (3/4) = 23 / 24 := 
by
  sorry

end snow_at_least_once_l1959_195968


namespace terminal_side_of_minus_330_in_first_quadrant_l1959_195945

def angle_quadrant (angle : ℤ) : ℕ :=
  let reduced_angle := ((angle % 360) + 360) % 360
  if reduced_angle < 90 then 1
  else if reduced_angle < 180 then 2
  else if reduced_angle < 270 then 3
  else 4

theorem terminal_side_of_minus_330_in_first_quadrant :
  angle_quadrant (-330) = 1 :=
by
  -- We need a proof to justify the theorem, so we leave it with 'sorry' as instructed.
  sorry

end terminal_side_of_minus_330_in_first_quadrant_l1959_195945


namespace kath_total_cost_l1959_195958

def admission_cost : ℝ := 8
def discount_percentage_pre6pm : ℝ := 0.25
def discount_percentage_student : ℝ := 0.10
def time_of_movie : ℝ := 4
def num_people : ℕ := 6
def num_students : ℕ := 2

theorem kath_total_cost :
  let discounted_price := admission_cost * (1 - discount_percentage_pre6pm)
  let student_price := discounted_price * (1 - discount_percentage_student)
  let num_non_students := num_people - num_students - 1 -- remaining people (total - 2 students - Kath)
  let kath_and_siblings_cost := 3 * discounted_price
  let student_friends_cost := num_students * student_price
  let non_student_friend_cost := num_non_students * discounted_price
  let total_cost := kath_and_siblings_cost + student_friends_cost + non_student_friend_cost
  total_cost = 34.80 := by
  let discounted_price := admission_cost * (1 - discount_percentage_pre6pm)
  let student_price := discounted_price * (1 - discount_percentage_student)
  let num_non_students := num_people - num_students - 1
  let kath_and_siblings_cost := 3 * discounted_price
  let student_friends_cost := num_students * student_price
  let non_student_friend_cost := num_non_students * discounted_price
  let total_cost := kath_and_siblings_cost + student_friends_cost + non_student_friend_cost
  sorry

end kath_total_cost_l1959_195958


namespace cos_double_angle_l1959_195904

theorem cos_double_angle {α : ℝ} (h1 : 0 < α ∧ α < 2 * Real.pi ∧ α > 3 * Real.pi / 2) 
  (h2 : Real.sin α + Real.cos α = Real.sqrt 3 / 3) : 
  Real.cos (2 * α) = Real.sqrt 5 / 3 := 
by
  sorry

end cos_double_angle_l1959_195904


namespace baseball_wins_l1959_195969

-- Define the constants and conditions
def total_games : ℕ := 130
def won_more_than_lost (L W : ℕ) : Prop := W = 3 * L + 14
def total_games_played (L W : ℕ) : Prop := W + L = total_games

-- Define the theorem statement
theorem baseball_wins (L W : ℕ) 
  (h1 : won_more_than_lost L W)
  (h2 : total_games_played L W) : 
  W = 101 :=
  sorry

end baseball_wins_l1959_195969


namespace number_of_boys_in_class_l1959_195996

theorem number_of_boys_in_class 
  (n : ℕ)
  (average_height : ℝ)
  (incorrect_height : ℝ)
  (correct_height : ℝ)
  (actual_average_height : ℝ)
  (initial_average_height : average_height = 185)
  (incorrect_record : incorrect_height = 166)
  (correct_record : correct_height = 106)
  (actual_avg : actual_average_height = 183) 
  (total_height_incorrect : ℝ) 
  (total_height_correct : ℝ) 
  (total_height_eq : total_height_incorrect = 185 * n)
  (correct_total_height_eq : total_height_correct = 185 * n - (incorrect_height - correct_height))
  (actual_total_height_eq : total_height_correct = actual_average_height * n) :
  n = 30 :=
by
  sorry

end number_of_boys_in_class_l1959_195996


namespace smallest_y_value_smallest_y_value_is_neg6_l1959_195993

theorem smallest_y_value :
  ∀ y : ℝ, (3 * y^2 + 21 * y + 18 = y * (2 * y + 12)) → (y = -3 ∨ y = -6) :=
by
  sorry

theorem smallest_y_value_is_neg6 :
  ∃ y : ℝ, (3 * y^2 + 21 * y + 18 = y * (2 * y + 12)) ∧ (y = -6) :=
by
  have H := smallest_y_value
  sorry

end smallest_y_value_smallest_y_value_is_neg6_l1959_195993


namespace lake_with_more_frogs_has_45_frogs_l1959_195915

-- Definitions for the problem.
variable (F : ℝ) -- Number of frogs in the lake with more frogs.
variable (F_less : ℝ) -- Number of frogs in Lake Crystal (the lake with fewer frogs).

-- Conditions
axiom fewer_frogs_condition : F_less = 0.8 * F
axiom total_frogs_condition : F + F_less = 81

-- Theorem statement: Proving that the number of frogs in the lake with more frogs is 45.
theorem lake_with_more_frogs_has_45_frogs :
  F = 45 :=
by
  sorry

end lake_with_more_frogs_has_45_frogs_l1959_195915


namespace x_when_y_is_125_l1959_195924

noncomputable def C : ℝ := (2^2) * (5^2)

theorem x_when_y_is_125 
  (x y : ℝ) 
  (h_pos : x > 0 ∧ y > 0) 
  (h_inv : x^2 * y^2 = C) 
  (h_initial : y = 5) 
  (h_x_initial : x = 2) 
  (h_y : y = 125) : 
  x = 2 / 25 :=
by
  sorry

end x_when_y_is_125_l1959_195924


namespace penelope_min_games_l1959_195918

theorem penelope_min_games (m w l: ℕ) (h1: 25 * w - 13 * l = 2007) (h2: m = w + l) : m = 87 := by
  sorry

end penelope_min_games_l1959_195918


namespace bretschneider_l1959_195920

noncomputable def bretschneider_theorem 
  (a b c d m n : ℝ) 
  (A C : ℝ) : Prop :=
  m^2 * n^2 = a^2 * c^2 + b^2 * d^2 - 2 * a * b * c * d * Real.cos (A + C)

theorem bretschneider (a b c d m n A C : ℝ) :
  bretschneider_theorem a b c d m n A C :=
sorry

end bretschneider_l1959_195920


namespace solution_of_system_l1959_195910

theorem solution_of_system :
  (∀ x : ℝ,
    (2 + x < 6 - 3 * x) ∧ (x ≤ (4 + x) / 2)
    → x < 1) :=
by
  sorry

end solution_of_system_l1959_195910


namespace shirt_final_price_is_correct_l1959_195943

noncomputable def final_price_percentage (initial_price : ℝ) : ℝ :=
  let first_discount := initial_price * 0.80
  let second_discount := first_discount * 0.90
  let anniversary_addition := second_discount * 1.05
  let final_price := anniversary_addition * 1.15
  final_price / initial_price * 100

theorem shirt_final_price_is_correct (initial_price : ℝ) : final_price_percentage initial_price = 86.94 := by
  sorry

end shirt_final_price_is_correct_l1959_195943


namespace determine_pairs_l1959_195991

theorem determine_pairs (a b : ℕ) (h : 2017^a = b^6 - 32 * b + 1) : 
  (a = 0 ∧ b = 0) ∨ (a = 0 ∧ b = 2) :=
by sorry

end determine_pairs_l1959_195991


namespace Karen_wall_paint_area_l1959_195992

theorem Karen_wall_paint_area :
  let height_wall := 10
  let width_wall := 15
  let height_window := 3
  let width_window := 5
  let height_door := 2
  let width_door := 6
  let area_wall := height_wall * width_wall
  let area_window := height_window * width_window
  let area_door := height_door * width_door
  let area_to_paint := area_wall - area_window - area_door
  area_to_paint = 123 := by
{
  sorry
}

end Karen_wall_paint_area_l1959_195992


namespace total_tablets_l1959_195961

-- Variables for the numbers of Lenovo, Samsung, and Huawei tablets
variables (n x y : ℕ)

-- Conditions based on problem statement
def condition1 : Prop := 2 * x + 6 + y < n / 3

def condition2 : Prop := (n - 2 * x - y - 6 = 3 * y)

def condition3 : Prop := (n - 6 * x - y - 6 = 59)

-- The statement to prove that the total number of tablets is 94
theorem total_tablets (h1 : condition1 n x y) (h2 : condition2 n x y) (h3 : condition3 n x y) : n = 94 :=
by
  sorry

end total_tablets_l1959_195961


namespace sum_of_coefficients_eq_l1959_195956

theorem sum_of_coefficients_eq :
  ∃ n : ℕ, (∀ a b : ℕ, (3 * a + 5 * b)^n = 2^15) → n = 5 :=
by
  sorry

end sum_of_coefficients_eq_l1959_195956


namespace smallest_percent_increase_from_2_to_3_l1959_195977

def percent_increase (initial final : ℕ) : ℚ := 
  ((final - initial : ℕ) : ℚ) / (initial : ℕ) * 100

def value_at_question : ℕ → ℕ
| 1 => 100
| 2 => 200
| 3 => 300
| 4 => 500
| 5 => 1000
| 6 => 2000
| 7 => 4000
| 8 => 8000
| 9 => 16000
| 10 => 32000
| 11 => 64000
| 12 => 125000
| 13 => 250000
| 14 => 500000
| 15 => 1000000
| _ => 0  -- Default case for questions out of range

theorem smallest_percent_increase_from_2_to_3 :
  let p1 := percent_increase (value_at_question 1) (value_at_question 2)
  let p2 := percent_increase (value_at_question 2) (value_at_question 3)
  let p3 := percent_increase (value_at_question 3) (value_at_question 4)
  let p11 := percent_increase (value_at_question 11) (value_at_question 12)
  let p14 := percent_increase (value_at_question 14) (value_at_question 15)
  p2 < p1 ∧ p2 < p3 ∧ p2 < p11 ∧ p2 < p14 :=
by
  sorry

end smallest_percent_increase_from_2_to_3_l1959_195977


namespace sum_of_997_lemons_l1959_195957

-- Define x and y as functions of k
def x (k : ℕ) := 1 + 9 * k
def y (k : ℕ) := 110 - 7 * k

-- The theorem we need to prove
theorem sum_of_997_lemons :
  ∃ (k : ℕ), 0 ≤ k ∧ k ≤ 15 ∧ 7 * (x k) + 9 * (y k) = 997 := 
by
  sorry -- Proof to be filled in

end sum_of_997_lemons_l1959_195957


namespace Danny_more_wrappers_than_caps_l1959_195990

-- Define the conditions
def bottle_caps_park := 11
def wrappers_park := 28

-- State the theorem representing the problem
theorem Danny_more_wrappers_than_caps:
  wrappers_park - bottle_caps_park = 17 :=
by
  sorry

end Danny_more_wrappers_than_caps_l1959_195990


namespace seventh_term_correct_l1959_195911

noncomputable def seventh_term_geometric_sequence (a r : ℝ) (h1 : a = 5) (h2 : a * r = 1/5) : ℝ :=
  a * r ^ 6

theorem seventh_term_correct :
  seventh_term_geometric_sequence 5 (1/25) (by rfl) (by norm_num) = 1 / 48828125 :=
  by
    unfold seventh_term_geometric_sequence
    sorry

end seventh_term_correct_l1959_195911


namespace total_cost_is_225_l1959_195938

def total_tickets : ℕ := 29
def cost_7_dollar_ticket : ℕ := 7
def cost_9_dollar_ticket : ℕ := 9
def number_of_9_dollar_tickets : ℕ := 11
def number_of_7_dollar_tickets : ℕ := total_tickets - number_of_9_dollar_tickets
def total_cost : ℕ := (number_of_9_dollar_tickets * cost_9_dollar_ticket) + (number_of_7_dollar_tickets * cost_7_dollar_ticket)

theorem total_cost_is_225 : total_cost = 225 := by
  sorry

end total_cost_is_225_l1959_195938


namespace percentage_increase_of_gross_l1959_195976

theorem percentage_increase_of_gross
  (P R : ℝ)
  (price_drop : ℝ := 0.20)
  (quantity_increase : ℝ := 0.60)
  (original_gross : ℝ := P * R)
  (new_price : ℝ := (1 - price_drop) * P)
  (new_quantity_sold : ℝ := (1 + quantity_increase) * R)
  (new_gross : ℝ := new_price * new_quantity_sold)
  (percentage_increase : ℝ := ((new_gross - original_gross) / original_gross) * 100) :
  percentage_increase = 28 :=
by
  sorry

end percentage_increase_of_gross_l1959_195976


namespace abes_age_l1959_195986

theorem abes_age (A : ℕ) (h : A + (A - 7) = 29) : A = 18 :=
by
  sorry

end abes_age_l1959_195986


namespace find_common_ratio_l1959_195998

noncomputable def common_ratio_of_geometric_sequence (a : ℕ → ℝ) (d : ℝ) 
  (h1 : d ≠ 0)
  (h2 : ∀ n, a (n + 1) = a n + d)
  (h3 : ∃ (r : ℝ), r ≠ 0 ∧ a 3 ^ 2 = a 1 * a 9) : ℝ :=
3

theorem find_common_ratio 
( a : ℕ → ℝ) 
( d : ℝ) 
(h1 : d ≠ 0)
(h2 : ∀ n, a (n + 1) = a n + d)
(h3 : ∃ (r : ℝ), r ≠ 0 ∧ a 3 ^ 2 = a 1 * a 9) :
common_ratio_of_geometric_sequence a d h1 h2 h3 = 3 :=
sorry

end find_common_ratio_l1959_195998


namespace jeff_total_run_is_290_l1959_195952

variables (monday_to_wednesday_run : ℕ)
variables (thursday_run : ℕ)
variables (friday_run : ℕ)

def jeff_weekly_run_total : ℕ :=
  monday_to_wednesday_run + thursday_run + friday_run

theorem jeff_total_run_is_290 :
  (60 * 3) + (60 - 20) + (60 + 10) = 290 :=
by
  sorry

end jeff_total_run_is_290_l1959_195952


namespace calculate_expression_l1959_195951

variables (a b : ℝ) -- declaring variables a and b to be real numbers

theorem calculate_expression :
  (-a * b^2) ^ 3 + (a * b^2) * (a * b) ^ 2 * (-2 * b) ^ 2 = 3 * a^3 * b^6 :=
by
  sorry

end calculate_expression_l1959_195951


namespace rationalize_denominator_l1959_195944

theorem rationalize_denominator (A B C : ℤ) (h : A + B * Real.sqrt C = -(9) - 4 * Real.sqrt 5) : A * B * C = 180 :=
by
  have hA : A = -9 := by sorry
  have hB : B = -4 := by sorry
  have hC : C = 5 := by sorry
  rw [hA, hB, hC]
  norm_num

end rationalize_denominator_l1959_195944


namespace cos_of_vector_dot_product_l1959_195964

open Real

noncomputable def cos_value (x : ℝ) : ℝ := cos (x + π / 4)

theorem cos_of_vector_dot_product (x : ℝ)
  (h1 : π / 4 < x)
  (h2 : x < π / 2)
  (h3 : (sqrt 2) * cos x + (sqrt 2) * sin x = 8 / 5) :
  cos_value x = - 3 / 5 :=
by
  sorry

end cos_of_vector_dot_product_l1959_195964


namespace max_gcd_of_15m_plus_4_and_14m_plus_3_l1959_195942

theorem max_gcd_of_15m_plus_4_and_14m_plus_3 (m : ℕ) (hm : 0 < m) :
  ∃ k : ℕ, k = gcd (15 * m + 4) (14 * m + 3) ∧ k = 11 :=
by {
  sorry
}

end max_gcd_of_15m_plus_4_and_14m_plus_3_l1959_195942


namespace four_digit_number_is_2561_l1959_195935

-- Define the problem domain based on given conditions
def unique_in_snowflake_and_directions (grid : Matrix (Fin 3) (Fin 6) ℕ) : Prop :=
  ∀ (i j : Fin 3), -- across all directions
    ∀ (x y : Fin 6), 
      (x ≠ y) → 
      (grid i x ≠ grid i y) -- uniqueness in i-direction
      ∧ (grid y x ≠ grid y y) -- uniqueness in j-direction

-- Assignment of numbers in the grid fulfilling the conditions
def grid : Matrix (Fin 3) (Fin 6) ℕ :=
![ ![2, 5, 2, 5, 1, 6], ![4, 3, 2, 6, 1, 1], ![6, 1, 4, 5, 3, 2] ]

-- Definition of the four-digit number
def ABCD : ℕ := grid 0 1 * 1000 + grid 0 2 * 100 + grid 0 3 * 10 + grid 0 4

-- The theorem to be proved
theorem four_digit_number_is_2561 :
  unique_in_snowflake_and_directions grid →
  ABCD = 2561 :=
sorry

end four_digit_number_is_2561_l1959_195935


namespace log_inequality_l1959_195907

theorem log_inequality : 
  ∀ (logπ2 log2π : ℝ), logπ2 = 1 / log2π → 0 < logπ2 → 0 < log2π → (1 / logπ2 + 1 / log2π > 2) :=
by
  intros logπ2 log2π h1 h2 h3
  have h4: logπ2 = 1 / log2π := h1
  have h5: 0 < logπ2 := h2
  have h6: 0 < log2π := h3
  -- To be completed with the actual proof steps if needed
  sorry

end log_inequality_l1959_195907


namespace set_roster_method_l1959_195959

open Set

theorem set_roster_method :
  { m : ℤ | ∃ n : ℕ, 12 = n * (m + 1) } = {0, 1, 2, 3, 5, 11} :=
  sorry

end set_roster_method_l1959_195959


namespace spending_Mar_Apr_May_l1959_195928

-- Define the expenditures at given points
def e_Feb : ℝ := 0.7
def e_Mar : ℝ := 1.2
def e_May : ℝ := 4.4

-- Define the amount spent from March to May
def amount_spent_Mar_Apr_May := e_May - e_Feb

-- The main theorem to prove
theorem spending_Mar_Apr_May : amount_spent_Mar_Apr_May = 3.7 := by
  sorry

end spending_Mar_Apr_May_l1959_195928


namespace victor_initial_books_l1959_195936

theorem victor_initial_books (x : ℕ) : (x + 3 = 12) → (x = 9) :=
by
  sorry

end victor_initial_books_l1959_195936


namespace geometric_series_sum_l1959_195984

-- Definition of the geometric sum function in Lean
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * ((1 - r^n) / (1 - r))

-- Specific terms for the problem
def a : ℚ := 2
def r : ℚ := 2 / 5
def n : ℕ := 5

-- The target sum we aim to prove
def target_sum : ℚ := 10310 / 3125

-- The theorem stating that the calculated sum equals the target sum
theorem geometric_series_sum : geometric_sum a r n = target_sum :=
by sorry

end geometric_series_sum_l1959_195984


namespace decimal_to_fraction_l1959_195966

theorem decimal_to_fraction (d : ℚ) (h : d = 2.35) : d = 47 / 20 := sorry

end decimal_to_fraction_l1959_195966


namespace exists_sequences_l1959_195927

theorem exists_sequences (m n : Nat → Nat) (h₁ : ∀ k, m k = 2 * k) (h₂ : ∀ k, n k = 5 * k * k)
  (h₃ : ∀ (i j : Nat), (i ≠ j) → (m i ≠ m j) ∧ (n i ≠ n j)) :
  (∀ k, Nat.sqrt (n k + (m k) * (m k)) = 3 * k) ∧
  (∀ k, Nat.sqrt (n k - (m k) * (m k)) = k) :=
by 
  sorry

end exists_sequences_l1959_195927


namespace profit_percentage_is_33_point_33_l1959_195914

variable (C S : ℝ)

-- Initial condition based on the problem statement
axiom cost_eq_sell : 20 * C = 15 * S

-- Statement to prove
theorem profit_percentage_is_33_point_33 (h : 20 * C = 15 * S) : (S - C) / C * 100 = 33.33 := 
sorry

end profit_percentage_is_33_point_33_l1959_195914


namespace not_divisible_l1959_195912

theorem not_divisible (n : ℕ) : ¬ ((4^n - 1) ∣ (5^n - 1)) :=
by
  sorry

end not_divisible_l1959_195912


namespace nate_walks_past_per_minute_l1959_195949

-- Define the conditions as constants
def rows_G := 15
def cars_per_row_G := 10
def rows_H := 20
def cars_per_row_H := 9
def total_minutes := 30

-- Define the problem statement
theorem nate_walks_past_per_minute :
  ((rows_G * cars_per_row_G) + (rows_H * cars_per_row_H)) / total_minutes = 11 := 
sorry

end nate_walks_past_per_minute_l1959_195949


namespace sin_cos_identity_l1959_195988

theorem sin_cos_identity (θ : Real) (h1 : 0 < θ ∧ θ < π) (h2 : Real.sin θ * Real.cos θ = - (1/8)) :
  Real.sin (2 * Real.pi + θ) - Real.sin ((Real.pi / 2) - θ) = (Real.sqrt 5) / 2 := by
  sorry

end sin_cos_identity_l1959_195988


namespace tree_planting_activity_l1959_195908

variables (trees_first_group trees_second_group people_first_group people_second_group : ℕ)
variable (average_trees_per_person_first_group average_trees_per_person_second_group : ℕ)

theorem tree_planting_activity :
  trees_first_group = 12 →
  trees_second_group = 36 →
  people_second_group = people_first_group + 6 →
  average_trees_per_person_first_group = trees_first_group / people_first_group →
  average_trees_per_person_second_group = trees_second_group / people_second_group →
  average_trees_per_person_first_group = average_trees_per_person_second_group →
  people_first_group = 3 := 
by 
  intros h1 h2 h3 h4 h5 h6
  sorry

end tree_planting_activity_l1959_195908


namespace no_base_6_digit_divisible_by_7_l1959_195917

theorem no_base_6_digit_divisible_by_7 :
  ∀ (d : ℕ), d < 6 → ¬ (7 ∣ (652 + 42 * d)) :=
by
  intros d hd
  sorry

end no_base_6_digit_divisible_by_7_l1959_195917


namespace quadratic_range_l1959_195970

noncomputable def f : ℝ → ℝ := sorry -- Quadratic function with a positive coefficient for its quadratic term

axiom symmetry_condition : ∀ x : ℝ, f x = f (4 - x)

theorem quadratic_range (x : ℝ) (h1 : f (1 - 2 * x ^ 2) < f (1 + 2 * x - x ^ 2)) : -2 < x ∧ x < 0 :=
by sorry

end quadratic_range_l1959_195970


namespace find_value_of_expression_l1959_195921

variable {a : ℕ → ℤ}

-- Define arithmetic sequence property
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variable (h1 : a 1 + 3 * a 8 + a 15 = 120)
variable (h2 : is_arithmetic_sequence a)

-- Theorem to be proved
theorem find_value_of_expression : 2 * a 6 - a 4 = 24 :=
sorry

end find_value_of_expression_l1959_195921


namespace ellipse_closer_to_circle_l1959_195906

variables (a : ℝ)

-- Conditions: 1 < a < 2 + sqrt 5
def in_range_a (a : ℝ) : Prop := 1 < a ∧ a < 2 + Real.sqrt 5

-- Ellipse eccentricity should decrease as 'a' increases for the given range 1 < a < 2 + sqrt 5
theorem ellipse_closer_to_circle (h_range : in_range_a a) :
    ∃ b : ℝ, b = Real.sqrt (1 - (a^2 - 1) / (4 * a)) ∧ ∀ a', (1 < a' ∧ a' < 2 + Real.sqrt 5 ∧ a < a') → b > Real.sqrt (1 - (a'^2 - 1) / (4 * a')) := 
sorry

end ellipse_closer_to_circle_l1959_195906


namespace little_john_initial_money_l1959_195980

def sweets_cost : ℝ := 2.25
def friends_donation : ℝ := 2 * 2.20
def money_left : ℝ := 3.85

theorem little_john_initial_money :
  sweets_cost + friends_donation + money_left = 10.50 :=
by
  sorry

end little_john_initial_money_l1959_195980


namespace value_of_a_minus_b_l1959_195933

theorem value_of_a_minus_b (a b : ℝ) (h : (a - 5)^2 + |b^3 - 27| = 0) : a - b = 2 :=
by
  sorry

end value_of_a_minus_b_l1959_195933


namespace dividend_percentage_shares_l1959_195934

theorem dividend_percentage_shares :
  ∀ (purchase_price market_value : ℝ) (interest_rate : ℝ),
  purchase_price = 56 →
  market_value = 42 →
  interest_rate = 0.12 →
  ( (interest_rate * purchase_price) / market_value * 100 = 16) :=
by
  intros purchase_price market_value interest_rate h1 h2 h3
  rw [h1, h2, h3]
  -- Calculations were done in solution
  sorry

end dividend_percentage_shares_l1959_195934


namespace intersection_A_B_l1959_195963

def A : Set ℝ := { x | abs x < 3 }
def B : Set ℝ := { x | 2 - x > 0 }

theorem intersection_A_B : A ∩ B = { x : ℝ | -3 < x ∧ x < 2 } :=
by
  sorry

end intersection_A_B_l1959_195963


namespace sum_of_common_ratios_l1959_195971

variable {k p r : ℝ}

-- Condition 1: geometric sequences with distinct common ratios
-- Condition 2: a_3 - b_3 = 3(a_2 - b_2)
def geometric_sequences (k p r : ℝ) : Prop :=
  (k ≠ 0) ∧ (p ≠ r) ∧ (k * p^2 - k * r^2 = 3 * (k * p - k * r))

theorem sum_of_common_ratios (k p r : ℝ) (h : geometric_sequences k p r) : p + r = 3 :=
by
  sorry

end sum_of_common_ratios_l1959_195971


namespace problem_a_proof_l1959_195903

variables {A B C D M K : Point}
variables {triangle_ABC : Triangle A B C}
variables {incircle : Circle} (ht : touches incircle AC D) 
variables (hdm : diameter incircle D M) 
variables (bm_line : Line B M) (intersect_bm_ac : intersects bm_line AC K)

theorem problem_a_proof : 
  AK = DC :=
sorry

end problem_a_proof_l1959_195903


namespace wheel_radius_correct_l1959_195926
noncomputable def wheel_radius (total_distance : ℝ) (n_revolutions : ℕ) : ℝ :=
  total_distance / (n_revolutions * 2 * Real.pi)

theorem wheel_radius_correct :
  wheel_radius 450.56 320 = 0.224 :=
by
  sorry

end wheel_radius_correct_l1959_195926


namespace op_proof_l1959_195940

-- Definition of the operation \(\oplus\)
def op (x y : ℝ) : ℝ := x^2 + y

-- Theorem statement for the given proof problem
theorem op_proof (h : ℝ) : op h (op h h) = 2 * h^2 + h :=
by 
  sorry

end op_proof_l1959_195940


namespace inequality_proof_l1959_195989

theorem inequality_proof (x : ℝ) : 3 * x - 6 > 5 * (x - 2) → x < 2 :=
by
  sorry

end inequality_proof_l1959_195989


namespace box_height_l1959_195939

theorem box_height (h : ℝ) :
  ∃ (h : ℝ), 
  let large_sphere_radius := 3
  let small_sphere_radius := 1.5
  let box_width := 6
  h = 12 := 
sorry

end box_height_l1959_195939


namespace train_speed_l1959_195905

noncomputable def speed_in_kmh (distance : ℕ) (time : ℕ) : ℚ :=
  (distance : ℚ) / (time : ℚ) * 3600 / 1000

theorem train_speed
  (distance : ℕ) (time : ℕ)
  (h_dist : distance = 150)
  (h_time : time = 9) :
  speed_in_kmh distance time = 60 :=
by
  rw [h_dist, h_time]
  sorry

end train_speed_l1959_195905


namespace end_digit_of_number_l1959_195948

theorem end_digit_of_number (n : ℕ) (h_n : n = 2022) (h_start : ∃ (f : ℕ → ℕ), f 0 = 4 ∧ 
    (∀ i < n - 1, (19 ∣ (10 * f i + f (i + 1))) ∨ (23 ∣ (10 * f i + f (i + 1))))) :
  ∃ (f : ℕ → ℕ), f (n - 1) = 8 :=
by {
  sorry
}

end end_digit_of_number_l1959_195948


namespace area_percentage_increase_l1959_195947

theorem area_percentage_increase (r1 r2 : ℝ) (π : ℝ) (area1 area2 : ℝ) (N : ℝ) :
  r1 = 6 → r2 = 4 → area1 = π * r1 ^ 2 → area2 = π * r2 ^ 2 →
  N = 125 →
  ((area1 - area2) / area2) * 100 = N :=
by {
  sorry
}

end area_percentage_increase_l1959_195947


namespace total_books_correct_l1959_195946

-- Definitions based on the conditions
def num_books_bottom_shelf (T : ℕ) := T / 3
def num_books_middle_shelf (T : ℕ) := T / 4
def num_books_top_shelf : ℕ := 30
def total_books (T : ℕ) := num_books_bottom_shelf T + num_books_middle_shelf T + num_books_top_shelf

theorem total_books_correct : total_books 72 = 72 :=
by
  sorry

end total_books_correct_l1959_195946


namespace actual_distance_is_correct_l1959_195979

noncomputable def actual_distance_in_meters (scale : ℕ) (map_distance_cm : ℝ) : ℝ :=
  (map_distance_cm * scale) / 100

theorem actual_distance_is_correct
  (scale : ℕ)
  (map_distance_cm : ℝ)
  (h_scale : scale = 3000000)
  (h_map_distance : map_distance_cm = 4) :
  actual_distance_in_meters scale map_distance_cm = 1.2 * 10^5 :=
by
  sorry

end actual_distance_is_correct_l1959_195979


namespace C_is_14_years_younger_than_A_l1959_195930

variable (A B C D : ℕ)

-- Conditions
axiom cond1 : A + B = (B + C) + 14
axiom cond2 : B + D = (C + A) + 10
axiom cond3 : D = C + 6

-- To prove
theorem C_is_14_years_younger_than_A : A - C = 14 :=
by
  sorry

end C_is_14_years_younger_than_A_l1959_195930


namespace systematic_sampling_removal_count_l1959_195975

theorem systematic_sampling_removal_count :
  ∀ (N n : ℕ), N = 3204 ∧ n = 80 → N % n = 4 := 
by
  sorry

end systematic_sampling_removal_count_l1959_195975


namespace must_divide_l1959_195994

-- Proving 5 is a divisor of q

variables {p q r s : ℕ}

theorem must_divide (h1 : Nat.gcd p q = 30) (h2 : Nat.gcd q r = 42)
                   (h3 : Nat.gcd r s = 66) (h4 : 80 < Nat.gcd s p)
                   (h5 : Nat.gcd s p < 120) :
                   5 ∣ q :=
sorry

end must_divide_l1959_195994


namespace angle_C_measure_ratio_inequality_l1959_195985

open Real

variables (A B C a b c : ℝ)

-- Assumptions
variable (ABC_is_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b)
variable (sin_condition : sin (2 * C - π / 2) = 1/2)
variable (inequality_condition : a^2 + b^2 < c^2)

theorem angle_C_measure :
  0 < C ∧ C < π ∧ C = 2 * π / 3 := sorry

theorem ratio_inequality :
  1 < (a + b) / c ∧ (a + b) / c ≤ 2 * sqrt 3 / 3 := sorry

end angle_C_measure_ratio_inequality_l1959_195985


namespace smallest_multiplier_to_perfect_square_l1959_195987

theorem smallest_multiplier_to_perfect_square : ∃ k : ℕ, k > 0 ∧ ∀ m : ℕ, (2010 * m = k * k) → m = 2010 :=
by
  sorry

end smallest_multiplier_to_perfect_square_l1959_195987


namespace tommy_initial_balloons_l1959_195941

theorem tommy_initial_balloons :
  ∃ x : ℝ, x + 78.5 = 132.25 ∧ x = 53.75 := by
  sorry

end tommy_initial_balloons_l1959_195941
