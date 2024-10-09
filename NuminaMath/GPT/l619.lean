import Mathlib

namespace solve_equation_l619_61972

variable (x : ℝ)

def equation := (x / (2 * x - 3)) + (5 / (3 - 2 * x)) = 4
def condition := x ≠ 3 / 2

theorem solve_equation : equation x ∧ condition x → x = 1 :=
by
  sorry

end solve_equation_l619_61972


namespace katie_has_more_games_l619_61924

   -- Conditions
   def katie_games : Nat := 81
   def friends_games : Nat := 59

   -- Problem statement
   theorem katie_has_more_games : (katie_games - friends_games) = 22 :=
   by
     -- Proof to be provided
     sorry
   
end katie_has_more_games_l619_61924


namespace employee_payments_l619_61978

theorem employee_payments :
  ∃ (A B C : ℤ), A = 900 ∧ B = 600 ∧ C = 500 ∧
    A + B + C = 2000 ∧
    A = 3 * B / 2 ∧
    C = 400 + 100 := 
by
  sorry

end employee_payments_l619_61978


namespace andy_l619_61994

theorem andy's_profit_per_cake :
  (∀ (cakes : ℕ), cakes = 2 → ∀ (ingredient_cost : ℕ), ingredient_cost = 12 →
                  ∀ (packaging_cost_per_cake : ℕ), packaging_cost_per_cake = 1 →
                  ∀ (selling_price_per_cake : ℕ), selling_price_per_cake = 15 →
                  ∀ (profit_per_cake : ℕ), profit_per_cake = selling_price_per_cake - (ingredient_cost / cakes + packaging_cost_per_cake) →
                    profit_per_cake = 8) :=
by
  sorry

end andy_l619_61994


namespace inequality_problem_l619_61939

theorem inequality_problem (x : ℝ) (hx : 0 < x) : 
  1 + x ^ 2018 ≥ (2 * x) ^ 2017 / (1 + x) ^ 2016 := 
by
  sorry

end inequality_problem_l619_61939


namespace arith_seq_sum_geom_mean_proof_l619_61944

theorem arith_seq_sum_geom_mean_proof (a_1 : ℝ) (a_n : ℕ → ℝ)
(common_difference : ℝ) (s_n : ℕ → ℝ)
(h_sequence : ∀ n, a_n n = a_1 + (n - 1) * common_difference)
(h_sum : ∀ n, s_n n = n / 2 * (2 * a_1 + (n - 1) * common_difference))
(h_geom_mean : (s_n 2) ^ 2 = s_n 1 * s_n 4)
(h_common_diff : common_difference = -1) :
a_1 = -1 / 2 :=
sorry

end arith_seq_sum_geom_mean_proof_l619_61944


namespace is_isosceles_of_x_eq_one_root_is_right_angled_of_equal_roots_l619_61956

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

-- Given that a, b, c are the sides of the triangle
axiom lengths_of_triangle : a > 0 ∧ b > 0 ∧ c > 0

-- Problem 1: Prove that triangle is isosceles if x=1 is a root
theorem is_isosceles_of_x_eq_one_root  : ((a - c) * (1:ℝ)^2 - 2 * b * (1:ℝ) + (a + c) = 0) → a = b ∧ a ≠ c := 
by
  intros h
  sorry

-- Problem 2: Prove that triangle is right-angled if the equation has two equal real roots
theorem is_right_angled_of_equal_roots : (b^2 = a^2 - c^2) → (a^2 = b^2 + c^2) := 
by 
  intros h
  sorry

end is_isosceles_of_x_eq_one_root_is_right_angled_of_equal_roots_l619_61956


namespace homes_distance_is_65_l619_61928

noncomputable def distance_between_homes
  (maxwell_speed : ℕ)
  (brad_speed : ℕ)
  (maxwell_distance : ℕ)
  (time : ℕ) : ℕ :=
  maxwell_distance + brad_speed * time

theorem homes_distance_is_65
  (maxwell_speed : ℕ := 2)
  (brad_speed : ℕ := 3)
  (maxwell_distance : ℕ := 26)
  (time : ℕ := maxwell_distance / maxwell_speed) :
  distance_between_homes maxwell_speed brad_speed maxwell_distance time = 65 :=
by 
  sorry

end homes_distance_is_65_l619_61928


namespace sandy_took_310_dollars_l619_61940

theorem sandy_took_310_dollars (X : ℝ) (h70percent : 0.70 * X = 217) : X = 310 := by
  sorry

end sandy_took_310_dollars_l619_61940


namespace invalid_prob_distribution_D_l619_61926

noncomputable def sum_of_probs_A : ℚ :=
  0 + 1/2 + 0 + 0 + 1/2

noncomputable def sum_of_probs_B : ℚ :=
  0.1 + 0.2 + 0.3 + 0.4

noncomputable def sum_of_probs_C (p : ℚ) (hp : 0 ≤ p ∧ p ≤ 1) : ℚ :=
  p + (1 - p)

noncomputable def sum_of_probs_D : ℚ :=
  (1/1*2) + (1/2*3) + (1/3*4) + (1/4*5) + (1/5*6) + (1/6*7) + (1/7*8)

theorem invalid_prob_distribution_D :
  sum_of_probs_D ≠ 1 := sorry

end invalid_prob_distribution_D_l619_61926


namespace maria_needs_green_beans_l619_61932

theorem maria_needs_green_beans :
  ∀ (potatoes carrots onions green_beans : ℕ), 
  (carrots = 6 * potatoes) →
  (onions = 2 * carrots) →
  (green_beans = onions / 3) →
  (potatoes = 2) →
  green_beans = 8 :=
by
  intros potatoes carrots onions green_beans h1 h2 h3 h4
  rw [h4, Nat.mul_comm 6 2] at h1
  rw [h1, Nat.mul_comm 2 12] at h2
  rw [h2] at h3
  sorry

end maria_needs_green_beans_l619_61932


namespace first_sequence_general_term_second_sequence_general_term_l619_61995

-- For the first sequence
def first_sequence_sum : ℕ → ℚ
| n => n^2 + 1/2 * n

theorem first_sequence_general_term (n : ℕ) : 
  (first_sequence_sum (n+1) - first_sequence_sum n) = (2 * (n+1) - 1/2) := 
sorry

-- For the second sequence
def second_sequence_sum : ℕ → ℚ
| n => 1/4 * n^2 + 2/3 * n + 3

theorem second_sequence_general_term (n : ℕ) : 
  (second_sequence_sum (n+1) - second_sequence_sum n) = 
  if n = 0 then 47/12 
  else (6 * (n+1) + 5)/12 := 
sorry

end first_sequence_general_term_second_sequence_general_term_l619_61995


namespace find_X_l619_61950

def star (a b : ℤ) : ℤ := 5 * a - 3 * b

theorem find_X (X : ℤ) (h1 : star X (star 3 2) = 18) : X = 9 :=
by
  sorry

end find_X_l619_61950


namespace age_proof_l619_61923

   variable (x : ℝ)
   
   theorem age_proof (h : 3 * (x + 5) - 3 * (x - 5) = x) : x = 30 :=
   by
     sorry
   
end age_proof_l619_61923


namespace batsman_average_after_17th_inning_l619_61954

theorem batsman_average_after_17th_inning (A : ℝ) :
  let total_runs_after_17_innings := 16 * A + 87
  let new_average := total_runs_after_17_innings / 17
  new_average = A + 3 → 
  (A + 3) = 39 :=
by
  sorry

end batsman_average_after_17th_inning_l619_61954


namespace commodities_price_difference_l619_61979

theorem commodities_price_difference : 
  ∀ (C1 C2 : ℕ), 
    C1 = 477 → 
    C1 + C2 = 827 → 
    C1 - C2 = 127 :=
by
  intros C1 C2 h1 h2
  sorry

end commodities_price_difference_l619_61979


namespace hot_dogs_total_l619_61907

theorem hot_dogs_total (D : ℕ)
  (h1 : 9 = 2 * D + D + 3) :
  (2 * D + 9 + D = 15) :=
by sorry

end hot_dogs_total_l619_61907


namespace sum_coordinates_point_C_l619_61980

/-
Let point A = (0,0), point B is on the line y = 6, and the slope of AB is 3/4.
Point C lies on the y-axis with a slope of 1/2 from B to C.
We need to prove that the sum of the coordinates of point C is 2.
-/
theorem sum_coordinates_point_C : 
  ∃ (A B C : ℝ × ℝ), 
  A = (0, 0) ∧ 
  B.2 = 6 ∧ 
  (B.2 - A.2) / (B.1 - A.1) = 3 / 4 ∧ 
  C.1 = 0 ∧ 
  (C.2 - B.2) / (C.1 - B.1) = 1 / 2 ∧ 
  C.1 + C.2 = 2 :=
by
  sorry

end sum_coordinates_point_C_l619_61980


namespace find_counterfeit_coin_l619_61910

def is_counterfeit (coins : Fin 9 → ℝ) (i : Fin 9) : Prop :=
  ∀ j : Fin 9, j ≠ i → coins j = coins 0 ∧ coins i < coins 0

def algorithm_exists (coins : Fin 9 → ℝ) : Prop :=
  ∃ f : (Fin 9 → ℝ) → Fin 9, is_counterfeit coins (f coins)

theorem find_counterfeit_coin (coins : Fin 9 → ℝ) (h : ∃ i : Fin 9, is_counterfeit coins i) : algorithm_exists coins :=
by sorry

end find_counterfeit_coin_l619_61910


namespace percentage_of_boys_and_additional_boys_l619_61930

theorem percentage_of_boys_and_additional_boys (total_students : ℕ) (boys_ratio : ℕ) (girls_ratio : ℕ)
  (total_students_eq : total_students = 42) (ratio_condition : boys_ratio = 3 ∧ girls_ratio = 4) :
  let total_groups := total_students / (boys_ratio + girls_ratio)
  let total_boys := boys_ratio * total_groups
  (total_boys * 100 / total_students = 300 / 7) ∧ (21 - total_boys = 3) :=
by {
  sorry
}

end percentage_of_boys_and_additional_boys_l619_61930


namespace factorization_ce_sum_eq_25_l619_61993

theorem factorization_ce_sum_eq_25 {C E : ℤ} (h : (C * x - 13) * (E * x - 7) = 20 * x^2 - 87 * x + 91) : 
  C * E + C = 25 :=
sorry

end factorization_ce_sum_eq_25_l619_61993


namespace unique_intersections_l619_61997

def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := 5 * x + y = 1
def line3 (x y : ℝ) : Prop := 6 * x - 4 * y = 2

theorem unique_intersections :
  (∃ x1 y1, line1 x1 y1 ∧ line2 x1 y1) ∧
  (∃ x2 y2, line2 x2 y2 ∧ line3 x2 y2) ∧
  ¬ (∃ x y, line1 x y ∧ line3 x y) ∧
  (∀ x y x' y', (line1 x y ∧ line2 x y ∧ line2 x' y' ∧ line3 x' y') → (x = x' ∧ y = y')) :=
by
  sorry

end unique_intersections_l619_61997


namespace remainder_sum_l619_61964

theorem remainder_sum (n : ℤ) (h : n % 21 = 13) : (n % 3 + n % 7) = 7 := by
  sorry

end remainder_sum_l619_61964


namespace find_s2_side_length_l619_61921

-- Define the variables involved
variables (r s : ℕ)

-- Conditions based on problem statement
def height_eq : Prop := 2 * r + s = 2160
def width_eq : Prop := 2 * r + 3 * s + 110 = 4020

-- The theorem stating that s = 875 given the conditions
theorem find_s2_side_length (h1 : height_eq r s) (h2 : width_eq r s) : s = 875 :=
by {
  sorry
}

end find_s2_side_length_l619_61921


namespace slope_undefined_iff_vertical_l619_61965

theorem slope_undefined_iff_vertical (m : ℝ) :
  let M := (2 * m + 3, m)
  let N := (m - 2, 1)
  (2 * m + 3 - (m - 2) = 0 ∧ m - 1 ≠ 0) ↔ m = -5 :=
by
  sorry

end slope_undefined_iff_vertical_l619_61965


namespace blake_spent_60_on_mangoes_l619_61981

def spent_on_oranges : ℕ := 40
def spent_on_apples : ℕ := 50
def initial_amount : ℕ := 300
def change : ℕ := 150
def total_spent := initial_amount - change
def total_spent_on_fruits := spent_on_oranges + spent_on_apples
def spending_on_mangoes := total_spent - total_spent_on_fruits

theorem blake_spent_60_on_mangoes : spending_on_mangoes = 60 := 
by
  -- The proof will go here
  sorry

end blake_spent_60_on_mangoes_l619_61981


namespace even_square_even_square_even_even_l619_61953

-- Definition for a natural number being even
def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

-- Statement 1: If p is even, then p^2 is even
theorem even_square_even (p : ℕ) (hp : is_even p) : is_even (p * p) :=
sorry

-- Statement 2: If p^2 is even, then p is even
theorem square_even_even (p : ℕ) (hp_squared : is_even (p * p)) : is_even p :=
sorry

end even_square_even_square_even_even_l619_61953


namespace cost_price_of_article_l619_61945

theorem cost_price_of_article (SP : ℝ) (profit_percentage : ℝ) (profit_fraction : ℝ) (CP : ℝ) : 
  SP = 120 → profit_percentage = 25 → profit_fraction = profit_percentage / 100 → 
  SP = CP + profit_fraction * CP → CP = 96 :=
by intros hSP hprofit_percentage hprofit_fraction heq
   sorry

end cost_price_of_article_l619_61945


namespace polynomial_sum_l619_61967

variable {R : Type*} [CommRing R] {x y : R}

/-- Given that the sum of a polynomial P and x^2 - y^2 is x^2 + y^2, we want to prove that P is 2y^2. -/
theorem polynomial_sum (P : R) (h : P + (x^2 - y^2) = x^2 + y^2) : P = 2 * y^2 :=
by
  sorry

end polynomial_sum_l619_61967


namespace inequality_range_l619_61983

theorem inequality_range (k : ℝ) : (∀ x : ℝ, abs (x + 1) - abs (x - 2) > k) → k < -3 :=
by
  sorry

end inequality_range_l619_61983


namespace find_original_polynomial_calculate_correct_result_l619_61987

variable {P : Polynomial ℝ}
variable (Q : Polynomial ℝ := 2 * X ^ 2 + X - 5)
variable (R : Polynomial ℝ := X ^ 2 + 3 * X - 1)

theorem find_original_polynomial (h : P - Q = R) : P = 3 * X ^ 2 + 4 * X - 6 :=
by
  sorry

theorem calculate_correct_result (h : P = 3 * X ^ 2 + 4 * X - 6) : P - Q = X ^ 2 + X + 9 :=
by
  sorry

end find_original_polynomial_calculate_correct_result_l619_61987


namespace cookies_in_second_type_l619_61913

theorem cookies_in_second_type (x : ℕ) (h1 : 50 * 12 + 80 * x + 70 * 16 = 3320) : x = 20 :=
by sorry

end cookies_in_second_type_l619_61913


namespace cars_overtake_distance_l619_61906

def speed_red_car : ℝ := 30
def speed_black_car : ℝ := 50
def time_to_overtake : ℝ := 1
def distance_between_cars : ℝ := 20

theorem cars_overtake_distance :
  (speed_black_car - speed_red_car) * time_to_overtake = distance_between_cars :=
by sorry

end cars_overtake_distance_l619_61906


namespace factorize_expression_l619_61918

variable {R : Type*} [CommRing R] (a b : R)

theorem factorize_expression : 2 * a^2 * b - 4 * a * b + 2 * b = 2 * b * (a - 1)^2 :=
by
  sorry

end factorize_expression_l619_61918


namespace cost_effective_combination_l619_61992

/--
Jackson wants to impress his girlfriend by filling her hot tub with champagne.
The hot tub holds 400 liters of liquid. He has three types of champagne bottles:
1. Small bottle: Holds 0.75 liters with a price of $70 per bottle.
2. Medium bottle: Holds 1.5 liters with a price of $120 per bottle.
3. Large bottle: Holds 3 liters with a price of $220 per bottle.

If he purchases more than 50 bottles of any type, he will get a 10% discount on 
that type. If he purchases over 100 bottles of any type, he will get 20% off 
on that type of bottles. 

Prove that the most cost-effective combination of bottles for 
Jackson to purchase is 134 large bottles for a total cost of $23,584 after the discount.
-/
theorem cost_effective_combination :
  let volume := 400
  let small_bottle_volume := 0.75
  let small_bottle_cost := 70
  let medium_bottle_volume := 1.5
  let medium_bottle_cost := 120
  let large_bottle_volume := 3
  let large_bottle_cost := 220
  let discount_50 := 0.10
  let discount_100 := 0.20
  let cost_134_large_bottles := (134 * large_bottle_cost) * (1 - discount_100)
  cost_134_large_bottles = 23584 :=
sorry

end cost_effective_combination_l619_61992


namespace problem_1_problem_2_l619_61970

theorem problem_1 (x y : ℝ) (h1 : x - y = 3) (h2 : 3*x - 8*y = 14) : x = 2 ∧ y = -1 :=
sorry

theorem problem_2 (x y : ℝ) (h1 : 3*x + 4*y = 16) (h2 : 5*x - 6*y = 33) : x = 6 ∧ y = -1/2 :=
sorry

end problem_1_problem_2_l619_61970


namespace lawnmower_percentage_drop_l619_61955

theorem lawnmower_percentage_drop :
  ∀ (initial_value value_after_one_year value_after_six_months : ℝ)
    (percentage_drop_in_year : ℝ),
  initial_value = 100 →
  value_after_one_year = 60 →
  percentage_drop_in_year = 20 →
  value_after_one_year = (1 - percentage_drop_in_year / 100) * value_after_six_months →
  (initial_value - value_after_six_months) / initial_value * 100 = 25 :=
by
  intros initial_value value_after_one_year value_after_six_months percentage_drop_in_year
  intros h_initial h_value_after_one_year h_percentage_drop_in_year h_value_equation
  sorry

end lawnmower_percentage_drop_l619_61955


namespace find_angle_l619_61933

variable (x : ℝ)

theorem find_angle (h1 : x + (180 - x) = 180) (h2 : x + (90 - x) = 90) (h3 : 180 - x = 3 * (90 - x)) : x = 45 := 
by
  sorry

end find_angle_l619_61933


namespace probability_of_specific_combination_l619_61911

def total_shirts : ℕ := 3
def total_shorts : ℕ := 7
def total_socks : ℕ := 4
def total_clothes : ℕ := total_shirts + total_shorts + total_socks
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
def favorable_outcomes : ℕ := (choose total_shirts 2) * (choose total_shorts 1) * (choose total_socks 1)
def total_outcomes : ℕ := choose total_clothes 4

theorem probability_of_specific_combination :
  favorable_outcomes / total_outcomes = 84 / 1001 :=
by
  -- Proof omitted
  sorry

end probability_of_specific_combination_l619_61911


namespace area_enclosed_by_equation_l619_61989

theorem area_enclosed_by_equation :
  ∀ (x y : ℝ), (x^2 + y^2 - 4 * x + 10 * y = -20) → (∃ r : ℝ, r^2 = 9 ∧ ∃ c : ℝ × ℝ, (∃ a b, (x - a)^2 + (y - b)^2 = r^2)) :=
by
  sorry

end area_enclosed_by_equation_l619_61989


namespace verification_equation_3_conjecture_general_equation_l619_61966

theorem verification_equation_3 : 
  4 * Real.sqrt (4 / 15) = Real.sqrt (4 * (4 / 15)) :=
sorry

theorem conjecture :
  Real.sqrt (5 * (5 / 24)) = 5 * Real.sqrt (5 / 24) :=
sorry

theorem general_equation (n : ℕ) (h : 2 ≤ n) :
  n * Real.sqrt (n / (n^2 - 1)) = Real.sqrt (n + n / (n^2 - 1)) :=
sorry

end verification_equation_3_conjecture_general_equation_l619_61966


namespace bond_face_value_l619_61982

theorem bond_face_value
  (F : ℝ)
  (S : ℝ)
  (hS : S = 3846.153846153846)
  (hI1 : I = 0.05 * F)
  (hI2 : I = 0.065 * S) :
  F = 5000 :=
by
  sorry

end bond_face_value_l619_61982


namespace identical_prob_of_painted_cubes_l619_61991

/-
  Given:
  - Each face of a cube can be painted in one of 3 colors.
  - Each cube has 6 faces.
  - The total possible ways to paint both cubes is 531441.
  - The total ways to paint them such that they are identical after rotation is 66.

  Prove:
  - The probability of two painted cubes being identical after rotation is 2/16101.
-/
theorem identical_prob_of_painted_cubes :
  let total_ways := 531441
  let identical_ways := 66
  (identical_ways : ℚ) / total_ways = 2 / 16101 := by
  sorry

end identical_prob_of_painted_cubes_l619_61991


namespace goat_can_circle_around_tree_l619_61963

/-- 
  Given a goat tied with a rope of length 4.7 meters (L) near an old tree with a cylindrical trunk of radius 0.5 meters (R), 
  with the shortest distance from the stake to the surface of the tree being 1 meter (d), 
  prove that the minimal required rope length to encircle the tree and return to the stake is less than 
  or equal to the given rope length of 4.7 meters (L).
-/ 
theorem goat_can_circle_around_tree (L R d : ℝ) (hR : R = 0.5) (hd : d = 1) (hL : L = 4.7) : 
  ∃ L_min, L_min ≤ L := 
by
  -- Detailed proof steps omitted.
  sorry

end goat_can_circle_around_tree_l619_61963


namespace fraction_eval_l619_61946

theorem fraction_eval : 
    (1 / (3 - (1 / (3 - (1 / (3 - (1 / 4))))))) = (11 / 29) := 
by
  sorry

end fraction_eval_l619_61946


namespace solve_equation_l619_61949

theorem solve_equation (x : ℝ) :
  (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) → x = -9 :=
by 
  sorry

end solve_equation_l619_61949


namespace find_k_l619_61998

open Real

noncomputable def chord_intersection (k : ℝ) : Prop :=
  let R : ℝ := 3
  let d := abs (k + 1) / sqrt (1 + k^2)
  d^2 + (12 * sqrt 5 / 10)^2 = R^2

theorem find_k (k : ℝ) (h : k > 1) (h_intersect : chord_intersection k) : k = 2 := by
  sorry

end find_k_l619_61998


namespace stratified_sampling_females_l619_61934

theorem stratified_sampling_females :
  let total_employees := 200
  let male_employees := 120
  let female_employees := 80
  let sample_size := 20
  number_of_female_in_sample = (female_employees / total_employees) * sample_size := by
  sorry

end stratified_sampling_females_l619_61934


namespace task_completion_days_l619_61959

theorem task_completion_days (a b c: ℕ) :
  (b = a + 6) → (c = b + 3) → 
  (3 / a + 4 / b = 9 / c) →
  a = 18 ∧ b = 24 ∧ c = 27 :=
  by
  sorry

end task_completion_days_l619_61959


namespace part_a_part_b_l619_61909

-- Definitions of the basic tiles, colorings, and the proposition

inductive Color
| black : Color
| white : Color

structure Tile :=
(c00 c01 c10 c11 : Color)

-- Ali's forbidden tiles (6 types for part (a))
def forbiddenTiles_6 : List Tile := 
[ Tile.mk Color.black Color.white Color.white Color.white,
  Tile.mk Color.black Color.white Color.black Color.white,
  Tile.mk Color.black Color.white Color.white Color.black,
  Tile.mk Color.black Color.white Color.black Color.black,
  Tile.mk Color.black Color.black Color.black Color.black,
  Tile.mk Color.white Color.white Color.white Color.white
]

-- Ali's forbidden tiles (7 types for part (b))
def forbiddenTiles_7 : List Tile := 
[ Tile.mk Color.black Color.white Color.white Color.white,
  Tile.mk Color.black Color.white Color.black Color.white,
  Tile.mk Color.black Color.white Color.white Color.black,
  Tile.mk Color.black Color.white Color.black Color.black,
  Tile.mk Color.black Color.black Color.black Color.black,
  Tile.mk Color.white Color.white Color.white Color.white,
  Tile.mk Color.black Color.white Color.black Color.white
]

-- Propositions to be proved

-- Part (a): Mohammad can color the infinite table with no forbidden tiles present
theorem part_a :
  ∃f : ℕ × ℕ → Color, ∀ t ∈ forbiddenTiles_6, ∃ x y : ℕ, ¬(f (x, y) = t.c00 ∧ f (x, y+1) = t.c01 ∧ 
  f (x+1, y) = t.c10 ∧ f (x+1, y+1) = t.c11) := 
sorry

-- Part (b): Ali can present 7 forbidden tiles such that Mohammad cannot achieve his goal
theorem part_b :
  ∀ f : ℕ × ℕ → Color, ∃ t ∈ forbiddenTiles_7, ∃ x y : ℕ, (f (x, y) = t.c00 ∧ f (x, y+1) = t.c01 ∧ 
  f (x+1, y) = t.c10 ∧ f (x+1, y+1) = t.c11) := 
sorry

end part_a_part_b_l619_61909


namespace divisible_sum_l619_61915

theorem divisible_sum (k : ℕ) (n : ℕ) (h : n = 2^(k-1)) : 
  ∀ (S : Finset ℕ), S.card = 2*n - 1 → ∃ T ⊆ S, T.card = n ∧ T.sum id % n = 0 :=
by
  sorry

end divisible_sum_l619_61915


namespace apples_not_ripe_l619_61931

theorem apples_not_ripe (total_apples good_apples : ℕ) (h1 : total_apples = 14) (h2 : good_apples = 8) : total_apples - good_apples = 6 :=
by {
  sorry
}

end apples_not_ripe_l619_61931


namespace solve_equation_in_integers_l619_61958
-- Import the necessary library for Lean

-- Define the main theorem to solve the equation in integers
theorem solve_equation_in_integers :
  ∃ (xs : List (ℕ × ℕ)), (∀ x y, (3^x - 2^y = 1 → (x, y) ∈ xs)) ∧ xs = [(1, 1), (2, 3)] :=
by
  sorry

end solve_equation_in_integers_l619_61958


namespace savings_after_one_year_l619_61990

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem savings_after_one_year :
  compound_interest 1000 0.10 2 1 = 1102.50 :=
by
  sorry

end savings_after_one_year_l619_61990


namespace min_number_knights_l619_61912

theorem min_number_knights (h1 : ∃ n : ℕ, n = 7) (h2 : ∃ s : ℕ, s = 42) (h3 : ∃ l : ℕ, l = 24) :
  ∃ k : ℕ, k ≥ 0 ∧ k ≤ 7 ∧ k * (7 - k) = 12 ∧ k = 3 :=
by
  sorry

end min_number_knights_l619_61912


namespace number_of_students_taking_statistics_l619_61952

theorem number_of_students_taking_statistics
  (total_students : ℕ)
  (history_students : ℕ)
  (history_or_statistics : ℕ)
  (history_only : ℕ)
  (history_and_statistics : ℕ := history_students - history_only)
  (statistics_only : ℕ := history_or_statistics - history_and_statistics - history_only)
  (statistics_students : ℕ := history_and_statistics + statistics_only) :
  total_students = 90 → history_students = 36 → history_or_statistics = 59 → history_only = 29 →
    statistics_students = 30 :=
by
  intros
  -- Proof goes here but is omitted.
  sorry

end number_of_students_taking_statistics_l619_61952


namespace min_value_4x_plus_3y_l619_61999

theorem min_value_4x_plus_3y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + y = 5 * x * y) :
  4 * x + 3 * y ≥ 5 :=
sorry

end min_value_4x_plus_3y_l619_61999


namespace expand_product_l619_61986

theorem expand_product (x : ℤ) : 
  (3 * x + 4) * (2 * x - 6) = 6 * x^2 - 10 * x - 24 :=
by
  sorry

end expand_product_l619_61986


namespace range_of_k_intersecting_AB_l619_61935

theorem range_of_k_intersecting_AB 
  (A B : ℝ × ℝ) 
  (hA : A = (2, 7)) 
  (hB : B = (9, 6)) 
  (k : ℝ) 
  (hk : k ≠ 0) 
  (H : ∃ x : ℝ, A.2 = k * A.1 ∧ B.2 = k * B.1):
  (2 / 3) ≤ k ∧ k ≤ 7 / 2 :=
by sorry

end range_of_k_intersecting_AB_l619_61935


namespace range_of_a_l619_61925

noncomputable def has_solutions (a : ℝ) : Prop :=
  ∀ x : ℝ, 2 * a * 9^(Real.sin x) + 4 * a * 3^(Real.sin x) + a - 8 = 0

theorem range_of_a : ∀ a : ℝ,
  (has_solutions a ↔ (8 / 31 <= a ∧ a <= 72 / 23)) := sorry

end range_of_a_l619_61925


namespace solve_for_x_l619_61905

-- Define the given condition
def condition (x : ℝ) : Prop := (x - 5) ^ 3 = -((1 / 27)⁻¹)

-- State the problem as a Lean theorem
theorem solve_for_x : ∃ x : ℝ, condition x ∧ x = 2 := by
  sorry

end solve_for_x_l619_61905


namespace trigonometric_identity_l619_61996

noncomputable def tan_alpha : ℝ := 4

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = tan_alpha) :
  (Real.sin (Real.pi - α) - Real.sin (Real.pi / 2 + α)) / Real.cos (-α) = 3 :=
by
  sorry

end trigonometric_identity_l619_61996


namespace bug_converges_to_final_position_l619_61973

noncomputable def bug_final_position : ℝ × ℝ := 
  let horizontal_sum := ∑' n, if n % 4 = 0 then (1 / 4) ^ (n / 4) else 0
  let vertical_sum := ∑' n, if n % 4 = 1 then (1 / 4) ^ (n / 4) else 0
  (horizontal_sum, vertical_sum)

theorem bug_converges_to_final_position : bug_final_position = (4 / 5, 2 / 5) := 
  sorry

end bug_converges_to_final_position_l619_61973


namespace div_expression_l619_61961

variable {α : Type*} [Field α]

theorem div_expression (a b c : α) : 4 * a^2 * b^2 * c / (-2 * a * b^2) = -2 * a * c := by
  sorry

end div_expression_l619_61961


namespace clock_angle_3_to_7_l619_61962

theorem clock_angle_3_to_7 : 
  let number_of_rays := 12
  let total_degrees := 360
  let degree_per_ray := total_degrees / number_of_rays
  let angle_3_to_7 := 4 * degree_per_ray
  angle_3_to_7 = 120 :=
by
  sorry

end clock_angle_3_to_7_l619_61962


namespace elastic_collision_inelastic_collision_l619_61971

-- Given conditions for Case A and Case B
variables (L V : ℝ) (m : ℝ) -- L is length of the rods, V is the speed, m is mass of each sphere

-- Prove Case A: The dumbbells separate maintaining their initial velocities
theorem elastic_collision (h1 : L > 0) (h2 : V > 0) (h3 : m > 0) :
  -- After a perfectly elastic collision, the dumbbells separate maintaining their initial velocities
  true := sorry

-- Prove Case B: The dumbbells start rotating around the collision point with angular velocity V / (2 * L)
theorem inelastic_collision (h1 : L > 0) (h2 : V > 0) (h3 : m > 0) :
  -- After a perfectly inelastic collision, the dumbbells start rotating around the collision point with angular velocity V / (2 * L)
  true := sorry

end elastic_collision_inelastic_collision_l619_61971


namespace sin_half_alpha_l619_61919

theorem sin_half_alpha (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
    Real.sin (α / 2) = (-1 + Real.sqrt 5) / 4 := 
by
  sorry

end sin_half_alpha_l619_61919


namespace average_tree_height_l619_61901

theorem average_tree_height : 
  ∀ (T₁ T₂ T₃ T₄ T₅ T₆ : ℕ),
  T₂ = 27 ->
  ((T₁ = 3 * T₂) ∨ (T₁ = T₂ / 3)) ->
  ((T₃ = 3 * T₂) ∨ (T₃ = T₂ / 3)) ->
  ((T₄ = 3 * T₃) ∨ (T₄ = T₃ / 3)) ->
  ((T₅ = 3 * T₄) ∨ (T₅ = T₄ / 3)) ->
  ((T₆ = 3 * T₅) ∨ (T₆ = T₅ / 3)) ->
  (T₁ + T₂ + T₃ + T₄ + T₅ + T₆) / 6 = 22 := 
by 
  intros T₁ T₂ T₃ T₄ T₅ T₆ hT2 hT1 hT3 hT4 hT5 hT6
  sorry

end average_tree_height_l619_61901


namespace cone_and_sphere_volume_l619_61916

theorem cone_and_sphere_volume (π : ℝ) (r h : ℝ) (V_cylinder : ℝ) (V_cone V_sphere V_total : ℝ) 
  (h_cylinder : V_cylinder = 54 * π) 
  (h_radius : h = 3 * r)
  (h_cone : V_cone = (1 / 3) * π * r^2 * h) 
  (h_sphere : V_sphere = (4 / 3) * π * r^3) :
  V_total = 42 * π := 
by
  sorry

end cone_and_sphere_volume_l619_61916


namespace count_multiples_of_15_l619_61903

theorem count_multiples_of_15 (a b n : ℕ) (h_gte : 25 ≤ a) (h_lte : b ≤ 205) (h15 : n = 15) : 
  (∃ (k : ℕ), a ≤ k * n ∧ k * n ≤ b ∧ 1 ≤ k - 1 ∧ k - 1 ≤ 12) :=
sorry

end count_multiples_of_15_l619_61903


namespace find_k_l619_61975

def vector_a : ℝ × ℝ := (2, 1)
def vector_b (k : ℝ) : ℝ × ℝ := (k, 3)

theorem find_k (k : ℝ) :
  let sum_vector := (vector_a.1 + 2 * (vector_b k).1, vector_a.2 + 2 * (vector_b k).2)
  let diff_vector := (2 * vector_a.1 - (vector_b k).1, 2 * vector_a.2 - (vector_b k).2)
  sum_vector.1 * diff_vector.2 = sum_vector.2 * diff_vector.1
  → k = 6 :=
by
  sorry

end find_k_l619_61975


namespace number_of_dogs_is_112_l619_61968

-- Definitions based on the given conditions.
def ratio_dogs_to_cats_to_bunnies (D C B : ℕ) : Prop := 4 * C = 7 * D ∧ 9 * C = 7 * B
def total_dogs_and_bunnies (D B : ℕ) (total : ℕ) : Prop := D + B = total

-- The hypothesis and conclusion of the problem.
theorem number_of_dogs_is_112 (D C B : ℕ) (x : ℕ) (h1: ratio_dogs_to_cats_to_bunnies D C B) (h2: total_dogs_and_bunnies D B 364) : D = 112 :=
by 
  sorry

end number_of_dogs_is_112_l619_61968


namespace symmetric_points_coords_l619_61920

theorem symmetric_points_coords (a b : ℝ) :
    let N := (a, -b)
    let P := (-a, -b)
    let Q := (b, a)
    N = (a, -b) ∧ P = (-a, -b) ∧ Q = (b, a) →
    Q = (b, a) :=
by
  intro h
  sorry

end symmetric_points_coords_l619_61920


namespace prime_sum_is_prime_l619_61984

def prime : ℕ → Prop := sorry 

theorem prime_sum_is_prime (A B : ℕ) (hA : prime A) (hB : prime B) (hAB : prime (A - B)) (hABB : prime (A - B - B)) : prime (A + B + (A - B) + (A - B - B)) :=
sorry

end prime_sum_is_prime_l619_61984


namespace has_four_digits_l619_61969

def least_number_divisible (n: ℕ) : Prop := 
  n = 9600 ∧ 
  (∃ k1 k2 k3 k4: ℕ, n = 15 * k1 ∧ n = 25 * k2 ∧ n = 40 * k3 ∧ n = 75 * k4)

theorem has_four_digits : ∀ n: ℕ, least_number_divisible n → (Nat.digits 10 n).length = 4 :=
by
  intros n h
  sorry

end has_four_digits_l619_61969


namespace ratio_of_A_to_B_l619_61947

theorem ratio_of_A_to_B (total_weight compound_A_weight compound_B_weight : ℝ)
  (h1 : total_weight = 108)
  (h2 : compound_B_weight = 90)
  (h3 : compound_A_weight = total_weight - compound_B_weight) :
  compound_A_weight / compound_B_weight = 1 / 5 :=
by
  sorry

end ratio_of_A_to_B_l619_61947


namespace trader_goal_l619_61927

theorem trader_goal 
  (profit : ℕ)
  (half_profit : ℕ)
  (donation : ℕ)
  (total_funds : ℕ)
  (made_above_goal : ℕ)
  (goal : ℕ)
  (h1 : profit = 960)
  (h2 : half_profit = profit / 2)
  (h3 : donation = 310)
  (h4 : total_funds = half_profit + donation)
  (h5 : made_above_goal = 180)
  (h6 : goal = total_funds - made_above_goal) :
  goal = 610 :=
by 
  sorry

end trader_goal_l619_61927


namespace sum_of_coefficients_eq_minus_36_l619_61951

noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x ^ 2 + b * x + c

theorem sum_of_coefficients_eq_minus_36 
  (a b c : ℝ)
  (h_min : ∀ x, quadratic a b c x ≥ -36)
  (h_points : quadratic a b c (-3) = 0 ∧ quadratic a b c 5 = 0)
  : a + b + c = -36 :=
sorry

end sum_of_coefficients_eq_minus_36_l619_61951


namespace thirtieth_term_of_arithmetic_seq_l619_61929

def arithmetic_seq (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

theorem thirtieth_term_of_arithmetic_seq : 
  arithmetic_seq 3 4 30 = 119 := 
by
  sorry

end thirtieth_term_of_arithmetic_seq_l619_61929


namespace red_ball_probability_l619_61976

theorem red_ball_probability 
  (red_balls : ℕ)
  (black_balls : ℕ)
  (total_balls : ℕ)
  (h1 : red_balls = 3)
  (h2 : black_balls = 9)
  (h3 : total_balls = red_balls + black_balls) :
  (red_balls : ℚ) / total_balls = 1 / 4 :=
by
  sorry

end red_ball_probability_l619_61976


namespace negation_of_p_l619_61902

theorem negation_of_p (p : Prop) : (∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔ (∀ x : ℝ, x > 0 → ¬ ((x + 1) * Real.exp x > 1)) :=
by
  sorry

end negation_of_p_l619_61902


namespace roots_are_integers_l619_61900

theorem roots_are_integers (a b : ℤ) (h_discriminant : ∃ (q r : ℚ), r ≠ 0 ∧ a^2 - 4 * b = (q/r)^2) : 
  ∃ x y : ℤ, x^2 - a * x + b = 0 ∧ y^2 - a * y + b = 0 := 
sorry

end roots_are_integers_l619_61900


namespace total_pencils_correct_l619_61974

def pencils_per_child := 4
def num_children := 8
def total_pencils := pencils_per_child * num_children

theorem total_pencils_correct : total_pencils = 32 := by
  sorry

end total_pencils_correct_l619_61974


namespace allocate_25_rubles_in_4_weighings_l619_61904

theorem allocate_25_rubles_in_4_weighings :
  ∃ (coins : ℕ) (coins5 : ℕ → ℕ), 
    (coins = 1600) ∧ 
    (coins5 0 = 800 ∧ coins5 1 = 800) ∧
    (coins5 2 = 400 ∧ coins5 3 = 400) ∧
    (coins5 4 = 200 ∧ coins5 5 = 200) ∧
    (coins5 6 = 100 ∧ coins5 7 = 100) ∧
    (
      25 = 20 + 5 ∧ 
      (∃ i j k l m n, coins5 i = 400 ∧ coins5 j = 400 ∧ coins5 k = 200 ∧
        coins5 l = 200 ∧ coins5 m = 100 ∧ coins5 n = 100)
    )
  := 
sorry

end allocate_25_rubles_in_4_weighings_l619_61904


namespace ArianaBoughtTulips_l619_61917

theorem ArianaBoughtTulips (total_flowers : ℕ) (fraction_roses : ℚ) (carnations : ℕ) 
    (h_total : total_flowers = 40) (h_fraction : fraction_roses = 2/5) (h_carnations : carnations = 14) : 
    total_flowers - (total_flowers * fraction_roses + carnations) = 10 := by
  sorry

end ArianaBoughtTulips_l619_61917


namespace probability_age_between_30_and_40_l619_61936

-- Assume total number of people in the group is 200
def total_people : ℕ := 200

-- Assume 80 people have an age of more than 40 years
def people_age_more_than_40 : ℕ := 80

-- Assume 70 people have an age between 30 and 40 years
def people_age_between_30_and_40 : ℕ := 70

-- Assume 30 people have an age between 20 and 30 years
def people_age_between_20_and_30 : ℕ := 30

-- Assume 20 people have an age of less than 20 years
def people_age_less_than_20 : ℕ := 20

-- The proof problem statement
theorem probability_age_between_30_and_40 :
  (people_age_between_30_and_40 : ℚ) / (total_people : ℚ) = 7 / 20 :=
by
  sorry

end probability_age_between_30_and_40_l619_61936


namespace problem1_solution_set_problem2_range_of_m_l619_61922

def f (x : ℝ) : ℝ := |x - 3| - 5
def g (x : ℝ) : ℝ := |x + 2| - 2

theorem problem1_solution_set :
  {x : ℝ | f x ≤ 2} = {x : ℝ | -4 ≤ x ∧ x ≤ 10} := 
sorry

theorem problem2_range_of_m (m : ℝ) (h : ∃ x : ℝ, f x - g x ≥ m - 3) :
  m ≤ 5 :=
sorry

end problem1_solution_set_problem2_range_of_m_l619_61922


namespace expenditure_representation_l619_61942

theorem expenditure_representation (income expenditure : ℤ)
  (h_income : income = 60)
  (h_expenditure : expenditure = 40) :
  -expenditure = -40 :=
by {
  sorry
}

end expenditure_representation_l619_61942


namespace ratio_of_c_to_d_l619_61941

theorem ratio_of_c_to_d (x y c d : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0)
    (h1 : 9 * x - 6 * y = c) (h2 : 15 * x - 10 * y = d) :
    c / d = -2 / 5 :=
by
  sorry

end ratio_of_c_to_d_l619_61941


namespace jan_clean_car_water_l619_61960

def jan_water_problem
  (initial_water : ℕ)
  (car_water : ℕ)
  (plant_additional : ℕ)
  (plate_clothes_water : ℕ)
  (remaining_water : ℕ)
  (used_water : ℕ)
  (car_cleaning_water : ℕ) : Prop :=
  initial_water = 65 ∧
  plate_clothes_water = 24 ∧
  plant_additional = 11 ∧
  remaining_water = 2 * plate_clothes_water ∧
  used_water = initial_water - remaining_water ∧
  car_water = used_water + plant_additional ∧
  car_cleaning_water = car_water / 4

theorem jan_clean_car_water : jan_water_problem 65 17 11 24 48 17 7 :=
by {
  sorry
}

end jan_clean_car_water_l619_61960


namespace cylinder_height_to_diameter_ratio_l619_61908

theorem cylinder_height_to_diameter_ratio
  (r h : ℝ)
  (inscribed_sphere : h = 2 * r)
  (cylinder_volume : π * r^2 * h = 3 * (4/3) * π * r^3) :
  (h / (2 * r)) = 2 :=
by
  sorry

end cylinder_height_to_diameter_ratio_l619_61908


namespace men_absent_l619_61943

theorem men_absent (n : ℕ) (d1 d2 : ℕ) (x : ℕ) 
  (h1 : n = 22) 
  (h2 : d1 = 20) 
  (h3 : d2 = 22) 
  (hc : n * d1 = (n - x) * d2) : 
  x = 2 := 
by {
  sorry
}

end men_absent_l619_61943


namespace investment_schemes_correct_l619_61977

-- Define the parameters of the problem
def num_projects : Nat := 3
def num_districts : Nat := 4

-- Function to count the number of valid investment schemes
def count_investment_schemes (num_projects num_districts : Nat) : Nat :=
  let total_schemes := num_districts ^ num_projects
  let invalid_schemes := num_districts
  total_schemes - invalid_schemes

-- Theorem statement
theorem investment_schemes_correct :
  count_investment_schemes num_projects num_districts = 60 := by
  sorry

end investment_schemes_correct_l619_61977


namespace exponentiation_comparison_l619_61914

theorem exponentiation_comparison :
  1.7 ^ 0.3 > 0.9 ^ 0.3 :=
by sorry

end exponentiation_comparison_l619_61914


namespace part_I_part_II_l619_61985

-- Definition of functions
def f (x a : ℝ) := |3 * x - a|
def g (x : ℝ) := |x + 1|

-- Part (I): Solution set for f(x) < 3 when a = 4
theorem part_I (x : ℝ) : f x 4 < 3 ↔ (1 / 3 < x ∧ x < 7 / 3) :=
by 
  sorry

-- Part (II): Range of a such that f(x) + g(x) > 1 for all x in ℝ
theorem part_II (a : ℝ) : (∀ x : ℝ, f x a + g x > 1) ↔ (a < -6 ∨ a > 0) :=
by 
  sorry

end part_I_part_II_l619_61985


namespace worth_of_entire_lot_l619_61948

theorem worth_of_entire_lot (half_share : ℝ) (amount_per_tenth : ℝ) (total_amount : ℝ) :
  half_share = 0.5 →
  amount_per_tenth = 460 →
  total_amount = (amount_per_tenth * 10) →
  (total_amount * 2) = 9200 :=
by
  intros h1 h2 h3
  sorry

end worth_of_entire_lot_l619_61948


namespace minimum_cost_l619_61937

theorem minimum_cost (
    x y m w : ℝ) 
    (h1 : 4 * x + 2 * y = 400)
    (h2 : 2 * x + 4 * y = 320)
    (h3 : m ≥ 16)
    (h4 : m + (80 - m) = 80)
    (h5 : w = 80 * m + 40 * (80 - m)) :
    x = 80 ∧ y = 40 ∧ w = 3840 :=
by 
  sorry

end minimum_cost_l619_61937


namespace vector_odot_not_symmetric_l619_61988

-- Define the vector operation ⊛
def vector_odot (a b : ℝ × ℝ) : ℝ :=
  let (m, n) := a
  let (p, q) := b
  m * q - n * p

-- Statement: Prove that the operation is not symmetric
theorem vector_odot_not_symmetric (a b : ℝ × ℝ) : vector_odot a b ≠ vector_odot b a := by
  sorry

end vector_odot_not_symmetric_l619_61988


namespace teacher_A_realizes_fish_l619_61938

variable (Teacher : Type) (has_fish : Teacher → Prop) (is_laughing : Teacher → Prop)
variables (A B C : Teacher)

-- Initial assumptions
axiom all_laughing : is_laughing A ∧ is_laughing B ∧ is_laughing C
axiom each_thinks_others_have_fish : (¬has_fish A ∧ has_fish B ∧ has_fish C) 
                                      ∨ (has_fish A ∧ ¬has_fish B ∧ has_fish C)
                                      ∨ (has_fish A ∧ has_fish B ∧ ¬has_fish C)

-- The logical conclusion
theorem teacher_A_realizes_fish : (∃ A B C : Teacher, 
  is_laughing A ∧ is_laughing B ∧ is_laughing C ∧
  ((¬has_fish A ∧ has_fish B ∧ has_fish C)
  ∨ (has_fish A ∧ ¬has_fish B ∧ has_fish C)
  ∨ (has_fish A ∧ has_fish B ∧ ¬has_fish C))) →
  (has_fish A ∧ is_laughing B ∧ is_laughing C) :=
sorry -- proof not required.

end teacher_A_realizes_fish_l619_61938


namespace larger_cross_section_distance_l619_61957

theorem larger_cross_section_distance
  (h_area1 : ℝ)
  (h_area2 : ℝ)
  (dist_planes : ℝ)
  (h_area1_val : h_area1 = 256 * Real.sqrt 2)
  (h_area2_val : h_area2 = 576 * Real.sqrt 2)
  (dist_planes_val : dist_planes = 10) :
  ∃ h : ℝ, h = 30 :=
by
  sorry

end larger_cross_section_distance_l619_61957
