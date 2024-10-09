import Mathlib

namespace hoseok_add_8_l322_32224

theorem hoseok_add_8 (x : ℕ) (h : 6 * x = 72) : x + 8 = 20 :=
sorry

end hoseok_add_8_l322_32224


namespace watermelon_juice_percentage_l322_32211

theorem watermelon_juice_percentage :
  ∀ (total_ounces orange_juice_percent grape_juice_ounces : ℕ), 
  orange_juice_percent = 25 →
  grape_juice_ounces = 70 →
  total_ounces = 200 →
  ((total_ounces - (orange_juice_percent * total_ounces / 100 + grape_juice_ounces)) / total_ounces) * 100 = 40 :=
by
  intros total_ounces orange_juice_percent grape_juice_ounces h1 h2 h3
  sorry

end watermelon_juice_percentage_l322_32211


namespace divides_5n_4n_iff_n_is_multiple_of_3_l322_32227

theorem divides_5n_4n_iff_n_is_multiple_of_3 (n : ℕ) (h : n > 0) : 
  61 ∣ (5^n - 4^n) ↔ ∃ k : ℕ, n = 3 * k :=
by
  sorry

end divides_5n_4n_iff_n_is_multiple_of_3_l322_32227


namespace koby_sparklers_correct_l322_32201

-- Define the number of sparklers in each of Koby's boxes as a variable
variable (S : ℕ)

-- Specify the conditions
def koby_sparklers : ℕ := 2 * S
def koby_whistlers : ℕ := 2 * 5
def cherie_sparklers : ℕ := 8
def cherie_whistlers : ℕ := 9
def total_fireworks : ℕ := koby_sparklers S + koby_whistlers + cherie_sparklers + cherie_whistlers

-- The theorem to prove that the number of sparklers in each of Koby's boxes is 3
theorem koby_sparklers_correct : total_fireworks S = 33 → S = 3 := by
  sorry

end koby_sparklers_correct_l322_32201


namespace percentage_increase_visitors_l322_32275

theorem percentage_increase_visitors 
  (V_Oct : ℕ)
  (V_Nov V_Dec : ℕ)
  (h1 : V_Oct = 100)
  (h2 : V_Dec = V_Nov + 15)
  (h3 : V_Oct + V_Nov + V_Dec = 345) : 
  (V_Nov - V_Oct) * 100 / V_Oct = 15 := 
by 
  sorry

end percentage_increase_visitors_l322_32275


namespace minimum_distance_l322_32297

theorem minimum_distance (m n : ℝ) (a : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ 4) 
  (h3 : m * Real.sqrt (Real.log a - 1 / 4) + 2 * a + 1 / 2 * n = 0) : 
  Real.sqrt (m^2 + n^2) = 4 * Real.sqrt (Real.log 2) / Real.log 2 :=
sorry

end minimum_distance_l322_32297


namespace surface_area_ratio_volume_ratio_l322_32208

-- Given conditions
def tetrahedron_surface_area (S : ℝ) : ℝ := 4 * S
def tetrahedron_volume (V : ℝ) : ℝ := 27 * V
def polyhedron_G_surface_area (S : ℝ) : ℝ := 28 * S
def polyhedron_G_volume (V : ℝ) : ℝ := 23 * V

-- Statements to prove
theorem surface_area_ratio (S : ℝ) (h1 : S > 0) :
  tetrahedron_surface_area S / polyhedron_G_surface_area S = 9 / 7 := by
  simp [tetrahedron_surface_area, polyhedron_G_surface_area]
  sorry

theorem volume_ratio (V : ℝ) (h1 : V > 0) :
  tetrahedron_volume V / polyhedron_G_volume V = 27 / 23 := by
  simp [tetrahedron_volume, polyhedron_G_volume]
  sorry

end surface_area_ratio_volume_ratio_l322_32208


namespace beaver_hid_36_carrots_l322_32256

variable (x y : ℕ)

-- Conditions
def beaverCarrots := 4 * x
def bunnyCarrots := 6 * y

-- Given that both animals hid the same total number of carrots
def totalCarrotsEqual := beaverCarrots x = bunnyCarrots y

-- Bunny used 3 fewer burrows than the beaver
def bunnyBurrows := y = x - 3

-- The goal is to show the beaver hid 36 carrots
theorem beaver_hid_36_carrots (H1 : totalCarrotsEqual x y) (H2 : bunnyBurrows x y) : beaverCarrots x = 36 := by
  sorry

end beaver_hid_36_carrots_l322_32256


namespace circle_radius_l322_32273

theorem circle_radius {C : ℝ → ℝ → Prop} (h1 : C 4 0) (h2 : C (-4) 0) : ∃ r : ℝ, r = 4 :=
by
  -- sorry for brevity
  sorry

end circle_radius_l322_32273


namespace pages_per_day_l322_32230

theorem pages_per_day (total_pages : ℕ) (days : ℕ) (h1 : total_pages = 63) (h2 : days = 3) : total_pages / days = 21 :=
by
  sorry

end pages_per_day_l322_32230


namespace equivalent_math_problem_l322_32289

noncomputable def P : ℝ := Real.sqrt 1011 + Real.sqrt 1012
noncomputable def Q : ℝ := - (Real.sqrt 1011 + Real.sqrt 1012)
noncomputable def R : ℝ := Real.sqrt 1011 - Real.sqrt 1012
noncomputable def S : ℝ := Real.sqrt 1012 - Real.sqrt 1011

theorem equivalent_math_problem :
  (P * Q)^2 * R * S = 8136957 :=
by
  sorry

end equivalent_math_problem_l322_32289


namespace gcd_of_459_and_357_l322_32247

theorem gcd_of_459_and_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_of_459_and_357_l322_32247


namespace angle_triple_complement_l322_32270

theorem angle_triple_complement (x : ℝ) (h1 : x + (180 - x) = 180) (h2 : x = 3 * (180 - x)) : x = 135 := 
by
  sorry

end angle_triple_complement_l322_32270


namespace transformed_graph_equation_l322_32292

theorem transformed_graph_equation (x y x' y' : ℝ)
  (h1 : x' = 5 * x)
  (h2 : y' = 3 * y)
  (h3 : x^2 + y^2 = 1) :
  x'^2 / 25 + y'^2 / 9 = 1 :=
by
  sorry

end transformed_graph_equation_l322_32292


namespace chimney_height_theorem_l322_32250

noncomputable def chimney_height :=
  let BCD := 75 * Real.pi / 180
  let BDC := 60 * Real.pi / 180
  let CBD := 45 * Real.pi / 180
  let CD := 40
  let BC := CD * Real.sin BDC / Real.sin CBD
  let CE := 1
  let elevation := 30 * Real.pi / 180
  let AB := CE + (Real.tan elevation * BC)
  AB

theorem chimney_height_theorem : chimney_height = 1 + 20 * Real.sqrt 2 :=
by
  sorry

end chimney_height_theorem_l322_32250


namespace exists_sum_of_three_l322_32281

theorem exists_sum_of_three {a b c d : ℕ} 
  (h1 : Nat.Coprime a b) 
  (h2 : Nat.Coprime a c) 
  (h3 : Nat.Coprime a d)
  (h4 : Nat.Coprime b c) 
  (h5 : Nat.Coprime b d) 
  (h6 : Nat.Coprime c d) 
  (h7 : a * b + c * d = a * c - 10 * b * d) :
  ∃ x y z, (x = a ∨ x = b ∨ x = c ∨ x = d) ∧ 
           (y = a ∨ y = b ∨ y = c ∨ y = d) ∧ 
           (z = a ∨ z = b ∨ z = c ∨ z = d) ∧ 
           x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ 
           (x = y + z ∨ y = x + z ∨ z = x + y) :=
by
  sorry

end exists_sum_of_three_l322_32281


namespace lyndee_friends_count_l322_32244

-- Definitions
variables (total_chicken total_garlic_bread : ℕ)
variables (lyndee_chicken lyndee_garlic_bread : ℕ)
variables (friends_large_chicken_count : ℕ)
variables (friends_large_chicken : ℕ)
variables (friend_garlic_bread_per_friend : ℕ)

def remaining_chicken (total_chicken lyndee_chicken friends_large_chicken_count friends_large_chicken : ℕ) : ℕ :=
  total_chicken - (lyndee_chicken + friends_large_chicken_count * friends_large_chicken)

def remaining_garlic_bread (total_garlic_bread lyndee_garlic_bread : ℕ) : ℕ :=
  total_garlic_bread - lyndee_garlic_bread

def total_friends (friends_large_chicken_count remaining_chicken remaining_garlic_bread friend_garlic_bread_per_friend : ℕ) : ℕ :=
  friends_large_chicken_count + remaining_chicken + remaining_garlic_bread / friend_garlic_bread_per_friend

-- Theorem statement
theorem lyndee_friends_count : 
  total_chicken = 11 → 
  total_garlic_bread = 15 →
  lyndee_chicken = 1 →
  lyndee_garlic_bread = 1 →
  friends_large_chicken_count = 3 →
  friends_large_chicken = 2 →
  friend_garlic_bread_per_friend = 3 →
  total_friends friends_large_chicken_count 
                (remaining_chicken total_chicken lyndee_chicken friends_large_chicken_count friends_large_chicken)
                (remaining_garlic_bread total_garlic_bread lyndee_garlic_bread)
                friend_garlic_bread_per_friend = 7 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7
  -- Proof omitted
  sorry

end lyndee_friends_count_l322_32244


namespace evaluate_expression_l322_32215

variable (x y : ℝ)

theorem evaluate_expression :
  (1 + x^2 + y^3) * (1 - x^3 - y^3) = 1 + x^2 - x^3 - y^3 - x^5 - x^2 * y^3 - x^3 * y^3 - y^6 :=
by
  sorry

end evaluate_expression_l322_32215


namespace Felix_can_lift_150_pounds_l322_32223

theorem Felix_can_lift_150_pounds : ∀ (weightFelix weightBrother : ℝ),
  (weightBrother = 2 * weightFelix) →
  (3 * weightBrother = 600) →
  (Felix_can_lift = 1.5 * weightFelix) →
  Felix_can_lift = 150 :=
by
  intros weightFelix weightBrother h1 h2 h3
  sorry

end Felix_can_lift_150_pounds_l322_32223


namespace max_neg_square_in_interval_l322_32209

variable (f : ℝ → ℝ)

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, 0 < x → x < y → f x < f y

noncomputable def neg_square_val (x : ℝ) : ℝ :=
  - (f x) ^ 2

theorem max_neg_square_in_interval : 
  (∀ x_1 x_2 : ℝ, f (x_1 + x_2) = f x_1 + f x_2) →
  f 1 = 2 →
  is_increasing f →
  (∀ x : ℝ, f (-x) = - f x) →
  ∃ b ∈ (Set.Icc (-3) (-2)), 
  ∀ x ∈ (Set.Icc (-3) (-2)), neg_square_val f x ≤ neg_square_val f b ∧ neg_square_val f b = -16 := 
sorry

end max_neg_square_in_interval_l322_32209


namespace transfers_l322_32222

variable (x : ℕ)
variable (gA gB gC : ℕ)

noncomputable def girls_in_A := x + 4
noncomputable def girls_in_B := x
noncomputable def girls_in_C := x - 1

variable (trans_A_to_B : ℕ)
variable (trans_B_to_C : ℕ)
variable (trans_C_to_A : ℕ)

axiom C_to_A_girls : trans_C_to_A = 2
axiom equal_girls : gA = x + 1 ∧ gB = x + 1 ∧ gC = x + 1

theorem transfers (hA : gA = girls_in_A - trans_A_to_B + trans_C_to_A)
                  (hB : gB = girls_in_B - trans_B_to_C + trans_A_to_B)
                  (hC : gC = girls_in_C - trans_C_to_A + trans_B_to_C) :
  trans_A_to_B = 5 ∧ trans_B_to_C = 4 :=
by
  sorry

end transfers_l322_32222


namespace correct_regression_equation_l322_32238

-- Problem Statement
def negatively_correlated (x y : ℝ) : Prop := sorry -- Define negative correlation for x, y
def sample_mean_x : ℝ := 3
def sample_mean_y : ℝ := 3.5
def regression_equation (b0 b1 : ℝ) (x : ℝ) : ℝ := b0 + b1 * x

theorem correct_regression_equation 
    (H_neg_corr : negatively_correlated x y) :
    regression_equation 9.5 (-2) sample_mean_x = sample_mean_y :=
by
    -- The proof will go here, skipping with sorry
    sorry

end correct_regression_equation_l322_32238


namespace customer_paid_amount_l322_32218

theorem customer_paid_amount (O : ℕ) (D : ℕ) (P : ℕ) (hO : O = 90) (hD : D = 20) (hP : P = O - D) : P = 70 :=
sorry

end customer_paid_amount_l322_32218


namespace range_of_expression_l322_32221

theorem range_of_expression (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a * b = 2) :
  (a^2 + b^2) / (a - b) ≤ -4 :=
sorry

end range_of_expression_l322_32221


namespace prove_incorrect_statement_l322_32277

-- Definitions based on given conditions
def isIrrational (x : ℝ) : Prop := ¬ ∃ a b : ℚ, x = a / b ∧ b ≠ 0
def isSquareRoot (x y : ℝ) : Prop := x * x = y
def hasSquareRoot (x : ℝ) : Prop := ∃ y : ℝ, isSquareRoot y x

-- Options translated into Lean
def optionA : Prop := ∀ x : ℝ, isIrrational x → ¬ hasSquareRoot x
def optionB (x : ℝ) : Prop := 0 < x → ∃ y : ℝ, y * y = x ∧ (-y) * (-y) = x
def optionC : Prop := isSquareRoot 0 0
def optionD (a : ℝ) : Prop := ∀ x : ℝ, x = -a → (x ^ 3 = - (a ^ 3))

-- The incorrect statement according to the solution
def incorrectStatement : Prop := optionA

-- The theorem to be proven
theorem prove_incorrect_statement : incorrectStatement :=
by
  -- Replace with the actual proof, currently a placeholder using sorry
  sorry

end prove_incorrect_statement_l322_32277


namespace car_travel_distance_l322_32246

-- Definitions based on the problem
def arith_seq_sum (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * a1 + d * (n * (n - 1)) / 2

-- Main statement to prove
theorem car_travel_distance : arith_seq_sum 40 (-12) 5 = 88 :=
by sorry

end car_travel_distance_l322_32246


namespace hyperbola_range_m_l322_32269

theorem hyperbola_range_m (m : ℝ) : 
  (∃ x y : ℝ, (x^2 / (16 - m)) + (y^2 / (9 - m)) = 1) → 9 < m ∧ m < 16 :=
by 
  sorry

end hyperbola_range_m_l322_32269


namespace sum_three_consecutive_integers_divisible_by_three_l322_32271

theorem sum_three_consecutive_integers_divisible_by_three (a : ℕ) (h : 1 < a) :
  (a - 1) + a + (a + 1) % 3 = 0 :=
by
  sorry

end sum_three_consecutive_integers_divisible_by_three_l322_32271


namespace Moscow_1975_p_q_r_equal_primes_l322_32272

theorem Moscow_1975_p_q_r_equal_primes (a b c : ℕ) (p q r : ℕ) 
  (hp : p = b^c + a) 
  (hq : q = a^b + c) 
  (hr : r = c^a + b) 
  (prime_p : Prime p) 
  (prime_q : Prime q) 
  (prime_r : Prime r) : 
  q = r :=
sorry

end Moscow_1975_p_q_r_equal_primes_l322_32272


namespace natural_numbers_solution_l322_32200

theorem natural_numbers_solution (a : ℕ) :
  ∃ k n : ℕ, k = 3 * a - 2 ∧ n = 2 * a - 1 ∧ (7 * k + 15 * n - 1) % (3 * k + 4 * n) = 0 :=
sorry

end natural_numbers_solution_l322_32200


namespace average_chem_math_l322_32255

theorem average_chem_math (P C M : ℕ) (h : P + C + M = P + 180) : (C + M) / 2 = 90 :=
  sorry

end average_chem_math_l322_32255


namespace isosceles_triangles_with_perimeter_21_l322_32235

theorem isosceles_triangles_with_perimeter_21 : 
  ∃ n : ℕ, n = 5 ∧ (∀ (a b c : ℕ), a ≤ b ∧ b = c ∧ a + 2*b = 21 → 1 ≤ a ∧ a ≤ 10) :=
sorry

end isosceles_triangles_with_perimeter_21_l322_32235


namespace find_k_exists_p3_p5_no_number_has_p2_and_p4_l322_32295

def has_prop_pk (n k : ℕ) : Prop := ∃ lst : List ℕ, (∀ x ∈ lst, x > 1) ∧ (lst.length = k) ∧ (lst.prod = n)

theorem find_k_exists_p3_p5 :
  ∃ (k : ℕ), (k = 3) ∧ ∃ (n : ℕ), has_prop_pk n k ∧ has_prop_pk n (k + 2) :=
by {
  sorry
}

theorem no_number_has_p2_and_p4 :
  ¬ ∃ (n : ℕ), has_prop_pk n 2 ∧ has_prop_pk n 4 :=
by {
  sorry
}

end find_k_exists_p3_p5_no_number_has_p2_and_p4_l322_32295


namespace problem_solution_l322_32205

variable {a b c d : ℝ}
variable (h_a : a = 4 * π / 3)
variable (h_b : b = 10 * π)
variable (h_c : c = 62)
variable (h_d : d = 30)

theorem problem_solution : (b * c) / (a * d) = 15.5 :=
by
  rw [h_a, h_b, h_c, h_d]
  -- Continued steps according to identified solution steps
  -- and arithmetic operations.
  sorry

end problem_solution_l322_32205


namespace quadratic_has_one_solution_l322_32294

theorem quadratic_has_one_solution (q : ℚ) (hq : q ≠ 0) : 
  (∃ x, ∀ y, q*y^2 - 18*y + 8 = 0 → x = y) ↔ q = 81 / 8 :=
by
  sorry

end quadratic_has_one_solution_l322_32294


namespace children_difference_l322_32296

-- Define the initial number of children on the bus
def initial_children : ℕ := 5

-- Define the number of children who got off the bus
def children_off : ℕ := 63

-- Define the number of children on the bus after more got on
def final_children : ℕ := 14

-- Define the number of children who got on the bus
def children_on : ℕ := (final_children + children_off) - initial_children

-- Prove the number of children who got on minus the number of children who got off is equal to 9
theorem children_difference :
  (children_on - children_off) = 9 :=
by
  -- Direct translation from the proof steps
  sorry

end children_difference_l322_32296


namespace r_daily_earning_l322_32239

-- Definitions from conditions in the problem
def earnings_of_all (P Q R : ℕ) : Prop := 9 * (P + Q + R) = 1620
def earnings_p_and_r (P R : ℕ) : Prop := 5 * (P + R) = 600
def earnings_q_and_r (Q R : ℕ) : Prop := 7 * (Q + R) = 910

-- Theorem to prove the daily earnings of r
theorem r_daily_earning (P Q R : ℕ) 
    (h1 : earnings_of_all P Q R)
    (h2 : earnings_p_and_r P R)
    (h3 : earnings_q_and_r Q R) : 
    R = 70 := 
by 
  sorry

end r_daily_earning_l322_32239


namespace cafeteria_seats_taken_l322_32229

def table1_count : ℕ := 10
def table1_seats : ℕ := 8
def table2_count : ℕ := 5
def table2_seats : ℕ := 12
def table3_count : ℕ := 5
def table3_seats : ℕ := 10
noncomputable def unseated_ratio1 : ℝ := 1/4
noncomputable def unseated_ratio2 : ℝ := 1/3
noncomputable def unseated_ratio3 : ℝ := 1/5

theorem cafeteria_seats_taken : 
  ((table1_count * table1_seats) - (unseated_ratio1 * (table1_count * table1_seats))) + 
  ((table2_count * table2_seats) - (unseated_ratio2 * (table2_count * table2_seats))) + 
  ((table3_count * table3_seats) - (unseated_ratio3 * (table3_count * table3_seats))) = 140 :=
by sorry

end cafeteria_seats_taken_l322_32229


namespace fraction_B_compared_to_A_and_C_l322_32274

theorem fraction_B_compared_to_A_and_C
    (A B C : ℕ) 
    (h1 : A = (B + C) / 3) 
    (h2 : A = B + 35) 
    (h3 : A + B + C = 1260) : 
    (∃ x : ℚ, B = x * (A + C) ∧ x = 2 / 7) :=
by
  sorry

end fraction_B_compared_to_A_and_C_l322_32274


namespace max_value_of_expression_l322_32214

theorem max_value_of_expression (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h_sum : x + y + z = 3) :
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2) ≤ 12 := 
sorry

end max_value_of_expression_l322_32214


namespace donna_babysitting_hours_l322_32217

theorem donna_babysitting_hours 
  (total_earnings : ℝ)
  (dog_walking_hours : ℝ)
  (dog_walking_rate : ℝ)
  (dog_walking_days : ℝ)
  (card_shop_hours : ℝ)
  (card_shop_rate : ℝ)
  (card_shop_days : ℝ)
  (babysitting_rate : ℝ)
  (days : ℝ)
  (total_dog_walking_earnings : ℝ := dog_walking_hours * dog_walking_rate * dog_walking_days)
  (total_card_shop_earnings : ℝ := card_shop_hours * card_shop_rate * card_shop_days)
  (total_earnings_dog_card : ℝ := total_dog_walking_earnings + total_card_shop_earnings)
  (babysitting_hours : ℝ := (total_earnings - total_earnings_dog_card) / babysitting_rate) :
  total_earnings = 305 → dog_walking_hours = 2 → dog_walking_rate = 10 → dog_walking_days = 5 →
  card_shop_hours = 2 → card_shop_rate = 12.5 → card_shop_days = 5 →
  babysitting_rate = 10 → babysitting_hours = 8 :=
by
  intros
  sorry

end donna_babysitting_hours_l322_32217


namespace MissyTotalTVTime_l322_32267

theorem MissyTotalTVTime :
  let reality_shows := [28, 35, 42, 39, 29]
  let cartoons := [10, 10]
  let ad_breaks := [8, 6, 12]
  let total_time := reality_shows.sum + cartoons.sum + ad_breaks.sum
  total_time = 219 := by
{
  -- Lean proof logic goes here (proof not requested)
  sorry
}

end MissyTotalTVTime_l322_32267


namespace product_of_dodecagon_l322_32258

open Complex

theorem product_of_dodecagon (Q : Fin 12 → ℂ) (h₁ : Q 0 = 2) (h₇ : Q 6 = 8) :
  (Q 0) * (Q 1) * (Q 2) * (Q 3) * (Q 4) * (Q 5) * (Q 6) * (Q 7) * (Q 8) * (Q 9) * (Q 10) * (Q 11) = 244140624 :=
sorry

end product_of_dodecagon_l322_32258


namespace find_unknown_number_l322_32202

theorem find_unknown_number
  (n : ℕ)
  (h_lcm : Nat.lcm n 1491 = 5964) :
  n = 4 :=
sorry

end find_unknown_number_l322_32202


namespace divisible_by_42_l322_32263

theorem divisible_by_42 (a : ℤ) : ∃ k : ℤ, a^7 - a = 42 * k := 
sorry

end divisible_by_42_l322_32263


namespace rationalize_denominator_l322_32293

theorem rationalize_denominator : (1 / (Real.sqrt 3 + 1) = (Real.sqrt 3 - 1) / 2) :=
by
  sorry

end rationalize_denominator_l322_32293


namespace min_a_plus_b_l322_32262

variable (a b : ℝ)
variable (ha_pos : a > 0)
variable (hb_pos : b > 0)
variable (h1 : a^2 - 12 * b ≥ 0)
variable (h2 : 9 * b^2 - 4 * a ≥ 0)

theorem min_a_plus_b (a b : ℝ) (ha_pos : a > 0) (hb_pos : b > 0)
  (h1 : a^2 - 12 * b ≥ 0) (h2 : 9 * b^2 - 4 * a ≥ 0) :
  a + b = 3.3442 := 
sorry

end min_a_plus_b_l322_32262


namespace expected_number_of_letters_in_mailbox_A_l322_32231

def prob_xi_0 : ℚ := 4 / 9
def prob_xi_1 : ℚ := 4 / 9
def prob_xi_2 : ℚ := 1 / 9

def expected_xi := 0 * prob_xi_0 + 1 * prob_xi_1 + 2 * prob_xi_2

theorem expected_number_of_letters_in_mailbox_A :
  expected_xi = 2 / 3 := by
  sorry

end expected_number_of_letters_in_mailbox_A_l322_32231


namespace distinct_roots_l322_32213

noncomputable def roots (a b c : ℝ) := ((b^2 - 4 * a * c) ≥ 0) ∧ ((-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a) * Real.sqrt (b^2 - 4 * a * c)) ≠ (0 : ℝ)

theorem distinct_roots{ p q r s : ℝ } (h1 : p ≠ q) (h2 : p ≠ r) (h3 : p ≠ s) (h4 : q ≠ r) 
(h5 : q ≠ s) (h6 : r ≠ s)
(h_roots_1 : roots 1 (-12*p) (-13*q))
(h_roots_2 : roots 1 (-12*r) (-13*s)) : 
(p + q + r + s = 2028) := sorry

end distinct_roots_l322_32213


namespace solve_triangle_l322_32236

noncomputable def triangle_side_lengths (a b c : ℝ) : Prop :=
  a = 10 ∧ b = 9 ∧ c = 17

theorem solve_triangle (a b c : ℝ) :
  (a ^ 2 - b ^ 2 = 19) ∧ 
  (126 + 52 / 60 + 12 / 3600 = 126.87) ∧ -- Converting the angle into degrees for simplicity
  (21.25 = 21.25)  -- Diameter given directly
  → triangle_side_lengths a b c :=
sorry

end solve_triangle_l322_32236


namespace students_absent_afternoon_l322_32234

theorem students_absent_afternoon
  (morning_registered afternoon_registered total_students morning_absent : ℕ)
  (h_morning_registered : morning_registered = 25)
  (h_morning_absent : morning_absent = 3)
  (h_afternoon_registered : afternoon_registered = 24)
  (h_total_students : total_students = 42) :
  (afternoon_registered - (total_students - (morning_registered - morning_absent))) = 4 :=
by
  sorry

end students_absent_afternoon_l322_32234


namespace basketball_team_wins_l322_32259

theorem basketball_team_wins (f : ℚ) (h1 : 40 + 40 * f + (40 + 40 * f) = 130) : f = 5 / 8 :=
by
  sorry

end basketball_team_wins_l322_32259


namespace sum_of_faces_l322_32280

variable (a d b c e f : ℕ)
variable (pos_a : a > 0) (pos_d : d > 0) (pos_b : b > 0) (pos_c : c > 0) 
variable (pos_e : e > 0) (pos_f : f > 0)
variable (h : a * b * e + a * b * f + a * c * e + a * c * f + d * b * e + d * b * f + d * c * e + d * c * f = 1176)

theorem sum_of_faces : a + d + b + c + e + f = 33 := by
  sorry

end sum_of_faces_l322_32280


namespace root_in_interval_l322_32260

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x - 1

theorem root_in_interval (k : ℤ) (h : ∃ x : ℝ, k < x ∧ x < k + 1 ∧ f x = 0) : k = 0 :=
by
  sorry

end root_in_interval_l322_32260


namespace no_natural_n_divisible_by_2019_l322_32286

theorem no_natural_n_divisible_by_2019 :
  ∀ n : ℕ, ¬ 2019 ∣ (n^2 + n + 2) :=
by sorry

end no_natural_n_divisible_by_2019_l322_32286


namespace find_f2_l322_32232

noncomputable def f : ℝ → ℝ := sorry

axiom function_property : ∀ (x : ℝ), f (2^x) + x * f (2^(-x)) = 1

theorem find_f2 : f 2 = 0 :=
by
  sorry

end find_f2_l322_32232


namespace polynomial_value_l322_32279

theorem polynomial_value (x : ℝ) (hx : x^2 - 4*x + 1 = 0) : 
  x^4 - 8*x^3 + 10*x^2 - 8*x + 1 = -56 - 32*Real.sqrt 3 ∨ 
  x^4 - 8*x^3 + 10*x^2 - 8*x + 1 = -56 + 32*Real.sqrt 3 :=
sorry

end polynomial_value_l322_32279


namespace primes_product_less_than_20_l322_32287

-- Define the primes less than 20
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the product of a list of natural numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

theorem primes_product_less_than_20 :
  product primes_less_than_20 = 9699690 :=
by
  sorry

end primes_product_less_than_20_l322_32287


namespace numWaysToChoosePairs_is_15_l322_32283

def numWaysToChoosePairs : ℕ :=
  let white := Nat.choose 5 2
  let brown := Nat.choose 3 2
  let blue := Nat.choose 2 2
  let black := Nat.choose 2 2
  white + brown + blue + black

theorem numWaysToChoosePairs_is_15 : numWaysToChoosePairs = 15 := by
  -- We will prove this theorem in actual proof
  sorry

end numWaysToChoosePairs_is_15_l322_32283


namespace pairs_sold_l322_32242

-- Define the given conditions
def initial_large_pairs : ℕ := 22
def initial_medium_pairs : ℕ := 50
def initial_small_pairs : ℕ := 24
def pairs_left : ℕ := 13

-- Translate to the equivalent proof problem
theorem pairs_sold : (initial_large_pairs + initial_medium_pairs + initial_small_pairs) - pairs_left = 83 := by
  sorry

end pairs_sold_l322_32242


namespace sculpture_and_base_total_height_l322_32248

noncomputable def sculpture_height_ft : Nat := 2
noncomputable def sculpture_height_in : Nat := 10
noncomputable def base_height_in : Nat := 4
noncomputable def inches_per_foot : Nat := 12

theorem sculpture_and_base_total_height :
  (sculpture_height_ft * inches_per_foot + sculpture_height_in + base_height_in = 38) :=
by
  sorry

end sculpture_and_base_total_height_l322_32248


namespace julia_money_left_l322_32225

def initial_money : ℕ := 40
def spent_on_game : ℕ := initial_money / 2
def money_left_after_game : ℕ := initial_money - spent_on_game
def spent_on_in_game_purchases : ℕ := money_left_after_game / 4
def final_money_left : ℕ := money_left_after_game - spent_on_in_game_purchases

theorem julia_money_left : final_money_left = 15 := by
  sorry

end julia_money_left_l322_32225


namespace simplify_expression_l322_32243

theorem simplify_expression : (-5) - (-4) + (-7) - (2) = -5 + 4 - 7 - 2 := 
by
  sorry

end simplify_expression_l322_32243


namespace find_x_l322_32219

theorem find_x (A B D : ℝ) (BC CD x : ℝ) 
  (hA : A = 60) (hB : B = 90) (hD : D = 90) 
  (hBC : BC = 2) (hCD : CD = 3) 
  (hResult : x = 8 / Real.sqrt 3) : 
  AB = x :=
by
  sorry

end find_x_l322_32219


namespace fractional_eq_solution_range_l322_32264

theorem fractional_eq_solution_range (x m : ℝ) (h : (2 * x - m) / (x + 1) = 1) (hx : x < 0) : 
  m < -1 ∧ m ≠ -2 := 
by 
  sorry

end fractional_eq_solution_range_l322_32264


namespace reading_speed_increase_factor_l322_32285

-- Define Tom's normal reading speed as a constant rate
def tom_normal_speed := 12 -- pages per hour

-- Define the time period
def hours := 2 -- hours

-- Define the number of pages read in the given time period
def pages_read := 72 -- pages

-- Calculate the expected pages read at normal speed in the given time
def expected_pages := tom_normal_speed * hours -- should be 24 pages

-- Define the calculated factor by which the reading speed has increased
def expected_factor := pages_read / expected_pages -- should be 3

-- Prove that the factor is indeed 3
theorem reading_speed_increase_factor :
  expected_factor = 3 := by
  sorry

end reading_speed_increase_factor_l322_32285


namespace arccos_neg1_l322_32228

theorem arccos_neg1 : Real.arccos (-1) = Real.pi := 
sorry

end arccos_neg1_l322_32228


namespace sum_of_xy_eq_20_l322_32299

theorem sum_of_xy_eq_20 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hx_lt : x < 30) (hy_lt : y < 30)
    (hxy : x + y + x * y = 119) : x + y = 20 :=
sorry

end sum_of_xy_eq_20_l322_32299


namespace liam_drinks_17_glasses_l322_32204

def minutes_in_hours (h : ℕ) : ℕ := h * 60

def total_time_in_minutes (hours : ℕ) (extra_minutes : ℕ) : ℕ := 
  minutes_in_hours hours + extra_minutes

def rate_of_drinking (drink_interval : ℕ) (total_minutes : ℕ) : ℕ :=
  total_minutes / drink_interval

theorem liam_drinks_17_glasses : 
  rate_of_drinking 20 (total_time_in_minutes 5 40) = 17 :=
by
  sorry

end liam_drinks_17_glasses_l322_32204


namespace general_term_formula_l322_32284

-- Define the sequence as given in the conditions
def seq (n : ℕ) : ℚ := 
  match n with 
  | 0       => 1
  | 1       => 2 / 3
  | 2       => 1 / 2
  | 3       => 2 / 5
  | (n + 1) => sorry   -- This is just a placeholder, to be proved

-- State the theorem
theorem general_term_formula (n : ℕ) : seq n = 2 / (n + 1) := 
by {
  -- Proof will be provided here
  sorry
}

end general_term_formula_l322_32284


namespace margin_in_terms_of_selling_price_l322_32253

variable (C S n M : ℝ)

theorem margin_in_terms_of_selling_price (h : M = (2 * C) / n) : M = (2 * S) / (n + 2) :=
sorry

end margin_in_terms_of_selling_price_l322_32253


namespace fred_cantaloupes_l322_32216

def num_cantaloupes_K : ℕ := 29
def num_cantaloupes_J : ℕ := 20
def total_cantaloupes : ℕ := 65

theorem fred_cantaloupes : ∃ F : ℕ, num_cantaloupes_K + num_cantaloupes_J + F = total_cantaloupes ∧ F = 16 :=
by
  sorry

end fred_cantaloupes_l322_32216


namespace median_perimeter_ratio_l322_32207

variables {A B C : Type*}
variables (AB BC AC AD BE CF : ℝ)
variable (l m : ℝ)

noncomputable def triangle_perimeter (AB BC AC : ℝ) : ℝ := AB + BC + AC
noncomputable def triangle_median_sum (AD BE CF : ℝ) : ℝ := AD + BE + CF

theorem median_perimeter_ratio (h1 : l = triangle_perimeter AB BC AC)
                                (h2 : m = triangle_median_sum AD BE CF) :
  m / l > 3 / 4 :=
by
  sorry

end median_perimeter_ratio_l322_32207


namespace youtube_dislikes_calculation_l322_32245

theorem youtube_dislikes_calculation :
  ∀ (l d_initial d_final : ℕ),
    l = 3000 →
    d_initial = (l / 2) + 100 →
    d_final = d_initial + 1000 →
    d_final = 2600 :=
by
  intros l d_initial d_final h_l h_d_initial h_d_final
  sorry

end youtube_dislikes_calculation_l322_32245


namespace remainder_of_875_div_by_170_l322_32241

theorem remainder_of_875_div_by_170 :
  ∃ r, (∀ x, x ∣ 680 ∧ x ∣ (875 - r) → x ≤ 170) ∧ 170 ∣ (875 - r) ∧ r = 25 :=
by
  sorry

end remainder_of_875_div_by_170_l322_32241


namespace evaluate_expression_l322_32266

theorem evaluate_expression : (∃ (x : Real), 6 < x ∧ x < 7 ∧ x = Real.sqrt 45) → (Int.floor (Real.sqrt 45))^2 + 2*Int.floor (Real.sqrt 45) + 1 = 49 := 
by
  sorry

end evaluate_expression_l322_32266


namespace total_ingredients_l322_32206

theorem total_ingredients (b f s : ℕ) (h_ratio : 2 * f = 5 * f) (h_flour : f = 15) : b + f + s = 30 :=
by 
  sorry

end total_ingredients_l322_32206


namespace business_transaction_loss_l322_32288

theorem business_transaction_loss (cost_price : ℝ) (final_price : ℝ) (markup_percent : ℝ) (reduction_percent : ℝ) : 
  (final_price = 96) ∧ (markup_percent = 0.2) ∧ (reduction_percent = 0.2) ∧ (cost_price * (1 + markup_percent) * (1 - reduction_percent) = final_price) → 
  (cost_price - final_price = -4) :=
by
sorry

end business_transaction_loss_l322_32288


namespace arcsin_cos_solution_l322_32298

theorem arcsin_cos_solution (x : ℝ) (h : -π/2 ≤ x/3 ∧ x/3 ≤ π/2) :
  x = 3*π/10 ∨ x = 3*π/8 := 
sorry

end arcsin_cos_solution_l322_32298


namespace combinations_of_three_toppings_l322_32282

def number_of_combinations (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem combinations_of_three_toppings : number_of_combinations 10 3 = 120 := by
  sorry

end combinations_of_three_toppings_l322_32282


namespace positive_difference_1010_1000_l322_32265

-- Define the arithmetic sequence
def arithmetic_sequence (a d n : ℕ) : ℕ :=
  a + (n - 1) * d

-- Define the specific terms
def a_1000 := arithmetic_sequence 5 7 1000
def a_1010 := arithmetic_sequence 5 7 1010

-- Proof statement
theorem positive_difference_1010_1000 : a_1010 - a_1000 = 70 :=
by
  sorry

end positive_difference_1010_1000_l322_32265


namespace gcd_of_repeated_three_digit_numbers_l322_32254

theorem gcd_of_repeated_three_digit_numbers :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → Int.gcd 1001001 n = 1001001 :=
by
  -- proof omitted
  sorry

end gcd_of_repeated_three_digit_numbers_l322_32254


namespace toby_breakfast_calories_l322_32249

noncomputable def calories_bread := 100
noncomputable def calories_peanut_butter_per_serving := 200
noncomputable def servings_peanut_butter := 2

theorem toby_breakfast_calories :
  1 * calories_bread + servings_peanut_butter * calories_peanut_butter_per_serving = 500 :=
by
  sorry

end toby_breakfast_calories_l322_32249


namespace dads_strawberries_l322_32212

variable (M D : ℕ)

theorem dads_strawberries (h1 : M + D = 22) (h2 : M = 36) (h3 : D ≤ 22) :
  D + 30 = 46 :=
by
  sorry

end dads_strawberries_l322_32212


namespace total_jellybeans_l322_32226

-- Define the conditions
def a := 3 * 12       -- Caleb's jellybeans
def b := a / 2        -- Sophie's jellybeans

-- Define the goal
def total := a + b    -- Total jellybeans

-- The theorem statement
theorem total_jellybeans : total = 54 :=
by
  -- Proof placeholder
  sorry

end total_jellybeans_l322_32226


namespace opposite_of_neg_2_is_2_l322_32233

theorem opposite_of_neg_2_is_2 : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_2_is_2_l322_32233


namespace min_value_func_l322_32237

noncomputable def func (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

theorem min_value_func : ∃ x : ℝ, func x = -2 :=
by
  existsi (Real.pi / 2 + Real.pi / 3)
  sorry

end min_value_func_l322_32237


namespace asian_games_tourists_scientific_notation_l322_32252

theorem asian_games_tourists_scientific_notation : 
  ∀ (n : ℕ), n = 18480000 → 1.848 * (10:ℝ) ^ 7 = (n : ℝ) :=
by
  intro n
  sorry

end asian_games_tourists_scientific_notation_l322_32252


namespace find_a_l322_32240

theorem find_a (x : ℝ) (hx1 : 0 < x)
  (hx2 : x + 1/x ≥ 2)
  (hx3 : x + 4/x^2 ≥ 3)
  (hx4 : x + 27/x^3 ≥ 4) :
  (x + a/x^4 ≥ 5) → a = 4^4 :=
sorry

end find_a_l322_32240


namespace maximize_profit_l322_32278

noncomputable def R (x : ℝ) : ℝ := 
  if x ≤ 40 then
    40 * x - (1 / 2) * x^2
  else
    1500 - 25000 / x

noncomputable def cost (x : ℝ) : ℝ := 2 + 0.1 * x

noncomputable def f (x : ℝ) : ℝ := R x - cost x

theorem maximize_profit :
  ∃ x : ℝ, x = 50 ∧ f 50 = 300 := by
  sorry

end maximize_profit_l322_32278


namespace rhombus_shorter_diagonal_l322_32276

theorem rhombus_shorter_diagonal (perimeter : ℝ) (angle_ratio : ℝ) (side_length diagonal_length : ℝ)
  (h₁ : perimeter = 9.6) 
  (h₂ : angle_ratio = 1 / 2) 
  (h₃ : side_length = perimeter / 4) 
  (h₄ : diagonal_length = side_length) :
  diagonal_length = 2.4 := 
sorry

end rhombus_shorter_diagonal_l322_32276


namespace rectangle_width_l322_32210

theorem rectangle_width (side_length square_len rect_len : ℝ) (h1 : side_length = 4) (h2 : rect_len = 4) (h3 : square_len = side_length * side_length) (h4 : square_len = rect_len * some_width) :
  some_width = 4 :=
by
  sorry

end rectangle_width_l322_32210


namespace complex_exponentiation_l322_32220

-- Define the imaginary unit i where i^2 = -1.
def i : ℂ := Complex.I

-- Lean statement for proving the problem.
theorem complex_exponentiation :
  (1 + i)^6 = -8 * i :=
sorry

end complex_exponentiation_l322_32220


namespace max_view_angle_dist_l322_32291

theorem max_view_angle_dist (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) : ∃ (x : ℝ), x = Real.sqrt (b * (a + b)) := by
  sorry

end max_view_angle_dist_l322_32291


namespace total_fish_count_l322_32203

def number_of_tables : ℕ := 32
def fish_per_table : ℕ := 2
def additional_fish_table : ℕ := 1
def total_fish : ℕ := (number_of_tables * fish_per_table) + additional_fish_table

theorem total_fish_count : total_fish = 65 := by
  sorry

end total_fish_count_l322_32203


namespace plane_equation_correct_l322_32268

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vectorBC (B C : Point3D) : Point3D :=
  { x := C.x - B.x, y := C.y - B.y, z := C.z - B.z }

def planeEquation (n : Point3D) (A : Point3D) (P : Point3D) : ℝ :=
  n.x * (P.x - A.x) + n.y * (P.y - A.y) + n.z * (P.z - A.z)

theorem plane_equation_correct :
  let A := ⟨3, -3, -6⟩
  let B := ⟨1, 9, -5⟩
  let C := ⟨6, 6, -4⟩
  let n := vectorBC B C
  ∀ P, planeEquation n A P = 0 ↔ 5 * (P.x - A.x) - 3 * (P.y - A.y) + 1 * (P.z - A.z) - 18 = 0 :=
by
  sorry

end plane_equation_correct_l322_32268


namespace winner_more_votes_than_second_place_l322_32290

theorem winner_more_votes_than_second_place :
  ∃ (W S T F : ℕ), 
    F = 199 ∧
    W = S + (W - S) ∧
    W = T + 79 ∧
    W = F + 105 ∧
    W + S + T + F = 979 ∧
    W - S = 53 :=
by
  sorry

end winner_more_votes_than_second_place_l322_32290


namespace little_john_gave_to_each_friend_l322_32257

noncomputable def little_john_total : ℝ := 10.50
noncomputable def sweets : ℝ := 2.25
noncomputable def remaining : ℝ := 3.85

theorem little_john_gave_to_each_friend :
  (little_john_total - sweets - remaining) / 2 = 2.20 :=
by
  sorry

end little_john_gave_to_each_friend_l322_32257


namespace find_m_l322_32251

-- Definitions
variable {A B C O H : Type}
variable {O_is_circumcenter : is_circumcenter O A B C}
variable {H_is_altitude_intersection : is_altitude_intersection H A B C}
variable (AH BH CH OA OB OC : ℝ)

-- Problem Statement
theorem find_m (h : AH * BH * CH = m * (OA * OB * OC)) : m = 1 :=
sorry

end find_m_l322_32251


namespace domain_f1_correct_f2_correct_f2_at_3_l322_32261

noncomputable def f1 (x : ℝ) : ℝ := Real.sqrt (4 - 2 * x) + 1 + 1 / (x + 1)

noncomputable def domain_f1 : Set ℝ := {x | 4 - 2 * x ≥ 0} \ (insert 1 (insert (-1) {}))

theorem domain_f1_correct : domain_f1 = { x | x ≤ 2 ∧ x ≠ 1 ∧ x ≠ -1 } :=
by
  sorry

noncomputable def f2 (x : ℝ) : ℝ := x^2 - 4 * x + 3

theorem f2_correct : ∀ x, f2 (x + 1) = x^2 - 2 * x :=
by
  sorry

theorem f2_at_3 : f2 3 = 0 :=
by
  sorry

end domain_f1_correct_f2_correct_f2_at_3_l322_32261
