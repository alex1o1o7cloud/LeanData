import Mathlib

namespace distinct_rectangles_l133_13304

theorem distinct_rectangles :
  ∃! (l w : ℝ), l * w = 100 ∧ l + w = 24 :=
sorry

end distinct_rectangles_l133_13304


namespace find_cool_triple_x_eq_5_find_cool_triple_x_eq_7_two_distinct_cool_triples_for_odd_x_find_cool_triple_x_even_l133_13329

-- Define the nature of a "cool" triple.
def is_cool_triple (x y z : ℕ) : Prop :=
  x > 0 ∧ y > 1 ∧ z > 0 ∧ x^2 - 3 * y^2 = z^2 - 3

-- Part (a) i: For x = 5.
theorem find_cool_triple_x_eq_5 : ∃ (y z : ℕ), is_cool_triple 5 y z := sorry

-- Part (a) ii: For x = 7.
theorem find_cool_triple_x_eq_7 : ∃ (y z : ℕ), is_cool_triple 7 y z := sorry

-- Part (b): For every x ≥ 5 and odd, there are at least two distinct cool triples.
theorem two_distinct_cool_triples_for_odd_x (x : ℕ) (h1 : x ≥ 5) (h2 : x % 2 = 1) : 
  ∃ (y₁ z₁ y₂ z₂ : ℕ), is_cool_triple x y₁ z₁ ∧ is_cool_triple x y₂ z₂ ∧ (y₁, z₁) ≠ (y₂, z₂) := sorry

-- Part (c): Find a cool type triple with x even.
theorem find_cool_triple_x_even : ∃ (x y z : ℕ), x % 2 = 0 ∧ is_cool_triple x y z := sorry

end find_cool_triple_x_eq_5_find_cool_triple_x_eq_7_two_distinct_cool_triples_for_odd_x_find_cool_triple_x_even_l133_13329


namespace roots_of_polynomial_l133_13338

def poly (x : ℝ) : ℝ := x^3 - 3 * x^2 - 4 * x + 12

theorem roots_of_polynomial : 
  (poly 2 = 0) ∧ (poly (-2) = 0) ∧ (poly 3 = 0) ∧ 
  (∀ x, poly x = 0 → x = 2 ∨ x = -2 ∨ x = 3) :=
by
  sorry

end roots_of_polynomial_l133_13338


namespace total_triangles_l133_13328

theorem total_triangles (small_triangles : ℕ)
    (triangles_4_small : ℕ)
    (triangles_9_small : ℕ)
    (triangles_16_small : ℕ)
    (number_small_triangles : small_triangles = 20)
    (number_triangles_4_small : triangles_4_small = 5)
    (number_triangles_9_small : triangles_9_small = 1)
    (number_triangles_16_small : triangles_16_small = 1) :
    small_triangles + triangles_4_small + triangles_9_small + triangles_16_small = 27 := 
by 
    -- proof omitted
    sorry

end total_triangles_l133_13328


namespace baker_cakes_total_l133_13342

-- Conditions
def initial_cakes : ℕ := 121
def cakes_sold : ℕ := 105
def cakes_bought : ℕ := 170

-- Proof Problem
theorem baker_cakes_total :
  initial_cakes - cakes_sold + cakes_bought = 186 :=
by
  sorry

end baker_cakes_total_l133_13342


namespace blue_tshirt_count_per_pack_l133_13381

theorem blue_tshirt_count_per_pack :
  ∀ (total_tshirts white_packs blue_packs tshirts_per_white_pack tshirts_per_blue_pack : ℕ), 
    white_packs = 3 →
    blue_packs = 2 → 
    tshirts_per_white_pack = 6 → 
    total_tshirts = 26 →
    total_tshirts = white_packs * tshirts_per_white_pack + blue_packs * tshirts_per_blue_pack →
  tshirts_per_blue_pack = 4 :=
by
  intros total_tshirts white_packs blue_packs tshirts_per_white_pack tshirts_per_blue_pack
  intros h1 h2 h3 h4 h5
  sorry

end blue_tshirt_count_per_pack_l133_13381


namespace gcd_sequence_condition_l133_13324

theorem gcd_sequence_condition (p q : ℕ) (hp : 0 < p) (hq : 0 < q)
  (a : ℕ → ℕ)
  (ha1 : a 1 = 1) (ha2 : a 2 = 1) 
  (ha_rec : ∀ n, a (n + 2) = p * a (n + 1) + q * a n) 
  (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (gcd (a m) (a n) = a (gcd m n)) ↔ (p = 1) := 
sorry

end gcd_sequence_condition_l133_13324


namespace intersection_eq_l133_13390

-- Define Set A based on the given condition
def setA : Set ℝ := {x | 1 < (3:ℝ)^x ∧ (3:ℝ)^x ≤ 9}

-- Define Set B based on the given condition
def setB : Set ℝ := {x | (x + 2) / (x - 1) ≤ 0}

-- Define the intersection of Set A and Set B
def intersection : Set ℝ := {x | x > 0 ∧ x < 1}

-- Prove that the intersection of setA and setB equals (0, 1)
theorem intersection_eq : {x | x > 0 ∧ x < 1} = {x | x ∈ setA ∧ x ∈ setB} :=
by
  sorry

end intersection_eq_l133_13390


namespace mod_product_prob_l133_13352

def prob_mod_product (a b : ℕ) : ℚ :=
  let quotient := a * b % 4
  if quotient = 0 then 1/2
  else if quotient = 1 then 1/8
  else if quotient = 2 then 1/4
  else if quotient = 3 then 1/8
  else 0

theorem mod_product_prob (a b : ℕ) :
  (∃ n : ℚ, n = prob_mod_product a b) :=
by
  sorry

end mod_product_prob_l133_13352


namespace good_numbers_100_2010_ex_good_and_not_good_x_y_l133_13365

-- Definition of a good number
def is_good_number (n : ℤ) : Prop := ∃ a b : ℤ, n = a^2 + 161 * b^2

-- (1) Prove 100 and 2010 are good numbers
theorem good_numbers_100_2010 : is_good_number 100 ∧ is_good_number 2010 :=
by sorry

-- (2) Prove there exist positive integers x and y such that x^161 + y^161 is a good number, 
-- but x + y is not a good number
theorem ex_good_and_not_good_x_y : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ is_good_number (x^161 + y^161) ∧ ¬ is_good_number (x + y) :=
by sorry

end good_numbers_100_2010_ex_good_and_not_good_x_y_l133_13365


namespace is_opposite_if_differ_in_sign_l133_13357

-- Define opposite numbers based on the given condition in the problem:
def opposite_numbers (a b : ℝ) : Prop := a = -b

-- State the theorem based on the translation in c)
theorem is_opposite_if_differ_in_sign (a b : ℝ) (h : a = -b) : opposite_numbers a b := by
  sorry

end is_opposite_if_differ_in_sign_l133_13357


namespace median_isosceles_right_triangle_leg_length_l133_13344

theorem median_isosceles_right_triangle_leg_length (m : ℝ) (h : ℝ) (x : ℝ)
  (H1 : m = 15)
  (H2 : m = h / 2)
  (H3 : 2 * x * x = h * h) : x = 15 * Real.sqrt 2 :=
by
  sorry

end median_isosceles_right_triangle_leg_length_l133_13344


namespace find_valid_pairs_l133_13323

def satisfies_condition (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ (a ^ 2017 + b) % (a * b) = 0

theorem find_valid_pairs : 
  ∀ (a b : ℕ), satisfies_condition a b → (a = 1 ∧ b = 1) ∨ (a = 2 ∧ b = 2 ^ 2017) := 
by
  sorry

end find_valid_pairs_l133_13323


namespace system_of_inequalities_solutions_l133_13309

theorem system_of_inequalities_solutions (x : ℤ) :
  (3 * x - 2 ≥ 2 * x - 5) ∧ ((x / 2 - (x - 2) / 3 < 1 / 2)) →
  (x = -3 ∨ x = -2) :=
by sorry

end system_of_inequalities_solutions_l133_13309


namespace base12_addition_l133_13366

theorem base12_addition : ∀ a b : ℕ, a = 956 ∧ b = 273 → (a + b) = 1009 := by
  sorry

end base12_addition_l133_13366


namespace non_adjacent_boys_arrangements_l133_13336

-- We define the number of boys and girls
def boys := 4
def girls := 6

-- The function to compute combinations C(n, k)
def combinations (n k : ℕ) : ℕ := Nat.choose n k

-- The function to compute permutations P(n, k)
def permutations (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

-- The total arrangements where 2 selected boys are not adjacent
def total_non_adjacent_arrangements : ℕ :=
  (combinations boys 2) * (combinations girls 3) * (permutations 3 3) * (permutations (3 + 1) 2)

theorem non_adjacent_boys_arrangements :
  total_non_adjacent_arrangements = 8640 := by
  sorry

end non_adjacent_boys_arrangements_l133_13336


namespace select_at_least_8_sticks_l133_13395

theorem select_at_least_8_sticks (S : Finset ℕ) (hS : S = (Finset.range 92 \ {0})) :
  ∃ (sticks : Finset ℕ) (h_sticks : sticks.card = 8),
    ∃ (a b c : ℕ) (h_a : a ∈ sticks) (h_b : b ∈ sticks) (h_c : c ∈ sticks),
    (a + b > c) ∧ (b + c > a) ∧ (c + a > b) :=
by
  -- Proof required here
  sorry

end select_at_least_8_sticks_l133_13395


namespace total_weight_l133_13325

axiom D : ℕ -- Daughter's weight
axiom C : ℕ -- Grandchild's weight
axiom M : ℕ -- Mother's weight

-- Given conditions from the problem
axiom h1 : D + C = 60
axiom h2 : C = M / 5
axiom h3 : D = 50

-- The statement to be proven
theorem total_weight : M + D + C = 110 :=
by sorry

end total_weight_l133_13325


namespace inequality_square_l133_13310

theorem inequality_square (a b : ℝ) (h : a > b ∧ b > 0) : a^2 > b^2 :=
by
  sorry

end inequality_square_l133_13310


namespace pages_per_day_l133_13302

def notebooks : Nat := 5
def pages_per_notebook : Nat := 40
def total_days : Nat := 50

theorem pages_per_day (H1 : notebooks = 5) (H2 : pages_per_notebook = 40) (H3 : total_days = 50) : 
  (notebooks * pages_per_notebook / total_days) = 4 := by
  sorry

end pages_per_day_l133_13302


namespace values_of_m_l133_13387

def A : Set ℝ := { -1, 2 }
def B (m : ℝ) : Set ℝ := { x | m * x + 1 = 0 }

theorem values_of_m (m : ℝ) : (A ∪ B m = A) ↔ (m = -1/2 ∨ m = 0 ∨ m = 1) := by
  sorry

end values_of_m_l133_13387


namespace total_rain_duration_l133_13358

theorem total_rain_duration:
  let first_day_duration := 10
  let second_day_duration := first_day_duration + 2
  let third_day_duration := 2 * second_day_duration
  first_day_duration + second_day_duration + third_day_duration = 46 :=
by
  sorry

end total_rain_duration_l133_13358


namespace us_supermarkets_count_l133_13373

-- Definition of variables and conditions
def total_supermarkets : ℕ := 84
def difference_us_canada : ℕ := 10

-- Proof statement
theorem us_supermarkets_count (C : ℕ) (H : 2 * C + difference_us_canada = total_supermarkets) :
  C + difference_us_canada = 47 :=
sorry

end us_supermarkets_count_l133_13373


namespace minimize_fractions_sum_l133_13351

theorem minimize_fractions_sum {A B C D E : ℕ}
  (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D) (h4 : A ≠ E)
  (h5 : B ≠ C) (h6 : B ≠ D) (h7 : B ≠ E)
  (h8 : C ≠ D) (h9 : C ≠ E) (h10 : D ≠ E)
  (h11 : A ≠ 9) (h12 : B ≠ 9) (h13 : C ≠ 9) (h14 : D ≠ 9) (h15 : E ≠ 9)
  (hA : 1 ≤ A) (hB : 1 ≤ B) (hC : 1 ≤ C) (hD : 1 ≤ D) (hE : 1 ≤ E)
  (hA' : A ≤ 9) (hB' : B ≤ 9) (hC' : C ≤ 9) (hD' : D ≤ 9) (hE' : E ≤ 9) :
  A / B + C / D + E / 9 = 125 / 168 :=
sorry

end minimize_fractions_sum_l133_13351


namespace abs_x_minus_y_zero_l133_13347

theorem abs_x_minus_y_zero (x y : ℝ) 
  (h_avg : (x + y + 30 + 29 + 31) / 5 = 30)
  (h_var : ((x - 30)^2 + (y - 30)^2 + (30 - 30)^2 + (29 - 30)^2 + (31 - 30)^2) / 5 = 2) : 
  |x - y| = 0 :=
  sorry

end abs_x_minus_y_zero_l133_13347


namespace driver_actual_speed_l133_13364

theorem driver_actual_speed (v t : ℝ) 
  (h1 : t > 0) 
  (h2 : v > 0) 
  (cond : v * t = (v + 18) * (2 / 3 * t)) : 
  v = 36 :=
by 
  sorry

end driver_actual_speed_l133_13364


namespace max_value_y_l133_13354

theorem max_value_y (x : ℝ) (h : x < -1) : x + 1/(x + 1) ≤ -3 :=
by sorry

end max_value_y_l133_13354


namespace six_digit_palindrome_count_l133_13367

def num_six_digit_palindromes : Nat :=
  let a_choices := 9
  let b_choices := 10
  let c_choices := 10
  let d_choices := 10
  a_choices * b_choices * c_choices * d_choices

theorem six_digit_palindrome_count : num_six_digit_palindromes = 9000 := by
  sorry

end six_digit_palindrome_count_l133_13367


namespace martin_speed_l133_13362

theorem martin_speed (distance : ℝ) (time : ℝ) (h₁ : distance = 12) (h₂ : time = 6) : (distance / time = 2) :=
by 
  -- Note: The proof is not required as per instructions, so we use 'sorry'
  sorry

end martin_speed_l133_13362


namespace spadesuit_evaluation_l133_13327

def spadesuit (a b : ℤ) : ℤ := Int.natAbs (a - b)

theorem spadesuit_evaluation :
  spadesuit 5 (spadesuit 3 9) = 1 := 
by 
  sorry

end spadesuit_evaluation_l133_13327


namespace price_restoration_percentage_l133_13339

noncomputable def original_price := 100
def reduced_price (P : ℝ) := 0.8 * P
def restored_price (P : ℝ) (x : ℝ) := P = x * reduced_price P

theorem price_restoration_percentage (P : ℝ) (x : ℝ) (h : restored_price P x) : x = 1.25 :=
by
  sorry

end price_restoration_percentage_l133_13339


namespace neg_p_l133_13369

variable {x : ℝ}

def p := ∀ x > 0, Real.sin x ≤ 1

theorem neg_p : ¬ p ↔ ∃ x > 0, Real.sin x > 1 :=
by
  sorry

end neg_p_l133_13369


namespace f_decreasing_increasing_find_b_range_l133_13396

-- Define the function f(x) and prove its properties for x > 0 and x < 0
noncomputable def f (x a : ℝ) : ℝ := x + a / x

theorem f_decreasing_increasing (a : ℝ) (h : a > 0):
  (∀ x : ℝ, 0 < x → x ≤ Real.sqrt a → ∀ x1 x2 : ℝ, (0 < x1 ∧ x1 < x2 ∧ x2 ≤ Real.sqrt a) → f x1 a > f x2 a) ∧ 
  (∀ x : ℝ, 0 < Real.sqrt a → Real.sqrt a ≤ x → ∀ x1 x2 : ℝ, (Real.sqrt a ≤ x1 ∧ x1 < x2) → f x1 a < f x2 a) ∧ 
  (∀ x : ℝ, x < 0 → -Real.sqrt a ≤ x ∧ x < 0 → f x1 a > f x2 a) ∧ 
  (∀ x : ℝ, x < 0 → x < -Real.sqrt a → f x1 a < f x2 a)
:= sorry

-- Define the function h(x) and find the range of b
noncomputable def h (x : ℝ) : ℝ := x + 4 / x - 8
noncomputable def g (x b : ℝ) : ℝ := -x - 2 * b

theorem find_b_range:
  (∀ x1 : ℝ, 1 ≤ x1 ∧ x1 ≤ 3 → ∃ x2 : ℝ, 1 ≤ x2 ∧ x2 ≤ 3 ∧ g x2 b = h x1) ↔
  1/2 ≤ b ∧ b ≤ 1
:= sorry

end f_decreasing_increasing_find_b_range_l133_13396


namespace isosceles_triangle_area_l133_13300

-- Definitions
def isosceles_triangle (b h : ℝ) : Prop :=
∃ a : ℝ, a * b / 2 = a * h

def square_of_area_one (a : ℝ) : Prop :=
a = 1

def centroids_coincide (g_triangle g_square : ℝ × ℝ) : Prop :=
g_triangle = g_square

-- The statement of the problem
theorem isosceles_triangle_area
  (b h : ℝ)
  (s : ℝ)
  (triangle_centroid : ℝ × ℝ)
  (square_centroid : ℝ × ℝ)
  (H1 : isosceles_triangle b h)
  (H2 : square_of_area_one s)
  (H3 : centroids_coincide triangle_centroid square_centroid)
  : b * h / 2 = 9 / 4 :=
by
  sorry

end isosceles_triangle_area_l133_13300


namespace product_of_five_consecutive_integers_divisible_by_120_l133_13307

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) : 
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_integers_divisible_by_120_l133_13307


namespace cricket_player_innings_l133_13333

theorem cricket_player_innings (n : ℕ) (T : ℕ) 
  (h1 : T = n * 48) 
  (h2 : T + 178 = (n + 1) * 58) : 
  n = 12 :=
by
  sorry

end cricket_player_innings_l133_13333


namespace largest_fraction_among_list_l133_13353

theorem largest_fraction_among_list :
  ∃ (f : ℚ), f = 105 / 209 ∧ 
  (f > 5 / 11) ∧ 
  (f > 9 / 20) ∧ 
  (f > 23 / 47) ∧ 
  (f > 205 / 409) := 
by
  sorry

end largest_fraction_among_list_l133_13353


namespace bad_carrots_l133_13349

theorem bad_carrots (carol_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) (total_carrots : ℕ) (bad_carrots : ℕ) 
  (h1 : carol_carrots = 29)
  (h2 : mom_carrots = 16)
  (h3 : good_carrots = 38)
  (h4 : total_carrots = carol_carrots + mom_carrots)
  (h5 : bad_carrots = total_carrots - good_carrots) :
  bad_carrots = 7 := by
  sorry

end bad_carrots_l133_13349


namespace gcd_of_items_l133_13332

theorem gcd_of_items :
  ∀ (plates spoons glasses bowls : ℕ),
  plates = 3219 →
  spoons = 5641 →
  glasses = 1509 →
  bowls = 2387 →
  Nat.gcd (Nat.gcd (Nat.gcd plates spoons) glasses) bowls = 1 :=
by
  intros plates spoons glasses bowls
  intros Hplates Hspoons Hglasses Hbowls
  rw [Hplates, Hspoons, Hglasses, Hbowls]
  sorry

end gcd_of_items_l133_13332


namespace no_two_digit_factorization_1729_l133_13335

noncomputable def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem no_two_digit_factorization_1729 :
  ¬ ∃ (a b : ℕ), a * b = 1729 ∧ is_two_digit a ∧ is_two_digit b :=
by
  sorry

end no_two_digit_factorization_1729_l133_13335


namespace problem_1_problem_2_l133_13360

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log (x + 1) / Real.log 2 else 2^(-x) - 1

theorem problem_1 : f (f (-2)) = 2 := by 
  sorry

theorem problem_2 (x_0 : ℝ) (h : f x_0 < 3) : -2 < x_0 ∧ x_0 < 7 := by
  sorry

end problem_1_problem_2_l133_13360


namespace tangent_sum_problem_l133_13372

theorem tangent_sum_problem
  (α β : ℝ)
  (h_eq_root : ∃ (x y : ℝ), (x = Real.tan α) ∧ (y = Real.tan β) ∧ (6*x^2 - 5*x + 1 = 0) ∧ (6*y^2 - 5*y + 1 = 0))
  (h_range_α : 0 < α ∧ α < π/2)
  (h_range_β : π < β ∧ β < 3*π/2) :
  (Real.tan (α + β) = 1) ∧ (α + β = 5*π/4) := 
sorry

end tangent_sum_problem_l133_13372


namespace percentage_discount_l133_13370

def cost_per_ball : ℝ := 0.1
def number_of_balls : ℕ := 10000
def amount_paid : ℝ := 700

theorem percentage_discount : (number_of_balls * cost_per_ball - amount_paid) / (number_of_balls * cost_per_ball) * 100 = 30 :=
by
  sorry

end percentage_discount_l133_13370


namespace combined_rate_l133_13394

theorem combined_rate
  (earl_rate : ℕ)
  (ellen_time : ℚ)
  (total_envelopes : ℕ)
  (total_time : ℕ)
  (combined_total_envelopes : ℕ)
  (combined_total_time : ℕ) :
  earl_rate = 36 →
  ellen_time = 1.5 →
  total_envelopes = 36 →
  total_time = 1 →
  combined_total_envelopes = 180 →
  combined_total_time = 3 →
  (earl_rate + (total_envelopes / ellen_time)) = 60 :=
by
  sorry

end combined_rate_l133_13394


namespace maximize_profit_l133_13363

noncomputable def profit (t : ℝ) : ℝ :=
  27 - (18 / t) - t

theorem maximize_profit : ∀ t > 0, profit t ≤ 27 - 6 * Real.sqrt 2 ∧ profit (3 * Real.sqrt 2) = 27 - 6 * Real.sqrt 2 := by {
  sorry
}

end maximize_profit_l133_13363


namespace monthly_fee_for_second_plan_l133_13318

theorem monthly_fee_for_second_plan 
  (monthly_fee_first_plan : ℝ) 
  (rate_first_plan : ℝ) 
  (rate_second_plan : ℝ) 
  (minutes : ℕ) 
  (monthly_fee_second_plan : ℝ) :
  monthly_fee_first_plan = 22 -> 
  rate_first_plan = 0.13 -> 
  rate_second_plan = 0.18 -> 
  minutes = 280 -> 
  (22 + 0.13 * 280 = monthly_fee_second_plan + 0.18 * 280) -> 
  monthly_fee_second_plan = 8 := 
by
  intros h_fee_first_plan h_rate_first_plan h_rate_second_plan h_minutes h_equal_costs
  sorry

end monthly_fee_for_second_plan_l133_13318


namespace y_is_multiple_of_3_y_is_multiple_of_9_y_is_multiple_of_27_y_is_multiple_of_81_l133_13315

noncomputable def y : ℕ := 81 + 243 + 729 + 1458 + 2187 + 6561 + 19683

theorem y_is_multiple_of_3 : y % 3 = 0 :=
sorry

theorem y_is_multiple_of_9 : y % 9 = 0 :=
sorry

theorem y_is_multiple_of_27 : y % 27 = 0 :=
sorry

theorem y_is_multiple_of_81 : y % 81 = 0 :=
sorry

end y_is_multiple_of_3_y_is_multiple_of_9_y_is_multiple_of_27_y_is_multiple_of_81_l133_13315


namespace great_dane_more_than_triple_pitbull_l133_13348

variables (C P G : ℕ)
variables (h1 : G = 307) (h2 : P = 3 * C) (h3 : C + P + G = 439)

theorem great_dane_more_than_triple_pitbull
  : G - 3 * P = 10 :=
by
  sorry

end great_dane_more_than_triple_pitbull_l133_13348


namespace perfect_square_fraction_l133_13386

open Nat

theorem perfect_square_fraction (a b : ℕ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (h : (ab + 1) ∣ (a^2 + b^2)) : ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) :=
by 
  sorry

end perfect_square_fraction_l133_13386


namespace parabola_vertex_coordinates_l133_13398

theorem parabola_vertex_coordinates :
  ∃ (x y : ℝ), (∀ x : ℝ, y = 3 * x^2 + 2) ∧ x = 0 ∧ y = 2 :=
by
  sorry

end parabola_vertex_coordinates_l133_13398


namespace total_number_of_flags_is_12_l133_13385

def number_of_flags : Nat :=
  3 * 2 * 2

theorem total_number_of_flags_is_12 : number_of_flags = 12 := by
  sorry

end total_number_of_flags_is_12_l133_13385


namespace divide_8_friends_among_4_teams_l133_13371

def num_ways_to_divide_friends (n : ℕ) (teams : ℕ) :=
  teams ^ n

theorem divide_8_friends_among_4_teams :
  num_ways_to_divide_friends 8 4 = 65536 :=
by sorry

end divide_8_friends_among_4_teams_l133_13371


namespace cliff_total_rocks_l133_13350

theorem cliff_total_rocks (I S : ℕ) (h1 : S = 2 * I) (h2 : I / 3 = 30) :
  I + S = 270 :=
sorry

end cliff_total_rocks_l133_13350


namespace cos_alpha_minus_pi_over_2_l133_13377

theorem cos_alpha_minus_pi_over_2 (α : ℝ) 
  (h1 : ∃ k : ℤ, α = k * (2 * Real.pi) ∨ α = k * (2 * Real.pi) + Real.pi / 2 ∨ α = k * (2 * Real.pi) + Real.pi ∨ α = k * (2 * Real.pi) + 3 * Real.pi / 2)
  (h2 : Real.cos α = 4 / 5)
  (h3 : Real.sin α = -3 / 5) : 
  Real.cos (α - Real.pi / 2) = -3 / 5 := 
by 
  sorry

end cos_alpha_minus_pi_over_2_l133_13377


namespace pentagon_largest_angle_l133_13343

theorem pentagon_largest_angle
  (F G H I J : ℝ)
  (hF : F = 90)
  (hG : G = 70)
  (hH_eq_I : H = I)
  (hJ : J = 2 * H + 20)
  (sum_angles : F + G + H + I + J = 540) :
  max F (max G (max H (max I J))) = 200 :=
by
  sorry

end pentagon_largest_angle_l133_13343


namespace sum_of_inradii_eq_height_l133_13308

variables (a b c h b1 a1 : ℝ)
variables (r r1 r2 : ℝ)

-- Assume CH is the height of the right-angled triangle ABC from the vertex of the right angle.
-- r, r1, r2 are the radii of the incircles of triangles ABC, AHC, and BHC respectively.
-- Given definitions:
-- BC = a
-- AC = b
-- AB = c
-- AH = b1
-- BH = a1
-- CH = h

-- Formulas for the radii of the respective triangles:
-- r : radius of incircle of triangle ABC = (a + b - h) / 2
-- r1 : radius of incircle of triangle AHC = (h + b1 - b) / 2
-- r2 : radius of incircle of triangle BHC = (h + a1 - a) / 2

theorem sum_of_inradii_eq_height 
  (H₁ : r = (a + b - h) / 2)
  (H₂ : r1 = (h + b1 - b) / 2) 
  (H₃ : r2 = (h + a1 - a) / 2) 
  (H₄ : b1 = b - h) 
  (H₅ : a1 = a - h) : 
  r + r1 + r2 = h :=
by
  sorry

end sum_of_inradii_eq_height_l133_13308


namespace optimal_garden_dimensions_l133_13361

theorem optimal_garden_dimensions :
  ∃ (l w : ℝ), 2 * l + 2 * w = 400 ∧ l ≥ 100 ∧ w ≥ 50 ∧ l ≥ w + 20 ∧ l * w = 9600 :=
by
  sorry

end optimal_garden_dimensions_l133_13361


namespace max_area_of_region_S_l133_13326

-- Define the radii of the circles
def radii : List ℕ := [2, 4, 6, 8]

-- Define the function for the maximum area of region S given the conditions
def max_area_region_S : ℕ := 75

-- Prove the maximum area of region S is 75π
theorem max_area_of_region_S {radii : List ℕ} (h : radii = [2, 4, 6, 8]) 
: max_area_region_S = 75 := by 
  sorry

end max_area_of_region_S_l133_13326


namespace sum_of_ages_l133_13305

-- Definitions for Robert's and Maria's current ages
variables (R M : ℕ)

-- Conditions based on the problem statement
theorem sum_of_ages
  (h1 : R = M + 8)
  (h2 : R + 5 = 3 * (M - 3)) :
  R + M = 30 :=
by
  sorry

end sum_of_ages_l133_13305


namespace solve_inequality_l133_13306

-- Define the conditions
def condition_inequality (x : ℝ) : Prop := abs x + abs (2 * x - 3) ≥ 6

-- Define the solution set form
def solution_set (x : ℝ) : Prop := x ≤ -1 ∨ x ≥ 3

-- State the theorem
theorem solve_inequality (x : ℝ) : condition_inequality x → solution_set x := 
by 
  sorry

end solve_inequality_l133_13306


namespace jenny_ate_65_chocolates_l133_13359

-- Define the number of chocolate squares Mike ate
def MikeChoc := 20

-- Define the function that calculates the chocolates Jenny ate
def JennyChoc (mikeChoc : ℕ) := 3 * mikeChoc + 5

-- The theorem stating the solution
theorem jenny_ate_65_chocolates (h : MikeChoc = 20) : JennyChoc MikeChoc = 65 := by
  -- Automatic proof step
  sorry

end jenny_ate_65_chocolates_l133_13359


namespace value_of_expression_l133_13312

theorem value_of_expression (x : ℤ) (h : x = -2) : (3 * x - 4)^2 = 100 :=
by
  -- Given the hypothesis h: x = -2
  -- Need to show: (3 * x - 4)^2 = 100
  sorry

end value_of_expression_l133_13312


namespace absolute_difference_distance_l133_13322

/-- Renaldo drove 15 kilometers, Ernesto drove 7 kilometers more than one-third of Renaldo's distance, 
Marcos drove -5 kilometers. Prove that the absolute difference between the total distances driven by 
Renaldo and Ernesto combined, and the distance driven by Marcos is 22 kilometers. -/
theorem absolute_difference_distance :
  let renaldo_distance := 15
  let ernesto_distance := 7 + (1 / 3) * renaldo_distance
  let marcos_distance := -5
  abs ((renaldo_distance + ernesto_distance) - marcos_distance) = 22 := by
  sorry

end absolute_difference_distance_l133_13322


namespace coal_extraction_in_four_months_l133_13346

theorem coal_extraction_in_four_months
  (x1 x2 x3 x4 : ℝ)
  (h1 : 4 * x1 + x2 + 2 * x3 + 5 * x4 = 10)
  (h2 : 2 * x1 + 3 * x2 + 2 * x3 + x4 = 7)
  (h3 : 5 * x1 + 2 * x2 + x3 + 4 * x4 = 14) :
  4 * (x1 + x2 + x3 + x4) = 12 :=
by
  sorry

end coal_extraction_in_four_months_l133_13346


namespace even_sum_of_digits_residue_l133_13388

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem even_sum_of_digits_residue (k : ℕ) (h : 2 ≤ k) (r : ℕ) (hr : r < k) :
  ∃ n : ℕ, sum_of_digits n % 2 = 0 ∧ n % k = r := 
sorry

end even_sum_of_digits_residue_l133_13388


namespace ratio_of_x_intercepts_l133_13382

theorem ratio_of_x_intercepts (c : ℝ) (u v : ℝ) (h1 : c ≠ 0) 
  (h2 : u = -c / 8) (h3 : v = -c / 4) : u / v = 1 / 2 :=
by {
  sorry
}

end ratio_of_x_intercepts_l133_13382


namespace intersection_M_N_l133_13331

open Set

noncomputable def M : Set ℝ := {-1, 0, 1}
noncomputable def N : Set ℝ := {x | x^2 + x ≤ 0}

theorem intersection_M_N : M ∩ N = {-1, 0} := sorry

end intersection_M_N_l133_13331


namespace probability_of_specific_roll_l133_13374

noncomputable def probability_event : ℚ :=
  let favorable_outcomes_first_die := 3 -- 1, 2, 3
  let total_outcomes_die := 8
  let probability_first_die := favorable_outcomes_first_die / total_outcomes_die
  
  let favorable_outcomes_second_die := 4 -- 5, 6, 7, 8
  let probability_second_die := favorable_outcomes_second_die / total_outcomes_die
  
  probability_first_die * probability_second_die

theorem probability_of_specific_roll :
  probability_event = 3 / 16 := 
  by
    sorry

end probability_of_specific_roll_l133_13374


namespace solution_count_l133_13330

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem solution_count (a : ℝ) : 
  (∃ x : ℝ, f x = a) ↔ 
  ((a > 2 ∨ a < -2 ∧ ∃! x₁, f x₁ = a) ∨ 
   ((a = 2 ∨ a = -2) ∧ ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = a ∧ f x₂ = a) ∨ 
   (-2 < a ∧ a < 2 ∧ ∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ = a ∧ f x₂ = a ∧ f x₃ = a)) := 
by sorry

end solution_count_l133_13330


namespace ab_value_l133_13389

theorem ab_value (a b : ℤ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 50) : a * b = 7 := by
  sorry

end ab_value_l133_13389


namespace interest_rate_is_20_percent_l133_13392

theorem interest_rate_is_20_percent (P A : ℝ) (t : ℝ) (r : ℝ) 
  (h1 : P = 500) (h2 : A = 1000) (h3 : t = 5) :
  A = P * (1 + r * t) → r = 0.20 :=
by
  intro h
  sorry

end interest_rate_is_20_percent_l133_13392


namespace deepak_current_age_l133_13321

variable (A D : ℕ)

def ratio_condition : Prop := A * 5 = D * 2
def arun_future_age (A : ℕ) : Prop := A + 10 = 30

theorem deepak_current_age (h1 : ratio_condition A D) (h2 : arun_future_age A) : D = 50 := sorry

end deepak_current_age_l133_13321


namespace circle_properties_intercept_length_l133_13301

theorem circle_properties (a r : ℝ) (h1 : a^2 + 16 = r^2) (h2 : (6 - a)^2 + 16 = r^2) (h3 : r > 0) :
  a = 3 ∧ r = 5 :=
by
  sorry

theorem intercept_length (m : ℝ) (h : |24 + m| / 5 = 3) :
  m = -4 ∨ m = -44 :=
by
  sorry

end circle_properties_intercept_length_l133_13301


namespace find_range_a_l133_13317

theorem find_range_a (x y a : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (hy : 2 ≤ y ∧ y ≤ 3) :
  (∀ x y, (1 ≤ x ∧ x ≤ 2) → (2 ≤ y ∧ y ≤ 3) → (xy ≤ a*x^2 + 2*y^2)) ↔ (-1/2 ≤ a) :=
sorry

end find_range_a_l133_13317


namespace functional_equation_solution_l133_13313

theorem functional_equation_solution :
  ∀ (f : ℤ → ℤ), (∀ (m n : ℤ), f (m + f (f n)) = -f (f (m + 1)) - n) → (∀ (p : ℤ), f p = 1 - p) :=
by
  intro f h
  sorry

end functional_equation_solution_l133_13313


namespace all_flowers_bloom_simultaneously_l133_13337

-- Define days of the week
inductive Day : Type
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
deriving DecidableEq

open Day

-- Define bloom conditions for the flowers
def sunflowers_bloom (d : Day) : Prop :=
  d ≠ Tuesday ∧ d ≠ Thursday ∧ d ≠ Sunday

def lilies_bloom (d : Day) : Prop :=
  d ≠ Thursday ∧ d ≠ Saturday

def peonies_bloom (d : Day) : Prop :=
  d ≠ Sunday

-- Define the main theorem
theorem all_flowers_bloom_simultaneously : ∃ d : Day, 
  sunflowers_bloom d ∧ lilies_bloom d ∧ peonies_bloom d ∧
  (∀ d', d' ≠ d → ¬ (sunflowers_bloom d' ∧ lilies_bloom d' ∧ peonies_bloom d')) :=
by
  sorry

end all_flowers_bloom_simultaneously_l133_13337


namespace last_number_is_two_l133_13316

theorem last_number_is_two (A B C D : ℝ)
  (h1 : A + B + C = 18)
  (h2 : B + C + D = 9)
  (h3 : A + D = 13) :
  D = 2 :=
sorry

end last_number_is_two_l133_13316


namespace sum_of_squares_l133_13378

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 14) (h2 : a * b + b * c + a * c = 72) : 
  a^2 + b^2 + c^2 = 52 :=
by
  sorry

end sum_of_squares_l133_13378


namespace nests_count_l133_13384

theorem nests_count (birds nests : ℕ) (h1 : birds = 6) (h2 : birds - nests = 3) : nests = 3 := by
  sorry

end nests_count_l133_13384


namespace find_M_coordinates_l133_13314

-- Definition of the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop :=
  y ^ 2 = 2 * p * x

-- Definition to check if point M lies according to given conditions
def matchesCondition
  (p : ℝ) (M P O F : ℝ × ℝ) : Prop :=
  let xO := O.1
  let yO := O.2
  let xP := P.1
  let yP := P.2
  let xM := M.1
  let yM := M.2
  let xF := F.1
  let yF := F.2
  (xP = 2) ∧ (yP = 2 * p) ∧
  (xO = 0) ∧ (yO = 0) ∧
  (xF = p / 2) ∧ (yF = 0) ∧
  (Real.sqrt ((xM - xP) ^ 2 + (yM - yP) ^ 2) =
  Real.sqrt ((xM - xO) ^ 2 + (yM - yO) ^ 2)) ∧
  (Real.sqrt ((xM - xP) ^ 2 + (yM - yP) ^ 2) =
  Real.sqrt ((xM - xF) ^ 2 + (yM - yF) ^ 2))

-- Prove the coordinates of M satisfy the conditions
theorem find_M_coordinates :
  ∀ p : ℝ, p > 0 →
  matchesCondition p (1/4, 7/4) (2, 2 * p) (0, 0) (p / 2, 0) :=
by
  intros p hp
  simp [parabola, matchesCondition]
  sorry

end find_M_coordinates_l133_13314


namespace smallest_portion_l133_13376

theorem smallest_portion
    (a_1 d : ℚ)
    (h1 : 5 * a_1 + 10 * d = 10)
    (h2 : (a_1 + 2 * d + a_1 + 3 * d + a_1 + 4 * d) / 7 = a_1 + a_1 + d) :
  a_1 = 1 / 6 := 
sorry

end smallest_portion_l133_13376


namespace distinct_pos_numbers_implies_not_zero_at_least_one_of_abc_impossible_for_all_neq_l133_13380

noncomputable section

variables (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) (h1 : 0 < a) 
(h2 : 0 < b) (h3 : 0 < c)

theorem distinct_pos_numbers_implies_not_zero :
  (a - b) ^ 2 + (b - c) ^ 2 + (c - a) ^ 2 ≠ 0 :=
sorry

theorem at_least_one_of_abc :
  a > b ∨ a < b ∨ a = b :=
sorry

theorem impossible_for_all_neq :
  ¬(a ≠ c ∧ b ≠ c ∧ a ≠ b) :=
sorry

end distinct_pos_numbers_implies_not_zero_at_least_one_of_abc_impossible_for_all_neq_l133_13380


namespace number_of_nurses_l133_13391

variables (D N : ℕ)

-- Condition: The total number of doctors and nurses is 250
def total_staff := D + N = 250

-- Condition: The ratio of doctors to nurses is 2 to 3
def ratio_doctors_to_nurses := D = (2 * N) / 3

-- Proof: The number of nurses is 150
theorem number_of_nurses (h1 : total_staff D N) (h2 : ratio_doctors_to_nurses D N) : N = 150 :=
sorry

end number_of_nurses_l133_13391


namespace arithmetic_expression_equality_l133_13397

theorem arithmetic_expression_equality : 
  (1/4 : ℝ) * 8 * (1/16) * 32 * (1/64) * 128 * (1/256) * 512 * (1/1024) * 2048 * (1/4096) * 8192 = 64 := 
by
  sorry

end arithmetic_expression_equality_l133_13397


namespace quadratic_inequality_condition_l133_13311

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - a * x + 1 > 0) → 0 ≤ a ∧ a < 4 :=
sorry

end quadratic_inequality_condition_l133_13311


namespace equality_of_expressions_l133_13303

theorem equality_of_expressions :
  (2^3 ≠ 2 * 3) ∧
  (-(-2)^2 ≠ (-2)^2) ∧
  (-3^2 ≠ 3^2) ∧
  (-2^3 = (-2)^3) :=
by
  sorry

end equality_of_expressions_l133_13303


namespace percent_increase_lines_l133_13383

theorem percent_increase_lines (final_lines increase : ℕ) (h1 : final_lines = 5600) (h2 : increase = 1600) :
  (increase * 100) / (final_lines - increase) = 40 := 
sorry

end percent_increase_lines_l133_13383


namespace part1_part2_part3_l133_13356

-- Definitions from the problem
def initial_cost_per_bottle := 16
def initial_selling_price := 20
def initial_sales_volume := 60
def sales_decrease_per_yuan_increase := 5

def daily_sales_volume (x : ℕ) : ℕ :=
  initial_sales_volume - sales_decrease_per_yuan_increase * x

def profit_per_bottle (x : ℕ) : ℕ :=
  (initial_selling_price - initial_cost_per_bottle) + x

def daily_profit (x : ℕ) : ℕ :=
  daily_sales_volume x * profit_per_bottle x

-- The proofs we need to establish
theorem part1 (x : ℕ) : 
  daily_sales_volume x = 60 - 5 * x ∧ profit_per_bottle x = 4 + x :=
sorry

theorem part2 (x : ℕ) : 
  daily_profit x = 300 → x = 6 ∨ x = 2 :=
sorry

theorem part3 : 
  ∃ x : ℕ, ∀ y : ℕ, (daily_profit x < daily_profit y) → 
              (daily_profit x = 320 ∧ x = 4) :=
sorry

end part1_part2_part3_l133_13356


namespace base4_to_base10_conversion_l133_13368

theorem base4_to_base10_conversion :
  2 * 4^4 + 0 * 4^3 + 3 * 4^2 + 1 * 4^1 + 2 * 4^0 = 566 :=
by
  sorry

end base4_to_base10_conversion_l133_13368


namespace minimum_ticket_cost_l133_13319

-- Definitions of the conditions in Lean
def southern_cities : ℕ := 4
def northern_cities : ℕ := 5
def one_way_ticket_cost (N : ℝ) : ℝ := N
def round_trip_ticket_cost (N : ℝ) : ℝ := 1.6 * N

-- The main theorem to prove
theorem minimum_ticket_cost (N : ℝ) : 
  (∀ (Y1 Y2 Y3 Y4 : ℕ), 
  (∀ (S1 S2 S3 S4 S5 : ℕ), 
  southern_cities = 4 → northern_cities = 5 →
  one_way_ticket_cost N = N →
  round_trip_ticket_cost N = 1.6 * N →
  ∃ (total_cost : ℝ), total_cost = 6.4 * N)) :=
sorry

end minimum_ticket_cost_l133_13319


namespace remi_water_bottle_capacity_l133_13375

-- Let's define the problem conditions
def daily_refills : ℕ := 3
def days : ℕ := 7
def total_spilled : ℕ := 5 + 8 -- Total spilled water in ounces
def total_intake : ℕ := 407 -- Total amount of water drunk in 7 days

-- The capacity of Remi's water bottle is the quantity we need to prove
def bottle_capacity (x : ℕ) : Prop :=
  daily_refills * days * x - total_spilled = total_intake

-- Statement of the proof problem
theorem remi_water_bottle_capacity : bottle_capacity 20 :=
by
  sorry

end remi_water_bottle_capacity_l133_13375


namespace solve_for_x_l133_13345

theorem solve_for_x (x: ℝ) (h: (x-3)^4 = 16): x = 5 := 
by
  sorry

end solve_for_x_l133_13345


namespace decrease_in_profit_due_to_idle_loom_correct_l133_13340

def loom_count : ℕ := 80
def total_sales_value : ℕ := 500000
def monthly_manufacturing_expenses : ℕ := 150000
def establishment_charges : ℕ := 75000
def efficiency_level_idle_loom : ℕ := 100
def sales_per_loom : ℕ := total_sales_value / loom_count
def expenses_per_loom : ℕ := monthly_manufacturing_expenses / loom_count
def profit_contribution_idle_loom : ℕ := sales_per_loom - expenses_per_loom

def decrease_in_profit_due_to_idle_loom : ℕ := 4375

theorem decrease_in_profit_due_to_idle_loom_correct :
  profit_contribution_idle_loom = decrease_in_profit_due_to_idle_loom :=
by sorry

end decrease_in_profit_due_to_idle_loom_correct_l133_13340


namespace probability_samantha_in_sam_not_l133_13320

noncomputable def probability_in_picture_but_not (time_samantha : ℕ) (lap_samantha : ℕ) (time_sam : ℕ) (lap_sam : ℕ) : ℚ :=
  let seconds_raced := 900
  let samantha_laps := seconds_raced / time_samantha
  let sam_laps := seconds_raced / time_sam
  let start_line_samantha := (samantha_laps - (samantha_laps % 1)) * time_samantha + ((samantha_laps % 1) * lap_samantha)
  let start_line_sam := (sam_laps - (sam_laps % 1)) * time_sam + ((sam_laps % 1) * lap_sam)
  let in_picture_duration := 80
  let overlapping_time := 30
  overlapping_time / in_picture_duration

theorem probability_samantha_in_sam_not : probability_in_picture_but_not 120 60 75 25 = 3 / 8 := by
  sorry

end probability_samantha_in_sam_not_l133_13320


namespace film_finishes_earlier_on_first_channel_l133_13393

-- Definitions based on conditions
def DurationSegmentFirstChannel (n : ℕ) : ℝ := n * 22
def DurationSegmentSecondChannel (k : ℕ) : ℝ := k * 11

-- The time when first channel starts the n-th segment
def StartNthSegmentFirstChannel (n : ℕ) : ℝ := (n - 1) * 22

-- The number of segments second channel shows by the time first channel starts the n-th segment
def SegmentsShownSecondChannel (n : ℕ) : ℕ := ((n - 1) * 22) / 11

-- If first channel finishes earlier than second channel
theorem film_finishes_earlier_on_first_channel (n : ℕ) (hn : 1 < n) :
  DurationSegmentFirstChannel n < DurationSegmentSecondChannel (SegmentsShownSecondChannel n + 1) :=
sorry

end film_finishes_earlier_on_first_channel_l133_13393


namespace mary_lambs_count_l133_13399

def initial_lambs : Nat := 6
def baby_lambs : Nat := 2 * 2
def traded_lambs : Nat := 3
def extra_lambs : Nat := 7

theorem mary_lambs_count : initial_lambs + baby_lambs - traded_lambs + extra_lambs = 14 := by
  sorry

end mary_lambs_count_l133_13399


namespace max_value_of_a_plus_b_l133_13341

def max_possible_sum (a b : ℝ) (h1 : 4 * a + 3 * b ≤ 10) (h2 : a + 2 * b ≤ 4) : ℝ :=
  a + b

theorem max_value_of_a_plus_b :
  ∃a b : ℝ, (4 * a + 3 * b ≤ 10) ∧ (a + 2 * b ≤ 4) ∧ (a + b = 14 / 5) :=
by {
  sorry
}

end max_value_of_a_plus_b_l133_13341


namespace slope_angle_at_point_l133_13379

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 4 * x + 8

-- Define the derivative of the function f
def f' (x : ℝ) : ℝ := 3 * x^2 - 4

-- State the problem: Prove the slope angle at point (1, 5) is 135 degrees
theorem slope_angle_at_point (θ : ℝ) (h : θ = 135) :
    f' 1 = -1 := 
by 
    sorry

end slope_angle_at_point_l133_13379


namespace three_digit_numbers_with_repeats_l133_13334

theorem three_digit_numbers_with_repeats :
  (let total_numbers := 9 * 10 * 10
   let non_repeating_numbers := 9 * 9 * 8
   total_numbers - non_repeating_numbers = 252) :=
by
  sorry

end three_digit_numbers_with_repeats_l133_13334


namespace marie_stamps_giveaway_l133_13355

theorem marie_stamps_giveaway :
  let notebooks := 4
  let stamps_per_notebook := 20
  let binders := 2
  let stamps_per_binder := 50
  let fraction_to_keep := 1/4
  let total_stamps := notebooks * stamps_per_notebook + binders * stamps_per_binder
  let stamps_to_keep := fraction_to_keep * total_stamps
  let stamps_to_give_away := total_stamps - stamps_to_keep
  stamps_to_give_away = 135 :=
by
  sorry

end marie_stamps_giveaway_l133_13355
