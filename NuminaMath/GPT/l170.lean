import Mathlib

namespace triangle_obtuse_at_15_l170_170634

-- Define the initial angles of the triangle
def x0 : ℝ := 59.999
def y0 : ℝ := 60
def z0 : ℝ := 60.001

-- Define the recurrence relations for the angles
def x (n : ℕ) : ℝ := (-2)^n * (x0 - 60) + 60
def y (n : ℕ) : ℝ := (-2)^n * (y0 - 60) + 60
def z (n : ℕ) : ℝ := (-2)^n * (z0 - 60) + 60

-- Define the obtuseness condition
def is_obtuse (a : ℝ) : Prop := a > 90

-- The main theorem stating the least positive integer n is 15 for which the triangle A_n B_n C_n is obtuse
theorem triangle_obtuse_at_15 : ∃ n : ℕ, n > 0 ∧ 
  (is_obtuse (x n) ∨ is_obtuse (y n) ∨ is_obtuse (z n)) ∧ n = 15 :=
sorry

end triangle_obtuse_at_15_l170_170634


namespace max_a_value_l170_170319

theorem max_a_value :
  ∀ (a x : ℝ), 
  (x - 1) * x - (a - 2) * (a + 1) ≥ 1 → a ≤ 3 / 2 := sorry

end max_a_value_l170_170319


namespace rectangle_area_theorem_l170_170936

def rectangle_area (d : ℝ) (area : ℝ) : Prop :=
  ∃ w : ℝ, 0 < w ∧ 9 * w^2 + w^2 = d^2 ∧ area = 3 * w^2

theorem rectangle_area_theorem (d : ℝ) : rectangle_area d (3 * d^2 / 10) :=
sorry

end rectangle_area_theorem_l170_170936


namespace mark_bananas_equals_mike_matt_fruits_l170_170239

theorem mark_bananas_equals_mike_matt_fruits :
  (∃ (bananas_mike matt_apples mark_bananas : ℕ),
    bananas_mike = 3 ∧
    matt_apples = 2 * bananas_mike ∧
    mark_bananas = 18 - (bananas_mike + matt_apples) ∧
    mark_bananas = (bananas_mike + matt_apples)) :=
sorry

end mark_bananas_equals_mike_matt_fruits_l170_170239


namespace part_a_part_b_part_c_l170_170443

/-- (a) Given that p = 33 and q = 216, show that the equation f(x) = 0 has 
three distinct integer solutions and the equation g(x) = 0 has two distinct integer solutions.
-/
theorem part_a (p q : ℕ) (h_p : p = 33) (h_q : q = 216) :
  (∃ x1 x2 x3 : ℤ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (x1 * x2 * x3 = 216 ∧ x1 + x2 + x3 = 33 ∧ x1 = 0))
  ∧ (∃ y1 y2 : ℤ, y1 ≠ y2 ∧ (3 * y1 * y2 = 216 ∧ y1 + y1 = 22)) := sorry

/-- (b) Suppose that the equation f(x) = 0 has three distinct integer solutions 
and the equation g(x) = 0 has two distinct integer solutions. Prove the necessary conditions 
for p and q.
-/
theorem part_b (p q : ℕ) 
  (h_f : ∃ x1 x2 x3 : ℤ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (x1 * x2 * x3 = q ∧ x1 + x2 + x3 = p))
  (h_g : ∃ y1 y2 : ℤ, y1 ≠ y2 ∧ (3 * y1 * y2 = q ∧ y1 + y1 = 2 * p)) :
  (∃ k : ℕ, p = 3 * k) ∧ (∃ l : ℕ, q = 9 * l) ∧ (∃ m n : ℕ, p^2 - 3 * q = m^2 ∧ p^2 - 4 * q = n^2) := sorry

/-- (c) Prove that there are infinitely many pairs of positive integers (p, q) for which:
1. The equation f(x) = 0 has three distinct integer solutions.
2. The equation g(x) = 0 has two distinct integer solutions.
3. The greatest common divisor of p and q is 3.
-/
theorem part_c :
  ∃ (p q : ℕ) (infinitely_many : ℕ → Prop),
  (∃ x1 x2 x3 : ℤ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (x1 * x2 * x3 = q ∧ x1 + x2 + x3 = p))
  ∧ (∃ y1 y2 : ℤ, y1 ≠ y2 ∧ (3 * y1 * y2 = q ∧ y1 + y1 = 2 * p))
  ∧ ∃ k : ℕ, gcd p q = 3 ∧ infinitely_many k := sorry

end part_a_part_b_part_c_l170_170443


namespace b_share_l170_170208

-- Definitions based on the conditions
def salary (a b c d : ℕ) : Prop :=
  ∃ x : ℕ, a = 2 * x ∧ b = 3 * x ∧ c = 4 * x ∧ d = 6 * x

def condition (d c : ℕ) : Prop :=
  d = c + 700

-- Proof problem based on the correct answer
theorem b_share (a b c d : ℕ) (x : ℕ) (salary_cond : salary a b c d) (cond : condition d c) :
  b = 1050 := by
  sorry

end b_share_l170_170208


namespace projectile_height_35_l170_170600

theorem projectile_height_35 (t : ℝ) : 
  (∃ t : ℝ, -4.9 * t ^ 2 + 30 * t = 35 ∧ t > 0) → t = 10 / 7 := 
sorry

end projectile_height_35_l170_170600


namespace round_robin_10_person_tournament_l170_170221

noncomputable def num_matches (n : ℕ) : ℕ :=
  n * (n - 1) / 2

theorem round_robin_10_person_tournament :
  num_matches 10 = 45 :=
by
  sorry

end round_robin_10_person_tournament_l170_170221


namespace sum_of_first_8_terms_l170_170025

-- Define the geometric sequence and its properties
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r

-- Define the sum of the first n terms of a sequence
def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

-- Given conditions
def c1 (a : ℕ → ℝ) : Prop := geometric_sequence a 2
def c2 (a : ℕ → ℝ) : Prop := sum_of_first_n_terms a 4 = 1

-- The statement to prove
theorem sum_of_first_8_terms (a : ℕ → ℝ) (h1 : c1 a) (h2 : c2 a) : sum_of_first_n_terms a 8 = 17 :=
by
  sorry

end sum_of_first_8_terms_l170_170025


namespace exists_function_f_l170_170219

-- Define the problem statement
theorem exists_function_f :
  ∃ (f : ℝ → ℝ), ∀ x : ℝ, f (abs (x + 1)) = x^2 + 2 * x :=
sorry

end exists_function_f_l170_170219


namespace integer_pairs_satisfy_equation_l170_170247

theorem integer_pairs_satisfy_equation :
  ∀ (a b : ℤ), b + 1 ≠ 0 → b + 2 ≠ 0 → a + b + 1 ≠ 0 →
    ( (a + 2)/(b + 1) + (a + 1)/(b + 2) = 1 + 6/(a + b + 1) ↔ 
      (a = 1 ∧ b = 0) ∨ (∃ t : ℤ, t ≠ 0 ∧ t ≠ -1 ∧ a = -3 - t ∧ b = t) ) :=
by
  intros a b h1 h2 h3
  sorry

end integer_pairs_satisfy_equation_l170_170247


namespace min_value_of_expression_l170_170007

theorem min_value_of_expression (a b : ℝ) (h_pos_b : 0 < b) (h_eq : 2 * a + b = 1) : 
  42 + b^2 + 1 / (a * b) ≥ 17 / 2 := 
sorry

end min_value_of_expression_l170_170007


namespace evaluate_expression_l170_170798

theorem evaluate_expression : 8^8 * 27^8 * 8^27 * 27^27 = 216^35 :=
by sorry

end evaluate_expression_l170_170798


namespace tenth_number_in_sixteenth_group_is_257_l170_170244

-- Define the general term of the sequence a_n = 2n - 3.
def a_n (n : ℕ) : ℕ := 2 * n - 3

-- Define the first number of the n-th group.
def first_number_of_group (n : ℕ) : ℕ := n^2 - n - 1

-- Define the m-th number in the n-th group.
def group_n_m (n m : ℕ) : ℕ := first_number_of_group n + (m - 1) * 2

theorem tenth_number_in_sixteenth_group_is_257 : group_n_m 16 10 = 257 := by
  sorry

end tenth_number_in_sixteenth_group_is_257_l170_170244


namespace Maryann_total_minutes_worked_l170_170043

theorem Maryann_total_minutes_worked (c a t : ℕ) (h1 : c = 70) (h2 : a = 7 * c) (h3 : t = c + a) : t = 560 := by
  sorry

end Maryann_total_minutes_worked_l170_170043


namespace expand_product_correct_l170_170907

noncomputable def expand_product (x : ℝ) : ℝ :=
  3 * (x + 4) * (x + 5)

theorem expand_product_correct (x : ℝ) :
  expand_product x = 3 * x^2 + 27 * x + 60 :=
by
  unfold expand_product
  sorry

end expand_product_correct_l170_170907


namespace units_digit_7_pow_2023_l170_170607

-- We start by defining a function to compute units digit of powers of 7 modulo 10.
def units_digit_of_7_pow (n : ℕ) : ℕ :=
  (7 ^ n) % 10

-- Define the problem statement: the units digit of 7^2023 is equal to 3.
theorem units_digit_7_pow_2023 : units_digit_of_7_pow 2023 = 3 := sorry

end units_digit_7_pow_2023_l170_170607


namespace max_discount_rate_l170_170353

theorem max_discount_rate (cp sp : ℝ) (h1 : cp = 4) (h2 : sp = 5) : 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 → x ≤ 12 := by
  use 12
  intro h
  sorry

end max_discount_rate_l170_170353


namespace sum_of_terms_l170_170881

theorem sum_of_terms (a d : ℕ) (h1 : a + d < a + 2 * d)
  (h2 : (a + d) * (a + 20) = (a + 2 * d) ^ 2)
  (h3 : a + 20 - a = 20) :
  a + (a + d) + (a + 2 * d) + (a + 20) = 46 :=
by
  sorry

end sum_of_terms_l170_170881


namespace roger_cookie_price_l170_170615

open Classical

theorem roger_cookie_price
  (art_base1 art_base2 art_height : ℕ) 
  (art_cookies_per_batch art_cookie_price roger_cookies_per_batch : ℕ)
  (art_area : ℕ := (art_base1 + art_base2) * art_height / 2)
  (total_dough : ℕ := art_cookies_per_batch * art_area)
  (roger_area : ℚ := total_dough / roger_cookies_per_batch)
  (art_total_earnings : ℚ := art_cookies_per_batch * art_cookie_price) :
  ∀ (roger_cookie_price : ℚ), roger_cookies_per_batch * roger_cookie_price = art_total_earnings →
  roger_cookie_price = 100 / 3 :=
sorry

end roger_cookie_price_l170_170615


namespace problem_statement_l170_170805

theorem problem_statement (x : ℝ) (h1 : x = 3 ∨ x = -3) : 6 * x^2 + 4 * x - 2 * (x^2 - 1) - 2 * (2 * x + x^2) = 20 := 
by {
  sorry
}

end problem_statement_l170_170805


namespace total_apples_l170_170977

theorem total_apples (A B C : ℕ) (h1 : A + B = 11) (h2 : B + C = 18) (h3 : A + C = 19) : A + B + C = 24 :=  
by
  -- Skip the proof
  sorry

end total_apples_l170_170977


namespace charlie_fewer_games_than_dana_l170_170271

theorem charlie_fewer_games_than_dana
  (P D C Ph : ℕ)
  (h1 : P = D + 5)
  (h2 : C < D)
  (h3 : Ph = C + 3)
  (h4 : Ph = 12)
  (h5 : P = Ph + 4) :
  D - C = 2 :=
by
  sorry

end charlie_fewer_games_than_dana_l170_170271


namespace strike_time_10_times_l170_170938

def time_to_strike (n : ℕ) : ℝ :=
  if n = 0 then 0 else (n - 1) * 6

theorem strike_time_10_times : time_to_strike 10 = 60 :=
  by {
    -- Proof outline
    -- time_to_strike 10 = (10 - 1) * 6 = 9 * 6 = 54. Thanks to provided solution -> we shall consider that time take 10 seconds for the clock to start striking.
    sorry
  }

end strike_time_10_times_l170_170938


namespace min_length_PQ_l170_170320

noncomputable def minimum_length (a : ℝ) : ℝ :=
  let x := 2 * a
  let y := a + 2
  let d := |2 * 2 - 2 * 0 + 4| / Real.sqrt (1^2 + (-2)^2)
  let r := Real.sqrt 5
  d - r

theorem min_length_PQ : ∀ (a : ℝ), P ∈ {P : ℝ × ℝ | (P.1 - 2)^2 + P.2^2 = 5} ∧ Q = (2 * a, a + 2) →
  minimum_length a = 3 * Real.sqrt 5 / 5 :=
by
  intro a
  intro h
  rcases h with ⟨hP, hQ⟩
  sorry

end min_length_PQ_l170_170320


namespace quadratic_to_square_form_l170_170420

theorem quadratic_to_square_form (x : ℝ) :
  (x^2 - 6*x + 7 = 0) ↔ ((x - 3)^2 = 2) :=
sorry

end quadratic_to_square_form_l170_170420


namespace trapezium_other_side_length_l170_170427

theorem trapezium_other_side_length :
  ∃ (x : ℝ), 1/2 * (18 + x) * 17 = 323 ∧ x = 20 :=
by
  sorry

end trapezium_other_side_length_l170_170427


namespace solve_for_x0_l170_170321

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 2 then x^2 + 2 else 2 * x

theorem solve_for_x0 (x0 : ℝ) (h : f x0 = 8) : x0 = 4 ∨ x0 = - Real.sqrt 6 :=
  by
  sorry

end solve_for_x0_l170_170321


namespace solution_set_characterization_l170_170636

noncomputable def satisfies_inequality (x : ℝ) : Bool :=
  (3 / (x + 2) + 4 / (x + 6)) > 1

theorem solution_set_characterization :
  ∀ x : ℝ, (satisfies_inequality x) ↔ (x < -7 ∨ (-6 < x ∧ x < -2) ∨ x > 2) :=
by
  intro x
  unfold satisfies_inequality
  -- here we would provide the proof
  sorry

end solution_set_characterization_l170_170636


namespace length_percentage_increase_l170_170109

/--
Given that the area of a rectangle is 460 square meters and the breadth is 20 meters,
prove that the percentage increase in length compared to the breadth is 15%.
-/
theorem length_percentage_increase (A : ℝ) (b : ℝ) (l : ℝ) (hA : A = 460) (hb : b = 20) (hl : l = A / b) :
  ((l - b) / b) * 100 = 15 :=
by
  sorry

end length_percentage_increase_l170_170109


namespace quadrilateral_EFGH_inscribed_in_circle_l170_170729

theorem quadrilateral_EFGH_inscribed_in_circle 
  (a b c : ℝ)
  (angle_EFG : ℝ := 60)
  (angle_EHG : ℝ := 50)
  (EH : ℝ := 5)
  (FG : ℝ := 7)
  (EG : ℝ := a)
  (EF : ℝ := b)
  (GH : ℝ := c)
  : EG = 7 * (Real.sin (70 * Real.pi / 180)) / (Real.sin (50 * Real.pi / 180)) :=
by
  sorry

end quadrilateral_EFGH_inscribed_in_circle_l170_170729


namespace max_value_expression_l170_170560

theorem max_value_expression : ∀ (a b c d : ℝ), 
  a ∈ Set.Icc (-4.5) 4.5 → 
  b ∈ Set.Icc (-4.5) 4.5 → 
  c ∈ Set.Icc (-4.5) 4.5 → 
  d ∈ Set.Icc (-4.5) 4.5 → 
  a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 90 :=
by sorry

end max_value_expression_l170_170560


namespace max_value_of_k_l170_170583

theorem max_value_of_k (m : ℝ) (k : ℝ) (h1 : 0 < m) (h2 : m < 1/2) 
  (h3 : ∀ m, 0 < m → m < 1/2 → (1 / m + 2 / (1 - 2 * m) ≥ k)) : k = 8 :=
sorry

end max_value_of_k_l170_170583


namespace juice_difference_is_eight_l170_170163

-- Defining the initial conditions
def initial_large_barrel : ℕ := 10
def initial_small_barrel : ℕ := 8
def poured_juice : ℕ := 3

-- Defining the final amounts
def final_large_barrel : ℕ := initial_large_barrel + poured_juice
def final_small_barrel : ℕ := initial_small_barrel - poured_juice

-- The statement we need to prove
theorem juice_difference_is_eight :
  final_large_barrel - final_small_barrel = 8 :=
by
  -- Skipping the proof
  sorry

end juice_difference_is_eight_l170_170163


namespace no_rational_roots_l170_170733

theorem no_rational_roots (a b c d : ℕ) (h1 : 1000 * a + 100 * b + 10 * c + d = p) (h2 : Prime p) (h3: Nat.digits 10 p = [a, b, c, d]) : 
  ¬ ∃ x : ℚ, a * x^3 + b * x^2 + c * x + d = 0 :=
by
  sorry

end no_rational_roots_l170_170733


namespace min_abs_x1_x2_l170_170684

noncomputable def f (a x : ℝ) : ℝ := a * Real.sin x - 2 * Real.sqrt 3 * Real.cos x

theorem min_abs_x1_x2 
  (a x1 x2 : ℝ)
  (h_symmetry : ∃ c : ℝ, c = -Real.pi / 6 ∧ (∀ x, f a (x - c) = f a x))
  (h_product : f a x1 * f a x2 = -16) :
  ∃ m : ℝ, m = abs (x1 + x2) ∧ m = 2 * Real.pi / 3 :=
by sorry

end min_abs_x1_x2_l170_170684


namespace christina_walking_speed_l170_170349

noncomputable def christina_speed : ℕ :=
  let distance_between := 270
  let jack_speed := 4
  let lindy_total_distance := 240
  let lindy_speed := 8
  let meeting_time := lindy_total_distance / lindy_speed
  let jack_covered := jack_speed * meeting_time
  let remaining_distance := distance_between - jack_covered
  remaining_distance / meeting_time

theorem christina_walking_speed : christina_speed = 5 := by
  -- Proof will be provided here to verify the theorem, but for now, we use sorry to skip it
  sorry

end christina_walking_speed_l170_170349


namespace find_number_l170_170885

theorem find_number
  (n : ℕ)
  (h : 80641 * n = 806006795) :
  n = 9995 :=
by 
  sorry

end find_number_l170_170885


namespace probability_of_drawing_three_white_marbles_l170_170132

noncomputable def probability_of_three_white_marbles : ℚ :=
  let total_marbles := 5 + 7 + 15
  let prob_first_white := 15 / total_marbles
  let prob_second_white := 14 / (total_marbles - 1)
  let prob_third_white := 13 / (total_marbles - 2)
  prob_first_white * prob_second_white * prob_third_white

theorem probability_of_drawing_three_white_marbles :
  probability_of_three_white_marbles = 2 / 13 := 
by 
  sorry

end probability_of_drawing_three_white_marbles_l170_170132


namespace sum_of_areas_of_circles_l170_170776

-- Definitions of conditions
def r : ℝ := by sorry  -- radius of the circle at vertex A
def s : ℝ := by sorry  -- radius of the circle at vertex B
def t : ℝ := by sorry  -- radius of the circle at vertex C

axiom sum_radii_r_s : r + s = 6
axiom sum_radii_r_t : r + t = 8
axiom sum_radii_s_t : s + t = 10

-- The statement we want to prove
theorem sum_of_areas_of_circles : π * r^2 + π * s^2 + π * t^2 = 56 * π :=
by
  -- Use given axioms and properties of the triangle and circles
  sorry

end sum_of_areas_of_circles_l170_170776


namespace martha_initial_blocks_l170_170850

theorem martha_initial_blocks (final_blocks : ℕ) (found_blocks : ℕ) (initial_blocks : ℕ) : 
  final_blocks = initial_blocks + found_blocks → 
  final_blocks = 84 →
  found_blocks = 80 → 
  initial_blocks = 4 :=
by
  intros h1 h2 h3
  sorry

end martha_initial_blocks_l170_170850


namespace probability_sequence_rw_10_l170_170933

noncomputable def probability_red_white_red : ℚ :=
  (4 / 10) * (6 / 9) * (3 / 8)

theorem probability_sequence_rw_10 :
    probability_red_white_red = 1 / 10 := by
  sorry

end probability_sequence_rw_10_l170_170933


namespace perfect_square_trinomial_m_l170_170724

theorem perfect_square_trinomial_m (m : ℤ) : (∀ x : ℤ, ∃ k : ℤ, x^2 + 2*m*x + 9 = (x + k)^2) ↔ m = 3 ∨ m = -3 :=
by
  sorry

end perfect_square_trinomial_m_l170_170724


namespace problem_f_2004_l170_170519

noncomputable def f (x : ℝ) (a : ℝ) (α : ℝ) (b : ℝ) (β : ℝ) : ℝ := 
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem problem_f_2004 (a α b β : ℝ) 
  (h_non_zero : a ≠ 0 ∧ b ≠ 0 ∧ α ≠ 0 ∧ β ≠ 0) 
  (h_condition : f 2003 a α b β = 6) : 
  f 2004 a α b β = 2 := 
by
  sorry

end problem_f_2004_l170_170519


namespace c_value_for_infinite_solutions_l170_170553

theorem c_value_for_infinite_solutions :
  ∀ (c : ℝ), (∀ (x : ℝ), 3 * (5 + c * x) = 15 * x + 15) ↔ c = 5 :=
by
  -- Proof
  sorry

end c_value_for_infinite_solutions_l170_170553


namespace trig_identity_1_trig_identity_2_l170_170552

theorem trig_identity_1 (θ : ℝ) (h₀ : 0 < θ ∧ θ < π / 2) (h₁ : Real.tan θ = 2) :
  (Real.sin (π - θ) + Real.sin (3 * π / 2 + θ)) / 
  (3 * Real.sin (π / 2 - θ) - 2 * Real.sin (π + θ)) = 1 / 7 :=
by sorry

theorem trig_identity_2 (θ : ℝ) (h₀ : 0 < θ ∧ θ < π / 2) (h₁ : Real.tan θ = 2) :
  (1 - Real.cos (2 * θ)) / 
  (Real.sin (2 * θ) + Real.cos (2 * θ)) = 8 :=
by sorry

end trig_identity_1_trig_identity_2_l170_170552


namespace factorization_problem_l170_170364

theorem factorization_problem 
    (a m n b : ℝ)
    (h1 : (x + 2) * (x + 4) = x^2 + a * x + m)
    (h2 : (x + 1) * (x + 9) = x^2 + n * x + b) :
    (x + 3) * (x + 3) = x^2 + a * x + b :=
by
  sorry

end factorization_problem_l170_170364


namespace intercept_x_parallel_lines_l170_170197

theorem intercept_x_parallel_lines (m : ℝ) 
    (line_l : ∀ x y : ℝ, y + m * (x + 1) = 0) 
    (parallel : ∀ x y : ℝ, y * m - (2 * m + 1) * x = 1) : 
    ∃ x : ℝ, x + 1 = -1 :=
by
  sorry

end intercept_x_parallel_lines_l170_170197


namespace boy_running_time_l170_170937

theorem boy_running_time (s : ℝ) (v : ℝ) (h1 : s = 35) (h2 : v = 9) : 
  (4 * s) / (v * 1000 / 3600) = 56 := by
  sorry

end boy_running_time_l170_170937


namespace smallest_number_greater_300_with_remainder_24_l170_170717

theorem smallest_number_greater_300_with_remainder_24 :
  ∃ n : ℕ, n > 300 ∧ n % 25 = 24 ∧ ∀ k : ℕ, k > 300 ∧ k % 25 = 24 → n ≤ k :=
sorry

end smallest_number_greater_300_with_remainder_24_l170_170717


namespace find_a_l170_170505

theorem find_a (a : ℝ) (h_pos : a > 0) :
  (∀ x y : ℤ, x^2 - a * (x : ℝ) + 4 * a = 0) →
  a = 25 ∨ a = 18 ∨ a = 16 :=
by
  sorry

end find_a_l170_170505


namespace problem_solution_l170_170723

theorem problem_solution
  (a b c : ℕ)
  (h_pos_a : 0 < a ∧ a ≤ 10)
  (h_pos_b : 0 < b ∧ b ≤ 10)
  (h_pos_c : 0 < c ∧ c ≤ 10)
  (h1 : abc % 11 = 2)
  (h2 : 7 * c % 11 = 3)
  (h3 : 8 * b % 11 = 4 + b % 11) : 
  (a + b + c) % 11 = 0 := 
by
  sorry

end problem_solution_l170_170723


namespace find_multiplier_l170_170268

theorem find_multiplier :
  ∀ (x n : ℝ), (x = 5) → (x * n = (16 - x) + 4) → n = 3 :=
by
  intros x n hx heq
  sorry

end find_multiplier_l170_170268


namespace num_ways_to_convert_20d_l170_170508

theorem num_ways_to_convert_20d (n d q : ℕ) (h : 5 * n + 10 * d + 25 * q = 2000) (hn : n ≥ 2) (hq : q ≥ 1) :
    ∃ k : ℕ, k = 130 := sorry

end num_ways_to_convert_20d_l170_170508


namespace divisible_12_or_36_l170_170228

theorem divisible_12_or_36 (x : ℕ) (n : ℕ) (h1 : Nat.Prime x) (h2 : 3 < x) (h3 : x = 3 * n + 1 ∨ x = 3 * n - 1) :
  12 ∣ (x^6 - x^3 - x^2 + x) ∨ 36 ∣ (x^6 - x^3 - x^2 + x) := 
by
  sorry

end divisible_12_or_36_l170_170228


namespace cut_ribbon_l170_170528

theorem cut_ribbon
    (length_ribbon : ℝ)
    (points : ℝ × ℝ × ℝ × ℝ × ℝ)
    (h_length : length_ribbon = 5)
    (h_points : points = (1, 2, 3, 4, 5)) :
    points.2.1 = (11 / 15) * length_ribbon :=
by
    sorry

end cut_ribbon_l170_170528


namespace each_player_plays_36_minutes_l170_170737

-- Definitions based on the conditions
def players := 10
def players_on_field := 8
def match_duration := 45
def equal_play_time (t : Nat) := 360 / players = t

-- Theorem: Each player plays for 36 minutes
theorem each_player_plays_36_minutes : ∃ t, equal_play_time t ∧ t = 36 :=
by
  -- Skipping the proof
  sorry

end each_player_plays_36_minutes_l170_170737


namespace simplify_expression_l170_170027

theorem simplify_expression :
  (3 + 4 + 5 + 7) / 3 + (3 * 6 + 9) / 4 = 157 / 12 :=
by
  sorry

end simplify_expression_l170_170027


namespace system_has_infinite_solutions_l170_170796

theorem system_has_infinite_solutions :
  ∀ (x y : ℝ), (3 * x - 4 * y = 5) ↔ (6 * x - 8 * y = 10) ∧ (9 * x - 12 * y = 15) :=
by
  sorry

end system_has_infinite_solutions_l170_170796


namespace triangle_area_ratio_l170_170984

theorem triangle_area_ratio :
  let base_jihye := 3
  let height_jihye := 2
  let base_donggeon := 3
  let height_donggeon := 6.02
  let area_jihye := (base_jihye * height_jihye) / 2
  let area_donggeon := (base_donggeon * height_donggeon) / 2
  (area_donggeon / area_jihye) = 3.01 :=
by
  sorry

end triangle_area_ratio_l170_170984


namespace negate_proposition_p_l170_170902

theorem negate_proposition_p (f : ℝ → ℝ) :
  (¬ ∀ (x₁ x₂ : ℝ), (f x₂ - f x₁) * (x₂ - x₁) >= 0) ↔ ∃ (x₁ x₂ : ℝ), (f x₂ - f x₁) * (x₂ - x₁) < 0 :=
sorry

end negate_proposition_p_l170_170902


namespace find_smallest_x_satisfying_condition_l170_170700

theorem find_smallest_x_satisfying_condition :
  ∃ x : ℝ, 0 < x ∧ (⌊x^2⌋ - x * ⌊x⌋ = 10) ∧ x = 131 / 11 :=
by
  sorry

end find_smallest_x_satisfying_condition_l170_170700


namespace largest_possible_distance_between_spheres_l170_170346

noncomputable def largest_distance_between_spheres : ℝ :=
  110 + Real.sqrt 1818

theorem largest_possible_distance_between_spheres :
  let center1 := (3, -5, 7)
  let radius1 := 15
  let center2 := (-10, 20, -25)
  let radius2 := 95
  ∀ A B : ℝ × ℝ × ℝ,
    (dist A center1 = radius1) →
    (dist B center2 = radius2) →
    dist A B ≤ largest_distance_between_spheres :=
  sorry

end largest_possible_distance_between_spheres_l170_170346


namespace final_amounts_total_l170_170671

variable {Ben_initial Tom_initial Max_initial: ℕ}
variable {Ben_final Tom_final Max_final: ℕ}

theorem final_amounts_total (h1: Ben_initial = 48) 
                           (h2: Max_initial = 48) 
                           (h3: Ben_final = ((Ben_initial - Tom_initial - Max_initial) * 3 / 2))
                           (h4: Max_final = ((Max_initial * 3 / 2))) 
                           (h5: Tom_final = (Tom_initial * 2 - ((Ben_initial - Tom_initial - Max_initial) / 2) - 48))
                           (h6: Max_final = 48) :
  Ben_final + Tom_final + Max_final = 144 := 
by 
  sorry

end final_amounts_total_l170_170671


namespace mary_cut_roses_l170_170051

theorem mary_cut_roses (initial_roses add_roses total_roses : ℕ) (h1 : initial_roses = 6) (h2 : total_roses = 16) (h3 : total_roses = initial_roses + add_roses) : add_roses = 10 :=
by
  sorry

end mary_cut_roses_l170_170051


namespace unattainable_value_l170_170068

theorem unattainable_value : ∀ x : ℝ, x ≠ -4/3 → (y = (2 - x) / (3 * x + 4) → y ≠ -1/3) :=
by
  intro x hx h
  rw [eq_comm] at h
  sorry

end unattainable_value_l170_170068


namespace t_shirt_price_increase_t_shirt_max_profit_l170_170787

theorem t_shirt_price_increase (x : ℝ) : (x + 10) * (300 - 10 * x) = 3360 → x = 2 := 
by 
  sorry

theorem t_shirt_max_profit (x : ℝ) : (-10 * x^2 + 200 * x + 3000) = 4000 ↔ x = 10 := 
by 
  sorry

end t_shirt_price_increase_t_shirt_max_profit_l170_170787


namespace pyramid_addition_totals_l170_170413

theorem pyramid_addition_totals 
  (initial_faces : ℕ) (initial_edges : ℕ) (initial_vertices : ℕ)
  (first_pyramid_new_faces : ℕ) (first_pyramid_new_edges : ℕ) (first_pyramid_new_vertices : ℕ)
  (second_pyramid_new_faces : ℕ) (second_pyramid_new_edges : ℕ) (second_pyramid_new_vertices : ℕ)
  (cancelling_faces_first : ℕ) (cancelling_faces_second : ℕ) :
  initial_faces = 5 → 
  initial_edges = 9 → 
  initial_vertices = 6 → 
  first_pyramid_new_faces = 3 →
  first_pyramid_new_edges = 3 →
  first_pyramid_new_vertices = 1 →
  second_pyramid_new_faces = 4 →
  second_pyramid_new_edges = 4 →
  second_pyramid_new_vertices = 1 →
  cancelling_faces_first = 1 →
  cancelling_faces_second = 1 →
  initial_faces + first_pyramid_new_faces - cancelling_faces_first 
  + second_pyramid_new_faces - cancelling_faces_second 
  + initial_edges + first_pyramid_new_edges + second_pyramid_new_edges
  + initial_vertices + first_pyramid_new_vertices + second_pyramid_new_vertices 
  = 34 := by sorry

end pyramid_addition_totals_l170_170413


namespace equation_infinitely_many_solutions_l170_170747

theorem equation_infinitely_many_solutions (a : ℝ) :
  (∀ x : ℝ, 4 * (3 * x - 2 * a) = 3 * (4 * x + 18)) ↔ a = -27 / 4 :=
sorry

end equation_infinitely_many_solutions_l170_170747


namespace october_birth_percentage_l170_170894

theorem october_birth_percentage 
  (jan feb mar apr may jun jul aug sep oct nov dec total : ℕ) 
  (h_total : total = 100)
  (h_jan : jan = 2) (h_feb : feb = 4) (h_mar : mar = 8) (h_apr : apr = 5) 
  (h_may : may = 4) (h_jun : jun = 9) (h_jul : jul = 7) (h_aug : aug = 12) 
  (h_sep : sep = 8) (h_oct : oct = 6) (h_nov : nov = 5) (h_dec : dec = 4) : 
  (oct : ℕ) * 100 / total = 6 := 
by
  sorry

end october_birth_percentage_l170_170894


namespace find_x_plus_y_squared_l170_170619

variable (x y a b : ℝ)

def condition1 := x * y = b
def condition2 := (1 / (x ^ 2)) + (1 / (y ^ 2)) = a

theorem find_x_plus_y_squared (h1 : condition1 x y b) (h2 : condition2 x y a) : 
  (x + y) ^ 2 = a * b ^ 2 + 2 * b :=
by
  sorry

end find_x_plus_y_squared_l170_170619


namespace graveling_cost_is_3900_l170_170613

noncomputable def cost_of_graveling_roads 
  (length : ℕ) (breadth : ℕ) (width_road : ℕ) (cost_per_sq_m : ℕ) : ℕ :=
  let area_road_length := length * width_road
  let area_road_breadth := (breadth - width_road) * width_road
  let total_area := area_road_length + area_road_breadth
  total_area * cost_per_sq_m

theorem graveling_cost_is_3900 :
  cost_of_graveling_roads 80 60 10 3 = 3900 := 
by 
  unfold cost_of_graveling_roads
  sorry

end graveling_cost_is_3900_l170_170613


namespace age_ratio_proof_l170_170502

-- Define the ages
def sonAge := 22
def manAge := sonAge + 24

-- Define the ratio computation statement
def ageRatioInTwoYears : ℚ := 
  let sonAgeInTwoYears := sonAge + 2
  let manAgeInTwoYears := manAge + 2
  manAgeInTwoYears / sonAgeInTwoYears

-- The theorem to prove
theorem age_ratio_proof : ageRatioInTwoYears = 2 :=
by
  sorry

end age_ratio_proof_l170_170502


namespace ab_value_l170_170695

-- Defining the conditions as Lean assumptions
theorem ab_value (a b c : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) (h3 : a + b + c = 10) : a * b = 9 :=
by
  sorry

end ab_value_l170_170695


namespace susan_took_longer_l170_170049
variables (M S J T x : ℕ)
theorem susan_took_longer (h1 : M = 2 * S)
                         (h2 : S = J + x)
                         (h3 : J = 30)
                         (h4 : T = M - 7)
                         (h5 : M + S + J + T = 223) : x = 10 :=
sorry

end susan_took_longer_l170_170049


namespace positive_difference_sums_l170_170587

theorem positive_difference_sums : 
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  sum_even_n - sum_odd_n = 250 :=
by
  intros
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  show sum_even_n - sum_odd_n = 250
  sorry

end positive_difference_sums_l170_170587


namespace color_dot_figure_l170_170304

-- Definitions reflecting the problem conditions
def num_colors : ℕ := 3
def first_triangle_coloring_ways : ℕ := 6
def subsequent_triangle_coloring_ways : ℕ := 3
def additional_dot_coloring_ways : ℕ := 2

-- The theorem stating the required proof
theorem color_dot_figure : first_triangle_coloring_ways * 
                           subsequent_triangle_coloring_ways * 
                           subsequent_triangle_coloring_ways * 
                           additional_dot_coloring_ways = 108 := by
sorry

end color_dot_figure_l170_170304


namespace nolan_monthly_savings_l170_170898

theorem nolan_monthly_savings (m k : ℕ) (H : 12 * m = 36 * k) : m = 3 * k := 
by sorry

end nolan_monthly_savings_l170_170898


namespace min_value_of_expression_l170_170036

theorem min_value_of_expression : ∃ x : ℝ, (8 - x) * (6 - x) * (8 + x) * (6 + x) ≥ -196 :=
by
  sorry

end min_value_of_expression_l170_170036


namespace not_necessarily_prime_sum_l170_170124

theorem not_necessarily_prime_sum (nat_ordered_sequence : ℕ → ℕ) :
  (∀ n1 n2 n3 : ℕ, n1 < n2 → n2 < n3 → nat_ordered_sequence n1 + nat_ordered_sequence n2 + nat_ordered_sequence n3 ≠ prime) :=
sorry

end not_necessarily_prime_sum_l170_170124


namespace gcd_m_n_15_lcm_m_n_45_l170_170444

-- Let m and n be integers greater than 0, and 3m + 2n = 225.
variables (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : 3 * m + 2 * n = 225)

-- First part: If the greatest common divisor of m and n is 15, then m + n = 105.
theorem gcd_m_n_15 (h4 : Int.gcd m n = 15) : m + n = 105 :=
sorry

-- Second part: If the least common multiple of m and n is 45, then m + n = 90.
theorem lcm_m_n_45 (h5 : Int.lcm m n = 45) : m + n = 90 :=
sorry

end gcd_m_n_15_lcm_m_n_45_l170_170444


namespace total_students_sampled_l170_170277

theorem total_students_sampled :
  ∀ (seniors juniors freshmen sampled_seniors sampled_juniors sampled_freshmen total_students : ℕ),
    seniors = 1000 →
    juniors = 1200 →
    freshmen = 1500 →
    sampled_freshmen = 75 →
    sampled_seniors = seniors * (sampled_freshmen / freshmen) →
    sampled_juniors = juniors * (sampled_freshmen / freshmen) →
    total_students = sampled_seniors + sampled_juniors + sampled_freshmen →
    total_students = 185 :=
by
sorry

end total_students_sampled_l170_170277


namespace committee_selection_correct_l170_170286

def num_ways_to_choose_committee : ℕ :=
  let total_people := 10
  let president_ways := total_people
  let vp_ways := total_people - 1
  let remaining_people := total_people - 2
  let committee_ways := Nat.choose remaining_people 2
  president_ways * vp_ways * committee_ways

theorem committee_selection_correct :
  num_ways_to_choose_committee = 2520 :=
by
  sorry

end committee_selection_correct_l170_170286


namespace anya_can_obtain_any_composite_number_l170_170417

theorem anya_can_obtain_any_composite_number (n : ℕ) (h : ∃ k, k > 1 ∧ k < n ∧ n % k = 0) : ∃ m ≥ 4, ∀ k, k > 1 → k < m → m % k = 0 → m = n :=
by
  sorry

end anya_can_obtain_any_composite_number_l170_170417


namespace population_increase_rate_is_20_percent_l170_170085

noncomputable def population_increase_rate 
  (initial_population final_population : ℕ) : ℕ :=
  ((final_population - initial_population) * 100) / initial_population

theorem population_increase_rate_is_20_percent :
  population_increase_rate 2000 2400 = 20 :=
by
  unfold population_increase_rate
  sorry

end population_increase_rate_is_20_percent_l170_170085


namespace total_area_equals_total_frequency_l170_170433

-- Definition of frequency and frequency distribution histogram
def frequency_distribution_histogram (frequencies : List ℕ) := ∀ i, (i < frequencies.length) → ℕ

-- Definition that the total area of the small rectangles is the sum of the frequencies
def total_area_of_rectangles (frequencies : List ℕ) : ℕ := frequencies.sum

-- Theorem stating the equivalence
theorem total_area_equals_total_frequency (frequencies : List ℕ) :
  total_area_of_rectangles frequencies = frequencies.sum := 
by
  sorry

end total_area_equals_total_frequency_l170_170433


namespace quadratic_distinct_real_roots_l170_170520

theorem quadratic_distinct_real_roots (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x^2 + m*x + (m + 3) = 0)) ↔ (m < -2 ∨ m > 6) := 
sorry

end quadratic_distinct_real_roots_l170_170520


namespace f_value_at_4_l170_170783

def f : ℝ → ℝ := sorry  -- Define f as a function from ℝ to ℝ

-- Specify the condition that f satisfies for all real numbers x
axiom f_condition (x : ℝ) : f (2^x) + x * f (2^(-x)) = 3

-- Statement to be proven: f(4) = -3
theorem f_value_at_4 : f 4 = -3 :=
by {
  -- Proof goes here
  sorry
}

end f_value_at_4_l170_170783


namespace distance_to_center_square_l170_170637

theorem distance_to_center_square (x y : ℝ) (h : x*x + y*y = 72) (h1 : x*x + (y + 8)*(y + 8) = 72) (h2 : (x + 4)*(x + 4) + y*y = 72) :
  x*x + y*y = 9 ∨ x*x + y*y = 185 :=
by
  sorry

end distance_to_center_square_l170_170637


namespace janet_total_earnings_l170_170076

-- Definitions based on conditions from step a)
def hourly_wage := 70
def hours_worked := 20
def rate_per_pound := 20
def weight_sculpture1 := 5
def weight_sculpture2 := 7

-- Statement for the proof problem
theorem janet_total_earnings : 
  let earnings_from_extermination := hourly_wage * hours_worked
  let earnings_from_sculpture1 := rate_per_pound * weight_sculpture1
  let earnings_from_sculpture2 := rate_per_pound * weight_sculpture2
  earnings_from_extermination + earnings_from_sculpture1 + earnings_from_sculpture2 = 1640 := 
by
  sorry

end janet_total_earnings_l170_170076


namespace shortest_distance_from_vertex_to_path_l170_170884

theorem shortest_distance_from_vertex_to_path
  (r l : ℝ)
  (hr : r = 1)
  (hl : l = 3) :
  ∃ d : ℝ, d = 1.5 :=
by
  -- Given a cone with a base radius of 1 cm and a slant height of 3 cm
  -- We need to prove the shortest distance from the vertex to the path P back to P is 1.5 cm
  sorry

end shortest_distance_from_vertex_to_path_l170_170884


namespace probability_of_passing_through_correct_l170_170489

def probability_of_passing_through (n k : ℕ) : ℚ :=
(2 * k * n - 2 * k^2 + 2 * k - 1) / n^2

theorem probability_of_passing_through_correct (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) :
  probability_of_passing_through n k = (2 * k * n - 2 * k^2 + 2 * k - 1) / n^2 := 
by
  sorry

end probability_of_passing_through_correct_l170_170489


namespace households_used_both_brands_l170_170868

theorem households_used_both_brands 
  (total_households : ℕ)
  (neither_AB : ℕ)
  (only_A : ℕ)
  (h3 : ∀ (both : ℕ), ∃ (only_B : ℕ), only_B = 3 * both)
  (h_sum : ∀ (both : ℕ), neither_AB + only_A + both + (3 * both) = total_households) :
  ∃ (both : ℕ), both = 10 :=
by 
  sorry

end households_used_both_brands_l170_170868


namespace ratio_of_girls_who_like_bb_to_boys_dont_like_bb_l170_170165

-- Definitions based on the given conditions.
def total_students : ℕ := 25
def percent_girls : ℕ := 60
def percent_boys_like_bb : ℕ := 40
def percent_girls_like_bb : ℕ := 80

-- Results from those conditions.
def num_girls : ℕ := percent_girls * total_students / 100
def num_boys : ℕ := total_students - num_girls
def num_boys_like_bb : ℕ := percent_boys_like_bb * num_boys / 100
def num_boys_dont_like_bb : ℕ := num_boys - num_boys_like_bb
def num_girls_like_bb : ℕ := percent_girls_like_bb * num_girls / 100

-- Proof Problem Statement
theorem ratio_of_girls_who_like_bb_to_boys_dont_like_bb :
  (num_girls_like_bb : ℕ) / num_boys_dont_like_bb = 2 / 1 :=
by
  sorry

end ratio_of_girls_who_like_bb_to_boys_dont_like_bb_l170_170165


namespace largest_fraction_l170_170823

theorem largest_fraction :
  let A := (5 : ℚ) / 11
  let B := (6 : ℚ) / 13
  let C := (18 : ℚ) / 37
  let D := (101 : ℚ) / 202
  let E := (200 : ℚ) / 399
  E > A ∧ E > B ∧ E > C ∧ E > D := by
  sorry

end largest_fraction_l170_170823


namespace compute_fraction_power_l170_170075

theorem compute_fraction_power :
  8 * (1 / 4) ^ 4 = 1 / 32 := 
by
  sorry

end compute_fraction_power_l170_170075


namespace abs_eq_three_system1_system2_l170_170447

theorem abs_eq_three : ∀ x : ℝ, |x| = 3 ↔ x = 3 ∨ x = -3 := 
by sorry

theorem system1 : ∀ x y : ℝ, (y * (x - 1) = 0) ∧ (2 * x + 5 * y = 7) → 
(x = 7 / 2 ∧ y = 0) ∨ (x = 1 ∧ y = 1) := 
by sorry

theorem system2 : ∀ x y : ℝ, (x * y - 2 * x - y + 2 = 0) ∧ (x + 6 * y = 3) ∧ (3 * x + y = 8) → 
(x = 1 ∧ y = 5) ∨ (x = 2 ∧ y = 2) := 
by sorry

end abs_eq_three_system1_system2_l170_170447


namespace vertical_asymptote_at_x_4_l170_170245

def P (x : ℝ) : ℝ := x^2 + 2 * x + 8
def Q (x : ℝ) : ℝ := x^2 - 8 * x + 16

theorem vertical_asymptote_at_x_4 : ∃ x : ℝ, Q x = 0 ∧ P x ≠ 0 ∧ x = 4 :=
by
  use 4
  -- Proof skipped
  sorry

end vertical_asymptote_at_x_4_l170_170245


namespace value_of_expression_l170_170872

theorem value_of_expression : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end value_of_expression_l170_170872


namespace greatest_integer_b_not_in_range_of_quadratic_l170_170567

theorem greatest_integer_b_not_in_range_of_quadratic :
  ∀ b : ℤ, (∀ x : ℝ, x^2 + (b : ℝ) * x + 20 ≠ 5) ↔ (b^2 < 60) ∧ (b ≤ 7) := by
  sorry

end greatest_integer_b_not_in_range_of_quadratic_l170_170567


namespace exist_colored_points_r_gt_pi_div_sqrt3_exist_colored_points_r_gt_pi_div_2_l170_170451

theorem exist_colored_points_r_gt_pi_div_sqrt3 (r : ℝ) (hr : r > π / Real.sqrt 3) 
    (coloring : ℝ × ℝ → Prop) : 
    (∀ p, 0 ≤ p.1 ∧ p.1^2 + p.2^2 < r^2 → coloring p ∨ ¬coloring p) → 
    ∃ A B : ℝ × ℝ, (0 ≤ A.1 ∧ 0 ≤ B.1 ∧ A.1^2 + A.2^2 < r^2 ∧ B.1^2 + B.2^2 < r^2 ∧ dist A B = π ∧ coloring A = coloring B) :=
sorry

theorem exist_colored_points_r_gt_pi_div_2 (r : ℝ) (hr : r > π / 2)
    (coloring : ℝ × ℝ → Prop) : 
    (∀ p, 0 ≤ p.1 ∧ p.1^2 + p.2^2 < r^2 → coloring p ∨ ¬coloring p) → 
    ∃ A B : ℝ × ℝ, (0 ≤ A.1 ∧ 0 ≤ B.1 ∧ A.1^2 + A.2^2 < r^2 ∧ B.1^2 + B.2^2 < r^2 ∧ dist A B = π ∧ coloring A = coloring B) :=
sorry

end exist_colored_points_r_gt_pi_div_sqrt3_exist_colored_points_r_gt_pi_div_2_l170_170451


namespace speed_of_stream_l170_170469

theorem speed_of_stream 
  (v : ℝ)
  (boat_speed : ℝ)
  (distance_downstream : ℝ)
  (distance_upstream : ℝ)
  (H1 : boat_speed = 12)
  (H2 : distance_downstream = 32)
  (H3 : distance_upstream = 16)
  (H4 : distance_downstream / (boat_speed + v) = distance_upstream / (boat_speed - v)) :
  v = 4 :=
by
  sorry

end speed_of_stream_l170_170469


namespace volume_of_new_cube_is_2744_l170_170435

-- Define the volume function for a cube given side length
def volume_of_cube (side : ℝ) : ℝ := side ^ 3

-- Given the original cube with a specific volume
def original_volume : ℝ := 343

-- Find the side length of the original cube by taking the cube root of the volume
def original_side_length := (original_volume : ℝ)^(1/3)

-- The side length of the new cube is twice the side length of the original cube
def new_side_length := 2 * original_side_length

-- The volume of the new cube should be calculated
def new_volume := volume_of_cube new_side_length

-- Theorem stating that the new volume is 2744 cubic feet
theorem volume_of_new_cube_is_2744 : new_volume = 2744 := sorry

end volume_of_new_cube_is_2744_l170_170435


namespace chord_length_l170_170039

noncomputable def circle_center (c: ℝ × ℝ) (r: ℝ): Prop := 
  ∃ x y: ℝ, 
    (x - c.1)^2 + (y - c.2)^2 = r^2

noncomputable def line_equation (a b c: ℝ): Prop := 
  ∀ x y: ℝ, 
    a*x + b*y + c = 0

theorem chord_length (a: ℝ): 
  circle_center (2, 1) 2 ∧ line_equation a 1 (-5) ∧
  ∃(chord_len: ℝ), chord_len = 4 → 
  a = 2 :=
by
  sorry

end chord_length_l170_170039


namespace sum_first_15_nat_eq_120_l170_170821

-- Define a function to sum the first n natural numbers
def sum_natural_numbers (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

-- Define the theorem to show that the sum of the first 15 natural numbers equals 120
theorem sum_first_15_nat_eq_120 : sum_natural_numbers 15 = 120 := 
  by
    sorry

end sum_first_15_nat_eq_120_l170_170821


namespace value_of_b_plus_c_l170_170527

theorem value_of_b_plus_c 
  (b c : ℝ) 
  (f : ℝ → ℝ)
  (h_def : ∀ x, f x = x^2 + 2 * b * x + c)
  (h_solution_set : ∀ x, f x ≤ 0 ↔ -1 ≤ x ∧ x ≤ 1) :
  b + c = -1 :=
sorry

end value_of_b_plus_c_l170_170527


namespace puzzle_pieces_count_l170_170330

variable (border_pieces : ℕ) (trevor_pieces : ℕ) (joe_pieces : ℕ) (missing_pieces : ℕ)

def total_puzzle_pieces (border_pieces trevor_pieces joe_pieces missing_pieces : ℕ) : ℕ :=
  border_pieces + trevor_pieces + joe_pieces + missing_pieces

theorem puzzle_pieces_count :
  border_pieces = 75 → 
  trevor_pieces = 105 → 
  joe_pieces = 3 * trevor_pieces → 
  missing_pieces = 5 → 
  total_puzzle_pieces border_pieces trevor_pieces joe_pieces missing_pieces = 500 :=
by
  intros h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃, h₄]
  -- proof step to get total_number_pieces = 75 + 105 + (3 * 105) + 5
  -- hence total_puzzle_pieces = 500
  sorry

end puzzle_pieces_count_l170_170330


namespace equation_is_correct_l170_170943

-- Define the numbers
def n1 : ℕ := 2
def n2 : ℕ := 2
def n3 : ℕ := 11
def n4 : ℕ := 11

-- Define the mathematical expression and the target result
def expression : ℚ := (n1 + n2 / n3) * n4
def target_result : ℚ := 24

-- The proof statement
theorem equation_is_correct : expression = target_result := by
  sorry

end equation_is_correct_l170_170943


namespace coupons_used_l170_170830

theorem coupons_used
  (initial_books : ℝ)
  (sold_books : ℝ)
  (coupons_per_book : ℝ)
  (remaining_books := initial_books - sold_books)
  (total_coupons := remaining_books * coupons_per_book) :
  initial_books = 40.0 →
  sold_books = 20.0 →
  coupons_per_book = 4.0 →
  total_coupons = 80.0 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end coupons_used_l170_170830


namespace simplest_square_root_among_choices_l170_170909

variable {x : ℝ}

def is_simplest_square_root (n : ℝ) : Prop :=
  ∀ m, (m^2 = n) → (m = n)

theorem simplest_square_root_among_choices :
  is_simplest_square_root 7 ∧ ∀ n, n = 24 ∨ n = 1/3 ∨ n = 0.2 → ¬ is_simplest_square_root n :=
by
  sorry

end simplest_square_root_among_choices_l170_170909


namespace total_journey_distance_l170_170089

variable (D : ℝ) (T : ℝ) (v₁ : ℝ) (v₂ : ℝ)

theorem total_journey_distance :
  T = 10 → 
  v₁ = 21 → 
  v₂ = 24 → 
  (T = (D / (2 * v₁)) + (D / (2 * v₂))) → 
  D = 224 :=
by
  intros hT hv₁ hv₂ hDistance
  -- Proof goes here
  sorry

end total_journey_distance_l170_170089


namespace sum_of_geometric_terms_l170_170624

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ q > 0, ∀ n, a (n + 1) = q * a n

theorem sum_of_geometric_terms {a : ℕ → ℝ} 
  (hseq : geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_a1 : a 1 = 1)
  (h_sum135 : a 1 + a 3 + a 5 = 21) :
  a 2 + a 4 + a 6 = 42 :=
sorry

end sum_of_geometric_terms_l170_170624


namespace competition_end_time_l170_170453

def time_in_minutes := 24 * 60  -- Total minutes in 24 hours

def competition_start_time := 14 * 60 + 30  -- 2:30 p.m. in minutes from midnight

theorem competition_end_time :
  competition_start_time + 1440 = competition_start_time :=
by 
  sorry

end competition_end_time_l170_170453


namespace inequality_0_lt_a_lt_1_l170_170120

theorem inequality_0_lt_a_lt_1 (a : ℝ) (h : 0 < a ∧ a < 1) : 
  (1 / a) + (4 / (1 - a)) ≥ 9 :=
by
  sorry

end inequality_0_lt_a_lt_1_l170_170120


namespace time_to_cross_man_l170_170591

-- Define the conversion from km/h to m/s
def kmh_to_ms (speed_kmh : ℕ) : ℕ := (speed_kmh * 1000) / 3600

-- Given conditions
def length_of_train : ℕ := 150
def speed_of_train_kmh : ℕ := 180

-- Calculate speed in m/s
def speed_of_train_ms : ℕ := kmh_to_ms speed_of_train_kmh

-- Proof problem statement
theorem time_to_cross_man : (length_of_train : ℕ) / (speed_of_train_ms : ℕ) = 3 := by
  sorry

end time_to_cross_man_l170_170591


namespace dalton_needs_more_money_l170_170711

theorem dalton_needs_more_money :
  let jump_rope_cost := 9
  let board_game_cost := 15
  let playground_ball_cost := 5
  let puzzle_cost := 8
  let saved_allowance := 7
  let uncle_gift := 14
  let total_cost := jump_rope_cost + board_game_cost + playground_ball_cost + puzzle_cost
  let total_money := saved_allowance + uncle_gift
  (total_cost - total_money) = 16 :=
by
  sorry

end dalton_needs_more_money_l170_170711


namespace greatest_non_sum_complex_l170_170169

def is_complex (n : ℕ) : Prop :=
  ∃ p q : ℕ, p ≠ q ∧ Nat.Prime p ∧ Nat.Prime q ∧ p ∣ n ∧ q ∣ n

theorem greatest_non_sum_complex : ∀ n : ℕ, (¬ ∃ a b : ℕ, is_complex a ∧ is_complex b ∧ a + b = n) → n ≤ 23 :=
by {
  sorry
}

end greatest_non_sum_complex_l170_170169


namespace count_N_less_than_2000_l170_170186

noncomputable def count_valid_N (N : ℕ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ x < 2000 ∧ N < 2000 ∧ x ^ (⌊x⌋₊) = N

theorem count_N_less_than_2000 : 
  ∃ (count : ℕ), count = 412 ∧ 
  (∀ (N : ℕ), N < 2000 → (∃ x : ℝ, x > 0 ∧ x < 2000 ∧ x ^ (⌊x⌋₊) = N) ↔ N < count) :=
sorry

end count_N_less_than_2000_l170_170186


namespace find_a4_b4_c4_l170_170215

theorem find_a4_b4_c4 (a b c : ℝ) (h1 : a + b + c = 3) (h2 : a^2 + b^2 + c^2 = 5) (h3 : a^3 + b^3 + c^3 = 15) : 
    a^4 + b^4 + c^4 = 35 := 
by 
  sorry

end find_a4_b4_c4_l170_170215


namespace pet_food_cost_is_correct_l170_170730

-- Define the given conditions
def rabbit_toy_cost := 6.51
def cage_cost := 12.51
def total_cost := 24.81
def found_dollar := 1.00

-- Define the cost of pet food
def pet_food_cost := total_cost - (rabbit_toy_cost + cage_cost) + found_dollar

-- The statement to prove
theorem pet_food_cost_is_correct : pet_food_cost = 6.79 :=
by
  -- proof steps here
  sorry

end pet_food_cost_is_correct_l170_170730


namespace no_integer_solutions_l170_170991

theorem no_integer_solutions :
  ∀ (m n : ℤ), (m^3 + 4 * m^2 + 3 * m ≠ 8 * n^3 + 12 * n^2 + 6 * n + 1) := by
  sorry

end no_integer_solutions_l170_170991


namespace trig_identity_l170_170298

open Real

theorem trig_identity (θ : ℝ) (h : tan θ = 2) :
  ((sin θ + cos θ) * cos (2 * θ)) / sin θ = -9 / 10 :=
sorry

end trig_identity_l170_170298


namespace two_solutions_exist_l170_170602

theorem two_solutions_exist 
  (a b c : ℝ) 
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_equation : (1 / a) + (1 / b) + (1 / c) = (1 / (a + b + c))) : 
  ∃ (a' b' c' : ℝ), 
    ((a' = 1/3 ∧ b' = 1/3 ∧ c' = 1/3) ∨ (a' = -1/3 ∧ b' = -1/3 ∧ c' = -1/3)) := 
sorry

end two_solutions_exist_l170_170602


namespace coin_problem_exists_l170_170960

theorem coin_problem_exists (n : ℕ) : 
  (∃ n, n % 8 = 6 ∧ n % 7 = 5 ∧ (∀ m, (m % 8 = 6 ∧ m % 7 = 5) → n ≤ m)) →
  (∃ n, (n % 8 = 6) ∧ (n % 7 = 5) ∧ (n % 9 = 0)) :=
by
  sorry

end coin_problem_exists_l170_170960


namespace max_sections_with_five_lines_l170_170106

def sections (n : ℕ) : ℕ :=
  if n = 0 then 1 else
  n * (n + 1) / 2 + 1

theorem max_sections_with_five_lines : sections 5 = 16 := by
  sorry

end max_sections_with_five_lines_l170_170106


namespace bill_left_with_money_l170_170982

def foolsgold (ounces_sold : Nat) (price_per_ounce : Nat) (fine : Nat): Int :=
  (ounces_sold * price_per_ounce) - fine

theorem bill_left_with_money :
  foolsgold 8 9 50 = 22 :=
by
  sorry

end bill_left_with_money_l170_170982


namespace neg_exists_x_sq_lt_one_eqv_forall_x_real_x_leq_neg_one_or_x_geq_one_l170_170259

theorem neg_exists_x_sq_lt_one_eqv_forall_x_real_x_leq_neg_one_or_x_geq_one :
  ¬(∃ x : ℝ, x^2 < 1) ↔ ∀ x : ℝ, x ≤ -1 ∨ x ≥ 1 := 
by 
  sorry

end neg_exists_x_sq_lt_one_eqv_forall_x_real_x_leq_neg_one_or_x_geq_one_l170_170259


namespace tetrahedron_volume_l170_170152

theorem tetrahedron_volume (R S1 S2 S3 S4 : ℝ) : 
    V = (1 / 3) * R * (S1 + S2 + S3 + S4) :=
sorry

end tetrahedron_volume_l170_170152


namespace geometric_sequence_properties_l170_170683

theorem geometric_sequence_properties (a b c : ℝ) (r : ℝ)
    (h1 : a = -2 * r)
    (h2 : b = a * r)
    (h3 : c = b * r)
    (h4 : -8 = c * r) :
    b = -4 ∧ a * c = 16 :=
by
  sorry

end geometric_sequence_properties_l170_170683


namespace arithmetic_sequence_sum_mod_l170_170466

theorem arithmetic_sequence_sum_mod (a d l k S n : ℕ) 
  (h_seq_start : a = 3)
  (h_common_difference : d = 5)
  (h_last_term : l = 103)
  (h_sum_formula : S = (k * (3 + 103)) / 2)
  (h_term_count : k = 21)
  (h_mod_condition : 1113 % 17 = n)
  (h_range_condition : 0 ≤ n ∧ n < 17) : 
  n = 8 :=
by
  sorry

end arithmetic_sequence_sum_mod_l170_170466


namespace range_of_a_for_three_zeros_l170_170745

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a_for_three_zeros (a : ℝ) (h : ∃ x1 x2 x3 : ℝ, f x1 a = 0 ∧ f x2 a = 0 ∧ f x3 a = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) : a < -3 :=
sorry

end range_of_a_for_three_zeros_l170_170745


namespace smallest_successive_number_l170_170061

theorem smallest_successive_number :
  ∃ n : ℕ, n * (n + 1) * (n + 2) = 1059460 ∧ ∀ m : ℕ, m * (m + 1) * (m + 2) = 1059460 → n ≤ m :=
sorry

end smallest_successive_number_l170_170061


namespace hot_dogs_per_pack_l170_170300

-- Define the givens / conditions
def total_hot_dogs : ℕ := 36
def buns_pack_size : ℕ := 9
def same_quantity (h : ℕ) (b : ℕ) := h = b

-- State the theorem to be proven
theorem hot_dogs_per_pack : ∃ h : ℕ, (total_hot_dogs / h = buns_pack_size) ∧ same_quantity (total_hot_dogs / h) (total_hot_dogs / buns_pack_size) := 
sorry

end hot_dogs_per_pack_l170_170300


namespace exp_add_exp_nat_mul_l170_170090

noncomputable def Exp (z : ℝ) : ℝ := Real.exp z

theorem exp_add (a b x : ℝ) :
  Exp ((a + b) * x) = Exp (a * x) * Exp (b * x) := sorry

theorem exp_nat_mul (x : ℝ) (k : ℕ) :
  Exp (k * x) = (Exp x) ^ k := sorry

end exp_add_exp_nat_mul_l170_170090


namespace find_roots_of_equation_l170_170203

theorem find_roots_of_equation
  (a b c d x : ℝ)
  (h1 : a + d = 2015)
  (h2 : b + c = 2015)
  (h3 : a ≠ c)
  (h4 : (x - a) * (x - b) = (x - c) * (x - d)) :
  x = 1007.5 :=
by
  sorry

end find_roots_of_equation_l170_170203


namespace olympiad_not_possible_l170_170011

theorem olympiad_not_possible (x : ℕ) (y : ℕ) (h1 : x + y = 1000) (h2 : y = x + 43) : false := by
  sorry

end olympiad_not_possible_l170_170011


namespace find_f_x_minus_1_l170_170348

theorem find_f_x_minus_1 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = x ^ 2 + 2 * x) :
  ∀ x : ℤ, f (x - 1) = x ^ 2 - 2 * x :=
by
  sorry

end find_f_x_minus_1_l170_170348


namespace max_value_of_f_on_interval_l170_170411

noncomputable def f (x : ℝ) : ℝ := 2^x + x * Real.log (1/4)

theorem max_value_of_f_on_interval :
  ∃ x ∈ Set.Icc (-2:ℝ) 2, f x = (1/4:ℝ) + 4 * Real.log 2 := 
sorry

end max_value_of_f_on_interval_l170_170411


namespace extra_interest_l170_170001

def principal : ℝ := 7000
def rate1 : ℝ := 0.18
def rate2 : ℝ := 0.12
def time : ℝ := 2

def interest (P R T : ℝ) : ℝ := P * R * T

theorem extra_interest :
  interest principal rate1 time - interest principal rate2 time = 840 := by
  sorry

end extra_interest_l170_170001


namespace find_m_l170_170337

theorem find_m (m : ℝ) :
  (∀ x y, x + (m^2 - m) * y = 4 * m - 1 → ∀ x y, 2 * x - y - 5 = 0 → (-1 / (m^2 - m)) = -1 / 2) → 
  (m = -1 ∨ m = 2) :=
sorry

end find_m_l170_170337


namespace cassidy_posters_l170_170014

theorem cassidy_posters (p_two_years_ago : ℕ) (p_double : ℕ) (p_current : ℕ) (p_added : ℕ) 
    (h1 : p_two_years_ago = 14) 
    (h2 : p_double = 2 * p_two_years_ago)
    (h3 : p_current = 22)
    (h4 : p_added = p_double - p_current) : 
    p_added = 6 := 
by
  sorry

end cassidy_posters_l170_170014


namespace num_of_original_numbers_l170_170674

theorem num_of_original_numbers
    (n : ℕ) 
    (S : ℤ) 
    (incorrect_avg correct_avg : ℤ)
    (incorrect_num correct_num : ℤ)
    (h1 : incorrect_avg = 46)
    (h2 : correct_avg = 51)
    (h3 : incorrect_num = 25)
    (h4 : correct_num = 75)
    (h5 : S + correct_num = correct_avg * n)
    (h6 : S + incorrect_num = incorrect_avg * n) :
  n = 10 := by
  sorry

end num_of_original_numbers_l170_170674


namespace Albert_eats_48_slices_l170_170622

theorem Albert_eats_48_slices (large_pizzas : ℕ) (small_pizzas : ℕ) (slices_large : ℕ) (slices_small : ℕ) 
  (h1 : large_pizzas = 2) (h2 : small_pizzas = 2) (h3 : slices_large = 16) (h4 : slices_small = 8) :
  (large_pizzas * slices_large + small_pizzas * slices_small) = 48 := 
by 
  -- sorry is used to skip the proof.
  sorry

end Albert_eats_48_slices_l170_170622


namespace accurate_measurement_l170_170418

-- Define the properties of Dr. Sharadek's tape
structure SharadekTape where
  startsWithHalfCM : Bool -- indicates if the tape starts with a half-centimeter bracket
  potentialError : ℝ -- potential measurement error

-- Define the conditions as an instance of the structure
noncomputable def drSharadekTape : SharadekTape :=
  { startsWithHalfCM := true,
    potentialError := 0.5 }

-- Define a segment with a known precise measurement
structure Segment where
  length : ℝ

noncomputable def AB (N : ℕ) : Segment :=
  { length := N + 0.5 }

-- The theorem stating the correct answer under the given conditions
theorem accurate_measurement (N : ℕ) : 
  ∃ AB : Segment, AB.length = N + 0.5 :=
by
  existsi AB N
  exact rfl

end accurate_measurement_l170_170418


namespace recipe_serves_correctly_l170_170279

theorem recipe_serves_correctly:
  ∀ (cream_fat_per_cup : ℝ) (cream_amount_cup : ℝ) (fat_per_serving : ℝ) (total_servings: ℝ),
    cream_fat_per_cup = 88 →
    cream_amount_cup = 0.5 →
    fat_per_serving = 11 →
    total_servings = (cream_amount_cup * cream_fat_per_cup) / fat_per_serving →
    total_servings = 4 :=
by
  intros cream_fat_per_cup cream_amount_cup fat_per_serving total_servings
  intros hcup hccup hfserv htserv
  sorry

end recipe_serves_correctly_l170_170279


namespace min_value_of_expression_l170_170732

noncomputable def min_expression := 4 * (Real.rpow 5 (1/4) - 1)^2

theorem min_value_of_expression (a b c : ℝ) (h₁ : 1 ≤ a) (h₂ : a ≤ b) (h₃ : b ≤ c) (h₄ : c ≤ 5) :
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 = min_expression :=
sorry

end min_value_of_expression_l170_170732


namespace total_rods_required_l170_170882

-- Define the number of rods needed per unit for each type
def rods_per_sheet_A : ℕ := 10
def rods_per_sheet_B : ℕ := 8
def rods_per_sheet_C : ℕ := 12
def rods_per_beam_A : ℕ := 6
def rods_per_beam_B : ℕ := 4
def rods_per_beam_C : ℕ := 5

-- Define the composition per panel
def sheets_A_per_panel : ℕ := 2
def sheets_B_per_panel : ℕ := 1
def beams_C_per_panel : ℕ := 2

-- Define the number of panels
def num_panels : ℕ := 10

-- Prove the total number of metal rods required for the entire fence
theorem total_rods_required : 
  (sheets_A_per_panel * rods_per_sheet_A + 
   sheets_B_per_panel * rods_per_sheet_B +
   beams_C_per_panel * rods_per_beam_C) * num_panels = 380 :=
by 
  sorry

end total_rods_required_l170_170882


namespace distinct_orders_scoops_l170_170384

-- Conditions
def total_scoops : ℕ := 4
def chocolate_scoops : ℕ := 2
def vanilla_scoops : ℕ := 1
def strawberry_scoops : ℕ := 1

-- Problem statement
theorem distinct_orders_scoops :
  (Nat.factorial total_scoops) / ((Nat.factorial chocolate_scoops) * (Nat.factorial vanilla_scoops) * (Nat.factorial strawberry_scoops)) = 12 := by
  sorry

end distinct_orders_scoops_l170_170384


namespace least_positive_integer_l170_170214

theorem least_positive_integer (N : ℕ) :
  (N % 11 = 10) ∧
  (N % 12 = 11) ∧
  (N % 13 = 12) ∧
  (N % 14 = 13) ∧
  (N % 15 = 14) ∧
  (N % 16 = 15) →
  N = 720719 :=
by
  sorry

end least_positive_integer_l170_170214


namespace algorithm_output_is_127_l170_170738
-- Import the entire Mathlib library

-- Define the possible values the algorithm can output
def possible_values : List ℕ := [15, 31, 63, 127]

-- Define the property where the value is of the form 2^n - 1
def is_exp2_minus_1 (x : ℕ) := ∃ n : ℕ, x = 2^n - 1

-- Define the main theorem to prove the algorithm's output is 127
theorem algorithm_output_is_127 : (∀ x ∈ possible_values, is_exp2_minus_1 x) →
                                      ∃ n : ℕ, 127 = 2^n - 1 :=
by
  -- Define the conditions and the proof steps are left out
  sorry

end algorithm_output_is_127_l170_170738


namespace factorize_expression1_factorize_expression2_l170_170793

section
variable (x y : ℝ)

theorem factorize_expression1 : (x^2 + y^2)^2 - 4 * x^2 * y^2 = (x + y)^2 * (x - y)^2 :=
sorry

theorem factorize_expression2 : 3 * x^3 - 12 * x^2 * y + 12 * x * y^2 = 3 * x * (x - 2 * y)^2 :=
sorry
end

end factorize_expression1_factorize_expression2_l170_170793


namespace digitalEarth_correct_l170_170899

-- Define the possible descriptions of "Digital Earth"
inductive DigitalEarthDescription
| optionA : DigitalEarthDescription
| optionB : DigitalEarthDescription
| optionC : DigitalEarthDescription
| optionD : DigitalEarthDescription

-- Define the correct description according to the solution
def correctDescription : DigitalEarthDescription := DigitalEarthDescription.optionB

-- Define the theorem to prove the equivalence
theorem digitalEarth_correct :
  correctDescription = DigitalEarthDescription.optionB :=
sorry

end digitalEarth_correct_l170_170899


namespace equivalent_multipliers_l170_170646

variable (a b c : ℝ)

theorem equivalent_multipliers :
  (a - 0.07 * a + 0.05 * b) / c = (0.93 * a + 0.05 * b) / c :=
sorry

end equivalent_multipliers_l170_170646


namespace correct_expression_l170_170234

theorem correct_expression (a b : ℝ) : (a - b) * (b + a) = a^2 - b^2 :=
by
  sorry

end correct_expression_l170_170234


namespace instantaneous_velocity_at_3_l170_170276

noncomputable def motion_equation (t : ℝ) : ℝ := 1 - t + t^2

theorem instantaneous_velocity_at_3 :
  (deriv (motion_equation) 3 = 5) :=
by
  sorry

end instantaneous_velocity_at_3_l170_170276


namespace percentage_reduction_is_correct_l170_170442

def percentage_reduction_alcohol_concentration (V_original V_added : ℚ) (C_original : ℚ) : ℚ :=
  let V_total := V_original + V_added
  let Amount_alcohol := V_original * C_original
  let C_new := Amount_alcohol / V_total
  ((C_original - C_new) / C_original) * 100

theorem percentage_reduction_is_correct :
  percentage_reduction_alcohol_concentration 12 28 0.20 = 70 := by
  sorry

end percentage_reduction_is_correct_l170_170442


namespace maria_scored_33_points_l170_170206

-- Defining constants and parameters
def num_shots := 40
def equal_distribution : ℕ := num_shots / 3 -- each type of shot

-- Given success rates
def success_rate_three_point : ℚ := 0.25
def success_rate_two_point : ℚ := 0.50
def success_rate_free_throw : ℚ := 0.80

-- Defining the points per successful shot
def points_per_successful_three_point_shot : ℕ := 3
def points_per_successful_two_point_shot : ℕ := 2
def points_per_successful_free_throw_shot : ℕ := 1

-- Calculating total points scored
def total_points_scored :=
  (success_rate_three_point * points_per_successful_three_point_shot * equal_distribution) +
  (success_rate_two_point * points_per_successful_two_point_shot * equal_distribution) +
  (success_rate_free_throw * points_per_successful_free_throw_shot * equal_distribution)

theorem maria_scored_33_points :
  total_points_scored = 33 := 
sorry

end maria_scored_33_points_l170_170206


namespace find_sum_uv_l170_170136

theorem find_sum_uv (u v : ℝ) (h1 : 3 * u - 7 * v = 29) (h2 : 5 * u + 3 * v = -9) : u + v = -3.363 := 
sorry

end find_sum_uv_l170_170136


namespace unique_digit_sum_l170_170725

theorem unique_digit_sum (X Y M Z F : ℕ) (H1 : X ≠ 0) (H2 : Y ≠ 0) (H3 : M ≠ 0) (H4 : Z ≠ 0) (H5 : F ≠ 0)
  (H6 : X ≠ Y) (H7 : X ≠ M) (H8 : X ≠ Z) (H9 : X ≠ F)
  (H10 : Y ≠ M) (H11 : Y ≠ Z) (H12 : Y ≠ F)
  (H13 : M ≠ Z) (H14 : M ≠ F)
  (H15 : Z ≠ F)
  (H16 : 10 * X + Y ≠ 0) (H17 : 10 * M + Z ≠ 0)
  (H18 : 111 * F = (10 * X + Y) * (10 * M + Z)) :
  X + Y + M + Z + F = 28 := by
  sorry

end unique_digit_sum_l170_170725


namespace area_inequality_l170_170584

theorem area_inequality 
  (α β γ : ℝ) 
  (P Q S : ℝ) 
  (h1 : P / Q = α * β * γ) 
  (h2 : S = Q * (α + 1) * (β + 1) * (γ + 1)) : 
  (S ^ (1 / 3)) ≥ (P ^ (1 / 3)) + (Q ^ (1 / 3)) :=
by
  sorry

end area_inequality_l170_170584


namespace farey_sequence_mediant_l170_170802

theorem farey_sequence_mediant (a b x y c d : ℕ) (h₁ : a * y < b * x) (h₂ : b * x < y * c) (farey_consecutiveness: bx - ay = 1 ∧ cy - dx = 1) : (x / y) = (a+c) / (b+d) := 
by
  sorry

end farey_sequence_mediant_l170_170802


namespace Joe_total_income_l170_170294

theorem Joe_total_income : 
  (∃ I : ℝ, 0.1 * 1000 + 0.2 * 3000 + 0.3 * (I - 500 - 4000) = 848 ∧ I - 500 > 4000) → I = 4993.33 :=
by
  sorry

end Joe_total_income_l170_170294


namespace rotor_permutations_l170_170927

-- Define the factorial function for convenience
def fact : Nat → Nat
| 0     => 1
| (n + 1) => (n + 1) * fact n

-- The main statement to prove
theorem rotor_permutations : (fact 5) / ((fact 2) * (fact 2)) = 30 := by
  sorry

end rotor_permutations_l170_170927


namespace inequality_one_inequality_two_l170_170904

-- Problem (1)
theorem inequality_one {a b : ℝ} (h1 : a ≥ b) (h2 : b > 0) : 2 * a ^ 3 - b ^ 3 ≥ 2 * a * b ^ 2 - a ^ 2 * b :=
sorry

-- Problem (2)
theorem inequality_two {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) : (a ^ 2 / b + b ^ 2 / c + c ^ 2 / a) ≥ 1 :=
sorry

end inequality_one_inequality_two_l170_170904


namespace range_of_a_l170_170278

theorem range_of_a {a : ℝ} : 
  (∃ x : ℝ, (1 / 2 < x ∧ x < 3) ∧ (x ^ 2 - a * x + 1 = 0)) ↔ (2 ≤ a ∧ a < 10 / 3) :=
by
  sorry

end range_of_a_l170_170278


namespace ratio_cost_to_marked_price_l170_170386

variables (x : ℝ) (marked_price : ℝ) (selling_price : ℝ) (cost_price : ℝ)

theorem ratio_cost_to_marked_price :
  (selling_price = marked_price - 1/4 * marked_price) →
  (cost_price = 2/3 * selling_price) →
  (cost_price / marked_price = 1/2) :=
by
  sorry

end ratio_cost_to_marked_price_l170_170386


namespace exactly_one_divisible_by_4_l170_170356

theorem exactly_one_divisible_by_4 :
  (777 % 4 = 1) ∧ (555 % 4 = 3) ∧ (999 % 4 = 3) →
  (∃! (x : ℕ),
    (x = 777 ^ 2021 * 999 ^ 2021 - 1 ∨
     x = 999 ^ 2021 * 555 ^ 2021 - 1 ∨
     x = 555 ^ 2021 * 777 ^ 2021 - 1) ∧
    x % 4 = 0) :=
by
  intros h
  sorry

end exactly_one_divisible_by_4_l170_170356


namespace ratio_of_two_numbers_l170_170914

theorem ratio_of_two_numbers (a b : ℝ) (h1 : a + b = 7 * (a - b)) (h2 : a > b) (h3 : a > 0) (h4 : b > 0) :
  a / b = 4 / 3 :=
by
  sorry

end ratio_of_two_numbers_l170_170914


namespace minimum_value_l170_170288

theorem minimum_value (x : ℝ) (h : x > 0) :
  x^3 + 12*x + 81 / x^4 = 24 := 
sorry

end minimum_value_l170_170288


namespace solve_expression_l170_170096

theorem solve_expression :
  ( (12.05 * 5.4 + 0.6) / (2.3 - 1.8) * (7/3) - (4.07 * 3.5 + 0.45) ^ 2) = 90.493 := 
by 
  sorry

end solve_expression_l170_170096


namespace a11_is_1_l170_170727

variable (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Condition 1: The sum of the first n terms S_n satisfies S_n + S_m = S_{n+m}
axiom sum_condition (n m : ℕ) : S n + S m = S (n + m)

-- Condition 2: a_1 = 1
axiom a1_condition : a 1 = 1

-- Question: prove a_{11} = 1
theorem a11_is_1 : a 11 = 1 :=
sorry


end a11_is_1_l170_170727


namespace mass_percentage_O_is_correct_l170_170539

noncomputable def molar_mass_Al : ℝ := 26.98
noncomputable def molar_mass_O : ℝ := 16.00
noncomputable def num_Al_atoms : ℕ := 2
noncomputable def num_O_atoms : ℕ := 3

noncomputable def molar_mass_Al2O3 : ℝ :=
  (num_Al_atoms * molar_mass_Al) + (num_O_atoms * molar_mass_O)

noncomputable def mass_percentage_O_in_Al2O3 : ℝ :=
  ((num_O_atoms * molar_mass_O) / molar_mass_Al2O3) * 100

theorem mass_percentage_O_is_correct :
  mass_percentage_O_in_Al2O3 = 47.07 :=
by
  sorry

end mass_percentage_O_is_correct_l170_170539


namespace sum_of_three_numbers_l170_170465

theorem sum_of_three_numbers :
  ∀ (a b c : ℕ), 
  a ≤ b ∧ b ≤ c → b = 10 →
  (a + b + c) / 3 = a + 20 →
  (a + b + c) / 3 = c - 30 →
  a + b + c = 60 :=
by
  sorry

end sum_of_three_numbers_l170_170465


namespace gcd_459_357_polynomial_at_neg4_l170_170504

-- Statement for the GCD problem
theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

-- Definition of the polynomial
def f (x : Int) : Int :=
  12 + 35 * x - 8 * x^2 + 79 * x^3 + 6 * x^4 + 5 * x^5 + 3 * x^6

-- Statement for the polynomial evaluation problem
theorem polynomial_at_neg4 : f (-4) = 3392 := by
  sorry

end gcd_459_357_polynomial_at_neg4_l170_170504


namespace roberto_outfit_combinations_l170_170970

-- Define the components of the problem
def trousers_count : ℕ := 5
def shirts_count : ℕ := 7
def jackets_count : ℕ := 4
def disallowed_combinations : ℕ := 7

-- Define the requirements
theorem roberto_outfit_combinations :
  (trousers_count * shirts_count * jackets_count) - disallowed_combinations = 133 := by
  sorry

end roberto_outfit_combinations_l170_170970


namespace worksheets_already_graded_l170_170389

theorem worksheets_already_graded {total_worksheets problems_per_worksheet problems_left_to_grade : ℕ} :
  total_worksheets = 9 →
  problems_per_worksheet = 4 →
  problems_left_to_grade = 16 →
  (total_worksheets - (problems_left_to_grade / problems_per_worksheet)) = 5 :=
by
  intros h1 h2 h3
  sorry

end worksheets_already_graded_l170_170389


namespace koala_fiber_intake_l170_170064

theorem koala_fiber_intake (x : ℝ) (h1 : 0.3 * x = 12) : x = 40 := 
by 
  sorry

end koala_fiber_intake_l170_170064


namespace newspapers_ratio_l170_170105

theorem newspapers_ratio :
  (∀ (j m : ℕ), j = 234 → m = 4 * j + 936 → (m / 4) / j = 2) :=
by
  sorry

end newspapers_ratio_l170_170105


namespace acute_triangle_conditions_l170_170104

-- Definitions exclusively from the conditions provided.
def condition_A (AB AC : ℝ) : Prop :=
  AB * AC > 0

def condition_B (sinA sinB sinC : ℝ) : Prop :=
  sinA / sinB = 4 / 5 ∧ sinA / sinC = 4 / 6 ∧ sinB / sinC = 5 / 6

def condition_C (cosA cosB cosC : ℝ) : Prop :=
  cosA * cosB * cosC > 0

def condition_D (tanA tanB : ℝ) : Prop :=
  tanA * tanB = 2

-- Prove which conditions guarantee that triangle ABC is acute.
theorem acute_triangle_conditions (AB AC sinA sinB sinC cosA cosB cosC tanA tanB : ℝ) :
  (condition_B sinA sinB sinC ∨ condition_C cosA cosB cosC ∨ condition_D tanA tanB) →
  (∀ (A B C : ℝ), A < π / 2 ∧ B < π / 2 ∧ C < π / 2) :=
sorry

end acute_triangle_conditions_l170_170104


namespace inradius_of_triangle_l170_170149

theorem inradius_of_triangle (A p s r : ℝ) (h1 : A = 2 * p) (h2 : A = r * s) (h3 : p = 2 * s) : r = 4 :=
by
  sorry

end inradius_of_triangle_l170_170149


namespace mean_marks_second_section_l170_170760

-- Definitions for the problem conditions
def num_students (section1 section2 section3 section4 : ℕ) : ℕ :=
  section1 + section2 + section3 + section4

def total_marks (section1 section2 section3 section4 : ℕ) (mean1 mean2 mean3 mean4 : ℝ) : ℝ :=
  section1 * mean1 + section2 * mean2 + section3 * mean3 + section4 * mean4

-- The final problem translated into a lean statement
theorem mean_marks_second_section :
  let section1 := 65
  let section2 := 35
  let section3 := 45
  let section4 := 42
  let mean1 := 50
  let mean3 := 55
  let mean4 := 45
  let overall_average := 51.95
  num_students section1 section2 section3 section4 = 187 →
  ((section1 : ℝ) * mean1 + (section2 : ℝ) * M + (section3 : ℝ) * mean3 + (section4 : ℝ) * mean4)
    = 187 * overall_average →
  M = 59.99 :=
by
  intros section1 section2 section3 section4 mean1 mean3 mean4 overall_average Hnum Htotal
  sorry

end mean_marks_second_section_l170_170760


namespace prime_in_A_l170_170137

open Nat

def is_in_A (x : ℕ) : Prop :=
  ∃ (a b : ℤ), x = a^2 + 2*b^2 ∧ a * b ≠ 0

theorem prime_in_A (p : ℕ) [Fact (Nat.Prime p)] (h : is_in_A (p^2)) : is_in_A p :=
  sorry

end prime_in_A_l170_170137


namespace min_value_of_expression_l170_170188

theorem min_value_of_expression (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (hxyz : x * y * z = 27) :
  x + 3 * y + 6 * z >= 27 :=
by
  sorry

end min_value_of_expression_l170_170188


namespace meaningful_sqrt_range_l170_170941

theorem meaningful_sqrt_range (x : ℝ) (h : 0 ≤ x + 3) : -3 ≤ x :=
by sorry

end meaningful_sqrt_range_l170_170941


namespace evaluate_expression_correct_l170_170877

def evaluate_expression : ℚ :=
  let a := 17
  let b := 19
  let c := 23
  let numerator := a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)
  let denominator := a * (1/b + 1/c) + b * (1/c + 1/a) + c * (1/a + 1/b)
  numerator / denominator

theorem evaluate_expression_correct : evaluate_expression = 59 := 
by {
  -- proof skipped
  sorry
}

end evaluate_expression_correct_l170_170877


namespace solve_for_x_l170_170329

theorem solve_for_x (x : ℝ) (h : 1 - 2 * (1 / (1 + x)) = 1 / (1 + x)) : x = 2 := 
  sorry

end solve_for_x_l170_170329


namespace least_positive_base_ten_seven_binary_digits_l170_170159

theorem least_positive_base_ten_seven_binary_digits : ∃ n : ℕ, n = 64 ∧ (n >= 2^6 ∧ n < 2^7) := 
by
  sorry

end least_positive_base_ten_seven_binary_digits_l170_170159


namespace jason_fires_weapon_every_15_seconds_l170_170302

theorem jason_fires_weapon_every_15_seconds
    (flame_duration_per_fire : ℕ)
    (total_flame_duration_per_minute : ℕ)
    (seconds_per_minute : ℕ)
    (h1 : flame_duration_per_fire = 5)
    (h2 : total_flame_duration_per_minute = 20)
    (h3 : seconds_per_minute = 60) :
    seconds_per_minute / (total_flame_duration_per_minute / flame_duration_per_fire) = 15 := 
by
  sorry

end jason_fires_weapon_every_15_seconds_l170_170302


namespace systematic_sampling_example_l170_170189

theorem systematic_sampling_example (rows seats : ℕ) (all_seats_filled : Prop) (chosen_seat : ℕ):
  rows = 50 ∧ seats = 60 ∧ all_seats_filled ∧ chosen_seat = 18 → sampling_method = "systematic_sampling" :=
by
  sorry

end systematic_sampling_example_l170_170189


namespace alyosha_cube_problem_l170_170344

theorem alyosha_cube_problem (n s : ℕ) (h1 : n > s) (h2 : n ^ 3 - s ^ 3 = 152) : 
  n = 6 := 
by
  sorry

end alyosha_cube_problem_l170_170344


namespace simplify_and_evaluate_expr_l170_170692

namespace SimplificationProof

variable (x : ℝ)

theorem simplify_and_evaluate_expr (h : x = Real.sqrt 5 - 1) :
  ((x / (x - 1) - 1) / ((x ^ 2 - 1) / (x ^ 2 - 2 * x + 1))) = Real.sqrt 5 / 5 :=
by
  sorry

end SimplificationProof

end simplify_and_evaluate_expr_l170_170692


namespace has_exactly_one_zero_point_l170_170084

noncomputable def f (x a b : ℝ) : ℝ := (x - 1) * Real.exp x - a * x^2 + b

theorem has_exactly_one_zero_point
  (a b : ℝ) 
  (h1 : (1/2 < a ∧ a ≤ Real.exp 2 / 2 ∧ b > 2 * a) ∨ (0 < a ∧ a < 1/2 ∧ b ≤ 2 * a)) :
  ∃! x : ℝ, f x a b = 0 := 
sorry

end has_exactly_one_zero_point_l170_170084


namespace central_angle_measure_l170_170531

-- Given conditions
def radius : ℝ := 2
def area : ℝ := 4

-- Central angle α
def central_angle : ℝ := 2

-- Theorem statement: The central angle measure is 2 radians
theorem central_angle_measure :
  ∃ α : ℝ, α = central_angle ∧ area = (1/2) * (α * radius) := 
sorry

end central_angle_measure_l170_170531


namespace similar_triangles_side_length_l170_170295

theorem similar_triangles_side_length
  (A1 A2 : ℕ) (k : ℕ) (h1 : A1 - A2 = 18)
  (h2 : A1 = k^2 * A2) (h3 : ∃ n : ℕ, A2 = n)
  (s : ℕ) (h4 : s = 3) :
  s * k = 6 :=
by
  sorry

end similar_triangles_side_length_l170_170295


namespace intersection_with_y_axis_l170_170066

-- Define the given function
def f (x : ℝ) := x^2 + x - 2

-- Prove that the intersection point with the y-axis is (0, -2)
theorem intersection_with_y_axis : f 0 = -2 :=
by {
  sorry
}

end intersection_with_y_axis_l170_170066


namespace annual_decrease_rate_l170_170735

theorem annual_decrease_rate (P : ℕ) (P2 : ℕ) (r : ℝ) : 
  (P = 10000) → (P2 = 8100) → (P2 = P * (1 - r / 100)^2) → (r = 10) :=
by
  intro hP hP2 hEq
  sorry

end annual_decrease_rate_l170_170735


namespace triangle_problem_l170_170252

/-- 
Given a triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively, 
if b = 2 and 2*b*cos B = a*cos C + c*cos A,
prove that B = π/3 and find the maximum area of ΔABC.
-/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) (h1 : b = 2) (h2 : 2 * b * Real.cos B = a * Real.cos C + c * Real.cos A) :
  B = Real.pi / 3 ∧
  (∃ (max_area : ℝ), max_area = Real.sqrt 3 ∧ max_area = (1/2) * a * c * Real.sin B) :=
by
  sorry

end triangle_problem_l170_170252


namespace range_of_a_l170_170305

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x → x ≤ 4 → (a * x^2 - 2 * x + 2) > 0) ↔ (a > 1 / 2) :=
by
  sorry

end range_of_a_l170_170305


namespace find_a1_and_d_l170_170111

-- Defining the arithmetic sequence and its properties
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Given conditions
def conditions (a : ℕ → ℤ) (a_1 : ℤ) (d : ℤ) : Prop :=
  (a 4 + a 5 + a 6 + a 7 = 56) ∧ (a 4 * a 7 = 187) ∧ (a 1 = a_1) ∧ is_arithmetic_sequence a d

-- Proving the solution
theorem find_a1_and_d :
  ∃ (a : ℕ → ℤ) (a_1 d : ℤ),
    conditions a a_1 d ∧ ((a_1 = 5 ∧ d = 2) ∨ (a_1 = 23 ∧ d = -2)) :=
by
  sorry

end find_a1_and_d_l170_170111


namespace train_crossing_time_l170_170019

-- Definitions from conditions
def length_of_train : ℕ := 120
def length_of_bridge : ℕ := 150
def speed_kmph : ℕ := 36
def speed_mps : ℕ := speed_kmph * 1000 / 3600 -- Convert km/h to m/s
def total_distance : ℕ := length_of_train + length_of_bridge

-- Theorem statement
theorem train_crossing_time : total_distance / speed_mps = 27 := by
  sorry

end train_crossing_time_l170_170019


namespace find_tax_rate_l170_170383

variable (total_spent : ℝ) (sales_tax : ℝ) (tax_free_cost : ℝ) (taxable_items_cost : ℝ) 
variable (T : ℝ)

theorem find_tax_rate (h1 : total_spent = 25) 
                      (h2 : sales_tax = 0.30)
                      (h3 : tax_free_cost = 21.7)
                      (h4 : taxable_items_cost = total_spent - tax_free_cost - sales_tax)
                      (h5 : sales_tax = (T / 100) * taxable_items_cost) :
  T = 10 := 
sorry

end find_tax_rate_l170_170383


namespace proof_problem_l170_170050

noncomputable def problem (x y : ℝ) : ℝ :=
  let A := 2 * x + y
  let B := 2 * x - y
  (A ^ 2 - B ^ 2) * (x - 2 * y)

theorem proof_problem : problem (-1) 2 = 80 := by
  sorry

end proof_problem_l170_170050


namespace rhombus_diagonal_l170_170797

theorem rhombus_diagonal (d1 d2 : ℝ) (area : ℝ) (h : d1 * d2 = 2 * area) (hd2 : d2 = 21) (h_area : area = 157.5) : d1 = 15 :=
by
  sorry

end rhombus_diagonal_l170_170797


namespace line_intersects_circle_l170_170315

theorem line_intersects_circle
  (a b r : ℝ)
  (r_nonzero : r ≠ 0)
  (h_outside : a^2 + b^2 > r^2) :
  ∃ x y : ℝ, (x^2 + y^2 = r^2) ∧ (a * x + b * y = r^2) :=
sorry

end line_intersects_circle_l170_170315


namespace integral_f_equals_neg_third_l170_170650

def f (x : ℝ) (c : ℝ) : ℝ := x^2 + 2 * c

theorem integral_f_equals_neg_third :
  (∫ x in (0 : ℝ)..(1 : ℝ), f x (∫ t in (0 : ℝ)..(1 : ℝ), f t (∫ t in (0 : ℝ)..(1 : ℝ), f t 0))) = -1/3 :=
by
  sorry

end integral_f_equals_neg_third_l170_170650


namespace fraction_of_raisins_in_mixture_l170_170947

def cost_of_raisins (R : ℝ) := 3 * R
def cost_of_nuts (R : ℝ) := 3 * (3 * R)
def total_cost (R : ℝ) := cost_of_raisins R + cost_of_nuts R

theorem fraction_of_raisins_in_mixture (R : ℝ) (hR_pos : R > 0) : 
  cost_of_raisins R / total_cost R = 1 / 4 :=
by
  sorry

end fraction_of_raisins_in_mixture_l170_170947


namespace factorize_x3_minus_9x_factorize_a3b_minus_2a2b_plus_ab_l170_170440

theorem factorize_x3_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
sorry

theorem factorize_a3b_minus_2a2b_plus_ab (a b : ℝ) : a^3 * b - 2 * a^2 * b + a * b = a * b * (a - 1)^2 :=
sorry

end factorize_x3_minus_9x_factorize_a3b_minus_2a2b_plus_ab_l170_170440


namespace complement_U_A_correct_l170_170617

-- Define the universal set U and set A
def U : Set Int := {-1, 0, 2}
def A : Set Int := {-1, 0}

-- Define the complement of A in U
def complement_U_A : Set Int := {x | x ∈ U ∧ x ∉ A}

-- Theorem stating the required proof
theorem complement_U_A_correct : complement_U_A = {2} :=
by
  sorry -- Proof will be filled in

end complement_U_A_correct_l170_170617


namespace union_complement_eq_l170_170514

open Set

variable (I A B : Set ℤ)
variable (I_def : I = {-3, -2, -1, 0, 1, 2})
variable (A_def : A = {-1, 1, 2})
variable (B_def : B = {-2, -1, 0})

theorem union_complement_eq :
  A ∪ (I \ B) = {-3, -1, 1, 2} :=
by 
  rw [I_def, A_def, B_def]
  sorry

end union_complement_eq_l170_170514


namespace topaz_sapphire_value_equal_l170_170871

/-
  Problem statement: Given the following conditions:
  1. One sapphire and two topazes are three times more valuable than an emerald: S + 2T = 3E
  2. Seven sapphires and one topaz are eight times more valuable than an emerald: 7S + T = 8E
  
  Prove that the value of one topaz is equal to the value of one sapphire (T = S).
-/

theorem topaz_sapphire_value_equal
  (S T E : ℝ) 
  (h1 : S + 2 * T = 3 * E) 
  (h2 : 7 * S + T = 8 * E) :
  T = S := 
  sorry

end topaz_sapphire_value_equal_l170_170871


namespace interest_rate_of_first_account_l170_170212

theorem interest_rate_of_first_account (r : ℝ) 
  (h1 : 7200 = 4000 + 4000)
  (h2 : 4000 * r = 4000 * 0.10) : 
  r = 0.10 :=
sorry

end interest_rate_of_first_account_l170_170212


namespace average_is_207_l170_170213

variable (x : ℕ)

theorem average_is_207 (h : (201 + 202 + 204 + 205 + 206 + 209 + 209 + 210 + 212 + x) / 10 = 207) :
  x = 212 :=
sorry

end average_is_207_l170_170213


namespace carla_catches_up_in_three_hours_l170_170828

-- Definitions as lean statements based on conditions
def john_speed : ℝ := 30
def carla_speed : ℝ := 35
def john_start_time : ℝ := 0
def carla_start_time : ℝ := 0.5

-- Lean problem statement to prove the catch-up time
theorem carla_catches_up_in_three_hours : 
  ∃ t : ℝ, 35 * t = 30 * (t + 0.5) ∧ t = 3 :=
by
  sorry

end carla_catches_up_in_three_hours_l170_170828


namespace no_information_loss_chart_is_stem_and_leaf_l170_170601

theorem no_information_loss_chart_is_stem_and_leaf :
  "The correct chart with no information loss" = "Stem-and-leaf plot" :=
sorry

end no_information_loss_chart_is_stem_and_leaf_l170_170601


namespace solution_set_of_inequality_l170_170606

theorem solution_set_of_inequality : 
  {x : ℝ | x < x^2} = {x | x < 0} ∪ {x | x > 1} :=
by sorry

end solution_set_of_inequality_l170_170606


namespace total_operation_time_correct_l170_170018

def accessories_per_doll := 2 + 3 + 1 + 5
def number_of_dolls := 12000
def time_per_doll := 45
def time_per_accessory := 10
def total_accessories := number_of_dolls * accessories_per_doll
def time_for_dolls := number_of_dolls * time_per_doll
def time_for_accessories := total_accessories * time_per_accessory
def total_combined_time := time_for_dolls + time_for_accessories

theorem total_operation_time_correct :
  total_combined_time = 1860000 :=
by
  sorry

end total_operation_time_correct_l170_170018


namespace range_of_m_for_function_l170_170360

noncomputable def isFunctionDefinedForAllReal (f : ℝ → ℝ) := ∀ x : ℝ, true

theorem range_of_m_for_function :
  (∀ x : ℝ, x^2 - 2 * m * x + m + 2 > 0) ↔ (-1 < m ∧ m < 2) :=
sorry

end range_of_m_for_function_l170_170360


namespace toothpaste_usage_l170_170157

-- Define the variables involved
variables (t : ℕ) -- total toothpaste in grams
variables (d : ℕ) -- grams used by dad per brushing
variables (m : ℕ) -- grams used by mom per brushing
variables (b : ℕ) -- grams used by Anne + brother per brushing
variables (r : ℕ) -- brushing rate per day
variables (days : ℕ) -- days for toothpaste to run out
variables (N : ℕ) -- family members

-- Given conditions
variables (ht : t = 105)         -- Total toothpaste is 105 grams
variables (hd : d = 3)           -- Dad uses 3 grams per brushing
variables (hm : m = 2)           -- Mom uses 2 grams per brushing
variables (hr : r = 3)           -- Each member brushes three times a day
variables (hdays : days = 5)     -- Toothpaste runs out in 5 days

-- Additional calculations
variable (total_brushing : ℕ)
variable (total_usage_d: ℕ)
variable (total_usage_m: ℕ)
variable (total_usage_parents: ℕ)
variable (total_usage_family: ℕ)

-- Helper expressions
def total_brushing_expr := days * r * 2
def total_usage_d_expr := d * r
def total_usage_m_expr := m * r
def total_usage_parents_expr := (total_usage_d_expr + total_usage_m_expr) * days
def total_usage_family_expr := t - total_usage_parents_expr

-- Assume calculations
variables (h1: total_usage_d = total_usage_d_expr)  
variables (h2: total_usage_m = total_usage_m_expr)
variables (h3: total_usage_parents = total_usage_parents_expr)
variables (h4: total_usage_family = total_usage_family_expr)
variables (h5 : total_brushing = total_brushing_expr)

-- Define the proof
theorem toothpaste_usage : 
  b = total_usage_family / total_brushing := 
  sorry

end toothpaste_usage_l170_170157


namespace simplify_fraction_l170_170438

theorem simplify_fraction : 
  ((2^12)^2 - (2^10)^2) / ((2^11)^2 - (2^9)^2) = 4 := 
by sorry

end simplify_fraction_l170_170438


namespace sequence_remainder_mod_10_l170_170545

def T : ℕ → ℕ := sorry -- Since the actual recursive definition is part of solution steps, we abstract it.
def remainder (n k : ℕ) : ℕ := n % k

theorem sequence_remainder_mod_10 (n : ℕ) (h: n = 2023) : remainder (T n) 10 = 6 :=
by 
  sorry

end sequence_remainder_mod_10_l170_170545


namespace larger_integer_value_l170_170183

theorem larger_integer_value
  (a b : ℕ)
  (h1 : a ≥ b)
  (h2 : ↑a / ↑b = 7 / 3)
  (h3 : a * b = 294) :
  a = 7 * Int.sqrt 14 := 
sorry

end larger_integer_value_l170_170183


namespace sequence_property_l170_170833

theorem sequence_property {m : ℤ} (h_m : |m| ≥ 2) (a : ℕ → ℤ)
  (h_nonzero : ¬(a 1 = 0 ∧ a 2 = 0))
  (h_rec : ∀ n : ℕ, a (n+2) = a (n+1) - m * a n)
  (r s : ℕ) (h_r_s : r > s ∧ s ≥ 2) (h_eq : a r = a s ∧ a s = a 1) :
  r - s ≥ |m| := sorry

end sequence_property_l170_170833


namespace isabella_hair_growth_l170_170108

theorem isabella_hair_growth :
  ∀ (initial final : ℤ), initial = 18 → final = 24 → final - initial = 6 :=
by
  intros initial final h_initial h_final
  rw [h_initial, h_final]
  exact rfl
-- sorry

end isabella_hair_growth_l170_170108


namespace actual_time_when_watch_shows_8_PM_l170_170292

-- Definitions based on the problem's conditions
def initial_time := 8  -- 8:00 AM
def incorrect_watch_time := 14 * 60 + 42  -- 2:42 PM converted to minutes
def actual_time := 15 * 60  -- 3:00 PM converted to minutes
def target_watch_time := 20 * 60  -- 8:00 PM converted to minutes

-- Define to calculate the rate of time loss
def time_loss_rate := (actual_time - incorrect_watch_time) / (actual_time - initial_time * 60)

-- Hypothesis that the watch loses time at a constant rate
axiom constant_rate : ∀ t, t >= initial_time * 60 ∧ t <= actual_time → (t * time_loss_rate) = (actual_time - incorrect_watch_time)

-- Define the target time based on watch reading 8:00 PM
noncomputable def target_actual_time := target_watch_time / time_loss_rate

-- Main theorem: Prove that given the conditions, the target actual time is 8:32 PM
theorem actual_time_when_watch_shows_8_PM : target_actual_time = (20 * 60 + 32) :=
sorry

end actual_time_when_watch_shows_8_PM_l170_170292


namespace total_germs_l170_170032

-- Define variables and constants
namespace BiologyLab

def petri_dishes : ℕ := 75
def germs_per_dish : ℕ := 48

-- The goal is to prove that the total number of germs is as expected.
theorem total_germs : (petri_dishes * germs_per_dish) = 3600 :=
by
  -- Proof is omitted for this example
  sorry

end BiologyLab

end total_germs_l170_170032


namespace positive_difference_of_two_numbers_l170_170173

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
sorry

end positive_difference_of_two_numbers_l170_170173


namespace reciprocal_neg_half_l170_170774

theorem reciprocal_neg_half : (1 / (- (1 / 2) : ℚ) = -2) :=
by
  sorry

end reciprocal_neg_half_l170_170774


namespace chipmunk_families_left_l170_170572

theorem chipmunk_families_left (orig : ℕ) (left : ℕ) (h1 : orig = 86) (h2 : left = 65) : orig - left = 21 := by
  sorry

end chipmunk_families_left_l170_170572


namespace number_reduced_by_10_eq_0_09_three_point_two_four_increased_to_three_two_four_zero_l170_170094

theorem number_reduced_by_10_eq_0_09 : ∃ (x : ℝ), x / 10 = 0.09 ∧ x = 0.9 :=
sorry

theorem three_point_two_four_increased_to_three_two_four_zero : ∃ (y : ℝ), 3.24 * y = 3240 ∧ y = 1000 :=
sorry

end number_reduced_by_10_eq_0_09_three_point_two_four_increased_to_three_two_four_zero_l170_170094


namespace time_ratio_xiao_ming_schools_l170_170345

theorem time_ratio_xiao_ming_schools
  (AB BC CD : ℝ) 
  (flat_speed uphill_speed downhill_speed : ℝ)
  (h1 : AB + BC + CD = 1) 
  (h2 : AB / BC = 1 / 2)
  (h3 : BC / CD = 2 / 1)
  (h4 : flat_speed / uphill_speed = 3 / 2)
  (h5 : uphill_speed / downhill_speed = 2 / 4) :
  (AB / flat_speed + BC / uphill_speed + CD / downhill_speed) / 
  (AB / flat_speed + BC / downhill_speed + CD / uphill_speed) = 19 / 16 :=
by
  sorry

end time_ratio_xiao_ming_schools_l170_170345


namespace simplify_fraction_sum_l170_170551

variable (a b c : ℝ)
variable (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)

theorem simplify_fraction_sum (x : ℝ) (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ( (x + a) ^ 2 / ((a - b) * (a - c))
  + (x + b) ^ 2 / ((b - a) * (b - c))
  + (x + c) ^ 2 / ((c - a) * (c - b)) )
  = a * x + b * x + c * x - a - b - c :=
sorry

end simplify_fraction_sum_l170_170551


namespace log_sum_correct_l170_170415

noncomputable def log_sum : Prop :=
  let x := (3/2)
  let y := (5/3)
  (x + y) = (19/6)

theorem log_sum_correct : log_sum :=
by
  sorry

end log_sum_correct_l170_170415


namespace find_students_l170_170813

theorem find_students (n : ℕ) (h1 : n % 8 = 5) (h2 : n % 6 = 1) (h3 : n < 50) : n = 13 :=
sorry

end find_students_l170_170813


namespace conditional_probability_event_B_given_event_A_l170_170503

-- Definitions of events A and B
def event_A := {outcomes | ∃ i j k, outcomes = (i, j, k) ∧ (i = 1 ∨ j = 1 ∨ k = 1)}
def event_B := {outcomes | ∃ i j k, outcomes = (i, j, k) ∧ (i + j + k = 1)}

-- Calculation of probabilities
def probability_AB := 3 / 8
def probability_A := 7 / 8

-- Prove conditional probability
theorem conditional_probability_event_B_given_event_A :
  (probability_AB / probability_A) = 3 / 7 :=
by
  sorry

end conditional_probability_event_B_given_event_A_l170_170503


namespace circle_condition_l170_170227

theorem circle_condition (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 - 2*m*x - 2*m*y + 2*m^2 + m - 1 = 0) →
  m < 1 :=
sorry

end circle_condition_l170_170227


namespace range_of_m_l170_170963

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 - 4 * x ≥ m) → m ≤ -3 := 
sorry

end range_of_m_l170_170963


namespace find_certain_number_l170_170676

def certain_number (x : ℚ) : Prop := 5 * 1.6 - (1.4 * x) / 1.3 = 4

theorem find_certain_number : certain_number (-(26/7)) :=
by 
  simp [certain_number]
  sorry

end find_certain_number_l170_170676


namespace monotonic_increase_interval_l170_170742

noncomputable def interval_of_monotonic_increase (k : ℤ) : Set ℝ :=
  {x : ℝ | k * Real.pi - Real.pi / 12 ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 12}

theorem monotonic_increase_interval 
    (ω : ℝ)
    (hω : 0 < ω)
    (hperiod : Real.pi = 2 * Real.pi / ω) :
    ∀ k : ℤ, ∃ I : Set ℝ, I = interval_of_monotonic_increase k := 
by
  sorry

end monotonic_increase_interval_l170_170742


namespace sum_6n_is_correct_l170_170378

theorem sum_6n_is_correct {n : ℕ} (h : (5 * n * (5 * n + 1)) / 2 = (n * (n + 1)) / 2 + 200) :
  (6 * n * (6 * n + 1)) / 2 = 300 :=
by sorry

end sum_6n_is_correct_l170_170378


namespace incorrect_inequality_transformation_l170_170130

theorem incorrect_inequality_transformation 
    (a b : ℝ) 
    (h : a > b) 
    : ¬(1 - a > 1 - b) := 
by {
  sorry 
}

end incorrect_inequality_transformation_l170_170130


namespace box_weight_difference_l170_170956

theorem box_weight_difference:
  let w1 := 2
  let w2 := 3
  let w3 := 13
  let w4 := 7
  let w5 := 10
  (max (max (max (max w1 w2) w3) w4) w5) - (min (min (min (min w1 w2) w3) w4) w5) = 11 :=
by
  sorry

end box_weight_difference_l170_170956


namespace num_of_elements_l170_170780

-- Lean statement to define and prove the problem condition
theorem num_of_elements (n S : ℕ) (h1 : (S + 26) / n = 5) (h2 : (S + 36) / n = 6) : n = 10 := by
  sorry

end num_of_elements_l170_170780


namespace polygon_sides_l170_170326

theorem polygon_sides (a : ℝ) (n : ℕ) (h1 : a = 140) (h2 : 180 * (n-2) = n * a) : n = 9 := 
by sorry

end polygon_sides_l170_170326


namespace total_cost_l170_170499

def cost(M R F : ℝ) := 10 * M = 24 * R ∧ 6 * F = 2 * R ∧ F = 23

theorem total_cost (M R F : ℝ) (h : cost M R F) : 
  4 * M + 3 * R + 5 * F = 984.40 :=
by
  sorry

end total_cost_l170_170499


namespace smallest_number_divisible_by_conditions_l170_170720

theorem smallest_number_divisible_by_conditions:
  ∃ n : ℕ, (∀ d ∈ [8, 12, 22, 24], d ∣ (n - 12)) ∧ (n = 252) :=
by
  sorry

end smallest_number_divisible_by_conditions_l170_170720


namespace sum_cube_eq_l170_170853

theorem sum_cube_eq (a b c : ℝ) (h : a + b + c = 0) : a^3 + b^3 + c^3 = 3 * a * b * c :=
by 
  sorry

end sum_cube_eq_l170_170853


namespace find_x_value_l170_170959

theorem find_x_value (x : ℝ) (h1 : Real.sin (π / 2 - x) = -Real.sqrt 3 / 2) (h2 : π < x ∧ x < 2 * π) : x = 7 * π / 6 :=
sorry

end find_x_value_l170_170959


namespace find_value_of_m_and_n_l170_170497

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^3 + 3*x^2 + m * x
noncomputable def g (x : ℝ) (n : ℝ) : ℝ := Real.log (x + 1) + n * x

theorem find_value_of_m_and_n (m n : ℝ) (h₀ : n > 0) 
  (h₁ : f (-1) m = -1) 
  (h₂ : ∀ x : ℝ, f x m = g x n → x = 0) :
  m + n = 5 := 
by 
  sorry

end find_value_of_m_and_n_l170_170497


namespace find_n_l170_170911

noncomputable def binom (n k : ℕ) := Nat.choose n k

theorem find_n 
  (n : ℕ)
  (h1 : (binom (n-6) 7) / binom n 7 = (6 * binom (n-7) 6) / binom n 7)
  : n = 48 := by
  sorry

end find_n_l170_170911


namespace snake_body_length_l170_170455

theorem snake_body_length (L : ℝ) (H : ℝ) (h1 : H = L / 10) (h2 : L = 10) : L - H = 9 :=
by
  sorry

end snake_body_length_l170_170455


namespace sum_of_first_15_terms_l170_170664

theorem sum_of_first_15_terms (S : ℕ → ℕ) (h1 : S 5 = 48) (h2 : S 10 = 60) : S 15 = 72 :=
sorry

end sum_of_first_15_terms_l170_170664


namespace green_fish_count_l170_170973

theorem green_fish_count (B O G : ℕ) (h1 : B = (2 / 5) * 200)
  (h2 : O = 2 * B - 30) (h3 : G = (3 / 2) * O) (h4 : B + O + G = 200) : 
  G = 195 :=
by
  sorry

end green_fish_count_l170_170973


namespace tunnel_length_scale_l170_170134

theorem tunnel_length_scale (map_length_cm : ℝ) (scale_ratio : ℝ) (convert_factor : ℝ) : 
  map_length_cm = 7 → scale_ratio = 38000 → convert_factor = 100000 →
  (map_length_cm * scale_ratio / convert_factor) = 2.66 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end tunnel_length_scale_l170_170134


namespace smallest_integer_CC6_DD8_l170_170922

def is_valid_digit_in_base (n : ℕ) (b : ℕ) : Prop :=
  n < b

theorem smallest_integer_CC6_DD8 : 
  ∃ C D : ℕ, is_valid_digit_in_base C 6 ∧ is_valid_digit_in_base D 8 ∧ 7 * C = 9 * D ∧ 7 * C = 63 :=
by
  sorry

end smallest_integer_CC6_DD8_l170_170922


namespace simplify_expression_l170_170432

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (a b : V)

theorem simplify_expression :
  (1/2 : ℝ) • (2 • a + 8 • b) - (4 • a - 2 • b) = 6 • b - 3 • a :=
by sorry

end simplify_expression_l170_170432


namespace total_payment_correct_l170_170112

noncomputable def calculate_total_payment : ℝ :=
  let original_price_vase := 200
  let discount_vase := 0.35 * original_price_vase
  let sale_price_vase := original_price_vase - discount_vase
  let tax_vase := 0.10 * sale_price_vase

  let original_price_teacups := 300
  let discount_teacups := 0.20 * original_price_teacups
  let sale_price_teacups := original_price_teacups - discount_teacups
  let tax_teacups := 0.08 * sale_price_teacups

  let original_price_plate := 500
  let sale_price_plate := original_price_plate
  let tax_plate := 0.10 * sale_price_plate

  (sale_price_vase + tax_vase) + (sale_price_teacups + tax_teacups) + (sale_price_plate + tax_plate)

theorem total_payment_correct : calculate_total_payment = 952.20 :=
by sorry

end total_payment_correct_l170_170112


namespace triangle_value_l170_170380

-- Define the operation \(\triangle\)
def triangle (m n p q : ℕ) : ℕ := (m * m) * p * q / n

-- Define the problem statement
theorem triangle_value : triangle 5 6 9 4 = 150 := by
  sorry

end triangle_value_l170_170380


namespace tangent_circles_pass_through_homothety_center_l170_170614

-- Define the necessary structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def is_tangent_to_line (ω : Circle) (L : ℝ → ℝ) : Prop :=
  sorry -- Definition of tangency to a line

def is_tangent_to_circle (ω : Circle) (C : Circle) : Prop :=
  sorry -- Definition of tangency to another circle

theorem tangent_circles_pass_through_homothety_center
  (L : ℝ → ℝ) (C : Circle) (ω : Circle)
  (H_ext H_int : ℝ × ℝ)
  (H_tangency_line : is_tangent_to_line ω L)
  (H_tangency_circle : is_tangent_to_circle ω C) :
  ∃ P Q : ℝ × ℝ, 
    (is_tangent_to_line ω L ∧ is_tangent_to_circle ω C) →
    (P = Q ∧ (P = H_ext ∨ P = H_int)) :=
by
  sorry

end tangent_circles_pass_through_homothety_center_l170_170614


namespace alice_speed_is_6_5_l170_170231

-- Definitions based on the conditions.
def a : ℝ := sorry -- Alice's speed
def b : ℝ := a + 3 -- Bob's speed

-- Alice cycles towards the park 80 miles away and Bob meets her 15 miles away from the park
def d_alice : ℝ := 65 -- Alice's distance traveled (80 - 15)
def d_bob : ℝ := 95 -- Bob's distance traveled (80 + 15)

-- Equating the times
def time_eqn := d_alice / a = d_bob / b

-- Alice's speed is 6.5 mph
theorem alice_speed_is_6_5 : a = 6.5 :=
by
  have h1 : b = a + 3 := sorry
  have h2 : a * 65 = (a + 3) * 95 := sorry
  have h3 : 30 * a = 195 := sorry
  have h4 : a = 6.5 := sorry
  exact h4

end alice_speed_is_6_5_l170_170231


namespace count_parallelograms_392_l170_170578

-- Define the conditions in Lean
def is_lattice_point (x y : ℕ) : Prop :=
  ∃ q : ℕ, x = q ∧ y = q

def on_line_y_eq_x (x y : ℕ) : Prop :=
  y = x ∧ is_lattice_point x y

def on_line_y_eq_mx (x y : ℕ) (m : ℕ) : Prop :=
  y = m * x ∧ is_lattice_point x y ∧ m > 1

def area_parallelogram (q s m : ℕ) : ℕ :=
  (m - 1) * q * s

-- Define the target theorem
theorem count_parallelograms_392 :
  (∀ (q s m : ℕ),
    on_line_y_eq_x q q →
    on_line_y_eq_mx s (m * s) m →
    area_parallelogram q s m = 250000) →
  (∃! n : ℕ, n = 392) :=
sorry

end count_parallelograms_392_l170_170578


namespace volume_of_pyramid_l170_170266

noncomputable def volume_of_pyramid_QEFGH : ℝ := 
  let EF := 10
  let FG := 3
  let base_area := EF * FG
  let height := 9
  (1/3) * base_area * height

theorem volume_of_pyramid {EF FG : ℝ} (hEF : EF = 10) (hFG : FG = 3)
  (QE_perpendicular_EF : true) (QE_perpendicular_EH : true) (QE_height : QE = 9) :
  volume_of_pyramid_QEFGH = 90 := by
  sorry

end volume_of_pyramid_l170_170266


namespace evaluate_expression_l170_170550

theorem evaluate_expression :
  let a := 24
  let b := 7
  3 * (a + b) ^ 2 - (a ^ 2 + b ^ 2) = 2258 :=
by
  let a := 24
  let b := 7
  sorry

end evaluate_expression_l170_170550


namespace eq_is_quadratic_iff_m_zero_l170_170910

theorem eq_is_quadratic_iff_m_zero (m : ℝ) : (|m| + 2 = 2 ∧ m - 3 ≠ 0) ↔ m = 0 := by
  sorry

end eq_is_quadratic_iff_m_zero_l170_170910


namespace equal_sum_seq_example_l170_170009

def EqualSumSeq (a : ℕ → ℕ) (c : ℕ) : Prop := ∀ n, a n + a (n + 1) = c

theorem equal_sum_seq_example (a : ℕ → ℕ) 
  (h1 : EqualSumSeq a 5) 
  (h2 : a 1 = 2) : a 6 = 3 :=
by 
  sorry

end equal_sum_seq_example_l170_170009


namespace avg_equivalence_l170_170387

-- Definition of binary average [a, b]
def avg2 (a b : ℤ) : ℤ := (a + b) / 2

-- Definition of ternary average {a, b, c}
def avg3 (a b c : ℤ) : ℤ := (a + b + c) / 3

-- Lean statement for proving the given problem
theorem avg_equivalence : avg3 (avg3 2 2 (-1)) (avg2 3 (-1)) 1 = 1 := by
  sorry

end avg_equivalence_l170_170387


namespace find_a_value_l170_170351

theorem find_a_value : 
  (∀ x, (3 * (x - 2) - 4 * (x - 5 / 4) = 0) ↔ ( ∃ a, ((2 * x - a) / 3 - (x - a) / 2 = x - 1) ∧ a = -11 )) := sorry

end find_a_value_l170_170351


namespace pairs_of_polygons_with_angle_difference_l170_170393

theorem pairs_of_polygons_with_angle_difference :
  ∃ (pairs : ℕ), pairs = 52 ∧ ∀ (n k : ℕ), n > k ∧ (360 / k - 360 / n = 1) :=
sorry

end pairs_of_polygons_with_angle_difference_l170_170393


namespace find_c_for_two_zeros_l170_170919

noncomputable def f (x c : ℝ) : ℝ := x^3 - 3*x + c

theorem find_c_for_two_zeros (c : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 c = 0 ∧ f x2 c = 0) ↔ c = -2 ∨ c = 2 :=
sorry

end find_c_for_two_zeros_l170_170919


namespace hyperbola_problem_l170_170658

theorem hyperbola_problem (s : ℝ) :
    (∃ b > 0, ∀ (x y : ℝ), (x, y) = (-4, 5) → (x^2 / 9) - (y^2 / b^2) = 1)
    ∧ (∀ (x y : ℝ), (x, y) = (-3, 0) → (x^2 / 9) - (y^2 / b^2) = 1)
    ∧ (∀ b > 0, (x, y) = (s, 3) → (x^2 / 9) - (7 * y^2 / 225) = 1)
    → s^2 = (288 / 25) :=
by
  sorry

end hyperbola_problem_l170_170658


namespace weather_on_july_15_l170_170521

theorem weather_on_july_15 
  (T: ℝ) (sunny: Prop) (W: ℝ) (crowded: Prop) 
  (h1: (T ≥ 85 ∧ sunny ∧ W < 15) → crowded) 
  (h2: ¬ crowded) : (T < 85 ∨ ¬ sunny ∨ W ≥ 15) :=
sorry

end weather_on_july_15_l170_170521


namespace stadium_length_in_yards_l170_170398

def length_in_feet := 183
def feet_per_yard := 3

theorem stadium_length_in_yards : length_in_feet / feet_per_yard = 61 := by
  sorry

end stadium_length_in_yards_l170_170398


namespace pascal_triangle_contains_53_once_l170_170119

theorem pascal_triangle_contains_53_once
  (h_prime : Nat.Prime 53) :
  ∃! n, ∃ k, n ≥ k ∧ n > 0 ∧ k > 0 ∧ Nat.choose n k = 53 := by
  sorry

end pascal_triangle_contains_53_once_l170_170119


namespace find_m_l170_170546

theorem find_m (x1 x2 m : ℝ)
  (h1 : ∀ x, x^2 - 4 * x + m = 0 → x = x1 ∨ x = x2)
  (h2 : x1 + x2 - x1 * x2 = 1) :
  m = 3 :=
sorry

end find_m_l170_170546


namespace number_of_married_men_at_least_11_l170_170913

-- Definitions based only on conditions from a)
def total_men := 100
def men_with_tv := 75
def men_with_radio := 85
def men_with_ac := 70
def married_with_tv_radio_ac := 11

-- Theorem that needs to be proven based on the conditions
theorem number_of_married_men_at_least_11 : total_men ≥ married_with_tv_radio_ac :=
by
  sorry

end number_of_married_men_at_least_11_l170_170913


namespace line_parallel_to_plane_line_perpendicular_to_plane_l170_170660

theorem line_parallel_to_plane (A B C D x1 y1 z1 m n p : ℝ) :
  A * m + B * n + C * p = 0 ↔ 
  ∀ x y z, ((A * x + B * y + C * z + D = 0) → 
  (∃ t, x = x1 + m * t ∧ y = y1 + n * t ∧ z = z1 + p * t)) :=
sorry

theorem line_perpendicular_to_plane (A B C D x1 y1 z1 m n p : ℝ) :
  (A / m = B / n ∧ B / n = C / p) ↔ 
  ∀ x y z, ((A * x + B * y + C * z + D = 0) → 
  (∃ t, x = x1 + m * t ∧ y = y1 + n * t ∧ z = z1 + p * t)) :=
sorry

end line_parallel_to_plane_line_perpendicular_to_plane_l170_170660


namespace both_participation_correct_l170_170604

-- Define the number of total participants
def total_participants : ℕ := 50

-- Define the number of participants in Chinese competition
def chinese_participants : ℕ := 30

-- Define the number of participants in Mathematics competition
def math_participants : ℕ := 38

-- Define the number of people who do not participate in either competition
def neither_participants : ℕ := 2

-- Define the number of people who participate in both competitions
def both_participants : ℕ :=
  chinese_participants + math_participants - (total_participants - neither_participants)

-- The theorem we want to prove
theorem both_participation_correct : both_participants = 20 :=
by
  sorry

end both_participation_correct_l170_170604


namespace positive_roots_of_x_pow_x_eq_one_over_sqrt_two_l170_170645

theorem positive_roots_of_x_pow_x_eq_one_over_sqrt_two (x : ℝ) (h : x > 0) : 
  (x^x = 1 / Real.sqrt 2) ↔ (x = 1 / 2 ∨ x = 1 / 4) := by
  sorry

end positive_roots_of_x_pow_x_eq_one_over_sqrt_two_l170_170645


namespace sum_first_seven_terms_of_arith_seq_l170_170770

-- Define an arithmetic sequence
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- Conditions: a_2 = 10 and a_5 = 1
def a_2 := 10
def a_5 := 1

-- The sum of the first 7 terms of the sequence
theorem sum_first_seven_terms_of_arith_seq (a d : ℤ) :
  arithmetic_seq a d 1 = a_2 →
  arithmetic_seq a d 4 = a_5 →
  (7 * a + (7 * 6 / 2) * d = 28) :=
by
  sorry

end sum_first_seven_terms_of_arith_seq_l170_170770


namespace find_a_for_even_function_l170_170477

theorem find_a_for_even_function :
  ∀ a : ℝ, (∀ x : ℝ, a * 3^x + 1 / 3^x = a * 3^(-x) + 1 / 3^(-x)) → a = 1 :=
by
  sorry

end find_a_for_even_function_l170_170477


namespace polygon_D_has_largest_area_l170_170856

noncomputable def area_A := 4 * 1 + 2 * (1 / 2) -- 5
noncomputable def area_B := 2 * 1 + 2 * (1 / 2) + Real.pi / 4 -- ≈ 3.785
noncomputable def area_C := 3 * 1 + 3 * (1 / 2) -- 4.5
noncomputable def area_D := 3 * 1 + 1 * (1 / 2) + 2 * (Real.pi / 4) -- ≈ 5.07
noncomputable def area_E := 1 * 1 + 3 * (1 / 2) + 3 * (Real.pi / 4) -- ≈ 4.855

theorem polygon_D_has_largest_area :
  area_D > area_A ∧
  area_D > area_B ∧
  area_D > area_C ∧
  area_D > area_E :=
by
  sorry

end polygon_D_has_largest_area_l170_170856


namespace inequality_solution_l170_170352

theorem inequality_solution (x : ℝ) : 2 * (3 * x - 2) > x + 1 ↔ x > 1 := by
  sorry

end inequality_solution_l170_170352


namespace tony_combined_lift_weight_l170_170013

noncomputable def tony_exercises :=
  let curl_weight := 90 -- pounds.
  let military_press_weight := 2 * curl_weight -- pounds.
  let squat_weight := 5 * military_press_weight -- pounds.
  let bench_press_weight := 1.5 * military_press_weight -- pounds.
  squat_weight + bench_press_weight

theorem tony_combined_lift_weight :
  tony_exercises = 1170 := by
  -- Here we will include the necessary proof steps
  sorry

end tony_combined_lift_weight_l170_170013


namespace min_value_inverse_sum_l170_170026

theorem min_value_inverse_sum (a m n : ℝ) (a_pos : 0 < a) (a_ne_one : a ≠ 1) (mn_pos : 0 < m * n) :
  (a^(1-1) = 1) ∧ (m + n = 1) → (1/m + 1/n) = 4 :=
by
  sorry

end min_value_inverse_sum_l170_170026


namespace pie_difference_l170_170818

theorem pie_difference (p1 p2 : ℚ) (h1 : p1 = 5 / 6) (h2 : p2 = 2 / 3) : p1 - p2 = 1 / 6 := 
by 
  sorry

end pie_difference_l170_170818


namespace sequence_an_correct_l170_170162

noncomputable def seq_an (n : ℕ) : ℚ :=
if h : n = 1 then 1 else (1 / (2 * n - 1) - 1 / (2 * n - 3))

theorem sequence_an_correct (n : ℕ) (S : ℕ → ℚ)
  (h1 : S 1 = 1)
  (h2 : ∀ n ≥ 2, S n ^ 2 = seq_an n * (S n - 0.5)) :
  seq_an n = if n = 1 then 1 else (1 / (2 * n - 1) - 1 / (2 * n - 3)) :=
sorry

end sequence_an_correct_l170_170162


namespace solve_equation_l170_170397

theorem solve_equation (x : ℝ) : 4 * (x - 1) ^ 2 = 9 ↔ x = 5 / 2 ∨ x = -1 / 2 := 
by 
  sorry

end solve_equation_l170_170397


namespace integer_solutions_count_l170_170462

theorem integer_solutions_count : 
  ∃ (S : Finset (ℤ × ℤ)), 
  (∀ x y, (x, y) ∈ S ↔ x^2 + x * y + 2 * y^2 = 29) ∧ 
  S.card = 4 := 
sorry

end integer_solutions_count_l170_170462


namespace meal_combinations_correct_l170_170715

-- Let E denote the total number of dishes on the menu
def E : ℕ := 12

-- Let V denote the number of vegetarian dishes on the menu
def V : ℕ := 5

-- Define the function that computes the number of different combinations of meals Elena and Nasir can order
def meal_combinations (e : ℕ) (v : ℕ) : ℕ :=
  e * v

-- The theorem to prove that the number of different combinations of meals Elena and Nasir can order is 60
theorem meal_combinations_correct : meal_combinations E V = 60 := by
  sorry

end meal_combinations_correct_l170_170715


namespace remainder_3_pow_2n_plus_8_l170_170946

theorem remainder_3_pow_2n_plus_8 (n : Nat) : (3 ^ (2 * n) + 8) % 8 = 1 := by
  sorry

end remainder_3_pow_2n_plus_8_l170_170946


namespace patrick_age_l170_170746

theorem patrick_age (r_age_future : ℕ) (years_future : ℕ) (half_age : ℕ → ℕ) 
  (h1 : r_age_future = 30) (h2 : years_future = 2) 
  (h3 : ∀ n, half_age n = n / 2) :
  half_age (r_age_future - years_future) = 14 :=
by
  sorry

end patrick_age_l170_170746


namespace max_imaginary_part_of_roots_l170_170135

noncomputable def find_phi : Prop :=
  ∃ z : ℂ, z^6 - z^4 + z^2 - 1 = 0 ∧ (∀ w : ℂ, w^6 - w^4 + w^2 - 1 = 0 → z.im ≤ w.im) ∧ z.im = Real.sin (Real.pi / 4)

theorem max_imaginary_part_of_roots : find_phi :=
sorry

end max_imaginary_part_of_roots_l170_170135


namespace min_value_of_fraction_l170_170470

theorem min_value_of_fraction (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : 2 * m + n = 1) : 
  (1 / m + 2 / n) = 8 := 
by 
  sorry

end min_value_of_fraction_l170_170470


namespace yoga_studio_women_count_l170_170365

theorem yoga_studio_women_count :
  ∃ W : ℕ, 
  (8 * 190) + (W * 120) = 14 * 160 ∧ W = 6 :=
by 
  existsi (6);
  sorry

end yoga_studio_women_count_l170_170365


namespace problem_part1_problem_part2_l170_170114

-- Statement for Part (1)
theorem problem_part1 (a b : ℝ) (h_sol : {x : ℝ | ax^2 - 3 * x + 2 > 0} = {x : ℝ | x < 1 ∨ x > 2}) :
  a = 1 ∧ b = 2 := sorry

-- Statement for Part (2)
theorem problem_part2 (x y k : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) 
  (h_eq : (1 / x) + (2 / y) = 1) (h_ineq : 2 * x + y ≥ k^2 + k + 2) :
  -3 ≤ k ∧ k ≤ 2 := sorry

end problem_part1_problem_part2_l170_170114


namespace cookies_eq_23_l170_170575

def total_packs : Nat := 27
def cakes : Nat := 4
def cookies : Nat := total_packs - cakes

theorem cookies_eq_23 : cookies = 23 :=
by
  -- Proof goes here
  sorry

end cookies_eq_23_l170_170575


namespace Q_finishes_in_6_hours_l170_170873

def Q_time_to_finish_job (T_Q : ℝ) : Prop :=
  let P_rate := 1 / 3
  let Q_rate := 1 / T_Q
  let work_together_2hr := 2 * (P_rate + Q_rate)
  let P_alone_work_40min := (2 / 3) * P_rate
  work_together_2hr + P_alone_work_40min = 1

theorem Q_finishes_in_6_hours : Q_time_to_finish_job 6 :=
  sorry -- Proof skipped

end Q_finishes_in_6_hours_l170_170873


namespace find_minimum_value_l170_170230

theorem find_minimum_value (a : ℝ) (x : ℝ) (h1 : a > 0) (h2 : x > 0): 
  (∃ m : ℝ, ∀ x > 0, (a^2 + x^2) / x ≥ m ∧ ∃ x₀ > 0, (a^2 + x₀^2) / x₀ = m) :=
sorry

end find_minimum_value_l170_170230


namespace g_of_2_l170_170537

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : x * g y = 2 * y * g x 
axiom g_of_10 : g 10 = 5

theorem g_of_2 : g 2 = 2 :=
by
    sorry

end g_of_2_l170_170537


namespace find_ac_and_area_l170_170779

variables {a b c : ℝ} {A B C : ℝ}
variables (triangle_abc : ∀ {a b c : ℝ} {A B C : ℝ}, a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π)
variables (h1 : (a ^ 2 + c ^ 2 - b ^ 2) / Real.cos B = 4)
variables (h2 : (2 * b * Real.cos C - 2 * c * Real.cos B) / (b * Real.cos C + c * Real.cos B) - c / a = 2)

noncomputable def ac_value := 2

noncomputable def area_of_triangle_abc := (Real.sqrt 15) / 4

theorem find_ac_and_area (triangle_abc : ∀ {a b c : ℝ} {A B C : ℝ}, a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π) 
                         (h1 : (a ^ 2 + c ^ 2 - b ^ 2) / Real.cos B = 4) 
                         (h2 : (2 * b * Real.cos C - 2 * c * Real.cos B) / (b * Real.cos C + c * Real.cos B) - c / a = 2):
  ac_value = 2 ∧
  area_of_triangle_abc = (Real.sqrt 15) / 4 := 
sorry

end find_ac_and_area_l170_170779


namespace number_of_camels_l170_170012

theorem number_of_camels (hens goats keepers camel_feet heads total_feet : ℕ)
  (h_hens : hens = 50) (h_goats : goats = 45) (h_keepers : keepers = 15)
  (h_feet_diff : total_feet = heads + 224)
  (h_heads : heads = hens + goats + keepers)
  (h_hens_feet : hens * 2 = 100)
  (h_goats_feet : goats * 4 = 180)
  (h_keepers_feet : keepers * 2 = 30)
  (h_camels_feet : camel_feet = 24)
  (h_total_feet : total_feet = 334)
  (h_feet_without_camels : 100 + 180 + 30 = 310) :
  camel_feet / 4 = 6 := sorry

end number_of_camels_l170_170012


namespace largest_angle_in_hexagon_l170_170490

theorem largest_angle_in_hexagon :
  ∀ (x : ℝ), (2 * x + 3 * x + 3 * x + 4 * x + 4 * x + 5 * x = 720) →
  5 * x = 1200 / 7 :=
by
  intros x h
  sorry

end largest_angle_in_hexagon_l170_170490


namespace range_of_expression_l170_170610

theorem range_of_expression (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y ≤ 1) : 
  ∃ (z : Set ℝ), z = Set.Icc (2 / 3) 4 ∧ (4*x^2 + 4*y^2 + (1 - x - y)^2) ∈ z :=
by
  sorry

end range_of_expression_l170_170610


namespace b_1001_value_l170_170767

theorem b_1001_value (b : ℕ → ℝ)
  (h1 : ∀ n ≥ 2, b n = b (n - 1) * b (n + 1)) 
  (h2 : b 1 = 3 + Real.sqrt 11)
  (h3 : b 888 = 17 + Real.sqrt 11) : 
  b 1001 = 7 * Real.sqrt 11 - 20 := sorry

end b_1001_value_l170_170767


namespace PQ_sum_l170_170752

-- Define the problem conditions
variable (P Q x : ℝ)
variable (h1 : (∀ x, x ≠ 3 → P / (x - 3) + Q * (x - 2) = (-4 * x^2 + 20 * x + 32) / (x - 3)))

-- Define the proof goal
theorem PQ_sum (h1 : (∀ x, x ≠ 3 → P / (x - 3) + Q * (x - 2) = (-4 * x^2 + 20 * x + 32) / (x - 3))) : P + Q = 52 :=
sorry

end PQ_sum_l170_170752


namespace american_literature_marks_l170_170565

variable (History HomeEconomics PhysicalEducation Art AverageMarks NumberOfSubjects TotalMarks KnownMarks : ℕ)
variable (A : ℕ)

axiom marks_history : History = 75
axiom marks_home_economics : HomeEconomics = 52
axiom marks_physical_education : PhysicalEducation = 68
axiom marks_art : Art = 89
axiom average_marks : AverageMarks = 70
axiom number_of_subjects : NumberOfSubjects = 5

def total_marks (AverageMarks NumberOfSubjects : ℕ) : ℕ := AverageMarks * NumberOfSubjects

def known_marks (History HomeEconomics PhysicalEducation Art : ℕ) : ℕ := History + HomeEconomics + PhysicalEducation + Art

axiom total_marks_eq : TotalMarks = total_marks AverageMarks NumberOfSubjects
axiom known_marks_eq : KnownMarks = known_marks History HomeEconomics PhysicalEducation Art

theorem american_literature_marks :
  A = TotalMarks - KnownMarks := by
  sorry

end american_literature_marks_l170_170565


namespace pq_false_implies_m_range_l170_170930

def p : Prop := ∀ x : ℝ, abs x + x ≥ 0

def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + m * x + 1 = 0

theorem pq_false_implies_m_range (m : ℝ) :
  (¬ (p ∧ q m)) → -2 < m ∧ m < 2 :=
by
  sorry

end pq_false_implies_m_range_l170_170930


namespace total_games_is_24_l170_170400

-- Definitions of conditions
def games_this_month : Nat := 9
def games_last_month : Nat := 8
def games_next_month : Nat := 7

-- Total games attended
def total_games_attended : Nat :=
  games_this_month + games_last_month + games_next_month

-- Problem statement
theorem total_games_is_24 : total_games_attended = 24 := by
  sorry

end total_games_is_24_l170_170400


namespace square_side_length_eq_area_and_perimeter_l170_170592

theorem square_side_length_eq_area_and_perimeter (a : ℝ) (h : a^2 = 4 * a) : a = 4 :=
by sorry

end square_side_length_eq_area_and_perimeter_l170_170592


namespace right_angled_triangle_count_in_pyramid_l170_170181

-- Define the cuboid and the triangular pyramid within it
variables (A B C D A₁ B₁ C₁ D₁ : Type)

-- Assume there exists a cuboid ABCD-A₁B₁C₁D₁
axiom cuboid : Prop

-- Define the triangular pyramid A₁-ABC
structure triangular_pyramid (A₁ A B C : Type) : Type :=
  (vertex₁ : A₁)
  (vertex₂ : A)
  (vertex₃ : B)
  (vertex4 : C)
  
-- The mathematical statement to prove: the number of right-angled triangles in A₁-ABC is 4
theorem right_angled_triangle_count_in_pyramid (A : Type) (B : Type) (C : Type) (A₁ : Type)
  (h_pyramid : triangular_pyramid A₁ A B C) (h_cuboid : cuboid) :
  ∃ n : ℕ, n = 4 :=
by
  sorry

end right_angled_triangle_count_in_pyramid_l170_170181


namespace sum_2_75_0_003_0_158_l170_170280

theorem sum_2_75_0_003_0_158 : 2.75 + 0.003 + 0.158 = 2.911 :=
by
  -- Lean proof goes here  
  sorry

end sum_2_75_0_003_0_158_l170_170280


namespace solve_for_y_in_terms_of_x_l170_170285

theorem solve_for_y_in_terms_of_x (x y : ℝ) (h : 2 * x - 7 * y = 5) : y = (2 * x - 5) / 7 :=
sorry

end solve_for_y_in_terms_of_x_l170_170285


namespace range_of_expressions_l170_170293

theorem range_of_expressions (x y : ℝ) (h1 : 30 < x ∧ x < 42) (h2 : 16 < y ∧ y < 24) :
  46 < x + y ∧ x + y < 66 ∧ -18 < x - 2 * y ∧ x - 2 * y < 10 ∧ (5 / 4) < (x / y) ∧ (x / y) < (21 / 8) :=
sorry

end range_of_expressions_l170_170293


namespace greatest_possible_red_points_l170_170439

theorem greatest_possible_red_points (R B : ℕ) (h1 : R + B = 25)
    (h2 : ∀ r1 r2, r1 < R → r2 < R → r1 ≠ r2 → ∃ (n : ℕ), (∃ b1 : ℕ, b1 < B) ∧ ¬∃ b2 : ℕ, b2 < B) :
  R ≤ 13 :=
by {
  sorry
}

end greatest_possible_red_points_l170_170439


namespace gcd_of_three_numbers_l170_170855

theorem gcd_of_three_numbers : 
  let a := 4560
  let b := 6080
  let c := 16560
  gcd (gcd a b) c = 80 := 
by {
  -- placeholder for the proof
  sorry
}

end gcd_of_three_numbers_l170_170855


namespace graph_of_eq_hyperbola_l170_170605

theorem graph_of_eq_hyperbola (x y : ℝ) : (x + y)^2 = x^2 + y^2 + 1 → ∃ a b : ℝ, a * b = x * y ∧ a * b = 1/2 := by
  sorry

end graph_of_eq_hyperbola_l170_170605


namespace height_of_table_l170_170934

variable (h l w h3 : ℝ)

-- Conditions from the problem
def condition1 : Prop := h3 = 4
def configurationA : Prop := l + h - w = 50
def configurationB : Prop := w + h + h3 - l = 44

-- Statement to prove
theorem height_of_table (h l w h3 : ℝ) 
  (cond1 : condition1 h3)
  (confA : configurationA h l w)
  (confB : configurationB h l w h3) : 
  h = 45 := 
by 
  sorry

end height_of_table_l170_170934


namespace value_of_x_l170_170243

theorem value_of_x : ∃ (x : ℚ), (10 - 2 * x) ^ 2 = 4 * x ^ 2 + 20 * x ∧ x = 5 / 3 :=
by
  sorry

end value_of_x_l170_170243


namespace least_positive_integer_k_l170_170240

noncomputable def least_k (a : ℝ) (n : ℕ) : ℝ :=
  (1 : ℝ) / ((n + 1 : ℝ) ^ 3)

theorem least_positive_integer_k :
  ∃ k : ℕ , (∀ a : ℝ, ∀ n : ℕ,
  (0 ≤ a ∧ a ≤ 1) → (a^k * (1 - a)^n < least_k a n)) ∧
  (∀ k' : ℕ, k' < 4 → ¬(∀ a : ℝ, ∀ n : ℕ, (0 ≤ a ∧ a ≤ 1) → (a^k' * (1 - a)^n < least_k a n))) :=
sorry

end least_positive_integer_k_l170_170240


namespace ball_more_than_bat_l170_170874

theorem ball_more_than_bat :
  ∃ x y : ℕ, (2 * x + 3 * y = 1300) ∧ (3 * x + 2 * y = 1200) ∧ (y - x = 100) :=
by
  sorry

end ball_more_than_bat_l170_170874


namespace algebraic_expression_value_l170_170686

variable {R : Type} [CommRing R]

theorem algebraic_expression_value (m n : R) (h1 : m - n = -2) (h2 : m * n = 3) :
  -m^3 * n + 2 * m^2 * n^2 - m * n^3 = -12 :=
sorry

end algebraic_expression_value_l170_170686


namespace little_twelve_conference_games_l170_170609

def teams_in_division : ℕ := 6
def divisions : ℕ :=  2

def games_within_division (t : ℕ) : ℕ := (t * (t - 1)) / 2 * 2

def games_between_divisions (d t : ℕ) : ℕ := t * t

def total_conference_games (d t : ℕ) : ℕ :=
  d * games_within_division t + games_between_divisions d t

theorem little_twelve_conference_games :
  total_conference_games divisions teams_in_division = 96 :=
by
  sorry

end little_twelve_conference_games_l170_170609


namespace common_number_exists_l170_170792

def sum_of_list (l : List ℚ) : ℚ := l.sum

theorem common_number_exists (l1 l2 : List ℚ) (commonNumber : ℚ) 
    (h1 : l1.length = 5) 
    (h2 : l2.length = 5) 
    (h3 : sum_of_list l1 / 5 = 7) 
    (h4 : sum_of_list l2 / 5 = 10) 
    (h5 : (sum_of_list l1 + sum_of_list l2 - commonNumber) / 9 = 74 / 9) 
    : commonNumber = 11 :=
sorry

end common_number_exists_l170_170792


namespace major_premise_is_false_l170_170678

-- Define the major premise
def major_premise (a : ℝ) : Prop := a^2 > 0

-- Define the minor premise
def minor_premise (a : ℝ) := true

-- Define the conclusion based on the premises
def conclusion (a : ℝ) : Prop := a^2 > 0

-- Show that the major premise is false by finding a counterexample
theorem major_premise_is_false : ¬ ∀ a : ℝ, major_premise a := by
  sorry

end major_premise_is_false_l170_170678


namespace neg_fraction_comparison_l170_170071

theorem neg_fraction_comparison : - (4 / 5 : ℝ) > - (5 / 6 : ℝ) :=
by {
  -- sorry to skip the proof
  sorry
}

end neg_fraction_comparison_l170_170071


namespace adam_has_10_apples_l170_170226

theorem adam_has_10_apples
  (Jackie_has_2_apples : ∀ Jackie_apples, Jackie_apples = 2)
  (Adam_has_8_more_apples : ∀ Adam_apples Jackie_apples, Adam_apples = Jackie_apples + 8)
  : ∀ Adam_apples, Adam_apples = 10 :=
by {
  sorry
}

end adam_has_10_apples_l170_170226


namespace negation_universal_proposition_l170_170589

theorem negation_universal_proposition :
  (¬ ∀ x : ℝ, 0 < x → x^2 + x + 1 > 0) ↔ (∃ x : ℝ, 0 < x ∧ x^2 + x + 1 ≤ 0) :=
by
  sorry

end negation_universal_proposition_l170_170589


namespace daves_initial_apps_l170_170536

theorem daves_initial_apps : ∃ (X : ℕ), X + 11 - 17 = 4 ∧ X = 10 :=
by {
  sorry
}

end daves_initial_apps_l170_170536


namespace isosceles_triangle_vertex_angle_l170_170158

theorem isosceles_triangle_vertex_angle (exterior_angle : ℝ) (h1 : exterior_angle = 40) : 
  ∃ vertex_angle : ℝ, vertex_angle = 140 :=
by
  sorry

end isosceles_triangle_vertex_angle_l170_170158


namespace f_le_one_l170_170939

open Real

theorem f_le_one (x : ℝ) (hx : 0 < x) : (1 + log x) / x ≤ 1 := 
sorry

end f_le_one_l170_170939


namespace tan_two_beta_l170_170333

variables {α β : Real}

theorem tan_two_beta (h1 : Real.tan (α + β) = 1) (h2 : Real.tan (α - β) = 7) : Real.tan (2 * β) = -3 / 4 :=
by
  sorry

end tan_two_beta_l170_170333


namespace solution_set_of_inequality_l170_170953

theorem solution_set_of_inequality (x : ℝ) (h : 2 * x + 3 ≤ 1) : x ≤ -1 :=
sorry

end solution_set_of_inequality_l170_170953


namespace sum_ratio_is_nine_l170_170722

open Nat

-- Predicate to define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Sum of the first n terms of an arithmetic sequence
noncomputable def S (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  (n * (a 0 + a (n - 1))) / 2

axiom a : ℕ → ℝ -- The arithmetic sequence
axiom h_arith : is_arithmetic_sequence a
axiom a5_eq_5a3 : a 4 = 5 * a 2

-- Statement of the problem
theorem sum_ratio_is_nine : S 9 a / S 5 a = 9 :=
sorry

end sum_ratio_is_nine_l170_170722


namespace total_missed_questions_l170_170414

-- Definitions
def missed_by_you : ℕ := 36
def missed_by_friend : ℕ := 7
def missed_by_you_friends : ℕ := missed_by_you + missed_by_friend

-- Theorem
theorem total_missed_questions (h1 : missed_by_you = 5 * missed_by_friend) :
  missed_by_you_friends = 43 :=
by
  sorry

end total_missed_questions_l170_170414


namespace shaded_ratio_l170_170580

theorem shaded_ratio (full_rectangles half_rectangles : ℕ) (n m : ℕ) (rectangle_area shaded_area total_area : ℝ)
  (h1 : n = 4) (h2 : m = 5) (h3 : rectangle_area = n * m) 
  (h4 : full_rectangles = 3) (h5 : half_rectangles = 4)
  (h6 : shaded_area = full_rectangles * 1 + 0.5 * half_rectangles * 1)
  (h7 : total_area = rectangle_area) :
  shaded_area / total_area = 1 / 4 := by
  sorry

end shaded_ratio_l170_170580


namespace marked_price_l170_170516

theorem marked_price (original_price : ℝ) 
                     (discount1_rate : ℝ) 
                     (profit_rate : ℝ) 
                     (discount2_rate : ℝ)
                     (marked_price : ℝ) : 
                     original_price = 40 → 
                     discount1_rate = 0.15 → 
                     profit_rate = 0.25 → 
                     discount2_rate = 0.10 → 
                     marked_price = 47.20 := 
by
  intros h₁ h₂ h₃ h₄
  sorry

end marked_price_l170_170516


namespace find_a_c_l170_170131

theorem find_a_c (a c : ℝ) (h_discriminant : ∀ x : ℝ, a * x^2 + 10 * x + c = 0 → ∃ k : ℝ, a * k^2 + 10 * k + c = 0 ∧ (a * x^2 + 10 * k + c = 0 → x = k))
  (h_sum : a + c = 12) (h_lt : a < c) : (a, c) = (6 - Real.sqrt 11, 6 + Real.sqrt 11) :=
sorry

end find_a_c_l170_170131


namespace tan_x_plus_pi_over_4_l170_170891

theorem tan_x_plus_pi_over_4 (x : ℝ) (hx : Real.tan x = 2) : Real.tan (x + Real.pi / 4) = -3 :=
by
  sorry

end tan_x_plus_pi_over_4_l170_170891


namespace Sabrina_pencils_l170_170037

variable (S : ℕ) (J : ℕ)

theorem Sabrina_pencils (h1 : S + J = 50) (h2 : J = 2 * S + 8) :
  S = 14 :=
by
  sorry

end Sabrina_pencils_l170_170037


namespace existence_of_five_regular_polyhedra_l170_170274

def regular_polyhedron (n m : ℕ) : Prop :=
  n ≥ 3 ∧ m ≥ 3 ∧ (2 / m + 2 / n > 1)

theorem existence_of_five_regular_polyhedra :
  ∃ (n m : ℕ), regular_polyhedron n m → 
    (n = 3 ∧ m = 3 ∨ 
     n = 4 ∧ m = 3 ∨ 
     n = 3 ∧ m = 4 ∨ 
     n = 5 ∧ m = 3 ∨ 
     n = 3 ∧ m = 5) :=
by
  sorry

end existence_of_five_regular_polyhedra_l170_170274


namespace hyperbola_slope_product_l170_170912

open Real

theorem hyperbola_slope_product
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (h : ∀ {x y : ℝ}, x ≠ 0 → (x^2 / a^2 - y^2 / b^2 = 1) → 
    ∀ {k1 k2 : ℝ}, (x = 0 ∨ y = 0) → (k1 * k2 = ((b^2) / (a^2)))) :
  (b^2 / a^2 = 3) :=
by 
  sorry

end hyperbola_slope_product_l170_170912


namespace arithmetic_mean_equality_l170_170340

variable (x y a b : ℝ)

theorem arithmetic_mean_equality (hx : x ≠ 0) (hy : y ≠ 0) :
  (1 / 2 * ((x + a) / y + (y - b) / x)) = (x^2 + a * x + y^2 - b * y) / (2 * x * y) :=
  sorry

end arithmetic_mean_equality_l170_170340


namespace alice_and_bob_pies_l170_170662

theorem alice_and_bob_pies (T : ℝ) : (T / 5 = T / 6 + 2) → T = 60 := by
  sorry

end alice_and_bob_pies_l170_170662


namespace value_of_s_l170_170789

-- Define the variables as integers (they represent non-zero digits)
variables {a p v e s r : ℕ}

-- Define the conditions as hypotheses
theorem value_of_s (h1 : a + p = v) (h2 : v + e = s) (h3 : s + a = r) (h4 : p + e + r = 14) :
  s = 7 :=
by
  sorry

end value_of_s_l170_170789


namespace city_a_location_l170_170788

theorem city_a_location (ϕ A_latitude : ℝ) (m : ℝ) (h_eq_height : true)
  (h_shadows_3x : true) 
  (h_angle: true) (h_southern : A_latitude < 0) 
  (h_rad_lat : ϕ = abs A_latitude):

  ϕ = 45 ∨ ϕ = 7.14 :=
by 
  sorry

end city_a_location_l170_170788


namespace number_of_subsets_of_three_element_set_l170_170126

theorem number_of_subsets_of_three_element_set :
  ∃ (S : Finset ℕ), S.card = 3 ∧ S.powerset.card = 8 :=
sorry

end number_of_subsets_of_three_element_set_l170_170126


namespace three_term_arithmetic_seq_l170_170382

noncomputable def arithmetic_sequence_squares (x y z : ℤ) : Prop :=
  x^2 + z^2 = 2 * y^2

theorem three_term_arithmetic_seq (x y z : ℤ) :
  (∃ a b : ℤ, a = (x + z) / 2 ∧ b = (x - z) / 2 ∧ x^2 + z^2 = 2 * y^2) ↔
  arithmetic_sequence_squares x y z :=
by
  sorry

end three_term_arithmetic_seq_l170_170382


namespace price_equivalence_l170_170775

theorem price_equivalence : 
  (∀ a o p : ℕ, 10 * a = 5 * o ∧ 4 * o = 6 * p) → 
  (∀ a o p : ℕ, 20 * a = 15 * p) :=
by
  intro h
  sorry

end price_equivalence_l170_170775


namespace original_cost_price_l170_170410

theorem original_cost_price ( C S : ℝ )
  (h1 : S = 1.05 * C)
  (h2 : S - 3 = 1.10 * 0.95 * C)
  : C = 600 :=
sorry

end original_cost_price_l170_170410


namespace f_3_neg3div2_l170_170010

noncomputable def f : ℝ → ℝ :=
sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom symm_f : ∀ t : ℝ, f t = f (1 - t)
axiom restriction_f : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1/2 → f x = -x^2

theorem f_3_neg3div2 :
  f 3 + f (-3/2) = -1/4 :=
sorry

end f_3_neg3div2_l170_170010


namespace rice_mixture_ratio_l170_170482

theorem rice_mixture_ratio (x y z : ℕ) (h : 16 * x + 24 * y + 30 * z = 18 * (x + y + z)) : 
  x = 9 * y + 18 * z :=
by
  sorry

end rice_mixture_ratio_l170_170482


namespace range_of_a_l170_170861

noncomputable def f (x a : ℝ) := Real.exp (-x) - 2 * x - a

def curve (x : ℝ) := x ^ 3 + x

def y_in_range (x : ℝ) := x >= -2 ∧ x <= 2

theorem range_of_a : ∀ (a : ℝ), (∃ x, y_in_range (curve x) ∧ f (curve x) a = curve x) ↔ a ∈ Set.Icc (Real.exp (-2) - 6) (Real.exp 2 + 6) := by
  sorry

end range_of_a_l170_170861


namespace janet_savings_l170_170021

def wall1_area := 5 * 8 -- wall 1 area
def wall2_area := 7 * 8 -- wall 2 area
def wall3_area := 6 * 9 -- wall 3 area
def total_area := wall1_area + wall2_area + wall3_area
def tiles_per_square_foot := 4
def total_tiles := total_area * tiles_per_square_foot

def turquoise_tile_cost := 13
def turquoise_labor_cost := 6
def total_cost_turquoise := (total_tiles * turquoise_tile_cost) + (total_area * turquoise_labor_cost)

def purple_tile_cost := 11
def purple_labor_cost := 8
def total_cost_purple := (total_tiles * purple_tile_cost) + (total_area * purple_labor_cost)

def orange_tile_cost := 15
def orange_labor_cost := 5
def total_cost_orange := (total_tiles * orange_tile_cost) + (total_area * orange_labor_cost)

def least_expensive_option := total_cost_purple
def most_expensive_option := total_cost_orange

def savings := most_expensive_option - least_expensive_option

theorem janet_savings : savings = 1950 := by
  sorry

end janet_savings_l170_170021


namespace functional_equation_solution_l170_170630

noncomputable def f : ℝ → ℝ := sorry 

theorem functional_equation_solution :
  (∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2) →
  10 * f 2006 + f 0 = 20071 :=
by
  intros h
  sorry

end functional_equation_solution_l170_170630


namespace intersecting_points_of_curves_l170_170325

theorem intersecting_points_of_curves :
  (∀ x y, (y = 2 * x^3 + x^2 - 5 * x + 2) ∧ (y = 3 * x^2 + 6 * x - 4) → 
   (x = -1 ∧ y = -7) ∨ (x = 3 ∧ y = 41)) := sorry

end intersecting_points_of_curves_l170_170325


namespace fourth_student_guess_l170_170629

theorem fourth_student_guess :
  let first_guess := 100
  let second_guess := 8 * first_guess
  let third_guess := second_guess - 200
  let total := first_guess + second_guess + third_guess
  let average := total / 3
  let fourth_guess := average + 25
  fourth_guess = 525 :=
by
  sorry

end fourth_student_guess_l170_170629


namespace conditions_for_a_and_b_l170_170168

variables (a b x y : ℝ)

theorem conditions_for_a_and_b (h1 : x^2 + x * y + y^2 - y = 0) (h2 : a * x^2 + b * x * y + x = 0) :
  (a + 1)^2 = 4 * (b + 1) ∧ b ≠ -1 :=
sorry

end conditions_for_a_and_b_l170_170168


namespace problem_sum_of_pairwise_prime_product_l170_170981

theorem problem_sum_of_pairwise_prime_product:
  ∃ a b c d: ℕ, a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1 ∧
  a * b * c * d = 288000 ∧
  gcd a b = 1 ∧ gcd a c = 1 ∧ gcd a d = 1 ∧
  gcd b c = 1 ∧ gcd b d = 1 ∧ gcd c d = 1 ∧
  a + b + c + d = 390 :=
sorry

end problem_sum_of_pairwise_prime_product_l170_170981


namespace distance_swim_against_current_l170_170479

-- Definitions based on problem conditions
def swimmer_speed_still_water : ℝ := 4 -- km/h
def water_current_speed : ℝ := 1 -- km/h
def time_swimming_against_current : ℝ := 2 -- hours

-- Calculation of effective speed against the current
def effective_speed_against_current : ℝ :=
  swimmer_speed_still_water - water_current_speed

-- Proof statement
theorem distance_swim_against_current :
  effective_speed_against_current * time_swimming_against_current = 6 :=
by
  -- By substituting values from the problem,
  -- effective_speed_against_current * time_swimming_against_current = 3 * 2
  -- which equals 6.
  sorry

end distance_swim_against_current_l170_170479


namespace linear_in_one_variable_linear_in_two_variables_l170_170579

namespace MathProof

-- Definition of the equation
def equation (k x y : ℝ) : ℝ := (k^2 - 1) * x^2 + (k + 1) * x + (k - 7) * y - (k + 2)

-- Theorem for linear equation in one variable
theorem linear_in_one_variable (k : ℝ) (x y : ℝ) :
  k = -1 → equation k x y = 0 → ∃ y' : ℝ, equation k 0 y' = 0 :=
by
  sorry

-- Theorem for linear equation in two variables
theorem linear_in_two_variables (k : ℝ) (x y : ℝ) :
  k = 1 → equation k x y = 0 → ∃ x' y' : ℝ, equation k x' y' = 0 :=
by
  sorry

end MathProof

end linear_in_one_variable_linear_in_two_variables_l170_170579


namespace simple_interest_rate_l170_170743

theorem simple_interest_rate :
  ∀ (P R : ℝ), 
  (R * 25 / 100 = 1) → 
  R = 4 := 
by
  intros P R h
  sorry

end simple_interest_rate_l170_170743


namespace dreamCarCost_l170_170458

-- Definitions based on given conditions
def monthlyEarnings : ℕ := 4000
def monthlySavings : ℕ := 500
def totalEarnings : ℕ := 360000

-- Theorem stating the desired result
theorem dreamCarCost :
  (totalEarnings / monthlyEarnings) * monthlySavings = 45000 :=
by
  sorry

end dreamCarCost_l170_170458


namespace triangle_existence_l170_170281

theorem triangle_existence (n : ℕ) (h : 2 * n > 0) (segments : Finset (ℕ × ℕ))
  (h_segments : segments.card = n^2 + 1)
  (points_in_segment : ∀ {a b : ℕ}, (a, b) ∈ segments → a < 2 * n ∧ b < 2 * n) :
  ∃ x y z, x < 2 * n ∧ y < 2 * n ∧ z < 2 * n ∧ (x ≠ y ∧ y ≠ z ∧ z ≠ x) ∧
  ((x, y) ∈ segments ∨ (y, x) ∈ segments) ∧
  ((y, z) ∈ segments ∨ (z, y) ∈ segments) ∧
  ((z, x) ∈ segments ∨ (x, z) ∈ segments) :=
by
  sorry

end triangle_existence_l170_170281


namespace horizontal_asymptote_at_3_l170_170616

noncomputable def rational_function (x : ℝ) : ℝ :=
  (15 * x^4 + 2 * x^3 + 11 * x^2 + 6 * x + 4) / (5 * x^4 + x^3 + 10 * x^2 + 4 * x + 2)

theorem horizontal_asymptote_at_3 : 
  (∀ ε > 0, ∃ N > 0, ∀ x > N, |rational_function x - 3| < ε) := 
by
  sorry

end horizontal_asymptote_at_3_l170_170616


namespace numBills_is_9_l170_170763

-- Define the conditions: Mike has 45 dollars in 5-dollar bills
def totalDollars : ℕ := 45
def billValue : ℕ := 5
def numBills : ℕ := 9

-- Prove that the number of 5-dollar bills Mike has is 9
theorem numBills_is_9 : (totalDollars = billValue * numBills) → (numBills = 9) :=
by
  intro h
  sorry

end numBills_is_9_l170_170763


namespace rank_identity_l170_170402

theorem rank_identity (n p : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) 
  (h1: 2 ≤ n) (h2: 2 ≤ p) (h3: A^(p+1) = A) : 
  Matrix.rank A + Matrix.rank (1 - A^p) = n := 
  sorry

end rank_identity_l170_170402


namespace colby_mango_sales_l170_170829

theorem colby_mango_sales
  (total_kg : ℕ)
  (mangoes_per_kg : ℕ)
  (remaining_mangoes : ℕ)
  (half_sold_to_market : ℕ) :
  total_kg = 60 →
  mangoes_per_kg = 8 →
  remaining_mangoes = 160 →
  half_sold_to_market = 20 := by
    sorry

end colby_mango_sales_l170_170829


namespace fifth_term_of_arithmetic_sequence_is_minus_three_l170_170538

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d

theorem fifth_term_of_arithmetic_sequence_is_minus_three (a d : ℤ) :
  (arithmetic_sequence a d 11 = 25) ∧ (arithmetic_sequence a d 12 = 29) →
  (arithmetic_sequence a d 4 = -3) :=
by 
  intros h
  sorry

end fifth_term_of_arithmetic_sequence_is_minus_three_l170_170538


namespace jeans_price_increase_l170_170093

theorem jeans_price_increase (manufacturing_cost customer_price : ℝ) 
  (h1 : customer_price = 1.40 * (1.40 * manufacturing_cost))
  : (customer_price - manufacturing_cost) / manufacturing_cost * 100 = 96 :=
by sorry

end jeans_price_increase_l170_170093


namespace evaluate_expression_l170_170249

theorem evaluate_expression :
  - (18 / 3 * 8 - 70 + 5 * 7) = -13 := by
  sorry

end evaluate_expression_l170_170249


namespace proof_problem1_proof_problem2_proof_problem3_l170_170376

-- Definition of the three mathematical problems
def problem1 : Prop := 8 / (-2) - (-4) * (-3) = -16

def problem2 : Prop := -2^3 + (-3) * ((-2)^3 + 5) = 1

def problem3 (x : ℝ) : Prop := (2 * x^2)^3 * x^2 - x^10 / x^2 = 7 * x^8

-- Statements of the proofs required
theorem proof_problem1 : problem1 :=
by sorry

theorem proof_problem2 : problem2 :=
by sorry

theorem proof_problem3 (x : ℝ) : problem3 x :=
by sorry

end proof_problem1_proof_problem2_proof_problem3_l170_170376


namespace inequality_solution_l170_170721

theorem inequality_solution (z : ℝ) : 
  z^2 - 40 * z + 400 ≤ 36 ↔ 14 ≤ z ∧ z ≤ 26 :=
by
  sorry

end inequality_solution_l170_170721


namespace twelve_people_pairing_l170_170175

noncomputable def num_ways_to_pair : ℕ := sorry

theorem twelve_people_pairing :
  (∀ (n : ℕ), n = 12 → (∃ f : ℕ → ℕ, ∀ i, f i = 2 ∨ f i = 12 ∨ f i = 7) → num_ways_to_pair = 3) := 
sorry

end twelve_people_pairing_l170_170175


namespace inverse_function_shift_l170_170121

-- Conditions
variable {f : ℝ → ℝ} {f_inv : ℝ → ℝ}
variable (hf : ∀ x : ℝ, f_inv (f x) = x ∧ f (f_inv x) = x)
variable (point_B : f 3 = -1)

-- Proof statement
theorem inverse_function_shift :
  f_inv (-3 + 2) = 3 :=
by
  -- Proof goes here
  sorry

end inverse_function_shift_l170_170121


namespace second_candy_cost_l170_170388

theorem second_candy_cost 
  (C : ℝ) 
  (hp := 25 * 8 + 50 * C = 75 * 6) : 
  C = 5 := 
  sorry

end second_candy_cost_l170_170388


namespace original_profit_percentage_l170_170535

theorem original_profit_percentage
  (C : ℝ) -- original cost
  (S : ℝ) -- selling price
  (y : ℝ) -- original profit percentage
  (hS : S = C * (1 + 0.01 * y)) -- condition for selling price based on original cost
  (hC' : S = 0.85 * C * (1 + 0.01 * (y + 20))) -- condition for selling price based on reduced cost
  : y = -89 :=
by
  sorry

end original_profit_percentage_l170_170535


namespace problem1_l170_170988

theorem problem1
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ)
  (h₁ : (3*x - 2)^(6) = a₀ + a₁ * (2*x - 1) + a₂ * (2*x - 1)^2 + a₃ * (2*x - 1)^3 + a₄ * (2*x - 1)^4 + a₅ * (2*x - 1)^5 + a₆ * (2*x - 1)^6)
  (h₂ : a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 1)
  (h₃ : a₀ - a₁ + a₂ - a₃ + a₄ - a₅ + a₆ = 64) :
  (a₁ + a₃ + a₅) / (a₀ + a₂ + a₄ + a₆) = -63 / 65 := by
  sorry

end problem1_l170_170988


namespace tom_average_speed_l170_170057

theorem tom_average_speed 
  (d1 d2 : ℝ) (s1 s2 t1 t2 : ℝ)
  (h_d1 : d1 = 30) 
  (h_d2 : d2 = 50) 
  (h_s1 : s1 = 30) 
  (h_s2 : s2 = 50) 
  (h_t1 : t1 = d1 / s1) 
  (h_t2 : t2 = d2 / s2)
  (h_total_distance : d1 + d2 = 80) 
  (h_total_time : t1 + t2 = 2) :
  (d1 + d2) / (t1 + t2) = 40 := 
by {
  sorry
}

end tom_average_speed_l170_170057


namespace mo_tea_cups_l170_170445

theorem mo_tea_cups (n t : ℤ) (h1 : 4 * n + 3 * t = 22) (h2 : 3 * t = 4 * n + 8) : t = 5 :=
by
  -- proof steps
  sorry

end mo_tea_cups_l170_170445


namespace largest_common_remainder_l170_170282

theorem largest_common_remainder : 
  ∃ n r, 2013 ≤ n ∧ n ≤ 2156 ∧ (n % 5 = r) ∧ (n % 11 = r) ∧ (n % 13 = r) ∧ (r = 4) := 
by
  sorry

end largest_common_remainder_l170_170282


namespace book_profit_percentage_l170_170143

noncomputable def profit_percentage (cost_price marked_price : ℝ) (discount_rate : ℝ) : ℝ :=
  let discount := discount_rate / 100 * marked_price
  let selling_price := marked_price - discount
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

theorem book_profit_percentage :
  profit_percentage 47.50 69.85 15 = 24.994736842105263 :=
by
  sorry

end book_profit_percentage_l170_170143


namespace multiply_by_15_is_225_l170_170795

-- Define the condition
def number : ℕ := 15

-- State the theorem with the conditions and the expected result
theorem multiply_by_15_is_225 : 15 * number = 225 := by
  -- Insert the proof here
  sorry

end multiply_by_15_is_225_l170_170795


namespace no_intersection_tangent_graph_l170_170429

theorem no_intersection_tangent_graph (k : ℝ) (m : ℤ) : 
  (∀ x: ℝ, x = (k * Real.pi) / 2 → (¬ 4 * k ≠ 4 * m + 1)) → 
  (-1 ≤ k ∧ k ≤ 1) →
  (k = 1 / 4 ∨ k = -3 / 4) :=
sorry

end no_intersection_tangent_graph_l170_170429


namespace sector_area_l170_170596

/--
The area of a sector with radius 6cm and central angle 15° is (3 * π / 2) cm².
-/
theorem sector_area (R : ℝ) (θ : ℝ) (h_radius : R = 6) (h_angle : θ = 15) :
    (S : ℝ) = (3 * Real.pi / 2) := by
  sorry

end sector_area_l170_170596


namespace probability_same_color_l170_170656

/-- Define the number of green plates. -/
def green_plates : ℕ := 7

/-- Define the number of yellow plates. -/
def yellow_plates : ℕ := 5

/-- Define the total number of plates. -/
def total_plates : ℕ := green_plates + yellow_plates

/-- Calculate the binomial coefficient for choosing k items from a set of n items. -/
def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- Prove that the probability of selecting three plates of the same color is 9/44. -/
theorem probability_same_color :
  (binomial_coeff green_plates 3 + binomial_coeff yellow_plates 3) / binomial_coeff total_plates 3 = 9 / 44 :=
by
  sorry

end probability_same_color_l170_170656


namespace probability_of_C_l170_170748

theorem probability_of_C (P_A P_B P_C P_D P_E : ℚ)
  (hA : P_A = 2/5)
  (hB : P_B = 1/5)
  (hCD : P_C = P_D)
  (hE : P_E = 2 * P_C)
  (h_total : P_A + P_B + P_C + P_D + P_E = 1) : P_C = 1/10 :=
by
  -- To prove this theorem, you will use the conditions provided in the hypotheses.
  -- Here's how you start the proof:
  sorry

end probability_of_C_l170_170748


namespace double_24_times_10_pow_8_l170_170626

theorem double_24_times_10_pow_8 : 2 * (2.4 * 10^8) = 4.8 * 10^8 :=
by
  sorry

end double_24_times_10_pow_8_l170_170626


namespace donald_juice_l170_170968

variable (P D : ℕ)

theorem donald_juice (h1 : P = 3) (h2 : D = 2 * P + 3) : D = 9 := by
  sorry

end donald_juice_l170_170968


namespace percent_motorists_no_ticket_l170_170125

theorem percent_motorists_no_ticket (M : ℝ) :
  (0.14285714285714285 * M - 0.10 * M) / (0.14285714285714285 * M) * 100 = 30 :=
by
  sorry

end percent_motorists_no_ticket_l170_170125


namespace surface_area_of_sphere_containing_prism_l170_170377

-- Assume the necessary geometric context and definitions are available.
def rightSquarePrism (a h : ℝ) (V : ℝ) := 
  a^2 * h = V

theorem surface_area_of_sphere_containing_prism 
  (a h V : ℝ) (S : ℝ) (π := Real.pi)
  (prism_on_sphere : ∀ (prism : rightSquarePrism a h V), True)
  (height_eq_4 : h = 4) 
  (volume_eq_16 : V = 16) :
  S = 4 * π * 24 :=
by
  -- proof steps would go here
  sorry

end surface_area_of_sphere_containing_prism_l170_170377


namespace equal_product_groups_exist_l170_170924

def numbers : List ℕ := [21, 22, 34, 39, 44, 45, 65, 76, 133, 153]

theorem equal_product_groups_exist :
  ∃ (g1 g2 : List ℕ), 
    g1.length = 5 ∧ g2.length = 5 ∧ 
    g1.prod = g2.prod ∧ g1.prod = 349188840 ∧ 
    (g1 ++ g2 = numbers ∨ g1 ++ g2 = numbers.reverse) :=
by
  sorry

end equal_product_groups_exist_l170_170924


namespace no_real_roots_range_l170_170588

theorem no_real_roots_range (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + 1 ≠ 0) ↔ (-2 < m ∧ m < 2) :=
by
  sorry

end no_real_roots_range_l170_170588


namespace initial_marbles_l170_170863

theorem initial_marbles (M : ℝ) (h0 : 0.2 * M + 0.35 * (0.8 * M) + 130 = M) : M = 250 :=
by
  sorry

end initial_marbles_l170_170863


namespace set_A_roster_l170_170118

def is_nat_not_greater_than_4 (x : ℕ) : Prop := x ≤ 4

def A : Set ℕ := {x | is_nat_not_greater_than_4 x}

theorem set_A_roster : A = {0, 1, 2, 3, 4} := by
  sorry

end set_A_roster_l170_170118


namespace scientific_notation_448000_l170_170878

theorem scientific_notation_448000 :
  ∃ a n, (448000 : ℝ) = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 4.48 ∧ n = 5 :=
by
  sorry

end scientific_notation_448000_l170_170878


namespace kolya_or_leva_wins_l170_170739

-- Definitions for segment lengths
variables (k l : ℝ)

-- Definition of the condition when Kolya wins
def kolya_wins (k l : ℝ) : Prop :=
  k > l

-- Definition of the condition when Leva wins
def leva_wins (k l : ℝ) : Prop :=
  k ≤ l

-- Theorem statement for the proof problem
theorem kolya_or_leva_wins (k l : ℝ) : kolya_wins k l ∨ leva_wins k l :=
sorry

end kolya_or_leva_wins_l170_170739


namespace ratio_of_volumes_l170_170816

noncomputable def volume_cone (r h : ℝ) : ℝ :=
  (1/3) * Real.pi * r^2 * h

theorem ratio_of_volumes :
  let r_C := 10
  let h_C := 20
  let r_D := 18
  let h_D := 12
  volume_cone r_C h_C / volume_cone r_D h_D = 125 / 243 :=
by
  sorry

end ratio_of_volumes_l170_170816


namespace total_people_in_cars_by_end_of_race_l170_170117

-- Define the initial conditions and question
def initial_num_cars : ℕ := 20
def initial_num_passengers_per_car : ℕ := 2
def initial_num_drivers_per_car : ℕ := 1
def extra_passengers_per_car : ℕ := 1

-- Define the number of people per car initially
def initial_people_per_car : ℕ := initial_num_passengers_per_car + initial_num_drivers_per_car

-- Define the number of people per car after gaining extra passenger
def final_people_per_car : ℕ := initial_people_per_car + extra_passengers_per_car

-- The statement to be proven
theorem total_people_in_cars_by_end_of_race : initial_num_cars * final_people_per_car = 80 := by
  -- Prove the theorem
  sorry

end total_people_in_cars_by_end_of_race_l170_170117


namespace max_value_l170_170657

-- Define the weights and values of gemstones
def weight_sapphire : ℕ := 6
def value_sapphire : ℕ := 15
def weight_ruby : ℕ := 3
def value_ruby : ℕ := 9
def weight_diamond : ℕ := 2
def value_diamond : ℕ := 5

-- Define the weight capacity
def max_weight : ℕ := 24

-- Define the availability constraint
def min_availability : ℕ := 10

-- The goal is to prove that the maximum value is 72
theorem max_value : ∃ (num_sapphire num_ruby num_diamond : ℕ),
  num_sapphire >= min_availability ∧
  num_ruby >= min_availability ∧
  num_diamond >= min_availability ∧
  num_sapphire * weight_sapphire + num_ruby * weight_ruby + num_diamond * weight_diamond ≤ max_weight ∧
  num_sapphire * value_sapphire + num_ruby * value_ruby + num_diamond * value_diamond = 72 :=
by sorry

end max_value_l170_170657


namespace tank_full_after_50_minutes_l170_170949

-- Define the conditions as constants
def tank_capacity : ℕ := 850
def pipe_a_rate : ℕ := 40
def pipe_b_rate : ℕ := 30
def pipe_c_rate : ℕ := 20
def cycle_duration : ℕ := 3  -- duration of each cycle in minutes
def net_water_per_cycle : ℕ := pipe_a_rate + pipe_b_rate - pipe_c_rate  -- net liters added per cycle

-- Define the statement to be proved: the tank will be full at exactly 50 minutes
theorem tank_full_after_50_minutes :
  ∀ minutes_elapsed : ℕ, (minutes_elapsed = 50) →
  ((minutes_elapsed / cycle_duration) * net_water_per_cycle = tank_capacity - pipe_c_rate) :=
sorry

end tank_full_after_50_minutes_l170_170949


namespace e_exp_ax1_ax2_gt_two_l170_170357

noncomputable def f (a x : ℝ) : ℝ := Real.exp (a * x) - a * (x + 2)

theorem e_exp_ax1_ax2_gt_two {a x1 x2 : ℝ} (h : a ≠ 0) (h1 : f a x1 = 0) (h2 : f a x2 = 0) (hx : x1 < x2) : 
  Real.exp (a * x1) + Real.exp (a * x2) > 2 :=
sorry

end e_exp_ax1_ax2_gt_two_l170_170357


namespace vector_subtraction_result_l170_170785

-- Defining the vectors a and b as given in the conditions
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

-- The main theorem stating that a - 2b results in the expected coordinates
theorem vector_subtraction_result :
  a - 2 • b = (7, -2) := by
  sorry

end vector_subtraction_result_l170_170785


namespace union_complement_correct_l170_170713

open Set

def A : Set ℝ := {0, 1, 2, 3}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≥ 0}

theorem union_complement_correct : A ∪ (compl B) = Ioo (-1 : ℝ) 3 ∪ {3} := by
  sorry

end union_complement_correct_l170_170713


namespace find_a3_l170_170314

-- Define the geometric sequence and its properties.
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Given conditions
variables (a : ℕ → ℝ) (q : ℝ)
variable (h_GeoSeq : is_geometric_sequence a q)
variable (h_a1 : a 1 = 1)
variable (h_a5 : a 5 = 9)

-- Define what we need to prove
theorem find_a3 : a 3 = 3 :=
sorry

end find_a3_l170_170314


namespace average_GPA_of_whole_class_l170_170296

variable (n : ℕ)

def GPA_first_group : ℕ := 54 * (n / 3)
def GPA_second_group : ℕ := 45 * (2 * n / 3)
def total_GPA : ℕ := GPA_first_group n + GPA_second_group n

theorem average_GPA_of_whole_class : total_GPA n / n = 48 := by
  sorry

end average_GPA_of_whole_class_l170_170296


namespace solve_for_m_l170_170472

theorem solve_for_m (a_0 a_1 a_2 a_3 a_4 a_5 m : ℝ)
  (h1 : (x : ℝ) → (x + m)^5 = a_0 + a_1 * (x+1) + a_2 * (x+1)^2 + a_3 * (x+1)^3 + a_4 * (x+1)^4 + a_5 * (x+1)^5)
  (h2 : a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 32) :
  m = 2 :=
sorry

end solve_for_m_l170_170472


namespace ratio_of_small_square_to_shaded_area_l170_170008

theorem ratio_of_small_square_to_shaded_area :
  let small_square_area := 2 * 2
  let large_square_area := 5 * 5
  let shaded_area := (large_square_area / 2) - (small_square_area / 2)
  (small_square_area : ℚ) / shaded_area = 8 / 21 :=
by
  sorry

end ratio_of_small_square_to_shaded_area_l170_170008


namespace batsman_average_after_17th_inning_l170_170915

theorem batsman_average_after_17th_inning 
    (A : ℕ) 
    (hA : A = 15) 
    (runs_17th_inning : ℕ)
    (increase_in_average : ℕ) 
    (hscores : runs_17th_inning = 100)
    (hincrease : increase_in_average = 5) :
    (A + increase_in_average = 20) :=
by
  sorry

end batsman_average_after_17th_inning_l170_170915


namespace count_whole_numbers_in_interval_l170_170035

theorem count_whole_numbers_in_interval :
  let a := (7 / 4)
  let b := (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℕ, a < x ∧ x < b → 2 ≤ x ∧ x ≤ 9 := by
  sorry

end count_whole_numbers_in_interval_l170_170035


namespace possible_n_values_l170_170473

theorem possible_n_values (x y z n : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
    x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2 → n = 1 ∨ n = 3 :=
by 
  sorry

end possible_n_values_l170_170473


namespace people_who_didnt_show_up_l170_170459

-- Definitions based on the conditions
def invited_people : ℕ := 68
def people_per_table : ℕ := 3
def tables_needed : ℕ := 6

-- Theorem statement
theorem people_who_didnt_show_up : 
  (invited_people - tables_needed * people_per_table = 50) :=
by 
  sorry

end people_who_didnt_show_up_l170_170459


namespace ratio_of_almonds_to_walnuts_l170_170488

theorem ratio_of_almonds_to_walnuts
  (A W : ℝ)
  (weight_almonds : ℝ)
  (total_weight : ℝ)
  (weight_walnuts : ℝ)
  (ratio : 2 * W = total_weight - weight_almonds)
  (given_almonds : weight_almonds = 107.14285714285714)
  (given_total_weight : total_weight = 150)
  (computed_weight_walnuts : weight_walnuts = 42.85714285714286)
  (proportion : A / (2 * W) = weight_almonds / weight_walnuts) :
  A / W = 5 :=
by
  sorry

end ratio_of_almonds_to_walnuts_l170_170488


namespace geometric_sequence_a6_l170_170396

theorem geometric_sequence_a6 (a : ℕ → ℝ) (a1 r : ℝ) (h1 : ∀ n, a n = a1 * r ^ (n - 1)) (h2 : (a 2) * (a 4) * (a 12) = 64) : a 6 = 4 :=
sorry

end geometric_sequence_a6_l170_170396


namespace thabo_total_books_l170_170424

-- Definitions and conditions mapped from the problem
def H : ℕ := 35
def P_NF : ℕ := H + 20
def P_F : ℕ := 2 * P_NF
def total_books : ℕ := H + P_NF + P_F

-- The theorem proving the total number of books
theorem thabo_total_books : total_books = 200 := by
  -- Proof goes here.
  sorry

end thabo_total_books_l170_170424


namespace ratio_eq_l170_170394

variable (a b c d : ℚ)

theorem ratio_eq :
  (a / b = 5 / 2) →
  (c / d = 7 / 3) →
  (d / b = 5 / 4) →
  (a / c = 6 / 7) :=
by
  intros h1 h2 h3
  sorry

end ratio_eq_l170_170394


namespace factor_quadratic_l170_170290

theorem factor_quadratic (x : ℝ) : 
  x^2 + 6 * x = 1 → (x + 3)^2 = 10 := 
by
  intro h
  sorry

end factor_quadratic_l170_170290


namespace total_eggs_l170_170554

def e0 : ℝ := 47.0
def ei : ℝ := 5.0

theorem total_eggs : e0 + ei = 52.0 := by
  sorry

end total_eggs_l170_170554


namespace max_sheets_one_participant_l170_170771

theorem max_sheets_one_participant
  (n : ℕ) (avg_sheets : ℕ) (h1 : n = 40) (h2 : avg_sheets = 7) 
  (h3 : ∀ i : ℕ, i < n → 1 ≤ 1) : 
  ∃ max_sheets : ℕ, max_sheets = 241 :=
by
  sorry

end max_sheets_one_participant_l170_170771


namespace number_multiplied_value_l170_170015

theorem number_multiplied_value (x : ℝ) :
  (4 / 6) * x = 8 → x = 12 :=
by
  sorry

end number_multiplied_value_l170_170015


namespace segment_lengths_l170_170731

noncomputable def radius : ℝ := 5
noncomputable def diameter : ℝ := 2 * radius
noncomputable def chord_length : ℝ := 8

-- The lengths of the segments AK and KB
theorem segment_lengths (x : ℝ) (y : ℝ) 
  (hx : 0 < x ∧ x < diameter) 
  (hy : 0 < y ∧ y < diameter) 
  (h1 : x + y = diameter) 
  (h2 : x * y = (diameter^2) / 4 - 16 / 4) : 
  x = 2.5 ∧ y = 7.5 := 
sorry

end segment_lengths_l170_170731


namespace number_of_team_members_l170_170961

theorem number_of_team_members (x x1 x2 : ℕ) (h₀ : x = x1 + x2) (h₁ : 3 * x1 + 4 * x2 = 33) : x = 6 :=
sorry

end number_of_team_members_l170_170961


namespace min_distance_to_circle_l170_170507

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2))

def is_on_circle (Q : ℝ × ℝ) : Prop :=
  (Q.1 - 1)^2 + Q.2^2 = 4

def P : ℝ × ℝ := (-2, -3)
def center : ℝ × ℝ := (1, 0)
def radius : ℝ := 2

theorem min_distance_to_circle : ∃ Q : ℝ × ℝ, is_on_circle Q ∧ distance P Q = 3 * (Real.sqrt 2) - radius :=
by
  sorry

end min_distance_to_circle_l170_170507


namespace simplify_fraction_l170_170310

theorem simplify_fraction : (3 : ℚ) / 462 + 17 / 42 = 95 / 231 :=
by sorry

end simplify_fraction_l170_170310


namespace sarah_problem_sum_l170_170091

theorem sarah_problem_sum (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 100 ≤ y ∧ y < 1000) (h : 1000 * x + y = 9 * x * y) :
  x + y = 126 :=
sorry

end sarah_problem_sum_l170_170091


namespace second_differences_of_cubes_l170_170192

-- Define the first difference for cubes of consecutive natural numbers
def first_difference (n : ℕ) : ℕ :=
  ((n + 1) ^ 3) - (n ^ 3)

-- Define the second difference for the first differences
def second_difference (n : ℕ) : ℕ :=
  first_difference (n + 1) - first_difference n

-- Proof statement: Prove that second differences are equal to 6n + 6
theorem second_differences_of_cubes (n : ℕ) : second_difference n = 6 * n + 6 :=
  sorry

end second_differences_of_cubes_l170_170192


namespace sequence_property_exists_l170_170836

theorem sequence_property_exists :
  ∃ a₁ a₂ a₃ a₄ : ℝ, 
  a₂ - a₁ = a₃ - a₂ ∧ a₃ - a₂ = a₄ - a₃ ∧
  (a₃ / a₁ = a₄ / a₃) ∧ ∃ r : ℝ, r ≠ 0 ∧ a₁ = -4 * r ∧ a₂ = -3 * r ∧ a₃ = -2 * r ∧ a₄ = -r :=
by
  sorry

end sequence_property_exists_l170_170836


namespace expected_value_of_N_l170_170628

noncomputable def expected_value_N : ℝ :=
  30

theorem expected_value_of_N :
  -- Suppose Bob chooses a 4-digit binary string uniformly at random,
  -- and examines an infinite sequence of independent random binary bits.
  -- Let N be the least number of bits Bob has to examine to find his chosen string.
  -- Then the expected value of N is 30.
  expected_value_N = 30 :=
by
  sorry

end expected_value_of_N_l170_170628


namespace books_before_addition_l170_170639

-- Let b be the initial number of books on the shelf
variable (b : ℕ)

theorem books_before_addition (h : b + 10 = 19) : b = 9 := by
  sorry

end books_before_addition_l170_170639


namespace sqrt_sum_difference_product_l170_170593

open Real

theorem sqrt_sum_difference_product :
  (sqrt 3 + sqrt 2) * (sqrt 3 - sqrt 2) = 1 := by
  sorry

end sqrt_sum_difference_product_l170_170593


namespace range_of_a_l170_170216

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x + Real.log x - (x^2 / (x - Real.log x))

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0) ↔
  1 < a ∧ a < (Real.exp 1) / (Real.exp 1 - 1) - 1 / Real.exp 1 :=
sorry

end range_of_a_l170_170216


namespace pyramid_transport_volume_l170_170714

-- Define the conditions of the problem
def pyramid_height : ℝ := 15
def pyramid_base_side_length : ℝ := 8
def box_length : ℝ := 10
def box_width : ℝ := 10
def box_height : ℝ := 15

-- Define the volume of the box
def box_volume : ℝ := box_length * box_width * box_height

-- State the theorem
theorem pyramid_transport_volume : box_volume = 1500 := by
  sorry

end pyramid_transport_volume_l170_170714


namespace minimum_value_of_polynomial_l170_170172

-- Define the polynomial expression
def polynomial_expr (x : ℝ) : ℝ := (8 - x) * (6 - x) * (8 + x) * (6 + x)

-- State the theorem with the minimum value
theorem minimum_value_of_polynomial : ∃ x : ℝ, polynomial_expr x = -196 := by
  sorry

end minimum_value_of_polynomial_l170_170172


namespace verify_salary_problem_l170_170343

def salary_problem (W : ℕ) (S_old : ℕ) (S_new : ℕ := 780) (n : ℕ := 9) : Prop :=
  (W + S_old) / n = 430 ∧ (W + S_new) / n = 420 → S_old = 870

theorem verify_salary_problem (W S_old : ℕ) (h1 : (W + S_old) / 9 = 430) (h2 : (W + 780) / 9 = 420) : S_old = 870 :=
by {
  sorry
}

end verify_salary_problem_l170_170343


namespace probability_at_least_seven_heads_or_tails_l170_170269

open Nat

-- Define the probability of getting at least seven heads or tails in eight coin flips
theorem probability_at_least_seven_heads_or_tails :
  let total_outcomes := 2^8
  let favorable_outcomes := (choose 8 7) + (choose 8 7) + 1 + 1
  let probability := (favorable_outcomes : ℝ) / total_outcomes
  probability = 9 / 128 := by
  sorry

end probability_at_least_seven_heads_or_tails_l170_170269


namespace mouse_jump_frog_jump_diff_l170_170562

open Nat

theorem mouse_jump_frog_jump_diff :
  let grasshopper_jump := 19
  let frog_jump := grasshopper_jump + 10
  let mouse_jump := grasshopper_jump + 30
  mouse_jump - frog_jump = 20 :=
by
  let grasshopper_jump := 19
  let frog_jump := grasshopper_jump + 10
  let mouse_jump := grasshopper_jump + 30
  have h1 : frog_jump = 29 := by decide
  have h2 : mouse_jump = 49 := by decide
  have h3 : mouse_jump - frog_jump = 20 := by decide
  exact h3

end mouse_jump_frog_jump_diff_l170_170562


namespace Tony_slices_left_after_week_l170_170467

-- Define the conditions and problem statement
def Tony_slices_per_day (days : ℕ) : ℕ := days * 2
def Tony_slices_on_Saturday : ℕ := 3 + 2
def Tony_slice_on_Sunday : ℕ := 1
def Total_slices_used (days : ℕ) : ℕ := Tony_slices_per_day days + Tony_slices_on_Saturday + Tony_slice_on_Sunday
def Initial_loaf : ℕ := 22
def Slices_left (days : ℕ) : ℕ := Initial_loaf - Total_slices_used days

-- Prove that Tony has 6 slices left after a week
theorem Tony_slices_left_after_week : Slices_left 5 = 6 := by
  sorry

end Tony_slices_left_after_week_l170_170467


namespace kate_spent_on_mouse_l170_170931

theorem kate_spent_on_mouse :
  let march := 27
  let april := 13
  let may := 28
  let saved := march + april + may
  let keyboard := 49
  let left := 14
  saved - left - keyboard = 5 :=
by
  let march := 27
  let april := 13
  let may := 28
  let saved := march + april + may
  let keyboard := 49
  let left := 14
  show saved - left - keyboard = 5
  sorry

end kate_spent_on_mouse_l170_170931


namespace cos_double_angle_l170_170755

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2 * θ) = -7 / 25 := 
sorry

end cos_double_angle_l170_170755


namespace area_relation_l170_170985

noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := 
  0.5 * (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_relation (A B C A' B' C' : ℝ × ℝ) (hAA'BB'CC'parallel: 
  ∃ k : ℝ, (A'.1 - A.1 = k * (B'.1 - B.1)) ∧ (A'.2 - A.2 = k * (B'.2 - B.2)) ∧ 
           (B'.1 - B.1 = k * (C'.1 - C.1)) ∧ (B'.2 - B.2 = k * (C'.2 - C.2))) :
  3 * (area_triangle A B C + area_triangle A' B' C') = 
    area_triangle A B' C' + area_triangle B C' A' + area_triangle C A' B' +
    area_triangle A' B C + area_triangle B' C A + area_triangle C' A B := 
sorry

end area_relation_l170_170985


namespace shaded_percentage_l170_170825

noncomputable def percent_shaded (side_len : ℕ) : ℝ :=
  let total_area := (side_len : ℝ) * side_len
  let shaded_area := (2 * 2) + (2 * 5) + (1 * 7)
  100 * (shaded_area / total_area)

theorem shaded_percentage (PQRS_side : ℕ) (hPQRS : PQRS_side = 7) :
  percent_shaded PQRS_side = 42.857 :=
  by
  rw [hPQRS]
  sorry

end shaded_percentage_l170_170825


namespace computation_problems_count_l170_170999

theorem computation_problems_count : 
  ∃ (x y : ℕ), 3 * x + 5 * y = 110 ∧ x + y = 30 ∧ x = 20 :=
by
  sorry

end computation_problems_count_l170_170999


namespace jerry_won_47_tickets_l170_170327

open Nat

-- Define the initial number of tickets
def initial_tickets : Nat := 4

-- Define the number of tickets spent on the beanie
def tickets_spent_on_beanie : Nat := 2

-- Define the current total number of tickets Jerry has
def current_tickets : Nat := 49

-- Define the number of tickets Jerry won later
def tickets_won_later : Nat := current_tickets - (initial_tickets - tickets_spent_on_beanie)

-- The theorem to prove
theorem jerry_won_47_tickets :
  tickets_won_later = 47 :=
by sorry

end jerry_won_47_tickets_l170_170327


namespace Janka_bottle_caps_l170_170643

theorem Janka_bottle_caps (n : ℕ) :
  (∃ k1 : ℕ, n = 3 * k1) ∧ (∃ k2 : ℕ, n = 4 * k2) ↔ n = 12 ∨ n = 24 :=
by
  sorry

end Janka_bottle_caps_l170_170643


namespace parallelogram_area_twice_quadrilateral_area_l170_170704

theorem parallelogram_area_twice_quadrilateral_area (S : ℝ) (LMNP_area : ℝ) 
  (h : LMNP_area = 2 * S) : LMNP_area = 2 * S := 
by {
  sorry
}

end parallelogram_area_twice_quadrilateral_area_l170_170704


namespace painting_area_l170_170042

theorem painting_area (wall_height wall_length bookshelf_height bookshelf_length : ℝ)
  (h_wall_height : wall_height = 10)
  (h_wall_length : wall_length = 15)
  (h_bookshelf_height : bookshelf_height = 3)
  (h_bookshelf_length : bookshelf_length = 5) :
  wall_height * wall_length - bookshelf_height * bookshelf_length = 135 := 
by
  sorry

end painting_area_l170_170042


namespace value_of_k_l170_170679

theorem value_of_k (k : ℤ) : 
  (∃ a b : ℤ, x^2 + k * x + 81 = a^2 * x^2 + 2 * a * b * x + b^2) → (k = 18 ∨ k = -18) :=
by
  sorry

end value_of_k_l170_170679


namespace average_visitors_on_Sundays_l170_170273

theorem average_visitors_on_Sundays (S : ℕ) 
  (h1 : 30 % 7 = 2)  -- The month begins with a Sunday
  (h2 : 25 = 30 - 5)  -- The month has 25 non-Sundays
  (h3 : (120 * 25) = 3000) -- Total visitors on non-Sundays
  (h4 : (125 * 30) = 3750) -- Total visitors for the month
  (h5 : 5 * 30 > 0) -- There are a positive number of Sundays
  : S = 150 :=
by
  sorry

end average_visitors_on_Sundays_l170_170273


namespace club_last_names_l170_170726

theorem club_last_names :
  ∃ A B C D E F : ℕ,
    A + B + C + D + E + F = 21 ∧
    A^2 + B^2 + C^2 + D^2 + E^2 + F^2 = 91 :=
by {
  sorry
}

end club_last_names_l170_170726


namespace find_a_l170_170265

theorem find_a (r s : ℚ) (a : ℚ) :
  (∀ x : ℚ, (ax^2 + 18 * x + 16 = (r * x + s)^2)) → 
  s = 4 ∨ s = -4 →
  a = (9 / 4) * (9 / 4)
:= sorry

end find_a_l170_170265


namespace geometric_sequence_n_value_l170_170518

theorem geometric_sequence_n_value (a₁ : ℕ) (q : ℕ) (a_n : ℕ) (n : ℕ) (h1 : a₁ = 1) (h2 : q = 2) (h3 : a_n = 64) (h4 : a_n = a₁ * q^(n-1)) : n = 7 :=
by
  sorry

end geometric_sequence_n_value_l170_170518


namespace eli_age_difference_l170_170297

theorem eli_age_difference (kaylin_age : ℕ) (freyja_age : ℕ) (sarah_age : ℕ) (eli_age : ℕ) 
  (H1 : kaylin_age = 33)
  (H2 : freyja_age = 10)
  (H3 : kaylin_age + 5 = sarah_age)
  (H4 : sarah_age = 2 * eli_age) :
  eli_age - freyja_age = 9 := 
sorry

end eli_age_difference_l170_170297


namespace percent_of_y_l170_170670

theorem percent_of_y (y : ℝ) (hy : y > 0) : (6 * y / 20) + (3 * y / 10) = 0.6 * y :=
by
  sorry

end percent_of_y_l170_170670


namespace find_number_l170_170267

theorem find_number (x : ℝ) :
  10 * x - 10 = 50 ↔ x = 6 := by
  sorry

end find_number_l170_170267


namespace unshaded_squares_in_tenth_figure_l170_170486

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + d * (n - 1)

theorem unshaded_squares_in_tenth_figure :
  arithmetic_sequence 8 4 10 = 44 :=
by
  sorry

end unshaded_squares_in_tenth_figure_l170_170486


namespace turtles_in_lake_l170_170029

-- Definitions based on conditions
def total_turtles : ℝ := 100
def percent_female : ℝ := 0.6
def percent_male : ℝ := 0.4
def percent_striped_male : ℝ := 0.25
def striped_turtle_babies : ℝ := 4
def percent_babies : ℝ := 0.4

-- Statement to prove
theorem turtles_in_lake : 
  (total_turtles * percent_male * percent_striped_male / percent_babies = striped_turtle_babies) →
  total_turtles = 100 :=
by
  sorry

end turtles_in_lake_l170_170029


namespace net_cannot_contain_2001_knots_l170_170599

theorem net_cannot_contain_2001_knots (knots : Nat) (ropes_per_knot : Nat) (total_knots : knots = 2001) (ropes_per_knot_eq : ropes_per_knot = 3) :
  false :=
by
  sorry

end net_cannot_contain_2001_knots_l170_170599


namespace number_of_short_trees_to_plant_l170_170145

-- Definitions of the conditions
def current_short_trees : ℕ := 41
def current_tall_trees : ℕ := 44
def total_short_trees_after_planting : ℕ := 98

-- The statement to be proved
theorem number_of_short_trees_to_plant :
  total_short_trees_after_planting - current_short_trees = 57 :=
by
  -- Proof goes here
  sorry

end number_of_short_trees_to_plant_l170_170145


namespace mul_101_101_l170_170500

theorem mul_101_101 : 101 * 101 = 10201 := 
by
  sorry

end mul_101_101_l170_170500


namespace part1_part2_l170_170207

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

-- Statement for part (1)
theorem part1 (m : ℝ) : (m > -2) → (∀ x : ℝ, m + f x > 0) :=
sorry

-- Statement for part (2)
theorem part2 (m : ℝ) : (m > 2) ↔ (∀ x : ℝ, m - f x > 0) :=
sorry

end part1_part2_l170_170207


namespace max_sum_at_1008_l170_170102

noncomputable def sum_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n * (a 1 + a n)) / 2

theorem max_sum_at_1008 (a : ℕ → ℝ) : 
  sum_sequence a 2015 > 0 → 
  sum_sequence a 2016 < 0 → 
  ∃ n, n = 1008 ∧ ∀ m, sum_sequence a m ≤ sum_sequence a 1008 :=
by
  intros h1 h2
  sorry

end max_sum_at_1008_l170_170102


namespace max_value_quadratic_l170_170559

theorem max_value_quadratic :
  (∃ x : ℝ, ∀ y : ℝ, -3*y^2 + 9*y + 24 ≤ -3*x^2 + 9*x + 24) ∧ (∃ x : ℝ, x = 3/2) :=
sorry

end max_value_quadratic_l170_170559


namespace identical_dice_probability_l170_170199

def num_ways_to_paint_die : ℕ := 3^6

def total_ways_to_paint_dice (n : ℕ) : ℕ := (num_ways_to_paint_die ^ n)

def count_identical_ways : ℕ := 1 + 324 + 8100

def probability_identical_dice : ℚ :=
  (count_identical_ways : ℚ) / (total_ways_to_paint_dice 2 : ℚ)

theorem identical_dice_probability : probability_identical_dice = 8425 / 531441 := by
  sorry

end identical_dice_probability_l170_170199


namespace problem_statement_l170_170944

variable {R : Type*} [LinearOrderedField R]

theorem problem_statement
  (x1 x2 x3 y1 y2 y3 : R)
  (h1 : x1 + x2 + x3 = 0)
  (h2 : y1 + y2 + y3 = 0)
  (h3 : x1 * y1 + x2 * y2 + x3 * y3 = 0)
  (h4 : (x1^2 + x2^2 + x3^2) * (y1^2 + y2^2 + y3^2) > 0) :
  (x1^2 / (x1^2 + x2^2 + x3^2) + y1^2 / (y1^2 + y2^2 + y3^2) = 2 / 3) := 
sorry

end problem_statement_l170_170944


namespace part_1_part_2_l170_170218

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 4 ≤ 0}

theorem part_1 (m : ℝ) : (A ∩ B m = {x | 0 ≤ x ∧ x ≤ 3}) → (m = 2) := 
by
  sorry

theorem part_2 (m : ℝ) : (A ⊆ (Set.univ \ B m)) → (m > 5 ∨ m < -3) := 
by
  sorry

end part_1_part_2_l170_170218


namespace prisoners_freedom_guaranteed_l170_170471

-- Definition of the problem strategy
def prisoners_strategy (n : ℕ) : Prop :=
  ∃ counter regular : ℕ → ℕ,
    (∀ i, i < n - 1 → regular i < 2) ∧ -- Each regular prisoner turns on the light only once
    (∃ count : ℕ, 
      counter count = 99 ∧  -- The counter counts to 99 based on the strategy
      (∀ k, k < 99 → (counter (k + 1) = counter k + 1))) -- Each turn off increases the count by one

-- The main proof statement that there is a strategy ensuring the prisoners' release
theorem prisoners_freedom_guaranteed : ∀ (n : ℕ), n = 100 →
  prisoners_strategy n :=
by {
  sorry -- The actual proof is omitted
}

end prisoners_freedom_guaranteed_l170_170471


namespace man_speed_with_stream_l170_170211

variable (V_m V_as : ℝ)
variable (V_s V_ws : ℝ)

theorem man_speed_with_stream
  (cond1 : V_m = 5)
  (cond2 : V_as = 8)
  (cond3 : V_as = V_m - V_s)
  (cond4 : V_ws = V_m + V_s) :
  V_ws = 8 := 
by
  sorry

end man_speed_with_stream_l170_170211


namespace no_perfect_square_m_in_range_l170_170918

theorem no_perfect_square_m_in_range : 
  ∀ m : ℕ, 4 ≤ m ∧ m ≤ 12 → ¬(∃ k : ℕ, 2 * m^2 + 3 * m + 2 = k^2) := by
sorry

end no_perfect_square_m_in_range_l170_170918


namespace smallest_six_consecutive_number_exists_max_value_N_perfect_square_l170_170060

-- Definition of 'six-consecutive numbers'
def is_six_consecutive (a b c d : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧
  b ≠ d ∧ c ≠ d ∧ (a + b) * (c + d) = 60

-- Definition of the function F
def F (a b c d : ℕ) : ℤ :=
  let p := (10 * a + c) - (10 * b + d)
  let q := (10 * a + d) - (10 * b + c)
  q - p

-- Exists statement for the smallest six-consecutive number
theorem smallest_six_consecutive_number_exists :
  ∃ (a b c d : ℕ), is_six_consecutive a b c d ∧ (1000 * a + 100 * b + 10 * c + d) = 1369 := 
sorry

-- Exists statement for the maximum N such that F(N) is perfect square
theorem max_value_N_perfect_square :
  ∃ (a b c d : ℕ), is_six_consecutive a b c d ∧ 
  (1000 * a + 100 * b + 10 * c + d) = 9613 ∧
  ∃ (k : ℤ), F a b c d = k ^ 2 := 
sorry

end smallest_six_consecutive_number_exists_max_value_N_perfect_square_l170_170060


namespace cos_sin_identity_l170_170255

theorem cos_sin_identity : 
  (Real.cos (75 * Real.pi / 180) + Real.sin (75 * Real.pi / 180)) * 
  (Real.cos (75 * Real.pi / 180) - Real.sin (75 * Real.pi / 180)) = -Real.sqrt 3 / 2 := 
  sorry

end cos_sin_identity_l170_170255


namespace rate_percent_per_annum_l170_170710

theorem rate_percent_per_annum (P : ℝ) (SI_increase : ℝ) (T_increase : ℝ) (R : ℝ) 
  (hP : P = 2000) (hSI_increase : SI_increase = 40) (hT_increase : T_increase = 4) 
  (h : SI_increase = P * R * T_increase / 100) : R = 0.5 :=
by  
  sorry

end rate_percent_per_annum_l170_170710


namespace division_of_decimals_l170_170719

theorem division_of_decimals : (0.45 : ℝ) / (0.005 : ℝ) = 90 := 
sorry

end division_of_decimals_l170_170719


namespace difference_of_roots_l170_170838

theorem difference_of_roots :
  ∀ (x : ℝ), (x^2 - 5*x + 6 = 0) → (∃ r1 r2 : ℝ, r1 > 2 ∧ r2 < r1 ∧ r1 - r2 = 1) :=
by
  sorry

end difference_of_roots_l170_170838


namespace pushups_percentage_l170_170541

def total_exercises : ℕ := 12 + 8 + 20

def percentage_pushups (total_ex: ℕ) : ℕ := (8 * 100) / total_ex

theorem pushups_percentage (h : total_exercises = 40) : percentage_pushups total_exercises = 20 :=
by
  sorry

end pushups_percentage_l170_170541


namespace slope_magnitude_l170_170123

-- Definitions based on given conditions
def parabola : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ y^2 = 4 * x }
def line (k m : ℝ) : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ y = k * x + m }
def focus : ℝ × ℝ := (1, 0)
def intersects (l p : Set (ℝ × ℝ)) : Prop := ∃ x1 y1 x2 y2, (x1, y1) ∈ l ∧ (x1, y1) ∈ p ∧ (x2, y2) ∈ l ∧ (x2, y2) ∈ p ∧ (x1, y1) ≠ (x2, y2)

theorem slope_magnitude (k m : ℝ) (h_k_nonzero : k ≠ 0) 
  (h_intersects : intersects (line k m) parabola) 
  (h_AF_2FB : ∀ x1 y1 x2 y2, (x1, y1) ∈ line k m → (x1, y1) ∈ parabola → 
                          (x2, y2) ∈ line k m → (x2, y2) ∈ parabola → 
                          (1 - x1 = 2 * (x2 - 1)) ∧ (-y1 = 2 * y2)) :
  |k| = 2 * Real.sqrt 2 :=
sorry

end slope_magnitude_l170_170123


namespace solve_diamond_l170_170083

theorem solve_diamond (d : ℕ) (hd : d < 10) (h : d * 9 + 6 = d * 10 + 3) : d = 3 :=
sorry

end solve_diamond_l170_170083


namespace quadratic_root_shift_l170_170962

theorem quadratic_root_shift (A B p : ℤ) (α β : ℤ) 
  (h1 : ∀ x, x^2 + p * x + 19 = 0 → x = α + 1 ∨ x = β + 1)
  (h2 : ∀ x, x^2 - A * x + B = 0 → x = α ∨ x = β)
  (h3 : α + β = A)
  (h4 : α * β = B) :
  A + B = 18 := 
sorry

end quadratic_root_shift_l170_170962


namespace Tony_packs_of_pens_l170_170460

theorem Tony_packs_of_pens (T : ℕ) 
  (Kendra_packs : ℕ := 4) 
  (pens_per_pack : ℕ := 3) 
  (Kendra_keep : ℕ := 2) 
  (Tony_keep : ℕ := 2)
  (friends_pens : ℕ := 14) 
  (total_pens_given : Kendra_packs * pens_per_pack - Kendra_keep + 3 * T - Tony_keep = friends_pens) :
  T = 2 :=
by {
  sorry
}

end Tony_packs_of_pens_l170_170460


namespace circle_radius_is_7_5_l170_170115

noncomputable def radius_of_circle (side_length : ℝ) : ℝ := sorry

theorem circle_radius_is_7_5 :
  radius_of_circle 12 = 7.5 := sorry

end circle_radius_is_7_5_l170_170115


namespace tanvi_rank_among_girls_correct_l170_170903

def Vikas_rank : ℕ := 9
def Tanvi_rank : ℕ := 17
def girls_between : ℕ := 2
def Tanvi_rank_among_girls : ℕ := 8

theorem tanvi_rank_among_girls_correct (Vikas_rank Tanvi_rank girls_between Tanvi_rank_among_girls : ℕ) 
  (h1 : Vikas_rank = 9) 
  (h2 : Tanvi_rank = 17) 
  (h3 : girls_between = 2)
  (h4 : Tanvi_rank_among_girls = 8): 
  Tanvi_rank_among_girls = 8 := by
  sorry

end tanvi_rank_among_girls_correct_l170_170903


namespace polynomial_at_neg_one_eq_neg_two_l170_170204

-- Define the polynomial f(x)
def polynomial (x : ℝ) : ℝ := 1 + x + 2 * x^2 + 3 * x^3 + 4 * x^4 + 5 * x^5

-- Define Horner's method process
def horner_method (x : ℝ) : ℝ :=
  let a5 := 5
  let a4 := 4
  let a3 := 3
  let a2 := 2
  let a1 := 1
  let a  := 1
  let u4 := a5 * x + a4
  let u3 := u4 * x + a3
  let u2 := u3 * x + a2
  let u1 := u2 * x + a1
  let u0 := u1 * x + a
  u0

-- Prove that the polynomial evaluated using Horner's method at x := -1 is equal to -2
theorem polynomial_at_neg_one_eq_neg_two : horner_method (-1) = -2 := by
  sorry

end polynomial_at_neg_one_eq_neg_two_l170_170204


namespace distinct_remainders_l170_170238

theorem distinct_remainders (p : ℕ) (a : Fin p → ℤ) (hp : Nat.Prime p) :
  ∃ k : ℤ, (Finset.univ.image (fun i : Fin p => (a i + i * k) % p)).card ≥ ⌈(p / 2 : ℚ)⌉ :=
sorry

end distinct_remainders_l170_170238


namespace total_sequins_is_162_l170_170031

/-- Jane sews 6 rows of 8 blue sequins each. -/
def rows_of_blue_sequins : Nat := 6
def sequins_per_blue_row : Nat := 8
def total_blue_sequins : Nat := rows_of_blue_sequins * sequins_per_blue_row

/-- Jane sews 5 rows of 12 purple sequins each. -/
def rows_of_purple_sequins : Nat := 5
def sequins_per_purple_row : Nat := 12
def total_purple_sequins : Nat := rows_of_purple_sequins * sequins_per_purple_row

/-- Jane sews 9 rows of 6 green sequins each. -/
def rows_of_green_sequins : Nat := 9
def sequins_per_green_row : Nat := 6
def total_green_sequins : Nat := rows_of_green_sequins * sequins_per_green_row

/-- The total number of sequins Jane adds to her costume. -/
def total_sequins : Nat := total_blue_sequins + total_purple_sequins + total_green_sequins

theorem total_sequins_is_162 : total_sequins = 162 := 
by
  sorry

end total_sequins_is_162_l170_170031


namespace total_balloons_correct_l170_170993

-- Definitions based on the conditions
def brookes_initial_balloons : Nat := 12
def brooke_additional_balloons : Nat := 8

def tracys_initial_balloons : Nat := 6
def tracy_additional_balloons : Nat := 24

-- Calculate the number of balloons each person has after the additions and Tracy popping half
def brookes_final_balloons : Nat := brookes_initial_balloons + brooke_additional_balloons
def tracys_balloons_after_addition : Nat := tracys_initial_balloons + tracy_additional_balloons
def tracys_final_balloons : Nat := tracys_balloons_after_addition / 2

-- Total number of balloons
def total_balloons : Nat := brookes_final_balloons + tracys_final_balloons

-- The proof statement
theorem total_balloons_correct : total_balloons = 35 := by
  -- Proof would go here (but we'll skip with sorry)
  sorry

end total_balloons_correct_l170_170993


namespace line_passes_through_circle_center_l170_170854

theorem line_passes_through_circle_center (a : ℝ) :
  (∃ x y : ℝ, (x^2 + y^2 + 2*x - 4*y = 0) ∧ (3*x + y + a = 0)) → a = 1 :=
by
  sorry

end line_passes_through_circle_center_l170_170854


namespace num_integer_solutions_prime_l170_170728

def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m > 0 ∧ m < n → n % m ≠ 0

def integer_solutions : List ℤ := [-1, 3]

theorem num_integer_solutions_prime :
  (∀ x ∈ integer_solutions, is_prime (|15 * x^2 - 32 * x - 28|)) ∧ (integer_solutions.length = 2) :=
by
  sorry

end num_integer_solutions_prime_l170_170728


namespace greatest_possible_n_l170_170564

theorem greatest_possible_n (n : ℤ) (h1 : 102 * n^2 ≤ 8100) : n ≤ 8 :=
sorry

end greatest_possible_n_l170_170564


namespace seq_a_2014_l170_170217

theorem seq_a_2014 {a : ℕ → ℕ}
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 0 < n → n * a (n + 1) = (n + 1) * a n) :
  a 2014 = 2014 :=
sorry

end seq_a_2014_l170_170217


namespace find_g_inverse_84_l170_170777

-- Definition of the function g
def g (x : ℝ) : ℝ := 3 * x^3 + 3

-- Definition stating the goal
theorem find_g_inverse_84 : g⁻¹ 84 = 3 :=
sorry

end find_g_inverse_84_l170_170777


namespace hiker_final_distance_l170_170832

theorem hiker_final_distance :
  let east := 24
  let north := 7
  let west := 15
  let south := 5
  let net_east := east - west
  let net_north := north - south
  net_east = 9 ∧ net_north = 2 →
  Real.sqrt ((net_east)^2 + (net_north)^2) = Real.sqrt 85 :=
by
  intros
  sorry

end hiker_final_distance_l170_170832


namespace correct_sum_after_digit_change_l170_170573

theorem correct_sum_after_digit_change :
  let d := 7
  let e := 8
  let num1 := 935641
  let num2 := 471850
  let correct_sum := num1 + num2
  let new_sum := correct_sum + 10000
  new_sum = 1417491 := 
sorry

end correct_sum_after_digit_change_l170_170573


namespace triangle_area_l170_170250

theorem triangle_area (r : ℝ) (a b c : ℝ) (h : a^2 + b^2 = c^2) (hc : c = 2 * r) (r_val : r = 5) (ratio : a / b = 3 / 4 ∧ b / c = 4 / 5) :
  (1 / 2) * a * b = 24 :=
by
  -- We assume statements are given
  sorry

end triangle_area_l170_170250


namespace original_number_of_people_is_fifteen_l170_170928

/-!
The average age of all the people who gathered at a family celebration was equal to the number of attendees. 
Aunt Beta, who was 29 years old, soon excused herself and left. 
Even after Aunt Beta left, the average age of all the remaining attendees was still equal to their number.
Prove that the original number of people at the celebration is 15.
-/

theorem original_number_of_people_is_fifteen
  (n : ℕ)
  (s : ℕ)
  (h1 : s = n^2)
  (h2 : s - 29 = (n - 1)^2):
  n = 15 :=
by
  sorry

end original_number_of_people_is_fifteen_l170_170928


namespace product_of_all_possible_values_l170_170799

theorem product_of_all_possible_values (x : ℝ) (h : 2 * |x + 3| - 4 = 2) :
  ∃ (a b : ℝ), (x = a ∨ x = b) ∧ a * b = 0 :=
by
  sorry

end product_of_all_possible_values_l170_170799


namespace number_of_sets_of_positive_integers_l170_170194

theorem number_of_sets_of_positive_integers : 
  ∃ n : ℕ, n = 3333 ∧ ∀ x y z : ℕ, x > 0 → y > 0 → z > 0 → x < y → y < z → x + y + z = 203 → n = 3333 :=
by
  sorry

end number_of_sets_of_positive_integers_l170_170194


namespace probability_of_drawing_3_black_and_2_white_l170_170399

noncomputable def total_ways_to_draw_5_balls : ℕ := Nat.choose 27 5
noncomputable def ways_to_choose_3_black : ℕ := Nat.choose 10 3
noncomputable def ways_to_choose_2_white : ℕ := Nat.choose 12 2
noncomputable def favorable_outcomes : ℕ := ways_to_choose_3_black * ways_to_choose_2_white
noncomputable def desired_probability : ℚ := favorable_outcomes / total_ways_to_draw_5_balls

theorem probability_of_drawing_3_black_and_2_white :
  desired_probability = 132 / 1345 := by
  sorry

end probability_of_drawing_3_black_and_2_white_l170_170399


namespace shared_property_l170_170966

-- Definitions of the shapes
structure Parallelogram where
  sides_equal    : Bool -- Parallelograms have opposite sides equal but not necessarily all four.

structure Rectangle where
  sides_equal    : Bool -- Rectangles have opposite sides equal.
  diagonals_equal: Bool

structure Rhombus where
  sides_equal: Bool -- Rhombuses have all sides equal.
  diagonals_perpendicular: Bool -- Diagonals of a rhombus are perpendicular.

structure Square where
  sides_equal: Bool -- Squares have all sides equal.
  diagonals_perpendicular: Bool -- Diagonals of a square are perpendicular.
  diagonals_equal: Bool -- Diagonals of a square are equal in length.

-- Definitions of properties
def all_sides_equal (p1 p2 p3 p4 : Parallelogram) := p1.sides_equal ∧ p2.sides_equal ∧ p3.sides_equal ∧ p4.sides_equal
def diagonals_equal (r1 r2 r3 : Rectangle) (s1 s2 : Square) := r1.diagonals_equal ∧ r2.diagonals_equal ∧ s1.diagonals_equal ∧ s2.diagonals_equal
def diagonals_perpendicular (r1 : Rhombus) (s1 s2 : Square) := r1.diagonals_perpendicular ∧ s1.diagonals_perpendicular ∧ s2.diagonals_perpendicular
def diagonals_bisect_each_other (p1 p2 p3 p4 : Parallelogram) (r1 : Rectangle) (r2 : Rhombus) (s1 s2 : Square) := True -- All these shapes have diagonals that bisect each other.

-- The statement we need to prove
theorem shared_property (p1 p2 p3 p4 : Parallelogram) (r1 r2 : Rectangle) (r3 : Rhombus) (s1 s2 : Square) : 
  (diagonals_bisect_each_other p1 p2 p3 p4 r1 r3 s1 s2) :=
by
  sorry

end shared_property_l170_170966


namespace MaryHasBlueMarbles_l170_170625

-- Define the number of blue marbles Dan has
def DanMarbles : Nat := 5

-- Define the relationship of Mary's marbles to Dan's marbles
def MaryMarbles : Nat := 2 * DanMarbles

-- State the theorem that we need to prove
theorem MaryHasBlueMarbles : MaryMarbles = 10 :=
by
  sorry

end MaryHasBlueMarbles_l170_170625


namespace ants_of_species_X_on_day_6_l170_170549

/-- Given the initial populations of Species X and Species Y and their growth rates,
    prove the number of Species X ants on Day 6. -/
theorem ants_of_species_X_on_day_6 
  (x y : ℕ)  -- Number of Species X and Y ants on Day 0
  (h1 : x + y = 40)  -- Total number of ants on Day 0
  (h2 : 64 * x + 4096 * y = 21050)  -- Total number of ants on Day 6
  :
  64 * x = 2304 := 
sorry

end ants_of_species_X_on_day_6_l170_170549


namespace ajay_income_l170_170153

theorem ajay_income
  (I : ℝ)
  (h₁ : I * 0.45 + I * 0.25 + I * 0.075 + 9000 = I) :
  I = 40000 :=
by
  sorry

end ajay_income_l170_170153


namespace Euler_theorem_l170_170316

theorem Euler_theorem {m a : ℕ} (hm : m ≥ 1) (h_gcd : Nat.gcd a m = 1) : a ^ Nat.totient m ≡ 1 [MOD m] :=
by
  sorry

end Euler_theorem_l170_170316


namespace exist_positive_int_for_arithmetic_mean_of_divisors_l170_170826

theorem exist_positive_int_for_arithmetic_mean_of_divisors
  (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_distinct : p ≠ q) :
  ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ 
  (∃ k : ℕ, k * (a + 1) * (b + 1) = (p^(a+1) - 1) / (p - 1) * (q^(b+1) - 1) / (q - 1)) :=
sorry

end exist_positive_int_for_arithmetic_mean_of_divisors_l170_170826


namespace cubic_sum_l170_170843

theorem cubic_sum (x y : ℝ) (h1 : x + y = 8) (h2 : x * y = 12) : x^3 + y^3 = 224 :=
by
  sorry

end cubic_sum_l170_170843


namespace willie_bananas_remain_same_l170_170640

variable (Willie_bananas Charles_bananas Charles_loses : ℕ)

theorem willie_bananas_remain_same (h_willie : Willie_bananas = 48) (h_charles_initial : Charles_bananas = 14) (h_charles_loses : Charles_loses = 35) :
  Willie_bananas = 48 :=
by
  sorry

end willie_bananas_remain_same_l170_170640


namespace smallest_positive_integer_l170_170423

theorem smallest_positive_integer (x : ℕ) (hx_pos : x > 0) (h : x < 15) : x = 1 :=
by
  sorry

end smallest_positive_integer_l170_170423


namespace find_s_of_2_l170_170935

-- Define t and s as per the given conditions
def t (x : ℚ) : ℚ := 4 * x - 9
def s (x : ℚ) : ℚ := x^2 + 4 * x - 5

-- The theorem that we need to prove
theorem find_s_of_2 : s 2 = 217 / 16 := by
  sorry

end find_s_of_2_l170_170935


namespace carol_is_inviting_friends_l170_170436

theorem carol_is_inviting_friends :
  ∀ (invitations_per_pack packs_needed friends_invited : ℕ), 
  invitations_per_pack = 2 → 
  packs_needed = 5 → 
  friends_invited = invitations_per_pack * packs_needed → 
  friends_invited = 10 :=
by
  intros invitations_per_pack packs_needed friends_invited h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end carol_is_inviting_friends_l170_170436


namespace original_number_fraction_l170_170964

theorem original_number_fraction (x : ℚ) (h : 1 + 1/x = 9/4) : x = 4/5 := by
  sorry

end original_number_fraction_l170_170964


namespace perfect_square_expression_l170_170110

theorem perfect_square_expression (n : ℕ) : ∃ t : ℕ, n^2 - 4 * n + 11 = t^2 ↔ n = 5 :=
by
  sorry

end perfect_square_expression_l170_170110


namespace max_rectangle_area_with_prime_dimension_l170_170987

theorem max_rectangle_area_with_prime_dimension :
  ∃ (l w : ℕ), 2 * (l + w) = 120 ∧ (Prime l ∨ Prime w) ∧ l * w = 899 :=
by
  sorry

end max_rectangle_area_with_prime_dimension_l170_170987


namespace number_of_zeros_l170_170741

noncomputable def f (a : ℝ) (x : ℝ) := a * x^2 - 2 * a * x + a + 1
noncomputable def g (b : ℝ) (x : ℝ) := b * x^3 - 2 * b * x^2 + b * x - 4 / 27

theorem number_of_zeros (a b : ℝ) (ha : a > 0) (hb : b > 1) :
  ∃! (x : ℝ), g b (f a x) = 0 := sorry

end number_of_zeros_l170_170741


namespace f_f_0_eq_zero_number_of_zeros_l170_170642

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x > 0 then 1 - 1/x else (a - 1) * x + 1

theorem f_f_0_eq_zero (a : ℝ) : f a (f a 0) = 0 := by
  sorry

theorem number_of_zeros (a : ℝ) : 
  if a = 1 then ∃! x, f a x = 0 else
  if a > 1 then ∃! x1, ∃! x2, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0 else
  ∃! x, f a x = 0 := by sorry

end f_f_0_eq_zero_number_of_zeros_l170_170642


namespace power_function_value_at_neg2_l170_170431

theorem power_function_value_at_neg2 
  (f : ℝ → ℝ)
  (a : ℝ)
  (h1 : ∀ x : ℝ, f x = x^a)
  (h2 : f 2 = 1 / 4) 
  : f (-2) = 1 / 4 := by
  sorry

end power_function_value_at_neg2_l170_170431


namespace total_volume_of_cubes_l170_170339

theorem total_volume_of_cubes :
  let sarah_side_length := 3
  let sarah_num_cubes := 8
  let tom_side_length := 4
  let tom_num_cubes := 4
  let sarah_volume := sarah_num_cubes * sarah_side_length^3
  let tom_volume := tom_num_cubes * tom_side_length^3
  sarah_volume + tom_volume = 472 := by
  -- Definitions coming from conditions
  let sarah_side_length := 3
  let sarah_num_cubes := 8
  let tom_side_length := 4
  let tom_num_cubes := 4
  let sarah_volume := sarah_num_cubes * sarah_side_length^3
  let tom_volume := tom_num_cubes * tom_side_length^3
  -- Total volume of all cubes
  have h : sarah_volume + tom_volume = 472 := by sorry

  exact h

end total_volume_of_cubes_l170_170339


namespace simplify_expression_l170_170246

variable (x y : ℝ)
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)

theorem simplify_expression : (x⁻¹ - y) ^ 2 = (1 / x ^ 2 - 2 * y / x + y ^ 2) :=
  sorry

end simplify_expression_l170_170246


namespace no_function_satisfies_condition_l170_170416

theorem no_function_satisfies_condition :
  ¬ ∃ f : ℤ → ℤ, ∀ x y : ℤ, f (x + f y) = f x - y :=
sorry

end no_function_satisfies_condition_l170_170416


namespace evaluate_expression_l170_170241

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluate_expression_l170_170241


namespace minimum_students_l170_170452

variables (b g : ℕ) -- Define variables for boys and girls

-- Define the conditions
def boys_passed : ℕ := (3 * b) / 4
def girls_passed : ℕ := (2 * g) / 3
def equal_passed := boys_passed b = girls_passed g

def total_students := b + g + 4

-- Statement to prove minimum students in the class
theorem minimum_students (h1 : equal_passed b g)
  (h2 : ∃ multiple_of_nine : ℕ, g = 9 * multiple_of_nine ∧ 3 * b = 4 * multiple_of_nine * 2) :
  total_students b g = 21 :=
sorry

end minimum_students_l170_170452


namespace find_original_number_l170_170925

theorem find_original_number (c : ℝ) (h₁ : c / 12.75 = 16) (h₂ : 2.04 / 1.275 = 1.6) : c = 204 :=
by
  sorry

end find_original_number_l170_170925


namespace correct_bio_experiment_technique_l170_170862

-- Let's define our conditions as hypotheses.
def yeast_count_method := "sampling_inspection"
def small_animal_group_method := "sampler_sampling"
def mitosis_rinsing_purpose := "wash_away_dissociation_solution"
def fat_identification_solution := "alcohol"

-- The question translated into a statement is to show that the method for counting yeast is the sampling inspection method.
theorem correct_bio_experiment_technique :
  yeast_count_method = "sampling_inspection" ∧
  small_animal_group_method ≠ "mark-recapture" ∧
  mitosis_rinsing_purpose ≠ "wash_away_dye" ∧
  fat_identification_solution ≠ "50%_hydrochloric_acid" :=
sorry

end correct_bio_experiment_technique_l170_170862


namespace angles_with_same_terminal_side_l170_170004

theorem angles_with_same_terminal_side (k : ℤ) :
  {θ : ℝ | ∃ k : ℤ, θ = k * 360 + 260} = 
  {θ : ℝ | ∃ k : ℤ, θ = k * 360 + (-460 % 360)} :=
by sorry

end angles_with_same_terminal_side_l170_170004


namespace evaluate_fx_plus_2_l170_170547

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 1)

theorem evaluate_fx_plus_2 (x : ℝ) (h : x ^ 2 ≠ 1) : 
  f (x + 2) = (x + 3) / (x + 1) :=
by
  sorry

end evaluate_fx_plus_2_l170_170547


namespace positive_integers_a_2014_b_l170_170232

theorem positive_integers_a_2014_b (a : ℕ) :
  (∃! b : ℕ, 2 ≤ a / b ∧ a / b ≤ 5) → a = 6710 ∨ a = 6712 ∨ a = 6713 :=
by
  sorry

end positive_integers_a_2014_b_l170_170232


namespace jump_length_third_frog_l170_170187

theorem jump_length_third_frog (A B C : ℝ) 
  (h1 : A = (B + C) / 2) 
  (h2 : B = (A + C) / 2) 
  (h3 : |B - A| + |(B - C) / 2| = 60) : 
  |C - (A + B) / 2| = 30 :=
sorry

end jump_length_third_frog_l170_170187


namespace crabapple_recipients_sequence_count_l170_170696

/-- Mrs. Crabapple teaches a class of 15 students and her advanced literature class meets three times a week.
    She picks a new student each period to receive a crabapple, ensuring no student receives more than one
    crabapple in a week. Prove that the number of different sequences of crabapple recipients is 2730. -/
theorem crabapple_recipients_sequence_count :
  ∃ sequence_count : ℕ, sequence_count = 15 * 14 * 13 ∧ sequence_count = 2730 :=
by
  sorry

end crabapple_recipients_sequence_count_l170_170696


namespace tan_beta_minus_2alpha_l170_170781

theorem tan_beta_minus_2alpha (alpha beta : ℝ) (h1 : Real.tan alpha = 2) (h2 : Real.tan (beta - alpha) = 3) : 
  Real.tan (beta - 2 * alpha) = 1 / 7 := 
sorry

end tan_beta_minus_2alpha_l170_170781


namespace unique_positive_real_solution_l170_170758

def f (x : ℝ) := x^11 + 5 * x^10 + 20 * x^9 + 1000 * x^8 - 800 * x^7

theorem unique_positive_real_solution :
  ∃! (x : ℝ), 0 < x ∧ f x = 0 :=
sorry

end unique_positive_real_solution_l170_170758


namespace max_visible_unit_cubes_l170_170342

def cube_size := 11
def total_unit_cubes := cube_size ^ 3

def visible_unit_cubes (n : ℕ) : ℕ :=
  (n * n) + (n * (n - 1)) + ((n - 1) * (n - 1))

theorem max_visible_unit_cubes : 
  visible_unit_cubes cube_size = 331 := by
  sorry

end max_visible_unit_cubes_l170_170342


namespace seq_problem_l170_170303

theorem seq_problem (a : ℕ → ℚ) (d : ℚ) (h_arith : ∀ n : ℕ, a (n + 1) = a n + d )
 (h1 : a 1 = 2)
 (h_geom : (a 1 - 1) * (a 5 + 5) = (a 3)^2) :
  a 2017 = 1010 := 
sorry

end seq_problem_l170_170303


namespace minimum_employees_needed_l170_170264

noncomputable def employees_needed (total_days : ℕ) (work_days : ℕ) (rest_days : ℕ) (min_on_duty : ℕ) : ℕ :=
  let comb := (total_days.choose rest_days)
  min_on_duty * comb / work_days

theorem minimum_employees_needed {total_days work_days rest_days min_on_duty : ℕ} (h_total_days: total_days = 7) (h_work_days: work_days = 5) (h_rest_days: rest_days = 2) (h_min_on_duty: min_on_duty = 45) : 
  employees_needed total_days work_days rest_days min_on_duty = 63 := by
  rw [h_total_days, h_work_days, h_rest_days, h_min_on_duty]
  -- detailed computation and proofs steps omitted
  -- the critical part is to ensure 63 is derived correctly based on provided values
  sorry

end minimum_employees_needed_l170_170264


namespace first_shaded_square_ensuring_all_columns_l170_170860

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def shaded_squares_in_columns (k : ℕ) : Prop :=
  ∀ j : ℕ, j < 7 → ∃ n : ℕ, triangular_number n % 7 = j ∧ triangular_number n ≤ k

theorem first_shaded_square_ensuring_all_columns:
  shaded_squares_in_columns 55 :=
by
  sorry

end first_shaded_square_ensuring_all_columns_l170_170860


namespace range_of_a_plus_abs_b_l170_170648

theorem range_of_a_plus_abs_b (a b : ℝ)
  (h1 : -1 ≤ a) (h2 : a ≤ 3)
  (h3 : -5 < b) (h4 : b < 3) :
  -1 ≤ a + |b| ∧ a + |b| < 8 := by
sorry

end range_of_a_plus_abs_b_l170_170648


namespace geometric_sequence_a6_l170_170176

theorem geometric_sequence_a6 (a : ℕ → ℝ) 
  (h1 : a 4 * a 8 = 9) 
  (h2 : a 4 + a 8 = 8) 
  (geom_seq : ∀ n m, a (n + m) = a n * a m): 
  a 6 = 3 :=
by
  -- skipped proof
  sorry

end geometric_sequence_a6_l170_170176


namespace josh_marbles_l170_170372

theorem josh_marbles (initial_marbles lost_marbles remaining_marbles : ℤ) 
  (h1 : initial_marbles = 19) 
  (h2 : lost_marbles = 11) 
  (h3 : remaining_marbles = initial_marbles - lost_marbles) : 
  remaining_marbles = 8 := 
by
  sorry

end josh_marbles_l170_170372


namespace child_ticket_price_l170_170581

theorem child_ticket_price
    (num_people : ℕ)
    (num_adults : ℕ)
    (num_seniors : ℕ)
    (num_children : ℕ)
    (adult_ticket_cost : ℝ)
    (senior_discount : ℝ)
    (total_bill : ℝ) :
    num_people = 50 →
    num_adults = 25 →
    num_seniors = 15 →
    num_children = 10 →
    adult_ticket_cost = 15 →
    senior_discount = 0.25 →
    total_bill = 600 →
    ∃ x : ℝ, x = 5.63 :=
by {
  sorry
}

end child_ticket_price_l170_170581


namespace average_speed_train_l170_170702

theorem average_speed_train (x : ℝ) (h1 : x ≠ 0) :
  let t1 := x / 40
  let t2 := 2 * x / 20
  let t3 := 3 * x / 60
  let total_time := t1 + t2 + t3
  let total_distance := 6 * x
  let average_speed := total_distance / total_time
  average_speed = 240 / 7 := by
  sorry

end average_speed_train_l170_170702


namespace number_of_classmates_l170_170395

theorem number_of_classmates (n : ℕ) (Alex Aleena : ℝ) 
  (h_Alex : Alex = 1/11) (h_Aleena : Aleena = 1/14) 
  (h_bound : ∀ (x : ℝ), Aleena ≤ x ∧ x ≤ Alex → ∃ c : ℕ, n - 2 > 0 ∧ c = n - 2) : 
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_l170_170395


namespace additional_savings_zero_l170_170184

noncomputable def windows_savings (purchase_price : ℕ) (free_windows : ℕ) (paid_windows : ℕ)
  (dave_needs : ℕ) (doug_needs : ℕ) : ℕ := sorry

theorem additional_savings_zero :
  windows_savings 100 2 5 12 10 = 0 := sorry

end additional_savings_zero_l170_170184


namespace cost_to_open_store_l170_170669

-- Define the conditions as constants
def revenue_per_month : ℕ := 4000
def expenses_per_month : ℕ := 1500
def months_to_payback : ℕ := 10

-- Theorem stating the cost to open the store
theorem cost_to_open_store : (revenue_per_month - expenses_per_month) * months_to_payback = 25000 :=
by
  sorry

end cost_to_open_store_l170_170669


namespace clock_hands_overlap_l170_170373

theorem clock_hands_overlap (t : ℝ) :
  (∀ (h_angle m_angle : ℝ), h_angle = 30 + 0.5 * t ∧ m_angle = 6 * t ∧ h_angle = m_angle ∧ h_angle = 45) → t = 8 :=
by
  intro h
  sorry

end clock_hands_overlap_l170_170373


namespace problem1_problem2_problem3_problem4_l170_170331

theorem problem1 (h : Real.cos 75 * Real.sin 75 = 1 / 2) : False :=
by
  sorry

theorem problem2 : (1 + Real.tan 15) / (1 - Real.tan 15) = Real.sqrt 3 :=
by
  sorry

theorem problem3 : Real.tan 20 + Real.tan 25 + Real.tan 20 * Real.tan 25 = 1 :=
by
  sorry

theorem problem4 (θ : Real) (h1 : Real.sin (2 * θ) ≠ 0) : (1 / Real.tan θ - 1 / Real.tan (2 * θ) = 1 / Real.sin (2 * θ)) :=
by
  sorry

end problem1_problem2_problem3_problem4_l170_170331


namespace ratio_blue_yellow_l170_170210

theorem ratio_blue_yellow (total_butterflies blue_butterflies black_butterflies : ℕ)
  (h_total : total_butterflies = 19)
  (h_blue : blue_butterflies = 6)
  (h_black : black_butterflies = 10) :
  (blue_butterflies : ℚ) / (total_butterflies - blue_butterflies - black_butterflies : ℚ) = 2 / 1 := 
by {
  sorry
}

end ratio_blue_yellow_l170_170210


namespace distance_covered_l170_170100

/-- 
Given the following conditions:
1. The speed of Abhay (A) is 5 km/h.
2. The time taken by Abhay to cover a distance is 2 hours more than the time taken by Sameer.
3. If Abhay doubles his speed, then he would take 1 hour less than Sameer.
Prove that the distance (D) they are covering is 30 kilometers.
-/
theorem distance_covered (D S : ℝ) (A : ℝ) (hA : A = 5) 
  (h1 : D / A = D / S + 2) 
  (h2 : D / (2 * A) = D / S - 1) : 
  D = 30 := by
    sorry

end distance_covered_l170_170100


namespace probability_of_answering_phone_in_4_rings_l170_170127

/-- A proof statement that asserts the probability of answering the phone within the first four rings is equal to 9/10. -/
theorem probability_of_answering_phone_in_4_rings :
  (1/10) + (3/10) + (2/5) + (1/10) = 9/10 :=
by
  sorry

end probability_of_answering_phone_in_4_rings_l170_170127


namespace age_difference_l170_170807

variables (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 14) : C = A - 14 :=
by sorry

end age_difference_l170_170807


namespace find_S20_l170_170291

variable {α : Type*} [AddCommGroup α] [Module ℝ α]
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
axiom sum_first_n_terms (n : ℕ) : S n = n * (a 1 + a n) / 2
axiom points_collinear (A B C O : α) : Collinear ℝ ({A, B, C} : Set α) ∧ O = 0
axiom vector_relationship (A B C O : α) : O = 0 → C = (a 12) • A + (a 9) • B
axiom line_not_through_origin (A B O : α) : ¬Collinear ℝ ({O, A, B} : Set α)

-- Question: To find S 20
theorem find_S20 (A B C O : α) (h_collinear : Collinear ℝ ({A, B, C} : Set α)) 
  (h_vector : O = 0 → C = (a 12) • A + (a 9) • B) 
  (h_origin : O = 0)
  (h_not_through_origin : ¬Collinear ℝ ({O, A, B} : Set α)) : 
  S 20 = 10 := by
  sorry

end find_S20_l170_170291


namespace problem_solution_l170_170449

noncomputable def f (x : ℝ) : ℝ := (1 / 4) * (x + 1)^2

theorem problem_solution :
  (∀ x : ℝ, (0 < x ∧ x ≤ 5) → x ≤ f x ∧ f x ≤ 2 * |x - 1| + 1) →
  (f 1 = 4 * (1 / 4) + 1) →
  (∃ (t m : ℝ), m > 1 ∧ 
               (∀ x : ℝ, (1 ≤ x ∧ x ≤ m) → f t ≤ (1 / 4) * (x + t + 1)^2)) →
  (1 / 4 = 1 / 4) ∧ (m = 2) :=
by
  intros h1 h2 h3
  sorry

end problem_solution_l170_170449


namespace tangent_line_l170_170844

variable (a b x₀ y₀ x y : ℝ)
variable (h_ab : a > b)
variable (h_b0 : b > 0)

def ellipse (a b x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

theorem tangent_line (h_el : ellipse a b x₀ y₀) : 
  (x₀ * x / a^2) + (y₀ * y / b^2) = 1 :=
sorry

end tangent_line_l170_170844


namespace part_I_part_II_l170_170067

noncomputable def f (x : ℝ) : ℝ := 2 * |x + 1| + |2 * x - 1|

theorem part_I (x : ℝ) : 
  (f x > f 1) ↔ (x < -3/2 ∨ x > 1) :=
by
  -- We leave the proof as sorry to indicate it needs to be filled in
  sorry

theorem part_II (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x : ℝ, f x ≥ 1/m + 1/n) → m + n ≥ 4/3 :=
by
  -- We leave the proof as sorry to indicate it needs to be filled in
  sorry

end part_I_part_II_l170_170067


namespace erased_angle_is_97_l170_170869

theorem erased_angle_is_97 (n : ℕ) (h1 : 3 ≤ n) (h2 : (n - 2) * 180 = 1703 + x) : 
  1800 - 1703 = 97 :=
by sorry

end erased_angle_is_97_l170_170869


namespace find_multiple_sales_l170_170095

theorem find_multiple_sales 
  (A : ℝ) 
  (M : ℝ)
  (h : M * A = 0.35294117647058826 * (11 * A + M * A)) 
  : M = 6 :=
sorry

end find_multiple_sales_l170_170095


namespace cube_minus_self_divisible_by_10_l170_170311

theorem cube_minus_self_divisible_by_10 (k : ℤ) : 10 ∣ ((5 * k) ^ 3 - 5 * k) :=
by sorry

end cube_minus_self_divisible_by_10_l170_170311


namespace joan_total_spent_on_clothing_l170_170948

theorem joan_total_spent_on_clothing :
  let shorts_cost := 15.00
  let jacket_cost := 14.82
  let shirt_cost := 12.51
  let shoes_cost := 21.67
  let hat_cost := 8.75
  let belt_cost := 6.34
  shorts_cost + jacket_cost + shirt_cost + shoes_cost + hat_cost + belt_cost = 79.09 :=
by
  sorry

end joan_total_spent_on_clothing_l170_170948


namespace compute_fraction_l170_170815

theorem compute_fraction (x y z : ℝ) (h : x ≠ y ∧ y ≠ z ∧ z ≠ x) (sum_eq : x + y + z = 12) :
  (xy + yz + zx) / (x^2 + y^2 + z^2) = (144 - (x^2 + y^2 + z^2)) / (2 * (x^2 + y^2 + z^2)) :=
by
  sorry

end compute_fraction_l170_170815


namespace probability_four_heads_l170_170754

-- Definitions for use in the conditions
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def biased_coin (h : ℚ) (n k : ℕ) : ℚ :=
  binomial_coefficient n k * (h ^ k) * ((1 - h) ^ (n - k))

-- Condition: probability of getting heads exactly twice is equal to getting heads exactly three times.
def condition (h : ℚ) : Prop :=
  biased_coin h 5 2 = biased_coin h 5 3

-- Theorem to be proven: probability of getting heads exactly four times out of five is 5/32.
theorem probability_four_heads (h : ℚ) (cond : condition h) : biased_coin h 5 4 = 5 / 32 :=
by
  sorry

end probability_four_heads_l170_170754


namespace value_range_of_f_l170_170248

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 then 2 * x - x^2
  else if -4 ≤ x ∧ x < 0 then x^2 + 6 * x
  else 0

theorem value_range_of_f : Set.range f = {y : ℝ | -9 ≤ y ∧ y ≤ 1} :=
by
  sorry

end value_range_of_f_l170_170248


namespace unique_combinations_bathing_suits_l170_170072

theorem unique_combinations_bathing_suits
  (men_styles : ℕ) (men_sizes : ℕ) (men_colors : ℕ)
  (women_styles : ℕ) (women_sizes : ℕ) (women_colors : ℕ)
  (h_men_styles : men_styles = 5) (h_men_sizes : men_sizes = 3) (h_men_colors : men_colors = 4)
  (h_women_styles : women_styles = 4) (h_women_sizes : women_sizes = 4) (h_women_colors : women_colors = 5) :
  men_styles * men_sizes * men_colors + women_styles * women_sizes * women_colors = 140 :=
by
  sorry

end unique_combinations_bathing_suits_l170_170072


namespace number_of_young_teachers_selected_l170_170017

theorem number_of_young_teachers_selected 
  (total_teachers elderly_teachers middle_aged_teachers young_teachers sample_size : ℕ)
  (h_total: total_teachers = 200)
  (h_elderly: elderly_teachers = 25)
  (h_middle_aged: middle_aged_teachers = 75)
  (h_young: young_teachers = 100)
  (h_sample_size: sample_size = 40)
  : young_teachers * sample_size / total_teachers = 20 := 
sorry

end number_of_young_teachers_selected_l170_170017


namespace triangle_area_is_120_l170_170852

-- Define the triangle sides
def a : ℕ := 10
def b : ℕ := 24
def c : ℕ := 26

-- Define a function to calculate the area of a right-angled triangle
noncomputable def right_triangle_area (a b : ℕ) : ℕ := (a * b) / 2

-- Statement to prove the area of the triangle
theorem triangle_area_is_120 : right_triangle_area 10 24 = 120 :=
by
  sorry

end triangle_area_is_120_l170_170852


namespace rhombus_other_diagonal_l170_170744

theorem rhombus_other_diagonal (d1 : ℝ) (area : ℝ) (d2 : ℝ) 
  (h1 : d1 = 50) 
  (h2 : area = 625) 
  (h3 : area = (d1 * d2) / 2) : 
  d2 = 25 :=
by
  sorry

end rhombus_other_diagonal_l170_170744


namespace max_red_socks_l170_170098

-- Define r (red socks), b (blue socks), t (total socks), with the given constraints
def socks_problem (r b t : ℕ) : Prop :=
  t = r + b ∧
  t ≤ 2023 ∧
  (2 * r * (r - 1) + 2 * b * (b - 1)) = 2 * 5 * t * (t - 1)

-- State the theorem that the maximum number of red socks is 990
theorem max_red_socks : ∃ r b t, socks_problem r b t ∧ r = 990 :=
sorry

end max_red_socks_l170_170098


namespace ellipse_slope_product_l170_170932

theorem ellipse_slope_product (x₀ y₀ : ℝ) (hp : x₀^2 / 4 + y₀^2 / 3 = 1) :
  (y₀ / (x₀ + 2)) * (y₀ / (x₀ - 2)) = -3 / 4 :=
by
  -- The proof is omitted.
  sorry

end ellipse_slope_product_l170_170932


namespace shell_count_l170_170122

theorem shell_count (initial_shells : ℕ) (ed_limpet : ℕ) (ed_oyster : ℕ) (ed_conch : ℕ) (jacob_extra : ℕ)
  (h1 : initial_shells = 2)
  (h2 : ed_limpet = 7) 
  (h3 : ed_oyster = 2) 
  (h4 : ed_conch = 4) 
  (h5 : jacob_extra = 2) : 
  (initial_shells + ed_limpet + ed_oyster + ed_conch + (ed_limpet + ed_oyster + ed_conch + jacob_extra)) = 30 := 
by 
  sorry

end shell_count_l170_170122


namespace sandys_average_price_l170_170706

noncomputable def average_price_per_book (priceA : ℝ) (discountA : ℝ) (booksA : ℕ) (priceB : ℝ) (discountB : ℝ) (booksB : ℕ) (conversion_rate : ℝ) : ℝ :=
  let costA := priceA / (1 - discountA)
  let priceB_in_usd := priceB / conversion_rate
  let costB := priceB_in_usd / (1 - discountB)
  let total_cost := costA + costB
  let total_books := booksA + booksB
  total_cost / total_books

theorem sandys_average_price :
  average_price_per_book 1380 0.15 65 900 0.10 55 0.85 = 23.33 :=
by
  sorry

end sandys_average_price_l170_170706


namespace rodney_probability_correct_guess_l170_170806

noncomputable def two_digit_integer (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

noncomputable def tens_digit (n : ℕ) : Prop :=
  (n / 10 = 7 ∨ n / 10 = 8 ∨ n / 10 = 9)

noncomputable def units_digit_even (n : ℕ) : Prop :=
  (n % 10 = 0 ∨ n % 10 = 2 ∨ n % 10 = 4 ∨ n % 10 = 6 ∨ n % 10 = 8)

noncomputable def greater_than_seventy_five (n : ℕ) : Prop := n > 75

theorem rodney_probability_correct_guess (n : ℕ) :
  two_digit_integer n →
  tens_digit n →
  units_digit_even n →
  greater_than_seventy_five n →
  (∃ m, m = 1 / 12) :=
sorry

end rodney_probability_correct_guess_l170_170806


namespace solve_for_x_l170_170006

theorem solve_for_x (x : ℝ) (h : (2 + x) / (4 + x) = (3 + x) / (7 + x)) : x = -1 :=
by {
  sorry
}

end solve_for_x_l170_170006


namespace smaller_circle_circumference_l170_170921

theorem smaller_circle_circumference (r r2 : ℝ) : 
  (60:ℝ) / 360 * 2 * Real.pi * r = 8 →
  r = 24 / Real.pi →
  1 / 4 * (24 / Real.pi)^2 = (24 / Real.pi - 2 * r2) * (24 / Real.pi) →
  2 * Real.pi * r2 = 36 :=
  by
    intros h1 h2 h3
    sorry

end smaller_circle_circumference_l170_170921


namespace sum_even_1_to_200_l170_170498

open Nat

/-- The sum of all even numbers from 1 to 200 is 10100. --/
theorem sum_even_1_to_200 :
  let first_term := 2
  let last_term := 200
  let common_diff := 2
  let n := (last_term - first_term) / common_diff + 1
  let sum := n / 2 * (first_term + last_term)
  sum = 10100 :=
by
  let first_term := 2
  let last_term := 200
  let common_diff := 2
  let n := (last_term - first_term) / common_diff + 1
  let sum := n / 2 * (first_term + last_term)
  show sum = 10100
  sorry

end sum_even_1_to_200_l170_170498


namespace find_a_squared_plus_b_squared_and_ab_l170_170736

theorem find_a_squared_plus_b_squared_and_ab (a b : ℝ) 
  (h1 : (a + b) ^ 2 = 7)
  (h2 : (a - b) ^ 2 = 3) : 
  a^2 + b^2 = 5 ∧ a * b = 1 :=
by 
  sorry

end find_a_squared_plus_b_squared_and_ab_l170_170736


namespace total_output_equal_at_20_l170_170883

noncomputable def total_output_A (x : ℕ) : ℕ :=
  200 + 20 * x

noncomputable def total_output_B (x : ℕ) : ℕ :=
  30 * x

theorem total_output_equal_at_20 :
  total_output_A 20 = total_output_B 20 :=
by
  sorry

end total_output_equal_at_20_l170_170883


namespace Linda_total_sales_l170_170586

theorem Linda_total_sales (necklaces_sold : ℕ) (rings_sold : ℕ) 
    (necklace_price : ℕ) (ring_price : ℕ) 
    (total_sales : ℕ) : 
    necklaces_sold = 4 → 
    rings_sold = 8 → 
    necklace_price = 12 → 
    ring_price = 4 → 
    total_sales = necklaces_sold * necklace_price + rings_sold * ring_price → 
    total_sales = 80 :=
by
  intros H1 H2 H3 H4 H5
  sorry

end Linda_total_sales_l170_170586


namespace shaded_area_of_octagon_l170_170082

def side_length := 12
def octagon_area := 288

theorem shaded_area_of_octagon (s : ℕ) (h0 : s = side_length):
  (2 * s * s - 2 * s * s / 2) * 2 / 2 = octagon_area :=
by
  skip
  sorry

end shaded_area_of_octagon_l170_170082


namespace inner_prod_sum_real_inner_prod_modulus_l170_170666

open Complex

-- Define the given mathematical expressions
noncomputable def pair (α β : ℂ) : ℝ := (1 / 4) * (norm (α + β) ^ 2 - norm (α - β) ^ 2)

noncomputable def inner_prod (α β : ℂ) : ℂ := pair α β + Complex.I * pair α (Complex.I * β)

-- Prove the given mathematical statements

-- 1. Prove that ⟨α, β⟩ + ⟨β, α⟩ is a real number
theorem inner_prod_sum_real (α β : ℂ) : (inner_prod α β + inner_prod β α).im = 0 := sorry

-- 2. Prove that |⟨α, β⟩| = |α| * |β|
theorem inner_prod_modulus (α β : ℂ) : Complex.abs (inner_prod α β) = Complex.abs α * Complex.abs β := sorry

end inner_prod_sum_real_inner_prod_modulus_l170_170666


namespace ball_total_distance_l170_170791

def total_distance (initial_height : ℝ) (bounce_factor : ℝ) (bounces : ℕ) : ℝ :=
  let rec loop (height : ℝ) (total : ℝ) (remaining : ℕ) : ℝ :=
    if remaining = 0 then total
    else loop (height * bounce_factor) (total + height + height * bounce_factor) (remaining - 1)
  loop initial_height 0 bounces

theorem ball_total_distance : 
  total_distance 20 0.8 4 = 106.272 :=
by
  sorry

end ball_total_distance_l170_170791


namespace solution_set_ineq_l170_170196

theorem solution_set_ineq (x : ℝ) : x^2 - 2 * abs x - 15 > 0 ↔ x < -5 ∨ x > 5 :=
sorry

end solution_set_ineq_l170_170196


namespace sum_of_remainders_l170_170842

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 13) : (n % 4) + (n % 5) = 4 := 
by {
  -- proof omitted
  sorry
}

end sum_of_remainders_l170_170842


namespace sanity_proof_l170_170047

-- Define the characters and their sanity status as propositions
variables (Griffin QuasiTurtle Lobster : Prop)

-- Conditions
axiom Lobster_thinks : (Griffin ∧ ¬QuasiTurtle ∧ ¬Lobster) ∨ (¬Griffin ∧ QuasiTurtle ∧ ¬Lobster) ∨ (¬Griffin ∧ ¬QuasiTurtle ∧ Lobster)
axiom QuasiTurtle_thinks : Griffin

-- Statement to prove
theorem sanity_proof : ¬Griffin ∧ ¬QuasiTurtle ∧ ¬Lobster :=
by {
  sorry
}

end sanity_proof_l170_170047


namespace sum_of_integers_sqrt_485_l170_170164

theorem sum_of_integers_sqrt_485 (x y : ℕ) (h1 : x^2 + y^2 = 245) (h2 : x * y = 120) : x + y = Real.sqrt 485 :=
sorry

end sum_of_integers_sqrt_485_l170_170164


namespace common_ratio_neg_two_l170_170582

theorem common_ratio_neg_two (a : ℕ → ℝ) (q : ℝ) 
  (h : ∀ n, a (n + 1) = a n * q)
  (H : 8 * a 2 + a 5 = 0) : 
  q = -2 :=
sorry

end common_ratio_neg_two_l170_170582


namespace tiling_condition_l170_170765

theorem tiling_condition (a b n : ℕ) : 
  (∃ f : ℕ → ℕ × ℕ, ∀ i < (a * b) / n, (f i).fst < a ∧ (f i).snd < b) ↔ (n ∣ a ∨ n ∣ b) :=
sorry

end tiling_condition_l170_170765


namespace number_of_women_more_than_men_l170_170002

variables (M W : ℕ)

def ratio_condition : Prop := M * 3 = 2 * W
def total_condition : Prop := M + W = 20
def correct_answer : Prop := W - M = 4

theorem number_of_women_more_than_men 
  (h1 : ratio_condition M W) 
  (h2 : total_condition M W) : 
  correct_answer M W := 
by 
  sorry

end number_of_women_more_than_men_l170_170002


namespace smaller_sphere_radius_l170_170366

theorem smaller_sphere_radius (R x : ℝ) (h1 : (4/3) * Real.pi * R^3 = (4/3) * Real.pi * x^3 + (4/3) * Real.pi * (2 * x)^3) 
  (h2 : ∀ r₁ r₂ : ℝ, r₁ / r₂ = 1 / 2 → r₁ = x ∧ r₂ = 2 * x) : x = R / 3 :=
by 
  sorry

end smaller_sphere_radius_l170_170366


namespace a_9_value_l170_170989

-- Define the sequence and its sum of the first n terms
def S (n : ℕ) : ℕ := n^2

-- Define the terms of the sequence
def a (n : ℕ) : ℕ := if n = 1 then S 1 else S n - S (n - 1)

-- The main statement to be proved
theorem a_9_value : a 9 = 17 :=
by
  sorry

end a_9_value_l170_170989


namespace intersection_of_complements_l170_170150

open Set

theorem intersection_of_complements (U : Set ℕ) (A B : Set ℕ)
  (hU : U = {1,2,3,4,5,6,7,8})
  (hA : A = {3,4,5})
  (hB : B = {1,3,6}) :
  (U \ A) ∩ (U \ B) = {2,7,8} := by
  rw [hU, hA, hB]
  sorry

end intersection_of_complements_l170_170150


namespace average_gas_mileage_round_trip_l170_170916

theorem average_gas_mileage_round_trip
  (d : ℝ) (ms mr : ℝ)
  (h1 : d = 150)
  (h2 : ms = 35)
  (h3 : mr = 15) :
  (2 * d) / ((d / ms) + (d / mr)) = 21 :=
by
  sorry

end average_gas_mileage_round_trip_l170_170916


namespace index_card_area_l170_170569

theorem index_card_area (a b : ℕ) (h1 : a = 5) (h2 : b = 7) (h3 : (a - 2) * b = 21) : (a * (b - 1)) = 30 := by
  sorry

end index_card_area_l170_170569


namespace xiao_wang_program_output_l170_170368

theorem xiao_wang_program_output (n : ℕ) (h : n = 8) : (n : ℝ) / (n^2 + 1) = 8 / 65 := by
  sorry

end xiao_wang_program_output_l170_170368


namespace movie_tickets_l170_170198

theorem movie_tickets (r h : ℕ) (h1 : r = 25) (h2 : h = 3 * r + 18) : h = 93 :=
by
  sorry

end movie_tickets_l170_170198


namespace num_spacy_subsets_15_l170_170820

def spacy_subsets (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | 1     => 2
  | 2     => 3
  | 3     => 4
  | n + 1 => spacy_subsets n + if n ≥ 2 then spacy_subsets (n - 2) else 1

theorem num_spacy_subsets_15 : spacy_subsets 15 = 406 := by
  sorry

end num_spacy_subsets_15_l170_170820


namespace complex_imaginary_unit_sum_l170_170544

theorem complex_imaginary_unit_sum (i : ℂ) (h : i^2 = -1) : i + i^2 + i^3 = -1 := 
by sorry

end complex_imaginary_unit_sum_l170_170544


namespace volume_is_correct_l170_170233

def condition1 (x y z : ℝ) : Prop := abs (x + 2 * y + 3 * z) + abs (x + 2 * y - 3 * z) ≤ 18
def condition2 (x y z : ℝ) : Prop := x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0
def region (x y z : ℝ) : Prop := condition1 x y z ∧ condition2 x y z

noncomputable def volume_of_region : ℝ :=
  60.75 -- the result obtained from the calculation steps

theorem volume_is_correct : ∀ (x y z : ℝ), region x y z → volume_of_region = 60.75 :=
by
  sorry

end volume_is_correct_l170_170233


namespace nonoverlapping_unit_squares_in_figure_50_l170_170698

def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

theorem nonoverlapping_unit_squares_in_figure_50 :
  f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 → f 50 = 7651 :=
by
  sorry

end nonoverlapping_unit_squares_in_figure_50_l170_170698


namespace stones_equally_distributed_l170_170627

theorem stones_equally_distributed (n k : ℕ) 
    (h : ∃ piles : Fin n → ℕ, (∀ i j, 2 * piles i + piles j = k * n)) :
  ∃ m : ℕ, k = 2^m :=
by
  sorry

end stones_equally_distributed_l170_170627


namespace count_odd_numbers_300_600_l170_170810

theorem count_odd_numbers_300_600 : ∃ n : ℕ, n = 149 ∧ ∀ k : ℕ, (301 ≤ k ∧ k < 600 ∧ k % 2 = 1) ↔ (301 ≤ k ∧ k < 600 ∧ k % 2 = 1 ∧ k - 301 < n * 2) :=
by {
  sorry
}

end count_odd_numbers_300_600_l170_170810


namespace equation_solutions_l170_170784

theorem equation_solutions :
  ∀ x y : ℤ, x^2 + x * y + y^2 + x + y - 5 = 0 → (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = -3) ∨ (x = -3 ∧ y = 1) :=
by
  intro x y h
  sorry

end equation_solutions_l170_170784


namespace statement_B_statement_D_l170_170338

variable {a b c d : ℝ}

theorem statement_B (h1 : a > b) (h2 : b > 0) (h3 : c < 0) : (c / a) > (c / b) := 
by sorry

theorem statement_D (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) : (a * c) < (b * d) := 
by sorry

end statement_B_statement_D_l170_170338


namespace max_capacity_tank_l170_170054

-- Definitions of the conditions
def water_loss_1 := 32000 * 5
def water_loss_2 := 10000 * 10
def total_loss := water_loss_1 + water_loss_2
def water_added := 40000 * 3
def missing_water := 140000

-- Definition of the maximum capacity
def max_capacity := total_loss + water_added + missing_water

-- The theorem to prove
theorem max_capacity_tank : max_capacity = 520000 := by
  sorry

end max_capacity_tank_l170_170054


namespace arithmetic_seq_middle_term_l170_170448

theorem arithmetic_seq_middle_term (a1 a3 y : ℤ) (h1 : a1 = 3^2) (h2 : a3 = 3^4)
    (h3 : y = (a1 + a3) / 2) : y = 45 :=
by
  rw [h1, h2] at h3
  simp at h3
  exact h3

end arithmetic_seq_middle_term_l170_170448


namespace expected_score_of_basketball_player_l170_170284

theorem expected_score_of_basketball_player :
  let p_inside : ℝ := 0.7
  let p_outside : ℝ := 0.4
  let attempts_inside : ℕ := 10
  let attempts_outside : ℕ := 5
  let points_inside : ℕ := 2
  let points_outside : ℕ := 3
  let E_inside : ℝ := attempts_inside * p_inside * points_inside
  let E_outside : ℝ := attempts_outside * p_outside * points_outside
  E_inside + E_outside = 20 :=
by
  sorry

end expected_score_of_basketball_player_l170_170284


namespace added_water_correct_l170_170154

theorem added_water_correct (initial_fullness : ℝ) (final_fullness : ℝ) (capacity : ℝ) (initial_amount : ℝ) (final_amount : ℝ) (added_water : ℝ) :
    initial_fullness = 0.30 →
    final_fullness = 3/4 →
    capacity = 60 →
    initial_amount = initial_fullness * capacity →
    final_amount = final_fullness * capacity →
    added_water = final_amount - initial_amount →
    added_water = 27 :=
by
  intros
  -- Insert the proof here
  sorry

end added_water_correct_l170_170154


namespace min_value_PQ_l170_170971

variable (t : ℝ) (x y : ℝ)

-- Parametric equations of line l
def line_l : Prop := (x = 4 * t - 1) ∧ (y = 3 * t - 3 / 2)

-- Polar equation of circle C
def polar_eq_circle_c (ρ θ : ℝ) : Prop :=
  ρ^2 = 2 * Real.sqrt 2 * ρ * Real.sin (θ - Real.pi / 4)

-- General equation of line l
def general_eq_line_l (x y : ℝ) : Prop := 3 * x - 4 * y = 3

-- Rectangular equation of circle C
def rectangular_eq_circle_c (x y : ℝ) : Prop :=
  (x + 1)^2 + (y - 1)^2 = 2

-- Definition of the condition where P is on line l
def p_on_line_l (x y : ℝ) : Prop := ∃ t : ℝ, line_l t x y

-- Minimum value of |PQ|
theorem min_value_PQ :
  p_on_line_l x y →
  general_eq_line_l x y →
  rectangular_eq_circle_c x y →
  ∃ d : ℝ, d = Real.sqrt 2 :=
by intros; sorry

end min_value_PQ_l170_170971


namespace range_of_c_l170_170761

theorem range_of_c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hab : a + b = a * b) (habc : a + b + c = a * b * c) : 1 < c ∧ c ≤ 4 / 3 :=
by
  sorry

end range_of_c_l170_170761


namespace third_podcast_length_correct_l170_170412

def first_podcast_length : ℕ := 45
def fourth_podcast_length : ℕ := 60
def next_podcast_length : ℕ := 60
def total_drive_time : ℕ := 360

def second_podcast_length := 2 * first_podcast_length

def total_time_other_than_third := first_podcast_length + second_podcast_length + fourth_podcast_length + next_podcast_length

theorem third_podcast_length_correct :
  total_drive_time - total_time_other_than_third = 105 := by
  -- Proof goes here
  sorry

end third_podcast_length_correct_l170_170412


namespace number_of_males_choosing_malt_l170_170701

-- Definitions of conditions as provided in the problem
def total_males : Nat := 10
def total_females : Nat := 16

def total_cheerleaders : Nat := total_males + total_females

def females_choosing_malt : Nat := 8
def females_choosing_coke : Nat := total_females - females_choosing_malt

noncomputable def cheerleaders_choosing_malt (M_males : Nat) : Nat :=
  females_choosing_malt + M_males

noncomputable def cheerleaders_choosing_coke (M_males : Nat) : Nat :=
  females_choosing_coke + (total_males - M_males)

theorem number_of_males_choosing_malt : ∃ (M_males : Nat), 
  cheerleaders_choosing_malt M_males = 2 * cheerleaders_choosing_coke M_males ∧
  cheerleaders_choosing_malt M_males + cheerleaders_choosing_coke M_males = total_cheerleaders ∧
  M_males = 9 := 
by
  sorry

end number_of_males_choosing_malt_l170_170701


namespace find_C_l170_170896

noncomputable def h (C D : ℝ) (x : ℝ) : ℝ := 2 * C * x - 3 * D ^ 2
def k (D : ℝ) (x : ℝ) := D * x

theorem find_C (C D : ℝ) (h_eq : h C D (k D 2) = 0) (hD : D ≠ 0) : C = 3 * D / 4 :=
by
  unfold h k at h_eq
  sorry

end find_C_l170_170896


namespace mike_pull_ups_per_week_l170_170171

theorem mike_pull_ups_per_week (pull_ups_per_entry entries_per_day days_per_week : ℕ)
  (h1 : pull_ups_per_entry = 2)
  (h2 : entries_per_day = 5)
  (h3 : days_per_week = 7)
  : pull_ups_per_entry * entries_per_day * days_per_week = 70 := 
by
  sorry

end mike_pull_ups_per_week_l170_170171


namespace three_digit_with_five_is_divisible_by_five_l170_170434

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def ends_in_five (n : ℕ) : Prop := n % 10 = 5

def divisible_by_five (n : ℕ) : Prop := n % 5 = 0

theorem three_digit_with_five_is_divisible_by_five (M : ℕ) :
  is_three_digit M ∧ ends_in_five M → divisible_by_five M :=
by
  sorry

end three_digit_with_five_is_divisible_by_five_l170_170434


namespace plane_perpendicular_l170_170644

-- Define types for lines and planes
axiom Line : Type
axiom Plane : Type

-- Define the relationships between lines and planes
axiom Parallel (l : Line) (p : Plane) : Prop
axiom Perpendicular (l : Line) (p : Plane) : Prop
axiom PlanePerpendicular (p1 p2 : Plane) : Prop

-- The setting conditions
variables (c : Line) (α β : Plane)

-- The given conditions
axiom c_perpendicular_β : Perpendicular c β
axiom c_parallel_α : Parallel c α

-- The proof goal (without the proof body)
theorem plane_perpendicular : PlanePerpendicular α β :=
by
  sorry

end plane_perpendicular_l170_170644


namespace real_part_of_z_l170_170608

variable (z : ℂ) (a : ℝ)

noncomputable def condition1 : Prop := z / (2 + (a : ℂ) * Complex.I) = 2 / (1 + Complex.I)
noncomputable def condition2 : Prop := z.im = -3

theorem real_part_of_z (h1 : condition1 z a) (h2 : condition2 z) : z.re = 1 := sorry

end real_part_of_z_l170_170608


namespace relationship_between_a_b_c_l170_170870

-- Define the given parabola function
def parabola (x : ℝ) (k : ℝ) : ℝ := -(x - 2)^2 + k

-- Define the points A, B, C with their respective coordinates and expressions on the parabola
variables {a b c k : ℝ}

-- Conditions: Points lie on the parabola
theorem relationship_between_a_b_c (hA : a = parabola (-2) k)
                                  (hB : b = parabola (-1) k)
                                  (hC : c = parabola 3 k) :
  a < b ∧ b < c :=
by
  sorry

end relationship_between_a_b_c_l170_170870


namespace find_a_plus_b_l170_170086

noncomputable def f (a b : ℝ) (x : ℝ) := a * x + b

noncomputable def h (x : ℝ) := 3 * x + 2

theorem find_a_plus_b (a b : ℝ) (x : ℝ) (h_condition : ∀ x, h (f a b x) = 4 * x - 1) :
  a + b = 1 / 3 := 
by
  sorry

end find_a_plus_b_l170_170086


namespace calculate_expression_value_l170_170253

theorem calculate_expression_value :
  (23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2) = 288 :=
by
  sorry

end calculate_expression_value_l170_170253


namespace alice_paper_cranes_l170_170301

theorem alice_paper_cranes (T : ℕ)
  (h1 : T / 2 - T / 10 = 400) : T = 1000 :=
sorry

end alice_paper_cranes_l170_170301


namespace pat_oj_consumption_l170_170272

def initial_oj : ℚ := 3 / 4
def alex_fraction : ℚ := 1 / 2
def pat_fraction : ℚ := 1 / 3

theorem pat_oj_consumption : pat_fraction * (initial_oj * (1 - alex_fraction)) = 1 / 8 := by
  -- This will be the proof part which can be filled later
  sorry

end pat_oj_consumption_l170_170272


namespace wire_length_ratio_l170_170517

noncomputable def total_wire_length_bonnie (pieces : Nat) (length_per_piece : Nat) := 
  pieces * length_per_piece

noncomputable def volume_of_cube (edge_length : Nat) := 
  edge_length ^ 3

noncomputable def wire_length_roark_per_cube (edges_per_cube : Nat) (length_per_edge : Nat) (num_cubes : Nat) :=
  edges_per_cube * length_per_edge * num_cubes

theorem wire_length_ratio : 
  let bonnie_pieces := 12
  let bonnie_length_per_piece := 8
  let bonnie_edge_length := 8
  let roark_length_per_edge := 2
  let roark_edges_per_cube := 12
  let bonnie_wire_length := total_wire_length_bonnie bonnie_pieces bonnie_length_per_piece
  let bonnie_cube_volume := volume_of_cube bonnie_edge_length
  let roark_num_cubes := bonnie_cube_volume
  let roark_wire_length := wire_length_roark_per_cube roark_edges_per_cube roark_length_per_edge roark_num_cubes
  bonnie_wire_length / roark_wire_length = 1 / 128 :=
by
  sorry

end wire_length_ratio_l170_170517


namespace line_through_point_and_area_l170_170893

theorem line_through_point_and_area (a b : ℝ) (x y : ℝ) 
  (hx : x = -2) (hy : y = 2) 
  (h_area : 1/2 * |a * b| = 1): 
  (2 * x + y + 2 = 0 ∨ x + 2 * y - 2 = 0) :=
  sorry

end line_through_point_and_area_l170_170893


namespace neg_q_sufficient_not_necc_neg_p_l170_170766

variable (p q : Prop)

theorem neg_q_sufficient_not_necc_neg_p (hp: p → q) (hnpq: ¬(q → p)) : (¬q → ¬p) ∧ (¬(¬p → ¬q)) :=
by
  sorry

end neg_q_sufficient_not_necc_neg_p_l170_170766


namespace quadratic_function_inequality_l170_170812

variable (a x x₁ x₂ : ℝ)

def f (x : ℝ) := a * x^2 + 2 * a * x + 4

theorem quadratic_function_inequality
  (h₀ : 0 < a) (h₁ : a < 3)
  (h₂ : x₁ + x₂ = 0)
  (h₃ : x₁ < x₂) :
  f a x₁ < f a x₂ := 
sorry

end quadratic_function_inequality_l170_170812


namespace warriors_truth_tellers_l170_170236

/-- There are 33 warriors. Each warrior is either a truth-teller or a liar, 
    with only one favorite weapon: a sword, a spear, an axe, or a bow. 
    They were asked four questions, and the number of "Yes" answers to the 
    questions are 13, 15, 20, and 27 respectively. Prove that the number of 
    warriors who always tell the truth is 12. -/
theorem warriors_truth_tellers
  (warriors : ℕ) (truth_tellers : ℕ)
  (yes_to_sword : ℕ) (yes_to_spear : ℕ)
  (yes_to_axe : ℕ) (yes_to_bow : ℕ)
  (h1 : warriors = 33)
  (h2 : yes_to_sword = 13)
  (h3 : yes_to_spear = 15)
  (h4 : yes_to_axe = 20)
  (h5 : yes_to_bow = 27)
  (h6 : yes_to_sword + yes_to_spear + yes_to_axe + yes_to_bow = 75) :
  truth_tellers = 12 := by
  -- Proof will be here
  sorry

end warriors_truth_tellers_l170_170236


namespace heaviest_weight_is_aq3_l170_170020

variable (a q : ℝ) (h : 0 < a) (hq : 1 < q)

theorem heaviest_weight_is_aq3 :
  let w1 := a
  let w2 := a * q
  let w3 := a * q^2
  let w4 := a * q^3
  w4 > w3 ∧ w4 > w2 ∧ w4 > w1 ∧ w1 + w4 > w2 + w3 :=
by
  sorry

end heaviest_weight_is_aq3_l170_170020


namespace students_with_screws_neq_bolts_l170_170190

-- Let's define the main entities
def total_students : ℕ := 40
def nails_neq_bolts : ℕ := 15
def screws_eq_nails : ℕ := 10

-- Main theorem statement
theorem students_with_screws_neq_bolts (total : ℕ) (neq_nails_bolts : ℕ) (eq_screws_nails : ℕ) :
  total = 40 → neq_nails_bolts = 15 → eq_screws_nails = 10 → ∃ k, k ≥ 15 ∧ k ≤ 40 - eq_screws_nails - neq_nails_bolts := 
by
  intros
  sorry

end students_with_screws_neq_bolts_l170_170190


namespace range_of_k_l170_170945

theorem range_of_k (k : ℝ) : 
  (∀ x : ℝ, k * x ^ 2 + 2 * k * x + 3 ≠ 0) ↔ (0 ≤ k ∧ k < 3) :=
by sorry

end range_of_k_l170_170945


namespace smallest_y_value_l170_170900

theorem smallest_y_value : ∃ y : ℝ, 2 * y ^ 2 + 7 * y + 3 = 5 ∧ (∀ y' : ℝ, 2 * y' ^ 2 + 7 * y' + 3 = 5 → y ≤ y') := sorry

end smallest_y_value_l170_170900


namespace minimum_value_of_PQ_l170_170529

theorem minimum_value_of_PQ {x y : ℝ} (P : ℝ × ℝ) (h₁ : (P.1 - 3)^2 + (P.2 - 4)^2 > 4)
  (h₂ : ∀ Q : ℝ × ℝ, (Q.1 - 3)^2 + (Q.2 - 4)^2 = 4 → (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (P.1)^2 + (P.2)^2) :
  ∃ PQ_min : ℝ, PQ_min = 17/2 := by
  sorry

end minimum_value_of_PQ_l170_170529


namespace average_price_of_tshirts_l170_170454

theorem average_price_of_tshirts
  (A : ℝ)
  (total_cost_seven_remaining : ℝ := 7 * 505)
  (total_cost_three_returned : ℝ := 3 * 673)
  (total_cost_eight : ℝ := total_cost_seven_remaining + 673) -- since (1 t-shirt with price is included in the total)
  (total_cost_eight_eq : total_cost_eight = 8 * A) :
  A = 526 :=
by sorry

end average_price_of_tshirts_l170_170454


namespace ratio_solution_l170_170533

theorem ratio_solution (x : ℚ) : (1 : ℚ) / 3 = 5 / 3 / x → x = 5 := 
by
  intro h
  sorry

end ratio_solution_l170_170533


namespace pay_docked_per_lateness_l170_170790

variable (hourly_rate : ℤ) (work_hours : ℤ) (times_late : ℕ) (actual_pay : ℤ) 

theorem pay_docked_per_lateness (h_rate : hourly_rate = 30) 
                                (w_hours : work_hours = 18) 
                                (t_late : times_late = 3) 
                                (a_pay : actual_pay = 525) :
                                (hourly_rate * work_hours - actual_pay) / times_late = 5 :=
by
  sorry

end pay_docked_per_lateness_l170_170790


namespace parallel_lines_implies_value_of_m_l170_170430

theorem parallel_lines_implies_value_of_m :
  ∀ (m : ℝ), (∀ (x y : ℝ), 3 * x + 2 * y - 2 = 0) ∧ (∀ (x y : ℝ), (2 * m - 1) * x + m * y + 1 = 0) → 
  m = 2 := 
by
  sorry

end parallel_lines_implies_value_of_m_l170_170430


namespace arithmetic_sequence_sum_l170_170908

theorem arithmetic_sequence_sum :
  ∃ x y z d : ℝ, 
  d = (31 - 4) / 5 ∧ 
  x = 4 + d ∧ 
  y = x + d ∧ 
  z = 16 + d ∧ 
  (x + y + z) = 45.6 :=
by
  sorry

end arithmetic_sequence_sum_l170_170908


namespace evaluate_expression_l170_170148

theorem evaluate_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end evaluate_expression_l170_170148


namespace vet_fees_cat_result_l170_170612

-- Given conditions
def vet_fees_dog : ℕ := 15
def families_dogs : ℕ := 8
def families_cats : ℕ := 3
def vet_donation : ℕ := 53

-- Mathematical equivalency in Lean
noncomputable def vet_fees_cat (C : ℕ) : Prop :=
  (1 / 3 : ℚ) * (families_dogs * vet_fees_dog + families_cats * C) = vet_donation

-- Prove the vet fees for cats are 13 using above conditions
theorem vet_fees_cat_result : ∃ (C : ℕ), vet_fees_cat C ∧ C = 13 :=
by {
  use 13,
  sorry
}

end vet_fees_cat_result_l170_170612


namespace percentage_answered_first_correctly_l170_170524

-- Defining the given conditions
def percentage_answered_second_correctly : ℝ := 0.25
def percentage_answered_neither_correctly : ℝ := 0.20
def percentage_answered_both_correctly : ℝ := 0.20

-- Lean statement for the proof problem
theorem percentage_answered_first_correctly :
  ∃ a : ℝ, a + percentage_answered_second_correctly - percentage_answered_both_correctly = 0.80 ∧ a = 0.75 := by
  sorry

end percentage_answered_first_correctly_l170_170524


namespace Barkley_bones_l170_170099

def bones_per_month : ℕ := 10
def months : ℕ := 5
def bones_received : ℕ := bones_per_month * months
def bones_buried : ℕ := 42
def bones_available : ℕ := 8

theorem Barkley_bones :
  bones_received - bones_buried = bones_available := by sorry

end Barkley_bones_l170_170099


namespace average_age_increase_l170_170257

theorem average_age_increase 
  (n : Nat) 
  (a : ℕ) 
  (b : ℕ) 
  (total_students : Nat)
  (avg_age_9 : ℕ) 
  (tenth_age : ℕ) 
  (original_total_age : Nat)
  (new_total_age : Nat)
  (new_avg_age : ℕ)
  (age_increase : ℕ) 
  (h1 : n = 9) 
  (h2 : avg_age_9 = 8) 
  (h3 : tenth_age = 28)
  (h4 : total_students = 10)
  (h5 : original_total_age = n * avg_age_9) 
  (h6 : new_total_age = original_total_age + tenth_age)
  (h7 : new_avg_age = new_total_age / total_students)
  (h8 : age_increase = new_avg_age - avg_age_9) :
  age_increase = 2 := 
by 
  sorry

end average_age_increase_l170_170257


namespace alice_walk_time_l170_170030

theorem alice_walk_time (bob_time : ℝ) 
  (bob_distance : ℝ) 
  (alice_distance1 : ℝ) 
  (alice_distance2 : ℝ) 
  (time_ratio : ℝ) 
  (expected_alice_time : ℝ) :
  bob_time = 36 →
  bob_distance = 6 →
  alice_distance1 = 4 →
  alice_distance2 = 7 →
  time_ratio = 1 / 3 →
  expected_alice_time = 21 →
  (expected_alice_time = alice_distance2 / (alice_distance1 / (bob_time * time_ratio))) := 
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h3, h5]
  have h_speed : ℝ := alice_distance1 / (bob_time * time_ratio)
  rw [h4, h6]
  linarith [h_speed]

end alice_walk_time_l170_170030


namespace infinite_geometric_series_sum_l170_170369

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 2
  let r := (1 : ℝ) / 2
  (a + a * r + a * r^2 + a * r^3 + ∑' n : ℕ, a * r^n) = 1 :=
by
  sorry

end infinite_geometric_series_sum_l170_170369


namespace find_value_of_a_l170_170897

theorem find_value_of_a (x a : ℝ) (h : 2 * x - a + 5 = 0) (h_x : x = -2) : a = 1 :=
by
  sorry

end find_value_of_a_l170_170897


namespace ellipse_product_l170_170140

noncomputable def AB_CD_product (a b c : ℝ) (h1 : a^2 - b^2 = c^2) (h2 : a + b = 14) : ℝ :=
  2 * a * 2 * b

-- The main statement
theorem ellipse_product (c : ℝ) (h_c : c = 8) (h_diameter : 6 = 6)
  (a b : ℝ) (h1 : a^2 - b^2 = c^2) (h2 : a + b = 14) :
  AB_CD_product a b c h1 h2 = 175 := sorry

end ellipse_product_l170_170140


namespace marbles_remaining_correct_l170_170926

-- Define the number of marbles Chris has
def marbles_chris : ℕ := 12

-- Define the number of marbles Ryan has
def marbles_ryan : ℕ := 28

-- Define the total number of marbles in the pile
def total_marbles : ℕ := marbles_chris + marbles_ryan

-- Define the number of marbles each person takes away from the pile
def marbles_taken_each : ℕ := total_marbles / 4

-- Define the total number of marbles taken away
def total_marbles_taken : ℕ := 2 * marbles_taken_each

-- Define the number of marbles remaining in the pile
def marbles_remaining : ℕ := total_marbles - total_marbles_taken

theorem marbles_remaining_correct : marbles_remaining = 20 := by
  sorry

end marbles_remaining_correct_l170_170926


namespace satisfies_conditions_l170_170406

theorem satisfies_conditions : ∃ (n : ℤ), 0 ≤ n ∧ n < 31 ∧ -250 % 31 = n % 31 ∧ n = 29 :=
by
  sorry

end satisfies_conditions_l170_170406


namespace car_r_speed_l170_170464

variable (v : ℝ)

theorem car_r_speed (h1 : (300 / v - 2 = 300 / (v + 10))) : v = 30 := 
sorry

end car_r_speed_l170_170464


namespace least_remaining_marbles_l170_170313

/-- 
There are 60 identical marbles forming a tetrahedral pile.
The formula for the number of marbles in a tetrahedral pile up to the k-th level is given by:
∑_(i=1)^k (i * (i + 1)) / 6 = k * (k + 1) * (k + 2) / 6.

We must show that the least number of remaining marbles when 60 marbles are used to form the pile is 4.
-/
theorem least_remaining_marbles : ∃ k : ℕ, (60 - k * (k + 1) * (k + 2) / 6) = 4 :=
by
  sorry

end least_remaining_marbles_l170_170313


namespace universal_proposition_l170_170822

def is_multiple_of_two (x : ℕ) : Prop :=
  ∃ k : ℕ, x = 2 * k

def is_even (x : ℕ) : Prop :=
  ∃ k : ℕ, x = 2 * k

theorem universal_proposition : 
  (∀ x : ℕ, is_multiple_of_two x → is_even x) :=
by
  sorry

end universal_proposition_l170_170822


namespace roots_cubic_identity_l170_170709

theorem roots_cubic_identity (p q r s : ℝ) (h1 : r + s = p) (h2 : r * s = -q) (h3 : ∀ x : ℝ, x^2 - p*x - q = 0 → (x = r ∨ x = s)) :
  r^3 + s^3 = p^3 + 3*p*q := by
  sorry

end roots_cubic_identity_l170_170709


namespace boat_speed_l170_170362

theorem boat_speed (v : ℝ) (h1 : 5 + v = 30) : v = 25 :=
by 
  -- Solve for the speed of the second boat
  sorry

end boat_speed_l170_170362


namespace no_nat_n_divisible_by_169_l170_170654

theorem no_nat_n_divisible_by_169 (n : ℕ) : ¬ ∃ k : ℕ, n^2 + 5 * n + 16 = 169 * k :=
sorry

end no_nat_n_divisible_by_169_l170_170654


namespace mutually_exclusive_pairs_l170_170350

-- Define the events based on the conditions
def event_two_red_one_white (bag : List String) (drawn : List String) : Prop :=
  drawn.length = 3 ∧ (drawn.count "red" = 2 ∧ drawn.count "white" = 1)

def event_one_red_two_white (bag : List String) (drawn : List String) : Prop :=
  drawn.length = 3 ∧ (drawn.count "red" = 1 ∧ drawn.count "white" = 2)

def event_three_red (bag : List String) (drawn : List String) : Prop :=
  drawn.length = 3 ∧ drawn.count "red" = 3

def event_at_least_one_white (bag : List String) (drawn : List String) : Prop :=
  drawn.length = 3 ∧ 1 ≤ drawn.count "white"

def event_three_white (bag : List String) (drawn : List String) : Prop :=
  drawn.length = 3 ∧ drawn.count "white" = 3

-- Define mutually exclusive property
def mutually_exclusive (A B : List String → List String → Prop) (bag : List String) : Prop :=
  ∀ drawn, A bag drawn → ¬ B bag drawn

-- Define the main theorem statement
theorem mutually_exclusive_pairs (bag : List String) (condition : bag = ["red", "red", "red", "red", "red", "white", "white", "white", "white", "white"]) :
  mutually_exclusive event_three_red event_at_least_one_white bag ∧
  mutually_exclusive event_three_red event_three_white bag :=
by
  sorry

end mutually_exclusive_pairs_l170_170350


namespace average_mark_of_excluded_students_l170_170794

theorem average_mark_of_excluded_students
  (N : ℕ) (A A_remaining : ℕ)
  (num_excluded : ℕ)
  (hN : N = 9)
  (hA : A = 60)
  (hA_remaining : A_remaining = 80)
  (h_excluded : num_excluded = 5) :
  (N * A - (N - num_excluded) * A_remaining) / num_excluded = 44 :=
by
  sorry

end average_mark_of_excluded_students_l170_170794


namespace class3_total_score_l170_170839

theorem class3_total_score 
  (total_points : ℕ)
  (class1_score class2_score class3_score : ℕ)
  (class1_places class2_places class3_places : ℕ)
  (total_places : ℕ)
  (points_1st  points_2nd  points_3rd : ℕ)
  (h1 : total_points = 27)
  (h2 : class1_score = class2_score)
  (h3 : 2 * class1_places = class2_places)
  (h4 : class1_places + class2_places + class3_places = total_places)
  (h5 : 3 * points_1st + 3 * points_2nd + 3 * points_3rd = total_points)
  (h6 : total_places = 9)
  (h7 : points_1st = 5)
  (h8 : points_2nd = 3)
  (h9 : points_3rd = 1) :
  class3_score = 7 :=
sorry

end class3_total_score_l170_170839


namespace solve_recurrence_relation_l170_170381

def recurrence_relation (a : ℕ → ℤ) : Prop :=
  ∀ n ≥ 3, a n = 3 * a (n - 1) - 3 * a (n - 2) + a (n - 3) + 24 * n - 6

def initial_conditions (a : ℕ → ℤ) : Prop :=
  a 0 = -4 ∧ a 1 = -2 ∧ a 2 = 2

def explicit_solution (n : ℕ) : ℤ :=
  -4 + 17 * n - 21 * n^2 + 5 * n^3 + n^4

theorem solve_recurrence_relation :
  ∀ (a : ℕ → ℤ),
    recurrence_relation a →
    initial_conditions a →
    ∀ n, a n = explicit_solution n := by
  intros a h_recur h_init n
  sorry

end solve_recurrence_relation_l170_170381


namespace factor_square_difference_l170_170263

theorem factor_square_difference (t : ℝ) : t^2 - 121 = (t - 11) * (t + 11) := 
  sorry

end factor_square_difference_l170_170263


namespace bowling_ball_weight_l170_170635

theorem bowling_ball_weight (b c : ℕ) (h1 : 8 * b = 4 * c) (h2 : 3 * c = 108) : b = 18 := 
by 
  sorry

end bowling_ball_weight_l170_170635


namespace cos_60_eq_sqrt3_div_2_l170_170059

theorem cos_60_eq_sqrt3_div_2 : Real.cos (60 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry

end cos_60_eq_sqrt3_div_2_l170_170059


namespace zero_of_function_l170_170335

theorem zero_of_function : ∃ x : ℝ, (x + 1)^2 = 0 :=
by
  use -1
  sorry

end zero_of_function_l170_170335


namespace fraction_subtraction_simplify_l170_170358

noncomputable def fraction_subtraction : ℚ :=
  (12 / 25) - (3 / 75)

theorem fraction_subtraction_simplify : fraction_subtraction = (11 / 25) :=
  by
    -- Proof goes here
    sorry

end fraction_subtraction_simplify_l170_170358


namespace find_other_number_l170_170487

theorem find_other_number (a b : ℕ) (h1 : Nat.gcd a b = 24) (h2 : Nat.lcm a b = 5040) (h3 : a = 240) : b = 504 :=
by {
  sorry
}

end find_other_number_l170_170487


namespace det_A_zero_l170_170685

theorem det_A_zero
  (x1 x2 x3 y1 y2 y3 : ℝ)
  (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℝ)
  (h1 : a11 = Real.sin (x1 - y1)) (h2 : a12 = Real.sin (x1 - y2)) (h3 : a13 = Real.sin (x1 - y3))
  (h4 : a21 = Real.sin (x2 - y1)) (h5 : a22 = Real.sin (x2 - y2)) (h6 : a23 = Real.sin (x2 - y3))
  (h7 : a31 = Real.sin (x3 - y1)) (h8 : a32 = Real.sin (x3 - y2)) (h9 : a33 = Real.sin (x3 - y3)) :
  (Matrix.det ![![a11, a12, a13], ![a21, a22, a23], ![a31, a32, a33]]) = 0 := sorry

end det_A_zero_l170_170685


namespace jane_mistake_l170_170976

theorem jane_mistake (x y z : ℤ) (h1 : x - y + z = 15) (h2 : x - y - z = 7) : x - y = 11 :=
by sorry

end jane_mistake_l170_170976


namespace volume_of_sphere_l170_170005

noncomputable def cuboid_volume (a b c : ℝ) := a * b * c

noncomputable def sphere_volume (r : ℝ) := (4/3) * Real.pi * r^3

theorem volume_of_sphere
  (a b c : ℝ) 
  (sphere_radius : ℝ)
  (h1 : a = 1)
  (h2 : b = Real.sqrt 3)
  (h3 : c = 2)
  (h4 : sphere_radius = Real.sqrt (a^2 + b^2 + c^2) / 2)
  : sphere_volume sphere_radius = (8 * Real.sqrt 2 / 3) * Real.pi := 
by
  sorry

end volume_of_sphere_l170_170005


namespace speed_of_man_in_still_water_l170_170478

-- Define the conditions as given in step (a)
axiom conditions :
  ∃ (v_m v_s : ℝ),
    (40 / 5 = v_m + v_s) ∧
    (30 / 5 = v_m - v_s)

-- State the theorem that proves the speed of the man in still water
theorem speed_of_man_in_still_water : ∃ v_m : ℝ, v_m = 7 :=
by
  obtain ⟨v_m, v_s, h1, h2⟩ := conditions
  have h3 : v_m + v_s = 8 := by sorry
  have h4 : v_m - v_s = 6 := by sorry
  have h5 : 2 * v_m = 14 := by sorry
  exact ⟨7, by linarith⟩

end speed_of_man_in_still_water_l170_170478


namespace no_one_is_always_largest_l170_170543

theorem no_one_is_always_largest (a b c d : ℝ) :
  a - 2 = b + 3 ∧ a - 2 = c * 2 ∧ a - 2 = d + 5 →
  ∀ x, (x = a ∨ x = b ∨ x = c ∨ x = d) → (x ≤ c ∨ x ≤ a) :=
by
  -- The proof requires assuming the conditions and showing that no variable is always the largest.
  intro h cond
  sorry

end no_one_is_always_largest_l170_170543


namespace largest_common_value_under_800_l170_170476

-- Let's define the problem conditions as arithmetic sequences
def sequence1 (a : ℤ) : Prop := ∃ n : ℤ, a = 4 + 5 * n
def sequence2 (a : ℤ) : Prop := ∃ m : ℤ, a = 7 + 8 * m

-- Now we state the theorem that the largest common value less than 800 is 799
theorem largest_common_value_under_800 : 
  ∃ a : ℤ, sequence1 a ∧ sequence2 a ∧ a < 800 ∧ ∀ b : ℤ, sequence1 b ∧ sequence2 b ∧ b < 800 → b ≤ a :=
sorry

end largest_common_value_under_800_l170_170476


namespace problem_statement_l170_170179

noncomputable def calculateValue (n : ℕ) : ℕ :=
  Nat.choose (3 * n) (38 - n) + Nat.choose (n + 21) (3 * n)

theorem problem_statement : calculateValue 10 = 466 := by
  sorry

end problem_statement_l170_170179


namespace cylinder_volume_l170_170689

theorem cylinder_volume (r h : ℝ) (hr : r = 5) (hh : h = 10) :
    π * r^2 * h = 250 * π := by
  -- We leave the actual proof as sorry for now
  sorry

end cylinder_volume_l170_170689


namespace annual_production_2010_l170_170308

-- Defining the parameters
variables (a x : ℝ)

-- Define the growth formula
def annual_growth (initial : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial * (1 + rate)^years

-- The statement we need to prove
theorem annual_production_2010 :
  annual_growth a x 5 = a * (1 + x) ^ 5 :=
by
  sorry

end annual_production_2010_l170_170308


namespace quadratic_distinct_real_roots_l170_170558

theorem quadratic_distinct_real_roots (k : ℝ) : k < 1 / 2 ∧ k ≠ 0 ↔ (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (k * x1^2 - 2 * x1 + 2 = 0) ∧ (k * x2^2 - 2 * x2 + 2 = 0)) := 
by 
  sorry

end quadratic_distinct_real_roots_l170_170558


namespace range_of_f_l170_170156

noncomputable def f (x : ℝ) : ℝ := 4^x - 2^(x + 1) + 2

theorem range_of_f (h : ∀ x : ℝ, x ≤ 1) : (f '' {x : ℝ | x ≤ 1}) = {y : ℝ | 1 ≤ y ∧ y ≤ 2} :=
by
  sorry

end range_of_f_l170_170156


namespace complement_union_M_N_eq_16_l170_170942

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the subsets M and N
def M : Set ℕ := {2, 3, 4}
def N : Set ℕ := {4, 5}

-- Define the union of M and N
def unionMN : Set ℕ := M ∪ N

-- Define the complement of M ∪ N in U
def complementUnionMN : Set ℕ := U \ unionMN

-- State the theorem that the complement is {1, 6}
theorem complement_union_M_N_eq_16 : complementUnionMN = {1, 6} := by
  sorry

end complement_union_M_N_eq_16_l170_170942


namespace hall_length_l170_170289

theorem hall_length (L h : ℝ) (width volume : ℝ) 
  (h_width : width = 6) 
  (h_volume : L * width * h = 108) 
  (h_area : 12 * L = 2 * L * h + 12 * h) : 
  L = 6 := 
  sorry

end hall_length_l170_170289


namespace find_value_l170_170045

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodicity_condition : ∀ x : ℝ, f (2 + x) = f (-x)
axiom value_at_half : f (1/2) = 1/2

theorem find_value : f (2023 / 2) = 1/2 := by
  sorry

end find_value_l170_170045


namespace stratified_sampling_sophomores_l170_170901

theorem stratified_sampling_sophomores
  (freshmen : ℕ) (sophomores : ℕ) (juniors : ℕ) (total_selected : ℕ)
  (H_freshmen : freshmen = 550) (H_sophomores : sophomores = 700) (H_juniors : juniors = 750) (H_total_selected : total_selected = 100) :
  sophomores * total_selected / (freshmen + sophomores + juniors) = 35 :=
by
  sorry

end stratified_sampling_sophomores_l170_170901


namespace fish_weight_l170_170155

-- Definitions of weights
variable (T B H : ℝ)

-- Given conditions
def cond1 : Prop := T = 9
def cond2 : Prop := H = T + (1/2) * B
def cond3 : Prop := B = H + T

-- Theorem to prove
theorem fish_weight (h1 : cond1 T) (h2 : cond2 T B H) (h3 : cond3 T B H) :
  T + B + H = 72 :=
by
  sorry

end fish_weight_l170_170155


namespace student_tickets_second_day_l170_170540

variable (S T x: ℕ)

theorem student_tickets_second_day (hT : T = 9) (h_eq1 : 4 * S + 3 * T = 79) (h_eq2 : 12 * S + x * T = 246) : x = 10 :=
by
  sorry

end student_tickets_second_day_l170_170540


namespace present_age_ratio_l170_170979

-- Define the conditions as functions in Lean.
def age_difference (M R : ℝ) : Prop := M - R = 7.5
def future_age_ratio (M R : ℝ) : Prop := (R + 10) / (M + 10) = 2 / 3

-- Define the goal as a proof problem in Lean.
theorem present_age_ratio (M R : ℝ) 
  (h1 : age_difference M R) 
  (h2 : future_age_ratio M R) : 
  R / M = 2 / 5 := 
by 
  sorry  -- Proof to be completed

end present_age_ratio_l170_170979


namespace kenneth_earnings_l170_170193

theorem kenneth_earnings (E : ℝ) (h1 : E - 0.1 * E = 405) : E = 450 :=
sorry

end kenneth_earnings_l170_170193


namespace max_a_value_l170_170022

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b

theorem max_a_value :
  (∀ x : ℝ, ∃ y : ℝ, f y a b = f x a b + y) → a ≤ 1/2 :=
by
  sorry

end max_a_value_l170_170022


namespace sum_of_digits_is_13_l170_170496

theorem sum_of_digits_is_13:
  ∀ (a b c d : ℕ),
  b + c = 10 ∧
  c + d = 1 ∧
  a + d = 2 →
  a + b + c + d = 13 :=
by {
  sorry
}

end sum_of_digits_is_13_l170_170496


namespace hamburger_combinations_l170_170336

def number_of_condiments := 8
def condiment_combinations := 2 ^ number_of_condiments
def number_of_meat_patties := 4
def total_hamburgers := number_of_meat_patties * condiment_combinations

theorem hamburger_combinations :
  total_hamburgers = 1024 :=
by
  sorry

end hamburger_combinations_l170_170336


namespace find_area_of_oblique_triangle_l170_170680

noncomputable def area_triangle (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1 / 2 * a * b * Real.sin C

theorem find_area_of_oblique_triangle
  (A B C a b c : ℝ)
  (h1 : c = Real.sqrt 21)
  (h2 : c * Real.sin A = Real.sqrt 3 * a * Real.cos C)
  (h3 : Real.sin C + Real.sin (B - A) = 5 * Real.sin (2 * A))
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum_ABC : A + B + C = Real.pi)
  (habc_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (tri_angle_pos : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) :
  area_triangle a b c A B C = 5 * Real.sqrt 3 / 4 := 
sorry

end find_area_of_oblique_triangle_l170_170680


namespace platinum_earrings_percentage_l170_170876

theorem platinum_earrings_percentage
  (rings_percentage ornaments_percentage : ℝ)
  (rings_percentage_eq : rings_percentage = 0.30)
  (earrings_percentage_eq : ornaments_percentage - rings_percentage = 0.70)
  (platinum_earrings_percentage : ℝ)
  (platinum_earrings_percentage_eq : platinum_earrings_percentage = 0.70) :
  ornaments_percentage * platinum_earrings_percentage = 0.49 :=
by 
  have earrings_percentage := 0.70
  have ornaments_percentage := 0.70
  sorry

end platinum_earrings_percentage_l170_170876


namespace cedar_vs_pine_height_cedar_vs_birch_height_l170_170483

-- Define the heights as rational numbers
def pine_tree_height := 14 + 1/4
def birch_tree_height := 18 + 1/2
def cedar_tree_height := 20 + 5/8

-- Theorem to prove the height differences
theorem cedar_vs_pine_height :
  cedar_tree_height - pine_tree_height = 6 + 3/8 :=
by
  sorry

theorem cedar_vs_birch_height :
  cedar_tree_height - birch_tree_height = 2 + 1/8 :=
by
  sorry

end cedar_vs_pine_height_cedar_vs_birch_height_l170_170483


namespace efficiency_difference_l170_170016

variables (Rp Rq : ℚ)

-- Given conditions
def p_rate := Rp = 1 / 21
def combined_rate := Rp + Rq = 1 / 11

-- Define the percentage efficiency difference
def percentage_difference := (Rp - Rq) / Rq * 100

-- Main statement to prove
theorem efficiency_difference : 
  p_rate Rp ∧ 
  combined_rate Rp Rq → 
  percentage_difference Rp Rq = 10 :=
sorry

end efficiency_difference_l170_170016


namespace minimum_value_expression_l170_170703

theorem minimum_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  (∃ x : ℝ, x = 9 ∧ ∀ y : ℝ, y = (1 / a^2 - 1) * (1 / b^2 - 1) → x ≤ y) :=
sorry

end minimum_value_expression_l170_170703


namespace time_to_pass_platform_l170_170223

-- Definitions
def train_length : ℕ := 1400
def platform_length : ℕ := 700
def time_to_cross_tree : ℕ := 100
def train_speed : ℕ := train_length / time_to_cross_tree
def total_distance : ℕ := train_length + platform_length

-- Prove that the time to pass the platform is 150 seconds
theorem time_to_pass_platform : total_distance / train_speed = 150 :=
by
  sorry

end time_to_pass_platform_l170_170223


namespace housing_price_growth_l170_170237

theorem housing_price_growth (x : ℝ) (h₁ : (5500 : ℝ) > 0) (h₂ : (7000 : ℝ) > 0) :
  5500 * (1 + x) ^ 2 = 7000 := 
sorry

end housing_price_growth_l170_170237


namespace octavio_can_reach_3_pow_2023_l170_170800

theorem octavio_can_reach_3_pow_2023 (n : ℤ) (hn : n ≥ 1) :
  ∃ (steps : ℕ → ℤ), steps 0 = n ∧ (∀ k, steps (k + 1) = 3 * (steps k)) ∧
  steps 2023 = 3 ^ 2023 :=
by
  sorry

end octavio_can_reach_3_pow_2023_l170_170800


namespace quadratic_equation_factored_form_l170_170040

theorem quadratic_equation_factored_form : 
  ∀ x : ℝ, x^2 - 6 * x - 6 = 0 ↔ (x - 3)^2 = 15 := 
by 
  sorry

end quadratic_equation_factored_form_l170_170040


namespace necessary_but_not_sufficient_l170_170385

def p (a : ℝ) : Prop := (a - 1) * (a - 2) = 0
def q (a : ℝ) : Prop := a = 1

theorem necessary_but_not_sufficient (a : ℝ) : 
  (q a → p a) ∧ (p a → q a → False) :=
by
  sorry

end necessary_but_not_sufficient_l170_170385


namespace chlorine_moles_l170_170751

theorem chlorine_moles (methane_used chlorine_used chloromethane_formed : ℕ)
  (h_combined_methane : methane_used = 3)
  (h_formed_chloromethane : chloromethane_formed = 3)
  (balanced_eq : methane_used = chloromethane_formed) :
  chlorine_used = 3 :=
by
  have h : chlorine_used = methane_used := by sorry
  rw [h_combined_methane] at h
  exact h

end chlorine_moles_l170_170751


namespace trigonometric_expression_l170_170141

open Real

theorem trigonometric_expression (α β : ℝ) (h : cos α ^ 2 = cos β ^ 2) :
  (sin β ^ 2 / sin α + cos β ^ 2 / cos α = sin α + cos α ∨ sin β ^ 2 / sin α + cos β ^ 2 / cos α = -sin α + cos α) :=
sorry

end trigonometric_expression_l170_170141


namespace expected_winnings_is_0_25_l170_170409

def prob_heads : ℚ := 3 / 8
def prob_tails : ℚ := 1 / 4
def prob_edge  : ℚ := 1 / 8
def prob_disappear : ℚ := 1 / 4

def winnings_heads : ℚ := 2
def winnings_tails : ℚ := 5
def winnings_edge  : ℚ := -2
def winnings_disappear : ℚ := -6

def expected_winnings : ℚ := 
  prob_heads * winnings_heads +
  prob_tails * winnings_tails +
  prob_edge  * winnings_edge +
  prob_disappear * winnings_disappear

theorem expected_winnings_is_0_25 : expected_winnings = 0.25 := by
  sorry

end expected_winnings_is_0_25_l170_170409


namespace find_parallel_line_l170_170750

/-- 
Given a line l with equation 3x - 2y + 1 = 0 and a point A(1,1).
Find the equation of a line that passes through A and is parallel to l.
-/
theorem find_parallel_line (a b c : ℝ) (p_x p_y : ℝ) 
    (h₁ : 3 * p_x - 2 * p_y + c = 0) 
    (h₂ : p_x = 1 ∧ p_y = 1)
    (h₃ : a = 3 ∧ b = -2) :
    3 * x - 2 * y - 1 = 0 := 
by 
  sorry

end find_parallel_line_l170_170750


namespace tangent_line_to_curve_perpendicular_l170_170113

noncomputable def perpendicular_tangent_line (x y : ℝ) : Prop :=
  y = x^4 ∧ (4*x - y - 3 = 0)

theorem tangent_line_to_curve_perpendicular {x y : ℝ} (h : y = x^4 ∧ (4*x - y - 3 = 0)) :
  ∃ (x y : ℝ), (x+4*y-8=0) ∧ (4*x - y - 3 = 0) :=
by
  sorry

end tangent_line_to_curve_perpendicular_l170_170113


namespace b_share_of_earnings_l170_170056

-- Definitions derived from conditions
def work_rate_a := 1 / 6
def work_rate_b := 1 / 8
def work_rate_c := 1 / 12
def total_earnings := 1170

-- Mathematically equivalent Lean statement
theorem b_share_of_earnings : 
  (work_rate_b / (work_rate_a + work_rate_b + work_rate_c)) * total_earnings = 390 := 
by
  sorry

end b_share_of_earnings_l170_170056


namespace smallest_positive_integer_modulo_l170_170063

theorem smallest_positive_integer_modulo {n : ℕ} (h : 19 * n ≡ 546 [MOD 13]) : n = 11 := by
  sorry

end smallest_positive_integer_modulo_l170_170063


namespace relationship_between_a_and_b_l170_170052

-- Define the given linear function
def linear_function (k : ℝ) (x : ℝ) : ℝ := (k^2 + 1) * x + 1

-- Formalize the relationship between a and b given the points and the linear function
theorem relationship_between_a_and_b (a b k : ℝ) 
  (hP : a = linear_function k (-4))
  (hQ : b = linear_function k 2) :
  a < b := 
by
  sorry  -- Proof to be filled in by the theorem prover

end relationship_between_a_and_b_l170_170052


namespace smallest_delightful_integer_l170_170468

-- Definition of "delightful" integer
def is_delightful (B : ℤ) : Prop :=
  ∃ (n : ℕ), (n > 0) ∧ ((n + 1) * (2 * B + n)) / 2 = 3050

-- Proving the smallest delightful integer
theorem smallest_delightful_integer : ∃ (B : ℤ), is_delightful B ∧ ∀ (B' : ℤ), is_delightful B' → B ≤ B' :=
  sorry

end smallest_delightful_integer_l170_170468


namespace not_solution_of_equation_l170_170768

theorem not_solution_of_equation (a : ℝ) (h : a ≠ 0) : ¬ (a^2 * 1^2 + (a + 1) * 1 + 1 = 0) :=
by {
  sorry
}

end not_solution_of_equation_l170_170768


namespace sequence_divisible_by_13_l170_170129

theorem sequence_divisible_by_13 (n : ℕ) (h : n ≤ 1000) : 
  ∃ m, m = 165 ∧ ∀ k, 1 ≤ k ∧ k ≤ m → (10^(6*k) + 1) % 13 = 0 := 
sorry

end sequence_divisible_by_13_l170_170129


namespace distance_from_point_to_focus_l170_170980

noncomputable def point_on_parabola (P : ℝ × ℝ) (y : ℝ) : Prop :=
  y^2 = 16 * P.1 ∧ (P.2 = y ∨ P.2 = -y)

noncomputable def parabola_focus : ℝ × ℝ :=
  (4, 0)

theorem distance_from_point_to_focus
  (P : ℝ × ℝ) (y : ℝ)
  (h1 : point_on_parabola P y)
  (h2 : dist P (0, P.2) = 12) :
  dist P parabola_focus = 13 :=
sorry

end distance_from_point_to_focus_l170_170980


namespace sequence_fifth_term_l170_170474

theorem sequence_fifth_term (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : a 2 = 2)
    (h₃ : ∀ n > 2, a n = a (n-1) + a (n-2)) : a 5 = 8 :=
sorry

end sequence_fifth_term_l170_170474


namespace intersection_distance_zero_l170_170484

noncomputable def A : Type := ℝ × ℝ

def P : A := (2, 0)

def line_intersects_parabola (x y : ℝ) : Prop :=
  y - 2 * x + 5 = 0 ∧ y^2 = 3 * x + 4

def distance (p1 p2 : A) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem intersection_distance_zero :
  ∀ (A1 A2 : A),
  line_intersects_parabola A1.1 A1.2 ∧ line_intersects_parabola A2.1 A2.2 →
  (abs (distance A1 P - distance A2 P) = 0) :=
sorry

end intersection_distance_zero_l170_170484


namespace area_perimeter_quadratic_l170_170225

theorem area_perimeter_quadratic (a x y : ℝ) (h1 : x = 4 * a) (h2 : y = a^2) : y = (x / 4)^2 :=
by sorry

end area_perimeter_quadratic_l170_170225


namespace number_of_books_l170_170461

theorem number_of_books (Maddie Luisa Amy Noah : ℕ)
  (H1 : Maddie = 15)
  (H2 : Luisa = 18)
  (H3 : Amy + Luisa = Maddie + 9)
  (H4 : Noah = Amy / 3)
  : Amy + Noah = 8 :=
sorry

end number_of_books_l170_170461


namespace sufficient_but_not_necessary_l170_170401

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, (0 < x ∧ x < 2) → (x < 2)) ∧ ¬(∀ x : ℝ, (0 < x ∧ x < 2) ↔ (x < 2)) :=
sorry

end sufficient_but_not_necessary_l170_170401


namespace prime_quadruples_unique_l170_170220

noncomputable def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → (m = 1 ∨ m = n)

theorem prime_quadruples_unique (p q r n : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r) (hn : n > 0)
  (h_eq : p^2 = q^2 + r^n) :
  (p, q, r, n) = (3, 2, 5, 1) ∨ (p, q, r, n) = (5, 3, 2, 4) :=
by
  sorry

end prime_quadruples_unique_l170_170220


namespace eddie_rate_l170_170954

variables (hours_sam hours_eddie rate_sam total_crates rate_eddie : ℕ)

def sam_conditions :=
  hours_sam = 6 ∧ rate_sam = 60

def eddie_conditions :=
  hours_eddie = 4 ∧ total_crates = hours_sam * rate_sam

theorem eddie_rate (hs : sam_conditions hours_sam rate_sam)
                   (he : eddie_conditions hours_sam hours_eddie rate_sam total_crates) :
  rate_eddie = 90 :=
by sorry

end eddie_rate_l170_170954


namespace percentage_of_childrens_books_l170_170058

/-- Conditions: 
- There are 160 books in total.
- 104 of them are for adults.
Prove that the percentage of books intended for children is 35%. --/
theorem percentage_of_childrens_books (total_books : ℕ) (adult_books : ℕ) 
  (h_total : total_books = 160) (h_adult : adult_books = 104) :
  (160 - 104) / 160 * 100 = 35 := 
by {
  sorry -- Proof skipped
}

end percentage_of_childrens_books_l170_170058


namespace valid_inequalities_l170_170923

theorem valid_inequalities (a b c : ℝ) (h : 0 < c) 
  (h1 : b > c - b)
  (h2 : c > a)
  (h3 : c > b - a) :
  a < c / 2 ∧ b < a + c / 2 :=
by
  sorry

end valid_inequalities_l170_170923


namespace locus_of_midpoint_of_square_l170_170324

theorem locus_of_midpoint_of_square (a : ℝ) (x y : ℝ) (h1 : x^2 + y^2 = 4 * a^2) :
  (∃ X Y : ℝ, 2 * X = x ∧ 2 * Y = y ∧ X^2 + Y^2 = a^2) :=
by {
  -- No proof is required, so we use 'sorry' here
  sorry
}

end locus_of_midpoint_of_square_l170_170324


namespace time_saved_is_35_minutes_l170_170174

-- Define the speed and distances for each day
def monday_distance := 3
def wednesday_distance := 3
def friday_distance := 3
def sunday_distance := 4
def speed_monday := 6
def speed_wednesday := 4
def speed_friday := 5
def speed_sunday := 3
def speed_uniform := 5

-- Calculate the total time spent on the treadmill originally
def time_monday := monday_distance / speed_monday
def time_wednesday := wednesday_distance / speed_wednesday
def time_friday := friday_distance / speed_friday
def time_sunday := sunday_distance / speed_sunday
def total_time := time_monday + time_wednesday + time_friday + time_sunday

-- Calculate the total time if speed was uniformly 5 mph 
def total_distance := monday_distance + wednesday_distance + friday_distance + sunday_distance
def total_time_uniform := total_distance / speed_uniform

-- Time saved if walking at 5 mph every day
def time_saved := total_time - total_time_uniform

-- Convert time saved to minutes
def minutes_saved := time_saved * 60

theorem time_saved_is_35_minutes : minutes_saved = 35 := by
  sorry

end time_saved_is_35_minutes_l170_170174


namespace compare_2_pow_n_n_sq_l170_170611

theorem compare_2_pow_n_n_sq (n : ℕ) (h : n > 0) :
  (n = 1 → 2^n > n^2) ∧
  (n = 2 → 2^n = n^2) ∧
  (n = 3 → 2^n < n^2) ∧
  (n = 4 → 2^n = n^2) ∧
  (n ≥ 5 → 2^n > n^2) :=
by sorry

end compare_2_pow_n_n_sq_l170_170611


namespace compute_fraction_l170_170390

theorem compute_fraction : (1922^2 - 1913^2) / (1930^2 - 1905^2) = (9 : ℚ) / 25 := by
  sorry

end compute_fraction_l170_170390


namespace find_number_l170_170317

theorem find_number (n : ℝ) : (2629.76 / n = 528.0642570281125) → n = 4.979 :=
by
  intro h
  sorry

end find_number_l170_170317


namespace no_real_solutions_for_equation_l170_170334

theorem no_real_solutions_for_equation (x : ℝ) :
  y = 3 * x ∧ y = (x^3 - 8) / (x - 2) → false :=
by {
  sorry
}

end no_real_solutions_for_equation_l170_170334


namespace reflection_across_x_axis_l170_170590

def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem reflection_across_x_axis :
  reflect_x_axis (-2, -3) = (-2, 3) :=
by
  sorry

end reflection_across_x_axis_l170_170590


namespace second_reduction_percentage_l170_170840

variable (P : ℝ) -- Original price
variable (x : ℝ) -- Second reduction percentage

-- Condition 1: After a 25% reduction
def first_reduction (P : ℝ) : ℝ := 0.75 * P

-- Condition 3: Combined reduction equivalent to 47.5%
def combined_reduction (P : ℝ) : ℝ := 0.525 * P

-- Question: Given the conditions, prove that the second reduction is 0.3
theorem second_reduction_percentage (P : ℝ) (x : ℝ) :
  (1 - x) * first_reduction P = combined_reduction P → x = 0.3 :=
by
  intro h
  sorry

end second_reduction_percentage_l170_170840


namespace no_b_for_221_square_l170_170044

theorem no_b_for_221_square (b : ℕ) (h : b ≥ 3) :
  ¬ ∃ n : ℕ, 2 * b^2 + 2 * b + 1 = n^2 :=
by
  sorry

end no_b_for_221_square_l170_170044


namespace angies_age_l170_170986

variable (A : ℝ)

theorem angies_age (h : 2 * A + 4 = 20) : A = 8 :=
sorry

end angies_age_l170_170986


namespace rectangle_area_l170_170718

theorem rectangle_area (x : ℕ) (hx : x > 0)
  (h₁ : (x + 5) * 2 * (x + 10) = 3 * x * (x + 10))
  (h₂ : (x - 10) = x + 10 - 10) :
  x * (x + 10) = 200 :=
by {
  sorry
}

end rectangle_area_l170_170718


namespace ball_bounces_to_less_than_two_feet_l170_170048

noncomputable def bounce_height (n : ℕ) : ℝ := 20 * (3 / 4) ^ n

theorem ball_bounces_to_less_than_two_feet : ∃ k : ℕ, bounce_height k < 2 ∧ k = 7 :=
by
  -- We need to show that bounce_height k < 2 when k = 7
  sorry

end ball_bounces_to_less_than_two_feet_l170_170048


namespace count_ways_to_choose_one_person_l170_170525

theorem count_ways_to_choose_one_person (A B : ℕ) (hA : A = 3) (hB : B = 5) : A + B = 8 :=
by
  sorry

end count_ways_to_choose_one_person_l170_170525


namespace initial_average_l170_170834

variable (A : ℝ)
variables (nums : Fin 5 → ℝ)
variables (h_sum : 5 * A = nums 0 + nums 1 + nums 2 + nums 3 + nums 4)
variables (h_num : nums 0 = 12)
variables (h_new_avg : (5 * A + 12) / 5 = 9.2)

theorem initial_average :
  A = 6.8 :=
sorry

end initial_average_l170_170834


namespace sum_of_odd_integers_l170_170101

theorem sum_of_odd_integers (n : ℕ) (h1 : 4970 = n * (1 + n)) : (n ^ 2 = 4900) :=
by
  sorry

end sum_of_odd_integers_l170_170101


namespace ellipse_standard_equation_l170_170450

theorem ellipse_standard_equation
  (F : ℝ × ℝ)
  (e : ℝ)
  (eq1 : F = (0, 1))
  (eq2 : e = 1 / 2) :
  ∃ (a b : ℝ), a = 2 ∧ b ^ 2 = 3 ∧ (∀ x y : ℝ, (y ^ 2 / 4) + (x ^ 2 / 3) = 1) :=
by
  sorry

end ellipse_standard_equation_l170_170450


namespace evaluate_seventy_two_square_minus_twenty_four_square_l170_170328

theorem evaluate_seventy_two_square_minus_twenty_four_square :
  72 ^ 2 - 24 ^ 2 = 4608 := 
by {
  sorry
}

end evaluate_seventy_two_square_minus_twenty_four_square_l170_170328


namespace share_of_C_l170_170687

variable (A B C x : ℝ)

theorem share_of_C (hA : A = (2/3) * B) 
(hB : B = (1/4) * C) 
(hTotal : A + B + C = 595) 
(hC : C = x) : x = 420 :=
by
  -- Proof will follow here
  sorry

end share_of_C_l170_170687


namespace abc_correct_and_c_not_true_l170_170509

theorem abc_correct_and_c_not_true (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  a^2 > b^2 ∧ ab > b^2 ∧ (1/(a+b) > 1/a) ∧ ¬(1/a < 1/b) :=
  sorry

end abc_correct_and_c_not_true_l170_170509


namespace root_conditions_l170_170024

theorem root_conditions (m : ℝ) : (∃ a b : ℝ, a < 2 ∧ b > 2 ∧ a * b = -1 ∧ a + b = m) ↔ m > 3 / 2 := sorry

end root_conditions_l170_170024


namespace arithmetic_prog_leq_l170_170361

def t3 (s : List ℤ) : ℕ := 
  sorry -- Placeholder for function calculating number of 3-term arithmetic progressions

theorem arithmetic_prog_leq (a : List ℤ) (k : ℕ) (h_sorted : a = List.range k)
  : t3 a ≤ t3 (List.range k) :=
sorry -- Proof here

end arithmetic_prog_leq_l170_170361


namespace strips_overlap_area_l170_170965

theorem strips_overlap_area (L1 L2 AL AR S : ℝ) (hL1 : L1 = 9) (hL2 : L2 = 7) (hAL : AL = 27) (hAR : AR = 18) 
    (hrel : (AL + S) / (AR + S) = L1 / L2) : S = 13.5 := 
by
  sorry

end strips_overlap_area_l170_170965


namespace average_wage_per_day_l170_170888

variable (numMaleWorkers : ℕ) (wageMale : ℕ) (numFemaleWorkers : ℕ) (wageFemale : ℕ) (numChildWorkers : ℕ) (wageChild : ℕ)

theorem average_wage_per_day :
  numMaleWorkers = 20 →
  wageMale = 35 →
  numFemaleWorkers = 15 →
  wageFemale = 20 →
  numChildWorkers = 5 →
  wageChild = 8 →
  (20 * 35 + 15 * 20 + 5 * 8) / (20 + 15 + 5) = 26 :=
by
  intros
  -- Proof would follow here
  sorry

end average_wage_per_day_l170_170888


namespace evaluate_fg_sum_at_1_l170_170603

def f (x : ℚ) : ℚ := (4 * x^2 + 3 * x + 6) / (x^2 + 2 * x + 5)
def g (x : ℚ) : ℚ := x + 1

theorem evaluate_fg_sum_at_1 : f (g 1) + g (f 1) = 497 / 104 :=
by
  sorry

end evaluate_fg_sum_at_1_l170_170603


namespace ratio_of_B_to_C_l170_170571

-- Definitions based on conditions
def A := 40
def C := A + 20
def total := 220
def B := total - A - C

-- Theorem statement
theorem ratio_of_B_to_C : B / C = 2 :=
by
  -- Placeholder for proof
  sorry

end ratio_of_B_to_C_l170_170571


namespace square_garden_tiles_l170_170481

theorem square_garden_tiles (n : ℕ) (h : 2 * n - 1 = 25) : n^2 = 169 :=
by
  sorry

end square_garden_tiles_l170_170481


namespace mixed_operations_with_rationals_l170_170103

theorem mixed_operations_with_rationals :
  let a := 1 / 4
  let b := 1 / 2
  let c := 2 / 3
  (a - b + c) * (-12) = -8 :=
by
  sorry

end mixed_operations_with_rationals_l170_170103


namespace smallest_integer_in_range_l170_170530

-- Given conditions
def is_congruent_6 (n : ℕ) : Prop := n % 6 = 1
def is_congruent_7 (n : ℕ) : Prop := n % 7 = 1
def is_congruent_8 (n : ℕ) : Prop := n % 8 = 1

-- Lean statement for the proof problem
theorem smallest_integer_in_range :
  ∃ n : ℕ, (n > 1) ∧ is_congruent_6 n ∧ is_congruent_7 n ∧ is_congruent_8 n ∧ (n = 169) ∧ (120 ≤ n ∧ n < 210) :=
by
  sorry

end smallest_integer_in_range_l170_170530


namespace bucket_weight_full_l170_170846

theorem bucket_weight_full (c d : ℝ) (x y : ℝ) 
  (h1 : x + (1 / 3) * y = c) 
  (h2 : x + (3 / 4) * y = d) : 
  x + y = (-3 * c + 8 * d) / 5 :=
sorry

end bucket_weight_full_l170_170846


namespace number_B_expression_l170_170998

theorem number_B_expression (A B : ℝ) (h : A = B - (4/5) * B) : B = (A + B) / (4 / 5) :=
sorry

end number_B_expression_l170_170998


namespace count_positive_integers_l170_170422

theorem count_positive_integers (n : ℕ) (m : ℕ) :
  (∀ (k : ℕ), 1 ≤ k ∧ k < 100 ∧ (∃ (n : ℕ), n = 2 * k + 1 ∧ n < 200) 
  ∧ (∃ (m : ℤ), m = k * (k + 1) ∧ m % 5 = 0)) → 
  ∃ (cnt : ℕ), cnt = 20 :=
by
  sorry

end count_positive_integers_l170_170422


namespace total_distance_traveled_is_960_l170_170566

-- Definitions of conditions
def first_day_distance : ℝ := 100
def second_day_distance : ℝ := 3 * first_day_distance
def third_day_distance : ℝ := second_day_distance + 110
def fourth_day_distance : ℝ := 150

-- The total distance traveled in four days
def total_distance : ℝ := first_day_distance + second_day_distance + third_day_distance + fourth_day_distance

-- Theorem statement
theorem total_distance_traveled_is_960 :
  total_distance = 960 :=
by
  sorry

end total_distance_traveled_is_960_l170_170566


namespace xyz_cubic_expression_l170_170809

theorem xyz_cubic_expression (x y z a b c : ℝ) (h1 : x * y = a) (h2 : x * z = b) (h3 : y * z = c) (h4 : x ≠ 0) (h5 : y ≠ 0) (h6 : z ≠ 0) (h7 : a ≠ 0) (h8 : b ≠ 0) (h9 : c ≠ 0) :
  x^3 + y^3 + z^3 = (a^3 + b^3 + c^3) / (a * b * c) :=
by
  sorry

end xyz_cubic_expression_l170_170809


namespace complex_number_z_l170_170598

theorem complex_number_z (z : ℂ) (i : ℂ) (hz : i^2 = -1) (h : (1 - i)^2 / z = 1 + i) : z = -1 - i :=
by
  sorry

end complex_number_z_l170_170598


namespace find_values_l170_170375

theorem find_values (a b: ℝ) (h1: a > b) (h2: b > 1)
  (h3: Real.log a / Real.log b + Real.log b / Real.log a = 5 / 2)
  (h4: a^b = b^a) :
  a = 4 ∧ b = 2 := 
sorry

end find_values_l170_170375


namespace race_result_130m_l170_170161

theorem race_result_130m (d : ℕ) (t_a t_b: ℕ) (a_speed b_speed : ℚ) (d_a_t : ℚ) (d_b_t : ℚ) (distance_covered_by_B_in_20_secs : ℚ) :
  d = 130 →
  t_a = 20 →
  t_b = 25 →
  a_speed = (↑d) / t_a →
  b_speed = (↑d) / t_b →
  d_a_t = a_speed * t_a →
  d_b_t = b_speed * t_b →
  distance_covered_by_B_in_20_secs = b_speed * 20 →
  (d - distance_covered_by_B_in_20_secs = 26) :=
by
  sorry

end race_result_130m_l170_170161


namespace edith_novel_count_l170_170978

-- Definitions based on conditions
variables (N W : ℕ)

-- Conditions from the problem
def condition1 : Prop := N = W / 2
def condition2 : Prop := N + W = 240

-- Target statement
theorem edith_novel_count (N W : ℕ) (h1 : N = W / 2) (h2 : N + W = 240) : N = 80 :=
by
  sorry

end edith_novel_count_l170_170978


namespace probability_last_passenger_own_seat_is_half_l170_170920

open Classical

-- Define the behavior and probability question:

noncomputable def probability_last_passenger_own_seat (n : ℕ) : ℚ :=
  if n = 0 then 0 else 1 / 2

-- The main theorem stating the probability for an arbitrary number of passengers n
-- The theorem that needs to be proved:
theorem probability_last_passenger_own_seat_is_half (n : ℕ) (h : n > 0) : 
  probability_last_passenger_own_seat n = 1 / 2 :=
by sorry

end probability_last_passenger_own_seat_is_half_l170_170920


namespace point_in_quadrant_I_l170_170570

theorem point_in_quadrant_I (x y : ℝ) (h1 : 4 * x + 6 * y = 24) (h2 : y = x + 3) : x > 0 ∧ y > 0 :=
by sorry

end point_in_quadrant_I_l170_170570


namespace max_area_rectangle_l170_170864

theorem max_area_rectangle (l w : ℝ) 
  (h1 : 2 * l + 2 * w = 60) 
  (h2 : l - w = 10) : 
  l * w = 200 := 
by
  sorry

end max_area_rectangle_l170_170864


namespace bob_daily_work_hours_l170_170170

theorem bob_daily_work_hours
  (total_hours_in_month : ℕ)
  (days_per_week : ℕ)
  (weeks_per_month : ℕ)
  (total_working_days : ℕ)
  (daily_working_hours : ℕ)
  (h1 : total_hours_in_month = 200)
  (h2 : days_per_week = 5)
  (h3 : weeks_per_month = 4)
  (h4 : total_working_days = days_per_week * weeks_per_month)
  (h5 : daily_working_hours = total_hours_in_month / total_working_days) :
  daily_working_hours = 10 := 
sorry

end bob_daily_work_hours_l170_170170


namespace Jim_weekly_savings_l170_170160

-- Define the given conditions
def Sara_initial_savings : ℕ := 4100
def Sara_weekly_savings : ℕ := 10
def weeks : ℕ := 820

-- Define the proof goal based on the conditions
theorem Jim_weekly_savings :
  let Sara_total_savings := Sara_initial_savings + (Sara_weekly_savings * weeks)
  let Jim_weekly_savings := Sara_total_savings / weeks
  Jim_weekly_savings = 15 := 
by 
  sorry

end Jim_weekly_savings_l170_170160


namespace office_distance_l170_170950

theorem office_distance (d t : ℝ) 
    (h1 : d = 40 * (t + 1.5)) 
    (h2 : d - 40 = 60 * (t - 2)) : 
    d = 340 :=
by
  -- The detailed proof omitted
  sorry

end office_distance_l170_170950


namespace sufficient_not_necessary_l170_170426

theorem sufficient_not_necessary (a b : ℝ) (h : b > a ∧ a > 0) : (1 / a > 1 / b) :=
by {
  sorry -- the proof steps are intentionally omitted
}

end sufficient_not_necessary_l170_170426


namespace caps_percentage_l170_170133

open Real

-- Define the conditions as given in part (a)
def total_caps : ℝ := 575
def red_caps : ℝ := 150
def green_caps : ℝ := 120
def blue_caps : ℝ := 175
def yellow_caps : ℝ := total_caps - (red_caps + green_caps + blue_caps)

-- Define the problem asking for the percentages of each color and proving the answer
theorem caps_percentage :
  (red_caps / total_caps) * 100 = 26.09 ∧
  (green_caps / total_caps) * 100 = 20.87 ∧
  (blue_caps / total_caps) * 100 = 30.43 ∧
  (yellow_caps / total_caps) * 100 = 22.61 :=
by
  -- proof steps would go here
  sorry

end caps_percentage_l170_170133


namespace trapezoid_area_l170_170222

theorem trapezoid_area (h : ℝ) : 
  let base1 := 3 * h 
  let base2 := 4 * h 
  let average_base := (base1 + base2) / 2 
  let area := average_base * h 
  area = (7 * h^2) / 2 := 
by
  sorry

end trapezoid_area_l170_170222


namespace real_y_iff_x_ranges_l170_170892

-- Definitions for conditions
variable (x y : ℝ)

-- Condition for the equation
def equation := 9 * y^2 - 6 * x * y + 2 * x + 7 = 0

-- Theorem statement
theorem real_y_iff_x_ranges :
  (∃ y : ℝ, equation x y) ↔ (x ≤ -2 ∨ x ≥ 7) :=
sorry

end real_y_iff_x_ranges_l170_170892


namespace chocolate_chip_cookie_count_l170_170371

-- Let cookies_per_bag be the number of cookies in each bag
def cookies_per_bag : ℕ := 5

-- Let oatmeal_cookies be the number of oatmeal cookies
def oatmeal_cookies : ℕ := 2

-- Let num_baggies be the number of baggies
def num_baggies : ℕ := 7

-- Define the total number of cookies as num_baggies * cookies_per_bag
def total_cookies : ℕ := num_baggies * cookies_per_bag

-- Define the number of chocolate chip cookies as total_cookies - oatmeal_cookies
def chocolate_chip_cookies : ℕ := total_cookies - oatmeal_cookies

-- Prove that the number of chocolate chip cookies is 33
theorem chocolate_chip_cookie_count : chocolate_chip_cookies = 33 := by
  sorry

end chocolate_chip_cookie_count_l170_170371


namespace point_in_second_quadrant_coordinates_l170_170191

theorem point_in_second_quadrant_coordinates (a : ℤ) (h1 : a + 1 < 0) (h2 : 2 * a + 6 > 0) :
  (a + 1, 2 * a + 6) = (-1, 2) :=
sorry

end point_in_second_quadrant_coordinates_l170_170191


namespace find_minimum_n_l170_170379

noncomputable def a_seq (n : ℕ) : ℕ := 3 ^ (n - 1)

noncomputable def S_n (n : ℕ) : ℕ := 1 / 2 * (3 ^ n - 1)

theorem find_minimum_n (S_n : ℕ → ℕ) (n : ℕ) :
  (3^n - 1) / 2 > 1000 → n = 7 := 
sorry

end find_minimum_n_l170_170379


namespace eq_3_solutions_l170_170716

theorem eq_3_solutions (p : ℕ) (hp : Nat.Prime p) :
  ∃! (x y : ℕ), (0 < x) ∧ (0 < y) ∧ ((1 / x) + (1 / y) = (1 / p)) ∧
  ((x = p + 1 ∧ y = p^2 + p) ∨ (x = p + p ∧ y = p + p) ∨ (x = p^2 + p ∧ y = p + 1)) :=
sorry

end eq_3_solutions_l170_170716


namespace younger_person_age_l170_170408

/-- Let E be the present age of the elder person and Y be the present age of the younger person.
Given the conditions :
1) E - Y = 20
2) E - 15 = 2 * (Y - 15)
Prove that Y = 35. -/
theorem younger_person_age (E Y : ℕ) 
  (h1 : E - Y = 20) 
  (h2 : E - 15 = 2 * (Y - 15)) : 
  Y = 35 :=
sorry

end younger_person_age_l170_170408


namespace carol_pennies_l170_170485

variable (a c : ℕ)

theorem carol_pennies (h₁ : c + 2 = 4 * (a - 2)) (h₂ : c - 2 = 3 * (a + 2)) : c = 62 :=
by
  sorry

end carol_pennies_l170_170485


namespace number_of_educated_employees_l170_170847

-- Define the context and input values
variable (T: ℕ) (I: ℕ := 20) (decrease_illiterate: ℕ := 15) (total_decrease_illiterate: ℕ := I * decrease_illiterate) (average_salary_decrease: ℕ := 10)

-- The theorem statement
theorem number_of_educated_employees (h1: total_decrease_illiterate / T = average_salary_decrease) (h2: T = I + 10): L = 10 := by
  sorry

end number_of_educated_employees_l170_170847


namespace bert_made_1_dollar_l170_170425

def bert_earnings (selling_price tax_rate markup : ℝ) : ℝ :=
  selling_price - (tax_rate * selling_price) - (selling_price - markup)

theorem bert_made_1_dollar :
  bert_earnings 90 0.1 10 = 1 :=
by 
  sorry

end bert_made_1_dollar_l170_170425


namespace zs_share_in_profit_l170_170672

noncomputable def calculateProfitShare (x_investment y_investment z_investment z_months total_profit : ℚ) : ℚ :=
  let x_invest_months := x_investment * 12
  let y_invest_months := y_investment * 12
  let z_invest_months := z_investment * z_months
  let total_invest_months := x_invest_months + y_invest_months + z_invest_months
  let z_share := z_invest_months / total_invest_months
  total_profit * z_share

theorem zs_share_in_profit :
  calculateProfitShare 36000 42000 48000 8 14190 = 2580 :=
by
  sorry

end zs_share_in_profit_l170_170672


namespace goods_train_speed_l170_170652

theorem goods_train_speed (length_train : ℝ) (length_platform : ℝ) (time_seconds : ℝ) (speed_kmph : ℝ) :
  length_train = 280.04 →
  length_platform = 240 →
  time_seconds = 26 →
  speed_kmph = (length_train + length_platform) / time_seconds * 3.6 →
  speed_kmph = 72 :=
by
  intros h_train h_platform h_time h_speed
  rw [h_train, h_platform, h_time] at h_speed
  sorry

end goods_train_speed_l170_170652


namespace find_k_l170_170128

theorem find_k (x y k : ℝ) (h₁ : 3 * x + y = k) (h₂ : -1.2 * x + y = -20) (hx : x = 7) : k = 9.4 :=
by
  sorry

end find_k_l170_170128


namespace quadratic_no_real_roots_l170_170492

theorem quadratic_no_real_roots (c : ℝ) : (∀ x : ℝ, x^2 + 2 * x + c ≠ 0) → c > 1 :=
by
  sorry

end quadratic_no_real_roots_l170_170492


namespace find_BC_l170_170139

variable (A B C : Type)
variables (a b : ℝ) -- Angles
variables (AB BC CA : ℝ) -- Sides of the triangle

-- Given conditions:
-- 1: Triangle ABC
-- 2: cos(a - b) + sin(a + b) = 2
-- 3: AB = 4

theorem find_BC (hAB : AB = 4) (hTrig : Real.cos (a - b) + Real.sin (a + b) = 2) :
  BC = 2 * Real.sqrt 2 := 
sorry

end find_BC_l170_170139


namespace kids_joined_in_l170_170041

-- Define the given conditions
def original : ℕ := 14
def current : ℕ := 36

-- State the goal
theorem kids_joined_in : (current - original = 22) :=
by
  sorry

end kids_joined_in_l170_170041


namespace initial_y_percentage_proof_l170_170975

variable (initial_volume : ℝ) (added_volume : ℝ) (initial_percentage_x : ℝ) (result_percentage_x : ℝ)

-- Conditions
def initial_volume_condition : Prop := initial_volume = 80
def added_volume_condition : Prop := added_volume = 20
def initial_percentage_x_condition : Prop := initial_percentage_x = 0.30
def result_percentage_x_condition : Prop := result_percentage_x = 0.44

-- Question
def initial_percentage_y (initial_volume added_volume initial_percentage_x result_percentage_x : ℝ) : ℝ :=
  1 - initial_percentage_x

-- Theorem
theorem initial_y_percentage_proof 
  (h1 : initial_volume_condition initial_volume)
  (h2 : added_volume_condition added_volume)
  (h3 : initial_percentage_x_condition initial_percentage_x)
  (h4 : result_percentage_x_condition result_percentage_x) :
  initial_percentage_y initial_volume added_volume initial_percentage_x result_percentage_x = 0.70 := 
sorry

end initial_y_percentage_proof_l170_170975


namespace order_of_arrival_l170_170940

noncomputable def position_order (P S O E R : ℕ) : Prop :=
  S = O - 10 ∧ S = R + 25 ∧ R = E - 5 ∧ E = P - 25

theorem order_of_arrival (P S O E R : ℕ) (h : position_order P S O E R) :
  P > (S + 10) ∧ S > (O - 10) ∧ O > (E + 5) ∧ E > R :=
sorry

end order_of_arrival_l170_170940


namespace vector_computation_l170_170835

def c : ℝ × ℝ × ℝ := (-3, 5, 2)
def d : ℝ × ℝ × ℝ := (5, -1, 3)

theorem vector_computation : 2 • c - 5 • d + c = (-34, 20, -9) := by
  sorry

end vector_computation_l170_170835


namespace same_solutions_implies_k_value_l170_170055

theorem same_solutions_implies_k_value (k : ℤ) : (∀ x : ℤ, 2 * x = 4 ↔ 3 * x + k = -2) → k = -8 :=
by
  sorry

end same_solutions_implies_k_value_l170_170055


namespace partA_partB_partC_partD_l170_170887

variable (α β : ℝ)
variable (hα : 0 < α) (hα1 : α < 1)
variable (hβ : 0 < β) (hβ1 : β < 1)

theorem partA : 
  (1 - β) * (1 - α) * (1 - β) = (1 - α) * (1 - β)^2 := by
  sorry

theorem partB :
  β * (1 - β)^2 = β * (1 - β)^2 := by
  sorry

theorem partC :
  β * (1 - β)^2 + (1 - β)^3 = β * (1 - β)^2 + (1 - β)^3 := by
  sorry

theorem partD (hα0 : α < 0.5) :
  (1 - α) * (α - α^2) < (1 - α) := by
  sorry

end partA_partB_partC_partD_l170_170887


namespace fish_per_black_duck_l170_170523

theorem fish_per_black_duck :
  ∀ (W_d B_d M_d : ℕ) (fish_per_W fish_per_M total_fish : ℕ),
    (fish_per_W = 5) →
    (fish_per_M = 12) →
    (W_d = 3) →
    (B_d = 7) →
    (M_d = 6) →
    (total_fish = 157) →
    (total_fish - (W_d * fish_per_W + M_d * fish_per_M)) = 70 →
    (70 / B_d) = 10 :=
by
  intros W_d B_d M_d fish_per_W fish_per_M total_fish hW hM hW_d hB_d hM_d htotal_fish hcalculation
  sorry

end fish_per_black_duck_l170_170523


namespace min_abs_sum_l170_170641

theorem min_abs_sum (x y : ℝ) : |x - 1| + |x| + |y - 1| + |y + 1| ≥ 3 :=
by sorry

end min_abs_sum_l170_170641


namespace Kyler_wins_1_game_l170_170299

theorem Kyler_wins_1_game
  (peter_wins : ℕ)
  (peter_losses : ℕ)
  (emma_wins : ℕ)
  (emma_losses : ℕ)
  (kyler_losses : ℕ)
  (total_games : ℕ)
  (kyler_wins : ℕ)
  (htotal : total_games = (peter_wins + peter_losses + emma_wins + emma_losses + kyler_wins + kyler_losses) / 2)
  (hpeter : peter_wins = 4 ∧ peter_losses = 2)
  (hemma : emma_wins = 3 ∧ emma_losses = 3)
  (hkyler_losses : kyler_losses = 3)
  (htotal_wins_losses : total_games = peter_wins + emma_wins + kyler_wins) : kyler_wins = 1 :=
by
  sorry

end Kyler_wins_1_game_l170_170299


namespace prove_ab_l170_170028

theorem prove_ab 
  (a b : ℝ)
  (h1 : a + b = 4)
  (h2 : a^2 + b^2 = 6) : 
  a * b = 5 :=
by
  sorry

end prove_ab_l170_170028


namespace total_texts_sent_is_97_l170_170202

def textsSentOnMondayAllison := 5
def textsSentOnMondayBrittney := 5
def textsSentOnMondayCarol := 5

def textsSentOnTuesdayAllison := 15
def textsSentOnTuesdayBrittney := 10
def textsSentOnTuesdayCarol := 12

def textsSentOnWednesdayAllison := 20
def textsSentOnWednesdayBrittney := 18
def textsSentOnWednesdayCarol := 7

def totalTextsAllison := textsSentOnMondayAllison + textsSentOnTuesdayAllison + textsSentOnWednesdayAllison
def totalTextsBrittney := textsSentOnMondayBrittney + textsSentOnTuesdayBrittney + textsSentOnWednesdayBrittney
def totalTextsCarol := textsSentOnMondayCarol + textsSentOnTuesdayCarol + textsSentOnWednesdayCarol

def totalTextsAllThree := totalTextsAllison + totalTextsBrittney + totalTextsCarol

theorem total_texts_sent_is_97 : totalTextsAllThree = 97 := by
  sorry

end total_texts_sent_is_97_l170_170202


namespace cost_hour_excess_is_1point75_l170_170556

noncomputable def cost_per_hour_excess (x : ℝ) : Prop :=
  let total_hours := 9
  let initial_cost := 15
  let excess_hours := total_hours - 2
  let total_cost := initial_cost + excess_hours * x
  let average_cost_per_hour := 3.0277777777777777
  (total_cost / total_hours) = average_cost_per_hour

theorem cost_hour_excess_is_1point75 : cost_per_hour_excess 1.75 :=
by
  sorry

end cost_hour_excess_is_1point75_l170_170556


namespace distance_range_l170_170740

variable (x : ℝ)
variable (starting_fare : ℝ := 6) -- fare in yuan for up to 2 kilometers
variable (surcharge : ℝ := 1) -- yuan surcharge per ride
variable (additional_fare : ℝ := 1) -- fare for every additional 0.5 kilometers
variable (additional_distance : ℝ := 0.5) -- distance in kilometers for every additional fare

theorem distance_range (h_total_fare : 9 = starting_fare + (x - 2) / additional_distance * additional_fare + surcharge) :
  2.5 < x ∧ x ≤ 3 :=
by
  -- Proof goes here
  sorry

end distance_range_l170_170740


namespace greatest_whole_number_satisfies_inequality_l170_170182

theorem greatest_whole_number_satisfies_inequality : 
  ∃ (x : ℕ), (∀ (y : ℕ), (6 * y - 4 < 5 - 3 * y) → y ≤ x) ∧ x = 0 := 
sorry

end greatest_whole_number_satisfies_inequality_l170_170182


namespace intersection_M_N_l170_170501

-- Define set M and N
def M : Set ℝ := {x | x - 1 < 0}
def N : Set ℝ := {x | x^2 - 5 * x + 6 > 0}

-- Problem statement to show their intersection
theorem intersection_M_N :
  M ∩ N = {x | x < 1} := 
sorry

end intersection_M_N_l170_170501


namespace area_ratio_equilateral_triangl_l170_170623

theorem area_ratio_equilateral_triangl (x : ℝ) :
  let sA : ℝ := x 
  let sB : ℝ := 3 * sA
  let sC : ℝ := 5 * sA
  let sD : ℝ := 4 * sA
  let area_ABC := (Real.sqrt 3 / 4) * (sA ^ 2)
  let s := (sB + sC + sD) / 2
  let area_A'B'C' := Real.sqrt (s * (s - sB) * (s - sC) * (s - sD))
  (area_A'B'C' / area_ABC) = 8 * Real.sqrt 3 := by
  sorry

end area_ratio_equilateral_triangl_l170_170623


namespace delta_x_not_zero_l170_170992

noncomputable def average_rate_of_change (f : ℝ → ℝ) (x : ℝ) (delta_x : ℝ) : ℝ :=
  (f (x + delta_x) - f x) / delta_x

theorem delta_x_not_zero (f : ℝ → ℝ) (x delta_x : ℝ) (h_neq : delta_x ≠ 0):
  average_rate_of_change f x delta_x ≠ 0 := 
by
  sorry

end delta_x_not_zero_l170_170992


namespace side_length_of_regular_pentagon_l170_170653

theorem side_length_of_regular_pentagon (perimeter : ℝ) (number_of_sides : ℕ) (h1 : perimeter = 23.4) (h2 : number_of_sides = 5) : 
  perimeter / number_of_sides = 4.68 :=
by
  sorry

end side_length_of_regular_pentagon_l170_170653


namespace contractor_fine_per_absent_day_l170_170306

theorem contractor_fine_per_absent_day :
  ∃ x : ℝ, (∀ (total_days absent_days worked_days earnings_per_day total_earnings : ℝ),
   total_days = 30 →
   earnings_per_day = 25 →
   total_earnings = 490 →
   absent_days = 8 →
   worked_days = total_days - absent_days →
   25 * worked_days - absent_days * x = total_earnings
  ) → x = 7.5 :=
by
  existsi 7.5
  intros
  sorry

end contractor_fine_per_absent_day_l170_170306


namespace song_book_cost_correct_l170_170275

/-- Define the constants for the problem. -/
def clarinet_cost : ℝ := 130.30
def pocket_money : ℝ := 12.32
def total_spent : ℝ := 141.54

/-- Prove the cost of the song book. -/
theorem song_book_cost_correct :
  (total_spent - clarinet_cost) = 11.24 :=
by
  sorry

end song_book_cost_correct_l170_170275


namespace farmer_trees_l170_170251

theorem farmer_trees (x n m : ℕ) 
  (h1 : x + 20 = n^2) 
  (h2 : x - 39 = m^2) : 
  x = 880 := 
by sorry

end farmer_trees_l170_170251


namespace gravitational_force_on_asteroid_l170_170318

theorem gravitational_force_on_asteroid :
  ∃ (k : ℝ), ∃ (f : ℝ), 
  (∀ (d : ℝ), f = k / d^2) ∧
  (d = 5000 → f = 700) →
  (∃ (f_asteroid : ℝ), f_asteroid = k / 300000^2 ∧ f_asteroid = 7 / 36) :=
sorry

end gravitational_force_on_asteroid_l170_170318


namespace rightmost_three_digits_of_7_pow_1994_l170_170576

theorem rightmost_three_digits_of_7_pow_1994 :
  (7 ^ 1994) % 800 = 49 :=
by
  sorry

end rightmost_three_digits_of_7_pow_1994_l170_170576


namespace cos_triple_angle_l170_170996

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (3 * θ) = -117/125 := 
by
  sorry

end cos_triple_angle_l170_170996


namespace avg_speed_l170_170405

noncomputable def jane_total_distance : ℝ := 120
noncomputable def time_period_hours : ℝ := 7

theorem avg_speed :
  jane_total_distance / time_period_hours = (120 / 7 : ℝ):=
by
  sorry

end avg_speed_l170_170405


namespace hawks_loss_percentage_is_30_l170_170034

-- Define the variables and the conditions
def matches_won (x : ℕ) : ℕ := 7 * x
def matches_lost (x : ℕ) : ℕ := 3 * x
def total_matches (x : ℕ) : ℕ := matches_won x + matches_lost x
def percent_lost (x : ℕ) : ℕ := (matches_lost x * 100) / total_matches x

-- The goal statement in Lean 4
theorem hawks_loss_percentage_is_30 (x : ℕ) (h : x > 0) : percent_lost x = 30 :=
by sorry

end hawks_loss_percentage_is_30_l170_170034


namespace space_shuttle_speed_kmph_l170_170957

-- Question: Prove that the speed of the space shuttle in kilometers per hour is 32400, given it travels at 9 kilometers per second and there are 3600 seconds in an hour.
theorem space_shuttle_speed_kmph :
  (9 * 3600 = 32400) :=
by
  sorry

end space_shuttle_speed_kmph_l170_170957


namespace students_catching_up_on_homework_l170_170618

-- Definitions for the given conditions
def total_students := 120
def silent_reading_students := (2/5 : ℚ) * total_students
def board_games_students := (3/10 : ℚ) * total_students
def group_discussions_students := (1/8 : ℚ) * total_students
def other_activities_students := silent_reading_students + board_games_students + group_discussions_students
def catching_up_homework_students := total_students - other_activities_students

-- Statement of the proof problem
theorem students_catching_up_on_homework : catching_up_homework_students = 21 := by
  sorry

end students_catching_up_on_homework_l170_170618


namespace calculation_l170_170421

theorem calculation :
  5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 :=
  by
    sorry

end calculation_l170_170421


namespace exponent_equation_l170_170690

theorem exponent_equation (n : ℕ) (h : 8^4 = 16^n) : n = 3 :=
by sorry

end exponent_equation_l170_170690


namespace trajectory_of_center_of_P_l170_170880

-- Define circles M and N
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define the conditions for the moving circle P
def externally_tangent (x y r : ℝ) : Prop := (x + 1)^2 + y^2 = (1 + r)^2
def internally_tangent (x y r : ℝ) : Prop := (x - 1)^2 + y^2 = (5 - r)^2

-- The statement we need to prove
theorem trajectory_of_center_of_P : ∃ (x y : ℝ), 
  (externally_tangent x y r) ∧ (internally_tangent x y r) →
  (x^2 / 9 + y^2 / 8 = 1) :=
by
  -- Proof will go here
  sorry

end trajectory_of_center_of_P_l170_170880


namespace birds_count_is_30_l170_170322

def total_animals : ℕ := 77
def number_of_kittens : ℕ := 32
def number_of_hamsters : ℕ := 15

def number_of_birds : ℕ := total_animals - number_of_kittens - number_of_hamsters

theorem birds_count_is_30 : number_of_birds = 30 := by
  sorry

end birds_count_is_30_l170_170322


namespace right_triangle_has_one_right_angle_l170_170065

def is_right_angle (θ : ℝ) : Prop := θ = 90

def sum_of_triangle_angles (α β γ : ℝ) : Prop := α + β + γ = 180

def right_triangle (α β γ : ℝ) : Prop := is_right_angle α ∨ is_right_angle β ∨ is_right_angle γ

theorem right_triangle_has_one_right_angle (α β γ : ℝ) :
  right_triangle α β γ → sum_of_triangle_angles α β γ →
  (is_right_angle α ∧ ¬is_right_angle β ∧ ¬is_right_angle γ) ∨
  (¬is_right_angle α ∧ is_right_angle β ∧ ¬is_right_angle γ) ∨
  (¬is_right_angle α ∧ ¬is_right_angle β ∧ is_right_angle γ) :=
by
  sorry

end right_triangle_has_one_right_angle_l170_170065


namespace german_team_goals_l170_170665

theorem german_team_goals :
  ∃ (x : ℕ), 10 < x ∧ x < 17 ∧ 11 < x ∧ x < 18 ∧ x % 2 = 1 ∧ 
             ((10 < x ∧ x < 17) ∧ (11 < x ∧ x < 18) ↔ (x % 2 = 0)) :=
sorry

end german_team_goals_l170_170665


namespace calculate_fraction_square_mul_l170_170691

theorem calculate_fraction_square_mul :
  ((8 / 9) ^ 2) * ((1 / 3) ^ 2) = 64 / 729 :=
by
  sorry

end calculate_fraction_square_mul_l170_170691


namespace calc_theoretical_yield_l170_170097
-- Importing all necessary libraries

-- Define the molar masses
def molar_mass_NaNO3 : ℝ := 85

-- Define the initial moles
def initial_moles_NH4NO3 : ℝ := 2
def initial_moles_NaOH : ℝ := 2

-- Define the final yield percentage
def yield_percentage : ℝ := 0.85

-- State the proof problem
theorem calc_theoretical_yield :
  let moles_NaNO3 := (2 : ℝ) * 2 * yield_percentage
  let grams_NaNO3 := moles_NaNO3 * molar_mass_NaNO3
  grams_NaNO3 = 289 :=
by 
  sorry

end calc_theoretical_yield_l170_170097


namespace shopkeeper_discount_and_selling_price_l170_170647

theorem shopkeeper_discount_and_selling_price :
  let CP := 100
  let MP := CP + 0.5 * CP
  let SP := CP + 0.15 * CP
  let Discount := (MP - SP) / MP * 100
  Discount = 23.33 ∧ SP = 115 :=
by
  sorry

end shopkeeper_discount_and_selling_price_l170_170647


namespace determine_expr_l170_170309

noncomputable def expr (a b c d : ℝ) : ℝ :=
  (1 + a + a * b) / (1 + a + a * b + a * b * c) +
  (1 + b + b * c) / (1 + b + b * c + b * c * d) +
  (1 + c + c * d) / (1 + c + c * d + c * d * a) +
  (1 + d + d * a) / (1 + d + d * a + d * a * b)

theorem determine_expr (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  expr a b c d = 2 :=
sorry

end determine_expr_l170_170309


namespace min_cubes_l170_170890

-- Define the conditions as properties
structure FigureViews :=
  (front_view : ℕ)
  (side_view : ℕ)
  (top_view : ℕ)
  (adjacency_requirement : Bool)

-- Define the given views
def given_views : FigureViews := {
  front_view := 3,  -- as described: 2 cubes at bottom + 1 on top
  side_view := 3,   -- same as front view
  top_view := 3,    -- L-shape consists of 3 cubes
  adjacency_requirement := true
}

-- The theorem to state that the minimum number of cubes is 3
theorem min_cubes (views : FigureViews) : views.front_view = 3 ∧ views.side_view = 3 ∧ views.top_view = 3 ∧ views.adjacency_requirement = true → ∃ n, n = 3 :=
by {
  sorry
}

end min_cubes_l170_170890


namespace turtle_minimum_distance_l170_170494

theorem turtle_minimum_distance 
  (constant_speed : ℝ)
  (turn_angle : ℝ)
  (total_time : ℕ) :
  constant_speed = 5 →
  turn_angle = 90 →
  total_time = 11 →
  ∃ (final_position : ℝ × ℝ), 
    (final_position = (5, 0) ∨ final_position = (-5, 0) ∨ final_position = (0, 5) ∨ final_position = (0, -5)) ∧
    dist final_position (0, 0) = 5 :=
by
  intros
  sorry

end turtle_minimum_distance_l170_170494


namespace nested_sqrt_eq_two_l170_170661

theorem nested_sqrt_eq_two (y : ℝ) (h : y = Real.sqrt (2 + y)) : y = 2 :=
by {
    -- Proof skipped
    sorry
}

end nested_sqrt_eq_two_l170_170661


namespace complex_point_quadrant_l170_170879

def inFourthQuadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem complex_point_quadrant :
  let z : ℂ := (2 - I) / (2 + I)
  inFourthQuadrant z :=
by
  sorry

end complex_point_quadrant_l170_170879


namespace sum_of_15_terms_l170_170677

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem sum_of_15_terms 
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_sum1 : a 1 + a 2 + a 3 = 1)
  (h_sum2 : a 4 + a 5 + a 6 = -2) :
  (a 1 + a 2 + a 3) + (a 4 + a 5 + a 6) + (a 7 + a 8 + a 9) +
  (a 10 + a 11 + a 12) + (a 13 + a 14 + a 15) = 11 :=
sorry

end sum_of_15_terms_l170_170677


namespace part1_factorization_part2_factorization_l170_170370

-- Part 1
theorem part1_factorization (x : ℝ) :
  (x - 1) * (6 * x + 5) = 6 * x^2 - x - 5 :=
by {
  sorry
}

-- Part 2
theorem part2_factorization (x : ℝ) :
  (x - 1) * (x + 3) * (x - 2) = x^3 - 7 * x + 6 :=
by {
  sorry
}

end part1_factorization_part2_factorization_l170_170370


namespace ordered_pair_solution_l170_170038

theorem ordered_pair_solution :
  ∃ (x y : ℤ), x + y = (6 - x) + (6 - y) ∧ x - y = (x - 2) + (y - 2) ∧ (x, y) = (2, 4) :=
by
  sorry

end ordered_pair_solution_l170_170038


namespace find_k_l170_170532

theorem find_k (k : ℝ) :
    (∀ x : ℝ, 4 * x^2 + k * x + 4 ≠ 0) → k = 8 :=
sorry

end find_k_l170_170532


namespace problem1_problem2_l170_170457

-- Problem 1: Proove that the given expression equals 1
theorem problem1 : (2021 * 2023) / (2022^2 - 1) = 1 :=
  by
  sorry

-- Problem 2: Proove that the given expression equals 45000
theorem problem2 : 2 * 101^2 + 2 * 101 * 98 + 2 * 49^2 = 45000 :=
  by
  sorry

end problem1_problem2_l170_170457


namespace maximum_sets_l170_170969

-- define the initial conditions
def dinner_forks : Nat := 6
def knives : Nat := dinner_forks + 9
def soup_spoons : Nat := 2 * knives
def teaspoons : Nat := dinner_forks / 2
def dessert_forks : Nat := teaspoons / 3
def butter_knives : Nat := 2 * dessert_forks

def max_capacity_g : Nat := 20000

def weight_dinner_fork : Nat := 80
def weight_knife : Nat := 100
def weight_soup_spoon : Nat := 85
def weight_teaspoon : Nat := 50
def weight_dessert_fork : Nat := 70
def weight_butter_knife : Nat := 65

-- Calculate the total weight of the existing cutlery
def total_weight_existing : Nat := 
  (dinner_forks * weight_dinner_fork) + 
  (knives * weight_knife) + 
  (soup_spoons * weight_soup_spoon) + 
  (teaspoons * weight_teaspoon) + 
  (dessert_forks * weight_dessert_fork) + 
  (butter_knives * weight_butter_knife)

-- Calculate the weight of one 2-piece cutlery set (1 knife + 1 dinner fork)
def weight_set : Nat := weight_knife + weight_dinner_fork

-- The remaining capacity in the drawer
def remaining_capacity_g : Nat := max_capacity_g - total_weight_existing

-- The maximum number of 2-piece cutlery sets that can be added
def max_2_piece_sets : Nat := remaining_capacity_g / weight_set

-- Theorem: maximum number of 2-piece cutlery sets that can be added is 84
theorem maximum_sets : max_2_piece_sets = 84 :=
by
  sorry

end maximum_sets_l170_170969


namespace average_score_of_seniors_l170_170428

theorem average_score_of_seniors
    (total_students : ℕ)
    (average_score_all : ℚ)
    (num_seniors num_non_seniors : ℕ)
    (mean_score_senior mean_score_non_senior : ℚ)
    (h1 : total_students = 120)
    (h2 : average_score_all = 84)
    (h3 : num_non_seniors = 2 * num_seniors)
    (h4 : mean_score_senior = 2 * mean_score_non_senior)
    (h5 : num_seniors + num_non_seniors = total_students)
    (h6 : num_seniors * mean_score_senior + num_non_seniors * mean_score_non_senior = total_students * average_score_all) :
  mean_score_senior = 126 :=
by
  sorry

end average_score_of_seniors_l170_170428


namespace garment_industry_initial_men_l170_170347

theorem garment_industry_initial_men (M : ℕ) :
  (M * 8 * 10 = 6 * 20 * 8) → M = 12 :=
by
  sorry

end garment_industry_initial_men_l170_170347


namespace initial_total_packs_l170_170177

def initial_packs (total_packs : ℕ) (regular_packs : ℕ) (unusual_packs : ℕ) (excellent_packs : ℕ) : Prop :=
  total_packs = regular_packs + unusual_packs + excellent_packs

def ratio_packs (regular_packs : ℕ) (unusual_packs : ℕ) (excellent_packs : ℕ) : Prop :=
  3 * (regular_packs + unusual_packs + excellent_packs) = 3 * regular_packs + 4 * unusual_packs + 6 * excellent_packs

def new_ratios (regular_packs : ℕ) (unusual_packs : ℕ) (excellent_packs : ℕ) (new_regular_packs : ℕ) (new_unusual_packs : ℕ) (new_excellent_packs : ℕ) : Prop :=
  2 * (new_regular_packs) + 5 * (new_unusual_packs) + 8 * (new_excellent_packs) = regular_packs + unusual_packs + excellent_packs + 8 * (regular_packs)

def pack_changes (initial_regular_packs : ℕ) (initial_unusual_packs : ℕ) (initial_excellent_packs : ℕ) (new_regular_packs : ℕ) (new_unusual_packs : ℕ) (new_excellent_packs : ℕ) : Prop :=
  initial_excellent_packs <= new_excellent_packs + 80 ∧ initial_regular_packs - new_regular_packs ≤ 10

theorem initial_total_packs (total_packs : ℕ) (regular_packs : ℕ) (unusual_packs : ℕ) (excellent_packs : ℕ) 
(new_regular_packs : ℕ) (new_unusual_packs : ℕ) (new_excellent_packs : ℕ) :
  initial_packs total_packs regular_packs unusual_packs excellent_packs ∧
  ratio_packs regular_packs unusual_packs excellent_packs ∧ 
  new_ratios regular_packs unusual_packs excellent_packs new_regular_packs new_unusual_packs new_excellent_packs ∧ 
  pack_changes regular_packs unusual_packs excellent_packs new_regular_packs new_unusual_packs new_excellent_packs 
  → total_packs = 260 := 
sorry

end initial_total_packs_l170_170177


namespace mohan_cookies_l170_170638

theorem mohan_cookies :
  ∃ a : ℕ, 
    a % 4 = 3 ∧
    a % 5 = 2 ∧
    a % 7 = 4 ∧
    a = 67 :=
by
  -- The proof will be written here.
  sorry

end mohan_cookies_l170_170638


namespace john_bought_slurpees_l170_170201

noncomputable def slurpees_bought (total_money paid change slurpee_cost : ℕ) : ℕ :=
  (paid - change) / slurpee_cost

theorem john_bought_slurpees :
  let total_money := 20
  let slurpee_cost := 2
  let change := 8
  slurpees_bought total_money total_money change slurpee_cost = 6 :=
by
  sorry

end john_bought_slurpees_l170_170201


namespace smallest_possible_value_of_M_l170_170367

theorem smallest_possible_value_of_M (a b c d e : ℕ) (h1 : a + b + c + d + e = 3060) 
    (h2 : a + e ≥ 1300) :
    ∃ M : ℕ, M = max (max (a + b) (max (b + c) (max (c + d) (d + e)))) ∧ M = 1174 :=
by
  sorry

end smallest_possible_value_of_M_l170_170367


namespace parabola_directrix_eq_l170_170621

theorem parabola_directrix_eq (a : ℝ) (h : - a / 4 = - (1 : ℝ) / 4) : a = 1 := by
  sorry

end parabola_directrix_eq_l170_170621


namespace difference_sweaters_Monday_Tuesday_l170_170917

-- Define conditions
def sweaters_knit_on_Monday : ℕ := 8
def sweaters_knit_on_Tuesday (T : ℕ) : Prop := T > 8
def sweaters_knit_on_Wednesday (T : ℕ) : ℕ := T - 4
def sweaters_knit_on_Thursday (T : ℕ) : ℕ := T - 4
def sweaters_knit_on_Friday : ℕ := 4

-- Define total sweaters knit in the week
def total_sweaters_knit (T : ℕ) : ℕ :=
  sweaters_knit_on_Monday + T + sweaters_knit_on_Wednesday T + sweaters_knit_on_Thursday T + sweaters_knit_on_Friday

-- Lean Theorem Statement
theorem difference_sweaters_Monday_Tuesday : ∀ T : ℕ, sweaters_knit_on_Tuesday T → total_sweaters_knit T = 34 → T - sweaters_knit_on_Monday = 2 :=
by
  intros T hT_total
  sorry

end difference_sweaters_Monday_Tuesday_l170_170917


namespace d_minus_r_eq_15_l170_170147

theorem d_minus_r_eq_15 (d r : ℤ) (h_d_gt_1 : d > 1)
  (h1 : 1059 % d = r)
  (h2 : 1417 % d = r)
  (h3 : 2312 % d = r) :
  d - r = 15 :=
sorry

end d_minus_r_eq_15_l170_170147


namespace find_y_l170_170079

def is_divisible_by (x y : ℕ) : Prop := x % y = 0

def ends_with_digit (x : ℕ) (d : ℕ) : Prop :=
  x % 10 = d

theorem find_y (y : ℕ) :
  (y > 0) ∧
  is_divisible_by y 4 ∧
  is_divisible_by y 5 ∧
  is_divisible_by y 7 ∧
  is_divisible_by y 13 ∧
  ¬ is_divisible_by y 8 ∧
  ¬ is_divisible_by y 15 ∧
  ¬ is_divisible_by y 50 ∧
  ends_with_digit y 0
  → y = 1820 :=
sorry

end find_y_l170_170079


namespace simplify_fraction_l170_170866

theorem simplify_fraction (x y : ℕ) (h₁ : x = 3) (h₂ : y = 4) : 
  (12 * x * y^3) / (9 * x^2 * y^2) = 16 / 9 :=
by
  sorry

end simplify_fraction_l170_170866


namespace num_ordered_triples_l170_170287

/-
Let Q be a right rectangular prism with integral side lengths a, b, and c such that a ≤ b ≤ c, and b = 2023.
A plane parallel to one of the faces of Q cuts Q into two prisms, one of which is similar to Q, and both have nonzero volume.
Prove that the number of ordered triples (a, b, c) such that b = 2023 is 7.
-/

theorem num_ordered_triples (a c : ℕ) (h : a ≤ 2023 ∧ 2023 ≤ c) (ac_eq_2023_squared : a * c = 2023^2) :
  ∃ count, count = 7 :=
by {
  sorry
}

end num_ordered_triples_l170_170287


namespace problem_inequality_l170_170205

noncomputable def A (x : ℝ) := (x - 3) ^ 2
noncomputable def B (x : ℝ) := (x - 2) * (x - 4)

theorem problem_inequality (x : ℝ) : A x > B x :=
  by
    sorry

end problem_inequality_l170_170205


namespace problem_statement_l170_170655

def f (x : ℤ) : ℤ := x^2 + 3
def g (x : ℤ) : ℤ := 2 * x + 5

theorem problem_statement : f (g 4) - g (f 4) = 129 := by
  sorry

end problem_statement_l170_170655


namespace problem1_problem2_l170_170074

-- Proof Problem for (1)
theorem problem1 : -15 - (-5) + 6 = -4 := sorry

-- Proof Problem for (2)
theorem problem2 : 81 / (-9 / 5) * (5 / 9) = -25 := sorry

end problem1_problem2_l170_170074


namespace remainder_1234567_127_l170_170631

theorem remainder_1234567_127 : (1234567 % 127) = 51 := 
by {
  sorry
}

end remainder_1234567_127_l170_170631


namespace problem_l170_170801

theorem problem (x : ℝ) (h : x^2 + x - 1 = 0) : x^3 + 2 * x^2 + 2007 = 2008 :=
by
  sorry

end problem_l170_170801


namespace fraction_expression_l170_170620

theorem fraction_expression :
  (3 / 7 + 5 / 8) / (5 / 12 + 2 / 9) = 531 / 322 :=
by sorry

end fraction_expression_l170_170620


namespace expand_polynomial_identity_l170_170116

variable {x : ℝ}

theorem expand_polynomial_identity : (7 * x + 5) * (5 * x ^ 2 - 2 * x + 4) = 35 * x ^ 3 + 11 * x ^ 2 + 18 * x + 20 := by
    sorry

end expand_polynomial_identity_l170_170116


namespace bag_of_chips_weight_l170_170952

theorem bag_of_chips_weight (c : ℕ) : 
  (∀ (t : ℕ), t = 9) → 
  (∀ (b : ℕ), b = 6) → 
  (∀ (x : ℕ), x = 4 * 6) → 
  (21 * 16 = 336) →
  (336 - 24 * 9 = 6 * c) → 
  c = 20 :=
by
  intros ht hb hx h_weight_total h_weight_chips
  sorry

end bag_of_chips_weight_l170_170952


namespace age_of_teacher_l170_170062

variables (age_students : ℕ) (age_all : ℕ) (teacher_age : ℕ)

def avg_age_students := 15
def num_students := 10
def num_people := 11
def avg_age_people := 16

theorem age_of_teacher
  (h1 : age_students = num_students * avg_age_students)
  (h2 : age_all = num_people * avg_age_people)
  (h3 : age_all = age_students + teacher_age) : teacher_age = 26 :=
by
  sorry

end age_of_teacher_l170_170062


namespace cubic_eq_solutions_l170_170506

theorem cubic_eq_solutions (x : ℝ) :
  x^3 - 4 * x = 0 ↔ x = 0 ∨ x = -2 ∨ x = 2 := by
  sorry

end cubic_eq_solutions_l170_170506


namespace inequality_holds_iff_m_lt_2_l170_170712

theorem inequality_holds_iff_m_lt_2 :
  (∀ x : ℝ, 1 < x ∧ x ≤ 4 → x^2 - m * x + m > 0) ↔ m < 2 :=
by
  sorry

end inequality_holds_iff_m_lt_2_l170_170712


namespace proposition2_and_4_correct_l170_170307

theorem proposition2_and_4_correct (a b : ℝ) : 
  (a > b ∧ b > 0 → a^2 - a > b^2 - b) ∧ 
  (a > 0 ∧ b > 0 ∧ 2 * a + b = 1 → a^2 + b^2 = 9) :=
by
  sorry

end proposition2_and_4_correct_l170_170307


namespace sum_powers_l170_170595

theorem sum_powers {a b : ℝ}
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^11 + b^11 = 199 :=
by
  sorry

end sum_powers_l170_170595


namespace smallest_class_size_l170_170967

variable (x : ℕ) 

theorem smallest_class_size
  (h1 : 5 * x + 2 > 40)
  (h2 : x ≥ 0) : 
  5 * 8 + 2 = 42 :=
by sorry

end smallest_class_size_l170_170967


namespace find_a5_div_b5_l170_170577

-- Definitions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d, ∀ n, a (n + 1) = a n + d
def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ := n * (a 0 + a (n - 1)) / 2

-- Main statement
theorem find_a5_div_b5 (a b : ℕ → ℤ) (S T : ℕ → ℤ)
  (h1 : is_arithmetic_sequence a)
  (h2 : is_arithmetic_sequence b)
  (h3 : ∀ n : ℕ, S n = sum_first_n_terms a n)
  (h4 : ∀ n : ℕ, T n = sum_first_n_terms b n)
  (h5 : ∀ n : ℕ, S n * (3 * n + 1) = 2 * n * T n) :
  (a 5 : ℚ) / b 5 = 9 / 14 :=
by
  sorry

end find_a5_div_b5_l170_170577


namespace intersection_A_B_l170_170594

def A : Set ℤ := {x | x > 0 }
def B : Set ℤ := {-1, 0, 1, 2, 3}

theorem intersection_A_B :
  A ∩ B = {1, 2, 3} :=
by
  sorry

end intersection_A_B_l170_170594


namespace shadow_length_to_time_l170_170209

theorem shadow_length_to_time (shadow_length_inches : ℕ) (stretch_rate_feet_per_hour : ℕ) (inches_per_foot : ℕ) 
                              (shadow_start_time : ℕ) :
  shadow_length_inches = 360 → stretch_rate_feet_per_hour = 5 → inches_per_foot = 12 → shadow_start_time = 0 →
  (shadow_length_inches / inches_per_foot) / stretch_rate_feet_per_hour = 6 := by
  intros h1 h2 h3 h4
  sorry

end shadow_length_to_time_l170_170209


namespace simplify_power_multiplication_l170_170849

theorem simplify_power_multiplication (x : ℝ) : (-x) ^ 3 * (-x) ^ 2 = -x ^ 5 :=
by sorry

end simplify_power_multiplication_l170_170849


namespace quadratic_vertex_l170_170675

noncomputable def quadratic_vertex_max (c d : ℝ) (h : -x^2 + c * x + d ≤ 0) : (ℝ × ℝ) :=
sorry

theorem quadratic_vertex 
  (c d : ℝ)
  (h : -x^2 + c * x + d ≤ 0)
  (root1 root2 : ℝ)
  (h_roots : root1 = -5 ∧ root2 = 3) :
  quadratic_vertex_max c d h = (4, 1) ∧ (∀ x: ℝ, (x - 4)^2 ≤ 1) :=
sorry

end quadratic_vertex_l170_170675


namespace general_term_formula_sum_first_n_terms_l170_170555

noncomputable def a_n (n : ℕ) : ℕ := 2^(n - 1)

def S (n : ℕ) : ℕ := n * (2^(n - 1))  -- Placeholder function for the sum of the first n terms

theorem general_term_formula (a_3_eq_2a_2 : 2^(3 - 1) = 2 * 2^(2 - 1)) (S3_eq_7 : S 3 = 7) :
  ∀ n, a_n n = 2^(n - 1) :=
sorry

def T (n : ℕ) : ℕ := 4 - ((4 + 2 * n) / 2^n) -- Placeholder function for calculating T_n

theorem sum_first_n_terms (a_3_eq_2a_2 : 2^(3 - 1) = 2 * 2^(2 - 1)) (S3_eq_7 : S 3 = 7) :
  ∀ n, T n = 4 - ((4 + 2*n) / 2^n) :=
sorry

end general_term_formula_sum_first_n_terms_l170_170555


namespace sufficient_but_not_necessary_condition_l170_170951

theorem sufficient_but_not_necessary_condition (k : ℝ) : 
  (k = 1 → ∃ x y : ℝ, x^2 + y^2 = 1 ∧ x - y + k = 0) ∧ 
  ¬(∀ k : ℝ, ∃ x y : ℝ, x^2 + y^2 = 1 ∧ x - y + k = 0 → k = 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l170_170951


namespace x_minus_y_eq_11_l170_170242

theorem x_minus_y_eq_11 (x y : ℝ) (h : |x - 6| + |y + 5| = 0) : x - y = 11 := by
  sorry

end x_minus_y_eq_11_l170_170242


namespace part_a_l170_170759

theorem part_a (x y : ℝ) (hx : x ≠ 1) (hy : y ≠ 1) (hxy : x * y ≠ 1) :
  (x * y) / (1 - x * y) = x / (1 - x) + y / (1 - y) :=
sorry

end part_a_l170_170759


namespace isosceles_right_triangle_ratio_l170_170023

theorem isosceles_right_triangle_ratio (a : ℝ) (h : a > 0) : (2 * a) / (Real.sqrt (a^2 + a^2)) = Real.sqrt 2 :=
by
  sorry

end isosceles_right_triangle_ratio_l170_170023


namespace count_perfect_square_factors_l170_170990

theorem count_perfect_square_factors : 
  let n := (2^10) * (3^12) * (5^15) * (7^7)
  ∃ (count : ℕ), count = 1344 ∧
    (∀ (a b c d : ℕ), 0 ≤ a ∧ a ≤ 10 ∧ 0 ≤ b ∧ b ≤ 12 ∧ 0 ≤ c ∧ c ≤ 15 ∧ 0 ≤ d ∧ d ≤ 7 →
      ((a % 2 = 0) ∧ (b % 2 = 0) ∧ (c % 2 = 0) ∧ (d % 2 = 0) →
        ∃ (k : ℕ), (2^a * 3^b * 5^c * 7^d) = k ∧ k ∣ n)) :=
by
  sorry

end count_perfect_square_factors_l170_170990


namespace sqrt_neg9_squared_l170_170651

theorem sqrt_neg9_squared : Real.sqrt ((-9: ℝ)^2) = 9 := by
  sorry

end sqrt_neg9_squared_l170_170651


namespace gas_volume_at_31_degrees_l170_170235

theorem gas_volume_at_31_degrees :
  (∀ T V : ℕ, (T = 45 → V = 30) ∧ (∀ k, T = 45 - 2 * k → V = 30 - 3 * k)) →
  ∃ V, (T = 31) ∧ (V = 9) :=
by
  -- The proof would go here
  sorry

end gas_volume_at_31_degrees_l170_170235


namespace product_of_possible_values_of_x_l170_170374

theorem product_of_possible_values_of_x :
  (∃ x, |x - 7| - 3 = -2) → ∃ y z, |y - 7| - 3 = -2 ∧ |z - 7| - 3 = -2 ∧ y * z = 48 :=
by
  sorry

end product_of_possible_values_of_x_l170_170374


namespace beach_ball_problem_l170_170185

noncomputable def change_in_radius (C₁ C₂ : ℝ) : ℝ := (C₂ - C₁) / (2 * Real.pi)

noncomputable def volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r ^ 3

noncomputable def percentage_increase_in_volume (V₁ V₂ : ℝ) : ℝ := (V₂ - V₁) / V₁ * 100

theorem beach_ball_problem (C₁ C₂ : ℝ) (hC₁ : C₁ = 30) (hC₂ : C₂ = 36) :
  change_in_radius C₁ C₂ = 3 / Real.pi ∧
  percentage_increase_in_volume (volume (C₁ / (2 * Real.pi))) (volume (C₂ / (2 * Real.pi))) = 72.78 :=
by
  sorry

end beach_ball_problem_l170_170185


namespace max_sum_square_pyramid_addition_l170_170557

def square_pyramid_addition_sum (faces edges vertices : ℕ) : ℕ :=
  let new_faces := faces - 1 + 4
  let new_edges := edges + 4
  let new_vertices := vertices + 1
  new_faces + new_edges + new_vertices

theorem max_sum_square_pyramid_addition :
  square_pyramid_addition_sum 6 12 8 = 34 :=
by
  sorry

end max_sum_square_pyramid_addition_l170_170557


namespace tedra_tomato_harvest_l170_170151

theorem tedra_tomato_harvest (W T F : ℝ) 
    (h1 : T = W / 2) 
    (h2 : W + T + F = 2000) 
    (h3 : F - 700 = 700) : 
    W = 400 := 
sorry

end tedra_tomato_harvest_l170_170151


namespace area_of_triangle_OAB_is_5_l170_170958

-- Define the parameters and assumptions
def OA : ℝ × ℝ := (-2, 1)
def OB : ℝ × ℝ := (4, 3)

noncomputable def area_triangle_OAB (OA OB : ℝ × ℝ) : ℝ :=
  1 / 2 * (OA.1 * OB.2 - OA.2 * OB.1)

-- The theorem we want to prove:
theorem area_of_triangle_OAB_is_5 : area_triangle_OAB OA OB = 5 := by
  sorry

end area_of_triangle_OAB_is_5_l170_170958


namespace time_to_cross_bridge_l170_170080

noncomputable def train_length := 300  -- in meters
noncomputable def train_speed_kmph := 72  -- in km/h
noncomputable def bridge_length := 1500  -- in meters

-- Define the conversion from km/h to m/s
noncomputable def train_speed_mps := (train_speed_kmph * 1000) / 3600  -- in m/s

-- Define the total distance to be traveled
noncomputable def total_distance := train_length + bridge_length  -- in meters

-- Define the time to cross the bridge
noncomputable def time_to_cross := total_distance / train_speed_mps  -- in seconds

theorem time_to_cross_bridge : time_to_cross = 90 := by
  -- skipping the proof
  sorry

end time_to_cross_bridge_l170_170080


namespace exists_integers_x_l170_170841

theorem exists_integers_x (a1 a2 a3 : ℤ) (h : 0 < a1 ∧ a1 < a2 ∧ a2 < a3) :
  ∃ (x1 x2 x3 : ℤ), (|x1| + |x2| + |x3| > 0) ∧ (a1 * x1 + a2 * x2 + a3 * x3 = 0) ∧ (max (max (|x1|) (|x2|)) (|x3|) < (2 / Real.sqrt 3 * Real.sqrt a3) + 1) := 
sorry

end exists_integers_x_l170_170841


namespace price_of_movie_ticket_l170_170261

theorem price_of_movie_ticket
  (M F : ℝ)
  (h1 : 8 * M = 2 * F)
  (h2 : 8 * M + 5 * F = 840) :
  M = 30 :=
by
  sorry

end price_of_movie_ticket_l170_170261


namespace students_owning_both_pets_l170_170865

theorem students_owning_both_pets:
  ∀ (students total students_dog students_cat : ℕ),
    total = 45 →
    students_dog = 28 →
    students_cat = 38 →
    -- Each student owning at least one pet means 
    -- total = students_dog ∪ students_cat
    total = students_dog + students_cat - students →
    students = 21 :=
by
  intros students total students_dog students_cat h_total h_dog h_cat h_union
  sorry

end students_owning_both_pets_l170_170865


namespace find_minimum_value_l170_170522

noncomputable def f (x y : ℝ) : ℝ :=
  x^2 + 6*y^2 - 2*x*y - 14*x - 6*y + 72

theorem find_minimum_value :
  let x := 9
  let y := 2
  (∀ x y : ℝ, f x y ≥ 3) ∧ (f 9 2 = 3) :=
by
  sorry

end find_minimum_value_l170_170522


namespace arithmetic_sequence_a5_l170_170354

theorem arithmetic_sequence_a5 (a_n : ℕ → ℝ) 
  (h_arith : ∀ n, a_n (n+1) - a_n n = a_n (n+2) - a_n (n+1))
  (h_condition : a_n 1 + a_n 9 = 10) :
  a_n 5 = 5 :=
sorry

end arithmetic_sequence_a5_l170_170354


namespace polygon_sides_eq_nine_l170_170851

theorem polygon_sides_eq_nine (n : ℕ) 
  (interior_sum : ℕ := (n - 2) * 180)
  (exterior_sum : ℕ := 360)
  (condition : interior_sum = 4 * exterior_sum - 180) : 
  n = 9 :=
by {
  sorry
}

end polygon_sides_eq_nine_l170_170851


namespace correct_formula_l170_170778

-- Given conditions
def table : List (ℕ × ℕ) := [(2, 0), (3, 2), (4, 6), (5, 12), (6, 20)]

-- Candidate formulas
def formulaA (x : ℕ) : ℕ := 2 * x - 4
def formulaB (x : ℕ) : ℕ := x^2 - 3 * x + 2
def formulaC (x : ℕ) : ℕ := x^3 - 3 * x^2 + 2 * x
def formulaD (x : ℕ) : ℕ := x^2 - 4 * x
def formulaE (x : ℕ) : ℕ := x^2 - 4

-- The statement to be proven
theorem correct_formula : ∀ (x y : ℕ), (x, y) ∈ table → y = formulaB x :=
by
  sorry

end correct_formula_l170_170778


namespace morgan_total_pens_l170_170694

def initial_red_pens : Nat := 65
def initial_blue_pens : Nat := 45
def initial_black_pens : Nat := 58
def initial_green_pens : Nat := 36
def initial_purple_pens : Nat := 27

def red_pens_given_away : Nat := 15
def blue_pens_given_away : Nat := 20
def green_pens_given_away : Nat := 10

def black_pens_bought : Nat := 12
def purple_pens_bought : Nat := 5

def final_red_pens : Nat := initial_red_pens - red_pens_given_away
def final_blue_pens : Nat := initial_blue_pens - blue_pens_given_away
def final_black_pens : Nat := initial_black_pens + black_pens_bought
def final_green_pens : Nat := initial_green_pens - green_pens_given_away
def final_purple_pens : Nat := initial_purple_pens + purple_pens_bought

def total_pens : Nat := final_red_pens + final_blue_pens + final_black_pens + final_green_pens + final_purple_pens

theorem morgan_total_pens : total_pens = 203 := 
by
  -- final_red_pens = 50
  -- final_blue_pens = 25
  -- final_black_pens = 70
  -- final_green_pens = 26
  -- final_purple_pens = 32
  -- Therefore, total_pens = 203
  sorry

end morgan_total_pens_l170_170694


namespace range_of_x_plus_y_l170_170511

theorem range_of_x_plus_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x * y - (x + y) = 1) : 
  x + y ≥ 2 + 2 * Real.sqrt 2 :=
sorry

end range_of_x_plus_y_l170_170511


namespace problem_proof_l170_170673

variable (P Q M N : ℝ)

axiom hp1 : M = 0.40 * Q
axiom hp2 : Q = 0.30 * P
axiom hp3 : N = 1.20 * P

theorem problem_proof : (M / N) = (1 / 10) := by
  sorry

end problem_proof_l170_170673


namespace Tim_has_7_times_more_l170_170260

-- Define the number of Dan's violet balloons
def Dan_violet_balloons : ℕ := 29

-- Define the number of Tim's violet balloons
def Tim_violet_balloons : ℕ := 203

-- Prove that the ratio of Tim's balloons to Dan's balloons is 7
theorem Tim_has_7_times_more (h : Tim_violet_balloons = 7 * Dan_violet_balloons) : 
  Tim_violet_balloons = 7 * Dan_violet_balloons := 
by {
  sorry
}

end Tim_has_7_times_more_l170_170260


namespace minimum_rows_required_l170_170563

theorem minimum_rows_required (total_students : ℕ) (max_students_per_school : ℕ) (seats_per_row : ℕ) (num_schools : ℕ) 
    (h_total_students : total_students = 2016) 
    (h_max_students_per_school : max_students_per_school = 45) 
    (h_seats_per_row : seats_per_row = 168) 
    (h_num_schools : num_schools = 46) : 
    ∃ (min_rows : ℕ), min_rows = 16 := 
by 
  -- Proof omitted
  sorry

end minimum_rows_required_l170_170563


namespace incorrect_operation_l170_170824

theorem incorrect_operation 
    (x y : ℝ) :
    (x - y) / (x + y) = (y - x) / (y + x) ↔ False := 
by 
  sorry

end incorrect_operation_l170_170824


namespace odds_against_C_win_l170_170341

def odds_against_winning (p : ℚ) : ℚ := (1 - p) / p

theorem odds_against_C_win (pA pB : ℚ) (hA : pA = 1/5) (hB : pB = 2/3) :
  odds_against_winning (1 - pA - pB) = 13 / 2 :=
by
  sorry

end odds_against_C_win_l170_170341


namespace trigonometric_cos_value_l170_170542

open Real

theorem trigonometric_cos_value (α : ℝ) (h : sin (α + π / 6) = 1 / 3) : 
  cos (2 * α - 2 * π / 3) = -7 / 9 := 
sorry

end trigonometric_cos_value_l170_170542


namespace exp_7pi_over_2_eq_i_l170_170441

theorem exp_7pi_over_2_eq_i : Complex.exp (7 * Real.pi * Complex.I / 2) = Complex.I :=
by
  sorry

end exp_7pi_over_2_eq_i_l170_170441


namespace vinnie_tips_l170_170837

variable (Paul Vinnie : ℕ)

def tips_paul := 14
def more_vinnie_than_paul := 16

theorem vinnie_tips :
  Vinnie = tips_paul + more_vinnie_than_paul :=
by
  unfold tips_paul more_vinnie_than_paul -- unfolding defined values
  exact sorry

end vinnie_tips_l170_170837


namespace min_a4_in_arithmetic_sequence_l170_170889

noncomputable def arithmetic_sequence_min_a4 (a1 d : ℝ) 
(S4 : ℝ := 4 * a1 + 6 * d)
(S5 : ℝ := 5 * a1 + 10 * d)
(a4 : ℝ := a1 + 3 * d) : Prop :=
  S4 ≤ 4 ∧ S5 ≥ 15 → a4 = 7

theorem min_a4_in_arithmetic_sequence (a1 d : ℝ) (h1 : 4 * a1 + 6 * d ≤ 4) 
(h2 : 5 * a1 + 10 * d ≥ 15) : 
arithmetic_sequence_min_a4 a1 d := 
by {
  sorry -- Proof is omitted
}

end min_a4_in_arithmetic_sequence_l170_170889


namespace sum_interest_l170_170200

noncomputable def simple_interest (P : ℝ) (R : ℝ) := P * R * 3 / 100

theorem sum_interest (P R : ℝ) (h : simple_interest P (R + 1) - simple_interest P R = 75) : P = 2500 :=
by
  sorry

end sum_interest_l170_170200


namespace option_A_is_incorrect_option_B_is_incorrect_option_C_is_correct_option_D_is_incorrect_l170_170561

variable (x y: ℝ)

theorem option_A_is_incorrect : 5 - 3 * (x + 1) ≠ 5 - 3 * x - 1 := 
by sorry

theorem option_B_is_incorrect : 2 - 4 * (x + 1/4) ≠ 2 - 4 * x + 1 := 
by sorry

theorem option_C_is_correct : 2 - 4 * (1/4 * x + 1) = 2 - x - 4 := 
by sorry

theorem option_D_is_incorrect : 2 * (x - 2) - 3 * (y - 1) ≠ 2 * x - 4 - 3 * y - 3 := 
by sorry

end option_A_is_incorrect_option_B_is_incorrect_option_C_is_correct_option_D_is_incorrect_l170_170561


namespace option_D_correct_l170_170003

-- Formal statement in Lean 4
theorem option_D_correct (m : ℝ) : 6 * m + (-2 - 10 * m) = -4 * m - 2 :=
by
  -- Proof is skipped per instruction
  sorry

end option_D_correct_l170_170003


namespace shaded_rectangle_area_l170_170875

def area_polygon : ℝ := 2016
def sides_polygon : ℝ := 18
def segments_persh : ℝ := 4

theorem shaded_rectangle_area :
  (area_polygon / sides_polygon) * segments_persh = 448 := 
sorry

end shaded_rectangle_area_l170_170875


namespace benny_number_of_kids_l170_170757

-- Define the conditions
def benny_has_dollars (d: ℕ): Prop := d = 360
def cost_per_apple (c: ℕ): Prop := c = 4
def apples_shared (num_kids num_apples: ℕ): Prop := num_apples = 5 * num_kids

-- State the main theorem
theorem benny_number_of_kids : 
  ∀ (d c k a : ℕ), benny_has_dollars d → cost_per_apple c → apples_shared k a → k = 18 :=
by
  intros d c k a hd hc ha
  -- The goal is to prove k = 18; use the provided conditions
  sorry

end benny_number_of_kids_l170_170757


namespace angle_C_is_pi_div_3_side_c_is_2_sqrt_3_l170_170786

-- Definitions of the sides and conditions in triangle
variables {a b c : ℝ} {A B C : ℝ}

-- Condition: a + b = 6
axiom sum_of_sides : a + b = 6

-- Condition: Area of triangle ABC is 2 * sqrt(3)
axiom area_of_triangle : 1/2 * a * b * Real.sin C = 2 * Real.sqrt 3

-- Condition: a cos B + b cos A = 2c cos C
axiom cos_condition : (a * Real.cos B + b * Real.cos A) / c = 2 * Real.cos C

-- Proof problem 1: Prove that C = π/3
theorem angle_C_is_pi_div_3 (h_cos : Real.cos C = 1/2) : C = Real.pi / 3 :=
sorry

-- Proof problem 2: Prove that c = 2 sqrt(3)
theorem side_c_is_2_sqrt_3 (h_sin : Real.sin C = Real.sqrt 3 / 2) : c = 2 * Real.sqrt 3 :=
sorry

end angle_C_is_pi_div_3_side_c_is_2_sqrt_3_l170_170786


namespace negation_of_one_even_is_all_odd_or_at_least_two_even_l170_170359

-- Definitions based on the problem conditions
def is_even (n : ℕ) : Prop := n % 2 = 0

def exactly_one_even (a b c : ℕ) : Prop :=
  (is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
  (¬ is_even a ∧ is_even b ∧ ¬ is_even c) ∨
  (¬ is_even a ∧ ¬ is_even b ∧ is_even c)

def all_odd (a b c : ℕ) : Prop :=
  ¬ is_even a ∧ ¬ is_even b ∧ ¬ is_even c

def at_least_two_even (a b c : ℕ) : Prop :=
  (is_even a ∧ is_even b) ∨
  (is_even a ∧ is_even c) ∨
  (is_even b ∧ is_even c)

-- The proposition to prove
theorem negation_of_one_even_is_all_odd_or_at_least_two_even (a b c : ℕ) :
  ¬ exactly_one_even a b c ↔ all_odd a b c ∨ at_least_two_even a b c :=
by sorry

end negation_of_one_even_is_all_odd_or_at_least_two_even_l170_170359


namespace factorize_cubic_expression_l170_170475

variable (a : ℝ)

theorem factorize_cubic_expression : a^3 - a = a * (a + 1) * (a - 1) := 
sorry

end factorize_cubic_expression_l170_170475


namespace hawks_score_l170_170419

theorem hawks_score (x y : ℕ) (h1 : x + y = 82) (h2 : x - y = 18) : y = 32 :=
sorry

end hawks_score_l170_170419


namespace biggest_number_in_ratio_l170_170138

theorem biggest_number_in_ratio (A B C D : ℕ) (h1 : 2 * D = 5 * A) (h2 : 3 * D = 5 * B) (h3 : 4 * D = 5 * C) (h_sum : A + B + C + D = 1344) : D = 480 := 
by
  sorry

end biggest_number_in_ratio_l170_170138


namespace A_formula_l170_170254

noncomputable def A (i : ℕ) (A₀ θ : ℝ) : ℝ :=
match i with
| 0     => A₀
| (i+1) => (A i A₀ θ * Real.cos θ + Real.sin θ) / (-A i A₀ θ * Real.sin θ + Real.cos θ)

theorem A_formula (A₀ θ : ℝ) (n : ℕ) :
  A n A₀ θ = (A₀ * Real.cos (n * θ) + Real.sin (n * θ)) / (-A₀ * Real.sin (n * θ) + Real.cos (n * θ)) :=
by
  sorry

end A_formula_l170_170254


namespace tree_height_l170_170283

theorem tree_height (B h : ℕ) (H : ℕ) (h_eq : h = 16) (B_eq : B = 12) (L : ℕ) (L_def : L ^ 2 = B ^ 2 + h ^ 2) (H_def : H = h + L) :
    H = 36 := by
  -- We do not need to provide the proof steps as per the instructions
  sorry

end tree_height_l170_170283


namespace solve_fractional_equation_l170_170827

theorem solve_fractional_equation (x : ℝ) (h : (3 / (x + 1) - 2 / (x - 1)) = 0) : x = 5 :=
sorry

end solve_fractional_equation_l170_170827


namespace sqrt_expression_evaluation_l170_170708

theorem sqrt_expression_evaluation : (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 1)))^4 = 3 + 2 * Real.sqrt 2 := 
by
  sorry

end sqrt_expression_evaluation_l170_170708


namespace same_heads_probability_l170_170663

theorem same_heads_probability
  (fair_coin : Real := 1/2)
  (biased_coin : Real := 5/8)
  (prob_Jackie_eq_Phil : Real := 77/225) :
  let m := 77
  let n := 225
  (m : ℕ) + (n : ℕ) = 302 := 
by {
  -- The proof would involve constructing the generating functions,
  -- calculating the sum of corresponding coefficients and showing that the
  -- resulting probability reduces to 77/225
  sorry
}

end same_heads_probability_l170_170663


namespace question1_solution_question2_solution_l170_170146

-- Definitions of the problem conditions
def f (x a : ℝ) : ℝ := abs (x - a)

-- First proof problem (Question 1)
theorem question1_solution (x : ℝ) : (f x 2) ≥ (4 - abs (x - 4)) ↔ (x ≥ 5 ∨ x ≤ 1) :=
by sorry

-- Second proof problem (Question 2)
theorem question2_solution (x : ℝ) (a : ℝ) (h_sol : 1 ≤ x ∧ x ≤ 2) 
  (h_ineq : abs (f (2 * x + a) a - 2 * f x a) ≤ 2) : a = 3 :=
by sorry

end question1_solution_question2_solution_l170_170146


namespace find_number_of_terms_l170_170180

variable {n : ℕ} {a : ℕ → ℤ}
variable (a_seq : ℕ → ℤ)

def sum_first_three_terms (a : ℕ → ℤ) : ℤ :=
  a 1 + a 2 + a 3

def sum_last_three_terms (n : ℕ) (a : ℕ → ℤ) : ℤ :=
  a (n-2) + a (n-1) + a n

def sum_all_terms (n : ℕ) (a : ℕ → ℤ) : ℤ :=
  (Finset.range n).sum a

theorem find_number_of_terms (h1 : sum_first_three_terms a_seq = 20)
    (h2 : sum_last_three_terms n a_seq = 130)
    (h3 : sum_all_terms n a_seq = 200) : n = 8 :=
sorry

end find_number_of_terms_l170_170180


namespace rectangle_area_l170_170929

-- Definitions:
variables (l w : ℝ)

-- Conditions:
def condition1 : Prop := l = 4 * w
def condition2 : Prop := 2 * l + 2 * w = 200

-- Theorem statement:
theorem rectangle_area (h1 : condition1 l w) (h2 : condition2 l w) : l * w = 1600 :=
sorry

end rectangle_area_l170_170929


namespace negate_universal_prop_l170_170548

theorem negate_universal_prop :
  (¬ ∀ x : ℝ, x^2 - 2*x + 2 > 0) ↔ ∃ x : ℝ, x^2 - 2*x + 2 ≤ 0 :=
sorry

end negate_universal_prop_l170_170548


namespace boat_speed_in_still_water_l170_170087

theorem boat_speed_in_still_water : 
  ∀ (V_b V_s : ℝ), 
  V_b + V_s = 15 → 
  V_b - V_s = 5 → 
  V_b = 10 :=
by
  intros V_b V_s h1 h2
  have h3 : 2 * V_b = 20 := by linarith
  linarith

end boat_speed_in_still_water_l170_170087


namespace quadratic_has_one_real_root_l170_170995

theorem quadratic_has_one_real_root (m : ℝ) (h : (6 * m)^2 - 4 * 1 * 4 * m = 0) : m = 4 / 9 :=
by sorry

end quadratic_has_one_real_root_l170_170995


namespace tan_add_formula_l170_170142

noncomputable def tan_subtract (a b : ℝ) : ℝ := (Real.tan a - Real.tan b) / (1 + Real.tan a * Real.tan b)
noncomputable def tan_add (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)

theorem tan_add_formula (α : ℝ) (hf : tan_subtract α (Real.pi / 4) = 1 / 4) :
  tan_add α (Real.pi / 4) = -4 :=
by
  sorry

end tan_add_formula_l170_170142


namespace shortest_chord_intercept_l170_170033

theorem shortest_chord_intercept (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 = 3 → x + m * y - m - 1 = 0 → m = 1) :=
sorry

end shortest_chord_intercept_l170_170033


namespace monotonic_increasing_interval_of_f_l170_170332

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.logb (1/2) (x^2))

theorem monotonic_increasing_interval_of_f : 
  (∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ < 0 ∧ -1 ≤ x₂ ∧ x₂ < 0 ∧ x₁ ≤ x₂ → f x₁ ≤ f x₂) ∧ 
  (∀ x : ℝ, f x ≥ 0) := sorry

end monotonic_increasing_interval_of_f_l170_170332


namespace pizza_topping_slices_l170_170705

theorem pizza_topping_slices 
  (total_slices pepperoni_slices mushroom_slices olive_slices : ℕ)
  (pepperoni_slices_has_at_least_one_topping : pepperoni_slices = 8)
  (mushroom_slices_has_at_least_one_topping : mushroom_slices = 12)
  (olive_slices_has_at_least_one_topping : olive_slices = 14)
  (total_slices_has_one_topping : total_slices = 16)
  (slices_with_at_least_one_topping : 8 + 12 + 14 - 2 * x = 16) :
  x = 9 :=
by
  sorry

end pizza_topping_slices_l170_170705


namespace exists_x_l170_170597

theorem exists_x (a b c : ℕ) (ha : 0 < a) (hc : 0 < c) :
  ∃ x : ℕ, (0 < x) ∧ (a ^ x + x) % c = b % c :=
sorry

end exists_x_l170_170597


namespace total_paid_is_201_l170_170392

def adult_ticket_price : ℕ := 8
def child_ticket_price : ℕ := 5
def total_tickets : ℕ := 33
def child_tickets : ℕ := 21
def adult_tickets : ℕ := total_tickets - child_tickets
def total_paid : ℕ := (child_tickets * child_ticket_price) + (adult_tickets * adult_ticket_price)

theorem total_paid_is_201 : total_paid = 201 :=
by
  sorry

end total_paid_is_201_l170_170392


namespace prime_square_sub_one_divisible_by_24_l170_170403

theorem prime_square_sub_one_divisible_by_24 (p : ℕ) (hp : p ≥ 5) (prime_p : Nat.Prime p) : 24 ∣ p^2 - 1 := by
  sorry

end prime_square_sub_one_divisible_by_24_l170_170403


namespace john_drove_total_distance_l170_170355

-- Define different rates and times for John's trip
def rate1 := 45 -- mph
def rate2 := 55 -- mph
def time1 := 2 -- hours
def time2 := 3 -- hours

-- Define the distances for each segment of the trip
def distance1 := rate1 * time1
def distance2 := rate2 * time2

-- Define the total distance
def total_distance := distance1 + distance2

-- The theorem to prove that John drove 255 miles in total
theorem john_drove_total_distance : total_distance = 255 :=
by
  sorry

end john_drove_total_distance_l170_170355


namespace least_integer_value_l170_170513

theorem least_integer_value :
  ∃ x : ℤ, (∀ x' : ℤ, (|3 * x' + 4| <= 18) → (x' >= x)) ∧ (|3 * x + 4| <= 18) ∧ x = -7 := 
sorry

end least_integer_value_l170_170513


namespace tim_weekly_earnings_l170_170144

-- Define the constants based on the conditions
def tasks_per_day : ℕ := 100
def pay_per_task : ℝ := 1.2
def days_per_week : ℕ := 6

-- Statement to prove Tim's weekly earnings
theorem tim_weekly_earnings : tasks_per_day * pay_per_task * days_per_week = 720 := by
  sorry

end tim_weekly_earnings_l170_170144


namespace sum_due_is_correct_l170_170526

-- Define constants for Banker's Discount and True Discount
def BD : ℝ := 288
def TD : ℝ := 240

-- Define Banker's Gain as the difference between BD and TD
def BG : ℝ := BD - TD

-- Define the sum due (S.D.) as the face value including True Discount and Banker's Gain
def SD : ℝ := TD + BG

-- Create a theorem to prove the sum due is Rs. 288
theorem sum_due_is_correct : SD = 288 :=
by
  -- Skipping proof with sorry; expect this statement to be true based on given conditions 
  sorry

end sum_due_is_correct_l170_170526


namespace brad_running_speed_l170_170749

-- Definitions based on the given conditions
def distance_between_homes : ℝ := 24
def maxwell_walking_speed : ℝ := 4
def maxwell_time_to_meet : ℝ := 3

/-- Brad's running speed is 6 km/h given the conditions of the problem. -/
theorem brad_running_speed : (distance_between_homes - (maxwell_walking_speed * maxwell_time_to_meet)) / (maxwell_time_to_meet - 1) = 6 := by
  sorry

end brad_running_speed_l170_170749


namespace symmetric_line_origin_l170_170811

theorem symmetric_line_origin (a b : ℝ) :
  (∀ (m n : ℝ), a * m + 3 * n = 9 → -m + 3 * -n + b = 0) ↔ a = -1 ∧ b = -9 :=
by
  sorry

end symmetric_line_origin_l170_170811


namespace period_is_seven_l170_170817

-- Define the conditions
def apples_per_sandwich (a : ℕ) := a = 4
def sandwiches_per_day (s : ℕ) := s = 10
def total_apples (t : ℕ) := t = 280

-- Define the question to prove the period
theorem period_is_seven (a s t d : ℕ) 
  (h1 : apples_per_sandwich a)
  (h2 : sandwiches_per_day s)
  (h3 : total_apples t)
  (h4 : d = t / (a * s)) 
  : d = 7 := 
sorry

end period_is_seven_l170_170817


namespace length_of_plot_57_meters_l170_170707

section RectangleProblem

variable (b : ℝ) -- breadth of the plot
variable (l : ℝ) -- length of the plot
variable (cost_per_meter : ℝ) -- cost per meter
variable (total_cost : ℝ) -- total cost

-- Given conditions
def length_eq_breadth_plus_14 (b l : ℝ) : Prop := l = b + 14
def cost_eq_perimeter_cost_per_meter (cost_per_meter total_cost perimeter : ℝ) : Prop :=
  total_cost = cost_per_meter * perimeter

-- Definition of perimeter
def perimeter (b l : ℝ) : ℝ := 2 * l + 2 * b

-- Problem statement
theorem length_of_plot_57_meters
  (h1 : length_eq_breadth_plus_14 b l)
  (h2 : cost_eq_perimeter_cost_per_meter cost_per_meter total_cost (perimeter b l))
  (h3 : cost_per_meter = 26.50)
  (h4 : total_cost = 5300) :
  l = 57 :=
by
  sorry

end RectangleProblem

end length_of_plot_57_meters_l170_170707


namespace fencing_required_l170_170166

theorem fencing_required
  (L : ℝ) (W : ℝ) (A : ℝ) (F : ℝ)
  (hL : L = 25)
  (hA : A = 880)
  (hArea : A = L * W)
  (hF : F = L + 2 * W) :
  F = 95.4 :=
by
  sorry

end fencing_required_l170_170166


namespace range_of_a_l170_170859

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Set.Ioo a (a + 1), ∃ f' : ℝ → ℝ, ∀ x, f' x = (x * Real.exp x) * (x + 2) ∧ f' x = 0) ↔ 
  a ∈ Set.Ioo (-3 : ℝ) (-2) ∪ Set.Ioo (-1) (0) := 
sorry

end range_of_a_l170_170859


namespace product_of_distinct_integers_l170_170270

def is2008thPower (n : ℕ) : Prop := ∃ k : ℕ, n = k ^ 2008

theorem product_of_distinct_integers {x y z : ℕ} (h1 : x ≠ y) (h2 : y ≠ z) (h3 : z ≠ x)
  (h4 : y = (x + z) / 2) (h5 : x > 0) (h6 : y > 0) (h7 : z > 0) 
  : is2008thPower (x * y * z) :=
  sorry

end product_of_distinct_integers_l170_170270


namespace ball_arrangement_divisibility_l170_170808

theorem ball_arrangement_divisibility :
  ∀ (n : ℕ), (∀ (i : ℕ), i < n → (∃ j k l m : ℕ, j < k ∧ k < l ∧ l < m ∧ m < n ∧ j ≠ k ∧ k ≠ l ∧ l ≠ m ∧ m ≠ j
    ∧ i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m)) →
  ¬((n = 2021) ∨ (n = 2022) ∨ (n = 2023) ∨ (n = 2024)) :=
sorry

end ball_arrangement_divisibility_l170_170808


namespace find_z_l170_170077

open Complex

theorem find_z (z : ℂ) (h1 : (z + 2 * I).im = 0) (h2 : ((z / (2 - I)).im = 0)) : z = 4 - 2 * I :=
by
  sorry

end find_z_l170_170077


namespace company_stores_l170_170480

theorem company_stores (total_uniforms : ℕ) (uniforms_per_store : ℕ) 
  (h1 : total_uniforms = 121) (h2 : uniforms_per_store = 4) : 
  total_uniforms / uniforms_per_store = 30 :=
by
  sorry

end company_stores_l170_170480


namespace min_value_of_expression_l170_170510

theorem min_value_of_expression (x : ℝ) (hx : 0 < x) : 4 * x + 1 / x ^ 6 ≥ 5 :=
sorry

end min_value_of_expression_l170_170510


namespace compare_squares_l170_170463

theorem compare_squares (a b c : ℝ) : a^2 + b^2 + c^2 ≥ ab + bc + ca :=
by
  sorry

end compare_squares_l170_170463


namespace minimum_omega_l170_170997

/-- Given function f and its properties, determine the minimum valid ω. -/
theorem minimum_omega {f : ℝ → ℝ} 
  (Hf : ∀ x : ℝ, f x = (1 / 2) * Real.cos (ω * x + φ) + 1)
  (Hsymmetry : ∃ k : ℤ, ω * (π / 3) + φ = k * π)
  (Hvalue : ∃ n : ℤ, f (π / 12) = 1 ∧ ω * (π / 12) + φ = n * π + π / 2)
  (Hpos : ω > 0) : ω = 2 := 
sorry

end minimum_omega_l170_170997


namespace back_wheel_revolutions_calculation_l170_170534

noncomputable def front_diameter : ℝ := 3 -- Diameter of the front wheel in feet
noncomputable def back_diameter : ℝ := 0.5 -- Diameter of the back wheel in feet
noncomputable def no_slippage : Prop := true -- No slippage condition
noncomputable def front_revolutions : ℕ := 150 -- Number of front wheel revolutions

theorem back_wheel_revolutions_calculation 
  (d_f : ℝ) (d_b : ℝ) (slippage : Prop) (n_f : ℕ) : 
  slippage → d_f = front_diameter → d_b = back_diameter → 
  n_f = front_revolutions → 
  ∃ n_b : ℕ, n_b = 900 := 
by
  sorry

end back_wheel_revolutions_calculation_l170_170534


namespace math_problem_l170_170167

theorem math_problem (x y : ℝ) (h1 : x - 2 * y = 4) (h2 : x * y = 8) :
  x^2 + 4 * y^2 = 48 :=
sorry

end math_problem_l170_170167


namespace baker_sold_cakes_l170_170994

theorem baker_sold_cakes :
  ∀ (C : ℕ),  -- C is the number of cakes Baker sold
    (∃ (cakes pastries : ℕ), 
      cakes = 14 ∧ 
      pastries = 153 ∧ 
      (∃ (sold_pastries : ℕ), sold_pastries = 8 ∧ 
      C = 89 + sold_pastries)) 
  → C = 97 :=
by
  intros C h
  rcases h with ⟨cakes, pastries, cakes_eq, pastries_eq, ⟨sold_pastries, sold_pastries_eq, C_eq⟩⟩
  -- Fill in the proof details
  sorry

end baker_sold_cakes_l170_170994


namespace achieve_100_with_fewer_threes_example1_achieve_100_with_fewer_threes_example2_l170_170107

theorem achieve_100_with_fewer_threes_example1 :
  ((333 / 3) - (33 / 3) = 100) :=
by
  sorry

theorem achieve_100_with_fewer_threes_example2 :
  ((33 * 3) + (3 / 3) = 100) :=
by
  sorry

end achieve_100_with_fewer_threes_example1_achieve_100_with_fewer_threes_example2_l170_170107


namespace number_of_pencils_l170_170258

theorem number_of_pencils (P : ℕ) (h : ∃ (n : ℕ), n * 4 = P) : ∃ k, 4 * k = P :=
  by
  sorry

end number_of_pencils_l170_170258


namespace min_mn_value_l170_170633

theorem min_mn_value (m n : ℕ) (hmn : m > n) (hn : n ≥ 1) 
  (hdiv : 1000 ∣ 1978 ^ m - 1978 ^ n) : m + n = 106 :=
sorry

end min_mn_value_l170_170633


namespace jenny_kenny_reunion_time_l170_170769

/-- Define initial conditions given in the problem --/
def jenny_initial_pos : ℝ × ℝ := (-60, 100)
def kenny_initial_pos : ℝ × ℝ := (-60, -100)
def building_radius : ℝ := 60
def kenny_speed : ℝ := 4
def jenny_speed : ℝ := 2
def distance_apa : ℝ := 200
def initial_distance : ℝ := 200

theorem jenny_kenny_reunion_time : ∃ t : ℚ, 
  (t = (10 * (Real.sqrt 35)) / 7) ∧ 
  (17 = (10 + 7)) :=
by
  -- conditions to be used
  let jenny_pos (t : ℝ) := (-60 + 2 * t, 100)
  let kenny_pos (t : ℝ) := (-60 + 4 * t, -100)
  let circle_eq (x y : ℝ) := (x^2 + y^2 = building_radius^2)
  
  sorry

end jenny_kenny_reunion_time_l170_170769


namespace sufficient_but_not_necessary_l170_170693

def p (x : ℝ) : Prop := x > 0
def q (x : ℝ) : Prop := |x| > 0

theorem sufficient_but_not_necessary (x : ℝ) : 
  (p x → q x) ∧ (¬(q x → p x)) :=
by
  sorry

end sufficient_but_not_necessary_l170_170693


namespace divide_milk_into_equal_parts_l170_170773

def initial_state : (ℕ × ℕ × ℕ) := (8, 0, 0)

def is_equal_split (state : ℕ × ℕ × ℕ) : Prop :=
  state.1 = 4 ∧ state.2 = 4

theorem divide_milk_into_equal_parts : 
  ∃ (state_steps : Fin 25 → ℕ × ℕ × ℕ),
  initial_state = state_steps 0 ∧
  is_equal_split (state_steps 24) :=
sorry

end divide_milk_into_equal_parts_l170_170773


namespace minimum_area_triangle_ABC_l170_170955

-- Define the vertices of the triangle
def A : ℤ × ℤ := (0,0)
def B : ℤ × ℤ := (30,18)

-- Define a function to calculate the area of the triangle using the Shoelace formula
def area_of_triangle (A B C : ℤ × ℤ) : ℤ := 15 * (C.2).natAbs

-- State the theorem
theorem minimum_area_triangle_ABC : 
  ∀ C : ℤ × ℤ, C ≠ (0,0) → area_of_triangle A B C ≥ 15 :=
by
  sorry -- Skip the proof

end minimum_area_triangle_ABC_l170_170955


namespace line_intersection_l170_170848

noncomputable def line1 (t : ℚ) : ℚ × ℚ := (1 - 2 * t, 4 + 3 * t)
noncomputable def line2 (u : ℚ) : ℚ × ℚ := (5 + u, 2 + 6 * u)

theorem line_intersection :
  ∃ t u : ℚ, line1 t = (21 / 5, -4 / 5) ∧ line2 u = (21 / 5, -4 / 5) :=
sorry

end line_intersection_l170_170848


namespace Earl_rate_36_l170_170493

theorem Earl_rate_36 (E : ℝ) (h1 : E + (2 / 3) * E = 60) : E = 36 :=
by {
  sorry
}

end Earl_rate_36_l170_170493


namespace triangle_side_lengths_approx_l170_170983

noncomputable def approx_side_lengths (AB : ℝ) (BAC ABC : ℝ) : ℝ × ℝ :=
  let α := BAC * Real.pi / 180
  let β := ABC * Real.pi / 180
  let c := AB
  let β1 := (90 - (BAC)) * Real.pi / 180
  let m := 2 * c * α * (β1 + 3) / (9 - α * β1)
  let c1 := 2 * c * β1 * (α + 3) / (9 - α * β1)
  let β2 := β1 - β
  let γ1 := α + β
  let a1 := β2 / γ1 * (γ1 + 3) / (β2 + 3) * m
  let a := (9 - β2 * γ1) / (2 * γ1 * (β2 + 3)) * m
  let b := c1 - a1
  (a, b)

theorem triangle_side_lengths_approx (AB : ℝ) (BAC ABC : ℝ) (hAB : AB = 441) (hBAC : BAC = 16.2) (hABC : ABC = 40.6) :
  approx_side_lengths AB BAC ABC = (147, 344) := by
  sorry

end triangle_side_lengths_approx_l170_170983


namespace jamestown_theme_parks_l170_170699

theorem jamestown_theme_parks (J : ℕ) (Venice := J + 25) (MarinaDelRay := J + 50) (total := J + Venice + MarinaDelRay) (h : total = 135) : J = 20 :=
by
  -- proof step to be done here
  sorry

end jamestown_theme_parks_l170_170699


namespace distance_to_airport_l170_170895

theorem distance_to_airport
  (t : ℝ)
  (d : ℝ)
  (h1 : 45 * (t + 1) + 20 = d)
  (h2 : d - 65 = 65 * (t - 1))
  : d = 390 := by
  sorry

end distance_to_airport_l170_170895


namespace smallest_lcm_l170_170262

theorem smallest_lcm (a b : ℕ) (h₁ : 1000 ≤ a ∧ a < 10000) (h₂ : 1000 ≤ b ∧ b < 10000) (h₃ : Nat.gcd a b = 5) : 
  Nat.lcm a b = 201000 :=
sorry

end smallest_lcm_l170_170262


namespace smallest_total_students_l170_170974

theorem smallest_total_students :
  (∃ (n : ℕ), 4 * n + (n + 2) > 50 ∧ ∀ m, 4 * m + (m + 2) > 50 → m ≥ n) → 4 * 10 + (10 + 2) = 52 :=
by
  sorry

end smallest_total_students_l170_170974


namespace fraction_one_bedroom_apartments_l170_170764

theorem fraction_one_bedroom_apartments :
  ∃ x : ℝ, (x + 0.33 = 0.5) ∧ x = 0.17 :=
by
  sorry

end fraction_one_bedroom_apartments_l170_170764


namespace find_B_l170_170195

theorem find_B (A B : ℝ) : (1 / 4 * 1 / 8 = 1 / (4 * A) ∧ 1 / 32 = 1 / B) → B = 32 := by
  intros h
  sorry

end find_B_l170_170195


namespace exists_multiple_sum_divides_l170_170224

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_multiple_sum_divides {n : ℕ} (hn : n > 0) :
  ∃ (n_ast : ℕ), n ∣ n_ast ∧ sum_of_digits n_ast ∣ n_ast :=
by
  sorry

end exists_multiple_sum_divides_l170_170224


namespace tim_kittens_l170_170688

theorem tim_kittens (initial_kittens : ℕ) (given_to_jessica_fraction : ℕ) (saras_kittens : ℕ) (adopted_fraction : ℕ) 
  (h_initial : initial_kittens = 12)
  (h_fraction_to_jessica : given_to_jessica_fraction = 3)
  (h_saras_kittens : saras_kittens = 14)
  (h_adopted_fraction : adopted_fraction = 2) :
  let kittens_after_jessica := initial_kittens - initial_kittens / given_to_jessica_fraction
  let total_kittens_after_sara := kittens_after_jessica + saras_kittens
  let adopted_kittens := saras_kittens / adopted_fraction
  let final_kittens := total_kittens_after_sara - adopted_kittens
  final_kittens = 15 :=
by {
  sorry
}

end tim_kittens_l170_170688


namespace mrs_hilt_current_rocks_l170_170659

-- Definitions based on conditions
def total_rocks_needed : ℕ := 125
def more_rocks_needed : ℕ := 61

-- Lean statement proving the required amount of currently held rocks
theorem mrs_hilt_current_rocks : (total_rocks_needed - more_rocks_needed) = 64 :=
by
  -- proof will be here
  sorry

end mrs_hilt_current_rocks_l170_170659


namespace total_students_l170_170512

-- Define the conditions
def ratio_girls_boys (G B : ℕ) : Prop := G / B = 1 / 2
def ratio_math_girls (M N : ℕ) : Prop := M / N = 3 / 1
def ratio_sports_boys (S T : ℕ) : Prop := S / T = 4 / 1

-- Define the problem statement
theorem total_students (G B M N S T : ℕ) 
  (h1 : ratio_girls_boys G B)
  (h2 : ratio_math_girls M N)
  (h3 : ratio_sports_boys S T)
  (h4 : M = 12)
  (h5 : G = M + N)
  (h6 : G = 16) 
  (h7 : B = 32) : 
  G + B = 48 :=
sorry

end total_students_l170_170512


namespace sequence_divisibility_condition_l170_170649

theorem sequence_divisibility_condition (t a b x1 : ℕ) (x : ℕ → ℕ)
  (h1 : a = 1) (h2 : b = t) (h3 : x1 = t) (h4 : x 1 = x1)
  (h5 : ∀ n, n ≥ 2 → x n = a * x (n - 1) + b) :
  (∀ m n, m ∣ n → x m ∣ x n) ↔ (a = 1 ∧ b = t ∧ x1 = t) := sorry

end sequence_divisibility_condition_l170_170649


namespace no_real_roots_iff_l170_170446

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2*x + a

theorem no_real_roots_iff (a : ℝ) : (∀ x : ℝ, f x a ≠ 0) → a > 1 :=
  by
    sorry

end no_real_roots_iff_l170_170446


namespace golden_section_AP_length_l170_170092

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

noncomputable def golden_ratio_recip : ℝ := (Real.sqrt 5 - 1) / 2

theorem golden_section_AP_length (AB : ℝ) (P : ℝ) 
  (h1 : AB = 2) (h2 : P = golden_ratio_recip * AB) : 
  P = Real.sqrt 5 - 1 :=
by
  sorry

end golden_section_AP_length_l170_170092


namespace hcf_of_numbers_is_five_l170_170574

theorem hcf_of_numbers_is_five (a b x : ℕ) (ratio : a = 3 * x) (ratio_b : b = 4 * x)
  (lcm_ab : Nat.lcm a b = 60) (hcf_ab : Nat.gcd a b = 5) : Nat.gcd a b = 5 :=
by
  sorry

end hcf_of_numbers_is_five_l170_170574


namespace circle_value_in_grid_l170_170069

theorem circle_value_in_grid :
  ∃ (min_circle_val : ℕ), min_circle_val = 21 ∧ (∀ (max_circle_val : ℕ), ∃ (L : ℕ), L > max_circle_val) :=
by
  sorry

end circle_value_in_grid_l170_170069


namespace range_of_k_for_circle_l170_170804

theorem range_of_k_for_circle (x y : ℝ) (k : ℝ) : 
  (x^2 + y^2 - 4*x + 2*y + 5*k = 0) → k < 1 :=
by 
  sorry

end range_of_k_for_circle_l170_170804


namespace faster_train_length_225_l170_170515

noncomputable def length_of_faster_train (speed_slower speed_faster : ℝ) (time : ℝ) : ℝ :=
  let relative_speed_kmph := speed_slower + speed_faster
  let relative_speed_mps := (relative_speed_kmph * 1000) / 3600
  relative_speed_mps * time

theorem faster_train_length_225 :
  length_of_faster_train 36 45 10 = 225 := by
  sorry

end faster_train_length_225_l170_170515


namespace find_constant_l170_170867

theorem find_constant (x1 x2 : ℝ) (C : ℝ) :
  x1 - x2 = 5.5 ∧
  x1 + x2 = -5 / 2 ∧
  x1 * x2 = C / 2 →
  C = -12 :=
by
  -- proof goes here
  sorry

end find_constant_l170_170867


namespace fraction_area_of_shaded_square_in_larger_square_is_one_eighth_l170_170845

theorem fraction_area_of_shaded_square_in_larger_square_is_one_eighth :
  let side_larger_square := 4
  let area_larger_square := side_larger_square^2
  let side_shaded_square := Real.sqrt (1^2 + 1^2)
  let area_shaded_square := side_shaded_square^2
  area_shaded_square / area_larger_square = 1 / 8 := 
by 
  sorry

end fraction_area_of_shaded_square_in_larger_square_is_one_eighth_l170_170845


namespace angle_of_inclination_l170_170831

theorem angle_of_inclination (x y : ℝ) (θ : ℝ) :
  (x - y - 1 = 0) → θ = 45 :=
by
  sorry

end angle_of_inclination_l170_170831


namespace min_value_theorem_l170_170391

noncomputable def min_value (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1) : ℝ :=
  1/a + 2/b

theorem min_value_theorem (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1) :
  min_value a b h₀ h₁ h₂ ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_value_theorem_l170_170391


namespace transformed_interval_l170_170000

noncomputable def transformation (x : ℝ) : ℝ := 8 * x - 2

theorem transformed_interval :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → -2 ≤ transformation x ∧ transformation x ≤ 6 := by
  intro x h
  unfold transformation
  sorry

end transformed_interval_l170_170000


namespace graph_fixed_point_l170_170667

theorem graph_fixed_point {a : ℝ} (h₁ : a > 0) (h₂ : a ≠ 1) : 
  ∃ A : ℝ × ℝ, A = (-2, -1) ∧ ∀ x : ℝ, y = a^(x + 2) - 2 ↔ (x, y) = A := 
by 
  sorry

end graph_fixed_point_l170_170667


namespace lily_remaining_money_l170_170053

def initial_amount := 55
def spent_on_shirt := 7
def spent_at_second_shop := 3 * spent_on_shirt
def total_spent := spent_on_shirt + spent_at_second_shop
def remaining_amount := initial_amount - total_spent

theorem lily_remaining_money : remaining_amount = 27 :=
by
  sorry

end lily_remaining_money_l170_170053


namespace find_fg3_l170_170229

def f (x : ℝ) : ℝ := 2 * x - 5
def g (x : ℝ) : ℝ := x^2 + 1

theorem find_fg3 : f (g 3) = 15 :=
by
  sorry

end find_fg3_l170_170229


namespace option_a_solution_l170_170814

theorem option_a_solution (x y : ℕ) (h₁: x = 2) (h₂: y = 2) : 2 * x + y = 6 := by
sorry

end option_a_solution_l170_170814


namespace point_not_in_first_quadrant_l170_170905

theorem point_not_in_first_quadrant (m n : ℝ) (h : m * n ≤ 0) : ¬ (m > 0 ∧ n > 0) :=
sorry

end point_not_in_first_quadrant_l170_170905


namespace cost_of_one_unit_each_l170_170585

variables (x y z : ℝ)

theorem cost_of_one_unit_each
  (h1 : 2 * x + 3 * y + z = 130)
  (h2 : 3 * x + 5 * y + z = 205) :
  x + y + z = 55 :=
by
  sorry

end cost_of_one_unit_each_l170_170585


namespace equivalent_statements_l170_170495

-- Definitions based on the problem
def is_not_negative (x : ℝ) : Prop := x >= 0
def is_not_positive (x : ℝ) : Prop := x <= 0
def is_positive (x : ℝ) : Prop := x > 0
def is_negative (x : ℝ) : Prop := x < 0

-- The main theorem statement
theorem equivalent_statements (x : ℝ) : 
  (is_not_negative x → is_not_positive (x^2)) ↔ (is_positive (x^2) → is_negative x) :=
by
  sorry

end equivalent_statements_l170_170495


namespace total_fish_sold_l170_170404

-- Define the conditions
def w1 : ℕ := 50
def w2 : ℕ := 3 * w1

-- Define the statement to prove
theorem total_fish_sold : w1 + w2 = 200 := by
  -- Insert the proof here 
  -- (proof omitted as per the instructions)
  sorry

end total_fish_sold_l170_170404


namespace abs_value_solutions_l170_170762

theorem abs_value_solutions (x : ℝ) : abs x = 6.5 ↔ x = 6.5 ∨ x = -6.5 :=
by
  sorry

end abs_value_solutions_l170_170762


namespace bicycle_speed_l170_170078

theorem bicycle_speed (x : ℝ) :
  (10 / x = 10 / (2 * x) + 1 / 3) → x = 15 :=
by
  intro h
  sorry

end bicycle_speed_l170_170078


namespace find_strawberry_jelly_amount_l170_170681

noncomputable def strawberry_jelly (t b : ℕ) : ℕ := t - b

theorem find_strawberry_jelly_amount (h₁ : 6310 = 4518 + s) : s = 1792 := by
  sorry

end find_strawberry_jelly_amount_l170_170681


namespace correct_operation_l170_170858

theorem correct_operation (a b : ℝ) : ((-3 * a^2 * b)^2 = 9 * a^4 * b^2) := sorry

end correct_operation_l170_170858


namespace percent_of_x_l170_170756

variable {x y z : ℝ}

-- Define the given conditions
def cond1 (z y : ℝ) : Prop := 0.45 * z = 0.9 * y
def cond2 (z x : ℝ) : Prop := z = 1.5 * x

-- State the theorem to prove
theorem percent_of_x (h1 : cond1 z y) (h2 : cond2 z x) : y = 0.75 * x :=
sorry

end percent_of_x_l170_170756


namespace largest_n_factors_l170_170437

theorem largest_n_factors (n : ℤ) :
  (∃ A B : ℤ, 3 * B + A = n ∧ A * B = 72) → n ≤ 217 :=
by {
  sorry
}

end largest_n_factors_l170_170437


namespace quadratic_unbounded_above_l170_170906

theorem quadratic_unbounded_above : ∀ (x y : ℝ), ∃ M : ℝ, ∀ z : ℝ, M < (2 * x^2 + 4 * x * y + 5 * y^2 + 8 * x - 6 * y + z) :=
by
  intro x y
  use 1000 -- Example to denote that for any point greater than 1000
  intro z
  have h1 : 2 * x^2 + 4 * x * y + 5 * y^2 + 8 * x - 6 * y + z ≥ 2 * 0^2 + 4 * 0 * y + 5 * y^2 + 8 * 0 - 6 * y + z := by sorry
  sorry

end quadratic_unbounded_above_l170_170906


namespace distance_halfway_along_orbit_l170_170456

variable {Zeta : Type}  -- Zeta is a type representing the planet
variable (distance_from_focus : Zeta → ℝ)  -- Function representing the distance from the sun (focus)

-- Conditions
variable (perigee_distance : ℝ := 3)
variable (apogee_distance : ℝ := 15)
variable (a : ℝ := (perigee_distance + apogee_distance) / 2)  -- semi-major axis

theorem distance_halfway_along_orbit (z : Zeta) (h1 : distance_from_focus z = perigee_distance) (h2 : distance_from_focus z = apogee_distance) :
  distance_from_focus z = a :=
sorry

end distance_halfway_along_orbit_l170_170456


namespace tire_price_l170_170734

theorem tire_price (x : ℝ) (h : 3 * x + 10 = 310) : x = 100 :=
sorry

end tire_price_l170_170734


namespace find_x_value_l170_170407

theorem find_x_value (x : ℝ) (y : ℝ) (h1 : y = (x^2 - 9) / (x - 3)) (h2 : y = 3 * x - 4) : x = 7 / 2 := 
sorry

end find_x_value_l170_170407


namespace sum_set_15_l170_170819

noncomputable def sum_nth_set (n : ℕ) : ℕ :=
  let first_element := 1 + (n - 1) * n / 2
  let last_element := first_element + n - 1
  n * (first_element + last_element) / 2

theorem sum_set_15 : sum_nth_set 15 = 1695 :=
  by sorry

end sum_set_15_l170_170819


namespace gerald_total_pieces_eq_672_l170_170070

def pieces_per_table : Nat := 12
def pieces_per_chair : Nat := 8
def num_tables : Nat := 24
def num_chairs : Nat := 48

def total_pieces : Nat := pieces_per_table * num_tables + pieces_per_chair * num_chairs

theorem gerald_total_pieces_eq_672 : total_pieces = 672 :=
by
  sorry

end gerald_total_pieces_eq_672_l170_170070


namespace sufficient_not_necessary_condition_l170_170491

-- Definitions of propositions
def propA (x : ℝ) : Prop := (x - 1)^2 < 9
def propB (x a : ℝ) : Prop := (x + 2) * (x + a) < 0

-- Lean statement of the problem
theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x, propA x → propB x a) ∧ (∃ x, ¬ propA x ∧ propB x a) ↔ a < -4 :=
sorry

end sufficient_not_necessary_condition_l170_170491


namespace herman_days_per_week_l170_170312

-- Defining the given conditions as Lean definitions
def total_meals : ℕ := 4
def cost_per_meal : ℕ := 4
def total_weeks : ℕ := 16
def total_cost : ℕ := 1280

-- Calculating derived facts based on given conditions
def cost_per_day : ℕ := total_meals * cost_per_meal
def cost_per_week : ℕ := total_cost / total_weeks

-- Our main theorem that states Herman buys breakfast combos 5 days per week
theorem herman_days_per_week : cost_per_week / cost_per_day = 5 :=
by
  -- Skipping the proof
  sorry

end herman_days_per_week_l170_170312


namespace ice_cream_orders_l170_170857

variables (V C S M O T : ℕ)

theorem ice_cream_orders :
  (V = 56) ∧ (C = 28) ∧ (S = 70) ∧ (M = 42) ∧ (O = 84) ↔
  (V = 2 * C) ∧
  (S = 25 * T / 100) ∧
  (M = 15 * T / 100) ∧
  (T = 280) ∧
  (V = 20 * T / 100) ∧
  (V + C + S + M + O = T) :=
by
  sorry

end ice_cream_orders_l170_170857


namespace not_divisible_by_n_l170_170753

theorem not_divisible_by_n (n : ℕ) (h : n > 1) : ¬n ∣ 2^n - 1 :=
by
  -- proof to be filled in
  sorry

end not_divisible_by_n_l170_170753


namespace green_peaches_count_l170_170782

def red_peaches : ℕ := 17
def green_peaches (x : ℕ) : Prop := red_peaches = x + 1

theorem green_peaches_count (x : ℕ) (h : green_peaches x) : x = 16 :=
by
  sorry

end green_peaches_count_l170_170782


namespace smallest_repunit_divisible_by_97_l170_170178

theorem smallest_repunit_divisible_by_97 :
  ∃ n : ℕ, (∃ d : ℤ, 10^n - 1 = 97 * 9 * d) ∧ (∀ m : ℕ, (∃ d : ℤ, 10^m - 1 = 97 * 9 * d) → n ≤ m) :=
by
  sorry

end smallest_repunit_divisible_by_97_l170_170178


namespace floor_sub_y_eq_zero_l170_170632

theorem floor_sub_y_eq_zero {y : ℝ} (h : ⌊y⌋ + ⌈y⌉ = 2 * y) : ⌊y⌋ - y = 0 :=
sorry

end floor_sub_y_eq_zero_l170_170632


namespace population_2002_l170_170256

-- Predicate P for the population of rabbits in a given year
def P : ℕ → ℝ := sorry

-- Given conditions
axiom cond1 : ∃ k : ℝ, P 2003 - P 2001 = k * P 2002
axiom cond2 : ∃ k : ℝ, P 2002 - P 2000 = k * P 2001
axiom condP2000 : P 2000 = 50
axiom condP2001 : P 2001 = 80
axiom condP2003 : P 2003 = 186

-- The statement we need to prove
theorem population_2002 : P 2002 = 120 :=
by
  sorry

end population_2002_l170_170256


namespace clean_house_time_l170_170668

theorem clean_house_time (B A: ℝ) (h1: A = 1/12) (h2: B + A = 1/4):
  (B + 2 * A) = 1/3 → 1 / (B + 2 * A) = 3 :=
by
  -- Proof omitted.
  sorry

end clean_house_time_l170_170668


namespace jeanne_additional_tickets_l170_170081

-- Define the costs
def ferris_wheel_cost : ℕ := 5
def roller_coaster_cost : ℕ := 4
def bumper_cars_cost : ℕ := 4
def jeanne_tickets : ℕ := 5

-- Calculate the total cost
def total_cost : ℕ := ferris_wheel_cost + roller_coaster_cost + bumper_cars_cost

-- Define the proof problem
theorem jeanne_additional_tickets : total_cost - jeanne_tickets = 8 :=
by sorry

end jeanne_additional_tickets_l170_170081


namespace parabola_focus_distance_l170_170568

theorem parabola_focus_distance
  (F P Q : ℝ × ℝ)
  (hF : F = (1 / 2, 0))
  (hP : ∃ y, P = (2 * y^2, y))
  (hQ : Q = (1 / 2, Q.2))
  (h_parallel : P.2 = Q.2)
  (h_distance : dist P Q = dist Q F) :
  dist P F = 2 :=
by
  sorry

end parabola_focus_distance_l170_170568


namespace directrix_eqn_of_parabola_l170_170772

theorem directrix_eqn_of_parabola : 
  ∀ y : ℝ, x = - (1 / 4 : ℝ) * y ^ 2 → x = 1 :=
by
  sorry

end directrix_eqn_of_parabola_l170_170772


namespace correct_statement_l170_170088

variable {a b : Type} -- Let a and b be types representing lines
variable {α β : Type} -- Let α and β be types representing planes

-- Define parallel relations for lines and planes
def parallel (L P : Type) : Prop := sorry

-- Define the subset relation for lines in planes
def subset (L P : Type) : Prop := sorry

-- Now state the theorem corresponding to the correct answer
theorem correct_statement (h1 : parallel α β) (h2 : subset a α) : parallel a β :=
sorry

end correct_statement_l170_170088


namespace solve_system_eqs_l170_170697
noncomputable section

theorem solve_system_eqs (x y z : ℝ) :
  (x * y = 5 * (x + y) ∧ x * z = 4 * (x + z) ∧ y * z = 2 * (y + z))
  ↔ (x = 0 ∧ y = 0 ∧ z = 0)
  ∨ (x = -40 ∧ y = 40 / 9 ∧ z = 40 / 11) := sorry

end solve_system_eqs_l170_170697


namespace less_than_reciprocal_l170_170682

theorem less_than_reciprocal (n : ℚ) : 
  n = -3 ∨ n = 3/4 ↔ (n = -1/2 → n >= 1/(-1/2)) ∧
                           (n = -3 → n < 1/(-3)) ∧
                           (n = 3/4 → n < 1/(3/4)) ∧
                           (n = 3 → n > 1/3) ∧
                           (n = 0 → false) := sorry

end less_than_reciprocal_l170_170682


namespace factorization_of_polynomial_l170_170363

noncomputable def p (x : ℤ) : ℤ := x^15 + x^10 + x^5 + 1
noncomputable def f (x : ℤ) : ℤ := x^3 + x^2 + x + 1
noncomputable def g (x : ℤ) : ℤ := x^12 - x^11 + x^9 - x^8 + x^6 - x^5 + x^3 - x^2 + x - 1

theorem factorization_of_polynomial : ∀ x : ℤ, p x = f x * g x :=
by sorry

end factorization_of_polynomial_l170_170363


namespace problem_solution_l170_170323

theorem problem_solution (a b c d e : ℤ) (h : (x - 3)^4 = ax^4 + bx^3 + cx^2 + dx + e) :
  b + c + d + e = 15 :=
by
  sorry

end problem_solution_l170_170323


namespace investment_value_change_l170_170886

theorem investment_value_change (k m : ℝ) : 
  let increaseFactor := 1 + k / 100
  let decreaseFactor := 1 - m / 100 
  let overallFactor := increaseFactor * decreaseFactor 
  let changeFactor := overallFactor - 1
  let percentageChange := changeFactor * 100 
  percentageChange = k - m - (k * m) / 100 := 
by 
  sorry

end investment_value_change_l170_170886


namespace range_of_a_l170_170073

noncomputable def geometric_seq (r : ℝ) (n : ℕ) (a₁ : ℝ) : ℝ := a₁ * r ^ (n - 1)

theorem range_of_a (a : ℝ) :
  (∃ a_seq b_seq : ℕ → ℝ, a_seq 1 = a ∧ (∀ n, b_seq n = (a_seq n - 2) / (a_seq n - 1)) ∧ (∀ n, a_seq n > a_seq (n+1)) ∧ (∀ n, b_seq (n + 1) = geometric_seq (2/3) (n + 1) (b_seq 1))) → 2 < a :=
by
  sorry

end range_of_a_l170_170073


namespace number_of_ways_to_win_championships_l170_170972

-- Definitions for the problem
def num_athletes := 5
def num_events := 3

-- Proof statement
theorem number_of_ways_to_win_championships : 
  (num_athletes ^ num_events) = 125 := 
by 
  sorry

end number_of_ways_to_win_championships_l170_170972


namespace triangle_area_l170_170046

theorem triangle_area (a b c : ℕ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (right_triangle : a^2 + b^2 = c^2) : 
  (1 / 2) * a * b = 54 := by
    sorry

end triangle_area_l170_170046


namespace percentage_of_one_pair_repeated_digits_l170_170803

theorem percentage_of_one_pair_repeated_digits (n : ℕ) (h1 : 10000 ≤ n) (h2 : n ≤ 99999) :
  ∃ (percentage : ℝ), percentage = 56.0 :=
by
  sorry

end percentage_of_one_pair_repeated_digits_l170_170803
