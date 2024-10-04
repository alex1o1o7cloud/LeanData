import Mathlib

namespace universal_proposition_l241_241067

def is_multiple_of_two (x : ℕ) : Prop :=
  ∃ k : ℕ, x = 2 * k

def is_even (x : ℕ) : Prop :=
  ∃ k : ℕ, x = 2 * k

theorem universal_proposition : 
  (∀ x : ℕ, is_multiple_of_two x → is_even x) :=
by
  sorry

end universal_proposition_l241_241067


namespace correct_average_wrong_reading_l241_241430

theorem correct_average_wrong_reading
  (initial_average : ℕ) (list_length : ℕ) (wrong_number : ℕ) (correct_number : ℕ) (correct_average : ℕ) 
  (h1 : initial_average = 18)
  (h2 : list_length = 10)
  (h3 : wrong_number = 26)
  (h4 : correct_number = 66)
  (h5 : correct_average = 22) :
  correct_average = ((initial_average * list_length) - wrong_number + correct_number) / list_length :=
sorry

end correct_average_wrong_reading_l241_241430


namespace binomial_12_10_eq_66_l241_241455

theorem binomial_12_10_eq_66 : binomial 12 10 = 66 :=
by
  sorry

end binomial_12_10_eq_66_l241_241455


namespace candy_box_original_price_l241_241127

theorem candy_box_original_price (P : ℝ) (h1 : 1.25 * P = 20) : P = 16 :=
sorry

end candy_box_original_price_l241_241127


namespace pencils_added_by_sara_l241_241572

-- Definitions based on given conditions
def original_pencils : ℕ := 115
def total_pencils : ℕ := 215

-- Statement to prove
theorem pencils_added_by_sara : total_pencils - original_pencils = 100 :=
by {
  -- Proof
  sorry
}

end pencils_added_by_sara_l241_241572


namespace remainder_of_9_6_plus_8_7_plus_7_8_mod_7_l241_241418

theorem remainder_of_9_6_plus_8_7_plus_7_8_mod_7 : (9^6 + 8^7 + 7^8) % 7 = 2 := 
by sorry

end remainder_of_9_6_plus_8_7_plus_7_8_mod_7_l241_241418


namespace drying_time_short_haired_dog_l241_241800

theorem drying_time_short_haired_dog (x : ℕ) (h1 : ∀ y, y = 2 * x) (h2 : 6 * x + 9 * (2 * x) = 240) : x = 10 :=
by
  sorry

end drying_time_short_haired_dog_l241_241800


namespace dan_minimum_speed_to_beat_cara_l241_241047

theorem dan_minimum_speed_to_beat_cara
  (distance : ℕ) (cara_speed : ℕ) (dan_delay : ℕ) :
  distance = 120 →
  cara_speed = 30 →
  dan_delay = 1 →
  ∃ (dan_speed : ℕ), dan_speed > 40 :=
by
  sorry

end dan_minimum_speed_to_beat_cara_l241_241047


namespace abs_nonneg_position_l241_241567

theorem abs_nonneg_position (a : ℝ) : 0 ≤ |a| ∧ |a| ≥ 0 → (exists x : ℝ, x = |a| ∧ x ≥ 0) :=
by 
  sorry

end abs_nonneg_position_l241_241567


namespace probability_factor_of_36_l241_241274

theorem probability_factor_of_36 : 
  let pos_ints := { n : ℕ | 1 ≤ n ∧ n ≤ 36 }
  let num_divisors := { d : ℕ | d ∣ 36 }
  (num_divisors.to_finset.card : ℚ) / (pos_ints.to_finset.card : ℚ) = 1 / 4 :=
by {
  sorry
}

end probability_factor_of_36_l241_241274


namespace ball_speed_is_20_l241_241447

def ball_flight_time : ℝ := 8
def collie_speed : ℝ := 5
def collie_catch_time : ℝ := 32

noncomputable def collie_distance : ℝ := collie_speed * collie_catch_time

theorem ball_speed_is_20 :
  collie_distance = ball_flight_time * 20 :=
by
  sorry

end ball_speed_is_20_l241_241447


namespace parabola_equation_l241_241002

theorem parabola_equation (p : ℝ) (h_pos : p > 0) (M : ℝ) (h_Mx : M = 3) (h_MF : abs (M + p/2) = 2 * p) :
  (forall x y, y^2 = 2 * p * x) -> (forall x y, y^2 = 4 * x) :=
by
  sorry

end parabola_equation_l241_241002


namespace length_of_AB_l241_241427

-- Definitions based on given conditions:
variables (AB BC CD DE AE AC : ℕ)
variables (h1 : BC = 3 * CD) (h2 : DE = 8) (h3 : AC = 11) (h4 : AE = 21)

-- The theorem stating the length of AB given the conditions.
theorem length_of_AB (AB BC CD DE AE AC : ℕ)
  (h1 : BC = 3 * CD) (h2 : DE = 8) (h3 : AC = 11) (h4 : AE = 21) : AB = 5 := by
  sorry

end length_of_AB_l241_241427


namespace find_S10_value_l241_241781

noncomputable def sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
∀ n : ℕ, 4 * S n = n * (a n + a (n + 1))

theorem find_S10_value (a S : ℕ → ℕ) (h1 : a 4 = 7) (h2 : sequence_sum a S) :
  S 10 = 100 :=
sorry

end find_S10_value_l241_241781


namespace probability_factor_36_l241_241280

def is_factor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

theorem probability_factor_36 : 
  (set.univ.filter (λ n : ℕ, n > 0 ∧ n ≤ 36 ∧ is_factor 36 n)).card.toRat / 36 = 1 / 4 := 
by
  sorry

end probability_factor_36_l241_241280


namespace inequality_solution_set_l241_241037

   theorem inequality_solution_set : 
     {x : ℝ | (4 * x - 5)^2 + (3 * x - 2)^2 < (x - 3)^2} = {x : ℝ | (2 / 3 : ℝ) < x ∧ x < (5 / 4 : ℝ)} :=
   by
     sorry
   
end inequality_solution_set_l241_241037


namespace forty_percent_more_than_seventyfive_by_fifty_l241_241077

def number : ℝ := 312.5

theorem forty_percent_more_than_seventyfive_by_fifty 
    (x : ℝ) 
    (h : 0.40 * x = 0.75 * 100 + 50) : 
    x = number :=
by
  sorry

end forty_percent_more_than_seventyfive_by_fifty_l241_241077


namespace probability_factor_36_l241_241238

theorem probability_factor_36 : 
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 36 ∧ (∀ k : ℕ, k ∣ n ↔ k ∣ 36) → (n ≠ 0 → (∃ p : ℚ, p = 1 / 4))) :=
begin
  sorry
end

end probability_factor_36_l241_241238


namespace binom_12_10_eq_66_l241_241464

theorem binom_12_10_eq_66 : nat.choose 12 10 = 66 :=
by
  sorry

end binom_12_10_eq_66_l241_241464


namespace geometric_sum_of_first_four_terms_eq_120_l241_241951

theorem geometric_sum_of_first_four_terms_eq_120
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (ha2 : a 2 = 9)
  (ha5 : a 5 = 243) :
  a 1 * (1 - r^4) / (1 - r) = 120 := 
sorry

end geometric_sum_of_first_four_terms_eq_120_l241_241951


namespace cost_of_song_book_l241_241548

def cost_of_trumpet : ℝ := 145.16
def total_amount_spent : ℝ := 151.00

theorem cost_of_song_book : (total_amount_spent - cost_of_trumpet) = 5.84 := by
  sorry

end cost_of_song_book_l241_241548


namespace sum_first_20_terms_arithmetic_seq_l241_241779

theorem sum_first_20_terms_arithmetic_seq :
  ∃ (a d : ℤ) (S_20 : ℤ), d > 0 ∧
  (a + 2 * d) * (a + 6 * d) = -12 ∧
  (a + 3 * d) + (a + 5 * d) = -4 ∧
  S_20 = 20 * a + (20 * 19 / 2) * d ∧
  S_20 = 180 :=
by
  sorry

end sum_first_20_terms_arithmetic_seq_l241_241779


namespace winner_percentage_l241_241949

theorem winner_percentage (total_votes winner_votes : ℕ) (h1 : winner_votes = 744) (h2 : total_votes - winner_votes = 288) :
  (winner_votes : ℤ) * 100 / total_votes = 62 := 
by
  sorry

end winner_percentage_l241_241949


namespace john_total_distance_l241_241024

-- Define the parameters according to the conditions
def daily_distance : ℕ := 1700
def number_of_days : ℕ := 6
def total_distance : ℕ := daily_distance * number_of_days

-- Lean theorem statement to prove the total distance run by John
theorem john_total_distance : total_distance = 10200 := by
  -- Here, the proof would go, but it is omitted as per instructions
  sorry

end john_total_distance_l241_241024


namespace factor_probability_l241_241265

theorem factor_probability : 
  let S := { n : ℕ | n > 0 ∧ n ≤ 36 } in
  let factor_count := (multiplicity 2 36 + 1) * (multiplicity 3 36 + 1) in
  let total_count := 36 in
  (factor_count : ℚ) / total_count = (1 : ℚ) / 4 := 
by
  sorry

end factor_probability_l241_241265


namespace problem_cos_tan_half_l241_241944

open Real

theorem problem_cos_tan_half
  (α : ℝ)
  (hcos : cos α = -4/5)
  (hquad : π < α ∧ α < 3 * π / 2) :
  (1 + tan (α / 2)) / (1 - tan (α / 2)) = -1 / 2 :=
  sorry

end problem_cos_tan_half_l241_241944


namespace work_completion_days_l241_241437

noncomputable def A_days : ℝ := 20
noncomputable def B_days : ℝ := 35
noncomputable def C_days : ℝ := 50

noncomputable def A_work_rate : ℝ := 1 / A_days
noncomputable def B_work_rate : ℝ := 1 / B_days
noncomputable def C_work_rate : ℝ := 1 / C_days

noncomputable def combined_work_rate : ℝ := A_work_rate + B_work_rate + C_work_rate
noncomputable def total_days : ℝ := 1 / combined_work_rate

theorem work_completion_days : total_days = 700 / 69 :=
by
  -- Proof steps would go here
  sorry

end work_completion_days_l241_241437


namespace log_x_y_eq_sqrt_3_l241_241397

variable (x y z : ℝ)
variable (hx : 1 < x) (hy : 1 < y) (hz : 1 < z)
variable (h1 : x ^ (Real.log z / Real.log y) = 2)
variable (h2 : y ^ (Real.log x / Real.log y) = 4)
variable (h3 : z ^ (Real.log y / Real.log x) = 8)

theorem log_x_y_eq_sqrt_3 : Real.log y / Real.log x = Real.sqrt 3 :=
by
  sorry

end log_x_y_eq_sqrt_3_l241_241397


namespace original_radius_of_cylinder_in_inches_l241_241650

theorem original_radius_of_cylinder_in_inches
  (r : ℝ) (h : ℝ) (V : ℝ → ℝ → ℝ → ℝ) 
  (h_increased_radius : V (r + 4) h π = V r (h + 4) π) 
  (h_original_height : h = 3) :
  r = 8 :=
by
  sorry

end original_radius_of_cylinder_in_inches_l241_241650


namespace reflect_origin_l241_241176

theorem reflect_origin (x y : ℝ) (h₁ : x = 4) (h₂ : y = -3) : 
  (-x, -y) = (-4, 3) :=
by {
  sorry
}

end reflect_origin_l241_241176


namespace reciprocal_of_neg_5_l241_241216

theorem reciprocal_of_neg_5 : (∃ r : ℚ, -5 * r = 1) ∧ r = -1 / 5 :=
by sorry

end reciprocal_of_neg_5_l241_241216


namespace total_reading_materials_l241_241098

theorem total_reading_materials 
  (books_per_shelf : ℕ) (magazines_per_shelf : ℕ) (newspapers_per_shelf : ℕ) (graphic_novels_per_shelf : ℕ) 
  (bookshelves : ℕ)
  (h_books : books_per_shelf = 23) 
  (h_magazines : magazines_per_shelf = 61) 
  (h_newspapers : newspapers_per_shelf = 17) 
  (h_graphic_novels : graphic_novels_per_shelf = 29) 
  (h_bookshelves : bookshelves = 37) : 
  (books_per_shelf * bookshelves + magazines_per_shelf * bookshelves + newspapers_per_shelf * bookshelves + graphic_novels_per_shelf * bookshelves) = 4810 := 
by {
  -- Condition definitions are already given; the proof is omitted here.
  sorry
}

end total_reading_materials_l241_241098


namespace magnitude_of_vec_sum_l241_241153

noncomputable def vec_a : ℝ × ℝ := (Real.cos (5 * Real.pi / 180), Real.sin (5 * Real.pi / 180))
noncomputable def vec_b : ℝ × ℝ := (Real.cos (65 * Real.pi / 180), Real.sin (65 * Real.pi / 180))
noncomputable def vec_sum : ℝ × ℝ := (vec_a.1 + 2 * vec_b.1, vec_a.2 + 2 * vec_b.2)
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem magnitude_of_vec_sum : magnitude vec_sum = Real.sqrt 7 := 
by 
  sorry

end magnitude_of_vec_sum_l241_241153


namespace probability_from_first_to_last_l241_241877

noncomputable def probability_path_possible (n : ℕ) : ℝ :=
  (2 ^ (n - 1) : ℝ) / (Nat.choose (2 * (n - 1)) (n - 1) : ℝ)

theorem probability_from_first_to_last (n : ℕ) (h : n > 1) :
  probability_path_possible n = 2 ^ (n - 1) / (Nat.choose (2 * (n - 1)) (n - 1)) := sorry

end probability_from_first_to_last_l241_241877


namespace find_g_eq_minus_x_l241_241031

-- Define the function g and the given conditions.
def g (x : ℝ) : ℝ := sorry

axiom g0 : g 0 = 2
axiom g_xy : ∀ (x y : ℝ), g (x * y) = g ((x^2 + 2 * y^2) / 3) + 3 * (x - y)^2

-- State the problem: proving that g(x) = -x.
theorem find_g_eq_minus_x : ∀ (x : ℝ), g x = -x := by
  sorry

end find_g_eq_minus_x_l241_241031


namespace find_p_l241_241422

open LinearMap

-- Define the vectors a, b, and the result p as constant vectors in ℝ^3
def a : ℝ^3 := ⟨1, -1, 2⟩
def b : ℝ^3 := ⟨-1, 2, 1⟩
def p : ℝ^3 := ⟨0, 1 / 2, 3 / 2⟩

-- Helper function to determine collinearity by a scalar multiple
def collinear (u v : ℝ^3) : Prop := ∃ k : ℝ, u = k • v

-- Statement of the problem where p is the given vector satisfying specified conditions
theorem find_p (v : ℝ^3) 
  (h1 : ∃ t : ℝ, p = a + t • (b - a))
  (h2 : collinear a p) 
  (h3 : collinear b p) :
  p = ⟨0, 1 / 2, 3 / 2⟩ := 
sorry

end find_p_l241_241422


namespace polynomial_is_perfect_cube_l241_241334

theorem polynomial_is_perfect_cube (p q n : ℚ) :
  (∃ a : ℚ, x^3 + p * x^2 + q * x + n = (x + a)^3) ↔ (q = p^2 / 3 ∧ n = p^3 / 27) :=
by sorry

end polynomial_is_perfect_cube_l241_241334


namespace find_f_2_l241_241940

noncomputable def f (a b x : ℝ) : ℝ := a * x^5 + b * x^3 - x + 2

theorem find_f_2 (a b : ℝ)
  (h : f a b (-2) = 5) : f a b 2 = -1 :=
by 
  sorry

end find_f_2_l241_241940


namespace evaluate_expression_l241_241100

theorem evaluate_expression :
  (Real.sqrt 5 * 5^(1/2) + 20 / 4 * 3 - 8^(3/2) + 5) = 25 - 16 * Real.sqrt 2 := 
by
  sorry

end evaluate_expression_l241_241100


namespace train_length_is_correct_l241_241428

noncomputable def lengthOfTrain (speed_km_hr : ℝ) (time_s : ℝ) : ℝ :=
  let speed_m_s := speed_km_hr * 1000 / 3600
  speed_m_s * time_s

theorem train_length_is_correct : lengthOfTrain 60 15 = 250.05 :=
by
  sorry

end train_length_is_correct_l241_241428


namespace max_value_of_3_pow_x_minus_9_pow_x_l241_241125

open Real

theorem max_value_of_3_pow_x_minus_9_pow_x :
  ∃ (x : ℝ), ∀ y : ℝ, 3^x - 9^x ≤ 3^y - 9^y := 
sorry

end max_value_of_3_pow_x_minus_9_pow_x_l241_241125


namespace yuebao_scientific_notation_l241_241446

-- Definition of converting a number to scientific notation
def scientific_notation (x : ℝ) (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ x = a * 10 ^ n

-- The specific problem statement
theorem yuebao_scientific_notation :
  scientific_notation (1853 * 10 ^ 9) 1.853 11 :=
by
  sorry

end yuebao_scientific_notation_l241_241446


namespace valentines_given_l241_241388

-- Let x be the number of boys and y be the number of girls
variables (x y : ℕ)

-- Condition 1: the number of valentines is 28 more than the total number of students.
axiom valentines_eq : x * y = x + y + 28

-- Theorem: Prove that the total number of valentines given is 60.
theorem valentines_given : x * y = 60 :=
by
  sorry

end valentines_given_l241_241388


namespace major_axis_length_l241_241601

theorem major_axis_length (r : ℝ) (minor_axis major_axis : ℝ) 
  (h1 : r = 2) 
  (h2 : minor_axis = 2 * r) 
  (h3 : major_axis = minor_axis + 0.8 * minor_axis) :
  major_axis = 7.2 :=
sorry

end major_axis_length_l241_241601


namespace range_m_l241_241929

open Real

theorem range_m (m : ℝ)
  (hP : ¬ (∃ x : ℝ, m * x^2 + 1 ≤ 0))
  (hQ : ¬ (∃ x : ℝ, x^2 + m * x + 1 < 0)) :
  0 ≤ m ∧ m ≤ 2 := 
sorry

end range_m_l241_241929


namespace jade_pieces_left_l241_241530

-- Define the initial number of pieces Jade has
def initial_pieces : Nat := 100

-- Define the number of pieces per level
def pieces_per_level : Nat := 7

-- Define the number of levels in the tower
def levels : Nat := 11

-- Define the resulting number of pieces Jade has left after building the tower
def pieces_left : Nat := initial_pieces - (pieces_per_level * levels)

-- The theorem stating that after building the tower, Jade has 23 pieces left
theorem jade_pieces_left : pieces_left = 23 := by
  -- Proof omitted
  sorry

end jade_pieces_left_l241_241530


namespace right_triangle_similarity_l241_241889

theorem right_triangle_similarity (y : ℝ) (h : 12 / y = 9 / 7) : y = 9.33 := 
by 
  sorry

end right_triangle_similarity_l241_241889


namespace f_2019_value_l241_241808

noncomputable def f : ℕ → ℕ := sorry

theorem f_2019_value
  (h : ∀ m n : ℕ, f (m + n) ≥ f m + f (f n) - 1) :
  f 2019 = 2019 :=
sorry

end f_2019_value_l241_241808


namespace solve_congruence_l241_241483

open Nat

theorem solve_congruence (x : ℕ) (h : x^2 + x - 6 ≡ 0 [MOD 143]) : 
  x = 2 ∨ x = 41 ∨ x = 101 ∨ x = 140 :=
by
  sorry

end solve_congruence_l241_241483


namespace arithmetic_sequence_max_sum_l241_241402

-- Condition: first term is 23
def a1 : ℤ := 23

-- Condition: common difference is -2
def d : ℤ := -2

-- Sum of the first n terms of the arithmetic sequence
def Sn (n : ℕ) : ℤ := n * a1 + (n * (n - 1)) / 2 * d

-- Problem Statement: Prove the maximum value of Sn(n)
theorem arithmetic_sequence_max_sum : ∃ n : ℕ, Sn n = 144 :=
sorry

end arithmetic_sequence_max_sum_l241_241402


namespace fifth_graders_more_than_seventh_l241_241656

theorem fifth_graders_more_than_seventh (price_per_pencil : ℕ) (price_per_pencil_pos : price_per_pencil > 0)
    (total_cents_7th : ℕ) (total_cents_7th_val : total_cents_7th = 201)
    (total_cents_5th : ℕ) (total_cents_5th_val : total_cents_5th = 243)
    (pencil_price_div_7th : total_cents_7th % price_per_pencil = 0)
    (pencil_price_div_5th : total_cents_5th % price_per_pencil = 0) :
    (total_cents_5th / price_per_pencil - total_cents_7th / price_per_pencil = 14) := 
by
    sorry

end fifth_graders_more_than_seventh_l241_241656


namespace factor_probability_36_l241_241287

-- Definitions based on conditions
def is_factor (n d : ℕ) : Prop := d ∣ n
def num_factors (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (is_factor n).card

def probability_factor (n : ℕ) : ℚ := num_factors n / n

-- Problem statement
theorem factor_probability_36 : probability_factor 36 = 1 / 4 :=
  by sorry

end factor_probability_36_l241_241287


namespace sin_675_eq_neg_sqrt2_div_2_l241_241756

theorem sin_675_eq_neg_sqrt2_div_2 : Real.sin (675 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  -- Proof goes here
  sorry

end sin_675_eq_neg_sqrt2_div_2_l241_241756


namespace find_A_l241_241856

theorem find_A (A B : ℕ) (hA : A < 10) (hB : B < 10) (h : 10 * A + 3 + 610 + B = 695) : A = 8 :=
by {
  sorry
}

end find_A_l241_241856


namespace third_median_length_l241_241091

theorem third_median_length 
  (m_A m_B : ℝ) -- lengths of the first two medians
  (area : ℝ)   -- area of the triangle
  (h_median_A : m_A = 5) -- the first median is 5 inches
  (h_median_B : m_B = 8) -- the second median is 8 inches
  (h_area : area = 6 * Real.sqrt 15) -- the area of the triangle is 6√15 square inches
  : ∃ m_C : ℝ, m_C = Real.sqrt 31 := -- the length of the third median is √31
sorry

end third_median_length_l241_241091


namespace total_bananas_in_collection_l241_241399

theorem total_bananas_in_collection (g b T : ℕ) (h₀ : g = 196) (h₁ : b = 2) (h₂ : T = 392) : g * b = T :=
by
  sorry

end total_bananas_in_collection_l241_241399


namespace lucille_house_difference_l241_241818

def height_lucille : ℕ := 80
def height_neighbor1 : ℕ := 70
def height_neighbor2 : ℕ := 99

def average_height (h1 h2 h3 : ℕ) : ℕ := (h1 + h2 + h3) / 3

def difference (h_average h_actual : ℕ) : ℕ := h_average - h_actual

theorem lucille_house_difference :
  difference (average_height height_lucille height_neighbor1 height_neighbor2) height_lucille = 3 :=
by
  unfold difference
  unfold average_height
  sorry

end lucille_house_difference_l241_241818


namespace probability_divisor_of_36_is_one_fourth_l241_241282

noncomputable def probability_divisor_of_36 : ℚ :=
  let total_divisors := 9 in
  let total_integers := 36 in
  total_divisors / total_integers

theorem probability_divisor_of_36_is_one_fourth :
  probability_divisor_of_36 = 1 / 4 :=
by
  sorry

end probability_divisor_of_36_is_one_fourth_l241_241282


namespace jebb_total_spent_l241_241802

theorem jebb_total_spent
  (cost_of_food : ℝ) (service_fee_rate : ℝ) (tip : ℝ)
  (h1 : cost_of_food = 50)
  (h2 : service_fee_rate = 0.12)
  (h3 : tip = 5) :
  cost_of_food + (cost_of_food * service_fee_rate) + tip = 61 := 
sorry

end jebb_total_spent_l241_241802


namespace angle_sum_acutes_l241_241637

theorem angle_sum_acutes (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) 
  (h_condition : |Real.sin α - 1/2| + Real.sqrt ((Real.tan β - 1)^2) = 0) : 
  α + β = π * 5/12 :=
by sorry

end angle_sum_acutes_l241_241637


namespace productivity_increase_l241_241570

theorem productivity_increase (a b : ℝ) : (7 / 8) * (1 + 20 / 100) = 1.05 :=
by
  sorry

end productivity_increase_l241_241570


namespace jebb_total_spent_l241_241803

theorem jebb_total_spent
  (cost_of_food : ℝ) (service_fee_rate : ℝ) (tip : ℝ)
  (h1 : cost_of_food = 50)
  (h2 : service_fee_rate = 0.12)
  (h3 : tip = 5) :
  cost_of_food + (cost_of_food * service_fee_rate) + tip = 61 := 
sorry

end jebb_total_spent_l241_241803


namespace solve_cubic_equation_l241_241117

theorem solve_cubic_equation (x : ℝ) (h : x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3) : x = 6 :=
by sorry

end solve_cubic_equation_l241_241117


namespace bally_subset_count_l241_241599

-- Define what it means for a set to be Bally
def is_bally_set (S : Set ℕ) : Prop :=
  ∀ m ∈ S, (S.filter (< m)).card < m / 2

-- The explicit set we are considering
def big_set : Set ℕ := {i | 1 ≤ i ∧ i ≤ 2020}

-- The main theorem stating the number of Bally subsets
theorem bally_subset_count :
  {T : Set ℕ | T ⊆ big_set ∧ is_bally_set T}.card = binom 2021 1010 - 1 :=
by
  sorry

end bally_subset_count_l241_241599


namespace minimum_value_of_v_l241_241006

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x

noncomputable def g (x u v : ℝ) : ℝ := (x - u)^3 - 3 * (x - u) - v

theorem minimum_value_of_v (u v : ℝ) (h_pos_u : u > 0) :
  ∀ u > 0, ∀ x : ℝ, f x = g x u v → v ≥ 4 :=
by
  sorry

end minimum_value_of_v_l241_241006


namespace binom_12_10_eq_66_l241_241466

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_12_10_eq_66 : binom 12 10 = 66 := 
by
  -- We state the given condition for binomial coefficients.
  let binom_coeff := binom 12 10
  -- We will use the formula \binom{n}{k} = \frac{n!}{k!(n-k)!}.
  change binom_coeff = Nat.factorial 12 / (Nat.factorial 10 * Nat.factorial (12 - 10))
  -- Simplify using factorial values and basic arithmetic to reach the conclusion.
  sorry

end binom_12_10_eq_66_l241_241466


namespace trajectory_and_min_area_l241_241634

theorem trajectory_and_min_area (C : ℝ → ℝ → Prop) (P : ℝ × ℝ → Prop)
  (l : ℝ → ℝ) (F : ℝ × ℝ) (M : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ)
  (k : ℝ) : 
  (∀ x y, P (x, y) ↔ x ^ 2 = 4 * y) → 
  P (0, 1) →
  (∀ y, l y = -1) →
  F = (0, 1) →
  (∀ x1 y1 x2 y2, x1 + x2 = 4 * k → x1 * x2 = -4 →
    M (x1, y1) (x2, y2) = (2 * k, -1)) →
  (min_area : ℝ) → 
  min_area = 4 :=
by
  intros
  sorry

end trajectory_and_min_area_l241_241634


namespace canvas_decreased_by_40_percent_l241_241096

noncomputable def canvas_decrease (P C : ℝ) (x d : ℝ) : Prop :=
  (P = 4 * C) ∧
  ((P - 0.60 * P) + (C - (x / 100) * C) = (1 - d / 100) * (P + C)) ∧
  (d = 55.99999999999999)

theorem canvas_decreased_by_40_percent (P C : ℝ) (x d : ℝ) 
  (h : canvas_decrease P C x d) : x = 40 :=
by
  sorry

end canvas_decreased_by_40_percent_l241_241096


namespace all_terms_are_integers_l241_241590

   noncomputable def a : ℕ → ℤ
   | 0 => 1
   | 1 => 1
   | 2 => 997
   | n + 3 => (1993 + a (n + 2) * a (n + 1)) / a n

   theorem all_terms_are_integers : ∀ n : ℕ, ∃ (a : ℕ → ℤ), 
     (a 1 = 1) ∧ 
     (a 2 = 1) ∧ 
     (a 3 = 997) ∧ 
     (∀ n : ℕ, a (n + 3) = (1993 + a (n + 2) * a (n + 1)) / a n) → 
     (∀ n : ℕ, ∃ k : ℤ, a n = k) := 
   by 
     sorry
   
end all_terms_are_integers_l241_241590


namespace triangle_PQR_area_l241_241988

/-- Given a triangle PQR where PQ = 4 miles, PR = 2 miles, and PQ is along Pine Street
and PR is along Quail Road, and there is a sub-triangle PQS within PQR
with PS = 2 miles along Summit Avenue and QS = 3 miles along Pine Street,
prove that the area of triangle PQR is 4 square miles --/
theorem triangle_PQR_area :
  ∀ (PQ PR PS QS : ℝ),
    PQ = 4 → PR = 2 → PS = 2 → QS = 3 →
    (1/2) * PQ * PR = 4 :=
by
  intros PQ PR PS QS hpq hpr hps hqs
  rw [hpq, hpr]
  norm_num
  done

end triangle_PQR_area_l241_241988


namespace probability_divisor_of_36_l241_241295

theorem probability_divisor_of_36 :
  (∃ (n : ℕ) (h₁ : 1 ≤ n ∧ n ≤ 36), (36 % n = 0) → (1 / 4 : ℝ)) :=
by 
  sorry

end probability_divisor_of_36_l241_241295


namespace probability_factor_of_36_l241_241257

def is_factor_of (n d : ℕ) : Prop := d % n = 0

theorem probability_factor_of_36 : 
  let total := 36
  let factors_of_36 := { n | n ≤ 36 ∧ is_factor_of n 36 }
  (factors_of_36.to_finset.card : ℚ) / total = 1 / 4 := 
by
  sorry

end probability_factor_of_36_l241_241257


namespace brianne_yard_length_l241_241475

theorem brianne_yard_length 
  (derrick_yard_length : ℝ)
  (h₁ : derrick_yard_length = 10)
  (alex_yard_length : ℝ)
  (h₂ : alex_yard_length = derrick_yard_length / 2)
  (brianne_yard_length : ℝ)
  (h₃ : brianne_yard_length = 6 * alex_yard_length) :
  brianne_yard_length = 30 :=
by sorry

end brianne_yard_length_l241_241475


namespace original_wattage_l241_241597

theorem original_wattage (W : ℝ) (new_W : ℝ) (h1 : new_W = 1.25 * W) (h2 : new_W = 100) : W = 80 :=
by
  sorry

end original_wattage_l241_241597


namespace rowing_time_to_place_and_back_l241_241743

open Real

/-- Definitions of the problem conditions -/
def rowing_speed_still_water : ℝ := 5
def current_speed : ℝ := 1
def distance_to_place : ℝ := 2.4

/-- Proof statement: the total time taken to row to the place and back is 1 hour -/
theorem rowing_time_to_place_and_back :
  (distance_to_place / (rowing_speed_still_water + current_speed)) + 
  (distance_to_place / (rowing_speed_still_water - current_speed)) =
  1 := by
  sorry

end rowing_time_to_place_and_back_l241_241743


namespace probability_of_3_black_face_cards_l241_241485

-- Definitions based on conditions
def total_cards : ℕ := 36
def total_black_face_cards : ℕ := 8
def total_other_cards : ℕ := total_cards - total_black_face_cards
def draw_cards : ℕ := 6
def draw_black_face_cards : ℕ := 3
def draw_other_cards := draw_cards - draw_black_face_cards

-- Calculation using combinations
noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def total_combinations : ℕ := combination total_cards draw_cards
noncomputable def favorable_combinations : ℕ := combination total_black_face_cards draw_black_face_cards * combination total_other_cards draw_other_cards

-- Calculating probability
noncomputable def probability : ℚ := favorable_combinations / total_combinations

-- The theorem to be proved
theorem probability_of_3_black_face_cards : probability = 11466 / 121737 := by
  -- proof
  sorry

end probability_of_3_black_face_cards_l241_241485


namespace total_number_of_ways_to_choose_courses_l241_241086

theorem total_number_of_ways_to_choose_courses :
  (∑ b in finset.range 3, nat.choose 2 b * nat.choose 4 (4 - b)) - nat.choose 4 4 = 14 := 
by {
  have sum_cases := (nat.choose 2 1 * nat.choose 4 3) + (nat.choose 2 2 * nat.choose 4 2),
  have subtract_for_no_b := sum_cases - nat.choose 4 4,
  exact subtract_for_no_b,
  sorry
}

end total_number_of_ways_to_choose_courses_l241_241086


namespace system1_solution_correct_system2_solution_correct_l241_241826

theorem system1_solution_correct (x y : ℝ) (h1 : x + y = 5) (h2 : 4 * x - 2 * y = 2) :
    x = 2 ∧ y = 3 :=
  sorry

theorem system2_solution_correct (x y : ℝ) (h1 : 3 * x - 2 * y = 13) (h2 : 4 * x + 3 * y = 6) :
    x = 3 ∧ y = -2 :=
  sorry

end system1_solution_correct_system2_solution_correct_l241_241826


namespace gcd_of_differences_is_10_l241_241070

theorem gcd_of_differences_is_10 (a b c : ℕ) (h1 : b > a) (h2 : c > b) (h3 : c > a)
  (h4 : b - a = 20) (h5 : c - b = 50) (h6 : c - a = 70) : Int.gcd (b - a) (Int.gcd (c - b) (c - a)) = 10 := 
sorry

end gcd_of_differences_is_10_l241_241070


namespace equal_probability_of_selection_l241_241627

-- Define a structure representing the scenario of the problem.
structure SamplingProblem :=
  (total_students : ℕ)
  (eliminated_students : ℕ)
  (remaining_students : ℕ)
  (selection_size : ℕ)
  (systematic_step : ℕ)

-- Instantiate the specific problem.
def problem_instance : SamplingProblem :=
  { total_students := 3001
  , eliminated_students := 1
  , remaining_students := 3000
  , selection_size := 50
  , systematic_step := 60 }

-- Define the main theorem to be proven.
theorem equal_probability_of_selection (prob : SamplingProblem) :
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ prob.remaining_students → 
  (prob.remaining_students - prob.systematic_step * ((i - 1) / prob.systematic_step) = i) :=
sorry

end equal_probability_of_selection_l241_241627


namespace equal_values_of_means_l241_241605

theorem equal_values_of_means (f : ℤ × ℤ → ℤ) 
  (h_pos : ∀ p, 0 < f p)
  (h_mean : ∀ p, f p = (f (p.1 + 1, p.2) + f (p.1 - 1, p.2) + f (p.1, p.2 + 1) + f (p.1, p.2 - 1)) / 4):
  ∃ m : ℤ, ∀ p, f p = m := sorry

end equal_values_of_means_l241_241605


namespace probability_of_reaching_last_floor_l241_241875

noncomputable def probability_of_open_paths (n : ℕ) : ℝ :=
  2^(n-1) / (Nat.choose (2*(n-1)) (n-1))

theorem probability_of_reaching_last_floor (n : ℕ) :
  probability_of_open_paths n = 2^(n-1) / (Nat.choose (2*(n-1)) (n-1)) :=
by
  sorry

end probability_of_reaching_last_floor_l241_241875


namespace alice_bob_meeting_point_l241_241750

def meet_same_point (turns : ℕ) : Prop :=
  ∃ n : ℕ, turns = 2 * n ∧ 18 ∣ (7 * n - (7 * n + n))

theorem alice_bob_meeting_point :
  meet_same_point 36 :=
by
  sorry

end alice_bob_meeting_point_l241_241750


namespace max_value_of_sum_on_ellipse_l241_241797

theorem max_value_of_sum_on_ellipse (x y : ℝ) (h : x^2 / 3 + y^2 = 1) : x + y ≤ 2 :=
sorry

end max_value_of_sum_on_ellipse_l241_241797


namespace milk_removal_replacement_l241_241092

theorem milk_removal_replacement (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 45) :
  (45 - x) * (45 - x) / 45 = 28.8 → x = 9 :=
by
  -- skipping the proof for now
  sorry

end milk_removal_replacement_l241_241092


namespace rotate_D_90_clockwise_l241_241556

structure Point (α : Type) :=
  (x : α)
  (y : α)

def rotate_90_clockwise (p : Point ℤ) : Point ℤ :=
  ⟨p.y, -p.x⟩

def D : Point ℤ := ⟨-3, 2⟩
def E : Point ℤ := ⟨0, 5⟩
def F : Point ℤ := ⟨0, 2⟩

theorem rotate_D_90_clockwise :
  rotate_90_clockwise D = Point.mk 2 (-3) :=
by
  sorry

end rotate_D_90_clockwise_l241_241556


namespace janice_time_left_l241_241956

def time_before_movie : ℕ := 2 * 60
def homework_time : ℕ := 30
def cleaning_time : ℕ := homework_time / 2
def walking_dog_time : ℕ := homework_time + 5
def taking_trash_time : ℕ := homework_time * 1 / 6

theorem janice_time_left : time_before_movie - (homework_time + cleaning_time + walking_dog_time + taking_trash_time) = 35 :=
by
  sorry

end janice_time_left_l241_241956


namespace find_number_l241_241486

theorem find_number (N : ℝ) (h : (0.47 * N - 0.36 * 1412) + 66 = 6) : N = 953.87 :=
  sorry

end find_number_l241_241486


namespace find_a_l241_241349

noncomputable def A (a : ℝ) : Set ℝ := {x | x^2 < a^2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}
def C : Set ℝ := {x | 1 < x ∧ x < 2}

theorem find_a (a : ℝ) (h : A a ∩ B = C) : a = 2 ∨ a = -2 := by
  sorry

end find_a_l241_241349


namespace cos_sum_is_one_or_cos_2a_l241_241638

open Real

theorem cos_sum_is_one_or_cos_2a (a b : ℝ) (h : ∫ x in a..b, sin x = 0) : cos (a + b) = 1 ∨ cos (a + b) = cos (2 * a) :=
  sorry

end cos_sum_is_one_or_cos_2a_l241_241638


namespace batsman_total_score_l241_241080

-- We establish our variables and conditions first
variables (T : ℕ) -- total score
variables (boundaries : ℕ := 3) -- number of boundaries
variables (sixes : ℕ := 8) -- number of sixes
variables (boundary_runs_per : ℕ := 4) -- runs per boundary
variables (six_runs_per : ℕ := 6) -- runs per six
variables (running_percentage : ℕ := 50) -- percentage of runs made by running

-- Define the amounts of runs from boundaries and sixes
def runs_from_boundaries := boundaries * boundary_runs_per
def runs_from_sixes := sixes * six_runs_per

-- Main theorem to prove
theorem batsman_total_score :
  T = runs_from_boundaries + runs_from_sixes + T / 2 → T = 120 :=
by
  sorry

end batsman_total_score_l241_241080


namespace max_value_of_linear_combination_of_m_n_k_l241_241140

-- The style grants us maximum flexibility for definitions.
theorem max_value_of_linear_combination_of_m_n_k 
  (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℕ) (m n k : ℕ)
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ m → a i % 3 = 1)
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n → b i % 3 = 2)
  (h3 : ∀ i, 1 ≤ i ∧ i ≤ k → c i % 3 = 0)
  (h4 : Function.Injective a)
  (h5 : Function.Injective b)
  (h6 : Function.Injective c)
  (h_distinct : ∀ i j, i ≠ j → a i ≠ b j ∧ a i ≠ c j ∧ b i ≠ c j)
  (h_sum : (Finset.range m).sum a + (Finset.range n).sum b + (Finset.range k).sum c = 2007)
  : 4 * m + 3 * n + 5 * k ≤ 256 := by
  sorry

end max_value_of_linear_combination_of_m_n_k_l241_241140


namespace trumpet_cost_l241_241969

variable (total_amount : ℝ) (book_cost : ℝ)

theorem trumpet_cost (h1 : total_amount = 151) (h2 : book_cost = 5.84) :
  (total_amount - book_cost = 145.16) :=
by
  sorry

end trumpet_cost_l241_241969


namespace potassium_salt_average_molar_mass_l241_241786

noncomputable def average_molar_mass (total_weight : ℕ) (num_moles : ℕ) : ℕ :=
  total_weight / num_moles

theorem potassium_salt_average_molar_mass :
  let total_weight := 672
  let num_moles := 4
  average_molar_mass total_weight num_moles = 168 := by
    sorry

end potassium_salt_average_molar_mass_l241_241786


namespace only_negative_integer_among_list_l241_241893

namespace NegativeIntegerProblem

def is_integer (n : ℚ) : Prop := ∃ m : ℤ, n = m

theorem only_negative_integer_among_list :
  (∃ x, x ∈ [0, -1, 2, -1.5] ∧ (x < 0) ∧ is_integer x) ↔ (x = -1) :=
by
  sorry

end NegativeIntegerProblem

end only_negative_integer_among_list_l241_241893


namespace prove_ab_l241_241647

theorem prove_ab 
  (a b : ℝ)
  (h1 : a + b = 4)
  (h2 : a^2 + b^2 = 6) : 
  a * b = 5 :=
by
  sorry

end prove_ab_l241_241647


namespace Katie_cupcakes_l241_241128

theorem Katie_cupcakes (initial_cupcakes sold_cupcakes final_cupcakes : ℕ) (h1 : initial_cupcakes = 26) (h2 : sold_cupcakes = 20) (h3 : final_cupcakes = 26) :
  (final_cupcakes - (initial_cupcakes - sold_cupcakes)) = 20 :=
by
  sorry

end Katie_cupcakes_l241_241128


namespace max_t_eq_one_l241_241378

theorem max_t_eq_one {x y : ℝ} (hx : x > 0) (hy : y > 0) : 
  max (min x (y / (x^2 + y^2))) 1 = 1 :=
sorry

end max_t_eq_one_l241_241378


namespace abs_x_plus_2_l241_241945

theorem abs_x_plus_2 (x : ℤ) (h : x = -3) : |x + 2| = 1 :=
by sorry

end abs_x_plus_2_l241_241945


namespace fencing_cost_l241_241051

def total_cost_of_fencing 
  (length breadth cost_per_meter : ℝ)
  (h1 : length = 62)
  (h2 : length = breadth + 24)
  (h3 : cost_per_meter = 26.50) : ℝ :=
  2 * (length + breadth) * cost_per_meter

theorem fencing_cost : total_cost_of_fencing 62 38 26.50 (by rfl) (by norm_num) (by norm_num) = 5300 := 
by 
  sorry

end fencing_cost_l241_241051


namespace number_of_alligators_l241_241016

theorem number_of_alligators (A : ℕ) 
  (num_snakes : ℕ := 18) 
  (total_eyes : ℕ := 56) 
  (eyes_per_snake : ℕ := 2) 
  (eyes_per_alligator : ℕ := 2) 
  (snakes_eyes : ℕ := num_snakes * eyes_per_snake) 
  (alligators_eyes : ℕ := A * eyes_per_alligator) 
  (total_animals_eyes : ℕ := snakes_eyes + alligators_eyes) 
  (total_eyes_eq : total_animals_eyes = total_eyes) 
: A = 10 :=
by 
  sorry

end number_of_alligators_l241_241016


namespace min_x8_x9_x10_eq_618_l241_241535

theorem min_x8_x9_x10_eq_618 (x : ℕ → ℕ) (h1 : ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 10 → x i < x j)
  (h2 : x 1 + x 2 + x 3 + x 4 + x 5 + x 6 + x 7 + x 8 + x 9 + x 10 = 2023) :
  x 8 + x 9 + x 10 = 618 :=
sorry

end min_x8_x9_x10_eq_618_l241_241535


namespace wrongly_written_height_is_176_l241_241046

-- Definitions and given conditions
def average_height_incorrect := 182
def average_height_correct := 180
def num_boys := 35
def actual_height := 106

-- The difference in total height due to the error
def total_height_incorrect := num_boys * average_height_incorrect
def total_height_correct := num_boys * average_height_correct
def height_difference := total_height_incorrect - total_height_correct

-- The wrongly written height
def wrongly_written_height := actual_height + height_difference

-- Proof statement
theorem wrongly_written_height_is_176 : wrongly_written_height = 176 := by
  sorry

end wrongly_written_height_is_176_l241_241046


namespace sqrt_36_eq_6_cube_root_neg_a_125_l241_241561

theorem sqrt_36_eq_6 : ∀ (x : ℝ), 0 ≤ x ∧ x^2 = 36 → x = 6 :=
by sorry

theorem cube_root_neg_a_125 : ∀ (a y : ℝ), y^3 = - a / 125 → y = - (a^(1/3)) / 5 :=
by sorry

end sqrt_36_eq_6_cube_root_neg_a_125_l241_241561


namespace solution_to_water_l241_241525

theorem solution_to_water (A W S T: ℝ) (h1: A = 0.04) (h2: W = 0.02) (h3: S = 0.06) (h4: T = 0.48) :
  (T * (W / S) = 0.16) :=
by
  sorry

end solution_to_water_l241_241525


namespace binomial_12_10_l241_241450

def binomial (n k : ℕ) : ℕ := n.choose k

theorem binomial_12_10 : binomial 12 10 = 66 := by
  -- The proof will go here
  sorry

end binomial_12_10_l241_241450


namespace rect_garden_width_l241_241431

theorem rect_garden_width (w l : ℝ) (h1 : l = 3 * w) (h2 : l * w = 768) : w = 16 := by
  sorry

end rect_garden_width_l241_241431


namespace solution_1_solution_2_l241_241004

noncomputable def f (x a : ℝ) : ℝ := |x - a| - |x - 3|

theorem solution_1 (x : ℝ) : (f x (-1) >= 2) ↔ (x >= 2) :=
by
  sorry

theorem solution_2 (a : ℝ) : 
  (∃ x : ℝ, f x a <= -(a / 2)) ↔ (a <= 2 ∨ a >= 6) :=
by
  sorry

end solution_1_solution_2_l241_241004


namespace smallest_five_digit_multiple_of_18_l241_241340

theorem smallest_five_digit_multiple_of_18 : ∃ n : ℕ, n = 10008 ∧ (n ≥ 10000 ∧ n < 100000) ∧ n % 18 = 0 ∧ (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) ∧ m % 18 = 0 → n ≤ m) := sorry

end smallest_five_digit_multiple_of_18_l241_241340


namespace find_x_l241_241493

def f (x : ℝ) : ℝ := 3 * x - 5

theorem find_x (x : ℝ) (h : 2 * f x - 19 = f (x - 4)) : x = 4 := 
by 
  sorry

end find_x_l241_241493


namespace range_of_m_l241_241943

theorem range_of_m (m : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + m * x + 2 * m - 3 < 0) ↔ 2 ≤ m ∧ m ≤ 6 := 
by
  sorry

end range_of_m_l241_241943


namespace point_P_quadrant_IV_l241_241591

theorem point_P_quadrant_IV (x y : ℝ) (h1 : x > 0) (h2 : y < 0) : x > 0 ∧ y < 0 :=
by
  sorry

end point_P_quadrant_IV_l241_241591


namespace value_of_a_if_lines_are_parallel_l241_241699

theorem value_of_a_if_lines_are_parallel (a : ℝ) :
  (∀ (x y : ℝ), x + a*y - 7 = 0 → (a+1)*x + 2*y - 14 = 0) → a = -2 :=
sorry

end value_of_a_if_lines_are_parallel_l241_241699


namespace reciprocal_of_neg_five_l241_241218

theorem reciprocal_of_neg_five: 
  ∃ x : ℚ, -5 * x = 1 ∧ x = -1 / 5 := 
sorry

end reciprocal_of_neg_five_l241_241218


namespace ab_value_l241_241631

theorem ab_value (a b : ℝ) (h1 : |a| = 3) (h2 : |b - 2| = 9) (h3 : a + b > 0) :
  ab = 33 ∨ ab = -33 :=
by
  sorry

end ab_value_l241_241631


namespace Euclid_Middle_School_AMC8_contest_l241_241604

theorem Euclid_Middle_School_AMC8_contest (students_Germain students_Newton students_Young : ℕ)
       (hG : students_Germain = 11) 
       (hN : students_Newton = 8) 
       (hY : students_Young = 9) : 
       students_Germain + students_Newton + students_Young = 28 :=
by
  sorry

end Euclid_Middle_School_AMC8_contest_l241_241604


namespace line_intersects_circle_l241_241568

theorem line_intersects_circle (a : ℝ) :
  ∃ (x y : ℝ), (y = a * x + 1) ∧ ((x - 1) ^ 2 + y ^ 2 = 4) :=
by
  sorry

end line_intersects_circle_l241_241568


namespace sum_of_cubes_l241_241186

variable {R : Type} [OrderedRing R] [Field R] [DecidableEq R]

theorem sum_of_cubes (a b c : R) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : c ≠ a)
    (h₄ : (a^3 + 12) / a = (b^3 + 12) / b) (h₅ : (b^3 + 12) / b = (c^3 + 12) / c) :
    a^3 + b^3 + c^3 = -36 := by
  sorry

end sum_of_cubes_l241_241186


namespace remainder_when_dividing_l241_241380

theorem remainder_when_dividing (c d : ℕ) (p q : ℕ) :
  c = 60 * p + 47 ∧ d = 45 * q + 14 → (c + d) % 15 = 1 :=
by
  sorry

end remainder_when_dividing_l241_241380


namespace intersection_complement_eq_C_l241_241930

def A := { x : ℝ | -3 < x ∧ x < 6 }
def B := { x : ℝ | 2 < x ∧ x < 7 }
def complement_B := { x : ℝ | x ≤ 2 ∨ x ≥ 7 }
def C := { x : ℝ | -3 < x ∧ x ≤ 2 }

theorem intersection_complement_eq_C :
  A ∩ complement_B = C :=
sorry

end intersection_complement_eq_C_l241_241930


namespace cos_alpha_value_l241_241629

noncomputable def cos_alpha (α : ℝ) : ℝ :=
  (3 - 4 * Real.sqrt 3) / 10

theorem cos_alpha_value (α : ℝ) (h1 : Real.sin (Real.pi / 6 + α) = 3 / 5) (h2 : Real.pi / 3 < α ∧ α < 5 * Real.pi / 6) :
  Real.cos α = cos_alpha α :=
by
sorry

end cos_alpha_value_l241_241629


namespace q_investment_time_l241_241073

-- Definitions from the conditions
def investment_ratio_p_q : ℚ := 7 / 5
def profit_ratio_p_q : ℚ := 7 / 13
def time_p : ℕ := 5

-- Problem statement
theorem q_investment_time
  (investment_ratio_p_q : ℚ)
  (profit_ratio_p_q : ℚ)
  (time_p : ℕ)
  (hpq_inv : investment_ratio_p_q = 7 / 5)
  (hpq_profit : profit_ratio_p_q = 7 / 13)
  (ht_p : time_p = 5) : 
  ∃ t_q : ℕ, 35 * t_q = 455 :=
sorry

end q_investment_time_l241_241073


namespace smallest_x_integer_value_l241_241853

theorem smallest_x_integer_value (x : ℤ) (h : (x - 5) ∣ 58) : x = -53 :=
by
  sorry

end smallest_x_integer_value_l241_241853


namespace fill_675_cans_time_l241_241314

theorem fill_675_cans_time :
  (∀ (cans_per_batch : ℕ) (time_per_batch : ℕ) (total_cans : ℕ),
    cans_per_batch = 150 →
    time_per_batch = 8 →
    total_cans = 675 →
    total_cans / cans_per_batch * time_per_batch = 36) :=
begin
  intros cans_per_batch time_per_batch total_cans h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
  sorry
end

end fill_675_cans_time_l241_241314


namespace eval_expr_at_sqrt3_minus_3_l241_241202

noncomputable def expr (a : ℝ) : ℝ :=
  (3 - a) / (2 * a - 4) / (a + 2 - 5 / (a - 2))

theorem eval_expr_at_sqrt3_minus_3 : expr (Real.sqrt 3 - 3) = -Real.sqrt 3 / 6 := 
  by sorry

end eval_expr_at_sqrt3_minus_3_l241_241202


namespace notebook_cost_l241_241745

theorem notebook_cost
  (n c : ℝ)
  (h1 : n + c = 2.20)
  (h2 : n = c + 2) :
  n = 2.10 :=
by
  sorry

end notebook_cost_l241_241745


namespace sin_2alpha_pos_of_tan_alpha_pos_l241_241646

theorem sin_2alpha_pos_of_tan_alpha_pos (α : Real) (h : Real.tan α > 0) : Real.sin (2 * α) > 0 :=
sorry

end sin_2alpha_pos_of_tan_alpha_pos_l241_241646


namespace solve_inequality_l241_241041

open Set

-- Define a predicate for the inequality solution sets
def inequality_solution_set (k : ℝ) : Set ℝ :=
  if h : k = 0 then {x | x < 1}
  else if h : 0 < k ∧ k < 2 then {x | x < 1 ∨ x > 2 / k}
  else if h : k = 2 then {x | True} \ {1}
  else if h : k > 2 then {x | x < 2 / k ∨ x > 1}
  else {x | 2 / k < x ∧ x < 1}

-- The statement of the proof
theorem solve_inequality (k : ℝ) :
  ∀ x : ℝ, k * x^2 - (k + 2) * x + 2 < 0 ↔ x ∈ inequality_solution_set k :=
by
  sorry

end solve_inequality_l241_241041


namespace best_fitting_regression_line_l241_241095

theorem best_fitting_regression_line
  (R2_A : ℝ) (R2_B : ℝ) (R2_C : ℝ) (R2_D : ℝ)
  (h_A : R2_A = 0.27)
  (h_B : R2_B = 0.85)
  (h_C : R2_C = 0.96)
  (h_D : R2_D = 0.5) :
  R2_C = 0.96 :=
by
  -- Proof goes here
  sorry

end best_fitting_regression_line_l241_241095


namespace problem_statement_l241_241925

namespace GeometricRelations

variables {Line Plane : Type} [Nonempty Line] [Nonempty Plane]

-- Define parallel and perpendicular relations
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry

-- Given conditions
variables (m n : Line) (α β : Plane)

-- The theorem to be proven
theorem problem_statement 
  (h1 : perpendicular m β) 
  (h2 : parallel α β) : 
  perpendicular m α :=
sorry

end GeometricRelations

end problem_statement_l241_241925


namespace jerry_remaining_debt_l241_241668

theorem jerry_remaining_debt :
  ∀ (paid_two_months_ago paid_last_month total_debt: ℕ),
  paid_two_months_ago = 12 →
  paid_last_month = paid_two_months_ago + 3 →
  total_debt = 50 →
  total_debt - (paid_two_months_ago + paid_last_month) = 23 :=
by
  intros paid_two_months_ago paid_last_month total_debt h1 h2 h3
  sorry

end jerry_remaining_debt_l241_241668


namespace no_such_integers_exists_l241_241532

theorem no_such_integers_exists :
  ∀ (P : ℕ → ℕ), (∀ x, P x = x ^ 2000 - x ^ 1000 + 1) →
  ¬(∃ (a : Fin 8002 → ℕ), Function.Injective a ∧
  (∀ i j k : Fin 8002, i ≠ j → j ≠ k → i ≠ k → a i * a j * a k ∣ P (a i) * P (a j) * P (a k))) := 
by
  intro P hP notExists
  have contra : ∃ (a : Fin 8002 → ℕ), Function.Injective a ∧
    (∀ i j k : Fin 8002, i ≠ j → j ≠ k → i ≠ k → a i * a j * a k ∣ P (a i) * P (a j) * P (a k)) := notExists
  sorry

end no_such_integers_exists_l241_241532


namespace salt_percentage_in_first_solution_l241_241748

variable (S : ℚ)
variable (H : 0 ≤ S ∧ S ≤ 100)  -- percentage constraints

theorem salt_percentage_in_first_solution (h : 0.75 * S / 100 + 7 = 16) : S = 12 :=
by { sorry }

end salt_percentage_in_first_solution_l241_241748


namespace distance_from_pole_to_line_l241_241368

/-- Definition of the line in polar coordinates -/
def line_polar (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2

/-- Definition of the pole in Cartesian coordinates -/
def pole_cartesian : ℝ × ℝ := (0, 0)

/-- Convert the line from polar to Cartesian -/
def line_cartesian (x y : ℝ) : Prop := x = 2

/-- The distance function between a point and a line in Cartesian coordinates -/
def distance_to_line (p : ℝ × ℝ) : ℝ := abs (p.1 - 2)

/-- Prove that the distance from the pole to the line is 2 -/
theorem distance_from_pole_to_line : distance_to_line pole_cartesian = 2 := by
  sorry

end distance_from_pole_to_line_l241_241368


namespace problem_statement_l241_241633

-- Definition of the arithmetic sequence {a_n}
def a (n : ℕ) : ℕ := n

-- Definition of the geometric sequence {b_n}
def b (n : ℕ) : ℕ := 2^n

-- Definition of the sequence {c_n}
def c (n : ℕ) : ℕ := a n + b n

-- Sum of first n terms of the sequence {c_n}
def S (n : ℕ) : ℕ := (n * (n + 1)) / 2 + 2^(n + 1) - 2

-- Prove the problem statement
theorem problem_statement :
  (a 1 + a 2 = 3) ∧
  (a 4 - a 3 = 1) ∧
  (b 2 = a 4) ∧
  (b 3 = a 8) ∧
  (∀ n : ℕ, c n = a n + b n) ∧
  (∀ n : ℕ, S n = (n * (n + 1)) / 2 + 2^(n + 1) - 2) :=
by {
  sorry -- Proof goes here
}

end problem_statement_l241_241633


namespace smallest_possible_a_l241_241206

theorem smallest_possible_a (a b c : ℝ) 
  (h1 : (∀ x, y = a * x ^ 2 + b * x + c ↔ y = a * (x + 1/3) ^ 2 + 5/9))
  (h2 : a > 0)
  (h3 : ∃ n : ℤ, a + b + c = n) : 
  a = 1/4 :=
sorry

end smallest_possible_a_l241_241206


namespace magic_trick_constant_l241_241058

theorem magic_trick_constant (a : ℚ) : ((2 * a + 8) / 4 - a / 2) = 2 :=
by
  sorry

end magic_trick_constant_l241_241058


namespace exists_large_subset_free_of_arithmetic_progressions_l241_241416

open Finset

noncomputable def isFreeOfArithmeticProgressions (A : Finset ℕ) : Prop :=
  ∀ ⦃a b c : ℕ⦄, a ∈ A → b ∈ A → c ∈ A → a ≠ b → a ≠ c → b ≠ c → a + b ≠ 2 * c

theorem exists_large_subset_free_of_arithmetic_progressions :
  ∃ (A : Finset ℕ), A ⊆ range (3^8) ∧ A.card ≥ 256 ∧ isFreeOfArithmeticProgressions A :=
begin
  sorry
end

end exists_large_subset_free_of_arithmetic_progressions_l241_241416


namespace num_books_second_shop_l241_241821

-- Define the conditions
def num_books_first_shop : ℕ := 32
def cost_first_shop : ℕ := 1500
def cost_second_shop : ℕ := 340
def avg_price_per_book : ℕ := 20

-- Define the proof statement
theorem num_books_second_shop : 
  (num_books_first_shop + (cost_second_shop + cost_first_shop) / avg_price_per_book) - num_books_first_shop = 60 := by
  sorry

end num_books_second_shop_l241_241821


namespace find_value_expression_l241_241676

theorem find_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x^2 + x * y + y^2 = 75)
  (h2 : y^2 + y * z + z^2 = 4)
  (h3 : z^2 + x * z + x^2 = 79) :
  x * y + y * z + x * z = 20 := 
sorry

end find_value_expression_l241_241676


namespace original_price_l241_241317

variable (a : ℝ)

-- Given the price after a 20% discount is a yuan per unit,
-- Prove that the original price per unit was (5/4) * a yuan.
theorem original_price (h : a > 0) : (a / (4 / 5)) = (5 / 4) * a :=
by sorry

end original_price_l241_241317


namespace carries_average_speed_is_approx_34_29_l241_241804

noncomputable def CarriesActualAverageSpeed : ℝ :=
  let jerry_speed := 40 -- in mph
  let jerry_time := 1/2 -- in hours, 30 minutes = 0.5 hours
  let jerry_distance := jerry_speed * jerry_time

  let beth_distance := jerry_distance + 5
  let beth_time := jerry_time + (20 / 60) -- converting 20 minutes to hours

  let carrie_distance := 2 * jerry_distance
  let carrie_time := 1 + (10 / 60) -- converting 10 minutes to hours

  carrie_distance / carrie_time

theorem carries_average_speed_is_approx_34_29 : 
  |CarriesActualAverageSpeed - 34.29| < 0.01 :=
sorry

end carries_average_speed_is_approx_34_29_l241_241804


namespace village_male_population_l241_241088

theorem village_male_population (total_population parts male_parts : ℕ) (h1 : total_population = 600) (h2 : parts = 4) (h3 : male_parts = 2) :
  male_parts * (total_population / parts) = 300 :=
by
  -- We are stating the problem as per the given conditions
  sorry

end village_male_population_l241_241088


namespace proof_A_intersection_C_U_B_l241_241499

open Set

-- Given sets U, A, and B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {4, 5}

-- Complement of B with respect to U
def C_U_B : Set ℕ := U \ B

-- Prove that the intersection of A and C_U_B is {2, 3}
theorem proof_A_intersection_C_U_B :
  A ∩ C_U_B = {2, 3} := by
  sorry

end proof_A_intersection_C_U_B_l241_241499


namespace binomial_12_10_eq_66_l241_241454

theorem binomial_12_10_eq_66 : binomial 12 10 = 66 :=
by
  sorry

end binomial_12_10_eq_66_l241_241454


namespace domain_of_c_is_all_reals_l241_241476

theorem domain_of_c_is_all_reals (k : ℝ) :
  (∀ x : ℝ, -3 * x^2 - 4 * x + k ≠ 0) ↔ k < -4 / 3 := 
by
  sorry

end domain_of_c_is_all_reals_l241_241476


namespace microorganism_half_filled_time_l241_241977

theorem microorganism_half_filled_time :
  (∀ x, 2^x = 2^9 ↔ x = 9) :=
by
  sorry

end microorganism_half_filled_time_l241_241977


namespace option_c_equals_9_l241_241423

theorem option_c_equals_9 : (3 * 3 - 3 + 3) = 9 :=
by
  sorry

end option_c_equals_9_l241_241423


namespace calculate_seasons_l241_241859

theorem calculate_seasons :
  ∀ (episodes_per_season : ℕ) (episodes_per_day : ℕ) (days : ℕ),
  episodes_per_season = 20 →
  episodes_per_day = 2 →
  days = 30 →
  (episodes_per_day * days) / episodes_per_season = 3 :=
by
  intros episodes_per_season episodes_per_day days h_eps h_epd h_d
  sorry

end calculate_seasons_l241_241859


namespace totalUniqueStudents_l241_241894

-- Define the club memberships and overlap
variable (mathClub scienceClub artClub overlap : ℕ)

-- Conditions based on the problem
def mathClubSize : Prop := mathClub = 15
def scienceClubSize : Prop := scienceClub = 10
def artClubSize : Prop := artClub = 12
def overlapSize : Prop := overlap = 5

-- Main statement to prove
theorem totalUniqueStudents : 
  mathClubSize mathClub → 
  scienceClubSize scienceClub →
  artClubSize artClub →
  overlapSize overlap →
  mathClub + scienceClub + artClub - overlap = 32 := by
  intros
  sorry

end totalUniqueStudents_l241_241894


namespace abs_eq_self_nonneg_l241_241212

theorem abs_eq_self_nonneg (x : ℝ) : abs x = x ↔ x ≥ 0 :=
sorry

end abs_eq_self_nonneg_l241_241212


namespace eval_expression_l241_241223

theorem eval_expression : (20 - 16) * (12 + 8) / 4 = 20 := 
by 
  sorry

end eval_expression_l241_241223


namespace correct_answer_l241_241131

noncomputable def sqrt_2 : ℝ := Real.sqrt 2

def P : Set ℝ := { x | x^2 - 2*x - 3 ≤ 0 }

theorem correct_answer : {sqrt_2} ⊆ P :=
sorry

end correct_answer_l241_241131


namespace george_collected_50_marbles_l241_241490

theorem george_collected_50_marbles (w y g r total : ℕ)
  (hw : w = total / 2)
  (hy : y = 12)
  (hg : g = y / 2)
  (hr : r = 7)
  (htotal : total = w + y + g + r) :
  total = 50 := by
  sorry

end george_collected_50_marbles_l241_241490


namespace solve_cos_sin_eq_one_l241_241824

theorem solve_cos_sin_eq_one (n : ℕ) (k : ℤ) :
  ∃ x : ℝ, cos (n * x) - sin (n * x) = 1 ∧ x = 2 * (↑k) * Real.pi / n := sorry

end solve_cos_sin_eq_one_l241_241824


namespace largest_b_for_box_volume_l241_241990

theorem largest_b_for_box_volume (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) 
                                 (h4 : c = 3) (volume : a * b * c = 360) : 
    b = 8 := 
sorry

end largest_b_for_box_volume_l241_241990


namespace maximum_sum_of_factors_exists_maximum_sum_of_factors_l241_241179

theorem maximum_sum_of_factors {A B C : ℕ} (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C)
  (h4 : A * B * C = 2023) : A + B + C ≤ 297 :=
sorry

theorem exists_maximum_sum_of_factors : ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A * B * C = 2023 ∧ A + B + C = 297 :=
sorry

end maximum_sum_of_factors_exists_maximum_sum_of_factors_l241_241179


namespace smallest_value_of_y1_y2_y3_sum_l241_241190

noncomputable def y_problem := 
  ∃ (y1 y2 y3 : ℝ), 
  (y1 + 3 * y2 + 5 * y3 = 120) 
  ∧ (y1^2 + y2^2 + y3^2 = 720 / 7)

theorem smallest_value_of_y1_y2_y3_sum :
  (∃ (y1 y2 y3 : ℝ), 0 < y1 ∧ 0 < y2 ∧ 0 < y3 ∧ (y1 + 3 * y2 + 5 * y3 = 120) 
  ∧ (y1^2 + y2^2 + y3^2 = 720 / 7)) :=
by 
  sorry

end smallest_value_of_y1_y2_y3_sum_l241_241190


namespace binomial_coefficient_12_10_l241_241471

theorem binomial_coefficient_12_10 : Nat.choose 12 10 = 66 := by
  sorry  -- The actual proof is omitted as instructed.

end binomial_coefficient_12_10_l241_241471


namespace complement_of_A_l241_241628

theorem complement_of_A (U : Set ℕ) (A : Set ℕ) (C_UA : Set ℕ) :
  U = {2, 3, 4} →
  A = {x | (x - 1) * (x - 4) < 0 ∧ x ∈ Set.univ} →
  C_UA = {x ∈ U | x ∉ A} →
  C_UA = {4} :=
by
  intros hU hA hCUA
  -- proof omitted, sorry placeholder
  sorry

end complement_of_A_l241_241628


namespace greatest_sum_solution_l241_241707

theorem greatest_sum_solution (x y : ℤ) (h : x^2 + y^2 = 20) : 
  x + y ≤ 6 :=
sorry

end greatest_sum_solution_l241_241707


namespace sum_of_all_possible_values_of_M_l241_241213

-- Given conditions
-- M * (M - 8) = -7
-- We need to prove that the sum of all possible values of M is 8

theorem sum_of_all_possible_values_of_M : 
  ∃ M1 M2 : ℝ, (M1 * (M1 - 8) = -7) ∧ (M2 * (M2 - 8) = -7) ∧ (M1 + M2 = 8) :=
by
  sorry

end sum_of_all_possible_values_of_M_l241_241213


namespace probability_factor_36_l241_241241

theorem probability_factor_36 : 
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 36 ∧ (∀ k : ℕ, k ∣ n ↔ k ∣ 36) → (n ≠ 0 → (∃ p : ℚ, p = 1 / 4))) :=
begin
  sorry
end

end probability_factor_36_l241_241241


namespace probability_factor_of_36_l241_241243

def is_factor (d n : ℕ) : Prop := n % d = 0

theorem probability_factor_of_36 : 
  (∑ i in finset.range (36+1), if is_factor i 36 then (1:ℚ) else 0) / 36 = 1 / 4 := by
sorry

end probability_factor_of_36_l241_241243


namespace grace_putting_down_mulch_hours_l241_241155

/-- Grace's earnings conditions and hours calculation in September. -/
theorem grace_putting_down_mulch_hours :
  ∃ h : ℕ, 
    6 * 63 + 11 * 9 + 9 * h = 567 ∧
    h = 10 :=
by
  sorry

end grace_putting_down_mulch_hours_l241_241155


namespace inverse_function_of_f_l241_241787

noncomputable def f (x : ℝ) : ℝ := (x - 1) ^ 2

noncomputable def f_inv (y : ℝ) : ℝ := 1 - Real.sqrt y

theorem inverse_function_of_f :
  ∀ x, x ≤ 1 → f_inv (f x) = x ∧ ∀ y, 0 ≤ y → f (f_inv y) = y :=
by
  intros
  sorry

end inverse_function_of_f_l241_241787


namespace rectangle_area_l241_241333

theorem rectangle_area (P : ℕ) (a : ℕ) (b : ℕ) (h₁ : P = 2 * (a + b)) (h₂ : P = 40) (h₃ : a = 5) : a * b = 75 :=
by
  sorry

end rectangle_area_l241_241333


namespace shaded_area_correct_l241_241987

-- Triangle and circle setup
def equilateral_triangle := (side_length : ℝ) (side_length = 12)
def circle (radius : ℝ) := (diameter = side_length) (radius = diameter / 2)

-- Calculations for the shaded regions
def angle_AEB := 60
def angle_AOC := 60
def area_sector (angle : ℝ) (radius : ℝ) := (angle / 360) * (Real.pi * radius ^ 2)
def area_triangle (side_length : ℝ) := (side_length ^ 2 * Real.sqrt 3) / 4
def shaded_region_area := λ radius, area_sector 60 radius - area_triangle radius
def total_shaded_area (radius : ℝ) := 2 * shaded_region_area radius

-- Verifying the final result
theorem shaded_area_correct :
  let radius := 6 in
  let a := 12 in
  let b := 18 in
  let c := 3 in
  total_shaded_area radius = a * Real.pi - b * Real.sqrt c ∧ a + b + c = 33 :=
by
  sorry

end shaded_area_correct_l241_241987


namespace find_largest_x_l241_241106

theorem find_largest_x : 
  ∃ x : ℝ, (4 * x ^ 3 - 17 * x ^ 2 + x + 10 = 0) ∧ 
           (∀ y : ℝ, 4 * y ^ 3 - 17 * y ^ 2 + y + 10 = 0 → y ≤ x) ∧ 
           x = (25 + Real.sqrt 545) / 8 :=
sorry

end find_largest_x_l241_241106


namespace initial_amount_is_1875_l241_241616

-- Defining the conditions as given in the problem
def initial_amount : ℝ := sorry
def spent_on_clothes : ℝ := 250
def spent_on_food (remaining : ℝ) : ℝ := 0.35 * remaining
def spent_on_electronics (remaining : ℝ) : ℝ := 0.50 * remaining

-- Given conditions
axiom condition1 : initial_amount - spent_on_clothes = sorry
axiom condition2 : initial_amount - spent_on_clothes - spent_on_food (initial_amount - spent_on_clothes) = sorry
axiom condition3 : initial_amount - spent_on_clothes - spent_on_food (initial_amount - spent_on_clothes) - spent_on_electronics (initial_amount - spent_on_clothes - spent_on_food (initial_amount - spent_on_clothes)) = 200

-- Prove that initial amount is $1875
theorem initial_amount_is_1875 : initial_amount = 1875 :=
sorry

end initial_amount_is_1875_l241_241616


namespace sandy_correct_sums_l241_241394

variables (x y : ℕ)

theorem sandy_correct_sums :
  (x + y = 30) →
  (3 * x - 2 * y = 50) →
  x = 22 :=
by
  intro h1 h2
  -- Proof will be filled in here
  sorry

end sandy_correct_sums_l241_241394


namespace total_animals_l241_241059

theorem total_animals (initial_elephants initial_hippos : ℕ) 
  (ratio_female_hippos : ℚ)
  (births_per_female_hippo : ℕ)
  (newborn_elephants_diff : ℕ)
  (he : initial_elephants = 20)
  (hh : initial_hippos = 35)
  (rfh : ratio_female_hippos = 5 / 7)
  (bpfh : births_per_female_hippo = 5)
  (ned : newborn_elephants_diff = 10) :
  ∃ (total_animals : ℕ), total_animals = 315 :=
by sorry

end total_animals_l241_241059


namespace question1_question2_l241_241147

noncomputable def f (x b c : ℝ) := x^2 + b * x + c

theorem question1 (b c : ℝ) (h : ∀ x : ℝ, 2 * x + b ≤ f x b c) (x : ℝ) (hx : 0 ≤ x) :
  f x b c ≤ (x + c)^2 :=
sorry

theorem question2 (b c m : ℝ) (h : ∀ b c : ℝ, b ≠ c → f c b b - f b b b ≤ m * (c^2 - b^2)) :
  m ≥ 3/2 :=
sorry

end question1_question2_l241_241147


namespace remainder_of_3045_div_32_l241_241717

theorem remainder_of_3045_div_32 : 3045 % 32 = 5 :=
by sorry

end remainder_of_3045_div_32_l241_241717


namespace fractional_eq_no_real_roots_l241_241652

theorem fractional_eq_no_real_roots (k : ℝ) :
  (∀ x : ℝ, (x - 1) ≠ 0 → (k / (x - 1) + 3 ≠ x / (1 - x))) → k = -1 :=
by
  sorry

end fractional_eq_no_real_roots_l241_241652


namespace remainder_sum_div7_l241_241770

theorem remainder_sum_div7 (a b c : ℕ) (h1 : a * b * c ≡ 2 [MOD 7])
  (h2 : 3 * c ≡ 4 [MOD 7])
  (h3 : 4 * b ≡ 2 + b [MOD 7]) :
  (a + b + c) % 7 = 6 := by
  sorry

end remainder_sum_div7_l241_241770


namespace probability_factor_36_l241_241281

def is_factor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

theorem probability_factor_36 : 
  (set.univ.filter (λ n : ℕ, n > 0 ∧ n ≤ 36 ∧ is_factor 36 n)).card.toRat / 36 = 1 / 4 := 
by
  sorry

end probability_factor_36_l241_241281


namespace min_value_fraction_l241_241835

theorem min_value_fraction (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (2 * a + b) = 2) : 
  ∃ x : ℝ, x = (8 * a + b) / (a * b) ∧ x = 9 :=
by
  sorry

end min_value_fraction_l241_241835


namespace sum_of_dimensions_l241_241090

theorem sum_of_dimensions
  (X Y Z : ℝ)
  (h1 : X * Y = 24)
  (h2 : X * Z = 48)
  (h3 : Y * Z = 72) :
  X + Y + Z = 22 := 
sorry

end sum_of_dimensions_l241_241090


namespace emily_three_blue_marbles_probability_l241_241618

noncomputable def probability_exactly_three_blue_marbles : ℝ :=
  let blue_marble_prob := (8:ℕ) / (14:ℕ)
  let red_marble_prob := (6:ℕ) / (14:ℕ)
  let n := 6
  let k := 3
  let comb := (nat.choose n k)
  in comb * (blue_marble_prob ^ k) * (red_marble_prob ^ (n - k))

theorem emily_three_blue_marbles_probability :
  probability_exactly_three_blue_marbles = 34560 / 117649 := by
  sorry

end emily_three_blue_marbles_probability_l241_241618


namespace sin_C_and_area_of_triangle_l241_241370

open Real

noncomputable section

theorem sin_C_and_area_of_triangle 
  (A B C : ℝ)
  (cos_A : Real := sqrt 3 / 3)
  (a b c : ℝ := (3 * sqrt 2)) 
  (cosA : cos A = sqrt 3 / 3)
  -- angles in radians, use radians for the angles when proving
  (side_c : c = sqrt 3)
  (side_a : a = 3 * sqrt 2) :
  (sin C = 1 / 3) ∧ (1 / 2 * a * b * sin C = 5 * sqrt 6 / 3) :=
by
  sorry

end sin_C_and_area_of_triangle_l241_241370


namespace greatest_possible_value_x_l241_241578

theorem greatest_possible_value_x :
  ∀ x : ℚ, (∃ y : ℚ, y = (5 * x - 25) / (4 * x - 5) ∧ y^2 + y = 18) →
  x ≤ 55 / 29 :=
by sorry

end greatest_possible_value_x_l241_241578


namespace probability_open_doors_l241_241881

variable (n : ℕ)

theorem probability_open_doors (h : n > 1) : 
  let num_doors := 2 * (n - 1)
      num_locked := n - 1
  in (2^(n-1) / (num_doors.choose num_locked) = 
  2^(n-1) / (nat.choose num_doors num_locked)) := sorry

end probability_open_doors_l241_241881


namespace supermarket_A_is_more_cost_effective_l241_241682

def price_A (kg : ℕ) : ℕ :=
  if kg <= 4 then kg * 10
  else 4 * 10 + (kg - 4) * 6

def price_B (kg : ℕ) : ℕ :=
  kg * 10 * 8 / 10

theorem supermarket_A_is_more_cost_effective :
  price_A 3 = 30 ∧ 
  price_A 5 = 46 ∧ 
  ∀ (x : ℕ), (x > 4) → price_A x = 6 * x + 16 ∧ 
  price_A 10 < price_B 10 :=
by 
  sorry

end supermarket_A_is_more_cost_effective_l241_241682


namespace train_ticket_product_l241_241205

theorem train_ticket_product
  (a b c d e : ℕ)
  (h1 : b = a + 1)
  (h2 : c = a + 2)
  (h3 : d = a + 3)
  (h4 : e = a + 4)
  (h_sum : a + b + c + d + e = 120) :
  a * b * c * d * e = 7893600 :=
sorry

end train_ticket_product_l241_241205


namespace no_such_number_exists_l241_241401

-- Definitions for conditions
def base_5_digit_number (x : ℕ) : Prop := 
  ∀ n, 0 ≤ n ∧ n < 2023 → x / 5^n % 5 < 5

def odd_plus_one (n m : ℕ) : Prop :=
  (∀ k < 1012, (n / 5^(2*k) % 25 / 5 = m / 5^(2*k) % 25 / 5 + 1)) ∧
  (∀ k < 1011, (n / 5^(2*k+1) % 25 / 5 = m / 5^(2*k+1) % 25 / 5 - 1))

def has_two_prime_factors_that_differ_by_two (x : ℕ) : Prop :=
  ∃ u v, u * v = x ∧ Prime u ∧ Prime v ∧ v = u + 2

-- Combined conditions for the hypothesized number x
def hypothesized_number (x : ℕ) : Prop := 
  base_5_digit_number x ∧
  odd_plus_one x x ∧
  has_two_prime_factors_that_differ_by_two x

-- The proof statement that the hypothesized number cannot exist
theorem no_such_number_exists : ¬ ∃ x, hypothesized_number x :=
by
  sorry

end no_such_number_exists_l241_241401


namespace simplify_expression_l241_241753

variable (x y : ℝ)

theorem simplify_expression : (-(3 * x * y - 2 * x ^ 2) - 2 * (3 * x ^ 2 - x * y)) = (-4 * x ^ 2 - x * y) :=
by
  sorry

end simplify_expression_l241_241753


namespace remaining_sugar_l241_241102

/-- Chelsea has 24 kilos of sugar. She divides them into 4 bags equally.
  Then one of the bags gets torn and half of the sugar falls to the ground.
  How many kilos of sugar remain? --/
theorem remaining_sugar (total_sugar : ℕ) (bags : ℕ) (torn_bag_fraction : ℚ) (initial_per_bag : ℕ) (fallen_sugar : ℕ) :
  total_sugar = 24 →
  bags = 4 →
  (total_sugar / bags) = initial_per_bag →
  initial_per_bag = 6 →
  (initial_per_bag * torn_bag_fraction) = fallen_sugar →
  torn_bag_fraction = 1/2 →
  fallen_sugar = 3 →
  (total_sugar - fallen_sugar) = 21 :=
begin
  intros h_total h_bags h_initial_per_bag_eq h_initial_per_bag h_torn_bag_fraction_eq h_torn_bag_fraction h_fallen_sugar,
  rw [h_total, h_initial_per_bag_eq.symm, h_bags],
  norm_num at *,
  sorry
end

end remaining_sugar_l241_241102


namespace parabola_tangents_min_area_l241_241850

noncomputable def parabola_tangents (p : ℝ) : Prop :=
  ∃ (y₀ : ℝ), p > 0 ∧ (2 * Real.sqrt (y₀^2 + 2 * p) = 4)

theorem parabola_tangents_min_area (p : ℝ) : parabola_tangents 2 :=
by
  sorry

end parabola_tangents_min_area_l241_241850


namespace matt_minus_sara_l241_241984

def sales_tax_rate : ℝ := 0.08
def original_price : ℝ := 120.00
def discount_rate : ℝ := 0.25

def matt_total : ℝ := (original_price * (1 + sales_tax_rate)) * (1 - discount_rate)
def sara_total : ℝ := (original_price * (1 - discount_rate)) * (1 + sales_tax_rate)

theorem matt_minus_sara : matt_total - sara_total = 0 :=
by
  sorry

end matt_minus_sara_l241_241984


namespace number_of_common_tangents_of_two_circles_l241_241145

theorem number_of_common_tangents_of_two_circles 
  (x y : ℝ)
  (circle1 : x^2 + y^2 = 1)
  (circle2 : x^2 + y^2 - 6 * x - 8 * y + 9 = 0) :
  ∃ n : ℕ, n = 3 :=
by
  sorry

end number_of_common_tangents_of_two_circles_l241_241145


namespace printing_shop_paper_boxes_l241_241308

variable (x y : ℕ) -- Assuming x and y are natural numbers since the number of boxes can't be negative.

theorem printing_shop_paper_boxes (h1 : 80 * x + 180 * y = 2660)
                                  (h2 : x = 5 * y - 3) :
    x = 22 ∧ y = 5 := sorry

end printing_shop_paper_boxes_l241_241308


namespace largest_multiple_of_three_l241_241432

theorem largest_multiple_of_three (n : ℕ) (h : 3 * n + (3 * n + 3) + (3 * n + 6) = 117) : 3 * n + 6 = 42 :=
by
  sorry

end largest_multiple_of_three_l241_241432


namespace choose_3_out_of_13_l241_241515

theorem choose_3_out_of_13: (Nat.choose 13 3) = 286 :=
by
  sorry

end choose_3_out_of_13_l241_241515


namespace compute_combination_product_l241_241104

def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem compute_combination_product :
  combination 10 3 * combination 8 3 = 6720 :=
by
  sorry

end compute_combination_product_l241_241104


namespace range_of_f_l241_241639

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem range_of_f :
  ∀ x ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 4), f x ∈ Set.Icc (1 : ℝ) (Real.sqrt 2) := 
by
  intro x hx
  rw [Set.mem_Icc] at hx
  have : ∀ x ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 4), f x ∈ Set.Icc 1 (Real.sqrt 2) := sorry
  exact this x hx

end range_of_f_l241_241639


namespace pizza_cost_per_slice_l241_241374

theorem pizza_cost_per_slice :
  let pizza_cost := 10
  let first_topping_cost := 2
  let next_two_toppings_cost := 2
  let remaining_toppings_cost := 2
  let total_cost := pizza_cost + first_topping_cost + next_two_toppings_cost + remaining_toppings_cost
  let slices := 8
  total_cost / slices = 2 := by
  let pizza_cost := 10
  let first_topping_cost := 2
  let next_two_toppings_cost := 2
  let remaining_toppings_cost := 2
  let total_cost := pizza_cost + first_topping_cost + next_two_toppings_cost + remaining_toppings_cost
  let slices := 8
  have h : total_cost = 16 := by
    -- calculations to show total_cost = 16 can be provided here
    sorry
  have hslices : slices = 8 := rfl
  calc
    total_cost / slices = 16 / 8 : by rw [h, hslices]
                  ... = 2         : by norm_num

end pizza_cost_per_slice_l241_241374


namespace sqrt_expression_eq_1720_l241_241607

theorem sqrt_expression_eq_1720 : Real.sqrt ((43 * 42 * 41 * 40) + 1) = 1720 := by
  sorry

end sqrt_expression_eq_1720_l241_241607


namespace parallel_lines_slope_l241_241065

-- Define the given conditions
def line1_slope (x : ℝ) : ℝ := 6
def line2_slope (c : ℝ) (x : ℝ) : ℝ := 3 * c

-- State the proof problem
theorem parallel_lines_slope (c : ℝ) : 
  (∀ x : ℝ, line1_slope x = line2_slope c x) → c = 2 :=
by
  intro h
  -- Intro provides a human-readable variable and corresponding proof obligation
  -- The remainder of the proof would follow here, but instead,
  -- we use "sorry" to indicate an incomplete proof
  sorry

end parallel_lines_slope_l241_241065


namespace ultratown_run_difference_l241_241169

/-- In Ultratown, the streets are all 25 feet wide, 
and the blocks they enclose are rectangular with lengths of 500 feet and widths of 300 feet. 
Hannah runs around the block on the longer 500-foot side of the street, 
while Harry runs on the opposite, outward side of the street. 
Prove that Harry runs 200 more feet than Hannah does for every lap around the block.
-/ 
theorem ultratown_run_difference :
  let street_width : ℕ := 25
  let inner_length : ℕ := 500
  let inner_width : ℕ := 300
  let outer_length := inner_length + 2 * street_width
  let outer_width := inner_width + 2 * street_width
  let inner_perimeter := 2 * (inner_length + inner_width)
  let outer_perimeter := 2 * (outer_length + outer_width)
  (outer_perimeter - inner_perimeter) = 200 :=
by
  sorry

end ultratown_run_difference_l241_241169


namespace square_root_combination_l241_241014

theorem square_root_combination (a : ℝ) (h : 1 + a = 4 - 2 * a) : a = 1 :=
by
  -- proof goes here
  sorry

end square_root_combination_l241_241014


namespace rectangular_field_diagonal_length_l241_241554

noncomputable def diagonal_length_of_rectangular_field (a : ℝ) (A : ℝ) : ℝ :=
  let b := A / a
  let d := Real.sqrt (a^2 + b^2)
  d

theorem rectangular_field_diagonal_length :
  let a : ℝ := 14
  let A : ℝ := 135.01111065390137
  abs (diagonal_length_of_rectangular_field a A - 17.002) < 0.001 := by
    sorry

end rectangular_field_diagonal_length_l241_241554


namespace parabola_directrix_standard_eq_l241_241220

theorem parabola_directrix_standard_eq (y : ℝ) (x : ℝ) : 
  (∃ (p : ℝ), p > 0 ∧ ∀ (P : {P // P ≠ x ∨ P ≠ y}), 
  (y + 1) = p) → x^2 = 4 * y :=
sorry

end parabola_directrix_standard_eq_l241_241220


namespace equivalent_problem_l241_241828

variable (x y : ℝ)
variable (hx_ne_zero : x ≠ 0)
variable (hy_ne_zero : y ≠ 0)
variable (h : (3 * x + y) / (x - 3 * y) = -2)

theorem equivalent_problem : (x + 3 * y) / (3 * x - y) = 2 :=
by
  sorry

end equivalent_problem_l241_241828


namespace gnollish_valid_sentences_count_l241_241398

/--
The Gnollish language consists of 4 words: "splargh," "glumph," "amr," and "bork."
A sentence is valid if "splargh" does not come directly before "glumph" or "bork."
Prove that there are 240 valid 4-word sentences in Gnollish.
-/
theorem gnollish_valid_sentences_count : 
  let words := ["splargh", "glumph", "amr", "bork"]
  let total_sentences := (words.length ^ 4)
  let invalid_conditions (w1 w2 : String) := 
    (w1 = "splargh" ∧ (w2 = "glumph" ∨ w2 = "bork"))
  let invalid_count : ℕ := 
    2 * words.length * words.length * (words.length - 1)
  let valid_sentences := total_sentences - invalid_count
  valid_sentences = 240 :=
by
  let words := ["splargh", "glumph", "amr", "bork"]
  let total_sentences := (words.length ^ 4)
  let invalid_conditions (w1 w2 : String) := 
    (w1 = "splargh" ∧ (w2 = "glumph" ∨ w2 = "bork"))
  let invalid_count : ℕ := 
    2 * words.length * words.length * (words.length - 1)
  let valid_sentences := total_sentences - invalid_count
  have : valid_sentences = 240 := by sorry
  exact this

end gnollish_valid_sentences_count_l241_241398


namespace binom_12_10_eq_66_l241_241465

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_12_10_eq_66 : binom 12 10 = 66 := 
by
  -- We state the given condition for binomial coefficients.
  let binom_coeff := binom 12 10
  -- We will use the formula \binom{n}{k} = \frac{n!}{k!(n-k)!}.
  change binom_coeff = Nat.factorial 12 / (Nat.factorial 10 * Nat.factorial (12 - 10))
  -- Simplify using factorial values and basic arithmetic to reach the conclusion.
  sorry

end binom_12_10_eq_66_l241_241465


namespace car_drive_time_60_kmh_l241_241438

theorem car_drive_time_60_kmh
  (t : ℝ)
  (avg_speed : ℝ := 80)
  (dist_speed_60 : ℝ := 60 * t)
  (time_speed_90 : ℝ := 2 / 3)
  (dist_speed_90 : ℝ := 90 * time_speed_90)
  (total_distance : ℝ := dist_speed_60 + dist_speed_90)
  (total_time : ℝ := t + time_speed_90)
  (avg_speed_eq : avg_speed = total_distance / total_time) :
  t = 1 / 3 := 
sorry

end car_drive_time_60_kmh_l241_241438


namespace cos_angle_of_vectors_l241_241641

variables (a b : EuclideanSpace ℝ (Fin 2))

theorem cos_angle_of_vectors (h1 : ‖a‖ = 2) (h2 : ‖b‖ = 1) (h3 : ‖a - b‖ = 2) :
  (inner a b) / (‖a‖ * ‖b‖) = 1/4 :=
by
  sorry

end cos_angle_of_vectors_l241_241641


namespace moles_of_CO2_formed_l241_241126

-- Define the reaction
def reaction (HCl NaHCO3 CO2 : ℕ) : Prop :=
  HCl = NaHCO3 ∧ HCl + NaHCO3 = CO2

-- Given conditions
def given_conditions : Prop :=
  ∃ (HCl NaHCO3 CO2 : ℕ),
    reaction HCl NaHCO3 CO2 ∧ HCl = 3 ∧ NaHCO3 = 3

-- Prove the number of moles of CO2 formed is 3.
theorem moles_of_CO2_formed : given_conditions → ∃ CO2 : ℕ, CO2 = 3 :=
  by
    intros h
    sorry

end moles_of_CO2_formed_l241_241126


namespace pythagorean_triplets_l241_241612

theorem pythagorean_triplets (a b c : ℤ) (h : a^2 + b^2 = c^2) :
  ∃ d p q : ℤ, a = 2 * d * p * q ∧ b = d * (q^2 - p^2) ∧ c = d * (p^2 + q^2) := sorry

end pythagorean_triplets_l241_241612


namespace no_integer_n_satisfies_conditions_l241_241488

theorem no_integer_n_satisfies_conditions :
  ¬ ∃ n : ℕ, 0 < n ∧ 1000 ≤ n / 5 ∧ n / 5 ≤ 9999 ∧ 1000 ≤ 5 * n ∧ 5 * n ≤ 9999 :=
by
  sorry

end no_integer_n_satisfies_conditions_l241_241488


namespace probability_penny_nickel_heads_l241_241044

noncomputable def num_outcomes : ℕ := 2^4
noncomputable def num_successful_outcomes : ℕ := 2 * 2

theorem probability_penny_nickel_heads :
  (num_successful_outcomes : ℚ) / num_outcomes = 1 / 4 :=
by
  sorry

end probability_penny_nickel_heads_l241_241044


namespace probability_factor_36_l241_241279

def is_factor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

theorem probability_factor_36 : 
  (set.univ.filter (λ n : ℕ, n > 0 ∧ n ≤ 36 ∧ is_factor 36 n)).card.toRat / 36 = 1 / 4 := 
by
  sorry

end probability_factor_36_l241_241279


namespace probability_factor_of_36_l241_241290

theorem probability_factor_of_36 : (∃ p : ℚ, p = 1 / 4 ∧
  let n := 36 in
  let factors := {d | d ∣ n ∧ d > 0 ∧ d ≤ n} in
  let total_pos_ints := 36 in
  let num_factors := set.card factors in
  p = num_factors / total_pos_ints) :=
begin
  sorry
end

end probability_factor_of_36_l241_241290


namespace max_k_l241_241537

theorem max_k (n : ℕ) (h : n ≥ 3) (M : Finset (fin n)) :
  ∃ k : ℕ, 
    (if n ≤ 5 then k = Nat.choose n 3 else k = Nat.choose (n-1) 2) := 
by
  sorry

end max_k_l241_241537


namespace find_max_marks_l241_241588

theorem find_max_marks (M : ℝ) (h1 : 0.60 * M = 80 + 100) : M = 300 := 
by
  sorry

end find_max_marks_l241_241588


namespace four_digit_number_properties_l241_241115

theorem four_digit_number_properties :
  ∃ (a b c d : ℕ), 
    a + b + c + d = 8 ∧ 
    a = 3 * b ∧ 
    d = 4 * c ∧ 
    1000 * a + 100 * b + 10 * c + d = 6200 :=
by
  sorry

end four_digit_number_properties_l241_241115


namespace find_levels_satisfying_surface_area_conditions_l241_241503

theorem find_levels_satisfying_surface_area_conditions (n : ℕ) :
  let A_total_lateral := n * (n + 1) * Real.pi
  let A_total_vertical := Real.pi * n^2
  let A_total := n * (3 * n + 1) * Real.pi
  A_total_lateral = 0.35 * A_total → n = 13 :=
by
  intros A_total_lateral A_total_vertical A_total h
  sorry

end find_levels_satisfying_surface_area_conditions_l241_241503


namespace domino_perfect_play_winner_l241_241660

theorem domino_perfect_play_winner :
  ∀ {PlayerI PlayerII : Type} 
    (legal_move : PlayerI → PlayerII → Prop)
    (initial_move : PlayerI → Prop)
    (next_moves : PlayerII → PlayerI → PlayerII → Prop),
    (∀ pI pII, legal_move pI pII) → 
    (∃ m, initial_move m) → 
    (∀ mI mII, next_moves mII mI mII) → 
    ∃ winner, winner = PlayerI :=
by
  sorry

end domino_perfect_play_winner_l241_241660


namespace original_number_of_members_l241_241364

-- Define the initial conditions
variables (x y : ℕ)

-- First condition: if five 9-year-old members leave
def condition1 : Prop := x * y - 45 = (y + 1) * (x - 5)

-- Second condition: if five 17-year-old members join
def condition2 : Prop := x * y + 85 = (y + 1) * (x + 5)

-- The theorem to be proven
theorem original_number_of_members (h1 : condition1 x y) (h2 : condition2 x y) : x = 20 :=
by sorry

end original_number_of_members_l241_241364


namespace probability_factor_of_36_l241_241253

theorem probability_factor_of_36 : 
  (finset.card (finset.filter (λ x : ℕ, 36 % x = 0) (finset.range (36 + 1))) : ℚ) / 36 = 1 / 4 := 
by 
sorry

end probability_factor_of_36_l241_241253


namespace book_pages_count_l241_241678

theorem book_pages_count :
  (∀ n : ℕ, n = 4 → 42 * n = 168) ∧
  (∀ n : ℕ, n = 2 → 50 * n = 100) ∧
  (∀ p1 p2 : ℕ, p1 = 168 ∧ p2 = 100 → p1 + p2 = 268) ∧
  (∀ p : ℕ, p = 268 → p + 30 = 298) →
  298 = 298 := by
  sorry

end book_pages_count_l241_241678


namespace probability_three_tails_one_head_l241_241161

theorem probability_three_tails_one_head :
  let p := (1 : ℝ) / 2 in
  (∃ t1 t2 t3 t4 : bool, (t1 = tt) ∨ (t2 = tt) ∨ (t3 = tt) ∨ (t4 = tt)) → 
  (∑ e in {t | t = tt ∨ t = ff}, p ^ 4) * 4 = 1 / 4 :=
by sorry

end probability_three_tails_one_head_l241_241161


namespace range_H_l241_241910

def H (x : ℝ) : ℝ := |x + 2| - |x - 3|

theorem range_H : Set.Iic (5 : ℝ) = Set.range H :=
by
  sorry

end range_H_l241_241910


namespace fish_weight_l241_241157

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

end fish_weight_l241_241157


namespace F_monotonically_decreasing_xf_vs_1divxf_l241_241932

open Real

variables {f : ℝ → ℝ}

-- Conditions
axiom f_pos : ∀ (x : ℝ), x > 0 → f(x) > 0
axiom f_deriv_neg_ratio : ∀ (x : ℝ), x > 0 → f'(x) / f(x) < -1

-- Question Ⅰ: Monotonicity of F(x) = e^x f(x)
theorem F_monotonically_decreasing (x : ℝ) (hx : x > 0) :
  deriv (λ x, exp x * f x) x < 0 :=
sorry

-- Question Ⅱ: Magnitude comparison for 0 < x < 1
theorem xf_vs_1divxf (x : ℝ) (hx1 : 0 < x) (hx2 : x < 1) :
  x * f x > (1 / x) * f (1 / x) :=
sorry

end F_monotonically_decreasing_xf_vs_1divxf_l241_241932


namespace find_a_l241_241732

variable (a : ℝ)

def average_condition (a : ℝ) : Prop :=
  ((2 * a + 16) + (3 * a - 8)) / 2 = 74

theorem find_a (h: average_condition a) : a = 28 :=
  sorry

end find_a_l241_241732


namespace megan_numbers_difference_l241_241681

theorem megan_numbers_difference 
  (x1 x2 x3 x4 x5 : ℝ) 
  (h_mean3 : (x1 + x2 + x3) / 3 = -3)
  (h_mean4 : (x1 + x2 + x3 + x4) / 4 = 4)
  (h_mean5 : (x1 + x2 + x3 + x4 + x5) / 5 = -5) :
  x4 - x5 = 66 :=
by
  sorry

end megan_numbers_difference_l241_241681


namespace tan_expression_l241_241898

theorem tan_expression (a : ℝ) (h₀ : 45 = 2 * a) (h₁ : Real.tan 45 = 1) 
  (h₂ : Real.tan (2 * a) = 2 * Real.tan a / (1 - Real.tan a * Real.tan a)) :
  Real.tan a / (1 - Real.tan a * Real.tan a) = 1 / 2 :=
by 
  sorry

end tan_expression_l241_241898


namespace smallest_number_l241_241718

theorem smallest_number (n : ℕ) : (∀ d ∈ [8, 14, 26, 28], (n - 18) % d = 0) → n = 746 := by
  sorry

end smallest_number_l241_241718


namespace molecular_weight_N2O5_l241_241417

variable {x : ℕ}

theorem molecular_weight_N2O5 (hx : 10 * 108 = 1080) : (108 * x = 1080 * x / 10) :=
by
  sorry

end molecular_weight_N2O5_l241_241417


namespace contrapositive_of_square_sum_zero_l241_241406

theorem contrapositive_of_square_sum_zero (a b : ℝ) :
  (a ≠ 0 ∨ b ≠ 0) → a^2 + b^2 ≠ 0 :=
by
  sorry

end contrapositive_of_square_sum_zero_l241_241406


namespace sum_infinite_series_l241_241759

theorem sum_infinite_series : 
  ∑' n : ℕ, (3 * (n + 1) - 1) / ((n + 1) * ((n + 1) + 1) * ((n + 1) + 3)) = 73 / 12 := 
by sorry

end sum_infinite_series_l241_241759


namespace unique_B_squared_l241_241185

theorem unique_B_squared (B : Matrix (Fin 2) (Fin 2) ℝ) (h : B^4 = 0) : 
  ∃! B2 : Matrix (Fin 2) (Fin 2) ℝ, B2 = B * B :=
sorry

end unique_B_squared_l241_241185


namespace impossible_to_transport_stones_l241_241180

-- Define the conditions of the problem
def stones : List ℕ := List.range' 370 (468 - 370 + 2 + 1) 2
def truck_capacity : ℕ := 3000
def number_of_trucks : ℕ := 7
def number_of_stones : ℕ := 50

-- Prove that it is impossible to transport the stones using the given trucks
theorem impossible_to_transport_stones :
  stones.length = number_of_stones →
  (∀ weights ∈ stones.sublists, (weights.sum ≤ truck_capacity → List.length weights ≤ number_of_trucks)) → 
  false :=
by
  sorry

end impossible_to_transport_stones_l241_241180


namespace unique_positive_integer_triples_l241_241614

theorem unique_positive_integer_triples (a b c : ℕ) (h1 : ab + 3 * b * c = 63) (h2 : ac + 3 * b * c = 39) : 
∃! (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ ab + 3 * b * c = 63 ∧ ac + 3 * b * c = 39 :=
by sorry

end unique_positive_integer_triples_l241_241614


namespace range_of_a_l241_241355

noncomputable def f (x a : ℝ) := x^2 + 2 * x - a
noncomputable def g (x : ℝ) := 2 * x + 2 * Real.log x

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2, (1/e) ≤ x1 ∧ x1 < x2 ∧ x2 ≤ e ∧ f x1 a = g x1 ∧ f x2 a = g x2) ↔ 
  1 < a ∧ a ≤ (1/(e^2)) + 2 := 
sorry

end range_of_a_l241_241355


namespace minimum_one_by_one_squares_l241_241763

theorem minimum_one_by_one_squares :
  ∀ (x y z : ℕ), 9 * x + 4 * y + z = 49 → (z = 3) :=
  sorry

end minimum_one_by_one_squares_l241_241763


namespace card_sequence_probability_l241_241996

noncomputable def probability_of_sequence : ℚ :=
  (4/52) * (4/51) * (4/50)

theorem card_sequence_probability :
  probability_of_sequence = 4/33150 := 
by 
  sorry

end card_sequence_probability_l241_241996


namespace discount_difference_is_correct_l241_241089

-- Define the successive discounts in percentage
def discount1 : ℝ := 0.25
def discount2 : ℝ := 0.15
def discount3 : ℝ := 0.10

-- Define the store's claimed discount
def claimed_discount : ℝ := 0.45

-- Calculate the true discount
def true_discount : ℝ := 1 - ((1 - discount1) * (1 - discount2) * (1 - discount3))

-- Calculate the difference between the true discount and the claimed discount
def discount_difference : ℝ := claimed_discount - true_discount

-- State the theorem to be proved
theorem discount_difference_is_correct : discount_difference = 2.375 / 100 := by
  sorry

end discount_difference_is_correct_l241_241089


namespace train_length_is_330_meters_l241_241749

noncomputable def train_speed : ℝ := 60 -- in km/hr
noncomputable def man_speed : ℝ := 6    -- in km/hr
noncomputable def time : ℝ := 17.998560115190788  -- in seconds

noncomputable def relative_speed_km_per_hr : ℝ := train_speed + man_speed
noncomputable def conversion_factor : ℝ := 5 / 18

noncomputable def relative_speed_m_per_s : ℝ := 
  relative_speed_km_per_hr * conversion_factor

theorem train_length_is_330_meters : 
  (relative_speed_m_per_s * time) = 330 := 
sorry

end train_length_is_330_meters_l241_241749


namespace faye_country_albums_l241_241857

theorem faye_country_albums (C : ℕ) (h1 : 6 * C + 18 = 30) : C = 2 :=
by
  -- This is the theorem statement with the necessary conditions and question
  sorry

end faye_country_albums_l241_241857


namespace target_hit_probability_l241_241327

-- Define the probabilities given in the problem
def prob_A_hits : ℚ := 9 / 10
def prob_B_hits : ℚ := 8 / 9

-- The required probability that at least one hits the target
def prob_target_hit : ℚ := 89 / 90

-- Theorem stating that the probability calculated matches the expected outcome
theorem target_hit_probability :
  1 - ((1 - prob_A_hits) * (1 - prob_B_hits)) = prob_target_hit :=
by
  sorry

end target_hit_probability_l241_241327


namespace lattice_midpoint_l241_241365

theorem lattice_midpoint (points : Fin 5 → ℤ × ℤ) :
  ∃ (i j : Fin 5), i ≠ j ∧ 
  let (x1, y1) := points i 
  let (x2, y2) := points j
  (x1 + x2) % 2 = 0 ∧ (y1 + y2) % 2 = 0 := 
sorry

end lattice_midpoint_l241_241365


namespace hotel_flat_fee_l241_241312

theorem hotel_flat_fee
  (f n : ℝ)
  (h1 : f + 3 * n = 195)
  (h2 : f + 7 * n = 380) :
  f = 56.25 :=
by sorry

end hotel_flat_fee_l241_241312


namespace least_number_of_tiles_l241_241741

-- Definitions for classroom dimensions
def classroom_length : ℕ := 624 -- in cm
def classroom_width : ℕ := 432 -- in cm

-- Definitions for tile dimensions
def rectangular_tile_length : ℕ := 60
def rectangular_tile_width : ℕ := 80
def triangular_tile_base : ℕ := 40
def triangular_tile_height : ℕ := 40

-- Definition for the area calculation
def area (length width : ℕ) : ℕ := length * width
def area_triangular_tile (base height : ℕ) : ℕ := (base * height) / 2

-- Define the area of the classroom and tiles
def classroom_area : ℕ := area classroom_length classroom_width
def rectangular_tile_area : ℕ := area rectangular_tile_length rectangular_tile_width
def triangular_tile_area : ℕ := area_triangular_tile triangular_tile_base triangular_tile_height

-- Define the number of tiles required
def number_of_rectangular_tiles : ℕ := (classroom_area + rectangular_tile_area - 1) / rectangular_tile_area -- ceiling division in lean
def number_of_triangular_tiles : ℕ := (classroom_area + triangular_tile_area - 1) / triangular_tile_area -- ceiling division in lean

-- Define the minimum number of tiles required
def minimum_number_of_tiles : ℕ := min number_of_rectangular_tiles number_of_triangular_tiles

-- The main theorem establishing the least number of tiles required
theorem least_number_of_tiles : minimum_number_of_tiles = 57 := by
    sorry

end least_number_of_tiles_l241_241741


namespace cost_of_soap_per_year_l241_241381

-- Conditions:
def duration_of_soap (bar: Nat) : Nat := 2
def cost_per_bar (bar: Nat) : Real := 8.0
def months_in_year : Nat := 12

-- Derived quantity
def bars_needed (months: Nat) (duration: Nat): Nat := months / duration

-- Theorem statement:
theorem cost_of_soap_per_year : 
  let n := bars_needed months_in_year (duration_of_soap 1)
  n * (cost_per_bar 1) = 48.0 := 
  by 
    -- Skipping proof
    sorry

end cost_of_soap_per_year_l241_241381


namespace coterminal_angle_l241_241444

theorem coterminal_angle :
  ∀ θ : ℤ, (θ - 60) % 360 = 0 → θ = -300 ∨ θ = -60 ∨ θ = 600 ∨ θ = 1380 :=
by
  sorry

end coterminal_angle_l241_241444


namespace find_dividend_l241_241414

def dividend_problem (dividend divisor : ℕ) : Prop :=
  (15 * divisor + 5 = dividend) ∧ (dividend + divisor + 15 + 5 = 2169)

theorem find_dividend : ∃ dividend, ∃ divisor, dividend_problem dividend divisor ∧ dividend = 2015 :=
sorry

end find_dividend_l241_241414


namespace arithmetic_geometric_sequence_k4_l241_241913

theorem arithmetic_geometric_sequence_k4 (a : ℕ → ℝ) (d : ℝ) (h_d_ne_zero : d ≠ 0)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_geo_seq : ∃ k : ℕ → ℕ, k 0 = 1 ∧ k 1 = 2 ∧ k 2 = 6 ∧ ∀ i, a (k i + 1) / a (k i) = a (k i + 2) / a (k i + 1)) :
  ∃ k4 : ℕ, k4 = 22 := 
by
  sorry

end arithmetic_geometric_sequence_k4_l241_241913


namespace perpendicular_lines_m_l241_241523

theorem perpendicular_lines_m (m : ℝ) :
  (∀ (x y : ℝ), x - 2 * y + 5 = 0 → 2 * x + m * y - 6 = 0) →
  m = 1 :=
by
  sorry

end perpendicular_lines_m_l241_241523


namespace simplify_expression_l241_241606

theorem simplify_expression : 
  3 * Real.sqrt 48 - 9 * Real.sqrt (1 / 3) - Real.sqrt 3 * (2 - Real.sqrt 27) = 7 * Real.sqrt 3 + 9 :=
by
  -- The proof is omitted as per the instructions
  sorry

end simplify_expression_l241_241606


namespace max_value_f_l241_241337

noncomputable def f (x : ℝ) : ℝ := 8 * Real.sin x - Real.tan x

theorem max_value_f : 
  ∃ x ∈ Set.Ioo 0 (Real.pi / 2), ∀ y ∈ Set.Ioo 0 (Real.pi / 2), f y ≤ f x ∧ f x = 3 * Real.sqrt 3 :=
by
  sorry

end max_value_f_l241_241337


namespace probability_length_error_in_interval_l241_241001

noncomputable def normal_dist_prob (μ σ : ℝ) (a b : ℝ) : ℝ :=
∫ x in a..b, (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-((x - μ) ^ 2) / (2 * σ ^ 2))

theorem probability_length_error_in_interval :
  normal_dist_prob 0 3 3 6 = 0.1359 :=
by
  sorry

end probability_length_error_in_interval_l241_241001


namespace exists_pairwise_coprime_product_of_two_consecutive_integers_l241_241038

theorem exists_pairwise_coprime_product_of_two_consecutive_integers (n : ℕ) (h : 0 < n) :
  ∃ (a : Fin n → ℕ), (∀ i, 2 ≤ a i) ∧ (Pairwise (IsCoprime on fun i => a i)) ∧ (∃ k : ℕ, (Finset.univ.prod a) - 1 = k * (k + 1)) := 
sorry

end exists_pairwise_coprime_product_of_two_consecutive_integers_l241_241038


namespace sequence_an_l241_241027

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

-- Conditions
axiom S_formula (n : ℕ) (h₁ : n > 0) : S n = 2 * a n - 2

-- Proof goal
theorem sequence_an (n : ℕ) (h₁ : n > 0) : a n = 2 ^ n := by
  sorry

end sequence_an_l241_241027


namespace number_of_Slurpees_l241_241958

theorem number_of_Slurpees
  (total_money : ℕ)
  (cost_per_Slurpee : ℕ)
  (change : ℕ)
  (spent_money := total_money - change)
  (number_of_Slurpees := spent_money / cost_per_Slurpee)
  (h1 : total_money = 20)
  (h2 : cost_per_Slurpee = 2)
  (h3 : change = 8) :
  number_of_Slurpees = 6 := by
  sorry

end number_of_Slurpees_l241_241958


namespace train_length_is_250_l241_241442

-- Define the length of the train
def train_length (L : ℝ) (V : ℝ) :=
  -- Condition 1
  (V = L / 10) → 
  -- Condition 2
  (V = (L + 1250) / 60) → 
  -- Question
  L = 250

-- Here's the statement that we expect to prove
theorem train_length_is_250 (L V : ℝ) : train_length L V :=
by {
  -- sorry is a placeholder to indicate the theorem proof is omitted
  sorry
}

end train_length_is_250_l241_241442


namespace isosceles_triangle_base_angle_l241_241175

theorem isosceles_triangle_base_angle (α : ℕ) (base_angle : ℕ) 
  (hα : α = 40) (hsum : α + 2 * base_angle = 180) : 
  base_angle = 70 :=
sorry

end isosceles_triangle_base_angle_l241_241175


namespace arith_seq_a1_eq_15_l241_241144

variable {a : ℕ → ℤ} (a_seq : ∀ n, a n = a 1 + (n-1) * d)
variable {a_4 : ℤ} (h4 : a 4 = 9)
variable {a_8 : ℤ} (h8 : a 8 = -a 9)

theorem arith_seq_a1_eq_15 (a_seq : ∀ n, a n = a 1 + (n-1) * d) (h4 : a 4 = 9) (h8 : a 8 = -a 9) : a 1 = 15 :=
by
  -- Proof should go here
  sorry

end arith_seq_a1_eq_15_l241_241144


namespace difference_before_flipping_l241_241708

-- Definitions based on the conditions:
variables (Y G : ℕ) -- Number of yellow and green papers

-- Condition: flipping 16 yellow papers changes counts as described
def papers_after_flipping (Y G : ℕ) : Prop :=
  Y - 16 = G + 16

-- Condition: after flipping, there are 64 more green papers than yellow papers.
def green_more_than_yellow_after_flipping (G Y : ℕ) : Prop :=
  G + 16 = (Y - 16) + 64

-- Statement: Prove the difference in the number of green and yellow papers before flipping was 32.
theorem difference_before_flipping (Y G : ℕ) (h1 : papers_after_flipping Y G) (h2 : green_more_than_yellow_after_flipping G Y) :
  (Y - G) = 32 :=
by
  sorry

end difference_before_flipping_l241_241708


namespace binomial_12_10_l241_241449

def binomial (n k : ℕ) : ℕ := n.choose k

theorem binomial_12_10 : binomial 12 10 = 66 := by
  -- The proof will go here
  sorry

end binomial_12_10_l241_241449


namespace malcolm_brushes_teeth_l241_241033

theorem malcolm_brushes_teeth :
  (∃ (M : ℕ), M = 180 ∧ (∃ (N : ℕ), N = 90 ∧ (M / N = 2))) :=
by
  sorry

end malcolm_brushes_teeth_l241_241033


namespace binom_12_10_eq_66_l241_241461

theorem binom_12_10_eq_66 : nat.choose 12 10 = 66 :=
by
  sorry

end binom_12_10_eq_66_l241_241461


namespace probability_factor_of_36_l241_241292

theorem probability_factor_of_36 : (∃ p : ℚ, p = 1 / 4 ∧
  let n := 36 in
  let factors := {d | d ∣ n ∧ d > 0 ∧ d ≤ n} in
  let total_pos_ints := 36 in
  let num_factors := set.card factors in
  p = num_factors / total_pos_ints) :=
begin
  sorry
end

end probability_factor_of_36_l241_241292


namespace least_n_for_perfect_square_l241_241336

theorem least_n_for_perfect_square (n : ℕ) :
  (∀ m : ℕ, 2^8 + 2^11 + 2^n = m * m) → n = 12 := sorry

end least_n_for_perfect_square_l241_241336


namespace central_angle_measure_l241_241137

-- Given conditions
def radius : ℝ := 2
def area : ℝ := 4

-- Central angle α
def central_angle : ℝ := 2

-- Theorem statement: The central angle measure is 2 radians
theorem central_angle_measure :
  ∃ α : ℝ, α = central_angle ∧ area = (1/2) * (α * radius) := 
sorry

end central_angle_measure_l241_241137


namespace complex_trajectory_is_ellipse_l241_241836

open Complex

theorem complex_trajectory_is_ellipse (z : ℂ) (h : abs (z - i) + abs (z + i) = 3) : 
  true := 
sorry

end complex_trajectory_is_ellipse_l241_241836


namespace total_dogs_l241_241231

theorem total_dogs (D : ℕ) 
(h1 : 12 = 12)
(h2 : D / 2 = D / 2)
(h3 : D / 4 = D / 4)
(h4 : 10 = 10)
(h_eq : 12 + D / 2 + D / 4 + 10 = D) : 
D = 88 := by
sorry

end total_dogs_l241_241231


namespace max_value_3x_sub_9x_l241_241121

open Real

theorem max_value_3x_sub_9x : ∃ x : ℝ, 3^x - 9^x ≤ 1/4 ∧ (∀ y : ℝ, 3^y - 9^y ≤ 3^x - 9^x) :=
by
  sorry

end max_value_3x_sub_9x_l241_241121


namespace discount_percentage_is_25_l241_241181

def piano_cost := 500
def lessons_count := 20
def lesson_price := 40
def total_paid := 1100

def lessons_cost := lessons_count * lesson_price
def total_cost := piano_cost + lessons_cost
def discount_amount := total_cost - total_paid
def discount_percentage := (discount_amount / lessons_cost) * 100

theorem discount_percentage_is_25 : discount_percentage = 25 := by
  sorry

end discount_percentage_is_25_l241_241181


namespace mangoes_per_kg_l241_241903

theorem mangoes_per_kg (total_kg : ℕ) (sold_market_kg : ℕ) (sold_community_factor : ℚ) (remaining_mangoes : ℕ) (mangoes_per_kg : ℕ) :
  total_kg = 60 ∧ sold_market_kg = 20 ∧ sold_community_factor = 1/2 ∧ remaining_mangoes = 160 → mangoes_per_kg = 8 :=
  by
  sorry

end mangoes_per_kg_l241_241903


namespace algebraic_expression_l241_241611

def ast (n : ℕ) : ℕ := sorry

axiom condition_1 : ast 1 = 1
axiom condition_2 : ∀ (n : ℕ), ast (n + 1) = 3 * ast n

theorem algebraic_expression (n : ℕ) :
  n > 0 → ast n = 3^(n - 1) :=
by
  -- Proof to be completed
  sorry

end algebraic_expression_l241_241611


namespace Maria_soap_cost_l241_241384
-- Import the entire Mathlib library
  
theorem Maria_soap_cost (soap_last_months : ℕ) (cost_per_bar : ℝ) (months_in_year : ℕ):
  (soap_last_months = 2) -> 
  (cost_per_bar = 8.00) ->
  (months_in_year = 12) -> 
  (months_in_year / soap_last_months * cost_per_bar = 48.00) := 
by
  intros h_soap_last h_cost h_year
  sorry

end Maria_soap_cost_l241_241384


namespace min_value_range_of_x_l241_241922

variables (a b x : ℝ)

-- Problem 1: Prove the minimum value of 1/a + 4/b given a + b = 1, a > 0, b > 0
theorem min_value (h1 : a + b = 1) (h2 : a > 0) (h3 : b > 0) : 
  ∃ c, c = 9 ∧ ∀ y, ∃ (a b : ℝ), a + b = 1 ∧ a > 0 ∧ b > 0 → (1/a + 4/b) ≥ y :=
sorry

-- Problem 2: Prove the range of x for which 1/a + 4/b ≥ |2x - 1| - |x + 1|
theorem range_of_x (h : ∀ (a b : ℝ), a + b = 1 → a > 0 → b > 0 → (1/a + 4/b) ≥ (|2*x - 1| - |x + 1|)) :
  -7 ≤ x ∧ x ≤ 11 :=
sorry

end min_value_range_of_x_l241_241922


namespace binom_12_10_eq_66_l241_241467

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_12_10_eq_66 : binom 12 10 = 66 := 
by
  -- We state the given condition for binomial coefficients.
  let binom_coeff := binom 12 10
  -- We will use the formula \binom{n}{k} = \frac{n!}{k!(n-k)!}.
  change binom_coeff = Nat.factorial 12 / (Nat.factorial 10 * Nat.factorial (12 - 10))
  -- Simplify using factorial values and basic arithmetic to reach the conclusion.
  sorry

end binom_12_10_eq_66_l241_241467


namespace filled_sandbag_weight_is_correct_l241_241412

-- Define the conditions
def sandbag_weight : ℝ := 250
def fill_percent : ℝ := 0.80
def heavier_factor : ℝ := 1.40

-- Define the intermediate weights
def sand_weight : ℝ := sandbag_weight * fill_percent
def extra_weight : ℝ := sand_weight * (heavier_factor - 1)
def filled_material_weight : ℝ := sand_weight + extra_weight

-- Define the total weight including the empty sandbag
def total_weight : ℝ := sandbag_weight + filled_material_weight

-- Prove the total weight is correct
theorem filled_sandbag_weight_is_correct : total_weight = 530 := 
by sorry

end filled_sandbag_weight_is_correct_l241_241412


namespace length_of_each_piece_l241_241156

theorem length_of_each_piece (rod_length : ℝ) (num_pieces : ℕ) (h₁ : rod_length = 42.5) (h₂ : num_pieces = 50) : (rod_length / num_pieces * 100) = 85 := 
by 
  sorry

end length_of_each_piece_l241_241156


namespace lines_are_parallel_l241_241851

theorem lines_are_parallel : 
  ∀ (x y : ℝ), (2 * x - y = 7) → (2 * x - y - 1 = 0) → False :=
by
  sorry  -- Proof will be filled in later

end lines_are_parallel_l241_241851


namespace find_a_l241_241928

noncomputable def S_n (n : ℕ) (a : ℝ) : ℝ := 2 * 3^n + a
noncomputable def a_1 (a : ℝ) : ℝ := S_n 1 a
noncomputable def a_2 (a : ℝ) : ℝ := S_n 2 a - S_n 1 a
noncomputable def a_3 (a : ℝ) : ℝ := S_n 3 a - S_n 2 a

theorem find_a (a : ℝ) : a_1 a * a_3 a = (a_2 a)^2 → a = -2 :=
by
  sorry

end find_a_l241_241928


namespace total_situps_l241_241896

def situps (b c j : ℕ) : ℕ := b * 1 + c * 2 + j * 3

theorem total_situps :
  ∀ (b c j : ℕ),
    b = 45 →
    c = 2 * b →
    j = c + 5 →
    situps b c j = 510 :=
by intros b c j hb hc hj
   sorry

end total_situps_l241_241896


namespace nested_fraction_expression_l241_241721

theorem nested_fraction_expression : 
  1 + (1 / (1 - (1 / (1 + (1 / 2))))) = 4 := 
by sorry

end nested_fraction_expression_l241_241721


namespace simplify_and_evaluate_expr_l241_241688

namespace SimplificationProof

variable (x : ℝ)

theorem simplify_and_evaluate_expr (h : x = Real.sqrt 5 - 1) :
  ((x / (x - 1) - 1) / ((x ^ 2 - 1) / (x ^ 2 - 2 * x + 1))) = Real.sqrt 5 / 5 :=
by
  sorry

end SimplificationProof

end simplify_and_evaluate_expr_l241_241688


namespace binomial_12_10_l241_241451

def binomial (n k : ℕ) : ℕ := n.choose k

theorem binomial_12_10 : binomial 12 10 = 66 := by
  -- The proof will go here
  sorry

end binomial_12_10_l241_241451


namespace son_l241_241709

variable (M S : ℕ)

theorem son's_age (h1 : M = 4 * S) (h2 : (M - 3) + (S - 3) = 49) : S = 11 :=
by
  sorry

end son_l241_241709


namespace prob_at_least_one_multiple_of_4_60_l241_241805

def num_multiples_of_4 (n : ℕ) : ℕ :=
  n / 4

def total_numbers_in_range (n : ℕ) : ℕ :=
  n

def num_not_multiples_of_4 (n : ℕ) : ℕ :=
  total_numbers_in_range n - num_multiples_of_4 n

def prob_no_multiple_of_4 (n : ℕ) : ℚ :=
  let p := num_not_multiples_of_4 n / total_numbers_in_range n
  p * p

def prob_at_least_one_multiple_of_4 (n : ℕ) : ℚ :=
  1 - prob_no_multiple_of_4 n

theorem prob_at_least_one_multiple_of_4_60 :
  prob_at_least_one_multiple_of_4 60 = 7 / 16 :=
by
  -- Proof is skipped.
  sorry

end prob_at_least_one_multiple_of_4_60_l241_241805


namespace quadratic_two_distinct_real_roots_l241_241643

theorem quadratic_two_distinct_real_roots (k : ℝ) :
  2 * k ≠ 0 → (8 * k + 1)^2 - 64 * k^2 > 0 → k > -1 / 16 ∧ k ≠ 0 :=
by
  sorry

end quadratic_two_distinct_real_roots_l241_241643


namespace area_isosceles_right_triangle_l241_241050

open Real

-- Define the condition that the hypotenuse of an isosceles right triangle is 4√2 units
def hypotenuse (a b : ℝ) : Prop := a^2 + b^2 = (4 * sqrt 2)^2

-- State the theorem to prove the area of the triangle is 8 square units
theorem area_isosceles_right_triangle (a b : ℝ) (h : hypotenuse a b) : 
  a = b → 1/2 * a * b = 8 := 
by 
  intros
  -- Proof steps are not required, so we use 'sorry'
  sorry

end area_isosceles_right_triangle_l241_241050


namespace second_pipe_filling_time_l241_241553

theorem second_pipe_filling_time (T : ℝ) :
  (∃ T : ℝ, (1 / 8 + 1 / T = 1 / 4.8) ∧ T = 12) :=
by
  sorry

end second_pipe_filling_time_l241_241553


namespace probability_B_more_points_than_A_l241_241329

open Nat

theorem probability_B_more_points_than_A 
  (teams : Finset ℕ)
  (h_teams_card : teams.card = 8)
  (h_play_conditions : ∀ (game : ℕ × ℕ), game ∈ teams ×ˢ teams → game.1 ≠ game.2)
  (h_first_game_B_win : ∃ game : ℕ × ℕ, game.1 = 1 ∧ game.2 = 0 ∧ game ∈ teams ×ˢ teams)
  (equal_chance : ∀ game : ℕ × ℕ, game ∈ teams ×ˢ teams → (game.1, game.2 ∈ teams → (0.5)))
  :
  let p := Rat.mk 793 2048 in
  (p = \frac{793}{2048}):
  true :=
sorry

end probability_B_more_points_than_A_l241_241329


namespace problem_trigonometric_identity_l241_241917

-- Define the problem conditions
theorem problem_trigonometric_identity
  (α : ℝ)
  (h : 3 * Real.sin (33 * Real.pi / 14 + α) = -5 * Real.cos (5 * Real.pi / 14 + α)) :
  Real.tan (5 * Real.pi / 14 + α) = -5 / 3 :=
sorry

end problem_trigonometric_identity_l241_241917


namespace battery_charge_to_60_percent_l241_241084

noncomputable def battery_charge_time (initial_charge_percent : ℝ) (initial_time_minutes : ℕ) (additional_time_minutes : ℕ) : ℕ :=
  let rate_per_minute := initial_charge_percent / initial_time_minutes
  let additional_charge_percent := additional_time_minutes * rate_per_minute
  let total_percent := initial_charge_percent + additional_charge_percent
  if total_percent = 60 then
    initial_time_minutes + additional_time_minutes
  else
    sorry

theorem battery_charge_to_60_percent : battery_charge_time 20 60 120 = 180 :=
by
  -- The formal proof will be provided here.
  sorry

end battery_charge_to_60_percent_l241_241084


namespace part1_part2_l241_241149

theorem part1 (x : ℝ) : -5 * x^2 + 3 * x + 2 > 0 ↔ -2/5 < x ∧ x < 1 :=
by sorry

theorem part2 (a x : ℝ) (h : 0 < a) :
  (a * x^2 + (a + 3) * x + 3 > 0 ↔
    (
      (0 < a ∧ a < 3 ∧ (x < -3/a ∨ x > -1)) ∨
      (a = 3 ∧ x ≠ -1) ∨
      (a > 3 ∧ (x < -1 ∨ x > -3/a))
    )
  ) :=
by sorry

end part1_part2_l241_241149


namespace minimum_k_for_mutual_criticism_l241_241113

theorem minimum_k_for_mutual_criticism (k : ℕ) (h1 : 15 * k > 105) : k ≥ 8 := by
  sorry

end minimum_k_for_mutual_criticism_l241_241113


namespace sufficient_condition_for_solution_l241_241863

theorem sufficient_condition_for_solution 
  (a : ℝ) (f g h : ℝ → ℝ) (h_a : 1 < a)
  (h_fg_h : ∀ x : ℝ, 0 ≤ f x + g x + h x) 
  (h_common_root : ∃ x : ℝ, f x = 0 ∧ g x = 0 ∧ h x = 0) : 
  ∃ x : ℝ, a^(f x) + a^(g x) + a^(h x) = 3 := 
by
  sorry

end sufficient_condition_for_solution_l241_241863


namespace probability_of_reaching_last_floor_l241_241876

noncomputable def probability_of_open_paths (n : ℕ) : ℝ :=
  2^(n-1) / (Nat.choose (2*(n-1)) (n-1))

theorem probability_of_reaching_last_floor (n : ℕ) :
  probability_of_open_paths n = 2^(n-1) / (Nat.choose (2*(n-1)) (n-1)) :=
by
  sorry

end probability_of_reaching_last_floor_l241_241876


namespace jane_total_worth_l241_241801

open Nat

theorem jane_total_worth (q d : ℕ) (h1 : q + d = 30)
  (h2 : 25 * q + 10 * d + 150 = 10 * q + 25 * d) :
  25 * q + 10 * d = 450 :=
by
  sorry

end jane_total_worth_l241_241801


namespace pink_cookies_eq_fifty_l241_241193

-- Define the total number of cookies
def total_cookies : ℕ := 86

-- Define the number of red cookies
def red_cookies : ℕ := 36

-- The property we want to prove
theorem pink_cookies_eq_fifty : (total_cookies - red_cookies = 50) :=
by
  sorry

end pink_cookies_eq_fifty_l241_241193


namespace find_tip_percentage_l241_241222

def original_bill : ℝ := 139.00
def per_person_share : ℝ := 30.58
def number_of_people : ℕ := 5

theorem find_tip_percentage (original_bill : ℝ) (per_person_share : ℝ) (number_of_people : ℕ) 
  (total_paid : ℝ := per_person_share * number_of_people) 
  (tip_amount : ℝ := total_paid - original_bill) : 
  (tip_amount / original_bill) * 100 = 10 :=
by
  sorry

end find_tip_percentage_l241_241222


namespace no_pairs_for_arithmetic_progression_l241_241477

-- Define the problem in Lean
theorem no_pairs_for_arithmetic_progression :
  ¬ ∃ (a b : ℝ), (2 * a = 5 + b) ∧ (2 * b = a * (1 + b)) :=
sorry

end no_pairs_for_arithmetic_progression_l241_241477


namespace base_k_sum_l241_241560

theorem base_k_sum (k : ℕ) (t : ℕ) (h1 : (k + 3) * (k + 4) * (k + 7) = 4 * k^3 + 7 * k^2 + 3 * k + 5)
    (h2 : t = (k + 3) + (k + 4) + (k + 7)) :
    t = 50 := sorry

end base_k_sum_l241_241560


namespace triangle_identity_l241_241663

variables (a b c h_a h_b h_c x y z : ℝ)

-- Define the given conditions
def condition1 := a / h_a = x
def condition2 := b / h_b = y
def condition3 := c / h_c = z

-- Statement of the theorem to be proved
theorem triangle_identity 
  (h1 : condition1 a h_a x) 
  (h2 : condition2 b h_b y) 
  (h3 : condition3 c h_c z) : 
  x^2 + y^2 + z^2 - 2 * x * y - 2 * y * z - 2 * z * x + 4 = 0 := 
  by 
    sorry

end triangle_identity_l241_241663


namespace exists_mutual_shooters_l241_241965

theorem exists_mutual_shooters (n : ℕ) (h : 0 ≤ n) (d : Fin (2 * n + 1) → Fin (2 * n + 1) → ℝ)
  (hdistinct : ∀ i j k l : Fin (2 * n + 1), i ≠ j → k ≠ l → d i j ≠ d k l)
  (hc : ∀ i : Fin (2 * n + 1), ∃ j : Fin (2 * n + 1), i ≠ j ∧ (∀ k : Fin (2 * n + 1), k ≠ j → d i j < d i k)) :
  ∃ i j : Fin (2 * n + 1), i ≠ j ∧
  (∀ k : Fin (2 * n + 1), k ≠ j → d i j < d i k) ∧
  (∀ k : Fin (2 * n + 1), k ≠ i → d j i < d j k) :=
by
  sorry

end exists_mutual_shooters_l241_241965


namespace pizzaCostPerSlice_l241_241373

/-- Define the constants and parameters for the problem --/
def largePizzaCost : ℝ := 10.00
def numberOfSlices : ℕ := 8
def firstToppingCost : ℝ := 2.00
def secondThirdToppingCost : ℝ := 1.00
def otherToppingCost : ℝ := 0.50
def toppings : List String := ["pepperoni", "sausage", "ham", "olives", "mushrooms", "bell peppers", "pineapple"]

/-- Calculate the total number of toppings --/
def numberOfToppings : ℕ := toppings.length

/-- Calculate the total cost of the pizza including all toppings --/
noncomputable def totalPizzaCost : ℝ :=
  largePizzaCost + 
  firstToppingCost + 
  2 * secondThirdToppingCost + 
  (numberOfToppings - 3) * otherToppingCost

/-- Calculate the cost per slice --/
noncomputable def costPerSlice : ℝ := totalPizzaCost / numberOfSlices

/-- Proof statement: The cost per slice is $2.00 --/
theorem pizzaCostPerSlice : costPerSlice = 2 := by
  sorry

end pizzaCostPerSlice_l241_241373


namespace family_boys_girls_l241_241167

theorem family_boys_girls (B G : ℕ) 
  (h1 : B - 1 = G) 
  (h2 : B = 2 * (G - 1)) : 
  B = 4 ∧ G = 3 := 
by {
  sorry
}

end family_boys_girls_l241_241167


namespace distinct_real_numbers_cubed_sum_l241_241187

theorem distinct_real_numbers_cubed_sum (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_eq : ∀ x ∈ {a, b, c}, (x^3 + 12) / x = (a^3 + 12) / a) : 
  a^3 + b^3 + c^3 = -36 :=
by
  sorry

end distinct_real_numbers_cubed_sum_l241_241187


namespace athlete_weight_l241_241811

theorem athlete_weight (a b c : ℤ) (k₁ k₂ k₃ : ℤ)
  (h1 : (a + b + c) / 3 = 42)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 43)
  (h4 : a = 5 * k₁)
  (h5 : b = 5 * k₂)
  (h6 : c = 5 * k₃) :
  b = 40 :=
by
  sorry

end athlete_weight_l241_241811


namespace total_players_must_be_square_l241_241171

variables (k m : ℕ)
def n : ℕ := k + m

theorem total_players_must_be_square (h: (k*(k-1) / 2) + (m*(m-1) / 2) = k * m) :
  ∃ (s : ℕ), n = s^2 :=
by sorry

end total_players_must_be_square_l241_241171


namespace positive_number_equals_seven_l241_241587

theorem positive_number_equals_seven (x : ℝ) (h_pos : x > 0) (h_eq : x - 4 = 21 / x) : x = 7 :=
sorry

end positive_number_equals_seven_l241_241587


namespace find_x_l241_241351

theorem find_x (x : ℝ) (h : x^2 ∈ ({1, 0, x} : Set ℝ)) : x = -1 :=
sorry

end find_x_l241_241351


namespace probability_factor_36_l241_241261

theorem probability_factor_36 : (∃ n : ℕ, n ≤ 36 ∧ ∃ k, 36 = k * n) → (9 / 36 = 1 / 4) :=
begin
  sorry
end

end probability_factor_36_l241_241261


namespace probability_factor_36_l241_241260

theorem probability_factor_36 : (∃ n : ℕ, n ≤ 36 ∧ ∃ k, 36 = k * n) → (9 / 36 = 1 / 4) :=
begin
  sorry
end

end probability_factor_36_l241_241260


namespace range_of_y_function_l241_241582

def range_of_function : Set ℝ :=
  {y : ℝ | ∃ (x : ℝ), x ≠ -2 ∧ y = (x^2 + 5*x + 6)/(x+2)}

theorem range_of_y_function :
  range_of_function = {y : ℝ | y ≠ 1} :=
by
  sorry

end range_of_y_function_l241_241582


namespace maximize_profit_l241_241107

theorem maximize_profit 
  (cost_per_product : ℝ)
  (initial_price : ℝ)
  (initial_sales : ℝ)
  (price_increase_effect : ℝ)
  (daily_sales_decrease : ℝ)
  (max_profit_price : ℝ)
  (max_profit : ℝ)
  :
  cost_per_product = 8 ∧ initial_price = 10 ∧ initial_sales = 100 ∧ price_increase_effect = 1 ∧ daily_sales_decrease = 10 → 
  max_profit_price = 14 ∧
  max_profit = 360 :=
by 
  intro h
  have h_cost := h.1
  have h_initial_price := h.2.1
  have h_initial_sales := h.2.2.1
  have h_price_increase_effect := h.2.2.2.1
  have h_daily_sales_decrease := h.2.2.2.2
  sorry

end maximize_profit_l241_241107


namespace probability_factor_of_36_l241_241269

theorem probability_factor_of_36 :
  let N := 36
  let S := {n : ℕ | n > 0 ∧ n ≤ N}
  let E := {n : ℕ | n ∈ S ∧ N % n = 0}
  (E.card : ℚ) / S.card = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l241_241269


namespace julia_game_difference_l241_241672

theorem julia_game_difference :
  let tag_monday := 28
  let hide_seek_monday := 15
  let tag_tuesday := 33
  let hide_seek_tuesday := 21
  let total_monday := tag_monday + hide_seek_monday
  let total_tuesday := tag_tuesday + hide_seek_tuesday
  let difference := total_tuesday - total_monday
  difference = 11 := by
  sorry

end julia_game_difference_l241_241672


namespace interval_between_segments_systematic_sampling_l241_241232

theorem interval_between_segments_systematic_sampling 
  (total_students : ℕ) (sample_size : ℕ) 
  (h_total_students : total_students = 1000) 
  (h_sample_size : sample_size = 40):
  total_students / sample_size = 25 :=
by
  sorry

end interval_between_segments_systematic_sampling_l241_241232


namespace exists_xy_l241_241673

-- Given conditions from the problem
variables (m x0 y0 : ℕ)
-- Integers x0 and y0 are relatively prime
variables (rel_prim : Nat.gcd x0 y0 = 1)
-- y0 divides x0^2 + m
variables (div_y0 : y0 ∣ x0^2 + m)
-- x0 divides y0^2 + m
variables (div_x0 : x0 ∣ y0^2 + m)

-- Main theorem statement
theorem exists_xy 
  (hm : m > 0) 
  (hx0 : x0 > 0) 
  (hy0 : y0 > 0) 
  (rel_prim : Nat.gcd x0 y0 = 1) 
  (div_y0 : y0 ∣ x0^2 + m) 
  (div_x0 : x0 ∣ y0^2 + m) : 
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ Nat.gcd x y = 1 ∧ y ∣ x^2 + m ∧ x ∣ y^2 + m ∧ x + y ≤ m + 1 := 
sorry

end exists_xy_l241_241673


namespace total_fare_for_20km_l241_241947

def base_fare : ℝ := 8
def fare_per_km_from_3_to_10 : ℝ := 1.5
def fare_per_km_beyond_10 : ℝ := 0.8

def fare_for_first_3km : ℝ := base_fare
def fare_for_3_to_10_km : ℝ := 7 * fare_per_km_from_3_to_10
def fare_for_beyond_10_km : ℝ := 10 * fare_per_km_beyond_10

theorem total_fare_for_20km : fare_for_first_3km + fare_for_3_to_10_km + fare_for_beyond_10_km = 26.5 :=
by
  sorry

end total_fare_for_20km_l241_241947


namespace binomial_coefficient_12_10_l241_241472

theorem binomial_coefficient_12_10 : Nat.choose 12 10 = 66 := by
  sorry  -- The actual proof is omitted as instructed.

end binomial_coefficient_12_10_l241_241472


namespace expected_value_is_minus_one_half_l241_241738

def prob_heads := 1 / 4
def prob_tails := 2 / 4
def prob_edge := 1 / 4
def win_heads := 4
def win_tails := -3
def win_edge := 0

theorem expected_value_is_minus_one_half :
  (prob_heads * win_heads + prob_tails * win_tails + prob_edge * win_edge) = -1 / 2 :=
by
  sorry

end expected_value_is_minus_one_half_l241_241738


namespace find_chemistry_marks_l241_241105

theorem find_chemistry_marks 
    (marks_english : ℕ := 70)
    (marks_math : ℕ := 63)
    (marks_physics : ℕ := 80)
    (marks_biology : ℕ := 65)
    (average_marks : ℚ := 68.2) :
    ∃ (marks_chemistry : ℕ), 
      (marks_english + marks_math + marks_physics + marks_biology + marks_chemistry) = 5 * average_marks 
      → marks_chemistry = 63 :=
by
  sorry

end find_chemistry_marks_l241_241105


namespace pyramid_surface_area_l241_241085

noncomputable def total_surface_area_of_pyramid (a b : ℝ) (theta : ℝ) (height : ℝ) : ℝ :=
  let base_area := a * b * Real.sin theta
  let slant_height := Real.sqrt (height ^ 2 + (a / 2) ^ 2)
  let lateral_area := 4 * (1 / 2 * a * slant_height)
  base_area + lateral_area

theorem pyramid_surface_area :
  total_surface_area_of_pyramid 12 14 (Real.pi / 3) 15 = 168 * Real.sqrt 3 + 216 * Real.sqrt 29 :=
by sorry

end pyramid_surface_area_l241_241085


namespace mrs_hilt_travel_distance_l241_241549

theorem mrs_hilt_travel_distance :
  let distance_water_fountain := 30
  let distance_main_office := 50
  let distance_teacher_lounge := 35
  let trips_water_fountain := 4
  let trips_main_office := 2
  let trips_teacher_lounge := 3
  (distance_water_fountain * trips_water_fountain +
   distance_main_office * trips_main_office +
   distance_teacher_lounge * trips_teacher_lounge) = 325 :=
by
  -- Proof goes here
  sorry

end mrs_hilt_travel_distance_l241_241549


namespace has_two_distinct_real_roots_parabola_equation_l241_241154

open Real

-- Define the quadratic polynomial
def quad_poly (m : ℝ) (x : ℝ) : ℝ := x^2 - 2 * m * x + m^2 - 4

-- Question 1: Prove that the quadratic equation has two distinct real roots
theorem has_two_distinct_real_roots (m : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (quad_poly m x₁ = 0) ∧ (quad_poly m x₂ = 0) := by
  sorry

-- Question 2: Prove the equation of the parabola given certain conditions
theorem parabola_equation (m : ℝ) (hx : quad_poly m 0 = 0) : 
  m = 0 ∧ ∀ x : ℝ, quad_poly m x = x^2 - 4 := by
  sorry

end has_two_distinct_real_roots_parabola_equation_l241_241154


namespace believe_more_blue_l241_241592

-- Define the conditions
def total_people : ℕ := 150
def more_green : ℕ := 90
def both_more_green_and_more_blue : ℕ := 40
def neither : ℕ := 20

-- Theorem statement: Prove that the number of people who believe teal is "more blue" is 80
theorem believe_more_blue : 
  total_people - neither - (more_green - both_more_green_and_more_blue) = 80 :=
by
  sorry

end believe_more_blue_l241_241592


namespace probability_divisor_of_36_l241_241294

theorem probability_divisor_of_36 :
  (∃ (n : ℕ) (h₁ : 1 ≤ n ∧ n ≤ 36), (36 % n = 0) → (1 / 4 : ℝ)) :=
by 
  sorry

end probability_divisor_of_36_l241_241294


namespace expression_evaluation_l241_241752

theorem expression_evaluation : 
  54 + (42 / 14) + (27 * 17) - 200 - (360 / 6) + 2^4 = 272 := by 
  sorry

end expression_evaluation_l241_241752


namespace least_number_to_subtract_l241_241420

theorem least_number_to_subtract (n m : ℕ) (h : n = 56783421) (d : m = 569) : (n % m) = 56783421 % 569 := 
by sorry

end least_number_to_subtract_l241_241420


namespace find_r_l241_241974

theorem find_r (k r : ℝ) (h1 : (5 = k * 3^r)) (h2 : (45 = k * 9^r)) : r = 2 :=
  sorry

end find_r_l241_241974


namespace largest_divisor_of_m_l241_241651

theorem largest_divisor_of_m (m : ℤ) (hm_pos : 0 < m) (h : 33 ∣ m^2) : 33 ∣ m :=
sorry

end largest_divisor_of_m_l241_241651


namespace probability_factor_of_36_l241_241245

def is_factor (d n : ℕ) : Prop := n % d = 0

theorem probability_factor_of_36 : 
  (∑ i in finset.range (36+1), if is_factor i 36 then (1:ℚ) else 0) / 36 = 1 / 4 := by
sorry

end probability_factor_of_36_l241_241245


namespace problem_divisible_by_64_l241_241912

theorem problem_divisible_by_64 (n : ℕ) (hn : n > 0) : (3 ^ (2 * n + 2) - 8 * n - 9) % 64 = 0 := 
by
  sorry

end problem_divisible_by_64_l241_241912


namespace fraction_value_l241_241520

theorem fraction_value (p q x : ℚ) (h₁ : p / q = 4 / 5) (h₂ : 2 * q + p ≠ 0) (h₃ : 2 * q - p ≠ 0) :
  x + (2 * q - p) / (2 * q + p) = 2 → x = 11 / 7 :=
by
  sorry

end fraction_value_l241_241520


namespace smallest_sum_proof_l241_241844

theorem smallest_sum_proof (N : ℕ) (p : ℝ) (h1 : 6 * N = 2022) (hp : p > 0) : (N * 1 = 337) :=
by 
  have hN : N = 2022 / 6 := by 
    sorry
  exact hN

end smallest_sum_proof_l241_241844


namespace solution_inequality_part1_solution_inequality_part2_l241_241007

noncomputable def f (x : ℝ) : ℝ := abs (x - 2)
noncomputable def g (x m : ℝ) : ℝ := -abs (x + 7) + 3 * m

theorem solution_inequality_part1 (x : ℝ) :
  (f x + x^2 - 4 > 0) ↔ (x > 2 ∨ x < -1) :=
sorry

theorem solution_inequality_part2 (m : ℝ) :
  (∃ x : ℝ, f x < g x m) → (m > 3) :=
sorry

end solution_inequality_part1_solution_inequality_part2_l241_241007


namespace range_a_l241_241478

theorem range_a (a : ℝ) :
  (∀ x : ℝ, a ≤ x ∧ x ≤ 3 → -1 ≤ -x^2 + 2 * x + 2 ∧ -x^2 + 2 * x + 2 ≤ 3) →
  -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_a_l241_241478


namespace solve_system_of_equations_l241_241203

def system_solution : Prop := ∃ x y : ℚ, 4 * x - 6 * y = -14 ∧ 8 * x + 3 * y = -15 ∧ x = -11 / 5 ∧ y = 2.6 / 3

theorem solve_system_of_equations : system_solution := sorry

end solve_system_of_equations_l241_241203


namespace area_of_inscribed_rectangle_l241_241082

open Real

theorem area_of_inscribed_rectangle (r l w : ℝ) (h_radius : r = 7) 
  (h_ratio : l / w = 3) (h_width : w = 2 * r) : l * w = 588 :=
by
  sorry

end area_of_inscribed_rectangle_l241_241082


namespace exists_small_factorable_subsequence_l241_241814

def small_factorable (n : ℕ) : Prop := ∃ (a b c d : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1 ∧ n = a * b * c * d

theorem exists_small_factorable_subsequence :
  ∃ (xs : ℕ), (100 ≤ xs ∧ xs ≤ 999) → 
  ∃ (ys : ℕ), (100 ≤ ys ∧ ys ≤ 999) → 
  ∃ subseq : ℕ, 0 ≤ subseq < 10^6 ∧ small_factorable subseq := 
sorry

end exists_small_factorable_subsequence_l241_241814


namespace zero_point_interval_l241_241837

noncomputable def f (x : ℝ) : ℝ := (4 / x) - (2^x)

theorem zero_point_interval : ∃ x : ℝ, (1 < x ∧ x < 1.5) ∧ f x = 0 :=
sorry

end zero_point_interval_l241_241837


namespace infinitely_many_n_l241_241736

theorem infinitely_many_n (h : ℤ) : ∃ (S : Set ℤ), S ≠ ∅ ∧ ∀ n ∈ S, ∃ k : ℕ, ⌊n * Real.sqrt (h^2 + 1)⌋ = k^2 :=
by
  sorry

end infinitely_many_n_l241_241736


namespace maria_total_earnings_l241_241385

-- Definitions of the conditions
def day1_tulips := 30
def day1_roses := 20
def day2_tulips := 2 * day1_tulips
def day2_roses := 2 * day1_roses
def day3_tulips := day2_tulips / 10
def day3_roses := 16
def tulip_price := 2
def rose_price := 3

-- Definition of the total earnings calculation
noncomputable def total_earnings : ℤ :=
  let total_tulips := day1_tulips + day2_tulips + day3_tulips
  let total_roses := day1_roses + day2_roses + day3_roses
  (total_tulips * tulip_price) + (total_roses * rose_price)

-- The proof statement
theorem maria_total_earnings : total_earnings = 420 := by
  sorry

end maria_total_earnings_l241_241385


namespace find_radius_l241_241997

noncomputable def radius_from_tangent_circles (AB : ℝ) (r : ℝ) : ℝ :=
  let O1O2 := 2 * r
  let proportion := AB / O1O2
  r + r * proportion

theorem find_radius
  (AB : ℝ) (r : ℝ)
  (hAB : AB = 11) (hr : r = 5) :
  radius_from_tangent_circles AB r = 55 :=
by
  sorry

end find_radius_l241_241997


namespace max_digit_sum_of_watch_display_l241_241742

-- Define the problem conditions
def valid_hour (h : ℕ) : Prop := 0 ≤ h ∧ h < 24
def valid_minute (m : ℕ) : Prop := 0 ≤ m ∧ m < 60
def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

-- Define the proof problem
theorem max_digit_sum_of_watch_display : 
  ∃ h m : ℕ, valid_hour h ∧ valid_minute m ∧ (digit_sum h + digit_sum m = 24) :=
sorry

end max_digit_sum_of_watch_display_l241_241742


namespace variance_of_yield_l241_241596

/-- Given a data set representing annual average yields,
    prove that the variance of this data set is approximately 171. --/
theorem variance_of_yield {yields : List ℝ} 
  (h_yields : yields = [450, 430, 460, 440, 450, 440, 470, 460]) :
  let mean := (yields.sum / yields.length : ℝ)
  let squared_diffs := (yields.map (fun x => (x - mean)^2))
  let variance := (squared_diffs.sum / (yields.length - 1 : ℝ))
  abs (variance - 171) < 1 :=
by
  sorry

end variance_of_yield_l241_241596


namespace mrs_sheridan_cats_l241_241550

theorem mrs_sheridan_cats (initial_cats : ℝ) (given_away_cats : ℝ) (remaining_cats : ℝ) :
  initial_cats = 17.0 → given_away_cats = 14.0 → remaining_cats = (initial_cats - given_away_cats) → remaining_cats = 3.0 :=
by
  intros
  sorry

end mrs_sheridan_cats_l241_241550


namespace max_sum_two_digit_primes_l241_241603

theorem max_sum_two_digit_primes : (89 + 97) = 186 := 
by
  sorry

end max_sum_two_digit_primes_l241_241603


namespace coins_difference_l241_241619

theorem coins_difference (p n d : ℕ) (h1 : p + n + d = 3030)
  (h2 : 1 ≤ p) (h3 : 1 ≤ n) (h4 : 1 ≤ d) (h5 : p ≤ 3029) (h6 : n ≤ 3029) (h7 : d ≤ 3029) :
  (max (p + 5 * n + 10 * d) (max (p + 5 * n + 10 * (3030 - p - n)) (3030 - n - d + 5 * d + 10 * p))) - 
  (min (p + 5 * n + 10 * d) (min (p + 5 * n + 10 * (3030 - p - n)) (3030 - n - d + 5 * d + 10 * p))) = 27243 := 
sorry

end coins_difference_l241_241619


namespace value_at_2013_l241_241143

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f x = -f (-x)
axiom periodic_5 : ∀ x : ℝ, f (x + 5) ≥ f x
axiom periodic_1 : ∀ x : ℝ, f (x + 1) ≤ f x

theorem value_at_2013 : f 2013 = 0 :=
by
  -- Proof goes here
  sorry

end value_at_2013_l241_241143


namespace friends_division_ways_l241_241514

theorem friends_division_ways : (4 ^ 8 = 65536) :=
by
  sorry

end friends_division_ways_l241_241514


namespace vector_coordinates_l241_241784

-- Define the given vectors.
def a : (ℝ × ℝ) := (1, 1)
def b : (ℝ × ℝ) := (1, -1)

-- Define the proof goal.
theorem vector_coordinates :
  -2 • a - b = (-3, -1) :=
by
  sorry -- Proof not required.

end vector_coordinates_l241_241784


namespace jerry_remaining_debt_l241_241666

variable (two_months_ago_payment last_month_payment total_debt : ℕ)

def remaining_debt : ℕ := total_debt - (two_months_ago_payment + last_month_payment)

theorem jerry_remaining_debt :
  two_months_ago_payment = 12 →
  last_month_payment = 12 + 3 →
  total_debt = 50 →
  remaining_debt two_months_ago_payment last_month_payment total_debt = 23 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end jerry_remaining_debt_l241_241666


namespace perpendicular_lines_l241_241653

theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, 2 * x - a * y - 1 = 0 → x + 2 * y = 0) →
  (a = 1) :=
by
  sorry

end perpendicular_lines_l241_241653


namespace unique_solution_for_a_l241_241003

theorem unique_solution_for_a (a : ℝ) :
  (∃! x : ℝ, 2 ^ |2 * x - 2| - a * Real.cos (1 - x) = 0) ↔ a = 1 :=
sorry

end unique_solution_for_a_l241_241003


namespace total_coffee_needed_l241_241009

-- Conditions as definitions
def weak_coffee_amount_per_cup : ℕ := 1
def strong_coffee_amount_per_cup : ℕ := 2 * weak_coffee_amount_per_cup
def cups_of_weak_coffee : ℕ := 12
def cups_of_strong_coffee : ℕ := 12

-- Prove that the total amount of coffee needed equals 36 tablespoons
theorem total_coffee_needed : (weak_coffee_amount_per_cup * cups_of_weak_coffee) + (strong_coffee_amount_per_cup * cups_of_strong_coffee) = 36 :=
by
  sorry

end total_coffee_needed_l241_241009


namespace probability_factor_of_36_l241_241277

theorem probability_factor_of_36 : 
  let pos_ints := { n : ℕ | 1 ≤ n ∧ n ≤ 36 }
  let num_divisors := { d : ℕ | d ∣ 36 }
  (num_divisors.to_finset.card : ℚ) / (pos_ints.to_finset.card : ℚ) = 1 / 4 :=
by {
  sorry
}

end probability_factor_of_36_l241_241277


namespace probability_open_path_l241_241868

-- Define necessary terms
def total_doors (n : ℕ) : ℕ := 2 * (n - 1)
def locked_doors (n : ℕ) : ℕ := total_doors n / 2

-- Helper function to compute binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Probability theorem
theorem probability_open_path (n : ℕ) (h : n > 1) : 
  ((locked_doors n) = (n-1)) → 
  (∃ p, p = (2^(n-1)) / (binom (total_doors n) (n-1))) :=
by {
  intro h1,
  use ((2^(n-1)) / (binom (total_doors n) (n-1))),
  sorry
}

end probability_open_path_l241_241868


namespace debt_amount_is_40_l241_241207

theorem debt_amount_is_40 (l n t debt remaining : ℕ) (h_l : l = 6)
  (h_n1 : n = 5 * l) (h_n2 : n = 3 * t) (h_remaining : remaining = 6) 
  (h_share : ∀ x y z : ℕ, x = y ∧ y = z ∧ z = 2) :
  debt = 40 := 
by
  sorry

end debt_amount_is_40_l241_241207


namespace find_r_l241_241674

theorem find_r (a b m p r : ℚ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a * b = 4)
  (h4 : ∀ x : ℚ, x^2 - m * x + 4 = (x - a) * (x - b)) :
  (a - 1 / b) * (b - 1 / a) = 9 / 4 := by
  sorry

end find_r_l241_241674


namespace find_original_numbers_l241_241407

theorem find_original_numbers (x y : ℕ) (hx : x + y = 2022) 
  (hy : (x - 5) / 10 + 10 * y + 1 = 2252) : x = 1815 ∧ y = 207 :=
by sorry

end find_original_numbers_l241_241407


namespace average_production_last_5_days_l241_241304

theorem average_production_last_5_days
  (avg_first_25_days : ℕ → ℕ → ℕ → ℕ → Prop)
  (avg_monthly : ℕ)
  (total_days : ℕ)
  (days_first_period : ℕ)
  (avg_production_first_period : ℕ)
  (avg_total_monthly : ℕ)
  (days_second_period : ℕ)
  (total_production_five_days : ℕ):
  (days_first_period = 25) →
  (avg_production_first_period = 50) →
  (avg_total_monthly = 48) →
  (total_production_five_days = 190) →
  (days_second_period = 5) →
  avg_first_25_days days_first_period avg_production_first_period 
  (days_first_period * avg_production_first_period) avg_total_monthly ∧
  avg_monthly = avg_total_monthly →
  ((days_first_period + days_second_period) * avg_monthly - 
  days_first_period * avg_production_first_period = total_production_five_days) →
  (total_production_five_days / days_second_period = 38) := sorry

end average_production_last_5_days_l241_241304


namespace Jeff_total_ounces_of_peanut_butter_l241_241023

theorem Jeff_total_ounces_of_peanut_butter
    (jars : ℕ)
    (equal_count : ℕ)
    (total_jars : jars = 9)
    (j16 : equal_count = 3) 
    (j28 : equal_count = 3)
    (j40 : equal_count = 3) :
    (3 * 16 + 3 * 28 + 3 * 40 = 252) :=
by
  sorry

end Jeff_total_ounces_of_peanut_butter_l241_241023


namespace compute_g_neg_101_l241_241975

noncomputable def g (x : ℝ) : ℝ := sorry

theorem compute_g_neg_101 (g_condition : ∀ x y : ℝ, g (x * y) + x = x * g y + g x)
                         (g1 : g 1 = 7) :
    g (-101) = -95 := 
by 
  sorry

end compute_g_neg_101_l241_241975


namespace handshake_count_l241_241229

theorem handshake_count :
  let total_people := 5 * 4
  let handshakes_per_person := total_people - 1 - 3
  let total_handshakes_with_double_count := total_people * handshakes_per_person
  let total_handshakes := total_handshakes_with_double_count / 2
  total_handshakes = 160 :=
by
-- We include "sorry" to indicate that the proof is not provided.
sorry

end handshake_count_l241_241229


namespace angle_B_is_pi_over_3_range_of_expression_l241_241655

variable {A B C a b c : ℝ}

-- Conditions
def sides_opposite_angles (A B C : ℝ) (a b c : ℝ): Prop :=
  (2 * c - a) * Real.cos B - b * Real.cos A = 0

-- Part 1: Prove B = π/3
theorem angle_B_is_pi_over_3 (h : sides_opposite_angles A B C a b c) : 
    B = Real.pi / 3 := 
  sorry

-- Part 2: Prove the range of sqrt(3) * (sin A + sin(C - π/6)) is (1, 2]
theorem range_of_expression (h : 0 < A ∧ A < 2 * Real.pi / 3) : 
    (1:ℝ) < Real.sqrt 3 * (Real.sin A + Real.sin (C - Real.pi / 6)) 
    ∧ Real.sqrt 3 * (Real.sin A + Real.sin (C - Real.pi / 6)) ≤ 2 := 
  sorry

end angle_B_is_pi_over_3_range_of_expression_l241_241655


namespace posters_count_l241_241665

-- Define the regular price per poster
def regular_price : ℕ := 4

-- Jeremy can buy 24 posters at regular price
def posters_at_regular_price : ℕ := 24

-- Total money Jeremy has is equal to the money needed to buy 24 posters
def total_money : ℕ := posters_at_regular_price * regular_price

-- The special deal: buy one get the second at half price
def cost_of_two_posters : ℕ := regular_price + regular_price / 2

-- Number of pairs Jeremy can buy with his total money
def number_of_pairs : ℕ := total_money / cost_of_two_posters

-- Total number of posters Jeremy can buy under the sale
def total_posters := number_of_pairs * 2

-- Prove that the total posters is 32
theorem posters_count : total_posters = 32 := by
  sorry

end posters_count_l241_241665


namespace only_odd_integer_option_l241_241725

theorem only_odd_integer_option : 
  (6 ^ 2 = 36 ∧ Even 36) ∧ 
  (23 - 17 = 6 ∧ Even 6) ∧ 
  (9 * 24 = 216 ∧ Even 216) ∧ 
  (96 / 8 = 12 ∧ Even 12) ∧ 
  (9 * 41 = 369 ∧ Odd 369)
:= by
  sorry

end only_odd_integer_option_l241_241725


namespace find_n_l241_241911

theorem find_n (n : ℕ) : (1 / (n + 1 : ℝ) + 2 / (n + 1 : ℝ) + (n + 1) / (n + 1 : ℝ) = 2) → (n = 2) :=
by
  sorry

end find_n_l241_241911


namespace probability_factor_of_36_l241_241242

def is_factor (d n : ℕ) : Prop := n % d = 0

theorem probability_factor_of_36 : 
  (∑ i in finset.range (36+1), if is_factor i 36 then (1:ℚ) else 0) / 36 = 1 / 4 := by
sorry

end probability_factor_of_36_l241_241242


namespace fundamental_events_in_A_expected_value_ξ_l241_241533

noncomputable def solution_set := {x : ℝ | x^2 - x - 6 ≤ 0}

theorem fundamental_events_in_A :
  {p : ℤ × ℤ | p.1 ∈ {m : ℤ | -2 ≤ m ∧ m ≤ 3} ∧ p.2 ∈ {n : ℤ | -2 ≤ n ∧ n ≤ 3} ∧ p.1 + p.2 = 0} =
  {(-2, 2), (-1, 1), (0, 0), (1, -1), (2, -2)} := sorry

noncomputable def ξ_distribution := 
  {0, 1, 4, 9} →ᵣ (λ x : ℝ, 
    if x = 0 then (1 / 6 : ℝ)
    else if x = 1 then (1 / 3 : ℝ)
    else if x = 4 then (1 / 3 : ℝ)
    else if x = 9 then (1 / 6 : ℝ)
    else 0)

theorem expected_value_ξ : 
  ∑ x in {0, 1, 4, 9}, x * ξ_distribution x = (19 / 6 : ℝ) := sorry

end fundamental_events_in_A_expected_value_ξ_l241_241533


namespace geometric_series_first_term_l241_241986

theorem geometric_series_first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 30)
  (h2 : a^2 / (1 - r^2) = 90)
  (hrange : |r| < 1) :
  a = 60 / 11 :=
by 
  sorry

end geometric_series_first_term_l241_241986


namespace exists_pos_integer_n_l241_241358

theorem exists_pos_integer_n (n : ℕ) (hn_pos : n > 0) (h : ∃ m : ℕ, m * m = 1575 * n) : n = 7 :=
sorry

end exists_pos_integer_n_l241_241358


namespace probability_factor_of_36_is_1_over_4_l241_241273

def totalPositiveIntegers := 36

def isDivisor (n: Nat) : Prop := ∃ (a b: Nat), (0 ≤ a ∧ a ≤ 2) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (n = 2^a * 3^b)

def numDivisors := 9
def probability := (numDivisors : ℚ) / (totalPositiveIntegers : ℚ)

theorem probability_factor_of_36_is_1_over_4 :
  probability = (1 / 4 : ℚ) :=
sorry

end probability_factor_of_36_is_1_over_4_l241_241273


namespace girls_in_class_l241_241705

theorem girls_in_class (total_students boys_ratio girls_ratio : ℕ) (h_total : total_students = 260)
  (h_ratio_boys : boys_ratio = 5) (h_ratio_girls : girls_ratio = 8) : 
  let total_ratio := boys_ratio + girls_ratio in
  let boys_fraction := boys_ratio / total_ratio in
  let boys := total_students * boys_fraction in
  let girls := total_students - boys in
  girls = 160 := 
by 
  sorry

end girls_in_class_l241_241705


namespace lucille_house_difference_l241_241817

def height_lucille : ℕ := 80
def height_neighbor1 : ℕ := 70
def height_neighbor2 : ℕ := 99

def average_height (h1 h2 h3 : ℕ) : ℕ := (h1 + h2 + h3) / 3

def difference (h_average h_actual : ℕ) : ℕ := h_average - h_actual

theorem lucille_house_difference :
  difference (average_height height_lucille height_neighbor1 height_neighbor2) height_lucille = 3 :=
by
  unfold difference
  unfold average_height
  sorry

end lucille_house_difference_l241_241817


namespace Mary_work_days_l241_241679

theorem Mary_work_days :
  ∀ (M : ℝ), (∀ R : ℝ, R = M / 1.30) → (R = 20) → M = 26 :=
by
  intros M h1 h2
  sorry

end Mary_work_days_l241_241679


namespace two_person_subcommittees_from_eight_l241_241508

theorem two_person_subcommittees_from_eight :
  (Nat.choose 8 2) = 28 :=
by
  sorry

end two_person_subcommittees_from_eight_l241_241508


namespace solve_inequality_l241_241702

-- Define the odd and monotonically decreasing function
noncomputable def f : ℝ → ℝ := sorry

-- Assume the given conditions
axiom odd_f : ∀ x, f (-x) = -f x
axiom decreasing_f : ∀ x y, x < y → y < 0 → f x > f y
axiom f_at_2 : f 2 = 0

-- The proof statement
theorem solve_inequality (x : ℝ) : (x - 1) * f (x + 1) > 0 ↔ -3 < x ∧ x < -1 :=
by
  -- Proof omitted
  sorry

end solve_inequality_l241_241702


namespace probability_factor_of_36_l241_241246

theorem probability_factor_of_36 :
  (∃ n ∈ finset.range 37, (∃ k : ℕ, 36 = k * n)) → 9 / 36 = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l241_241246


namespace find_x_l241_241158

open Real

theorem find_x 
  (x y : ℝ) 
  (hx_pos : 0 < x)
  (hy_pos : 0 < y) 
  (h_eq : 7 * x^2 + 21 * x * y = 2 * x^3 + 3 * x^2 * y) 
  : x = 7 := 
sorry

end find_x_l241_241158


namespace calc_sub_neg_eq_add_problem_0_sub_neg_3_l241_241323

theorem calc_sub_neg_eq_add (a b : Int) : a - (-b) = a + b := by
  sorry

theorem problem_0_sub_neg_3 : 0 - (-3) = 3 := by
  exact calc_sub_neg_eq_add 0 3

end calc_sub_neg_eq_add_problem_0_sub_neg_3_l241_241323


namespace lucille_house_difference_l241_241815

-- Define the heights of the houses as given in the conditions.
def height_lucille : ℕ := 80
def height_neighbor1 : ℕ := 70
def height_neighbor2 : ℕ := 99

-- Define the total height of the houses.
def total_height : ℕ := height_neighbor1 + height_lucille + height_neighbor2

-- Define the average height of the houses.
def average_height : ℕ := total_height / 3

-- Define the height difference between Lucille's house and the average height.
def height_difference : ℕ := average_height - height_lucille

-- The theorem to prove.
theorem lucille_house_difference :
  height_difference = 3 := by
  sorry

end lucille_house_difference_l241_241815


namespace evaluate_expression_l241_241621

theorem evaluate_expression : 1273 + 120 / 60 - 173 = 1102 := by
  sorry

end evaluate_expression_l241_241621


namespace weight_of_replaced_person_l241_241948

theorem weight_of_replaced_person (avg_weight : ℝ) (new_person_weight : ℝ)
  (h1 : new_person_weight = 65)
  (h2 : ∀ (initial_avg_weight : ℝ), 8 * (initial_avg_weight + 2.5) - 8 * initial_avg_weight = new_person_weight - avg_weight) :
  avg_weight = 45 := 
by
  -- Proof goes here
  sorry

end weight_of_replaced_person_l241_241948


namespace binomial_12_10_eq_66_l241_241456

theorem binomial_12_10_eq_66 : binomial 12 10 = 66 :=
by
  sorry

end binomial_12_10_eq_66_l241_241456


namespace geometric_series_sum_l241_241751

theorem geometric_series_sum :
  let a := -1
  let r := -3
  let n := 8
  let S := (a * (r ^ n - 1)) / (r - 1)
  S = 1640 :=
by 
  sorry 

end geometric_series_sum_l241_241751


namespace problem1_problem2_problem3_l241_241491

-- Definitions and conditions for Problem 1
def symmetric (l : List ℤ) := l = l.reverse

def is_arithmetic (l : List ℤ) := 
∃ d, ∀ i < l.length - 1, l.nth_le i (by linarith) + d = l.nth_le (i + 1) (by linarith)

def b_seq := [2, 5, 8, 11, 8, 5, 2]
def b_seq_question : Prop := 
symmetric b_seq ∧ is_arithmetic b_seq.take 4 ∧ b_seq.head = 2 ∧ b_seq.nth_le 3 (by linarith) = 11

theorem problem1 : b_seq_question → b_seq = [2, 5, 8, 11, 8, 5, 2] := sorry

-- Definitions and conditions for Problem 2
def c_seq (k : ℕ) (n : ℕ) := 
if n < k then c_seq (k - n + 1) else 50 - 4 * (n - k)

def sum_seq (c : ℕ → ℤ) (n : ℕ) := 
∑ i in range n, c i

def S (k : ℕ) := 
sum_seq (c_seq k) (2 * k - 1)

theorem problem2 : (∀ k ≥ 1, symmetric (List.map (c_seq k) (List.range (2 * k - 1))) → 
∀ k ≥ 3, S k ≤ 626 → S k = 626 → k = 13 := sorry

-- Definitions and conditions for Problem 3
def sequence_m (m : ℕ) : List ℤ :=
let l := List.range m
l.concat1 (List.range' 1 m ++ List.reverse (List.range' 1 (m-1)))

def T (n : ℕ) := 
if n = 2008 then sequence_m n else []

def S2008 (m : ℕ) := sum_list (T (m))

theorem problem3 : ∀ m > 1500, one_possible_sum_of_first_2008_terms (sequence_m m) = 2 ^ 2008 - 1 := sorry

end problem1_problem2_problem3_l241_241491


namespace average_score_of_class_l241_241884

variable (students_total : ℕ) (group1_students : ℕ) (group2_students : ℕ)
variable (group1_avg : ℝ) (group2_avg : ℝ)

theorem average_score_of_class :
  students_total = 20 → 
  group1_students = 10 → 
  group2_students = 10 → 
  group1_avg = 80 → 
  group2_avg = 60 → 
  (group1_students * group1_avg + group2_students * group2_avg) / students_total = 70 := 
by
  intros students_total_eq group1_students_eq group2_students_eq group1_avg_eq group2_avg_eq
  rw [students_total_eq, group1_students_eq, group2_students_eq, group1_avg_eq, group2_avg_eq]
  simp
  sorry

end average_score_of_class_l241_241884


namespace expected_value_correct_l241_241487

-- Define the problem conditions
def num_balls : ℕ := 5

def prob_swapped_twice : ℚ := (2 / 25)
def prob_never_swapped : ℚ := (9 / 25)
def prob_original_position : ℚ := prob_swapped_twice + prob_never_swapped

-- Define the expected value calculation
def expected_num_in_original_position : ℚ :=
  num_balls * prob_original_position

-- Claim: The expected number of balls that occupy their original positions after two successive transpositions is 2.2.
theorem expected_value_correct :
  expected_num_in_original_position = 2.2 :=
sorry

end expected_value_correct_l241_241487


namespace xiao_ming_incorrect_l241_241421

theorem xiao_ming_incorrect : ∃ (a b : ℚ), a > 0 ∧ b < 0 ∧ a > b ∧ (1/a) > (1/b) :=
by
  use (1 : ℚ),
  use (-1 : ℚ),
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  norm_num

end xiao_ming_incorrect_l241_241421


namespace cost_price_of_bicycle_l241_241730

variables {CP_A SP_AB SP_BC : ℝ}

theorem cost_price_of_bicycle (h1 : SP_AB = CP_A * 1.2)
                             (h2 : SP_BC = SP_AB * 1.25)
                             (h3 : SP_BC = 225) :
                             CP_A = 150 :=
by sorry

end cost_price_of_bicycle_l241_241730


namespace mike_profit_l241_241546

theorem mike_profit 
  (num_acres_bought : ℕ) (price_per_acre_buy : ℤ) 
  (fraction_sold : ℚ) (price_per_acre_sell : ℤ) :
  num_acres_bought = 200 →
  price_per_acre_buy = 70 →
  fraction_sold = 1/2 →
  price_per_acre_sell = 200 →
  let cost_of_land := price_per_acre_buy * num_acres_bought,
      num_acres_sold := (fraction_sold * num_acres_bought),
      revenue_from_sale := price_per_acre_sell * num_acres_sold,
      profit := revenue_from_sale - cost_of_land
  in profit = 6000 := by
  intros h1 h2 h3 h4
  let cost_of_land := price_per_acre_buy * num_acres_bought
  let num_acres_sold := (fraction_sold * num_acres_bought)
  let revenue_from_sale := price_per_acre_sell * num_acres_sold
  let profit := revenue_from_sale - cost_of_land
  rw [h1, h2, h3, h4]
  sorry

end mike_profit_l241_241546


namespace geometric_sequence_common_ratio_l241_241178

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * q)
  (h_a1 : a 1 = 1/2) 
  (h_a4 : a 4 = -4) : 
  q = -2 := 
by
  sorry

end geometric_sequence_common_ratio_l241_241178


namespace simplified_expression_evaluation_l241_241686

-- Problem and conditions
def x := Real.sqrt 5 - 1

-- Statement of the proof problem
theorem simplified_expression_evaluation : 
  ( (x / (x - 1) - 1) / (x^2 - 1) / (x^2 - 2 * x + 1) ) = Real.sqrt 5 / 5 :=
sorry

end simplified_expression_evaluation_l241_241686


namespace opposite_numbers_abs_eq_l241_241426

theorem opposite_numbers_abs_eq (a : ℚ) : abs a = abs (-a) :=
by
  sorry

end opposite_numbers_abs_eq_l241_241426


namespace max_marks_l241_241715

theorem max_marks (M : ℝ) (h : 0.80 * M = 240) : M = 300 :=
sorry

end max_marks_l241_241715


namespace instrument_failure_probability_l241_241170

noncomputable def probability_of_instrument_not_working (m : ℕ) (P : ℝ) : ℝ :=
  1 - (1 - P)^m

theorem instrument_failure_probability (m : ℕ) (P : ℝ) :
  0 ≤ P → P ≤ 1 → probability_of_instrument_not_working m P = 1 - (1 - P)^m :=
by
  intros _ _
  sorry

end instrument_failure_probability_l241_241170


namespace red_pieces_count_l241_241410

-- Define the conditions
def total_pieces : ℕ := 3409
def blue_pieces : ℕ := 3264

-- Prove the number of red pieces
theorem red_pieces_count : total_pieces - blue_pieces = 145 :=
by sorry

end red_pieces_count_l241_241410


namespace solve_for_y_l241_241130

theorem solve_for_y (x y : ℝ) (h : 2 * x - y = 6) : y = 2 * x - 6 :=
by
  sorry

end solve_for_y_l241_241130


namespace all_equal_l241_241150

theorem all_equal (a : Fin 100 → ℝ) 
  (h1 : a 0 - 3 * a 1 + 2 * a 2 ≥ 0)
  (h2 : a 1 - 3 * a 2 + 2 * a 3 ≥ 0)
  (h3 : a 2 - 3 * a 3 + 2 * a 4 ≥ 0)
  -- ...
  (h99: a 98 - 3 * a 99 + 2 * a 0 ≥ 0)
  (h100: a 99 - 3 * a 0 + 2 * a 1 ≥ 0) : 
    ∀ i : Fin 100, a i = a 0 := 
by 
  sorry

end all_equal_l241_241150


namespace smallest_possible_sum_l241_241847

theorem smallest_possible_sum (N : ℕ) (p : ℚ) (h : p > 0) (hsum : 6 * N = 2022) : 
  ∃ (N : ℕ), N * 1 = 337 :=
by 
  use 337
  sorry

end smallest_possible_sum_l241_241847


namespace arithmetic_seq_third_term_l241_241165

theorem arithmetic_seq_third_term
  (a d : ℝ)
  (h : a + (a + 2 * d) = 10) :
  a + d = 5 := by
  sorry

end arithmetic_seq_third_term_l241_241165


namespace new_quadratic_coeff_l241_241785

theorem new_quadratic_coeff (r s p q : ℚ) 
  (h1 : 3 * r^2 + 4 * r + 2 = 0)
  (h2 : 3 * s^2 + 4 * s + 2 = 0)
  (h3 : r + s = -4 / 3)
  (h4 : r * s = 2 / 3) 
  (h5 : r^3 + s^3 = - p) :
  p = 16 / 27 :=
by
  sorry

end new_quadratic_coeff_l241_241785


namespace supercomputer_transformation_stops_l241_241389

def transformation_rule (n : ℕ) : ℕ :=
  let A : ℕ := n / 100
  let B : ℕ := n % 100
  2 * A + 8 * B

theorem supercomputer_transformation_stops (n : ℕ) :
  let start := (10^900 - 1) / 9 -- 111...111 with 900 ones
  (n = start) → (∀ m, transformation_rule m < 100 → false) :=
by
  sorry

end supercomputer_transformation_stops_l241_241389


namespace find_m_if_polynomial_is_square_l241_241013

theorem find_m_if_polynomial_is_square (m : ℝ) :
  (∀ x, ∃ k : ℝ, x^2 + 2 * (m - 3) * x + 16 = (x + k)^2) ↔ (m = 7 ∨ m = -1) :=
by
  sorry

end find_m_if_polynomial_is_square_l241_241013


namespace infinite_possible_matrices_A_squared_l241_241028

theorem infinite_possible_matrices_A_squared (A : Matrix (Fin 3) (Fin 3) ℝ) (hA : A^4 = 0) :
  ∃ (S : Set (Matrix (Fin 3) (Fin 3) ℝ)), (∀ B ∈ S, B = A^2) ∧ S.Infinite :=
sorry

end infinite_possible_matrices_A_squared_l241_241028


namespace January_to_November_ratio_l241_241309

variable (N D J : ℝ)

-- Condition 1: November revenue is 3/5 of December revenue
axiom revenue_Nov : N = (3 / 5) * D

-- Condition 2: December revenue is 2.5 times the average of November and January revenues
axiom revenue_Dec : D = 2.5 * (N + J) / 2

-- Goal: Prove the ratio of January revenue to November revenue is 1/3
theorem January_to_November_ratio : J / N = 1 / 3 :=
by
  -- We will use the given axioms to derive the proof
  sorry

end January_to_November_ratio_l241_241309


namespace simplify_and_evaluate_expr_l241_241689

namespace SimplificationProof

variable (x : ℝ)

theorem simplify_and_evaluate_expr (h : x = Real.sqrt 5 - 1) :
  ((x / (x - 1) - 1) / ((x ^ 2 - 1) / (x ^ 2 - 2 * x + 1))) = Real.sqrt 5 / 5 :=
by
  sorry

end SimplificationProof

end simplify_and_evaluate_expr_l241_241689


namespace expected_value_N_given_S_2_l241_241301

noncomputable def expected_value_given_slip_two (P_N_n : ℕ → ℝ) : ℝ :=
  ∑' n, n * (2^(-n) / (n * (Real.log 2 - 0.5)))

theorem expected_value_N_given_S_2 (P_N_n : ℕ → ℝ) (h : ∀ n, P_N_n n = 2^(-n)) :
  expected_value_given_slip_two P_N_n = 1 / (2 * Real.log 2 - 1) :=
by
  sorry

end expected_value_N_given_S_2_l241_241301


namespace sum_of_three_consecutive_cubes_divisible_by_9_l241_241395

theorem sum_of_three_consecutive_cubes_divisible_by_9 (n : ℕ) : 
  (n^3 + (n + 1)^3 + (n + 2)^3) % 9 = 0 := 
by
  sorry

end sum_of_three_consecutive_cubes_divisible_by_9_l241_241395


namespace value_of_a_l241_241497

noncomputable def M : Set ℝ := {x | x^2 = 2}
noncomputable def N (a : ℝ) : Set ℝ := {x | a*x = 1}

theorem value_of_a (a : ℝ) : N a ⊆ M → a = 0 ∨ a = -Real.sqrt 2 / 2 ∨ a = Real.sqrt 2 / 2 :=
by
  intro h
  sorry

end value_of_a_l241_241497


namespace continuous_function_solution_l241_241116

theorem continuous_function_solution (f : ℝ → ℝ) (a : ℝ) (h_continuous : Continuous f) (h_pos : 0 < a)
    (h_equation : ∀ x, f x = a^x * f (x / 2)) :
    ∃ C : ℝ, ∀ x, f x = C * a^(2 * x) := 
sorry

end continuous_function_solution_l241_241116


namespace tangent_line_properties_l241_241524

noncomputable def curve (x : ℝ) (a b : ℝ) : ℝ := x^2 + a*x + b

theorem tangent_line_properties (a b : ℝ) :
  (∀ x : ℝ, curve 0 a b = b) →
  (∀ x : ℝ, x - (curve x a b - b) + 1 = 0 → (∀ x : ℝ, 2*0 + a = 1)) →
  a + b = 2 :=
by
  intros h_curve h_tangent
  have h_b : b = 1 := by sorry
  have h_a : a = 1 := by sorry
  rw [h_a, h_b]
  norm_num

end tangent_line_properties_l241_241524


namespace x_k_expr_a_x_k_expr_b_x_k_expr_c_y_k_expr_a_y_k_expr_b_y_k_expr_c_l241_241775

theorem x_k_expr_a (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  x k = 6 * x (k - 1) - x (k - 2) := 
by sorry

theorem x_k_expr_b (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  x k = 34 * x (k - 2) - x (k - 4) := 
by sorry

theorem x_k_expr_c (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  x k = 198 * x (k - 3) - x (k - 6) := 
by sorry

theorem y_k_expr_a (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  y k = 6 * y (k - 1) - y (k - 2) := 
by sorry

theorem y_k_expr_b (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  y k = 34 * y (k - 2) - y (k - 4) := 
by sorry

theorem y_k_expr_c (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  y k = 198 * y (k - 3) - y (k - 6) := 
by sorry

end x_k_expr_a_x_k_expr_b_x_k_expr_c_y_k_expr_a_y_k_expr_b_y_k_expr_c_l241_241775


namespace fare_calculation_l241_241706

-- Definitions for given conditions
def initial_mile_fare : ℝ := 3.00
def additional_rate : ℝ := 0.30
def initial_miles : ℝ := 0.5
def available_fare : ℝ := 15 - 3  -- Total minus tip

-- Proof statement
theorem fare_calculation (miles : ℝ) : initial_mile_fare + additional_rate * (miles - initial_miles) / 0.10 = available_fare ↔ miles = 3.5 :=
by
  sorry

end fare_calculation_l241_241706


namespace cos_beta_half_l241_241931

theorem cos_beta_half (α β : ℝ) (hα_ac : 0 < α ∧ α < π / 2) (hβ_ac : 0 < β ∧ β < π / 2) 
  (h1 : Real.tan α = 4 * Real.sqrt 3) (h2 : Real.cos (α + β) = -11 / 14) : 
  Real.cos β = 1 / 2 :=
by
  sorry

end cos_beta_half_l241_241931


namespace hoseok_position_l241_241819

variable (total_people : ℕ) (pos_from_back : ℕ)

theorem hoseok_position (h₁ : total_people = 9) (h₂ : pos_from_back = 5) :
  (total_people - pos_from_back + 1) = 5 :=
by
  sorry

end hoseok_position_l241_241819


namespace total_pay_is_correct_l241_241998

-- Define the weekly pay for employee B
def pay_B : ℝ := 228

-- Define the multiplier for employee A's pay relative to employee B's pay
def multiplier_A : ℝ := 1.5

-- Define the weekly pay for employee A
def pay_A : ℝ := multiplier_A * pay_B

-- Define the total weekly pay for both employees
def total_pay : ℝ := pay_A + pay_B

-- Prove the total pay
theorem total_pay_is_correct : total_pay = 570 := by
  -- Use the definitions and compute the total pay
  sorry

end total_pay_is_correct_l241_241998


namespace largest_positive_integer_n_l241_241769

 

theorem largest_positive_integer_n (n : ℕ) :
  (∀ p : ℕ, Nat.Prime p ∧ 2 < p ∧ p < n → Nat.Prime (n - p)) →
  ∀ m : ℕ, (∀ q : ℕ, Nat.Prime q ∧ 2 < q ∧ q < m → Nat.Prime (m - q)) → n ≥ m → n = 10 :=
by
  sorry

end largest_positive_integer_n_l241_241769


namespace middle_integer_is_five_l241_241057

-- Define the conditions of the problem
def consecutive_one_digit_positive_odd_integers (a b c : ℤ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧
  a + 2 = b ∧ b + 2 = c ∨ a + 2 = c ∧ c + 2 = b

def sum_is_one_seventh_of_product (a b c : ℤ) : Prop :=
  a + b + c = (a * b * c) / 7

-- Define the theorem to prove
theorem middle_integer_is_five :
  ∃ (b : ℤ), consecutive_one_digit_positive_odd_integers (b - 2) b (b + 2) ∧
             sum_is_one_seventh_of_product (b - 2) b (b + 2) ∧
             b = 5 :=
sorry

end middle_integer_is_five_l241_241057


namespace points_per_touchdown_l241_241694

theorem points_per_touchdown (number_of_touchdowns : ℕ) (total_points : ℕ) (h1 : number_of_touchdowns = 3) (h2 : total_points = 21) : (total_points / number_of_touchdowns) = 7 :=
by
  sorry

end points_per_touchdown_l241_241694


namespace max_value_3x_sub_9x_l241_241120

open Real

theorem max_value_3x_sub_9x : ∃ x : ℝ, 3^x - 9^x ≤ 1/4 ∧ (∀ y : ℝ, 3^y - 9^y ≤ 3^x - 9^x) :=
by
  sorry

end max_value_3x_sub_9x_l241_241120


namespace probability_factor_36_l241_241259

theorem probability_factor_36 : (∃ n : ℕ, n ≤ 36 ∧ ∃ k, 36 = k * n) → (9 / 36 = 1 / 4) :=
begin
  sorry
end

end probability_factor_36_l241_241259


namespace triangle_ABC_is_isosceles_l241_241390

open EuclideanGeometry

-- Define the problem context
variables {A B C P Q : Point}

-- Translate the conditions into Lean definitions
axiom triangle_ABC : Triangle A B C
axiom point_P_on_BC : P ∈ line (B, C)
axiom angle_PAB_45 : ∠ P A B = 45
axiom Q_on_perpendicular_bisector : ¬Colinear A P Q ∧ (Q ∈ line (A, C) ∧ dist A Q = dist P Q)
axiom PQ_perpendicular_BC : Perpendicular (segment P Q) (line (B, C))

-- Prove the desired theorem
theorem triangle_ABC_is_isosceles : Isosceles_triangle A B C :=
by
  sorry

end triangle_ABC_is_isosceles_l241_241390


namespace increasing_interval_a_geq_neg2_l241_241008

theorem increasing_interval_a_geq_neg2
  (f : ℝ → ℝ)
  (h : ∀ x, f x = x^2 + 2 * (a - 2) * x + 5)
  (h_inc : ∀ x > 4, f (x + 1) > f x) :
  a ≥ -2 :=
sorry

end increasing_interval_a_geq_neg2_l241_241008


namespace sin_690_degree_l241_241692

theorem sin_690_degree : Real.sin (690 * Real.pi / 180) = -1/2 :=
by
  sorry

end sin_690_degree_l241_241692


namespace dog_bones_l241_241302

theorem dog_bones (initial_bones found_bones : ℕ) (h₁ : initial_bones = 15) (h₂ : found_bones = 8) : initial_bones + found_bones = 23 := by
  sorry

end dog_bones_l241_241302


namespace probability_mass_range_l241_241774

/-- Let ξ be a random variable representing the mass of a badminton product. 
    Suppose P(ξ < 4.8) = 0.3 and P(ξ ≥ 4.85) = 0.32. 
    We want to prove that the probability that the mass is in the range [4.8, 4.85) is 0.38. -/
theorem probability_mass_range (P : ℝ → ℝ) (h1 : P (4.8) = 0.3) (h2 : P (4.85) = 0.32) :
  P (4.8) - P (4.85) = 0.38 :=
by 
  sorry

end probability_mass_range_l241_241774


namespace taller_tree_height_is_108_l241_241234

variables (H : ℝ)

-- Conditions
def taller_tree_height := H
def shorter_tree_height := H - 18
def ratio_condition := (H - 18) / H = 5 / 6

-- Theorem to prove
theorem taller_tree_height_is_108 (hH : 0 < H) (h_ratio : ratio_condition H) : taller_tree_height H = 108 :=
sorry

end taller_tree_height_is_108_l241_241234


namespace truck_transportation_l241_241415

theorem truck_transportation
  (x y t : ℕ) 
  (h1 : xt - yt = 60)
  (h2 : (x - 4) * (t + 10) = xt)
  (h3 : (y - 3) * (t + 10) = yt)
  (h4 : xt = x * t)
  (h5 : yt = y * t) : 
  x - 4 = 8 ∧ y - 3 = 6 ∧ t + 10 = 30 := 
by
  sorry

end truck_transportation_l241_241415


namespace binomial_12_10_eq_66_l241_241458

theorem binomial_12_10_eq_66 : (Nat.choose 12 10) = 66 :=
by
  sorry

end binomial_12_10_eq_66_l241_241458


namespace jerome_time_6_hours_l241_241529

theorem jerome_time_6_hours (T: ℝ) (s_J: ℝ) (t_N: ℝ) (s_N: ℝ)
  (h1: s_J = 4) 
  (h2: t_N = 3) 
  (h3: s_N = 8): T = 6 :=
by
  -- Given s_J = 4, t_N = 3, and s_N = 8,
  -- we need to prove that T = 6.
  sorry

end jerome_time_6_hours_l241_241529


namespace problem_solution_l241_241854

def complex_expression : ℕ := 3 * (3 * (4 * (3 * (4 * (2 + 1) + 1) + 2) + 1) + 2) + 1

theorem problem_solution : complex_expression = 1492 := by
  sorry

end problem_solution_l241_241854


namespace cube_root_of_neg_125_l241_241562

theorem cube_root_of_neg_125 : (-5)^3 = -125 := 
by sorry

end cube_root_of_neg_125_l241_241562


namespace compare_cubic_terms_l241_241923

theorem compare_cubic_terms (a b : ℝ) :
    (a ≥ b → a^3 - b^3 ≥ a * b^2 - a^2 * b) ∧
    (a < b → a^3 - b^3 ≤ a * b^2 - a^2 * b) :=
by sorry

end compare_cubic_terms_l241_241923


namespace reciprocal_of_neg_5_l241_241215

theorem reciprocal_of_neg_5 : (∃ r : ℚ, -5 * r = 1) ∧ r = -1 / 5 :=
by sorry

end reciprocal_of_neg_5_l241_241215


namespace relatively_prime_positive_integers_l241_241029

theorem relatively_prime_positive_integers (a b : ℕ) (h1 : a > b) (h2 : gcd a b = 1) (h3 : (a^3 - b^3) / (a - b)^3 = 91 / 7) : a - b = 1 := 
by 
  sorry

end relatively_prime_positive_integers_l241_241029


namespace melted_mixture_weight_l241_241727

/-- 
If the ratio of zinc to copper is 9:11 and 27 kg of zinc has been consumed, then the total weight of the melted mixture is 60 kg.
-/
theorem melted_mixture_weight (zinc_weight : ℕ) (ratio_zinc_to_copper : ℕ → ℕ → Prop)
  (h_ratio : ratio_zinc_to_copper 9 11) (h_zinc : zinc_weight = 27) :
  ∃ (total_weight : ℕ), total_weight = 60 :=
by
  sorry

end melted_mixture_weight_l241_241727


namespace cosine_double_angle_identity_l241_241494

theorem cosine_double_angle_identity (α : ℝ) (h : Real.sin (α + 7 * Real.pi / 6) = 1) :
  Real.cos (2 * α - 2 * Real.pi / 3) = 1 := by
  sorry

end cosine_double_angle_identity_l241_241494


namespace min_shirts_to_save_l241_241891

theorem min_shirts_to_save (x : ℕ) :
  (75 + 10 * x < if x < 30 then 15 * x else 14 * x) → x = 20 :=
by
  sorry

end min_shirts_to_save_l241_241891


namespace arithmetic_progression_common_difference_l241_241908

theorem arithmetic_progression_common_difference :
  ∀ (A1 An n d : ℕ), A1 = 3 → An = 103 → n = 21 → An = A1 + (n - 1) * d → d = 5 :=
by
  intros A1 An n d h1 h2 h3 h4
  sorry

end arithmetic_progression_common_difference_l241_241908


namespace cuboid_surface_area_correct_l241_241696

-- Define the dimensions of the cuboid
def l : ℕ := 4
def w : ℕ := 5
def h : ℕ := 6

-- Define the function to calculate the surface area of the cuboid
def surface_area (l w h : ℕ) : ℕ := 2 * (l * w + w * h + h * l)

-- The theorem stating that the surface area of the cuboid is 148 cm²
theorem cuboid_surface_area_correct : surface_area l w h = 148 := by
  sorry

end cuboid_surface_area_correct_l241_241696


namespace binary_1011_is_11_decimal_124_is_174_l241_241609

-- Define the conversion from binary to decimal
def binaryToDecimal (n : Nat) : Nat :=
  (n % 10) * 2^0 + ((n / 10) % 10) * 2^1 + ((n / 100) % 10) * 2^2 + ((n / 1000) % 10) * 2^3

-- Define the conversion from decimal to octal through division and remainder
noncomputable def decimalToOctal (n : Nat) : String := 
  let rec aux (n : Nat) (acc : List Nat) : List Nat :=
    if n = 0 then acc else aux (n / 8) ((n % 8) :: acc)
  (aux n []).foldr (fun d s => s ++ d.repr) ""

-- Prove that the binary number 1011 (base 2) equals the decimal number 11
theorem binary_1011_is_11 : binaryToDecimal 1011 = 11 := by
  sorry

-- Prove that the decimal number 124 equals the octal number 174 (base 8)
theorem decimal_124_is_174 : decimalToOctal 124 = "174" := by
  sorry

end binary_1011_is_11_decimal_124_is_174_l241_241609


namespace value_of_a_minus_b_l241_241012

theorem value_of_a_minus_b (a b c : ℝ) 
    (h1 : 2011 * a + 2015 * b + c = 2021)
    (h2 : 2013 * a + 2017 * b + c = 2023)
    (h3 : 2012 * a + 2016 * b + 2 * c = 2026) : 
    a - b = -2 := 
by
  sorry

end value_of_a_minus_b_l241_241012


namespace exists_two_linear_functions_l241_241915

-- Define the quadratic trinomials and their general forms
variables (a b c d e f : ℝ)
-- Assuming coefficients a and d are non-zero
variable (ha : a ≠ 0)
variable (hd : d ≠ 0)

-- Define the linear function
def ell (m n x : ℝ) : ℝ := m * x + n

-- Define the quadratic trinomials P(x) and Q(x) 
def P (x : ℝ) := a * x^2 + b * x + c
def Q (x : ℝ) := d * x^2 + e * x + f

-- Prove that there exist exactly two linear functions ell(x) that satisfy the condition for all x
theorem exists_two_linear_functions : 
  ∃ (m1 m2 n1 n2 : ℝ), 
  (∀ x, P a b c x = Q d e f (ell m1 n1 x)) ∧ 
  (∀ x, P a b c x = Q d e f (ell m2 n2 x)) := 
sorry

end exists_two_linear_functions_l241_241915


namespace third_side_length_l241_241790

/-- Given two sides of a triangle with lengths 4cm and 9cm, prove that the valid length of the third side must be 9cm. -/
theorem third_side_length (a b c : ℝ) (h₀ : a = 4) (h₁ : b = 9) :
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) → (c = 9) :=
by {
  sorry
}

end third_side_length_l241_241790


namespace total_points_correct_l241_241684

def points_from_two_pointers (t : ℕ) : ℕ := 2 * t
def points_from_three_pointers (th : ℕ) : ℕ := 3 * th
def points_from_free_throws (f : ℕ) : ℕ := f

def total_points (two_points three_points free_throws : ℕ) : ℕ :=
  points_from_two_pointers two_points + points_from_three_pointers three_points + points_from_free_throws free_throws

def sam_points : ℕ := total_points 20 5 10
def alex_points : ℕ := total_points 15 6 8
def jake_points : ℕ := total_points 10 8 5
def lily_points : ℕ := total_points 12 3 16

def game_total_points : ℕ := sam_points + alex_points + jake_points + lily_points

theorem total_points_correct : game_total_points = 219 :=
by
  sorry

end total_points_correct_l241_241684


namespace trapezium_shorter_side_l241_241623

theorem trapezium_shorter_side (a b h : ℝ) (H1 : a = 10) (H2 : b = 18) (H3 : h = 10.00001) : a = 10 :=
by
  sorry

end trapezium_shorter_side_l241_241623


namespace rearranged_number_divisible_by_27_l241_241794

theorem rearranged_number_divisible_by_27 (n m : ℕ) (hn : m = 3 * n) 
  (hdigits : ∀ a b : ℕ, (a ∈ n.digits 10 ↔ b ∈ m.digits 10)) : 27 ∣ m :=
sorry

end rearranged_number_divisible_by_27_l241_241794


namespace black_piece_probability_l241_241224

-- Definitions based on conditions
def total_pieces : ℕ := 10 + 5
def black_pieces : ℕ := 10

-- Probability calculation
def probability_black : ℚ := black_pieces / total_pieces

-- Statement to prove
theorem black_piece_probability : probability_black = 2/3 := by
  sorry -- proof to be filled in later

end black_piece_probability_l241_241224


namespace probability_of_two_non_defective_pens_l241_241429

-- Definitions for conditions from the problem
def total_pens : ℕ := 16
def defective_pens : ℕ := 3
def selected_pens : ℕ := 2
def non_defective_pens : ℕ := total_pens - defective_pens

-- Function to calculate probability of drawing non-defective pens
noncomputable def probability_no_defective (total : ℕ) (defective : ℕ) (selected : ℕ) : ℚ :=
  (non_defective_pens / total_pens) * ((non_defective_pens - 1) / (total_pens - 1))

-- Theorem stating the correct answer
theorem probability_of_two_non_defective_pens : 
  probability_no_defective total_pens defective_pens selected_pens = 13 / 20 :=
by
  sorry

end probability_of_two_non_defective_pens_l241_241429


namespace sum_of_elements_in_S_l241_241806

def is_repeating_decimal_form (x : ℝ) : Prop :=
∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ x = (a + b / 10 + c / 100) * (1 / 999)

noncomputable def S : set ℝ := { x | is_repeating_decimal_form x }

theorem sum_of_elements_in_S : ∑ x in S, x = 360 := sorry

end sum_of_elements_in_S_l241_241806


namespace value_of_x_plus_2y_l241_241640

theorem value_of_x_plus_2y (x y : ℝ) (h1 : (x + y) / 3 = 1.6666666666666667) (h2 : 2 * x + y = 7) : x + 2 * y = 8 := by
  sorry

end value_of_x_plus_2y_l241_241640


namespace intersection_A_B_l241_241026

-- Definition of set A
def A : Set ℝ := { x | x ≤ 3 }

-- Definition of set B
def B : Set ℝ := {2, 3, 4, 5}

-- Proof statement
theorem intersection_A_B :
  A ∩ B = {2, 3} :=
sorry

end intersection_A_B_l241_241026


namespace max_m_n_l241_241377

theorem max_m_n (m n: ℕ) (h: m + 3*n - 5 = 2 * Nat.lcm m n - 11 * Nat.gcd m n) : 
  m + n ≤ 70 :=
sorry

end max_m_n_l241_241377


namespace cost_of_soap_per_year_l241_241382

-- Conditions:
def duration_of_soap (bar: Nat) : Nat := 2
def cost_per_bar (bar: Nat) : Real := 8.0
def months_in_year : Nat := 12

-- Derived quantity
def bars_needed (months: Nat) (duration: Nat): Nat := months / duration

-- Theorem statement:
theorem cost_of_soap_per_year : 
  let n := bars_needed months_in_year (duration_of_soap 1)
  n * (cost_per_bar 1) = 48.0 := 
  by 
    -- Skipping proof
    sorry

end cost_of_soap_per_year_l241_241382


namespace quadratic_inequality_solution_l241_241762

theorem quadratic_inequality_solution (m : ℝ) (h : m ≠ 0) : 
  (∃ x : ℝ, m * x^2 - x + 1 < 0) ↔ (m ∈ Set.Iio 0 ∨ m ∈ Set.Ioo 0 (1 / 4)) :=
by
  sorry

end quadratic_inequality_solution_l241_241762


namespace binomial_coefficient_12_10_l241_241469

theorem binomial_coefficient_12_10 : Nat.choose 12 10 = 66 := by
  sorry  -- The actual proof is omitted as instructed.

end binomial_coefficient_12_10_l241_241469


namespace daily_wage_of_a_man_l241_241569

theorem daily_wage_of_a_man (M W : ℝ) 
  (h1 : 24 * M + 16 * W = 11600) 
  (h2 : 12 * M + 37 * W = 11600) : 
  M = 350 :=
by
  sorry

end daily_wage_of_a_man_l241_241569


namespace range_of_f_l241_241581

open Set

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_of_f : range f = {y : ℝ | y ≠ 1} :=
by
  sorry

end range_of_f_l241_241581


namespace incorrect_statement_D_l241_241914

theorem incorrect_statement_D : ∃ a : ℝ, a > 0 ∧ (1 - 1 / (2 * a) < 0) := by
  sorry

end incorrect_statement_D_l241_241914


namespace find_a_l241_241563

theorem find_a (a b c : ℤ) (vertex_cond : ∀ x : ℝ, -a * (x - 1)^2 + 3 = -a * (x - 1)^2 + 3)
    (point_cond : (0, 1) ∈ {p : ℝ × ℝ | p.2 = a * p.1^2 + b * p.1 + c}) :
    a = -2 := by 
sorry

end find_a_l241_241563


namespace log_comparison_l241_241103

theorem log_comparison :
  (Real.log 80 / Real.log 20) < (Real.log 640 / Real.log 80) :=
by
  sorry

end log_comparison_l241_241103


namespace total_ages_l241_241347

variable (Frank : ℕ) (Gabriel : ℕ)
variables (h1 : Frank = 10) (h2 : Gabriel = Frank - 3)

theorem total_ages (hF : Frank = 10) (hG : Gabriel = Frank - 3) : Frank + Gabriel = 17 :=
by
  rw [hF, hG]
  norm_num
  sorry

end total_ages_l241_241347


namespace cloth_sold_l241_241087

theorem cloth_sold (total_sell_price : ℤ) (loss_per_meter : ℤ) (cost_price_per_meter : ℤ) (x : ℤ) 
    (h1 : total_sell_price = 18000) 
    (h2 : loss_per_meter = 5) 
    (h3 : cost_price_per_meter = 50) 
    (h4 : (cost_price_per_meter - loss_per_meter) * x = total_sell_price) : 
    x = 400 :=
by
  sorry

end cloth_sold_l241_241087


namespace probability_factor_of_36_l241_241252

theorem probability_factor_of_36 : 
  (finset.card (finset.filter (λ x : ℕ, 36 % x = 0) (finset.range (36 + 1))) : ℚ) / 36 = 1 / 4 := 
by 
sorry

end probability_factor_of_36_l241_241252


namespace probability_divisor_of_36_l241_241296

theorem probability_divisor_of_36 :
  (∃ (n : ℕ) (h₁ : 1 ≤ n ∧ n ≤ 36), (36 % n = 0) → (1 / 4 : ℝ)) :=
by 
  sorry

end probability_divisor_of_36_l241_241296


namespace compound_weight_l241_241325

noncomputable def weightB : ℝ := 275
noncomputable def ratioAtoB : ℝ := 2 / 10

theorem compound_weight (weightA weightB total_weight : ℝ) 
  (h1 : ratioAtoB = 2 / 10) 
  (h2 : weightB = 275) 
  (h3 : weightA = weightB * (2 / 10)) 
  (h4 : total_weight = weightA + weightB) : 
  total_weight = 330 := 
by sorry

end compound_weight_l241_241325


namespace students_no_A_l241_241658

theorem students_no_A (T AH AM AHAM : ℕ) (h1 : T = 35) (h2 : AH = 10) (h3 : AM = 15) (h4 : AHAM = 5) :
  T - (AH + AM - AHAM) = 15 :=
by
  sorry

end students_no_A_l241_241658


namespace count_ordered_sets_eq_catalan_l241_241339

open BigOperators

/-- The number of ordered sets (a_1, a_2, ..., a_n) of n natural numbers such that 
    1 ≤ a_1 ≤ a_2 ≤ ... ≤ a_n and a_i ≤ i for all i = 1, 2, ..., n 
    is given by the Catalan number A_n = (2n choose n) / (n + 1) --/
theorem count_ordered_sets_eq_catalan (n : ℕ) :
  (∑ a : ℕ, (a ≤ n).toInt * (a ≤ n).toInt) =
    Nat.factorial (2 * n) / (Nat.factorial n * Nat.factorial (n + 1)) :=
  sorry

end count_ordered_sets_eq_catalan_l241_241339


namespace probability_divisor_of_36_l241_241297

theorem probability_divisor_of_36 :
  (∃ (n : ℕ) (h₁ : 1 ≤ n ∧ n ≤ 36), (36 % n = 0) → (1 / 4 : ℝ)) :=
by 
  sorry

end probability_divisor_of_36_l241_241297


namespace smallest_n_l241_241298

theorem smallest_n (n : ℕ) : 17 * n ≡ 136 [MOD 5] → n = 3 := 
by sorry

end smallest_n_l241_241298


namespace total_distance_run_l241_241318

def track_meters : ℕ := 9
def laps_already_run : ℕ := 6
def laps_to_run : ℕ := 5

theorem total_distance_run :
  (laps_already_run * track_meters) + (laps_to_run * track_meters) = 99 := by
  sorry

end total_distance_run_l241_241318


namespace xyz_inequality_l241_241032

theorem xyz_inequality (x y z : ℝ) (h : x + y + z = 0) : 
  6 * (x^3 + y^3 + z^3)^2 ≤ (x^2 + y^2 + z^2)^3 := 
by sorry

end xyz_inequality_l241_241032


namespace range_of_y_function_l241_241583

def range_of_function : Set ℝ :=
  {y : ℝ | ∃ (x : ℝ), x ≠ -2 ∧ y = (x^2 + 5*x + 6)/(x+2)}

theorem range_of_y_function :
  range_of_function = {y : ℝ | y ≠ 1} :=
by
  sorry

end range_of_y_function_l241_241583


namespace sum_of_possible_values_of_cardinality_l241_241772

noncomputable def sum_possible_cardinalities (A : Set ℝ) 
  (h : (Set.Image2 (λ a b => a - b) A A).Finite ∧ (Set.Image2 (λ a b => a - b) A A).toFinset.card = 25) : ℕ := by
  sorry

theorem sum_of_possible_values_of_cardinality (A : Set ℝ)
  (h : (Set.Image2 (λ a b => a - b) A A).Finite ∧ (Set.Image2 (λ a b => a - b) A A).toFinset.card = 25):
  sum_possible_cardinalities A h = 76 := by
  sorry

end sum_of_possible_values_of_cardinality_l241_241772


namespace cost_of_paving_floor_l241_241305

-- Conditions
def length_of_room : ℝ := 8
def width_of_room : ℝ := 4.75
def rate_per_sq_metre : ℝ := 900

-- Statement to prove
theorem cost_of_paving_floor : (length_of_room * width_of_room * rate_per_sq_metre) = 34200 :=
by
  sorry

end cost_of_paving_floor_l241_241305


namespace determinant_scalar_multiplication_l241_241920

theorem determinant_scalar_multiplication (x y z w : ℝ) (h : abs (x * w - y * z) = 10) :
  abs (3*x * 3*w - 3*y * 3*z) = 90 :=
by
  sorry

end determinant_scalar_multiplication_l241_241920


namespace original_number_is_76_l241_241076

-- Define the original number x and the condition given
def original_number_condition (x : ℝ) : Prop :=
  (3 / 4) * x = x - 19

-- State the theorem that the original number x must be 76 if it satisfies the condition
theorem original_number_is_76 (x : ℝ) (h : original_number_condition x) : x = 76 :=
sorry

end original_number_is_76_l241_241076


namespace a4_value_a_n_formula_l241_241019

theorem a4_value : a_4 = 30 := 
by
    sorry

noncomputable def a_n (n : ℕ) : ℕ :=
    (n * (n + 1)^2 * (2 * n + 1)) / 6

theorem a_n_formula (n : ℕ) : a_n n = (n * (n + 1)^2 * (2 * n + 1)) / 6 := 
by
    sorry

end a4_value_a_n_formula_l241_241019


namespace handshake_count_l241_241228

-- Define the conditions
def num_companies : ℕ := 5
def reps_per_company : ℕ := 4
def total_people : ℕ := num_companies * reps_per_company
def handshakes_per_person : ℕ := total_people - 1 - (reps_per_company - 1)

-- Define the theorem to prove
theorem handshake_count : 
  total_people * (total_people - 1 - (reps_per_company - 1)) / 2 = 160 := 
by
  -- Here should be the proof, but it is omitted
  sorry

end handshake_count_l241_241228


namespace x_finishes_work_alone_in_18_days_l241_241735

theorem x_finishes_work_alone_in_18_days
  (y_days : ℕ) (y_worked : ℕ) (x_remaining_days : ℝ)
  (hy : y_days = 15) (hy_worked : y_worked = 10) 
  (hx_remaining : x_remaining_days = 6.000000000000001) :
  ∃ (x_days : ℝ), x_days = 18 :=
by 
  sorry

end x_finishes_work_alone_in_18_days_l241_241735


namespace min_value_expression_l241_241966

theorem min_value_expression (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 1) : 
  ∃(x : ℝ), x ≤ (a - b) * (b - c) * (c - d) * (d - a) ∧ x = -1/8 :=
sorry

end min_value_expression_l241_241966


namespace range_of_f_l241_241580

open Set

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_of_f : range f = {y : ℝ | y ≠ 1} :=
by
  sorry

end range_of_f_l241_241580


namespace car_clock_problem_l241_241042

-- Define the conditions and statements required for the proof
variable (t₀ : ℕ) -- Initial time in minutes corresponding to 2:00 PM
variable (t₁ : ℕ) -- Time in minutes when the accurate watch shows 2:40 PM
variable (t₂ : ℕ) -- Time in minutes when the car clock shows 2:50 PM
variable (t₃ : ℕ) -- Time in minutes when the car clock shows 8:00 PM
variable (rate : ℚ) -- Rate of the car clock relative to real time

-- Define the initial condition
def initial_time := (t₀ = 0)

-- Define the time gain from 2:00 PM to 2:40 PM on the accurate watch
def accurate_watch_time := (t₁ = 40)

-- Define the time gain for car clock from 2:00 PM to 2:50 PM
def car_clock_time := (t₂ = 50)

-- Define the rate of the car clock relative to real time as 5/4
def car_clock_rate := (rate = 5/4)

-- Define the car clock reading at 8:00 PM
def car_clock_later := (t₃ = 8 * 60)

-- Define the actual time corresponding to the car clock reading 8:00 PM
def actual_time : ℚ := (t₀ + (t₃ - t₀) * (4/5))

-- Define the statement theorem using the defined conditions and variables
theorem car_clock_problem 
  (h₀ : initial_time t₀) 
  (h₁ : accurate_watch_time t₁) 
  (h₂ : car_clock_time t₂) 
  (h₃ : car_clock_rate rate) 
  (h₄ : car_clock_later t₃) 
  : actual_time t₀ t₃ = 8 * 60 + 24 :=
by sorry

end car_clock_problem_l241_241042


namespace intersection_is_solution_l241_241522

theorem intersection_is_solution (a b : ℝ) :
  (b = 3 * a + 6 ∧ b = 2 * a - 4) ↔ (3 * a - b = -6 ∧ 2 * a - b = 4) := 
by sorry

end intersection_is_solution_l241_241522


namespace second_person_fraction_removed_l241_241916

theorem second_person_fraction_removed (teeth_total : ℕ) 
    (removed1 removed3 removed4 : ℕ)
    (total_removed: ℕ)
    (h1: teeth_total = 32)
    (h2: removed1 = teeth_total / 4)
    (h3: removed3 = teeth_total / 2)
    (h4: removed4 = 4)
    (h5 : total_removed = 40):
    ((total_removed - (removed1 + removed3 + removed4)) : ℚ) / teeth_total = 3 / 8 :=
by
  sorry

end second_person_fraction_removed_l241_241916


namespace intersection_complement_l241_241677

def U : Set ℤ := Set.univ
def M : Set ℤ := {1, 2}
def P : Set ℤ := {-2, -1, 0, 1, 2}
def CUM : Set ℤ := {x : ℤ | x ∉ M}

theorem intersection_complement :
  P ∩ CUM = {-2, -1, 0} :=
by
  sorry

end intersection_complement_l241_241677


namespace smallest_sum_with_probability_l241_241845

theorem smallest_sum_with_probability (N : ℕ) (p : ℝ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ 6) (h2 : 6 * N = 2022) (h3 : p > 0) :
  ∃ M, M = 337 ∧ (∀ sum, sum = 2022 → P(sum) = p) ∧ (∀ min_sum, min_sum = N → P(min_sum) = p):=
begin
  sorry
end

end smallest_sum_with_probability_l241_241845


namespace coordinates_on_y_axis_l241_241521

theorem coordinates_on_y_axis (m : ℝ) (h : m + 1 = 0) : (m + 1, m + 4) = (0, 3) :=
by
  sorry

end coordinates_on_y_axis_l241_241521


namespace staffing_ways_l241_241901

open Nat

def suitable_candidates (total_resumes : ℕ) (ratio : ℚ) : ℕ :=
  (total_resumes * ratio.num) / ratio.denom

theorem staffing_ways :
  ∃ (ways : ℕ),
    let total_resumes := 30;
    let ratio := (2 : ℚ) / 3;
    let suitable := suitable_candidates total_resumes ratio;
    let positions := 5;
    ways = (suitable * (suitable - 1) * (suitable - 2) * (suitable - 3) * (suitable - 4)) ∧ ways = 930240 :=
by
  sorry

end staffing_ways_l241_241901


namespace percentage_of_loss_is_10_l241_241598

-- Definitions based on conditions
def cost_price : ℝ := 1800
def selling_price : ℝ := 1620
def loss : ℝ := cost_price - selling_price

-- The goal: prove the percentage of loss equals 10%
theorem percentage_of_loss_is_10 :
  (loss / cost_price) * 100 = 10 := by
  sorry

end percentage_of_loss_is_10_l241_241598


namespace polynomial_no_negative_roots_l241_241199

theorem polynomial_no_negative_roots (x : ℝ) (h : x < 0) : x^4 - 4 * x^3 - 6 * x^2 - 3 * x + 9 ≠ 0 := 
by 
  sorry

end polynomial_no_negative_roots_l241_241199


namespace exist_coprime_sums_l241_241967

theorem exist_coprime_sums (n k : ℕ) (h1 : 0 < n) (h2 : Even (k * (n - 1))) :
  ∃ x y : ℕ, Nat.gcd x n = 1 ∧ Nat.gcd y n = 1 ∧ (x + y) % n = k % n :=
  sorry

end exist_coprime_sums_l241_241967


namespace determine_x_l241_241906

noncomputable def proof_problem (x : ℝ) (y : ℝ) : Prop :=
  y > 0 → 2 * (x * y^2 + x^2 * y + 2 * y^2 + 2 * x * y) / (x + y) > 3 * x^2 * y

theorem determine_x (x : ℝ) : 
  (∀ (y : ℝ), y > 0 → proof_problem x y) ↔ 0 ≤ x ∧ x < (1 + Real.sqrt 13) / 3 := 
sorry

end determine_x_l241_241906


namespace seating_arrangement_l241_241796

theorem seating_arrangement (boys girls : ℕ) (alternate : boys = 5 ∧ girls = 4 ∧ alternate_seating : boys = 5 ∧ girls = 4): 
  5! * nat.choose 6 4 * 4! = 43200 :=
begin
  sorry
end

end seating_arrangement_l241_241796


namespace characteristics_of_function_l241_241783

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem characteristics_of_function :
  (∀ x : ℝ, x ≠ 0 → deriv f x = -1 / x^2) ∧
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → x < y → f x > f y) ∧
  (∀ L : ℝ, ∃ x : ℝ, x > 0 ∧ f x > L) ∧
  (∀ L : ℝ, ∃ x : ℝ, x < 0 ∧ f x < -L) :=
by
  sorry

end characteristics_of_function_l241_241783


namespace prairie_total_area_l241_241310

theorem prairie_total_area (dust : ℕ) (untouched : ℕ) (total : ℕ) 
  (h1 : dust = 64535) (h2 : untouched = 522) : total = dust + untouched :=
by
  sorry

end prairie_total_area_l241_241310


namespace square_perimeter_l241_241316

theorem square_perimeter (s : ℝ)
  (h1 : ∃ (s : ℝ), 4 * s = s * 1 + s / 4 * 1 + s * 1 + s / 4 * 1)
  (h2 : ∃ (P : ℝ), P = 4 * s)
  : (5/2) * s = 40 → 4 * s = 64 :=
by
  intro h
  sorry

end square_perimeter_l241_241316


namespace evaporation_period_days_l241_241311

theorem evaporation_period_days
    (initial_water : ℝ)
    (daily_evaporation : ℝ)
    (evaporation_percentage : ℝ)
    (total_evaporated_water : ℝ)
    (number_of_days : ℝ) :
    initial_water = 10 ∧
    daily_evaporation = 0.06 ∧
    evaporation_percentage = 0.12 ∧
    total_evaporated_water = initial_water * evaporation_percentage ∧
    number_of_days = total_evaporated_water / daily_evaporation →
    number_of_days = 20 :=
by
  sorry

end evaporation_period_days_l241_241311


namespace probability_exactly_one_girl_two_boys_l241_241440

-- Defining the probability mass functions for boy and girl as 0.5 each
noncomputable def p_boy : ℝ := 0.5
noncomputable def p_girl : ℝ := 0.5

-- The main theorem stating the problem
theorem probability_exactly_one_girl_two_boys :
  (p_boy = 0.5) → (p_girl = 0.5) →
  let P_1G2B := 3 * (0.5 ^ 3) in
  P_1G2B = 0.375 :=
by
  intros h_boy h_girl
  -- Definitions and calculations would usually go here
  sorry

end probability_exactly_one_girl_two_boys_l241_241440


namespace Sam_has_38_dollars_l241_241200

theorem Sam_has_38_dollars (total_money erica_money sam_money : ℕ) 
  (h1 : total_money = 91)
  (h2 : erica_money = 53) 
  (h3 : total_money = erica_money + sam_money) : 
  sam_money = 38 := 
by 
  sorry

end Sam_has_38_dollars_l241_241200


namespace coord_sum_D_l241_241555

def is_midpoint (M C D : ℝ × ℝ) := M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)

theorem coord_sum_D (M C D : ℝ × ℝ) (h : is_midpoint M C D) (hM : M = (4, 6)) (hC : C = (10, 2)) :
  D.1 + D.2 = 8 :=
sorry

end coord_sum_D_l241_241555


namespace nap_hours_in_70_days_l241_241539

-- Define the variables and conditions
variable (n d a b c e : ℕ)  -- assuming they are natural numbers

-- Define the total nap hours function
noncomputable def total_nap_hours (n d a b c e : ℕ) : ℕ :=
  (a + b) * 10

-- The statement to prove
theorem nap_hours_in_70_days (n d a b c e : ℕ) :
  total_nap_hours n d a b c e = (a + b) * 10 :=
by sorry

end nap_hours_in_70_days_l241_241539


namespace ed_lost_seven_marbles_l241_241112

theorem ed_lost_seven_marbles (D L : ℕ) (h1 : ∃ (Ed_init Tim_init : ℕ), Ed_init = D + 19 ∧ Tim_init = D - 10)
(h2 : ∃ (Ed_final Tim_final : ℕ), Ed_final = D + 19 - L - 4 ∧ Tim_final = D - 10 + 4 + 3)
(h3 : ∀ (Ed_final : ℕ), Ed_final = D + 8)
(h4 : ∀ (Tim_final : ℕ), Tim_final = D):
  L = 7 :=
by
  sorry

end ed_lost_seven_marbles_l241_241112


namespace sum_first_60_terms_l241_241219

theorem sum_first_60_terms {a : ℕ → ℤ}
  (h : ∀ n, a (n + 1) + (-1)^n * a n = 2 * n - 1) :
  (Finset.range 60).sum a = 1830 :=
sorry

end sum_first_60_terms_l241_241219


namespace probability_at_least_three_copresidents_l241_241842

noncomputable def binomial_coeff (n k : ℕ) : ℕ :=
  if h : k ≤ n then (n.choose k) else 0

/-- The probability of selecting at least three co-presidents when a club is randomly chosen and 
then four members are randomly selected from the club -/
theorem probability_at_least_three_copresidents :
  let p1 := (4 * (10 - 4) + 1) / (binomial_coeff 10 4) in
  let p2 := (4 * (12 - 4) + 1) / (binomial_coeff 12 4) in
  let p3 := (4 * (15 - 4) + 1) / (binomial_coeff 15 4) in
  (1 / 3) * (p1 + p2 + p3) ≈ 0.035 :=
by sorry

end probability_at_least_three_copresidents_l241_241842


namespace probability_even_product_l241_241765

-- Conditions
def box := {1, 2, 4}
def draw := List.product box box box

-- The Problem: Prove the probability that the product of numbers on three drawn chips is even equals 26/27.
theorem probability_even_product :
  let total_outcomes := 27
  let favorable_outcomes := 26
  favorable_outcomes / total_outcomes = (26/27 : ℚ) :=
by
  -- sorry is used to indicate the proof is omitted
  sorry

end probability_even_product_l241_241765


namespace area_of_shaded_region_l241_241332

theorem area_of_shaded_region :
  let ABCD_area := 36
  let EFGH_area := 1 * 3
  let IJKL_area := 2 * 4
  let MNOP_area := 3 * 1
  let shaded_area := ABCD_area - (EFGH_area + IJKL_area + MNOP_area)
  shaded_area = 22 :=
by
  let ABCD_area := 36
  let EFGH_area := 1 * 3
  let IJKL_area := 2 * 4
  let MNOP_area := 3 * 1
  let shaded_area := ABCD_area - (EFGH_area + IJKL_area + MNOP_area)
  sorry

end area_of_shaded_region_l241_241332


namespace solution_set_of_gx_lt_0_l241_241353

noncomputable def f (x : ℝ) : ℝ := 2 ^ x

noncomputable def f_inv (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def g (x : ℝ) : ℝ := f_inv (1 - x) - f_inv (1 + x)

theorem solution_set_of_gx_lt_0 : { x : ℝ | g x < 0 } = Set.Ioo 0 1 := by
  sorry

end solution_set_of_gx_lt_0_l241_241353


namespace determine_alpha_l241_241935

variables (m n : ℝ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_mn : m + n = 1)
variables (α : ℝ)

-- Defining the minimum value condition
def minimum_value_condition : Prop :=
  (1 / m + 16 / n) = 25

-- Defining the curve passing through point P
def passes_through_P : Prop :=
  (m / 5) ^ α = (m / 4)

theorem determine_alpha
  (h_min_value : minimum_value_condition m n)
  (h_passes_through : passes_through_P m α) :
  α = 1 / 2 :=
sorry

end determine_alpha_l241_241935


namespace ball_maximum_height_l241_241307
-- Import necessary libraries

-- Define the height function
def ball_height (t : ℝ) : ℝ := -5 * t^2 + 50 * t + 20

-- Proposition asserting that the maximum height of the ball is 145 meters
theorem ball_maximum_height : ∃ t : ℝ, ball_height t = 145 :=
  sorry

end ball_maximum_height_l241_241307


namespace lucille_house_difference_l241_241816

-- Define the heights of the houses as given in the conditions.
def height_lucille : ℕ := 80
def height_neighbor1 : ℕ := 70
def height_neighbor2 : ℕ := 99

-- Define the total height of the houses.
def total_height : ℕ := height_neighbor1 + height_lucille + height_neighbor2

-- Define the average height of the houses.
def average_height : ℕ := total_height / 3

-- Define the height difference between Lucille's house and the average height.
def height_difference : ℕ := average_height - height_lucille

-- The theorem to prove.
theorem lucille_house_difference :
  height_difference = 3 := by
  sorry

end lucille_house_difference_l241_241816


namespace probability_factor_of_36_l241_241291

theorem probability_factor_of_36 : (∃ p : ℚ, p = 1 / 4 ∧
  let n := 36 in
  let factors := {d | d ∣ n ∧ d > 0 ∧ d ≤ n} in
  let total_pos_ints := 36 in
  let num_factors := set.card factors in
  p = num_factors / total_pos_ints) :=
begin
  sorry
end

end probability_factor_of_36_l241_241291


namespace smallest_five_digit_multiple_of_18_l241_241341

def is_multiple_of (x : ℕ) (k : ℕ) : Prop := ∃ n : ℕ, x = k * n

theorem smallest_five_digit_multiple_of_18 : 
  ∃ x : ℕ, 
    (10000 ≤ x ∧ x < 100000) ∧ 
    is_multiple_of x 18 ∧ 
    (∀ y : ℕ, (10000 ≤ y ∧ y < 100000) ∧ is_multiple_of y 18 → x ≤ y) :=
begin
  use 10008,
  -- The details of the proof are omitted.
  sorry
end

end smallest_five_digit_multiple_of_18_l241_241341


namespace total_pupils_correct_l241_241174

-- Definitions of the conditions
def number_of_girls : ℕ := 308
def number_of_boys : ℕ := 318

-- Definition of the number of pupils
def total_number_of_pupils : ℕ := number_of_girls + number_of_boys

-- The theorem to be proven
theorem total_pupils_correct : total_number_of_pupils = 626 := by
  -- The proof would go here
  sorry

end total_pupils_correct_l241_241174


namespace machine_fill_time_l241_241313

theorem machine_fill_time (filled_cans : ℕ) (time_per_batch : ℕ) (total_cans : ℕ) (expected_time : ℕ)
  (h1 : filled_cans = 150)
  (h2 : time_per_batch = 8)
  (h3 : total_cans = 675)
  (h4 : expected_time = 36) :
  (total_cans / filled_cans) * time_per_batch = expected_time :=
by 
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end machine_fill_time_l241_241313


namespace probability_factor_of_36_l241_241251

theorem probability_factor_of_36 : 
  (finset.card (finset.filter (λ x : ℕ, 36 % x = 0) (finset.range (36 + 1))) : ℚ) / 36 = 1 / 4 := 
by 
sorry

end probability_factor_of_36_l241_241251


namespace value_of_fraction_l241_241830

variable {x y : ℝ}

theorem value_of_fraction (hx : x ≠ 0) (hy : y ≠ 0) (h : (3 * x + y) / (x - 3 * y) = -2) :
  (x + 3 * y) / (3 * x - y) = 2 :=
sorry

end value_of_fraction_l241_241830


namespace range_of_m_range_of_x_l241_241134

variable {a b m : ℝ}

-- Given conditions
axiom a_pos : a > 0
axiom b_pos : b > 0
axiom sum_eq_one : a + b = 1

-- Problem (I): Prove range of m
theorem range_of_m (h : ab ≤ m) : m ≥ 1 / 4 := by
  sorry

variable {x : ℝ}

-- Problem (II): Prove range of x
theorem range_of_x (h : 4 / a + 1 / b ≥ |2 * x - 1| - |x + 2|) : -2 ≤ x ∧ x ≤ 6 := by
  sorry

end range_of_m_range_of_x_l241_241134


namespace width_of_first_sheet_paper_l241_241210

theorem width_of_first_sheet_paper :
  ∀ (w : ℝ),
  2 * 11 * w = 2 * 4.5 * 11 + 100 → 
  w = 199 / 22 := 
by
  intro w
  intro h
  sorry

end width_of_first_sheet_paper_l241_241210


namespace simplify_and_evaluate_l241_241823

noncomputable def simplified_expr (x y : ℝ) : ℝ :=
  ((-2 * x + y)^2 - (2 * x - y) * (y + 2 * x) - 6 * y) / (2 * y)

theorem simplify_and_evaluate :
  let x := -1
  let y := 2
  simplified_expr x y = 1 :=
by
  -- Proof will go here
  sorry

end simplify_and_evaluate_l241_241823


namespace smallest_positive_integer_n_l241_241419

theorem smallest_positive_integer_n :
  ∃ (n : ℕ), 5 * n ≡ 1978 [MOD 26] ∧ n = 16 :=
by
  sorry

end smallest_positive_integer_n_l241_241419


namespace expression_evaluates_to_4_l241_241899

theorem expression_evaluates_to_4 :
  2 * Real.cos (Real.pi / 6) + (- 1 / 2 : ℝ)⁻¹ + |Real.sqrt 3 - 2| + (2 * Real.sqrt (9 / 4))^0 + Real.sqrt 9 = 4 := 
by
  sorry

end expression_evaluates_to_4_l241_241899


namespace find_ellipse_equation_l241_241352

-- Definitions based on conditions
def ellipse_centered_at_origin (x y : ℝ) (m n : ℝ) := m * x ^ 2 + n * y ^ 2 = 1

def passes_through_points_A_and_B (m n : ℝ) := 
  (ellipse_centered_at_origin 0 (-2) m n) ∧ (ellipse_centered_at_origin (3 / 2) (-1) m n)

-- Statement to be proved
theorem find_ellipse_equation : 
  ∃ (m n : ℝ), (m > 0) ∧ (n > 0) ∧ (m ≠ n) ∧ 
  passes_through_points_A_and_B m n ∧ 
  m = 1 / 3 ∧ n = 1 / 4 :=
by sorry

end find_ellipse_equation_l241_241352


namespace fraction_meaningful_l241_241849

theorem fraction_meaningful (x : ℝ) : x ≠ 1 ↔ ∃ (f : ℝ → ℝ), f x = (x + 2) / (x - 1) :=
by
  sorry

end fraction_meaningful_l241_241849


namespace correct_calculation_l241_241424

theorem correct_calculation (a : ℝ) :
  2 * a^4 * 3 * a^5 = 6 * a^9 :=
by
  sorry

end correct_calculation_l241_241424


namespace sum_eq_sqrt_122_l241_241030

theorem sum_eq_sqrt_122 
  (a b c : ℝ) 
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h1 : a^2 + b^2 + c^2 = 58) 
  (h2 : a * b + b * c + c * a = 32) :
  a + b + c = Real.sqrt 122 := 
by
  sorry

end sum_eq_sqrt_122_l241_241030


namespace meal_combinations_count_l241_241526

/-- Define the number of menu items -/
def num_menu_items : ℕ := 15

/-- Define the number of distinct combinations of meals Maryam and Jorge can order,
    considering they may choose the same dish and distinguishing who orders what -/
theorem meal_combinations_count (maryam_dishes jorge_dishes : ℕ) : 
  maryam_dishes = num_menu_items ∧ jorge_dishes = num_menu_items → 
  maryam_dishes * jorge_dishes = 225 :=
by
  intros h
  simp only [num_menu_items] at h -- Utilize the definition of num_menu_items
  sorry

end meal_combinations_count_l241_241526


namespace water_fee_relationship_xiao_qiangs_water_usage_l241_241883

variable (x y : ℝ)
variable (H1 : x > 10)
variable (H2 : y = 3 * x - 8)

theorem water_fee_relationship : y = 3 * x - 8 := 
  by 
    exact H2

theorem xiao_qiangs_water_usage : y = 67 → x = 25 :=
  by
    intro H
    have H_eq : 67 = 3 * x - 8 := by 
      rw [←H2, H]
    linarith

end water_fee_relationship_xiao_qiangs_water_usage_l241_241883


namespace handshake_count_l241_241230

theorem handshake_count :
  let total_people := 5 * 4
  let handshakes_per_person := total_people - 1 - 3
  let total_handshakes_with_double_count := total_people * handshakes_per_person
  let total_handshakes := total_handshakes_with_double_count / 2
  total_handshakes = 160 :=
by
-- We include "sorry" to indicate that the proof is not provided.
sorry

end handshake_count_l241_241230


namespace mike_profit_l241_241544

-- Define the conditions
def total_acres : ℕ := 200
def cost_per_acre : ℕ := 70
def sold_acres := total_acres / 2
def selling_price_per_acre : ℕ := 200

-- Statement to prove the profit Mike made is $6,000
theorem mike_profit :
  let total_cost := total_acres * cost_per_acre
  let total_revenue := sold_acres * selling_price_per_acre
  total_revenue - total_cost = 6000 := 
by
  sorry

end mike_profit_l241_241544


namespace max_value_of_f_l241_241123

noncomputable def f (x : ℝ) : ℝ := 3^x - 9^x

theorem max_value_of_f : ∃ x : ℝ, f x = 1 / 4 := sorry

end max_value_of_f_l241_241123


namespace garden_snake_length_l241_241892

theorem garden_snake_length :
  ∀ (garden_snake boa_constrictor : ℝ),
    boa_constrictor * 7.0 = garden_snake →
    boa_constrictor = 1.428571429 →
    garden_snake = 10.0 :=
by
  intros garden_snake boa_constrictor H1 H2
  sorry

end garden_snake_length_l241_241892


namespace modular_inverse_l241_241139

theorem modular_inverse :
  (24 * 22) % 53 = 1 :=
by
  have h1 : (24 * -29) % 53 = (53 * 0 - 29 * 24) % 53 := by sorry
  have h2 : (24 * -29) % 53 = (-29 * 24) % 53 := by sorry
  have h3 : (-29 * 24) % 53 = (-29 % 53 * 24 % 53 % 53) := by sorry
  have h4 : -29 % 53 = 53 - 24 := by sorry
  have h5 : (53 - 29) % 53 = (22 * 22) % 53 := by sorry
  have h6 : (22 * 22) % 53 = (24 * 22) % 53 := by sorry
  have h7 : (24 * 22) % 53 = 1 := by sorry
  exact h7

end modular_inverse_l241_241139


namespace no_such_function_exists_l241_241764

theorem no_such_function_exists (f : ℕ → ℕ) (h : ∀ n, f (f n) = n + 2019) : false :=
sorry

end no_such_function_exists_l241_241764


namespace store_A_has_highest_capacity_l241_241994

noncomputable def total_capacity_A : ℕ := 5 * 6 * 9
noncomputable def total_capacity_B : ℕ := 8 * 4 * 7
noncomputable def total_capacity_C : ℕ := 10 * 3 * 8

theorem store_A_has_highest_capacity : total_capacity_A = 270 ∧ total_capacity_A > total_capacity_B ∧ total_capacity_A > total_capacity_C := 
by 
  -- Proof skipped with a placeholder
  sorry

end store_A_has_highest_capacity_l241_241994


namespace sum_of_roots_l241_241648

theorem sum_of_roots (a β : ℝ) 
  (h1 : a^2 - 2 * a = 1) 
  (h2 : β^2 - 2 * β - 1 = 0) 
  (hne : a ≠ β) 
  : a + β = 2 := 
sorry

end sum_of_roots_l241_241648


namespace quadratic_completing_square_sum_l241_241482

theorem quadratic_completing_square_sum (q t : ℝ) :
    (∃ (x : ℝ), 9 * x^2 - 54 * x - 36 = 0 ∧ (x + q)^2 = t) →
    q + t = 10 := sorry

end quadratic_completing_square_sum_l241_241482


namespace area_triangle_DEF_l241_241527

theorem area_triangle_DEF 
  (DE EL EF : ℝ) (H1 : DE = 15) (H2 : EL = 12) (H3 : EF = 20) 
  (DL : ℝ) (H4 : DE^2 = EL^2 + DL^2) (H5 : DL * EF = DL * 20) :
  1/2 * EF * DL = 90 :=
by
  -- Use the assumptions and conditions to state the theorem.
  sorry

end area_triangle_DEF_l241_241527


namespace polar_coordinate_conversion_l241_241366

theorem polar_coordinate_conversion :
  ∃ (r θ : ℝ), (r = 2) ∧ (θ = 11 * Real.pi / 8) ∧ 
    ∀ (r1 θ1 : ℝ), (r1 = -2) ∧ (θ1 = 3 * Real.pi / 8) →
      (abs r1 = r) ∧ (θ1 + Real.pi = θ) :=
by
  sorry

end polar_coordinate_conversion_l241_241366


namespace factor_probability_36_l241_241286

-- Definitions based on conditions
def is_factor (n d : ℕ) : Prop := d ∣ n
def num_factors (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (is_factor n).card

def probability_factor (n : ℕ) : ℚ := num_factors n / n

-- Problem statement
theorem factor_probability_36 : probability_factor 36 = 1 / 4 :=
  by sorry

end factor_probability_36_l241_241286


namespace find_dividend_l241_241862

noncomputable def dividend (divisor quotient remainder : ℕ) : ℕ :=
  (divisor * quotient) + remainder

theorem find_dividend :
  ∀ (divisor quotient remainder : ℕ), 
  divisor = 16 → 
  quotient = 8 → 
  remainder = 4 → 
  dividend divisor quotient remainder = 132 :=
by
  intros divisor quotient remainder hdiv hquo hrem
  sorry

end find_dividend_l241_241862


namespace tangent_point_condition_l241_241005

open Function

def f (x : ℝ) : ℝ := x^3 - 3 * x
def tangent_line (s : ℝ) (x t : ℝ) : ℝ := (3 * s^2 - 3) * (x - 2) + s^3 - 3 * s

theorem tangent_point_condition (t : ℝ) (h_tangent : ∃s : ℝ, tangent_line s 2 t = t) 
  (h_not_on_curve : ∀ s, (2, t) ≠ (s, f s)) : t = -6 :=
by
  sorry

end tangent_point_condition_l241_241005


namespace bags_total_on_next_day_l241_241411

def bags_on_monday : ℕ := 7
def additional_bags : ℕ := 5
def bags_on_next_day : ℕ := bags_on_monday + additional_bags

theorem bags_total_on_next_day : bags_on_next_day = 12 := by
  unfold bags_on_next_day
  unfold bags_on_monday
  unfold additional_bags
  sorry

end bags_total_on_next_day_l241_241411


namespace tanya_bought_11_pears_l241_241045

variable (P : ℕ)

-- Define the given conditions about the number of different fruits Tanya bought
def apples : ℕ := 4
def pineapples : ℕ := 2
def basket_of_plums : ℕ := 1

-- Define the total number of fruits initially and the remaining fruits
def initial_fruit_total : ℕ := 18
def remaining_fruit_total : ℕ := 9
def half_fell_out_of_bag : ℕ := remaining_fruit_total * 2

-- The main theorem to prove
theorem tanya_bought_11_pears (h : P + apples + pineapples + basket_of_plums = initial_fruit_total) : P = 11 := by
  -- providing a placeholder for the proof
  sorry

end tanya_bought_11_pears_l241_241045


namespace value_of_fraction_l241_241829

variable {x y : ℝ}

theorem value_of_fraction (hx : x ≠ 0) (hy : y ≠ 0) (h : (3 * x + y) / (x - 3 * y) = -2) :
  (x + 3 * y) / (3 * x - y) = 2 :=
sorry

end value_of_fraction_l241_241829


namespace points_within_distance_5_l241_241952

noncomputable def distance (x y z : ℝ) : ℝ := Real.sqrt (x^2 + y^2 + z^2)

def within_distance (x y z : ℝ) (d : ℝ) : Prop := distance x y z ≤ d

def A := (1, 1, 1)
def B := (1, 2, 2)
def C := (2, -3, 5)
def D := (3, 0, 4)

theorem points_within_distance_5 :
  within_distance 1 1 1 5 ∧
  within_distance 1 2 2 5 ∧
  ¬ within_distance 2 (-3) 5 5 ∧
  within_distance 3 0 4 5 :=
by {
  sorry
}

end points_within_distance_5_l241_241952


namespace inequality_must_hold_l241_241516

theorem inequality_must_hold (m n : ℝ) (h : m > n) : 2 + m > 2 + n :=
sorry

end inequality_must_hold_l241_241516


namespace probability_open_path_l241_241869

-- Define necessary terms
def total_doors (n : ℕ) : ℕ := 2 * (n - 1)
def locked_doors (n : ℕ) : ℕ := total_doors n / 2

-- Helper function to compute binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Probability theorem
theorem probability_open_path (n : ℕ) (h : n > 1) : 
  ((locked_doors n) = (n-1)) → 
  (∃ p, p = (2^(n-1)) / (binom (total_doors n) (n-1))) :=
by {
  intro h1,
  use ((2^(n-1)) / (binom (total_doors n) (n-1))),
  sorry
}

end probability_open_path_l241_241869


namespace work_rate_B_l241_241594

theorem work_rate_B :
  (∀ A B : ℝ, A = 30 → (1 / A + 1 / B = 1 / 19.411764705882355) → B = 55) := by 
    intro A B A_cond combined_rate
    have hA : A = 30 := A_cond
    rw [hA] at combined_rate
    sorry

end work_rate_B_l241_241594


namespace fran_speed_l241_241670

-- Definitions for conditions
def joann_speed : ℝ := 15 -- in miles per hour
def joann_time : ℝ := 4 -- in hours
def fran_time : ℝ := 2 -- in hours
def joann_distance : ℝ := joann_speed * joann_time -- distance Joann traveled

-- Proof Goal Statement
theorem fran_speed (hf: fran_time ≠ 0) : (joann_speed * joann_time) / fran_time = 30 :=
by
  -- Sorry placeholder skips the proof steps
  sorry

end fran_speed_l241_241670


namespace janice_time_left_l241_241955

def time_before_movie : ℕ := 2 * 60
def homework_time : ℕ := 30
def cleaning_time : ℕ := homework_time / 2
def walking_dog_time : ℕ := homework_time + 5
def taking_trash_time : ℕ := homework_time * 1 / 6

theorem janice_time_left : time_before_movie - (homework_time + cleaning_time + walking_dog_time + taking_trash_time) = 35 :=
by
  sorry

end janice_time_left_l241_241955


namespace reciprocal_of_neg_five_l241_241217

theorem reciprocal_of_neg_five: 
  ∃ x : ℚ, -5 * x = 1 ∧ x = -1 / 5 := 
sorry

end reciprocal_of_neg_five_l241_241217


namespace probability_path_from_first_to_last_floor_open_doors_l241_241873

noncomputable
def probability_path_possible (n : ℕ) : ℚ :=
  (2 ^ (n - 1)) / (nat.choose (2 * (n - 1)) (n - 1))

theorem probability_path_from_first_to_last_floor_open_doors (n : ℕ) :
  probability_path_possible n = (2 ^ (n - 1)) / (nat.choose (2 * (n - 1)) (n - 1)) :=
by
  sorry

end probability_path_from_first_to_last_floor_open_doors_l241_241873


namespace arithmetic_seq_third_term_l241_241166

theorem arithmetic_seq_third_term
  (a d : ℝ)
  (h : a + (a + 2 * d) = 10) :
  a + d = 5 := by
  sorry

end arithmetic_seq_third_term_l241_241166


namespace absolute_value_inequality_l241_241204

theorem absolute_value_inequality (x : ℝ) : ¬ (|x - 3| + |x + 4| < 6) :=
sorry

end absolute_value_inequality_l241_241204


namespace sum_of_two_numbers_l241_241233

theorem sum_of_two_numbers (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h1 : x * y = 12) (h2 : 1 / x = 3 * (1 / y)) : x + y = 8 :=
by
  sorry

end sum_of_two_numbers_l241_241233


namespace probability_factor_of_36_l241_241254

def is_factor_of (n d : ℕ) : Prop := d % n = 0

theorem probability_factor_of_36 : 
  let total := 36
  let factors_of_36 := { n | n ≤ 36 ∧ is_factor_of n 36 }
  (factors_of_36.to_finset.card : ℚ) / total = 1 / 4 := 
by
  sorry

end probability_factor_of_36_l241_241254


namespace lcm_Anthony_Bethany_Casey_Dana_l241_241445

theorem lcm_Anthony_Bethany_Casey_Dana : Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 10) = 120 := 
by
  sorry

end lcm_Anthony_Bethany_Casey_Dana_l241_241445


namespace sum_of_two_numbers_l241_241985

theorem sum_of_two_numbers (x y : ℕ) (h1 : y = x + 4) (h2 : y = 30) : x + y = 56 :=
by
  -- Asserts the conditions and goal statement
  sorry

end sum_of_two_numbers_l241_241985


namespace dividing_by_10_l241_241094

theorem dividing_by_10 (x : ℤ) (h : x + 8 = 88) : x / 10 = 8 :=
by
  sorry

end dividing_by_10_l241_241094


namespace johns_average_speed_l241_241671

theorem johns_average_speed :
  let distance1 := 20
  let speed1 := 10
  let distance2 := 30
  let speed2 := 20
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 14.29 :=
by
  sorry

end johns_average_speed_l241_241671


namespace solve_quadratic_eq_l241_241396

theorem solve_quadratic_eq (x : ℝ) : x ^ 2 + 2 * x - 5 = 0 → (x = -1 + Real.sqrt 6 ∨ x = -1 - Real.sqrt 6) :=
by 
  intro h
  sorry

end solve_quadratic_eq_l241_241396


namespace rahul_batting_average_l241_241393

theorem rahul_batting_average:
  ∃ (A : ℝ), A = 46 ∧
  (∀ (R : ℝ), R = 138 → R = 54 * 4 - 78 → A = R / 3) ∧
  ∃ (n_matches : ℕ), n_matches = 3 :=
by
  sorry

end rahul_batting_average_l241_241393


namespace blue_face_probability_l241_241015

def sides : ℕ := 12
def green_faces : ℕ := 5
def blue_faces : ℕ := 4
def red_faces : ℕ := 3

theorem blue_face_probability : 
  (blue_faces : ℚ) / sides = 1 / 3 :=
by
  sorry

end blue_face_probability_l241_241015


namespace find_y_value_l241_241980

theorem find_y_value :
  (∃ m b : ℝ, (∀ x y : ℝ, (x = 2 ∧ y = 5) ∨ (x = 6 ∧ y = 17) ∨ (x = 10 ∧ y = 29) → y = m * x + b))
  → (∃ y : ℝ, x = 40 → y = 119) := by
  sorry

end find_y_value_l241_241980


namespace worker_b_days_l241_241728

variables (W_A W_B W : ℝ)
variables (h1 : W_A = 2 * W_B)
variables (h2 : (W_A + W_B) * 10 = W)
variables (h3 : W = 30 * W_B)

theorem worker_b_days : ∃ days : ℝ, days = 30 :=
by
  sorry

end worker_b_days_l241_241728


namespace problem1_problem2_l241_241132

variable {a b : ℝ}

theorem problem1
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : a^3 + b^3 = 2)
  : (a + b) * (a^5 + b^5) ≥ 4 := sorry

theorem problem2
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : a^3 + b^3 = 2)
  : a + b ≤ 2 := sorry

end problem1_problem2_l241_241132


namespace jerry_remaining_debt_l241_241667

variable (two_months_ago_payment last_month_payment total_debt : ℕ)

def remaining_debt : ℕ := total_debt - (two_months_ago_payment + last_month_payment)

theorem jerry_remaining_debt :
  two_months_ago_payment = 12 →
  last_month_payment = 12 + 3 →
  total_debt = 50 →
  remaining_debt two_months_ago_payment last_month_payment total_debt = 23 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end jerry_remaining_debt_l241_241667


namespace set_of_points_plane_z_eq_one_distance_point_P_to_plane_xOy_l241_241798

open EuclideanGeometry

-- Define the plane z = 1
def plane_z_eq_one : Plane ℝ := { normal := ⟨0, 0, 1⟩, point := ⟨0, 0, 1⟩ }

-- Define the point P(2, 3, 5)
def point_P : EuclideanGeometry.Point ℝ 3 := ⟨2, 3, 5⟩

-- Define the plane xOy (z=0)
def plane_xOy : Plane ℝ := { normal := ⟨0, 0, 1⟩, point := ⟨0, 0, 0⟩ }

theorem set_of_points_plane_z_eq_one :
  ∀ (p : EuclideanGeometry.Point ℝ 3), p.z = 1 ↔ p ∈ plane_z_eq_one :=
by
  sorry

theorem distance_point_P_to_plane_xOy : distance point_P plane_xOy = 5 :=
by
  sorry


end set_of_points_plane_z_eq_one_distance_point_P_to_plane_xOy_l241_241798


namespace intersection_of_A_and_B_l241_241942

def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}
def B : Set ℤ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {-2, 0, 2} := by
  sorry

end intersection_of_A_and_B_l241_241942


namespace probability_factor_of_36_l241_241276

theorem probability_factor_of_36 : 
  let pos_ints := { n : ℕ | 1 ≤ n ∧ n ≤ 36 }
  let num_divisors := { d : ℕ | d ∣ 36 }
  (num_divisors.to_finset.card : ℚ) / (pos_ints.to_finset.card : ℚ) = 1 / 4 :=
by {
  sorry
}

end probability_factor_of_36_l241_241276


namespace trigonometric_identity_1_l241_241306

theorem trigonometric_identity_1 :
  ( (Real.sqrt 3 * Real.sin (-1200 * Real.pi / 180)) / (Real.tan (11 * Real.pi / 3)) 
  - Real.cos (585 * Real.pi / 180) * Real.tan (-37 * Real.pi / 4) = (Real.sqrt 3 / 2) - (Real.sqrt 2 / 2) ) :=
by
  sorry

end trigonometric_identity_1_l241_241306


namespace probability_divisor_of_36_is_one_fourth_l241_241284

noncomputable def probability_divisor_of_36 : ℚ :=
  let total_divisors := 9 in
  let total_integers := 36 in
  total_divisors / total_integers

theorem probability_divisor_of_36_is_one_fourth :
  probability_divisor_of_36 = 1 / 4 :=
by
  sorry

end probability_divisor_of_36_is_one_fourth_l241_241284


namespace sequence_properties_l241_241937

open BigOperators

-- Given conditions
def is_geometric_sequence (a : ℕ → ℝ) := ∃ q > 0, ∀ n, a (n + 1) = a n * q
def sequence_a (n : ℕ) : ℝ := 2^(n - 1)

-- Definitions for b_n and S_n
def sequence_b (n : ℕ) : ℕ := n - 1
def sequence_c (n : ℕ) : ℝ := sequence_a n * (sequence_b n) -- c_n = a_n * b_n

-- Statement of the problem
theorem sequence_properties (a : ℕ → ℝ) (hgeo : is_geometric_sequence a) (h1 : a 1 = 1) (h2 : a 2 * a 4 = 16) : 
 (∀ n, sequence_b n = n - 1 ) ∧ S_n = ∑ i in Finset.range n, sequence_c (i + 1) := sorry

end sequence_properties_l241_241937


namespace linear_equation_value_m_l241_241162

theorem linear_equation_value_m (m : ℝ) (h : ∀ x : ℝ, 2 * x^(m - 1) + 3 = 0 → x ≠ 0) : m = 2 :=
sorry

end linear_equation_value_m_l241_241162


namespace pencils_placed_by_sara_l241_241574

theorem pencils_placed_by_sara (initial_pencils final_pencils : ℕ) (h1 : initial_pencils = 115) (h2 : final_pencils = 215) : final_pencils - initial_pencils = 100 := by
  sorry

end pencils_placed_by_sara_l241_241574


namespace grid_with_value_exists_possible_values_smallest_possible_value_l241_241620

open Nat

def isGridValuesP (P : ℕ) (a b c d e f g h i : ℕ) : Prop :=
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i) ∧
  (b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
   d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
   e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
   f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
   g ≠ h ∧ g ≠ i ∧
   h ≠ i) ∧
  (a * b * c = P) ∧ (d * e * f = P) ∧
  (g * h * i = P) ∧ (a * d * g = P) ∧
  (b * e * h = P) ∧ (c * f * i = P)

theorem grid_with_value_exists (P : ℕ) :
  ∃ a b c d e f g h i : ℕ, isGridValuesP P a b c d e f g h i :=
sorry

theorem possible_values (P : ℕ) :
  P ∈ [1992, 1995] ↔ 
  ∃ a b c d e f g h i : ℕ, isGridValuesP P a b c d e f g h i :=
sorry

theorem smallest_possible_value : 
  ∃ P a b c d e f g h i : ℕ, isGridValuesP P a b c d e f g h i ∧ 
  ∀ Q, (∃ w x y z u v s t q : ℕ, isGridValuesP Q w x y z u v s t q) → Q ≥ 120 :=
sorry

end grid_with_value_exists_possible_values_smallest_possible_value_l241_241620


namespace circle_area_pi_l241_241976

def circle_eq := ∀ x y : ℝ, x^2 + y^2 + 4 * x + 3 = 0 → (x + 2) ^ 2 + y ^ 2 = 1

theorem circle_area_pi (h : ∀ x y : ℝ, x^2 + y^2 + 4 * x + 3 = 0 → (x + 2) ^ 2 + y ^ 2 = 1) :
  ∃ S : ℝ, S = π :=
by {
  sorry
}

end circle_area_pi_l241_241976


namespace inverse_property_l241_241142

-- Given conditions
variable (f : ℝ → ℝ) (f_inv : ℝ → ℝ)
variable (hf_injective : Function.Injective f)
variable (hf_surjective : Function.Surjective f)
variable (h_inverse : ∀ y : ℝ, f (f_inv y) = y)
variable (hf_property : ∀ x : ℝ, f (-x) + f (x) = 3)

-- The proof goal
theorem inverse_property (x : ℝ) : (f_inv (x - 1) + f_inv (4 - x)) = 0 :=
by
  sorry

end inverse_property_l241_241142


namespace problem_a2_sum_eq_364_l241_241357

noncomputable def a (n : ℕ) : ℕ :=
  (polynomial.repr ((1 + X + X^2) ^ 6)).coeff n

theorem problem_a2_sum_eq_364 :
  (a 2) + (a 4) + (a 6) + (a 8) + (a 10) + (a 12) = 364 :=
sorry

end problem_a2_sum_eq_364_l241_241357


namespace pencils_count_l241_241839

theorem pencils_count (initial_pencils additional_pencils : ℕ) (h1 : initial_pencils = 115) (h2 : additional_pencils = 100) : initial_pencils + additional_pencils = 215 :=
by sorry

end pencils_count_l241_241839


namespace eval_expression_l241_241767

theorem eval_expression : 
  3000^3 - 2998 * 3000^2 - 2998^2 * 3000 + 2998^3 = 23992 := 
by 
  sorry

end eval_expression_l241_241767


namespace two_person_subcommittees_l241_241512

theorem two_person_subcommittees (n : ℕ) (hn : n = 8) : (finset.univ : finset (fin 8)).powerset.card = 28 :=
by
  rw hn
  -- use the fact that the number of k-combinations (sub-committees) is given by binomial coefficient
  exact (nat.choose_symm 8 2).symm ▸ nat.choose 8 2

end two_person_subcommittees_l241_241512


namespace average_weight_whole_class_l241_241433

def sectionA_students : Nat := 36
def sectionB_students : Nat := 44
def avg_weight_sectionA : Float := 40.0 
def avg_weight_sectionB : Float := 35.0
def total_weight_sectionA := avg_weight_sectionA * Float.ofNat sectionA_students
def total_weight_sectionB := avg_weight_sectionB * Float.ofNat sectionB_students
def total_students := sectionA_students + sectionB_students
def total_weight := total_weight_sectionA + total_weight_sectionB
def avg_weight_class := total_weight / Float.ofNat total_students

theorem average_weight_whole_class :
  avg_weight_class = 37.25 := by
  sorry

end average_weight_whole_class_l241_241433


namespace shaded_region_area_l241_241979

theorem shaded_region_area
  (R r : ℝ)
  (h : r^2 = R^2 - 2500)
  : π * (R^2 - r^2) = 2500 * π :=
by
  sorry

end shaded_region_area_l241_241979


namespace min_value_of_squares_l241_241789

theorem min_value_of_squares (a b c d : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : 0 < d) (h₅ : a + b + c + d = Real.sqrt 7960) : 
  a^2 + b^2 + c^2 + d^2 ≥ 1990 :=
sorry

end min_value_of_squares_l241_241789


namespace sin_675_eq_neg_sqrt2_div_2_l241_241757

axiom angle_reduction (a : ℝ) : (a - 360 * (floor (a / 360))) * π / 180 = a * π / 180 - 2 * π * (floor (a / 360))

theorem sin_675_eq_neg_sqrt2_div_2 : real.sin (675 * real.pi / 180) = - (real.sqrt 2 / 2) :=
by
  have h1: real.sin (675 * real.pi / 180) = real.sin (315 * real.pi / 180),
  { rw [← angle_reduction 675, show floor(675 / 360:ℝ) = 1 by norm_num, int.cast_one, sub_self, zero_mul, add_zero] },
  have h2: 315 = 360 - 45,
  { norm_num },
  have h3: real.sin (315 * real.pi / 180) = - real.sin (45 * real.pi / 180),
  { rw [eq_sub_of_add_eq $ show real.sin (2 * π - 45 * real.pi / 180) = -real.sin (45 * real.pi / 180) by simp] },
  rw [h1, h3, real.sin_pi_div_four],
  norm_num

end sin_675_eq_neg_sqrt2_div_2_l241_241757


namespace find_theta_l241_241714

def rectangle : Type := sorry
def angle (α : ℝ) : Prop := 0 ≤ α ∧ α < 180

-- Given conditions in the problem
variables {α β γ δ θ : ℝ}

axiom angle_10 : angle 10
axiom angle_14 : angle 14
axiom angle_33 : angle 33
axiom angle_26 : angle 26

axiom zig_zag_angles (a b c d e f : ℝ) :
  a = 26 ∧ f = 10 ∧
  26 + b = 33 ∧ b = 7 ∧
  e + 10 = 14 ∧ e = 4 ∧
  c = b ∧ d = e ∧
  θ = c + d

theorem find_theta : θ = 11 :=
sorry

end find_theta_l241_241714


namespace cost_of_popsicle_sticks_l241_241905

theorem cost_of_popsicle_sticks
  (total_money : ℕ)
  (cost_of_molds : ℕ)
  (cost_per_bottle : ℕ)
  (popsicles_per_bottle : ℕ)
  (sticks_used : ℕ)
  (sticks_left : ℕ)
  (number_of_sticks : ℕ)
  (remaining_money : ℕ) :
  total_money = 10 →
  cost_of_molds = 3 →
  cost_per_bottle = 2 →
  popsicles_per_bottle = 20 →
  sticks_left = 40 →
  number_of_sticks = 100 →
  remaining_money = total_money - cost_of_molds - (sticks_used / popsicles_per_bottle * cost_per_bottle) →
  sticks_used = number_of_sticks - sticks_left →
  remaining_money = 1 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end cost_of_popsicle_sticks_l241_241905


namespace quadratic_real_root_m_l241_241792

theorem quadratic_real_root_m (m : ℝ) (h : 4 - 4 * m ≥ 0) : m = 0 ∨ m = 2 ∨ m = 4 ∨ m = 6 ↔ m = 0 :=
by
  sorry

end quadratic_real_root_m_l241_241792


namespace g_five_eq_one_l241_241049

noncomputable def g : ℝ → ℝ := sorry

axiom g_mul (x y : ℝ) : g (x * y) = g x * g y
axiom g_zero_ne_zero : g 0 ≠ 0

theorem g_five_eq_one : g 5 = 1 := by
  sorry

end g_five_eq_one_l241_241049


namespace garage_motorcycles_l241_241408

theorem garage_motorcycles (bicycles cars motorcycles total_wheels : ℕ)
  (hb : bicycles = 20)
  (hc : cars = 10)
  (hw : total_wheels = 90)
  (wb : bicycles * 2 = 40)
  (wc : cars * 4 = 40)
  (wm : motorcycles * 2 = total_wheels - (bicycles * 2 + cars * 4)) :
  motorcycles = 5 := 
  by 
  sorry

end garage_motorcycles_l241_241408


namespace abscissa_of_point_P_l241_241000

open Real

noncomputable def hyperbola_abscissa (x y : ℝ) : Prop :=
  (x^2 - y^2 = 4) ∧
  (x > 0) ∧
  ((x + 2 * sqrt 2) * (x - 2 * sqrt 2) = -y^2)

theorem abscissa_of_point_P :
  ∃ (x y : ℝ), hyperbola_abscissa x y ∧ x = sqrt 6 := by
  sorry

end abscissa_of_point_P_l241_241000


namespace max_sequence_is_ten_l241_241579

noncomputable def max_int_sequence_length : Prop :=
  ∀ (a : ℕ → ℤ), 
    (∀ i : ℕ, a i + a (i+1) + a (i+2) + a (i+3) + a (i+4) > 0) ∧
    (∀ i : ℕ, a i + a (i+1) + a (i+2) + a (i+3) + a (i+4) + a (i+5) + a (i+6) < 0) →
    (∃ n ≤ 10, ∀ i ≥ n, a i = 0)

theorem max_sequence_is_ten : max_int_sequence_length :=
sorry

end max_sequence_is_ten_l241_241579


namespace dave_fifth_store_car_count_l241_241726

theorem dave_fifth_store_car_count :
  let cars_first_store := 30
  let cars_second_store := 14
  let cars_third_store := 14
  let cars_fourth_store := 21
  let mean := 20.8
  let total_cars := mean * 5
  let total_cars_first_four := cars_first_store + cars_second_store + cars_third_store + cars_fourth_store
  total_cars - total_cars_first_four = 25 := by
sorry

end dave_fifth_store_car_count_l241_241726


namespace students_answered_both_correctly_l241_241589

theorem students_answered_both_correctly
  (enrolled : ℕ)
  (did_not_take_test : ℕ)
  (answered_q1_correctly : ℕ)
  (answered_q2_correctly : ℕ)
  (total_students_answered_both : ℕ) :
  enrolled = 29 →
  did_not_take_test = 5 →
  answered_q1_correctly = 19 →
  answered_q2_correctly = 24 →
  total_students_answered_both = 19 :=
by
  intros
  sorry

end students_answered_both_correctly_l241_241589


namespace carrots_picked_by_Carol_l241_241902

theorem carrots_picked_by_Carol (total_carrots mom_carrots : ℕ) (h1 : total_carrots = 38 + 7) (h2 : mom_carrots = 16) :
  total_carrots - mom_carrots = 29 :=
by {
  sorry
}

end carrots_picked_by_Carol_l241_241902


namespace expression_equals_33_l241_241900

noncomputable def calculate_expression : ℚ :=
  let part1 := 25 * 52
  let part2 := 46 * 15
  let diff := part1 - part2
  (2013 / diff) * 10

theorem expression_equals_33 : calculate_expression = 33 := sorry

end expression_equals_33_l241_241900


namespace polynomial_roots_l241_241405

-- The statement that we need to prove
theorem polynomial_roots (a b : ℚ) (h : (2 + Real.sqrt 3) ^ 3 + 4 * (2 + Real.sqrt 3) ^ 2 + a * (2 + Real.sqrt 3) + b = 0) :
  ((Polynomial.C a * Polynomial.X ^ 3 + Polynomial.C (4 : ℚ) * Polynomial.X ^ 2 + Polynomial.C a * Polynomial.X + Polynomial.C b) = Polynomial.X ^ 3 + Polynomial.C (4 : ℚ) * Polynomial.X ^ 2 + Polynomial.C a * Polynomial.X + Polynomial.C b) →
  (2 - Real.sqrt 3) ^ 3 + 4 * (2 - Real.sqrt 3) ^ 2 + a * (2 - Real.sqrt 3) + b = 0 ∧ -8 ^ 3 + 4 * (-8) ^ 2 + a * (-8) + b = 0 := sorry

end polynomial_roots_l241_241405


namespace system_of_equations_implies_quadratic_l241_241109

theorem system_of_equations_implies_quadratic (x y : ℝ) :
  (3 * x^2 + 9 * x + 4 * y + 2 = 0) ∧ (3 * x + y + 4 = 0) → (y^2 + 11 * y - 14 = 0) := by
  sorry

end system_of_equations_implies_quadratic_l241_241109


namespace induction_proof_l241_241036

open Nat

noncomputable def S (n : ℕ) : ℚ :=
  match n with
  | 0     => 0
  | (n+1) => S n + 1 / ((n+1) * (n+2))

theorem induction_proof : ∀ n : ℕ, S n = n / (n + 1) := by
  intro n
  induction n with
  | zero => 
    -- Base case: S(1) = 1/2
    sorry
  | succ n ih =>
    -- Induction step: Assume S(n) = n / (n + 1), prove S(n+1) = (n+1) / (n+2)
    sorry

end induction_proof_l241_241036


namespace triangle_is_isosceles_right_l241_241793

theorem triangle_is_isosceles_right (a b S : ℝ) (h : S = (1/4) * (a^2 + b^2)) :
  ∃ C : ℝ, C = 90 ∧ a = b :=
by
  sorry

end triangle_is_isosceles_right_l241_241793


namespace find_minimum_value_of_quadratic_l241_241342

theorem find_minimum_value_of_quadratic :
  ∀ (x : ℝ), (x = 5/2) -> (∀ y, y = 3 * x ^ 2 - 15 * x + 7 -> ∀ z, z ≥ y) := 
sorry

end find_minimum_value_of_quadratic_l241_241342


namespace find_functions_l241_241933

theorem find_functions (M N : ℝ × ℝ)
  (hM : M.fst = -4) (hM_quad2 : 0 < M.snd)
  (hN : N = (-6, 0))
  (h_area : 1 / 2 * 6 * M.snd = 15) :
  (∃ k, ∀ x, (M = (-4, 5) → N = (-6, 0) → x * k = -5 / 4 * x)) ∧ 
  (∃ a b, ∀ x, (M = (-4, 5) → N = (-6, 0) → x * a + b = 5 / 2 * x + 15)) := 
sorry

end find_functions_l241_241933


namespace store_discount_percentage_l241_241971

theorem store_discount_percentage
  (total_without_discount : ℝ := 350)
  (final_price : ℝ := 252)
  (coupon_percentage : ℝ := 0.1) :
  ∃ (x : ℝ), total_without_discount * (1 - x / 100) * (1 - coupon_percentage) = final_price ∧ x = 20 :=
by
  use 20
  sorry

end store_discount_percentage_l241_241971


namespace part_I_solution_set_part_II_solution_range_l241_241146

-- Part I: Defining the function and proving the solution set for m = 3
def f (x m : ℝ) : ℝ := |x + 1| + |m - x|

theorem part_I_solution_set (x : ℝ) :
  (f x 3 ≥ 6) ↔ (x ≤ -2 ∨ x ≥ 4) :=
sorry

-- Part II: Proving the range of values for m such that f(x) ≥ 8 for any real number x
theorem part_II_solution_range (m : ℝ) :
  (∀ x : ℝ, f x m ≥ 8) ↔ (m ≤ -9 ∨ m ≥ 7) :=
sorry

end part_I_solution_set_part_II_solution_range_l241_241146


namespace BD_length_l241_241528

theorem BD_length
  (A B C D : Type)
  (dist_AC : ℝ := 10)
  (dist_BC : ℝ := 10)
  (dist_AD : ℝ := 12)
  (dist_CD : ℝ := 5) : (BD : ℝ) = 95 / 12 :=
by
  sorry

end BD_length_l241_241528


namespace chocolates_divisible_l241_241071

theorem chocolates_divisible (n m : ℕ) (h1 : n > 0) (h2 : m > 0) : 
  (n ≤ m) ∨ (m % (n - m) = 0) :=
sorry

end chocolates_divisible_l241_241071


namespace two_baskets_of_peaches_l241_241993

theorem two_baskets_of_peaches (R G : ℕ) (h1 : G = R + 2) (h2 : 2 * R + 2 * G = 12) : R = 2 :=
by
  sorry

end two_baskets_of_peaches_l241_241993


namespace determine_k_l241_241625

theorem determine_k (k : ℝ) (h : 2 - 2^2 = k * (2)^2 + 1) : k = -3/4 :=
by
  sorry

end determine_k_l241_241625


namespace binom_12_10_eq_66_l241_241468

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_12_10_eq_66 : binom 12 10 = 66 := 
by
  -- We state the given condition for binomial coefficients.
  let binom_coeff := binom 12 10
  -- We will use the formula \binom{n}{k} = \frac{n!}{k!(n-k)!}.
  change binom_coeff = Nat.factorial 12 / (Nat.factorial 10 * Nat.factorial (12 - 10))
  -- Simplify using factorial values and basic arithmetic to reach the conclusion.
  sorry

end binom_12_10_eq_66_l241_241468


namespace equivalent_problem_l241_241827

variable (x y : ℝ)
variable (hx_ne_zero : x ≠ 0)
variable (hy_ne_zero : y ≠ 0)
variable (h : (3 * x + y) / (x - 3 * y) = -2)

theorem equivalent_problem : (x + 3 * y) / (3 * x - y) = 2 :=
by
  sorry

end equivalent_problem_l241_241827


namespace sale_price_same_as_original_l241_241575

theorem sale_price_same_as_original (x : ℝ) :
  let increased_price := 1.25 * x
  let sale_price := 0.8 * increased_price
  sale_price = x := 
by
  let increased_price := 1.25 * x
  let sale_price := 0.8 * increased_price
  sorry

end sale_price_same_as_original_l241_241575


namespace arithmetic_sequence_sum_l241_241099

-- Condition definitions
def a : Int := 3
def d : Int := 2
def a_n : Int := 25
def n : Int := 12

-- Sum formula for an arithmetic sequence proof
theorem arithmetic_sequence_sum :
    let n := 12
    let S_n := (n * (a + a_n)) / 2
    S_n = 168 := by
  sorry

end arithmetic_sequence_sum_l241_241099


namespace people_from_second_row_joined_l241_241843

theorem people_from_second_row_joined
  (initial_first_row : ℕ) (initial_second_row : ℕ) (initial_third_row : ℕ) (people_waded : ℕ) (remaining_people : ℕ)
  (H1 : initial_first_row = 24)
  (H2 : initial_second_row = 20)
  (H3 : initial_third_row = 18)
  (H4 : people_waded = 3)
  (H5 : remaining_people = 54) :
  initial_second_row - (initial_first_row + initial_second_row + initial_third_row - initial_first_row - people_waded - remaining_people) = 5 :=
by
  sorry

end people_from_second_row_joined_l241_241843


namespace minimize_expression_l241_241236

theorem minimize_expression (x : ℝ) : 
  ∃ (m : ℝ), m = 2023 ∧ ∀ x, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ m :=
sorry

end minimize_expression_l241_241236


namespace probability_factor_of_36_l241_241268

theorem probability_factor_of_36 :
  let N := 36
  let S := {n : ℕ | n > 0 ∧ n ≤ N}
  let E := {n : ℕ | n ∈ S ∧ N % n = 0}
  (E.card : ℚ) / S.card = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l241_241268


namespace table_length_l241_241886

theorem table_length (area_m2 : ℕ) (width_cm : ℕ) (length_cm : ℕ) 
  (h_area : area_m2 = 54)
  (h_width : width_cm = 600)
  :
  length_cm = 900 := 
  sorry

end table_length_l241_241886


namespace segment_ratios_l241_241680

theorem segment_ratios 
  (AB_parts BC_parts : ℝ) 
  (hAB: AB_parts = 3) 
  (hBC: BC_parts = 4) 
  : AB_parts / (AB_parts + BC_parts) = 3 / 7 ∧ BC_parts / (AB_parts + BC_parts) = 4 / 7 := 
  sorry

end segment_ratios_l241_241680


namespace pencil_count_l241_241840

/-- 
If there are initially 115 pencils in the drawer, and Sara adds 100 more pencils, 
then the total number of pencils in the drawer is 215.
-/
theorem pencil_count (initial_pencils added_pencils : ℕ) (h1 : initial_pencils = 115) (h2 : added_pencils = 100) : 
  initial_pencils + added_pencils = 215 := by
  sorry

end pencil_count_l241_241840


namespace jacob_find_more_l241_241111

theorem jacob_find_more :
  let initial_shells := 2
  let ed_limpet_shells := 7
  let ed_oyster_shells := 2
  let ed_conch_shells := 4
  let total_shells := 30
  let ed_shells := ed_limpet_shells + ed_oyster_shells + ed_conch_shells + initial_shells
  let jacob_shells := total_shells - ed_shells
  (jacob_shells - ed_limpet_shells - ed_oyster_shells - ed_conch_shells = 2) := 
by 
  sorry

end jacob_find_more_l241_241111


namespace maximize_profit_l241_241209

def cost_price_A (x y : ℕ) := x = y + 20
def cost_sum_eq_200 (x y : ℕ) := x + 2 * y = 200
def linear_function (m n : ℕ) := m = -((1/2) : ℚ) * n + 90
def profit_function (w n : ℕ) : ℚ := (-((1/2) : ℚ) * ((n : ℚ) - 130)^2) + 1250

theorem maximize_profit
  (x y m n : ℕ)
  (hx : cost_price_A x y)
  (hsum : cost_sum_eq_200 x y)
  (hlin : linear_function m n)
  (hmaxn : 80 ≤ n ∧ n ≤ 120)
  : y = 60 ∧ x = 80 ∧ n = 120 ∧ profit_function 120 120 = 1200 := 
sorry

end maximize_profit_l241_241209


namespace probability_factor_of_36_is_1_over_4_l241_241271

def totalPositiveIntegers := 36

def isDivisor (n: Nat) : Prop := ∃ (a b: Nat), (0 ≤ a ∧ a ≤ 2) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (n = 2^a * 3^b)

def numDivisors := 9
def probability := (numDivisors : ℚ) / (totalPositiveIntegers : ℚ)

theorem probability_factor_of_36_is_1_over_4 :
  probability = (1 / 4 : ℚ) :=
sorry

end probability_factor_of_36_is_1_over_4_l241_241271


namespace total_students_l241_241064

-- Define two-thirds of the class having brown eyes
def brown_eyes (T : ℤ) : Prop :=
  (2 / 3 : ℤ) * T = 2 * T / 3

-- Define half of the students with brown eyes have black hair
def black_hair_given_brown_eyes (T : ℤ) : Prop :=
  (1 / 2 : ℤ) * (2 * T / 3) = (1 / 3 : ℤ) * T

-- Given 6 students have brown eyes and black hair
def students_with_brown_eyes_black_hair : Prop :=
  (1 / 3 : ℤ) * ?m_1 = 6

-- The theorem to prove
theorem total_students (T : ℤ) (H1 : brown_eyes T) (H2 : black_hair_given_brown_eyes T) (H3 : students_with_brown_eyes_black_hair T) : T = 18 := by
  sorry

end total_students_l241_241064


namespace proteges_57_l241_241595

def divisors (n : ℕ) : List ℕ := (List.range (n + 1)).filter (λ d => n % d = 0)

def units_digit (n : ℕ) : ℕ := n % 10

def proteges (n : ℕ) : List ℕ := (divisors n).map units_digit

theorem proteges_57 : proteges 57 = [1, 3, 9, 7] :=
sorry

end proteges_57_l241_241595


namespace min_value_of_expression_l241_241534

noncomputable def min_value_expression (a b c d : ℝ) : ℝ :=
  (a - 2)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (d / c - 1)^2

theorem min_value_of_expression (a b c d : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≥ 2) :
  min_value_expression a b c d = 1 / 4 :=
sorry

end min_value_of_expression_l241_241534


namespace restaurant_total_dishes_l241_241867

noncomputable def total_couscous_received : ℝ := 15.4 + 45
noncomputable def total_chickpeas_received : ℝ := 19.8 + 33

-- Week 1, ratio of 5:3 (couscous:chickpeas)
noncomputable def sets_of_ratio_week1_couscous : ℝ := total_couscous_received / 5
noncomputable def sets_of_ratio_week1_chickpeas : ℝ := total_chickpeas_received / 3
noncomputable def dishes_week1 : ℝ := min sets_of_ratio_week1_couscous sets_of_ratio_week1_chickpeas

-- Week 2, ratio of 3:2 (couscous:chickpeas)
noncomputable def sets_of_ratio_week2_couscous : ℝ := total_couscous_received / 3
noncomputable def sets_of_ratio_week2_chickpeas : ℝ := total_chickpeas_received / 2
noncomputable def dishes_week2 : ℝ := min sets_of_ratio_week2_couscous sets_of_ratio_week2_chickpeas

-- Total dishes rounded down
noncomputable def total_dishes : ℝ := dishes_week1 + dishes_week2

theorem restaurant_total_dishes :
  ⌊total_dishes⌋ = 32 :=
by {
  sorry
}

end restaurant_total_dishes_l241_241867


namespace binomial_12_10_eq_66_l241_241459

theorem binomial_12_10_eq_66 : (Nat.choose 12 10) = 66 :=
by
  sorry

end binomial_12_10_eq_66_l241_241459


namespace estimate_students_scores_l241_241443

noncomputable def students_scoring_within_interval 
  (xi : ℝ → ℝ) 
  (N : ℕ)
  (mu : ℝ) 
  (sigma : ℝ) 
  (P_interval1 : ℝ) 
  (P_interval2 : ℝ) : ℝ :=
100000 * (P_interval2 - P_interval1)

theorem estimate_students_scores 
  : students_scoring_within_interval 
      (Normal 70 25) 
      100000 
      70 
      5 
      0.3413 
      0.4772 = 13590 :=
sorry

end estimate_students_scores_l241_241443


namespace total_students_l241_241063

theorem total_students (total_students_with_brown_eyes total_students_with_black_hair: ℕ)
    (h1: ∀ (total_students : ℕ), (2 * total_students_with_brown_eyes) = 3 * total_students)
    (h2: (2 * total_students_with_black_hair) = total_students_with_brown_eyes)
    (h3: total_students_with_black_hair = 6) : 
    ∃ total_students : ℕ, total_students = 18 :=
by
  sorry

end total_students_l241_241063


namespace probability_factor_of_36_l241_241293

theorem probability_factor_of_36 : (∃ p : ℚ, p = 1 / 4 ∧
  let n := 36 in
  let factors := {d | d ∣ n ∧ d > 0 ∧ d ≤ n} in
  let total_pos_ints := 36 in
  let num_factors := set.card factors in
  p = num_factors / total_pos_ints) :=
begin
  sorry
end

end probability_factor_of_36_l241_241293


namespace find_fraction_l241_241963

theorem find_fraction (a b : ℝ) (h : a ≠ b) (h_eq : a / b + (a + 20 * b) / (b + 20 * a) = 3) : a / b = 0.33 :=
sorry

end find_fraction_l241_241963


namespace triangle_acute_angle_contradiction_l241_241713

theorem triangle_acute_angle_contradiction
  (α β γ : ℝ)
  (h_sum : α + β + γ = 180)
  (h_tri : 0 < α ∧ 0 < β ∧ 0 < γ)
  (h_at_most_one_acute : (α < 90 ∧ β ≥ 90 ∧ γ ≥ 90) 
                         ∨ (α ≥ 90 ∧ β < 90 ∧ γ ≥ 90) 
                         ∨ (α ≥ 90 ∧ β ≥ 90 ∧ γ < 90)) :
  false :=
by
  sorry

end triangle_acute_angle_contradiction_l241_241713


namespace factor_probability_36_l241_241289

-- Definitions based on conditions
def is_factor (n d : ℕ) : Prop := d ∣ n
def num_factors (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (is_factor n).card

def probability_factor (n : ℕ) : ℚ := num_factors n / n

-- Problem statement
theorem factor_probability_36 : probability_factor 36 = 1 / 4 :=
  by sorry

end factor_probability_36_l241_241289


namespace unique_digit_sum_l241_241177

theorem unique_digit_sum (X Y M Z F : ℕ) (H1 : X ≠ 0) (H2 : Y ≠ 0) (H3 : M ≠ 0) (H4 : Z ≠ 0) (H5 : F ≠ 0)
  (H6 : X ≠ Y) (H7 : X ≠ M) (H8 : X ≠ Z) (H9 : X ≠ F)
  (H10 : Y ≠ M) (H11 : Y ≠ Z) (H12 : Y ≠ F)
  (H13 : M ≠ Z) (H14 : M ≠ F)
  (H15 : Z ≠ F)
  (H16 : 10 * X + Y ≠ 0) (H17 : 10 * M + Z ≠ 0)
  (H18 : 111 * F = (10 * X + Y) * (10 * M + Z)) :
  X + Y + M + Z + F = 28 := by
  sorry

end unique_digit_sum_l241_241177


namespace coordinates_OQ_quadrilateral_area_range_l241_241502

variables {p : ℝ} (p_pos : 0 < p)
variables {x0 x1 x2 y0 y1 y2 : ℝ} (h_parabola_A : y1^2 = 2*p*x1) (h_parabola_B : y2^2 = 2*p*x2) (h_parabola_M : y0^2 = 2*p*x0)
variables {a : ℝ} (h_focus_x : a = x0 + p) 

variables {FA FM FB : ℝ}
variables (h_arith_seq : ( FM = FA - (FA - FB) / 2 ))

-- Step 1: Prove the coordinates of OQ
theorem coordinates_OQ : (x0 + p, 0) = (a, 0) :=
by
  -- proof will be completed here
  sorry 

variables {x0_val : ℝ} (x0_eq : x0 = 2) {FM_val : ℝ} (FM_eq : FM = 5 / 2)

-- Step 2: Prove the area range of quadrilateral ABB1A1
theorem quadrilateral_area_range : ∀ (p : ℝ), 0 < p →
  ∀ (x0 x1 x2 y1 y2 FM OQ : ℝ), 
    x0 = 2 → FM = 5 / 2 → OQ = 3 → (y1^2 = 2*p*x1) → (y2^2 = 2*p*x2) →
  ( ∃ S : ℝ, 0 < S ∧ S ≤ 10) :=
by
  -- proof will be completed here
  sorry 

end coordinates_OQ_quadrilateral_area_range_l241_241502


namespace evaluate_expression_l241_241768

theorem evaluate_expression : 
  let expr := (15 / 8) ^ 2
  let ceil_expr := Nat.ceil expr
  let mult_expr := ceil_expr * (21 / 5)
  Nat.floor mult_expr = 16 := by
  sorry

end evaluate_expression_l241_241768


namespace fraction_sent_afternoon_l241_241586

-- Defining the problem conditions
def total_fliers : ℕ := 1000
def fliers_sent_morning : ℕ := total_fliers * 1/5
def fliers_left_afternoon : ℕ := total_fliers - fliers_sent_morning
def fliers_left_next_day : ℕ := 600
def fliers_sent_afternoon : ℕ := fliers_left_afternoon - fliers_left_next_day

-- Proving the fraction of fliers sent in the afternoon
theorem fraction_sent_afternoon : (fliers_sent_afternoon : ℚ) / fliers_left_afternoon = 1/4 :=
by
  -- proof goes here
  sorry

end fraction_sent_afternoon_l241_241586


namespace two_person_subcommittees_l241_241511

theorem two_person_subcommittees (n : ℕ) (hn : n = 8) : (finset.univ : finset (fin 8)).powerset.card = 28 :=
by
  rw hn
  -- use the fact that the number of k-combinations (sub-committees) is given by binomial coefficient
  exact (nat.choose_symm 8 2).symm ▸ nat.choose 8 2

end two_person_subcommittees_l241_241511


namespace increase_in_area_l241_241061

theorem increase_in_area :
  let original_side := 6
  let increase := 1
  let new_side := original_side + increase
  let original_area := original_side * original_side
  let new_area := new_side * new_side
  let area_increase := new_area - original_area
  area_increase = 13 :=
by
  let original_side := 6
  let increase := 1
  let new_side := original_side + increase
  let original_area := original_side * original_side
  let new_area := new_side * new_side
  let area_increase := new_area - original_area
  sorry

end increase_in_area_l241_241061


namespace product_of_roots_l241_241536

theorem product_of_roots (Q : Polynomial ℚ) (hQ : Q.degree = 1) (h_root : Q.eval 6 = 0) :
  (Q.roots : Multiset ℚ).prod = 6 :=
sorry

end product_of_roots_l241_241536


namespace fibonacci_sum_of_squares_l241_241184

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_sum_of_squares (n : ℕ) (hn : n ≥ 1) :
  (Finset.range n).sum (λ i => (fibonacci (i + 1))^2) = fibonacci n * fibonacci (n + 1) :=
sorry

end fibonacci_sum_of_squares_l241_241184


namespace total_animal_count_l241_241060

theorem total_animal_count (initial_hippos : ℕ) (initial_elephants : ℕ) 
  (female_hippo_factor : ℚ) (newborn_per_female_hippo : ℕ) 
  (extra_newborn_elephants : ℕ) 
  (h_initial_hippos : initial_hippos = 35)
  (h_initial_elephants : initial_elephants = 20)
  (h_female_hippo_factor : female_hippo_factor = 5 / 7)
  (h_newborn_per_female_hippo : newborn_per_female_hippo = 5)
  (h_extra_newborn_elephants : extra_newborn_elephants = 10) : 
  (initial_elephants + initial_hippos + 
  (initial_hippos * female_hippo_factor).to_nat * newborn_per_female_hippo + 
  (initial_hippos * female_hippo_factor).to_nat * newborn_per_female_hippo + 
  extra_newborn_elephants) = 315 :=
by
  sorry

end total_animal_count_l241_241060


namespace area_of_shaded_region_l241_241978

-- Define the conditions
def concentric_circles (O : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  r1 < r2 ∧ ∀ P, (P.1 - O.1)^2 + (P.2 - O.2)^2 = r1^2 → (P.1 - O.1)^2 + (P.2 - O.2)^2 = r2^2

-- Define the lengths and given properties
def chord_tangent_smaller_circle (O A B : ℝ × ℝ) (AB_length : ℝ) (r1 : ℝ) : Prop :=
  ∥A - B∥ = AB_length ∧ ∥A - O∥ = r1 ∧ ∥B - O∥ = r1 ∧
  let P := (A + B) / 2 in
  ∥P - O∥ = r1 ∧ ∥A - P∥ = AB_length / 2

-- Main theorem
theorem area_of_shaded_region
  (O A B : ℝ × ℝ) (r1 r2 : ℝ) (AB_length : ℝ)
  (hcc : concentric_circles O r1 r2)
  (hct : chord_tangent_smaller_circle O A B AB_length r1) :
  π * (r2^2 - r1^2) = 2500 * π :=
by
  sorry

end area_of_shaded_region_l241_241978


namespace log_comparison_l241_241188

open Real

noncomputable def a := log 4 / log 5  -- a = log_5(4)
noncomputable def b := log 5 / log 3  -- b = log_3(5)
noncomputable def c := log 5 / log 4  -- c = log_4(5)

theorem log_comparison : a < c ∧ c < b := 
by
  sorry

end log_comparison_l241_241188


namespace JackEmails_l241_241664

theorem JackEmails (E : ℕ) (h1 : 10 = E + 7) : E = 3 :=
by
  sorry

end JackEmails_l241_241664


namespace find_a_minus_c_l241_241733

section
variables (a b c : ℝ)
variables (h₁ : (a + b) / 2 = 110) (h₂ : (b + c) / 2 = 170)

theorem find_a_minus_c : a - c = -120 :=
by
  sorry
end

end find_a_minus_c_l241_241733


namespace remainder_when_divided_by_8_l241_241066

theorem remainder_when_divided_by_8:
  ∀ (n : ℕ), (∃ (q : ℕ), n = 7 * q + 5) → n % 8 = 1 :=
by
  intro n h
  rcases h with ⟨q, hq⟩
  sorry

end remainder_when_divided_by_8_l241_241066


namespace jerry_remaining_debt_l241_241669

theorem jerry_remaining_debt :
  ∀ (paid_two_months_ago paid_last_month total_debt: ℕ),
  paid_two_months_ago = 12 →
  paid_last_month = paid_two_months_ago + 3 →
  total_debt = 50 →
  total_debt - (paid_two_months_ago + paid_last_month) = 23 :=
by
  intros paid_two_months_ago paid_last_month total_debt h1 h2 h3
  sorry

end jerry_remaining_debt_l241_241669


namespace album_count_l241_241319

theorem album_count (A B S : ℕ) (hA : A = 23) (hB : B = 9) (hS : S = 15) : 
  (A - S) + B = 17 :=
by
  -- Variables and conditions
  have Andrew_unique : ℕ := A - S
  have Bella_unique : ℕ := B
  -- Proof starts here
  sorry

end album_count_l241_241319


namespace nested_sqrt_eq_two_l241_241517

theorem nested_sqrt_eq_two (x : ℝ) (h : x = Real.sqrt (2 + x)) : x = 2 :=
sorry

end nested_sqrt_eq_two_l241_241517


namespace sum_first_60_natural_numbers_l241_241303

theorem sum_first_60_natural_numbers : (60 * (60 + 1)) / 2 = 1830 := by
  sorry

end sum_first_60_natural_numbers_l241_241303


namespace intersect_curves_l241_241053

theorem intersect_curves (R : ℝ) (hR : R > 0) :
  (∃ (x y : ℝ), x^2 + y^2 = R^2 ∧ x - y - 2 = 0) ↔ R ≥ Real.sqrt 2 :=
sorry

end intersect_curves_l241_241053


namespace similar_triangles_y_value_l241_241890

noncomputable def y_value := 9.33

theorem similar_triangles_y_value (y : ℝ) 
    (h : ∃ (a b : ℝ), a = 12 ∧ b = 9 ∧ (a / y = b / 7)) : y = y_value := 
  sorry

end similar_triangles_y_value_l241_241890


namespace hourly_rate_is_7_l241_241552

-- Define the fixed fee, the total payment, and the number of hours
def fixed_fee : ℕ := 17
def total_payment : ℕ := 80
def num_hours : ℕ := 9

-- Define the function calculating the hourly rate based on the given conditions
def hourly_rate (fixed_fee total_payment num_hours : ℕ) : ℕ :=
  (total_payment - fixed_fee) / num_hours

-- Prove that the hourly rate is 7 dollars per hour
theorem hourly_rate_is_7 :
  hourly_rate fixed_fee total_payment num_hours = 7 := 
by 
  -- proof is skipped
  sorry

end hourly_rate_is_7_l241_241552


namespace probability_path_from_first_to_last_floor_open_doors_l241_241871

noncomputable
def probability_path_possible (n : ℕ) : ℚ :=
  (2 ^ (n - 1)) / (nat.choose (2 * (n - 1)) (n - 1))

theorem probability_path_from_first_to_last_floor_open_doors (n : ℕ) :
  probability_path_possible n = (2 ^ (n - 1)) / (nat.choose (2 * (n - 1)) (n - 1)) :=
by
  sorry

end probability_path_from_first_to_last_floor_open_doors_l241_241871


namespace find_m_l241_241700

theorem find_m (m : ℝ) : (m + 2) * (m - 2) + 3 * m * (m + 2) = 0 ↔ m = 1/2 ∨ m = -2 :=
by
  sorry

end find_m_l241_241700


namespace increasing_interval_implication_l241_241361

theorem increasing_interval_implication (a : ℝ) :
  (∀ x ∈ Set.Ioo (1 / 2) 2, (1 / x + 2 * a * x > 0)) → a > -1 / 8 :=
by
  intro h
  sorry

end increasing_interval_implication_l241_241361


namespace division_multiplication_calculation_l241_241320

theorem division_multiplication_calculation :
  (30 / (7 + 2 - 3)) * 4 = 20 :=
by
  sorry

end division_multiplication_calculation_l241_241320


namespace smallest_integer_form_l241_241719

theorem smallest_integer_form (m n : ℤ) : ∃ (a : ℤ), a = 2011 * m + 55555 * n ∧ a > 0 → a = 1 :=
by
  sorry

end smallest_integer_form_l241_241719


namespace tangent_circumcircle_bisects_EC_l241_241887

open EuclideanGeometry

variables {A B C D E : Point}

theorem tangent_circumcircle_bisects_EC
  (ABC_isosceles : is_isosceles_triangle A B C)
  (D_on_BC : lies_on D (line_through B C))
  (E_on_parallel_B_AC : ∃ l, is_parallel l (line_through A C) ∧ E = intersection (line_through B l) (line_through A D))
  (circumcircle_ABD : ∃ O, is_circumcircle_of O A B D) :
  ∃ F, is_tangent_at F (circumcircle_ABD), is_midpoint F E C :=
by
  sorry

end tangent_circumcircle_bisects_EC_l241_241887


namespace ratio_of_area_of_inscribed_circle_to_triangle_l241_241832

theorem ratio_of_area_of_inscribed_circle_to_triangle (h r : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  let a := (3 / 5) * h
  let b := (4 / 5) * h
  let A := (1 / 2) * a * b
  let s := (a + b + h) / 2
  (π * r) / s = (5 * π * r) / (12 * h) :=
by
  let a := (3 / 5) * h
  let b := (4 / 5) * h
  let A := (1 / 2) * a * b
  let s := (a + b + h) / 2
  sorry

end ratio_of_area_of_inscribed_circle_to_triangle_l241_241832


namespace center_of_circle_from_diameter_l241_241400

theorem center_of_circle_from_diameter (x1 y1 x2 y2 : ℝ) 
  (h1 : x1 = 3) (h2 : y1 = -3) (h3 : x2 = 13) (h4 : y2 = 17) :
  (x1 + x2) / 2 = 8 ∧ (y1 + y2) / 2 = 7 :=
by
  sorry

end center_of_circle_from_diameter_l241_241400


namespace mike_profit_l241_241547

theorem mike_profit 
  (num_acres_bought : ℕ) (price_per_acre_buy : ℤ) 
  (fraction_sold : ℚ) (price_per_acre_sell : ℤ) :
  num_acres_bought = 200 →
  price_per_acre_buy = 70 →
  fraction_sold = 1/2 →
  price_per_acre_sell = 200 →
  let cost_of_land := price_per_acre_buy * num_acres_bought,
      num_acres_sold := (fraction_sold * num_acres_bought),
      revenue_from_sale := price_per_acre_sell * num_acres_sold,
      profit := revenue_from_sale - cost_of_land
  in profit = 6000 := by
  intros h1 h2 h3 h4
  let cost_of_land := price_per_acre_buy * num_acres_bought
  let num_acres_sold := (fraction_sold * num_acres_bought)
  let revenue_from_sale := price_per_acre_sell * num_acres_sold
  let profit := revenue_from_sale - cost_of_land
  rw [h1, h2, h3, h4]
  sorry

end mike_profit_l241_241547


namespace profit_is_correct_l241_241072

-- Definitions of the conditions
def initial_outlay : ℕ := 10000
def cost_per_set : ℕ := 20
def price_per_set : ℕ := 50
def sets_sold : ℕ := 500

-- Derived calculations
def revenue (sets_sold : ℕ) (price_per_set : ℕ) : ℕ :=
  sets_sold * price_per_set

def manufacturing_costs (initial_outlay : ℕ) (cost_per_set : ℕ) (sets_sold : ℕ) : ℕ :=
  initial_outlay + (cost_per_set * sets_sold)

def profit (revenue : ℕ) (manufacturing_costs : ℕ) : ℕ :=
  revenue - manufacturing_costs

-- Theorem stating the problem
theorem profit_is_correct : 
  profit (revenue sets_sold price_per_set) (manufacturing_costs initial_outlay cost_per_set sets_sold) = 5000 :=
by
  sorry

end profit_is_correct_l241_241072


namespace incorrect_expression_among_options_l241_241723

theorem incorrect_expression_among_options :
  ¬(0.75 ^ (-0.3) < 0.75 ^ (0.1)) :=
by
  sorry

end incorrect_expression_among_options_l241_241723


namespace sum_of_x_and_y_l241_241703

theorem sum_of_x_and_y (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y)
    (hx : ∃ (a : ℕ), 720 * x = a^2)
    (hy : ∃ (b : ℕ), 720 * y = b^4) :
    x + y = 1130 :=
sorry

end sum_of_x_and_y_l241_241703


namespace total_savings_l241_241489

-- Define the given conditions
def number_of_tires : ℕ := 4
def sale_price : ℕ := 75
def original_price : ℕ := 84

-- State the proof problem
theorem total_savings : (original_price - sale_price) * number_of_tires = 36 :=
by
  -- Proof omitted
  sorry

end total_savings_l241_241489


namespace triangle_area_inscribed_rectangle_area_l241_241052

theorem triangle_area (m n : ℝ) : ∃ (S : ℝ), S = m * n := 
sorry

theorem inscribed_rectangle_area (m n : ℝ) : ∃ (A : ℝ), A = (2 * m^2 * n^2) / (m + n)^2 :=
sorry

end triangle_area_inscribed_rectangle_area_l241_241052


namespace mike_profit_l241_241542

-- Definition of initial conditions
def acres_bought := 200
def cost_per_acre := 70
def fraction_sold := 1 / 2
def selling_price_per_acre := 200

-- Definitions derived from conditions
def total_cost := acres_bought * cost_per_acre
def acres_sold := acres_bought * fraction_sold
def total_revenue := acres_sold * selling_price_per_acre
def profit := total_revenue - total_cost

-- Theorem stating the question and answer tuple
theorem mike_profit : profit = 6000 := by
  -- Proof omitted
  sorry

end mike_profit_l241_241542


namespace probability_of_two_green_apples_l241_241372

theorem probability_of_two_green_apples (total_apples green_apples choose_apples : ℕ)
  (h_total : total_apples = 8)
  (h_green : green_apples = 4)
  (h_choose : choose_apples = 2) 
: (Nat.choose green_apples choose_apples : ℚ) / (Nat.choose total_apples choose_apples) = 3 / 14 := 
by
  -- This part we would provide a proof, but for now we will use sorry
  sorry

end probability_of_two_green_apples_l241_241372


namespace probability_two_white_balls_l241_241083

noncomputable def probability_of_two_white_balls (total_balls white_balls black_balls: ℕ) : ℚ :=
  if white_balls + black_balls = total_balls ∧ total_balls = 15 ∧ white_balls = 7 ∧ black_balls = 8 then
    (white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1))
  else 0

theorem probability_two_white_balls : 
  probability_of_two_white_balls 15 7 8 = 1/5
:= sorry

end probability_two_white_balls_l241_241083


namespace more_girls_than_boys_l241_241659

theorem more_girls_than_boys (girls boys total_pupils : ℕ) (h1 : girls = 692) (h2 : total_pupils = 926) (h3 : boys = total_pupils - girls) : girls - boys = 458 :=
by
  sorry

end more_girls_than_boys_l241_241659


namespace find_n_l241_241434

theorem find_n (n : ℕ) (h : n > 0) : 
  (3^n + 5^n) % (3^(n-1) + 5^(n-1)) = 0 ↔ n = 1 := 
by sorry

end find_n_l241_241434


namespace eq1_solution_eq2_solution_l241_241825

theorem eq1_solution (x : ℝ) : (3 * x * (x - 1) = 2 - 2 * x) ↔ (x = 1 ∨ x = -2/3) :=
sorry

theorem eq2_solution (x : ℝ) : (3 * x^2 - 6 * x + 2 = 0) ↔ (x = 1 + (Real.sqrt 3) / 3 ∨ x = 1 - (Real.sqrt 3) / 3) :=
sorry

end eq1_solution_eq2_solution_l241_241825


namespace solve_for_y_l241_241040

theorem solve_for_y (y : ℝ) : 7 - y = 4 → y = 3 :=
by
  sorry

end solve_for_y_l241_241040


namespace cost_of_15_brown_socks_is_3_dollars_l241_241712

def price_of_brown_sock (price_white_socks : ℚ) (price_white_more_than_brown : ℚ) : ℚ :=
  (price_white_socks - price_white_more_than_brown) / 2

def cost_of_15_brown_socks (price_brown_sock : ℚ) : ℚ :=
  15 * price_brown_sock

theorem cost_of_15_brown_socks_is_3_dollars
  (price_white_socks : ℚ) (price_white_more_than_brown : ℚ) 
  (h1 : price_white_socks = 0.45) (h2 : price_white_more_than_brown = 0.25) :
  cost_of_15_brown_socks (price_of_brown_sock price_white_socks price_white_more_than_brown) = 3 := 
by
  sorry

end cost_of_15_brown_socks_is_3_dollars_l241_241712


namespace largest_four_digit_divisible_by_8_l241_241999

/-- The largest four-digit number that is divisible by 8 is 9992. -/
theorem largest_four_digit_divisible_by_8 : ∃ x : ℕ, x = 9992 ∧ x < 10000 ∧ x % 8 = 0 ∧
  ∀ y : ℕ, y < 10000 ∧ y % 8 = 0 → y ≤ 9992 := 
by 
  sorry

end largest_four_digit_divisible_by_8_l241_241999


namespace total_time_for_seven_flights_l241_241159

theorem total_time_for_seven_flights :
  let a := 15
  let d := 8
  let n := 7
  let l := a + (n - 1) * d
  let S_n := n * (a + l) / 2
  S_n = 273 :=
by
  sorry

end total_time_for_seven_flights_l241_241159


namespace range_of_function_l241_241584

def range_exclusion (x : ℝ) : Prop :=
  x ≠ 1

theorem range_of_function :
  set.range (λ x : ℝ, if x = -2 then (0 : ℝ) else x + 3) = {y : ℝ | range_exclusion y} :=
by 
  sorry

end range_of_function_l241_241584


namespace min_value_my_function_l241_241404

noncomputable def my_function (x : ℝ) : ℝ :=
  abs (x - 1) + 2 * abs (x - 2) + 3 * abs (x - 3) + 4 * abs (x - 4)

theorem min_value_my_function :
  ∃ (x : ℝ), my_function x = 8 ∧ (∀ y : ℝ, my_function y ≥ 8) :=
sorry

end min_value_my_function_l241_241404


namespace equal_sets_d_l241_241724

theorem equal_sets_d : 
  (let M := {x | x^2 - 3*x + 2 = 0}
   let N := {1, 2}
   M = N) :=
by 
  sorry

end equal_sets_d_l241_241724


namespace remainder_of_1998_to_10_mod_10k_l241_241114

theorem remainder_of_1998_to_10_mod_10k : 
  let x := 1998
  let y := 10^4
  x^10 % y = 1024 := 
by
  let x := 1998
  let y := 10^4
  sorry

end remainder_of_1998_to_10_mod_10k_l241_241114


namespace curve_is_circle_l241_241484

theorem curve_is_circle (r θ : ℝ) (h : r = 1 / (Real.sin θ + Real.cos θ)) : 
  ∃ k : ℝ, ∀ x y : ℝ, (x^2 + y^2 = k^2) → 
    (r^2 = x^2 + y^2 ∧ ∃ (θ : ℝ), x/r = Real.cos θ ∧ y/r = Real.sin θ) :=
sorry

end curve_is_circle_l241_241484


namespace range_of_function_l241_241585

def range_exclusion (x : ℝ) : Prop :=
  x ≠ 1

theorem range_of_function :
  set.range (λ x : ℝ, if x = -2 then (0 : ℝ) else x + 3) = {y : ℝ | range_exclusion y} :=
by 
  sorry

end range_of_function_l241_241585


namespace problem_solution_l241_241473

noncomputable def g (x : ℝ) (P : ℝ) (Q : ℝ) (R : ℝ) : ℝ := x^2 / (P * x^2 + Q * x + R)

theorem problem_solution (P Q R : ℤ) 
  (h1 : ∀ x > 5, g x P Q R > 0.5)
  (h2 : P * (-3)^2 + Q * (-3) + R = 0)
  (h3 : P * 4^2 + Q * 4 + R = 0)
  (h4 : ∃ y : ℝ, y = 1 / P ∧ ∀ x : ℝ, abs (g x P Q R - y) < ε):
  P + Q + R = -24 :=
by
  sorry

end problem_solution_l241_241473


namespace pencils_count_l241_241838

theorem pencils_count (initial_pencils additional_pencils : ℕ) (h1 : initial_pencils = 115) (h2 : additional_pencils = 100) : initial_pencils + additional_pencils = 215 :=
by sorry

end pencils_count_l241_241838


namespace order_a_c_b_l241_241348

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 4 / Real.log 3
noncomputable def c : ℝ := Real.log 8 / Real.log 5

theorem order_a_c_b : a > c ∧ c > b := 
by {
  sorry
}

end order_a_c_b_l241_241348


namespace initial_students_l241_241062

def students_got_off : ℕ := 3
def students_left : ℕ := 7

theorem initial_students (h1 : students_got_off = 3) (h2 : students_left = 7) :
    students_got_off + students_left = 10 :=
by
  sorry

end initial_students_l241_241062


namespace determine_k_l241_241964

theorem determine_k (a b c k : ℤ) (h1 : c = -a - b) 
  (h2 : 60 < 6 * (8 * a + b) ∧ 6 * (8 * a + b) < 70)
  (h3 : 80 < 7 * (9 * a + b) ∧ 7 * (9 * a + b) < 90)
  (h4 : 2000 * k < (50^2 * a + 50 * b + c) ∧ (50^2 * a + 50 * b + c) < 2000 * (k + 1)) :
  k = 1 :=
  sorry

end determine_k_l241_241964


namespace right_triangle_side_length_l241_241237

theorem right_triangle_side_length (c a b : ℕ) (h1 : c = 5) (h2 : a = 3) (h3 : c^2 = a^2 + b^2) : b = 4 :=
  by
  sorry

end right_triangle_side_length_l241_241237


namespace speed_in_still_water_l241_241729

-- Defining the terms as given conditions in the problem
def speed_downstream (v_m v_s : ℝ) : ℝ := v_m + v_s
def speed_upstream (v_m v_s : ℝ) : ℝ := v_m - v_s

-- Given conditions translated into Lean definitions
def downstream_condition : Prop :=
  ∃ (v_m v_s : ℝ), speed_downstream v_m v_s = 7

def upstream_condition : Prop :=
  ∃ (v_m v_s : ℝ), speed_upstream v_m v_s = 4

-- The problem statement to prove
theorem speed_in_still_water : 
  downstream_condition ∧ upstream_condition → ∃ v_m : ℝ, v_m = 5.5 :=
by 
  intros
  sorry

end speed_in_still_water_l241_241729


namespace fraction_inhabitable_earth_surface_l241_241946

theorem fraction_inhabitable_earth_surface 
  (total_land_fraction: ℚ) 
  (inhabitable_land_fraction: ℚ) 
  (h1: total_land_fraction = 1/3) 
  (h2: inhabitable_land_fraction = 2/3) 
  : (total_land_fraction * inhabitable_land_fraction) = 2/9 :=
by
  sorry

end fraction_inhabitable_earth_surface_l241_241946


namespace probability_factor_of_36_l241_241266

theorem probability_factor_of_36 :
  let N := 36
  let S := {n : ℕ | n > 0 ∧ n ≤ N}
  let E := {n : ℕ | n ∈ S ∧ N % n = 0}
  (E.card : ℚ) / S.card = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l241_241266


namespace license_plate_palindrome_probability_l241_241191

theorem license_plate_palindrome_probability : 
  let p := 775 
  let q := 67600  
  p + q = 776 :=
by
  let p := 775
  let q := 67600
  show p + q = 776
  sorry

end license_plate_palindrome_probability_l241_241191


namespace probability_factor_of_36_l241_241247

theorem probability_factor_of_36 :
  (∃ n ∈ finset.range 37, (∃ k : ℕ, 36 = k * n)) → 9 / 36 = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l241_241247


namespace additional_people_needed_l241_241617

theorem additional_people_needed (k m : ℕ) (h1 : 8 * 3 = k) (h2 : m * 2 = k) : (m - 8) = 4 :=
by
  sorry

end additional_people_needed_l241_241617


namespace probability_factor_of_36_l241_241256

def is_factor_of (n d : ℕ) : Prop := d % n = 0

theorem probability_factor_of_36 : 
  let total := 36
  let factors_of_36 := { n | n ≤ 36 ∧ is_factor_of n 36 }
  (factors_of_36.to_finset.card : ℚ) / total = 1 / 4 := 
by
  sorry

end probability_factor_of_36_l241_241256


namespace fixed_monthly_fee_l241_241968

variable (x y : Real)

theorem fixed_monthly_fee :
  (x + y = 15.30) →
  (x + 1.5 * y = 20.55) →
  (x = 4.80) :=
by
  intros h1 h2
  sorry

end fixed_monthly_fee_l241_241968


namespace bridge_length_l241_241593

theorem bridge_length
  (train_length : ℝ) (train_speed : ℝ) (time_taken : ℝ)
  (h_train_length : train_length = 280)
  (h_train_speed : train_speed = 18)
  (h_time_taken : time_taken = 20) : ∃ L : ℝ, L = 80 :=
by
  let distance_covered := train_speed * time_taken
  have h_distance_covered : distance_covered = 360 := by sorry
  let bridge_length := distance_covered - train_length
  have h_bridge_length : bridge_length = 80 := by sorry
  existsi bridge_length
  exact h_bridge_length

end bridge_length_l241_241593


namespace cosine_square_plus_alpha_sine_l241_241198

variable (α : ℝ)

theorem cosine_square_plus_alpha_sine (h1 : 0 ≤ α) (h2 : α ≤ Real.pi / 2) : 
  Real.cos α * Real.cos α + α * Real.sin α ≥ 1 :=
sorry

end cosine_square_plus_alpha_sine_l241_241198


namespace simplify_expression_eval_l241_241690

noncomputable def simplify_expression (x : ℝ) : ℝ :=
  ((x / (x - 1) - 1) / ((x^2 - 1) / (x^2 - 2 * x + 1)))

theorem simplify_expression_eval : simplify_expression (Real.sqrt 5 - 1) = (Real.sqrt 5) / 5 :=
by
  sorry

end simplify_expression_eval_l241_241690


namespace ratio_of_ap_l241_241654

theorem ratio_of_ap (a d : ℕ) (h : 30 * a + 435 * d = 3 * (15 * a + 105 * d)) : a = 8 * d :=
by
  sorry

end ratio_of_ap_l241_241654


namespace probability_of_three_tails_one_head_in_four_tosses_l241_241160

noncomputable def probability_three_tails_one_head (n : ℕ) : ℚ :=
  if n = 4 then 1 / 4 else 0

theorem probability_of_three_tails_one_head_in_four_tosses :
  probability_three_tails_one_head 4 = 1 / 4 :=
by sorry

end probability_of_three_tails_one_head_in_four_tosses_l241_241160


namespace probability_from_first_to_last_l241_241879

noncomputable def probability_path_possible (n : ℕ) : ℝ :=
  (2 ^ (n - 1) : ℝ) / (Nat.choose (2 * (n - 1)) (n - 1) : ℝ)

theorem probability_from_first_to_last (n : ℕ) (h : n > 1) :
  probability_path_possible n = 2 ^ (n - 1) / (Nat.choose (2 * (n - 1)) (n - 1)) := sorry

end probability_from_first_to_last_l241_241879


namespace chelsea_sugar_problem_l241_241101

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

end chelsea_sugar_problem_l241_241101


namespace no_integer_solutions_l241_241538

theorem no_integer_solutions (x y z : ℤ) : x^3 + y^6 ≠ 7 * z + 3 :=
by sorry

end no_integer_solutions_l241_241538


namespace determinant_scaled_l241_241918

theorem determinant_scaled
  (x y z w : ℝ)
  (h : x * w - y * z = 10) :
  (3 * x) * (3 * w) - (3 * y) * (3 * z) = 90 :=
by sorry

end determinant_scaled_l241_241918


namespace num_people_at_gathering_l241_241895

noncomputable def total_people_at_gathering : ℕ :=
  let wine_soda := 12
  let wine_juice := 10
  let wine_coffee := 6
  let wine_tea := 4
  let soda_juice := 8
  let soda_coffee := 5
  let soda_tea := 3
  let juice_coffee := 7
  let juice_tea := 2
  let coffee_tea := 4
  let wine_soda_juice := 3
  let wine_soda_coffee := 1
  let wine_soda_tea := 2
  let wine_juice_coffee := 3
  let wine_juice_tea := 1
  let wine_coffee_tea := 2
  let soda_juice_coffee := 3
  let soda_juice_tea := 1
  let soda_coffee_tea := 2
  let juice_coffee_tea := 3
  let all_five := 1
  wine_soda + wine_juice + wine_coffee + wine_tea +
  soda_juice + soda_coffee + soda_tea + juice_coffee +
  juice_tea + coffee_tea + wine_soda_juice + wine_soda_coffee +
  wine_soda_tea + wine_juice_coffee + wine_juice_tea +
  wine_coffee_tea + soda_juice_coffee + soda_juice_tea +
  soda_coffee_tea + juice_coffee_tea + all_five

theorem num_people_at_gathering : total_people_at_gathering = 89 := by
  sorry

end num_people_at_gathering_l241_241895


namespace increasing_sequence_nec_but_not_suf_l241_241369

theorem increasing_sequence_nec_but_not_suf (a : ℕ → ℝ) :
  (∀ n, abs (a (n + 1)) > a n) → (∀ n, a (n + 1) > a n) ↔ 
  ∃ (n : ℕ), ¬ (abs (a (n + 1)) > a n) ∧ (a (n + 1) > a n) :=
sorry

end increasing_sequence_nec_but_not_suf_l241_241369


namespace solve_for_k_l241_241118

theorem solve_for_k (p q : ℝ) (k : ℝ) (hpq : 3 * p^2 + 6 * p + k = 0) (hq : 3 * q^2 + 6 * q + k = 0) 
    (h_diff : |p - q| = (1 / 2) * (p^2 + q^2)) : k = -16 + 12 * Real.sqrt 2 ∨ k = -16 - 12 * Real.sqrt 2 :=
by
  sorry

end solve_for_k_l241_241118


namespace simplified_expression_evaluation_l241_241687

-- Problem and conditions
def x := Real.sqrt 5 - 1

-- Statement of the proof problem
theorem simplified_expression_evaluation : 
  ( (x / (x - 1) - 1) / (x^2 - 1) / (x^2 - 2 * x + 1) ) = Real.sqrt 5 / 5 :=
sorry

end simplified_expression_evaluation_l241_241687


namespace two_person_subcommittees_l241_241510

theorem two_person_subcommittees (n : ℕ) (hn : n = 8) : (finset.univ : finset (fin 8)).powerset.card = 28 :=
by
  rw hn
  -- use the fact that the number of k-combinations (sub-committees) is given by binomial coefficient
  exact (nat.choose_symm 8 2).symm ▸ nat.choose 8 2

end two_person_subcommittees_l241_241510


namespace max_value_of_3_pow_x_minus_9_pow_x_l241_241124

open Real

theorem max_value_of_3_pow_x_minus_9_pow_x :
  ∃ (x : ℝ), ∀ y : ℝ, 3^x - 9^x ≤ 3^y - 9^y := 
sorry

end max_value_of_3_pow_x_minus_9_pow_x_l241_241124


namespace triangle_inequality_l241_241810

theorem triangle_inequality
  (a b c x y z : ℝ)
  (h_order : a < b ∧ b < c ∧ 0 < x)
  (h_area_eq : c * x = a * y + b * z) :
  x < y + z :=
by
  sorry

end triangle_inequality_l241_241810


namespace pencils_added_by_sara_l241_241571

-- Definitions based on given conditions
def original_pencils : ℕ := 115
def total_pencils : ℕ := 215

-- Statement to prove
theorem pencils_added_by_sara : total_pencils - original_pencils = 100 :=
by {
  -- Proof
  sorry
}

end pencils_added_by_sara_l241_241571


namespace zhou_yu_age_at_death_l241_241683

theorem zhou_yu_age_at_death (x : ℕ) (h₁ : 1 ≤ x ∧ x ≤ 9)
    (h₂ : ∃ age : ℕ, age = 10 * (x - 3) + x)
    (h₃ : x^2 = 10 * (x - 3) + x) :
    x^2 = 10 * (x - 3) + x :=
by
  sorry

end zhou_yu_age_at_death_l241_241683


namespace exists_positive_integer_n_with_N_distinct_prime_factors_l241_241807

open Nat

/-- Let \( N \) be a positive integer. Prove that there exists a positive integer \( n \) such that \( n^{2013} - n^{20} + n^{13} - 2013 \) has at least \( N \) distinct prime factors. -/
theorem exists_positive_integer_n_with_N_distinct_prime_factors (N : ℕ) (h : 0 < N) : 
  ∃ n : ℕ, 0 < n ∧ (n ^ 2013 - n ^ 20 + n ^ 13 - 2013).primeFactors.card ≥ N :=
sorry

end exists_positive_integer_n_with_N_distinct_prime_factors_l241_241807


namespace probability_factor_of_36_is_1_over_4_l241_241270

def totalPositiveIntegers := 36

def isDivisor (n: Nat) : Prop := ∃ (a b: Nat), (0 ≤ a ∧ a ≤ 2) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (n = 2^a * 3^b)

def numDivisors := 9
def probability := (numDivisors : ℚ) / (totalPositiveIntegers : ℚ)

theorem probability_factor_of_36_is_1_over_4 :
  probability = (1 / 4 : ℚ) :=
sorry

end probability_factor_of_36_is_1_over_4_l241_241270


namespace compute_expression_l241_241904

theorem compute_expression : 45 * (28 + 72) + 55 * 45 = 6975 := 
  by
  sorry

end compute_expression_l241_241904


namespace part_I_part_II_l241_241776

noncomputable def vector_a : ℝ × ℝ := (4, 3)
noncomputable def vector_b : ℝ × ℝ := (5, -12)
noncomputable def vector_sum := (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2)
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)
noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
noncomputable def vector_magnitude_sum := magnitude vector_sum
noncomputable def magnitude_a := magnitude vector_a
noncomputable def magnitude_b := magnitude vector_b
noncomputable def cos_theta := dot_product vector_a vector_b / (magnitude_a * magnitude_b)

-- Prove the magnitude of the sum of vectors is 9√2
theorem part_I : vector_magnitude_sum = 9 * Real.sqrt 2 :=
by
  sorry

-- Prove the cosine of the angle between the vectors is -16/65
theorem part_II : cos_theta = -16 / 65 :=
by
  sorry

end part_I_part_II_l241_241776


namespace measure_of_angle_Q_l241_241010

theorem measure_of_angle_Q (a b c d e Q : ℝ)
  (ha : a = 138) (hb : b = 85) (hc : c = 130) (hd : d = 120) (he : e = 95)
  (h_hex : a + b + c + d + e + Q = 720) : 
  Q = 152 :=
by
  rw [ha, hb, hc, hd, he] at h_hex
  linarith

end measure_of_angle_Q_l241_241010


namespace binom_12_10_eq_66_l241_241463

theorem binom_12_10_eq_66 : nat.choose 12 10 = 66 :=
by
  sorry

end binom_12_10_eq_66_l241_241463


namespace binom_subtract_l241_241755

theorem binom_subtract :
  (Nat.choose 7 4) - 5 = 30 :=
by
  -- proof goes here
  sorry

end binom_subtract_l241_241755


namespace find_X_l241_241078

theorem find_X : ∃ X : ℝ, 0.60 * X = 0.30 * 800 + 370 ∧ X = 1016.67 := by
  sorry

end find_X_l241_241078


namespace blu_ray_movies_returned_l241_241744

theorem blu_ray_movies_returned (D B x : ℕ)
  (h1 : D / B = 17 / 4)
  (h2 : D + B = 378)
  (h3 : D / (B - x) = 9 / 2) :
  x = 4 := by
  sorry

end blu_ray_movies_returned_l241_241744


namespace third_term_arithmetic_sequence_l241_241164

theorem third_term_arithmetic_sequence (a x : ℝ) 
  (h : 2 * a + 2 * x = 10) : a + x = 5 := 
by
  sorry

end third_term_arithmetic_sequence_l241_241164


namespace maria_earnings_correct_l241_241386

/-- Maria's earnings calculation over three days -/
def maria_total_earnings : ℕ :=
  let first_day := (30 * 2) + (20 * 3) in
  let second_day := ((30 * 2) * 2) + ((20 * 3) * 2) in
  let third_day := ((30 * 2) * 0.1 * 2) + (16 * 3) in
  first_day + second_day + third_day
  
theorem maria_earnings_correct : 
  maria_total_earnings = 420 :=
by
  -- Formal proof steps would go here
  sorry

end maria_earnings_correct_l241_241386


namespace john_needs_more_usd_l241_241025

noncomputable def additional_usd (needed_eur needed_sgd : ℝ) (has_usd has_jpy : ℝ) : ℝ :=
  let eur_to_usd := 1 / 0.84
  let sgd_to_usd := 1 / 1.34
  let jpy_to_usd := 1 / 110.35
  let total_needed_usd := needed_eur * eur_to_usd + needed_sgd * sgd_to_usd
  let total_has_usd := has_usd + has_jpy * jpy_to_usd
  total_needed_usd - total_has_usd

theorem john_needs_more_usd :
  ∀ (needed_eur needed_sgd : ℝ) (has_usd has_jpy : ℝ),
    needed_eur = 7.50 → needed_sgd = 5.00 → has_usd = 2.00 → has_jpy = 500 →
    additional_usd needed_eur needed_sgd has_usd has_jpy = 6.13 :=
by
  intros needed_eur needed_sgd has_usd has_jpy
  intros hneeded_eur hneeded_sgd hhas_usd hhas_jpy
  unfold additional_usd
  rw [hneeded_eur, hneeded_sgd, hhas_usd, hhas_jpy]
  sorry

end john_needs_more_usd_l241_241025


namespace Kaleb_total_games_l241_241961

-- Define the conditions as variables and parameters
variables (W L T : ℕ) -- the number of games won, lost, and tied
variable h_ratio : W : L : T = 7 : 4 : 5
variable h_won : W = 42

-- Define the theorem to prove the total number of games played
theorem Kaleb_total_games (W L T : ℕ) (h_ratio : W : L : T = 7 : 4 : 5) (h_won : W = 42) : W + L + T = 96 :=
by sorry

end Kaleb_total_games_l241_241961


namespace simplify_expression_eval_l241_241691

noncomputable def simplify_expression (x : ℝ) : ℝ :=
  ((x / (x - 1) - 1) / ((x^2 - 1) / (x^2 - 2 * x + 1)))

theorem simplify_expression_eval : simplify_expression (Real.sqrt 5 - 1) = (Real.sqrt 5) / 5 :=
by
  sorry

end simplify_expression_eval_l241_241691


namespace total_games_played_l241_241960

theorem total_games_played (won_games : ℕ) (won_ratio : ℕ) (lost_ratio : ℕ) (tied_ratio : ℕ) (total_games : ℕ) :
  won_games = 42 →
  won_ratio = 7 →
  lost_ratio = 4 →
  tied_ratio = 5 →
  total_games = won_games + lost_ratio * (won_games / won_ratio) + tied_ratio * (won_games / won_ratio) →
  total_games = 96 :=
by
  intros h_won h_won_ratio h_lost_ratio h_tied_ratio h_total
  sorry

end total_games_played_l241_241960


namespace binary_arithmetic_l241_241754

theorem binary_arithmetic 
  : (0b10110 + 0b1011 - 0b11100 + 0b11101 = 0b100010) :=
by
  sorry

end binary_arithmetic_l241_241754


namespace area_ratio_ADE_BCED_l241_241799

open EuclideanGeometry

variables {A B C D E F : Point}
variables {AB BC AC AD AE: ℝ}
variables {AB_pos : 0 < AB} {BC_pos : 0 < BC} {AC_pos : 0 < AC}
variables {AD_pos : 0 < AD} {AE_pos : 0 < AE}

-- Conditions specified
def triangle_ABC : Triangle := ⟨A, B, C⟩
def point_D_on_AB : Line := Line.mk A B
def point_E_on_AC : Line := Line.mk A C
def point_F_on_BC : Line := Line.mk B C
def parallel_DF_BE := parallel <| Segment.mk D F <| Segment.mk B E

-- Given numerical lengths
axiom AB_eq_24 : AB = 24
axiom BC_eq_45 : BC = 45
axiom AC_eq_48 : AC = 48
axiom AD_eq_18 : AD = 18
axiom AE_eq_16 : AE = 16

-- The main theorem
theorem area_ratio_ADE_BCED :
  ¬ collinear A B C →
  point_of_line D point_D_on_AB →
  point_of_line E point_E_on_AC →
  point_of_line F point_F_on_BC →
  parallel_DF_BE →
  (area_ratio (triangle a d e) (quadrilateral b c e d) = 1/15) :=
sorry

end area_ratio_ADE_BCED_l241_241799


namespace two_person_subcommittees_l241_241504

def committee_size := 8
def sub_committee_size := 2
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def combination (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem two_person_subcommittees : combination committee_size sub_committee_size = 28 := by
  sorry

end two_person_subcommittees_l241_241504


namespace equivalent_expression_l241_241577

def evaluate_expression : ℚ :=
  let part1 := (2/3) * ((35/100) * 250)
  let part2 := ((75/100) * 150) / 16
  let part3 := (1/2) * ((40/100) * 500)
  part1 - part2 + part3

theorem equivalent_expression :
  evaluate_expression = 151.3020833333 :=  
by 
  sorry

end equivalent_expression_l241_241577


namespace union_complement_l241_241498

def universalSet : Set ℤ := { x | x^2 < 9 }

def A : Set ℤ := {1, 2}

def B : Set ℤ := {-2, -1, 2}

def complement_I_B : Set ℤ := universalSet \ B

theorem union_complement :
  A ∪ complement_I_B = {0, 1, 2} :=
by
  sorry

end union_complement_l241_241498


namespace reciprocal_of_neg3_l241_241214

theorem reciprocal_of_neg3 : 1 / (-3 : ℝ) = - (1 / 3) :=
by
  sorry

end reciprocal_of_neg3_l241_241214


namespace triangle_area_example_l241_241778

-- Define the right triangle DEF with angle at D being 45 degrees and DE = 8 units
noncomputable def area_of_45_45_90_triangle (DE : ℝ) (angle_d : ℝ) (h_angle : angle_d = 45) (h_DE : DE = 8) : ℝ :=
  1 / 2 * DE * DE

-- State the theorem to prove the area
theorem triangle_area_example {DE : ℝ} {angle_d : ℝ} (h_angle : angle_d = 45) (h_DE : DE = 8) :
  area_of_45_45_90_triangle DE angle_d h_angle h_DE = 32 := 
sorry

end triangle_area_example_l241_241778


namespace minimize_q_neg_1_l241_241602

noncomputable def respectful_polynomial (a b : ℝ) : (ℝ → ℝ) := λ x : ℝ, x^2 + a * x + b

theorem minimize_q_neg_1 (a b : ℝ) (h1 : b = -a - 1) (h2 : ∀ x, ((respectful_polynomial a b) ((respectful_polynomial a b) x) = 0) → (x^2 + (2 * a) * x + (a^2 + 2 * b + 1) * x + (2 * a * b + a) + (b^2 + a * b + b) = 0)) :
  (respectful_polynomial a b) (-1) = 0 :=
by {
  sorry
}

end minimize_q_neg_1_l241_241602


namespace solve_equation_l241_241039

theorem solve_equation :
  ∃ x : ℝ, (x - 2)^3 + (x - 6)^3 = 54 ∧ x = 7 := by
sorry

end solve_equation_l241_241039


namespace symmetric_origin_coordinates_l241_241716

def symmetric_coordinates (x y : ℚ) (x_line y_line : ℚ) : Prop :=
  x_line - 2 * y_line + 2 = 0 ∧ y_line = -2 * x_line ∧ x = -4/5 ∧ y = 8/5

theorem symmetric_origin_coordinates :
  ∃ (x_0 y_0 : ℚ), symmetric_coordinates x_0 y_0 (-4/5) (8/5) :=
by
  use -4/5, 8/5
  sorry

end symmetric_origin_coordinates_l241_241716


namespace nth_equation_l241_241551

theorem nth_equation (n : ℕ) (h : 0 < n) : 9 * (n - 1) + n = 10 * n - 9 := 
  sorry

end nth_equation_l241_241551


namespace area_of_ABCD_proof_l241_241822

noncomputable def point := ℝ × ℝ

structure Rectangle :=
  (A B C D : point)
  (angle_C_trisected_by_CE_CF : Prop)
  (E_on_AB : Prop)
  (F_on_AD : Prop)
  (AF : ℝ)
  (BE : ℝ)

def area_of_rectangle (rect : Rectangle) : ℝ :=
  let (x1, y1) := rect.A
  let (x2, y2) := rect.C
  (x2 - x1) * (y2 - y1)

theorem area_of_ABCD_proof :
  ∀ (ABCD : Rectangle),
    ABCD.angle_C_trisected_by_CE_CF →
    ABCD.E_on_AB →
    ABCD.F_on_AD →
    ABCD.AF = 2 →
    ABCD.BE = 6 →
    abs (area_of_rectangle ABCD - 150) < 1 :=
by
  sorry

end area_of_ABCD_proof_l241_241822


namespace final_number_is_odd_l241_241035

theorem final_number_is_odd : 
  ∃ (n : ℤ), n % 2 = 1 ∧ n ≥ 1 ∧ n < 1024 := sorry

end final_number_is_odd_l241_241035


namespace total_population_of_cities_l241_241225

theorem total_population_of_cities (n : ℕ) (avg_pop : ℕ) (pn : (n = 20)) (avg_factor: (avg_pop = (4500 + 5000) / 2)) : 
  n * avg_pop = 95000 := 
by 
  sorry

end total_population_of_cities_l241_241225


namespace price_increase_needed_l241_241981

theorem price_increase_needed (P : ℝ) (hP : P > 0) : (100 * ((P / (0.85 * P)) - 1)) = 17.65 :=
by
  sorry

end price_increase_needed_l241_241981


namespace last_digit_1993_2002_plus_1995_2002_l241_241564

theorem last_digit_1993_2002_plus_1995_2002 :
  (1993 ^ 2002 + 1995 ^ 2002) % 10 = 4 :=
by sorry

end last_digit_1993_2002_plus_1995_2002_l241_241564


namespace jina_teddies_l241_241957

variable (T : ℕ)

def initial_teddies (bunnies : ℕ) (koala : ℕ) (add_teddies : ℕ) (total : ℕ) :=
  T + bunnies + add_teddies + koala

theorem jina_teddies (bunnies : ℕ) (koala : ℕ) (add_teddies : ℕ) (total : ℕ) :
  bunnies = 3 * T ∧ koala = 1 ∧ add_teddies = 2 * bunnies ∧ total = 51 → T = 5 :=
by
  sorry

end jina_teddies_l241_241957


namespace determinant_scaled_l241_241919

theorem determinant_scaled
  (x y z w : ℝ)
  (h : x * w - y * z = 10) :
  (3 * x) * (3 * w) - (3 * y) * (3 * z) = 90 :=
by sorry

end determinant_scaled_l241_241919


namespace perpendicular_lines_slope_product_l241_241635

theorem perpendicular_lines_slope_product (a : ℝ) (x y : ℝ) :
  let l1 := ax + y + 2 = 0
  let l2 := x + y = 0
  ( -a * -1 = -1 ) -> a = -1 :=
sorry

end perpendicular_lines_slope_product_l241_241635


namespace find_b_value_l241_241624

theorem find_b_value
  (b : ℝ) :
  (∃ x y : ℝ, x = 3 ∧ y = -5 ∧ b * x + (b + 2) * y = b - 1) → b = -3 :=
by
  sorry

end find_b_value_l241_241624


namespace hyperbola_eccentricity_l241_241812

theorem hyperbola_eccentricity (a b m n e : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_mn : m * n = 2 / 9)
  (h_hyperbola : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) : e = 3 * Real.sqrt 2 / 4 :=
sorry

end hyperbola_eccentricity_l241_241812


namespace g_at_2_l241_241359

def g (x : ℝ) : ℝ := x^2 - 4

theorem g_at_2 : g 2 = 0 := by
  sorry

end g_at_2_l241_241359


namespace equivalent_multipliers_l241_241649

variable (a b c : ℝ)

theorem equivalent_multipliers :
  (a - 0.07 * a + 0.05 * b) / c = (0.93 * a + 0.05 * b) / c :=
sorry

end equivalent_multipliers_l241_241649


namespace parabola_directrix_l241_241642

theorem parabola_directrix (vertex_origin : ∀ (x y : ℝ), x = 0 ∧ y = 0)
    (directrix : ∀ (y : ℝ), y = 4) : ∃ p, x^2 = -2 * p * y ∧ p = 8 ∧ x^2 = -16 * y := 
sorry

end parabola_directrix_l241_241642


namespace sum_ad_eq_two_l241_241861

theorem sum_ad_eq_two (a b c d : ℝ) 
  (h1 : a + b = 4) 
  (h2 : b + c = 7) 
  (h3 : c + d = 5) : 
  a + d = 2 :=
by
  sorry

end sum_ad_eq_two_l241_241861


namespace problem1_part1_problem1_part2_l241_241448

theorem problem1_part1 : (3 - Real.pi)^0 - 2 * Real.cos (Real.pi / 6) + abs (1 - Real.sqrt 3) + (1 / 2)⁻¹ = 2 := by
  sorry

theorem problem1_part2 {x : ℝ} : x^2 - 2 * x - 9 = 0 -> (x = 1 + Real.sqrt 10 ∨ x = 1 - Real.sqrt 10) := by
  sorry

end problem1_part1_problem1_part2_l241_241448


namespace arrange_numbers_l241_241495

namespace MathProofs

theorem arrange_numbers (a b : ℚ) (h1 : a > 0) (h2 : b < 0) (h3 : a + b < 0) :
  b < -a ∧ -a < a ∧ a < -b :=
by
  -- Proof to be completed
  sorry

end MathProofs

end arrange_numbers_l241_241495


namespace isosceles_triangle_leg_length_l241_241097

-- Define the necessary condition for the isosceles triangle
def isosceles_triangle (a b c : ℕ) : Prop :=
  b = c ∧ a + b + c = 16 ∧ a = 4

-- State the theorem we want to prove
theorem isosceles_triangle_leg_length :
  ∃ (b c : ℕ), isosceles_triangle 4 b c ∧ b = 6 :=
by
  -- Formal proof will be provided here
  sorry

end isosceles_triangle_leg_length_l241_241097


namespace part1_real_values_part2_imaginary_values_l241_241343

namespace ComplexNumberProblem

-- Definitions of conditions for part 1
def imaginaryZero (x : ℝ) : Prop :=
  x^2 + 3*x + 2 = 0

def realPositive (x : ℝ) : Prop :=
  x^2 - 2*x - 2 > 0

-- Definition of question for part 1
def realValues (x : ℝ) : Prop :=
  x = -1 ∨ x = -2

-- Proof problem for part 1
theorem part1_real_values (x : ℝ) (h1 : imaginaryZero x) (h2 : realPositive x) : realValues x :=
by
  have h : realValues x := sorry
  exact h

-- Definitions of conditions for part 2
def realPartOne (x : ℝ) : Prop :=
  x^2 - 2*x - 2 = 1

def imaginaryNonZero (x : ℝ) : Prop :=
  x^2 + 3*x + 2 ≠ 0

-- Definition of question for part 2
def imaginaryValues (x : ℝ) : Prop :=
  x = 3

-- Proof problem for part 2
theorem part2_imaginary_values (x : ℝ) (h1 : realPartOne x) (h2 : imaginaryNonZero x) : imaginaryValues x :=
by
  have h : imaginaryValues x := sorry
  exact h

end ComplexNumberProblem

end part1_real_values_part2_imaginary_values_l241_241343


namespace pairs_solution_l241_241760

theorem pairs_solution (x y : ℝ) :
  (4 * x^2 - y^2)^2 + (7 * x + 3 * y - 39)^2 = 0 ↔ (x = 3 ∧ y = 6) ∨ (x = 39 ∧ y = -78) := 
by
  sorry

end pairs_solution_l241_241760


namespace problem_l241_241133

variable (m : ℝ)

def p (m : ℝ) : Prop := m ≤ 2
def q (m : ℝ) : Prop := 0 < m ∧ m < 1

theorem problem (hpq : ¬ (p m ∧ q m)) (hlpq : p m ∨ q m) : m ≤ 0 ∨ (1 ≤ m ∧ m ≤ 2) := 
sorry

end problem_l241_241133


namespace value_of_livestock_l241_241435

variable (x y : ℝ)

theorem value_of_livestock :
  (5 * x + 2 * y = 10) ∧ (2 * x + 5 * y = 8) :=
sorry

end value_of_livestock_l241_241435


namespace linear_regression_change_l241_241777

theorem linear_regression_change (x : ℝ) :
  let y1 := 2 - 1.5 * x
  let y2 := 2 - 1.5 * (x + 1)
  y2 - y1 = -1.5 := by
  -- y1 = 2 - 1.5 * x
  -- y2 = 2 - 1.5 * x - 1.5
  -- Δ y = y2 - y1
  sorry

end linear_regression_change_l241_241777


namespace ratio_of_small_rectangle_length_to_width_l241_241973

-- Define the problem conditions
variables (s : ℝ)

-- Define the length and width of the small rectangle
def length_of_small_rectangle := 3 * s
def width_of_small_rectangle := s

-- Prove that the ratio of the length to the width of the small rectangle is 3
theorem ratio_of_small_rectangle_length_to_width : 
  length_of_small_rectangle s / width_of_small_rectangle s = 3 :=
by
  sorry

end ratio_of_small_rectangle_length_to_width_l241_241973


namespace intersection_A_B_complement_l241_241813

def universal_set : Set ℝ := {x : ℝ | True}
def A : Set ℝ := {x : ℝ | x^2 - 2 * x < 0}
def B : Set ℝ := {x : ℝ | x > 1}
def B_complement : Set ℝ := {x : ℝ | x ≤ 1}

theorem intersection_A_B_complement :
  (A ∩ B_complement) = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_A_B_complement_l241_241813


namespace slope_perpendicular_l241_241852

theorem slope_perpendicular (x1 y1 x2 y2 m : ℚ) 
  (hx1 : x1 = 3) (hy1 : y1 = -4) (hx2 : x2 = -6) (hy2 : y2 = 2) 
  (hm : m = (y2 - y1) / (x2 - x1)) :
  ∀ m_perpendicular: ℚ, m_perpendicular = (-1 / m) → m_perpendicular = 3/2 := 
sorry

end slope_perpendicular_l241_241852


namespace students_per_group_l241_241226

theorem students_per_group (n m : ℕ) (h_n : n = 36) (h_m : m = 9) : 
  (n - m) / 3 = 9 := 
by
  sorry

end students_per_group_l241_241226


namespace cylinder_original_radius_l241_241481

theorem cylinder_original_radius
    (r h: ℝ)
    (h₀: h = 4)
    (h₁: π * (r + 8)^2 * 4 = π * r^2 * 12) :
    r = 12 :=
by
  -- Insert your proof here
  sorry

end cylinder_original_radius_l241_241481


namespace min_p_q_sum_l241_241043

theorem min_p_q_sum (p q : ℕ) (hp : 0 < p) (hq : 0 < q) (h : 162 * p = q^3) : p + q = 54 :=
sorry

end min_p_q_sum_l241_241043


namespace sin_675_eq_neg_sqrt2_over_2_l241_241758

theorem sin_675_eq_neg_sqrt2_over_2 :
  sin (675 * Real.pi / 180) = - (Real.sqrt 2 / 2) := 
by
  -- problem states that 675° reduces to 315°
  have h₁ : (675 : ℝ) ≡ 315 [MOD 360], by norm_num,
  
  -- recognize 315° as 360° - 45°
  have h₂ : (315 : ℝ) = 360 - 45, by norm_num,

  -- in the fourth quadrant, sin(315°) = -sin(45°)
  have h₃ : sin (315 * Real.pi / 180) = - (sin (45 * Real.pi / 180)), by
    rw [Real.sin_angle_sub_eq_sin_add, Real.sin_angle_eq_sin_add],
    
  -- sin(45°) = sqrt(2)/2
  have h₄ : sin (45 * Real.pi / 180) = Real.sqrt 2 / 2, by
    -- As an assumed known truth for this problem
    exact Real.sin_pos_of_angle,

  -- combine above facts
  rw [h₃, h₄],
  norm_num
  -- sorry is needed if proof steps aren't complete
  sorry

end sin_675_eq_neg_sqrt2_over_2_l241_241758


namespace jessica_purchase_cost_l241_241531

noncomputable def c_toy : Real := 10.22
noncomputable def c_cage : Real := 11.73
noncomputable def c_total : Real := c_toy + c_cage

theorem jessica_purchase_cost : c_total = 21.95 :=
by
  sorry

end jessica_purchase_cost_l241_241531


namespace calculateSurfaceArea_l241_241208

noncomputable def totalSurfaceArea (r : ℝ) : ℝ :=
  let hemisphereCurvedArea := 2 * Real.pi * r^2
  let cylinderLateralArea := 2 * Real.pi * r * r
  hemisphereCurvedArea + cylinderLateralArea

theorem calculateSurfaceArea :
  ∃ r : ℝ, (Real.pi * r^2 = 144 * Real.pi) ∧ totalSurfaceArea r = 576 * Real.pi :=
by
  exists 12
  constructor
  . sorry -- Proof that 144π = π*12^2 can be shown
  . sorry -- Proof that 576π = 288π + 288π can be shown

end calculateSurfaceArea_l241_241208


namespace mike_profit_l241_241545

-- Define the conditions
def total_acres : ℕ := 200
def cost_per_acre : ℕ := 70
def sold_acres := total_acres / 2
def selling_price_per_acre : ℕ := 200

-- Statement to prove the profit Mike made is $6,000
theorem mike_profit :
  let total_cost := total_acres * cost_per_acre
  let total_revenue := sold_acres * selling_price_per_acre
  total_revenue - total_cost = 6000 := 
by
  sorry

end mike_profit_l241_241545


namespace mike_profit_l241_241543

-- Definition of initial conditions
def acres_bought := 200
def cost_per_acre := 70
def fraction_sold := 1 / 2
def selling_price_per_acre := 200

-- Definitions derived from conditions
def total_cost := acres_bought * cost_per_acre
def acres_sold := acres_bought * fraction_sold
def total_revenue := acres_sold * selling_price_per_acre
def profit := total_revenue - total_cost

-- Theorem stating the question and answer tuple
theorem mike_profit : profit = 6000 := by
  -- Proof omitted
  sorry

end mike_profit_l241_241543


namespace books_about_fish_l241_241858

theorem books_about_fish (F : ℕ) (spent : ℕ) (cost_whale_books : ℕ) (cost_magazines : ℕ) (cost_fish_books_per_unit : ℕ) (whale_books : ℕ) (magazines : ℕ) :
  whale_books = 9 →
  magazines = 3 →
  cost_whale_books = 11 →
  cost_magazines = 1 →
  spent = 179 →
  99 + 11 * F + 3 = spent → F = 7 :=
by
  sorry

end books_about_fish_l241_241858


namespace binomial_12_10_eq_66_l241_241460

theorem binomial_12_10_eq_66 : (Nat.choose 12 10) = 66 :=
by
  sorry

end binomial_12_10_eq_66_l241_241460


namespace binomial_12_10_eq_66_l241_241453

theorem binomial_12_10_eq_66 : binomial 12 10 = 66 :=
by
  sorry

end binomial_12_10_eq_66_l241_241453


namespace p_iff_q_l241_241148

def f (x a : ℝ) := x * (x - a) * (x - 2)

def p (a : ℝ) := 0 < a ∧ a < 2

def q (a : ℝ) := 
  let f' x := 3 * x^2 - 2 * (a + 2) * x + 2 * a
  f' a < 0

theorem p_iff_q (a : ℝ) : (p a) ↔ (q a) := by
  sorry

end p_iff_q_l241_241148


namespace count_symmetric_numbers_count_symmetric_divisible_by_4_sum_symmetric_numbers_l241_241441

-- Definitions based on conditions
def is_symmetric_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 1 ∨ d = 8 ∨ d = 6 ∨ d = 9

def symmetric_pair (a b : ℕ) : Prop :=
  (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 1) ∨ (a = 8 ∧ b = 8) ∨ (a = 6 ∧ b = 9) ∨ (a = 9 ∧ b = 6)

-- 1. Prove the total number of 7-digit symmetric numbers
theorem count_symmetric_numbers : ∃ n, n = 300 := by
  sorry

-- 2. Prove the number of symmetric numbers divisible by 4
theorem count_symmetric_divisible_by_4 : ∃ n, n = 75 := by
  sorry

-- 3. Prove the total sum of these 7-digit symmetric numbers
theorem sum_symmetric_numbers : ∃ s, s = 1959460200 := by
  sorry

end count_symmetric_numbers_count_symmetric_divisible_by_4_sum_symmetric_numbers_l241_241441


namespace find_phi_l241_241403

theorem find_phi (ϕ : ℝ) (h1 : |ϕ| < π / 2)
  (h2 : ∃ k : ℤ, 3 * (π / 12) + ϕ = k * π + π / 2) :
  ϕ = π / 4 :=
by sorry

end find_phi_l241_241403


namespace abc_unique_l241_241675

theorem abc_unique (n : ℕ) (hn : 0 < n) (p : ℕ) (hp : Nat.Prime p) 
                   (a b c : ℤ) 
                   (h : a^n + p * b = b^n + p * c ∧ b^n + p * c = c^n + p * a) 
                   : a = b ∧ b = c :=
by
  sorry

end abc_unique_l241_241675


namespace sum_of_areas_of_tangent_circles_l241_241989

theorem sum_of_areas_of_tangent_circles
  (r s t : ℝ)
  (h1 : r + s = 6)
  (h2 : s + t = 8)
  (h3 : r + t = 10) :
  π * (r^2 + s^2 + t^2) = 56 * π :=
by
  sorry

end sum_of_areas_of_tangent_circles_l241_241989


namespace area_of_figure_enclosed_by_curve_l241_241021

theorem area_of_figure_enclosed_by_curve (θ : ℝ) : 
  ∃ (A : ℝ), A = 4 * Real.pi ∧ (∀ θ, (4 * Real.cos θ)^2 = (4 * Real.cos θ) * 4 * Real.cos θ) :=
sorry

end area_of_figure_enclosed_by_curve_l241_241021


namespace longest_side_of_triangle_l241_241141

theorem longest_side_of_triangle (a d : ℕ) (h1 : d = 2) (h2 : a - d > 0) (h3 : a + d > 0)
    (h_angle : ∃ C : ℝ, C = 120) 
    (h_arith_seq : ∃ (b c : ℕ), b = a - d ∧ c = a ∧ b + 2 * d = c + d) : 
    a + d = 7 :=
by
  -- The proof will be provided here
  sorry

end longest_side_of_triangle_l241_241141


namespace probability_factor_of_36_l241_241275

theorem probability_factor_of_36 : 
  let pos_ints := { n : ℕ | 1 ≤ n ∧ n ≤ 36 }
  let num_divisors := { d : ℕ | d ∣ 36 }
  (num_divisors.to_finset.card : ℚ) / (pos_ints.to_finset.card : ℚ) = 1 / 4 :=
by {
  sorry
}

end probability_factor_of_36_l241_241275


namespace two_person_subcommittees_l241_241506

def committee_size := 8
def sub_committee_size := 2
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def combination (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem two_person_subcommittees : combination committee_size sub_committee_size = 28 := by
  sorry

end two_person_subcommittees_l241_241506


namespace find_b_l241_241565

variable (b : ℝ)

theorem find_b 
    (h₁ : 0 < b)
    (h₂ : b < 4)
    (area_ratio : ∃ k : ℝ, k = 4/16 ∧ (4 + b) / -b = 2 * k) :
  b = -4/3 :=
by
  sorry

end find_b_l241_241565


namespace smallest_x_for_div_by9_l241_241791

-- Define the digit sum of the number 761*829 with a placeholder * for x
def digit_sum_with_x (x : Nat) : Nat :=
  7 + 6 + 1 + x + 8 + 2 + 9

-- State the theorem to prove the smallest value of x makes the sum divisible by 9
theorem smallest_x_for_div_by9 : ∃ x : Nat, digit_sum_with_x x % 9 = 0 ∧ (∀ y : Nat, y < x → digit_sum_with_x y % 9 ≠ 0) :=
sorry

end smallest_x_for_div_by9_l241_241791


namespace inverse_of_2_is_46_l241_241938

-- Given the function f(x) = 5x^3 + 6
def f (x : ℝ) : ℝ := 5 * x^3 + 6

-- Prove the statement
theorem inverse_of_2_is_46 : (∃ y, f y = x) ∧ f (2 : ℝ) = 46 → x = 46 :=
by
  sorry

end inverse_of_2_is_46_l241_241938


namespace least_subtracted_number_l241_241722

theorem least_subtracted_number (a b c d e : ℕ) 
  (h₁ : a = 2590) 
  (h₂ : b = 9) 
  (h₃ : c = 11) 
  (h₄ : d = 13) 
  (h₅ : e = 6) 
  : ∃ (x : ℕ), a - x % b = e ∧ a - x % c = e ∧ a - x % d = e := by
  sorry

end least_subtracted_number_l241_241722


namespace first_offset_length_l241_241119

theorem first_offset_length (diagonal : ℝ) (offset2 : ℝ) (area : ℝ) (h_diagonal : diagonal = 50) (h_offset2 : offset2 = 8) (h_area : area = 450) :
  ∃ offset1 : ℝ, offset1 = 10 :=
by
  sorry

end first_offset_length_l241_241119


namespace solve_arithmetic_series_l241_241322

theorem solve_arithmetic_series : 
  25^2 - 23^2 + 21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 338 :=
by sorry

end solve_arithmetic_series_l241_241322


namespace evaluate_g_at_6_l241_241379

def g (x : ℝ) := 3 * x^4 - 19 * x^3 + 31 * x^2 - 27 * x - 72

theorem evaluate_g_at_6 : g 6 = 666 := by
  sorry

end evaluate_g_at_6_l241_241379


namespace average_weight_l241_241168

theorem average_weight (w : ℕ) : 
  (64 < w ∧ w ≤ 67) → w = 66 :=
by sorry

end average_weight_l241_241168


namespace quadratic_roots_new_equation_l241_241136

theorem quadratic_roots_new_equation (a b c x1 x2 : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : x1 + x2 = -b / a) 
  (h3 : x1 * x2 = c / a) : 
  ∃ (a' b' c' : ℝ), a' * x^2 + b' * x + c' = 0 ∧ a' = a^2 ∧ b' = 3 * a * b ∧ c' = 2 * b^2 + a * c :=
sorry

end quadratic_roots_new_equation_l241_241136


namespace erased_number_l241_241834

theorem erased_number (n i : ℕ) (h : (n * (n + 1) / 2 - i) / (n - 1) = 602 / 17) : i = 7 :=
sorry

end erased_number_l241_241834


namespace total_situps_performed_l241_241897

theorem total_situps_performed :
  let Barney_situps_per_min := 45 in
  let Carrie_situps_per_min := 2 * Barney_situps_per_min in
  let Jerrie_situps_per_min := Carrie_situps_per_min + 5 in
  let total_situps :=
    (Barney_situps_per_min * 1) +
    (Carrie_situps_per_min * 2) +
    (Jerrie_situps_per_min * 3) in
  total_situps = 510 :=
by
  sorry

end total_situps_performed_l241_241897


namespace paul_and_lisa_total_dollars_l241_241970

def total_dollars_of_paul_and_lisa (paul_dol : ℚ) (lisa_dol : ℚ) : ℚ :=
  paul_dol + lisa_dol

theorem paul_and_lisa_total_dollars (paul_dol := (5 / 6 : ℚ)) (lisa_dol := (2 / 5 : ℚ)) :
  total_dollars_of_paul_and_lisa paul_dol lisa_dol = (123 / 100 : ℚ) :=
by
  sorry

end paul_and_lisa_total_dollars_l241_241970


namespace sheela_monthly_income_l241_241734

variable (deposits : ℝ) (percentage : ℝ) (monthly_income : ℝ)

-- Conditions
axiom deposit_condition : deposits = 3400
axiom percentage_condition : percentage = 0.15
axiom income_condition : deposits = percentage * monthly_income

-- Proof goal
theorem sheela_monthly_income :
  monthly_income = 3400 / 0.15 :=
sorry

end sheela_monthly_income_l241_241734


namespace series_converges_to_limit_l241_241608

theorem series_converges_to_limit :
  let a := 3
  let r := (1 / 3 : ℝ)
  has_sum (λ n : ℕ, a * r^n) (9 / 2) :=
begin
  sorry
end

end series_converges_to_limit_l241_241608


namespace prove_total_rent_of_field_l241_241864

def totalRentField (A_cows A_months B_cows B_months C_cows C_months 
                    D_cows D_months E_cows E_months F_cows F_months 
                    G_cows G_months A_rent : ℕ) : ℕ := 
  let A_cow_months := A_cows * A_months
  let B_cow_months := B_cows * B_months
  let C_cow_months := C_cows * C_months
  let D_cow_months := D_cows * D_months
  let E_cow_months := E_cows * E_months
  let F_cow_months := F_cows * F_months
  let G_cow_months := G_cows * G_months
  let total_cow_months := A_cow_months + B_cow_months + C_cow_months + 
                          D_cow_months + E_cow_months + F_cow_months + G_cow_months
  let rent_per_cow_month := A_rent / A_cow_months
  total_cow_months * rent_per_cow_month

theorem prove_total_rent_of_field : totalRentField 24 3 10 5 35 4 21 3 15 6 40 2 28 (7/2) 720 = 5930 :=
  by
  sorry

end prove_total_rent_of_field_l241_241864


namespace solve_for_x_l241_241518

theorem solve_for_x (x : ℝ) (h : x^4 = (-3)^4) : x = 3 ∨ x = -3 :=
sorry

end solve_for_x_l241_241518


namespace cuboid_volume_l241_241135

theorem cuboid_volume (x y z : ℝ)
  (h1 : 2 * (x + y) = 20)
  (h2 : 2 * (y + z) = 32)
  (h3 : 2 * (x + z) = 28) : x * y * z = 240 := 
by
  sorry

end cuboid_volume_l241_241135


namespace average_score_l241_241173

theorem average_score (m n : ℝ) (hm : m ≥ 0) (hn : n ≥ 0) :
  (20 * m + 23 * n) / (20 + 23) = 20 / 43 * m + 23 / 43 * n := sorry

end average_score_l241_241173


namespace geometric_sum_S12_l241_241221

theorem geometric_sum_S12 (a r : ℝ) (h₁ : r ≠ 1) (S4_eq : a * (1 - r^4) / (1 - r) = 24) (S8_eq : a * (1 - r^8) / (1 - r) = 36) : a * (1 - r^12) / (1 - r) = 42 := 
sorry

end geometric_sum_S12_l241_241221


namespace find_ab_cd_l241_241189

variables (a b c d : ℝ)

def special_eq (x : ℝ) := 
  (min (20 * x + 19) (19 * x + 20) = (a * x + b) - abs (c * x + d))

theorem find_ab_cd (h : ∀ x : ℝ, special_eq a b c d x) :
  a * b + c * d = 380 := 
sorry

end find_ab_cd_l241_241189


namespace necessary_but_not_sufficient_l241_241151

open Set

namespace Mathlib

noncomputable def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
noncomputable def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem necessary_but_not_sufficient (a : ℝ) : 
  (a ∈ M → a ∈ N) ∧ ¬(a ∈ N → a ∈ M) :=
by
  sorry

end Mathlib

end necessary_but_not_sufficient_l241_241151


namespace labor_cost_calculation_l241_241018

def num_men : Nat := 5
def num_women : Nat := 8
def num_boys : Nat := 10

def base_wage_man : Nat := 100
def base_wage_woman : Nat := 80
def base_wage_boy : Nat := 50

def efficiency_man_woman_ratio : Nat := 2
def efficiency_man_boy_ratio : Nat := 3

def overtime_rate_multiplier : Nat := 3 / 2 -- 1.5 as a ratio
def holiday_rate_multiplier : Nat := 2

def num_men_working_overtime : Nat := 3
def hours_worked_overtime : Nat := 10
def regular_workday_hours : Nat := 8

def is_holiday : Bool := true

theorem labor_cost_calculation : 
  (num_men * base_wage_man * holiday_rate_multiplier
    + num_women * base_wage_woman * holiday_rate_multiplier
    + num_boys * base_wage_boy * holiday_rate_multiplier
    + num_men_working_overtime * (hours_worked_overtime - regular_workday_hours) * (base_wage_man * overtime_rate_multiplier)) 
  = 4180 :=
by
  sorry

end labor_cost_calculation_l241_241018


namespace no_two_positive_roots_l241_241367

theorem no_two_positive_roots (a : Fin 2022 → ℕ)
  (h : ∀ i : Fin 2022, 2 + i < 2024)
  (distinct : Function.Injective a)
  (coeffs : ∀ i : Fin 2022, (a i) ∈ {n | 2 ≤ n ∧ n ≤ 2023}) :
  ¬ ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧ (x₁ ^ 2022 - ∑ i in Finset.range 2022, (a ⟨i, Fin.is_lt (i, 2022)⟩) * x₁ ^ (2021 - i)) = 2023 ∧ (x₂ ^ 2022 - ∑ i in Finset.range 2022, (a ⟨i, Fin.is_lt (i, 2022)⟩) * x₂ ^ (2021 - i)) = 2023 := sorry

end no_two_positive_roots_l241_241367


namespace sum_remainder_l241_241360

theorem sum_remainder (n : ℤ) : ((9 - n) + (n + 4)) % 9 = 4 := 
by 
  sorry

end sum_remainder_l241_241360


namespace sequence_sum_l241_241299

theorem sequence_sum :
  1 - 4 + 7 - 10 + 13 - 16 + 19 - 22 + 25 - 28 + 31 - 34 + 37 - 40 + 43 - 46 + 49 - 52 + 55 = 28 :=
by
  sorry

end sequence_sum_l241_241299


namespace cricket_initial_overs_l241_241172

/-- Prove that the number of initial overs played was 10. -/
theorem cricket_initial_overs 
  (target : ℝ) 
  (initial_run_rate : ℝ) 
  (remaining_run_rate : ℝ) 
  (remaining_overs : ℕ)
  (h_target : target = 282)
  (h_initial_run_rate : initial_run_rate = 4.6)
  (h_remaining_run_rate : remaining_run_rate = 5.9)
  (h_remaining_overs : remaining_overs = 40) 
  : ∃ x : ℝ, x = 10 := 
by
  sorry

end cricket_initial_overs_l241_241172


namespace angle_BMA_half_arc_diff_l241_241820

variable {C : Type} [MetricSpace C] [NormedAddTorsor ℝ C] -- Circle context
variables (O A B M C : C) -- Points involved

-- Condition: M is the point of tangency, MB is tangent, MAC is secant
def is_tangent (O M B : C) : Prop := sorry -- Property defining tangency at M
def is_secant (O M A C : C) : Prop := sorry -- Property defining secancy through A and C
def arc_length (O A B : C) : ℝ := sorry -- Length of the arc AB in circle centered at O

theorem angle_BMA_half_arc_diff
  (h_tangent : is_tangent O M B)
  (h_secant : is_secant O M A C) :
  let θ := angle O B A in -- Angle BAC = θ
  let φ := angle O A C in -- Angle BMA = φ
  φ = 1 / 2 * (arc_length O B C - arc_length O A B) := sorry

end angle_BMA_half_arc_diff_l241_241820


namespace binom_12_10_eq_66_l241_241462

theorem binom_12_10_eq_66 : nat.choose 12 10 = 66 :=
by
  sorry

end binom_12_10_eq_66_l241_241462


namespace max_value_expression_l241_241927

theorem max_value_expression (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) :
  ∃ M, M = 3 ∧ ∀ a b c, 0 < a → 0 < b → 0 < c → 
    let A := (a^3 * (b + c) + b^3 * (c + a) + c^3 * (a + b)) / 
             ((a + b + c)^4 - 79 * (a * b * c)^(4/3))
    A ≤ M := 
begin
  use 3,
  sorry
end

end max_value_expression_l241_241927


namespace two_person_subcommittees_from_eight_l241_241509

theorem two_person_subcommittees_from_eight :
  (Nat.choose 8 2) = 28 :=
by
  sorry

end two_person_subcommittees_from_eight_l241_241509


namespace not_multiple_of_121_l241_241197

theorem not_multiple_of_121 (n : ℤ) : ¬ ∃ k : ℤ, n^2 + 2*n + 12 = 121*k := 
sorry

end not_multiple_of_121_l241_241197


namespace conditional_probability_l241_241300

-- Definitions of the events and probabilities given in the conditions
def event_A (red : ℕ) : Prop := red % 3 = 0
def event_B (red blue : ℕ) : Prop := red + blue > 8

-- The actual values of probabilities calculated in the solution
def P_A : ℚ := 1/3
def P_B : ℚ := 1/3
def P_AB : ℚ := 5/36

-- Definition of conditional probability
def P_B_given_A : ℚ := P_AB / P_A

-- The claim we want to prove
theorem conditional_probability :
  P_B_given_A = 5 / 12 :=
sorry

end conditional_probability_l241_241300


namespace largest_angle_of_pentagon_l241_241795

-- Define the angles of the pentagon and the conditions on them.
def is_angle_of_pentagon (A B C D E : ℝ) :=
  A = 108 ∧ B = 72 ∧ C = D ∧ E = 3 * C ∧
  A + B + C + D + E = 540

-- Prove the largest angle in the pentagon is 216
theorem largest_angle_of_pentagon (A B C D E : ℝ) (h : is_angle_of_pentagon A B C D E) :
  max (max (max (max A B) C) D) E = 216 :=
by
  sorry

end largest_angle_of_pentagon_l241_241795


namespace problem_equivalent_l241_241138

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 2) / Real.log 4 - 1

theorem problem_equivalent : 
  (∀ x : ℝ, f x = f (-x)) →
  (∀ x : ℝ, f x = Real.log (x + 2) / Real.log 4 - 1) →
  {x : ℝ | f (x - 2) > 0} = {x | x < 0 ∨ x > 4} :=
by
  intro h_even h_def
  sorry

end problem_equivalent_l241_241138


namespace problem_a_problem_b_problem_c_l241_241362

open Finset

-- Part (a)
theorem problem_a : (card {s : finset (fin 10) | s.card = 2 ∧ (∃ i j, i ≠ j ∧
  (s = {i, j} ∧ (i + 5) % 10 = j ∨ (j + 5) % 10 = i))}) / (choose 10 2) = 1 / 9 := sorry

-- Part (b)
theorem problem_b : (card {s : finset (fin 10) | s.card = 3 ∧ (∃ i j k, 
  (s = {i, j, k}) ∧ (i ≠ j) ∧ (j ≠ k) ∧ (k ≠ i) ∧ 
  (i + 5) % 10 ≠ j ∧ (j + 5) % 10 ≠ k ∧ (k + 5) % 10 ≠ i ∧ 
  (((i + 5) % 10 = k ∨ (k + 5) % 10 = i ∨ (j + 5) % 10 = i) ∨ 
  ((i - 5) % 10 = j ∨ (j - 5) % 10 = k ∨ (k - 5) % 10 = i)) )}) / 
  (choose 10 3) = 1 / 3 := sorry

-- Part (c)
theorem problem_c : (card {s : finset (fin 10) | s.card = 4 ∧ (∃ i j k l, 
  i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i ∧ (s = {i, j, k, l} ∧ 
  (({i, j} ∪ {k, l} ⊆ {0, 5}) ∨ ({i, k} ∪ {j, l} ⊆ {1, 6}) ∨ 
  ({i, l} ∪ {j, k} ⊆ {2, 7}) ∨ ({i, j} ∪ {k, l} ⊆ {3, 8}) ∨ 
  ({i, k} ∪ {j, l} ⊆ {4, 9})))}) / (choose 10 4) = 1 / 21 := sorry

end problem_a_problem_b_problem_c_l241_241362


namespace find_length_AE_l241_241074

theorem find_length_AE (AB BC CD DE AC CE AE : ℕ) 
  (h1 : AB = 2) 
  (h2 : BC = 2) 
  (h3 : CD = 5) 
  (h4 : DE = 7)
  (h5 : AC > 2) 
  (h6 : AC < 4) 
  (h7 : CE > 2) 
  (h8 : CE < 5)
  (h9 : AC ≠ CE)
  (h10 : AC ≠ AE)
  (h11 : CE ≠ AE)
  : AE = 5 :=
sorry

end find_length_AE_l241_241074


namespace factor_probability_l241_241262

theorem factor_probability : 
  let S := { n : ℕ | n > 0 ∧ n ≤ 36 } in
  let factor_count := (multiplicity 2 36 + 1) * (multiplicity 3 36 + 1) in
  let total_count := 36 in
  (factor_count : ℚ) / total_count = (1 : ℚ) / 4 := 
by
  sorry

end factor_probability_l241_241262


namespace boat_speed_in_still_water_l241_241068

-- Define the conditions
def speed_of_stream : ℝ := 3 -- (speed in km/h)
def time_downstream : ℝ := 1 -- (time in hours)
def time_upstream : ℝ := 1.5 -- (time in hours)

-- Define the goal by proving the speed of the boat in still water
theorem boat_speed_in_still_water : 
  ∃ V_b : ℝ, (V_b + speed_of_stream) * time_downstream = (V_b - speed_of_stream) * time_upstream ∧ V_b = 15 :=
by
  sorry -- (Proof will be provided here)

end boat_speed_in_still_water_l241_241068


namespace number_of_boxes_l241_241848

-- Define the conditions
def apples_per_crate : ℕ := 180
def number_of_crates : ℕ := 12
def rotten_apples : ℕ := 160
def apples_per_box : ℕ := 20

-- Define the statement to prove
theorem number_of_boxes : (apples_per_crate * number_of_crates - rotten_apples) / apples_per_box = 100 := 
by 
  sorry -- Proof skipped

end number_of_boxes_l241_241848


namespace total_lives_remaining_l241_241995

theorem total_lives_remaining (initial_players quit_players : Nat) 
  (lives_3_players lives_4_players lives_2_players bonus_lives : Nat)
  (h1 : initial_players = 16)
  (h2 : quit_players = 7)
  (h3 : lives_3_players = 10)
  (h4 : lives_4_players = 8)
  (h5 : lives_2_players = 6)
  (h6 : bonus_lives = 4)
  (remaining_players : Nat)
  (h7 : remaining_players = initial_players - quit_players)
  (lives_before_bonus : Nat)
  (h8 : lives_before_bonus = 3 * lives_3_players + 4 * lives_4_players + 2 * lives_2_players)
  (bonus_total : Nat)
  (h9 : bonus_total = remaining_players * bonus_lives) :
  3 * lives_3_players + 4 * lives_4_players + 2 * lives_2_players + remaining_players * bonus_lives = 110 :=
by
  sorry

end total_lives_remaining_l241_241995


namespace compare_negatives_l241_241324

theorem compare_negatives : -1 > -2 := 
by 
  sorry

end compare_negatives_l241_241324


namespace probability_factor_of_36_l241_241249

theorem probability_factor_of_36 :
  (∃ n ∈ finset.range 37, (∃ k : ℕ, 36 = k * n)) → 9 / 36 = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l241_241249


namespace expected_reflection_value_l241_241020

noncomputable def expected_reflections : ℝ :=
  (2 / Real.pi) *
  (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4))

theorem expected_reflection_value :
  expected_reflections = (2 / Real.pi) *
    (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4)) :=
by
  sorry

end expected_reflection_value_l241_241020


namespace factor_probability_l241_241263

theorem factor_probability : 
  let S := { n : ℕ | n > 0 ∧ n ≤ 36 } in
  let factor_count := (multiplicity 2 36 + 1) * (multiplicity 3 36 + 1) in
  let total_count := 36 in
  (factor_count : ℚ) / total_count = (1 : ℚ) / 4 := 
by
  sorry

end factor_probability_l241_241263


namespace add_decimals_l241_241321

theorem add_decimals : 5.763 + 2.489 = 8.252 := 
by
  sorry

end add_decimals_l241_241321


namespace partition_nats_100_subsets_l241_241557

theorem partition_nats_100_subsets :
  ∃ (S : ℕ → ℕ), (∀ n, 1 ≤ S n ∧ S n ≤ 100) ∧
    (∀ a b c : ℕ, a + 99 * b = c → S a = S c ∨ S a = S b ∨ S b = S c) :=
by
  sorry

end partition_nats_100_subsets_l241_241557


namespace bus_speed_excluding_stoppages_l241_241330

variable (v : ℝ)

-- Given conditions
def speed_including_stoppages := 45 -- kmph
def stoppage_time_ratio := 1/6 -- 10 minutes per hour is 1/6 of the time

-- Prove that the speed excluding stoppages is 54 kmph
theorem bus_speed_excluding_stoppages (h1 : speed_including_stoppages = 45) 
                                      (h2 : stoppage_time_ratio = 1/6) : 
                                      v = 54 := by
  sorry

end bus_speed_excluding_stoppages_l241_241330


namespace min_value_at_x_zero_l241_241773

noncomputable def f (x : ℝ) := Real.sqrt (x^2 + (x + 1)^2) + Real.sqrt (x^2 + (x - 1)^2)

theorem min_value_at_x_zero : ∀ x : ℝ, f x ≥ f 0 := by
  sorry

end min_value_at_x_zero_l241_241773


namespace binomial_coefficient_12_10_l241_241470

theorem binomial_coefficient_12_10 : Nat.choose 12 10 = 66 := by
  sorry  -- The actual proof is omitted as instructed.

end binomial_coefficient_12_10_l241_241470


namespace pictures_per_album_l241_241737

theorem pictures_per_album (phone_pics camera_pics albums pics_per_album : ℕ)
  (h1 : phone_pics = 7) (h2 : camera_pics = 13) (h3 : albums = 5)
  (h4 : pics_per_album * albums = phone_pics + camera_pics) :
  pics_per_album = 4 :=
by
  sorry

end pictures_per_album_l241_241737


namespace cyclic_determinant_zero_l241_241962

open Matrix

-- Define the roots of the polynomial and the polynomial itself.
variables {α β γ δ : ℂ} -- We assume the roots are complex numbers.
variable (p q r : ℂ) -- Coefficients of the polynomial x^4 + px^2 + qx + r = 0

-- Define the matrix whose determinant we want to compute
def cyclic_matrix (α β γ δ : ℂ) : Matrix (Fin 4) (Fin 4) ℂ :=
  ![
    ![α, β, γ, δ],
    ![β, γ, δ, α],
    ![γ, δ, α, β],
    ![δ, α, β, γ]
  ]

-- Statement of the theorem
theorem cyclic_determinant_zero :
  ∀ (α β γ δ : ℂ) (p q r : ℂ),
  (∀ x : ℂ, x ^ 4 + p * x ^ 2 + q * x + r = 0 → x = α ∨ x = β ∨ x = γ ∨ x = δ) →
  det (cyclic_matrix α β γ δ) = 0 :=
by
  intros α β γ δ p q r hRoots
  sorry

end cyclic_determinant_zero_l241_241962


namespace smallest_sum_symmetrical_dice_l241_241846

theorem smallest_sum_symmetrical_dice (p : ℝ) (N : ℕ) (h₁ : p > 0) (h₂ : 6 * N = 2022) : N = 337 := 
by
  -- Proof can be filled in here
  sorry

end smallest_sum_symmetrical_dice_l241_241846


namespace expectation_of_binomial_l241_241936

noncomputable def binomial_expectation (n : ℕ) (p : ℝ) : ℝ := n * p

theorem expectation_of_binomial :
  binomial_expectation 6 (1/3) = 2 :=
by
  sorry

end expectation_of_binomial_l241_241936


namespace abs_z_bounds_l241_241780

open Complex

theorem abs_z_bounds (z : ℂ) (h : abs (z + 1/z) = 1) : 
  (Real.sqrt 5 - 1) / 2 ≤ abs z ∧ abs z ≤ (Real.sqrt 5 + 1) / 2 := 
sorry

end abs_z_bounds_l241_241780


namespace arithmetic_sequence_term_count_l241_241907

def first_term : ℕ := 5
def common_difference : ℕ := 3
def last_term : ℕ := 203

theorem arithmetic_sequence_term_count :
  ∃ n : ℕ, last_term = first_term + (n - 1) * common_difference ∧ n = 67 :=
by
  sorry

end arithmetic_sequence_term_count_l241_241907


namespace sum_of_roots_of_equation_l241_241720

theorem sum_of_roots_of_equation : 
  (∀ x, 5 = (x^3 - 2*x^2 - 8*x) / (x + 2)) → 
  (∃ x1 x2, (5 = x1) ∧ (5 = x2) ∧ (x1 + x2 = 4)) := 
by
  sorry

end sum_of_roots_of_equation_l241_241720


namespace probability_factor_of_36_l241_241267

theorem probability_factor_of_36 :
  let N := 36
  let S := {n : ℕ | n > 0 ∧ n ≤ N}
  let E := {n : ℕ | n ∈ S ∧ N % n = 0}
  (E.card : ℚ) / S.card = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l241_241267


namespace probability_factor_of_36_l241_241248

theorem probability_factor_of_36 :
  (∃ n ∈ finset.range 37, (∃ k : ℕ, 36 = k * n)) → 9 / 36 = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l241_241248


namespace probability_five_chords_form_convex_pentagon_l241_241480

-- Definitions of problem conditions
variable (n : ℕ) (k : ℕ)

-- Eight points on a circle
def points_on_circle : ℕ := 8

-- Number of chords selected
def selected_chords : ℕ := 5

-- Total number of ways to select 5 chords from 28 possible chords
def total_ways : ℕ := Nat.choose 28 5

-- Number of ways to select 5 points from 8, forming a convex pentagon
def favorable_ways : ℕ := Nat.choose 8 5

-- The probability computation
def probability_pentagon (total : ℕ) (favorable : ℕ) : ℚ :=
  favorable / total

theorem probability_five_chords_form_convex_pentagon :
  probability_pentagon total_ways favorable_ways = 1 / 1755 :=
by
  sorry

end probability_five_chords_form_convex_pentagon_l241_241480


namespace b_investment_l241_241731

noncomputable def B_share := 880
noncomputable def A_share := 560
noncomputable def A_investment := 7000
noncomputable def C_investment := 18000
noncomputable def total_investment (B: ℝ) := A_investment + B + C_investment

theorem b_investment (B : ℝ) (P : ℝ)
    (h1 : 7000 / total_investment B * P = A_share)
    (h2 : B / total_investment B * P = B_share) : B = 8000 :=
by
  sorry

end b_investment_l241_241731


namespace probability_of_reaching_last_floor_l241_241874

noncomputable def probability_of_open_paths (n : ℕ) : ℝ :=
  2^(n-1) / (Nat.choose (2*(n-1)) (n-1))

theorem probability_of_reaching_last_floor (n : ℕ) :
  probability_of_open_paths n = 2^(n-1) / (Nat.choose (2*(n-1)) (n-1)) :=
by
  sorry

end probability_of_reaching_last_floor_l241_241874


namespace cubes_with_one_colored_face_l241_241698

theorem cubes_with_one_colored_face (n : ℕ) (c1 : ℕ) (c2 : ℕ) :
  (n = 64) ∧ (c1 = 4) ∧ (c2 = 4) → ((4 * n) * 2) / n = 32 :=
by 
  sorry

end cubes_with_one_colored_face_l241_241698


namespace solution_set_of_inequality_l241_241645

variable (a x : ℝ)

theorem solution_set_of_inequality (h : 0 < a ∧ a < 1) :
  (a - x) * (x - (1/a)) > 0 ↔ a < x ∧ x < 1/a :=
sorry

end solution_set_of_inequality_l241_241645


namespace probability_open_doors_l241_241880

variable (n : ℕ)

theorem probability_open_doors (h : n > 1) : 
  let num_doors := 2 * (n - 1)
      num_locked := n - 1
  in (2^(n-1) / (num_doors.choose num_locked) = 
  2^(n-1) / (nat.choose num_doors num_locked)) := sorry

end probability_open_doors_l241_241880


namespace two_person_subcommittees_l241_241505

def committee_size := 8
def sub_committee_size := 2
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def combination (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem two_person_subcommittees : combination committee_size sub_committee_size = 28 := by
  sorry

end two_person_subcommittees_l241_241505


namespace empire_state_building_height_l241_241034

theorem empire_state_building_height (h_top_floor : ℕ) (h_antenna_spire : ℕ) (total_height : ℕ) :
  h_top_floor = 1250 ∧ h_antenna_spire = 204 ∧ total_height = h_top_floor + h_antenna_spire → total_height = 1454 :=
by
  sorry

end empire_state_building_height_l241_241034


namespace pencil_count_l241_241841

/-- 
If there are initially 115 pencils in the drawer, and Sara adds 100 more pencils, 
then the total number of pencils in the drawer is 215.
-/
theorem pencil_count (initial_pencils added_pencils : ℕ) (h1 : initial_pencils = 115) (h2 : added_pencils = 100) : 
  initial_pencils + added_pencils = 215 := by
  sorry

end pencil_count_l241_241841


namespace probability_factor_36_l241_241278

def is_factor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

theorem probability_factor_36 : 
  (set.univ.filter (λ n : ℕ, n > 0 ∧ n ≤ 36 ∧ is_factor 36 n)).card.toRat / 36 = 1 / 4 := 
by
  sorry

end probability_factor_36_l241_241278


namespace prob_ace_then_king_l241_241710

theorem prob_ace_then_king :
  let total_cards := 52
  let total_aces := 4
  let total_kings := 4
  let prob_first_ace := total_aces / total_cards
  let prob_second_king := total_kings / (total_cards - 1)
  prob_first_ace * prob_second_king = 4 / 663 := by
{
  -- Definitions
  let total_cards := 52
  let total_aces := 4
  let total_kings := 4

  -- Calculation of probability
  let prob_first_ace := total_aces / total_cards
  let prob_second_king := total_kings / (total_cards - 1)
  have h : prob_first_ace * prob_second_king = 4 / 663 := by sorry,

  -- Return the result
  h,
}

end prob_ace_then_king_l241_241710


namespace inequality_abc_l241_241972

theorem inequality_abc
  (a b c : ℝ)
  (ha : 0 ≤ a) (ha_le : a ≤ 1)
  (hb : 0 ≤ b) (hb_le : b ≤ 1)
  (hc : 0 ≤ c) (hc_le : c ≤ 1) :
  a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 :=
sorry

end inequality_abc_l241_241972


namespace batsman_average_46_innings_l241_241079

variable (A : ℕ) (highest_score : ℕ) (lowest_score : ℕ) (average_excl : ℕ)
variable (n_innings n_without_highest_lowest : ℕ)

theorem batsman_average_46_innings
  (h_diff: highest_score - lowest_score = 190)
  (h_avg_excl: average_excl = 58)
  (h_highest: highest_score = 199)
  (h_innings: n_innings = 46)
  (h_innings_excl: n_without_highest_lowest = 44) :
  A = (44 * 58 + 199 + 9) / 46 := by
  sorry

end batsman_average_46_innings_l241_241079


namespace probability_factor_36_l241_241239

theorem probability_factor_36 : 
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 36 ∧ (∀ k : ℕ, k ∣ n ↔ k ∣ 36) → (n ≠ 0 → (∃ p : ℚ, p = 1 / 4))) :=
begin
  sorry
end

end probability_factor_36_l241_241239


namespace intersection_eq_l241_241540

namespace SetIntersection

open Set

-- Definitions of sets A and B
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}

-- Prove the intersection of A and B is {1, 2}
theorem intersection_eq : A ∩ B = {1, 2} :=
by
  sorry

end SetIntersection

end intersection_eq_l241_241540


namespace bee_flight_time_l241_241885

theorem bee_flight_time (t : ℝ) : 
  let speed_daisy_to_rose := 2.6
  let speed_rose_to_poppy := speed_daisy_to_rose + 3
  let distance_daisy_to_rose := speed_daisy_to_rose * 10
  let distance_rose_to_poppy := distance_daisy_to_rose - 8
  distance_rose_to_poppy = speed_rose_to_poppy * t
  ∧ abs (t - 3) < 1 := 
sorry

end bee_flight_time_l241_241885


namespace assign_grades_l241_241747

-- Definitions based on the conditions:
def num_students : ℕ := 12
def num_grades : ℕ := 4

-- Statement of the theorem
theorem assign_grades : num_grades ^ num_students = 16777216 := by
  sorry

end assign_grades_l241_241747


namespace Maria_soap_cost_l241_241383
-- Import the entire Mathlib library
  
theorem Maria_soap_cost (soap_last_months : ℕ) (cost_per_bar : ℝ) (months_in_year : ℕ):
  (soap_last_months = 2) -> 
  (cost_per_bar = 8.00) ->
  (months_in_year = 12) -> 
  (months_in_year / soap_last_months * cost_per_bar = 48.00) := 
by
  intros h_soap_last h_cost h_year
  sorry

end Maria_soap_cost_l241_241383


namespace number_of_common_tangents_l241_241501

def circleM (x y : ℝ) : Prop := x^2 + y^2 - 4 * y = 0
def circleN (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

theorem number_of_common_tangents : ∃ n : ℕ, n = 2 ∧ 
  (∀ x y : ℝ, circleM x y → circleN x y → false) :=
by
  sorry

end number_of_common_tangents_l241_241501


namespace molecular_weight_of_oxygen_part_l241_241338

-- Define the known variables as constants
def atomic_weight_oxygen : ℝ := 16.00
def num_oxygen_atoms : ℕ := 2
def molecular_weight_compound : ℝ := 88.00

-- Define the problem as a theorem
theorem molecular_weight_of_oxygen_part :
  16.00 * 2 = 32.00 :=
by
  -- The proof will be filled in here
  sorry

end molecular_weight_of_oxygen_part_l241_241338


namespace vehicles_travelled_last_year_l241_241183

theorem vehicles_travelled_last_year (V : ℕ) : 
  (∀ (x : ℕ), (96 : ℕ) * (V / 100000000) = 2880) → V = 3000000000 := 
by 
  sorry

end vehicles_travelled_last_year_l241_241183


namespace smallest_visible_sum_of_3x3x3_cube_is_90_l241_241866

theorem smallest_visible_sum_of_3x3x3_cube_is_90 
: ∀ (dices: Fin 27 → Fin 6 → ℕ),
    (∀ i j k, dices (3*i+j) k = 7 - dices (3*i+j) (5-k)) → 
    (∃ s, s = 90 ∧
    s = (8 * (dices 0 0 + dices 0 1 + dices 0 2)) + 
        (12 * (dices 0 0 + dices 0 1)) +
        (6 * (dices 0 0))) := sorry

end smallest_visible_sum_of_3x3x3_cube_is_90_l241_241866


namespace Jennifer_more_boxes_l241_241182

-- Definitions based on conditions
def Kim_boxes : ℕ := 54
def Jennifer_boxes : ℕ := 71

-- Proof statement (no actual proof needed, just the statement)
theorem Jennifer_more_boxes : Jennifer_boxes - Kim_boxes = 17 := by
  sorry

end Jennifer_more_boxes_l241_241182


namespace binomial_12_10_eq_66_l241_241457

theorem binomial_12_10_eq_66 : (Nat.choose 12 10) = 66 :=
by
  sorry

end binomial_12_10_eq_66_l241_241457


namespace probability_crane_reaches_lily_pad_14_l241_241054

theorem probability_crane_reaches_lily_pad_14 :
  let num_pads := 16
  let predators := {4, 7, 12}
  let food_pad := 14
  let start_pad := 0
  let hop_prob := (1 : ℚ) / 2
  let leap_prob := (1 : ℚ) / 2
  let reach_prob := (3 : ℚ) / 512
  start_pad < num_pads ∧ food_pad < num_pads ∧ (∀ p ∈ predators, p < num_pads)
  → ∃ path: list ℕ, (path.head = some start_pad ∧ path.last = some food_pad ∧ path.forall (λ p, p ∉ predators))
  → (reach_prob = 3 / 512 : ℚ) := by
  sorry

end probability_crane_reaches_lily_pad_14_l241_241054


namespace complement_union_l241_241941

noncomputable def A : Set ℝ := { x : ℝ | x^2 - x - 2 ≤ 0 }
noncomputable def B : Set ℝ := { x : ℝ | 1 < x ∧ x ≤ 3 }
noncomputable def CR (S : Set ℝ) : Set ℝ := { x : ℝ | x ∉ S }

theorem complement_union (A B : Set ℝ) :
  (CR A ∪ B) = (Set.univ \ A ∪ Set.Ioo 1 3) := by
  sorry

end complement_union_l241_241941


namespace decreasing_function_l241_241496

def f (a x : ℝ) : ℝ := a * x^3 - x

theorem decreasing_function (a : ℝ) 
  (h : ∀ x y : ℝ, x < y → f a y ≤ f a x) : a ≤ 0 :=
by
  sorry

end decreasing_function_l241_241496


namespace ticket_cost_l241_241110

noncomputable def calculate_cost (x : ℝ) : ℝ :=
  6 * (1.1 * x) + 5 * (x / 2)

theorem ticket_cost (x : ℝ) (h : 4 * (1.1 * x) + 3 * (x / 2) = 28.80) : 
  calculate_cost x = 44.41 := by
  sorry

end ticket_cost_l241_241110


namespace ricardo_coins_difference_l241_241558

theorem ricardo_coins_difference :
  ∃ (x y : ℕ), (x + y = 2020) ∧ (x ≥ 1) ∧ (y ≥ 1) ∧ ((5 * x + y) - (x + 5 * y) = 8072) :=
by
  sorry

end ricardo_coins_difference_l241_241558


namespace probability_factor_of_36_l241_241250

theorem probability_factor_of_36 : 
  (finset.card (finset.filter (λ x : ℕ, 36 % x = 0) (finset.range (36 + 1))) : ℚ) / 36 = 1 / 4 := 
by 
sorry

end probability_factor_of_36_l241_241250


namespace degree_equality_l241_241934

theorem degree_equality (m : ℕ) :
  (∀ x y z : ℕ, 2 + 4 = 1 + (m + 2)) → 3 * m - 2 = 7 :=
by
  intro h
  sorry

end degree_equality_l241_241934


namespace sequence_bound_l241_241632

theorem sequence_bound (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) (h_seq : ∀ n, (a n) ^ 2 ≤ a (n + 1)) :
  ∀ n, a n < 1 / n :=
by
  intros
  sorry

end sequence_bound_l241_241632


namespace total_interest_paid_l241_241375

-- Define the problem as a theorem in Lean 4
theorem total_interest_paid
  (initial_investment : ℝ)
  (interest_6_months : ℝ)
  (interest_10_months : ℝ)
  (interest_18_months : ℝ)
  (total_interest : ℝ) :
  initial_investment = 10000 ∧ 
  interest_6_months = 0.02 * initial_investment ∧
  interest_10_months = 0.03 * (initial_investment + interest_6_months) ∧
  interest_18_months = 0.04 * (initial_investment + interest_6_months + interest_10_months) ∧
  total_interest = interest_6_months + interest_10_months + interest_18_months →
  total_interest = 926.24 :=
by
  sorry

end total_interest_paid_l241_241375


namespace sum_of_palindromes_l241_241983

theorem sum_of_palindromes (a b : ℕ) (ha : a > 99) (ha' : a < 1000) (hb : b > 99) (hb' : b < 1000) 
  (hpal_a : ∀ i j k, a = 100*i + 10*j + k → a = 100*k + 10*j + i) 
  (hpal_b : ∀ i j k, b = 100*i + 10*j + k → b = 100*k + 10*j + i) 
  (hprod : a * b = 589185) : a + b = 1534 :=
sorry

end sum_of_palindromes_l241_241983


namespace Mickey_less_than_twice_Minnie_l241_241610

def Minnie_horses_per_day : ℕ := 10
def Mickey_horses_per_day : ℕ := 14

theorem Mickey_less_than_twice_Minnie :
  2 * Minnie_horses_per_day - Mickey_horses_per_day = 6 := by
  sorry

end Mickey_less_than_twice_Minnie_l241_241610


namespace tommy_total_balloons_l241_241413

-- Define the conditions from part (a)
def original_balloons : Nat := 26
def additional_balloons : Nat := 34

-- Define the proof problem from part (c)
theorem tommy_total_balloons : original_balloons + additional_balloons = 60 := by
  -- Skip the actual proof
  sorry

end tommy_total_balloons_l241_241413


namespace addition_addends_l241_241387

theorem addition_addends (a b : ℕ) (c₁ c₂ : ℕ) (d : ℕ) : 
  a + b = c₁ ∧ a + (b - d) = c₂ ∧ d = 50 ∧ c₁ = 982 ∧ c₂ = 577 → 
  a = 450 ∧ b = 532 :=
by
  sorry

end addition_addends_l241_241387


namespace Jamal_crayon_cost_l241_241022

/-- Jamal bought 4 half dozen colored crayons at $2 per crayon. 
    He got a 10% discount on the total cost, and an additional 5% discount on the remaining amount. 
    After paying in US Dollars (USD), we want to know how much he spent in Euros (EUR) and British Pounds (GBP) 
    given that 1 USD is equal to 0.85 EUR and 1 USD is equal to 0.75 GBP. 
    This statement proves that the total cost was 34.884 EUR and 30.78 GBP. -/
theorem Jamal_crayon_cost :
  let number_of_crayons := 4 * 6
  let initial_cost := number_of_crayons * 2
  let first_discount := 0.10 * initial_cost
  let cost_after_first_discount := initial_cost - first_discount
  let second_discount := 0.05 * cost_after_first_discount
  let final_cost_usd := cost_after_first_discount - second_discount
  let final_cost_eur := final_cost_usd * 0.85
  let final_cost_gbp := final_cost_usd * 0.75
  final_cost_eur = 34.884 ∧ final_cost_gbp = 30.78 := 
by
  sorry

end Jamal_crayon_cost_l241_241022


namespace pencils_placed_by_sara_l241_241573

theorem pencils_placed_by_sara (initial_pencils final_pencils : ℕ) (h1 : initial_pencils = 115) (h2 : final_pencils = 215) : final_pencils - initial_pencils = 100 := by
  sorry

end pencils_placed_by_sara_l241_241573


namespace problem_intersection_l241_241356

noncomputable def A (x : ℝ) : Prop := 1 < x ∧ x < 4
noncomputable def B (x : ℝ) : Prop := 0 < x ∧ x < 2

theorem problem_intersection : {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | 1 < x ∧ x < 2} :=
by sorry

end problem_intersection_l241_241356


namespace maximum_ab_value_l241_241939

noncomputable def ab_max (a b : ℝ) : ℝ :=
  if a > 0 then 2 * a * a - a * a * Real.log a else 0

theorem maximum_ab_value : ∀ (a b : ℝ), (∀ (x : ℝ), (Real.exp x - a * x + a) ≥ b) →
   ab_max a b ≤ if a = Real.exp (3 / 2) then (Real.exp 3) / 2 else sorry :=
by
  intros a b h
  sorry

end maximum_ab_value_l241_241939


namespace perp_bisector_eq_l241_241695

/-- The circles x^2+y^2=4 and x^2+y^2-4x+6y=0 intersect at points A and B. 
Find the equation of the perpendicular bisector of line segment AB. -/

theorem perp_bisector_eq : 
  let C1 := (0, 0)
  let C2 := (2, -3)
  ∃ (a b c : ℝ), a = 3 ∧ b = 2 ∧ c = 0 ∧ ∀ (x y : ℝ), a * x + b * y + c = 0 := 
by
  sorry

end perp_bisector_eq_l241_241695


namespace mike_travel_miles_l241_241192

theorem mike_travel_miles
  (toll_fees_mike : ℝ) (toll_fees_annie : ℝ) (mike_start_fee : ℝ) 
  (annie_start_fee : ℝ) (mike_per_mile : ℝ) (annie_per_mile : ℝ) 
  (annie_travel_time : ℝ) (annie_speed : ℝ) (mike_cost : ℝ) 
  (annie_cost : ℝ) 
  (h_mike_cost_eq : mike_cost = mike_start_fee + toll_fees_mike + mike_per_mile * 36)
  (h_annie_cost_eq : annie_cost = annie_start_fee + toll_fees_annie + annie_per_mile * annie_speed * annie_travel_time)
  (h_equal_costs : mike_cost = annie_cost)
  : 36 = 36 :=
by 
  sorry

end mike_travel_miles_l241_241192


namespace determine_positive_integers_l241_241761

theorem determine_positive_integers (x y z : ℕ) (h : x^2 + y^2 - 15 = 2^z) :
  (x = 0 ∧ y = 4 ∧ z = 0) ∨ (x = 4 ∧ y = 0 ∧ z = 0) ∨
  (x = 1 ∧ y = 4 ∧ z = 1) ∨ (x = 4 ∧ y = 1 ∧ z = 1) :=
sorry

end determine_positive_integers_l241_241761


namespace inequality_proof_l241_241500

theorem inequality_proof (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 1) :
  (x * y / Real.sqrt (x * y + y * z) + y * z / Real.sqrt (y * z + z * x) + z * x / Real.sqrt (z * x + x * y)) 
  ≤ (Real.sqrt 2) / 2 :=
by
  sorry

end inequality_proof_l241_241500


namespace spherical_to_rectangular_conversion_l241_241326

-- Define spherical to rectangular coordinate conversion
def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_conversion :
  spherical_to_rectangular 4 π (π / 3) = (-2 * Real.sqrt 3, 0, 2) :=
by
  sorry

end spherical_to_rectangular_conversion_l241_241326


namespace pages_left_after_all_projects_l241_241701

-- Definitions based on conditions
def initial_pages : ℕ := 120
def pages_for_science : ℕ := (initial_pages * 25) / 100
def pages_for_math : ℕ := 10
def pages_after_science_and_math : ℕ := initial_pages - pages_for_science - pages_for_math
def pages_for_history : ℕ := (initial_pages * 15) / 100
def pages_after_history : ℕ := pages_after_science_and_math - pages_for_history
def remaining_pages : ℕ := pages_after_history / 2

theorem pages_left_after_all_projects :
  remaining_pages = 31 :=
  by
  sorry

end pages_left_after_all_projects_l241_241701


namespace find_number_l241_241436

theorem find_number (x : ℤ) (h1 : x - 2 + 4 = 9) : x = 7 :=
by
  sorry

end find_number_l241_241436


namespace common_integer_count_l241_241201

open Set
open Polynomial

/-- Define Set A as the set of integers from 3 to 30 inclusive -/
def SetA : Set ℤ := {i | 3 ≤ i ∧ i ≤ 30}

/-- Define Set B as the set of integers from 10 to 40 inclusive -/
def SetB : Set ℤ := {i | 10 ≤ i ∧ i ≤ 40}

/-- Define the condition polynomial f(i) = i^2 - 5i - 6 -/
def condition (i : ℤ) : Prop := eval i (X^2 - 5 * X - 6) = 0

/-- The number of distinct integers i that belong to both Set A and Set B and satisfy the condition equals 0 -/
theorem common_integer_count : 
  card {i | i ∈ SetA ∧ i ∈ SetB ∧ condition i} = 0 :=
sorry

end common_integer_count_l241_241201


namespace probability_diff_color_balls_l241_241991

-- Definition of the problem
def total_balls := 3 + 2
def total_pairs := (total_balls * (total_balls - 1)) / 2

def white_balls := 3
def black_balls := 2

def different_color_pairs := white_balls * black_balls

-- Stating the theorem
theorem probability_diff_color_balls :
  (different_color_pairs : ℚ) / total_pairs = 3 / 5 :=
by
  sorry

end probability_diff_color_balls_l241_241991


namespace typeB_lines_l241_241644

noncomputable def isTypeBLine (line : Real → Real) : Prop :=
  ∃ P : ℝ × ℝ, line P.1 = P.2 ∧ (Real.sqrt ((P.1 + 5)^2 + P.2^2) - Real.sqrt ((P.1 - 5)^2 + P.2^2) = 6)

theorem typeB_lines :
  isTypeBLine (fun x => x + 1) ∧ isTypeBLine (fun x => 2) :=
by sorry

end typeB_lines_l241_241644


namespace find_y_l241_241888

noncomputable def similar_triangles (a b x z : ℝ) :=
  (a / x = b / z)

theorem find_y 
  (a b x z : ℝ)
  (ha : a = 12)
  (hb : b = 9)
  (hz : z = 7)
  (h_sim : similar_triangles a b x z) :
  x = 28 / 3 :=
begin
  subst ha,
  subst hb,
  subst hz,
  unfold similar_triangles at h_sim,
  field_simp [h_sim],
  ring,
end

end find_y_l241_241888


namespace number_of_2_dollar_socks_l241_241559

-- Given conditions
def total_pairs (a b c : ℕ) := a + b + c = 15
def total_cost (a b c : ℕ) := 2 * a + 4 * b + 5 * c = 41
def min_each_pair (a b c : ℕ) := a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1

-- To be proved
theorem number_of_2_dollar_socks (a b c : ℕ) (h1 : total_pairs a b c) (h2 : total_cost a b c) (h3 : min_each_pair a b c) : 
  a = 11 := 
  sorry

end number_of_2_dollar_socks_l241_241559


namespace expression_value_l241_241855

theorem expression_value : 3 * (15 + 7)^2 - (15^2 + 7^2) = 1178 := by
    sorry

end expression_value_l241_241855


namespace three_digit_numbers_with_product_24_l241_241833

-- Definitions based on the conditions
def digits (n : ℕ) : Finset ℕ :=
{ k | k < 10 }

-- Condition: Product of digits equals 24
def product_of_digits_eq_24 (n : ℕ) : Prop :=
  (digits n).prod id = 24

-- Statement of the problem
theorem three_digit_numbers_with_product_24 :
  (Finset.card { n : ℕ | n ≥ 100 ∧ n < 1000 ∧ product_of_digits_eq_24 n } = 21) :=
by {
  sorry  -- proof omitted
}

end three_digit_numbers_with_product_24_l241_241833


namespace kiana_and_her_siblings_age_sum_l241_241376

theorem kiana_and_her_siblings_age_sum :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 256 ∧ a + b + c = 38 :=
by
sorry

end kiana_and_her_siblings_age_sum_l241_241376


namespace total_percentage_of_failed_candidates_is_correct_l241_241661

def total_candidates : ℕ := 2000
def number_of_girls : ℕ := 900
def number_of_boys : ℕ := total_candidates - number_of_girls
def percentage_boys_passed : ℚ := 38 / 100
def percentage_girls_passed : ℚ := 32 / 100
def number_of_boys_passed : ℚ := percentage_boys_passed * number_of_boys
def number_of_girls_passed : ℚ := percentage_girls_passed * number_of_girls
def total_candidates_passed : ℚ := number_of_boys_passed + number_of_girls_passed
def total_candidates_failed : ℚ := total_candidates - total_candidates_passed
def total_percentage_failed : ℚ := (total_candidates_failed / total_candidates) * 100

theorem total_percentage_of_failed_candidates_is_correct :
  total_percentage_failed = 64.7 := by
  sorry

end total_percentage_of_failed_candidates_is_correct_l241_241661


namespace selling_price_is_correct_l241_241346

def wholesale_cost : ℝ := 24.35
def gross_profit_percentage : ℝ := 0.15

def gross_profit : ℝ := gross_profit_percentage * wholesale_cost
def selling_price : ℝ := wholesale_cost + gross_profit

theorem selling_price_is_correct :
  selling_price = 28.00 :=
by
  sorry

end selling_price_is_correct_l241_241346


namespace original_wire_length_l241_241093

theorem original_wire_length (S L : ℝ) (h1: S = 30) (h2: S = (3 / 5) * L) : S + L = 80 := by 
  sorry

end original_wire_length_l241_241093


namespace not_basic_logic_structure_l241_241425

def SequenceStructure : Prop := true
def ConditionStructure : Prop := true
def LoopStructure : Prop := true
def DecisionStructure : Prop := true

theorem not_basic_logic_structure : ¬ (SequenceStructure ∨ ConditionStructure ∨ LoopStructure) -> DecisionStructure := by
  sorry

end not_basic_logic_structure_l241_241425


namespace probability_of_selecting_shirt_short_sock_l241_241011

/-
  Define the problem setup:
  6 shirts, 3 pairs of shorts, 8 pairs of socks.
  Total articles of clothing: 17
-/

def total_articles := 6 + 3 + 8  -- total number of articles

def choose (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.fact n / (Nat.fact k * Nat.fact (n - k))) else 0

def total_ways_to_choose_four : ℕ :=
  choose total_articles 4

/-
  Define the favorable outcomes:
  1. At least one shirt.
  2. Exactly one pair of shorts.
  3. Exactly one pair of socks.
-/
def favorable_outcomes : ℕ :=
  let shirts := 6
  let shorts := 3
  let socks := 8
  choose shorts 1 * choose socks 1 * (choose shirts 2 + choose shirts 1)

def expected_probability : ℚ :=
  favorable_outcomes / total_ways_to_choose_four

theorem probability_of_selecting_shirt_short_sock :
  expected_probability = 84 / 397 := by
  sorry

end probability_of_selecting_shirt_short_sock_l241_241011


namespace data_median_and_mode_l241_241492

theorem data_median_and_mode :
  let data := [3, 5, 7, 8, 8] in
  List.median data = 7 ∧ List.mode data = 8 :=
by
  sorry

end data_median_and_mode_l241_241492


namespace valid_triples_l241_241613

theorem valid_triples (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x ∣ (y + 1) ∧ y ∣ (z + 1) ∧ z ∣ (x + 1) ↔ (x, y, z) = (1, 1, 1) ∨ 
                                                      (x, y, z) = (1, 1, 2) ∨ 
                                                      (x, y, z) = (1, 3, 2) ∨ 
                                                      (x, y, z) = (3, 5, 4) :=
by
  sorry

end valid_triples_l241_241613


namespace min_value_fraction_ineq_l241_241345

-- Define the conditions and statement to be proved
theorem min_value_fraction_ineq (x : ℝ) (hx : x > 4) : 
  ∃ M, M = 4 * Real.sqrt 5 ∧ ∀ y : ℝ, y > 4 → (y + 16) / Real.sqrt (y - 4) ≥ M := 
sorry

end min_value_fraction_ineq_l241_241345


namespace girls_count_l241_241055

-- Define the constants according to the conditions
def boys_on_team : ℕ := 28
def groups : ℕ := 8
def members_per_group : ℕ := 4

-- Calculate the total number of members
def total_members : ℕ := groups * members_per_group

-- Calculate the number of girls by subtracting the number of boys from the total members
def girls_on_team : ℕ := total_members - boys_on_team

-- The proof statement: prove that the number of girls on the team is 4
theorem girls_count : girls_on_team = 4 := by
  -- Skip the proof, completing the statement
  sorry

end girls_count_l241_241055


namespace find_number_l241_241315

-- Definitions based on conditions
def sum : ℕ := 555 + 445
def difference : ℕ := 555 - 445
def quotient : ℕ := 2 * difference
def remainder : ℕ := 70
def divisor : ℕ := sum

-- Statement to be proved
theorem find_number : (divisor * quotient + remainder) = 220070 := by
  sorry

end find_number_l241_241315


namespace area_of_T_shaped_region_l241_241075

theorem area_of_T_shaped_region :
  let ABCD_area : ℝ := 48
  let EFHG_area : ℝ := 4
  let EFGI_area : ℝ := 8
  let EFCD_area : ℝ := 12
  (ABCD_area - (EFHG_area + EFGI_area + EFCD_area)) = 24 :=
by
  let ABCD_area : ℝ := 48
  let EFHG_area : ℝ := 4
  let EFGI_area : ℝ := 8
  let EFCD_area : ℝ := 12
  exact sorry

end area_of_T_shaped_region_l241_241075


namespace salty_cookies_initial_at_least_34_l241_241391

variable {S : ℕ}  -- S will represent the initial number of salty cookies

-- Conditions from the problem
def sweet_cookies_initial := 8
def sweet_cookies_ate := 20
def salty_cookies_ate := 34
def more_salty_than_sweet := 14

theorem salty_cookies_initial_at_least_34 :
  8 = sweet_cookies_initial ∧
  20 = sweet_cookies_ate ∧
  34 = salty_cookies_ate ∧
  salty_cookies_ate = sweet_cookies_ate + more_salty_than_sweet
  → S ≥ 34 :=
by sorry

end salty_cookies_initial_at_least_34_l241_241391


namespace solve_system_of_equations_l241_241693

theorem solve_system_of_equations : 
  ∃ (x y : ℤ), 2 * x + 5 * y = 8 ∧ 3 * x - 5 * y = -13 ∧ x = -1 ∧ y = 2 :=
by
  sorry

end solve_system_of_equations_l241_241693


namespace number_of_true_propositions_l241_241615

theorem number_of_true_propositions : 
  let original_p := ∀ (a : ℝ), a > -1 → a > -2
  let converse_p := ∀ (a : ℝ), a > -2 → a > -1
  let inverse_p := ∀ (a : ℝ), a ≤ -1 → a ≤ -2
  let contrapositive_p := ∀ (a : ℝ), a ≤ -2 → a ≤ -1
  (original_p ∧ contrapositive_p ∧ ¬converse_p ∧ ¬inverse_p) → (2 = 2) :=
by
  intros
  sorry

end number_of_true_propositions_l241_241615


namespace well_diameter_l241_241740

theorem well_diameter (V h : ℝ) (pi : ℝ) (r : ℝ) :
  h = 8 ∧ V = 25.132741228718345 ∧ pi = 3.141592653589793 ∧ V = pi * r^2 * h → 2 * r = 2 :=
by
  sorry

end well_diameter_l241_241740


namespace product_of_dice_divisible_by_9_l241_241235

-- Define the probability of rolling a number divisible by 3
def prob_roll_div_by_3 : ℚ := 1/6

-- Define the probability of rolling a number not divisible by 3
def prob_roll_not_div_by_3 : ℚ := 2/3

-- Define the probability that the product of numbers rolled on 6 dice is divisible by 9
def prob_product_div_by_9 : ℚ := 449/729

-- Main statement of the problem
theorem product_of_dice_divisible_by_9 :
  (1 - ((prob_roll_not_div_by_3^6) + 
        (6 * prob_roll_div_by_3 * (prob_roll_not_div_by_3^5)) + 
        (15 * (prob_roll_div_by_3^2) * (prob_roll_not_div_by_3^4)))) = prob_product_div_by_9 :=
by {
  sorry
}

end product_of_dice_divisible_by_9_l241_241235


namespace multiples_of_15_between_17_and_202_l241_241513

theorem multiples_of_15_between_17_and_202 : 
  ∃ n : ℕ, (∀ k : ℤ, 17 < k * 15 ∧ k * 15 < 202 → k = n + 1) ∧ n = 12 :=
sorry

end multiples_of_15_between_17_and_202_l241_241513


namespace projection_of_AB_onto_CD_l241_241636

noncomputable def A : ℝ × ℝ := (-1, 1)
noncomputable def B : ℝ × ℝ := (1, 2)
noncomputable def C : ℝ × ℝ := (-2, -1)
noncomputable def D : ℝ × ℝ := (3, 4)

noncomputable def vector_sub (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  (p2.1 - p1.1, p2.2 - p1.2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem projection_of_AB_onto_CD :
  let AB := vector_sub A B
  let CD := vector_sub C D
  (magnitude AB) * (dot_product AB CD) / (magnitude CD) ^ 2 = 3 * Real.sqrt 2 / 2 :=
by
  sorry

end projection_of_AB_onto_CD_l241_241636


namespace simplify_f_value_f_given_conditions_value_f_given_alpha_l241_241924

noncomputable def f (alpha : ℝ) : ℝ := (Real.sin (π - alpha) * Real.cos (2 * π - alpha) * Real.tan (-alpha + (3 * π / 2)) * Real.tan (-alpha - π)) / Real.sin (-π - alpha)

theorem simplify_f (alpha : ℝ) : 
  f(alpha) = Real.sin(alpha) * Real.tan(alpha) :=
sorry

theorem value_f_given_conditions (alpha : ℝ) (h1 : Real.cos (alpha - (3 * π / 2)) = 1 / 5) (third_quadrant : π < alpha ∧ alpha < 3 * π / 2) : 
  f(alpha) = -Real.sqrt(6) / 60 :=
sorry

theorem value_f_given_alpha : 
  f(-1860 * (π/180)) = 3 / 2 :=
sorry

end simplify_f_value_f_given_conditions_value_f_given_alpha_l241_241924


namespace num_tents_needed_l241_241541

def count_people : ℕ :=
  let matts_family := 1 + 1 + 1 + 1 + 1 + 4 + 1 + 1 + 2 + 2
  let joes_family := 1 + 1 + 3 + 1
  matts_family + joes_family

def house_capacity : ℕ := 6

def tent_capacity : ℕ := 2

theorem num_tents_needed : (count_people - house_capacity) / tent_capacity = 7 := by
  sorry

end num_tents_needed_l241_241541


namespace resultingPoint_is_correct_l241_241196

def Point (α : Type _) := (x : α) × (y : α)

def initialPoint : Point Int := (-2, -3)

def moveLeft (p : Point Int) (units : Int) : Point Int :=
  (p.1 - units, p.2)

def moveUp (p : Point Int) (units : Int) : Point Int :=
  (p.1, p.2 + units)

theorem resultingPoint_is_correct : 
  (moveUp (moveLeft initialPoint 1) 3) = (-3, 0) :=
by
  sorry

end resultingPoint_is_correct_l241_241196


namespace badminton_costs_l241_241831

variables (x : ℕ) (h : x > 16)

-- Define costs at Store A and Store B
def cost_A : ℕ := 1760 + 40 * x
def cost_B : ℕ := 1920 + 32 * x

-- Lean statement to prove the costs
theorem badminton_costs : 
  cost_A x = 1760 + 40 * x ∧ cost_B x = 1920 + 32 * x :=
by {
  -- This proof is expected but not required for the task
  sorry
}

end badminton_costs_l241_241831


namespace probability_from_first_to_last_l241_241878

noncomputable def probability_path_possible (n : ℕ) : ℝ :=
  (2 ^ (n - 1) : ℝ) / (Nat.choose (2 * (n - 1)) (n - 1) : ℝ)

theorem probability_from_first_to_last (n : ℕ) (h : n > 1) :
  probability_path_possible n = 2 ^ (n - 1) / (Nat.choose (2 * (n - 1)) (n - 1)) := sorry

end probability_from_first_to_last_l241_241878


namespace geometric_sequence_common_ratio_l241_241950

theorem geometric_sequence_common_ratio (a_1 a_4 q : ℕ) (h1 : a_1 = 8) (h2 : a_4 = 64) (h3 : a_4 = a_1 * q^3) : q = 2 :=
by {
  -- Given: a_1 = 8
  --        a_4 = 64
  --        a_4 = a_1 * q^3
  -- Prove: q = 2
  sorry
}

end geometric_sequence_common_ratio_l241_241950


namespace trailing_zeros_of_7_factorial_in_base_8_l241_241566

def trailing_zeros_in_base (n : ℕ) (b : ℕ) : ℕ :=
  if b ≤ 1 then 0
  else (List.range n.succ).count (λ k, b^k | n)

theorem trailing_zeros_of_7_factorial_in_base_8 :
  trailing_zeros_in_base 7! 8 = 1 :=
by
  sorry

end trailing_zeros_of_7_factorial_in_base_8_l241_241566


namespace probability_open_path_l241_241870

-- Define necessary terms
def total_doors (n : ℕ) : ℕ := 2 * (n - 1)
def locked_doors (n : ℕ) : ℕ := total_doors n / 2

-- Helper function to compute binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Probability theorem
theorem probability_open_path (n : ℕ) (h : n > 1) : 
  ((locked_doors n) = (n-1)) → 
  (∃ p, p = (2^(n-1)) / (binom (total_doors n) (n-1))) :=
by {
  intro h1,
  use ((2^(n-1)) / (binom (total_doors n) (n-1))),
  sorry
}

end probability_open_path_l241_241870


namespace max_value_of_f_l241_241122

noncomputable def f (x : ℝ) : ℝ := 3^x - 9^x

theorem max_value_of_f : ∃ x : ℝ, f x = 1 / 4 := sorry

end max_value_of_f_l241_241122


namespace janice_remaining_time_l241_241954

theorem janice_remaining_time
  (homework_time : ℕ := 30)
  (clean_room_time : ℕ := homework_time / 2)
  (walk_dog_time : ℕ := homework_time + 5)
  (take_out_trash_time : ℕ := homework_time / 6)
  (total_time_before_movie : ℕ := 120) :
  (total_time_before_movie - (homework_time + clean_room_time + walk_dog_time + take_out_trash_time)) = 35 :=
by
  sorry

end janice_remaining_time_l241_241954


namespace round_robin_games_l241_241739

theorem round_robin_games (x : ℕ) (h : 45 = (1 / 2) * x * (x - 1)) : (1 / 2) * x * (x - 1) = 45 :=
sorry

end round_robin_games_l241_241739


namespace point_transform_l241_241195

theorem point_transform : 
  let P := (-2, -3)
  let P' := (P.1 - 1, P.2 + 3)
  P' = (-3, 0) :=
by
  let P := (-2, -3)
  let P' := (P.1 - 1, P.2 + 3)
  show P' = (-3, 0)
  sorry

end point_transform_l241_241195


namespace expression_max_value_l241_241771

open Real

theorem expression_max_value (x : ℝ) : ∃ M, M = 1/7 ∧ (∀ y : ℝ, y = x -> (y^3) / (y^6 + y^4 + y^3 - 3*y^2 + 9) ≤ M) :=
sorry

end expression_max_value_l241_241771


namespace bus_driver_earnings_l241_241081

variables (rate : ℝ) (regular_hours overtime_hours : ℕ) (regular_rate overtime_rate : ℝ)

def calculate_regular_earnings (regular_rate : ℝ) (regular_hours : ℕ) : ℝ :=
  regular_rate * regular_hours

def calculate_overtime_earnings (overtime_rate : ℝ) (overtime_hours : ℕ) : ℝ :=
  overtime_rate * overtime_hours

def total_compensation (regular_rate overtime_rate : ℝ) (regular_hours overtime_hours : ℕ) : ℝ :=
  calculate_regular_earnings regular_rate regular_hours + calculate_overtime_earnings overtime_rate overtime_hours

theorem bus_driver_earnings :
  let regular_rate := 16
  let overtime_rate := regular_rate * 1.75
  let regular_hours := 40
  let total_hours := 44
  let overtime_hours := total_hours - regular_hours
  total_compensation regular_rate overtime_rate regular_hours overtime_hours = 752 :=
by
  sorry

end bus_driver_earnings_l241_241081


namespace find_c_of_perpendicular_lines_l241_241479

theorem find_c_of_perpendicular_lines (c : ℤ) :
  (∀ x y : ℤ, y = -3 * x + 4 → ∃ y' : ℤ, y' = (c * x + 18) / 9) →
  c = 3 :=
by
  sorry

end find_c_of_perpendicular_lines_l241_241479


namespace probability_factor_of_36_is_1_over_4_l241_241272

def totalPositiveIntegers := 36

def isDivisor (n: Nat) : Prop := ∃ (a b: Nat), (0 ≤ a ∧ a ≤ 2) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (n = 2^a * 3^b)

def numDivisors := 9
def probability := (numDivisors : ℚ) / (totalPositiveIntegers : ℚ)

theorem probability_factor_of_36_is_1_over_4 :
  probability = (1 / 4 : ℚ) :=
sorry

end probability_factor_of_36_is_1_over_4_l241_241272


namespace cyclist_speed_l241_241576

variable (circumference : ℝ) (v₂ : ℝ) (t : ℝ)

theorem cyclist_speed (h₀ : circumference = 180) (h₁ : v₂ = 8) (h₂ : t = 12)
  (h₃ : (7 * t + v₂ * t) = circumference) : 7 = 7 :=
by
  -- From given conditions, we derived that v₁ should be 7
  sorry

end cyclist_speed_l241_241576


namespace total_households_in_apartment_complex_l241_241992

theorem total_households_in_apartment_complex :
  let buildings := 25
  let floors_per_building := 10
  let households_per_floor := 8
  buildings * floors_per_building * households_per_floor = 2000 :=
by
  sorry

end total_households_in_apartment_complex_l241_241992


namespace fractional_expression_value_l241_241782

theorem fractional_expression_value (x y z : ℝ) (hz : z ≠ 0) 
  (h1 : 2 * x - 3 * y - z = 0)
  (h2 : x + 3 * y - 14 * z = 0) :
  (x^2 + 3 * x * y) / (y^2 + z^2) = 7 := 
by sorry

end fractional_expression_value_l241_241782


namespace probability_open_doors_l241_241882

variable (n : ℕ)

theorem probability_open_doors (h : n > 1) : 
  let num_doors := 2 * (n - 1)
      num_locked := n - 1
  in (2^(n-1) / (num_doors.choose num_locked) = 
  2^(n-1) / (nat.choose num_doors num_locked)) := sorry

end probability_open_doors_l241_241882


namespace cost_of_eight_CDs_l241_241711

theorem cost_of_eight_CDs (cost_of_two_CDs : ℕ) (h : cost_of_two_CDs = 36) : 8 * (cost_of_two_CDs / 2) = 144 := by
  sorry

end cost_of_eight_CDs_l241_241711


namespace handshake_count_l241_241227

-- Define the conditions
def num_companies : ℕ := 5
def reps_per_company : ℕ := 4
def total_people : ℕ := num_companies * reps_per_company
def handshakes_per_person : ℕ := total_people - 1 - (reps_per_company - 1)

-- Define the theorem to prove
theorem handshake_count : 
  total_people * (total_people - 1 - (reps_per_company - 1)) / 2 = 160 := 
by
  -- Here should be the proof, but it is omitted
  sorry

end handshake_count_l241_241227


namespace total_pages_in_book_l241_241626

theorem total_pages_in_book (pages_per_day : ℕ) (days : ℕ) (total_pages : ℕ) 
  (h1 : pages_per_day = 22) (h2 : days = 569) : total_pages = 12518 :=
by
  sorry

end total_pages_in_book_l241_241626


namespace factorization_identity_l241_241331

theorem factorization_identity (a : ℝ) : (a + 3) * (a - 7) + 25 = (a - 2) ^ 2 :=
by
  sorry

end factorization_identity_l241_241331


namespace hair_cut_length_l241_241371

theorem hair_cut_length (original_length after_haircut : ℕ) (h1 : original_length = 18) (h2 : after_haircut = 9) :
  original_length - after_haircut = 9 :=
by
  sorry

end hair_cut_length_l241_241371


namespace unique_not_in_range_of_g_l241_241048

noncomputable def g (m n p q : ℝ) (x : ℝ) : ℝ := (m * x + n) / (p * x + q)

theorem unique_not_in_range_of_g (m n p q : ℝ) (hne1 : m ≠ 0) (hne2 : n ≠ 0) (hne3 : p ≠ 0) (hne4 : q ≠ 0)
  (h₁ : g m n p q 23 = 23) (h₂ : g m n p q 53 = 53) (h₃ : ∀ (x : ℝ), x ≠ -q / p → g m n p q (g m n p q x) = x) :
  ∃! x : ℝ, ¬ ∃ y : ℝ, g m n p q y = x ∧ x = -38 :=
sorry

end unique_not_in_range_of_g_l241_241048


namespace geometric_sequence_xz_eq_three_l241_241630

theorem geometric_sequence_xz_eq_three 
  (x y z : ℝ)
  (h1 : ∃ r : ℝ, x = -1 * r ∧ y = x * r ∧ z = y * r ∧ -3 = z * r) :
  x * z = 3 :=
by
  -- skip the proof
  sorry

end geometric_sequence_xz_eq_three_l241_241630


namespace third_term_arithmetic_sequence_l241_241163

theorem third_term_arithmetic_sequence (a x : ℝ) 
  (h : 2 * a + 2 * x = 10) : a + x = 5 := 
by
  sorry

end third_term_arithmetic_sequence_l241_241163


namespace derrick_yard_length_l241_241474

def alex_yard (derrick_yard : ℝ) := derrick_yard / 2
def brianne_yard (alex_yard : ℝ) := 6 * alex_yard

theorem derrick_yard_length : brianne_yard (alex_yard derrick_yard) = 30 → derrick_yard = 10 :=
by
  intro h
  sorry

end derrick_yard_length_l241_241474


namespace passenger_rides_each_car_once_l241_241069

noncomputable theory

open ProbabilityTheory

variable (Ω : Type) [ProbabilitySpace Ω]

/-- The probability that a passenger will ride in each of the 2 cars exactly once given two rides. -/
theorem passenger_rides_each_car_once (Rides : (Fin 2) → Ω) (eventA eventB : Event Ω) 
  (hA : eventA = {ω | (Rides 0) ω = 0 ∨ (Rides 1) ω = 1}) 
  (hB : eventB = {ω | (Rides 0) ω = 1 ∨ (Rides 1) ω = 0}) 
  (hIndependent : indep (λ _, Rides 0) (λ _, Rides 1)) :
  (condProb eventA {ω | true}) = 1/2 :=
by sorry

end passenger_rides_each_car_once_l241_241069


namespace path_count_l241_241865

theorem path_count :
  let is_valid_path (path : List (ℕ × ℕ)) : Prop :=
    ∃ (n : ℕ), path = List.range n    -- This is a simplification for definition purposes
  let count_paths_outside_square (start finish : (ℤ × ℤ)) (steps : ℕ) : ℕ :=
    43826                              -- Hardcoded the result as this is the correct answer
  ∀ start finish : (ℤ × ℤ),
    start = (-5, -5) → 
    finish = (5, 5) → 
    count_paths_outside_square start finish 20 = 43826
:= 
sorry

end path_count_l241_241865


namespace shape_is_plane_l241_241344

-- Define cylindrical coordinates
structure CylindricalCoord :=
  (r : ℝ) (theta : ℝ) (z : ℝ)

-- Define the condition
def condition (c : ℝ) (coord : CylindricalCoord) : Prop :=
  coord.z = c

-- The shape is described as a plane
def is_plane : Prop := ∀ (coord1 coord2 : CylindricalCoord), (coord1.z = coord2.z)

theorem shape_is_plane (c : ℝ) : 
  (∀ coord : CylindricalCoord, condition c coord) ↔ is_plane :=
by 
  sorry

end shape_is_plane_l241_241344


namespace three_point_two_four_two_times_twelve_div_one_hundred_equals_zero_point_three_eight_nine_zero_four_l241_241860

theorem three_point_two_four_two_times_twelve_div_one_hundred_equals_zero_point_three_eight_nine_zero_four :
  (3.242 * 12) / 100 = 0.38904 :=
by 
  sorry

end three_point_two_four_two_times_twelve_div_one_hundred_equals_zero_point_three_eight_nine_zero_four_l241_241860


namespace basketball_game_points_half_l241_241017

theorem basketball_game_points_half (a d b r : ℕ) (h_arith_seq : a + (a + d) + (a + 2 * d) + (a + 3 * d) ≤ 100)
    (h_geo_seq : b + b * r + b * r^2 + b * r^3 ≤ 100)
    (h_win_by_two : 4 * a + 6 * d = b * (1 + r + r^2 + r^3) + 2) :
    (a + (a + d)) + (b + b * r) = 14 :=
sorry

end basketball_game_points_half_l241_241017


namespace CauchySchwarz_l241_241392

theorem CauchySchwarz' (a b x y : ℝ) : (a^2 + b^2) * (x^2 + y^2) ≥ (a * x + b * y)^2 := by
  sorry

end CauchySchwarz_l241_241392


namespace max_value_of_A_l241_241926

theorem max_value_of_A (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  (a^3 * (b + c) + b^3 * (c + a) + c^3 * (a + b)) / ((a + b + c)^4 - 79 * (a * b * c)^(4 / 3)) ≤ 3 :=
sorry

end max_value_of_A_l241_241926


namespace min_value_of_quadratic_l241_241809

theorem min_value_of_quadratic (y1 y2 y3 : ℝ) (h1 : 0 < y1) (h2 : 0 < y2) (h3 : 0 < y3) (h_eq : 2 * y1 + 3 * y2 + 4 * y3 = 75) :
  y1^2 + 2 * y2^2 + 3 * y3^2 ≥ 5625 / 29 :=
sorry

end min_value_of_quadratic_l241_241809


namespace value_of_a_l241_241409

theorem value_of_a 
  (a b c d e : ℤ)
  (h1 : a + 4 = b + 2)
  (h2 : a + 2 = b)
  (h3 : a + c = 146)
  (he : e = 79)
  (h4 : e = d + 2)
  (h5 : d = c + 2)
  (h6 : c = b + 2) :
  a = 71 :=
by
  sorry

end value_of_a_l241_241409


namespace find_width_of_river_l241_241746

theorem find_width_of_river
    (total_distance : ℕ)
    (river_width : ℕ)
    (prob_find_item : ℚ)
    (h1 : total_distance = 500)
    (h2 : prob_find_item = 4/5)
    : river_width = 100 :=
by
    sorry

end find_width_of_river_l241_241746


namespace simplify_and_evaluate_expression_l241_241685

variable (x y : ℚ)

theorem simplify_and_evaluate_expression (hx : x = 1) (hy : y = 1 / 2) :
  (3 * x + 2 * y) * (3 * x - 2 * y) - (x - y) ^ 2 = 31 / 4 :=
by
  rw [hx, hy]
  sorry

end simplify_and_evaluate_expression_l241_241685


namespace two_person_subcommittees_from_eight_l241_241507

theorem two_person_subcommittees_from_eight :
  (Nat.choose 8 2) = 28 :=
by
  sorry

end two_person_subcommittees_from_eight_l241_241507


namespace probability_divisor_of_36_is_one_fourth_l241_241283

noncomputable def probability_divisor_of_36 : ℚ :=
  let total_divisors := 9 in
  let total_integers := 36 in
  total_divisors / total_integers

theorem probability_divisor_of_36_is_one_fourth :
  probability_divisor_of_36 = 1 / 4 :=
by
  sorry

end probability_divisor_of_36_is_one_fourth_l241_241283


namespace angle_sum_around_point_l241_241662

theorem angle_sum_around_point (x y : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) : 
    x + y + 130 = 360 → x + y = 230 := by
  sorry

end angle_sum_around_point_l241_241662


namespace sum_of_two_numbers_l241_241056

theorem sum_of_two_numbers (S L : ℝ) (h1 : S = 10.0) (h2 : 7 * S = 5 * L) : S + L = 24.0 :=
by
  -- proof goes here
  sorry

end sum_of_two_numbers_l241_241056


namespace factor_probability_l241_241264

theorem factor_probability : 
  let S := { n : ℕ | n > 0 ∧ n ≤ 36 } in
  let factor_count := (multiplicity 2 36 + 1) * (multiplicity 3 36 + 1) in
  let total_count := 36 in
  (factor_count : ℚ) / total_count = (1 : ℚ) / 4 := 
by
  sorry

end factor_probability_l241_241264


namespace binomial_12_10_l241_241452

def binomial (n k : ℕ) : ℕ := n.choose k

theorem binomial_12_10 : binomial 12 10 = 66 := by
  -- The proof will go here
  sorry

end binomial_12_10_l241_241452


namespace part1_part2_l241_241766

noncomputable section

def f (x : ℝ) (a : ℝ) : ℝ := abs (x + 2 * a)

theorem part1 (a : ℝ) :
  (∀ x : ℝ, -4 < x ∧ x < 4 ↔ f x a < 4 - 2 * a) →
  a = 0 := 
sorry

theorem part2 (m : ℝ) :
  (∀ x : ℝ, f x 1 - f (-2 * x) 1 ≤ x + m) →
  2 ≤ m :=
sorry

end part1_part2_l241_241766


namespace probability_divisor_of_36_is_one_fourth_l241_241285

noncomputable def probability_divisor_of_36 : ℚ :=
  let total_divisors := 9 in
  let total_integers := 36 in
  total_divisors / total_integers

theorem probability_divisor_of_36_is_one_fourth :
  probability_divisor_of_36 = 1 / 4 :=
by
  sorry

end probability_divisor_of_36_is_one_fourth_l241_241285


namespace find_n_l241_241211

noncomputable def arithmeticSequenceTerm (a b : ℝ) (n : ℕ) : ℝ :=
  let A := Real.log a
  let B := Real.log b
  6 * B + (n - 1) * 11 * B

theorem find_n 
  (a b : ℝ) 
  (h1 : Real.log (a^2 * b^4) = 2 * Real.log a + 4 * Real.log b)
  (h2 : Real.log (a^6 * b^11) = 6 * Real.log a + 11 * Real.log b)
  (h3 : Real.log (a^12 * b^20) = 12 * Real.log a + 20 * Real.log b) 
  (h_diff : (6 * Real.log a + 11 * Real.log b) - (2 * Real.log a + 4 * Real.log b) = 
            (12 * Real.log a + 20 * Real.log b) - (6 * Real.log a + 11 * Real.log b))
  : ∃ n : ℕ, arithmeticSequenceTerm a b 15 = Real.log (b^n) ∧ n = 160 :=
by
  use 160
  sorry

end find_n_l241_241211


namespace find_a_l241_241519

theorem find_a (a : ℚ) (h : a + a / 3 + a / 4 = 4) : a = 48 / 19 := by
  sorry

end find_a_l241_241519


namespace find_interest_rate_l241_241982

-- conditions
def P : ℝ := 6200
def t : ℕ := 10

def interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * r * t
def I : ℝ := P - 3100

-- problem statement
theorem find_interest_rate (r : ℝ) :
  interest P r t = I → r = 0.05 :=
by
  sorry

end find_interest_rate_l241_241982


namespace probability_factor_of_36_l241_241255

def is_factor_of (n d : ℕ) : Prop := d % n = 0

theorem probability_factor_of_36 : 
  let total := 36
  let factors_of_36 := { n | n ≤ 36 ∧ is_factor_of n 36 }
  (factors_of_36.to_finset.card : ℚ) / total = 1 / 4 := 
by
  sorry

end probability_factor_of_36_l241_241255


namespace probability_path_from_first_to_last_floor_open_doors_l241_241872

noncomputable
def probability_path_possible (n : ℕ) : ℚ :=
  (2 ^ (n - 1)) / (nat.choose (2 * (n - 1)) (n - 1))

theorem probability_path_from_first_to_last_floor_open_doors (n : ℕ) :
  probability_path_possible n = (2 ^ (n - 1)) / (nat.choose (2 * (n - 1)) (n - 1)) :=
by
  sorry

end probability_path_from_first_to_last_floor_open_doors_l241_241872


namespace coordinate_sum_condition_l241_241354

open Function

theorem coordinate_sum_condition :
  (∃ (g : ℝ → ℝ), g 6 = 5 ∧
    (∃ y : ℝ, 4 * y = g (3 * 2) + 4 ∧ y = 9 / 4 ∧ 2 + y = 17 / 4)) :=
by
  sorry

end coordinate_sum_condition_l241_241354


namespace vector_magnitude_l241_241152

noncomputable def a : ℝ × ℝ := (Real.cos (5 * Real.pi / 180), Real.sin (5 * Real.pi / 180))
noncomputable def b : ℝ × ℝ := (Real.cos (65 * Real.pi / 180), Real.sin (65 * Real.pi / 180))

theorem vector_magnitude : |(a.1 + 2 * b.1, a.2 + 2 * b.2)| = Real.sqrt 7 :=
by sorry

end vector_magnitude_l241_241152


namespace number_of_girls_l241_241704

theorem number_of_girls
  (B G : ℕ)
  (ratio_condition : B * 8 = 5 * G)
  (total_condition : B + G = 260) :
  G = 160 :=
by
  sorry

end number_of_girls_l241_241704


namespace ms_cole_students_l241_241194

theorem ms_cole_students (S6 S4 S7 : ℕ)
  (h1: S6 = 40)
  (h2: S4 = 4 * S6)
  (h3: S7 = 2 * S4) :
  S6 + S4 + S7 = 520 :=
by
  sorry

end ms_cole_students_l241_241194


namespace probability_factor_of_36_l241_241244

def is_factor (d n : ℕ) : Prop := n % d = 0

theorem probability_factor_of_36 : 
  (∑ i in finset.range (36+1), if is_factor i 36 then (1:ℚ) else 0) / 36 = 1 / 4 := by
sorry

end probability_factor_of_36_l241_241244


namespace find_n_l241_241788

theorem find_n (P s k m n : ℝ) (h : P = s / (1 + k + m) ^ n) :
  n = (Real.log (s / P)) / (Real.log (1 + k + m)) :=
sorry

end find_n_l241_241788


namespace number_of_real_solutions_l241_241909

noncomputable def f (x : ℝ) : ℝ :=
  (Finset.range 50).sum (λ n => (n + 1 : ℝ) / (x - (n + 1 : ℝ)))

theorem number_of_real_solutions : ∃ n : ℕ, n = 51 ∧ ∀ x : ℝ, f x = x + 1 ↔ n = 51 :=
by
  sorry

end number_of_real_solutions_l241_241909


namespace perimeter_of_photo_l241_241600

theorem perimeter_of_photo 
  (frame_width : ℕ)
  (frame_area : ℕ)
  (outer_edge_length : ℕ)
  (photo_perimeter : ℕ) :
  frame_width = 2 → 
  frame_area = 48 → 
  outer_edge_length = 10 →
  photo_perimeter = 16 :=
by
  intros h1 h2 h3
  -- proof goes here
  sorry

end perimeter_of_photo_l241_241600


namespace tangent_circle_radius_l241_241363

theorem tangent_circle_radius (O A B C : ℝ) (r1 r2 : ℝ) :
  (O = 5) →
  (abs (A - B) = 8) →
  (C = (2 * A + B) / 3) →
  r1 = 8 / 9 ∨ r2 = 32 / 9 :=
sorry

end tangent_circle_radius_l241_241363


namespace five_digit_numbers_l241_241697

def divisible_by_4_and_9 (n : ℕ) : Prop :=
  (n % 4 = 0) ∧ (n % 9 = 0)

def is_candidate (n : ℕ) : Prop :=
  ∃ a b, n = 10000 * a + 1000 + 200 + 30 + b ∧ a < 10 ∧ b < 10

theorem five_digit_numbers :
  ∀ (n : ℕ), is_candidate n → divisible_by_4_and_9 n → n = 11232 ∨ n = 61236 :=
by
  sorry

end five_digit_numbers_l241_241697


namespace probability_factor_36_l241_241240

theorem probability_factor_36 : 
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 36 ∧ (∀ k : ℕ, k ∣ n ↔ k ∣ 36) → (n ≠ 0 → (∃ p : ℚ, p = 1 / 4))) :=
begin
  sorry
end

end probability_factor_36_l241_241240


namespace factor_probability_36_l241_241288

-- Definitions based on conditions
def is_factor (n d : ℕ) : Prop := d ∣ n
def num_factors (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (is_factor n).card

def probability_factor (n : ℕ) : ℚ := num_factors n / n

-- Problem statement
theorem factor_probability_36 : probability_factor 36 = 1 / 4 :=
  by sorry

end factor_probability_36_l241_241288


namespace distance_between_trees_l241_241657

-- Definitions based on conditions
def yard_length : ℝ := 360
def number_of_trees : ℕ := 31
def number_of_gaps : ℕ := number_of_trees - 1

-- The proposition to prove
theorem distance_between_trees : yard_length / number_of_gaps = 12 := sorry

end distance_between_trees_l241_241657


namespace determinant_scalar_multiplication_l241_241921

theorem determinant_scalar_multiplication (x y z w : ℝ) (h : abs (x * w - y * z) = 10) :
  abs (3*x * 3*w - 3*y * 3*z) = 90 :=
by
  sorry

end determinant_scalar_multiplication_l241_241921


namespace probability_factor_36_l241_241258

theorem probability_factor_36 : (∃ n : ℕ, n ≤ 36 ∧ ∃ k, 36 = k * n) → (9 / 36 = 1 / 4) :=
begin
  sorry
end

end probability_factor_36_l241_241258


namespace two_m_plus_three_b_l241_241108

noncomputable def m : ℚ := (-(3/2) - (1/2)) / (2 - (-1))

noncomputable def b : ℚ := (1/2) - m * (-1)

theorem two_m_plus_three_b :
  2 * m + 3 * b = -11 / 6 :=
by
  sorry

end two_m_plus_three_b_l241_241108


namespace stratified_sampling_l241_241129

variable (H M L total_sample : ℕ)
variable (H_fams M_fams L_fams : ℕ)

-- Conditions
def community : Prop := H_fams = 150 ∧ M_fams = 360 ∧ L_fams = 90
def total_population : Prop := H_fams + M_fams + L_fams = 600
def sample_size : Prop := total_sample = 100

-- Statement
theorem stratified_sampling (H_fams M_fams L_fams : ℕ) (total_sample : ℕ)
  (h_com : community H_fams M_fams L_fams)
  (h_total_pop : total_population H_fams M_fams L_fams)
  (h_sample_size : sample_size total_sample)
  : H = 25 ∧ M = 60 ∧ L = 15 :=
by
  sorry

end stratified_sampling_l241_241129


namespace number_of_Slurpees_l241_241959

theorem number_of_Slurpees
  (total_money : ℕ)
  (cost_per_Slurpee : ℕ)
  (change : ℕ)
  (spent_money := total_money - change)
  (number_of_Slurpees := spent_money / cost_per_Slurpee)
  (h1 : total_money = 20)
  (h2 : cost_per_Slurpee = 2)
  (h3 : change = 8) :
  number_of_Slurpees = 6 := by
  sorry

end number_of_Slurpees_l241_241959


namespace min_value_is_five_l241_241350

noncomputable def min_value (x y : ℝ) : ℝ :=
  if x + 3 * y = 5 * x * y then 3 * x + 4 * y else 0

theorem min_value_is_five {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : min_value x y = 5 :=
by
  sorry

end min_value_is_five_l241_241350


namespace cistern_fill_time_l241_241439

/--
  A cistern can be filled by tap A in 4 hours,
  emptied by tap B in 6 hours,
  and filled by tap C in 3 hours.
  If all the taps are opened simultaneously,
  then the cistern will be filled in exactly 2.4 hours.
-/
theorem cistern_fill_time :
  let rate_A := 1 / 4
  let rate_B := -1 / 6
  let rate_C := 1 / 3
  let combined_rate := rate_A + rate_B + rate_C
  let fill_time := 1 / combined_rate
  fill_time = 2.4 := by
  sorry

end cistern_fill_time_l241_241439


namespace find_k_for_minimum_value_l241_241335

theorem find_k_for_minimum_value :
  ∃ (k : ℝ), (∀ (x y : ℝ), 9 * x^2 - 6 * k * x * y + (3 * k^2 + 1) * y^2 - 6 * x - 6 * y + 7 ≥ 1)
  ∧ (∃ (x y : ℝ), 9 * x^2 - 6 * k * x * y + (3 * k^2 + 1) * y^2 - 6 * x - 6 * y + 7 = 1)
  ∧ k = 3 :=
sorry

end find_k_for_minimum_value_l241_241335


namespace cos_neg_pi_over_3_l241_241622

theorem cos_neg_pi_over_3 : Real.cos (-π / 3) = 1 / 2 :=
by
  sorry

end cos_neg_pi_over_3_l241_241622


namespace janice_remaining_time_l241_241953

theorem janice_remaining_time
  (homework_time : ℕ := 30)
  (clean_room_time : ℕ := homework_time / 2)
  (walk_dog_time : ℕ := homework_time + 5)
  (take_out_trash_time : ℕ := homework_time / 6)
  (total_time_before_movie : ℕ := 120) :
  (total_time_before_movie - (homework_time + clean_room_time + walk_dog_time + take_out_trash_time)) = 35 :=
by
  sorry

end janice_remaining_time_l241_241953


namespace not_possible_to_construct_l241_241328

/-- The frame consists of 54 unit segments. -/
def frame_consists_of_54_units : Prop := sorry

/-- Each part of the construction set consists of three unit segments. -/
def part_is_three_units : Prop := sorry

/-- Each vertex of a cube is shared by three edges. -/
def vertex_shares_three_edges : Prop := sorry

/-- Six segments emerge from the center of the cube. -/
def center_has_six_segments : Prop := sorry

/-- It is not possible to construct the frame with exactly 18 parts. -/
theorem not_possible_to_construct
  (h1 : frame_consists_of_54_units)
  (h2 : part_is_three_units)
  (h3 : vertex_shares_three_edges)
  (h4 : center_has_six_segments) : 
  ¬ ∃ (parts : ℕ), parts = 18 :=
sorry

end not_possible_to_construct_l241_241328
