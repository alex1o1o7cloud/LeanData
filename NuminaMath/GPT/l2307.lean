import Mathlib

namespace NUMINAMATH_GPT_vector_parallel_and_on_line_l2307_230722

noncomputable def is_point_on_line (x y t : ℝ) : Prop :=
  x = 5 * t + 3 ∧ y = 2 * t + 4

noncomputable def is_parallel (a b c d : ℝ) : Prop :=
  ∃ k : ℝ, a = k * c ∧ b = k * d

theorem vector_parallel_and_on_line :
  ∃ (a b t : ℝ), 
      (a = (5 * t + 3) - 1) ∧ (b = (2 * t + 4) - 1) ∧ 
      is_parallel a b 3 2 ∧ is_point_on_line (5 * t + 3) (2 * t + 4) t := 
by
  use (33 / 4), (11 / 2), (5 / 4)
  sorry

end NUMINAMATH_GPT_vector_parallel_and_on_line_l2307_230722


namespace NUMINAMATH_GPT_digit_Q_is_0_l2307_230718

theorem digit_Q_is_0 (M N P Q : ℕ) (hM : M < 10) (hN : N < 10) (hP : P < 10) (hQ : Q < 10) 
  (add_eq : 10 * M + N + 10 * P + M = 10 * Q + N) 
  (sub_eq : 10 * M + N - (10 * P + M) = N) : Q = 0 := 
by
  sorry

end NUMINAMATH_GPT_digit_Q_is_0_l2307_230718


namespace NUMINAMATH_GPT_slope_of_given_line_eq_l2307_230728

theorem slope_of_given_line_eq : (∀ x y : ℝ, (4 / x + 5 / y = 0) → (x ≠ 0 ∧ y ≠ 0) → ∀ y x : ℝ, y = - (5 * x / 4) → ∃ m, m = -5/4) :=
by
  sorry

end NUMINAMATH_GPT_slope_of_given_line_eq_l2307_230728


namespace NUMINAMATH_GPT_find_tuition_l2307_230721

def tuition_problem (T : ℝ) : Prop :=
  75 = T + (T - 15)

theorem find_tuition (T : ℝ) (h : tuition_problem T) : T = 45 :=
by
  sorry

end NUMINAMATH_GPT_find_tuition_l2307_230721


namespace NUMINAMATH_GPT_total_shaded_area_l2307_230738

/-- 
Given a 6-foot by 12-foot floor tiled with 1-foot by 1-foot tiles,
where each tile has four white quarter circles of radius 1/3 foot at its corners,
prove that the total shaded area of the floor is 72 - 8π square feet.
-/
theorem total_shaded_area :
  let floor_length := 6
  let floor_width := 12
  let tile_size := 1
  let radius := 1 / 3
  let area_of_tile := tile_size * tile_size
  let white_area_per_tile := (Real.pi * radius^2 / 4) * 4
  let shaded_area_per_tile := area_of_tile - white_area_per_tile
  let number_of_tiles := floor_length * floor_width
  let total_shaded_area := number_of_tiles * shaded_area_per_tile
  total_shaded_area = 72 - 8 * Real.pi :=
by
  let floor_length := 6
  let floor_width := 12
  let tile_size := 1
  let radius := 1 / 3
  let area_of_tile := tile_size * tile_size
  let white_area_per_tile := (Real.pi * radius^2 / 4) * 4
  let shaded_area_per_tile := area_of_tile - white_area_per_tile
  let number_of_tiles := floor_length * floor_width
  let total_shaded_area := number_of_tiles * shaded_area_per_tile
  sorry

end NUMINAMATH_GPT_total_shaded_area_l2307_230738


namespace NUMINAMATH_GPT_sum_of_smallest_and_largest_prime_l2307_230701

def primes_between (a b : ℕ) : List ℕ := List.filter Nat.Prime (List.range' a (b - a + 1))

def smallest_prime_in_range (a b : ℕ) : ℕ :=
  match primes_between a b with
  | [] => 0
  | h::t => h

def largest_prime_in_range (a b : ℕ) : ℕ :=
  match List.reverse (primes_between a b) with
  | [] => 0
  | h::t => h

theorem sum_of_smallest_and_largest_prime : smallest_prime_in_range 1 50 + largest_prime_in_range 1 50 = 49 := 
by
  -- Let the Lean prover take over from here
  sorry

end NUMINAMATH_GPT_sum_of_smallest_and_largest_prime_l2307_230701


namespace NUMINAMATH_GPT_divisible_by_a_minus_one_squared_l2307_230727

theorem divisible_by_a_minus_one_squared (a n : ℕ) (h : n > 0) :
  (a^(n+1) - n * (a - 1) - a) % (a - 1)^2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_divisible_by_a_minus_one_squared_l2307_230727


namespace NUMINAMATH_GPT_find_max_value_l2307_230771

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 - x + a

theorem find_max_value (a x : ℝ) (h_min : f 1 a = 1) : 
  ∃ x : ℝ, f (-1/3) 2 = 59/27 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_max_value_l2307_230771


namespace NUMINAMATH_GPT_lara_additional_miles_needed_l2307_230768

theorem lara_additional_miles_needed :
  ∀ (d1 d2 d_total t1 speed1 speed2 avg_speed : ℝ),
    d1 = 20 →
    speed1 = 25 →
    speed2 = 40 →
    avg_speed = 35 →
    t1 = d1 / speed1 →
    d_total = d1 + d2 →
    avg_speed = (d_total) / (t1 + d2 / speed2) →
    d2 = 64 :=
by sorry

end NUMINAMATH_GPT_lara_additional_miles_needed_l2307_230768


namespace NUMINAMATH_GPT_ratio_siblings_l2307_230744

theorem ratio_siblings (M J C : ℕ) 
  (hM : M = 60)
  (hJ : J = 4 * M - 60)
  (hJ_C : J = C + 135) :
  (C : ℚ) / M = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_siblings_l2307_230744


namespace NUMINAMATH_GPT_percentage_distance_l2307_230753

theorem percentage_distance (start : ℝ) (end_point : ℝ) (point : ℝ) (total_distance : ℝ)
  (distance_from_start : ℝ) :
  start = -55 → end_point = 55 → point = 5.5 → total_distance = end_point - start →
  distance_from_start = point - start →
  (distance_from_start / total_distance) * 100 = 55 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_percentage_distance_l2307_230753


namespace NUMINAMATH_GPT_solve_equation_l2307_230716

theorem solve_equation : ∃ x : ℤ, (x - 15) / 3 = (3 * x + 11) / 8 ∧ x = -153 := 
by
  use -153
  sorry

end NUMINAMATH_GPT_solve_equation_l2307_230716


namespace NUMINAMATH_GPT_find_integers_l2307_230756

theorem find_integers (x y : ℤ) 
  (h1 : x * y + (x + y) = 95) 
  (h2 : x * y - (x + y) = 59) : 
  (x = 11 ∧ y = 7) ∨ (x = 7 ∧ y = 11) :=
by
  sorry

end NUMINAMATH_GPT_find_integers_l2307_230756


namespace NUMINAMATH_GPT_factorization_problem_l2307_230739

theorem factorization_problem :
  ∃ (a b : ℤ), (25 * x^2 - 130 * x - 120 = (5 * x + a) * (5 * x + b)) ∧ (a + 3 * b = -86) := by
  sorry

end NUMINAMATH_GPT_factorization_problem_l2307_230739


namespace NUMINAMATH_GPT_rectangle_circle_area_ratio_l2307_230706

theorem rectangle_circle_area_ratio (w r : ℝ) (h1 : 2 * 2 * w + 2 * w = 2 * pi * r) :
  ((2 * w) * w) / (pi * r^2) = 2 * pi / 9 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_circle_area_ratio_l2307_230706


namespace NUMINAMATH_GPT_average_cookies_l2307_230710

theorem average_cookies (cookie_counts : List ℕ) (h : cookie_counts = [8, 10, 12, 15, 16, 17, 20]) :
  (cookie_counts.sum : ℚ) / cookie_counts.length = 14 := by
    -- Proof goes here
  sorry

end NUMINAMATH_GPT_average_cookies_l2307_230710


namespace NUMINAMATH_GPT_cos_sum_identity_cosine_30_deg_l2307_230760

theorem cos_sum_identity : 
  (Real.cos (Real.pi * 43 / 180) * Real.cos (Real.pi * 13 / 180) + 
   Real.sin (Real.pi * 43 / 180) * Real.sin (Real.pi * 13 / 180)) = 
   (Real.cos (Real.pi * 30 / 180)) :=
sorry

theorem cosine_30_deg : 
  Real.cos (Real.pi * 30 / 180) = (Real.sqrt 3 / 2) :=
sorry

end NUMINAMATH_GPT_cos_sum_identity_cosine_30_deg_l2307_230760


namespace NUMINAMATH_GPT_sum_of_zeros_l2307_230797

-- Defining the conditions and the result
theorem sum_of_zeros (f : ℝ → ℝ) (h : ∀ x, f (1 - x) = f (3 + x)) (a b c : ℝ)
  (h1 : f a = 0) (h2 : f b = 0) (h3 : f c = 0) : 
  a + b + c = 3 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_zeros_l2307_230797


namespace NUMINAMATH_GPT_determine_b_l2307_230707

theorem determine_b (b : ℝ) : (∀ x1 x2 : ℝ, x1^2 - x2^2 = 7 → x1 * x2 = 12 → x1 + x2 = b) → (b = 7 ∨ b = -7) := 
by {
  -- Proof needs to be provided
  sorry
}

end NUMINAMATH_GPT_determine_b_l2307_230707


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_is_18_l2307_230792

variable (a : ℕ → ℕ)

theorem arithmetic_sequence_sum_is_18
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_a5 : a 5 = 2) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 18 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_is_18_l2307_230792


namespace NUMINAMATH_GPT_subset_condition_l2307_230736

def A : Set ℝ := {2, 0, 1, 6}
def B (a : ℝ) : Set ℝ := {x | x + a > 0}

theorem subset_condition (a : ℝ) (h : A ⊆ B a) : a > 0 :=
sorry

end NUMINAMATH_GPT_subset_condition_l2307_230736


namespace NUMINAMATH_GPT_robert_books_l2307_230780

/-- Given that Robert reads at a speed of 75 pages per hour, books have 300 pages, and Robert reads for 9 hours,
    he can read 2 complete 300-page books in that time. -/
theorem robert_books (reading_speed : ℤ) (pages_per_book : ℤ) (hours_available : ℤ) 
(h1 : reading_speed = 75) 
(h2 : pages_per_book = 300) 
(h3 : hours_available = 9) : 
  hours_available / (pages_per_book / reading_speed) = 2 := 
by {
  -- adding placeholder for proof
  sorry
}

end NUMINAMATH_GPT_robert_books_l2307_230780


namespace NUMINAMATH_GPT_line_divides_circle_1_3_l2307_230749

noncomputable def circle_equidistant_from_origin : Prop := 
  ∃ l : ℝ → ℝ, ∀ x y : ℝ, ((x-1)^2 + (y-1)^2 = 2) → 
                     (l 0 = 0 ∧ (l x = l y) ∧ 
                     ((x = 0) ∨ (y = 0)))

theorem line_divides_circle_1_3 (x y : ℝ) : 
  (x - 1)^2 + (y - 1)^2 = 2 → 
  (x = 0 ∨ y = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_divides_circle_1_3_l2307_230749


namespace NUMINAMATH_GPT_rectangle_length_l2307_230794

theorem rectangle_length (L W : ℝ) (h1 : L = 4 * W) (h2 : L * W = 100) : L = 20 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_length_l2307_230794


namespace NUMINAMATH_GPT_original_work_days_l2307_230764

-- Definitions based on conditions
noncomputable def L : ℕ := 7  -- Number of laborers originally employed
noncomputable def A : ℕ := 3  -- Number of absent laborers
noncomputable def t : ℕ := 14 -- Number of days it took the remaining laborers to finish the work

-- Theorem statement to prove
theorem original_work_days : (L - A) * t = L * 8 := by
  sorry

end NUMINAMATH_GPT_original_work_days_l2307_230764


namespace NUMINAMATH_GPT_positive_perfect_squares_multiples_of_36_lt_10_pow_8_l2307_230715

theorem positive_perfect_squares_multiples_of_36_lt_10_pow_8 :
  ∃ (count : ℕ), count = 1666 ∧ 
    (∀ n : ℕ, (n * n < 10^8 → n % 6 = 0) ↔ (n < 10^4 ∧ n % 6 = 0)) :=
sorry

end NUMINAMATH_GPT_positive_perfect_squares_multiples_of_36_lt_10_pow_8_l2307_230715


namespace NUMINAMATH_GPT_friend_spent_13_50_l2307_230711

noncomputable def amount_you_spent : ℝ := 
  let x := (22 - 5) / 2
  x

noncomputable def amount_friend_spent (x : ℝ) : ℝ := 
  x + 5

theorem friend_spent_13_50 :
  ∃ x : ℝ, (x + (x + 5) = 22) ∧ (x + 5 = 13.5) :=
by
  sorry

end NUMINAMATH_GPT_friend_spent_13_50_l2307_230711


namespace NUMINAMATH_GPT_quadratic_roots_real_find_m_value_l2307_230782

theorem quadratic_roots_real (m : ℝ) (h_roots : ∃ x1 x2 : ℝ, x1 * x1 + 4 * x1 + (m - 1) = 0 ∧ x2 * x2 + 4 * x2 + (m - 1) = 0) :
  m ≤ 5 :=
by {
  sorry
}

theorem find_m_value (m : ℝ) (x1 x2 : ℝ) (h_eq1 : x1 * x1 + 4 * x1 + (m - 1) = 0) (h_eq2 : x2 * x2 + 4 * x2 + (m - 1) = 0) (h_cond : 2 * (x1 + x2) + x1 * x2 + 10 = 0) :
  m = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_roots_real_find_m_value_l2307_230782


namespace NUMINAMATH_GPT_number_of_valid_six_tuples_l2307_230798

def is_valid_six_tuple (p : ℕ) (a b c d e f : ℕ) : Prop :=
  a + b + c + d + e + f = 3 * p ∧
  (a + b) % (c + d) = 0 ∧
  (b + c) % (d + e) = 0 ∧
  (c + d) % (e + f) = 0 ∧
  (d + e) % (f + a) = 0 ∧
  (e + f) % (a + b) = 0

theorem number_of_valid_six_tuples (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) : 
  ∃! n, n = p + 2 ∧ ∀ (a b c d e f : ℕ), is_valid_six_tuple p a b c d e f → n = p + 2 :=
sorry

end NUMINAMATH_GPT_number_of_valid_six_tuples_l2307_230798


namespace NUMINAMATH_GPT_find_m_l2307_230769

theorem find_m (m : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) (h : a = (2, -4) ∧ b = (-3, m) ∧ (‖a‖ * ‖b‖ + (a.1 * b.1 + a.2 * b.2)) = 0) : m = 6 := 
by 
  sorry

end NUMINAMATH_GPT_find_m_l2307_230769


namespace NUMINAMATH_GPT_complement_B_in_U_l2307_230758

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {x | x = 1}
def U : Set ℕ := A ∪ B

theorem complement_B_in_U : (U \ B) = {2, 3} := by
  sorry

end NUMINAMATH_GPT_complement_B_in_U_l2307_230758


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l2307_230741

theorem sum_of_squares_of_roots :
  ∃ x1 x2 : ℝ, (10 * x1 ^ 2 + 15 * x1 - 20 = 0) ∧ (10 * x2 ^ 2 + 15 * x2 - 20 = 0) ∧ (x1 ≠ x2) ∧ x1^2 + x2^2 = 25/4 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l2307_230741


namespace NUMINAMATH_GPT_largest_inscribed_circle_radius_l2307_230757

theorem largest_inscribed_circle_radius (k : ℝ) (h_perimeter : 0 < k) :
  ∃ (r : ℝ), r = (k / 2) * (3 - 2 * Real.sqrt 2) :=
by
  have h_r : ∃ (r : ℝ), r = (k / 2) * (3 - 2 * Real.sqrt 2)
  exact ⟨(k / 2) * (3 - 2 * Real.sqrt 2), rfl⟩
  exact h_r

end NUMINAMATH_GPT_largest_inscribed_circle_radius_l2307_230757


namespace NUMINAMATH_GPT_f_f_neg1_l2307_230790

def f (x : ℝ) : ℝ := x^2 + 1

theorem f_f_neg1 : f (f (-1)) = 5 :=
  by
    sorry

end NUMINAMATH_GPT_f_f_neg1_l2307_230790


namespace NUMINAMATH_GPT_spherical_to_rectangular_conversion_l2307_230700

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_conversion :
  spherical_to_rectangular 4 (Real.pi / 2) (Real.pi / 4) = (0, 2 * Real.sqrt 2, 2 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_spherical_to_rectangular_conversion_l2307_230700


namespace NUMINAMATH_GPT_volume_of_given_tetrahedron_l2307_230720

noncomputable def volume_of_tetrahedron (radius : ℝ) (total_length : ℝ) : ℝ := 
  let R := radius
  let L := total_length
  let a := (2 * Real.sqrt 33) / 3
  let V := (a^3 * Real.sqrt 2) / 12
  V

theorem volume_of_given_tetrahedron :
  volume_of_tetrahedron (Real.sqrt 22 / 2) (8 * Real.pi) = 48 := 
  sorry

end NUMINAMATH_GPT_volume_of_given_tetrahedron_l2307_230720


namespace NUMINAMATH_GPT_polyhedron_space_diagonals_l2307_230705

theorem polyhedron_space_diagonals (V E F T Q P : ℕ) (hV : V = 30) (hE : E = 70) (hF : F = 42)
                                    (hT : T = 26) (hQ : Q = 12) (hP : P = 4) : 
  ∃ D : ℕ, D = 321 :=
by
  have total_pairs := (30 * 29) / 2
  have triangular_face_diagonals := 0
  have quadrilateral_face_diagonals := 12 * 2
  have pentagon_face_diagonals := 4 * 5
  have total_face_diagonals := triangular_face_diagonals + quadrilateral_face_diagonals + pentagon_face_diagonals
  have total_edges_and_diagonals := total_pairs - 70 - total_face_diagonals
  use total_edges_and_diagonals
  sorry

end NUMINAMATH_GPT_polyhedron_space_diagonals_l2307_230705


namespace NUMINAMATH_GPT_min_distance_l2307_230762

open Complex

theorem min_distance (z : ℂ) (hz : abs (z + 2 - 2*I) = 1) : abs (z - 2 - 2*I) = 3 :=
sorry

end NUMINAMATH_GPT_min_distance_l2307_230762


namespace NUMINAMATH_GPT_john_spent_30_l2307_230772

/-- At a supermarket, John spent 1/5 of his money on fresh fruits and vegetables, 1/3 on meat products, and 1/10 on bakery products. If he spent the remaining $11 on candy, how much did John spend at the supermarket? -/
theorem john_spent_30 (X : ℝ) (h1 : X * (1/5) + X * (1/3) + X * (1/10) + 11 = X) : X = 30 := 
by 
  sorry

end NUMINAMATH_GPT_john_spent_30_l2307_230772


namespace NUMINAMATH_GPT_jessica_walks_distance_l2307_230750

theorem jessica_walks_distance (rate time : ℝ) (h_rate : rate = 4) (h_time : time = 2) :
  rate * time = 8 :=
by 
  rw [h_rate, h_time]
  norm_num

end NUMINAMATH_GPT_jessica_walks_distance_l2307_230750


namespace NUMINAMATH_GPT_blue_books_count_l2307_230767

def number_of_blue_books (R B : ℕ) (p : ℚ) : Prop :=
  R = 4 ∧ p = 3/14 → B^2 + 7 * B - 44 = 0

theorem blue_books_count :
  ∃ B : ℕ, number_of_blue_books 4 B (3/14) ∧ B = 4 :=
by
  sorry

end NUMINAMATH_GPT_blue_books_count_l2307_230767


namespace NUMINAMATH_GPT_determine_k_l2307_230746

theorem determine_k (k : ℝ) : (1 - 3 * k * (-2/3) = 7 * 3) → k = 10 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_determine_k_l2307_230746


namespace NUMINAMATH_GPT_circle_area_l2307_230730

theorem circle_area : 
    (∃ x y : ℝ, 3 * x^2 + 3 * y^2 - 9 * x + 12 * y + 27 = 0) →
    (∃ A : ℝ, A = (7 / 4) * Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_circle_area_l2307_230730


namespace NUMINAMATH_GPT_concentration_proof_l2307_230751

noncomputable def newConcentration (vol1 vol2 vol3 : ℝ) (perc1 perc2 perc3 : ℝ) (totalVol : ℝ) (finalVol : ℝ) :=
  (vol1 * perc1 + vol2 * perc2 + vol3 * perc3) / finalVol

theorem concentration_proof : 
  newConcentration 2 6 4 0.2 0.55 0.35 (12 : ℝ) (15 : ℝ) = 0.34 := 
by 
  sorry

end NUMINAMATH_GPT_concentration_proof_l2307_230751


namespace NUMINAMATH_GPT_chess_tournament_rounds_needed_l2307_230704

theorem chess_tournament_rounds_needed
  (num_players : ℕ)
  (num_games_per_round : ℕ)
  (H1 : num_players = 20)
  (H2 : num_games_per_round = 10) :
  (num_players * (num_players - 1)) / num_games_per_round = 38 :=
by
  sorry

end NUMINAMATH_GPT_chess_tournament_rounds_needed_l2307_230704


namespace NUMINAMATH_GPT_mechanism_parts_l2307_230740

theorem mechanism_parts (L S : ℕ) (h1 : L + S = 30) (h2 : L ≤ 11) (h3 : S ≤ 19) :
  L = 11 ∧ S = 19 :=
by
  sorry

end NUMINAMATH_GPT_mechanism_parts_l2307_230740


namespace NUMINAMATH_GPT_find_n_l2307_230761

theorem find_n : ∃ n : ℤ, 0 ≤ n ∧ n < 103 ∧ 100 * n % 103 = 65 % 103 ∧ n = 68 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l2307_230761


namespace NUMINAMATH_GPT_third_price_reduction_l2307_230778

theorem third_price_reduction (original_price final_price : ℝ) (x : ℝ) 
  (h1 : (original_price * (1 - x)^2 = final_price))
  (h2 : final_price = 100)
  (h3 : original_price = 100 / (1 - 0.19)) :
  (original_price * (1 - x)^3 = 90) :=
by
  sorry

end NUMINAMATH_GPT_third_price_reduction_l2307_230778


namespace NUMINAMATH_GPT_probability_of_both_types_probability_distribution_and_expectation_of_X_l2307_230724

-- Definitions
def total_zongzi : ℕ := 8
def red_bean_paste_zongzi : ℕ := 2
def date_zongzi : ℕ := 6
def selected_zongzi : ℕ := 3

-- Part 1: The probability of selecting both red bean paste and date zongzi
theorem probability_of_both_types :
  let total_combinations := Nat.choose total_zongzi selected_zongzi
  let one_red_two_date := Nat.choose red_bean_paste_zongzi 1 * Nat.choose date_zongzi 2
  let two_red_one_date := Nat.choose red_bean_paste_zongzi 2 * Nat.choose date_zongzi 1
  (one_red_two_date + two_red_one_date) / total_combinations = 9 / 14 :=
by sorry

-- Part 2: The probability distribution and expectation of X
theorem probability_distribution_and_expectation_of_X :
  let P_X_0 := (Nat.choose red_bean_paste_zongzi 0 * Nat.choose date_zongzi 3) / Nat.choose total_zongzi selected_zongzi
  let P_X_1 := (Nat.choose red_bean_paste_zongzi 1 * Nat.choose date_zongzi 2) / Nat.choose total_zongzi selected_zongzi
  let P_X_2 := (Nat.choose red_bean_paste_zongzi 2 * Nat.choose date_zongzi 1) / Nat.choose total_zongzi selected_zongzi
  P_X_0 = 5 / 14 ∧ P_X_1 = 15 / 28 ∧ P_X_2 = 3 / 28 ∧
  (0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2 = 3 / 4) :=
by sorry

end NUMINAMATH_GPT_probability_of_both_types_probability_distribution_and_expectation_of_X_l2307_230724


namespace NUMINAMATH_GPT_total_cost_of_trick_decks_l2307_230723

theorem total_cost_of_trick_decks (cost_per_deck: ℕ) (victor_decks: ℕ) (friend_decks: ℕ) (total_spent: ℕ) : 
  cost_per_deck = 8 → victor_decks = 6 → friend_decks = 2 → total_spent = cost_per_deck * victor_decks + cost_per_deck * friend_decks → total_spent = 64 :=
by 
  sorry

end NUMINAMATH_GPT_total_cost_of_trick_decks_l2307_230723


namespace NUMINAMATH_GPT_determine_k_l2307_230776

theorem determine_k (k : ℚ) (h_collinear : ∃ (f : ℚ → ℚ), 
  f 0 = 3 ∧ f 7 = k ∧ f 21 = 2) : k = 8 / 3 :=
by
  sorry

end NUMINAMATH_GPT_determine_k_l2307_230776


namespace NUMINAMATH_GPT_monkey2_peach_count_l2307_230733

noncomputable def total_peaches : ℕ := 81
def monkey1_share (p : ℕ) : ℕ := (5 * p) / 6
def remaining_after_monkey1 (p : ℕ) : ℕ := p - monkey1_share p
def monkey2_share (p : ℕ) : ℕ := (5 * remaining_after_monkey1 p) / 9
def remaining_after_monkey2 (p : ℕ) : ℕ := remaining_after_monkey1 p - monkey2_share p
def monkey3_share (p : ℕ) : ℕ := remaining_after_monkey2 p

theorem monkey2_peach_count : monkey2_share total_peaches = 20 :=
by
  sorry

end NUMINAMATH_GPT_monkey2_peach_count_l2307_230733


namespace NUMINAMATH_GPT_division_value_l2307_230719

theorem division_value (n x : ℝ) (h₀ : n = 4.5) (h₁ : (n / x) * 12 = 9) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_division_value_l2307_230719


namespace NUMINAMATH_GPT_find_value_of_a_l2307_230726

-- Definitions based on the conditions
def x (k : ℕ) : ℕ := 3 * k
def y (k : ℕ) : ℕ := 4 * k
def z (k : ℕ) : ℕ := 6 * k

-- Setting up the sum equation
def sum_eq_52 (k : ℕ) : Prop := x k + y k + z k = 52

-- Defining the y equation
def y_eq (a : ℚ) (k : ℕ) : Prop := y k = 15 * a + 5

-- Stating the main problem
theorem find_value_of_a (a : ℚ) (k : ℕ) : sum_eq_52 k → y_eq a k → a = 11 / 15 := by
  sorry

end NUMINAMATH_GPT_find_value_of_a_l2307_230726


namespace NUMINAMATH_GPT_max_smart_winners_min_total_prize_l2307_230747

-- Define relevant constants and conditions
def total_winners := 25
def prize_smart : ℕ := 15
def prize_comprehensive : ℕ := 30

-- Problem 1: Maximum number of winners in "Smartest Brain" competition
theorem max_smart_winners (x : ℕ) (h1 : total_winners = 25)
  (h2 : total_winners - x ≥ 5 * x) : x ≤ 4 :=
sorry

-- Problem 2: Minimum total prize amount
theorem min_total_prize (y : ℕ) (h1 : y ≤ 4)
  (h2 : total_winners = 25)
  (h3 : (total_winners - y) ≥ 5 * y)
  (h4 : prize_smart = 15)
  (h5 : prize_comprehensive = 30) :
  15 * y + 30 * (25 - y) = 690 :=
sorry

end NUMINAMATH_GPT_max_smart_winners_min_total_prize_l2307_230747


namespace NUMINAMATH_GPT_sqrt_of_square_neg7_l2307_230799

theorem sqrt_of_square_neg7 : Real.sqrt ((-7:ℝ)^2) = 7 := by
  sorry

end NUMINAMATH_GPT_sqrt_of_square_neg7_l2307_230799


namespace NUMINAMATH_GPT_find_divisor_l2307_230783

theorem find_divisor (D : ℕ) : 
  (242 % D = 15) ∧ 
  (698 % D = 27) ∧ 
  ((242 + 698) % D = 5) → 
  D = 42 := 
by 
  sorry

end NUMINAMATH_GPT_find_divisor_l2307_230783


namespace NUMINAMATH_GPT_die_total_dots_l2307_230791

theorem die_total_dots :
  ∀ (face1 face2 face3 face4 face5 face6 : ℕ),
    face1 < face2 ∧ face2 < face3 ∧ face3 < face4 ∧ face4 < face5 ∧ face5 < face6 ∧
    (face2 - face1 ≥ 2) ∧ (face3 - face2 ≥ 2) ∧ (face4 - face3 ≥ 2) ∧ (face5 - face4 ≥ 2) ∧ (face6 - face5 ≥ 2) ∧
    (face3 ≠ face1 + 2) ∧ (face4 ≠ face2 + 2) ∧ (face5 ≠ face3 + 2) ∧ (face6 ≠ face4 + 2)
    → face1 + face2 + face3 + face4 + face5 + face6 = 27 :=
by {
  sorry
}

end NUMINAMATH_GPT_die_total_dots_l2307_230791


namespace NUMINAMATH_GPT_find_smaller_angle_l2307_230731

theorem find_smaller_angle (x : ℝ) (h1 : (x + (x + 18) = 180)) : x = 81 := 
by 
  sorry

end NUMINAMATH_GPT_find_smaller_angle_l2307_230731


namespace NUMINAMATH_GPT_one_eq_a_l2307_230712

theorem one_eq_a (x y z a : ℝ) (h₁: x + y + z = a) (h₂: 1/x + 1/y + 1/z = 1/a) :
  x = a ∨ y = a ∨ z = a :=
  sorry

end NUMINAMATH_GPT_one_eq_a_l2307_230712


namespace NUMINAMATH_GPT_inequality_problem_l2307_230725

-- Define the conditions and the problem statement
theorem inequality_problem (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  (a - b * c) / (a + b * c) + (b - c * a) / (b + c * a) + (c - a * b) / (c + a * b) ≤ 3 / 2 :=
sorry

end NUMINAMATH_GPT_inequality_problem_l2307_230725


namespace NUMINAMATH_GPT_blue_length_is_2_l2307_230737

-- Define the lengths of the parts
def total_length : ℝ := 4
def purple_length : ℝ := 1.5
def black_length : ℝ := 0.5

-- Define the length of the blue part with the given conditions
def blue_length : ℝ := total_length - (purple_length + black_length)

-- State the theorem we need to prove
theorem blue_length_is_2 : blue_length = 2 :=
by 
  sorry

end NUMINAMATH_GPT_blue_length_is_2_l2307_230737


namespace NUMINAMATH_GPT_courtyard_width_l2307_230763

theorem courtyard_width (length : ℕ) (brick_length brick_width : ℕ) (num_bricks : ℕ) (W : ℕ)
  (H1 : length = 25)
  (H2 : brick_length = 20)
  (H3 : brick_width = 10)
  (H4 : num_bricks = 18750)
  (H5 : 2500 * (W * 100) = num_bricks * (brick_length * brick_width)) :
  W = 15 :=
by sorry

end NUMINAMATH_GPT_courtyard_width_l2307_230763


namespace NUMINAMATH_GPT_arithmetic_sequence_third_term_l2307_230789

theorem arithmetic_sequence_third_term (a d : ℤ) 
  (h20 : a + 19 * d = 17) (h21 : a + 20 * d = 20) : a + 2 * d = -34 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_third_term_l2307_230789


namespace NUMINAMATH_GPT_common_ratio_of_gp_l2307_230729

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
if r = 1 then a * n else a * (1 - r^n) / (1 - r)

theorem common_ratio_of_gp (a : ℝ) (r : ℝ) (h : geometric_sum a r 6 / geometric_sum a r 3 = 28) : r = 3 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_of_gp_l2307_230729


namespace NUMINAMATH_GPT_increasing_interval_l2307_230775

noncomputable def f (x : ℝ) := Real.log x / Real.log (1 / 2)

def is_monotonically_increasing (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y

def h (x : ℝ) : ℝ := x^2 + x - 2

theorem increasing_interval :
  is_monotonically_increasing (f ∘ h) {x : ℝ | x < -2} :=
sorry

end NUMINAMATH_GPT_increasing_interval_l2307_230775


namespace NUMINAMATH_GPT_positive_difference_of_two_numbers_l2307_230717

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_of_two_numbers_l2307_230717


namespace NUMINAMATH_GPT_num_members_in_league_l2307_230742

theorem num_members_in_league :
  let sock_cost := 5
  let tshirt_cost := 11
  let total_exp := 3100
  let cost_per_member_before_discount := 2 * (sock_cost + tshirt_cost)
  let discount := 3
  let effective_cost_per_member := cost_per_member_before_discount - discount
  let num_members := total_exp / effective_cost_per_member
  num_members = 150 :=
by
  let sock_cost := 5
  let tshirt_cost := 11
  let total_exp := 3100
  let cost_per_member_before_discount := 2 * (sock_cost + tshirt_cost)
  let discount := 3
  let effective_cost_per_member := cost_per_member_before_discount - discount
  let num_members := total_exp / effective_cost_per_member
  sorry

end NUMINAMATH_GPT_num_members_in_league_l2307_230742


namespace NUMINAMATH_GPT_find_angle_l2307_230745

variable (a b : ℝ × ℝ) (α : ℝ)
variable (θ : ℝ)

-- Conditions provided in the problem
def condition1 := (a.1^2 + a.2^2 = 4)
def condition2 := (b = (4 * Real.cos α, -4 * Real.sin α))
def condition3 := (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0)

-- Desired result
theorem find_angle (h1 : condition1 a) (h2 : condition2 b α) (h3 : condition3 a b) :
  θ = Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_find_angle_l2307_230745


namespace NUMINAMATH_GPT_popsicles_consumed_l2307_230734

def total_minutes (hours : ℕ) (additional_minutes : ℕ) : ℕ :=
  hours * 60 + additional_minutes

def popsicles_in_time (total_time : ℕ) (interval : ℕ) : ℕ :=
  total_time / interval

theorem popsicles_consumed : popsicles_in_time (total_minutes 4 30) 15 = 18 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_popsicles_consumed_l2307_230734


namespace NUMINAMATH_GPT_maximum_revenue_l2307_230785

def ticket_price (x : ℕ) (y : ℤ) : Prop :=
  (6 ≤ x ∧ x ≤ 10 ∧ y = 1000 * x - 5750) ∨
  (10 < x ∧ x ≤ 38 ∧ y = -30 * x^2 + 1300 * x - 5750)

theorem maximum_revenue :
  ∃ x y, ticket_price x y ∧ y = 8830 ∧ x = 22 :=
by {
  sorry
}

end NUMINAMATH_GPT_maximum_revenue_l2307_230785


namespace NUMINAMATH_GPT_prob_qualified_bulb_factory_a_l2307_230774

-- Define the given probability of a light bulb being produced by Factory A
def prob_factory_a : ℝ := 0.7

-- Define the given pass rate (conditional probability) of Factory A's light bulbs
def pass_rate_factory_a : ℝ := 0.95

-- The goal is to prove that the probability of getting a qualified light bulb produced by Factory A is 0.665
theorem prob_qualified_bulb_factory_a : prob_factory_a * pass_rate_factory_a = 0.665 :=
by
  -- This is where the proof would be, but we'll use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_prob_qualified_bulb_factory_a_l2307_230774


namespace NUMINAMATH_GPT_abs_ineq_solution_l2307_230714

theorem abs_ineq_solution (x : ℝ) :
  (|x - 2| + |x + 1| < 4) ↔ (x ∈ Set.Ioo (-7 / 2) (-1) ∪ Set.Ico (-1) (5 / 2)) := by
  sorry

end NUMINAMATH_GPT_abs_ineq_solution_l2307_230714


namespace NUMINAMATH_GPT_original_price_of_car_l2307_230781

theorem original_price_of_car (P : ℝ) 
  (h₁ : 0.561 * P + 200 = 7500) : 
  P = 13012.48 := 
sorry

end NUMINAMATH_GPT_original_price_of_car_l2307_230781


namespace NUMINAMATH_GPT_johns_ratio_l2307_230793

-- Definitions for initial counts
def initial_pink := 26
def initial_green := 15
def initial_yellow := 24
def initial_total := initial_pink + initial_green + initial_yellow

-- Definitions for Carl's and John's actions
def carl_pink_taken := 4
def john_pink_taken := 6
def remaining_pink := initial_pink - carl_pink_taken - john_pink_taken

-- Definition for remaining hard hats
def total_remaining := 43

-- Compute John's green hat withdrawal
def john_green_taken := (initial_total - carl_pink_taken - john_pink_taken) - total_remaining
def ratio := john_green_taken / john_pink_taken

theorem johns_ratio : ratio = 2 :=
by
  -- Proof details omitted
  sorry

end NUMINAMATH_GPT_johns_ratio_l2307_230793


namespace NUMINAMATH_GPT_find_inequality_solution_l2307_230702

theorem find_inequality_solution :
  {x : ℝ | (x + 1) / (x - 2) + (x + 3) / (2 * x + 1) ≤ 2}
  = {x : ℝ | -1 / 2 ≤ x ∧ x ≤ 1 ∨ 2 ≤ x ∧ x ≤ 9} :=
by
  -- The proof steps are omitted.
  sorry

end NUMINAMATH_GPT_find_inequality_solution_l2307_230702


namespace NUMINAMATH_GPT_sum_in_base_b_l2307_230708

-- Definitions needed to articulate the problem
def base_b_value (n : ℕ) (b : ℕ) : ℕ :=
  match n with
  | 12 => b + 2
  | 15 => b + 5
  | 16 => b + 6
  | 3146 => 3 * b^3 + 1 * b^2 + 4 * b + 6
  | _  => 0

def s_in_base_b (b : ℕ) : ℕ :=
  base_b_value 12 b + base_b_value 15 b + base_b_value 16 b

theorem sum_in_base_b (b : ℕ) (h : (base_b_value 12 b) * (base_b_value 15 b) * (base_b_value 16 b) = base_b_value 3146 b) :
  s_in_base_b b = 44 := by
  sorry

end NUMINAMATH_GPT_sum_in_base_b_l2307_230708


namespace NUMINAMATH_GPT_pythagorean_triangle_inscribed_circle_radius_is_integer_l2307_230743

theorem pythagorean_triangle_inscribed_circle_radius_is_integer 
  (a b c : ℕ)
  (h1 : c^2 = a^2 + b^2) 
  (h2 : r = (a + b - c) / 2) :
  ∃ (r : ℕ), r = (a + b - c) / 2 :=
sorry

end NUMINAMATH_GPT_pythagorean_triangle_inscribed_circle_radius_is_integer_l2307_230743


namespace NUMINAMATH_GPT_parabola_sum_l2307_230796

theorem parabola_sum (a b c : ℝ)
  (h1 : 4 = a * 1^2 + b * 1 + c)
  (h2 : -1 = a * (-2)^2 + b * (-2) + c)
  (h3 : ∀ x : ℝ, a * x^2 + b * x + c = a * (x + 1)^2 - 2)
  : a + b + c = 5 := by
  sorry

end NUMINAMATH_GPT_parabola_sum_l2307_230796


namespace NUMINAMATH_GPT_remaining_sweet_potatoes_l2307_230735

def harvested_sweet_potatoes : ℕ := 80
def sold_sweet_potatoes_mrs_adams : ℕ := 20
def sold_sweet_potatoes_mr_lenon : ℕ := 15
def traded_sweet_potatoes : ℕ := 10
def donated_sweet_potatoes : ℕ := 5

theorem remaining_sweet_potatoes :
  harvested_sweet_potatoes - (sold_sweet_potatoes_mrs_adams + sold_sweet_potatoes_mr_lenon + traded_sweet_potatoes + donated_sweet_potatoes) = 30 :=
by
  sorry

end NUMINAMATH_GPT_remaining_sweet_potatoes_l2307_230735


namespace NUMINAMATH_GPT_ratio_D_E_equal_l2307_230786

variable (total_characters : ℕ) (initial_A : ℕ) (initial_C : ℕ) (initial_D : ℕ) (initial_E : ℕ)

def mary_story_conditions (total_characters : ℕ) (initial_A : ℕ) (initial_C : ℕ) (initial_D : ℕ) (initial_E : ℕ) : Prop :=
  total_characters = 60 ∧
  initial_A = 1 / 2 * total_characters ∧
  initial_C = 1 / 2 * initial_A ∧
  initial_D + initial_E = total_characters - (initial_A + initial_C)

theorem ratio_D_E_equal (total_characters initial_A initial_C initial_D initial_E : ℕ) :
  mary_story_conditions total_characters initial_A initial_C initial_D initial_E →
  initial_D = initial_E :=
sorry

end NUMINAMATH_GPT_ratio_D_E_equal_l2307_230786


namespace NUMINAMATH_GPT_min_value_of_expression_l2307_230777

theorem min_value_of_expression : 
  ∃ x y : ℝ, (z = x^2 + 2*x*y + 2*y^2 + 2*x + 4*y + 3) ∧ z = 1 ∧ x = 0 ∧ y = -1 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l2307_230777


namespace NUMINAMATH_GPT_initial_total_quantity_l2307_230779

theorem initial_total_quantity
  (x : ℝ)
  (milk_water_ratio : 5 / 9 = 5 * x / (3 * x + 12))
  (milk_juice_ratio : 5 / 8 = 5 * x / (4 * x + 6)) :
  5 * x + 3 * x + 4 * x = 24 :=
by
  sorry

end NUMINAMATH_GPT_initial_total_quantity_l2307_230779


namespace NUMINAMATH_GPT_total_trapezoid_area_l2307_230766

def large_trapezoid_area (AB CD altitude_L : ℝ) : ℝ :=
  0.5 * (AB + CD) * altitude_L

def small_trapezoid_area (EF GH altitude_S : ℝ) : ℝ :=
  0.5 * (EF + GH) * altitude_S

def total_area (large_area small_area : ℝ) : ℝ :=
  large_area + small_area

theorem total_trapezoid_area :
  large_trapezoid_area 60 30 15 + small_trapezoid_area 25 10 5 = 762.5 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_trapezoid_area_l2307_230766


namespace NUMINAMATH_GPT_hypotenuse_length_l2307_230784

theorem hypotenuse_length (a b c : ℝ) (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l2307_230784


namespace NUMINAMATH_GPT_vector_BC_l2307_230795

def vector_subtraction (v1 v2 : ℤ × ℤ) : ℤ × ℤ :=
(v1.1 - v2.1, v1.2 - v2.2)

theorem vector_BC (BA CA BC : ℤ × ℤ) (hBA : BA = (2, 3)) (hCA : CA = (4, 7)) :
  BC = vector_subtraction BA CA → BC = (-2, -4) :=
by
  intro hBC
  rw [vector_subtraction, hBA, hCA] at hBC
  simpa using hBC

end NUMINAMATH_GPT_vector_BC_l2307_230795


namespace NUMINAMATH_GPT_trivia_team_total_points_l2307_230787

def totalPoints : Nat := 182

def points_member_A : Nat := 3 * 2
def points_member_B : Nat := 5 * 4 + 1 * 6
def points_member_C : Nat := 2 * 6
def points_member_D : Nat := 4 * 2 + 2 * 4
def points_member_E : Nat := 1 * 2 + 3 * 4
def points_member_F : Nat := 5 * 6
def points_member_G : Nat := 2 * 4 + 1 * 2
def points_member_H : Nat := 3 * 6 + 2 * 2
def points_member_I : Nat := 1 * 4 + 4 * 6
def points_member_J : Nat := 7 * 2 + 1 * 4

theorem trivia_team_total_points : 
  points_member_A + points_member_B + points_member_C + points_member_D + points_member_E + 
  points_member_F + points_member_G + points_member_H + points_member_I + points_member_J = totalPoints := 
by
  repeat { sorry }

end NUMINAMATH_GPT_trivia_team_total_points_l2307_230787


namespace NUMINAMATH_GPT_simplify_expression_l2307_230752

theorem simplify_expression (p : ℤ) : 
  ((7 * p + 3) - 3 * p * 2) * 4 + (5 - 2 / 2) * (8 * p - 12) = 36 * p - 36 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2307_230752


namespace NUMINAMATH_GPT_least_positive_integer_satifies_congruences_l2307_230773

theorem least_positive_integer_satifies_congruences :
  ∃ x : ℕ, x ≡ 1 [MOD 4] ∧ x ≡ 2 [MOD 5] ∧ x ≡ 3 [MOD 6] ∧ x = 17 :=
sorry

end NUMINAMATH_GPT_least_positive_integer_satifies_congruences_l2307_230773


namespace NUMINAMATH_GPT_abs_diff_of_two_numbers_l2307_230755

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 216) : |x - y| = 6 := 
sorry

end NUMINAMATH_GPT_abs_diff_of_two_numbers_l2307_230755


namespace NUMINAMATH_GPT_interest_group_selections_l2307_230748

-- Define the number of students and the number of interest groups
def num_students : ℕ := 4
def num_groups : ℕ := 3

-- Theorem statement: The total number of different possible selections of interest groups is 81.
theorem interest_group_selections : num_groups ^ num_students = 81 := by
  sorry

end NUMINAMATH_GPT_interest_group_selections_l2307_230748


namespace NUMINAMATH_GPT_complement_intersection_l2307_230765

def M : Set ℝ := {x | x ≥ 1}
def N : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

theorem complement_intersection : (M ∩ N)ᶜ = { x : ℝ | x < 1 ∨ x > 3 } :=
  sorry

end NUMINAMATH_GPT_complement_intersection_l2307_230765


namespace NUMINAMATH_GPT_max_crystalline_polyhedron_volume_l2307_230759

theorem max_crystalline_polyhedron_volume (n : ℕ) (R : ℝ) (h_n : n > 1) :
  ∃ V : ℝ, 
    V = (32 / 81) * (n - 1) * (R ^ 3) * Real.sin (2 * Real.pi / (n - 1)) :=
sorry

end NUMINAMATH_GPT_max_crystalline_polyhedron_volume_l2307_230759


namespace NUMINAMATH_GPT_total_toes_on_bus_l2307_230709

-- Definitions based on conditions
def toes_per_hand_hoopit : Nat := 3
def hands_per_hoopit : Nat := 4
def number_of_hoopits_on_bus : Nat := 7

def toes_per_hand_neglart : Nat := 2
def hands_per_neglart : Nat := 5
def number_of_neglarts_on_bus : Nat := 8

-- We need to prove that the total number of toes on the bus is 164
theorem total_toes_on_bus :
  (toes_per_hand_hoopit * hands_per_hoopit * number_of_hoopits_on_bus) +
  (toes_per_hand_neglart * hands_per_neglart * number_of_neglarts_on_bus) = 164 :=
by sorry

end NUMINAMATH_GPT_total_toes_on_bus_l2307_230709


namespace NUMINAMATH_GPT_original_cost_of_horse_l2307_230754

theorem original_cost_of_horse (x : ℝ) (h : x - x^2 / 100 = 24) : x = 40 ∨ x = 60 := 
by 
  sorry

end NUMINAMATH_GPT_original_cost_of_horse_l2307_230754


namespace NUMINAMATH_GPT_first_discount_percentage_l2307_230788

-- Given conditions
def initial_price : ℝ := 390
def final_price : ℝ := 285.09
def second_discount : ℝ := 0.15

-- Definition for the first discount percentage
noncomputable def first_discount (D : ℝ) : ℝ :=
initial_price * (1 - D / 100) * (1 - second_discount)

-- Theorem statement
theorem first_discount_percentage : ∃ D : ℝ, first_discount D = final_price ∧ D = 13.99 :=
by
  sorry

end NUMINAMATH_GPT_first_discount_percentage_l2307_230788


namespace NUMINAMATH_GPT_repayment_correct_l2307_230732

noncomputable def repayment_amount (a γ : ℝ) : ℝ :=
  a * γ * (1 + γ) ^ 5 / ((1 + γ) ^ 5 - 1)

theorem repayment_correct (a γ : ℝ) (γ_pos : γ > 0) : 
  repayment_amount a γ = a * γ * (1 + γ) ^ 5 / ((1 + γ) ^ 5 - 1) :=
by
   sorry

end NUMINAMATH_GPT_repayment_correct_l2307_230732


namespace NUMINAMATH_GPT_mohan_least_cookies_l2307_230713

theorem mohan_least_cookies :
  ∃ b : ℕ, 
    b % 6 = 5 ∧
    b % 8 = 3 ∧
    b % 9 = 6 ∧
    b = 59 :=
by
  sorry

end NUMINAMATH_GPT_mohan_least_cookies_l2307_230713


namespace NUMINAMATH_GPT_maria_travel_fraction_l2307_230703

theorem maria_travel_fraction (x : ℝ) (total_distance : ℝ)
  (h1 : ∀ d1 d2, d1 + d2 = total_distance)
  (h2 : total_distance = 360)
  (h3 : ∃ d1 d2 d3, d1 = 360 * x ∧ d2 = (1 / 4) * (360 - 360 * x) ∧ d3 = 135)
  (h4 : d1 + d2 + d3 = total_distance)
  : x = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_maria_travel_fraction_l2307_230703


namespace NUMINAMATH_GPT_min_value_expression_l2307_230770

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ∃ a b, a > 0 ∧ b > 0 ∧ (∀ x y, x > 0 ∧ y > 0 → (1 / x + x / y^2 + y ≥ 2 * Real.sqrt 2)) := 
sorry

end NUMINAMATH_GPT_min_value_expression_l2307_230770
