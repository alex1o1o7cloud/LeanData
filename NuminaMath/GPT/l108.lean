import Mathlib

namespace count_digit_7_in_range_l108_108048

theorem count_digit_7_in_range : ∀ n ∈ finset.Icc 100 199, (n % 10 = 7 ∨ n / 10 % 10 = 7) →  ∃ counts, counts = 20 :=
by
  sorry

end count_digit_7_in_range_l108_108048


namespace sqrt_mul_sqrt_l108_108247

theorem sqrt_mul_sqrt (h1 : Real.sqrt 25 = 5) : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_mul_sqrt_l108_108247


namespace box_surface_area_l108_108717

theorem box_surface_area (w l s tab : ℕ):
  w = 40 → l = 60 → s = 8 → tab = 2 →
  (40 * 60 - 4 * 8 * 8 + 2 * (2 * (60 - 2 * 8) + 2 * (40 - 2 * 8))) = 2416 :=
by
  intros _ _ _ _
  sorry

end box_surface_area_l108_108717


namespace chords_even_arcs_even_l108_108225

theorem chords_even_arcs_even (N : ℕ) 
  (h1 : ∀ k : ℕ, k ≤ N → ¬ ((k : ℤ) % 2 = 1)) : 
  N % 2 = 0 := 
sorry

end chords_even_arcs_even_l108_108225


namespace area_of_triangle_ABC_with_B_45_degrees_area_of_triangle_ABC_with_c_sqrt3_b_l108_108192

open Real

-- Definitions
def is_triangle (A B C: ℝ) := A + B + C = π
def side_opposite_to_angle (a b c A B C : ℝ) := is_triangle A B C ∧
  (a^2 + b^2 - 2 * a * b * cos A = c^2)

-- Given conditions
def given_conditions (A B c: ℝ) := sin A + sqrt 3 * cos A = 2 ∧ A = π / 6 ∧ B = π / 4 ∧ c = sqrt 3

-- 1. Prove that, given a = 2 and B = 45 degrees, the area is sqrt(3) + 1
theorem area_of_triangle_ABC_with_B_45_degrees :
  ∀ (a b c A B C : ℝ), a = 2 → given_conditions A B c →
    side_opposite_to_angle a b c A B C →
    (1 / 2) * a * b * sin C = sqrt 3 + 1 :=
begin
  intros a b c A B C h_a h_cond h_triangle,
  sorry,
end

-- 2. Prove that, given a = 2 and c = sqrt(3)b, the area is sqrt(3)
theorem area_of_triangle_ABC_with_c_sqrt3_b :
  ∀ (a b c A B C : ℝ), a = 2 → given_conditions A B c →
    side_opposite_to_angle a b c A B C →
    (1 / 2) * b * c * sin A = sqrt 3 :=
begin
  intros a b c A B C h_a h_cond h_triangle,
  sorry,
end

end area_of_triangle_ABC_with_B_45_degrees_area_of_triangle_ABC_with_c_sqrt3_b_l108_108192


namespace percentage_of_tomato_plants_is_20_l108_108070

-- Define the conditions
def garden1_plants := 20
def garden1_tomato_percentage := 0.10
def garden2_plants := 15
def garden2_tomato_percentage := 1 / 3

-- Define the question as a theorem to be proved
theorem percentage_of_tomato_plants_is_20 :
  let total_plants := garden1_plants + garden2_plants in
  let total_tomato_plants := (garden1_tomato_percentage * garden1_plants) + (garden2_tomato_percentage * garden2_plants) in
  (total_tomato_plants / total_plants) * 100 = 20 :=
by
  sorry

end percentage_of_tomato_plants_is_20_l108_108070


namespace sqrt_mul_sqrt_l108_108259

theorem sqrt_mul_sqrt (a b c : ℝ) (hac : a = 49) (hbc : b = sqrt 25) (hc : c = 7 * sqrt 5) : sqrt (a * b) = c := 
by
  sorry

end sqrt_mul_sqrt_l108_108259


namespace sum_of_solutions_eq_zero_l108_108134

theorem sum_of_solutions_eq_zero :
  let f (x : ℝ) := 2^|x| + 4 * |x|
  (∀ x : ℝ, f x = 20) →
  (∃ x₁ x₂ : ℝ, f x₁ = 20 ∧ f x₂ = 20 ∧ x₁ + x₂ = 0) :=
sorry

end sum_of_solutions_eq_zero_l108_108134


namespace art_club_artworks_l108_108236

-- Define the conditions
def students := 25
def artworks_per_student_per_quarter := 3
def quarters_per_year := 4
def years := 3

-- Calculate total artworks
theorem art_club_artworks : 
  students * artworks_per_student_per_quarter * quarters_per_year * years = 900 :=
by
  sorry

end art_club_artworks_l108_108236


namespace surface_area_of_sphere_with_diameter_two_l108_108879

theorem surface_area_of_sphere_with_diameter_two :
  let diameter := 2
  let radius := diameter / 2
  4 * Real.pi * radius ^ 2 = 4 * Real.pi :=
by
  sorry

end surface_area_of_sphere_with_diameter_two_l108_108879


namespace sqrt_49_mul_sqrt_25_l108_108279

theorem sqrt_49_mul_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_l108_108279


namespace sqrt_49_times_sqrt_25_eq_7sqrt5_l108_108261

theorem sqrt_49_times_sqrt_25_eq_7sqrt5 :
  (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  have h1 : Real.sqrt 25 = 5 := by sorry
  have h2 : 49 * 5 = 245 := by sorry
  have h3 : Real.sqrt 245 = 7 * Real.sqrt 5 := by sorry
  sorry

end sqrt_49_times_sqrt_25_eq_7sqrt5_l108_108261


namespace smallest_n_boxes_cookies_l108_108058

theorem smallest_n_boxes_cookies (n : ℕ) (h : (17 * n - 1) % 12 = 0) : n = 5 :=
sorry

end smallest_n_boxes_cookies_l108_108058


namespace no_real_root_for_3_in_g_l108_108144

noncomputable def g (x c : ℝ) : ℝ := x^2 + 3 * x + c

theorem no_real_root_for_3_in_g (c : ℝ) :
  (21 - 4 * c) < 0 ↔ c > 21 / 4 := by
sorry

end no_real_root_for_3_in_g_l108_108144


namespace distinct_parallel_lines_l108_108726

theorem distinct_parallel_lines (k : ℝ) :
  (∃ (L1 L2 : ℝ × ℝ → Prop), 
    (∀ x y, L1 (x, y) ↔ x - 2 * y - 3 = 0) ∧ 
    (∀ x y, L2 (x, y) ↔ 18 * x - k^2 * y - 9 * k = 0)) → 
  (∃ slope1 slope2, 
    slope1 = 1/2 ∧ 
    slope2 = 18 / k^2 ∧
    (slope1 = slope2) ∧
    (¬ (∀ x y, x - 2 * y - 3 = 18 * x - k^2 * y - 9 * k))) → 
  k = -6 :=
by 
  sorry

end distinct_parallel_lines_l108_108726


namespace at_least_six_destinations_l108_108114

theorem at_least_six_destinations (destinations : ℕ) (tickets_sold : ℕ) (h_dest : destinations = 200) (h_tickets : tickets_sold = 3800) :
  ∃ k ≥ 6, ∃ t : ℕ, (∃ f : Fin destinations → ℕ, (∀ i : Fin destinations, f i ≤ t) ∧ (tickets_sold ≤ t * destinations) ∧ ((∃ i : Fin destinations, f i = k) → k ≥ 6)) :=
by
  sorry

end at_least_six_destinations_l108_108114


namespace problem1_problem2_l108_108522

open Real

noncomputable def f (a x : ℝ) : ℝ := a * log x + 0.5 * x^2 - a * x

theorem problem1 (a : ℝ) : (a > 4) ↔ (∀ x : ℝ, a * log x + 0.5 * x^2 - a * x has two distinct positive roots where x^2 - a * x + a = 0) := sorry

theorem problem2 (a : ℝ) (h : a > 4) : ∀ x1 x2 : ℝ, (x1 + x2 = a) ∧ (x1 * x2 = a) → (f a x1 + f a x2 < λ * (x1 + x2)) → (λ ≥ log 4 - 3) := sorry

end problem1_problem2_l108_108522


namespace coeff_x5_term_l108_108045

-- We define the binomial coefficient function C(n, k)
def C (n k : ℕ) : ℕ := Nat.choose n k

-- We define the expression in question
noncomputable def expr (x : ℝ) : ℝ := (1/x + 2*x)^7

-- The coefficient of x^5 term in the expansion
theorem coeff_x5_term : 
  let general_term (r : ℕ) (x : ℝ) := (2:ℝ)^r * C 7 r * x^(2 * r - 7)
  -- r is chosen such that the power of x is 5
  let r := 6
  -- The coefficient for r=6
  general_term r 1 = 448 := 
by sorry

end coeff_x5_term_l108_108045


namespace part1_part2_l108_108669

theorem part1 (m : ℝ) :
  ∀ x : ℝ, x^2 + ( (2 * m - 1) : ℝ) * x + m^2 = 0 → m ≤ 1 / 4 :=
sorry

theorem part2 (m : ℝ) 
  (h : ∀ x1 x2 : ℝ, (x1^2 + (2*m -1)*x1 + m^2 = 0) ∧ (x2^2 + (2*m -1)*x2 + m^2 = 0) ∧ (x1*x2 + x1 + x2 = 4)) :
    m = -1 :=
sorry

end part1_part2_l108_108669


namespace digit_7_appears_20_times_in_range_100_to_199_l108_108049

theorem digit_7_appears_20_times_in_range_100_to_199 : 
  ∃ (d7_count : ℕ), (d7_count = 20) ∧ (∀ (n : ℕ), 100 ≤ n ∧ n ≤ 199 → 
  d7_count = (n.to_string.count '7')) :=
sorry

end digit_7_appears_20_times_in_range_100_to_199_l108_108049


namespace mickys_sticks_more_l108_108216

theorem mickys_sticks_more 
  (simons_sticks : ℕ := 36)
  (gerrys_sticks : ℕ := (2 * simons_sticks) / 3)
  (total_sticks_needed : ℕ := 129)
  (total_simons_and_gerrys_sticks : ℕ := simons_sticks + gerrys_sticks)
  (mickys_sticks : ℕ := total_sticks_needed - total_simons_and_gerrys_sticks) :
  mickys_sticks - total_simons_and_gerrys_sticks = 9 :=
by
  sorry

end mickys_sticks_more_l108_108216


namespace positive_real_solutions_l108_108359

noncomputable def x1 := (75 + Real.sqrt 5773) / 2
noncomputable def x2 := (-50 + Real.sqrt 2356) / 2

theorem positive_real_solutions :
  ∀ x : ℝ, 
  0 < x → 
  (1/2 * (4*x^2 - 1) = (x^2 - 75*x - 15) * (x^2 + 50*x + 10)) ↔ 
  (x = x1 ∨ x = x2) :=
by
  sorry

end positive_real_solutions_l108_108359


namespace john_trip_l108_108194

theorem john_trip (t : ℝ) (h : t ≥ 0) : 
  ∀ t : ℝ, 60 * t + 90 * ((7 / 2) - t) = 300 :=
by sorry

end john_trip_l108_108194


namespace problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l108_108237

-- Define the conditions for each problem explicitly
def cond1 : Prop := ∃ (A B C : Type), -- "A" can only be in the middle or on the sides (positions are constrainted)
  True -- (specific arrangements are abstracted here)

def cond2 : Prop := ∃ (A B C : Type), -- male students must be grouped together
  True

def cond3 : Prop := ∃ (A B C : Type), -- male students cannot be grouped together
  True

def cond4 : Prop := ∃ (A B C : Type), -- the order of "A", "B", "C" from left to right remains unchanged
  True

def cond5 : Prop := ∃ (A B C : Type), -- "A" is not on the far left and "B" is not on the far right
  True

def cond6 : Prop := ∃ (A B C D : Type), -- One more female student, males and females are not next to each other
  True

def cond7 : Prop := ∃ (A B C : Type), -- arranged in two rows, with 3 people in the front row and 2 in the back row
  True

def cond8 : Prop := ∃ (A B C : Type), -- there must be 1 person between "A" and "B"
  True

-- Prove each condition results in the specified number of arrangements

theorem problem1 : cond1 → True := by
  -- Problem (1) is to show 72 arrangements given conditions
  sorry

theorem problem2 : cond2 → True := by
  -- Problem (2) is to show 36 arrangements given conditions
  sorry

theorem problem3 : cond3 → True := by
  -- Problem (3) is to show 12 arrangements given conditions
  sorry

theorem problem4 : cond4 → True := by
  -- Problem (4) is to show 20 arrangements given conditions
  sorry

theorem problem5 : cond5 → True := by
  -- Problem (5) is to show 78 arrangements given conditions
  sorry

theorem problem6 : cond6 → True := by
  -- Problem (6) is to show 144 arrangements given conditions
  sorry

theorem problem7 : cond7 → True := by
  -- Problem (7) is to show 120 arrangements given conditions
  sorry

theorem problem8 : cond8 → True := by
  -- Problem (8) is to show 36 arrangements given conditions
  sorry

end problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l108_108237


namespace probability_of_yellow_ball_is_correct_l108_108445

-- Defining the conditions
def red_balls : ℕ := 2
def yellow_balls : ℕ := 5
def blue_balls : ℕ := 4

-- Define the total number of balls
def total_balls : ℕ := red_balls + yellow_balls + blue_balls

-- Define the probability of choosing a yellow ball
def probability_yellow_ball : ℚ := yellow_balls / total_balls

-- The theorem statement we need to prove
theorem probability_of_yellow_ball_is_correct :
  probability_yellow_ball = 5 / 11 :=
sorry

end probability_of_yellow_ball_is_correct_l108_108445


namespace computer_price_after_9_years_l108_108434

theorem computer_price_after_9_years 
  (initial_price : ℝ) (decrease_factor : ℝ) (years : ℕ) 
  (initial_price_eq : initial_price = 8100)
  (decrease_factor_eq : decrease_factor = 1 - 1/3)
  (years_eq : years = 9) :
  initial_price * (decrease_factor ^ (years / 3)) = 2400 := 
by
  sorry

end computer_price_after_9_years_l108_108434


namespace average_of_roots_l108_108612

theorem average_of_roots (p q : ℝ) (h : ∃ x1 x2 : ℝ, 3*p*x1^2 - 6*p*x1 + q = 0 ∧ 3*p*x2^2 - 6*p*x2 + q = 0 ∧ x1 ≠ x2):
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 3*p*x1^2 - 6*p*x1 + q = 0 ∧ 3*p*x2^2 - 6*p*x2 + q = 0) → 
  (x1 + x2) / 2 = 1 :=
by
  sorry

end average_of_roots_l108_108612


namespace rolls_remaining_to_sell_l108_108956

-- Conditions
def total_rolls_needed : ℕ := 45
def rolls_sold_to_grandmother : ℕ := 1
def rolls_sold_to_uncle : ℕ := 10
def rolls_sold_to_neighbor : ℕ := 6

-- Theorem statement
theorem rolls_remaining_to_sell : (total_rolls_needed - (rolls_sold_to_grandmother + rolls_sold_to_uncle + rolls_sold_to_neighbor) = 28) :=
by
  sorry

end rolls_remaining_to_sell_l108_108956


namespace convex_polygons_from_fifteen_points_l108_108502

theorem convex_polygons_from_fifteen_points 
    (h : ∀ (n : ℕ), n = 15) :
    ∃ (k : ℕ), k = 32192 :=
by
  sorry

end convex_polygons_from_fifteen_points_l108_108502


namespace number_of_articles_l108_108989

theorem number_of_articles (C S : ℝ) (h_gain : S = 1.4285714285714286 * C) (h_cost : ∃ X : ℝ, X * C = 35 * S) : ∃ X : ℝ, X = 50 :=
by
  -- Define the specific existence and equality proof here
  sorry

end number_of_articles_l108_108989


namespace distance_from_origin_to_point_l108_108181

-- Define the specific points in the coordinate system
def origin : ℝ × ℝ := (0, 0)
def point : ℝ × ℝ := (8, -15)

-- The distance formula in Euclidean space
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- The theorem statement
theorem distance_from_origin_to_point : distance origin point = 17 := 
by
  sorry

end distance_from_origin_to_point_l108_108181


namespace translation_min_point_correct_l108_108438

-- Define the original equation
def original_eq (x : ℝ) := |x| - 5

-- Define the translation function
def translate_point (p : ℝ × ℝ) (tx ty : ℝ) : ℝ × ℝ := (p.1 + tx, p.2 + ty)

-- Define the minimum point of the original equation
def original_min_point : ℝ × ℝ := (0, original_eq 0)

-- Translate the original minimum point three units right and four units up
def new_min_point := translate_point original_min_point 3 4

-- Prove that the new minimum point is (3, -1)
theorem translation_min_point_correct : new_min_point = (3, -1) :=
by
  sorry

end translation_min_point_correct_l108_108438


namespace diamond_and_face_card_probability_l108_108239

noncomputable def probability_first_diamond_second_face_card : ℚ :=
  let total_cards := 52
  let total_faces := 12
  let diamond_faces := 3
  let diamond_non_faces := 10
  (9/52) * (12/51) + (3/52) * (11/51)

theorem diamond_and_face_card_probability :
  probability_first_diamond_second_face_card = 47 / 884 := 
by {
  sorry
}

end diamond_and_face_card_probability_l108_108239


namespace sqrt_mul_sqrt_l108_108243

theorem sqrt_mul_sqrt (h1 : Real.sqrt 25 = 5) : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_mul_sqrt_l108_108243


namespace scientists_speculation_reasonable_uranus_will_not_affect_earth_next_observation_l108_108576

-- Define the given conditions as constants and theorems in Lean
theorem scientists_speculation_reasonable : 
  ∃ (a b c : ℝ), 
  (64*a - 8*b + c = 32) ∧ 
  (36*a - 6*b + c = 28.5) ∧ 
  (16*a - 4*b + c = 26) ∧ 
  (∀ (x y : ℝ), (y = a*x^2 + b*x + c) → (x = 0) → y < 24.5) :=
by -- sorry is a placeholder for the proof
sorry

theorem uranus_will_not_affect_earth_next_observation : 
  ∃ (a b c : ℝ), 
  (64*a - 8*b + c = 32) ∧ 
  (36*a - 6*b + c = 28.5) ∧ 
  (16*a - 4*b + c = 26) ∧ 
  (∀ (x y : ℝ), (y = a*x^2 + b*x + c) → (x = 2) → y ≥ 24.5) :=
by -- sorry is a placeholder for the proof
sorry

end scientists_speculation_reasonable_uranus_will_not_affect_earth_next_observation_l108_108576


namespace fraction_of_water_l108_108481

/-- 
  Prove that the fraction of the mixture that is water is (\frac{2}{5}) 
  given the total weight of the mixture is 40 pounds, 
  1/4 of the mixture is sand, 
  and the remaining 14 pounds of the mixture is gravel. 
-/
theorem fraction_of_water 
  (total_weight : ℝ)
  (weight_sand : ℝ)
  (weight_gravel : ℝ)
  (weight_water : ℝ)
  (h1 : total_weight = 40)
  (h2 : weight_sand = (1/4) * total_weight)
  (h3 : weight_gravel = 14)
  (h4 : weight_water = total_weight - (weight_sand + weight_gravel)) :
  (weight_water / total_weight) = 2/5 :=
by
  sorry

end fraction_of_water_l108_108481


namespace find_c_l108_108986

theorem find_c (a c : ℤ) (h1 : 3 * a + 2 = 2) (h2 : c - a = 3) : c = 3 := by
  sorry

end find_c_l108_108986


namespace find_digit_D_l108_108582

theorem find_digit_D (A B C D : ℕ) (h1 : A + B = A + 10 * (B / 10)) (h2 : D + 10 * (A / 10) = A + C)
  (h3 : A + 10 * (B / 10) - C = A) (h4 : 0 ≤ A) (h5 : A ≤ 9) (h6 : 0 ≤ B) (h7 : B ≤ 9)
  (h8 : 0 ≤ C) (h9 : C ≤ 9) (h10 : 0 ≤ D) (h11 : D ≤ 9) : D = 9 := 
sorry

end find_digit_D_l108_108582


namespace integral_eval_l108_108386

theorem integral_eval : ∫ x in (1:ℝ)..(2:ℝ), (2*x + 1/x) = 3 + Real.log 2 := by
  sorry

end integral_eval_l108_108386


namespace find_angle_x_l108_108044

theorem find_angle_x (x : ℝ) (h1 : 3 * x + 2 * x = 90) : x = 18 :=
  by
    sorry

end find_angle_x_l108_108044


namespace distance_M_to_AB_l108_108724

noncomputable def distance_to_ab : ℝ := 5.8

theorem distance_M_to_AB
  (M : Point)
  (A B C : Point)
  (d_AC d_BC : ℝ)
  (AB BC AC : ℝ)
  (H1 : d_AC = 2)
  (H2 : d_BC = 4)
  (H3 : AB = 10)
  (H4 : BC = 17)
  (H5 : AC = 21) :
  distance_to_ab = 5.8 :=
by
  sorry

end distance_M_to_AB_l108_108724


namespace find_x_l108_108098

theorem find_x (x y : ℝ) (h₁ : 2 * x - y = 14) (h₂ : y = 2) : x = 8 :=
by
  sorry

end find_x_l108_108098


namespace percentage_of_hindu_boys_l108_108539

-- Define the total number of boys in the school
def total_boys := 700

-- Define the percentage of Muslim boys
def muslim_percentage := 44 / 100

-- Define the percentage of Sikh boys
def sikh_percentage := 10 / 100

-- Define the number of boys from other communities
def other_communities_boys := 126

-- State the main theorem to prove the percentage of Hindu boys
theorem percentage_of_hindu_boys (h1 : total_boys = 700)
                                 (h2 : muslim_percentage = 44 / 100)
                                 (h3 : sikh_percentage = 10 / 100)
                                 (h4 : other_communities_boys = 126) : 
                                 ((total_boys - (total_boys * muslim_percentage + total_boys * sikh_percentage + other_communities_boys)) / total_boys) * 100 = 28 :=
by {
  sorry
}

end percentage_of_hindu_boys_l108_108539


namespace weekly_earnings_l108_108406

-- Definition of the conditions
def hourly_rate : ℕ := 20
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 4

-- Theorem that conforms to the problem statement
theorem weekly_earnings : hourly_rate * hours_per_day * days_per_week = 640 := by
  sorry

end weekly_earnings_l108_108406


namespace equilateral_triangle_of_equal_inradii_l108_108440

theorem equilateral_triangle_of_equal_inradii
  (ABC : Triangle)
  (h_medians_divide : ∀ Δ ∈ (ABC.medians_divide), area Δ = (1/6) * area ABC)
  (h_four_inradii_equal : ∃ (Δ₁ Δ₂ Δ₃ Δ₄ : Triangle), (Δ₁ ∈ (ABC.medians_divide) ∧ Δ₂ ∈ (ABC.medians_divide) ∧ Δ₃ ∈ (ABC.medians_divide) ∧ Δ₄ ∈ (ABC.medians_divide)) ∧ (inradius Δ₁ = inradius Δ₂ ∧ inradius Δ₂ = inradius Δ₃ ∧ inradius Δ₃ = inradius Δ₄)) :
  is_equilateral ABC :=
sorry

end equilateral_triangle_of_equal_inradii_l108_108440


namespace hypotenuse_of_45_45_90_triangle_l108_108568

noncomputable def leg_length : ℝ := 15
noncomputable def angle_opposite_leg : ℝ := Real.pi / 4  -- 45 degrees in radians

theorem hypotenuse_of_45_45_90_triangle (h_leg : ℝ) (h_angle : ℝ) 
  (h_leg_cond : h_leg = leg_length) (h_angle_cond : h_angle = angle_opposite_leg) :
  ∃ h_hypotenuse : ℝ, h_hypotenuse = h_leg * Real.sqrt 2 :=
sorry

end hypotenuse_of_45_45_90_triangle_l108_108568


namespace dean_marathon_time_l108_108859

/-- 
Micah runs 2/3 times as fast as Dean, and it takes Jake 1/3 times more time to finish the marathon
than it takes Micah. The total time the three take to complete the marathon is 23 hours.
Prove that the time it takes Dean to finish the marathon is approximately 7.67 hours.
-/
theorem dean_marathon_time (D M J : ℝ)
  (h1 : M = D * (3 / 2))
  (h2 : J = M + (1 / 3) * M)
  (h3 : D + M + J = 23) : 
  D = 23 / 3 :=
by
  sorry

end dean_marathon_time_l108_108859


namespace leak_emptying_time_l108_108482

theorem leak_emptying_time (fill_rate_no_leak : ℝ) (combined_rate_with_leak : ℝ) (L : ℝ) :
  fill_rate_no_leak = 1/10 →
  combined_rate_with_leak = 1/12 →
  fill_rate_no_leak - L = combined_rate_with_leak →
  1 / L = 60 :=
by
  intros h1 h2 h3
  sorry

end leak_emptying_time_l108_108482


namespace largest_number_is_a_l108_108823

-- Define the numbers in their respective bases
def a := 8 * 9 + 5
def b := 3 * 5^2 + 0 * 5 + 1 * 5^0
def c := 1 * 2^3 + 0 * 2^2 + 0 * 2^1 + 1 * 2^0

theorem largest_number_is_a : a > b ∧ a > c :=
by
  -- These are the expected results, rest is the proof steps which we skip using sorry
  have ha : a = 77 := rfl
  have hb : b = 76 := rfl
  have hc : c = 9 := rfl
  sorry

end largest_number_is_a_l108_108823


namespace complementary_angles_l108_108156

theorem complementary_angles (angle1 angle2 : ℝ) (h1 : angle1 + angle2 = 90) (h2 : angle1 = 25) : angle2 = 65 :=
by 
  sorry

end complementary_angles_l108_108156


namespace area_of_ABCM_l108_108800

-- Definitions of the problem conditions
def length_of_sides (P : ℕ) := 4
def forms_right_angle (P : ℕ) := True
def M_intersection (AG CH : ℝ) := True

-- Proposition that quadrilateral ABCM has the correct area
theorem area_of_ABCM (a b c m : ℝ) :
  (length_of_sides 12 = 4) ∧
  (forms_right_angle 12) ∧
  (M_intersection a b) →
  ∃ area_ABCM : ℝ, area_ABCM = 88/5 :=
by
  sorry

end area_of_ABCM_l108_108800


namespace grocer_sales_l108_108107

theorem grocer_sales (sale1 sale2 sale3 sale4 sale5 sale6 : ℕ)
  (h1 : sale2 = 900)
  (h2 : sale3 = 1000)
  (h3 : sale4 = 700)
  (h4 : sale5 = 800)
  (h5 : sale6 = 900)
  (h6 : (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = 850) :
  sale1 = 800 :=
by
  sorry

end grocer_sales_l108_108107


namespace coat_price_reduction_l108_108102

variable (original_price reduction : ℝ)

theorem coat_price_reduction
  (h_orig : original_price = 500)
  (h_reduct : reduction = 350)
  : reduction / original_price * 100 = 70 := 
sorry

end coat_price_reduction_l108_108102


namespace least_overlap_coffee_tea_l108_108718

open BigOperators

-- Define the percentages in a way that's compatible in Lean
def percentage (n : ℕ) := n / 100

noncomputable def C := percentage 75
noncomputable def T := percentage 80
noncomputable def B := percentage 55

-- The theorem statement
theorem least_overlap_coffee_tea : C + T - 1 = B := sorry

end least_overlap_coffee_tea_l108_108718


namespace asian_games_volunteer_selection_l108_108400

-- Define the conditions.

def total_volunteers : ℕ := 5
def volunteer_A_cannot_serve_language_services : Prop := true

-- Define the main problem.
-- We are supposed to find the number of ways to assign three roles given the conditions.
def num_ways_to_assign_roles : ℕ :=
  let num_ways_language_services := 4 -- A cannot serve this role, so 4 choices
  let num_ways_other_roles := 4 * 3 -- We need to choose and arrange 2 volunteers out of remaining
  num_ways_language_services * num_ways_other_roles

-- The target theorem.
theorem asian_games_volunteer_selection : num_ways_to_assign_roles = 48 :=
by
  sorry

end asian_games_volunteer_selection_l108_108400


namespace smallest_n_2000_divides_a_n_l108_108197

theorem smallest_n_2000_divides_a_n (a : ℕ → ℤ) 
  (h_rec : ∀ n, n ≥ 1 → (n - 1) * a (n + 1) = (n + 1) * a n - 2 * (n - 1)) 
  (h2000 : 2000 ∣ a 1999) : 
  ∃ n, n ≥ 2 ∧ 2000 ∣ a n ∧ n = 249 := 
by 
  sorry

end smallest_n_2000_divides_a_n_l108_108197


namespace cylinder_radius_in_cone_l108_108109

theorem cylinder_radius_in_cone :
  ∀ (r : ℚ), (2 * r = r) → (0 < r) → (∀ (h : ℚ), h = 2 * r → 
  (∀ (c_r : ℚ), c_r = 4 (c_r is radius of cone)  ∧ (h_c : ℚ), h_c = 10 (h_c is height of cone) ∧ 
  (10 - h) / r = h_c / c_r) → r = 20 / 9) :=
begin
  sorry,
end

end cylinder_radius_in_cone_l108_108109


namespace solve_system_of_inequalities_l108_108430

theorem solve_system_of_inequalities (x y : ℤ) :
  (2 * x - y > 3 ∧ 3 - 2 * x + y > 0) ↔ (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) := 
by { sorry }

end solve_system_of_inequalities_l108_108430


namespace part1_real_roots_part2_specific_roots_l108_108667

-- Part 1: Real roots condition
theorem part1_real_roots (m : ℝ) (h : ∃ x : ℝ, x^2 + (2 * m - 1) * x + m^2 = 0) : m ≤ 1/4 :=
by sorry

-- Part 2: Specific roots condition
theorem part2_specific_roots (m : ℝ) (x1 x2 : ℝ) 
  (h1 : x1^2 + (2 * m - 1) * x1 + m^2 = 0) 
  (h2 : x2^2 + (2 * m - 1) * x2 + m^2 = 0) 
  (h3 : x1 * x2 + x1 + x2 = 4) : m = -1 :=
by sorry

end part1_real_roots_part2_specific_roots_l108_108667


namespace Grisha_probability_expected_flips_l108_108040

theorem Grisha_probability (h_even_heads : ∀ n, (n % 2 = 0 → coin_flip n = heads), 
                            t_odd_tails : ∀ n, (n % 2 = 1 → coin_flip n = tails)) : 
  (probability (Grisha wins)) = 1 / 3 :=
by 
  sorry

theorem expected_flips (h_even_heads : ∀ n, (n % 2 = 0 → coin_flip n = heads), 
                        t_odd_tails : ∀ n, (n % 2 = 1 → coin_flip n = tails)) : 
  (expected_flips) = 2 :=
by
  sorry

end Grisha_probability_expected_flips_l108_108040


namespace find_x_of_orthogonal_vectors_l108_108804

theorem find_x_of_orthogonal_vectors (x : ℝ) : 
  (⟨3, -4, 1⟩ : ℝ × ℝ × ℝ) • (⟨x, 2, -7⟩ : ℝ × ℝ × ℝ) = 0 → x = 5 := 
by
  sorry

end find_x_of_orthogonal_vectors_l108_108804


namespace smallest_k_mod_19_7_3_l108_108738

theorem smallest_k_mod_19_7_3 : ∃ k : ℕ, k > 1 ∧ (k % 19 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) ∧ k = 400 := 
by {
  -- statements of conditions in form of hypotheses
  let h1 := k > 1,
  let h2 := k % 19 = 1,
  let h3 := k % 7 = 1,
  let h4 := k % 3 = 1,
  -- goal of the theorem
  exact ⟨400, _⟩ sorry -- we indicate the goal should be of the form ⟨value, proof⟩, and fill in the proof with 'sorry'
}

end smallest_k_mod_19_7_3_l108_108738


namespace pairs_satisfying_x2_minus_y2_eq_45_l108_108678

theorem pairs_satisfying_x2_minus_y2_eq_45 :
  (∃ p : Finset (ℕ × ℕ), (∀ (x y : ℕ), ((x, y) ∈ p → x^2 - y^2 = 45) ∧ (∀ (x y : ℕ), (x, y) ∈ p → 0 < x ∧ 0 < y)) ∧ p.card = 3) :=
by
  sorry

end pairs_satisfying_x2_minus_y2_eq_45_l108_108678


namespace kids_played_on_tuesday_l108_108195

-- Definitions of the conditions
def kids_played_on_wednesday (julia : Type) : Nat := 4
def kids_played_on_monday (julia : Type) : Nat := 6
def difference_monday_wednesday (julia : Type) : Nat := 2

-- Define the statement to prove
theorem kids_played_on_tuesday (julia : Type) :
  (kids_played_on_monday julia - difference_monday_wednesday julia) = kids_played_on_wednesday julia :=
by
  sorry

end kids_played_on_tuesday_l108_108195


namespace regular_octagon_interior_angle_l108_108464

theorem regular_octagon_interior_angle : 
  (∀ (n : ℕ), n = 8 → ∀ (sum_of_interior_angles : ℕ), sum_of_interior_angles = (n - 2) * 180 → ∀ (each_angle : ℕ), each_angle = sum_of_interior_angles / n → each_angle = 135) :=
  sorry

end regular_octagon_interior_angle_l108_108464


namespace adjacent_books_probability_l108_108882

def chinese_books : ℕ := 2
def math_books : ℕ := 2
def physics_books : ℕ := 1
def total_books : ℕ := chinese_books + math_books + physics_books

theorem adjacent_books_probability :
  (total_books = 5) →
  (chinese_books = 2) →
  (math_books = 2) →
  (physics_books = 1) →
  (∃ p : ℚ, p = 1 / 5) :=
by
  intros h1 h2 h3 h4
  -- Proof omitted.
  exact ⟨1 / 5, rfl⟩

end adjacent_books_probability_l108_108882


namespace evaluate_powers_of_i_l108_108000

theorem evaluate_powers_of_i : (complex.I ^ 22 + complex.I ^ 222) = -2 :=
by
  -- Using by to start the proof block and ending it with sorry.
  sorry

end evaluate_powers_of_i_l108_108000


namespace sqrt_mul_sqrt_l108_108248

theorem sqrt_mul_sqrt (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : sqrt (a * sqrt b) = sqrt a * sqrt (sqrt b) := by
  sorry

example : sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  have h1 : sqrt 25 = 5 := by 
    norm_num
  have h2 : sqrt (49 * 5) = sqrt 245 := by  
    unfold sqrt
  have h3 : sqrt 245 = 7 * sqrt 5 := by 
    norm_num
  rw [h1, h2, h3]
  norm_num

end sqrt_mul_sqrt_l108_108248


namespace all_positive_rationals_are_red_l108_108124

-- Define the property of being red for rational numbers
def is_red (x : ℚ) : Prop :=
  ∃ n : ℕ, ∃ (f : ℕ → ℚ), f 0 = 1 ∧ (∀ m : ℕ, f (m + 1) = f m + 1 ∨ f (m + 1) = f m / (f m + 1)) ∧ f n = x

-- Proposition stating that all positive rational numbers are red
theorem all_positive_rationals_are_red :
  ∀ x : ℚ, 0 < x → is_red x :=
  by sorry

end all_positive_rationals_are_red_l108_108124


namespace percentage_error_calculation_l108_108618

theorem percentage_error_calculation (x : ℝ) :
  let correct_value := x * (5 / 3)
  let incorrect_value := x * (3 / 5)
  let difference := correct_value - incorrect_value
  let percentage_error := (difference / correct_value) * 100
  percentage_error = 64 := 
by
  let correct_value := x * (5 / 3)
  let incorrect_value := x * (3 / 5)
  let difference := correct_value - incorrect_value
  let percentage_error := (difference / correct_value) * 100
  sorry

end percentage_error_calculation_l108_108618


namespace grid_points_circumference_l108_108708

def numGridPointsOnCircumference (R : ℝ) : ℕ := sorry

def isInteger (x : ℝ) : Prop := ∃ (n : ℤ), x = n

theorem grid_points_circumference (R : ℝ) (h : numGridPointsOnCircumference R = 1988) : 
  isInteger R ∨ isInteger (Real.sqrt 2 * R) :=
by
  sorry

end grid_points_circumference_l108_108708


namespace silly_bills_count_l108_108475

theorem silly_bills_count (x : ℕ) (h1 : x + 2 * (x + 11) + 3 * (x - 18) = 100) : x = 22 :=
by { sorry }

end silly_bills_count_l108_108475


namespace max_value_of_expression_l108_108699

open Real

theorem max_value_of_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x + y + z = 1) : 
  x^4 * y^2 * z ≤ 1024 / 7^7 :=
sorry

end max_value_of_expression_l108_108699


namespace hypotenuse_of_454590_triangle_l108_108558

theorem hypotenuse_of_454590_triangle (l : ℝ) (angle : ℝ) (h : ℝ) (h_leg : l = 15) (h_angle : angle = 45) :
  h = l * Real.sqrt 2 := 
  sorry

end hypotenuse_of_454590_triangle_l108_108558


namespace count_digit_7_from_100_to_199_l108_108052

theorem count_digit_7_from_100_to_199 : 
  (list.range' 100 100).countp (λ n, (70 ≤ n ∧ n ≤ 79) ∨ (n % 10 = 7)) = 20 :=
by
  simp
  sorry

end count_digit_7_from_100_to_199_l108_108052


namespace part1_l108_108600

   noncomputable def sin_20_deg_sq : ℝ := (Real.sin (20 * Real.pi / 180))^2
   noncomputable def cos_80_deg_sq : ℝ := (Real.sin (10 * Real.pi / 180))^2
   noncomputable def sqrt3_sin20_cos80 : ℝ := Real.sqrt 3 * Real.sin (20 * Real.pi / 180) * Real.sin (10 * Real.pi / 180)
   noncomputable def value : ℝ := sin_20_deg_sq + cos_80_deg_sq + sqrt3_sin20_cos80

   theorem part1 : value = 1 / 4 := by
     sorry
   
end part1_l108_108600


namespace heptagon_angle_in_arithmetic_progression_l108_108578

theorem heptagon_angle_in_arithmetic_progression (a d : ℝ) :
  a + 3 * d = 128.57 → 
  (7 * a + 21 * d = 900) → 
  ∃ angle : ℝ, angle = 128.57 :=
by
  sorry

end heptagon_angle_in_arithmetic_progression_l108_108578


namespace factor_difference_of_squares_l108_108936

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) :=
by
  sorry

end factor_difference_of_squares_l108_108936


namespace train_length_l108_108598

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (length_m : ℕ) 
  (h1 : speed_kmh = 180)
  (h2 : time_s = 18)
  (h3 : 1 = 1000 / 3600) :
  length_m = (speed_kmh * 1000 / 3600) * time_s :=
by
  sorry

end train_length_l108_108598


namespace price_of_third_variety_l108_108864

theorem price_of_third_variety 
    (price1 price2 price3 : ℝ)
    (mix_ratio1 mix_ratio2 mix_ratio3 : ℝ)
    (mixture_price : ℝ)
    (h1 : price1 = 126)
    (h2 : price2 = 135)
    (h3 : mix_ratio1 = 1)
    (h4 : mix_ratio2 = 1)
    (h5 : mix_ratio3 = 2)
    (h6 : mixture_price = 153) :
    price3 = 175.5 :=
by
  sorry

end price_of_third_variety_l108_108864


namespace calculate_total_earnings_l108_108316

theorem calculate_total_earnings :
  let num_floors := 10
  let rooms_per_floor := 20
  let hours_per_room := 8
  let earnings_per_hour := 20
  let total_rooms := num_floors * rooms_per_floor
  let total_hours := total_rooms * hours_per_room
  let total_earnings := total_hours * earnings_per_hour
  total_earnings = 32000 := by sorry

end calculate_total_earnings_l108_108316


namespace ivanov_should_receive_12_l108_108766

variable (x : ℝ) -- price per car in million rubles
variable (iv : ℝ) -- Ivanov's monetary contribution to balance
variable (p : ℝ) -- Petrov's monetary contribution to balance

-- Define the given conditions as Lean hypotheses
variable (h_iv_cars : iv = 70 * x)
variable (h_p_cars : p = 40 * x)
variable (h_s_contrib : 44) -- Sidorov's contribution
variable (h_balance : (iv + p + 44) / 3 = 44)

-- The amount Ivanov is entitled to receive
def ivanov_gets_back : ℝ := iv - 44

-- The theorem to prove
theorem ivanov_should_receive_12 : ivanov_gets_back x iv p = 12 :=
by
  -- add proof here
  sorry

end ivanov_should_receive_12_l108_108766


namespace speed_of_boat_in_still_water_l108_108442

theorem speed_of_boat_in_still_water
    (speed_stream : ℝ)
    (distance_downstream : ℝ)
    (distance_upstream : ℝ)
    (t : ℝ)
    (x : ℝ)
    (h1 : speed_stream = 10)
    (h2 : distance_downstream = 80)
    (h3 : distance_upstream = 40)
    (h4 : t = distance_downstream / (x + speed_stream))
    (h5 : t = distance_upstream / (x - speed_stream)) :
  x = 30 :=
by sorry

end speed_of_boat_in_still_water_l108_108442


namespace sqrt_49_times_sqrt_25_eq_7sqrt5_l108_108260

theorem sqrt_49_times_sqrt_25_eq_7sqrt5 :
  (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  have h1 : Real.sqrt 25 = 5 := by sorry
  have h2 : 49 * 5 = 245 := by sorry
  have h3 : Real.sqrt 245 = 7 * Real.sqrt 5 := by sorry
  sorry

end sqrt_49_times_sqrt_25_eq_7sqrt5_l108_108260


namespace exist_integers_xy_divisible_by_p_l108_108698

theorem exist_integers_xy_divisible_by_p (p : ℕ) [Fact (Nat.Prime p)] : ∃ x y : ℤ, (x^2 + y^2 + 2) % p = 0 := by
  sorry

end exist_integers_xy_divisible_by_p_l108_108698


namespace find_number_l108_108778

-- Definition to calculate the sum of the digits of a number
def sumOfDigits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Problem statement
theorem find_number :
  ∃ n : ℕ, n > 0 ∧ (n * sumOfDigits n = 2008) ∧ (n = 251) :=
by
  use 251
  split
  · exact nat.succ_pos'
  split
  · show 251 * sumOfDigits 251 = 2008
    sorry
  · exact rfl

end find_number_l108_108778


namespace cindy_hit_section_8_l108_108996

inductive Player : Type
| Alice | Ben | Cindy | Dave | Ellen
deriving DecidableEq

structure DartContest :=
(player : Player)
(score : ℕ)

def ContestConditions (dc : DartContest) : Prop :=
  match dc with
  | ⟨Player.Alice, 10⟩ => True
  | ⟨Player.Ben, 6⟩ => True
  | ⟨Player.Cindy, 9⟩ => True
  | ⟨Player.Dave, 15⟩ => True
  | ⟨Player.Ellen, 19⟩ => True
  | _ => False

def isScoreSection8 (dc : DartContest) : Prop :=
  dc.player = Player.Cindy ∧ dc.score = 8

theorem cindy_hit_section_8 
  (cond : ∀ (dc : DartContest), ContestConditions dc) : 
  ∃ (dc : DartContest), isScoreSection8 dc := by
  sorry

end cindy_hit_section_8_l108_108996


namespace distance_from_origin_to_point_l108_108182

-- Define the specific points in the coordinate system
def origin : ℝ × ℝ := (0, 0)
def point : ℝ × ℝ := (8, -15)

-- The distance formula in Euclidean space
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- The theorem statement
theorem distance_from_origin_to_point : distance origin point = 17 := 
by
  sorry

end distance_from_origin_to_point_l108_108182


namespace geometric_sequence_sum_ratio_l108_108960

theorem geometric_sequence_sum_ratio (a_n : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h_nonzero_q : q ≠ 0) 
  (a2 : a_n 2 = a_n 1 * q) (a5 : a_n 5 = a_n 1 * q^4) 
  (h_condition : 8 * a_n 2 + a_n 5 = 0)
  (h_sum : ∀ n, S n = a_n 1 * (1 - q^n) / (1 - q)) : 
  S 5 / S 2 = -11 :=
by 
  sorry

end geometric_sequence_sum_ratio_l108_108960


namespace sandwiches_left_l108_108702

theorem sandwiches_left (S G K L : ℕ) (h1 : S = 20) (h2 : G = 4) (h3 : K = 2 * G) (h4 : L = S - G - K) : L = 8 :=
sorry

end sandwiches_left_l108_108702


namespace maximize_profit_l108_108322

noncomputable def profit (x : ℕ) : ℝ :=
  if x ≤ 200 then
    (0.40 - 0.24) * 30 * x
  else if x ≤ 300 then
    (0.40 - 0.24) * 10 * 200 + (0.40 - 0.24) * 20 * x - (0.24 - 0.08) * 10 * (x - 200)
  else
    (0.40 - 0.24) * 10 * 200 + (0.40 - 0.24) * 20 * 300 - (0.24 - 0.08) * 10 * (x - 200) - (0.24 - 0.08) * 20 * (x - 300)

theorem maximize_profit : ∀ x : ℕ, 
  profit 300 = 1120 ∧ (∀ y : ℕ, profit y ≤ 1120) :=
by
  sorry

end maximize_profit_l108_108322


namespace michelle_sandwiches_l108_108704

def sandwiches_left (total : ℕ) (given_to_coworker : ℕ) (kept : ℕ) : ℕ :=
  total - given_to_coworker - kept

theorem michelle_sandwiches : sandwiches_left 20 4 (4 * 2) = 8 :=
by
  sorry

end michelle_sandwiches_l108_108704


namespace smallest_fraction_gt_five_sevenths_l108_108786

theorem smallest_fraction_gt_five_sevenths (a b : ℕ) (h1 : 10 ≤ a ∧ a ≤ 99) (h2 : 10 ≤ b ∧ b ≤ 99) (h3 : 7 * a > 5 * b) : a = 68 ∧ b = 95 :=
sorry

end smallest_fraction_gt_five_sevenths_l108_108786


namespace factor_diff_of_squares_l108_108927

theorem factor_diff_of_squares (t : ℝ) : 
  t^2 - 64 = (t - 8) * (t + 8) :=
by
  -- The purpose here is to state the theorem only, without the proof.
  sorry

end factor_diff_of_squares_l108_108927


namespace gcd_lcm_condition_implies_divisibility_l108_108060

theorem gcd_lcm_condition_implies_divisibility
  (a b : ℤ) (h : Int.gcd a b + Int.lcm a b = a + b) : a ∣ b ∨ b ∣ a := 
sorry

end gcd_lcm_condition_implies_divisibility_l108_108060


namespace probability_of_yellow_ball_l108_108447

theorem probability_of_yellow_ball 
  (red_balls : ℕ) 
  (yellow_balls : ℕ) 
  (blue_balls : ℕ) 
  (total_balls : ℕ)
  (h1 : red_balls = 2)
  (h2 : yellow_balls = 5)
  (h3 : blue_balls = 4)
  (h4 : total_balls = red_balls + yellow_balls + blue_balls) :
  (yellow_balls / total_balls : ℚ) = 5 / 11 :=
by 
  rw [h1, h2, h3] at h4  -- Substitute the ball counts into the total_balls definition.
  norm_num at h4  -- Simplify to verify the total is indeed 11.
  rw [h2, h4] -- Use the number of yellow balls and total number of balls to state the ratio.
  norm_num -- Normalize the fraction to show it equals 5/11.

#check probability_of_yellow_ball

end probability_of_yellow_ball_l108_108447


namespace original_number_doubled_added_trebled_l108_108323

theorem original_number_doubled_added_trebled (x : ℤ) : 3 * (2 * x + 9) = 75 → x = 8 :=
by
  intro h
  -- The proof is omitted as instructed.
  sorry

end original_number_doubled_added_trebled_l108_108323


namespace arithmetic_geom_seq_a1_over_d_l108_108768

theorem arithmetic_geom_seq_a1_over_d (a1 a2 a3 a4 d : ℝ) (hne : d ≠ 0)
  (hgeom1 : (a1 + 2*d)^2 = a1 * (a1 + 3*d))
  (hgeom2 : (a1 + d)^2 = a1 * (a1 + 3*d)) :
  (a1 / d = -4) ∨ (a1 / d = 1) :=
by
  sorry

end arithmetic_geom_seq_a1_over_d_l108_108768


namespace problem2_l108_108397

noncomputable def problem1 (a b c : ℝ) (A B C : ℝ) (h1 : 2 * (Real.sin A)^2 + (Real.sin B)^2 = (Real.sin C)^2)
    (h2 : b = 2 * a) (h3 : a = 2) : (1 / 2) * a * b * Real.sin C = Real.sqrt 15 :=
by
  sorry

theorem problem2 (a b c : ℝ) (h : 2 * a^2 + b^2 = c^2) :
  ∃ m : ℝ, (m = 2 * Real.sqrt 2) ∧ (∀ x y z : ℝ, 2 * x^2 + y^2 = z^2 → (z^2 / (x * y)) ≥ m) ∧ ((c / a) = 2) :=
by
  sorry

end problem2_l108_108397


namespace sum_of_remainders_l108_108626

-- Definitions of the given problem
def a : ℕ := 1234567
def b : ℕ := 123

-- First remainder calculation
def r1 : ℕ := a % b

-- Second remainder calculation with the power
def r2 : ℕ := (2 ^ r1) % b

-- The proof statement
theorem sum_of_remainders : r1 + r2 = 29 := by
  sorry

end sum_of_remainders_l108_108626


namespace part1_part2_l108_108700

noncomputable def f (x a : ℝ) : ℝ := abs (x - 1) - 2 * abs (x + a)

theorem part1 (x : ℝ) : (∃ a, a = 1) → f x 1 > 1 ↔ -2 < x ∧ x < -(2/3) := by
  sorry

theorem part2 (a : ℝ) : (∀ x, 2 ≤ x → x ≤ 3 → f x a > 0) ↔ (-5/2) < a ∧ a < -2 := by
  sorry

end part1_part2_l108_108700


namespace circles_tangent_dist_l108_108518

theorem circles_tangent_dist (t : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 4) ∧ 
  (∀ x y : ℝ, (x - t)^2 + y^2 = 1) ∧ 
  (∀ x1 y1 x2 y2 : ℝ, x1^2 + y1^2 = 4 → (x2 - t)^2 + y2^2 = 1 → 
    dist (x1, y1) (x2, y2) = 3) → 
  t = 3 ∨ t = -3 :=
by 
  sorry

end circles_tangent_dist_l108_108518


namespace sqrt_49_times_sqrt25_eq_7_sqrt5_l108_108301

-- Definitions based on conditions
def sqrt_25 := Real.sqrt 25
def sqrt_49 := Real.sqrt 49

theorem sqrt_49_times_sqrt25_eq_7_sqrt5 :
  Real.sqrt (49 * sqrt_25) = 7 * Real.sqrt 5 := by
sorry

end sqrt_49_times_sqrt25_eq_7_sqrt5_l108_108301


namespace sakshi_days_l108_108212

theorem sakshi_days (Sakshi_efficiency Tanya_efficiency : ℝ) (Sakshi_days Tanya_days : ℝ) (h_efficiency : Tanya_efficiency = 1.25 * Sakshi_efficiency) (h_days : Tanya_days = 8) : Sakshi_days = 10 :=
by
  sorry

end sakshi_days_l108_108212


namespace sqrt_49_mul_sqrt_25_l108_108283

theorem sqrt_49_mul_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_l108_108283


namespace roots_value_l108_108017

theorem roots_value (m n : ℝ) (h1 : Polynomial.eval m (Polynomial.C 1 + Polynomial.C 3 * Polynomial.X + Polynomial.X ^ 2) = 0) (h2 : Polynomial.eval n (Polynomial.C 1 + Polynomial.C 3 * Polynomial.X + Polynomial.X ^ 2) = 0) : m^2 + 4 * m + n = -2 := 
sorry

end roots_value_l108_108017


namespace correct_option_l108_108423

-- Definitions for universe set, and subsets A and B
def S : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 5}

-- The proof goal
theorem correct_option : A ⊆ S \ B :=
by
  sorry

end correct_option_l108_108423


namespace polyhedron_has_circumscribed_sphere_l108_108580

theorem polyhedron_has_circumscribed_sphere
  (A B C D S C1 B1 D1 : Point)
  (h_plane : plane_contains_point A (perpendicular_plane S C))
  (h_intersect_SC : h_plane ∩ SC = {C1})
  (h_intersect_SB : h_plane ∩ SB = {B1})
  (h_intersect_SD : h_plane ∩ SD = {D1})
  (h_base : rectangle A B C D)
  (h_perpendicular : perpendicular S (plane_of_rectangle A B C D)) :
  ∃ O : Point, circumscribed_sphere (polyhedron A B C D B1 C1 D1) O := sorry

end polyhedron_has_circumscribed_sphere_l108_108580


namespace angles_of_triangle_l108_108870

theorem angles_of_triangle (a b c m_a m_b : ℝ) (h1 : m_a ≥ a) (h2 : m_b ≥ b) : 
  ∃ (α β γ : ℝ), ∀ t, 
  (t = 90) ∧ (α = 45) ∧ (β = 45) := 
sorry

end angles_of_triangle_l108_108870


namespace find_k_l108_108155

theorem find_k (a b k : ℝ) (h1 : 2^a = k) (h2 : 3^b = k) (hk : k ≠ 1) (h3 : 2 * a + b = a * b) : 
  k = 18 :=
sorry

end find_k_l108_108155


namespace range_of_mu_l108_108817

theorem range_of_mu (a b μ : ℝ) (ha : 0 < a) (hb : 0 < b) (hμ : 0 < μ) (h : 1 / a + 9 / b = 1) : μ ≤ 16 :=
by
  sorry

end range_of_mu_l108_108817


namespace challenge_Jane_l108_108075

def is_vowel (c : Char) : Prop :=
  c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U'

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def card_pairs : List (Char ⊕ ℕ) :=
  [Sum.inl 'A', Sum.inl 'T', Sum.inl 'U', Sum.inr 5, Sum.inr 8, Sum.inr 10, Sum.inr 14]

def Jane_claim (c : Char ⊕ ℕ) : Prop :=
  match c with
  | Sum.inl v => is_vowel v → ∃ n, Sum.inr n ∈ card_pairs ∧ is_even n
  | Sum.inr n => false

theorem challenge_Jane (cards : List (Char ⊕ ℕ)) (h : card_pairs = cards) :
  ∃ c ∈ cards, c = Sum.inr 5 ∧ ¬Jane_claim (Sum.inr 5) :=
sorry

end challenge_Jane_l108_108075


namespace count_4_letter_words_with_A_l108_108380

-- Define the alphabet set and the properties
def Alphabet : Finset (Char) := {'A', 'B', 'C', 'D', 'E'}
def total_words := (Alphabet.card ^ 4 : ℕ)
def total_words_without_A := (Alphabet.erase 'A').card ^ 4
def total_words_with_A := total_words - total_words_without_A

-- The key theorem to prove
theorem count_4_letter_words_with_A : total_words_with_A = 369 := sorry

end count_4_letter_words_with_A_l108_108380


namespace total_salaries_l108_108584

variable (A_salary B_salary : ℝ)

def A_saves : ℝ := 0.05 * A_salary
def B_saves : ℝ := 0.15 * B_salary

theorem total_salaries (h1 : A_salary = 5250) 
                       (h2 : A_saves = B_saves) : 
    A_salary + B_salary = 7000 := by
  sorry

end total_salaries_l108_108584


namespace blue_square_area_percentage_l108_108332

theorem blue_square_area_percentage (k : ℝ) (H1 : 0 < k) 
(Flag_area : ℝ := k^2) -- total area of the flag
(Cross_area : ℝ := 0.49 * Flag_area) -- total area of the cross and blue squares 
(one_blue_square_area : ℝ := Cross_area / 3) -- area of one blue square
(percentage : ℝ := one_blue_square_area / Flag_area * 100) :
percentage = 16.33 :=
by
  sorry

end blue_square_area_percentage_l108_108332


namespace weight_of_each_bag_of_planks_is_14_l108_108633

-- Definitions
def crate_capacity : Nat := 20
def num_crates : Nat := 15
def num_bags_nails : Nat := 4
def weight_bag_nails : Nat := 5
def num_bags_hammers : Nat := 12
def weight_bag_hammers : Nat := 5
def num_bags_planks : Nat := 10
def weight_to_leave_out : Nat := 80

-- Total weight calculations
def weight_nails := num_bags_nails * weight_bag_nails
def weight_hammers := num_bags_hammers * weight_bag_hammers
def total_weight_nails_hammers := weight_nails + weight_hammers
def total_crate_capacity := num_crates * crate_capacity
def weight_that_can_be_loaded := total_crate_capacity - weight_to_leave_out
def weight_available_for_planks := weight_that_can_be_loaded - total_weight_nails_hammers
def weight_each_bag_planks := weight_available_for_planks / num_bags_planks

-- Theorem statement
theorem weight_of_each_bag_of_planks_is_14 : weight_each_bag_planks = 14 :=
by {
  sorry
}

end weight_of_each_bag_of_planks_is_14_l108_108633


namespace prime_divides_a_minus_3_l108_108198

theorem prime_divides_a_minus_3 (a p : ℤ) (hp : Prime p) (h1 : p ∣ 5 * a - 1) (h2 : p ∣ a - 10) : p ∣ a - 3 := by
  sorry

end prime_divides_a_minus_3_l108_108198


namespace jose_profit_share_correct_l108_108089

-- Definitions for the conditions
def tom_investment : ℕ := 30000
def tom_months : ℕ := 12
def jose_investment : ℕ := 45000
def jose_months : ℕ := 10
def total_profit : ℕ := 36000

-- Capital months calculations
def tom_capital_months : ℕ := tom_investment * tom_months
def jose_capital_months : ℕ := jose_investment * jose_months
def total_capital_months : ℕ := tom_capital_months + jose_capital_months

-- Jose's share of the profit calculation
def jose_share_of_profit : ℕ := (jose_capital_months * total_profit) / total_capital_months

-- The theorem to prove
theorem jose_profit_share_correct : jose_share_of_profit = 20000 := by
  -- This is where the proof steps would go
  sorry

end jose_profit_share_correct_l108_108089


namespace find_values_l108_108477

variable (circle triangle : ℕ)

axiom condition1 : triangle = circle + circle + circle
axiom condition2 : triangle + circle = 40

theorem find_values : circle = 10 ∧ triangle = 30 :=
by
  sorry

end find_values_l108_108477


namespace cubic_expression_l108_108387

theorem cubic_expression (a b c : ℝ) (h1 : a + b + c = 15) (h2 : ab + ac + bc = 50) : 
  a^3 + b^3 + c^3 - 3 * a * b * c = 1125 :=
sorry

end cubic_expression_l108_108387


namespace distance_traveled_by_car_l108_108334

theorem distance_traveled_by_car (total_distance : ℕ) (fraction_foot : ℚ) (fraction_bus : ℚ)
  (h_total : total_distance = 40) (h_fraction_foot : fraction_foot = 1/4)
  (h_fraction_bus : fraction_bus = 1/2) :
  (total_distance * (1 - fraction_foot - fraction_bus)) = 10 :=
by
  sorry

end distance_traveled_by_car_l108_108334


namespace square_root_value_l108_108099

-- Define the problem conditions
def x : ℝ := 5

-- Prove the solution
theorem square_root_value : (Real.sqrt (x - 3)) = Real.sqrt 2 :=
by
  -- Proof steps skipped
  sorry

end square_root_value_l108_108099


namespace problem_solution_l108_108385

noncomputable def positiveIntPairsCount : ℕ :=
  sorry

theorem problem_solution :
  positiveIntPairsCount = 2 :=
sorry

end problem_solution_l108_108385


namespace quadratic_distinct_real_roots_iff_l108_108839

theorem quadratic_distinct_real_roots_iff (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (∀ (z : ℝ), z^2 - 2 * (m - 2) * z + m^2 = (z - x) * (z - y))) ↔ m < 1 :=
by
  sorry

end quadratic_distinct_real_roots_iff_l108_108839


namespace intersection_sets_l108_108516

-- Define set A as all x such that x >= -2
def setA : Set ℝ := {x | x >= -2}

-- Define set B as all x such that x < 1
def setB : Set ℝ := {x | x < 1}

-- The statement to prove in Lean 4
theorem intersection_sets : (setA ∩ setB) = {x | -2 <= x ∧ x < 1} :=
by
  sorry

end intersection_sets_l108_108516


namespace inequality_solution_l108_108583

theorem inequality_solution (x : ℝ) : (5 * x + 3 > 9 - 3 * x ∧ x ≠ 3) ↔ (x > 3 / 4 ∧ x ≠ 3) :=
by {
  sorry
}

end inequality_solution_l108_108583


namespace sugar_recipes_l108_108606

theorem sugar_recipes (container_sugar recipe_sugar : ℚ) 
  (h1 : container_sugar = 56 / 3) 
  (h2 : recipe_sugar = 3 / 2) :
  container_sugar / recipe_sugar = 112 / 9 := sorry

end sugar_recipes_l108_108606


namespace inequality_solution_l108_108002

open Set Real

theorem inequality_solution (x : ℝ) :
  (1 / (x + 1) + 3 / (x + 7) ≥ 2 / 3) ↔ (x ∈ Ioo (-7 : ℝ) (-4) ∪ Ioo (-1) (2) ∪ {(-4 : ℝ), 2}) :=
by sorry

end inequality_solution_l108_108002


namespace sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l108_108289

theorem sqrt_49_mul_sqrt_25_eq_7_sqrt_5 : (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l108_108289


namespace polygon_sides_sum_l108_108079

theorem polygon_sides_sum
  (area_ABCDEF : ℕ) (AB BC FA DE EF : ℕ)
  (h1 : area_ABCDEF = 78)
  (h2 : AB = 10)
  (h3 : BC = 11)
  (h4 : FA = 7)
  (h5 : DE = 4)
  (h6 : EF = 8) :
  DE + EF = 12 := 
by
  sorry

end polygon_sides_sum_l108_108079


namespace trig_expression_simplification_l108_108647

theorem trig_expression_simplification (α : Real) :
  Real.cos (3/2 * Real.pi + 4 * α)
  + Real.sin (3 * Real.pi - 8 * α)
  - Real.sin (4 * Real.pi - 12 * α)
  = 4 * Real.cos (2 * α) * Real.cos (4 * α) * Real.sin (6 * α) :=
sorry

end trig_expression_simplification_l108_108647


namespace expected_number_of_digits_is_1_55_l108_108121

/-- Brent rolls a fair icosahedral die with numbers 1 through 20 on its faces -/
noncomputable def expectedNumberOfDigits : ℚ :=
  let P_one_digit := 9 / 20
  let P_two_digit := 11 / 20
  (P_one_digit * 1) + (P_two_digit * 2)

/-- The expected number of digits Brent will roll is 1.55 -/
theorem expected_number_of_digits_is_1_55 : expectedNumberOfDigits = 1.55 := by
  sorry

end expected_number_of_digits_is_1_55_l108_108121


namespace find_natural_number_l108_108008

theorem find_natural_number (n : ℕ) (h1 : ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n ∨ d = 3 ∨ d = 5 ∨ d = 9 ∨ d = 15)
  (h2 : 1 + 3 + 5 + 9 + 15 + n = 78) : n = 45 := sorry

end find_natural_number_l108_108008


namespace arrangement_count_equivalent_problem_l108_108587

noncomputable def number_of_unique_arrangements : Nat :=
  let n : Nat := 6 -- Number of balls and boxes
  let match_3_boxes_ways := Nat.choose n 3 -- Choosing 3 boxes out of 6
  let permute_remaining_boxes := 2 -- Permutations of the remaining 3 boxes such that no numbers match
  match_3_boxes_ways * permute_remaining_boxes

theorem arrangement_count_equivalent_problem :
  number_of_unique_arrangements = 40 := by
  sorry

end arrangement_count_equivalent_problem_l108_108587


namespace sum_two_consecutive_sum_three_consecutive_sum_five_consecutive_sum_six_consecutive_l108_108894

theorem sum_two_consecutive : ∃ x : ℕ, 75 = x + (x + 1) := by
  sorry

theorem sum_three_consecutive : ∃ x : ℕ, 75 = x + (x + 1) + (x + 2) := by
  sorry

theorem sum_five_consecutive : ∃ x : ℕ, 75 = x + (x + 1) + (x + 2) + (x + 3) + (x + 4) := by
  sorry

theorem sum_six_consecutive : ∃ x : ℕ, 75 = x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) := by
  sorry

end sum_two_consecutive_sum_three_consecutive_sum_five_consecutive_sum_six_consecutive_l108_108894


namespace positive_difference_of_two_numbers_l108_108878

theorem positive_difference_of_two_numbers 
  (x y : ℝ) 
  (h1 : x + y = 10) 
  (h2 : x^2 - y^2 = 24) : 
  |x - y| = 2.4 := 
sorry

end positive_difference_of_two_numbers_l108_108878


namespace quadratic_has_two_distinct_roots_l108_108150

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem quadratic_has_two_distinct_roots 
  (a b c : ℝ) (h : 2016 + a^2 + a * c < a * b) : 
  discriminant a b c > 0 :=
sorry

end quadratic_has_two_distinct_roots_l108_108150


namespace least_seven_digit_binary_number_l108_108750

theorem least_seven_digit_binary_number : ∃ n : ℕ, (nat.binary_digits n = 7) ∧ (n = 64) := by
  sorry

end least_seven_digit_binary_number_l108_108750


namespace number_tower_proof_l108_108981

theorem number_tower_proof : 123456 * 9 + 7 = 1111111 := 
  sorry

end number_tower_proof_l108_108981


namespace intersection_M_N_l108_108015

def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x^2 + 2*x - 3 ≤ 0}

theorem intersection_M_N :
  M ∩ N = {-2, -1, 0, 1} :=
by
  sorry

end intersection_M_N_l108_108015


namespace brody_battery_life_left_l108_108793

-- Define the conditions
def full_battery_life : ℕ := 60
def used_fraction : ℚ := 3 / 4
def exam_duration : ℕ := 2

-- The proof statement
theorem brody_battery_life_left :
  let remaining_battery_initial := full_battery_life * (1 - used_fraction).toRat
  let remaining_battery := remaining_battery_initial - exam_duration
  remaining_battery = 13 := 
by
  sorry

end brody_battery_life_left_l108_108793


namespace smallest_n_l108_108465

theorem smallest_n (n : ℕ) (h1 : ∃ k : ℕ, 3 * n = k ^ 2) (h2 : ∃ m : ℕ, 5 * n = m ^ 5) : n = 151875 := sorry

end smallest_n_l108_108465


namespace larger_of_two_numbers_l108_108763

theorem larger_of_two_numbers (hcf : ℕ) (f1 : ℕ) (f2 : ℕ) 
(h_hcf : hcf = 10) 
(h_f1 : f1 = 11) 
(h_f2 : f2 = 15) 
: max (hcf * f1) (hcf * f2) = 150 :=
by
  have lcm := hcf * f1 * f2
  have num1 := hcf * f1
  have num2 := hcf * f2
  sorry

end larger_of_two_numbers_l108_108763


namespace larger_root_eq_5_over_8_l108_108664

noncomputable def find_larger_root : ℝ := 
    let x := ((5:ℝ) / 8)
    let y := ((23:ℝ) / 48)
    if x > y then x else y

theorem larger_root_eq_5_over_8 (x : ℝ) (y : ℝ) : 
  (x - ((5:ℝ) / 8)) * (x - ((5:ℝ) / 8)) + (x - ((5:ℝ) / 8)) * (x - ((1:ℝ) / 3)) = 0 → 
  find_larger_root = ((5:ℝ) / 8) :=
by
  intro h
  -- proof goes here
  sorry

end larger_root_eq_5_over_8_l108_108664


namespace tan_theta_parallel_l108_108677

theorem tan_theta_parallel (θ : ℝ) : 
  let a := (2, 3)
  let b := (Real.cos θ, Real.sin θ)
  (b.1 * a.2 = b.2 * a.1) → Real.tan θ = 3 / 2 :=
by
  intros h
  sorry

end tan_theta_parallel_l108_108677


namespace student_weight_l108_108471

variable (S W : ℕ)

theorem student_weight (h1 : S - 5 = 2 * W) (h2 : S + W = 110) : S = 75 :=
by
  sorry

end student_weight_l108_108471


namespace minimal_guests_l108_108903

-- Problem statement: For 120 chairs arranged in a circle,
-- determine the smallest number of guests (N) needed 
-- so that any additional guest must sit next to an already seated guest.

theorem minimal_guests (N : ℕ) : 
  (∀ (chairs : ℕ), chairs = 120 → 
    ∃ (N : ℕ), N = 20 ∧ 
      (∀ (new_guest : ℕ), new_guest + chairs = 120 → 
        new_guest ≤ N + 1 ∧ new_guest ≤ N - 1)) :=
by
  sorry

end minimal_guests_l108_108903


namespace part1_real_roots_part2_specific_roots_l108_108668

-- Part 1: Real roots condition
theorem part1_real_roots (m : ℝ) (h : ∃ x : ℝ, x^2 + (2 * m - 1) * x + m^2 = 0) : m ≤ 1/4 :=
by sorry

-- Part 2: Specific roots condition
theorem part2_specific_roots (m : ℝ) (x1 x2 : ℝ) 
  (h1 : x1^2 + (2 * m - 1) * x1 + m^2 = 0) 
  (h2 : x2^2 + (2 * m - 1) * x2 + m^2 = 0) 
  (h3 : x1 * x2 + x1 + x2 = 4) : m = -1 :=
by sorry

end part1_real_roots_part2_specific_roots_l108_108668


namespace cost_price_per_meter_l108_108762

-- Definitions based on the conditions given in the problem
def meters_of_cloth : ℕ := 45
def selling_price : ℕ := 4500
def profit_per_meter : ℕ := 12

-- Statement to prove
theorem cost_price_per_meter :
  (selling_price - (profit_per_meter * meters_of_cloth)) / meters_of_cloth = 88 :=
by
  sorry

end cost_price_per_meter_l108_108762


namespace arithmetic_sequence_sum_19_l108_108200

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_19 (h1 : is_arithmetic_sequence a)
  (h2 : a 9 = 11) (h3 : a 11 = 9) (h4 : ∀ n, S n = n / 2 * (a 1 + a n)) :
  S 19 = 190 :=
sorry

end arithmetic_sequence_sum_19_l108_108200


namespace tetrahedron_edges_sum_of_squares_l108_108426

-- Given conditions
variables {a b c d e f x y z : ℝ}

-- Mathematical statement
theorem tetrahedron_edges_sum_of_squares :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 4 * (x^2 + y^2 + z^2) :=
sorry

end tetrahedron_edges_sum_of_squares_l108_108426


namespace hypotenuse_of_45_45_90_triangle_l108_108560

theorem hypotenuse_of_45_45_90_triangle (a : ℝ) (h : ℝ) 
  (ha : a = 15) 
  (angle_opposite_leg : ℝ) 
  (h_angle : angle_opposite_leg = 45) 
  (right_triangle : ∃ θ : ℝ, θ = 90) : 
  h = 15 * Real.sqrt 2 := 
sorry

end hypotenuse_of_45_45_90_triangle_l108_108560


namespace solve_for_s_l108_108173

theorem solve_for_s (k s : ℝ) 
  (h1 : 7 = k * 3^s) 
  (h2 : 126 = k * 9^s) : 
  s = 2 + Real.log 2 / Real.log 3 := by
  sorry

end solve_for_s_l108_108173


namespace fraction_of_girls_on_trip_l108_108842

variable {g b : ℚ}

theorem fraction_of_girls_on_trip (h : g = b) (hg_trip : g_trip = (3/5) * g) (hb_trip : b_trip = (3/4) * b) :
  let total_trip := g_trip + b_trip in
  (g_trip / total_trip) = 4 / 9 :=
by sorry

end fraction_of_girls_on_trip_l108_108842


namespace sqrt_mul_sqrt_l108_108258

theorem sqrt_mul_sqrt (a b c : ℝ) (hac : a = 49) (hbc : b = sqrt 25) (hc : c = 7 * sqrt 5) : sqrt (a * b) = c := 
by
  sorry

end sqrt_mul_sqrt_l108_108258


namespace percentage_calculation_l108_108028

theorem percentage_calculation : 
  (0.8 * 90) = ((P / 100) * 60.00000000000001 + 30) → P = 70 := by
  sorry

end percentage_calculation_l108_108028


namespace polyhedron_space_diagonals_l108_108772

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

end polyhedron_space_diagonals_l108_108772


namespace perp_lines_of_parallel_planes_l108_108821

variables {Line Plane : Type} 
variables (m n : Line) (α β : Plane)
variable (is_parallel : Line → Plane → Prop)
variable (is_perpendicular : Line → Plane → Prop)
variable (planes_parallel : Plane → Plane → Prop)
variable (lines_perpendicular : Line → Line → Prop)

-- Given Conditions
variables (h1 : planes_parallel α β) (h2 : is_perpendicular m α) (h3 : is_parallel n β)

-- Prove that
theorem perp_lines_of_parallel_planes (h1 : planes_parallel α β) (h2 : is_perpendicular m α) (h3 : is_parallel n β) : lines_perpendicular m n := 
sorry

end perp_lines_of_parallel_planes_l108_108821


namespace milk_water_equal_l108_108450

theorem milk_water_equal (a : ℕ) :
  let glass_a_initial := a
  let glass_b_initial := a
  let mixture_in_a := glass_a_initial + 1
  let milk_portion_in_a := 1 / mixture_in_a
  let water_portion_in_a := glass_a_initial / mixture_in_a
  let water_in_milk_glass := water_portion_in_a
  let milk_in_water_glass := milk_portion_in_a
  water_in_milk_glass = milk_in_water_glass := by
  sorry

end milk_water_equal_l108_108450


namespace sqrt_mul_sqrt_l108_108251

theorem sqrt_mul_sqrt (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : sqrt (a * sqrt b) = sqrt a * sqrt (sqrt b) := by
  sorry

example : sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  have h1 : sqrt 25 = 5 := by 
    norm_num
  have h2 : sqrt (49 * 5) = sqrt 245 := by  
    unfold sqrt
  have h3 : sqrt 245 = 7 * sqrt 5 := by 
    norm_num
  rw [h1, h2, h3]
  norm_num

end sqrt_mul_sqrt_l108_108251


namespace factor_expression_l108_108501

theorem factor_expression (y : ℝ) : 
  3 * y * (2 * y + 5) + 4 * (2 * y + 5) = (3 * y + 4) * (2 * y + 5) :=
by
  sorry

end factor_expression_l108_108501


namespace minimize_expr_l108_108807

theorem minimize_expr : ∃ c : ℝ, (∀ d : ℝ, (3/4 * c^2 - 9 * c + 5) ≤ (3/4 * d^2 - 9 * d + 5)) ∧ c = 6 :=
by
  use 6
  sorry

end minimize_expr_l108_108807


namespace find_original_prices_and_discount_l108_108100

theorem find_original_prices_and_discount :
  ∃ x y a : ℝ,
  (6 * x + 5 * y = 1140) ∧
  (3 * x + 7 * y = 1110) ∧
  (((9 * x + 8 * y) - 1062) / (9 * x + 8 * y) = a) ∧
  x = 90 ∧
  y = 120 ∧
  a = 0.4 :=
by
  sorry

end find_original_prices_and_discount_l108_108100


namespace functional_equation_solution_l108_108809

theorem functional_equation_solution:
  (∀ f : ℝ → ℝ, (∀ x y z : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x * y * z = 1 →
  f x ^ 2 - f y * f z = x * (x + y + z) * (f x + f y + f z)) →
  (∀ x : ℝ, x ≠ 0 → ( (f x = x^2 - 1/x) ∨ (f x = 0)))) :=
by
  sorry

end functional_equation_solution_l108_108809


namespace min_selling_price_l108_108615

-- Average sales per month
def avg_sales := 50

-- Cost per refrigerator
def cost_per_fridge := 1200

-- Shipping fee per refrigerator
def shipping_fee_per_fridge := 20

-- Monthly storefront fee
def monthly_storefront_fee := 10000

-- Monthly repair costs
def monthly_repair_costs := 5000

-- Profit margin requirement
def profit_margin := 0.2

-- The minimum selling price for the shop to maintain at least 20% profit margin
theorem min_selling_price 
  (avg_sales : ℕ) 
  (cost_per_fridge : ℕ) 
  (shipping_fee_per_fridge : ℕ) 
  (monthly_storefront_fee : ℕ) 
  (monthly_repair_costs : ℕ) 
  (profit_margin : ℝ) : 
  ∃ x : ℝ, 
    (50 * x - ((cost_per_fridge + shipping_fee_per_fridge) * avg_sales + monthly_storefront_fee + monthly_repair_costs)) 
    ≥ (cost_per_fridge + shipping_fee_per_fridge) * avg_sales + monthly_storefront_fee + monthly_repair_costs * profit_margin 
    → x ≥ 1824 :=
by 
  sorry

end min_selling_price_l108_108615


namespace geometric_sequence_general_formula_l108_108401

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ a1 q : ℝ, ∀ n : ℕ, a n = a1 * q ^ (n - 1)

variables (a : ℕ → ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := a 1 + a 3 = 10
def condition2 : Prop := a 4 + a 6 = 5 / 4

-- The final statement to prove
theorem geometric_sequence_general_formula (h : geometric_sequence a) (h1 : condition1 a) (h2 : condition2 a) :
  ∀ n : ℕ, a n = 2 ^ (4 - n) :=
sorry

end geometric_sequence_general_formula_l108_108401


namespace smallest_k_mod_19_7_3_l108_108736

theorem smallest_k_mod_19_7_3 : ∃ k : ℕ, k > 1 ∧ (k % 19 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) ∧ k = 400 := 
by {
  -- statements of conditions in form of hypotheses
  let h1 := k > 1,
  let h2 := k % 19 = 1,
  let h3 := k % 7 = 1,
  let h4 := k % 3 = 1,
  -- goal of the theorem
  exact ⟨400, _⟩ sorry -- we indicate the goal should be of the form ⟨value, proof⟩, and fill in the proof with 'sorry'
}

end smallest_k_mod_19_7_3_l108_108736


namespace hyperbola_symmetric_asymptotes_l108_108675

noncomputable def M : ℝ := 225 / 16

theorem hyperbola_symmetric_asymptotes (M_val : ℝ) :
  (∀ x y : ℝ, (x^2 / 9 - y^2 / 16 = 1 → y = x * (4 / 3) ∨ y = -x * (4 / 3))
  ∧ (y^2 / 25 - x^2 / M_val = 1 → y = x * (5 / Real.sqrt M_val) ∨ y = -x * (5 / Real.sqrt M_val)))
  → M_val = M := by
  sorry

end hyperbola_symmetric_asymptotes_l108_108675


namespace probability_of_yellow_ball_l108_108446

theorem probability_of_yellow_ball 
  (red_balls : ℕ) 
  (yellow_balls : ℕ) 
  (blue_balls : ℕ) 
  (total_balls : ℕ)
  (h1 : red_balls = 2)
  (h2 : yellow_balls = 5)
  (h3 : blue_balls = 4)
  (h4 : total_balls = red_balls + yellow_balls + blue_balls) :
  (yellow_balls / total_balls : ℚ) = 5 / 11 :=
by 
  rw [h1, h2, h3] at h4  -- Substitute the ball counts into the total_balls definition.
  norm_num at h4  -- Simplify to verify the total is indeed 11.
  rw [h2, h4] -- Use the number of yellow balls and total number of balls to state the ratio.
  norm_num -- Normalize the fraction to show it equals 5/11.

#check probability_of_yellow_ball

end probability_of_yellow_ball_l108_108446


namespace geometric_sequence_example_l108_108683

theorem geometric_sequence_example
  (a : ℕ → ℝ)
  (h1 : ∀ n, 0 < a n)
  (h2 : ∃ r, ∀ n, a (n + 1) = r * a n)
  (h3 : Real.log (a 2) / Real.log 2 + Real.log (a 8) / Real.log 2 = 1) :
  a 3 * a 7 = 2 :=
sorry

end geometric_sequence_example_l108_108683


namespace solve_diff_l108_108071

-- Definitions based on conditions
def equation (e y : ℝ) : Prop := y^2 + e^2 = 3 * e * y + 1

theorem solve_diff (e a b : ℝ) (h1 : equation e a) (h2 : equation e b) (h3 : a ≠ b) : 
  |a - b| = Real.sqrt (5 * e^2 + 4) := 
sorry

end solve_diff_l108_108071


namespace sum_smallest_largest_2y_l108_108818

variable (a n y : ℤ)

noncomputable def is_even (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k
noncomputable def is_odd (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k + 1

theorem sum_smallest_largest_2y 
  (h1 : is_odd a) 
  (h2 : n % 2 = 0) 
  (h3 : y = a + n) : 
  a + (a + 2 * n) = 2 * y := 
by 
  sorry

end sum_smallest_largest_2y_l108_108818


namespace Total_toys_l108_108623

-- Definitions from the conditions
def Mandy_toys : ℕ := 20
def Anna_toys : ℕ := 3 * Mandy_toys
def Amanda_toys : ℕ := Anna_toys + 2

-- The statement to be proven
theorem Total_toys : Mandy_toys + Anna_toys + Amanda_toys = 142 :=
by
  -- Add proof here
  sorry

end Total_toys_l108_108623


namespace claire_apple_pies_l108_108752

theorem claire_apple_pies (N : ℤ) 
  (h1 : N % 6 = 4) 
  (h2 : N % 8 = 5) 
  (h3 : N < 30) : 
  N = 22 :=
by
  sorry

end claire_apple_pies_l108_108752


namespace aquarium_final_volume_l108_108066

theorem aquarium_final_volume :
  let length := 4
  let width := 6
  let height := 3
  let total_volume := length * width * height
  let initial_volume := total_volume / 2
  let spilled_volume := initial_volume / 2
  let remaining_volume := initial_volume - spilled_volume
  let final_volume := remaining_volume * 3
  final_volume = 54 :=
by sorry

end aquarium_final_volume_l108_108066


namespace equilibrium_point_stability_l108_108893

open Real

noncomputable def jacobian_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![-1, 0, 1],
    ![0, -2, -1],
    ![0, 1, -1]
  ]

def characteristic_polynomial (A : Matrix (Fin 3) (Fin 3) ℝ) : Polynomial ℝ :=
  (Polynomial.C (1:ℝ) • Polynomial.X) * Polynomial.det (Polynomial.C A - Polynomial.X • (1:Matrix (Fin 3) (Fin 3) ℝ))

theorem equilibrium_point_stability : Prop :=
  let λ := characteristic_polynomial jacobian_matrix in
  ∀ λ_i ∈ λ.rootSet ℂ, λ_i.re < 0

end equilibrium_point_stability_l108_108893


namespace eq_g_of_f_l108_108153

def f (x : ℝ) : ℝ := 3 * x - 5
def g (x : ℝ) : ℝ := 6 * x - 29

theorem eq_g_of_f (x : ℝ) : 2 * (f x) - 19 = g x :=
by 
  sorry

end eq_g_of_f_l108_108153


namespace positive_difference_of_two_numbers_l108_108875

theorem positive_difference_of_two_numbers :
  ∃ (x y : ℝ), x + y = 10 ∧ x^2 - y^2 = 24 ∧ |x - y| = 12 / 5 :=
by
  sorry

end positive_difference_of_two_numbers_l108_108875


namespace star_evaluation_l108_108959

noncomputable def star (a b : ℝ) : ℝ := (a + b) / (a - b)

theorem star_evaluation : (star (star 2 3) 4) = 1 / 9 := 
by sorry

end star_evaluation_l108_108959


namespace unique_prime_value_l108_108802

theorem unique_prime_value :
  ∃! n : ℕ, n > 0 ∧ Nat.Prime (n^3 - 7 * n^2 + 17 * n - 11) :=
by {
  sorry
}

end unique_prime_value_l108_108802


namespace factor_difference_of_squares_l108_108937

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) :=
by
  sorry

end factor_difference_of_squares_l108_108937


namespace determine_xyz_l108_108854

theorem determine_xyz (x y z : ℂ) (h1 : x * y + 3 * y = -9) (h2 : y * z + 3 * z = -9) (h3 : z * x + 3 * x = -9) : 
  x * y * z = 27 := 
by
  sorry

end determine_xyz_l108_108854


namespace solve_equation_l108_108874

theorem solve_equation : ∃ x : ℤ, 3 * x - 2 * x = 7 ∧ x = 7 :=
by
  sorry

end solve_equation_l108_108874


namespace part_a_part_b_part_c_l108_108886

theorem part_a (a b c : ℝ) (h : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) 
  (ineq : a^4 + b^4 + c^4 ≤ 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)) :
  a ≤ b + c ∧ b ≤ a + c ∧ c ≤ a + b := 
sorry

theorem part_b (a b c : ℝ) (h : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) 
  (ineq : a^4 + b^4 + c^4 ≤ 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)) :
  a^2 + b^2 + c^2 ≤ 2 * (a * b + b * c + c * a) := 
sorry

theorem part_c (a b c : ℝ) (h : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) :
  ¬ (a^2 + b^2 + c^2 ≤ 2 * (a * b + b * c + c * a) → 
     a^4 + b^4 + c^4 ≤ 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)) :=
sorry

end part_a_part_b_part_c_l108_108886


namespace digit_7_count_in_range_100_to_199_l108_108051

/-- 
Given the range of numbers from 100 to 199 inclusive,
prove that the digit 7 appears exactly 20 times.
-/
theorem digit_7_count_in_range_100_to_199 : 
  let count_7 := (λ n : ℕ, 
    (n / 100 = 7) +  -- hundreds place
    ((n % 100) / 10 = 7) + -- tens place
    ((n % 10) = 7)) in
  (Finset.sum (Finset.range' 100 100) count_7 = 20) :=
by sorry

end digit_7_count_in_range_100_to_199_l108_108051


namespace sequences_cover_naturals_without_repetition_l108_108525

theorem sequences_cover_naturals_without_repetition
  (x y : Real) 
  (hx : Irrational x) 
  (hy : Irrational y) 
  (hxy : 1/x + 1/y = 1) :
  (∀ n : ℕ, ∃! k : ℕ, (⌊k * x⌋ = n) ∨ (⌊k * y⌋ = n)) :=
sorry

end sequences_cover_naturals_without_repetition_l108_108525


namespace geometric_sequence_26th_term_l108_108727

noncomputable def r : ℝ := (8 : ℝ)^(1/6)

noncomputable def a (n : ℕ) (a₁ : ℝ) (r : ℝ) : ℝ := a₁ * r^(n - 1)

theorem geometric_sequence_26th_term :
  (a 26 (a 14 10 r) r = 640) :=
by
  have h₁ : a 14 10 r = 10 := sorry
  have h₂ : r^6 = 8 := sorry
  sorry

end geometric_sequence_26th_term_l108_108727


namespace least_positive_base_ten_seven_binary_digits_l108_108747

theorem least_positive_base_ten_seven_binary_digits : ∃ n : ℕ, n = 64 ∧ (n >= 2^6 ∧ n < 2^7) := 
by
  sorry

end least_positive_base_ten_seven_binary_digits_l108_108747


namespace rectangle_area_l108_108327

theorem rectangle_area (w l d : ℝ) 
  (h1 : l = 2 * w) 
  (h2 : d = 10)
  (h3 : d^2 = w^2 + l^2) : 
  l * w = 40 := 
by
  sorry

end rectangle_area_l108_108327


namespace determine_color_sum_or_product_l108_108235

theorem determine_color_sum_or_product {x : ℕ → ℝ} (h_distinct: ∀ i j : ℕ, i < j → x i < x j) (x_pos : ∀ i : ℕ, x i > 0) :
  ∃ c : ℕ → ℝ, (∀ i : ℕ, c i > 0) ∧
  (∀ i j : ℕ, i < j → (∃ r1 r2 : ℕ, (r1 ≠ r2) ∧ (c r1 + c r2 = x₆₄ + x₆₃) ∧ (c r1 * c r2 = x₆₄ * x₆₃))) :=
sorry

end determine_color_sum_or_product_l108_108235


namespace average_next_seven_l108_108214

variable (c : ℕ) (h : c > 0)

theorem average_next_seven (d : ℕ) (h1 : d = (2 * c + 3)) 
  : (d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7 = 2 * c + 6 := by
  sorry

end average_next_seven_l108_108214


namespace probability_of_two_germinates_is_48_over_125_l108_108314

noncomputable def probability_of_exactly_two_germinates : ℚ :=
  let p := 4/5
  let n := 3
  let k := 2
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_of_two_germinates_is_48_over_125 :
  probability_of_exactly_two_germinates = 48/125 := by
    sorry

end probability_of_two_germinates_is_48_over_125_l108_108314


namespace factor_diff_of_squares_l108_108930

theorem factor_diff_of_squares (t : ℝ) : 
  t^2 - 64 = (t - 8) * (t + 8) :=
by
  -- The purpose here is to state the theorem only, without the proof.
  sorry

end factor_diff_of_squares_l108_108930


namespace smallest_number_after_removal_l108_108046

-- Define the original number as a list of digits
def original_digits : List ℕ := [3, 7, 2, 8, 9, 5, 4, 1, 0, 6]

-- Define the function to check the smallest number by removing three digits
def smallest_seven_digit_number (digits: List ℕ) : List ℕ :=
  [2, 4, 5, 1, 0, 6, 7] -- correct seven digits by removing 3, 2, 8

theorem smallest_number_after_removal (original_digits : List ℕ) : 
  (smallest_seven_digit_number original_digits) = [2, 4, 5, 1, 0, 6, 7] :=
by
  sorry

end smallest_number_after_removal_l108_108046


namespace rectangle_x_satisfy_l108_108019

theorem rectangle_x_satisfy (x : ℝ) (h1 : 3 * x = 3 * x) (h2 : x + 5 = x + 5) (h3 : (3 * x) * (x + 5) = 2 * (3 * x) + 2 * (x + 5)) : x = 1 :=
sorry

end rectangle_x_satisfy_l108_108019


namespace distance_from_origin_l108_108183

theorem distance_from_origin :
  let origin := (0, 0)
      pt := (8, -15)
  in Real.sqrt ((pt.1 - origin.1)^2 + (pt.2 - origin.2)^2) = 17 :=
by
  let origin := (0, 0)
  let pt := (8, -15)
  have h1 : pt.1 - origin.1 = 8 := by rfl
  have h2 : pt.2 - origin.2 = -15 := by rfl
  simp [h1, h2]
  -- sorry will be used to skip the actual proof steps and to ensure the code builds successfully
  sorry

end distance_from_origin_l108_108183


namespace perfect_square_trinomial_m_l108_108026

theorem perfect_square_trinomial_m (m : ℝ) :
  (∃ a : ℝ, (x^2 + 2 * (m - 1) * x + 4) = (x + a)^2) → (m = 3 ∨ m = -1) :=
by
  sorry

end perfect_square_trinomial_m_l108_108026


namespace multiple_of_3_l108_108431

theorem multiple_of_3 (a b : ℤ) (h1 : ∃ m : ℤ, a = 3 * m) (h2 : ∃ n : ℤ, b = 9 * n) : ∃ k : ℤ, a + b = 3 * k :=
by
  sorry

end multiple_of_3_l108_108431


namespace vase_net_gain_l108_108425

theorem vase_net_gain 
  (selling_price : ℝ)
  (V1_cost : ℝ)
  (V2_cost : ℝ)
  (hyp1 : selling_price = 2.50)
  (hyp2 : 1.25 * V1_cost = selling_price)
  (hyp3 : 0.85 * V2_cost = selling_price) :
  (selling_price + selling_price) - (V1_cost + V2_cost) = 0.06 := 
by 
  sorry

end vase_net_gain_l108_108425


namespace hyperbola_eccentricity_proof_l108_108378

noncomputable def hyperbola_eccentricity_problem
  (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b)
  (Pf : ∃ P : ℝ × ℝ, (P.1 ^ 2 / a ^ 2 - P.2 ^ 2 / b ^ 2 = 1) ∧ 
                       (dist ⟨P.1, P.2⟩ ⟨-c, 0⟩ = a) ∧ 
                       (dist ⟨P.1, P.2⟩ ⟨c, 0⟩ = 3 * a)) : ℝ :=
let c := sqrt (a^2 + b^2) / 2 in
let e := c / a in
have : 2 * c * c = 10 * a * a, sorry,
(eccentricity (sqrt (1 + (b/a)^2) : ℝ)): ℝ :=
√10 / 2

theorem hyperbola_eccentricity_proof
  {a b : ℝ} (h_a_pos : 0 < a) (h_b_pos : 0 < b)
  (Pf : ∃ P : ℝ × ℝ, (P.1 ^ 2 / a ^ 2 - P.2 ^ 2 / b ^ 2 = 1) ∧ 
                       (dist ⟨P.1, P.2⟩ ⟨-c, 0⟩ = a) ∧ 
                       (dist ⟨P.1, P.2⟩ ⟨c, 0⟩ = 3 * a)) :
  h_eccentricity (c, a) =
  e := sorry

end hyperbola_eccentricity_proof_l108_108378


namespace divisor_is_three_l108_108908

theorem divisor_is_three (n d q p : ℕ) (h1 : n = d * q + 3) (h2 : n^2 = d * p + 3) : d = 3 := 
sorry

end divisor_is_three_l108_108908


namespace base_square_eq_l108_108684

theorem base_square_eq (b : ℕ) (h : (3*b + 3)^2 = b^3 + 2*b^2 + 3*b) : b = 9 :=
sorry

end base_square_eq_l108_108684


namespace number_of_ordered_tuples_l108_108660

noncomputable def count_tuples 
  (a1 a2 a3 a4 : ℕ) 
  (H_distinct : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)
  (H_range : 1 ≤ a1 ∧ a1 ≤ 100 ∧ 1 ≤ a2 ∧ a2 ≤ 100 ∧ 1 ≤ a3 ∧ a3 ≤ 100 ∧ 1 ≤ a4 ∧ a4 ≤ 100)
  (H_eqn : (a1^2 + a2^2 + a3^2) * (a2^2 + a3^2 + a4^2) = (a1 * a2 + a2 * a3 + a3 * a4)^2): ℕ :=
40

theorem number_of_ordered_tuples 
  (a1 a2 a3 a4 : ℕ)
  (H_distinct : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)
  (H_range : 1 ≤ a1 ∧ a1 ≤ 100 ∧ 1 ≤ a2 ∧ a2 ≤ 100 ∧ 1 ≤ a3 ∧ a3 ≤ 100 ∧ 1 ≤ a4 ∧ a4 ≤ 100)
  (H_eqn : (a1^2 + a2^2 + a3^2) * (a2^2 + a3^2 + a4^2) = (a1 * a2 + a2 * a3 + a3 * a4)^2) : 
  count_tuples a1 a2 a3 a4 H_distinct H_range H_eqn = 40 :=
sorry

end number_of_ordered_tuples_l108_108660


namespace ladder_distance_from_wall_l108_108774

noncomputable def dist_from_wall (ladder_length : ℝ) (angle_deg : ℝ) : ℝ :=
  ladder_length * Real.cos (angle_deg * Real.pi / 180)

theorem ladder_distance_from_wall :
  ∀ (ladder_length : ℝ) (angle_deg : ℝ), ladder_length = 19 → angle_deg = 60 → dist_from_wall ladder_length angle_deg = 9.5 :=
by
  intros ladder_length angle_deg h1 h2
  sorry

end ladder_distance_from_wall_l108_108774


namespace find_angle_NCB_l108_108680

def triangle_ABC_with_point_N (A B C N : Point) : Prop :=
  ∃ (angle_ABC angle_ACB angle_NAB angle_NBC : ℝ),
    angle_ABC = 50 ∧
    angle_ACB = 20 ∧
    angle_NAB = 40 ∧
    angle_NBC = 30 

theorem find_angle_NCB (A B C N : Point) 
  (h : triangle_ABC_with_point_N A B C N) :
  ∃ (angle_NCB : ℝ), 
  angle_NCB = 10 :=
sorry

end find_angle_NCB_l108_108680


namespace sum_of_octal_numbers_l108_108808

theorem sum_of_octal_numbers :
  let a := 0o1275
  let b := 0o164
  let sum := 0o1503
  a + b = sum :=
by
  -- Proof is omitted here with sorry
  sorry

end sum_of_octal_numbers_l108_108808


namespace compute_g_neg_x_l108_108498

noncomputable def g (x : ℝ) : ℝ := (x^2 + 3*x + 2) / (x^2 - 3*x + 2)

theorem compute_g_neg_x (x : ℝ) (h : x^2 ≠ 2) : g (-x) = 1 / g x := 
  by sorry

end compute_g_neg_x_l108_108498


namespace four_divides_sum_of_squares_iff_even_l108_108201

theorem four_divides_sum_of_squares_iff_even (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (4 ∣ (a^2 + b^2 + c^2)) ↔ (Even a ∧ Even b ∧ Even c) :=
by
  sorry

end four_divides_sum_of_squares_iff_even_l108_108201


namespace yield_percentage_of_stock_l108_108104

noncomputable def annual_dividend (par_value : ℝ) : ℝ := 0.21 * par_value
noncomputable def market_price : ℝ := 210
noncomputable def yield_percentage (annual_dividend : ℝ) (market_price : ℝ) : ℝ :=
  (annual_dividend / market_price) * 100

theorem yield_percentage_of_stock (par_value : ℝ)
  (h_par_value : par_value = 100) :
  yield_percentage (annual_dividend par_value) market_price = 10 :=
by
  sorry

end yield_percentage_of_stock_l108_108104


namespace arithmetic_progression_sum_l108_108466

theorem arithmetic_progression_sum (a d : ℝ) (n : ℕ) : 
  a + 10 * d = 5.25 → 
  a + 6 * d = 3.25 → 
  (n : ℝ) / 2 * (2 * a + (n - 1) * d) = 56.25 → 
  n = 15 :=
by
  intros h1 h2 h3
  sorry

end arithmetic_progression_sum_l108_108466


namespace hypotenuse_of_454590_triangle_l108_108557

theorem hypotenuse_of_454590_triangle (l : ℝ) (angle : ℝ) (h : ℝ) (h_leg : l = 15) (h_angle : angle = 45) :
  h = l * Real.sqrt 2 := 
  sorry

end hypotenuse_of_454590_triangle_l108_108557


namespace initial_population_l108_108313

/--
Suppose 5% of people in a village died by bombardment,
15% of the remaining population left the village due to fear,
and the population is now reduced to 3294.
Prove that the initial population was 4080.
-/
theorem initial_population (P : ℝ) 
  (H1 : 0.05 * P + 0.15 * (1 - 0.05) * P + 3294 = P) : P = 4080 :=
sorry

end initial_population_l108_108313


namespace positive_difference_of_two_numbers_l108_108876

theorem positive_difference_of_two_numbers :
  ∃ (x y : ℝ), x + y = 10 ∧ x^2 - y^2 = 24 ∧ |x - y| = 12 / 5 :=
by
  sorry

end positive_difference_of_two_numbers_l108_108876


namespace distance_origin_to_point_l108_108180

theorem distance_origin_to_point :
  let distance (x1 y1 x2 y2 : ℝ) := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  distance 0 0 8 (-15) = 17 :=
by
  sorry

end distance_origin_to_point_l108_108180


namespace surface_area_of_interior_box_l108_108860

def original_sheet_width : ℕ := 40
def original_sheet_length : ℕ := 50
def corner_cut_side : ℕ := 8
def corners_count : ℕ := 4

def area_of_original_sheet : ℕ := original_sheet_width * original_sheet_length
def area_of_one_corner_cut : ℕ := corner_cut_side * corner_cut_side
def total_area_removed : ℕ := corners_count * area_of_one_corner_cut
def area_of_remaining_sheet : ℕ := area_of_original_sheet - total_area_removed

theorem surface_area_of_interior_box : area_of_remaining_sheet = 1744 :=
by
  sorry

end surface_area_of_interior_box_l108_108860


namespace inclination_angle_l108_108086

theorem inclination_angle (θ : ℝ) (h : 0 ≤ θ ∧ θ < 180) :
  (∀ x y : ℝ, x - y + 3 = 0 → θ = 45) :=
sorry

end inclination_angle_l108_108086


namespace sin_double_angle_l108_108146

open Real

theorem sin_double_angle (θ : ℝ) (h : cos (θ + π) = -1 / 3) : sin (2 * θ + π / 2) = -7 / 9 :=
by
  sorry

end sin_double_angle_l108_108146


namespace ratio_expression_l108_108963

theorem ratio_expression (a b c : ℝ) (ha : a / b = 20) (hb : b / c = 10) : (a + b) / (b + c) = 210 / 11 := by
  sorry

end ratio_expression_l108_108963


namespace heptagon_divisibility_impossible_l108_108537

theorem heptagon_divisibility_impossible (a b c d e f g : ℕ) :
  (b ∣ a ∨ a ∣ b) ∧ (c ∣ b ∨ b ∣ c) ∧ (d ∣ c ∨ c ∣ d) ∧ (e ∣ d ∨ d ∣ e) ∧
  (f ∣ e ∨ e ∣ f) ∧ (g ∣ f ∨ f ∣ g) ∧ (a ∣ g ∨ g ∣ a) →
  ¬((a ∣ c ∨ c ∣ a) ∧ (a ∣ d ∨ d ∣ a) ∧ (a ∣ e ∨ e ∣ a) ∧ (a ∣ f ∨ f ∣ a) ∧
    (a ∣ g ∨ g ∣ a) ∧ (b ∣ d ∨ d ∣ b) ∧ (b ∣ e ∨ e ∣ b) ∧ (b ∣ f ∨ f ∣ b) ∧
    (b ∣ g ∨ g ∣ b) ∧ (c ∣ e ∨ e ∣ c) ∧ (c ∣ f ∨ f ∣ c) ∧ (c ∣ g ∨ g ∣ c) ∧
    (d ∣ f ∨ f ∣ d) ∧ (d ∣ g ∨ g ∣ d) ∧ (e ∣ g ∨ g ∣ e)) :=
 by
  sorry

end heptagon_divisibility_impossible_l108_108537


namespace sequence_n_500_l108_108540

theorem sequence_n_500 (a : ℕ → ℤ) 
  (h1 : a 1 = 1010) 
  (h2 : a 2 = 1011) 
  (h3 : ∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = 2 * n + 3) : 
  a 500 = 3003 := 
sorry

end sequence_n_500_l108_108540


namespace find_third_number_l108_108479

theorem find_third_number (x : ℕ) : 9548 + 7314 = x + 13500 ↔ x = 3362 :=
by
  sorry

end find_third_number_l108_108479


namespace distance_between_trains_l108_108091

theorem distance_between_trains (d1 d2 : ℝ) (t1 t2 : ℝ) (s1 s2 : ℝ) (x : ℝ) :
  d1 = d2 + 100 →
  s1 = 50 →
  s2 = 40 →
  d1 = s1 * t1 →
  d2 = s2 * t2 →
  t1 = t2 →
  d2 = 400 →
  d1 + d2 = 900 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end distance_between_trains_l108_108091


namespace puja_runs_distance_in_meters_l108_108073

noncomputable def puja_distance (time_in_seconds : ℝ) (speed_kmph : ℝ) : ℝ :=
  let time_in_hours := time_in_seconds / 3600
  let distance_km := speed_kmph * time_in_hours
  distance_km * 1000

theorem puja_runs_distance_in_meters :
  abs (puja_distance 59.995200383969284 30 - 499.96) < 0.01 :=
by
  sorry

end puja_runs_distance_in_meters_l108_108073


namespace smallest_k_l108_108739

theorem smallest_k :
  ∃ k : ℕ, k > 1 ∧ (k % 19 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) ∧ k = 400 :=
by
  sorry

end smallest_k_l108_108739


namespace number_of_solutions_depends_on_a_l108_108128

theorem number_of_solutions_depends_on_a (a : ℝ) : 
  (∀ x : ℝ, 2^(3 * x) + 4 * a * 2^(2 * x) + a^2 * 2^x - 6 * a^3 = 0) → 
  (if a = 0 then 0 else if a > 0 then 1 else 2) = 
  (if a = 0 then 0 else if a > 0 then 1 else 2) :=
by 
  sorry

end number_of_solutions_depends_on_a_l108_108128


namespace train_crossing_time_l108_108171

namespace TrainCrossingProblem

def length_of_train : ℕ := 250
def length_of_bridge : ℕ := 300
def speed_of_train_kmph : ℕ := 36
def speed_of_train_mps : ℕ := 10 -- conversion from 36 kmph to m/s
def total_distance : ℕ := length_of_train + length_of_bridge -- 250 + 300
def expected_time : ℕ := 55

theorem train_crossing_time : 
  (total_distance / speed_of_train_mps) = expected_time :=
by
  sorry
end TrainCrossingProblem

end train_crossing_time_l108_108171


namespace sqrt_49_times_sqrt_25_l108_108291

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 :=
by
  sorry

end sqrt_49_times_sqrt_25_l108_108291


namespace area_of_flowerbed_l108_108628

theorem area_of_flowerbed :
  ∀ (a b : ℕ), 2 * (a + b) = 24 → b + 1 = 3 * (a + 1) → 
  let shorter_side := 3 * a
  let longer_side := 3 * b
  shorter_side * longer_side = 144 :=
by
  sorry

end area_of_flowerbed_l108_108628


namespace SufficientCondition_l108_108467

theorem SufficientCondition :
  ∀ x y z : ℤ, x = z ∧ y = x - 1 → x * (x - y) + y * (y - z) + z * (z - x) = 2 :=
by
  intros x y z h
  cases h with
  | intro h1 h2 =>
  sorry

end SufficientCondition_l108_108467


namespace find_vector_coordinates_l108_108004

structure Point3D :=
  (x y z : ℝ)

def vector_sub (a b : Point3D) : Point3D :=
  Point3D.mk (b.x - a.x) (b.y - a.y) (b.z - a.z)

theorem find_vector_coordinates (A B : Point3D)
  (hA : A = { x := 1, y := -3, z := 4 })
  (hB : B = { x := -3, y := 2, z := 1 }) :
  vector_sub A B = { x := -4, y := 5, z := -3 } :=
by
  -- Proof is omitted
  sorry

end find_vector_coordinates_l108_108004


namespace find_X_value_l108_108135

theorem find_X_value (X : ℝ) : 
  (1.5 * ((3.6 * 0.48 * 2.5) / (0.12 * X * 0.5)) = 1200.0000000000002) → 
  X = 0.225 :=
by
  sorry

end find_X_value_l108_108135


namespace solution_exists_l108_108219

-- Defining the variables x and y
variables (x y : ℝ)

-- Defining the conditions
def condition_1 : Prop :=
  3 * x ≥ 2 * y + 16

def condition_2 : Prop :=
  x^4 + 2 * (x^2) * (y^2) + y^4 + 25 - 26 * (x^2) - 26 * (y^2) = 72 * x * y

-- Stating the theorem that (6, 1) satisfies the conditions
theorem solution_exists : condition_1 6 1 ∧ condition_2 6 1 :=
by
  -- Convert conditions into expressions
  have h1 : condition_1 6 1 := by sorry
  have h2 : condition_2 6 1 := by sorry
  -- Conjunction of both conditions is satisfied
  exact ⟨h1, h2⟩

end solution_exists_l108_108219


namespace time_spent_per_egg_in_seconds_l108_108356

-- Definitions based on the conditions in the problem
def minutes_per_roll : ℕ := 30
def number_of_rolls : ℕ := 7
def total_cleaning_time : ℕ := 225
def number_of_eggs : ℕ := 60

-- Problem statement
theorem time_spent_per_egg_in_seconds :
  (total_cleaning_time - number_of_rolls * minutes_per_roll) * 60 / number_of_eggs = 15 := by
  sorry

end time_spent_per_egg_in_seconds_l108_108356


namespace prove_weight_loss_l108_108761

variable (W : ℝ) -- Original weight
variable (x : ℝ) -- Percentage of weight lost

def weight_equation := W - (x / 100) * W + (2 / 100) * W = (89.76 / 100) * W

theorem prove_weight_loss (h : weight_equation W x) : x = 12.24 :=
by
  sorry

end prove_weight_loss_l108_108761


namespace four_letter_words_with_A_l108_108379

theorem four_letter_words_with_A :
  let letters := ['A', 'B', 'C', 'D', 'E']
  in let total_4_letter_words := 5^4
  in let words_without_A := 4^4
  in total_4_letter_words - words_without_A = 369 := by
  sorry

end four_letter_words_with_A_l108_108379


namespace sqrt_neg_sq_eq_two_l108_108919

theorem sqrt_neg_sq_eq_two : Real.sqrt ((-2 : ℝ)^2) = 2 := by
  -- Proof intentionally omitted.
  sorry

end sqrt_neg_sq_eq_two_l108_108919


namespace max_area_of_triangle_l108_108994

theorem max_area_of_triangle (a b c : ℝ) (hC : C = 60) (h1 : 3 * a * b = 25 - c^2) :
  (∃ S : ℝ, S = (a * b * (Real.sqrt 3)) / 4 ∧ S = 25 * (Real.sqrt 3) / 16) :=
sorry

end max_area_of_triangle_l108_108994


namespace max_A_excircle_area_ratio_max_A_excircle_area_ratio_eq_l108_108411

noncomputable def A_excircle_area_ratio (α : Real) (s : Real) : Real :=
  0.5 * Real.sin α

theorem max_A_excircle_area_ratio (α : Real) (s : Real) : (A_excircle_area_ratio α s) ≤ 0.5 :=
by
  sorry

theorem max_A_excircle_area_ratio_eq (s : Real) : 
  (A_excircle_area_ratio (Real.pi / 2) s) = 0.5 :=
by
  sorry

end max_A_excircle_area_ratio_max_A_excircle_area_ratio_eq_l108_108411


namespace evaporation_days_l108_108315

theorem evaporation_days
    (initial_water : ℝ)
    (evap_rate : ℝ)
    (percent_evaporated : ℝ)
    (evaporated_water : ℝ)
    (days : ℝ)
    (h1 : initial_water = 10)
    (h2 : evap_rate = 0.012)
    (h3 : percent_evaporated = 0.06)
    (h4 : evaporated_water = initial_water * percent_evaporated)
    (h5 : days = evaporated_water / evap_rate) :
  days = 50 :=
by
  sorry

end evaporation_days_l108_108315


namespace inequality_solution_l108_108585

theorem inequality_solution (y : ℝ) : 
  (3 ≤ |y - 4| ∧ |y - 4| ≤ 7) ↔ (7 ≤ y ∧ y ≤ 11 ∨ -3 ≤ y ∧ y ≤ 1) :=
by
  sorry

end inequality_solution_l108_108585


namespace intersection_of_A_and_B_l108_108980

-- Definitions from conditions
def A : Set ℤ := {x | x - 1 ≥ 0}
def B : Set ℤ := {0, 1, 2}

-- Proof statement
theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end intersection_of_A_and_B_l108_108980


namespace sufficient_but_not_necessary_condition_x_gt_5_x_gt_3_l108_108369

theorem sufficient_but_not_necessary_condition_x_gt_5_x_gt_3 :
  ∀ x : ℝ, (x > 5 → x > 3) ∧ (∃ x : ℝ, x > 3 ∧ x ≤ 5) :=
by
  sorry

end sufficient_but_not_necessary_condition_x_gt_5_x_gt_3_l108_108369


namespace diff_of_squares_odd_divisible_by_8_l108_108215

theorem diff_of_squares_odd_divisible_by_8 (m n : ℤ) :
  ((2 * m + 1) ^ 2 - (2 * n + 1) ^ 2) % 8 = 0 :=
by 
  sorry

end diff_of_squares_odd_divisible_by_8_l108_108215


namespace ratio_buses_to_cars_l108_108231

theorem ratio_buses_to_cars (B C : ℕ) (h1 : B = C - 60) (h2 : C = 65) : B / C = 1 / 13 :=
by 
  sorry

end ratio_buses_to_cars_l108_108231


namespace man_l108_108608

-- Define all given conditions using Lean definitions
def speed_with_current_wind : ℝ := 22
def speed_of_current : ℝ := 5
def wind_resistance_factor : ℝ := 0.15
def current_increase_factor : ℝ := 0.10

-- Define the key quantities (man's speed in still water, effective speed in still water, new current speed against)
def speed_in_still_water : ℝ := speed_with_current_wind - speed_of_current
def effective_speed_in_still_water : ℝ := speed_in_still_water - (wind_resistance_factor * speed_in_still_water)
def new_speed_of_current_against : ℝ := speed_of_current + (current_increase_factor * speed_of_current)

-- Proof goal: Prove that the man's speed against the current is 8.95 km/hr considering all the conditions
theorem man's_speed_against_current_is_correct : 
  (effective_speed_in_still_water - new_speed_of_current_against) = 8.95 := 
by
  sorry

end man_l108_108608


namespace range_of_f_l108_108827

noncomputable def f (x : ℝ) : ℝ :=
  (real.sqrt 3) * real.sin x - real.cos x

theorem range_of_f : set.range f = set.Icc (-2 : ℝ) (2 : ℝ) :=
sorry

end range_of_f_l108_108827


namespace percentage_error_calculation_l108_108619

theorem percentage_error_calculation (x : ℝ) :
  let correct_value := x * (5 / 3)
  let incorrect_value := x * (3 / 5)
  let difference := correct_value - incorrect_value
  let percentage_error := (difference / correct_value) * 100
  percentage_error = 64 := 
by
  let correct_value := x * (5 / 3)
  let incorrect_value := x * (3 / 5)
  let difference := correct_value - incorrect_value
  let percentage_error := (difference / correct_value) * 100
  sorry

end percentage_error_calculation_l108_108619


namespace perfect_cubes_l108_108649

theorem perfect_cubes (n : ℕ) (h : n > 0) : 
  (n = 7 ∨ n = 11 ∨ n = 12 ∨ n = 25) ↔ ∃ k : ℤ, (n^3 - 18*n^2 + 115*n - 391) = k^3 :=
by exact sorry

end perfect_cubes_l108_108649


namespace dividend_is_correct_l108_108599

theorem dividend_is_correct :
  ∃ (R D Q V: ℕ), R = 6 ∧ D = 5 * Q ∧ D = 3 * R + 2 ∧ V = D * Q + R ∧ V = 86 :=
by
  sorry

end dividend_is_correct_l108_108599


namespace james_weekly_earnings_l108_108408

def hourly_rate : ℕ := 20
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 4

theorem james_weekly_earnings : hourly_rate * (hours_per_day * days_per_week) = 640 := by
  sorry

end james_weekly_earnings_l108_108408


namespace percentage_increase_l108_108108

theorem percentage_increase
  (W R : ℝ)
  (H1 : 0.70 * R = 1.04999999999999982 * W) :
  (R - W) / W * 100 = 50 :=
by
  sorry

end percentage_increase_l108_108108


namespace sqrt_49_times_sqrt25_eq_7_sqrt5_l108_108298

-- Definitions based on conditions
def sqrt_25 := Real.sqrt 25
def sqrt_49 := Real.sqrt 49

theorem sqrt_49_times_sqrt25_eq_7_sqrt5 :
  Real.sqrt (49 * sqrt_25) = 7 * Real.sqrt 5 := by
sorry

end sqrt_49_times_sqrt25_eq_7_sqrt5_l108_108298


namespace smallest_k_l108_108743

theorem smallest_k (k : ℕ) (h1 : k > 1) (h2 : k % 19 = 1) (h3 : k % 7 = 1) (h4 : k % 3 = 1) : k = 400 :=
by
  sorry

end smallest_k_l108_108743


namespace sum_of_all_possible_values_l108_108550

theorem sum_of_all_possible_values (x y : ℝ) (h : x * y - x^2 - y^2 = 4) :
  (x - 2) * (y - 2) = 4 :=
sorry

end sum_of_all_possible_values_l108_108550


namespace tan_symmetric_about_k_pi_over_2_min_value_cos2x_plus_sinx_l108_108521

theorem tan_symmetric_about_k_pi_over_2 (k : ℤ) : 
  (∀ x : ℝ, Real.tan (x + k * Real.pi / 2) = Real.tan x) := 
sorry

theorem min_value_cos2x_plus_sinx : 
  (∀ x : ℝ, Real.cos x ^ 2 + Real.sin x ≥ -1) ∧ (∃ x : ℝ, Real.cos x ^ 2 + Real.sin x = -1) :=
sorry

end tan_symmetric_about_k_pi_over_2_min_value_cos2x_plus_sinx_l108_108521


namespace length_PQ_circle_line_l108_108663

def circle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y = 0

def line_parametric (t : ℝ) (x y : ℝ) : Prop := 
  x = -1 + t ∧ y = t

def polar_ray (theta : ℝ) : Prop := theta = 3 * Real.pi / 4

theorem length_PQ_circle_line :
  let P := (2 * Real.sqrt 2, 3 * Real.pi / 4)
  let Q := (Real.sqrt 2, 3 * Real.pi / 4)
  dist P Q = Real.sqrt 2 :=
sorry

end length_PQ_circle_line_l108_108663


namespace compute_value_l108_108204

def diamond_op (x y : ℕ) : ℕ := 3 * x + 5 * y
def heart_op (z x : ℕ) : ℕ := 4 * z + 2 * x

theorem compute_value : heart_op (diamond_op 4 3) 8 = 124 := by
  sorry

end compute_value_l108_108204


namespace intersection_PQ_l108_108199

def P := {x : ℝ | x < 1}
def Q := {x : ℝ | x^2 < 4}
def PQ_intersection := {x : ℝ | -2 < x ∧ x < 1}

theorem intersection_PQ : P ∩ Q = PQ_intersection := by
  sorry

end intersection_PQ_l108_108199


namespace count_special_numbers_l108_108858

theorem count_special_numbers : 
  {n // 2010 ≤ n ∧ n ≤ 2099 ∧ ∃ x y : ℕ, n = 2000 + x * 10 + y ∧ (20 * y = x * x)}.card = 3 :=
by sorry

end count_special_numbers_l108_108858


namespace range_of_m_l108_108392

theorem range_of_m (m : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - m * x - m < 0) ↔ (-4 ≤ m ∧ m ≤ 0) := 
by sorry

end range_of_m_l108_108392


namespace pebbles_ratio_l108_108424

variable (S : ℕ)

theorem pebbles_ratio :
  let initial_pebbles := 18
  let skipped_pebbles := 9
  let additional_pebbles := 30
  let final_pebbles := 39
  initial_pebbles - skipped_pebbles + additional_pebbles = final_pebbles →
  (skipped_pebbles : ℚ) / initial_pebbles = 1 / 2 :=
by
  intros
  sorry

end pebbles_ratio_l108_108424


namespace initial_range_calculation_l108_108906

variable (initial_range telescope_range : ℝ)
variable (increased_by : ℝ)
variable (h_telescope : telescope_range = increased_by * initial_range)

theorem initial_range_calculation 
  (h_telescope_range : telescope_range = 150)
  (h_increased_by : increased_by = 3)
  (h_telescope : telescope_range = increased_by * initial_range) :
  initial_range = 50 :=
  sorry

end initial_range_calculation_l108_108906


namespace average_of_roots_l108_108610

theorem average_of_roots (p q : ℝ) (h : ∀ r : ℝ, r^2 * (3 * p) + r * (-6 * p) + q = 0 → ∃ a b : ℝ, r = a ∨ r = b) : 
  ∀ (r1 r2 : ℝ), (3 * p) * r1^2 + (-6 * p) * r1 + q = 0 ∧ (3 * p) * r2^2 + (-6 * p) * r2 + q = 0 → 
  (r1 + r2) / 2 = 1 :=
by {
  sorry
}

end average_of_roots_l108_108610


namespace find_complex_z_l108_108360

open Complex

noncomputable def is_real (z : ℂ) : Prop := ∃ (r : ℝ), z = r

theorem find_complex_z (z : ℂ) (h1 : ∥conj z - 3∥ = ∥conj z - 3 * I∥)
  (h2 : is_real (z - 1 + 5 / (z - 1))) : z = 2 - 2 * I ∨ z = -1 + I :=
by
  sorry

end find_complex_z_l108_108360


namespace power_function_no_origin_l108_108529

theorem power_function_no_origin (m : ℝ) : 
  (m^2 - m - 1 <= 0) ∧ (m^2 - 3 * m + 3 = 1) → m = 1 :=
by
  intros
  sorry

end power_function_no_origin_l108_108529


namespace asimov_books_l108_108500

theorem asimov_books (h p : Nat) (condition1 : h + p = 12) (condition2 : 30 * h + 20 * p = 300) : h = 6 := by
  sorry

end asimov_books_l108_108500


namespace total_cost_first_3_years_l108_108056

def monthly_fee : ℕ := 12
def down_payment : ℕ := 50
def years : ℕ := 3

theorem total_cost_first_3_years :
  (years * 12 * monthly_fee + down_payment) = 482 :=
by
  sorry

end total_cost_first_3_years_l108_108056


namespace div_mul_neg_one_third_l108_108122

theorem div_mul_neg_one_third : (2 : ℚ) / 3 * (-1/3) = -2/9 := by
  sorry

end div_mul_neg_one_third_l108_108122


namespace cosine_shift_right_eq_l108_108452

notation "π" => Real.pi

theorem cosine_shift_right_eq :
  ∀ (x : ℝ), 2 * cos (2 * (x - π / 8)) = 2 * cos (2 * x - π / 4) :=
by
  intro x
  sorry

end cosine_shift_right_eq_l108_108452


namespace mechanic_hourly_rate_l108_108775

-- Definitions and conditions
def total_bill : ℕ := 450
def parts_charge : ℕ := 225
def hours_worked : ℕ := 5

-- The main theorem to prove
theorem mechanic_hourly_rate : (total_bill - parts_charge) / hours_worked = 45 := by
  sorry

end mechanic_hourly_rate_l108_108775


namespace number_of_distinct_real_roots_l108_108021

theorem number_of_distinct_real_roots (k : ℕ) :
  (∃ k : ℕ, ∀ x : ℝ, |x| - 4 = (3 * |x|) / 2 → 0 = k) :=
  sorry

end number_of_distinct_real_roots_l108_108021


namespace carla_water_drank_l108_108349

theorem carla_water_drank (W S : ℝ) (h1 : W + S = 54) (h2 : S = 3 * W - 6) : W = 15 :=
by
  sorry

end carla_water_drank_l108_108349


namespace find_x_l108_108082

variable {a b x : ℝ}

-- Defining the given conditions
def is_linear_and_unique_solution (a b : ℝ) : Prop :=
  3 * a + 2 * b = 0 ∧ a ≠ 0

-- The proof problem: prove that x = 1.5, given the conditions.
theorem find_x (ha : is_linear_and_unique_solution a b) : x = 1.5 :=
  sorry

end find_x_l108_108082


namespace length_of_train_l108_108311

-- We state the conditions as definitions.
def length_of_train_equals_length_of_platform (l_train l_platform : ℝ) : Prop :=
l_train = l_platform

def speed_of_train (s : ℕ) : Prop :=
s = 216

def crossing_time (t : ℕ) : Prop :=
t = 1

-- Defining the goal according to the problem statement.
theorem length_of_train (l_train l_platform : ℝ) (s t : ℕ) 
  (h1 : length_of_train_equals_length_of_platform l_train l_platform) 
  (h2 : speed_of_train s) 
  (h3 : crossing_time t) : 
  l_train = 1800 :=
by
  sorry

end length_of_train_l108_108311


namespace rolls_remaining_to_sell_l108_108955

-- Conditions
def total_rolls_needed : ℕ := 45
def rolls_sold_to_grandmother : ℕ := 1
def rolls_sold_to_uncle : ℕ := 10
def rolls_sold_to_neighbor : ℕ := 6

-- Theorem statement
theorem rolls_remaining_to_sell : (total_rolls_needed - (rolls_sold_to_grandmother + rolls_sold_to_uncle + rolls_sold_to_neighbor) = 28) :=
by
  sorry

end rolls_remaining_to_sell_l108_108955


namespace find_distance_from_origin_l108_108780

-- Define the conditions as functions
def point_distance_from_x_axis (y : ℝ) : Prop := abs y = 15
def distance_from_point (x y : ℝ) (x₀ y₀ : ℝ) (d : ℝ) : Prop := (x - x₀)^2 + (y - y₀)^2 = d^2

-- Define the proof problem
theorem find_distance_from_origin (x y : ℝ) (n : ℝ) (hx : x = 2 + Real.sqrt 105) (hy : point_distance_from_x_axis y) (hx_gt : x > 2) (hdist : distance_from_point x y 2 7 13) :
  n = Real.sqrt (334 + 4 * Real.sqrt 105) :=
sorry

end find_distance_from_origin_l108_108780


namespace hypotenuse_length_l108_108111

theorem hypotenuse_length (x y : ℝ) 
  (h1 : (1/3) * Real.pi * y^2 * x = 1080 * Real.pi) 
  (h2 : (1/3) * Real.pi * x^2 * y = 2430 * Real.pi) : 
  Real.sqrt (x^2 + y^2) = 6 * Real.sqrt 13 := 
  sorry

end hypotenuse_length_l108_108111


namespace smallest_fraction_numerator_l108_108784

theorem smallest_fraction_numerator :
  ∃ (a b : ℕ), (10 ≤ a ∧ a < 100) ∧ (10 ≤ b ∧ b < 100) ∧ (5 * b < 7 * a) ∧ 
    ∀ (a' b' : ℕ), (10 ≤ a' ∧ a' < 100) ∧ (10 ≤ b' ∧ b' < 100) ∧ (5 * b' < 7 * a') →
    (a * b' ≤ a' * b) → a = 68 :=
sorry

end smallest_fraction_numerator_l108_108784


namespace convex_polygons_on_circle_l108_108507

theorem convex_polygons_on_circle:
  let points := 15 in
  ∑ i in finset.range (points + 1), choose points i - (choose points 0 + choose points 1 + choose points 2 + choose points 3) = 32192 :=
begin
  sorry
end

end convex_polygons_on_circle_l108_108507


namespace quadratic_real_roots_and_a_value_l108_108977

-- Define the quadratic equation (a-5)x^2 - 4x - 1 = 0
def quadratic_eq (a : ℝ) (x : ℝ) := (a - 5) * x^2 - 4 * x - 1

-- Define the discriminant for the quadratic equation
def discriminant (a : ℝ) := 4 - 4 * (a - 5) * (-1)

-- Main theorem statement
theorem quadratic_real_roots_and_a_value
    (a : ℝ) (x1 x2 : ℝ) 
    (h_roots : (a - 5) ≠ 0)
    (h_eq : quadratic_eq a x1 = 0 ∧ quadratic_eq a x2 = 0)
    (h_sum_product : x1 + x2 + x1 * x2 = 3) :
    (a ≥ 1) ∧ (a = 6) :=
  sorry

end quadratic_real_roots_and_a_value_l108_108977


namespace sqrt_49_mul_sqrt_25_l108_108278

theorem sqrt_49_mul_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_l108_108278


namespace algebraic_expression_evaluation_l108_108835

theorem algebraic_expression_evaluation (a b : ℝ) (h₁ : a ≠ b) 
  (h₂ : a^2 - 8 * a + 5 = 0) (h₃ : b^2 - 8 * b + 5 = 0) :
  (b - 1) / (a - 1) + (a - 1) / (b - 1) = -20 :=
by
  sorry

end algebraic_expression_evaluation_l108_108835


namespace max_k_for_ineq_l108_108007

theorem max_k_for_ineq (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : m^3 + n^3 > (m + n)^2) :
  m^3 + n^3 ≥ (m + n)^2 + 10 :=
sorry

end max_k_for_ineq_l108_108007


namespace Gage_skating_minutes_l108_108145

theorem Gage_skating_minutes (d1 d2 d3 : ℕ) (m1 m2 : ℕ) (avg : ℕ) (h1 : d1 = 6) (h2 : d2 = 4) (h3 : d3 = 1) (h4 : m1 = 80) (h5 : m2 = 105) (h6 : avg = 95) : 
  (d1 * m1 + d2 * m2 + d3 * x) / (d1 + d2 + d3) = avg ↔ x = 145 := 
by 
  sorry

end Gage_skating_minutes_l108_108145


namespace three_point_seven_five_as_fraction_l108_108594

theorem three_point_seven_five_as_fraction :
  (15 : ℚ) / 4 = 3.75 :=
sorry

end three_point_seven_five_as_fraction_l108_108594


namespace find_b_minus_a_l108_108059

theorem find_b_minus_a (a b : ℤ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a - 9 * b + 18 * a * b = 2018) : b - a = 223 :=
sorry

end find_b_minus_a_l108_108059


namespace abs_x_minus_1_le_1_is_equivalent_to_x_le_2_l108_108987

theorem abs_x_minus_1_le_1_is_equivalent_to_x_le_2 (x : ℝ) :
  (|x - 1| ≤ 1) ↔ (x ≤ 2) := sorry

end abs_x_minus_1_le_1_is_equivalent_to_x_le_2_l108_108987


namespace smallest_x_undefined_l108_108892

theorem smallest_x_undefined : ∃ x : ℝ, (10 * x^2 - 90 * x + 20 = 0) ∧ x = 1 / 4 :=
by sorry

end smallest_x_undefined_l108_108892


namespace margo_pairing_probability_l108_108682

theorem margo_pairing_probability (students : Finset ℕ)
  (H_50_students : students.card = 50)
  (margo irma jess kurt : ℕ)
  (H_margo_in_students : margo ∈ students)
  (H_irma_in_students : irma ∈ students)
  (H_jess_in_students : jess ∈ students)
  (H_kurt_in_students : kurt ∈ students)
  (possible_partners : Finset ℕ := students.erase margo) :
  (3: ℝ) / 49 = ((3: ℝ) / (possible_partners.card: ℝ)) :=
by
  -- The actual steps of the proof will be here
  sorry

end margo_pairing_probability_l108_108682


namespace find_xyz_l108_108358

theorem find_xyz
  (x y z : ℝ)
  (h1 : x + y + z = 38)
  (h2 : x * y * z = 2002)
  (h3 : 0 < x ∧ x ≤ 11)
  (h4 : z ≥ 14) :
  x = 11 ∧ y = 13 ∧ z = 14 :=
sorry

end find_xyz_l108_108358


namespace find_g_of_3_l108_108174

theorem find_g_of_3 (f g : ℝ → ℝ) (h₁ : ∀ x, f x = 2 * x + 3) (h₂ : ∀ x, g (x + 2) = f x) :
  g 3 = 5 :=
sorry

end find_g_of_3_l108_108174


namespace least_positive_base_ten_with_seven_binary_digits_l108_108751

theorem least_positive_base_ten_with_seven_binary_digits : 
  ∃ n : ℕ, (n >= 1 ∧ 7 ≤ n.digits 2 .length) → n = 64 :=
begin
  sorry
end

end least_positive_base_ten_with_seven_binary_digits_l108_108751


namespace sum_of_other_endpoint_l108_108325

theorem sum_of_other_endpoint (x y : ℝ) (h₁ : (9 + x) / 2 = 5) (h₂ : (-6 + y) / 2 = -8) :
  x + y = -9 :=
sorry

end sum_of_other_endpoint_l108_108325


namespace sequence_properties_l108_108671

-- Define the sequence according to the problem
def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ (∀ n : ℕ, n ≥ 2 → a n = (n * a (n - 1)) / (n - 1))

-- State the theorem to be proved
theorem sequence_properties :
  ∃ (a : ℕ → ℕ), 
    seq a ∧ a 2 = 6 ∧ a 3 = 9 ∧ (∀ n : ℕ, n ≥ 1 → a n = 3 * n) :=
by
  -- Existence quantifier and properties (sequence definition, first three terms, and general term)
  sorry

end sequence_properties_l108_108671


namespace choose_president_and_vice_president_l108_108572

theorem choose_president_and_vice_president :
  let total_members := 24
  let boys := 8
  let girls := 16
  let senior_members := 4
  let senior_boys := 2
  let senior_girls := 2
  let president_choices := senior_members
  let vice_president_choices_boy_pres := girls
  let vice_president_choices_girl_pres := boys - senior_boys
  let total_ways :=
    (senior_boys * vice_president_choices_boy_pres) + 
    (senior_girls * vice_president_choices_girl_pres)
  total_ways = 44 := 
by
  sorry

end choose_president_and_vice_president_l108_108572


namespace clock_angle_at_3_40_l108_108888

theorem clock_angle_at_3_40
  (hour_position : ℕ → ℝ)
  (minute_position : ℕ → ℝ)
  (h_hour : hour_position 3 = 3 * 30)
  (h_minute : minute_position 40 = 40 * 6)
  : abs (minute_position 40 - (hour_position 3 + 20 * 30 / 60)) = 130 :=
by
  -- Insert proof here
  sorry

end clock_angle_at_3_40_l108_108888


namespace problem_statement_l108_108925

noncomputable def f (x : ℝ) : ℝ :=
  if (0 < x ∧ x < 1) then 4^x
  else if (-1 < x ∧ x < 0) then -4^(-x)
  else if (-2 < x ∧ x < -1) then -4^(x + 2)
  else if (1 < x ∧ x < 2) then 4^(x - 2)
  else 0

theorem problem_statement :
  (f (-5 / 2) + f 1) = -2 :=
sorry

end problem_statement_l108_108925


namespace trig_comparison_l108_108856

theorem trig_comparison 
  (a : ℝ) (b : ℝ) (c : ℝ) :
  a = Real.sin (3 * Real.pi / 5) → 
  b = Real.cos (2 * Real.pi / 5) → 
  c = Real.tan (2 * Real.pi / 5) → 
  b < a ∧ a < c :=
by
  intro ha hb hc
  sorry

end trig_comparison_l108_108856


namespace students_basketball_cricket_l108_108535

theorem students_basketball_cricket (A B: ℕ) (AB: ℕ):
  A = 12 →
  B = 8 →
  AB = 3 →
  (A + B - AB) = 17 :=
by
  intros
  sorry

end students_basketball_cricket_l108_108535


namespace twice_a_minus_4_nonnegative_l108_108132

theorem twice_a_minus_4_nonnegative (a : ℝ) : 2 * a - 4 ≥ 0 ↔ 2 * a - 4 = 0 ∨ 2 * a - 4 > 0 := 
by
  sorry

end twice_a_minus_4_nonnegative_l108_108132


namespace range_of_x_l108_108230

theorem range_of_x (x : ℝ) : (∃ y : ℝ, y = (2 / (Real.sqrt (x - 1)))) → (x > 1) :=
by
  sorry

end range_of_x_l108_108230


namespace average_of_roots_l108_108611

theorem average_of_roots (p q : ℝ) (h : ∀ r : ℝ, r^2 * (3 * p) + r * (-6 * p) + q = 0 → ∃ a b : ℝ, r = a ∨ r = b) : 
  ∀ (r1 r2 : ℝ), (3 * p) * r1^2 + (-6 * p) * r1 + q = 0 ∧ (3 * p) * r2^2 + (-6 * p) * r2 + q = 0 → 
  (r1 + r2) / 2 = 1 :=
by {
  sorry
}

end average_of_roots_l108_108611


namespace intersection_of_A_and_B_l108_108154

def A : Set (ℝ × ℝ) := {p | p.snd = 3 * p.fst - 2}
def B : Set (ℝ × ℝ) := {p | p.snd = p.fst ^ 2}

theorem intersection_of_A_and_B :
  {p : ℝ × ℝ | p ∈ A ∧ p ∈ B} = {(1, 1), (2, 4)} :=
by
  sorry

end intersection_of_A_and_B_l108_108154


namespace total_time_spent_l108_108493

-- Define time spent on each step
def time_first_step : ℕ := 30
def time_second_step : ℕ := time_first_step / 2
def time_third_step : ℕ := time_first_step + time_second_step

-- Prove the total time spent
theorem total_time_spent : 
  time_first_step + time_second_step + time_third_step = 90 := by
  sorry

end total_time_spent_l108_108493


namespace impossible_to_get_60_pieces_possible_to_get_more_than_60_pieces_l108_108469

theorem impossible_to_get_60_pieces :
  ¬ ∃ (n m : ℕ), 1 + 7 * n + 11 * m = 60 :=
sorry

theorem possible_to_get_more_than_60_pieces :
  ∀ k > 60, ∃ (n m : ℕ), 1 + 7 * n + 11 * m = k :=
sorry

end impossible_to_get_60_pieces_possible_to_get_more_than_60_pieces_l108_108469


namespace perfect_square_polynomial_l108_108035

theorem perfect_square_polynomial (m : ℝ) :
  (∃ a b : ℝ, (a * x + b)^2 = m - 10 * x + x^2) → m = 25 :=
sorry

end perfect_square_polynomial_l108_108035


namespace find_m_l108_108168

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-1, 2)

-- Define the function to calculate m * a + b
def m_a_plus_b (m : ℝ) : ℝ × ℝ := (2 * m - 1, 3 * m + 2)

-- Define the vector a - 2 * b
def a_minus_2b : ℝ × ℝ := (4, -1)

-- Define the condition for parallelism
def parallel (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, v = (k * w.1, k * w.2)

-- The theorem that states the equivalence
theorem find_m (m : ℝ) (H : parallel (m_a_plus_b m) a_minus_2b) : m = -1/2 :=
by
  sorry

end find_m_l108_108168


namespace car_trip_time_l108_108457

theorem car_trip_time (walking_mixed: 1.5 = 1.25 + x) 
                      (walking_both: 2.5 = 2 * 1.25) : 
  2 * x * 60 = 30 :=
by sorry

end car_trip_time_l108_108457


namespace problem_1_problem_2_l108_108696

open Set Real

noncomputable def A : Set ℝ := {x | x^2 - 3 * x - 18 ≤ 0}

noncomputable def B (m : ℝ) : Set ℝ := {x | m - 8 ≤ x ∧ x ≤ m + 4}

theorem problem_1 : (m = 3) → ((compl A) ∩ (B m) = {x | (-5 ≤ x ∧ x < -3) ∨ (6 < x ∧ x ≤ 7)}) :=
by
  sorry

theorem problem_2 : (A ∩ (B m) = A) → (2 ≤ m ∧ m ≤ 5) :=
by
  sorry

end problem_1_problem_2_l108_108696


namespace part_a_l108_108312

theorem part_a (n : ℕ) (hn : 0 < n) : 
  ∃ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x^(n-1) + y^n = z^(n+1) :=
sorry

end part_a_l108_108312


namespace distance_origin_to_point_l108_108179

theorem distance_origin_to_point :
  let distance (x1 y1 x2 y2 : ℝ) := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  distance 0 0 8 (-15) = 17 :=
by
  sorry

end distance_origin_to_point_l108_108179


namespace distinct_values_in_expression_rearrangement_l108_108162

theorem distinct_values_in_expression_rearrangement : 
  ∀ (exp : ℕ), exp = 3 → 
  (∃ n : ℕ, n = 3 ∧ 
    let a := exp ^ (exp ^ exp)
    let b := exp ^ ((exp ^ exp) ^ exp)
    let c := ((exp ^ exp) ^ exp) ^ exp
    let d := (exp ^ (exp ^ exp)) ^ exp
    let e := (exp ^ exp) ^ (exp ^ exp)
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) :=
by
  sorry

end distinct_values_in_expression_rearrangement_l108_108162


namespace larger_number_l108_108454

variables (x y : ℕ)

theorem larger_number (h1 : x + y = 47) (h2 : x - y = 7) : x = 27 :=
sorry

end larger_number_l108_108454


namespace subtraction_of_7_305_from_neg_3_219_l108_108891

theorem subtraction_of_7_305_from_neg_3_219 :
  -3.219 - 7.305 = -10.524 :=
by
  -- The proof would go here
  sorry

end subtraction_of_7_305_from_neg_3_219_l108_108891


namespace customer_ordered_bags_l108_108096

def bags_per_batch : Nat := 10
def initial_bags : Nat := 20
def days : Nat := 4
def batches_per_day : Nat := 1

theorem customer_ordered_bags : 
  initial_bags + days * batches_per_day * bags_per_batch = 60 :=
by
  sorry

end customer_ordered_bags_l108_108096


namespace part1_part2_part3_l108_108012

noncomputable def distance (p q : ℝ × ℝ) : ℝ := 
  real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

noncomputable def line_eq (x y k : ℝ) : Prop := 
  y = k * x

noncomputable def line_distance (l_coeff: ℝ × ℝ × ℝ) (p: ℝ × ℝ) : ℝ := 
  abs (l_coeff.1 * p.1 + l_coeff.2 * p.2 + l_coeff.3) / real.sqrt (l_coeff.1 ^ 2 + l_coeff.2 ^ 2)

theorem part1 (hC : (3, -2) = (3, -2)) : 
  ∀ x y: ℝ, (x - 3)^2 + (y + 2)^2 = 25 := 
sorry

theorem part2 
  (tangent_point : ℝ × ℝ := (0, 3)) 
  (center := (3, -2)) 
  (radius := 5) : 
  ∀ x y k: ℝ, (line_eq x y k → 
  line_distance (1, -k, 3 * k - 3) center = radius 
  → (15 * x - 8 * y + 24 = 0 ∨ y = 3)) := 
sorry

theorem part3 
  (center := (3, -2))
  (radius := 5)
  (line_coeff := (3, 4)) 
  (dist := 1) : 
  ∀ (m : ℝ),  
  |line_distance (3, 4, m) center - radius| = dist → 
  (m = 21 ∨ m = 19) := 
sorry

end part1_part2_part3_l108_108012


namespace find_n_divisible_by_35_l108_108143

-- Define the five-digit number for some digit n
def num (n : ℕ) : ℕ := 80000 + n * 1000 + 975

-- Define the conditions
def divisible_by_5 (d : ℕ) : Prop := d % 5 = 0
def divisible_by_7 (d : ℕ) : Prop := d % 7 = 0
def divisible_by_35 (d : ℕ) : Prop := divisible_by_5 d ∧ divisible_by_7 d

-- Statement of the problem for proving given conditions and the correct answer
theorem find_n_divisible_by_35 : ∃ (n : ℕ), (num n % 35 = 0) ∧ n = 6 := by
  sorry

end find_n_divisible_by_35_l108_108143


namespace factor_expression_l108_108943

theorem factor_expression (x : ℝ) : 
  75 * x^11 + 225 * x^22 = 75 * x^11 * (1 + 3 * x^11) :=
by sorry

end factor_expression_l108_108943


namespace f_at_5_l108_108064

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the function definition

axiom odd_function (f: ℝ → ℝ) : ∀ x : ℝ, f (-x) = -f x
axiom functional_equation (f: ℝ → ℝ) : ∀ x : ℝ, f (x + 1) + f x = 0

theorem f_at_5 : f 5 = 0 :=
by {
  -- Proof to be provided here
  sorry
}

end f_at_5_l108_108064


namespace IAOC_seating_arrangements_l108_108433

-- Definitions based on the conditions
def numMercury : ℕ := 4
def numVenus : ℕ := 4
def numEarth : ℕ := 4
def numChairs : ℕ := 12
def chairMercury : ℕ := 1
def chairVenus : ℕ := 12
def numArrangements := 1

-- Given conditions are translated to functions to check immediate left seats
noncomputable def isValidArrangement (arrangement : List ℕ) : Bool := 
  let chairs := (List.range numChairs).rotate' chairMercury
  ¬ (arrangement.nth! ((chairs.indexOf chairVenus) - 1) = numMercury) &&
  ¬ (arrangement.nth! ((chairs.indexOf numEarth) - 1) = numVenus) &&
  ¬ (arrangement.nth! ((chairs.indexOf numMercury) - 1) = numEarth)

-- Main theorem statement translated to Lean 4
theorem IAOC_seating_arrangements (N : ℕ) 
  (validArrangements : ℕ := 
    List.permutations [numMercury, numVenus, numEarth]
    .filter isValidArrangement 
    .length)
  : validArrangements * (fac numMercury) * (fac numVenus) * (fac numEarth) = N * (4!) ^ 3 := 
  by {
    have h : validArrangements = 216, sorry,
    rw h,
    ring
  }

end IAOC_seating_arrangements_l108_108433


namespace sum_first_8_terms_eq_8_l108_108822

noncomputable def arithmetic_sequence_sum (n : ℕ) (a₁ d : ℤ) : ℤ :=
  n / 2 * (2 * a₁ + (n - 1) * d)

theorem sum_first_8_terms_eq_8
  (a : ℕ → ℤ)
  (h_arith_seq : ∀ n : ℕ, a (n + 1) = a 1 + ↑n * d)
  (h_a1 : a 1 = 8)
  (h_a4_a6 : a 4 + a 6 = 0) :
  arithmetic_sequence_sum 8 8 (-2) = 8 := 
by
  sorry

end sum_first_8_terms_eq_8_l108_108822


namespace carla_drinks_water_l108_108348

-- Definitions from the conditions
def total_liquid (s w : ℕ) : Prop := s + w = 54
def soda_water_relation (s w : ℕ) : Prop := s = 3 * w - 6

-- Proof statement
theorem carla_drinks_water : ∀ (s w : ℕ), total_liquid s w ∧ soda_water_relation s w → w = 15 :=
by
  intros s w h,
  sorry

end carla_drinks_water_l108_108348


namespace sqrt_49_times_sqrt_25_eq_7sqrt5_l108_108265

theorem sqrt_49_times_sqrt_25_eq_7sqrt5 :
  (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  have h1 : Real.sqrt 25 = 5 := by sorry
  have h2 : 49 * 5 = 245 := by sorry
  have h3 : Real.sqrt 245 = 7 * Real.sqrt 5 := by sorry
  sorry

end sqrt_49_times_sqrt_25_eq_7sqrt5_l108_108265


namespace guppies_to_angelfish_ratio_l108_108921

noncomputable def goldfish : ℕ := 8
noncomputable def angelfish : ℕ := goldfish + 4
noncomputable def total_fish : ℕ := 44
noncomputable def guppies : ℕ := total_fish - (goldfish + angelfish)

theorem guppies_to_angelfish_ratio :
    guppies / angelfish = 2 := by
    sorry

end guppies_to_angelfish_ratio_l108_108921


namespace NataliesSisterInitialDiaries_l108_108067

theorem NataliesSisterInitialDiaries (D : ℕ)
  (h1 : 2 * D - (1 / 4) * 2 * D = 18) : D = 12 :=
by sorry

end NataliesSisterInitialDiaries_l108_108067


namespace heartsuit_example_l108_108924

def heartsuit (a b : ℤ) : ℤ := a * b^3 - 2 * b + 3

theorem heartsuit_example : heartsuit 2 3 = 51 :=
by
  sorry

end heartsuit_example_l108_108924


namespace relationship_y1_y2_y3_l108_108967

theorem relationship_y1_y2_y3 :
  let y1 := -(((-4):ℝ)^2) + 5 in
  let y2 := -(((-1):ℝ)^2) + 5 in
  let y3 := -((2:ℝ)^2) + 5 in
  y2 > y3 ∧ y3 > y1 :=
by
  sorry

end relationship_y1_y2_y3_l108_108967


namespace principal_amount_is_approx_1200_l108_108222

noncomputable def find_principal_amount : Real :=
  let R := 0.10
  let n := 2
  let T := 1
  let SI (P : Real) := P * R * T / 100
  let CI (P : Real) := P * ((1 + R / n) ^ (n * T)) - P
  let diff (P : Real) := CI P - SI P
  let target_diff := 2.999999999999936
  let P := target_diff / (0.1025 - 0.10)
  P

theorem principal_amount_is_approx_1200 : abs (find_principal_amount - 1200) < 0.0001 := 
by
  sorry

end principal_amount_is_approx_1200_l108_108222


namespace functional_equation_solution_l108_108642

theorem functional_equation_solution (f : ℚ → ℚ)
  (H : ∀ x y : ℚ, f (x + y) + f (x - y) = 2 * f x + 2 * f y) :
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x^2 :=
by
  sorry

end functional_equation_solution_l108_108642


namespace gcd_2750_9450_l108_108889

theorem gcd_2750_9450 : Nat.gcd 2750 9450 = 50 := by
  sorry

end gcd_2750_9450_l108_108889


namespace factor_t_sq_minus_64_l108_108940

def isDifferenceOfSquares (a b : Int) : Prop := a = b^2

theorem factor_t_sq_minus_64 (t : Int) (h : isDifferenceOfSquares 64 8) : (t^2 - 64) = (t - 8) * (t + 8) := by
  sorry

end factor_t_sq_minus_64_l108_108940


namespace average_last_30_l108_108080

theorem average_last_30 (avg_first_40 : ℝ) 
  (avg_all_70 : ℝ) 
  (sum_first_40 : ℝ := 40 * avg_first_40)
  (sum_all_70 : ℝ := 70 * avg_all_70) 
  (total_results: ℕ := 70):
  (30 : ℝ) * (40: ℝ) + (30: ℝ) * (40: ℝ) = 70 * 34.285714285714285 :=
by
  sorry

end average_last_30_l108_108080


namespace rational_root_of_polynomial_l108_108926

-- Polynomial definition
def P (x : ℚ) : ℚ := 3 * x^4 - 7 * x^3 + 4 * x^2 + 6 * x - 8

-- Theorem statement
theorem rational_root_of_polynomial : ∀ x : ℚ, P x = 0 ↔ x = -1 :=
by
  sorry

end rational_root_of_polynomial_l108_108926


namespace y_relation_l108_108968

theorem y_relation (y1 y2 y3 : ℝ) : 
  (-4, y1) ∈ set_of (λ p : ℝ × ℝ, p.snd = -p.fst^2 + 5) ∧
  (-1, y2) ∈ set_of (λ p : ℝ × ℝ, p.snd = -p.fst^2 + 5) ∧
  (2, y3) ∈ set_of (λ p : ℝ × ℝ, p.snd = -p.fst^2 + 5) →
  y2 > y3 ∧ y3 > y1 :=
begin
  sorry
end

end y_relation_l108_108968


namespace sqrt_nested_l108_108305

theorem sqrt_nested : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_nested_l108_108305


namespace largest_divisor_of_square_difference_l108_108419

theorem largest_divisor_of_square_difference (m n : ℤ) (hm : m % 2 = 0) (hn : n % 2 = 0) (h : n < m) : 
  ∃ d, ∀ m n, (m % 2 = 0) → (n % 2 = 0) → (n < m) → d ∣ (m^2 - n^2) ∧ ∀ k, (∀ m n, (m % 2 = 0) → (n % 2 = 0) → (n < m) → k ∣ (m^2 - n^2)) → k ≤ d :=
sorry

end largest_divisor_of_square_difference_l108_108419


namespace quadratic_solution_l108_108142

theorem quadratic_solution (m : ℝ) (x₁ x₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂) 
  (h_roots : ∀ x, 2 * x^2 + 4 * m * x + m = 0 ↔ x = x₁ ∨ x = x₂) 
  (h_sum_squares : x₁^2 + x₂^2 = 3 / 16) :
  m = -1 / 8 :=
by
  sorry

end quadratic_solution_l108_108142


namespace probability_one_defective_l108_108756

theorem probability_one_defective (g d : Nat) (h1 : g = 3) (h2 : d = 1) : 
  let total_combinations := (g + d).choose 2
  let favorable_outcomes := g * d
  favorable_outcomes / total_combinations = 1 / 2 := by
sorry

end probability_one_defective_l108_108756


namespace find_biology_marks_l108_108126

variables (e m p c b : ℕ)
variable (a : ℝ)

def david_marks_in_biology : Prop :=
  e = 72 ∧
  m = 45 ∧
  p = 72 ∧
  c = 77 ∧
  a = 68.2 ∧
  (e + m + p + c + b) / 5 = a

theorem find_biology_marks (h : david_marks_in_biology e m p c b a) : b = 75 :=
sorry

end find_biology_marks_l108_108126


namespace tan_alpha_through_point_l108_108976

theorem tan_alpha_through_point (α : ℝ) (x y : ℝ) (h : (x, y) = (3, 4)) : Real.tan α = 4 / 3 :=
sorry

end tan_alpha_through_point_l108_108976


namespace number_of_m_values_l108_108545

theorem number_of_m_values (m : ℕ) (h1 : 4 * m > 11) (h2 : m < 12) : 
  11 - 3 + 1 = 9 := 
sorry

end number_of_m_values_l108_108545


namespace measure_of_each_interior_angle_of_regular_octagon_l108_108462

theorem measure_of_each_interior_angle_of_regular_octagon 
  (n : ℕ) (h_n : n = 8) : 
  let sum_of_interior_angles := 180 * (n - 2) in
  let measure_of_interior_angle := sum_of_interior_angles / n in
  measure_of_interior_angle = 135 :=
by
  sorry

end measure_of_each_interior_angle_of_regular_octagon_l108_108462


namespace hypotenuse_of_45_45_90_triangle_l108_108566

noncomputable def leg_length : ℝ := 15
noncomputable def angle_opposite_leg : ℝ := Real.pi / 4  -- 45 degrees in radians

theorem hypotenuse_of_45_45_90_triangle (h_leg : ℝ) (h_angle : ℝ) 
  (h_leg_cond : h_leg = leg_length) (h_angle_cond : h_angle = angle_opposite_leg) :
  ∃ h_hypotenuse : ℝ, h_hypotenuse = h_leg * Real.sqrt 2 :=
sorry

end hypotenuse_of_45_45_90_triangle_l108_108566


namespace how_many_more_rolls_needed_l108_108957

variable (total_needed sold_to_grandmother sold_to_uncle sold_to_neighbor : ℕ)

theorem how_many_more_rolls_needed (h1 : total_needed = 45)
  (h2 : sold_to_grandmother = 1)
  (h3 : sold_to_uncle = 10)
  (h4 : sold_to_neighbor = 6) :
  total_needed - (sold_to_grandmother + sold_to_uncle + sold_to_neighbor) = 28 := by
  sorry

end how_many_more_rolls_needed_l108_108957


namespace quadratic_has_two_distinct_roots_l108_108148

theorem quadratic_has_two_distinct_roots (a b c : ℝ) (h : 2016 + a^2 + a * c < a * b) : 
  (b^2 - 4 * a * c) > 0 :=
by {
  sorry
}

end quadratic_has_two_distinct_roots_l108_108148


namespace calculate_t_minus_d_l108_108887

def tom_pays : ℕ := 150
def dorothy_pays : ℕ := 190
def sammy_pays : ℕ := 240
def nancy_pays : ℕ := 320
def total_expenses := tom_pays + dorothy_pays + sammy_pays + nancy_pays
def individual_share := total_expenses / 4
def tom_needs_to_pay := individual_share - tom_pays
def dorothy_needs_to_pay := individual_share - dorothy_pays
def sammy_should_receive := sammy_pays - individual_share
def nancy_should_receive := nancy_pays - individual_share
def t := tom_needs_to_pay
def d := dorothy_needs_to_pay

theorem calculate_t_minus_d : t - d = 40 :=
by
  sorry

end calculate_t_minus_d_l108_108887


namespace cost_of_milkshake_is_correct_l108_108689

-- Definitions related to the problem conditions
def initial_amount : ℕ := 15
def spent_on_cupcakes : ℕ := initial_amount * (1 / 3)
def remaining_after_cupcakes : ℕ := initial_amount - spent_on_cupcakes
def spent_on_sandwich : ℕ := remaining_after_cupcakes * (20 / 100)
def remaining_after_sandwich : ℕ := remaining_after_cupcakes - spent_on_sandwich
def remaining_after_milkshake : ℕ := 4
def cost_of_milkshake : ℕ := remaining_after_sandwich - remaining_after_milkshake

-- The theorem stating the equivalent proof problem
theorem cost_of_milkshake_is_correct :
  cost_of_milkshake = 4 :=
sorry

end cost_of_milkshake_is_correct_l108_108689


namespace scientific_notation_of_105000_l108_108206

theorem scientific_notation_of_105000 : (105000 : ℝ) = 1.05 * 10^5 := 
by {
  sorry
}

end scientific_notation_of_105000_l108_108206


namespace sqrt_mul_sqrt_l108_108256

theorem sqrt_mul_sqrt (a b c : ℝ) (hac : a = 49) (hbc : b = sqrt 25) (hc : c = 7 * sqrt 5) : sqrt (a * b) = c := 
by
  sorry

end sqrt_mul_sqrt_l108_108256


namespace pizza_slice_volume_l108_108328

-- Define the parameters given in the conditions
def pizza_thickness : ℝ := 0.5
def pizza_diameter : ℝ := 16.0
def num_slices : ℝ := 16.0

-- Define the volume of one slice
theorem pizza_slice_volume : (π * (pizza_diameter / 2) ^ 2 * pizza_thickness / num_slices) = 2 * π := by
  sorry

end pizza_slice_volume_l108_108328


namespace find_complementary_angle_l108_108158

theorem find_complementary_angle (angle1 angle2 : ℝ) (h1 : angle1 + angle2 = 90) (h2 : angle1 = 25) : angle2 = 65 := 
by 
  sorry

end find_complementary_angle_l108_108158


namespace sqrt_49_times_sqrt_25_eq_7sqrt5_l108_108263

theorem sqrt_49_times_sqrt_25_eq_7sqrt5 :
  (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  have h1 : Real.sqrt 25 = 5 := by sorry
  have h2 : 49 * 5 = 245 := by sorry
  have h3 : Real.sqrt 245 = 7 * Real.sqrt 5 := by sorry
  sorry

end sqrt_49_times_sqrt_25_eq_7sqrt5_l108_108263


namespace radius_of_cylinder_is_correct_l108_108110

/-- 
  A right circular cylinder is inscribed in a right circular cone such that:
  - The diameter of the cylinder is equal to its height.
  - The cone has a diameter of 8.
  - The cone has an altitude of 10.
  - The axes of the cylinder and cone coincide.
  Prove that the radius of the cylinder is 20/9.
-/
theorem radius_of_cylinder_is_correct :
  ∀ (r : ℚ), 
    (2 * r = 8 - 2 * r ∧ 10 - 2 * r = (10 / 4) * r) → 
    r = 20 / 9 :=
by
  intro r
  intro h
  sorry

end radius_of_cylinder_is_correct_l108_108110


namespace initial_pairs_l108_108851

variable (p1 p2 p3 p4 p_initial : ℕ)

def week1_pairs := 12
def week2_pairs := week1_pairs + 4
def week3_pairs := (week1_pairs + week2_pairs) / 2
def week4_pairs := week3_pairs - 3
def total_pairs := 57

theorem initial_pairs :
  let p1 := week1_pairs
  let p2 := week2_pairs
  let p3 := week3_pairs
  let p4 := week4_pairs
  p1 + p2 + p3 + p4 + p_initial = 57 → p_initial = 4 :=
by
  sorry

end initial_pairs_l108_108851


namespace base_b_three_digit_count_l108_108845

-- Define the condition that counts the valid three-digit numbers in base b
def num_three_digit_numbers (b : ℕ) : ℕ :=
  (b - 1) ^ 2 * b

-- Define the specific problem statement
theorem base_b_three_digit_count :
  num_three_digit_numbers 4 = 72 :=
by
  -- Proof skipped as per the instruction
  sorry

end base_b_three_digit_count_l108_108845


namespace sqrt_product_l108_108273

theorem sqrt_product (a b : ℝ) (h1 : a = 49) (h2 : b = 25) : sqrt (a * sqrt b) = 7 * sqrt 5 :=
by sorry

end sqrt_product_l108_108273


namespace sqrt_product_l108_108274

theorem sqrt_product (a b : ℝ) (h1 : a = 49) (h2 : b = 25) : sqrt (a * sqrt b) = 7 * sqrt 5 :=
by sorry

end sqrt_product_l108_108274


namespace repeated_letter_adjacent_probability_l108_108614

open Finset

noncomputable def probability_repeated_adjacent
  (letters : Finset Char) (X : Char) (H : X ∈ letters) (k : Nat) : ℝ :=
  let n := 10
  let factorial (n : Nat) := if n = 0 then 1 else n * factorial (n - 1)
  let total_arrangements := factorial 10 / factorial 2
  let adjacent_arrangements := factorial 9
  adjacent_arrangements / total_arrangements

theorem repeated_letter_adjacent_probability 
  (letters : Finset Char) (X : Char) (H : X ∈ letters) :
  letters = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', X } → 
  probability_repeated_adjacent letters X H 10 = 0.2 :=
by
  intro h1
  rw [probability_repeated_adjacent, h1]
  sorry

end repeated_letter_adjacent_probability_l108_108614


namespace kerosene_consumption_reduction_l108_108899

variable (P C : ℝ)

/-- In the new budget, with the price of kerosene oil rising by 25%, 
    we need to prove that consumption must be reduced by 20% to maintain the same expenditure. -/
theorem kerosene_consumption_reduction (h : 1.25 * P * C_new = P * C) : C_new = 0.8 * C := by
  sorry

end kerosene_consumption_reduction_l108_108899


namespace arithmetic_sequences_sum_l108_108202

theorem arithmetic_sequences_sum
  (a b : ℕ → ℕ)
  (d1 d2 : ℕ)
  (h1 : ∀ n, a (n + 1) = a n + d1)
  (h2 : ∀ n, b (n + 1) = b n + d2)
  (h3 : a 1 + b 1 = 7)
  (h4 : a 3 + b 3 = 21) :
  a 5 + b 5 = 35 :=
sorry

end arithmetic_sequences_sum_l108_108202


namespace double_given_number_l108_108527

def given_number : ℝ := 1.2 * 10^6

def double_number (x: ℝ) : ℝ := x * 2

theorem double_given_number : double_number given_number = 2.4 * 10^6 :=
by sorry

end double_given_number_l108_108527


namespace chromosome_structure_l108_108787

-- Definitions related to the conditions of the problem
def chromosome : Type := sorry  -- Define type for chromosome (hypothetical representation)
def has_centromere (c : chromosome) : Prop := sorry  -- Predicate indicating a chromosome has centromere
def contains_one_centromere (c : chromosome) : Prop := sorry  -- Predicate indicating a chromosome contains one centromere
def has_one_chromatid (c : chromosome) : Prop := sorry  -- Predicate indicating a chromosome has one chromatid
def has_two_chromatids (c : chromosome) : Prop := sorry  -- Predicate indicating a chromosome has two chromatids
def is_chromatin (c : chromosome) : Prop := sorry  -- Predicate indicating a chromosome is chromatin

-- Define the problem statement
theorem chromosome_structure (c : chromosome) :
  contains_one_centromere c ∧ ¬has_one_chromatid c ∧ ¬has_two_chromatids c ∧ ¬is_chromatin c := sorry

end chromosome_structure_l108_108787


namespace proof_of_diagonal_length_l108_108872

noncomputable def length_of_diagonal (d : ℝ) : Prop :=
  d^2 = 325 ∧ 17^2 + 36 = 325

theorem proof_of_diagonal_length (d : ℝ) : length_of_diagonal d → d = 5 * Real.sqrt 13 :=
by
  intro h
  sorry

end proof_of_diagonal_length_l108_108872


namespace sqrt_expression_eval_l108_108795

theorem sqrt_expression_eval :
  (Real.sqrt 8) + (Real.sqrt (1 / 2)) + (Real.sqrt 3 - 1) ^ 2 + (Real.sqrt 6 / (1 / 2 * Real.sqrt 2)) = (5 / 2) * Real.sqrt 2 + 4 := 
by
  sorry

end sqrt_expression_eval_l108_108795


namespace coordinates_of_point_P_l108_108873

-- Define the function y = x^3
def cubic (x : ℝ) : ℝ := x^3

-- Define the derivative of the function
def derivative_cubic (x : ℝ) : ℝ := 3 * x^2

-- Define the condition for the slope of the tangent line to the function at point P
def slope_tangent_line := 3

-- Prove that the coordinates of point P are (1, 1) or (-1, -1) when the slope of the tangent line is 3
theorem coordinates_of_point_P (x : ℝ) (y : ℝ) 
    (h1 : y = cubic x) 
    (h2 : derivative_cubic x = slope_tangent_line) : 
    (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) :=
by
  sorry

end coordinates_of_point_P_l108_108873


namespace sqrt_nested_l108_108306

theorem sqrt_nested : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_nested_l108_108306


namespace probability_of_sum_digits_eq_9_l108_108815

def probability_sum_digits_eq_9 (s : Finset ℕ) (n : ℕ) : ℚ :=
  let possible_digits := {1, 2, 3, 4, 5}
  let possible_combinations := finset.powersetLen 3 possible_digits
  let digit_sum_9_combinations := possible_combinations.filter (λ x, x.sum id = 9)
  (digit_sum_9_combinations.card : ℚ) / (possible_combined_digit_permutations.possible_digits.card^(n:ℚ))

theorem probability_of_sum_digits_eq_9 :
  probability_sum_digits_eq_9 {1, 2, 3, 4, 5} 3 = 19 / 125 :=
by
  sorry

end probability_of_sum_digits_eq_9_l108_108815


namespace necessary_sufficient_condition_l108_108971

theorem necessary_sufficient_condition (a b x_0 : ℝ) (h : a > 0) :
  (x_0 = b / a) ↔ (∀ x : ℝ, (1/2) * a * x^2 - b * x ≥ (1/2) * a * x_0^2 - b * x_0) :=
sorry

end necessary_sufficient_condition_l108_108971


namespace complex_number_property_l108_108394

noncomputable def imaginary_unit : Complex := Complex.I

theorem complex_number_property (n : ℕ) (hn : 4^n = 256) : (1 + imaginary_unit)^n = -4 :=
by
  sorry

end complex_number_property_l108_108394


namespace count_valid_numbers_l108_108732

theorem count_valid_numbers : 
  let count_A := 10 
  let count_B := 2 
  count_A * count_B = 20 :=
by 
  let count_A := 10
  let count_B := 2
  have : count_A * count_B = 20 := by norm_num
  exact this

end count_valid_numbers_l108_108732


namespace sqrt_mul_sqrt_l108_108249

theorem sqrt_mul_sqrt (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : sqrt (a * sqrt b) = sqrt a * sqrt (sqrt b) := by
  sorry

example : sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  have h1 : sqrt 25 = 5 := by 
    norm_num
  have h2 : sqrt (49 * 5) = sqrt 245 := by  
    unfold sqrt
  have h3 : sqrt 245 = 7 * sqrt 5 := by 
    norm_num
  rw [h1, h2, h3]
  norm_num

end sqrt_mul_sqrt_l108_108249


namespace blackjack_payment_l108_108904

def casino_payout (b: ℤ) (r: ℤ): ℤ := b + r
def blackjack_payout (bet: ℤ) (ratio_numerator: ℤ) (ratio_denominator: ℤ): ℤ :=
  (ratio_numerator * bet) / ratio_denominator

theorem blackjack_payment (bet: ℤ) (ratio_numerator: ℤ) (ratio_denominator: ℤ) (payout: ℤ):
  ratio_numerator = 3 → 
  ratio_denominator = 2 → 
  bet = 40 →
  payout = blackjack_payout bet ratio_numerator ratio_denominator → 
  casino_payout bet payout = 100 :=
by
  sorry

end blackjack_payment_l108_108904


namespace fraction_zero_when_x_eq_3_l108_108799

theorem fraction_zero_when_x_eq_3 : ∀ x : ℝ, x = 3 → (x^6 - 54 * x^3 + 729) / (x^3 - 27) = 0 :=
by
  intro x hx
  rw [hx]
  sorry

end fraction_zero_when_x_eq_3_l108_108799


namespace part_a_part_b_l108_108769

theorem part_a (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : 16 * x * y * z = (x + y)^2 * (x + z)^2) : x + y + z ≤ 4 :=
sorry

theorem part_b : ∃ (S : Set (ℚ × ℚ × ℚ)), S.Countable ∧
  (∀ (x y z : ℚ), (x, y, z) ∈ S → 0 < x ∧ 0 < y ∧ 0 < z ∧ 16 * x * y * z = (x + y)^2 * (x + z)^2 ∧ x + y + z = 4) ∧ 
  Infinite S :=
sorry

end part_a_part_b_l108_108769


namespace largest_apartment_size_l108_108624

theorem largest_apartment_size (rent_per_sqft : ℝ) (budget : ℝ) (s : ℝ) :
  rent_per_sqft = 0.9 →
  budget = 630 →
  s = budget / rent_per_sqft →
  s = 700 :=
by
  sorry

end largest_apartment_size_l108_108624


namespace convex_polygons_on_circle_l108_108506

theorem convex_polygons_on_circle:
  let points := 15 in
  ∑ i in finset.range (points + 1), choose points i - (choose points 0 + choose points 1 + choose points 2 + choose points 3) = 32192 :=
begin
  sorry
end

end convex_polygons_on_circle_l108_108506


namespace y_is_one_y_is_neg_two_thirds_l108_108167

-- Define the vectors a and b
def vector_a : ℝ × ℝ := (1, 3)
def vector_b (y : ℝ) : ℝ × ℝ := (2, y)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Prove y = 1 given dot_product(vector_a, vector_b(y)) = 5
theorem y_is_one (h : dot_product vector_a (vector_b y) = 5) : y = 1 :=
by
  -- We assume the proof (otherwise it would go here)
  sorry

-- Prove y = -2/3 given |vector_a + vector_b(y)| = |vector_a - vector_b(y)|
theorem y_is_neg_two_thirds (h : (vector_a.1 + (vector_b y).1)^2 + (vector_a.2 + (vector_b y).2)^2 =
                                (vector_a.1 - (vector_b y).1)^2 + (vector_a.2 - (vector_b y).2)^2) : y = -2/3 :=
by
  -- We assume the proof (otherwise it would go here)
  sorry

end y_is_one_y_is_neg_two_thirds_l108_108167


namespace man_salary_l108_108470

variable (S : ℝ)

theorem man_salary (S : ℝ) (h1 : S - (1/3) * S - (1/4) * S - (1/5) * S = 1760) : S = 8123 := 
by 
  sorry

end man_salary_l108_108470


namespace sqrt_mul_sqrt_l108_108254

theorem sqrt_mul_sqrt (a b c : ℝ) (hac : a = 49) (hbc : b = sqrt 25) (hc : c = 7 * sqrt 5) : sqrt (a * b) = c := 
by
  sorry

end sqrt_mul_sqrt_l108_108254


namespace sqrt_49_times_sqrt_25_l108_108290

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 :=
by
  sorry

end sqrt_49_times_sqrt_25_l108_108290


namespace fewer_sevens_l108_108054

def seven_representation (n : ℕ) : ℕ :=
  (7 * (10^n - 1)) / 9

theorem fewer_sevens (n : ℕ) :
  ∃ m, m < n ∧ 
    (∃ expr : ℕ → ℕ, (∀ i < n, expr i = 7) ∧ seven_representation n = expr m) :=
sorry

end fewer_sevens_l108_108054


namespace exist_integers_not_div_by_7_l108_108412

theorem exist_integers_not_div_by_7 (k : ℕ) (hk : 0 < k) :
  ∃ (x y : ℤ), (¬ (7 ∣ x)) ∧ (¬ (7 ∣ y)) ∧ (x^2 + 6 * y^2 = 7^k) :=
sorry

end exist_integers_not_div_by_7_l108_108412


namespace probability_kwoes_non_intersect_breads_l108_108346

-- Define the total number of ways to pick 3 points from 7
def total_combinations : ℕ := Nat.choose 7 3

-- Define the number of ways to pick 3 consecutive points from 7
def favorable_combinations : ℕ := 7

-- Define the probability of non-intersection
def non_intersection_probability : ℚ := favorable_combinations / total_combinations

-- Assert the final required probability
theorem probability_kwoes_non_intersect_breads :
  non_intersection_probability = 1 / 5 :=
by
  sorry

end probability_kwoes_non_intersect_breads_l108_108346


namespace problem_statement_l108_108023

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 4}
def B : Set ℕ := {4, 5}
def C_U (B : Set ℕ) : Set ℕ := U \ B

-- Statement
theorem problem_statement : A ∩ (C_U B) = {2} :=
  sorry

end problem_statement_l108_108023


namespace grid_problem_l108_108541

theorem grid_problem 
  (n m : ℕ) 
  (h1 : ∀ (blue_cells : ℕ), blue_cells = m + n - 1 → (n * m ≠ 0) → (blue_cells = (n * m) / 2010)) :
  ∃ (k : ℕ), k = 96 :=
by
  sorry

end grid_problem_l108_108541


namespace largest_integer_base7_four_digits_l108_108420

theorem largest_integer_base7_four_digits :
  ∃ M : ℕ, (∀ m : ℕ, 7^3 ≤ m^2 ∧ m^2 < 7^4 → m ≤ M) ∧ M = 48 :=
sorry

end largest_integer_base7_four_digits_l108_108420


namespace temperature_difference_correct_l108_108038

def avg_high : ℝ := 9
def avg_low : ℝ := -5
def temp_difference : ℝ := avg_high - avg_low

theorem temperature_difference_correct : temp_difference = 14 := by
  sorry

end temperature_difference_correct_l108_108038


namespace sqrt_expression_simplified_l108_108268

theorem sqrt_expression_simplified :
  sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  sorry

end sqrt_expression_simplified_l108_108268


namespace hyperbola_center_l108_108003

theorem hyperbola_center (x y : ℝ) :
    9 * x^2 - 54 * x - 36 * y^2 + 360 * y - 864 = 0 → (x, y) = (3, 5) :=
by
  sorry

end hyperbola_center_l108_108003


namespace three_point_seven_five_as_fraction_l108_108595

theorem three_point_seven_five_as_fraction :
  (15 : ℚ) / 4 = 3.75 :=
sorry

end three_point_seven_five_as_fraction_l108_108595


namespace find_other_package_size_l108_108627

variable (total_coffee : ℕ)
variable (total_5_ounce_packages : ℕ)
variable (num_other_packages : ℕ)
variable (other_package_size : ℕ)

theorem find_other_package_size
  (h1 : total_coffee = 85)
  (h2 : total_5_ounce_packages = num_other_packages + 2)
  (h3 : num_other_packages = 5)
  (h4 : 5 * total_5_ounce_packages + other_package_size * num_other_packages = total_coffee) :
  other_package_size = 10 :=
sorry

end find_other_package_size_l108_108627


namespace trains_clear_time_l108_108240

theorem trains_clear_time
  (length_train1 : ℕ) (length_train2 : ℕ)
  (speed_train1_kmph : ℕ) (speed_train2_kmph : ℕ)
  (conversion_factor : ℕ) -- 5/18 as a rational number (for clarity)
  (approx_rel_speed : ℚ) -- Approximate relative speed 
  (total_distance : ℕ) 
  (total_time : ℚ) :
  length_train1 = 160 →
  length_train2 = 280 →
  speed_train1_kmph = 42 →
  speed_train2_kmph = 30 →
  conversion_factor = 5 / 18 →
  approx_rel_speed = (42 * (5 / 18) + 30 * (5 / 18)) →
  total_distance = length_train1 + length_train2 →
  total_time = total_distance / approx_rel_speed →
  total_time = 22 := 
by
  sorry

end trains_clear_time_l108_108240


namespace part1_part2_l108_108670

theorem part1 (m : ℝ) :
  ∀ x : ℝ, x^2 + ( (2 * m - 1) : ℝ) * x + m^2 = 0 → m ≤ 1 / 4 :=
sorry

theorem part2 (m : ℝ) 
  (h : ∀ x1 x2 : ℝ, (x1^2 + (2*m -1)*x1 + m^2 = 0) ∧ (x2^2 + (2*m -1)*x2 + m^2 = 0) ∧ (x1*x2 + x1 + x2 = 4)) :
    m = -1 :=
sorry

end part1_part2_l108_108670


namespace smallest_k_l108_108742

theorem smallest_k (k : ℕ) (h1 : k > 1) (h2 : k % 19 = 1) (h3 : k % 7 = 1) (h4 : k % 3 = 1) : k = 400 :=
by
  sorry

end smallest_k_l108_108742


namespace new_tax_rate_l108_108991

theorem new_tax_rate
  (old_rate : ℝ) (income : ℝ) (savings : ℝ) (new_rate : ℝ)
  (h1 : old_rate = 0.46)
  (h2 : income = 36000)
  (h3 : savings = 5040)
  (h4 : new_rate = (income * old_rate - savings) / income) :
  new_rate = 0.32 :=
by {
  sorry
}

end new_tax_rate_l108_108991


namespace solve_quartic_eq_l108_108715

theorem solve_quartic_eq {x : ℝ} : (x - 4)^4 + (x - 6)^4 = 16 → (x = 4 ∨ x = 6) :=
by
  sorry

end solve_quartic_eq_l108_108715


namespace no_solution_for_m_l108_108031

theorem no_solution_for_m (m : ℝ) : ¬ ∃ x : ℝ, x ≠ 1 ∧ (mx - 1) / (x - 1) = 3 ↔ m = 1 ∨ m = 3 := 
by sorry

end no_solution_for_m_l108_108031


namespace distance_from_origin_to_point_l108_108189

theorem distance_from_origin_to_point : 
  ∀ (x y : ℝ), x = 8 → y = -15 → real.sqrt ((x-0)^2 + (y-0)^2) = 17 :=
by
  intros x y hx hy
  rw [hx, hy]
  norm_num
  rw [real.sqrt_eq_rpow]
  norm_num
  sorry

end distance_from_origin_to_point_l108_108189


namespace intersection_A_B_l108_108819

-- Define set A and its condition
def A : Set ℝ := { y | ∃ (x : ℝ), y = x^2 }

-- Define set B and its condition
def B : Set ℝ := { x | ∃ (y : ℝ), y = Real.sqrt (1 - x^2) }

-- Define the set intersection A ∩ B
def A_intersect_B : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

-- The theorem statement
theorem intersection_A_B :
  A ∩ B = { x : ℝ | 0 ≤ x ∧ x ≤ 1 } :=
sorry

end intersection_A_B_l108_108819


namespace digit_6_count_1_to_700_l108_108984

theorem digit_6_count_1_to_700 :
  let countNumbersWithDigit6 := 700 - (7 * 9 * 9)
  countNumbersWithDigit6 = 133 := 
by
  let countNumbersWithDigit6 := 700 - (7 * 9 * 9)
  show countNumbersWithDigit6 = 133
  sorry

end digit_6_count_1_to_700_l108_108984


namespace quadratic_has_one_solution_implies_m_l108_108862

theorem quadratic_has_one_solution_implies_m (m : ℚ) :
  (∀ x : ℚ, 3 * x^2 - 7 * x + m = 0 → (b^2 - 4 * a * m = 0)) ↔ m = 49 / 12 :=
by
  sorry

end quadratic_has_one_solution_implies_m_l108_108862


namespace smallest_k_l108_108744

theorem smallest_k (k : ℕ) (h1 : k > 1) (h2 : k % 19 = 1) (h3 : k % 7 = 1) (h4 : k % 3 = 1) : k = 400 :=
by
  sorry

end smallest_k_l108_108744


namespace triangle_statements_l108_108846

-- Definitions of internal angles and sides of the triangle
def Triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  A + B + C = Real.pi ∧ a > 0 ∧ b > 0 ∧ c > 0

-- Statement A: If ABC is an acute triangle, then sin A > cos B
lemma statement_A (A B C a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_acute : A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2) :
  Real.sin A > Real.cos B := 
sorry

-- Statement B: If A > B, then sin A > sin B
lemma statement_B (A B C a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_AB : A > B) : 
  Real.sin A > Real.sin B := 
sorry

-- Statement C: If ABC is a non-right triangle, then tan A + tan B + tan C = tan A * tan B * tan C
lemma statement_C (A B C a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_non_right : A ≠ Real.pi / 2 ∧ B ≠ Real.pi / 2 ∧ C ≠ Real.pi / 2) : 
  Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C := 
sorry

-- Statement D: If a cos A = b cos B, then triangle ABC must be isosceles
lemma statement_D (A B C a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_cos : a * Real.cos A = b * Real.cos B) : 
  ¬(A = B) ∧ ¬(B = C) := 
sorry

-- Theorem to combine all statements
theorem triangle_statements (A B C a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_acute : A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2)
  (h_AB : A > B)
  (h_non_right : A ≠ Real.pi / 2 ∧ B ≠ Real.pi / 2 ∧ C ≠ Real.pi / 2)
  (h_cos : a * Real.cos A = b * Real.cos B) : 
  Real.sin A > Real.cos B ∧ Real.sin A > Real.sin B ∧ 
  (Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C) ∧ 
  ¬(A = B) ∧ ¬(B = C) := 
by
  exact ⟨statement_A A B C a b c h_triangle h_acute, statement_B A B C a b c h_triangle h_AB, statement_C A B C a b c h_triangle h_non_right, statement_D A B C a b c h_triangle h_cos⟩

end triangle_statements_l108_108846


namespace annie_total_distance_traveled_l108_108344

-- Definitions of conditions
def walk_distance : ℕ := 5
def bus_distance : ℕ := 7
def total_distance_one_way : ℕ := walk_distance + bus_distance
def total_distance_round_trip : ℕ := total_distance_one_way * 2

-- Theorem statement to prove the total number of blocks traveled
theorem annie_total_distance_traveled : total_distance_round_trip = 24 :=
by
  sorry

end annie_total_distance_traveled_l108_108344


namespace find_b_from_ellipse_l108_108661

-- Definitions used in conditions
variables {F₁ F₂ : ℝ → ℝ} -- foci
variables (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b)
noncomputable def ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Conditions
def point_on_ellipse (P : ℝ × ℝ) : Prop := ellipse a b P.1 P.2
def perpendicular_vectors (P : ℝ × ℝ) : Prop := true -- Simplified, use correct condition in detailed proof
def area_of_triangle (P : ℝ × ℝ) (F₁ F₂ : ℝ → ℝ) : ℝ := 9

-- The target statement
theorem find_b_from_ellipse (P : ℝ × ℝ) (condition1 : point_on_ellipse a b P)
  (condition2 : perpendicular_vectors P) 
  (condition3 : area_of_triangle P F₁ F₂ = 9) : 
  b = 3 := 
sorry

end find_b_from_ellipse_l108_108661


namespace f_2023_eq_1375_l108_108437

-- Define the function f and the conditions
noncomputable def f : ℕ → ℕ := sorry

axiom f_ff_eq (n : ℕ) (h : n > 0) : f (f n) = 3 * n
axiom f_3n2_eq (n : ℕ) (h : n > 0) : f (3 * n + 2) = 3 * n + 1

-- Prove the specific value for f(2023)
theorem f_2023_eq_1375 : f 2023 = 1375 := sorry

end f_2023_eq_1375_l108_108437


namespace convex_polygons_from_fifteen_points_l108_108503

theorem convex_polygons_from_fifteen_points 
    (h : ∀ (n : ℕ), n = 15) :
    ∃ (k : ℕ), k = 32192 :=
by
  sorry

end convex_polygons_from_fifteen_points_l108_108503


namespace sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l108_108286

theorem sqrt_49_mul_sqrt_25_eq_7_sqrt_5 : (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l108_108286


namespace valid_outfits_count_l108_108310

-- Definitions based on problem conditions
def shirts : Nat := 5
def pants : Nat := 6
def invalid_combination : Nat := 1

-- Problem statement
theorem valid_outfits_count : shirts * pants - invalid_combination = 29 := by 
  sorry

end valid_outfits_count_l108_108310


namespace sequence_total_sum_is_correct_l108_108794

-- Define the sequence pattern
def sequence_sum : ℕ → ℤ
| 0       => 1
| 1       => -2
| 2       => -4
| 3       => 8
| (n + 4) => sequence_sum n + 4

-- Define the number of groups in the sequence
def num_groups : ℕ := 319

-- Define the sum of each individual group
def group_sum : ℤ := 3

-- Define the total sum of the sequence
def total_sum : ℤ := num_groups * group_sum

theorem sequence_total_sum_is_correct : total_sum = 957 := by
  sorry

end sequence_total_sum_is_correct_l108_108794


namespace sqrt_49_times_sqrt_25_l108_108292

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 :=
by
  sorry

end sqrt_49_times_sqrt_25_l108_108292


namespace julia_miles_l108_108632

theorem julia_miles (total_miles darius_miles julia_miles : ℕ) 
  (h1 : darius_miles = 679)
  (h2 : total_miles = 1677)
  (h3 : total_miles = darius_miles + julia_miles) :
  julia_miles = 998 :=
by
  sorry

end julia_miles_l108_108632


namespace girls_more_than_boys_l108_108871

theorem girls_more_than_boys : ∃ (b g x : ℕ), b = 3 * x ∧ g = 4 * x ∧ b + g = 35 ∧ g - b = 5 :=
by  -- We just define the theorem, no need for a proof, added "by sorry"
  sorry

end girls_more_than_boys_l108_108871


namespace find_f_neg_l108_108011

noncomputable def f (a b x : ℝ) := a * x^3 + b * x - 2

theorem find_f_neg (a b : ℝ) (f_2017 : f a b 2017 = 7) : f a b (-2017) = -11 :=
by
  sorry

end find_f_neg_l108_108011


namespace penny_makes_total_revenue_l108_108497

def price_per_slice : ℕ := 7
def slices_per_pie : ℕ := 6
def pies_sold : ℕ := 7

theorem penny_makes_total_revenue :
  (pies_sold * slices_per_pie) * price_per_slice = 294 := by
  sorry

end penny_makes_total_revenue_l108_108497


namespace find_complementary_angle_l108_108159

theorem find_complementary_angle (angle1 angle2 : ℝ) (h1 : angle1 + angle2 = 90) (h2 : angle1 = 25) : angle2 = 65 := 
by 
  sorry

end find_complementary_angle_l108_108159


namespace sqrt_mul_sqrt_l108_108253

theorem sqrt_mul_sqrt (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : sqrt (a * sqrt b) = sqrt a * sqrt (sqrt b) := by
  sorry

example : sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  have h1 : sqrt 25 = 5 := by 
    norm_num
  have h2 : sqrt (49 * 5) = sqrt 245 := by  
    unfold sqrt
  have h3 : sqrt 245 = 7 * sqrt 5 := by 
    norm_num
  rw [h1, h2, h3]
  norm_num

end sqrt_mul_sqrt_l108_108253


namespace route_difference_l108_108205

noncomputable def time_route_A (distance_A : ℝ) (speed_A : ℝ) : ℝ :=
  (distance_A / speed_A) * 60

noncomputable def time_route_B (distance1_B distance2_B distance3_B : ℝ) (speed1_B speed2_B speed3_B : ℝ) : ℝ :=
  ((distance1_B / speed1_B) * 60) + 
  ((distance2_B / speed2_B) * 60) + 
  ((distance3_B / speed3_B) * 60)

theorem route_difference
  (distance_A : ℝ := 8)
  (speed_A : ℝ := 25)
  (distance1_B : ℝ := 2)
  (distance2_B : ℝ := 0.5)
  (speed1_B : ℝ := 50)
  (speed2_B : ℝ := 20)
  (distance_total_B : ℝ := 7)
  (speed3_B : ℝ := 35) :
  time_route_A distance_A speed_A - time_route_B distance1_B distance2_B (distance_total_B - distance1_B - distance2_B) speed1_B speed2_B speed3_B = 7.586 :=
by
  sorry

end route_difference_l108_108205


namespace parallel_lines_l108_108517

noncomputable def line1 (x y : ℝ) : Prop := x - y + 1 = 0
noncomputable def line2 (a x y : ℝ) : Prop := x + a * y + 3 = 0

theorem parallel_lines (a x y : ℝ) : (∀ (x y : ℝ), line1 x y → line2 a x y → x = y ∨ (line1 x y ∧ x ≠ y)) → 
  (a = -1 ∧ ∃ d : ℝ, d = Real.sqrt 2) :=
sorry

end parallel_lines_l108_108517


namespace hypotenuse_of_45_45_90_triangle_l108_108571

theorem hypotenuse_of_45_45_90_triangle (leg : ℝ) (angle_opposite_leg : ℝ) (h_leg : leg = 15) (h_angle : angle_opposite_leg = 45) :
  ∃ hypotenuse, hypotenuse = leg * Real.sqrt 2 :=
by
  use leg * Real.sqrt 2
  rw [h_leg]
  rw [h_angle]
  sorry

end hypotenuse_of_45_45_90_triangle_l108_108571


namespace determinant_of_sine_matrix_is_zero_l108_108630

open Matrix

theorem determinant_of_sine_matrix_is_zero :
  let A := λ (i j : Fin 3) => sin ((i.val * 3 + j.val + 1) : ℝ)
  det A = 0 := 
by
  let A := λ (i j : Fin 3) => sin ((i.val * 3 + j.val + 1) : ℝ)
  have : det A = 0 := sorry
  exact this

end determinant_of_sine_matrix_is_zero_l108_108630


namespace full_price_tickets_revenue_l108_108106

theorem full_price_tickets_revenue (f h d p : ℕ) 
  (h1 : f + h + d = 200) 
  (h2 : f * p + h * (p / 2) + d * (2 * p) = 5000) 
  (h3 : p = 50) : 
  f * p = 4500 :=
by
  sorry

end full_price_tickets_revenue_l108_108106


namespace hypotenuse_of_45_45_90_triangle_l108_108561

theorem hypotenuse_of_45_45_90_triangle (a : ℝ) (h : ℝ) 
  (ha : a = 15) 
  (angle_opposite_leg : ℝ) 
  (h_angle : angle_opposite_leg = 45) 
  (right_triangle : ∃ θ : ℝ, θ = 90) : 
  h = 15 * Real.sqrt 2 := 
sorry

end hypotenuse_of_45_45_90_triangle_l108_108561


namespace percentage_error_l108_108616

theorem percentage_error (x : ℚ) : 
  let incorrect_result := (3/5 : ℚ) * x
  let correct_result := (5/3 : ℚ) * x
  let ratio := incorrect_result / correct_result
  let percentage_error := (1 - ratio) * 100
  percentage_error = 64 :=
by
  let incorrect_result := (3/5 : ℚ) * x
  let correct_result := (5/3 : ℚ) * x
  let ratio := incorrect_result / correct_result
  let percentage_error := (1 - ratio) * 100
  sorry

end percentage_error_l108_108616


namespace binom_2000_3_eq_l108_108798

theorem binom_2000_3_eq : Nat.choose 2000 3 = 1331000333 := by
  sorry

end binom_2000_3_eq_l108_108798


namespace battery_life_after_exam_l108_108791

-- Define the conditions
def full_battery_life : ℕ := 60
def used_battery_fraction : ℚ := 3 / 4
def exam_duration : ℕ := 2

-- Define the theorem to prove the remaining battery life after the exam
theorem battery_life_after_exam (full_battery_life : ℕ) (used_battery_fraction : ℚ) (exam_duration : ℕ) : ℕ :=
  let remaining_battery_life := full_battery_life * (1 - used_battery_fraction)
  remaining_battery_life - exam_duration = 13

end battery_life_after_exam_l108_108791


namespace different_pronunciation_in_group_C_l108_108788

theorem different_pronunciation_in_group_C :
  let groupC := [("戏谑", "xuè"), ("虐待", "nüè"), ("瘠薄", "jí"), ("脊梁", "jǐ"), ("赝品", "yàn"), ("义愤填膺", "yīng")]
  ∀ {a : String} {b : String}, (a, b) ∈ groupC → a ≠ b :=
by
  intro groupC h
  sorry

end different_pronunciation_in_group_C_l108_108788


namespace find_radius_of_circle_l108_108542

theorem find_radius_of_circle :
  ∀ (r : ℝ) (α : ℝ) (ρ : ℝ) (θ : ℝ), r > 0 →
  (∀ (x y : ℝ), x = r * Real.cos α ∧ y = r * Real.sin α → x^2 + y^2 = r^2) →
  (∃ (x y: ℝ), x - y + 2 = 0 ∧ 2 * Real.sqrt (r^2 - 2) = 2 * Real.sqrt 2) →
  r = 2 :=
by
  intro r α ρ θ r_pos curve_eq polar_eq
  sorry

end find_radius_of_circle_l108_108542


namespace tangent_line_through_point_l108_108830

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x

theorem tangent_line_through_point (x y : ℝ) (h₁ : y = 2 * Real.log x - x) (h₂ : (1 : ℝ)  ≠ 0) 
  (h₃ : (-1 : ℝ) ≠ 0):
  (x - y - 2 = 0) :=
sorry

end tangent_line_through_point_l108_108830


namespace probability_grisha_wins_expectation_coin_flips_l108_108039

-- Define the conditions as predicates
def grishaWins (toss_seq : List Bool) : Bool := 
  (toss_seq.length % 2 == 0) && (toss_seq.last = some true)

def vanyaWins (toss_seq : List Bool) : Bool := 
  (toss_seq.length % 2 == 1) && (toss_seq.last = some false)

-- State the probability theorem
theorem probability_grisha_wins : 
  (∑ x in ((grishaWins ≤ 1 / 3))) = 1/3 := by sorry

-- State the expectation theorem
theorem expectation_coin_flips : 
  (E[flips until (grishaWins ∨ vanyaWins)]) = 2 := by sorry

end probability_grisha_wins_expectation_coin_flips_l108_108039


namespace hypotenuse_of_45_45_90_triangle_l108_108567

noncomputable def leg_length : ℝ := 15
noncomputable def angle_opposite_leg : ℝ := Real.pi / 4  -- 45 degrees in radians

theorem hypotenuse_of_45_45_90_triangle (h_leg : ℝ) (h_angle : ℝ) 
  (h_leg_cond : h_leg = leg_length) (h_angle_cond : h_angle = angle_opposite_leg) :
  ∃ h_hypotenuse : ℝ, h_hypotenuse = h_leg * Real.sqrt 2 :=
sorry

end hypotenuse_of_45_45_90_triangle_l108_108567


namespace tan_add_l108_108652

theorem tan_add (α β : ℝ) (h1 : Real.tan (α - π / 6) = 3 / 7) (h2 : Real.tan (π / 6 + β) = 2 / 5) : Real.tan (α + β) = 1 := by
  sorry

end tan_add_l108_108652


namespace equation_represents_3x_minus_7_equals_2x_plus_5_l108_108436

theorem equation_represents_3x_minus_7_equals_2x_plus_5 (x : ℝ) :
  (3 * x - 7 = 2 * x + 5) :=
sorry

end equation_represents_3x_minus_7_equals_2x_plus_5_l108_108436


namespace distinct_convex_polygons_of_four_or_more_sides_l108_108505

noncomputable def total_subsets (n : Nat) : Nat := 2^n

noncomputable def subsets_with_fewer_than_four_members (n : Nat) : Nat := 
  (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)

noncomputable def valid_subsets (n : Nat) : Nat := 
  total_subsets n - subsets_with_fewer_than_four_members n

theorem distinct_convex_polygons_of_four_or_more_sides (n : Nat) (h : n = 15) : valid_subsets n = 32192 := by
  sorry

end distinct_convex_polygons_of_four_or_more_sides_l108_108505


namespace tank_capacity_l108_108318

theorem tank_capacity
  (x : ℝ) -- define x as the full capacity of the tank in gallons
  (h1 : (5/6) * x - (2/3) * x = 15) -- first condition
  (h2 : (2/3) * x = y) -- second condition, though not actually needed
  : x = 90 := 
by sorry

end tank_capacity_l108_108318


namespace number_of_cipher_keys_l108_108869

theorem number_of_cipher_keys (n : ℕ) (h : n % 2 = 0) : 
  ∃ K : ℕ, K = 4^(n^2 / 4) :=
by 
  sorry

end number_of_cipher_keys_l108_108869


namespace arithmetic_identity_l108_108001

theorem arithmetic_identity : 3 * 5 * 7 + 15 / 3 = 110 := by
  sorry

end arithmetic_identity_l108_108001


namespace father_son_age_problem_l108_108484

theorem father_son_age_problem
  (F S Y : ℕ)
  (h1 : F = 3 * S)
  (h2 : F = 45)
  (h3 : F + Y = 2 * (S + Y)) :
  Y = 15 :=
sorry

end father_son_age_problem_l108_108484


namespace face_opposite_to_A_l108_108607

-- Define the faces and their relationships
inductive Face : Type
| A | B | C | D | E | F
open Face

def adjacent (x y : Face) : Prop :=
  match x, y with
  | A, B => true
  | B, A => true
  | C, A => true
  | A, C => true
  | D, A => true
  | A, D => true
  | C, D => true
  | D, C => true
  | E, F => true
  | F, E => true
  | _, _ => false

-- Theorem stating that "F" is opposite to "A" given the provided conditions.
theorem face_opposite_to_A : ∀ x : Face, (adjacent A x = false) → (x = B ∨ x = C ∨ x = D → false) → (x = E ∨ x = F) → x = F := 
  by
    intros x h1 h2 h3
    sorry

end face_opposite_to_A_l108_108607


namespace probability_sum_18_l108_108083

def total_outcomes := 100

def successful_pairs := [(8, 10), (9, 9), (10, 8)]

def num_successful_outcomes := successful_pairs.length

theorem probability_sum_18 : (num_successful_outcomes / total_outcomes : ℚ) = 3 / 100 := 
by
  -- The actual proof should go here
  sorry

end probability_sum_18_l108_108083


namespace common_difference_of_arithmetic_sequence_l108_108367

variable {a : ℕ → ℝ} {S : ℕ → ℝ}
noncomputable def S_n (n : ℕ) : ℝ := -n^2 + 4*n

theorem common_difference_of_arithmetic_sequence :
  (∀ n : ℕ, S n = S_n n) →
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d ∧ d = -2 :=
by
  intro h
  use -2
  sorry

end common_difference_of_arithmetic_sequence_l108_108367


namespace probability_multiple_of_four_l108_108850

theorem probability_multiple_of_four :
  let spinner_outcome : ℕ → ℕ
        | 0 => 2
        | 1 => 4
        | 2 => 1
        | 3 => 3
        | _ => 0
  in let starting_points := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
  in let multiples_of_4 := {x ∈ starting_points | x % 4 = 0}
  in let prob_starting_at_multiple_of_4 := (multiples_of_4.card : ℚ) / (starting_points.card)
  in let prob_reaching_multiple_of_4 :=
       prob_starting_at_multiple_of_4 * 1 + 
       (4 / starting_points.card) * 2 * (1 / 16)
  in prob_reaching_multiple_of_4 = 7 / 24 :=
by
  let spinner_probabilities (sp1 sp2 : ℕ → ℕ) :=
    (Prob.elem {spinner_outcome sp1, spinner_outcome sp2} : ℚ) / 16
  let total_prob := Prob.event (λ n => n % 4 = 0)
  sorry

end probability_multiple_of_four_l108_108850


namespace angle_terminal_side_equiv_l108_108722

theorem angle_terminal_side_equiv (k : ℤ) : 
  ∀ θ α : ℝ, θ = - (π / 3) → α = 5 * π / 3 → α = θ + 2 * k * π := by
  intro θ α hθ hα
  sorry

end angle_terminal_side_equiv_l108_108722


namespace range_of_a_l108_108832

theorem range_of_a (a : ℝ) : 1 ∉ {x : ℝ | x^2 - 2 * x + a > 0} → a ≤ 1 :=
by
  sorry

end range_of_a_l108_108832


namespace intersection_A_complement_B_eq_minus_three_to_zero_l108_108165

-- Define the set A
def A : Set ℝ := { x : ℝ | x^2 + x - 6 ≤ 0 }

-- Define the set B
def B : Set ℝ := { y : ℝ | ∃ x : ℝ, y = Real.sqrt x ∧ 0 ≤ x ∧ x ≤ 4 }

-- Define the complement of B
def C_RB : Set ℝ := { y : ℝ | ¬ (y ∈ B) }

-- The proof problem
theorem intersection_A_complement_B_eq_minus_three_to_zero :
  (A ∩ C_RB) = { x : ℝ | -3 ≤ x ∧ x < 0 } :=
by
  sorry

end intersection_A_complement_B_eq_minus_three_to_zero_l108_108165


namespace value_of_C_l108_108885

theorem value_of_C (C : ℝ) (h : 4 * C + 3 = 25) : C = 5.5 :=
by
  sorry

end value_of_C_l108_108885


namespace divisor_is_three_l108_108909

theorem divisor_is_three (n d q p : ℕ) (h1 : n = d * q + 3) (h2 : n^2 = d * p + 3) : d = 3 := 
sorry

end divisor_is_three_l108_108909


namespace solve_p_value_l108_108077

noncomputable def solve_for_p (n m p : ℚ) : Prop :=
  (5 / 6 = n / 90) ∧ ((m + n) / 105 = (p - m) / 150) ∧ (p = 137.5)

theorem solve_p_value (n m p : ℚ) (h1 : 5 / 6 = n / 90) (h2 : (m + n) / 105 = (p - m) / 150) : 
  p = 137.5 :=
by
  sorry

end solve_p_value_l108_108077


namespace sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l108_108288

theorem sqrt_49_mul_sqrt_25_eq_7_sqrt_5 : (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l108_108288


namespace range_3x_plus_2y_l108_108030

theorem range_3x_plus_2y (x y : ℝ) : -1 < x + y ∧ x + y < 4 → 2 < x - y ∧ x - y < 3 → 
  -3/2 < 3*x + 2*y ∧ 3*x + 2*y < 23/2 :=
by
  sorry

end range_3x_plus_2y_l108_108030


namespace no_periodic_sum_l108_108472

def is_periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x : ℝ, f (x + p) = f x

theorem no_periodic_sum (g h : ℝ → ℝ) :
  (is_periodic g 2) → (is_periodic h (π / 2)) → ¬ ∃ T > 0, is_periodic (λ x, g x + h x) T :=
by {
  sorry
}

end no_periodic_sum_l108_108472


namespace sqrt_D_irrational_l108_108413

variable (k : ℤ)

def a := 3 * k
def b := 3 * k + 3
def c := a k + b k
def D := a k * a k + b k * b k + c k * c k

theorem sqrt_D_irrational : ¬ ∃ (r : ℚ), r * r = D k := 
by sorry

end sqrt_D_irrational_l108_108413


namespace solution_system2_l108_108992

-- Given first system and its solution
variables {a1 a2 c1 c2 : ℝ}
variables {x y : ℝ}

-- Conditions from system 1 and its solution
def system1_eq1 :=  a1 * 2 + 3 = c1
def system1_eq2 :=  a2 * 2 + 3 = c2

-- Conditions from second system
def system2_eq1 :=  a1 * x + y = a1 - c1
def system2_eq2 :=  a2 * x + y = a2 - c2

-- Goal
theorem solution_system2 : system1_eq1 ∧ system1_eq2 → system2_eq1 ∧ system2_eq2 → x = -1 ∧ y = -3 :=
by
  intros h1 h2
  sorry

end solution_system2_l108_108992


namespace bottom_row_bricks_l108_108898

theorem bottom_row_bricks (x : ℕ) 
    (h : x + (x - 1) + (x - 2) + (x - 3) + (x - 4) = 200) : x = 42 :=
sorry

end bottom_row_bricks_l108_108898


namespace max_distance_proof_area_of_coverage_ring_proof_l108_108812

noncomputable def maxDistanceFromCenterToRadars : ℝ :=
  24 / Real.sin (Real.pi / 7)

noncomputable def areaOfCoverageRing : ℝ :=
  960 * Real.pi / Real.tan (Real.pi / 7)

theorem max_distance_proof :
  ∀ (r n : ℕ) (width : ℝ),  n = 7 → r = 26 → width = 20 → 
  maxDistanceFromCenterToRadars = 24 / Real.sin (Real.pi / 7) :=
by
  intros r n width hn hr hwidth
  sorry

theorem area_of_coverage_ring_proof :
  ∀ (r n : ℕ) (width : ℝ), n = 7 → r = 26 → width = 20 → 
  areaOfCoverageRing = 960 * Real.pi / Real.tan (Real.pi / 7) :=
by
  intros r n width hn hr hwidth
  sorry

end max_distance_proof_area_of_coverage_ring_proof_l108_108812


namespace larger_number_l108_108455

theorem larger_number (x y : ℤ) (h1 : x + y = 47) (h2 : x - y = 7) : x = 27 :=
  sorry

end larger_number_l108_108455


namespace triangle_is_isosceles_l108_108840

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (triangle : Type)

noncomputable def is_isosceles_triangle (A B C : ℝ) (a b c : ℝ) (triangle : Type) : Prop :=
  c = 2 * a * Real.cos B → A = B ∨ B = C ∨ C = A

theorem triangle_is_isosceles (A B C : ℝ) (a b c : ℝ) (triangle : Type) (h : c = 2 * a * Real.cos B) :
  is_isosceles_triangle A B C a b c triangle :=
sorry

end triangle_is_isosceles_l108_108840


namespace minimum_tanA_9tanB_l108_108975

variable (a b c A B : ℝ)
variable (Aacute : A > 0 ∧ A < π / 2)
variable (h1 : a^2 = b^2 + 2*b*c * Real.sin A)
variable (habc : a = b * Real.sin A)

theorem minimum_tanA_9tanB : 
  ∃ (A B : ℝ), (A > 0 ∧ A < π / 2) ∧ (a^2 = b^2 + 2*b*c * Real.sin A) ∧ (a = b * Real.sin A) ∧ 
  (min ((Real.tan A) - 9*(Real.tan B)) = -2) := 
  sorry

end minimum_tanA_9tanB_l108_108975


namespace factor_difference_of_squares_l108_108934
noncomputable def factored (t : ℝ) : ℝ := (t - 8) * (t + 8)

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = factored t := by
  sorry

end factor_difference_of_squares_l108_108934


namespace find_angle_l108_108203

variables {V : Type*} [inner_product_space ℝ V] (a b : V)

theorem find_angle (ahyp : ∥a∥ = 2) (bhyp : ∥b∥ = 1) (dot_hyp : ⟪a, a - b⟫ = 3) :
  real.angle a b = real.pi / 3 :=
by sorry

end find_angle_l108_108203


namespace range_of_a_l108_108662

-- Define the even function property
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the monotonically increasing property on [0, ∞)
def mono_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  even_function f →
  mono_increasing_on_nonneg f →
  (f (Real.log a / Real.log 2) + f (Real.log a / Real.log (1/2)) ≤ 2 * f 1) →
  (0 < a ∧ a ≤ 2) :=
by
  intros h_even h_mono h_ineq
  sorry

end range_of_a_l108_108662


namespace sum_S6_l108_108964

variable (a_n : ℕ → ℚ)
variable (d : ℚ)
variable (S : ℕ → ℚ)
variable (a1 : ℚ)

/-- Define arithmetic sequence with common difference -/
def arithmetic_seq (n : ℕ) := a1 + n * d

/-- Define the sum of the first n terms of the sequence -/
def sum_of_arith_seq (n : ℕ) := n * a1 + (n * (n - 1) / 2) * d

/-- The given conditions -/
axiom h1 : d = 5
axiom h2 : (a_n 1 = a1) ∧ (a_n 2 = a1 + d) ∧ (a_n 5 = a1 + 4 * d)
axiom geom_seq : (a1 + d)^2 = a1 * (a1 + 4 * d)

theorem sum_S6 : S 6 = 90 := by
  sorry

end sum_S6_l108_108964


namespace hyperbola_focus_and_asymptotes_l108_108338

def is_focus_on_y_axis (a b : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
∃ c : ℝ, eq (c^2 * a) (c^2 * b)

def are_asymptotes_perpendicular (eq : ℝ → ℝ → Prop) : Prop :=
∃ k1 k2 : ℝ, (k1 != 0 ∧ k2 != 0 ∧ eq k1 k2 ∧ eq (-k1) k2)

theorem hyperbola_focus_and_asymptotes :
  is_focus_on_y_axis 1 (-1) (fun y x => y^2 - x^2 = 4) ∧ are_asymptotes_perpendicular (fun y x => y = x) :=
by
  sorry

end hyperbola_focus_and_asymptotes_l108_108338


namespace find_c_and_d_l108_108923

theorem find_c_and_d (c d : ℝ) (h : ℝ → ℝ) (f : ℝ → ℝ) (finv : ℝ → ℝ) 
  (h_def : ∀ x, h x = 6 * x - 5)
  (finv_eq : ∀ x, finv x = 6 * x - 3)
  (f_def : ∀ x, f x = c * x + d)
  (inv_prop : ∀ x, f (finv x) = x ∧ finv (f x) = x) :
  4 * c + 6 * d = 11 / 3 :=
by
  sorry

end find_c_and_d_l108_108923


namespace james_weekly_earnings_l108_108407

def hourly_rate : ℕ := 20
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 4

theorem james_weekly_earnings : hourly_rate * (hours_per_day * days_per_week) = 640 := by
  sorry

end james_weekly_earnings_l108_108407


namespace initial_paint_l108_108137

variable (total_needed : ℕ) (paint_bought : ℕ) (still_needed : ℕ)

theorem initial_paint (h_total_needed : total_needed = 70)
                      (h_paint_bought : paint_bought = 23)
                      (h_still_needed : still_needed = 11) : 
                      ∃ x : ℕ, x = 36 :=
by
  sorry

end initial_paint_l108_108137


namespace intersection_M_N_l108_108166

def M : Set ℝ := { x | -5 < x ∧ x < 3 }
def N : Set ℝ := { x | -2 < x ∧ x < 4 }

theorem intersection_M_N : M ∩ N = { x | -2 < x ∧ x < 3 } := 
by sorry

end intersection_M_N_l108_108166


namespace triple_apply_l108_108389

def f (x : ℝ) : ℝ := 5 * x - 4

theorem triple_apply : f (f (f 2)) = 126 :=
by
  rw [f, f, f]
  sorry

end triple_apply_l108_108389


namespace quadratic_has_two_distinct_roots_l108_108151

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem quadratic_has_two_distinct_roots 
  (a b c : ℝ) (h : 2016 + a^2 + a * c < a * b) : 
  discriminant a b c > 0 :=
sorry

end quadratic_has_two_distinct_roots_l108_108151


namespace poly_degree_and_terms_correct_l108_108081

-- Define the specific polynomial
def poly : Polynomial (ℤ × ℤ) := Polynomial.C (1, 0) * Polynomial.x ^ (2, 0) * Polynomial.y ^ (3, 0) 
                                   - Polynomial.C (3, 0) * Polynomial.x ^ (1, 0) * Polynomial.y ^ (3, 0) 
                                   - Polynomial.C (2, 0)

-- Degree calculation
def poly_degree := 5

-- Number of terms calculation
def poly_num_terms := 3

-- The theorem stating the degree and number of terms of the polynomial
theorem poly_degree_and_terms_correct :
  deg poly = poly_degree ∧ count_terms poly = poly_num_terms :=
by
  sorry

end poly_degree_and_terms_correct_l108_108081


namespace volume_ratio_l108_108530

theorem volume_ratio (x : ℝ) (h : x > 0) : 
  let V_Q := x^3
  let V_P := (3 * x)^3
  (V_Q / V_P) = (1 / 27) :=
by
  sorry

end volume_ratio_l108_108530


namespace parabola_focus_coordinates_l108_108593

theorem parabola_focus_coordinates : 
  ∀ (x y : ℝ), x = 4 * y^2 → (∃ (y₀ : ℝ), (x, y₀) = (1/16, 0)) :=
by
  intro x y hxy
  sorry

end parabola_focus_coordinates_l108_108593


namespace original_price_l108_108309

theorem original_price (total_payment : ℝ) (num_units : ℕ) (discount_rate : ℝ) 
(h1 : total_payment = 500) (h2 : num_units = 18) (h3 : discount_rate = 0.20) : 
  (total_payment / (1 - discount_rate) * num_units) = 625.05 :=
by
  sorry

end original_price_l108_108309


namespace sqrt_49_times_sqrt_25_l108_108294

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 :=
by
  sorry

end sqrt_49_times_sqrt_25_l108_108294


namespace regular_octagon_interior_angle_l108_108463

theorem regular_octagon_interior_angle : 
  (∀ (n : ℕ), n = 8 → ∀ (sum_of_interior_angles : ℕ), sum_of_interior_angles = (n - 2) * 180 → ∀ (each_angle : ℕ), each_angle = sum_of_interior_angles / n → each_angle = 135) :=
  sorry

end regular_octagon_interior_angle_l108_108463


namespace cd_cost_l108_108025

theorem cd_cost (mp3_cost savings father_amt lacks cd_cost : ℝ) :
  mp3_cost = 120 ∧ savings = 55 ∧ father_amt = 20 ∧ lacks = 64 →
  120 + cd_cost - (savings + father_amt) = lacks → 
  cd_cost = 19 :=
by
  intros
  sorry

end cd_cost_l108_108025


namespace three_person_subcommittees_from_seven_l108_108383

-- Definition of the combinations formula (binomial coefficient)
def choose : ℕ → ℕ → ℕ
| n, k => if k = 0 then 1 else (n * choose (n - 1) (k - 1)) / k 

-- Problem statement in Lean 4
theorem three_person_subcommittees_from_seven : choose 7 3 = 35 :=
by
  -- We would fill in the steps here or use a sorry to skip the proof
  sorry

end three_person_subcommittees_from_seven_l108_108383


namespace volume_of_one_pizza_piece_l108_108331

theorem volume_of_one_pizza_piece
  (h : ℝ) (d : ℝ) (n : ℕ)
  (h_eq : h = 1 / 2)
  (d_eq : d = 16)
  (n_eq : n = 16) :
  ((π * (d / 2)^2 * h) / n) = 2 * π :=
by
  rw [h_eq, d_eq, n_eq]
  sorry

end volume_of_one_pizza_piece_l108_108331


namespace janessa_keeps_cards_l108_108849

theorem janessa_keeps_cards (l1 l2 l3 l4 l5 : ℕ) :
  -- Conditions
  l1 = 4 →
  l2 = 13 →
  l3 = 36 →
  l4 = 4 →
  l5 = 29 →
  -- The total number of cards Janessa initially has is l1 + l2.
  let initial_cards := l1 + l2 in
  -- After ordering additional cards from eBay, she has initial_cards + l3 cards.
  let cards_after_order := initial_cards + l3 in
  -- After discarding bad cards, she has cards_after_order - l4 cards.
  let cards_after_discard := cards_after_order - l4 in
  -- She gives l5 cards to Dexter, so she keeps cards_after_discard - l5 cards.
  cards_after_discard - l5 = 20 :=
by
  intros h1 h2 h3 h4 h5
  let initial_cards := l1 + l2
  let cards_after_order := initial_cards + l3
  let cards_after_discard := cards_after_order - l4
  show cards_after_discard - l5 = 20, from sorry

end janessa_keeps_cards_l108_108849


namespace monotonic_increasing_k_l108_108866

noncomputable def f (k x : ℝ) : ℝ := k * x^2 + (3 * k - 2) * x - 5

theorem monotonic_increasing_k (k : ℝ) : (∀ x y : ℝ, 1 ≤ x → x ≤ y → f k x ≤ f k y) ↔ k ∈ Set.Ici (2 / 5) :=
by
  sorry

end monotonic_increasing_k_l108_108866


namespace relationship_y1_y2_y3_l108_108966

-- Conditions
def y1 := -((-4)^2) + 5
def y2 := -((-1)^2) + 5
def y3 := -(2^2) + 5

-- Statement to be proved
theorem relationship_y1_y2_y3 : y2 > y3 ∧ y3 > y1 := by
  sorry

end relationship_y1_y2_y3_l108_108966


namespace sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l108_108287

theorem sqrt_49_mul_sqrt_25_eq_7_sqrt_5 : (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l108_108287


namespace original_difference_of_weights_l108_108588

variable (F S T : ℝ)

theorem original_difference_of_weights :
  (F + S + T = 75) →
  (F - 2 = 0.7 * (S + 2)) →
  (S + 1 = 0.8 * (T + 1)) →
  T - F = 10.16 :=
by
  intro h1 h2 h3
  sorry

end original_difference_of_weights_l108_108588


namespace train_travel_distance_l108_108115

theorem train_travel_distance (speed time: ℕ) (h1: speed = 85) (h2: time = 4) : speed * time = 340 :=
by
-- Given: speed = 85 km/hr and time = 4 hr
-- To prove: speed * time = 340
-- Since speed = 85 and time = 4, then 85 * 4 = 340
sorry

end train_travel_distance_l108_108115


namespace hypotenuse_of_45_45_90_triangle_15_l108_108565

theorem hypotenuse_of_45_45_90_triangle_15 (a : ℝ) (h : a = 15) : 
  ∃ (c : ℝ), c = a * Real.sqrt 2 :=
by
  use a * Real.sqrt 2
  rw h
  sorry

end hypotenuse_of_45_45_90_triangle_15_l108_108565


namespace altitude_eqn_median_eqn_l108_108152

def Point := (ℝ × ℝ)

def A : Point := (4, 0)
def B : Point := (6, 7)
def C : Point := (0, 3)

theorem altitude_eqn (B C: Point) : 
  ∃ (k b : ℝ), (b = 6) ∧ (k = - 3 / 2) ∧ (∀ x y : ℝ, y = k * x + b →
  3 * x + 2 * y - 12 = 0)
:=
sorry

theorem median_eqn (A B C : Point) :
  ∃ (k b : ℝ), (b = 20) ∧ (k = -3/5) ∧ (∀ x y : ℝ, y = k * x + b →
  5 * x + y - 20 = 0)
:=
sorry

end altitude_eqn_median_eqn_l108_108152


namespace percentage_error_l108_108617

theorem percentage_error (x : ℚ) : 
  let incorrect_result := (3/5 : ℚ) * x
  let correct_result := (5/3 : ℚ) * x
  let ratio := incorrect_result / correct_result
  let percentage_error := (1 - ratio) * 100
  percentage_error = 64 :=
by
  let incorrect_result := (3/5 : ℚ) * x
  let correct_result := (5/3 : ℚ) * x
  let ratio := incorrect_result / correct_result
  let percentage_error := (1 - ratio) * 100
  sorry

end percentage_error_l108_108617


namespace train_length_l108_108336

-- Definitions and conditions based on the problem
def time : ℝ := 28.997680185585153
def bridge_length : ℝ := 150
def train_speed : ℝ := 10

-- The theorem to prove
theorem train_length : (train_speed * time) - bridge_length = 139.97680185585153 :=
by
  sorry

end train_length_l108_108336


namespace positive_number_square_roots_l108_108395

theorem positive_number_square_roots (a : ℝ) 
  (h1 : (2 * a - 1) ^ 2 = (a - 2) ^ 2) 
  (h2 : ∃ b : ℝ, b > 0 ∧ ((2 * a - 1) = b ∨ (a - 2) = b)) : 
  ∃ n : ℝ, n = 1 :=
by
  sorry

end positive_number_square_roots_l108_108395


namespace isosceles_triangle_inequality_l108_108965

theorem isosceles_triangle_inequality
  (a b : ℝ)
  (hb : b > 0)
  (h₁₂ : 12 * (π / 180) = π / 15) 
  (h_sin6 : Real.sin (6 * (π / 180)) > 1 / 10)
  (h_eq : a = 2 * b * Real.sin (6 * (π / 180))) : 
  b < 5 * a := 
by
  sorry

end isosceles_triangle_inequality_l108_108965


namespace bananas_per_friend_l108_108241

-- Define the conditions
def total_bananas : ℕ := 40
def number_of_friends : ℕ := 40

-- Define the theorem to be proved
theorem bananas_per_friend : 
  (total_bananas / number_of_friends) = 1 :=
by
  sorry

end bananas_per_friend_l108_108241


namespace actual_time_of_storm_l108_108224

theorem actual_time_of_storm
  (malfunctioned_hours_tens_digit : ℕ)
  (malfunctioned_hours_units_digit : ℕ)
  (malfunctioned_minutes_tens_digit : ℕ)
  (malfunctioned_minutes_units_digit : ℕ)
  (original_time : ℕ × ℕ)
  (hours_tens_digit : ℕ := 2)
  (hours_units_digit : ℕ := 0)
  (minutes_tens_digit : ℕ := 0)
  (minutes_units_digit : ℕ := 9) :
  (malfunctioned_hours_tens_digit = hours_tens_digit + 1 ∨ malfunctioned_hours_tens_digit = hours_tens_digit - 1) →
  (malfunctioned_hours_units_digit = hours_units_digit + 1 ∨ malfunctioned_hours_units_digit = hours_units_digit - 1) →
  (malfunctioned_minutes_tens_digit = minutes_tens_digit + 1 ∨ malfunctioned_minutes_tens_digit = minutes_tens_digit - 1) →
  (malfunctioned_minutes_units_digit = minutes_units_digit + 1 ∨ malfunctioned_minutes_units_digit = minutes_units_digit - 1) →
  original_time = (11, 18) :=
by
  sorry

end actual_time_of_storm_l108_108224


namespace intersection_domain_range_l108_108979

-- Define domain and function
def domain : Set ℝ := {-1, 0, 1}
def f (x : ℝ) : ℝ := |x|

-- Prove the theorem
theorem intersection_domain_range :
  let range : Set ℝ := {y | ∃ x ∈ domain, f x = y}
  let A : Set ℝ := domain
  let B : Set ℝ := range 
  A ∩ B = {0, 1} :=
by
  -- The proof is skipped with sorry
  sorry

end intersection_domain_range_l108_108979


namespace exponents_subtraction_l108_108657

theorem exponents_subtraction (m n : ℕ) (hm : 3 ^ m = 8) (hn : 3 ^ n = 2) : 3 ^ (m - n) = 4 := 
by
  sorry

end exponents_subtraction_l108_108657


namespace three_digit_cubes_divisible_by_8_l108_108024

theorem three_digit_cubes_divisible_by_8 : ∃ (S : Finset ℕ), S.card = 2 ∧ ∀ x ∈ S, x ^ 3 ≥ 100 ∧ x ^ 3 ≤ 999 ∧ x ^ 3 % 8 = 0 :=
by
  sorry

end three_digit_cubes_divisible_by_8_l108_108024


namespace expression_values_l108_108138

-- Define the conditions as a predicate
def conditions (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a^2 - b * c = b^2 - a * c ∧ b^2 - a * c = c^2 - a * b

-- The main theorem statement
theorem expression_values (a b c : ℝ) (h : conditions a b c) :
  (∃ x : ℝ, x = (a / (b + c) + 2 * b / (a + c) + 4 * c / (a + b)) ∧ (x = 7 / 2 ∨ x = -7)) :=
by
  sorry

end expression_values_l108_108138


namespace subset_P1_P2_l108_108549

def P1 (a : ℝ) : Set ℝ := {x | x^2 + a*x + 1 > 0}
def P2 (a : ℝ) : Set ℝ := {x | x^2 + a*x + 2 > 0}

theorem subset_P1_P2 (a : ℝ) : P1 a ⊆ P2 a :=
by intros x hx; sorry

end subset_P1_P2_l108_108549


namespace sum_of_two_numbers_l108_108993

theorem sum_of_two_numbers (x y : ℝ) 
  (h1 : x^2 + y^2 = 220) 
  (h2 : x * y = 52) : 
  x + y = 18 :=
by
  sorry

end sum_of_two_numbers_l108_108993


namespace unique_integral_root_l108_108637

theorem unique_integral_root {x : ℤ} :
  x - 12 / (x - 3) = 5 - 12 / (x - 3) ↔ x = 5 :=
by
  sorry

end unique_integral_root_l108_108637


namespace hex_B2F_to_base10_l108_108352

theorem hex_B2F_to_base10 :
  let b := 11
  let two := 2
  let f := 15
  let base := 16
  (b * base^2 + two * base^1 + f * base^0) = 2863 :=
by
  sorry

end hex_B2F_to_base10_l108_108352


namespace how_many_more_rolls_needed_l108_108958

variable (total_needed sold_to_grandmother sold_to_uncle sold_to_neighbor : ℕ)

theorem how_many_more_rolls_needed (h1 : total_needed = 45)
  (h2 : sold_to_grandmother = 1)
  (h3 : sold_to_uncle = 10)
  (h4 : sold_to_neighbor = 6) :
  total_needed - (sold_to_grandmother + sold_to_uncle + sold_to_neighbor) = 28 := by
  sorry

end how_many_more_rolls_needed_l108_108958


namespace find_D_l108_108767

-- Definitions
variable (A B C D E F : ℕ)

-- Conditions
axiom sum_AB : A + B = 16
axiom sum_BC : B + C = 12
axiom sum_EF : E + F = 8
axiom total_sum : A + B + C + D + E + F = 18

-- Theorem statement
theorem find_D : D = 6 :=
by
  sorry

end find_D_l108_108767


namespace triangle_perimeter_l108_108533

/-- In a triangle ABC, where sides a, b, c are opposite to angles A, B, C respectively.
Given the area of the triangle = 15 * sqrt 3 / 4, 
angle A = 60 degrees and 5 * sin B = 3 * sin C,
prove that the perimeter of triangle ABC is 8 + sqrt 19. -/
theorem triangle_perimeter
  (a b c : ℝ)
  (A B C : ℝ)
  (hA : A = 60)
  (h_area : (1 / 2) * b * c * (Real.sin (A / (180 / Real.pi))) = 15 * Real.sqrt 3 / 4)
  (h_sin : 5 * Real.sin B = 3 * Real.sin C) :
  a + b + c = 8 + Real.sqrt 19 :=
sorry

end triangle_perimeter_l108_108533


namespace stock_price_calculation_l108_108353

def stock_price_end_of_first_year (initial_price : ℝ) (increase_percent : ℝ) : ℝ :=
  initial_price * (1 + increase_percent)

def stock_price_end_of_second_year (price_first_year : ℝ) (decrease_percent : ℝ) : ℝ :=
  price_first_year * (1 - decrease_percent)

theorem stock_price_calculation 
  (initial_price : ℝ)
  (increase_percent : ℝ)
  (decrease_percent : ℝ)
  (final_price : ℝ) :
  initial_price = 120 ∧ 
  increase_percent = 0.80 ∧
  decrease_percent = 0.30 ∧
  final_price = 151.20 → 
  stock_price_end_of_second_year (stock_price_end_of_first_year initial_price increase_percent) decrease_percent = final_price :=
by
  sorry

end stock_price_calculation_l108_108353


namespace volume_of_one_pizza_piece_l108_108330

theorem volume_of_one_pizza_piece
  (h : ℝ) (d : ℝ) (n : ℕ)
  (h_eq : h = 1 / 2)
  (d_eq : d = 16)
  (n_eq : n = 16) :
  ((π * (d / 2)^2 * h) / n) = 2 * π :=
by
  rw [h_eq, d_eq, n_eq]
  sorry

end volume_of_one_pizza_piece_l108_108330


namespace sqrt_product_l108_108276

theorem sqrt_product (a b : ℝ) (h1 : a = 49) (h2 : b = 25) : sqrt (a * sqrt b) = 7 * sqrt 5 :=
by sorry

end sqrt_product_l108_108276


namespace hypotenuse_of_45_45_90_triangle_15_l108_108564

theorem hypotenuse_of_45_45_90_triangle_15 (a : ℝ) (h : a = 15) : 
  ∃ (c : ℝ), c = a * Real.sqrt 2 :=
by
  use a * Real.sqrt 2
  rw h
  sorry

end hypotenuse_of_45_45_90_triangle_15_l108_108564


namespace point_in_third_quadrant_l108_108229

theorem point_in_third_quadrant :
  let sin2018 := Real.sin (2018 * Real.pi / 180)
  let tan117 := Real.tan (117 * Real.pi / 180)
  sin2018 < 0 ∧ tan117 < 0 → 
  (sin2018 < 0 ∧ tan117 < 0) :=
by
  intros
  sorry

end point_in_third_quadrant_l108_108229


namespace trader_cloth_sale_l108_108335

theorem trader_cloth_sale (total_SP : ℕ) (profit_per_meter : ℕ) (cost_per_meter : ℕ) (SP_per_meter : ℕ)
  (h1 : total_SP = 8400) (h2 : profit_per_meter = 12) (h3 : cost_per_meter = 128) (h4 : SP_per_meter = cost_per_meter + profit_per_meter) :
  ∃ (x : ℕ), SP_per_meter * x = total_SP ∧ x = 60 :=
by
  -- We will skip the proof using sorry
  sorry

end trader_cloth_sale_l108_108335


namespace smallest_solution_neg_two_l108_108362

-- We set up the expressions and then state the smallest solution
def smallest_solution (x : ℝ) : Prop :=
  x * abs x = 3 * x + 2

theorem smallest_solution_neg_two :
  ∃ x : ℝ, smallest_solution x ∧ (∀ y : ℝ, smallest_solution y → y ≥ x) ∧ x = -2 :=
by
  sorry

end smallest_solution_neg_two_l108_108362


namespace parabola_intersects_xaxis_at_least_one_l108_108972

theorem parabola_intersects_xaxis_at_least_one {a b c : ℝ} (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  (∃ x1 x2, x1 ≠ x2 ∧ (a * x1^2 + 2 * b * x1 + c = 0) ∧ (a * x2^2 + 2 * b * x2 + c = 0)) ∨
  (∃ x1 x2, x1 ≠ x2 ∧ (b * x1^2 + 2 * c * x1 + a = 0) ∧ (b * x2^2 + 2 * c * x2 + a = 0)) ∨
  (∃ x1 x2, x1 ≠ x2 ∧ (c * x1^2 + 2 * a * x1 + b = 0) ∧ (c * x2^2 + 2 * a * x2 + b = 0)) :=
by
  sorry

end parabola_intersects_xaxis_at_least_one_l108_108972


namespace first_pipe_time_l108_108609

noncomputable def pool_filling_time (T : ℝ) : Prop :=
  (1 / T + 1 / 12 = 1 / 4.8) → (T = 8)

theorem first_pipe_time :
  ∃ T : ℝ, pool_filling_time T := by
  use 8
  sorry

end first_pipe_time_l108_108609


namespace factor_difference_of_squares_l108_108938

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) :=
by
  sorry

end factor_difference_of_squares_l108_108938


namespace there_are_six_bases_ending_in_one_for_625_in_decimal_l108_108810

theorem there_are_six_bases_ending_in_one_for_625_in_decimal :
  (∃ ls : List ℕ, ls = [2, 3, 4, 6, 8, 12] ∧ ∀ b ∈ ls, 2 ≤ b ∧ b ≤ 12 ∧ 624 % b = 0 ∧ List.length ls = 6) :=
by
  sorry

end there_are_six_bases_ending_in_one_for_625_in_decimal_l108_108810


namespace find_t_l108_108974

theorem find_t (t : ℝ) (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) :
  (∀ n, a_n n = a_n 1 * (a_n 2 / a_n 1)^(n-1)) → -- Geometric sequence condition
  (∀ n, S_n n = 2017 * 2016^n - 2018 * t) →     -- Given sum formula
  t = 2017 / 2018 :=
by
  sorry

end find_t_l108_108974


namespace find_k_l108_108390

noncomputable def is_perfect_square (k : ℝ) : Prop :=
  ∀ x : ℝ, ∃ a : ℝ, x^2 + 2*(k-1)*x + 64 = (x + a)^2

theorem find_k (k : ℝ) : is_perfect_square k ↔ (k = 9 ∨ k = -7) :=
sorry

end find_k_l108_108390


namespace polynomial_coeff_sum_eq_neg_two_l108_108679

/-- If (1 - 2 * x) ^ 9 = a₉ * x ^ 9 + a₈ * x ^ 8 + ... + a₂ * x ^ 2 + a₁ * x + a₀, 
then a₁ + a₂ + ... + a₈ + a₉ = -2. -/
theorem polynomial_coeff_sum_eq_neg_two 
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℤ) 
  (h : (1 - 2 * x) ^ 9 = a₉ * x ^ 9 + a₈ * x ^ 8 + a₇ * x ^ 7 + a₆ * x ^ 6 + a₅ * x ^ 5 + a₄ * x ^ 4 + a₃ * x ^ 3 + a₂ * x ^ 2 + a₁ * x + a₀) : 
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -2 :=
by sorry

end polynomial_coeff_sum_eq_neg_two_l108_108679


namespace frank_bought_2_bags_of_chips_l108_108650

theorem frank_bought_2_bags_of_chips
  (cost_choco_bar : ℕ)
  (num_choco_bar : ℕ)
  (total_money : ℕ)
  (change : ℕ)
  (cost_bag_chip : ℕ)
  (num_bags_chip : ℕ)
  (h1 : cost_choco_bar = 2)
  (h2 : num_choco_bar = 5)
  (h3 : total_money = 20)
  (h4 : change = 4)
  (h5 : cost_bag_chip = 3)
  (h6 : total_money - change = (cost_choco_bar * num_choco_bar) + (cost_bag_chip * num_bags_chip)) :
  num_bags_chip = 2 := by
  sorry

end frank_bought_2_bags_of_chips_l108_108650


namespace find_n_l108_108946

theorem find_n (n : ℚ) (h : (1 / (n + 2) + 3 / (n + 2) + n / (n + 2) = 4)) : n = -4 / 3 :=
by
  sorry

end find_n_l108_108946


namespace smallest_solution_eq_l108_108133

theorem smallest_solution_eq :
  (∀ x : ℝ, x ≠ 3 →
  (3 * x / (x - 3) + (3 * x^2 - 27) / x = 15) → 
  x = 1 - Real.sqrt 10 ∨ (∃ y : ℝ, y ≤ 1 - Real.sqrt 10 ∧ y ≠ 3 ∧ 3 * y / (y - 3) + (3 * y^2 - 27) / y = 15)) :=
sorry

end smallest_solution_eq_l108_108133


namespace distance_from_origin_to_point_l108_108190

theorem distance_from_origin_to_point : 
  ∀ (x y : ℝ), x = 8 → y = -15 → real.sqrt ((x-0)^2 + (y-0)^2) = 17 :=
by
  intros x y hx hy
  rw [hx, hy]
  norm_num
  rw [real.sqrt_eq_rpow]
  norm_num
  sorry

end distance_from_origin_to_point_l108_108190


namespace black_lambs_correct_l108_108131

-- Define the total number of lambs
def total_lambs : ℕ := 6048

-- Define the number of white lambs
def white_lambs : ℕ := 193

-- Define the number of black lambs
def black_lambs : ℕ := total_lambs - white_lambs

-- The goal is to prove that the number of black lambs is 5855
theorem black_lambs_correct : black_lambs = 5855 := by
  sorry

end black_lambs_correct_l108_108131


namespace sqrt_49_times_sqrt25_eq_7_sqrt5_l108_108297

-- Definitions based on conditions
def sqrt_25 := Real.sqrt 25
def sqrt_49 := Real.sqrt 49

theorem sqrt_49_times_sqrt25_eq_7_sqrt5 :
  Real.sqrt (49 * sqrt_25) = 7 * Real.sqrt 5 := by
sorry

end sqrt_49_times_sqrt25_eq_7_sqrt5_l108_108297


namespace find_a_plus_c_l108_108868

theorem find_a_plus_c (a b c d : ℝ)
  (h₁ : -(3 - a) ^ 2 + b = 6) (h₂ : (3 - c) ^ 2 + d = 6)
  (h₃ : -(7 - a) ^ 2 + b = 2) (h₄ : (7 - c) ^ 2 + d = 2) :
  a + c = 10 := sorry

end find_a_plus_c_l108_108868


namespace necessary_but_not_sufficient_condition_l108_108988

theorem necessary_but_not_sufficient_condition (x : ℝ) (h₁ : 0 < x ∧ x < 5) :
    (| x - 1 | < 1) → False :=
by
  intro h
  have h₂ : 0 < x ∧ x < 2 := ⟨sorry, sorry⟩
  sorry

end necessary_but_not_sufficient_condition_l108_108988


namespace circle_intersection_range_l108_108514

theorem circle_intersection_range (m : ℝ) :
  (x^2 + y^2 - 4*x + 2*m*y + m + 6 = 0) ∧ 
  (∀ A B : ℝ, 
    (A - y = 0) ∧ (B - y = 0) → A * B > 0
  ) → 
  (m > 2 ∨ (-6 < m ∧ m < -2)) :=
by 
  sorry

end circle_intersection_range_l108_108514


namespace no_solution_for_x_l108_108033

theorem no_solution_for_x (m : ℝ) :
  (∀ x : ℝ, x ≠ 1 → (mx - 1) / (x - 1) ≠ 3) ↔ (m = 1 ∨ m = 3) :=
by
  sorry

end no_solution_for_x_l108_108033


namespace area_of_square_l108_108901

theorem area_of_square (side_length : ℝ) (h : side_length = 17) : side_length * side_length = 289 :=
by
  sorry

end area_of_square_l108_108901


namespace investment_payoff_period_l108_108720

noncomputable theory

def initialInvestment (systemUnitCost : ℕ) (graphicsCardCost : ℕ) (numGraphicsCards : ℕ) : ℕ :=
  systemUnitCost + (numGraphicsCards * graphicsCardCost)

def dailyRevenue (ethPerCardPerDay : ℝ) (numGraphicsCards : ℕ) (ethToRubRate : ℝ) : ℝ :=
  (ethPerCardPerDay * numGraphicsCards) * ethToRubRate

def dailyEnergyCost (systemUnitConsumption : ℕ) (graphicsCardConsumption : ℕ) (numGraphicsCards : ℕ) (electricityCostPerKWh : ℝ) : ℝ :=
  let totalWattage := systemUnitConsumption + (graphicsCardConsumption * numGraphicsCards)
  let dailyKWh := (totalWattage / 1000.0) * 24
  dailyKWh * electricityCostPerKWh

def netDailyProfit (dailyRevenue : ℝ) (dailyEnergyCost : ℝ) : ℝ :=
  dailyRevenue - dailyEnergyCost

def paybackPeriod (initialInvestment : ℕ) (netDailyProfit : ℝ) : ℝ :=
  initialInvestment / netDailyProfit

theorem investment_payoff_period
    (systemUnitCost : ℕ := 9499)
    (graphicsCardCost : ℕ := 20990)
    (numGraphicsCards : ℕ := 2)
    (ethPerCardPerDay : ℝ := 0.00630)
    (ethToRubRate : ℝ := 27790.37)
    (systemUnitConsumption : ℕ := 120)
    (graphicsCardConsumption : ℕ := 185)
    (electricityCostPerKWh : ℝ := 5.38)
    : paybackPeriod (initialInvestment systemUnitCost graphicsCardCost numGraphicsCards)
                    (netDailyProfit (dailyRevenue ethPerCardPerDay numGraphicsCards ethToRubRate)
                                    (dailyEnergyCost systemUnitConsumption graphicsCardConsumption numGraphicsCards electricityCostPerKWh)) ≈ 179 := by
  sorry

end investment_payoff_period_l108_108720


namespace natural_number_with_six_divisors_two_prime_sum_78_is_45_l108_108009

def has_six_divisors (n : ℕ) : Prop :=
  (∃ p1 p2 : ℕ, p1.prime ∧ p2.prime ∧ p1 ≠ p2 ∧ 
  (∃ α1 α2 : ℕ, α1 + α2 > 0 ∧ n = p1 ^ α1 * p2 ^ α2 ∧ 
  (α1 + 1) * (α2 + 1) = 6))

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ d, d > 0 ∧ n % d = 0) (Finset.range (n + 1))).sum id

theorem natural_number_with_six_divisors_two_prime_sum_78_is_45 (n : ℕ) :
  has_six_divisors n ∧ sum_of_divisors n = 78 → n = 45 := 
by 
  sorry

end natural_number_with_six_divisors_two_prime_sum_78_is_45_l108_108009


namespace evaluate_expression_l108_108639

theorem evaluate_expression : (1 / (2 + (1 / (3 + (1 / 4))))) = (13 / 30) :=
by
  sorry

end evaluate_expression_l108_108639


namespace f_12_eq_12_l108_108061

noncomputable def f : ℕ → ℤ := sorry

axiom f_int (n : ℕ) (hn : 0 < n) : ∃ k : ℤ, f n = k
axiom f_2 : f 2 = 2
axiom f_mul (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : f (m * n) = f m * f n
axiom f_monotonic (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : m > n → f m > f n

theorem f_12_eq_12 : f 12 = 12 := sorry

end f_12_eq_12_l108_108061


namespace loss_percentage_initial_selling_l108_108113

theorem loss_percentage_initial_selling (CP SP' : ℝ) 
  (hCP : CP = 1250) 
  (hSP' : SP' = CP * 1.15) 
  (h_diff : SP' - 500 = 937.5) : 
  (CP - 937.5) / CP * 100 = 25 := 
by 
  sorry

end loss_percentage_initial_selling_l108_108113


namespace find_N_l108_108890

theorem find_N : ∃ (N : ℤ), N > 0 ∧ (36^2 * 60^2 = 30^2 * N^2) ∧ (N = 72) :=
by
  sorry

end find_N_l108_108890


namespace find_a_value_l108_108022

theorem find_a_value (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) 
  (h3 : (∃ l : ℝ, ∃ f : ℝ → ℝ, f x = a^x ∧ deriv f 0 = -1)) :
  a = 1 / Real.exp 1 := by
  sorry

end find_a_value_l108_108022


namespace polar_to_rectangular_l108_108125

theorem polar_to_rectangular (r θ : ℝ) (hr : r = 6) (hθ : θ = 5 * Real.pi / 3) :
  (r * Real.cos θ, r * Real.sin θ) = (3, -3 * Real.sqrt 3) :=
by
  -- Definitions and assertions from the conditions
  have cos_theta : Real.cos (5 * Real.pi / 3) = 1 / 2 :=
    by sorry  -- detailed trigonometric proof is omitted
  have sin_theta : Real.sin (5 * Real.pi / 3) = - Real.sqrt 3 / 2 :=
    by sorry  -- detailed trigonometric proof is omitted

  -- Proof that the converted coordinates match the expected result
  rw [hr, hθ, cos_theta, sin_theta]
  simp
  -- Detailed proof steps to verify (6 * (1 / 2), 6 * (- Real.sqrt 3 / 2)) = (3, -3 * Real.sqrt 3) omitted
  sorry

end polar_to_rectangular_l108_108125


namespace gasoline_tank_capacity_l108_108320

theorem gasoline_tank_capacity (x : ℕ) (h1 : 5 * x / 6 - 2 * x / 3 = 15) : x = 90 :=
sorry

end gasoline_tank_capacity_l108_108320


namespace negation_p_l108_108857

open Nat

def p : Prop := ∀ n : ℕ, n^2 ≤ 2^n

theorem negation_p : ¬p ↔ ∃ n : ℕ, n^2 > 2^n :=
by
  sorry

end negation_p_l108_108857


namespace sum_in_range_l108_108631

theorem sum_in_range : 
    let a := (2:ℝ) + 1/8
    let b := (3:ℝ) + 1/3
    let c := (5:ℝ) + 1/18
    10.5 < a + b + c ∧ a + b + c < 11 := 
by 
    sorry

end sum_in_range_l108_108631


namespace initial_speed_100_l108_108907

/-- Conditions of the problem:
1. The total distance from A to D is 100 km.
2. At point B, the navigator shows that 30 minutes are remaining.
3. At point B, the motorist reduces his speed by 10 km/h.
4. At point C, the navigator shows 20 km remaining, and the motorist again reduces his speed by 10 km/h.
5. The distance from C to D is 20 km.
6. The journey from B to C took 5 minutes longer than from C to D.
-/
theorem initial_speed_100 (x v : ℝ) (h1 : x = 100 - v / 2)
  (h2 : ∀ t, t = x / v)
  (h3 : ∀ t1 t2, t1 = (80 - x) / (v - 10) ∧ t2 = 20 / (v - 20))
  (h4 : (80 - x) / (v - 10) - 20 / (v - 20) = 1/12) :
  v = 100 := 
sorry

end initial_speed_100_l108_108907


namespace probability_of_selecting_one_defective_l108_108758

-- Definitions based on conditions from the problem
def items : List ℕ := [0, 1, 2, 3]  -- 0 represents defective, 1, 2, 3 represent genuine

def sample_space : List (ℕ × ℕ) := 
  [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

def event_A : List (ℕ × ℕ) := 
  [(0, 1), (0, 2), (0, 3)]

-- The probability of event A, calculated based on the classical method
def probability_event_A : ℚ := event_A.length / sample_space.length

theorem probability_of_selecting_one_defective : 
  probability_event_A = 1 / 2 := by
  sorry

end probability_of_selecting_one_defective_l108_108758


namespace can_determine_counterfeit_l108_108355

-- Define the conditions of the problem
structure ProblemConditions where
  totalCoins : ℕ := 100
  exaggeration : ℕ

-- Define the problem statement
theorem can_determine_counterfeit (P : ProblemConditions) : 
  ∃ strategy : ℕ → Prop, 
    ∀ (k : ℕ), strategy P.exaggeration -> 
    (∀ i, i < 100 → (P.totalCoins = 100 ∧ ∃ n, n > 0 ∧ 
     ∀ j, j < P.totalCoins → (P.totalCoins = j + 1 ∨ P.totalCoins = 99 + j))) := 
sorry

end can_determine_counterfeit_l108_108355


namespace proof_BH_length_equals_lhs_rhs_l108_108047

noncomputable def calculate_BH_length : ℝ :=
  let AB := 3
  let BC := 4
  let CA := 5
  let AG := 4  -- Since AB < AG
  let AH := 6  -- AG < AH
  let GI := 3
  let HI := 8
  let GH := Real.sqrt (GI ^ 2 + HI ^ 2)
  let p := 3
  let q := 2
  let r := 73
  let s := 1
  3 + 2 * Real.sqrt 73

theorem proof_BH_length_equals_lhs_rhs :
  let BH := 3 + 2 * Real.sqrt 73
  calculate_BH_length = BH := by
    sorry

end proof_BH_length_equals_lhs_rhs_l108_108047


namespace sqrt_49_mul_sqrt_25_l108_108281

theorem sqrt_49_mul_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_l108_108281


namespace precision_of_21_658_billion_is_hundred_million_l108_108814

theorem precision_of_21_658_billion_is_hundred_million :
  (21.658 : ℝ) * 10^9 % (10^8) = 0 :=
by
  sorry

end precision_of_21_658_billion_is_hundred_million_l108_108814


namespace distance_from_origin_to_point_l108_108186

open Real

theorem distance_from_origin_to_point (x y : ℝ) :
  x = 8 →
  y = -15 →
  sqrt (x^2 + y^2) = 17 :=
by
  intros hx hy
  rw [hx, hy]
  norm_num
  rw ←sqrt_sq (show 0 ≤ 289 by norm_num)
  norm_num

end distance_from_origin_to_point_l108_108186


namespace sqrt_mul_sqrt_l108_108244

theorem sqrt_mul_sqrt (h1 : Real.sqrt 25 = 5) : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_mul_sqrt_l108_108244


namespace find_symmetric_point_l108_108511

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def plane (x y z : ℝ) : ℝ := 
  4 * x + 6 * y + 4 * z - 25

def symmetric_point (M M_prime : Point3D) (plane_eq : ℝ → ℝ → ℝ → ℝ) : Prop :=
  let t : ℝ := (1 / 4)
  let M0 : Point3D := { x := (1 + 4 * t), y := (6 * t), z := (1 + 4 * t) }
  let midpoint_x := (M.x + M_prime.x) / 2
  let midpoint_y := (M.y + M_prime.y) / 2
  let midpoint_z := (M.z + M_prime.z) / 2
  M0.x = midpoint_x ∧ M0.y = midpoint_y ∧ M0.z = midpoint_z ∧
  plane_eq M0.x M0.y M0.z = 0

def M : Point3D := { x := 1, y := 0, z := 1 }

def M_prime : Point3D := { x := 3, y := 3, z := 3 }

theorem find_symmetric_point : symmetric_point M M_prime plane := by
  -- the proof is omitted here
  sorry

end find_symmetric_point_l108_108511


namespace tan_alpha_eq_2_l108_108366

theorem tan_alpha_eq_2 (α : ℝ) (h : Real.tan α = 2) : (Real.cos α + 3 * Real.sin α) / (3 * Real.cos α - Real.sin α) = 7 := by
  sorry

end tan_alpha_eq_2_l108_108366


namespace remainder_when_divided_by_x_minus_1_l108_108018

noncomputable def p (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + a * x^2 + b * x + 12

theorem remainder_when_divided_by_x_minus_1 (a b : ℝ)
  (h1 : p a b (-2) = 0)
  (h2 : p a b 3 = 0) :
  p a b 1 = 18 :=
begin
  sorry
end

end remainder_when_divided_by_x_minus_1_l108_108018


namespace probability_one_defective_l108_108753

theorem probability_one_defective (g d : ℕ) (h_g : g = 3) (h_d : d = 1) : 
  let total_items := g + d in
  let sample_space := (total_items.choose 2).toFinset in
  let event_A := {x ∈ sample_space | x.count (0 = ∘ id) = 1} in
  (event_A.card : ℚ) / (sample_space.card : ℚ) = 1 / 2 :=
by
  sorry

end probability_one_defective_l108_108753


namespace find_f_nine_l108_108085

-- Define the function f that satisfies the conditions
def f (x : ℝ) : ℝ := sorry

-- Define the condition that f(x + y) = f(x) * f(y) for all real x and y
axiom functional_equation : ∀ (x y : ℝ), f (x + y) = f x * f y

-- Define the condition that f(3) = 4
axiom f_three : f 3 = 4

-- State the main theorem to prove that f(9) = 64
theorem find_f_nine : f 9 = 64 := by
  sorry

end find_f_nine_l108_108085


namespace reduced_less_than_scaled_l108_108337

-- Define the conditions
def original_flow_rate : ℝ := 5.0
def reduced_flow_rate : ℝ := 2.0
def scaled_flow_rate : ℝ := 0.6 * original_flow_rate

-- State the theorem we need to prove
theorem reduced_less_than_scaled : scaled_flow_rate - reduced_flow_rate = 1.0 := 
by
  -- insert the detailed proof steps here
  sorry

end reduced_less_than_scaled_l108_108337


namespace sin_sum_eq_sin_l108_108813

open Real

theorem sin_sum_eq_sin (a b : ℝ) :
  sin a + sin b = sin (a + b) →
  ∃ k l m : ℤ, (a = 2 * π * k) ∨ (b = 2 * π * l) ∨ (a + b = π + 2 * π * m) :=
by
  sorry

end sin_sum_eq_sin_l108_108813


namespace average_height_Heidi_Lola_l108_108983

theorem average_height_Heidi_Lola :
  (2.1 + 1.4) / 2 = 1.75 := by
  sorry

end average_height_Heidi_Lola_l108_108983


namespace trig_identity_l108_108656

theorem trig_identity (x : ℝ) (h : 2 * Real.cos x - 5 * Real.sin x = 3) :
  (Real.sin x + 2 * Real.cos x = 1 / 2) ∨ (Real.sin x + 2 * Real.cos x = 83 / 29) := sorry

end trig_identity_l108_108656


namespace two_students_follow_all_celebrities_l108_108238

theorem two_students_follow_all_celebrities :
  ∀ (students : Finset ℕ) (celebrities_followers : ℕ → Finset ℕ),
    (students.card = 120) →
    (∀ c : ℕ, c < 10 → (celebrities_followers c).card ≥ 85 ∧ (celebrities_followers c) ⊆ students) →
    ∃ (s1 s2 : ℕ), s1 ∈ students ∧ s2 ∈ students ∧ s1 ≠ s2 ∧
      (∀ c : ℕ, c < 10 → (s1 ∈ celebrities_followers c ∨ s2 ∈ celebrities_followers c)) :=
by
  intros students celebrities_followers h_students_card h_followers_cond
  sorry

end two_students_follow_all_celebrities_l108_108238


namespace area_of_field_l108_108912

-- Define the given conditions and the problem
theorem area_of_field (L W A : ℝ) (hL : L = 20) (hFencing : 2 * W + L = 88) (hA : A = L * W) : 
  A = 680 :=
by
  sorry

end area_of_field_l108_108912


namespace sum_of_numbers_is_60_l108_108729

-- Define the primary values used in the conditions
variables (a b c : ℝ)

-- Define the conditions in the problem
def mean_condition_1 : Prop := (a + b + c) / 3 = a + 20
def mean_condition_2 : Prop := (a + b + c) / 3 = c - 30
def median_condition : Prop := b = 10

-- Prove that the sum of the numbers is 60 given the conditions
theorem sum_of_numbers_is_60 (hac1 : mean_condition_1 a b c) (hac2 : mean_condition_2 a b c) (hbm : median_condition b) : a + b + c = 60 :=
by 
  sorry

end sum_of_numbers_is_60_l108_108729


namespace cuboid_surface_area_4_8_6_l108_108095

noncomputable def cuboid_surface_area (length width height : ℕ) : ℕ :=
  2 * (length * width + length * height + width * height)

theorem cuboid_surface_area_4_8_6 : cuboid_surface_area 4 8 6 = 208 := by
  sorry

end cuboid_surface_area_4_8_6_l108_108095


namespace hypotenuse_of_45_45_90_triangle_l108_108562

theorem hypotenuse_of_45_45_90_triangle (a : ℝ) (h : ℝ) 
  (ha : a = 15) 
  (angle_opposite_leg : ℝ) 
  (h_angle : angle_opposite_leg = 45) 
  (right_triangle : ∃ θ : ℝ, θ = 90) : 
  h = 15 * Real.sqrt 2 := 
sorry

end hypotenuse_of_45_45_90_triangle_l108_108562


namespace probability_of_yellow_ball_is_correct_l108_108444

-- Defining the conditions
def red_balls : ℕ := 2
def yellow_balls : ℕ := 5
def blue_balls : ℕ := 4

-- Define the total number of balls
def total_balls : ℕ := red_balls + yellow_balls + blue_balls

-- Define the probability of choosing a yellow ball
def probability_yellow_ball : ℚ := yellow_balls / total_balls

-- The theorem statement we need to prove
theorem probability_of_yellow_ball_is_correct :
  probability_yellow_ball = 5 / 11 :=
sorry

end probability_of_yellow_ball_is_correct_l108_108444


namespace inequality_proof_l108_108368

theorem inequality_proof
  (a b c d : ℝ) 
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d)
  (h_cond : a * b + b * c + c * d + d * a = 1) :
  (a^3 / (b + c + d) + b^3 / (c + d + a) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1 / 3) :=
by {
  sorry
}

end inequality_proof_l108_108368


namespace gcd_f100_f101_l108_108422

def f (x : ℤ) : ℤ := x^2 - 3 * x + 2023

theorem gcd_f100_f101 : Int.gcd (f 100) (f 101) = 2 :=
by
  sorry

end gcd_f100_f101_l108_108422


namespace derek_history_test_l108_108634

theorem derek_history_test :
  let ancient_questions := 20
  let medieval_questions := 25
  let modern_questions := 35
  let total_questions := ancient_questions + medieval_questions + modern_questions

  let derek_ancient_correct := 0.60 * ancient_questions
  let derek_medieval_correct := 0.56 * medieval_questions
  let derek_modern_correct := 0.70 * modern_questions

  let derek_total_correct := derek_ancient_correct + derek_medieval_correct + derek_modern_correct

  let passing_score := 0.65 * total_questions
  (derek_total_correct < passing_score) →
  passing_score - derek_total_correct = 2
  := by
  sorry

end derek_history_test_l108_108634


namespace sqrt_nested_l108_108304

theorem sqrt_nested : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_nested_l108_108304


namespace marked_price_percentage_l108_108112

theorem marked_price_percentage
  (CP MP SP : ℝ)
  (h_profit : SP = 1.08 * CP)
  (h_discount : SP = 0.8307692307692308 * MP) :
  MP = CP * 1.3 :=
by sorry

end marked_price_percentage_l108_108112


namespace sum_of_numbers_l108_108228

noncomputable def mean (a b c : ℕ) : ℕ := (a + b + c) / 3

theorem sum_of_numbers (a b c : ℕ) (h1 : mean a b c = a + 8)
  (h2 : mean a b c = c - 20) (h3 : b = 7) (h_le1 : a ≤ b) (h_le2 : b ≤ c) :
  a + b + c = 57 :=
by {
  sorry
}

end sum_of_numbers_l108_108228


namespace payback_period_l108_108719

def system_unit_cost : ℕ := 9499 -- cost in RUB
def graphics_card_cost : ℕ := 20990 -- cost per card in RUB
def num_graphics_cards : ℕ := 2
def system_unit_power : ℕ := 120 -- power in watts
def graphics_card_power : ℕ := 185 -- power per card in watts
def earnings_per_card_per_day_ethereum : ℚ := 0.00630
def ethereum_to_rub : ℚ := 27790.37 -- RUB per ETH
def electricity_cost_per_kwh : ℚ := 5.38 -- RUB per kWh
def total_investment : ℕ := system_unit_cost + num_graphics_cards * graphics_card_cost
def total_power_consumption_watts : ℕ := system_unit_power + num_graphics_cards * graphics_card_power
def total_power_consumption_kwh_per_day : ℚ := total_power_consumption_watts / 1000 * 24
def daily_earnings_rub : ℚ := earnings_per_card_per_day_ethereum * num_graphics_cards * ethereum_to_rub
def daily_energy_cost : ℚ := total_power_consumption_kwh_per_day * electricity_cost_per_kwh
def net_daily_profit : ℚ := daily_earnings_rub - daily_energy_cost

theorem payback_period : total_investment / net_daily_profit = 179 := by
  sorry

end payback_period_l108_108719


namespace obtuse_triangle_side_range_l108_108374

theorem obtuse_triangle_side_range
  (a : ℝ)
  (h1 : a > 0)
  (h2 : (a + 4)^2 > a^2 + (a + 2)^2)
  (h3 : (a + 2)^2 + (a + 4)^2 < a^2) : 
  2 < a ∧ a < 6 := 
sorry

end obtuse_triangle_side_range_l108_108374


namespace bus_car_ratio_l108_108087

variable (R C Y : ℝ)

noncomputable def ratio_of_bus_to_car (R C Y : ℝ) : ℝ :=
  R / C

theorem bus_car_ratio 
  (h1 : R = 48) 
  (h2 : Y = 3.5 * C) 
  (h3 : Y = R - 6) : 
  ratio_of_bus_to_car R C Y = 4 :=
by sorry

end bus_car_ratio_l108_108087


namespace total_population_expr_l108_108534

-- Definitions of the quantities
variables (b g t : ℕ)

-- Conditions
axiom boys_as_girls : b = 3 * g
axiom girls_as_teachers : g = 9 * t

-- Theorem to prove
theorem total_population_expr : b + g + t = 37 * b / 27 :=
by
  sorry

end total_population_expr_l108_108534


namespace investment_at_6_percent_l108_108487

theorem investment_at_6_percent
  (x y : ℝ) 
  (total_investment : x + y = 15000)
  (total_interest : 0.06 * x + 0.075 * y = 1023) :
  x = 6800 :=
sorry

end investment_at_6_percent_l108_108487


namespace find_r_s_l108_108016

noncomputable def r_s_proof_problem (r s : ℕ) (h1 : Nat.gcd r s = 1) (h2 : (r : ℝ) / s = (2 * (Real.sqrt 2 + Real.sqrt 10)) / (5 * Real.sqrt (3 + Real.sqrt 5))) : Prop :=
(r, s) = (4, 5)

theorem find_r_s (r s : ℕ) (h1 : Nat.gcd r s = 1) (h2 : (r : ℝ) / s = (2 * (Real.sqrt 2 + Real.sqrt 10)) / (5 * Real.sqrt (3 + Real.sqrt 5))) : r_s_proof_problem r s h1 h2 :=
sorry

end find_r_s_l108_108016


namespace andy_tomatoes_left_l108_108340

theorem andy_tomatoes_left :
  let plants := 50
  let tomatoes_per_plant := 15
  let total_tomatoes := plants * tomatoes_per_plant
  let tomatoes_dried := (2 / 3) * total_tomatoes
  let tomatoes_left_after_drying := total_tomatoes - tomatoes_dried
  let tomatoes_for_marinara := (1 / 2) * tomatoes_left_after_drying
  let tomatoes_left := tomatoes_left_after_drying - tomatoes_for_marinara
  tomatoes_left = 125 := sorry

end andy_tomatoes_left_l108_108340


namespace ellipse_foci_k_value_l108_108725

theorem ellipse_foci_k_value 
    (k : ℝ) 
    (h1 : 5 * (0:ℝ)^2 + k * (2:ℝ)^2 = 5): 
    k = 1 := 
by 
  sorry

end ellipse_foci_k_value_l108_108725


namespace sqrt_49_times_sqrt25_eq_7_sqrt5_l108_108299

-- Definitions based on conditions
def sqrt_25 := Real.sqrt 25
def sqrt_49 := Real.sqrt 49

theorem sqrt_49_times_sqrt25_eq_7_sqrt5 :
  Real.sqrt (49 * sqrt_25) = 7 * Real.sqrt 5 := by
sorry

end sqrt_49_times_sqrt25_eq_7_sqrt5_l108_108299


namespace rolls_sold_to_uncle_l108_108363

theorem rolls_sold_to_uncle (total_rolls needed_rolls rolls_to_grandmother rolls_to_neighbor rolls_to_uncle : ℕ)
  (h1 : total_rolls = 45)
  (h2 : needed_rolls = 28)
  (h3 : rolls_to_grandmother = 1)
  (h4 : rolls_to_neighbor = 6)
  (h5 : rolls_to_uncle + rolls_to_grandmother + rolls_to_neighbor + needed_rolls = total_rolls) :
  rolls_to_uncle = 10 :=
by {
  sorry
}

end rolls_sold_to_uncle_l108_108363


namespace geometric_proportion_l108_108458

theorem geometric_proportion (a b c d : ℝ) (h1 : a / b = c / d) (h2 : a / b = d / c) :
  (a = b ∧ b = c ∧ c = d) ∨ (|a| = |b| ∧ |b| = |c| ∧ |c| = |d| ∧ (a * b * c * d < 0)) :=
by
  sorry

end geometric_proportion_l108_108458


namespace base_case_n_equals_1_l108_108573

variable {a : ℝ}
variable {n : ℕ}

theorem base_case_n_equals_1 (h1 : a ≠ 1) (h2 : n = 1) : 1 + a = 1 + a :=
by
  sorry

end base_case_n_equals_1_l108_108573


namespace remainder_11_pow_1000_mod_500_l108_108596

theorem remainder_11_pow_1000_mod_500 : (11 ^ 1000) % 500 = 1 :=
by
  have h1 : 11 % 5 = 1 := by norm_num
  have h2 : (11 ^ 10) % 100 = 1 := by
    -- Some steps omitted to satisfy conditions; normally would be generalized
    sorry
  have h3 : 500 = 5 * 100 := by norm_num
  -- Further omitted steps aligning with the Chinese Remainder Theorem application.
  sorry

end remainder_11_pow_1000_mod_500_l108_108596


namespace minimal_polynomial_correct_l108_108645

noncomputable def minimal_polynomial : Polynomial ℚ :=
  (Polynomial.X^2 - 4 * Polynomial.X + 1) * (Polynomial.X^2 - 6 * Polynomial.X + 2)

theorem minimal_polynomial_correct :
  Polynomial.X^4 - 10 * Polynomial.X^3 + 29 * Polynomial.X^2 - 26 * Polynomial.X + 2 = minimal_polynomial :=
  sorry

end minimal_polynomial_correct_l108_108645


namespace find_B_divisible_by_6_l108_108735

theorem find_B_divisible_by_6 (B : ℕ) : (5170 + B) % 6 = 0 ↔ (B = 2 ∨ B = 8) :=
by
  -- Conditions extracted from the problem are directly used here:
  sorry -- Proof would be here

end find_B_divisible_by_6_l108_108735


namespace mass_percentage_Al_in_AlBr3_l108_108460

theorem mass_percentage_Al_in_AlBr3 
  (molar_mass_Al : Real := 26.98) 
  (molar_mass_Br : Real := 79.90) 
  (molar_mass_AlBr3 : Real := molar_mass_Al + 3 * molar_mass_Br)
  : (molar_mass_Al / molar_mass_AlBr3) * 100 = 10.11 := 
by 
  -- Here we would provide the proof; skipping with sorry
  sorry

end mass_percentage_Al_in_AlBr3_l108_108460


namespace price_equivalence_l108_108836

theorem price_equivalence : 
  (∀ a o p : ℕ, 10 * a = 5 * o ∧ 4 * o = 6 * p) → 
  (∀ a o p : ℕ, 20 * a = 15 * p) :=
by
  intro h
  sorry

end price_equivalence_l108_108836


namespace total_blocks_traveled_l108_108341

-- Given conditions as definitions
def annie_walked_blocks : ℕ := 5
def annie_rode_blocks : ℕ := 7

-- The total blocks Annie traveled
theorem total_blocks_traveled : annie_walked_blocks + annie_rode_blocks + (annie_walked_blocks + annie_rode_blocks) = 24 := by
  sorry

end total_blocks_traveled_l108_108341


namespace johns_friends_count_l108_108694

-- Define the conditions
def total_cost : ℕ := 12100
def cost_per_person : ℕ := 1100

-- Define the theorem to prove the number of friends John is going with
theorem johns_friends_count (total_cost cost_per_person : ℕ) (h1 : total_cost = 12100) (h2 : cost_per_person = 1100) : (total_cost / cost_per_person) - 1 = 10 := by
  -- Providing the proof is not required, so we use sorry to skip it
  sorry

end johns_friends_count_l108_108694


namespace find_d_plus_f_l108_108449

noncomputable def a : ℂ := sorry
noncomputable def c : ℂ := sorry
noncomputable def e : ℂ := -2 * a - c
noncomputable def d : ℝ := sorry
noncomputable def f : ℝ := sorry

theorem find_d_plus_f (a c e : ℂ) (d f : ℝ) (h₁ : e = -2 * a - c) (h₂ : a.im + d + f = 4) (h₃ : a.re + c.re + e.re = 0) (h₄ : 2 + d + f = 4) : d + f = 2 :=
by
  -- proof goes here
  sorry

end find_d_plus_f_l108_108449


namespace roberto_raise_percentage_l108_108428

theorem roberto_raise_percentage
    (starting_salary : ℝ)
    (previous_salary : ℝ)
    (current_salary : ℝ)
    (h1 : starting_salary = 80000)
    (h2 : previous_salary = starting_salary * 1.40)
    (h3 : current_salary = 134400) :
    ((current_salary - previous_salary) / previous_salary) * 100 = 20 :=
by sorry

end roberto_raise_percentage_l108_108428


namespace correctly_subtracted_value_l108_108601

theorem correctly_subtracted_value (x : ℤ) (h1 : 122 = x - 64) : 
  x - 46 = 140 :=
by
  -- Proof goes here
  sorry

end correctly_subtracted_value_l108_108601


namespace sqrt_mul_sqrt_l108_108246

theorem sqrt_mul_sqrt (h1 : Real.sqrt 25 = 5) : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_mul_sqrt_l108_108246


namespace digit_7_count_in_range_l108_108050

def count_digit_7 : ℕ :=
  let units_place := (107 - 100) / 10 + 1
  let tens_place := 10
  units_place + tens_place

theorem digit_7_count_in_range : count_digit_7 = 20 := by
  sorry

end digit_7_count_in_range_l108_108050


namespace distance_to_canada_l108_108546

theorem distance_to_canada (speed : ℝ) (total_time : ℝ) (stop_time : ℝ) (driving_time : ℝ) (distance : ℝ) :
  speed = 60 ∧ total_time = 7 ∧ stop_time = 1 ∧ driving_time = total_time - stop_time ∧
  distance = speed * driving_time → distance = 360 :=
by
  sorry

end distance_to_canada_l108_108546


namespace billiard_ball_hits_top_left_pocket_l108_108486

/--
A ball is released from the bottom left pocket of a rectangular billiard table with dimensions
26 × 1965 (with the longer side 1965 running left to right and the shorter side 26 running top
to bottom) at an angle of 45° to the sides. Pockets are located at the corners of the rectangle.
Prove that after several reflections off the sides, the ball will fall into the top left pocket.
--/
theorem billiard_ball_hits_top_left_pocket 
  (table_width : ℕ) (table_height : ℕ) (angle : ℝ)
  (initial_position : ℕ × ℕ) (target_position : ℕ × ℕ) :
  table_width = 1965 → 
  table_height = 26 → 
  angle = real.pi / 4 → 
  initial_position = (0, 0) → 
  target_position = (0, 26) →
  ∃ (m n : ℕ), 2 * m = 151 * n :=
by sorry

end billiard_ball_hits_top_left_pocket_l108_108486


namespace minimum_value_of_expression_l108_108922

theorem minimum_value_of_expression (x y : ℝ) : 
    ∃ (x y : ℝ), (2 * x * y - 1) ^ 2 + (x - y) ^ 2 = 0 :=
by
  sorry

end minimum_value_of_expression_l108_108922


namespace numbers_not_as_difference_of_squares_l108_108945

theorem numbers_not_as_difference_of_squares :
  {n : ℕ | ¬ ∃ x y : ℕ, x^2 - y^2 = n} = {1, 4} ∪ {4*k + 2 | k : ℕ} :=
by sorry

end numbers_not_as_difference_of_squares_l108_108945


namespace factor_difference_of_squares_l108_108935

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) :=
by
  sorry

end factor_difference_of_squares_l108_108935


namespace pawpaws_basket_l108_108863

variable (total_fruits mangoes pears lemons kiwis : ℕ)
variable (pawpaws : ℕ)

theorem pawpaws_basket
  (h1 : total_fruits = 58)
  (h2 : mangoes = 18)
  (h3 : pears = 10)
  (h4 : lemons = 9)
  (h5 : kiwis = 9)
  (h6 : total_fruits = mangoes + pears + lemons + kiwis + pawpaws) :
  pawpaws = 12 := by
  sorry

end pawpaws_basket_l108_108863


namespace problem_statement_l108_108391

theorem problem_statement (x : ℚ) (h : 8 * x = 3) : 200 * (1 / x) = 1600 / 3 :=
by
  sorry

end problem_statement_l108_108391


namespace not_perfect_square_infinitely_many_l108_108695

theorem not_perfect_square_infinitely_many (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_gt : b > a) (h_prime : Prime (b - a)) :
  ∃ᶠ n in at_top, ¬ IsSquare ((a ^ n + a + 1) * (b ^ n + b + 1)) :=
sorry

end not_perfect_square_infinitely_many_l108_108695


namespace Valleyball_Soccer_League_members_l108_108543

theorem Valleyball_Soccer_League_members (cost_socks cost_tshirt total_expenditure cost_per_member: ℕ) (h1 : cost_socks = 6) (h2 : cost_tshirt = cost_socks + 8) (h3 : total_expenditure = 3740) (h4 : cost_per_member = cost_socks + 2 * cost_tshirt) : 
  total_expenditure = 3740 → cost_per_member = 34 → total_expenditure / cost_per_member = 110 :=
sorry

end Valleyball_Soccer_League_members_l108_108543


namespace exists_x_eq_1_l108_108448

theorem exists_x_eq_1 (x y z t : ℕ) (h : x + y + z + t = 10) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (ht : 0 < t) :
  ∃ x, x = 1 :=
sorry

end exists_x_eq_1_l108_108448


namespace present_age_of_B_l108_108897

-- Definitions
variables (a b : ℕ)

-- Conditions
def condition1 (a b : ℕ) : Prop := a + 10 = 2 * (b - 10)
def condition2 (a b : ℕ) : Prop := a = b + 7

-- Theorem to prove
theorem present_age_of_B (a b : ℕ) (h1 : condition1 a b) (h2 : condition2 a b) : b = 37 := by
  sorry

end present_age_of_B_l108_108897


namespace hypotenuse_of_454590_triangle_l108_108559

theorem hypotenuse_of_454590_triangle (l : ℝ) (angle : ℝ) (h : ℝ) (h_leg : l = 15) (h_angle : angle = 45) :
  h = l * Real.sqrt 2 := 
  sorry

end hypotenuse_of_454590_triangle_l108_108559


namespace power_function_value_l108_108838

theorem power_function_value (a : ℝ) (f : ℝ → ℝ)
  (h₁ : ∀ x, f x = x ^ a)
  (h₂ : f 2 = 4) :
  f 9 = 81 :=
by
  sorry

end power_function_value_l108_108838


namespace gold_tetrahedron_volume_l108_108350

theorem gold_tetrahedron_volume (side_length : ℝ) (h : side_length = 8) : 
  volume_of_tetrahedron_with_gold_vertices = 170.67 := 
by 
  sorry

end gold_tetrahedron_volume_l108_108350


namespace smallest_x_l108_108130

theorem smallest_x (M : ℤ) (x : ℕ) (hx_pos : 0 < x)
  (h_factorization : 1800 = 2^3 * 3^2 * 5^2)
  (h_eq : 1800 * x = M^3) : x = 15 :=
sorry

end smallest_x_l108_108130


namespace quadratic_roots_and_T_range_l108_108418

theorem quadratic_roots_and_T_range
  (m : ℝ)
  (h1 : m ≥ -1)
  (x1 x2 : ℝ)
  (h2 : x1^2 + 2*(m-2)*x1 + (m^2 - 3*m + 3) = 0)
  (h3 : x2^2 + 2*(m-2)*x2 + (m^2 - 3*m + 3) = 0)
  (h4 : x1 ≠ x2)
  (h5 : x1^2 + x2^2 = 6) :
  m = (5 - Real.sqrt 17) / 2 ∧ (0 < ((m * x1) / (1 - x1) + (m * x2) / (1 - x2)) ∧ ((m * x1) / (1 - x1) + (m * x2) / (1 - x2)) ≤ 4 ∧ ((m * x1) / (1 - x1) + (m * x2) / (1 - x2)) ≠ 2) :=
by
  sorry

end quadratic_roots_and_T_range_l108_108418


namespace sufficient_not_necessary_example_l108_108659

lemma sufficient_but_not_necessary_condition (x y : ℝ) (hx : x >= 2) (hy : y >= 2) : x^2 + y^2 >= 4 :=
by
  -- We only need to state the lemma, so the proof is omitted.
  sorry

theorem sufficient_not_necessary_example :
  ¬(∀ x y : ℝ, (x^2 + y^2 >= 4) -> (x >= 2) ∧ (y >= 2)) :=
by 
  -- We only need to state the theorem, so the proof is omitted.
  sorry

end sufficient_not_necessary_example_l108_108659


namespace gcd_consecutive_digits_l108_108947

theorem gcd_consecutive_digits (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) 
  (h₁ : b = a + 1) (h₂ : c = a + 2) (h₃ : d = a + 3) :
  ∃ g, g = gcd (1000 * a + 100 * b + 10 * c + d - (1000 * d + 100 * c + 10 * b + a)) 3096 :=
by {
  sorry
}

end gcd_consecutive_digits_l108_108947


namespace jean_candy_count_l108_108547

theorem jean_candy_count : ∃ C : ℕ, 
  C - 7 = 16 ∧ 
  (C - 7 + 7 = C) ∧ 
  (C - 7 = 16) ∧ 
  (C + 0 = C) ∧
  (C - 7 = 16) :=
by 
  sorry 

end jean_candy_count_l108_108547


namespace min_sum_areas_of_triangles_l108_108969

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1 / 4, 0)

def parabola (p : ℝ × ℝ) : Prop := p.2^2 = p.1

def O := (0, 0)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def on_opposite_sides_x_axis (p q : ℝ × ℝ) : Prop := p.2 * q.2 < 0

theorem min_sum_areas_of_triangles 
  (A B : ℝ × ℝ)
  (hA : parabola A)
  (hB : parabola B)
  (hAB : on_opposite_sides_x_axis A B)
  (h_dot : dot_product A B = 2) :
  ∃ m : ℝ, m = 3 := by
  sorry

end min_sum_areas_of_triangles_l108_108969


namespace range_of_a_l108_108666

theorem range_of_a (a : ℝ) (x y : ℝ) (hxy : x * y > 0) (hx : 0 < x) (hy : 0 < y) :
  (x + y) * (1 / x + a / y) ≥ 9 → a ≥ 4 :=
by
  intro h
  sorry

end range_of_a_l108_108666


namespace divisor_of_number_l108_108910

theorem divisor_of_number (n d q p : ℤ) 
  (h₁ : n = d * q + 3)
  (h₂ : n ^ 2 = d * p + 3) : 
  d = 6 := 
sorry

end divisor_of_number_l108_108910


namespace distinct_convex_polygons_of_four_or_more_sides_l108_108504

noncomputable def total_subsets (n : Nat) : Nat := 2^n

noncomputable def subsets_with_fewer_than_four_members (n : Nat) : Nat := 
  (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)

noncomputable def valid_subsets (n : Nat) : Nat := 
  total_subsets n - subsets_with_fewer_than_four_members n

theorem distinct_convex_polygons_of_four_or_more_sides (n : Nat) (h : n = 15) : valid_subsets n = 32192 := by
  sorry

end distinct_convex_polygons_of_four_or_more_sides_l108_108504


namespace age_of_teacher_l108_108579

theorem age_of_teacher (avg_age_students : ℕ) (num_students : ℕ) (inc_avg_with_teacher : ℕ) (num_people_with_teacher : ℕ) :
  avg_age_students = 21 →
  num_students = 20 →
  inc_avg_with_teacher = 22 →
  num_people_with_teacher = 21 →
  let total_age_students := num_students * avg_age_students
  let total_age_with_teacher := num_people_with_teacher * inc_avg_with_teacher
  total_age_with_teacher - total_age_students = 42 :=
by
  intros
  sorry

end age_of_teacher_l108_108579


namespace sqrt_product_l108_108277

theorem sqrt_product (a b : ℝ) (h1 : a = 49) (h2 : b = 25) : sqrt (a * sqrt b) = 7 * sqrt 5 :=
by sorry

end sqrt_product_l108_108277


namespace largest_x_l108_108948

theorem largest_x (x : ℝ) : 
  (∃ x, (15 * x ^ 2 - 40 * x + 18) / (4 * x - 3) + 6 * x = 7 * x - 2) → 
  (x ≤ 1) := sorry

end largest_x_l108_108948


namespace sum_of_bases_l108_108399

theorem sum_of_bases (S₁ S₂ G₁ G₂ : ℚ)
  (h₁ : G₁ = 4 * S₁ / (S₁^2 - 1) + 8 / (S₁^2 - 1))
  (h₂ : G₂ = 8 * S₁ / (S₁^2 - 1) + 4 / (S₁^2 - 1))
  (h₃ : G₁ = 3 * S₂ / (S₂^2 - 1) + 6 / (S₂^2 - 1))
  (h₄ : G₂ = 6 * S₂ / (S₂^2 - 1) + 3 / (S₂^2 - 1)) :
  S₁ + S₂ = 23 :=
by
  sorry

end sum_of_bases_l108_108399


namespace prove_a2_minus_b2_l108_108175

theorem prove_a2_minus_b2 : 
  ∀ (a b : ℚ), 
  a + b = 9 / 17 ∧ a - b = 1 / 51 → a^2 - b^2 = 3 / 289 :=
by
  intros a b h
  cases' h
  sorry

end prove_a2_minus_b2_l108_108175


namespace original_amount_l108_108913

theorem original_amount {P : ℕ} {R : ℕ} {T : ℕ} (h1 : P = 1000) (h2 : T = 5) 
  (h3 : ∃ R, (1000 * (R + 5) * 5) / 100 + 1000 = 1750) : 
  1000 + (1000 * R * 5 / 100) = 1500 :=
by
  sorry

end original_amount_l108_108913


namespace players_taking_chemistry_l108_108496

theorem players_taking_chemistry (total_players biology_players both_sci_players: ℕ) 
  (h1 : total_players = 12)
  (h2 : biology_players = 7)
  (h3 : both_sci_players = 2)
  (h4 : ∀ p, p <= total_players) : 
  ∃ chemistry_players, chemistry_players = 7 := 
sorry

end players_taking_chemistry_l108_108496


namespace complex_expression_evaluation_l108_108375

theorem complex_expression_evaluation (z : ℂ) (h : z = 1 - I) :
  (z^2 - 2 * z) / (z - 1) = -2 * I :=
by
  sorry

end complex_expression_evaluation_l108_108375


namespace tissue_magnification_l108_108916

theorem tissue_magnification
  (diameter_magnified : ℝ)
  (diameter_actual : ℝ)
  (h1 : diameter_magnified = 5)
  (h2 : diameter_actual = 0.005) :
  diameter_magnified / diameter_actual = 1000 :=
by
  -- proof goes here
  sorry

end tissue_magnification_l108_108916


namespace factor_difference_of_squares_l108_108931
noncomputable def factored (t : ℝ) : ℝ := (t - 8) * (t + 8)

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = factored t := by
  sorry

end factor_difference_of_squares_l108_108931


namespace order_of_abc_l108_108653

noncomputable def a : ℝ := 0.1 * Real.exp 0.1
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := -Real.log 0.9

theorem order_of_abc : b > a ∧ a > c :=
by
  sorry

end order_of_abc_l108_108653


namespace problem1_problem2_l108_108951

theorem problem1 (x : ℝ) (h : 4 * x^2 - 9 = 0) : x = 3/2 ∨ x = -3/2 :=
by
  sorry

theorem problem2 (x : ℝ) (h : 64 * (x-2)^3 - 1 = 0) : x = 2 + 1/4 :=
by
  sorry

end problem1_problem2_l108_108951


namespace sqrt_nested_l108_108307

theorem sqrt_nested : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_nested_l108_108307


namespace michelle_sandwiches_l108_108705

def sandwiches_left (total : ℕ) (given_to_coworker : ℕ) (kept : ℕ) : ℕ :=
  total - given_to_coworker - kept

theorem michelle_sandwiches : sandwiches_left 20 4 (4 * 2) = 8 :=
by
  sorry

end michelle_sandwiches_l108_108705


namespace solve_inequality_range_of_a_l108_108829

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |2 * x + 2|

theorem solve_inequality : {x : ℝ | f x > 5} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 4 / 3} :=
by
  sorry

theorem range_of_a (a : ℝ) (h : ∀ x, ¬ (f x < a)) : a ≤ 2 :=
by
  sorry

end solve_inequality_range_of_a_l108_108829


namespace smallest_positive_b_l108_108129

theorem smallest_positive_b (b : ℕ) : 
  (b % 3 = 2) ∧ 
  (b % 4 = 3) ∧ 
  (b % 5 = 4) ∧ 
  (b % 6 = 5) ↔ 
  b = 59 :=
by
  sorry

end smallest_positive_b_l108_108129


namespace arabella_total_learning_time_l108_108494

-- Define the conditions
def arabella_first_step_time := 30 -- in minutes
def arabella_second_step_time := arabella_first_step_time / 2 -- half the time of the first step
def arabella_third_step_time := arabella_first_step_time + arabella_second_step_time -- sum of the first and second steps

-- Define the total time spent
def arabella_total_time := arabella_first_step_time + arabella_second_step_time + arabella_third_step_time

-- The theorem to prove
theorem arabella_total_learning_time : arabella_total_time = 90 := 
  sorry

end arabella_total_learning_time_l108_108494


namespace six_digit_number_condition_l108_108415

theorem six_digit_number_condition (a b c : ℕ) (h : 1 ≤ a ∧ a ≤ 9) (hb : b < 10) (hc : c < 10) : 
  ∃ k : ℕ, 100000 * a + 10000 * b + 1000 * c + 100 * (2 * a) + 10 * (2 * b) + 2 * c = 2 * k := 
by
  sorry

end six_digit_number_condition_l108_108415


namespace angle_sum_l108_108685

theorem angle_sum (y : ℝ) (h : 3 * y + y = 120) : y = 30 :=
sorry

end angle_sum_l108_108685


namespace Nellie_needs_to_sell_more_rolls_l108_108953

-- Define the conditions
def total_needed : ℕ := 45
def sold_grandmother : ℕ := 1
def sold_uncle : ℕ := 10
def sold_neighbor : ℕ := 6

-- Define the total sold
def total_sold : ℕ := sold_grandmother + sold_uncle + sold_neighbor

-- Define the remaining rolls needed
def remaining_rolls := total_needed - total_sold

-- Statement to prove that remaining_rolls equals 28
theorem Nellie_needs_to_sell_more_rolls : remaining_rolls = 28 := by
  unfold remaining_rolls
  unfold total_sold
  unfold total_needed sold_grandmother sold_uncle sold_neighbor
  calc
  45 - (1 + 10 + 6) = 45 - 17 : by rw [Nat.add_assoc]
  ... = 28 : by norm_num

end Nellie_needs_to_sell_more_rolls_l108_108953


namespace arabella_total_learning_time_l108_108495

-- Define the conditions
def arabella_first_step_time := 30 -- in minutes
def arabella_second_step_time := arabella_first_step_time / 2 -- half the time of the first step
def arabella_third_step_time := arabella_first_step_time + arabella_second_step_time -- sum of the first and second steps

-- Define the total time spent
def arabella_total_time := arabella_first_step_time + arabella_second_step_time + arabella_third_step_time

-- The theorem to prove
theorem arabella_total_learning_time : arabella_total_time = 90 := 
  sorry

end arabella_total_learning_time_l108_108495


namespace functional_equation_solution_l108_108641

theorem functional_equation_solution (f : ℝ → ℝ)
  (h : ∀ x y, f (x ^ 2) - f (y ^ 2) + 2 * x + 1 = f (x + y) * f (x - y)) :
  (∀ x, f x = x + 1) ∨ (∀ x, f x = -x - 1) :=
by
  sorry

end functional_equation_solution_l108_108641


namespace julia_money_given_l108_108998

-- Define the conditions
def num_snickers : ℕ := 2
def num_mms : ℕ := 3
def cost_snickers : ℚ := 1.5
def cost_mms : ℚ := 2 * cost_snickers
def change_received : ℚ := 8

-- The total cost Julia had to pay
def total_cost : ℚ := (num_snickers * cost_snickers) + (num_mms * cost_mms)

-- Julia gave this amount of money to the cashier
def money_given : ℚ := total_cost + change_received

-- The problem to prove
theorem julia_money_given : money_given = 20 := by
  sorry

end julia_money_given_l108_108998


namespace quadratic_solution_l108_108141

theorem quadratic_solution (m : ℝ) (x₁ x₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂) 
  (h_roots : ∀ x, 2 * x^2 + 4 * m * x + m = 0 ↔ x = x₁ ∨ x = x₂) 
  (h_sum_squares : x₁^2 + x₂^2 = 3 / 16) :
  m = -1 / 8 :=
by
  sorry

end quadratic_solution_l108_108141


namespace factorize_expr_l108_108803

theorem factorize_expr (a b : ℝ) : 2 * a^2 - a * b = a * (2 * a - b) := 
by
  sorry

end factorize_expr_l108_108803


namespace sum_numbers_l108_108347

theorem sum_numbers :
  2345 + 3452 + 4523 + 5234 + 3245 + 2453 + 4532 + 5324 = 8888 := by
  sorry

end sum_numbers_l108_108347


namespace seeds_total_l108_108069

def seedsPerWatermelon : Nat := 345
def numberOfWatermelons : Nat := 27
def totalSeeds : Nat := seedsPerWatermelon * numberOfWatermelons

theorem seeds_total :
  totalSeeds = 9315 :=
by
  sorry

end seeds_total_l108_108069


namespace factor_diff_of_squares_l108_108929

theorem factor_diff_of_squares (t : ℝ) : 
  t^2 - 64 = (t - 8) * (t + 8) :=
by
  -- The purpose here is to state the theorem only, without the proof.
  sorry

end factor_diff_of_squares_l108_108929


namespace fraction_people_eating_pizza_l108_108844

variable (people : ℕ) (initial_pizza : ℕ) (pieces_per_person : ℕ) (remaining_pizza : ℕ)
variable (fraction : ℚ)

theorem fraction_people_eating_pizza (h1 : people = 15)
    (h2 : initial_pizza = 50)
    (h3 : pieces_per_person = 4)
    (h4 : remaining_pizza = 14)
    (h5 : 4 * 15 * fraction = initial_pizza - remaining_pizza) :
    fraction = 3 / 5 := 
  sorry

end fraction_people_eating_pizza_l108_108844


namespace sqrt_mul_sqrt_l108_108255

theorem sqrt_mul_sqrt (a b c : ℝ) (hac : a = 49) (hbc : b = sqrt 25) (hc : c = 7 * sqrt 5) : sqrt (a * b) = c := 
by
  sorry

end sqrt_mul_sqrt_l108_108255


namespace arithmetic_seq_sum_2017_l108_108370

theorem arithmetic_seq_sum_2017 
  (S : ℕ → ℝ) 
  (a : ℕ → ℝ) 
  (a1 : a 1 = -2017) 
  (h1 : ∀ n : ℕ, S n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1))
  (h2 : (S 2014) / 2014 - (S 2008) / 2008 = 6) : 
  S 2017 = -2017 :=
by
  sorry

end arithmetic_seq_sum_2017_l108_108370


namespace bases_representing_625_have_final_digit_one_l108_108811

theorem bases_representing_625_have_final_digit_one :
  (finset.count (λ b, 624 % b = 0) (finset.range (12 + 1)).filter (λ b, b ≥ 2)) = 7 :=
begin
  sorry
end

end bases_representing_625_have_final_digit_one_l108_108811


namespace radius_increase_of_pizza_l108_108029

/-- 
Prove that the percent increase in radius from a medium pizza to a large pizza is 20% 
given the following conditions:
1. The radius of the large pizza is some percent larger than that of a medium pizza.
2. The percent increase in area between a medium and a large pizza is approximately 44%.
3. The area of a circle is given by the formula A = π * r^2.
--/
theorem radius_increase_of_pizza
  (r R : ℝ) -- r and R are the radii of the medium and large pizza respectively
  (h1 : R = (1 + k) * r) -- The radius of the large pizza is some percent larger than that of a medium pizza
  (h2 : π * R^2 = 1.44 * π * r^2) -- The percent increase in area between a medium and a large pizza is approximately 44%
  : k = 0.2 := 
sorry

end radius_increase_of_pizza_l108_108029


namespace probability_of_selecting_one_defective_l108_108757

-- Definitions based on conditions from the problem
def items : List ℕ := [0, 1, 2, 3]  -- 0 represents defective, 1, 2, 3 represent genuine

def sample_space : List (ℕ × ℕ) := 
  [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

def event_A : List (ℕ × ℕ) := 
  [(0, 1), (0, 2), (0, 3)]

-- The probability of event A, calculated based on the classical method
def probability_event_A : ℚ := event_A.length / sample_space.length

theorem probability_of_selecting_one_defective : 
  probability_event_A = 1 / 2 := by
  sorry

end probability_of_selecting_one_defective_l108_108757


namespace grisha_wins_probability_expected_flips_l108_108041

-- Define conditions
def even_flip_heads_win (n : ℕ) (flip : ℕ → bool) : Prop :=
  n % 2 = 0 ∧ flip n = tt

def odd_flip_tails_win (n : ℕ) (flip : ℕ → bool) : Prop :=
  n % 2 = 1 ∧ flip n = ff

-- 1. Prove the probability P of Grisha winning the coin toss game is 1/3
theorem grisha_wins_probability (flip : ℕ → bool) (P : ℝ) :
  (∀ n, even_flip_heads_win n flip ∨ odd_flip_tails_win n flip) →
  P = 1/3 := sorry

-- 2. Prove the expected number E of coin flips until the outcome is decided is 2
theorem expected_flips (flip : ℕ → bool) (E : ℝ) :
  (∀ n, even_flip_heads_win n flip ∨ odd_flip_tails_win n flip) →
  E = 2 := sorry

end grisha_wins_probability_expected_flips_l108_108041


namespace ellipse_properties_l108_108013

noncomputable def ellipse_equation (x y a b : ℝ) : Prop :=
  (x * x) / (a * a) + (y * y) / (b * b) = 1

theorem ellipse_properties (a b c k : ℝ) (h_ab : a > b) (h_b : b > 1) (h_c : 2 * c = 2) 
  (h_area : (2 * Real.sqrt 3 / 3)^2 = 4 / 3) (h_slope : k ≠ 0)
  (h_PD : |(c - 4 * k^2 / (3 + 4 * k^2))^2 + (-3 * k / (3 + 4 * k^2))^2| = 3 * Real.sqrt 2 / 7) :
  (ellipse_equation 1 0 a b ∧
   (a = 2 ∧ b = Real.sqrt 3) ∧
   k = 1 ∨ k = -1) :=
by
  -- Prove the standard equation of the ellipse C and the value of k
  sorry

end ellipse_properties_l108_108013


namespace problem_conditions_l108_108365

theorem problem_conditions (a : ℕ → ℤ) :
  (1 + x)^6 = a 0 + a 1 * (1 - x) + a 2 * (1 - x)^2 + a 3 * (1 - x)^3 + a 4 * (1 - x)^4 + a 5 * (1 - x)^5 + a 6 * (1 - x)^6 →
  a 6 = 1 ∧ a 1 + a 3 + a 5 = -364 :=
by sorry

end problem_conditions_l108_108365


namespace complementary_angles_l108_108157

theorem complementary_angles (angle1 angle2 : ℝ) (h1 : angle1 + angle2 = 90) (h2 : angle1 = 25) : angle2 = 65 :=
by 
  sorry

end complementary_angles_l108_108157


namespace factor_difference_of_squares_l108_108933
noncomputable def factored (t : ℝ) : ℝ := (t - 8) * (t + 8)

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = factored t := by
  sorry

end factor_difference_of_squares_l108_108933


namespace measure_of_each_interior_angle_of_regular_octagon_l108_108461

theorem measure_of_each_interior_angle_of_regular_octagon 
  (n : ℕ) (h_n : n = 8) : 
  let sum_of_interior_angles := 180 * (n - 2) in
  let measure_of_interior_angle := sum_of_interior_angles / n in
  measure_of_interior_angle = 135 :=
by
  sorry

end measure_of_each_interior_angle_of_regular_octagon_l108_108461


namespace seashells_initial_count_l108_108706

theorem seashells_initial_count (S : ℝ) (h : S + 4.0 = 10) : S = 6.0 :=
by
  sorry

end seashells_initial_count_l108_108706


namespace parabola_whose_directrix_is_tangent_to_circle_l108_108491

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 9

noncomputable def is_tangent (line_eq : ℝ → ℝ → Prop) (circle_eq : ℝ → ℝ → Prop) : Prop := 
  ∃ p : ℝ × ℝ, (line_eq p.1 p.2) ∧ (circle_eq p.1 p.2) ∧ 
  (∀ q : ℝ × ℝ, (circle_eq q.1 q.2) → (line_eq q.1 q.2) → q = p)

-- Definitions of parabolas
noncomputable def parabola_A_directrix (x y : ℝ) : Prop := y = 2

noncomputable def parabola_B_directrix (x y : ℝ) : Prop := x = 2

noncomputable def parabola_C_directrix (x y : ℝ) : Prop := x = -4

noncomputable def parabola_D_directrix (x y : ℝ) : Prop := y = -1

-- The final statement to prove
theorem parabola_whose_directrix_is_tangent_to_circle :
  is_tangent parabola_D_directrix circle_eq ∧ ¬ is_tangent parabola_A_directrix circle_eq ∧ 
  ¬ is_tangent parabola_B_directrix circle_eq ∧ ¬ is_tangent parabola_C_directrix circle_eq :=
sorry

end parabola_whose_directrix_is_tangent_to_circle_l108_108491


namespace find_a_and_b_l108_108853

noncomputable def a_and_b (x y : ℝ) (a b : ℝ) : Prop :=
  a = Real.sqrt x + Real.sqrt y ∧ b = Real.sqrt (x + 2) + Real.sqrt (y + 2) ∧
  ∃ n : ℤ, a = n ∧ b = n + 2

theorem find_a_and_b (x y : ℝ) (a b : ℝ)
  (h₁ : 0 ≤ x)
  (h₂ : 0 ≤ y)
  (h₃ : a_and_b x y a b)
  (h₄ : ∃ n : ℤ, a = n ∧ b = n + 2) :
  a = 1 ∧ b = 3 := by
  sorry

end find_a_and_b_l108_108853


namespace battery_life_remaining_l108_108792

variables (full_battery_life : ℕ) (used_fraction : ℚ) (exam_duration : ℕ) (remaining_battery : ℕ)

def brody_calculator_conditions :=
  full_battery_life = 60 ∧
  used_fraction = 3 / 4 ∧
  exam_duration = 2

theorem battery_life_remaining
  (h : brody_calculator_conditions full_battery_life used_fraction exam_duration) :
  remaining_battery = 13 :=
by 
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end battery_life_remaining_l108_108792


namespace number_of_frames_bought_l108_108074

/- 
   Define the problem conditions:
   1. Each photograph frame costs 3 dollars.
   2. Sally paid with a 20 dollar bill.
   3. Sally got 11 dollars in change.
-/ 

def frame_cost : Int := 3
def initial_payment : Int := 20
def change_received : Int := 11

/- 
   Prove that the number of photograph frames Sally bought is 3.
-/

theorem number_of_frames_bought : (initial_payment - change_received) / frame_cost = 3 := 
by
  sorry

end number_of_frames_bought_l108_108074


namespace rectangle_within_l108_108065

theorem rectangle_within (a b c d : ℝ) (h1 : a < c) (h2 : c ≤ d) (h3 : d < b) (h4 : a * b < c * d) :
  (b^2 - a^2)^2 ≤ (b * d - a * c)^2 + (b * c - a * d)^2 :=
by
  sorry

end rectangle_within_l108_108065


namespace no_intersection_curves_l108_108147

theorem no_intersection_curves (k : ℕ) (hn : k > 0) 
  (h_intersection : ∀ x y : ℝ, ¬(x^2 + y^2 = k^2 ∧ x * y = k)) : 
  k = 1 := 
sorry

end no_intersection_curves_l108_108147


namespace expected_value_is_correct_l108_108116

-- Given conditions
def prob_heads : ℚ := 2 / 5
def prob_tails : ℚ := 3 / 5
def win_amount_heads : ℚ := 5
def loss_amount_tails : ℚ := -3

-- Expected value calculation
def expected_value : ℚ := prob_heads * win_amount_heads + prob_tails * loss_amount_tails

-- Property to prove
theorem expected_value_is_correct : expected_value = 0.2 := sorry

end expected_value_is_correct_l108_108116


namespace sqrt_49_times_sqrt25_eq_7_sqrt5_l108_108296

-- Definitions based on conditions
def sqrt_25 := Real.sqrt 25
def sqrt_49 := Real.sqrt 49

theorem sqrt_49_times_sqrt25_eq_7_sqrt5 :
  Real.sqrt (49 * sqrt_25) = 7 * Real.sqrt 5 := by
sorry

end sqrt_49_times_sqrt25_eq_7_sqrt5_l108_108296


namespace hyperbola_eccentricity_squared_l108_108820

theorem hyperbola_eccentricity_squared (a b : ℝ) (hapos : a > 0) (hbpos : b > 0) 
(F1 F2 : ℝ × ℝ) (e : ℝ)
(hfoci1 : F1 = (-a * e, 0))
(hfoci2 : F2 = (a * e, 0)) :
  (-2*F1.1).pow 2 + (F1.1 + a*e - (F1.1)).pow 2 + (a/b).pow 2 = (4 + 2*real.sqrt 2) :=
sorry

end hyperbola_eccentricity_squared_l108_108820


namespace prob_greater_first_card_l108_108210

theorem prob_greater_first_card :
  (∃ (draw : fin 5 → fin 5 → bool), 
    let count := (Sum (λ (i : fin 5), Sum (λ (j : fin 5), if j.val < i.val then 1 else 0)))
    count = 10 
  ) → p = (2 : ℚ) / 5 :=
by
  sorry

end prob_greater_first_card_l108_108210


namespace problem_statement_l108_108531

theorem problem_statement (x : ℤ) (h : 3 - x = -2) : x + 1 = 6 := 
by {
  -- Proof would be provided here
  sorry
}

end problem_statement_l108_108531


namespace probability_quadratic_real_roots_l108_108806

noncomputable def probability_real_roots : ℝ := 3 / 4

theorem probability_quadratic_real_roots :
  (∀ a b : ℝ, -π ≤ a ∧ a ≤ π ∧ -π ≤ b ∧ b ≤ π →
  (∃ x : ℝ, x^2 + 2*a*x - b^2 + π = 0) ↔ a^2 + b^2 ≥ π) →
  (probability_real_roots = 3 / 4) :=
sorry

end probability_quadratic_real_roots_l108_108806


namespace required_run_rate_l108_108191

/-
In the first 10 overs of a cricket game, the run rate was 3.5. 
What should be the run rate in the remaining 40 overs to reach the target of 320 runs?
-/

def run_rate_in_10_overs : ℝ := 3.5
def overs_played : ℕ := 10
def target_runs : ℕ := 320 
def remaining_overs : ℕ := 40

theorem required_run_rate : 
  (target_runs - (run_rate_in_10_overs * overs_played)) / remaining_overs = 7.125 := by 
sorry

end required_run_rate_l108_108191


namespace ray_walks_to_park_l108_108574

theorem ray_walks_to_park (x : ℤ) (h1 : 3 * (x + 7 + 11) = 66) : x = 4 :=
by
  -- solving steps are skipped
  sorry

end ray_walks_to_park_l108_108574


namespace max_area_parabola_l108_108371

open Real

noncomputable def max_area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1))

theorem max_area_parabola (a b c : ℝ) 
  (ha : a^2 = (a * a))
  (hb : b^2 = (b * b))
  (hc : c^2 = (c * c))
  (centroid_cond1 : (a + b + c) = 4)
  (centroid_cond2 : (a^2 + b^2 + c^2) = 6)
  : max_area_of_triangle (a^2, a) (b^2, b) (c^2, c) = (sqrt 3) / 9 := 
sorry

end max_area_parabola_l108_108371


namespace wine_ages_l108_108865

-- Define the ages of the wines as variables
variable (C F T B Bo M : ℝ)

-- Define the six conditions
axiom h1 : F = 3 * C
axiom h2 : C = 4 * T
axiom h3 : B = (1 / 2) * T
axiom h4 : Bo = 2 * F
axiom h5 : M^2 = Bo
axiom h6 : C = 40

-- Prove the ages of the wines 
theorem wine_ages : 
  F = 120 ∧ 
  T = 10 ∧ 
  B = 5 ∧ 
  Bo = 240 ∧ 
  M = Real.sqrt 240 :=
by
  sorry

end wine_ages_l108_108865


namespace stratified_sampling_l108_108398

-- Definition of the given variables and conditions
def total_students_grade10 : ℕ := 30
def total_students_grade11 : ℕ := 40
def selected_students_grade11 : ℕ := 8

-- Implementation of the stratified sampling proportion requirement
theorem stratified_sampling (x : ℕ) (hx : (x : ℚ) / total_students_grade10 = (selected_students_grade11 : ℚ) / total_students_grade11) :
  x = 6 :=
by
  sorry

end stratified_sampling_l108_108398


namespace jackson_holidays_l108_108402

theorem jackson_holidays (holidays_per_month : ℕ) (months_per_year : ℕ) (total_holidays : ℕ) :
  holidays_per_month = 3 → months_per_year = 12 → total_holidays = holidays_per_month * months_per_year →
  total_holidays = 36 :=
by
  intros
  sorry

end jackson_holidays_l108_108402


namespace smallest_sum_of_relatively_prime_numbers_l108_108196

open Nat

/-- Leonhard has five cards. Each card has a nonnegative integer written on it,
and any two cards show relatively prime numbers. Compute the smallest possible value of the sum of
the numbers on Leonhard's cards. -/

theorem smallest_sum_of_relatively_prime_numbers :
  ∃ (a b c d e : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
  (gcd a b = 1) ∧ (gcd a c = 1) ∧ (gcd a d = 1) ∧ (gcd a e = 1) ∧ 
  (gcd b c = 1) ∧ (gcd b d = 1) ∧ (gcd b e = 1) ∧ 
  (gcd c d = 1) ∧ (gcd c e = 1) ∧ (gcd d e = 1) ∧ 
  a + b + c + d + e = 4 :=
begin
  sorry
end

end smallest_sum_of_relatively_prime_numbers_l108_108196


namespace ellipse_standard_eq_l108_108658

-- Assumptions and definitions
variables {a b c x y : ℝ}
variables {F1 F2 P M : ℝ × ℝ}
variables λ : ℝ
variables t : ℝ
variables {A B C D : ℝ × ℝ}

-- Conditions
def ellipse_eq := x^2 / a^2 + y^2 / b^2 = 1
def point_on_ellipse := P = (-1, (sqrt 2) / 2) ∧ ellipse_eq P.1 P.2
def midpoint_condition := M = ((P.1 + F2.1) / 2, (P.2 + F2.2) / 2) ∧ (M = (0, M.2))
def dot_product := λ ∈ [2/3, 1]

-- Proving the range of area S given conditions
theorem ellipse_standard_eq (h1: ellipse_eq) (h2: point_on_ellipse) (h3: midpoint_condition) (h4: dot_product) :
  (∃ a² = 2, b² = 1) ∧ (∃ S ∈ [4 * sqrt 3 / 5, 4 * sqrt 6 / 7]) :=
by sorry

end ellipse_standard_eq_l108_108658


namespace log_relationship_l108_108388

theorem log_relationship (a b c : ℝ) 
  (ha : a = Real.log 3 / Real.log 2) 
  (hb : b = Real.log 4 / Real.log 3) 
  (hc : c = Real.log 5 / Real.log 4) : 
  c < b ∧ b < a :=
by 
  sorry

end log_relationship_l108_108388


namespace second_largest_geometric_sum_l108_108586

theorem second_largest_geometric_sum {a r : ℕ} (h_sum: a + a * r + a * r^2 + a * r^3 = 1417) (h_geometric: 1 + r + r^2 + r^3 ∣ 1417) : (a * r^2 = 272) :=
sorry

end second_largest_geometric_sum_l108_108586


namespace sum_of_parts_l108_108602

theorem sum_of_parts (x y : ℝ) (h1 : x + y = 52) (h2 : y = 30.333333333333332) :
  10 * x + 22 * y = 884 :=
sorry

end sum_of_parts_l108_108602


namespace sqrt_expression_simplified_l108_108271

theorem sqrt_expression_simplified :
  sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  sorry

end sqrt_expression_simplified_l108_108271


namespace tram_speed_l108_108713

/-- 
Given:
1. The pedestrian's speed is 1 km per 10 minutes, which converts to 6 km/h.
2. The speed of the trams is V km/h.
3. The relative speed of oncoming trams is V + 6 km/h.
4. The relative speed of overtaking trams is V - 6 km/h.
5. The ratio of the number of oncoming trams to overtaking trams is 700/300.
Prove:
The speed of the trams V is 15 km/h.
-/
theorem tram_speed (V : ℝ) (h1 : (V + 6) / (V - 6) = 700 / 300) : V = 15 :=
by
  sorry

end tram_speed_l108_108713


namespace john_total_spent_l108_108410

noncomputable def calculate_total_spent : ℝ :=
  let orig_price_A := 900.0
  let discount_A := 0.15 * orig_price_A
  let price_A := orig_price_A - discount_A
  let tax_A := 0.06 * price_A
  let total_A := price_A + tax_A
  let orig_price_B := 600.0
  let discount_B := 0.25 * orig_price_B
  let price_B := orig_price_B - discount_B
  let tax_B := 0.09 * price_B
  let total_B := price_B + tax_B
  let total_other_toys := total_A + total_B
  let price_lightsaber := 2 * total_other_toys
  let tax_lightsaber := 0.04 * price_lightsaber
  let total_lightsaber := price_lightsaber + tax_lightsaber
  total_other_toys + total_lightsaber

theorem john_total_spent : calculate_total_spent = 4008.312 := by
  sorry

end john_total_spent_l108_108410


namespace factor_t_sq_minus_64_l108_108939

def isDifferenceOfSquares (a b : Int) : Prop := a = b^2

theorem factor_t_sq_minus_64 (t : Int) (h : isDifferenceOfSquares 64 8) : (t^2 - 64) = (t - 8) * (t + 8) := by
  sorry

end factor_t_sq_minus_64_l108_108939


namespace probability_qualifies_team_expected_value_number_competitions_l108_108995

open ProbabilityTheory

noncomputable def probability_qualifies : ℚ := 67 / 256
noncomputable def expected_value_competitions : ℚ := 65 / 16

theorem probability_qualifies_team {p q : ℚ} (h1 : p = 1 / 4) (h2 : q = 3 / 4) :
  (let event_a := p * p * (q * q * q  + q ^ 4) in
    event_a) = probability_qualifies := 
by 
  sorry

theorem expected_value_number_competitions {p q : ℚ} (h1 : p = 1 / 4) (h2 : q = 3 / 4) :
  (let ξ := 2 * (p * p) + 3 * (2 * p * q * p) + 4 * (3 * p * q * q * p + q ^ 4) + 5 * (4 * p * q ^ 3) in
    ξ / probability_qualifies) = expected_value_competitions := 
by 
  sorry

end probability_qualifies_team_expected_value_number_competitions_l108_108995


namespace postcards_remainder_l108_108915

theorem postcards_remainder :
  let amelia := 45
  let ben := 55
  let charles := 23
  let total := amelia + ben + charles
  total % 15 = 3 :=
by
  let amelia := 45
  let ben := 55
  let charles := 23
  let total := amelia + ben + charles
  show total % 15 = 3
  sorry

end postcards_remainder_l108_108915


namespace units_digit_of_product_of_odds_between_10_and_50_l108_108918

def product_of_odds_units_digit : ℕ :=
  let odds := [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49]
  let product := odds.foldl (· * ·) 1
  product % 10

theorem units_digit_of_product_of_odds_between_10_and_50 : product_of_odds_units_digit = 5 :=
  sorry

end units_digit_of_product_of_odds_between_10_and_50_l108_108918


namespace points_per_enemy_l108_108591

theorem points_per_enemy (kills: ℕ) (bonus_threshold: ℕ) (bonus_multiplier: ℝ) (total_score_with_bonus: ℕ) (P: ℝ) 
(hk: kills = 150) (hbt: bonus_threshold = 100) (hbm: bonus_multiplier = 1.5) (hts: total_score_with_bonus = 2250)
(hP: 150 * P * bonus_multiplier = total_score_with_bonus) : 
P = 10 := sorry

end points_per_enemy_l108_108591


namespace michael_initial_money_l108_108556

theorem michael_initial_money (M : ℝ) 
  (half_give_away_to_brother : ∃ (m_half : ℝ), M / 2 = m_half)
  (brother_initial_money : ℝ := 17)
  (candy_cost : ℝ := 3)
  (brother_ends_up_with : ℝ := 35) :
  brother_initial_money + M / 2 - candy_cost = brother_ends_up_with ↔ M = 42 :=
sorry

end michael_initial_money_l108_108556


namespace james_payment_l108_108403

theorem james_payment (james_meal : ℕ) (friend_meal : ℕ) (tip_percent : ℕ) (final_payment : ℕ) : 
  james_meal = 16 → 
  friend_meal = 14 → 
  tip_percent = 20 → 
  final_payment = 18 :=
by
  -- Definitions
  let total_bill_before_tip := james_meal + friend_meal
  let tip := total_bill_before_tip * tip_percent / 100
  let final_bill := total_bill_before_tip + tip
  let half_bill := final_bill / 2
  -- Proof (to be filled in)
  sorry

end james_payment_l108_108403


namespace ivanov_entitled_to_12_million_rubles_l108_108765

def equal_contributions (x : ℝ) : Prop :=
  let ivanov_contribution := 70 * x
  let petrov_contribution := 40 * x
  let sidorov_contribution := 44
  ivanov_contribution = 44 ∧ petrov_contribution = 44 ∧ (ivanov_contribution + petrov_contribution + sidorov_contribution) / 3 = 44

def money_ivanov_receives (x : ℝ) : ℝ :=
  let ivanov_contribution := 70 * x
  ivanov_contribution - 44

theorem ivanov_entitled_to_12_million_rubles :
  ∃ x : ℝ, equal_contributions x → money_ivanov_receives x = 12 :=
sorry

end ivanov_entitled_to_12_million_rubles_l108_108765


namespace find_distance_between_PQ_l108_108094

-- Defining distances and speeds
def distance_by_first_train (t : ℝ) : ℝ := 50 * t
def distance_by_second_train (t : ℝ) : ℝ := 40 * t
def distance_between_PQ (t : ℝ) : ℝ := distance_by_first_train t + (distance_by_first_train t - 100)

-- Main theorem stating the problem
theorem find_distance_between_PQ : ∃ t : ℝ, distance_by_first_train t - distance_by_second_train t = 100 ∧ distance_between_PQ t = 900 := 
sorry

end find_distance_between_PQ_l108_108094


namespace distance_from_origin_to_point_l108_108185

open Real

theorem distance_from_origin_to_point (x y : ℝ) :
  x = 8 →
  y = -15 →
  sqrt (x^2 + y^2) = 17 :=
by
  intros hx hy
  rw [hx, hy]
  norm_num
  rw ←sqrt_sq (show 0 ≤ 289 by norm_num)
  norm_num

end distance_from_origin_to_point_l108_108185


namespace max_imaginary_part_of_root_l108_108490

theorem max_imaginary_part_of_root (z : ℂ) (h : z^6 - z^4 + z^2 - 1 = 0) (hne : z^2 ≠ 1) : 
  ∃ θ : ℝ, -90 ≤ θ ∧ θ ≤ 90 ∧ Complex.im z = Real.sin θ ∧ θ = 90 := 
sorry

end max_imaginary_part_of_root_l108_108490


namespace find_m_plus_n_l108_108523

-- Define the sets and variables
def M : Set ℝ := {x | x^2 - 4 * x < 0}
def N (m : ℝ) : Set ℝ := {x | m < x ∧ x < 5}
def K (n : ℝ) : Set ℝ := {x | 3 < x ∧ x < n}

theorem find_m_plus_n (m n : ℝ) 
  (hM: M = {x | 0 < x ∧ x < 4})
  (hK_true: K n = M ∩ N m) :
  m + n = 7 := 
  sorry

end find_m_plus_n_l108_108523


namespace range_of_a_l108_108435

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ax^2 + 3 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 4 / 9) :=
sorry

end range_of_a_l108_108435


namespace sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l108_108284

theorem sqrt_49_mul_sqrt_25_eq_7_sqrt_5 : (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l108_108284


namespace divisor_of_number_l108_108911

theorem divisor_of_number (n d q p : ℤ) 
  (h₁ : n = d * q + 3)
  (h₂ : n ^ 2 = d * p + 3) : 
  d = 6 := 
sorry

end divisor_of_number_l108_108911


namespace simplify_expression_l108_108712

theorem simplify_expression (x : ℝ) : 
  x * (x * (x * (3 - x) - 6) + 7) + 2 = -x^4 + 3 * x^3 - 6 * x^2 + 7 * x + 2 := 
by 
  sorry

end simplify_expression_l108_108712


namespace hypotenuse_of_45_45_90_triangle_l108_108570

theorem hypotenuse_of_45_45_90_triangle (leg : ℝ) (angle_opposite_leg : ℝ) (h_leg : leg = 15) (h_angle : angle_opposite_leg = 45) :
  ∃ hypotenuse, hypotenuse = leg * Real.sqrt 2 :=
by
  use leg * Real.sqrt 2
  rw [h_leg]
  rw [h_angle]
  sorry

end hypotenuse_of_45_45_90_triangle_l108_108570


namespace num_students_l108_108489

theorem num_students (x : ℕ) (h1 : ∃ z : ℕ, z = 10 * x + 6) (h2 : ∃ z : ℕ, z = 12 * x - 6) : x = 6 :=
by
  sorry

end num_students_l108_108489


namespace money_brought_to_store_l108_108982

theorem money_brought_to_store : 
  let sheet_cost := 42
  let rope_cost := 18
  let propane_and_burner_cost := 14
  let helium_cost_per_ounce := 1.5
  let height_per_ounce := 113
  let max_height := 9492
  let total_item_cost := sheet_cost + rope_cost + propane_and_burner_cost
  let helium_needed := max_height / height_per_ounce
  let helium_total_cost := helium_needed * helium_cost_per_ounce
  total_item_cost + helium_total_cost = 200 :=
by
  sorry

end money_brought_to_store_l108_108982


namespace cubic_roots_equal_l108_108036

theorem cubic_roots_equal (k : ℚ) (h1 : k > 0)
  (h2 : ∃ a b : ℚ, a ≠ b ∧ (a + a + b = -3) ∧ (2 * a * b + a^2 = -54) ∧ (3 * x^3 + 9 * x^2 - 162 * x + k = 0)) : 
  k = 7983 / 125 :=
sorry

end cubic_roots_equal_l108_108036


namespace andrew_age_l108_108119

variables (a g : ℝ)

theorem andrew_age (h1 : g = 15 * a) (h2 : g - a = 60) : a = 30 / 7 :=
by sorry

end andrew_age_l108_108119


namespace distance_to_point_is_17_l108_108188

def distance_from_origin (x y : ℝ) : ℝ :=
  Real.sqrt ((x - 0)^2 + (y - 0)^2)

theorem distance_to_point_is_17 :
  distance_from_origin 8 (-15) = 17 :=
by
  sorry

end distance_to_point_is_17_l108_108188


namespace scientific_notation_104000000_l108_108707

theorem scientific_notation_104000000 :
  104000000 = 1.04 * 10^8 :=
sorry

end scientific_notation_104000000_l108_108707


namespace sqrt_49_times_sqrt_25_eq_7sqrt5_l108_108262

theorem sqrt_49_times_sqrt_25_eq_7sqrt5 :
  (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  have h1 : Real.sqrt 25 = 5 := by sorry
  have h2 : 49 * 5 = 245 := by sorry
  have h3 : Real.sqrt 245 = 7 * Real.sqrt 5 := by sorry
  sorry

end sqrt_49_times_sqrt_25_eq_7sqrt5_l108_108262


namespace func_C_increasing_l108_108117

open Set

noncomputable def func_A (x : ℝ) : ℝ := 3 - x
noncomputable def func_B (x : ℝ) : ℝ := x^2 - x
noncomputable def func_C (x : ℝ) : ℝ := -1 / (x + 1)
noncomputable def func_D (x : ℝ) : ℝ := -abs x

theorem func_C_increasing : ∀ x y : ℝ, 0 < x → 0 < y → x < y → func_C x < func_C y := by
  sorry

end func_C_increasing_l108_108117


namespace part_one_solution_set_part_two_range_of_a_l108_108377

def f (x : ℝ) (a : ℝ) : ℝ := |x - a| - 2

theorem part_one_solution_set (a : ℝ) (h : a = 1) : { x : ℝ | f x a + |2 * x - 3| > 0 } = { x : ℝ | x > 2 ∨ x < 2 / 3 } := 
sorry

theorem part_two_range_of_a : (∃ x : ℝ, f x (a) > |x - 3|) ↔ (a < 1 ∨ a > 5) :=
sorry

end part_one_solution_set_part_two_range_of_a_l108_108377


namespace annie_total_distance_traveled_l108_108343

-- Definitions of conditions
def walk_distance : ℕ := 5
def bus_distance : ℕ := 7
def total_distance_one_way : ℕ := walk_distance + bus_distance
def total_distance_round_trip : ℕ := total_distance_one_way * 2

-- Theorem statement to prove the total number of blocks traveled
theorem annie_total_distance_traveled : total_distance_round_trip = 24 :=
by
  sorry

end annie_total_distance_traveled_l108_108343


namespace correct_divisor_l108_108843

theorem correct_divisor :
  ∀ (D : ℕ), (D = 12 * 63) → (D = x * 36) → (x = 21) := 
by 
  intros D h1 h2
  sorry

end correct_divisor_l108_108843


namespace min_value_fraction_l108_108973

theorem min_value_fraction {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 1) :
  (∀ y : ℝ,  y > 0 → (∀ x : ℝ, x > 0 → x + 3 * y = 1 → (1/x + 1/(3*y)) ≥ 4)) :=
sorry

end min_value_fraction_l108_108973


namespace max_coins_identifiable_l108_108105

theorem max_coins_identifiable (n : ℕ) : exists (c : ℕ), c = 2 * n^2 + 1 :=
by
  sorry

end max_coins_identifiable_l108_108105


namespace ping_pong_ball_probability_l108_108326

noncomputable def multiple_of_6_9_or_both_probability : ℚ :=
  let total_numbers := 72
  let multiples_of_6 := 12
  let multiples_of_9 := 8
  let multiples_of_both := 4
  (multiples_of_6 + multiples_of_9 - multiples_of_both) / total_numbers

theorem ping_pong_ball_probability :
  multiple_of_6_9_or_both_probability = 2 / 9 :=
by
  sorry

end ping_pong_ball_probability_l108_108326


namespace factor_difference_of_squares_l108_108932
noncomputable def factored (t : ℝ) : ℝ := (t - 8) * (t + 8)

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = factored t := by
  sorry

end factor_difference_of_squares_l108_108932


namespace cost_of_1000_gums_in_dollars_l108_108221

theorem cost_of_1000_gums_in_dollars :
  let cost_per_piece_in_cents := 1
  let pieces := 1000
  let cents_per_dollar := 100
  ∃ cost_in_dollars : ℝ, cost_in_dollars = (cost_per_piece_in_cents * pieces) / cents_per_dollar :=
sorry

end cost_of_1000_gums_in_dollars_l108_108221


namespace total_students_in_school_district_l108_108088

def CampusA_students : Nat :=
  let students_per_grade : Nat := 100
  let num_grades : Nat := 5
  let special_education : Nat := 30
  (students_per_grade * num_grades) + special_education

def CampusB_students : Nat :=
  let students_per_grade : Nat := 120
  let num_grades : Nat := 5
  students_per_grade * num_grades

def CampusC_students : Nat :=
  let students_per_grade : Nat := 150
  let num_grades : Nat := 2
  let international_program : Nat := 50
  (students_per_grade * num_grades) + international_program

def total_students : Nat :=
  CampusA_students + CampusB_students + CampusC_students

theorem total_students_in_school_district : total_students = 1480 := by
  sorry

end total_students_in_school_district_l108_108088


namespace remainder_x2023_plus_1_l108_108950

noncomputable def remainder (a b : Polynomial ℂ) : Polynomial ℂ :=
a % b

theorem remainder_x2023_plus_1 :
  remainder (Polynomial.X ^ 2023 + 1) (Polynomial.X ^ 8 - Polynomial.X ^ 6 + Polynomial.X ^ 4 - Polynomial.X ^ 2 + 1) =
  - Polynomial.X ^ 3 + 1 :=
by
  sorry

end remainder_x2023_plus_1_l108_108950


namespace complete_square_solution_l108_108622

theorem complete_square_solution (a b c : ℤ) (h1 : a^2 = 25) (h2 : 10 * b = 30) (h3 : (a * x + b)^2 = 25 * x^2 + 30 * x + c) :
  a + b + c = -58 :=
by
  sorry

end complete_square_solution_l108_108622


namespace pizzas_in_park_l108_108409

-- Define the conditions and the proof problem
def pizza_cost : ℕ := 12
def delivery_charge : ℕ := 2
def park_distance : ℕ := 100  -- in meters
def building_distance : ℕ := 2000  -- in meters
def pizzas_delivered_to_building : ℕ := 2
def total_payment_received : ℕ := 64

-- Prove the number of pizzas delivered in the park
theorem pizzas_in_park : (64 - (pizzas_delivered_to_building * pizza_cost + delivery_charge)) / pizza_cost = 3 :=
by
  sorry -- Proof not required

end pizzas_in_park_l108_108409


namespace translation_4_units_upwards_l108_108208

theorem translation_4_units_upwards (M N : ℝ × ℝ) (hx : M.1 = N.1) (hy_diff : N.2 - M.2 = 4) :
  N = (M.1, M.2 + 4) :=
by
  sorry

end translation_4_units_upwards_l108_108208


namespace triangle_subsegment_length_l108_108733

theorem triangle_subsegment_length (DF DE EF DG GF : ℚ)
  (h_ratio : ∃ x : ℚ, DF = 3 * x ∧ DE = 4 * x ∧ EF = 5 * x)
  (h_EF_len : EF = 20)
  (h_angle_bisector : DG + GF = DE ∧ DG / GF = DE / DF) :
  DF < DE ∧ DE < EF →
  min DG GF = 48 / 7 :=
by
  sorry

end triangle_subsegment_length_l108_108733


namespace invested_sum_l108_108900

theorem invested_sum (P r : ℝ) 
  (peter_total : P + 3 * P * r = 815) 
  (david_total : P + 4 * P * r = 870) 
  : P = 650 := 
by
  sorry

end invested_sum_l108_108900


namespace gasoline_tank_capacity_l108_108321

theorem gasoline_tank_capacity (x : ℕ) (h1 : 5 * x / 6 - 2 * x / 3 = 15) : x = 90 :=
sorry

end gasoline_tank_capacity_l108_108321


namespace liking_songs_proof_l108_108211

def num_ways_liking_songs : ℕ :=
  let total_songs := 6
  let pair1 := 1
  let pair2 := 2
  let ways_to_choose_pair1 := Nat.choose total_songs pair1
  let remaining_songs := total_songs - pair1
  let ways_to_choose_pair2 := Nat.choose remaining_songs pair2 * Nat.choose (remaining_songs - pair2) pair2
  let final_song_choices := 4
  ways_to_choose_pair1 * ways_to_choose_pair2 * final_song_choices * 3 -- multiplied by 3 for the three possible pairs

theorem liking_songs_proof :
  num_ways_liking_songs = 2160 :=
  by sorry

end liking_songs_proof_l108_108211


namespace set_equivalence_l108_108063

open Set

def set_A : Set ℝ := { x | x^2 - 2 * x > 0 }
def set_B : Set ℝ := { y | ∃ x : ℝ, y = 2^x }

theorem set_equivalence : (univ \ set_B) ∪ set_A = (Iic 1) ∪ Ioi 2 :=
sorry

end set_equivalence_l108_108063


namespace density_of_second_part_l108_108730

theorem density_of_second_part (V m : ℝ) (h1 : ∀ V m : ℝ, V_1 = 0.3 * V) 
  (h2 : ∀ V m : ℝ, m_1 = 0.6 * m) 
  (rho1 : ρ₁ = 7800) : 
  ∃ ρ₂, ρ₂ = 2229 :=
by sorry

end density_of_second_part_l108_108730


namespace problem_result_l108_108603

def elongation_A : List ℕ := [545, 533, 551, 522, 575, 544, 541, 568, 596, 548]
def elongation_B : List ℕ := [536, 527, 543, 530, 560, 533, 522, 550, 576, 536]

def z_i : List ℤ := List.zipWith (λ x y => x - y) elongation_A elongation_B

def sample_mean (lst : List ℤ) : ℚ :=
  (List.sum lst : ℚ) / List.length lst

def sample_variance (lst : List ℤ) : ℚ :=
  let mean := sample_mean lst
  (List.sum (lst.map (λ z => (z - mean) * (z - mean))) : ℚ) / List.length lst

def improvement_significance (mean : ℚ) (variance : ℚ) : Prop :=
  mean ≥ 2 * Real.sqrt (variance / 10)

theorem problem_result :
  sample_mean z_i = 11 ∧
  sample_variance z_i = 61 ∧
  improvement_significance (sample_mean z_i) (sample_variance z_i) :=
by
  sorry

end problem_result_l108_108603


namespace prove_ordered_pair_l108_108643

-- Definition of the problem
def satisfies_equation1 (x y : ℚ) : Prop :=
  3 * x - 4 * y = -7

def satisfies_equation2 (x y : ℚ) : Prop :=
  7 * x - 3 * y = 5

-- Definition of the correct answer
def correct_answer (x y : ℚ) : Prop :=
  x = -133 / 57 ∧ y = 64 / 19

-- Main theorem to prove
theorem prove_ordered_pair :
  correct_answer (-133 / 57) (64 / 19) :=
by
  unfold correct_answer
  constructor
  { sorry }
  { sorry }

end prove_ordered_pair_l108_108643


namespace right_triangle_legs_solutions_l108_108949

theorem right_triangle_legs_solutions (R r : ℝ) (h_cond : R / r ≥ 1 + Real.sqrt 2) :
  ∃ (a b : ℝ), 
    a = r + R + Real.sqrt (R^2 - 2 * r * R - r^2) ∧ 
    b = r + R - Real.sqrt (R^2 - 2 * r * R - r^2) ∧ 
    (2 * R)^2 = a^2 + b^2 := by
  sorry

end right_triangle_legs_solutions_l108_108949


namespace sqrt_nested_l108_108302

theorem sqrt_nested : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_nested_l108_108302


namespace ratio_of_water_level_increase_l108_108592

noncomputable def volume_narrow_cone (h₁ : ℝ) : ℝ := (16 / 3) * Real.pi * h₁
noncomputable def volume_wide_cone (h₂ : ℝ) : ℝ := (64 / 3) * Real.pi * h₂
noncomputable def volume_marble_narrow : ℝ := (32 / 3) * Real.pi
noncomputable def volume_marble_wide : ℝ := (4 / 3) * Real.pi

theorem ratio_of_water_level_increase :
  ∀ (h₁ h₂ h₁' h₂' : ℝ),
  h₁ = 4 * h₂ →
  h₁' = h₁ + 2 →
  h₂' = h₂ + (1 / 16) →
  volume_narrow_cone h₁ = volume_wide_cone h₂ →
  volume_narrow_cone h₁ + volume_marble_narrow = volume_narrow_cone h₁' →
  volume_wide_cone h₂ + volume_marble_wide = volume_wide_cone h₂' →
  (h₁' - h₁) / (h₂' - h₂) = 32 :=
by
  intros h₁ h₂ h₁' h₂' h₁_eq_4h₂ h₁'_eq_h₁_add_2 h₂'_eq_h₂_add_1_div_16 vol_h₁_eq_vol_h₂ vol_nar_eq vol_wid_eq
  sorry

end ratio_of_water_level_increase_l108_108592


namespace sqrt_49_mul_sqrt_25_l108_108282

theorem sqrt_49_mul_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_l108_108282


namespace boat_distance_against_stream_l108_108043

variable (v_s : ℝ)
variable (effective_speed_stream : ℝ := 15)
variable (speed_still_water : ℝ := 10)
variable (distance_along_stream : ℝ := 15)

theorem boat_distance_against_stream : 
  distance_along_stream / effective_speed_stream = 1 ∧ effective_speed_stream = speed_still_water + v_s →
  10 - v_s = 5 :=
by
  intros
  sorry

end boat_distance_against_stream_l108_108043


namespace sqrt_mul_sqrt_l108_108250

theorem sqrt_mul_sqrt (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : sqrt (a * sqrt b) = sqrt a * sqrt (sqrt b) := by
  sorry

example : sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  have h1 : sqrt 25 = 5 := by 
    norm_num
  have h2 : sqrt (49 * 5) = sqrt 245 := by  
    unfold sqrt
  have h3 : sqrt 245 = 7 * sqrt 5 := by 
    norm_num
  rw [h1, h2, h3]
  norm_num

end sqrt_mul_sqrt_l108_108250


namespace jogging_distance_apart_l108_108914

theorem jogging_distance_apart
  (alice_speed : ℝ)
  (bob_speed : ℝ)
  (time_in_minutes : ℝ)
  (distance_apart : ℝ)
  (h1 : alice_speed = 1 / 12)
  (h2 : bob_speed = 3 / 40)
  (h3 : time_in_minutes = 120)
  (h4 : distance_apart = alice_speed * time_in_minutes + bob_speed * time_in_minutes) :
  distance_apart = 19 := by
  sorry

end jogging_distance_apart_l108_108914


namespace quadratic_roots_l108_108140

theorem quadratic_roots (m x1 x2 : ℝ) 
  (h1 : 2*x1^2 + 4*m*x1 + m = 0)
  (h2 : 2*x2^2 + 4*m*x2 + m = 0)
  (h3 : x1 ≠ x2)
  (h4 : x1^2 + x2^2 = 3/16) : 
  m = -1/8 := 
sorry

end quadratic_roots_l108_108140


namespace sqrt_mul_sqrt_l108_108252

theorem sqrt_mul_sqrt (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : sqrt (a * sqrt b) = sqrt a * sqrt (sqrt b) := by
  sorry

example : sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  have h1 : sqrt 25 = 5 := by 
    norm_num
  have h2 : sqrt (49 * 5) = sqrt 245 := by  
    unfold sqrt
  have h3 : sqrt 245 = 7 * sqrt 5 := by 
    norm_num
  rw [h1, h2, h3]
  norm_num

end sqrt_mul_sqrt_l108_108252


namespace problem_equiv_math_problem_l108_108710
-- Lean Statement for the proof problem

variable {x y z : ℝ}

theorem problem_equiv_math_problem (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (eq1 : x^2 + x * y + y^2 / 3 = 25) 
  (eq2 : y^2 / 3 + z^2 = 9) 
  (eq3 : z^2 + z * x + x^2 = 16) :
  x * y + 2 * y * z + 3 * z * x = 24 * Real.sqrt 3 :=
by
  sorry

end problem_equiv_math_problem_l108_108710


namespace parallel_lines_intersect_hyperbola_l108_108090

theorem parallel_lines_intersect_hyperbola :
  ∀ (k : ℝ) (xK xL xM xN : ℝ),
  (∃ K L M N : ℝ × ℝ,
    K = (xK, k * xK + 14) ∧ L = (xL, k * xL + 14) ∧
    M = (xM, k * xM + 4)  ∧ N = (xN, k * xN + 4) ∧
    k * xK^2 + 14 * xK - 1 = 0 ∧
    k * xL^2 + 14 * xL - 1 = 0 ∧
    k * xM^2 + 4 * xM - 1 = 0 ∧
    k * xN^2 + 4 * xN - 1 = 0 ) →
  ∃ AL AK BN BM : ℝ,
    (AL - AK) / (BN - BM) = 3.5 := by 
  sorry

end parallel_lines_intersect_hyperbola_l108_108090


namespace least_positive_base_ten_number_with_seven_digit_binary_representation_l108_108745

theorem least_positive_base_ten_number_with_seven_digit_binary_representation :
  ∃ n : ℤ, n > 0 ∧ (∀ k : ℤ, k > 0 ∧ k < n → digit_length binary_digit_representation k < 7) ∧ digit_length binary_digit_representation n = 7 :=
sorry

end least_positive_base_ten_number_with_seven_digit_binary_representation_l108_108745


namespace incorrect_statements_l108_108648

variables (a b c : ℝ × ℝ)

def is_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k • w)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  (v.1 * w.1) + (v.2 * w.2)

theorem incorrect_statements :
  ¬ ((is_parallel a b ∧ is_parallel b c) → is_parallel a c) ∧
  (dot_product a b = dot_product a c ∧ a ≠ (0, 0) → b ≠ c) ∧
  ¬ ((dot_product a b) * c = a * (dot_product b c)) :=
sorry

end incorrect_statements_l108_108648


namespace average_of_roots_l108_108613

theorem average_of_roots (p q : ℝ) (h : ∃ x1 x2 : ℝ, 3*p*x1^2 - 6*p*x1 + q = 0 ∧ 3*p*x2^2 - 6*p*x2 + q = 0 ∧ x1 ≠ x2):
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 3*p*x1^2 - 6*p*x1 + q = 0 ∧ 3*p*x2^2 - 6*p*x2 + q = 0) → 
  (x1 + x2) / 2 = 1 :=
by
  sorry

end average_of_roots_l108_108613


namespace find_f_neg_8point5_l108_108852

def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodicity : ∀ x : ℝ, f (x + 2) = -f x
axiom initial_condition : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x

theorem find_f_neg_8point5 : f (-8.5) = -0.5 :=
by
  -- Expect this proof to follow the outlined logic
  sorry

end find_f_neg_8point5_l108_108852


namespace correct_mean_after_correction_l108_108101

theorem correct_mean_after_correction
  (n : ℕ) (incorrect_mean : ℝ) (incorrect_value : ℝ) (correct_value : ℝ)
  (h : n = 30) (h_mean : incorrect_mean = 150) (h_incorrect_value : incorrect_value = 135) (h_correct_value : correct_value = 165) :
  (incorrect_mean * n - incorrect_value + correct_value) / n = 151 :=
  by
  sorry

end correct_mean_after_correction_l108_108101


namespace positive_difference_of_two_numbers_l108_108877

theorem positive_difference_of_two_numbers 
  (x y : ℝ) 
  (h1 : x + y = 10) 
  (h2 : x^2 - y^2 = 24) : 
  |x - y| = 2.4 := 
sorry

end positive_difference_of_two_numbers_l108_108877


namespace grisha_win_probability_expected_number_coin_flips_l108_108042

-- Problem 1: Probability of Grisha's win
theorem grisha_win_probability : 
  ∀ (even_heads_criteria : ℤ → ℚ) (odd_tails_criteria : ℤ → ℚ), 
    even_heads_criteria ∈ {2 * n | n : ℤ} → 
    odd_tails_criteria ∈ {2 * n + 1 | n : ℤ} → 
    (grisha_win : ℚ) :=
  sorry

-- Problem 2: Expected number of coin flips
theorem expected_number_coin_flips : 
  ∀ (even_heads_criteria : ℤ → ℚ) (odd_tails_criteria : ℤ → ℚ), 
    even_heads_criteria ∈ {2 * n | n : ℤ} → 
    odd_tails_criteria ∈ {2 * n + 1 | n : ℤ} → 
    (expected_tosses : ℚ) :=
  sorry

end grisha_win_probability_expected_number_coin_flips_l108_108042


namespace probability_of_drawing_A_l108_108605

-- Definitions of the conditions
variables (n : ℕ) (A_units : ℕ := 10) (B_units : ℕ := 15) (total_products : ℕ := A_units + B_units)

-- Probabilities of combinations
noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

noncomputable def comb (n k : ℕ) : ℕ :=
factorial n / (factorial k * factorial (n - k))

noncomputable def prob_draw_A : ℚ :=
(comb A_units 2 + comb A_units 1 * comb B_units 1) / comb total_products 2

-- Main proof statement
theorem probability_of_drawing_A :
  prob_draw_A = 39 / 60 :=
sorry

end probability_of_drawing_A_l108_108605


namespace ordering_of_means_l108_108697

variable {a b : ℝ}

-- Define that a and b are positive and unequal
def conditions : Prop := a > 0 ∧ b > 0 ∧ a ≠ b

-- Define the arithmetic mean
def AM (a b : ℝ) : ℝ := (a^2 + b^2) / 2

-- Define the geometric mean
def GM (a b : ℝ) : ℝ := a * b

-- Define the harmonic mean
def HM (a b : ℝ) : ℝ := 2 * a^2 * b^2 / (a^2 + b^2)

theorem ordering_of_means :
  conditions → AM a b > GM a b ∧ GM a b > HM a b :=
by
  intros
  sorry

end ordering_of_means_l108_108697


namespace statement_A_statement_D_l108_108468

variable (a b c d : ℝ)

-- Statement A: If ac² > bc², then a > b
theorem statement_A (h1 : a * c^2 > b * c^2) (h2 : c ≠ 0) : a > b := by
  sorry

-- Statement D: If a > b > 0, then a + 1/b > b + 1/a
theorem statement_D (h1 : a > b) (h2 : b > 0) : a + 1 / b > b + 1 / a := by
  sorry

end statement_A_statement_D_l108_108468


namespace find_x_l108_108676

-- Define the vectors a, b, and c
def a : ℝ × ℝ := (-2, 0)
def b : ℝ × ℝ := (2, 1)
def c (x : ℝ) : ℝ × ℝ := (x, 1)

-- Define the collinearity condition
def collinear_with_3a_plus_b (x : ℝ) : Prop :=
  ∃ k : ℝ, c x = k • (3 • a + b)

theorem find_x :
  ∀ x : ℝ, collinear_with_3a_plus_b x → x = -4 := 
sorry

end find_x_l108_108676


namespace probability_one_defective_l108_108755

theorem probability_one_defective (g d : Nat) (h1 : g = 3) (h2 : d = 1) : 
  let total_combinations := (g + d).choose 2
  let favorable_outcomes := g * d
  favorable_outcomes / total_combinations = 1 / 2 := by
sorry

end probability_one_defective_l108_108755


namespace no_solution_for_x_l108_108034

theorem no_solution_for_x (m : ℝ) :
  (∀ x : ℝ, x ≠ 1 → (mx - 1) / (x - 1) ≠ 3) ↔ (m = 1 ∨ m = 3) :=
by
  sorry

end no_solution_for_x_l108_108034


namespace probability_five_cards_l108_108952

/-- 
There are a standard deck of 52 cards. 
The sequence of drawing cards is considered:
1. The first card is a King.
2. The second card is a heart.
3. The third card is a Jack.
4. The fourth card is a spade.
5. The fifth card is a Queen.

The goal is to compute the probability of this sequence.
-/
noncomputable def card_probability : ℚ :=
  let prob_king := (4 / 52: ℚ)
  let prob_heart := (12 / 51: ℚ)
  let prob_jack := (4 / 50: ℚ)
  let prob_spade := (12 / 49: ℚ)
  let prob_queen := (4 / 48: ℚ)
  prob_king * prob_heart * prob_jack * prob_spade * prob_queen

theorem probability_five_cards :
  card_probability = (3 / 10125: ℚ) :=
by
  -- Probability calculations
  let prob_king := (4 / 52: ℚ)
  let prob_heart := (12 / 51: ℚ)
  let prob_jack := (4 / 50: ℚ)
  let prob_spade := (12 / 49: ℚ)
  let prob_queen := (4 / 48: ℚ)
  have h : prob_king * prob_heart * prob_jack * prob_spade * prob_queen = (3 / 10125: ℚ)
  sorry

end probability_five_cards_l108_108952


namespace focal_length_is_correct_l108_108084

def hyperbola_eqn : Prop := (∀ x y : ℝ, (x^2 / 4) - (y^2 / 9) = 1 → True)

noncomputable def focal_length_of_hyperbola : ℝ :=
  2 * Real.sqrt (4 + 9)

theorem focal_length_is_correct : hyperbola_eqn → focal_length_of_hyperbola = 2 * Real.sqrt 13 := by
  intro h
  sorry

end focal_length_is_correct_l108_108084


namespace sqrt_expression_simplified_l108_108270

theorem sqrt_expression_simplified :
  sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  sorry

end sqrt_expression_simplified_l108_108270


namespace bobby_total_candy_l108_108345

theorem bobby_total_candy (candy1 candy2 : ℕ) (h1 : candy1 = 26) (h2 : candy2 = 17) : candy1 + candy2 = 43 := 
by 
  sorry

end bobby_total_candy_l108_108345


namespace parallelogram_side_length_l108_108441

theorem parallelogram_side_length (a b : ℕ) (h1 : 2 * (a + b) = 16) (h2 : a = 5) : b = 3 :=
by
  sorry

end parallelogram_side_length_l108_108441


namespace percentage_greater_than_l108_108674

-- Definitions of the variables involved
variables (X Y Z : ℝ)

-- Lean statement to prove the formula
theorem percentage_greater_than (X Y Z : ℝ) : 
  (100 * (X - Y)) / (Y + Z) = (100 * (X - Y)) / (Y + Z) :=
by
  -- skipping the actual proof
  sorry

end percentage_greater_than_l108_108674


namespace maximum_students_l108_108620

theorem maximum_students (x : ℕ) (hx : x / 2 + x / 4 + x / 7 + 6 > x) : x ≤ 28 :=
by sorry

end maximum_students_l108_108620


namespace width_of_lawn_is_30_m_l108_108781

-- Define the conditions
def lawn_length : ℕ := 70
def lawn_width : ℕ := 30
def road_width : ℕ := 5
def gravel_rate_per_sqm : ℕ := 4
def gravel_cost : ℕ := 1900

-- Mathematically equivalent proof problem statement
theorem width_of_lawn_is_30_m 
  (H1 : lawn_length = 70)
  (H2 : road_width = 5)
  (H3 : gravel_rate_per_sqm = 4)
  (H4 : gravel_cost = 1900)
  (H5 : 2*road_width*5 + (lawn_length - road_width) * 5 * gravel_rate_per_sqm = gravel_cost) :
  lawn_width = 30 := 
sorry

end width_of_lawn_is_30_m_l108_108781


namespace storm_time_l108_108223

def time_corrected (hours tens units : ℕ) (minutes tens units : ℕ) : Prop :=
(hours tens units = 1 ∨ hours tens units = 3) ∧
(hours units = 1 ∨ hours units = 9) ∧
(minutes tens units = 1 ∨ minutes tens units = 9) ∧
(minutes units = 8 ∨ minutes units = 0)

theorem storm_time :
  ∃ hours tens units: ℕ, ∃ hours units: ℕ, ∃ minutes tens units: ℕ, ∃ minutes units: ℕ,
  time_corrected hours tens units minutes tens units ∧
  (hours tens units = 1) ∧ (hours units = 1) ∧ (minutes tens units = 1) ∧ (minutes units = 8) :=
begin
  use 1,
  use 1,
  use 1,
  use 8,
  split,
  { split,
    { left, refl },
    split,
    { left, refl },
    split,
    { left, refl },
    { left, refl }},
  exact ⟨rfl, rfl, rfl, rfl⟩
end

end storm_time_l108_108223


namespace find_y_l108_108723

variable (x y z : ℕ)

-- Conditions
def condition1 : Prop := 100 + 200 + 300 + x = 1000
def condition2 : Prop := 300 + z + 100 + x + y = 1000

-- Theorem to be proven
theorem find_y (h1 : condition1 x) (h2 : condition2 x y z) : z + y = 200 :=
sorry

end find_y_l108_108723


namespace smallest_k_mod_19_7_3_l108_108737

theorem smallest_k_mod_19_7_3 : ∃ k : ℕ, k > 1 ∧ (k % 19 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) ∧ k = 400 := 
by {
  -- statements of conditions in form of hypotheses
  let h1 := k > 1,
  let h2 := k % 19 = 1,
  let h3 := k % 7 = 1,
  let h4 := k % 3 = 1,
  -- goal of the theorem
  exact ⟨400, _⟩ sorry -- we indicate the goal should be of the form ⟨value, proof⟩, and fill in the proof with 'sorry'
}

end smallest_k_mod_19_7_3_l108_108737


namespace sqrt_expression_simplified_l108_108267

theorem sqrt_expression_simplified :
  sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  sorry

end sqrt_expression_simplified_l108_108267


namespace women_in_department_l108_108686

theorem women_in_department : 
  ∀ (total_students men women : ℕ) (men_percentage women_percentage : ℝ),
  men_percentage = 0.70 →
  women_percentage = 0.30 →
  men = 420 →
  total_students = men / men_percentage →
  women = total_students * women_percentage →
  women = 180 :=
by
  intros total_students men women men_percentage women_percentage
  intros h1 h2 h3 h4 h5
  sorry

end women_in_department_l108_108686


namespace consecutive_numbers_count_l108_108883

-- Definitions and conditions
variables (n : ℕ) (x : ℕ)
axiom avg_condition : (2 * 33 = 2 * x + n - 1)
axiom highest_num_condition : (x + (n - 1) = 36)

-- Thm statement
theorem consecutive_numbers_count : n = 7 :=
by
  sorry

end consecutive_numbers_count_l108_108883


namespace point_on_y_axis_is_zero_l108_108824

-- Given conditions
variables (m : ℝ) (y : ℝ)
-- \( P(m, 2) \) lies on the y-axis
def point_on_y_axis (m y : ℝ) : Prop := (m = 0)

-- Proof statement: Prove that if \( P(m, 2) \) lies on the y-axis, then \( m = 0 \)
theorem point_on_y_axis_is_zero (h : point_on_y_axis m 2) : m = 0 :=
by 
  -- the proof would go here
  sorry

end point_on_y_axis_is_zero_l108_108824


namespace compose_f_g_f_l108_108416

def f (x : ℝ) : ℝ := 2 * x + 5
def g (x : ℝ) : ℝ := 3 * x + 4

theorem compose_f_g_f (x : ℝ) : f (g (f 3)) = 79 := by
  sorry

end compose_f_g_f_l108_108416


namespace find_mother_age_l108_108213

-- Definitions for the given conditions
def serena_age_now := 9
def years_in_future := 6
def serena_age_future := serena_age_now + years_in_future
def mother_age_future (M : ℕ) := 3 * serena_age_future

-- The main statement to prove
theorem find_mother_age (M : ℕ) (h1 : M = mother_age_future M - years_in_future) : M = 39 :=
by
  sorry

end find_mother_age_l108_108213


namespace mo_tea_cups_l108_108709

theorem mo_tea_cups (n t : ℤ) (h1 : 4 * n + 3 * t = 22) (h2 : 3 * t = 4 * n + 8) : t = 5 :=
by
  -- proof steps
  sorry

end mo_tea_cups_l108_108709


namespace find_v2002_l108_108867

def g (x : ℕ) : ℕ :=
  match x with
  | 1 => 5
  | 2 => 3
  | 3 => 6
  | 4 => 2
  | 5 => 1
  | 6 => 7
  | 7 => 4
  | _ => 0

def seq_v : ℕ → ℕ
| 0       => 5
| (n + 1) => g (seq_v n)

theorem find_v2002 : seq_v 2002 = 5 :=
  sorry

end find_v2002_l108_108867


namespace exactly_one_pair_probability_l108_108354

def four_dice_probability : ℚ :=
  sorry  -- Here we skip the actual computation and proof

theorem exactly_one_pair_probability : four_dice_probability = 5/9 := by {
  -- Placeholder for proof, explanation, and calculation
  sorry
}

end exactly_one_pair_probability_l108_108354


namespace sqrt_mul_sqrt_l108_108245

theorem sqrt_mul_sqrt (h1 : Real.sqrt 25 = 5) : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_mul_sqrt_l108_108245


namespace distinct_distances_l108_108478

theorem distinct_distances (points : Finset (ℝ × ℝ)) (h : points.card = 2016) :
  ∃ s : Finset ℝ, s.card ≥ 45 ∧ ∀ p ∈ points, ∃ q ∈ points, p ≠ q ∧ 
    (s = (points.image (λ r => dist p r)).filter (λ x => x ≠ 0)) :=
by
  sorry

end distinct_distances_l108_108478


namespace function_relation4_l108_108673

open Set

section
  variable (M : Set ℤ) (N : Set ℤ)

  def relation1 (x : ℤ) := x ^ 2
  def relation2 (x : ℤ) := x + 1
  def relation3 (x : ℤ) := x - 1
  def relation4 (x : ℤ) := abs x

  theorem function_relation4 : 
    M = {-1, 1, 2, 4} →
    N = {1, 2, 4} →
    (∀ x ∈ M, relation4 x ∈ N) :=
  by
    intros hM hN
    simp [relation4]
    sorry
end

end function_relation4_l108_108673


namespace correct_proposition_four_l108_108169

universe u

-- Definitions
variable {Point : Type u} (A B : Point) (a α : Set Point)
variable (h5 : A ∉ α)
variable (h6 : a ⊂ α)

-- The statement to be proved
theorem correct_proposition_four : A ∉ a :=
sorry

end correct_proposition_four_l108_108169


namespace inequality_of_sums_l108_108421

theorem inequality_of_sums
  (a1 a2 b1 b2 : ℝ)
  (h1 : 0 < a1)
  (h2 : 0 < a2)
  (h3 : a1 > a2)
  (h4 : b1 ≥ a1)
  (h5 : b1 * b2 ≥ a1 * a2) :
  b1 + b2 ≥ a1 + a2 :=
by
  -- Here we don't provide the proof
  sorry

end inequality_of_sums_l108_108421


namespace correct_equation_D_l108_108308

theorem correct_equation_D : (|5 - 3| = - (3 - 5)) :=
by
  sorry

end correct_equation_D_l108_108308


namespace sqrt_49_times_sqrt_25_l108_108293

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 :=
by
  sorry

end sqrt_49_times_sqrt_25_l108_108293


namespace abby_and_damon_weight_l108_108488

variables {a b c d : ℝ}

theorem abby_and_damon_weight (h1 : a + b = 260) (h2 : b + c = 245) 
(h3 : c + d = 270) (h4 : a + c = 220) : a + d = 285 := 
by 
  sorry

end abby_and_damon_weight_l108_108488


namespace sqrt_product_l108_108275

theorem sqrt_product (a b : ℝ) (h1 : a = 49) (h2 : b = 25) : sqrt (a * sqrt b) = 7 * sqrt 5 :=
by sorry

end sqrt_product_l108_108275


namespace sqrt_expression_simplified_l108_108269

theorem sqrt_expression_simplified :
  sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  sorry

end sqrt_expression_simplified_l108_108269


namespace range_of_x_l108_108373

theorem range_of_x (S : ℕ → ℕ) (a : ℕ → ℕ) (x : ℕ) :
  (∀ n, n ≥ 2 → S (n - 1) + S n = 2 * n^2 + 1) →
  S 0 = 0 →
  a 1 = x →
  (∀ n, a n ≤ a (n + 1)) →
  2 < x ∧ x < 3 := 
sorry

end range_of_x_l108_108373


namespace probability_of_green_l108_108485

-- Define the conditions
def P_R : ℝ := 0.15
def P_O : ℝ := 0.35
def P_B : ℝ := 0.2
def total_probability (P_Y P_G : ℝ) : Prop := P_R + P_O + P_B + P_Y + P_G = 1

-- State the theorem to be proven
theorem probability_of_green (P_Y : ℝ) (P_G : ℝ) (h : total_probability P_Y P_G) (P_Y_assumption : P_Y = 0.15) : P_G = 0.15 :=
by
  sorry

end probability_of_green_l108_108485


namespace students_from_other_communities_l108_108999

noncomputable def percentageMuslims : ℝ := 0.41
noncomputable def percentageHindus : ℝ := 0.32
noncomputable def percentageSikhs : ℝ := 0.12
noncomputable def totalStudents : ℝ := 1520

theorem students_from_other_communities : 
  totalStudents * (1 - (percentageMuslims + percentageHindus + percentageSikhs)) = 228 := 
by 
  sorry

end students_from_other_communities_l108_108999


namespace circle_center_coordinates_l108_108520

-- Definition of the circle's equation
def circle_eq : Prop := ∀ x y : ℝ, (x - 2)^2 + (y + 1)^2 = 3

-- Proof of the circle's center coordinates
theorem circle_center_coordinates : ∃ h k : ℝ, (h, k) = (2, -1) := 
sorry

end circle_center_coordinates_l108_108520


namespace green_yarn_length_l108_108581

/-- The length of the green piece of yarn given the red yarn is 8 cm more 
than three times the length of the green yarn and the total length 
for 2 pieces of yarn is 632 cm. -/
theorem green_yarn_length (G R : ℕ) 
  (h1 : R = 3 * G + 8)
  (h2 : G + R = 632) : 
  G = 156 := 
by
  sorry

end green_yarn_length_l108_108581


namespace moles_of_HCl_formed_l108_108010

theorem moles_of_HCl_formed
  (C2H6_initial : Nat)
  (Cl2_initial : Nat)
  (HCl_expected : Nat)
  (balanced_reaction : C2H6_initial + Cl2_initial = C2H6_initial + HCl_expected):
  C2H6_initial = 2 → Cl2_initial = 2 → HCl_expected = 2 :=
by
  intros
  sorry

end moles_of_HCl_formed_l108_108010


namespace Nellie_needs_to_sell_more_rolls_l108_108954

-- Define the conditions
def total_needed : ℕ := 45
def sold_grandmother : ℕ := 1
def sold_uncle : ℕ := 10
def sold_neighbor : ℕ := 6

-- Define the total sold
def total_sold : ℕ := sold_grandmother + sold_uncle + sold_neighbor

-- Define the remaining rolls needed
def remaining_rolls := total_needed - total_sold

-- Statement to prove that remaining_rolls equals 28
theorem Nellie_needs_to_sell_more_rolls : remaining_rolls = 28 := by
  unfold remaining_rolls
  unfold total_sold
  unfold total_needed sold_grandmother sold_uncle sold_neighbor
  calc
  45 - (1 + 10 + 6) = 45 - 17 : by rw [Nat.add_assoc]
  ... = 28 : by norm_num

end Nellie_needs_to_sell_more_rolls_l108_108954


namespace p_arithmetic_fibonacci_term_correct_l108_108005

noncomputable def p_arithmetic_fibonacci_term (p : ℕ) : ℝ :=
  5 ^ ((p - 1) / 2)

theorem p_arithmetic_fibonacci_term_correct (p : ℕ) : p_arithmetic_fibonacci_term p = 5 ^ ((p - 1) / 2) := 
by 
  rfl -- direct application of the definition

#check p_arithmetic_fibonacci_term_correct

end p_arithmetic_fibonacci_term_correct_l108_108005


namespace chips_count_l108_108178

theorem chips_count (B G P R x : ℕ) 
  (hx1 : 5 < x) (hx2 : x < 11) 
  (h : 1^B * 5^G * x^P * 11^R = 28160) : 
  P = 2 :=
by 
  -- Hint: Prime factorize 28160 to apply constraints and identify corresponding exponents.
  have prime_factorization_28160 : 28160 = 2^6 * 5^1 * 7^2 := by sorry
  -- Given 5 < x < 11 and by prime factorization, x can only be 7 (since it factors into the count of 7)
  -- Complete the rest of the proof
  sorry

end chips_count_l108_108178


namespace sqrt_expression_eq_three_l108_108234

theorem sqrt_expression_eq_three (h: (Real.sqrt 81) = 9) : Real.sqrt ((Real.sqrt 81 + Real.sqrt 81) / 2) = 3 :=
by 
  sorry

end sqrt_expression_eq_three_l108_108234


namespace retail_profit_percent_l108_108339

variable (CP : ℝ) (MP : ℝ) (SP : ℝ)
variable (h_marked : MP = CP + 0.60 * CP)
variable (h_discount : SP = MP - 0.25 * MP)

theorem retail_profit_percent : CP = 100 → MP = CP + 0.60 * CP → SP = MP - 0.25 * MP → 
       (SP - CP) / CP * 100 = 20 := 
by
  intros h1 h2 h3
  sorry

end retail_profit_percent_l108_108339


namespace thursday_to_wednesday_ratio_l108_108770

-- Let M, T, W, Th be the number of messages sent on Monday, Tuesday, Wednesday, and Thursday respectively.
variables (M T W Th : ℕ)

-- Conditions are given as follows
axiom hM : M = 300
axiom hT : T = 200
axiom hW : W = T + 300
axiom hSum : M + T + W + Th = 2000

-- Define the function to compute the ratio
def ratio (a b : ℕ) : ℚ := a / b

-- The target is to prove that the ratio of the messages sent on Thursday to those sent on Wednesday is 2 / 1
theorem thursday_to_wednesday_ratio : ratio Th W = 2 :=
by {
  sorry
}

end thursday_to_wednesday_ratio_l108_108770


namespace four_gt_sqrt_fifteen_l108_108629

theorem four_gt_sqrt_fifteen : 4 > Real.sqrt 15 := 
sorry

end four_gt_sqrt_fifteen_l108_108629


namespace max_lines_with_specific_angles_l108_108721

def intersecting_lines : ℕ := 6

theorem max_lines_with_specific_angles :
  ∀ (n : ℕ), (∀ (i j : ℕ), i ≠ j → (∃ θ : ℝ, θ = 30 ∨ θ = 60 ∨ θ = 90)) → n ≤ 6 :=
  sorry

end max_lines_with_specific_angles_l108_108721


namespace upstream_distance_l108_108233

variable (Vb Vs Vdown Vup Dup : ℕ)

def boatInStillWater := Vb = 36
def speedStream := Vs = 12
def downstreamSpeed := Vdown = Vb + Vs
def upstreamSpeed := Vup = Vb - Vs
def timeEquality := 80 / Vdown = Dup / Vup

theorem upstream_distance (Vb Vs Vdown Vup Dup : ℕ) 
  (h1 : boatInStillWater Vb)
  (h2 : speedStream Vs)
  (h3 : downstreamSpeed Vb Vs Vdown)
  (h4 : upstreamSpeed Vb Vs Vup)
  (h5 : timeEquality Vdown Vup Dup) : Dup = 40 := 
sorry

end upstream_distance_l108_108233


namespace total_blocks_traveled_l108_108342

-- Given conditions as definitions
def annie_walked_blocks : ℕ := 5
def annie_rode_blocks : ℕ := 7

-- The total blocks Annie traveled
theorem total_blocks_traveled : annie_walked_blocks + annie_rode_blocks + (annie_walked_blocks + annie_rode_blocks) = 24 := by
  sorry

end total_blocks_traveled_l108_108342


namespace carla_games_won_l108_108651

theorem carla_games_won (F C : ℕ) (h1 : F + C = 30) (h2 : F = C / 2) : C = 20 :=
by
  sorry

end carla_games_won_l108_108651


namespace range_of_m_l108_108163

noncomputable def y (m x : ℝ) := m * (1/4)^x - (1/2)^x + 1

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, y m x = 0) → (m ≤ 0 ∨ m = 1 / 4) := sorry

end range_of_m_l108_108163


namespace find_m_values_l108_108990

theorem find_m_values {m : ℝ} :
  (∀ x : ℝ, mx^2 + (m+2) * x + (1 / 2) * m + 1 = 0 → x = 0) 
  ↔ (m = 0 ∨ m = 2 ∨ m = -2) :=
by sorry

end find_m_values_l108_108990


namespace wire_length_ratio_l108_108790

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

end wire_length_ratio_l108_108790


namespace least_value_expr_l108_108097

   variable {x y : ℝ}

   theorem least_value_expr : ∃ x y : ℝ, (x^3 * y - 1)^2 + (x + y)^2 = 1 :=
   by
     sorry
   
end least_value_expr_l108_108097


namespace max_value_y_l108_108961

theorem max_value_y (x : ℝ) (h : x < 5 / 4) : 
  (4 * x - 2 + 1 / (4 * x - 5)) ≤ 1 :=
sorry

end max_value_y_l108_108961


namespace island_inhabitants_even_l108_108068

theorem island_inhabitants_even 
  (total : ℕ) 
  (knights liars : ℕ)
  (H : total = knights + liars)
  (H1 : ∃ (knk : Prop), (knk → (knights % 2 = 0)) ∧ (¬knk → (knights % 2 = 1)))
  (H2 : ∃ (lkr : Prop), (lkr → (liars % 2 = 1)) ∧ (¬lkr → (liars % 2 = 0)))
  : (total % 2 = 0) := sorry

end island_inhabitants_even_l108_108068


namespace total_time_taken_l108_108232

theorem total_time_taken
  (speed_boat : ℝ)
  (speed_stream : ℝ)
  (distance : ℝ)
  (h_boat : speed_boat = 12)
  (h_stream : speed_stream = 5)
  (h_distance : distance = 325) :
  (distance / (speed_boat - speed_stream) + distance / (speed_boat + speed_stream)) = 65.55 :=
by
  sorry

end total_time_taken_l108_108232


namespace factor_t_sq_minus_64_l108_108942

def isDifferenceOfSquares (a b : Int) : Prop := a = b^2

theorem factor_t_sq_minus_64 (t : Int) (h : isDifferenceOfSquares 64 8) : (t^2 - 64) = (t - 8) * (t + 8) := by
  sorry

end factor_t_sq_minus_64_l108_108942


namespace num_words_with_A_l108_108381

theorem num_words_with_A :
  let total_words := 5^4,
      words_without_A := 4^4 in
  total_words - words_without_A = 369 :=
by
  sorry

end num_words_with_A_l108_108381


namespace inequality_abc_l108_108372

variable (a b c : ℝ)

theorem inequality_abc (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) :
  a / (a^3 - a^2 + 3) + b / (b^3 - b^2 + 3) + c / (c^3 - c^2 + 3) ≤ 1 := 
sorry

end inequality_abc_l108_108372


namespace pizza_slice_volume_l108_108329

-- Define the parameters given in the conditions
def pizza_thickness : ℝ := 0.5
def pizza_diameter : ℝ := 16.0
def num_slices : ℝ := 16.0

-- Define the volume of one slice
theorem pizza_slice_volume : (π * (pizza_diameter / 2) ^ 2 * pizza_thickness / num_slices) = 2 * π := by
  sorry

end pizza_slice_volume_l108_108329


namespace sequence_solution_l108_108524

theorem sequence_solution
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_a1 : a 1 = 10)
  (h_b1 : b 1 = 10)
  (h_recur_a : ∀ n : ℕ, a (n + 1) = 1 / (a n * b n))
  (h_recur_b : ∀ n : ℕ, b (n + 1) = (a n)^4 * b n) :
  (∀ n : ℕ, n > 0 → a n = 10^((2 - 3 * n) * (-1 : ℝ)^n) ∧ b n = 10^((6 * n - 7) * (-1 : ℝ)^n)) :=
by
  sorry

end sequence_solution_l108_108524


namespace sqrt_49_times_sqrt_25_eq_7sqrt5_l108_108264

theorem sqrt_49_times_sqrt_25_eq_7sqrt5 :
  (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  have h1 : Real.sqrt 25 = 5 := by sorry
  have h2 : 49 * 5 = 245 := by sorry
  have h3 : Real.sqrt 245 = 7 * Real.sqrt 5 := by sorry
  sorry

end sqrt_49_times_sqrt_25_eq_7sqrt5_l108_108264


namespace denominator_of_fraction_l108_108459

theorem denominator_of_fraction (n : ℕ) (h1 : n = 20) (h2 : num = 35) (dec_value : ℝ) (h3 : dec_value = 2 / 10^n) : denom = 175 * 10^20 :=
by
  sorry

end denominator_of_fraction_l108_108459


namespace tan_alpha_fraction_value_l108_108160

theorem tan_alpha_fraction_value {α : Real} (h : Real.tan α = 2) : 
  (3 * Real.sin α + Real.cos α) / (5 * Real.sin α + 2 * Real.cos α) = 7 / 12 :=
by
  sorry

end tan_alpha_fraction_value_l108_108160


namespace cupboard_cost_price_l108_108764

theorem cupboard_cost_price (C : ℝ) 
  (h1 : ∀ C₀, C = C₀ → C₀ * 0.88 + 1500 = C₀ * 1.12) :
  C = 6250 := by
  sorry

end cupboard_cost_price_l108_108764


namespace sandy_age_l108_108575

theorem sandy_age (S M : ℕ) (h1 : M = S + 14) (h2 : S / M = 7 / 9) : S = 49 :=
sorry

end sandy_age_l108_108575


namespace equal_roots_of_quadratic_l108_108364

theorem equal_roots_of_quadratic (k : ℝ) : 
  ( ∀ x : ℝ, 2 * k * x^2 + 7 * k * x + 2 = 0 → x = x ) ↔ k = 16 / 49 :=
by
  sorry

end equal_roots_of_quadratic_l108_108364


namespace ratio_of_boys_l108_108536

theorem ratio_of_boys 
  (p : ℚ) 
  (h : p = (3/4) * (1 - p)) : 
  p = 3 / 7 :=
by
  sorry

end ratio_of_boys_l108_108536


namespace range_of_heights_l108_108728

theorem range_of_heights (max_height min_height : ℝ) (h_max : max_height = 175) (h_min : min_height = 100) :
  (max_height - min_height) = 75 :=
by
  -- Defer proof
  sorry

end range_of_heights_l108_108728


namespace distance_between_trains_l108_108092

theorem distance_between_trains (d1 d2 : ℝ) (t1 t2 : ℝ) (s1 s2 : ℝ) (x : ℝ) :
  d1 = d2 + 100 →
  s1 = 50 →
  s2 = 40 →
  d1 = s1 * t1 →
  d2 = s2 * t2 →
  t1 = t2 →
  d2 = 400 →
  d1 + d2 = 900 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end distance_between_trains_l108_108092


namespace sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l108_108285

theorem sqrt_49_mul_sqrt_25_eq_7_sqrt_5 : (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l108_108285


namespace sqrt_expression_simplified_l108_108266

theorem sqrt_expression_simplified :
  sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  sorry

end sqrt_expression_simplified_l108_108266


namespace simplify_complex_expression_l108_108076

variables (x y : ℝ) (i : ℂ)

theorem simplify_complex_expression (h : i^2 = -1) :
  (x + 3 * i * y) * (x - 3 * i * y) = x^2 - 9 * y^2 :=
by sorry

end simplify_complex_expression_l108_108076


namespace a_n_formula_l108_108655

variable {a : ℕ+ → ℝ}  -- Defining a_n as a sequence from positive natural numbers to real numbers
variable {S : ℕ+ → ℝ}  -- Defining S_n as a sequence from positive natural numbers to real numbers

-- Given conditions
axiom S_def (n : ℕ+) : S n = a n / 2 + 1 / a n - 1
axiom a_pos (n : ℕ+) : a n > 0

-- Conjecture to be proved
theorem a_n_formula (n : ℕ+) : a n = Real.sqrt (2 * n + 1) - Real.sqrt (2 * n - 1) := 
sorry -- proof to be done

end a_n_formula_l108_108655


namespace production_time_l108_108896

-- Define the conditions
def machineProductionRate (machines: ℕ) (units: ℕ) (hours: ℕ): ℕ := units / machines / hours

-- The question we need to answer: How long will it take for 10 machines to produce 100 units?
theorem production_time (h1 : machineProductionRate 5 20 10 = 4 / 10)
  : 10 * 0.4 * 25 = 100 :=
by sorry

end production_time_l108_108896


namespace number_of_ways_to_choose_water_polo_team_l108_108207

theorem number_of_ways_to_choose_water_polo_team :
  let total_members := 15
  let team_size := 7
  let goalie_positions := 1
  let player_choices := team_size - goalie_positions
  ∃ (total_ways : ℕ), 
  total_ways = total_members * Nat.choose (total_members - 1) player_choices ∧ 
  total_ways = 45045 :=
by
  let total_members := 15
  let team_size := 7
  let goalie_positions := 1
  let player_choices := team_size - goalie_positions
  have total_ways : ℕ := total_members * Nat.choose (total_members - 1) player_choices
  use total_ways
  sorry

end number_of_ways_to_choose_water_polo_team_l108_108207


namespace extreme_value_at_one_l108_108837

noncomputable def f (x : ℝ) (a : ℝ) := (x^2 + a) / (x + 1)

theorem extreme_value_at_one (a : ℝ) :
  (∃ x : ℝ, x = 1 ∧ (∀ ε > 0, ∃ δ > 0, ∀ y : ℝ, abs (y-1) < δ → abs (f y a - f 1 a) < ε)) →
  a = 3 :=
by
  sorry

end extreme_value_at_one_l108_108837


namespace quadratic_has_two_distinct_roots_l108_108149

theorem quadratic_has_two_distinct_roots (a b c : ℝ) (h : 2016 + a^2 + a * c < a * b) : 
  (b^2 - 4 * a * c) > 0 :=
by {
  sorry
}

end quadratic_has_two_distinct_roots_l108_108149


namespace distance_from_origin_l108_108184

theorem distance_from_origin :
  let origin := (0, 0)
      pt := (8, -15)
  in Real.sqrt ((pt.1 - origin.1)^2 + (pt.2 - origin.2)^2) = 17 :=
by
  let origin := (0, 0)
  let pt := (8, -15)
  have h1 : pt.1 - origin.1 = 8 := by rfl
  have h2 : pt.2 - origin.2 = -15 := by rfl
  simp [h1, h2]
  -- sorry will be used to skip the actual proof steps and to ensure the code builds successfully
  sorry

end distance_from_origin_l108_108184


namespace cricket_target_run_rate_cricket_wicket_partnership_score_l108_108687

noncomputable def remaining_runs_needed (initial_runs : ℕ) (target_runs : ℕ) : ℕ :=
  target_runs - initial_runs

noncomputable def required_run_rate (remaining_runs : ℕ) (remaining_overs : ℕ) : ℚ :=
  (remaining_runs : ℚ) / remaining_overs

theorem cricket_target_run_rate (initial_runs : ℕ) (target_runs : ℕ) (remaining_overs : ℕ)
  (initial_wickets : ℕ) :
  initial_runs = 32 → target_runs = 282 → remaining_overs = 40 → initial_wickets = 3 →
  required_run_rate (remaining_runs_needed initial_runs target_runs) remaining_overs = 6.25 :=
by
  sorry


theorem cricket_wicket_partnership_score (initial_runs : ℕ) (target_runs : ℕ)
  (initial_wickets : ℕ) :
  initial_runs = 32 → target_runs = 282 → initial_wickets = 3 →
  remaining_runs_needed initial_runs target_runs = 250 :=
by
  sorry

end cricket_target_run_rate_cricket_wicket_partnership_score_l108_108687


namespace least_positive_base_ten_number_with_seven_binary_digits_l108_108749

theorem least_positive_base_ten_number_with_seven_binary_digits :
  ∃ n : ℕ, (n > 0) ∧ (n < 2^7) ∧ (n >= 2^6) ∧ (nat.binary_length n = 7) ∧ n = 64 :=
begin
  sorry
end

end least_positive_base_ten_number_with_seven_binary_digits_l108_108749


namespace largest_integer_satisfying_inequality_l108_108127

theorem largest_integer_satisfying_inequality :
  ∃ x : ℤ, (6 * x - 5 < 3 * x + 4) ∧ (∀ y : ℤ, (6 * y - 5 < 3 * y + 4) → y ≤ x) ∧ x = 2 :=
by
  sorry

end largest_integer_satisfying_inequality_l108_108127


namespace no_integers_with_cube_sum_l108_108646

theorem no_integers_with_cube_sum (a b : ℤ) (h1 : a^3 + b^3 = 4099) (h2 : Prime 4099) : false :=
sorry

end no_integers_with_cube_sum_l108_108646


namespace larger_number_l108_108453

variables (x y : ℕ)

theorem larger_number (h1 : x + y = 47) (h2 : x - y = 7) : x = 27 :=
sorry

end larger_number_l108_108453


namespace find_f1_l108_108432

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f1
  (h1 : ∀ x : ℝ, |f x - x^2| ≤ 1/4)
  (h2 : ∀ x : ℝ, |f x + 1 - x^2| ≤ 3/4) :
  f 1 = 3/4 := 
sorry

end find_f1_l108_108432


namespace triangle_perimeter_range_expression_l108_108020

-- Part 1: Prove the perimeter of △ABC
theorem triangle_perimeter (a b c : ℝ) (cosB : ℝ) (area : ℝ)
  (h1 : b^2 = a * c) (h2 : cosB = 3 / 5) (h3 : area = 2) :
  a + b + c = Real.sqrt 5 + Real.sqrt 21 :=
sorry

-- Part 2: Prove the range for the given expression
theorem range_expression (a b c : ℝ) (q : ℝ)
  (h1 : b = a * q) (h2 : c = a * q^2) :
  (Real.sqrt 5 - 1) / 2 < q ∧ q < (Real.sqrt 5 + 1) / 2 :=
sorry

end triangle_perimeter_range_expression_l108_108020


namespace kids_played_on_monday_l108_108548

theorem kids_played_on_monday (m t a : Nat) (h1 : t = 7) (h2 : a = 19) (h3 : a = m + t) : m = 12 := 
by 
  sorry

end kids_played_on_monday_l108_108548


namespace factor_t_sq_minus_64_l108_108941

def isDifferenceOfSquares (a b : Int) : Prop := a = b^2

theorem factor_t_sq_minus_64 (t : Int) (h : isDifferenceOfSquares 64 8) : (t^2 - 64) = (t - 8) * (t + 8) := by
  sorry

end factor_t_sq_minus_64_l108_108941


namespace range_of_k_l108_108519

theorem range_of_k (k : ℝ) (a : ℕ+ → ℝ) (h : ∀ n : ℕ+, a n = 2 * (n:ℕ)^2 + k * (n:ℕ)) 
  (increasing : ∀ n : ℕ+, a n < a (n + 1)) : 
  k > -6 := 
by 
  sorry

end range_of_k_l108_108519


namespace fill_pool_time_l108_108692

theorem fill_pool_time 
  (pool_volume : ℕ) (num_hoses : ℕ) (flow_rate_per_hose : ℕ)
  (H_pool_volume : pool_volume = 36000)
  (H_num_hoses : num_hoses = 6)
  (H_flow_rate_per_hose : flow_rate_per_hose = 3) :
  (pool_volume : ℚ) / (num_hoses * flow_rate_per_hose * 60) = 100 / 3 :=
by sorry

end fill_pool_time_l108_108692


namespace interest_rate_is_five_percent_l108_108805

-- Define the problem parameters
def principal : ℝ := 1200
def amount_after_period : ℝ := 1344
def time_period : ℝ := 2.4

-- Define the simple interest formula
def interest (P R T : ℝ) : ℝ := P * R * T

-- The goal is to prove that the rate of interest is 5% per year
theorem interest_rate_is_five_percent :
  ∃ R, interest principal R time_period = amount_after_period - principal ∧ R = 0.05 :=
by
  sorry

end interest_rate_is_five_percent_l108_108805


namespace determine_n_l108_108638

theorem determine_n (n : ℕ) : (2 : ℕ)^n = 2 * 4^2 * 16^3 ↔ n = 17 := 
by
  sorry

end determine_n_l108_108638


namespace moles_NaClO4_formed_l108_108510

-- Condition: Balanced chemical reaction
def reaction : Prop := ∀ (NaOH HClO4 NaClO4 H2O : ℕ), NaOH + HClO4 = NaClO4 + H2O

-- Given: 3 moles of NaOH and 3 moles of HClO4
def initial_moles_NaOH : ℕ := 3
def initial_moles_HClO4 : ℕ := 3

-- Question: number of moles of NaClO4 formed
def final_moles_NaClO4 : ℕ := 3

-- Proof Problem: Given the balanced chemical reaction and initial moles, prove the final moles of NaClO4
theorem moles_NaClO4_formed : reaction → initial_moles_NaOH = 3 → initial_moles_HClO4 = 3 → final_moles_NaClO4 = 3 :=
by
  intros
  sorry

end moles_NaClO4_formed_l108_108510


namespace jerry_needed_tonight_l108_108055

def jerry_earnings :=
  [20, 60, 15, 40]

def target_avg : ℕ := 50
def nights : ℕ := 5

def target_total_earnings := nights * target_avg
def actual_earnings_so_far := jerry_earnings.sum

theorem jerry_needed_tonight : (target_total_earnings - actual_earnings_so_far) = 115 :=
by
  have target_total : target_total_earnings = 250 := by rfl
  have actual_so_far : actual_earnings_so_far = 135 := by rfl
  rw [target_total, actual_so_far]
  norm_num
  -- 250 - 135 = 115
  rfl

end jerry_needed_tonight_l108_108055


namespace base5_number_l108_108759

/-- A base-5 number only contains the digits 0, 1, 2, 3, and 4.
    Given the number 21340, we need to prove that it could possibly be a base-5 number. -/
theorem base5_number (n : ℕ) (h : n = 21340) : 
  ∀ d ∈ [2, 1, 3, 4, 0], d < 5 :=
by sorry

end base5_number_l108_108759


namespace max_value_of_ab_l108_108553

theorem max_value_of_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 5 * a + 3 * b < 90) :
  ab * (90 - 5 * a - 3 * b) ≤ 1800 :=
sorry

end max_value_of_ab_l108_108553


namespace distance_to_point_is_17_l108_108187

def distance_from_origin (x y : ℝ) : ℝ :=
  Real.sqrt ((x - 0)^2 + (y - 0)^2)

theorem distance_to_point_is_17 :
  distance_from_origin 8 (-15) = 17 :=
by
  sorry

end distance_to_point_is_17_l108_108187


namespace intersection_of_A_and_B_l108_108672

def A := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 1) / Real.log 2}
def B := {x : ℝ | x < 2}

theorem intersection_of_A_and_B : (A ∩ B) = {x : ℝ | 1 < x ∧ x < 2} := by
  sorry

end intersection_of_A_and_B_l108_108672


namespace time_after_12345_seconds_is_13_45_45_l108_108688

def seconds_in_a_minute := 60
def minutes_in_an_hour := 60
def initial_hour := 10
def initial_minute := 45
def initial_second := 0
def total_seconds := 12345

def time_after_seconds (hour minute second : Nat) (elapsed_seconds : Nat) : (Nat × Nat × Nat) :=
  let total_initial_seconds := hour * 3600 + minute * 60 + second
  let total_final_seconds := total_initial_seconds + elapsed_seconds
  let final_hour := total_final_seconds / 3600
  let remaining_seconds_after_hour := total_final_seconds % 3600
  let final_minute := remaining_seconds_after_hour / 60
  let final_second := remaining_seconds_after_hour % 60
  (final_hour, final_minute, final_second)

theorem time_after_12345_seconds_is_13_45_45 :
  time_after_seconds initial_hour initial_minute initial_second total_seconds = (13, 45, 45) :=
by
  sorry

end time_after_12345_seconds_is_13_45_45_l108_108688


namespace files_more_than_apps_l108_108801

-- Defining the initial conditions
def initial_apps : ℕ := 11
def initial_files : ℕ := 3

-- Defining the conditions after some changes
def apps_left : ℕ := 2
def files_left : ℕ := 24

-- Statement to prove
theorem files_more_than_apps : (files_left - apps_left) = 22 := 
by
  sorry

end files_more_than_apps_l108_108801


namespace problem_l108_108172

variable (x : ℝ) (Q : ℝ)

theorem problem (h : 2 * (5 * x + 3 * Real.pi) = Q) : 4 * (10 * x + 6 * Real.pi + 2) = 4 * Q + 8 :=
by
  sorry

end problem_l108_108172


namespace evaluate_expression_l108_108962

theorem evaluate_expression (x y : ℕ) (hx : x = 4) (hy : y = 5) :
  (1 / (y : ℚ) / (1 / (x : ℚ)) + 2) = 14 / 5 :=
by
  rw [hx, hy]
  simp
  sorry

end evaluate_expression_l108_108962


namespace factor_diff_of_squares_l108_108928

theorem factor_diff_of_squares (t : ℝ) : 
  t^2 - 64 = (t - 8) * (t + 8) :=
by
  -- The purpose here is to state the theorem only, without the proof.
  sorry

end factor_diff_of_squares_l108_108928


namespace find_tan_α_l108_108513

variable (α : ℝ) (h1 : Real.sin (α - Real.pi / 3) = 3 / 5)
variable (h2 : Real.pi / 4 < α ∧ α < Real.pi / 2)

theorem find_tan_α (h1 : Real.sin (α - Real.pi / 3) = 3 / 5) (h2 : Real.pi / 4 < α ∧ α < Real.pi / 2) : 
  Real.tan α = - (48 + 25 * Real.sqrt 3) / 11 :=
sorry

end find_tan_α_l108_108513


namespace largest_C_inequality_l108_108361

theorem largest_C_inequality :
  ∃ C : ℝ, C = Real.sqrt (8 / 3) ∧ ∀ x y z : ℝ, x^2 + y^2 + z^2 + 2 ≥ C * (x + y + z) :=
by
  sorry

end largest_C_inequality_l108_108361


namespace abs_sqrt3_minus_1_sub_2_cos30_eq_neg1_l108_108796

theorem abs_sqrt3_minus_1_sub_2_cos30_eq_neg1 :
  |(Real.sqrt 3) - 1| - 2 * Real.cos (Real.pi / 6) = -1 := by
  sorry

end abs_sqrt3_minus_1_sub_2_cos30_eq_neg1_l108_108796


namespace sqrt_mul_sqrt_l108_108242

theorem sqrt_mul_sqrt (h1 : Real.sqrt 25 = 5) : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_mul_sqrt_l108_108242


namespace unique_solution_l108_108357

def one_digit_divisors := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def four_digit_numbers := Finset.Icc 1000 9999

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def one_digit_divisor_count (n : ℕ) : Prop := 
  (Finset.filter (λ d, d ∈ one_digit_divisors ∧ n % d = 0) one_digit_divisors).card = 9

def four_digit_divisor_count (n : ℕ) : Prop := 
  (Finset.filter (λ d, is_four_digit d ∧ n % d = 0) four_digit_numbers).card = 5

theorem unique_solution :
  ∃! (n : ℕ), is_four_digit n ∧ one_digit_divisor_count n ∧ four_digit_divisor_count n :=
begin
  use 5040,
  split,
  { unfold is_four_digit one_digit_divisor_count four_digit_divisor_count,
    split,
    exact ⟨by norm_num, by norm_num⟩,
    split,
    { sorry },  -- Proof that 5040 has exactly 9 one-digit divisors
    { sorry }   -- Proof that 5040 has exactly 5 four-digit divisors
  },
  { intros y hy,
    have h₁ : is_four_digit y := hy.1,
    have h₂ : one_digit_divisor_count y := hy.2.1,
    have h₃ : four_digit_divisor_count y := hy.2.2,
    sorry,  -- Proof that any other number satisfying these conditions must be 5040
  }
end

end unique_solution_l108_108357


namespace least_binary_seven_digits_l108_108746

theorem least_binary_seven_digits : (n : ℕ) → (dig : ℕ) 
  (h : bit_length n = 7) : n = 64 := 
begin
  assume n dig h,
  sorry
end

end least_binary_seven_digits_l108_108746


namespace solution_ne_zero_l108_108734

theorem solution_ne_zero (a x : ℝ) (h : x = a * x + 1) : x ≠ 0 := sorry

end solution_ne_zero_l108_108734


namespace joes_current_weight_l108_108693

theorem joes_current_weight (W : ℕ) (R : ℕ) : 
  (W = 222 - 4 * R) →
  (W - 3 * R = 180) →
  W = 198 :=
by
  intros h1 h2
  -- Skip the proof for now
  sorry

end joes_current_weight_l108_108693


namespace grade_distribution_sum_l108_108176

theorem grade_distribution_sum (a b c d : ℝ) (ha : a = 0.6) (hb : b = 0.25) (hc : c = 0.1) (hd : d = 0.05) :
  a + b + c + d = 1.0 :=
by
  -- Introduce the hypothesis
  rw [ha, hb, hc, hd]
  -- Now the goal simplifies to: 0.6 + 0.25 + 0.1 + 0.05 = 1.0
  sorry

end grade_distribution_sum_l108_108176


namespace safe_travel_exists_l108_108589

def total_travel_time : ℕ := 16
def first_crater_cycle : ℕ := 18
def first_crater_duration : ℕ := 1
def second_crater_cycle : ℕ := 10
def second_crater_duration : ℕ := 1

theorem safe_travel_exists : 
  ∃ t : ℕ, t ∈ { t | (∀ k : ℕ, t % first_crater_cycle ≠ k ∨ t % first_crater_cycle ≥ first_crater_duration) 
  ∧ (∀ k : ℕ, t % second_crater_cycle ≠ k ∨ t % second_crater_cycle ≥ second_crater_duration) 
  ∧ (∀ k : ℕ, (t + total_travel_time) % first_crater_cycle ≠ k ∨ (t + total_travel_time) % first_crater_cycle ≥ first_crater_duration) 
  ∧ (∀ k : ℕ, (t + total_travel_time) % second_crater_cycle ≠ k ∨ (t + total_travel_time) % second_crater_cycle ≥ second_crater_duration) } :=
sorry

end safe_travel_exists_l108_108589


namespace sqrt_mul_sqrt_l108_108257

theorem sqrt_mul_sqrt (a b c : ℝ) (hac : a = 49) (hbc : b = sqrt 25) (hc : c = 7 * sqrt 5) : sqrt (a * b) = c := 
by
  sorry

end sqrt_mul_sqrt_l108_108257


namespace sqrt_nested_l108_108303

theorem sqrt_nested : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_nested_l108_108303


namespace weekly_earnings_l108_108405

-- Definition of the conditions
def hourly_rate : ℕ := 20
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 4

-- Theorem that conforms to the problem statement
theorem weekly_earnings : hourly_rate * hours_per_day * days_per_week = 640 := by
  sorry

end weekly_earnings_l108_108405


namespace frequency_of_group5_l108_108621

-- Define the total number of students and the frequencies of each group
def total_students : ℕ := 40
def freq_group1 : ℕ := 12
def freq_group2 : ℕ := 10
def freq_group3 : ℕ := 6
def freq_group4 : ℕ := 8

-- Define the frequency of the fifth group in terms of the above frequencies
def freq_group5 : ℕ := total_students - (freq_group1 + freq_group2 + freq_group3 + freq_group4)

-- The theorem to be proven
theorem frequency_of_group5 : freq_group5 = 4 := by
  -- Proof goes here, skipped with sorry
  sorry

end frequency_of_group5_l108_108621


namespace pyramid_edge_length_correct_l108_108443

-- Definitions for the conditions
def total_length (sum_of_edges : ℝ) := sum_of_edges = 14.8
def edges_count (num_of_edges : ℕ) := num_of_edges = 8

-- Definition for the question and corresponding answer to prove
def length_of_one_edge (sum_of_edges : ℝ) (num_of_edges : ℕ) (one_edge_length : ℝ) :=
  sum_of_edges / num_of_edges = one_edge_length

-- The statement that needs to be proven
theorem pyramid_edge_length_correct : total_length 14.8 → edges_count 8 → length_of_one_edge 14.8 8 1.85 :=
by
  intros h1 h2
  sorry

end pyramid_edge_length_correct_l108_108443


namespace point_of_tangency_l108_108828

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)
noncomputable def f_deriv (x a : ℝ) : ℝ := Real.exp x - a * Real.exp (-x)

theorem point_of_tangency (a : ℝ) (h1 : ∀ x, f_deriv (-x) a = -f_deriv x a)
  (h2 : ∃ x0, f_deriv x0 1 = 3/2) :
  ∃ x0 y0, x0 = Real.log 2 ∧ y0 = f (Real.log 2) 1 ∧ y0 = 5/2 :=
by
  sorry

end point_of_tangency_l108_108828


namespace compound_oxygen_atoms_l108_108317

theorem compound_oxygen_atoms (H C O : Nat) (mw : Nat) (H_weight C_weight O_weight : Nat) 
  (h_H : H = 2)
  (h_C : C = 1)
  (h_mw : mw = 62)
  (h_H_weight : H_weight = 1)
  (h_C_weight : C_weight = 12)
  (h_O_weight : O_weight = 16)
  : O = 3 :=
by
  sorry

end compound_oxygen_atoms_l108_108317


namespace find_N_mod_inverse_l108_108855

-- Definitions based on given conditions
def A := 111112
def B := 142858
def M := 1000003
def AB : Nat := (A * B) % M
def N := 513487

-- Statement to prove
theorem find_N_mod_inverse : (711812 * N) % M = 1 := by
  -- Proof skipped as per instruction
  sorry

end find_N_mod_inverse_l108_108855


namespace section_b_students_can_be_any_nonnegative_integer_l108_108881

def section_a_students := 36
def avg_weight_section_a := 30
def avg_weight_section_b := 30
def avg_weight_whole_class := 30

theorem section_b_students_can_be_any_nonnegative_integer (x : ℕ) :
  let total_weight_section_a := section_a_students * avg_weight_section_a
  let total_weight_section_b := x * avg_weight_section_b
  let total_weight_whole_class := (section_a_students + x) * avg_weight_whole_class
  (total_weight_section_a + total_weight_section_b = total_weight_whole_class) :=
by 
  sorry

end section_b_students_can_be_any_nonnegative_integer_l108_108881


namespace janessa_kept_20_cards_l108_108848

-- Definitions based on conditions
def initial_cards : Nat := 4
def father_cards : Nat := 13
def ebay_cards : Nat := 36
def bad_shape_cards : Nat := 4
def cards_given_to_dexter : Nat := 29

-- Prove that Janessa kept 20 cards for herself
theorem janessa_kept_20_cards :
  (initial_cards + father_cards  + ebay_cards - bad_shape_cards) - cards_given_to_dexter = 20 :=
by
  sorry

end janessa_kept_20_cards_l108_108848


namespace no_valid_pairs_l108_108833

theorem no_valid_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ¬ (a * b + 82 = 25 * Nat.lcm a b + 15 * Nat.gcd a b) :=
by {
  sorry
}

end no_valid_pairs_l108_108833


namespace remainder_of_x_l108_108776

theorem remainder_of_x (x : ℤ) (h : 2 * x - 3 = 7) : x % 2 = 1 := by
  sorry

end remainder_of_x_l108_108776


namespace major_axis_length_is_three_l108_108324

-- Given the radius of the cylinder
def cylinder_radius : ℝ := 1

-- Given the percentage longer of the major axis than the minor axis
def percentage_longer (r : ℝ) : ℝ := 1.5

-- Given the function to calculate the minor axis using the radius
def minor_axis (r : ℝ) : ℝ := 2 * r

-- Given the function to calculate the major axis using the minor axis
def major_axis (minor_axis : ℝ) (factor : ℝ) : ℝ := minor_axis * factor

-- The conjecture states that the major axis length is 3
theorem major_axis_length_is_three : 
  major_axis (minor_axis cylinder_radius) (percentage_longer cylinder_radius) = 3 :=
by 
  -- Proof goes here
  sorry

end major_axis_length_is_three_l108_108324


namespace solve_ratios_l108_108218

theorem solve_ratios (q m n : ℕ) (h1 : 7 / 9 = n / 108) (h2 : 7 / 9 = (m + n) / 126) (h3 : 7 / 9 = (q - m) / 162) : q = 140 :=
by
  sorry

end solve_ratios_l108_108218


namespace total_area_at_stage_4_l108_108834

/-- Define the side length of the square at a given stage -/
def side_length (n : ℕ) : ℕ := n + 2

/-- Define the area of the square at a given stage -/
def area (n : ℕ) : ℕ := (side_length n) ^ 2

/-- State the theorem -/
theorem total_area_at_stage_4 : 
  (area 0) + (area 1) + (area 2) + (area 3) = 86 :=
by
  -- proof goes here
  sorry

end total_area_at_stage_4_l108_108834


namespace pascal_triangle_10_to_30_l108_108170

-- Definitions
def pascal_row_numbers (n : ℕ) : ℕ := n + 1

def total_numbers_up_to (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

-- Proof Statement
theorem pascal_triangle_10_to_30 :
  total_numbers_up_to 29 - total_numbers_up_to 9 = 400 := by
  sorry

end pascal_triangle_10_to_30_l108_108170


namespace tank_capacity_l108_108319

theorem tank_capacity
  (x : ℝ) -- define x as the full capacity of the tank in gallons
  (h1 : (5/6) * x - (2/3) * x = 15) -- first condition
  (h2 : (2/3) * x = y) -- second condition, though not actually needed
  : x = 90 := 
by sorry

end tank_capacity_l108_108319


namespace no_such_integers_exists_l108_108427

theorem no_such_integers_exists 
  (a b c d : ℤ) 
  (h1 : a * 19^3 + b * 19^2 + c * 19 + d = 1) 
  (h2 : a * 62^3 + b * 62^2 + c * 62 + d = 2) : 
  false :=
by
  sorry

end no_such_integers_exists_l108_108427


namespace how_many_roses_cut_l108_108451

theorem how_many_roses_cut :
  ∀ (r_i r_f r_c : ℕ), r_i = 6 → r_f = 16 → r_c = r_f - r_i → r_c = 10 :=
by
  intros r_i r_f r_c hri hrf heq
  rw [hri, hrf] at heq
  exact heq

end how_many_roses_cut_l108_108451


namespace albums_not_in_both_l108_108118

-- Definitions representing the problem conditions
def andrew_albums : ℕ := 23
def common_albums : ℕ := 11
def john_unique_albums : ℕ := 8

-- Proof statement (not the actual proof)
theorem albums_not_in_both : 
  (andrew_albums - common_albums) + john_unique_albums = 20 :=
by
  sorry

end albums_not_in_both_l108_108118


namespace inequality_and_equality_condition_l108_108552

variable {x y : ℝ}

theorem inequality_and_equality_condition
  (hx : 0 < x) (hy : 0 < y) :
  (x + y^2 / x ≥ 2 * y) ∧ (x + y^2 / x = 2 * y ↔ x = y) := sorry

end inequality_and_equality_condition_l108_108552


namespace four_digit_number_l108_108006

theorem four_digit_number (a b c d : ℕ)
    (h1 : 0 ≤ a) (h2 : a ≤ 9)
    (h3 : 0 ≤ b) (h4 : b ≤ 9)
    (h5 : 0 ≤ c) (h6 : c ≤ 9)
    (h7 : 0 ≤ d) (h8 : d ≤ 9)
    (h9 : 2 * (1000 * a + 100 * b + 10 * c + d) + 1000 = 1000 * d + 100 * c + 10 * b + a)
    : (1000 * a + 100 * b + 10 * c + d) = 2996 :=
by
  sorry

end four_digit_number_l108_108006


namespace ratio_a_c_l108_108393

theorem ratio_a_c (a b c : ℕ) (h1 : a / b = 5 / 3) (h2 : b / c = 1 / 5) : a / c = 1 / 3 :=
sorry

end ratio_a_c_l108_108393


namespace significant_improvement_l108_108604

-- Definition of experiment data
def experiment_data (x y : Fin 10 → ℝ) : Prop :=
  x = ![545, 533, 551, 522, 575, 544, 541, 568, 596, 548] ∧
  y = ![536, 527, 543, 530, 560, 533, 522, 550, 576, 536]

-- Definition of z_i
def z (x y : Fin 10 → ℝ) (i : Fin 10) : ℝ := x i - y i

-- Sample mean of z
def mean_z (z : Fin 10 → ℝ) : ℝ := (1 / 10) * ∑ i, z i

-- Sample variance of z
def variance_z (z : Fin 10 → ℝ) : ℝ := (1 / 10) * ∑ i, (z i - mean_z z)^2

-- Proof problem to check significant improvement
theorem significant_improvement (x y : Fin 10 → ℝ)
  (h_data : experiment_data x y) :
  mean_z (z x y) ≥ 2 * (Real.sqrt (variance_z (z x y) / 10)) :=
by
  sorry

end significant_improvement_l108_108604


namespace caleb_spent_more_on_ice_cream_l108_108920

theorem caleb_spent_more_on_ice_cream :
  ∀ (number_of_ic_cartons number_of_fy_cartons : ℕ)
    (cost_per_ic_carton cost_per_fy_carton : ℝ)
    (discount_rate sales_tax_rate : ℝ),
    number_of_ic_cartons = 10 →
    number_of_fy_cartons = 4 →
    cost_per_ic_carton = 4 →
    cost_per_fy_carton = 1 →
    discount_rate = 0.15 →
    sales_tax_rate = 0.05 →
    (number_of_ic_cartons * cost_per_ic_carton * (1 - discount_rate) + 
     (number_of_ic_cartons * cost_per_ic_carton * (1 - discount_rate) + 
      number_of_fy_cartons * cost_per_fy_carton) * sales_tax_rate) -
    (number_of_fy_cartons * cost_per_fy_carton) = 30 :=
by
  intros number_of_ic_cartons number_of_fy_cartons cost_per_ic_carton cost_per_fy_carton discount_rate sales_tax_rate
  sorry

end caleb_spent_more_on_ice_cream_l108_108920


namespace even_square_even_square_even_even_l108_108062

-- Definition for a natural number being even
def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

-- Statement 1: If p is even, then p^2 is even
theorem even_square_even (p : ℕ) (hp : is_even p) : is_even (p * p) :=
sorry

-- Statement 2: If p^2 is even, then p is even
theorem square_even_even (p : ℕ) (hp_squared : is_even (p * p)) : is_even p :=
sorry

end even_square_even_square_even_even_l108_108062


namespace relationship_between_Q_and_t_remaining_power_after_5_hours_distance_with_40_power_l108_108590

-- Define the relationship between Q and t
def remaining_power (t : ℕ) : ℕ := 80 - 15 * t

-- Question 1: Prove relationship between Q and t
theorem relationship_between_Q_and_t : ∀ t : ℕ, remaining_power t = 80 - 15 * t :=
by sorry

-- Question 2: Prove remaining power after 5 hours
theorem remaining_power_after_5_hours : remaining_power 5 = 5 :=
by sorry

-- Question 3: Prove distance the car can travel with 40 kW·h remaining power
theorem distance_with_40_power 
  (remaining_power : ℕ := (80 - 15 * t)) 
  (t := 8 / 3)
  (speed : ℕ := 90) : (90 * (8 / 3)) = 240 :=
by sorry

end relationship_between_Q_and_t_remaining_power_after_5_hours_distance_with_40_power_l108_108590


namespace find_f_pi_six_value_l108_108376

noncomputable def f (x : ℝ) (f'₀ : ℝ) : ℝ := f'₀ * Real.sin x + Real.cos x

theorem find_f_pi_six_value (f'₀ : ℝ) (h : f'₀ = 2 + Real.sqrt 3) : f (π / 6) f'₀ = 1 + Real.sqrt 3 := 
by
  -- condition from the problem
  let f₀ := f (π / 6) f'₀
  -- final goal to prove
  sorry

end find_f_pi_six_value_l108_108376


namespace W_555_2_last_three_digits_l108_108474

noncomputable def W : ℕ → ℕ → ℕ
| n, 0     => n ^ n
| n, (k+1) => W (W n k) k

theorem W_555_2_last_three_digits :
  (W 555 2) % 1000 = 875 :=
sorry

end W_555_2_last_three_digits_l108_108474


namespace quadratic_inequality_solution_set_l108_108429

theorem quadratic_inequality_solution_set :
  (∃ x : ℝ, 2 * x + 3 - x^2 > 0) ↔ (-1 < x ∧ x < 3) :=
sorry

end quadratic_inequality_solution_set_l108_108429


namespace C_paisa_for_A_rupee_l108_108771

variable (A B C : ℝ)
variable (C_share : ℝ) (total_sum : ℝ)
variable (B_per_A : ℝ)

noncomputable def C_paisa_per_A_rupee (A B C C_share total_sum B_per_A : ℝ) : ℝ :=
  let C_paisa := C_share * 100
  C_paisa / A

theorem C_paisa_for_A_rupee : C_share = 32 ∧ total_sum = 164 ∧ B_per_A = 0.65 → 
  C_paisa_per_A_rupee A B C C_share total_sum B_per_A = 40 := by
  sorry

end C_paisa_for_A_rupee_l108_108771


namespace sqrt_product_l108_108272

theorem sqrt_product (a b : ℝ) (h1 : a = 49) (h2 : b = 25) : sqrt (a * sqrt b) = 7 * sqrt 5 :=
by sorry

end sqrt_product_l108_108272


namespace ratio_b_a_4_l108_108532

theorem ratio_b_a_4 (a b : ℚ) (h1 : b / a = 4) (h2 : b = 15 - 6 * a) : a = 3 / 2 :=
by
  sorry

end ratio_b_a_4_l108_108532


namespace torn_pages_count_l108_108681

theorem torn_pages_count (pages : Finset ℕ) (h1 : ∀ p ∈ pages, 1 ≤ p ∧ p ≤ 100) (h2 : pages.sum id = 4949) : 
  100 - pages.card = 3 := 
by
  sorry

end torn_pages_count_l108_108681


namespace locus_of_P_is_single_ray_l108_108654
  
noncomputable def M : ℝ × ℝ := (1, 0)
noncomputable def N : ℝ × ℝ := (3, 0)

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2))

theorem locus_of_P_is_single_ray (P : ℝ × ℝ) (h : distance P M - distance P N = 2) : 
∃ α : ℝ, P = (3 + α * (P.1 - 3), α * P.2) :=
sorry

end locus_of_P_is_single_ray_l108_108654


namespace value_of_sum_l108_108078

theorem value_of_sum (a b c d : ℤ) 
  (h1 : a - b + c = 7)
  (h2 : b - c + d = 8)
  (h3 : c - d + a = 5)
  (h4 : d - a + b = 4) : a + b + c + d = 12 := 
  sorry

end value_of_sum_l108_108078


namespace vector_add_sub_l108_108797

open Matrix

section VectorProof

/-- Define the vectors a, b, and c. -/
def a : Matrix (Fin 2) (Fin 1) ℤ := ![![3], ![-6]]
def b : Matrix (Fin 2) (Fin 1) ℤ := ![![-1], ![5]]
def c : Matrix (Fin 2) (Fin 1) ℤ := ![![5], ![-20]]

/-- State the proof problem. -/
theorem vector_add_sub :
  2 • a + 4 • b - c = ![![-3], ![28]] :=
by
  sorry

end VectorProof

end vector_add_sub_l108_108797


namespace money_problem_solution_l108_108711

theorem money_problem_solution (a b : ℝ) (h1 : 7 * a + b < 100) (h2 : 4 * a - b = 40) (h3 : b = 0.5 * a) : 
  a = 80 / 7 ∧ b = 40 / 7 :=
by
  sorry

end money_problem_solution_l108_108711


namespace multiplication_problem_division_problem_l108_108123

theorem multiplication_problem :
  125 * 76 * 4 * 8 * 25 = 7600000 :=
sorry

theorem division_problem :
  (6742 + 6743 + 6738 + 6739 + 6741 + 6743) / 6 = 6741 :=
sorry

end multiplication_problem_division_problem_l108_108123


namespace two_a_minus_five_d_eq_zero_l108_108417

variables {α : Type*} [Field α]

def f (a b c d x : α) : α :=
  (2*a*x + 3*b) / (4*c*x - 5*d)

theorem two_a_minus_five_d_eq_zero
  (a b c d : α) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (hf : ∀ x, f a b c d (f a b c d x) = x) :
  2*a - 5*d = 0 :=
sorry

end two_a_minus_five_d_eq_zero_l108_108417


namespace desired_average_l108_108333

theorem desired_average (P1 P2 P3 : ℝ) (A : ℝ) 
  (hP1 : P1 = 74) 
  (hP2 : P2 = 84) 
  (hP3 : P3 = 67) 
  (hA : A = (P1 + P2 + P3) / 3) : 
  A = 75 :=
  sorry

end desired_average_l108_108333


namespace kenneth_distance_past_finish_l108_108625

noncomputable def distance_past_finish_line (race_distance : ℕ) (biff_speed : ℕ) (kenneth_speed : ℕ) : ℕ :=
  let biff_time := race_distance / biff_speed
  let kenneth_distance := kenneth_speed * biff_time
  kenneth_distance - race_distance

theorem kenneth_distance_past_finish (race_distance : ℕ) (biff_speed : ℕ) (kenneth_speed : ℕ) (finish_line_distance : ℕ) : 
  race_distance = 500 ->
  biff_speed = 50 -> 
  kenneth_speed = 51 ->
  finish_line_distance = 10 ->
  distance_past_finish_line race_distance biff_speed kenneth_speed = finish_line_distance := by
  sorry

end kenneth_distance_past_finish_l108_108625


namespace exists_f_with_f3_eq_9_forall_f_f3_le_9_l108_108351

-- Define the real-valued function f satisfying the given conditions
variable (f : ℝ → ℝ)
variable (f_real : ∀ x : ℝ, true)  -- f is real-valued and defined for all real numbers
variable (f_mul : ∀ x y : ℝ, f (x * y) = f x * f y)  -- f(xy) = f(x)f(y)
variable (f_add : ∀ x y : ℝ, f (x + y) ≤ 2 * (f x + f y))  -- f(x+y) ≤ 2(f(x) + f(y))
variable (f_2 : f 2 = 4)  -- f(2) = 4

-- Part a
theorem exists_f_with_f3_eq_9 : ∃ f : ℝ → ℝ, (∀ x : ℝ, true) ∧ 
                              (∀ x y : ℝ, f (x * y) = f x * f y) ∧ 
                              (∀ x y : ℝ, f (x + y) ≤ 2 * (f x + f y)) ∧ 
                              (f 2 = 4) ∧ 
                              (f 3 = 9) := 
sorry

-- Part b
theorem forall_f_f3_le_9 : ∀ f : ℝ → ℝ, 
                        (∀ x : ℝ, true) → 
                        (∀ x y : ℝ, f (x * y) = f x * f y) → 
                        (∀ x y : ℝ, f (x + y) ≤ 2 * (f x + f y)) → 
                        (f 2 = 4) → 
                        (f 3 ≤ 9) := 
sorry

end exists_f_with_f3_eq_9_forall_f_f3_le_9_l108_108351


namespace ten_times_six_x_plus_fourteen_pi_l108_108528

theorem ten_times_six_x_plus_fourteen_pi (x : ℝ) (Q : ℝ) (h : 5 * (3 * x + 7 * Real.pi) = Q) : 
  10 * (6 * x + 14 * Real.pi) = 4 * Q :=
by
  sorry

end ten_times_six_x_plus_fourteen_pi_l108_108528


namespace george_older_than_christopher_l108_108816

theorem george_older_than_christopher
  (G C F : ℕ)
  (h1 : C = 18)
  (h2 : F = C - 2)
  (h3 : G + C + F = 60) :
  G - C = 8 := by
  sorry

end george_older_than_christopher_l108_108816


namespace original_price_of_second_pair_l108_108691

variable (P : ℝ) -- original price of the second pair of shoes
variable (discounted_price : ℝ := P / 2)
variable (total_before_discount : ℝ := 40 + discounted_price)
variable (final_payment : ℝ := (3 / 4) * total_before_discount)
variable (payment : ℝ := 60)

theorem original_price_of_second_pair (h : final_payment = payment) : P = 80 :=
by
  -- Skipping the proof with sorry.
  sorry

end original_price_of_second_pair_l108_108691


namespace no_common_period_l108_108473

theorem no_common_period (g h : ℝ → ℝ) 
  (hg : ∀ x, g (x + 2) = g x) 
  (hh : ∀ x, h (x + π/2) = h x) : 
  ¬ (∃ T > 0, ∀ x, g (x + T) + h (x + T) = g x + h x) :=
sorry

end no_common_period_l108_108473


namespace gloves_needed_l108_108226

theorem gloves_needed (participants : ℕ) (gloves_per_participant : ℕ) (total_gloves : ℕ)
  (h1 : participants = 82)
  (h2 : gloves_per_participant = 2)
  (h3 : total_gloves = participants * gloves_per_participant) :
  total_gloves = 164 :=
by
  sorry

end gloves_needed_l108_108226


namespace average_speed_l108_108480

theorem average_speed (v : ℝ) (h : 500 / v - 500 / (v + 10) = 2) : v = 45.25 :=
by
  sorry

end average_speed_l108_108480


namespace only_n_equal_1_is_natural_number_for_which_2_pow_n_plus_n_pow_2016_is_prime_l108_108944

theorem only_n_equal_1_is_natural_number_for_which_2_pow_n_plus_n_pow_2016_is_prime (n : ℕ) : 
  Prime (2^n + n^2016) ↔ n = 1 := by
  sorry

end only_n_equal_1_is_natural_number_for_which_2_pow_n_plus_n_pow_2016_is_prime_l108_108944


namespace no_solution_for_m_l108_108032

theorem no_solution_for_m (m : ℝ) : ¬ ∃ x : ℝ, x ≠ 1 ∧ (mx - 1) / (x - 1) = 3 ↔ m = 1 ∨ m = 3 := 
by sorry

end no_solution_for_m_l108_108032


namespace smallest_k_l108_108740

theorem smallest_k :
  ∃ k : ℕ, k > 1 ∧ (k % 19 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) ∧ k = 400 :=
by
  sorry

end smallest_k_l108_108740


namespace four_letter_words_with_A_at_least_once_l108_108382

theorem four_letter_words_with_A_at_least_once (A B C D E : Type) :
  let total := 5^4 in
  let without_A := 4^4 in
  total - without_A = 369 :=
by {
  let total := 5^4;
  let without_A := 4^4;
  have : total - without_A = 369 := by sorry;
  exact this;
}

end four_letter_words_with_A_at_least_once_l108_108382


namespace cost_of_each_pant_l108_108847

theorem cost_of_each_pant (shirts pants : ℕ) (cost_shirt cost_total : ℕ) (cost_pant : ℕ) :
  shirts = 10 ∧ pants = (shirts / 2) ∧ cost_shirt = 6 ∧ cost_total = 100 →
  (shirts * cost_shirt + pants * cost_pant = cost_total) →
  cost_pant = 8 :=
by
  sorry

end cost_of_each_pant_l108_108847


namespace beth_sold_coins_l108_108120

theorem beth_sold_coins :
  let initial_coins := 125
  let gift_coins := 35
  let total_coins := initial_coins + gift_coins
  let sold_coins := total_coins / 2
  sold_coins = 80 :=
by
  sorry

end beth_sold_coins_l108_108120


namespace find_b_of_expression_l108_108396

theorem find_b_of_expression (y : ℝ) (b : ℝ) (hy : y > 0)
  (h : (7 / 10) * y = (8 * y) / b + (3 * y) / 10) : b = 20 :=
sorry

end find_b_of_expression_l108_108396


namespace range_of_a_l108_108831

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (1 + a) * x^2 + a * x + a < x^2 + 1) : a ≤ 0 := 
sorry

end range_of_a_l108_108831


namespace sqrt_49_times_sqrt25_eq_7_sqrt5_l108_108300

-- Definitions based on conditions
def sqrt_25 := Real.sqrt 25
def sqrt_49 := Real.sqrt 49

theorem sqrt_49_times_sqrt25_eq_7_sqrt5 :
  Real.sqrt (49 * sqrt_25) = 7 * Real.sqrt 5 := by
sorry

end sqrt_49_times_sqrt25_eq_7_sqrt5_l108_108300


namespace sequence_problem_l108_108731

noncomputable def b_n (n : ℕ) : ℝ := 5 * (5/3)^(n-2)

theorem sequence_problem 
  (a_n : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : ∀ n, a_n (n + 1) = a_n n + d)
  (h2 : d ≠ 0)
  (h3 : a_n 8 = a_n 5 + 3 * d)
  (h4 : a_n 13 = a_n 8 + 5 * d)
  (b_2 : ℝ)
  (hb2 : b_2 = 5)
  (h5 : ∀ n, b_n n = (match n with | 2 => b_2 | _ => sorry))
  (conseq_terms : ∀ (n : ℕ), (a_n 5 + 3 * d)^2 = a_n 5 * (a_n 5 + 8 * d)) 
  : ∀ n, b_n n = b_n 2 * (5/3)^(n-2) := 
by 
  sorry

end sequence_problem_l108_108731


namespace subcommittees_count_l108_108384

theorem subcommittees_count 
  (n : ℕ) (k : ℕ) (hn : n = 7) (hk : k = 3) : 
  (nat.choose n k) = 35 := by 
  have h1 : 7 = 7 := rfl
  have h2 : 3 = 3 := rfl
  sorry

end subcommittees_count_l108_108384


namespace hypotenuse_of_45_45_90_triangle_l108_108569

theorem hypotenuse_of_45_45_90_triangle (leg : ℝ) (angle_opposite_leg : ℝ) (h_leg : leg = 15) (h_angle : angle_opposite_leg = 45) :
  ∃ hypotenuse, hypotenuse = leg * Real.sqrt 2 :=
by
  use leg * Real.sqrt 2
  rw [h_leg]
  rw [h_angle]
  sorry

end hypotenuse_of_45_45_90_triangle_l108_108569


namespace find_a_l108_108970

noncomputable def f (x : ℝ) : ℝ := x^2 + 9
noncomputable def g (x : ℝ) : ℝ := x^2 - 5

theorem find_a (a : ℝ) (h1 : a > 0) (h2 : f (g a) = 9) : a = Real.sqrt 5 :=
by
  sorry

end find_a_l108_108970


namespace find_n_l108_108439

theorem find_n :
  ∃ (n : ℤ), 50 ≤ n ∧ n ≤ 120 ∧ (n % 8 = 0) ∧ (n % 7 = 5) ∧ (n % 6 = 3) ∧ n = 208 := 
by {
  sorry
}

end find_n_l108_108439


namespace least_7_digit_binary_number_is_64_l108_108748

theorem least_7_digit_binary_number_is_64 : ∃ n : ℕ, n = 64 ∧ (∀ m : ℕ, (m < 64 ∧ m >= 64) → false) ∧ nat.log2 64 = 6 :=
by
  sorry

end least_7_digit_binary_number_is_64_l108_108748


namespace solve_for_k_l108_108985

theorem solve_for_k (x k : ℝ) (h : k ≠ 0) 
(h_eq : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 7)) : k = 7 :=
by
  -- Proof would go here
  sorry

end solve_for_k_l108_108985


namespace solve_system_l108_108861

theorem solve_system :
  ∃ x y : ℝ, x - y = 1 ∧ 3 * x + 2 * y = 8 ∧ x = 2 ∧ y = 1 := by
  sorry

end solve_system_l108_108861


namespace arithmetic_sequence_sum_l108_108014

theorem arithmetic_sequence_sum : 
  ∀ (a : ℕ → ℝ) (d : ℝ), (a 1 = 2 ∨ a 1 = 8) → (a 2017 = 2 ∨ a 2017 = 8) → 
  (∀ n : ℕ, a (n + 1) = a n + d) →
  a 2 + a 1009 + a 2016 = 15 := 
by
  intro a d h1 h2017 ha
  sorry

end arithmetic_sequence_sum_l108_108014


namespace average_marks_l108_108880

theorem average_marks (M P C : ℕ) (h1 : M + P = 60) (h2 : C = P + 10) : (M + C) / 2 = 35 := 
by
  sorry

end average_marks_l108_108880


namespace sum_of_four_primes_div_by_60_l108_108551

open Nat

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem sum_of_four_primes_div_by_60
  (p q r s : ℕ)
  (hp : is_prime p)
  (hq : is_prime q)
  (hr : is_prime r)
  (hs : is_prime s)
  (horder : 5 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < p + 10) :
  (p + q + r + s) % 60 = 0 :=
by
  sorry


end sum_of_four_primes_div_by_60_l108_108551


namespace portia_high_school_students_l108_108209

variables (P L M : ℕ)
axiom h1 : P = 4 * L
axiom h2 : P = 2 * M
axiom h3 : P + L + M = 4800

theorem portia_high_school_students : P = 2740 :=
by sorry

end portia_high_school_students_l108_108209


namespace part1_intersection_part2_range_of_m_l108_108414

-- Define the universal set and the sets A and B
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 0 ∨ x > 3}
def B (m : ℝ) : Set ℝ := {x | x < m - 1 ∨ x > 2 * m}

-- Part (1): When m = 3, find A ∩ B
theorem part1_intersection:
  A ∩ B 3 = {x | x < 0 ∨ x > 6} :=
sorry

-- Part (2): If B ∪ A = B, find the range of values for m
theorem part2_range_of_m (m : ℝ) :
  (B m ∪ A = B m) → (1 ≤ m ∧ m ≤ 3 / 2) :=
sorry

end part1_intersection_part2_range_of_m_l108_108414


namespace sqrt_49_times_sqrt_25_l108_108295

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 :=
by
  sorry

end sqrt_49_times_sqrt_25_l108_108295


namespace sqrt_49_mul_sqrt_25_l108_108280

theorem sqrt_49_mul_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_l108_108280


namespace part1_part2_l108_108476

section PartOne

variables (x y : ℕ)
def condition1 := x + y = 360
def condition2 := x - y = 110

theorem part1 (h1 : condition1 x y) (h2 : condition2 x y) : x = 235 ∧ y = 125 := by {
  sorry
}

end PartOne

section PartTwo

variables (t W : ℕ)
def tents_capacity (t : ℕ) := 40 * t + 20 * (9 - t)
def food_capacity (t : ℕ) := 10 * t + 20 * (9 - t)
def transportation_cost (t : ℕ) := 4000 * t + 3600 * (9 - t)

theorem part2 
  (htents : tents_capacity t ≥ 235) 
  (hfood : food_capacity t ≥ 125) : 
  W = transportation_cost t → t = 3 ∧ W = 33600 := by {
  sorry
}

end PartTwo

end part1_part2_l108_108476


namespace largest_integer_m_property_l108_108635

theorem largest_integer_m_property (M : ℝ) (hM : M > 1) :
  (∀ (s : Finset ℝ), s.card = 10 → (∀ (a b c : ℝ), a ∈ s → b ∈ s → c ∈ s → a < b → b < c → (a * b * c) < (b ^ 2))) ↔ (M ≤ 4 ^ 255) :=
sorry

end largest_integer_m_property_l108_108635


namespace find_distance_between_PQ_l108_108093

-- Defining distances and speeds
def distance_by_first_train (t : ℝ) : ℝ := 50 * t
def distance_by_second_train (t : ℝ) : ℝ := 40 * t
def distance_between_PQ (t : ℝ) : ℝ := distance_by_first_train t + (distance_by_first_train t - 100)

-- Main theorem stating the problem
theorem find_distance_between_PQ : ∃ t : ℝ, distance_by_first_train t - distance_by_second_train t = 100 ∧ distance_between_PQ t = 900 := 
sorry

end find_distance_between_PQ_l108_108093


namespace number_satisfying_condition_l108_108777

-- The sum of the digits function
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Main theorem
theorem number_satisfying_condition : ∃ n : ℕ, n * sum_of_digits n = 2008 ∧ n = 251 :=
by
  sorry

end number_satisfying_condition_l108_108777


namespace age_problem_l108_108841

open Classical

variable (A B C : ℕ)

theorem age_problem (h1 : A + 10 = 2 * (B - 10))
                    (h2 : C = 3 * (A - 5))
                    (h3 : A = B + 9)
                    (h4 : C = A + 4) :
  B = 39 :=
sorry

end age_problem_l108_108841


namespace sector_area_is_correct_l108_108161

noncomputable def area_of_sector (r : ℝ) (α : ℝ) : ℝ := 1/2 * α * r^2

theorem sector_area_is_correct (circumference : ℝ) (central_angle : ℝ) (r : ℝ) (area : ℝ) 
  (h1 : circumference = 8) 
  (h2 : central_angle = 2) 
  (h3 : circumference = central_angle * r + 2 * r)
  (h4 : r = 2) : area = 4 :=
by
  have h5: area = 1/2 * central_angle * r^2 := sorry
  exact sorry

end sector_area_is_correct_l108_108161


namespace center_circle_sum_l108_108597

theorem center_circle_sum (h k : ℝ) :
  (∃ h k : ℝ, h + k = 6 ∧ ∃ R, (x - h)^2 + (y - k)^2 = R^2) ↔ ∃ h k : ℝ, h = 3 ∧ k = 3 ∧ h + k = 6 := 
by
  sorry

end center_circle_sum_l108_108597


namespace one_third_of_6_3_eq_21_10_l108_108509

theorem one_third_of_6_3_eq_21_10 : (6.3 / 3) = (21 / 10) := by
  sorry

end one_third_of_6_3_eq_21_10_l108_108509


namespace membership_relation_l108_108512

-- Definitions of M and N
def M (x : ℝ) : Prop := abs (x + 1) < 4
def N (x : ℝ) : Prop := x / (x - 3) < 0

theorem membership_relation (a : ℝ) (h : M a) : N a → M a := by
  sorry

end membership_relation_l108_108512


namespace functional_equation_solution_l108_108508

theorem functional_equation_solution (f : ℚ → ℚ) (h : ∀ x y : ℚ, f (x + y) = f x + f y) :
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x :=
sorry

end functional_equation_solution_l108_108508


namespace solve_system_l108_108716

theorem solve_system :
  ∃ x y : ℤ, (x - 3 * y = 7) ∧ (5 * x + 2 * y = 1) ∧ (x = 1) ∧ (y = -2) :=
by
  sorry

end solve_system_l108_108716


namespace fixed_point_l108_108136

variable (p : ℝ)

def f (x : ℝ) : ℝ := 9 * x^2 + p * x - 5 * p

theorem fixed_point : ∀ c d : ℝ, (∀ p : ℝ, f p c = d) → (c = 5 ∧ d = 225) :=
by
  intro c d h
  -- This is a placeholder for the proof
  sorry

end fixed_point_l108_108136


namespace find_number_l108_108884

theorem find_number (N : ℤ) (h1 : ∃ k : ℤ, N - 3 = 5 * k) (h2 : ∃ l : ℤ, N - 2 = 7 * l) (h3 : 50 < N ∧ N < 70) : N = 58 :=
by
  sorry

end find_number_l108_108884


namespace find_marks_in_biology_l108_108057

-- Definitions based on conditions in a)
def marks_english : ℕ := 76
def marks_math : ℕ := 60
def marks_physics : ℕ := 72
def marks_chemistry : ℕ := 65
def num_subjects : ℕ := 5
def average_marks : ℕ := 71

-- The theorem that needs to be proved
theorem find_marks_in_biology : 
  let total_marks := marks_english + marks_math + marks_physics + marks_chemistry 
  let total_marks_all := average_marks * num_subjects
  let marks_biology := total_marks_all - total_marks
  marks_biology = 82 := 
by
  sorry

end find_marks_in_biology_l108_108057


namespace exponent_is_23_l108_108027

theorem exponent_is_23 (k : ℝ) : (1/2: ℝ) ^ 23 * (1/81: ℝ) ^ k = (1/18: ℝ) ^ 23 → 23 = 23 := by
  intro h
  sorry

end exponent_is_23_l108_108027


namespace mortar_shell_hits_the_ground_at_50_seconds_l108_108538

noncomputable def mortar_shell_firing_equation (x : ℝ) : ℝ :=
  - (1 / 5) * x^2 + 10 * x

theorem mortar_shell_hits_the_ground_at_50_seconds : 
  ∃ x : ℝ, mortar_shell_firing_equation x = 0 ∧ x = 50 :=
by
  sorry

end mortar_shell_hits_the_ground_at_50_seconds_l108_108538


namespace a_plus_b_is_18_over_5_l108_108227

noncomputable def a_b_sum (a b : ℚ) : Prop :=
  (∃ (x y : ℚ), x = 2 ∧ y = 3 ∧ x = (1 / 3) * y + a ∧ y = (1 / 5) * x + b) → a + b = (18 / 5)

-- No proof provided, just the statement.
theorem a_plus_b_is_18_over_5 (a b : ℚ) : a_b_sum a b :=
sorry

end a_plus_b_is_18_over_5_l108_108227


namespace tickets_used_63_l108_108789

def rides_ferris_wheel : ℕ := 5
def rides_bumper_cars : ℕ := 4
def cost_per_ride : ℕ := 7
def total_rides : ℕ := rides_ferris_wheel + rides_bumper_cars
def total_tickets_used : ℕ := total_rides * cost_per_ride

theorem tickets_used_63 : total_tickets_used = 63 := by
  unfold total_tickets_used
  unfold total_rides
  unfold rides_ferris_wheel
  unfold rides_bumper_cars
  unfold cost_per_ride
  -- proof goes here
  sorry

end tickets_used_63_l108_108789


namespace range_of_a_l108_108978

noncomputable def f (x : ℝ) : ℝ := x + 1 / Real.exp x

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x > a * x) ↔ (1 - Real.exp 1 < a ∧ a ≤ 1) := by
  sorry

end range_of_a_l108_108978


namespace sample_capacity_l108_108177

theorem sample_capacity (n : ℕ) (A B C : ℕ) (h_ratio : A / (A + B + C) = 3 / 14) (h_A : A = 15) : n = 70 :=
by
  sorry

end sample_capacity_l108_108177


namespace total_dogs_in_kennel_l108_108917

-- Definition of the given conditions
def T := 45       -- Number of dogs that wear tags
def C := 40       -- Number of dogs that wear flea collars
def B := 6        -- Number of dogs that wear both tags and collars
def D_neither := 1 -- Number of dogs that wear neither a collar nor tags

-- Theorem statement
theorem total_dogs_in_kennel : T + C - B + D_neither = 80 := 
by
  -- Proof omitted
  sorry

end total_dogs_in_kennel_l108_108917


namespace inequality_proof_l108_108072

theorem inequality_proof (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  (a / b + b / c + c / a) ^ 2 ≥ 3 * (a / c + c / b + b / a) :=
  sorry

end inequality_proof_l108_108072


namespace collinear_probability_theorem_l108_108773

open Probability

/-- Define the possible outcomes of rolling a die. -/
def die_outcome := {n : ℕ | 1 ≤ n ∧ n ≤ 6}

/-- Define the vector \( \overrightarrow{q} = (3, 6) \). -/
def q : ℕ × ℕ := (3, 6)

/-- Define the event that vectors \( \overrightarrow{p} \) and \( \overrightarrow{q} \) are collinear. -/
def collinearity_event (m n : ℕ) : Prop := (n = 2 * m)

/-- Define the probability space for rolling a die twice. -/
def dice_prob_space := finset.pi_finset (finset.univ : finset die_outcome) (λ _, (finset.univ : finset die_outcome))

/-- Compute the probability that the vectors \( \overrightarrow{p} \) and \( \overrightarrow{q} \) are collinear. -/
def collinear_probability : ℝ :=
  let event := {x : die_outcome × die_outcome | collinearity_event x.1 x.2} in
  (event.card : ℝ) / (dice_prob_space.card : ℝ)

/-- Main theorem: The probability that the vectors \( \overrightarrow{p} = (m, n) \)
and \( \overrightarrow{q} = (3, 6)\) are collinear is \( \frac{1}{12} \). -/
theorem collinear_probability_theorem : collinear_probability = 1 / 12 :=
sorry

end collinear_probability_theorem_l108_108773


namespace train_crosses_second_platform_in_20_sec_l108_108782

theorem train_crosses_second_platform_in_20_sec
  (length_train : ℝ)
  (length_first_platform : ℝ)
  (time_first_platform : ℝ)
  (length_second_platform : ℝ)
  (time_second_platform : ℝ):

  length_train = 100 ∧
  length_first_platform = 350 ∧
  time_first_platform = 15 ∧
  length_second_platform = 500 →
  time_second_platform = 20 := by
  sorry

end train_crosses_second_platform_in_20_sec_l108_108782


namespace simplify_expression_l108_108577

theorem simplify_expression (x y : ℤ) (h1 : x = 1) (h2 : y = -2) :
  2 * x ^ 2 - (3 * (-5 / 3 * x ^ 2 + 2 / 3 * x * y) - (x * y - 3 * x ^ 2)) + 2 * x * y = 2 :=
by {
  sorry
}

end simplify_expression_l108_108577


namespace find_initial_pens_l108_108895

-- Conditions in the form of definitions
def initial_pens (P : ℕ) : ℕ := P
def after_mike (P : ℕ) : ℕ := P + 20
def after_cindy (P : ℕ) : ℕ := 2 * after_mike P
def after_sharon (P : ℕ) : ℕ := after_cindy P - 19

-- The final condition
def final_pens (P : ℕ) : ℕ := 31

-- The goal is to prove that the initial number of pens is 5
theorem find_initial_pens : 
  ∃ (P : ℕ), after_sharon P = final_pens P → P = 5 :=
by 
  sorry

end find_initial_pens_l108_108895


namespace track_circumference_l108_108902

def same_start_point (A B : ℕ) : Prop := A = B

def opposite_direction (a_speed b_speed : ℕ) : Prop := a_speed > 0 ∧ b_speed > 0

def first_meet_after (A B : ℕ) (a_distance b_distance : ℕ) : Prop := a_distance = 150 ∧ b_distance = 150

def second_meet_near_full_lap (B : ℕ) (lap_length short_distance : ℕ) : Prop := short_distance = 90

theorem track_circumference
    (A B : ℕ) (a_speed b_speed lap_length : ℕ)
    (h1 : same_start_point A B)
    (h2 : opposite_direction a_speed b_speed)
    (h3 : first_meet_after A B 150 150)
    (h4 : second_meet_near_full_lap B lap_length 90) :
    lap_length = 300 :=
sorry

end track_circumference_l108_108902


namespace smallest_fraction_l108_108760

theorem smallest_fraction (f1 f2 f3 f4 f5 : ℚ) (h1 : f1 = 2 / 3) (h2 : f2 = 3 / 4) (h3 : f3 = 5 / 6) 
  (h4 : f4 = 5 / 8) (h5 : f5 = 11 / 12) : f4 = 5 / 8 ∧ f4 < f1 ∧ f4 < f2 ∧ f4 < f3 ∧ f4 < f5 := 
by 
  sorry

end smallest_fraction_l108_108760


namespace eq_cont_fracs_l108_108714

noncomputable def cont_frac : Nat -> Rat
| 0       => 0
| (n + 1) => (n : Rat) + 1 / (cont_frac n)

theorem eq_cont_fracs (n : Nat) : 
  1 - cont_frac n = cont_frac n - 1 :=
sorry

end eq_cont_fracs_l108_108714


namespace option_C_true_l108_108636

theorem option_C_true (a b : ℝ):
    (a^2 + b^2 ≥ 2 * a * b) ↔ ((a^2 + b^2 > 2 * a * b) ∨ (a^2 + b^2 = 2 * a * b)) :=
by
  sorry

end option_C_true_l108_108636


namespace hyperbola_asymptotes_angle_l108_108665

noncomputable def angle_between_asymptotes 
  (a b : ℝ) (e : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : e = 2 * Real.sqrt 3 / 3) : ℝ :=
  2 * Real.arctan (b / a)

theorem hyperbola_asymptotes_angle (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : e = 2 * Real.sqrt 3 / 3) 
  (b_eq : b = Real.sqrt (e^2 * a^2 - a^2)) : 
  angle_between_asymptotes a b e h1 h2 h3 = π / 3 := 
by
  -- proof omitted
  sorry
  
end hyperbola_asymptotes_angle_l108_108665


namespace collinear_EFN_l108_108515

open EuclideanGeometry

-- Defining the cyclic quadrilateral and the midpoint
variables {A B C D M N E F : Point}
variable [CyclicQuadrilateral ABCD]

-- Definitions of M and N with given properties
variable (M_mid: isMidpoint M C D) (N_circumcircle_abm: OnCircle N (Circumcircle A B M)) 
variable (N_neq_M : N ≠ M)
variable (ratio_condition : (LineSegmentRatio A N B = LineSegmentRatio A M B))

-- Intersections defining points E and F
variable (E_def : E = intersection_point (line_through A C) (line_through B D))
variable (F_def : F = intersection_point (line_through B C) (line_through D A))

-- The theorem to prove collinearity
theorem collinear_EFN : Collinear {E, F, N} :=
by
  sorry

end collinear_EFN_l108_108515


namespace cards_per_set_is_13_l108_108193

-- Definitions based on the conditions
def total_cards : ℕ := 365
def sets_to_brother : ℕ := 8
def sets_to_sister : ℕ := 5
def sets_to_friend : ℕ := 2
def total_sets_given : ℕ := sets_to_brother + sets_to_sister + sets_to_friend
def total_cards_given : ℕ := 195

-- The problem to prove
theorem cards_per_set_is_13 : total_cards_given / total_sets_given = 13 :=
  by
  -- Here we would provide the proof, but for now, we use sorry
  sorry

end cards_per_set_is_13_l108_108193


namespace find_c_plus_one_div_b_l108_108220

-- Assume that a, b, and c are positive real numbers such that the given conditions hold.
variables (a b c : ℝ)
variables (habc : a * b * c = 1)
variables (hac : a + 1 / c = 7)
variables (hba : b + 1 / a = 11)

-- The goal is to show that c + 1 / b = 5 / 19.
theorem find_c_plus_one_div_b : c + 1 / b = 5 / 19 :=
by 
  sorry

end find_c_plus_one_div_b_l108_108220


namespace james_payment_correct_l108_108404

-- Definitions from conditions
def cost_steak_eggs : ℝ := 16
def cost_chicken_fried_steak : ℝ := 14
def total_cost := cost_steak_eggs + cost_chicken_fried_steak
def half_share := total_cost / 2
def tip := total_cost * 0.2
def james_total_payment := half_share + tip

-- Statement to be proven
theorem james_payment_correct : james_total_payment = 21 :=
by
  sorry

end james_payment_correct_l108_108404


namespace count_digit_7_to_199_l108_108053

theorem count_digit_7_to_199 : (Nat.countDigitInRange 7 100 199) = 20 := by
  sorry

end count_digit_7_to_199_l108_108053


namespace simplify_expr_l108_108217

variable (a b : ℤ)  -- assuming a and b are elements of the ring ℤ

theorem simplify_expr : 105 * a - 38 * a + 27 * b - 12 * b = 67 * a + 15 * b := 
by
  sorry

end simplify_expr_l108_108217


namespace quadratic_roots_l108_108139

theorem quadratic_roots (m x1 x2 : ℝ) 
  (h1 : 2*x1^2 + 4*m*x1 + m = 0)
  (h2 : 2*x2^2 + 4*m*x2 + m = 0)
  (h3 : x1 ≠ x2)
  (h4 : x1^2 + x2^2 = 3/16) : 
  m = -1/8 := 
sorry

end quadratic_roots_l108_108139


namespace silverware_probability_l108_108997

def numWaysTotal (totalPieces : ℕ) (choosePieces : ℕ) : ℕ :=
  Nat.choose totalPieces choosePieces

def numWaysForks (forks : ℕ) (chooseForks : ℕ) : ℕ :=
  Nat.choose forks chooseForks

def numWaysSpoons (spoons : ℕ) (chooseSpoons : ℕ) : ℕ :=
  Nat.choose spoons chooseSpoons

def numWaysKnives (knives : ℕ) (chooseKnives : ℕ) : ℕ :=
  Nat.choose knives chooseKnives

def favorableOutcomes (forks : ℕ) (spoons : ℕ) (knives : ℕ) : ℕ :=
  numWaysForks forks 2 * numWaysSpoons spoons 1 * numWaysKnives knives 1

def probability (totalWays : ℕ) (favorableWays : ℕ) : ℚ :=
  favorableWays / totalWays

theorem silverware_probability :
  probability (numWaysTotal 18 4) (favorableOutcomes 5 7 6) = 7 / 51 := by
  sorry

end silverware_probability_l108_108997


namespace cake_fraction_eaten_l108_108554

theorem cake_fraction_eaten (total_slices kept_slices slices_eaten : ℕ) 
  (h1 : total_slices = 12)
  (h2 : kept_slices = 9)
  (h3 : slices_eaten = total_slices - kept_slices) :
  (slices_eaten : ℚ) / total_slices = 1 / 4 := 
sorry

end cake_fraction_eaten_l108_108554


namespace min_value_quadratic_l108_108826

theorem min_value_quadratic (x : ℝ) : x = -1 ↔ (∀ y : ℝ, x^2 + 2*x + 4 ≤ y) := by
  sorry

end min_value_quadratic_l108_108826


namespace smallest_fraction_gt_five_sevenths_l108_108785

theorem smallest_fraction_gt_five_sevenths (a b : ℕ) (h1 : 10 ≤ a ∧ a ≤ 99) (h2 : 10 ≤ b ∧ b ≤ 99) (h3 : 7 * a > 5 * b) : a = 68 ∧ b = 95 :=
sorry

end smallest_fraction_gt_five_sevenths_l108_108785


namespace ratio_shiny_igneous_to_total_l108_108037

-- Define the conditions
variable (S I SI : ℕ)
variable (SS : ℕ)
variable (h1 : I = S / 2)
variable (h2 : SI = 40)
variable (h3 : S + I = 180)
variable (h4 : SS = S / 5)

-- Statement to prove
theorem ratio_shiny_igneous_to_total (S I SI SS : ℕ) 
  (h1 : I = S / 2) 
  (h2 : SI = 40) 
  (h3 : S + I = 180) 
  (h4 : SS = S / 5) : 
  SI / I = 2 / 3 := 
sorry

end ratio_shiny_igneous_to_total_l108_108037


namespace min_function_value_in_domain_l108_108644

theorem min_function_value_in_domain :
  ∃ (x y : ℝ), (1 / 3 ≤ x ∧ x ≤ 3 / 5) ∧ (1 / 4 ≤ y ∧ y ≤ 1 / 2) ∧ (∀ (x y : ℝ), (1 / 3 ≤ x ∧ x ≤ 3 / 5) ∧ (1 / 4 ≤ y ∧ y ≤ 1 / 2) → (xy / (x^2 + y^2)) ≥ (60 / 169)) :=
sorry

end min_function_value_in_domain_l108_108644


namespace number_of_zeros_of_f_l108_108164

def f (x : ℝ) : ℝ := 2 * x - 3 * x

theorem number_of_zeros_of_f :
  ∃ (n : ℕ), n = 2 ∧ (∀ x, f x = 0 → x ∈ {x | f x = 0}) :=
by {
  sorry
}

end number_of_zeros_of_f_l108_108164


namespace find_values_of_a_b_solve_inequality_l108_108825

variable (a b : ℝ)
variable (h1 : ∀ x : ℝ, a * x^2 + b * x + 2 = 0 ↔ x = -1/2 ∨ x = 2)

theorem find_values_of_a_b (h2 : a = -2) (h3 : b = 3) : 
  a = -2 ∧ b = 3 :=
by
  constructor
  exact h2
  exact h3


theorem solve_inequality 
  (h2 : a = -2) (h3 : b = 3) :
  ∀ x : ℝ, (a * x^2 + b * x - 1 > 0) ↔ (1/2 < x ∧ x < 1) :=
by
  sorry

end find_values_of_a_b_solve_inequality_l108_108825


namespace chemistry_club_student_count_l108_108640

theorem chemistry_club_student_count (x : ℕ) (h1 : x % 3 = 0)
  (h2 : x % 4 = 0) (h3 : x % 6 = 0)
  (h4 : (x / 3) = (x / 4) + 3) :
  (x / 6) = 6 :=
by {
  -- Proof goes here
  sorry
}

end chemistry_club_student_count_l108_108640


namespace factor_in_form_of_2x_l108_108905

theorem factor_in_form_of_2x (w : ℕ) (hw : w = 144) : ∃ x : ℕ, 936 * w = 2^x * P → x = 4 :=
by
  sorry

end factor_in_form_of_2x_l108_108905


namespace larger_number_l108_108456

theorem larger_number (x y : ℤ) (h1 : x + y = 47) (h2 : x - y = 7) : x = 27 :=
  sorry

end larger_number_l108_108456


namespace number_of_good_card_groups_l108_108483

noncomputable def card_value (k : ℕ) : ℕ := 2 ^ k

def is_good_card_group (cards : Finset ℕ) : Prop :=
  (cards.sum card_value = 2004)

theorem number_of_good_card_groups : 
  ∃ n : ℕ, n = 1006009 ∧ ∃ (cards : Finset ℕ), is_good_card_group cards :=
sorry

end number_of_good_card_groups_l108_108483


namespace smallest_k_l108_108741

theorem smallest_k :
  ∃ k : ℕ, k > 1 ∧ (k % 19 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) ∧ k = 400 :=
by
  sorry

end smallest_k_l108_108741


namespace smallest_fraction_numerator_l108_108783

theorem smallest_fraction_numerator :
  ∃ (a b : ℕ), (10 ≤ a ∧ a < 100) ∧ (10 ≤ b ∧ b < 100) ∧ (5 * b < 7 * a) ∧ 
    ∀ (a' b' : ℕ), (10 ≤ a' ∧ a' < 100) ∧ (10 ≤ b' ∧ b' < 100) ∧ (5 * b' < 7 * a') →
    (a * b' ≤ a' * b) → a = 68 :=
sorry

end smallest_fraction_numerator_l108_108783


namespace initial_number_correct_l108_108103

def initial_number_problem : Prop :=
  ∃ (x : ℝ), x + 3889 - 47.80600000000004 = 3854.002 ∧
            x = 12.808000000000158

theorem initial_number_correct : initial_number_problem :=
by
  -- proof goes here
  sorry

end initial_number_correct_l108_108103


namespace sheepdog_speed_l108_108555

theorem sheepdog_speed 
  (T : ℝ) (t : ℝ) (sheep_speed : ℝ) (initial_distance : ℝ)
  (total_distance_speed : ℝ) :
  T = 20  →
  t = 20 →
  sheep_speed = 12 →
  initial_distance = 160 →
  total_distance_speed = 20 →
  total_distance_speed * T = initial_distance + sheep_speed * t := 
by sorry

end sheepdog_speed_l108_108555


namespace sandwiches_left_l108_108703

theorem sandwiches_left (S G K L : ℕ) (h1 : S = 20) (h2 : G = 4) (h3 : K = 2 * G) (h4 : L = S - G - K) : L = 8 :=
sorry

end sandwiches_left_l108_108703


namespace total_time_spent_l108_108492

-- Define time spent on each step
def time_first_step : ℕ := 30
def time_second_step : ℕ := time_first_step / 2
def time_third_step : ℕ := time_first_step + time_second_step

-- Prove the total time spent
theorem total_time_spent : 
  time_first_step + time_second_step + time_third_step = 90 := by
  sorry

end total_time_spent_l108_108492


namespace jacob_walked_8_miles_l108_108690

theorem jacob_walked_8_miles (rate time : ℝ) (h_rate : rate = 4) (h_time : time = 2) :
  rate * time = 8 := by
  -- conditions
  have hr : rate = 4 := h_rate
  have ht : time = 2 := h_time
  -- problem
  sorry

end jacob_walked_8_miles_l108_108690


namespace minimum_x_y_sum_l108_108526

theorem minimum_x_y_sum (x y : ℕ) (hx : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h : (1 / (x : ℚ)) + (1 / (y : ℚ)) = 1 / 15) : x + y = 64 :=
  sorry

end minimum_x_y_sum_l108_108526


namespace probability_one_defective_l108_108754

theorem probability_one_defective (g d : ℕ) (h_g : g = 3) (h_d : d = 1) : 
  let total_items := g + d in
  let sample_space := (total_items.choose 2).toFinset in
  let event_A := {x ∈ sample_space | x.count (0 = ∘ id) = 1} in
  (event_A.card : ℚ) / (sample_space.card : ℚ) = 1 / 2 :=
by
  sorry

end probability_one_defective_l108_108754


namespace largest_expression_is_A_l108_108499

noncomputable def A : ℝ := 3009 / 3008 + 3009 / 3010
noncomputable def B : ℝ := 3011 / 3010 + 3011 / 3012
noncomputable def C : ℝ := 3010 / 3009 + 3010 / 3011

theorem largest_expression_is_A : A > B ∧ A > C := by
  sorry

end largest_expression_is_A_l108_108499


namespace smallest_nat_divisible_by_225_l108_108544

def has_digits_0_or_1 (n : ℕ) : Prop := 
  ∀ (d : ℕ), d ∈ n.digits 10 → d = 0 ∨ d = 1

def divisible_by_225 (n : ℕ) : Prop := 225 ∣ n

theorem smallest_nat_divisible_by_225 :
  ∃ (n : ℕ), has_digits_0_or_1 n ∧ divisible_by_225 n 
    ∧ ∀ (m : ℕ), has_digits_0_or_1 m ∧ divisible_by_225 m → n ≤ m 
    ∧ n = 11111111100 := 
  sorry

end smallest_nat_divisible_by_225_l108_108544


namespace complement_of_intersection_l108_108701

open Set

-- Define the universal set U
def U := @univ ℝ
-- Define the sets M and N
def M : Set ℝ := {x | x >= 2}
def N : Set ℝ := {x | 0 <= x ∧ x < 5}

-- Define M ∩ N
def M_inter_N := M ∩ N

-- Define the complement of M ∩ N with respect to U
def C_U (A : Set ℝ) := Aᶜ

theorem complement_of_intersection :
  C_U M_inter_N = {x : ℝ | x < 2 ∨ x ≥ 5} := 
by 
  sorry

end complement_of_intersection_l108_108701


namespace snakes_in_each_cage_l108_108779

theorem snakes_in_each_cage (total_snakes : ℕ) (total_cages : ℕ) (h_snakes: total_snakes = 4) (h_cages: total_cages = 2) 
  (h_even_distribution : (total_snakes % total_cages) = 0) : (total_snakes / total_cages) = 2 := 
by sorry

end snakes_in_each_cage_l108_108779


namespace hypotenuse_of_45_45_90_triangle_15_l108_108563

theorem hypotenuse_of_45_45_90_triangle_15 (a : ℝ) (h : a = 15) : 
  ∃ (c : ℝ), c = a * Real.sqrt 2 :=
by
  use a * Real.sqrt 2
  rw h
  sorry

end hypotenuse_of_45_45_90_triangle_15_l108_108563
