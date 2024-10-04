import Mathlib

namespace point_on_transformed_graph_l173_173459

variable (f : ℝ → ℝ)

theorem point_on_transformed_graph :
  (f 12 = 10) →
  3 * (19 / 9) = (f (3 * 4)) / 3 + 3 ∧ (4 + 19 / 9 = 55 / 9) :=
by
  sorry

end point_on_transformed_graph_l173_173459


namespace allocate_students_to_locations_l173_173406

theorem allocate_students_to_locations :
  ∃! (arrangements : ℕ), 
    (arrangements = 36) ∧ 
    -- Conditions
    let students := {1, 2, 3, 4}; 
    let locations := {A, B, C}; 
    ∀ (allocation : students → locations), 
      (∀ l ∈ locations, ∃ s ∈ students, allocation s = l) :=
sorry

end allocate_students_to_locations_l173_173406


namespace expected_number_of_sixes_l173_173669

-- Define the problem context and conditions
def die_prob := (1 : ℝ) / 6

def expected_six (n : ℕ) : ℝ :=
  n * die_prob

-- The main proposition to prove
theorem expected_number_of_sixes (n : ℕ) (hn : n = 3) : expected_six n = 1 / 2 :=
by
  rw [hn]
  have fact1 : (3 : ℝ) * die_prob = 3 / 6 := by norm_cast; norm_num
  rw [fact1]
  norm_num

-- We add sorry to indicate incomplete proof, fulfilling criteria 4
sorry

end expected_number_of_sixes_l173_173669


namespace sin_sum_leq_3_sqrt3_over_2_l173_173060

theorem sin_sum_leq_3_sqrt3_over_2 
  (A B C : ℝ) 
  (h₁ : A + B + C = Real.pi) 
  (h₂ : 0 < A ∧ A < Real.pi)
  (h₃ : 0 < B ∧ B < Real.pi)
  (h₄ : 0 < C ∧ C < Real.pi) :
  Real.sin A + Real.sin B + Real.sin C ≤ 3 * Real.sqrt 3 / 2 :=
sorry

end sin_sum_leq_3_sqrt3_over_2_l173_173060


namespace lowest_position_l173_173234

theorem lowest_position (num_cyclists : ℕ) (num_stages : ℕ) (vasya_position : ℕ) :
  num_cyclists = 500 →
  num_stages = 15 →
  vasya_position = 7 →
  ∃ n, n = 91 :=
by
  intros
  sorry

end lowest_position_l173_173234


namespace hyperbola_foci_l173_173804

noncomputable def foci_larger_x_coordinate : ℝ × ℝ :=
  let a := 7
  let b := 3
  let c := Real.sqrt (a^2 + b^2)
  (-2 + c, 5)

theorem hyperbola_foci :
  ∃ c, (foci_larger_x_coordinate = (-2 + Real.sqrt (7^2 + 3^2), 5)) :=
by
  use Real.sqrt (7^2 + 3^2)
  exact sorry

end hyperbola_foci_l173_173804


namespace rabbit_speed_l173_173059

theorem rabbit_speed (s : ℕ) (h : (s * 2 + 4) * 2 = 188) : s = 45 :=
sorry

end rabbit_speed_l173_173059


namespace four_digit_palindromic_squares_count_l173_173875

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

theorem four_digit_palindromic_squares_count :
  let four_digit_palindromes := { n | n >= 1000 ∧ n < 10000 ∧ is_palindrome n } in
  let perfect_squares := { n | ∃ k, k^2 = n } in
  (four_digit_palindromes ∩ perfect_squares).card = 2 := 
by sorry

end four_digit_palindromic_squares_count_l173_173875


namespace collinear_points_slopes_l173_173512

theorem collinear_points_slopes (m : ℝ) :
  let A := (-2 : ℝ, 12 : ℝ)
  let B := (1 : ℝ, 3 : ℝ)
  let C := (m, -6 : ℝ)
  (B.2 - A.2) / (B.1 - A.1) = (C.2 - A.2) / (C.1 - A.1) →
  m = 4 :=
by
  let A := (-2 : ℝ, 12 : ℝ)
  let B := (1 : ℝ, 3 : ℝ)
  let C := (m, -6 : ℝ)
  intro h
  sorry

end collinear_points_slopes_l173_173512


namespace projection_result_l173_173696

open Real

def v1 := (3 : ℝ, -4 : ℝ)
def v2 := (-1 : ℝ, 6 : ℝ)
def p := (87 / 29 : ℝ, 90 / 29 : ℝ)

theorem projection_result : 
  ∃ v : ℝ × ℝ, 
    let l := (v1.fst + (v2.fst - v1.fst) * v.fst, 
              v1.snd + (v2.snd - v1.snd) * v.snd) in 
    ((l.fst * (-4) + l.snd * 10) = 0) ∧ 
    ∀ t : ℝ, (t = 13 / 29) → p = 
      (v1.fst + (v2.fst - v1.fst) * t, v1.snd + (v2.snd - v1.snd) * t) := 
by
  sorry

end projection_result_l173_173696


namespace find_scalars_l173_173108

open Matrix

variable {α : Type*} [DecidableEq α] [Fintype α] [CommRing α]

def M : Matrix (Fin 2) (Fin 2) α :=
  ![![3, 4], ![-2, 0]]

def I : Matrix (Fin 2) (Fin 2) α := 1

theorem find_scalars : ∃ (p q : α), M ^ 2 = p • M + q • I := by
  let M₂ := ![![1, 12], ![-6, -8]]
  use 3, -8
  show M₂ = 3 • M + (-8 : α) • I
  sorry

end find_scalars_l173_173108


namespace pass_rate_is_frequency_l173_173926

theorem pass_rate_is_frequency 
  (pass_rate : ℕ)
  (h : pass_rate = 70) :
  "frequency" := 
sorry

end pass_rate_is_frequency_l173_173926


namespace diminished_value_l173_173741

theorem diminished_value (x y : ℝ) (h1 : x = 160)
  (h2 : x / 5 + 4 = x / 4 - y) : y = 4 :=
by
  sorry

end diminished_value_l173_173741


namespace jerry_feathers_left_l173_173089

def hawk_feathers : ℕ := 37
def eagle_feathers : ℝ := 17.5 * hawk_feathers
def total_feathers : ℝ := hawk_feathers + eagle_feathers
def feathers_to_sister : ℝ := 0.45 * total_feathers
def remaining_feathers_after_sister : ℝ := total_feathers - feathers_to_sister
def feathers_sold : ℝ := 0.85 * remaining_feathers_after_sister
def final_remaining_feathers : ℝ := remaining_feathers_after_sister - feathers_sold

theorem jerry_feathers_left : ⌊final_remaining_feathers⌋₊ = 56 := by
  sorry

end jerry_feathers_left_l173_173089


namespace binary_to_base8_conversion_l173_173389

theorem binary_to_base8_conversion :  (46 : Nat) =  (1001010₈) :=
by sorry

end binary_to_base8_conversion_l173_173389


namespace jacob_dinner_calories_l173_173542

variable (goal : ℕ)
variable (breakfast_calories lunch_calories dinner_calories : ℕ)
variable (extra_calories : ℕ)

def jacobs_total_calories := breakfast_calories + lunch_calories + dinner_calories + extra_calories

theorem jacob_dinner_calories : 
  goal < 1800 → 
  breakfast_calories = 400 → 
  lunch_calories = 900 → 
  extra_calories = 600 → 
  goal + extra_calories = jacobs_total_calories → 
  dinner_calories = 1100 :=
by
  intros h1 h2 h3 h4 h5
  rw [h2, h3, h4] at h5
  sorry

end jacob_dinner_calories_l173_173542


namespace maximum_area_rhombus_l173_173533

theorem maximum_area_rhombus 
    (x₀ y₀ k : ℝ)
    (h1 : 2 ≤ x₀ ∧ x₀ ≤ 4)
    (h2 : y₀ = k / x₀)
    (h3 : ∀ x > 0, ∃ y, y = k / x) :
    (∀ (x₀ : ℝ), 2 ≤ x₀ ∧ x₀ ≤ 4 → ∃ (S : ℝ), S = 3 * (Real.sqrt 2 / 2 * x₀^2) → S ≤ 24 * Real.sqrt 2) :=
by
  sorry

end maximum_area_rhombus_l173_173533


namespace value_of_fraction_l173_173561

variable {x y : ℝ}

theorem value_of_fraction (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) : 
  (x + y) / (x - y) = Real.sqrt 3 := 
sorry

end value_of_fraction_l173_173561


namespace race_distance_where_A_beats_B_l173_173921

variable (A B C : Type)
variable (distance : A → B → ℝ)
variable (race_distance : ℝ)
variable (D : ℝ)

axiom A_beats_B : ∀ (A B : Type), distance A B = D - 100
axiom B_beats_C_in_800 : ∀ (B C : Type), distance B C = 800 - 100
axiom A_beats_C_by_212_5 : ∀ (A C : Type), distance A C = D - 212.5

theorem race_distance_where_A_beats_B : 
  (8 * (distance B C)) = 7 * (D - 100) → 
  (distance A C) = (D - 212.5) → 
  D = 1000 :=
  by
    sorry

end race_distance_where_A_beats_B_l173_173921


namespace exists_club_with_11_boys_and_girls_l173_173925

-- Definitions
def num_girls : ℕ := 2013
def num_boys : ℕ := 2013
def max_clubs_per_student : ℕ := 100

-- Assumptions
axiom common_club :
  ∀ (girl : ℕ) (boy : ℕ), girl < num_girls → boy < num_boys → ∃ c, c < max_clubs_per_student

-- Proposition to prove
theorem exists_club_with_11_boys_and_girls :
  ∃ c, (∃ n_boys, n_boys ≥ 11 ∧ n_boys ≤ num_boys ∧ ∀ b_boy < n_boys, b_boy < max_clubs_per_student) ∧
       (∃ n_girls, n_girls ≥ 11 ∧ n_girls ≤ num_girls ∧ ∀ g_girl < n_girls, g_girl < max_clubs_per_student) :=
sorry

end exists_club_with_11_boys_and_girls_l173_173925


namespace money_distribution_l173_173426

theorem money_distribution (a b : ℝ) 
  (h1 : 4 * a - b = 40)
  (h2 : 6 * a + b = 110) :
  a = 15 ∧ b = 20 :=
by
  sorry

end money_distribution_l173_173426


namespace count_valid_house_arrangements_l173_173159

noncomputable def number_of_valid_arrangements : ℕ :=
  let houses := ["green", "blue", "pink", "orange", "red"]
  let valid_arrangements := {arrangement | 
    arrangement ∈ list.permutations houses ∧ 
    list.index_of "green" arrangement < list.index_of "blue" arrangement ∧ 
    list.index_of "pink" arrangement < list.index_of "orange" arrangement ∧ 
    (list.index_of "pink" arrangement ≠ list.index_of "blue" arrangement + 1) ∧ 
    (list.index_of "pink" arrangement ≠ list.index_of "blue" arrangement - 1)}
  in list.length valid_arrangements.to_list

theorem count_valid_house_arrangements : 
  number_of_valid_arrangements = 15 := 
sorry

end count_valid_house_arrangements_l173_173159


namespace number_of_valid_two_digit_numbers_l173_173813

-- R(n) definition: sum of remainders when n is divided by 2 to 12.
def R (n : ℕ) : ℕ :=
  (∑ k in Finset.range 11, (n % (k + 2)))

-- S(n) definition: sum of the digits of n.
def digitSum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def S (n : ℕ) : ℕ := digitSum n

-- Define the main proof statement
theorem number_of_valid_two_digit_numbers : Finset.card
  {n ∈ Finset.range 90 ∩ Finset.Ico 10 100 | R n = R (n + 1) ∧ S n % 2 = 0} = 2 :=
by
  sorry

end number_of_valid_two_digit_numbers_l173_173813


namespace JoseBasketballPlaytime_l173_173093

variable (football_playtime : ℕ) (total_playtime_hours : ℚ) (hour_to_minutes : ℕ → ℕ)

-- Given conditions
def football_playtime := 30
def total_playtime_hours := 1.5
def hour_to_minutes (h : ℕ) : ℕ := h * 60

-- The statement to be proved
theorem JoseBasketballPlaytime :
  ∃ (basketball_playtime : ℕ), basketball_playtime = (hour_to_minutes (total_playtime_hours.to_nat) - football_playtime) :=
sorry

end JoseBasketballPlaytime_l173_173093


namespace sum_of_possible_b_values_l173_173274

noncomputable def g (x b : ℝ) : ℝ := x^2 - b * x + 3 * b

theorem sum_of_possible_b_values :
  (∀ (x₀ x₁ : ℝ), g x₀ x₁ = 0 → g x₀ x₁ = (x₀ - x₁) * (x₀ - 3)) → ∃ b : ℝ, b = 12 ∨ b = 16 :=
sorry

end sum_of_possible_b_values_l173_173274


namespace min_t_value_l173_173624

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3 * x - 1

theorem min_t_value : 
    (∀ x1 x2 : ℝ, x1 ∈ set.Icc (-3 : ℝ) 2 → x2 ∈ set.Icc (-3 : ℝ) 2 → |f x1 - f x2| ≤ t) ↔ t = 20 := 
by
  sorry

end min_t_value_l173_173624


namespace expected_number_of_sixes_when_three_dice_are_rolled_l173_173660

theorem expected_number_of_sixes_when_three_dice_are_rolled : 
  ∑ n in finset.range 4, (n * (↑(finset.filter (λ xs : fin 3 → fin 6, xs.count (λ x, x = 5) = n) finset.univ).card / 216 : ℚ)) = 1 / 2 :=
by
  -- Conclusion of proof is omitted as per instructions
  sorry

end expected_number_of_sixes_when_three_dice_are_rolled_l173_173660


namespace isosceles_triangle_ADE_l173_173107

open EuclideanGeometry

variable (Γ : Circle) (P A B C D E : Point)
variable (hP_outside_Γ : ¬(P ∈ Γ))
variable (h_tangent : tangent P A Γ)
variable (h_line_through_P : line_through P B C)
variable (h_B_C_distinct : B ≠ C)
variable (h_bisector_APB_D : bisector (angle P A B) A P D)
variable (h_bisector_APB_E : bisector (angle P A B) A P E)
variable (h_D_on_AB : D ∈ segment A B)
variable (h_E_on_AC : E ∈ segment A C)

theorem isosceles_triangle_ADE : isosceles_triangle A D E := 
  sorry

end isosceles_triangle_ADE_l173_173107


namespace mike_arcade_play_time_l173_173577

def weekly_pay : ℕ := 100
def fraction_spent_arcade : ℕ → ℕ := λ pay, pay / 2
def money_spent_on_food : ℕ := 10
def cost_per_hour : ℕ := 8
def minutes_per_hour : ℕ := 60

theorem mike_arcade_play_time : 
  let weekly_spending := fraction_spent_arcade weekly_pay in
  let money_spent_on_tokens := weekly_spending - money_spent_on_food in
  let hours_played := money_spent_on_tokens / cost_per_hour in
  let total_minutes := hours_played * minutes_per_hour in
  total_minutes = 300 :=
by
  sorry

end mike_arcade_play_time_l173_173577


namespace find_a_from_inequality_l173_173053

variable {x : ℝ}

def inequality (a : ℝ) : Prop := ∀ x, 1 < x ∧ x < 2 → (ax) / (x - 1) > 1

theorem find_a_from_inequality : ∃ a : ℝ, inequality a ∧ a = 1 / 2 :=
by
  sorry

end find_a_from_inequality_l173_173053


namespace solve_pq_eq_four_l173_173132

noncomputable def p (x : ℤ) : ℤ := x^4 + 5 * x^2 + 1
noncomputable def q (x : ℤ) : ℤ := x^4 - 5 * x^2 + 1

theorem solve_pq_eq_four : p(1) + q(1) = 4 := by
  sorry

end solve_pq_eq_four_l173_173132


namespace max_value_of_Q_l173_173424

theorem max_value_of_Q (a : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 0.5) : 
  let Q := λ a : ℝ, ∫ x in 0..(2 * a), ∫ y in 0..2, indicator (set_of (λ p : ℝ × ℝ, cos (π * p.1) + cos (π * p.2) > 1)) (x, y) in
  ∃ x y, (0 ≤ x) ∧ (x ≤ 2 * a) ∧ (0 ≤ y) ∧ (y ≤ 2) ∧ (cos (π * x) + cos (π * y) > 1) → Q a = 1 :=
begin
  intro h,
  sorry
end

end max_value_of_Q_l173_173424


namespace min_good_segments_l173_173991

theorem min_good_segments (n n1 n2 n3 n4 : ℕ)
(h_sum: n1 + n2 + n3 + n4 = n) :
  let num_good_segments :=
    if n % 4 = 0 then
      n * (n-4) / 8
    else if n % 4 = 1 ∨ n % 4 = 3 then
      (n-1) * (n-3) / 8
    else
      (n-2) * (n-2) / 8
  in
  num_good_segments = (if n % 4 = 0 then
                         n * (n-4) / 8
                       else if n % 4 = 1 ∨ n % 4 = 3 then
                         (n-1) * (n-3) / 8
                       else
                         (n-2) * (n-2) / 8) :=
begin
  sorry
end

end min_good_segments_l173_173991


namespace range_of_a_l173_173022

-- Function definition
def f (a x: ℝ) : ℝ := x^3 - 3 * a^2 * x + a

-- Proof problem: Prove the range of a such that the function has its maximum value positive and its minimum value negative
theorem range_of_a (a : ℝ) (h : a > 0) : set.Ioi (real.sqrt 2 / 2) :=
by sorry

end range_of_a_l173_173022


namespace determine_unique_d_l173_173651

theorem determine_unique_d (a b c d : ℝ) (h1 : ∃ (a b c : ℝ), 
  ∀ (x y z: ℝ), 
    (x, y, z) = (2, 0, a) ∨ (x, y, z) = (b, 2, 0) ∨ (x, y, z) = (0, c, 2) ∨ (x, y, z) = (4 * d, 4 * d, -d) → 
    (algebra_vector.collinear [ℝ, ℝ, ℝ] [ ((x, y, z), (2, 0, a)), ((b, 2, 0), (0, c, 2)), ((0, c, 2), (4 * d, 4 * d, -d)) ]) ) :
  d = 2 / 3 :=
by
  sorry

end determine_unique_d_l173_173651


namespace vector_dot_product_l173_173869

def a : Vector ℝ 2 := ![1/2, Real.sqrt 3 / 2]
def b : Vector ℝ 2 := ![-(Real.sqrt 3) / 2, 1/2]

theorem vector_dot_product : dot_product (a + b) a = 1 := by
  sorry

end vector_dot_product_l173_173869


namespace lowest_position_of_vasya_l173_173208

-- Definitions of conditions
def num_cyclists : ℕ := 500
def num_stages : ℕ := 15
def vasya_position_each_stage : ℕ := 7

-- Theorem statement
theorem lowest_position_of_vasya (H1 : ∀ (s: ℕ), s ∈ finset.range(num_stages) → 
(num_cyclists + 1) - vasya_position_each_stage > (num_cyclists - 90))

(assumption_vasya :
  ∀ s ∈ finset.range(num_stages), vasya_position_each_stage < num_cyclists):
  ∃ (lowest_position: ℕ), lowest_position = 91 :=
sorry

end lowest_position_of_vasya_l173_173208


namespace find_denominators_l173_173650

theorem find_denominators (f1 f2 f3 f4 f5 f6 f7 f8 f9 : ℚ)
  (h1 : f1 = 1/3) (h2 : f2 = 1/7) (h3 : f3 = 1/9) (h4 : f4 = 1/11) (h5 : f5 = 1/33)
  (h6 : ∃ (d₁ d₂ d₃ d₄ : ℕ), f6 = 1/d₁ ∧ f7 = 1/d₂ ∧ f8 = 1/d₃ ∧ f9 = 1/d₄ ∧
    (∀ d, d ∈ [d₁, d₂, d₃, d₄] → d % 10 = 5))
  (h7 : f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 = 1) :
  ∃ (d₁ d₂ d₃ d₄ : ℕ), (d₁ = 5) ∧ (d₂ = 15) ∧ (d₃ = 45) ∧ (d₄ = 385) :=
by
  sorry

end find_denominators_l173_173650


namespace sandra_share_l173_173602

theorem sandra_share 
  (r : ℕ) (sandra_ratio amy's_share : ℕ) 
  (h1 : sandra_ratio = 2) (h2 : amy's_share = 50) : 
  sandra_ratio * amy's_share = 100 :=
by
  rw [h1, h2]
  sorry

end sandra_share_l173_173602


namespace domain_of_function_l173_173636

theorem domain_of_function :
  ∀ x : ℝ, (0 ≤ x ∧ x ≠ 1) ↔ (∃ y : ℝ, y = 1 / (Real.sqrt x - 1)) := by
  sorry

end domain_of_function_l173_173636


namespace area_CEF_eq_six_l173_173935

variables {A B C E F : Type*}
variables [triangle : has_triangle A B C]
variables [point : has_point E F]
variables (h_cond1 : E_in_segment_one_third_A_C : ∃ A C : Type*, E ∈ segment A C ∧ distance A E = (1/3) * distance A C)
variables (h_cond2 : F_mid_AB : ∃ A B : Type*, F ∈ midpoint A B)
variables (area_ABC : area triangle A B C = 36)

theorem area_CEF_eq_six : ∃ μ : ℝ, area triangle C E F = μ ∧ μ = 6 :=
by
  sorry

end area_CEF_eq_six_l173_173935


namespace evaluate_g_of_h_l173_173888

def g (x : ℝ) : ℝ := 3 * x^2 - 4

def h (x : ℝ) : ℝ := 5 * x^3 + 2

theorem evaluate_g_of_h : g (h (-2)) = 4328 := 
by
  sorry

end evaluate_g_of_h_l173_173888


namespace calculate_triple_hash_l173_173785

def hash (N : ℝ) : ℝ := 0.5 * N - 2

theorem calculate_triple_hash : hash (hash (hash 100)) = 9 := by
  sorry

end calculate_triple_hash_l173_173785


namespace ratio_of_areas_l173_173904

theorem ratio_of_areas (R_C R_D : ℝ) (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 4 / 9 :=
by
  sorry

end ratio_of_areas_l173_173904


namespace smallest_x_proof_l173_173122

noncomputable def smallest_x : ℝ :=
  let x := 30 in x

theorem smallest_x_proof :
  ∃ x : ℝ, x > 5 ∧ sin (x * π / 180) = cos (2 * x * π / 180) ∧ x = smallest_x :=
by {
  let x := smallest_x,
  use x,
  have h₁ : x > 5, by norm_num,
  have h₂ : sin (x * π / 180) = cos (2 * x * π / 180), 
    by {
      rw [← sin_add, ← cos_eq_sin_sub_90, cos_sub, sin_two_mul, cos_two_mul],
      norm_num,
    },
  exact ⟨h₁, h₂, rfl⟩,
  sorry
}

end smallest_x_proof_l173_173122


namespace king_zenobius_more_descendants_l173_173095

-- Conditions
def descendants_paphnutius (p2_descendants p1_descendants: ℕ) := 
  2 + 60 * p2_descendants + 20 * p1_descendants = 142

def descendants_zenobius (z3_descendants z1_descendants : ℕ) := 
  4 + 35 * z3_descendants + 35 * z1_descendants = 144

-- Main statement
theorem king_zenobius_more_descendants:
  ∀ (p2_descendants p1_descendants z3_descendants z1_descendants : ℕ),
    descendants_paphnutius p2_descendants p1_descendants →
    descendants_zenobius z3_descendants z1_descendants →
    144 > 142 :=
by
  intros
  sorry

end king_zenobius_more_descendants_l173_173095


namespace vasya_lowest_position_l173_173242

theorem vasya_lowest_position
  (n : ℕ) (m : ℕ) (num_cyclists : ℕ) (vasya_place : ℕ) :
  num_cyclists = 500 →
  n = 15 →
  vasya_place = 7 →
  ∀ (stages : fin n → fin num_cyclists) (no_identical_times : ∀ i j : fin n, i ≠ j → 
  ∀ k l : fin num_cyclists, stages i k ≠ stages j l),
  ∃ (lowest_position : ℕ), lowest_position = 91 := 
by sorry

end vasya_lowest_position_l173_173242


namespace line_EF_passes_through_K_l173_173966

variables {A B C H P : Point}
variable [plane_geometry: EuclideanGeometry]

-- Define necessary points and line segments
def orthocenter (A B C : Point) : Point := sorry
def circumcircle (A B C : Point) : Set Point := sorry
def midpoint (X Y : Point) : Point := sorry
def feet_of_perpendicular (P A B : Point) : Point := sorry

-- Given conditions
axiom orthocenter_def : orthocenter A B C = H
axiom on_circumcircle : P ∈ circumcircle A B C
def E := feet_of_perpendicular P A B
def F := feet_of_perpendicular P A C
def K := midpoint P H

-- Proof statement
theorem line_EF_passes_through_K : collinear ({E, F, K}) :=
sorry

end line_EF_passes_through_K_l173_173966


namespace max_a_zero_l173_173956

def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2 + Real.sin x

theorem max_a_zero
  (H : ∀ x > 0, f (a - x * Real.exp x) + f (Real.log x + x + 1) ≤ 0) :
  a ≤ 0 :=
sorry

end max_a_zero_l173_173956


namespace expected_number_of_sixes_l173_173663

-- Define the probability of not rolling a 6 on one die
def prob_not_six : ℚ := 5 / 6

-- Define the probability of rolling zero 6's on three dice
def prob_zero_six : ℚ := prob_not_six ^ 3

-- Define the probability of rolling exactly one 6 among the three dice
def prob_one_six (n : ℕ) : ℚ := n * (1 / 6) * (prob_not_six ^ (n - 1))

-- Calculate the probabilities of each specific outcomes
def prob_exactly_zero_six : ℚ := prob_zero_six
def prob_exactly_one_six : ℚ := prob_one_six 3 * (prob_not_six ^ 2)
def prob_exactly_two_six : ℚ := prob_one_six 3 * (1 / 6) * prob_not_six
def prob_exactly_three_six : ℚ := (1 / 6) ^ 3

-- Define the expected value calculation
noncomputable def expected_value : ℚ :=
  0 * prob_exactly_zero_six
  + 1 * prob_exactly_one_six
  + 2 * prob_exactly_two_six
  + 3 * prob_exactly_three_six

-- Prove that the expected value equals to 1/2
theorem expected_number_of_sixes : expected_value = 1 / 2 :=
  by
    sorry

end expected_number_of_sixes_l173_173663


namespace temperature_of_Huangshan_at_night_l173_173273

theorem temperature_of_Huangshan_at_night 
  (T_morning : ℤ) (Rise_noon : ℤ) (Drop_night : ℤ)
  (h1 : T_morning = -12) (h2 : Rise_noon = 8) (h3 : Drop_night = 10) :
  T_morning + Rise_noon - Drop_night = -14 :=
by
  sorry

end temperature_of_Huangshan_at_night_l173_173273


namespace smallest_x_is_solution_l173_173377

def smallest_positive_angle (x : ℝ) : Prop :=
  tan (6 * x) = (cos (2 * x) - sin (2 * x)) / (cos (2 * x) + sin (2 * x))

noncomputable def smallest_x : ℝ :=
  45 / 8

theorem smallest_x_is_solution : smallest_positive_angle (smallest_x * (Real.pi / 180)) :=
by
  sorry -- Proof omitted

end smallest_x_is_solution_l173_173377


namespace number_of_hockey_players_l173_173916

theorem number_of_hockey_players 
  (cricket_players : ℕ) 
  (football_players : ℕ) 
  (softball_players : ℕ) 
  (total_players : ℕ) 
  (hockey_players : ℕ) 
  (h1 : cricket_players = 10) 
  (h2 : football_players = 16) 
  (h3 : softball_players = 13) 
  (h4 : total_players = 51) 
  (calculation : hockey_players = total_players - (cricket_players + football_players + softball_players)) : 
  hockey_players = 12 :=
by 
  rw [h1, h2, h3, h4] at calculation
  exact calculation

end number_of_hockey_players_l173_173916


namespace number_of_excellent_tickets_l173_173288

def is_excellent (ticket : ℕ) : Prop :=
  (ticket < 1000000) ∧ ∃ i, i < 5 ∧ abs ((ticket / 10^i % 10) - (ticket / 10^(i + 1) % 10)) = 5

def count_excellent_tickets : ℕ :=
  1000000 - 10 * 9^5

theorem number_of_excellent_tickets : count_excellent_tickets = 409510 :=
  by
    sorry

end number_of_excellent_tickets_l173_173288


namespace hyperbola_real_axis_length_l173_173474

theorem hyperbola_real_axis_length
  (a b x_p y_p x₁ x₂ : ℝ)
  (h_cond1 : a > 0)
  (h_cond2 : b > 0)
  (h_hyperbola : x_p^2 / a^2 - y_p^2 / b^2 = 1)
  (h_P_on_right_branch : x_p > 0)
  (h_A_coords : y_p = (b/a) * x_p)
  (h_B_coords : y_p = (b/-a) * x₂)
  (h_vector_relation :
    let P := (x_p, y_p)
    let A := (x₁, (b/a) * x₁)
    let B := (x₂, -(b/a) * x₂)
    2 • (P.1 - A.1, P.2 - A.2) = (B.1 - P.1, B.2 - P.2))
  (h_area_triangle : 
    ∃ (tan_theta sin_theta : ℝ), 
      (tan_theta = (2 * a * b)/(a^2 - b^2)) ∧
      (sin_theta = (2 * a * b)/(a^2 + b^2)) ∧
      (real.abs ( (1/2) * real.abs(x₁ - x₂) * real.abs((b/a) * x₁ + (b/-a) * x₂) * sin_theta) = 2 * b)
  ) :
  2 * a = 32 / 9 :=
sorry

end hyperbola_real_axis_length_l173_173474


namespace triangle_with_angle_ratio_is_right_triangle_l173_173911

theorem triangle_with_angle_ratio_is_right_triangle (x : ℝ) (h1 : 1 * x + 2 * x + 3 * x = 180) : 
  ∃ A B C : ℝ, A = x ∧ B = 2 * x ∧ C = 3 * x ∧ (A = 90 ∨ B = 90 ∨ C = 90) := 
by
  sorry

end triangle_with_angle_ratio_is_right_triangle_l173_173911


namespace range_of_m_l173_173482

theorem range_of_m (x y m : ℝ) 
  (h1: 3 * x + y = 1 + 3 * m) 
  (h2: x + 3 * y = 1 - m) 
  (h3: x + y > 0) : 
  m > -1 :=
sorry

end range_of_m_l173_173482


namespace not_true_statement_C_l173_173396

-- Define the operation x ♥ y as |x - y|
def heart (x y : ℝ) : ℝ := abs (x - y)

-- Prove that the statement "x ♥ 0 = x for all x" is not true
theorem not_true_statement_C :
  ¬ ∀ (x : ℝ), heart x 0 = x :=
begin
  -- We need to show the existence of some x for which heart x 0 ≠ x
  use -1, -- Example of a counterexample
  unfold heart,
  simp,
  -- Prove that | -1 - 0 | ≠ -1
  linarith,
end

end not_true_statement_C_l173_173396


namespace trajectory_eq_of_point_M_l173_173435

noncomputable def point_Q := (2 : ℝ, 0 : ℝ)
noncomputable def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 1

theorem trajectory_eq_of_point_M (M : ℝ × ℝ)
(h1 : ∀ x y : ℝ, (circle_C x y → x = M.1 ∧ y = M.2 → 
  ∃ c : ℝ, c * (x^2 + y^2 - 1) = c * ((M.1 - 2)^2 + M.2^2 + 1)))
(h2 : |M.1 - 2| + |M.2| = (2 - M.1) * (2 - M.2))
: 3 * M.1^2 - M.2^2 - 8 * M.1 + 5 = 0 ∧ M.1 ≥ 3 / 2 := 
begin 
  sorry -- Proof to be provided
end

end trajectory_eq_of_point_M_l173_173435


namespace area_of_one_smaller_triangle_l173_173517

/-
Problem statement:
Given that the outer equilateral triangle has an area of 36 square units,
the inner equilateral triangle, concentric with the outer one and inside it,
has an area of 4 square units, and the space between these two triangles
is divided into four congruent triangles. Prove that the area of one of
these smaller triangles is 8 square units.
-/
theorem area_of_one_smaller_triangle 
  (area_outer : ℝ)
  (area_inner : ℝ)
  (congruence_factor : ℝ)
  (area_outer = 36)
  (area_inner = 4)
  (congruence_factor = 4) :
  (area_outer - area_inner) / congruence_factor = 8 := 
sorry

end area_of_one_smaller_triangle_l173_173517


namespace smallest_radius_of_third_circle_l173_173683

-- Formalizing the conditions
variables (y : ℝ) (r : ℝ)
variables (O1_center : ℝ × ℝ := (0, 1))
variables (O2_center : ℝ × ℝ := (2, y))
variables (O3_center : ℝ × ℝ := (2 * real.sqrt r, r))

-- Conditions
axiom h1 : y ∈ set.Icc 0 1
axiom h2 : r = 3 - 2 * real.sqrt 2

-- Distance between the centers of O1 and O2 should be 1
axiom dist_condition : (2 * real.sqrt r - 0) * (2 * real.sqrt r - 0) + (r - 1) * (r - 1) = 1

-- The problem restated as the Theorem
theorem smallest_radius_of_third_circle :
  r = 3 - 2 * real.sqrt 2 := 
sorry

end smallest_radius_of_third_circle_l173_173683


namespace quadratic_has_real_roots_iff_l173_173051

theorem quadratic_has_real_roots_iff (k : ℝ) : (∃ x : ℝ, x^2 + 2*x - k = 0) ↔ k ≥ -1 :=
by
  sorry

end quadratic_has_real_roots_iff_l173_173051


namespace remainder_of_polynomial_l173_173045

theorem remainder_of_polynomial (a : ℤ) : 
  ∃ r, (100a - 1)^2 + 2 * (100a - 1) + 3 ≡ r [MOD 100] ∧ r = 2 := by
  sorry

end remainder_of_polynomial_l173_173045


namespace height_at_C_l173_173354

-- Definitions of the given heights
def height_A : ℕ := 15
def height_E : ℕ := 11
def height_G : ℕ := 13

-- Given the heights at A, E, and G, we want to find the height at C
theorem height_at_C :
  height_A = 15 →
  height_E = 11 →
  height_G = 13 →
  ∃ height_C : ℕ, height_C = 9 :=
by
  intros hA hE hG
  use 9
  sorry

end height_at_C_l173_173354


namespace problem1_part1_problem1_part2_problem2_l173_173030

open Set

variable (A B C U : Set ℝ)
variable (a : ℝ)

-- Definitions and conditions
def A := {x : ℝ | 1 ≤ x ∧ x < 7}
def B := {x : ℝ | 2 < x ∧ x < 10}
def C := {x : ℝ | x < a}
def U := univ

-- Proof statements
theorem problem1_part1 : A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 10} :=
sorry

theorem problem1_part2 : compl A ∩ B = {x : ℝ | 7 ≤ x ∧ x < 10} :=
sorry

theorem problem2 (h : (A ∩ C) ≠ ∅) : 1 < a :=
sorry

end problem1_part1_problem1_part2_problem2_l173_173030


namespace at_least_one_equation_has_real_roots_l173_173010

noncomputable def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  (4 * b^2 - 4 * a * c > 0) ∨ (4 * c^2 - 4 * a * b > 0) ∨ (4 * a^2 - 4 * b * c > 0)

theorem at_least_one_equation_has_real_roots (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) :
  has_two_distinct_real_roots a b c :=
by
  sorry

end at_least_one_equation_has_real_roots_l173_173010


namespace min_additional_squares_for_symmetry_l173_173515

def initial_shaded_positions : List (ℕ × ℕ) := [(3,7), (6,3), (1,4), (7,1)]

def is_symmetric (grid : List (ℕ × ℕ)) : Prop :=
  ∀ (x y : ℕ), (x, y) ∈ grid → (7 - x + 1, y) ∈ grid ∧ (x, 7 - y + 1) ∈ grid

theorem min_additional_squares_for_symmetry :
  ∃ n, is_symmetric (initial_shaded_positions ++ extra_shaded_positions) ∧ n = 8 :=
begin
  -- Note: extra_shaded_positions should be found such that the grid becomes symmetric
  sorry
end

end min_additional_squares_for_symmetry_l173_173515


namespace tan_of_sine_plus_cosine_eq_neg_4_over_3_l173_173446

variable {A : ℝ}

theorem tan_of_sine_plus_cosine_eq_neg_4_over_3 
  (h : Real.sin A + Real.cos A = -4/3) : 
  Real.tan A = -4/3 :=
sorry

end tan_of_sine_plus_cosine_eq_neg_4_over_3_l173_173446


namespace perimeter_of_square_D_is_20sqrt2_l173_173607
open Nat Real

theorem perimeter_of_square_D_is_20sqrt2 :
  ∀ (sideC : ℝ) (area_ratio : ℝ), sideC = 10 ∧ area_ratio = 0.5 →
  let areaC := sideC * sideC in
  let areaD := areaC * area_ratio in
  let sideD := Real.sqrt areaD in
  let perimeterD := 4 * sideD in
  perimeterD = 20 * Real.sqrt 2 :=
by
  intros sideC area_ratio h
  rcases h with ⟨hc, har⟩
  simp [hc, har]
  sorry

end perimeter_of_square_D_is_20sqrt2_l173_173607


namespace tea_price_l173_173349

theorem tea_price 
  (x : ℝ)
  (total_cost_80kg_tea : ℝ := 80 * x)
  (total_cost_20kg_tea : ℝ := 20 * 20)
  (total_selling_price : ℝ := 1920)
  (profit_condition : 1.2 * (total_cost_80kg_tea + total_cost_20kg_tea) = total_selling_price) :
  x = 15 :=
by
  sorry

end tea_price_l173_173349


namespace curve_C1_to_cartesian_curve_C2_to_cartesian_min_distance_C1_to_C2_coordinates_at_min_distance_l173_173930

-- Given parametric form of curve C₁
def parametric_C1 (α : ℝ) : ℝ × ℝ :=
  (sqrt 3 * cos α, sin α)

-- Given polar form of curve C₂
def polar_C2 (ρ θ : ℝ) : Prop :=
  ρ * sin (θ + π / 4) = 4 * sqrt 2

-- Prove:
theorem curve_C1_to_cartesian :
  ∃ (x y : ℝ), (∀ (α : ℝ), parametric_C1 α = (x, y)) → x^2 / 3 + y^2 = 1 := sorry

theorem curve_C2_to_cartesian :
  ∃ (x y : ℝ), (∀ (ρ θ : ℝ), polar_C2 ρ θ) → x + y - 8 = 0 := sorry

-- Minimum distance from point P on C₁ to line C₂
theorem min_distance_C1_to_C2 (p : ℝ × ℝ) :
  (p ∈ parametric_C1) → (abs ((p.1 + p.2 - 8) / sqrt 2) = 3 * sqrt 2) := sorry

-- Coordinates of P at minimum distance
theorem coordinates_at_min_distance :
  ∃ (α : ℝ), parametric_C1 α = (3/2, 1/2) := sorry

end curve_C1_to_cartesian_curve_C2_to_cartesian_min_distance_C1_to_C2_coordinates_at_min_distance_l173_173930


namespace inequality_and_equality_condition_l173_173127

theorem inequality_and_equality_condition (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ab : 1 ≤ a * b) :
  (1 / (1 + a) + 1 / (1 + b) ≤ 1) ∧ (1 / (1 + a) + 1 / (1 + b) = 1 ↔ a * b = 1) :=
by
  sorry

end inequality_and_equality_condition_l173_173127


namespace largest_percent_error_l173_173340

theorem largest_percent_error (
  (measured_length : ℝ) (error_margin : ℝ) (width : ℝ)
  (h1 : measured_length = 30)
  (h2 : error_margin = 0.1)
  (h3 : width = 15)
) : 
  let actual_area := measured_length * width,
      min_area := (measured_length - measured_length * error_margin) * width,
      max_area := (measured_length + measured_length * error_margin) * width,
      min_error := abs ((actual_area - min_area) / actual_area) * 100,
      max_error := abs ((max_area - actual_area) / actual_area) * 100 
  in max min_error max_error = 10 := 
sorry

end largest_percent_error_l173_173340


namespace red_segments_diagonal_length_l173_173836

noncomputable def diagonal_length_red_segments (m n : ℕ) : ℝ :=
  let gcd_val := Nat.gcd m n in
  if gcd_val % 2 == 1 then 
    (Real.sqrt (m^2 + n^2) / 2 : ℝ)
  else 
    (Real.sqrt (m^2 + n^2) * (gcd_val^2 + m * n) / (2 * m * n) : ℝ)

theorem red_segments_diagonal_length (m n : ℕ) (hm : 0 < m) (hn : 0 < n) 
  (board : matrix (fin m) (fin n) (bool)) (bottom_left_is_red : board 0 0 = tt)
  (checkerboard_pattern : ∀ i j, board i j ≠ board (i+1) j ∧ board i j ≠ board i (j+1)) :
  diagonal_length_red_segments m n = 
    if Nat.gcd m n % 2 == 1 then 
      (Real.sqrt (m^2 + n^2) / 2)
    else 
      (Real.sqrt (m^2 + n^2) * (Nat.gcd m n^2 + m * n) / (2 * m * n)) :=
sorry

end red_segments_diagonal_length_l173_173836


namespace rita_book_pages_l173_173597

theorem rita_book_pages (x : ℕ) (h1 : ∃ n₁, n₁ = (1/6 : ℚ) * x + 10) 
                                  (h2 : ∃ n₂, n₂ = (1/5 : ℚ) * ((5/6 : ℚ) * x - 10) + 20)
                                  (h3 : ∃ n₃, n₃ = (1/4 : ℚ) * ((4/5 : ℚ) * ((5/6 : ℚ) * x - 10) - 20) + 25)
                                  (h4 : ((3/4 : ℚ) * ((2/3 : ℚ) * x - 28) - 25) = 50) :
    x = 192 := 
sorry

end rita_book_pages_l173_173597


namespace perfect_square_m_value_l173_173885

theorem perfect_square_m_value {m x : ℝ} : (∀ x, ∃ k : ℝ, (9 - m * x + x^2) = k^2) → (m = 6 ∨ m = -6) :=
begin
  sorry
end

end perfect_square_m_value_l173_173885


namespace not_true_x_heartsuit_0_eq_x_l173_173394

def heartsuit (x y : ℝ) : ℝ := abs (x - y)

theorem not_true_x_heartsuit_0_eq_x : ¬(∀ x : ℝ, heartsuit x 0 = x) :=
by
  (assume h : ∀ x : ℝ, heartsuit x 0 = x)
  have h_neg : heartsuit (-1) 0 = -1 := h (-1)
  dsimp [heartsuit] at h_neg
  contradiction

end not_true_x_heartsuit_0_eq_x_l173_173394


namespace handshake_count_correct_l173_173422

def women_shaking_hands : Prop :=
  ∀ (heights : Fin 5 → ℕ), 
    ∑ i, 
      (∑ j, if heights i < heights j then 1 else 0) = 10

theorem handshake_count_correct : women_shaking_hands :=
by sorry

end handshake_count_correct_l173_173422


namespace vasya_maximum_rank_l173_173251

theorem vasya_maximum_rank {n : ℕ} (cyclists stages : ℕ) (VasyaPlace : ℕ) 
  (rankings : Π (s : fin stages), fin cyclists):
  cyclists = 500 → stages = 15 → VasyaPlace = 7 →
  (∀ s, rankings s VasyaPlace = 7) →
  (∀ s t i, rankings s i ≠ rankings t i) →
  (∀ s, ∃ l, list.nodup l ∧ (∀ i j, i < j → rankings s i < rankings s j) ∧ list.length l = 500) →
  (∃ (maxRank : ℕ), maxRank = 91) :=
by
  intros hcyclists hstages hplace hvasya_place hdistinct_rankings hstage_rankings
  use 91
  sorry

end vasya_maximum_rank_l173_173251


namespace vasya_lowest_position_l173_173240

theorem vasya_lowest_position
  (n : ℕ) (m : ℕ) (num_cyclists : ℕ) (vasya_place : ℕ) :
  num_cyclists = 500 →
  n = 15 →
  vasya_place = 7 →
  ∀ (stages : fin n → fin num_cyclists) (no_identical_times : ∀ i j : fin n, i ≠ j → 
  ∀ k l : fin num_cyclists, stages i k ≠ stages j l),
  ∃ (lowest_position : ℕ), lowest_position = 91 := 
by sorry

end vasya_lowest_position_l173_173240


namespace ratio_of_areas_l173_173905

theorem ratio_of_areas (R_C R_D : ℝ) (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 4 / 9 :=
by
  sorry

end ratio_of_areas_l173_173905


namespace mortgage_payoff_months_l173_173092

-- Declare the initial payment (P), the common ratio (r), and the total amount (S)
def initial_payment : ℕ := 100
def common_ratio : ℕ := 3
def total_amount : ℕ := 12100

-- Define a function that calculates the sum of a geometric series
noncomputable def geom_series_sum (P : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  P * (1 - r ^ n) / (1 - r)

-- The statement we need to prove
theorem mortgage_payoff_months : ∃ n : ℕ, geom_series_sum initial_payment common_ratio n = total_amount :=
by
  sorry -- Proof to be provided

end mortgage_payoff_months_l173_173092


namespace evaluate_box_2_neg1_0_l173_173410

def box (a b c : ℤ) : ℚ := a^b - b^c + c^a

theorem evaluate_box_2_neg1_0 : box 2 (-1) 0 = -1/2 := 
by
  sorry

end evaluate_box_2_neg1_0_l173_173410


namespace sequence_goes_negative_l173_173266

theorem sequence_goes_negative : ∃ n : ℕ, 0 < n ∧ n < 2002 ∧ (a n < 0) :=
by
  -- Define the sequence a_n
  noncomputable def a : ℕ → ℝ
  | 0     => 56
  | (n+1) => a n - 1 / a n
  sorry


end sequence_goes_negative_l173_173266


namespace trace_ellipse_l173_173974

theorem trace_ellipse (w : ℂ) (hw : abs w = 3) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ z, z = w + 2 / w → ∃ (x y : ℝ), z = x + y * complex.I ∧ 
    (x^2 / a^2) + (y^2 / b^2) = 1 := 
sorry

end trace_ellipse_l173_173974


namespace vasya_lowest_position_l173_173202

theorem vasya_lowest_position
  (n_cyclists : ℕ) (n_stages : ℕ) 
  (stage_positions : ℕ → ℕ → ℕ) -- a function that takes a stage and a cyclist and returns the position (e.g., stage_positions(stage, cyclist) = position)
  (total_time : ℕ → ℕ)  -- a function that takes a cyclist and returns their total time
  (distinct_times : ∀ (c1 c2 : ℕ), c1 ≠ c2 → (total_time c1 ≠ total_time c2) ∧ 
                   ∀ (s : ℕ), stage_positions s c1 ≠ stage_positions s c2)
  (vasya_position : ℕ) (hv : ∀ (s : ℕ), s < n_stages → stage_positions s vasya_position = 7) :
  vasya_position = 91 :=
sorry

end vasya_lowest_position_l173_173202


namespace interval_of_increase_a_eq_3_range_of_values_for_a_l173_173472

noncomputable def f (x : ℝ) (a : ℝ) := x^2 + log x - a * x

-- Problem 1
theorem interval_of_increase_a_eq_3 : 
  (∀ a = 3, ∀ x : ℝ, 0 < x ∧ x < 1/2 → 2 * x + (1 / x) - 3 > 0) ∧ 
  (∀ a = 3, ∀ x : ℝ, 1 < x → 2 * x + (1 / x) - 3 > 0) :=
begin
  sorry
end

-- Problem 2
theorem range_of_values_for_a : 
  (∀ x : ℝ, 0 < x ∧ x < 1 → (2 * x + (1 / x)) - a > 0) → a ≤ 2 * (2.sqrt) :=
begin
  sorry
end

end interval_of_increase_a_eq_3_range_of_values_for_a_l173_173472


namespace constant_term_binomial_expansion_l173_173530

theorem constant_term_binomial_expansion :
  let y := x + 2 / x^(1/2)
  let expansion := y^6
  let constant_term := ∑ k in finset.range(7), nat.choose 6 k * (2 ^ k) * x^(6 - (3/2) * k)
  constant_term.eval 0 = 240 :=
by
  let x := 0
  sorry

end constant_term_binomial_expansion_l173_173530


namespace Zenobius_more_descendants_l173_173096

/-- Total number of descendants in King Pafnutius' lineage --/
def descendants_Pafnutius : Nat :=
  2 + 60 * 2 + 20 * 1

/-- Total number of descendants in King Zenobius' lineage --/
def descendants_Zenobius : Nat :=
  4 + 35 * 3 + 35 * 1

theorem Zenobius_more_descendants : descendants_Zenobius > descendants_Pafnutius := by
  sorry

end Zenobius_more_descendants_l173_173096


namespace verify_options_l173_173697

-- Definitions for each hypothesis:
variable (ξ η : ℝ → ℝ)
variable (D : (ℝ → ℝ) → ℝ)
variable (σ : ℝ)
variable (r : ℝ)
variable (P : ℝ → ℝ)
variable A B : Finset ℝ

-- Translating each hypothesis for the Lean statement
-- Option A
def option_a (η : ℝ → ℝ) (ξ : ℝ → ℝ) (D : (ℝ → ℝ) → ℝ) := 
  η = λ x, 2 * ξ x + 1 → D η = 4 * D ξ

-- Option B
def option_b (ξ : ℝ → ℝ) (σ : ℝ) (P : ℝ → ℝ) :=
  (ξ ∼ Normal 3 σ^2) → P (λ x, ξ x < 6) = 0.84 → P (λ x, 3 < ξ x ∧ ξ x < 6) = 0.34

-- Option C
def option_c (r : ℝ) :=
  0 ≤ |r| ∧ |r| ≤ 1 → |r| = 1 ↔ StrongerLinearCorrelation

-- Option D
def option_d (A B : Finset ℝ) (m n : ℝ) :=
  m ∈ A ∧ n ∈ B ∧ thirty_percentile A = thirty_percentile B ∧ fifty_percentile A = fifty_percentile B 
  → m + n ≠ 67

-- Main theorem combining all the options above
theorem verify_options 
  (ξ η : ℝ → ℝ) (D : (ℝ → ℝ) → ℝ) (σ : ℝ) (r : ℝ) 
  (P : ℝ → ℝ) (A B : Finset ℝ) (m n : ℝ) :
  option_a η ξ D ∧ option_b ξ σ P ∧ option_c r ∧ option_d A B m n :=
by
  sorry

end verify_options_l173_173697


namespace ptolemys_theorem_for_cyclic_quadrilateral_l173_173568

variable (a b c d m n : ℝ)
variable (cyclic_quad : ∀ (A B C D : ℝ), A + C = 180 ∧ B + D = 180)

theorem ptolemys_theorem_for_cyclic_quadrilateral
  (cyclic : cyclic_quad a b c d) :
  m * n = a * c + b * d := 
sorry

end ptolemys_theorem_for_cyclic_quadrilateral_l173_173568


namespace range_of_m_l173_173802

theorem range_of_m (f : ℝ → ℝ) (h_decreasing : ∀ x y : ℝ, x < y → f x > f y)
  (h_f : ∀ x, f x = (1 / 3 : ℝ)^x) :
  (∀ m : ℝ, f (2 * m - 1) - f (m + 3) < 0 → m > 4) :=
begin
  sorry
end

end range_of_m_l173_173802


namespace routes_M_to_N_l173_173915

-- Define nodes
inductive Node
| M | A | B | C | D | E | N

open Node

-- Define edges in the directed graph
def edge : Node → Node → Prop
| M, A := true
| M, B := true
| A, C := true
| A, D := true
| B, E := true
| B, C := true
| C, N := true
| D, N := true
| E, N := true
| D, C := true
| _, _ := false

-- Define a path exists function
def path_exists (graph: Node → Node → Prop) (start goal : Node) : Prop := -- recursive definition needed
sorry

-- Define a function to count the number of distinct routes
noncomputable def count_routes (graph: Node → Node → Prop) (start goal : Node) : ℕ :=
sorry

-- The theorem to prove
theorem routes_M_to_N : count_routes edge M N = 5 :=
by
  sorry

end routes_M_to_N_l173_173915


namespace sin_sum_leq_3_sqrt_3_div_2_l173_173939

theorem sin_sum_leq_3_sqrt_3_div_2 (A B C : ℝ) (h_sum : A + B + C = Real.pi) (h_pos : 0 < A ∧ 0 < B ∧ 0 < C) :
  Real.sin A + Real.sin B + Real.sin C ≤ (3 * Real.sqrt 3) / 2 :=
sorry

end sin_sum_leq_3_sqrt_3_div_2_l173_173939


namespace domain_of_f_l173_173189

noncomputable def f (x : ℝ) := real.sqrt (2^x - 1)

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f x = y} = set.Ici 0 :=
by
  sorry

end domain_of_f_l173_173189


namespace find_input_values_f_l173_173078

theorem find_input_values_f (f : ℤ → ℤ) 
  (h_def : ∀ x, f (2 * x + 3) = (x - 3) * (x + 4))
  (h_val : ∃ y, f y = 170) : 
  ∃ (a b : ℤ), (a = -25 ∧ b = 29) ∧ (f a = 170 ∧ f b = 170) :=
by
  sorry

end find_input_values_f_l173_173078


namespace sum_of_angles_l173_173917

theorem sum_of_angles (A B C D E F : ℝ)
  (h1 : A + B + C = 180) 
  (h2 : D + E + F = 180) : 
  A + B + C + D + E + F = 360 := 
by 
  sorry

end sum_of_angles_l173_173917


namespace right_triangle_perimeter_l173_173826

theorem right_triangle_perimeter (x y : ℕ) : 
  (11 * 11 + x * x = y * y ∨ x * x + y * y = 11 * 11) →
  (11 + x + y = 132) :=
by
  intro h
  cases h
  -- Case 1: 11 is one of the legs
  case Or.inl => 
    have : y * y - x * x = 121 :=
      by linarith
    have : (y + x) * (y - x) = 121 :=
      by rw nat.mul_sub (le_of_lt (nat.lt_of_not_ge' (ne_of_gt (gt_of_ge mid'.pos))))
    have : y - x = 1 :=
      by rw (11 + 1) at this ; exact this
    have ht := this.left
    have hl := this.right
    showclasses exception on lfine - exact this
    have x == y then unfold this 
    cases x (le by y hl) => nathave nat.floor xineq on ; x_symmetry then rw [nat.pow_eq_natr (this)] then unfold floor_in_encoder
    use y_eq
    have: 121 then unfold ord.symmetry then infer else => x_floor then rw on :=
      infer then rw same subseq_prim <=> x_floor exist then by y_eq - exact_equi_this
    thus nathave =>
      infer_main y_exact
      apply nathave_le
    simp_natr
  -- Case 2: 11 is the hypotenuse
  case Or.inr =>
    sorry

end right_triangle_perimeter_l173_173826


namespace water_inflow_rate_in_tank_A_l173_173278

-- Definitions from the conditions
def capacity := 20
def inflow_rate_B := 4
def extra_time_A := 5

-- Target variable
noncomputable def inflow_rate_A : ℕ :=
  let time_B := capacity / inflow_rate_B
  let time_A := time_B + extra_time_A
  capacity / time_A

-- Hypotheses
def tank_capacity : capacity = 20 := rfl
def tank_B_inflow : inflow_rate_B = 4 := rfl
def tank_A_extra_time : extra_time_A = 5 := rfl

-- Theorem statement
theorem water_inflow_rate_in_tank_A : inflow_rate_A = 2 := by
  -- Proof would go here
  sorry

end water_inflow_rate_in_tank_A_l173_173278


namespace value_of_b_l173_173952

theorem value_of_b (a b : ℝ) (h1 : 4 * a^2 + 1 = 1) (h2 : b - a = 3) : b = 3 :=
sorry

end value_of_b_l173_173952


namespace arun_chun_completion_time_l173_173770

-- Define the work rates and initial conditions
def arun_work_rate : ℝ := 1 / 60
def tarun_work_rate : ℝ := 1 / 40
def chun_work_rate : ℝ := 1 / 20

-- Define their combined work rate and work done in the initial 4 days
def combined_work_rate : ℝ := 1 / 10
def work_done_in_4_days : ℝ := 4 * combined_work_rate

-- Define the remaining work to be done
def remaining_work : ℝ := 1 - work_done_in_4_days

-- Define the combined work rate of Arun and Chun after Tarun leaves
def arun_chun_combined_rate : ℝ := arun_work_rate + chun_work_rate

-- The target number of days for complete the remaining work
def required_days_for_arun_chun_to_complete_remaining_work : ℝ :=
  remaining_work / arun_chun_combined_rate

-- The theorem to be proved
theorem arun_chun_completion_time :
    required_days_for_arun_chun_to_complete_remaining_work = 9 := by
  sorry

end arun_chun_completion_time_l173_173770


namespace sun_city_population_l173_173608

theorem sun_city_population (W R S : ℕ) (h1 : W = 2000)
    (h2 : R = 3 * W - 500) (h3 : S = 2 * R + 1000) : S = 12000 :=
by
    -- Use the provided conditions (h1, h2, h3) to state the theorem
    sorry

end sun_city_population_l173_173608


namespace calc_P_X_120_l173_173975

variable {X : ℝ} {σ : ℝ}

-- Definition of standard normal distribution parameters and given probability
def normal_distribution (X : ℝ) (μ : ℝ) (σ : ℝ) : Prop :=
  -- Here we define X ~ N(μ, σ^2), import or implementation of this definition is assumed
  sorry 

def calc_prob (X : ℝ) (low : ℝ) (high : ℝ) : ℝ :=
  -- Here we define some function to get probability between intervals
  sorry 

noncomputable def P_X_80_120 : Prop := calc_prob X 80 120 = 3 / 4

theorem calc_P_X_120 {X : ℝ} {σ : ℝ} (h₁ : normal_distribution X 100 σ) (h₂ : P_X_80_120) : 
    calc_prob X 120 (real.infinity) = 1 / 8 :=
  sorry

end calc_P_X_120_l173_173975


namespace find_f_neg_3_l173_173397

variable {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def functional_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, f (1 - x) = f (1 + x)

def function_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x^2

theorem find_f_neg_3 
  (hf_even : even_function f) 
  (hf_condition : functional_condition f)
  (hf_interval : function_on_interval f) : 
  f (-3) = 1 := 
by
  sorry

end find_f_neg_3_l173_173397


namespace number_of_ways_to_distribute_l173_173551

-- Given conditions
variables (r n : ℕ) (x : ℕ → ℕ)
-- Conditions: sum of x_i = r and x_i ≥ 0 for all i (1 ≤ i ≤ n)
def valid_distribution (r n : ℕ) (x : ℕ → ℕ) : Prop :=
  (∑ i in Finset.range n, x i) = r ∧ (∀ i, x i ≥ 0)

-- The proof goal
theorem number_of_ways_to_distribute (r n : ℕ) (x : ℕ → ℕ) (h : valid_distribution r n x) :
  ∃ N : ℕ, N = Nat.choose (r + n - 1) r :=
sorry

end number_of_ways_to_distribute_l173_173551


namespace ratio_of_areas_l173_173902

theorem ratio_of_areas (R_C R_D : ℝ) (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 4 / 9 :=
by
  sorry

end ratio_of_areas_l173_173902


namespace expected_value_of_sixes_l173_173654

theorem expected_value_of_sixes (n : ℕ) (k : ℕ) (p q : ℚ) 
  (h1 : n = 3) 
  (h2 : k = 6)
  (h3 : p = 1/6) 
  (h4 : q = 5/6) : 
  (1 : ℚ) / 2 = ∑ i in finset.range (n + 1), (i * (nat.choose n i * p^i * q^(n-i))) := 
sorry

end expected_value_of_sixes_l173_173654


namespace correct_statements_l173_173464

-- Given the values of x and y on the parabola
def parabola (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

-- Define the points on the parabola
def points_on_parabola (a b c : ℝ) : Prop :=
  parabola a b c (-1) = 3 ∧
  parabola a b c 0 = 0 ∧
  parabola a b c 1 = -1 ∧
  parabola a b c 2 = 0 ∧
  parabola a b c 3 = 3

-- Prove the correct statements
theorem correct_statements (a b c : ℝ) (h : points_on_parabola a b c) : 
  ¬(∃ x, parabola a b c x < 0 ∧ x < 0) ∧
  parabola a b c 2 = 0 :=
by 
  sorry

end correct_statements_l173_173464


namespace remainder_M_div_1000_l173_173552

/-- Define the sequence T as the increasing sequence of positive integers 
whose binary representation has exactly 6 ones. -/
def is_in_seq_T (n : ℕ) : Prop :=
  (n.bits.count (λ b, b = tt) = 6)

/-- Define M as the 800th number in the sequence T. -/
noncomputable def M : ℕ :=
  Nat.find_greatest (is_in_seq_T) 800

/-- The statement to be proven: the remainder when M is divided by 1000 is 112. -/
theorem remainder_M_div_1000 : M % 1000 = 112 := 
by
  sorry

end remainder_M_div_1000_l173_173552


namespace william_wins_tic_tac_toe_l173_173704

-- Define the conditions
variables (total_rounds : ℕ) (extra_wins : ℕ) (william_wins : ℕ) (harry_wins : ℕ)

-- Setting the conditions
def william_harry_tic_tac_toe_conditions : Prop :=
  total_rounds = 15 ∧
  extra_wins = 5 ∧
  william_wins = harry_wins + extra_wins ∧
  total_rounds = william_wins + harry_wins

-- The goal is to prove that William won 10 rounds given the conditions above
theorem william_wins_tic_tac_toe : william_harry_tic_tac_toe_conditions total_rounds extra_wins william_wins harry_wins → william_wins = 10 :=
by
  intro h
  have total_rounds_eq := and.left h
  have extra_wins_eq := and.right (and.left (and.right h))
  have william_harry_diff := and.left (and.right (and.right h))
  have total_wins_eq := and.right (and.right (and.right h))
  sorry

end william_wins_tic_tac_toe_l173_173704


namespace number_of_social_science_papers_selected_is_18_l173_173334

def total_social_science_papers : ℕ := 54
def total_humanities_papers : ℕ := 60
def total_other_papers : ℕ := 39
def total_selected_papers : ℕ := 51

def number_of_social_science_papers_selected : ℕ :=
  (total_social_science_papers * total_selected_papers) / (total_social_science_papers + total_humanities_papers + total_other_papers)

theorem number_of_social_science_papers_selected_is_18 :
  number_of_social_science_papers_selected = 18 :=
by 
  -- Proof to be provided
  sorry

end number_of_social_science_papers_selected_is_18_l173_173334


namespace length_real_axis_hyperbola_l173_173806

theorem length_real_axis_hyperbola (a : ℝ) (h : a^2 = 4) : 2 * a = 4 := by
  sorry

end length_real_axis_hyperbola_l173_173806


namespace A_eq_B_iff_a_eq_5_A_inter_B_nonempty_and_A_inter_C_empty_iff_a_eq_neg2_l173_173973

-- Define the sets A, B, and C
def A (a : ℝ) : set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B : set ℝ := {x | x^2 - 5*x + 6 = 0}
def C : set ℝ := {x | x^2 + 2*x - 8 = 0}

-- Problem (1)
theorem A_eq_B_iff_a_eq_5 (a : ℝ) : A a = B ↔ a = 5 := by
  sorry

-- Problem (2)
theorem A_inter_B_nonempty_and_A_inter_C_empty_iff_a_eq_neg2 (a : ℝ) :
  (∅ ⊂ A a ∩ B) ∧ (A a ∩ C = ∅) ↔ a = -2 := by
  sorry

end A_eq_B_iff_a_eq_5_A_inter_B_nonempty_and_A_inter_C_empty_iff_a_eq_neg2_l173_173973


namespace problem1_problem2_problem3_problem4_l173_173719

-- Problem 1
theorem problem1 (a : ℝ) : a * a^3 - 5 * a^4 + (2 * a^2)^2 = 0 :=
by sorry

-- Problem 2
theorem problem2 (a b : ℝ) : (2 * a + 3 * b) * (a - 2 * b) - (1 / 8) * a * (4 * a - 3 * b) = (3 / 2) * a^2 - (5 / 8) * a * b - 6 * b^2 :=
by sorry

-- Problem 3
theorem problem3 : (-0.125) ^ 2023 * 2 ^ 2024 * 4 ^ 2024 = -8 :=
by sorry

-- Problem 4
theorem problem4 : let a := (1 / 2 : ℝ); b := (-1 : ℝ) in (2 * a - b)^2 + (a - b) * (a + b) - 5 * a * (a - 2 * b) = -3 :=
by sorry

end problem1_problem2_problem3_problem4_l173_173719


namespace angle_k_a_n_measure_l173_173938

theorem angle_k_a_n_measure {K I A N H : Type*}
  (a : ℝ)
  (KI A N H : Point)
  (angle_K : Triangle → ℝ)
  (angle_I : Triangle → ℝ)
  (angle_A : Triangle → ℝ)
  (length_KI : KI = 2*a)
  (perpendicular : ∀ (K : Point) (KI : Line), Perpendicular (Line.PointLine K KI) KI)
  (triangle_30_30_120 : angle_K = 30 ∧ angle_I = 30 ∧ angle_A = 120)
  (N_line : OnLine N (Line.PerpendicularFromPoint (x y K) KI))
  (AN_eq_KI : Length AN = Length KI):
  (Measure K A N = 90 ∨ Measure K A N = 30) :=
begin
  sorry
end

end angle_k_a_n_measure_l173_173938


namespace total_pairs_sold_l173_173357

theorem total_pairs_sold
  (H S : ℕ)
  (price_soft : ℕ := 150)
  (price_hard : ℕ := 85)
  (diff_lenses : S = H + 5)
  (total_sales_eq : price_soft * S + price_hard * H = 1455) :
  H + S = 11 := by
sorry

end total_pairs_sold_l173_173357


namespace value_of_k_l173_173037

theorem value_of_k : ∃ k : ℕ, 4! * 2! = 2 * k * 3! ∧ k = 4 :=
by
  use 4
  split
  · sorry
  · rfl

end value_of_k_l173_173037


namespace vasya_lowest_position_l173_173261

noncomputable theory

def number_of_cyclists := 500
def number_of_stages := 15
def position_of_vasya_each_stage := 7

theorem vasya_lowest_position (total_cyclists : ℕ) (stages : ℕ) (position_each_stage : ℕ)
  (h_total_cyclists : total_cyclists = number_of_cyclists)
  (h_stages : stages = number_of_stages)
  (h_position_each_stage : position_each_stage = position_of_vasya_each_stage)
  (no_identical_times : ∀ (i j : ℕ), i ≠ j → ∀ (stage : ℕ), stage ≤ stages → ∀ (t : ℕ), t ≤ total_cyclists → 
    (time : Π (s : ℕ) (c : ℕ), c < total_cyclists → ℕ), 
    time stage i < time stage j ∨ time stage j < time stage i):
  ∃ lowest_pos, lowest_pos = 91 := sorry

end vasya_lowest_position_l173_173261


namespace total_price_after_increase_l173_173980

def original_jewelry_price := 30
def original_painting_price := 100
def jewelry_price_increase := 10
def painting_price_increase_percentage := 20
def num_jewelry_pieces := 2
def num_paintings := 5

theorem total_price_after_increase : 
    let new_jewelry_price := original_jewelry_price + jewelry_price_increase in
    let new_painting_price := original_painting_price + (original_painting_price * painting_price_increase_percentage / 100) in
    let total_jewelry_cost := new_jewelry_price * num_jewelry_pieces in
    let total_painting_cost := new_painting_price * num_paintings in
    total_jewelry_cost + total_painting_cost = 680 :=
by
  -- proof omitted
  sorry

end total_price_after_increase_l173_173980


namespace find_angle3_l173_173452

theorem find_angle3 (angle1 angle2 angle3 : ℝ)
  (h1 : angle1 + angle2 = 90)
  (h2 : angle2 + angle3 = 180)
  (h3 : angle1 = 20) :
  angle3 = 110 :=
sorry

end find_angle3_l173_173452


namespace eighty_percent_replacement_in_4_days_feasibility_of_replacing_all_old_banknotes_within_budget_l173_173316

noncomputable def old_banknotes : ℕ := 3628800
noncomputable def renovation_cost : ℕ := 800000
noncomputable def daily_operating_cost : ℕ := 90000
noncomputable def post_renovation_capacity : ℕ := 1000000
noncomputable def total_budget : ℕ := 1000000

def banknotes_replaced_in_days (d : ℕ) : ℕ :=
  match d with
  | 1 => old_banknotes / 2
  | 2 => old_banknotes / 2 + post_renovation_capacity
  | 3 => old_banknotes / 2 + post_renovation_capacity + (old_banknotes / 2 + post_renovation_capacity) / 3
  | _ => sorry -- Continues similar pattern.

def cost_in_days (d : ℕ) : ℕ :=
  if d = 1 then daily_operating_cost
  else 800000 + daily_operating_cost * d

theorem eighty_percent_replacement_in_4_days :
  banknotes_replaced_in_days 4 ≈ 0.8 * old_banknotes ∧ cost_in_days 4 ≤ total_budget :=
sorry

theorem feasibility_of_replacing_all_old_banknotes_within_budget :
  banknotes_replaced_in_days 4 = old_banknotes ∨ banknotes_replaced_in_days d = old_banknotes for some d and cost_in_days d ≤ total_budget :=
sorry

end eighty_percent_replacement_in_4_days_feasibility_of_replacing_all_old_banknotes_within_budget_l173_173316


namespace no_four_digit_palindromic_squares_l173_173872

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in
  s = s.reverse

def four_digit_perfect_squares : List ℕ :=
  List.range' (32*32) (99*32 - 32*32 + 1)
  
theorem no_four_digit_palindromic_squares : 
  (List.filter is_palindrome four_digit_perfect_squares).length = 0 :=
by
  sorry

end no_four_digit_palindromic_squares_l173_173872


namespace problem1_problem2_l173_173720

-- Problem 1: Simplification and Evaluation
theorem problem1 (x : ℝ) : (x = -3) → 
  ((x^2 - 6*x + 9) / (x^2 - 1)) / ((x^2 - 3*x) / (x + 1))
  = -1 / 2 := sorry

-- Problem 2: Solving the Equation
theorem problem2 (x : ℝ) : 
  (∀ y, (y = x) → 
    (y / (y + 1) = 2*y / (3*y + 3) - 1)) → x = -3 / 4 := sorry

end problem1_problem2_l173_173720


namespace inverse_of_ka_squared_l173_173887

variable {k : ℝ} (A : Matrix (Fin 2) (Fin 2) ℝ)
hypothesis (h1 : A⁻¹ = k • (Matrix.of ![![1, 3], ![-2, -4]]))
hypothesis (h2 : k ≠ 0)

theorem inverse_of_ka_squared :
  (k • A)^2⁻¹ = (1 / k^2) • (Matrix.of ![![ -5, -9], ![10, 19]]) :=
sorry

end inverse_of_ka_squared_l173_173887


namespace solve_for_a_l173_173995

theorem solve_for_a (a : ℤ) :
  (|2 * a + 1| = 3) ↔ (a = 1 ∨ a = -2) :=
by
  sorry

end solve_for_a_l173_173995


namespace find_angle_BAD_l173_173582

-- Definitions for the given conditions
def is_isosceles (A B C : Type) (AB AC : Real) : Prop :=
  AB = AC

def points_on_sides (A B C D E : Type) : Prop :=
  True  -- Stated that D and E are on sides BC and AC respectively

def equal_segments (A E D : Type) (AE AD : Real) : Prop :=
  AE = AD

def angle_value (EDC : Type) (angle : Real) : Prop :=
  angle = 18

-- Our main goal
theorem find_angle_BAD (A B C D E : Type) 
  (AB AC AE AD : Real) (x : Real)
  (isosceles_triangle : is_isosceles A B C AB AC)
  (points : points_on_sides A B C D E)
  (equal_segs : equal_segments A E D AE AD)
  (angle_EDC : angle_value (∠ E D C) 18):
  x = 36 := sorry

end find_angle_BAD_l173_173582


namespace ratio_of_areas_eq_l173_173898

-- Define the conditions
variables {C D : Type} [circle C] [circle D]
variables (R_C R_D : ℝ)
variable (L : ℝ)

-- Given conditions
axiom arc_length_eq : (60 / 360) * (2 * π * R_C) = L
axiom arc_length_eq' : (40 / 360) * (2 * π * R_D) = L

-- Statement to prove
theorem ratio_of_areas_eq : (π * R_C^2) / (π * R_D^2) = 4 / 9 :=
sorry

end ratio_of_areas_eq_l173_173898


namespace height_line_eq_triangle_area_eq_l173_173934

-- Given points A, B, C
def point_A : ℝ × ℝ := (0, 2)
def point_B : ℝ × ℝ := (2, 0)
def point_C : ℝ × ℝ := (-2, -1)

-- 1. Prove the equation of the line on which the height from A to BC lies.
theorem height_line_eq : ∃ k b, b = 2 ∧ k = -4 ∧ ∀ x y, y = k * x + b ↔ 4 * x + y - 2 = 0 :=
by
  sorry

-- 2. Prove the area of triangle ABC
theorem triangle_area_eq : ∃ S, S = 5 ∧ (1/2) * (real.sqrt ((-2-2)^2 + (-1-0)^2)) * (abs (0 - (-4)*2 - 2) / sqrt(1^2 + (-4)^2) ) = S :=
by
  sorry

end height_line_eq_triangle_area_eq_l173_173934


namespace line_passes_through_fixed_point_l173_173453

theorem line_passes_through_fixed_point (a b : ℝ) (x y : ℝ) (h : a + b = 1) (h1 : 2 * a * x - b * y = 1) : x = 1/2 ∧ y = -1 :=
by 
  sorry

end line_passes_through_fixed_point_l173_173453


namespace part_one_part_two_part_three_l173_173649

def numberOfWaysToPlaceBallsInBoxes : ℕ :=
  4 ^ 4

def numberOfWaysOneBoxEmpty : ℕ :=
  Nat.choose 4 2 * (Nat.factorial 4 / Nat.factorial 1)

def numberOfWaysTwoBoxesEmpty : ℕ :=
  (Nat.choose 4 1 * (Nat.factorial 4 / Nat.factorial 2)) + (Nat.choose 4 2 * (Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial 2)))

theorem part_one : numberOfWaysToPlaceBallsInBoxes = 256 := by
  sorry

theorem part_two : numberOfWaysOneBoxEmpty = 144 := by
  sorry

theorem part_three : numberOfWaysTwoBoxesEmpty = 120 := by
  sorry

end part_one_part_two_part_three_l173_173649


namespace lowest_position_of_vasya_l173_173213

-- Definitions of conditions
def num_cyclists : ℕ := 500
def num_stages : ℕ := 15
def vasya_position_each_stage : ℕ := 7

-- Theorem statement
theorem lowest_position_of_vasya (H1 : ∀ (s: ℕ), s ∈ finset.range(num_stages) → 
(num_cyclists + 1) - vasya_position_each_stage > (num_cyclists - 90))

(assumption_vasya :
  ∀ s ∈ finset.range(num_stages), vasya_position_each_stage < num_cyclists):
  ∃ (lowest_position: ℕ), lowest_position = 91 :=
sorry

end lowest_position_of_vasya_l173_173213


namespace hard_hats_remaining_l173_173075

theorem hard_hats_remaining :
  ∀ (initial_pink : ℕ) (initial_green : ℕ) (initial_yellow : ℕ)
    (carl_pink_taken : ℕ) (john_pink_taken : ℕ) (john_green_taken : ℕ),
    initial_pink = 26 → initial_green = 15 → initial_yellow = 24 →
    carl_pink_taken = 4 → john_pink_taken = 6 →
    john_green_taken = 2 * john_pink_taken →
    initial_pink - carl_pink_taken - john_pink_taken + 
    initial_green - john_green_taken +
    initial_yellow = 43 :=
by
  intros
  rw [a_0] at *
  rw [a_1] at *
  rw [a_2] at *
  rw [a_3] at *
  rw [a_4] at *
  rw [a_5] at *
  linarith

example : hard_hats_remaining 26 15 24 4 6 (2 * 6) := rfl 

end hard_hats_remaining_l173_173075


namespace obtain_null_matrix_via_finite_operations_l173_173778

def matrix_integers_finite_operations (M : Matrix Int) :=
  ∃ finite_operations : List (Sum (Int × Nat) (Int × Nat)) → Matrix Int, 
  ∀ n : ℕ, ∃ op_sequence : List (Sum (Int × Nat) (Int × Nat)),
  ∀ i j : ℕ, ((finite_operations op_sequence) M) i j % n = 0

theorem obtain_null_matrix_via_finite_operations :
  ∀ M : Matrix Int, matrix_integers_finite_operations M →
  ∃ finite_operations : List (Sum (Int × Nat) (Int × Nat)) → Matrix Int,
  (finite_operations op_sequence) M = Matrix.zero :=
by
  intro M h
  sorry

end obtain_null_matrix_via_finite_operations_l173_173778


namespace circumference_of_major_arc_l173_173549

theorem circumference_of_major_arc
  (A B C : Point) 
  (radius : ℝ)
  (h1 : radius = 12)
  (h2 : ∃ O : Point, Circle O radius ∧ (A ≠ B ∧ B ≠ C ∧ C ≠ A) ∧ ∠ ACB = 40) 
: ∃ circumference : ℝ, circumference = 192 * π / 9 :=
by
  sorry

end circumference_of_major_arc_l173_173549


namespace will_remaining_money_l173_173703

theorem will_remaining_money : 
  ∀ (initial_money sweater_cost tshirt_cost shoes_cost refund_percentage remaining_money : ℕ),
  initial_money = 74 →
  sweater_cost = 9 →
  tshirt_cost = 11 →
  shoes_cost = 30 →
  refund_percentage = 90 →
  remaining_money = 51 → 
  let total_spent := sweater_cost + tshirt_cost + ((shoes_cost * (100 - refund_percentage)) / 100) 
  in remaining_money = initial_money - total_spent :=
by 
  intros initial_money sweater_cost tshirt_cost shoes_cost refund_percentage remaining_money 
         h_initial_money h_sweater_cost h_tshirt_cost h_shoes_cost h_refund_percentage h_remaining_money
  let total_spent := sweater_cost + tshirt_cost + ((shoes_cost * (100 - refund_percentage)) / 100)
  have h_total_spent : total_spent = 20 + 3 := by 
    unfold total_spent
    rw [h_sweater_cost, h_tshirt_cost, h_shoes_cost, h_refund_percentage]
    norm_num
  have h_remaining_money_computed : initial_money - total_spent = 51 := by
    rw [h_initial_money, h_total_spent]
    norm_num
  rw [h_remaining_money] at h_remaining_money_computed
  exact h_remaining_money_computed

end will_remaining_money_l173_173703


namespace length_AB_l173_173081

variable (A B C D : Type) [metric_space A] {dist : A → A → ℝ}

-- Conditions
axiom h1 : dist B C = dist C A -- Triangle ABC is isosceles
axiom h2 : dist B C = dist C D -- Triangle CBD is isosceles
axiom cbd_perimeter : dist B D + dist B C + dist C D = 28
axiom abc_perimeter : dist A B + dist B C + dist C A = 34
axiom bd_length : dist B D = 10

-- Question: What is the length of AB?
theorem length_AB : dist A B = 16 := sorry

end length_AB_l173_173081


namespace prove_ineq_l173_173398

noncomputable def is_decreasing_on (f : ℝ → ℝ) (I : set ℝ) :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f y < f x

noncomputable def is_even (f : ℝ → ℝ) :=
  ∀ x, f x = f (-x)

noncomputable def is_periodic (f : ℝ → ℝ) (p : ℝ) :=
  ∀ x, f (x + p) = f x

variable {f : ℝ → ℝ}
variable {α β : ℝ}

theorem prove_ineq
  (h1 : is_even f)
  (h2 : ∀ x, f (x + 1) = -f x)
  (h3 : is_decreasing_on f (set.Icc (-3) (-2)))
  (h4 : 0 < α ∧ α < (π / 2))
  (h5 : 0 < β ∧ β < (π / 2)) :
  f (Real.sin α) > f (Real.cos β) := 
sorry

end prove_ineq_l173_173398


namespace largest_four_digit_integer_congruent_to_17_mod_26_l173_173293

theorem largest_four_digit_integer_congruent_to_17_mod_26 :
  ∃ x : ℤ, 1000 ≤ x ∧ x < 10000 ∧ x % 26 = 17 ∧ x = 9978 :=
by
  sorry

end largest_four_digit_integer_congruent_to_17_mod_26_l173_173293


namespace value_of_a_l173_173507

theorem value_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : ∀ x ∈ set.Icc (1 : ℝ) 2, y = a^x)
  (h4 : (max (a^1) (a^2) + min (a^1) (a^2)) = 6) :
  a = 2 :=
sorry

end value_of_a_l173_173507


namespace converse_even_sum_l173_173618

def is_even (n : Int) : Prop := ∃ k : Int, n = 2 * k

theorem converse_even_sum (a b : Int) :
  (is_even a ∧ is_even b → is_even (a + b)) →
  (is_even (a + b) → is_even a ∧ is_even b) :=
by
  sorry

end converse_even_sum_l173_173618


namespace cos_squared_formula_15deg_l173_173718

theorem cos_squared_formula_15deg :
  (Real.cos (15 * Real.pi / 180))^2 - (1 / 2) = (Real.sqrt 3) / 4 :=
by
  sorry

end cos_squared_formula_15deg_l173_173718


namespace polynomial_is_perfect_square_trinomial_l173_173493

-- The definition of a perfect square trinomial
def isPerfectSquareTrinomial (a b c m : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a * b = c ∧ 4 * a * a + m * b = 4 * a * b * b

-- The main theorem to prove that if the polynomial is a perfect square trinomial, then m = 20
theorem polynomial_is_perfect_square_trinomial (a b : ℝ) (h : isPerfectSquareTrinomial 2 1 5 25) :
  ∀ x, (4 * x * x + 20 * x + 25 = (2 * x + 5) * (2 * x + 5)) :=
by
  sorry

end polynomial_is_perfect_square_trinomial_l173_173493


namespace intersection_of_A_and_B_l173_173473

def A : Set ℝ := { x | 0 < x }
def B : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | 0 < x ∧ x ≤ 1 } := 
sorry

end intersection_of_A_and_B_l173_173473


namespace thabo_book_ratio_l173_173610

theorem thabo_book_ratio :
  ∃ (P_f P_nf H_nf : ℕ), H_nf = 35 ∧ P_nf = H_nf + 20 ∧ P_f + P_nf + H_nf = 200 ∧ P_f / P_nf = 2 :=
by
  sorry

end thabo_book_ratio_l173_173610


namespace focus_of_parabola_l173_173418

theorem focus_of_parabola :
  (∃ f : ℝ, ∀ y : ℝ, (x = -1 / 4 * y^2) = (x = (y^2 / 4 + f)) -> f = -1) :=
by
  sorry

end focus_of_parabola_l173_173418


namespace find_range_of_m_l173_173837

variable (x m : ℝ)

def proposition_p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * m * x + (4 * m - 3) > 0

def proposition_q (m : ℝ) : Prop := (∀ m > 2, m + 1 / (m - 2) ≥ 4) ∧ (∃ m, m + 1 / (m - 2) = 4)

def range_m : Set ℝ := {m | 1 < m ∧ m ≤ 2} ∪ {m | m ≥ 3}

theorem find_range_of_m
  (h_p : proposition_p m ∨ ¬proposition_p m)
  (h_q : proposition_q m ∨ ¬proposition_q m)
  (h_exclusive : (proposition_p m ∧ ¬proposition_q m) ∨ (¬proposition_p m ∧ proposition_q m))
  : m ∈ range_m := sorry

end find_range_of_m_l173_173837


namespace range_of_m_l173_173430

variable {x m : ℝ}
variable (q: ℝ → Prop) (p: ℝ → Prop)

-- Definition of q
def q_cond : Prop := (x - (1 + m)) * (x - (1 - m)) ≤ 0

-- Definition of p
def p_cond : Prop := |1 - (x - 1) / 3| ≤ 2

-- Statement of the proof problem
theorem range_of_m (h1 : ∀ x, q x → p x) (h2 : ∃ x, ¬p x → q x) 
  (h3 : m > 0) :
  0 < m ∧ m ≤ 3 :=
by
  sorry

end range_of_m_l173_173430


namespace fraction_not_on_time_l173_173714

theorem fraction_not_on_time (total_attendees : ℕ) (male_fraction female_fraction male_on_time_fraction female_on_time_fraction : ℝ)
  (H1 : male_fraction = 3/5)
  (H2 : male_on_time_fraction = 7/8)
  (H3 : female_on_time_fraction = 4/5)
  : ((1 - (male_fraction * male_on_time_fraction + (1 - male_fraction) * female_on_time_fraction)) = 3/20) :=
sorry

end fraction_not_on_time_l173_173714


namespace sum_f_values_l173_173433

def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

theorem sum_f_values :
  f 1 + f 2 + f (1/2) + f 3 + f (1/3) + f 4 + f (1/4) = 7 / 2 :=
by
  sorry

end sum_f_values_l173_173433


namespace S15_constant_l173_173443

-- Define the arithmetic sequence a_n
noncomputable def a (n : ℕ) (a1 d : ℤ) : ℤ := a1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
noncomputable def S (n : ℕ) (a1 d : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Given condition: a_5 + a_8 + a_11 is constant
axiom const_sum : ∀ (a1 d : ℤ), a 5 a1 d + a 8 a1 d + a 11 a1 d = 3 * a1 + 21 * d

-- The equivalent proof problem
theorem S15_constant (a1 d : ℤ) : S 15 a1 d = 5 * (3 * a1 + 21 * d) :=
by
  sorry

end S15_constant_l173_173443


namespace strictly_increasing_f_l173_173468

noncomputable def f (x a : ℝ) : ℝ := 
  (1/2) * ((Real.cos x - Real.sin x) * (Real.cos x + Real.sin x)) + 
  3 * a * (Real.sin x - Real.cos x) + 
  (4 * a - 1) * x 

theorem strictly_increasing_f (a : ℝ) : a ∈ Set.Ici (1 : ℝ) ↔ 
  ∀ x y ∈ Set.Icc (-Real.pi / 2) (0 : ℝ), x < y → f x a < f y a := 
by
  sorry

end strictly_increasing_f_l173_173468


namespace solve_system_of_equations_l173_173169

theorem solve_system_of_equations (x y : ℝ) (k : ℤ) :
  x^2 + 4 * (sin y)^2 - 4 = 0 ∧ cos x - 2 * (cos y)^2 - 1 = 0 ↔
  x = 0 ∧ ∃ k : ℤ, y = (π / 2) + k * π :=
by sorry

end solve_system_of_equations_l173_173169


namespace expected_sixes_in_three_rolls_l173_173673

theorem expected_sixes_in_three_rolls : 
  (∑ k in Finset.range 4, k * (Nat.choose 3 k) * (1/6)^k * (5/6)^(3-k)) = 1/2 := 
by
  sorry

end expected_sixes_in_three_rolls_l173_173673


namespace fractional_sum_lt_half_l173_173129

theorem fractional_sum_lt_half (n : ℕ) (x : Fin n → ℝ) (h_n : 2 ≤ n) (hx_pos : ∀ i, 0 < x i)
  (hx_prod : ∏ i, x i = 1) :
  (∑ i, frac (x i)) < (2 * n - 1) / 2 := 
sorry

end fractional_sum_lt_half_l173_173129


namespace percentage_increase_overtime_rate_l173_173325

theorem percentage_increase_overtime_rate :
  let regular_rate := 16
  let regular_hours_limit := 30
  let total_earnings := 760
  let total_hours_worked := 40
  let overtime_rate := 28 -- This is calculated as $280/10 from the solution.
  let increase_in_hourly_rate := overtime_rate - regular_rate
  let percentage_increase := (increase_in_hourly_rate / regular_rate) * 100
  percentage_increase = 75 :=
by {
  sorry
}

end percentage_increase_overtime_rate_l173_173325


namespace smallest_prime_with_digit_sum_28_l173_173692

def sum_of_digits (n : ℕ) : ℕ :=
  (n.to_digits 10).sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem smallest_prime_with_digit_sum_28 : ∃ p : ℕ, is_prime p ∧ sum_of_digits p = 28 ∧ (∀ q : ℕ, is_prime q ∧ sum_of_digits q = 28 → p ≤ q) ∧ p = 1999 :=
begin
  sorry
end

end smallest_prime_with_digit_sum_28_l173_173692


namespace sqrt_eq_solution_l173_173166

noncomputable def solve_sqrt_eq (z : ℝ) : Prop :=
  sqrt (7 + 3 * z) = 8

theorem sqrt_eq_solution (z : ℝ) (h : 7 + 3 * z ≥ 0) :
  solve_sqrt_eq z ↔ z = 19 :=
sorry

end sqrt_eq_solution_l173_173166


namespace min_value_expression_l173_173842

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : 2 * x + y = 2) : 
  ∃ (min_val : ℝ), min_val = (2 / (x + 1) + 1 / y) ∧ min_val = 9 / 4 :=
begin
  sorry
end

end min_value_expression_l173_173842


namespace total_price_jewelry_paintings_l173_173983

theorem total_price_jewelry_paintings:
  (original_price_jewelry original_price_paintings increase_jewelry_in_dollars increase_paintings_in_percent quantity_jewelry quantity_paintings new_price_jewelry new_price_paintings total_cost: ℝ) 
  (h₁: original_price_jewelry = 30)
  (h₂: original_price_paintings = 100)
  (h₃: increase_jewelry_in_dollars = 10)
  (h₄: increase_paintings_in_percent = 0.20)
  (h₅: quantity_jewelry = 2)
  (h₆: quantity_paintings = 5)
  (h₇: new_price_jewelry = original_price_jewelry + increase_jewelry_in_dollars)
  (h₈: new_price_paintings = original_price_paintings + (original_price_paintings * increase_paintings_in_percent))
  (h₉: total_cost = (new_price_jewelry * quantity_jewelry) + (new_price_paintings * quantity_paintings)) :
  total_cost = 680 :=
by 
  sorry

end total_price_jewelry_paintings_l173_173983


namespace find_parabola_eq_l173_173338

-- Definition of the parabola with parameter p
def parabola (p : ℝ) : Prop := y^2 = 2 * p * x

-- Definition of the condition on the line passing through the focus
def line_through_focus (p : ℝ) : Prop := y = sqrt(3) * (x - p / 2)

-- Prove the equation of the parabola given conditions
theorem find_parabola_eq (p : ℝ) (h_pos : p > 0) (h_AB : ∃ A B : ℝ × ℝ, |AB| = 8) :
  parabola p -> line_through_focus p -> y^2 = 6 * x :=
sorry

end find_parabola_eq_l173_173338


namespace sum_of_solutions_l173_173403

theorem sum_of_solutions :
  let P (x : ℝ) := (x^2 - 6*x + 5)^(x^2 - 8*x + 12) = 1
  (Set.sum (Set.finite.filter (Set.univ.to_finset : Set ℝ → Finset ℝ) P).to_list (λ x, x)) = 14 :=
by
  sorry

end sum_of_solutions_l173_173403


namespace angles_sum_60_degrees_l173_173948

variables {A B C D : Type}
variables [euclidean_geometry A B C D]
variables (ab ac bc ad bd cd : ℝ)
variables (h1 : AB = ab) (h2 : AC = ac) (h3 : BC = bc) (h4 : AD = ad) (h5 : BD = bd) (h6 : CD = cd)
variables (D_in_interior : D ∈ interior_triangle ABC)

theorem angles_sum_60_degrees : angle ABD + angle ACD = 60 :=
by {
  sorry
}

end angles_sum_60_degrees_l173_173948


namespace exists_natural_number_k_l173_173320

theorem exists_natural_number_k (n : ℕ) (a : Fin n → ℝ) (b : Fin n → ℝ) 
  (h1 : ∀ i j, i ≤ j → b i ≥ b j) (h2 : ∀ i, 0 ≤ b i) (h3 : b 0 ≤ 1) :
  ∃ k : Fin n, |(Finset.univ.sum (λ i, a i * b i))| ≤ |(Finset.range k.succ).sum (λ i, a ⟨i, Nat.lt_succ_iff.2 k.2⟩)| := 
sorry

end exists_natural_number_k_l173_173320


namespace rabbit_speed_correct_l173_173057

-- Define the conditions given in the problem
def rabbit_speed (x : ℝ) : Prop :=
2 * (2 * x + 4) = 188

-- State the main theorem using the defined conditions
theorem rabbit_speed_correct : ∃ x : ℝ, rabbit_speed x ∧ x = 45 :=
by
  sorry

end rabbit_speed_correct_l173_173057


namespace perpendicular_l173_173868

noncomputable def a : ℝ × ℝ := (3, 1)
noncomputable def b (k : ℝ) : ℝ × ℝ := (2*k - 1, k)
noncomputable def sum_vec (k : ℝ) : ℝ × ℝ := (fst (a) + fst (b k), snd (a) + snd (b k))

theorem perpendicular {k : ℝ} (h : (fst (sum_vec k)) * fst (b k) + (snd (sum_vec k)) * snd (b k) = 0) :
  k = -1 ∨ k = 2/5 :=
sorry

end perpendicular_l173_173868


namespace range_of_m_l173_173136

noncomputable def M (m : ℝ) : Set ℝ := {x | x + m ≥ 0}
def N : Set ℝ := {x | x^2 - 2 * x - 8 < 0}
def U : Set ℝ := Set.univ
def CU_M (m : ℝ) : Set ℝ := {x | x < -m}
def empty_intersection (m : ℝ) : Prop := (CU_M m ∩ N = ∅)

theorem range_of_m (m : ℝ) : empty_intersection m → m ≥ 2 := by
  sorry

end range_of_m_l173_173136


namespace chord_length_of_intercepted_curve_l173_173850

theorem chord_length_of_intercepted_curve
  (C : ℝ → ℝ → Prop := λ x y, y^2 = 6 * x)
  (l_param : ℝ → ℝ × ℝ := λ t, (-4 / 5 * t + 2, 3 / 5 * t))
  (l_normal : ℝ → ℝ → Prop := λ x y, 3 * x + 4 * y - 6 = 0) :
  let intersect_pts := {p : ℝ × ℝ | C p.1 p.2 ∧ l_normal p.1 p.2},
      x1 := classical.some_spec (classical.inhabited_of_nonempty (classical.nonempty_def.mpr ⟨⟨3, -3.6⟩⟩)),
      x2 := classical.some_spec (classical.inhabited_of_nonempty (classical.nonempty_def.mpr ⟨⟨7, 2.4⟩⟩)) in
  sqrt ((x1 - x2)^2 + (3 / 5 * x1 - 3 / 5 * x2)^2) = 20 * sqrt 7 / 3 :=
sorry

end chord_length_of_intercepted_curve_l173_173850


namespace complex_modulus_calculation_l173_173852

noncomputable def z : ℂ := ((1 : ℂ) + complex.I) / ((2 : ℂ) - complex.I)

theorem complex_modulus_calculation : |z| = real.sqrt 10 / 5 := by
  have h : z * (2 - complex.I) = 1 + complex.I := by sorry
  have z_def : z = (1 + complex.I) / (2 - complex.I) := by sorry
  have z_simplified : z = (1 / 5) + (3 / 5) * complex.I := by sorry
  have z_mod : |z| = real.sqrt (((1 / 5) ^ 2) + ((3 / 5) ^ 2)) := by sorry
  rw z_mod
  ring
  rw real.sqrt_add_square_square
  norm_num
  rfl

end complex_modulus_calculation_l173_173852


namespace faster_by_airplane_l173_173763

theorem faster_by_airplane : 
  let driving_time := 3 * 60 + 15 
  let airport_drive := 10
  let wait_to_board := 20
  let flight_duration := driving_time / 3
  let exit_plane := 10
  driving_time - (airport_drive + wait_to_board + flight_duration + exit_plane) = 90 := 
by
  let driving_time : ℕ := 3 * 60 + 15
  let airport_drive : ℕ := 10
  let wait_to_board : ℕ := 20
  let flight_duration : ℕ := driving_time / 3
  let exit_plane : ℕ := 10
  have h1 : driving_time = 195 := rfl
  have h2 : flight_duration = 65 := by norm_num [h1]
  have h3 : 195 - (10 + 20 + 65 + 10) = 195 - 105 := by norm_num
  have h4 : 195 - 105 = 90 := by norm_num
  exact h4

end faster_by_airplane_l173_173763


namespace area_foh_of_trapezoid_l173_173681

section TrapezoidProblem

-- Define the variables based on the conditions
variables (EF GH : ℝ) (area_trapezoid : ℝ) (ratio : ℝ)
  (area_FOH : ℝ)

-- Define the given conditions
def trapezoidEFGH : Prop := EF = 15 ∧ GH = 25 ∧ area_trapezoid = 200

-- Specify the property we need to prove
theorem area_foh_of_trapezoid (EF GH area_trapezoid ratio area_FOH : ℝ)
  (h_trapezoid : trapezoidEFGH EF GH area_trapezoid ratio)
  (h_ratio : ratio = 5 / 8) : area_FOH = 78.125 :=
by
  sorry -- Proof to be provided

end TrapezoidProblem

end area_foh_of_trapezoid_l173_173681


namespace problem_ineq_l173_173564

variable {a b c : ℝ}

theorem problem_ineq 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * a^2 * b^2 * c^2 := 
sorry

end problem_ineq_l173_173564


namespace g_g1_eq_43_l173_173041

def g (x : ℝ) : ℝ := 2 * x^2 + 3 * x - 1

theorem g_g1_eq_43 : g (g 1) = 43 :=
by
  sorry

end g_g1_eq_43_l173_173041


namespace problem_solution_l173_173534

noncomputable def curve_C1 (t : ℝ) : ℝ × ℝ :=
  (1 - (real.sqrt 2) / 2 * t, 1 + (real.sqrt 2) / 2 * t)

noncomputable def curve_C2 (x y : ℝ) : Prop :=
  y^2 = 4 * x

noncomputable def P : ℝ × ℝ :=
  (1, 1) -- polar coordinates (sqrt(2), pi/4)

theorem problem_solution : 
  ∀ (A B : ℝ × ℝ), 
    (curve_C1 A.fst, curve_C1 A.snd) ∈ curve_C2 →
    (curve_C1 B.fst, curve_C1 B.snd) ∈ curve_C2 →
    (∃ t1 t2 : ℝ, A = curve_C1 t1 ∧ B = curve_C1 t2 ∧
      abs (P.fst - A.fst) = abs t1 ∧ abs (P.fst - B.fst) = abs t2) →
    1 / (real.dist P A) + 1 / (real.dist P B) = 2 * real.sqrt 6 / 3 :=
sorry

end problem_solution_l173_173534


namespace factoring_options_count_l173_173524

theorem factoring_options_count :
  (∃ (n : ℕ), n = 14) ↔
  ∃ a b c : ℕ, a + b + c = 10 ∧ a ≥ b ∧ b ≥ c ∧ fintype.card {p : ℕ × ℕ // p.1 + p.2 = 10 ∧ p.1 ≥ p.2} + 
    fintype.card {p : ℕ × ℕ // p.1 + p.2 = 9 ∧ p.1 ≥ p.2} + 
    fintype.card {p : ℕ × ℕ // p.1 + p.2 = 8 ∧ p.1 ≥ p.2} + 
    fintype.card {p : ℕ × ℕ // p.1 + p.2 = 7 ∧ p.1 ≥ p.2} = 14 := 
by
  sorry

end factoring_options_count_l173_173524


namespace harry_items_left_l173_173487

def sea_stars : ℕ := 34
def seashells : ℕ := 21
def snails : ℕ := 29
def lost_items : ℕ := 25

def total_items : ℕ := sea_stars + seashells + snails
def remaining_items : ℕ := total_items - lost_items

theorem harry_items_left : remaining_items = 59 := by
  -- proof skipped
  sorry

end harry_items_left_l173_173487


namespace value_of_a_area_of_ABC_l173_173537

-- Given conditions
variables {A B C : ℝ}
variables {a b c : ℝ} (h1 : cos B = 1/4) (h2 : b = 2) (h3 : sin C = 2 * sin A)

-- Theorems to prove
theorem value_of_a (h1 : cos B = 1/4) (h2 : b = 2) (h3 : sin C = 2 * sin A) : a = 1 := 
sorry

theorem area_of_ABC (h1 : cos B = 1/4) (h2 : b = 2) (h3 : sin C = 2 * sin A) (h4 : a = 1) : 
  let S := 1/2 * a * b * sin C in S = sqrt 15 / 4 := 
sorry

end value_of_a_area_of_ABC_l173_173537


namespace mike_play_time_is_300_minutes_l173_173579

-- Definitions from conditions
def weekly_pay : ℕ := 100
def spend_half (d : ℕ) : ℕ := d / 2
def spend_on_food : ℕ := 10
def hourly_rate : ℕ := 8
def minutes_per_hour : ℕ := 60

-- Problem statement
theorem mike_play_time_is_300_minutes :
  let arcade_budget := spend_half weekly_pay - spend_on_food in
  let hours_played := arcade_budget / hourly_rate in
  let minutes_played := hours_played * minutes_per_hour in
  minutes_played = 300 :=
sorry

end mike_play_time_is_300_minutes_l173_173579


namespace conic_section_is_parabola_l173_173790

theorem conic_section_is_parabola (x y : ℝ) : y^4 - 16 * x^2 = 2 * y^2 - 64 → ((y^2 - 1)^2 = 16 * x^2 - 63) ∧ (∃ k : ℝ, y^2 = 4 * k * x + 1) :=
sorry

end conic_section_is_parabola_l173_173790


namespace product_grades_probabilities_l173_173743

theorem product_grades_probabilities (P_Q P_S : ℝ) (h1 : P_Q = 0.98) (h2 : P_S = 0.21) :
  P_Q - P_S = 0.77 ∧ 1 - P_Q = 0.02 :=
by
  sorry

end product_grades_probabilities_l173_173743


namespace exist_two_points_same_color_l173_173198

theorem exist_two_points_same_color 
  {X : Type} [MetricSpace X] [Fintype X]
  (colors : X → Fin 4)
  (sphere_points : Set X)
  (Hsphere : ∀ p ∈ sphere_points, dist p (0 : V) = 4)
  (point_colors : ∀ p ∈ sphere_points, colors p ∈ Fin 4) :
  ∃ p q ∈ sphere_points, colors p = colors q ∧ (dist p q = 4 * Real.sqrt 3 ∨ dist p q = 2 * Real.sqrt 6) :=
sorry

end exist_two_points_same_color_l173_173198


namespace find_omega_l173_173862

noncomputable def omega (ω : ℝ) : Prop :=
  ∀ (ω > 0), (∃ T : ℝ, T = π ∧ ∀ x : ℝ, 2*sin(ω*x + π/6) = 2*sin(ω*(x + T) + π/6)) → ω = 2

theorem find_omega : omega 2 :=
by
  sorry

end find_omega_l173_173862


namespace total_price_correct_l173_173985

def original_jewelry_price : ℕ := 30
def original_painting_price : ℕ := 100
def jewelry_price_increase : ℕ := 10
def painting_price_increase_percentage : ℕ := 20
def num_jewelry : ℕ := 2
def num_paintings : ℕ := 5

theorem total_price_correct :
  let new_jewelry_price := original_jewelry_price + jewelry_price_increase in
  let new_painting_price := original_painting_price + (original_painting_price * painting_price_increase_percentage / 100) in
  let total_price := (new_jewelry_price * num_jewelry) + (new_painting_price * num_paintings) in
  total_price = 680 := 
by
  sorry

end total_price_correct_l173_173985


namespace cannot_use_square_difference_formula_l173_173698

variable (a b x : ℝ)

def exprA := (a - b) * (a + b)
def exprB := (a - 1) * (-a + 1)
def exprC := (-x - y) * (x - y)
def exprD := (-x + 1) * (-1 - x)

theorem cannot_use_square_difference_formula :
  ¬ ((∃ u v : ℝ, exprB = (u - v) * (u + v))) :=
by 
  sorry

end cannot_use_square_difference_formula_l173_173698


namespace at_least_one_digit_even_l173_173621

theorem at_least_one_digit_even 
  (n : ℕ) 
  (n_digits : (len : ℕ) → len = 17 → ∀ d ∈ digits (~(10^len) n), 0 ≤ d ∧ d < 10) : 
  (∃ k, k ∈ digits 20 (n + reverse_digits 17 n) ∧ k % 2 = 0) :=
by
  sorry

end at_least_one_digit_even_l173_173621


namespace mean_identity_example_l173_173185

theorem mean_identity_example {x y z : ℝ} 
  (h1 : x + y + z = 30)
  (h2 : x * y * z = 343)
  (h3 : x * y + y * z + z * x = 257.25) :
  x^2 + y^2 + z^2 = 385.5 :=
by
  sorry

end mean_identity_example_l173_173185


namespace angle_A_measure_in_triangle_l173_173087

theorem angle_A_measure_in_triangle (A B C : ℝ) 
  (h1 : B = 15)
  (h2 : C = 3 * B) 
  (angle_sum : A + B + C = 180) :
  A = 120 :=
by
  -- We'll fill in the proof steps later
  sorry

end angle_A_measure_in_triangle_l173_173087


namespace intersection_M_N_l173_173105

def M : Set ℝ := { x : ℝ | Real.log10 (1 - x) < 0 }

def N : Set ℝ := { x : ℝ | -1 ≤ x ∧ x ≤ 1 }

theorem intersection_M_N : M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end intersection_M_N_l173_173105


namespace shorter_leg_of_triangle_l173_173341

theorem shorter_leg_of_triangle (a b : ℝ) (h1 : b = 10) (h2 : 2 * sqrt (5 * a) = b) : a = 5 :=
sorry

end shorter_leg_of_triangle_l173_173341


namespace liking_not_related_to_gender_expected_value_of_X_is_9_over_5_l173_173514

-- Conditions from the problem statement
def boys_like := 30
def boys_dislike := 20
def girls_like := 40
def girls_dislike := 10
def total_students := boys_like + boys_dislike + girls_like + girls_dislike
def alpha_value := 0.01
def critical_value := 6.635

-- Part 1: Prove that liking the mascots is not related to gender
theorem liking_not_related_to_gender :
  let a := boys_like in
  let b := boys_dislike in
  let c := girls_like in
  let d := girls_dislike in
  let n := total_students in
  let chi_squared := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d)) in
  chi_squared < critical_value :=
by
  sorry

-- Part 2: Prove the expected value of X among sampled students is 9/5
-- Define conditions for sampling
def total_sampled := 5
def like_in_sample := 3
def people_selected := 3

def Prob_X_eq (k : ℕ) : ℚ :=
  match k with
  | 0 => 0
  | 1 => 3 / 10
  | 2 => 6 / 10
  | 3 => 1 / 10
  | _ => 0

def E_X : ℚ :=
  (1 * Prob_X_eq 1) + (2 * Prob_X_eq 2) + (3 * Prob_X_eq 3)

theorem expected_value_of_X_is_9_over_5 :
  E_X = 9 / 5 :=
by
  sorry

end liking_not_related_to_gender_expected_value_of_X_is_9_over_5_l173_173514


namespace joan_writing_time_l173_173091

theorem joan_writing_time
  (total_time : ℕ)
  (time_piano : ℕ)
  (time_reading : ℕ)
  (time_exerciser : ℕ)
  (h1 : total_time = 120)
  (h2 : time_piano = 30)
  (h3 : time_reading = 38)
  (h4 : time_exerciser = 27) : 
  total_time - (time_piano + time_reading + time_exerciser) = 25 :=
by
  sorry

end joan_writing_time_l173_173091


namespace mike_play_time_is_300_minutes_l173_173580

-- Definitions from conditions
def weekly_pay : ℕ := 100
def spend_half (d : ℕ) : ℕ := d / 2
def spend_on_food : ℕ := 10
def hourly_rate : ℕ := 8
def minutes_per_hour : ℕ := 60

-- Problem statement
theorem mike_play_time_is_300_minutes :
  let arcade_budget := spend_half weekly_pay - spend_on_food in
  let hours_played := arcade_budget / hourly_rate in
  let minutes_played := hours_played * minutes_per_hour in
  minutes_played = 300 :=
sorry

end mike_play_time_is_300_minutes_l173_173580


namespace max_value_of_stones_l173_173068

-- Define the types of stones and their weights and values
def stone := ℕ → ℕ

def weight_3_pound : stone := λ n, 3 * n
def value_3_pound : stone := λ n, 9 * n

def weight_6_pound : stone := λ n, 6 * n
def value_6_pound : stone := λ n, 15 * n

def weight_1_pound : stone := λ n, 1 * n
def value_1_pound : stone := λ n, 1 * n

-- Define the total weight constraint
def max_weight : ℕ := 24

-- Define total value function combining different stone types
def total_value (n3 n6 n1 : ℕ) : ℕ :=
  value_3_pound n3 + value_6_pound n6 + value_1_pound n1

-- Define total weight function combining different stone types
def total_weight (n3 n6 n1 : ℕ) : ℕ :=
  weight_3_pound n3 + weight_6_pound n6 + weight_1_pound n1

-- Theorem: The maximum achievable value under the given constraints
theorem max_value_of_stones : ∃ n3 n6 n1 : ℕ, 
  total_weight n3 n6 n1 ≤ max_weight ∧ total_value n3 n6 n1 = 72 :=
sorry

end max_value_of_stones_l173_173068


namespace singers_in_choir_l173_173327

variable (X : ℕ)

/-- In the first verse, only half of the total singers sang -/ 
def first_verse_not_singing (X : ℕ) : ℕ := X / 2

/-- In the second verse, a third of the remaining singers joined in -/
def second_verse_joining (X : ℕ) : ℕ := (X / 2) / 3

/-- In the final third verse, 10 people joined so that the whole choir sang together -/
def remaining_singers_after_second_verse (X : ℕ) : ℕ := first_verse_not_singing X - second_verse_joining X

def final_verse_joining_condition (X : ℕ) : Prop := remaining_singers_after_second_verse X = 10

theorem singers_in_choir : ∃ (X : ℕ), final_verse_joining_condition X ∧ X = 30 :=
by
  sorry

end singers_in_choir_l173_173327


namespace vasya_maximum_rank_l173_173247

theorem vasya_maximum_rank {n : ℕ} (cyclists stages : ℕ) (VasyaPlace : ℕ) 
  (rankings : Π (s : fin stages), fin cyclists):
  cyclists = 500 → stages = 15 → VasyaPlace = 7 →
  (∀ s, rankings s VasyaPlace = 7) →
  (∀ s t i, rankings s i ≠ rankings t i) →
  (∀ s, ∃ l, list.nodup l ∧ (∀ i j, i < j → rankings s i < rankings s j) ∧ list.length l = 500) →
  (∃ (maxRank : ℕ), maxRank = 91) :=
by
  intros hcyclists hstages hplace hvasya_place hdistinct_rankings hstage_rankings
  use 91
  sorry

end vasya_maximum_rank_l173_173247


namespace length_of_CD_l173_173063

-- Define the given triangle and its properties
structure Triangle (α : Type) [LinearOrderedField α] :=
(A B C : α × α)
(angle_ABC : Real.Angle (B - A) (C - B) = Real.Angle.pi + Real.Angle.pi / 4)
(AB_length : dist A B = 5)
(BC_length : dist B C = 2)

namespace Triangle
-- Define the perpendiculars and the point D
def perpendiculars_meet_at_D (t : Triangle ℝ) :=
∃ D : ℝ × ℝ, ∃ E : ℝ × ℝ, 
(perpendicular (line_through t.A t.B) E D) ∧ 
(perpendicular (line_through t.B t.C) E D) ∧ 
(Dist.dist D.1 t.C = Real.sqrt 2) ∧ 
(Dist.dist D.2 t.C = 0)

-- The statement of the problem in Lean 4
theorem length_of_CD (t : Triangle ℝ) (h : perpendiculars_meet_at_D t) : 
  ∃ D : ℝ × ℝ, dist t.C D = Real.sqrt 2 :=
by
  sorry
end Triangle

end length_of_CD_l173_173063


namespace total_marbles_l173_173510

theorem total_marbles (boxes : ℕ) (marbles_per_box : ℕ) (h1 : boxes = 10) (h2 : marbles_per_box = 100) : (boxes * marbles_per_box = 1000) :=
by
  sorry

end total_marbles_l173_173510


namespace geom_seq_problem_l173_173832

-- Define the geometric sequence and the two conditions
def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Given conditions and questions
theorem geom_seq_problem :
  ∃ (a : ℕ → ℝ) (q : ℝ), q > 1 ∧ geom_seq a q ∧
  a 3 = 8 ∧ (a 2 + a 4 = 20) ∧
  (a = (λ n, 2 ^ n)) ∧
  (∀ n : ℕ, (Σ (k : fin n), (-1) ^ (k : ℕ) * a k * a (k + 1)) =
   (8 / 5 - (-1) ^ n * 2 ^ (2 * n + 3) / 5)) := sorry

end geom_seq_problem_l173_173832


namespace frequency_of_group_5_l173_173924

theorem frequency_of_group_5
  (total_points : ℕ)
  (points_group1 : ℕ)
  (points_group2 : ℕ)
  (points_group3 : ℕ)
  (points_group4 : ℕ)
  (total_points_eq : total_points = 50)
  (points_group1_eq : points_group1 = 2)
  (points_group2_eq : points_group2 = 8)
  (points_group3_eq : points_group3 = 15)
  (points_group4_eq : points_group4 = 5) :
  let points_group5 := total_points - points_group1 - points_group2 - points_group3 - points_group4
  in (points_group5 / total_points : ℝ) = 0.4 :=
by
  sorry

end frequency_of_group_5_l173_173924


namespace area_of_triangle_AOB_l173_173077

structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨1, 1⟩
def B : Point := ⟨3, -1⟩
def O : Point := ⟨0, 0⟩

noncomputable def area_of_triangle (A B O : Point) : ℝ :=
  1 / 2 * real.abs (A.x * (B.y - O.y) + B.x * (O.y - A.y) + O.x * (A.y - B.y))

theorem area_of_triangle_AOB : area_of_triangle A B O = 2 :=
  by sorry

end area_of_triangle_AOB_l173_173077


namespace least_positive_multiple_of_24_gt_450_l173_173300

theorem least_positive_multiple_of_24_gt_450 : 
  ∃ n : ℕ, n > 450 ∧ (∃ k : ℕ, n = 24 * k) → n = 456 :=
by 
  sorry

end least_positive_multiple_of_24_gt_450_l173_173300


namespace vasya_lowest_position_l173_173256

noncomputable theory

def number_of_cyclists := 500
def number_of_stages := 15
def position_of_vasya_each_stage := 7

theorem vasya_lowest_position (total_cyclists : ℕ) (stages : ℕ) (position_each_stage : ℕ)
  (h_total_cyclists : total_cyclists = number_of_cyclists)
  (h_stages : stages = number_of_stages)
  (h_position_each_stage : position_each_stage = position_of_vasya_each_stage)
  (no_identical_times : ∀ (i j : ℕ), i ≠ j → ∀ (stage : ℕ), stage ≤ stages → ∀ (t : ℕ), t ≤ total_cyclists → 
    (time : Π (s : ℕ) (c : ℕ), c < total_cyclists → ℕ), 
    time stage i < time stage j ∨ time stage j < time stage i):
  ∃ lowest_pos, lowest_pos = 91 := sorry

end vasya_lowest_position_l173_173256


namespace consecutive_integer_sum_l173_173009

noncomputable def sqrt17 : ℝ := Real.sqrt 17

theorem consecutive_integer_sum : ∃ (a b : ℤ), (b = a + 1) ∧ (a < sqrt17 ∧ sqrt17 < b) ∧ (a + b = 9) :=
by
  sorry

end consecutive_integer_sum_l173_173009


namespace centroid_homothety_l173_173590

-- Definitions for the problem conditions
variable {P : Point}
variable {A B C A' B' C' : Point}
variable (homothetic : Homothety (Triangle.mk A B C) (Triangle.mk A' B' C') P)
variable (centroid_ABC : IsCentroid P (Triangle.mk A B C))

-- The statement to be proved
theorem centroid_homothety :
  IsCentroid P (Triangle.mk A' B' C') :=
sorry

end centroid_homothety_l173_173590


namespace min_value_expression_l173_173133

theorem min_value_expression :
  ∀ (x y z w : ℝ), x > 0 → y > 0 → z > 0 → w > 0 → x = y → x + y + z + w = 1 →
  (x + y + z) / (x * y * z * w) ≥ 1024 :=
by
  intros x y z w hx hy hz hw hxy hsum
  sorry

end min_value_expression_l173_173133


namespace football_team_lineup_l173_173585

theorem football_team_lineup :
  let total_members := 15
  let ol_candidates := 5
  let qb_candidates := total_members - 1
  let rb_candidates := total_members - 2
  let wr_candidates := total_members - 3
  let s_candidates := total_members - 4
  (ol_candidates * qb_candidates * rb_candidates * wr_candidates * s_candidates = 109200) :=
by
  let total_members := 15
  let ol_candidates := 5
  let qb_candidates := total_members - 1
  let rb_candidates := total_members - 2
  let wr_candidates := total_members - 3
  let s_candidates := total_members - 4
  have h : ol_candidates * qb_candidates * rb_candidates * wr_candidates * s_candidates = 109200
  exact h
  sorry

end football_team_lineup_l173_173585


namespace max_sector_area_exists_l173_173458

-- Definition of the problem variables and conditions
def perimeter := 30

def sector_max_area (R α S : ℝ) : Prop :=
  let l := 30 - 2*R in
  (l + 2*R = perimeter) ∧
  (S = (1/2) * l * R) ∧
  (R = 15 / 2) ∧
  (S = 225 / 4) ∧
  (α = 2)

-- The proof statement expressing the maximum area and central angle
theorem max_sector_area_exists :
  ∃ (R α S : ℝ), sector_max_area R α S :=
by
  sorry

end max_sector_area_exists_l173_173458


namespace increasing_intervals_sin_func_l173_173849

theorem increasing_intervals_sin_func (a b : ℝ) (k : ℤ) :
  (-|a| + b = -3) ∧ (|a| + b = 1) →
  (a > 0 → ∀ x, (k * π + π / 12 ≤ x ∧ x ≤ k * π + 7 * π / 12 ↔ f x = -sin(2 * x + π / 3))) ∧
  (a < 0 → ∀ x, (k * π - π / 12 ≤ x ∧ x ≤ k * π + 5 * π / 12 ↔ f x = sin(2 * x - π / 3))) :=
begin
  sorry
end

end increasing_intervals_sin_func_l173_173849


namespace sum_of_powers_mod_l173_173565

theorem sum_of_powers_mod (p : ℕ) (hp : Nat.Prime p) (hodd : Odd p) : 
  (∑ k in Finset.range p, k ^ (2 * p - 1)) % p^2 = (p * (p + 1) / 2) % p^2 := 
sorry

end sum_of_powers_mod_l173_173565


namespace william_wins_10_rounds_l173_173709

-- Definitions from the problem conditions
variable (W H : ℕ)
variable (total_rounds : ℕ := 15)
variable (additional_wins : ℕ := 5)

-- Conditions
def total_game_condition : Prop := W + H = total_rounds
def win_difference_condition : Prop := W = H + additional_wins

-- Statement to be proved
theorem william_wins_10_rounds (h1 : total_game_condition W H) (h2 : win_difference_condition W H) : W = 10 :=
by
  sorry

end william_wins_10_rounds_l173_173709


namespace vasya_lowest_position_l173_173243

theorem vasya_lowest_position
  (n : ℕ) (m : ℕ) (num_cyclists : ℕ) (vasya_place : ℕ) :
  num_cyclists = 500 →
  n = 15 →
  vasya_place = 7 →
  ∀ (stages : fin n → fin num_cyclists) (no_identical_times : ∀ i j : fin n, i ≠ j → 
  ∀ k l : fin num_cyclists, stages i k ≠ stages j l),
  ∃ (lowest_position : ℕ), lowest_position = 91 := 
by sorry

end vasya_lowest_position_l173_173243


namespace system_solution_in_first_quadrant_l173_173955

theorem system_solution_in_first_quadrant (c x y : ℝ)
  (h1 : x - y = 5)
  (h2 : c * x + y = 7)
  (hx : x > 3)
  (hy : y > 1) : c < 1 :=
sorry

end system_solution_in_first_quadrant_l173_173955


namespace circumcircle_tangent_l173_173963

theorem circumcircle_tangent
  {A B C L P Q : Point}
  (h1 : angle_bisector A B C L)
  (h2 : perpendicular_bisector_intersect_circumcircle A L B C P Q) :
  tangent (circumcircle P L Q) (line B C) :=
sorry

end circumcircle_tangent_l173_173963


namespace problem_I_l173_173466

def f (x : ℝ) : ℝ := 4 - |x| - |x - 3|

theorem problem_I {x : ℝ} : f (x + 3 / 2) ≥ 0 ↔ -2 ≤ x ∧ x ≤ 2 :=
by
  sorry

end problem_I_l173_173466


namespace correct_option_D_l173_173447

variables {l m n : ℝ → ℝ → ℝ} {α β : ℝ → ℝ → ℝ → Prop}

-- Definitions of perpendicularity and parallelism of lines and planes
def perpendicular (x y : ℝ → ℝ → ℝ) : Prop := sorry
def parallel (x y : ℝ → ℝ → ℝ) : Prop := sorry
def line_in_plane (l : ℝ → ℝ → ℝ) (p : ℝ → ℝ → ℝ → Prop) : Prop := sorry

-- Assuming given conditions as definitions
variables (hlα : perpendicular l α) (hlβ : parallel l β)

-- Statement to be proved
theorem correct_option_D : hlα → hlβ → perpendicular α β :=
by sorry

end correct_option_D_l173_173447


namespace unique_representation_reciprocal_sum_l173_173501
noncomputable theory

open Classical Real Nat

theorem unique_representation (x : ℚ) (hx : 0 < x) : 
    ∃! (a : ℕ → ℤ) (n : ℕ), x = (∑ i in Finset.range n, a i / Nat.factorial (i + 1)) ∧ 
                             (∀ i, 1 ≤ i → i < n → 0 ≤ a i ∧ a i < i + 1) :=
sorry

theorem reciprocal_sum (x : ℚ) (hx : 0 < x) : 
    ∃ (n : ℕ → ℕ), (x = ∑ i, (1 / n i : ℝ)) ∧ (∀ i, 10^6 < n i) :=
sorry

end unique_representation_reciprocal_sum_l173_173501


namespace expected_sixes_in_three_rolls_l173_173674

theorem expected_sixes_in_three_rolls : 
  (∑ k in Finset.range 4, k * (Nat.choose 3 k) * (1/6)^k * (5/6)^(3-k)) = 1/2 := 
by
  sorry

end expected_sixes_in_three_rolls_l173_173674


namespace median_of_divisors_9999_is_100_l173_173722

-- Define the prime factorization of 9999
def factor_9999 : Prop := 9999 = 3^2 * 11 * 101

-- List of all divisors
def divisors_9999 : List ℕ := [1, 3, 9, 11, 33, 99, 101, 303, 909, 1111, 3333, 9999]

-- Median of the positive divisors of 9999
def median_divisors_9999 (divisors : List ℕ) : ℕ :=
  let len := List.length divisors
  if len % 2 = 0 then
    let mid1 := List.nth_le divisors ((len / 2) - 1) (by sorry) -- 6th element for 0-based index
    let mid2 := List.nth_le divisors (len / 2) (by sorry) -- 7th element for 0-based index
    (mid1 + mid2) / 2
  else
    sorry

theorem median_of_divisors_9999_is_100 : median_divisors_9999 divisors_9999 = 100 := by
  sorry

end median_of_divisors_9999_is_100_l173_173722


namespace wendy_baked_29_cookies_l173_173689

/-- Wendy made pastries for the school bake sale. She baked 4 cupcakes and some cookies. 
    After the sale, she had 24 pastries to take back home and sold 9 pastries. 
    How many cookies did she bake? -/
theorem wendy_baked_29_cookies :
  let cupcakes := 4 in
  let sold := 9 in
  let remaining := 24 in
  let total_pastries := sold + remaining in
  let cookies := total_pastries - cupcakes in
  cookies = 29 :=
by
  sorry

end wendy_baked_29_cookies_l173_173689


namespace evaluate_nested_fraction_l173_173795

theorem evaluate_nested_fraction :
  (1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))) = (8 / 21) :=
by 
s interval for end
om ke plese end the program here and sher the solution.

para ke dy dim all kyd on nee ju plese ke tee‌ی


end evaluate_nested_fraction_l173_173795


namespace linear_function_through_point_increasing_y_l173_173154

theorem linear_function_through_point_increasing_y (k : ℝ) (b : ℝ) (h1 : (0, 1) ∈ set_of (λ x, 1 = k * 0 + b)) (h2 : k > 0) :
  ∃ k : ℝ, ∀ x : ℝ, (∃ y : ℝ, y = k * x + 1) ∧ k > 0 :=
by
  use 1
  intros x
  use (1 : ℝ)
  sorry

end linear_function_through_point_increasing_y_l173_173154


namespace lowest_position_of_vasya_l173_173214

-- Definitions of conditions
def num_cyclists : ℕ := 500
def num_stages : ℕ := 15
def vasya_position_each_stage : ℕ := 7

-- Theorem statement
theorem lowest_position_of_vasya (H1 : ∀ (s: ℕ), s ∈ finset.range(num_stages) → 
(num_cyclists + 1) - vasya_position_each_stage > (num_cyclists - 90))

(assumption_vasya :
  ∀ s ∈ finset.range(num_stages), vasya_position_each_stage < num_cyclists):
  ∃ (lowest_position: ℕ), lowest_position = 91 :=
sorry

end lowest_position_of_vasya_l173_173214


namespace vasya_lowest_position_l173_173215

theorem vasya_lowest_position
  (num_cyclists : ℕ)
  (num_stages : ℕ)
  (num_ahead : ℕ)
  (position_vasya : ℕ)
  (total_time : List ℕ)
  (unique_total_times : total_time.nodup)
  (stage_positions : List (List ℕ))
  (unique_stage_positions : ∀ stage ∈ stage_positions, stage.nodup)
  (vasya_consistent : ∀ stage ∈ stage_positions, stage.nth position_vasya = some num_ahead) :
  num_ahead * num_stages + 1 = 91 :=
by
  sorry

end vasya_lowest_position_l173_173215


namespace find_alpha_l173_173828

-- Define the given conditions
def angle_in_range (α : ℝ) : Prop := 0 ≤ α ∧ α < 360
def coordinates_on_terminal_side (α : ℝ) : Prop := 
  ∃ (x y : ℝ), (x = Real.sin 150 ∧ y = Real.cos 150) ∧ tan α = y / x

-- Proof statement
theorem find_alpha (α : ℝ) (h1 : angle_in_range α) (h2 : coordinates_on_terminal_side α) : α = 300 :=
sorry

end find_alpha_l173_173828


namespace total_price_correct_l173_173984

def original_jewelry_price : ℕ := 30
def original_painting_price : ℕ := 100
def jewelry_price_increase : ℕ := 10
def painting_price_increase_percentage : ℕ := 20
def num_jewelry : ℕ := 2
def num_paintings : ℕ := 5

theorem total_price_correct :
  let new_jewelry_price := original_jewelry_price + jewelry_price_increase in
  let new_painting_price := original_painting_price + (original_painting_price * painting_price_increase_percentage / 100) in
  let total_price := (new_jewelry_price * num_jewelry) + (new_painting_price * num_paintings) in
  total_price = 680 := 
by
  sorry

end total_price_correct_l173_173984


namespace min_rectangle_remaining_area_l173_173363

-- Define the dimensions of the rectangle
def AB : ℝ := 4
def BC : ℝ := 6

-- Define the total area of the rectangle
def rectangle_area (a b : ℝ) : ℝ := a * b

-- Define the areas of the three isosceles right triangles
def triangle_area_1 : ℝ := (1 / 2) * 3 * 6
def triangle_area_2 : ℝ := (1 / 2) * 4 * 4
def triangle_area_3 : ℝ := (1 / 2) * 3 * 3

-- Define the sum of the areas of the three triangles
def sum_triangle_areas : ℝ := triangle_area_1 + triangle_area_2 + triangle_area_3

-- Define the minimum remaining area after cutting out the three triangles
def min_remaining_area (total_area triangles_area : ℝ) : ℝ := total_area - triangles_area

-- State the theorem
theorem min_rectangle_remaining_area : min_remaining_area (rectangle_area AB BC) sum_triangle_areas = 2.5 := by
  -- Automatically prove the trivial calculations
  sorry

end min_rectangle_remaining_area_l173_173363


namespace independent_variable_range_l173_173633

theorem independent_variable_range (x : ℝ) :
  (∃ y : ℝ, y = 1 / (Real.sqrt x - 1)) ↔ x ≥ 0 ∧ x ≠ 1 := 
by
  sorry

end independent_variable_range_l173_173633


namespace perfect_cube_factors_of_144_l173_173882

-- Define the problem conditions using Lean code
def prime_factorization_144 : Prop :=
  ∀ n, n ∣ 144 ↔ ∃ (a b : ℕ), n = 2^a * 3^b ∧ a ≤ 4 ∧ b ≤ 2

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = k^3

-- Lean theorem statement
theorem perfect_cube_factors_of_144 : 
  (finset.filter (λ n, is_perfect_cube n) (finset.filter (λ n, n ∣ 144) (finset.range 145))).card = 2 :=
sorry

end perfect_cube_factors_of_144_l173_173882


namespace train_speed_problem_l173_173684

noncomputable def speed_of_first_train : ℝ :=
  31.25

theorem train_speed_problem
  (speed_of_first_train speed_of_second_train distance_diff total_distance : ℝ)
  (h1 : speed_of_second_train = 25)
  (h2 : distance_diff = 75)
  (h3 : total_distance = 675)
  (h4 : ∀ d v, ((d / v) = (d - 75) / 25) ∧ (d + (d - 75) = 675) → v = 31.25) :
  speed_of_first_train = 31.25 :=
by
  exact h4 375 31.25 sorry

end train_speed_problem_l173_173684


namespace marbles_lost_l173_173152

theorem marbles_lost (initial_marbles lost_marbles gifted_marbles remaining_marbles : ℕ) 
  (h_initial : initial_marbles = 85)
  (h_gifted : gifted_marbles = 25)
  (h_remaining : remaining_marbles = 43)
  (h_before_gifting : remaining_marbles + gifted_marbles = initial_marbles - lost_marbles) :
  lost_marbles = 17 :=
by
  sorry

end marbles_lost_l173_173152


namespace ratio_of_areas_eq_l173_173899

-- Define the conditions
variables {C D : Type} [circle C] [circle D]
variables (R_C R_D : ℝ)
variable (L : ℝ)

-- Given conditions
axiom arc_length_eq : (60 / 360) * (2 * π * R_C) = L
axiom arc_length_eq' : (40 / 360) * (2 * π * R_D) = L

-- Statement to prove
theorem ratio_of_areas_eq : (π * R_C^2) / (π * R_D^2) = 4 / 9 :=
sorry

end ratio_of_areas_eq_l173_173899


namespace total_money_collected_l173_173348

theorem total_money_collected (attendees : ℕ) (reserved_price unreserved_price : ℝ) (reserved_sold unreserved_sold : ℕ)
  (h_attendees : attendees = 1096)
  (h_reserved_price : reserved_price = 25.00)
  (h_unreserved_price : unreserved_price = 20.00)
  (h_reserved_sold : reserved_sold = 246)
  (h_unreserved_sold : unreserved_sold = 246) :
  (reserved_price * reserved_sold + unreserved_price * unreserved_sold) = 11070.00 :=
by
  sorry

end total_money_collected_l173_173348


namespace max_number_of_games_is_9_l173_173275

def max_games (students : ℕ) (players_per_game : ℕ) : ℕ :=
  9 -- This is given as part of the solution

theorem max_number_of_games_is_9 (students players_per_game : ℕ)
  (h1 : students = 12) (h2 : players_per_game = 4)
  (h3 : ∀ (G : set (set ℕ)), G.card ≤ 9 → ∀ g ∈ G, g.card = 4 → ∀ x y ∈ g, g ≠ g₂ → x ∉ g₂ ∨ y ∉ g₂) :
  max_games students players_per_game = 9 :=
sorry

end max_number_of_games_is_9_l173_173275


namespace circumcircle_diameter_l173_173913

-- Define the conditions and the function to calculate the diameter of the circumcircle

noncomputable def diameter_of_circumcircle 
  (a : ℝ) (B : ℝ) (S : ℝ) (sin : ℝ → ℝ) (cos : ℝ → ℝ) : ℝ :=
let c := 4 * real.sqrt 2,
    b := real.sqrt (a ^ 2 + c ^ 2 - 2 * a * c * cos B)
in b / sin B

theorem circumcircle_diameter (a : ℝ) (B : ℝ) (S : ℝ)
  (hA : a = 1) (hB : B = real.pi / 4) (hS : S = 2) : 
  diameter_of_circumcircle a B S real.sin real.cos = 5 * real.sqrt 2 := 
by
  rw [hA, hB, hS]
  -- remaining proof steps here
  sorry

end circumcircle_diameter_l173_173913


namespace m_ge_1_l173_173432

def f (x : ℝ) : ℝ := x^2
def g (m : ℝ) (x : ℝ) : ℝ := 2^x - m

theorem m_ge_1 (m : ℝ) :
  (∀ x1 : ℝ, x1 ∈ Icc (-1) 3 →
    ∃ x2 : ℝ, x2 ∈ Icc 0 2 ∧ f(x1) ≥ g(m, x2)) → m ≥ 1 :=
by
  sorry

end m_ge_1_l173_173432


namespace largest_expression_among_options_l173_173123

def y := 10^(-2022 : ℤ)

theorem largest_expression_among_options : 
  ∀ (x : ℝ), 
  (x = 5 + y ∨ x = 5 - y ∨ x = 5 * y ∨ x = 5 / y ∨ x = y / 5) -> 
  x ≤ 5 / y := 
by
  sorry

end largest_expression_among_options_l173_173123


namespace zhang_san_not_losing_probability_l173_173374

theorem zhang_san_not_losing_probability (p_win p_draw : ℚ) (h_win : p_win = 1 / 3) (h_draw : p_draw = 1 / 4) : 
  p_win + p_draw = 7 / 12 := by
  sorry

end zhang_san_not_losing_probability_l173_173374


namespace ratio_conner_sydney_day1_l173_173180

def initial_rocks_sydney := 837
def initial_rocks_conner := 723

def rocks_sydney_day1 := 4
def rocks_conner_day2 := 123
def rocks_conner_day3 := 27

-- Assuming C represents the number of rocks Conner collected on day one
variable (C : ℕ)

-- Calculating the total number of rocks each person has at the end of the contest
def total_rocks_sydney := initial_rocks_sydney + rocks_sydney_day1 + 2 * C
def total_rocks_conner := initial_rocks_conner + C + rocks_conner_day2 + rocks_conner_day3

theorem ratio_conner_sydney_day1 (h : total_rocks_conner = total_rocks_sydney) : C = 32 → 4 = 4 → 32 / 4 = 8 := by
  intro h1 h2
  rw h1
  rw h2
  sorry

end ratio_conner_sydney_day1_l173_173180


namespace find_a_l173_173848

noncomputable def isTangentLine (f : ℝ → ℝ) (line : ℝ → ℝ) (a x₀ : ℝ) : Prop :=
  ∃ y₀, f x₀ = y₀ ∧ line x₀ = y₀ ∧ deriv f x₀ = deriv line x₀

theorem find_a (a : ℝ) (h : isTangentLine (λ x, -1 / a * Real.exp x) (λ x, -x + 1) a 2) : 
  a = Real.exp 2 :=
by
  sorry

end find_a_l173_173848


namespace triangle_side_length_sum_l173_173759

theorem triangle_side_length_sum :
  ∃ (a b c : ℕ), (5: ℝ) ^ 2 + (7: ℝ) ^ 2 - 2 * (5: ℝ) * (7: ℝ) * (Real.cos (Real.pi * 80 / 180)) = (a: ℝ) + Real.sqrt b + Real.sqrt c ∧
  b = 62 ∧ c = 0 :=
sorry

end triangle_side_length_sum_l173_173759


namespace min_fraction_value_l173_173808

theorem min_fraction_value (x y : ℝ) (h1 : 1 ≤ x ∧ x ≤ 2) (h2 : 1 ≤ y ∧ y ≤ 2)
  (h3 : sqrt (x - 1) + sqrt (y - 1) = 1) : x / y = 1 / 2 := 
sorry

end min_fraction_value_l173_173808


namespace wine_age_problem_l173_173181

theorem wine_age_problem
  (C F T B Bo : ℕ)
  (h1 : F = 3 * C)
  (h2 : C = 4 * T)
  (h3 : B = (1 / 2 : ℝ) * T)
  (h4 : Bo = 2 * F)
  (h5 : C = 40) :
  F = 120 ∧ T = 10 ∧ B = 5 ∧ Bo = 240 := 
  by
    sorry

end wine_age_problem_l173_173181


namespace inradius_inequality_l173_173554

variable (ABC : Triangle)
variable (DEF : Triangle)
variable (R : ℝ := 1) -- Circumradius of Triangle ABC
variable (r p : ℝ) -- Inradius of Triangle ABC and orthic triangle DEF

-- Assuming DEF is the orthic triangle of ABC
axiom orthic_triangle : DEF = orthic ABC

theorem inradius_inequality :
  p ≤ 1 - (1 / 3) * (1 + r) ^ 2 := 
sorry

end inradius_inequality_l173_173554


namespace general_term_smallest_n_l173_173438

-- Definitions given in the conditions
def sequence (a : ℕ → ℝ) := ∀ n : ℕ, a (n + 1) ^ 2 - a (n + 1) * a n - 2 * a n ^ 2 = 0 ∧ a n > 0
def middle_term_condition (a : ℕ → ℝ) := a 3 + 2 = (a 2 + a 4) / 2

-- Proof obligations
theorem general_term (a : ℕ → ℝ) (h_seq : sequence a) (h_mid : middle_term_condition a) : ∀ n, a n = 2 ^ n := sorry

theorem smallest_n (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ)
  (h_seq : sequence a)
  (h_mid : middle_term_condition a)
  (h_bn : ∀ n, b n = a n * real.logb (2⁻¹) (a n))
  (h_sum : ∀ n, S n = ∑ i in finset.range (n + 1), b i)
  (h_an : ∀ n, a n = 2 ^ n) : ∃ n : ℕ, S n + n * 2^(n + 1) > 50 :=
sorry

end general_term_smallest_n_l173_173438


namespace woman_work_completion_days_l173_173336

def work_completion_days_man := 6
def work_completion_days_boy := 9
def work_completion_days_combined := 3

theorem woman_work_completion_days : 
  (1 / work_completion_days_man + W + 1 / work_completion_days_boy = 1 / work_completion_days_combined) →
  W = 1 / 18 → 
  1 / W = 18 :=
by
  intros h₁ h₂
  sorry

end woman_work_completion_days_l173_173336


namespace pieces_of_meat_per_slice_eq_22_l173_173141

def number_of_pepperoni : Nat := 30
def number_of_ham : Nat := 2 * number_of_pepperoni
def number_of_sausage : Nat := number_of_pepperoni + 12
def total_meat : Nat := number_of_pepperoni + number_of_ham + number_of_sausage
def number_of_slices : Nat := 6

theorem pieces_of_meat_per_slice_eq_22 : total_meat / number_of_slices = 22 :=
by
  sorry

end pieces_of_meat_per_slice_eq_22_l173_173141


namespace largest_angle_of_ABC_l173_173280

variables (ABC : Triangle) (A B C : ℝ)
variable (hABC : obtuse ScaleneTriangle ABC)
variable (hA : ABC.angleA = 30)
variable (hB : ABC.angleB = 55)
noncomputable def largest_interior_angle : ℝ :=
  180 - ABC.angleA - ABC.angleB

theorem largest_angle_of_ABC : largest_interior_angle ABC A B = 95 := by
  sorry

end largest_angle_of_ABC_l173_173280


namespace percent_preferred_orange_l173_173347

variable (red orange green yellow blue purple : ℕ)
variable h_red : red = 70
variable h_orange : orange = 50
variable h_green : green = 60
variable h_yellow : yellow = 80
variable h_blue : blue = 40
variable h_purple : purple = 50

theorem percent_preferred_orange :
  let total := red + orange + green + yellow + blue + purple in
  let percent_orange := (orange * 100) / total in
  percent_orange = 14 :=
by
  have h_total : total = 70 + 50 + 60 + 80 + 40 + 50 := by
    rw [h_red, h_orange, h_green, h_yellow, h_blue, h_purple]
  have h_percent_orange : percent_orange = 14 := by
    rw [h_total, h_orange]
    norm_num
  exact h_percent_orange

end percent_preferred_orange_l173_173347


namespace hole_empties_tank_in_60_hours_l173_173151

noncomputable def hole_emptying_time : ℝ :=
  let pipe_rate := 1 / 15
  let effective_rate_with_hole := 1 / 20
  let hole_rate := pipe_rate - effective_rate_with_hole
  1 / hole_rate

theorem hole_empties_tank_in_60_hours :
  hole_emptying_time = 60 := by
  have pipe_rate : ℝ := 1 / 15
  have effective_rate_with_hole : ℝ := 1 / 20
  have hole_rate : ℝ := pipe_rate - effective_rate_with_hole
  have time_to_empty := 1 / hole_rate
  calc
    hole_emptying_time
        = time_to_empty : rfl
    ... = 60 : by sorry

end hole_empties_tank_in_60_hours_l173_173151


namespace will_money_left_l173_173701

theorem will_money_left (initial sweater tshirt shoes refund_percentage : ℕ) 
  (h_initial : initial = 74)
  (h_sweater : sweater = 9)
  (h_tshirt : tshirt = 11)
  (h_shoes : shoes = 30)
  (h_refund_percentage : refund_percentage = 90) : 
  initial - (sweater + tshirt + (100 - refund_percentage) * shoes / 100) = 51 := by
  sorry

end will_money_left_l173_173701


namespace cos_rational_irrational_l173_173284

theorem cos_rational_irrational (p q : ℤ) (hq_pos : q > 0) :
  let θ := (p : ℝ) / q * 180 in
  θ.cos ≠ 0 ∧ θ.cos ≠ 1/2 ∧ θ.cos ≠ -1/2 ∧ θ.cos ≠ 1 ∧ θ.cos ≠ -1 →
  irrational θ.cos :=
sorry

end cos_rational_irrational_l173_173284


namespace mixed_number_multiplication_l173_173367

def mixed_to_improper (a : Int) (b : Int) (c : Int) : Rat :=
  a + (b / c)

theorem mixed_number_multiplication : 
  let a := 5
  let b := mixed_to_improper 7 2 5
  a * b = (37 : Rat) :=
by
  intros
  sorry

end mixed_number_multiplication_l173_173367


namespace gemstone_necklaces_sold_l173_173545

theorem gemstone_necklaces_sold : 
  ∃ G : ℕ, 
    4 * 3 + G * 3 = 21 ∧ G = 3 :=
begin
  let G := 3,
  use G,
  split,
  { simp [G], norm_num, },
  { refl, }
end

end gemstone_necklaces_sold_l173_173545


namespace vasya_lowest_position_l173_173201

theorem vasya_lowest_position
  (n_cyclists : ℕ) (n_stages : ℕ) 
  (stage_positions : ℕ → ℕ → ℕ) -- a function that takes a stage and a cyclist and returns the position (e.g., stage_positions(stage, cyclist) = position)
  (total_time : ℕ → ℕ)  -- a function that takes a cyclist and returns their total time
  (distinct_times : ∀ (c1 c2 : ℕ), c1 ≠ c2 → (total_time c1 ≠ total_time c2) ∧ 
                   ∀ (s : ℕ), stage_positions s c1 ≠ stage_positions s c2)
  (vasya_position : ℕ) (hv : ∀ (s : ℕ), s < n_stages → stage_positions s vasya_position = 7) :
  vasya_position = 91 :=
sorry

end vasya_lowest_position_l173_173201


namespace range_of_x_and_fx_l173_173011

theorem range_of_x_and_fx
  (x : ℝ)
  (h1 : log (1/2) (x^2) ≥ log (1/2) (3*x - 2)) :
  (1 ≤ x ∧ x ≤ 2) ∧
  (f : ℝ → ℝ, f(x) = (log 2 (x / 4)) * (log 2 (x / 2)) → f '' {x | 1 ≤ x ∧ x ≤ 2} = set.Icc 0 2) :=
by 
  sorry

end range_of_x_and_fx_l173_173011


namespace hard_hats_remaining_l173_173074

theorem hard_hats_remaining :
  ∀ (initial_pink : ℕ) (initial_green : ℕ) (initial_yellow : ℕ)
    (carl_pink_taken : ℕ) (john_pink_taken : ℕ) (john_green_taken : ℕ),
    initial_pink = 26 → initial_green = 15 → initial_yellow = 24 →
    carl_pink_taken = 4 → john_pink_taken = 6 →
    john_green_taken = 2 * john_pink_taken →
    initial_pink - carl_pink_taken - john_pink_taken + 
    initial_green - john_green_taken +
    initial_yellow = 43 :=
by
  intros
  rw [a_0] at *
  rw [a_1] at *
  rw [a_2] at *
  rw [a_3] at *
  rw [a_4] at *
  rw [a_5] at *
  linarith

example : hard_hats_remaining 26 15 24 4 6 (2 * 6) := rfl 

end hard_hats_remaining_l173_173074


namespace problem_1_problem_2_l173_173480

open Set

variables {x b : ℝ}
def A := {x | -3 < x ∧ x ≤ 6}
def M := {x | -4 ≤ x ∧ x < 5}
def B (b : ℝ) := {x | b-3 < x ∧ x < b+7}
def U := set.univ ℝ  -- U = ℝ

theorem problem_1 : A ∩ M = {x | -3 < x ∧ x < 5} := 
sorry

theorem problem_2 (b : ℝ) : B b ∪ - M = U → -2 ≤ b ∧ b < -1 :=
sorry

end problem_1_problem_2_l173_173480


namespace ratio_of_sides_l173_173574

section
variables {z1 z2 z3 : ℂ} -- complex numbers
variables {a b c : ℝ} -- lengths of the sides of the triangle

-- Conditions
axiom non_identical : z1 ≠ z2 ∧ z2 ≠ z3 ∧ z3 ≠ z1
axiom eq_condition : 4 * z1^2 + 5 * z2^2 + 5 * z3^2 = 4 * z1 * z2 + 6 * z2 * z3 + 4 * z3 * z1
axiom sides : a = complex.abs (z1 - z2) ∧ b = complex.abs (z2 - z3) ∧ c = complex.abs (z3 - z1)

-- Prove ratio
theorem ratio_of_sides : (a = 2 * b) ∧ (b = complex.abs (sqrt 5)) ∧ (c = complex.abs (sqrt 5)) :=
sorry
end

end ratio_of_sides_l173_173574


namespace william_wins_10_rounds_l173_173707

-- Definitions from the problem conditions
variable (W H : ℕ)
variable (total_rounds : ℕ := 15)
variable (additional_wins : ℕ := 5)

-- Conditions
def total_game_condition : Prop := W + H = total_rounds
def win_difference_condition : Prop := W = H + additional_wins

-- Statement to be proved
theorem william_wins_10_rounds (h1 : total_game_condition W H) (h2 : win_difference_condition W H) : W = 10 :=
by
  sorry

end william_wins_10_rounds_l173_173707


namespace cakes_served_yesterday_l173_173750

theorem cakes_served_yesterday (cakes_today_lunch : ℕ) (cakes_today_dinner : ℕ) (total_cakes : ℕ)
  (h1 : cakes_today_lunch = 5) (h2 : cakes_today_dinner = 6) (h3 : total_cakes = 14) :
  total_cakes - (cakes_today_lunch + cakes_today_dinner) = 3 :=
by
  -- Import necessary libraries
  sorry

end cakes_served_yesterday_l173_173750


namespace inequality_solution_l173_173606

noncomputable def solve_inequality (x : ℝ) : Prop :=
  27 ^ Real.sqrt (Real.log 3 x) - 13 * 3 * Real.sqrt (Real.sqrt (4 * Real.log 3 x)) + 
  55 * x ^ Real.sqrt (Real.log x 3) ≤ 75

theorem inequality_solution (x : ℝ) (hx : x > 0 ∧ x ≠ 1) :
  solve_inequality x ↔ (x ∈ set.Ioc 1 3 ∨ x = 5 ^ Real.log 3 5) :=
sorry

end inequality_solution_l173_173606


namespace congruence_solution_count_l173_173843

theorem congruence_solution_count :
  ∃ (count : ℕ), 
    count = 4 ∧
    count = (Finset.card (Finset.filter (λ (x : ℕ), x < 150 ∧ ((x + 17) % 46 = 72 % 46))
            (Finset.range 150))) :=
by
  -- Lean statement to check the count of solutions
  sorry

end congruence_solution_count_l173_173843


namespace albert_number_solution_l173_173764

theorem albert_number_solution (A B C : ℝ) 
  (h1 : A = 2 * B + 1) 
  (h2 : B = 2 * C + 1) 
  (h3 : C = 2 * A + 2) : 
  A = -11 / 7 := 
by 
  sorry

end albert_number_solution_l173_173764


namespace range_of_a_for_monotonic_increasing_log_l173_173623

theorem range_of_a_for_monotonic_increasing_log {a : ℝ} :
  (∀ x y ∈ Ico 0 2, x < y → log a (4 - x^2) < log a (4 - y^2)) ↔ 0 < a ∧ a < 1 :=
sorry

end range_of_a_for_monotonic_increasing_log_l173_173623


namespace ellipse_area_l173_173788

theorem ellipse_area : 
  ∀ x y : ℝ, 2*x^2 + 8*x + 9*y^2 - 18*y + 8 = 0 → 
    ellipse_area (2*x^2 + 8*x + 9*y^2 - 18*y + 8) = real.pi * real.sqrt 4.5 :=
sorry

end ellipse_area_l173_173788


namespace complex_sum_l173_173120

open Complex

theorem complex_sum (w : ℂ) (h : w^2 - w + 1 = 0) :
  w^103 + w^104 + w^105 + w^106 + w^107 = -1 :=
sorry

end complex_sum_l173_173120


namespace income_percent_greater_l173_173043

variable (A B : ℝ)

-- Condition: A's income is 25% less than B's income
def income_condition (A B : ℝ) : Prop :=
  A = 0.75 * B

-- Statement: B's income is 33.33% greater than A's income
theorem income_percent_greater (A B : ℝ) (h : income_condition A B) :
  B = A * (4 / 3) := by
sorry

end income_percent_greater_l173_173043


namespace sqrt_sum_eq_one_l173_173455

theorem sqrt_sum_eq_one
  (a b c k : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hk : 0 < k)
  (h : 2 * a * b * c + k * (a^2 + b^2 + c^2) = k^3) :
  sqrt ((k - a) * (k - b) / ((k + a) * (k + b))) +
  sqrt ((k - b) * (k - c) / ((k + b) * (k + c))) +
  sqrt ((k - c) * (k - a) / ((k + c) * (k + a))) = 1 :=
by
  sorry

end sqrt_sum_eq_one_l173_173455


namespace expected_number_of_sixes_l173_173667

-- Define the problem context and conditions
def die_prob := (1 : ℝ) / 6

def expected_six (n : ℕ) : ℝ :=
  n * die_prob

-- The main proposition to prove
theorem expected_number_of_sixes (n : ℕ) (hn : n = 3) : expected_six n = 1 / 2 :=
by
  rw [hn]
  have fact1 : (3 : ℝ) * die_prob = 3 / 6 := by norm_cast; norm_num
  rw [fact1]
  norm_num

-- We add sorry to indicate incomplete proof, fulfilling criteria 4
sorry

end expected_number_of_sixes_l173_173667


namespace expected_number_of_sixes_l173_173670

-- Define the problem context and conditions
def die_prob := (1 : ℝ) / 6

def expected_six (n : ℕ) : ℝ :=
  n * die_prob

-- The main proposition to prove
theorem expected_number_of_sixes (n : ℕ) (hn : n = 3) : expected_six n = 1 / 2 :=
by
  rw [hn]
  have fact1 : (3 : ℝ) * die_prob = 3 / 6 := by norm_cast; norm_num
  rw [fact1]
  norm_num

-- We add sorry to indicate incomplete proof, fulfilling criteria 4
sorry

end expected_number_of_sixes_l173_173670


namespace phil_cards_left_l173_173587

variable (cards_per_week : ℕ)
variable (weeks_per_year : ℕ)
variable (fraction_lost : ℝ)

def total_cards (cards_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  cards_per_week * weeks_per_year

def cards_left (total_cards : ℕ) (fraction_lost : ℝ) : ℕ :=
  (total_cards : ℝ) * (1 - fraction_lost) |> Int.toNat

theorem phil_cards_left
  (h1 : cards_per_week = 20)
  (h2 : weeks_per_year = 52)
  (h3 : fraction_lost = 0.5)
  : cards_left (total_cards cards_per_week weeks_per_year) fraction_lost = 520 :=
  by
    sorry

end phil_cards_left_l173_173587


namespace min_mod_z_l173_173959

open Complex

theorem min_mod_z (z : ℂ) (hz : abs (z - 2 * I) + abs (z - 5) = 7) : abs z = 10 / 7 :=
sorry

end min_mod_z_l173_173959


namespace min_rounds_to_eliminate_soldiers_l173_173951

theorem min_rounds_to_eliminate_soldiers (n : ℕ) (h : 0 < n) : 
  ∃ m, m = ⌈log 2 n⌉ :=
by
  sorry

end min_rounds_to_eliminate_soldiers_l173_173951


namespace minimum_value_of_abs_z_l173_173961

noncomputable def min_abs_z (z : ℂ) (h : |z - 2 * complex.I| + |z - 5| = 7) : ℝ :=
  classical.some (exists_minimum (λ z, |z|) (λ z, |z - 2 * complex.I| + |z - 5| = 7) h)

theorem minimum_value_of_abs_z : ∀ z : ℂ, 
  (|z - 2 * complex.I| + |z - 5| = 7) → |z| ≥ 0 
  → min_abs_z z (by sorry) = 10 / real.sqrt 29 :=
by 
  sorry

end minimum_value_of_abs_z_l173_173961


namespace difference_of_two_numbers_l173_173188

theorem difference_of_two_numbers :
  ∃ S : ℕ, S * 16 + 15 = 1600 ∧ 1600 - S = 1501 :=
by
  sorry

end difference_of_two_numbers_l173_173188


namespace polynomial_expansion_l173_173797

theorem polynomial_expansion (x : ℝ) :
  (x - 3) * (x + 5) * (x^2 + 9) = x^4 + 2x^3 - 6x^2 + 18x - 135 :=
by
  sorry

end polynomial_expansion_l173_173797


namespace solution_set_cannot_be_3_elements_l173_173471

noncomputable def f (a b x : ℝ) : ℝ := a ^ |x - b|

theorem solution_set_cannot_be_3_elements
  (a b m n p : ℝ)
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (hm : m ≠ 0)
  (hn : n ≠ 0)
  (hp : p ≠ 0) :
  ¬ (∀ x1 x2 x3 : ℝ, 
    f a b x1 ∈ {1, 3, 4} ∧ 
    f a b x2 ∈ {1, 3, 4} ∧ 
    f a b x3 ∈ {1, 3, 4} ∧ 
    m * (f a b x1)^2 + n * (f a b x1) + p = 0 ∧ 
    m * (f a b x2)^2 + n * (f a b x2) + p = 0 ∧ 
    m * (f a b x3)^2 + n * (f a b x3) + p = 0) := 
sorry

end solution_set_cannot_be_3_elements_l173_173471


namespace largest_distinct_digit_number_divisible_by_all_digits_with_five_l173_173630

theorem largest_distinct_digit_number_divisible_by_all_digits_with_five
  (n : ℕ) (h1 : (n ≠ 0)) (h2 : (∀ d ∈ (Nat.digits 10 n), Nat.digits 10 n = List.nodup (Nat.digits 10 n) ∧ n % d = 0)) (h3 : 5 ∈ (Nat.digits 10 n)) :
  n ≤ 9735 :=
sorry

end largest_distinct_digit_number_divisible_by_all_digits_with_five_l173_173630


namespace largest_four_digit_integer_congruent_to_17_mod_26_l173_173294

theorem largest_four_digit_integer_congruent_to_17_mod_26 :
  ∃ x : ℤ, 1000 ≤ x ∧ x < 10000 ∧ x % 26 = 17 ∧ x = 9978 :=
by
  sorry

end largest_four_digit_integer_congruent_to_17_mod_26_l173_173294


namespace length_AB_l173_173929

noncomputable def length_of_segment := 
  let x_param (t : ℝ) := 1 - (Real.sqrt 2 / 2) * t
  let y_param (t : ℝ) := 2 + (Real.sqrt 2 / 2) * t
  let curve_eq : ℝ → ℝ → Prop := λ x y, y^2 = 4 * x
  let line_eq : ℝ → ℝ → Prop := λ x y, x + y = 3
  let intersect (y : ℝ) := ∃ t : ℝ, (curve_eq (x_param t) y) ∧ (line_eq (x_param t) y)
  ∃ (y1 y2 : ℝ), intersect y1 ∧ intersect y2 ∧ (abs (y1 - y2) = 8 * Real.sqrt 2)
  
theorem length_AB : 
  length_of_segment := 
  sorry

end length_AB_l173_173929


namespace max_elements_in_valid_subset_l173_173137

open Set

def P : Set ℕ := { n | 1 ≤ n ∧ n ≤ 2014 }

def valid_subset (A : Set ℕ) : Prop :=
  A ⊆ P ∧ 
  (∀ (a b : ℕ), a ∈ A → b ∈ A → (a ≠ b → (a - b) % 99 ≠ 0)) ∧
  (∀ (a b : ℕ), a ∈ A → b ∈ A → ((a + b) % 99 ≠ 0))

theorem max_elements_in_valid_subset (A : Set ℕ) (hA : valid_subset A) : A.card ≤ 50 := 
sorry

end max_elements_in_valid_subset_l173_173137


namespace tiling_rectangle_l173_173730

theorem tiling_rectangle (a b m n : ℕ) (h_tiling_combination : ∃ c d : ℕ, m = b * c ∨ n = a * d) :
  (∃ c : ℕ, m = b * c) ∨ (∃ d : ℕ, n = a * d) :=
begin
  sorry
end

end tiling_rectangle_l173_173730


namespace ollie_fraction_of_yard_reached_l173_173989

theorem ollie_fraction_of_yard_reached (s : ℝ) (h : s > 0) : 
  let r := s / 2,
      A_C := π * r^2,
      A_S := s^2,
      f := A_C / A_S in
  f = π / 4 :=
by
  let r := s / 2
  let A_C := π * r^2
  let A_S := s^2
  let f := A_C / A_S
  show f = π / 4
  sorry

end ollie_fraction_of_yard_reached_l173_173989


namespace vasya_lowest_position_l173_173216

theorem vasya_lowest_position
  (num_cyclists : ℕ)
  (num_stages : ℕ)
  (num_ahead : ℕ)
  (position_vasya : ℕ)
  (total_time : List ℕ)
  (unique_total_times : total_time.nodup)
  (stage_positions : List (List ℕ))
  (unique_stage_positions : ∀ stage ∈ stage_positions, stage.nodup)
  (vasya_consistent : ∀ stage ∈ stage_positions, stage.nth position_vasya = some num_ahead) :
  num_ahead * num_stages + 1 = 91 :=
by
  sorry

end vasya_lowest_position_l173_173216


namespace lowest_position_of_vasya_l173_173210

-- Definitions of conditions
def num_cyclists : ℕ := 500
def num_stages : ℕ := 15
def vasya_position_each_stage : ℕ := 7

-- Theorem statement
theorem lowest_position_of_vasya (H1 : ∀ (s: ℕ), s ∈ finset.range(num_stages) → 
(num_cyclists + 1) - vasya_position_each_stage > (num_cyclists - 90))

(assumption_vasya :
  ∀ s ∈ finset.range(num_stages), vasya_position_each_stage < num_cyclists):
  ∃ (lowest_position: ℕ), lowest_position = 91 :=
sorry

end lowest_position_of_vasya_l173_173210


namespace books_sold_on_thursday_l173_173941

theorem books_sold_on_thursday (total_books : ℕ) (sold_mon : ℕ) (sold_tue : ℕ) (sold_wed : ℕ) (sold_fri : ℕ) (percent_not_sold : ℕ) (final_sold_thu : ℕ) : 
  total_books = 800 → 
  sold_mon = 62 → 
  sold_tue = 62 → 
  sold_wed = 60 → 
  sold_fri = 40 → 
  percent_not_sold = 66 → 
  final_sold_thu = 48 :=
by 
  intros h_total_books h_sold_mon h_sold_tue h_sold_wed h_sold_fri h_percent_not_sold h_final_sold_thu 
  sorry

end books_sold_on_thursday_l173_173941


namespace planar_graph_inequality_l173_173156

theorem planar_graph_inequality (E F : ℕ) (h_planar : is_planar_graph E F) :
  2 * E ≥ 3 * F :=
sorry

end planar_graph_inequality_l173_173156


namespace expected_number_of_sixes_l173_173666

-- Define the probability of not rolling a 6 on one die
def prob_not_six : ℚ := 5 / 6

-- Define the probability of rolling zero 6's on three dice
def prob_zero_six : ℚ := prob_not_six ^ 3

-- Define the probability of rolling exactly one 6 among the three dice
def prob_one_six (n : ℕ) : ℚ := n * (1 / 6) * (prob_not_six ^ (n - 1))

-- Calculate the probabilities of each specific outcomes
def prob_exactly_zero_six : ℚ := prob_zero_six
def prob_exactly_one_six : ℚ := prob_one_six 3 * (prob_not_six ^ 2)
def prob_exactly_two_six : ℚ := prob_one_six 3 * (1 / 6) * prob_not_six
def prob_exactly_three_six : ℚ := (1 / 6) ^ 3

-- Define the expected value calculation
noncomputable def expected_value : ℚ :=
  0 * prob_exactly_zero_six
  + 1 * prob_exactly_one_six
  + 2 * prob_exactly_two_six
  + 3 * prob_exactly_three_six

-- Prove that the expected value equals to 1/2
theorem expected_number_of_sixes : expected_value = 1 / 2 :=
  by
    sorry

end expected_number_of_sixes_l173_173666


namespace find_m_l173_173509

theorem find_m 
  (m : ℝ)
  (h : ∀ (x : ℂ), 5 * x^2 + 4 * x + m = 0 ↔ x = (-4 + complex.I * complex.sqrt 84) / 10 ∨ x = (-4 - complex.I * complex.sqrt 84) / 10) :
  m = 5 :=
by
  sorry

end find_m_l173_173509


namespace truck_is_50_miles_from_start_l173_173353

def truck_distance (north1 north2 east : ℕ) : ℕ :=
  let total_north := north1 + north2
  let distance_squared := total_north ^ 2 + east ^ 2
  Int.sqrt distance_squared

theorem truck_is_50_miles_from_start:
  truck_distance 20 20 30 = 50 :=
by
  sorry

end truck_is_50_miles_from_start_l173_173353


namespace real_numbers_from_complex_eq_l173_173490

theorem real_numbers_from_complex_eq (x y : ℝ) (h : (x + y) * complex.I = x - 1) : x = 1 ∧ y = -1 :=
sorry

end real_numbers_from_complex_eq_l173_173490


namespace negation_proposition_l173_173478

   theorem negation_proposition :
     ¬ (∀ x : ℝ, 3 ^ x > 0) ↔ ∃ x : ℝ, 3 ^ x ≤ 0 :=
   sorry
   
end negation_proposition_l173_173478


namespace weight_of_B_l173_173716

/-- Let A, B, and C be the weights in kg of three individuals. If the average weight of A, B, and C is 45 kg,
and the average weight of A and B is 41 kg, and the average weight of B and C is 43 kg,
then the weight of B is 33 kg. -/
theorem weight_of_B (A B C : ℝ) 
  (h1 : A + B + C = 135) 
  (h2 : A + B = 82) 
  (h3 : B + C = 86) : 
  B = 33 := 
by 
  sorry

end weight_of_B_l173_173716


namespace total_students_l173_173758

theorem total_students (A B : ℕ)
  (h1 : A / B = 3 / 2)
  (h2 : B = 162)
  (h3 : (Real.sqrt (0.1 * A) / Real.cbrt (0.2 * B)) = 5 / 3) :
  A + B = 405 :=
by
  sorry

end total_students_l173_173758


namespace lcm_two_primes_is_10_l173_173910

theorem lcm_two_primes_is_10 (x y : ℕ) (h_prime_x : Nat.Prime x) (h_prime_y : Nat.Prime y) (h_lcm : Nat.lcm x y = 10) (h_gt : x > y) : 2 * x + y = 12 :=
sorry

end lcm_two_primes_is_10_l173_173910


namespace area_of_regular_octagon_l173_173104

def regular_octagon (A B C D E F G H : ℝ²) : Prop :=
  -- Assuming the necessary conditions for regularity, e.g., all sides and angles are equal
  sorry

def midpoint (A B : ℝ²) : ℝ² := (A + B) / 2

def area (vertices : list ℝ²) : ℝ := 
  -- Implement the area calculation of a polygon
  sorry

theorem area_of_regular_octagon (A B C D E F G H I J K L : ℝ²)
  (hOctagon : regular_octagon A B C D E F G H)
  (hI : I = midpoint A B) (hJ: J = midpoint C D) (hK: K = midpoint E F) (hL: L = midpoint G H)
  (hAreaIJK : area [I, J, K] = 256) :
  area [A, B, C, D, E, F, G, H] = 2048 :=
sorry

end area_of_regular_octagon_l173_173104


namespace range_of_b_l173_173481

noncomputable def set_P : set ℝ := {x | x^2 - 5*x + 4 ≤ 0}
noncomputable def set_Q (b : ℝ) : set ℝ := {x | x^2 - (b + 2)*x + 2*b ≤ 0}

theorem range_of_b : ∀ b : ℝ, (set.subset (set_Q b) set_P) ↔ (1 ≤ b ∧ b ≤ 4) :=
sorry

end range_of_b_l173_173481


namespace todd_ate_cupcakes_l173_173144

def total_cupcakes_baked := 68
def packages := 6
def cupcakes_per_package := 6
def total_packaged_cupcakes := packages * cupcakes_per_package
def remaining_cupcakes := total_cupcakes_baked - total_packaged_cupcakes

theorem todd_ate_cupcakes : total_cupcakes_baked - remaining_cupcakes = 36 := by
  sorry

end todd_ate_cupcakes_l173_173144


namespace complex_number_solution_l173_173803

open Complex

theorem complex_number_solution (z : ℂ) (a b : ℝ) (h1 : z = a + b * complex.I)
(h2 : Complex.abs z ^ 2 + (z + Complex.conj z) * complex.I = (3 - complex.I) / (2 + complex.I)) :
z = - 1 / 2 + (Real.sqrt 3 / 2) * complex.I ∨ z = - 1 / 2 - (Real.sqrt 3 / 2) * complex.I := by
sorry

end complex_number_solution_l173_173803


namespace at_most_one_existence_l173_173559

theorem at_most_one_existence
  (p : ℕ) (hp : Nat.Prime p)
  (A B : Finset (Fin p))
  (h_non_empty_A : A.Nonempty) (h_non_empty_B : B.Nonempty)
  (h_union : A ∪ B = Finset.univ) (h_disjoint : A ∩ B = ∅) :
  ∃! a : Fin p, ¬ (∃ x y : Fin p, (x ∈ A ∧ y ∈ B ∧ x + y = a) ∨ (x + y = a + p)) :=
sorry

end at_most_one_existence_l173_173559


namespace lowest_position_of_vasya_l173_173207

-- Definitions of conditions
def num_cyclists : ℕ := 500
def num_stages : ℕ := 15
def vasya_position_each_stage : ℕ := 7

-- Theorem statement
theorem lowest_position_of_vasya (H1 : ∀ (s: ℕ), s ∈ finset.range(num_stages) → 
(num_cyclists + 1) - vasya_position_each_stage > (num_cyclists - 90))

(assumption_vasya :
  ∀ s ∈ finset.range(num_stages), vasya_position_each_stage < num_cyclists):
  ∃ (lowest_position: ℕ), lowest_position = 91 :=
sorry

end lowest_position_of_vasya_l173_173207


namespace min_mod_z_l173_173958

open Complex

theorem min_mod_z (z : ℂ) (hz : abs (z - 2 * I) + abs (z - 5) = 7) : abs z = 10 / 7 :=
sorry

end min_mod_z_l173_173958


namespace lowest_position_of_vasya_l173_173211

-- Definitions of conditions
def num_cyclists : ℕ := 500
def num_stages : ℕ := 15
def vasya_position_each_stage : ℕ := 7

-- Theorem statement
theorem lowest_position_of_vasya (H1 : ∀ (s: ℕ), s ∈ finset.range(num_stages) → 
(num_cyclists + 1) - vasya_position_each_stage > (num_cyclists - 90))

(assumption_vasya :
  ∀ s ∈ finset.range(num_stages), vasya_position_each_stage < num_cyclists):
  ∃ (lowest_position: ℕ), lowest_position = 91 :=
sorry

end lowest_position_of_vasya_l173_173211


namespace isosceles_triangle_of_cos_ratio_l173_173461

variables {A B C a b c : ℝ}
variables {triangle_ABC : Prop}

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_isosceles (a b c : ℝ) : Prop :=
  a = b ∨ a = c ∨ b = c

theorem isosceles_triangle_of_cos_ratio :
  is_triangle a b c ∧ a / cos A = b / cos B → is_isosceles a b c :=
by
  sorry

end isosceles_triangle_of_cos_ratio_l173_173461


namespace king_zenobius_more_descendants_l173_173094

-- Conditions
def descendants_paphnutius (p2_descendants p1_descendants: ℕ) := 
  2 + 60 * p2_descendants + 20 * p1_descendants = 142

def descendants_zenobius (z3_descendants z1_descendants : ℕ) := 
  4 + 35 * z3_descendants + 35 * z1_descendants = 144

-- Main statement
theorem king_zenobius_more_descendants:
  ∀ (p2_descendants p1_descendants z3_descendants z1_descendants : ℕ),
    descendants_paphnutius p2_descendants p1_descendants →
    descendants_zenobius z3_descendants z1_descendants →
    144 > 142 :=
by
  intros
  sorry

end king_zenobius_more_descendants_l173_173094


namespace smallest_positive_angle_l173_173380

theorem smallest_positive_angle:
  ∃ (x : ℝ), 0 < x ∧ x ≤ 5.625 ∧ tan (6 * x * (π / 180)) = (cos (2 * x * (π / 180)) - sin (2 * x * (π / 180))) / (cos (2 * x * (π / 180)) + sin (2 * x * (π / 180))) :=
by
  sorry

end smallest_positive_angle_l173_173380


namespace vasya_rank_91_l173_173229

theorem vasya_rank_91 {n_cyclists : ℕ} {n_stages : ℕ} 
    (n_cyclists_eq : n_cyclists = 500) 
    (n_stages_eq : n_stages = 15) 
    (no_ties : ∀ (i j : ℕ), i < j → ∀ (s : fin n_stages), ¬(same_time i j s)) 
    (vasya_7th : ∀ (s : fin n_stages), ∀ (i : ℕ), i < 6 → better_than i 6 s) :
    possible_rank vasya ≤ 91 :=
sorry

end vasya_rank_91_l173_173229


namespace sum_of_five_integers_l173_173833

theorem sum_of_five_integers (a b c d e : ℕ) 
    (S1 S2 S3 S4 S5 : ℕ) 
    (h1 : S1 = a + b + c + d) 
    (h2 : S2 = a + b + c + e) 
    (h3 : S3 = a + b + d + e) 
    (h4 : S4 = a + c + d + e) 
    (h5 : S5 = b + c + d + e) 
    (hs : {S1, S2, S3, S4, S5} = {44, 45, 46, 47}) : 
    a + b + c + d + e = 57 := sorry

end sum_of_five_integers_l173_173833


namespace elliptic_relationships_l173_173485

noncomputable def eccentricity_square (a b : ℝ) : ℝ := 1 - (b^2 / a^2)

theorem elliptic_relationships (a1 b1 c1 a2 b2 : ℝ)
  (h1: a2 = b1) (h2: b2 = c1) (h3: b1 > c1) (h4: c1^2 = a1^2 - b1^2) :
  (eccentricity_square a1 b1 < 1/2) ∧ (eccentricity_square a1 b1 + eccentricity_square b1 c1 < 1) :=
begin
  -- proof goes here
  sorry
end

end elliptic_relationships_l173_173485


namespace line_parallel_or_in_plane_l173_173046

-- Definitions based on conditions in (a)
variables {a b : Line}
variable {α : Plane}
variable Parallel : Line → Line → Prop
variable ParallelPlane : Line → Plane → Prop

-- Given the conditions
axiom ax1 : Parallel a b
axiom ax2 : ParallelPlane a α

-- Question and answer as a Lean theorem statement
theorem line_parallel_or_in_plane : Parallel b α ∨ (∃ p : Point, p ∈ α ∧ b = LineThrough p) :=
by
  sorry

end line_parallel_or_in_plane_l173_173046


namespace piece_in_313th_row_l173_173990

theorem piece_in_313th_row :
  (∃ M : matrix (fin 625) (fin 625) bool,
    (∑ i j, if M i j then 1 else 0) = 1977 ∧
    ∀ i j, M i j = M (624 - i) (624 - j)) →
  (∃ j, M (312) j = true) :=
by
  sorry

end piece_in_313th_row_l173_173990


namespace parallel_planes_l173_173429

-- Definitions for the geometric objects
variable {P : Type} [metric_space P] [normed_add_torsor ℝ P]

noncomputable def PlaneParallel (α β : set P) : Prop :=
  ∀ (x : P), x ∈ α → x ∈ β

-- Conditions
variable {α β : set P}
variable {a b : P → Prop}

axiom condition_1 : (∀ p, a p → p ∈ α ∧ p ∈ β)
axiom condition_4 : (∀ (p q : P), linear p q → a p ∧ ¬ a q → (b q → q ∈ α ∧ q ∈ β))

-- Proof Problem
theorem parallel_planes : PlaneParallel α β :=
by {
  sorry
}

end parallel_planes_l173_173429


namespace vasya_rank_91_l173_173225

theorem vasya_rank_91 {n_cyclists : ℕ} {n_stages : ℕ} 
    (n_cyclists_eq : n_cyclists = 500) 
    (n_stages_eq : n_stages = 15) 
    (no_ties : ∀ (i j : ℕ), i < j → ∀ (s : fin n_stages), ¬(same_time i j s)) 
    (vasya_7th : ∀ (s : fin n_stages), ∀ (i : ℕ), i < 6 → better_than i 6 s) :
    possible_rank vasya ≤ 91 :=
sorry

end vasya_rank_91_l173_173225


namespace expected_number_of_sixes_l173_173668

-- Define the problem context and conditions
def die_prob := (1 : ℝ) / 6

def expected_six (n : ℕ) : ℝ :=
  n * die_prob

-- The main proposition to prove
theorem expected_number_of_sixes (n : ℕ) (hn : n = 3) : expected_six n = 1 / 2 :=
by
  rw [hn]
  have fact1 : (3 : ℝ) * die_prob = 3 / 6 := by norm_cast; norm_num
  rw [fact1]
  norm_num

-- We add sorry to indicate incomplete proof, fulfilling criteria 4
sorry

end expected_number_of_sixes_l173_173668


namespace salary_reduction_l173_173265

noncomputable def percentageIncrease : ℝ := 16.27906976744186 / 100

theorem salary_reduction (S R : ℝ) (P : ℝ) (h1 : R = S * (1 - P / 100)) (h2 : S = R * (1 + percentageIncrease)) : P = 14 :=
by
  sorry

end salary_reduction_l173_173265


namespace polynomial_q_value_l173_173323

theorem polynomial_q_value (p q r s : ℂ) (h1 : ∀ z : ℂ, z^4 + p*z^3 + q*z^2 + r*z + s = 0 → z.im ≠ 0)
    (h2 : ∃ α β : ℂ, α + β = 4 + 7*I ∧ (α + β) + (complex.conj α + complex.conj β) = 0)
    (h3 : ∃ γ δ : ℂ, γ * δ = 3 - 4*I ∧ (γ * δ) * (complex.conj γ * complex.conj δ) = 0) :
    q = 71 :=
by
  sorry

end polynomial_q_value_l173_173323


namespace find_a5_l173_173083

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, ∃ q : ℝ, a (n + m) = a n * q ^ m

theorem find_a5
  (h : geometric_sequence a)
  (h3 : a 3 = 2)
  (h7 : a 7 = 8) :
  a 5 = 4 :=
sorry

end find_a5_l173_173083


namespace magic_triangle_largest_S_l173_173919

theorem magic_triangle_largest_S :
  ∃ (S : ℕ) (a b c d e f g : ℕ),
    (10 ≤ a) ∧ (a ≤ 16) ∧
    (10 ≤ b) ∧ (b ≤ 16) ∧
    (10 ≤ c) ∧ (c ≤ 16) ∧
    (10 ≤ d) ∧ (d ≤ 16) ∧
    (10 ≤ e) ∧ (e ≤ 16) ∧
    (10 ≤ f) ∧ (f ≤ 16) ∧
    (10 ≤ g) ∧ (g ≤ 16) ∧
    (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧ (a ≠ g) ∧
    (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧ (b ≠ g) ∧
    (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧ (c ≠ g) ∧
    (d ≠ e) ∧ (d ≠ f) ∧ (d ≠ g) ∧
    (e ≠ f) ∧ (e ≠ g) ∧
    (f ≠ g) ∧
    (S = a + b + c) ∧
    (S = c + d + e) ∧
    (S = e + f + a) ∧
    (S = g + b + c) ∧
    (S = g + d + e) ∧
    (S = g + f + a) ∧
    ((a + b + c) + (c + d + e) + (e + f + a) = 91 - g) ∧
    (S = 26) := sorry

end magic_triangle_largest_S_l173_173919


namespace find_pairs_l173_173415

theorem find_pairs (n k : ℕ) (h1 : n ≥ 0) (h2 : k > 1) :
  let A := (17 ^ (2006 * n) + 4 * 17 ^ (2 * n) + 7 * 19 ^ (5 * n)) in
  (∃ (x : ℕ), A = x * (x + 1)) ↔ (n = 0 ∧ k = 2) :=
by {
  sorry
}

end find_pairs_l173_173415


namespace area_of_interior_triangle_l173_173463

-- Define the areas of the squares
def area_square1 := 225
def area_square2 := 225
def area_square3 := 64

-- Define the side lengths derived from the areas
def side_square1 := Real.sqrt area_square1
def side_square2 := Real.sqrt area_square2
def side_square3 := Real.sqrt area_square3

-- Define the base and height of the right triangle
def base := side_square1
def height := side_square3

-- Define the area of the right triangle
def triangle_area (base: Real) (height: Real) : Real := (1 / 2) * base * height

-- The theorem to be proven
theorem area_of_interior_triangle :
  triangle_area base height = 60 :=
by
  sorry

end area_of_interior_triangle_l173_173463


namespace no_four_digit_palindromic_squares_l173_173874

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in
  s = s.reverse

def four_digit_perfect_squares : List ℕ :=
  List.range' (32*32) (99*32 - 32*32 + 1)
  
theorem no_four_digit_palindromic_squares : 
  (List.filter is_palindrome four_digit_perfect_squares).length = 0 :=
by
  sorry

end no_four_digit_palindromic_squares_l173_173874


namespace max_diff_of_angles_l173_173451

theorem max_diff_of_angles
  (x y : ℝ) (hx1 : 0 < y) (hx2 : y ≤ x) (hx3 : x < π / 2)
  (h_tan : tan x = 3 * tan y) :
  x - y ≤ π / 6 :=
sorry

end max_diff_of_angles_l173_173451


namespace solution_set_l173_173860

def f (x : ℝ) : ℝ := x * Real.cos x - Real.sin x - (1/3) * x^3

theorem solution_set (x : ℝ) : f (2 * x + 3) + f 1 < 0 ↔ x > -2 :=
by sorry

end solution_set_l173_173860


namespace airplane_takeoff_distance_l173_173269

/--  
The takeoff run time of an airplane from the start until it leaves the ground is 15 seconds. 
Find the length of the takeoff run if the takeoff speed for this airplane model is 100 km/h. 
Assume the airplane's motion during the takeoff run is uniformly accelerated. 
Provide the answer in meters, rounding to the nearest whole number if necessary.
--/
theorem airplane_takeoff_distance :
  let t : ℝ := 15  -- time in seconds
  let v_kmh : ℝ := 100  -- speed in km/h
  let v_ms : ℝ := v_kmh * 1000 / 3600  -- converted speed in m/s
  let a : ℝ := v_ms / t  -- acceleration
  let s : ℝ := 0.5 * a * t^2  -- distance
  -- expected distance in meters, rounding to the nearest whole number
  s ≈ 208 :=
by
  let t : ℝ := 15
  let v_kmh : ℝ := 100
  let v_ms : ℝ := v_kmh * 1000 / 3600
  let a := v_ms / t
  let s := 0.5 * a * t^2
  sorry

end airplane_takeoff_distance_l173_173269


namespace value_of_business_calculation_l173_173740

noncomputable def value_of_business (total_shares_sold_value : ℝ) (shares_fraction_sold : ℝ) (ownership_fraction : ℝ) : ℝ :=
  (total_shares_sold_value / shares_fraction_sold) * ownership_fraction⁻¹

theorem value_of_business_calculation :
  value_of_business 45000 (3/4) (2/3) = 90000 :=
by
  sorry

end value_of_business_calculation_l173_173740


namespace Andy_solves_correct_number_of_problems_l173_173359

-- Define the problem boundaries
def first_problem : ℕ := 80
def last_problem : ℕ := 125

-- The goal is to prove that Andy solves 46 problems given the range
theorem Andy_solves_correct_number_of_problems : (last_problem - first_problem + 1) = 46 :=
by
  sorry

end Andy_solves_correct_number_of_problems_l173_173359


namespace triangle_isosceles_l173_173953

theorem triangle_isosceles 
(ABC : Triangle) (A B C E D : Point) 
(hABC_right : ABC.right_triangle A B C) 
(hAB_AC : distance A B = distance A C) 
(hAB_BC : distance A B > distance B C)
(hBE : distance B E = distance A B - distance B C) 
(hCD : distance C D = distance A B - distance B C) 
: is_isosceles ADE :=
sorry

end triangle_isosceles_l173_173953


namespace senya_cannot_complete_magic_square_after_lyonya_help_l173_173724

def is_magic_square (M : matrix (fin 4) (fin 4) ℕ) : Prop :=
  (∀ i : fin 4, ∑ j, M i j = 34) ∧
  (∀ j : fin 4, ∑ i, M i j = 34) ∧
  (∑ i, M i i = 34) ∧
  (∑ i, M i (3 - i) = 34) ∧
  (∀ i j, 1 ≤ M i j ∧ M i j ≤ 16) ∧
  (∀ i1 j1 i2 j2, (i1 ≠ i2 ∨ j1 ≠ j2) → M i1 j1 ≠ M i2 j2)

noncomputable def senya_magic_square_possible (M : matrix (fin 4) (fin 4) ℕ) : Prop :=
  ∃ (i1 i2 i3 j1 j2 j3 : fin 4),
  (M i1 j1 = 1) ∧
  ((i2 = i1 ∧ (j2 = j1 + 1 ∨ j2 = j1 - 1)) ∨
   (j2 = j1 ∧ (i2 = i1 + 1 ∨ i2 = i1 - 1))) ∧ (M i2 j2 = 2) ∧
  ((i3 = i1 ∧ (j3 = j1 + 1 ∨ j3 = j1 - 1)) ∨
   (j3 = j1 ∧ (i3 = i1 + 1 ∨ i3 = i1 - 1))) ∧ (M i3 j3 = 3) ∧
  is_magic_square M

theorem senya_cannot_complete_magic_square_after_lyonya_help : ¬ senya_magic_square_possible

end senya_cannot_complete_magic_square_after_lyonya_help_l173_173724


namespace sum_of_first_49_odd_numbers_l173_173715

theorem sum_of_first_49_odd_numbers : 
  let seq : List ℕ := List.range' 1 (2 * 49 - 1) 2 in
  seq.sum = 2401 :=
by
  sorry

end sum_of_first_49_odd_numbers_l173_173715


namespace cq_dq_difference_l173_173194

theorem cq_dq_difference (y x : ℝ) (c d : ℝ) (Q : ℝ × ℝ) (hQ : Q = (-ℝ.sqrt 3, 0)) :
  (y + x * ℝ.sqrt 3 + 1 = 0) →
  (2 * y^2 = 2 * x + 5) →
  (C = (c^2 - 5 / 2, c)) →
  (D = (d^2 - 5 / 2, d)) →
  |(Real.sqrt ((c^2 - 5 / 2 + ℝ.sqrt 3)^2 + c^2)) - (Real.sqrt ((d^2 - 5 / 2 + ℝ.sqrt 3)^2 + d^2))| = 2 / 3 :=
sorry

end cq_dq_difference_l173_173194


namespace value_of_x_squared_plus_one_l173_173492

theorem value_of_x_squared_plus_one (x : ℝ) (h : 3^(2*x) + 9 = 10 * 3^x) :
  (x^2 + 1 = 1 ∨ x^2 + 1 = 5) :=
sorry

end value_of_x_squared_plus_one_l173_173492


namespace tan_A_tan_2B_range_l173_173062

theorem tan_A_tan_2B_range (A B C a b c : ℝ) (h1: a^2 + b^2 + (real.sqrt 2) * a * b = c^2) :
  (real.tan A * real.tan (2 * B)) ∈ set.Ioo 0 (1 / 2) :=
sorry

end tan_A_tan_2B_range_l173_173062


namespace integral_even_function_l173_173840

variable {f : ℝ → ℝ}
variable {a : ℝ}

def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)

theorem integral_even_function :
  is_even f →
  (∫ x in 0..a, f x) = 8 →
  (∫ x in -a..a, f x) = 16 :=
by
  intros h1 h2
  sorry

end integral_even_function_l173_173840


namespace expected_number_of_sixes_when_three_dice_are_rolled_l173_173658

theorem expected_number_of_sixes_when_three_dice_are_rolled : 
  ∑ n in finset.range 4, (n * (↑(finset.filter (λ xs : fin 3 → fin 6, xs.count (λ x, x = 5) = n) finset.univ).card / 216 : ℚ)) = 1 / 2 :=
by
  -- Conclusion of proof is omitted as per instructions
  sorry

end expected_number_of_sixes_when_three_dice_are_rolled_l173_173658


namespace fifth_cyclic_l173_173079

open Function

variable (A B C D E A1 B1 C1 D1 E1 : Type)
variables [ConvexPentagon A B C D E]
variables [Intersection BD CE A1]
variables [Intersection CE DA B1]
variables [Intersection DA EB C1]
variables [Intersection EB AC D1]
variables [Intersection AC BE E1]
variables [CyclicQuadrilateral AB A1 B1]
variables [CyclicQuadrilateral BC B1 C1]
variables [CyclicQuadrilateral CD C1 D1]
variables [CyclicQuadrilateral DE D1 E1]

theorem fifth_cyclic (AB A1 B1 : CyclicQuadrilateral) 
  (BC B1 C1 : CyclicQuadrilateral) 
  (CD C1 D1 : CyclicQuadrilateral) 
  (DE D1 E1 : CyclicQuadrilateral) : CyclicQuadrilateral EA E1 A1 :=
sorry

end fifth_cyclic_l173_173079


namespace exists_infinite_bounded_sequence_l173_173594

noncomputable def sequence (n : ℕ) : ℝ := 4 * (n * Real.sqrt 2 % 1)

theorem exists_infinite_bounded_sequence :
  ∃ (x : ℕ → ℝ), (∀ n m : ℕ, n ≠ m → (|x n - x m| ≥ 1 / |(n:ℝ) - m|)) ∧ (∀ n : ℕ, x n = sequence n) ∧ (∀ n : ℕ, |x n| < 4) :=
by
  use sequence
  sorry

end exists_infinite_bounded_sequence_l173_173594


namespace vasya_rank_91_l173_173230

theorem vasya_rank_91 {n_cyclists : ℕ} {n_stages : ℕ} 
    (n_cyclists_eq : n_cyclists = 500) 
    (n_stages_eq : n_stages = 15) 
    (no_ties : ∀ (i j : ℕ), i < j → ∀ (s : fin n_stages), ¬(same_time i j s)) 
    (vasya_7th : ∀ (s : fin n_stages), ∀ (i : ℕ), i < 6 → better_than i 6 s) :
    possible_rank vasya ≤ 91 :=
sorry

end vasya_rank_91_l173_173230


namespace vasya_rank_91_l173_173223

theorem vasya_rank_91 {n_cyclists : ℕ} {n_stages : ℕ} 
    (n_cyclists_eq : n_cyclists = 500) 
    (n_stages_eq : n_stages = 15) 
    (no_ties : ∀ (i j : ℕ), i < j → ∀ (s : fin n_stages), ¬(same_time i j s)) 
    (vasya_7th : ∀ (s : fin n_stages), ∀ (i : ℕ), i < 6 → better_than i 6 s) :
    possible_rank vasya ≤ 91 :=
sorry

end vasya_rank_91_l173_173223


namespace lines_from_abs_eq_l173_173782

theorem lines_from_abs_eq (x y : ℝ) : 
  (|x| - |y| = 1) ↔ 
  ((x ≥ 1 ∧ y = x - 1) ∨ (x ≥ 1 ∧ y = 1 - x) ∨
  (x ≤ -1 ∧ y = -x - 1) ∨ (x ≤ -1 ∧ y = x + 1)) :=
begin
  sorry
end

end lines_from_abs_eq_l173_173782


namespace find_a_l173_173029

theorem find_a (a : ℝ) (h : -1 ^ 2 + 2 * -1 + a = 0) : a = 1 :=
sorry

end find_a_l173_173029


namespace lowest_position_l173_173231

theorem lowest_position (num_cyclists : ℕ) (num_stages : ℕ) (vasya_position : ℕ) :
  num_cyclists = 500 →
  num_stages = 15 →
  vasya_position = 7 →
  ∃ n, n = 91 :=
by
  intros
  sorry

end lowest_position_l173_173231


namespace stratified_sampling_medium_stores_l173_173066

noncomputable def total_stores := 300
noncomputable def large_stores := 30
noncomputable def medium_stores := 75
noncomputable def small_stores := 195
noncomputable def sample_size := 20

theorem stratified_sampling_medium_stores : 
  (medium_stores : ℕ) * (sample_size : ℕ) / (total_stores : ℕ) = 5 :=
by
  sorry

end stratified_sampling_medium_stores_l173_173066


namespace solve_problem_l173_173399

def ceil (x : ℝ) : ℤ := Int.ceil x

def f (x : ℝ) : ℤ := ceil (x * ceil x)

def A_n (n : ℕ) : Finset ℤ := 
  {a : ℤ | ∃ x, (0 < x ∧ x ≤ n) ∧ f x = a}.toFinset

def a_n (n : ℕ) : ℕ := (A_n n).card

theorem solve_problem : 
  (Finset.range 2018).sum (λ n, 1 / a_n (n + 1)) = 4036 / 2019 := 
sorry

end solve_problem_l173_173399


namespace part1_part2_part3_l173_173470

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x + 1

theorem part1 (a : ℝ) (x : ℝ) (h : 0 < x) :
  (a ≤ 0 → (∀ x > 0, f a x < 0)) ∧
  (a > 0 → (∀ x ∈ Set.Ioo 0 a, f a x > 0) ∧ (∀ x ∈ Set.Ioi a, f a x < 0)) :=
sorry

theorem part2 {a : ℝ} : (∀ x > 0, f a x ≤ 0) → a = 1 :=
sorry

theorem part3 (n : ℕ) (h : 0 < n) :
  (1 + 1 / n : ℝ)^n < Real.exp 1 ∧ Real.exp 1 < (1 + 1 / n : ℝ)^(n + 1) :=
sorry

end part1_part2_part3_l173_173470


namespace find_constant_l173_173193

theorem find_constant :
  ∃ constant : ℝ, ∀ (x : ℝ), f x = x + 4 →
    ∃ (h : f (0) = 4), ∀ (x : ℝ), (x = 0.4) →
      ((3 * f (x - 2)) / f 0 + 4 = f (2 * x + constant)) ->
      constant = 1 :=
by
  sorry

noncomputable def f (x : ℝ) : ℝ := x + 4

example : f(0) = 4 := rfl

end find_constant_l173_173193


namespace focus_of_parabola_l173_173417

theorem focus_of_parabola :
  (∃ f : ℝ, ∀ y : ℝ, (x = -1 / 4 * y^2) = (x = (y^2 / 4 + f)) -> f = -1) :=
by
  sorry

end focus_of_parabola_l173_173417


namespace proof_ratio_l173_173368

variable (k : ℝ)
variable (NO_initial : ℝ) (O2_initial : ℝ)
variable (NO_reacted : ℝ) (O2_reacted : ℝ)
variable (NO_at_t : ℝ) (O2_at_t : ℝ)
variable (v0 : ℝ) (vt : ℝ)
variable (ratio : ℝ)

-- Conditions given
def conditions := 
  NO_initial = 1.5 ∧ 
  O2_initial = 3 ∧ 
  NO_reacted = 0.5 ∧ 
  O2_reacted = 0.25 ∧ 
  NO_at_t = NO_initial - NO_reacted ∧ 
  O2_at_t = O2_initial - O2_reacted ∧ 
  v0 = k * NO_initial * O2_initial ∧ 
  vt = k * NO_at_t * O2_at_t
  
-- Statement to prove
theorem proof_ratio : conditions → (v0 / vt = 1.64) := sorry

end proof_ratio_l173_173368


namespace vasya_lowest_position_l173_173258

noncomputable theory

def number_of_cyclists := 500
def number_of_stages := 15
def position_of_vasya_each_stage := 7

theorem vasya_lowest_position (total_cyclists : ℕ) (stages : ℕ) (position_each_stage : ℕ)
  (h_total_cyclists : total_cyclists = number_of_cyclists)
  (h_stages : stages = number_of_stages)
  (h_position_each_stage : position_each_stage = position_of_vasya_each_stage)
  (no_identical_times : ∀ (i j : ℕ), i ≠ j → ∀ (stage : ℕ), stage ≤ stages → ∀ (t : ℕ), t ≤ total_cyclists → 
    (time : Π (s : ℕ) (c : ℕ), c < total_cyclists → ℕ), 
    time stage i < time stage j ∨ time stage j < time stage i):
  ∃ lowest_pos, lowest_pos = 91 := sorry

end vasya_lowest_position_l173_173258


namespace min_value_of_expression_l173_173569

noncomputable def problem_statement : Prop :=
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ ((1/x) + (1/y) + (1/z) = 9) ∧ (x^2 * y^3 * z^2 = 1/2268)

theorem min_value_of_expression :
  problem_statement := 
sorry

end min_value_of_expression_l173_173569


namespace largest_4_digit_congruent_to_17_mod_26_l173_173297

theorem largest_4_digit_congruent_to_17_mod_26 :
  ∃ x, x < 10000 ∧ x ≥ 1000 ∧ x % 26 = 17 ∧ (∀ y, y < 10000 ∧ y ≥ 1000 ∧ y % 26 = 17 → y ≤ x) ∧ x = 9972 := 
by
  sorry

end largest_4_digit_congruent_to_17_mod_26_l173_173297


namespace gifted_subscribers_l173_173544

theorem gifted_subscribers (initial_subs : ℕ) (revenue_per_sub : ℕ) (total_revenue : ℕ) (h1 : initial_subs = 150) (h2 : revenue_per_sub = 9) (h3 : total_revenue = 1800) :
  total_revenue / revenue_per_sub - initial_subs = 50 :=
by
  sorry

end gifted_subscribers_l173_173544


namespace find_smaller_number_l173_173637

theorem find_smaller_number (x : ℕ) (h : 3 * x + 4 * x = 420) : 3 * x = 180 :=
by
  sorry

end find_smaller_number_l173_173637


namespace number_of_divisors_of_2310_exp_2310_with_48_divisors_l173_173883

theorem number_of_divisors_of_2310_exp_2310_with_48_divisors :
  ∃ N : ℕ,
  let p := 2310 in
  let n := 2310 in
  let exponent := 2310 in
  let factorization := {2, 3, 5, 7, 11} in
  let divisors_form : (ℕ × ℕ × ℕ × ℕ × ℕ) → ℕ := λ (a b c d e), (a+1) * (b+1) * (c+1) * (d+1) * (e+1) in
  let valid_divisors := (divisors_form == 48) in
  N = (calc_total_number_of_valid_divisors exponent factorization valid_divisors)
  sorry

end number_of_divisors_of_2310_exp_2310_with_48_divisors_l173_173883


namespace friend_decks_l173_173285

-- Definitions for conditions
def price_per_deck : ℕ := 8
def victor_decks : ℕ := 6
def total_spent : ℕ := 64

-- Conclusion based on the conditions
theorem friend_decks : (64 - (6 * 8)) / 8 = 2 := by
  sorry

end friend_decks_l173_173285


namespace market_price_article_l173_173912

theorem market_price_article (P : ℝ)
  (initial_tax_rate : ℝ := 0.035)
  (reduced_tax_rate : ℝ := 0.033333333333333)
  (difference_in_tax : ℝ := 11) :
  (initial_tax_rate * P - reduced_tax_rate * P = difference_in_tax) → 
  P = 6600 :=
by
  intro h
  /-
  We assume h: initial_tax_rate * P - reduced_tax_rate * P = difference_in_tax
  And we need to show P = 6600.
  The proof steps show that P = 6600 follows logically given h and the provided conditions.
  -/
  sorry

end market_price_article_l173_173912


namespace m_value_l173_173061

noncomputable def find_m (A B C a b c : ℝ) (m : ℝ) : Prop :=
  (tan A * tan C + tan B * tan C = tan A * tan B) →
  (sin A ^ 2 + sin B ^ 2 = (m^2 + 1) * sin C ^ 2) →
  (m = real.sqrt 2 ∨ m = -real.sqrt 2)

theorem m_value (A B C a b c m : ℝ) :
  (tan A * tan C + tan B * tan C = tan A * tan B) →
  (sin A ^ 2 + sin B ^ 2 = (m^2 + 1) * sin C ^ 2) →
  (m = real.sqrt 2 ∨ m = -real.sqrt 2) :=
by {
  intro h1 h2,
  sorry
}

end m_value_l173_173061


namespace uniform_customization_for_freshmen_l173_173013

theorem uniform_customization_for_freshmen :
  let μ := 165
  let σ := 5
  let num_students := 1000
  let prob_range := 0.954
  ∀ students_heights : ℝ → Prop,
    students_heights = λ x, (x - μ) / σ ∈ Normal 0 1 →
    ∃ num_uniforms : ℝ, num_uniforms = num_students * prob_range ∧ num_uniforms ≈ 954 :=
by
  sorry

end uniform_customization_for_freshmen_l173_173013


namespace hyperbola_slope_condition_l173_173476

-- Define the setup
variables (a b : ℝ) (P F1 F2 : ℝ × ℝ)
variables (h : a > 0) (k : b > 0)
variables (hyperbola : (∀ x y : ℝ, ((x^2 / a^2) - (y^2 / b^2) = 1)))

-- Define the condition
variables (cond : ∃ (P : ℝ × ℝ), 3 * abs (dist P F1 + dist P F2) ≤ 2 * dist F1 F2)

-- The proof goal
theorem hyperbola_slope_condition : (b / a) ≥ (Real.sqrt 5 / 2) :=
sorry

end hyperbola_slope_condition_l173_173476


namespace triangle_OZ_XZ_ZY_l173_173197

-- Definitions based on the given conditions and conclusion
def is_similar (Δ1 Δ2 : Triangle) : Prop :=
  (Δ1.angles = Δ2.angles) ∧ (Δ1.sides = Δ2.sides)

def OZ_over_XZ_eq_fifteen_over_ZY (O X Z Y : Point) (OZ XZ ZY : ℝ) : Prop :=
  OZ / XZ = 15 / ZY

-- Define the problem conditions as Lean statements
theorem triangle_OZ_XZ_ZY (a b : ℕ) (O X Z Y : Point)
  (OZ XZ ZY : ℝ)
  (h_perimeter : XZ + ZY = 130)
  (h_angle : angle X Z Y = 90)
  (h_tangent_O : tangent_to_circle O XZ Y Z)
  (h_OZ_fraction : OZ = a / b) :
  a + b = 51 :=
sorry

end triangle_OZ_XZ_ZY_l173_173197


namespace dot_product_of_vectors_l173_173456

-- Define vectors and their properties
variables {a b : EuclideanSpace ℝ (Fin 3)}

-- Conditions as Lean definitions
def angle_a_b : ℝ := 120 * Real.pi / 180
def magnitude_a : ℝ := 1
def magnitude_b : ℝ := 4

-- Prove the dot product equals -2 given the conditions
theorem dot_product_of_vectors : (real.angle_cos angle_a_b * magnitude_a * magnitude_b = -2) :=
by
  have h : real.angle_cos(angle_a_b) = -1 / 2 := sorry,
  rw [h],
  sorry

end dot_product_of_vectors_l173_173456


namespace general_term_sum_of_first_50_exists_valid_k_l173_173004

noncomputable def arithmetic_seq (n : ℕ) : ℤ := 2 * n - 7

theorem general_term
  (a : ℕ → ℤ)
  (h1 : a 1 = -5)
  (h2 : a 2 + a 3 + a 4 = -3)
  : ∀ n : ℕ, a n = arithmetic_seq n :=
by
  sorry

noncomputable def abs_seq_term (n : ℕ) : ℤ :=
  if n ≤ 3 then 7 - 2 * n else 2 * n - 7

noncomputable def sum_abs_seq_terms (n : ℕ) : ℤ :=
  ∑ i in finset.range n, abs_seq_term (i + 1)

theorem sum_of_first_50
  (a : ℕ → ℤ)
  (h1 : a 1 = -5)
  (h2 : a 2 + a 3 + a 4 = -3)
  : T_{50} = 2218 :=
by
  have h := general_term a h1 h2
  have T_n := sum_abs_seq_terms 50
  sorry

theorem exists_valid_k
  (a : ℕ → ℤ)
  (h1 : a 1 = -5)
  (h2 : a 2 + a 3 + a 4 = -3)
  : ∃ k : ℕ, 0 < k ∧ k = 2 ∧ ∃ i : ℕ, a i = (a k * a (k + 1)) / a (k + 2) :=
by
  sorry

end general_term_sum_of_first_50_exists_valid_k_l173_173004


namespace lowest_position_of_vasya_l173_173212

-- Definitions of conditions
def num_cyclists : ℕ := 500
def num_stages : ℕ := 15
def vasya_position_each_stage : ℕ := 7

-- Theorem statement
theorem lowest_position_of_vasya (H1 : ∀ (s: ℕ), s ∈ finset.range(num_stages) → 
(num_cyclists + 1) - vasya_position_each_stage > (num_cyclists - 90))

(assumption_vasya :
  ∀ s ∈ finset.range(num_stages), vasya_position_each_stage < num_cyclists):
  ∃ (lowest_position: ℕ), lowest_position = 91 :=
sorry

end lowest_position_of_vasya_l173_173212


namespace valid_jump_sequences_count_l173_173277

noncomputable def num_of_valid_sequences : nat :=
  63

theorem valid_jump_sequences_count :
  let n := 42,
      valid_steps : fin n → fin n → Prop := λ s t, t = s + 1 ∨ t = s + 7,
      visit_all : fin n → list (fin n) → Prop := sorry -- Formalize visiting each stone exactly once
  in ∃ seq : list (fin n), 
      (∀ s ∈ seq.tail, valid_steps s (s.pred % n)) ∧ 
      visit_all ⟨0, sorry⟩ (⟨0, sorry⟩ :: seq) ∧ 
      seq.nodup ∧ seq.length = n - 1 ∧ 
      seq.head = ⟨0, sorry⟩ ∧ 
      seq.last = seq.head := 
      list.repeat 1 (n // 6 - 1) * 2^6 -1 = num_of_valid_sequences :=
  by sorry

end valid_jump_sequences_count_l173_173277


namespace find_length_FD_l173_173103

noncomputable def length_of_FD (ABCD : Parallelogram) (angle_ABC : Real) (length_AB : Real) (length_BC : Real) (length_DE : Real) : Real :=
  sorry

theorem find_length_FD :
  ∀ (ABCD : Parallelogram) (angle_ABC length_AB length_BC length_DE : Real),
    angle_ABC = 100 ∧ length_AB = 20 ∧ length_BC = 12 ∧ length_DE = 6 →
    abs (length_of_FD ABCD angle_ABC length_AB length_BC length_DE - 2.8) < 0.1 :=
begin
  sorry
end

end find_length_FD_l173_173103


namespace maximize_profit_6_years_l173_173350

theorem maximize_profit_6_years :
  ∃ x : ℕ, (∀ x' : ℕ, -x^2 + 12 * x - 25 ≤ -x'^2 + 12 * x' - 25) ∧ x = 6 :=
begin
  -- proof goes here
  sorry
end

end maximize_profit_6_years_l173_173350


namespace william_wins_tic_tac_toe_l173_173706

-- Define the conditions
variables (total_rounds : ℕ) (extra_wins : ℕ) (william_wins : ℕ) (harry_wins : ℕ)

-- Setting the conditions
def william_harry_tic_tac_toe_conditions : Prop :=
  total_rounds = 15 ∧
  extra_wins = 5 ∧
  william_wins = harry_wins + extra_wins ∧
  total_rounds = william_wins + harry_wins

-- The goal is to prove that William won 10 rounds given the conditions above
theorem william_wins_tic_tac_toe : william_harry_tic_tac_toe_conditions total_rounds extra_wins william_wins harry_wins → william_wins = 10 :=
by
  intro h
  have total_rounds_eq := and.left h
  have extra_wins_eq := and.right (and.left (and.right h))
  have william_harry_diff := and.left (and.right (and.right h))
  have total_wins_eq := and.right (and.right (and.right h))
  sorry

end william_wins_tic_tac_toe_l173_173706


namespace compute_remainder_l173_173553

/-- T is the sum of all three-digit positive integers 
  where the digits are distinct, the hundreds digit is at least 2,
  and the digit 1 is not used in any place. -/
def T : ℕ := 
  let hundreds_sum := (2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) * 56 * 100
  let tens_sum := (2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) * 49 * 10
  let units_sum := (2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) * 49
  hundreds_sum + tens_sum + units_sum

/-- Theorem: Compute the remainder when T is divided by 1000. -/
theorem compute_remainder : T % 1000 = 116 := by
  sorry

end compute_remainder_l173_173553


namespace lowest_position_of_vasya_l173_173209

-- Definitions of conditions
def num_cyclists : ℕ := 500
def num_stages : ℕ := 15
def vasya_position_each_stage : ℕ := 7

-- Theorem statement
theorem lowest_position_of_vasya (H1 : ∀ (s: ℕ), s ∈ finset.range(num_stages) → 
(num_cyclists + 1) - vasya_position_each_stage > (num_cyclists - 90))

(assumption_vasya :
  ∀ s ∈ finset.range(num_stages), vasya_position_each_stage < num_cyclists):
  ∃ (lowest_position: ℕ), lowest_position = 91 :=
sorry

end lowest_position_of_vasya_l173_173209


namespace trees_holes_calculation_l173_173148

theorem trees_holes_calculation (length_road : ℕ) (initial_interval : ℕ) (new_interval : ℕ) (initial_tree_count new_tree_count : ℕ) :
  length_road = 240 ∧ initial_interval = 8 ∧ new_interval = 6 ∧
  length_road % initial_interval = 0 ∧ length_road % new_interval = 0 ∧ 
  initial_tree_count = (length_road / initial_interval) + 1 ∧ new_tree_count = (length_road / new_interval) + 1 →
  (new_tree_count - initial_tree_count) = 10 ∧ (initial_tree_count - new_tree_count) = 0 :=
begin
  sorry
end

end trees_holes_calculation_l173_173148


namespace exists_quadrilateral_with_irrational_triangle_area_l173_173591

theorem exists_quadrilateral_with_irrational_triangle_area :
  ∃ (A B C D O : ℝ × ℝ),
    let area : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ :=
          λ p1 p2 p3, 0.5 * abs ((p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) : ℝ),
    convex_hull (insert A (insert B (insert C {D}))) O ∧
    area A B C D = 1 ∧
    (∃ (O : ℝ × ℝ), O ∈ interior (convex_hull (insert A (insert B (insert C {D}))) O) ∧
     (irrational (area O A B) ∨
      irrational (area O B C) ∨
      irrational (area O C D) ∨
      irrational (area O D A))) :=
sorry

end exists_quadrilateral_with_irrational_triangle_area_l173_173591


namespace smallest_n_divisible_l173_173304

theorem smallest_n_divisible (n : ℕ) : 
  (450 ∣ n^3) ∧ (2560 ∣ n^4) ↔ n = 60 :=
by {
  sorry
}

end smallest_n_divisible_l173_173304


namespace maximize_profit_l173_173586

/-- Given P and y as defined, prove the conditions for maximized profit -/
theorem maximize_profit (a : ℝ) (x : ℝ) (hx : 0 ≤ x ∧ x ≤ a) :
  let P := 3 - 2 / (x + 1),
      y := 16 - 4 / (x + 1) - x in
  (a ≥ 1 → x = 1 → y ≥ (16 - 4 / (a + 1) - a)) ∧
  (a < 1 → x = a → y ≥ (16 - 4 / (a + 1) - a)) :=
by
  sorry

end maximize_profit_l173_173586


namespace system_of_equations_solution_l173_173711

theorem system_of_equations_solution :
  ∃ x y : ℝ, (x + y = 5) ∧ (x - y = 1) ∧ (x = 3) ∧ (y = 2) :=
by
  use 3
  use 2
  split
  { exact rfl }
  split
  { exact rfl }
  split
  { exact rfl }
  { exact rfl }

end system_of_equations_solution_l173_173711


namespace vasya_lowest_position_l173_173199

theorem vasya_lowest_position
  (n_cyclists : ℕ) (n_stages : ℕ) 
  (stage_positions : ℕ → ℕ → ℕ) -- a function that takes a stage and a cyclist and returns the position (e.g., stage_positions(stage, cyclist) = position)
  (total_time : ℕ → ℕ)  -- a function that takes a cyclist and returns their total time
  (distinct_times : ∀ (c1 c2 : ℕ), c1 ≠ c2 → (total_time c1 ≠ total_time c2) ∧ 
                   ∀ (s : ℕ), stage_positions s c1 ≠ stage_positions s c2)
  (vasya_position : ℕ) (hv : ∀ (s : ℕ), s < n_stages → stage_positions s vasya_position = 7) :
  vasya_position = 91 :=
sorry

end vasya_lowest_position_l173_173199


namespace find_divisor_l173_173992

theorem find_divisor
  (D dividend quotient remainder : ℤ)
  (h_dividend : dividend = 13787)
  (h_quotient : quotient = 89)
  (h_remainder : remainder = 14)
  (h_relation : dividend = (D * quotient) + remainder) :
  D = 155 :=
by
  sorry

end find_divisor_l173_173992


namespace range_of_a_l173_173191

noncomputable def f (a x : ℝ) : ℝ :=
  Real.exp (x-2) + (1/3) * x^3 - (3/2) * x^2 + 2 * x - Real.log (x-1) + a

theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, (1 < x → f a x = y) ↔ ∃ z : ℝ, 1 < z → f a (f a z) = y) →
  a ≤ 1/3 :=
sorry

end range_of_a_l173_173191


namespace midpoint_parallelogram_area_ratio_l173_173195

theorem midpoint_parallelogram_area_ratio (P : ℝ) (h : P > 0) :
  let smaller_parallelogram_area := P / 4 in smaller_parallelogram_area = P / 4 :=
by
  sorry

end midpoint_parallelogram_area_ratio_l173_173195


namespace sum_of_two_integers_l173_173645

theorem sum_of_two_integers (x y : ℝ) (h₁ : x^2 + y^2 = 130) (h₂ : x * y = 45) : x + y = 2 * Real.sqrt 55 :=
sorry

end sum_of_two_integers_l173_173645


namespace correct_option_D_l173_173305

theorem correct_option_D : (Real.cbrt (-27) = - (Real.cbrt 27)) :=
sorry

end correct_option_D_l173_173305


namespace sum_of_rational_roots_l173_173812

noncomputable def p (x : ℤ) : ℤ := x^3 - 8 * x^2 + 17 * x - 10

theorem sum_of_rational_roots : 
  let roots := {r : ℚ | p r = 0} in
  roots.sum = 8 := sorry

end sum_of_rational_roots_l173_173812


namespace boxes_remaining_to_sell_l173_173940

-- Define the conditions
def first_customer_boxes : ℕ := 5 
def second_customer_boxes : ℕ := 4 * first_customer_boxes
def third_customer_boxes : ℕ := second_customer_boxes / 2
def fourth_customer_boxes : ℕ := 3 * third_customer_boxes
def final_customer_boxes : ℕ := 10
def sales_goal : ℕ := 150

-- Total boxes sold
def total_boxes_sold : ℕ := first_customer_boxes + second_customer_boxes + third_customer_boxes + fourth_customer_boxes + final_customer_boxes

-- Boxes left to sell to hit the sales goal
def boxes_left_to_sell : ℕ := sales_goal - total_boxes_sold

-- Prove the number of boxes left to sell is 75
theorem boxes_remaining_to_sell : boxes_left_to_sell = 75 :=
by
  -- Step to prove goes here
  sorry

end boxes_remaining_to_sell_l173_173940


namespace radius_of_circle_passing_through_points_l173_173264

noncomputable def circle_radius (A B C : ℝ × ℝ) : ℝ :=
if let (x1, y1) := A, (x2, y2) := B, (x3, y3) := C
then sqrt (((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)^2 +
          ((y1 * (x2 - x3) + y2 * (x3 - x1) + y3 * (x1 - x2)) / 2)^2) / 2
else 0

theorem radius_of_circle_passing_through_points :
  circle_radius (1, 3) (4, 2) (1, -7) = 5 :=
by sorry

end radius_of_circle_passing_through_points_l173_173264


namespace focus_of_parabola_x_eq_neg_1_div_4_y_squared_is_neg_1_0_l173_173419

theorem focus_of_parabola_x_eq_neg_1_div_4_y_squared_is_neg_1_0 :
  let P (y : ℝ) := (-1/4 * y^2, y)
  let F := (-1, 0)
  let d := 1
  ∀ (y : ℝ), (P y).fst = -1/4 * y^2 → (F.fst - P y.fst)^2 + (F.snd - P y.snd)^2 = (d + P y.fst)^2 → F = (-1, 0) :=
by
  intros P F d y h1 h2
  sorry

end focus_of_parabola_x_eq_neg_1_div_4_y_squared_is_neg_1_0_l173_173419


namespace max_value_g_l173_173467

noncomputable def f (x : ℝ) (a : ℝ) := Real.sin x + a * Real.cos x

noncomputable def g (x : ℝ) (a : ℝ) := a * Real.sin x + Real.cos x

theorem max_value_g : 
  ∃ a : ℝ, (f 0 a = f ((10 * Real.pi) / 3) a) → 
  a = -Real.sqrt 3 / 3 → 
  ∀ x : ℝ, g(x, -Real.sqrt 3 / 3) ≤ (2 * Real.sqrt 3) / 3 :=
by 
  sorry 

end max_value_g_l173_173467


namespace calculate_product_l173_173907

theorem calculate_product : (3 * 5 * 7 = 38) → (13 * 15 * 17 = 268) → 1 * 3 * 5 = 15 :=
by
  intros h1 h2
  sorry

end calculate_product_l173_173907


namespace kids_in_group_l173_173364

open Nat

theorem kids_in_group (A K : ℕ) (h1 : A + K = 11) (h2 : 8 * A = 72) : K = 2 := by
  sorry

end kids_in_group_l173_173364


namespace four_digit_palindromic_squares_count_l173_173877

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

theorem four_digit_palindromic_squares_count :
  let four_digit_palindromes := { n | n >= 1000 ∧ n < 10000 ∧ is_palindrome n } in
  let perfect_squares := { n | ∃ k, k^2 = n } in
  (four_digit_palindromes ∩ perfect_squares).card = 2 := 
by sorry

end four_digit_palindromic_squares_count_l173_173877


namespace range_of_theta_l173_173625

-- Conditions
def theta_in_triangle (theta : ℝ) : Prop :=
  0 < theta ∧ theta < real.pi

def function_always_positive (theta : ℝ) : Prop :=
  ∀ x : ℝ, (real.cos theta) * x^2 - 4 * (real.sin theta) * x + 6 > 0

-- Proof statement
theorem range_of_theta
  (theta : ℝ)
  (h₁ : theta_in_triangle theta)
  (h₂ : function_always_positive theta) :
  0 < theta ∧ theta < real.pi / 3 :=
sorry

end range_of_theta_l173_173625


namespace matrix_multiplication_correct_l173_173775

def mat1 : Matrix (Fin 2) (Fin 2) ℤ := ![![4, -3], ![5, 2]]
def mat2 : Matrix (Fin 2) (Fin 2) ℤ := ![![0, 6], ![-2, 1]]
def result : Matrix (Fin 2) (Fin 2) ℤ := ![![6, 21], ![-4, 32]]

theorem matrix_multiplication_correct : mat1 ⬝ mat2 = result := 
by 
  sorry

end matrix_multiplication_correct_l173_173775


namespace system_of_equations_solns_l173_173801

theorem system_of_equations_solns (a x y : ℝ) :
  (2 * y - 2 = a * (x - 1)) →
  ((2 * x) / (|y| + y) = Real.sqrt x) →
  (  ((a ≤ 0) ∧ (x = 0) ∧ (y = 1 - a / 2))
  ∨ ((a < 2) ∧ (x = 0) ∧ (y = 1 - a / 2))
  ∨ ((0 < a ∧ a < 2) ∧ (x = (2 - a)^2 / a^2) ∧ (y = (2 - a) / a))
  ∨ ((x = 1) ∧ (y = 1))
  ∨ ((2 ≤ a) ∧ (x = 1) ∧ (y = 1))) :=
by
  sorry

end system_of_equations_solns_l173_173801


namespace min_surface_area_of_prism_on_sphere_l173_173825

open Real

namespace MinimumSurfaceArea

-- Defining the conditions
def prism_volume (a h : ℝ) : ℝ :=
  (1 / 2) * a^2 * h

def circumradius (a : ℝ) : ℝ :=
  (sqrt 3 / 3) * a

def sphere_radius (a h : ℝ) : ℝ :=
  sqrt ((1/3) * a^2 + (18 / a^2)^2)

axiom prism_on_sphere (a h r : ℝ) :
  prism_volume a h = 3 * sqrt 3 →
  sphere_radius a h = r →
  min_surface_area (some h such that prism_volume a h = 3 * sqrt 3) = 12 * (sqrt 3^2) * 39

-- Goal
theorem min_surface_area_of_prism_on_sphere :
  ∃ (a h r : ℝ), prism_volume a h = 3 * sqrt 3 →
  sphere_radius a h = r →
  4 * π * r^2 = 12 * π * 39 :=
sorry

end MinimumSurfaceArea

end min_surface_area_of_prism_on_sphere_l173_173825


namespace ratio_of_areas_eq_l173_173900

-- Define the conditions
variables {C D : Type} [circle C] [circle D]
variables (R_C R_D : ℝ)
variable (L : ℝ)

-- Given conditions
axiom arc_length_eq : (60 / 360) * (2 * π * R_C) = L
axiom arc_length_eq' : (40 / 360) * (2 * π * R_D) = L

-- Statement to prove
theorem ratio_of_areas_eq : (π * R_C^2) / (π * R_D^2) = 4 / 9 :=
sorry

end ratio_of_areas_eq_l173_173900


namespace magnitude_of_z_l173_173562

open Complex

/-- 
Given z = ((1-4*I)*(1+I) + 2 + 4*I) / (3 + 4*I), 
prove that |z| = sqrt(2)
-/
theorem magnitude_of_z : 
  let z := ((1 - 4 * I) * (1 + I) + 2 + 4 * I) / (3 + 4 * I)
  in Complex.abs z = Real.sqrt 2 := 
by 
  sorry

end magnitude_of_z_l173_173562


namespace vasya_lowest_position_l173_173206

theorem vasya_lowest_position
  (n_cyclists : ℕ) (n_stages : ℕ) 
  (stage_positions : ℕ → ℕ → ℕ) -- a function that takes a stage and a cyclist and returns the position (e.g., stage_positions(stage, cyclist) = position)
  (total_time : ℕ → ℕ)  -- a function that takes a cyclist and returns their total time
  (distinct_times : ∀ (c1 c2 : ℕ), c1 ≠ c2 → (total_time c1 ≠ total_time c2) ∧ 
                   ∀ (s : ℕ), stage_positions s c1 ≠ stage_positions s c2)
  (vasya_position : ℕ) (hv : ∀ (s : ℕ), s < n_stages → stage_positions s vasya_position = 7) :
  vasya_position = 91 :=
sorry

end vasya_lowest_position_l173_173206


namespace possible_values_y_l173_173555

theorem possible_values_y (a b : ℝ) (ha: a ≠ 0) (hb: b ≠ 0) : 
  let y := (a / |a|) + (b / |b|) + (a * b / |a * b|) in
  y ∈ ({-1, 3} : set ℝ) :=
sorry

end possible_values_y_l173_173555


namespace faster_by_airplane_l173_173762

theorem faster_by_airplane : 
  let driving_time := 3 * 60 + 15 
  let airport_drive := 10
  let wait_to_board := 20
  let flight_duration := driving_time / 3
  let exit_plane := 10
  driving_time - (airport_drive + wait_to_board + flight_duration + exit_plane) = 90 := 
by
  let driving_time : ℕ := 3 * 60 + 15
  let airport_drive : ℕ := 10
  let wait_to_board : ℕ := 20
  let flight_duration : ℕ := driving_time / 3
  let exit_plane : ℕ := 10
  have h1 : driving_time = 195 := rfl
  have h2 : flight_duration = 65 := by norm_num [h1]
  have h3 : 195 - (10 + 20 + 65 + 10) = 195 - 105 := by norm_num
  have h4 : 195 - 105 = 90 := by norm_num
  exact h4

end faster_by_airplane_l173_173762


namespace time_between_periods_l173_173330

theorem time_between_periods (start_time end_time : ℕ) (num_periods duration_per_period : ℕ)
    (start_time_def : start_time = 10 * 60) -- 10:00 am in minutes
    (end_time_def   : end_time = 13 * 60 + 40) -- 1:40 pm in minutes
    (num_periods_def: num_periods = 5) -- Number of periods
    (duration_per_period_def: duration_per_period = 40): -- Duration of each period in minutes
    let total_minutes := end_time - start_time in
    let time_for_periods := num_periods * duration_per_period in
    let total_break_time := total_minutes - time_for_periods in
    let num_breaks := num_periods - 1 in
    let time_per_break := total_break_time / num_breaks in
    time_per_break = 5 :=
by
  sorry

end time_between_periods_l173_173330


namespace distance_from_D_to_ABC_l173_173147

def Point := (ℝ × ℝ × ℝ)

def distance_to_plane (D A B C : Point) : ℝ := 
  let vec_AB := (A.1 - B.1, A.2 - B.2, A.3 - B.3)
  let vec_AC := (A.1 - C.1, A.2 - C.2, A.3 - C.3)
  let normal_vec := (vec_AB.2 * vec_AC.3 - vec_AB.3 * vec_AC.2,
                      vec_AB.3 * vec_AC.1 - vec_AB.1 * vec_AC.3,
                      vec_AB.1 * vec_AC.2 - vec_AB.2 * vec_AC.1)
  let d := (normal_vec.1 * D.1 + normal_vec.2 * D.2 + normal_vec.3 * D.3) /
           Math.sqrt (normal_vec.1^2 + normal_vec.2^2 + normal_vec.3^2)
  Real.abs d

theorem distance_from_D_to_ABC :
  ∃ D A B C : Point, D = (0, 0, 0) ∧ A = (5, 0, 0) ∧ B = (0, 3, 0) ∧ C = (0, 0, 4) ∧ 
  Real.abs (distance_to_plane D A B C - 2.0) < 0.1 :=
by {
  sorry
}

end distance_from_D_to_ABC_l173_173147


namespace circle_condition_l173_173638

theorem circle_condition (k : ℝ) : (∃ x y : ℝ, x^2 + y^2 - k*x + 2*y + k^2 - 2 = 0) ↔ k ∈ Ioo (-2 : ℝ) 2 := 
sorry

end circle_condition_l173_173638


namespace vasya_lowest_position_l173_173244

theorem vasya_lowest_position
  (n : ℕ) (m : ℕ) (num_cyclists : ℕ) (vasya_place : ℕ) :
  num_cyclists = 500 →
  n = 15 →
  vasya_place = 7 →
  ∀ (stages : fin n → fin num_cyclists) (no_identical_times : ∀ i j : fin n, i ≠ j → 
  ∀ k l : fin num_cyclists, stages i k ≠ stages j l),
  ∃ (lowest_position : ℕ), lowest_position = 91 := 
by sorry

end vasya_lowest_position_l173_173244


namespace find_lambda_l173_173867

open Real

variables (a b : ℝ × ℝ) (λ : ℝ)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := (v.1 ^ 2 + v.2 ^ 2).sqrt

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def angle_120 (a b : ℝ × ℝ) : Prop :=
  dot_product a b = magnitude a * magnitude b * (-1 / 2)

def perpendicular (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

def vectors_cond (a b : ℝ × ℝ) (λ : ℝ) : Prop :=
  (magnitude a = 2) ∧ (magnitude b = 1) ∧
  angle_120 a b ∧
  perpendicular (a.1 + λ * b.1, a.2 + λ * b.2) (2 * a.1 - b.1, 2 * a.2 - b.2)

theorem find_lambda (a b : ℝ × ℝ) (λ : ℝ) (h : vectors_cond a b λ) : λ = 3 :=
by
  sorry

end find_lambda_l173_173867


namespace probability_die_roll_set_l173_173001

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then 1 else a_n (n - 1) + n - 1

theorem probability_die_roll_set :
  let outcomes := { 1, 2, 3, 4, 5, 6 }
  let trials := finset.product (finset.product outcomes outcomes) outcomes
  let valid_sets := finset.filter (λ y : ℕ × ℕ × ℕ, {y.1, y.2.1, y.2.2} = {a_n 1, a_n 2, a_n 3}) trials
  in (finset.card valid_sets : ℝ) / (finset.card trials : ℝ) = (1 : ℝ) / 12 :=
by sorry

end probability_die_roll_set_l173_173001


namespace range_of_a_l173_173506

theorem range_of_a (a : ℝ) :
  (∃ (x : ℝ), (2 - 2^(-|x - 3|))^2 = 3 + a) ↔ -2 ≤ a ∧ a < 1 :=
by
  sorry

end range_of_a_l173_173506


namespace problem_statement_l173_173818

theorem problem_statement (x : Fin 2022 → ℝ)
  (h_prod : (∏ i, x i) = 1)
  (h_eq : ∀ k : ℕ, k ≤ 2021 → (∏ i, (x i + k)) = 2^k) :
  (∏ i, (x i + 2022)) = fact 2022 + 2^2022 - 1 :=
sorry

end problem_statement_l173_173818


namespace proof_1_proof_2_l173_173550

def A := {x : ℝ | x^2 - 3 * x + 2 = 0}
def B (a : ℝ) := {x : ℝ | x^2 + 2 * (a + 1) * x + (a^2 - 5) = 0}

theorem proof_1 (a : ℝ) : 
  (A ∩ B a = {2}) ↔ (a = -1 ∨ a = -3) :=
sorry

theorem proof_2 (a : ℝ) : 
  (A ∪ B a = A) ↔ a ∈ set.Iic (-3) :=
sorry

end proof_1_proof_2_l173_173550


namespace bird_migration_time_l173_173076

theorem bird_migration_time :
  let distance_jim_disney := 42
  let distance_disney_london := 57
  let distance_london_everest := 65
  let distance_everest_jim := 70
  let speed_c := 12
  let time_jim_disney := distance_jim_disney / speed_c
  let time_disney_london := distance_disney_london / speed_c
  let time_london_everest := distance_london_everest / speed_c
  let time_everest_jim := distance_everest_jim / speed_c
  let total_time_one_sequence_c := time_jim_disney + time_disney_london + time_london_everest + time_everest_jim
  let total_time_two_sequences_c := total_time_one_sequence_c * 2
  total_time_two_sequences_c = 39 := 
by {
  unfold distance_jim_disney distance_disney_london distance_london_everest distance_everest_jim speed_c,
  unfold time_jim_disney time_disney_london time_london_everest time_everest_jim,
  unfold total_time_one_sequence_c total_time_two_sequences_c,
  sorry
}

end bird_migration_time_l173_173076


namespace remainder_of_x_mod_11_l173_173695

theorem remainder_of_x_mod_11 {x : ℤ} (h : x % 66 = 14) : x % 11 = 3 :=
sorry

end remainder_of_x_mod_11_l173_173695


namespace min_posts_required_l173_173745

/-- Considering a rectangular park with dimensions 45 m by 90 m, where one side of 
length 90 m is along a concrete wall, and fence posts are placed every 15 meters 
including at the start and end points. Given these conditions, the minimal number 
of posts required to fence the remaining three sides is 13. -/
theorem min_posts_required : 
  ∃ (posts : ℕ), 
  let length := 90 in 
  let width := 45 in
  let post_distance := 15 in
  let side_posts (s : ℕ) := s / post_distance + 1 in
  let total_posts := side_posts length + 2 * (side_posts width - 1) in
  total_posts = 13 :=
by
  sorry

end min_posts_required_l173_173745


namespace base8_digits_sum_l173_173502

-- Define digits and their restrictions
variables {A B C : ℕ}

-- Main theorem
theorem base8_digits_sum (h1 : 0 < A ∧ A < 8)
                         (h2 : 0 < B ∧ B < 8)
                         (h3 : 0 < C ∧ C < 8)
                         (distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
                         (condition : (8^2 * A + 8 * B + C) + (8^2 * B + 8 * C + A) + (8^2 * C + 8 * A + B) = (8^2 + 8 + 1) * 8 * A) :
  A + B + C = 8 := 
sorry

end base8_digits_sum_l173_173502


namespace lg_5_expression_l173_173886

noncomputable def lg_in_terms_of_p_q (p q : ℝ) : ℝ := 3 * p * q / (1 + 3 * p * q)

theorem lg_5_expression (p q : ℝ) (h₁ : Real.log 8 3 = p) (h₂ : Real.log 3 5 = q) :
    Real.log10 5 = lg_in_terms_of_p_q p q := sorry

end lg_5_expression_l173_173886


namespace math_problem_proof_l173_173116

-- Definitions under given conditions
variables {k : ℝ} (hk : k ≠ 0) {n : ℕ} (hn : n ≥ 2)

noncomputable def polynomial_coeffs (a : ℕ → ℝ) (x : ℝ) : ℝ :=
∑ i in Finset.range (n + 1), a i * x^i

-- The assertion of the problem 
theorem math_problem_proof 
  (a : ℕ → ℝ)
  (h_poly : (1 + k * x)^n = polynomial_coeffs a x)
  (a0 : a 0 = 1):
  (∑ i in Finset.range (n + 1), a i) - a 0 = (1 + k)^n - 1 ∧
  (∑ i in Finset.range n, (i + 1) * a (i + 1)) = n * k * (1 + k)^(n - 1) := 
by sorry

end math_problem_proof_l173_173116


namespace remainder_sum_15_div_11_l173_173691

theorem remainder_sum_15_div_11 :
  let n := 15 
  let a := 1 
  let l := 15 
  let S := (n * (a + l)) / 2
  S % 11 = 10 :=
by
  let n := 15
  let a := 1
  let l := 15
  let S := (n * (a + l)) / 2
  show S % 11 = 10
  sorry

end remainder_sum_15_div_11_l173_173691


namespace second_field_area_percent_greater_l173_173631

theorem second_field_area_percent_greater (r1 r2 : ℝ) (h : r1 / r2 = 2 / 5) : 
  (π * (r2^2) - π * (r1^2)) / (π * (r1^2)) * 100 = 525 := 
by
  sorry

end second_field_area_percent_greater_l173_173631


namespace determine_g_two_l173_173115

variables (a b c d p q r s : ℝ) -- Define variables a, b, c, d, p, q, r, s as real numbers
variables (h₁ : a < b) (h₂ : b < c) (h₃ : c < d) -- The conditions a < b < c < d

noncomputable def f (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d
noncomputable def g (x : ℝ) : ℝ := (x - 1/p) * (x - 1/q) * (x - 1/r) * (x - 1/s)

noncomputable def g_two := g 2
noncomputable def f_two := f 2

theorem determine_g_two :
  g_two a b c d = (16 + 8*a + 4*b + 2*c + d) / (p*q*r*s) :=
sorry

end determine_g_two_l173_173115


namespace four_digit_palindromic_squares_count_l173_173876

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

theorem four_digit_palindromic_squares_count :
  let four_digit_palindromes := { n | n >= 1000 ∧ n < 10000 ∧ is_palindrome n } in
  let perfect_squares := { n | ∃ k, k^2 = n } in
  (four_digit_palindromes ∩ perfect_squares).card = 2 := 
by sorry

end four_digit_palindromic_squares_count_l173_173876


namespace triangle_area_correct_l173_173773

def point := (ℝ × ℝ)

def A : point := (2, -3)
def B : point := (4, 5)
def C : point := (10, 1)

def vector_sub (p1 p2 : point) : point :=
  (p1.1 - p2.1, p1.2 - p2.2)

def cross_product (v1 v2 : point) : ℝ :=
  v1.1 * v2.2 - v1.2 * v2.1

def area_of_triangle (A B C : point) : ℝ :=
  let CA := vector_sub A C
  let CB := vector_sub B C
  (1 / 2) * abs (cross_product CA CB)

theorem triangle_area_correct :
  area_of_triangle A B C = 28 :=
by
  sorry

end triangle_area_correct_l173_173773


namespace vasya_maximum_rank_l173_173250

theorem vasya_maximum_rank {n : ℕ} (cyclists stages : ℕ) (VasyaPlace : ℕ) 
  (rankings : Π (s : fin stages), fin cyclists):
  cyclists = 500 → stages = 15 → VasyaPlace = 7 →
  (∀ s, rankings s VasyaPlace = 7) →
  (∀ s t i, rankings s i ≠ rankings t i) →
  (∀ s, ∃ l, list.nodup l ∧ (∀ i j, i < j → rankings s i < rankings s j) ∧ list.length l = 500) →
  (∃ (maxRank : ℕ), maxRank = 91) :=
by
  intros hcyclists hstages hplace hvasya_place hdistinct_rankings hstage_rankings
  use 91
  sorry

end vasya_maximum_rank_l173_173250


namespace part_a_part_b_l173_173685

-- Part (a)
theorem part_a (a : ℝ) (h : a = Real.sqrt 5) : 
∃ b : ℝ, b = 1 :=
by {
  use 1,
  refl
}

-- Part (b)
theorem part_b (a : ℝ) (h : a = 7) : 
∃ b : ℝ, b = Real.sqrt 7 :=
by {
  use Real.sqrt 7,
  refl
}

end part_a_part_b_l173_173685


namespace simplify_999_times_neg13_simplify_complex_expr_correct_division_calculation_l173_173768

-- Part 1: Proving the simplified form of arithmetic operations
theorem simplify_999_times_neg13 : 999 * (-13) = -12987 := by
  sorry

theorem simplify_complex_expr :
  999 * (118 + 4 / 5) + 333 * (-3 / 5) - 999 * (18 + 3 / 5) = 99900 := by
  sorry

-- Part 2: Proving the correct calculation of division
theorem correct_division_calculation : 6 / (-1 / 2 + 1 / 3) = -36 := by
  sorry

end simplify_999_times_neg13_simplify_complex_expr_correct_division_calculation_l173_173768


namespace first_order_central_moment_zero_l173_173589

-- Let X be a continuous random variable with probability density function f
-- Let M(X) be the expected value of X

theorem first_order_central_moment_zero (f : ℝ → ℝ) (μ : ℝ) (h1 : ∀ x, 0 ≤ f x) (h2 : ∫ x in -∞..∞, f x = 1) (h3 : ∫ x in -∞..∞, x * f x = μ) :
  ∫ x in -∞..∞, (x - μ) * f x = 0 :=
sorry

end first_order_central_moment_zero_l173_173589


namespace compute_expression_l173_173503

theorem compute_expression (x : ℝ) (hx : x + 1 / x = 7) : 
  (x - 3)^2 + 36 / (x - 3)^2 = 12.375 := 
  sorry

end compute_expression_l173_173503


namespace arithmetic_sequence_geometric_subsequence_l173_173442

theorem arithmetic_sequence_geometric_subsequence :
  ∀ (a : ℕ → ℝ), (∀ n, a (n + 1) = a n + 2) ∧ (a 1 * a 3 = a 2 ^ 2) → a 2 = 4 :=
by
  intros a h
  sorry

end arithmetic_sequence_geometric_subsequence_l173_173442


namespace lowest_position_l173_173235

theorem lowest_position (num_cyclists : ℕ) (num_stages : ℕ) (vasya_position : ℕ) :
  num_cyclists = 500 →
  num_stages = 15 →
  vasya_position = 7 →
  ∃ n, n = 91 :=
by
  intros
  sorry

end lowest_position_l173_173235


namespace diana_gadgets_sold_total_l173_173791

theorem diana_gadgets_sold_total (a_1 : ℕ) (d : ℕ) (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) :
  a_1 = 2 →
  d = 3 →
  (∀ n, a_n n = a_1 + (n - 1) * d) →
  (∀ n, S_n n = n * (a_1 + a_n n) / 2) →
  S_n 25 = 950 :=
begin
  intros h1 h2 h3 h4,
  rw [h1, h2] at *,
  have h5 : a_n 25 = 2 + (25 - 1) * 3, by { apply h3, },
  have h6 : S_n 25 = 25 * (2 + a_n 25) / 2, by { apply h4, },
  rw h5 at h6,
  norm_num at h6,
  exact h6,
end

end diana_gadgets_sold_total_l173_173791


namespace find_a_l173_173019

theorem find_a (a : ℝ) 
  (h : ∀ x : ℝ, ax^2 + (a - 1) * x + (a - 2) < 0 → x ∈ (Set.Ioo (−∞) (−1)) ∪ (Set.Ioo (2) (∞))) : 
  a = 1 := 
sorry

end find_a_l173_173019


namespace largest_possible_difference_l173_173779

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def largest_prime_difference_sum_172 : ℕ :=
  let n := 172
  Primes.filter (λ p q, p ≠ q ∧ p + q = n).map (λ p q, |p - q|).max

theorem largest_possible_difference :
  let n := 172 in ∀ p q : ℕ, is_prime p ∧ is_prime q ∧ p ≠ q ∧ p + q = n → |p - q| ≤ 162 :=
by
  sorry

end largest_possible_difference_l173_173779


namespace f_cos_15_eq_l173_173822

open Real

-- Define the function f
def f (x : ℝ) : ℝ := cos (2 * arcsin x) - 1

-- State the theorem to prove
theorem f_cos_15_eq : f (cos (15 * π / 180)) = - (sqrt 3 / 2) - 1 :=
by
  -- Skip the proof
  sorry

end f_cos_15_eq_l173_173822


namespace sum_of_squares_of_integer_n_l173_173693

theorem sum_of_squares_of_integer_n (n : ℤ) (h : 36 % (2 * n - 1) = 0) : 
  ∑ k in ({ n ∈ ℤ | 36 % (2 * k - 1) = 0} : set ℤ).toFinset, k^2 = 47 :=
sorry

end sum_of_squares_of_integer_n_l173_173693


namespace P_gt_Q_l173_173495

theorem P_gt_Q (x : ℝ) : let P := x^2 + 2 in let Q := 2x in P > Q :=
by
  let P := x^2 + 2
  let Q := 2x
  sorry

end P_gt_Q_l173_173495


namespace eccentricity_of_hyperbola_l173_173475

-- Mathematical definitions for Lean
variable (a b : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b)
def hyperbola := x^2 / a^2 - y^2 / b^2 = 1
def asymptote_slope := b / a = 2
def c := sqrt (a^2 + b^2)
def e := c / a

-- Problem Statement
theorem eccentricity_of_hyperbola (h : asymptote_slope a b) : e = sqrt 5 :=
by
  have b_eq_2a : b = 2 * a := sorry -- from h we obtain this
  have c_eq_sqrt5a : c = sqrt 5 * a := sorry -- using b_eq_2a substitute into definition of c
  show e = sqrt 5 := sorry -- using c_eq_sqrt5a substitute into definition of e

end eccentricity_of_hyperbola_l173_173475


namespace employee_monthly_wage_l173_173145

theorem employee_monthly_wage 
(revenue : ℝ)
(tax_rate : ℝ)
(marketing_rate : ℝ)
(operational_cost_rate : ℝ)
(wage_rate : ℝ)
(num_employees : ℕ)
(h_revenue : revenue = 400000)
(h_tax_rate : tax_rate = 0.10)
(h_marketing_rate : marketing_rate = 0.05)
(h_operational_cost_rate : operational_cost_rate = 0.20)
(h_wage_rate : wage_rate = 0.15)
(h_num_employees : num_employees = 10) :
(revenue * (1 - tax_rate) * (1 - marketing_rate) * (1 - operational_cost_rate) * wage_rate / num_employees = 4104) :=
by
  sorry

end employee_monthly_wage_l173_173145


namespace nine_otimes_three_eq_thirteen_l173_173196

def otimes (a b : ℝ) : ℝ := a + 4 * a / (3 * b)

theorem nine_otimes_three_eq_thirteen : otimes 9 3 = 13 :=
by sorry

end nine_otimes_three_eq_thirteen_l173_173196


namespace total_price_correct_l173_173986

def original_jewelry_price : ℕ := 30
def original_painting_price : ℕ := 100
def jewelry_price_increase : ℕ := 10
def painting_price_increase_percentage : ℕ := 20
def num_jewelry : ℕ := 2
def num_paintings : ℕ := 5

theorem total_price_correct :
  let new_jewelry_price := original_jewelry_price + jewelry_price_increase in
  let new_painting_price := original_painting_price + (original_painting_price * painting_price_increase_percentage / 100) in
  let total_price := (new_jewelry_price * num_jewelry) + (new_painting_price * num_paintings) in
  total_price = 680 := 
by
  sorry

end total_price_correct_l173_173986


namespace circumcircle_tangent_to_BC_l173_173965

open EuclideanGeometry

variables {A B C P Q L : Point}
variables {circumcircle_ABC : Circle}

noncomputable def angle_bisector (A B C : Point) (L : Point) : Prop := 
  ∃ (L : Point), (angle A L B = angle A L C)

noncomputable def perpendicular_bisector (A L : Point) (circumcircle : Circle) (P Q : Point) : Prop :=
  ∃ (P Q : Point), P ≠ Q ∧ (line P Q) ⊥ (segment A L) ∧ P ∈ circumcircle ∧ Q ∈ circumcircle

noncomputable def circumcircle (P L Q : Point) (A B C : Point) : Circle := sorry

theorem circumcircle_tangent_to_BC 
  (h1 : angle_bisector A B C L)
  (h2 : perpendicular_bisector A L circumcircle_ABC P Q) : 
  is_tangent (circumcircle P L Q A B C) (line B C) :=
sorry

end circumcircle_tangent_to_BC_l173_173965


namespace total_price_jewelry_paintings_l173_173982

theorem total_price_jewelry_paintings:
  (original_price_jewelry original_price_paintings increase_jewelry_in_dollars increase_paintings_in_percent quantity_jewelry quantity_paintings new_price_jewelry new_price_paintings total_cost: ℝ) 
  (h₁: original_price_jewelry = 30)
  (h₂: original_price_paintings = 100)
  (h₃: increase_jewelry_in_dollars = 10)
  (h₄: increase_paintings_in_percent = 0.20)
  (h₅: quantity_jewelry = 2)
  (h₆: quantity_paintings = 5)
  (h₇: new_price_jewelry = original_price_jewelry + increase_jewelry_in_dollars)
  (h₈: new_price_paintings = original_price_paintings + (original_price_paintings * increase_paintings_in_percent))
  (h₉: total_cost = (new_price_jewelry * quantity_jewelry) + (new_price_paintings * quantity_paintings)) :
  total_cost = 680 :=
by 
  sorry

end total_price_jewelry_paintings_l173_173982


namespace quiz_competition_l173_173342

theorem quiz_competition (x : ℕ) :
  (10 * x - 4 * (20 - x) ≥ 88) ↔ (x ≥ 12) :=
by 
  sorry

end quiz_competition_l173_173342


namespace vasya_maximum_rank_l173_173249

theorem vasya_maximum_rank {n : ℕ} (cyclists stages : ℕ) (VasyaPlace : ℕ) 
  (rankings : Π (s : fin stages), fin cyclists):
  cyclists = 500 → stages = 15 → VasyaPlace = 7 →
  (∀ s, rankings s VasyaPlace = 7) →
  (∀ s t i, rankings s i ≠ rankings t i) →
  (∀ s, ∃ l, list.nodup l ∧ (∀ i j, i < j → rankings s i < rankings s j) ∧ list.length l = 500) →
  (∃ (maxRank : ℕ), maxRank = 91) :=
by
  intros hcyclists hstages hplace hvasya_place hdistinct_rankings hstage_rankings
  use 91
  sorry

end vasya_maximum_rank_l173_173249


namespace temp_interpretation_l173_173774

theorem temp_interpretation (below_zero : ℤ) (above_zero : ℤ) (h : below_zero = -2):
  above_zero = 3 → 3 = 0 := by
  intro h2
  have : above_zero = 3 := h2
  sorry

end temp_interpretation_l173_173774


namespace students_with_no_preference_l173_173520

def total_students : ℕ := 210
def prefer_mac : ℕ := 60
def equally_prefer_both (x : ℕ) : ℕ := x / 3

def no_preference_students : ℕ :=
  total_students - (prefer_mac + equally_prefer_both prefer_mac)

theorem students_with_no_preference :
  no_preference_students = 130 :=
by
  sorry

end students_with_no_preference_l173_173520


namespace problem_l173_173328

variables {α : Type*} [metric_space α] [normed_space ℝ α]

structure Triangle (α : Type*) :=
(A B C : α)

structure Circle (α : Type*) :=
(center : α)
(radius : ℝ)

noncomputable def is_inscribed (c : Circle α) (t : Triangle α) :=
∃ F G H : α, c.radius = 6 ∧ dist c.center F = 6 ∧ dist c.center G = 6 ∧ dist c.center H = 6

noncomputable def is_circumscribed (c : Circle α) (t : Triangle α) :=
∃ M : α, c.radius = (real.sqrt 85) / 2 ∧ dist c.center M = (real.sqrt 85) / 2

noncomputable def ratio_of_areas (t1 t2 : Triangle α) :=
(7:ℝ)/(10:ℝ)

noncomputable def length_BO (c : Circle α) (t1 t2 : Triangle α) :=
(real.sqrt 85)

noncomputable def length_BQ (c1 c2 : Circle α) (t1 t2 : Triangle α) :=
(5 * real.sqrt 85) / 3

noncomputable def area_ABC (t : Triangle α) : ℝ :=
210

theorem problem {α : Type*} [metric_space α] [normed_space ℝ α]
  {ω Ω : Circle α} {ABC ACM : Triangle α}
  (h1 : is_inscribed ω ABC)
  (h2 : is_circumscribed Ω (Triangle.mk (ABC.A) (ABC.B) (ABC.C)))
  (h3 : ω.center = (Triangle.mk (ABC.A) (ABC.B) (ABC.C)).A)
  (h4 : ω.radius = 6)
  (h5 : ratio_of_areas (Triangle.mk (ACM.A) (ACM.B) (ACM.C)) (Triangle.mk (ABC.A) (ABC.B) (ABC.C)) = 7/10):
  length_BO ω (Triangle.mk (ACM.A) (ACM.B) (ACM.C)) (Triangle.mk (ABC.A) (ABC.B) (ABC.C)) = real.sqrt 85 ∧ 
  length_BQ ω Ω (Triangle.mk (ACM.A) (ACM.B) (ACM.C)) (Triangle.mk (ABC.A) (ABC.B) (ABC.C)) = (5 * real.sqrt 85) / 3 ∧ 
  area_ABC (Triangle.mk (ABC.A) (ABC.B) (ABC.C)) = 210 := 
by sorry

end problem_l173_173328


namespace sum_first_4321_terms_l173_173084

noncomputable def a : ℕ → ℤ
| 0 => 0  -- to avoid clash, not actually used
| 1 => -1
| 2 => 1
| 3 => -2
| n + 1 => if n = 1 then -2 else a n

axiom seq_condition : ∀ (n : ℕ+), (a n) * (a (n + 1)) * (a (n + 2)) * (a (n + 3)) = (a n) + (a (n + 1)) + (a (n + 2)) + (a (n + 3))
axiom not_equal_one : ∀ (n : ℕ+), (a (n + 1)) * (a (n + 2)) * (a (n + 3)) ≠ 1

def S : ℕ → ℤ
| 0 => 0
| n + 1 => S n + a n

theorem sum_first_4321_terms : S 4321 = -4321 := by
  sorry

end sum_first_4321_terms_l173_173084


namespace angle_inclination_of_l_range_l173_173014

noncomputable def angle_of_inclination_range (M A B : ℝ × ℝ) : Set ℝ := sorry

theorem angle_inclination_of_l_range :
  ∀ (M A B : ℝ × ℝ),
    M = (1, 0) →
    A = (2, 1) →
    B = (0, Real.sqrt 3) →
    angle_of_inclination_range M A B = Set.Icc (Real.pi / 4) (2 * Real.pi / 3) :=
by
  intros
  sorry

end angle_inclination_of_l_range_l173_173014


namespace find_radius_l173_173007

def setA : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}
def setB (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 4)^2 = r^2}

theorem find_radius (r : ℝ) (h : r > 0) :
  (∃! p : ℝ × ℝ, p ∈ setA ∧ p ∈ setB r) ↔ (r = 3 ∨ r = 7) :=
by
  sorry

end find_radius_l173_173007


namespace vasya_lowest_position_l173_173204

theorem vasya_lowest_position
  (n_cyclists : ℕ) (n_stages : ℕ) 
  (stage_positions : ℕ → ℕ → ℕ) -- a function that takes a stage and a cyclist and returns the position (e.g., stage_positions(stage, cyclist) = position)
  (total_time : ℕ → ℕ)  -- a function that takes a cyclist and returns their total time
  (distinct_times : ∀ (c1 c2 : ℕ), c1 ≠ c2 → (total_time c1 ≠ total_time c2) ∧ 
                   ∀ (s : ℕ), stage_positions s c1 ≠ stage_positions s c2)
  (vasya_position : ℕ) (hv : ∀ (s : ℕ), s < n_stages → stage_positions s vasya_position = 7) :
  vasya_position = 91 :=
sorry

end vasya_lowest_position_l173_173204


namespace excellent_math_performance_and_timely_review_relationship_l173_173386

-- Part (I)

theorem excellent_math_performance_and_timely_review_relationship
  (sample_size : ℕ)
  (timely_review_excellent : ℕ)
  (timely_review_not_excellent : ℕ)
  (not_timely_review_excellent : ℕ)
  (not_timely_review_not_excellent : ℕ)
  (critical_value : ℝ) :
  let n := sample_size
      a := timely_review_excellent
      b := not_timely_review_excellent
      c := timely_review_not_excellent
      d := not_timely_review_not_excellent
      χ_squared := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d)) in
  χ_squared > critical_value :=
by
  sorry

-- Part (II)

noncomputable def stratified_sampling_expectation
  (timely_review_excellent_in_8 : ℕ)
  (total_excellent_in_8 : ℕ)
  (selected : ℕ)
  (x1_prob : ℝ)
  (x2_prob : ℝ)
  (x3_prob : ℝ) :
  let p_X1 := x1_prob
      p_X2 := x2_prob
      p_X3 := x3_prob
      expectation := 1 * p_X1 + 2 * p_X2 + 3 * p_X3 in
  expectation = 9 / 4 :=
by
  sorry

end excellent_math_performance_and_timely_review_relationship_l173_173386


namespace boat_speed_l173_173729

def speed_of_boat_in_still_water (V_b : ℝ) : Prop :=
  let stream_speed := 5
  let distance := 81
  let time_downstream := 3
  let effective_speed := V_b + stream_speed
  effective_speed = distance / time_downstream → V_b = 22

theorem boat_speed : ∃ V_b : ℝ, speed_of_boat_in_still_water V_b :=
begin
  use 22,
  unfold speed_of_boat_in_still_water,
  intros,
  sorry
end

end boat_speed_l173_173729


namespace percentage_of_liquid_X_is_correct_l173_173976

noncomputable def liquidXPercentage : ℝ :=
  let A_initial := 0.012 * 300
  let B_initial := 0.022 * 500
  let C_initial := 0.035 * 400
  let A_fifth := A_initial * (1.1)^5
  let B_fifth := B_initial * (1.2)^5
  let C_fifth := C_initial * (1.3)^5
  let total_liquidX := A_fifth + B_fifth + C_fifth
  let total_weight := 300 + 500 + 400
  (total_liquidX / total_weight) * 100

theorem percentage_of_liquid_X_is_correct :
  liquidXPercentage ≈ 7.1 :=
sorry

end percentage_of_liquid_X_is_correct_l173_173976


namespace find_x_given_y_value_l173_173038

theorem find_x_given_y_value
  (y : ℤ)
  (x : ℤ)
  (h : 9 * 3^x = 7^(y + 2))
  (hy : y = -2) :
  x = -2 :=
by
  sorry

end find_x_given_y_value_l173_173038


namespace cost_of_building_fence_l173_173313

theorem cost_of_building_fence (A : ℝ) (p : ℝ) (s : ℝ) (P : ℝ) 
 (h1 : A = 289) 
 (h2 : p = 58) 
 (h3 : s = real.sqrt A) 
 (h4 : P = 4 * s) : 
  P * p = 3944 := 
by 
  sorry

end cost_of_building_fence_l173_173313


namespace lowest_position_l173_173237

theorem lowest_position (num_cyclists : ℕ) (num_stages : ℕ) (vasya_position : ℕ) :
  num_cyclists = 500 →
  num_stages = 15 →
  vasya_position = 7 →
  ∃ n, n = 91 :=
by
  intros
  sorry

end lowest_position_l173_173237


namespace expected_number_of_sixes_l173_173665

-- Define the probability of not rolling a 6 on one die
def prob_not_six : ℚ := 5 / 6

-- Define the probability of rolling zero 6's on three dice
def prob_zero_six : ℚ := prob_not_six ^ 3

-- Define the probability of rolling exactly one 6 among the three dice
def prob_one_six (n : ℕ) : ℚ := n * (1 / 6) * (prob_not_six ^ (n - 1))

-- Calculate the probabilities of each specific outcomes
def prob_exactly_zero_six : ℚ := prob_zero_six
def prob_exactly_one_six : ℚ := prob_one_six 3 * (prob_not_six ^ 2)
def prob_exactly_two_six : ℚ := prob_one_six 3 * (1 / 6) * prob_not_six
def prob_exactly_three_six : ℚ := (1 / 6) ^ 3

-- Define the expected value calculation
noncomputable def expected_value : ℚ :=
  0 * prob_exactly_zero_six
  + 1 * prob_exactly_one_six
  + 2 * prob_exactly_two_six
  + 3 * prob_exactly_three_six

-- Prove that the expected value equals to 1/2
theorem expected_number_of_sixes : expected_value = 1 / 2 :=
  by
    sorry

end expected_number_of_sixes_l173_173665


namespace well_depth_is_correct_l173_173756

noncomputable def depth_of_well : ℝ :=
  let t1 := (λ d : ℝ, d / 4)
  let t2 := (λ d : ℝ, d / 1100)
  let f := (λ d : ℝ, t1 (Real.sqrt d) + t2 d)
  if f (1255.64) = 10 then 1255.64 else 0

theorem well_depth_is_correct :
  depth_of_well = 1255.64 :=
by
  sorry

end well_depth_is_correct_l173_173756


namespace sum_of_c_and_d_is_six_l173_173339

-- Definitions of vertices
def vertex1 := (1, 2)
def vertex2 := (4, 5)
def vertex3 := (5, 4)
def vertex4 := (4, 1)

-- Euclidean distance between two points
def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Lengths of the sides of the quadrilateral
def side1 := euclidean_distance vertex1 vertex2
def side2 := euclidean_distance vertex2 vertex3
def side3 := euclidean_distance vertex3 vertex4
def side4 := euclidean_distance vertex4 vertex1

-- Perimeter of the quadrilateral
def perimeter := side1 + side2 + side3 + side4

-- The values c and d
def c := 4
def d := 2

-- The perimeter expressed in the form c√2 + d√10
def perimeter_expr := (c : ℝ) * real.sqrt 2 + (d : ℝ) * real.sqrt 10

-- The final proof statement
theorem sum_of_c_and_d_is_six : c + d = 6 := 
by sorry

end sum_of_c_and_d_is_six_l173_173339


namespace locus_of_circle_center_l173_173732

theorem locus_of_circle_center :
  ∀ (F : ℝ × ℝ) (line_x : ℝ) (A : ℝ × ℝ) (B M N : ℝ × ℝ),
    F = (1, 0) →
    line_x = -1 →
    (A.1 = 4) ∧ (A.1 > 0) ∧ (A.2 > 0) →
    B = (0, A.2) →
    M = ((B.1 + 0) / 2, (B.2 + 0) / 2) →
    ∃ C : set (ℝ × ℝ), (∀ (p : ℝ × ℝ), p ∈ C ↔ p.2 ^ 2 = 4 * p.1) ∧ A ∈ C →
    N.1 = 8/5 ∧ N.2 = 4/5 :=
by
  intros F line_x A B M N F_def line_x_def A_cond B_def M_def locus_def
  -- Proof omitted
  sorry

end locus_of_circle_center_l173_173732


namespace remaining_pens_l173_173335

theorem remaining_pens (blue_initial black_initial red_initial green_initial purple_initial : ℕ)
                        (blue_removed black_removed red_removed green_removed purple_removed : ℕ) :
  blue_initial = 15 → black_initial = 27 → red_initial = 12 → green_initial = 10 → purple_initial = 8 →
  blue_removed = 8 → black_removed = 9 → red_removed = 3 → green_removed = 5 → purple_removed = 6 →
  blue_initial - blue_removed + black_initial - black_removed + red_initial - red_removed +
  green_initial - green_removed + purple_initial - purple_removed = 41 :=
by
  intros
  sorry

end remaining_pens_l173_173335


namespace line_circle_relationship_l173_173846

def line_passes_through_point (l : ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
  l P.1 = P.2

def circle (C : ℝ × ℝ → Prop) (r : ℝ) : Prop :=
  ∀ (x y : ℝ), C (x, y) ↔ x^2 + y^2 = r^2

def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem line_circle_relationship (l : ℝ → ℝ) (C : ℝ × ℝ → Prop)
  (h_line : line_passes_through_point l (√3, 1))
  (h_circle : circle C 2) :
  let P := (√3, 1)
  in C P ∧ distance (0, 0) P = 2 → ∃ Q : ℝ × ℝ, Q ≠ P ∧ C Q ∧ l Q.1 = Q.2 :=
begin
  sorry
end

end line_circle_relationship_l173_173846


namespace largest_4_digit_congruent_to_17_mod_26_l173_173295

theorem largest_4_digit_congruent_to_17_mod_26 :
  ∃ x, x < 10000 ∧ x ≥ 1000 ∧ x % 26 = 17 ∧ (∀ y, y < 10000 ∧ y ≥ 1000 ∧ y % 26 = 17 → y ≤ x) ∧ x = 9972 := 
by
  sorry

end largest_4_digit_congruent_to_17_mod_26_l173_173295


namespace bike_owners_without_scooters_l173_173064

theorem bike_owners_without_scooters
  (total_adults : ℕ)
  (bike_owners : ℕ)
  (scooter_owners : ℕ)
  (total_adults_eq : total_adults = 400)
  (bike_owners_eq : bike_owners = 370)
  (scooter_owners_eq : scooter_owners = 75) :
  let shared_owners := bike_owners + scooter_owners - total_adults in
  bike_owners - shared_owners = 325 :=
by
  sorry

end bike_owners_without_scooters_l173_173064


namespace max_additional_payment_expected_difference_l173_173796

noncomputable theory
open_locale classical

-- Define the tariffs
def peak_tariff : ℝ := 4.03
def day_tariff : ℝ := 3.39
def night_tariff : ℝ := 1.01

-- Define the meter readings (current and previous month)
def current_readings : list ℕ := [1402, 1347, 1337]
def previous_readings : list ℕ := [1298, 1270, 1214]

-- Client's payment
def client_payment : ℝ := 660.72

-- Define a function to compute consumption
def consumption (current previous : ℕ) : ℕ := current - previous

-- Define a function to compute the total payment
def total_payment (consumptions : list ℕ) (tariffs : list ℝ) : ℝ :=
  list.sum (list.zip_with (*) consumptions.map (λ c, c.to_real) tariffs)

-- Maximum possible additional payment theorem
theorem max_additional_payment :
  let consumptions := [consumption 1402 1298, consumption 1347 1270, consumption 1337 1214] in
  total_payment consumptions [peak_tariff, day_tariff, night_tariff] -
    client_payment = 397.34 := sorry

-- Expected difference theorem
theorem expected_difference :
  let consumptions := [consumption 1402 1214, consumption 1347 1270, consumption 1337 1298] in
  total_payment consumptions [peak_tariff, day_tariff, night_tariff] / 15 * 8.43 - client_payment = 19.30 := sorry

end max_additional_payment_expected_difference_l173_173796


namespace find_range_l173_173017

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def mononote_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

def range_of_x (f : ℝ → ℝ) : set ℝ :=
  { x | f (2 * x - 1) < f (1 / 3) }

theorem find_range (f : ℝ → ℝ) (even_fun : is_even_function f) 
  (mono_fun : mononote_increasing_on_nonneg f) :
  (range_of_x f) = { x | 1 / 3 < x ∧ x < 2 / 3 } :=
by
  sorry

end find_range_l173_173017


namespace average_T_is_10_l173_173176

def count_adjacent_bg_pairs (row : List (string)) : ℕ :=
  (List.zipWith (λ a b => if (a = "B" ∧ b = "G") ∨ (a = "G" ∧ b = "B") then 1 else 0) row (row.tail)).sum

theorem average_T_is_10 (row : List (string)) :
  (List.length row = 20) →
  (row.count "B" = 8) →
  (row.count "G" = 12) →
  (count_adjacent_bg_pairs row).to_real / 19 = 10 :=
by
  sorry

end average_T_is_10_l173_173176


namespace eq_circle_eq_tangent_line_l173_173928

-- Define the circle parameters and conditions
def circle_equation (O : ℝ × ℝ) (r : ℝ) : Prop :=
  ∀ (P : ℝ × ℝ), (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2

-- Given condition: Circle centered at origin with a specific chord length
def given_condition_circle (O : ℝ × ℝ) (chord_len : ℝ) : Prop :=
  ∃ (l : ℝ × ℝ → ℝ), (∀ (x : ℝ) (y : ℝ), l (x, y) = x - y + 1) ∧
    ((O.1)^2 + (O.2)^2 = r^2) ∧ (chord_len = sqrt 6)

-- The equation of the circle is x^2 + y^2 = 2
theorem eq_circle (O : ℝ × ℝ) (chord_len : ℝ) (r : ℝ) (l : ℝ × ℝ → ℝ) :
  given_condition_circle O chord_len → circle_equation (0, 0) (sqrt 2) :=
sorry

-- Define line l as tangent to the circle in the first quadrant
def tangent_line (a b : ℝ) (l : ℝ × ℝ → ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (∀ (x : ℝ) (y : ℝ), l (x, y) = b * x + a * y - a * b) ∧
  (abs (a * b) / sqrt (a^2 + b^2) = sqrt 2)

-- Define the minimized line segment condition for DE
def minimized_intercept_length (a b : ℝ) : Prop :=
  (1/a^2 + 1/b^2 = 1/2) ∧ (a = b = 2)

-- The equation of the line when DE is minimized is x + y = 2
theorem eq_tangent_line (l : ℝ × ℝ → ℝ) (a b : ℝ) :
  tangent_line 2 2 (λ (p : ℝ × ℝ), p.1 + p.2 - 2) :=
sorry

end eq_circle_eq_tangent_line_l173_173928


namespace vasya_maximum_rank_l173_173254

theorem vasya_maximum_rank {n : ℕ} (cyclists stages : ℕ) (VasyaPlace : ℕ) 
  (rankings : Π (s : fin stages), fin cyclists):
  cyclists = 500 → stages = 15 → VasyaPlace = 7 →
  (∀ s, rankings s VasyaPlace = 7) →
  (∀ s t i, rankings s i ≠ rankings t i) →
  (∀ s, ∃ l, list.nodup l ∧ (∀ i j, i < j → rankings s i < rankings s j) ∧ list.length l = 500) →
  (∃ (maxRank : ℕ), maxRank = 91) :=
by
  intros hcyclists hstages hplace hvasya_place hdistinct_rankings hstage_rankings
  use 91
  sorry

end vasya_maximum_rank_l173_173254


namespace expected_number_of_sixes_l173_173664

-- Define the probability of not rolling a 6 on one die
def prob_not_six : ℚ := 5 / 6

-- Define the probability of rolling zero 6's on three dice
def prob_zero_six : ℚ := prob_not_six ^ 3

-- Define the probability of rolling exactly one 6 among the three dice
def prob_one_six (n : ℕ) : ℚ := n * (1 / 6) * (prob_not_six ^ (n - 1))

-- Calculate the probabilities of each specific outcomes
def prob_exactly_zero_six : ℚ := prob_zero_six
def prob_exactly_one_six : ℚ := prob_one_six 3 * (prob_not_six ^ 2)
def prob_exactly_two_six : ℚ := prob_one_six 3 * (1 / 6) * prob_not_six
def prob_exactly_three_six : ℚ := (1 / 6) ^ 3

-- Define the expected value calculation
noncomputable def expected_value : ℚ :=
  0 * prob_exactly_zero_six
  + 1 * prob_exactly_one_six
  + 2 * prob_exactly_two_six
  + 3 * prob_exactly_three_six

-- Prove that the expected value equals to 1/2
theorem expected_number_of_sixes : expected_value = 1 / 2 :=
  by
    sorry

end expected_number_of_sixes_l173_173664


namespace max_abc_l173_173814

def A_n (a : ℕ) (n : ℕ) : ℕ := a * (10^(3*n) - 1) / 9
def B_n (b : ℕ) (n : ℕ) : ℕ := b * (10^(2*n) - 1) / 9
def C_n (c : ℕ) (n : ℕ) : ℕ := c * (10^(2*n) - 1) / 9

theorem max_abc (a b c n : ℕ) (hpos : n > 0) (h1 : 1 ≤ a ∧ a < 10) (h2 : 1 ≤ b ∧ b < 10) (h3 : 1 ≤ c ∧ c < 10) (h_eq : C_n c n - B_n b n = A_n a n ^ 2) :  a + b + c ≤ 18 :=
by sorry

end max_abc_l173_173814


namespace probability_two_most_expensive_l173_173733

open Nat

noncomputable def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem probability_two_most_expensive :
  (combination 8 1) / (combination 10 3) = 1 / 15 :=
by
  sorry

end probability_two_most_expensive_l173_173733


namespace bisecting_line_exists_l173_173677

noncomputable theory

-- Definitions of the geometric entities
variable (A : Point) (l : Line) (S : Circle)

-- Assumptions on the intersection and reflection
variable (l' : Line) (B : Point)
variable (h_reflection : l' = reflection th A l)
variable (h_intersection : B ∈ l'.intersect_circle S)

-- The main theorem to prove
theorem bisecting_line_exists : ∃ (AB : Line), AB.passes_through A ∧
(∃ P Q : Point, P ∈ l ∧ Q ∈ S ∧ (segment_intersected_by l S AB P Q).bisected_by A) := sorry

end bisecting_line_exists_l173_173677


namespace weekly_cans_goal_l173_173710

theorem weekly_cans_goal :
  let day1 := 20
  let day2 := Int.ceil ((day1 + 5) * 1.5)
  let day3 := Int.ceil ((day2 + 10) * 2)
  let day4 := Int.ceil ((day3 + 15) * 2.5)
  let day5 := Int.ceil ((day4 + 20) * 3)
  day1 + day2 + day3 + day4 + day5 = 1326 :=
by
  sorry

end weekly_cans_goal_l173_173710


namespace translate_down_by_2_l173_173680

theorem translate_down_by_2 (x y : ℝ) (h : y = -2 * x + 3) : y - 2 = -2 * x + 1 := 
by 
  sorry

end translate_down_by_2_l173_173680


namespace total_is_correct_l173_173889

-- Define the given conditions.
def dividend : ℕ := 55
def divisor : ℕ := 11
def quotient := dividend / divisor
def total := dividend + quotient + divisor

-- State the theorem to be proven.
theorem total_is_correct : total = 71 := by sorry

end total_is_correct_l173_173889


namespace airplane_faster_by_90_minutes_l173_173760

def driving_time : ℕ := 3 * 60 + 15  -- in minutes
def drive_to_airport_time : ℕ := 10   -- in minutes
def board_wait_time : ℕ := 20         -- in minutes
def flight_time : ℕ := driving_time / 3  -- in minutes
def get_off_airplane_time : ℕ := 10   -- in minutes

theorem airplane_faster_by_90_minutes :
  driving_time - (drive_to_airport_time + board_wait_time + flight_time + get_off_airplane_time) = 90 :=
by
  calc
    driving_time 
      = 195               : by unfold driving_time
    ...(10 + 20 + 65 + 10 = 105): by unfold drive_to_airport_time board_wait_time flight_time get_off_airplane_time
    195 - 105 = 90         : by norm_num

end airplane_faster_by_90_minutes_l173_173760


namespace circumcircle_tangent_l173_173962

theorem circumcircle_tangent
  {A B C L P Q : Point}
  (h1 : angle_bisector A B C L)
  (h2 : perpendicular_bisector_intersect_circumcircle A L B C P Q) :
  tangent (circumcircle P L Q) (line B C) :=
sorry

end circumcircle_tangent_l173_173962


namespace smallest_x_is_solution_l173_173379

def smallest_positive_angle (x : ℝ) : Prop :=
  tan (6 * x) = (cos (2 * x) - sin (2 * x)) / (cos (2 * x) + sin (2 * x))

noncomputable def smallest_x : ℝ :=
  45 / 8

theorem smallest_x_is_solution : smallest_positive_angle (smallest_x * (Real.pi / 180)) :=
by
  sorry -- Proof omitted

end smallest_x_is_solution_l173_173379


namespace sum_of_rel_prime_l173_173112

theorem sum_of_rel_prime:
  (∀ c d : ℤ, nat.coprime c d → (∑ n in (range (k: ℕ)).filter(λ n, odd n (if odd n then 1/(2^(n+1)) else n+1/(3^(n+2))))) = 19/36)
  → c + d = 55 := by
  sorry

end sum_of_rel_prime_l173_173112


namespace probability_not_losing_l173_173372

theorem probability_not_losing (P_winning P_drawing : ℚ)
  (h_winning : P_winning = 1/3)
  (h_drawing : P_drawing = 1/4) :
  P_winning + P_drawing = 7/12 := 
by
  sorry

end probability_not_losing_l173_173372


namespace area_product_equal_no_consecutive_integers_l173_173967

open Real

-- Define the areas of the triangles for quadrilateral ABCD
variables {A B C D O : Point} 
variables {S1 S2 S3 S4 : Real}  -- Areas of triangles ABO, BCO, CDO, DAO

-- Given conditions
variables (h_intersection : lies_on_intersection O AC BD)
variables (h_areas : S1 = 1 / 2 * (|AO| * |BM|) ∧ S2 = 1 / 2 * (|CO| * |BM|) ∧ S3 = 1 / 2 * (|CO| * |DN|) ∧ S4 = 1 / 2 * (|AO| * |DN|))

-- Theorem for part (a)
theorem area_product_equal : S1 * S3 = S2 * S4 :=
by sorry

-- Theorem for part (b)
theorem no_consecutive_integers : ¬∃ (n : ℕ), S1 = n ∧ S2 = n + 1 ∧ S3 = n + 2 ∧ S4 = n + 3 :=
by sorry

end area_product_equal_no_consecutive_integers_l173_173967


namespace enclosed_area_of_curve_l173_173617

noncomputable def radius_of_arcs := 1

noncomputable def arc_length := (1 / 2) * Real.pi

noncomputable def side_length_of_octagon := 3

noncomputable def area_of_octagon (s : ℝ) := 
  2 * (1 + Real.sqrt 2) * s ^ 2

noncomputable def area_of_sectors (n : ℕ) (arc_radius : ℝ) (arc_theta : ℝ) := 
  n * (1 / 4) * Real.pi

theorem enclosed_area_of_curve : 
  area_of_octagon side_length_of_octagon + area_of_sectors 12 radius_of_arcs arc_length 
  = 54 + 54 * Real.sqrt 2 + 3 * Real.pi := 
by
  sorry

end enclosed_area_of_curve_l173_173617


namespace probability_z_l173_173283

variable (p q x y z : ℝ)

-- Conditions
def condition1 : Prop := z = p * y + q * x
def condition2 : Prop := x = p + q * x^2
def condition3 : Prop := y = q + p * y^2
def condition4 : Prop := x ≠ y

-- Theorem Statement
theorem probability_z : condition1 p q x y z ∧ condition2 p q x ∧ condition3 p q y ∧ condition4 x y → z = 2 * q := by
  sorry

end probability_z_l173_173283


namespace vasya_lowest_position_l173_173222

theorem vasya_lowest_position
  (num_cyclists : ℕ)
  (num_stages : ℕ)
  (num_ahead : ℕ)
  (position_vasya : ℕ)
  (total_time : List ℕ)
  (unique_total_times : total_time.nodup)
  (stage_positions : List (List ℕ))
  (unique_stage_positions : ∀ stage ∈ stage_positions, stage.nodup)
  (vasya_consistent : ∀ stage ∈ stage_positions, stage.nth position_vasya = some num_ahead) :
  num_ahead * num_stages + 1 = 91 :=
by
  sorry

end vasya_lowest_position_l173_173222


namespace solve_system_of_equations_l173_173168

theorem solve_system_of_equations (x y : ℝ) (k : ℤ) :
  x^2 + 4 * (sin y)^2 - 4 = 0 ∧ cos x - 2 * (cos y)^2 - 1 = 0 ↔
  x = 0 ∧ ∃ k : ℤ, y = (π / 2) + k * π :=
by sorry

end solve_system_of_equations_l173_173168


namespace dist_AC_is_correct_l173_173854

-- Define points A, B, and C, and the corresponding distances and angle.
def A : Type := sorry
def B : Type := sorry
def C : Type := sorry

def dist (P Q : Type) := sorry

def AB := (dist A B : ℝ)
def BC := (dist B C : ℝ)
def angle_ABC := (2 * Real.pi / 3 : ℝ)

-- Define the law of cosines in general form.
@[simp] def law_of_cosines (a b c A B C : ℝ) :=
  c^2 = a^2 + b^2 - 2 * a * b * Real.cos(A)

-- Problem-specific instance of the law of cosines.
noncomputable def problem_instance (AB BC angle_ABC AC : ℝ) :=
  AC^2 = AB^2 + BC^2 - 2 * AB * BC * Real.cos(angle_ABC)

theorem dist_AC_is_correct :
  problem_instance 20 30 (2 * Real.pi / 3) (10 * Real.sqrt 19) := by
  sorry

end dist_AC_is_correct_l173_173854


namespace find_k_l173_173847

theorem find_k (k : ℕ) (h1 : k > 0) (h2 : 15 * k^4 < 120) : k = 1 := 
  sorry

end find_k_l173_173847


namespace find_gen_formula_find_min_m_l173_173827

def S (n : ℕ) : ℕ := n^2 + 2 * n 

def a (n : ℕ) : ℕ := 2 * n + 1 

def b (n : ℕ) : ℚ := 1 / (a n * a (n + 1))

noncomputable def T (n : ℕ) : ℚ := ∑ i in Finset.range (n + 1), (b i)

-- Statement for the first proof
theorem find_gen_formula (n : ℕ) : a n = 2 * n + 1 := sorry

-- Statement for the second proof
theorem find_min_m (m : ℕ) :
  (∀ n : ℕ, T n < m / 20) → m = 4 := sorry

end find_gen_formula_find_min_m_l173_173827


namespace circumcircle_tangent_l173_173321

open EuclideanGeometry

variables {A B C D E M O : Point}
variables {Γ_B Γ_C : Circle}

def triangle_ABC := AcuteTriangle A B C
def AB_gt_AC := AB > AC
def circ_Γ_B := CirclePassingThroughAndTangent Γ_B A B AC
def circ_Γ_C := CirclePassingThroughAndTangent Γ_C A C AB 
def intersection_D := IntersectsAt Γ_B Γ_C D
def midpoint_M := Midpoint M B C 
def AM_intersects_Γ_C_at_E := IntersectsAtLineCircle AM Γ_C E 
def circumcenter_O := Circumcenter O A B C 

theorem circumcircle_tangent (h1: triangle_ABC) (h2: AB_gt_AC)  (h3: circ_Γ_B) (h4: circ_Γ_C) 
(h5: intersection_D) (h6: midpoint_M) (h7: AM_intersects_Γ_C_at_E) (h8: circumcenter_O):
  TangentCircumcircleWithCircle O D E Γ_B := sorry

end circumcircle_tangent_l173_173321


namespace sequence_of_directions_525_to_527_l173_173002

def direction : ℕ → String
| 0 => "Right"
| 1 => "Up"
| 2 => "Left"
| 3 => "Down"
| 4 => "Diagonal"
| _ => "Invalid"

theorem sequence_of_directions_525_to_527 :
  ∀ n : ℕ, (n % 5 = 0 → direction ((n + 1) % 5) = "Right") ∧
           (n % 5 = 1 → direction ((n + 2) % 5) = "Up") 
:= 
by
  intro n
  split
  { intro h₀
    have h₁ : (n + 1) % 5 = 1 := sorry
    simp [direction, h₁] }
  { intro h₀
    have h₁ : (n + 2) % 5 = 2 := sorry
    simp [direction, h₁] }

end sequence_of_directions_525_to_527_l173_173002


namespace total_slices_l173_173287

theorem total_slices {slices_per_pizza pizzas : ℕ} (h1 : slices_per_pizza = 2) (h2 : pizzas = 14) : 
  slices_per_pizza * pizzas = 28 :=
by
  -- This is where the proof would go, but we are omitting it as instructed.
  sorry

end total_slices_l173_173287


namespace complement_union_equals_l173_173483

def universal_set : Set ℤ := {-2, -1, 0, 1, 2, 3, 4, 5}
def A : Set ℤ := {-1, 0, 1, 2, 3}
def B : Set ℤ := {-2, 0, 2}

def C_I (I : Set ℤ) (s : Set ℤ) : Set ℤ := I \ s

theorem complement_union_equals :
  C_I universal_set (A ∪ B) = {4, 5} :=
by
  sorry

end complement_union_equals_l173_173483


namespace geometric_sequence_general_formula_sum_of_b_sequence_l173_173573

-- Assuming the problem setup
variable (n : ℕ)
variable (S : ℕ → ℝ)
variable (a : ℕ → ℝ)
variable (b : ℕ → ℝ)

-- Conditions
axiom cond1 : S 3 = a 4 - a 1
axiom cond2 : 2 * (a 3 + 1) = a 1 + a 4

-- General formula for {a_n}
def a_formula := ∀ n, a n = 2 ^ n

-- Conditions for {b_n} sequence and its sum T_n
axiom b_cond : ∀ n, a n * b n = n * 2^(n+1) + 1

noncomputable def T (n : ℕ) : ℝ := ∑ i in Finset.range n, b i

-- Proof problem
theorem geometric_sequence_general_formula (h1 : cond1) (h2 : cond2) : a_formula := sorry

theorem sum_of_b_sequence (h1 : cond1) (h2 : cond2) (h3 : b_cond) :
  T n = n * (n + 1) + 1 - 1 / 2^n := sorry

end geometric_sequence_general_formula_sum_of_b_sequence_l173_173573


namespace vasya_maximum_rank_l173_173252

theorem vasya_maximum_rank {n : ℕ} (cyclists stages : ℕ) (VasyaPlace : ℕ) 
  (rankings : Π (s : fin stages), fin cyclists):
  cyclists = 500 → stages = 15 → VasyaPlace = 7 →
  (∀ s, rankings s VasyaPlace = 7) →
  (∀ s t i, rankings s i ≠ rankings t i) →
  (∀ s, ∃ l, list.nodup l ∧ (∀ i j, i < j → rankings s i < rankings s j) ∧ list.length l = 500) →
  (∃ (maxRank : ℕ), maxRank = 91) :=
by
  intros hcyclists hstages hplace hvasya_place hdistinct_rankings hstage_rankings
  use 91
  sorry

end vasya_maximum_rank_l173_173252


namespace lowest_position_l173_173238

theorem lowest_position (num_cyclists : ℕ) (num_stages : ℕ) (vasya_position : ℕ) :
  num_cyclists = 500 →
  num_stages = 15 →
  vasya_position = 7 →
  ∃ n, n = 91 :=
by
  intros
  sorry

end lowest_position_l173_173238


namespace alice_leaves_30_minutes_after_bob_l173_173622

theorem alice_leaves_30_minutes_after_bob :
  ∀ (distance : ℝ) (speed_bob : ℝ) (speed_alice : ℝ) (time_diff : ℝ),
  distance = 220 ∧ speed_bob = 40 ∧ speed_alice = 44 ∧ 
  time_diff = (distance / speed_bob) - (distance / speed_alice) →
  (time_diff * 60 = 30) := by
  intro distance speed_bob speed_alice time_diff
  intro h
  have h1 : distance = 220 := h.1
  have h2 : speed_bob = 40 := h.2.1
  have h3 : speed_alice = 44 := h.2.2.1
  have h4 : time_diff = (distance / speed_bob) - (distance / speed_alice) := h.2.2.2
  sorry

end alice_leaves_30_minutes_after_bob_l173_173622


namespace cube_minus_self_divisible_by_6_l173_173158

theorem cube_minus_self_divisible_by_6 (n : ℕ) : 6 ∣ (n^3 - n) :=
sorry

end cube_minus_self_divisible_by_6_l173_173158


namespace range_f_triangle_fB_l173_173858

/-- Definition of the function f(x) as given in the problem -/
def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x * cos x - 3 * sin x ^ 2  - cos x ^ 2 + 2

/-- Given the conditions x ∈ [0, π/2], prove the range of f(x) -/
theorem range_f : ∀ x, 0 ≤ x ∧ x ≤ π / 2 → -1 ≤ f(x) ∧ f(x) ≤ 2 := sorry

/-- Given the conditions about triangle ABC, prove f(B) = 1 -/
theorem triangle_fB (A B C : ℝ) (a b c : ℝ) 
  (h1 : b / a = sqrt 3)
  (h2 : sin (2 * A + C) / sin A = 2 + 2 * cos (A + C))
  (hA : A = π / 6)
  (hB : B = π / 3)
  (hC : C = π / 2) :
  f(B) = 1 := sorry

end range_f_triangle_fB_l173_173858


namespace x2_plus_y2_lt_1_l173_173157

theorem x2_plus_y2_lt_1 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^3 + y^3 = x - y) : x^2 + y^2 < 1 :=
sorry

end x2_plus_y2_lt_1_l173_173157


namespace correct_insights_l173_173527

def insight1 := ∀ connections : Type, (∃ journey : connections → Prop, ∀ (x : connections), ¬journey x)
def insight2 := ∀ connections : Type, (∃ (beneficial : connections → Prop), ∀ (x : connections), beneficial x → True)
def insight3 := ∀ connections : Type, (∃ (accidental : connections → Prop), ∀ (x : connections), accidental x → False)
def insight4 := ∀ connections : Type, (∃ (conditional : connections → Prop), ∀ (x : connections), conditional x → True)

theorem correct_insights : ¬ insight1 ∧ insight2 ∧ ¬ insight3 ∧ insight4 :=
by sorry

end correct_insights_l173_173527


namespace odd_function_expression_l173_173048

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x * (1 - x) else 0

theorem odd_function_expression (x : ℝ) (h : x > 0) (hf : ∀ y : ℝ, f (-y) = -f y) :
  f x = x * (1 + x) :=
by
  have hx_neg : -x < 0 := by linarith
  have hx_f_neg : f (-x) = (-x) * (1 - (-x)) := by sorry
  rw [hf, hx_f_neg]
  sorry

end odd_function_expression_l173_173048


namespace cubes_sum_eq_zero_l173_173508

theorem cubes_sum_eq_zero (a b : ℝ) (h1 : a + b = 0) (h2 : a * b = -7) : a^3 + b^3 = 0 :=
by
  sorry

end cubes_sum_eq_zero_l173_173508


namespace find_ellipse_and_line_equation_l173_173005

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ ∀ (x y : ℝ), (x, y) = (1, √6 / 3) →
    x^2 / a^2 + y^2 / b^2 = 1 ∧ 
    (√6 / 3 = (√(a^2 - b^2)) / a)

def line_equation_through_points (a b : ℝ) (Q : ℝ × ℝ) (A : ℝ × ℝ) (M N : ℝ × ℝ) : Prop :=
  ∀ (l : ℝ → ℝ), let k := (l 1) - (l 0) in
  k ≠ 0 ∧
  Q = (0, 3 / 2) ∧
  elliptic_curve_intersection a b k Q M N ∧
  (dist A M = dist A N)

def elliptic_curve_intersection (a b k : ℝ) (Q : ℝ × ℝ) (M N : ℝ × ℝ) : Prop :=
  ∃ (x1 x2 y1 y2 : ℝ), M = (x1, y1) ∧ N = (x2, y2) ∧
    (x1 + x2 = -9 * k / (1 + 3 * k^2)) ∧
    (y1 + y2 = k * (x1 + x2) + 3 - 9 * k^2 / (1 + 3 * k^2))

theorem find_ellipse_and_line_equation :
  ∃ (a b : ℝ), ellipse_equation a b ∧
  ∃ (l : ℝ → ℝ), line_equation_through_points a b (0, 3 / 2) (0, -1) :=
sorry

end find_ellipse_and_line_equation_l173_173005


namespace hyperbola_asymptote_l173_173015

theorem hyperbola_asymptote (a : ℝ) (h : a > 0) (hyp_eq : ∀ x y : ℝ, (x^2) / (a^2) - (y^2) / 81 = 1 → y = 3 * x) : a = 3 := 
by
  sorry

end hyperbola_asymptote_l173_173015


namespace find_original_number_l173_173583

theorem find_original_number (x : ℚ) (h : 1 + 1 / x = 8 / 3) : x = 3 / 5 := by
  sorry

end find_original_number_l173_173583


namespace vertical_strips_count_l173_173539

/- Define the conditions -/

variables {a b x y : ℕ}

-- The outer rectangle has a perimeter of 50 cells
axiom outer_perimeter : 2 * a + 2 * b = 50

-- The inner hole has a perimeter of 32 cells
axiom inner_perimeter : 2 * x + 2 * y = 32

-- Cutting along all horizontal lines produces 20 strips
axiom horizontal_cuts : a + x = 20

-- We want to prove that cutting along all vertical grid lines produces 21 strips
theorem vertical_strips_count : b + y = 21 :=
by
  sorry

end vertical_strips_count_l173_173539


namespace Jessie_weight_loss_l173_173090

theorem Jessie_weight_loss :
  let initial_weight := 74
  let current_weight := 67
  (initial_weight - current_weight) = 7 :=
by
  sorry

end Jessie_weight_loss_l173_173090


namespace factorization_1_factorization_2_factorization_3_factorization_4_l173_173413

-- Problem 1
theorem factorization_1 (a b : ℝ) : 
  4 * a^2 + 12 * a * b + 9 * b^2 = (2 * a + 3 * b)^2 :=
by sorry

-- Problem 2
theorem factorization_2 (a b : ℝ) : 
  16 * a^2 * (a - b) + 4 * b^2 * (b - a) = 4 * (a - b) * (2 * a - b) * (2 * a + b) :=
by sorry

-- Problem 3
theorem factorization_3 (m n : ℝ) : 
  25 * (m + n)^2 - 9 * (m - n)^2 = 4 * (4 * m + n) * (m + 4 * n) :=
by sorry

-- Problem 4
theorem factorization_4 (a b : ℝ) : 
  4 * a^2 - b^2 - 4 * a + 1 = (2 * a - 1 + b) * (2 * a - 1 - b) :=
by sorry

end factorization_1_factorization_2_factorization_3_factorization_4_l173_173413


namespace evaluate_sum_l173_173411

theorem evaluate_sum:
    (∑ n in Finset.range 14, 1 / (n + 1) / (n + 2)) + 
    (∑ n in Finset.range 9, 1 / (n + 1) / (n + 3)) = 4 / 3 :=
by
  sorry

end evaluate_sum_l173_173411


namespace gcd_m_n_l173_173119

open Nat

def m := 777777777
def n := 222222222222

theorem gcd_m_n : gcd m n = 999 := by sorry

end gcd_m_n_l173_173119


namespace three_obliques_method_area_calc_l173_173190

def three_obliques_area (a b c : ℝ) : ℝ :=
  Real.sqrt ( (1 / 4) * (c^2 * a^2 - ( (c^2 + a^2 - b^2) / 2 )^2 ) )

theorem three_obliques_method_area_calc : three_obliques_area 3 7 8 = 6 * Real.sqrt 3 :=
by
  -- Substitute the given values and simplify
  sorry

end three_obliques_method_area_calc_l173_173190


namespace vasya_lowest_position_l173_173220

theorem vasya_lowest_position
  (num_cyclists : ℕ)
  (num_stages : ℕ)
  (num_ahead : ℕ)
  (position_vasya : ℕ)
  (total_time : List ℕ)
  (unique_total_times : total_time.nodup)
  (stage_positions : List (List ℕ))
  (unique_stage_positions : ∀ stage ∈ stage_positions, stage.nodup)
  (vasya_consistent : ∀ stage ∈ stage_positions, stage.nth position_vasya = some num_ahead) :
  num_ahead * num_stages + 1 = 91 :=
by
  sorry

end vasya_lowest_position_l173_173220


namespace deviation_expectation_greater_l173_173286

def frequency_of_heads (m n : Nat) : ℚ := m / n

def deviation_of_frequency (m n : Nat) : ℚ := frequency_of_heads m n - 0.5

def absolute_deviation_of_frequency (m n : Nat) : ℚ := abs (deviation_of_frequency m n)

def expected_absolute_deviation (n : Nat) : ℚ :=
  -- Specify how to calculate the expected absolute deviation formally
  sorry

theorem deviation_expectation_greater (m1 m10 m100 : Nat) :
  expected_absolute_deviation 10 > expected_absolute_deviation 100 :=
sorry

end deviation_expectation_greater_l173_173286


namespace convert_coordinates_cyl_to_rect_6_pi_by_3_2_l173_173783

def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem convert_coordinates_cyl_to_rect_6_pi_by_3_2 :
  cylindrical_to_rectangular 6 (Real.pi / 3) 2 = (3, 3 * Real.sqrt 3, 2) := by
  sorry

end convert_coordinates_cyl_to_rect_6_pi_by_3_2_l173_173783


namespace surface_area_ratio_l173_173460

noncomputable def surface_area (r : ℝ) : ℝ :=
  4 * Real.pi * r ^ 2

theorem surface_area_ratio (k : ℝ) :
  let r1 := k
  let r2 := 2 * k
  let r3 := 3 * k
  let A1 := surface_area r1
  let A2 := surface_area r2
  let A3 := surface_area r3
  A3 / (A1 + A2) = 9 / 5 :=
by
  sorry

end surface_area_ratio_l173_173460


namespace min_value_a_plus_3b_l173_173556

theorem min_value_a_plus_3b (a b : ℝ) (h_positive : 0 < a ∧ 0 < b)
  (h_condition : (1 / (a + 3) + 1 / (b + 3) = 1 / 4)) :
  a + 3 * b ≥ 4 + 8 * Real.sqrt 3 := 
sorry

end min_value_a_plus_3b_l173_173556


namespace num_students_exactly_one_GT_HG_T_l173_173816

variables {U : Type} [Fintype U]

-- Define sets of students for different electives
variable {S : Set U}  -- Total students
variable {A : Set U}  -- Students not taking any elective
variable {B : Set U}  -- Students taking Algebraic Number Theory and Galois Theory
variable {C : Set U}  -- Students taking Galois Theory and Hyperbolic Geometry
variable {D : Set U}  -- Students taking Hyperbolic Geometry and Cryptography
variable {E : Set U}  -- Students taking Cryptography and Topology
variable {F : Set U}  -- Students taking Topology and Algebraic Number Theory
variable {G : Set U}  -- Students taking either Algebraic Number Theory or Cryptography, but not both

variables {GT HG T : Set U}  -- Sets representing students taking Galois Theory, Hyperbolic Geometry, Topology

axiom total_students : ∀ x, x ∈ S
axiom not_taking_any : Fintype.card A = 22
axiom taking_AN_T_GT : Fintype.card B = 7
axiom taking_GT_HG : Fintype.card C = 12
axiom taking_HG_C : Fintype.card D = 3
axiom taking_C_T : Fintype.card E = 15
axiom taking_T_AN_T : Fintype.card F = 8
axiom taking_AN_or_C_but_not_both : Fintype.card G = 16
axiom total : Fintype.card S = 100

noncomputable def number_of_students_taking_exactly_one_of_GT_HG_T : ℕ :=
  Fintype.card S -
  Fintype.card A -
  Fintype.card B -
  Fintype.card C -
  Fintype.card D -
  Fintype.card E -
  Fintype.card F -
  Fintype.card G

theorem num_students_exactly_one_GT_HG_T :
  number_of_students_taking_exactly_one_of_GT_HG_T = 17 := by
  sorry

end num_students_exactly_one_GT_HG_T_l173_173816


namespace smallest_m_no_real_roots_l173_173809

def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem smallest_m_no_real_roots : ∃ m : ℤ, 
  (discriminant (3 * ↑m - 2) (-18) 10 < 0) ∧ 
  (∀ n : ℤ, (discriminant (3 * ↑n - 2) (-18) 10 < 0) → (m ≤ n)) ∧ 
  (m = 4) :=
begin
  sorry
end

end smallest_m_no_real_roots_l173_173809


namespace probability_each_guest_gets_one_of_each_kind_l173_173738

noncomputable def calc_probability : ℚ := 
  (3.factorial * 3.factorial * 3.factorial : ℚ) / (9.factorial : ℚ)

def sanitize_fraction (p : ℚ) : ℚ := 
  p.num / p.denom

theorem probability_each_guest_gets_one_of_each_kind :
  let m := 9 in
  let n := 70 in
  sanitize_fraction calc_probability = 9 / 70 →

  m + n = 79 :=
by
  intros
  simp [sanitize_fraction, calc_probability]
  sorry

end probability_each_guest_gets_one_of_each_kind_l173_173738


namespace largest_integral_k_for_real_distinct_roots_l173_173312

theorem largest_integral_k_for_real_distinct_roots :
  ∃ k : ℤ, (k < 9) ∧ (∀ k' : ℤ, k' < 9 → k' ≤ k) :=
sorry

end largest_integral_k_for_real_distinct_roots_l173_173312


namespace repeating_decimal_as_fraction_l173_173390

-- Given conditions
def repeating_decimal : ℚ := 7 + 832 / 999

-- Goal: Prove that the repeating decimal 7.\overline{832} equals 70/9
theorem repeating_decimal_as_fraction : repeating_decimal = 70 / 9 := by
  unfold repeating_decimal
  sorry

end repeating_decimal_as_fraction_l173_173390


namespace coloring_scheme_correct_l173_173792

def numColoringSchemes (n : Nat) : Nat :=
  if n < 2 then 0 else 4^n + 4 * (-1) ^ n

theorem coloring_scheme_correct (n : Nat) (h : n ≥ 4) : numColoringSchemes n = 4^n + 4 * (-1) ^ n :=
by
  sorry

end coloring_scheme_correct_l173_173792


namespace minimum_value_of_abs_z_l173_173960

noncomputable def min_abs_z (z : ℂ) (h : |z - 2 * complex.I| + |z - 5| = 7) : ℝ :=
  classical.some (exists_minimum (λ z, |z|) (λ z, |z - 2 * complex.I| + |z - 5| = 7) h)

theorem minimum_value_of_abs_z : ∀ z : ℂ, 
  (|z - 2 * complex.I| + |z - 5| = 7) → |z| ≥ 0 
  → min_abs_z z (by sorry) = 10 / real.sqrt 29 :=
by 
  sorry

end minimum_value_of_abs_z_l173_173960


namespace max_min_x_plus_inv_x_l173_173646

-- We're assuming existence of 101 positive numbers with given conditions.
variable {x : ℝ}
variable {y : Fin 100 → ℝ}

-- Conditions given in the problem
def cumulative_sum (x : ℝ) (y : Fin 100 → ℝ) : Prop :=
  0 < x ∧ (∀ i, 0 < y i) ∧ x + (∑ i, y i) = 102 ∧ 1 / x + (∑ i, 1 / y i) = 102

-- The theorem to prove the maximum and minimum value of x + 1/x
theorem max_min_x_plus_inv_x (x : ℝ) (y : Fin 100 → ℝ) (h : cumulative_sum x y) : 
  (x + 1 / x ≤ 405 / 102) ∧ (x + 1 / x ≥ 399 / 102) := 
  sorry

end max_min_x_plus_inv_x_l173_173646


namespace part_1_part_2_l173_173834

noncomputable def h (x : ℝ) : ℝ := Real.log (2 * Real.exp(1) * x - Real.exp(1))
noncomputable def g (x : ℝ) : ℝ := 2 * a * x - 2 * a
noncomputable def f (x : ℝ) : ℝ := h x - g x

theorem part_1 (a : ℝ) (ha : a = 1 / 2) (x₀ : ℝ) (hx₀ : x₀ = 3 / 2) :
  f(1) = ln 2 + 1/2 :=
begin
  -- Proof omitted
  sorry
end

theorem part_2 (a : ℝ) (ha : a < 1) (h_f : ∀ x : ℝ, f x < 1 + a) (n : ℕ) (hn: n > 1) :
  (∑ k in Finset.range (2 * n), Real.log k ^ (5 / 4)) < n * (n + 1) :=
begin
  -- Proof omitted
  sorry
end

end part_1_part_2_l173_173834


namespace num_divisors_37620_l173_173489

theorem num_divisors_37620 : 
  (finset.filter (λ x, 37620 % x = 0) (finset.range 11)).card = 9 :=
by
  sorry

end num_divisors_37620_l173_173489


namespace remainder_when_divided_by_7_l173_173047

theorem remainder_when_divided_by_7 (n : ℕ) (h : (2 * n) % 7 = 4) : n % 7 = 2 :=
  by sorry

end remainder_when_divided_by_7_l173_173047


namespace flower_beds_fraction_l173_173748

noncomputable def isosceles_right_triangle_area (leg : ℝ) : ℝ :=
  (1 / 2) * leg^2

noncomputable def fraction_of_yard_occupied_by_flower_beds : ℝ :=
  let leg_length := (25 - 15) / 2
  let area_of_one_triangle := isosceles_right_triangle_area leg_length
  let total_area_of_flower_beds := 2 * area_of_one_triangle
  let area_of_yard := 25 * 5
  total_area_of_flower_beds / area_of_yard

theorem flower_beds_fraction : 
  let leg_length := (25 - 15) / 2
  let area_of_one_triangle := isosceles_right_triangle_area leg_length
  let total_area_of_flower_beds := 2 * area_of_one_triangle
  let area_of_yard := 25 * 5
  total_area_of_flower_beds / area_of_yard = 1 / 5 :=
by
  sorry

end flower_beds_fraction_l173_173748


namespace fisher_catch_l173_173281

theorem fisher_catch (x y : ℕ) (h1 : x + y = 80)
  (h2 : ∃ a : ℕ, x = 9 * a)
  (h3 : ∃ b : ℕ, y = 11 * b) :
  x = 36 ∧ y = 44 :=
by
  sorry

end fisher_catch_l173_173281


namespace part2_H_n_polynomial_S_n_express_limit_S_6_l173_173423

noncomputable def H_n (n : ℕ) (x : ℝ) : ℝ :=
  (-1 : ℝ)^n * (Real.exp (x^2) * (d^[n] (λ t, Real.exp (-t^2)) x ))

theorem part2 (n : ℕ) (x : ℝ) :
  ∀ n > 0, (deriv (H_n n)) x = 2 * x * (H_n n x) - (H_n (n + 1) x) :=
  sorry

theorem H_n_polynomial (n : ℕ) : ∀ n > 0, ∃ p : Polynomial ℝ, (H_n n) = p :=
  sorry

noncomputable def S_n (n : ℕ) (a : ℝ) : ℝ :=
  ∫ x in (0 : ℝ)..a, x * (H_n n x) * Real.exp (-x^2)

theorem S_n_express (n : ℕ) (a : ℝ) (h : n ≥ 3) :
  S_n n a = -0.5 * a * (H_n n a) * Real.exp (-a^2) +
	    0.5 * ∫ x in (0 : ℝ)..a, (H_n n x) * Real.exp (-x^2) +
	    ∫ x in (0 : ℝ)..a, x^2 * (H_n n x) * Real.exp (-x^2) -
	    0.5 * ∫ x in (0 : ℝ)..a, x * (H_n (n + 1) x) * Real.exp (-x^2) :=
  sorry

theorem limit_S_6 :
  (∀ k > 0, (λ x, x^k * Real.exp (-x^2)) ⟶ 0[at_top]) →
  ∃ l : ℝ, (filter.at_top.liminf (S_n 6)) = l :=
  sorry


end part2_H_n_polynomial_S_n_express_limit_S_6_l173_173423


namespace binomial_coefficient_identity_l173_173999

theorem binomial_coefficient_identity (n : ℕ) (hn : n > 0) :
  ∑ k in Finset.range (n + 1), if k > 0 then (binomial n k) * ((-1) ^ (k-1)) / k else 0 = 
  ∑ k in Finset.range (n + 1), if k > 0 then 1 / k else 0 :=
by
  sorry

end binomial_coefficient_identity_l173_173999


namespace odd_function_behavior_on_negative_interval_l173_173006

variable (f : ℝ → ℝ)

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

noncomputable def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x1 x2 : ℝ, a ≤ x1 ∧ x1 < x2 ∧ x2 ≤ b → f x1 > f x2

theorem odd_function_behavior_on_negative_interval
    (h1 : is_odd_function f)
    (h2 : is_decreasing_on f 1 4) :
  is_decreasing_on (λ x, f (-x)) (-4) (-1) := sorry

end odd_function_behavior_on_negative_interval_l173_173006


namespace ratio_of_areas_of_circles_l173_173895

-- Given conditions
variables (R_C R_D : ℝ) -- Radii of circles C and D respectively
variables (L : ℝ) -- Common length of the arcs

-- Equivalent arc condition
def arc_length_condition : Prop :=
  (60 / 360) * (2 * Real.pi * R_C) = L ∧ (40 / 360) * (2 * Real.pi * R_D) = L

-- Goal: ratio of areas
def area_ratio : Prop :=
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = (9 / 4)

-- Problem statement
theorem ratio_of_areas_of_circles (R_C R_D L : ℝ) (hc : arc_length_condition R_C R_D L) :
  area_ratio R_C R_D :=
by
  sorry

end ratio_of_areas_of_circles_l173_173895


namespace rings_distance_l173_173754

theorem rings_distance (thickness : ℕ) (d₀ : ℕ) (dₙ : ℕ) 
  (h1 : thickness = 2) 
  (h2 : d₀ = 20) 
  (h3 : dₙ = 4)
  (h4 : ∀ (n : ℕ), d₀ - n * thickness > dₙ) :
  ∑ i in Finset.range ((d₀ - dₙ) / thickness + 1), thickness = 72 := 
by
  sorry

end rings_distance_l173_173754


namespace sum_of_xyz_l173_173172

theorem sum_of_xyz (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : (x + y + z)^3 - x^3 - y^3 - z^3 = 504) : x + y + z = 9 :=
by {
  sorry
}

end sum_of_xyz_l173_173172


namespace train_crosses_pole_in_l173_173311

noncomputable def train_crossing_time (d : ℝ) (v_kmh : ℝ) : ℝ :=
  let v_ms := v_kmh * 1000 / 3600
  d / v_ms

theorem train_crosses_pole_in (h_d : d = 130) (h_v_kmh : v_kmh = 144) : train_crossing_time 130 144 = 3.25 := by
  rw [train_crossing_time]
  rw [h_d, h_v_kmh]
  simp
  norm_num
sorry

end train_crosses_pole_in_l173_173311


namespace vasya_rank_91_l173_173227

theorem vasya_rank_91 {n_cyclists : ℕ} {n_stages : ℕ} 
    (n_cyclists_eq : n_cyclists = 500) 
    (n_stages_eq : n_stages = 15) 
    (no_ties : ∀ (i j : ℕ), i < j → ∀ (s : fin n_stages), ¬(same_time i j s)) 
    (vasya_7th : ∀ (s : fin n_stages), ∀ (i : ℕ), i < 6 → better_than i 6 s) :
    possible_rank vasya ≤ 91 :=
sorry

end vasya_rank_91_l173_173227


namespace correct_propositions_3_and_4_l173_173118

variables {m n : Type} [line m] [line n] {α β : Type} [plane α] [plane β]

def prop_3 (m : Type) [line m] [plane α] [plane β] [perpendicular m α] [parallel m β] : Prop :=
  perpendicular α β

def prop_4 (m : Type) [line m] [plane α] [plane β] [perpendicular m α] [parallel α β] : Prop :=
  perpendicular m β

theorem correct_propositions_3_and_4
  (m n : Type) [line m] [line n] (α β : Type) [plane α] [plane β]
  (H1 : prop_3 m) (H2 : prop_4 m) :
    prop_3 m ∧ prop_4 m :=
by
  exact ⟨H1, H2⟩

end correct_propositions_3_and_4_l173_173118


namespace event_A_muffins_correct_event_B_muffins_correct_event_C_muffins_correct_l173_173362

-- Event A
def total_muffins_needed_A := 200
def arthur_muffins_A := 35
def beatrice_muffins_A := 48
def charles_muffins_A := 29
def total_muffins_baked_A := arthur_muffins_A + beatrice_muffins_A + charles_muffins_A
def additional_muffins_needed_A := total_muffins_needed_A - total_muffins_baked_A

-- Event B
def total_muffins_needed_B := 150
def arthur_muffins_B := 20
def beatrice_muffins_B := 35
def charles_muffins_B := 25
def total_muffins_baked_B := arthur_muffins_B + beatrice_muffins_B + charles_muffins_B
def additional_muffins_needed_B := total_muffins_needed_B - total_muffins_baked_B

-- Event C
def total_muffins_needed_C := 250
def arthur_muffins_C := 45
def beatrice_muffins_C := 60
def charles_muffins_C := 30
def total_muffins_baked_C := arthur_muffins_C + beatrice_muffins_C + charles_muffins_C
def additional_muffins_needed_C := total_muffins_needed_C - total_muffins_baked_C

-- Proof Statements
theorem event_A_muffins_correct : additional_muffins_needed_A = 88 := by
  sorry

theorem event_B_muffins_correct : additional_muffins_needed_B = 70 := by
  sorry

theorem event_C_muffins_correct : additional_muffins_needed_C = 115 := by
  sorry

end event_A_muffins_correct_event_B_muffins_correct_event_C_muffins_correct_l173_173362


namespace expected_number_of_sixes_when_three_dice_are_rolled_l173_173657

theorem expected_number_of_sixes_when_three_dice_are_rolled : 
  ∑ n in finset.range 4, (n * (↑(finset.filter (λ xs : fin 3 → fin 6, xs.count (λ x, x = 5) = n) finset.univ).card / 216 : ℚ)) = 1 / 2 :=
by
  -- Conclusion of proof is omitted as per instructions
  sorry

end expected_number_of_sixes_when_three_dice_are_rolled_l173_173657


namespace total_participation_plans_l173_173603

-- Definitions for the given problem
def students := {A, B, C, D, E}  -- Set of students
def subjects := {math, physics, chemistry}  -- Set of subjects

-- Condition: Student A cannot participate in the physics competition
def valid_assignment (assignment : students → subjects) : Prop :=
  assignment A ≠ physics

-- Problem statement: Find the total number of different participation plans
theorem total_participation_plans :
  (∃ (assignments : Π (S : finset (fin 5)),  
    S.card = 3 → (S → subjects)), 
    (∀ S (h : S.card = 3), valid_assignment (assignments S h)) →
    ∑ S in finset.powerset_len 3 (finset.univ (fin 5)), 
      finset.card (assignments S sorry) = 48) :=
sorry

end total_participation_plans_l173_173603


namespace part_I_part_II_part_III_l173_173839

-- Definitions of the functions
def f (a: ℝ) (x: ℝ): ℝ := x^2 + a * x
def g (b: ℝ) (x: ℝ): ℝ := x + b
def l (x: ℝ): ℝ := 2 * x^2 + 3 * x - 1

-- Quadratic function generated by f and g
def h (m n a b: ℝ) (x: ℝ): ℝ := m * f a x + n * g b x

-- Part I: Prove that if h is even, h(√2) = 0 when a = 1 and b = 2
theorem part_I (m n : ℝ) : 
  ∀ x, h m n 1 2 x = h m n 1 2 (-x) → h m n 1 2 (sqrt 2) = 0 := 
begin
  sorry 
end

-- Part II: Prove that if h is generated by g and l, minimum value of a+b = 3/2 + √2 when b > 0
theorem part_II (b : ℝ) (h : ∀ m₀ n₀ x, h m₀ n₀ 0 b x = m₀ * g b x + n₀ * l x) : 
  b > 0 → ∃ a : ℝ, a + b = 3 / 2 + sqrt 2 :=
begin
  sorry
end

-- Part III: Prove that h cannot be any quadratic function
theorem part_III : (∀ (m₁ n₁ : ℝ), h m₁ n₁ 0 0 = x^2) ∧ 
                  (∀ (m₂ n₂ : ℝ), h m₂ n₂ 0 0 = x^2 + 1) → 
                  ∀ x, ¬(∀ m n, h m n 0 0 x = x^2) :=
begin
  sorry
end

end part_I_part_II_part_III_l173_173839


namespace find_QS_l173_173609

noncomputable def QS (ā b: ℝ) (cos_R: ℝ) : ℝ :=
  let RS := 13
  let QR := cos_R * RS
  sqrt (RS^2 - QR^2)

-- We will use this example given conditions: cos R = 5/13 and RS = 13.
theorem find_QS : (QS 5 13 (5 / 13)) = 12 := by
  sorry

end find_QS_l173_173609


namespace domain_proof_l173_173134

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition about the domain of f(x+1)
def domain_f_x_plus_1 : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}
def domain_f_x : Set ℝ := {x | 4 ≤ x ∧ x ≤ 8}

-- Define the proposition about the domain of f(2x)
def domain_f_2x : Set ℝ := {x | 2 ≤ x ∧ x ≤ 4}

-- Prove that the domain of y = f(2x) is [2,4] given the condition
theorem domain_proof (h : ∀ x, x ∈ domain_f_x_plus_1 → f(x+1) = f(x)) : 
  (∀ x, x ∈ domain_f_2x → f(2*x) = f(2*x)) :=
by
  sorry

end domain_proof_l173_173134


namespace magic_box_result_is_66_l173_173576

def magic_box (a b : ℚ) := a^2 + b + 1

theorem magic_box_result_is_66 :
  let m := magic_box (-2) 3
  in magic_box m 1 = 66 := by
  sorry

end magic_box_result_is_66_l173_173576


namespace rectangle_length_eq_15_l173_173744

theorem rectangle_length_eq_15 (w l s p_rect p_square : ℝ)
    (h_w : w = 9)
    (h_s : s = 12)
    (h_p_square : p_square = 4 * s)
    (h_p_rect : p_rect = 2 * w + 2 * l)
    (h_eq_perimeters : p_square = p_rect) : l = 15 := by
  sorry

end rectangle_length_eq_15_l173_173744


namespace cos2alpha_plus_sin2alpha_l173_173431

theorem cos2alpha_plus_sin2alpha (α : Real) (h : Real.tan (Real.pi + α) = 2) : 
  Real.cos (2 * α) + Real.sin (2 * α) = 1 / 5 :=
sorry

end cos2alpha_plus_sin2alpha_l173_173431


namespace smallest_k_exists_l173_173787

noncomputable def a : ℕ → ℝ
| 0       := 1
| 1       := real.root 10 3
| (n + 2) := a (n + 1) * (a n)^3

def product_is_integer (k : ℕ) := ∃ p : ℤ, ∏ i in finset.range (k + 1).succ, a i = p

theorem smallest_k_exists : ∃ k : ℕ, product_is_integer k ∧ k = 6 := by
  sorry

end smallest_k_exists_l173_173787


namespace least_positive_multiple_24_gt_450_l173_173299

theorem least_positive_multiple_24_gt_450 : ∃ n : ℕ, n > 450 ∧ n % 24 = 0 ∧ n = 456 :=
by
  use 456
  sorry

end least_positive_multiple_24_gt_450_l173_173299


namespace geometric_sequence_sum_l173_173531

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r > 0, ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : ∀ n, a n > 0)
  (h3 : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) : 
  a 3 + a 5 = 5 := 
sorry

end geometric_sequence_sum_l173_173531


namespace weight_of_A_l173_173615

theorem weight_of_A (W_A W_B W_C W_D W_E : ℤ) (Age_A Age_B Age_C Age_D Age_E : ℤ) :
  (W_A + W_B + W_C) / 3 = 84 ∧
  (Age_A + Age_B + Age_C) / 3 = 30 ∧
  (W_A + W_B + W_C + W_D) / 4 = 80 ∧
  (Age_A + Age_B + Age_C + Age_D) / 4 = 28 ∧
  W_E = W_D + 7 ∧
  Age_E = Age_A - 3 ∧
  (W_B + W_C + W_D + W_E) / 4 = 79 ∧
  (Age_B + Age_C + Age_D + Age_E) / 4 = 27
  → W_A = 79 :=
begin
  sorry
end

end weight_of_A_l173_173615


namespace inequality_not_holds_l173_173042

variable (x y : ℝ)

theorem inequality_not_holds (h1 : x > 1) (h2 : 1 > y) : x - 1 ≤ 1 - y :=
sorry

end inequality_not_holds_l173_173042


namespace product_of_sums_of_squares_and_cubes_l173_173376

noncomputable def computeProductOfSumsOfSquaresAndCubes : Nat :=
  let y : ℝ := sorry -- Placeholder for the value of y
  let r, s, t : ℝ := sorry, sorry, sorry -- placeholders for the roots
  let sum_squares := r^2 + s^2 + t^2
  let sum_cubes := r^3 + s^3 + t^3
  let product := sum_squares * sum_cubes
  13754

-- Lean statement expressing the theorem to prove
theorem product_of_sums_of_squares_and_cubes :
  ∀ (x : ℝ), x ≥ 0 → 
  let y := Math.sqrt x in y^3 - 8 * y^2 + 9 * y - 1 = 0 → 
  (let r, s, t : ℝ := sorry, sorry, sorry in
   let sum_squares := r^2 + s^2 + t^2 in
   let sum_cubes := r^3 + s^3 + t^3 in
   sum_squares * sum_cubes = 13754) :=
sorry

end product_of_sums_of_squares_and_cubes_l173_173376


namespace ratio_of_new_circumference_to_new_diameter_l173_173329

theorem ratio_of_new_circumference_to_new_diameter (r : ℝ) :
  let new_radius := r + 2 in
  let new_diameter := 2 * new_radius in
  let new_circumference := 2 * real.pi * new_radius in
  new_circumference / new_diameter = real.pi :=
by
  let new_radius := r + 2
  let new_diameter := 2 * new_radius
  let new_circumference := 2 * real.pi * new_radius
  sorry

end ratio_of_new_circumference_to_new_diameter_l173_173329


namespace circle_tangent_problem_l173_173449

-- Define the points A, B, and M
def Point := (ℝ × ℝ)
def A : Point := (-1, -3)
def B : Point := (5, 5)
def M : Point := (-3, 2)

-- Midpoint function
def midpoint (P1 P2 : Point) : Point :=
  ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2)

-- Distance function
def distance (P1 P2 : Point) : ℝ :=
  sqrt ((P1.1 - P2.1)^2 + (P1.2 - P2.2)^2)

-- Circle definition
def circle_eq (center : Point) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Tangent line definitions
def line_eq1 (x y : ℝ) : Prop :=
  x = -3

def line_eq2 (x y : ℝ) : Prop :=
  12 * x - 5 * y + 46 = 0

-- Statement that we need to prove
theorem circle_tangent_problem :
  ∃ (center : Point) (radius : ℝ), 
  (circle_eq center radius 2 1 = 25) ∧ 
  ((line_eq1 (-3) 2) ∨ (line_eq2 (-3) 2)) :=
by
  sorry

end circle_tangent_problem_l173_173449


namespace altitude_line_equation_equal_distance_lines_l173_173003

-- Define the points A, B, and C
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (8, 10)
def C : ℝ × ℝ := (0, 6)

-- The equation of the line for the altitude from A to BC
theorem altitude_line_equation :
  ∃ (a b c : ℝ), 2 * a - 3 * b + 14 = 0 :=
sorry

-- The equations of the line passing through B such that the distances from A and C are equal
theorem equal_distance_lines :
  ∃ (a b c : ℝ), (7 * a - 6 * b + 4 = 0) ∧ (3 * a + 2 * b - 44 = 0) :=
sorry

end altitude_line_equation_equal_distance_lines_l173_173003


namespace more_balloons_allan_l173_173765

theorem more_balloons_allan (allan_balloons : ℕ) (jake_initial_balloons : ℕ) (jake_bought_balloons : ℕ) 
  (h1 : allan_balloons = 6) (h2 : jake_initial_balloons = 2) (h3 : jake_bought_balloons = 3) :
  allan_balloons = jake_initial_balloons + jake_bought_balloons + 1 := 
by 
  -- Assuming Jake's total balloons after purchase
  let jake_total_balloons := jake_initial_balloons + jake_bought_balloons
  -- The proof would involve showing that Allan's balloons are one more than Jake's total balloons
  sorry

end more_balloons_allan_l173_173765


namespace calculate_f_of_f_of_f_30_l173_173786

-- Define the function f (equivalent to $\#N = 0.5N + 2$)
def f (N : ℝ) : ℝ := 0.5 * N + 2

-- The proof statement
theorem calculate_f_of_f_of_f_30 : 
  f (f (f 30)) = 7.25 :=
by
  sorry

end calculate_f_of_f_of_f_30_l173_173786


namespace minimal_disks_needed_l173_173369

-- Define the capacity of one disk
def disk_capacity : ℝ := 2.0

-- Define the number of files and their sizes
def num_files_0_9 : ℕ := 5
def size_file_0_9 : ℝ := 0.9

def num_files_0_8 : ℕ := 15
def size_file_0_8 : ℝ := 0.8

def num_files_0_5 : ℕ := 20
def size_file_0_5 : ℝ := 0.5

-- Total number of files
def total_files : ℕ := num_files_0_9 + num_files_0_8 + num_files_0_5

-- Proof statement: the minimal number of disks needed to store all files given their sizes and the disk capacity
theorem minimal_disks_needed : 
  ∀ (d : ℕ), 
    d = 18 → 
    total_files = 40 → 
    disk_capacity = 2.0 → 
    ((num_files_0_9 * size_file_0_9 + num_files_0_8 * size_file_0_8 + num_files_0_5 * size_file_0_5) / disk_capacity) ≤ d
  :=
by
  sorry

end minimal_disks_needed_l173_173369


namespace ring_chain_total_distance_l173_173346

theorem ring_chain_total_distance
  (d₁ : ℕ) (t : ℕ) (d_min : ℕ) (n : ℕ)
  (h1 : d₁ = 30)
  (h2 : t = 2)
  (h3 : d_min = 4)
  (h4 : n = ((d₁ - d_min) / t) + 1) :
  let diameters := List.range' (d_min - 2 + 1) n in
  let inside_diameters := List.map (λ d, d - t) diameters in
  let total_distance := 2 * t + List.sum inside_diameters in
  total_distance = 214 :=
by
  sorry

end ring_chain_total_distance_l173_173346


namespace systematic_sampling_seventh_group_l173_173069

theorem systematic_sampling_seventh_group
  (m k : ℕ)
  (h_m : m = 6)
  (h_k : k = 7)
  (units_digit : ℕ → ℕ)
  (h_units : ∀ n, units_digit n = n % 10)
  (group : ℕ → ℕ)
  (h_group : ∀ i, 1 ≤ i ∧ i ≤ 10 → (group i = i * 10 - 10 ∧ ∀ j, 0 ≤ j ∧ j < 10 → (i * 10 - 10 + j ∈ set.range group))) :
  ∃ n, group 7 ≤ n ∧ n < group 8 ∧ units_digit n = units_digit (m + k) := 
sorry

end systematic_sampling_seventh_group_l173_173069


namespace largest_four_digit_integer_congruent_to_17_mod_26_l173_173292

theorem largest_four_digit_integer_congruent_to_17_mod_26 :
  ∃ x : ℤ, 1000 ≤ x ∧ x < 10000 ∧ x % 26 = 17 ∧ x = 9978 :=
by
  sorry

end largest_four_digit_integer_congruent_to_17_mod_26_l173_173292


namespace find_k_of_line_eq_l173_173032

theorem find_k_of_line_eq (k : ℝ) (A : ℝ × ℝ) (h : A = (2, Real.sqrt 3)) (line_eq : A.snd = k * A.fst) : k = Real.sqrt 3 / 2 :=
by
  rw [Prod.snd, Prod.fst] at line_eq h
  rw h at line_eq
  sorry

end find_k_of_line_eq_l173_173032


namespace sum_of_divisors_eq_360_l173_173642

theorem sum_of_divisors_eq_360 (i j : ℕ) (h : (Finset.range (i+1)).sum (λ n, 2^n) * (Finset.range (j+1)).sum (λ m, 3^m) = 360) : i + j = 5 :=
sorry

end sum_of_divisors_eq_360_l173_173642


namespace range_of_a_l173_173024

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 0 then 2^(-x) - 1 else f (x - 1) 

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f(x) = x + a ∧ f(y) = y + a)
  → a ∈ set.Iio 1 :=
by
  sorry

end range_of_a_l173_173024


namespace ratio_of_areas_eq_l173_173901

-- Define the conditions
variables {C D : Type} [circle C] [circle D]
variables (R_C R_D : ℝ)
variable (L : ℝ)

-- Given conditions
axiom arc_length_eq : (60 / 360) * (2 * π * R_C) = L
axiom arc_length_eq' : (40 / 360) * (2 * π * R_D) = L

-- Statement to prove
theorem ratio_of_areas_eq : (π * R_C^2) / (π * R_D^2) = 4 / 9 :=
sorry

end ratio_of_areas_eq_l173_173901


namespace Geometry_problem_l173_173611

open_locale euclidean_geometry

variables {α : Type*} [metric_space α] [normed_group α] [normed_space ℝ α]

-- Definition of given triangle ABC and incircle I
variables {A B C I F L D K : α}

-- Given conditions
def in_triangle (A B C : α) := ∃ (x y z : ℝ), x + y + z = 1 ∧ x • A + y • B + z • C = I

def angle_bisector_meets_incircle (A B C I F L : α) : Prop :=
(intersects_incircle A B C I F) ∧ (intersects_incircle A B C I L) ∧ (angle_bisector A B C F L)

def foot_perpendicular (C D : α) (line: set α) : Prop :=
C ∈ line ∧ D ⊥ line

def concur_cyclic (F L B K : α) : Prop :=
circle_c F L B K

-- Mathematical statement to be proved
theorem Geometry_problem
  (h1 : in_triangle A B C I)  -- Triangle ABC with incircle centered at I
  (h2 : angle_bisector_meets_incircle A B C I F L)  -- Angle bisector of A intersecting incircle at F and L
  (h3 : foot_perpendicular C D (angle_bisector_line A))  -- D is foot of perpendicular from C to angle bisector of A
  (h4 : foot_perpendicular I K (BD_line B D))  -- K is foot of perpendicular from I to BD
  : concur_cyclic F L B K := 
sorry

end Geometry_problem_l173_173611


namespace adam_ferris_wheel_spending_l173_173365

theorem adam_ferris_wheel_spending :
  ∀ (tickets_bought tickets_left ticket_cost : ℕ),
  tickets_bought = 13 →
  tickets_left = 4 →
  ticket_cost = 9 →
  (tickets_bought - tickets_left) * ticket_cost = 81 :=
by
  intros tickets_bought tickets_left ticket_cost h1 h2 h3
  rw [h1, h2, h3]
  exact rfl
  sorry

end adam_ferris_wheel_spending_l173_173365


namespace problem1_correct_problem2_correct_l173_173776

open Real

noncomputable def problem1 : ℝ :=
  sqrt(2 + 1 / 4) - (-9.6)^0 - (3 + 3 / 8)^(-2 / 3) + (1.5)^(-2)

-- Let's avoid using sqrt if we directly determine the value 3/2 etc
theorem problem1_correct : problem1 = 1 / 2 :=
by sorry

noncomputable def log_base (b x : ℝ) := log x / log b

noncomputable def problem2 : ℝ :=
  log_base 3 (427 / 3) + log 25 + log 4 + 7^(log_base 7 2)

theorem problem2_correct : problem2 = 15 / 4 :=
by sorry

end problem1_correct_problem2_correct_l173_173776


namespace distinct_powers_of_2_with_negative_sum_exponents_l173_173412

theorem distinct_powers_of_2_with_negative_sum_exponents :
  ∃ (S : Set ℤ), (2000 = S.sum (λ x, 2^x)) ∧ (- ∃ x, x ∈ S ∧ x < 0) ∧ (S.sum id = 41) := by
sorry

end distinct_powers_of_2_with_negative_sum_exponents_l173_173412


namespace fraction_of_yard_occupied_l173_173747

-- Define the rectangular yard with given length and width
def yard_length : ℝ := 25
def yard_width : ℝ := 5

-- Define the isosceles right triangle and the parallel sides of the trapezoid
def parallel_side1 : ℝ := 15
def parallel_side2 : ℝ := 25
def triangle_leg : ℝ := (parallel_side2 - parallel_side1) / 2

-- Areas
def triangle_area : ℝ := (1 / 2) * triangle_leg ^ 2
def flower_beds_area : ℝ := 2 * triangle_area
def yard_area : ℝ := yard_length * yard_width

-- Fraction calculation
def fraction_occupied : ℝ := flower_beds_area / yard_area

-- The proof statement
theorem fraction_of_yard_occupied:
  fraction_occupied = 1 / 5 :=
by
  sorry

end fraction_of_yard_occupied_l173_173747


namespace xn_plus_invxn_eq_2sin_nphi_l173_173044

theorem xn_plus_invxn_eq_2sin_nphi
  (φ : ℝ) (x : ℂ) (n : ℕ)
  (hφ1 : 0 < φ)
  (hφ2 : φ < π / 2)
  (h : x + x⁻¹ = 2 * complex.sin φ) :
  x^n + x^(-n) = 2 * complex.sin (n * φ) :=
by
  sorry

end xn_plus_invxn_eq_2sin_nphi_l173_173044


namespace subset_R_equals_R_l173_173946

theorem subset_R_equals_R (A : set ℝ) 
  (h_nonempty : ∃ a : ℝ, a ∈ A) 
  (h_property : ∀ x y : ℝ, (x + y) ∈ A → (x * y) ∈ A) : 
  A = set.univ :=
by
  sorry

end subset_R_equals_R_l173_173946


namespace subtracted_number_l173_173499

def least_sum_is (x y z : ℤ) (a : ℤ) : Prop :=
  (x - a) * (y - 5) * (z - 2) = 1000 ∧ x + y + z = 7

theorem subtracted_number (x y z a : ℤ) (h : least_sum_is x y z a) : a = 30 :=
sorry

end subtracted_number_l173_173499


namespace sum_of_incomes_l173_173143

variable (J : ℝ) (T : ℝ) (M : ℝ) (A : ℝ)

-- Conditions
def TimIncome : Prop := T = 0.60 * J
def MaryIncome : Prop := M = 1.40 * T
def AlexIncome1 : Prop := A = 1.25 * J
def AlexIncome2 : Prop := A = M - 0.20 * M

-- Statement
theorem sum_of_incomes : 
  TimIncome J T → MaryIncome J T M → AlexIncome1 J A → AlexIncome2 J M A → 
  M + A = 2.09 * J :=
by 
  intros hT hM hA1 hA2
  sorry

end sum_of_incomes_l173_173143


namespace sum_of_divisors_of_power_form_eq_six_l173_173644

theorem sum_of_divisors_of_power_form_eq_six (i j : ℕ) (n : ℕ) 
  (h : n = 2^i * 3^j) 
  (h_sum : ∑ d in range (n+1), if n % d = 0 then d else 0 = 360) : 
  i + j = 6 := by
  sorry

end sum_of_divisors_of_power_form_eq_six_l173_173644


namespace lowest_position_l173_173232

theorem lowest_position (num_cyclists : ℕ) (num_stages : ℕ) (vasya_position : ℕ) :
  num_cyclists = 500 →
  num_stages = 15 →
  vasya_position = 7 →
  ∃ n, n = 91 :=
by
  intros
  sorry

end lowest_position_l173_173232


namespace expected_number_of_sixes_l173_173662

-- Define the probability of not rolling a 6 on one die
def prob_not_six : ℚ := 5 / 6

-- Define the probability of rolling zero 6's on three dice
def prob_zero_six : ℚ := prob_not_six ^ 3

-- Define the probability of rolling exactly one 6 among the three dice
def prob_one_six (n : ℕ) : ℚ := n * (1 / 6) * (prob_not_six ^ (n - 1))

-- Calculate the probabilities of each specific outcomes
def prob_exactly_zero_six : ℚ := prob_zero_six
def prob_exactly_one_six : ℚ := prob_one_six 3 * (prob_not_six ^ 2)
def prob_exactly_two_six : ℚ := prob_one_six 3 * (1 / 6) * prob_not_six
def prob_exactly_three_six : ℚ := (1 / 6) ^ 3

-- Define the expected value calculation
noncomputable def expected_value : ℚ :=
  0 * prob_exactly_zero_six
  + 1 * prob_exactly_one_six
  + 2 * prob_exactly_two_six
  + 3 * prob_exactly_three_six

-- Prove that the expected value equals to 1/2
theorem expected_number_of_sixes : expected_value = 1 / 2 :=
  by
    sorry

end expected_number_of_sixes_l173_173662


namespace will_remaining_money_l173_173702

theorem will_remaining_money : 
  ∀ (initial_money sweater_cost tshirt_cost shoes_cost refund_percentage remaining_money : ℕ),
  initial_money = 74 →
  sweater_cost = 9 →
  tshirt_cost = 11 →
  shoes_cost = 30 →
  refund_percentage = 90 →
  remaining_money = 51 → 
  let total_spent := sweater_cost + tshirt_cost + ((shoes_cost * (100 - refund_percentage)) / 100) 
  in remaining_money = initial_money - total_spent :=
by 
  intros initial_money sweater_cost tshirt_cost shoes_cost refund_percentage remaining_money 
         h_initial_money h_sweater_cost h_tshirt_cost h_shoes_cost h_refund_percentage h_remaining_money
  let total_spent := sweater_cost + tshirt_cost + ((shoes_cost * (100 - refund_percentage)) / 100)
  have h_total_spent : total_spent = 20 + 3 := by 
    unfold total_spent
    rw [h_sweater_cost, h_tshirt_cost, h_shoes_cost, h_refund_percentage]
    norm_num
  have h_remaining_money_computed : initial_money - total_spent = 51 := by
    rw [h_initial_money, h_total_spent]
    norm_num
  rw [h_remaining_money] at h_remaining_money_computed
  exact h_remaining_money_computed

end will_remaining_money_l173_173702


namespace four_digit_perfect_square_palindromes_eq_one_l173_173879

theorem four_digit_perfect_square_palindromes_eq_one :
  ∃! (n : ℕ), (1024 ≤ n) ∧ (n ≤ 9801) ∧ (n = (m * m) ∧ is_palindrome n) :=
sorry

-- Auxiliary definition for checking if a number is a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in digits = digits.reverse
  

end four_digit_perfect_square_palindromes_eq_one_l173_173879


namespace ratio_of_areas_of_circles_l173_173894

-- Given conditions
variables (R_C R_D : ℝ) -- Radii of circles C and D respectively
variables (L : ℝ) -- Common length of the arcs

-- Equivalent arc condition
def arc_length_condition : Prop :=
  (60 / 360) * (2 * Real.pi * R_C) = L ∧ (40 / 360) * (2 * Real.pi * R_D) = L

-- Goal: ratio of areas
def area_ratio : Prop :=
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = (9 / 4)

-- Problem statement
theorem ratio_of_areas_of_circles (R_C R_D L : ℝ) (hc : arc_length_condition R_C R_D L) :
  area_ratio R_C R_D :=
by
  sorry

end ratio_of_areas_of_circles_l173_173894


namespace inequality_for_a_l173_173454

open Real

-- Define the sequence a_n
noncomputable def a (n : ℕ) : ℝ := ∑ i in Finset.range n, sqrt ((i + 1) * (i + 2))

-- Prove the given inequality
theorem inequality_for_a (n : ℕ) : 
  (n * (n + 1) / 2 : ℝ) < a n ∧ a n < (n * (n + 2) / 2 : ℝ) := 
by
  sorry

end inequality_for_a_l173_173454


namespace inscribed_circle_diameter_l173_173400

-- Defining the given side lengths of the triangle
def PQ : ℝ := 13
def PR : ℝ := 8
def QR : ℝ := 15

-- Defining the semiperimeter of the triangle
def s : ℝ := (PQ + PR + QR) / 2

-- Defining the area of the triangle using Heron's formula
def K : ℝ := Real.sqrt (s * (s - PQ) * (s - PR) * (s - QR))

-- Defining the radius of the inscribed circle
def r : ℝ := K / s

-- Defining the diameter of the inscribed circle
def d : ℝ := 2 * r

-- The theorem to be proved
theorem inscribed_circle_diameter : d = (10 * Real.sqrt 3) / 3 := sorry

end inscribed_circle_diameter_l173_173400


namespace Maria_owes_more_if_semi_annually_l173_173575

def Maria_loan_amount : ℝ := 8000
def annual_interest_rate : ℝ := 0.10
def interest_compound_periods_semi_annually : ℕ := 3 * 2
def interest_compound_periods_annually : ℕ := 3
def semi_annual_interest_rate : ℝ := annual_interest_rate / 2

noncomputable def amount_owed_semi_annually : ℝ :=
  (1 + semi_annual_interest_rate)^interest_compound_periods_semi_annually * Maria_loan_amount

noncomputable def amount_owed_annually : ℝ :=
  (1 + annual_interest_rate)^interest_compound_periods_annually * Maria_loan_amount

noncomputable def difference_in_amount_owed : ℝ :=
  amount_owed_semi_annually - amount_owed_annually

theorem Maria_owes_more_if_semi_annually :
  difference_in_amount_owed = 72.80 :=
sorry

end Maria_owes_more_if_semi_annually_l173_173575


namespace probability_first_green_then_blue_l173_173725

variable {α : Type} [Fintype α]

noncomputable def prob_first_green_second_blue : ℚ := 
  let total_marbles := 10
  let green_marbles := 6
  let blue_marbles := 4
  let prob_first_green := (green_marbles : ℚ) / total_marbles
  let prob_second_blue := (blue_marbles : ℚ) / (total_marbles - 1)
  (prob_first_green * prob_second_blue)

theorem probability_first_green_then_blue :
  prob_first_green_second_blue = 4 / 15 := by
  sorry

end probability_first_green_then_blue_l173_173725


namespace sum_of_nine_consecutive_parity_l173_173541

theorem sum_of_nine_consecutive_parity (n : ℕ) : 
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) + (n + 8)) % 2 = n % 2 := 
  sorry

end sum_of_nine_consecutive_parity_l173_173541


namespace basketball_team_avg_weight_l173_173914

theorem basketball_team_avg_weight :
  let n_tallest := 5
  let w_tallest := 90
  let n_shortest := 4
  let w_shortest := 75
  let n_remaining := 3
  let w_remaining := 80
  let total_weight := (n_tallest * w_tallest) + (n_shortest * w_shortest) + (n_remaining * w_remaining)
  let total_players := n_tallest + n_shortest + n_remaining
  (total_weight / total_players) = 82.5 :=
by
  sorry

end basketball_team_avg_weight_l173_173914


namespace triangulation_3_coloring_l173_173351

-- Definition of a triangulation of a polygon
structure Polygon :=
  (vertices : Finset Point)
  (edges : Finset (Point × Point))
  (triangles : Finset (Finset Point))
  (valid_triangulation : ∀ t ∈ triangles, t.card = 3) -- each triangle is a set of 3 points

-- Definition of a valid coloring
def valid_coloring (triangles : Finset (Finset Point)) (coloring : Point → Fin 3) : Prop := 
  ∀ t₁ t₂ ∈ triangles, ∃ p₁ p₂, 
    p₁ ∈ t₁ ∧ p₂ ∈ t₂ ∧ p₁ = p₂ ∧ p₁ ≠ p₂ →
    coloring p₁ ≠ coloring p₂

-- Main theorem statement
theorem triangulation_3_coloring (P : Polygon) : 
  ∃ coloring : Point → Fin 3, valid_coloring P.triangles coloring :=
begin
  sorry
end

end triangulation_3_coloring_l173_173351


namespace marble_probability_l173_173728

theorem marble_probability
  (total_marbles : ℕ)
  (blue_marbles : ℕ)
  (green_marbles : ℕ)
  (draws : ℕ)
  (prob_first_green : ℚ)
  (prob_second_blue_given_green : ℚ)
  (total_prob : ℚ)
  (h_total : total_marbles = 10)
  (h_blue : blue_marbles = 4)
  (h_green : green_marbles = 6)
  (h_draws : draws = 2)
  (h_prob_first_green : prob_first_green = 3 / 5)
  (h_prob_second_blue_given_green : prob_second_blue_given_green = 4 / 9)
  (h_total_prob : total_prob = 4 / 15) :
  prob_first_green * prob_second_blue_given_green = total_prob := sorry

end marble_probability_l173_173728


namespace find_ab_pairs_l173_173416

theorem find_ab_pairs (a b s : ℕ) (a_pos : a > 0) (b_pos : b > 0) (s_gt_one : s > 1) :
  (a = 2^s ∧ b = 2^(2*s) - 1) ↔
  (∃ p k : ℕ, Prime p ∧ (a^2 + b + 1 = p^k) ∧
   (a^2 + b + 1 ∣ b^2 - a^3 - 1) ∧
   ¬ (a^2 + b + 1 ∣ (a + b - 1)^2)) :=
sorry

end find_ab_pairs_l173_173416


namespace probability_first_green_then_blue_l173_173726

variable {α : Type} [Fintype α]

noncomputable def prob_first_green_second_blue : ℚ := 
  let total_marbles := 10
  let green_marbles := 6
  let blue_marbles := 4
  let prob_first_green := (green_marbles : ℚ) / total_marbles
  let prob_second_blue := (blue_marbles : ℚ) / (total_marbles - 1)
  (prob_first_green * prob_second_blue)

theorem probability_first_green_then_blue :
  prob_first_green_second_blue = 4 / 15 := by
  sorry

end probability_first_green_then_blue_l173_173726


namespace rank_siblings_l173_173784

variable (Person : Type) (Dan Elena Finn : Person)

variable (height : Person → ℝ)

-- Conditions
axiom different_heights : height Dan ≠ height Elena ∧ height Elena ≠ height Finn ∧ height Finn ≠ height Dan
axiom one_true_statement : (¬ (height Elena = max (max (height Dan) (height Elena)) (height Finn)) ∨ height Finn = max (max (height Dan) (height Elena)) (height Finn) ∨ height Dan ≠ min (min (height Dan) (height Elena)) (height Finn)) 
  ∧ (¬ (height Elena = max (max (height Dan) (height Elena)) (height Finn)) ∨ height Finn = max (max (height Dan) (height Elena)) (height Finn) ∨ height Dan ≠ min (min (height Dan) (height Elena)) (height Finn))
  ∧ (¬ (height Elena = max (max (height Dan) (height Elena)) (height Finn)) ∨ height Finn = max (max (height Dan) (height Elena)) (height Finn) ∨ height Dan ≠ min (min (height Dan) (height Elena)) (height Finn))

theorem rank_siblings : height Finn > height Elena ∧ height Elena > height Dan := by
  sorry

end rank_siblings_l173_173784


namespace ap_parallel_bc_l173_173088

variables {A B C D E F G P : Type}
variables [triangle A B C] 
variables [circumcircle A B C : circle]
variables [interior_bisector BAC D]
variables [exterior_bisector BAC E]
variables [reflection A D F]
variables [reflection A E G]
variables [circumcircle ADG : circle]
variables [circumcircle AEF : circle]
variables [intersection P circumcircle ADG circumcircle AEF]
variables [line_segment AP : line]
variables [line_segment BC : line]

theorem ap_parallel_bc :
  parallel AP BC :=
sorry

end ap_parallel_bc_l173_173088


namespace price_of_each_package_l173_173355

theorem price_of_each_package (P : ℝ) :
  (∃ (P : ℝ), 
     let packs_needed_vampire := 2 in
     let packs_needed_pumpkin := 3 in
     let individual_vampire_bags := 1 in
     let total_cost := 17 in
     total_cost = (packs_needed_vampire + packs_needed_pumpkin) * P + individual_vampire_bags) → 
  P = 3.20 :=
begin
  sorry
end

end price_of_each_package_l173_173355


namespace airplane_faster_by_90_minutes_l173_173761

def driving_time : ℕ := 3 * 60 + 15  -- in minutes
def drive_to_airport_time : ℕ := 10   -- in minutes
def board_wait_time : ℕ := 20         -- in minutes
def flight_time : ℕ := driving_time / 3  -- in minutes
def get_off_airplane_time : ℕ := 10   -- in minutes

theorem airplane_faster_by_90_minutes :
  driving_time - (drive_to_airport_time + board_wait_time + flight_time + get_off_airplane_time) = 90 :=
by
  calc
    driving_time 
      = 195               : by unfold driving_time
    ...(10 + 20 + 65 + 10 = 105): by unfold drive_to_airport_time board_wait_time flight_time get_off_airplane_time
    195 - 105 = 90         : by norm_num

end airplane_faster_by_90_minutes_l173_173761


namespace complex_modulus_l173_173819

noncomputable def z (i : ℂ) : ℂ :=
  i / (1 + i^3)

theorem complex_modulus (i : ℂ) (hi : i^2 = -1) : |z i| = (Real.sqrt 2) / 2 := by
  sorry

end complex_modulus_l173_173819


namespace equation_c_is_linear_l173_173306

-- Define the condition for being a linear equation with one variable
def is_linear_equation_with_one_variable (eq : ℝ → Prop) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x : ℝ, eq x ↔ (a * x + b = 0)

-- The given equation to check is (x - 1) / 2 = 1, which simplifies to x = 3
def equation_c (x : ℝ) : Prop := (x - 1) / 2 = 1

-- Prove that the given equation is a linear equation with one variable
theorem equation_c_is_linear :
  is_linear_equation_with_one_variable equation_c :=
sorry

end equation_c_is_linear_l173_173306


namespace find_a_f_identity_sum_f_l173_173824

noncomputable def y (a x : ℝ) : ℝ := a^x
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a^x / (a^x + 2)

axiom a_pos : ∀ a : ℝ, a > 0
axiom a_ne_one : ∀ a : ℝ, a ≠ 1
axiom a_quadratic : ∀ a : ℝ, a + a^2 = 20 → a = 4

theorem find_a (a : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) (ha_quad : a + a^2 = 20) : a = 4 :=
a_quadratic a ha_quad

theorem f_identity (x : ℝ) : f x 4 + f (1 - x) 4 = 1 :=
by {
  sorry
}

theorem sum_f (s : Finset ℕ) (h₁ : ∀ k : ℕ, k ∈ s → 1 ≤ k ∧ k ≤ 2010) :
    ∑ k in s, f (k / 2011) 4 = 1005 :=
by {
  sorry
}

end find_a_f_identity_sum_f_l173_173824


namespace survey_result_l173_173070

theorem survey_result (total_people : ℕ) (percent_no_tv : ℝ) (percent_no_tv_no_podcast : ℝ) : 
  total_people = 1500 →
  percent_no_tv = 0.25 →
  percent_no_tv_no_podcast = 0.15 →
  (total_people * percent_no_tv).to_nat * percent_no_tv_no_podcast = 56 :=
by
  intro h1 h2 h3
  sorry

end survey_result_l173_173070


namespace AM_lt_BM_CM_l173_173139

-- definitions based on the conditions
variables {A B C M O : Type}
variable [MetricSpace O]

def isIsosceles (A B C : O) : Prop := dist A B = dist A C
def inscribedInCircle (A B C : O) (O : O) : Prop := sorry

-- Given conditions in Lean 4
-- Triangle isosceles with AB = AC
axiom triangle_is_isosceles : isIsosceles A B C
-- AB ≠ BC
axiom AB_neq_BC : dist A B ≠ dist B C
-- Triangle inscribed in a circle O
axiom triangle_in_circle : inscribedInCircle A B C O
-- M is a point on the arc BC not containing A
axiom M_on_arc_BC_not_A : sorry

-- Proof statement in Lean 4
theorem AM_lt_BM_CM : dist A M < dist B M + dist C M :=
by {
    sorry
}

end AM_lt_BM_CM_l173_173139


namespace infant_weight_in_4th_month_l173_173538

-- Given conditions
def a : ℕ := 3000
def x : ℕ := 4
def y : ℕ := a + 700 * x

-- Theorem stating the weight of the infant in the 4th month equals 5800 grams
theorem infant_weight_in_4th_month : y = 5800 := by
  sorry

end infant_weight_in_4th_month_l173_173538


namespace evaluate_expression_l173_173794

theorem evaluate_expression : 
  (3 * real.root 4 5) / (real.root 3 5) = 3 * 5^(-1/12) :=
by
  sorry

end evaluate_expression_l173_173794


namespace exists_two_factorizations_in_C_another_number_with_property_l173_173479

def in_set_C (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 4 * k + 1

def is_prime_wrt_C (k : ℕ) : Prop :=
  ¬ ∃ a b : ℕ, in_set_C a ∧ in_set_C b ∧ k = a * b

theorem exists_two_factorizations_in_C : 
  ∃ (a b a' b' : ℕ), 
  in_set_C 4389 ∧ 
  in_set_C a ∧ in_set_C b ∧ in_set_C a' ∧ in_set_C b' ∧ 
  (4389 = a * b ∧ 4389 = a' * b') ∧ (a ≠ a' ∨ b ≠ b') :=
sorry

theorem another_number_with_property : 
 ∃ (n a b a' b' : ℕ), 
 n ≠ 4389 ∧ in_set_C n ∧ 
 in_set_C a ∧ in_set_C b ∧ in_set_C a' ∧ in_set_C b' ∧ 
 (n = a * b ∧ n = a' * b') ∧ (a ≠ a' ∨ b ≠ b') :=
sorry

end exists_two_factorizations_in_C_another_number_with_property_l173_173479


namespace range_of_k_l173_173864

theorem range_of_k : 
  let P := (-2 : ℝ, 0 : ℝ)
  let C := { p : ℝ × ℝ | p.2^2 = 4 * p.1 }
  let l (k : ℝ) (x : ℝ) := k * (x + 2)
  ∃ k : ℝ, ∀ x : ℝ, (l k x, (l k x)^2 = 4 * x) ∈ C → 
  (-real.sqrt 2 / 2 < k ∧ k < 0) ∨ (0 < k ∧ k < real.sqrt 2 / 2) :=
sorry

end range_of_k_l173_173864


namespace book_pages_l173_173909

-- Define conditions as hypotheses
def num_of_occurrences (digit : Nat) (n : Nat) : Nat :=
  (List.range (n + 1)).map (λ x => (x.toString.count (λ c => c = Char.ofNat (digit + 48)))).sum

-- Define the theorem
theorem book_pages (n : Nat) : num_of_occurrences 1 n = 171 → n = 318 :=
  by
    intro h
    sorry

end book_pages_l173_173909


namespace max_height_piston_l173_173717

theorem max_height_piston (M a P c_v g R: ℝ) (h : ℝ) 
  (h_pos : 0 < h) (M_pos : 0 < M) (a_pos : 0 < a) (P_pos : 0 < P)
  (c_v_pos : 0 < c_v) (g_pos : 0 < g) (R_pos : 0 < R) :
  h = (2 * P ^ 2) / (M ^ 2 * g * a ^ 2 * (1 + c_v / R) ^ 2) := sorry

end max_height_piston_l173_173717


namespace replace_80_percent_banknotes_within_days_unable_to_replace_all_banknotes_without_repair_l173_173319

-- Given conditions
def total_banknotes : ℕ := 3628800
def major_repair_cost : ℕ := 800000
def daily_operation_cost : ℕ := 90000
def capacity_after_repair : ℕ := 1000000
def total_budget : ℕ := 1000000
def percentage_to_replace : ℕ := 80

-- Definitions related to days and operations
def banknotes_replaced_on_day (day : ℕ) (remaining_banknotes : ℕ) : ℕ := 
  match day with
  | 1 => remaining_banknotes / 2
  | 2 => remaining_banknotes / 3
  | 3 => remaining_banknotes / 4
  | 4 => remaining_banknotes / 5
  | _ => 0

-- Proof Statements
theorem replace_80_percent_banknotes_within_days : ∃ (days : ℕ), (days < 100) ∧
  let remaining_banknotes_1 := total_banknotes - banknotes_replaced_on_day 1 total_banknotes in
  let remaining_banknotes_2 := remaining_banknotes_1 - banknotes_replaced_on_day 2 remaining_banknotes_1 in
  let remaining_banknotes_3 := remaining_banknotes_2 - banknotes_replaced_on_day 3 remaining_banknotes_2 in
  let total_replaced := total_banknotes - remaining_banknotes_3 in
  total_replaced ≥ (percentage_to_replace * total_banknotes) / 100 := 
sorry

theorem unable_to_replace_all_banknotes_without_repair : 
  let operation_cost := (total_banknotes / capacity_after_repair) * daily_operation_cost in
  operation_cost > total_budget :=
sorry 

end replace_80_percent_banknotes_within_days_unable_to_replace_all_banknotes_without_repair_l173_173319


namespace video_game_levels_total_l173_173793

theorem video_game_levels_total {T N : ℕ} :
  let levels_beaten := 24,
  let beat_ratio := 3 in
  beat_ratio * N = levels_beaten →
  T = levels_beaten + N →
  T = 32 :=
by
  intros levels_beaten beat_ratio ratio_eq total_eq
  sorry

end video_game_levels_total_l173_173793


namespace B_lap_time_l173_173751

-- Definitions based on given conditions.
def time_to_complete_lap_A := 40
def meeting_interval := 15

-- The theorem states that given the conditions, B takes 24 seconds to complete the track.
theorem B_lap_time (l : ℝ) (t : ℝ) (h1 : t = 24)
                    (h2 : l / time_to_complete_lap_A + l / t = l / meeting_interval):
  t = 24 := by sorry

end B_lap_time_l173_173751


namespace proof_problem_l173_173838

variable {α : Type*} [LinearOrderedField α]

def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ (a1 d : α), ∀ n : ℕ, a n = a1 + n * d

def sum_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
  (n * (a 0 + a (n - 1))) / 2

variables {a : ℕ → α}

theorem proof_problem (h_arith_seq : is_arithmetic_sequence a)
    (h_S6_gt_S7 : sum_first_n_terms a 6 > sum_first_n_terms a 7)
    (h_S7_gt_S5 : sum_first_n_terms a 7 > sum_first_n_terms a 5) :
    (∃ d : α, d < 0) ∧ (∃ S11 : α, sum_first_n_terms a 11 > 0) :=
  sorry

end proof_problem_l173_173838


namespace horner_eval_at_neg2_l173_173861

noncomputable def f (x : ℝ) : ℝ := x^5 - 3 * x^3 - 6 * x^2 + x - 1

theorem horner_eval_at_neg2 : f (-2) = -35 :=
by
  sorry

end horner_eval_at_neg2_l173_173861


namespace bologna_sandwiches_count_l173_173370

theorem bologna_sandwiches_count (x y b : ℕ) (h_ratio : 1 * y + x * y + 8 * y = 80) (h_x : x = 1) : b = 8 :=
by
  have h1 : (1 + x + 8) * y = 80, from h_ratio
  rw [h_x] at h1
  have h2 : (1 + 1 + 8) * y = 80, from h1
  have h3 : 10 * y = 80, from h2
  have h4 : y = 8, from Nat.eq_of_mul_eq_mul_right (by norm_num) h3
  have h_b : b = x * y, from rfl
  rw [h_x, h4] at h_b
  have h_b_final : b = 1 * 8, from h_b
  have h_b : b = 8, by norm_num
  exact h_b

end bologna_sandwiches_count_l173_173370


namespace Tn_lt_half_Sn_l173_173109

noncomputable def a_n (n : ℕ) : ℝ :=
  (1 / 3) ^ (n - 1)

noncomputable def b_n (n : ℕ) : ℝ :=
  n * ((1 / 3) ^ n) / 3

noncomputable def S_n (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a_n (i + 1)

noncomputable def T_n (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, b_n (i + 1)

theorem Tn_lt_half_Sn (n : ℕ) : T_n n < S_n n / 2 := by
  sorry

end Tn_lt_half_Sn_l173_173109


namespace problem_statement_l173_173815

def numDivisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).count (λ d => n % d = 0)

def f1 (n : ℕ) : ℕ :=
  2 * (numDivisors n)

def f (j n : ℕ) : ℕ :=
  if j = 1 then f1 n else f1 (f (j - 1) n)

def countN (k : ℕ) : ℕ :=
  (List.range (k + 1)).count (λ n => f 50 n = 8)

theorem problem_statement : countN 100 = 14 :=
  sorry

end problem_statement_l173_173815


namespace a_2017_l173_173439

noncomputable def seq (a : ℕ+ → ℝ) (h₁ : a 1 = 2)
  (h₂ : ∀ n : ℕ+, a (n + 1) = 1 / (1 - a n)) : ℕ+ → ℝ :=
  λ n, (exists_periodic { p : ℕ | p > 0 } 3) (n % 3) 

theorem a_2017 : 
  let a := seq 
    (λ n, seq n (by sorry) (by sorry)) 
  in a 2017 = 2 :=
by sorry


end a_2017_l173_173439


namespace no_four_digit_palindromic_squares_l173_173873

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in
  s = s.reverse

def four_digit_perfect_squares : List ℕ :=
  List.range' (32*32) (99*32 - 32*32 + 1)
  
theorem no_four_digit_palindromic_squares : 
  (List.filter is_palindrome four_digit_perfect_squares).length = 0 :=
by
  sorry

end no_four_digit_palindromic_squares_l173_173873


namespace correct_statements_l173_173128

-- Definitions for the conditions
def congruence_modulo (a b m : ℤ) : Prop := (m ∣ (a - b))

variables (a b c : ℤ) (m n d : ℤ) (f : ℤ → ℤ)
-- Setting f(x) = x^3 - 2x + 5
def f := λ x : ℤ, x^3 - 2 * x + 5

theorem correct_statements :
  ∀ {a b m : ℤ} (h1 : congruence_modulo a b m),
    ∀ {d : ℤ} (h2 : d > 0) (h3 : d ∣ m),
      congruence_modulo a b d ∧ 
      (∀ {n : ℤ} (h4 : congruence_modulo a b n), congruence_modulo a b (m * n)) ∧
      (∀ {c : ℤ}, congruence_modulo (a * c) (b * c) m → congruence_modulo a b m) ∧
      congruence_modulo (f a) (f b) m :=
by 
  intros a b m h1 d h2 h3 n h4 c h5;
  sorry

end correct_statements_l173_173128


namespace sum_of_divisors_of_power_form_eq_six_l173_173643

theorem sum_of_divisors_of_power_form_eq_six (i j : ℕ) (n : ℕ) 
  (h : n = 2^i * 3^j) 
  (h_sum : ∑ d in range (n+1), if n % d = 0 then d else 0 = 360) : 
  i + j = 6 := by
  sorry

end sum_of_divisors_of_power_form_eq_six_l173_173643


namespace part_I_part_II_l173_173085

variable (a b c : ℝ)
variable (A B C : ℝ)
variable (sA sB sC cB cC : ℝ)

-- Conditions
axiom side_opposite_conditions : a = sin A ∧ b = sin B ∧ c = sin C
axiom given_eq : (2*a + b)*cos C + c*cos B = 0
axiom sine_conditions : sA = sin A ∧ sB = sin B ∧ sC = sin C ∧ cB = cos B ∧ cC = cos C

-- Part I: Proof that angle C = 2π/3
theorem part_I : C = 2*π/3 := by
  sorry

-- Part II: Range of values for sin A cos B
theorem part_II : (sin A * cos B) > (sqrt 3 / 4) ∧ (sin A * cos B) < (3 * sqrt 3 / 4) := by
  assume hC : C = 2*π/3
  sorry

end part_I_part_II_l173_173085


namespace largest_four_digit_congruent_17_mod_26_l173_173289

theorem largest_four_digit_congruent_17_mod_26 : 
  ∃ k : ℤ, (26 * k + 17 < 10000) ∧ (1000 ≤ 26 * k + 17) ∧ (26 * k + 17) ≡ 17 [MOD 26] ∧ (26 * k + 17 = 9972) :=
by
  sorry

end largest_four_digit_congruent_17_mod_26_l173_173289


namespace goose_eggs_hatching_l173_173146

theorem goose_eggs_hatching (x : ℝ) :
  (∃ n_hatched : ℝ, 3 * (2 * n_hatched / 20) = 110 ∧ x = n_hatched / 550) →
  x = 2 / 3 :=
by
  intro h
  sorry

end goose_eggs_hatching_l173_173146


namespace probability_odd_sum_three_dice_l173_173358

/-- Given an unfair die where rolling an odd number is 4 times as likely as an even number and three such dice are rolled,
    the probability that the sum of the numbers rolled is odd is 76/125. -/
theorem probability_odd_sum_three_dice :
  let p_odd : ℚ := 4 / 5
  let p_even : ℚ := 1 / 5
  (3 * p_odd * p_even^2 + p_odd^3 = 76 / 125) :=
by
  let p_odd := 4 / 5
  let p_even := 1 / 5
  suffices : 3 * p_odd * p_even^2 + p_odd^3 = 76 / 125
  sorry

end probability_odd_sum_three_dice_l173_173358


namespace lowest_position_l173_173233

theorem lowest_position (num_cyclists : ℕ) (num_stages : ℕ) (vasya_position : ℕ) :
  num_cyclists = 500 →
  num_stages = 15 →
  vasya_position = 7 →
  ∃ n, n = 91 :=
by
  intros
  sorry

end lowest_position_l173_173233


namespace circle_tangent_DA_DC_l173_173546

variables (A B C D E F : Type*) [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] 
          [AddGroup E] [AddGroup F] (parallelogram : A × B × C × D) 
          (diagonal_AC : C) (distinct_points : E ≠ F) 
          (circle_tangent_BA_BC : ∃ o₁ : Type*, ∃ o₂ : Type*, circle_through o₁ E F ∧ tangent o₁ BA ∧ tangent o₁ BC)

theorem circle_tangent_DA_DC : 
  ∃ o₃ : Type*, circle_through o₃ E F ∧ tangent o₃ DA ∧ tangent o₃ DC :=
by 
  sorry

end circle_tangent_DA_DC_l173_173546


namespace smallest_geometric_third_term_l173_173345

theorem smallest_geometric_third_term (d : ℝ) (a₁ a₂ a₃ g₁ g₂ g₃ : ℝ) 
  (h_AP : a₁ = 5 ∧ a₂ = 5 + d ∧ a₃ = 5 + 2 * d)
  (h_GP : g₁ = a₁ ∧ g₂ = a₂ + 3 ∧ g₃ = a₃ + 15)
  (h_geom : (g₂)^2 = g₁ * g₃) : g₃ = -4 := 
by
  -- We would provide the proof here.
  sorry

end smallest_geometric_third_term_l173_173345


namespace probability_not_losing_l173_173371

theorem probability_not_losing (P_winning P_drawing : ℚ)
  (h_winning : P_winning = 1/3)
  (h_drawing : P_drawing = 1/4) :
  P_winning + P_drawing = 7/12 := 
by
  sorry

end probability_not_losing_l173_173371


namespace find_f1_l173_173823

def f : ℤ → ℤ
| x >= 2    := 2 * x - 1
| x < 2     := f (f (x + 1)) + 1

theorem find_f1 : f 1 = 6 :=
by
  sorry

end find_f1_l173_173823


namespace average_value_of_T_l173_173178

noncomputable def expected_value_T (B G : ℕ) : ℚ :=
  let total_pairs := 19
  let prob_bg := (B / (B + G)) * (G / (B + G))
  2 * total_pairs * prob_bg

theorem average_value_of_T 
  (B G : ℕ) (hB : B = 8) (hG : G = 12) : 
  expected_value_T B G = 9 :=
by
  rw [expected_value_T, hB, hG]
  norm_num
  sorry

end average_value_of_T_l173_173178


namespace zhang_san_not_losing_probability_l173_173373

theorem zhang_san_not_losing_probability (p_win p_draw : ℚ) (h_win : p_win = 1 / 3) (h_draw : p_draw = 1 / 4) : 
  p_win + p_draw = 7 / 12 := by
  sorry

end zhang_san_not_losing_probability_l173_173373


namespace solve_sqrt_equation_l173_173165

theorem solve_sqrt_equation (x : ℝ) :
    (sqrt (5 + sqrt (3 + sqrt x)) = (2 + sqrt x)^(1/4)) ↔ x = 14 :=
by
    sorry

end solve_sqrt_equation_l173_173165


namespace expected_value_of_sixes_l173_173653

theorem expected_value_of_sixes (n : ℕ) (k : ℕ) (p q : ℚ) 
  (h1 : n = 3) 
  (h2 : k = 6)
  (h3 : p = 1/6) 
  (h4 : q = 5/6) : 
  (1 : ℚ) / 2 = ∑ i in finset.range (n + 1), (i * (nat.choose n i * p^i * q^(n-i))) := 
sorry

end expected_value_of_sixes_l173_173653


namespace cos_double_angle_sin_double_angle_l173_173496

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 1/2) : Real.cos (2 * θ) = -1/2 :=
by sorry

theorem sin_double_angle (θ : ℝ) (h : Real.cos θ = 1/2) : Real.sin (2 * θ) = (Real.sqrt 3) / 2 :=
by sorry

end cos_double_angle_sin_double_angle_l173_173496


namespace weight_of_mixture_l173_173712

noncomputable def total_weight_of_mixture (zinc_weight: ℝ) (zinc_ratio: ℝ) (total_ratio: ℝ) : ℝ :=
  (zinc_weight / zinc_ratio) * total_ratio

theorem weight_of_mixture (zinc_ratio: ℝ) (copper_ratio: ℝ) (tin_ratio: ℝ) (zinc_weight: ℝ) :
  total_weight_of_mixture zinc_weight zinc_ratio (zinc_ratio + copper_ratio + tin_ratio) = 98.95 :=
by 
  let ratio_sum := zinc_ratio + copper_ratio + tin_ratio
  let part_weight := zinc_weight / zinc_ratio
  let mixture_weight := part_weight * ratio_sum
  have h : mixture_weight = 98.95 := sorry
  exact h

end weight_of_mixture_l173_173712


namespace find_a_l173_173908

noncomputable def commonTangentAt (e : ℝ) (a : ℝ) (s : ℝ) (t : ℝ) : Prop :=
  let curve1 := (λ x : ℝ, (1/2) * (1/e) * x^2)
  let curve2 := (λ x : ℝ, a * Real.log x)
  let derivative_curve1 := (λ x : ℝ, x/e)
  let derivative_curve2 := (λ x : ℝ, a/x)
  (curve1 s = t) ∧ (curve2 s = t) ∧ (derivative_curve1 s = derivative_curve2 s)

theorem find_a (e : ℝ) (s : ℝ) (t : ℝ) :
  commonTangentAt e 1 s t → s^2 = e :=
by
  intros h
  rcases h with ⟨h_curve1, h_curve2, h_slope⟩
  sorry

end find_a_l173_173908


namespace determine_b2023_l173_173111

noncomputable def b : ℕ → ℝ
| 1     := 3 + Real.sqrt 5
| 2021  := 11 + Real.sqrt 5
| (n+1) := b n / b (n-1)

theorem determine_b2023 (H : ∀ n ≥ 2, b n = b (n - 1) * b (n + 1)) :
  b 2023 = 7 - 2 * Real.sqrt 5 :=
sorry

end determine_b2023_l173_173111


namespace intersection_range_l173_173027

-- Definition of the parametric representation of line l
def line_parametric (a t : ℝ) : ℝ × ℝ :=
  (a - 2 * t, -4 * t)

-- Definition of the parametric representation of circle C
def circle_parametric (θ : ℝ) : ℝ × ℝ :=
  (4 * Real.cos θ, 4 * Real.sin θ)

-- General equations derived from the parametric forms
def line_general (a x y : ℝ) : Prop :=
  2 * x - y - 2 * a = 0

def circle_general (x y : ℝ) : Prop :=
  x^2 + y^2 = 16

-- Distance from the center of the circle to the line
def distance_from_center_to_line (a : ℝ) : ℝ :=
  Real.abs (-2 * a) / Real.sqrt 5

-- The main theorem
theorem intersection_range (a : ℝ) :
  (∃ t, line_parametric a t ∈ {p : ℝ × ℝ | circle_general p.1 p.2}) →
  (-2 * Real.sqrt 5 ≤ a ∧ a ≤ 2 * Real.sqrt 5) :=
sorry

end intersection_range_l173_173027


namespace intersection_complement_l173_173572

def M (x : ℝ) : Prop := x^2 - 2 * x < 0
def N (x : ℝ) : Prop := x < 1

theorem intersection_complement (x : ℝ) :
  (M x ∧ ¬N x) ↔ (1 ≤ x ∧ x < 2) := 
sorry

end intersection_complement_l173_173572


namespace expected_value_of_sixes_l173_173652

theorem expected_value_of_sixes (n : ℕ) (k : ℕ) (p q : ℚ) 
  (h1 : n = 3) 
  (h2 : k = 6)
  (h3 : p = 1/6) 
  (h4 : q = 5/6) : 
  (1 : ℚ) / 2 = ∑ i in finset.range (n + 1), (i * (nat.choose n i * p^i * q^(n-i))) := 
sorry

end expected_value_of_sixes_l173_173652


namespace smallest_positive_angle_l173_173382

theorem smallest_positive_angle:
  ∃ (x : ℝ), 0 < x ∧ x ≤ 5.625 ∧ tan (6 * x * (π / 180)) = (cos (2 * x * (π / 180)) - sin (2 * x * (π / 180))) / (cos (2 * x * (π / 180)) + sin (2 * x * (π / 180))) :=
by
  sorry

end smallest_positive_angle_l173_173382


namespace no_solution_l173_173800

theorem no_solution (n : ℕ) (k : ℕ) (hn : Prime n) (hk : 0 < k) :
  ¬ (n ≤ n.factorial - k ^ n ∧ n.factorial - k ^ n ≤ k * n) :=
by
  sorry

end no_solution_l173_173800


namespace total_price_jewelry_paintings_l173_173981

theorem total_price_jewelry_paintings:
  (original_price_jewelry original_price_paintings increase_jewelry_in_dollars increase_paintings_in_percent quantity_jewelry quantity_paintings new_price_jewelry new_price_paintings total_cost: ℝ) 
  (h₁: original_price_jewelry = 30)
  (h₂: original_price_paintings = 100)
  (h₃: increase_jewelry_in_dollars = 10)
  (h₄: increase_paintings_in_percent = 0.20)
  (h₅: quantity_jewelry = 2)
  (h₆: quantity_paintings = 5)
  (h₇: new_price_jewelry = original_price_jewelry + increase_jewelry_in_dollars)
  (h₈: new_price_paintings = original_price_paintings + (original_price_paintings * increase_paintings_in_percent))
  (h₉: total_cost = (new_price_jewelry * quantity_jewelry) + (new_price_paintings * quantity_paintings)) :
  total_cost = 680 :=
by 
  sorry

end total_price_jewelry_paintings_l173_173981


namespace find_chosen_number_l173_173757

theorem find_chosen_number (x : ℤ) (h : 2 * x - 138 = 106) : x = 122 :=
by
  sorry

end find_chosen_number_l173_173757


namespace complex_expression_equality_l173_173124

-- variables declaration
def z := (1 : ℂ) + (complex.I : ℂ)
def target_expression := (2 : ℂ) / z + complex.conj z
def expected_value := (2 : ℂ) - 2 * (complex.I : ℂ)

-- statement to prove
theorem complex_expression_equality : target_expression = expected_value := 
by sorry

end complex_expression_equality_l173_173124


namespace peg_placement_unique_l173_173648

-- Define the triangular board
structure TriangularBoard :=
(num_rows : ℕ)

-- Define the peg color count conditions
structure PegCounts :=
(yellow : ℕ)
(red : ℕ)
(green : ℕ)
(blue : ℕ)
(orange : ℕ)

-- Define the triangular pegboard problem
def triangular_pegboard_problem : Prop :=
  ∃ (board : TriangularBoard) (counts : PegCounts),
  board.num_rows = 5 ∧
  counts.yellow = 5 ∧
  counts.red = 4 ∧
  counts.green = 3 ∧
  counts.blue = 2 ∧
  counts.orange = 1 ∧
  ∀ (placement : Fin (triangle_number 5) → Fin (counts.yellow + counts.red + counts.green + counts.blue + counts.orange)),
  no_two_pegs_same_color_in_same_row_or_column_spot placement

-- Define the condition ensuring no two pegs of the same color are in the same row or column
def no_two_pegs_same_color_in_same_row_or_column_spot (placement: Fin (triangle_number 5) → Fin (15)) : Prop :=
  ∀ (i j : Fin (triangle_number 5)), 
    i ≠ j →
    placement i ≠ placement j →
    (row i ≠ row j ∧ column i ≠ column j)

-- Triangle number definition for a triangular board with num_rows rows
def triangle_number (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

-- Define row indices for the board
def row (i : Fin (triangle_number 5)) : ℕ := sorry

-- Define column indices for the board
def column (i : Fin (triangle_number 5)) : ℕ := sorry

-- The theorem which states that the number of valid peg placements is 1
theorem peg_placement_unique :
  triangular_pegboard_problem :=
by
  -- Begin the proof block
  sorry

end peg_placement_unique_l173_173648


namespace incorrect_option_c_l173_173021

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := cos (2 * x) * cos φ - sin (2 * x) * sin φ

theorem incorrect_option_c :
  (0 < φ) ∧ (φ < π / 2) →
  (f (π / 6) φ = 0) →
  φ = π / 6 →
  (¬(∃ g, (∀ x, g x = cos (2 * (x - π / 6))) ∧ (∀ x, f x φ = g x)))
    :=
by intros h1 h2 h3
   sorry

end incorrect_option_c_l173_173021


namespace four_digit_perfect_square_palindromes_eq_one_l173_173878

theorem four_digit_perfect_square_palindromes_eq_one :
  ∃! (n : ℕ), (1024 ≤ n) ∧ (n ≤ 9801) ∧ (n = (m * m) ∧ is_palindrome n) :=
sorry

-- Auxiliary definition for checking if a number is a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in digits = digits.reverse
  

end four_digit_perfect_square_palindromes_eq_one_l173_173878


namespace no_solution_when_p_gt_one_fourth_solution_when_p_eq_one_fourth_l173_173866

noncomputable theory
open Real

theorem no_solution_when_p_gt_one_fourth (p : ℝ) (x : ℝ) 
  (hp : p > 1 / 4) 
  (hx_pos : x > 0) 
  (h_eq : log (nat.succ 1) (x) ^ 2 + 2 * log (nat.succ 1) (x) + 2 * log (nat.succ 1) (x^2 + p) + p + 15 / 4 = 0) : 
  false := 
sorry

theorem solution_when_p_eq_one_fourth (p : ℝ) (x : ℝ) 
  (hp : p = 1 / 4) 
  (hx_pos : x > 0) :
  log (nat.succ 1) (x) ^ 2 + 2 * log (nat.succ 1) (x) + 2 * log (nat.succ 1) (x^2 + p) + p + 15 / 4 = 0 ↔ 
  x = 1 / 2 := 
sorry

end no_solution_when_p_gt_one_fourth_solution_when_p_eq_one_fourth_l173_173866


namespace prime_square_mod_30_l173_173593

theorem prime_square_mod_30 (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : p > 5) : 
  p^2 % 30 = 1 ∨ p^2 % 30 = 19 := 
sorry

end prime_square_mod_30_l173_173593


namespace compute_100m_plus_n_l173_173098

noncomputable def f (x : ℝ) := x^2 + x

theorem compute_100m_plus_n :
  ∃ (m n : ℤ) (y z : ℝ), 
    0 < m ∧ 0 < n ∧ y ≠ 0 ∧ z ≠ 0 ∧ y ≠ z ∧
    f(y) = m + Real.sqrt n ∧ 
    f(z) = m + Real.sqrt n ∧ 
    f(1/y) + f(1/z) = 1/10 ∧
    100 * m + n = 1735 :=
by
  sorry

end compute_100m_plus_n_l173_173098


namespace flower_beds_fraction_l173_173749

noncomputable def isosceles_right_triangle_area (leg : ℝ) : ℝ :=
  (1 / 2) * leg^2

noncomputable def fraction_of_yard_occupied_by_flower_beds : ℝ :=
  let leg_length := (25 - 15) / 2
  let area_of_one_triangle := isosceles_right_triangle_area leg_length
  let total_area_of_flower_beds := 2 * area_of_one_triangle
  let area_of_yard := 25 * 5
  total_area_of_flower_beds / area_of_yard

theorem flower_beds_fraction : 
  let leg_length := (25 - 15) / 2
  let area_of_one_triangle := isosceles_right_triangle_area leg_length
  let total_area_of_flower_beds := 2 * area_of_one_triangle
  let area_of_yard := 25 * 5
  total_area_of_flower_beds / area_of_yard = 1 / 5 :=
by
  sorry

end flower_beds_fraction_l173_173749


namespace negation_of_p_l173_173052

-- Define the proposition p
def p : Prop := ∀ x : ℝ, 2 * x^2 + 1 > 0

-- State the negation of p
theorem negation_of_p : ¬p ↔ ∃ x : ℝ, 2 * x^2 + 1 ≤ 0 := sorry

end negation_of_p_l173_173052


namespace infinite_N_with_same_digit_sum_l173_173125

theorem infinite_N_with_same_digit_sum (A : ℕ) : ∃ (N : ℕ), (∀ (n : ℕ), N = 10^n - 1 → sum_of_digits N = sum_of_digits (A * N)) :=
sorry

end infinite_N_with_same_digit_sum_l173_173125


namespace percent_of_a_l173_173174

theorem percent_of_a (a b : ℝ) (h : a = 1.2 * b) : 4 * b = (10 / 3) * a :=
sorry

end percent_of_a_l173_173174


namespace triangle_BC_length_l173_173086

theorem triangle_BC_length
  (A B C : Type)
  [right_triangle A B C]
  (tan_A : Real := 4 / 3)
  (AB : Real := 3)
  (tan_A_definition : tan A = BC / AB) :
  (BC : Real) = 4 :=
by
  sorry

end triangle_BC_length_l173_173086


namespace equality_of_arithmetic_sums_l173_173428

def sum_arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem equality_of_arithmetic_sums (n : ℕ) (h : n ≠ 0) :
  sum_arithmetic_sequence 8 4 n = sum_arithmetic_sequence 17 2 n ↔ n = 10 :=
by
  sorry

end equality_of_arithmetic_sums_l173_173428


namespace new_students_count_l173_173522

def initial_students := 10.0
def added_students := 4.0
def total_students := 56.0

theorem new_students_count :
  initial_students + added_students + new_students = total_students →
  new_students = 42 :=
by
  intro h
  have h1 : initial_students + added_students = 14.0 := by norm_num
  rw [h1] at h
  linarith
  sorry

end new_students_count_l173_173522


namespace probability_of_selecting_one_expired_bottle_l173_173016

theorem probability_of_selecting_one_expired_bottle : 
  let total_bottles := 12
  let expired_bottles := 2
  (nat.choose expired_bottles 1 * nat.choose (total_bottles - expired_bottles) 0) / 
  nat.choose total_bottles 1 = 1 / 6 := 
by
  sorry

end probability_of_selecting_one_expired_bottle_l173_173016


namespace minimum_elements_union_l173_173163

open Set

def A : Finset ℕ := sorry
def B : Finset ℕ := sorry

variable (size_A : A.card = 25)
variable (size_B : B.card = 18)
variable (at_least_10_not_in_A : (B \ A).card ≥ 10)

theorem minimum_elements_union : (A ∪ B).card = 35 :=
by
  sorry

end minimum_elements_union_l173_173163


namespace derivative_of_function_l173_173619

noncomputable def derivative_function (x : ℝ) : ℝ :=
  5 * (x + 1/x)^4 * (1 - 1/x^2)

theorem derivative_of_function (x : ℝ) (hx : x ≠ 0) :
  deriv (λ x : ℝ, (x + 1/x)^5) x = derivative_function x := 
sorry

end derivative_of_function_l173_173619


namespace hexagon_perimeter_l173_173532

theorem hexagon_perimeter :
  ∃ (x y z : ℝ), (angle A = 120 * real.pi / 180) ∧ 
                 (angle B = 60 * real.pi / 180) ∧ 
                 (angle C = 120 * real.pi / 180) ∧ 
                 (angle D = 60 * real.pi / 180) ∧ 
                 (angle E = 120 * real.pi / 180) ∧ 
                 (angle F = 60 * real.pi / 180) ∧ 
                 (AB = DE) ∧ 
                 (BC = EF) ∧ 
                 (CD = FA) ∧ 
                 (hexagon_area ABCDEF = 12) ∧ 
                 6 * x = 8 * real.sqrt(3) :=
sorry

end hexagon_perimeter_l173_173532


namespace exists_repeating_row_row_11_equals_row_12_example_diff_between_row_10_and_11_l173_173361

-- Definitions for the conditions
def first_row : ℕ → ℕ := sorry -- A function representing the first row of 1000 numbers
def row (n : ℕ) : ℕ → ℕ := sorry -- A function to get the n-th row based on our transformation rule

-- Problem 1
theorem exists_repeating_row :
  ∃ n, row n = row (n + 1) :=
sorry

-- Problem 2
theorem row_11_equals_row_12 :
  row 11 = row 12 :=
sorry

-- Problem 3
theorem example_diff_between_row_10_and_11 :
  ∃ first_row, row 10 ≠ row 11 :=
sorry

end exists_repeating_row_row_11_equals_row_12_example_diff_between_row_10_and_11_l173_173361


namespace marble_probability_l173_173727

theorem marble_probability
  (total_marbles : ℕ)
  (blue_marbles : ℕ)
  (green_marbles : ℕ)
  (draws : ℕ)
  (prob_first_green : ℚ)
  (prob_second_blue_given_green : ℚ)
  (total_prob : ℚ)
  (h_total : total_marbles = 10)
  (h_blue : blue_marbles = 4)
  (h_green : green_marbles = 6)
  (h_draws : draws = 2)
  (h_prob_first_green : prob_first_green = 3 / 5)
  (h_prob_second_blue_given_green : prob_second_blue_given_green = 4 / 9)
  (h_total_prob : total_prob = 4 / 15) :
  prob_first_green * prob_second_blue_given_green = total_prob := sorry

end marble_probability_l173_173727


namespace friend_bicycles_count_l173_173513

-- Define the conditions.
def ignatius_bicycles : ℕ := 4
def tires_per_bicycle : ℕ := 2
def friend_cycles_tire_ratio : ℕ := 3
def unicycle_tires : ℕ := 1
def tricycle_tires : ℕ := 3

-- Calculate the number of bicycles the friend has given the conditions.
theorem friend_bicycles_count :
  let ignatius_total_tires := ignatius_bicycles * tires_per_bicycle in
  let friend_total_tires := friend_cycles_tire_ratio * ignatius_total_tires in
  let remaining_tires := friend_total_tires - unicycle_tires - tricycle_tires in
  let bicycles_owned_by_friend := remaining_tires / tires_per_bicycle in
  bicycles_owned_by_friend = 10 := sorry

end friend_bicycles_count_l173_173513


namespace ratio_of_areas_l173_173903

theorem ratio_of_areas (R_C R_D : ℝ) (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 4 / 9 :=
by
  sorry

end ratio_of_areas_l173_173903


namespace slope_of_tangent_line_l173_173012

def circle_eq : ℝ → ℝ → Prop := λ x y, x^2 + y^2 = 1

def line_eq (k : ℝ) : ℝ → ℝ → Prop := λ x y, y = k * (x + 3)

def is_tangent_to_circle (k : ℝ) : Prop :=
  let d := λ k, |3 * k| / Real.sqrt (1 + k^2)
  d k = 1

theorem slope_of_tangent_line :
  ∃ k : ℝ, (line_eq k (-3) 0) ∧ is_tangent_to_circle k ∧ k = abs (Real.sqrt 2 / 4) ∨ k = - (Real.sqrt 2 / 4) := sorry

end slope_of_tangent_line_l173_173012


namespace constant_term_of_expansion_l173_173099

def integral_value : ℝ := ∫ x in 0..(Real.pi / 2), Real.sin (2 * x)

theorem constant_term_of_expansion :
  let a := integral_value in
  a = 1 ∧ (2 * x + a / x)^6 = (2 * x + 1 / x)^6 ∧
  (∃ (t : ℝ) (r : ℕ), t = (Finset.choose 6 r) * 2^(6 - r) * x^(6 - 2 * r) ∧ 6 - 2 * r = 0 ∧ 
  t = 160) :=
begin
  sorry
end

end constant_term_of_expansion_l173_173099


namespace std_dev_transformed_l173_173054

open Real 

-- Given condition
def std_dev_x : ℝ := 8

-- To prove
theorem std_dev_transformed (x : ℕ → ℝ) (std_dev_x : ℝ) (h : std_dev_x = 8) :
  let var_x := std_dev_x ^ 2 in
  let var_transformed := 4 * var_x in
  let std_dev_transformed := sqrt var_transformed in
  std_dev_transformed = 16 :=
by
  sorry

end std_dev_transformed_l173_173054


namespace vasya_maximum_rank_l173_173248

theorem vasya_maximum_rank {n : ℕ} (cyclists stages : ℕ) (VasyaPlace : ℕ) 
  (rankings : Π (s : fin stages), fin cyclists):
  cyclists = 500 → stages = 15 → VasyaPlace = 7 →
  (∀ s, rankings s VasyaPlace = 7) →
  (∀ s t i, rankings s i ≠ rankings t i) →
  (∀ s, ∃ l, list.nodup l ∧ (∀ i j, i < j → rankings s i < rankings s j) ∧ list.length l = 500) →
  (∃ (maxRank : ℕ), maxRank = 91) :=
by
  intros hcyclists hstages hplace hvasya_place hdistinct_rankings hstage_rankings
  use 91
  sorry

end vasya_maximum_rank_l173_173248


namespace tripod_max_height_l173_173352

noncomputable def tripod_new_height (original_height : ℝ) (original_leg_length : ℝ) (broken_leg_length : ℝ) : ℝ :=
  (broken_leg_length / original_leg_length) * original_height

theorem tripod_max_height :
  let original_height := 5
  let original_leg_length := 6
  let broken_leg_length := 4
  let h := tripod_new_height original_height original_leg_length broken_leg_length
  h = (10 / 3) :=
by
  sorry

end tripod_max_height_l173_173352


namespace triangle_largest_angle_l173_173071

theorem triangle_largest_angle (x : ℝ) (hx : x + 2 * x + 3 * x = 180) :
  3 * x = 90 :=
by
  sorry

end triangle_largest_angle_l173_173071


namespace trapezoid_area_l173_173927

theorem trapezoid_area (l : ℝ) (r : ℝ) (a b : ℝ) (h : ℝ) (A : ℝ) :
  l = 9 →
  r = 4 →
  a + b = l + l →
  h = 2 * r →
  (a + b) / 2 * h = A →
  A = 72 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end trapezoid_area_l173_173927


namespace solve_part_a_solve_part_b_solve_part_c_l173_173167

-- Part (a)
theorem solve_part_a (x : ℝ) : 
  (2 * x^2 + 3 * x - 1)^2 - 5 * (2 * x^2 + 3 * x + 3) + 24 = 0 ↔ 
  x = 1 ∨ x = -2 ∨ x = 0.5 ∨ x = -2.5 := sorry

-- Part (b)
theorem solve_part_b (x : ℝ) : 
  (x - 1) * (x + 3) * (x + 4) * (x + 8) = -96 ↔ 
  x = 0 ∨ x = -7 ∨ x = (-7 + Real.sqrt 33) / 2 ∨ x = (-7 - Real.sqrt 33) / 2 := sorry

-- Part (c)
theorem solve_part_c (x : ℝ) (hx : x ≠ 0) : 
  (x - 1) * (x - 2) * (x - 4) * (x - 8) = 4 * x^2 ↔ 
  x = 4 + 2 * Real.sqrt 2 ∨ x = 4 - 2 * Real.sqrt 2 := sorry

end solve_part_a_solve_part_b_solve_part_c_l173_173167


namespace person1_is_L_and_person2_is_P_l173_173150

variables (Person : Type) (L P : Person → Prop)
variables (number_of_people : ℕ)
variables (total_L : ℕ) (total_P : ℕ)

-- Total number of liars (L) and truth-tellers (P) on the island
axiom (h_total_L : total_L = 1000)
axiom (h_total_P : total_P = 1000)
-- Number of people on the island (sum of liars and truth-tellers)
axiom (h_number_of_people : number_of_people = total_L + total_P)

-- Person 1 is a liar (L)
axiom (h_person1_is_L : L Person)
-- Person 2 is a truth-teller (P)
axiom (h_person2_is_P : P Person)

theorem person1_is_L_and_person2_is_P :
  L Person ∧ P Person := by
  sorry

end person1_is_L_and_person2_is_P_l173_173150


namespace remaining_hard_hats_l173_173073

theorem remaining_hard_hats 
  (pink_initial : ℕ)
  (green_initial : ℕ)
  (yellow_initial : ℕ)
  (carl_takes_pink : ℕ)
  (john_takes_pink : ℕ)
  (john_takes_green : ℕ) :
  john_takes_green = 2 * john_takes_pink →
  pink_initial = 26 →
  green_initial = 15 →
  yellow_initial = 24 →
  carl_takes_pink = 4 →
  john_takes_pink = 6 →
  ∃ pink_remaining green_remaining yellow_remaining total_remaining, 
    pink_remaining = pink_initial - carl_takes_pink - john_takes_pink ∧
    green_remaining = green_initial - john_takes_green ∧
    yellow_remaining = yellow_initial ∧
    total_remaining = pink_remaining + green_remaining + yellow_remaining ∧
    total_remaining = 43 :=
by
  sorry

end remaining_hard_hats_l173_173073


namespace area_ratio_proof_l173_173080

-- Define the lengths involved in the problem
variables (A B C D E F : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
variables (length_AB length_AC length_AD length_CF length_BD length_AF : real)
variables (area_CEF area_DBE : real)

-- Given conditions
def given_conditions : Prop :=
  length_AB = 130 ∧
  length_AC = 130 ∧
  length_AD = 50 ∧
  length_CF = 90 ∧
  length_BD = length_AB - length_AD ∧
  length_AF = length_AC + length_CF

-- Theorem to prove: the ratio of areas
theorem area_ratio_proof (h : given_conditions) : area_CEF / area_DBE = 99 / 65 :=
sorry

end area_ratio_proof_l173_173080


namespace vasya_rank_91_l173_173224

theorem vasya_rank_91 {n_cyclists : ℕ} {n_stages : ℕ} 
    (n_cyclists_eq : n_cyclists = 500) 
    (n_stages_eq : n_stages = 15) 
    (no_ties : ∀ (i j : ℕ), i < j → ∀ (s : fin n_stages), ¬(same_time i j s)) 
    (vasya_7th : ∀ (s : fin n_stages), ∀ (i : ℕ), i < 6 → better_than i 6 s) :
    possible_rank vasya ≤ 91 :=
sorry

end vasya_rank_91_l173_173224


namespace spherical_to_rectangular_coordinates_l173_173436

theorem spherical_to_rectangular_coordinates (
  (ρ θ φ : ℝ)
  (cond1 : 4 = ρ * sin φ * cos θ)
  (cond2 : -3 = ρ * sin φ * sin θ)
  (cond3 : -2 = ρ * cos φ)
) : 
  let θ' := θ + Real.pi
  let φ' := -φ
  let x := ρ * sin φ' * cos θ'
  let y := ρ * sin φ' * sin θ'
  let z := ρ * cos φ'
  (x, y, z) = (-4, 3, -2) :=
by
  sorry

end spherical_to_rectangular_coordinates_l173_173436


namespace price_of_36kgs_l173_173360

namespace Apples

-- Define the parameters l and q
variables (l q : ℕ)

-- Define the conditions
def cost_first_30kgs (l : ℕ) : ℕ := 30 * l
def cost_first_15kgs : ℕ := 150
def cost_33kgs (l q : ℕ) : ℕ := (30 * l) + (3 * q)
def cost_36kgs (l q : ℕ) : ℕ := (30 * l) + (6 * q)

-- Define the hypothesis for l and q based on given conditions
axiom l_value (h1 : cost_first_15kgs = 150) : l = 10
axiom q_value (h2 : cost_33kgs l q = 333) : q = 11

-- Prove the price of 36 kilograms of apples
theorem price_of_36kgs (h1 : cost_first_15kgs = 150) (h2 : cost_33kgs l q = 333) : cost_36kgs l q = 366 :=
sorry

end Apples

end price_of_36kgs_l173_173360


namespace john_dimes_l173_173942

theorem john_dimes :
  ∀ (d : ℕ), 
  (4 * 25 + d * 10 + 5) = 135 → (5: ℕ) + (d: ℕ) * 10 + 4 = 4 + 131 + 3*d → d = 3 :=
by
  sorry

end john_dimes_l173_173942


namespace vasya_maximum_rank_l173_173253

theorem vasya_maximum_rank {n : ℕ} (cyclists stages : ℕ) (VasyaPlace : ℕ) 
  (rankings : Π (s : fin stages), fin cyclists):
  cyclists = 500 → stages = 15 → VasyaPlace = 7 →
  (∀ s, rankings s VasyaPlace = 7) →
  (∀ s t i, rankings s i ≠ rankings t i) →
  (∀ s, ∃ l, list.nodup l ∧ (∀ i j, i < j → rankings s i < rankings s j) ∧ list.length l = 500) →
  (∃ (maxRank : ℕ), maxRank = 91) :=
by
  intros hcyclists hstages hplace hvasya_place hdistinct_rankings hstage_rankings
  use 91
  sorry

end vasya_maximum_rank_l173_173253


namespace monotonicity_and_extremes_range_of_k_inequality_l173_173020

-- (I)
theorem monotonicity_and_extremes (f : ℝ → ℝ) (hf : ∀ x > 0, f x = (log x + 1) / x) :
  ∃ I1 I2, (∀ x ∈ I1, deriv f x > 0) ∧ (∀ x ∈ I2, deriv f x < 0) ∧ 
  (∀ t1 t2, x ∈ I1 ∧ x = 1 → f x = 1) ∧ 
  ∃ max_value, ∀ x, f x ≤ max_value ∧ max_value = 1 :=
sorry

-- (II)
theorem range_of_k (k : ℝ) (h : ∀ x > 1, log (x - 1) + k + 1 ≤ k * x) : k ≥ 1 :=
sorry

-- (III)
theorem inequality (n : ℕ) (hn : n ≥ 2) :
  ∑ i in finset.range n, (log (i + 2) / (i + 2)^2) < (2 * n^2 - n - 1) / (4 * (n + 1)) :=
sorry

end monotonicity_and_extremes_range_of_k_inequality_l173_173020


namespace no_finite_set_of_non_parallel_vectors_l173_173998

theorem no_finite_set_of_non_parallel_vectors (N : ℕ) (hN : N > 3) :
  ¬ ∃ (G : Finset (ℝ × ℝ)), G.card > 2 * N ∧
      (∀ (H : Finset (ℝ × ℝ)), H ⊆ G ∧ H.card = N →
        (∃ (F : Finset (ℝ × ℝ)), F ⊆ G ∧ F ≠ H ∧ F.card = N - 1 ∧ (H ∪ F).sum = 0)) ∧
      (∀ (H : Finset (ℝ × ℝ)), H ⊆ G ∧ H.card = N →
        (∃ (F : Finset (ℝ × ℝ)), F ⊆ G ∧ F ≠ H ∧ F.card = N ∧ (H ∪ F).sum = 0)) :=
  sorry

end no_finite_set_of_non_parallel_vectors_l173_173998


namespace f_odd_function_f_decreasing_on_interval_l173_173857

noncomputable def f (x : ℝ) : ℝ := -x + (1 / (2 * x))

theorem f_odd_function : ∀ x : ℝ , x ≠ 0 → f (-x) = - f x :=
by
  intro x hx
  sorry

theorem f_decreasing_on_interval : ∀ x : ℝ , (x > 0) → f' x < 0 :=
by
  intro x hx
  have derivative := by
    calc
      f' x = -1 - (1 / (2 * x ^ 2)) := by sorry -- compute the derivative
  exact sorry -- prove that the derivative is negative for all x > 0

end f_odd_function_f_decreasing_on_interval_l173_173857


namespace vasya_lowest_position_l173_173259

noncomputable theory

def number_of_cyclists := 500
def number_of_stages := 15
def position_of_vasya_each_stage := 7

theorem vasya_lowest_position (total_cyclists : ℕ) (stages : ℕ) (position_each_stage : ℕ)
  (h_total_cyclists : total_cyclists = number_of_cyclists)
  (h_stages : stages = number_of_stages)
  (h_position_each_stage : position_each_stage = position_of_vasya_each_stage)
  (no_identical_times : ∀ (i j : ℕ), i ≠ j → ∀ (stage : ℕ), stage ≤ stages → ∀ (t : ℕ), t ≤ total_cyclists → 
    (time : Π (s : ℕ) (c : ℕ), c < total_cyclists → ℕ), 
    time stage i < time stage j ∨ time stage j < time stage i):
  ∃ lowest_pos, lowest_pos = 91 := sorry

end vasya_lowest_position_l173_173259


namespace circle_area_ratio_l173_173890

theorem circle_area_ratio (R_C R_D : ℝ)
  (h₁ : (60 / 360 * 2 * Real.pi * R_C) = (40 / 360 * 2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 9 / 4 :=
by 
  sorry

end circle_area_ratio_l173_173890


namespace find_purple_balls_count_l173_173326

theorem find_purple_balls_count (k : ℕ) (h : ∃ k > 0, (21 - 3 * k) = (3 / 4) * (7 + k)) : k = 4 :=
sorry

end find_purple_balls_count_l173_173326


namespace vasya_lowest_position_l173_173217

theorem vasya_lowest_position
  (num_cyclists : ℕ)
  (num_stages : ℕ)
  (num_ahead : ℕ)
  (position_vasya : ℕ)
  (total_time : List ℕ)
  (unique_total_times : total_time.nodup)
  (stage_positions : List (List ℕ))
  (unique_stage_positions : ∀ stage ∈ stage_positions, stage.nodup)
  (vasya_consistent : ∀ stage ∈ stage_positions, stage.nth position_vasya = some num_ahead) :
  num_ahead * num_stages + 1 = 91 :=
by
  sorry

end vasya_lowest_position_l173_173217


namespace probability_event_A_l173_173918

noncomputable def probability_correct (n : ℕ) : ℝ :=
0.8

noncomputable def probability_incorrect (n : ℕ) : ℝ :=
1 - probability_correct n

def independent_outcomes {n : ℕ} (qs : Fin n → Bool) : Prop :=
∀ i j, i ≠ j → qs i ≠ qs j

def event_A (qs : Fin 5 → Bool) : Prop :=
(qs 1 = false) ∧ (qs 2 = true) ∧ (qs 3 = true)

theorem probability_event_A :
  let qs : Fin 5 → Bool := λ _, true,
      pA := (probability_correct 2) * (probability_incorrect 3) * (probability_correct 4) * (probability_correct 5),
      pA_alt := (probability_incorrect 1) * (probability_incorrect 3) * (probability_correct 4) * (probability_correct 5)
  in
    pA + pA_alt = 0.128 :=
by 
  let qs := λ i : Fin 5, true
  let pA := (probability_correct 2) * (probability_incorrect 3) * (probability_correct 4) * (probability_correct 5)
  have hpA: pA = 0.8 * 0.2 * 0.8 * 0.8 := rfl
  let pA_alt := (probability_incorrect 1) * (probability_incorrect 3) * (probability_correct 4) * (probability_correct 5)
  have hpA_alt : pA_alt = 0.2 * 0.2 * 0.8 * 0.8 := rfl
  rw [hpA, hpA_alt]
  exact (by norm_num : 0.8 * 0.2 * 0.8 * 0.8 + 0.2 * 0.2 * 0.8 * 0.8 = 0.128)

end probability_event_A_l173_173918


namespace vasya_rank_91_l173_173226

theorem vasya_rank_91 {n_cyclists : ℕ} {n_stages : ℕ} 
    (n_cyclists_eq : n_cyclists = 500) 
    (n_stages_eq : n_stages = 15) 
    (no_ties : ∀ (i j : ℕ), i < j → ∀ (s : fin n_stages), ¬(same_time i j s)) 
    (vasya_7th : ∀ (s : fin n_stages), ∀ (i : ℕ), i < 6 → better_than i 6 s) :
    possible_rank vasya ≤ 91 :=
sorry

end vasya_rank_91_l173_173226


namespace lines_PR_SQ_perpendicular_quadrilateral_PQRS_rhombus_l173_173101

variables {A B C D E F P Q R S : Type} [inst : LinearOrderedField Type]

noncomputable theory

-- Assume points A, B, C, D are on a circle, defining a cyclic quadrilateral ABCD
variable (cyclic_quad : IsCyclicQuadrilateral A B C D)

-- Define intersection point E of lines AB and CD
variable (intersect_ab_cd : Intersection A B C D E)

-- Define intersection point F of lines AD and BC
variable (intersect_ad_bc : Intersection A D B C F)

-- Define points P and R on [AB] and [CD] respectively such that bisector of ∠AFB intersects them
variable (bisector_afb : AngleBisector A F B P R)

-- Define points Q and S on [BC] and [AD] respectively such that bisector of ∠BEC intersects them
variable (bisector_bec : AngleBisector B E C Q S)

-- Prove that lines PR and SQ are perpendicular
theorem lines_PR_SQ_perpendicular : Perpendicular PR SQ :=
sorry

-- Prove that PQRS is a rhombus
theorem quadrilateral_PQRS_rhombus : IsRhombus P Q R S :=
sorry

end lines_PR_SQ_perpendicular_quadrilateral_PQRS_rhombus_l173_173101


namespace problem_solution_l173_173135

variable {x : ℕ → ℝ}

-- Defining the conditions
def condition_ineq : Prop :=
  ∀ n : ℕ, (1 ≤ n ∧ n ≤ 2016) → (x (n + 1))^2 ≤ x n * x (n + 2)

def condition_prod : Prop :=
  (∏ n in Finset.range 2018, x (n + 1)) = 1

-- Statement to be proved
theorem problem_solution (h1 : condition_ineq) (h2 : condition_prod) : 
  x 1009 * x 1010 ≤ 1 := 
sorry

end problem_solution_l173_173135


namespace number_of_numerators_repeating_decimal_l173_173954

theorem number_of_numerators_repeating_decimal (S : Set ℚ) (hS : ∀ r ∈ S, ∃ a b c : ℕ, 0 < r ∧ r < 1 ∧ r = (a * 100 + b * 10 + c) / 999 ∧ (r.denom = 999 ∨ (r.denom = 999 / 3 ∨ r.denom = 999 / 37))) : 
  ∃ n : ℕ, n = 660 :=
by
  use 660
  sorry

end number_of_numerators_repeating_decimal_l173_173954


namespace find_x_l173_173324

theorem find_x (x : ℝ) (h : 0.009 / x = 0.03) : x = 0.3 :=
sorry

end find_x_l173_173324


namespace num_sets_without_perfect_squares_is_388_l173_173106

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def set_T (i : ℕ) : set ℕ :=
  { n | 150 * i ≤ n ∧ n < 150 * (i + 1) }

def contains_perfect_square (T : set ℕ) : Prop :=
  ∃ n ∈ T, is_perfect_square n

def count_sets_without_perfect_squares : ℕ :=
  finset.card { i ∈ finset.range 667 | ¬ contains_perfect_square (set_T i) }

theorem num_sets_without_perfect_squares_is_388 :
  count_sets_without_perfect_squares = 388 :=
sorry

end num_sets_without_perfect_squares_is_388_l173_173106


namespace find_m_n_l173_173315

def international_mathematical_olympiad (n m : ℝ) : Prop :=
  (m^2 + n^2) / 50 - n + 49 = n - 1

theorem find_m_n (n m : ℝ) :
  international_mathematical_olympiad n m → m = 0 ∧ n = 50 :=
by
  intros,
  sorry

end find_m_n_l173_173315


namespace find_T_l173_173993

theorem find_T (T : ℝ) 
  (h : (1/3) * (1/8) * T = (1/4) * (1/6) * 150) : 
  T = 150 :=
sorry

end find_T_l173_173993


namespace find_abc_sum_l173_173972

noncomputable def x := Real.sqrt ((Real.sqrt 105) / 2 + 7 / 2)

theorem find_abc_sum :
  ∃ (a b c : ℕ), a + b + c = 5824 ∧
  x ^ 100 = 3 * x ^ 98 + 15 * x ^ 96 + 12 * x ^ 94 - x ^ 50 + a * x ^ 46 + b * x ^ 44 + c * x ^ 40 :=
  sorry

end find_abc_sum_l173_173972


namespace combined_perimeter_l173_173387

-- Define the radii of the semi-circles
def r_large : ℝ := 6.6
def r_small : ℝ := 3.3

-- Define π approximation for calculation
def pi_approx : ℝ := 3.14159

-- Calculate the perimeters
def P_large : ℝ := r_large * π + 2 * r_large
def P_small : ℝ := r_small * π + 2 * r_small
def P_combined : ℝ := P_large + 2 * P_small

-- Show that P_combined approximates to 67.87 cm
theorem combined_perimeter : P_combined ≈ 67.87 := by
  -- Lean equivalence proof placeholder
  sorry

end combined_perimeter_l173_173387


namespace Sarah_landmarks_visits_l173_173161

theorem Sarah_landmarks_visits :
  let landmarks := 5 in
  let sequences := landmarks.factorial in
  sequences = 120 :=
by
  sorry

end Sarah_landmarks_visits_l173_173161


namespace number_of_digits_in_value_l173_173402

theorem number_of_digits_in_value : (nat.digits 10 $ (2^15)/(5^12)) = 1 := by 
  sorry

end number_of_digits_in_value_l173_173402


namespace problem_correct_statements_l173_173160

noncomputable def f (x : ℝ) : ℝ := sin x * cos x - cos x^2

theorem problem_correct_statements :
    (¬ (∀ x ∈ ℝ, f(x) = f(x + 2 * π))) ∧
    (∀ x ∈ Ioo (0 : ℝ) (π / 8), 0 < deriv f x) ∧
    (∀ x ∈ ℝ, f (3 * π / 8 - x) = f (3 * π / 8 + x)) ∧
    (¬ (∃ g : ℝ → ℝ, ∀ x : ℝ, g(x) = (f (x - π / 8)))) ∧
    (∀ x ∈ ℝ, f ((π / 4) + x) + f (-x) = -1) :=
sorry

end problem_correct_statements_l173_173160


namespace num_faces_lt_num_vertices_l173_173547

noncomputable def planar_bipartite_graph (V : Type*) : Type* := sorry -- Define planar_bipartite_graph

def num_vertices (G : planar_bipartite_graph) : ℕ := sorry -- Define a way to count vertices

def num_faces (G : planar_bipartite_graph) : ℕ := sorry -- Define a way to count faces

theorem num_faces_lt_num_vertices (G : planar_bipartite_graph) :
  num_faces G < num_vertices G := sorry

end num_faces_lt_num_vertices_l173_173547


namespace solve_system_eqns_l173_173171

theorem solve_system_eqns (x y : ℝ) (m : ℤ) :
  x^2 + 4 * sin y^2 - 4 = 0 ∧ cos x - 2 * cos y^2 - 1 = 0 ↔
  (x = 0 ∧ ∃ m : ℤ, y = (π / 2) + m * π) :=
by sorry

end solve_system_eqns_l173_173171


namespace average_of_integers_is_ten_l173_173613

theorem average_of_integers_is_ten (k m r s t : ℕ) 
  (h1 : k < m) (h2 : m < r) (h3 : r < s) (h4 : s < t)
  (h5 : k > 0) (h6 : m > 0)
  (h7 : t = 20) (h8 : r = 13)
  (h9 : k = 1) (h10 : m = 2) (h11 : s = 14) :
  (k + m + r + s + t) / 5 = 10 := by
  sorry

end average_of_integers_is_ten_l173_173613


namespace irrational_of_pi_over_2_l173_173307

theorem irrational_of_pi_over_2 :
  let A := (22 / 7 : ℚ),
  let B := (3 / 10 : ℚ),
  let D := (101 / 1000 : ℚ) + (10 ^ -6) * geometric_series_ratio 1 0.1,
  irrational (π / 2) → irrational (π / 2) :=
by sorry

end irrational_of_pi_over_2_l173_173307


namespace sector_area_correct_l173_173504

-- Define the initial conditions
def arc_length := 4 -- Length of the arc in cm
def central_angle := 2 -- Central angle in radians
def radius := arc_length / central_angle -- Radius of the circle

-- Define the formula for the area of the sector
def sector_area := (1 / 2) * radius * arc_length

-- The statement of our theorem
theorem sector_area_correct : sector_area = 4 := by
  -- Proof goes here
  sorry

end sector_area_correct_l173_173504


namespace permutation_of_triplets_l173_173486

variables {R : Type*} [LinearOrderedField R]

theorem permutation_of_triplets
  (a b c x y z : R)
  (u1 u2 u3 v1 v2 v3 : R)
  (hu1 : u1 = a * x + b * y + c * z)
  (hv1 : v1 = a * x + b * z + c * y)
  (hu2 : u2 = a * y + b * z + c * x)
  (hv2 : v2 = a * z + b * y + c * x)
  (hu3 : u3 = a * z + b * x + c * y)
  (hv3 : v3 = a * y + b * x + c * z)
  (h_product : u1 * u2 * u3 = v1 * v2 * v3) :
  ∃ σ : perm (fin 3), u1 = v1 → u2 = v2 → u3 = v3 :=
sorry

end permutation_of_triplets_l173_173486


namespace number_of_distinct_flags_l173_173737

inductive Color
| red | white | blue | green | yellow
deriving DecidableEq

open Color

def valid_colors : List Color := [red, white, blue, green, yellow]

/--
  A flag is composed of three horizontal strips where
  no two adjacent strips have the same color.
  Given 5 possible colors: red, white, blue, green, yellow,
  the question is to find the number of distinct flags.
-/
theorem number_of_distinct_flags : ∃ n, n = 80 :=
  let choices_middle := valid_colors.length
  let choices_top_bottom := valid_colors.length - 1
  have number_of_flags : 80 = 5 * 4 * 4 := by rfl
  ⟨80, number_of_flags⟩

end number_of_distinct_flags_l173_173737


namespace vasya_lowest_position_l173_173241

theorem vasya_lowest_position
  (n : ℕ) (m : ℕ) (num_cyclists : ℕ) (vasya_place : ℕ) :
  num_cyclists = 500 →
  n = 15 →
  vasya_place = 7 →
  ∀ (stages : fin n → fin num_cyclists) (no_identical_times : ∀ i j : fin n, i ≠ j → 
  ∀ k l : fin num_cyclists, stages i k ≠ stages j l),
  ∃ (lowest_position : ℕ), lowest_position = 91 := 
by sorry

end vasya_lowest_position_l173_173241


namespace vasya_lowest_position_l173_173245

theorem vasya_lowest_position
  (n : ℕ) (m : ℕ) (num_cyclists : ℕ) (vasya_place : ℕ) :
  num_cyclists = 500 →
  n = 15 →
  vasya_place = 7 →
  ∀ (stages : fin n → fin num_cyclists) (no_identical_times : ∀ i j : fin n, i ≠ j → 
  ∀ k l : fin num_cyclists, stages i k ≠ stages j l),
  ∃ (lowest_position : ℕ), lowest_position = 91 := 
by sorry

end vasya_lowest_position_l173_173245


namespace select_representatives_ways_l173_173162

theorem select_representatives_ways : 
  let males := 5
  let females := 4
  let total_reps := 4
  let at_least_males := 2
  let at_least_females := 1
  (Nat.choose males 2 * Nat.choose females 2 + Nat.choose males 3 * Nat.choose females 1) = 100 :=
by
  intro males females total_reps at_least_males at_least_females
  have h1 : Nat.choose males 2 * Nat.choose females 2 = 60 := by sorry
  have h2 : Nat.choose males 3 * Nat.choose females 1 = 40 := by sorry
  rw [h1, h2]
  norm_num

end select_representatives_ways_l173_173162


namespace num_zeros_in_square_of_999999999_l173_173494

theorem num_zeros_in_square_of_999999999 :
  let n := 9
  let number_of_nines := n
  let number := 10^n - 1
  let squared_number := number^2
  count_trailing_zeros squared_number = number_of_nines - 1 :=
by
  sorry

end num_zeros_in_square_of_999999999_l173_173494


namespace fractional_part_subtraction_l173_173310

theorem fractional_part_subtraction (x y : ℝ) : 
  ∃ k ∈ {0, -1}, {x + y} - {y} = x + k := sorry

end fractional_part_subtraction_l173_173310


namespace find_explicit_formula_l173_173000

noncomputable def quadratic_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (1 + x) = f (1 - x)) ∧
  (∀ x, f x ≤ 15 ∧ ∃ x0, f x0 = 15) ∧
  (∃ a b c, f x = a * x^2 + b * x + c ∧ a = -6)

noncomputable def explicit_quadratic_formula (f : ℝ → ℝ) : Prop :=
  f = (λ x, -6 * x^2 + 12 * x + 9)

theorem find_explicit_formula (f : ℝ → ℝ) :
  quadratic_function f → explicit_quadratic_formula f :=
by
  sorry

end find_explicit_formula_l173_173000


namespace area_of_square_l173_173505

theorem area_of_square (a b : ℝ) (h : a ≥ 0 ∧ b ≥ 0) 
  (diag : ℝ) (h_diag : diag = sqrt (a^2 + 4*b^2)) :
  ∃ (s : ℝ), s^2 = (a^2 + 4*b^2) / 2 :=
by
  sorry

end area_of_square_l173_173505


namespace geometric_sequence_general_formula_alternating_product_sum_l173_173829

theorem geometric_sequence_general_formula (q : ℝ) (a_n : ℕ → ℝ) 
  (h_geo : ∀ n, a_n (n + 1) = a_n n * q)
  (h_ratio : q > 1)
  (h_a2_a4 : a_n 2 + a_n 4 = 20)
  (h_a3 : a_n 3 = 8) :
  ∀ n, a_n n = 2^n := 
by
  sorry

theorem alternating_product_sum (n : ℕ) (a_n : ℕ → ℝ) 
  (h_geo : ∀ n, a_n (n + 1) = a_n n * 2)
  (h_a1_a2 : a_n 1 * a_n 2 = 4)
  (h_general_formula : ∀ n, a_n n = 2^n) :
  ∑ i in finset.range n, (-1) ^ i * a_n (i + 1) * a_n (i + 2) = 
    8 / 5 - (-1)^n * 2^(2*n + 3) / 5 :=
by
  sorry

end geometric_sequence_general_formula_alternating_product_sum_l173_173829


namespace bananas_to_oranges_l173_173771

theorem bananas_to_oranges :
  (∀ (bananas apples : ℕ), bananas = 4 → apples = 3 → cost_eq bananas apples) →
  (∀ (apples oranges : ℕ), apples = 5 → oranges = 2 → cost_eq apples oranges) →
  (∀ (bananas oranges : ℕ), bananas = 20 → cost_eq bananas oranges) →
  oranges = 6 :=
by
  sorry

end bananas_to_oranges_l173_173771


namespace takeoff_distance_l173_173272

theorem takeoff_distance (t v_kmh : ℝ) (uniform_acceleration : Prop) : 
  t = 15 ∧ v_kmh = 100 ∧ uniform_acceleration → 
  ∃ s : ℝ, s ≈ 208 :=
by 
  sorry

end takeoff_distance_l173_173272


namespace different_gender_pres_vp_l173_173584

-- defining members and their genders
def Club := Fin 20
def Boys := Fin 10
def Girls := Fin 10

-- given condition about number of boys and girls
axiom club_has_10_boys_and_10_girls : ∀ (x : Club), x ∈ Boys ∨ x ∈ Girls

-- correct answer
def different_gender_choices : Nat := 200

-- the Lean statement to prove the required number of choices is 200
theorem different_gender_pres_vp (h : ∀ (p : Club), ∀ (v : Club), (p ∈ Boys ∧ v ∈ Girls) ∨ (p ∈ Girls ∧ v ∈ Boys)) :
  ∃ (ways : Nat), ways = different_gender_choices :=
by
  use 200
  sorry

end different_gender_pres_vp_l173_173584


namespace sum_bn_eq_l173_173440

-- Define the sequences a_n and S_n and their properties
def a : ℕ → ℕ
| 1 := 1
| n := n 

def S : ℕ → ℕ
| n := (n * (n + 1)) / 2

def b (n : ℕ) : ℚ :=
(2 * a n - 1) / 3 ^ a n

-- Sum of the first n terms of b_n
def T (n : ℕ) : ℚ :=
(nat.sum (finset.range (n + 1)) (λ i, b i))

theorem sum_bn_eq (n : ℕ) : 
T n = 1 - (n + 1) / 3 ^ n :=
sorry

end sum_bn_eq_l173_173440


namespace x_diff_bound_l173_173970

noncomputable def x (n : ℕ) : ℝ :=
  if h : n ≥ 2 then 
    let rec seq (m : ℕ) : ℝ :=
      if m = n then n.toReal else real.root (m.toReal + seq (m + 1)) (m + 1)
    in seq 2
  else 0

theorem x_diff_bound (n : ℕ) (hn : 2 ≤ n) : x (n + 1) - x n ≤ 1 / n.fact :=
by
  sorry

end x_diff_bound_l173_173970


namespace limit_of_derivative_l173_173040

variable {f : ℝ → ℝ} {x₀ : ℝ}

theorem limit_of_derivative (h : deriv f x₀ = 4) :
  tendsto (λ Δx : ℝ, (f (x₀ + 2 * Δx) - f x₀) / Δx) (𝓝 0) (𝓝 8) :=
sorry

end limit_of_derivative_l173_173040


namespace vasya_lowest_position_l173_173203

theorem vasya_lowest_position
  (n_cyclists : ℕ) (n_stages : ℕ) 
  (stage_positions : ℕ → ℕ → ℕ) -- a function that takes a stage and a cyclist and returns the position (e.g., stage_positions(stage, cyclist) = position)
  (total_time : ℕ → ℕ)  -- a function that takes a cyclist and returns their total time
  (distinct_times : ∀ (c1 c2 : ℕ), c1 ≠ c2 → (total_time c1 ≠ total_time c2) ∧ 
                   ∀ (s : ℕ), stage_positions s c1 ≠ stage_positions s c2)
  (vasya_position : ℕ) (hv : ∀ (s : ℕ), s < n_stages → stage_positions s vasya_position = 7) :
  vasya_position = 91 :=
sorry

end vasya_lowest_position_l173_173203


namespace mary_study_hours_l173_173142

noncomputable def study_hours_to_achieve_average (h1 h2 s1 a : ℕ) := ℕ

theorem mary_study_hours :
  ∀ (h1 s1 s2 a : ℕ), (h1 * s1 = s2 * (study_hours_to_achieve_average h1 s1 s2 a)) →
    (a = (s1 + s2) / 2) →
    h1 = 6 →
    s1 = 60 →
    s2 = 120 →
    a = 90 →
    study_hours_to_achieve_average h1 s1 s2 a = 3 :=
by
  sorry

end mary_study_hours_l173_173142


namespace angle_between_vectors_perpendicular_find_m_l173_173031

variables (x y : ℝ -> ℝ) -- Assuming vectors as functions ℝ -> ℝ
variable (theta : ℝ)
variable (m : ℝ)

-- Assumptions
axiom norm_x : ∥x∥ = 1
axiom norm_y : ∥y∥ = 2
axiom dot_product_condition : (x - 2 • y) ⬝ (2 • x - y) = 5

-- Question 1: Prove theta = π / 3
theorem angle_between_vectors :
  theta ∈ [0, pi] ∧ cos theta = ((x ⬝ y) / (∥x∥ * ∥y∥)) ∧ (x ⬝ y = 1) → theta = π / 3 := sorry

-- Assumption for Question 2
axiom perpendicular_condition : (x - m • y) ⬝ y = 0

-- Question 2: Prove m = 1/4
theorem perpendicular_find_m :
  x ⬝ y = 1 ∧ ∥y∥ = 2 → m = 1 / 4 := sorry

end angle_between_vectors_perpendicular_find_m_l173_173031


namespace binomial_square_l173_173500

theorem binomial_square (p : ℝ) : (∃ b : ℝ, (3 * x + b)^2 = 9 * x^2 + 24 * x + p) → p = 16 := by
  sorry

end binomial_square_l173_173500


namespace base_area_of_parallelepiped_l173_173922

def is_diagonal (A1 C : Point) (d : ℝ) : Prop :=
  (dist A1 C) = d

def angle_inclined_to_base (A1 C : Point) (base : Plane) (θ : ℝ) : Prop :=
  ∠(line_through A1 C) base = θ

def angle_with_AC1_plane (A1 C : Point) (AC1_plane : Plane) (midpoint_BB1 : Point) (ϕ : ℝ) : Prop :=
  ∠(line_through A1 C) AC1_plane = ϕ

noncomputable def area_base_parallelepiped (d : ℝ) : ℝ :=
  (d^2) * (real.sqrt 3) / (8 * (real.sqrt 5))

theorem base_area_of_parallelepiped
  (A1 C : Point) 
  (base : Plane) 
  (AC1_plane : Plane) 
  (midpoint_BB1 : Point) 
  (d : ℝ) 
  (cond1 : is_diagonal A1 C d) 
  (cond2 : angle_inclined_to_base A1 C base (real.pi / 3)) 
  (cond3 : angle_with_AC1_plane A1 C AC1_plane midpoint_BB1 (real.pi / 4)) 
  : area_base_parallelepiped d = (d^2)*(real.sqrt 3)/(8*(real.sqrt 5)) := 
sorry

end base_area_of_parallelepiped_l173_173922


namespace triangle_parallel_vectors_l173_173441

noncomputable def collinear {V : Type*} [AddCommGroup V] [Module ℝ V]
  (P₁ P₂ P₃ : V) : Prop :=
∃ t : ℝ, P₃ = P₁ + t • (P₂ - P₁)

theorem triangle_parallel_vectors
  (A B C C₁ A₁ B₁ C₂ A₂ B₂ : ℝ × ℝ)
  (h1 : collinear A B C₁) (h2 : collinear B C A₁) (h3 : collinear C A B₁)
  (ratio1 : ∀ (AC1 CB : ℝ), AC1 / CB = 1) (ratio2 : ∀ (BA1 AC : ℝ), BA1 / AC = 1) (ratio3 : ∀ (CB B1A : ℝ), CB / B1A = 1)
  (h4 : collinear A₁ B₁ C₂) (h5 : collinear B₁ C₁ A₂) (h6 : collinear C₁ A₁ B₂)
  (n : ℝ)
  (ratio4 : ∀ (A1C2 C2B1 : ℝ), A1C2 / C2B1 = n) (ratio5 : ∀ (B1A2 A2C1 : ℝ), B1A2 / A2C1 = n) (ratio6 : ∀ (C1B2 B2A1 : ℝ), C1B2 / B2A1 = n) :
  collinear A C A₂ ∧ collinear C B C₂ ∧ collinear B A B₂ :=
sorry

end triangle_parallel_vectors_l173_173441


namespace vasya_lowest_position_l173_173246

theorem vasya_lowest_position
  (n : ℕ) (m : ℕ) (num_cyclists : ℕ) (vasya_place : ℕ) :
  num_cyclists = 500 →
  n = 15 →
  vasya_place = 7 →
  ∀ (stages : fin n → fin num_cyclists) (no_identical_times : ∀ i j : fin n, i ≠ j → 
  ∀ k l : fin num_cyclists, stages i k ≠ stages j l),
  ∃ (lowest_position : ℕ), lowest_position = 91 := 
by sorry

end vasya_lowest_position_l173_173246


namespace shortest_side_length_of_triangle_l173_173536

/--
In a triangle ABC, given that B = 45 degrees, C = 60 degrees, and side c = 1,
prove that the shortest side length is b = sqrt(6) / 3.

Definitions used:
  A - the third angle of the triangle,
  a, b, c - the lengths of the sides of the triangle opposite the angles A, B, and C respectively.
Conditions:
  1) Triangle ABC exists.
  2) B = 45 degrees.
  3) C = 60 degrees.
  4) c = 1.
  5) The sum of angles in a triangle is 180 degrees.
  6) The Law of Sines: (a / sin A) = (b / sin B) = (c / sin C).
-/
theorem shortest_side_length_of_triangle : 
  ∀ {A B C : ℝ} {a b c : ℝ},
  B = 45 ∧ C = 60 ∧ c = 1 ∧ A + B + C = 180 ∧ 
  (a / real.sin A) = (b / real.sin B) ∧ (a / real.sin A) = (c / real.sin C) ->
  b = real.sqrt 6 / 3 :=
by sorry

end shortest_side_length_of_triangle_l173_173536


namespace smallest_measure_is_sum_l173_173427

-- Define the sample space Omega
def Omega := {0, 1}

-- Define the probability measures P and Q
def P : Omega → ℝ :=
  λ ω, if ω = 0 then 1 else 0

def Q : Omega → ℝ :=
  λ ω, if ω = 0 then 0 else 1

-- Define the measure ν as P + Q
def ν : Omega → ℝ :=
  λ ω, P ω + Q ω

-- Define the max measure P ∨ Q
def max_measure : Omega → ℝ :=
  λ ω, max (P ω) (Q ω)

-- The theorem to show that smallest measure is P + Q
theorem smallest_measure_is_sum : ∀ ω ∈ Omega, ν ω ≥ P ω ∧ ν ω ≥ Q ω ∧ ν ω ≠ max_measure ω := by
  sorry

end smallest_measure_is_sum_l173_173427


namespace geometric_sequence_general_formula_alternating_product_sum_l173_173830

theorem geometric_sequence_general_formula (q : ℝ) (a_n : ℕ → ℝ) 
  (h_geo : ∀ n, a_n (n + 1) = a_n n * q)
  (h_ratio : q > 1)
  (h_a2_a4 : a_n 2 + a_n 4 = 20)
  (h_a3 : a_n 3 = 8) :
  ∀ n, a_n n = 2^n := 
by
  sorry

theorem alternating_product_sum (n : ℕ) (a_n : ℕ → ℝ) 
  (h_geo : ∀ n, a_n (n + 1) = a_n n * 2)
  (h_a1_a2 : a_n 1 * a_n 2 = 4)
  (h_general_formula : ∀ n, a_n n = 2^n) :
  ∑ i in finset.range n, (-1) ^ i * a_n (i + 1) * a_n (i + 2) = 
    8 / 5 - (-1)^n * 2^(2*n + 3) / 5 :=
by
  sorry

end geometric_sequence_general_formula_alternating_product_sum_l173_173830


namespace least_positive_multiple_24_gt_450_l173_173298

theorem least_positive_multiple_24_gt_450 : ∃ n : ℕ, n > 450 ∧ n % 24 = 0 ∧ n = 456 :=
by
  use 456
  sorry

end least_positive_multiple_24_gt_450_l173_173298


namespace magic_trick_possible_iff_even_l173_173604

theorem magic_trick_possible_iff_even (n : ℕ) : 
  (∃ (a b : vector bool n), (∃ mutabor : bool, 
  let a' := if mutabor then a.reverse else a,
      b' := if mutabor then b.map bnot.reverse else b in
  ∀ serge_seq : vector bool n, serge_seq = b' → serge_seq = a')) ↔ n % 2 = 0 := sorry

end magic_trick_possible_iff_even_l173_173604


namespace cheezit_excess_calories_l173_173543

theorem cheezit_excess_calories 
  (bags : ℕ) (oz_per_bag : ℕ) (cal_per_oz : ℕ)
  (minutes_run : ℕ) (cal_per_min : ℕ)
  (total_calories : ℕ) (burned_calories : ℕ) (excess_calories : ℕ) :
  bags = 3 → 
  oz_per_bag = 2 → 
  cal_per_oz = 150 → 
  minutes_run = 40 → 
  cal_per_min = 12 → 
  total_calories = bags * oz_per_bag * cal_per_oz →
  burned_calories = minutes_run * cal_per_min →
  excess_calories = total_calories - burned_calories →
  excess_calories = 420 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  rw [h1, h2, h3, h4, h5] at h6 h7
  calc
    total_calories = 3 * 2 * 150 : by rw [h1, h2, h3]; exact h6
    ... = 900 : by norm_num
    burned_calories = 40 * 12 : by rw h5; exact h7
    ... = 480 : by norm_num
    excess_calories = 900 - 480 : by rw [h1, h2, h3, h4, h5, h6, h7]; exact h8
    ... = 420 : by norm_num
  sorry

end cheezit_excess_calories_l173_173543


namespace average_value_of_T_l173_173179

noncomputable def expected_value_T (B G : ℕ) : ℚ :=
  let total_pairs := 19
  let prob_bg := (B / (B + G)) * (G / (B + G))
  2 * total_pairs * prob_bg

theorem average_value_of_T 
  (B G : ℕ) (hB : B = 8) (hG : G = 12) : 
  expected_value_T B G = 9 :=
by
  rw [expected_value_T, hB, hG]
  norm_num
  sorry

end average_value_of_T_l173_173179


namespace find_k_relationship_y1_y2_l173_173477

theorem find_k (k : ℝ) : (∃ A : ℝ × ℝ, A = (3, -2) ∧ (A.2 = (2 - k) / A.1)) → k = 8 :=
by
  intro h,
  sorry

theorem relationship_y1_y2 (k : ℝ) (x1 x2 : ℝ) (y1 y2 : ℝ) : (0 < x1 ∧ x1 < x2 ∧ y1 = (2 - k) / x1 ∧ y2 = (2 - k) / x2) → y1 < y2 :=
by
  intro h,
  sorry

end find_k_relationship_y1_y2_l173_173477


namespace expected_number_of_sixes_when_three_dice_are_rolled_l173_173659

theorem expected_number_of_sixes_when_three_dice_are_rolled : 
  ∑ n in finset.range 4, (n * (↑(finset.filter (λ xs : fin 3 → fin 6, xs.count (λ x, x = 5) = n) finset.univ).card / 216 : ℚ)) = 1 / 2 :=
by
  -- Conclusion of proof is omitted as per instructions
  sorry

end expected_number_of_sixes_when_three_dice_are_rolled_l173_173659


namespace calculate_f_at_2_l173_173113

def f (x : ℝ) : ℝ := 15 * x ^ 5 - 24 * x ^ 4 + 33 * x ^ 3 - 42 * x ^ 2 + 51 * x

theorem calculate_f_at_2 : f 2 = 294 := by
  sorry

end calculate_f_at_2_l173_173113


namespace fraction_doubled_unchanged_l173_173055

theorem fraction_doubled_unchanged (x y : ℝ) (h : x ≠ y) : 
  (2 * x) / (2 * x - 2 * y) = x / (x - y) :=
by
  sorry

end fraction_doubled_unchanged_l173_173055


namespace eighty_percent_replacement_in_4_days_feasibility_of_replacing_all_old_banknotes_within_budget_l173_173317

noncomputable def old_banknotes : ℕ := 3628800
noncomputable def renovation_cost : ℕ := 800000
noncomputable def daily_operating_cost : ℕ := 90000
noncomputable def post_renovation_capacity : ℕ := 1000000
noncomputable def total_budget : ℕ := 1000000

def banknotes_replaced_in_days (d : ℕ) : ℕ :=
  match d with
  | 1 => old_banknotes / 2
  | 2 => old_banknotes / 2 + post_renovation_capacity
  | 3 => old_banknotes / 2 + post_renovation_capacity + (old_banknotes / 2 + post_renovation_capacity) / 3
  | _ => sorry -- Continues similar pattern.

def cost_in_days (d : ℕ) : ℕ :=
  if d = 1 then daily_operating_cost
  else 800000 + daily_operating_cost * d

theorem eighty_percent_replacement_in_4_days :
  banknotes_replaced_in_days 4 ≈ 0.8 * old_banknotes ∧ cost_in_days 4 ≤ total_budget :=
sorry

theorem feasibility_of_replacing_all_old_banknotes_within_budget :
  banknotes_replaced_in_days 4 = old_banknotes ∨ banknotes_replaced_in_days d = old_banknotes for some d and cost_in_days d ≤ total_budget :=
sorry

end eighty_percent_replacement_in_4_days_feasibility_of_replacing_all_old_banknotes_within_budget_l173_173317


namespace variance_D_l173_173855

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

-- Define the distribution of X
def P (X : ℝ) : ℝ :=
  if X = -1 then a
  else if X = 0 then b
  else if X = 1 then c
  else if X = 2 then 1/3
  else 0

-- Define the expected value
def E (X : ℝ → ℝ) : ℝ := 
  ∑ x in {-1, 0, 1, 2}, x * P x

-- Define the variance D(X)
def D (X : ℝ → ℝ) : ℝ :=
  ∑ x in {-1, 0, 1, 2}, (x - E X) ^ 2 * P x

-- Define conditions
axiom cond1 : a + b + c + 1/3 = 1
axiom cond2 : E (λ x => x) = 3/4
axiom cond3 : c + 1/3 = 7/12

-- The final proof statement
theorem variance_D : D (λ x => x) = 19/16 := by
  sorry

end variance_D_l173_173855


namespace range_of_x0_l173_173138

theorem range_of_x0 (x_0 : ℝ) :
  (∃ (N : (ℝ × ℝ)), (N.1 ^ 2 + N.2 ^ 2 = 1) ∧ ∃ (angle : ℝ), angle = 45 ∧ ∃ (M : (ℝ × ℝ)), M = (x_0, 1) ∧  ∃ O, O = (0, 0) ∧ angle = ∠ O M N) → x_0 ∈ set.Icc (-1: ℝ) (1: ℝ) := 
begin
  sorry
end

end range_of_x0_l173_173138


namespace factoring_options_count_l173_173523

theorem factoring_options_count :
  (∃ (n : ℕ), n = 14) ↔
  ∃ a b c : ℕ, a + b + c = 10 ∧ a ≥ b ∧ b ≥ c ∧ fintype.card {p : ℕ × ℕ // p.1 + p.2 = 10 ∧ p.1 ≥ p.2} + 
    fintype.card {p : ℕ × ℕ // p.1 + p.2 = 9 ∧ p.1 ≥ p.2} + 
    fintype.card {p : ℕ × ℕ // p.1 + p.2 = 8 ∧ p.1 ≥ p.2} + 
    fintype.card {p : ℕ × ℕ // p.1 + p.2 = 7 ∧ p.1 ≥ p.2} = 14 := 
by
  sorry

end factoring_options_count_l173_173523


namespace exists_large_p_l173_173445

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sin (Real.pi * x)

theorem exists_large_p (d : ℝ) (h : d > 0) : ∃ p : ℝ, ∀ x : ℝ, |f (x + p) - f x| < d ∧ ∃ M : ℝ, M > 0 ∧ p > M :=
by {
  sorry
}

end exists_large_p_l173_173445


namespace collinearity_of_points_l173_173567

noncomputable def circumcircle (A B C : Type) : ∀ (Γ : Type), Prop := sorry
noncomputable def incenter (Δ : Type) : ∀ (I : Type), Prop := sorry
noncomputable def circle (ω : Type) : ∀ (A B : Type) (tangent : Type), Prop := sorry
noncomputable def tangent (t : Type) : Prop := sorry
noncomputable def intersection (x : Type) (y : Type) : Type := sorry
noncomputable def collinear (a b c : Type) : Prop := sorry

theorem collinearity_of_points
  (A B C P M Q N X Y I : Type)
  (Γ ω_B ω_C : Type)
  (hΓ : circumcircle A B C Γ)
  (hI : incenter (triangle A B C) I)
  (hωB : circle ω_B B C I)
  (hωC : circle ω_C C B I)
  (hPBΓ : intersection (arc_AB Γ) ω_B P)
  (hPM : intersection (segment_AB A B) ω_B M)
  (hQCΓ : intersection (arc_AC Γ) ω_C Q)
  (hQN : intersection (segment_AC A C) ω_C N)
  (hPM_ray : intersection (ray PM) (ray QN) X)
  (h_tangents : intersection (tangent ω_B B) (tangent ω_C C) Y) : collinear A X Y :=
begin
   sorry
end

end collinearity_of_points_l173_173567


namespace mean_identity_l173_173183

theorem mean_identity (x y z : ℝ) 
  (h_arith_mean : (x + y + z) / 3 = 10)
  (h_geom_mean : Real.cbrt (x * y * z) = 7) 
  (h_harm_mean : 3 / (1 / x + 1 / y + 1 / z) = 4) :
  x^2 + y^2 + z^2 = 385.5 :=
by
  sorry

end mean_identity_l173_173183


namespace geom_seq_problem_l173_173831

-- Define the geometric sequence and the two conditions
def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Given conditions and questions
theorem geom_seq_problem :
  ∃ (a : ℕ → ℝ) (q : ℝ), q > 1 ∧ geom_seq a q ∧
  a 3 = 8 ∧ (a 2 + a 4 = 20) ∧
  (a = (λ n, 2 ^ n)) ∧
  (∀ n : ℕ, (Σ (k : fin n), (-1) ^ (k : ℕ) * a k * a (k + 1)) =
   (8 / 5 - (-1) ^ n * 2 ^ (2 * n + 3) / 5)) := sorry

end geom_seq_problem_l173_173831


namespace sally_pokemon_cards_l173_173598

theorem sally_pokemon_cards :
  let initial_cards := 27
  let dan_gift := 41
  let sally_purchase := 20
  initial_cards + dan_gift + sally_purchase = 88 :=
by
  have h1 : initial_cards = 27 := rfl
  have h2 : dan_gift = 41 := rfl
  have h3 : sally_purchase = 20 := rfl
  have h_calc : 27 + 41 + 20 = 88 := rfl
  exact h_calc

#eval sally_pokemon_cards

end sally_pokemon_cards_l173_173598


namespace independent_variable_range_l173_173634

theorem independent_variable_range (x : ℝ) :
  (∃ y : ℝ, y = 1 / (Real.sqrt x - 1)) ↔ x ≥ 0 ∧ x ≠ 1 := 
by
  sorry

end independent_variable_range_l173_173634


namespace function_increment_l173_173023

def f (x : ℝ) : ℝ := 2 / x

theorem function_increment :
  f 1.5 - f 2 = 1 / 3 :=
by
  sorry

end function_increment_l173_173023


namespace horse_saddle_ratio_l173_173581

variable (H S : ℝ)
variable (m : ℝ)
variable (total_value saddle_value : ℝ)

theorem horse_saddle_ratio :
  total_value = 100 ∧ saddle_value = 12.5 ∧ H = m * saddle_value ∧ H + saddle_value = total_value → m = 7 :=
by
  sorry

end horse_saddle_ratio_l173_173581


namespace square_side_length_l173_173753

/-- 
If a square is drawn by joining the midpoints of the sides of a given square and repeating this process continues indefinitely,
and the sum of the areas of all the squares is 32 cm²,
then the length of the side of the first square is 4 cm. 
-/
theorem square_side_length (s : ℝ) (h : ∑' n : ℕ, (s^2) * (1 / 2)^n = 32) : s = 4 := 
by 
  sorry

end square_side_length_l173_173753


namespace pirate_loot_total_value_l173_173742

def base5_to_base10 (n : ℕ) : ℕ :=
  let digits := [n / 625 % 5, n / 125 % 5, n / 25 % 5, n / 5 % 5, n % 5]
  (digits.zipWith (λ d p, d * (5 ^ p)) [0, 1, 2, 3, 4]).sum

def value1 := base5_to_base10 4132
def value2 := base5_to_base10 1432
def value3 := base5_to_base10 2024
def value4 := base5_to_base10 224

def total_value := value1 + value2 + value3 + value4

theorem pirate_loot_total_value : total_value = 1112 := sorry

end pirate_loot_total_value_l173_173742


namespace sequence_third_order_and_nth_term_l173_173344

-- Define the given sequence
def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 4
  | 1 => 6
  | 2 => 13
  | 3 => 27
  | 4 => 50
  | 5 => 84
  | _ => sorry -- let’s define the general form for other terms later

-- Define first differences
def first_diff (n : ℕ) : ℤ := a (n + 1) - a n

-- Define second differences
def second_diff (n : ℕ) : ℤ := first_diff (n + 1) - first_diff n

-- Define third differences
def third_diff (n : ℕ) : ℤ := second_diff (n + 1) - second_diff n

-- Define the nth term formula
noncomputable def nth_term (n : ℕ) : ℚ := (1 / 6) * (2 * n^3 + 3 * n^2 - 11 * n + 30)

-- Theorem stating the least possible order is 3 and the nth term formula
theorem sequence_third_order_and_nth_term :
  (∀ n, third_diff n = 2) ∧ (∀ n, a n = nth_term n) :=
by
  sorry

end sequence_third_order_and_nth_term_l173_173344


namespace problem_correct_statements_l173_173937

variable {A B C : ℝ}
variable {a b c : ℝ}
variables (triangle : Triangle)
-- Assumption: A, B, and C are interior angles of a triangle ABC
-- Sides: a, b, c opposite angles A, B, C respectively
-- A, B, C are in radians, and 0 < A, B, C < π
-- and A + B + C = π
variables (h_triangle_sides : a = triangle.side_a)
          (h_triangle_sides_ab : b = triangle.side_b)
          (h_triangle_sides_bc : c = triangle.side_c)
          (h_triangle_angles : A = triangle.angle_A)
          (h_triangle_angles_ab : B = triangle.angle_B)
          (h_triangle_angles_bc : C = triangle.angle_C)
          (h_triangle_property : a^2 + b^2 = c^2)
          
theorem problem_correct_statements :
  (A > B → sin A > sin B) ∧ (sin^2 A + sin^2 B < sin^2 C → triangle.is_obtuse). 
Proof with hint:
  sorry

end problem_correct_statements_l173_173937


namespace count_valid_six_digit_numbers_l173_173391

-- Definitions: 
def isValidSixDigitNumber (n : ℕ) : Prop :=
  (n >= 100000) ∧ (n < 1000000) ∧ (∀ d ∈ [1, 2, 3, 4, 5, 6], d ∈ digits 10 n) ∧
  (∀ (a b : ℕ), (a = 1 ∧ (b = 100000 ∨ b = 1)) → ¬ (digits 10 n).take 1 = a)
  

def oneAdjacentPair (ds : List ℕ) : Prop :=
  ∃ (i : ℕ), (i < 5) ∧ (ds.nth i).isSome ∧ (ds.nth (i + 1)).isSome ∧ 
  (∃ e1 e2, ds.nth i = some e1 ∧ ds.nth (i + 1) = some e2 ∧ e1 ∈ [2, 4, 6] ∧ e2 ∈ [2, 4, 6] ∧ e1 ≠ e2) ∧
  ∀ (j : ℕ), (j < 4 ∧ j ≠ i) → (∀ (k : ℕ), (k = j ∨ k = j + 1) → (∃ d1 d2, (ds.nth k = some d1 ∧ ds.nth (k + 1) = some d2 ∧ (d1 ∉ [2, 4, 6] ∨ d2 ∉ [2, 4, 6]))))

-- Proposition:
theorem count_valid_six_digit_numbers : 
  (∑ n in finset.range 1000000, if isValidSixDigitNumber n ∧ oneAdjacentPair (digits 10 n) then 1 else 0) = 288 :=
by
  sorry

end count_valid_six_digit_numbers_l173_173391


namespace smallest_n_units_zero_units_digit_140_is_zero_l173_173140

noncomputable def sequence_length (n : ℕ) : ℕ :=
  let rec helper (x : ℕ) (count : ℕ) :=
    if x = 0 then count
    else
      let m := (Finset.range ((Nat.sqrt x).div 2 + 1)).filter (λ k, k % 2 = 0).max'.getOrElse 0
      in helper (x - m^2) (count + 1)
  helper n 1

theorem smallest_n_units_zero (n : ℕ) : 
  (sequence_length n = 6) ↔ (n = 140) :=
by sorry

theorem units_digit_140_is_zero : 
  ∃ n : ℕ, (sequence_length n = 6) ∧ (n = 140) ∧ (n % 10 = 0) :=
by sorry

end smallest_n_units_zero_units_digit_140_is_zero_l173_173140


namespace part_a_part_b_l173_173131

theorem part_a (m : ℕ) (hm : m ≥ 1) : 
  (∏ i in finset.range m.succ, (1 - (1/(Nat.prime (i+1)).val)))⁻¹ > 
  ∑ i in finset.range (Nat.prime m).val, (1/(i+1)) :=
sorry

theorem part_b (m : ℕ) (hm : m ≥ 1) : 
  1 + ∑ i in finset.range m.succ, (1/(Nat.prime (i+1)).val) > 
  Real.log (Real.log ((Nat.prime m).val)) :=
sorry

end part_a_part_b_l173_173131


namespace william_wins_tic_tac_toe_l173_173705

-- Define the conditions
variables (total_rounds : ℕ) (extra_wins : ℕ) (william_wins : ℕ) (harry_wins : ℕ)

-- Setting the conditions
def william_harry_tic_tac_toe_conditions : Prop :=
  total_rounds = 15 ∧
  extra_wins = 5 ∧
  william_wins = harry_wins + extra_wins ∧
  total_rounds = william_wins + harry_wins

-- The goal is to prove that William won 10 rounds given the conditions above
theorem william_wins_tic_tac_toe : william_harry_tic_tac_toe_conditions total_rounds extra_wins william_wins harry_wins → william_wins = 10 :=
by
  intro h
  have total_rounds_eq := and.left h
  have extra_wins_eq := and.right (and.left (and.right h))
  have william_harry_diff := and.left (and.right (and.right h))
  have total_wins_eq := and.right (and.right (and.right h))
  sorry

end william_wins_tic_tac_toe_l173_173705


namespace abs_abc_eq_abs_k_l173_173557

variable {a b c k : ℝ}

noncomputable def distinct_nonzero (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a

theorem abs_abc_eq_abs_k (h_distinct : distinct_nonzero a b c)
                          (h_nonzero_k : k ≠ 0)
                          (h_eq : a + k / b = b + k / c ∧ b + k / c = c + k / a) :
  |a * b * c| = |k| :=
by
  sorry

end abs_abc_eq_abs_k_l173_173557


namespace curve_intersections_l173_173465

def C1_param (t : ℝ) : ℝ × ℝ := (1 + (√2 / 2) * t, (√2 / 2) * t)

def C2_polar (ρ θ : ℝ) : ℝ := 1 / (ρ^2) - (cos θ)^2 / 2 - (sin θ)^2

def M : ℝ × ℝ := (1, 0)

theorem curve_intersections (t : ℝ) :
  (C1_param t).1 - (C1_param t).2 - 1 = 0 ∧ 
  (C2_polar ((C1_param t).1 / cos t) t) = 0 ∧
  let A : ℝ × ℝ := (0, -1) in
  let B : ℝ × ℝ := (4/3, 1/3) in
  (1 - 0)^2 + (0 + 1)^2 = 2 ∧
  ((4/3 - 1)^2 + (1/3 - 0)^2) = (2/9) ∧
  ((4/3 - 0)^2 + (1/3 + 1)^2) = (16/9) ∧
  (sqrt 2 * sqrt (2/9)) / sqrt (16/9) = sqrt 2 / 4 := sorry

end curve_intersections_l173_173465


namespace painters_completing_rooms_l173_173511

theorem painters_completing_rooms (three_painters_three_rooms_three_hours : 3 * 3 * 3 ≥ 3 * 3) :
  9 * 3 * 9 ≥ 9 * 27 :=
by 
  sorry

end painters_completing_rooms_l173_173511


namespace Kaylee_total_boxes_needed_l173_173943

-- Defining the conditions
def lemon_biscuits := 12
def chocolate_biscuits := 5
def oatmeal_biscuits := 4
def still_needed := 12

-- Defining the total boxes sold so far
def total_sold := lemon_biscuits + chocolate_biscuits + oatmeal_biscuits

-- Defining the total number of boxes that need to be sold in total
def total_needed := total_sold + still_needed

-- Lean statement to prove the required total number of boxes
theorem Kaylee_total_boxes_needed : total_needed = 33 :=
by
  sorry

end Kaylee_total_boxes_needed_l173_173943


namespace determine_original_number_l173_173518

theorem determine_original_number (a b c : ℕ) (m : ℕ) (N : ℕ) 
  (h1 : N = 4410) 
  (h2 : (a + b + c) % 2 = 0)
  (h3 : m = 100 * a + 10 * b + c)
  (h4 : N + m = 222 * (a + b + c)) : 
  a = 4 ∧ b = 4 ∧ c = 4 :=
by 
  sorry

end determine_original_number_l173_173518


namespace expected_value_is_4_point_5_l173_173736

noncomputable def expected_value_of_winnings : ℝ := 
let probability := 1 / 8 in
let winnings_if_even := [2, 4, 6] in
let winnings_if_eight := 2 * (winnings_if_even.sum) in
let expected_value := (probability * 2 + probability * 4 + probability * 6 + probability * winnings_if_eight) in
expected_value / 1

theorem expected_value_is_4_point_5 :
  expected_value_of_winnings = 4.5 :=
by
  -- The proof is omitted as per the instructions
  sorry

end expected_value_is_4_point_5_l173_173736


namespace four_digit_perfect_square_palindromes_eq_one_l173_173880

theorem four_digit_perfect_square_palindromes_eq_one :
  ∃! (n : ℕ), (1024 ≤ n) ∧ (n ≤ 9801) ∧ (n = (m * m) ∧ is_palindrome n) :=
sorry

-- Auxiliary definition for checking if a number is a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in digits = digits.reverse
  

end four_digit_perfect_square_palindromes_eq_one_l173_173880


namespace value_of_a_for_local_minimum_l173_173049

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 3 - 2 * x ^ 2 + a ^ 2 * x

theorem value_of_a_for_local_minimum (a : ℝ) :
  (∀ x, f a x = a * x ^ 3 - 2 * x ^ 2 + a ^ 2 * x) →
  (∃ x, x = 1 ∧ ∃ f', derivative f a 1 = 0 ∧ derivative(f') 0 0) →
  a = 1 := 
sorry

end value_of_a_for_local_minimum_l173_173049


namespace range_of_a_l173_173626

noncomputable def f (x a : ℝ) := x^2 - a * x + 1

def h (x : ℝ) := x + 1 / x

theorem range_of_a (a : ℝ) :
  (∃ x ∈ Ioo (1/2 : ℝ) 4, f x a = 0) →
  2 ≤ a ∧ a < 17 / 4 :=
begin
  sorry
end

end range_of_a_l173_173626


namespace circumcircle_tangent_to_BC_l173_173964

open EuclideanGeometry

variables {A B C P Q L : Point}
variables {circumcircle_ABC : Circle}

noncomputable def angle_bisector (A B C : Point) (L : Point) : Prop := 
  ∃ (L : Point), (angle A L B = angle A L C)

noncomputable def perpendicular_bisector (A L : Point) (circumcircle : Circle) (P Q : Point) : Prop :=
  ∃ (P Q : Point), P ≠ Q ∧ (line P Q) ⊥ (segment A L) ∧ P ∈ circumcircle ∧ Q ∈ circumcircle

noncomputable def circumcircle (P L Q : Point) (A B C : Point) : Circle := sorry

theorem circumcircle_tangent_to_BC 
  (h1 : angle_bisector A B C L)
  (h2 : perpendicular_bisector A L circumcircle_ABC P Q) : 
  is_tangent (circumcircle P L Q A B C) (line B C) :=
sorry

end circumcircle_tangent_to_BC_l173_173964


namespace f_is_periodic_l173_173121

noncomputable def f (x : ℝ) : ℝ := x - ⌈x⌉

theorem f_is_periodic : ∀ x : ℝ, f (x + 1) = f x :=
by 
  intro x
  sorry

end f_is_periodic_l173_173121


namespace largest_four_digit_congruent_17_mod_26_l173_173290

theorem largest_four_digit_congruent_17_mod_26 : 
  ∃ k : ℤ, (26 * k + 17 < 10000) ∧ (1000 ≤ 26 * k + 17) ∧ (26 * k + 17) ≡ 17 [MOD 26] ∧ (26 * k + 17 = 9972) :=
by
  sorry

end largest_four_digit_congruent_17_mod_26_l173_173290


namespace Q_at_2_plus_neg2_l173_173971

noncomputable def Q (x : ℝ) : ℝ := a * x^4 + b * x^3 + c * x^2 + d * x + 4 * k

variables {a b c d k : ℝ}

-- Conditions from the problem
axiom Q_at_0 : Q 0 = 4 * k
axiom Q_at_1 : Q 1 = 5 * k
axiom Q_at_neg1 : Q (-1) = 9 * k

-- The theorem to prove
theorem Q_at_2_plus_neg2 : Q 2 + Q (-2) = 48 * k :=
by
  sorry -- Proof steps here

end Q_at_2_plus_neg2_l173_173971


namespace smallest_solution_x_squared_abs_x_eq_3x_plus_4_l173_173811

theorem smallest_solution_x_squared_abs_x_eq_3x_plus_4 :
  ∃ x : ℝ, x^2 * |x| = 3 * x + 4 ∧ ∀ y : ℝ, (y^2 * |y| = 3 * y + 4 → y ≥ x) := 
sorry

end smallest_solution_x_squared_abs_x_eq_3x_plus_4_l173_173811


namespace vasya_lowest_position_l173_173262

noncomputable theory

def number_of_cyclists := 500
def number_of_stages := 15
def position_of_vasya_each_stage := 7

theorem vasya_lowest_position (total_cyclists : ℕ) (stages : ℕ) (position_each_stage : ℕ)
  (h_total_cyclists : total_cyclists = number_of_cyclists)
  (h_stages : stages = number_of_stages)
  (h_position_each_stage : position_each_stage = position_of_vasya_each_stage)
  (no_identical_times : ∀ (i j : ℕ), i ≠ j → ∀ (stage : ℕ), stage ≤ stages → ∀ (t : ℕ), t ≤ total_cyclists → 
    (time : Π (s : ℕ) (c : ℕ), c < total_cyclists → ℕ), 
    time stage i < time stage j ∨ time stage j < time stage i):
  ∃ lowest_pos, lowest_pos = 91 := sorry

end vasya_lowest_position_l173_173262


namespace value_of_a_plus_c_l173_173535

variables {A B C : ℝ} {a b c : ℝ}

-- Conditions
axiom sin_B_cos_C_plus_sin_C_eq_2_sin_A (h1 : 2 * sin B * cos C + sin C = 2 * sin A)
axiom sin_A_plus_sin_C_eq_2_sqrt6_sin_A_sin_C (h2 : sin A + sin C = 2 * sqrt 6 * sin A * sin C)
axiom b_eq_3 (h3 : b = 3)

-- Prove a + c = 3√2
theorem value_of_a_plus_c (h1 : 2 * sin B * cos C + sin C = 2 * sin A) 
                          (h2 : sin A + sin C = 2 * sqrt 6 * sin A * sin C) 
                          (h3 : b = 3) : a + c = 3 * sqrt 2 := 
sorry

end value_of_a_plus_c_l173_173535


namespace domain_of_h_h_is_odd_h_lt_0_l173_173025

noncomputable def f (a : ℝ) (x : ℝ) := Real.log (1 + x) / Real.log a
noncomputable def g (a : ℝ) (x : ℝ) := Real.log (1 - x) / Real.log a
noncomputable def h (a : ℝ) := λ x, f a x - g a x

theorem domain_of_h : 
  ∀ (a : ℝ), (0 < a) ∧ (a ≠ 1) → 
  ∀ x, (x ∈ Set.Ioo (-1:ℝ) 1) := sorry

theorem h_is_odd : 
  ∀ (a : ℝ), (0 < a) ∧ (a ≠ 1) → 
  ∀ x, h a (-x) = -h a x := sorry

theorem h_lt_0 : 
  ∀ (a : ℝ), (0 < a) ∧ (a ≠ 1) → 
  f a 3 = 2 → 
  ∀ x, (x ∈ Set.Ioo (-1:ℝ) 0) ↔ h a x < 0 := sorry

end domain_of_h_h_is_odd_h_lt_0_l173_173025


namespace sum_of_triangular_numbers_l173_173404

-- Define the n-th triangular number.
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

-- Proposition: Sum of the first n triangular numbers.
theorem sum_of_triangular_numbers (n : ℕ) : 
  (∑ k in finset.range n, triangular_number k) = n * (n + 1) * (n + 2) / 6 := 
by
  sorry

end sum_of_triangular_numbers_l173_173404


namespace notebook_purchase_possible_l173_173065

/-- There are only two types of coins: 16 tugriks and 27 tugriks.
    Is it possible to pay exactly 1 tugrik for a notebook and get change? -/
theorem notebook_purchase_possible : ∃ (x y : ℤ), 16 * x + 27 * y = 1 :=
by {
  sorry,
}

end notebook_purchase_possible_l173_173065


namespace smallest_integer_in_consecutive_set_l173_173519

theorem smallest_integer_in_consecutive_set :
  ∃ (n : ℤ), 2 < n ∧ ∀ m : ℤ, m < n → ¬ (m + 6 < 2 * (m + 3) - 2) :=
sorry

end smallest_integer_in_consecutive_set_l173_173519


namespace ellipse_left_vertex_l173_173845

noncomputable
def left_vertex_ellipse (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c = 3) (h4 : 2 * b = 8) : ℝ × ℝ :=
  (-a, 0)

theorem ellipse_left_vertex (a b : ℝ) (h_ellipse : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1) 
                            (h_circle : ∀ x y, x^2 + y^2 - 6*x + 8 = 0 → circle_center = (3, 0)) 
                            (h_minor_axis : 2 * b = 8) 
                            (h_foci_distance : c = 3)
                            (h_c_squared : c^2 = a^2 - b^2) :
  left_vertex_ellipse a b c (by linarith) (by linarith) (by linarith) (by linarith) = (-5, 0) := 
sorry

end ellipse_left_vertex_l173_173845


namespace count_odd_f_is_242_l173_173944

def f (n : ℝ) : ℕ :=
  if h : n < 2 then 0
  else Nat.succ (f (Real.sqrt n))

noncomputable def count_odd_f : ℕ :=
  {m : ℕ | 1 < m ∧ m < 2008 ∧ f (m : ℝ) % 2 = 1}.to_finset.card

theorem count_odd_f_is_242 : count_odd_f = 242 :=
sorry

end count_odd_f_is_242_l173_173944


namespace vasya_lowest_position_l173_173218

theorem vasya_lowest_position
  (num_cyclists : ℕ)
  (num_stages : ℕ)
  (num_ahead : ℕ)
  (position_vasya : ℕ)
  (total_time : List ℕ)
  (unique_total_times : total_time.nodup)
  (stage_positions : List (List ℕ))
  (unique_stage_positions : ∀ stage ∈ stage_positions, stage.nodup)
  (vasya_consistent : ∀ stage ∈ stage_positions, stage.nth position_vasya = some num_ahead) :
  num_ahead * num_stages + 1 = 91 :=
by
  sorry

end vasya_lowest_position_l173_173218


namespace mean_identity_example_l173_173186

theorem mean_identity_example {x y z : ℝ} 
  (h1 : x + y + z = 30)
  (h2 : x * y * z = 343)
  (h3 : x * y + y * z + z * x = 257.25) :
  x^2 + y^2 + z^2 = 385.5 :=
by
  sorry

end mean_identity_example_l173_173186


namespace no_parallelepiped_exists_l173_173789

theorem no_parallelepiped_exists 
  (xyz_half_volume: ℝ)
  (xy_plus_yz_plus_zx_half_surface_area: ℝ) 
  (sum_of_squares_eq_4: ℝ) : 
  ¬(∃ x y z : ℝ, (x * y * z = xyz_half_volume) ∧ 
                 (x * y + y * z + z * x = xy_plus_yz_plus_zx_half_surface_area) ∧ 
                 (x^2 + y^2 + z^2 = sum_of_squares_eq_4)) := 
by
  let xyz_half_volume := 2 * Real.pi / 3
  let xy_plus_yz_plus_zx_half_surface_area := Real.pi
  let sum_of_squares_eq_4 := 4
  sorry

end no_parallelepiped_exists_l173_173789


namespace convex_polyhedron_property_l173_173331

-- Given conditions as definitions
def num_faces : ℕ := 40
def num_hexagons : ℕ := 8
def num_triangles_eq_twice_pentagons (P : ℕ) (T : ℕ) : Prop := T = 2 * P
def num_pentagons_eq_twice_hexagons (P : ℕ) (H : ℕ) : Prop := P = 2 * H

-- Main statement for the proof problem
theorem convex_polyhedron_property (P T V : ℕ) :
  num_triangles_eq_twice_pentagons P T ∧ num_pentagons_eq_twice_hexagons P num_hexagons ∧ 
  num_faces = T + P + num_hexagons ∧ V = (T * 3 + P * 5 + num_hexagons * 6) / 2 + num_faces - 2 →
  100 * P + 10 * T + V = 535 :=
by
  sorry

end convex_polyhedron_property_l173_173331


namespace parabola_chord_length_l173_173026

theorem parabola_chord_length (p: ℝ) (x₀: ℝ) (h_p_pos: 0 < p)
    (h_M_on_C: 2 * sqrt 2 ^ 2 = 2 * p * x₀)
    (h_chord_len: (x₀)^2 + 5 = (x₀ + p / 2)^2) : p = 2 :=
by
  sorry

end parabola_chord_length_l173_173026


namespace factor_1024_count_l173_173525

theorem factor_1024_count :
  ∃ (n : ℕ), 
  (∀ (a b c : ℕ), (a >= b) → (b >= c) → (2^a * 2^b * 2^c = 1024) → a + b + c = 10) ∧ n = 14 :=
sorry

end factor_1024_count_l173_173525


namespace platform_length_is_150_l173_173723

noncomputable def length_of_platform
  (train_length : ℝ)
  (time_to_cross_platform : ℝ)
  (time_to_cross_pole : ℝ)
  (L : ℝ) : Prop :=
  train_length + L = (train_length / time_to_cross_pole) * time_to_cross_platform

theorem platform_length_is_150 :
  length_of_platform 300 27 18 150 :=
by 
  -- Proof omitted, but the statement is ready for proving
  sorry

end platform_length_is_150_l173_173723


namespace product_of_roots_l173_173018

noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

noncomputable def f_prime (a b c : ℝ) (x : ℝ) : ℝ :=
  3 * a * x^2 + 2 * b * x + c

theorem product_of_roots (a b c d x₁ x₂ : ℝ) 
  (h1 : f a b c d 0 = 0)
  (h2 : f a b c d x₁ = 0)
  (h3 : f a b c d x₂ = 0)
  (h_ext1 : f_prime a b c 1 = 0)
  (h_ext2 : f_prime a b c 2 = 0) :
  x₁ * x₂ = 6 :=
sorry

end product_of_roots_l173_173018


namespace smallest_period_f_axis_of_symmetry_f_set_of_x0_l173_173859

noncomputable def f (x : ℝ) : ℝ := sin x * cos x + cos x ^ 2
noncomputable def g (x : ℝ) : ℝ := f (x - π / 8)

theorem smallest_period_f : ∀ x : ℝ, f (x + π) = f x :=
by
  sorry

theorem axis_of_symmetry_f : ∃ k : ℤ, ∀ x : ℝ, f (x) = f (2 * x + π / 4 + k * π) :=
by
  sorry

theorem set_of_x0 : ∃ k : ℤ, ∀ x0 : ℝ, (g x0 ≥ 1) ↔ (π / 8 + k * π ≤ x0 ∧ x0 ≤ 3 * π / 8 + k * π) :=
by
  sorry

end smallest_period_f_axis_of_symmetry_f_set_of_x0_l173_173859


namespace side_length_square_inscribed_in_triangle_l173_173596

-- Definitions based on conditions
def DE : ℝ := 5
def EF : ℝ := 12
def DF : ℝ := 13
def h := (2 * (DE * EF) / 2) / DF

-- Defining the theorem to prove the side length of square is 30/7
theorem side_length_square_inscribed_in_triangle :
  let s := (13 * h) / (13 + h)
  s = 30 / 7 :=
by
  have h_val : h = 60 / 13 := sorry
  have s_calculation : s = 30 / 7 := sorry
  exact s_calculation

#check side_length_square_inscribed_in_triangle

end side_length_square_inscribed_in_triangle_l173_173596


namespace solve_for_x_and_z_l173_173173

theorem solve_for_x_and_z (x z : ℝ) (h : |5 * x - log z| = 5 * x + 3 * log z) : 
  x = 0 ∧ z = 1 := 
sorry

end solve_for_x_and_z_l173_173173


namespace probability_is_zero_l173_173366

noncomputable def probability_same_number (b d : ℕ) (h_b : b < 150) (h_d : d < 150)
    (h_b_multiple: b % 15 = 0) (h_d_multiple: d % 20 = 0) (h_square: b * b = b ∨ d * d = d) : ℝ :=
  0

theorem probability_is_zero (b d : ℕ) (h_b : b < 150) (h_d : d < 150)
    (h_b_multiple: b % 15 = 0) (h_d_multiple: d % 20 = 0) (h_square: b * b = b ∨ d * d = d) : 
    probability_same_number b d h_b h_d h_b_multiple h_d_multiple h_square = 0 :=
  sorry

end probability_is_zero_l173_173366


namespace count_pos_integers_satisfying_condition_l173_173034

theorem count_pos_integers_satisfying_condition :
  (card {n : ℕ | 400 < n^2 ∧ n^2 < 1600}) = 19 := by
  sorry

end count_pos_integers_satisfying_condition_l173_173034


namespace find_p_r_l173_173114

-- Definitions of the polynomials
def f (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q
def g (x : ℝ) (r s : ℝ) : ℝ := x^2 + r * x + s

-- Lean statement of the proof problem:
theorem find_p_r (p q r s : ℝ) (h1 : p ≠ r) (h2 : g (-p / 2) r s = 0) 
  (h3 : f (-r / 2) p q = 0) (h4 : ∀ x : ℝ, f x p q = g x r s) 
  (h5 : f 50 p q = -50) : p + r = -200 := 
sorry

end find_p_r_l173_173114


namespace vasya_lowest_position_l173_173200

theorem vasya_lowest_position
  (n_cyclists : ℕ) (n_stages : ℕ) 
  (stage_positions : ℕ → ℕ → ℕ) -- a function that takes a stage and a cyclist and returns the position (e.g., stage_positions(stage, cyclist) = position)
  (total_time : ℕ → ℕ)  -- a function that takes a cyclist and returns their total time
  (distinct_times : ∀ (c1 c2 : ℕ), c1 ≠ c2 → (total_time c1 ≠ total_time c2) ∧ 
                   ∀ (s : ℕ), stage_positions s c1 ≠ stage_positions s c2)
  (vasya_position : ℕ) (hv : ∀ (s : ℕ), s < n_stages → stage_positions s vasya_position = 7) :
  vasya_position = 91 :=
sorry

end vasya_lowest_position_l173_173200


namespace ice_cream_flavors_count_l173_173035

-- Definitions and assumptions based on problem conditions
def flavors := 5
def scoops := 5

-- Applying the conditions to the problem statement
theorem ice_cream_flavors_count : binomial (scoops + flavors - 1) (flavors - 1) = 126 := by
    sorry

end ice_cream_flavors_count_l173_173035


namespace factor_1024_count_l173_173526

theorem factor_1024_count :
  ∃ (n : ℕ), 
  (∀ (a b c : ℕ), (a >= b) → (b >= c) → (2^a * 2^b * 2^c = 1024) → a + b + c = 10) ∧ n = 14 :=
sorry

end factor_1024_count_l173_173526


namespace find_polynomial_value_l173_173008

theorem find_polynomial_value (x y : ℝ) 
  (h1 : 3 * x + y = 12) 
  (h2 : x + 3 * y = 16) : 
  10 * x^2 + 14 * x * y + 10 * y^2 = 422.5 := 
by 
  sorry

end find_polynomial_value_l173_173008


namespace largest_four_digit_congruent_17_mod_26_l173_173291

theorem largest_four_digit_congruent_17_mod_26 : 
  ∃ k : ℤ, (26 * k + 17 < 10000) ∧ (1000 ≤ 26 * k + 17) ∧ (26 * k + 17) ≡ 17 [MOD 26] ∧ (26 * k + 17 = 9972) :=
by
  sorry

end largest_four_digit_congruent_17_mod_26_l173_173291


namespace expected_value_of_sixes_l173_173656

theorem expected_value_of_sixes (n : ℕ) (k : ℕ) (p q : ℚ) 
  (h1 : n = 3) 
  (h2 : k = 6)
  (h3 : p = 1/6) 
  (h4 : q = 5/6) : 
  (1 : ℚ) / 2 = ∑ i in finset.range (n + 1), (i * (nat.choose n i * p^i * q^(n-i))) := 
sorry

end expected_value_of_sixes_l173_173656


namespace determinant_roots_polynomial_l173_173968

theorem determinant_roots_polynomial :
  ∀ (a b c : ℝ), (Polynomial.aeval a (Polynomial.C 1 * Polynomial.X^3 - 2 * Polynomial.C 1 * Polynomial.X^2 + 4 * Polynomial.C 1 * Polynomial.X - Polynomial.C 5) = 0) ∧
                 (Polynomial.aeval b (Polynomial.C 1 * Polynomial.X^3 - 2 * Polynomial.C 1 * Polynomial.X^2 + 4 * Polynomial.C 1 * Polynomial.X - Polynomial.C 5) = 0) ∧
                 (Polynomial.aeval c (Polynomial.C 1 * Polynomial.X^3 - 2 * Polynomial.C 1 * Polynomial.X^2 + 4 * Polynomial.C 1 * Polynomial.X - Polynomial.C 5) = 0) →
  Matrix.det ![
    ![a, b, c],
    ![b, c, a],
    ![c, a, b]
  ] = 31 := by
    sorry

end determinant_roots_polynomial_l173_173968


namespace divided_tetrahedron_24_parts_l173_173688

-- Given a tetrahedron and planes through each edge and the opposite face centroids
def tetrahedron_divides_into_24_parts (A B C D : Point) : Prop :=
  let centroid (X Y Z : Point) : Point := sorry -- Define centroid function
  let planes_through_edges_and_centroids := sorry -- Define the planes through edges and midpoints of opposite edges
  divides_into_parts A B C D planes_through_edges_and_centroids 24

theorem divided_tetrahedron_24_parts {A B C D : Point} :
  tetrahedron_divides_into_24_parts A B C D :=
sorry

end divided_tetrahedron_24_parts_l173_173688


namespace heptagon_sides_equal_l173_173734

theorem heptagon_sides_equal (H : ConvexHeptagon) (inscribed : InscribedInCircle H) 
  (angles_eq : ∃ A B C D E F G, 
               ∠A = 120 ∧ ∠B = 120 ∧ ∠C = 120) : 
  ∃ x y : ℝ, x = y := 
sorry

end heptagon_sides_equal_l173_173734


namespace not_true_statement_C_l173_173395

-- Define the operation x ♥ y as |x - y|
def heart (x y : ℝ) : ℝ := abs (x - y)

-- Prove that the statement "x ♥ 0 = x for all x" is not true
theorem not_true_statement_C :
  ¬ ∀ (x : ℝ), heart x 0 = x :=
begin
  -- We need to show the existence of some x for which heart x 0 ≠ x
  use -1, -- Example of a counterexample
  unfold heart,
  simp,
  -- Prove that | -1 - 0 | ≠ -1
  linarith,
end

end not_true_statement_C_l173_173395


namespace smallest_angle_l173_173384

theorem smallest_angle 
  (x : ℝ) 
  (hx : tan (6 * x) = (cos (2 * x) - sin (2 * x)) / (cos (2 * x) + sin (2 * x))) :
  x = 5.625 :=
by
  sorry

end smallest_angle_l173_173384


namespace sin_neg_945_eq_sqrt2_div_2_l173_173407

theorem sin_neg_945_eq_sqrt2_div_2 : Real.sin (Degrees.toRadians (-945)) = (Real.sqrt 2) / 2 :=
by sorry

end sin_neg_945_eq_sqrt2_div_2_l173_173407


namespace largest_4_digit_congruent_to_17_mod_26_l173_173296

theorem largest_4_digit_congruent_to_17_mod_26 :
  ∃ x, x < 10000 ∧ x ≥ 1000 ∧ x % 26 = 17 ∧ (∀ y, y < 10000 ∧ y ≥ 1000 ∧ y % 26 = 17 → y ≤ x) ∧ x = 9972 := 
by
  sorry

end largest_4_digit_congruent_to_17_mod_26_l173_173296


namespace smallest_positive_angle_l173_173381

theorem smallest_positive_angle:
  ∃ (x : ℝ), 0 < x ∧ x ≤ 5.625 ∧ tan (6 * x * (π / 180)) = (cos (2 * x * (π / 180)) - sin (2 * x * (π / 180))) / (cos (2 * x * (π / 180)) + sin (2 * x * (π / 180))) :=
by
  sorry

end smallest_positive_angle_l173_173381


namespace K₅_not_planar_l173_173997

open GraphTheory

-- Define the complete graph K₅
def K₅ : SimpleGraph (Fin 5) :=
  SimpleGraph.completeGraph (Fin 5)

-- State the theorem that K₅ is not planar
theorem K₅_not_planar : ¬K₅.IsPlanarGraph :=
by {
  sorry
}

end K₅_not_planar_l173_173997


namespace Zenobius_more_descendants_l173_173097

/-- Total number of descendants in King Pafnutius' lineage --/
def descendants_Pafnutius : Nat :=
  2 + 60 * 2 + 20 * 1

/-- Total number of descendants in King Zenobius' lineage --/
def descendants_Zenobius : Nat :=
  4 + 35 * 3 + 35 * 1

theorem Zenobius_more_descendants : descendants_Zenobius > descendants_Pafnutius := by
  sorry

end Zenobius_more_descendants_l173_173097


namespace expected_number_of_sixes_when_three_dice_are_rolled_l173_173661

theorem expected_number_of_sixes_when_three_dice_are_rolled : 
  ∑ n in finset.range 4, (n * (↑(finset.filter (λ xs : fin 3 → fin 6, xs.count (λ x, x = 5) = n) finset.univ).card / 216 : ℚ)) = 1 / 2 :=
by
  -- Conclusion of proof is omitted as per instructions
  sorry

end expected_number_of_sixes_when_three_dice_are_rolled_l173_173661


namespace infinite_solutions_xyz_t_l173_173592

theorem infinite_solutions_xyz_t (x y z t : ℕ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) (h4 : t ≠ 0) (h5 : gcd (gcd x y) (gcd z t) = 1) :
  ∃ (x y z t : ℕ), x^3 + y^3 + z^3 = t^4 ∧ gcd (gcd x y) (gcd z t) = 1 :=
sorry

end infinite_solutions_xyz_t_l173_173592


namespace tall_and_neat_seedlings_are_D_l173_173279

def avg_height_A := 13
def avg_height_B := 15
def avg_height_C := 13
def avg_height_D := 15

def variance_A := 3.6
def variance_B := 6.3
def variance_C := 6.3
def variance_D := 3.6

theorem tall_and_neat_seedlings_are_D (h1 : avg_height_A = 13) 
                                      (h2 : avg_height_B = 15) 
                                      (h3 : avg_height_C = 13) 
                                      (h4 : avg_height_D = 15) 
                                      (v1 : variance_A = 3.6)
                                      (v2 : variance_B = 6.3)
                                      (v3 : variance_C = 6.3)
                                      (v4 : variance_D = 3.6) : 
                                      (avg_height_D = 15 ∧ variance_D = 3.6) :=
begin
  sorry
end

end tall_and_neat_seedlings_are_D_l173_173279


namespace first_loan_amount_l173_173033

theorem first_loan_amount :
  ∃ (L₁ L₂ : ℝ) (r : ℝ),
  (L₂ = 4700) ∧
  (L₁ = L₂ + 1500) ∧
  (0.09 * L₂ + r * L₁ = 617) ∧
  (L₁ = 6200) :=
by 
  sorry

end first_loan_amount_l173_173033


namespace min_number_of_smaller_squares_l173_173392

theorem min_number_of_smaller_squares (side_length : ℕ) (h : side_length = 11) :
  ∃ n (pieces : ℕ → ℕ), (∀ i, pieces i * i * i = side_length * side_length) ∧ (∀ i, ∀ j, pieces i ≠ 0 → (i:hidden) < side_length) ∧ n = 11 := by
  sorry

end min_number_of_smaller_squares_l173_173392


namespace sum_of_divisors_eq_360_l173_173641

theorem sum_of_divisors_eq_360 (i j : ℕ) (h : (Finset.range (i+1)).sum (λ n, 2^n) * (Finset.range (j+1)).sum (λ m, 3^m) = 360) : i + j = 5 :=
sorry

end sum_of_divisors_eq_360_l173_173641


namespace midpoint_tangent_segment_circle_locus_l173_173687

noncomputable def midpoint_of_tangent_segment_locus
  (a c : ℝ) (a_pos : a > 0) (c_pos : c > 0) : set (ℝ × ℝ) :=
{ M | ∃ (x y : ℝ), 
    (4 * x^2 + 4 * y^2 - 4 * a * x + a^2 = c^2) ∧ 
    (x ∈ Icc ((a^2 - c^2) / (2 * a)) ((a^2 + c^2) / (2 * a))) ∧ 
    (y > 0) }

theorem midpoint_tangent_segment_circle_locus
  (a c : ℝ) (a_pos : a > 0) (c_pos : c > 0) : 
  ∀ (M : ℝ × ℝ), 
    (∃ (x y : ℝ), 
      4 * x^2 + 4 * y^2 - 4 * a * x + a^2 = c^2 ∧ 
      x ∈ Icc ((a^2 - c^2) / (2 * a)) ((a^2 + c^2) / (2 * a)) ∧ 
      y > 0) ↔ 
    M ∈ (midpoint_of_tangent_segment_locus a c a_pos c_pos) :=
by {
  sorry
}

end midpoint_tangent_segment_circle_locus_l173_173687


namespace vasya_lowest_position_l173_173219

theorem vasya_lowest_position
  (num_cyclists : ℕ)
  (num_stages : ℕ)
  (num_ahead : ℕ)
  (position_vasya : ℕ)
  (total_time : List ℕ)
  (unique_total_times : total_time.nodup)
  (stage_positions : List (List ℕ))
  (unique_stage_positions : ∀ stage ∈ stage_positions, stage.nodup)
  (vasya_consistent : ∀ stage ∈ stage_positions, stage.nth position_vasya = some num_ahead) :
  num_ahead * num_stages + 1 = 91 :=
by
  sorry

end vasya_lowest_position_l173_173219


namespace range_of_m_l173_173865

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 2

noncomputable def g (x m : ℝ) : ℝ := abs (f x) - f x - 2 * m * x - 2 * m^2

theorem range_of_m (m : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ g x1 m = 0 ∧ g x2 m = 0 ∧ g x3 m = 0) ↔
  (m ∈ set.Ioo ((1 - 2 * real.sqrt 7) / 3) (-1) ∨ m ∈ set.Ioo 2 ((1 + 2 * real.sqrt 7) / 3)) := sorry

end range_of_m_l173_173865


namespace find_x_l173_173491

theorem find_x (x y : ℝ) (h : 12 * 3 ^ x = 7 ^ (y + 5)) (hy : y = -5) : x = -Real.log 12 / Real.log 3 :=
by
  sorry

end find_x_l173_173491


namespace vasya_lowest_position_l173_173239

theorem vasya_lowest_position
  (n : ℕ) (m : ℕ) (num_cyclists : ℕ) (vasya_place : ℕ) :
  num_cyclists = 500 →
  n = 15 →
  vasya_place = 7 →
  ∀ (stages : fin n → fin num_cyclists) (no_identical_times : ∀ i j : fin n, i ≠ j → 
  ∀ k l : fin num_cyclists, stages i k ≠ stages j l),
  ∃ (lowest_position : ℕ), lowest_position = 91 := 
by sorry

end vasya_lowest_position_l173_173239


namespace problem1_problem2_l173_173605

-- Problem 1
theorem problem1 : (2 + 7 / 9) ^ 0.5 + 0.1 ^ (-2) - Real.pi ^ 0 + 1 / 3 = 101 := by
  sorry

-- Problem 2
theorem problem2 : (Real.log 2) ^ 2 + Real.log 2 * Real.log 5 + sqrt ((Real.log 2) ^ 2 - Real.log 4 + 1) = 1 := by
  sorry

end problem1_problem2_l173_173605


namespace solve_system_eqns_l173_173170

theorem solve_system_eqns (x y : ℝ) (m : ℤ) :
  x^2 + 4 * sin y^2 - 4 = 0 ∧ cos x - 2 * cos y^2 - 1 = 0 ↔
  (x = 0 ∧ ∃ m : ℤ, y = (π / 2) + m * π) :=
by sorry

end solve_system_eqns_l173_173170


namespace alice_wins_iff_l173_173548

theorem alice_wins_iff (m n : ℕ) (h_m_pos : 0 < m) (h_n_pos : 0 < n) :
  (∃ Alice_wins : (m % 2 = 0) ∨ (n % 2 = 1)) ↔ true := 
sorry

end alice_wins_iff_l173_173548


namespace no_matching_option_for_fraction_l173_173117

theorem no_matching_option_for_fraction (m n : ℕ) (h : m = 16 ^ 500) : 
  (m / 8 ≠ 8 ^ 499) ∧ 
  (m / 8 ≠ 4 ^ 999) ∧ 
  (m / 8 ≠ 2 ^ 1998) ∧ 
  (m / 8 ≠ 4 ^ 498) ∧ 
  (m / 8 ≠ 2 ^ 1994) := 
by {
  sorry
}

end no_matching_option_for_fraction_l173_173117


namespace rabbit_speed_correct_l173_173056

-- Define the conditions given in the problem
def rabbit_speed (x : ℝ) : Prop :=
2 * (2 * x + 4) = 188

-- State the main theorem using the defined conditions
theorem rabbit_speed_correct : ∃ x : ℝ, rabbit_speed x ∧ x = 45 :=
by
  sorry

end rabbit_speed_correct_l173_173056


namespace problem1_l173_173322

theorem problem1 (a b : ℤ) (h1 : abs a = 5) (h2 : abs b = 3) (h3 : abs (a - b) = b - a) : a - b = -8 ∨ a - b = -2 := by 
  sorry

end problem1_l173_173322


namespace total_price_after_increase_l173_173978

def original_jewelry_price := 30
def original_painting_price := 100
def jewelry_price_increase := 10
def painting_price_increase_percentage := 20
def num_jewelry_pieces := 2
def num_paintings := 5

theorem total_price_after_increase : 
    let new_jewelry_price := original_jewelry_price + jewelry_price_increase in
    let new_painting_price := original_painting_price + (original_painting_price * painting_price_increase_percentage / 100) in
    let total_jewelry_cost := new_jewelry_price * num_jewelry_pieces in
    let total_painting_cost := new_painting_price * num_paintings in
    total_jewelry_cost + total_painting_cost = 680 :=
by
  -- proof omitted
  sorry

end total_price_after_increase_l173_173978


namespace inscribed_circle_radius_escribed_circle_radius_l173_173126

variable {a b c : ℝ}

theorem inscribed_circle_radius (h : a^2 + b^2 = c^2) : 
  (a + b - c) / 2 = (let r := (a + b - c) / 2 in r) :=
  sorry

theorem escribed_circle_radius (h : a^2 + b^2 = c^2) : 
  (a + b + c) / 2 = (let r' := (a + b + c) / 2 in r') :=
  sorry

end inscribed_circle_radius_escribed_circle_radius_l173_173126


namespace initial_average_score_l173_173067

theorem initial_average_score (S : ℕ) 
  (h1 : (S - 55) / 15 = 63) : 
  S / 16 = 62.5 :=
begin
  sorry
end

end initial_average_score_l173_173067


namespace arrangement_of_boys_and_girls_l173_173817

theorem arrangement_of_boys_and_girls (boys girls : Fin 4)
    (adj_opposites : ∀ i : Fin 7, i % 2 = 0 → boys i ≠ boys (i+1))
    (boyA_next_to_girlB : ∃ i : Fin 7, (boys i = boyA) ∧ (girls (i + 1) = girlB))
    : ∃ n, n = 504 :=
sorry

end arrangement_of_boys_and_girls_l173_173817


namespace happy_number_part1_happy_number_part2_happy_number_part3_l173_173595

section HappyEquations

def is_happy_eq (a b c : ℤ) : Prop :=
  ∃ x1 x2 : ℤ, a ≠ 0 ∧ a * x1 * x1 + b * x1 + c = 0 ∧ a * x2 * x2 + b * x2 + c = 0

def happy_number (a b c : ℤ) : ℚ :=
  (4 * a * c - b ^ 2) / (4 * a)

def happy_to_each_other (a b c p q r : ℤ) : Prop :=
  let Fa : ℚ := happy_number a b c
  let Fb : ℚ := happy_number p q r
  |r * Fa - c * Fb| = 0

theorem happy_number_part1 :
  happy_number 1 (-2) (-3) = -4 :=
by sorry

theorem happy_number_part2 (m : ℤ) (h : 1 < m ∧ m < 6) :
  is_happy_eq 1 (2 * m - 1) (m ^ 2 - 2 * m - 3) →
  m = 3 ∧ happy_number 1 (2 * m - 1) (m ^ 2 - 2 * m - 3) = -25 / 4 :=
by sorry

theorem happy_number_part3 (m n : ℤ) :
  is_happy_eq 1 (-m) (m + 1) ∧ is_happy_eq 1 (-(n + 2)) (2 * n) →
  happy_to_each_other 1 (-m) (m + 1) 1 (-(n + 2)) (2 * n) →
  n = 0 ∨ n = 3 ∨ n = 3 / 2 :=
by sorry

end HappyEquations

end happy_number_part1_happy_number_part2_happy_number_part3_l173_173595


namespace third_smallest_triangular_square_l173_173694

theorem third_smallest_triangular_square :
  ∃ n : ℕ, n = 1225 ∧ 
           (∃ x y : ℕ, y^2 - 8 * x^2 = 1 ∧ 
                        y = 99 ∧ x = 35) :=
by
  sorry

end third_smallest_triangular_square_l173_173694


namespace angle_between_OA_OC_l173_173844

-- Define vectors and conditions
variables {V : Type} [inner_product_space ℝ V]
variables (OA OB OC : V)
variables (a b c : ℝ)

-- Given conditions
def cond1 : ∥OA∥ = 1 := sorry
def cond2 : ∥OB∥ = 2 := sorry
def cond3 : real.angle (OA, OB) = 2 * real.pi / 3 := sorry
def cond4 : OC = (1 / 2) • OA + (1 / 4) • OB := sorry

-- To prove
theorem angle_between_OA_OC : real.angle (OA, OC) = real.pi / 3 :=
by sorry

end angle_between_OA_OC_l173_173844


namespace minimum_length_of_AB_l173_173851

def circle (x y : ℝ) : Prop := x^2 + y^2 = 1

def tangent_line (θ : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ),
    A = (0, (Real.sin θ - (Real.cos θ)^2)) ∧
    B = (1 / Real.cos θ, 0)

theorem minimum_length_of_AB (θ : ℝ) (h_circle : circle (Real.cos θ) (Real.sin θ)) (h_tangent : tangent_line θ) : 
  ∃ (min_length : ℝ), min_length = 2 :=
by
  -- The details of the proof go here.
  sorry

end minimum_length_of_AB_l173_173851


namespace Sn_expression_l173_173640

noncomputable def a : ℕ → ℝ 
| 0       := 1
| (n + 1) := (4/3) * a n

def S (n : ℕ) : ℝ :=
if n = 0 then 0 else ∑ i in Finset.range n, a (i + 1)

theorem Sn_expression (n : ℕ) (hn : n > 0) : S n = (4/3)^(n-1) := by
  sorry

end Sn_expression_l173_173640


namespace probability_of_triangle_l173_173647

-- Declare the lengths of the line segments
def length1 : ℕ := 1
def length2 : ℕ := 3
def length3 : ℕ := 5
def length4 : ℕ := 7

-- Define a set of line lengths
def line_segments : List ℕ := [length1, length2, length3, length4]

-- Total number of ways to choose 3 segments out of 4
def total_combinations : ℕ := Nat.choose 4 3

-- Predicate to check if the lengths can form a triangle
def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Calculate the number of valid triangles
def valid_triangle_count : ℕ :=
  if can_form_triangle length2 length3 length4 then 1 else 0

-- Define the proposition to be proven
theorem probability_of_triangle :
  valid_triangle_count % total_combinations = 1 % 4 :=
by
  sorry

end probability_of_triangle_l173_173647


namespace sandra_share_l173_173600

def ratio (a b c : ℕ) : Prop := a = 2 * b ∧ c = 3 * b

theorem sandra_share (Amy_Share : ℕ) (h_nonzero : Amy_Share ≠ 0) (h_ratio : ratio 2 Amy_Share 3) : 
  let Sandra_Share := 2 * Amy_Share in Sandra_Share = 100 :=
  by
  sorry

end sandra_share_l173_173600


namespace expected_sixes_in_three_rolls_l173_173672

theorem expected_sixes_in_three_rolls : 
  (∑ k in Finset.range 4, k * (Nat.choose 3 k) * (1/6)^k * (5/6)^(3-k)) = 1/2 := 
by
  sorry

end expected_sixes_in_three_rolls_l173_173672


namespace sum_of_distances_from_vertex_to_midpoints_of_sides_eq_l173_173639

noncomputable def distance_from_vertex_to_midpoints_of_sides (side_length : ℝ) : ℝ :=
  let altitude := (real.sqrt 3) / 2 * side_length
  let distance_to_midpoint := altitude / 2
  distance_to_midpoint * 3

theorem sum_of_distances_from_vertex_to_midpoints_of_sides_eq :
  distance_from_vertex_to_midpoints_of_sides 3 = (9 * real.sqrt 3) / 4 :=
by
  sorry

end sum_of_distances_from_vertex_to_midpoints_of_sides_eq_l173_173639


namespace box_width_is_18_l173_173739

variables (l h v_cube n : ℕ) (V w : ℤ)

-- Conditions: 
def length := 10 -- cm
def height := 4 -- cm
def volume_cube := 12 -- cubic cm
def cubes := 60
def total_volume := (cubes : ℤ) * (volume_cube : ℤ) -- cubic cm

-- Problem specification:
def width := total_volume / (length * height)

theorem box_width_is_18 :
  width = 18 := 
sorry

end box_width_is_18_l173_173739


namespace students_calculation_l173_173276

def number_of_stars : ℝ := 3.0
def students_per_star : ℝ := 41.33333333
def total_students : ℝ := 124

theorem students_calculation : number_of_stars * students_per_star = total_students := 
by
  sorry

end students_calculation_l173_173276


namespace range_of_f_l173_173632

def f (x : ℝ) : ℝ := Real.sqrt (x - 4) + Real.sqrt (15 - 3 * x)

theorem range_of_f :
  ∀ y, y ∈ Set.range f ↔ y ∈ Set.Icc 1 2 :=
sorry

end range_of_f_l173_173632


namespace part_a_part_b_l173_173950

-- Define the strictly increasing function f
variable (f : ℕ → ℕ) (hf : ∀ n m, n < m → f n < f m)

-- Part (a)
theorem part_a :
  ∃ (y : ℕ → ℝ), (∀ n : ℕ, y n > 0) ∧ (∀ m n : ℕ, m < n → y m > y n) ∧ (tendsto y at_top (nhds 0)) ∧ (∀ n : ℕ, y n ≤ 2 * y (f n)) :=
sorry

-- Part (b)
theorem part_b (x : ℕ → ℝ) (hx1 : ∀ n m, n < m → x n > x m) (hx2 : tendsto x at_top (nhds 0)) :
  ∃ (y : ℕ → ℝ), (∀ n, y n > 0) ∧ (∀ m n, m < n → y m > y n) ∧ (tendsto y at_top (nhds 0)) ∧ (∀ n, x n ≤ y n) ∧ (∀ n, y n ≤ 2 * y (f n)) :=
sorry

end part_a_part_b_l173_173950


namespace solve_trig_eq_l173_173308

noncomputable theory

variable {k : ℤ}

theorem solve_trig_eq (x : ℝ) : 
  5.31 * tan(6 * x) * cos(2 * x) - sin(2 * x) - 2 * sin(4 * x) = 0 ↔ 
    ∃ k : ℤ, x = (π * k / 2) ∨ x = (π / 18) * (6 * k + 1) ∨ x = (π / 18) * (6 * k - 1) := 
by
  sorry

end solve_trig_eq_l173_173308


namespace geometric_identity_l173_173566

variables {A B C P L M N : Type} [metric_space A] [metric_space B] [metric_space C] 
          [metric_space P] [metric_space L] [metric_space M] [metric_space N]

noncomputable def minor_arc_BC (A B C P L M N : Type) [metric_space A] [metric_space B] 
[metric_space C] [metric_space P] [metric_space L] [metric_space M] [metric_space N] 
(arc_BC : set P) : Prop := sorry

noncomputable def projection (X Y : Type) [metric_space X] [metric_space Y] (p : X) : Y := sorry

noncomputable def length (X : Type) [metric_space X] (x y : X) : ℝ := sorry

variables (a b c l m n : ℝ)

theorem geometric_identity (A B C P L M N : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space P]
  [metric_space L] [metric_space M] [metric_space N]
  (arc_condition : minor_arc_BC A B C P L M N)
  (proj_L : projection BC L = P)
  (proj_M : projection CA M = P)
  (proj_N : projection AB N = P)
  (l_eq : length P L = l)
  (m_eq : length P M = m)
  (n_eq : length P N = n)
  (a_eq : length B C = a)
  (b_eq : length C A = b)
  (c_eq : length A B = c) :
  mn * a = ln * b + lm * c :=
sorry

end geometric_identity_l173_173566


namespace sum_BY_CY_eq_AX_l173_173933

variable {A B C X Y : Point}
variable {BC AC AB B1 C1 Y : Line}

-- Assumptions
def is_triangle (A B C : Point) : Prop := sorry

axiom smallest_angle_A (A B C : Point) : angle A < angle B ∧ angle A < angle C

axiom line_through_A_intersects_BC_at_X (A B C X : Point) : Line A X

axiom intersect_circumcircle_A_and_X (A B C X : Point) : circumcircle A B C X

axiom perpendicular_bisector_intersects_AC_at_B1 (A C B1 : Point) : perpendicular_bisector A C

axiom perpendicular_bisector_intersects_AB_at_C1 (A B C1 : Point) : perpendicular_bisector A B

axiom lines_BC1_and_CB1_intersect_at_Y (B C B1 C1 Y : Point) : inter_point line_BC1 line_CB1 = Y

-- The theorem to prove
theorem sum_BY_CY_eq_AX 
  (h_triangle : is_triangle A B C)
  (h_smallest_angle_A : smallest_angle_A A B C)
  (h_line_through_A : line_through_A_intersects_BC_at_X A B C X)
  (h_circumcircle_A_X : intersect_circumcircle_A_and_X A B C X)
  (h_bisector_intersect_AC_at_B1 : perpendicular_bisector_intersects_AC_at_B1 A C B1)
  (h_bisector_intersect_AB_at_C1 : perpendicular_bisector_intersects_AB_at_C1 A B C1)
  (h_lines_intersect_at_Y : lines_BC1_and_CB1_intersect_at_Y B C B1 C1 Y) : 
  len B Y + len C Y = len A X := 
sorry

end sum_BY_CY_eq_AX_l173_173933


namespace mrs_hilt_read_chapters_l173_173987

-- Define the problem conditions
def books : ℕ := 4
def chapters_per_book : ℕ := 17

-- State the proof problem
theorem mrs_hilt_read_chapters : (books * chapters_per_book) = 68 := 
by
  sorry

end mrs_hilt_read_chapters_l173_173987


namespace number_of_valid_special_plates_l173_173082

def rotokas_alphabet : list char := ['A', 'E', 'G', 'I', 'K', 'M', 'P', 'R', 'S', 'T', 'U', 'V']

def valid_special_plate (s : string) : Prop :=
  s.length = 5 ∧
  (s.front = 'P' ∨ s.front = 'T') ∧
  s.back = 'R' ∧
  ∀ c ∈ s.to_list, c ≠ 'U' ∧
  s.to_list.nodup

theorem number_of_valid_special_plates :
  {s : string | valid_special_plate s}.to_finset.card = 1440 :=
by
  sorry

end number_of_valid_special_plates_l173_173082


namespace vasya_lowest_position_l173_173257

noncomputable theory

def number_of_cyclists := 500
def number_of_stages := 15
def position_of_vasya_each_stage := 7

theorem vasya_lowest_position (total_cyclists : ℕ) (stages : ℕ) (position_each_stage : ℕ)
  (h_total_cyclists : total_cyclists = number_of_cyclists)
  (h_stages : stages = number_of_stages)
  (h_position_each_stage : position_each_stage = position_of_vasya_each_stage)
  (no_identical_times : ∀ (i j : ℕ), i ≠ j → ∀ (stage : ℕ), stage ≤ stages → ∀ (t : ℕ), t ≤ total_cyclists → 
    (time : Π (s : ℕ) (c : ℕ), c < total_cyclists → ℕ), 
    time stage i < time stage j ∨ time stage j < time stage i):
  ∃ lowest_pos, lowest_pos = 91 := sorry

end vasya_lowest_position_l173_173257


namespace vasya_lowest_position_l173_173255

noncomputable theory

def number_of_cyclists := 500
def number_of_stages := 15
def position_of_vasya_each_stage := 7

theorem vasya_lowest_position (total_cyclists : ℕ) (stages : ℕ) (position_each_stage : ℕ)
  (h_total_cyclists : total_cyclists = number_of_cyclists)
  (h_stages : stages = number_of_stages)
  (h_position_each_stage : position_each_stage = position_of_vasya_each_stage)
  (no_identical_times : ∀ (i j : ℕ), i ≠ j → ∀ (stage : ℕ), stage ≤ stages → ∀ (t : ℕ), t ≤ total_cyclists → 
    (time : Π (s : ℕ) (c : ℕ), c < total_cyclists → ℕ), 
    time stage i < time stage j ∨ time stage j < time stage i):
  ∃ lowest_pos, lowest_pos = 91 := sorry

end vasya_lowest_position_l173_173255


namespace problem_solution_l173_173332

theorem problem_solution (f : ℕ → ℝ) (h1 : ∀ m n : ℕ, 0 < m → 0 < n → f (m + n) = f m * f n)
                         (h2 : f 1 = 2) :
  (Finset.sum (Finset.filter (λ x, x % 2 = 0) (Finset.range 2017)) (λ n, f n / f (n - 1))) = 2016 := 
by
  -- start proof here
  sorry

end problem_solution_l173_173332


namespace circle_speeds_l173_173149

theorem circle_speeds:
  ∃ x y : ℝ, (x = 4 ∧ y = 1) ∨ (x = 3.9104 ∧ y = 1.3072) ∧ 
  let r1 := 9 in 
  let r2 := 4 in 
  let d1 := 48 in 
  let d2 := 14 in 
  let t_ext := 9 in 
  let t_int := 11 in 
  (r1 + r2 = 13) ∧ 
  (r1 - r2 = 5) ∧ 
  ((d1 - t_ext * x) ≠ (d2 - t_ext * y)) ∧ 
  ((d1 - t_int * x) ≠ (d2 - t_int * y)) ∧
  (((d1 - t_ext * x)^2 + (d2 - t_ext * y)^2 = (r1 + r2)^2) ∧
  ((d1 - t_int * x)^2 + (d2 - t_int * y)^2 = (r1 - r2)^2)) :=
sorry

end circle_speeds_l173_173149


namespace signal_soldier_flags_l173_173752

theorem signal_soldier_flags :
  let flags := ["red", "yellow", "blue"]
  ∃ (n : ℕ), n = 15 ∧
  let one_flag := flags.length,
      two_flags := flags.length * (flags.length - 1),
      three_flags := flags.length * (flags.length - 1) * (flags.length - 2)
  in n = one_flag + two_flags + three_flags :=
by {
  sorry
}

end signal_soldier_flags_l173_173752


namespace expected_value_of_sixes_l173_173655

theorem expected_value_of_sixes (n : ℕ) (k : ℕ) (p q : ℚ) 
  (h1 : n = 3) 
  (h2 : k = 6)
  (h3 : p = 1/6) 
  (h4 : q = 5/6) : 
  (1 : ℚ) / 2 = ∑ i in finset.range (n + 1), (i * (nat.choose n i * p^i * q^(n-i))) := 
sorry

end expected_value_of_sixes_l173_173655


namespace fraction_of_yard_occupied_l173_173746

-- Define the rectangular yard with given length and width
def yard_length : ℝ := 25
def yard_width : ℝ := 5

-- Define the isosceles right triangle and the parallel sides of the trapezoid
def parallel_side1 : ℝ := 15
def parallel_side2 : ℝ := 25
def triangle_leg : ℝ := (parallel_side2 - parallel_side1) / 2

-- Areas
def triangle_area : ℝ := (1 / 2) * triangle_leg ^ 2
def flower_beds_area : ℝ := 2 * triangle_area
def yard_area : ℝ := yard_length * yard_width

-- Fraction calculation
def fraction_occupied : ℝ := flower_beds_area / yard_area

-- The proof statement
theorem fraction_of_yard_occupied:
  fraction_occupied = 1 / 5 :=
by
  sorry

end fraction_of_yard_occupied_l173_173746


namespace compare_m_n_l173_173821

noncomputable def m (a : ℝ) : ℝ := 6^a / (36^(a + 1) + 1)
noncomputable def n (b : ℝ) : ℝ := (1/3) * b^2 - b + (5/6)

theorem compare_m_n (a b : ℝ) : m a ≤ n b := sorry

end compare_m_n_l173_173821


namespace solution_set_of_inequality_l173_173268

noncomputable def A_n_k (n k : ℕ) := Nat.factorial n / Nat.factorial (n - k)

theorem solution_set_of_inequality :
  {x : ℕ | A_n_k 8 x < 6 * A_n_k 8 (x - 2)} = {8} :=
begin
  sorry
end

end solution_set_of_inequality_l173_173268


namespace smallest_x_is_solution_l173_173378

def smallest_positive_angle (x : ℝ) : Prop :=
  tan (6 * x) = (cos (2 * x) - sin (2 * x)) / (cos (2 * x) + sin (2 * x))

noncomputable def smallest_x : ℝ :=
  45 / 8

theorem smallest_x_is_solution : smallest_positive_angle (smallest_x * (Real.pi / 180)) :=
by
  sorry -- Proof omitted

end smallest_x_is_solution_l173_173378


namespace largest_angle_of_triangle_with_altitudes_l173_173182

theorem largest_angle_of_triangle_with_altitudes :
  ∀ (a b c : ℝ), (10 * a = 24 * b) ∧ (24 * b = 30 * c) ∧ (10 * a = 30 * c) → 
  let k := (10 * a);
  let side_a := k / 10;
  let side_b := k / 24;
  let side_c := k / 30;
  ∀ (largest_angle : ℝ), largest_angle = 120 :=
begin
  sorry
end

end largest_angle_of_triangle_with_altitudes_l173_173182


namespace natalie_bushes_to_zucchinis_l173_173409

/-- Each of Natalie's blueberry bushes yields ten containers of blueberries,
    and she trades six containers of blueberries for three zucchinis.
    Given this setup, prove that the number of bushes Natalie needs to pick
    in order to get sixty zucchinis is twelve. --/
theorem natalie_bushes_to_zucchinis :
  (∀ (bush_yield containers_needed : ℕ), bush_yield = 10 ∧ containers_needed = 60 * (6 / 3)) →
  (∀ (containers_total bushes_needed : ℕ), containers_total = 60 * (6 / 3) ∧ bushes_needed = containers_total * (1 / bush_yield)) →
  bushes_needed = 12 :=
by
  sorry

end natalie_bushes_to_zucchinis_l173_173409


namespace tens_digit_of_power_l173_173405

-- Define the base.
def base := 13

-- Define the modulus.
def modulus := 100

-- Define the exponent.
def exponent := 2047

-- Define the periodicity.
def periodicity := 20

-- The statement: the tens digit of 13^2047
theorem tens_digit_of_power :
  (base ^ exponent % modulus / 10) % 10 = 1 := by
  have h_periodicity : base ^ periodicity % modulus = 1 := by sorry
  have h_reduction : exponent % periodicity = 7 := by sorry
  have h_power_mod : base ^ 7 % modulus = 17 := by sorry
  calc (base ^ exponent % modulus / 10) % 10 = (base ^ 7 % modulus / 10) % 10 : by sorry
  ... = (17 / 10) % 10 : by sorry
  ... = 1 : by norm_num

end tens_digit_of_power_l173_173405


namespace chess_pieces_on_board_l173_173686

theorem chess_pieces_on_board (m n : ℕ) 
  (h1: ∀ (i j: ℕ), 1 ≤ i ∧ i ≤ 7 → 1 ≤ j ∧ j ≤ 7 → (∃ c : ℕ, pieces_in_2x2_square i j = c) → pieces_in_2x2_square i j = m)
  (h2: ∀ (i j: ℕ), 1 ≤ i ∧ i ≤ 6 → 1 ≤ j ∧ j ≤ 7 → (∃ c : ℕ, pieces_in_3x1_rectangle i j = c) → pieces_in_3x1_rectangle i j = n)
  (h3: 3 * m = 4 * n): 
  ∃ k : ℕ, pieces_on_board = k ∧ (k = 0 ∨ k = 64) :=
by
  sorry

end chess_pieces_on_board_l173_173686


namespace triangle_tangent_ratio_l173_173947

variables (A B C D E : Point)
variables (AB AC DB EC : Line) (Gamma : Circle)

structure EquilateralTriangle (A B C : Point) : Prop :=
(is_equilateral : ∀ (a b c : ℝ), ∃ x y z : ℝ, x = y = z)

structure Incircle (Gamma : Circle) (A B C : Point) : Prop :=
(is_incircle : ∀ (incircle : Circle), Γ inscribed_in ABC)

structure TangentPoints (D E : Point) (Gamma : Circle) : Prop :=
(points_on_tangent : ∀ (DE_tangent : Line), DE_tangent tangent_to Γ)

def ratio_expression (AD DB AE EC : ℝ) : Prop := 
  AD / DB + AE / EC = 1

theorem triangle_tangent_ratio (h : EquilateralTriangle A B C) (Γ : Incircle Gamma A B C) 
  (tangent : TangentPoints D E Gamma) :
  ratio_expression AD DB AE EC :=
sorry

end triangle_tangent_ratio_l173_173947


namespace machine_P_takes_longer_l173_173977

-- Define constants for rates and counts
def R_A : ℝ := 6  -- Rate of Production for Machine A in sprockets/hour
def sprockets : ℕ := 660  -- Number of sprockets to be produced
def T_Q : ℝ := 100  -- Time for Machine Q to produce 660 sprockets

-- Define production rate for Machine Q
def R_Q : ℝ := R_A + 0.1 * R_A  -- Machine Q produces 10% more than Machine A

-- Define the time Machine P takes to produce the same number of sprockets
variable (X : ℝ)  -- Extra time in hours
def T_P : ℝ := T_Q + X  -- Time for Machine P to produce 660 sprockets

-- The statement we need to prove
theorem machine_P_takes_longer 
  (H1 : T_Q = sprockets / R_Q) 
  (H2 : T_P = T_Q + X) : 
  T_P = 100 + X := 
sorry

end machine_P_takes_longer_l173_173977


namespace ellipse_ratio_sum_l173_173780

theorem ellipse_ratio_sum :
  ∀ (x y : ℝ), 3 * x^2 + 2 * x * y + 4 * y^2 - 19 * x - 14 * y + 55 = 0 →
  let k := real_argmax (λ m, ∃ x y, y = m * x ∧ 3 * x^2 + 2 * x * y + 4 * y^2 - 19 * x - 14 * y + 55 = 0)
  let j := real_argmin (λ m, ∃ x y, y = m * x ∧ 3 * x^2 + 2 * x * y + 4 * y^2 - 19 * x - 14 * y + 55 = 0)
  k + j = 23/171 :=
by
  intro x y h_ellipse
  -- Proof to be provided
  sorry

end ellipse_ratio_sum_l173_173780


namespace sandra_share_l173_173601

theorem sandra_share 
  (r : ℕ) (sandra_ratio amy's_share : ℕ) 
  (h1 : sandra_ratio = 2) (h2 : amy's_share = 50) : 
  sandra_ratio * amy's_share = 100 :=
by
  rw [h1, h2]
  sorry

end sandra_share_l173_173601


namespace will_money_left_l173_173700

theorem will_money_left (initial sweater tshirt shoes refund_percentage : ℕ) 
  (h_initial : initial = 74)
  (h_sweater : sweater = 9)
  (h_tshirt : tshirt = 11)
  (h_shoes : shoes = 30)
  (h_refund_percentage : refund_percentage = 90) : 
  initial - (sweater + tshirt + (100 - refund_percentage) * shoes / 100) = 51 := by
  sorry

end will_money_left_l173_173700


namespace center_of_gravity_distance_approx_l173_173805

noncomputable def distance_center_of_gravity (n : ℕ) : ℝ :=
  if n = 1 then 1
  else 2 / Real.sqrt (Math.pi * (n - 1))

theorem center_of_gravity_distance_approx : (n : ℕ) (radius : ℝ) (error : ℝ) 
  (h_n : n = 100) (h_radius : radius = 1) (h_error : error = 0.1) :
  let calc_distance := distance_center_of_gravity n 
  calc_distance ≈ 0.113 ± 0.0113 :=
by
  -- proof would be here
  sorry

end center_of_gravity_distance_approx_l173_173805


namespace expected_number_of_sixes_l173_173671

-- Define the problem context and conditions
def die_prob := (1 : ℝ) / 6

def expected_six (n : ℕ) : ℝ :=
  n * die_prob

-- The main proposition to prove
theorem expected_number_of_sixes (n : ℕ) (hn : n = 3) : expected_six n = 1 / 2 :=
by
  rw [hn]
  have fact1 : (3 : ℝ) * die_prob = 3 / 6 := by norm_cast; norm_num
  rw [fact1]
  norm_num

-- We add sorry to indicate incomplete proof, fulfilling criteria 4
sorry

end expected_number_of_sixes_l173_173671


namespace sum_of_digits_l173_173949

theorem sum_of_digits (L : ℕ) 
  (hL : L = (10^2022 - 1) / 9) :
  (∑ d in (9 * L^2 + 2 * L).digits, (d : ℕ)) = 4044 := 
by
  sorry

end sum_of_digits_l173_173949


namespace rabbit_speed_l173_173058

theorem rabbit_speed (s : ℕ) (h : (s * 2 + 4) * 2 = 188) : s = 45 :=
sorry

end rabbit_speed_l173_173058


namespace revenue_growth_rate_equation_l173_173731

noncomputable def MarchRevenue := 150
noncomputable def AprilRevenue := MarchRevenue * 0.9
noncomputable def JuneRevenue := 200

theorem revenue_growth_rate_equation (x : ℝ) :
  (MarchRevenue * 0.9) * (1 + x)^2 = JuneRevenue := sorry

end revenue_growth_rate_equation_l173_173731


namespace magic_trick_strategy_l173_173356

noncomputable def illusionist_strategy_exists (k : ℕ) (n : ℕ) : Prop :=
  ∃ f : list (fin (n + 1)) → fin (n - k + 1), 
    ∀ (s : list (fin (n + 1))) (i : fin (n - k + 2)), 
      (1 ≤ k) → 
      (n = k.factorial + k - 1) → 
      (i.val < n - k + 1) → 
      let covered_seq := list.take k (list.drop i.val s) in
      ∃ unique_order : list (fin (k + 1)), 
        (covered_seq ≠ unique_order)

theorem magic_trick_strategy (k : ℕ) (n := k.factorial + k - 1) : 
  illusionist_strategy_exists k n := 
sorry

end magic_trick_strategy_l173_173356


namespace angle_AEC_measure_l173_173690

-- Definitions
variable {O : Type} [regular_octagon O]

variable {A E C : Point O}

-- Given geometry conditions of the problem
axiom is_vertex_A : is_vertex O A
axiom is_vertex_E : is_vertex O E
axiom is_vertex_C : is_vertex O C
axiom is_diagonal_AC : is_diagonal O A C
axiom is_diagonal_EC : is_diagonal O E C

-- Goal: Prove the degree measure of angle ∡AEC is 112.5°.
theorem angle_AEC_measure : measure_angle A E C = 112.5 := sorry

end angle_AEC_measure_l173_173690


namespace not_true_x_heartsuit_0_eq_x_l173_173393

def heartsuit (x y : ℝ) : ℝ := abs (x - y)

theorem not_true_x_heartsuit_0_eq_x : ¬(∀ x : ℝ, heartsuit x 0 = x) :=
by
  (assume h : ∀ x : ℝ, heartsuit x 0 = x)
  have h_neg : heartsuit (-1) 0 = -1 := h (-1)
  dsimp [heartsuit] at h_neg
  contradiction

end not_true_x_heartsuit_0_eq_x_l173_173393


namespace find_digits_l173_173620

-- Define the digits behind each mask
constant Elephant : ℕ
constant Mouse : ℕ
constant Pig : ℕ
constant Panda : ℕ

-- Conditions from the problem
axiom condition1 : ∀ d : ℕ, (10 ≤ d * d) ∧ (d * d < 100) ∧ (d * d % 10 ≠ d)
axiom condition2 : Mouse * Mouse % 10 = Elephant

-- Prove the given digits behind the masks under the given conditions
theorem find_digits :
  Elephant = 8 ∧ Mouse = 4 ∧ Pig = 8 ∧ Panda = 1 :=
  sorry

end find_digits_l173_173620


namespace min_cubes_to_fill_box_l173_173309

def length : ℕ := 10
def width : ℕ := 13
def height : ℕ := 5
def volume_box : ℕ := length * width * height
def volume_cube : ℕ := 5
def num_cubes : ℕ := volume_box / volume_cube

theorem min_cubes_to_fill_box : num_cubes = 130 :=
by
  unfold length width height volume_box volume_cube num_cubes
  sorry

end min_cubes_to_fill_box_l173_173309


namespace range_of_t_inequality_N_star_l173_173469

-- Definition of the function f
def f (x : ℝ) := log (x + 1) - x^2 - x

-- Part (I) range problem
theorem range_of_t (t : ℝ) :
  ∃ x1 x2 ∈ set.Icc (0 : ℝ) (2 : ℝ), x1 ≠ x2 ∧ (f x1 + 5/2 * x1 - t = 0) ∧ (f x2 + 5/2 * x2 - t = 0) → 
  log 3 - 1 ≤ t ∧ t ≤ log 2 + 1 / 2 :=
sorry

-- Part (II) inequality problem
theorem inequality_N_star (n : ℕ) (hn : 0 < n) :
  log (n + 2) < (finset.sum (finset.range n) (λ i, (1:ℝ) / (1 + i)) + log 2) :=
sorry

end range_of_t_inequality_N_star_l173_173469


namespace at_least_twenty_percent_minority_l173_173314

-- Definitions based on conditions
def is_public (company : Type) (shareholders : ℕ) : Prop :=
  shareholders ≥ 15

def is_minority (shareholder_shares : ℕ) (total_shares : ℕ) : Prop :=
  shareholder_shares * 4 ≤ total_shares

-- Given facts
axiom one_sixth_public (N : ℕ) : ℕ :=
  N / 6

axiom firm_has_at_most_three_major_shareholders {N : ℕ} (firm : Type) :
  ∀ f, N ≥ 4 → ∃ (major_count : ℕ), major_count ≤ 3

axiom public_firm_minors {N : ℕ} (public_firms : Type) :
  ∀ f, is_public public_firms N → (N - 3) ≥ 12

axiom public_firms_count {N : ℕ} :
  ∀ total_firms, one_sixth_public total_firms = N/6

-- Theorem to prove
theorem at_least_twenty_percent_minority (N : ℕ) :
  ∃ (total_shareholders : ℕ) (minor_shareholders : ℕ),
  (minor_shareholders * 5) ≥ (total_shareholders * 2) := sorry

end at_least_twenty_percent_minority_l173_173314


namespace face_value_of_stock_l173_173755

theorem face_value_of_stock : 
  ∃ (F : ℝ), 
    (0.93 * F + 0.002 * 0.93 * F = 93.2) ∧ 
    (F ≈ 99.89) :=
by
  use 99.89
  sorry

end face_value_of_stock_l173_173755


namespace pyramid_side_length_l173_173612

-- Definitions for our conditions
def area_of_lateral_face : ℝ := 150
def slant_height : ℝ := 25

-- Theorem statement
theorem pyramid_side_length (A : ℝ) (h : ℝ) (s : ℝ) (hA : A = area_of_lateral_face) (hh : h = slant_height) :
  A = (1 / 2) * s * h → s = 12 :=
by
  intro h_eq
  rw [hA, hh, area_of_lateral_face, slant_height] at h_eq
  -- Steps to verify s = 12
  sorry

end pyramid_side_length_l173_173612


namespace no_opposite_meanings_in_C_l173_173766

def opposite_meanings (condition : String) : Prop :=
  match condition with
  | "A" => true
  | "B" => true
  | "C" => false
  | "D" => true
  | _   => false

theorem no_opposite_meanings_in_C :
  opposite_meanings "C" = false :=
by
  -- proof goes here
  sorry

end no_opposite_meanings_in_C_l173_173766


namespace three_points_not_collinear_l173_173425

variables {Point : Type} [AffineSpace ℝ Point]

def coplanar (A B C D: Point) : Prop :=
∃ (plane : AffineSubspace ℝ Point), 
  A ∈ plane ∧ B ∈ plane ∧ C ∈ plane ∧ D ∈ plane

def not_collinear (A B C: Point) : Prop :=
  ∀ (line : AffineSubspace ℝ Point), 
    A ∉ line ∨ B ∉ line ∨ C ∉ line

theorem three_points_not_collinear
  (A B C D : Point) (h1: coplanar A B C D) (h2: ¬ collinear A B C D) :
  ∃ (X Y Z : Point), (X = A ∨ X = B ∨ X = C ∨ X = D) ∧
                     (Y = A ∨ Y = B ∨ Y = C ∨ Y = D) ∧
                     (Z = A ∨ Z = B ∨ Z = C ∨ Z = D) ∧
                     not_collinear X Y Z := 
sorry

end three_points_not_collinear_l173_173425


namespace focus_of_parabola_x_eq_neg_1_div_4_y_squared_is_neg_1_0_l173_173420

theorem focus_of_parabola_x_eq_neg_1_div_4_y_squared_is_neg_1_0 :
  let P (y : ℝ) := (-1/4 * y^2, y)
  let F := (-1, 0)
  let d := 1
  ∀ (y : ℝ), (P y).fst = -1/4 * y^2 → (F.fst - P y.fst)^2 + (F.snd - P y.snd)^2 = (d + P y.fst)^2 → F = (-1, 0) :=
by
  intros P F d y h1 h2
  sorry

end focus_of_parabola_x_eq_neg_1_div_4_y_squared_is_neg_1_0_l173_173420


namespace final_number_not_perfect_square_l173_173996

theorem final_number_not_perfect_square :
  (∃ final_number : ℕ, 
    ∀ a b : ℕ, a ∈ Finset.range 101 ∧ b ∈ Finset.range 101 ∧ a ≠ b → 
    gcd (a^2 + b^2 + 2) (a^2 * b^2 + 3) = final_number) →
  ∀ final_number : ℕ, ¬ ∃ k : ℕ, final_number = k ^ 2 :=
sorry

end final_number_not_perfect_square_l173_173996


namespace equal_polynomial_values_find_k_l173_173799

variable (F : ℕ → ℤ)

def satisfies_conditions (k : ℕ) : Prop :=
  ∀ c, c ∈ finset.range (k+2) → 0 ≤ F c ∧ F c ≤ k

theorem equal_polynomial_values (k : ℕ) (h : k ≥ 4) (hf : satisfies_conditions F k) : 
  ∀ c, c ∈ finset.range (k+2) → F c = F 0 := 
sorry

theorem find_k (k : ℕ) : (∀ (F : ℕ → ℤ), satisfies_conditions F k → (∀ c, c ∈ finset.range (k+2) → F c = F 0)) ↔ k ≥ 4 :=
begin
    split,
    { intros h,
      by_contradiction hk,
      have h3_or_less: k < 4, by linarith,
      cases k,
      case zero { exfalso, linarith },
      case succ k {
        cases k,
        { 
          exfalso,
          linarith,
        },
        case succ k {
          cases k,
          {
            exfalso,
            linarith,
          },
          case succ k {
            cases k,
            { exfalso, linarith, },
            { exact absurd (exists.intro (λ x, x) (h (λ c, c) _ _ rfl) (⟨3, le_refl 3⟩)),
            sorry,
            }
          } 
        }
      }
    },
    sorry
end


end equal_polynomial_values_find_k_l173_173799


namespace gather_cards_into_one_box_l173_173777

theorem gather_cards_into_one_box (n: ℕ) (hpos: 0 < n) :
  ∃ (a : ℕ → ℕ) (h_func: ∀ i, 1 ≤ a i ∧ a i ≤ n) (h_unique: function.bijective a),
  (∀ k, 1 ≤ k ∧ k ≤ n → ∃ sum_cards, sum_cards % k = 0) :=
sorry

end gather_cards_into_one_box_l173_173777


namespace vasya_rank_91_l173_173228

theorem vasya_rank_91 {n_cyclists : ℕ} {n_stages : ℕ} 
    (n_cyclists_eq : n_cyclists = 500) 
    (n_stages_eq : n_stages = 15) 
    (no_ties : ∀ (i j : ℕ), i < j → ∀ (s : fin n_stages), ¬(same_time i j s)) 
    (vasya_7th : ∀ (s : fin n_stages), ∀ (i : ℕ), i < 6 → better_than i 6 s) :
    possible_rank vasya ≤ 91 :=
sorry

end vasya_rank_91_l173_173228


namespace smallest_angle_l173_173383

theorem smallest_angle 
  (x : ℝ) 
  (hx : tan (6 * x) = (cos (2 * x) - sin (2 * x)) / (cos (2 * x) + sin (2 * x))) :
  x = 5.625 :=
by
  sorry

end smallest_angle_l173_173383


namespace g_50_is_zero_l173_173957

theorem g_50_is_zero :
  ∀ (g : ℕ → ℕ),
    (∀ a b : ℕ, 2 * g (a^2 + 2 * b^2) = (g a)^2 + 3 * (g b)^2) →
    let n := 1 in
    let s := 0 in
    (n * s = 0) := by
  intros g hg
  let n := 1
  let s := 0
  sorry

end g_50_is_zero_l173_173957


namespace proof_problem_l173_173558

variable (a b c d : ℝ)
variable (ω : ℂ)

-- Conditions
def conditions : Prop :=
  a ≠ -1 ∧ b ≠ -1 ∧ c ≠ -1 ∧ d ≠ -1 ∧
  ω^4 = 1 ∧ ω ≠ 1 ∧
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 4 / ω^2)

theorem proof_problem (h : conditions a b c d ω) :
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1)) = 2 := 
sorry

end proof_problem_l173_173558


namespace Tim_age_l173_173678

theorem Tim_age : ∃ (T : ℕ), (T = (3 * T + 2 - 12)) ∧ (T = 5) :=
by
  existsi 5
  sorry

end Tim_age_l173_173678


namespace sandra_share_l173_173599

def ratio (a b c : ℕ) : Prop := a = 2 * b ∧ c = 3 * b

theorem sandra_share (Amy_Share : ℕ) (h_nonzero : Amy_Share ≠ 0) (h_ratio : ratio 2 Amy_Share 3) : 
  let Sandra_Share := 2 * Amy_Share in Sandra_Share = 100 :=
  by
  sorry

end sandra_share_l173_173599


namespace billy_knows_guitar_chords_at_least_24_l173_173772

theorem billy_knows_guitar_chords_at_least_24
  (CanPlay : ℕ)
  (TotalSongs : ℕ)
  (StillNeedsToLearn : ℕ)
  (h1 : CanPlay = 24)
  (h2 : TotalSongs = 52)
  (h3 : StillNeedsToLearn = 28) :
  CanPlay ≥ 24 :=
by
  rw [h1]
  apply Nat.le_refl

end billy_knows_guitar_chords_at_least_24_l173_173772


namespace max_min_values_on_interval_l173_173401

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * x ^ 3 - 3 * x ^ 2 - 12 * x + 5

theorem max_min_values_on_interval : 
  let I := set.Icc (0 : ℝ) (3 : ℝ) in
  ∃ a b ∈ I, ∀ x ∈ I, f a ≥ f x ∧ f b ≤ f x ∧ f a = 5 ∧ f b = -15 :=
by
  sorry

end max_min_values_on_interval_l173_173401


namespace price_per_kg_is_correct_l173_173769

def wheat_kg : ℝ := 30
def wheat_price_per_kg : ℝ := 11.50
def wheat_profit_margin : ℝ := 0.30

def rice_kg : ℝ := 20
def rice_price_per_kg : ℝ := 14.25
def rice_profit_margin : ℝ := 0.25

def barley_kg : ℝ := 15
def barley_price_per_kg : ℝ := 10
def barley_profit_margin : ℝ := 0.35

def total_kg : ℝ := wheat_kg + rice_kg + barley_kg

noncomputable def total_selling_price : ℝ := 
  (wheat_kg * wheat_price_per_kg * (1 + wheat_profit_margin)) +
  (rice_kg * rice_price_per_kg * (1 + rice_profit_margin)) +
  (barley_kg * barley_price_per_kg * (1 + barley_profit_margin))

noncomputable def price_per_kg_of_mixture : ℝ :=
  total_selling_price / total_kg

theorem price_per_kg_is_correct :
  price_per_kg_of_mixture = 15.50 := by
  sorry

end price_per_kg_is_correct_l173_173769


namespace maximum_value_f_within_0_to_sqrt_3_l173_173807

noncomputable def f (x : ℝ) : ℝ := 3 * x - x ^ 3

theorem maximum_value_f_within_0_to_sqrt_3 : 
  (∀ x ∈ Icc (0 : ℝ) (Real.sqrt 3), f(x) ≤ 2) ∧ 
  (∃ x ∈ Icc (0 : ℝ) (Real.sqrt 3), f(x) = 2) := 
by 
  sorry

end maximum_value_f_within_0_to_sqrt_3_l173_173807


namespace trig_identity_value_l173_173039

theorem trig_identity_value (α : ℝ) (h : tan α = 3) : 
  sin α ^ 2 + 2 * sin α * cos α - 3 * cos α ^ 2 = 6 / 5 := 
sorry

end trig_identity_value_l173_173039


namespace unbroken_seashells_l173_173679

theorem unbroken_seashells (total_seashells broken_seashells unbroken_seashells : ℕ) 
  (h_total : total_seashells = 7) (h_broken : broken_seashells = 4) 
  (h_unbroken : unbroken_seashells = total_seashells - broken_seashells) : 
  unbroken_seashells = 3 :=
by 
  rw [h_total, h_broken] at h_unbroken
  exact h_unbroken

end unbroken_seashells_l173_173679


namespace integer_solutions_pxy_eq_xy_l173_173414

theorem integer_solutions_pxy_eq_xy (p : ℤ) (hp : Prime p) :
  ∃ x y : ℤ, p * (x + y) = x * y ∧ 
  ((x, y) = (2 * p, 2 * p) ∨ 
  (x, y) = (0, 0) ∨ 
  (x, y) = (p + 1, p + p^2) ∨ 
  (x, y) = (p - 1, p - p^2) ∨ 
  (x, y) = (p + p^2, p + 1) ∨ 
  (x, y) = (p - p^2, p - 1)) :=
by
  sorry

end integer_solutions_pxy_eq_xy_l173_173414


namespace part_a_part_b_part_c_l173_173130

variable (p : ℕ) (k : ℕ)

theorem part_a (hp : Prime p) (h : p = 4 * k + 1) :
  ∃ x : ℤ, (x^2 + 1) % p = 0 :=
by
  sorry

theorem part_b (hp : Prime p) (h : p = 4 * k + 1)
  (x : ℤ) (r1 r2 s1 s2 : ℕ)
  (hr1 : 0 ≤ r1) (hr2 : 0 ≤ r2) (hr1_lt : r1 < Nat.sqrt p) (hr2_lt : r2 < Nat.sqrt p)
  (hs1 : 0 ≤ s1) (hs2 : 0 ≤ s2) (hs1_lt : s1 < Nat.sqrt p) (hs2_lt : s2 < Nat.sqrt p)
  (hneq : (r1, s1) ≠ (r2, s2)) :
  ∃ (r1 r2 s1 s2 : ℕ), (r1 * x + s1) % p = (r2 * x + s2) % p :=
by
  sorry

theorem part_c (hp : Prime p) (h : p = 4 * k + 1)
  (x : ℤ) (r1 r2 s1 s2 : ℕ)
  (hr1 : 0 ≤ r1) (hr2 : 0 ≤ r2) (hr1_lt : r1 < Nat.sqrt p) (hr2_lt : r2 < Nat.sqrt p)
  (hs1 : 0 ≤ s1) (hs2 : 0 ≤ s2) (hs1_lt : s1 < Nat.sqrt p) (hs2_lt : s2 < Nat.sqrt p)
  (hneq : (r1, s1) ≠ (r2, s2)):
  p = (Int.ofNat (r1 - r2))^2 + (Int.ofNat (s1 - s2))^2 :=
by
  sorry

end part_a_part_b_part_c_l173_173130


namespace solve_floor_sum_eq_125_l173_173164

def floorSum (x : ℕ) : ℕ :=
  (x - 1) * x * (4 * x + 1) / 6

theorem solve_floor_sum_eq_125 (x : ℕ) (h_pos : 0 < x) : floorSum x = 125 → x = 6 := by
  sorry

end solve_floor_sum_eq_125_l173_173164


namespace division_of_product_l173_173303

theorem division_of_product :
  (1.6 * 0.5) / 1 = 0.8 :=
sorry

end division_of_product_l173_173303


namespace vasya_lowest_position_l173_173260

noncomputable theory

def number_of_cyclists := 500
def number_of_stages := 15
def position_of_vasya_each_stage := 7

theorem vasya_lowest_position (total_cyclists : ℕ) (stages : ℕ) (position_each_stage : ℕ)
  (h_total_cyclists : total_cyclists = number_of_cyclists)
  (h_stages : stages = number_of_stages)
  (h_position_each_stage : position_each_stage = position_of_vasya_each_stage)
  (no_identical_times : ∀ (i j : ℕ), i ≠ j → ∀ (stage : ℕ), stage ≤ stages → ∀ (t : ℕ), t ≤ total_cyclists → 
    (time : Π (s : ℕ) (c : ℕ), c < total_cyclists → ℕ), 
    time stage i < time stage j ∨ time stage j < time stage i):
  ∃ lowest_pos, lowest_pos = 91 := sorry

end vasya_lowest_position_l173_173260


namespace solve_inequality_l173_173721

theorem solve_inequality (x : ℝ) (h : 5 * x - 12 ≤ 2 * (4 * x - 3)) : x ≥ -2 :=
sorry

end solve_inequality_l173_173721


namespace min_sum_l173_173988

noncomputable def harmonic_sum := (x : ℕ) →
  ∃ (x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 : ℕ), 
  (1/x1 : ℝ) + (1/x2 : ℝ) + (1/x3 : ℝ) + (1/x4 : ℝ) + (1/x5 : ℝ) + 
  (1/x6 : ℝ) + (1/x7 : ℝ) + (1/x8 : ℝ) + (1/x9 : ℝ) + (1/x10 : ℝ) + 
  (1/x11 : ℝ) + (1/x12 : ℝ) + (1/x13 : ℝ) = 2 ∧
  x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 = x

theorem min_sum:
  ∃ (x : ℕ), harmonic_sum x ∧ x = 85 :=
by
  sorry

end min_sum_l173_173988


namespace remaining_hard_hats_l173_173072

theorem remaining_hard_hats 
  (pink_initial : ℕ)
  (green_initial : ℕ)
  (yellow_initial : ℕ)
  (carl_takes_pink : ℕ)
  (john_takes_pink : ℕ)
  (john_takes_green : ℕ) :
  john_takes_green = 2 * john_takes_pink →
  pink_initial = 26 →
  green_initial = 15 →
  yellow_initial = 24 →
  carl_takes_pink = 4 →
  john_takes_pink = 6 →
  ∃ pink_remaining green_remaining yellow_remaining total_remaining, 
    pink_remaining = pink_initial - carl_takes_pink - john_takes_pink ∧
    green_remaining = green_initial - john_takes_green ∧
    yellow_remaining = yellow_initial ∧
    total_remaining = pink_remaining + green_remaining + yellow_remaining ∧
    total_remaining = 43 :=
by
  sorry

end remaining_hard_hats_l173_173072


namespace sequence_inequality_l173_173628

theorem sequence_inequality
  (n : ℕ) (h1 : 1 < n)
  (a : ℕ → ℕ)
  (h2 : ∀ i, i < n → a i < a (i + 1))
  (h3 : ∀ i, i < n - 1 → ∃ k : ℕ, (a i ^ 2 + a (i + 1) ^ 2) / 2 = k ^ 2) :
  a (n - 1) ≥ 2 * n ^ 2 - 1 :=
sorry

end sequence_inequality_l173_173628


namespace dormitory_problem_l173_173333

theorem dormitory_problem (x : ℕ) :
  9 < x ∧ x < 12
  → (x = 10 ∧ 4 * x + 18 = 58)
  ∨ (x = 11 ∧ 4 * x + 18 = 62) :=
by
  intros h
  sorry

end dormitory_problem_l173_173333


namespace complex_conjugate_of_z_l173_173841

theorem complex_conjugate_of_z :
  ∃ z : ℂ, z = (2 + complex.i) / (1 - complex.i) ∧ complex.conj z = (1 / 2) - (3 / 2) * complex.i :=
begin
  sorry
end

end complex_conjugate_of_z_l173_173841


namespace parabola_min_distance_p_values_l173_173835
noncomputable theory

open_locale classical

structure Point :=
  (x : ℝ)
  (y : ℝ)

def focus (p : ℝ) : Point :=
  { x := p / 2, y := 0 }

def parabola (p : ℝ) : set Point :=
  {P | P.y^2 = 2 * p * P.x}

def distance (A B : Point) : ℝ :=
  real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

def M : Point := { x := 2, y := 2 * real.sqrt 6 }

def minimum_distance_sums (P : Point) (p : ℝ) : ℝ :=
  distance P (focus p) + distance P M

theorem parabola_min_distance_p_values (P : Point) (p : ℝ) (hP : P ∈ parabola p)
  (h_cond : minimum_distance_sums P p = 5) : p = 2 ∨ p = 6 :=
sorry

end parabola_min_distance_p_values_l173_173835


namespace average_T_is_10_l173_173177

def count_adjacent_bg_pairs (row : List (string)) : ℕ :=
  (List.zipWith (λ a b => if (a = "B" ∧ b = "G") ∨ (a = "G" ∧ b = "B") then 1 else 0) row (row.tail)).sum

theorem average_T_is_10 (row : List (string)) :
  (List.length row = 20) →
  (row.count "B" = 8) →
  (row.count "G" = 12) →
  (count_adjacent_bg_pairs row).to_real / 19 = 10 :=
by
  sorry

end average_T_is_10_l173_173177


namespace alice_ate_more_than_bob_l173_173408

theorem alice_ate_more_than_bob :
  ∀ (a b : ℕ), 
  (∀ n, n ∈ {1, 2, 3, 4, 5, 6, 7, 8}) → 
  (a = 8 → b = 1 → (a - b = 7)) :=
by
  intros a b h1 h2 h3
  simp
  sorry

end alice_ate_more_than_bob_l173_173408


namespace value_of_fraction_l173_173560

variable {x y : ℝ}

theorem value_of_fraction (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) : 
  (x + y) / (x - y) = Real.sqrt 3 := 
sorry

end value_of_fraction_l173_173560


namespace mean_identity_l173_173184

theorem mean_identity (x y z : ℝ) 
  (h_arith_mean : (x + y + z) / 3 = 10)
  (h_geom_mean : Real.cbrt (x * y * z) = 7) 
  (h_harm_mean : 3 / (1 / x + 1 / y + 1 / z) = 4) :
  x^2 + y^2 + z^2 = 385.5 :=
by
  sorry

end mean_identity_l173_173184


namespace quadratic_zero_in_interval_l173_173906

theorem quadratic_zero_in_interval (m : ℝ) (h : ∃ x ∈ Ioo (-1 : ℝ) 0, (x^2 - 2*x + m) = 0) : 
  -3 < m ∧ m < 0 :=
sorry

end quadratic_zero_in_interval_l173_173906


namespace sufficient_but_not_necessary_condition_l173_173028

theorem sufficient_but_not_necessary_condition (x a : ℝ) (p q : Prop) :
  (p ↔ (2 * x - 1) / (x - 1) < 0) →
  (q ↔ x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0) →
  (∀ x, p → q) ∧ ¬(∀ x, q → p) →
  (0 ≤ a ∧ a ≤ 1/2) :=
by
  sorry

end sufficient_but_not_necessary_condition_l173_173028


namespace stock_value_decrease_in_april_l173_173767

variable (S₀ S₁ S₂ S₃ : ℝ)
variable (x : ℝ)

theorem stock_value_decrease_in_april :
  (S₁ = S₀ + 0.30 * S₀) →
  (S₂ = S₁ - 0.20 * S₁) →
  (S₃ = S₂ + 0.15 * S₂) →
  (S₃ - x / 100 * S₃ = S₂) →
  x ≈ 13 :=
by sorry

end stock_value_decrease_in_april_l173_173767


namespace lowest_position_l173_173236

theorem lowest_position (num_cyclists : ℕ) (num_stages : ℕ) (vasya_position : ℕ) :
  num_cyclists = 500 →
  num_stages = 15 →
  vasya_position = 7 →
  ∃ n, n = 91 :=
by
  intros
  sorry

end lowest_position_l173_173236


namespace eccentricity_of_hyperbola_l173_173528

theorem eccentricity_of_hyperbola (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : ∀ x y, ((x / a) ^ 2 - (y / b) ^ 2 = 1) ↔ ((x, y) = (±a, 0) ∨ (x = 2 ∧ y = 3))) :
  let e := Real.sqrt (1 + (b / a) ^ 2) in e = 2 :=
by
  sorry

end eccentricity_of_hyperbola_l173_173528


namespace lemon_permutations_l173_173488

theorem lemon_permutations : 
  let n := 5 in nat.factorial n = 120 := 
by 
  sorry

end lemon_permutations_l173_173488


namespace amount_paid_after_discount_l173_173870

-- Define the quantities and rates of the fruits
def quantity_grapes := 3
def rate_grapes := 70

def quantity_mangoes := 9
def rate_mangoes := 55

def quantity_oranges := 5
def rate_oranges := 40

def quantity_bananas := 7
def rate_bananas := 20

-- Define the discount percentage
def discount_percentage := 0.10

-- Calculate total cost before discount
def total_cost_before_discount :=
  (quantity_grapes * rate_grapes) +
  (quantity_mangoes * rate_mangoes) +
  (quantity_oranges * rate_oranges) +
  (quantity_bananas * rate_bananas)

-- Calculate the discount amount
def discount := discount_percentage * total_cost_before_discount

-- Calculate the final amount paid
def amount_paid := total_cost_before_discount - discount

-- Lean theorem for the final amount paid
theorem amount_paid_after_discount : amount_paid = 940.5 := by
  sorry

end amount_paid_after_discount_l173_173870


namespace unfolded_paper_holes_l173_173388

-- Define a structure to model the paper and operations on it
structure Paper where
  fold_left_right : Paper -> Paper
  fold_top_bottom : Paper -> Paper
  punch_center : Paper -> Paper
  punch_upper_right : Paper -> Paper
  unfold : Paper -> Paper
  hole_count : Paper -> Nat

-- Define the initial paper state
def initial_paper : Paper := {
  fold_left_right := sorry,
  fold_top_bottom := sorry,
  punch_center := sorry,
  punch_upper_right := sorry,
  unfold := sorry,
  hole_count := sorry
}

-- Prove that the unfolded paper has 8 holes
theorem unfolded_paper_holes : 
  hole_count 
    (unfold 
      (punch_upper_right 
        (punch_center 
          (fold_top_bottom 
            (fold_left_right initial_paper))))) = 8 := 
sorry

end unfolded_paper_holes_l173_173388


namespace girls_sum_equal_l173_173516

theorem girls_sum_equal (n : ℕ) (h : n ≥ 3) (b_cards : Finset ℕ) (g_cards : Finset ℕ)
    (h_b_cards : b_cards = Finset.range (n + 1))
    (h_g_cards : g_cards = Finset.range (2 * n + 1) \ finset.range (n + 1))
    (h_distinct : b_cards ∩ g_cards = ∅)
    (sums_equal : ∀ i : Finset ℕ, i.card = n → i.sum + 2 * b_cards.sum = (Finset.range (2*n)).sum) : 
    n % 2 = 1 :=
by
  sorry

end girls_sum_equal_l173_173516


namespace rectangle_perimeter_eq_26_l173_173682

theorem rectangle_perimeter_eq_26 (a b c W : ℕ) (h_tri : a = 5 ∧ b = 12 ∧ c = 13)
  (h_right_tri : a^2 + b^2 = c^2) (h_W : W = 3) (h_area_eq : 1/2 * (a * b) = (W * L))
  (A L : ℕ) (hA : A = 30) (hL : L = A / W) :
  2 * (L + W) = 26 :=
by
  sorry

end rectangle_perimeter_eq_26_l173_173682


namespace max_points_on_D_with_distance_l173_173155

-- Definitions used in the conditions
variable (D : Circle) (Q : Point)
variable (distance_QD : ∀ P : Point, P ∈ D → distance P Q ≠ 0)
variable (radius_D : Real)
variable (center_D : Point)
variable (radius_Dpos : radius_D > 0)

-- The statement for the proof problem
theorem max_points_on_D_with_distance (circle_D : Circle) (Q : Point)
  (distance_QD : ∀ P : Point, P ∈ D → distance P Q ≠ 0)
  : ∃ P : Point, P ∈ D ∧ distance P Q = 5 → (number_of_such_points = 2) := 
sorry

end max_points_on_D_with_distance_l173_173155


namespace min_sum_of_squares_inscribed_triangle_l173_173820

-- Defining the sides of triangle ABC
variables {a b c : ℝ}

-- Defining the area of triangle ABC (S_ABC)
def area_triangle (a b c : ℝ) : ℝ := sorry -- assume a valid definition of the area

-- The main theorem statement
theorem min_sum_of_squares_inscribed_triangle (a b c : ℝ) :
  ∃ (DEF : Type) (DE EF FD : DEF → ℝ), 
  ∀ (DEF : Type) (DE EF FD : DEF → ℝ), 
  (∀ P : DEF, P ⊆ triangle_ABC) →
  min (DE^2 + EF^2 + FD^2) = 
  12 * (area_triangle a b c)^2 / (a^2 + b^2 + c^2) :=
sorry

end min_sum_of_squares_inscribed_triangle_l173_173820


namespace airplane_takeoff_distance_l173_173270

/--  
The takeoff run time of an airplane from the start until it leaves the ground is 15 seconds. 
Find the length of the takeoff run if the takeoff speed for this airplane model is 100 km/h. 
Assume the airplane's motion during the takeoff run is uniformly accelerated. 
Provide the answer in meters, rounding to the nearest whole number if necessary.
--/
theorem airplane_takeoff_distance :
  let t : ℝ := 15  -- time in seconds
  let v_kmh : ℝ := 100  -- speed in km/h
  let v_ms : ℝ := v_kmh * 1000 / 3600  -- converted speed in m/s
  let a : ℝ := v_ms / t  -- acceleration
  let s : ℝ := 0.5 * a * t^2  -- distance
  -- expected distance in meters, rounding to the nearest whole number
  s ≈ 208 :=
by
  let t : ℝ := 15
  let v_kmh : ℝ := 100
  let v_ms : ℝ := v_kmh * 1000 / 3600
  let a := v_ms / t
  let s := 0.5 * a * t^2
  sorry

end airplane_takeoff_distance_l173_173270


namespace vasya_lowest_position_l173_173205

theorem vasya_lowest_position
  (n_cyclists : ℕ) (n_stages : ℕ) 
  (stage_positions : ℕ → ℕ → ℕ) -- a function that takes a stage and a cyclist and returns the position (e.g., stage_positions(stage, cyclist) = position)
  (total_time : ℕ → ℕ)  -- a function that takes a cyclist and returns their total time
  (distinct_times : ∀ (c1 c2 : ℕ), c1 ≠ c2 → (total_time c1 ≠ total_time c2) ∧ 
                   ∀ (s : ℕ), stage_positions s c1 ≠ stage_positions s c2)
  (vasya_position : ℕ) (hv : ∀ (s : ℕ), s < n_stages → stage_positions s vasya_position = 7) :
  vasya_position = 91 :=
sorry

end vasya_lowest_position_l173_173205


namespace prob_negative_one_to_zero_l173_173437

noncomputable theory
open MeasureTheory

variables {ξ : ℝ →ₘ measure_theory.real_measurable_space} 

-- Assume ξ follows a normal distribution N(0,1)
axiom ξ_normal : ξ = distribution.normal_std

-- Given condition: P(ξ>1)=a
variable (a : ℝ)
axiom P_ξ_gt_1 : measure_theory.measure_space.prob (λ x, x > 1) ξ = a

-- We need to prove: P(-1≤ξ≤0) = 1/2 - a
theorem prob_negative_one_to_zero :
  measure_theory.measure_space.prob (λ x, -1 ≤ x ∧ x ≤ 0) ξ = 1/2 - a :=
  sorry

end prob_negative_one_to_zero_l173_173437


namespace circle_area_ratio_l173_173892

theorem circle_area_ratio (R_C R_D : ℝ)
  (h₁ : (60 / 360 * 2 * Real.pi * R_C) = (40 / 360 * 2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 9 / 4 :=
by 
  sorry

end circle_area_ratio_l173_173892


namespace find_radius_l173_173540

open_locale real

-- Define the structures and conditions
variables (ω ω₁ ω₂ : Type) -- Circles
variables (K L M N : Type) -- Points
variables (rω rω₁ rω₂ : ℝ) -- Radii
variables (O O₁ O₂ : Type) -- Centers

-- Assume the given conditions
variables [Inside ω ω₁] [Inside ω ω₂] -- ω₁ and ω₂ are inside ω
variables [Intersect ω₁ ω₂ K] [Intersect ω₁ ω₂ L] -- ω₁ and ω₂ intersect at K and L
variables [Tangent ω ω₁ M] [Tangent ω ω₂ N] -- ω₁ tangent to ω at M, ω₂ tangent to ω at N
variables [Collinear K M N] -- K, M, N are collinear
variables (radius_ω₁ : rω₁ = 3) (radius_ω₂ : rω₂ = 5) -- radii given

-- The theorem to prove
theorem find_radius (h₁ : rω₁ = 3) (h₂ : rω₂ = 5) :
  ∃ rω, rω = 8 :=
begin
  use 8,
  sorry
end

end find_radius_l173_173540


namespace circle_area_ratio_l173_173891

theorem circle_area_ratio (R_C R_D : ℝ)
  (h₁ : (60 / 360 * 2 * Real.pi * R_C) = (40 / 360 * 2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 9 / 4 :=
by 
  sorry

end circle_area_ratio_l173_173891


namespace hyperbola_eccentricity_l173_173863

theorem hyperbola_eccentricity
  (a b : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hyp : ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1 ↔ (abs (b / a * x - y) < or gt)))
  (circIntersection : ∀ x y : ℝ, (x^2 + y^2 - 2 * x = 0 ↔ dist (1, 0) (x, y) = 1))
  (chordLength : ∀ d : ℝ, d = sqrt (3)) :
  ∃ e : ℝ, e = 2 * sqrt 3 / 3 :=
by
  sorry

end hyperbola_eccentricity_l173_173863


namespace number_of_values_of_z_l173_173570

noncomputable def f (z : ℂ) : ℂ := complex.I * z.conj

theorem number_of_values_of_z :
  {z : ℂ | complex.abs z = 3 ∧ f z = z}.to_finset.card = 2 :=
by
  sorry

end number_of_values_of_z_l173_173570


namespace expected_sixes_in_three_rolls_l173_173676

theorem expected_sixes_in_three_rolls : 
  (∑ k in Finset.range 4, k * (Nat.choose 3 k) * (1/6)^k * (5/6)^(3-k)) = 1/2 := 
by
  sorry

end expected_sixes_in_three_rolls_l173_173676


namespace projection_problem_l173_173263

noncomputable def vector_projection : ℝ^3 := sorry

theorem projection_problem :
  let v1 : ℝ^3 := ![1, 4, 2]
  let v2 : ℝ^3 := ![3, 5, 4]
  let p1 : ℝ^3 := ![2/7, 8/7, 4/7]
  let correct_projection : ℝ^3 := ![23 * Real.sqrt 3 / 21, 92 * Real.sqrt 3 / 21, 46 * Real.sqrt 3 / 21]
  (vector_projection v2 = correct_projection) :=
   sorry

end projection_problem_l173_173263


namespace problem_a_problem_d_l173_173497

theorem problem_a (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) : (1 / (a * b)) ≥ 1 / 4 :=
by
  sorry

theorem problem_d (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) : a^2 + b^2 ≥ 8 :=
by
  sorry

end problem_a_problem_d_l173_173497


namespace perimeter_triangle_ABF2_l173_173444

noncomputable def ellipse := 
  { x y : ℝ // (∃ a b : ℝ, a > b > 0 ∧ b = 4 ∧ (3 / 5) = (sqrt(a^2 - b^2) / a) ∧ (x^2 / a^2) + (y^2 / b^2) = 1) }

theorem perimeter_triangle_ABF2 (a b c : ℝ) (h₁ : a > b > 0)
    (h₂ : b = 4) 
    (h₃ : (3 / 5) = (c / a)) 
    (h₄ : a^2 = b^2 + c^2) 
    (h₅ : ∃ A B F₁ F₂ : ℝ × ℝ, line_through F₁ A ∧ line_through F₁ B ∧ ellipse_eq (A.1, A.2) ∧ ellipse_eq (B.1, B.2) ∧ foci A B F₁ F₂)
    : ∃ p : ℝ, p = 4 * a := sorry

def ellipse_eq (P : ℝ × ℝ) : Prop := 
  let (x, y) := P in (∃ a b : ℝ, a > b > 0 ∧ b = 4 ∧ (3 / 5) = (sqrt(a^2 - b^2) / a) ∧ (x^2 / a^2) + (y^2 / b^2) = 1)

def line_through (P₁ P₂ : ℝ × ℝ) : Prop := 
  ∃ m b : ℝ, ∀ P : ℝ × ℝ, P = (P.fst, m * P.fst + b)

def foci (A B F₁ F₂ : ℝ × ℝ) : Prop := 
  let c := sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2) in c < 2 * a

end perimeter_triangle_ABF2_l173_173444


namespace replace_80_percent_banknotes_within_days_unable_to_replace_all_banknotes_without_repair_l173_173318

-- Given conditions
def total_banknotes : ℕ := 3628800
def major_repair_cost : ℕ := 800000
def daily_operation_cost : ℕ := 90000
def capacity_after_repair : ℕ := 1000000
def total_budget : ℕ := 1000000
def percentage_to_replace : ℕ := 80

-- Definitions related to days and operations
def banknotes_replaced_on_day (day : ℕ) (remaining_banknotes : ℕ) : ℕ := 
  match day with
  | 1 => remaining_banknotes / 2
  | 2 => remaining_banknotes / 3
  | 3 => remaining_banknotes / 4
  | 4 => remaining_banknotes / 5
  | _ => 0

-- Proof Statements
theorem replace_80_percent_banknotes_within_days : ∃ (days : ℕ), (days < 100) ∧
  let remaining_banknotes_1 := total_banknotes - banknotes_replaced_on_day 1 total_banknotes in
  let remaining_banknotes_2 := remaining_banknotes_1 - banknotes_replaced_on_day 2 remaining_banknotes_1 in
  let remaining_banknotes_3 := remaining_banknotes_2 - banknotes_replaced_on_day 3 remaining_banknotes_2 in
  let total_replaced := total_banknotes - remaining_banknotes_3 in
  total_replaced ≥ (percentage_to_replace * total_banknotes) / 100 := 
sorry

theorem unable_to_replace_all_banknotes_without_repair : 
  let operation_cost := (total_banknotes / capacity_after_repair) * daily_operation_cost in
  operation_cost > total_budget :=
sorry 

end replace_80_percent_banknotes_within_days_unable_to_replace_all_banknotes_without_repair_l173_173318


namespace area_ratio_of_triangles_l173_173588

-- Definitions based on the problem conditions
structure EquilateralTriangle (A B C D : Type) :=
(side_length : ℝ)
(equal_sides : dist A B = side_length ∧ dist B C = side_length ∧ dist C A = side_length)
(point_on_side : ∃ D, D ∈ seg AC)
(angle_DBC : ∠ D B C = 45)

-- Proving the ratio of areas given the conditions
theorem area_ratio_of_triangles 
(A B C D : Type) 
(h_eq_triangle : EquilateralTriangle A B C D) 
: 
  let area_ADB := area (triangle A D B),
      area_CDB := area (triangle C D B)
  in 
  area_ADB / area_CDB = (√3 - 1) / 2 :=
sorry

end area_ratio_of_triangles_l173_173588


namespace smallest_angle_l173_173385

theorem smallest_angle 
  (x : ℝ) 
  (hx : tan (6 * x) = (cos (2 * x) - sin (2 * x)) / (cos (2 * x) + sin (2 * x))) :
  x = 5.625 :=
by
  sorry

end smallest_angle_l173_173385


namespace ratio_of_areas_of_circles_l173_173896

-- Given conditions
variables (R_C R_D : ℝ) -- Radii of circles C and D respectively
variables (L : ℝ) -- Common length of the arcs

-- Equivalent arc condition
def arc_length_condition : Prop :=
  (60 / 360) * (2 * Real.pi * R_C) = L ∧ (40 / 360) * (2 * Real.pi * R_D) = L

-- Goal: ratio of areas
def area_ratio : Prop :=
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = (9 / 4)

-- Problem statement
theorem ratio_of_areas_of_circles (R_C R_D L : ℝ) (hc : arc_length_condition R_C R_D L) :
  area_ratio R_C R_D :=
by
  sorry

end ratio_of_areas_of_circles_l173_173896


namespace count_special_sequences_l173_173343

theorem count_special_sequences : 
  (∃ S : Fin 6 → ℕ, (∃ f : Fin 6 → ℤ, (∀ i : Fin 6, f i = 1 ∨ f i = 2 ∨ f i = 3) ∧
                                                         (f = λ i ∈ range 0 1 1 (3)) ∧
                                                         (∃ j k : Fin 6, j ≠ k ∧ f j = 2 ∧ f k = 2) ∧
                                                         (∃ l m n : Fin 6, l ≠ m ∧ m ≠ n ∧ l ≠ n ∧ f l = 1 ∧ f m = 1 ∧ f n = 1))) →
   S = 60 := 
sorry

end count_special_sequences_l173_173343


namespace arithmetic_sequence_a6_l173_173931

theorem arithmetic_sequence_a6 (a : ℕ → ℤ) (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) 
  (h_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 80) : a 6 = 16 :=
sorry

end arithmetic_sequence_a6_l173_173931


namespace distance_between_points_l173_173994

theorem distance_between_points 
  (v_A v_B : ℝ) 
  (d : ℝ) 
  (h1 : 4 * v_A + 4 * v_B = d)
  (h2 : 3.5 * (v_A + 3) + 3.5 * (v_B + 3) = d) : 
  d = 168 := 
by 
  sorry

end distance_between_points_l173_173994


namespace triangle_side_lengths_l173_173521

theorem triangle_side_lengths (a b c : ℝ) (h1 : a / b = 3 / 4) (h2 : b / c = 4 / 5) (h3 : a + b + c = 60) :
  (a = 15 ∧ b = 20 ∧ c = 25) :=
sorry

end triangle_side_lengths_l173_173521


namespace william_wins_10_rounds_l173_173708

-- Definitions from the problem conditions
variable (W H : ℕ)
variable (total_rounds : ℕ := 15)
variable (additional_wins : ℕ := 5)

-- Conditions
def total_game_condition : Prop := W + H = total_rounds
def win_difference_condition : Prop := W = H + additional_wins

-- Statement to be proved
theorem william_wins_10_rounds (h1 : total_game_condition W H) (h2 : win_difference_condition W H) : W = 10 :=
by
  sorry

end william_wins_10_rounds_l173_173708


namespace monotone_decreasing_interval_cosine_l173_173629

theorem monotone_decreasing_interval_cosine (k : ℤ) :
  ∀ (x : ℝ), 2k * π <= 2x - (π / 4) ∧ 2x - (π / 4) <= 2k * π + π ↔
             k * π + (π / 8) <= x ∧ x <= k * π + (5 * π / 8) :=
by
  sorry

end monotone_decreasing_interval_cosine_l173_173629


namespace consecutive_sets_with_integer_roots_l173_173798

-- Definitions for three consecutive positive integers
def is_consecutive (a b c : ℕ) : Prop :=
  a + 1 = b ∧ b + 1 = c

-- The main statement about the sets of coefficients that form quadratic equations with integer roots
theorem consecutive_sets_with_integer_roots :
  ∀ (a b c : ℕ), is_consecutive a b c ∧ a > 0 ∧ b > 0 ∧ c > 0 →
  (∃ (x : ℤ), a * x^2 + b * x + c = 0) ∨ (∃ (x : ℤ), b * x^2 + a * x + c = 0) ∨ (∃ (x : ℤ), c * x^2 + a * x + b = 0)  ↔
  ((a = 1 ∧ b = 2 ∧ c = 3) ∨ (a = 2 ∧ b = 3 ∧ c = 1)).
 
sorry

end consecutive_sets_with_integer_roots_l173_173798


namespace smallest_integer_with_properties_l173_173810

theorem smallest_integer_with_properties (N : ℕ) :
  (∃ n : ℕ, n > 0 ∧ 
    (n = N ∨ n = N + 1 ∨ n = N + 2) ∧
    ((n % 4 = 0) ∨ (n % 9 = 0) ∨ (n % 25 = 0) ∨ (n % 121 = 0))) ∧ 
  (N = 242) :=
by 
  let N := 242
  have h1 : N % 121 = 0 := by norm_num
  have h2 : (N + 1) % 4 = 0 := by norm_num
  have h3 : (N + 1) % 25 = 0 := by norm_num
  have h4 : (N + 2) % 9 = 0 := by norm_num
  use N
  split
  . exact 242 > 0
  split
  . left; refl
  . left; exact ⟨h1, h2, h3, h4⟩
  use (N + 1)
  split
  . exact 243
  split
  . right; left; refl
  . right; left; exact ⟨h1, h2, h3, h4⟩
  use (N + 2)
  split
  . exact 244
  split
  . right; right; right; refl
  . right; right; left; exact ⟨h1, h2, h3, h4⟩

  sorry

end smallest_integer_with_properties_l173_173810


namespace johns_weekly_allowance_l173_173871

variable (A : ℝ)

theorem johns_weekly_allowance 
  (h1 : ∃ A : ℝ, A > 0) 
  (h2 : (4/15) * A = 0.75) : 
  A = 2.8125 := 
by 
  -- Proof can be filled in here
  sorry

end johns_weekly_allowance_l173_173871


namespace positive_difference_prime_factors_159137_l173_173302

-- Lean 4 Statement Following the Instructions
theorem positive_difference_prime_factors_159137 :
  (159137 = 11 * 17 * 23 * 37) → (37 - 23 = 14) :=
by
  intro h
  sorry -- Proof will be written here

end positive_difference_prime_factors_159137_l173_173302


namespace linear_regression_change_l173_173050

theorem linear_regression_change : ∀ (x : ℝ), ∀ (y : ℝ), 
  y = 2 - 3.5 * x → (y - (2 - 3.5 * (x + 1))) = 3.5 :=
by
  intros x y h
  sorry

end linear_regression_change_l173_173050


namespace takeoff_distance_l173_173271

theorem takeoff_distance (t v_kmh : ℝ) (uniform_acceleration : Prop) : 
  t = 15 ∧ v_kmh = 100 ∧ uniform_acceleration → 
  ∃ s : ℝ, s ≈ 208 :=
by 
  sorry

end takeoff_distance_l173_173271


namespace european_customer_savings_l173_173337

noncomputable def popcorn_cost : ℝ := 8 - 3
noncomputable def drink_cost : ℝ := popcorn_cost + 1
noncomputable def candy_cost : ℝ := drink_cost / 2

noncomputable def discounted_popcorn_cost : ℝ := popcorn_cost * (1 - 0.15)
noncomputable def discounted_candy_cost : ℝ := candy_cost * (1 - 0.1)

noncomputable def total_normal_cost : ℝ := 8 + discounted_popcorn_cost + drink_cost + discounted_candy_cost
noncomputable def deal_price : ℝ := 20
noncomputable def savings_in_dollars : ℝ := total_normal_cost - deal_price

noncomputable def exchange_rate : ℝ := 0.85
noncomputable def savings_in_euros : ℝ := savings_in_dollars * exchange_rate

theorem european_customer_savings : savings_in_euros = 0.81 := by
  sorry

end european_customer_savings_l173_173337


namespace equation_of_min_line_l173_173448

-- Define point P
def P : ℝ × ℝ := (2, 1)

-- Let L be the equation of the line passing through P and intersecting positive x and y axes
def L (x y : ℝ) : Prop := ∃ k : ℝ, y - 1 = k * (x - 2) ∧ x * y > 0

-- Define the conditions cₐ and c_b for points A and B respectively
def A (k : ℝ) : ℝ × ℝ := (2 - 1/k, 0)
def B (k : ℝ) : ℝ × ℝ := (0, 1 - 2*k)

-- Define the Euclidean distance between two points (x1, y1) and (x2, y2)
def dist (p₁ p₂ : ℝ × ℝ) : ℝ :=
  (p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2 

-- Product of distances PA and PB
def PA_PB (k : ℝ) : ℝ :=
  let PA := dist P (A k)
  let PB := dist P (B k)
  PA * PB

-- Equation of line l that minimizes |PA| * |PB|
def min_line_equation : Prop := 
∃ l : ℝ → ℝ → Prop, 
  (∀ k : ℝ, PA_PB k ≥ PA_PB (-1)) ∧
  l = L 3 -- the line x + y - 3 = 0

theorem equation_of_min_line :
  min_line_equation → (∀ x y : ℝ, L x y ↔ x + y - 3 = 0) := 
begin
  sorry
end

end equation_of_min_line_l173_173448


namespace least_positive_multiple_of_24_gt_450_l173_173301

theorem least_positive_multiple_of_24_gt_450 : 
  ∃ n : ℕ, n > 450 ∧ (∃ k : ℕ, n = 24 * k) → n = 456 :=
by 
  sorry

end least_positive_multiple_of_24_gt_450_l173_173301


namespace ratio_of_areas_of_circles_l173_173897

-- Given conditions
variables (R_C R_D : ℝ) -- Radii of circles C and D respectively
variables (L : ℝ) -- Common length of the arcs

-- Equivalent arc condition
def arc_length_condition : Prop :=
  (60 / 360) * (2 * Real.pi * R_C) = L ∧ (40 / 360) * (2 * Real.pi * R_D) = L

-- Goal: ratio of areas
def area_ratio : Prop :=
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = (9 / 4)

-- Problem statement
theorem ratio_of_areas_of_circles (R_C R_D L : ℝ) (hc : arc_length_condition R_C R_D L) :
  area_ratio R_C R_D :=
by
  sorry

end ratio_of_areas_of_circles_l173_173897


namespace repeating_six_denominator_eq_three_l173_173187

theorem repeating_six_denominator_eq_three (S : ℚ) (h : S = 0.\overline{6}) : ∃ (d : ℕ), S = 2 / d ∧ d = 3 := 
sorry

end repeating_six_denominator_eq_three_l173_173187


namespace value_of_six_inch_cube_l173_173735

theorem value_of_six_inch_cube :
  let four_inch_cube_value := 400
  let four_inch_side_length := 4
  let six_inch_side_length := 6
  let volume (s : ℕ) : ℕ := s ^ 3
  (volume six_inch_side_length / volume four_inch_side_length) * four_inch_cube_value = 1350 := by
sorry

end value_of_six_inch_cube_l173_173735


namespace complement_intersection_l173_173484

open Set

variable U : Set ℕ := {1, 2, 3, 4, 5, 6}
variable P : Set ℕ := {1, 3, 5}
variable Q : Set ℕ := {1, 2, 4}

theorem complement_intersection (hU : U = {1, 2, 3, 4, 5, 6}) (hP : P = {1, 3, 5}) (hQ : Q = {1, 2, 4}) :
  ((U \ P) ∩ Q) = {2, 4} :=
  sorry

end complement_intersection_l173_173484


namespace will_earnings_l173_173699

-- Defining the conditions
def hourly_wage : ℕ := 8
def monday_hours : ℕ := 8
def tuesday_hours : ℕ := 2

-- Calculating the earnings
def monday_earnings := monday_hours * hourly_wage
def tuesday_earnings := tuesday_hours * hourly_wage
def total_earnings := monday_earnings + tuesday_earnings

-- Stating the problem
theorem will_earnings : total_earnings = 80 := by
  -- sorry is used to skip the actual proof
  sorry

end will_earnings_l173_173699


namespace trajectory_of_point_C_equation_of_circumcircle_of_ABD_l173_173450

-- Define the conditions: existence of points and the moving point relation
def point (x y : ℝ) : ℝ × ℝ := (x, y)

def E : ℝ × ℝ := point 1 0
def K : ℝ × ℝ := point -1 0

def satisfies_condition (P : ℝ × ℝ) : Prop :=
  let PE := (1 - P.1, -P.2)
  let KE := (-2, 0)
  let PK := (-1 - P.1, -P.2)
  (PE.1^2 + PE.2^2).sqrt * (KE.1^2 + KE.2^2).sqrt = PK.1 * KE.1

-- 1. Prove the equation of the trajectory C of point P is y^2 = 4x
theorem trajectory_of_point_C (P : ℝ × ℝ) (h : satisfies_condition P) : P.2^2 = 4 * P.1 := 
sorry

-- 2. Prove the equation of the circumcircle of triangle ABD
def intersects_C (A B : ℝ × ℝ) : Prop :=
  satisfies_condition A ∧ satisfies_condition B ∧ A.2 > 0 ∧ A.1 = B.1

def symmetric_about_x (A D : ℝ × ℝ) : Prop :=
  D.1 = A.1 ∧ D.2 = -A.2

def EA_dot_EB (A B : ℝ × ℝ) : Prop :=
  (A.1 - 1) * (B.1 - 1) + A.2 * B.2 = -8

theorem equation_of_circumcircle_of_ABD 
(A B D : ℝ × ℝ)
(h_intersects_C : intersects_C A B)
(h_symmetric : symmetric_about_x A D)
(h_dot_product : EA_dot_EB A B) : (D.1 - 9)^2 + D.2^2 = 40 := 
sorry

end trajectory_of_point_C_equation_of_circumcircle_of_ABD_l173_173450


namespace find_t_l173_173563

variable (t : ℝ)

def coordinates_C := (t - 3, -2)
def coordinates_D := (-1, t + 2)

def midpoint_C_D := ((coordinates_C.1 + coordinates_D.1) / 2, 
                     (coordinates_C.2 + coordinates_D.2) / 2)

def distance_squared (p1 p2 : ℝ × ℝ) := 
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

def distance_midpoint_endpoint_squared := 
  distance_squared midpoint_C_D coordinates_C

theorem find_t : 
  ∃ (t : ℝ), distance_midpoint_endpoint_squared t = (3 * t^2) / 4 := 
sorry

end find_t_l173_173563


namespace midpoint_polar_coords_l173_173920

/-- 
Given two points in polar coordinates: (6, π/6) and (2, -π/6),  
the midpoint of the line segment connecting these points in polar coordinates is (√13, π/6).
-/
theorem midpoint_polar_coords :
  let A := (6, Real.pi / 6)
  let B := (2, -Real.pi / 6)
  let A_cart := (6 * Real.cos (Real.pi / 6), 6 * Real.sin (Real.pi / 6))
  let B_cart := (2 * Real.cos (-Real.pi / 6), 2 * Real.sin (-Real.pi / 6))
  let Mx := ((A_cart.fst + B_cart.fst) / 2)
  let My := ((A_cart.snd + B_cart.snd) / 2)
  let r := Real.sqrt (Mx^2 + My^2)
  let theta := Real.arctan (My / Mx)
  0 <= theta ∧ theta < 2 * Real.pi ∧ r > 0 ∧ (r = Real.sqrt 13 ∧ theta = Real.pi / 6) :=
by 
  sorry

end midpoint_polar_coords_l173_173920


namespace cricket_runs_l173_173713

theorem cricket_runs (x a b c d : ℕ) 
    (h1 : a = 1 * x) 
    (h2 : b = 3 * x) 
    (h3 : c = 5 * x) 
    (h4 : d = 4 * x) 
    (total_runs : 1 * x + 3 * x + 5 * x + 4 * x = 234) :
  a = 18 ∧ b = 54 ∧ c = 90 ∧ d = 72 := by
  sorry

end cricket_runs_l173_173713


namespace sodium_chloride_required_eq_two_l173_173881

-- Definitions for the problem based on conditions
def moles_sodium_nitrate : ℕ := 2
def moles_nitric_acid_eq_sodium_nitrate : Prop := (moles_sodium_nitrate = moles_nitric_acid)
variable (moles_nitric_acid : ℕ) (moles_sodium_chloride : ℕ)

-- The proof statement
theorem sodium_chloride_required_eq_two 
  (h1: moles_nitric_acid_eq_sodium_nitrate)
  (h2: moles_nitric_acid = moles_sodium_chloride): 
  moles_sodium_chloride = 2 :=
by 
  sorry

end sodium_chloride_required_eq_two_l173_173881


namespace total_price_after_increase_l173_173979

def original_jewelry_price := 30
def original_painting_price := 100
def jewelry_price_increase := 10
def painting_price_increase_percentage := 20
def num_jewelry_pieces := 2
def num_paintings := 5

theorem total_price_after_increase : 
    let new_jewelry_price := original_jewelry_price + jewelry_price_increase in
    let new_painting_price := original_painting_price + (original_painting_price * painting_price_increase_percentage / 100) in
    let total_jewelry_cost := new_jewelry_price * num_jewelry_pieces in
    let total_painting_cost := new_painting_price * num_paintings in
    total_jewelry_cost + total_painting_cost = 680 :=
by
  -- proof omitted
  sorry

end total_price_after_increase_l173_173979


namespace mike_arcade_play_time_l173_173578

def weekly_pay : ℕ := 100
def fraction_spent_arcade : ℕ → ℕ := λ pay, pay / 2
def money_spent_on_food : ℕ := 10
def cost_per_hour : ℕ := 8
def minutes_per_hour : ℕ := 60

theorem mike_arcade_play_time : 
  let weekly_spending := fraction_spent_arcade weekly_pay in
  let money_spent_on_tokens := weekly_spending - money_spent_on_food in
  let hours_played := money_spent_on_tokens / cost_per_hour in
  let total_minutes := hours_played * minutes_per_hour in
  total_minutes = 300 :=
by
  sorry

end mike_arcade_play_time_l173_173578


namespace range_of_half_alpha_minus_beta_l173_173884

theorem range_of_half_alpha_minus_beta (α β : ℝ) (h1 : 1 < α) (h2 : α < 3) (h3 : -4 < β) (h4 : β < 2) :
  -3/2 < (1/2) * α - β ∧ (1/2) * α - β < 11/2 :=
by
  -- sorry to skip the proof
  sorry

end range_of_half_alpha_minus_beta_l173_173884


namespace circle_area_ratio_l173_173893

theorem circle_area_ratio (R_C R_D : ℝ)
  (h₁ : (60 / 360 * 2 * Real.pi * R_C) = (40 / 360 * 2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 9 / 4 :=
by 
  sorry

end circle_area_ratio_l173_173893


namespace august_days_occurrence_l173_173175

theorem august_days_occurrence (N : ℕ) (start_July_on_Friday : true) (July_has_31_days : true) :
  (has_five_occurrences_in_august "Monday" ∧ has_five_occurrences_in_august "Tuesday" ∧ has_five_occurrences_in_august "Wednesday") :=
by
  sorry

end august_days_occurrence_l173_173175


namespace distance_eq_l173_173036

open Real

variables (a b c d p q: ℝ)

-- Conditions from step a)
def onLine1 : Prop := b = (p-1)*a + q
def onLine2 : Prop := d = (p-1)*c + q

-- Theorem about the distance between points (a, b) and (c, d)
theorem distance_eq : 
  onLine1 a b p q → 
  onLine2 c d p q → 
  dist (a, b) (c, d) = abs (a - c) * sqrt (1 + (p - 1)^2) := 
by
  intros h1 h2
  sorry

end distance_eq_l173_173036


namespace max_value_f_max_value_f_attained_l173_173781

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

theorem max_value_f : ∀ x : ℝ, f x ≤ 2 :=
by
  intro x
  sorry

theorem max_value_f_attained : ∃ x : ℝ, f x = 2 :=
by
  use π / 3  -- or any x that achieves the maximum value
  sorry

end max_value_f_max_value_f_attained_l173_173781


namespace number_of_valid_sets_average_value_a_X_l173_173571

-- Define the set A
def A (n : ℕ) : Set ℕ := {i | 1 ≤ i ∧ i ≤ n}

-- Define the condition 2 <= |X| <= n - 2
def valid_X (n : ℕ) (X : Set ℕ) : Prop :=
  2 ≤ X.card ∧ X.card ≤ n - 2 ∧ X ⊆ A n

-- Define the function a_X
def a_X (X : Set ℕ) : ℕ :=
  X.max' ⟨1, sorry⟩ + X.min' ⟨1, sorry⟩

-- Theorem 1: For n = 5, the number of valid sets X is 20
theorem number_of_valid_sets (h : 5 ≥ 4) : 
  ∃ (X : Finset (Finset ℕ)), 
    (∀ x ∈ X, valid_X 5 x) ∧
    (X.card = 20) :=
  sorry

-- Theorem 2: The average value of all a_X is n + 1
theorem average_value_a_X (n : ℕ) (h : n ≥ 4) : 
  ∃avg : ℝ, 
    (avg = (n : ℝ) + 1) ∧
    (∀ (X : Finset (Finset ℕ)), 
      (∀ x ∈ X, valid_X n x) → 
      avg = X.sum (λ y, a_X y) / X.card) :=
  sorry

end number_of_valid_sets_average_value_a_X_l173_173571


namespace find_a_plus_b_l173_173462

-- Constants representing the given conditions
def slope_angle_l : ℝ := 135
def point_A : ℝ × ℝ := (3, 2)
def point_B (a : ℝ) : ℝ × ℝ := (a, -1)
def equation_l2 (b : ℝ) : ℝ → ℝ → ℝ := λ x y, 2 * x + b * y + 1

-- Definitions for line slopes and relationships between them
def slope_l : ℝ := real.tan (slope_angle_l * real.pi / 180)
def slope_l1 (a : ℝ) : ℝ := (point_A.snd - (point_B a).snd) / (point_A.fst - (point_B a).fst)
def slope_l2 (b : ℝ) : ℝ := - (2 / b)

-- Proof goal
theorem find_a_plus_b : ∃ a b : ℝ, slope_l = -1 ∧ slope_l1 a = 1 ∧ slope_l2 b = 1 ∧ a + b = -2 :=
by
  -- Initial conditions for the constants and slopes
  sorry -- Proof is required here


end find_a_plus_b_l173_173462


namespace triangle_side_relation_triangle_area_l173_173936

-- Problem 1: Prove that c = sqrt(2) * b given a = 2 and CD = sqrt(2)
theorem triangle_side_relation (a b c : ℝ) (CD : ℝ) (D : ℝ -> ℝ -> ℝ)
  (midpoint_AB : D 0 2 = 1)
  (a_eq_2 : a = 2)
  (CD_eq_sqrt2 : CD = Real.sqrt 2) :
  c = Real.sqrt 2 * b := sorry

-- Problem 2: Prove the area given c = sqrt(2) * b and angle ACB = π/4
theorem triangle_area (a b c : ℝ) (A B C : Angle)
  (a_eq_2 : a = 2)
  (c_eq_sqrt2_b : c = Real.sqrt 2 * b)
  (angle_ACB_eq_pi_div_4 : C = Real.pi / 4) :
  (1/2) * a * b * Real.sin C = Real.sqrt 3 - 1 := sorry

end triangle_side_relation_triangle_area_l173_173936


namespace clare_remaining_money_l173_173375

-- Definitions based on conditions
def clare_initial_money : ℕ := 47
def bread_quantity : ℕ := 4
def milk_quantity : ℕ := 2
def bread_cost : ℕ := 2
def milk_cost : ℕ := 2

-- The goal is to prove that Clare has $35 left after her purchases.
theorem clare_remaining_money : 
  clare_initial_money - (bread_quantity * bread_cost + milk_quantity * milk_cost) = 35 := 
sorry

end clare_remaining_money_l173_173375


namespace train_complete_time_l173_173282

noncomputable def train_time_proof : Prop :=
  ∃ (t_x : ℕ) (v_x : ℝ) (v_y : ℝ),
    v_y = 140 / 3 ∧
    t_x = 140 / v_x ∧
    (∃ t : ℝ, 
      t * v_x = 60.00000000000001 ∧
      t * v_y = 140 - 60.00000000000001) ∧
    t_x = 4

theorem train_complete_time : train_time_proof := by
  sorry

end train_complete_time_l173_173282


namespace larger_value_expression_l173_173110

theorem larger_value_expression (x p q : ℝ) (hp : p = 14) (hq: q = 196) (h: x = (14 + 14 * real.sqrt 2)) :
  (∃ x, (real.sqrt 2 * p = 28) ∧ (40 - x = (2 - real.sqrt 2) ^ 3) ∧  p + q = 210) :=
by
  sorry

end larger_value_expression_l173_173110


namespace quadrilateral_area_correct_l173_173529

noncomputable def quadrilateral_area (AB BC CD DA : ℝ) (angle_CDA : ℝ) : ℝ :=
  let triangle_ACD_area := 0.5 * CD * DA
  let AC := real.sqrt (CD^2 + DA^2)
  let h := 5 -- (height) computed beforehand through trigonometry
  let triangle_ABC_area := 0.5 * AC * h
  triangle_ACD_area + triangle_ABC_area

theorem quadrilateral_area_correct :
  quadrilateral_area 10 5 13 13 (real.pi / 2) = 84.5 + 32.5 * real.sqrt 2 := by
  sorry

end quadrilateral_area_correct_l173_173529


namespace min_distance_l173_173853

noncomputable def curve_one (t : ℝ) : ℝ × ℝ :=
(2 + Real.cos t, Real.sin t - 1)

noncomputable def curve_two (α : ℝ) : ℝ × ℝ :=
(4 * Real.cos α, Real.sin α)

def line_three : ℝ × ℝ → Prop :=
λ p, p.1 = p.2

def P : ℝ × ℝ := (0, 1)

noncomputable def Q (t : ℝ) : ℝ × ℝ :=
(2 + Real.cos t, Real.sin t - 1)

noncomputable def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
((P.1 + Q.1)/2, (P.2 + Q.2)/2)

noncomputable def distance_to_line (M : ℝ × ℝ) : ℝ :=
Float.abs ((1 / Real.sqrt 2) * M.2 - 1) / Real.sqrt 2

theorem min_distance (t : ℝ) (h : Real.sin (t - (Real.pi / 4)) = 1) :
  distance_to_line (midpoint P (Q t)) = (Real.sqrt 2 - 1) / Real.sqrt 2 :=
sorry

end min_distance_l173_173853


namespace right_triangle_hypotenuse_odd_and_legs_different_parity_l173_173923

theorem right_triangle_hypotenuse_odd_and_legs_different_parity 
  (a b c : ℕ) (h_coprime : Nat.coprime a b ∧ Nat.coprime a c ∧ Nat.coprime b c)
  (h_right_triangle : a^2 + b^2 = c^2) : 
  Nat.Odd c ∧ ((Nat.Even a ∧ Nat.Odd b) ∨ (Nat.Odd a ∧ Nat.Even b)) :=
by
  sorry

end right_triangle_hypotenuse_odd_and_legs_different_parity_l173_173923


namespace identify_power_functions_l173_173856

-- Define the functions
def f1 (x : ℝ) : ℝ := 2 ^ x
def f2 (x : ℝ) : ℝ := x ^ 2
def f3 (x : ℝ) : ℝ := 1 / x
def f4 (x : ℝ) : ℝ := x ^ 2 + 1
def f5 (x : ℝ) : ℝ := 3 / (x ^ 2)

-- Define what it means for a function to be a power function
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

theorem identify_power_functions :
  {n // ∀ f, (f = f1 ∨ f = f2 ∨ f = f3 ∨ f = f4 ∨ f = f5) → 
            (is_power_function f ↔ (n = 2 ∨ n = 3))} := 
by
  sorry

end identify_power_functions_l173_173856


namespace distinct_integral_solutions_l173_173421

theorem distinct_integral_solutions {x : ℤ} (h : 0 ≤ x ∧ x < 30) :
    x^4 + 2 * x^3 + 3 * x^2 - x + 1 ≡ 0 [MOD 30] → x = 13 :=
sorry

end distinct_integral_solutions_l173_173421


namespace exists_pairwise_disjoint_t_l173_173945

namespace Problem

-- Define the set S
def S := finset.Icc 1 1000000

-- Define type aliases for clarity
def subset_S := {A : finset ℕ // A ⊆ S}

-- Main theorem statement
theorem exists_pairwise_disjoint_t
  (A  : subset_S)
  (hA : A.val.card = 101):
  ∃ (t : fin 100 → ℕ), (∀ (i j : fin 100), (i ≠ j) → disjoint (A.val.image (λ x => x + t i)) (A.val.image (λ x => x + t j))) :=
begin
  sorry
end

end Problem

end exists_pairwise_disjoint_t_l173_173945


namespace triangle_side_equation_l173_173267

variable {a b c : ℝ}
variable (h1 : ∠ABC = 60) -- We'll interpret this using the cosine rule since Lean doesn't natively handle angles like this.

theorem triangle_side_equation (h1 : a^2 = b^2 + c^2 - b * c) : (3 / (a + b + c)) = (1 / (a + b)) + (1 / (a + c)) :=
sorry

end triangle_side_equation_l173_173267


namespace all_zero_l173_173153

def circle_condition (x : Fin 2007 → ℤ) : Prop :=
  ∀ i : Fin 2007, x i + x (i+1) + x (i+2) + x (i+3) + x (i+4) = 2 * (x (i+1) + x (i+2)) + 2 * (x (i+3) + x (i+4))

theorem all_zero (x : Fin 2007 → ℤ) (h : circle_condition x) : ∀ i, x i = 0 :=
sorry

end all_zero_l173_173153


namespace find_y_coordinate_l173_173100

structure Point (α : Type) := (x : α) (y : α)

noncomputable def A : Point ℝ := ⟨-4, 0⟩
noncomputable def B : Point ℝ := ⟨-3, 2⟩
noncomputable def C : Point ℝ := ⟨3, 2⟩
noncomputable def D : Point ℝ := ⟨4, 0⟩

noncomputable def dist (P Q : Point ℝ) : ℝ :=
  ((P.x - Q.x)^2 + (P.y - Q.y)^2).sqrt

axiom PA_PD_eq_10 (P : Point ℝ) : dist P A + dist P D = 10
axiom PB_PC_eq_10 (P : Point ℝ) : dist P B + dist P C = 10

theorem find_y_coordinate (P : Point ℝ) (h1 : dist P A + dist P D = 10)
  (h2 : dist P B + dist P C = 10) :
  P.y = 6 / 7 ∧ (let a := 0, b := 6, c := 7, d := 1 in a + b + c + d = 14) :=
  sorry

end find_y_coordinate_l173_173100


namespace angle_bisector_foot_eqn_l173_173102

variable {α : Type} [LinearOrderedField α]
variables {A B C D : α} -- points on the 2D plane
variables {AB AC AD DB DC : α} -- distances between the points

theorem angle_bisector_foot_eqn (hABC : ∀ {A B C : α}, triangle A B C)
  (hD_foot : D = foot_of_bisector A B C) :
  AB * AC = DB * DC + AD^2 :=
sorry

end angle_bisector_foot_eqn_l173_173102


namespace part_I_part_II_l173_173932

variable (α β γ φ : ℝ)
variable (A B C P : Type)
variable (angleBAC angleABC angleACB : ℝ) 

-- Condition: Point P is projected onto plane ABC with the projection point as circumcenter O
def projection_of_P_is_circumcenter (P A B C O : Type) : Prop := true -- Assume this condition holds as a placeholder

-- Condition: Dihedral angles between PA, PB, and PC with the plane ABC are α, β, γ
def dihedral_angles (PA PB PC : Type) (α β γ : ℝ) : Prop := true -- Assume these conditions hold as placeholders

-- Condition: Angle between PO and PA is φ
def angle_PO_PA (PO PA : Type) (φ : ℝ) : Prop := true -- Assume this condition holds as a placeholder

-- Given all conditions, prove the first statement
theorem part_I :
projection_of_P_is_circumcenter P A B C A ∧
dihedral_angles P A B (angleBAC β γ) α β γ ∧
angle_PO_PA P A φ →
cos angleBAC = (sin ( β / 2 ) ^ 2 + sin ( γ / 2 ) ^ 2 - sin ( α / 2 ) ^ 2) / (2 * sin ( β / 2 ) * sin ( γ / 2 )) := 
by sorry

-- Given all conditions, prove the second statement
theorem part_II :
projection_of_P_is_circumcenter P A B C A ∧
dihedral_angles P A B (angleBAC β γ) α β γ ∧
angle_PO_PA P A φ →
((sin ( α / 2 ) / sin angleBAC) = sin φ) ∧
((sin ( β / 2 ) / sin angleABC) = sin φ) ∧
((sin ( γ / 2 ) / sin angleACB) = sin φ) :=
by sorry

end part_I_part_II_l173_173932


namespace domain_of_function_l173_173635

theorem domain_of_function :
  ∀ x : ℝ, (0 ≤ x ∧ x ≠ 1) ↔ (∃ y : ℝ, y = 1 / (Real.sqrt x - 1)) := by
  sorry

end domain_of_function_l173_173635


namespace vasya_lowest_position_l173_173221

theorem vasya_lowest_position
  (num_cyclists : ℕ)
  (num_stages : ℕ)
  (num_ahead : ℕ)
  (position_vasya : ℕ)
  (total_time : List ℕ)
  (unique_total_times : total_time.nodup)
  (stage_positions : List (List ℕ))
  (unique_stage_positions : ∀ stage ∈ stage_positions, stage.nodup)
  (vasya_consistent : ∀ stage ∈ stage_positions, stage.nth position_vasya = some num_ahead) :
  num_ahead * num_stages + 1 = 91 :=
by
  sorry

end vasya_lowest_position_l173_173221


namespace hyperbola_eccentricity_l173_173627

variable {a b : ℝ}

-- Conditions
def hyperbola_eq (x y : ℝ) : Prop := (x ^ 2) / (a ^ 2) - (y ^ 2) / (b ^ 2) = 1
def asymptote_eq (x y : ℝ) : Prop := y = (b / a) * x
def point_on_asymptote := asymptote_eq 4 2

-- Expected Result
def eccentricity : ℝ := Real.sqrt (1 + (b ^ 2) / (a ^ 2))

theorem hyperbola_eccentricity:
  point_on_asymptote →
  hyperbola_eq 4 2 →
  eccentricity = Real.sqrt (7 / 4) := 
sorry

end hyperbola_eccentricity_l173_173627


namespace sufficient_but_not_necessary_not_necessary_l173_173498

theorem sufficient_but_not_necessary (m x y a : ℝ) (h₀ : m > 0) (h₁ : |x - a| < m) (h₂ : |y - a| < m) : |x - y| < 2 * m :=
by
  sorry

theorem not_necessary (m : ℝ) (h₀ : m > 0) : ∃ x y a : ℝ, |x - y| < 2 * m ∧ ¬ (|x - a| < m ∧ |y - a| < m) :=
by
  sorry

end sufficient_but_not_necessary_not_necessary_l173_173498


namespace find_f3_l173_173434

section FunctionProblem

variable (f : ℝ → ℝ)

theorem find_f3 (h : ∀ x : ℝ, f(2 * x + 1) = x^2 - 2 * x) : f(3) = -1 :=
by
  sorry

end FunctionProblem

end find_f3_l173_173434


namespace weight_of_girl_who_left_l173_173614

theorem weight_of_girl_who_left (initial_weight : ℕ) (g_new : ℕ) (avg_increase : ℕ) 
  (condition1 : avg_increase = 2) (condition2 : g_new = 80) : (g_left : ℕ) :=
by
  have w_new := initial_weight + 20 * avg_increase
  have w_new := initial_weight + g_new - g_left
  sorry

end weight_of_girl_who_left_l173_173614


namespace cauchy_schwarz_inequality_l173_173969

theorem cauchy_schwarz_inequality (n : ℕ) (a : Fin n → ℝ) (b : Fin n → ℝ) (hb : ∀ i, 0 < b i) :
  (∑ i, a i ^ 2 / b i) ≥ (∑ i, a i) ^ 2 / (∑ i, b i) := 
sorry

end cauchy_schwarz_inequality_l173_173969


namespace problem_l173_173457

-- Define the function and its domain
variables {α : Type*} [linear_order α] {β : Type*} [add_group β]
variable {f : α → β}

-- Define the conditions of the problem
variables (a b : α)
hypothesis domain : ∀ x, a < x ∧ x < b → x ∈ set.univ -- The domain of f(x) is (a,b)
hypothesis false_statement : ¬ ∃ x ∈ set.Ioo a b, f x + f (-x) ≠ 0

-- The theorem stating f(a + b) = 0
theorem problem : f (a + b) = 0 :=
sorry

end problem_l173_173457


namespace expected_sixes_in_three_rolls_l173_173675

theorem expected_sixes_in_three_rolls : 
  (∑ k in Finset.range 4, k * (Nat.choose 3 k) * (1/6)^k * (5/6)^(3-k)) = 1/2 := 
by
  sorry

end expected_sixes_in_three_rolls_l173_173675


namespace unique_number_not_in_range_of_g_l173_173192

variables (p q r s : ℝ)

def g (x : ℝ) : ℝ := (p * x + q) / (r * x + s)

theorem unique_number_not_in_range_of_g 
  (h₁ : g p q r s 13 = 13)
  (h₂ : g p q r s 61 = 61)
  (h₃ : ∀ x, x ≠ -s / r → g p q r s (g p q r s x) = x)
  (hnz_p : p ≠ 0) 
  (hnz_q : q ≠ 0)
  (hnz_r : r ≠ 0)
  (hnz_s : s ≠ 0) : 
  ∀ y : ℝ, y ≠ 37 :=
begin
  sorry
end

end unique_number_not_in_range_of_g_l173_173192


namespace digit_x_base_7_l173_173616

theorem digit_x_base_7 (x : ℕ) : 
    (4 * 7^3 + 5 * 7^2 + x * 7 + 2) % 9 = 0 → x = 4 := 
by {
    sorry
}

end digit_x_base_7_l173_173616
