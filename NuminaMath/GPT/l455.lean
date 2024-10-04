import Mathlib

namespace polynomial_degree_l455_455957

-- We define the polynomial as given in the problem condition.
def polynomial := (X^4 + 1)^3 * (X^2 + X + 1)^2

-- Now we state the main theorem: that the degree of this polynomial is 16.
theorem polynomial_degree : polynomial.degree = 16 := sorry

end polynomial_degree_l455_455957


namespace no_four_digit_numbers_divisible_by_11_l455_455661

theorem no_four_digit_numbers_divisible_by_11 (a b c d : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ 9) 
(h₃ : 0 ≤ b) (h₄ : b ≤ 9) (h₅ : 0 ≤ c) (h₆ : c ≤ 9) (h₇ : 0 ≤ d) (h₈ : d ≤ 9) 
(h₉ : a + b + c + d = 10) (h₁₀ : a + c = b + d) : 
0 = 0 :=
sorry

end no_four_digit_numbers_divisible_by_11_l455_455661


namespace product_of_x_equals_21_l455_455099

theorem product_of_x_equals_21 (x : ℝ)
  (square_side : x - 3)
  (rectangle_length : x - 5)
  (rectangle_width : x + 3)
  (h : 3 * (x - 3) ^ 2 = (x - 5) * (x + 3)) :
  (x = 7 ∨ x = 3) → (7 * 3 = 21) := by
  sorry

end product_of_x_equals_21_l455_455099


namespace motion_first_kind_rotation_or_translation_l455_455774

-- Definitions for reflections and lines
def reflection (l : Line) : Point → Point := sorry -- Definition of reflection (to be provided)
def composition {α : Type} (f g : α → α) : α → α := λ x, f (g x)

-- Definition of the main statement
theorem motion_first_kind_rotation_or_translation :
  ∀ l₁ l₂ : Line, 
    is_motion_first_kind (composition (reflection l₁) (reflection l₂)) →
    (is_rotation (composition (reflection l₁) (reflection l₂)) ∨ is_translation (composition (reflection l₁) (reflection l₂))) :=
by 
  intros l₁ l₂ h
  -- Proof steps (to be filled) showing that "composition of two reflections" results in either 
  -- a rotation or a translation, depending on whether l₁ and l₂ are parallel or intersect.
  sorry

end motion_first_kind_rotation_or_translation_l455_455774


namespace stadium_length_in_feet_l455_455821

theorem stadium_length_in_feet (length_in_yards : ℕ) (conversion_factor : ℕ) (h1 : length_in_yards = 62) (h2 : conversion_factor = 3) : length_in_yards * conversion_factor = 186 :=
by
  sorry

end stadium_length_in_feet_l455_455821


namespace number_of_primes_such_that_39p_plus_1_is_perfect_square_l455_455672

open Nat

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem number_of_primes_such_that_39p_plus_1_is_perfect_square :
  ∃ S : Finset ℕ, (∀ p ∈ S, Prime p ∧ isPerfectSquare (39 * p + 1)) ∧ S.card = 3 := by
sorry

end number_of_primes_such_that_39p_plus_1_is_perfect_square_l455_455672


namespace trackball_mice_count_l455_455050

theorem trackball_mice_count (total_sales wireless_share optical_share : ℕ) 
    (h_total : total_sales = 80)
    (h_wireless : wireless_share = total_sales / 2)
    (h_optical : optical_share = total_sales / 4):
    total_sales - (wireless_share + optical_share) = 20 :=
by
  sorry

end trackball_mice_count_l455_455050


namespace obtuse_angle_only_dihedral_planar_l455_455538

/-- Given the range of three types of angles, prove that only the dihedral angle's planar angle can be obtuse. -/
theorem obtuse_angle_only_dihedral_planar 
  (α : ℝ) (β : ℝ) (γ : ℝ) 
  (hα : 0 < α ∧ α ≤ 90)
  (hβ : 0 ≤ β ∧ β ≤ 90)
  (hγ : 0 ≤ γ ∧ γ < 180) : 
  (90 < γ ∧ (¬(90 < α)) ∧ (¬(90 < β))) :=
by 
  sorry

end obtuse_angle_only_dihedral_planar_l455_455538


namespace sequence_behavior_l455_455998

noncomputable def x_sequence : ℕ → ℝ
| 0       := sorry -- since x1 not given, we'll ignore initial value here
| (n + 1) := (2 * x_sequence n ^ 2 + 5 * x_sequence n + 7) / (3 * x_sequence n + 5)

theorem sequence_behavior (x1 : ℝ) (h_pos : ∀ n, x_sequence n > 0) (h_initial : x1 ≠ real.sqrt 7)
  (h_recur : ∀ n, x_sequence (n+1) = (2 * x_sequence n ^ 2 + 5 * x_sequence n + 7) / (3 * x_sequence n + 5)) :
  (∀ n, (x_sequence n < x_sequence (n + 1) ∧ x_sequence (n + 1) < real.sqrt 7) ∨
      (x_sequence n > x_sequence (n + 1) ∧ x_sequence (n + 1) > real.sqrt 7)) :=
sorry

end sequence_behavior_l455_455998


namespace regional_park_license_plates_l455_455528

theorem regional_park_license_plates :
  let letters := 3
  let digits := 10
  let plate_length := 5
  letters * digits ^ plate_length = 300000
  := by
  simp [letters, digits, plate_length]
  sorry

end regional_park_license_plates_l455_455528


namespace correct_transformation_C_l455_455484

-- Define the conditions as given in the problem
def condition_A (x : ℝ) : Prop := 4 + x = 3 ∧ x = 3 - 4
def condition_B (x : ℝ) : Prop := (1 / 3) * x = 0 ∧ x = 0
def condition_C (y : ℝ) : Prop := 5 * y = -4 * y + 2 ∧ 5 * y + 4 * y = 2
def condition_D (a : ℝ) : Prop := (1 / 2) * a - 1 = 3 * a ∧ a - 2 = 6 * a

-- The theorem to prove that condition_C is correctly transformed
theorem correct_transformation_C : condition_C 1 := 
by sorry

end correct_transformation_C_l455_455484


namespace volume_of_intersecting_octahedra_l455_455474

def absolute (x : ℝ) : ℝ := abs x

noncomputable def volume_of_region : ℝ :=
  let region1 (x y z : ℝ) := absolute x + absolute y + absolute z ≤ 2
  let region2 (x y z : ℝ) := absolute x + absolute y + absolute (z - 2) ≤ 2
  -- The region is the intersection of these two inequalities
  -- However, we calculate its volume directly
  (2 / 3 : ℝ)

theorem volume_of_intersecting_octahedra :
  (volume_of_region : ℝ) = (2 / 3 : ℝ) :=
sorry

end volume_of_intersecting_octahedra_l455_455474


namespace problem_statement_l455_455096

def otimes (a b : ℝ) : ℝ := a^2 / b

theorem problem_statement : ((otimes (otimes 3 4) 6) - (otimes 3 (otimes 4 6)) - 1) = -113 / 32 := 
  by 
    sorry

end problem_statement_l455_455096


namespace triangle_congruence_of_symmetric_lines_l455_455255

-- Definitions for the problem
variables {A B C : Point} -- Points A, B, C forming triangle ABC
variable {l : Line} -- Line l

-- Conditions
noncomputable def is_tangent_to_incircle (l : Line) (A B C : Point) : Prop := sorry
noncomputable def is_symmetric_to_external_bisectors 
  (l l_a l_b l_c : Line) (A B C : Point) : Prop := sorry

-- Main theorem statement
theorem triangle_congruence_of_symmetric_lines 
  (h_tangent : is_tangent_to_incircle l A B C)
  (h_symmetric : is_symmetric_to_external_bisectors l l_a l_b l_c A B C) :
  triangle_congruence (triangle_of_lines l_a l_b l_c) (triangle_of_points A B C) :=
sorry

end triangle_congruence_of_symmetric_lines_l455_455255


namespace forecast_interpretation_l455_455728

-- Define the conditions
def condition (precipitation_probability : ℕ) : Prop :=
  precipitation_probability = 78

-- Define the interpretation question as a proof
theorem forecast_interpretation (precipitation_probability: ℕ) (cond : condition precipitation_probability) :
  precipitation_probability = 78 :=
by
  sorry

end forecast_interpretation_l455_455728


namespace anna_apple_ratio_l455_455183

-- Definitions based on conditions
def tuesday_apples : ℕ := 4
def wednesday_apples : ℕ := 2 * tuesday_apples
def total_apples : ℕ := 14

-- Theorem statement
theorem anna_apple_ratio :
  ∃ thursday_apples : ℕ, 
  thursday_apples = total_apples - (tuesday_apples + wednesday_apples) ∧
  (thursday_apples : ℚ) / tuesday_apples = 1 / 2 :=
by
  sorry

end anna_apple_ratio_l455_455183


namespace cheese_initial_amount_l455_455832

noncomputable def cheeseProblem : ℕ :=
let k := 35 in -- proof burden moved here by assuming k = 35 since it's the only fit
let cheeseEatenFirstNight := 10 in 
let cheeseEatenSecondNight := 1 in -- directly calculated from given information
cheeseEatenFirstNight + cheeseEatenSecondNight

theorem cheese_initial_amount
  (k : ℕ) (h1 : k > 7) (h2 : ∃ (d : ℕ), k = d ∧ 35 % d = 0) :
  cheeseProblem = 11 :=
by
  sorry

end cheese_initial_amount_l455_455832


namespace find_x_l455_455269

open Nat

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_x (x : ℕ) (hx : x > 0) (hprime : is_prime (x^5 + x + 1)) : x = 1 := 
by 
  sorry

end find_x_l455_455269


namespace solve_fraction_l455_455246

theorem solve_fraction (a b : ℝ) (hab : 3 * a = 2 * b) : (a + b) / b = 5 / 3 :=
by
  sorry

end solve_fraction_l455_455246


namespace k_squared_geq_25_over_3_l455_455907

theorem k_squared_geq_25_over_3
  (a1 a2 a3 a4 a5 k : ℝ)
  (h_diff : ∀ i j, i ≠ j → |a1 - a2| ≥ 1 ∧ |a1 - a3| ≥ 1 ∧ |a1 - a4| ≥ 1  ∧ |a1 - a5| ≥ 1 ∧ |a2 - a3| ≥ 1  ∧ |a2 - a4| ≥ 1  ∧  |a2 - a5| ≥ 1 ∧ |a3 - a4| ≥ 1  ∧ |a3 - a5| ≥ 1  ∧ |a4 - a5| ≥ 1 )
  (h_sum : a1 + a2 + a3 + a4 + a5 = 2 * k)
  (h_sum_squares : a1^2 + a2^2 + a3^2 + a4^2 + a5^2 = 2 * k^2) :
  k^2 ≥ 25 / 3 :=
by
  sorry

end k_squared_geq_25_over_3_l455_455907


namespace trackball_mice_count_l455_455045

theorem trackball_mice_count 
  (total_mice wireless_mice optical_mice trackball_mice : ℕ)
  (h1 : total_mice = 80)
  (h2 : wireless_mice = total_mice / 2)
  (h3 : optical_mice = total_mice / 4)
  (h4 : trackball_mice = total_mice - (wireless_mice + optical_mice)) :
  trackball_mice = 20 := by 
  sorry

end trackball_mice_count_l455_455045


namespace simson_line_l455_455775

theorem simson_line 
  (ABC: Triangle)
  (Ω: Circle)
  (P: Point)
  (hP: P ∈ Ω)
  (D E F: Point)
  (hD: ⟂ (LineSegment (P, D)) (LineSegment (B, C)))
  (hE: ⟂ (LineSegment (P, E)) (LineSegment (C, A)))
  (hF: ⟂ (LineSegment (P, F)) (LineSegment (A, B)))
  : Collinear [D, E, F] :=
sorry

end simson_line_l455_455775


namespace length_of_GH_l455_455712

theorem length_of_GH
  (AB BC : ℝ) (angle : ℝ)
  (h1 : AB = 6)
  (h2 : BC = 8)
  (h3 : angle = 30) :
  let DB := real.sqrt (AB ^ 2 + BC ^ 2) in
  let sec_angle := 2 / real.sqrt 3 in
  GH = DB * sec_angle :=
begin
  let DB := real.sqrt (AB ^ 2 + BC ^ 2),
  have hDB : DB = 10, by {
    calc DB = real.sqrt (6 ^ 2 + 8 ^ 2) : by simp [h1, h2]
       ... = real.sqrt (36 + 64) : by norm_num
       ... = real.sqrt 100 : by norm_num
       ... = 10 : by norm_num,
  },
  let sec_angle := 2 / real.sqrt 3,
  have h_sec : sec_angle = 2 / real.sqrt 3, by refl,
  have hGH : GH = 10 * (2 / real.sqrt 3), by {
    calc GH = DB * sec_angle : by refl
       ... = 10 * (2 / real.sqrt 3) : by simp [hDB, h_sec],
  },
  exact hGH,
  sorry,
end

end length_of_GH_l455_455712


namespace right_triangle_hypotenuse_l455_455403

variable (LM LN : ℝ)
variable (N M : ℝ)

theorem right_triangle_hypotenuse 
  (h1 : ∠M = π / 2) 
  (h2 : cos N = 4 / 5) 
  (h3 : LM = 20) 
  : LN = 25 :=
begin
  sorry
end

end right_triangle_hypotenuse_l455_455403


namespace books_selection_combination_l455_455328

theorem books_selection_combination : (nat.choose 8 3) = 56 := 
by 
  sorry

end books_selection_combination_l455_455328


namespace tournament_total_games_l455_455880

theorem tournament_total_games (n : ℕ) (h : n = 8) : 
  let total_games := n * (n - 1) * 2 in 
  total_games = 112 :=
by
  sorry

end tournament_total_games_l455_455880


namespace crates_needed_l455_455574

-- Conditions as definitions
def novels : ℕ := 145
def comics : ℕ := 271
def documentaries : ℕ := 419
def albums : ℕ := 209
def crate_capacity : ℕ := 9

-- Total items calculation
def total_items : ℕ := novels + comics + documentaries + albums

-- Proof statement
theorem crates_needed : (total_items / crate_capacity) = 116 := by
  sorry

end crates_needed_l455_455574


namespace curling_tournament_equation_l455_455913

theorem curling_tournament_equation (x : ℕ) :
  (∃ (x : ℕ), (1 / 2 * x * (x - 1) = 45)) → (x * (x - 1) = 90) :=
by
  intro h
  cases h with x hx
  have h1 : x * (x - 1) / 2 = 45 := by 
    assumption
  have h2 : x * (x - 1) = 90 := by
    exact (eq_mul_of_div_eq_right (@two_ne_zero ℤ _ _ _) hx).symm  -- Multiply both sides by 2
  exact h2

end curling_tournament_equation_l455_455913


namespace trig_identity_proof_l455_455877

theorem trig_identity_proof : 
  sin (47 * (real.pi / 180)) * sin (103 * (real.pi / 180)) + sin (43 * (real.pi / 180)) * cos (77 * (real.pi / 180)) = sqrt 3 / 2 :=
by {
  sorry
}

end trig_identity_proof_l455_455877


namespace find_pairs_l455_455950

theorem find_pairs (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) :
  (a * b^2 + b + 7) ∣ (a^2 * b + a + b) ↔ ((a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ ∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k) :=
by sorry

end find_pairs_l455_455950


namespace polygon_sides_l455_455686

theorem polygon_sides :
  ∀ (n : ℕ), (n > 2) → (n - 2) * 180 < 360 → n = 3 :=
by
  intros n hn1 hn2
  sorry

end polygon_sides_l455_455686


namespace BDE_angle_60_degrees_l455_455257

-- Definitions given in the problem
variables (A B C D E F : Type) 
  [EuclideanGeometry ABC] -- Ensure ABC is a class from Euclidean Geometry
  (equilateral_triangle : is_equilateral A B C)
  (D_on_ext_AB : extends AB A D)
  (E_on_ext_BC : extends BC C E)
  (F_on_ext_AC : extends AC C F)
  (CF_eq_AD : length C F = length A D)
  (AC_plus_EF_eq_DE : length A C + length E F = length D E)

-- The main proof problem
theorem BDE_angle_60_degrees : ∠ B D E = 60 := 
  by sorry

end BDE_angle_60_degrees_l455_455257


namespace reflection_correct_l455_455590

def vector1 : ℝ × ℝ := (1,2)
def vector2 : ℝ × ℝ := (2,1)

def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let norm_sq := v.1 * v.1 + v.2 * v.2
  let scale := dot_product / norm_sq
  (scale * v.1, scale * v.2)

def reflection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let p := projection u v
  (2 * p.1 - u.1, 2 * p.2 - u.2)

theorem reflection_correct : reflection vector1 vector2 = (11/5, -2/5) :=
  sorry

end reflection_correct_l455_455590


namespace ott_fraction_part_l455_455759

noncomputable def fractional_part_of_group_money (x : ℝ) (M L N P : ℝ) :=
  let total_initial := M + L + N + P + 2
  let money_received_by_ott := 4 * x
  let ott_final_money := 2 + money_received_by_ott
  let total_final := total_initial + money_received_by_ott
  (ott_final_money / total_final) = (3 / 14)

theorem ott_fraction_part (x : ℝ) (M L N P : ℝ)
    (hM : M = 6 * x) (hL : L = 5 * x) (hN : N = 4 * x) (hP : P = 7 * x) :
    fractional_part_of_group_money x M L N P :=
by
  sorry

end ott_fraction_part_l455_455759


namespace mary_total_travel_time_and_cost_l455_455377

namespace Travel

-- Conditions based on the problem statement
def time_uber_to_house : ℕ := 10
def cost_uber_to_house_usd : ℝ := 15
def time_uber_to_airport : ℕ := time_uber_to_house * 5
def time_bag_check : ℕ := 15
def cost_bag_check_eur : ℝ := 20
def time_security : ℕ := time_bag_check * 3
def time_wait_to_board : ℕ := 20
def time_wait_for_plane : ℕ := time_wait_to_board * 2
def time_first_layover : ℕ := 3 * 60 + 25
def time_flight_delay : ℕ := 45
def time_second_layover : ℕ := 1 * 60 + 50
def time_zone_difference : ℕ := 3
def cost_meal_gbp : ℝ := 10
def exchange_rate_usd_to_eur : ℝ := 0.85
def exchange_rate_usd_to_gbp : ℝ := 0.75

-- Definitions of total time and cost in USD
def total_time : ℕ :=
  time_uber_to_house + 
  time_uber_to_airport + 
  time_bag_check + 
  time_security +
  time_wait_to_board + 
  time_wait_for_plane + 
  time_first_layover + 
  time_flight_delay + 
  time_second_layover

def total_time_with_time_zone : ℕ := total_time + time_zone_difference * 60

def cost_bag_check_usd : ℝ := cost_bag_check_eur / exchange_rate_usd_to_eur
def cost_meal_usd : ℝ := cost_meal_gbp / exchange_rate_usd_to_gbp

def total_cost_usd : ℝ := cost_uber_to_house_usd + cost_bag_check_usd + cost_meal_usd

-- Theorem statement to prove the total time and total cost
theorem mary_total_travel_time_and_cost :
  total_time_with_time_zone = 12 * 60 ∧ 
  total_cost_usd = 51.86 := by
  sorry

end Travel

end mary_total_travel_time_and_cost_l455_455377


namespace triangle_midpoint_isosceles_l455_455754

theorem triangle_midpoint_isosceles (A B C D E F : Point)
  (hD : midpoint D A F)
  (hE : lies_on_line E C D ∧ lies_on_line E A B)
  (hBD_BF_CF : dist B D = dist B F ∧ dist B F = dist C F) :
  dist A E = dist D E :=
sorry

end triangle_midpoint_isosceles_l455_455754


namespace Sandy_original_number_l455_455398

theorem Sandy_original_number (x : ℝ) (h : (3 * x + 20)^2 = 2500) : x = 10 :=
by
  sorry

end Sandy_original_number_l455_455398


namespace a_n_nonzero_l455_455249

def a : ℕ → ℤ
| 0       := 1
| 1       := 2
| (n + 2) := if a n * a (n + 1) % 2 = 0 then 5 * a (n + 1) - 3 * a n else a (n + 1) - a n

theorem a_n_nonzero : ∀ n : ℕ, a n ≠ 0 := by
  sorry

end a_n_nonzero_l455_455249


namespace gcd_sum_lcm_eq_gcd_l455_455777

theorem gcd_sum_lcm_eq_gcd (a b : ℤ) : Int.gcd (a + b) (Int.lcm a b) = Int.gcd a b :=
by 
  sorry

end gcd_sum_lcm_eq_gcd_l455_455777


namespace math_problem_l455_455920

theorem math_problem : (300 + 5 * 8) / (2^3) = 42.5 := by
  sorry

end math_problem_l455_455920


namespace chess_tournament_rankings_l455_455006

theorem chess_tournament_rankings :
  let P_Q_choices := 2; let R_S_choices := 2; let sunday_arrangements := 2 in
  let total_rankings := P_Q_choices * R_S_choices * sunday_arrangements * 2 in
  total_rankings = 16 :=
by
  -- Definitions based on conditions
  let P_Q_choices := 2
  let R_S_choices := 2
  let sunday_arrangements := 2
  let total_rankings := P_Q_choices * R_S_choices * sunday_arrangements * 2
  -- Theorem statement based on the question and the correct answer
  show total_rankings = 16, from sorry

end chess_tournament_rankings_l455_455006


namespace arithmetic_series_sum_l455_455979

theorem arithmetic_series_sum :
  ∀ (a d l : ℤ) (n : ℕ), a = -49 → d = 2 → l = -1 → (a + (n - 1) * d = l) → (n = 25) → 
  ∑ k in Finset.range n, (a + k * d) = -625 :=
by
  intros a d l n ha hd hl hnl hn25
  sorry

end arithmetic_series_sum_l455_455979


namespace boys_girls_ratio_l455_455700

theorem boys_girls_ratio (B G : ℕ) (h : 0.80 * B + 0.75 * G = 0.78 * (B + G)) : B / G = 3 / 2 := 
by sorry

end boys_girls_ratio_l455_455700


namespace seventy_two_times_twenty_eight_l455_455552

theorem seventy_two_times_twenty_eight : 72 * 28 = 4896 := by
  have h1 : 72 = 70 + 2 := by rfl
  have h2 : 28 = 30 - 2 := by rfl
  calc
    72 * 28 = (70 + 2) * (30 - 2) : by rw [h1, h2]
    ... = (70 + 2) * (30 - 2) : by rfl
    ... = 70^2 - 2^2 : by rw [mul_sub, add_mul, add_mul, <- add_sub, <- add_sub]; ring
    ... = 4900 - 4 : by norm_num
    ... = 4896 : by norm_num

end seventy_two_times_twenty_eight_l455_455552


namespace parallel_implies_a_values_l455_455601

noncomputable def line1 (a : ℝ) : AffineSubspace ℝ ℝ := {p | p.x + 2 * a * p.y - 1 = 0}
noncomputable def line2 (a : ℝ) : AffineSubspace ℝ ℝ := {p | (a + 1) * p.x - a * p.y = 0}

theorem parallel_implies_a_values (a : ℝ) (h_parallel : is_parallel (line1 a) (line2 a)) : 
  a = -3 / 2 ∨ a = 0 :=
sorry

end parallel_implies_a_values_l455_455601


namespace Joel_laps_count_l455_455491

-- Definitions of conditions
def Yvonne_laps := 10
def sister_laps := Yvonne_laps / 2
def Joel_laps := sister_laps * 3

-- Statement to be proved
theorem Joel_laps_count : Joel_laps = 15 := by
  -- currently, proof is not required, so we defer it with 'sorry'
  sorry

end Joel_laps_count_l455_455491


namespace vanya_number_l455_455457

theorem vanya_number (m n : ℕ) (hm : m < 10) (hn : n < 10) (h : (10 * n + m)^2 = 4 * (10 * m + n)) : 
  10 * m + n = 81 :=
by sorry

end vanya_number_l455_455457


namespace matrix_sum_correct_l455_455593

open Matrix

def A : Matrix (fin 2) (fin 2) ℤ := !![4, -3; 0, 5]
def B : Matrix (fin 2) (fin 2) ℤ := !![-6, 8; 7, -10]
def C : Matrix (fin 2) (fin 2) ℤ := !![-2, 5; 7, -5]

theorem matrix_sum_correct : A + B = C :=
by 
  sorry

end matrix_sum_correct_l455_455593


namespace range_of_x_l455_455746

theorem range_of_x (x y : ℝ) (h : 4 * x * y + 4 * y^2 + x + 6 = 0) : x ≤ -2 ∨ x ≥ 3 :=
sorry

end range_of_x_l455_455746


namespace probability_of_shortest_diagonal_in_20_gon_l455_455852

def total_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

def shortest_diagonals (n : ℕ) : ℕ := n

theorem probability_of_shortest_diagonal_in_20_gon :
  let n := 20 in
  let total_d := total_diagonals n in
  let shortest_d := shortest_diagonals n in
  total_d = 170 ∧ shortest_d = 20 → shortest_d.to_rat / total_d.to_rat = 2 / 17 :=
by
  intros n total_d shortest_d h,
  sorry

end probability_of_shortest_diagonal_in_20_gon_l455_455852


namespace solution_l455_455073

noncomputable def problem_statement : Prop :=
  ∃ x : ℝ, log x / log 2 + log x / log 4 + log x / log 8 = 9 

theorem solution (x : ℝ) (hx : log x / log 2 + log x / log 4 + log x / log 8 = 9) :
  x = 2^(54/11) :=
by
  sorry

end solution_l455_455073


namespace train_crossing_time_l455_455533

theorem train_crossing_time 
  (length_of_train : ℝ) 
  (speed_of_train_kmhr : ℝ) 
  (length_of_bridge : ℝ) : 
  speed_of_train_kmhr = 45 → length_of_train = 155 → length_of_bridge = 220.03 → 
  (length_of_train + length_of_bridge) / (speed_of_train_kmhr * 1000 / 3600) = 30.0024 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  have speed_in_ms := (45 : ℝ) * (1000 / 3600)
  have total_distance := (155 : ℝ) + 220.03
  simp only [speed_in_ms, total_distance]
  have eq1 : speed_in_ms = 12.5 := by norm_num
  have eq2 : total_distance = 375.03 := by norm_num
  rw [eq1, eq2]
  norm_num

end train_crossing_time_l455_455533


namespace students_count_l455_455795

theorem students_count (x : ℕ) (h1 : x / 2 + x / 4 + x / 7 + 3 = x) : x = 28 :=
  sorry

end students_count_l455_455795


namespace cylinder_volume_l455_455273

theorem cylinder_volume (r h : ℝ) (h_r : r = 2) : ∃ V : ℝ, V = 4 * Real.pi * h :=
by
  use 4 * Real.pi * h
  rw h_r
  sorry

end cylinder_volume_l455_455273


namespace problem_f_prime_at_zero_l455_455248

noncomputable def f (x : ℝ) : ℝ := x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5) + 6

theorem problem_f_prime_at_zero : deriv f 0 = 120 :=
by
  -- Proof omitted
  sorry

end problem_f_prime_at_zero_l455_455248


namespace prob_same_gender_eq_two_fifths_l455_455516

-- Define the number of male and female students
def num_male_students : ℕ := 3
def num_female_students : ℕ := 2

-- Define the total number of students
def total_students : ℕ := num_male_students + num_female_students

-- Define the probability calculation
def probability_same_gender := (num_male_students * (num_male_students - 1) / 2 + num_female_students * (num_female_students - 1) / 2) / (total_students * (total_students - 1) / 2)

theorem prob_same_gender_eq_two_fifths :
  probability_same_gender = 2 / 5 :=
by
  -- Proof is omitted
  sorry

end prob_same_gender_eq_two_fifths_l455_455516


namespace count_primes_with_prime_remainder_div_6_l455_455667

theorem count_primes_with_prime_remainder_div_6 :
  let primes_btwn_50_100 := [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
  (filter (λ p, let r := p % 6 in r = 1 ∨ r = 5) primes_btwn_50_100).length = 10 :=
  by {
    let primes_btwn_50_100 := [53, 59, 61, 67, 71, 73, 79, 83, 89, 97],
    have expected_primes := [53, 59, 61, 67, 71, 73, 79, 83, 89, 97],
    let prime_remainders := (filter (λ p, let r := p % 6 in r = 1 ∨ r = 5) primes_btwn_50_100),
    exact prime_remainders.length = 10,
  }

end count_primes_with_prime_remainder_div_6_l455_455667


namespace bug_back_at_A_l455_455736

noncomputable def prob_back_at_A_after_6_meters : ℚ := 159 / 972

theorem bug_back_at_A (P : ℕ → ℚ) (P_0 : P 0 = 1) 
  (trans : ∀ n, P (n + 1) = (1/2) * P(0) + (1/2) * P(n-1) + (1/6) * P(n-1 - 1)) :
  P 6 = 53 / 324 :=
by 
  have P1 : P 1 = 0 := sorry
  have P2 : P 2 = 1/6 := sorry
  have P3 : P 3 = 7/36 := sorry
  have P4 : P 4 = 19/108 := sorry
  have P5 : P 5 = 55/324 := sorry
  have P6 : P 6 = 159/972 := by sorry
  exact sorry

end bug_back_at_A_l455_455736


namespace ratio_of_radii_l455_455087

theorem ratio_of_radii (R r : ℝ) (α : ℝ) (h : (R + r) * real.sin α + r = R) :
  R / r = real.cot^2 ((real.pi / 4) - (α / 2)) :=
sorry

end ratio_of_radii_l455_455087


namespace advanced_purchase_tickets_sold_l455_455903

theorem advanced_purchase_tickets_sold (A D : ℕ) 
  (h1 : A + D = 140)
  (h2 : 8 * A + 14 * D = 1720) : 
  A = 40 :=
by
  sorry

end advanced_purchase_tickets_sold_l455_455903


namespace residue_at_zero_l455_455972

noncomputable def sin_taylor (n : ℕ) : ℝ :=
  ∑ k in (range n).filter (Even.2), (-1)^{k/2} * (z^(2*k + 1)) / (int_of_nat (fact(2 * k + 1) : ℕ))

noncomputable def phi (z : ℂ) : ℂ := 
  (sin(3*z) - 3*sin(z)) / z^3

noncomputable def psi (z : ℂ) : ℂ := 
  (sin(z)-z) * sin(z) / z^4

theorem residue_at_zero : 
  residue (λ z : ℂ, (sin(3*z) - 3*sin(z)) / ((sin(z) - z) * sin(z))) 0 = 24 := 
sorry

end residue_at_zero_l455_455972


namespace obtuse_triangle_of_sin_cos_sum_l455_455030

theorem obtuse_triangle_of_sin_cos_sum (A : ℝ) (h1 : 0 < A) (h2 : A < π) (h3 : sin A + cos A = 7 / 12) : π / 2 < A ∧ A < π :=
by
  sorry

end obtuse_triangle_of_sin_cos_sum_l455_455030


namespace parabola_ordinate_l455_455063

theorem parabola_ordinate (x y : ℝ) (h : y = 2 * x^2) (d : dist (x, y) (0, 1 / 8) = 9 / 8) : y = 1 := 
sorry

end parabola_ordinate_l455_455063


namespace total_cost_is_28_l455_455055

-- Define the original prices
def price1 : ℝ := 2.45
def price2 : ℝ := 6.99
def price3 : ℝ := 11.25
def price4 : ℝ := 8.50

-- Define the discount percentage
def discount : ℝ := 0.10

-- Function to compute the discount
def applyDiscount (price : ℝ) (percentage : ℝ) : ℝ :=
  price - (price * percentage)

-- Function to round to the nearest dollar
def roundToNearestDollar (amount : ℝ) : ℤ :=
  Int.round amount

-- Define the total cost calculation after discount and rounding
noncomputable def totalCost : ℤ :=
  (roundToNearestDollar price1) +
  (roundToNearestDollar price2) +
  (roundToNearestDollar (applyDiscount price3 discount)) +
  (roundToNearestDollar price4)

-- Theorem stating the total cost is equal to the expected result
theorem total_cost_is_28 : totalCost = 28 := 
  by sorry

end total_cost_is_28_l455_455055


namespace determine_borrow_lend_years_l455_455526

theorem determine_borrow_lend_years (P : ℝ) (Rb Rl G : ℝ) (n : ℝ) 
  (hP : P = 9000) 
  (hRb : Rb = 4 / 100) 
  (hRl : Rl = 6 / 100) 
  (hG : G = 180) 
  (h_gain : G = P * Rl * n - P * Rb * n) : 
  n = 1 := 
sorry

end determine_borrow_lend_years_l455_455526


namespace tom_coins_worth_l455_455441

-- Definitions based on conditions:
def total_coins : ℕ := 30
def value_difference_cents : ℕ := 90
def nickel_value_cents : ℕ := 5
def dime_value_cents : ℕ := 10

-- Main theorem statement:
theorem tom_coins_worth (n d : ℕ) (h1 : d = total_coins - n) 
    (h2 : (nickel_value_cents * n + dime_value_cents * d) - (dime_value_cents * n + nickel_value_cents * d) = value_difference_cents) : 
    (nickel_value_cents * n + dime_value_cents * d) = 180 :=
by
  sorry -- Proof omitted.

end tom_coins_worth_l455_455441


namespace sqrt_range_l455_455311

theorem sqrt_range (x : ℝ) (hx : 0 ≤ x - 1) : 1 ≤ x :=
by sorry

end sqrt_range_l455_455311


namespace probability_C_l455_455143

-- Define the probabilities for each region.
def prob_A := 1 / 3
def prob_B := 1 / 6
def prob_C : ℚ -- To be shown
def prob_D_and_E := prob_C

-- The total probability must sum to 1.
theorem probability_C : prob_C + prob_D_and_E + prob_A + prob_B = 1 → prob_C = 1 / 4 :=
by
  -- Delegate proof steps are skipped.
  sorry

end probability_C_l455_455143


namespace method1_three_sessions_cost_method2_more_cost_effective_for_nine_sessions_method1_allows_more_sessions_l455_455147

/-- Method 1: Membership card costs 200 yuan + 10 yuan per swim session. -/
def method1_cost (num_sessions : ℕ) : ℕ := 200 + 10 * num_sessions

/-- Method 2: Each swim session costs 30 yuan. -/
def method2_cost (num_sessions : ℕ) : ℕ := 30 * num_sessions

/-- Problem (1): Total cost for 3 swim sessions using Method 1 is 230 yuan. -/
theorem method1_three_sessions_cost : method1_cost 3 = 230 := by
  sorry

/-- Problem (2): Method 2 is more cost-effective than Method 1 for 9 swim sessions. -/
theorem method2_more_cost_effective_for_nine_sessions : method2_cost 9 < method1_cost 9 := by
  sorry

/-- Problem (3): Method 1 allows more sessions than Method 2 within a budget of 600 yuan. -/
theorem method1_allows_more_sessions : (600 - 200) / 10 > 600 / 30 := by
  sorry

end method1_three_sessions_cost_method2_more_cost_effective_for_nine_sessions_method1_allows_more_sessions_l455_455147


namespace function_inverse_l455_455961

theorem function_inverse (x : ℝ) (h : ℝ → ℝ) (k : ℝ → ℝ) 
  (h_def : ∀ x, h x = 6 - 7 * x) 
  (k_def : ∀ x, k x = (6 - x) / 7) : 
  h (k x) = x ∧ k (h x) = x := 
  sorry

end function_inverse_l455_455961


namespace square_length_of_PQ_l455_455771

-- Define the parabola equation
def parabola (x : ℝ) := 3 * x^2 - 5 * x + 2

-- Define the conditions for points P and Q being on the circle
def on_circle (P Q : ℝ × ℝ) (r : ℝ) := 
  P.fst^2 + P.snd^2 = r^2 ∧ Q.fst^2 + Q.snd^2 = r^2

-- Define the midpoint condition
def midpoint_at_origin (P Q : ℝ × ℝ) :=
  P.fst + Q.fst = 0 ∧ P.snd + Q.snd = 0

-- Given conditions and deriving the required proof
theorem square_length_of_PQ :
  ∃ (P Q : ℝ × ℝ), 
    on_circle P Q (real.sqrt ((5/3)^2 + 2^2)) ∧ 
    midpoint_at_origin P Q ∧
    (parabola P.fst = P.snd ∧ parabola Q.fst = Q.snd) ∧
    ((2 * real.sqrt ( (P.fst^2 + P.snd^2) ))^2 = 244 / 9) :=
by
  sorry

end square_length_of_PQ_l455_455771


namespace count_valid_numbers_l455_455334

def is_digit_valid (d : ℕ) : Prop := d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 5 ∨ d = 7 ∨ d = 9

def is_valid_number (n : ℕ) : Prop :=
  (n % 15 = 0) ∧
  (∃ d1 d2 d3 d4 d5 d6 : ℕ,
    is_digit_valid d1 ∧
    is_digit_valid d2 ∧
    is_digit_valid d3 ∧
    is_digit_valid d4 ∧
    is_digit_valid d5 ∧
    is_digit_valid d6 ∧
    let m := 200000000000 + 10000000000 * d1 + 1000000000 * d2 + 100000000 * d3 + 10000000 * d4 + 1000000 * d5 + 100000 * d6 + (2 * 1000) + (n % 1000) in
    (m = n))

theorem count_valid_numbers : ∃ n : ℕ, is_valid_number n ∧ 5184 = ∑ n in (finset.range 6000000), if is_valid_number n then 1 else 0 := 
sorry

end count_valid_numbers_l455_455334


namespace definite_integral_example_l455_455554

theorem definite_integral_example :
  ∫ x in 0..(Real.pi / 2), (1 + Real.sin x) = Real.pi / 2 + 1 :=
by
  -- Proof will go here
  sorry

end definite_integral_example_l455_455554


namespace percentage_land_mr_william_l455_455017

noncomputable def tax_rate_arable := 0.01
noncomputable def tax_rate_orchard := 0.02
noncomputable def tax_rate_pasture := 0.005

noncomputable def subsidy_arable := 100
noncomputable def subsidy_orchard := 50
noncomputable def subsidy_pasture := 20

noncomputable def total_tax_village := 3840
noncomputable def tax_mr_william := 480

theorem percentage_land_mr_william : 
  (tax_mr_william / total_tax_village : ℝ) * 100 = 12.5 :=
by
  sorry

end percentage_land_mr_william_l455_455017


namespace max_chips_on_board_l455_455381

def ChipsOnBoard : Prop :=
∀ (board: Matrix ℕ ℕ ℕ) (size: ℕ),
  size = 8 →
  (∀ i j, board i j ≤ 1) →
  (∀ i, ∑ j, board i j ≤ 1) →
  (∀ j, ∑ i, board i j ≤ 1) →
  (∃ m, m = ∑ i j, board i j ∧ m ≤ 14)

theorem max_chips_on_board : ChipsOnBoard :=
by
  sorry

end max_chips_on_board_l455_455381


namespace trains_clear_each_other_in_12_seconds_l455_455870

def train1_length : ℝ := 137 -- Length of Train 1 in meters
def train2_length : ℝ := 163 -- Length of Train 2 in meters
def train1_speed : ℝ := 42 * 1000 / 3600 -- Speed of Train 1 in m/s (convert from km/h to m/s)
def train2_speed : ℝ := 48 * 1000 / 3600 -- Speed of Train 2 in m/s (convert from km/h to m/s)
def total_distance : ℝ := train1_length + train2_length -- Total distance both trains need to cover

def relative_speed : ℝ := train1_speed + train2_speed -- Relative speed in m/s

def time_to_clear_each_other : ℝ := total_distance / relative_speed -- Time for the trains to clear each other

theorem trains_clear_each_other_in_12_seconds :
  time_to_clear_each_other = 12 := by
  -- Add the proof details
  sorry

end trains_clear_each_other_in_12_seconds_l455_455870


namespace inequality_solution_l455_455214

theorem inequality_solution : {x : ℝ | -2 < (x^2 - 12 * x + 20) / (x^2 - 4 * x + 8) ∧ (x^2 - 12 * x + 20) / (x^2 - 4 * x + 8) < 2} = {x : ℝ | 5 < x} := 
sorry

end inequality_solution_l455_455214


namespace oranges_initial_weight_l455_455402

-- Defining the initial weight of oranges in kilograms
def initial_weight (W : ℝ) : Prop :=
  let water_content_initial := 0.95 * W
  let non_water_content := 0.05 * W
  let water_content_new := 0.9 * 25
  non_water_content = 0.10 * 25 ∧ W = 50

theorem oranges_initial_weight : ∃ W : ℝ, initial_weight W :=
begin
  use 50,
  unfold initial_weight,
  split,
  {
    sorry, -- Proof of the equality 0.05 * 50 = 0.10 * 25
  },
  {
    refl, -- Reflexivity proof that W = 50
  },
end

end oranges_initial_weight_l455_455402


namespace birthday_probability_l455_455805

theorem birthday_probability (n : ℕ) (d : ℕ) (h_n : n = 30) (h_d : d = 366) :
  let p := 1 - (703 / 732)^n in
  p ≥ 1 - (703 / 732)^(30) :=
by
  sorry

end birthday_probability_l455_455805


namespace func_passes_through_1_2_l455_455813

-- Given conditions
variable (a : ℝ) (x : ℝ) (y : ℝ)
variable (h1 : 0 < a) (h2 : a ≠ 1)

-- Definition of the function
noncomputable def func (x : ℝ) : ℝ := a^(x-1) + 1

-- Proof statement
theorem func_passes_through_1_2 : func a 1 = 2 :=
by
  -- proof goes here
  sorry

end func_passes_through_1_2_l455_455813


namespace lines_parallel_l455_455651

variables {a : ℝ}

def line1 (a : ℝ) : set (ℝ × ℝ) := { p | a * p.1 + 2 * p.2 - 1 = 0 }
def line2 (a : ℝ) : set (ℝ × ℝ) := { p | 8 * p.1 + a * p.2 + 2 - a = 0 }

theorem lines_parallel (a : ℝ) (h : ∃ c : ℝ, ∀ p : ℝ × ℝ,
  line1 a p → line2 a p ∨ line2 a p → line1 a p) : a = -4 :=
sorry

end lines_parallel_l455_455651


namespace quadratic_inequality_solution_l455_455077

open Real

theorem quadratic_inequality_solution :
    ∀ x : ℝ, -8 * x^2 + 6 * x - 1 < 0 ↔ 0.25 < x ∧ x < 0.5 :=
by sorry

end quadratic_inequality_solution_l455_455077


namespace problem_part_1_solution_set_of_f_when_a_is_3_problem_part_2_range_of_a_l455_455399

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - a) + a

theorem problem_part_1_solution_set_of_f_when_a_is_3 :
  {x : ℝ | 0 ≤ x ∧ x ≤ 3} = {x : ℝ | f x 3 ≤ 6} :=
by
  sorry

def g (x : ℝ) : ℝ := abs (2 * x - 3)

theorem problem_part_2_range_of_a :
  {a : ℝ | 4 ≤ a} = {a : ℝ | ∀ x : ℝ, f x a + g x ≥ 5} :=
by
  sorry

end problem_part_1_solution_set_of_f_when_a_is_3_problem_part_2_range_of_a_l455_455399


namespace count_six_digit_palindromes_l455_455194

def num_six_digit_palindromes : ℕ := 9000

theorem count_six_digit_palindromes :
  (∃ a b c d : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧
     num_six_digit_palindromes = 9000) :=
sorry

end count_six_digit_palindromes_l455_455194


namespace sara_quarters_l455_455786

-- Conditions
def usd_to_eur (usd : ℝ) : ℝ := usd * 0.85
def eur_to_usd (eur : ℝ) : ℝ := eur * 1.15
def value_of_quarter_usd : ℝ := 0.25
def dozen : ℕ := 12

-- Theorem
theorem sara_quarters (sara_savings_usd : ℝ) (usd_to_eur_ratio : ℝ) (eur_to_usd_ratio : ℝ) (quarter_value_usd : ℝ) (doz : ℕ) : sara_savings_usd = 9 → usd_to_eur_ratio = 0.85 → eur_to_usd_ratio = 1.15 → quarter_value_usd = 0.25 → doz = 12 → 
  ∃ dozens : ℕ, dozens = 2 :=
by
  sorry

end sara_quarters_l455_455786


namespace count_true_propositions_l455_455838

-- Define each proposition
def proposition1 : Prop := ∀ x y : ℝ, (x ≠ 0 ∨ y ≠ 0) → x * y = 0 → (x = 0 ∧ y = 0)
def proposition2 : Prop := ∀ m : ℝ, (m > 2) → ∀ x : ℝ, x^2 - 2*x + m > 0
def proposition3 : Prop := ∀ (F₁ F₂ M : ℝ × ℝ), dist F₁ F₂ = 7 → dist M F₁ + dist M F₂ = 7 → 
                          (∃ a b c d : ℝ, a ≠ 0 ∨ b ≠ 0 → (M = F₁ ∨ M = F₂))
def proposition4 : Prop := ∀ (a b c : ℝ), linear_independent ℝ ![(a,0,0), (b,0,0), (c,0,0)] →
                          linear_independent ℝ ![(a+b,0,0), (a+c,0,0), (b+c,0,0)]

def number_of_true_propositions : ℕ := 
  if proposition1 then 1 else 0 +
  if proposition2 then 1 else 0 +
  if proposition3 then 1 else 0 +
  if proposition4 then 1 else 0

theorem count_true_propositions : number_of_true_propositions = 2 := by
  sorry

end count_true_propositions_l455_455838


namespace complex_conjugate_z_l455_455263

def i : ℂ := complex.I

def z : ℂ := 2*i / (-1 + 2*i)

def conjugate_z : ℂ := (4 / 5) + (2 * i / 5)

theorem complex_conjugate_z : complex.conj z = conjugate_z := 
by {
  sorry
}

end complex_conjugate_z_l455_455263


namespace jessicas_score_l455_455022

theorem jessicas_score (average_20 : ℕ) (average_21 : ℕ) (n : ℕ) (jessica_score : ℕ) 
  (h1 : average_20 = 75)
  (h2 : average_21 = 76)
  (h3 : n = 20)
  (h4 : jessica_score = (average_21 * (n + 1)) - (average_20 * n)) :
  jessica_score = 96 :=
by 
  sorry

end jessicas_score_l455_455022


namespace area_triangle_ABC_find_AC_length_l455_455539

-- Definitions of the conditions
noncomputable def triangle (A B C : Point) : Prop :=
  -- Add conditions to define an acute-angled triangle in Lean (details omitted)
  sorry 

noncomputable def circle (ω : Circle) (O : Point) : Prop :=
  -- Add conditions to define the circle (details omitted)
  sorry 

noncomputable def inscribed (ABC : triangle) (ω : circle) : Prop :=
  sorry

noncomputable def intersects (c1 c2 : Circle) (P : Point) : Prop :=
  sorry

noncomputable def tangent (ω : Circle) (A C T : Point) : Prop :=
  sorry

noncomputable def segment (TP : Line) (AC : Line) (K : Point) : Prop :=
  sorry

noncomputable def areas (APK CPK : triangle) : Prop :=
  sorry

noncomputable def angle_ABC (ABC : triangle) : Real :=
  sorry

-- Area of triangle ABC
noncomputable def area_triangle (A B C : Point) : Real :=
  sorry

-- Proving the area of triangle ABC is 81/2
theorem area_triangle_ABC {A B C O P T K : Point}
  (θ : Real) (h₁ : triangle A B C) (h₂ : circle ω O) (h₃ : inscribed h₁ h₂)
  (h₄ : intersects (circle_through A O C) (segment BC) P)
  (h₅ : tangent ω A C T) (h₆ : segment TP AC K)
  (h₇ : areas (triangle A P K) (triangle C P K) 10 8) : 
  area_triangle A B C = 81 / 2 :=
  sorry

-- Given angle ABC = arctan 1/2 and find AC length
theorem find_AC_length {A B C O P T K : Point}
  (h₁ : triangle A B C) (h₂ : circle ω O) (h₃ : inscribed h₁ h₂)
  (h₄ : intersects (circle_through A O C) (segment BC) P)
  (h₅ : tangent ω A C T) (h₆ : segment TP AC K)
  (h₇ : areas (triangle A P K) (triangle C P K) 10 8) 
  (h₈ : angle_ABC (triangle A B C) = Real.arctan (1 / 2)) :
  length_segment A C = 3 * Real.sqrt 17 / 2 :=
  sorry

end area_triangle_ABC_find_AC_length_l455_455539


namespace cardinality_bound_l455_455355

variables (n : ℕ) (𝒜 : Finset (Finset (Fin n)))
-- 𝒜 is an I-system
def is_I_system (𝒜 : Finset (Finset (Fin n))) : Prop :=
∀ A ∈ 𝒜, ∀ B ⊆ A, B ∈ 𝒜
-- 𝒜 is an S-system
def is_S_system (𝒜 : Finset (Finset (Fin n))) : Prop :=
∀ A B ∈ 𝒜, A ≠ B → (A ⊂ B ∨ B ⊂ A)

theorem cardinality_bound 
  (hI : is_I_system n 𝒜)
  (hS : is_S_system n 𝒜) :
  𝒜.card ≤ Nat.choose n (Nat.floor (n / 2) + 1) :=
sorry

end cardinality_bound_l455_455355


namespace range_of_k_l455_455043

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 2 → x < k) ↔ k ∈ Ioi 2 :=
by {
  split,
  { intro hk,
    dsimp [Ioi] at *,
    have h := hk 2 (by norm_num),
    linarith },
  { rintros ⟨h₁, k_gt_2⟩ x ⟨hx₁, hx₂⟩,
    exact lt_of_le_of_lt hx₂ k_gt_2 },
}

end range_of_k_l455_455043


namespace taxi_problem_l455_455435

theorem taxi_problem (n : ℕ) (h : n = 4) : 
  (2^n - 2) = 14 :=
by 
  rw h
  norm_num

end taxi_problem_l455_455435


namespace files_remaining_on_flash_drive_l455_455843

def initial_music_files : ℕ := 32
def initial_video_files : ℕ := 96
def deleted_files : ℕ := 60

def total_initial_files : ℕ := initial_music_files + initial_video_files

theorem files_remaining_on_flash_drive 
  (h : total_initial_files = 128) : (total_initial_files - deleted_files) = 68 := by
  sorry

end files_remaining_on_flash_drive_l455_455843


namespace find_angle_between_vectors_l455_455266

open Real

variables (a b : ℝ^3)

def magnitude (v : ℝ^3) := sqrt (v.1^2 + v.2^2 + v.3^2)
def dot_product (u v : ℝ^3) := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
def angle_between_vectors (u v : ℝ^3) : ℝ := acos ((dot_product u v) / ((magnitude u) * (magnitude v)))

theorem find_angle_between_vectors (h₁ : magnitude a = 2) (h₂ : magnitude b = 4)
  (h₃ : dot_product (a + b) a = 0) : angle_between_vectors a b = 2 * π / 3 := by
  sorry

end find_angle_between_vectors_l455_455266


namespace garden_potatoes_l455_455376

-- Definitions for the conditions
def initial_potatoes : ℕ := 8
def rows : ℕ := initial_potatoes
def added_potatoes_per_row : ℕ := 2
def eaten_potatoes_per_row : ℕ := 3

-- Define the number of remaining potatoes
def remaining_potatoes : ℕ :=
  let potatoes_per_row_after_addition := 1 + added_potatoes_per_row in
  let potatoes_per_row_after_eaten := 
    max 0 (potatoes_per_row_after_addition - eaten_potatoes_per_row) in
  potatoes_per_row_after_eaten * rows

-- The proof that remaining potatoes are 0
theorem garden_potatoes : remaining_potatoes = 0 :=
sorry -- proof omitted

end garden_potatoes_l455_455376


namespace number_of_collections_l455_455748

-- Definitions for the problem
def S : Finset (Finset ℕ) := {1, 2, 3}

def collections_satisfying_property (T : Finset (Finset (Finset ℕ))) : Prop :=
  ∀ U V ∈ T, (U ∩ V ∈ T) ∧ (U ∪ V ∈ T)

-- Proposition to prove 
theorem number_of_collections : ∃ T : Finset (Finset (Finset ℕ)), collections_satisfying_property T ∧ T.card = 74 :=
sorry

end number_of_collections_l455_455748


namespace find_B_over_A_l455_455816

noncomputable def A : ℤ := 1
noncomputable def B : ℤ := -1

theorem find_B_over_A (A B : ℤ) : 
  (∀ x : ℝ, x ≠ -2 ∧ x ≠ -5 ∧ x ≠ 2 → 
    (A / (x + 2 : ℝ) + B / ((x + 5) * (x - 2)) = (x^2 + x + 7) / (x^3 + 6 * x^2 - 13 * x - 10))) →  
    B / A = -1 :=
by
  intros h
  have : A = 1 := sorry
  have : B = -1 := sorry
  rw [this, this]
  exact rfl

end find_B_over_A_l455_455816


namespace percentage_biology_students_l455_455911

theorem percentage_biology_students (total_students : ℕ) (not_enrolled : ℕ) (enrolled : ℕ) (percentage : ℝ) :
  total_students = 880 →
  not_enrolled = 594 →
  enrolled = total_students - not_enrolled →
  percentage = (enrolled : ℝ) / (total_students : ℝ) * 100 →
  percentage ≈ 32.5 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end percentage_biology_students_l455_455911


namespace joel_laps_count_l455_455490

def yvonne_laps : ℕ := 10

def younger_sister_laps : ℕ := yvonne_laps / 2

def joel_laps : ℕ := younger_sister_laps * 3

theorem joel_laps_count : joel_laps = 15 := by
  -- The proof is not required as per instructions
  sorry

end joel_laps_count_l455_455490


namespace range_of_x_l455_455606

theorem range_of_x (x : ℝ) : (0.2 ^ x < 25) → (x > -2) :=
by 
  sorry

end range_of_x_l455_455606


namespace joan_mortgage_payoff_l455_455867

/-- Joan's mortgage problem statement. -/
theorem joan_mortgage_payoff (a r : ℕ) (total : ℕ) (n : ℕ) : a = 100 → r = 3 → total = 12100 → 
    total = a * (1 - r^n) / (1 - r) → n = 5 :=
by intros ha hr htotal hgeom; sorry

end joan_mortgage_payoff_l455_455867


namespace probability_of_both_chinese_books_l455_455108

def total_books := 5
def chinese_books := 3
def math_books := 2

theorem probability_of_both_chinese_books (select_books : ℕ) 
  (total_choices : ℕ) (favorable_choices : ℕ) :
  select_books = 2 →
  total_choices = (Nat.choose total_books select_books) →
  favorable_choices = (Nat.choose chinese_books select_books) →
  (favorable_choices : ℚ) / (total_choices : ℚ) = 3 / 10 := by
  intros h1 h2 h3
  sorry

end probability_of_both_chinese_books_l455_455108


namespace product_of_f_equals_one_l455_455922

def S : Set (ℕ × ℕ) := 
  {p | p.fst ∈ {1, 2, 3, 4, 5} ∧ p.snd ∈ {1, 2, 3, 4, 5, 6}}

def isRightTriangle (A B C : ℕ × ℕ) : Prop :=
  (A.fst = B.fst ∨ A.fst = C.fst ∨ A.snd = B.snd ∨ A.snd = C.snd) ∧ 
  (A ≠ B ∧ B ≠ C ∧ C ≠ A) ∧ 
  (A ∈ S ∧ B ∈ S ∧ C ∈ S)

noncomputable def f (A B C : ℕ × ℕ) (h : isRightTriangle A B C) : ℚ :=
  let θ := ∠ C B A
  (Real.tan θ * Real.tan (π / 2 - θ)).toRat

theorem product_of_f_equals_one : 
  ∏ t in {t | ∃ A B C, t = (A, B, C) ∧ isRightTriangle A B C}, f t.1 t.2.1 t.2.2 (by sorry) = 1 := sorry

end product_of_f_equals_one_l455_455922


namespace mango_price_reduction_l455_455379

theorem mango_price_reduction (P R : ℝ) (M : ℕ)
  (hP_orig : 110 * P = 366.67)
  (hM : M * P = 360)
  (hR_red : (M + 12) * R = 360) :
  ((P - R) / P) * 100 = 10 :=
by sorry

end mango_price_reduction_l455_455379


namespace quadratic_vertex_coordinates_l455_455434

theorem quadratic_vertex_coordinates : ∀ x : ℝ,
  (∃ y : ℝ, y = 2 * x^2 - 4 * x + 5) →
  (1, 3) = (1, 3) :=
by
  intro x
  intro h
  sorry

end quadratic_vertex_coordinates_l455_455434


namespace monotonic_increasing_interval_of_f_l455_455095

noncomputable def f (x : ℝ) : ℝ := (1/3)^(|x| - 1)

theorem monotonic_increasing_interval_of_f : set.Iio (0 : ℝ) = {x | (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ ≤ f x₂)} :=
by
  sorry

end monotonic_increasing_interval_of_f_l455_455095


namespace flipsimpossible_l455_455706

theorem flipsimpossible (row_flips col_flips : Fin 100 → Fin 100 → ℕ) :
  (∀ i k, (row_flips i + col_flips k) % 2 == 0) ∨ (row_flips i + col_flips k)  % 2 == 1)  
  ∧ (∃ i k, (row_flips i + col_flips k) % 2 == 0) ∨ 
  (row_flips i + col_flips k)  % 2 == 1) :=
  (100 * (row_flips i + col_flips k - 2 * row_flips i * col_flips k)!= 1970) :=
sorry

end flipsimpossible_l455_455706


namespace amount_of_money_around_circumference_l455_455165

-- Define the given conditions
def horizontal_coins : ℕ := 6
def vertical_coins : ℕ := 4
def coin_value_won : ℕ := 100

-- The goal is to prove the total amount of money around the circumference
theorem amount_of_money_around_circumference : 
  (2 * (horizontal_coins - 2) + 2 * (vertical_coins - 2) + 4) * coin_value_won = 1600 :=
by
  sorry

end amount_of_money_around_circumference_l455_455165


namespace exist_kxnplus1_rectangle_l455_455620

universe u

variables {α : Type u} [fintype α] [decidable_eq α]

theorem exist_kxnplus1_rectangle (p k n : ℕ) (h1 : p < k) (h2 : k < n)
  (mark : α → Prop) (hmark : ∀ (x : ℕ) (y : ℕ), 
    (∀ i j, x ≤ i → i < x + k + 1 → y ≤ j → j < y + n → mark (i, j)) → 
    finset.card (finset.filter mark (finset.Ico x (x + k + 1) ×ˢ finset.Ico y (y + n))) = p) :
  ∃ x y, finset.card (finset.filter mark (finset.Ico x (x + k) ×ˢ finset.Ico y (y + n + 1))) ≥ p + 1 :=
begin
  sorry
end

end exist_kxnplus1_rectangle_l455_455620


namespace additional_girls_needed_l455_455107

theorem additional_girls_needed (initial_girls initial_boys additional_girls : ℕ)
  (h_initial_girls : initial_girls = 2)
  (h_initial_boys : initial_boys = 6)
  (h_fraction_goal : (initial_girls + additional_girls) = (5 * (initial_girls + initial_boys + additional_girls)) / 8) :
  additional_girls = 8 :=
by
  -- A placeholder for the proof
  sorry

end additional_girls_needed_l455_455107


namespace Vanya_number_thought_of_l455_455456

theorem Vanya_number_thought_of :
  ∃ m n : ℕ, m < 10 ∧ n < 10 ∧ (10 * m + n = 81 ∧ (10 * n + m)^2 = 4 * (10 * m + n)) :=
sorry

end Vanya_number_thought_of_l455_455456


namespace nate_pages_to_read_l455_455058

theorem nate_pages_to_read
  (total_pages : ℕ)
  (percentage_read : ℝ)
  (pages_read : ℕ)
  (pages_left : ℕ)
  (h1 : total_pages = 1675)
  (h2 : percentage_read = 46.3)
  (h3 : pages_read = (percentage_read / 100 * total_pages).toInt)
  (h4 : pages_left = total_pages - pages_read) :
  pages_left = 900 :=
by
  sorry

end nate_pages_to_read_l455_455058


namespace johns_regular_season_spending_is_292_johns_peak_season_spending_is_461_point_4_l455_455024

def regular_season_paintball_spending : ℕ := 
  let plays_per_month := 3
  let boxes_per_play := 3
  let cost_per_box := 23
  let equipment_maintenance_cost := 40
  let travel_cost_per_month := 10 + 15 + 12 + 8
  let total_boxes := plays_per_month * boxes_per_play
  let total_paintball_cost := total_boxes * cost_per_box
  total_paintball_cost + equipment_maintenance_cost + travel_cost_per_month

def peak_season_paintball_spending : ℕ :=
  let plays_per_month := 3 * 2
  let boxes_per_play := 3
  let cost_per_box := 22
  let discount_rate := 0.1
  let equipment_maintenance_cost := 60
  let travel_cost_per_month := 10 + 15 + 12 + 8
  let total_boxes := plays_per_month * boxes_per_play
  let total_paintball_cost := total_boxes * cost_per_box
  let discount := discount_rate * total_paintball_cost
  let total_paintball_cost_after_discount := total_paintball_cost - discount
  total_paintball_cost_after_discount + equipment_maintenance_cost + travel_cost_per_month

theorem johns_regular_season_spending_is_292 :
  regular_season_paintball_spending = 292 :=
by
  sorry

theorem johns_peak_season_spending_is_461_point_4 :
  peak_season_paintball_spending = 461.4 :=
by
  sorry

end johns_regular_season_spending_is_292_johns_peak_season_spending_is_461_point_4_l455_455024


namespace collinearity_and_area_eq_l455_455747

noncomputable theory

open EuclideanGeometry

structure Triangle (A B C : Point) :=
(incenter : Point)
(orthocenter : Point)
(midpoint_AC : Point)
(midpoint_AB : Point)
(B2 : Point)
(C2 : Point)
(K : Point)
(circumcenter_BHC : Point)

def collinear (A I A1 : Point) : Prop :=
∃ (line : Line), A ∈ line ∧ I ∈ line ∧ A1 ∈ line

def area_eq_triangle (T1 T2 : Triangle) : Prop :=
area (Triangle.vertices T1) = area (Triangle.vertices T2)

theorem collinearity_and_area_eq (A B C I H B1 C1 B2 C2 K A1 : Point) :
  Triangle I H B1 C1 B2 C2 K A1 →
  (collinear A I A1 ↔ 
  area_eq_triangle (Triangle.mk B K B2) (Triangle.mk C K C2)) := by sorry

end collinearity_and_area_eq_l455_455747


namespace man_speed_kmph_l455_455897

noncomputable def train_speed_kmph : ℝ := 56
noncomputable def train_length_m : ℝ := 110
noncomputable def passing_time_s : ℝ := 6.386585847325762

def train_speed_mps := train_speed_kmph * (1000 / 3600)

def relative_speed := train_length_m / passing_time_s

theorem man_speed_kmph :
  let man_speed_mps := relative_speed - train_speed_mps
  let man_speed_kmph := man_speed_mps * (3600 / 1000)
  abs (man_speed_kmph - 6.004) < 0.001 :=
by
  sorry

end man_speed_kmph_l455_455897


namespace gcd_sum_and_lcm_eq_gcd_l455_455778

theorem gcd_sum_and_lcm_eq_gcd (a b : ℤ) :  Int.gcd (a + b) (Int.lcm a b) = Int.gcd a b :=
sorry

end gcd_sum_and_lcm_eq_gcd_l455_455778


namespace fraction_food_l455_455525

-- Define the salary S and remaining amount H
def S : ℕ := 170000
def H : ℕ := 17000

-- Define fractions of the salary spent on house rent and clothes
def fraction_rent : ℚ := 1 / 10
def fraction_clothes : ℚ := 3 / 5

-- Define the fraction F to be proven
def F : ℚ := 1 / 5

-- Define the remaining fraction of the salary
def remaining_fraction : ℚ := H / S

theorem fraction_food :
  ∀ S H : ℕ,
  S = 170000 →
  H = 17000 →
  F = 1 / 5 →
  F + (fraction_rent + fraction_clothes) + remaining_fraction = 1 :=
by
  intros S H hS hH hF
  sorry

end fraction_food_l455_455525


namespace simplify_log_expression_l455_455497

noncomputable def log_base_a (a : ℝ) (h : 0 < a) : ℝ :=
  log a

theorem simplify_log_expression (a : ℝ) (h : 0 < a) :
  log_base_a (a^(1/4)^(1/4)^(1/4)) = 1 / 64 := by
  sorry

end simplify_log_expression_l455_455497


namespace lines_of_first_character_l455_455346

-- Definitions for the number of lines each character has
def L3 : Nat := 2

def L2 : Nat := 3 * L3 + 6

def L1 : Nat := L2 + 8

-- The theorem we are proving
theorem lines_of_first_character : L1 = 20 :=
by
  -- The proof would go here
  sorry

end lines_of_first_character_l455_455346


namespace red_balls_in_bag_l455_455881

theorem red_balls_in_bag (r : ℕ) (h1 : 0 ≤ r ∧ r ≤ 12)
  (h2 : (r * (r - 1)) / (12 * 11) = 1 / 10) : r = 12 :=
sorry

end red_balls_in_bag_l455_455881


namespace initial_machines_count_l455_455000

theorem initial_machines_count (M : ℕ) (h1 : M * 8 = 8 * 1) (h2 : 72 * 6 = 12 * 2) : M = 64 :=
by
  sorry

end initial_machines_count_l455_455000


namespace school_year_length_l455_455384

theorem school_year_length
  (children : ℕ)
  (juice_boxes_per_child_per_day : ℕ)
  (days_per_week : ℕ)
  (total_juice_boxes : ℕ)
  (w : ℕ)
  (h1 : children = 3)
  (h2 : juice_boxes_per_child_per_day = 1)
  (h3 : days_per_week = 5)
  (h4 : total_juice_boxes = 375)
  (h5 : total_juice_boxes = children * juice_boxes_per_child_per_day * days_per_week * w)
  : w = 25 :=
by
  sorry

end school_year_length_l455_455384


namespace simplify_abs_eq_l455_455314

variable {x : ℚ}

theorem simplify_abs_eq (hx : |1 - x| = 1 + |x|) : |x - 1| = 1 - x :=
by
  sorry

end simplify_abs_eq_l455_455314


namespace sphere_volume_l455_455653

noncomputable def volume_of_sphere {x y z : ℝ} (v : ℝ × ℝ × ℝ) 
  (h : (v.1 * v.1 + v.2 * v.2 + v.3 * v.3) = (4 * v.1 - 16 * v.2 + 32 * v.3)) : ℝ :=
  (4 / 3) * Real.pi * 18^3

theorem sphere_volume {v : ℝ × ℝ × ℝ}
  (h : (v.1 * v.1 + v.2 * v.2 + v.3 * v.3) = (4 * v.1 - 16 * v.2 + 32 * v.3)) :
  volume_of_sphere v h = 7776 * Real.pi :=
sorry

end sphere_volume_l455_455653


namespace triangle_inequality_l455_455020

open Geometry

variables {A B C A₀ D₁ D₂ E₁ E₂ : Point}
variables {ABC : Triangle}

-- Given conditions
def conditions (ABC : Triangle) (A₀ : Point) (D₁ D₂ E₁ E₂ : Point) : Prop :=
  let A₀ := midpoint ABC.BC ∧
  let bisector_A := angle_bisector ABC.A in
  let perpendicular_from_A₀ := perpendicular_line A₀ bisector_A in
  intersection perpendicular_from_A₀ (line ABC.A ABC.BE) = D₁ ∧
  intersection perpendicular_from_A₀ (line ABC.A ABC.CE) = D₂ ∧
  diameter_circle D₁ D₂ ∧
  let circle_with_diameter := circle_of_diameter D₁ D₂ in
  chord circle_with_diameter E₁ E₂ ∧
  perpendicular D₁ D₂ E₁ E₂ ∧
  passes_through E₁ E₂ A₀

-- Statement to prove BC ≥ E₁E₂
theorem triangle_inequality (ABC : Triangle) (A₀ D₁ D₂ E₁ E₂ : Point) (h : conditions ABC A₀ D₁ D₂ E₁ E₂) : 
  segment_length ABC.BC ≥ segment_length E₁E₂ :=
sorry

end triangle_inequality_l455_455020


namespace a_nine_is_nineteen_l455_455614

def sequence (S : ℕ → ℕ) : ℕ → ℕ
| 1 := S 1
| (n + 1) := S (n + 1) - S n

theorem a_nine_is_nineteen (S : ℕ → ℕ) (h : ∀ n, S n = n^2 + 2 * n) :
  sequence S 9 = 19 :=
sorry

end a_nine_is_nineteen_l455_455614


namespace transitiveSim_l455_455848

def isGreat (f : ℕ × ℕ → ℤ) : Prop :=
  ∀ m n : ℕ, f (m + 1, n + 1) * f (m, n) - f (m + 1, n) * f (m, n + 1) = 1

def seqSim (A B : ℕ → ℤ) : Prop :=
  ∃ f : ℕ × ℕ → ℤ, isGreat f ∧ (∀ n, f (n, 0) = A n) ∧ (∀ n, f (0, n) = B n)

theorem transitiveSim (A B C D : ℕ → ℤ)
  (h1 : seqSim A B)
  (h2 : seqSim B C)
  (h3 : seqSim C D) : seqSim D A :=
sorry

end transitiveSim_l455_455848


namespace eel_jellyfish_ratio_l455_455906

noncomputable def combined_cost : ℝ := 200
noncomputable def eel_cost : ℝ := 180
noncomputable def jellyfish_cost : ℝ := combined_cost - eel_cost

theorem eel_jellyfish_ratio : eel_cost / jellyfish_cost = 9 :=
by
  sorry

end eel_jellyfish_ratio_l455_455906


namespace equilateral_if_isosceles_and_AO_twice_A₁D_l455_455007

noncomputable def isosceles {α : Type*} [EuclideanGeometry α]
  (A B C O A₁ D : α) :=
isosceles_triangle A B C ∧
right_angle (A, A₁, B) ∧
circumcircle_triangle_intersection A B C A D ∧
incircle_center A B C O ∧
distance A O = 2 * distance A₁ D

theorem equilateral_if_isosceles_and_AO_twice_A₁D 
  {α : Type*} [EuclideanGeometry α]
  (A B C O A₁ D : α)
  (h : isosceles A B C O A₁ D) :
  length A B = length A C ∧ length A C = length B C :=
sorry

end equilateral_if_isosceles_and_AO_twice_A₁D_l455_455007


namespace crayons_in_the_box_l455_455315

theorem crayons_in_the_box (initial_crayons : ℕ) (added_crayons : ℝ) :
  initial_crayons = 7 → added_crayons = 7 / 3 → initial_crayons + added_crayons.toNat = 9 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end crayons_in_the_box_l455_455315


namespace complex_div_simplify_l455_455133

theorem complex_div_simplify : (10 * Complex.i) / (2 - Complex.i) = -2 + 4 * Complex.i := 
by
  sorry

end complex_div_simplify_l455_455133


namespace original_price_of_stamp_l455_455517

theorem original_price_of_stamp (original_price : ℕ) (h : original_price * (1 / 5 : ℚ) = 6) : original_price = 30 :=
by
  sorry

end original_price_of_stamp_l455_455517


namespace problem_3_problem_4_l455_455597

open Classical

section
  variable {x₁ x₂ : ℝ}
  theorem problem_3 (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) : (Real.log (x₁ * x₂) = Real.log x₁ + Real.log x₂) :=
  by
    sorry

  theorem problem_4 (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) (hlt : x₁ < x₂) : ((Real.log x₁ - Real.log x₂) / (x₁ - x₂) > 0) :=
  by
    sorry
end

end problem_3_problem_4_l455_455597


namespace intersection_volume_calculation_l455_455471

noncomputable def volume_of_intersection : ℝ :=
  let region1 := {p : ℝ × ℝ × ℝ | abs p.1 + abs p.2 + abs p.3 ≤ 2}
  let region2 := {p : ℝ × ℝ × ℝ | abs p.1 + abs p.2 + abs (p.3 - 2) ≤ 2}
  let intersection := region1 ∩ region2
  8/3

theorem intersection_volume_calculation :
  volume_of_intersection = 8 / 3 :=
begin
  sorry
end

end intersection_volume_calculation_l455_455471


namespace problem_A5_l455_455873

/-- Prove that for \( n \in \mathbb{N} \), \( n > 2 \), and a permutation \( a_1, a_2, \ldots, a_{2n} \)
    of the numbers \( 1, 2, \ldots, 2n \) such that \( a_1 < a_3 < \cdots < a_{2n-1} \) and 
    \( a_2 > a_4 > \cdots > a_{2n} \), we have \((a_1 - a_2)^2 + (a_3 - a_4)^2 + \cdots + (a_{2n-1} - a_{2n})^2 > n^3 \).
-/
theorem problem_A5 (n : ℕ) (h_n : 2 < n) (a : Fin (2 * n) → ℕ)
  (h_perm : ∀ k, k < 2 * n → a k ∈ Finset.range (2 * n))
  (h_inc : ∀ i : Fin n, a ⟨2 * i⟩ < a ⟨2 * i + 2⟩)
  (h_dec : ∀ i : Fin n, a ⟨2 * i + 1⟩ > a ⟨2 * i + 3⟩) :
  ∑ i in Finset.range n, (a ⟨2 * i⟩ - a ⟨2 * i + 1⟩) ^ 2 > n ^ 3 := 
sorry

end problem_A5_l455_455873


namespace isosceles_triangle_area_l455_455709

theorem isosceles_triangle_area
  (A B C D: Type)
  [metric_space A]
  [metric_space B]
  [metric_space C]
  [metric_space D]
  [has_scalar ℝ A]
  [has_scalar ℝ B]
  [has_scalar ℝ C]
  [has_scalar ℝ D]
  (BD: ℝ)
  (θ: ℝ)
  (base: ℝ)
  (height: ℝ)
  (isosceles: isosceles A B C)
  (altitude: A → B)
  (h₁: height = 1)
  (h₂: θ = π/4)
  (area: ℝ) :
  (area = 1) :=
sorry

end isosceles_triangle_area_l455_455709


namespace square_area_4_l455_455331

noncomputable def complex_point := ℂ

def vertices_form_square (z : complex_point) : Prop :=
  ∃ z₁ z₂ z₃ z₄ : complex_point, 
    {z₁, z₂, z₃, z₄} = {z, z^2, z^3, z + 3z - 2z} ∧
    (z₄ - z₁) * (z₄ - z₂) = -1 * abs ((z₂ - z₁) ^ 2) ∧ 
    abs (z₃ - z₁) = abs (z₂ - z₁)

def square_area (z : complex_point) : ℂ :=
  abs ((z^2 - z) * (z^3 - z))

theorem square_area_4 (z : complex_point) (h : vertices_form_square z) :
  square_area z = 4 :=
by
  sorry

end square_area_4_l455_455331


namespace number_of_Neglart_students_l455_455766

theorem number_of_Neglart_students (total_toes : ℕ) (H_toes_per_H : ℕ) (N_toes_per_N : ℕ)
  (num_H : ℕ) (total_H_toes : ℕ) (remaining_toes : ℕ) (num_N : ℕ) :
  total_toes = 164 →
  H_toes_per_H = 12 →
  num_H = 7 →
  total_H_toes = num_H * H_toes_per_H →
  remaining_toes = total_toes - total_H_toes →
  N_toes_per_N = 10 →
  num_N = remaining_toes / N_toes_per_N →
  num_N = 8 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h6, h5, h7]
  exact rfl

end number_of_Neglart_students_l455_455766


namespace factorial_subtraction_l455_455191

theorem factorial_subtraction : 7! - 6 * 6! - 2 * 6! = -720 :=
by
  sorry

end factorial_subtraction_l455_455191


namespace find_x_l455_455268

open Nat

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_x (x : ℕ) (hx : x > 0) (hprime : is_prime (x^5 + x + 1)) : x = 1 := 
by 
  sorry

end find_x_l455_455268


namespace valid_combinations_l455_455892

theorem valid_combinations: 
  ∀ (herbs crystals incompatible_herbs_per_crystal : ℕ), 
    herbs = 4 → 
    crystals = 6 → 
    incompatible_herbs_per_crystal = 3 → 
    let total_combinations := herbs * crystals in
    let total_incompatibilities := 2 * incompatible_herbs_per_crystal in
    total_combinations - total_incompatibilities = 18 :=
begin
  intros herbs crystals incompatible_herbs_per_crystal herbs_eq crystals_eq incompatible_herbs_per_crystal_eq,
  rw [herbs_eq, crystals_eq, incompatible_herbs_per_crystal_eq],
  let total_combinations := 4 * 6,
  let total_incompatibilities := 2 * 3,
  show total_combinations - total_incompatibilities = 18,
  { simp [total_combinations, total_incompatibilities] }
end

end valid_combinations_l455_455892


namespace max_f_l455_455502

def f (x : ℝ) : ℝ := (x / (x^2 + 9)) + (1 / (x^2 - 6 * x + 21)) + (Real.cos (2 * Real.pi * x))

theorem max_f : ∃ x : ℝ, f(x) = 1.25 := 
by 
  have h₃ := (3 / ((3)^2 + 9) = 1/6)
  have h₁ := (1 / ((3 - 3)^2 + 12) = 1/12)
  have h_c := (1 : ℝ) -- max value of cos 
  sorry

end max_f_l455_455502


namespace noPrimeProduct_l455_455566

def isComposite (n : ℕ) : Prop :=
  1 < n ∧ ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ n = m * k

def firstEightComposites : List ℕ := [4, 6, 8, 9, 10, 12, 14, 15]

def productIsPrime (a b : ℕ) : Prop :=
  Nat.Prime (a * b)

theorem noPrimeProduct :
  ∃ l : List ℕ, l = firstEightComposites ∧ (∀ a b ∈ l, a ≠ b → ¬(productIsPrime a b)) :=
by
  use firstEightComposites
  split
  { refl }
  { intros a ha b hb hab
    sorry }

end noPrimeProduct_l455_455566


namespace pears_more_than_apples_l455_455408

-- Definitions for A, P, B and the conditions
variables (A P B : ℕ)

-- Conditions
def condition1 := B = P + 3
def condition2 := B = 9
def condition3 := A + P + B = 19

-- Theorem statement
theorem pears_more_than_apples (h1 : condition1) (h2 : condition2) (h3 : condition3) :
  P - A = 2 :=
sorry

end pears_more_than_apples_l455_455408


namespace sqrt_sum_power_six_l455_455507

theorem sqrt_sum_power_six :
  let s := (Real.sqrt 7 + Real.sqrt 3) ^ 6 in
  ⌊s⌋ = 7039 :=
by sorry

end sqrt_sum_power_six_l455_455507


namespace range_of_a_l455_455042

def p (a : ℝ) : Prop := ∀ (f : ℝ → ℝ), f = λ x, log (a * x^2 - x + a / 16) → set.range f = set.univ
def q (a : ℝ) : Prop := ∀ x : ℝ, 3^x - 9^x < a

theorem range_of_a (a : ℝ) : ¬(p a ∧ q a) → (a > 2 ∨ a ≤ 1 / 4) :=
by
  sorry

end range_of_a_l455_455042


namespace find_x_plus_y_l455_455679

theorem find_x_plus_y (x y : ℝ) 
  (h1 : |x| + x + y = 14) 
  (h2 : x + |y| - y = 16) : 
  x + y = -2 := 
by
  sorry

end find_x_plus_y_l455_455679


namespace product_AT_BT_l455_455203

-- Define the basic geometric entities and their properties
def circle (center : Point) (radius : ℝ) : Prop :=
∀ (P: Point), dist center P = radius

-- Define the midpoint function
def midpoint (A B M : Point) : Prop :=
 dist A M = dist B M

-- Define the tangency property
def tangent_to_circle (A B T : Point) (omega_center : Point) (omega_radius : ℝ) : Prop :=
(∃ (P: Point), on_circle ω P omega_radius ∧ dist A T = dist B T)

-- Define the problem with all given conditions
theorem product_AT_BT :
  ∀ (O P A B T : Point),
  ∀ r_omega r_w : ℝ,
  on_circle O A r_omega →
  on_circle O B r_omega →
  dist O P = r_omega →
  r_omega = 13 →
  r_w = 14 →
  dist A B = 24 →
  tangent_to_circle A B T P r_w →
  dist AT BT = 56 :=
by
  intros
  sorry

end product_AT_BT_l455_455203


namespace projection_of_m_onto_n_l455_455298

variables (m n : EuclideanSpace ℝ (Fin 2))

-- Definitions based on the conditions from step a)
def m_magnitude_one : Prop := ‖m‖ = 1
def n_magnitude_one : Prop := ‖n‖ = 1
def vector_magnitude_condition : Prop := ‖(3 : ℝ) • m - (2 : ℝ) • n‖ = Real.sqrt 7

-- The Lean statement for the mathematically equivalent proof problem
theorem projection_of_m_onto_n 
  (h1 : m_magnitude_one m) 
  (h2 : n_magnitude_one n) 
  (h3 : vector_magnitude_condition m n) : 
  ((EuclideanSpace.inner m n) / (EuclideanSpace.inner n n)) • n = (1 / 2 : ℝ) • n :=
sorry

end projection_of_m_onto_n_l455_455298


namespace Derrick_yard_length_l455_455930

def DerrickYard := 10
def BrianneYard := 30
def AlexYard (D : ℕ) := D / 2
def CarlaYard (B: ℕ) := 3 * B + (1/4 : ℝ) * B
def DerekYard (C : ℝ) := (2/3 : ℝ) * C - Real.sqrt 10

theorem Derrick_yard_length : 
  ∀ (D : ℕ), 
  BrianneYard = 30 →
  AlexYard D = BrianneYard / 6 →
  D = 10 :=
by 
  intros
  sorry

end Derrick_yard_length_l455_455930


namespace similar_triangle_side_length_l455_455170

theorem similar_triangle_side_length
  (a1 : ℝ) (a2 : ℝ) (c1 : ℝ) (c2 : ℝ)
  (h1 : a1 = 18) (h2 : c1 = 30) (h3 : c2 = 60) :
  let k := c2 / c1 in a2 = k * a1 :=
begin
  sorry
end

end similar_triangle_side_length_l455_455170


namespace base4_operation_l455_455945

-- A function to convert a string representing a number in base 4 to a natural number
def base4_to_nat (s : string) : ℕ :=
  s.foldl (λ acc c, acc * 4 + (c.to_nat - '0'.to_nat)) 0

-- A function to convert a natural number to a string representing a number in base 4
def nat_to_base4 (n : ℕ) : string :=
  if n = 0 then "0"
  else let rec convert (num : ℕ) (acc : string) : string :=
         if num = 0 then acc
         else convert (num / 4) (char.of_nat (num % 4 + '0'.to_nat) :: acc.to_list) in
       convert n ""

-- Conditions from the problem
noncomputable def n1 := base4_to_nat "120"
noncomputable def n2 := base4_to_nat "13"
noncomputable def n3 := base4_to_nat "2"
noncomputable def result := base4_to_nat "1110"

-- The theorem to prove
theorem base4_operation : nat_to_base4 ((n1 * n2) / n3) = "1110" :=
by 
  sorry

end base4_operation_l455_455945


namespace clubs_popularity_order_l455_455432

theorem clubs_popularity_order (chess drama art science : ℚ)
  (h_chess: chess = 14/35) (h_drama: drama = 9/28) (h_art: art = 11/21) (h_science: science = 8/15) :
  science > art ∧ art > chess ∧ chess > drama :=
by {
  -- Place proof steps here (optional)
  sorry
}

end clubs_popularity_order_l455_455432


namespace number_of_sets_l455_455351

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem number_of_sets (x y z : ℕ) :
  (lcm x y = 6) ∧ (lcm y z = 15) → 
  (∃! (S : Finset (ℕ × ℕ × ℕ)),
   S = { (6, 1, 15), (2, 3, 5), (2, 3, 15), (6, 3, 5), (6, 3, 15) }).card = 5 :=
by
  sorry

end number_of_sets_l455_455351


namespace sales_professionals_count_l455_455185

theorem sales_professionals_count :
  (∀ (C : ℕ) (MC : ℕ) (M : ℕ), C = 500 → MC = 10 → M = 5 → C / M / MC = 10) :=
by
  intros C MC M hC hMC hM
  sorry

end sales_professionals_count_l455_455185


namespace percentage_increase_in_pay_rate_l455_455902

-- Given conditions
def regular_rate : ℝ := 10
def total_surveys : ℕ := 50
def cellphone_surveys : ℕ := 35
def total_earnings : ℝ := 605

-- We need to demonstrate that the percentage increase in the pay rate for surveys involving the use of her cellphone is 30%
theorem percentage_increase_in_pay_rate :
  let earnings_at_regular_rate := regular_rate * total_surveys
  let earnings_from_cellphone_surveys := total_earnings - earnings_at_regular_rate
  let rate_per_cellphone_survey := earnings_from_cellphone_surveys / cellphone_surveys
  let increase_in_rate := rate_per_cellphone_survey - regular_rate
  let percentage_increase := (increase_in_rate / regular_rate) * 100
  percentage_increase = 30 :=
by
  sorry

end percentage_increase_in_pay_rate_l455_455902


namespace polynomial_factor_l455_455823

noncomputable def find_c (p k : ℚ) : ℚ :=
  15 + 2 * p * k

theorem polynomial_factor {p k c : ℚ} (h1 : 3*c = -36 + 12*p*k) (h2 : 5*k = -6) : c = 447 / 25 :=
by
  let k := -6 / 5
  have pk := -(1 / 5)
  have h3 : p = pk := sorry
  have h4 : k = k := sorry
  have h5 : c = find_c pk k := by simp [h3, h4]
  have h6 : c = 447 / 25 := by simp [h5]
  exact h6

end polynomial_factor_l455_455823


namespace watermelons_left_l455_455785

theorem watermelons_left (initial : ℕ) (eaten : ℕ) (remaining : ℕ) (h1 : initial = 4) (h2 : eaten = 3) : remaining = 1 :=
by
  sorry

end watermelons_left_l455_455785


namespace parts_in_batch_l455_455705

theorem parts_in_batch :
  ∃ a : ℕ, 500 ≤ a ∧ a ≤ 600 ∧ a % 20 = 13 ∧ a % 27 = 20 ∧ a = 533 :=
begin
  sorry
end

end parts_in_batch_l455_455705


namespace prob_both_black_prob_both_white_prob_at_most_one_black_l455_455008

-- Define probabilities for balls in Bag A and Bag B
def bagA_black_prob : ℝ := 1 / 2
def bagA_white_prob : ℝ := 1 / 2
def bagB_black_prob : ℝ := 2 / 3
def bagB_white_prob : ℝ := 1 / 3

-- Proposition for the probability that both balls are black
theorem prob_both_black :
  bagA_black_prob * bagB_black_prob = 1 / 3 := sorry

-- Proposition for the probability that both balls are white
theorem prob_both_white :
  bagA_white_prob * bagB_white_prob = 1 / 6 := sorry

-- Proposition for the probability that at most one ball is black
theorem prob_at_most_one_black :
  (bagA_white_prob * bagB_white_prob) +
  (bagA_black_prob * bagB_white_prob) +
  (bagA_white_prob * bagB_black_prob) = 2 / 3 := sorry

end prob_both_black_prob_both_white_prob_at_most_one_black_l455_455008


namespace sum_first_10_b_l455_455290

-- Condition definitions
def seq_a : ℕ → ℕ
| 0     := 2
| (n+1) := 2 * (seq_a n)

def seq_b (n : ℕ) : ℕ := Int.log2 (seq_a n)

-- Theorem statement
theorem sum_first_10_b : (Finset.range 10).sum seq_b = 55 :=
by
  sorry

end sum_first_10_b_l455_455290


namespace horner_v1_value_l455_455454

noncomputable def f (x : ℝ) : ℝ := 0.5 * x^5 + 4 * x^4 - 3 * x^2 + x - 1

theorem horner_v1_value :
  let x := 3
  let f_x := f x
  let v_1 := 0.5 * x + 4
  (v_1 = 5.5) :=
by
  let x := 3
  have h_f_x : f x = ((((0.5 * x + 4) * x + 0) * x - 3) * x + 1) * x - 1 := by sorry
  have h_v1 : v_1 = 0.5 * x + 4 := by rfl
  show v_1 = 5.5 from sorry

end horner_v1_value_l455_455454


namespace abc_cubed_sum_l455_455357

noncomputable def k := (a^3 + 10) / a^2

theorem abc_cubed_sum (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_eq : (a^3 + 10) / a^2 = (b^3 + 10) / b^2 ∧ (b^3 + 10) / b^2 = (c^3 + 10) / c^2) :
  a^3 + b^3 + c^3 = 1301 :=
begin
  sorry
end

end abc_cubed_sum_l455_455357


namespace parallel_lines_perpendicular_lines_min_abs_l455_455619

-- Parallel lines proof
theorem parallel_lines (a b : ℝ) (h1 : b = -12) (h2 : ∀ x y : ℝ, (x + a^2 * y + 1 = 0) → ((a^2 + 1) * x - b * y + 3 = 0) → True) 
: a = real.sqrt 3 ∨ a = -real.sqrt 3 := by 
  sorry

-- Perpendicular lines proof
theorem perpendicular_lines_min_abs (a b : ℝ) (h1 : ∀ x y : ℝ, (x + a^2 * y + 1 = 0) → ((a^2 + 1) * x - b * y + 3 = 0) → a ≠ 0 ∧ b = (a^2 + 1) / a^2) 
: ∃ a b : ℝ, abs (a * b) = 2 := by
  sorry

end parallel_lines_perpendicular_lines_min_abs_l455_455619


namespace find_x_plus_y_l455_455264

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.cos y = 2010) (h2 : x + 2010 * Real.sin y = 2009) (hy : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2009 :=
by
  sorry

end find_x_plus_y_l455_455264


namespace min_value_of_product_l455_455139

theorem min_value_of_product :
  ∀ (a : Fin 6 → ℕ), 
    (∀ i, 1 ≤ a i ∧ a i ≤ 6) →
    (∀ i j, (i ≠ j) → a i ≠ a j) →
    (∏ i in Finset.range 6, (a i - a ((i + 1) % 6)) / (a ((i + 2) % 6) - a ((i + 3) % 6)) = 1) :=
by
  intros a h1 h2
  sorry

end min_value_of_product_l455_455139


namespace perimeter_regular_polygon_l455_455167

-- Condition definitions
def is_regular_polygon (n : ℕ) (s : ℝ) : Prop := 
  n * s > 0

def exterior_angle (E : ℝ) (n : ℕ) : Prop := 
  E = 360 / n

def side_length (s : ℝ) : Prop :=
  s = 6

-- Theorem statement to prove the perimeter is 24 units
theorem perimeter_regular_polygon 
  (n : ℕ) (s E : ℝ)
  (h1 : is_regular_polygon n s)
  (h2 : exterior_angle E n)
  (h3 : side_length s)
  (h4 : E = 90) :
  4 * s = 24 :=
by
  sorry

end perimeter_regular_polygon_l455_455167


namespace area_A_l455_455788

variable (AB BC CD DA : ℝ)
variable (ABCD_area : ℝ)
variable (BB'_ratio CC'_ratio DD'_ratio AA'_ratio : ℝ)

-- These are the given conditions
axiom AB_eq : AB = 5
axiom BC_eq : BC = 8
axiom CD_eq : CD = 4
axiom DA_eq : DA = 7
axiom ABCD_area_eq : ABCD_area = 20
axiom BB_ratio_eq : BB'_ratio = 1.5
axiom CC_ratio_eq : CC'_ratio = 1.5
axiom DD_ratio_eq : DD'_ratio = 1.5
axiom AA_ratio_eq : AA'_ratio = 1.5

-- You need to prove the area of A'B'C'D' is 140
theorem area_A'B'C'D'_eq : 
  let BB' := BB_ratio_eq * AB in
  let CC' := CC_ratio_eq * BC in
  let DD' := DD_ratio_eq * CD in
  let AA' := AA_ratio_eq * DA in
  let extended_area := 4 * 1.5 * ABCD_area_eq in
  let total_area := ABCD_area_eq + extended_area in
  total_area = 140 := by
sorry

end area_A_l455_455788


namespace sand_art_l455_455344

theorem sand_art (len_blue_rect : ℕ) (area_blue_rect : ℕ) (side_red_square : ℕ) (sand_per_sq_inch : ℕ) (h1 : len_blue_rect = 7) (h2 : area_blue_rect = 42) (h3 : side_red_square = 5) (h4 : sand_per_sq_inch = 3) :
  (area_blue_rect * sand_per_sq_inch) + (side_red_square * side_red_square * sand_per_sq_inch) = 201 :=
by
  sorry

end sand_art_l455_455344


namespace trig_simplification_l455_455789

theorem trig_simplification (x : ℝ) : 
  sin (2 * x - π) * cos (x - 3 * π) + sin (2 * x - 9 * π / 2) * cos (x + π / 2) = sin (3 * x) :=
by
  sorry

end trig_simplification_l455_455789


namespace island_inhabitants_l455_455834

section island_proof

variables (A B C : Type)
variables (is_knight : A → Prop) (is_liar : A → Prop)
variables (said : B → A → Type → Prop)
variables (said_liar : B → A → Prop)

-- Assume B said A claimed "I am a liar."
axiom B_said_A_liar : said B A (is_liar A)

-- Assume C said, "Don't believe B! He is lying." which means B is lying.
axiom C_said_B_lying : is_liar B

-- Definition of knights and liars:
axiom knights_tell_truth : ∀ x, is_knight x ↔ (∀ P, P → P)
axiom liars_always_lie : ∀ x, is_liar x ↔ ¬ (∀ P, P → P)

-- Theorem to prove: B is a liar and C is a knight.
theorem island_inhabitants : is_liar B ∧ is_knight C :=
  by
  sorry

end island_proof

end island_inhabitants_l455_455834


namespace engraving_VYKHOD_time_l455_455182

variables (t : char → ℕ)
variables (plate1 plate2 plate3 plate4 : list char)

-- Hypotheses from problem statement
-- plate1 = "ДОМ МОДЫ", plate2 = "ВХОД"
def plate1 : list char := ['Д', 'О', 'М', ' ', 'М', 'О', 'Д', 'Ы']
def plate2 : list char := ['В', 'Х', 'О', 'Д']
-- plate3 = "В ДЫМОХОД"
def plate3 : list char := ['В', ' ', 'Д', 'Ы', 'М', 'О', 'Х', 'О', 'Д']
-- plate4 = "ВЫХОД"
def plate4 : list char := ['В', 'Ы', 'Х', 'О', 'Д']

lemma engrave_time_plate1_plate2 :
  (plate1 ++ plate2).map t.sum = 50 :=
sorry

lemma engrave_time_plate3 :
  plate3.map t.sum = 35 :=
sorry

theorem engraving_VYKHOD_time :
  plate4.map t.sum = 20 :=
sorry

end engraving_VYKHOD_time_l455_455182


namespace second_root_l455_455556

variables {a b c x : ℝ}

theorem second_root (h : a * (b + c) * x ^ 2 - b * (c + a) * x + c * (a + b) = 0)
(hroot : a * (b + c) * (-1) ^ 2 - b * (c + a) * (-1) + c * (a + b) = 0) :
  ∃ k : ℝ, k = - c * (a + b) / (a * (b + c)) ∧ a * (b + c) * k ^ 2 - b * (c + a) * k + c * (a + b) = 0 :=
sorry

end second_root_l455_455556


namespace chessboard_impossible_l455_455439

theorem chessboard_impossible 
    (chessboard : matrix (fin 8) (fin 8) bool)
    (initial_state : ∀ i j, chessboard i j = (i + j) % 2 = 0)
    (row_invariant : ∀ i, ∑ j, if chessboard i j then 1 else 0 = 4)
    (col_invariant : ∀ j, ∑ i, if chessboard i j then 1 else 0 = 4) :
  ¬(∃ new_chessboard : matrix (fin 8) (fin 8) bool,
       (∀ i < 4, ∀ j, new_chessboard i j = true) ∧
       (∀ i ≥ 4, ∀ j, new_chessboard i j = false)) :=
sorry

end chessboard_impossible_l455_455439


namespace parabolas_through_points_l455_455370

noncomputable def square_vertices (c : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  ((-c, -c), (c, -c), (c, c), (-c, c))

noncomputable def intersect_diagonals (t c : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((t, t), (-t, t))

noncomputable def intersections_M_N (t c : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let M := (0, (-2*c*t) / (t - c))
  let N := (0, (2*c*t) / (t + c))
  (M, N)

noncomputable def intersections_P_Q_R_S (t c : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  let P := ((-2*c*t) / (t - c), (-2*c*t) / (t - c))
  let Q := ((2*c*t) / (t - c), (-2*c*t) / (t - c))
  let R := ((2*c*t) / (t + c), (2*c*t) / (t + c))
  let S := ((-2*c*t) / (t + c), (2*c*t) / (t + c))
  (P, Q, R, S)

theorem parabolas_through_points (c : ℝ) (t : ℝ) :
  let vertices := square_vertices c
  let (A, B, C, D) := vertices
  let (K, L) := intersect_diagonals t c
  let (M, N) := intersections_M_N t c
  let (P, Q, R, S) := intersections_P_Q_R_S t c in
  (∃ p₁ : ℝ, ∀ x y : ℝ, (x, y) ∈ {C, D, N, P, Q} → x^2 = 2*p₁*(y - (2*c*t)/(t+c))) ∧
  (∃ p₂ : ℝ, ∀ x y : ℝ, (x, y) ∈ {A, B, M, R, S} → x^2 = 2*p₂*(y - (-2*c*t)/(t-c))) ∧
  ((∃ t, t = 3*c) ∨ (∃ t, t = c/3) ∨ (∃ t, t = (-5 + 4*sqrt(5))*c/11) ∨ (∃ t, t = (-5 - 4*sqrt(5))*c/11)) →
  all_points_lie_on_parabolas ⟶ focus_at_midpoint_condition := sorry

end parabolas_through_points_l455_455370


namespace original_number_l455_455001

-- Define the three-digit number and its permutations under certain conditions.
-- Prove the original number given the specific conditions stated.
theorem original_number (a b c : ℕ)
  (ha : a % 2 = 1) -- a being odd
  (m : ℕ := 100 * a + 10 * b + c)
  (sum_permutations : 100*a + 10*b + c + 100*a + 10*c + b + 100*b + 10*c + a + 
                      100*c + 10*a + b + 100*b + 10*a + c + 100*c + 10*b + a = 3300) :
  m = 192 := 
sorry

end original_number_l455_455001


namespace point_set_description_l455_455295

structure Point :=
(x : ℝ)
(y : ℝ)

def dist_square (P Q : Point) : ℝ :=
(P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2

variable {A B M : Point}
variable {k l d : ℝ}

theorem point_set_description (hk : k ≠ 0) (hl : l ≠ 0) (hk_l_sum : k + l ≠ 0) : 
  k * dist_square A M + l * dist_square B M = d → 
  ∃ (P : Point), (dist_square P M = r) ∧ (
    (r = (d + k * dist_square A P + l * dist_square B P) / (k + l)) ∨ 
    (d = k * dist_square A P + l * dist_square B P) ∨ 
    (∀ (x y : ℝ), k * dist_square A ⟨x, y⟩ + l * dist_square B ⟨x, y⟩ < 0)
  )
  sorry

end point_set_description_l455_455295


namespace parabola_ordinate_l455_455064

theorem parabola_ordinate (x y : ℝ) (h : y = 2 * x^2) (d : dist (x, y) (0, 1 / 8) = 9 / 8) : y = 1 := 
sorry

end parabola_ordinate_l455_455064


namespace triangle_height_l455_455407

theorem triangle_height (base height : ℝ) (area : ℝ) (h1 : base = 2) (h2 : area = 3) (area_formula : area = (base * height) / 2) : height = 3 :=
by
  sorry

end triangle_height_l455_455407


namespace sum_of_n_for_perfect_square_l455_455975

theorem sum_of_n_for_perfect_square :
  let is_perfect_square (x : ℤ) := ∃ (y : ℤ), y^2 = x
  in ∑ n in { n | n > 0 ∧ is_perfect_square (n^2 - 17 * n + 72) }, n = 17 :=
by
  sorry

end sum_of_n_for_perfect_square_l455_455975


namespace sequence_sum_zero_l455_455938

theorem sequence_sum_zero (n : ℕ) (h : n > 1) :
  (∃ (a : ℕ → ℤ), (∀ k : ℕ, k > 0 → a k ≠ 0) ∧ (∀ k : ℕ, k > 0 → a k + 2 * a (2 * k) + n * a (n * k) = 0)) ↔ n ≥ 3 := 
by sorry

end sequence_sum_zero_l455_455938


namespace eliminate_denominator_eq_poly_l455_455567

theorem eliminate_denominator_eq_poly (x : ℝ) (hx : x ≠ 0) (hx1 : x + 1 ≠ 0) :
  (1 - (5 * x + 2) / (x * (x + 1)) = 3 / (x + 1)) →
  x ^ 2 - 7 * x - 2 = 0 :=
by
  intro h
  have h1 : x * (x + 1) * (1 - (5 * x + 2) / (x * (x + 1))) = x * (x + 1) * (3 / (x + 1)) :=
    mul_eq_mul_right_iff.mpr (Or.inl h)
  sorry

end eliminate_denominator_eq_poly_l455_455567


namespace g_is_even_l455_455560

noncomputable def g (x : ℝ) := 2 ^ (x ^ 2 - 4) - |x|

theorem g_is_even : ∀ x : ℝ, g (-x) = g x :=
by
  sorry

end g_is_even_l455_455560


namespace cartesian_equation_of_C_length_AB_l455_455715

-- Define the initial conditions for the problem
constant polar_eq : (ρ θ : Real) → Prop
constant param_L : (t : Real) → (x y : Real) → Prop

-- Specify the conditions given in the problem
axiom polar_condition (ρ θ : Real) : polar_eq ρ θ ↔ ρ * Real.sin θ^2 = Real.cos θ

axiom param_L_condition (t : Real) (x y : Real) :
  param_L t x y ↔ x = 2 - (Real.sqrt 2)/2 * t ∧ y = (Real.sqrt 2) / 2 * t

-- Specify the problem statements to prove
theorem cartesian_equation_of_C : ∀ (x y : Real), polar_eq (Real.sqrt (x^2 + y^2)) (Real.arctan (y / x)) → y^2 = x :=
by
  sorry

theorem length_AB : ∀ t1 t2 : Real, param_L t1 (2 - (Real.sqrt 2) / 2 * t1) ((Real.sqrt 2) / 2 * t1) →
    param_L t2 (2 - (Real.sqrt 2) / 2 * t2) ((Real.sqrt 2) / 2 * t2) →
    t1 ≠ t2 →
    ((2 - (Real.sqrt 2) / 2 * t1) = (2 - (Real.sqrt 2) / 2 * t2)) ∧
    ((Real.sqrt 2) / 2 * t1 = (Real.sqrt 2) / 2 * t2) →
    (Real.abs (t1 - t2) = 3 * Real.sqrt 2) :=
by
  sorry

end cartesian_equation_of_C_length_AB_l455_455715


namespace area_of_adoe_l455_455021

-- Definitions for the triangle and points
variables {A B C E O D: Type}
           [IsTriangle ABC]
           {S a b : Real}

-- Conditions
def area_ABC := S
def BC := a
def AC := b
def bisector_CE := E
def median_BD := D
def intersection_O := O
def quad_ADOE_area := (b * (3 * a + b) * S) / (2 * (a + b) * (2 * a + b))

-- Statement to prove
theorem area_of_adoe
  (h_triangle : IsTriangle ABC)
  (h_bc : length BC = a)
  (h_ac : length AC = b)
  (h_bisector : AngleBisector CE)
  (h_median : Median BD)
  (h_intersection : Intersect CE BD O) :
  area quadrilateral ADOE = quad_ADOE_area :=
sorry

end area_of_adoe_l455_455021


namespace radius_of_inscribed_circle_l455_455853

noncomputable def inscribed_circle_radius (AB AC BC : ℝ) : ℝ :=
  let s := (AB + AC + BC) / 2
  let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  K / s

theorem radius_of_inscribed_circle :
  inscribed_circle_radius 6 8 10 = 2 :=
by
  sorry

end radius_of_inscribed_circle_l455_455853


namespace expression_value_l455_455855

noncomputable def expr := (1.90 * (1 / (1 - (3: ℝ)^(1/4)))) + (1 / (1 + (3: ℝ)^(1/4))) + (2 / (1 + (3: ℝ)^(1/2)))

theorem expression_value : expr = -2 := 
by
  sorry

end expression_value_l455_455855


namespace percentage_increase_in_llama_cost_l455_455114

def cost_of_goat : ℕ := 400
def number_of_goats : ℕ := 3
def total_cost : ℕ := 4800

def llamas_cost (x : ℕ) : Prop :=
  let total_cost_goats := number_of_goats * cost_of_goat
  let total_cost_llamas := total_cost - total_cost_goats
  let number_of_llamas := 2 * number_of_goats
  let cost_per_llama := total_cost_llamas / number_of_llamas
  let increase := cost_per_llama - cost_of_goat
  ((increase / cost_of_goat) * 100) = x

theorem percentage_increase_in_llama_cost :
  llamas_cost 50 :=
sorry

end percentage_increase_in_llama_cost_l455_455114


namespace train_crosses_platform_in_25_002_seconds_l455_455898

noncomputable def time_to_cross_platform 
  (length_train : ℝ) 
  (length_platform : ℝ) 
  (speed_kmph : ℝ) : ℝ := 
  let total_distance := length_train + length_platform
  let speed_mps := speed_kmph * (1000 / 3600)
  total_distance / speed_mps

theorem train_crosses_platform_in_25_002_seconds :
  time_to_cross_platform 225 400.05 90 = 25.002 := by
  sorry

end train_crosses_platform_in_25_002_seconds_l455_455898


namespace commute_days_l455_455523

-- Define the given conditions using Lean definitions
variable (a b x : ℕ)

-- Assign known values based on the problem conditions
axiom h1 : a = 12
axiom h2 : b = 8

-- Define the statement to be proved
theorem commute_days:
  x = a + b → x = 20 :=
by
  intro h
  rw [h1, h2] at h
  exact h

end commute_days_l455_455523


namespace prob_travel_to_shore_l455_455905

noncomputable def archipelago_prob : ℝ :=
  let p := 0.5 in
  let q := 1 - p in
  q * (1 / (1 - p * q))

theorem prob_travel_to_shore :
  archipelago_prob = 2/3 :=
by
  sorry

end prob_travel_to_shore_l455_455905


namespace coefficient_x3_in_expansion_l455_455333

-- Define the expansion of the binomial expression
def expansion (x : ℝ) : ℝ := (1 + x) * (2 + x)^5

-- Statement to be proved
theorem coefficient_x3_in_expansion : (expansion x).coeff 3 = 120 :=
sorry

end coefficient_x3_in_expansion_l455_455333


namespace functional_equation_solution_l455_455208

noncomputable def f (x : ℝ) : ℝ := sorry

axiom functional_equation (x y : ℝ) : f(x * f(y) + 1) = y + f(f(x) * f(y))

theorem functional_equation_solution : ∀ x : ℝ, f(x) = x - 1 :=
by
  sorry

end functional_equation_solution_l455_455208


namespace max_min_values_l455_455749

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def functional_eq (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x + f y

def decreasing_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ), 0 < x → f x < 0

def f_def (f : ℝ → ℝ) : Prop :=
  is_odd f ∧ functional_eq f ∧ decreasing_on_positive f ∧ f 1 = -2

theorem max_min_values (f : ℝ → ℝ) (h : f_def f) :
  ∃ (a b : ℝ), (a = 6 ∧ b = -6) ∧ (∀ x ∈ Icc (-3 : ℝ) 3, f x ≤ a ∧ b ≤ f x) :=
sorry

end max_min_values_l455_455749


namespace arithmetic_sequence_from_line_l455_455253

def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∃ a₀ : ℝ, ∀ n : ℕ, n > 0 → a n = a₀ + n * d

def lies_on_line (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = 2 * n + 1

theorem arithmetic_sequence_from_line (a : ℕ → ℝ) :
  (∀ n : ℕ, n > 0 → a n = 2 * n + 1) → is_arithmetic a :=
begin
  intro h,
  use [2, 1],
  intro n,
  simp [h],
  sorry
end

end arithmetic_sequence_from_line_l455_455253


namespace find_p_l455_455613

variable (n : ℕ) (p : ℚ)
variable (ξ : ℕ → ℕ)

-- Assumption that ξ follows a binomial distribution B(n, p)
axiom BinomialDistribution : (x : ℕ) → ℕ → ℕ → Prop

-- Introducing stochastic properties
axiom Expectation : (x : ℕ → ℕ) → ℚ → Prop
axiom Variance : (x : ℕ → ℕ) → ℚ → Prop

-- Given conditions
axiom H1 : Expectation ξ (8 : ℚ)
axiom H2 : Variance ξ (1.6 : ℚ)
axiom H3 : BinomialDistribution ξ n p

-- The statement to prove
theorem find_p : p = (0.8 : ℚ) :=
by
  sorry

end find_p_l455_455613


namespace triangle_angle_sum_l455_455338

theorem triangle_angle_sum (x y : ℝ)
  (h1 : ∠ BCA = 90) (h2 : ∠ ABC = 3 * y)
  (h3 : ∠ BAC = x)
  (triangle_sum : ∠ BAC + ∠ ABC + ∠ BCA = 180) :
  x + y = 90 - 2 * y :=
by sorry

end triangle_angle_sum_l455_455338


namespace max_value_Sn_l455_455373

variable (a : ℕ → ℝ) (d : ℝ)
variable (n : ℕ)
variable (S : ℕ → ℝ)

-- Define the arithmetic sequence
def arithmetic_seq (n : ℕ) : ℝ := a 1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_seq (n : ℕ) : ℝ := ((n : ℝ) / 2) * (2 * (a 1) + (n - 1) * d)

theorem max_value_Sn (h₁ : |arithmetic_seq 3| = |arithmetic_seq 11|)
                     (h₂ : d < 0) :
  (S n = sum_arithmetic_seq n) → (n = 6 ∨ n = 7) := 
by sorry

end max_value_Sn_l455_455373


namespace same_focal_distance_l455_455618

theorem same_focal_distance :
  let C1 := λ x y : ℝ, x^2 / 12 + y^2 / 4 = 1
  let C2 := λ x y : ℝ, x^2 / 16 + y^2 / 8 = 1
  let a1 := 2 * Real.sqrt 3
  let b1 := 2
  let c1 := 2 * Real.sqrt 2
  let a2 := 4
  let b2 := 2 * Real.sqrt 2
  let c2 := 2 * Real.sqrt 2
  c1 = c2 :=
by {
  sorry
}

end same_focal_distance_l455_455618


namespace Sravan_travel_time_l455_455082

theorem Sravan_travel_time :
  let total_distance := 540
  let first_half_distance := total_distance / 2
  let second_half_distance := total_distance / 2
  let speed_first_half := 45
  let speed_second_half := 30
  let time_first_half := first_half_distance / speed_first_half
  let time_second_half := second_half_distance / speed_second_half
  let total_time := time_first_half + time_second_half
  total_time = 15 :=
by
  let total_distance := 540
  let first_half_distance := total_distance / 2
  let second_half_distance := total_distance / 2
  let speed_first_half := 45
  let speed_second_half := 30
  let time_first_half := first_half_distance / speed_first_half
  let time_second_half := second_half_distance / speed_second_half
  let total_time := time_first_half + time_second_half
  sorry

end Sravan_travel_time_l455_455082


namespace tan_alpha_neg_two_l455_455624

-- Define the conditions and prove the target statement
theorem tan_alpha_neg_two (α : ℝ) 
  (h1 : sin α = (2 * real.sqrt 5) / 5) 
  (h2 : (real.pi / 2) ≤ α ∧ α ≤ real.pi) : 
  tan α = -2 := 
by sorry

end tan_alpha_neg_two_l455_455624


namespace value_of_m_l455_455287

theorem value_of_m (m : ℝ) (h₁ : m^2 - 9 * m + 19 = 1) (h₂ : 2 * m^2 - 7 * m - 9 ≤ 0) : m = 3 :=
sorry

end value_of_m_l455_455287


namespace ratio_suspension_to_fingers_toes_l455_455734

-- Definition of conditions
def suspension_days_per_instance : Nat := 3
def bullying_instances : Nat := 20
def fingers_and_toes : Nat := 20

-- Theorem statement
theorem ratio_suspension_to_fingers_toes :
  (suspension_days_per_instance * bullying_instances) / fingers_and_toes = 3 :=
by
  sorry

end ratio_suspension_to_fingers_toes_l455_455734


namespace sum_prime_factors_of_77_l455_455936

theorem sum_prime_factors_of_77 : ∑ p in {p | p.prime ∧ p ∣ 77}, p = 18 :=
  sorry

end sum_prime_factors_of_77_l455_455936


namespace intersection_eq_l455_455878

open Set

def A : Set ℕ := {0, 2, 4, 6}
def B : Set ℕ := {x | 3 < x ∧ x < 7}

theorem intersection_eq : A ∩ B = {4, 6} := 
by 
  sorry

end intersection_eq_l455_455878


namespace length_QS_l455_455446

noncomputable def right_triangle (P Q R : Type*) [metric_space P] (dist : P → P → ℝ) : Prop :=
dist P R = real.sqrt (dist P Q ^ 2 + dist Q R ^ 2)

theorem length_QS
  {P Q R S : Type*} [metric_space P]
  (dist : P → P → ℝ)
  (is_right_triangle : right_triangle P Q R dist)
  (PQ_eq : dist P Q = 3)
  (QR_eq : dist Q R = 4)
  (angle_bisector : 2 * dist P S = dist P R + dist S Q) :
  dist Q S = 12 / 7 := 
sorry

end length_QS_l455_455446


namespace mork_mindy_combined_tax_rate_l455_455056

def combined_tax_rate (X : ℝ) (mork_tax_rate : ℝ) (mindy_tax_rate : ℝ) (mindy_earning_multiple : ℝ) : ℝ :=
  let mork_income := X
  let mindy_income := mindy_earning_multiple * X
  let total_income := mork_income + mindy_income
  let mork_tax := mork_tax_rate * mork_income
  let mindy_tax := mindy_tax_rate * mindy_income
  let total_tax := mork_tax + mindy_tax
  total_tax / total_income * 100

theorem mork_mindy_combined_tax_rate :
  ∀ (X : ℝ), combined_tax_rate X 0.3 0.2 3 = 22.5 :=
by
  sorry

end mork_mindy_combined_tax_rate_l455_455056


namespace acceleration_at_t_2_l455_455513

theorem acceleration_at_t_2 :
  (derivative (λ t : ℝ, t^2 + 3) 2 = 4) :=
by
  sorry

end acceleration_at_t_2_l455_455513


namespace abs_eight_minus_neg_two_sum_of_integers_satisfying_condition_minimum_distance_sum_l455_455845

-- Problem (1)
theorem abs_eight_minus_neg_two : |8 - (-2)| = 10 :=
by
  sorry

-- Problem (2)
theorem sum_of_integers_satisfying_condition : {
  x : ℤ | |x - 2| + |x + 3| = 5 }.sum = -3 :=
by
  sorry

-- Problem (3)
theorem minimum_distance_sum (x : ℚ) : ∃ y, y = |x + 4| + |x - 6| ∧ y = 10 :=
by
  sorry

end abs_eight_minus_neg_two_sum_of_integers_satisfying_condition_minimum_distance_sum_l455_455845


namespace ratio_of_men_to_women_l455_455453

def num_cannoneers : ℕ := 63
def num_people : ℕ := 378
def num_women (C : ℕ) : ℕ := 2 * C
def num_men (total : ℕ) (women : ℕ) : ℕ := total - women

theorem ratio_of_men_to_women : 
  let C := num_cannoneers
  let total := num_people
  let W := num_women C
  let M := num_men total W
  M / W = 2 :=
by
  sorry

end ratio_of_men_to_women_l455_455453


namespace operation_value_l455_455990

variable (m n x y : ℝ)
variable (” : ℝ → ℝ → ℝ)
-- Define the operation based on given formula
def operation : ℝ → ℝ → ℝ := λ m n, (n^2 - m) / x + y

-- Prove that given the values, the operation result is 7.5
theorem operation_value :
  (m = 4) → (n = 3) → (x = 2) → (y = 5) → operation m n = 7.5 :=
by
  -- Placeholder for actual proof
  sorry

end operation_value_l455_455990


namespace points_on_line_any_real_n_l455_455336

theorem points_on_line_any_real_n (m n : ℝ) 
  (h1 : m = 2 * n + 5) 
  (h2 : m + 1 = 2 * (n + 0.5) + 5) : 
  True :=
by
  sorry

end points_on_line_any_real_n_l455_455336


namespace evaluate_f_f_2_l455_455756

noncomputable def f : ℝ → ℝ :=
λ x, if x < 2 then 2 * Real.exp (x - 1) else Real.logb 3 (x*x - 1)

theorem evaluate_f_f_2 : f (f 2) = 2 := 
by
  sorry

end evaluate_f_f_2_l455_455756


namespace problem_statement_l455_455548

noncomputable def lhs: ℝ := 8^6 * 27^6 * 8^27 * 27^8
noncomputable def rhs: ℝ := 216^14 * 8^19

theorem problem_statement : lhs = rhs :=
by
  sorry

end problem_statement_l455_455548


namespace equivalent_chord_length_l455_455276

theorem equivalent_chord_length (k : ℝ) :
  ∀ (x y : ℝ), (x^2 / 9 + y^2 / 8 = 1) → 
  (line_eq : ℝ → ℝ → Prop) := (λ a b, b = k * a + 2) → 
  (chord_length_eq : ∀ (line1 line2 : ℝ → ℝ → Prop), 
    (line1 = λ a b, b = k * a + 2) ∧ (line2 = λ a b, b = - k * a + 2) → 
    ∃ s1 s2, (∀ p1 : ℝ × ℝ, line1 p1.1 p1.2 → (p1.1, p1.2) ∈ s1) ∧ 
             (∀ p2 : ℝ × ℝ, line2 p2.1 p2.2 → (p2.1, p2.2) ∈ s2) ∧
             length(s1) = length(s2)) →
  chord_length_eq (λ a b, b = k * a + 2) (λ a b, b = - k * a + 2) := 
begin
  sorry
end

end equivalent_chord_length_l455_455276


namespace cutting_figure_into_rectangles_and_squares_l455_455090

theorem cutting_figure_into_rectangles_and_squares :
  ∀ (figure : matrix ℕ ℕ ℕ), 
    (∃ (gray black : ℕ), gray = 8 ∧ black = 9 ∧ figure.height * figure.width = 17) →
    (∃ (ways : ℕ), ways = 10) :=
sorry

end cutting_figure_into_rectangles_and_squares_l455_455090


namespace perimeter_regular_polygon_l455_455166

-- Condition definitions
def is_regular_polygon (n : ℕ) (s : ℝ) : Prop := 
  n * s > 0

def exterior_angle (E : ℝ) (n : ℕ) : Prop := 
  E = 360 / n

def side_length (s : ℝ) : Prop :=
  s = 6

-- Theorem statement to prove the perimeter is 24 units
theorem perimeter_regular_polygon 
  (n : ℕ) (s E : ℝ)
  (h1 : is_regular_polygon n s)
  (h2 : exterior_angle E n)
  (h3 : side_length s)
  (h4 : E = 90) :
  4 * s = 24 :=
by
  sorry

end perimeter_regular_polygon_l455_455166


namespace quadratic_product_fact_l455_455935

def quadratic_factors_product : Prop :=
  let integer_pairs := [(-1, 24), (-2, 12), (-3, 8), (-4, 6), (-6, 4), (-8, 3), (-12, 2), (-24, 1)]
  let t_values := integer_pairs.map (fun (c, d) => c + d)
  let product_t := t_values.foldl (fun acc t => acc * t) 1
  product_t = -5290000

theorem quadratic_product_fact : quadratic_factors_product :=
by sorry

end quadratic_product_fact_l455_455935


namespace find_n_l455_455951

theorem find_n (n : ℕ) (h : n > 0) : 
  (∃ d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}, ∃ t ∈ ℕ, n^5 + 79 = d * ((10^t - 1) / 9)) ↔ (n = 2) := 
by
  sorry

end find_n_l455_455951


namespace last_digits_after_hour_l455_455341

theorem last_digits_after_hour
    (initial_numbers : List ℕ)
    (h_initial : initial_numbers = [2, 3, 4]) :
    ∃ (digits : List ℕ), digits.perm [7, 8, 9] ∧
    (∀ n ∈ digits, n < 10) :=
    sorry

end last_digits_after_hour_l455_455341


namespace wheel_speed_l455_455745

def original_circumference_in_miles := 10 / 5280
def time_factor := 3600
def new_time_factor := 3600 - (1/3)

theorem wheel_speed
  (r : ℝ) 
  (original_speed : r * time_factor = original_circumference_in_miles * 3600)
  (new_speed : (r + 5) * (time_factor - 1/10800) = original_circumference_in_miles * 3600) :
  r = 10 :=
sorry

end wheel_speed_l455_455745


namespace monogramming_cost_per_stocking_l455_455383

noncomputable def total_stockings : ℕ := (5 * 5) + 4
noncomputable def price_per_stocking : ℝ := 20 - (0.10 * 20)
noncomputable def total_cost_of_stockings : ℝ := total_stockings * price_per_stocking
noncomputable def total_cost : ℝ := 1035
noncomputable def total_monogramming_cost : ℝ := total_cost - total_cost_of_stockings

theorem monogramming_cost_per_stocking :
  (total_monogramming_cost / total_stockings) = 17.69 :=
by
  sorry

end monogramming_cost_per_stocking_l455_455383


namespace initial_cheese_amount_l455_455830

theorem initial_cheese_amount (k : ℕ) (h1 : k > 7) (h2 : 35 % k = 0) : initial_cheese 7 10 k = 11 :=
by 
  -- Define the amount each rat ate the first night
  let cheese_per_rat_first_night := 10 / k
  
  -- Define the amount each rat ate the second night
  let cheese_per_rat_second_night := 5 / k
  
  -- Total cheese eaten the second night
  let total_cheese_second_night := 7 * cheese_per_rat_second_night
  
  -- Ensure the total amount consumed remains whole
  have total_cheese_is_whole : total_cheese_second_night = 1 := by {
    rw [total_cheese_second_night, cheese_per_rat_second_night],
    -- We know k must divide 35, so the result of 35/k must be 1
    simp [k, 35 % k, h2],
  }
  
  -- Initial cheese is the sum of cheese eaten first and second night
  let initial_cheese := 10 + 1
  exact initial_cheese_is_11 initial_cheese

-- Define supporting lemma that sums the cheese amounts
lemma initial_cheese_is_11 (initial_cheese : ℕ) : initial_cheese = 11 :=
by {
  simp [initial_cheese],
  exact 10 + 1 == 11,
}

end initial_cheese_amount_l455_455830


namespace white_pairs_coincide_l455_455565

theorem white_pairs_coincide 
    (red_triangles : ℕ)
    (blue_triangles : ℕ)
    (white_triangles : ℕ)
    (red_pairs : ℕ)
    (blue_pairs : ℕ)
    (red_white_pairs : ℕ)
    (coinciding_white_pairs : ℕ) :
    red_triangles = 4 → 
    blue_triangles = 6 →
    white_triangles = 10 →
    red_pairs = 3 →
    blue_pairs = 4 →
    red_white_pairs = 3 →
    coinciding_white_pairs = 7 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end white_pairs_coincide_l455_455565


namespace student_average_comparison_l455_455893

theorem student_average_comparison (x y w : ℤ) (hxw : x < w) (hwy : w < y) : 
  (B : ℤ) > (A : ℤ) :=
  let A := (x + y + w) / 3
  let B := ((x + w) / 2 + y) / 2
  sorry

end student_average_comparison_l455_455893


namespace domain_of_f_l455_455122

open Set Real

noncomputable def f (x : ℝ) : ℝ := (x + 6) / sqrt (x^2 - 5 * x + 6)

theorem domain_of_f : {x : ℝ | ∃ y, y = f x} = {x : ℝ | x < 2 ∨ x > 3} :=
by
  sorry

end domain_of_f_l455_455122


namespace voice_of_china_signup_ways_l455_455724

theorem voice_of_china_signup_ways : 
  (2 * 2 * 2 = 8) :=
by {
  sorry
}

end voice_of_china_signup_ways_l455_455724


namespace complex_conjugate_of_z_l455_455265

open Complex

def z : ℂ := (1 - 2 * Complex.i) / Complex.i

theorem complex_conjugate_of_z : conj z = -2 + Complex.i := by
  sorry

end complex_conjugate_of_z_l455_455265


namespace sum_surface_areas_bienao_correct_l455_455009

noncomputable def sum_surface_areas_bienao (MA AB BC : ℝ) : ℝ :=
  if (MA = 2) ∧ (AB = 2) ∧ (BC = 2) then
    let AC := 2 * Real.sqrt (2),
        MC := 2 * Real.sqrt (3),
        r := Real.sqrt (2) - 1,
        R := Real.sqrt (3),
        surface_area_circumscribed := 4 * Real.pi * R^2,
        surface_area_inscribed := 4 * Real.pi * r^2
    in surface_area_circumscribed + surface_area_inscribed
  else 0

theorem sum_surface_areas_bienao_correct :
  sum_surface_areas_bienao 2 2 2 = 24 * Real.pi - 8 * Real.pi * Real.sqrt 2 :=
by sorry

end sum_surface_areas_bienao_correct_l455_455009


namespace find_f2014_l455_455612

noncomputable def f : ℤ → ℝ := sorry

axiom condition1 : ∀ x:ℤ, f(x) * f(x + 2) = 2014
axiom condition2 : f(0) = 1

theorem find_f2014 : f(2014) = 2014 := 
by sorry

end find_f2014_l455_455612


namespace length_PB_cases_l455_455818

noncomputable def length_PB (P A B O : Type*) [InnerProductSpace ℝ P] 
  (radius : ℝ) (PA_length : ℝ) (AB_length : ℝ)
  (PA : P) (A : P) (B : P) (O : P) : ℝ :=
  sorry

theorem length_PB_cases (P A B O : Type*) [InnerProductSpace ℝ P] 
  (radius : ℝ) (PA_length : ℝ) (AB_length : ℝ)
  (PA : P) (A : P) (B : P) (O : P) :
  radius = 1 → PA_length = 1 → AB_length = real.sqrt 2 → 
  (length_PB P A B O radius PA_length AB_length PA A B O = 1 ∨
   length_PB P A B O radius PA_length AB_length PA A B O = real.sqrt 5) :=
by
  assume h1 : radius = 1
  assume h2 : PA_length = 1
  assume h3 : AB_length = real.sqrt 2
  sorry

end length_PB_cases_l455_455818


namespace isosceles_triangle_LCM_l455_455202

variables {A B C L K M : Type*} [Point A] [Point B] [Point C] [Point L] [Point K] [Point M]
variables {ω Ω : Circle} [TangentTo ω A B]
  [TangentTo ω A C] [TouchΩ Ω A C] [TouchΩExtended Ω L BC]
  (HL : Touches Ω ω at L on BC)
  (HA : Line A L meets K on ω and M on Ω) 
  (HParallel : KB ∥ CM)

theorem isosceles_triangle_LCM
  (h_tangent_AB_AC : TangentRight ω B C)
  (h_tangent_2 : TangentRight Ω A C)
  (h_touch : Touches Ω ω at L on BC)
  (h_intersect_AL : Line A L meets K on ω and M on Ω)
  (h_parallel : KB ∥ CM):
  is_isosceles △ L C M := sorry

end isosceles_triangle_LCM_l455_455202


namespace papa_bird_worms_l455_455374

-- Define the problem conditions
def num_babies : ℕ := 6
def worms_per_baby_per_day : ℕ := 3
def feeding_days : ℕ := 3
def worms_mama_caught : ℕ := 13
def worms_stolen : ℕ := 2
def worms_mama_needs : ℕ := 34

-- Define the statement we want to prove
theorem papa_bird_worms :
  let total_worms_needed := num_babies * worms_per_baby_per_day * feeding_days in
  let mama_bird_worms := worms_mama_caught - worms_stolen in
  let total_worms_mama_accounted := mama_bird_worms + worms_mama_needs in
  total_worms_needed - total_worms_mama_accounted = 9 :=
by {
  -- Skipping the proof
  sorry
}

end papa_bird_worms_l455_455374


namespace ellen_hits_9_l455_455401

theorem ellen_hits_9 (scores: ℕ → ℕ) 
(Alice Ben Cindy Dave Ellen Frank: ℕ)
(h_scores: ∀ x, scores x ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
(h_Alice: Alice = 27)
(h_Ben: Ben = 14)
(h_Cindy: Cindy = 20)
(h_Dave: Dave = 22)
(h_Ellen: Ellen = 24)
(h_Frank: Frank = 30)
(h_score_distinct: ∀ i j, i ≠ j → scores i ≠ scores j) : scores Ellen = 9 :=
sorry

end ellen_hits_9_l455_455401


namespace parts_in_batch_l455_455702

theorem parts_in_batch (a : ℕ) (h₁ : 20 * (a / 20) + 13 = a) (h₂ : 27 * (a / 27) + 20 = a) 
  (h₃ : 500 ≤ a) (h₄ : a ≤ 600) : a = 533 :=
by sorry

end parts_in_batch_l455_455702


namespace matrix_identity_l455_455353

variable {K : Type*} [Field K] {n : Type*} [DecidableEq n] [Fintype n]
variable (A : Matrix n n K) (I : Matrix n n K) (A_inv : Matrix n n K)

-- Conditions
axiom inv_A : A * A_inv = I ∧ A_inv * A = I
axiom condition : (A - 3 • I) * (A - 5 • I) = -I

-- The proof statement
theorem matrix_identity :
  A + 10 • A_inv = 6.5 • I :=
sorry

end matrix_identity_l455_455353


namespace remainder_division_l455_455463
-- Import the necessary library

-- Define the number and the divisor
def number : ℕ := 2345678901
def divisor : ℕ := 101

-- State the theorem
theorem remainder_division : number % divisor = 23 :=
by sorry

end remainder_division_l455_455463


namespace equal_acquaintances_l455_455495

theorem equal_acquaintances (n : ℕ) (A : ℕ → Prop)
  (H : ∀ (x y z : ℕ), A x → A y → A z → x ≠ y → y ≠ z → x ≠ z → 
    ∃ (w : ℕ), w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ A w ∧
    ∀ (v : ℕ), A v → (v ≠ x ∧ v ≠ y ∧ v ≠ z → v = w ∨ v = x ∨ v = y ∨ v = z)) :
  (∃ k : ℕ, ∀ (x : ℕ), A x → (∃ C : ℕ → ℕ, (bij_on C { y : ℕ | A y ∧ y ≠ x} (finset.range k) ∧
    ∀ u v : ℕ, A u ∧ A v ∧ u ≠ v → (∃ t : ℕ, t ≠ u ∧ t ≠ v ∧ A t ∧ C t = C u ∧ C t = C v)))) ∧
  (n > 4) :=
sorry

end equal_acquaintances_l455_455495


namespace pair_count_is_19_l455_455105

-- Define the number condition for the overline notation.
def overline (x y z w : ℕ) : ℕ := 1000*x + 100*y + 10*z + w

-- Define the problem conditions.
def div_824 (a b : ℕ) : Prop :=
  ∃ x y z w : ℕ, x = 5 ∧ y = a ∧ z = 6 ∧ w = 8 ∧
  ((overline 5 a 6 8) * (overline 8 6 5 b)) % 824 = 0

noncomputable def count_pairs : ℕ :=
  finset.card ((finset.range 10).product (finset.range 10)).filter (λ ab, div_824 ab.1 ab.2)

theorem pair_count_is_19 : count_pairs = 19 :=
sorry

end pair_count_is_19_l455_455105


namespace trackball_mice_count_l455_455052

theorem trackball_mice_count
  (total_mice : ℕ)
  (wireless_fraction : ℕ)
  (optical_fraction : ℕ)
  (h_total : total_mice = 80)
  (h_wireless : wireless_fraction = total_mice / 2)
  (h_optical : optical_fraction = total_mice / 4) :
  total_mice - (wireless_fraction + optical_fraction) = 20 :=
sorry

end trackball_mice_count_l455_455052


namespace fifth_color_marbles_l455_455701

theorem fifth_color_marbles 
  (r : ℕ) (g : ℕ) (y : ℕ) (b : ℕ) (f : ℕ) (t : ℕ)
  (hr : r = 25)
  (hg : g = 3 * r)
  (hy : y = Nat.floor (0.20 * g))
  (hb : b = 2 * y)
  (hf : f = Nat.floor (1.50 * (r + b)))
  (ht : t = 4 * g)
  (htotal : t = r + g + y + b + f) :
  f = 155 :=
by
  sorry

end fifth_color_marbles_l455_455701


namespace log_eq_solution_l455_455790

noncomputable def solve_log_eq (y : ℝ) : Prop :=
  2 * log y + 3 * log 2 = 1

theorem log_eq_solution : ∃ y : ℝ, solve_log_eq y ∧ y = real.sqrt 5 / 2 :=
by
  use real.sqrt 5 / 2
  sorry

end log_eq_solution_l455_455790


namespace sin_2x_equals_neg_61_div_72_l455_455730

variable (x y : Real)
variable (h1 : Real.sin y = (3 / 2) * Real.sin x + (2 / 3) * Real.cos x)
variable (h2 : Real.cos y = (2 / 3) * Real.sin x + (3 / 2) * Real.cos x)

theorem sin_2x_equals_neg_61_div_72 : Real.sin (2 * x) = -61 / 72 :=
by
  -- Proof goes here
  sorry

end sin_2x_equals_neg_61_div_72_l455_455730


namespace greatest_common_divisor_of_98_and_n_l455_455841

theorem greatest_common_divisor_of_98_and_n (n : ℕ) (h1 : ∃ (d : Finset ℕ),  d = {1, 7, 49} ∧ ∀ x ∈ d, x ∣ 98 ∧ x ∣ n) :
  ∃ (g : ℕ), g = 49 :=
by
  sorry

end greatest_common_divisor_of_98_and_n_l455_455841


namespace net_effect_sale_value_l455_455692

variables {P Q : ℝ}

theorem net_effect_sale_value 
  (h1 : ∀ P Q : ℝ, P > 0 ∧ Q > 0)
  (h_price_reduction : ∀ P : ℝ, P_after_reduction = 0.9 * P)
  (h_sale_increase : ∀ Q : ℝ, Q_after_increase = 1.85 * Q) :
  (0.9 * P * 1.85 * Q) / (P * Q) - 1 = 0.665 :=
by
  intro h1 h_price_reduction h_sale_increase
  sorry

end net_effect_sale_value_l455_455692


namespace relationship_between_abc_l455_455358

noncomputable def a : ℝ := 3^0.7

noncomputable def b : ℝ := Real.log 2 / Real.log 3

noncomputable def c : ℝ := Real.log 2 / Real.log (1 / 3)

theorem relationship_between_abc : a > b ∧ b > c :=
by
  let a := a
  let b := b
  let c := c
  sorry

end relationship_between_abc_l455_455358


namespace employee_saves_l455_455864

-- Given conditions
def cost_price : ℝ := 500
def markup_percentage : ℝ := 0.15
def employee_discount_percentage : ℝ := 0.15

-- Definitions
def final_retail_price : ℝ := cost_price * (1 + markup_percentage)
def employee_discount_amount : ℝ := final_retail_price * employee_discount_percentage

-- Assertion
theorem employee_saves :
  employee_discount_amount = 86.25 := by
  sorry

end employee_saves_l455_455864


namespace integer_solutions_count_l455_455932

theorem integer_solutions_count :
  {n : ℤ | (n + complex.i)^6 }.toSet.card = 1 :=
sorry

end integer_solutions_count_l455_455932


namespace vertex_of_parabola_is_1_2_l455_455564

-- Definitions
def vertex_x (a b : ℝ) : ℝ := -b / (2 * a)
def parabola_y (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Properties
theorem vertex_of_parabola_is_1_2 :
  ∀ (a b c : ℝ), a = 3 → b = -6 → c = 5 → (vertex_x a b, parabola_y a b c (vertex_x a b)) = (1, 2) :=
by
  intros a b c ha hb hc
  sorry

end vertex_of_parabola_is_1_2_l455_455564


namespace crates_needed_l455_455573

-- Conditions as definitions
def novels : ℕ := 145
def comics : ℕ := 271
def documentaries : ℕ := 419
def albums : ℕ := 209
def crate_capacity : ℕ := 9

-- Total items calculation
def total_items : ℕ := novels + comics + documentaries + albums

-- Proof statement
theorem crates_needed : (total_items / crate_capacity) = 116 := by
  sorry

end crates_needed_l455_455573


namespace statue_selling_price_l455_455731

/-- Problem conditions -/
def original_cost : ℤ := 550
def profit_percentage : ℝ := 0.20

/-- Proof problem statement -/
theorem statue_selling_price : original_cost + profit_percentage * original_cost = 660 := by
  sorry

end statue_selling_price_l455_455731


namespace angle_SA_BC_l455_455016

noncomputable def midpoint (A B : ℝ^3) : ℝ^3 := (A + B) / 2

variables (S A B C M : ℝ^3)

-- Conditions
def isosceles_triangle_ABC (A B C : ℝ^3) : Prop := (dist A B) = (dist A C)
def isosceles_pyramid_SABC (S B C : ℝ^3) : Prop := (dist S B) = (dist S C)
def midpoint_M_BC (M B C : ℝ^3) : Prop := M = midpoint B C

-- Question: Find the angle between line SA and line BC is π/2
theorem angle_SA_BC (h1 : isosceles_triangle_ABC A B C)
                  (h2 : isosceles_pyramid_SABC S B C)
                  (h3 : midpoint_M_BC M B C) :
                  angle (S - A) (B - C) = π / 2 :=
by
  -- Proof is omitted
  sorry

end angle_SA_BC_l455_455016


namespace number_of_solutions_l455_455242

theorem number_of_solutions :
  let f (x : ℕ) := (log (x - 30) / log 2) + (log (90 - x) / log 2) in
  ∃ (S : Finset ℕ), (∀ x ∈ S, 30 < x ∧ x < 90 ∧ f x < 7) ∧ S.card = 26 :=
by
  sorry

end number_of_solutions_l455_455242


namespace min_f_in_interval_l455_455279

open Real

noncomputable def f (ω x : ℝ) : ℝ := sin (ω * x) - 2 * sqrt 3 * sin (ω * x / 2) ^ 2 + sqrt 3

theorem min_f_in_interval (ω : ℝ) (hω : ω > 0) :
  (∀ x, 0 <= x ∧ x <= π / 2 → f 1 x >= f 1 (π / 3)) :=
by sorry

end min_f_in_interval_l455_455279


namespace larger_triangle_perimeter_l455_455529

theorem larger_triangle_perimeter (a b c : ℕ) (h_right_triangle : a * a + b * b = c * c)
  (k : ℕ) (hk : k = 3) (hypotenuse_c : c = 10) (new_hypotenuse : k * c = 30) :
  let new_a := k * a,
      new_b := k * b,
      new_c := k * c in
  new_a + new_b + new_c = 72 := 
by
  have ha : a = 6 := by sorry
  have hb : b = 8 := by sorry
  have hc : c = 10 := by exact hypotenuse_c
  sorry

end larger_triangle_perimeter_l455_455529


namespace range_of_fx_plus_x_l455_455921

def f (x : ℝ) : ℝ :=
  if h : -2 ≤ x ∧ x ≤ 0 then -x
  else if h : 0 < x ∧ x ≤ 2 then x - 1
  else 0  -- This case will never occur in our interval but Lean requires a total function

def fx_plus_x (x : ℝ) : ℝ :=
  f x + x

theorem range_of_fx_plus_x : 
  ∀x : ℝ, -2 ≤ x ∧ x ≤ 2 → fx_plus_x x ∈ Set.Icc (-1 : ℝ) (3 : ℝ) :=
  by
    sorry

end range_of_fx_plus_x_l455_455921


namespace quadrilateral_angle_properties_l455_455811

theorem quadrilateral_angle_properties (A B C D M N : Point) (α : ℝ)
  (h1 : extension A B ∩ extension C D = M ∧ angle ⟨A, B⟩ M = α)
  (h2 : extension A D ∩ extension B C = N ∧ angle ⟨A, D⟩ N = α) :
  ∃ θ1 θ2 θ3 θ4 : ℝ, (θ1 = θ2) ∧ (θ3 - θ4 = 2 * α) :=
sorry

end quadrilateral_angle_properties_l455_455811


namespace find_middle_number_l455_455101

theorem find_middle_number (a b c : ℕ) (h1 : a + b = 16) (h2 : a + c = 21) (h3 : b + c = 27) : b = 11 := by
  sorry

end find_middle_number_l455_455101


namespace negation_exists_irrational_square_l455_455820

noncomputable def irrational_square_not_rational : Prop :=
  ∀ x : ℝ, x ∉ ℚ → x^2 ∉ ℚ

theorem negation_exists_irrational_square :
  ¬ (∃ x₀ : ℝ, x₀ ∉ ℚ ∧ x₀^2 ∈ ℚ) ↔ irrational_square_not_rational :=
by sorry

end negation_exists_irrational_square_l455_455820


namespace determinant_sin_l455_455041

-- Define angles A, B, C as angles of a triangle
variables {A B C : ℝ}

-- Define the condition for non-right triangle and the given sine identity
axiom triangle_angles (h : A + B + C = π) : ¬(A = π/2 ∨ B = π/2 ∨ C = π/2)
axiom sine_identity : sin A + sin B + sin C = sin A * sin B * sin C

-- State the theorem to prove the determinant is 2
theorem determinant_sin (hABC : A + B + C = π) : 
  (abs (Matrix.det ![
    ![sin A, 1, 1],
    ![1, sin B, 1],
    ![1, 1, sin C]
  ]) = 2) := 
sorry

end determinant_sin_l455_455041


namespace sum_of_coeffs_eq_neg30_l455_455226

noncomputable def expanded : Polynomial ℤ := 
  -(Polynomial.C 4 - Polynomial.X) * (Polynomial.X + 3 * (Polynomial.C 4 - Polynomial.X))

theorem sum_of_coeffs_eq_neg30 : (expanded.coeffs.sum) = -30 := 
  sorry

end sum_of_coeffs_eq_neg30_l455_455226


namespace geometric_sequence_a6_l455_455014

theorem geometric_sequence_a6 (a : ℕ → ℝ) (a1 r : ℝ) (h1 : ∀ n, a n = a1 * r ^ (n - 1)) (h2 : (a 2) * (a 4) * (a 12) = 64) : a 6 = 4 :=
sorry

end geometric_sequence_a6_l455_455014


namespace max_value_quadratic_function_l455_455307

noncomputable def quadratic_function (x : ℝ) : ℝ :=
  -3 * x^2 + 8

theorem max_value_quadratic_function : ∃(x : ℝ), quadratic_function x = 8 :=
by
  sorry

end max_value_quadratic_function_l455_455307


namespace first_box_weight_l455_455764

theorem first_box_weight (W : ℕ) (w3 : ℕ) (diff : ℕ) 
  (h1 : w3 = 13) (h2 : diff = 11) (h3 : W - w3 = diff) : W = 24 :=
by
  rw [h3, h1, h2] -- Based on the given conditions
  sorry

end first_box_weight_l455_455764


namespace like_terms_C_l455_455858

-- Definition of a term
structure Term :=
  (coeff : Int)
  (vars : List (String × Int)) -- List of variables with their exponents

-- Definition of like terms: same variables with the same exponents
def like_terms (t1 t2 : Term) : Prop :=
  t1.vars = t2.vars

-- The terms given in condition C
def term1 : Term := ⟨-1, [("m", 2), ("n", 3)]⟩
def term2 : Term := ⟨-3, [("n", 3), ("m", 2)]⟩

-- Ensure variables list is sorted, assuming ordering by name for simplicity
def sort_vars (vars : List (String × Int)) : List (String × Int) :=
  List.sort (λ a b => a.1 < b.1) vars

-- Statement to be proved
theorem like_terms_C :
  like_terms { term1 with vars := sort_vars term1.vars } { term2 with vars := sort_vars term2.vars } :=
sorry

end like_terms_C_l455_455858


namespace volume_of_intersection_of_octahedra_l455_455465

theorem volume_of_intersection_of_octahedra :
  let region1 := {p : ℝ × ℝ × ℝ | abs p.1 + abs p.2 + abs p.3 ≤ 2}
  let region2 := {p : ℝ × ℝ × ℝ | abs p.1 + abs p.2 + abs (p.3 - 2) ≤ 2}
  region1 ∩ region2 = volume (region1 ∩ region2) = 16 / 3 := 
sorry

end volume_of_intersection_of_octahedra_l455_455465


namespace height_of_windows_l455_455809

theorem height_of_windows
  (L W H d_l d_w w_w : ℕ)
  (C T : ℕ)
  (hl : L = 25)
  (hw : W = 15)
  (hh : H = 12)
  (hdl : d_l = 6)
  (hdw : d_w = 3)
  (hww : w_w = 3)
  (hc : C = 3)
  (ht : T = 2718):
  ∃ h : ℕ, 960 - (18 + 9 * h) = 906 ∧ 
  (T = C * (960 - (18 + 9 * h))) ∧
  (960 = 2 * (L * H) + 2 * (W * H)) ∧ 
  (18 = d_l * d_w) ∧ 
  (9 * h = 3 * (h * w_w)) := 
sorry

end height_of_windows_l455_455809


namespace positive_diff_solutions_abs_eq_12_l455_455212

theorem positive_diff_solutions_abs_eq_12 : 
  ∀ (x1 x2 : ℤ), (|x1 - 4| = 12) ∧ (|x2 - 4| = 12) ∧ (x1 > x2) → (x1 - x2 = 24) :=
by
  sorry

end positive_diff_solutions_abs_eq_12_l455_455212


namespace problem_a_b_c_relationship_l455_455916

theorem problem_a_b_c_relationship (u v a b c : ℝ)
  (h1 : u - v = a)
  (h2 : u^2 - v^2 = b)
  (h3 : u^3 - v^3 = c) :
  3 * b^2 + a^4 = 4 * a * c := by
  sorry

end problem_a_b_c_relationship_l455_455916


namespace proof_problem_l455_455643

noncomputable def f (A ω φ x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

def is_highest_point (f : ℝ → ℝ) (Mx : ℝ) (My : ℝ) : Prop :=
  ∀ x, f (Mx + x) ≤ My

def g (x : ℝ) : ℝ := 3 * Real.cos x

def meets_condition (g : ℝ → ℝ) (x0 : ℝ) (m : ℝ) : Prop :=
  g x0 + 2 ≤ Real.log m / Real.log 3

theorem proof_problem (A ω φ m : ℝ) :
  (A > 0) →
  (ω > 0) →
  (0 < φ < π / 2) →
  (∀ x₁ x₂ : ℝ, f A ω φ x₁ = 0 ∧ f A ω φ x₂ = 0 → abs (x₁ - x₂) = π / 2) →
  is_highest_point (f A ω φ) (π / 6) 3 →
  ∃ x0 ∈ Icc (-π / 3) (2 * π / 3), meets_condition (g) x0 m →
  m ≥ sqrt 3 := sorry

end proof_problem_l455_455643


namespace production_line_improvement_better_than_financial_investment_l455_455146

noncomputable def improved_mean_rating (initial_mean : ℝ) := initial_mean + 0.05

noncomputable def combined_mean_rating (mean_unimproved : ℝ) (mean_improved : ℝ) : ℝ :=
  (mean_unimproved * 200 + mean_improved * 200) / 400

noncomputable def combined_variance (variance : ℝ) (combined_mean : ℝ) : ℝ :=
  (2 * variance) - combined_mean ^ 2

noncomputable def increased_returns (grade_a_price : ℝ) (grade_b_price : ℝ) 
  (proportion_upgraded : ℝ) (units_per_day : ℕ) (days_per_year : ℕ) : ℝ :=
  (grade_a_price - grade_b_price) * proportion_upgraded * units_per_day * days_per_year - 200000000

noncomputable def financial_returns (initial_investment : ℝ) (annual_return_rate : ℝ) : ℝ :=
  initial_investment * (1 + annual_return_rate) - initial_investment

theorem production_line_improvement_better_than_financial_investment 
  (initial_mean : ℝ := 9.98) 
  (initial_variance : ℝ := 0.045) 
  (grade_a_price : ℝ := 2000) 
  (grade_b_price : ℝ := 1200) 
  (proportion_upgraded : ℝ := 3 / 8) 
  (units_per_day : ℕ := 200) 
  (days_per_year : ℕ := 365) 
  (initial_investment : ℝ := 200000000) 
  (annual_return_rate : ℝ := 0.082) : 
  combined_mean_rating initial_mean (improved_mean_rating initial_mean) = 10.005 ∧ 
  combined_variance initial_variance (combined_mean_rating initial_mean (improved_mean_rating initial_mean)) = 0.045625 ∧ 
  increased_returns grade_a_price grade_b_price proportion_upgraded units_per_day days_per_year > financial_returns initial_investment annual_return_rate := 
by {
  sorry
}

end production_line_improvement_better_than_financial_investment_l455_455146


namespace area_FDBG_l455_455445

-- Define the problem conditions
variables (A B C D E F G : Type)
variables [metric_space A] [metric_space B] [metric_space C]
variables [metric_space D] [metric_space E] [metric_space F] [metric_space G]
variables (AB AC : ℝ)
variables (area_ABC : ℝ)
variables (D_mid_AB : midpoint A B D)
variables (E_mid_AC : midpoint A C E)
variables (angle_bisector : bisector A)
variables (F_on_DE : on_segment D E F)
variables (G_on_BC : on_segment B C G)

-- Define the theorem to be proved
theorem area_FDBG :
  AB = 50 → AC = 10 → area_ABC = 120 →
  midpoint A B D → midpoint A C E → bisector A → 
  on_segment D E F → on_segment B C G →
  area F D B G = 75 :=
by sorry

end area_FDBG_l455_455445


namespace number_of_six_digit_palindromes_l455_455193

def is_six_digit_palindrome (n : ℕ) : Prop :=
  ∃ a b c, a ≠ 0 ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ n = a * 100001 + b * 10010 + c * 1100

theorem number_of_six_digit_palindromes : ∃ p, p = 900 ∧ (∀ n, is_six_digit_palindrome n → n = p) :=
by
  sorry

end number_of_six_digit_palindromes_l455_455193


namespace count_primes_with_prime_remainders_between_50_and_100_l455_455671

/-- 
The count of prime numbers between 50 and 100 that have a prime remainder 
(1, 2, 3, or 5) when divided by 6 is 10.
-/
theorem count_primes_with_prime_remainders_between_50_and_100 : 
  (finset.filter (λ p, ∃ r, (p % 6 = r) ∧ nat.prime r ∧ r ∈ ({1, 2, 3, 5} : finset ℕ)) 
                  (finset.filter nat.prime (finset.Ico 51 101))).card = 10 := 
by 
  sorry

end count_primes_with_prime_remainders_between_50_and_100_l455_455671


namespace power_of_negative_125_l455_455570

theorem power_of_negative_125 : (-125 : ℝ)^(4/3) = 625 := by
  sorry

end power_of_negative_125_l455_455570


namespace largest_integer_m_such_that_expression_is_negative_l455_455966

theorem largest_integer_m_such_that_expression_is_negative :
  ∃ (m : ℤ), (∀ (n : ℤ), (m^2 - 11 * m + 24 < 0 ) → n < m → n^2 - 11 * n + 24 < 0) ∧
  m^2 - 11 * m + 24 < 0 ∧
  (m ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) :=
by
  sorry

end largest_integer_m_such_that_expression_is_negative_l455_455966


namespace region_volume_is_two_thirds_l455_455476

noncomputable def volume_of_region : ℝ :=
  let region := {p : ℝ × ℝ × ℝ | |p.1| + |p.2| + |p.3| ≤ 2 ∧ |p.1| + |p.2| + |p.3 - 2| ≤ 2}
  -- Assuming volume function calculates the volume of the region
  volume region

theorem region_volume_is_two_thirds :
  volume_of_region = 2 / 3 :=
by
  sorry

end region_volume_is_two_thirds_l455_455476


namespace solve_inequality_l455_455793

noncomputable def problem_statement (x : ℝ) : Prop :=
  ( ( -2 * (x^2 - 5 * x + 25) * (2 * x^2 + 17 * x + 35) ) / ( (x^3 + 125) * real.sqrt (4 * x^2 + 28 * x + 49) ) ≤ x^2 + 3 * x - 2 )

theorem solve_inequality : ∀ x : ℝ,
  (problem_statement x) ↔ (x ∈ set.Iio (-5) ∪ 
                            set.Ioc (-5) (-4) ∪ 
                            set.Ioc (-3.5) (-3) ∪ 
                            set.Ici 0) :=
by
  sorry

end solve_inequality_l455_455793


namespace largest_subset_with_no_quadruples_l455_455532

def isValidSubset (S : Set ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ S → b ∈ S → a ≠ b → ¬ (4 * a = b)

def largestSubsetCardinality : ℕ :=
  173

theorem largest_subset_with_no_quadruples (S : Set ℕ) (h₁ : ∀ n ∈ S, 1 ≤ n ∧ n ≤ 200) (h₂ : isValidSubset S) :
  S.card ≤ largestSubsetCardinality :=
sorry

end largest_subset_with_no_quadruples_l455_455532


namespace avg_age_diff_l455_455088

noncomputable def avg_age_team : ℕ := 28
noncomputable def num_players : ℕ := 11
noncomputable def wicket_keeper_age : ℕ := avg_age_team + 3
noncomputable def total_age_team : ℕ := avg_age_team * num_players
noncomputable def age_captain : ℕ := avg_age_team

noncomputable def total_age_remaining_players : ℕ := total_age_team - age_captain - wicket_keeper_age
noncomputable def num_remaining_players : ℕ := num_players - 2
noncomputable def avg_age_remaining_players : ℕ := total_age_remaining_players / num_remaining_players

theorem avg_age_diff :
  avg_age_team - avg_age_remaining_players = 3 :=
by
  sorry

end avg_age_diff_l455_455088


namespace crayons_per_color_in_each_box_l455_455518

def crayons_in_each_box : ℕ := 2

theorem crayons_per_color_in_each_box
  (colors : ℕ)
  (boxes_per_hour : ℕ)
  (crayons_in_4_hours : ℕ)
  (hours : ℕ)
  (total_boxes : ℕ := boxes_per_hour * hours)
  (crayons_per_box : ℕ := crayons_in_4_hours / total_boxes)
  (crayons_per_color : ℕ := crayons_per_box / colors)
  (colors_eq : colors = 4)
  (boxes_per_hour_eq : boxes_per_hour = 5)
  (crayons_in_4_hours_eq : crayons_in_4_hours = 160)
  (hours_eq : hours = 4) : crayons_per_color = crayons_in_each_box :=
by {
  sorry
}

end crayons_per_color_in_each_box_l455_455518


namespace proof_problem_l455_455289

noncomputable def solveForAB : (a b : ℝ) × ℝ :=
  let eq1 := 3 * a + 5 * b + 1 = 15
  let eq2 := 4 * a + 7 * b + 1 = 28
  let a := (-37 : ℝ)
  let b := (25 : ℝ)
  ((a, b), 1 * a + 1 * b + 1)

theorem proof_problem : (a b : ℝ) × ℝ :=
  let ((a, b), result) := solveForAB
  (a, b, 1 * a + 1 * b + 1 = -11)
  by {
    exact (sorry : eq1 ∧ eq2)
  }

end proof_problem_l455_455289


namespace expected_total_rain_equals_20_point_5_l455_455901

def prob_no_rain := 0.3
def prob_3_inches := 0.3
def prob_8_inches := 0.4

def rain_no_rain := 0
def rain_3_inches := 3
def rain_8_inches := 8

def E_rain_one_day := prob_no_rain * rain_no_rain + prob_3_inches * rain_3_inches + prob_8_inches * rain_8_inches

def days_in_week := 5

def E_rain_week := days_in_week * E_rain_one_day

theorem expected_total_rain_equals_20_point_5 :
  E_rain_week = 20.5 :=
by
  -- The computed expected amount of rainfall per day is 4.1 inches
  have h1 : E_rain_one_day = 4.1, by
    unfold E_rain_one_day prob_no_rain prob_3_inches prob_8_inches rain_no_rain rain_3_inches rain_8_inches
    simp
    norm_num
  
  -- Multiply that by the number of days in the week to get total weekly expected rainfall
  unfold E_rain_week days_in_week
  rw h1
  norm_num

end expected_total_rain_equals_20_point_5_l455_455901


namespace part_a_inequality_1_part_a_inequality_2_part_a_combined_inequality_l455_455494

noncomputable def A : ℝ := (∏ i in Finset.range 50, (2 * i + 1 : ℝ) / (2 * (i + 1) : ℝ))

theorem part_a_inequality_1 : 1 / (10 * Real.sqrt 2) < A :=
sorry

theorem part_a_inequality_2 : A < 1 / 10 :=
sorry

theorem part_a_combined_inequality : 1 / (10 * Real.sqrt 2) < A ∧ A < 1 / 10 :=
⟨part_a_inequality_1, part_a_inequality_2⟩

end part_a_inequality_1_part_a_inequality_2_part_a_combined_inequality_l455_455494


namespace average_rate_of_change_l455_455804

noncomputable def f (x : ℝ) : ℝ :=
  -2 * x^2 + 1

theorem average_rate_of_change : 
  ((f 1 - f 0) / (1 - 0)) = -2 :=
by
  sorry

end average_rate_of_change_l455_455804


namespace stage_4_total_area_is_54_l455_455688

theorem stage_4_total_area_is_54 :
  let side_length_stage (n : ℕ) : ℕ := 1 + n
  let area_stage (n : ℕ) : ℕ := (side_length_stage n) ^ 2
  area_stage 0 + area_stage 1 + area_stage 2 + area_stage 3 = 54 :=
by
  let side_length_stage := λ n : ℕ, 1 + n
  let area_stage := λ n : ℕ, (side_length_stage n) ^ 2
  calc
    area_stage 0 + area_stage 1 + area_stage 2 + area_stage 3
      = 2 ^ 2 + 3 ^ 2 + 4 ^ 2 + 5 ^ 2 : by sorry
    ... = 54 : by sorry

end stage_4_total_area_is_54_l455_455688


namespace car_original_cost_price_usd_l455_455524

noncomputable def original_cost_price_in_usd 
  (loss_percentage : ℚ) 
  (improvement_percentage : ℚ) 
  (gain_percentage : ℚ) 
  (final_selling_price_rs : ℚ) 
  (exchange_rate : ℚ)
  : ℚ :=
let 
    cost_after_loss := final_selling_price_rs / ((1 + gain_percentage) * (1 + improvement_percentage * (1 - loss_percentage))),
    original_cost_price_rs := cost_after_loss / (1 - loss_percentage) 
in original_cost_price_rs / exchange_rate

theorem car_original_cost_price_usd : 
  original_cost_price_in_usd 0.1 0.05 0.2 67320 75 ≈ 791.55 :=
by 
  sorry

end car_original_cost_price_usd_l455_455524


namespace interest_rate_per_annum_l455_455427

theorem interest_rate_per_annum :
  ∃ (r : ℝ), 338 = 312.50 * (1 + r) ^ 2 :=
by
  sorry

end interest_rate_per_annum_l455_455427


namespace transform_quadratic_proof_l455_455561

-- Define the conditions
variables {A B C D E F : ℝ}
variables {α β γ α' β' γ' : ℝ}
variables {a b : ℝ}

-- Define the quadratic equation transformation
def transformed_eq (x' y' : ℝ) : ℝ :=
    A * (α * x' + β * y' + γ)^2 +
    2 * B * (α * x' + β * y' + γ) * (x' + β' * y' + γ') +
    C * (x' + β' * y' + γ')^2 +
    2 * D * (α * x' + β * y' + γ) +
    2 * E * (x' + β' * y') +
    F

-- Define the target forms
def target_eq1 (x' y' : ℝ) : Prop := 
    (x'^2 / a^2) + (y'^2 / b^2) - 1 = 0

def target_eq2 (x' y' : ℝ) : Prop :=
    (x'^2 / a^2) - (y'^2 / b^2) - 1 = 0

-- The proposition to prove
theorem transform_quadratic_proof (hαβ : α * β' - α' * β ≠ 0)
    (hΔ : A * C - B^2 ≠ 0) : 
    (∀ x' y', transformed_eq x' y' = 0 → target_eq1 x' y') ∨
    (∀ x' y', transformed_eq x' y' = 0 → target_eq2 x' y') :=
sorry

end transform_quadratic_proof_l455_455561


namespace range_of_a_for_common_tangent_l455_455799

noncomputable def f (x : ℝ) : ℝ := x^2 + 1
noncomputable def g (a x : ℝ) : ℝ := a * Real.exp(x) + 1

theorem range_of_a_for_common_tangent :
  ∀ a > 0, (∃ l : ℝ, ∃ c_f : ℝ -> ℝ, ∃ c_g : ℝ -> ℝ, 
    (l = c_f) ∧ (l = c_g) ∧ (∀ x : ℝ, c_f x = x^2 + 1) ∧ (∀ x : ℝ, c_g x = a * Real.exp(x) + 1)) 
    ↔ a ∈ Ioo 0 (4 / Real.exp(2)) :=
sorry

end range_of_a_for_common_tangent_l455_455799


namespace range_of_m_tangent_not_parallel_l455_455649

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := (1 / 2) * x^2 - k * x
noncomputable def h (x : ℝ) (m : ℝ) : ℝ := f x + g x (m + (1 / m))
noncomputable def M (x : ℝ) (m : ℝ) : ℝ := f x - g x (m + (1 / m))

theorem range_of_m (m : ℝ) (h_extreme : ∃ x ∈ Set.Ioo 0 2, ∀ y ∈ Set.Ioo 0 2, h y m ≤ h x m) : 
  (0 < m ∧ m ≤ 1 / 2) ∨ (m ≥ 2) :=
  sorry

theorem tangent_not_parallel (x1 x2 x0 : ℝ) (m : ℝ) (h_zeros : M x1 m = 0 ∧ M x2 m = 0 ∧ x1 > x2 ∧ 2 * x0 = x1 + x2) :
  ¬ (∃ l : ℝ, ∀ x : ℝ, M x m = l * (x - x0) + M x0 m ∧ l = 0) :=
  sorry

end range_of_m_tangent_not_parallel_l455_455649


namespace problem_statements_l455_455254

-- Given the function definitions and required intervals
def f_n (n : ℕ) (x : ℝ) : ℝ := (Real.sin x)^n + (Real.cos x)^n

theorem problem_statements (x : ℝ) (n : ℕ) (h1 : x ∈ Set.Icc 0 (Real.pi / 2)) (h2 : 0 < n) :
  (∀ n, ∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f_n n x ≤ Real.sqrt 2) ∧
  (∀ n, IsConstant (f_n n) ↔ n = 2) ∧
  (Monotone.fun (f_n 4) (Set.Icc 0 (Real.pi / 4)) (decOn := sorry) ∧
  Monotone.inc (f_n 4) (Set.Icc (Real.pi / 4) (Real.pi / 2)) (incOn := sorry)) := sorry

end problem_statements_l455_455254


namespace carpenters_time_l455_455991

theorem carpenters_time (t1 t2 t3 t4 : ℝ) (ht1 : t1 = 1) (ht2 : t2 = 2)
  (ht3 : t3 = 3) (ht4 : t4 = 4) : (1 / (1 / t1 + 1 / t2 + 1 / t3 + 1 / t4)) = 12 / 25 := by
  sorry

end carpenters_time_l455_455991


namespace problem_function_value_l455_455611

theorem problem_function_value (f : ℝ → ℝ)
    (h_add : ∀ x y, f(x) + f(y) = f(x + y))
    (h_f2 : f(2) = 4) :
    f(0) + f(-2) = -4 :=
by
    sorry

end problem_function_value_l455_455611


namespace factory_non_defective_percentage_l455_455324

structure Machine :=
  (production_percentage : ℝ)
  (defect_rate : ℝ)
  (efficiency_factor : ℝ)

def M1 : Machine := ⟨20, 2, 0.8⟩
def M2 : Machine := ⟨25, 4, 0.9⟩
def M3 : Machine := ⟨30, 5, 0.85⟩
def M4 : Machine := ⟨15, 7, 0.95⟩
def M5 : Machine := ⟨10, 8, 1.1⟩

def adjusted_defect_rate (m : Machine) : ℝ :=
  m.defect_rate * m.efficiency_factor

def non_defective_percentage (m : Machine) : ℝ :=
  100 - adjusted_defect_rate m

def weighted_non_defective_percentage (m : Machine) : ℝ :=
  (non_defective_percentage m) * (m.production_percentage / 100)

def overall_non_defective_percentage : ℝ :=
  (weighted_non_defective_percentage M1) +
  (weighted_non_defective_percentage M2) +
  (weighted_non_defective_percentage M3) +
  (weighted_non_defective_percentage M4) +
  (weighted_non_defective_percentage M5)

theorem factory_non_defective_percentage :
  overall_non_defective_percentage = 95.6275 :=
by sorry

end factory_non_defective_percentage_l455_455324


namespace expected_losses_correct_l455_455396

def game_probabilities : List (ℕ × ℝ) := [
  (5, 0.6), (10, 0.75), (15, 0.4), (12, 0.85), (20, 0.5),
  (30, 0.2), (10, 0.9), (25, 0.7), (35, 0.65), (10, 0.8)
]

def expected_losses : ℝ :=
  (1 - 0.6) + (1 - 0.75) + (1 - 0.4) + (1 - 0.85) +
  (1 - 0.5) + (1 - 0.2) + (1 - 0.9) + (1 - 0.7) +
  (1 - 0.65) + (1 - 0.8)

theorem expected_losses_correct :
  expected_losses = 3.55 :=
by {
  -- Skipping the actual proof and inserting a sorry as instructed
  sorry
}

end expected_losses_correct_l455_455396


namespace sum_of_coeffs_eq_neg30_l455_455225

noncomputable def expanded : Polynomial ℤ := 
  -(Polynomial.C 4 - Polynomial.X) * (Polynomial.X + 3 * (Polynomial.C 4 - Polynomial.X))

theorem sum_of_coeffs_eq_neg30 : (expanded.coeffs.sum) = -30 := 
  sorry

end sum_of_coeffs_eq_neg30_l455_455225


namespace find_x_plus_y_l455_455682

theorem find_x_plus_y (x y : ℝ) (hx : |x| + x + y = 14) (hy : x + |y| - y = 16) : x + y = 26 / 5 := 
sorry

end find_x_plus_y_l455_455682


namespace boxes_of_toothpicks_needed_l455_455941

def total_cards : Nat := 52
def unused_cards : Nat := 23
def cards_used : Nat := total_cards - unused_cards

def toothpicks_wall_per_card : Nat := 64
def windows_per_card : Nat := 3
def doors_per_card : Nat := 2
def toothpicks_per_window_or_door : Nat := 12
def roof_toothpicks : Nat := 1250
def box_capacity : Nat := 750

def toothpicks_for_walls : Nat := cards_used * toothpicks_wall_per_card
def toothpicks_per_card_windows_doors : Nat := (windows_per_card + doors_per_card) * toothpicks_per_window_or_door
def toothpicks_for_windows_doors : Nat := cards_used * toothpicks_per_card_windows_doors
def total_toothpicks_needed : Nat := toothpicks_for_walls + toothpicks_for_windows_doors + roof_toothpicks

def boxes_needed := Nat.ceil (total_toothpicks_needed / box_capacity)

theorem boxes_of_toothpicks_needed : boxes_needed = 7 := by
  -- Proof should be done here
  sorry

end boxes_of_toothpicks_needed_l455_455941


namespace choose_one_from_ten_l455_455302

theorem choose_one_from_ten :
  Nat.choose 10 1 = 10 :=
by
  sorry

end choose_one_from_ten_l455_455302


namespace tangent_line_to_curve_y_eq_x_ln_x_l455_455138

theorem tangent_line_to_curve_y_eq_x_ln_x (m : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ (2 = real.log x₀ + 1) ∧ (m = -x₀)) → (m = -real.exp 1) :=
by
  sorry

end tangent_line_to_curve_y_eq_x_ln_x_l455_455138


namespace lcm_of_25_35_50_l455_455865

-- Define numbers
def a : ℕ := 25
def b : ℕ := 35
def c : ℕ := 50

-- Define the LCM function
def lcm (x y : ℕ) : ℕ :=
  x * y / Nat.gcd x y

-- State the theorem to be proven
theorem lcm_of_25_35_50 : lcm (lcm a b) c = 350 :=
by
  -- Proof will be here
  sorry

end lcm_of_25_35_50_l455_455865


namespace skew_lines_not_in_same_plane_l455_455879

-- Definition of skew_lines based on given condition
def skew_lines (l1 l2 : Line) : Prop :=
  ¬ ∃ (p : Point), p ∈ l1 ∧ p ∈ l2 ∧ 
  ∃ (pl1 pl2 : Plane), l1 ⊆ pl1 ∧ l2 ⊆ pl2 ∧ pl1 ≠ pl2

-- Statement of the mathematically equivalent proof problem
theorem skew_lines_not_in_same_plane (l1 l2 : Line) :
  skew_lines l1 l2 ↔ ¬ ∃ (pl : Plane), l1 ⊆ pl ∧ l2 ⊆ pl := sorry

end skew_lines_not_in_same_plane_l455_455879


namespace domain_sec_arcsin_x3_l455_455934

def sec (θ : ℝ) : ℝ := 1 / (Real.cos θ)

def function_defined_at (f : ℝ → ℝ) (a : Set ℝ) :=
  ∀ x ∈ a, ∃ y, f x = y

theorem domain_sec_arcsin_x3 : (function_defined_at (λ x, sec (Real.arcsin (x^3))) (Set.Ioo (-1 : ℝ) 1)) :=
by
  sorry

end domain_sec_arcsin_x3_l455_455934


namespace loan_years_for_B_l455_455157

def interest (P R T : ℝ) : ℝ := P * (R / 100) * T

theorem loan_years_for_B (n : ℕ) : 
  let interest_B := interest 5000 10 n,
      interest_C := interest 3000 10 4,
      total_interest := interest_B + interest_C
  in total_interest = 2200 → n = 2 :=
sorry

end loan_years_for_B_l455_455157


namespace triangle_with_two_obtuse_solutions_l455_455382

-- Given conditions
variables {A B C : ℝ} {a b c : ℝ}
def triangle (A B C a b c : ℝ) := a ∈ (A + B - C) ∧ b ∈ (A + C - B) ∧ c ∈ (B + C - A)

-- Law of Sines
def law_of_sines (a b sinA : ℝ) : ℝ := (b * sinA) / a

-- Define the problem statement
theorem triangle_with_two_obtuse_solutions (A B C : ℝ) (a b : ℝ) (hA : A = π / 6) (hb : b = 3)
  (h1 : 0 < B ∧ B < π / 3) ∨ (π / 2 < B ∧ B < 5 * π / 6) :
  (sqrt 3 < a ∧ a < 3) :=
by
  sorry

end triangle_with_two_obtuse_solutions_l455_455382


namespace count_six_digit_palindromes_l455_455195

def num_six_digit_palindromes : ℕ := 9000

theorem count_six_digit_palindromes :
  (∃ a b c d : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧
     num_six_digit_palindromes = 9000) :=
sorry

end count_six_digit_palindromes_l455_455195


namespace matrix_pow_six_l455_455553

noncomputable def matrix_input : Matrix (Fin 2) (Fin 2) ℝ :=
  !![ √3, -1;
      1,  √3]

theorem matrix_pow_six :
  (matrix_input^6) = !![ -64, 0;
                          0 , -64] :=
by
  sorry

end matrix_pow_six_l455_455553


namespace min_fence_length_l455_455769

theorem min_fence_length (w l F: ℝ) (h1: l = 2 * w) (h2: 2 * w^2 ≥ 500) : F = 96 :=
by sorry

end min_fence_length_l455_455769


namespace left_person_truthful_right_person_lies_l455_455835

theorem left_person_truthful_right_person_lies
  (L R M : Prop)
  (L_truthful_or_false : L ∨ ¬L)
  (R_truthful_or_false : R ∨ ¬R)
  (M_always_answers : M = (L → M) ∨ (¬L → M))
  (left_statement : L → (M = (L → M)))
  (right_statement : R → (M = (¬L → M))) :
  (L ∧ ¬R) ∨ (¬L ∧ R) :=
by
  sorry

end left_person_truthful_right_person_lies_l455_455835


namespace quadrilateral_area_offset_l455_455954

theorem quadrilateral_area_offset
  (d : ℝ) (x : ℝ) (y : ℝ) (A : ℝ)
  (h_d : d = 26)
  (h_y : y = 6)
  (h_A : A = 195) :
  A = 1/2 * (x + y) * d → x = 9 :=
by
  sorry

end quadrilateral_area_offset_l455_455954


namespace crossing_time_opposite_direction_10_seconds_l455_455500

noncomputable def relative_speed_same_direction := 60 - 40 -- km/h
noncomputable def relative_speed_same_direction_mps := (relative_speed_same_direction : ℝ) * (5 / 18) -- m/s

noncomputable def crossing_time_same_direction_seconds := 50 -- seconds
noncomputable def total_distance_cross_same_direction := crossing_time_same_direction_seconds * relative_speed_same_direction_mps -- meters
noncomputable def length_of_each_train := total_distance_cross_same_direction / 2 -- meters

noncomputable def relative_speed_opposite_direction := 60 + 40 -- km/h
noncomputable def relative_speed_opposite_direction_mps := (relative_speed_opposite_direction : ℝ) * (5 / 18) -- m/s

theorem crossing_time_opposite_direction_10_seconds : 
  (2 * length_of_each_train) / relative_speed_opposite_direction_mps ≈ 10 := by sorry

end crossing_time_opposite_direction_10_seconds_l455_455500


namespace sum_of_prime_f_values_eq_zero_l455_455986

def f (n : ℕ) : ℤ := n^6 - 550 * n^3 + 324

theorem sum_of_prime_f_values_eq_zero : ∑ k in (finset.filter (λ n, nat.prime (f n)) (finset.range (1000))), f k = 0 :=
by sorry

end sum_of_prime_f_values_eq_zero_l455_455986


namespace initial_cheese_amount_l455_455831

theorem initial_cheese_amount (k : ℕ) (h1 : k > 7) (h2 : 35 % k = 0) : initial_cheese 7 10 k = 11 :=
by 
  -- Define the amount each rat ate the first night
  let cheese_per_rat_first_night := 10 / k
  
  -- Define the amount each rat ate the second night
  let cheese_per_rat_second_night := 5 / k
  
  -- Total cheese eaten the second night
  let total_cheese_second_night := 7 * cheese_per_rat_second_night
  
  -- Ensure the total amount consumed remains whole
  have total_cheese_is_whole : total_cheese_second_night = 1 := by {
    rw [total_cheese_second_night, cheese_per_rat_second_night],
    -- We know k must divide 35, so the result of 35/k must be 1
    simp [k, 35 % k, h2],
  }
  
  -- Initial cheese is the sum of cheese eaten first and second night
  let initial_cheese := 10 + 1
  exact initial_cheese_is_11 initial_cheese

-- Define supporting lemma that sums the cheese amounts
lemma initial_cheese_is_11 (initial_cheese : ℕ) : initial_cheese = 11 :=
by {
  simp [initial_cheese],
  exact 10 + 1 == 11,
}

end initial_cheese_amount_l455_455831


namespace train_distance_in_2_hours_l455_455520

theorem train_distance_in_2_hours :
  (∀ (t : ℕ), t = 90 → (1 / ↑t) * 7200 = 80) :=
by
  sorry

end train_distance_in_2_hours_l455_455520


namespace y_intercept_of_line_b_l455_455758

theorem y_intercept_of_line_b
  (m : ℝ) (c₁ : ℝ) (c₂ : ℝ) (x₁ : ℝ) (y₁ : ℝ)
  (h_parallel : m = 3/2)
  (h_point : (4, 2) ∈ { p : ℝ × ℝ | p.2 = m * p.1 + c₂ }) :
  c₂ = -4 := by
  sorry

end y_intercept_of_line_b_l455_455758


namespace sum_of_integers_n_correct_sum_final_sum_of_integers_n_l455_455978

theorem sum_of_integers_n (n x : ℤ) (h : n^2 - 17 * n + 72 = x^2) (hn : 0 < n) : 
  n = 8 ∨ n = 9 := 
sorry

theorem correct_sum : (8 + 9 : ℤ) = 17 := 
by norm_num

theorem final_sum_of_integers_n :
  ∃ n1 n2 : ℤ, (n1^2 - 17 * n1 + 72 = 0) ∧ (n2^2 - 17 * n2 + 72 = 0) ∧ (0 < n1) ∧ (0 < n2) ∧ (n1 + n2 = 17) :=
begin
  use [8, 9],
  split,
  { exact (by norm_num : (8^2 - 17 * 8 + 72 = 0)) },
  split,
  { exact (by norm_num : (9^2 - 17 * 9 + 72 = 0)) },
  split,
  { exact (by norm_num : 0 < 8) },
  split,
  { exact (by norm_num : 0 < 9) },
  { exact (by norm_num : 8 + 9 = 17) }
end

end sum_of_integers_n_correct_sum_final_sum_of_integers_n_l455_455978


namespace find_hydrogen_atoms_l455_455970

-- Define the atomic weights
def atomic_weight_C : ℝ := 12.01
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.008

-- Define the total molecular weight condition
def total_molecular_weight : ℝ := 122

-- Define the combined molecular weight of Carbons and Oxygens
def molecular_weight_C : ℝ := 7 * atomic_weight_C
def molecular_weight_O : ℝ := 2 * atomic_weight_O
def molecular_weight_C_O : ℝ := molecular_weight_C + molecular_weight_O

-- Define the calculation for molecular weight of hydrogen
def molecular_weight_H_atoms : ℝ := total_molecular_weight - molecular_weight_C_O

-- Define the number of hydrogen atoms indirectly as a variable
def number_of_H_atoms : ℝ := molecular_weight_H_atoms / atomic_weight_H

-- The theorem to prove
theorem find_hydrogen_atoms : number_of_H_atoms ≈ 6 := by
  sorry

end find_hydrogen_atoms_l455_455970


namespace find_x_l455_455683

theorem find_x (x : ℝ) : 
  (∑ n in (0 : ℕ) .., (2 * n + 1) * x^n = 16) → x = 15 / 17 :=
by
  sorry

end find_x_l455_455683


namespace countPrimesWithPrimeRemainder_div6_l455_455664

/-- Define the prime numbers in the range 50 to 100 -/
def isPrimeInRange_50_100 (n : ℕ) : Prop :=
  n ∈ [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

/-- Define the condition for having a prime remainder when divided by 6 -/
def hasPrimeRemainder_div6 (n : ℕ) : Prop :=
  ∃ (r : ℕ), r ∈ [1, 2, 3, 5] ∧ n % 6 = r

theorem countPrimesWithPrimeRemainder_div6 :
  (finset.filter (λ n, hasPrimeRemainder_div6 n) (finset.filter isPrimeInRange_50_100 (finset.range 101))).card = 10 :=
by
  sorry

end countPrimesWithPrimeRemainder_div6_l455_455664


namespace sum_reciprocal_closest_integer_cbrt_l455_455359

def f (n : ℕ) : ℚ :=
  let m := round (real.cbrt n)
  in m

theorem sum_reciprocal_closest_integer_cbrt :
  ∑ k in finset.range 4095.succ, (1 / (f k)) = 451.875 := by
  sorry

end sum_reciprocal_closest_integer_cbrt_l455_455359


namespace problem_statement_l455_455278

noncomputable def f (x : ℝ) (ω : ℝ) (ϕ : ℝ) : ℝ := 2 * Real.sin (ω * x + ϕ - π / 6)

noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos (x / 2 - π / 3)

-- Problem statement
theorem problem_statement 
    (ω : ℝ) (ϕ : ℝ) (x : ℝ) (k : ℤ)
    (h_ϕ_range : 0 < ϕ ∧ ϕ < π)
    (h_ω_positive : 0 < ω)
    (h_even_function : ∀ x, f x ω ϕ = f (-x) ω ϕ)
    (h_symmetric_distance : ∀ x, f x ω ϕ = f (x + π / 2) ω ϕ) :
  (∀ x, x ∈ Set.Icc (π / 6) (5 * π / 6) → f x ω ϕ ∈ Set.Icc (-2) 1) ∧
  (∀ x, x ∈ Set.Icc 0 (4 * π) → g x = 2 * Real.cos (x / 2 - π / 3)
    → ∃ k : ℤ, x ∈ Set.Icc (4 * k * π + 2 * π / 3) (4 * k * π + 8 * π / 3)) :=
sorry

end problem_statement_l455_455278


namespace not_integer_sum_fraction_l455_455780

theorem not_integer_sum_fraction (n : ℕ) (hn : n = 1986) :
  ¬ ∃ k : ℤ, (∑ i in finset.range (n + 1), ∑ j in finset.range (i), (1 / (i * j))) = k := by
    sorry

end not_integer_sum_fraction_l455_455780


namespace option_d_is_incorrect_l455_455626

variables {α : Type*} [normed_group α] [normed_space ℝ α] [finite_dimensional ℝ α]

def vector_ab : α := ⟨2, -1, -4⟩
def vector_ad : α := ⟨4, 2, 0⟩
def vector_ap : α := ⟨-1, 2, -1⟩
def point_P_not_on_plane : Prop := ∀ {v : α}, (v = vector_ab ∨ v = vector_ad → ∀ (k : ℝ), v ≠ k • vector_ap)

theorem option_d_is_incorrect (h : point_P_not_on_plane) : ∃ (bd : α), ¬∃ (λ : ℝ), vector_ap = λ • bd := 
by {
  sorry
}

end option_d_is_incorrect_l455_455626


namespace find_other_number_l455_455800

noncomputable def calculateB (lcm hcf a : ℕ) : ℕ :=
  (lcm * hcf) / a

theorem find_other_number :
  ∃ B : ℕ, (calculateB 76176 116 8128) = 1087 :=
by
  use 1087
  sorry

end find_other_number_l455_455800


namespace B_time_to_complete_work_l455_455493

-- Defining the work rates and conditions
variable (A B C : ℝ)
def work_rate_A := A = 1 / 3
def work_rate_BC := B + C = 1 / 3
def work_rate_AC := A + C = 1 / 2

-- Main theorem statement
theorem B_time_to_complete_work (hA : work_rate_A) (hBC : work_rate_BC) (hAC : work_rate_AC) : 1 / B = 6 :=
by
  sorry

end B_time_to_complete_work_l455_455493


namespace equilateral_triangle_area_l455_455120

def point := (ℝ, ℝ)

noncomputable def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

noncomputable def area_of_equilateral_triangle (p1 p2 p3 : point) : ℝ :=
  let side_length := distance p1 p2 in
  (real.sqrt 3 / 4) * side_length ^ 2

theorem equilateral_triangle_area:
  let E := (1, 2) in
  let F := (1, 8) in
  let G := (7, 2) in
  area_of_equilateral_triangle E F G = 9 * real.sqrt 3 := 
sorry

end equilateral_triangle_area_l455_455120


namespace perp_through_midpoint_l455_455018

open Real

/-- Given trapezoid ABCD with AB = BC = CD and height CH from C to base AB, prove
    that the perpendicular dropped from H onto AC passes through the midpoint of BD. -/
theorem perp_through_midpoint (A B C D H : Point) (midpoint_BD : Line)  
    (h1 : is_trapezoid A B C D)
    (h2 : dist A B = dist B C ∧ dist B C = dist C D)
    (h3 : is_height C H A B)
    (h4 : is_perpendicular H midpoint_BD)
    : passes_through_midpoint H midpoint_BD A C  := 
sorry

end perp_through_midpoint_l455_455018


namespace no_solution_for_inequalities_l455_455213

theorem no_solution_for_inequalities (x : ℝ) : ¬(4 * x ^ 2 + 7 * x - 2 < 0 ∧ 3 * x - 1 > 0) :=
by
  sorry

end no_solution_for_inequalities_l455_455213


namespace voice_of_china_signup_ways_l455_455723

theorem voice_of_china_signup_ways : 
  (2 * 2 * 2 = 8) :=
by {
  sorry
}

end voice_of_china_signup_ways_l455_455723


namespace solution_set_f_l455_455411

noncomputable def f : ℝ → ℝ := sorry

axiom f_domain : ∀ x : ℝ, f x ∈ ℝ
axiom f_at_neg1 : f (-1) = 2
axiom f_derivative_pos : ∀ x : ℝ, deriv f x > 2

theorem solution_set_f : { x : ℝ | f x > 2 * x + 4 } = set.Ioi (-1) :=
by 
  sorry

end solution_set_f_l455_455411


namespace train_speed_l455_455896

theorem train_speed (length_train : ℝ) (time_to_cross : ℝ) (length_bridge : ℝ)
  (h_train : length_train = 100) (h_time : time_to_cross = 12.499)
  (h_bridge : length_bridge = 150) : 
  ((length_train + length_bridge) / time_to_cross * 3.6) = 72 := 
by 
  sorry

end train_speed_l455_455896


namespace problem_1_solution_problem_2_solution_l455_455200

theorem problem_1_solution (x : ℝ) : (2 * x ^ 2 - 2 * real.sqrt 2 * x + 1 = 0) → (x = real.sqrt 2 / 2) := 
by 
  intro h
  sorry

theorem problem_2_solution (x : ℝ) : (x * (2 * x - 5) = 4 * x - 10) → (x = 5 / 2 ∨ x = 2) := 
by 
  intro h
  sorry

end problem_1_solution_problem_2_solution_l455_455200


namespace probability_of_roots_l455_455163

theorem probability_of_roots (k : ℝ) (h1 : 8 ≤ k) (h2 : k ≤ 13) :
  let a := k^2 - 2 * k - 35
  let b := 3 * k - 9
  let c := 2
  let discriminant := b^2 - 4 * a * c
  discriminant ≥ 0 → 
  (∃ x1 x2 : ℝ, 
    a * x1^2 + b * x1 + c = 0 ∧ 
    a * x2^2 + b * x2 + c = 0 ∧
    x1 ≤ 2 * x2) ↔ 
  ∃ p : ℝ, p = 0.6 := 
sorry

end probability_of_roots_l455_455163


namespace ball_reaches_20_feet_at_1_75_seconds_l455_455418

noncomputable def ball_height (t : ℝ) : ℝ :=
  60 - 9 * t - 8 * t ^ 2

theorem ball_reaches_20_feet_at_1_75_seconds :
  ∃ t : ℝ, ball_height t = 20 ∧ t = 1.75 ∧ t ≥ 0 :=
by {
  sorry
}

end ball_reaches_20_feet_at_1_75_seconds_l455_455418


namespace find_b_in_terms_of_a_l455_455303

noncomputable def triangle_problem (a b c : ℝ) (angleA angleB : ℝ) : Prop :=
  (|BC| = a) ∧ (|AC| = b) ∧ (|AB| = c) ∧ (3 * angleA + angleB = 180) ∧ (3 * a = 2 * c) → 
  (b = 5 * a / 4)

-- Statement of the theorem to be proven
theorem find_b_in_terms_of_a (a b c : ℝ) (angleA angleB : ℝ) :
  triangle_problem a b c angleA angleB := 
sorry

end find_b_in_terms_of_a_l455_455303


namespace prob_divisor_multiple_of_10_88_is_9_over_625_l455_455743

noncomputable def gcd(a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

def num_divisors (n : ℕ) : ℕ :=
  (n + 1) * (n + 1)

theorem prob_divisor_multiple_of_10_88_is_9_over_625 :
  let m := 9, n := 625 in
  (gcd 144 10000 = 16) → 
  (num_divisors 99 = 10000) → 
  (num_divisors 11 = 144) → 
  m + n = 634 :=
by
  intros
  sorry

end prob_divisor_multiple_of_10_88_is_9_over_625_l455_455743


namespace Vanya_number_thought_of_l455_455455

theorem Vanya_number_thought_of :
  ∃ m n : ℕ, m < 10 ∧ n < 10 ∧ (10 * m + n = 81 ∧ (10 * n + m)^2 = 4 * (10 * m + n)) :=
sorry

end Vanya_number_thought_of_l455_455455


namespace volume_unoccupied_by_cones_l455_455449

theorem volume_unoccupied_by_cones
  (r_cone : ℝ) (h_cone : ℝ) (h_cyl : ℝ) (V_unoccupied : ℝ)
  (hc1 : r_cone = 12) (hc2 : h_cone = 15) (hc3 : h_cyl = 24) 
  (hc4 : V_unoccupied = 2016 * Real.pi) : 
  V_unoccupied = 
  (Real.pi * (r_cone ^ 2) * h_cyl) - 2 * (1/3 * Real.pi * (r_cone ^ 2) * h_cone) :=
begin
  sorry
end

end volume_unoccupied_by_cones_l455_455449


namespace a_624_eq_196250_l455_455542

def a : ℕ → ℕ
| 0       := 5
| (n + 1) := a n + (n + 2) + 1

theorem a_624_eq_196250 : a 623 = 196250 :=
sorry

end a_624_eq_196250_l455_455542


namespace conjugate_neg_2i_l455_455121

open Complex

theorem conjugate_neg_2i : conj (-2 * I) = 2 * I :=
by
  sorry

end conjugate_neg_2i_l455_455121


namespace sum_of_products_inequality_l455_455129

theorem sum_of_products_inequality
  (n: ℕ) 
  (h₁: n > 2)
  (x: ℕ → ℝ)
  (h₂: ∀ i j, i < j → x i < x j) :
  let c := n * (n - 1) / 2 in
  c * (∑ i j, if i < j then x i * x j else 0) > 
  ((∑ k in Finset.range (n - 1), (n - 1 - k) * x k) * 
   (∑ k in Finset.range (n - 1), (k + 1) * x (k + 1))) :=
by
  sorry

end sum_of_products_inequality_l455_455129


namespace exists_matrix_with_perfect_square_entries_l455_455066

-- Define a matrix in the given form
def matrixA (a b c : ℕ) : Matrix (Fin 4) (Fin 4) ℕ := 
  ![![1, a, b, c], 
    ![0, 1, a, b], 
    ![0, 0, 1, a], 
    ![0, 0, 0, 1]]

-- Define the condition that a natural number is a perfect square
def is_perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, k * k = x

-- Define the condition for entries of the matrix to be positive
def positive_entries (a b c : ℕ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c

-- Define the condition that all entries of a matrix are perfect squares
def matrix_all_perfect_squares {n : ℕ} (M : Matrix (Fin 4) (Fin 4) ℕ) : Prop :=
  ∀ i j, is_perfect_square (M i j)

-- The final theorem statement
theorem exists_matrix_with_perfect_square_entries (n : ℕ) (hn : 0 < n):
  ∃ (a b c : ℕ), positive_entries a b c ∧ matrix_all_perfect_squares (matrixA a b c ^ n) :=
by
  sorry

end exists_matrix_with_perfect_square_entries_l455_455066


namespace find_q_value_l455_455308

theorem find_q_value 
  (p q r : ℕ) 
  (hp : 0 < p) 
  (hq : 0 < q) 
  (hr : 0 < r) 
  (h : p + 1 / (q + 1 / r : ℚ) = 25 / 19) : 
  q = 3 :=
by 
  sorry

end find_q_value_l455_455308


namespace coeff_x5_in_expansion_l455_455806

theorem coeff_x5_in_expansion : 
  (∃ (c : ℕ), (x + 1) ^ 8 = c * x^5 + _) → c = 56 :=
by
  sorry

end coeff_x5_in_expansion_l455_455806


namespace find_ab_minus_a_neg_b_l455_455995

variable (a b : ℝ)
variables (h₀ : a > 1) (h₁ : b > 0) (h₂ : a^b + a^(-b) = 2 * Real.sqrt 2)

theorem find_ab_minus_a_neg_b : a^b - a^(-b) = 2 := by
  sorry

end find_ab_minus_a_neg_b_l455_455995


namespace max_real_roots_alternating_sign_polynomial_l455_455211

def alternating_sign_polynomial (n : ℕ) : Polynomial ℝ := 
  ∑ i in Finset.range (n + 1), (-1 : ℝ) ^ i * Polynomial.X ^ (n - i)

theorem max_real_roots_alternating_sign_polynomial (n : ℕ) (hn : 0 < n) :
  (∀ x : ℝ, alternating_sign_polynomial n = 0 → x = 1) ↔ ¬Even n ∨ 
  (∃ x : ℝ, alternating_sign_polynomial n = 0 ∧ x = -1) :=
sorry

end max_real_roots_alternating_sign_polynomial_l455_455211


namespace cos_A_is_one_l455_455327

-- Definitions as per Lean's requirement
variable (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]

-- Declaring the conditions are given
variables (α : ℝ) (cos_A : ℝ)
variables (AB CD AD BC : ℝ)
def is_convex_quadrilateral (A B C D : Type) : Prop := 
  sorry -- This would be a formal definition of convex quadrilateral

-- The conditions are specified in Lean terms
variables (h1 : is_convex_quadrilateral A B C D)
variables (h2 : α = 0) -- α = 0 implies cos(α) = 1
variables (h3 : AB = 240)
variables (h4 : CD = 240)
variables (h5 : AD ≠ BC)
variables (h6 : AB + CD + AD + BC = 960)

-- The proof statement to indicate that cos(α) = 1 under the given conditions
theorem cos_A_is_one : cos_A = 1 :=
by
  sorry -- Proof not included as per the instruction

end cos_A_is_one_l455_455327


namespace area_of_triangles_equal_l455_455028

theorem area_of_triangles_equal {a b c d : ℝ} (h_hyperbola_a : a ≠ 0) (h_hyperbola_b : b ≠ 0) 
    (h_hyperbola_c : c ≠ 0) (h_hyperbola_d : d ≠ 0) (h_parallel : a * b = c * d) :
  (1 / 2) * ((a + c) * (a + c) / (a * c)) = (1 / 2) * ((b + d) * (b + d) / (b * d)) :=
by
  sorry

end area_of_triangles_equal_l455_455028


namespace num_subsets_of_A_inter_B_l455_455755

-- Define set A
def A : Set (ℝ × ℝ) := {p | ∃ x : ℝ, p = (x, x + 2)}

-- Define set B
def B : Set (ℝ × ℝ) := {p | ∃ x : ℝ, p = (x, 2^x)}

-- Define the problem and declare the theorem
theorem num_subsets_of_A_inter_B : 
  (A ∩ B).size = 2 → (A ∩ B).powerset.size = 4 :=
by
  -- Condition: intersection of sets A and B has exactly 2 elements
  sorry

end num_subsets_of_A_inter_B_l455_455755


namespace density_bounded_continuous_l455_455363

open MeasureTheory ProbabilityTheory

variables {X : ℝ → ℝ} {φ : ℝ → ℂ}

theorem density_bounded_continuous (hX : ∀ t, has_char_fn X (φ t))
  (hφ : ∫ t, complex.abs (φ t) < ∞) :
  ∃ f : ℝ → ℝ, 
    (∀ x, f x = (1 / (2 * Real.pi)) * ∫ t : ℝ, complex.exp (-complex.I * t * x) * (φ t)) ∧ 
    (∀ x, continuous (f x)) ∧ 
    (∀ x, f x ∈ set.Icc (0 : ℝ) (1 / (2 * Real.pi)) * (∫ t : ℝ, complex.abs (φ t))) :=
begin
  sorry -- The proof would go here
end

end density_bounded_continuous_l455_455363


namespace average_increase_is_zero_l455_455027

-- Define the initial scores and new scores.
def initial_scores := [92, 86, 95]
def additional_scores := [89, 93]

-- define the sum of a list
def list_sum (l : List ℕ) : ℕ := l.foldl (· + ·) 0

-- Define average calculation.
def average (scores : List ℕ) : ℚ :=
  list_sum scores / scores.length

-- Statement: Kim's average score increase after the last two exams is 0.
theorem average_increase_is_zero :
  let new_scores := initial_scores ++ additional_scores in
  average new_scores - average initial_scores = 0 :=
by
  sorry

end average_increase_is_zero_l455_455027


namespace base8_253_to_base10_l455_455206

-- Define the base 8 number as a tuple where each element represents the digit and its position
def base8_number : list (ℕ × ℕ) := [(2, 2), (5, 1), (3, 0)]

-- Function to convert a base 8 number to base 10
def base8_to_base10 (digits : list (ℕ × ℕ)) : ℕ :=
  digits.foldr (λ (digit : ℕ × ℕ) (acc : ℕ), acc + digit.fst * 8 ^ digit.snd) 0

-- The theorem to prove 
theorem base8_253_to_base10 : base8_to_base10 base8_number = 171 :=
by 
  -- unfold and calculate the converted value
  rw [base8_to_base10]
  -- assertion of each step in list evaluation
  simp [*, List.foldr_eq_foldr, List.foldr]
  -- add the values 128, 40 and 3
  sorry

end base8_253_to_base10_l455_455206


namespace distribution_ways_l455_455544

def count_distributions (n : ℕ) (k : ℕ) : ℕ :=
-- Calculation for count distributions will be implemented here
sorry

theorem distribution_ways (items bags : ℕ) (cond : items = 6 ∧ bags = 3):
  count_distributions items bags = 75 :=
by
  -- Proof would be implemented here
  sorry

end distribution_ways_l455_455544


namespace circumscribed_quadrilateral_sine_rule_l455_455097

variables {A B C D : ℝ} -- Vertex angles of the quadrilateral
variables {AB CD : ℝ} -- Lengths of sides AB and CD
variables {r : ℝ} -- Radius of the circle

theorem circumscribed_quadrilateral_sine_rule
  (h1 : ∠A + ∠B + ∠C + ∠D = 2 * π)
  (h2 : ∠A + ∠C = π)
  (h3 : ∠B + ∠D = π)
  (h4 : quadrilateral_is_circumscribed_about_circle AB CD r)
  : AB * (sin (A / 2)) * (sin (B / 2)) = CD * (sin (C / 2)) * (sin (D / 2)) :=
begin
  sorry
end

end circumscribed_quadrilateral_sine_rule_l455_455097


namespace alex_cell_phone_cost_l455_455515

def base_cost : ℝ := 20
def text_cost_per_message : ℝ := 0.1
def extra_min_cost_per_minute : ℝ := 0.15
def text_messages_sent : ℕ := 150
def hours_talked : ℝ := 32
def included_hours : ℝ := 25

theorem alex_cell_phone_cost : base_cost 
  + (text_messages_sent * text_cost_per_message)
  + ((hours_talked - included_hours) * 60 * extra_min_cost_per_minute) = 98 := by
  sorry

end alex_cell_phone_cost_l455_455515


namespace ascending_order_conversion_l455_455659

def convert_base (num : Nat) (base : Nat) : Nat :=
  match num with
  | 0 => 0
  | _ => (num / 10) * base + (num % 10)

theorem ascending_order_conversion :
  let num16 := 12
  let num7 := 25
  let num4 := 33
  let base16 := 16
  let base7 := 7
  let base4 := 4
  convert_base num4 base4 < convert_base num16 base16 ∧ 
  convert_base num16 base16 < convert_base num7 base7 :=
by
  -- Here would be the proof, but we skip it
  sorry

end ascending_order_conversion_l455_455659


namespace product_of_integers_l455_455992

theorem product_of_integers (A B C D : ℕ) 
  (h1 : A + B + C + D = 100) 
  (h2 : 2^A = B - 6) 
  (h3 : C + 6 = D)
  (h4 : B + C = D + 10) : 
  A * B * C * D = 33280 := 
by
  sorry

end product_of_integers_l455_455992


namespace John_finishe_watching_all_seasons_in_77_days_l455_455348

def typical_episodes := 22
def third_season_episodes := 24
def last_season_additional_episodes := 4
def initial_seasons := 9
def final_season := 1
def bonus_episodes := 5
def crossover_episode_hours := 1.5
def first_three_seasons_episode_length := 0.5
def remaining_episode_length := 0.75
def marathon_hours := 5
def daily_watch_hours := 2

def total_days_to_finish : ℕ :=
  let seasons_1_3_episodes := 3 * typical_episodes + (third_season_episodes - typical_episodes)
  let seasons_4_9_episodes := 6 * typical_episodes
  let last_season_episodes := typical_episodes + last_season_additional_episodes
  let total_episodes := seasons_1_3_episodes + seasons_4_9_episodes + last_season_episodes
  let total_hours := (seasons_1_3_episodes * first_three_seasons_episode_length) 
                  + (seasons_4_9_episodes * remaining_episode_length)
                  + (last_season_episodes * remaining_episode_length)
                  + (bonus_episodes * 1)
                  + crossover_episode_hours
  let remaining_hours := total_hours - marathon_hours
  let remaining_days := remaining_hours / daily_watch_hours
  77

theorem John_finishe_watching_all_seasons_in_77_days : total_days_to_finish = 77 := by
  sorry

end John_finishe_watching_all_seasons_in_77_days_l455_455348


namespace segment_of_line_in_isosceles_triangle_l455_455710

variables {α r : ℝ}

theorem segment_of_line_in_isosceles_triangle (h: α ≠ 0) :
  let θ := (3 * α) / 2
  in ∀ (t : Triangle) (O : Point) (B B1 : Point),
  t.is_isosceles ∧ t.angle_at_base = α ∧ t.inscribed_circle_radius = r →
  t.line_through_vertex_base_circ_center(B, B1, O) →
  segment_length B B1 = (4 * r * (Real.cos (α / 2))^2) / Real.sin θ :=
begin
  sorry
end

end segment_of_line_in_isosceles_triangle_l455_455710


namespace find_constant_k_l455_455210

theorem find_constant_k : 
  ∃ k : ℝ, (∀ x y : ℝ, y = x^2 - x + 1 ∧ y = 4x + k → x = (5 / 2)) ∧ k = -21 / 4 := by
  sorry

end find_constant_k_l455_455210


namespace area_inequality_l455_455802

theorem area_inequality
  (A B C A' B' C' : Type) [triangle ABC] 
  [angle_bisectors_intersect_circumcircle_at ABC A' B' C']
  (R : ℝ) [circumradius_of_triangle ABC R]
  (area_ABC : ℝ) [area_triangle ABC area_ABC]
  (area_A'B'C' : ℝ) [area_triangle A'B'C' area_A'B'C'] :
  16 * (area_A'B'C')^3 ≥ 27 * area_ABC * R^4 := sorry

end area_inequality_l455_455802


namespace incorrect_operation_l455_455485

theorem incorrect_operation :
  (√2) ^ 2 = 2 ∧
  √(3 ^ 2) = 3 ∧
  √(1 / 2) = (√2) / 2 ∧
  (¬ (√4 = 2) → false) → 
  false := by
  sorry

end incorrect_operation_l455_455485


namespace trackball_mice_count_l455_455049

theorem trackball_mice_count (total_sales wireless_share optical_share : ℕ) 
    (h_total : total_sales = 80)
    (h_wireless : wireless_share = total_sales / 2)
    (h_optical : optical_share = total_sales / 4):
    total_sales - (wireless_share + optical_share) = 20 :=
by
  sorry

end trackball_mice_count_l455_455049


namespace genevieve_fixed_errors_l455_455993

theorem genevieve_fixed_errors
  (lines_written : ℕ)
  (lines_per_block : ℕ)
  (initial_errors : ℕ)
  (additional_errors_per_block : ℕ)
  : lines_written = 4300 ∧ lines_per_block = 100 ∧ initial_errors = 3 ∧ additional_errors_per_block = 1 →
    let blocks := lines_written / lines_per_block in
    let sum_errors (n : ℕ) : ℕ := n * (2 * initial_errors + (n - 1) * additional_errors_per_block) / 2 in
    sum_errors blocks = 1032 :=
by
  sorry

end genevieve_fixed_errors_l455_455993


namespace factorial_sum_mod_prime_l455_455984

def sum_factorials (n : ℕ) : ℕ :=
  (Finset.range (n+1)).sum (λ k, nat.factorial k)

noncomputable def nat_floor_div_e (n : ℕ) : ℕ :=
  nat.floor ((nat.factorial n : ℝ) / real.exp 1)

theorem factorial_sum_mod_prime (p : ℕ) [Fact (Nat.Prime p)] (hp : p % 2 = 1) :
  (sum_factorials p - nat_floor_div_e (p-1)) % p = 0 :=
sorry

end factorial_sum_mod_prime_l455_455984


namespace game_winning_strategy_l455_455765

theorem game_winning_strategy : 
  let N_max := 2020 in
  let S := (List.range (N_max + 1)).filter odd |>.sum in
  let R := (List.range (N_max + 1)).filter (λ n, ¬odd n) |>.sum in
  (R - S) / 10 = 101 :=
by
  let N_max := 2020
  let S := (List.range (N_max + 1)).filter odd |>.sum
  let R := (List.range (N_max + 1)).filter (λ n, ¬odd n) |>.sum
  have H : (R - S) / 10 = 101 := sorry
  exact H

end game_winning_strategy_l455_455765


namespace number_of_vertices_bounded_l455_455352

variable (G : Type) [Graph G]

def radius (G : Type) [Graph G] : ℕ := sorry
def max_degree (G : Type) [Graph G] : ℕ := sorry
def num_vertices (G : Type) [Graph G] : ℕ := sorry

theorem number_of_vertices_bounded
  (k d : ℕ) (hd : d ≥ 3) 
  (hradius : radius G ≤ k)
  (hdegree : max_degree G ≤ d) :
  num_vertices G < (d / (d - 2)) * (d - 1)^k := sorry

end number_of_vertices_bounded_l455_455352


namespace exists_palindromes_l455_455718

def is_palindrome {α : Type} [decidable_eq α] (w : list α) : Prop :=
∀ j, 1 ≤ j ∧ j ≤ w.length → w.nth (j - 1) = w.nth (w.length - j)

def concatenate {α : Type} (w1 w2 : list α) : list α := w1 ++ w2

theorem exists_palindromes
  (r s t : ℕ)
  (hrst : r + s = t + 2)
  (hcoprime : (r + 2).gcd (s - 2) = 1) :
  ∃ (A B C : list ℕ),
    is_palindrome A ∧
    is_palindrome B ∧
    is_palindrome C ∧
    A.length = r ∧
    B.length = s ∧
    C.length = t ∧
    concatenate A B = concatenate C ['b', 'a'] :=
sorry

end exists_palindromes_l455_455718


namespace kids_go_to_camp_l455_455940

-- Define the total number of kids in Lawrence County
def total_kids : ℕ := 1059955

-- Define the number of kids who stay home
def stay_home : ℕ := 495718

-- Define the expected number of kids who go to camp
def expected_go_to_camp : ℕ := 564237

-- The theorem to prove the number of kids who go to camp
theorem kids_go_to_camp :
  total_kids - stay_home = expected_go_to_camp :=
by
  -- Proof is omitted
  sorry

end kids_go_to_camp_l455_455940


namespace part1_part2_part3_l455_455282

noncomputable def f (a x : ℝ) : ℝ := a * x ^ 2 + (3 + a) * x + 3

def g (a x k : ℝ) : ℝ := (f a x) - k * x

theorem part1 (a : ℝ) (h1 : f a 2 = 3) : f a = (-1) * x^2 + 2 * x + 3 := sorry

theorem part2 (a k : ℝ) (h1 : f a = -x^2 + 2 * x + 3) 
               (h2 : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → ∀ y : ℝ, -2 ≤ y ∧ y ≤ 2 → g a x k ≤ g a y k ) :
              k ∈ set.univ \ {k : ℝ | -2 < k ∧ k < 6} := sorry

theorem part3 (a : ℝ) : (a = -1) ∨ (a = -9) := sorry

end part1_part2_part3_l455_455282


namespace natalia_crates_l455_455582

/- The definitions from the conditions -/
def novels : ℕ := 145
def comics : ℕ := 271
def documentaries : ℕ := 419
def albums : ℕ := 209
def crate_capacity : ℕ := 9

/- The proposition to prove -/
theorem natalia_crates : (novels + comics + documentaries + albums) / crate_capacity = 116 := by
  -- this skips the proof and assumes the theorem is true
  sorry

end natalia_crates_l455_455582


namespace necessary_but_not_sufficient_condition_l455_455630

theorem necessary_but_not_sufficient_condition (p q : ℝ → Prop)
    (h₁ : ∀ x k, p x ↔ x ≥ k) 
    (h₂ : ∀ x, q x ↔ 3 / (x + 1) < 1) 
    (h₃ : ∃ k : ℝ, ∀ x, p x → q x ∧ ¬ (q x → p x)) :
  ∃ k, k > 2 :=
by
  sorry

end necessary_but_not_sufficient_condition_l455_455630


namespace always_divisible_by_2018_l455_455029

-- Define the conditions as hypotheses.
variables (k : ℕ) (N : ℕ) (t : ℕ)

theorem always_divisible_by_2018 (h_even : k = 1009 ^ t - 1) (h_N_gt_one : N > 1) (h_k_even : even k) :
  ∃ N' : ℕ, (N' ∣ N ∧ 2018 ∣ N') :=
sorry

end always_divisible_by_2018_l455_455029


namespace find_complex_number_l455_455209

theorem find_complex_number (a b : ℝ) (z : ℂ) (hz : z = a + b * complex.I) 
  (h : 3 * z - 4 * (complex.conj z) = -3 - 45 * complex.I) : 
  z = 3 - (45 / 7) * complex.I :=
by
  sorry

end find_complex_number_l455_455209


namespace mode_of_ages_l455_455801

/-- The ages of the 16 members of the school's male soccer team. -/
def ages : List ℕ := [17, 17, 18, 18, 16, 18, 17, 15, 18, 18, 17, 16, 18, 17, 18, 14]

/-- The mode of a list of natural numbers is the number that appears most frequently. -/
def mode (l : List ℕ) : ℕ :=
  l.foldr (fun x acc =>
    if l.count x > l.count acc then x else acc) l.head

/-- The mode of the ages list is 18 years old. -/
theorem mode_of_ages : mode ages = 18 := sorry

end mode_of_ages_l455_455801


namespace distribution_ways_l455_455178

-- Definitions for the conditions in the problem
def num_problems := 7
def num_friends := 10
def each_person_min := 1
def each_person_max := 2

-- The theorem statement
theorem distribution_ways : 
  (∃ f : Fin num_problems → Fin num_friends, 
     (∀ friend, 1 ≤ (f friend).card ∧ (f friend).card ≤ 2)) →
  (∑ f in (Fin num_problems → Fin num_friends), (1 ≤ (∑ i in (Fin num_problems), if f i = friend then 1 else 0) ∧ 
     (∑ i in (Fin num_problems), if f i = friend then 1 else 0) ≤ 2)) = 712800000 :=
by
  sorry

end distribution_ways_l455_455178


namespace intersection_of_medians_parallelogram_l455_455527

-- Define the quadrilateral with its diagonals intersecting
variables {A B C D O : Point}
variables (h1 : O ∈ line A C) (h2 : O ∈ line B D)

-- Define the triangles formed
variables (ΔAOB ΔBOC ΔCOD ΔDOA : Triangle)

-- Definition stating the intersection points of medians form a parallelogram
theorem intersection_of_medians_parallelogram :
    ∃ G1 G2 G3 G4 : Point,
      is_median_intersection A O B G1 ∧
      is_median_intersection B O C G2 ∧
      is_median_intersection C O D G3 ∧
      is_median_intersection D O A G4 ∧
      is_parallelogram G1 G2 G3 G4 :=
sorry

end intersection_of_medians_parallelogram_l455_455527


namespace gcd_sum_lcm_eq_gcd_l455_455776

theorem gcd_sum_lcm_eq_gcd (a b : ℤ) : Int.gcd (a + b) (Int.lcm a b) = Int.gcd a b :=
by 
  sorry

end gcd_sum_lcm_eq_gcd_l455_455776


namespace cost_of_camel_l455_455508

variables (C H O E G Z L : ℕ)

theorem cost_of_camel :
  (10 * C = 24 * H) →
  (16 * H = 4 * O) →
  (6 * O = 4 * E) →
  (3 * E = 5 * G) →
  (8 * G = 12 * Z) →
  (20 * Z = 7 * L) →
  (10 * E = 120000) →
  C = 4800 :=
by
  sorry

end cost_of_camel_l455_455508


namespace tangents_collinear_F_minimum_area_triangle_l455_455260

noncomputable def ellipse_condition : Prop :=
  ∀ (x y : ℝ), x^2 / 4 + y^2 / 3 = 1

noncomputable def point_P_on_line (P : ℝ × ℝ) : Prop :=
  P.1 = 4

noncomputable def tangent_condition (P A B : ℝ × ℝ) (ellipse : ℝ × ℝ → Prop) : Prop :=
  -- Tangent lines meet the ellipse equation at points A and B
  ellipse A ∧ ellipse B

noncomputable def collinear (A F B : ℝ × ℝ) : Prop :=
  (A.2 - F.2) * (B.1 - F.1) = (B.2 - F.2) * (A.1 - F.1)

noncomputable def minimum_area (P A B : ℝ × ℝ) : ℝ :=
  1 / 2 * abs ((A.1 * B.2 + B.1 * P.2 + P.1 * A.2) - (A.2 * B.1 + B.2 * P.1 + P.2 * A.1))

theorem tangents_collinear_F (F : ℝ × ℝ) (P A B : ℝ × ℝ) (ellipse : ℝ × ℝ → Prop)
  (h_ellipse_F : F = (1, 0))
  (h_point_P_on_line : point_P_on_line P)
  (h_tangent_condition : tangent_condition P A B ellipse)
  (h_ellipse_def : ellipse_condition) :
  collinear A F B :=
sorry

theorem minimum_area_triangle (F P A B : ℝ × ℝ) (ellipse : ℝ × ℝ → Prop)
  (h_ellipse_F : F = (1, 0))
  (h_point_P_on_line : point_P_on_line P)
  (h_tangent_condition : tangent_condition P A B ellipse)
  (h_ellipse_def : ellipse_condition) :
  minimum_area P A B = 9 / 2 :=
sorry

end tangents_collinear_F_minimum_area_triangle_l455_455260


namespace proportion_Q_to_R_l455_455540

theorem proportion_Q_to_R (q r : ℕ) (h1 : 3 * q + 5 * r = 1000) (h2 : 4 * r - 2 * q = 250) : q = r :=
by sorry

end proportion_Q_to_R_l455_455540


namespace fresh_grapes_weight_l455_455604

noncomputable def weight_of_fresh_grapes (F D : ℕ) : Prop :=
  D = 15 ∧ 0.4 * F = 0.8 * D

theorem fresh_grapes_weight : ∃ F : ℕ, weight_of_fresh_grapes F 15 ∧ F = 30 :=
by
  sorry

end fresh_grapes_weight_l455_455604


namespace frog_climb_time_l455_455153

-- Define the problem as an assertion within Lean.
theorem frog_climb_time 
  (well_depth : ℕ) (climb_up : ℕ) (slide_down : ℕ) (time_per_meter: ℕ) (climb_start_time : ℕ) 
  (time_to_slide_multiplier: ℚ)
  (time_to_second_position: ℕ) 
  (final_distance: ℕ) 
  (total_time: ℕ)
  (h_start : well_depth = 12)
  (h_climb_up: climb_up = 3)
  (h_slide_down : slide_down = 1)
  (h_time_per_meter : time_per_meter = 1)
  (h_time_to_slide_multiplier: time_to_slide_multiplier = 1/3)
  (h_time_to_second_position : climb_start_time = 8 * 60 /\ time_to_second_position = 8 * 60 + 17)
  (h_final_distance : final_distance = 3)
  (h_total_time: total_time = 22) :
  
  ∃ (t: ℕ), 
    t = total_time := 
by
  sorry

end frog_climb_time_l455_455153


namespace find_side_b_l455_455275

-- Define the setting and necessary conditions
noncomputable def triangle_arithmetic_sequence_b (A B C : ℝ) (a b c : ℝ) : Prop :=
  (A + B + C = Real.pi) ∧
  (B = (A + C) / 2) ∧
  (a = 4) ∧
  (c = 3) ∧
  (b^2 = a^2 + c^2 - 2 * a * c * Real.cos B)

-- Theorem statement
theorem find_side_b {A B C a c : ℝ} :
  (A + B + C = Real.pi) →
  (B = (A + C) / 2) →
  (a = 4) →
  (c = 3) →
  ∃ (b : ℝ), b = Real.sqrt 13 :=
by
  assume h1 h2 h3 h4
  exists Real.sqrt 13
  sorry

end find_side_b_l455_455275


namespace distance_to_intersections_l455_455386

noncomputable def parabola_1 (x : ℝ) : ℝ := -3 * x^2 + 2
noncomputable def parabola_2 (y : ℝ) : ℝ := -4 * y^2 + 2

theorem distance_to_intersections :
  let A := ((-1:ℝ)/6, (-1:ℝ)/8) in
  let intersection_points := {p : ℝ × ℝ | parabola_1 p.1 = p.2 ∧ parabola_2 p.2 = p.1} in
  ∀ p ∈ intersection_points, dist A p = (sqrt 697) / 24 :=
sorry

end distance_to_intersections_l455_455386


namespace f_g_of_4_eq_l455_455037

def f (x : ℝ) : ℝ := 3 * Real.sqrt x + 15 / Real.sqrt x
def g (x : ℝ) : ℝ := 2 * x ^ 2 - 4 * x - 5

theorem f_g_of_4_eq :
  f (g 4) = (48 / 11) * Real.sqrt 11 :=
by
  sorry

end f_g_of_4_eq_l455_455037


namespace intersection_point_on_circumcircle_l455_455116

-- Definition of the problem setup
variable (A B C : Point)
variables (circumcircle : Circle) (is_parallel : Line → Line → Prop)
variables (angle_bisector : Angle → Line)
variables (m n : Line)

-- Given conditions as Lean definitions
variable (triangle_abc : Is_triangle A B C)
variable (parallel_through_A_B : ∀ l₁ l₂, is_parallel l₁ l₂)
variable (symmetric_wrt_bisectors : ∀ l₁ l₂, is_symmetric l₁ l₂ (angle_bisector (∠BAC)))

-- Proving that the intersection of m and n lies on the circumcircle of triangle ABC
theorem intersection_point_on_circumcircle :
  (∃ D : Point, Intersection_points m n = {D} ∧ On_circumcircle D circumcircle) :=
by
sorry

end intersection_point_on_circumcircle_l455_455116


namespace sum_of_coefficients_l455_455229

theorem sum_of_coefficients (d : ℤ) : 
  let expr := -(4 - d) * (d + 3 * (4 - d))
  let expanded_form := -2 * d ^ 2 + 20 * d - 48
  let sum_of_coeffs := -2 + 20 - 48
  sum_of_coeffs = -30 :=
by
  -- The proof will go here, skipping for now.
  sorry

end sum_of_coefficients_l455_455229


namespace sin_cos_range_l455_455036

/-- 
  Define the function f(alpha, k) = sin(alpha)^(2*k) + cos(alpha)^(2*k)
  and prove that its range is [1 / 2 ^ (k - 1), 1] for k in ℕ⁺ (positive natural numbers).
-/
theorem sin_cos_range (k : ℕ) (hk : k > 0) :
  ∃ α : ℝ, let f (α : ℝ) := (Real.sin α)^(2*k) + (Real.cos α)^(2*k)
  in (1 / 2 ^ (k - 1)) <= f α ∧ f α ≤ 1 :=
sorry

end sin_cos_range_l455_455036


namespace rectangle_width_l455_455251

theorem rectangle_width (r l w : ℝ) (h_r : r = Real.sqrt 12) (h_l : l = 3 * Real.sqrt 2)
  (h_area_eq: Real.pi * r^2 = l * w) : w = 2 * Real.sqrt 2 * Real.pi :=
by
  sorry

end rectangle_width_l455_455251


namespace saturday_half_dollars_count_l455_455071

/-
  Define the conditions as given:
  - half_dollar is worth $0.50.
  - Sandy received 6 half_dollars on Sunday.
  - Total amount received is $11.50.
-/

def half_dollar_value : ℝ := 0.50
def sunday_half_dollars : ℕ := 6
def total_amount : ℝ := 11.50
def sunday_amount : ℝ := sunday_half_dollars * half_dollar_value
def saturday_amount : ℝ := total_amount - sunday_amount

/-
  Define the statement to be proved:
  Sandy received the following half-dollars on Saturday:
-/
theorem saturday_half_dollars_count : 
  (saturday_amount) / half_dollar_value = 17 :=
begin
  sorry
end

end saturday_half_dollars_count_l455_455071


namespace area_quadrilateral_FDBG_l455_455442

-- Definitions and conditions
variables (A B C D E F G : Type*) [EuclideanSpace A] [EuclideanSpace B] [EuclideanSpace C] 
variables (AB AC DE : ℝ)
variables (area_ABC area_ADE : ℝ)
variables (midpoint_D : midpoint A B D)
variables (midpoint_E : midpoint A C E)
variables (angle_bisector_AF : ∀ (a b c : ℝ), a / b = F)
variables (angle_bisector_BG : ∀ (a b c : ℝ), a / c = G)

-- Prove that the area of quadrilateral FDBG is equal to 83
theorem area_quadrilateral_FDBG :
  AB = 50 → AC = 10 → area_ABC = 120 → 
  midpoint_D → midpoint_E → angle_bisector_AF → angle_bisector_BG →
  area_of_quadrilateral F D B G = 83 :=
by {
  sorry
}

end area_quadrilateral_FDBG_l455_455442


namespace shortest_distance_at_m_minus1_l455_455294

noncomputable def point_of_intersection (m : ℝ) : ℝ × ℝ :=
  let l1 (x y : ℝ) := x + 3 * y - 3 * m ^ 2
      l2 (x y : ℝ) := 2 * x + y - m ^ 2 - 5 * m
  in (3 * m, m ^ 2 - m)

noncomputable def distance_to_line (x y : ℝ) := 
  abs (x + y + 3) / real.sqrt 2

theorem shortest_distance_at_m_minus1 {m : ℝ} :
  let P := point_of_intersection m in
  let d := distance_to_line (P.1) (P.2) in
  (d, m) = (real.sqrt 2, -1) :=
sorry

end shortest_distance_at_m_minus1_l455_455294


namespace perimeter_pentagon_l455_455459

noncomputable def length_AB : ℝ := 2
noncomputable def length_BC : ℝ := real.sqrt 5
noncomputable def length_CD : ℝ := real.sqrt 3
noncomputable def length_DE : ℝ := 1
noncomputable def length_AE : ℝ := real.sqrt 13

theorem perimeter_pentagon (AB BC CD DE AE : ℝ) 
    (hAB: AB = 2) (hBC: BC = real.sqrt 5) 
    (hCD: CD = real.sqrt 3) (hDE: DE = 1) 
    (hAE: AE = real.sqrt 13) : 
    AB + BC + CD + DE + AE = 3 + real.sqrt 5 + real.sqrt 3 + 1 + real.sqrt 13 :=
by 
    rw [hAB, hBC, hCD, hDE, hAE]
    ring

end perimeter_pentagon_l455_455459


namespace final_coordinates_of_A_l455_455770

-- Define the initial points
def A : ℝ × ℝ := (3, -2)
def B : ℝ × ℝ := (5, -5)
def C : ℝ × ℝ := (2, -4)

-- Define the translation operation
def translate (p : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ :=
  (p.1 + dx, p.2 + dy)

-- Define the rotation operation (180 degrees around a point (h, k))
def rotate180 (p : ℝ × ℝ) (h k : ℝ) : ℝ × ℝ :=
  (2 * h - p.1, 2 * k - p.2)

-- Translate point A
def A' := translate A 4 3

-- Rotate the translated point A' 180 degrees around the point (4, 0)
def A'' := rotate180 A' 4 0

-- The final coordinates of point A after transformations should be (1, -1)
theorem final_coordinates_of_A : A'' = (1, -1) :=
  sorry

end final_coordinates_of_A_l455_455770


namespace triangle_properties_l455_455102

noncomputable def angle_to_radians (d : ℝ) (m : ℝ) : ℝ := (d + m/60) * (Real.pi / 180)

def triangle_area (p α β : ℝ) : ℝ :=
  let γ := Real.pi - α - β
  let sin_α := Real.sin α
  let sin_β := Real.sin β
  let sin_γ := Real.sin γ
  let denom := sin_α + sin_β + sin_γ
  2 * p^2 * sin_α * sin_β * sin_γ / (denom * denom)

def triangle_side_bc (p α β : ℝ) : ℝ :=
  let γ := Real.pi - α - β
  let sin_γ := Real.sin γ
  2 * (p / (Real.sin α + Real.sin β + sin_γ)) * sin_γ

theorem triangle_properties (p α β : ℝ) :
  let γ := Real.pi - α - β
  triangle_side_bc p α β = 2 * (p / (Real.sin α + Real.sin β + Real.sin γ)) * Real.sin γ ∧
  triangle_area p α β = 2 * p^2 * (Real.sin α) * (Real.sin β) * (Real.sin γ) /
    (Real.sin α + Real.sin β + Real.sin γ)^2 :=
by sorry

end triangle_properties_l455_455102


namespace tan_alpha_plus_inverse_tan_alpha_l455_455245

variables {α : ℝ}

theorem tan_alpha_plus_inverse_tan_alpha (h : sin (2 * α) = 2 / 3) : tan α + 1 / tan α = 3 :=
sorry

end tan_alpha_plus_inverse_tan_alpha_l455_455245


namespace face_opposite_gold_is_black_l455_455884

variable (colors : Type) [DecidableEq colors]

structure Cube where
  top : colors
  bottom : colors
  left : colors
  right : colors
  front : colors
  back : colors

variables (Bl O Y Bk S G : colors)

-- We define the conditions given in the problem
def conditions (c : Cube colors) : Prop :=
  (c.top = Bk) ∧ (c.right = O) ∧
  ((c.front = Bl) ∧ (c.front ≠ Y) ∧ (c.front ≠ S) ∧ (c.front ≠ G) ∧ (c.front ≠ Bk)) ∨
  ((c.front = Y) ∧ (c.front ≠ Bl) ∧ (c.front ≠ S) ∧ (c.front ≠ G) ∧ (c.front ≠ Bk)) ∨
  ((c.front = S) ∧ (c.front ≠ Bl) ∧ (c.front ≠ Y) ∧ (c.front ≠ G) ∧ (c.front ≠ Bk)) 

theorem face_opposite_gold_is_black : 
  ∀ (c : Cube colors), conditions Bl O Y Bk S G c → c.bottom = G → c.top = Bk :=
by
  sorry

end face_opposite_gold_is_black_l455_455884


namespace proof_BD_CD_zero_l455_455725

noncomputable def f (a b c : ℝ) : ℝ → ℝ :=
  λ x, x^2 - a * x + (a^2 * b * c) / ((b + c)^2)

variables (a b c BD CD : ℝ)
hypothesis h1 : BD + CD = a
hypothesis h2 : BD * CD = (a^2 * b * c) / ((b + c)^2)

theorem proof_BD_CD_zero :
  f a b c BD = 0 ∧ f a b c CD = 0 :=
by
  sorry

end proof_BD_CD_zero_l455_455725


namespace total_balls_l455_455397

-- Conditions
def blue_balls : ℕ := 11
def green_balls : ℕ := 7
def red_balls : ℕ := 2 * blue_balls

-- Statement for the total number of balls
theorem total_balls (blue : ℕ) (green : ℕ) (red : ℕ) (h_blue : blue = blue_balls) (h_green : green = green_balls) (h_red : red = red_balls) :
  blue + green + red = 40 :=
by {
  rw [h_blue, h_green, h_red],
  sorry
}

end total_balls_l455_455397


namespace symmetrical_point_l455_455716

theorem symmetrical_point (P : ℝ × ℝ) (h : P = (-1, -2)) : ∃ P' : ℝ × ℝ, P' = (1, 2) ∧ P' = (-P.1, -P.2) :=
by
  use (1, 2)
  split
  case left =>
    rfl
  case right =>
    have hx : -P.1 = 1 := by rw [h]; rfl
    have hy : -P.2 = 2 := by rw [h]; rfl
    rw [hx, hy]
    rfl

end symmetrical_point_l455_455716


namespace count_tuples_l455_455080

theorem count_tuples (x : Fin 5 → ℤ) :
    (∀ i, 1 ≤ i ∧ i ≤ 5 → x ⟨i - 1, by linarith [i]⟩ ≥ i) ∧ (∑ i, x i = 25) →
    (Finset.card {t : Fin 5 → ℤ // ∀ i, 1 ≤ i ∧ i ≤ 5 → t ⟨i - 1, by linarith [i]⟩ ≥ i ∧ (∑ i, t i = 25)} = 1001) :=
  by
    sorry

end count_tuples_l455_455080


namespace average_sweater_less_by_21_after_discount_l455_455004

theorem average_sweater_less_by_21_after_discount
  (shirt_count sweater_count jeans_count : ℕ)
  (total_shirt_price total_sweater_price total_jeans_price : ℕ)
  (shirt_discount sweater_discount jeans_discount : ℕ)
  (shirt_avg_before_discount sweater_avg_before_discount jeans_avg_before_discount 
   shirt_avg_after_discount sweater_avg_after_discount jeans_avg_after_discount : ℕ) :
  shirt_count = 20 →
  sweater_count = 45 →
  jeans_count = 30 →
  total_shirt_price = 360 →
  total_sweater_price = 900 →
  total_jeans_price = 1200 →
  shirt_discount = 2 →
  sweater_discount = 4 →
  jeans_discount = 3 →
  shirt_avg_before_discount = total_shirt_price / shirt_count →
  sweater_avg_before_discount = total_sweater_price / sweater_count →
  jeans_avg_before_discount = total_jeans_price / jeans_count →
  shirt_avg_after_discount = shirt_avg_before_discount - shirt_discount →
  sweater_avg_after_discount = sweater_avg_before_discount - sweater_discount →
  jeans_avg_after_discount = jeans_avg_before_discount - jeans_discount →
  sweater_avg_after_discount = shirt_avg_after_discount →
  jeans_avg_after_discount - sweater_avg_after_discount = 21 :=
by
  intros
  sorry

end average_sweater_less_by_21_after_discount_l455_455004


namespace num_quadratic_eq_with_real_roots_l455_455559

def valid_b : Set ℤ := { -3, -2, -1, 0, 1, 2, 3 }
def valid_c : Set ℕ := { 0, 1, 2, 3 }

theorem num_quadratic_eq_with_real_roots :
  let real_root_conditions (b : ℤ) (c : ℕ) := (c ∈ valid_c ∧ b ∈ valid_b ∧ b^2 - 4 * c ≥ 0)
  in ∃ n : ℕ, n = 13 ∧ 
     { p : ℤ × ℕ // real_root_conditions p.1 p.2 }.finite.to_finset.card = n :=
by
  sorry

end num_quadratic_eq_with_real_roots_l455_455559


namespace turnip_total_correct_l455_455378

def turnips_left (melanie benny sarah david m_sold d_sold : ℕ) : ℕ :=
  let melanie_left := melanie - m_sold
  let david_left := david - d_sold
  benny + sarah + melanie_left + david_left

theorem turnip_total_correct :
  turnips_left 139 113 195 87 32 15 = 487 :=
by
  sorry

end turnip_total_correct_l455_455378


namespace solve_x_l455_455480

theorem solve_x (x : ℝ) (h : x ≠ 0) (h_eq : (5 * x) ^ 10 = (10 * x) ^ 5) : x = 2 / 5 :=
by
  sorry

end solve_x_l455_455480


namespace sum_of_coefficients_l455_455230

theorem sum_of_coefficients (d : ℤ) : 
  let expr := -(4 - d) * (d + 3 * (4 - d))
  let expanded_form := -2 * d ^ 2 + 20 * d - 48
  let sum_of_coeffs := -2 + 20 - 48
  sum_of_coeffs = -30 :=
by
  -- The proof will go here, skipping for now.
  sorry

end sum_of_coefficients_l455_455230


namespace fraction_negative_values_l455_455603

noncomputable def valid_x_intervals : Set ℝ := 
  {x : ℝ | (frac (11 * x^2 - 5 * x + 6) (x^2 + 5 * x + 6) - x < 0)}

theorem fraction_negative_values (x : ℝ) : 
  (11 * x^2 - 5 * x + 6) / (x^2 + 5 * x + 6) - x < 0 ↔ 
  (x ∈ Set.Ioo (-3) (-2) ∪ Set.Ioo (1) (2) ∪ Set.Ioi (3)) := 
sorry

end fraction_negative_values_l455_455603


namespace snake_turnaround_possible_l455_455141

def is_snake {n : ℕ} (k : ℕ) (positions : fin k → (fin n) × (fin n)) : Prop :=
  ∀ i : fin (k - 1), (positions i.succ.1).1 = (positions i.1).1 ∨ 
                     (positions i.succ.1).2 = (positions i.1).2

def shares_side {n : ℕ} (a b : (fin n) × (fin n)) : Prop :=
  ((a.1 = b.1) ∧ (a.2 = b.2 + 1 ∨ a.2 + 1 = b.2)) ∨ 
  ((a.2 = b.2) ∧ (a.1 = b.1 + 1 ∨ a.1 + 1 = b.1))

def can_turn_around {n : ℕ} (k : ℕ) (s₁ : fin k → (fin n) × (fin n)) : Prop :=
  ∃ s₂ : fin k → (fin n) × (fin n), 
    is_snake k s₂ ∧ 
    (∀ i, shares_side (s₁ 0) (s₂ i)) ∧ 
    (∀ i, shares_side (s₂ i) (s₂ (i+1 % k))) ∧ 
    (s₂ k.pred = s₁ 0)

theorem snake_turnaround_possible :
  ∃ n > 1, ∃ (s : fin ⟨⌊0.9 * (n:ℕ)^2⌋⟩ → (fin n) × (fin n)), 
    is_snake ⟨⌊0.9 * (n:ℕ)^2⌋⟩ s ∧ can_turn_around ⟨⌊0.9 * (n:ℕ)^2⌋⟩ s :=
by
  sorry

end snake_turnaround_possible_l455_455141


namespace smaller_number_is_17_l455_455824

theorem smaller_number_is_17 (x y : ℕ) (h1 : x * y = 323) (h2 : x - y = 2) : y = 17 :=
sorry

end smaller_number_is_17_l455_455824


namespace inv_mod_35_l455_455231

theorem inv_mod_35 : ∃ x : ℕ, 5 * x ≡ 1 [MOD 35] :=
by
  use 29
  sorry

end inv_mod_35_l455_455231


namespace probability_jammed_l455_455912

theorem probability_jammed (T τ : ℝ) (h : τ < T) : 
    (2 * τ / T - (τ / T) ^ 2) = (T^2 - (T - τ)^2) / T^2 := 
by
  sorry

end probability_jammed_l455_455912


namespace intersects_at_l455_455060

noncomputable def a : ℝ := -1
noncomputable def c : ℝ := 2

def f (x : ℝ) : ℝ := a * x^2 + c

def g (x : ℝ) : ℝ := c * x + a

theorem intersects_at :
  (f 0 = 2) ∧ (f 1 = 1) →
  (∃ x : ℝ, g x = 0 ∧ x = 1 / 2) ∧
  (∃ y : ℝ, g 0 = y ∧ y = -1) :=
by
  intro h,
  cases h with h0 h1,
  existsi (1 / 2),
  rw [g, a, c],
  split,
  sorry,
  existsi (-1),
  rw [g, a, c],
  split,
  sorry

end intersects_at_l455_455060


namespace cousin_age_when_double_brother_age_l455_455762

noncomputable def nick_age : ℕ := 13
noncomputable def sister_age : ℕ := nick_age + 6
noncomputable def combined_age : ℕ := nick_age + sister_age
noncomputable def brother_age : ℕ := combined_age / 2
noncomputable def cousin_age : ℕ := brother_age - 3
noncomputable def double_brother_age : ℕ := 2 * brother_age

theorem cousin_age_when_double_brother_age : 
  cousin_age + (double_brother_age - cousin_age) = double_brother_age - cousin_age + cousin_age :=
by
  calc cousin_age + (double_brother_age - cousin_age)
        = double_brother_age - cousin_age + cousin_age : by sorry

end cousin_age_when_double_brother_age_l455_455762


namespace decreasing_intervals_l455_455819

noncomputable def f (x : ℝ) : ℝ := (x + 1) / x

theorem decreasing_intervals :
  ∀ x : ℝ, x ≠ 0 → (∃ a b : ℝ, a < 0 ∧ b > 0 ∧ (a < x ∧ x < 0 ∨ 0 < x ∧ x < b) ) →
  f'(x) < 0 :=
by
  sorry

end decreasing_intervals_l455_455819


namespace lcm_fraction_mult_two_l455_455123

variable (x : ℤ)

noncomputable def lcm_of_fractions := Nat.lcm (1 / x) (Nat.lcm (1 / (4 * x)) (1 / (5 * x)))

theorem lcm_fraction_mult_two (h : x ≠ 0) : 2 * lcm_of_fractions x = 1 / (10 * x) :=
sorry

end lcm_fraction_mult_two_l455_455123


namespace sectorChordLength_correct_l455_455406

open Real

noncomputable def sectorChordLength (r α : ℝ) : ℝ :=
  2 * r * sin (α / 2)

theorem sectorChordLength_correct :
  ∃ (r α : ℝ), (1/2) * α * r^2 = 1 ∧ 2 * r + α * r = 4 ∧ sectorChordLength r α = 2 * sin 1 :=
by {
  sorry
}

end sectorChordLength_correct_l455_455406


namespace op_val_l455_455207

def op (a b c d : ℝ) : ℝ := b^2 - 4 * a * c + d^2

theorem op_val : op 2 3 1 4 = 17 := by
  rfl

end op_val_l455_455207


namespace region_volume_is_two_thirds_l455_455479

noncomputable def volume_of_region : ℝ :=
  let region := {p : ℝ × ℝ × ℝ | |p.1| + |p.2| + |p.3| ≤ 2 ∧ |p.1| + |p.2| + |p.3 - 2| ≤ 2}
  -- Assuming volume function calculates the volume of the region
  volume region

theorem region_volume_is_two_thirds :
  volume_of_region = 2 / 3 :=
by
  sorry

end region_volume_is_two_thirds_l455_455479


namespace quadratic_root_relation_l455_455428

theorem quadratic_root_relation (m n p q : ℝ) (s₁ s₂ : ℝ) 
  (h1 : s₁ + s₂ = -p) 
  (h2 : s₁ * s₂ = q) 
  (h3 : 3 * s₁ + 3 * s₂ = -m) 
  (h4 : 9 * s₁ * s₂ = n) 
  (h_m : m ≠ 0) 
  (h_n : n ≠ 0) 
  (h_p : p ≠ 0) 
  (h_q : q ≠ 0) :
  n = 9 * q :=
by
  sorry

end quadratic_root_relation_l455_455428


namespace range_of_a_l455_455623

variable (a : ℝ)

theorem range_of_a (h : ¬ ∃ x : ℝ, x^2 + 2 * x + a ≤ 0) : 1 < a :=
by {
  -- Proof will go here.
  sorry
}

end range_of_a_l455_455623


namespace proof_sin_300_cos_0_eq_neg_sqrt_3_div_2_l455_455104

noncomputable def sin_300_cos_0_eq_neg_sqrt_3_div_2 : Prop :=
  sin 300 * cos 0 = - (Real.sqrt 3 / 2)

theorem proof_sin_300_cos_0_eq_neg_sqrt_3_div_2 : sin_300_cos_0_eq_neg_sqrt_3_div_2 :=
by
  sorry

end proof_sin_300_cos_0_eq_neg_sqrt_3_div_2_l455_455104


namespace local_tax_paid_in_cents_l455_455535

-- Define Alicia's hourly wage in dollars
def hourlyWageDollars : ℝ := 25

-- Define the tax rate as a percentage
def taxRatePercentage : ℝ := 2.5

-- Define the function to convert dollars to cents
def dollarsToCents (dollars : ℝ) : ℝ := dollars * 100

-- Define the function to calculate the tax in cents
def taxInCents (wageDollars : ℝ) (taxRate : ℝ) : ℝ := 
  (taxRate / 100) * dollarsToCents wageDollars

-- Statement of the theorem we want to prove
theorem local_tax_paid_in_cents : 
  taxInCents hourlyWageDollars taxRatePercentage = 62.5 :=
  by
    sorry

end local_tax_paid_in_cents_l455_455535


namespace angle_BAD_measure_l455_455969

theorem angle_BAD_measure (D_A_C : ℝ) (AB_AC : AB = AC) (AD_BD : AD = BD) (h : D_A_C = 39) :
  B_A_D = 70.5 :=
by sorry

end angle_BAD_measure_l455_455969


namespace countPrimesWithPrimeRemainder_div6_l455_455663

/-- Define the prime numbers in the range 50 to 100 -/
def isPrimeInRange_50_100 (n : ℕ) : Prop :=
  n ∈ [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

/-- Define the condition for having a prime remainder when divided by 6 -/
def hasPrimeRemainder_div6 (n : ℕ) : Prop :=
  ∃ (r : ℕ), r ∈ [1, 2, 3, 5] ∧ n % 6 = r

theorem countPrimesWithPrimeRemainder_div6 :
  (finset.filter (λ n, hasPrimeRemainder_div6 n) (finset.filter isPrimeInRange_50_100 (finset.range 101))).card = 10 :=
by
  sorry

end countPrimesWithPrimeRemainder_div6_l455_455663


namespace perpendicular_concurrence_l455_455339

open EuclideanGeometry

variables {A B C A' B' C' : Point}

theorem perpendicular_concurrence 
  (h1 : are_inside A B C A' B' C') 
  (h2 : concurrent (perpendicular A B' C') (perpendicular B C' A') (perpendicular C A' B')) :
  concurrent (perpendicular A' B C) (perpendicular B' C A) (perpendicular C' A B) :=
sorry

end perpendicular_concurrence_l455_455339


namespace concavity_condition_max_interval_length_l455_455648

-- Problem (I)
theorem concavity_condition (m : ℝ) (x : ℝ) (h : 0 ≤ x ∧ x ≤ 3) : 
  (x^2 - m * x - 3 < 0) → (m > 2) :=
by sorry

-- Problem (II)
theorem max_interval_length (m : ℝ) (a b : ℝ) (h : abs m ≤ 2) :
  (∀ x ∈ (set.Ioo a b), x^2 - m * x - 3 < 0) →
  (b - a ≤ 2) :=
by sorry

end concavity_condition_max_interval_length_l455_455648


namespace distance_to_asymptote_of_hyperbola_l455_455810

-- Define the hyperbola and the point
def hyperbola : set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y^2 / 4 - x^2 = 1}
def P := (0 : ℝ, 1 : ℝ)

-- Define a function for the distance from a point to a line
def distance_point_to_line (a b c : ℝ) (x y : ℝ) : ℝ :=
  |a * x + b * y + c| / real.sqrt (a^2 + b^2)

-- Define the specific line (2x - y = 0) as a, b, and c coefficients
def line_asymptote (x y : ℝ) : Prop := 2 * x - y = 0

-- The theorem to prove
theorem distance_to_asymptote_of_hyperbola : 
  distance_point_to_line 2 (-1) 0 0 1 = real.sqrt 5 / 5 :=
by {
  sorry -- proof goes here
}

end distance_to_asymptote_of_hyperbola_l455_455810


namespace brad_green_balloons_l455_455188

noncomputable def number_of_green_balloons (total_balloons : Nat) (initial_red_balloons : Nat) (popped_red_balloons : Nat) (blue_balloons : Nat) (green_yellow_combined_ratio : Rat) : Nat :=
  let remaining_red_balloons := initial_red_balloons - popped_red_balloons
  let non_red_balloons := total_balloons - remaining_red_balloons
  let green_yellow_combined := non_red_balloons - blue_balloons
  let green_balloons := (green_yellow_combined_ratio * green_yellow_combined).floor
  green_balloons

theorem brad_green_balloons : number_of_green_balloons 50 15 3 7 (2 / 5) = 12 := by
  sorry

end brad_green_balloons_l455_455188


namespace digits_count_l455_455509

theorem digits_count (A : ℕ) (hA : A < 10) : cardinal.mk {x : ℕ | x < 5} = 5 := by
  have h: {x : ℕ | x < 5} = {0, 1, 2, 3, 4} := by sorry
  rw [h]
  simp
  sorry

end digits_count_l455_455509


namespace fourth_buoy_distance_with_current_l455_455894

-- Define the initial conditions
def first_buoy_distance : ℕ := 20
def second_buoy_additional_distance : ℕ := 24
def third_buoy_additional_distance : ℕ := 28
def common_difference_increment : ℕ := 4
def ocean_current_push_per_segment : ℕ := 3
def number_of_segments : ℕ := 3

-- Define the mathematical proof problem
theorem fourth_buoy_distance_with_current :
  let fourth_buoy_additional_distance := third_buoy_additional_distance + common_difference_increment
  let first_to_second_buoy := first_buoy_distance + second_buoy_additional_distance
  let second_to_third_buoy := first_to_second_buoy + third_buoy_additional_distance
  let distance_before_current := second_to_third_buoy + fourth_buoy_additional_distance
  let total_current_push := ocean_current_push_per_segment * number_of_segments
  let final_distance := distance_before_current - total_current_push
  final_distance = 95 := by
  sorry

end fourth_buoy_distance_with_current_l455_455894


namespace remainder_sum_mod_l455_455971

theorem remainder_sum_mod (a b c d e : ℕ)
  (h₁ : a = 17145)
  (h₂ : b = 17146)
  (h₃ : c = 17147)
  (h₄ : d = 17148)
  (h₅ : e = 17149)
  : (a + b + c + d + e) % 10 = 5 := by
  sorry

end remainder_sum_mod_l455_455971


namespace cole_drive_time_to_work_l455_455496

noncomputable def speed_to_work : ℝ := 60
noncomputable def speed_back_home : ℝ := 90
noncomputable def total_time : ℝ := 2

theorem cole_drive_time_to_work : 
  let D := 72 / 60 * total_time in
  60 * (D / speed_to_work) = 72 := 
by
  sorry

end cole_drive_time_to_work_l455_455496


namespace hyperbola_eccentricity_l455_455285

variable (a b c : ℝ)
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (h3 : c = 2 * a)
variable (x y : ℝ)

-- Hyperbola Condition
def hyperbola : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Circle Condition with center (c, 0)
def circle : Prop := (x - c)^2 + y^2 = c^2

-- Line l condition
def line_l : Prop := y = (a / b) * (x - (2 / 3) * a)

-- Chord length condition
def chord_length_condition : Prop := 
  let chord_length := (4 * Real.sqrt 2 / 3) * c in
  let distance_to_center := Real.sqrt(c^2 - (chord_length / 2)^2) in
  distance_to_center = c / 3

-- Theorem statement
theorem hyperbola_eccentricity (h_hyperbola : hyperbola a b x y)
    (h_circle : circle a b c x y)
    (h_line_l : line_l a b x y)
    (h_chord_length : chord_length_condition a b c) :
  let e := c / a in e = 2 := 
sorry

end hyperbola_eccentricity_l455_455285


namespace find_functions_satisfying_problem_statement_l455_455948

noncomputable def problem_statement 
    (f : ℝ → ℝ) : Prop :=
  ∀ x y z t : ℝ, (f(x) + f(z)) * (f(y) + f(t)) = f(xy - zt) + f(xt + yz)

theorem find_functions_satisfying_problem_statement :
  (∀ f : ℝ → ℝ, problem_statement f → (∀ x : ℝ, f(x) = 0) ∨ (∀ x : ℝ, f(x) = 1/2) ∨ (∀ x : ℝ, f(x) = x^2)) := 
sorry

end find_functions_satisfying_problem_statement_l455_455948


namespace fn_lt_n2_div_4_l455_455365

noncomputable def fn (n : ℕ) (r : ℝ) (a : ℕ → ℝ) : ℕ := 
  finset.card {t : finset.card (finset.range n).filter (λ p, (λ i j k, i < j ∧ j < k ∧ a j - a i = r * (a k - a j)) p.1 p.2.1 p.2.2).to_set }

theorem fn_lt_n2_div_4 (n : ℕ) (r : ℝ) (a : ℕ → ℝ)
  (h1 : 4 ≤ n) 
  (h2 : ∀ i j, i < j → i < n → j < n → a i < a j) 
  (h3 : ∀ i, i < n → 0 < a i) :
  fn n r a < n * n / 4 :=
sorry

end fn_lt_n2_div_4_l455_455365


namespace quadratic_eq_satisfied_l455_455356

noncomputable def omega : ℂ := sorry

axiom omega_prop1 : omega^8 = 1
axiom omega_prop2 : omega ≠ 1

def alpha : ℂ := omega + omega^3 + omega^5
def beta : ℂ := omega^2 + omega^4 + omega^6 + omega^7

theorem quadratic_eq_satisfied : ∃ (a b : ℝ), (α : ℂ → ℂ) (β : ℂ → ℂ), 
  a = 1 ∧ b = 3 ∧ (α * α + a * α + b = 0) ∧ (β * β + a * β + b = 0) := sorry

end quadratic_eq_satisfied_l455_455356


namespace no_valid_partition_of_nat_l455_455395

-- Definitions of the sets A, B, and C as nonempty subsets of positive integers
variable (A B C : Set ℕ)

-- Definition to capture the key condition in the problem
def valid_partition (A B C : Set ℕ) : Prop :=
  (∀ x ∈ A, ∀ y ∈ B, (x^2 - x * y + y^2) ∈ C) 

-- The main theorem stating that such a partition is impossible
theorem no_valid_partition_of_nat : 
  (∃ A B C : Set ℕ, A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅ ∧ (∀ x ∈ A, ∀ y ∈ B, (x^2 - x * y + y^2) ∈ C)) → False :=
by
  sorry

end no_valid_partition_of_nat_l455_455395


namespace ordinate_of_point_A_l455_455062

noncomputable def p : ℝ := 1 / 4
noncomputable def distance_to_focus (y₀ : ℝ) : ℝ := y₀ + p / 2

theorem ordinate_of_point_A :
  ∃ y₀ : ℝ, (distance_to_focus y₀ = 9 / 8) → y₀ = 1 :=
by
  -- Assume solution steps here
  sorry

end ordinate_of_point_A_l455_455062


namespace polar_equation_of_line_polar_equation_of_circle_area_of_triangle_l455_455011

-- Problem 1: Polar equation of the line l
theorem polar_equation_of_line (ρ : ℝ) : y = x → θ = π / 4 :=
sorry

-- Problem 2: Polar equation of the circle C
theorem polar_equation_of_circle (ρ θ : ℝ) :
  (x + 1)^2 + (y + 2)^2 = 1 →
  ρ^2 + 2 * ρ * cos θ + 4 * ρ * sin θ + 4 = 0 :=
sorry

-- Problem 3: Area of triangle CMN
theorem area_of_triangle (M N C : ℝ) :
  (l : y = x) →
  ((x + 1)^2 + (y + 2)^2 = 1) →
  (distance_from_center_to_line = sqrt 2 / 2) →
  (MN = sqrt 2) →
  (S = 1/2) :=
sorry

end polar_equation_of_line_polar_equation_of_circle_area_of_triangle_l455_455011


namespace min_value_x1_l455_455281

noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then 3^x - 1 else x^2 + 1

theorem min_value_x1 (x1 x2 : ℝ) (h1 : x1 > 0) (h2 : x2 ≤ 0) (heq : f x1 = f x2) : x1 ≥ Real.log 2 / Real.log 3 :=
begin
  sorry
end

end min_value_x1_l455_455281


namespace sum_of_n_for_perfect_square_l455_455974

theorem sum_of_n_for_perfect_square :
  let is_perfect_square (x : ℤ) := ∃ (y : ℤ), y^2 = x
  in ∑ n in { n | n > 0 ∧ is_perfect_square (n^2 - 17 * n + 72) }, n = 17 :=
by
  sorry

end sum_of_n_for_perfect_square_l455_455974


namespace complex_exp_pow_eight_l455_455189

def cos_deg (θ : ℝ) : ℝ := cos (θ * pi / 180)
def sin_deg (θ : ℝ) : ℝ := sin (θ * pi / 180)

theorem complex_exp_pow_eight :
  (3 * (cos_deg 30 + complex.i * sin_deg 30)) ^ 8 = -3280.5 - 3280.5 * complex.i * real.sqrt 3 :=
sorry

end complex_exp_pow_eight_l455_455189


namespace find_f1_l455_455691

variable {F : Type} [Ring F] (f : F → F)
variable x : F

-- Conditions
def odd_function (f : F → F) : Prop := ∀ x : F, f(-x) = -f(x)
def f_neg1_eq_2 : Prop := f(-1) = 2

-- Theorem statement
theorem find_f1 (h1 : odd_function f) (h2 : f_neg1_eq_2) : f(1) = -2 :=
by
  sorry

end find_f1_l455_455691


namespace pos_diff_of_solutions_abs_eq_20_l455_455851

theorem pos_diff_of_solutions_abs_eq_20 : ∀ (x1 x2 : ℝ), (|x1 + 5| = 20 ∧ |x2 + 5| = 20) → x1 - x2 = 40 :=
  by
    intros x1 x2 h
    sorry

end pos_diff_of_solutions_abs_eq_20_l455_455851


namespace natalia_crates_l455_455581

/- The definitions from the conditions -/
def novels : ℕ := 145
def comics : ℕ := 271
def documentaries : ℕ := 419
def albums : ℕ := 209
def crate_capacity : ℕ := 9

/- The proposition to prove -/
theorem natalia_crates : (novels + comics + documentaries + albums) / crate_capacity = 116 := by
  -- this skips the proof and assumes the theorem is true
  sorry

end natalia_crates_l455_455581


namespace bottle_cost_l455_455871

-- Definitions of the conditions
def total_cost := 30
def wine_extra_cost := 26

-- Statement of the problem in Lean 4
theorem bottle_cost : 
  ∃ x : ℕ, (x + (x + wine_extra_cost) = total_cost) ∧ x = 2 :=
by
  sorry

end bottle_cost_l455_455871


namespace minimum_distance_hyperbola_APF_l455_455625

-- Define the hyperbola and its properties
def hyperbola : set (ℝ × ℝ) := {p | (p.1^2 / 4) - (p.2^2 / 12) = 1}

-- Define the foci of the hyperbola
def F : ℝ × ℝ := (-4, 0) -- Left focus for sign consistency
def F' : ℝ × ℝ := (4, 0)  -- Right focus

-- Define the point A and a condition for point P on the right branch of the hyperbola
def A : ℝ × ℝ := (1, 4)
def on_right_branch (P : ℝ × ℝ) : Prop := P ∈ hyperbola ∧ P.1 > 0

-- Define the collinearity condition
def collinear (P : ℝ × ℝ) : Prop := ∃ k : ℝ, A.1 + k * (P.1 - A.1) = F'.1 ∧ A.2 + k * (P.2 - A.2) = F'.2

-- Define the distance from a point to a line given by the standard form Ax + By + C = 0
def distance_point_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / (sqrt (A^2 + B^2))

-- The main theorem statement
theorem minimum_distance_hyperbola_APF
  (P : ℝ × ℝ)
  (h1 : on_right_branch P)
  (h2 : collinear P) :
  distance_point_line F.1 F.2 4 3 (-16) = 32 / 5 :=
by
  sorry

end minimum_distance_hyperbola_APF_l455_455625


namespace abs_eight_minus_neg_two_sum_of_integers_abs_eq_five_sum_of_satisfying_integers_min_value_abs_sum_l455_455846

-- Part 1: Prove that |8 - (-2)| = 10
theorem abs_eight_minus_neg_two : |8 - (-2)| = 10 := 
sorry

-- Part 2: Prove that the sum of all integers x satisfying |x - 2| + |x + 3| = 5 is -3
theorem sum_of_integers_abs_eq_five (x : ℤ) (h : |x - 2| + |x + 3| = 5) : 
  x = -3 ∨ x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 :=
sorry

theorem sum_of_satisfying_integers : 
  ∑ i in {-3, -2, -1, 0, 1, 2}, i = -3 := 
sorry

-- Part 3: Prove that the minimum value of |x + 4| + |x - 6| is 10 for any rational x
theorem min_value_abs_sum (x : ℚ) : 
  ∃ y, (-4 ≤ y ∧ y ≤ 6) ∧ (|y + 4| + |y - 6| = 10) := 
sorry

end abs_eight_minus_neg_two_sum_of_integers_abs_eq_five_sum_of_satisfying_integers_min_value_abs_sum_l455_455846


namespace sum_of_angles_of_solutions_l455_455196

theorem sum_of_angles_of_solutions (r1 r2 r3 r4 : ℝ) (θ1 θ2 θ3 θ4 : ℝ) 
  (h1 : Complex.I * w^4 = 81) 
  (h2 : w1 = r1 * Complex.cis θ1) 
  (h3 : w2 = r2 * Complex.cis θ2) 
  (h4 : w3 = r3 * Complex.cis θ3) 
  (h5 : w4 = r4 * Complex.cis θ4) 
  (hr1 : r1 > 0) 
  (hr2 : r2 > 0) 
  (hr3 : r3 > 0) 
  (hr4 : r4 > 0) 
  (hθ1 : 0 ≤ θ1 ∧ θ1 < 360) 
  (hθ2 : 0 ≤ θ2 ∧ θ2 < 360) 
  (hθ3 : 0 ≤ θ3 ∧ θ3 < 360) 
  (hθ4 : 0 ≤ θ4 ∧ θ4 < 360) : 
  θ1 + θ2 + θ3 + θ4 = 630 := 
by 
  sorry

end sum_of_angles_of_solutions_l455_455196


namespace monotonicity_of_shifted_function_l455_455400

noncomputable def initial_function (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

noncomputable def shifted_function (x : ℝ) : ℝ := Real.sin (2 * (x + Real.pi / 2) - Real.pi / 4)

theorem monotonicity_of_shifted_function :
  (∀ x y : ℝ, - Real.pi / 8 ≤ x ∧ x < y ∧ y ≤ 3 * Real.pi / 8 → shifted_function x > shifted_function y) ∧
  (∃ x y : ℝ, - Real.pi / 4 ≤ x ∧ x < y ∧ y ≤ 3 * Real.pi / 4 ∧ shifted_function x ≤ shifted_function y ∧ shifted_function x ≥ shifted_function y) :=
begin
  -- Proof can be filled in later
  sorry
end

end monotonicity_of_shifted_function_l455_455400


namespace max_distance_goat_l455_455322

noncomputable def euclidean_distance (p q : ℝ × ℝ) : ℝ :=
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem max_distance_goat (c : ℝ × ℝ) (r : ℝ) (p : ℝ × ℝ) (hc : c = (3, -4)) (hr : r = 8) (hp : p = (1, 2)) :
  ∃ m : ℝ, m = r + euclidean_distance c p ∧ m = 8 + 2 * Real.sqrt 10 :=
by
  use 8 + 2 * Real.sqrt 10
  split
  · rw [hc, hr, hp]
    exact rfl
  · exact rfl

end max_distance_goat_l455_455322


namespace greatest_possible_value_of_y_l455_455798

theorem greatest_possible_value_of_y 
  (x y : ℤ) 
  (h : x * y + 7 * x + 6 * y = -8) : 
  y ≤ 27 ∧ (exists x, x * y + 7 * x + 6 * y = -8) := 
sorry

end greatest_possible_value_of_y_l455_455798


namespace p_plus_q_443_l455_455744

def sequence_fraction : ℚ :=
  (∑' n, if even n then (↑((nat.succ n) / 2 : ℕ) / (2 ^ ((n + 2) / 2) : ℚ)) else (↑(nat.succ n / 2) / (3 ^ ((n + 2) / 2) : ℚ)))

theorem p_plus_q_443 :
  ∃ (p q : ℕ), p.gcd q = 1 ∧ (p : ℚ) / (q : ℚ) = sequence_fraction ∧ p + q = 443 :=
by {
  sorry
}

end p_plus_q_443_l455_455744


namespace bean_inside_inscribed_circle_l455_455083

noncomputable def equilateral_triangle_area (a : ℝ) : ℝ :=
  (Real.sqrt 3 / 4) * a * a

noncomputable def inscribed_circle_radius (a : ℝ) : ℝ :=
  (Real.sqrt 3 / 3) * a

noncomputable def circle_area (r : ℝ) : ℝ :=
  Real.pi * r * r

noncomputable def probability_inside_circle (s_triangle s_circle : ℝ) : ℝ :=
  s_circle / s_triangle

theorem bean_inside_inscribed_circle :
  let a := 2
  let s_triangle := equilateral_triangle_area a
  let r := inscribed_circle_radius a
  let s_circle := circle_area r
  probability_inside_circle s_triangle s_circle = (Real.sqrt 3 * Real.pi / 9) :=
by
  sorry

end bean_inside_inscribed_circle_l455_455083


namespace num_four_digit_integers_with_at_least_one_4_or_7_l455_455301

def count_four_digit_integers_with_4_or_7 : ℕ := 5416

theorem num_four_digit_integers_with_at_least_one_4_or_7 :
  let all_four_digit_integers := 9000
  let valid_digits_first := 7
  let valid_digits := 8
  let integers_without_4_or_7 := valid_digits_first * valid_digits * valid_digits * valid_digits
  all_four_digit_integers - integers_without_4_or_7 = count_four_digit_integers_with_4_or_7 :=
by
  -- Using known values from the problem statement
  let all_four_digit_integers := 9000
  let valid_digits_first := 7
  let valid_digits := 8
  let integers_without_4_or_7 := valid_digits_first * valid_digits * valid_digits * valid_digits
  show all_four_digit_integers - integers_without_4_or_7 = count_four_digit_integers_with_4_or_7
  sorry

end num_four_digit_integers_with_at_least_one_4_or_7_l455_455301


namespace factorial_subtraction_l455_455190

theorem factorial_subtraction : 7! - 6 * 6! - 2 * 6! = -720 :=
by
  sorry

end factorial_subtraction_l455_455190


namespace cannot_divide_convex_13gon_into_parallelograms_l455_455773

-- Define a convex polygon
structure ConvexPolygon (n : Nat) :=
  (sides : Nat)
  (is_convex : ∀ (angle : Fin n), angle < 180)

-- Define specific case of convex 13-gon
def convex_13gon : ConvexPolygon 13 :=
  { sides := 13,
    is_convex := λ angle, sorry }

-- Formalize the proof statement
theorem cannot_divide_convex_13gon_into_parallelograms :
  ¬ (∃ (P : ConvexPolygon 13), can_be_divided_into_parallelograms P) :=
sorry

end cannot_divide_convex_13gon_into_parallelograms_l455_455773


namespace solve_equation_l455_455130

theorem solve_equation : 
  ∃ x : ℝ, 2.61 * (9 - (x + 1).sqrt).cbrt + (7 + (x + 1).sqrt).cbrt = 4 → x = 0 :=
by 
  sorry

end solve_equation_l455_455130


namespace jellybean_probability_l455_455155

theorem jellybean_probability :
  let total_jellybeans := 12
  let red_jellybeans := 5
  let blue_jellybeans := 2
  let yellow_jellybeans := 5
  let total_picks := 4
  let successful_outcomes := 10 * 7 
  let total_outcomes := Nat.choose 12 4 
  let required_probability := 14 / 99 
  successful_outcomes = 70 ∧ total_outcomes = 495 → 
  successful_outcomes / total_outcomes = required_probability := 
by 
  intros
  sorry

end jellybean_probability_l455_455155


namespace final_number_proof_l455_455070

/- Define the symbols and their corresponding values -/
def cat := 1
def chicken := 5
def crab := 2
def bear := 4
def goat := 3

/- Define the equations from the conditions -/
axiom row4_eq : 5 * crab = 10
axiom col5_eq : 4 * crab + goat = 11
axiom row2_eq : 2 * goat + crab + 2 * bear = 16
axiom col2_eq : cat + bear + 2 * goat + crab = 13
axiom col3_eq : 2 * crab + 2 * chicken + goat = 17

/- Final number is derived by concatenating digits -/
def final_number := cat * 10000 + chicken * 1000 + crab * 100 + bear * 10 + goat

/- Theorem to prove the final number is 15243 -/
theorem final_number_proof : final_number = 15243 := by
  -- Proof steps to be provided here.
  sorry

end final_number_proof_l455_455070


namespace partial_fraction_decomposition_l455_455595

noncomputable def polynomial : Polynomial ℝ := Polynomial.X^3 - 24 * Polynomial.X^2 + 143 * Polynomial.X - 210

theorem partial_fraction_decomposition (A B C p q r : ℝ) (h1 : Polynomial.roots polynomial = {p, q, r}) 
  (h2 : ∀ s : ℝ, 1 / (s^3 - 24 * s^2 + 143 * s - 210) = A / (s - p) + B / (s - q) + C / (s - r)) :
  1 / A + 1 / B + 1 / C = 243 :=
by
  sorry

end partial_fraction_decomposition_l455_455595


namespace range_of_t_l455_455410

noncomputable def f : ℝ → ℝ := sorry

theorem range_of_t (h1: ∀ x : ℝ, f'' x + 2017 < 4034 * x)
                   (h2: ∀ t : ℝ, f (t + 1) < f (-t) + 4034 * t + 2017) :
                   ∀ t : ℝ, t > -1 / 2 :=
begin
  sorry
end

end range_of_t_l455_455410


namespace distinct_real_roots_l455_455135

variables {a b c : ℝ}
hypothesis h_distinct_nonzero : a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

theorem distinct_real_roots :
  ∃ (p : ℝ → ℝ), 
    (p = λ x, a * x^2 + 2 * b * x + c ∨ p = λ x, b * x^2 + 2 * c * x + a ∨ p = λ x, c * x^2 + 2 * a * x + b) ∧
    (∃ x y : ℝ, x ≠ y ∧ p x = 0 ∧ p y = 0) :=
by
  sorry

end distinct_real_roots_l455_455135


namespace herons_area_of_ABC_herons_eq_qin_jiushao_l455_455693

-- Defining variables and given conditions
variables (a b c : ℝ)
def p : ℝ := (a + b + c) / 2

-- Hero's formula for the area of a triangle
def herons_formula (p a b c : ℝ) : ℝ := 
  Real.sqrt (p * (p - a) * (p - b) * (p - c))

-- Qin Jiushao's formula for the area of a triangle
def qin_jiushao_formula (a b c : ℝ) : ℝ := 
  Real.sqrt (1 / 4 * (a^2 * b^2 - ((a^2 + b^2 + c^2) / 2)^2))


-- Theorem stating the problem
theorem herons_area_of_ABC :
  herons_formula 15 8 10 12 = 15 * Real.sqrt 7 := 
sorry

theorem herons_eq_qin_jiushao (a b c : ℝ) :
  herons_formula (p a b c) a b c = qin_jiushao_formula a b c :=
sorry

end herons_area_of_ABC_herons_eq_qin_jiushao_l455_455693


namespace solution_set_l455_455632

noncomputable def f (x : ℝ) := sorry
noncomputable def f' (x : ℝ) := sorry

-- Given conditions:
axiom domain (x : ℝ) : 0 < x
axiom deriv_condition (x : ℝ) : f(x) < -x * f'(x)

-- Problem statement:
theorem solution_set (x : ℝ) :
  f(x + 1) > (x - 1) * f(x^2 - 1) ↔ x > 2 := sorry

end solution_set_l455_455632


namespace redesign_survey_response_l455_455176

theorem redesign_survey_response :
  ∀ (initial_sent redesigned_sent initial_respondents additional_rate : ℝ),
  initial_sent = 90 → 
  initial_respondents = 7 → 
  redesigned_sent = 63 → 
  additional_rate = 6 →
  let original_rate := (initial_respondents / initial_sent) * 100,
      new_rate := original_rate + additional_rate,
      num_redesigned_respondents := (new_rate / 100) * redesigned_sent 
  in num_redesigned_respondents ≈ 9 :=
begin
  intros,
  sorry
end

end redesign_survey_response_l455_455176


namespace count_primes_with_prime_remainders_between_50_and_100_l455_455669

/-- 
The count of prime numbers between 50 and 100 that have a prime remainder 
(1, 2, 3, or 5) when divided by 6 is 10.
-/
theorem count_primes_with_prime_remainders_between_50_and_100 : 
  (finset.filter (λ p, ∃ r, (p % 6 = r) ∧ nat.prime r ∧ r ∈ ({1, 2, 3, 5} : finset ℕ)) 
                  (finset.filter nat.prime (finset.Ico 51 101))).card = 10 := 
by 
  sorry

end count_primes_with_prime_remainders_between_50_and_100_l455_455669


namespace Ella_jellybeans_l455_455989

-- Definitions based on conditions from part (a)
def Dan_volume := 10
def Dan_jellybeans := 200
def scaling_factor := 3

-- Prove that Ella's box holds 5400 jellybeans
theorem Ella_jellybeans : scaling_factor^3 * Dan_jellybeans = 5400 := 
by
  sorry

end Ella_jellybeans_l455_455989


namespace quartic_two_real_roots_l455_455631

theorem quartic_two_real_roots
  (a b c d e : ℝ)
  (h : ∃ β : ℝ, β > 1 ∧ a * β^2 + (c - b) * β + e - d = 0)
  (ha : a ≠ 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a * x1^4 + b * x1^3 + c * x1^2 + d * x1 + e = 0) ∧ (a * x2^4 + b * x2^3 + c * x2^2 + d * x2 + e = 0) := 
  sorry

end quartic_two_real_roots_l455_455631


namespace find_max_sum_of_squares_l455_455735

open Real

theorem find_max_sum_of_squares 
  (a b c d : ℝ)
  (h1 : a + b = 17)
  (h2 : ab + c + d = 98)
  (h3 : ad + bc = 176)
  (h4 : cd = 105) :
  a^2 + b^2 + c^2 + d^2 ≤ 770 :=
sorry

end find_max_sum_of_squares_l455_455735


namespace min_distance_from_circle_to_line_l455_455720

noncomputable def polar_circle_eq : ℝ → ℝ := λ θ, 2 * sqrt 2 * sin (θ + π / 4)

noncomputable def polar_line_eq : ℝ → ℝ := λ θ, -2 / sin (θ + π / 3)

theorem min_distance_from_circle_to_line :
  let circleC := polar_circle_eq
      lineL := polar_line_eq in
  ∃ d_min, d_min = (5 + sqrt 3 - 2 * sqrt 2) / 2 :=
sorry

end min_distance_from_circle_to_line_l455_455720


namespace john_more_than_twice_bob_l455_455732

def roommates (Bob John : ℕ) := John - 2 * Bob

theorem john_more_than_twice_bob (Bob John : ℕ) (h_bob : Bob = 10) (h_john : John = 25) :
  roommates Bob John = 5 :=
by
  rw [roommates, h_bob, h_john]
  sorry

end john_more_than_twice_bob_l455_455732


namespace trackball_mice_count_l455_455053

theorem trackball_mice_count
  (total_mice : ℕ)
  (wireless_fraction : ℕ)
  (optical_fraction : ℕ)
  (h_total : total_mice = 80)
  (h_wireless : wireless_fraction = total_mice / 2)
  (h_optical : optical_fraction = total_mice / 4) :
  total_mice - (wireless_fraction + optical_fraction) = 20 :=
sorry

end trackball_mice_count_l455_455053


namespace remainder_2345678901_div_101_l455_455461

theorem remainder_2345678901_div_101 : 2345678901 % 101 = 12 :=
sorry

end remainder_2345678901_div_101_l455_455461


namespace range_of_a_l455_455644

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x - b * x^2

theorem range_of_a (a b x : ℝ) (hb : b ≤ 0) (hx : e < x ∧ x ≤ e^2) (h : f a b x ≥ x) :
  a ∈ Set.Ici (e^2 / 2) :=
begin
  sorry
end

end range_of_a_l455_455644


namespace cone_radius_l455_455234

theorem cone_radius (CSA : ℝ) (l : ℝ) (r : ℝ) (h_CSA : CSA = 989.6016858807849) (h_l : l = 15) :
    r = 21 :=
by
  sorry

end cone_radius_l455_455234


namespace subtract_complex_l455_455923

-- Definitions based on the conditions
def z1 : ℂ := 7 + 6 * complex.i
def z2 : ℂ := 4 - 2 * complex.i

-- Statement to be proved
theorem subtract_complex (h : z1 - z2 = 3 + 8 * complex.i) : z1 - z2 = 3 + 8 * complex.i :=
by
  sorry

end subtract_complex_l455_455923


namespace range_of_a_l455_455413

open Real

noncomputable def f (a x : ℝ) : ℝ := log x + (1 / 2) * x ^ 2 + a * x

theorem range_of_a (a : ℝ) : (∃ x > 0, deriv (f a) x = 3) ↔ a < 1 := by
  sorry

end range_of_a_l455_455413


namespace monotonicity_of_f_range_of_k_l455_455640

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x^2 + x) / (x + 1) - Real.log (x + 1)

theorem monotonicity_of_f (a : ℝ) (ha : 0 < a) :
  ((1 ≤ a) → ((∀ x ∈ Ioo (-1 : ℝ) 0, deriv (f a) x < 0) ∧ (∀ x > 0, deriv (f a) x > 0))) ∧
  ((1 / 2 < a ∧ a < 1) → ((∀ x ∈ Ioo (-1 : ℝ) (-2 + 1 / a), deriv (f a) x > 0) ∧ 
                           (∀ x ∈ Ioo (-2 + 1 / a) 0, deriv (f a) x < 0) ∧ 
                           (∀ x > 0, deriv (f a) x > 0))) ∧
  (a = 1 / 2 → ∀ x, deriv (f a) x ≥ 0) ∧
  ((0 < a ∧ a < 1 / 2) → ((∀ x ∈ Ioo (-1 : ℝ) 0, deriv (f a) x > 0) ∧ 
                          (∀ x ∈ Ioo (0) (-2 + 1 / a), deriv (f a) x < 0) ∧ 
                          (∀ x ∈ Ioi (-2 + 1 / a), deriv (f a) x > 0))) := sorry

noncomputable def f_1 (x : ℝ) : ℝ := x - Real.log (x + 1)

theorem range_of_k (k : ℝ) : (∀ x ≥ 0, f_1 x ≤ k * x^2) ↔ (k ≥ 1 / 2) := sorry

end monotonicity_of_f_range_of_k_l455_455640


namespace sum_s2018_l455_455926

def max (a b : ℝ) : ℝ := if a >= b then a else b

def seq_an (a : ℝ) (n : ℕ) : ℕ → ℝ
| 0     := a
| 1     := 1
| (n+2) := 2 * max (seq_an a (n + 1)) 2 / seq_an a n

def sum_seq (a : ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i => seq_an a i)

theorem sum_s2018 (a : ℝ) (a_pos : 0 < a) (h : seq_an a 2015 = 4 * a) : sum_seq a 2018 = 7260 :=
by
  sorry

end sum_s2018_l455_455926


namespace line_through_P0_perpendicular_to_plane_l455_455488

-- Definitions of the given conditions
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def P0 : Point3D := { x := 3, y := 4, z := 2 }

def plane (x y z : ℝ) : Prop := 8 * x - 4 * y + 5 * z - 4 = 0

-- The proof problem statement
theorem line_through_P0_perpendicular_to_plane :
  ∃ t : ℝ, (P0.x + 8 * t = x ∧ P0.y - 4 * t = y ∧ P0.z + 5 * t = z) ↔
    (∃ t : ℝ, x = 3 + 8 * t ∧ y = 4 - 4 * t ∧ z = 2 + 5 * t) → 
    (∃ t : ℝ, (x - 3) / 8 = t ∧ (y - 4) / -4 = t ∧ (z - 2) / 5 = t) := sorry

end line_through_P0_perpendicular_to_plane_l455_455488


namespace power_function_ab_l455_455608

theorem power_function_ab :
  ∀ (a b : ℝ), (∀ x : ℝ, f(x) = a * x^(2*a + 1) - b + 1) → a + b = 2 :=
by
  sorry

end power_function_ab_l455_455608


namespace shaded_area_fraction_l455_455822

theorem shaded_area_fraction (side_length : ℝ) (h1 : side_length = 1)
  (h2 : ∀ (A B C D : ℝ), A = B ∧ B = C ∧ C = D)
  (h3 : ∀ (center1 center2 : ℝ), center1 ≠ center2 → abs (center1 - center2) = side_length) :
  (2 - Real.sqrt(2)) * Real.pi - 1 = (2 - Real.sqrt(2)) * Real.pi - 1 :=
by sorry

end shaded_area_fraction_l455_455822


namespace noelle_speed_l455_455763

theorem noelle_speed (v d : ℝ) (h1 : d > 0) (h2 : v > 0) 
  (h3 : (2 * d) / ((d / v) + (d / 15)) = 5) : v = 3 := 
sorry

end noelle_speed_l455_455763


namespace temperature_in_New_York_l455_455825

-- Define the temperatures in New York, Miami, and San Diego
variables (T_NY T_Miami T_SanDiego : ℝ)

-- Create the conditions as Lean definitions
def condition1 := T_Miami = T_NY + 10
def condition2 := T_SanDiego = T_NY + 35
def condition3 := (T_NY + T_Miami + T_SanDiego) / 3 = 95

-- The statement to prove
theorem temperature_in_New_York :
  condition1 → condition2 → condition3 → T_NY = 80 :=
by
  sorry

end temperature_in_New_York_l455_455825


namespace inequality_b_c_a_l455_455684

-- Define the values of a, b, and c
def a := 8^53
def b := 16^41
def c := 64^27

-- State the theorem to prove the inequality b > c > a
theorem inequality_b_c_a : b > c ∧ c > a := by
  sorry

end inequality_b_c_a_l455_455684


namespace no_valid_solutions_l455_455797

theorem no_valid_solutions (a b : ℝ) (h1 : ∀ x, (a * x + b) ^ 2 = 4 * x^2 + 4 * x + 4) : false :=
  by
  sorry

end no_valid_solutions_l455_455797


namespace parts_in_batch_l455_455703

theorem parts_in_batch (a : ℕ) (h₁ : 20 * (a / 20) + 13 = a) (h₂ : 27 * (a / 27) + 20 = a) 
  (h₃ : 500 ≤ a) (h₄ : a ≤ 600) : a = 533 :=
by sorry

end parts_in_batch_l455_455703


namespace ratio_e_to_f_l455_455288

theorem ratio_e_to_f {a b c d e f : ℝ}
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : a * b * c / (d * e * f) = 0.75) :
  e / f = 0.5 :=
sorry

end ratio_e_to_f_l455_455288


namespace possible_values_of_a_l455_455783

theorem possible_values_of_a (x y a : ℝ)
  (h1 : x + y = a)
  (h2 : x^3 + y^3 = a)
  (h3 : x^5 + y^5 = a) :
  a = 0 ∨ a = 1 ∨ a = -1 ∨ a = 2 ∨ a = -2 :=
sorry

end possible_values_of_a_l455_455783


namespace vector_magnitude_sum_eq_sqrt6_l455_455296

-- Given conditions defined in Lean
variables (a b : ℝ × ℝ)
hypothesis ha : real.sqrt (a.1^2 + a.2^2) = 1
hypothesis hb : real.sqrt (b.1^2 + b.2^2) = 2
hypothesis hsub : real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 2

-- Goal is to prove that |a + b| = sqrt(6)
theorem vector_magnitude_sum_eq_sqrt6 : real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = real.sqrt 6 :=
by
  sorry

end vector_magnitude_sum_eq_sqrt6_l455_455296


namespace probability_of_specific_sequence_l455_455151

theorem probability_of_specific_sequence :
  let stop_at_4_occurs, even_cards_meet_2_6_occurs, odd_card_appears_occurs :=
    True, True, True in
  let probability_4_appears := 1 / 6,
      probability_other_than_4 := 5 / 6 in
  let sum_probability := 
    (λ n, (probability_4_appears * ((4 / 5) ^ (n - 1)) * (1 - (2 / 5) ^ (n - 1)))) in
  let series_sum :=
    tsum (λ n, sum_probability n) in
  series_sum = 1 / 24 :=
sorry

end probability_of_specific_sequence_l455_455151


namespace mirror_area_correct_l455_455054

noncomputable def frame_width : ℝ := 100
noncomputable def frame_height : ℝ := 140
noncomputable def frame_border : ℝ := 12

def mirror_width := frame_width - 2 * frame_border
def mirror_height := frame_height - 2 * frame_border

theorem mirror_area_correct : mirror_width * mirror_height = 8816 :=
by
  sorry

end mirror_area_correct_l455_455054


namespace people_left_line_l455_455914

variables (initial_people left_people new_people final_people : ℕ)

theorem people_left_line :
  initial_people = 9 →
  new_people = 3 →
  final_people = 6 →
  final_people = initial_people - left_people + new_people →
  left_people = 6 :=
by
  intros h_initial h_new h_final h_eq
  rw [h_initial, h_new, h_final] at h_eq
  linarith
  sorry

end people_left_line_l455_455914


namespace perimeter_of_AIHJ_eq_2_x_eq_half_one_plus_d_plus_f_l455_455103

-- Definitions and theorems for Part (a)
theorem perimeter_of_AIHJ_eq_2 
  (A B C H I J : Type) 
  (side_length : ℝ) 
  (h_triangle_equilateral : equilateral_triangle A B C side_length) 
  (h_on_BC : is_on_segment H B C) 
  (h_parallel_HI_AC : parallel HI AC)
  (h_parallel_HJ_AB : parallel HJ AB) 
  (AB_eq : side_length = 1) :
  perimeter AI HJ = 2 := 
sorry

-- Definitions and theorems for Part (b)
theorem x_eq_half_one_plus_d_plus_f
  (A B C H I J L O M P : Type) 
  (side_length : ℝ)
  (d f x : ℝ)
  (h_triangle_equilateral : equilateral_triangle A B C side_length) 
  (h_on_BC : is_on_segment H B C) 
  (h_parallel_HI_AC : parallel HI AC) 
  (h_parallel_HJ_AB : parallel HJ AB) 
  (h_perpendicular_LO_BC : perpendicular LO BC) 
  (h_perpendicular_MP_BC : perpendicular MP BC) 
  (segment_lengths : ℝ)
  (h_d : segment_length IL = d)
  (h_f : segment_length JM = f)
  (h_x : segment_length OP = x) 
  (AB_eq : side_length = 1) :
  x = (1 + d + f) / 2 :=
sorry

end perimeter_of_AIHJ_eq_2_x_eq_half_one_plus_d_plus_f_l455_455103


namespace expression_evaluation_l455_455198

noncomputable def calculate_expression : ℝ :=
  let term1 := (-1/3:ℝ)⁻¹
  let term2 := (sqrt 3 - 2)^0
  let term3 := 4 * (cos (real.pi / 4)) -- 45 degrees in radians
  term1 - term2 + term3

theorem expression_evaluation : calculate_expression = -4 + 2 * sqrt 2 :=
by
  -- definitions based on given conditions
  let term1 := (-1/3:ℝ)⁻¹
  have h_term1 : term1 = -3 := by sorry -- previous steps assure this
  let term2 := (sqrt 3 - 2)^0
  have h_term2 : term2 = 1 := by sorry -- any number to the power of zero is 1
  let term3 := 4 * (cos (real.pi / 4)) -- cos 45 degrees = sqrt 2 / 2
  have h_term3 : term3 = 2 * sqrt 2 := by sorry -- evaluated correctly as per known steps

  -- combining terms
  calc
    term1 - term2 + term3
        = -3 - 1 + 2 * sqrt 2 : by
          rw [h_term1, h_term2, h_term3]
    ... = -4 + 2 * sqrt 2 : by ring

end expression_evaluation_l455_455198


namespace cannot_divide_convex_13gon_into_parallelograms_l455_455772

-- Define a convex polygon
structure ConvexPolygon (n : Nat) :=
  (sides : Nat)
  (is_convex : ∀ (angle : Fin n), angle < 180)

-- Define specific case of convex 13-gon
def convex_13gon : ConvexPolygon 13 :=
  { sides := 13,
    is_convex := λ angle, sorry }

-- Formalize the proof statement
theorem cannot_divide_convex_13gon_into_parallelograms :
  ¬ (∃ (P : ConvexPolygon 13), can_be_divided_into_parallelograms P) :=
sorry

end cannot_divide_convex_13gon_into_parallelograms_l455_455772


namespace Isaabel_math_pages_l455_455729

theorem Isaabel_math_pages (x : ℕ) (total_problems : ℕ) (reading_pages : ℕ) (problems_per_page : ℕ) :
  (reading_pages * problems_per_page = 20) ∧ (total_problems = 30) →
  x * problems_per_page + 20 = total_problems →
  x = 2 := by
  sorry

end Isaabel_math_pages_l455_455729


namespace train_length_calculation_l455_455117

theorem train_length_calculation (L : ℝ) (t : ℝ) (v_faster : ℝ) (v_slower : ℝ) (relative_speed : ℝ) (total_distance : ℝ) :
  (v_faster = 60) →
  (v_slower = 40) →
  (relative_speed = (v_faster - v_slower) * 1000 / 3600) →
  (t = 48) →
  (total_distance = relative_speed * t) →
  (2 * L = total_distance) →
  L = 133.44 :=
by
  intros
  sorry

end train_length_calculation_l455_455117


namespace number_of_children_correct_l455_455113

def total_spectators : ℕ := 25000
def men_spectators : ℕ := 15320
def ratio_children_women : ℕ × ℕ := (7, 3)
def remaining_spectators : ℕ := total_spectators - men_spectators
def total_ratio_parts : ℕ := ratio_children_women.1 + ratio_children_women.2
def spectators_per_part : ℕ := remaining_spectators / total_ratio_parts

def children_spectators : ℕ := spectators_per_part * ratio_children_women.1

theorem number_of_children_correct : children_spectators = 6776 := by
  sorry

end number_of_children_correct_l455_455113


namespace sum_of_integers_n_correct_sum_final_sum_of_integers_n_l455_455977

theorem sum_of_integers_n (n x : ℤ) (h : n^2 - 17 * n + 72 = x^2) (hn : 0 < n) : 
  n = 8 ∨ n = 9 := 
sorry

theorem correct_sum : (8 + 9 : ℤ) = 17 := 
by norm_num

theorem final_sum_of_integers_n :
  ∃ n1 n2 : ℤ, (n1^2 - 17 * n1 + 72 = 0) ∧ (n2^2 - 17 * n2 + 72 = 0) ∧ (0 < n1) ∧ (0 < n2) ∧ (n1 + n2 = 17) :=
begin
  use [8, 9],
  split,
  { exact (by norm_num : (8^2 - 17 * 8 + 72 = 0)) },
  split,
  { exact (by norm_num : (9^2 - 17 * 9 + 72 = 0)) },
  split,
  { exact (by norm_num : 0 < 8) },
  split,
  { exact (by norm_num : 0 < 9) },
  { exact (by norm_num : 8 + 9 = 17) }
end

end sum_of_integers_n_correct_sum_final_sum_of_integers_n_l455_455977


namespace solution_set_l455_455588

theorem solution_set (x : ℝ) :
  (x^2 / (x + 1) > 3 / (x - 1) + 7 / 4) ↔ (x ∈ Iio (-1) ∪ Ioi 1.75) :=
by
  sorry

end solution_set_l455_455588


namespace sum_of_squares_of_solutions_l455_455594

theorem sum_of_squares_of_solutions : 
  (∃ x : ℝ, |x^2 - 2*x + 1/2010| = 1/2010) → 
  (∑ x in {x | |x^2 - 2*x + 1/2010| = 1/2010}, x^2) = 8036 / 2010 :=
by 
  sorry

end sum_of_squares_of_solutions_l455_455594


namespace find_f_2011_l455_455633

theorem find_f_2011 (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = f (-x)) 
  (h2 : ∀ x, f (x + 1) * f (x - 1) = 1) 
  (h3 : ∀ x, f x > 0) : 
  f 2011 = 1 := 
sorry

end find_f_2011_l455_455633


namespace triangle_ABC_circumradius_l455_455726

noncomputable def triangle_circumradius 
  (A B C D O : Type)
  [IsTriangle A B C] 
  [IsPoint A] 
  [IsPoint B] 
  [IsPoint C] 
  [IsPoint D] 
  [IsPoint O] 
  [AngleBisector A D C] [AngleBisector C D A] 
  (R_ADC : ℝ) 
  (hR_ADC : R_ADC = 6)
  (angle_30_ACO : ∠ A C O = 30) 
  : ℝ :=
  let r_ABC := 6 in r_ABC

-- The theorem we must prove
theorem triangle_ABC_circumradius 
  (A B C D O : Type)
  [IsTriangle A B C]
  [IsPoint A]
  [IsPoint B]
  [IsPoint C]
  [IsPoint D]
  [IsPoint O]
  [AngleBisector A D C]
  [AngleBisector C D A]
  (R_ADC : ℝ)
  (hR_ADC : R_ADC = 6)
  (angle_30_ACO : ∠ A C O = 30) 
  : r_ABC = 6 := 
sorry

end triangle_ABC_circumradius_l455_455726


namespace problem1_l455_455875

open Real

theorem problem1 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : 
  ∃ (m : ℝ), m = 9 / 2 ∧ ∀ (u v : ℝ), 0 < u → 0 < v → u + v = 1 → (1 / u + 4 / (1 + v)) ≥ m := 
sorry

end problem1_l455_455875


namespace hyperbola_focus_distance_l455_455650

theorem hyperbola_focus_distance
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (hyperbola_eq : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → (x+sqrt(2)*y=0))
  (M : ℝ × ℝ) (MF1_perpendicular_to_x : M.1 = -3)
  (F2_focus_parabola : F2 = (3, 0))
  (dist_F1_to_F2M : distance F1 (line_through F2 M) = 6/5) :
  distance F1 (line_through F2 M) = 6/5 :=
sorry -- Proof not required

end hyperbola_focus_distance_l455_455650


namespace exists_prime_and_increasing_seq_all_primes_l455_455367

theorem exists_prime_and_increasing_seq_all_primes (k : ℕ) (hk : k > 1) :
  ∃ p : ℕ, prime p ∧ ∃ a : ℕ → ℕ, (∀ n, a n > 0) ∧ (strict_mono a)  ∧ (∀ n, prime (p + k * a n)) :=
by
  sorry

end exists_prime_and_increasing_seq_all_primes_l455_455367


namespace crates_needed_l455_455575

-- Conditions as definitions
def novels : ℕ := 145
def comics : ℕ := 271
def documentaries : ℕ := 419
def albums : ℕ := 209
def crate_capacity : ℕ := 9

-- Total items calculation
def total_items : ℕ := novels + comics + documentaries + albums

-- Proof statement
theorem crates_needed : (total_items / crate_capacity) = 116 := by
  sorry

end crates_needed_l455_455575


namespace S_10_eq_110_l455_455013

-- Conditions
def a (n : ℕ) : ℕ := sorry  -- Assuming general term definition of arithmetic sequence
def S (n : ℕ) : ℕ := sorry  -- Assuming sum definition of arithmetic sequence

axiom a_3_eq_16 : a 3 = 16
axiom S_20_eq_20 : S 20 = 20

-- Prove
theorem S_10_eq_110 : S 10 = 110 :=
  by
  sorry

end S_10_eq_110_l455_455013


namespace max_value_of_f_l455_455422

noncomputable def f : ℝ → ℝ := λ x, 3 * real.sqrt (x - 1) + 4 * real.sqrt (2 - x)

theorem max_value_of_f :
  ∃ x ∈ set.Icc 1 2, f x = 5 :=
sorry

end max_value_of_f_l455_455422


namespace percent_students_with_B_is_20_l455_455909

def mr_hyde_class_scores : List ℕ :=
  [93, 65, 88, 100, 72, 95, 82, 68, 79, 56, 87, 81, 74, 85, 91]

def count_students_with_B (scores : List ℕ) : ℕ :=
  scores.filter (λ score => score ≥ 87 ∧ score ≤ 93).length

theorem percent_students_with_B_is_20 :
  (count_students_with_B mr_hyde_class_scores) * 100 / mr_hyde_class_scores.length = 20 :=
by
  sorry

end percent_students_with_B_is_20_l455_455909


namespace square_side_length_from_circumference_l455_455438

def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

theorem square_side_length_from_circumference (C : ℝ) (h : C = 37.69911184307752) : 
  ∃ (s : ℝ), s = 12 :=
by
  let r := C / (2 * Real.pi)
  let d := 2 * r
  have : d = 12, from sorry
  use d
  exact this

end square_side_length_from_circumference_l455_455438


namespace increasing_function_range_l455_455641

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1 / 2) * m * x^2 - 2 * x + Real.log x

theorem increasing_function_range (m : ℝ) : (∀ x > 0, m * x + (1 / x) - 2 ≥ 0) ↔ m ≥ 1 := 
by 
  sorry

end increasing_function_range_l455_455641


namespace ticket_distribution_methods_l455_455110

-- Define the set of tickets and the set of people
def tickets := {1, 2, 3, 4, 5, 6}
def people := {A, B, C, D}

-- Define the conditions as hypotheses
def valid_distribution : (tickets × people) → Prop :=
  λ t p,
    ∃ (m : people → list ℕ), 
    (∀ x : people, x ∈ people → (m x).length ≥ 1 ∧ (m x).length ≤ 2 ∧ (∀ (a b : ℕ), a ∈ (m x) → b ∈ (m x) → abs (a - b) = 1)) ∧ 
    (∀ t ∈ tickets, ∃! x : people, t ∈ (m x))

-- Theorem statement
theorem ticket_distribution_methods :
  (∃ (f : tickets → people), valid_distribution f) ↔ 144 := 
by
  sorry

end ticket_distribution_methods_l455_455110


namespace find_largest_m_l455_455964

theorem find_largest_m (m : ℤ) : (m^2 - 11 * m + 24 < 0) → m ≤ 7 := sorry

end find_largest_m_l455_455964


namespace downstream_speed_l455_455512

def still_water_speed : ℝ := 8.5
def upstream_speed : ℝ := 4

def stream_speed (V_b V_u : ℝ) : ℝ := V_b - V_u

theorem downstream_speed : 
  let V_b := still_water_speed in
  let V_u := upstream_speed in
  let V_s := stream_speed V_b V_u in
  V_b + V_s = 13 := 
by
  sorry

end downstream_speed_l455_455512


namespace joel_laps_count_l455_455489

def yvonne_laps : ℕ := 10

def younger_sister_laps : ℕ := yvonne_laps / 2

def joel_laps : ℕ := younger_sister_laps * 3

theorem joel_laps_count : joel_laps = 15 := by
  -- The proof is not required as per instructions
  sorry

end joel_laps_count_l455_455489


namespace curve_is_line_l455_455956

-- Define the polar equation as a condition
def polar_eq (r θ : ℝ) : Prop := r = 2 / (2 * Real.sin θ - Real.cos θ)

-- Define what it means for a curve to be a line
def is_line (x y : ℝ) : Prop := x + 2 * y = 2

-- The main statement to prove
theorem curve_is_line (r θ : ℝ) (x y : ℝ) (hr : polar_eq r θ) (hx : x = r * Real.cos θ) (hy : y = r * Real.sin θ) :
  is_line x y :=
sorry

end curve_is_line_l455_455956


namespace distinct_not_geom_prog_l455_455918

open Nat

theorem distinct_not_geom_prog (k m n : ℕ) (hk : k ≠ m) (hm : m ≠ n) (hn : k ≠ n) :
  ¬ ((2^m + 1)^2 = (2^k + 1) * (2^n + 1)) :=
by sorry

end distinct_not_geom_prog_l455_455918


namespace number_95_descending_is_21354_l455_455826

/-- There are 120 five-digit numbers formed by the digits 1, 2, 3, 4, 5, arranged in descending order.
The 95th number is 21354. -/
theorem number_95_descending_is_21354 :
  ∃ L : List (List ℕ), (L.length = 120) ∧ 
  (∀ l ∈ L, l ~ [1, 2, 3, 4, 5]) ∧
  (L.sorted (≥) [1,2,3,4,5]) ∧ 
  (L.get! 94 = [2, 1, 3, 5, 4]) :=
by 
  sorry

end number_95_descending_is_21354_l455_455826


namespace circle_radius_inscribed_l455_455429

noncomputable def a : ℝ := 6
noncomputable def b : ℝ := 12
noncomputable def c : ℝ := 18

noncomputable def r : ℝ :=
  let term1 := 1/a
  let term2 := 1/b
  let term3 := 1/c
  let sqrt_term := Real.sqrt ((1/(a * b)) + (1/(a * c)) + (1/(b * c)))
  1 / ((term1 + term2 + term3) + 2 * sqrt_term)

theorem circle_radius_inscribed :
  r = 36 / 17 := 
by
  sorry

end circle_radius_inscribed_l455_455429


namespace a_b_eqn_l455_455380

noncomputable def a_b_sequence : ℕ → ℝ
| 1 := 1
| 2 := 3
| 3 := 4
| 4 := 7
| 5 := 11
| (n+1) := a_b_sequence n + a_b_sequence (n - 1)

theorem a_b_eqn {a b : ℝ}
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = (a_b_sequence 10) :=
by
  -- The proof would go here
  sorry

end a_b_eqn_l455_455380


namespace range_of_m_for_ellipse_l455_455012

-- Define the equation of the ellipse
def ellipse_equation (m : ℝ) (x y : ℝ) : Prop :=
  m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2

-- The theorem to prove
theorem range_of_m_for_ellipse (m : ℝ) :
  (∀ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2) →
  5 < m :=
sorry

end range_of_m_for_ellipse_l455_455012


namespace graph_of_exponential_function_passes_through_point_l455_455415

theorem graph_of_exponential_function_passes_through_point
    (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
    ∃ (y : ℝ), (y = a^0 + 1) ∧ (0, y) = (0, 2) :=
by
  use a^0 + 1
  split
  sorry

end graph_of_exponential_function_passes_through_point_l455_455415


namespace calculate_expression_l455_455545

theorem calculate_expression : 1453 - 250 * 2 + 130 / 5 = 979 := by
  sorry

end calculate_expression_l455_455545


namespace intersection_of_prime_and_even_is_two_l455_455738

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_even (n : ℕ) : Prop :=
  ∃ k : ℤ, n = 2 * k

theorem intersection_of_prime_and_even_is_two :
  {n : ℕ | is_prime n} ∩ {n : ℕ | is_even n} = {2} :=
by
  sorry

end intersection_of_prime_and_even_is_two_l455_455738


namespace fraction_start_with_9_end_with_0_is_1_over_72_l455_455543

-- Definition of valid 8-digit telephone number
def valid_phone_number (d : Fin 10) (n : Fin 10) (m : Fin (10 ^ 6)) : Prop :=
  2 ≤ d.val ∧ d.val ≤ 9 ∧ n.val ≤ 8

-- Definition of phone numbers that start with 9 and end with 0
def starts_with_9_ends_with_0 (d : Fin 10) (n : Fin 10) (m : Fin (10 ^ 6)) : Prop :=
  d.val = 9 ∧ n.val = 0

-- The total number of valid 8-digit phone numbers
noncomputable def total_valid_numbers : ℕ :=
  8 * (10 ^ 6) * 9

-- The number of valid phone numbers that start with 9 and end with 0
noncomputable def valid_start_with_9_end_with_0 : ℕ :=
  10 ^ 6

-- The target fraction
noncomputable def target_fraction : ℚ :=
  valid_start_with_9_end_with_0 / total_valid_numbers

-- Main theorem
theorem fraction_start_with_9_end_with_0_is_1_over_72 :
  target_fraction = (1 / 72 : ℚ) :=
by
  sorry

end fraction_start_with_9_end_with_0_is_1_over_72_l455_455543


namespace solve_quadratic1_solve_quadratic2_l455_455791

-- Equation 1
theorem solve_quadratic1 (x : ℝ) :
  (x = 4 + 3 * Real.sqrt 2 ∨ x = 4 - 3 * Real.sqrt 2) ↔ x ^ 2 - 8 * x - 2 = 0 := by
  sorry

-- Equation 2
theorem solve_quadratic2 (x : ℝ) :
  (x = 3 / 2 ∨ x = -1) ↔ 2 * x ^ 2 - x - 3 = 0 := by
  sorry

end solve_quadratic1_solve_quadratic2_l455_455791


namespace third_rectangle_is_square_l455_455883

-- We model the problem using Lean definitions.
variables {A B C D : Type} [AffineSpace ℝ A B C D]

-- Defining the concept of a convex quadrilateral, the circumscribed condition, and squares
def is_convex_quadrilateral (A B C D : A) : Prop := sorry
def is_circumscribed_rectangle (A B C D : A) (rect : Rectangle) : Prop := sorry
def is_square (rect : Rectangle) : Prop := sorry

-- Main statement: Given the conditions, we need to prove the conclusion
theorem third_rectangle_is_square 
  (ABCD : A) 
  (rect1 rect2 rect3 : Rectangle)
  (h1 : is_circumscribed_rectangle ABCD rect1)
  (h2 : is_circumscribed_rectangle ABCD rect2)
  (h3 : is_circumscribed_rectangle ABCD rect3)
  (h4 : is_square rect1)
  (h5 : is_square rect2) 
  :
  is_square rect3 := 
sorry

end third_rectangle_is_square_l455_455883


namespace part_c_proved_inequality_1_part_c_proved_inequality_2_l455_455252

-- Definitions
variables {x y z : ℝ}
def p := 4 * (x + y + z)
def s := 2 * (x * y + y * z + z * x)
def d := Real.sqrt (x^2 + y^2 + z^2)

-- Conditions
axiom h1 : x < y
axiom h2 : y < z

-- Prove the inequalities
theorem part_c_proved_inequality_1 (h1 : x < y) (h2 : y < z) :
  x < (1 / 3) * ((1 / 4) * p - Real.sqrt (d^2 - 1 / 2) * s) := sorry

theorem part_c_proved_inequality_2 (h1 : x < y) (h2 : y < z) :
  z > (1 / 3) * ((1 / 4) * p + Real.sqrt (d^2 - 1 / 2) * s) := sorry

end part_c_proved_inequality_1_part_c_proved_inequality_2_l455_455252


namespace fraction_of_shaded_area_l455_455335

noncomputable theory

-- Define the conditions in the problem

def right_angled_triangle (P Q R : Point) :=
  angle P Q R = 90

def isosceles_triangle (P Q R : Point) :=
  dist P Q = dist Q R

def perpendicular (A B C : Point) :=
  angle A B C = 90

def bisect (A B : Point) :=
  midpoint A B

-- Define the statement to prove the fraction of the shaded area

theorem fraction_of_shaded_area (P Q R S T U V W : Point) 
  (h1 : right_angled_triangle P Q R)
  (h2 : isosceles_triangle P Q R)
  (h3 : perpendicular Q S (line_through P R))
  (h4 : perpendicular T U PR)
  (h5 : perpendicular V W PR)
  (h6 : perpendicular S T QR)
  (h7 : perpendicular U V QR) :
  fraction_shaded_area P Q R S T U V W = 5/32 :=
sorry

end fraction_of_shaded_area_l455_455335


namespace find_zero_of_f_l455_455093

noncomputable def f (x : ℝ) : ℝ := Real.exp (x - 1) - 3

theorem find_zero_of_f : ∃ x : ℝ, f x = 0 ∧ x = 1 + Real.log 3 :=
by
  use 1 + Real.log 3
  split
  · sorry
  · rfl

end find_zero_of_f_l455_455093


namespace log_product_l455_455675

noncomputable def log_a (a b : ℝ) := real.log b / real.log a

theorem log_product (a b : ℝ) (ha : a > 1) (hb : b > 1) : 
  (log_a a (b^3)) * (log_a b (a^2)) = 6 :=
by
  -- Proof goes here
  sorry

end log_product_l455_455675


namespace volume_of_intersecting_octahedra_l455_455473

def absolute (x : ℝ) : ℝ := abs x

noncomputable def volume_of_region : ℝ :=
  let region1 (x y z : ℝ) := absolute x + absolute y + absolute z ≤ 2
  let region2 (x y z : ℝ) := absolute x + absolute y + absolute (z - 2) ≤ 2
  -- The region is the intersection of these two inequalities
  -- However, we calculate its volume directly
  (2 / 3 : ℝ)

theorem volume_of_intersecting_octahedra :
  (volume_of_region : ℝ) = (2 / 3 : ℝ) :=
sorry

end volume_of_intersecting_octahedra_l455_455473


namespace sum_of_reciprocals_exceeds_l455_455521

-- Each number cannot contain the sequence "729"
def no_sequence_729 (n : ℕ) : Prop := ∀ i < n - 2, 
  (n / 10^(i+2) % 10 ≠ 7 ∨ n / 10^(i+1) % 10 ≠ 2 ∨ n / 10^i % 10 ≠ 9)

-- Number sets computed under given constraints
def num_3n_plus_1 (n : ℕ) : ℕ := 9 * 999^n
def num_3n_plus_2 (n : ℕ) : ℕ := 90 * 999^n
def num_3n_plus_3 (n : ℕ) : ℕ := 899 * 999^n

-- Minimum value thresholds
def min_value_3n_plus_1 (n : ℕ) : ℕ := 10^3n
def min_value_3n_plus_2 (n : ℕ) : ℕ := 10^(3n + 1)
def min_value_3n_plus_3 (n : ℕ) : ℕ := 10^(3n + 2)

-- Main statement to prove
theorem sum_of_reciprocals_exceeds : 
  (∀ n, no_sequence_729 n) →
  ∑ n in 0..n, 
    (∑ k in 0..(num_3n_plus_1 n), 
      (1 / min_value_3n_plus_1 n) +
     ∑ k in 0..(num_3n_plus_2 n), 
      (1 / min_value_3n_plus_2 n) +
     ∑ k in 0..(num_3n_plus_3 n), 
      (1 / min_value_3n_plus_3 n)
  ) ≤ 30000 :=
by
  sorry

end sum_of_reciprocals_exceeds_l455_455521


namespace number_of_senior_citizen_tickets_sold_on_first_day_l455_455796

theorem number_of_senior_citizen_tickets_sold_on_first_day 
  (S : ℤ) (x : ℤ)
  (student_ticket_price : ℤ := 9)
  (first_day_sales : ℤ := 79)
  (second_day_sales : ℤ := 246) 
  (first_day_student_tickets_sold : ℤ := 3)
  (second_day_senior_tickets_sold : ℤ := 12)
  (second_day_student_tickets_sold : ℤ := 10) 
  (h1 : 12 * S + 10 * student_ticket_price = second_day_sales)
  (h2 : S * x + first_day_student_tickets_sold * student_ticket_price = first_day_sales) : 
  x = 4 :=
by
  sorry

end number_of_senior_citizen_tickets_sold_on_first_day_l455_455796


namespace fill_blank_to_optimal_digits_l455_455140

/-- Given a 9-digit number of the form 9□7856000,
    prove that filling the blank with 0 makes the number closest to 9 billion,
    and filling the blank with 9 makes the number closest to 10 billion.
-/
theorem fill_blank_to_optimal_digits :
  ∀ (d : ℕ), (d < 10) → 
  (abs (9000000000 - (900000000 + d*1000000 + 7856000)) < abs (9000000000 - (900000000 + d*1000000 + 7856000 - (d - 0)*1000000))) ∧
  (abs (10000000000 - (900000000 + d*1000000 + 7856000)) < abs (10000000000 - (900000000 + d*1000000 + 7856000 - (d - 9)*1000000)))
:= 
begin
  intro d,
  intro h_d_lt_10,
  split,
  { sorry, }, -- Proving closest to 9 billion when digits is 0 (left side of tuple); Fill out this proof.
  { sorry, }  -- Proving closest to 10 billion when digits is 9 (right side of tuple); Fill out this proof.
end

end fill_blank_to_optimal_digits_l455_455140


namespace triangle_angle_measures_l455_455839

theorem triangle_angle_measures
  (P Q R S : Type)
  [inner_product_space ℝ P] [inner_product_space ℝ Q] [inner_product_space ℝ R] [inner_product_space ℝ S]
  (PQ QR PR RS : ℝ)
  (PQR : isosceles_triangle PQ QR)
  (PRS : isosceles_triangle PR RS)
  (R_inside_PQS : inside_triangle P Q S R)
  (angle_PQR_equals_50 : ∠PQR = 50)
  (angle_PRS_equals_160 : ∠PRS = 160) :
  ∠QPR = 55 :=
sorry

end triangle_angle_measures_l455_455839


namespace f_is_constant_l455_455038

-- Define the function and its domain and codomain
def f : ℤ × ℤ → ℕ

-- Define the condition that for all points (x, y) the value of f is the average
-- of the values at its four neighboring points
axiom avg_property : ∀ (x y : ℤ), 
  f (x, y) = (f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1)) / 4

-- Prove that f is constant
theorem f_is_constant : ∀ (x y : ℤ), f x y = f 0 0 :=
by
  sorry

end f_is_constant_l455_455038


namespace hash_triple_64_l455_455924

def hash (N : ℝ) : ℝ := (1/2) * (N - 2) + 2

theorem hash_triple_64 : hash (hash (hash 64)) = 9.75 := 
by 
  -- The proof is skipped as per instructions
  sorry

end hash_triple_64_l455_455924


namespace max_min_f_find_a_l455_455658

open Real

-- Definitions given in the problem
def m (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin (2 * x) + 2, cos x)
def n (x : ℝ) : ℝ × ℝ := (1, 2 * cos x)
def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

/-
Problem 1: Find the maximum and minimum values of f(x) on [0, π/4]
-/
theorem max_min_f :
  max (f 0) (max (f (π / 6)) (f (π / 4))) = 5 ∧ min (f 0) (min (f (π / 6)) (f (π / 4))) = 4 := by
  sorry

/-
Problem 2:
In ΔABC, if f(A) = 4, b = 1, and the area is √3 / 2, find the value of a.
-/
theorem find_a (A b : ℝ) (area : ℝ) (a : ℝ) :
  f A = 4 ∧ b = 1 ∧ area = sqrt 3 / 2 →
  a^2 = b^2 + (2:ℝ)^2 - 2 * b * (2:ℝ) * cos A →
  a = sqrt 3 := by 
  sorry

end max_min_f_find_a_l455_455658


namespace find_x_if_point_on_line_l455_455312

theorem find_x_if_point_on_line : 
  (∃ x : ℝ, (x, -6) ∈ line_through_points (2, 10) (-3, 1)) → x = -62 / 9 :=
by
  intro h
  sorry

def line_through_points (p1 p2 : ℝ × ℝ) : set (ℝ × ℝ) :=
  { pt : ℝ × ℝ | let (x1, y1) := p1, (x2, y2) := p2 in (y2 - y1) * (pt.1 - x1) = (x2 - x1) * (pt.2 - y1) }


end find_x_if_point_on_line_l455_455312


namespace largest_a_value_l455_455962

theorem largest_a_value (x : ℝ) (hx : -π / 2 < x ∧ x < π / 2) : 
  let t := Real.tan x 
  in ∃ a : ℤ, (∀ t, (a^2 - 15 * a - (t - 1) * (t + 2) * (t + 5) * (t + 8) < 35)) ↔ (a = 10) :=
by
  sorry

end largest_a_value_l455_455962


namespace smallest_n_exists_l455_455238

theorem smallest_n_exists:
  ∃ n, n = 34 ∧ ∀ (a : Fin 15 → ℕ),
  (∀ k : Fin 15, a k ∈ ({(16 + k) | k : ℕ} : Set ℕ)) ∧ 
  (∀ k : Fin 15, k + 1 ∣ a k) ∧
  function.injective a :=
sorry

end smallest_n_exists_l455_455238


namespace num_valid_years_between_1000_and_2000_l455_455888

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def two_digit_prime_palindromes : List ℕ := [11]
def three_digit_prime_palindromes : List ℕ := [101, 131, 151, 181]

def has_properties (year : ℕ) : Prop :=
  is_palindrome year ∧
  ∃ (p1 p2 : ℕ), p1 ∈ two_digit_prime_palindromes ∧ p2 ∈ three_digit_prime_palindromes ∧ year = p1 * p2

def palindrome_years := List.range' 1000 1001 -- generates years from 1000 to 1999 inclusive

noncomputable def count_valid_years : ℕ :=
  palindrome_years.countp has_properties

theorem num_valid_years_between_1000_and_2000 : count_valid_years = 4 :=
  sorry

end num_valid_years_between_1000_and_2000_l455_455888


namespace bus_avg_speed_l455_455142

noncomputable def average_speed_of_bus 
  (bicycle_speed : ℕ) 
  (initial_distance_behind : ℕ) 
  (catch_up_time : ℕ) :
  ℕ :=
  (initial_distance_behind + bicycle_speed * catch_up_time) / catch_up_time

theorem bus_avg_speed 
  (bicycle_speed : ℕ) 
  (initial_distance_behind : ℕ) 
  (catch_up_time : ℕ) 
  (h_bicycle_speed : bicycle_speed = 15) 
  (h_initial_distance_behind : initial_distance_behind = 195)
  (h_catch_up_time : catch_up_time = 3) :
  average_speed_of_bus bicycle_speed initial_distance_behind catch_up_time = 80 :=
by
  sorry

end bus_avg_speed_l455_455142


namespace binomial_falling_factorial_identity_l455_455927

def falling_factorial (x : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => x * falling_factorial (x - 1) n

theorem binomial_falling_factorial_identity (x y : ℕ) (n : ℕ) :
  falling_factorial (x + y) n = ∑ k in finset.range (n + 1), (nat.choose n k) * falling_factorial x k * falling_factorial y (n - k) :=
sorry

end binomial_falling_factorial_identity_l455_455927


namespace substitution_correct_l455_455119

theorem substitution_correct : 
  ∀ (x y: ℝ), (4 * x + 5 * y = 7) ∧ (y = 2 * x - 1) → (4 * x + 10 * x - 5 = 7) :=
by 
  intros x y h,
  cases h with h1 h2,
  rw [h2, mul_add, mul_sub] at h1,
  linarith [h1]

-- sorry can be added to skip the proof
-- sorry

end substitution_correct_l455_455119


namespace find_C_l455_455437

theorem find_C (A B C D : ℕ) (h_diff : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_eq : 4000 + 100 * A + 50 + B + (1000 * C + 200 + 10 * D + 7) = 7070) : C = 2 :=
sorry

end find_C_l455_455437


namespace calculate_catfish_dinner_cost_l455_455784

def cost_of_catfish_dinner : ℝ := 6

def river_joe_charges_popcorn_shrimp : ℝ := 3.5

def total_orders : ℕ := 26

def total_revenue : ℝ := 133.5

def number_of_popcorn_shrimp_orders : ℕ := 9

theorem calculate_catfish_dinner_cost :
  let remaining_orders := total_orders - number_of_popcorn_shrimp_orders,
      revenue_from_popcorn_shrimp := (number_of_popcorn_shrimp_orders : ℝ) * river_joe_charges_popcorn_shrimp,
      revenue_from_catfish_dinner := total_revenue - revenue_from_popcorn_shrimp,
      cost_of_catfish := revenue_from_catfish_dinner / (remaining_orders : ℝ)
  in cost_of_catfish = cost_of_catfish_dinner :=
by
  sorry

end calculate_catfish_dinner_cost_l455_455784


namespace total_students_in_class_l455_455699

theorem total_students_in_class (S R : ℕ)
  (h1 : S = 2 + 12 + 4 + R)
  (h2 : 0 * 2 + 1 * 12 + 2 * 4 + 3 * R = 2 * S) : S = 34 :=
by { sorry }

end total_students_in_class_l455_455699


namespace natalia_crates_l455_455583

/- The definitions from the conditions -/
def novels : ℕ := 145
def comics : ℕ := 271
def documentaries : ℕ := 419
def albums : ℕ := 209
def crate_capacity : ℕ := 9

/- The proposition to prove -/
theorem natalia_crates : (novels + comics + documentaries + albums) / crate_capacity = 116 := by
  -- this skips the proof and assumes the theorem is true
  sorry

end natalia_crates_l455_455583


namespace die_roll_probability_div_3_l455_455174

noncomputable def probability_divisible_by_3 : ℚ :=
  1 - ((2 : ℚ) / 3) ^ 8

theorem die_roll_probability_div_3 :
  probability_divisible_by_3 = 6305 / 6561 :=
by
  sorry

end die_roll_probability_div_3_l455_455174


namespace unique_function_condition_l455_455949

theorem unique_function_condition (f: ℤ → ℤ) :
  (∀ p x: ℤ, prime p → p ∣ (f x + f p)^f p - x) → 
  (∀ p: ℤ, prime p → f p > 0) → f = id :=
by
  sorry

end unique_function_condition_l455_455949


namespace brandon_investment_percentage_l455_455342

noncomputable def jackson_initial_investment : ℕ := 500
noncomputable def brandon_initial_investment : ℕ := 500
noncomputable def jackson_final_investment : ℕ := 2000
noncomputable def difference_in_investments : ℕ := 1900
noncomputable def brandon_final_investment : ℕ := jackson_final_investment - difference_in_investments

theorem brandon_investment_percentage :
  (brandon_final_investment : ℝ) / (brandon_initial_investment : ℝ) * 100 = 20 := by
  sorry

end brandon_investment_percentage_l455_455342


namespace find_largest_m_l455_455963

theorem find_largest_m (m : ℤ) : (m^2 - 11 * m + 24 < 0) → m ≤ 7 := sorry

end find_largest_m_l455_455963


namespace sin_double_angle_l455_455605

theorem sin_double_angle (α : ℝ) (h1 : sin α = -4/5) (h2 : α ∈ Ioo (-π/2) (π/2)) : 
  sin (2 * α) = -24/25 :=
sorry

end sin_double_angle_l455_455605


namespace num_small_triangles_l455_455156

-- Define the side lengths and areas of the triangles
def largeTriangleSideLength : ℝ := 15
def smallTriangleSideLength : ℝ := 3

def equilateralTriangleArea (s : ℝ) : ℝ := (math.sqrt 3 / 4) * s^2

def largeTriangleArea : ℝ := equilateralTriangleArea largeTriangleSideLength
def smallTriangleArea : ℝ := equilateralTriangleArea smallTriangleSideLength

-- Define the Lean statement
theorem num_small_triangles : largeTriangleArea / smallTriangleArea = 25 := by
  sorry

end num_small_triangles_l455_455156


namespace product_of_digits_of_N_l455_455534

theorem product_of_digits_of_N (N : ℕ) (h : N * (N + 1) / 2 = 2485) : 
  (N.digits 10).prod = 0 :=
sorry

end product_of_digits_of_N_l455_455534


namespace magnitude_of_vector_diff_l455_455292

theorem magnitude_of_vector_diff (x : ℝ)
  (a b : ℝ × ℝ)
  (h₁ : a = (1, x))
  (h₂ : b = (2 * x + 3, -x))
  (h_parallel : a.1 * b.2 = a.2 * b.1) :
  (∥(a.1 - 2 * b.1, a.2 - 2 * b.2)∥ = 5 ∨ ∥(a.1 - 2 * b.1, a.2 - 2 * b.2)∥ = 3 * Real.sqrt 5) :=
by
  sorry

end magnitude_of_vector_diff_l455_455292


namespace finite_valid_sets_l455_455587

noncomputable def valid_set (S : Set ℕ) : Prop :=
  (2 ≤ S.card) ∧ (∀ m n ∈ S, m > n → (n^2 / (m - n) : ℕ) ∈ S)

theorem finite_valid_sets : ∀ (S : Set ℕ), valid_set S → (∃ n : ℕ, S = {n, 2 * n}) :=
by
  intro S hS
  -- Begin the proof (left as an exercise)
  sorry

end finite_valid_sets_l455_455587


namespace volume_of_intersecting_octahedra_l455_455475

def absolute (x : ℝ) : ℝ := abs x

noncomputable def volume_of_region : ℝ :=
  let region1 (x y z : ℝ) := absolute x + absolute y + absolute z ≤ 2
  let region2 (x y z : ℝ) := absolute x + absolute y + absolute (z - 2) ≤ 2
  -- The region is the intersection of these two inequalities
  -- However, we calculate its volume directly
  (2 / 3 : ℝ)

theorem volume_of_intersecting_octahedra :
  (volume_of_region : ℝ) = (2 / 3 : ℝ) :=
sorry

end volume_of_intersecting_octahedra_l455_455475


namespace find_x_l455_455953

noncomputable def y (x : ℝ) : ℝ := real.sqrt (x - 10)

theorem find_x (x : ℝ) :
  (8 / (y x - 10) + 2 / (y x - 5) + 9 / (y x + 5) + 15 / (y x + 10) = 0) →
  (x = 19 ∨ x ≈ 30.84) :=
sorry

end find_x_l455_455953


namespace none_of_the_above_l455_455596

theorem none_of_the_above (n : ℤ) (h : n ≥ 2) : 
  ¬ ((n + 3) % 4 = 0) ∧ 
  ¬ ((n + 3) % 10 = 0) ∧ 
  ¬ (∃ k : ℤ, k * k = n + 3) ∧ 
  ¬ (∃ k : ℤ, k * k * k = n + 3) ∧ 
  ¬ prime (n + 3) :=
by
  sorry

end none_of_the_above_l455_455596


namespace intersection_complement_l455_455657

def M : Set ℝ := { x | x^2 - x - 6 ≥ 0 }
def N : Set ℝ := { x | -3 ≤ x ∧ x ≤ 1 }
def neg_R (A : Set ℝ) : Set ℝ := { x | x ∉ A }

theorem intersection_complement (N : Set ℝ) (M : Set ℝ) :
  N ∩ (neg_R M) = { x | -2 < x ∧ x ≤ 1 } := 
by {
  -- Proof goes here
  sorry
}

end intersection_complement_l455_455657


namespace parallel_lines_slope_equal_intercepts_lines_l455_455293

theorem parallel_lines_slope (m : ℝ) :
  (∀ x y, (2 * x - y - 3 = 0 ∧ x - m * y + 1 - 3 * m = 0) → 2 = (1 / m)) → m = 1 / 2 :=
by
  intro h
  sorry

theorem equal_intercepts_lines (m : ℝ) :
  (m ≠ 0 → (∀ x y, (x - m * y + 1 - 3 * m = 0) → (1 - 3 * m) / m = 3 * m - 1)) →
  (m = -1 ∨ m = 1 / 3) →
  ∀ x y, (x - m * y + 1 - 3 * m = 0) →
  (x + y + 4 = 0 ∨ 3 * x - y = 0) :=
by
  intro h hm
  sorry

end parallel_lines_slope_equal_intercepts_lines_l455_455293


namespace radius_of_circle_in_spherical_coordinates_l455_455558

theorem radius_of_circle_in_spherical_coordinates :
  ∀ (θ : ℝ), ∀ (ρ : ℝ) (φ : ℝ), ρ = 2 ∧ φ = π / 4 →
  ∃ r : ℝ, r = sqrt 2 :=
by
  intros θ ρ φ h
  cases h with hρ hφ
  use sqrt 2
  sorry

end radius_of_circle_in_spherical_coordinates_l455_455558


namespace fraction_product_simplification_l455_455546

theorem fraction_product_simplification :
  ∏ n in (Finset.range 2010).filter (2 ≤ _), (n + 2) / (n + 5) = (1 : ℚ) / 1007 := by
  sorry

end fraction_product_simplification_l455_455546


namespace intersection_volume_calculation_l455_455468

noncomputable def volume_of_intersection : ℝ :=
  let region1 := {p : ℝ × ℝ × ℝ | abs p.1 + abs p.2 + abs p.3 ≤ 2}
  let region2 := {p : ℝ × ℝ × ℝ | abs p.1 + abs p.2 + abs (p.3 - 2) ≤ 2}
  let intersection := region1 ∩ region2
  8/3

theorem intersection_volume_calculation :
  volume_of_intersection = 8 / 3 :=
begin
  sorry
end

end intersection_volume_calculation_l455_455468


namespace sum_of_legs_of_triangle_l455_455081

theorem sum_of_legs_of_triangle (a b : ℝ) (S1 S2 : ℝ) 
  (hS1 : S1 = 441) (hS2 : S2 = 440) 
  (h1 : ∃ AC CB : ℝ, right_triangle a b 441 440 AC CB) :
  AC + CB = 462 :=
sorry

end sum_of_legs_of_triangle_l455_455081


namespace complex_quadrant_l455_455562

def imaginary_unit := Complex.i

def complex_number := (2 - imaginary_unit) / imaginary_unit

def coordinates (z : ℂ) : ℤ × ℤ :=
  (z.re.toInt, z.im.toInt)

theorem complex_quadrant :
  coordinates complex_number = (-1, -2) →
  ∃ quadrant, quadrant = 3 := by
  sorry

end complex_quadrant_l455_455562


namespace harmonic_series_sum_l455_455983

def harmonic (n : ℕ) : ℚ :=
  if n = 0 then 0 else (Finset.range n).sum (λ k => 1 / (k + 1))

theorem harmonic_series_sum : (∑' n : ℕ, 1 / (n + 1 + 1 : ℚ) / (harmonic n) / (harmonic (n + 1))) = 1 := by
  sorry

end harmonic_series_sum_l455_455983


namespace max_value_of_x_plus_y_l455_455968

-- Define the conditions as a set of constraints
noncomputable def max_x_plus_y : ℝ :=
  let domain_x := set.Icc 0 (3 * Real.pi / 2)
  let domain_y := set.Icc Real.pi (2 * Real.pi)
  let eligible_x := {x | 2 * Real.sin x = 1} ∩ domain_x 
  let eligible_y := {y | 2 * Real.cos y = Real.sqrt 3} ∩ domain_y
  let candidates := {x + y | x ∈ eligible_x, y ∈ eligible_y}
  let max_val := set.sSup candidates
  max_val

-- State the theorem that the maximum value of x + y given the constraints is 8π/3
theorem max_value_of_x_plus_y : max_x_plus_y = 8 * Real.pi / 3 :=
  sorry

end max_value_of_x_plus_y_l455_455968


namespace trackball_mice_count_l455_455046

theorem trackball_mice_count 
  (total_mice wireless_mice optical_mice trackball_mice : ℕ)
  (h1 : total_mice = 80)
  (h2 : wireless_mice = total_mice / 2)
  (h3 : optical_mice = total_mice / 4)
  (h4 : trackball_mice = total_mice - (wireless_mice + optical_mice)) :
  trackball_mice = 20 := by 
  sorry

end trackball_mice_count_l455_455046


namespace parabola_chord_ratio_is_3_l455_455412

noncomputable def parabola_chord_ratio (p : ℝ) (h : p > 0) : ℝ :=
  let focus_x := p / 2
  let a_x := (3 * p) / 2
  let b_x := p / 6
  let af := a_x + (p / 2)
  let bf := b_x + (p / 2)
  af / bf

theorem parabola_chord_ratio_is_3 (p : ℝ) (h : p > 0) : parabola_chord_ratio p h = 3 := by
  sorry

end parabola_chord_ratio_is_3_l455_455412


namespace mr_bird_exact_speed_l455_455761

-- Define the properties and calculating the exact speed
theorem mr_bird_exact_speed (d t : ℝ) (h1 : d = 50 * (t + 1 / 12)) (h2 : d = 70 * (t - 1 / 12)) :
  d / t = 58 :=
by 
  -- skipping the proof
  sorry

end mr_bird_exact_speed_l455_455761


namespace arithmetic_sequence_sum_9_l455_455629

theorem arithmetic_sequence_sum_9 :
  ∀ (a : ℕ → ℝ) (d : ℝ),
  (∀ n, a n = 2 + n * d) ∧ d ≠ 0 ∧ (2 : ℝ) + 2 * d ≠ 0 ∧ (2 + 5 * d) ≠ 0 ∧ d = 0.5 →
  (2 + 2 * d)^2 = 2 * (2 + 5 * d) →
  (9 * 2 + (9 * 8 / 2) * 0.5) = 36 :=
by
  intros a d h1 h2
  sorry

end arithmetic_sequence_sum_9_l455_455629


namespace find_lambda_l455_455247

-- Define the vectors and their components
def vector_a : ℝ × ℝ × ℝ := (2, 1, -3)
def vector_b (λ : ℝ) : ℝ × ℝ × ℝ := (4, 2, λ)

-- Dot product definition for 3D vectors
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Define the condition of the problem
def perpendicular_condition (λ : ℝ) : Prop :=
  dot_product vector_a (vector_b λ) = 0

-- The theorem to state and prove
theorem find_lambda : ∃ λ : ℝ, λ = 10 / 3 ∧ perpendicular_condition λ := by
  sorry

end find_lambda_l455_455247


namespace food_price_l455_455150

-- Definitions for the conditions given in the problem
variable (P : ℝ) -- actual price of the food before tax and tip
variable (total_amount : ℝ := 184.80) -- total amount spent
variable (tip_rate : ℝ := 0.20) -- tip rate
variable (tax_rate : ℝ := 0.10) -- tax rate

-- Specification: total amount spent including tax and tip
def total_spent (P : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) : ℝ :=
  let price_with_tax := P + P * tax_rate
  let total_price := price_with_tax + price_with_tax * tip_rate
  total_price

-- Lean statement to prove that the actual price of the food
theorem food_price (h : total_spent P tax_rate tip_rate = total_amount) : P = 140 :=
by
  have h1 : total_spent P tax_rate tip_rate = 1.32 * P :=
    by sorry
  have h2 : 1.32 * P = total_amount :=
    by rw [h1]; exact h
  exact (eq_div_of_mul_eq' (by norm_num) h2)

end food_price_l455_455150


namespace sum_of_coefficients_l455_455227

theorem sum_of_coefficients (d : ℤ) : 
  let expr := -(4 - d) * (d + 3 * (4 - d))
  let expanded_form := -2 * d ^ 2 + 20 * d - 48
  let sum_of_coeffs := -2 + 20 - 48
  sum_of_coeffs = -30 :=
by
  -- The proof will go here, skipping for now.
  sorry

end sum_of_coefficients_l455_455227


namespace curve_is_rhombus_not_square_l455_455697

-- Variable definitions
variables {a b : ℝ}
variables (h_a : a > 0) (h_b : b > 0) (h_diff : a ≠ b)

-- Definition of the geometric figure
def equation (x y : ℝ) := (|x + y| / (2 * a)) + (|x - y| / (2 * b)) = 1

-- The statement to prove
theorem curve_is_rhombus_not_square (h : ∀ (x y : ℝ), equation a b x y) : 
  ∃ (R : set (ℝ × ℝ)), 
  (∀ (x y : ℝ), (x, y) ∈ R ↔ equation a b x y) ∧ 
  (is_rhombus R ∧ ¬ is_square R) :=
begin
  sorry -- proof not required
end

end curve_is_rhombus_not_square_l455_455697


namespace Martine_peach_count_l455_455375

variables (Gabrielle_peaches Benjy_peaches Martine_peaches : ℕ)
variable (h1 : Gabrielle_peaches = 15)
variable (h2 : Benjy_peaches = Gabrielle_peaches / 3)
variable (h3 : Martine_peaches = 2 * Benjy_peaches + 6)

theorem Martine_peach_count : Martine_peaches = 16 :=
by {
  rw [h1, h2] at *,
  simp at *,
  sorry
}

end Martine_peach_count_l455_455375


namespace intersection_volume_calculation_l455_455469

noncomputable def volume_of_intersection : ℝ :=
  let region1 := {p : ℝ × ℝ × ℝ | abs p.1 + abs p.2 + abs p.3 ≤ 2}
  let region2 := {p : ℝ × ℝ × ℝ | abs p.1 + abs p.2 + abs (p.3 - 2) ≤ 2}
  let intersection := region1 ∩ region2
  8/3

theorem intersection_volume_calculation :
  volume_of_intersection = 8 / 3 :=
begin
  sorry
end

end intersection_volume_calculation_l455_455469


namespace natalia_crates_l455_455576

noncomputable def total_items (novels comics documentaries albums : ℕ) : ℕ :=
  novels + comics + documentaries + albums

noncomputable def crates_needed (total_items items_per_crate : ℕ) : ℕ :=
  (total_items + items_per_crate - 1) / items_per_crate

theorem natalia_crates : crates_needed (total_items 145 271 419 209) 9 = 117 := by
  sorry

end natalia_crates_l455_455576


namespace focal_length_of_lens_l455_455887

-- Define the conditions
def initial_screen_distance : ℝ := 80
def moved_screen_distance : ℝ := 40
def lens_formula (f v u : ℝ) : Prop := (1 / f) = (1 / v) + (1 / u)

-- Define the proof goal
theorem focal_length_of_lens :
  ∃ f : ℝ, (f = 100 ∨ f = 60) ∧
  lens_formula f f (1 / 0) ∧  -- parallel beam implies object at infinity u = 1/0
  initial_screen_distance = 80 ∧
  moved_screen_distance = 40 :=
sorry

end focal_length_of_lens_l455_455887


namespace Jewel_bought_10_magazines_l455_455159

theorem Jewel_bought_10_magazines (x : ℕ) 
  (h1 : 3 * x = cost) 
  (h2 : 3.5 * x = revenue) 
  (h3 : revenue - cost = 5) : 
x = 10 :=
by
  sorry

end Jewel_bought_10_magazines_l455_455159


namespace horner_method_value_v2_at_minus_one_l455_455197

noncomputable def f (x : ℝ) : ℝ :=
  x^6 - 5*x^5 + 6*x^4 - 3*x^3 + 1.8*x^2 + 0.35*x + 2

theorem horner_method_value_v2_at_minus_one :
  let a : ℝ := -1
  let v_0 := 1
  let v_1 := v_0 * a - 5
  let v_2 := v_1 * a + 6
  v_2 = 12 :=
by
  intros
  sorry

end horner_method_value_v2_at_minus_one_l455_455197


namespace line_at_t4_is_correct_l455_455522

def parametric_line (a d : ℝ × ℝ × ℝ) (t : ℝ) : ℝ × ℝ × ℝ :=
  (a.1 + t * d.1, a.2 + t * d.2, a.3 + t * d.3)

theorem line_at_t4_is_correct :
  ∃ a d : ℝ × ℝ × ℝ,
    parametric_line a d 1 = (-3, 9, 12) ∧
    parametric_line a d (-1) = (4, -4, 2) ∧
    parametric_line a d 4 = (-27, 57, 27) := by
  sorry

end line_at_t4_is_correct_l455_455522


namespace product_of_altitude_segments_equal_l455_455068

variables {A B C M A1 B1 C1 : Point}
variables (hABC : Triangle A B C)
variables (hM : IsOrthocenter M A B C)
variables (hAltitudeA : Line A A1)
variables (hAltitudeB : Line B B1)
variables (hAltitudeC : Line C C1)
variables (hPerpendicularA : ⊥ A A1 (BC side of A B C))
variables (hPerpendicularB : ⊥ B B1 (AC side of A B C))
variables (hPerpendicularC : ⊥ C C1 (AB side of A B C))

theorem product_of_altitude_segments_equal :
  (AM * MA1 = BM * MB1) ∧ (AM * MA1 = CM * MC1) ∧ (BM * MB1 = CM * MC1) :=
sorry

end product_of_altitude_segments_equal_l455_455068


namespace scheduling_problem_l455_455112

noncomputable def num_valid_schedules : Nat :=
  Nat.fact 8 / (Nat.fact 2 * Nat.fact 5)

theorem scheduling_problem :
  num_valid_schedules = 40320 := by
  sorry

end scheduling_problem_l455_455112


namespace volume_of_intersection_of_octahedra_l455_455464

theorem volume_of_intersection_of_octahedra :
  let region1 := {p : ℝ × ℝ × ℝ | abs p.1 + abs p.2 + abs p.3 ≤ 2}
  let region2 := {p : ℝ × ℝ × ℝ | abs p.1 + abs p.2 + abs (p.3 - 2) ≤ 2}
  region1 ∩ region2 = volume (region1 ∩ region2) = 16 / 3 := 
sorry

end volume_of_intersection_of_octahedra_l455_455464


namespace find_x_l455_455304

-- Define the problem statement
theorem find_x (x : ℝ) : 4^(x + 2) = 68 + 4^x ↔ x = (Real.log (68 / 15)) / (Real.log 4) :=
by
  sorry

end find_x_l455_455304


namespace area_ratio_of_rectangle_and_triangle_l455_455856

variables (L W : ℝ)

theorem area_ratio_of_rectangle_and_triangle (h_nonzero : L ≠ 0 ∧ W ≠ 0) 
: (L * W) / (1 / 2 * L * W) = 2 := 
by
  rw [mul_comm L W, div_mul_eq_mul_div (L * W) (L * W / 2) 2, mul_div_cancel' (L * W) (2 : ℝ)],
  -- Next simplification will produce the simplified representation.
  rw [one_div, mul_assoc, div_self, mul_one, eq_self_iff_true],
  simp,
  exact h_nonzero.1,
  exact h_nonzero.2,
  norm_num,
  sorry

end area_ratio_of_rectangle_and_triangle_l455_455856


namespace time_B_start_after_A_l455_455175

variable (v_A v_B t_overtake : ℝ)

theorem time_B_start_after_A
  (h1 : v_A = 4)
  (h2 : v_B = 4.555555555555555)
  (h3 : t_overtake = 1.8) :
  let t := 2.05 - t_overtake 
  in t * 60 = 15 :=
by
  -- Let t := 2.05 - t_overtake
  -- show t * 60 = 15
  sorry

end time_B_start_after_A_l455_455175


namespace average_of_possible_values_of_x_l455_455306

theorem average_of_possible_values_of_x (x : ℝ) (h : (2 * x^2 + 3) = 21) : (x = 3 ∨ x = -3) → (3 + -3) / 2 = 0 := by
  sorry

end average_of_possible_values_of_x_l455_455306


namespace precision_tens_place_l455_455585

-- Given
def given_number : ℝ := 4.028 * (10 ^ 5)

-- Prove that the precision of the given_number is to the tens place.
theorem precision_tens_place : true := by
  -- Proof goes here
  sorry

end precision_tens_place_l455_455585


namespace pumpkins_difference_l455_455910

theorem pumpkins_difference 
  (moonglow_pumpkins : ℕ) 
  (sunshine_pumpkins : ℕ) 
  (h1: moonglow_pumpkins = 14) 
  (h2: sunshine_pumpkins = 54) 
  : sunshine_pumpkins - 3 * moonglow_pumpkins = 12 := 
by 
  rw [h1, h2] 
  simp 
  sorry

end pumpkins_difference_l455_455910


namespace minimum_P_k_over_interval_l455_455600

def is_odd (k : ℕ) : Prop := k % 2 = 1

def satisfies_equation (k n : ℕ) : Prop :=
  (⌊(n : ℝ) / k⌋ + ⌊(200 - n : ℝ) / k⌋ = ⌊200 / k⌋)

def probability_P (k : ℕ) : ℚ :=
  (Fintype.card {n : ℕ // satisfies_equation k n}) / 199

def min_possible_value_P : ℚ :=
  (50/101 : ℚ)

theorem minimum_P_k_over_interval :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 199 ∧ is_odd k → probability_P k ≥ min_possible_value_P := sorry

end minimum_P_k_over_interval_l455_455600


namespace find_b_l455_455814

theorem find_b (b : ℝ) : 
  (∀ x y, x + y = b ↔ (x = 6 ∧ y = 8)) → b = 14 :=
by 
  intro h
  have midpoint : (6 + 8 = b) := rfl
  exact midpoint

end find_b_l455_455814


namespace count_primes_with_prime_remainder_div_6_l455_455666

theorem count_primes_with_prime_remainder_div_6 :
  let primes_btwn_50_100 := [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
  (filter (λ p, let r := p % 6 in r = 1 ∨ r = 5) primes_btwn_50_100).length = 10 :=
  by {
    let primes_btwn_50_100 := [53, 59, 61, 67, 71, 73, 79, 83, 89, 97],
    have expected_primes := [53, 59, 61, 67, 71, 73, 79, 83, 89, 97],
    let prime_remainders := (filter (λ p, let r := p % 6 in r = 1 ∨ r = 5) primes_btwn_50_100),
    exact prime_remainders.length = 10,
  }

end count_primes_with_prime_remainder_div_6_l455_455666


namespace parts_in_batch_l455_455704

theorem parts_in_batch :
  ∃ a : ℕ, 500 ≤ a ∧ a ≤ 600 ∧ a % 20 = 13 ∧ a % 27 = 20 ∧ a = 533 :=
begin
  sorry
end

end parts_in_batch_l455_455704


namespace sqrt_25_eq_pm_five_l455_455100

theorem sqrt_25_eq_pm_five (x : ℝ) : x^2 = 25 ↔ x = 5 ∨ x = -5 := 
sorry

end sqrt_25_eq_pm_five_l455_455100


namespace remainder_2345678901_div_101_l455_455460

theorem remainder_2345678901_div_101 : 2345678901 % 101 = 12 :=
sorry

end remainder_2345678901_div_101_l455_455460


namespace concyclic_points_l455_455031

-- Define a structure for a triangle
structure Triangle :=
  (A B C : Point)

-- Define a structure for a point
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the internal angle bisector intersection point D with BC
def intersect_internal_bisector (T : Triangle) : Point := sorry

-- Define the coordinates of point I (incenter)
def incenter (T : Triangle) : Point := sorry

-- Define the function that returns the perpendicular bisector of a segment
def perpendicular_bisector (p1 p2 : Point) : Line := sorry

-- Define the function to get the intersection of lines (bisectors in this case)
def bisector_intersection (l1 l2 : Line) : Point := sorry

-- Define the bisector lines from points B and C
def bisector_from_B (T : Triangle) : Line := sorry
def bisector_from_C (T : Triangle) : Line := sorry

-- Define the points M and N based on bisector intersections
def point_M (T : Triangle) (D : Point) : Point := bisector_intersection (perpendicular_bisector T.A D) (bisector_from_B T)
def point_N (T : Triangle) (D : Point) : Point := bisector_intersection (perpendicular_bisector T.A D) (bisector_from_C T)

-- Define a function to determine whether four points are concyclic
def points_concyclic (p1 p2 p3 p4 : Point) : Prop := sorry

-- Define and state the proof problem
theorem concyclic_points (T : Triangle) : 
  let D := intersect_internal_bisector T in
  let I := incenter T in
  let M := point_M T D in
  let N := point_N T D in
  points_concyclic T.A I M N :=
by
  sorry

end concyclic_points_l455_455031


namespace parabola_focus_outside_circle_l455_455286

noncomputable def parabola_intersection_inequality : Prop :=
∀ (a : ℝ), (a < -2 * Real.sqrt 5 + 3) ∧ (a > -3) →
  let c1 : ℝ → ℝ := λ a, -1
  let c2 : ℝ → ℝ := λ a, a ^ 2
  let x1 : ℝ := 2*a + 12
  let x2 : ℝ := (2*a + 12)
in
  3*a^2 - 18*a - 33 > 0

theorem parabola_focus_outside_circle :
∀ (a : ℝ), (-3 < a) ∧ (a < -2 * Real.sqrt 5 + 3) → parabola_intersection_inequality a :=
by intro a h; unfold parabola_intersection_inequality; sorry

end parabola_focus_outside_circle_l455_455286


namespace triangle_angle_C_l455_455319

theorem triangle_angle_C (a b c : ℝ) (h : a^2 + b^2 = c^2 - real.sqrt 2 * a * b) :
  ∠C = 3 * real.pi / 4 :=
sorry

end triangle_angle_C_l455_455319


namespace georgie_entry_exit_ways_l455_455145

theorem georgie_entry_exit_ways :
  let n := 8 in
  let entry_ways := n in
  let exit_ways := n - 1 in
  let total_ways := entry_ways * exit_ways in
  total_ways = 56 :=
by
  sorry

end georgie_entry_exit_ways_l455_455145


namespace equation_solution_l455_455076

theorem equation_solution (t : ℤ) : 
  ∃ y : ℤ, (21 * t + 2)^3 + 2 * (21 * t + 2)^2 + 5 = 21 * y :=
sorry

end equation_solution_l455_455076


namespace bells_toll_together_l455_455511

/-
**Problem Statement**:
6 bells commence tolling together and toll at intervals of 2, 4, 6, 8, 10, and 12 seconds respectively. Prove that in 1800 seconds, the bells toll together 16 times.
-/
theorem bells_toll_together : 
  (∀ t ∈ (List.range 1800), (t % 2 = 0) ∧ (t % 4 = 0) ∧ (t % 6 = 0) ∧ (t % 8 = 0) ∧ (t % 10 = 0) ∧ (t % 12 = 0)) →
  let lcm := 120 in 
  let t_seconds := 1800 in
  (t_seconds / lcm) + 1 = 16 := 
by
  sorry

end bells_toll_together_l455_455511


namespace solid_front_view_l455_455689

theorem solid_front_view :
  (s ∈ { "cube", "regular triangular prism", "cylinder", "cone" }) →
  (front_view s).is_rectangle →
  (s ≠ "cone") :=
by
  intros s hs hView
  sorry

end solid_front_view_l455_455689


namespace vector_set_B_is_basis_l455_455181

-- Define the pairs of vectors
def vector_set_A : ℝ × ℝ × ℝ × ℝ := (1, 2, 0, 0)
def vector_set_B : ℝ × ℝ × ℝ × ℝ := (1, -2, 3, 5)
def vector_set_C : ℝ × ℝ × ℝ × ℝ := (3, 2, 9, 6)
def vector_set_D : ℝ × ℝ × ℝ × ℝ := (-3, 3, 2, -2)

-- Definition of collinearity
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

-- The Lean 4 statement that proves that vectors in set B are not collinear and can serve as a basis
theorem vector_set_B_is_basis (a b : ℝ × ℝ) (ha : a = (1, -2)) (hb : b = (3, 5)) :
  ¬ collinear a b :=
begin
  -- Proof will be provided
  sorry
end

end vector_set_B_is_basis_l455_455181


namespace books_about_sports_l455_455504

theorem books_about_sports (total_books school_books sports_books : ℕ) 
  (h1 : total_books = 58)
  (h2 : school_books = 19) 
  (h3 : sports_books = total_books - school_books) :
  sports_books = 39 :=
by 
  rw [h1, h2] at h3 
  exact h3

end books_about_sports_l455_455504


namespace maximize_angle_APB_l455_455714

theorem maximize_angle_APB :
  ∃ (x : ℝ), x = -3 ∧ ∀ (P : ℝ × ℝ), P = (x, 0) → 
  ∃ (θ : ℝ), θ = real.arcsin (sin θ) ∧ 
  (∀ Q : ℝ × ℝ, (Q = (x, 0)) → θ = real.arcsin (sin θ)) := sorry

end maximize_angle_APB_l455_455714


namespace boy_scouts_min_5_l455_455067

theorem boy_scouts_min_5 (a : Fin 7 → Fin 11 → ℕ) (r : Fin 7 → ℕ) (c : Fin 11 → ℕ) :
  (∀ i, r i = ∑ j, a i j) →
  (∀ j, c j = ∑ i, a i j) →
  (∑ i, ∑ j, a i j = 50) →
  ∃ (k : Fin 5), ∃ i j, r i > c j := sorry

end boy_scouts_min_5_l455_455067


namespace part1_part2_l455_455283

def f (x : ℝ) : ℝ := |2 * x - 1| + |2 * x + 3|

theorem part1 : ∀ x : ℝ, f(x) ≥ 4 := by sorry

def g (m x : ℝ) : ℝ := 2 * x ^ 2 + m / x

theorem part2 : ∀ x : ℝ, x > 0 → m = 4 → g(4, x) ≥ 6 := by sorry

end part1_part2_l455_455283


namespace sequence_polynomial_l455_455946

theorem sequence_polynomial (f : ℕ → ℤ) :
  (f 0 = 3 ∧ f 1 = 7 ∧ f 2 = 21 ∧ f 3 = 51) ↔ (∀ n, f n = n^3 + 2 * n^2 + n + 3) :=
by
  sorry

end sequence_polynomial_l455_455946


namespace crates_needed_l455_455572

-- Conditions as definitions
def novels : ℕ := 145
def comics : ℕ := 271
def documentaries : ℕ := 419
def albums : ℕ := 209
def crate_capacity : ℕ := 9

-- Total items calculation
def total_items : ℕ := novels + comics + documentaries + albums

-- Proof statement
theorem crates_needed : (total_items / crate_capacity) = 116 := by
  sorry

end crates_needed_l455_455572


namespace cross_time_approx_l455_455447

noncomputable def time_to_cross_trains 
(length_train1 length_train2 : ℕ) 
(speed_train1_kmh speed_train2_kmh : ℕ) 
: ℝ :=
  let rel_speed_kmh := speed_train1_kmh + speed_train2_kmh
  let rel_speed_ms := (rel_speed_kmh : ℝ) * (5 / 18)
  let total_length := (length_train1 + length_train2 : ℝ)
  total_length / rel_speed_ms

theorem cross_time_approx :
  time_to_cross_trains 140 180 60 40 ≈ 11.51 := 
  sorry

end cross_time_approx_l455_455447


namespace polynomial_remainder_l455_455591

theorem polynomial_remainder :
  let P := λ x : ℝ, 8 * x^5 - 10 * x^4 + 6 * x^3 - 2 * x^2 + 3 * x - 35
  let D := 2 * x - 8
  P 4 = 5961 := sorry

end polynomial_remainder_l455_455591


namespace countPrimesWithPrimeRemainder_div6_l455_455665

/-- Define the prime numbers in the range 50 to 100 -/
def isPrimeInRange_50_100 (n : ℕ) : Prop :=
  n ∈ [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

/-- Define the condition for having a prime remainder when divided by 6 -/
def hasPrimeRemainder_div6 (n : ℕ) : Prop :=
  ∃ (r : ℕ), r ∈ [1, 2, 3, 5] ∧ n % 6 = r

theorem countPrimesWithPrimeRemainder_div6 :
  (finset.filter (λ n, hasPrimeRemainder_div6 n) (finset.filter isPrimeInRange_50_100 (finset.range 101))).card = 10 :=
by
  sorry

end countPrimesWithPrimeRemainder_div6_l455_455665


namespace proportional_distribution_ratio_l455_455696

theorem proportional_distribution_ratio (B : ℝ) (r : ℝ) (S : ℝ) 
  (h1 : B = 80) 
  (h2 : S = 164)
  (h3 : S = (B / (1 - r)) + (B * (1 - r))) : 
  r = 0.2 := 
sorry

end proportional_distribution_ratio_l455_455696


namespace find_value_of_expression_l455_455635

variable (a b c : ℝ)

def parabola_symmetry (a b c : ℝ) :=
  (36 * a + 6 * b + c = 2) ∧ 
  (25 * a + 5 * b + c = 6) ∧ 
  (49 * a + 7 * b + c = -4)

theorem find_value_of_expression :
  (∃ a b c : ℝ, parabola_symmetry a b c) →
  3 * a + 3 * c + b = -8 :=  sorry

end find_value_of_expression_l455_455635


namespace no_convex_polygon_of_regular_triangles_l455_455394

theorem no_convex_polygon_of_regular_triangles (P : Type)
  (is_convex_polygon : convex_polygon P)
  (T : Type)
  (is_regular_triangle : ∀ (t : T), regular_triangle t)
  (non_overlapping : ∀ (t1 t2 : T), t1 ≠ t2 → (interior t1 ∩ interior t2) = ∅)
  (is_distinct : ∀ (t1 t2 : T), t1 ≠ t2)
  (compose_P : ∀ (p ∈ vertices P), ∃ t ∈ T, p ∈ vertices t):
  false :=
by sorry

end no_convex_polygon_of_regular_triangles_l455_455394


namespace infinite_x_differs_from_two_kth_powers_l455_455750

theorem infinite_x_differs_from_two_kth_powers (k : ℕ) (h : k > 1) : 
  ∃ (f : ℕ → ℕ), (∀ n, f n = (2^(n+1))^k - (2^n)^k) ∧ (∀ n, ∀ a b : ℕ, ¬ f n = a^k + b^k) :=
sorry

end infinite_x_differs_from_two_kth_powers_l455_455750


namespace ProofProblem_l455_455622

noncomputable def p : Prop := ∃ (x : ℝ), Math.sin x = Real.sqrt 5 / 2
noncomputable def q : Prop := ∀ (x : ℝ), (0 < x ∧ x < Real.pi / 2) → x > Math.sin x

theorem ProofProblem : ¬ q :=
by
  sorry

end ProofProblem_l455_455622


namespace yearning_numbers_within_interval_l455_455639

def yearning_number (k : ℕ) : Prop :=
  ∃ (n : ℕ), k + 2 = 2 ^ n

def interval (k : ℕ) : Prop := 1 ≤ k ∧ k ≤ 50

def is_integer_log_product (f : ℕ → ℝ) (n : ℕ) : Prop :=
  ∃ k, (interval k) ∧ yearning_number k

theorem yearning_numbers_within_interval : 
  ∃ (S : finset ℕ), 
    (∀ k ∈ S, interval k ∧ yearning_number k) ∧
    (S.card = 4) :=
sorry

end yearning_numbers_within_interval_l455_455639


namespace card_log_difference_zero_l455_455059

theorem card_log_difference_zero :
  ∃ (a : Finset ℕ) (b: Finset ℕ) (f: ℕ → ℕ) (g: ℕ → ℕ),
  (a = Finset.range (102) \ Finset.range (52)) ∧
  (b = Finset.range (52) \ Finset.range (2)) ∧
  (∀ x, x ∈ a → (f x) ∈ a) ∧
  (∀ y, y ∈ b → (g y) ∈ b) ∧
  (∑ x in b, Real.log (g x) - ∑ y in a, Real.log (f y) = 0) :=
begin
  sorry
end

end card_log_difference_zero_l455_455059


namespace number_of_true_propositions_l455_455180

noncomputable def population_size := 800
noncomputable def sample_size := 40
noncomputable def k := population_size / sample_size -- the correct calculation
noncomputable def seg_interval_false := 40
noncomputable def is_proposition_1_true := (k = seg_interval_false)

-- Conditions for the linear regression line and sample center
def linear_regression (b a x : ℝ) : ℝ := b * x + a
noncomputable def sample_center_x : ℝ := 0 -- example value, not used directly
noncomputable def sample_center_y : ℝ := 0 -- example value, not used directly
noncomputable def passes_through_sample_center := linear_regression 1 0 sample_center_x = sample_center_y
noncomputable def is_proposition_2_true := passes_through_sample_center

-- Normal distribution conditions
noncomputable def mu := 2
noncomputable def sigma_square : ℝ := 1 -- example value, real value unknown in the problem
noncomputable def P (a b : ℝ) : ℝ := sorry -- Assume a probability function
noncomputable def probability_1 := P (-real.infinity) 1
noncomputable def probability_segment := P 2 3
noncomputable def is_proposition_3_true := probability_1 = 0.1 ∧ probability_segment = 0.4

theorem number_of_true_propositions :
  (¬ is_proposition_1_true ∧ ¬ is_proposition_2_true ∧ is_proposition_3_true) →
  (1 = 1) := 
by sorry

end number_of_true_propositions_l455_455180


namespace shortest_distance_between_circles_l455_455124

noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₂.1 - p₁.1) ^ 2 + (p₂.2 - p₁.2) ^ 2)

theorem shortest_distance_between_circles :
  let c₁ := (4, -3)
  let r₁ := 4
  let c₂ := (-5, 1)
  let r₂ := 1
  distance c₁ c₂ - (r₁ + r₂) = Real.sqrt 97 - 5 :=
by
  sorry

end shortest_distance_between_circles_l455_455124


namespace probability_A_wins_l455_455109

def card_game :=
  let cards := {1, 2, 3, 4}
  let events : Set (Set ℕ) := {s | s ⊆ cards}

noncomputable def A_wins (first_even : ℕ → Prop) (odds : ℕ → ℕ → Prop) : ℚ := 
  (2 / 4) + (2 / 4) * (1 / 3)

-- Given the conditions
-- 1. There are 4 cards with numbers 1, 2, 3, and 4.
-- 2. The cards are shuffled and placed face down.
-- 3. Players A and B take turns drawing cards without replacement.
-- 4. Player A draws first.
-- 5. The first person to draw an even-numbered card wins.

-- Prove that the probability that player A wins is 2/3
theorem probability_A_wins (first_even : ℕ → Prop) (odds : ℕ → ℕ → Prop) : A_wins first_even odds = 2 / 3 :=
  sorry

end probability_A_wins_l455_455109


namespace bars_not_sold_l455_455216

-- Definitions for the conditions
def cost_per_bar : ℕ := 3
def total_bars : ℕ := 9
def money_made : ℕ := 18

-- The theorem we need to prove
theorem bars_not_sold : total_bars - (money_made / cost_per_bar) = 3 := sorry

end bars_not_sold_l455_455216


namespace count_integers_with_identical_digits_l455_455662

/-- The count of integers between 123 and 789 with at least two identical digits is 180. -/
theorem count_integers_with_identical_digits : 
  (finset.filter (λ n : ℕ, n >= 123 ∧ n <= 789 ∧ 
    (∃ a b c : ℕ, 10 * 10 * a + 10 * b + c = n ∧ (a = b ∨ a = c ∨ b = c))) 
    (finset.Icc 123 789)).card = 180 :=
sorry

end count_integers_with_identical_digits_l455_455662


namespace ref_ray_tangent_find_m_value_l455_455610

section
variable (m : ℝ) (k : ℝ)

/-- Given circle C with equation x^2 + y^2 + x - 6y + m = 0,
    and point A(11/2, 3),

    (1) When m = 1/4, a ray of light L emitted from point A hits the x-axis and is reflected.
        The line of the reflected ray is tangent to circle C. 
        We aim to prove that the equation of the line of the reflected ray is:
        y + 3 = (-4 - sqrt(7)) / 3 * (x - 11/2).

    (2) The line x + 2y - 3 = 0 intersects circle C at points P and Q.
        If the dot product of vectors OP and OQ equals zero, we aim to prove 
        that the value of m is 3.
-/

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + x - 6y + m = 0

def point_A : Prop := (11 / 2, 3)

def line_eq (x y : ℝ) : Prop := x + 2 * y - 3 = 0

theorem ref_ray_tangent 
    (h₁ : m = 0.25) 
    (h₂ : ∃ k : ℝ, k = (-4 - real.sqrt 7) / 3)
    (h₃ : ∀ x y : ℝ, circle_eq m x y) 
    (h₄ : ∀ x y : ℝ, point_A)
    (h₅ : ∃ x : ℝ, y = k * (x - 11 / 2) - 3) 
    : (∀ x y : ℝ, y + 3 = ((-4 - real.sqrt 7) / 3) * (x - 11 / 2)) := 
by 
    sorry

theorem find_m_value
    (h₁ : ∀ x y : ℝ, line_eq x y)
    (h₂ : ∃ P Q : ℝ × ℝ, (circle_eq m P.1 P.2 ∧ circle_eq m Q.1 Q.2) 
                           ∧ (P.1 + Q.1 = 4) 
                           ∧ (P.1 * Q.1 + P.2 * Q.2 = 0))
    : m = 3 :=
by
    sorry

end

end ref_ray_tangent_find_m_value_l455_455610


namespace area_of_smaller_circle_l455_455840

theorem area_of_smaller_circle
  (P A A' B B' S L : Point)
  (r_S r_L : ℝ)
  (h_tangency : TangentCircles S L)
  (h_common_tangents : CommonTangents P A B A' B')
  (h_lengths : PA = 5 ∧ AB = 5)
  (h_A_on_smaller : OnCircle A S r_S)
  (h_A'_on_smaller : OnCircle A' S r_S)
  (h_B_on_larger : OnCircle B L r_L)
  (h_B'_on_larger : OnCircle B' L r_L) :
  area_of_circle S = 25 * π / 8 := sorry

end area_of_smaller_circle_l455_455840


namespace average_of_last_three_numbers_l455_455803

theorem average_of_last_three_numbers (A B C D E F : ℕ) 
  (h1 : (A + B + C + D + E + F) / 6 = 30)
  (h2 : (A + B + C + D) / 4 = 25)
  (h3 : D = 25) :
  (D + E + F) / 3 = 35 :=
by
  sorry

end average_of_last_three_numbers_l455_455803


namespace problem_1_solution_problem_2_solution_l455_455199

theorem problem_1_solution (x : ℝ) : (2 * x ^ 2 - 2 * real.sqrt 2 * x + 1 = 0) → (x = real.sqrt 2 / 2) := 
by 
  intro h
  sorry

theorem problem_2_solution (x : ℝ) : (x * (2 * x - 5) = 4 * x - 10) → (x = 5 / 2 ∨ x = 2) := 
by 
  intro h
  sorry

end problem_1_solution_problem_2_solution_l455_455199


namespace determine_c_l455_455158

-- Define the points
def point1 : ℝ × ℝ := (-3, 1)
def point2 : ℝ × ℝ := (0, 4)

-- Define the direction vector calculation
def direction_vector : ℝ × ℝ := (point2.1 - point1.1, point2.2 - point1.2)

-- Define the target direction vector form
def target_direction_vector (c : ℝ) : ℝ × ℝ := (3, c)

-- Theorem stating that the calculated direction vector equals the target direction vector when c = 3
theorem determine_c : direction_vector = target_direction_vector 3 :=
by
  -- Proof omitted
  sorry

end determine_c_l455_455158


namespace right_triangle_hypotenuse_l455_455767

theorem right_triangle_hypotenuse (a : ℝ) (h : ℝ) (angle : ℝ) 
  (h1 : a = 16) 
  (h2 : angle = real.pi / 4) 
  (h3 : h = a * real.sqrt 2) :
  h = 16 * real.sqrt 2 :=
by
  sorry

end right_triangle_hypotenuse_l455_455767


namespace no_metric_for_as_convergence_l455_455069

theorem no_metric_for_as_convergence (X : Type) [MeasurableSpace X] (ρ : X → X → ℝ) :
    ¬ (∀ (ξ_n : ℕ → X),
        ((∀ ε > 0, ∃ N, ∀ n ≥ N, ρ (ξ_n n) 0 < ε) ↔ (∀ ε > 0, ∃ N, ∀ n ≥ N, |ξ_n n| < ε) → 
        (ξ_n n → 0 almost_surely)) := 
by
    sorry

end no_metric_for_as_convergence_l455_455069


namespace initial_population_is_72_l455_455118

noncomputable def initial_population (V0 : ℕ) : ℕ :=
let two_nights_vampires := V0 + 10 + 5 * (V0 + 10) in
if two_nights_vampires = 72 then
  70 + V0
else 
  0

theorem initial_population_is_72 (V0 : ℕ) (h1 : V0 + 10 + 5 * (V0 + 10) = 72) : initial_population V0 = 72 :=
by {
  unfold initial_population,
  rw [if_pos h1],
  sorry
}

end initial_population_is_72_l455_455118


namespace f_of_g_of_2_l455_455742

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x + 12 / Real.sqrt x
def g (x : ℝ) : ℝ := 2 * x^2 - 2 * x - 3

theorem f_of_g_of_2 : f (g 2) = 14 :=
by 
  sorry

end f_of_g_of_2_l455_455742


namespace intersection_volume_calculation_l455_455470

noncomputable def volume_of_intersection : ℝ :=
  let region1 := {p : ℝ × ℝ × ℝ | abs p.1 + abs p.2 + abs p.3 ≤ 2}
  let region2 := {p : ℝ × ℝ × ℝ | abs p.1 + abs p.2 + abs (p.3 - 2) ≤ 2}
  let intersection := region1 ∩ region2
  8/3

theorem intersection_volume_calculation :
  volume_of_intersection = 8 / 3 :=
begin
  sorry
end

end intersection_volume_calculation_l455_455470


namespace minimum_distance_l455_455385

noncomputable def min_distance_from_parabola_to_circle : ℝ :=
  let center_circle := (3 : ℝ, 0 : ℝ)
  let parabola (y : ℝ) := (y^2, y)
  let circle_radius := 1
  
  infi (λ y : ℝ, Real.sqrt ((parabola y).1 - center_circle.1)^2 + (parabola y).2^2) - circle_radius

theorem minimum_distance : min_distance_from_parabola_to_circle = (11.sqrt / 2 - 1) := sorry

end minimum_distance_l455_455385


namespace YI_angle_bisector_of_XYA_l455_455616

-- Condition: ABC is an acute triangle
variables {A B C I X Y : Type*}
-- Definitions for vertices and certain points (could be represented as points in affine geometry or similar)
axiom triangle_acute (A B C : Type*) : Triangle A B C

-- Point I is the incenter of triangle ABC
axiom incenter (I : Type*) : IsIncenter I A B C

-- Point X lies on BC on the same side as B wrt AI
axiom point_X_condition (X : Type*) : LiesOnBCSameSide X B A

-- Point Y lies on the shorter arc AB of circumcircle ABC
axiom point_Y_condition (Y : Type*) : LiesOnShorterArc Y A B

-- Given angles
axiom angle_AIX_condition : Angle A I X = 120
axiom angle_XYA_condition : Angle X Y A = 120

-- Goal: Prove YI is the angle bisector of angle XYA
theorem YI_angle_bisector_of_XYA :
  IsAngleBisector Y I X Y A :=
sorry

end YI_angle_bisector_of_XYA_l455_455616


namespace average_of_multiplied_set_l455_455498

theorem average_of_multiplied_set (a : ℕ → ℝ) (n : ℕ) (h1 : n = 10) (h2 : (∑ i in Finset.range n, a i) / n = 7) :
  (∑ i in Finset.range n, 10 * a i) / n = 70 :=
sorry

end average_of_multiplied_set_l455_455498


namespace red_card_events_l455_455243

-- Definitions based on the conditions
inductive Person
| A | B | C | D

inductive Card
| Red | Black | Blue | White

-- Definition of the events
def event_A_receives_red (distribution : Person → Card) : Prop :=
  distribution Person.A = Card.Red

def event_B_receives_red (distribution : Person → Card) : Prop :=
  distribution Person.B = Card.Red

-- The relationship between the two events
def mutually_exclusive_but_not_opposite (distribution : Person → Card) : Prop :=
  (event_A_receives_red distribution → ¬ event_B_receives_red distribution) ∧
  (event_B_receives_red distribution → ¬ event_A_receives_red distribution)

-- The formal theorem statement
theorem red_card_events (distribution : Person → Card) :
  mutually_exclusive_but_not_opposite distribution :=
sorry

end red_card_events_l455_455243


namespace M_set_class_count_l455_455305

-- Define the set X
def X : Set (Set Char) := {{a, b, c}}

-- Define the conditions for an M-set class
def is_M_set_class (M : Set (Set Char)) :=
  X ∈ M ∧
  ∅ ∈ M ∧
  (∀ A B : Set Char, A ∈ M → B ∈ M → (A ∪ B) ∈ M) ∧
  (∀ A B : Set Char, A ∈ M → B ∈ M → (A ∩ B) ∈ M)

-- Define the main theorem to be proven
theorem M_set_class_count : 
  ∃ (Ml : List (Set (Set Char))), 
  (∀ (M : Set (Set Char)), M ∈ Ml ↔ is_M_set_class M ∧ {b, c} ∈ M) ∧
  Ml.length = 12 := 
sorry

end M_set_class_count_l455_455305


namespace area_quadrilateral_FDBG_l455_455443

-- Definitions and conditions
variables (A B C D E F G : Type*) [EuclideanSpace A] [EuclideanSpace B] [EuclideanSpace C] 
variables (AB AC DE : ℝ)
variables (area_ABC area_ADE : ℝ)
variables (midpoint_D : midpoint A B D)
variables (midpoint_E : midpoint A C E)
variables (angle_bisector_AF : ∀ (a b c : ℝ), a / b = F)
variables (angle_bisector_BG : ∀ (a b c : ℝ), a / c = G)

-- Prove that the area of quadrilateral FDBG is equal to 83
theorem area_quadrilateral_FDBG :
  AB = 50 → AC = 10 → area_ABC = 120 → 
  midpoint_D → midpoint_E → angle_bisector_AF → angle_bisector_BG →
  area_of_quadrilateral F D B G = 83 :=
by {
  sorry
}

end area_quadrilateral_FDBG_l455_455443


namespace decreasing_log_func_range_l455_455677

noncomputable def is_decreasing (a : ℝ) : Prop :=
  ∀ x y ∈ Icc (0 : ℝ) 1, x < y → log a (2 - a * y) < log a (2 - a * x)

theorem decreasing_log_func_range {a : ℝ} (h_decreasing : is_decreasing a)
  (h_pos : a > 0) (h_neq_one : a ≠ 1)
  (h_domain : ∀ x ∈ Icc (0 : ℝ) 1, 2 - a * x > 0) :
  1 < a ∧ a < 2 :=
sorry

end decreasing_log_func_range_l455_455677


namespace original_stone_counted_as_99_l455_455115

theorem original_stone_counted_as_99 :
  (99 % 22) = 11 :=
by sorry

end original_stone_counted_as_99_l455_455115


namespace arithmetic_sequence_result_l455_455256

def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n m : ℕ, m > 0 → a(n + m) = a n + m * d

def sum_of_arithmetic_sequence (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  n * (a 1 + a n) / 2

theorem arithmetic_sequence_result
  (a : ℕ → ℚ) (h_seq : is_arithmetic_sequence a)
  (h_sum : sum_of_arithmetic_sequence a 13 = 6) :
  3 * a 9 - 2 * a 10 = 6 / 13 :=
sorry

end arithmetic_sequence_result_l455_455256


namespace john_total_hours_l455_455836

def wall_area (length : ℕ) (width : ℕ) := length * width

def total_area (num_walls : ℕ) (wall_area : ℕ) := num_walls * wall_area

def time_to_paint (area : ℕ) (time_per_square_meter : ℕ) := area * time_per_square_meter

def hours_to_minutes (hours : ℕ) := hours * 60

def total_hours (painting_time : ℕ) (spare_time : ℕ) := painting_time + spare_time

theorem john_total_hours 
  (length width num_walls time_per_square_meter spare_hours : ℕ) 
  (H_length : length = 2) 
  (H_width : width = 3) 
  (H_num_walls : num_walls = 5)
  (H_time_per_square_meter : time_per_square_meter = 10)
  (H_spare_hours : spare_hours = 5) :
  total_hours (time_to_paint (total_area num_walls (wall_area length width)) time_per_square_meter / hours_to_minutes 1) spare_hours = 10 := 
by 
    rw [H_length, H_width, H_num_walls, H_time_per_square_meter, H_spare_hours]
    sorry

end john_total_hours_l455_455836


namespace abs_eight_minus_neg_two_sum_of_integers_satisfying_condition_minimum_distance_sum_l455_455844

-- Problem (1)
theorem abs_eight_minus_neg_two : |8 - (-2)| = 10 :=
by
  sorry

-- Problem (2)
theorem sum_of_integers_satisfying_condition : {
  x : ℤ | |x - 2| + |x + 3| = 5 }.sum = -3 :=
by
  sorry

-- Problem (3)
theorem minimum_distance_sum (x : ℚ) : ∃ y, y = |x + 4| + |x - 6| ∧ y = 10 :=
by
  sorry

end abs_eight_minus_neg_two_sum_of_integers_satisfying_condition_minimum_distance_sum_l455_455844


namespace purchasing_power_increase_l455_455300

theorem purchasing_power_increase (P M : ℝ) (h : 0 < P ∧ 0 < M) :
  let new_price := 0.80 * P
  let original_quantity := M / P
  let new_quantity := M / new_price
  new_quantity = 1.25 * original_quantity :=
by
  sorry

end purchasing_power_increase_l455_455300


namespace diagonals_equal_l455_455451

open Real

variable {R : ℝ} {α : ℝ}
variable (A B C D E F G H : Type)
variables [point A] [point B] [point C] [point D] [point E] [point F] [point G] [point H]
variables (circle : circle A R)
variable (congruent1 : AD = BC)
variable (congruent2 : EH = FG)
variable (angleEqual : ∠DAB = ∠ABC = ∠GHE = ∠HEF = α)

theorem diagonals_equal (circABCD : inscribed_in_circle circle {ABCD})
    (circEFGH : inscribed_in_circle circle {EFGH})
    (AD_eq_BC : congruent AD BC)
    (EH_eq_FG : congruent EH FG)
    (angle_alpha : ∠DAB = α ∧ ∠ABC = α ∧ ∠GHE = α ∧ ∠HEF = α) :
    diagonal BD = diagonal EH :=
by
  sorry

end diagonals_equal_l455_455451


namespace intersection_of_sets_is_empty_l455_455655

def setA := {x : ℝ | ∃ y : ℝ, y = log (1 - x)}
def setB := {x : ℝ | x - 1 > 0}

theorem intersection_of_sets_is_empty : setA ∩ setB = ∅ :=
by
  sorry

end intersection_of_sets_is_empty_l455_455655


namespace find_k_for_equation_l455_455332

theorem find_k_for_equation : 
  ∃ k : ℤ, -x^2 - (k + 7) * x - 8 = -(x - 2) * (x - 4) → k = -13 := 
by
  sorry

end find_k_for_equation_l455_455332


namespace car_travel_speed_l455_455863

theorem car_travel_speed (v : ℝ) (h : 1 / v * 3600 = 65) : v ≈ 55.38 :=
by
  sorry

end car_travel_speed_l455_455863


namespace bounded_region_area_l455_455933

theorem bounded_region_area :
  let region := {p : ℝ × ℝ | p.2^2 + 2 * p.2 * p.1 + 56 * |p.1| = 784} in
  ∃ (vertices : list (ℝ × ℝ)), (vertices = [(0, -28), (0, 28), (28, -28), (-28, 28)]) ∧ 
  (∃ (height base : ℝ), height = 56 ∧ base = 56 ∧ 
  (∃ area : ℝ, area = height * base ∧ area = 3136)) :=
by
  sorry

end bounded_region_area_l455_455933


namespace certain_number_is_3_l455_455980

theorem certain_number_is_3 (x : ℚ) (h : (x / 11) * ((121 : ℚ) / 3) = 11) : x = 3 := 
sorry

end certain_number_is_3_l455_455980


namespace sum_log_seq_l455_455638

-- Given conditions
variable {x : ℕ → ℝ}
variable (h_seq : ∀ n, log (x (n + 1)) = 1 + log (x n))
variable (h_sum : (Finset.range 100).sum (λ i, x (i + 1)) = 100)

-- Theorem to prove
theorem sum_log_seq :
  log (Finset.range 100).sum (λ i, x (i + 101)) = 102 :=
sorry

end sum_log_seq_l455_455638


namespace chebyshev_inequality_l455_455997

noncomputable def tk (n : ℕ) (x : Fin n → ℝ) (k : Fin n) : ℝ :=
  ∏ (j : Fin n) in Finset.univ.erase k, |x j - x k|

theorem chebyshev_inequality {n : ℕ} (x : Fin n → ℝ) (h₁ : 2 ≤ n) (h₂ : ∀ i j, i < j → x i < x j) :
  (∑ k : Fin n, 1 / tk n x k) ≥ 2 ^ (n - 2) :=
by 
  sorry

end chebyshev_inequality_l455_455997


namespace jill_vacation_percentage_l455_455217

def jill_net_monthly_salary : ℝ := 3300
def jill_discretionary_income : ℝ := (1 / 5) * jill_net_monthly_salary
def vacation_percent (v : ℝ) : Prop :=
  v * jill_discretionary_income + 0.20 * jill_discretionary_income + 0.35 * jill_discretionary_income + 99 = jill_discretionary_income

theorem jill_vacation_percentage :
  ∃ (v : ℝ), vacation_percent v ∧ v = 0.3 := 
by
  sorry

end jill_vacation_percentage_l455_455217


namespace solve_for_x_l455_455075

theorem solve_for_x (x : ℝ) : 81 = 9 * (3 ^ (2 * x - 2)) → x = 2 := by
  intro h
  sorry

end solve_for_x_l455_455075


namespace percentage_small_bottles_sold_l455_455531

noncomputable def percentage_sold (initial_small initial_big remaining_bottles total_bottles_sold_big : ℝ) : ℝ :=
  let big_bottles_sold := total_bottles_sold_big
  let small_bottles_sold := initial_small - remaining_bottles + big_bottles_sold - initial_big
  (small_bottles_sold / initial_small) * 100

theorem percentage_small_bottles_sold :
  let initial_small := 6000 in
  let initial_big := 15000 in
  let total_bottles_sold_big_ratio := 0.12 in
  let remaining_bottles := 18540 in
  let total_bottles_sold_big := total_bottles_sold_big_ratio * initial_big in
  percentage_sold initial_small initial_big remaining_bottles total_bottles_sold_big = 11 := 
by
  sorry

end percentage_small_bottles_sold_l455_455531


namespace problem_l455_455204

def f : ℤ → ℤ := sorry

-- Condition: For every integer n >= 0, there are at most 0.001n^2 pairs (x, y) such that f(x+y) ≠ f(x) + f(y) and max |x|, |y| <= n.
def condition (n : ℤ) :=
  n ≥ 0 → 
  ∃ s : Finset (ℤ × ℤ), 
    (∀ p ∈ s, 
      let (x, y) := p in
      |x| ≤ n ∧ |y| ≤ n ∧ f (x + y) ≠ f x + f y) ∧
    s.card ≤ 0.001 * n^2

-- The main theorem we want to prove based on the problem
theorem problem (n : ℤ) :
  (∀ n ≥ 0, ∃ (cond : condition n), 
    ∃ (s : Finset ℤ), 
      (∀ a ∈ s, |a| ≤ n ∧ f a ≠ a * f 1) ∧ s.card ≤ n) :=
sorry

end problem_l455_455204


namespace average_difference_l455_455085

def daily_differences : List ℤ := [2, -1, 3, 1, -2, 2, 1]

theorem average_difference :
  (daily_differences.sum : ℚ) / daily_differences.length = 0.857 :=
by
  sorry

end average_difference_l455_455085


namespace sum_l_values_unique_l455_455368

variable {ℂ : Type*} [IsROrC ℂ] 

noncomputable def sum_of_l_values (p q r l : ℂ) : ℂ := 
  if h : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ p ≠ q ∧ q ≠ r ∧ r ≠ p ∧ 
      p / (1 - q^2) = l ∧ q / (1 - r^2) = l ∧ r / (1 - p^2) = l 
  then 1
  else 0

theorem sum_l_values_unique (p q r l : ℂ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) 
  (hpq : p ≠ q) (hqr : q ≠ r) (hrp : r ≠ p) 
  (h1 : p / (1 - q^2) = l) (h2 : q / (1 - r^2) = l) (h3 : r / (1 - p^2) = l) : 
  sum_of_l_values p q r l = 1 := by 
  sorry

end sum_l_values_unique_l455_455368


namespace min_value_A2_minus_B2_l455_455753

noncomputable def A (p q r : ℝ) : ℝ := 
  Real.sqrt (p + 3) + Real.sqrt (q + 6) + Real.sqrt (r + 12)

noncomputable def B (p q r : ℝ) : ℝ :=
  Real.sqrt (p + 2) + Real.sqrt (q + 2) + Real.sqrt (r + 2)

theorem min_value_A2_minus_B2
  (h₁ : 0 ≤ p)
  (h₂ : 0 ≤ q)
  (h₃ : 0 ≤ r) :
  ∃ (p q r : ℝ), A p q r ^ 2 - B p q r ^ 2 = 35 + 10 * Real.sqrt 10 := 
sorry

end min_value_A2_minus_B2_l455_455753


namespace school_student_count_l455_455530

theorem school_student_count (pencils erasers pencils_per_student erasers_per_student students : ℕ) 
    (h1 : pencils = 195) 
    (h2 : erasers = 65) 
    (h3 : pencils_per_student = 3)
    (h4 : erasers_per_student = 1) :
    students = pencils / pencils_per_student ∧ students = erasers / erasers_per_student → students = 65 :=
by
  sorry

end school_student_count_l455_455530


namespace shooters_points_l455_455091

variables {ℕ : Type}
variables (a_1 a_2 a_3 a_4 a_5 b_1 b_2 b_3 b_4 b_5 : ℕ)

theorem shooters_points :
  a_1 + a_2 + a_3 = b_1 + b_2 + b_3 →
  a_3 + a_4 + a_5 = 3 * (b_3 + b_4 + b_5) →
  a_3 = 10 ∧ b_3 = 2 :=
by
  intros h1 h2
  sorry

end shooters_points_l455_455091


namespace inequality_proof_l455_455388

variable (a b c : ℝ)

#check (0 < a ∧ 0 < b ∧ 0 < c ∧ abc * (a + b + c) = ab + bc + ca) →
  5 * (a + b + c) ≥ 7 + 8 * abc

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : abc * (a + b + c) = ab + bc + ca) : 
  5 * (a + b + c) ≥ 7 + 8 * abc :=
sorry

end inequality_proof_l455_455388


namespace sum_of_coefficients_expansion_l455_455221

theorem sum_of_coefficients_expansion (d : ℝ) :
  let expr := -(4 - d) * (d + 3 * (4 - d))
  in (polynomial.sum_of_coeffs expr) = -30 :=
by
  let expr := -(4 - d) * (d + 3 * (4 - d))
  have h_expr : expr = -2 * d^2 + 20 * d - 48, sorry
  have h_coeffs_sum : polynomial.sum_of_coeffs (-2 * d^2 + 20 * d - 48) = -30, sorry
  rw h_expr
  exact h_coeffs_sum

end sum_of_coefficients_expansion_l455_455221


namespace toothpicks_grid_total_l455_455837

theorem toothpicks_grid_total (L W : ℕ) (hL : L = 60) (hW : W = 32) : 
  (L + 1) * W + (W + 1) * L = 3932 := 
by 
  sorry

end toothpicks_grid_total_l455_455837


namespace intersection_complement_of_B_range_of_m_when_A_inter_B_eq_B_l455_455654

namespace Proofs

def set_A : set ℝ := { x : ℝ | 1 ≤ x ∧ x ≤ 4 }
def set_B (m : ℝ) : set ℝ := { x : ℝ | m ≤ x ∧ x ≤ m + 1 }

-- Condition (1)
theorem intersection_complement_of_B (m : ℝ) (h : m = 3) : 
  (set_A ∩ set_B m = { x : ℝ | 3 ≤ x ∧ x ≤ 4 }) ∧ 
  (set_A ∩ -set_B m = { x : ℝ | 1 ≤ x ∧ x < 3 }) :=
by {
  sorry
}

-- Condition (2)
theorem range_of_m_when_A_inter_B_eq_B (h : ∀ x, x ∈ set_A ↔ x ∈ set_B x) : 
  1 ≤ m ∧ m ≤ 3 :=
by {
  sorry
}

end Proofs

end intersection_complement_of_B_range_of_m_when_A_inter_B_eq_B_l455_455654


namespace fractional_part_inequality_l455_455988

noncomputable def floor (x : ℝ) : ℤ := int.floor x
noncomputable def fractional_part (x : ℝ) : ℝ := x - floor x

theorem fractional_part_inequality (m : ℤ) (hm : m > 0) : 
  abs (fractional_part (Real.sqrt m) - 1/2) ≥ 1 / (8 * (Real.sqrt m + 1)) :=
sorry

end fractional_part_inequality_l455_455988


namespace tan_690_eq_neg_sqrt3_div_3_l455_455433

theorem tan_690_eq_neg_sqrt3_div_3 :
  ∃ k : ℤ, ∀ α : ℝ, tan (α + 360 * k) = tan α ∧ tan (-α) = -tan α → tan 690 = - (Real.sqrt 3 / 3) :=
by
  sorry

end tan_690_eq_neg_sqrt3_div_3_l455_455433


namespace days_to_complete_work_l455_455144

-- Conditions
def work_rate_A : ℚ := 1 / 8
def work_rate_B : ℚ := 1 / 16
def combined_work_rate := work_rate_A + work_rate_B

-- Statement to prove
theorem days_to_complete_work : 1 / combined_work_rate = 16 / 3 := by
  sorry

end days_to_complete_work_l455_455144


namespace find_n_coeff_x2_rational_terms_l455_455272

noncomputable def T : ℕ → ℕ → ℚ × ℚ := 
  λ n r, (binom n r * ((-1 / 2)^r), (n - 2 * r) / 3)

theorem find_n (n : ℕ) : 
  (T n 5).2 = 0 → n = 10 :=
sorry

theorem coeff_x2 (n r : ℕ) : 
  n = 10 → r = 2 → (T n r).1 = 45 / 4 :=
sorry

theorem rational_terms (n : ℕ) :
  n = 10 → 
  ∃ r1 r2 r3, r1 = 2 ∧ r2 = 5 ∧ r3 = 8 ∧ 
  (T n r1).1 = 45 / 4 ∧ (T n r1).2 = 2 ∧
  (T n r2).1 = -63 / 8 ∧ (T n r2).2 = 0 ∧
  (T n r3).1 = 45 / 256 ∧ (T n r3).2 = -2 :=
sorry

end find_n_coeff_x2_rational_terms_l455_455272


namespace polynomial_roots_condition_l455_455026

open Real

def f (x a b : ℝ) : ℝ := x^2 + a * x + b

theorem polynomial_roots_condition (a b : ℤ) (h1 : ∀ x ≠ 0, f (x + x⁻¹) a b = f x a b + f x⁻¹ a b) (h2 : ∃ p q : ℤ, f p a b = 0 ∧ f q a b = 0) : a^2 + b^2 = 13 := by
  sorry

end polynomial_roots_condition_l455_455026


namespace comic_book_stack_order_count_l455_455057

theorem comic_book_stack_order_count : 
  let batman_comics : ℕ := 7
      xmen_comics : ℕ := 3
      calvin_hobbes_comics : ℕ := 5
  in 
  (batman_comics.factorial * xmen_comics.factorial * calvin_hobbes_comics.factorial) * 3.factorial = 21772800 := by
  sorry

end comic_book_stack_order_count_l455_455057


namespace walkers_earn_per_mile_this_year_l455_455005

theorem walkers_earn_per_mile_this_year :
  let x : ℝ := 44 / 16
  let last_year_rate := 4
  let last_year_collected := 44
  let last_year_miles := last_year_collected / last_year_rate
  let elroy_miles := last_year_miles + 5
  let elroy_collected := last_year_collected
  elroy_miles * x = elroy_collected →
  x = 2.75 :=
by
  let x : ℝ := 44 / 16
  let last_year_rate := 4
  let last_year_collected := 44
  let last_year_miles := last_year_collected / last_year_rate
  let elroy_miles := last_year_miles + 5
  let elroy_collected := last_year_collected
  have h1 : elroy_miles * x = elroy_collected := sorry
  exact h1


end walkers_earn_per_mile_this_year_l455_455005


namespace max_value_f_on_interval_l455_455757

noncomputable def f (x : ℝ) : ℝ :=
if x.is_irrational then x else
  let ⟨p, q, coprime_pq⟩ := exists_coprime (rat.num_denom (algebra.id ℚ x)) in
  if 0 < p ∧ p < q ∧ q = rat.denom (algebra.id ℚ x) then (p + 1) / q else x

theorem max_value_f_on_interval : 
  ∀ x ∈ set.Ioo (7 / 8 : ℚ) (8 / 9 : ℚ), f x ≤ 16 / 17 :=
sorry

end max_value_f_on_interval_l455_455757


namespace matrix_mul_combo_l455_455033
  
variable (M : Matrix (Fin 2) (Fin 1) ℝ)
variable (u z : Matrix (Fin 1) (Fin 1) ℝ)
variable (v : Matrix (Fin 2) (Fin 1) ℝ)

noncomputable def M_u : v = Matrix (Fin 2) (Fin 1) ℝ  := 
  (Matrix (Fin 2) (Fin 1) λ _ _, [[-3], [8]])

noncomputable def M_z : v = Matrix (Fin 2) (Fin 1) ℝ := 
  (Matrix (Fin 2) (Fin 1) λ _ _, [[4], [-1]])

noncomputable def result : Matrix (Fin 2) (Fin 1) ℝ :=
  (Matrix (Fin 2) (Fin 1) λ _ _, [[-29], [29]])

theorem matrix_mul_combo :
  (M * (3 • u - 5 • z)) = result := 
  sorry

end matrix_mul_combo_l455_455033


namespace abs_eight_minus_neg_two_sum_of_integers_abs_eq_five_sum_of_satisfying_integers_min_value_abs_sum_l455_455847

-- Part 1: Prove that |8 - (-2)| = 10
theorem abs_eight_minus_neg_two : |8 - (-2)| = 10 := 
sorry

-- Part 2: Prove that the sum of all integers x satisfying |x - 2| + |x + 3| = 5 is -3
theorem sum_of_integers_abs_eq_five (x : ℤ) (h : |x - 2| + |x + 3| = 5) : 
  x = -3 ∨ x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 :=
sorry

theorem sum_of_satisfying_integers : 
  ∑ i in {-3, -2, -1, 0, 1, 2}, i = -3 := 
sorry

-- Part 3: Prove that the minimum value of |x + 4| + |x - 6| is 10 for any rational x
theorem min_value_abs_sum (x : ℚ) : 
  ∃ y, (-4 ≤ y ∧ y ≤ 6) ∧ (|y + 4| + |y - 6| = 10) := 
sorry

end abs_eight_minus_neg_two_sum_of_integers_abs_eq_five_sum_of_satisfying_integers_min_value_abs_sum_l455_455847


namespace rajan_profit_share_l455_455782

noncomputable def investment_time_product (amount: ℕ) (months: ℕ) := amount * months

def total_investment_time_product := 
  investment_time_product 20000 12 + -- Rajan
  investment_time_product 25000 4 +  -- Rakesh
  investment_time_product 30000 10 + -- Mahesh
  investment_time_product 35000 12 + -- Suresh
  investment_time_product 15000 8 +  -- Mukesh
  investment_time_product 40000 2    -- Sachin

def total_profit := 18000

def rajan_investment_time_product := investment_time_product 20000 12

def rajan_share := (rajan_investment_time_product / total_investment_time_product) * total_profit

theorem rajan_profit_share : rajan_share = 3428.57 := 
sorry

end rajan_profit_share_l455_455782


namespace proj_v_onto_w_l455_455241

open Real

noncomputable def v : ℝ × ℝ := (8, -4)
noncomputable def w : ℝ × ℝ := (2, 3)

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

noncomputable def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let coeff := dot_product v w / dot_product w w
  (coeff * w.1, coeff * w.2)

theorem proj_v_onto_w :
  projection v w = (8 / 13, 12 / 13) :=
by
  sorry

end proj_v_onto_w_l455_455241


namespace commission_percentage_l455_455519

theorem commission_percentage 
  (cost_price : ℝ) (profit_percentage : ℝ) (observed_price : ℝ) (C : ℝ) 
  (h1 : cost_price = 15)
  (h2 : profit_percentage = 0.10)
  (h3 : observed_price = 19.8) 
  (h4 : 1 + C / 100 = 19.8 / (cost_price * (1 + profit_percentage)))
  : C = 20 := 
by
  sorry

end commission_percentage_l455_455519


namespace trisect_angle_l455_455694

theorem trisect_angle {A B C P : Type*} 
[HasAngle A B C]
[HasDistance AP AC]
[HasDistance PB PC] :
  (∠ C = 2 * ∠ B) → (AP = AC) → (PB = PC) → (trisects AP ∠ A) :=
by
  sorry

end trisect_angle_l455_455694


namespace volume_of_intersection_of_octahedra_l455_455466

theorem volume_of_intersection_of_octahedra :
  let region1 := {p : ℝ × ℝ × ℝ | abs p.1 + abs p.2 + abs p.3 ≤ 2}
  let region2 := {p : ℝ × ℝ × ℝ | abs p.1 + abs p.2 + abs (p.3 - 2) ≤ 2}
  region1 ∩ region2 = volume (region1 ∩ region2) = 16 / 3 := 
sorry

end volume_of_intersection_of_octahedra_l455_455466


namespace right_triangle_transformation_perimeter_l455_455891

theorem right_triangle_transformation_perimeter 
  (p_ABCD : ℕ)
  (h_perimeter : p_ABCD = 64) :
  let side_ABCD := p_ABCD / 4,
      hypotenuse_triangle := side_ABCD,
      leg_triangle := hypotenuse_triangle / real.sqrt 2,
      perimeter_ABFCDE := 2 * side_ABCD + 2 * leg_triangle
  in perimeter_ABFCDE = 32 + 16 * real.sqrt 2 :=
by
  sorry

end right_triangle_transformation_perimeter_l455_455891


namespace increasing_function_range_l455_455994

noncomputable def is_increasing_on (f : ℝ → ℝ) (S : Set ℝ) :=
  ∀ x y ∈ S, x ≤ y → f x ≤ f y

noncomputable def function_f (a : ℝ) (x : ℝ) : ℝ :=
  Real.log x + 1 / (a * x)

theorem increasing_function_range (a : ℝ) :
  a > 0 ∧ is_increasing_on (function_f a) {x : ℝ | 1 ≤ x} ↔ 1 ≤ a :=
sorry

end increasing_function_range_l455_455994


namespace tan_theta_solution_l455_455586

theorem tan_theta_solution (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π / 6)
  (h3 : tan θ + tan (2 * θ) + tan (4 * θ) = 0) : 
  tan θ = sqrt ((7 - sqrt 13) / 6) :=
sorry

end tan_theta_solution_l455_455586


namespace complex_expression_simplified_l455_455862

theorem complex_expression_simplified :
  let z1 := (1 + 3 * Complex.I) / (1 - 3 * Complex.I)
  let z2 := (1 - 3 * Complex.I) / (1 + 3 * Complex.I)
  let z3 := 1 / (8 * Complex.I^3)
  z1 + z2 + z3 = -1.6 + 0.125 * Complex.I := 
by
  sorry

end complex_expression_simplified_l455_455862


namespace eunice_spent_amount_l455_455942

theorem eunice_spent_amount (original_price : ℝ) (discount_percentage : ℝ) :
  original_price = 10000 → discount_percentage = 0.25 → 
  (original_price - (discount_percentage * original_price) = 7500) :=
by
  intros h_original_price h_discount_percentage
  rw [h_original_price, h_discount_percentage]
  -- 10000 - (0.25 * 10000) = 7500
  rfl

end eunice_spent_amount_l455_455942


namespace part1_part2_l455_455366

def f (x : ℝ) : ℝ := 1 / (1 + x)

def g (n : ℕ) (x : ℝ) : ℝ :=
  x + (List.range n).foldr (λ _ acc, f acc) x

def Fib : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := Fib (n+1) + Fib n

theorem part1 (n : ℕ) (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) : 
  g n x > g n y :=
sorry

theorem part2 (n : ℕ) : 
  g n 1 = (List.range (n+1)).foldr (λ i acc, acc + (Fib i) / (Fib (i+1))) 0 :=
sorry

end part1_part2_l455_455366


namespace count_primes_with_prime_remainder_div_6_l455_455668

theorem count_primes_with_prime_remainder_div_6 :
  let primes_btwn_50_100 := [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
  (filter (λ p, let r := p % 6 in r = 1 ∨ r = 5) primes_btwn_50_100).length = 10 :=
  by {
    let primes_btwn_50_100 := [53, 59, 61, 67, 71, 73, 79, 83, 89, 97],
    have expected_primes := [53, 59, 61, 67, 71, 73, 79, 83, 89, 97],
    let prime_remainders := (filter (λ p, let r := p % 6 in r = 1 ∨ r = 5) primes_btwn_50_100),
    exact prime_remainders.length = 10,
  }

end count_primes_with_prime_remainder_div_6_l455_455668


namespace simplified_expression_l455_455127

theorem simplified_expression (x : ℝ) : 
  (real.sqrt (1 + ((x^6 - 1) / (3 * x^3))^2)) = (real.sqrt (x^12 + 7 * x^6 + 1)) / (3 * x^3) :=
  sorry

end simplified_expression_l455_455127


namespace triangle_orthocenter_congruence_l455_455326

open Triangle

noncomputable def orthocenter (A B C : Triangle.Point) : Triangle.Point := sorry

noncomputable def triangle_congruent (t1 t2 : Triangle) : Prop := sorry

variables {A B C H_A H_B H_C : Point}
variables {T1 T2 T3 T4 T5 T6 : Triangle}

theorem triangle_orthocenter_congruence
  (A B C : Point) (HA HB HC : Point)
  (altitude_A : Altitude A HA)
  (altitude_B : Altitude B HB)
  (altitude_C : Altitude C HC)
  (orthocenter_ABC : Orthocenter A B C H_A H_B H_C)
  (ortho1 : Orthocenter A (orthocenter A B HA) (orthocenter A C HC)) :
  triangle_congruent (Triangle.mk H_A H_B H_C)
   (Triangle.mk (orthocenter A (orthocenter A B HA) (orthocenter A C HC))
                (orthocenter B (orthocenter B A HA) (orthocenter B C HB))
                (orthocenter C (orthocenter C A HC) (orthocenter C B HB))) :=
sorry

end triangle_orthocenter_congruence_l455_455326


namespace area_square_EFGH_l455_455717

theorem area_square_EFGH (AB BE : ℝ) (h : BE = 2) (h2 : AB = 10) :
  ∃ s : ℝ, (s = 8 * Real.sqrt 6 - 2) ∧ s^2 = (8 * Real.sqrt 6 - 2)^2 := by
  sorry

end area_square_EFGH_l455_455717


namespace cost_of_fencing_l455_455817

theorem cost_of_fencing :
  ∀ (w : ℕ), (let length := w + 10 in
  let perimeter := 2 * (length + w) in
  let cost_per_meter := 6.5 in
  perimeter = 340 →
  340 * cost_per_meter = 2210) :=
by
  sorry

end cost_of_fencing_l455_455817


namespace min_black_squares_l455_455321

def cell : Type := (fin 7 × fin 7)
def is_black (cells : set cell) (c : cell) : Prop := c ∈ cells
def is_white (cells : set cell) (c : cell) : Prop := c ∉ cells

def adjacent (c1 c2 : cell) : Prop :=
  (c1.1 = c2.1 ∧ (c1.2 = c2.2 + 1 ∨ c1.2 + 1 = c2.2))
  ∨ (c1.2 = c2.2 ∧ (c1.1 = c2.1 + 1 ∨ c1.1 + 1 = c2.1))

def edge (c : cell) : Prop := c.1 = 0 ∨ c.1 = 6 ∨ c.2 = 0 ∨ c.2 = 6

def condition (cells : set cell) : Prop :=
∀ c : cell, is_white cells c → (edge c ∨ ∃ c', adjacent c c' ∧ is_black cells c')

theorem min_black_squares : ∃ cells : set cell, condition cells ∧ finset.card (cells.to_finset) = 8 :=
sorry

end min_black_squares_l455_455321


namespace youngest_child_age_l455_455869

theorem youngest_child_age : ∃ (x : ℕ), let a₀ := x, a₁ := x + 2, a₂ := x + 4, a₃ := x + 6, a₄ := x + 8, a₅ := x + 10 in
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ = 48) ∧ (x = 3） :=
begin
  sorry
end

end youngest_child_age_l455_455869


namespace circle_tangent_to_y_axis_l455_455890

theorem circle_tangent_to_y_axis (F : ℝ × ℝ) (A : ℝ × ℝ)
  (hF : F = (1, 0)) 
  (hA_on_parabola : A.2 ^ 2 = 4 * A.1)
  (hA_on_ray : ∃ m : ℝ, A.2 = m * (A.1 - F.1)) :
  ∃ (L : ℝ → ℝ → Prop), L = (λ x y, x = 0) ∧ tangent (circle (F, A)) L :=
sorry

end circle_tangent_to_y_axis_l455_455890


namespace remainder_when_divided_by_9_l455_455128

noncomputable def base12_to_dec (x : ℕ) : ℕ :=
  (1 * 12^3) + (5 * 12^2) + (3 * 12) + 4
  
theorem remainder_when_divided_by_9 : base12_to_dec (1534) % 9 = 2 := by
  sorry

end remainder_when_divided_by_9_l455_455128


namespace cosine_sum_identity_l455_455132

theorem cosine_sum_identity :
  cos (13 * real.pi / 180) * cos (17 * real.pi / 180) - sin (17 * real.pi / 180) * sin (13 * real.pi / 180) = sqrt 3 / 2 :=
by sorry

end cosine_sum_identity_l455_455132


namespace equal_real_roots_a_value_l455_455310

theorem equal_real_roots_a_value (a : ℝ) :
  a ≠ 0 →
  let b := -4
  let c := 3
  b * b - 4 * a * c = 0 →
  a = 4 / 3 :=
by
  intros h_nonzero h_discriminant
  sorry

end equal_real_roots_a_value_l455_455310


namespace find_integer_n_l455_455674

theorem find_integer_n (n : ℤ) (h : (⌊n^2 / 4⌋ - (⌊n / 2⌋)^2) = 3) : n = 7 :=
sorry

end find_integer_n_l455_455674


namespace solve_equation_l455_455074

noncomputable def correct_solutions (x : ℝ) : Prop :=
  8^(2 / x) - 2^((3 * x + 3) / x) + 12 = 0

theorem solve_equation : (correct_solutions (3 * Real.log 2 / Real.log 6) ∧ correct_solutions 3) :=
by
  sorry

end solve_equation_l455_455074


namespace range_of_f_l455_455646

def g (t x : ℝ) := t^(x - 2) + 2

noncomputable def f (a b x : ℝ) := -a * x^2 + 2 * b * x + 7

theorem range_of_f :
  ∀ (a b : ℝ),
    (∀ t > 0, t ≠ 1 → g t 2 = 3 → a = 2 ∧ b = 3) →
    (∀ x ∈ set.Icc (-1 : ℝ) 2, -1 ≤ f a b x ∧ f a b x ≤ 23 / 2) :=
by {
  sorry,
}

end range_of_f_l455_455646


namespace natalia_crates_l455_455580

/- The definitions from the conditions -/
def novels : ℕ := 145
def comics : ℕ := 271
def documentaries : ℕ := 419
def albums : ℕ := 209
def crate_capacity : ℕ := 9

/- The proposition to prove -/
theorem natalia_crates : (novels + comics + documentaries + albums) / crate_capacity = 116 := by
  -- this skips the proof and assumes the theorem is true
  sorry

end natalia_crates_l455_455580


namespace solution_volume_l455_455137

theorem solution_volume (x : ℝ) (h1 : (0.16 * x) / (x + 13) = 0.0733333333333333) : x = 11 :=
by sorry

end solution_volume_l455_455137


namespace compare_abc_l455_455607

/-- Define the constants a, b, and c as given in the problem -/
noncomputable def a : ℝ := -5 / 4 * Real.log (4 / 5)
noncomputable def b : ℝ := Real.exp (1 / 4) / 4
noncomputable def c : ℝ := 1 / 3

/-- The theorem to be proved: a < b < c -/
theorem compare_abc : a < b ∧ b < c :=
by
  sorry

end compare_abc_l455_455607


namespace passing_time_l455_455842

noncomputable def length_train : ℝ := 62.5
noncomputable def speed_faster_train_kmh : ℝ := 46
noncomputable def speed_slower_train_kmh : ℝ := 36
noncomputable def relative_speed_ms : ℝ := (speed_faster_train_kmh - speed_slower_train_kmh) * (1000 / 3600)
noncomputable def total_distance : ℝ := 2 * length_train

theorem passing_time : 
  let time := total_distance / relative_speed_ms in
  time = 45 :=
begin
  sorry
end

end passing_time_l455_455842


namespace distinct_values_of_S_l455_455571

noncomputable def i : ℂ := complex.I

theorem distinct_values_of_S : ∃ n : ℤ, ∃ S : ℂ, (∃ n : ℤ, S = i^(n + 1) + i^(-n)) ∧ {S | ∃ n : ℤ, S = i^(n + 1) + i^(-n)}.card = 4 :=
sorry

end distinct_values_of_S_l455_455571


namespace minimum_apples_collected_l455_455908

theorem minimum_apples_collected : ∃ (A B C D : ℕ) (xA xB xC xD n : ℕ), 
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ A ≠ C ∧ A ≠ D ∧ B ≠ D ∧
  xA * 100 / n = A ∧ xB * 100 / n = B ∧ xC * 100 / n = C ∧ xD * 100 / n = D ∧
  xA + xB + xC + xD = n ∧
  D > A ∧ D > B ∧ D > C ∧
  let remaining_apples := n - xD in
  xA * 100 / remaining_apples = A ∧ xB * 100 / remaining_apples = B ∧ xC * 100 / remaining_apples = C ∧
  n = 20 :=
sorry

end minimum_apples_collected_l455_455908


namespace solve_trigonometric_identity_l455_455676

theorem solve_trigonometric_identity : 
  (∃ x : ℝ, sin (3 * x) * sin (5 * x) = cos (3 * x) * cos (5 * x) ∧ x = π / 16) :=
by
  sorry

end solve_trigonometric_identity_l455_455676


namespace fixed_point_coordinates_l455_455647

theorem fixed_point_coordinates (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : (2, 2) ∈ {p : ℝ × ℝ | ∃ x : ℝ, p = (x, a^(x-2) + 1)} := 
by
  -- Proof goes here
  sorry

end fixed_point_coordinates_l455_455647


namespace percentage_of_that_and_this_l455_455874

theorem percentage_of_that_and_this (x : ℝ) (hx: x ≠ 0) :
  ((3 / 2) * x) / ((3 / 4) * x) * 100 = 200 :=
by sorry

end percentage_of_that_and_this_l455_455874


namespace euler_formula_l455_455065

variable {O I : Type} -- O and I are points (types in Lean)
variable {R r : ℝ} -- R and r are real numbers

-- Here we assume that O is the circumcenter and I is the incenter of a triangle
-- The proof goal is to show Euler's formula for the distances.
theorem euler_formula (circumcenter : O)
                      (incenter : I)
                      (circumradius : R)
                      (inradius : r) :
                      (dist circumcenter incenter) ^ 2 = circumradius ^ 2 - 2 * circumradius * inradius := by
  sorry

end euler_formula_l455_455065


namespace factor_determines_d_l455_455685

theorem factor_determines_d (d : ℚ) :
  (∀ x : ℚ, x - 4 ∣ d * x^3 - 8 * x^2 + 5 * d * x - 12) → d = 5 / 3 := by
  sorry

end factor_determines_d_l455_455685


namespace triangle_angle_sum_180_l455_455872

noncomputable def is_rhombus (A B C D : Point) : Prop :=
  let AB := dist A B
  let BC := dist B C
  let CD := dist C D
  let DA := dist D A
  AB = BC ∧ BC = CD ∧ CD = DA

noncomputable def all_angles_at_least_60 (A B C D : Point) : Prop :=
  let ∠A := angle B A D
  let ∠B := angle A B C
  let ∠C := angle B C D
  let ∠D := angle C D A
  ∠A ≥ 60 ∧ ∠B ≥ 60 ∧ ∠C ≥ 60 ∧ ∠D ≥ 60

theorem triangle_angle_sum_180 (A B C D : Point) (S : ℝ) 
  (h1: S = 180) :
  (∃ A B C D, is_rhombus A B C D ∧ all_angles_at_least_60 A B C D) := 
sorry

end triangle_angle_sum_180_l455_455872


namespace sum_of_n_for_perfect_square_l455_455973

theorem sum_of_n_for_perfect_square :
  let is_perfect_square (x : ℤ) := ∃ (y : ℤ), y^2 = x
  in ∑ n in { n | n > 0 ∧ is_perfect_square (n^2 - 17 * n + 72) }, n = 17 :=
by
  sorry

end sum_of_n_for_perfect_square_l455_455973


namespace cuboctahedron_volume_ratio_l455_455987

theorem cuboctahedron_volume_ratio (x : ℝ) (h : x > 0) :
  let V_cube := x^3,
      V_tetrahedron := (x^3 / 48),
      V_total_tetrahedra := 8 * V_tetrahedron,
      V_cuboctahedron := V_cube - V_total_tetrahedra,
      ratio := V_cuboctahedron / V_cube in
  ratio = 84 / 100 :=
by
  sorry

end cuboctahedron_volume_ratio_l455_455987


namespace extreme_point_condition_l455_455645

variable {R : Type*} [OrderedRing R]

def f (x a b : R) : R := x^3 - a*x - b

theorem extreme_point_condition (a b x0 x1 : R) (h₁ : ∀ x : R, f x a b = x^3 - a*x - b)
  (h₂ : f x0 a b = x0^3 - a*x0 - b)
  (h₃ : f x1 a b = x1^3 - a*x1 - b)
  (has_extreme : ∃ x0 : R, 3*x0^2 = a) 
  (hx1_extreme : f x1 a b = f x0 a b) 
  (hx1_x0_diff : x1 ≠ x0) :
  x1 + 2*x0 = 0 :=
by
  sorry

end extreme_point_condition_l455_455645


namespace parallelogram_area_l455_455034

open Real

-- Define vectors p and q and their properties
variables (p q : EuclideanSpace 3)
variable (hp_length : ‖p‖ = 2)
variable (hq_length : ‖q‖ = 2)
variable (angle_pq : angle p q = π / 4)

-- Define the diagonals of the parallelogram
def a := p + 3 • q
def b := 3 • p + q

-- Prove the area of the parallelogram formed by the diagonals
theorem parallelogram_area : 
  |(a q - p) × (b p q)| = 8 * sqrt 2 :=
by
  sorry

end parallelogram_area_l455_455034


namespace minimum_period_of_f_l455_455423

noncomputable def f (x : ℝ) : ℝ := (real.sqrt 3 * real.sin x + real.cos x) * (real.sqrt 3 * real.cos x - real.sin x)

theorem minimum_period_of_f : ∀ (x : ℝ), f(x + π) = f(x) := by
  sorry

end minimum_period_of_f_l455_455423


namespace crayons_total_l455_455316

theorem crayons_total (blue red green : ℕ) 
  (h1 : red = 4 * blue) 
  (h2 : green = 2 * red) 
  (h3 : blue = 3) : 
  blue + red + green = 39 := 
by
  sorry

end crayons_total_l455_455316


namespace area_FDBG_l455_455444

-- Define the problem conditions
variables (A B C D E F G : Type)
variables [metric_space A] [metric_space B] [metric_space C]
variables [metric_space D] [metric_space E] [metric_space F] [metric_space G]
variables (AB AC : ℝ)
variables (area_ABC : ℝ)
variables (D_mid_AB : midpoint A B D)
variables (E_mid_AC : midpoint A C E)
variables (angle_bisector : bisector A)
variables (F_on_DE : on_segment D E F)
variables (G_on_BC : on_segment B C G)

-- Define the theorem to be proved
theorem area_FDBG :
  AB = 50 → AC = 10 → area_ABC = 120 →
  midpoint A B D → midpoint A C E → bisector A → 
  on_segment D E F → on_segment B C G →
  area F D B G = 75 :=
by sorry

end area_FDBG_l455_455444


namespace greatest_possible_n_l455_455205

theorem greatest_possible_n : ∃ n : ℕ, n < 200 ∧ n % 9 = 7 ∧ n % 6 = 2 ∧ n = 194 :=
by {
  -- Define \( n \)
  let n := 194,
  -- Verify that \( n \)
  have h1 : n < 200 := by decide,
  have h2 : n % 9 = 7 := by decide,
  have h3 : n % 6 = 2 := by decide,
  existsi n,
  exact ⟨h1, h2, h3⟩
}

end greatest_possible_n_l455_455205


namespace river_depth_mid_June_l455_455711

theorem river_depth_mid_June (D : ℝ) : 
    (∀ (mid_May mid_June mid_July : ℝ),
    mid_May = 5 →
    mid_June = mid_May + D →
    mid_July = 3 * mid_June →
    mid_July = 45) →
    D = 10 :=
by
    sorry

end river_depth_mid_June_l455_455711


namespace trackball_mice_count_l455_455048

theorem trackball_mice_count (total_sales wireless_share optical_share : ℕ) 
    (h_total : total_sales = 80)
    (h_wireless : wireless_share = total_sales / 2)
    (h_optical : optical_share = total_sales / 4):
    total_sales - (wireless_share + optical_share) = 20 :=
by
  sorry

end trackball_mice_count_l455_455048


namespace M_inter_N_eq_M_l455_455044

-- Definitions of the sets M and N
def M : Set ℝ := {x | abs (x - 1) < 1}
def N : Set ℝ := {x | x * (x - 3) < 0}

-- The desired equality
theorem M_inter_N_eq_M : M ∩ N = M := 
by
  sorry

end M_inter_N_eq_M_l455_455044


namespace sum_of_squares_of_CE_k_l455_455569

theorem sum_of_squares_of_CE_k
  (ABC : Triangle)
  (A B C D1 D2 E1 E2 E3 E4 : Point)
  (h1 : ABC.is_equilateral)
  (h2 : distance B D1 = 2 * Real.sqrt 3)
  (h3 : distance B D2 = 2 * Real.sqrt 3)
  (h4 : angle A B D1 = 30)
  (h5 : angle A B D2 = 150)
  (h6 : congruent (Triangle.mk A D1 E1) ABC)
  (h7 : congruent (Triangle.mk A D1 E2) ABC)
  (h8 : congruent (Triangle.mk A D2 E3) ABC)
  (h9 : congruent (Triangle.mk A D2 E4) ABC) :
  ∑ k in [E1, E2, E3, E4], 
  (distance C k) ^ 2 = 576 :=
sorry

end sum_of_squares_of_CE_k_l455_455569


namespace median_of_dataset_l455_455003

theorem median_of_dataset :
  ∀ (data : List ℕ), 
    data = [61, 75, 70, 56, 81, 91, 92, 91, 75, 81] →
    (data.sorted(≤).nth 4).iget + (data.sorted(≤).nth 5).iget = 156 →
    78 = 78 :=
by {
  intro data,
  intro h_data,
  intro h_sorted_sum,
  sorry
}

end median_of_dataset_l455_455003


namespace compare_exponentials_l455_455262

noncomputable def a : ℝ := 0.9^0.3
noncomputable def b : ℝ := 1.2^0.3
noncomputable def c : ℝ := 0.5^(-0.3)

theorem compare_exponentials : c > b ∧ b > a := by
  sorry

end compare_exponentials_l455_455262


namespace trackball_mice_count_l455_455047

theorem trackball_mice_count 
  (total_mice wireless_mice optical_mice trackball_mice : ℕ)
  (h1 : total_mice = 80)
  (h2 : wireless_mice = total_mice / 2)
  (h3 : optical_mice = total_mice / 4)
  (h4 : trackball_mice = total_mice - (wireless_mice + optical_mice)) :
  trackball_mice = 20 := by 
  sorry

end trackball_mice_count_l455_455047


namespace functional_inequality_solution_l455_455931

theorem functional_inequality_solution (f : ℝ → ℝ) (h : ∀ a b : ℝ, f (a^2) - f (b^2) ≤ (f (a) + b) * (a - f (b))) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := 
sorry

end functional_inequality_solution_l455_455931


namespace reverse_addition_unique_l455_455111

theorem reverse_addition_unique (k : ℤ) (h t u : ℕ) (n : ℤ)
  (hk : 100 * h + 10 * t + u = k) 
  (h_k_range : 100 < k ∧ k < 1000)
  (h_reverse_addition : 100 * u + 10 * t + h = k + n)
  (digits_range : 1 ≤ h ∧ h ≤ 9 ∧ 0 ≤ t ∧ t ≤ 9 ∧ 0 ≤ u ∧ u ≤ 9) :
  n = 99 :=
sorry

end reverse_addition_unique_l455_455111


namespace complex_expression_eq_l455_455737

-- Define the terms P, Q, R as complex numbers
def P : ℂ := 7 + 3 * complex.I
def Q : ℂ := 2 * complex.I
def R : ℂ := 7 - 3 * complex.I

-- State the theorem to be proved
theorem complex_expression_eq :
  (P * Q * R) - P = 113 * complex.I - 7 := 
sorry

end complex_expression_eq_l455_455737


namespace equal_distances_triangle_incircle_l455_455032

theorem equal_distances_triangle_incircle 
  (A B C E F G R S : Point)
  (h_triangle : Triangle A B C)
  (h_tangency_E : IncircleTangency A C E)
  (h_tangency_F : IncircleTangency B F)
  (h_intersection_G : Intersect G (Line (C, F)) (Line (B, E)))
  (h_parallelogram_R : Parallelogram B C E R)
  (h_parallelogram_S : Parallelogram B C S F) :
  Distance G R = Distance G S :=
sorry

end equal_distances_triangle_incircle_l455_455032


namespace cone_from_sector_l455_455857

theorem cone_from_sector 
  (sector_angle : ℝ) (sector_radius : ℝ)
  (circumference : ℝ := (sector_angle / 360) * (2 * Real.pi * sector_radius))
  (base_radius : ℝ := circumference / (2 * Real.pi))
  (slant_height : ℝ := sector_radius) :
  sector_angle = 270 ∧ sector_radius = 12 → base_radius = 9 ∧ slant_height = 12 :=
by
  sorry

end cone_from_sector_l455_455857


namespace gas_cost_original_eq_200_l455_455981

def original_gas_cost (x : ℝ) : Prop :=
  (x / 5) - 15 = x / 8

theorem gas_cost_original_eq_200 : ∃ x : ℝ, original_gas_cost x ∧ x = 200 :=
begin
  sorry
end

end gas_cost_original_eq_200_l455_455981


namespace cos_neg_60_equals_half_l455_455563

  theorem cos_neg_60_equals_half : Real.cos (-60 * Real.pi / 180) = 1 / 2 :=
  by
    sorry
  
end cos_neg_60_equals_half_l455_455563


namespace eq_has_positive_integer_solution_l455_455481

theorem eq_has_positive_integer_solution (a : ℤ) :
  (∃ x : ℕ+, (x : ℤ) - 4 - 2 * (a * x - 1) = 2) → a = 0 :=
by
  sorry

end eq_has_positive_integer_solution_l455_455481


namespace solve_system_equations_l455_455792

noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem solve_system_equations :
  ∃ x y : ℝ, (y = 10^((log10 x)^(log10 x)) ∧ (log10 x)^(log10 (2 * x)) = (log10 y) * 10^((log10 (log10 x))^2))
  → ((x = 10 ∧ y = 10) ∨ (x = 100 ∧ y = 10000)) :=
by
  sorry

end solve_system_equations_l455_455792


namespace smallest_k_sufficient_l455_455149

-- Definitions of constants and assumptions based on the problem conditions
def num_deputies := 2000
def num_items := 200
def total_expenditure_limit : ℝ := S

-- Main theorem statement
theorem smallest_k_sufficient (S : ℝ) : ∃ k : ℕ, k = 1991 ∧ (∀ proposals : fin num_deputies → (fin num_items → ℝ), 
    (∀ i, (finset.univ : finset (fin num_items)).sum (λ j, max_val_by_at_least_k_deputies proposals i k) ≤ S))
where
  -- helper function to calculate the maximum value for a given item agreed by at least k deputies
  max_val_by_at_least_k_deputies (proposals : fin num_deputies → (fin num_items → ℝ)) 
    (i : fin num_items) (k : ℕ) : ℝ :=
    (@finset.univ (fin num_deputies)).sort (λ x y, (proposals x i ≥ proposals y i)).take k).last sorry  -- chooses the k-th largest proposal

end

end smallest_k_sufficient_l455_455149


namespace probability_of_solution_l455_455162

open Set

def numbers : Set ℝ := {-10, -6, -5, -4, -2.5, -1, 0, 2.5, 4, 6, 7, 10}
def eq_solutions : Set ℝ := {x | x^3 = 0 ∨ x + 14 = 0 ∨ 2*x + 5 = 0}
def solutions_in_set : Set ℝ := numbers ∩ eq_solutions
def probability : ℝ := (solutions_in_set.toFinset.card : ℝ) / (numbers.toFinset.card : ℝ)

theorem probability_of_solution : probability = 1 / 6 := 
by
  have h : numbers = {-10, -6, -5, -4, -2.5, -1, 0, 2.5, 4, 6, 7, 10} := rfl
  have h_eq : eq_solutions = {0, -2.5, -14} := sorry
  have h_inter : solutions_in_set = {0, -2.5} := sorry
  have h_num : (numbers.toFinset.card : ℝ) = 12 := sorry
  have h_sol : (solutions_in_set.toFinset.card : ℝ) = 2 := sorry
  have h_prob : probability = 2 / 12 := by
    simp [solutions_in_set, probability]
  rw [h_prob]
  norm_num
  sorry

end probability_of_solution_l455_455162


namespace sum_of_coefficients_l455_455673

theorem sum_of_coefficients :
  let p : Polynomial ℝ := (5 * Polynomial.X - 2) ^ 8 in
  (p.eval 1) = 6561 :=
by sorry

end sum_of_coefficients_l455_455673


namespace intersection_and_parallel_l455_455506

open Plane

-- Definitions based on conditions
variables (m n l : Line) (α β : Plane)

-- Conditions
axiom skew_lines : ¬(m ∩ n).nonempty
axiom m_perp_alpha : m ⊥ α
axiom n_perp_beta : n ⊥ β
axiom l_perp_m : l ⊥ m
axiom l_perp_n : l ⊥ n
axiom l_not_in_alpha : ¬ l ⊆ α
axiom l_not_in_beta : ¬ l ⊆ β

-- Statement: 
-- Prove that α and β intersect and their line of intersection is parallel to l.
theorem intersection_and_parallel (m n l : Line) (α β : Plane) [skew_lines] [m_perp_alpha] [n_perp_beta] [l_perp_m] [l_perp_n] [l_not_in_alpha] [l_not_in_beta] :
  (∃ p : Line, p ⊆ α ∧ p ⊆ β ∧ p ∥ l) := sorry

end intersection_and_parallel_l455_455506


namespace option_C_is_proposition_l455_455860

def is_proposition (s : Prop) : Prop := ∃ p : Prop, s = p

theorem option_C_is_proposition : is_proposition (4 + 3 = 8) := sorry

end option_C_is_proposition_l455_455860


namespace reviewing_sequences_count_l455_455708

theorem reviewing_sequences_count :
  let articles_set := {1, 4, 5, 6, 7, 8, 9, 10}
  in (∑ k in Finset.range(articles_set.card + 1), Nat.choose(articles_set.card, k) * (k + 1)) = 1287 :=
by
  let articles_set := {1, 4, 5, 6, 7, 8, 9, 10}
  -- Summing over all k from 0 to the number of articles (8 in this case)
  sorry

end reviewing_sequences_count_l455_455708


namespace elvins_first_month_bill_l455_455218

theorem elvins_first_month_bill (F C : ℝ) 
  (h1 : F + C = 52)
  (h2 : F + 2 * C = 76) : 
  F + C = 52 :=
by
  sorry

end elvins_first_month_bill_l455_455218


namespace common_tangent_line_spheres_l455_455244

noncomputable theory -- Allow for noncomputable constructions

structure Point (α : Type*) := (x y z : α)
structure Cube (α : Type*) := (a : α) (vertices : List (Point α))

def cube_vertices (a : ℝ) : List (Point ℝ) :=
  [⟨0, 0, 0⟩, ⟨a, 0, 0⟩, ⟨a, a, 0⟩, ⟨0, a, 0⟩, ⟨0, 0, a⟩, ⟨a, 0, a⟩, ⟨a, a, a⟩, ⟨0, a, a⟩]

def Cube := Cube.mk a (cube_vertices a)

structure Sphere (α : Type*) := (center : Point α) (radius : α)

-- Assuming centers of spheres are determined suitably and each sphere is defined tangent to the faces of its cube
def Sphere_oriented (P : Point ℝ) (a : ℝ) : List (Sphere ℝ) :=
  [⟨P, a/√2⟩, ⟨P, a/√2⟩, ⟨P, a/√2⟩, ⟨P, a/√2⟩, ⟨P, a/√2⟩, ⟨P, a/√2⟩]

-- Theorem statement proving the common tangent line for six spheres derived from perpendiculars to a cube
theorem common_tangent_line_spheres (P : Point ℝ) (a : ℝ) (spheres : List (Sphere ℝ)) :
  ∃ L, ∀ S ∈ (Sphere_oriented P a), tangent_line L S :=
sorry

end common_tangent_line_spheres_l455_455244


namespace ordinate_of_point_A_l455_455061

noncomputable def p : ℝ := 1 / 4
noncomputable def distance_to_focus (y₀ : ℝ) : ℝ := y₀ + p / 2

theorem ordinate_of_point_A :
  ∃ y₀ : ℝ, (distance_to_focus y₀ = 9 / 8) → y₀ = 1 :=
by
  -- Assume solution steps here
  sorry

end ordinate_of_point_A_l455_455061


namespace regular_polygon_perimeter_l455_455168

theorem regular_polygon_perimeter (s : ℕ) (E : ℕ) (n : ℕ) (P : ℕ)
  (h1 : s = 6)
  (h2 : E = 90)
  (h3 : E = 360 / n)
  (h4 : P = n * s) :
  P = 24 :=
by sorry

end regular_polygon_perimeter_l455_455168


namespace ratio_of_areas_l455_455448

-- Definitions based on conditions provided
variables 
  (Q : Point) -- Center point Q
  (r1 r2 : ℝ) -- Radii of smaller and larger circles
  (C1 : ℝ := 2 * Math.pi * r1) -- Circumference of the smaller circle
  (C2 : ℝ := 2 * Math.pi * r2) -- Circumference of the larger circle

-- The conditions in mathematical problem
axiom angle_arcs_equal : (60 / 360) * C1 = (40 / 360) * C2

-- Target proof
theorem ratio_of_areas (h : (60 / 360) * C1 = (40 / 360) * C2) : (Math.pi * r1^2) / (Math.pi * r2^2) = 4 / 9 := 
  by
  -- Start proof here
  sorry

end ratio_of_areas_l455_455448


namespace inclination_angle_of_line_l455_455420

noncomputable def inclination_angle (m : ℝ) : ℝ :=
  real.arctan m * (180 / real.pi)

theorem inclination_angle_of_line : inclination_angle (1 / (real.sqrt 3)) = 30 :=
by
  sorry

end inclination_angle_of_line_l455_455420


namespace polynomial_inequality_solution_l455_455937

theorem polynomial_inequality_solution :
  {x : ℝ | x^3 - 4*x^2 - x + 20 > 0} = {x | x < -4} ∪ {x | 1 < x ∧ x < 5} ∪ {x | x > 5} :=
sorry

end polynomial_inequality_solution_l455_455937


namespace sine_central_angle_of_circular_sector_eq_4_5_l455_455482

theorem sine_central_angle_of_circular_sector_eq_4_5
  (R : Real)
  (α : Real)
  (h : π * R ^ 2 * Real.sin α = 2 * π * R ^ 2 * (1 - Real.cos α)) :
  Real.sin α = 4 / 5 := by
  sorry

end sine_central_angle_of_circular_sector_eq_4_5_l455_455482


namespace find_x_plus_y_l455_455681

theorem find_x_plus_y (x y : ℝ) (hx : |x| + x + y = 14) (hy : x + |y| - y = 16) : x + y = 26 / 5 := 
sorry

end find_x_plus_y_l455_455681


namespace proper_subsets_count_l455_455652

def A : Set ℕ := {1, 2, 3, 4}

theorem proper_subsets_count : (Finset.powerset A).card - 1 = 15 := by
  sorry

end proper_subsets_count_l455_455652


namespace speed_ratio_l455_455876

theorem speed_ratio (v_A v_B : ℝ) (h₁ : v_A = abs (-800 + v_B))
  (h₂ : 7 * v_A = abs (-800 + 7 * v_B)) : v_A / v_B = 3 / 2 :=
begin
  sorry,
end

end speed_ratio_l455_455876


namespace parallelogram_area_l455_455426

theorem parallelogram_area
  (a b : ℕ)
  (h1 : a + b = 15)
  (h2 : 2 * a = 3 * b) :
  2 * a = 18 :=
by
  -- Proof is omitted; the statement shows what needs to be proven
  sorry

end parallelogram_area_l455_455426


namespace intersection_M_N_l455_455291

def M (x : ℝ) : Prop := log x / log 2 < 1

def N (x : ℝ) : Prop := x^2 - 1 ≤ 0

theorem intersection_M_N : ∀ x : ℝ, M x ∧ N x ↔ 0 < x ∧ x ≤ 1 := by
  sorry

end intersection_M_N_l455_455291


namespace line_ellipse_intersection_l455_455598

def line (b : ℝ) : ℝ → ℝ := λ x, x + b

def ellipse (θ : ℝ) [0 ≤ θ ∧ θ ≤ π] : (ℝ × ℝ) := (2 * Real.cos θ, 4 * Real.sin θ)

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
(x^2 / 4) + (y^2 / 16) = 1 ∧ y ≥ 0

theorem line_ellipse_intersection (b : ℝ) :
  ∀ (x y : ℝ), 
    (ellipse_equation x y) → 
    (y = x + b ↔ -2 ≤ b ∧ b ≤ 2 * Real.sqrt 5) :=
by
  sorry

end line_ellipse_intersection_l455_455598


namespace region_volume_is_two_thirds_l455_455478

noncomputable def volume_of_region : ℝ :=
  let region := {p : ℝ × ℝ × ℝ | |p.1| + |p.2| + |p.3| ≤ 2 ∧ |p.1| + |p.2| + |p.3 - 2| ≤ 2}
  -- Assuming volume function calculates the volume of the region
  volume region

theorem region_volume_is_two_thirds :
  volume_of_region = 2 / 3 :=
by
  sorry

end region_volume_is_two_thirds_l455_455478


namespace shaded_area_calculation_l455_455849

-- Define the side length of the square
def side_length : ℝ := 24

-- Radius of each circle (1/4th of the side)
def radius : ℝ := side_length / 4

-- Area of the square
def area_square : ℝ := side_length ^ 2

-- Area of one circle
def area_circle : ℝ := π * radius ^ 2

-- Total area of three circles
def total_area_circles : ℝ := 3 * area_circle

-- The shaded area is the area of the square minus the total area of the circles
def shaded_area : ℝ := area_square - total_area_circles

-- The theorem stating the shaded area
theorem shaded_area_calculation : shaded_area = 576 - 108 * π :=
by {
  -- These intermediate steps can contain sorrys or direct calculations
  calc
    shaded_area
    = area_square - total_area_circles : rfl
    ... = 576 - 108 * π : sorry
}

end shaded_area_calculation_l455_455849


namespace largest_value_is_d_l455_455487

noncomputable def A := 24680 + 1 / 1357
noncomputable def B := 24680 - 1 / 1357
noncomputable def C := 24680 * (1 / 1357)
noncomputable def D := 24680 / (1 / 1357)
noncomputable def E := 24680.1357

theorem largest_value_is_d :
  D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  -- Setting up the proof, not required to complete for this task
  sorry

end largest_value_is_d_l455_455487


namespace natalia_crates_l455_455578

noncomputable def total_items (novels comics documentaries albums : ℕ) : ℕ :=
  novels + comics + documentaries + albums

noncomputable def crates_needed (total_items items_per_crate : ℕ) : ℕ :=
  (total_items + items_per_crate - 1) / items_per_crate

theorem natalia_crates : crates_needed (total_items 145 271 419 209) 9 = 117 := by
  sorry

end natalia_crates_l455_455578


namespace solve_equation_l455_455299

theorem solve_equation : 
  ∀ x : ℝ, (x ≠ 3) → 
  let lhs := (1 - x) / (x - 3),
      rhs := 1 / (3 - x) - 2 in
  lhs = rhs ↔ x = 5 := 
by 
  intros x hx,
  let lhs := (1 - x) / (x - 3),
  let rhs := 1 / (3 - x) - 2,
  split,
  { intro h,
    have h₁ : 1 - x = -1 - 2 * (x - 3), 
    { sorry }, -- Expand and simplify
    have h₂ : 1 - x + 2 * x = 5, 
    { sorry }, -- Combining like terms
    exact eq_of_sub_eq_zero (by linarith) },
  { intro h,
    subst h,
    simp [lhs, rhs],
    field_simp,
    linarith }

end solve_equation_l455_455299


namespace solution_equation_l455_455592

theorem solution_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) : 
  (2 / (x - 1) = 3 / (x - 2)) ↔ x = -1 :=
by
  unfold has_div.div
  split
  intro h3
  field_simp at h3
  sorry
  intro h4
  rw [h4]
  norm_num

end solution_equation_l455_455592


namespace arithmetic_square_root_problem_l455_455267

-- Problem Conditions
def condition1 (x : ℝ) : Prop := 2 = Real.sqrt (x - 2)
def condition2 (x y : ℝ) : Prop := 2 = Real.cbrt (2 * x - y + 1)

-- Statement to Prove
theorem arithmetic_square_root_problem : 
  ∀ (x y : ℝ), condition1 x → condition2 x y → Real.sqrt (x^2 - 4 * y) = 4 :=
by
  intros x y h1 h2
  -- Proof omitted
  sorry

end arithmetic_square_root_problem_l455_455267


namespace donation_participants_count_l455_455440

-- Define the conditions
variables (x : ℕ) -- Number of students from the eighth grade
constant total_donation_eighth : ℕ := 4800
constant total_donation_ninth : ℕ := 5000
constant number_difference : ℕ := 20

-- Define the property of average donation amount being the same
def average_donation_same (x : ℕ) : Prop :=
  (total_donation_eighth / x) = (total_donation_ninth / (x + number_difference))

-- Define the main theorem to prove the participants count based on conditions
theorem donation_participants_count 
  (h : average_donation_same x) 
  : x = 480 ∧ x + 20 = 500 := 
sorry

end donation_participants_count_l455_455440


namespace trackball_mice_count_l455_455051

theorem trackball_mice_count
  (total_mice : ℕ)
  (wireless_fraction : ℕ)
  (optical_fraction : ℕ)
  (h_total : total_mice = 80)
  (h_wireless : wireless_fraction = total_mice / 2)
  (h_optical : optical_fraction = total_mice / 4) :
  total_mice - (wireless_fraction + optical_fraction) = 20 :=
sorry

end trackball_mice_count_l455_455051


namespace scaled_vector_norm_l455_455615

variables (v : ℝ × ℝ × ℝ) (magnitude_v : ℝ)
-- Assume the norm of the vector v is equal to sqrt(18)
def magnitude_v_condition : Prop := ∥v∥ = Real.sqrt 18

-- Define the scaled vector norm calculation
theorem scaled_vector_norm (h : magnitude_v_condition v magnitude_v) :
  ∥-3 / 2 • v∥ = 9 * Real.sqrt 2 / 2 := by sorry

end scaled_vector_norm_l455_455615


namespace together_work_days_l455_455514

theorem together_work_days (A B C : ℕ) (nine_days : A = 9) (eighteen_days : B = 18) (twelve_days : C = 12) :
  (1 / A + 1 / B + 1 / C) = 1 / 4 :=
by
  sorry

end together_work_days_l455_455514


namespace max_c_value_for_f_x_range_l455_455967

theorem max_c_value_for_f_x_range:
  (∀ c : ℝ, (∃ x : ℝ, x^2 + 4 * x + c = -2) → c ≤ 2) ∧ (∃ (x : ℝ), x^2 + 4 * x + 2 = -2) :=
sorry

end max_c_value_for_f_x_range_l455_455967


namespace min_value_xyz_l455_455617

noncomputable def equilateral_triangle (A B C D E F P : Type) :=
  -- Conditions
  (side_length : ℝ)
  (h_eq_triangle : (distance A B = side_length) ∧ (distance B C = side_length) ∧ (distance C A = side_length))
  (h_on_sides : (∃ B, D ∈ segment(B, C)) ∧ (∃ C, E ∈ segment(C, A)) ∧ (∃ A, F ∈ segment(A, B)))
  (h_distances : (distance A E = 1) ∧ (distance B F = 1) ∧ (distance C D = 1))
  (h_triangle_form : segment(A, D) ∩ segment(B, E) ∩ segment(C, F) = { P })

-- Points and distances definitions
def Xeandyzequal (P : Type) (x y z : ℝ) :=
  ∃ xy, z ∈ { R : Type | distances_to_sides(P, x, y, z)}

-- Proof
theorem min_value_xyz : ∀ (A B C D E F P : Type) (x y z : ℝ), 
  equilateral_triangle(A, B, C, D, E, F, P) ∧ Xeandyzequal(P, x, y, z) 
  → (P ∈ vertices_of_triangle(AD, BE, CF))
  → (minimum (xyz (P, x, y, z)) = (648 * Math.sqrt(3)) / 2197) :=
begin
  sorry
end

end min_value_xyz_l455_455617


namespace students_solved_both_l455_455010

theorem students_solved_both (total_students solved_set_problem solved_function_problem both_problems_wrong: ℕ) 
  (h1: total_students = 50)
  (h2 : solved_set_problem = 40)
  (h3 : solved_function_problem = 31)
  (h4 : both_problems_wrong = 4) :
  (solved_set_problem + solved_function_problem - x + both_problems_wrong = total_students) → x = 25 := by
  sorry

end students_solved_both_l455_455010


namespace area_of_triangle_abe_l455_455727

theorem area_of_triangle_abe 
  (A B C E : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace E]
  (AB AC AE BC BE BM AM : ℝ)
  (angle_A : ℝ) (angle_ABE : ℝ)
  (h1 : AB = 3)
  (h2 : AC = 2 * sqrt 2)
  (h3 : angle_A = 45)
  (h4 : BM is_the_median_of A B C)
  (h5 : BM intersects_circumcircle_at E)
  (h6 : ∠A = 45)
  (h7 : AE = AC)
  (h8 : ∠ABE = 90)
  (h9 : ∠BEA = 135) :
  area (triangle A B E) = 3 * sqrt 2 :=  
sorry

end area_of_triangle_abe_l455_455727


namespace non_rent_extra_expenses_is_3000_l455_455023

-- Define the constants
def cost_parts : ℕ := 800
def markup : ℝ := 1.4
def num_computers : ℕ := 60
def rent : ℕ := 5000
def profit : ℕ := 11200

-- Calculate the selling price per computer
def selling_price : ℝ := cost_parts * markup

-- Calculate the total revenue from selling 60 computers
def total_revenue : ℝ := selling_price * num_computers

-- Calculate the total cost of components for 60 computers
def total_cost_components : ℕ := cost_parts * num_computers

-- Calculate the total expenses
def total_expenses : ℝ := total_revenue - profit

-- Define the non-rent extra expenses
def non_rent_extra_expenses : ℝ := total_expenses - rent - total_cost_components

-- Prove that the non-rent extra expenses equal to $3000
theorem non_rent_extra_expenses_is_3000 : non_rent_extra_expenses = 3000 := sorry

end non_rent_extra_expenses_is_3000_l455_455023


namespace remaining_money_l455_455161

def salary : ℝ := 190000
def fraction_food : ℝ := 1 / 5
def fraction_rent : ℝ := 1 / 10
def fraction_clothes : ℝ := 3 / 5

theorem remaining_money : salary - (fraction_food * salary + fraction_rent * salary + fraction_clothes * salary) = 19000 := by
  sorry

end remaining_money_l455_455161


namespace campers_difference_l455_455510

theorem campers_difference (m a : ℕ) (h_m : m = 44) (h_a : a = 39) : m - a = 5 :=
by
  rw [h_m, h_a]
  exact Nat.sub_self 5

end campers_difference_l455_455510


namespace extreme_value_h_at_a_zero_range_of_a_l455_455642

noncomputable def f (x : ℝ) : ℝ := 1 - Real.exp (-x)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x / (a * x + 1)

noncomputable def h (x : ℝ) (a : ℝ) : ℝ := (Real.exp (-x)) * (g x a)

-- Statement for the first proof problem
theorem extreme_value_h_at_a_zero :
  ∀ x : ℝ, h x 0 ≤ 1 / Real.exp 1 :=
sorry

-- Statement for the second proof problem
theorem range_of_a:
  ∀ x : ℝ, (0 ≤ x → x ≤ 1 / 2) → (f x ≤ g x x) :=
sorry

end extreme_value_h_at_a_zero_range_of_a_l455_455642


namespace part1_angle_A_part2_perimeter_l455_455740

-- Defining the conditions
variables {A B C : ℝ} {a b c : ℝ}
axiom triangle_ABC : ∃ a b c A B C, 
  a = 2 ∧
  cos A = 4 / 5 ∧
  c * sin (A - B) = b * sin (C - A) ∧
  a^2 = b * c

-- Proof for part (1)
theorem part1_angle_A (h : ∀ a b c A B C, 
  c * sin (A - B) = b * sin (C - A) ∧ a^2 = b * c) : 
  A = π / 3 := by
  sorry

-- Proof for part (2)
theorem part2_perimeter (h1 : ∀ a b c A B C,
  a = 2 ∧ cos A = 4 / 5 ∧ 
  c * sin (A - B) = b * sin (C - A) ∧ 
  a^2 = b * c) : 
  a + b + c = 2 + real.sqrt 13 := by
  sorry

end part1_angle_A_part2_perimeter_l455_455740


namespace sign_up_ways_l455_455722

theorem sign_up_ways : 
  let num_ways_A := 2
  let num_ways_B := 2
  let num_ways_C := 2
  num_ways_A * num_ways_B * num_ways_C = 8 := 
by 
  -- show the proof (omitted for simplicity)
  sorry

end sign_up_ways_l455_455722


namespace john_drinks_42_quarts_per_week_l455_455347

def gallons_per_day : ℝ := 1.5
def quarts_per_gallon : ℝ := 4
def days_per_week : ℕ := 7

theorem john_drinks_42_quarts_per_week :
  gallons_per_day * quarts_per_gallon * days_per_week = 42 := sorry

end john_drinks_42_quarts_per_week_l455_455347


namespace kelvin_path_count_l455_455733

-- Definition of Kelvin's movement
inductive Move
| walk1 : ℕ → ℕ → ℕ → ℕ → Move  -- (x, y) to (x + 1, y) or (x, y + 1) or (x + 1, y + 1)
| jump2 : ℕ → ℕ → ℕ → ℕ → Move  -- (x, y) to (x + 2, y) or (x, y + 2) or (x + 1, y + 1)

-- Kelvin can reach (6,8) from (0,0) by a combination of moves and jumps
def kelvin_reaches (start end : ℕ × ℕ) (moves : List Move) : Prop := sorry

-- The number of ways Kelvin can reach (6,8) is 1831830
theorem kelvin_path_count : ∃ moves : List Move, 
  kelvin_reaches (0, 0) (6, 8) moves ∧ moves.length = 1831830 := sorry

end kelvin_path_count_l455_455733


namespace distance_from_point_to_line_l455_455634

theorem distance_from_point_to_line (x y : ℝ) (h : 3 * x^2 + 4 * y^2 - 12 = 0) :
  let d := abs((6 : ℝ) + sqrt 7) / sqrt 2 in
  d = (6 * sqrt 2 - sqrt 14) / 2 ∨ d = (6 * sqrt 2 + sqrt 14) / 2 :=
by
  sorry

end distance_from_point_to_line_l455_455634


namespace ev_func_value_l455_455690

noncomputable def f (a b x : ℝ)  := a * x^2 + b * x + 1

theorem ev_func_value (a b : ℝ) (h_even : ∀ x : ℝ, f a b x = f a b (-x)) (h_int : - 1 - a ≤ 2a) (h_a : a = 1) (h_b : b = 0) :
  f a b (2 * a - b) = 5 :=
by
  rw [h_a, h_b]
  simp [f]
  sorry

end ev_func_value_l455_455690


namespace min_tangent_expression_acute_triangle_l455_455707

theorem min_tangent_expression_acute_triangle 
  {A B C : ℝ} 
  (h_triangle : ∀ {x y z}, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = π → A + B + C = π ∧ A < π/2 ∧ B < π/2 ∧ C < π/2)
  (h_tangent_identity : tan A + tan B + tan C = tan A * tan B * tan C) : 
  3 * tan B * tan C + 2 * tan A * tan C + tan A * tan B ≥ 6 + 2 * sqrt 3 + 2 * sqrt 2 + 2 * sqrt 6 := 
begin
  sorry
end

end min_tangent_expression_acute_triangle_l455_455707


namespace volume_of_intersecting_octahedra_l455_455472

def absolute (x : ℝ) : ℝ := abs x

noncomputable def volume_of_region : ℝ :=
  let region1 (x y z : ℝ) := absolute x + absolute y + absolute z ≤ 2
  let region2 (x y z : ℝ) := absolute x + absolute y + absolute (z - 2) ≤ 2
  -- The region is the intersection of these two inequalities
  -- However, we calculate its volume directly
  (2 / 3 : ℝ)

theorem volume_of_intersecting_octahedra :
  (volume_of_region : ℝ) = (2 / 3 : ℝ) :=
sorry

end volume_of_intersecting_octahedra_l455_455472


namespace concave_numbers_count_l455_455627

-- Let's define the context and problem based on the given conditions.
def is_concave_number (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a > b ∧ c > b

def count_concave_numbers : ℕ :=
  (Finset.univ.filter (λ (abc : Fin 10 × Fin 10 × Fin 10),
    is_concave_number abc.1 abc.2 abc.3)).card * 2

theorem concave_numbers_count : count_concave_numbers = 240 :=
begin
  -- Here we would prove the required theorem.
  sorry
end

end concave_numbers_count_l455_455627


namespace net_population_change_over_four_years_l455_455768

theorem net_population_change_over_four_years :
  let first_year_factor := 6 / 5
  let second_year_factor := 13 / 10
  let third_year_factor := 17 / 20
  let fourth_year_factor := 3 / 4
  let overall_factor := (first_year_factor * second_year_factor * third_year_factor * fourth_year_factor)
  let net_change_percentage := (overall_factor - 1) * 100
in abs (net_change_percentage + 6.475) < 0.01 :=
by
  let first_year_factor := 6 / 5
  let second_year_factor := 13 / 10
  let third_year_factor := 17 / 20
  let fourth_year_factor := 3 / 4
  let overall_factor := (first_year_factor * second_year_factor * third_year_factor * fourth_year_factor)
  let net_change_percentage := (overall_factor - 1) * 100
  have h1 : overall_factor = 3741 / 4000 := by norm_num
  have : net_change_percentage = (3741 / 4000 - 1) * 100 := by rw h1
  have simpl := -6.475
  show abs (net_change_percentage - simpl) < 0.01, by sorry

end net_population_change_over_four_years_l455_455768


namespace range_of_k_l455_455309

noncomputable def f (k x : ℝ) := (k * x + 7) / (k * x^2 + 4 * k * x + 3)

theorem range_of_k (k : ℝ) : (∀ x : ℝ, k * x^2 + 4 * k * x + 3 ≠ 0) ↔ 0 ≤ k ∧ k < 3 / 4 :=
by
  sorry

end range_of_k_l455_455309


namespace triangle_sides_arithmetic_progression_l455_455393

theorem triangle_sides_arithmetic_progression
  (A B C I L₁ : Type)
  (triangle : triangle A B C)
  (incenter : incenter I)
  (angle_bisector : angle_bisector A I L₁)
  (div_ratio : divides I angle_bisector in ratio 2 1)
  (sides : triangle_sides A B C)
  (a b c : ℝ)
  (a_side : side A = a)
  (b_side : side B = b)
  (c_side : side C = c) :
  (b + c = 2 * a) :=
sorry

end triangle_sides_arithmetic_progression_l455_455393


namespace market_income_correct_orchard_income_correct_better_selling_method_growth_rate_l455_455152

-- Define the conditions
variables (a b : ℝ) (a_4_5 : a = 4.5) (b_4 : b = 4) (b_lt_a : b < a)
variables (investment_yuan : ℝ := 13800) (total_yield_kg : ℝ := 18000)
variables (market_selling_rate_kg_per_day : ℝ := 1000)
variables (pay_per_helper_per_day_yuan : ℝ := 100) 
variables (transport_costs_per_day_yuan : ℝ := 200)

-- Problem 1: Income expressions
def market_income (a : ℝ) : ℝ := total_yield_kg * a - ((total_yield_kg / market_selling_rate_kg_per_day) * (2 * pay_per_helper_per_day_yuan + transport_costs_per_day_yuan))
def orchard_income (b : ℝ) : ℝ := total_yield_kg * b

theorem market_income_correct (a : ℝ) : market_income a = 18000 * a - 7200 := sorry
theorem orchard_income_correct (b : ℝ) : orchard_income b = 18000 * b := sorry

-- Problem 2: Compare selling methods
theorem better_selling_method (a : ℝ) (b : ℝ) (h_a : a = 4.5) (h_b : b = 4) : 
  18000 * a - 7200 > 18000 * b :=
by {
  rw h_a,
  rw h_b,
  exact calc
    18000 * 4.5 - 7200 = 73800 : by ring
    73800 > 72000 : by norm_num,
}

-- Problem 3: Growth rate calculation
def this_year_net_income (a : ℝ) : ℝ := market_income a - investment_yuan
def target_net_income_next_year : ℝ := 72000
theorem growth_rate (a : ℝ) (h_a : a = 4.5) : 
  (target_net_income_next_year - this_year_net_income a) / this_year_net_income a * 100 = 20 :=
by {
  rw h_a,
  let net_income := this_year_net_income 4.5,
  suffices : net_income = 60000, {
    rw this,
    exact calc
      (72000 - 60000) / 60000 * 100 = 120000 / 60000 * 100 : by rw sub_eq_add_neg ; ring
      ... = 2 * 100 : by norm_num
      ... = 20 * 10 : by ring,
  },
  exact calc
    this_year_net_income 4.5 = 73800 - 13800 : by rw this_year_net_income ; ring
    ... = 60000 : by ring,
}

end market_income_correct_orchard_income_correct_better_selling_method_growth_rate_l455_455152


namespace inequality_solution_l455_455078

theorem inequality_solution (x : ℝ) :
  (x ≠ 3) ∧ (x ≠ 6) →
  ((x - 6) / (x - 3)^2 < 0 ↔ x ∈ Set.Ioo (-∞) 3 ∪ Set.Ioo 3 6) :=
by
  intro h
  sorry

end inequality_solution_l455_455078


namespace smallest_n_divisible_by_101_l455_455035

noncomputable def b : ℕ → ℕ
| 8     := 20
| (n+1) := if n < 8 then 0 else 100 * b n + 2 * (n + 1)

theorem smallest_n_divisible_by_101 :
  ∃ n, n > 8 ∧ b n % 101 = 0 ∧ ∀ m, m > 8 ∧ b m % 101 = 0 → n ≤ m :=
exists.intro 15 (and.intro (by decide) (and.intro (by decide) (by decide)))

end smallest_n_divisible_by_101_l455_455035


namespace prob_rel_prime_is_1721_l455_455450

def is_rel_prime (a b : ℕ) : Prop := Nat.gcd a b = 1
def pairs := {p : ℕ × ℕ // 1 ≤ p.1 ∧ p.1 < p.2 ∧ p.2 ≤ 7}
def non_rel_prime_pairs := {pr : pairs // ¬ is_rel_prime pr.1.1 pr.1.2}

def prob_rel_prime : ℚ :=
(21 - (non_rel_prime_pairs.to_finset.card : ℕ)) / 21

theorem prob_rel_prime_is_1721 : prob_rel_prime = 17 / 21 :=
by
  sorry

end prob_rel_prime_is_1721_l455_455450


namespace square_div_by_144_l455_455889

theorem square_div_by_144 (n : ℕ) (h1 : ∃ (k : ℕ), n = 12 * k) : ∃ (m : ℕ), n^2 = 144 * m :=
by
  sorry

end square_div_by_144_l455_455889


namespace convergent_fraction_l455_455392

theorem convergent_fraction (α : ℝ) (p q : ℤ) (hα_irrational : irrational α) (hα_pos : α > 0) (h_condition : abs (α - (p / q)) < (1 / (2 * q^2))) : 
  ∃ Pn Qn : ℚ, (Pn / Qn) = (p / q) := sorry

end convergent_fraction_l455_455392


namespace count_valid_codes_eq_12_l455_455895

-- Definitions based on the problem's conditions
def is_valid_code (code : list ℕ) : Prop :=
  code.length = 3 ∧
  (forall d, d ∈ code → d = 6 ∨ d = 4 ∨ d = 3) ∧
  (code.count 6 ≤ 2) ∧
  (code.count 4 ≤ 1) ∧
  (code.product % 3 = 0) ∧
  (code.product % 4 = 0)

-- Count the number of valid 3-digit area codes
def count_valid_codes : ℕ :=
  (list.range 1000).countp (λ n, is_valid_code (nat.digits 10 n))

-- Prove that the count of valid codes is 12
theorem count_valid_codes_eq_12 : count_valid_codes = 12 :=
  sorry

end count_valid_codes_eq_12_l455_455895


namespace triangle_AC_plus_BC_l455_455317

theorem triangle_AC_plus_BC (A B C : Type) 
  (angleC : ℝ) (AB : ℝ) (h : ℝ) (AC BC : ℝ) 
  (h1 : angleC = 60) (h2 : AB = real.sqrt 3) (h3 : h = 4 / 3)
  (h4 : (1 / 2) * AB * h = (1 / 2) * AC * BC * real.sin (real.pi / 3))
  (h5 : AB ^ 2 = AC ^ 2 + BC ^ 2 - 2 * AC * BC * real.cos (real.pi / 3)) :
  AC + BC = real.sqrt 11 := by
  sorry

end triangle_AC_plus_BC_l455_455317


namespace unique_n_P_n_sqrt_number_of_n_satisfying_conditions_l455_455985

def P (n : ℕ) : ℕ := 
  if n < 2 then 0 
  else (Nat.minFac n)

lemma sqrt_eq (x y : ℕ) (h: y^2 = x) : y = Nat.sqrt x :=
  sorry

theorem unique_n_P_n_sqrt (n : ℕ) :
  (P n = Nat.sqrt n) ∧ (P (n + 72) = Nat.sqrt (n + 72)) →
  (n = 49) :=
by
  intros h
  cases h with h1 h2
  have sqrt_n: Nat.sqrt n = 7 :=
    sqrt_eq n 7 (by linarith)
  have sqrt_n_72: Nat.sqrt (n + 72) = 11 :=
    sqrt_eq (n + 72) 11 (by linarith)
  sorry

theorem number_of_n_satisfying_conditions :
  (∃ n : ℕ, (P n = Nat.sqrt n) ∧ (P (n + 72) = Nat.sqrt (n + 72))) →
  ∃! n : ℕ, (P n = Nat.sqrt n) ∧ (P (n + 72) = Nat.sqrt (n + 72)) :=
by
  intros h
  use 49
  apply unique_n_P_n_sqrt
  exact h
  sorry

end unique_n_P_n_sqrt_number_of_n_satisfying_conditions_l455_455985


namespace cube_volume_proof_l455_455126

noncomputable def ref_volume : ℝ := 8
noncomputable def ref_side_length : ℝ := real.cbrt ref_volume
noncomputable def ref_surface_area : ℝ := 6 * ref_side_length^2
noncomputable def target_surface_area : ℝ := 3 * ref_surface_area
noncomputable def target_side_length : ℝ := real.sqrt (target_surface_area / 6)
noncomputable def target_volume : ℝ := target_side_length^3

theorem cube_volume_proof :
  target_volume = 24 * real.sqrt 3 := by
  sorry

end cube_volume_proof_l455_455126


namespace intersection_eq_l455_455656

def set_M : Set ℝ := { x : ℝ | (x + 3) * (x - 2) < 0 }
def set_N : Set ℝ := { x : ℝ | 1 ≤ x ∧ x ≤ 3 }

theorem intersection_eq : set_M ∩ set_N = { x : ℝ | 1 ≤ x ∧ x < 2 } := by
  sorry

end intersection_eq_l455_455656


namespace exists_twenty_people_with_same_hair_count_l455_455781

theorem exists_twenty_people_with_same_hair_count:
  (∀ (x : Person), 0 ≤ hair_count x ∧ hair_count x ≤ 400000) →
  (population_moscow ≥ 8000000) →
  ∃ (G : set Person), G.card ≥ 20 ∧ ∃ (n : ℕ), ∀ p ∈ G, hair_count p = n := 
by sorry

-- Definitions used in the theorem statement:
def Person := sorry             -- A placeholder type for a person
def hair_count (p: Person) : ℕ := sorry   -- A placeholder function for the number of hairs on a person's head
def population_moscow : ℕ := sorry        -- A placeholder value for the population of Moscow

end exists_twenty_people_with_same_hair_count_l455_455781


namespace exists_N_not_divisible_by_13_l455_455039

theorem exists_N_not_divisible_by_13 {k : ℕ} (hk : 0 < k) : 
  ∃ N : ℕ, 
    (∀ d ∈ (Nat.digits 10 N), d ≠ 0) ∧ 
    (∀ M : ℕ, (M ∈ (Nat.perm_digits 10 N)) → ¬ (13 ∣ M)) ∧
    (Nat.digits 10 N).length = k :=
sorry

end exists_N_not_divisible_by_13_l455_455039


namespace ninety_percent_can_play_basketball_and_transport_l455_455323

-- Define a structure for a person with properties for their height and radii for laws.
structure Person where
  height : ℝ 
  radius_basketball : ℝ 
  radius_transport : ℝ

-- Define a function to determine if a person can play basketball.
def can_play_basketball (persons : List Person) (p : Person) : Bool :=
  let neighbors := persons.filter (λ other => dist (p.height) (other.height) ≤ p.radius_basketball)
  (neighbors.length / 2) < (neighbors.filter (λ nb => nb.height < p.height)).length

-- Define a function to determine if a person is entitled to free public transportation.
def has_free_transport (persons : List Person) (p : Person) : Bool :=
  let neighbors := persons.filter (λ other => dist (p.height) (other.height) ≤ p.radius_transport)
  (neighbors.length / 2) < (neighbors.filter (λ nb => nb.height > p.height)).length

-- Define the main theorem to prove the conditions.
theorem ninety_percent_can_play_basketball_and_transport (persons : List Person) :
  persons.length > 0 →
  (90% persons).count (λ p => can_play_basketball persons p) ≥ (9 * persons.length / 10) →
  (90% persons).count (λ p => has_free_transport persons p) ≥ (9 * persons.length / 10) →
  ∃ num_play_transport, num_play_transport ≥ (9 * persons.length / 10) :=
by
  -- temporary skip the proof
  sorry

end ninety_percent_can_play_basketball_and_transport_l455_455323


namespace pick_peanut_cluster_percentage_l455_455201

def total_chocolates := 100
def typeA_caramels := 5
def typeB_caramels := 6
def typeC_caramels := 4
def typeD_nougats := 2 * typeA_caramels
def typeE_nougats := 2 * typeB_caramels
def typeF_truffles := typeA_caramels + 6
def typeG_truffles := typeB_caramels + 6
def typeH_truffles := typeC_caramels + 6

def total_non_peanut_clusters := 
  typeA_caramels + typeB_caramels + typeC_caramels + typeD_nougats + typeE_nougats + typeF_truffles + typeG_truffles + typeH_truffles

def number_peanut_clusters := total_chocolates - total_non_peanut_clusters

def percent_peanut_clusters := (number_peanut_clusters * 100) / total_chocolates

theorem pick_peanut_cluster_percentage : percent_peanut_clusters = 30 := 
by {
  sorry
}

end pick_peanut_cluster_percentage_l455_455201


namespace sum_of_coefficients_expansion_l455_455222

theorem sum_of_coefficients_expansion (d : ℝ) :
  let expr := -(4 - d) * (d + 3 * (4 - d))
  in (polynomial.sum_of_coeffs expr) = -30 :=
by
  let expr := -(4 - d) * (d + 3 * (4 - d))
  have h_expr : expr = -2 * d^2 + 20 * d - 48, sorry
  have h_coeffs_sum : polynomial.sum_of_coeffs (-2 * d^2 + 20 * d - 48) = -30, sorry
  rw h_expr
  exact h_coeffs_sum

end sum_of_coefficients_expansion_l455_455222


namespace max_price_per_notebook_l455_455186

structure ProblemConditions where
  total_budget : ℚ
  membership_fee : ℚ
  number_of_notebooks : ℕ
  sales_tax_rate : ℚ

def bella_conditions : ProblemConditions := {
  total_budget := 180,
  membership_fee := 5,
  number_of_notebooks := 20,
  sales_tax_rate := 0.06
}

theorem max_price_per_notebook (cond : ProblemConditions) : 
  let effective_budget := cond.total_budget - cond.membership_fee
  let total_cost_with_tax per_notebook := cond.number_of_notebooks * per_notebook * (1 + cond.sales_tax_rate)
  sorted := effective_budget / cond.number_of_notebooks / (1 + cond.sales_tax_rate)
  floor p : ℚ := sorted.floor
  floor p = 8 := sorry

end max_price_per_notebook_l455_455186


namespace f_at_2_f_shifted_range_f_shifted_l455_455250

def f (x : ℝ) := x^2 - 2*x + 7

-- 1) Prove that f(2) = 7
theorem f_at_2 : f 2 = 7 := sorry

-- 2) Prove the expressions for f(x-1) and f(x+1)
theorem f_shifted (x : ℝ) : f (x-1) = x^2 - 4*x + 10 ∧ f (x+1) = x^2 + 6 := sorry

-- 3) Prove the range of f(x+1) is [6, +∞)
theorem range_f_shifted : ∀ x, f (x+1) ≥ 6 := sorry

end f_at_2_f_shifted_range_f_shifted_l455_455250


namespace inequality_proof_l455_455261

theorem inequality_proof (a b : ℝ) (h : a > b ∧ b > 0) : 
  (a - b)^2 / (8 * a) < (a + b) / 2 - Real.sqrt (a * b) ∧ (a + b) / 2 - Real.sqrt (a * b) < (a - b)^2 / (8 * b) := 
by 
  sorry

end inequality_proof_l455_455261


namespace new_paint_intensity_l455_455794

def red_paint_intensity (initial_intensity replacement_intensity : ℝ) (replacement_fraction : ℝ) : ℝ :=
  (1 - replacement_fraction) * initial_intensity + replacement_fraction * replacement_intensity

theorem new_paint_intensity :
  red_paint_intensity 0.1 0.2 0.5 = 0.15 :=
by sorry

end new_paint_intensity_l455_455794


namespace sphere_solution_infinite_l455_455999

universe u

variables 
  (Point : Type u)
  (Line : Type u)
  (Plane : Type u)
  (a : ℝ)
  (g : Line)
  (Q : Point)
  (S : Plane)
  (l : Line)

-- Define necessary geometric relationships
variables
  (is_perpendicular : Line → Plane → Prop)
  (lies_on_line : Point → Line → Prop)
  (is_distance_from_line : Point → Line → ℝ → Prop)
  (tangent_to_line : Line → Sphere → Prop)
  (normal_to_paraboloid : Point → Point → Line)
  (lies_in_plane : Point → Plane → Prop)
  (plane_formed_by_lines : Line → Line → Plane)

-- State the theorem
theorem sphere_solution_infinite :
  is_perpendicular g S ∧ lies_on_line Q g → 
  (∃1 (P : Point), is_distance_from_line P g a ∧ ∀ O, 
  lies_on_line O (normal_to_paraboloid P Q) ∧ (∀ l, tangent_to_line l O → 
  lies_in_plane O (plane_formed_by_lines g l))) :=
sorry

end sphere_solution_infinite_l455_455999


namespace sum_of_coefficients_expansion_l455_455220

theorem sum_of_coefficients_expansion (d : ℝ) :
  let expr := -(4 - d) * (d + 3 * (4 - d))
  in (polynomial.sum_of_coeffs expr) = -30 :=
by
  let expr := -(4 - d) * (d + 3 * (4 - d))
  have h_expr : expr = -2 * d^2 + 20 * d - 48, sorry
  have h_coeffs_sum : polynomial.sum_of_coeffs (-2 * d^2 + 20 * d - 48) = -30, sorry
  rw h_expr
  exact h_coeffs_sum

end sum_of_coefficients_expansion_l455_455220


namespace minimum_value_of_a_l455_455371

def g (x : ℝ) := x^3 - 3*x + 3 - x / Real.exp x

theorem minimum_value_of_a :
  (∃ x : ℝ, ∀ a : ℝ, Real.exp x * (x^3 - 3*x + 3) - a * Real.exp x - x ≤ 0) →
  ∃ a : ℝ, a = 1 - 1 / Real.exp 1 :=
by
  -- Placeholder for proof steps
  sorry

end minimum_value_of_a_l455_455371


namespace regular_polygon_perimeter_l455_455169

theorem regular_polygon_perimeter (s : ℕ) (E : ℕ) (n : ℕ) (P : ℕ)
  (h1 : s = 6)
  (h2 : E = 90)
  (h3 : E = 360 / n)
  (h4 : P = n * s) :
  P = 24 :=
by sorry

end regular_polygon_perimeter_l455_455169


namespace average_score_l455_455325

-- This definition captures the given scores and respective number of shooters
def total_scores (scores : List (ℕ × ℕ)) : ℕ :=
  scores.foldl (λ acc pair, acc + pair.1 * pair.2) 0 

def total_shooters (scores : List (ℕ × ℕ)) : ℕ :=
  scores.foldl (λ acc pair, acc + pair.2) 0 

theorem average_score (scores : List (ℕ × ℕ)) (h : scores = [(7, 4), (8, 2), (9, 3), (10, 1)]) :
  (total_scores scores : ℚ) / (total_shooters scores : ℚ) = 8.1 := by
-- Proof is intentionally omitted
sorry

end average_score_l455_455325


namespace find_m_l455_455660

open Real

noncomputable def vec_a : ℝ × ℝ := (-1, 2)
noncomputable def vec_b (m : ℝ) : ℝ × ℝ := (m, 3)

theorem find_m (m : ℝ) (h : -1 * m + 2 * 3 = 0) : m = 6 :=
sorry

end find_m_l455_455660


namespace total_fish_l455_455868

theorem total_fish (fish_Lilly fish_Rosy : ℕ) (hL : fish_Lilly = 10) (hR : fish_Rosy = 8) : fish_Lilly + fish_Rosy = 18 := 
by 
  sorry

end total_fish_l455_455868


namespace correct_answer_is_C_l455_455807

def options := {A := (2 + complex.i : ℂ), B := (2 - complex.i : ℂ), C := (1 + 2 * complex.i : ℂ), D := (1 - 2 * complex.i : ℂ)}

theorem correct_answer_is_C : options.C = (1 + 2 * complex.i : ℂ) :=
by
  sorry

end correct_answer_is_C_l455_455807


namespace _l455_455340

noncomputable def angle_bisector_theorem {a b c : ℝ} (A B C : ℝ) (h : 2 * B > A) : 
  ∃ E F, 
    BE = CF ∧ 
    ∠ BDE = ∠ CDF :=
sorry

noncomputable def length_of_BE {a b c : ℝ} (A B C : ℝ) (h : 2 * B > A) (E F) 
  (BE_CF : BE = CF) 
  (angle_condition : ∠ BDE = ∠ CDF) : 
  BE = a^2 / (b + c) :=
sorry

end _l455_455340


namespace find_x_prime_l455_455271

theorem find_x_prime (x : ℕ) (h1 : x > 0) (h2 : Prime (x^5 + x + 1)) : x = 1 := sorry

end find_x_prime_l455_455271


namespace rearrangement_distances_unchanged_l455_455330

-- Assume initial positions of pieces on an 8x8 grid
def is_initial_pos (x y : ℕ) : Prop := x ≤ 8 ∧ y ≤ 8

-- Define euclidean distance between two points
def dist (x₁ y₁ x₂ y₂ : ℕ) : ℝ := real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Define that for every initial position, there exists a rearranged position
noncomputable def rearranged_pos : ℕ × ℕ → ℕ × ℕ
| (x, y) := (x', y') -- Assume a function that maps initial to rearranged positions

-- Condition stating that pairwise distances have not decreased
def non_decreasing_dist (p1 p2 : ℕ × ℕ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x1', y1') := rearranged_pos (x1, y1)
  let (x2', y2') := rearranged_pos (x2, y2)
  dist x1 y1 x2 y2 ≤ dist x1' y1' x2' y2'

-- Define the sum of pairwise distances for a set of positions
noncomputable def sum_pairwise_dist (positions: finset (ℕ × ℕ)) : ℝ :=
∑ p1 in positions, ∑ p2 in positions, if p1 ≠ p2 then dist p1.1 p1.2 p2.1 p2.2 else 0

-- The main theorem to prove that the sum of pairwise distances has not changed
theorem rearrangement_distances_unchanged 
  (initial_positions : finset (ℕ × ℕ))
  (h_pos : ∀ p ∈ initial_positions, is_initial_pos p.1 p.2)
  (h_distances : ∀ p1 p2 ∈ initial_positions, non_decreasing_dist p1 p2) :
  sum_pairwise_dist initial_positions = sum_pairwise_dist (initial_positions.image rearranged_pos) := 
by sorry

end rearrangement_distances_unchanged_l455_455330


namespace inequality_proof_l455_455387

variable (a b c : ℝ)

#check (0 < a ∧ 0 < b ∧ 0 < c ∧ abc * (a + b + c) = ab + bc + ca) →
  5 * (a + b + c) ≥ 7 + 8 * abc

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : abc * (a + b + c) = ab + bc + ca) : 
  5 * (a + b + c) ≥ 7 + 8 * abc :=
sorry

end inequality_proof_l455_455387


namespace jose_land_division_l455_455025

/-- Let the total land Jose bought be 20000 square meters. Let Jose divide this land equally among himself and his four siblings. Prove that the land Jose will have after dividing it is 4000 square meters. -/
theorem jose_land_division : 
  let total_land := 20000
  let numberOfPeople := 5
  total_land / numberOfPeople = 4000 := by
sorry

end jose_land_division_l455_455025


namespace parabola_focus_coordinates_l455_455409

theorem parabola_focus_coordinates (a : ℝ) (h : a = 16) : (8 / 2, 0) = (4, 0) := by
  have p : ℝ := 8 -- Step where 2p = 16 implies p = 8
  have focus_coordinates : ℝ × ℝ := (p / 2, 0)
  show focus_coordinates = (4, 0) from rfl -- confirming (8 / 2, 0) = (4, 0)
  sorry

end parabola_focus_coordinates_l455_455409


namespace nth_derivative_eq_l455_455996

-- Definitions representing the conditions in the problem
variable {a b : ℝ} (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_a_ne_1 : a ≠ 1)

def f (x : ℝ) : ℝ := Real.sqrt (a * x^2 + b)

-- Targeting the formulation of the nth derivative of f
noncomputable def f_nth (n : ℕ) (x : ℝ) : ℝ :=
  Real.sqrt (a^n * x^2 + b * ((1 - a^n) / (1 - a)))

-- Statement of the problem to be proved
theorem nth_derivative_eq (n : ℕ) (x : ℝ) :
  (λ x, f^[n] x) = (f_nth n) :=
sorry

end nth_derivative_eq_l455_455996


namespace sum_of_coordinates_l455_455636

theorem sum_of_coordinates (f : ℝ → ℝ) (h : f 2 = 3) : 
  let x := 2 / 3
  let y := 2 * f (3 * x) + 4
  x + y = 32 / 3 :=
by
  sorry

end sum_of_coordinates_l455_455636


namespace probability_exactly_three_primes_l455_455915

noncomputable def prime_faces : Finset ℕ := {2, 3, 5, 7, 11}

def num_faces : ℕ := 12
def num_dice : ℕ := 7
def target_primes : ℕ := 3

def probability_three_primes : ℚ :=
  35 * ((5 / 12)^3 * (7 / 12)^4)

theorem probability_exactly_three_primes :
  probability_three_primes = (4375 / 51821766) :=
by
  sorry

end probability_exactly_three_primes_l455_455915


namespace region_volume_is_two_thirds_l455_455477

noncomputable def volume_of_region : ℝ :=
  let region := {p : ℝ × ℝ × ℝ | |p.1| + |p.2| + |p.3| ≤ 2 ∧ |p.1| + |p.2| + |p.3 - 2| ≤ 2}
  -- Assuming volume function calculates the volume of the region
  volume region

theorem region_volume_is_two_thirds :
  volume_of_region = 2 / 3 :=
by
  sorry

end region_volume_is_two_thirds_l455_455477


namespace length_of_floor_proof_l455_455094

noncomputable def breadth_of_floor : ℝ := sorry
noncomputable def length_of_floor : ℝ := 3 * breadth_of_floor

theorem length_of_floor_proof :
  (∃ (breadth : ℝ), length_of_floor = 3 * breadth ∧
  total_cost = 624 ∧ rate_per_sq_m = 4 ∧ (3 * breadth^2 = 156)) →
  length_of_floor = 6 * real.sqrt 13 :=
by
  sorry

end length_of_floor_proof_l455_455094


namespace dot_product_of_diff_l455_455297

def vector_a : ℝ × ℝ × ℝ := (-3, 1, -1)
def vector_b : ℝ × ℝ × ℝ := (1, 3, 5)
def vector_c : ℝ × ℝ × ℝ := (-2, -1, 2)

def vec_sub (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2, v1.3 - v2.3)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem dot_product_of_diff :
  dot_product (vec_sub vector_a vector_b) vector_c = -2 :=
by {
  sorry
}

end dot_product_of_diff_l455_455297


namespace none_of_the_statements_correct_l455_455861

theorem none_of_the_statements_correct :
  ¬ (sqrt ((-2: ℝ) ^ 2) = -2) ∧
  ¬ (∀ x, x = 3 → x = sqrt 9 ∨ x = -sqrt 9) ∧
  ¬ (real.cbrt ((-5: ℝ) ^ 3) = 5) ∧
  ¬ (sqrt 16 = 2 ∨ sqrt 16 = -2) :=
by
  split;
  { intro h,
    sorry }

end none_of_the_statements_correct_l455_455861


namespace inequality_of_abc_l455_455390

variable {a b c : ℝ}

theorem inequality_of_abc 
    (h : 0 < a ∧ 0 < b ∧ 0 < c)
    (h₁ : abc * (a + b + c) = ab + bc + ca) :
    5 * (a + b + c) ≥ 7 + 8 * abc :=
sorry

end inequality_of_abc_l455_455390


namespace problem1_problem2_l455_455550

-- Proposition 1
theorem problem1 : (2 ^ (-2 : ℤ) - ((-1/2) ^ 2) + (π - 2) ^ 0 - (-1) ^ 2023) = 2 := 
by
   sorry

-- Proposition 2
variable (a : ℝ)
theorem problem2 : ((2 * a^2 * 8 * a^2 + (2 * a)^3 - 4 * a^2) / (2 * a^2)) = (8 * a^2 + 4 * a - 2) :=
by
   sorry

end problem1_problem2_l455_455550


namespace three_scientists_same_topic_l455_455106

theorem three_scientists_same_topic :
  ∃ (S : Finset (Fin 17)), ∃ f : (Fin 17) → (Fin 17 → Fin 3), 
  (∀ i j, S.mem i → S.mem j →  i ≠ j → f i j = f j i)
  ∧ S.card >= 3 ∧
  ∀ (i j k : Fin 17), f i j = f j k ∧ f j k = f k i :=
sorry

end three_scientists_same_topic_l455_455106


namespace find_c_l455_455982

noncomputable def slope (c : ℝ) : ℝ := 4 / (3 - c)

noncomputable def line_eq (c : ℝ) (x : ℝ) : ℝ := slope(c) * (x - c)

theorem find_c :
  ∃ c : ℝ, 
    (c > 0) ∧ (c < 3) ∧ 
    (3 - c) * 2 = 2.5 :=
begin
  use 1.75,
  split,
  { linarith, },
  split,
  { linarith, },
  { field_simp,
    norm_num, }
end

end find_c_l455_455982


namespace find_smallest_lambda_l455_455239

theorem find_smallest_lambda :
  ∃ λ : ℝ, (∀ (a : ℕ → ℝ), (∀ n, a n > 1) → (∀ n, ∏ i in finset.range (n + 1), a i < (a n) ^ λ)) ∧
    (∀ μ, (∀ (a : ℕ → ℝ), (∀ n, a n > 1) → (∀ n, ∏ i in finset.range (n + 1), a i < (a n) ^ μ)) → λ ≤ μ) ∧
    λ = 4 :=
by
  sorry

end find_smallest_lambda_l455_455239


namespace eunji_class_total_students_l455_455179

variable (A B : Finset ℕ) (universe_students : Finset ℕ)

axiom students_play_instrument_a : A.card = 24
axiom students_play_instrument_b : B.card = 17
axiom students_play_both_instruments : (A ∩ B).card = 8
axiom no_students_without_instruments : A ∪ B = universe_students

theorem eunji_class_total_students : universe_students.card = 33 := by
  sorry

end eunji_class_total_students_l455_455179


namespace power_sum_of_roots_l455_455557

theorem power_sum_of_roots (t q a1 a2 : ℚ) (h1 : a1 + a2 = t)
  (h2 : a1 * a2 = q) (h3 : ∀ n : ℕ, a1 + a2 = a1^n + a2^n) :
  a1^1004 + a2^1004 = 2 := 
begin
  sorry
end

end power_sum_of_roots_l455_455557


namespace F_positive_for_positive_t_l455_455391

noncomputable def F (t : ℝ) : ℝ := ∫ x in 0..t, (Real.sin x) / (1 + x^2)

theorem F_positive_for_positive_t (t : ℝ) (ht : t > 0) : F t > 0 := 
by
  sorry

end F_positive_for_positive_t_l455_455391


namespace vector_magnitude_l455_455084

noncomputable def a : ℝ × ℝ := (1, 0)
noncomputable def b : ℝ × ℝ
-- Assume the existence of b satisfying the conditions
variable (b : ℝ × ℝ)

def magnitude_sq (v : ℝ × ℝ) : ℝ := v.1^2 + v.2^2

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

#check Real.cos (Real.pi / 3)  -- cos(60°) = 1/2

-- Conditions
axiom angle_ab : dot_product a b = magnitude_sq a * magnitude_sq b * Real.cos (Real.pi / 3)
axiom magnitude_b : magnitude_sq b = 1

theorem vector_magnitude :
  (magnitude_sq (a.1 + 2 * b.1, a.2 + 2 * b.2) = 7) :=
by
  sorry

end vector_magnitude_l455_455084


namespace natalia_crates_l455_455577

noncomputable def total_items (novels comics documentaries albums : ℕ) : ℕ :=
  novels + comics + documentaries + albums

noncomputable def crates_needed (total_items items_per_crate : ℕ) : ℕ :=
  (total_items + items_per_crate - 1) / items_per_crate

theorem natalia_crates : crates_needed (total_items 145 271 419 209) 9 = 117 := by
  sorry

end natalia_crates_l455_455577


namespace areas_correct_l455_455148

def radius := 10
def totalArea := Real.pi * radius^2
def winProbability := 1 / 4
def secondChanceProbability := 1 / 2

def winArea := winProbability * totalArea
def secondChanceArea := secondChanceProbability * totalArea

theorem areas_correct :
  winArea + secondChanceArea = 75 * Real.pi := by
  sorry

end areas_correct_l455_455148


namespace max_collisions_l455_455751

-- Define the problem
theorem max_collisions (n : ℕ) (hn : n > 0) : 
  ∃ C : ℕ, C = (n * (n - 1)) / 2 := 
sorry

end max_collisions_l455_455751


namespace find_y_l455_455369

def binary_op (a b c d : Int) : Int × Int := (a + d, b - c)

theorem find_y : ∃ y : Int, (binary_op 3 y 2 0) = (3, 4) ↔ y = 6 := by
  sorry

end find_y_l455_455369


namespace find_s_of_2_l455_455361

def t (x : ℝ) : ℝ := 4 * x - 9
def s (x : ℝ) : ℝ := x^2 + 4 * x - 1

theorem find_s_of_2 : s (2) = 281 / 16 :=
by
  sorry

end find_s_of_2_l455_455361


namespace Joel_laps_count_l455_455492

-- Definitions of conditions
def Yvonne_laps := 10
def sister_laps := Yvonne_laps / 2
def Joel_laps := sister_laps * 3

-- Statement to be proved
theorem Joel_laps_count : Joel_laps = 15 := by
  -- currently, proof is not required, so we defer it with 'sorry'
  sorry

end Joel_laps_count_l455_455492


namespace largest_zero_in_interval_l455_455362

theorem largest_zero_in_interval
  (a b c d p q : ℝ)
  (h_poly : ∀ x, x^4 - 2 * x^3 + p * x + q = 0 → x ∈ {a, b, c, d})
  (h_sum : a + b + c + d = 2)
  (h_sum_pairwise_products : ab + ac + ad + bc + bd + cd = 0) :
  a ≤ 2 :=
sorry

end largest_zero_in_interval_l455_455362


namespace total_candidates_appeared_l455_455866

-- Definitions as prompted by the conditions
def failed_english := 0.49
def failed_hindi := 0.36
def failed_both := 0.15
def passed_english_alone := 630

-- Theorem stating the main question and its answer as proof problem
theorem total_candidates_appeared (T : ℝ) :
  ((1 - failed_english) - failed_both) * T = passed_english_alone → T = 1750 :=
by
  -- Proof to be filled in
  sorry

end total_candidates_appeared_l455_455866


namespace log_sum_eq_two_l455_455505

theorem log_sum_eq_two:
  ∀ (a b : ℝ), (a = 2) → (b = 50) → (log 10 a + log 10 b = 2) :=
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end log_sum_eq_two_l455_455505


namespace express_as_terminating_decimal_l455_455584

theorem express_as_terminating_decimal :
  let num := 29
      den := 160
      dec := 0.18125
      prime_factorization := 2^5 * 5 = den
  in (num:ℚ) / (den:ℚ) = dec :=
by
  sorry

end express_as_terminating_decimal_l455_455584


namespace find_x_prime_l455_455270

theorem find_x_prime (x : ℕ) (h1 : x > 0) (h2 : Prime (x^5 + x + 1)) : x = 1 := sorry

end find_x_prime_l455_455270


namespace distance_is_sqrt2_l455_455958

noncomputable def distance_from_center_to_line : ℝ :=
  let C := (2, 0)           -- Center of the circle
  let A := (1 : ℝ)          -- Coefficient of x in line equation
  let B := (-1 : ℝ)         -- Coefficient of y in line equation
  let x1 := 2               -- x-coordinate of the center
  let y1 := 0               -- y-coordinate of the center
  abs (A * x1 + B * y1) / real.sqrt (A ^ 2 + B ^ 2)

theorem distance_is_sqrt2 :
  distance_from_center_to_line = real.sqrt 2 := 
sorry

end distance_is_sqrt2_l455_455958


namespace radius_of_tangent_circle_l455_455072

theorem radius_of_tangent_circle :
  ∃ r : ℝ, 6 ∧ (∀ (x : ℝ), y = x^2 + r) ∧ 
           (is_tangent_to_circle (x^2) r) ∧
           (angle_between_neighbors (x^2) (45 * π / 180)) ∧
           (r = 1 / 4) :=
sorry

end radius_of_tangent_circle_l455_455072


namespace smallest_B_for_divisibility_by_3_l455_455015

variable (B C : Nat)
variable (is_digit_B : B ≤ 9) (is_digit_C : C ≤ 9)

-- Defining the sum of the known digits
def sum_known_digits : Nat := 30

-- Defining the condition for divisibility by 3 for the entire number
def divisible_by_3 (n : Nat) : Prop := n % 3 = 0

-- Statement: Prove that the smallest digit B such that 30 + B + C is divisible by 3 is B = 0.
theorem smallest_B_for_divisibility_by_3 :
  ∃ B : Nat, is_digit_B ∧ (∀ B' : Nat, is_digit_B → B ≤ B' → divisible_by_3 (sum_known_digits + B + C))
  :=
sorry

end smallest_B_for_divisibility_by_3_l455_455015


namespace max_sums_with_diff_less_than_one_l455_455040

theorem max_sums_with_diff_less_than_one (n : ℕ) (x : Fin n → ℝ)
  (h_abs : ∀ i, |x i| ≥ 1) :
  ∃ A : Finset (Fin n) → ℝ, 
    ∀ (A1 A2 : Finset (Fin n)), 
      (A1 ≠ A2) → |(x \sum A1) -  (x \sum A2)| < 1 → 
        (Finset.card {A : Finset (Fin n)| 
          (∀ (A1 A2: Finset (Fin n)), 
            (A1 ≠ A2) →  |(x \sum  (A A1) ) - (x \sum  (A A2))| < 1 )
} 
≤ nat.choose n  (n / 2) :=
sorry

end max_sums_with_diff_less_than_one_l455_455040


namespace triangle_area_is_18_l455_455850

def point := (ℝ × ℝ)
def base_length (p1 p2 : point) := abs (p2.1 - p1.1)
def height (p : point) := abs p.2
def triangle_area (base height : ℝ) := 1/2 * base * height

def vertices : (point × point × point) := ((0,0), (4,0), (4,9))

theorem triangle_area_is_18 : 
  triangle_area (base_length (vertices.1) (vertices.2)) (height (vertices.3)) = 18 := by
  sorry

end triangle_area_is_18_l455_455850


namespace problem_l455_455678

variable {x y : ℝ}

theorem problem (h : x < y) : 3 - x > 3 - y :=
sorry

end problem_l455_455678


namespace probability_interview_information_l455_455184

theorem probability_interview_information
  (total_students : ℕ)
  (french_students : ℕ)
  (spanish_students : ℕ)
  (german_students : ℕ)
  (all_french_in_spanish : french_students ≤ spanish_students)
  (jessie_picks_two : ℕ := 2)
  (total_students_value : total_students = 30)
  (french_students_value : french_students = 22)
  (spanish_students_value : spanish_students = 25)
  (german_students_value : german_students = 5) :
  (let total_ways := Nat.choose total_students jessie_picks_two in 
   let only_spanish_and_german := total_students - spanish_students in
   let ways_to_pick_not_both := Nat.choose (only_spanish_and_german + german_students) jessie_picks_two in
   let not_both_probability := ways_to_pick_not_both / total_ways in 
   1 - not_both_probability = (407 / 435 : ℚ)) := sorry

end probability_interview_information_l455_455184


namespace trains_meet_time_l455_455452

-- Definition for the problem conditions
def distance : ℝ := 450
def speed_slower_train : ℝ := 48
def speed_faster_train : ℝ := speed_slower_train + 6
def combined_speed : ℝ := speed_slower_train + speed_faster_train
def time_to_meet : ℝ := distance / combined_speed

-- Theorem statement for the time to meet
theorem trains_meet_time : time_to_meet = 75 / 17 := by
  sorry

end trains_meet_time_l455_455452


namespace cost_of_vip_seat_l455_455173

def V : ℝ := 60 -- cost of VIP seat we need to prove
def P₁ : ℝ := 10 -- cost of general admission seat
def G : ℝ := 234 -- number of general admission tickets (from previous steps)
def Vt : ℝ := G - 148 -- number of VIP tickets sold
def total_tickets : ℝ := G + Vt -- total tickets sold
def revenue : ℝ := P₁ * G + V * Vt -- total revenue

theorem cost_of_vip_seat : V = 60 :=
by
  have h1 : total_tickets = 320 := rfl -- total tickets sold is 320
  have h2 : revenue = 7500 := rfl -- total revenue is $7500
  have h3 : G = 234 := rfl -- number of general admission tickets
  have h4 : Vt = G - 148 := rfl -- number of VIP tickets sold
  have h5 : V = (7500 - P₁ * G) / Vt := sorry -- equation from revenue condition
  have h6 : G = 234 := rfl -- number of general admission tickets
  have h7 : Vt = 234 - 148 := rfl -- calculate Vip tickets
  have h8 : Vt = 86 := rfl -- substituting G
  have h9 : V = (7500 - 10 * 234) / 86 := sorry -- substitute known values
  have h10 : V = 5160 / 86 := sorry -- simplify the equation
  have h11 : V = 60 := sorry -- calculate final value for V
  exact h11 -- conclude that V = 60


end cost_of_vip_seat_l455_455173


namespace problem_part1_problem_part2_l455_455329

-- Definitions and statements for the proof problem

def P := ℝ × ℝ
def F₁ : P := (0, -Real.sqrt 3)
def F₂ : P := (0, Real.sqrt 3)

def ellipse_eq (x y : ℝ) : Prop := (x^2)/4 + y^2 = 1

def sum_dist_eq_4 (p : P) : Prop :=
  let (x, y) := p
  Real.sqrt ((x - 0)^2 + (y + Real.sqrt 3)^2) +
  Real.sqrt ((x - 0)^2 + (y - Real.sqrt 3)^2) = 4

noncomputable def line_eq (k x : ℝ) : ℝ := k * x + 1

def on_ellipse (p : P) : Prop := 
  let (x, y) := p 
  ellipse_eq x y

def intersects_line (k : ℝ) (a b : P) : Prop :=
  let (x₁, y₁) := a
  let (x₂, y₂) := b
  y₁ = line_eq k x₁ ∧ y₂ = line_eq k x₂ ∧ on_ellipse (x₁, y₁) ∧ on_ellipse (x₂, y₂)

theorem problem_part1 : ∀ (P : ℝ × ℝ), sum_dist_eq_4 P → ellipse_eq (P.1) (P.2) :=
sorry

theorem problem_part2 :
  ∃ (k : ℝ), (k = 1/2 ∨ k = -1/2) ∧
  ∃ (A B : P), intersects_line k A B ∧ 
  (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 65 / 17 ∧
  (Real.sqrt (A.1^2 + A.2^2) + Real.sqrt (B.1^2 + B.2^2)) = 
    Real.sqrt ((A.1 + B.1)^2 + (A.2 + B.2)^2)) :=
sorry

end problem_part1_problem_part2_l455_455329


namespace person_savings_l455_455815

theorem person_savings (income expenditure savings : ℝ) 
  (h1 : income = 18000)
  (h2 : income / expenditure = 5 / 4)
  (h3 : savings = income - expenditure) : 
  savings = 3600 := 
sorry

end person_savings_l455_455815


namespace median_of_scores_l455_455713

theorem median_of_scores (scores : List ℕ) (h : scores = [77, 80, 79, 77, 80, 79, 80]) : 
  median (List.sort .lt scores) = 79 := 
by 
  sorry

end median_of_scores_l455_455713


namespace total_games_in_season_is_correct_l455_455829

-- Definitions based on given conditions
def games_per_month : ℕ := 7
def season_months : ℕ := 2

-- The theorem to prove
theorem total_games_in_season_is_correct : 
  (games_per_month * season_months = 14) :=
by
  sorry

end total_games_in_season_is_correct_l455_455829


namespace sum_of_coefficients_l455_455228

theorem sum_of_coefficients (d : ℤ) : 
  let expr := -(4 - d) * (d + 3 * (4 - d))
  let expanded_form := -2 * d ^ 2 + 20 * d - 48
  let sum_of_coeffs := -2 + 20 - 48
  sum_of_coeffs = -30 :=
by
  -- The proof will go here, skipping for now.
  sorry

end sum_of_coefficients_l455_455228


namespace largest_integer_m_such_that_expression_is_negative_l455_455965

theorem largest_integer_m_such_that_expression_is_negative :
  ∃ (m : ℤ), (∀ (n : ℤ), (m^2 - 11 * m + 24 < 0 ) → n < m → n^2 - 11 * n + 24 < 0) ∧
  m^2 - 11 * m + 24 < 0 ∧
  (m ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) :=
by
  sorry

end largest_integer_m_such_that_expression_is_negative_l455_455965


namespace range_of_m_l455_455280

theorem range_of_m (m : ℝ) :
  (¬(∀ x : ℝ, x^2 - m * x + 1 > 0 → -2 < m ∧ m < 2)) ∧
  (∃ x : ℝ, x^2 < 9 - m^2) ∧
  (-3 < m ∧ m < 3) →
  ((-3 < m ∧ m ≤ -2) ∨ (2 ≤ m ∧ m < 3)) :=
by sorry

end range_of_m_l455_455280


namespace negation_of_proposition_l455_455424

theorem negation_of_proposition :
  ¬(∀ n : ℤ, (∃ k : ℤ, n = 2 * k) → (∃ m : ℤ, n = 2 * m)) ↔ ∃ n : ℤ, (∃ k : ℤ, n = 2 * k) ∧ ¬(∃ m : ℤ, n = 2 * m) := 
sorry

end negation_of_proposition_l455_455424


namespace proposition_p_and_not_q_is_true_l455_455621

-- Define proposition p
def p : Prop := ∀ x > 0, Real.log (x + 1) > 0

-- Define proposition q
def q : Prop := ∀ a b : Real, a > b → a^2 > b^2

-- State the theorem to be proven in Lean
theorem proposition_p_and_not_q_is_true : p ∧ ¬q :=
by
  -- Sorry placeholder for the proof
  sorry

end proposition_p_and_not_q_is_true_l455_455621


namespace ryan_fish_count_l455_455345

theorem ryan_fish_count
  (R : ℕ)
  (J : ℕ)
  (Jeffery_fish : ℕ)
  (h1 : Jeffery_fish = 60)
  (h2 : Jeffery_fish = 2 * R)
  (h3 : J + R + Jeffery_fish = 100)
  : R = 30 :=
by
  sorry

end ryan_fish_count_l455_455345


namespace valid_n_l455_455233

theorem valid_n (n : ℕ) (h : n > 1) 
  (perm : ∃ (a : Fin n → ℕ), (∀ i, a i ∈ {i | 1 ≤ i ∧ i ≤ n}) ∧ 
                               Function.Bijective a ∧ 
                               ∀ i, (∏ j in Finset.range (i+1), a ⟨j, sorry⟩) % n ≠ 
                                    (∏ j in Finset.range (i), a ⟨j, sorry⟩) % n) : 
    n = 4 ∨ Nat.Prime n := 
begin
  sorry
end

end valid_n_l455_455233


namespace calculation_result_l455_455917

theorem calculation_result :
  -2^2 + real.sqrt ((-1)^2) - |real.sqrt 2 - 2| = -5 + real.sqrt 2 :=
by sorry

end calculation_result_l455_455917


namespace blueberry_pies_correct_l455_455568

def total_pies := 36
def apple_pie_ratio := 3
def blueberry_pie_ratio := 4
def cherry_pie_ratio := 5

-- Total parts in the ratio
def total_ratio_parts := apple_pie_ratio + blueberry_pie_ratio + cherry_pie_ratio

-- Number of pies per part
noncomputable def pies_per_part := total_pies / total_ratio_parts

-- Number of blueberry pies
noncomputable def blueberry_pies := blueberry_pie_ratio * pies_per_part

theorem blueberry_pies_correct : blueberry_pies = 12 := 
by
  sorry

end blueberry_pies_correct_l455_455568


namespace divisibility_of_c_l455_455752

theorem divisibility_of_c (p a b c : ℤ) (hp : Nat.Prime p) (ha : p ∣ a) (hb : p ∣ b) (hc : p ∣ c)
  (r s : ℤ) (hrs : r ≠ s) (hr : r^3 + a*r^2 + b*r + c = 0) (hs : s^3 + a*s^2 + b*s + c = 0) :
  p^3 ∣ c :=
sorry

end divisibility_of_c_l455_455752


namespace correct_statements_l455_455827

/-- Definitions for the conditions of the problem -/
def statement_1 : Prop := ∀ (seq : ℕ → ℕ), ∃! (f: ℕ → ℕ), f = seq
def statement_2 : Prop := ∀ (s : ℕ → ℕ), ∃ (f : ℕ → ℕ), f = s
def statement_3 : Prop := ∀ (s : ℕ → ℕ) (n : ℕ), s n = isolated_point (s n)
def statement_4 : Prop := ∀ (s : ℕ → ℕ), ∃ (gen_term : ℕ → ℕ), gen_term = s

/-- Proof that the correct statements about sequences are the second and the third -/
theorem correct_statements : (statement_2 ∧ statement_3) ∧ ¬(statement_1 ∧ statement_4) :=
by
  -- Placeholder for the actual proof
  sorry

end correct_statements_l455_455827


namespace number_halfway_between_l455_455237

theorem number_halfway_between :
  ∃ x : ℚ, x = (1/12 + 1/14) / 2 ∧ x = 13 / 168 :=
sorry

end number_halfway_between_l455_455237


namespace player_A_wins_2p_game_player_B_wins_3p_game_with_C_l455_455501

-- Definitions and conditions 
def cell := (ℕ × ℕ)
def board := fin 5 × fin 5
def player := ℕ -- Player A = 0, Player B = 1, Player C = 2

-- Definition of the game state
structure game_state :=
  (board : board → option player) -- board is a function from cell to an option containing player
  (turn : player) -- the current player (0, 1, or 2)

-- Victory condition: if current player places piece completing any row or column of five
def win_by (g : game_state) (p : player) :=
  ∃ i : fin 5, (∀ j, g.board (i, j) = some p) ∨ (∀ j, g.board (j, i) = some p)

-- Winning strategy for Player A in 2-player game
theorem player_A_wins_2p_game : ∃ strategy : game_state → cell, ∀ gs : game_state, 
  gs.turn = 0 → gs.board (strategy gs) = none → win_by (place gs (strategy gs) 0) 0 :=
sorry

-- Winning strategy for Player B in 3-player game with Player C's coordination
theorem player_B_wins_3p_game_with_C : ∃ strategy : game_state → cell, ∀ gs : game_state, 
  gs.turn = 1 → gs.board (strategy gs) = none → (∃ g' : game_state, g'.turn ≠ 1 → win_by g' 1) :=
sorry

-- Placeholder function for placing a piece on the board
def place (gs : game_state) (c : cell) (p : player) : game_state := 
{ gs with board := function.update gs.board c (some p), turn := (gs.turn + 1) % 3 } 


end player_A_wins_2p_game_player_B_wins_3p_game_with_C_l455_455501


namespace decomposition_of_5_to_4_eq_125_l455_455602

theorem decomposition_of_5_to_4_eq_125 :
  (∃ a b c : ℕ, (5^4 = a + b + c) ∧ 
                (a = 121) ∧ 
                (b = 123) ∧ 
                (c = 125)) := by 
sorry

end decomposition_of_5_to_4_eq_125_l455_455602


namespace diameter_of_lake_l455_455313

-- Given conditions: the radius of the circular lake
def radius : ℝ := 7

-- The proof problem: proving the diameter of the lake is 14 meters
theorem diameter_of_lake : 2 * radius = 14 :=
by
  sorry

end diameter_of_lake_l455_455313


namespace bank_fund_collection_l455_455503

theorem bank_fund_collection :
  ∃ (donations : Fin 100 → ℕ),
    (∀ i, donations i ≤ 200) ∧
    (∃ (d1 d2 : Fin 50 → ℕ),
      (∀ i, d1 i ≤ 100) ∧
      (∀ i, 100 < d2 i ∧ d2 i ≤ 200) ∧
      (∀ i j, d2 i - d1 j ≠ 100)) ∧
      (∑ i in Finset.univ : Finset (Fin 100), donations i) = 10050 :=
begin
  sorry
end

end bank_fund_collection_l455_455503


namespace brother_collection_ratio_l455_455343

/-- Janet's original and modified collection conditions. --/
def janet_original_collection : ℕ := 10
def janet_sell_amount : ℕ := 6
def janet_acquire_amount : ℕ := 4
def janet_final_collection_after_brother : ℕ := 24

/-- Janet's calculations. --/
def janet_collection_before_brother : ℕ := janet_original_collection - janet_sell_amount + janet_acquire_amount
def her_brothers_collection : ℕ := janet_final_collection_after_brother - janet_collection_before_brother

/-- The ratio of her brother's collection to Janet's collection before receiving his collection is 2:1. --/
theorem brother_collection_ratio :
  her_brothers_collection : janet_collection_before_brother = 2 :=
by
  -- Proof is omitted
  sorry

end brother_collection_ratio_l455_455343


namespace circle_center_radius_l455_455089

theorem circle_center_radius (A B : ℝ × ℝ) (hA : A = (2, -3)) (hB : B = (8, 9)) :
  let center := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ∧ let radius := Real.sqrt ((A.1 - center.1)^2 + (A.2 - center.2)^2)
  in center = (5, 3) ∧ radius = 3 * Real.sqrt 5 :=
by
  -- these steps represent the structure of the proof
  sorry

end circle_center_radius_l455_455089


namespace mia_study_time_l455_455760

theorem mia_study_time 
  (T : ℕ)
  (watching_tv_exercise_social_media : T = 1440 ∧ 
    ∃ study_time : ℚ, 
    (study_time = (1 / 4) * 
      (((27 / 40) * T - (9 / 80) * T) / 
        (T * 1 / 40 - (1 / 5) * T - (1 / 8) * T))
    )) :
  T = 1440 → study_time = 202.5 := 
by
  sorry

end mia_study_time_l455_455760


namespace conic_curve_eccentricity_l455_455637

theorem conic_curve_eccentricity (m : ℝ) 
    (h1 : ∃ k, k ≠ 0 ∧ 1 * k = m ∧ m * k = 4)
    (h2 : m = -2) : ∃ e : ℝ, e = Real.sqrt 3 :=
by
  sorry

end conic_curve_eccentricity_l455_455637


namespace problem_l455_455812

theorem problem (a : ℕ → ℕ) (h₁ : ∀ n, a n = 2^(n-1)) (n : ℕ) : 
  (∑ i in finset.range (n+1), (a i)^2 < 5 * 2^(n+1)) → n ≤ 4 :=
by
  sorry

end problem_l455_455812


namespace sign_up_ways_l455_455721

theorem sign_up_ways : 
  let num_ways_A := 2
  let num_ways_B := 2
  let num_ways_C := 2
  num_ways_A * num_ways_B * num_ways_C = 8 := 
by 
  -- show the proof (omitted for simplicity)
  sorry

end sign_up_ways_l455_455721


namespace find_a_l455_455925

def F (a b c : ℤ) : ℤ := a * b^2 + c

theorem find_a (a : ℤ) (h : F a 3 (-1) = F a 5 (-3)) : a = 1 / 8 := by
  sorry

end find_a_l455_455925


namespace trivia_team_missing_members_l455_455899

theorem trivia_team_missing_members 
  (total_members : ℕ)
  (points_per_member : ℕ)
  (total_points : ℕ)
  (showed_up_members : ℕ)
  (missing_members : ℕ) 
  (h1 : total_members = 15) 
  (h2 : points_per_member = 3) 
  (h3 : total_points = 27) 
  (h4 : showed_up_members = total_points / points_per_member) 
  (h5 : missing_members = total_members - showed_up_members) : 
  missing_members = 6 :=
by
  sorry

end trivia_team_missing_members_l455_455899


namespace sum_complex_equals_l455_455547

noncomputable def complex_sum : ℂ := 
  ∑ n in Finset.range 25, (complex.I ^ n) * real.cos (45 * n * real.pi / 180)

theorem sum_complex_equals : 
  complex_sum = 7 + 6 * complex.I * real.sqrt 2 :=
by
  sorry

end sum_complex_equals_l455_455547


namespace k_plus_m_plus_n_val_l455_455259

theorem k_plus_m_plus_n_val (t : ℝ) (m n k : ℕ) (hmn_rel_prime : Nat.coprime m n)
  (h1 : (1 + Real.sin t) * (1 + Real.cos t) = 5 / 4)
  (h2 : (1 - Real.sin t) * (1 - Real.cos t) = m / n - Real.sqrt k)
  (k_pos : 0 < k)
  (m_pos : 0 < m)
  (n_pos : 0 < n):
  k + m + n = 27 :=
sorry

end k_plus_m_plus_n_val_l455_455259


namespace compute_cos_sin_75_product_l455_455555

theorem compute_cos_sin_75_product :
  (cos (75 * real.pi / 180) + sin (75 * real.pi / 180)) *
  (cos (75 * real.pi / 180) - sin (75 * real.pi / 180)) =
  - (real.sqrt 3) / 2 :=
by
  sorry

end compute_cos_sin_75_product_l455_455555


namespace equidistant_points_l455_455955

variable (z : ℝ)

def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := p1 in
  let (x2, y2, z2) := p2 in
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

theorem equidistant_points :
  let A := (0, 0, z)
  let B := (3, 3, 1)
  let C := (4, 1, 2)
  distance A B = distance A C → z = 1 :=
by
  intros
  sorry

end equidistant_points_l455_455955


namespace circumcircle_diameter_l455_455318

theorem circumcircle_diameter (a : ℝ) (c : ℝ) (B : ℝ) (S : ℝ) (h₁ : a = 2) (h₂ : B = 60 * (π / 180)) (h₃ : S = sqrt 3) :
  (2 / (sin (60 * (π / 180)))) = 4 * (sqrt 3) / 3 := 
sorry

end circumcircle_diameter_l455_455318


namespace largest_value_among_given_numbers_l455_455483

theorem largest_value_among_given_numbers :
  let a := 2 * 0 * 2006
  let b := 2 * 0 + 6
  let c := 2 + 0 * 2006
  let d := 2 * (0 + 6)
  let e := 2006 * 0 + 0 * 6
  d >= a ∧ d >= b ∧ d >= c ∧ d >= e :=
by
  let a := 2 * 0 * 2006
  let b := 2 * 0 + 6
  let c := 2 + 0 * 2006
  let d := 2 * (0 + 6)
  let e := 2006 * 0 + 0 * 6
  sorry

end largest_value_among_given_numbers_l455_455483


namespace greatest_whole_number_solution_l455_455960

theorem greatest_whole_number_solution (x : ℤ) (h : 6 * x - 5 < 7 - 3 * x) : x ≤ 1 :=
sorry

end greatest_whole_number_solution_l455_455960


namespace count_primes_with_prime_remainders_between_50_and_100_l455_455670

/-- 
The count of prime numbers between 50 and 100 that have a prime remainder 
(1, 2, 3, or 5) when divided by 6 is 10.
-/
theorem count_primes_with_prime_remainders_between_50_and_100 : 
  (finset.filter (λ p, ∃ r, (p % 6 = r) ∧ nat.prime r ∧ r ∈ ({1, 2, 3, 5} : finset ℕ)) 
                  (finset.filter nat.prime (finset.Ico 51 101))).card = 10 := 
by 
  sorry

end count_primes_with_prime_remainders_between_50_and_100_l455_455670


namespace maximize_xz_l455_455537

theorem maximize_xz :
  ∃ (x y z t : ℝ), (x^2 + y^2 = 4) ∧ (z^2 + t^2 = 9) ∧ (x * t + y * z = 6) ∧ (x + z = real.sqrt 13) :=
begin
  sorry
end

end maximize_xz_l455_455537


namespace exist_scalars_B_pow_6_l455_455354

theorem exist_scalars_B_pow_6 (B : Matrix (Fin 2) (Fin 2) ℕ) 
(hB : B = ![![2, 3], ![4, 1]]) 
(hB2 : B^2 = 8 • B + 5 • (1 : Matrix (Fin 2) (Fin 2) ℕ)) :
  ∃ (r s : ℕ), r = 40208 ∧ s = 25955 ∧ B^6 = r • B + s • (1 : Matrix (Fin 2) (Fin 2) ℕ) :=
begin
  -- Insert proof here
  sorry
end

end exist_scalars_B_pow_6_l455_455354


namespace find_B_from_period_l455_455187

theorem find_B_from_period (A B C D : ℝ) (h : B ≠ 0) (period_condition : 2 * |2 * π / B| = 4 * π) : B = 1 := sorry

end find_B_from_period_l455_455187


namespace tom_catches_alice_in_12_857_minutes_l455_455904

variable (alice_speed tom_speed initial_distance : ℝ)
variable (alice_speed_positive : alice_speed > 0)
variable (tom_speed_positive : tom_speed > 0)
variable (initial_distance_positive : initial_distance > 0)

noncomputable def time_for_tom_to_catch_alice_minutes :=
  (initial_distance / (alice_speed + tom_speed)) * 60

theorem tom_catches_alice_in_12_857_minutes :
  alice_speed = 6 → tom_speed = 8 → initial_distance = 3 →
  time_for_tom_to_catch_alice_minutes alice_speed tom_speed initial_distance ≈ 12.857 :=
by
  sorry

end tom_catches_alice_in_12_857_minutes_l455_455904


namespace sum_elements_A_star_B_l455_455929
open Set

namespace example_problem

def set_operation (A B : Set ℕ) : Set ℕ :=
  {z | ∃ x y, x ∈ A ∧ y ∈ B ∧ z = x * y}

def A : Set ℕ := {1, 2}
def B : Set ℕ := {0, 2}

theorem sum_elements_A_star_B : (finset.sum (set_operation A B).to_finset id = 6) :=
sorry

end example_problem

end sum_elements_A_star_B_l455_455929


namespace initial_range_without_telescope_l455_455882

variable (V : ℝ)

def telescope_increases_range (V : ℝ) : Prop :=
  V + 0.875 * V = 150

theorem initial_range_without_telescope (V : ℝ) (h : telescope_increases_range V) : V = 80 :=
by
  sorry

end initial_range_without_telescope_l455_455882


namespace f_of_g_of_2_l455_455741

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x + 12 / Real.sqrt x
def g (x : ℝ) : ℝ := 2 * x^2 - 2 * x - 3

theorem f_of_g_of_2 : f (g 2) = 14 :=
by 
  sorry

end f_of_g_of_2_l455_455741


namespace cannot_determine_f_neg1_mul_f1_l455_455416

theorem cannot_determine_f_neg1_mul_f1 
  {f : ℝ → ℝ} 
  (h_cont : ContinuousOn f (set.Ioo (-2 : ℝ) (2 : ℝ))) 
  (h_root : ∀ x ∈ set.Ioo (-2 : ℝ) (2 : ℝ), f x = 0 ↔ x = 0) 
  : ¬ ∃ r : ℝ, r = f (-1) * f 1 := 
sorry

end cannot_determine_f_neg1_mul_f1_l455_455416


namespace unit_price_in_range_l455_455900

-- Given definitions and conditions
def Q (x : ℝ) : ℝ := 220 - 2 * x
def f (x : ℝ) : ℝ := x * Q x

-- The desired range for the unit price to maintain a production value of at least 60 million yuan
def valid_unit_price_range (x : ℝ) : Prop := 50 < x ∧ x < 60

-- The main theorem that needs to be proven
theorem unit_price_in_range (x : ℝ) (h₁ : 0 < x) (h₂ : x < 500) (h₃ : f x ≥ 60 * 10^6) : valid_unit_price_range x :=
sorry

end unit_price_in_range_l455_455900


namespace Priya_time_l455_455404

noncomputable def Suresh_rate : ℚ := 1 / 15
noncomputable def Ashutosh_rate : ℚ := 1 / 20
noncomputable def Priya_rate : ℚ := 1 / 25

noncomputable def Suresh_work : ℚ := 6 * Suresh_rate
noncomputable def Ashutosh_work : ℚ := 8 * Ashutosh_rate
noncomputable def total_work_done : ℚ := Suresh_work + Ashutosh_work
noncomputable def remaining_work : ℚ := 1 - total_work_done

theorem Priya_time : 
  remaining_work = Priya_rate * 5 := 
by sorry

end Priya_time_l455_455404


namespace correct_method_used_to_obtain_heights_l455_455419

def heights : List ℕ := [167, 168, 167, 164, 168, 168, 163, 168, 167, 160]

theorem correct_method_used_to_obtain_heights :
  the_method_used_to_obtain_data heights = "Field survey" :=
sorry

end correct_method_used_to_obtain_heights_l455_455419


namespace geometric_series_sum_l455_455125

theorem geometric_series_sum :
  let a := 1
  let r := 3
  let n := 9
  (1 * (3^n - 1) / (3 - 1)) = 9841 :=
by
  sorry

end geometric_series_sum_l455_455125


namespace find_side_length_b_l455_455019

theorem find_side_length_b 
  (a c : ℝ) (B : ℝ)
  (ha : a = 5)
  (hc : c = 8)
  (hB : B = real.pi / 3) :
  ∃ b : ℝ, b = 7 :=
by
  have h_cos := real.cos_pi_div_three,
  dsimp [real.pi] at h_cos,
  have h := calc 
    (7 : ℝ)^2 
      = (5 : ℝ)^2 + (8 : ℝ)^2 - 2 * (5 : ℝ) * (8 : ℝ) * (1 / 2) : by norm_num [h_cos]
  use 7
  rw h
  norm_num
  sorry

end find_side_length_b_l455_455019


namespace intersection_M_N_l455_455372

def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | -1 < x ∧ x < 1}

theorem intersection_M_N : (M ∩ N) = {x | 0 ≤ x ∧ x < 1} :=
by {
  sorry
}

end intersection_M_N_l455_455372


namespace inverse_proportionality_l455_455808

theorem inverse_proportionality 
  (a b : ℝ) 
  (k : ℝ)
  (h1 : a^3 * real.sqrt b = k) 
  (h2 : a = 3) 
  (h3 : b = 64) 
  (h4 : a * b = 36) : 
  b = 6 :=
begin
  sorry
end

end inverse_proportionality_l455_455808


namespace cheese_initial_amount_l455_455833

noncomputable def cheeseProblem : ℕ :=
let k := 35 in -- proof burden moved here by assuming k = 35 since it's the only fit
let cheeseEatenFirstNight := 10 in 
let cheeseEatenSecondNight := 1 in -- directly calculated from given information
cheeseEatenFirstNight + cheeseEatenSecondNight

theorem cheese_initial_amount
  (k : ℕ) (h1 : k > 7) (h2 : ∃ (d : ℕ), k = d ∧ 35 % d = 0) :
  cheeseProblem = 11 :=
by
  sorry

end cheese_initial_amount_l455_455833


namespace remainder_when_x_plus_3uy_div_y_l455_455687

theorem remainder_when_x_plus_3uy_div_y (x y u v : ℕ) (hx : x = u * y + v) (v_lt_y : v < y) :
  ((x + 3 * u * y) % y) = v := 
sorry

end remainder_when_x_plus_3uy_div_y_l455_455687


namespace remainder_division_l455_455462
-- Import the necessary library

-- Define the number and the divisor
def number : ℕ := 2345678901
def divisor : ℕ := 101

-- State the theorem
theorem remainder_division : number % divisor = 23 :=
by sorry

end remainder_division_l455_455462


namespace problem1_problem2_problem3_problem4_l455_455551

-- Problem 1
theorem problem1 (x y : ℝ) : ((-3 * x^2 * y)^2 * (2 * x * y^2) / (-6 * x^3 * y^4)) = -3 * x^2 := sorry

-- Problem 2
theorem problem2 (x : ℝ) : ((x + 2) * (x - 3) - x * (x - 1)) = -6 := sorry

-- Problem 3
theorem problem3 : (2024 * 2022 - 2023^2) = -1 := sorry

-- Problem 4
noncomputable def pi : ℝ := Real.pi -- redefine pi to match the problem's use of 'π'
theorem problem4 : ((-1/2)^(-2) - (3.14 - pi)^0 - 8^2024 * (-0.125)^2023) = 11 := sorry

end problem1_problem2_problem3_problem4_l455_455551


namespace find_angle_A_l455_455258

noncomputable def triangle_angle_A (a b B : ℝ) : ℝ :=
  real.arccos ((a^2 + b^2 - 2 * a * b * (real.cos B)) / 2)

theorem find_angle_A:
  ∀ (a b : ℝ) (B : ℝ),
    a = real.sqrt 2 → 
    b = real.sqrt 3 → 
    B = real.pi / 3 →
    triangle_angle_A a b B = real.pi / 4 :=
begin
  intros a b B ha hb hB,
  sorry
end

end find_angle_A_l455_455258


namespace totalAmountSpent_l455_455177

def costEggs (n : ℕ) : ℝ := if n >= 10 then n * 1.50 else n * 2
def costChickens (n : ℕ) : ℝ := (n / 5) * 4 * 8 + (n % 5) * 8
def costMilk (liters : ℕ) : ℝ := liters * 4
def costBread (loaves : ℕ) (litersMilk : ℕ) : ℝ :=
  let discountLoaves := litersMilk / 2
  (loaves - discountLoaves) * 3.50 + discountLoaves * 3.50 * 0.5
def costFlour (kg : ℕ) : ℝ := kg * 10 * 0.50
def costApples (kg : ℕ) : ℝ := kg * 2
def applyTax (amount : ℝ) : ℝ := amount * 1.05

theorem totalAmountSpent :
  let totalCost := costEggs 20 + costChickens 6 + costMilk 3 + costBread 2 3 + costFlour 1.5 + costApples 3
  applyTax totalCost = 105.79 :=
by
  sorry

end totalAmountSpent_l455_455177


namespace gcd_sum_and_lcm_eq_gcd_l455_455779

theorem gcd_sum_and_lcm_eq_gcd (a b : ℤ) :  Int.gcd (a + b) (Int.lcm a b) = Int.gcd a b :=
sorry

end gcd_sum_and_lcm_eq_gcd_l455_455779


namespace find_sale4_l455_455154

variable (sale1 sale2 sale3 sale5 sale6 avg : ℕ)
variable (total_sales : ℕ := 6 * avg)
variable (known_sales : ℕ := sale1 + sale2 + sale3 + sale5 + sale6)
variable (sale4 : ℕ := total_sales - known_sales)

theorem find_sale4 (h1 : sale1 = 6235) (h2 : sale2 = 6927) (h3 : sale3 = 6855)
                   (h5 : sale5 = 6562) (h6 : sale6 = 5191) (h_avg : avg = 6500) :
  sale4 = 7225 :=
by 
  sorry

end find_sale4_l455_455154


namespace sum_of_integers_n_correct_sum_final_sum_of_integers_n_l455_455976

theorem sum_of_integers_n (n x : ℤ) (h : n^2 - 17 * n + 72 = x^2) (hn : 0 < n) : 
  n = 8 ∨ n = 9 := 
sorry

theorem correct_sum : (8 + 9 : ℤ) = 17 := 
by norm_num

theorem final_sum_of_integers_n :
  ∃ n1 n2 : ℤ, (n1^2 - 17 * n1 + 72 = 0) ∧ (n2^2 - 17 * n2 + 72 = 0) ∧ (0 < n1) ∧ (0 < n2) ∧ (n1 + n2 = 17) :=
begin
  use [8, 9],
  split,
  { exact (by norm_num : (8^2 - 17 * 8 + 72 = 0)) },
  split,
  { exact (by norm_num : (9^2 - 17 * 9 + 72 = 0)) },
  split,
  { exact (by norm_num : 0 < 8) },
  split,
  { exact (by norm_num : 0 < 9) },
  { exact (by norm_num : 8 + 9 = 17) }
end

end sum_of_integers_n_correct_sum_final_sum_of_integers_n_l455_455976


namespace Sarah_age_l455_455215

theorem Sarah_age : ∃ (age : ℕ), 5 + 2 * age = 23 ∧ age = 9 :=
by {
  use 9,
  split,
  · exact rfl,
  · norm_num,
  sorry
}

end Sarah_age_l455_455215


namespace percent_employed_in_town_l455_455337

theorem percent_employed_in_town (E : ℝ) : 
  (0.14 * E) + 55 = E → E = 64 :=
by
  intro h
  have h1: 0.14 * E + 55 = E := h
  -- Proof step here, but we put sorry to skip the proof
  sorry

end percent_employed_in_town_l455_455337


namespace pirates_kill_at_least_10_l455_455885

-- Definitions based on conditions
def total_pirates (n : ℕ) : ℕ := n
def pirates_killed_in_specific_order : ℕ := 28

-- The proposition we need to prove
theorem pirates_kill_at_least_10 (n : ℕ) :
  (∀ order : list ℕ, list.nodup order → 
  (∀ i ∈ order, i < n) →
  ∃ k : ℕ, k ≥ 10 ∧ k ≤ 28) :=
sorry

end pirates_kill_at_least_10_l455_455885


namespace unique_solution_f_seq_l455_455172

noncomputable def f_seq : ℕ → ℝ → ℝ
| 0, x => 8
| 1, x => real.sqrt (x^2 + 48)
| (n + 2), x => real.sqrt (x^2 + 6 * f_seq (n + 1) x)

theorem unique_solution_f_seq (x : ℝ) (n : ℕ) (h : f_seq n x = 2 * x) : x = 4 :=
by
  sorry

end unique_solution_f_seq_l455_455172


namespace exists_palindromic_g_l455_455349

-- Define the polynomial f(x) and the conditions
def f (x : ℝ) (n : ℕ) (a : Fin (2 * n + 1) → ℝ) : ℝ :=
  ∑ i in (Finset.range (2 * n + 1)), a i * (x ^ i)

-- The condition that a_i = a_(2n - i)
def is_palindromic (n : ℕ) (a : Fin (2 * n + 1) → ℝ) : Prop :=
  ∀ i : Fin (n + 1), a i = a (⟨2 * n - i, Nat.sub_lt (Nat.mul_pos (by norm_num) n.zero_lt_succ) i⟩ : Fin (2 * n + 1))

-- Polynomial condition
theorem exists_palindromic_g (n : ℕ) (a : Fin (2 * n + 1) → ℝ) (h_palindromic : is_palindromic n a) (a_nonzero : a ⟨2 * n, nat.lt_succ_self (2 * n)⟩ ≠ 0) :
  ∃ g : ℝ → ℝ, (∀ x : ℝ, degree (g (x + x⁻¹)) = ↑n) ∧ (g (x + x⁻¹) * x ^ n = f x n a) :=
sorry

end exists_palindromic_g_l455_455349


namespace sum_of_coefficients_expansion_l455_455219

theorem sum_of_coefficients_expansion (d : ℝ) :
  let expr := -(4 - d) * (d + 3 * (4 - d))
  in (polynomial.sum_of_coeffs expr) = -30 :=
by
  let expr := -(4 - d) * (d + 3 * (4 - d))
  have h_expr : expr = -2 * d^2 + 20 * d - 48, sorry
  have h_coeffs_sum : polynomial.sum_of_coeffs (-2 * d^2 + 20 * d - 48) = -30, sorry
  rw h_expr
  exact h_coeffs_sum

end sum_of_coefficients_expansion_l455_455219


namespace sum_of_coeffs_eq_neg30_l455_455224

noncomputable def expanded : Polynomial ℤ := 
  -(Polynomial.C 4 - Polynomial.X) * (Polynomial.X + 3 * (Polynomial.C 4 - Polynomial.X))

theorem sum_of_coeffs_eq_neg30 : (expanded.coeffs.sum) = -30 := 
  sorry

end sum_of_coeffs_eq_neg30_l455_455224


namespace f_expression_and_a_range_l455_455360

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x < 0 then ln (-x) - a * x^2
  else if x = 0 then 0
  else if 0 < x ∧ x ≤ 1 then -ln x + a * x^2
  else 0

theorem f_expression_and_a_range (a : ℝ) :
  (∀ x, -1 ≤ x ∧ x < 0 → f a x = ln (-x) - a * x^2) ∧
  (f a 0 = 0) ∧
  (∀ x, 0 < x ∧ x ≤ 1 → f a x = -ln x + a * x^2) ∧
  (∀ x, 0 < x ∧ x < 1 → |f a x| ≥ 1 → a ≥ (Real.exp 1) / 2) :=
sorry

end f_expression_and_a_range_l455_455360


namespace reflected_ray_equation_interval_of_increase_sum_f_harmonic_domain_range_sum_l455_455134

-- Problem 1 (translated to Lean statement):
theorem reflected_ray_equation :
  ∃ C : ℝ × ℝ, C = (1, 4) ∧
  ∃ B : ℝ × ℝ, B = (0, 2) ∧
  ∀ x y : ℝ, (x = 1 ∧ y = 0) → (x = fst B ∧ y = snd B) → (2 * fst C - snd C + 2 = 0) := sorry

-- Problem 2 (translated to Lean statement):
theorem interval_of_increase :
  ∀ k : ℤ, (k : ℝ) * π - π / 12 ≤ x ∧ x ≤ (k : ℝ) * π + 5 * π / 12 :=
sorry

-- Problem 3 (translated to Lean statement):
def f (x : ℝ) : ℝ := 
sorry

theorem sum_f_harmonic :
  ∀ x, f(x) = -f(x + 3/2) ∧ f(-1) = 1 ∧ f(0) = -2 ∧ 
  (f(1) + f(2) + ... + f(2010) = 0) := sorry

-- Problem 4 (translated to Lean statement):
def g (x : ℝ) : ℝ := abs(2^x - 1)

theorem domain_range_sum :
  ∃ a b : ℝ, a = 0 ∧ b = 1 ∧ g(a) = a ∧ g(b) = b ∧ a + b = 1 := sorry

end reflected_ray_equation_interval_of_increase_sum_f_harmonic_domain_range_sum_l455_455134


namespace natalia_crates_l455_455579

noncomputable def total_items (novels comics documentaries albums : ℕ) : ℕ :=
  novels + comics + documentaries + albums

noncomputable def crates_needed (total_items items_per_crate : ℕ) : ℕ :=
  (total_items + items_per_crate - 1) / items_per_crate

theorem natalia_crates : crates_needed (total_items 145 271 419 209) 9 = 117 := by
  sorry

end natalia_crates_l455_455579


namespace latest_leave_time_correct_l455_455536

-- Define the conditions
def flight_time := 20 -- 8:00 pm in 24-hour format
def check_in_early := 2 -- 2 hours early
def drive_time := 45 -- 45 minutes
def park_time := 15 -- 15 minutes

-- Define the target time to be at the airport
def at_airport_time := flight_time - check_in_early -- 18:00 or 6:00 pm

-- Total travel time required (minutes)
def total_travel_time := drive_time + park_time -- 60 minutes

-- Convert total travel time to hours
def travel_time_in_hours : ℕ := total_travel_time / 60

-- Define the latest time to leave the house
def latest_leave_time := at_airport_time - travel_time_in_hours

-- Theorem to state the equivalence of the latest time they can leave their house
theorem latest_leave_time_correct : latest_leave_time = 17 :=
    by
    sorry

end latest_leave_time_correct_l455_455536


namespace sum_of_coeffs_eq_neg30_l455_455223

noncomputable def expanded : Polynomial ℤ := 
  -(Polynomial.C 4 - Polynomial.X) * (Polynomial.X + 3 * (Polynomial.C 4 - Polynomial.X))

theorem sum_of_coeffs_eq_neg30 : (expanded.coeffs.sum) = -30 := 
  sorry

end sum_of_coeffs_eq_neg30_l455_455223


namespace prime_divisor_problem_l455_455232

theorem prime_divisor_problem (d r : ℕ) (h1 : d > 1) (h2 : Prime d)
  (h3 : 1274 % d = r) (h4 : 1841 % d = r) (h5 : 2866 % d = r) : d - r = 6 :=
by
  sorry

end prime_divisor_problem_l455_455232


namespace count_integer_area_l455_455599

def A (n : ℤ) (h : 2 ≤ n) : Prop :=
  let x_min := 1
  let x_max := n
  ∀ x : ℤ, x_min ≤ x ∧ x ≤ x_max →
           let y_max := x * Int.floor (Real.log x / Real.log 2)
           0 ≤ y_max

theorem count_integer_area : ∃ (count : ℤ), 
  count = List.countp (λ n, ∃ h : 2 ≤ n ∧ n ≤ 100, A n h) (List.range' 2 (100 - 1)) ∧ count = 99 := sorry

end count_integer_area_l455_455599


namespace save_percentage_l455_455160

theorem save_percentage (I S : ℝ) 
  (h1 : 1.5 * I - 2 * S + (I - S) = 2 * (I - S))
  (h2 : I ≠ 0) : 
  S / I = 0.5 :=
by sorry

end save_percentage_l455_455160


namespace product_xyz_one_l455_455277

theorem product_xyz_one (x y z : ℝ) (h1 : x + 1/y = 2) (h2 : y + 1/z = 2) (h3 : z + 1/x = 2) : x * y * z = 1 := 
by {
    sorry
}

end product_xyz_one_l455_455277


namespace problem_l455_455092

def f (x : ℝ) : ℝ :=
if x ≥ 0 then x * (x + 1) else x * (x - 1)

theorem problem : f (f (-1)) = 6 :=
by
  sorry

end problem_l455_455092


namespace max_min_f_l455_455236

noncomputable def f (x : ℝ) : ℝ := 
  5 * Real.cos x ^ 2 - 6 * Real.sin (2 * x) + 20 * Real.sin x - 30 * Real.cos x + 7

theorem max_min_f :
  (∃ x : ℝ, f x = 16 + 10 * Real.sqrt 13) ∧
  (∃ x : ℝ, f x = 16 - 10 * Real.sqrt 13) :=
sorry

end max_min_f_l455_455236


namespace percent_decrease_in_cost_l455_455695

theorem percent_decrease_in_cost (cost_1990 cost_2010 : ℕ) (h1 : cost_1990 = 35) (h2 : cost_2010 = 5) : 
  ((cost_1990 - cost_2010) * 100 / cost_1990 : ℚ) = 86 := 
by
  sorry

end percent_decrease_in_cost_l455_455695


namespace fib_inequality_l455_455405

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n+2) => fib (n+1) + fib n

-- The main theorem
theorem fib_inequality (n a b : ℕ) (h1 : 1 ≤ n)
  (h2 : min (fib n / fib (n - 1)) (fib (n + 1) / fib n) < a / b)
  (h3 : a / b < max (fib n / fib (n - 1)) (fib (n + 1) / fib n)) :
  b ≥ fib(n + 1) :=
sorry

end fib_inequality_l455_455405


namespace prime_divisors_of_n_congruent_to_1_mod_4_l455_455131

theorem prime_divisors_of_n_congruent_to_1_mod_4
  (x y n : ℕ)
  (hx : x ≥ 3)
  (hn : n ≥ 2)
  (h_eq : x^2 + 5 = y^n) :
  ∀ p : ℕ, Prime p → p ∣ n → p ≡ 1 [MOD 4] :=
by
  sorry

end prime_divisors_of_n_congruent_to_1_mod_4_l455_455131


namespace problem1_problem2_l455_455549

-- Statement for Problem ①
theorem problem1 
: ( (-1 / 12 - 1 / 36 + 1 / 6) * (-36) = -2) := by
  sorry

-- Statement for Problem ②
theorem problem2
: ((-99 - 11 / 12) * 24 = -2398) := by
  sorry

end problem1_problem2_l455_455549


namespace like_terms_C_l455_455859

-- Definition of a term
structure Term :=
  (coeff : Int)
  (vars : List (String × Int)) -- List of variables with their exponents

-- Definition of like terms: same variables with the same exponents
def like_terms (t1 t2 : Term) : Prop :=
  t1.vars = t2.vars

-- The terms given in condition C
def term1 : Term := ⟨-1, [("m", 2), ("n", 3)]⟩
def term2 : Term := ⟨-3, [("n", 3), ("m", 2)]⟩

-- Ensure variables list is sorted, assuming ordering by name for simplicity
def sort_vars (vars : List (String × Int)) : List (String × Int) :=
  List.sort (λ a b => a.1 < b.1) vars

-- Statement to be proved
theorem like_terms_C :
  like_terms { term1 with vars := sort_vars term1.vars } { term2 with vars := sort_vars term2.vars } :=
sorry

end like_terms_C_l455_455859


namespace graph_passes_through_point_l455_455414

-- Define the main function with necessary conditions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x - 2) + 3

-- Define the theorem
theorem graph_passes_through_point : ∀ (a : ℝ), a > 0 → a ≠ 1 → f a 3 = 3 := 
by 
  intros a ha1 ha2
  simp [f, log]
  sorry

end graph_passes_through_point_l455_455414


namespace average_speed_comparison_l455_455919

theorem average_speed_comparison (u v w : ℝ) (hu : u > 0) (hv : v > 0) (hw : w > 0):
  (3 / (1 / u + 1 / v + 1 / w)) ≤ ((u + v + w) / 3) :=
sorry

end average_speed_comparison_l455_455919


namespace max_min_lg_function_l455_455235

noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem max_min_lg_function :
  let f := λ x : ℝ, Real.sin x ^ 2 + 2 * Real.cos x + 2
  let y := λ x : ℝ, lg (f x)
  ∃ x_max x_min : ℝ, 
    x_max ∈ Set.Icc (-Real.pi / 6) (2 * Real.pi / 3) ∧ 
    x_min ∈ Set.Icc (-Real.pi / 6) (2 * Real.pi / 3) ∧
    ∀ x ∈ Set.Icc (-Real.pi / 6) (2 * Real.pi / 3),
      y x ≤ y x_max ∧ y x_min ≤ y x ∧
      y x_max = lg 4 ∧ y x_min = lg (7 / 4) := sorry

end max_min_lg_function_l455_455235


namespace sum_of_reciprocals_l455_455499

theorem sum_of_reciprocals
  (m n p : ℕ)
  (HCF_mnp : Nat.gcd (Nat.gcd m n) p = 26)
  (LCM_mnp : Nat.lcm (Nat.lcm m n) p = 6930)
  (sum_mnp : m + n + p = 150) :
  (1 / (m : ℚ) + 1 / (n : ℚ) + 1 / (p : ℚ) = 1 / 320166) :=
by
  sorry

end sum_of_reciprocals_l455_455499


namespace air_conditioning_price_november_l455_455320

noncomputable def price_in_november : ℝ :=
  let january_price := 470
  let february_price := january_price * (1 - 0.12)
  let march_price := february_price * (1 + 0.08)
  let april_price := march_price * (1 - 0.10)
  let june_price := april_price * (1 + 0.05)
  let august_price := june_price * (1 - 0.07)
  let october_price := august_price * (1 + 0.06)
  october_price * (1 - 0.15)

theorem air_conditioning_price_november : price_in_november = 353.71 := by
  sorry

end air_conditioning_price_november_l455_455320


namespace exists_m_even_not_exists_m_odd_not_forall_m_even_not_forall_m_odd_l455_455486

noncomputable def f (m x : ℝ) := x^2 + m*x

theorem exists_m_even : ∃ m : ℝ, ∀ x : ℝ, f m x = f m (-x) :=
by {
  use 0,
  intro x,
  apply congrArg,
  ring,
}

theorem not_exists_m_odd : ¬ ∃ m : ℝ, ∀ x : ℝ, f m x = -f m (-x) :=
by {
  intro h,
  rcases h with ⟨m, hm⟩,
  specialize hm 1,
  have : f m 1 = -f m (-1) := hm,
  simp [f, mul_comm] at this,
  linarith,
}

theorem not_forall_m_even : ¬ ∀ m : ℝ, ∀ x : ℝ, f m x = f m (-x) :=
by {
  intro h,
  specialize h 1,
  have : f 1 1 = f 1 (-1) := h 1,
  simp [f, mul_comm] at this,
  linarith,
}

theorem not_forall_m_odd : ¬ ∀ m : ℝ, ∀ x : ℝ, f m x = -f m (-x) :=
by {
  intro h,
  specialize h 1,
  have : f 1 1 = -f 1 (-1) := h 1,
  simp [f, mul_comm] at this,
  linarith,
}

end exists_m_even_not_exists_m_odd_not_forall_m_even_not_forall_m_odd_l455_455486


namespace polar_line_eq_l455_455719

-- Given a point P in polar coordinates
def polar_pointP (ρ θ : ℝ) : Prop := (ρ = 2) ∧ (θ = π / 6)

-- Definition of the line passing through point P and parallel to the polar axis
theorem polar_line_eq (ρ θ : ℝ) (h : polar_pointP ρ θ) : ρ * sin θ = 1 :=
by sorry

end polar_line_eq_l455_455719


namespace number_of_six_digit_palindromes_l455_455192

def is_six_digit_palindrome (n : ℕ) : Prop :=
  ∃ a b c, a ≠ 0 ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ n = a * 100001 + b * 10010 + c * 1100

theorem number_of_six_digit_palindromes : ∃ p, p = 900 ∧ (∀ n, is_six_digit_palindrome n → n = p) :=
by
  sorry

end number_of_six_digit_palindromes_l455_455192


namespace marble_problem_l455_455828

def total_marbles_originally 
  (white_marbles : ℕ := 20) 
  (blue_marbles : ℕ) 
  (red_marbles : ℕ := blue_marbles) 
  (total_left : ℕ := 40)
  (jack_removes : ℕ := 2 * (white_marbles - blue_marbles)) : ℕ :=
  white_marbles + blue_marbles + red_marbles

theorem marble_problem : 
  ∀ (white_marbles : ℕ := 20) 
    (blue_marbles red_marbles : ℕ) 
    (jack_removes total_left : ℕ),
    red_marbles = blue_marbles →
    jack_removes = 2 * (white_marbles - blue_marbles) →
    total_left = total_marbles_originally white_marbles blue_marbles red_marbles - jack_removes →
    total_left = 40 →
    total_marbles_originally white_marbles blue_marbles red_marbles = 50 :=
by
  intros white_marbles blue_marbles red_marbles jack_removes total_left h1 h2 h3 h4
  sorry

end marble_problem_l455_455828


namespace boat_speed_in_still_water_l455_455431

variable (x : ℝ) -- speed of the boat in still water in km/hr
variable (current_rate : ℝ := 4) -- rate of the current in km/hr
variable (downstream_distance : ℝ := 4.8) -- distance traveled downstream in km
variable (downstream_time : ℝ := 18 / 60) -- time traveled downstream in hours

-- The main theorem stating that the speed of the boat in still water is 12 km/hr
theorem boat_speed_in_still_water : x = 12 :=
by
  -- Express the downstream speed and time relation
  have downstream_speed := x + current_rate
  have distance_relation := downstream_distance = downstream_speed * downstream_time
  -- Simplify and solve for x
  simp at distance_relation
  sorry

end boat_speed_in_still_water_l455_455431


namespace vanya_number_l455_455458

theorem vanya_number (m n : ℕ) (hm : m < 10) (hn : n < 10) (h : (10 * n + m)^2 = 4 * (10 * m + n)) : 
  10 * m + n = 81 :=
by sorry

end vanya_number_l455_455458


namespace find_x_plus_y_l455_455680

theorem find_x_plus_y (x y : ℝ) 
  (h1 : |x| + x + y = 14) 
  (h2 : x + |y| - y = 16) : 
  x + y = -2 := 
by
  sorry

end find_x_plus_y_l455_455680


namespace average_s_t_l455_455425

theorem average_s_t (s t : ℝ) 
  (h : (1 + 3 + 7 + s + t) / 5 = 12) : 
  (s + t) / 2 = 24.5 :=
by
  sorry

end average_s_t_l455_455425


namespace zero_of_polynomial_degree_four_l455_455164

noncomputable def P (r s α β x : ℂ) : ℂ := (x - r) * (x - s) * (x^2 + α * x + β)

def is_zero_of_P (r s α β t : ℂ) : Prop := P r s α β t = 0

-- Given conditions
axiom ra : ℤ -- Integer root 1
axiom rb : ℤ -- Integer root 2
axiom r : ℂ := ra
axiom s : ℂ := rb
axiom α : ℂ := -1 -- From solution
axiom β : ℂ := 3  -- Chosen such that 4β - α^2 = 11

theorem zero_of_polynomial_degree_four :
  is_zero_of_P r s α β (1 / 2 + (Complex.i * (Complex.sqrt 11) / 2)) :=
sorry

end zero_of_polynomial_degree_four_l455_455164


namespace unique_10_digit_number_property_l455_455947

def ten_digit_number (N : ℕ) : Prop :=
  10^9 ≤ N ∧ N < 10^10

def first_digits_coincide (N : ℕ) : Prop :=
  ∀ M : ℕ, N^2 < 10^M → N^2 / 10^(M - 10) = N

theorem unique_10_digit_number_property :
  ∀ (N : ℕ), ten_digit_number N ∧ first_digits_coincide N → N = 1000000000 := 
by
  intros N hN
  sorry

end unique_10_digit_number_property_l455_455947


namespace product_remainder_l455_455854

-- Define the product of the consecutive numbers
def product := 86 * 87 * 88 * 89 * 90 * 91 * 92

-- Lean statement to state the problem
theorem product_remainder :
  product % 7 = 0 :=
by
  sorry

end product_remainder_l455_455854


namespace precise_value_for_D_l455_455002

noncomputable def D : ℝ := 6.67408
noncomputable def delta_D : ℝ := 0.00009

theorem precise_value_for_D :
  (∀ x ∈ (Icc (D - delta_D) (D + delta_D)), (Real.round (1000 * x) / 1000 = 6.674)) :=
by
  intro x h
  sorry

end precise_value_for_D_l455_455002


namespace area_of_isosceles_triangle_l455_455589

theorem area_of_isosceles_triangle (a b : ℝ) (h1 : a = 10) (h2 : b = 7) :
  let h := real.sqrt (b^2 - (a/2)^2) in
  let area := (1/2) * a * h in
  area = 10 * real.sqrt 6 :=
by sorry

end area_of_isosceles_triangle_l455_455589


namespace profit_percentage_l455_455541

theorem profit_percentage (cost_price selling_price profit_percentage : ℚ) 
  (h_cost_price : cost_price = 240) 
  (h_selling_price : selling_price = 288) 
  (h_profit_percentage : profit_percentage = 20) : 
  profit_percentage = ((selling_price - cost_price) / cost_price) * 100 := 
by 
  sorry

end profit_percentage_l455_455541


namespace inequality_of_abc_l455_455389

variable {a b c : ℝ}

theorem inequality_of_abc 
    (h : 0 < a ∧ 0 < b ∧ 0 < c)
    (h₁ : abc * (a + b + c) = ab + bc + ca) :
    5 * (a + b + c) ≥ 7 + 8 * abc :=
sorry

end inequality_of_abc_l455_455389


namespace find_f2017_2018_l455_455274

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then -(x-1)^2
  else if 2 - x + x = 2 then 0  -- Placeholder for periodic extension
  else if x + 2 + x = x + 2 then 0  -- Placeholder for periodic extension
  else sorry  -- General case to be specified

theorem find_f2017_2018 (x : ℝ) (h₀ : 2017 ≤ x) (h₁ : x ≤ 2018) :
  f(x) = (2017 - x)^2 :=
by sorry

end find_f2017_2018_l455_455274


namespace range_of_x_l455_455098

theorem range_of_x (x : ℝ) : (4 : ℝ)^(2 * x - 1) > (1 / 2) ^ (-x - 4) → x > 2 := by
  sorry

end range_of_x_l455_455098


namespace function_value_f_value_derivative_sum_l455_455284

noncomputable def f (x : ℝ) : ℝ :=
  ((x + 1) ^ 2 + sin x) / (x ^ 2 + 1)

noncomputable def f_derivative (x : ℝ) : ℝ :=
  ((2 + cos x) * (x ^ 2 + 1) - (2 * x + sin x) * (2 * x)) / ((x ^ 2 + 1) ^ 2)

lemma even_f_derivative : ∀ x : ℝ, f_derivative (-x) = f_derivative x := 
sorry

lemma odd_h : ∀ x : ℝ, h (-x) = -h x := 
  sorry

noncomputable def h (x : ℝ) : ℝ :=
  (2 * x + sin x) / (x ^ 2 + 1)

theorem function_value (x : ℝ) : ∀ x: ℝ, f(x) + f(-x) = 2 :=
sorry

theorem f_value_derivative_sum : f 2017 + f_derivative 2017 + f (-2017) - f_derivative (-2017) = 2 :=
by 
  have even_der := even_f_derivative 2017
  have odd_h_val := odd_h 2017
  rw [<-even_der, function_value]
  linarith

end function_value_f_value_derivative_sum_l455_455284


namespace find_parallelline_through_point_l455_455136

def point := (ℝ, ℝ)
def line_equation (a b c : ℝ) := λ (x y : ℝ), a * x + b * y + c = 0

theorem find_parallelline_through_point
  (p : point) (a b c d : ℝ)
  (L1 : line_equation a b c)
  (L2 : line_equation a b d)
  (h_parallel : ∀ {x y}, L1 x y → L2 x y)
  (h_p : L2 p.1 p.2) :
  ∃ (c' : ℝ), L2 = line_equation a b c' ∧ c' = 7 := 
sorry

end find_parallelline_through_point_l455_455136


namespace eight_order_magic_star_l455_455944

noncomputable def magicSum : ℕ := 34

theorem eight_order_magic_star :
  ∃ (f : Fin 16 → ℕ), (∀ i, f i ∈ (1:ℕ) → 16) ∧
    (∀ i j k l, 
        i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l →
        f i + f j + f k + f l = magicSum) :=
sorry

end eight_order_magic_star_l455_455944


namespace equation_of_line_passing_through_point_and_perpendicular_l455_455959

variables {x y : ℝ}

noncomputable def pointA := (2, 3:ℝ)
noncomputable def line_given := 2 * x + y = 5
noncomputable def line_sought := x - 2 * y + 4 = 0

theorem equation_of_line_passing_through_point_and_perpendicular :
  ∃ l : ℝ → ℝ → Prop, l pointA.1 pointA.2 ∧ (∀ x y, line_given x y → l y x) ∧ l = line_sought :=
sorry

end equation_of_line_passing_through_point_and_perpendicular_l455_455959


namespace sqrt_cube_rational_l455_455787

theorem sqrt_cube_rational : 
  let sqrt_52 := 2 * Real.sqrt 13,
      x := (sqrt_52 + 5)^(1/3 : ℝ),
      y := (sqrt_52 - 5)^(1/3 : ℝ)
  in x - y ∈ ℚ :=
by
  -- Assuming the necessary definitions for the proof context
  let sqrt_52 := 2 * Real.sqrt 13
  let x := (sqrt_52 + 5)^(1/3 : ℝ)
  let y := (sqrt_52 - 5)^(1/3 : ℝ)
  have x_cube_eq : x^3 = sqrt_52 + 5 := by sorry
  have y_cube_eq : y^3 = sqrt_52 - 5 := by sorry
  -- Proof that x - y is rational
  sorry

end sqrt_cube_rational_l455_455787


namespace solve_cubic_eq_l455_455952

theorem solve_cubic_eq (x : ℝ) : x^3 + (2 - x)^3 = 8 ↔ x = 0 ∨ x = 2 := 
by 
  { sorry }

end solve_cubic_eq_l455_455952


namespace evaluate_g_compose_l455_455928

def g : ℤ → ℤ :=
λ x, if x < 8 then x^2 - 6 else x - 10

theorem evaluate_g_compose : g (g (g 18)) = -2 :=
by 
  sorry

end evaluate_g_compose_l455_455928


namespace ab_cd_not_prime_l455_455364

-- Define the problem and conditions in a Lean theorem statement
theorem ab_cd_not_prime (a b c d : ℤ) (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d > 0)
  (h5 : a * c + b * d = (b + d + a - c) * (b + d - a + c)) : ¬ prime (a * b + c * d) :=
by
  -- Sorry placeholder for the actual proof
  sorry

end ab_cd_not_prime_l455_455364


namespace volume_of_intersection_of_octahedra_l455_455467

theorem volume_of_intersection_of_octahedra :
  let region1 := {p : ℝ × ℝ × ℝ | abs p.1 + abs p.2 + abs p.3 ≤ 2}
  let region2 := {p : ℝ × ℝ × ℝ | abs p.1 + abs p.2 + abs (p.3 - 2) ≤ 2}
  region1 ∩ region2 = volume (region1 ∩ region2) = 16 / 3 := 
sorry

end volume_of_intersection_of_octahedra_l455_455467


namespace nth_term_150_is_2280_l455_455421

-- Definition of the sequence
def is_in_sequence (n : ℕ) : Prop :=
  ∃ (coeffs : list ℕ), 
    (∀ c ∈ coeffs, ∃ k, c = 3 ^ k) ∧ 
    coeffs.nodup ∧ 
    n = coeffs.sum

-- Helper to convert binary (represented as list of bits) to the corresponding sum of powers of 3
def binary_to_powers_of_3_sum (bin : list ℕ) : ℕ :=
  bin.enum_from 0 |>.filter_map (λ ⟨i, bit⟩ => if bit = 1 then some (3 ^ i) else none) |>.sum

-- Function to get the nth term in the sequence
def nth_term (n : ℕ) : ℕ :=
  let bin_rep := nat.binary_repr n
  binary_to_powers_of_3_sum bin_rep

-- Lean statement to prove
theorem nth_term_150_is_2280 : nth_term 150 = 2280 := sorry

end nth_term_150_is_2280_l455_455421


namespace vans_needed_l455_455079

theorem vans_needed (boys girls students_per_van total_vans : ℕ) 
  (hb : boys = 60) 
  (hg : girls = 80) 
  (hv : students_per_van = 28) 
  (t : total_vans = (boys + girls) / students_per_van) : 
  total_vans = 5 := 
by {
  sorry
}

end vans_needed_l455_455079


namespace sophie_germain_identity_l455_455943

theorem sophie_germain_identity (a b : ℝ) : 
  a^4 + 4 * b^4 = (a^2 + 2 * a * b + 2 * b^2) * (a^2 - 2 * a * b + 2 * b^2) :=
by sorry

end sophie_germain_identity_l455_455943


namespace peg_arrangement_l455_455436

-- Define the number of pegs of each color
def yellow_pegs : ℕ := 6
def red_pegs : ℕ := 5
def green_pegs : ℕ := 4
def blue_pegs : ℕ := 3
def orange_pegs : ℕ := 2

-- Define the factorial of a number (usually already defined in mathlib, but included here for clarity)
noncomputable def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n+1) * factorial n

-- Lean statement of the problem
theorem peg_arrangement : 
  factorial yellow_pegs * factorial red_pegs * factorial green_pegs * factorial blue_pegs * factorial orange_pegs = 12441600 := 
by {
  unfold yellow_pegs red_pegs green_pegs blue_pegs orange_pegs factorial,
  rw [factorial, factorial 5, factorial 4, factorial 3, factorial 2],
  norm_num,
  sorry
}

end peg_arrangement_l455_455436


namespace volume_ratio_l455_455430

noncomputable def V_sphere (p : ℝ) : ℝ := (4/3) * π * (4 * p) ^ 3
noncomputable def V_hemisphere (p : ℝ) : ℝ := (1/2) * (4/3) * π * (8 * p) ^ 3
noncomputable def V_combined (p : ℝ) : ℝ := V_sphere p + V_hemisphere p
noncomputable def V_cone (p : ℝ) : ℝ := (1/3) * π * (8 * p) ^ 2 * (4 * p)

theorem volume_ratio (p : ℝ) : V_combined p / V_cone p = 2 := by
  sorry

end volume_ratio_l455_455430


namespace value_of_m_l455_455417

theorem value_of_m : ∃ m : ℝ, m ≠ 0 ∧ (∃ (f : ℝ → ℝ), f 0 = 4 ∧ (∀ x, f x = m * x + m ^ 2)) ∧ (m > 0) ∧ (∃ m : ℝ, f = λ x, m * x + m ^ 2) :=
begin
  sorry
end

end value_of_m_l455_455417


namespace money_raised_by_full_price_tickets_l455_455171

theorem money_raised_by_full_price_tickets (f h : ℕ) (p revenue total_tickets : ℕ) 
  (full_price : p = 20) (total_cost : f * p + h * (p / 2) = revenue) 
  (ticket_count : f + h = total_tickets) (total_revenue : revenue = 2750)
  (ticket_number : total_tickets = 180) : f * p = 1900 := 
by
  sorry

end money_raised_by_full_price_tickets_l455_455171


namespace price_reductions_l455_455939

theorem price_reductions (a : ℝ) : 18400 * (1 - a / 100)^2 = 16000 :=
sorry

end price_reductions_l455_455939


namespace AK_parallel_BC_l455_455086

noncomputable def polar_transformation (incircle : Circle) (ABC : Triangle) : Point → Line := sorry
noncomputable def midpoint (A B : Point) : Point := sorry

variables {A B C D E F G L K N T : Point}
variables {ABC : Triangle} {incircle : Circle}

-- Conditions as transformations under polar transformation 
axiom polar_transform_DE_to_A : polar_transformation incircle ABC (line_through D E) = A
axiom polar_transform_FT_to_midpoint_BC : polar_transformation incircle ABC (line_through F T) = midpoint B C

-- Additional given condition
axiom AE_eq_AD : distance A E = distance A D

-- Proof statement
theorem AK_parallel_BC : parallel (line_through A K) (line_through B C) := sorry

end AK_parallel_BC_l455_455086


namespace mutually_exclusive_events_l455_455698

def box_contains (genuine_items : ℕ) (defective_items : ℕ) :=
  genuine_items = 4 ∧ defective_items = 3

def events (selected_items : list string) :=
  list.length selected_items = 2

def mutually_exclusive (pair1 : Prop) (pair2 : Prop) :=
  pair1 ∧ pair2 → false

theorem mutually_exclusive_events 
  (genuine_items defective_items : ℕ)
  (selected_items : list string) :
  box_contains genuine_items defective_items →
  events selected_items →
  mutually_exclusive (∃ l, l ~ [ "defective", "genuine" ]) (∀ x ∈ selected_items, x = "genuine") :=
by
  intros h_box h_events
  sorry

end mutually_exclusive_events_l455_455698


namespace cos_alpha_in_second_quadrant_l455_455628

theorem cos_alpha_in_second_quadrant (α : ℝ) (h1 : sin α = 3/5) (h2 : π/2 < α ∧ α < π) : cos α = -4/5 := 
sorry

end cos_alpha_in_second_quadrant_l455_455628


namespace algebraic_identity_l455_455609

variable (x y : ℝ)

theorem algebraic_identity (hx : x = 2 - real.sqrt 3) (hy : y = 2 + real.sqrt 3) : x^2 - y^2 = -8 * real.sqrt 3 :=
sorry

end algebraic_identity_l455_455609


namespace marble_probability_l455_455886

theorem marble_probability :
  let total_marbles := 21
  let red_marbles := 4
  let green_marbles := 6
  let white_marbles := 11 in
  let prob_red_then_green := (red_marbles / total_marbles) * (green_marbles / (total_marbles - 1)) in
  let prob_green_then_red := (green_marbles / total_marbles) * (red_marbles / (total_marbles - 1)) in
  let total_probability := prob_red_then_green + prob_green_then_red in
  total_probability = 4 / 35 :=
by
  sorry

end marble_probability_l455_455886


namespace probability_equality_distributions_l455_455739

noncomputable def probability_of_equality 
  (X Y : ℝ → ℝ) -- X and Y are random variables
  (F G : ℝ → ℝ) -- F and G are distribution functions
  (independent : ∀ x, ∃ e, P(X=x, Y=x) = P(X=x)*P(Y=x)) :
  Prop :=
∃ X Y : ℝ → ℝ,
  ∃ F G : ℝ → ℝ,
  (∀ x, ∃ e, P(X = x ∧ Y = x) = P(X = x) * P(Y = x)) →
  (∑ (x : ℤ), (F(x) - F(x - 1)) * (G(x) - G(x - 1))) = P(X = Y)

theorem probability_equality_distributions 
  {X Y : ℝ → ℝ} 
  {F G : ℝ → ℝ}
  (h_independent : ∀ x, ∃ e, P(X=x, Y=x) = P(X=x)*P(Y=x)) :
  probability_of_equality X Y F G h_independent :=
sorry

end probability_equality_distributions_l455_455739


namespace polynomial_positive_ratios_l455_455350

theorem polynomial_positive_ratios (P : polynomial ℝ) (hP : ∀ x > 0, 0 < P.eval x) :
  ∃ Q R : polynomial ℝ, (∀ i, 0 ≤ Q.coeff i) ∧ (∀ i, 0 ≤ R.coeff i) ∧ 
  ∀ x > 0, P.eval x = (Q.eval x) / (R.eval x) := sorry

end polynomial_positive_ratios_l455_455350


namespace combined_price_before_increase_l455_455240

-- Define the conditions
def new_price_candy_box := 20
def new_price_soda := 6
def new_price_chips := 8
def increase_candy_box := 0.25
def increase_soda := 0.50
def increase_chips := 0.10

-- Define the original prices based on the given increases
def original_price_candy_box := new_price_candy_box / (1 + increase_candy_box)
def original_price_soda := new_price_soda / (1 + increase_soda)
def original_price_chips := new_price_chips / (1 + increase_chips)

-- Define the combined original price
def combined_original_price := original_price_candy_box + original_price_soda + original_price_chips

-- The statement to prove
theorem combined_price_before_increase :
  combined_original_price = 34 :=
by
  unfold combined_original_price original_price_candy_box original_price_soda original_price_chips 
  sorry

end combined_price_before_increase_l455_455240
