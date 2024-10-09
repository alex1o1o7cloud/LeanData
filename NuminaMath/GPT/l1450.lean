import Mathlib

namespace red_button_probability_l1450_145025

/-
Mathematical definitions derived from the problem:
Initial setup:
- Jar A has 6 red buttons and 10 blue buttons.
- Same number of red and blue buttons are removed. Jar A retains 3/4 of original buttons.
- Calculate the final number of red buttons in Jar A and B, and determine the probability both selected buttons are red.
-/
theorem red_button_probability :
  let initial_red := 6
  let initial_blue := 10
  let total_buttons := initial_red + initial_blue
  let removal_fraction := 3 / 4
  let final_buttons := (3 / 4 : ℚ) * total_buttons
  let removed_buttons := total_buttons - final_buttons
  let removed_each_color := removed_buttons / 2
  let final_red_A := initial_red - removed_each_color
  let final_red_B := removed_each_color
  let prob_red_A := final_red_A / final_buttons
  let prob_red_B := final_red_B / removed_buttons
  prob_red_A * prob_red_B = 1 / 6 :=
by
  sorry

end red_button_probability_l1450_145025


namespace room_dimension_l1450_145022

theorem room_dimension {a : ℝ} (h1 : a > 0) 
  (h2 : 4 = 2^2) 
  (h3 : 14 = 2 * (7)) 
  (h4 : 2 * a = 14) :
  (a + 2 * a - 2 = 19) :=
sorry

end room_dimension_l1450_145022


namespace outfits_count_l1450_145006

theorem outfits_count (shirts ties pants belts : ℕ) (h_shirts : shirts = 7) (h_ties : ties = 5) (h_pants : pants = 4) (h_belts : belts = 2) : 
  (shirts * pants * (ties + 1) * (belts + 1 + 1) = 504) :=
by
  rw [h_shirts, h_ties, h_pants, h_belts]
  sorry

end outfits_count_l1450_145006


namespace work_completion_days_l1450_145024

theorem work_completion_days (a b c : ℝ) :
  (1/a) = 1/90 → (1/b) = 1/45 → (1/a + 1/b + 1/c) = 1/5 → c = 6 :=
by
  intros ha hb habc
  sorry

end work_completion_days_l1450_145024


namespace allan_balloons_l1450_145029

def initial_balloons : ℕ := 5
def additional_balloons : ℕ := 3
def total_balloons : ℕ := initial_balloons + additional_balloons

theorem allan_balloons :
  total_balloons = 8 :=
sorry

end allan_balloons_l1450_145029


namespace download_time_l1450_145092

theorem download_time (avg_speed : ℤ) (size_A size_B size_C : ℤ) (gb_to_mb : ℤ) (secs_in_min : ℤ) :
  avg_speed = 30 →
  size_A = 450 →
  size_B = 240 →
  size_C = 120 →
  gb_to_mb = 1000 →
  secs_in_min = 60 →
  ( (size_A * gb_to_mb + size_B * gb_to_mb + size_C * gb_to_mb ) / avg_speed ) / secs_in_min = 450 := by
  intros h_avg h_A h_B h_C h_gb h_secs
  sorry

end download_time_l1450_145092


namespace smallest_number_of_coins_l1450_145041

theorem smallest_number_of_coins : ∃ (n : ℕ), 
  n ≡ 2 [MOD 5] ∧ 
  n ≡ 1 [MOD 4] ∧ 
  n ≡ 0 [MOD 3] ∧ 
  n = 57 := 
by
  sorry

end smallest_number_of_coins_l1450_145041


namespace sally_out_of_pocket_l1450_145095

-- Definitions based on conditions
def g : ℕ := 320 -- Amount given by the school
def c : ℕ := 12  -- Cost per book
def n : ℕ := 30  -- Number of students

-- Definition derived from conditions
def total_cost : ℕ := n * c
def out_of_pocket : ℕ := total_cost - g

-- Proof statement
theorem sally_out_of_pocket : out_of_pocket = 40 := by
  -- The proof steps would go here
  sorry

end sally_out_of_pocket_l1450_145095


namespace total_oranges_picked_l1450_145075

/-- Michaela needs 20 oranges to get full --/
def oranges_michaela_needs : ℕ := 20

/-- Cassandra needs twice as many oranges as Michaela to get full --/
def oranges_cassandra_needs : ℕ := 2 * oranges_michaela_needs

/-- After both have eaten until they are full, 30 oranges remain --/
def oranges_remaining : ℕ := 30

/-- The total number of oranges eaten by both Michaela and Cassandra --/
def oranges_eaten : ℕ := oranges_michaela_needs + oranges_cassandra_needs

/-- Prove that the total number of oranges picked from the farm is 90 --/
theorem total_oranges_picked : oranges_eaten + oranges_remaining = 90 := by
  sorry

end total_oranges_picked_l1450_145075


namespace find_a_l1450_145055

-- Conditions: x = 5 is a solution to the equation 2x - a = -5
-- We need to prove that a = 15 under these conditions

theorem find_a (x a : ℤ) (h1 : x = 5) (h2 : 2 * x - a = -5) : a = 15 :=
by
  -- We are required to prove the statement, so we skip the proof part here
  sorry

end find_a_l1450_145055


namespace cylindrical_plane_l1450_145040

open Set

-- Define a cylindrical coordinate point (r, θ, z)
structure CylindricalCoord where
  r : ℝ
  theta : ℝ
  z : ℝ

-- Condition 1: In cylindrical coordinates, z is the height
def height_in_cylindrical := λ coords : CylindricalCoord => coords.z 

-- Condition 2: z is constant c
variable (c : ℝ)

-- The theorem to be proven
theorem cylindrical_plane (c : ℝ) :
  {p : CylindricalCoord | p.z = c} = {q : CylindricalCoord | q.z = c} :=
by
  sorry

end cylindrical_plane_l1450_145040


namespace isosceles_triangle_l1450_145009

-- Given: sides a, b, c of a triangle satisfying a specific condition
-- To Prove: the triangle is isosceles (has at least two equal sides)

theorem isosceles_triangle (a b c : ℝ)
  (h : (c - b) / a + (a - c) / b + (b - a) / c = 0) :
  (a = b ∨ b = c ∨ a = c) :=
sorry

end isosceles_triangle_l1450_145009


namespace prob_select_math_books_l1450_145020

theorem prob_select_math_books :
  let total_books := 5
  let math_books := 3
  let total_ways_select_2 := Nat.choose total_books 2
  let ways_select_2_math := Nat.choose math_books 2
  let probability := (ways_select_2_math : ℚ) / total_ways_select_2
  probability = 3 / 10 :=
by
  sorry

end prob_select_math_books_l1450_145020


namespace tom_watches_movies_total_duration_l1450_145032

-- Define the running times for each movie
def M := 120
def A := M - 30
def B := A + 10
def D := 2 * B - 20

-- Define the number of times Tom watches each movie
def watch_B := 2
def watch_A := 3
def watch_M := 1
def watch_D := 4

-- Calculate the total time spent watching each movie
def total_time_B := watch_B * B
def total_time_A := watch_A * A
def total_time_M := watch_M * M
def total_time_D := watch_D * D

-- Calculate the total duration Tom spends watching these movies in a week
def total_duration := total_time_B + total_time_A + total_time_M + total_time_D

-- The statement to prove
theorem tom_watches_movies_total_duration :
  total_duration = 1310 := 
by
  sorry

end tom_watches_movies_total_duration_l1450_145032


namespace evaluate_expression_l1450_145046

theorem evaluate_expression : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end evaluate_expression_l1450_145046


namespace buratino_correct_l1450_145083

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def valid_nine_digit_number (n : ℕ) : Prop :=
  n >= 10^8 ∧ n < 10^9 ∧ (∀ i j : ℕ, i < 9 ∧ j < 9 ∧ i ≠ j → ((n / 10^i) % 10 ≠ (n / 10^j) % 10)) ∧
  (∀ i : ℕ, i < 9 → (n / 10^i) % 10 ≠ 7)

def can_form_prime (n : ℕ) : Prop :=
  ∃ m : ℕ, valid_nine_digit_number n ∧ (m < 1000 ∧ is_prime m ∧
   (∃ erase_indices : List ℕ, erase_indices.length = 6 ∧ 
    ∀ i : ℕ, i ∈ erase_indices → i < 9 ∧ 
    (n % 10^(9 - i)) / 10^(3 - i) = m))

theorem buratino_correct : 
  ∀ n : ℕ, valid_nine_digit_number n → ¬ can_form_prime n :=
by
  sorry

end buratino_correct_l1450_145083


namespace additional_time_proof_l1450_145074

-- Given the charging rate of the battery and the additional time required to reach a percentage
noncomputable def charging_rate := 20 / 60
noncomputable def initial_time := 60
noncomputable def additional_time := 150

-- Define the total time required to reach a certain percentage
noncomputable def total_time := initial_time + additional_time

-- The proof statement to verify the additional time required beyond the initial 60 minutes
theorem additional_time_proof : total_time - initial_time = additional_time := sorry

end additional_time_proof_l1450_145074


namespace bracelet_pairing_impossible_l1450_145077

/--
Elizabeth has 100 different bracelets, and each day she wears three of them to school. 
Prove that it is impossible for any pair of bracelets to appear together on her wrist exactly once.
-/
theorem bracelet_pairing_impossible : 
  (∃ (bracelet_set : Finset (Finset (Fin 100))), 
    (∀ (a b : Fin 100), a ≠ b → ∃ t ∈ bracelet_set, {a, b} ⊆ t) ∧ (∀ t ∈ bracelet_set, t.card = 3) ∧ (bracelet_set.card * 3 / 2 ≠ 99)) :=
sorry

end bracelet_pairing_impossible_l1450_145077


namespace ratio_second_to_first_l1450_145008

-- Define the given conditions and variables
variables 
  (total_water : ℕ := 1200)
  (neighborhood1_usage : ℕ := 150)
  (neighborhood4_usage : ℕ := 350)
  (x : ℕ) -- water usage by second neighborhood

-- Define the usage by third neighborhood in terms of the second neighborhood usage
def neighborhood3_usage := x + 100

-- Define remaining water usage after substracting neighborhood 4 usage
def remaining_water := total_water - neighborhood4_usage

-- The sum of water used by neighborhoods
def total_usage_neighborhoods := neighborhood1_usage + neighborhood3_usage x + x

theorem ratio_second_to_first (h : total_usage_neighborhoods x = remaining_water) :
  (x : ℚ) / neighborhood1_usage = 2 := 
by
  sorry

end ratio_second_to_first_l1450_145008


namespace shark_sightings_in_cape_may_l1450_145018

theorem shark_sightings_in_cape_may (x : ℕ) (hx : x + (2 * x - 8) = 40) : 2 * x - 8 = 24 := 
by 
  sorry

end shark_sightings_in_cape_may_l1450_145018


namespace fourth_competitor_jump_l1450_145049

theorem fourth_competitor_jump :
  let first_jump := 22
  let second_jump := first_jump + 1
  let third_jump := second_jump - 2
  let fourth_jump := third_jump + 3
  fourth_jump = 24 := by
  sorry

end fourth_competitor_jump_l1450_145049


namespace min_area_quadrilateral_l1450_145037

theorem min_area_quadrilateral
  (S_AOB S_COD : ℝ) (h₁ : S_AOB = 4) (h₂ : S_COD = 9) :
  ∃ S_BOC S_AOD, S_AOB + S_COD + S_BOC + S_AOD = 25 :=
by
  sorry

end min_area_quadrilateral_l1450_145037


namespace cross_section_area_l1450_145013

-- Definitions for the conditions stated in the problem
def frustum_height : ℝ := 6
def upper_base_side : ℝ := 4
def lower_base_side : ℝ := 8

-- The main statement to be proved
theorem cross_section_area :
  (exists (cross_section_area : ℝ),
    cross_section_area = 16 * Real.sqrt 6) :=
sorry

end cross_section_area_l1450_145013


namespace sufficient_but_not_necessary_l1450_145097

theorem sufficient_but_not_necessary (x y : ℝ) :
  (x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 4) ∧ ∃ x y : ℝ, x^2 + y^2 ≥ 4 ∧ ¬(x ≥ 2 ∧ y ≥ 2) :=
by
  sorry

end sufficient_but_not_necessary_l1450_145097


namespace length_of_FD_l1450_145039

theorem length_of_FD
  (ABCD_is_square : ∀ (A B C D : ℝ), A = 8 ∧ B = 8 ∧ C = 8 ∧ D = 8)
  (E_midpoint_AD : ∀ (A D E : ℝ), E = (A + D) / 2)
  (F_on_BD : ∀ (B D F E : ℝ), B = 8 ∧ F = 3 ∧ D = 8 ∧ E = 4):
  ∃ (FD : ℝ), FD = 3 := by
  sorry

end length_of_FD_l1450_145039


namespace collinear_points_sum_l1450_145099

theorem collinear_points_sum (x y : ℝ) : 
  (∃ a b : ℝ, a * x + b * 3 + (1 - a - b) * 2 = a * x + b * y + (1 - a - b) * y ∧ 
               a * y + b * 4 + (1 - a - b) * y = a * x + b * y + (1 - a - b) * x) → 
  x = 2 → y = 4 → x + y = 6 :=
by sorry

end collinear_points_sum_l1450_145099


namespace solve_x_from_equation_l1450_145066

theorem solve_x_from_equation :
  ∀ (x : ℝ), x ≠ 0 ∧ (3 * x)^5 = (9 * x)^4 → x = 27 :=
by
  intro x
  rintro ⟨hx, h⟩
  sorry

end solve_x_from_equation_l1450_145066


namespace total_volume_of_pyramids_l1450_145056

theorem total_volume_of_pyramids :
  let base := 40
  let height_base := 20
  let height_pyramid := 30
  let area_base := (1 / 2) * base * height_base
  let volume_pyramid := (1 / 3) * area_base * height_pyramid
  3 * volume_pyramid = 12000 :=
by 
  sorry

end total_volume_of_pyramids_l1450_145056


namespace min_turns_for_route_l1450_145028

-- Define the number of parallel and intersecting streets
def num_parallel_streets := 10
def num_intersecting_streets := 10

-- Define the grid as a product of these two numbers
def num_intersections := num_parallel_streets * num_intersecting_streets

-- Define the minimum number of turns necessary for a closed bus route passing through all intersections
def min_turns (grid_size : Nat) : Nat :=
  if grid_size = num_intersections then 20 else 0

-- The main theorem statement
theorem min_turns_for_route : min_turns num_intersections = 20 :=
  sorry

end min_turns_for_route_l1450_145028


namespace minimum_value_expression_l1450_145051

theorem minimum_value_expression (x : ℝ) (h : -3 < x ∧ x < 2) :
  ∃ y, y = (x^2 + 4 * x + 5) / (2 * x + 6) ∧ y = 3 / 4 :=
by
  sorry

end minimum_value_expression_l1450_145051


namespace roots_cubic_identity_l1450_145093

theorem roots_cubic_identity (r s : ℚ) (h1 : 3 * r^2 + 5 * r + 2 = 0) (h2 : 3 * s^2 + 5 * s + 2 = 0) :
  (1 / r^3) + (1 / s^3) = -27 / 35 :=
sorry

end roots_cubic_identity_l1450_145093


namespace sum_of_remainders_mod_53_l1450_145072

theorem sum_of_remainders_mod_53 (x y z : ℕ) (h1 : x % 53 = 31) (h2 : y % 53 = 17) (h3 : z % 53 = 8) : 
  (x + y + z) % 53 = 3 :=
by {
  sorry
}

end sum_of_remainders_mod_53_l1450_145072


namespace radian_measure_of_central_angle_l1450_145065

-- Given conditions
variables (l r : ℝ)
variables (h1 : (1 / 2) * l * r = 1)
variables (h2 : 2 * r + l = 4)

-- The theorem to prove
theorem radian_measure_of_central_angle (l r : ℝ) (h1 : (1 / 2) * l * r = 1) (h2 : 2 * r + l = 4) : 
  l / r = 2 :=
by 
  -- Proof steps are not provided as per the requirement
  sorry

end radian_measure_of_central_angle_l1450_145065


namespace find_a5_l1450_145061

open Nat

def increasing_seq (a : Nat → Nat) : Prop :=
  ∀ m n : Nat, m < n → a m < a n

theorem find_a5
  (a : Nat → Nat)
  (h1 : ∀ n : Nat, a (a n) = 3 * n)
  (h2 : increasing_seq a)
  (h3 : ∀ n : Nat, a n > 0) :
  a 5 = 8 :=
by
  sorry

end find_a5_l1450_145061


namespace probability_of_two_red_shoes_is_0_1332_l1450_145053

def num_red_shoes : ℕ := 4
def num_green_shoes : ℕ := 6
def total_shoes : ℕ := num_red_shoes + num_green_shoes

def probability_first_red_shoe : ℚ := num_red_shoes / total_shoes
def remaining_red_shoes_after_first_draw : ℕ := num_red_shoes - 1
def remaining_shoes_after_first_draw : ℕ := total_shoes - 1
def probability_second_red_shoe : ℚ := remaining_red_shoes_after_first_draw / remaining_shoes_after_first_draw

def probability_two_red_shoes : ℚ := probability_first_red_shoe * probability_second_red_shoe

theorem probability_of_two_red_shoes_is_0_1332 : probability_two_red_shoes = 1332 / 10000 :=
by
  sorry

end probability_of_two_red_shoes_is_0_1332_l1450_145053


namespace find_m_value_l1450_145010

theorem find_m_value (m a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ)
  (h1 : (x + m)^9 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + a_3 * (x + 1)^3 + 
  a_4 * (x + 1)^4 + a_5 * (x + 1)^5 + a_6 * (x + 1)^6 + a_7 * (x + 1)^7 + 
  a_8 * (x + 1)^8 + a_9 * (x + 1)^9)
  (h2 : a_0 - a_1 + a_2 - a_3 + a_4 - a_5 + a_6 - a_7 + a_8 - a_9 = 3^9) :
  m = 4 :=
by
  sorry

end find_m_value_l1450_145010


namespace div_expr_l1450_145058

namespace Proof

theorem div_expr (x : ℝ) (h : x = 3.242 * 10) : x / 100 = 0.3242 := by
  sorry

end Proof

end div_expr_l1450_145058


namespace interest_rate_supposed_to_be_invested_l1450_145060

variable (P T : ℕ) (additional_interest interest_rate_15 interest_rate_R : ℚ)

def simple_interest (principal: ℚ) (time: ℚ) (rate: ℚ) : ℚ := (principal * time * rate) / 100

theorem interest_rate_supposed_to_be_invested :
  P = 15000 → T = 2 → additional_interest = 900 → interest_rate_15 = 15 →
  simple_interest P T interest_rate_15 = simple_interest P T interest_rate_R + additional_interest →
  interest_rate_R = 12 := by
  intros hP hT h_add h15 h_interest
  simp [simple_interest] at *
  sorry

end interest_rate_supposed_to_be_invested_l1450_145060


namespace prove_f_2_eq_10_prove_f_4_eq_10_prove_f_prove_f_l1450_145033

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom f_cond1 : ∀ x : ℝ, f x + (deriv g x) = 10
axiom f_cond2 : ∀ x : ℝ, f x - (deriv g (4 - x)) = 10
axiom g_even : ∀ x : ℝ, g x = g (-x)

theorem prove_f_2_eq_10 : f 2 = 10 := sorry
theorem prove_f_4_eq_10 : f 4 = 10 := sorry
theorem prove_f'_neg1_eq_f'_neg3 : deriv f (-1) = deriv f (-3) := sorry
theorem prove_f'_2023_ne_0 : deriv f 2023 ≠ 0 := sorry

end prove_f_2_eq_10_prove_f_4_eq_10_prove_f_prove_f_l1450_145033


namespace positive_integer_solution_l1450_145067

/-- Given that x, y, and t are all equal to 1, and x + y + z + t = 10, we need to prove that z = 7. -/
theorem positive_integer_solution {x y z t : ℕ} (hx : x = 1) (hy : y = 1) (ht : t = 1) (h : x + y + z + t = 10) : z = 7 :=
by {
  -- We would provide the proof here, but for now, we use sorry
  sorry
}

end positive_integer_solution_l1450_145067


namespace total_cakes_served_l1450_145027

def weekday_cakes_lunch : Nat := 6 + 8 + 10
def weekday_cakes_dinner : Nat := 9 + 7 + 5 + 13
def weekday_cakes_total : Nat := weekday_cakes_lunch + weekday_cakes_dinner

def weekend_cakes_lunch : Nat := 2 * (6 + 8 + 10)
def weekend_cakes_dinner : Nat := 2 * (9 + 7 + 5 + 13)
def weekend_cakes_total : Nat := weekend_cakes_lunch + weekend_cakes_dinner

def total_weekday_cakes : Nat := 5 * weekday_cakes_total
def total_weekend_cakes : Nat := 2 * weekend_cakes_total

def total_week_cakes : Nat := total_weekday_cakes + total_weekend_cakes

theorem total_cakes_served : total_week_cakes = 522 := by
  sorry

end total_cakes_served_l1450_145027


namespace div_problem_l1450_145017

theorem div_problem : 150 / (6 / 3) = 75 := by
  sorry

end div_problem_l1450_145017


namespace gcd_204_85_l1450_145000

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := 
by sorry

end gcd_204_85_l1450_145000


namespace find_years_l1450_145084

variable (p m x : ℕ)

def two_years_ago := p - 2 = 2 * (m - 2)
def four_years_ago := p - 4 = 3 * (m - 4)
def ratio_in_x_years (x : ℕ) := (p + x) * 2 = (m + x) * 3

theorem find_years (h1 : two_years_ago p m) (h2 : four_years_ago p m) : ratio_in_x_years p m 2 :=
by
  sorry

end find_years_l1450_145084


namespace determine_f_2014_l1450_145098

open Function

noncomputable def f : ℕ → ℕ :=
  sorry

theorem determine_f_2014
  (h1 : f 2 = 0)
  (h2 : f 3 > 0)
  (h3 : f 6042 = 2014)
  (h4 : ∀ m n : ℕ, f (m + n) - f m - f n ∈ ({0, 1} : Set ℕ)) :
  f 2014 = 671 :=
sorry

end determine_f_2014_l1450_145098


namespace number_of_frames_bought_l1450_145078

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

end number_of_frames_bought_l1450_145078


namespace find_x4_l1450_145044

theorem find_x4 (x_1 x_2 : ℝ) (h1 : 0 < x_1) (h2 : x_1 < x_2) 
  (P : (ℝ × ℝ)) (Q : (ℝ × ℝ)) (hP : P = (2, Real.log 2)) 
  (hQ : Q = (500, Real.log 500)) 
  (R : (ℝ × ℝ)) (x_4 : ℝ) :
  R = ((x_1 + x_2) / 2, (Real.log x_1 + Real.log x_2) / 2) →
  Real.log x_4 = (Real.log x_1 + Real.log x_2) / 2 →
  x_4 = Real.sqrt 1000 :=
by 
  intro hR hT
  sorry

end find_x4_l1450_145044


namespace extreme_values_range_of_a_l1450_145030

noncomputable def f (x : ℝ) := x^2 * Real.exp x
noncomputable def y (x : ℝ) (a : ℝ) := f x - a * x

theorem extreme_values :
  ∃ x_max x_min,
    (x_max = -2 ∧ f x_max = 4 / Real.exp 2) ∧
    (x_min = 0 ∧ f x_min = 0) := sorry

theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂, x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ y x₁ a = 0 ∧ y x₂ a = 0) ↔
  -1 / Real.exp 1 < a ∧ a < 0 := sorry

end extreme_values_range_of_a_l1450_145030


namespace max_value_of_g_l1450_145001

def g (n : ℕ) : ℕ :=
  if n < 15 then n + 15 else g (n - 7)

theorem max_value_of_g : ∃ m, ∀ n, g n ≤ m ∧ (∃ k, g k = m) :=
by
  use 29
  sorry

end max_value_of_g_l1450_145001


namespace solve_for_k_l1450_145021

def sameLine (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

theorem solve_for_k :
  (sameLine (3, 10) (1, k) (-7, 2)) → k = 8.4 :=
by
  sorry

end solve_for_k_l1450_145021


namespace third_side_length_is_six_l1450_145023

-- Defining the lengths of the sides of the triangle
def side1 : ℕ := 2
def side2 : ℕ := 6

-- Defining that the third side is an even number between 4 and 8
def is_even (x : ℕ) : Prop := x % 2 = 0
def valid_range (x : ℕ) : Prop := 4 < x ∧ x < 8

-- Stating the theorem
theorem third_side_length_is_six (x : ℕ) (h1 : is_even x) (h2 : valid_range x) : x = 6 :=
by
  sorry

end third_side_length_is_six_l1450_145023


namespace number_of_m_gons_proof_l1450_145088

noncomputable def number_of_m_gons_with_two_acute_angles (m n : ℕ) (h1 : 4 < m) (h2 : m < n) : ℕ :=
  (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1))

theorem number_of_m_gons_proof {m n : ℕ} (h1 : 4 < m) (h2 : m < n) :
  number_of_m_gons_with_two_acute_angles m n h1 h2 =
  (2 * n + 1) * ((Nat.choose (n + 1) (m - 1)) + (Nat.choose n (m - 1))) :=
sorry

end number_of_m_gons_proof_l1450_145088


namespace infinite_points_inside_circle_l1450_145063

theorem infinite_points_inside_circle:
  ∀ c : ℝ, c = 3 → ∀ x y : ℚ, 0 < x ∧ 0 < y  ∧ x^2 + y^2 < 9 → ∃ a b : ℚ, 0 < a ∧ 0 < b ∧ a^2 + b^2 < 9 :=
sorry

end infinite_points_inside_circle_l1450_145063


namespace football_cost_l1450_145080

theorem football_cost (cost_shorts cost_shoes money_have money_need : ℝ)
  (h_shorts : cost_shorts = 2.40)
  (h_shoes : cost_shoes = 11.85)
  (h_have : money_have = 10)
  (h_need : money_need = 8) :
  (money_have + money_need - (cost_shorts + cost_shoes) = 3.75) :=
by
  -- Proof goes here
  sorry

end football_cost_l1450_145080


namespace find_S_l1450_145047

variable (R S T c : ℝ)
variable (h1 : R = c * (S^2 / T^2))
variable (c_value : c = 8)
variable (h2 : R = 2) (h3 : T = 2) (h4 : S = 1)
variable (R_new : R = 50) (T_new : T = 5)

theorem find_S : S = 12.5 := by
  sorry

end find_S_l1450_145047


namespace find_remainder_l1450_145048

-- Definition of N based on given conditions
def N : ℕ := 44 * 432

-- Definition of next multiple of 432
def next_multiple_of_432 : ℕ := N + 432

-- Statement to prove the remainder when next_multiple_of_432 is divided by 39 is 12
theorem find_remainder : next_multiple_of_432 % 39 = 12 := 
by sorry

end find_remainder_l1450_145048


namespace min_value_expr_l1450_145045

theorem min_value_expr (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_sum : a + b = 1) : 
  (b / (3 * a)) + (3 / b) ≥ 5 := 
sorry

end min_value_expr_l1450_145045


namespace proof_P_l1450_145082

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the complement of P in U
def CU_P : Set ℕ := {4, 5}

-- Define the set P as the difference between U and CU_P
def P : Set ℕ := U \ CU_P

-- Prove that P = {1, 2, 3}
theorem proof_P :
  P = {1, 2, 3} :=
by
  sorry

end proof_P_l1450_145082


namespace min_value_ratio_l1450_145042

variable {α : Type*} [LinearOrderedField α]

theorem min_value_ratio (a : ℕ → α) (h1 : a 7 = a 6 + 2 * a 5) (h2 : ∃ m n : ℕ, a m * a n = 8 * a 1^2) :
  ∃ m n : ℕ, (1 / m + 4 / n = 11 / 6) :=
by
  sorry

end min_value_ratio_l1450_145042


namespace solution_set_of_inequality_l1450_145068

theorem solution_set_of_inequality
  (a b : ℝ)
  (h1 : a < 0) 
  (h2 : b / a = 1) :
  { x : ℝ | (x - 1) * (a * x + b) < 0 } = { x : ℝ | x < -1 } ∪ {x : ℝ | 1 < x} :=
by
  sorry

end solution_set_of_inequality_l1450_145068


namespace final_statement_l1450_145086

variable (f : ℝ → ℝ)

-- Conditions
axiom even_function : ∀ x, f (x) = f (-x)
axiom periodic_minus_one : ∀ x, f (x + 1) = -f (x)
axiom increasing_on_neg_one_to_zero : ∀ x y, -1 ≤ x ∧ x < y ∧ y ≤ 0 → f (x) < f (y)

-- Statement
theorem final_statement :
  (∀ x, f (x + 2) = f (x)) ∧
  (¬ (∀ x, 0 ≤ x ∧ x ≤ 1 → f (x) < f (x + 1))) ∧
  (∀ x y, 1 ≤ x ∧ x < y ∧ y ≤ 2 → f (x) < f (y)) ∧
  (f (2) = f (0)) :=
by
  sorry

end final_statement_l1450_145086


namespace at_least_one_not_less_than_two_l1450_145062

theorem at_least_one_not_less_than_two
  (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  let a := x + 1 / y
  let b := y + 1 / z
  let c := z + 1 / x
  a >= 2 ∨ b >= 2 ∨ c >= 2 := 
sorry

end at_least_one_not_less_than_two_l1450_145062


namespace meena_work_days_l1450_145085

theorem meena_work_days (M : ℝ) : 1/5 + 1/M = 3/10 → M = 10 :=
by
  sorry

end meena_work_days_l1450_145085


namespace housewife_more_oil_l1450_145014

theorem housewife_more_oil 
    (reduction_percent : ℝ := 10)
    (reduced_price : ℝ := 16)
    (budget : ℝ := 800)
    (approx_answer : ℝ := 5.01) :
    let P := reduced_price / (1 - reduction_percent / 100)
    let Q_original := budget / P
    let Q_reduced := budget / reduced_price
    let delta_Q := Q_reduced - Q_original
    abs (delta_Q - approx_answer) < 0.02 := 
by
  -- Let the goal be irrelevant to the proof because the proof isn't provided
  sorry

end housewife_more_oil_l1450_145014


namespace box_dimensions_sum_l1450_145012

theorem box_dimensions_sum (A B C : ℝ)
  (h1 : A * B = 18)
  (h2 : A * C = 32)
  (h3 : B * C = 50) :
  A + B + C = 57.28 := 
sorry

end box_dimensions_sum_l1450_145012


namespace determine_a_l1450_145073

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then 2 * x + a else -x - 2 * a

theorem determine_a (a : ℝ) (h : a ≠ 0) (h_eq : f a (1 - a) = f a (1 + a)) : a = -3/4 :=
by
  sorry

end determine_a_l1450_145073


namespace gray_region_area_l1450_145015

noncomputable def area_of_gray_region (C_center D_center : ℝ × ℝ) (C_radius D_radius : ℝ) :=
  let rect_area := 35
  let semicircle_C_area := (25 * Real.pi) / 2
  let quarter_circle_D_area := (16 * Real.pi) / 4
  rect_area - semicircle_C_area - quarter_circle_D_area

theorem gray_region_area :
  area_of_gray_region (5, 5) (12, 5) 5 4 = 35 - 16.5 * Real.pi :=
by
  simp [area_of_gray_region]
  sorry

end gray_region_area_l1450_145015


namespace inscribed_rectangle_l1450_145043

theorem inscribed_rectangle (b h : ℝ) : ∃ x : ℝ, 
  (∃ q : ℝ, x = q / 2) → 
  ∃ x : ℝ, 
    (∃ q : ℝ, q = 2 * x ∧ x = h * q / (2 * h + b)) :=
sorry

end inscribed_rectangle_l1450_145043


namespace Xiao_Ming_max_notebooks_l1450_145035

-- Definitions of the given conditions
def total_yuan : ℝ := 30
def total_books : ℕ := 30
def notebook_cost : ℝ := 4
def exercise_book_cost : ℝ := 0.4

-- Definition of the variables used in the inequality
def x (max_notebooks : ℕ) : ℝ := max_notebooks
def exercise_books (max_notebooks : ℕ) : ℝ := total_books - x max_notebooks

-- Definition of the total cost inequality
def total_cost (max_notebooks : ℕ) : ℝ :=
  x max_notebooks * notebook_cost + exercise_books max_notebooks * exercise_book_cost

theorem Xiao_Ming_max_notebooks (max_notebooks : ℕ) : total_cost max_notebooks ≤ total_yuan → max_notebooks ≤ 5 :=
by
  -- Proof goes here
  sorry

end Xiao_Ming_max_notebooks_l1450_145035


namespace shelby_gold_stars_l1450_145071

theorem shelby_gold_stars (stars_yesterday stars_today : ℕ) (h1 : stars_yesterday = 4) (h2 : stars_today = 3) :
  stars_yesterday + stars_today = 7 := 
by
  sorry

end shelby_gold_stars_l1450_145071


namespace chessboard_disk_cover_l1450_145054

noncomputable def chessboardCoveredSquares : ℕ :=
  let D : ℝ := 1 -- assuming D is a positive real number; actual value irrelevant as it gets cancelled in the comparison
  let grid_size : ℕ := 8
  let total_squares : ℕ := grid_size * grid_size
  let boundary_squares : ℕ := 28 -- pre-calculated in the insides steps
  let interior_squares : ℕ := total_squares - boundary_squares
  let non_covered_corners : ℕ := 4
  interior_squares - non_covered_corners

theorem chessboard_disk_cover : chessboardCoveredSquares = 32 := sorry

end chessboard_disk_cover_l1450_145054


namespace max_ballpoint_pens_l1450_145064

theorem max_ballpoint_pens (x y z : ℕ) (hx : x + y + z = 15)
  (hy : 10 * x + 40 * y + 60 * z = 500) (hz : x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1) :
  x ≤ 6 :=
sorry

end max_ballpoint_pens_l1450_145064


namespace john_school_year_hours_l1450_145007

theorem john_school_year_hours (summer_earnings : ℝ) (summer_hours_per_week : ℝ) (summer_weeks : ℝ) (target_school_earnings : ℝ) (school_weeks : ℝ) :
  summer_earnings = 4000 → summer_hours_per_week = 40 → summer_weeks = 8 → target_school_earnings = 5000 → school_weeks = 25 →
  (target_school_earnings / (summer_earnings / (summer_hours_per_week * summer_weeks)) / school_weeks) = 16 :=
by
  sorry

end john_school_year_hours_l1450_145007


namespace integral_eq_exp_integral_eq_one_l1450_145004

noncomputable
def y1 (τ : ℝ) (t : ℝ) (y : ℝ → ℝ) : Prop :=
  y τ = ∫ x in (0 : ℝ)..t, y x + 1

theorem integral_eq_exp (y : ℝ → ℝ) : 
  (∀ τ t, y1 τ t y) ↔ (∀ t, y t = Real.exp t) := 
  sorry

noncomputable
def y2 (t : ℝ) (y : ℝ → ℝ) : Prop :=
  ∫ x in (0 : ℝ)..t, y x * Real.sin (t - x) = 1 - Real.cos t

theorem integral_eq_one (y : ℝ → ℝ) : 
  (∀ t, y2 t y) ↔ (∀ t, y t = 1) :=
  sorry

end integral_eq_exp_integral_eq_one_l1450_145004


namespace price_per_acre_is_1863_l1450_145034

-- Define the conditions
def totalAcres : ℕ := 4
def numLots : ℕ := 9
def pricePerLot : ℤ := 828
def totalRevenue : ℤ := numLots * pricePerLot
def totalCost (P : ℤ) : ℤ := totalAcres * P

-- The proof problem: Prove that the price per acre P is 1863
theorem price_per_acre_is_1863 (P : ℤ) (h : totalCost P = totalRevenue) : P = 1863 :=
by
  sorry

end price_per_acre_is_1863_l1450_145034


namespace min_value_a_l1450_145069

theorem min_value_a (a : ℝ) : (∀ x : ℝ, a < x → 2 * x + 2 / (x - a) ≥ 7) → a ≥ 3 / 2 :=
by
  sorry

end min_value_a_l1450_145069


namespace seating_arrangement_l1450_145096

def valid_arrangements := 6

def Alice_refusal (A B C : Prop) := (¬ (A ∧ B)) ∧ (¬ (A ∧ C))
def Derek_refusal (D E C : Prop) := (¬ (D ∧ E)) ∧ (¬ (D ∧ C))

theorem seating_arrangement (A B C D E : Prop) : 
  Alice_refusal A B C ∧ Derek_refusal D E C → valid_arrangements = 6 := 
  sorry

end seating_arrangement_l1450_145096


namespace find_b1_over_b2_l1450_145057

variable {a b k a1 a2 b1 b2 : ℝ}

-- Assuming a is inversely proportional to b
def inversely_proportional (a b : ℝ) (k : ℝ) : Prop :=
  a * b = k

-- Define that a_1 and a_2 are nonzero and their ratio is 3/4
def a1_a2_ratio (a1 a2 : ℝ) (ratio : ℝ) : Prop :=
  a1 / a2 = ratio

-- Define that b_1 and b_2 are nonzero
def nonzero (x : ℝ) : Prop :=
  x ≠ 0

theorem find_b1_over_b2 (a1 a2 b1 b2 : ℝ) (h1 : inversely_proportional a b k)
  (h2 : a1_a2_ratio a1 a2 (3 / 4))
  (h3 : nonzero a1) (h4 : nonzero a2) (h5 : nonzero b1) (h6 : nonzero b2) :
  b1 / b2 = 4 / 3 := 
sorry

end find_b1_over_b2_l1450_145057


namespace empty_can_mass_l1450_145079

-- Define the mass of the full can
def full_can_mass : ℕ := 35

-- Define the mass of the can with half the milk
def half_can_mass : ℕ := 18

-- The theorem stating the mass of the empty can
theorem empty_can_mass : full_can_mass - (2 * (full_can_mass - half_can_mass)) = 1 := by
  sorry

end empty_can_mass_l1450_145079


namespace min_value_z_l1450_145031

theorem min_value_z (x y z : ℤ) (h1 : x + y + z = 100) (h2 : x < y) (h3 : y < 2 * z) : z ≥ 21 :=
sorry

end min_value_z_l1450_145031


namespace parabola_directrix_l1450_145059

theorem parabola_directrix (x y : ℝ) (h : y = x^2) : 4 * y + 1 = 0 := 
sorry

end parabola_directrix_l1450_145059


namespace lisa_phone_spending_l1450_145081

variable (cost_phone : ℕ) (cost_contract_per_month : ℕ) (case_percentage : ℕ) (headphones_ratio : ℕ)

/-- Given the cost of the phone, the monthly contract cost, 
    the percentage cost of the case, and ratio cost of headphones,
    prove that the total spending in the first year is correct.
-/ 
theorem lisa_phone_spending 
    (h_cost_phone : cost_phone = 1000) 
    (h_cost_contract_per_month : cost_contract_per_month = 200) 
    (h_case_percentage : case_percentage = 20)
    (h_headphones_ratio : headphones_ratio = 2) :
    cost_phone + (cost_phone * case_percentage / 100) + 
    ((cost_phone * case_percentage / 100) / headphones_ratio) + 
    (cost_contract_per_month * 12) = 3700 :=
by
  sorry

end lisa_phone_spending_l1450_145081


namespace smallest_angle_in_triangle_l1450_145087

theorem smallest_angle_in_triangle (a b c x : ℝ) 
  (h1 : a + b + c = 180)
  (h2 : a = 5 * x)
  (h3 : b = 3 * x) :
  x = 20 :=
by
  sorry

end smallest_angle_in_triangle_l1450_145087


namespace slope_of_tangent_at_4_l1450_145076

def f (x : ℝ) : ℝ := x^3 - 7 * x^2 + 1

theorem slope_of_tangent_at_4 : (deriv f 4) = -8 := by
  sorry

end slope_of_tangent_at_4_l1450_145076


namespace lily_pads_cover_entire_lake_l1450_145038

/-- 
If a patch of lily pads doubles in size every day and takes 57 days to cover half the lake,
then it will take 58 days to cover the entire lake.
-/
theorem lily_pads_cover_entire_lake (days_to_half : ℕ) (h : days_to_half = 57) : (days_to_half + 1 = 58) := by
  sorry

end lily_pads_cover_entire_lake_l1450_145038


namespace calculate_expression_l1450_145019

theorem calculate_expression :
  |(-1 : ℝ)| + Real.sqrt 9 - (1 - Real.sqrt 3)^0 - (1/2)^(-1 : ℝ) = 1 :=
by
  sorry

end calculate_expression_l1450_145019


namespace y_work_time_l1450_145026

noncomputable def total_work := 1 

noncomputable def work_rate_x := 1 / 40
noncomputable def work_x_in_8_days := 8 * work_rate_x
noncomputable def remaining_work := total_work - work_x_in_8_days

noncomputable def work_rate_y := remaining_work / 36

theorem y_work_time :
  (1 / work_rate_y) = 45 :=
by
  sorry

end y_work_time_l1450_145026


namespace value_of_f_8_minus_f_4_l1450_145011

noncomputable def f : ℝ → ℝ := sorry

theorem value_of_f_8_minus_f_4 :
  -- Conditions
  (∀ x, f (-x) = -f x) ∧              -- odd function
  (∀ x, f (x + 5) = f x) ∧            -- period of 5
  (f 1 = 1) ∧                         -- f(1) = 1
  (f 2 = 3) →                         -- f(2) = 3
  -- Goal
  f 8 - f 4 = -2 :=
sorry

end value_of_f_8_minus_f_4_l1450_145011


namespace country_albums_count_l1450_145089

-- Definitions based on conditions
def pop_albums : Nat := 8
def songs_per_album : Nat := 7
def total_songs : Nat := 70

-- Theorem to prove the number of country albums
theorem country_albums_count : (total_songs - pop_albums * songs_per_album) / songs_per_album = 2 := by
  sorry

end country_albums_count_l1450_145089


namespace rational_root_k_values_l1450_145052

theorem rational_root_k_values (k : ℤ) :
  (∃ x : ℚ, x^2017 - x^2016 + x^2 + k * x + 1 = 0) ↔ (k = 0 ∨ k = -2) :=
by
  sorry

end rational_root_k_values_l1450_145052


namespace remainder_2027_div_28_l1450_145016

theorem remainder_2027_div_28 : 2027 % 28 = 3 :=
by
  sorry

end remainder_2027_div_28_l1450_145016


namespace percentage_reduction_correct_l1450_145094

-- Define the initial conditions
def initial_conditions (P S : ℝ) (new_sales_increase_percentage net_sale_value_increase_percentage: ℝ) :=
  new_sales_increase_percentage = 0.72 ∧ net_sale_value_increase_percentage = 0.4104

-- Define the statement for the required percentage reduction
theorem percentage_reduction_correct (P S : ℝ) (x : ℝ) 
  (h : initial_conditions P S 0.72 0.4104) : 
  (S:ℝ) * (1 - x / 100) = 1.4104 * S := 
sorry

end percentage_reduction_correct_l1450_145094


namespace find_m_l1450_145091

theorem find_m (m : ℝ) (x : ℝ) (h : 2*x + m = 1) (hx : x = -1) : m = 3 := 
by
  rw [hx] at h
  linarith

end find_m_l1450_145091


namespace circle_C2_equation_line_l_equation_l1450_145050

-- Proof problem 1: Finding the equation of C2
theorem circle_C2_equation (C1_center_x C1_center_y : ℝ) (A_x A_y : ℝ) 
  (C2_center_x : ℝ) (C1_radius : ℝ) :
  C1_center_x = 6 ∧ C1_center_y = 7 ∧ C1_radius = 5 →
  A_x = 2 ∧ A_y = 4 →
  C2_center_x = 6 →
  (∀ y : ℝ, ((y - C1_center_y = C1_radius + (C1_radius + (y - C1_center_y)))) →
    (x - C2_center_x)^2 + (y - C2_center_y)^2 = 1) :=
sorry

-- Proof problem 2: Finding the equation of the line l
theorem line_l_equation (O_x O_y A_x A_y : ℝ) 
  (C1_center_x C1_center_y : ℝ) 
  (A_BC_dist : ℝ) :
  O_x = 0 ∧ O_y = 0 →
  A_x = 2 ∧ A_y = 4 →
  C1_center_x = 6 ∧ C1_center_y = 7 →
  A_BC_dist = 2 * (25^(1 / 2)) →
  ((2 : ℝ)*x - y + 5 = 0 ∨ (2 : ℝ)*x - y - 15 = 0) :=
sorry

end circle_C2_equation_line_l_equation_l1450_145050


namespace solve_inequality_l1450_145036

theorem solve_inequality :
  ∀ x : ℝ, (3 * x^2 - 4 * x - 7 < 0) ↔ (-1 < x ∧ x < 7 / 3) :=
by
  sorry

end solve_inequality_l1450_145036


namespace john_weekly_earnings_increase_l1450_145003

theorem john_weekly_earnings_increase (original_earnings new_earnings : ℕ) 
  (h₀ : original_earnings = 60) 
  (h₁ : new_earnings = 72) : 
  ((new_earnings - original_earnings) / original_earnings) * 100 = 20 :=
by
  sorry

end john_weekly_earnings_increase_l1450_145003


namespace boards_per_package_calculation_l1450_145002

-- Defining the conditions
def total_boards : ℕ := 154
def num_packages : ℕ := 52

-- Defining the division of total_boards by num_packages within rationals
def boards_per_package : ℚ := total_boards / num_packages

-- Prove that the boards per package is mathematically equal to the total boards divided by the number of packages
theorem boards_per_package_calculation :
  boards_per_package = 154 / 52 := by
  sorry

end boards_per_package_calculation_l1450_145002


namespace find_a_l1450_145090

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - Real.exp (1 - x) - a * x
noncomputable def g (a x : ℝ) : ℝ := Real.exp x + Real.exp (1 - x) - a

theorem find_a (x₁ x₂ a : ℝ) (h₁ : g a x₁ = 0) (h₂ : g a x₂ = 0) (hf : f a x₁ + f a x₂ = -4) : a = 4 :=
sorry

end find_a_l1450_145090


namespace berengere_contribution_l1450_145070

noncomputable def exchange_rate : ℝ := (1.5 : ℝ)
noncomputable def pastry_cost_euros : ℝ := (8 : ℝ)
noncomputable def lucas_money_cad : ℝ := (10 : ℝ)
noncomputable def lucas_money_euros : ℝ := lucas_money_cad / exchange_rate

theorem berengere_contribution :
  pastry_cost_euros - lucas_money_euros = (4 / 3 : ℝ) :=
by
  sorry

end berengere_contribution_l1450_145070


namespace rowing_distance_l1450_145005

theorem rowing_distance
  (v_still : ℝ) (v_current : ℝ) (time : ℝ)
  (h1 : v_still = 15) (h2 : v_current = 3) (h3 : time = 17.998560115190784) :
  (v_still + v_current) * 1000 / 3600 * time = 89.99280057595392 :=
by
  rw [h1, h2, h3] -- Apply the given conditions
  -- This will reduce to proving (15 + 3) * 1000 / 3600 * 17.998560115190784 = 89.99280057595392
  sorry

end rowing_distance_l1450_145005
