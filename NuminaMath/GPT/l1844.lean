import Mathlib

namespace find_r_and_s_l1844_184433

theorem find_r_and_s (r s : ℝ) :
  (∀ m : ℝ, ¬(∃ x : ℝ, x^2 + 10 * x = m * (x - 10) + 5) ↔ r < m ∧ m < s) →
  r + s = 60 :=
sorry

end find_r_and_s_l1844_184433


namespace calculate_expression_l1844_184405

theorem calculate_expression : 5 * 7 + 10 * 4 - 36 / 3 + 6 * 3 = 81 :=
by
  -- Proof steps would be included here if they were needed, but the proof is left as sorry for now.
  sorry

end calculate_expression_l1844_184405


namespace every_positive_integer_displayable_l1844_184479

-- Definitions based on the conditions of the problem
def flip_switch_up (n : ℕ) : ℕ := n + 1
def flip_switch_down (n : ℕ) : ℕ := n - 1
def press_red_button (n : ℕ) : ℕ := n * 3
def press_yellow_button (n : ℕ) : ℕ := if n % 3 = 0 then n / 3 else n
def press_green_button (n : ℕ) : ℕ := n * 5
def press_blue_button (n : ℕ) : ℕ := if n % 5 = 0 then n / 5 else n

-- Prove that every positive integer can appear on the calculator display
theorem every_positive_integer_displayable : ∀ n : ℕ, n > 0 → 
  ∃ m : ℕ, m = n ∧
    (m = flip_switch_up m ∨ m = flip_switch_down m ∨ 
     m = press_red_button m ∨ m = press_yellow_button m ∨ 
     m = press_green_button m ∨ m = press_blue_button m) := 
sorry

end every_positive_integer_displayable_l1844_184479


namespace find_simple_interest_rate_l1844_184489

theorem find_simple_interest_rate (P A T SI R : ℝ)
  (hP : P = 750)
  (hA : A = 1125)
  (hT : T = 5)
  (hSI : SI = A - P)
  (hSI_def : SI = (P * R * T) / 100) : R = 10 :=
by
  -- Proof would go here
  sorry

end find_simple_interest_rate_l1844_184489


namespace distance_to_valley_l1844_184441

theorem distance_to_valley (car_speed_kph : ℕ) (time_seconds : ℕ) (sound_speed_mps : ℕ) 
  (car_speed_mps : ℕ) (distance_by_car : ℕ) (distance_by_sound : ℕ) 
  (total_distance_equation : 2 * x + distance_by_car = distance_by_sound) : x = 640 :=
by
  have car_speed_kph := 72
  have time_seconds := 4
  have sound_speed_mps := 340
  have car_speed_mps := car_speed_kph * 1000 / 3600
  have distance_by_car := time_seconds * car_speed_mps
  have distance_by_sound := time_seconds * sound_speed_mps
  have total_distance_equation := (2 * x + distance_by_car = distance_by_sound)
  exact sorry

end distance_to_valley_l1844_184441


namespace gcd_of_three_numbers_l1844_184454

-- Definition of the numbers we are interested in
def a : ℕ := 9118
def b : ℕ := 12173
def c : ℕ := 33182

-- Statement of the problem to prove GCD
theorem gcd_of_three_numbers : Int.gcd (Int.gcd a b) c = 47 := 
sorry  -- Proof skipped

end gcd_of_three_numbers_l1844_184454


namespace circumscribed_sphere_surface_area_l1844_184444

noncomputable def surface_area_of_circumscribed_sphere_from_volume (V : ℝ) : ℝ :=
  let s := V^(1/3 : ℝ)
  let d := s * Real.sqrt 3
  4 * Real.pi * (d / 2) ^ 2

theorem circumscribed_sphere_surface_area (V : ℝ) (h : V = 27) : surface_area_of_circumscribed_sphere_from_volume V = 27 * Real.pi :=
by
  rw [h]
  unfold surface_area_of_circumscribed_sphere_from_volume
  sorry

end circumscribed_sphere_surface_area_l1844_184444


namespace largest_and_smallest_correct_l1844_184453

noncomputable def largest_and_smallest (x y : ℝ) (hx : x < 0) (hy : -1 < y ∧ y < 0) : ℝ × ℝ :=
  if hx_y : x * y > 0 then
    if hx_y_sq : x * y * y > x then
      (x * y, x)
    else
      sorry
  else
    sorry

theorem largest_and_smallest_correct {x y : ℝ} (hx : x < 0) (hy : -1 < y ∧ y < 0) :
  largest_and_smallest x y hx hy = (x * y, x) :=
by {
  sorry
}

end largest_and_smallest_correct_l1844_184453


namespace number_of_real_values_of_p_l1844_184427

theorem number_of_real_values_of_p :
  ∃ p_values : Finset ℝ, (∀ p ∈ p_values, ∀ x, x^2 - 2 * p * x + 3 * p = 0 → (x = p)) ∧ Finset.card p_values = 2 :=
by
  sorry

end number_of_real_values_of_p_l1844_184427


namespace range_of_real_roots_l1844_184476

theorem range_of_real_roots (a : ℝ) :
  (∃ x : ℝ, x^2 + 4*a*x - 4*a + 3 = 0) ∨
  (∃ x : ℝ, x^2 + (a-1)*x + a^2 = 0) ∨
  (∃ x : ℝ, x^2 + 2*a*x - 2*a = 0) ↔
  a >= -1 ∨ a <= -3/2 :=
  sorry

end range_of_real_roots_l1844_184476


namespace binom_identity1_binom_identity2_l1844_184474

section Combinatorics

variable (n k m : ℕ)

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.choose n k

-- Prove the identity: C(n, k) + C(n, k-1) = C(n+1, k)
theorem binom_identity1 : binomial n k + binomial n (k-1) = binomial (n+1) k :=
  sorry

-- Using the identity, prove: C(n, m) + C(n-1, m) + ... + C(n-10, m) = C(n+1, m+1) - C(n-10, m+1)
theorem binom_identity2 :
  (binomial n m + binomial (n-1) m + binomial (n-2) m + binomial (n-3) m
   + binomial (n-4) m + binomial (n-5) m + binomial (n-6) m + binomial (n-7) m
   + binomial (n-8) m + binomial (n-9) m + binomial (n-10) m)
   = binomial (n+1) (m+1) - binomial (n-10) (m+1) :=
  sorry

end Combinatorics

end binom_identity1_binom_identity2_l1844_184474


namespace Marc_watch_episodes_l1844_184400

theorem Marc_watch_episodes : ∀ (episodes per_day : ℕ), episodes = 50 → per_day = episodes / 10 → (episodes / per_day) = 10 :=
by
  intros episodes per_day h1 h2
  sorry

end Marc_watch_episodes_l1844_184400


namespace incorrect_statement_isosceles_trapezoid_l1844_184465

-- Define the properties of an isosceles trapezoid
structure IsoscelesTrapezoid (a b c d : ℝ) :=
  (parallel_bases : a = c ∨ b = d)  -- Bases are parallel
  (equal_diagonals : a = b) -- Diagonals are equal
  (equal_angles : ∀ α β : ℝ, α = β)  -- Angles on the same base are equal
  (axisymmetric : ∀ x : ℝ, x = -x)  -- Is an axisymmetric figure

-- Prove that the statement "The two bases of an isosceles trapezoid are parallel and equal" is incorrect
theorem incorrect_statement_isosceles_trapezoid (a b c d : ℝ) (h : IsoscelesTrapezoid a b c d) :
  ¬ (a = c ∧ b = d) :=
sorry

end incorrect_statement_isosceles_trapezoid_l1844_184465


namespace christen_potatoes_and_total_time_l1844_184426

-- Variables representing the given conditions
variables (homer_rate : ℕ) (christen_rate : ℕ) (initial_potatoes : ℕ) 
(homer_time_alone : ℕ) (total_time : ℕ)

-- Specific values for the given problem
def homerRate := 4
def christenRate := 6
def initialPotatoes := 60
def homerTimeAlone := 5

-- Function to calculate the number of potatoes peeled by Homer alone
def potatoesPeeledByHomerAlone :=
  homerRate * homerTimeAlone

-- Function to calculate the number of remaining potatoes
def remainingPotatoes :=
  initialPotatoes - potatoesPeeledByHomerAlone

-- Function to calculate the total peeling rate when Homer and Christen are working together
def combinedRate :=
  homerRate + christenRate

-- Function to calculate the time taken to peel the remaining potatoes
def timePeelingTogether :=
  remainingPotatoes / combinedRate

-- Function to calculate the total time spent peeling potatoes
def totalTime :=
  homerTimeAlone + timePeelingTogether

-- Function to calculate the number of potatoes peeled by Christen
def potatoesPeeledByChristen :=
  christenRate * timePeelingTogether

/- The theorem to be proven: Christen peeled 24 potatoes, and it took 9 minutes to peel all the potatoes. -/
theorem christen_potatoes_and_total_time :
  (potatoesPeeledByChristen = 24) ∧ (totalTime = 9) :=
by {
  sorry
}

end christen_potatoes_and_total_time_l1844_184426


namespace abcde_sum_to_628_l1844_184408

theorem abcde_sum_to_628 (a b c d e : ℕ) (h_distinct : (a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 4 ∨ a = 5) ∧ 
                                                 (b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4 ∨ b = 5) ∧ 
                                                 (c = 1 ∨ c = 2 ∨ c = 3 ∨ c = 4 ∨ c = 5) ∧ 
                                                 (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5) ∧ 
                                                 (e = 1 ∨ e = 2 ∨ e = 3 ∨ e = 4 ∨ e = 5) ∧
                                                 a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
                                                 b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
                                                 c ≠ d ∧ c ≠ e ∧
                                                 d ≠ e)
  (h1 : b ≤ d)
  (h2 : c ≥ a)
  (h3 : a ≤ e)
  (h4 : b ≥ e)
  (h5 : d ≠ 5) :
  a^b + c^d + e = 628 := sorry

end abcde_sum_to_628_l1844_184408


namespace length_of_platform_l1844_184482

theorem length_of_platform 
  (speed_kmph : ℕ)
  (time_cross_platform : ℕ)
  (time_cross_man : ℕ)
  (speed_mps : ℕ)
  (length_of_train : ℕ)
  (distance_platform : ℕ)
  (length_of_platform : ℕ) :
  speed_kmph = 72 →
  time_cross_platform = 30 →
  time_cross_man = 16 →
  speed_mps = speed_kmph * 1000 / 3600 →
  length_of_train = speed_mps * time_cross_man →
  distance_platform = speed_mps * time_cross_platform →
  length_of_platform = distance_platform - length_of_train →
  length_of_platform = 280 := by
  sorry

end length_of_platform_l1844_184482


namespace second_negative_integer_l1844_184448

theorem second_negative_integer (n : ℤ) (h : -11 * n + 5 = 93) : n = -8 :=
by
  sorry

end second_negative_integer_l1844_184448


namespace coordinates_of_point_l1844_184447

theorem coordinates_of_point (a : ℝ) (P : ℝ × ℝ) (hy : P = (a^2 - 1, a + 1)) (hx : (a^2 - 1) = 0) :
  P = (0, 2) ∨ P = (0, 0) :=
sorry

end coordinates_of_point_l1844_184447


namespace vertex_angle_isosceles_l1844_184431

theorem vertex_angle_isosceles (a b c : ℝ)
  (isosceles: (a = b ∨ b = c ∨ c = a))
  (angle_sum : a + b + c = 180)
  (one_angle_is_70 : a = 70 ∨ b = 70 ∨ c = 70) :
  a = 40 ∨ a = 70 ∨ b = 40 ∨ b = 70 ∨ c = 40 ∨ c = 70 :=
by sorry

end vertex_angle_isosceles_l1844_184431


namespace equation_has_one_negative_and_one_zero_root_l1844_184484

theorem equation_has_one_negative_and_one_zero_root :
  ∃ x y : ℝ, x < 0 ∧ y = 0 ∧ 3^x + x^2 + 2 * x - 1 = 0 ∧ 3^y + y^2 + 2 * y - 1 = 0 :=
sorry

end equation_has_one_negative_and_one_zero_root_l1844_184484


namespace solution_is_correct_l1844_184499

-- Define the options
inductive Options
| A_some_other
| B_someone_else
| C_other_person
| D_one_other

-- Define the condition as a function that returns the correct option
noncomputable def correct_option : Options :=
Options.B_someone_else

-- The theorem stating that the correct option must be the given choice
theorem solution_is_correct : correct_option = Options.B_someone_else :=
by
  sorry

end solution_is_correct_l1844_184499


namespace chalkboard_area_l1844_184472

def width : Float := 3.5
def length : Float := 2.3 * width
def area : Float := length * width

theorem chalkboard_area : area = 28.175 :=
by 
  sorry

end chalkboard_area_l1844_184472


namespace maximize_profit_constraints_l1844_184452

variable (a1 a2 b1 b2 d1 d2 c1 c2 x y z : ℝ)

theorem maximize_profit_constraints (a1 a2 b1 b2 d1 d2 c1 c2 x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (a1 * x + a2 * y ≤ c1) ∧ (b1 * x + b2 * y ≤ c2) :=
sorry

end maximize_profit_constraints_l1844_184452


namespace total_strawberries_weight_is_72_l1844_184470

-- Define the weights
def Marco_strawberries_weight := 19
def dad_strawberries_weight := Marco_strawberries_weight + 34 

-- The total weight of their strawberries
def total_strawberries_weight := Marco_strawberries_weight + dad_strawberries_weight

-- Prove that the total weight is 72 pounds
theorem total_strawberries_weight_is_72 : total_strawberries_weight = 72 := by
  sorry

end total_strawberries_weight_is_72_l1844_184470


namespace hiker_distance_l1844_184401

-- Prove that the length of the path d is 90 miles
theorem hiker_distance (x t d : ℝ) (h1 : d = x * t)
                             (h2 : d = (x + 1) * (3 / 4) * t)
                             (h3 : d = (x - 1) * (t + 3)) :
  d = 90 := 
sorry

end hiker_distance_l1844_184401


namespace congruence_equiv_l1844_184488

theorem congruence_equiv (x : ℤ) (h : 5 * x + 9 ≡ 3 [ZMOD 18]) : 3 * x + 14 ≡ 14 [ZMOD 18] :=
sorry

end congruence_equiv_l1844_184488


namespace original_price_l1844_184415

theorem original_price (P : ℝ) (S : ℝ) (h1 : S = 1.3 * P) (h2 : S = P + 650) : P = 2166.67 :=
by
  sorry

end original_price_l1844_184415


namespace boys_variance_greater_than_girls_l1844_184412

noncomputable def variance (scores : List ℝ) : ℝ :=
  let mean := (List.sum scores) / (scores.length : ℝ)
  List.sum (scores.map (λ x => (x - mean) ^ 2)) / (scores.length : ℝ)

def boys_scores : List ℝ := [86, 94, 88, 92, 90]
def girls_scores : List ℝ := [88, 93, 93, 88, 93]

theorem boys_variance_greater_than_girls :
  variance boys_scores > variance girls_scores :=
by
  sorry

end boys_variance_greater_than_girls_l1844_184412


namespace Louisa_travel_distance_l1844_184478

variables (D : ℕ)

theorem Louisa_travel_distance : 
  (200 / 50 + 3 = D / 50) → D = 350 :=
by
  intros h
  sorry

end Louisa_travel_distance_l1844_184478


namespace soccer_game_points_ratio_l1844_184442

theorem soccer_game_points_ratio :
  ∃ B1 A1 A2 B2 : ℕ,
    A1 = 8 ∧
    B2 = 8 ∧
    A2 = 6 ∧
    (A1 + B1 + A2 + B2 = 26) ∧
    (B1 / A1 = 1 / 2) := by
  sorry

end soccer_game_points_ratio_l1844_184442


namespace acute_triangle_tangent_difference_range_l1844_184464

theorem acute_triangle_tangent_difference_range {A B C a b c : ℝ} 
    (h1 : a^2 + b^2 > c^2) (h2 : b^2 + c^2 > a^2) (h3 : c^2 + a^2 > b^2)
    (hb2_minus_ha2_eq_ac : b^2 - a^2 = a * c) :
    1 < (1 / Real.tan A - 1 / Real.tan B) ∧ (1 / Real.tan A - 1 / Real.tan B) < (2 * Real.sqrt 3 / 3) :=
by
  sorry

end acute_triangle_tangent_difference_range_l1844_184464


namespace martin_ring_fraction_l1844_184410

theorem martin_ring_fraction (f : ℚ) :
  (36 + (36 * f + 4) = 52) → (f = 1 / 3) :=
by
  intro h
  -- Solution steps would go here
  sorry

end martin_ring_fraction_l1844_184410


namespace no_such_n_exists_l1844_184413

noncomputable def is_partitionable (s : Finset ℕ) : Prop :=
  ∃ (A B : Finset ℕ), A ∪ B = s ∧ A ∩ B = ∅ ∧ (A.prod id = B.prod id)

theorem no_such_n_exists :
  ¬ ∃ n : ℕ, n > 0 ∧ is_partitionable {n, n+1, n+2, n+3, n+4, n+5} :=
by
  sorry

end no_such_n_exists_l1844_184413


namespace total_bill_l1844_184493

theorem total_bill (total_people : ℕ) (children : ℕ) (adult_cost : ℕ) (child_cost : ℕ)
  (h : total_people = 201) (hc : children = 161) (ha : adult_cost = 8) (hc_cost : child_cost = 4) :
  (201 - 161) * 8 + 161 * 4 = 964 :=
by
  rw [←h, ←hc, ←ha, ←hc_cost]
  sorry

end total_bill_l1844_184493


namespace west_movement_80_eq_neg_80_l1844_184436

-- Define conditions
def east_movement (distance : ℤ) : ℤ := distance

-- Prove that moving westward is represented correctly
theorem west_movement_80_eq_neg_80 : east_movement (-80) = -80 :=
by
  -- Theorem proof goes here
  sorry

end west_movement_80_eq_neg_80_l1844_184436


namespace nails_per_plank_l1844_184451

theorem nails_per_plank {total_nails planks : ℕ} (h1 : total_nails = 4) (h2 : planks = 2) :
  total_nails / planks = 2 := by
  sorry

end nails_per_plank_l1844_184451


namespace michelle_sandwiches_l1844_184438

def sandwiches_left (total : ℕ) (given_to_coworker : ℕ) (kept : ℕ) : ℕ :=
  total - given_to_coworker - kept

theorem michelle_sandwiches : sandwiches_left 20 4 (4 * 2) = 8 :=
by
  sorry

end michelle_sandwiches_l1844_184438


namespace concentric_circle_ratio_l1844_184491

theorem concentric_circle_ratio (r R : ℝ) (hRr : R > r)
  (new_circles_tangent : ∀ (C1 C2 C3 : ℝ), C1 = C2 ∧ C2 = C3 ∧ C1 < R ∧ r < C1): 
  R = 3 * r := by sorry

end concentric_circle_ratio_l1844_184491


namespace maria_miles_after_second_stop_l1844_184419

theorem maria_miles_after_second_stop (total_distance : ℕ)
    (h1 : total_distance = 360)
    (distance_first_stop : ℕ)
    (h2 : distance_first_stop = total_distance / 2)
    (remaining_distance_after_first_stop : ℕ)
    (h3 : remaining_distance_after_first_stop = total_distance - distance_first_stop)
    (distance_second_stop : ℕ)
    (h4 : distance_second_stop = remaining_distance_after_first_stop / 4)
    (remaining_distance_after_second_stop : ℕ)
    (h5 : remaining_distance_after_second_stop = remaining_distance_after_first_stop - distance_second_stop) :
    remaining_distance_after_second_stop = 135 := by
  sorry

end maria_miles_after_second_stop_l1844_184419


namespace smaller_tablet_diagonal_l1844_184430

theorem smaller_tablet_diagonal :
  ∀ (A_large A_small : ℝ)
    (d : ℝ),
    A_large = (8 / Real.sqrt 2) ^ 2 →
    A_small = (d / Real.sqrt 2) ^ 2 →
    A_large = A_small + 7.5 →
    d = 7
:= by
  intros A_large A_small d h1 h2 h3
  sorry

end smaller_tablet_diagonal_l1844_184430


namespace sum_of_center_coordinates_l1844_184411

theorem sum_of_center_coordinates (x y : ℝ) :
    (x^2 + y^2 - 6*x + 8*y = 18) → (x = 3) → (y = -4) → x + y = -1 := 
by
    intro h1 hx hy
    rw [hx, hy]
    norm_num

end sum_of_center_coordinates_l1844_184411


namespace apples_first_year_l1844_184432

theorem apples_first_year (A : ℕ) 
  (second_year_prod : ℕ := 2 * A + 8)
  (third_year_prod : ℕ := 3 * (2 * A + 8) / 4)
  (total_prod : ℕ := A + second_year_prod + third_year_prod) :
  total_prod = 194 → A = 40 :=
by
  sorry

end apples_first_year_l1844_184432


namespace neg_of_exists_l1844_184490

theorem neg_of_exists (P : ℝ → Prop) : 
  (¬ ∃ x: ℝ, x ≥ 3 ∧ x^2 - 2 * x + 3 < 0) ↔ (∀ x: ℝ, x ≥ 3 → x^2 - 2 * x + 3 ≥ 0) :=
by
  sorry

end neg_of_exists_l1844_184490


namespace roots_sum_squares_l1844_184460

theorem roots_sum_squares (a b c : ℝ) (h₁ : Polynomial.eval a (3 * X^3 - 2 * X^2 + 5 * X + 15) = 0)
  (h₂ : Polynomial.eval b (3 * X^3 - 2 * X^2 + 5 * X + 15) = 0)
  (h₃ : Polynomial.eval c (3 * X^3 - 2 * X^2 + 5 * X + 15) = 0) :
  a^2 + b^2 + c^2 = -26 / 9 :=
sorry

end roots_sum_squares_l1844_184460


namespace a_is_4_when_b_is_3_l1844_184498

theorem a_is_4_when_b_is_3 
  (a : ℝ) (b : ℝ) (k : ℝ)
  (h1 : ∀ b, a * b^2 = k)
  (h2 : a = 9 ∧ b = 2) :
  a = 4 :=
by
  sorry

end a_is_4_when_b_is_3_l1844_184498


namespace keaton_earns_yearly_l1844_184468

/-- Keaton's total yearly earnings from oranges and apples given the harvest cycles and prices. -/
theorem keaton_earns_yearly : 
  let orange_harvest_cycle := 2
  let orange_harvest_price := 50
  let apple_harvest_cycle := 3
  let apple_harvest_price := 30
  let months_in_a_year := 12
  
  let orange_harvests_per_year := months_in_a_year / orange_harvest_cycle
  let apple_harvests_per_year := months_in_a_year / apple_harvest_cycle
  
  let orange_yearly_earnings := orange_harvests_per_year * orange_harvest_price
  let apple_yearly_earnings := apple_harvests_per_year * apple_harvest_price
    
  orange_yearly_earnings + apple_yearly_earnings = 420 :=
by
  sorry

end keaton_earns_yearly_l1844_184468


namespace remainder_of_7_pow_145_mod_9_l1844_184462

theorem remainder_of_7_pow_145_mod_9 : (7 ^ 145) % 9 = 7 := by
  sorry

end remainder_of_7_pow_145_mod_9_l1844_184462


namespace jonah_fish_count_l1844_184407

theorem jonah_fish_count :
  let initial_fish := 14
  let added_fish := 2
  let eaten_fish := 6
  let removed_fish := 2
  let new_fish := 3
  initial_fish + added_fish - eaten_fish - removed_fish + new_fish = 11 := 
by
  sorry

end jonah_fish_count_l1844_184407


namespace inequality_example_l1844_184458

theorem inequality_example (a b c : ℝ) (hac : a ≠ 0) (hbc : b ≠ 0) (hcc : c ≠ 0) :
  (a^4) / (4 * a^4 + b^4 + c^4) + (b^4) / (a^4 + 4 * b^4 + c^4) + (c^4) / (a^4 + b^4 + 4 * c^4) ≤ 1 / 2 :=
sorry

end inequality_example_l1844_184458


namespace intersection_M_N_l1844_184421

noncomputable def M : Set ℝ := {x | x^2 - x ≤ 0}
noncomputable def N : Set ℝ := {x | x < 1}

theorem intersection_M_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l1844_184421


namespace james_total_cost_l1844_184485

def courseCost (units: Nat) (cost_per_unit: Nat) : Nat :=
  units * cost_per_unit

def totalCostForFall : Nat :=
  courseCost 12 60 + courseCost 8 45

def totalCostForSpring : Nat :=
  let science_cost := courseCost 10 60
  let science_scholarship := science_cost / 2
  let humanities_cost := courseCost 10 45
  (science_cost - science_scholarship) + humanities_cost

def totalCostForSummer : Nat :=
  courseCost 6 80 + courseCost 4 55

def totalCostForWinter : Nat :=
  let science_cost := courseCost 6 80
  let science_scholarship := 3 * science_cost / 4
  let humanities_cost := courseCost 4 55
  (science_cost - science_scholarship) + humanities_cost

def totalAmountSpent : Nat :=
  totalCostForFall + totalCostForSpring + totalCostForSummer + totalCostForWinter

theorem james_total_cost: totalAmountSpent = 2870 :=
  by sorry

end james_total_cost_l1844_184485


namespace goods_train_length_is_420_l1844_184450

/-- The man's train speed in km/h. -/
def mans_train_speed_kmph : ℝ := 64

/-- The goods train speed in km/h. -/
def goods_train_speed_kmph : ℝ := 20

/-- The time taken for the trains to pass each other in seconds. -/
def passing_time_s : ℝ := 18

/-- The relative speed of two trains traveling in opposite directions in m/s. -/
noncomputable def relative_speed_mps : ℝ := 
  (mans_train_speed_kmph + goods_train_speed_kmph) * 1000 / 3600

/-- The length of the goods train in meters. -/
noncomputable def goods_train_length_m : ℝ := relative_speed_mps * passing_time_s

/-- The theorem stating the length of the goods train is 420 meters. -/
theorem goods_train_length_is_420 :
  goods_train_length_m = 420 :=
sorry

end goods_train_length_is_420_l1844_184450


namespace students_can_do_both_l1844_184487

variable (total_students swimmers gymnasts neither : ℕ)

theorem students_can_do_both (h1 : total_students = 60)
                             (h2 : swimmers = 27)
                             (h3 : gymnasts = 28)
                             (h4 : neither = 15) : 
                             total_students - (total_students - swimmers + total_students - gymnasts - neither) = 10 := 
by 
  sorry

end students_can_do_both_l1844_184487


namespace find_k_l1844_184435

-- Define the function y = kx
def linear_function (k x : ℝ) : ℝ := k * x

-- Define the point P(3,1)
def P : ℝ × ℝ := (3, 1)

theorem find_k (k : ℝ) (h : linear_function k 3 = 1) : k = 1 / 3 :=
by
  sorry

end find_k_l1844_184435


namespace find_x_l1844_184486

theorem find_x (x : ℝ) (h : x + 2.75 + 0.158 = 2.911) : x = 0.003 :=
sorry

end find_x_l1844_184486


namespace total_appetizers_l1844_184417

theorem total_appetizers (hotdogs cheese_pops chicken_nuggets mini_quiches stuffed_mushrooms total_portions : Nat)
  (h1 : hotdogs = 60)
  (h2 : cheese_pops = 40)
  (h3 : chicken_nuggets = 80)
  (h4 : mini_quiches = 100)
  (h5 : stuffed_mushrooms = 50)
  (h6 : total_portions = hotdogs + cheese_pops + chicken_nuggets + mini_quiches + stuffed_mushrooms) :
  total_portions = 330 :=
by sorry

end total_appetizers_l1844_184417


namespace triangle_inequality_l1844_184439

variables {A B C P D E F : Type} -- Variables representing points in the plane.
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P]
variables (PD PE PF PA PB PC : ℝ) -- Distances corresponding to the points.

-- Condition stating P lies inside or on the boundary of triangle ABC
axiom P_in_triangle_ABC : ∀ (A B C P : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P], 
  (PD > 0 ∧ PE > 0 ∧ PF > 0 ∧ PA > 0 ∧ PB > 0 ∧ PC > 0)

-- Objective statement to prove
theorem triangle_inequality (PD PE PF PA PB PC : ℝ) 
  (h1 : PA ≥ 0) 
  (h2 : PB ≥ 0) 
  (h3 : PC ≥ 0) 
  (h4 : PD ≥ 0) 
  (h5 : PE ≥ 0) 
  (h6 : PF ≥ 0) :
  PA + PB + PC ≥ 2 * (PD + PE + PF) := 
sorry -- Proof to be provided later.

end triangle_inequality_l1844_184439


namespace largest_possible_a_l1844_184466

theorem largest_possible_a 
  (a b c d : ℕ) 
  (h1 : a < 2 * b)
  (h2 : b < 3 * c)
  (h3 : c < 2 * d)
  (h4 : d < 100) :
  a ≤ 1179 :=
sorry

end largest_possible_a_l1844_184466


namespace final_amount_after_two_years_l1844_184428

open BigOperators

/-- Given an initial amount A0 and a percentage increase p, calculate the amount after n years -/
def compound_increase (A0 : ℝ) (p : ℝ) (n : ℕ) : ℝ :=
  (A0 * (1 + p)^n)

theorem final_amount_after_two_years (A0 : ℝ) (p : ℝ) (A2 : ℝ) :
  A0 = 1600 ∧ p = 1 / 8 ∧ compound_increase 1600 (1 / 8) 2 = 2025 :=
  sorry

end final_amount_after_two_years_l1844_184428


namespace arithmetic_sequence_a5_l1844_184492

theorem arithmetic_sequence_a5 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ)
  (h1 : a 1 = 1)
  (h2 : S 4 = 16)
  (h_sum : ∀ n, S n = (n * (2 * (a 1) + (n - 1) * (a 2 - a 1))) / 2)
  (h_a : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) :
  a 5 = 9 :=
by 
  sorry

end arithmetic_sequence_a5_l1844_184492


namespace celsius_equals_fahrenheit_l1844_184469

-- Define the temperature scales.
def celsius_to_fahrenheit (T_C : ℝ) : ℝ := 1.8 * T_C + 32

-- The Lean statement for the problem.
theorem celsius_equals_fahrenheit : ∃ (T : ℝ), T = celsius_to_fahrenheit T ↔ T = -40 :=
by
  sorry -- Proof is not required, just the statement.

end celsius_equals_fahrenheit_l1844_184469


namespace determinant_triangle_l1844_184402

theorem determinant_triangle (A B C : ℝ) (h : A + B + C = Real.pi) :
  Matrix.det ![![Real.cos A ^ 2, Real.tan A, 1],
               ![Real.cos B ^ 2, Real.tan B, 1],
               ![Real.cos C ^ 2, Real.tan C, 1]] = 0 := by
  sorry

end determinant_triangle_l1844_184402


namespace complement_of_S_in_U_l1844_184471

variable (U : Set ℕ)
variable (S : Set ℕ)

theorem complement_of_S_in_U (hU : U = {1, 2, 3, 4}) (hS : S = {1, 3}) : U \ S = {2, 4} := by
  sorry

end complement_of_S_in_U_l1844_184471


namespace sequence_terms_are_integers_l1844_184467

theorem sequence_terms_are_integers (a : ℕ → ℕ)
  (h0 : a 0 = 1) 
  (h1 : a 1 = 2) 
  (h_recurrence : ∀ n : ℕ, (n + 3) * a (n + 2) = (6 * n + 9) * a (n + 1) - n * a n) :
  ∀ n : ℕ, ∃ k : ℤ, a n = k := 
by
  -- Initialize the proof
  sorry

end sequence_terms_are_integers_l1844_184467


namespace potato_slice_length_l1844_184434

theorem potato_slice_length (x : ℕ) (h1 : 600 = x + (x + 50)) : x + 50 = 325 :=
by
  sorry

end potato_slice_length_l1844_184434


namespace students_per_group_correct_l1844_184409

def total_students : ℕ := 850
def number_of_teachers : ℕ := 23
def students_per_group : ℕ := total_students / number_of_teachers

theorem students_per_group_correct : students_per_group = 36 := sorry

end students_per_group_correct_l1844_184409


namespace gcd_437_323_eq_19_l1844_184494

theorem gcd_437_323_eq_19 : Int.gcd 437 323 = 19 := 
by 
  sorry

end gcd_437_323_eq_19_l1844_184494


namespace sin_14pi_over_5_eq_sin_36_degree_l1844_184423

noncomputable def sin_14pi_over_5 : ℝ :=
  Real.sin (14 * Real.pi / 5)

noncomputable def sin_36_degree : ℝ :=
  Real.sin (36 * Real.pi / 180)

theorem sin_14pi_over_5_eq_sin_36_degree :
  sin_14pi_over_5 = sin_36_degree :=
sorry

end sin_14pi_over_5_eq_sin_36_degree_l1844_184423


namespace sequence_term_condition_l1844_184422

theorem sequence_term_condition (n : ℕ) : (n^2 - 8 * n + 15 = 3) ↔ (n = 2 ∨ n = 6) :=
by 
  sorry

end sequence_term_condition_l1844_184422


namespace sample_size_proof_l1844_184414

-- Conditions
def investigate_height_of_students := "To investigate the height of junior high school students in Rui State City in early 2016, 200 students were sampled for the survey."

-- Definition of sample size based on the condition
def sample_size_condition (students_sampled : ℕ) : ℕ := students_sampled

-- Prove the sample size is 200 given the conditions
theorem sample_size_proof : sample_size_condition 200 = 200 := 
by
  sorry

end sample_size_proof_l1844_184414


namespace school_club_profit_l1844_184497

-- Definition of the problem conditions
def candy_bars_bought : ℕ := 800
def cost_per_four_bars : ℚ := 3
def bars_per_four_bars : ℕ := 4
def sell_price_per_three_bars : ℚ := 2
def bars_per_three_bars : ℕ := 3
def sales_fee_per_bar : ℚ := 0.05

-- Definition for cost calculations
def cost_per_bar : ℚ := cost_per_four_bars / bars_per_four_bars
def total_cost : ℚ := candy_bars_bought * cost_per_bar

-- Definition for revenue calculations
def sell_price_per_bar : ℚ := sell_price_per_three_bars / bars_per_three_bars
def total_revenue : ℚ := candy_bars_bought * sell_price_per_bar

-- Definition for total sales fee
def total_sales_fee : ℚ := candy_bars_bought * sales_fee_per_bar

-- Definition of profit
def profit : ℚ := total_revenue - total_cost - total_sales_fee

-- The statement to be proved
theorem school_club_profit : profit = -106.64 := by sorry

end school_club_profit_l1844_184497


namespace minimize_b_plus_c_l1844_184445

theorem minimize_b_plus_c (a b c : ℝ) (h1 : 0 < a)
  (h2 : ∀ x, (y : ℝ) = a * x^2 + b * x + c)
  (h3 : ∀ x, (yr : ℝ) = a * (x + 2)^2 + (a - 1)^2) :
  a = 1 :=
by
  sorry

end minimize_b_plus_c_l1844_184445


namespace sphere_surface_area_l1844_184416

theorem sphere_surface_area
  (V : ℝ)
  (r : ℝ)
  (h : ℝ)
  (R : ℝ)
  (V_cone : V = (2 * π) / 3)
  (r_cone_base : r = 1)
  (cone_height : h = 2 * V / (π * r^2))
  (sphere_radius : R^2 - (R - h)^2 = r^2):
  4 * π * R^2 = 25 * π / 4 :=
by
  sorry

end sphere_surface_area_l1844_184416


namespace monkey_reach_top_in_20_hours_l1844_184473

-- Defining the conditions
def tree_height : ℕ := 21
def hop_distance : ℕ := 3
def slip_distance : ℕ := 2

-- Defining the net distance gain per hour
def net_gain_per_hour : ℕ := hop_distance - slip_distance

-- Proof statement
theorem monkey_reach_top_in_20_hours :
  ∃ t : ℕ, t = 20 ∧ 20 * net_gain_per_hour + hop_distance = tree_height :=
by
  sorry

end monkey_reach_top_in_20_hours_l1844_184473


namespace dealer_cannot_prevent_goal_l1844_184418

theorem dealer_cannot_prevent_goal (m n : ℕ) :
  (m + n) % 4 = 0 :=
sorry

end dealer_cannot_prevent_goal_l1844_184418


namespace number_of_ping_pong_balls_l1844_184437

def sales_tax_rate : ℝ := 0.16

def total_cost_with_tax (B x : ℝ) : ℝ := B * x * (1 + sales_tax_rate)

def total_cost_without_tax (B x : ℝ) : ℝ := (B + 3) * x

theorem number_of_ping_pong_balls
  (B x : ℝ) (h₁ : total_cost_with_tax B x = total_cost_without_tax B x) :
  B = 18.75 := 
sorry

end number_of_ping_pong_balls_l1844_184437


namespace total_children_on_playground_l1844_184456

theorem total_children_on_playground (boys girls : ℕ) (hb : boys = 27) (hg : girls = 35) : boys + girls = 62 :=
  by
  -- Proof goes here
  sorry

end total_children_on_playground_l1844_184456


namespace diaz_age_twenty_years_later_l1844_184424

theorem diaz_age_twenty_years_later (D S : ℕ) (h₁ : 10 * D - 40 = 10 * S + 20) (h₂ : S = 30) : D + 20 = 56 :=
sorry

end diaz_age_twenty_years_later_l1844_184424


namespace ab_inequality_smaller_than_fourth_sum_l1844_184483

theorem ab_inequality_smaller_than_fourth_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a * b) / (a + b + 2 * c) + (b * c) / (b + c + 2 * a) + (c * a) / (c + a + 2 * b) ≤ (1 / 4) * (a + b + c) := 
by
  sorry

end ab_inequality_smaller_than_fourth_sum_l1844_184483


namespace part1_k_real_part2_find_k_l1844_184477

-- Part 1: Discriminant condition
theorem part1_k_real (k : ℝ) (h : x^2 + (2*k - 1)*x + k^2 - 1 = 0) : k ≤ 5 / 4 :=
by
  sorry

-- Part 2: Given additional conditions, find k
theorem part2_find_k (x1 x2 k : ℝ) (h_eq : x^2 + (2 * k - 1) * x + k^2 - 1 = 0)
  (h1 : x1 + x2 = 1 - 2 * k) (h2 : x1 * x2 = k^2 - 1) (h3 : x1^2 + x2^2 = 16 + x1 * x2) : k = -2 :=
by
  sorry

end part1_k_real_part2_find_k_l1844_184477


namespace hexagon_area_l1844_184481

-- Definition of an equilateral triangle with a given perimeter.
def is_equilateral_triangle (P Q R : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R] :=
  ∀ (a b c : ℝ), a = b ∧ b = c ∧ a + b + c = 42 ∧ ∀ (angle : ℝ), angle = 60

-- Statement of the problem
theorem hexagon_area (P Q R P' Q' R' : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R]
  [MetricSpace P'] [MetricSpace Q'] [MetricSpace R']
  (h1 : is_equilateral_triangle P Q R) :
  ∃ (area : ℝ), area = 49 * Real.sqrt 3 := 
sorry

end hexagon_area_l1844_184481


namespace subtract_two_decimals_l1844_184420

theorem subtract_two_decimals : 3.75 - 1.46 = 2.29 := by
  sorry

end subtract_two_decimals_l1844_184420


namespace smallest_x_l1844_184461

theorem smallest_x :
  ∃ (x : ℕ), x % 4 = 3 ∧ x % 5 = 4 ∧ x % 6 = 5 ∧ ∀ y : ℕ, (y % 4 = 3 ∧ y % 5 = 4 ∧ y % 6 = 5) → y ≥ x := 
sorry

end smallest_x_l1844_184461


namespace number_of_friends_l1844_184404

-- Define the initial amount of money John had
def initial_money : ℝ := 20.10 

-- Define the amount spent on sweets
def sweets_cost : ℝ := 1.05 

-- Define the amount given to each friend
def money_per_friend : ℝ := 1.00 

-- Define the amount of money left after giving to friends
def final_money : ℝ := 17.05 

-- Define a theorem to find the number of friends John gave money to
theorem number_of_friends (init_money sweets_cost money_per_friend final_money : ℝ) : 
  (init_money - sweets_cost - final_money) / money_per_friend = 2 :=
by
  sorry

end number_of_friends_l1844_184404


namespace polynomial_divisibility_l1844_184406

theorem polynomial_divisibility (m : ℤ) : (4 * m + 5) ^ 2 - 9 ∣ 8 := by
  sorry

end polynomial_divisibility_l1844_184406


namespace Bulgaria_f_1992_divisibility_l1844_184443

def f (m n : ℕ) : ℕ := m^(3^(4 * n) + 6) - m^(3^(4 * n) + 4) - m^5 + m^3

theorem Bulgaria_f_1992_divisibility (n : ℕ) (m : ℕ) :
  ( ∀ m : ℕ, m > 0 → f m n ≡ 0 [MOD 1992] ) ↔ ( n % 2 = 1 ) :=
by
  sorry

end Bulgaria_f_1992_divisibility_l1844_184443


namespace graph_is_line_l1844_184480

theorem graph_is_line : {p : ℝ × ℝ | (p.1 - p.2)^2 = 2 * (p.1^2 + p.2^2)} = {p : ℝ × ℝ | p.2 = -p.1} :=
by 
  sorry

end graph_is_line_l1844_184480


namespace value_for_real_value_for_pure_imaginary_l1844_184449

def is_real (z : ℂ) : Prop := z.im = 0
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

def value_conditions (k : ℝ) : ℂ := ⟨k^2 - 3*k - 4, k^2 - 5*k - 6⟩

theorem value_for_real (k : ℝ) : is_real (value_conditions k) ↔ (k = 6 ∨ k = -1) :=
by
  sorry

theorem value_for_pure_imaginary (k : ℝ) : is_pure_imaginary (value_conditions k) ↔ (k = 4) :=
by
  sorry

end value_for_real_value_for_pure_imaginary_l1844_184449


namespace logic_problem_l1844_184440

variable (p q : Prop)

theorem logic_problem (h₁ : p ∨ q) (h₂ : ¬ p) : ¬ p ∧ q :=
by
  sorry

end logic_problem_l1844_184440


namespace total_spent_is_correct_l1844_184475

def trumpet : ℝ := 149.16
def music_tool : ℝ := 9.98
def song_book : ℝ := 4.14
def trumpet_maintenance_accessories : ℝ := 21.47
def valve_oil_original : ℝ := 8.20
def valve_oil_discount_rate : ℝ := 0.20
def valve_oil_discounted : ℝ := valve_oil_original * (1 - valve_oil_discount_rate)
def band_t_shirt : ℝ := 14.95
def sales_tax_rate : ℝ := 0.065

def total_before_tax : ℝ :=
  trumpet + music_tool + song_book + trumpet_maintenance_accessories + valve_oil_discounted + band_t_shirt

def sales_tax : ℝ := total_before_tax * sales_tax_rate

def total_amount_spent : ℝ := total_before_tax + sales_tax

theorem total_spent_is_correct : total_amount_spent = 219.67 := by
  sorry

end total_spent_is_correct_l1844_184475


namespace blue_socks_count_l1844_184463

theorem blue_socks_count (total_socks : ℕ) (two_thirds_white : ℕ) (one_third_blue : ℕ) 
  (h1 : total_socks = 180) 
  (h2 : two_thirds_white = (2 / 3) * total_socks) 
  (h3 : one_third_blue = total_socks - two_thirds_white) : 
  one_third_blue = 60 :=
by
  sorry

end blue_socks_count_l1844_184463


namespace fractional_eq_solution_l1844_184425

theorem fractional_eq_solution (k : ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → (3 / x + 6 / (x - 1) - (x + k) / (x * (x - 1)) = 0)) →
  k ≠ -3 ∧ k ≠ 5 :=
by 
  sorry

end fractional_eq_solution_l1844_184425


namespace mrs_sheridan_final_cats_l1844_184403

def initial_cats : ℝ := 17.5
def given_away_cats : ℝ := 6.2
def returned_cats : ℝ := 2.8
def additional_given_away_cats : ℝ := 1.3

theorem mrs_sheridan_final_cats : 
  initial_cats - given_away_cats + returned_cats - additional_given_away_cats = 12.8 :=
by
  sorry

end mrs_sheridan_final_cats_l1844_184403


namespace Mike_age_l1844_184457

-- We define the ages of Mike and Barbara
variables (M B : ℕ)

-- Conditions extracted from the problem
axiom h1 : B = M / 2
axiom h2 : M - B = 8

-- The theorem to prove
theorem Mike_age : M = 16 :=
by sorry

end Mike_age_l1844_184457


namespace equidistant_point_on_y_axis_l1844_184459

theorem equidistant_point_on_y_axis :
  ∃ (y : ℝ), 0 < y ∧ 
  (dist (0, y) (-3, 0) = dist (0, y) (-2, 5)) ∧ 
  y = 2 :=
by
  sorry

end equidistant_point_on_y_axis_l1844_184459


namespace divisible_by_900_l1844_184495

theorem divisible_by_900 (n : ℕ) : 900 ∣ (6 ^ (2 * (n + 1)) - 2 ^ (n + 3) * 3 ^ (n + 2) + 36) := 
by 
  sorry

end divisible_by_900_l1844_184495


namespace part1_cos_A_part2_c_l1844_184429

-- We define a triangle with sides a, b, c opposite to angles A, B, C respectively.
variables (a b c : ℝ) (A B C : ℝ)
-- Given conditions for the problem:
variable (h1 : 3 * a * Real.cos A = c * Real.cos B + b * Real.cos C)
variable (h_cos_sum : Real.cos B + Real.cos C = (2 * Real.sqrt 3) / 3)
variable (ha : a = 2 * Real.sqrt 3)

-- The first part of the problem statement proving cos A = 1/3 given the conditions.
theorem part1_cos_A : Real.cos A = 1 / 3 :=
by
  sorry

-- The second part of the problem statement proving c = 3 given the conditions.
theorem part2_c : c = 3 :=
by
  sorry

end part1_cos_A_part2_c_l1844_184429


namespace max_2x_plus_y_value_l1844_184496

open Real

def on_ellipse (P : ℝ × ℝ) : Prop := 
  (P.1^2 / 4 + P.2^2 = 1)

def max_value_2x_plus_y (P : ℝ × ℝ) (h : on_ellipse P) : ℝ := 
  2 * P.1 + P.2

theorem max_2x_plus_y_value (P : ℝ × ℝ) (h : on_ellipse P):
  ∃ (m : ℝ), max_value_2x_plus_y P h = m ∧ m = sqrt 17 :=
sorry

end max_2x_plus_y_value_l1844_184496


namespace train_speed_l1844_184455

theorem train_speed :
  ∀ (length : ℝ) (time : ℝ),
    length = 135 ∧ time = 3.4711508793582233 →
    (length / time) * 3.6 = 140.0004 :=
by
  sorry

end train_speed_l1844_184455


namespace proof_triangle_inequality_l1844_184446

noncomputable def proof_statement (a b c: ℝ) (h: a + b > c ∧ b + c > a ∧ c + a > b) : Prop :=
  a * b * c ≥ (-a + b + c) * (a - b + c) * (a + b - c)

-- Proof statement without the proof
theorem proof_triangle_inequality (a b c: ℝ) (h: a + b > c ∧ b + c > a ∧ c + a > b) : 
  proof_statement a b c h :=
sorry

end proof_triangle_inequality_l1844_184446
