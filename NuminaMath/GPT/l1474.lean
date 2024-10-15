import Mathlib

namespace NUMINAMATH_GPT_difference_between_possible_values_of_x_l1474_147455

noncomputable def difference_of_roots (x : ℝ) (h : (x + 3)^2 / (3 * x + 65) = 2) : ℝ :=
  let sol1 := 11  -- First root
  let sol2 := -11 -- Second root
  sol1 - sol2

theorem difference_between_possible_values_of_x (x : ℝ) (h : (x + 3)^2 / (3 * x + 65) = 2) :
  difference_of_roots x h = 22 :=
sorry

end NUMINAMATH_GPT_difference_between_possible_values_of_x_l1474_147455


namespace NUMINAMATH_GPT_minimum_omega_l1474_147409

theorem minimum_omega (ω : ℕ) (h_pos : ω ∈ {n : ℕ | n > 0}) (h_cos_center : ∃ k : ℤ, ω * (π / 6) + (π / 6) = k * π + π / 2) :
  ω = 2 :=
by { sorry }

end NUMINAMATH_GPT_minimum_omega_l1474_147409


namespace NUMINAMATH_GPT_total_cost_of_fencing_l1474_147404

theorem total_cost_of_fencing (length breadth : ℕ) (cost_per_metre : ℕ) 
  (h1 : length = breadth + 20) 
  (h2 : length = 200) 
  (h3 : cost_per_metre = 26): 
  2 * (length + breadth) * cost_per_metre = 20140 := 
by sorry

end NUMINAMATH_GPT_total_cost_of_fencing_l1474_147404


namespace NUMINAMATH_GPT_how_many_correct_l1474_147463

def calc1 := (2 * Real.sqrt 3) * (3 * Real.sqrt 3) = 6 * Real.sqrt 3
def calc2 := Real.sqrt 2 + Real.sqrt 3 = Real.sqrt 5
def calc3 := (5 * Real.sqrt 5) - (2 * Real.sqrt 2) = 3 * Real.sqrt 3
def calc4 := (Real.sqrt 2) / (Real.sqrt 3) = (Real.sqrt 6) / 3

theorem how_many_correct : (¬ calc1) ∧ (¬ calc2) ∧ (¬ calc3) ∧ calc4 → 1 = 1 :=
by { sorry }

end NUMINAMATH_GPT_how_many_correct_l1474_147463


namespace NUMINAMATH_GPT_find_sum_of_terms_l1474_147494

noncomputable def arithmetic_sequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d
noncomputable def sum_of_first_n_terms (a₁ d n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem find_sum_of_terms (a₁ d : ℕ) (S : ℕ → ℕ) (h1 : S 4 = 8) (h2 : S 8 = 20) :
    S 4 = 4 * (2 * a₁ + 3 * d) / 2 → S 8 = 8 * (2 * a₁ + 7 * d) / 2 →
    a₁ = 13 / 8 ∧ d = 1 / 4 →
    a₁ + 10 * d + a₁ + 11 * d + a₁ + 12 * d + a₁ + 13 * d = 18 :=
by 
  sorry

end NUMINAMATH_GPT_find_sum_of_terms_l1474_147494


namespace NUMINAMATH_GPT_all_values_equal_l1474_147469

noncomputable def f : ℤ × ℤ → ℕ :=
sorry

theorem all_values_equal (f : ℤ × ℤ → ℕ)
  (h_pos : ∀ p, 0 < f p)
  (h_mean : ∀ x y, f (x, y) = 1/4 * (f (x+1, y) + f (x-1, y) + f (x, y+1) + f (x, y-1))) :
  ∀ (x1 y1 x2 y2 : ℤ), f (x1, y1) = f (x2, y2) := 
sorry

end NUMINAMATH_GPT_all_values_equal_l1474_147469


namespace NUMINAMATH_GPT_fraction_of_kiwis_l1474_147408

theorem fraction_of_kiwis (total_fruits : ℕ) (num_strawberries : ℕ) (h₁ : total_fruits = 78) (h₂ : num_strawberries = 52) :
  (total_fruits - num_strawberries) / total_fruits = 1 / 3 :=
by
  -- proof to be provided, this is just the statement
  sorry

end NUMINAMATH_GPT_fraction_of_kiwis_l1474_147408


namespace NUMINAMATH_GPT_find_number_l1474_147459

theorem find_number (a b : ℕ) (h₁ : a = 555) (h₂ : b = 445) :
  let S := a + b
  let D := a - b
  let Q := 2 * D
  let R := 30
  let N := (S * Q) + R
  N = 220030 := by
  sorry

end NUMINAMATH_GPT_find_number_l1474_147459


namespace NUMINAMATH_GPT_teena_speed_l1474_147401

theorem teena_speed (T : ℝ) : 
  (∀ (d₀ d_poe d_ahead : ℝ), 
    d₀ = 7.5 ∧ d_poe = 40 * 1.5 ∧ d_ahead = 15 →
    T = (d₀ + d_poe + d_ahead) / 1.5) → 
  T = 55 :=
by
  intros
  sorry

end NUMINAMATH_GPT_teena_speed_l1474_147401


namespace NUMINAMATH_GPT_cover_rectangle_with_polyomino_l1474_147464

-- Defining the conditions under which the m x n rectangle can be covered by the given polyomino
theorem cover_rectangle_with_polyomino (m n : ℕ) :
  (6 ∣ (m * n)) →
  (m ≠ 1 ∧ m ≠ 2 ∧ m ≠ 5) →
  (n ≠ 1 ∧ n ≠ 2 ∧ n ≠ 5) →
  ((3 ∣ m ∧ 4 ∣ n) ∨ (3 ∣ n ∧ 4 ∣ m) ∨ (12 ∣ (m * n))) :=
sorry

end NUMINAMATH_GPT_cover_rectangle_with_polyomino_l1474_147464


namespace NUMINAMATH_GPT_perpendicular_lines_a_value_l1474_147437

theorem perpendicular_lines_a_value :
  ∀ a : ℝ, 
    (∀ x y : ℝ, 2*x + a*y - 7 = 0) → 
    (∀ x y : ℝ, (a-3)*x + y + 4 = 0) → a = 2 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_a_value_l1474_147437


namespace NUMINAMATH_GPT_M_equals_N_l1474_147415

-- Define the sets M and N
def M : Set ℝ := {x | 0 ≤ x}
def N : Set ℝ := {y | 0 ≤ y}

-- State the main proof goal
theorem M_equals_N : M = N :=
by
  sorry

end NUMINAMATH_GPT_M_equals_N_l1474_147415


namespace NUMINAMATH_GPT_rectangular_field_length_l1474_147491

theorem rectangular_field_length (w l : ℝ) (h1 : l = w + 10) (h2 : l^2 + w^2 = 22^2) : l = 22 := 
sorry

end NUMINAMATH_GPT_rectangular_field_length_l1474_147491


namespace NUMINAMATH_GPT_friend_spent_more_l1474_147496

/-- Given that the total amount spent for lunch is $15 and your friend spent $8 on their lunch,
we need to prove that your friend spent $1 more than you did. -/
theorem friend_spent_more (total_spent friend_spent : ℤ) (h1 : total_spent = 15) (h2 : friend_spent = 8) :
  friend_spent - (total_spent - friend_spent) = 1 :=
by
  sorry

end NUMINAMATH_GPT_friend_spent_more_l1474_147496


namespace NUMINAMATH_GPT_chenny_friends_count_l1474_147449

def initial_candies := 10
def additional_candies := 4
def candies_per_friend := 2

theorem chenny_friends_count :
  (initial_candies + additional_candies) / candies_per_friend = 7 := by
  sorry

end NUMINAMATH_GPT_chenny_friends_count_l1474_147449


namespace NUMINAMATH_GPT_animals_per_aquarium_l1474_147473

theorem animals_per_aquarium (total_animals : ℕ) (number_of_aquariums : ℕ) (h1 : total_animals = 40) (h2 : number_of_aquariums = 20) : 
  total_animals / number_of_aquariums = 2 :=
by
  sorry

end NUMINAMATH_GPT_animals_per_aquarium_l1474_147473


namespace NUMINAMATH_GPT_solve_inequality_l1474_147422

theorem solve_inequality (x : ℝ) : 
  (3 * x^2 - 5 * x + 2 > 0) ↔ (x < 2 / 3 ∨ x > 1) := 
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1474_147422


namespace NUMINAMATH_GPT_no_seating_in_four_consecutive_seats_l1474_147468

theorem no_seating_in_four_consecutive_seats :
  let total_arrangements := Nat.factorial 10
  let grouped_arrangements := Nat.factorial 7 * Nat.factorial 4
  let acceptable_arrangements := total_arrangements - grouped_arrangements
  acceptable_arrangements = 3507840 :=
by
  sorry

end NUMINAMATH_GPT_no_seating_in_four_consecutive_seats_l1474_147468


namespace NUMINAMATH_GPT_shopkeeper_weight_l1474_147438

/-- A shopkeeper sells his goods at cost price but uses a certain weight instead of kilogram weight.
    His profit percentage is 25%. Prove that the weight he uses is 0.8 kilograms. -/
theorem shopkeeper_weight (c s p : ℝ) (x : ℝ) (h1 : s = c * (1 + p / 100))
  (h2 : p = 25) (h3 : c = 1) (h4 : s = 1.25) : x = 0.8 :=
by
  sorry

end NUMINAMATH_GPT_shopkeeper_weight_l1474_147438


namespace NUMINAMATH_GPT_books_sold_l1474_147434

theorem books_sold (original_books : ℕ) (remaining_books : ℕ) (sold_books : ℕ) 
  (h1 : original_books = 51) 
  (h2 : remaining_books = 6) 
  (h3 : sold_books = original_books - remaining_books) : 
  sold_books = 45 :=
by 
  sorry

end NUMINAMATH_GPT_books_sold_l1474_147434


namespace NUMINAMATH_GPT_how_many_both_books_l1474_147483

-- Definitions based on the conditions
def total_workers : ℕ := 40
def saramago_workers : ℕ := total_workers / 4
def kureishi_workers : ℕ := (total_workers * 5) / 8
def both_books (B : ℕ) : Prop :=
  B + (saramago_workers - B) + (kureishi_workers - B) + (9 - B) = total_workers

theorem how_many_both_books : ∃ B : ℕ, both_books B ∧ B = 4 := by
  use 4
  -- Proof goes here, skipped by using sorry
  sorry

end NUMINAMATH_GPT_how_many_both_books_l1474_147483


namespace NUMINAMATH_GPT_log5_x_equals_neg_two_log5_2_l1474_147406

theorem log5_x_equals_neg_two_log5_2 (x : ℝ) (h : x = (Real.log 3 / Real.log 9) ^ (Real.log 9 / Real.log 3)) :
  Real.log x / Real.log 5 = -2 * (Real.log 2 / Real.log 5) :=
by
  sorry

end NUMINAMATH_GPT_log5_x_equals_neg_two_log5_2_l1474_147406


namespace NUMINAMATH_GPT_exponent_fraction_equals_five_fourths_l1474_147466

theorem exponent_fraction_equals_five_fourths :
  (3^2016 + 3^2014) / (3^2016 - 3^2014) = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_exponent_fraction_equals_five_fourths_l1474_147466


namespace NUMINAMATH_GPT_sarahs_monthly_fee_l1474_147440

noncomputable def fixed_monthly_fee (x y : ℝ) : Prop :=
  x + 4 * y = 30.72 ∧ 1.1 * x + 8 * y = 54.72

theorem sarahs_monthly_fee : ∃ x y : ℝ, fixed_monthly_fee x y ∧ x = 7.47 :=
by
  sorry

end NUMINAMATH_GPT_sarahs_monthly_fee_l1474_147440


namespace NUMINAMATH_GPT_max_min_sums_l1474_147499

def P (x y : ℤ) := x^2 + y^2 = 50

theorem max_min_sums : 
  ∃ (x₁ y₁ x₂ y₂ : ℤ), P x₁ y₁ ∧ P x₂ y₂ ∧ 
    (x₁ + y₁ = 8) ∧ (x₂ + y₂ = -8) :=
by
  sorry

end NUMINAMATH_GPT_max_min_sums_l1474_147499


namespace NUMINAMATH_GPT_parabola_focus_l1474_147442

theorem parabola_focus (x : ℝ) : ∃ f : ℝ × ℝ, f = (0, 1 / 4) ∧ ∀ y : ℝ, y = x^2 → f = (0, 1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_l1474_147442


namespace NUMINAMATH_GPT_throws_to_return_to_elsa_l1474_147486

theorem throws_to_return_to_elsa :
  ∃ n, n = 5 ∧ (∀ (k : ℕ), k < n → ((1 + 5 * k) % 13 ≠ 1)) ∧ (1 + 5 * n) % 13 = 1 :=
by
  sorry

end NUMINAMATH_GPT_throws_to_return_to_elsa_l1474_147486


namespace NUMINAMATH_GPT_α_plus_β_eq_two_l1474_147429

noncomputable def α : ℝ := sorry
noncomputable def β : ℝ := sorry

theorem α_plus_β_eq_two
  (hα : α^3 - 3*α^2 + 5*α - 4 = 0)
  (hβ : β^3 - 3*β^2 + 5*β - 2 = 0) :
  α + β = 2 := 
sorry

end NUMINAMATH_GPT_α_plus_β_eq_two_l1474_147429


namespace NUMINAMATH_GPT_focus_of_parabola_l1474_147410

-- Problem statement
theorem focus_of_parabola (x y : ℝ) : (2 * x^2 = -y) → (focus_coordinates = (0, -1 / 8)) :=
by
  sorry

end NUMINAMATH_GPT_focus_of_parabola_l1474_147410


namespace NUMINAMATH_GPT_packing_objects_in_boxes_l1474_147436

theorem packing_objects_in_boxes 
  (n k : ℕ) (n_pos : 0 < n) (k_pos : 0 < k) 
  (objects : Fin (n * k) → Fin k) 
  (boxes : Fin k → Fin n → Fin k) :
  ∃ (pack : Fin (n * k) → Fin k), 
    (∀ i, ∃ c1 c2, 
      ∀ j, pack i = pack j → 
      (objects i = c1 ∨ objects i = c2 ∧
      objects j = c1 ∨ objects j = c2)) := 
sorry

end NUMINAMATH_GPT_packing_objects_in_boxes_l1474_147436


namespace NUMINAMATH_GPT_determine_a7_l1474_147458

noncomputable def arithmetic_seq (a1 d : ℤ) : ℕ → ℤ
| 0     => a1
| (n+1) => a1 + n * d

noncomputable def sum_arithmetic_seq (a1 d : ℤ) : ℕ → ℤ
| 0     => 0
| (n+1) => a1 * (n + 1) + (n * (n + 1) * d) / 2

theorem determine_a7 (a1 d : ℤ) (a2 : a1 + d = 7) (S7 : sum_arithmetic_seq a1 d 7 = -7) : arithmetic_seq a1 d 7 = -13 :=
by
  sorry

end NUMINAMATH_GPT_determine_a7_l1474_147458


namespace NUMINAMATH_GPT_angle_of_elevation_proof_l1474_147411

noncomputable def height_of_lighthouse : ℝ := 100

noncomputable def distance_between_ships : ℝ := 273.2050807568877

noncomputable def angle_of_elevation_second_ship : ℝ := 45

noncomputable def distance_from_second_ship := height_of_lighthouse

noncomputable def distance_from_first_ship := distance_between_ships - distance_from_second_ship

noncomputable def tanθ := height_of_lighthouse / distance_from_first_ship

noncomputable def angle_of_elevation_first_ship := Real.arctan tanθ

theorem angle_of_elevation_proof :
  angle_of_elevation_first_ship = 30 := by
    sorry

end NUMINAMATH_GPT_angle_of_elevation_proof_l1474_147411


namespace NUMINAMATH_GPT_quadratic_has_only_positive_roots_l1474_147476

theorem quadratic_has_only_positive_roots (m : ℝ) :
  (∀ (x : ℝ), x^2 + (m + 2) * x + (m + 5) = 0 → x > 0) →
  -5 < m ∧ m ≤ -4 :=
by 
  -- added sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_quadratic_has_only_positive_roots_l1474_147476


namespace NUMINAMATH_GPT_sum_geometric_sequence_l1474_147405

theorem sum_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n, a (n + 1) = a n * a 1)
  (h_a1 : a 1 = 1)
  (h_arithmetic : 4 * a 2 + a 4 = 2 * a 3) : 
  a 2 + a 3 + a 4 = 14 :=
sorry

end NUMINAMATH_GPT_sum_geometric_sequence_l1474_147405


namespace NUMINAMATH_GPT_arccos_neg_one_eq_pi_l1474_147493

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
by
  sorry

end NUMINAMATH_GPT_arccos_neg_one_eq_pi_l1474_147493


namespace NUMINAMATH_GPT_amount_daria_needs_l1474_147474

theorem amount_daria_needs (ticket_cost : ℕ) (total_tickets : ℕ) (current_money : ℕ) (needed_money : ℕ) 
  (h1 : ticket_cost = 90) 
  (h2 : total_tickets = 4) 
  (h3 : current_money = 189) 
  (h4 : needed_money = 360 - 189) 
  : needed_money = 171 := by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_amount_daria_needs_l1474_147474


namespace NUMINAMATH_GPT_solution_of_inequality_l1474_147453

theorem solution_of_inequality (a b : ℝ) (h : ∀ x : ℝ, (1 < x ∧ x < 3) ↔ (x^2 < a * x + b)) :
  b^a = 81 := 
sorry

end NUMINAMATH_GPT_solution_of_inequality_l1474_147453


namespace NUMINAMATH_GPT_customer_initial_amount_l1474_147423

theorem customer_initial_amount (d c : ℕ) (h1 : c = 100 * d) (h2 : c = 2 * d) : d = 0 ∧ c = 0 := by
  sorry

end NUMINAMATH_GPT_customer_initial_amount_l1474_147423


namespace NUMINAMATH_GPT_contrapositive_p_l1474_147481

-- Definitions
def A_score := 70
def B_score := 70
def C_score := 65
def p := ∀ (passing_score : ℕ), passing_score < 70 → (A_score < passing_score ∧ B_score < passing_score ∧ C_score < passing_score)

-- Statement to be proved
theorem contrapositive_p : 
  ∀ (passing_score : ℕ), (A_score ≥ passing_score ∨ B_score ≥ passing_score ∨ C_score ≥ passing_score) → (¬ passing_score < 70) := 
by
  sorry

end NUMINAMATH_GPT_contrapositive_p_l1474_147481


namespace NUMINAMATH_GPT_total_distance_race_l1474_147477

theorem total_distance_race
  (t_Sadie : ℝ) (s_Sadie : ℝ) (t_Ariana : ℝ) (s_Ariana : ℝ) 
  (s_Sarah : ℝ) (tt : ℝ)
  (h_Sadie : t_Sadie = 2) (hs_Sadie : s_Sadie = 3) 
  (h_Ariana : t_Ariana = 0.5) (hs_Ariana : s_Ariana = 6) 
  (hs_Sarah : s_Sarah = 4)
  (h_tt : tt = 4.5) : 
  (s_Sadie * t_Sadie + s_Ariana * t_Ariana + s_Sarah * (tt - (t_Sadie + t_Ariana))) = 17 := 
  by {
    sorry -- proof goes here
  }

end NUMINAMATH_GPT_total_distance_race_l1474_147477


namespace NUMINAMATH_GPT_sin_330_eq_neg_one_half_l1474_147487

theorem sin_330_eq_neg_one_half : 
  Real.sin (330 * Real.pi / 180) = -1 / 2 := 
sorry

end NUMINAMATH_GPT_sin_330_eq_neg_one_half_l1474_147487


namespace NUMINAMATH_GPT_satisfy_conditions_l1474_147457

variable (x : ℝ)

theorem satisfy_conditions :
  (3 * x^2 + 4 * x - 9 < 0) ∧ (x ≥ -2) ↔ (-2 ≤ x ∧ x < 1) := by
  sorry

end NUMINAMATH_GPT_satisfy_conditions_l1474_147457


namespace NUMINAMATH_GPT_earnings_difference_l1474_147427

theorem earnings_difference (x y : ℕ) 
  (h1 : 3 * 6 + 4 * 5 + 5 * 4 = 58)
  (h2 : x * y = 12500) 
  (total_earnings : (3 * 6 * x * y / 100 + 4 * 5 * x * y / 100 + 5 * 4 * x * y / 100) = 7250) :
  4 * 5 * x * y / 100 - 3 * 6 * x * y / 100 = 250 := 
by 
  sorry

end NUMINAMATH_GPT_earnings_difference_l1474_147427


namespace NUMINAMATH_GPT_no_three_digit_numbers_meet_conditions_l1474_147460

theorem no_three_digit_numbers_meet_conditions :
  ∀ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (n % 10 = 5) ∧ (n % 10 = 0) → false := 
by {
  sorry
}

end NUMINAMATH_GPT_no_three_digit_numbers_meet_conditions_l1474_147460


namespace NUMINAMATH_GPT_find_two_digit_integers_l1474_147412

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem find_two_digit_integers
    (a b : ℕ) :
    10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ 
    (a = b + 12 ∨ b = a + 12) ∧
    (a / 10 = b / 10 ∨ a % 10 = b % 10) ∧
    (sum_of_digits a = sum_of_digits b + 3 ∨ sum_of_digits b = sum_of_digits a + 3) :=
sorry

end NUMINAMATH_GPT_find_two_digit_integers_l1474_147412


namespace NUMINAMATH_GPT_g_g_is_odd_l1474_147400

def f (x : ℝ) : ℝ := x^3

def g (x : ℝ) : ℝ := f (f x)

theorem g_g_is_odd : ∀ x : ℝ, g (g (-x)) = -g (g x) :=
by 
-- proof will go here
sorry

end NUMINAMATH_GPT_g_g_is_odd_l1474_147400


namespace NUMINAMATH_GPT_sandwich_cost_90_cents_l1474_147451

def sandwich_cost (bread_cost ham_cost cheese_cost : ℕ) : ℕ :=
  2 * bread_cost + ham_cost + cheese_cost

theorem sandwich_cost_90_cents :
  sandwich_cost 15 25 35 = 90 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sandwich_cost_90_cents_l1474_147451


namespace NUMINAMATH_GPT_distance_home_to_school_l1474_147421

theorem distance_home_to_school :
  ∃ T D : ℝ, 6 * (T + 7/60) = D ∧ 12 * (T - 8/60) = D ∧ 9 * T = D ∧ D = 2.1 :=
by
  sorry

end NUMINAMATH_GPT_distance_home_to_school_l1474_147421


namespace NUMINAMATH_GPT_sum_of_distances_l1474_147418

theorem sum_of_distances (d_1 d_2 : ℝ) (h1 : d_2 = d_1 + 5) (h2 : d_1 + d_2 = 13) :
  d_1 + d_2 = 13 :=
by sorry

end NUMINAMATH_GPT_sum_of_distances_l1474_147418


namespace NUMINAMATH_GPT_combined_share_is_50000_l1474_147414

def profit : ℝ := 80000

def majority_owner_share : ℝ := 0.25 * profit

def remaining_profit : ℝ := profit - majority_owner_share

def partner_share : ℝ := 0.25 * remaining_profit

def combined_share_majority_two_owners : ℝ := majority_owner_share + 2 * partner_share

theorem combined_share_is_50000 :
  combined_share_majority_two_owners = 50000 := 
by 
  sorry

end NUMINAMATH_GPT_combined_share_is_50000_l1474_147414


namespace NUMINAMATH_GPT_lily_typing_break_time_l1474_147428

theorem lily_typing_break_time :
  ∃ t : ℝ, (15 * t + 15 * t = 255) ∧ (19 = 2 * t + 2) ∧ (t = 8) := 
sorry

end NUMINAMATH_GPT_lily_typing_break_time_l1474_147428


namespace NUMINAMATH_GPT_goldfish_equal_after_8_months_l1474_147475

noncomputable def B (n : ℕ) : ℝ := 3^(n + 1)
noncomputable def G (n : ℕ) : ℝ := 243 * 1.5^n

theorem goldfish_equal_after_8_months :
  ∃ n : ℕ, B n = G n ∧ n = 8 :=
by
  sorry

end NUMINAMATH_GPT_goldfish_equal_after_8_months_l1474_147475


namespace NUMINAMATH_GPT_sequence_geometric_and_sum_l1474_147445

variables {S : ℕ → ℝ} (a1 : S 1 = 1)
variable (n : ℕ)
def a := (S (n+1) - 2 * S n, S n)
def b := (2, n)

/-- Prove that the sequence {S n / n} is a geometric sequence 
with first term 1 and common ratio 2, and find the sum of the first 
n terms of the sequence {S n} -/
theorem sequence_geometric_and_sum {S : ℕ → ℝ} (a1 : S 1 = 1)
  (n : ℕ)
  (parallel : ∀ n, n * (S (n + 1) - 2 * S n) = 2 * S n) :
  ∃ r : ℝ, r = 2 ∧ ∃ T : ℕ → ℝ, T n = (n-1)*2^n + 1 :=
by
  sorry

end NUMINAMATH_GPT_sequence_geometric_and_sum_l1474_147445


namespace NUMINAMATH_GPT_students_playing_long_tennis_l1474_147433

theorem students_playing_long_tennis (n F B N L : ℕ)
  (h1 : n = 35)
  (h2 : F = 26)
  (h3 : B = 17)
  (h4 : N = 6)
  (h5 : L = (n - N) - (F - B)) :
  L = 20 :=
by
  sorry

end NUMINAMATH_GPT_students_playing_long_tennis_l1474_147433


namespace NUMINAMATH_GPT_find_x2_l1474_147472

theorem find_x2 (x1 x2 x3 : ℝ) (h1 : x1 + x2 = 14) (h2 : x1 + x3 = 17) (h3 : x2 + x3 = 33) : x2 = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_x2_l1474_147472


namespace NUMINAMATH_GPT_mary_received_more_l1474_147416

theorem mary_received_more (investment_Mary investment_Harry profit : ℤ)
  (one_third_profit divided_equally remaining_profit : ℤ)
  (total_Mary total_Harry difference : ℤ)
  (investment_ratio_Mary investment_ratio_Harry : ℚ) :
  investment_Mary = 700 →
  investment_Harry = 300 →
  profit = 3000 →
  one_third_profit = profit / 3 →
  divided_equally = one_third_profit / 2 →
  remaining_profit = profit - one_third_profit →
  investment_ratio_Mary = 7/10 →
  investment_ratio_Harry = 3/10 →
  total_Mary = divided_equally + investment_ratio_Mary * remaining_profit →
  total_Harry = divided_equally + investment_ratio_Harry * remaining_profit →
  difference = total_Mary - total_Harry →
  difference = 800 := by
  sorry

end NUMINAMATH_GPT_mary_received_more_l1474_147416


namespace NUMINAMATH_GPT_problem_statement_l1474_147407

theorem problem_statement (x : ℝ) (h : 2 * x^2 + 1 = 17) : 4 * x^2 + 1 = 33 :=
by sorry

end NUMINAMATH_GPT_problem_statement_l1474_147407


namespace NUMINAMATH_GPT_kim_pairs_of_shoes_l1474_147420

theorem kim_pairs_of_shoes : ∃ n : ℕ, 2 * n + 1 = 14 ∧ (1 : ℚ) / (2 * n - 1) = (0.07692307692307693 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_kim_pairs_of_shoes_l1474_147420


namespace NUMINAMATH_GPT_haley_money_l1474_147461

variable (x : ℕ)

def initial_amount : ℕ := 2
def difference : ℕ := 11
def total_amount (x : ℕ) : ℕ := x

theorem haley_money : total_amount x - initial_amount = difference → total_amount x = 13 := by
  sorry

end NUMINAMATH_GPT_haley_money_l1474_147461


namespace NUMINAMATH_GPT_triangle_side_ratio_l1474_147465

variables (a b c S : ℝ)
variables (A B C : ℝ)

/-- In triangle ABC, if the sides opposite to angles A, B, and C are a, b, and c respectively,
    and given a=1, B=π/4, and the area S=2, we prove that b / sin(B) = 5√2. -/
theorem triangle_side_ratio (h₁ : a = 1) (h₂ : B = Real.pi / 4) (h₃ : S = 2) : b / Real.sin B = 5 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_triangle_side_ratio_l1474_147465


namespace NUMINAMATH_GPT_alcohol_water_ratio_l1474_147470

theorem alcohol_water_ratio (a b : ℚ) (h₁ : a = 3/5) (h₂ : b = 2/5) : a / b = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_alcohol_water_ratio_l1474_147470


namespace NUMINAMATH_GPT_number_of_rabbits_l1474_147452

theorem number_of_rabbits (C D : ℕ) (hC : C = 49) (hD : D = 37) (h : D + R = C + 9) :
  R = 21 :=
by
    sorry

end NUMINAMATH_GPT_number_of_rabbits_l1474_147452


namespace NUMINAMATH_GPT_circle_equation_l1474_147495

-- Defining the points A and B
def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (1, 3)

-- Defining the center M of the circle on the x-axis
def M (a : ℝ) : ℝ × ℝ := (a, 0)

-- Defining the squared distance function between two points
def dist_sq (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2

-- Statement: Prove that the standard equation of the circle is (x - 2)² + y² = 10
theorem circle_equation : ∃ a : ℝ, (dist_sq (M a) A = dist_sq (M a) B) ∧ ((M a).1 = 2) ∧ (dist_sq (M a) A = 10) :=
sorry

end NUMINAMATH_GPT_circle_equation_l1474_147495


namespace NUMINAMATH_GPT_fraction_saved_l1474_147492

-- Definitions and given conditions
variables {P : ℝ} {f : ℝ}

-- Worker saves the same fraction each month, the same take-home pay each month
-- Total annual savings = 12fP and total annual savings = 2 * (amount not saved monthly)
theorem fraction_saved (h : 12 * f * P = 2 * (1 - f) * P) (P_ne_zero : P ≠ 0) : f = 1 / 7 :=
by
  -- The proof of the theorem goes here
  sorry

end NUMINAMATH_GPT_fraction_saved_l1474_147492


namespace NUMINAMATH_GPT_spherical_to_rectangular_correct_l1474_147450

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z)

theorem spherical_to_rectangular_correct : spherical_to_rectangular 4 (Real.pi / 6) (Real.pi / 3) = (3, Real.sqrt 3, 2) :=
by
  sorry

end NUMINAMATH_GPT_spherical_to_rectangular_correct_l1474_147450


namespace NUMINAMATH_GPT_product_of_conversions_l1474_147403

-- Define the binary number 1101
def binary_number := 1101

-- Convert binary 1101 to decimal
def binary_to_decimal : ℕ := 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0

-- Define the ternary number 212
def ternary_number := 212

-- Convert ternary 212 to decimal
def ternary_to_decimal : ℕ := 2 * 3^2 + 1 * 3^1 + 2 * 3^0

-- Statement to prove
theorem product_of_conversions : (binary_to_decimal) * (ternary_to_decimal) = 299 := by
  sorry

end NUMINAMATH_GPT_product_of_conversions_l1474_147403


namespace NUMINAMATH_GPT_age_ratio_l1474_147448

theorem age_ratio (S : ℕ) (M : ℕ) (h1 : S = 28) (h2 : M = S + 30) : 
  ((M + 2) / (S + 2) = 2) := 
by
  sorry

end NUMINAMATH_GPT_age_ratio_l1474_147448


namespace NUMINAMATH_GPT_proof_expression_l1474_147484

open Real

theorem proof_expression (x y : ℝ) (h1 : P = 2 * (x + y)) (h2 : Q = 3 * (x - y)) :
  (P + Q) / (P - Q) - (P - Q) / (P + Q) + (x + y) / (x - y) = (28 * x^2 - 20 * y^2) / ((x - y) * (5 * x - y) * (-x + 5 * y)) :=
by
  sorry

end NUMINAMATH_GPT_proof_expression_l1474_147484


namespace NUMINAMATH_GPT_compound_interest_rate_l1474_147402

theorem compound_interest_rate
(SI : ℝ) (CI : ℝ) (P1 : ℝ) (r : ℝ) (t1 t2 : ℕ) (P2 R : ℝ)
(h1 : SI = (P1 * r * t1) / 100)
(h2 : SI = CI / 2)
(h3 : CI = P2 * (1 + R / 100) ^ t2 - P2)
(h4 : P1 = 3500)
(h5 : r = 6)
(h6 : t1 = 2)
(h7 : P2 = 4000)
(h8 : t2 = 2) : R = 10 := by
  sorry

end NUMINAMATH_GPT_compound_interest_rate_l1474_147402


namespace NUMINAMATH_GPT_total_cost_of_purchase_l1474_147425

theorem total_cost_of_purchase :
  let sandwich_cost := 3
  let soda_cost := 2
  let num_sandwiches := 5
  let num_sodas := 8
  let total_cost := (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost)
  total_cost = 31 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_purchase_l1474_147425


namespace NUMINAMATH_GPT_inheritance_problem_l1474_147498

variables (x1 x2 x3 x4 : ℕ)

theorem inheritance_problem
  (h1 : x1 + x2 + x3 + x4 = 1320)
  (h2 : x1 + x4 = x2 + x3)
  (h3 : x2 + x4 = 2 * (x1 + x3))
  (h4 : x3 + x4 = 3 * (x1 + x2)) :
  x1 = 55 ∧ x2 = 275 ∧ x3 = 385 ∧ x4 = 605 :=
by sorry

end NUMINAMATH_GPT_inheritance_problem_l1474_147498


namespace NUMINAMATH_GPT_MutualExclusivity_Of_A_C_l1474_147467

-- Definitions of events using conditions from a)
def EventA (products : List Bool) : Prop :=
  products.all (λ p => p = true)

def EventB (products : List Bool) : Prop :=
  products.all (λ p => p = false)

def EventC (products : List Bool) : Prop :=
  products.any (λ p => p = false)

-- The main theorem using correct answer from b)
theorem MutualExclusivity_Of_A_C (products : List Bool) :
  EventA products → ¬ EventC products :=
by
  sorry

end NUMINAMATH_GPT_MutualExclusivity_Of_A_C_l1474_147467


namespace NUMINAMATH_GPT_Debby_daily_bottles_is_six_l1474_147446

def daily_bottles (total_bottles : ℕ) (total_days : ℕ) : ℕ :=
  total_bottles / total_days

theorem Debby_daily_bottles_is_six : daily_bottles 12 2 = 6 := by
  sorry

end NUMINAMATH_GPT_Debby_daily_bottles_is_six_l1474_147446


namespace NUMINAMATH_GPT_sequence_periodic_mod_l1474_147413

-- Define the sequence (u_n) recursively
def sequence_u (a : ℕ) : ℕ → ℕ
  | 0     => a  -- Note: u_1 is defined as the initial term a, treating the starting index as 0 for compatibility with Lean's indexing.
  | (n+1) => a ^ (sequence_u a n)

-- The theorem stating there exist integers k and N such that for all n ≥ N, u_{n+k} ≡ u_n (mod m)
theorem sequence_periodic_mod (a m : ℕ) (hm : 0 < m) (ha : 0 < a) :
  ∃ k N : ℕ, ∀ n : ℕ, N ≤ n → (sequence_u a (n + k) ≡ sequence_u a n [MOD m]) :=
by
  sorry

end NUMINAMATH_GPT_sequence_periodic_mod_l1474_147413


namespace NUMINAMATH_GPT_greatest_n_4022_l1474_147480

noncomputable def arithmetic_sequence_greatest_n 
  (a : ℕ → ℝ)
  (a1_pos : a 1 > 0)
  (cond1 : a 2011 + a 2012 > 0)
  (cond2 : a 2011 * a 2012 < 0) : ℕ :=
  4022

theorem greatest_n_4022 
  (a : ℕ → ℝ)
  (h_arith : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h1 : a 1 > 0)
  (h2 : a 2011 + a 2012 > 0)
  (h3 : a 2011 * a 2012 < 0):
  arithmetic_sequence_greatest_n a h1 h2 h3 = 4022 :=
sorry

end NUMINAMATH_GPT_greatest_n_4022_l1474_147480


namespace NUMINAMATH_GPT_simplify_polynomial_problem_l1474_147435

theorem simplify_polynomial_problem (p : ℝ) :
  (5 * p^4 - 4 * p^3 + 3 * p + 2) + (-3 * p^4 + 2 * p^3 - 7 * p^2 + 8) = 2 * p^4 - 2 * p^3 - 7 * p^2 + 3 * p + 10 := 
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_problem_l1474_147435


namespace NUMINAMATH_GPT_min_possible_A_div_C_l1474_147443

theorem min_possible_A_div_C (x : ℝ) (A C : ℝ) (h1 : x^2 + (1/x)^2 = A) (h2 : x + 1/x = C) (h3 : 0 < A) (h4 : 0 < C) :
  ∃ (C : ℝ), C = Real.sqrt 2 ∧ (∀ B, (x^2 + (1/x)^2 = B) → (x + 1/x = C) → (B / C = 0 → B = 0)) :=
by
  sorry

end NUMINAMATH_GPT_min_possible_A_div_C_l1474_147443


namespace NUMINAMATH_GPT_first_divisor_l1474_147441

theorem first_divisor (k : ℤ) (h1 : k % 5 = 2) (h2 : k % 6 = 5) (h3 : k % 7 = 3) (h4 : k < 42) (hk : k = 17) : 5 ≤ 6 ∧ 5 ≤ 7 ∧ 5 = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_first_divisor_l1474_147441


namespace NUMINAMATH_GPT_negation_proposition_l1474_147419

theorem negation_proposition (p : Prop) : 
  (∀ x : ℝ, 2 * x^2 - 1 > 0) ↔ ¬ (∃ x : ℝ, 2 * x^2 - 1 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_l1474_147419


namespace NUMINAMATH_GPT_gcd_lcm_lemma_l1474_147482

theorem gcd_lcm_lemma (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 33) (h_lcm : Nat.lcm a b = 90) : Nat.gcd a b = 3 :=
by
  sorry

end NUMINAMATH_GPT_gcd_lcm_lemma_l1474_147482


namespace NUMINAMATH_GPT_car_meeting_points_l1474_147430

-- Define the conditions for the problem
variables {A B : ℝ}
variables {speed_ratio : ℝ} (ratio_pos : speed_ratio = 5 / 4)
variables {T1 T2 : ℝ} (T1_pos : T1 = 145) (T2_pos : T2 = 201)

-- The proof problem statement
theorem car_meeting_points (A B : ℝ) (ratio_pos : speed_ratio = 5 / 4) 
  (T1 T2 : ℝ) (T1_pos : T1 = 145) (T2_pos : T2 = 201) :
  A = 103 ∧ B = 229 :=
sorry

end NUMINAMATH_GPT_car_meeting_points_l1474_147430


namespace NUMINAMATH_GPT_volume_calculation_l1474_147424

-- Define the dimensions of the rectangular parallelepiped
def a : ℕ := 2
def b : ℕ := 3
def c : ℕ := 4

-- Define the radius for spheres and cylinders
def r : ℝ := 2

theorem volume_calculation : 
  let l := 384
  let o := 140
  let q := 3
  (l + o + q = 527) :=
by
  sorry

end NUMINAMATH_GPT_volume_calculation_l1474_147424


namespace NUMINAMATH_GPT_total_red_and_green_peaches_l1474_147485

def red_peaches : ℕ := 6
def green_peaches : ℕ := 16

theorem total_red_and_green_peaches :
  red_peaches + green_peaches = 22 :=
  by 
    sorry

end NUMINAMATH_GPT_total_red_and_green_peaches_l1474_147485


namespace NUMINAMATH_GPT_proof_l1474_147488

noncomputable def M : Set ℝ := {x | 1 - (2 / x) > 0}
noncomputable def N : Set ℝ := {x | x ≥ 1}

theorem proof : (Mᶜ ∪ N) = {x | x ≥ 0} := sorry

end NUMINAMATH_GPT_proof_l1474_147488


namespace NUMINAMATH_GPT_sum_of_prime_factors_is_prime_l1474_147471

/-- Define the specific number in question -/
def num := 30030

/-- List the prime factors of the number -/
def prime_factors := [2, 3, 5, 7, 11, 13]

/-- Sum of the prime factors -/
def sum_prime_factors := prime_factors.sum

theorem sum_of_prime_factors_is_prime :
  sum_prime_factors = 41 ∧ Prime 41 := 
by
  -- The conditions are encapsulated in the definitions above
  -- Now, establish the required proof goal using these conditions
  sorry

end NUMINAMATH_GPT_sum_of_prime_factors_is_prime_l1474_147471


namespace NUMINAMATH_GPT_cost_of_marker_l1474_147447

theorem cost_of_marker (n m : ℝ) (h1 : 3 * n + 2 * m = 7.45) (h2 : 4 * n + 3 * m = 10.40) : m = 1.40 :=
  sorry

end NUMINAMATH_GPT_cost_of_marker_l1474_147447


namespace NUMINAMATH_GPT_angle_A_measure_l1474_147456

theorem angle_A_measure (A B C D E : ℝ) 
(h1 : A = 3 * B)
(h2 : A = 4 * C)
(h3 : A = 5 * D)
(h4 : A = 6 * E)
(h5 : A + B + C + D + E = 540) : 
A = 277 :=
by
  sorry

end NUMINAMATH_GPT_angle_A_measure_l1474_147456


namespace NUMINAMATH_GPT_num_int_values_N_l1474_147444

theorem num_int_values_N (N : ℕ) : 
  (∃ M, M ∣ 72 ∧ M > 3 ∧ N = M - 3) ↔ N ∈ ({1, 3, 5, 6, 9, 15, 21, 33, 69} : Finset ℕ) :=
by
  sorry

end NUMINAMATH_GPT_num_int_values_N_l1474_147444


namespace NUMINAMATH_GPT_trapezoid_prob_l1474_147454

noncomputable def trapezoid_probability_not_below_x_axis : ℝ :=
  let P := (4, 4)
  let Q := (-4, -4)
  let R := (-10, -4)
  let S := (-2, 4)
  -- Coordinates of intersection points
  let T := (0, 0)
  let U := (-6, 0)
  -- Compute the probability
  (16 * Real.sqrt 2 + 16) / (16 * Real.sqrt 2 + 40)

theorem trapezoid_prob :
  trapezoid_probability_not_below_x_axis = (16 * Real.sqrt 2 + 16) / (16 * Real.sqrt 2 + 40) :=
sorry

end NUMINAMATH_GPT_trapezoid_prob_l1474_147454


namespace NUMINAMATH_GPT_annie_miles_l1474_147490

theorem annie_miles (x : ℝ) :
  2.50 + (0.25 * 42) = 2.50 + 5.00 + (0.25 * x) → x = 22 :=
by
  sorry

end NUMINAMATH_GPT_annie_miles_l1474_147490


namespace NUMINAMATH_GPT_set_intersection_eq_l1474_147497

theorem set_intersection_eq (M N : Set ℝ) (hM : M = { x : ℝ | 0 < x ∧ x < 1 }) (hN : N = { x : ℝ | -2 < x ∧ x < 2 }) :
  M ∩ N = M :=
sorry

end NUMINAMATH_GPT_set_intersection_eq_l1474_147497


namespace NUMINAMATH_GPT_binary_op_property_l1474_147426

variable (X : Type)
variable (star : X → X → X)
variable (h : ∀ x y : X, star (star x y) x = y)

theorem binary_op_property (x y : X) : star x (star y x) = y := 
by 
  sorry

end NUMINAMATH_GPT_binary_op_property_l1474_147426


namespace NUMINAMATH_GPT_incorrect_statement_d_l1474_147478

variable (x : ℝ)
variables (p q : Prop)

-- Proving D is incorrect given defined conditions
theorem incorrect_statement_d :
  ∀ (x : ℝ), (¬ (x = 1) → ¬ (x^2 - 3 * x + 2 = 0)) ∧
  ((x > 2) → (x^2 - 3 * x + 2 > 0) ∧
  (¬ (x^2 + x + 1 = 0))) ∧
  ((p ∨ q) → ¬ (p ∧ q)) :=
by
  -- A detailed proof would be required here
  sorry

end NUMINAMATH_GPT_incorrect_statement_d_l1474_147478


namespace NUMINAMATH_GPT_paper_area_difference_l1474_147432

def sheet1_length : ℕ := 14
def sheet1_width : ℕ := 12
def sheet2_length : ℕ := 9
def sheet2_width : ℕ := 14

def area (length : ℕ) (width : ℕ) : ℕ := length * width

def combined_area (length : ℕ) (width : ℕ) : ℕ := 2 * area length width

theorem paper_area_difference :
  combined_area sheet1_length sheet1_width - combined_area sheet2_length sheet2_width = 84 := 
by 
  sorry

end NUMINAMATH_GPT_paper_area_difference_l1474_147432


namespace NUMINAMATH_GPT_abs_fraction_eq_sqrt_three_over_two_l1474_147439

variable (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ a^2 + b^2 = 10 * a * b)

theorem abs_fraction_eq_sqrt_three_over_two (h : a ≠ 0 ∧ b ≠ 0 ∧ a^2 + b^2 = 10 * a * b) : 
  |(a + b) / (a - b)| = Real.sqrt (3 / 2) := by
  sorry

end NUMINAMATH_GPT_abs_fraction_eq_sqrt_three_over_two_l1474_147439


namespace NUMINAMATH_GPT_question_proof_l1474_147417

theorem question_proof (x y : ℝ) (h : x * (x + y) = x^2 + y + 12) : xy + y^2 = y^2 + y + 12 :=
by
  sorry

end NUMINAMATH_GPT_question_proof_l1474_147417


namespace NUMINAMATH_GPT_polynomial_value_l1474_147479
variable {x y : ℝ}
theorem polynomial_value (h : 3 * x^2 + 4 * y + 9 = 8) : 9 * x^2 + 12 * y + 8 = 5 :=
by
   sorry

end NUMINAMATH_GPT_polynomial_value_l1474_147479


namespace NUMINAMATH_GPT_xiaohongs_mother_deposit_l1474_147462

theorem xiaohongs_mother_deposit (x : ℝ) :
  x + x * 3.69 / 100 * 3 * (1 - 20 / 100) = 5442.8 :=
by
  sorry

end NUMINAMATH_GPT_xiaohongs_mother_deposit_l1474_147462


namespace NUMINAMATH_GPT_part_I_part_II_l1474_147431

def f (x a : ℝ) : ℝ := |x - 4 * a| + |x|

theorem part_I (a : ℝ) (h : -4 ≤ a ∧ a ≤ 4) :
  ∀ x : ℝ, f x a ≥ a^2 := 
sorry

theorem part_II (x y z : ℝ) (h : 4 * x + 2 * y + z = 4) :
  (x + y)^2 + y^2 + z^2 ≥ 16 / 21 :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l1474_147431


namespace NUMINAMATH_GPT_find_dividend_l1474_147489

-- Definitions from conditions
def divisor : ℕ := 14
def quotient : ℕ := 12
def remainder : ℕ := 8

-- The problem statement to prove
theorem find_dividend : (divisor * quotient + remainder) = 176 := by
  sorry

end NUMINAMATH_GPT_find_dividend_l1474_147489
