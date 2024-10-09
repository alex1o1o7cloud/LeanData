import Mathlib

namespace calculate_expression_l2164_216451

theorem calculate_expression : 12 * (1 / (2 / 3 - 1 / 4 + 1 / 6)) = 144 / 7 :=
by
  sorry

end calculate_expression_l2164_216451


namespace length_of_place_mat_l2164_216430

noncomputable def radius : ℝ := 6
noncomputable def width : ℝ := 1.5
def inner_corner_touch (n : ℕ) : Prop := n = 6

theorem length_of_place_mat (y : ℝ) (h1 : radius = 6) (h2 : width = 1.5) (h3 : inner_corner_touch 6) :
  y = (Real.sqrt 141.75 + 1.5) / 2 :=
sorry

end length_of_place_mat_l2164_216430


namespace Larry_spends_108_minutes_l2164_216477

-- Define conditions
def half_hour_twice_daily := 30 * 2
def fifth_of_an_hour_daily := 60 / 5
def quarter_hour_twice_daily := 15 * 2
def tenth_of_an_hour_daily := 60 / 10

-- Define total times spent on each pet
def total_time_dog := half_hour_twice_daily + fifth_of_an_hour_daily
def total_time_cat := quarter_hour_twice_daily + tenth_of_an_hour_daily

-- Define the total time spent on pets
def total_time_pets := total_time_dog + total_time_cat

-- Lean theorem statement
theorem Larry_spends_108_minutes : total_time_pets = 108 := 
  by 
    sorry

end Larry_spends_108_minutes_l2164_216477


namespace smallest_integer_solution_m_l2164_216405

theorem smallest_integer_solution_m :
  (∃ x y m : ℝ, 2 * x + y = 4 ∧ x + 2 * y = -3 * m + 2 ∧ x - y > -3/2) →
  ∃ m : ℤ, (∀ x y : ℝ, 2 * x + y = 4 ∧ x + 2 * y = -3 * m + 2 ∧ x - y > -3/2) ↔ m = -1 :=
by
  sorry

end smallest_integer_solution_m_l2164_216405


namespace free_time_left_after_cleaning_l2164_216489

-- Define the time it takes for each task
def vacuuming_time : ℤ := 45
def dusting_time : ℤ := 60
def mopping_time : ℤ := 30
def brushing_time_per_cat : ℤ := 5
def number_of_cats : ℤ := 3
def total_free_time_in_minutes : ℤ := 3 * 60 -- 3 hours converted to minutes

-- Define the total cleaning time
def total_cleaning_time : ℤ := vacuuming_time + dusting_time + mopping_time + (brushing_time_per_cat * number_of_cats)

-- Prove that the free time left after cleaning is 30 minutes
theorem free_time_left_after_cleaning : (total_free_time_in_minutes - total_cleaning_time) = 30 :=
by
  sorry

end free_time_left_after_cleaning_l2164_216489


namespace milkshakes_per_hour_l2164_216491

variable (L : ℕ) -- number of milkshakes Luna can make per hour

theorem milkshakes_per_hour
  (h1 : ∀ (A : ℕ), A = 3) -- Augustus makes 3 milkshakes per hour
  (h2 : ∀ (H : ℕ), H = 8) -- they have been making milkshakes for 8 hours
  (h3 : ∀ (Total : ℕ), Total = 80) -- together they made 80 milkshakes
  (h4 : ∀ (Augustus_milkshakes : ℕ), Augustus_milkshakes = 3 * 8) -- Augustus made 24 milkshakes in 8 hours
 : L = 7 := sorry

end milkshakes_per_hour_l2164_216491


namespace framed_painting_ratio_correct_l2164_216439

/-- Define the conditions -/
def painting_height : ℕ := 30
def painting_width : ℕ := 20
def width_ratio : ℕ := 3

/-- Calculate the framed dimensions and check the area conditions -/
def framed_smaller_dimension (x : ℕ) : ℕ := painting_width + 2 * x
def framed_larger_dimension (x : ℕ) : ℕ := painting_height + 6 * x

theorem framed_painting_ratio_correct (x : ℕ) (h : (painting_width + 2 * x) * (painting_height + 6 * x) = 2 * (painting_width * painting_height)) :
  framed_smaller_dimension x / framed_larger_dimension x = 4 / 7 :=
by
  sorry

end framed_painting_ratio_correct_l2164_216439


namespace solve_xy_eq_x_plus_y_l2164_216459

theorem solve_xy_eq_x_plus_y (x y : ℤ) (h : x * y = x + y) : (x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = 2) :=
by {
  sorry
}

end solve_xy_eq_x_plus_y_l2164_216459


namespace no_positive_integer_solutions_l2164_216449

theorem no_positive_integer_solutions (A : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) :
  ¬(∃ x : ℕ, x^2 - 2 * A * x + A0 = 0) :=
by sorry

end no_positive_integer_solutions_l2164_216449


namespace lcm_six_ten_fifteen_is_30_l2164_216483

-- Define the numbers and their prime factorizations
def six := 6
def ten := 10
def fifteen := 15

noncomputable def lcm_six_ten_fifteen : ℕ :=
  Nat.lcm (Nat.lcm six ten) fifteen

-- The theorem to prove the LCM
theorem lcm_six_ten_fifteen_is_30 : lcm_six_ten_fifteen = 30 :=
  sorry

end lcm_six_ten_fifteen_is_30_l2164_216483


namespace find_a_if_lines_perpendicular_l2164_216473

-- Define the lines and the statement about their perpendicularity
theorem find_a_if_lines_perpendicular 
    (a : ℝ)
    (h_perpendicular : (2 * a) / (3 * (a - 1)) = 1) :
    a = 3 :=
by
  sorry

end find_a_if_lines_perpendicular_l2164_216473


namespace polynomial_value_l2164_216414

theorem polynomial_value : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 :=
by
  sorry

end polynomial_value_l2164_216414


namespace gain_amount_l2164_216421

theorem gain_amount (gain_percent : ℝ) (gain : ℝ) (amount : ℝ) 
  (h_gain_percent : gain_percent = 1) 
  (h_gain : gain = 0.70) 
  : amount = 70 :=
by
  sorry

end gain_amount_l2164_216421


namespace find_smallest_divisor_l2164_216497

theorem find_smallest_divisor {n : ℕ} 
  (h : n = 44402) 
  (hdiv1 : (n + 2) % 30 = 0) 
  (hdiv2 : (n + 2) % 48 = 0) 
  (hdiv3 : (n + 2) % 74 = 0) 
  (hdiv4 : (n + 2) % 100 = 0) : 
  ∃ d, d = 37 ∧ d ∣ (n + 2) :=
sorry

end find_smallest_divisor_l2164_216497


namespace factorize_mn_minus_mn_cubed_l2164_216438

theorem factorize_mn_minus_mn_cubed (m n : ℝ) : 
  m * n - m * n ^ 3 = m * n * (1 + n) * (1 - n) :=
by {
  sorry
}

end factorize_mn_minus_mn_cubed_l2164_216438


namespace gabby_fruit_total_l2164_216450

-- Definitions based on conditions
def watermelon : ℕ := 1
def peaches : ℕ := watermelon + 12
def plums : ℕ := peaches * 3
def total_fruit : ℕ := watermelon + peaches + plums

-- Proof statement
theorem gabby_fruit_total : total_fruit = 53 := 
by {
  sorry
}

end gabby_fruit_total_l2164_216450


namespace relationship_l2164_216485

-- Define sequences
variable (a b : ℕ → ℝ)

-- Define conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, 1 < m → m < n → a m = a 1 + (m - 1) * (a n - a 1) / (n - 1)

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, 1 < m → m < n → b m = b 1 * (b n / b 1)^(m - 1) / (n - 1)

noncomputable def sequences_conditions : Prop :=
  a 1 = b 1 ∧ a 1 > 0 ∧ ∀ n, a n = b n ∧ b n > 0

-- The main theorem
theorem relationship (h: sequences_conditions a b) : ∀ m n : ℕ, 1 < m → m < n → a m ≥ b m := 
by
  sorry

end relationship_l2164_216485


namespace combined_total_pets_l2164_216466

structure People := 
  (dogs : ℕ)
  (cats : ℕ)

def Teddy : People := {dogs := 7, cats := 8}
def Ben : People := {dogs := 7 + 9, cats := 0}
def Dave : People := {dogs := 7 - 5, cats := 8 + 13}

def total_pets (p : People) : ℕ := p.dogs + p.cats

theorem combined_total_pets : 
  total_pets Teddy + total_pets Ben + total_pets Dave = 54 := by
  sorry

end combined_total_pets_l2164_216466


namespace graph_intersect_x_axis_exactly_once_l2164_216452

theorem graph_intersect_x_axis_exactly_once (a : ℝ) :
    (∀ x : ℝ, (a-1) * x^2 - 4 * x + 2 * a = 0 → x = -(1/2)) ∨ -- Quadratic condition with one real root giving unique intersection
    ((a-1) = 0 ∧ ∃ x : ℝ, -4 * x + 2 * a = 0) -- Linear condition giving unique intersection
    ↔ a = -1 ∨ a = 2 ∨ a = 1 :=
by
    sorry

end graph_intersect_x_axis_exactly_once_l2164_216452


namespace number_of_bottles_l2164_216419

-- Define the weights and total weight based on given conditions
def weight_of_two_bags_chips : ℕ := 800
def total_weight_five_bags_and_juices : ℕ := 2200
def weight_difference_chip_Juice : ℕ := 350

-- Considering 1 bag of chips weighs 400 g (derived from the condition)
def weight_of_one_bag_chips : ℕ := 400
def weight_of_one_bottle_juice : ℕ := weight_of_one_bag_chips - weight_difference_chip_Juice

-- Define the proof of the question
theorem number_of_bottles :
  (total_weight_five_bags_and_juices - (5 * weight_of_one_bag_chips)) / weight_of_one_bottle_juice = 4 := by sorry

end number_of_bottles_l2164_216419


namespace only_a_zero_is_perfect_square_l2164_216469

theorem only_a_zero_is_perfect_square (a : ℕ) : (∃ (k : ℕ), a^2 + 2 * a = k^2) → a = 0 := by
  sorry

end only_a_zero_is_perfect_square_l2164_216469


namespace mapleton_math_team_combinations_l2164_216484

open Nat

theorem mapleton_math_team_combinations (girls boys : ℕ) (team_size girl_on_team boy_on_team : ℕ)
    (h_girls : girls = 4) (h_boys : boys = 5) (h_team_size : team_size = 4)
    (h_girl_on_team : girl_on_team = 3) (h_boy_on_team : boy_on_team = 1) :
    (Nat.choose girls girl_on_team) * (Nat.choose boys boy_on_team) = 20 := by
  sorry

end mapleton_math_team_combinations_l2164_216484


namespace find_angle_A_l2164_216480

theorem find_angle_A (a b : ℝ) (B A : ℝ) 
  (ha : a = Real.sqrt 2) 
  (hb : b = Real.sqrt 3) 
  (hB : B = Real.pi / 3) : 
  A = Real.pi / 4 := 
sorry

end find_angle_A_l2164_216480


namespace project_completion_days_l2164_216455

theorem project_completion_days 
  (total_mandays : ℕ)
  (initial_workers : ℕ)
  (leaving_workers : ℕ)
  (remaining_workers : ℕ)
  (days_total : ℕ) :
  total_mandays = 200 →
  initial_workers = 10 →
  leaving_workers = 4 →
  remaining_workers = 6 →
  days_total = 40 :=
by
  intros h0 h1 h2 h3
  sorry

end project_completion_days_l2164_216455


namespace candy_not_chocolate_l2164_216467

theorem candy_not_chocolate (candy_total : ℕ) (bags : ℕ) (choc_heart_bags : ℕ) (choc_kiss_bags : ℕ) : 
  candy_total = 63 ∧ bags = 9 ∧ choc_heart_bags = 2 ∧ choc_kiss_bags = 3 → 
  (candy_total - (choc_heart_bags * (candy_total / bags) + choc_kiss_bags * (candy_total / bags))) = 28 :=
by
  intros h
  sorry

end candy_not_chocolate_l2164_216467


namespace deepak_age_l2164_216429

theorem deepak_age : ∀ (R D : ℕ), (R / D = 4 / 3) ∧ (R + 6 = 18) → D = 9 :=
by
  sorry

end deepak_age_l2164_216429


namespace rectangle_area_eq_2a_squared_l2164_216407

variable {α : Type} [Semiring α] (a : α)

-- Conditions
def width (a : α) : α := a
def length (a : α) : α := 2 * a

-- Proof statement
theorem rectangle_area_eq_2a_squared (a : α) : (length a) * (width a) = 2 * a^2 := 
sorry

end rectangle_area_eq_2a_squared_l2164_216407


namespace halfway_fraction_l2164_216445

theorem halfway_fraction (a b : ℚ) (h1 : a = 1/5) (h2 : b = 1/3) : (a + b) / 2 = 4 / 15 :=
by 
  rw [h1, h2]
  norm_num

end halfway_fraction_l2164_216445


namespace red_flowers_count_l2164_216447

-- Let's define the given conditions
def total_flowers : ℕ := 10
def white_flowers : ℕ := 2
def blue_percentage : ℕ := 40

-- Calculate the number of blue flowers
def blue_flowers : ℕ := (blue_percentage * total_flowers) / 100

-- The property we want to prove is the number of red flowers
theorem red_flowers_count :
  total_flowers - (blue_flowers + white_flowers) = 4 :=
by
  sorry

end red_flowers_count_l2164_216447


namespace smallest_positive_integer_form_l2164_216402

theorem smallest_positive_integer_form (m n : ℤ) : 
  ∃ (d : ℤ), d > 0 ∧ d = 1205 * m + 27090 * n ∧ (∀ e, e > 0 → (∃ x y : ℤ, d = 1205 * x + 27090 * y) → d ≤ e) :=
sorry

end smallest_positive_integer_form_l2164_216402


namespace compute_expression_l2164_216460

theorem compute_expression (x y z : ℝ) (h₀ : x ≠ y) (h₁ : y ≠ z) (h₂ : z ≠ x) (h₃ : x + y + z = 3) :
  (xy + yz + zx) / (x^2 + y^2 + z^2) = 9 / (2 * (x^2 + y^2 + z^2)) - 1 / 2 :=
by
  sorry

end compute_expression_l2164_216460


namespace constant_function_l2164_216446

theorem constant_function {f : ℕ → ℕ} (h : ∀ x y : ℕ, x * f y + y * f x = (x + y) * f (x^2 + y^2)) : ∃ c : ℕ, ∀ x, f x = c := 
sorry

end constant_function_l2164_216446


namespace two_roots_iff_a_gt_neg1_l2164_216415

theorem two_roots_iff_a_gt_neg1 (a : ℝ) :
  (∃! x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2*x1 + 2*|x1 + 1| = a) ∧ (x2^2 + 2*x2 + 2*|x2 + 1| = a)) ↔ a > -1 :=
by sorry

end two_roots_iff_a_gt_neg1_l2164_216415


namespace union_when_m_equals_4_subset_implies_m_range_l2164_216420

-- Define the sets and conditions
def set_A := { x : ℝ | -2 ≤ x ∧ x ≤ 5 }
def set_B (m : ℝ) := { x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1 }

-- Problem 1: When m = 4, find the union of A and B
theorem union_when_m_equals_4 : ∀ x, x ∈ set_A ∪ set_B 4 ↔ -2 ≤ x ∧ x ≤ 7 :=
by sorry

-- Problem 2: If B ⊆ A, find the range of the real number m
theorem subset_implies_m_range (m : ℝ) : (∀ x, x ∈ set_B m → x ∈ set_A) ↔ m ≤ 3 :=
by sorry

end union_when_m_equals_4_subset_implies_m_range_l2164_216420


namespace find_principal_amount_l2164_216486

theorem find_principal_amount (P r : ℝ) (A2 A3 : ℝ) (n2 n3 : ℕ) 
  (h1 : n2 = 2) (h2 : n3 = 3) 
  (h3 : A2 = 8820) 
  (h4 : A3 = 9261) 
  (h5 : r = 0.05) 
  (h6 : A2 = P * (1 + r)^n2) 
  (h7 : A3 = P * (1 + r)^n3) : 
  P = 8000 := 
by 
  sorry

end find_principal_amount_l2164_216486


namespace exists_x_y_l2164_216453

theorem exists_x_y (n : ℕ) (hn : 0 < n) :
  ∃ x y : ℕ, n < x ∧ ¬ x ∣ y ∧ x^x ∣ y^y :=
by sorry

end exists_x_y_l2164_216453


namespace max_min_page_difference_l2164_216428

-- Define the number of pages in each book
variables (Poetry Documents Rites Changes SpringAndAutumn : ℤ)

-- Define the conditions as given in the problem
axiom h1 : abs (Poetry - Documents) = 24
axiom h2 : abs (Documents - Rites) = 17
axiom h3 : abs (Rites - Changes) = 27
axiom h4 : abs (Changes - SpringAndAutumn) = 19
axiom h5 : abs (SpringAndAutumn - Poetry) = 15

-- Assertion to prove
theorem max_min_page_difference : 
  ∃ a b c d e : ℤ, a = Poetry ∧ b = Documents ∧ c = Rites ∧ d = Changes ∧ e = SpringAndAutumn ∧ 
  abs (a - b) = 24 ∧ abs (b - c) = 17 ∧ abs (c - d) = 27 ∧ abs (d - e) = 19 ∧ abs (e - a) = 15 ∧ 
  (max a (max b (max c (max d e))) - min a (min b (min c (min d e)))) = 34 :=
by {
  sorry
}

end max_min_page_difference_l2164_216428


namespace min_value_abc2_l2164_216461

variables (a b c d : ℝ)

def condition_1 : Prop := a + b = 9 / (c - d)
def condition_2 : Prop := c + d = 25 / (a - b)

theorem min_value_abc2 :
  condition_1 a b c d → condition_2 a b c d → (a^2 + b^2 + c^2 + d^2) = 34 :=
by
  intros h1 h2
  sorry

end min_value_abc2_l2164_216461


namespace dressing_p_percentage_l2164_216494

-- Define the percentages of vinegar and oil in dressings p and q
def vinegar_in_p : ℝ := 0.30
def vinegar_in_q : ℝ := 0.10

-- Define the desired percentage of vinegar in the new dressing
def vinegar_in_new_dressing : ℝ := 0.12

-- Define the total mass of the new dressing
def total_mass_new_dressing : ℝ := 100.0

-- Define the mass of dressing p in the new dressing
def mass_of_p (x : ℝ) : ℝ := x

-- Define the mass of dressing q in the new dressing
def mass_of_q (x : ℝ) : ℝ := total_mass_new_dressing - x

-- Define the amount of vinegar contributed by dressings p and q
def vinegar_from_p (x : ℝ) : ℝ := vinegar_in_p * mass_of_p x
def vinegar_from_q (x : ℝ) : ℝ := vinegar_in_q * mass_of_q x

-- Define the total vinegar in the new dressing
def total_vinegar (x : ℝ) : ℝ := vinegar_from_p x + vinegar_from_q x

-- Problem statement: prove the percentage of dressing p in the new dressing
theorem dressing_p_percentage (x : ℝ) (hx : total_vinegar x = vinegar_in_new_dressing * total_mass_new_dressing) :
  (mass_of_p x / total_mass_new_dressing) * 100 = 10 :=
by
  sorry

end dressing_p_percentage_l2164_216494


namespace min_segments_for_octagon_perimeter_l2164_216492

/-- Given an octagon formed by cutting a smaller rectangle from a larger rectangle,
the minimum number of distinct line segment lengths needed to calculate the perimeter 
of this octagon is 3. --/
theorem min_segments_for_octagon_perimeter (a b c d e f g h : ℝ)
  (cond : a = c ∧ b = d ∧ e = g ∧ f = h) :
  ∃ (u v w : ℝ), u ≠ v ∧ v ≠ w ∧ u ≠ w :=
by
  sorry

end min_segments_for_octagon_perimeter_l2164_216492


namespace jim_gave_away_cards_l2164_216487

theorem jim_gave_away_cards
  (sets_brother : ℕ := 15)
  (sets_sister : ℕ := 8)
  (sets_friend : ℕ := 4)
  (sets_cousin : ℕ := 6)
  (sets_classmate : ℕ := 3)
  (cards_per_set : ℕ := 25) :
  (sets_brother + sets_sister + sets_friend + sets_cousin + sets_classmate) * cards_per_set = 900 :=
by
  sorry

end jim_gave_away_cards_l2164_216487


namespace customers_left_l2164_216499

-- Definitions based on problem conditions
def initial_customers : ℕ := 14
def remaining_customers : ℕ := 3

-- Theorem statement based on the question and the correct answer
theorem customers_left : initial_customers - remaining_customers = 11 := by
  sorry

end customers_left_l2164_216499


namespace specific_certain_event_l2164_216435

theorem specific_certain_event :
  ∀ (A B C D : Prop), 
    (¬ A) →
    (¬ B) →
    (¬ C) →
    D →
    D :=
by
  intros A B C D hA hB hC hD
  exact hD

end specific_certain_event_l2164_216435


namespace MrC_loses_240_after_transactions_l2164_216424

theorem MrC_loses_240_after_transactions :
  let house_initial_value := 12000
  let first_transaction_loss_percent := 0.15
  let second_transaction_gain_percent := 0.20
  let house_value_after_first_transaction :=
    house_initial_value * (1 - first_transaction_loss_percent)
  let house_value_after_second_transaction :=
    house_value_after_first_transaction * (1 + second_transaction_gain_percent)
  house_value_after_second_transaction - house_initial_value = 240 :=
by
  sorry

end MrC_loses_240_after_transactions_l2164_216424


namespace triangle_side_b_l2164_216481

open Real

variable {a b c : ℝ} (A B C : ℝ)

theorem triangle_side_b (h1 : a^2 - c^2 = 2 * b) (h2 : sin B = 6 * cos A * sin C) : b = 3 :=
sorry

end triangle_side_b_l2164_216481


namespace pond_field_area_ratio_l2164_216493

theorem pond_field_area_ratio (w l s A_field A_pond : ℕ) (h1 : l = 2 * w) (h2 : l = 96) (h3 : s = 8) (h4 : A_field = l * w) (h5 : A_pond = s * s) :
  A_pond.toFloat / A_field.toFloat = 1 / 72 := 
by
  sorry

end pond_field_area_ratio_l2164_216493


namespace product_and_quotient_l2164_216406

theorem product_and_quotient : (16 * 0.0625 / 4 * 0.5 * 2) = (1 / 4) :=
by
  -- The proof steps would go here
  sorry

end product_and_quotient_l2164_216406


namespace parabola_focus_coordinates_l2164_216400

theorem parabola_focus_coordinates (h : ∀ y, y^2 = 4 * x) : ∃ x, x = 1 ∧ y = 0 := 
sorry

end parabola_focus_coordinates_l2164_216400


namespace quadratic_sum_l2164_216498

theorem quadratic_sum (x : ℝ) :
  ∃ a h k : ℝ, (5*x^2 - 10*x - 3 = a*(x - h)^2 + k) ∧ (a + h + k = -2) :=
sorry

end quadratic_sum_l2164_216498


namespace functional_eq_solution_l2164_216433

theorem functional_eq_solution (f : ℝ → ℝ) 
  (H : ∀ x y : ℝ, f (2 * x + f y) = x + y + f x) : 
  ∀ x : ℝ, f x = x := 
by 
  sorry

end functional_eq_solution_l2164_216433


namespace sum_of_fractions_and_decimal_l2164_216462

theorem sum_of_fractions_and_decimal : 
    (3 / 25 : ℝ) + (1 / 5) + 55.21 = 55.53 :=
by 
  sorry

end sum_of_fractions_and_decimal_l2164_216462


namespace abs_ab_cd_leq_one_fourth_l2164_216441

theorem abs_ab_cd_leq_one_fourth (a b c d : ℝ) (h0 : 0 ≤ a) (h1 : 0 ≤ b) (h2 : 0 ≤ c) (h3 : 0 ≤ d) (h_sum : a + b + c + d = 1) :
  |a * b - c * d| ≤ 1 / 4 :=
sorry

end abs_ab_cd_leq_one_fourth_l2164_216441


namespace arithmetic_seq_term_ratio_l2164_216434

-- Assume two arithmetic sequences a and b
def arithmetic_seq_a (n : ℕ) : ℕ := sorry
def arithmetic_seq_b (n : ℕ) : ℕ := sorry

-- Sum of first n terms of the sequences
def sum_a (n : ℕ) : ℕ := (List.range (n+1)).map arithmetic_seq_a |>.sum
def sum_b (n : ℕ) : ℕ := (List.range (n+1)).map arithmetic_seq_b |>.sum

-- The given condition: Sn / Tn = (7n + 2) / (n + 3)
axiom sum_condition (n : ℕ) : (sum_a n) / (sum_b n) = (7 * n + 2) / (n + 3)

-- The goal: a4 / b4 = 51 / 10
theorem arithmetic_seq_term_ratio : (arithmetic_seq_a 4 : ℚ) / (arithmetic_seq_b 4 : ℚ) = 51 / 10 :=
by
  sorry

end arithmetic_seq_term_ratio_l2164_216434


namespace parallelogram_probability_l2164_216425

theorem parallelogram_probability (P Q R S : ℝ × ℝ) 
  (hP : P = (4, 2)) 
  (hQ : Q = (-2, -2)) 
  (hR : R = (-6, -6)) 
  (hS : S = (0, -2)) :
  let parallelogram_area := 24 -- given the computed area based on provided geometry
  let divided_area := parallelogram_area / 2
  let not_above_x_axis_area := divided_area
  (not_above_x_axis_area / parallelogram_area) = (1 / 2) :=
by
  sorry

end parallelogram_probability_l2164_216425


namespace imag_part_of_complex_l2164_216426

open Complex

theorem imag_part_of_complex : (im ((5 + I) / (1 + I))) = -2 :=
by
  sorry

end imag_part_of_complex_l2164_216426


namespace find_remainder_l2164_216436

theorem find_remainder (n : ℕ) 
  (h1 : n^2 % 7 = 3)
  (h2 : n^3 % 7 = 6) : 
  n % 7 = 6 := 
by sorry

end find_remainder_l2164_216436


namespace factorization_of_polynomial_l2164_216458

theorem factorization_of_polynomial :
  ∀ x : ℝ, x^2 + 6 * x + 9 - 64 * x^4 = (-8 * x^2 + x + 3) * (8 * x^2 + x + 3) :=
by
  intro x
  -- Sorry placeholder for the proof
  sorry

end factorization_of_polynomial_l2164_216458


namespace range_of_a_l2164_216413

open Real

noncomputable def p (a : ℝ) := ∀ (x : ℝ), x ≥ 1 → (2 * x - 3 * a) ≥ 0
noncomputable def q (a : ℝ) := (0 < 2 * a - 1) ∧ (2 * a - 1 < 1)

theorem range_of_a (a : ℝ) : p a ∧ q a ↔ (1/2 < a ∧ a ≤ 2/3) := by
  sorry

end range_of_a_l2164_216413


namespace find_n_l2164_216443

theorem find_n (x y : ℤ) (n : ℕ) (h1 : (x:ℝ)^n + (y:ℝ)^n = 91) (h2 : (x:ℝ) * y = 11.999999999999998) :
  n = 3 := 
sorry

end find_n_l2164_216443


namespace integral_percentage_l2164_216444

variable (a b : ℝ)

theorem integral_percentage (h : ∀ x, x^2 > 0) :
  (∫ x in a..b, (1 / 20 * x^2 + 3 / 10 * x^2)) = 0.35 * (∫ x in a..b, x^2) :=
by
  sorry

end integral_percentage_l2164_216444


namespace prove_a5_l2164_216470

-- Definition of the conditions
def expansion (x : ℤ) (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℤ) :=
  (x - 1) ^ 8 = a_0 + a_1 * (1 + x) + a_2 * (1 + x)^2 + a_3 * (1 + x)^3 + a_4 * (1 + x)^4 + 
               a_5 * (1 + x)^5 + a_6 * (1 + x)^6 + a_7 * (1 + x)^7 + a_8 * (1 + x)^8

-- Given condition
axiom condition (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℤ) : ∀ x : ℤ, expansion x a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8

-- The target problem: proving a_5 = -448
theorem prove_a5 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℤ) : a_5 = -448 :=
by
  sorry

end prove_a5_l2164_216470


namespace find_triples_of_positive_integers_l2164_216431

theorem find_triples_of_positive_integers :
  ∀ (x y z : ℕ), 2 * (x + y + z + 2 * x * y * z)^2 = (2 * x * y + 2 * y * z + 2 * z * x + 1)^2 + 2023 ↔ 
  (x = 3 ∧ y = 3 ∧ z = 2) ∨
  (x = 3 ∧ y = 2 ∧ z = 3) ∨
  (x = 2 ∧ y = 3 ∧ z = 3) ∨
  (x = 2 ∧ y = 3 ∧ z = 3) ∨
  (x = 3 ∧ y = 2 ∧ z = 3) ∨
  (x = 3 ∧ y = 3 ∧ z = 2) := 
by 
  sorry

end find_triples_of_positive_integers_l2164_216431


namespace solve_frac_eqn_l2164_216471

theorem solve_frac_eqn (x : ℝ) :
  (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) +
   1 / ((x - 5) * (x - 7)) + 1 / ((x - 7) * (x - 9)) = 1 / 8) ↔ 
  (x = 13 ∨ x = -3) :=
by
  sorry

end solve_frac_eqn_l2164_216471


namespace remainder_of_large_number_l2164_216454

theorem remainder_of_large_number :
  let number := 65985241545898754582556898522454889
  let last_four_digits := 4889
  last_four_digits % 16 = 9 := 
by
  let number := 65985241545898754582556898522454889
  let last_four_digits := 4889
  show last_four_digits % 16 = 9
  sorry

end remainder_of_large_number_l2164_216454


namespace problem1_problem2_problem3_problem4_l2164_216416

-- Problem 1
theorem problem1 (x : ℝ) (h : x * (5 * x + 4) = 5 * x + 4) : x = -4 / 5 ∨ x = 1 := 
sorry

-- Problem 2
theorem problem2 (x : ℝ) (h : -3 * x^2 + 22 * x - 24 = 0) : x = 6 ∨ x = 4 / 3 := 
sorry

-- Problem 3
theorem problem3 (x : ℝ) (h : (x + 8) * (x + 1) = -12) : x = -4 ∨ x = -5 := 
sorry

-- Problem 4
theorem problem4 (x : ℝ) (h : (3 * x + 2) * (x + 3) = x + 14) : x = -4 ∨ x = 2 / 3 := 
sorry

end problem1_problem2_problem3_problem4_l2164_216416


namespace problem1_problem2_problem3_l2164_216478

-- 1. Prove that (3ab³)² = 9a²b⁶
theorem problem1 (a b : ℝ) : (3 * a * b^3)^2 = 9 * a^2 * b^6 :=
by sorry

-- 2. Prove that x ⋅ x³ + x² ⋅ x² = 2x⁴
theorem problem2 (x : ℝ) : x * x^3 + x^2 * x^2 = 2 * x^4 :=
by sorry

-- 3. Prove that (12x⁴ - 6x³) ÷ 3x² = 4x² - 2x
theorem problem3 (x : ℝ) : (12 * x^4 - 6 * x^3) / (3 * x^2) = 4 * x^2 - 2 * x :=
by sorry

end problem1_problem2_problem3_l2164_216478


namespace cans_of_type_B_purchased_l2164_216409

variable (T P R : ℕ)

-- Conditions
def cost_per_can_A : ℕ := P / T
def cost_per_can_B : ℕ := 2 * cost_per_can_A T P
def quarters_in_dollar : ℕ := 4

-- Question and proof target
theorem cans_of_type_B_purchased (T P R : ℕ) (hT : T > 0) (hP : P > 0) (hR : R > 0) :
  (4 * R) / (2 * P / T) = 2 * R * T / P :=
by
  sorry

end cans_of_type_B_purchased_l2164_216409


namespace number_of_solutions_proof_l2164_216495

noncomputable def number_of_real_solutions (x y z w : ℝ) : ℝ :=
  if (x = z + w + 2 * z * w * x) ∧ (y = w + x + 2 * w * x * y) ∧ (z = x + y + 2 * x * y * z) ∧ (w = y + z + 2 * y * z * w) then
    5
  else
    0

theorem number_of_solutions_proof :
  ∃ x y z w : ℝ, x = z + w + 2 * z * w * x ∧ y = w + x + 2 * w * x * y ∧ z = x + y + 2 * x * y * z ∧ w = y + z + 2 * y * z * w → number_of_real_solutions x y z w = 5 :=
by
  sorry

end number_of_solutions_proof_l2164_216495


namespace number_equals_14_l2164_216403

theorem number_equals_14 (n : ℕ) (h1 : 2^n - 2^(n-2) = 3 * 2^12) (h2 : n = 14) : n = 14 := 
by 
  sorry

end number_equals_14_l2164_216403


namespace area_ratio_l2164_216479

noncomputable def pentagon_area (R s : ℝ) := (5 / 2) * R * s * Real.sin (Real.pi * 2 / 5)
noncomputable def triangle_area (s : ℝ) := (s^2) / 4

theorem area_ratio (R s : ℝ) (h : R = s / (2 * Real.sin (Real.pi / 5))) :
  (pentagon_area R s) / (triangle_area s) = 5 * (Real.sin ((2 * Real.pi) / 5) / Real.sin (Real.pi / 5)) :=
by
  sorry

end area_ratio_l2164_216479


namespace max_value_of_f_in_interval_l2164_216475

noncomputable def f (x m : ℝ) : ℝ := -x^3 + 3 * x^2 + m

theorem max_value_of_f_in_interval (m : ℝ) (h₁ : ∀ x ∈ [-2, 2], - x^3 + 3 * x^2 + m ≥ 1) : 
  ∃ x ∈ [-2, 2], f x m = 21 :=
by
  sorry

end max_value_of_f_in_interval_l2164_216475


namespace red_car_count_l2164_216440

-- Define the ratio and the given number of black cars
def ratio_red_to_black (R B : ℕ) : Prop := R * 8 = B * 3

-- Define the given number of black cars
def black_cars : ℕ := 75

-- State the theorem we want to prove
theorem red_car_count : ∃ R : ℕ, ratio_red_to_black R black_cars ∧ R = 28 :=
by
  sorry

end red_car_count_l2164_216440


namespace number_of_elements_in_sequence_l2164_216496

theorem number_of_elements_in_sequence :
  ∀ (a₀ d : ℕ) (n : ℕ), 
  a₀ = 4 →
  d = 2 →
  n = 64 →
  (a₀ + (n - 1) * d = 130) →
  n = 64 := 
by
  -- We will skip the proof steps as indicated
  sorry

end number_of_elements_in_sequence_l2164_216496


namespace oa_dot_ob_eq_neg2_l2164_216401

/-!
# Problem Statement
Given AB as the diameter of the smallest radius circle centered at C(0,1) that intersects 
the graph of y = 1 / (|x| - 1), where O is the origin. Prove that the dot product 
\overrightarrow{OA} · \overrightarrow{OB} equals -2.
-/

noncomputable def smallest_radius_circle_eqn (x : ℝ) : ℝ :=
  x^2 + ((1 / (|x| - 1)) - 1)^2

noncomputable def radius_of_circle (x : ℝ) : ℝ :=
  Real.sqrt (smallest_radius_circle_eqn x)

noncomputable def OA (x : ℝ) : ℝ × ℝ :=
  (x, (1 / (|x| - 1)) + 1)

noncomputable def OB (x : ℝ) : ℝ × ℝ :=
  (-x, 1 - (1 / (|x| - 1)))

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

theorem oa_dot_ob_eq_neg2 (x : ℝ) (hx : |x| > 1) :
  let a := OA x
  let b := OB x
  dot_product a b = -2 :=
by
  sorry

end oa_dot_ob_eq_neg2_l2164_216401


namespace factorization_correct_l2164_216423

-- Define the input expression
def expr (x y : ℝ) : ℝ := 2 * x^3 - 18 * x * y^2

-- Define the factorized form
def factorized_expr (x y : ℝ) : ℝ := 2 * x * (x + 3*y) * (x - 3*y)

-- Prove that the original expression is equal to the factorized form
theorem factorization_correct (x y : ℝ) : expr x y = factorized_expr x y := 
by sorry

end factorization_correct_l2164_216423


namespace arithmetic_sequence_120th_term_l2164_216418

theorem arithmetic_sequence_120th_term :
  let a1 := 6
  let d := 6
  let n := 120
  let a_n := a1 + (n - 1) * d
  a_n = 720 := by
  sorry

end arithmetic_sequence_120th_term_l2164_216418


namespace factory_output_exceeds_by_20_percent_l2164_216411

theorem factory_output_exceeds_by_20_percent 
  (planned_output : ℝ) (actual_output : ℝ)
  (h_planned : planned_output = 20)
  (h_actual : actual_output = 24) :
  ((actual_output - planned_output) / planned_output) * 100 = 20 := 
by
  sorry

end factory_output_exceeds_by_20_percent_l2164_216411


namespace increasing_function_range_l2164_216448

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 * a - 1) * x - 1 else x + 1

theorem increasing_function_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (1 / 2 < a ∧ a ≤ 2) :=
sorry

end increasing_function_range_l2164_216448


namespace equal_numbers_l2164_216476

namespace MathProblem

theorem equal_numbers 
  (x y z : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (h : x^2 / y + y^2 / z + z^2 / x = x^2 / z + z^2 / y + y^2 / x) : 
  x = y ∨ x = z ∨ y = z :=
by
  sorry

end MathProblem

end equal_numbers_l2164_216476


namespace arithmetic_sequence_sum_l2164_216488

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ)     -- arithmetic sequence
  (d : ℝ)         -- common difference
  (h: ∀ n, a (n + 1) = a n + d)     -- definition of arithmetic sequence
  (h_sum : a 2 + a 4 + a 5 + a 6 + a 8 = 25) : 
  a 2 + a 8 = 10 := 
  sorry

end arithmetic_sequence_sum_l2164_216488


namespace length_of_first_platform_l2164_216442

theorem length_of_first_platform 
  (t1 t2 : ℝ) 
  (length_train : ℝ) 
  (length_second_platform : ℝ) 
  (time1 : ℝ) 
  (time2 : ℝ) 
  (speed_eq : (t1 + length_train) / time1 = (length_second_platform + length_train) / time2) 
  (time1_eq : time1 = 15) 
  (time2_eq : time2 = 20) 
  (length_train_eq : length_train = 100) 
  (length_second_platform_eq: length_second_platform = 500) :
  t1 = 350 := 
  by 
  sorry

end length_of_first_platform_l2164_216442


namespace evaluate_expression_l2164_216408

def binom (n k : ℕ) : ℕ := if h : k ≤ n then Nat.choose n k else 0

theorem evaluate_expression : 
  (binom 2 5 * 3 ^ 5) / binom 10 5 = 0 := by
  -- Given conditions:
  have h1 : binom 2 5 = 0 := by sorry
  have h2 : binom 10 5 = 252 := by sorry
  -- Proof goal:
  sorry

end evaluate_expression_l2164_216408


namespace athlete_D_is_selected_l2164_216457

-- Define the average scores and variances of athletes
def avg_A : ℝ := 9.5
def var_A : ℝ := 6.6
def avg_B : ℝ := 9.6
def var_B : ℝ := 6.7
def avg_C : ℝ := 9.5
def var_C : ℝ := 6.7
def avg_D : ℝ := 9.6
def var_D : ℝ := 6.6

-- Define what it means for an athlete to be good and stable
def good_performance (avg : ℝ) : Prop := avg ≥ 9.6
def stable_play (variance : ℝ) : Prop := variance ≤ 6.6

-- Combine conditions for selecting the athlete
def D_is_suitable : Prop := good_performance avg_D ∧ stable_play var_D

-- State the theorem to be proved
theorem athlete_D_is_selected : D_is_suitable := 
by 
  sorry

end athlete_D_is_selected_l2164_216457


namespace op_add_mul_example_l2164_216490

def op_add (a b : ℤ) : ℤ := a + b - 1
def op_mul (a b : ℤ) : ℤ := a * b - 1

theorem op_add_mul_example : op_mul (op_add 6 8) (op_add 3 5) = 90 :=
by
  -- Rewriting it briefly without proof steps
  sorry

end op_add_mul_example_l2164_216490


namespace sum_and_product_of_reciprocals_l2164_216404

theorem sum_and_product_of_reciprocals (x y : ℝ) (h_sum : x + y = 12) (h_prod : x * y = 32) :
  (1/x + 1/y = 3/8) ∧ (1/x * 1/y = 1/32) :=
by
  sorry

end sum_and_product_of_reciprocals_l2164_216404


namespace longest_segment_cylinder_l2164_216456

theorem longest_segment_cylinder (r h : ℤ) (c : ℝ) (hr : r = 4) (hh : h = 9) : 
  c = Real.sqrt (2 * r * r + h * h) ↔ c = Real.sqrt 145 :=
by
  sorry

end longest_segment_cylinder_l2164_216456


namespace gcd_m_n_l2164_216427

noncomputable def m : ℕ := 5 * 11111111
noncomputable def n : ℕ := 111111111

theorem gcd_m_n : gcd m n = 11111111 := by
  sorry

end gcd_m_n_l2164_216427


namespace incorrect_statement_A_l2164_216464

theorem incorrect_statement_A (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b) :
  ¬ (a - a^2 > b - b^2) := sorry

end incorrect_statement_A_l2164_216464


namespace angle_BAC_eq_69_l2164_216422

-- Definitions and conditions
def AM_Squared_EQ_CM_MN (AM CM MN : ℝ) : Prop := AM^2 = CM * MN
def AM_EQ_MK (AM MK : ℝ) : Prop := AM = MK
def angle_AMN_EQ_CMK (angle_AMN angle_CMK : ℝ) : Prop := angle_AMN = angle_CMK
def angle_B : ℝ := 47
def angle_C : ℝ := 64

-- Final proof statement
theorem angle_BAC_eq_69 (AM CM MN MK : ℝ)
  (h1: AM_Squared_EQ_CM_MN AM CM MN)
  (h2: AM_EQ_MK AM MK)
  (h3: angle_AMN_EQ_CMK 70 70) -- Placeholder angle values since angles must be given/defined
  : ∃ angle_BAC : ℝ, angle_BAC = 69 :=
sorry

end angle_BAC_eq_69_l2164_216422


namespace geometric_sequence_sum_div_l2164_216437

theorem geometric_sequence_sum_div :
  ∀ {a : ℕ → ℝ} {q : ℝ},
  (∀ n, a (n + 1) = a n * q) →
  q = -1 / 3 →
  (a 1 + a 3 + a 5 + a 7) / (a 2 + a 4 + a 6 + a 8) = -3 :=
by
  intros a q geometric_seq common_ratio
  sorry

end geometric_sequence_sum_div_l2164_216437


namespace point_in_second_quadrant_l2164_216482

-- Define the point in question
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Given conditions based on the problem statement
def P (x : ℝ) : Point :=
  Point.mk (-2) (x^2 + 1)

-- The theorem we aim to prove
theorem point_in_second_quadrant (x : ℝ) : (P x).x < 0 ∧ (P x).y > 0 → 
  -- This condition means that the point is in the second quadrant
  (P x).x < 0 ∧ (P x).y > 0 :=
by
  sorry

end point_in_second_quadrant_l2164_216482


namespace conic_section_pair_of_lines_l2164_216412

theorem conic_section_pair_of_lines : 
  (∀ x y : ℝ, 4 * x^2 - 9 * y^2 = 0 → (2 * x - 3 * y = 0 ∨ 2 * x + 3 * y = 0)) :=
by
  sorry

end conic_section_pair_of_lines_l2164_216412


namespace sufficient_but_not_necessary_l2164_216463

variable (x : ℚ)

def is_integer (n : ℚ) : Prop := ∃ (k : ℤ), n = k

theorem sufficient_but_not_necessary :
  (is_integer x → is_integer (2 * x + 1)) ∧
  (¬ (is_integer (2 * x + 1) → is_integer x)) :=
by
  sorry

end sufficient_but_not_necessary_l2164_216463


namespace line_through_two_points_l2164_216417

theorem line_through_two_points (A B : ℝ × ℝ)
  (hA : A = (2, -3))
  (hB : B = (1, 4)) :
  ∃ (m b : ℝ), (∀ x y : ℝ, (y = m * x + b) ↔ ((x, y) = A ∨ (x, y) = B)) ∧ m = -7 ∧ b = 11 := by
  sorry

end line_through_two_points_l2164_216417


namespace child_tickets_sold_l2164_216468

theorem child_tickets_sold
  (A C : ℕ) 
  (h1 : A + C = 900)
  (h2 : 7 * A + 4 * C = 5100) :
  C = 400 :=
by
  sorry

end child_tickets_sold_l2164_216468


namespace pizza_slices_left_l2164_216465

theorem pizza_slices_left (initial_slices : ℕ) (people : ℕ) (slices_per_person : ℕ) 
  (h1 : initial_slices = 16) (h2 : people = 6) (h3 : slices_per_person = 2) : 
  initial_slices - people * slices_per_person = 4 := 
by
  sorry

end pizza_slices_left_l2164_216465


namespace inequality_system_solution_l2164_216472

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_system_solution_l2164_216472


namespace find_k_and_b_l2164_216474

theorem find_k_and_b (k b : ℝ) :
  (∃ P Q : ℝ × ℝ, P ≠ Q ∧
  ((P.1 - 1)^2 + P.2^2 = 1) ∧ 
  ((Q.1 - 1)^2 + Q.2^2 = 1) ∧ 
  (P.2 = k * P.1) ∧ 
  (Q.2 = k * Q.1) ∧ 
  (P.1 - P.2 + b = 0) ∧ 
  (Q.1 - Q.2 + b = 0) ∧ 
  ((P.1 + Q.1) / 2 = (P.2 + Q.2) / 2)) →
  k = -1 ∧ b = -1 :=
sorry

end find_k_and_b_l2164_216474


namespace at_least_one_negative_l2164_216410

theorem at_least_one_negative (a b : ℝ) (h : a + b < 0) : a < 0 ∨ b < 0 := by
  sorry

end at_least_one_negative_l2164_216410


namespace standard_equation_of_ellipse_l2164_216432

-- Define the conditions
def isEccentricity (e : ℝ) := e = (Real.sqrt 3) / 3
def segmentLength (L : ℝ) := L = (4 * Real.sqrt 3) / 3

-- Define properties
def is_ellipse (a b c : ℝ) := a > b ∧ b > 0 ∧ (a^2 = b^2 + c^2) ∧ (c = (Real.sqrt 3) / 3 * a)

-- The problem statement
theorem standard_equation_of_ellipse
(a b c : ℝ) (E L : ℝ)
(hE : isEccentricity E)
(hL : segmentLength L)
(h : is_ellipse a b c)
: (a = Real.sqrt 3) ∧ (c = 1) ∧ (b = Real.sqrt 2) ∧ (segmentLength L)
  → ( ∀ x y : ℝ, ((x^2 / 3) + (y^2 / 2) = 1) ) := by
  sorry

end standard_equation_of_ellipse_l2164_216432
