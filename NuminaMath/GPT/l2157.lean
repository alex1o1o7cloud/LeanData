import Mathlib

namespace NUMINAMATH_GPT_problem_equivalent_l2157_215704

theorem problem_equivalent (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : (4 * x + 2 * y) / (x - 4 * y) = 3) (hz_eq : z = 10 * y) :
  (x + 4 * y + z) / (4 * x - y - z) = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_equivalent_l2157_215704


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l2157_215734

-- Define the conditions
def abs_value_condition (x : ℝ) : Prop := |x| < 2
def quadratic_condition (x : ℝ) : Prop := x^2 - x - 6 < 0

-- Theorem statement
theorem sufficient_but_not_necessary : (∀ x : ℝ, abs_value_condition x → quadratic_condition x) ∧ ¬ (∀ x : ℝ, quadratic_condition x → abs_value_condition x) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l2157_215734


namespace NUMINAMATH_GPT_Jame_tears_30_cards_at_a_time_l2157_215770

theorem Jame_tears_30_cards_at_a_time
    (cards_per_deck : ℕ)
    (times_per_week : ℕ)
    (decks : ℕ)
    (weeks : ℕ)
    (total_cards : ℕ := decks * cards_per_deck)
    (total_times : ℕ := weeks * times_per_week)
    (cards_at_a_time : ℕ := total_cards / total_times)
    (h1 : cards_per_deck = 55)
    (h2 : times_per_week = 3)
    (h3 : decks = 18)
    (h4 : weeks = 11) :
    cards_at_a_time = 30 := by
  -- Proof can be added here
  sorry

end NUMINAMATH_GPT_Jame_tears_30_cards_at_a_time_l2157_215770


namespace NUMINAMATH_GPT_brown_beads_initial_l2157_215742

theorem brown_beads_initial (B : ℕ) 
  (h1 : 1 = 1) -- There is 1 green bead in the container.
  (h2 : 3 = 3) -- There are 3 red beads in the container.
  (h3 : 4 = 4) -- Tom left 4 beads in the container.
  (h4 : 2 = 2) -- Tom took out 2 beads.
  (h5 : 6 = 2 + 4) -- Total initial beads before Tom took any out.
  : B = 2 := sorry

end NUMINAMATH_GPT_brown_beads_initial_l2157_215742


namespace NUMINAMATH_GPT_collinear_c1_c2_l2157_215754

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (3, 7, 0)
def b : ℝ × ℝ × ℝ := (1, -3, 4)

-- Define the vectors c1 and c2 based on a and b
def c1 : ℝ × ℝ × ℝ := (4 * 3, 4 * 7, 4 * 0) - (2 * 1, 2 * -3, 2 * 4)
def c2 : ℝ × ℝ × ℝ := (1, -3, 4) - (2 * 3, 2 * 7, 2 * 0)

-- The theorem to prove that c1 and c2 are collinear
theorem collinear_c1_c2 : c1 = (-2 : ℝ) • c2 := by sorry

end NUMINAMATH_GPT_collinear_c1_c2_l2157_215754


namespace NUMINAMATH_GPT_cylinder_height_to_radius_ratio_l2157_215726

theorem cylinder_height_to_radius_ratio (V r h : ℝ) (hV : V = π * r^2 * h) (hS : sorry) :
  h / r = 2 :=
sorry

end NUMINAMATH_GPT_cylinder_height_to_radius_ratio_l2157_215726


namespace NUMINAMATH_GPT_jan_discount_percentage_l2157_215768

theorem jan_discount_percentage :
  ∃ percent_discount : ℝ,
    ∀ (roses_bought dozen : ℕ) (rose_cost amount_paid : ℝ),
      roses_bought = 5 * dozen → dozen = 12 →
      rose_cost = 6 →
      amount_paid = 288 →
      (roses_bought * rose_cost - amount_paid) / (roses_bought * rose_cost) * 100 = percent_discount →
      percent_discount = 20 :=
by
  sorry

end NUMINAMATH_GPT_jan_discount_percentage_l2157_215768


namespace NUMINAMATH_GPT_absolute_value_inequality_l2157_215790

theorem absolute_value_inequality (x y z : ℝ) : 
  |x| + |y| + |z| ≤ |x + y - z| + |x - y + z| + |-x + y + z| :=
by
  sorry

end NUMINAMATH_GPT_absolute_value_inequality_l2157_215790


namespace NUMINAMATH_GPT_farmer_total_profit_l2157_215793

theorem farmer_total_profit :
  let group1_revenue := 3 * 375
  let group1_cost := (8 * 13 + 3 * 15) * 3
  let group1_profit := group1_revenue - group1_cost

  let group2_revenue := 4 * 425
  let group2_cost := (5 * 14 + 9 * 16) * 4
  let group2_profit := group2_revenue - group2_cost

  let group3_revenue := 2 * 475
  let group3_cost := (10 * 15 + 8 * 18) * 2
  let group3_profit := group3_revenue - group3_cost

  let group4_revenue := 1 * 550
  let group4_cost := 20 * 20 * 1
  let group4_profit := group4_revenue - group4_cost

  let total_profit := group1_profit + group2_profit + group3_profit + group4_profit
  total_profit = 2034 :=
by
  sorry

end NUMINAMATH_GPT_farmer_total_profit_l2157_215793


namespace NUMINAMATH_GPT_intersecting_point_value_l2157_215713

theorem intersecting_point_value (c d : ℤ) (h1 : d = 5 * (-5) + c) (h2 : -5 = 5 * d + c) : 
  d = -5 := 
sorry

end NUMINAMATH_GPT_intersecting_point_value_l2157_215713


namespace NUMINAMATH_GPT_prove_expression_value_l2157_215718

theorem prove_expression_value (x y : ℝ) (h1 : 4 * x + y = 18) (h2 : x + 4 * y = 20) :
  20 * x^2 + 16 * x * y + 20 * y^2 = 724 :=
sorry

end NUMINAMATH_GPT_prove_expression_value_l2157_215718


namespace NUMINAMATH_GPT_domain_of_f_l2157_215703

-- Define the function domain transformation
theorem domain_of_f (f : ℝ → ℝ) : 
  (∀ (x : ℝ), -2 ≤ x ∧ x ≤ 2 → -7 ≤ 2*x - 3 ∧ 2*x - 3 ≤ 1) ↔ (∀ (y : ℝ), -7 ≤ y ∧ y ≤ 1) :=
sorry

end NUMINAMATH_GPT_domain_of_f_l2157_215703


namespace NUMINAMATH_GPT_multiplication_more_than_subtraction_l2157_215776

def x : ℕ := 22

def multiplication_result : ℕ := 3 * x
def subtraction_result : ℕ := 62 - x
def difference : ℕ := multiplication_result - subtraction_result

theorem multiplication_more_than_subtraction : difference = 26 :=
by
  sorry

end NUMINAMATH_GPT_multiplication_more_than_subtraction_l2157_215776


namespace NUMINAMATH_GPT_mean_is_not_51_l2157_215730

def frequencies : List Nat := [5, 8, 7, 13, 7]
def pH_values : List Float := [4.8, 4.9, 5.0, 5.2, 5.3]

def total_observations : Nat := List.sum frequencies

def mean (freqs : List Nat) (values : List Float) : Float :=
  let weighted_sum := List.sum (List.zipWith (· * ·) values (List.map (Float.ofNat) freqs))
  weighted_sum / (Float.ofNat total_observations)

theorem mean_is_not_51 : mean frequencies pH_values ≠ 5.1 := by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_mean_is_not_51_l2157_215730


namespace NUMINAMATH_GPT_euler_disproven_conjecture_solution_l2157_215780

theorem euler_disproven_conjecture_solution : 
  ∃ (n : ℕ), n^5 = 133^5 + 110^5 + 84^5 + 27^5 ∧ n = 144 :=
by
  use 144
  have h : 144^5 = 133^5 + 110^5 + 84^5 + 27^5 := sorry
  exact ⟨h, rfl⟩

end NUMINAMATH_GPT_euler_disproven_conjecture_solution_l2157_215780


namespace NUMINAMATH_GPT_value_of_A_l2157_215731

theorem value_of_A (A : ℕ) : (A * 1000 + 567) % 100 < 50 → (A * 1000 + 567) / 10 * 10 = 2560 → A = 2 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_value_of_A_l2157_215731


namespace NUMINAMATH_GPT_eval_polynomial_correct_l2157_215771

theorem eval_polynomial_correct (y : ℝ) (hy : y^2 - 3 * y - 9 = 0) (hy_pos : 0 < y) :
  y^3 - 3 * y^2 - 9 * y + 3 = 3 :=
sorry

end NUMINAMATH_GPT_eval_polynomial_correct_l2157_215771


namespace NUMINAMATH_GPT_leak_drain_time_l2157_215710

/-- Statement: Given the rates at which a pump fills a tank and a leak drains the tank, 
prove that the leak can drain all the water in the tank in 14 hours. -/
theorem leak_drain_time :
  (∀ P L: ℝ, P = 1/2 → (P - L) = 3/7 → L = 1/14 → (1 / L) = 14) := 
by
  intros P L hP hPL hL
  -- Proof is omitted (to be provided)
  sorry

end NUMINAMATH_GPT_leak_drain_time_l2157_215710


namespace NUMINAMATH_GPT_order_of_abcd_l2157_215737

-- Define the rational numbers a, b, c, d
variables {a b c d : ℚ}

-- State the conditions as assumptions
axiom h1 : a + b = c + d
axiom h2 : a + d < b + c
axiom h3 : c < d

-- The goal is to prove the correct order of a, b, c, d
theorem order_of_abcd (a b c d : ℚ) (h1 : a + b = c + d) (h2 : a + d < b + c) (h3 : c < d) :
  b > d ∧ d > c ∧ c > a :=
sorry

end NUMINAMATH_GPT_order_of_abcd_l2157_215737


namespace NUMINAMATH_GPT_polar_to_cartesian_l2157_215760

-- Define the conditions
def polar_eq (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- Define the goal as a theorem
theorem polar_to_cartesian : ∀ (x y : ℝ), 
  (∃ θ : ℝ, polar_eq (Real.sqrt (x^2 + y^2)) θ ∧ x = (Real.sqrt (x^2 + y^2)) * Real.cos θ 
  ∧ y = (Real.sqrt (x^2 + y^2)) * Real.sin θ) → (x-1)^2 + y^2 = 1 :=
by
  intro x y
  intro h
  sorry

end NUMINAMATH_GPT_polar_to_cartesian_l2157_215760


namespace NUMINAMATH_GPT_axis_of_symmetry_parabola_l2157_215706

theorem axis_of_symmetry_parabola (x y : ℝ) :
  x^2 + 2*x*y + y^2 + 3*x + y = 0 → x + y + 1 = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_axis_of_symmetry_parabola_l2157_215706


namespace NUMINAMATH_GPT_complement_angle_l2157_215722

theorem complement_angle (A : Real) (h : A = 55) : 90 - A = 35 := by
  sorry

end NUMINAMATH_GPT_complement_angle_l2157_215722


namespace NUMINAMATH_GPT_find_sample_size_l2157_215759

-- Definitions based on conditions
def ratio_students : ℕ := 2 + 3 + 5
def grade12_ratio : ℚ := 5 / ratio_students
def sample_grade12_students : ℕ := 150

-- The goal is to find n such that the proportion is maintained
theorem find_sample_size (n : ℕ) (h : grade12_ratio = sample_grade12_students / ↑n) : n = 300 :=
by sorry


end NUMINAMATH_GPT_find_sample_size_l2157_215759


namespace NUMINAMATH_GPT_onion_harvest_scientific_notation_l2157_215746

theorem onion_harvest_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 325000000 = a * 10^n ∧ a = 3.25 ∧ n = 8 := 
by
  sorry

end NUMINAMATH_GPT_onion_harvest_scientific_notation_l2157_215746


namespace NUMINAMATH_GPT_next_divisor_after_391_l2157_215738

theorem next_divisor_after_391 (m : ℕ) (h1 : m % 2 = 0) (h2 : m ≥ 1000 ∧ m < 10000) (h3 : 391 ∣ m) : 
  ∃ n, n > 391 ∧ n ∣ m ∧ (∀ k, k > 391 ∧ k < n → ¬ k ∣ m) ∧ n = 782 :=
sorry

end NUMINAMATH_GPT_next_divisor_after_391_l2157_215738


namespace NUMINAMATH_GPT_E_plays_2_games_l2157_215789

-- Definitions for the students and the number of games they played
def students := ["A", "B", "C", "D", "E"]
def games_played_by (S : String) : Nat :=
  if S = "A" then 4 else
  if S = "B" then 3 else
  if S = "C" then 2 else 
  if S = "D" then 1 else
  2  -- this is the number of games we need to prove for student E 

-- Theorem stating the number of games played by E
theorem E_plays_2_games : games_played_by "E" = 2 :=
  sorry

end NUMINAMATH_GPT_E_plays_2_games_l2157_215789


namespace NUMINAMATH_GPT_complement_union_in_set_l2157_215736

open Set

theorem complement_union_in_set {U A B : Set ℕ} 
  (hU : U = {1, 3, 5, 9}) 
  (hA : A = {1, 3, 9}) 
  (hB : B = {1, 9}) : 
  (U \ (A ∪ B)) = {5} := 
  by sorry

end NUMINAMATH_GPT_complement_union_in_set_l2157_215736


namespace NUMINAMATH_GPT_transformation_identity_l2157_215774

theorem transformation_identity (n : Nat) (h : 2 ≤ n) : 
  n * Real.sqrt (n / (n ^ 2 - 1)) = Real.sqrt (n + n / (n ^ 2 - 1)) := 
sorry

end NUMINAMATH_GPT_transformation_identity_l2157_215774


namespace NUMINAMATH_GPT_flavors_remaining_to_try_l2157_215761

def total_flavors : ℕ := 100
def flavors_tried_two_years_ago (total_flavors : ℕ) : ℕ := total_flavors / 4
def flavors_tried_last_year (flavors_tried_two_years_ago : ℕ) : ℕ := 2 * flavors_tried_two_years_ago

theorem flavors_remaining_to_try
  (total_flavors : ℕ)
  (flavors_tried_two_years_ago : ℕ)
  (flavors_tried_last_year : ℕ) :
  flavors_tried_two_years_ago = total_flavors / 4 →
  flavors_tried_last_year = 2 * flavors_tried_two_years_ago →
  total_flavors - (flavors_tried_two_years_ago + flavors_tried_last_year) = 25 :=
by
  sorry

end NUMINAMATH_GPT_flavors_remaining_to_try_l2157_215761


namespace NUMINAMATH_GPT_child_is_late_l2157_215758

theorem child_is_late 
  (distance : ℕ)
  (rate1 rate2 : ℕ) 
  (early_arrival : ℕ)
  (time_late_at_rate1 : ℕ)
  (time_required_by_rate1 : ℕ)
  (time_required_by_rate2 : ℕ)
  (actual_time : ℕ)
  (T : ℕ) :
  distance = 630 ∧ 
  rate1 = 5 ∧ 
  rate2 = 7 ∧ 
  early_arrival = 30 ∧
  (time_required_by_rate1 = distance / rate1) ∧
  (time_required_by_rate2 = distance / rate2) ∧
  (actual_time + T = time_required_by_rate1) ∧
  (actual_time - early_arrival = time_required_by_rate2) →
  T = 6 := 
by
  intros
  sorry

end NUMINAMATH_GPT_child_is_late_l2157_215758


namespace NUMINAMATH_GPT_average_person_funding_l2157_215739

-- Define the conditions from the problem
def total_amount_needed : ℝ := 1000
def amount_already_have : ℝ := 200
def number_of_people : ℝ := 80

-- Define the correct answer
def average_funding_per_person : ℝ := 10

-- Formulate the proof statement
theorem average_person_funding :
  (total_amount_needed - amount_already_have) / number_of_people = average_funding_per_person :=
by
  sorry

end NUMINAMATH_GPT_average_person_funding_l2157_215739


namespace NUMINAMATH_GPT_tetrahedron_edge_length_l2157_215707

theorem tetrahedron_edge_length (a : ℝ) (V : ℝ) 
  (h₀ : V = 0.11785113019775793) 
  (h₁ : V = (Real.sqrt 2 / 12) * a^3) : a = 1 := by
  sorry

end NUMINAMATH_GPT_tetrahedron_edge_length_l2157_215707


namespace NUMINAMATH_GPT_adam_change_l2157_215743

-- Defining the given amount Adam has and the cost of the airplane.
def amountAdamHas : ℝ := 5.00
def costOfAirplane : ℝ := 4.28

-- Statement of the theorem to be proven.
theorem adam_change : amountAdamHas - costOfAirplane = 0.72 := by
  sorry

end NUMINAMATH_GPT_adam_change_l2157_215743


namespace NUMINAMATH_GPT_percent_of_number_l2157_215787

theorem percent_of_number (N : ℝ) (h : (4 / 5) * (3 / 8) * N = 24) : 2.5 * N = 200 :=
by
  sorry

end NUMINAMATH_GPT_percent_of_number_l2157_215787


namespace NUMINAMATH_GPT_exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l2157_215720

-- Definition: A number is composite if it has more than two distinct positive divisors
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ (d : ℕ), d > 1 ∧ d < n ∧ n % d = 0

-- There exists a sequence of nine consecutive composite numbers within the first 500
theorem exists_nine_consecutive_composites :
  ∃ (seq : Fin 500 → ℕ), (∀ i : Fin 500, seq i > 0 ∧ seq i ≤ 500 ∧ is_composite (seq i)) ∧ 
                           ∃ (start : ℕ), start + 8 < 500 ∧
                           (∀ i, i < 9 -> is_composite (seq (⟨start + i, sorry⟩ : Fin 500))) := sorry

-- There exists a sequence of eleven consecutive composite numbers within the first 500
theorem exists_eleven_consecutive_composites :
  ∃ (seq : Fin 500 → ℕ), (∀ i : Fin 500, seq i > 0 ∧ seq i ≤ 500 ∧ is_composite (seq i)) ∧ 
                           ∃ (start : ℕ), start + 10 < 500 ∧
                           (∀ i, i < 11 -> is_composite (seq (⟨start + i, sorry⟩ : Fin 500))) := sorry

end NUMINAMATH_GPT_exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l2157_215720


namespace NUMINAMATH_GPT_parabola_tangent_hyperbola_l2157_215729

theorem parabola_tangent_hyperbola (m : ℝ) :
  (∀ x : ℝ, (x^2 + 5)^2 - m * x^2 = 4 → y = x^2 + 5)
  ∧ (∀ y : ℝ, y ≥ 5 → y^2 - m * x^2 = 4) →
  (m = 10 + 2 * Real.sqrt 21 ∨ m = 10 - 2 * Real.sqrt 21) :=
  sorry

end NUMINAMATH_GPT_parabola_tangent_hyperbola_l2157_215729


namespace NUMINAMATH_GPT_probability_single_trial_l2157_215767

theorem probability_single_trial (p : ℚ) (h₁ : (1 - p)^4 = 16 / 81) : p = 1 / 3 :=
sorry

end NUMINAMATH_GPT_probability_single_trial_l2157_215767


namespace NUMINAMATH_GPT_find_sum_of_abcd_l2157_215716

theorem find_sum_of_abcd (a b c d : ℚ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 10) :
  a + b + c + d = -26 / 3 :=
sorry

end NUMINAMATH_GPT_find_sum_of_abcd_l2157_215716


namespace NUMINAMATH_GPT_jasmine_total_cost_l2157_215711

noncomputable def total_cost_jasmine
  (coffee_beans_amount : ℕ)
  (milk_amount : ℕ)
  (coffee_beans_cost : ℝ)
  (milk_cost : ℝ)
  (discount_combined : ℝ)
  (additional_discount_milk : ℝ)
  (tax_rate : ℝ) : ℝ :=
  let total_before_discounts := coffee_beans_amount * coffee_beans_cost + milk_amount * milk_cost
  let total_after_combined_discount := total_before_discounts - discount_combined * total_before_discounts
  let milk_cost_after_additional_discount := milk_amount * milk_cost - additional_discount_milk * (milk_amount * milk_cost)
  let total_after_all_discounts := coffee_beans_amount * coffee_beans_cost + milk_cost_after_additional_discount
  let tax := tax_rate * total_after_all_discounts
  total_after_all_discounts + tax

theorem jasmine_total_cost :
  total_cost_jasmine 4 2 2.50 3.50 0.10 0.05 0.08 = 17.98 :=
by
  unfold total_cost_jasmine
  sorry

end NUMINAMATH_GPT_jasmine_total_cost_l2157_215711


namespace NUMINAMATH_GPT_final_reduced_price_l2157_215784

noncomputable def original_price (P : ℝ) (Q : ℝ) : ℝ := 800 / Q

noncomputable def price_after_first_week (P : ℝ) : ℝ := 0.90 * P
noncomputable def price_after_second_week (price1 : ℝ) : ℝ := 0.85 * price1
noncomputable def price_after_third_week (price2 : ℝ) : ℝ := 0.80 * price2

noncomputable def reduced_price (P : ℝ) : ℝ :=
  let price1 := price_after_first_week P
  let price2 := price_after_second_week price1
  price_after_third_week price2

theorem final_reduced_price :
  ∃ P Q : ℝ, 
    800 = Q * P ∧
    800 = (Q + 5) * reduced_price P ∧
    abs (reduced_price P - 62.06) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_final_reduced_price_l2157_215784


namespace NUMINAMATH_GPT_find_principal_l2157_215773

theorem find_principal
  (P R : ℝ)
  (h : (P * (R + 2) * 7) / 100 = (P * R * 7) / 100 + 140) :
  P = 1000 := by
sorry

end NUMINAMATH_GPT_find_principal_l2157_215773


namespace NUMINAMATH_GPT_part_I_solution_part_II_solution_l2157_215786

noncomputable def f (x a : ℝ) : ℝ := |x - a| - 2 * |x - 1|

theorem part_I_solution :
  ∀ x : ℝ, f x 3 ≥ 1 ↔ 0 ≤ x ∧ x ≤ (4 / 3) := by
  sorry

theorem part_II_solution :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x a - |2*x - 5| ≤ 0) ↔ (-1 ≤ a ∧ a ≤ 4) := by
  sorry

end NUMINAMATH_GPT_part_I_solution_part_II_solution_l2157_215786


namespace NUMINAMATH_GPT_ball_bounce_height_l2157_215732

theorem ball_bounce_height (initial_height : ℝ) (r : ℝ) (k : ℕ) : 
  initial_height = 1000 → r = 1/2 → (r ^ k * initial_height < 1) → k = 10 := by
sorry

end NUMINAMATH_GPT_ball_bounce_height_l2157_215732


namespace NUMINAMATH_GPT_sqrt_condition_iff_l2157_215702

theorem sqrt_condition_iff (x : ℝ) : (∃ y : ℝ, y = (2 * x + 3) ∧ (0 ≤ y)) ↔ (x ≥ -3 / 2) :=
by sorry

end NUMINAMATH_GPT_sqrt_condition_iff_l2157_215702


namespace NUMINAMATH_GPT_find_b_l2157_215791

theorem find_b (b : ℚ) (m : ℚ) 
  (h1 : x^2 + b*x + 1/6 = (x + m)^2 + 1/18) 
  (h2 : b < 0) : 
  b = -2/3 := 
sorry

end NUMINAMATH_GPT_find_b_l2157_215791


namespace NUMINAMATH_GPT_WillyLucyHaveMoreCrayons_l2157_215705

-- Definitions from the conditions
def WillyCrayons : ℕ := 1400
def LucyCrayons : ℕ := 290
def MaxCrayons : ℕ := 650

-- Theorem statement
theorem WillyLucyHaveMoreCrayons : WillyCrayons + LucyCrayons - MaxCrayons = 1040 := 
by 
  sorry

end NUMINAMATH_GPT_WillyLucyHaveMoreCrayons_l2157_215705


namespace NUMINAMATH_GPT_chocolates_divisible_l2157_215781

theorem chocolates_divisible (n m : ℕ) (h1 : n > 0) (h2 : m > 0) : 
  (n ≤ m) ∨ (m % (n - m) = 0) :=
sorry

end NUMINAMATH_GPT_chocolates_divisible_l2157_215781


namespace NUMINAMATH_GPT_range_of_x_l2157_215772

theorem range_of_x (f : ℝ → ℝ) (h_increasing : ∀ x y, x ≤ y → f x ≤ f y) (h_defined : ∀ x, -1 ≤ x ∧ x ≤ 1)
  (h_condition : ∀ x, f (x-2) < f (1-x)) : ∀ x, 1 ≤ x ∧ x < 3/2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l2157_215772


namespace NUMINAMATH_GPT_unpainted_unit_cubes_l2157_215721

theorem unpainted_unit_cubes (total_units : ℕ) (painted_per_face : ℕ) (painted_edges_adjustment : ℕ) :
  total_units = 216 → painted_per_face = 12 → painted_edges_adjustment = 36 → 
  total_units - (painted_per_face * 6 - painted_edges_adjustment) = 108 :=
by
  intros h_tot_units h_painted_face h_edge_adj
  sorry

end NUMINAMATH_GPT_unpainted_unit_cubes_l2157_215721


namespace NUMINAMATH_GPT_perfect_square_trinomial_l2157_215765

theorem perfect_square_trinomial (k : ℝ) : (∃ a b : ℝ, (a * x + b) ^ 2 = x^2 - k * x + 4) → (k = 4 ∨ k = -4) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l2157_215765


namespace NUMINAMATH_GPT_valid_numbers_are_135_and_144_l2157_215749

noncomputable def find_valid_numbers : List ℕ :=
  let numbers := [135, 144]
  numbers.filter (λ n =>
    let a := n / 100
    let b := (n / 10) % 10
    let c := n % 10
    n = (100 * a + 10 * b + c) ∧ n = a * b * c * (a + b + c)
  )

theorem valid_numbers_are_135_and_144 :
  find_valid_numbers = [135, 144] :=
by
  sorry

end NUMINAMATH_GPT_valid_numbers_are_135_and_144_l2157_215749


namespace NUMINAMATH_GPT_minimum_value_l2157_215763

theorem minimum_value (n : ℝ) (h : n > 0) : n + 32 / n^2 ≥ 6 := 
sorry

end NUMINAMATH_GPT_minimum_value_l2157_215763


namespace NUMINAMATH_GPT_arithmetic_sequence_50th_term_l2157_215748

theorem arithmetic_sequence_50th_term :
  let a_1 := 3
  let d := 7
  let n := 50
  (a_1 + (n - 1) * d) = 346 :=
by
  let a_1 := 3
  let d := 7
  let n := 50
  show (a_1 + (n - 1) * d) = 346
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_50th_term_l2157_215748


namespace NUMINAMATH_GPT_sophomores_selected_correct_l2157_215798

-- Define the number of students in each grade and the total spots for the event
def freshmen : ℕ := 240
def sophomores : ℕ := 260
def juniors : ℕ := 300
def totalSpots : ℕ := 40

-- Calculate the total number of students
def totalStudents : ℕ := freshmen + sophomores + juniors

-- The correct answer we want to prove
def numberOfSophomoresSelected : ℕ := (sophomores * totalSpots) / totalStudents

-- Statement to be proved
theorem sophomores_selected_correct : numberOfSophomoresSelected = 26 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_sophomores_selected_correct_l2157_215798


namespace NUMINAMATH_GPT_wendy_points_earned_l2157_215709

-- Define the conditions
def points_per_bag : ℕ := 5
def total_bags : ℕ := 11
def bags_not_recycled : ℕ := 2

-- Define the statement to be proved
theorem wendy_points_earned : (total_bags - bags_not_recycled) * points_per_bag = 45 :=
by
  sorry

end NUMINAMATH_GPT_wendy_points_earned_l2157_215709


namespace NUMINAMATH_GPT_square_of_binomial_l2157_215708

theorem square_of_binomial (c : ℝ) : (∃ b : ℝ, ∀ x : ℝ, 9 * x^2 - 30 * x + c = (3 * x + b)^2) ↔ c = 25 :=
by
  sorry

end NUMINAMATH_GPT_square_of_binomial_l2157_215708


namespace NUMINAMATH_GPT_problem_statement_l2157_215727

noncomputable def f (n : ℕ) (x : ℝ) : ℝ := x^n

variable (a : ℝ)
variable (h : a ≠ 1)

theorem problem_statement :
  (f 11 (f 13 a)) ^ 14 = f 2002 a ∧
  f 11 (f 13 (f 14 a)) = f 2002 a :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2157_215727


namespace NUMINAMATH_GPT_arithmetic_prog_includes_1999_l2157_215744

-- Definitions based on problem conditions
def is_in_arithmetic_progression (a d n : ℕ) : ℕ := a + (n - 1) * d

theorem arithmetic_prog_includes_1999
  (d : ℕ) (h_pos : d > 0) 
  (h_includes7 : ∃ n:ℕ, is_in_arithmetic_progression 7 d n = 7)
  (h_includes15 : ∃ n:ℕ, is_in_arithmetic_progression 7 d n = 15)
  (h_includes27 : ∃ n:ℕ, is_in_arithmetic_progression 7 d n = 27) :
  ∃ n:ℕ, is_in_arithmetic_progression 7 d n = 1999 := 
sorry

end NUMINAMATH_GPT_arithmetic_prog_includes_1999_l2157_215744


namespace NUMINAMATH_GPT_max_singular_words_l2157_215714

theorem max_singular_words (alphabet_length : ℕ) (word_length : ℕ) (strip_length : ℕ) 
  (num_non_overlapping_pieces : ℕ) (h_alphabet : alphabet_length = 25)
  (h_word_length : word_length = 17) (h_strip_length : strip_length = 5^18)
  (h_non_overlapping : num_non_overlapping_pieces = 5^16) : 
  ∃ max_singular_words, max_singular_words = 2 * 5^17 :=
by {
  -- proof to be completed
  sorry
}

end NUMINAMATH_GPT_max_singular_words_l2157_215714


namespace NUMINAMATH_GPT_net_investment_change_l2157_215725

def initial_investment : ℝ := 100
def first_year_increase (init : ℝ) : ℝ := init * 1.50
def second_year_decrease (value : ℝ) : ℝ := value * 0.70

theorem net_investment_change :
  second_year_decrease (first_year_increase initial_investment) - initial_investment = 5 :=
by
  -- This will be placeholder proof
  sorry

end NUMINAMATH_GPT_net_investment_change_l2157_215725


namespace NUMINAMATH_GPT_cindy_total_time_to_travel_one_mile_l2157_215797

-- Definitions for the conditions
def run_speed : ℝ := 3 -- Cindy's running speed in miles per hour.
def walk_speed : ℝ := 1 -- Cindy's walking speed in miles per hour.
def run_distance : ℝ := 0.5 -- Distance run by Cindy in miles.
def walk_distance : ℝ := 0.5 -- Distance walked by Cindy in miles.

-- Theorem statement
theorem cindy_total_time_to_travel_one_mile : 
  ((run_distance / run_speed) + (walk_distance / walk_speed)) * 60 = 40 := 
by
  sorry

end NUMINAMATH_GPT_cindy_total_time_to_travel_one_mile_l2157_215797


namespace NUMINAMATH_GPT_log_expression_value_l2157_215799

noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

theorem log_expression_value :
  (log2 8 * (log2 2 / log2 8)) + log2 4 = 3 :=
by
  sorry

end NUMINAMATH_GPT_log_expression_value_l2157_215799


namespace NUMINAMATH_GPT_PartA_l2157_215751

variable (f : ℝ → ℝ)
variable (h_diff : Differentiable ℝ f)
variable (h_eq : ∀ x, f (f x) = f x)

theorem PartA : ∀ x, (deriv f x = 0) ∨ (deriv f (f x) = 1) :=
by
  sorry

end NUMINAMATH_GPT_PartA_l2157_215751


namespace NUMINAMATH_GPT_find_b_compare_f_l2157_215735

-- Definition from conditions
def f (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := -x^2 + b*x + c

-- Part 1: Prove that b = 4
theorem find_b (b c : ℝ) (h : ∀ x : ℝ, f (2 + x) b c = f (2 - x) b c) : b = 4 :=
sorry

-- Part 2: Prove the comparison of f(\frac{5}{4}) and f(-a^2 - a + 1)
theorem compare_f (c : ℝ) (a : ℝ) (h₁ : ∀ x : ℝ, f (2 + x) 4 c = f (2 - x) 4 c) (h₂ : f (5/4) 4 c < f (-(a^2 + a - 1)) 4 c) :
f (5/4) 4 c < f (-(a^2 + a - 1)) 4 c := 
sorry

end NUMINAMATH_GPT_find_b_compare_f_l2157_215735


namespace NUMINAMATH_GPT_optimal_strategy_l2157_215747

-- Define the conditions
def valid_N (N : ℤ) : Prop :=
  0 ≤ N ∧ N ≤ 20

def score (N : ℤ) (other_teams_count : ℤ) : ℤ :=
  if other_teams_count > N then N else 0

-- The mathematical problem statement
theorem optimal_strategy : ∃ N : ℤ, valid_N N ∧ (∀ other_teams_count : ℤ, score 1 other_teams_count ≥ score N other_teams_count ∧ score 1 other_teams_count ≠ 0) :=
sorry

end NUMINAMATH_GPT_optimal_strategy_l2157_215747


namespace NUMINAMATH_GPT_napkins_total_l2157_215757

theorem napkins_total (o a w : ℕ) (ho : o = 10) (ha : a = 2 * o) (hw : w = 15) :
  w + o + a = 45 :=
by
  sorry

end NUMINAMATH_GPT_napkins_total_l2157_215757


namespace NUMINAMATH_GPT_unique_n_divides_2_pow_n_minus_1_l2157_215777

theorem unique_n_divides_2_pow_n_minus_1 (n : ℕ) (h : n ∣ 2^n - 1) : n = 1 :=
sorry

end NUMINAMATH_GPT_unique_n_divides_2_pow_n_minus_1_l2157_215777


namespace NUMINAMATH_GPT_blocks_given_by_father_l2157_215745

theorem blocks_given_by_father :
  ∀ (blocks_original total_blocks blocks_given : ℕ), 
  blocks_original = 2 →
  total_blocks = 8 →
  blocks_given = total_blocks - blocks_original →
  blocks_given = 6 :=
by
  intros blocks_original total_blocks blocks_given h1 h2 h3
  sorry

end NUMINAMATH_GPT_blocks_given_by_father_l2157_215745


namespace NUMINAMATH_GPT_height_of_pyramid_equal_to_cube_volume_l2157_215712

theorem height_of_pyramid_equal_to_cube_volume :
  (∃ h : ℝ, (5:ℝ)^3 = (1/3:ℝ) * (10:ℝ)^2 * h) ↔ h = 3.75 :=
by
  sorry

end NUMINAMATH_GPT_height_of_pyramid_equal_to_cube_volume_l2157_215712


namespace NUMINAMATH_GPT_bridge_must_hold_weight_l2157_215700

def weight_of_full_can (soda_weight empty_can_weight : ℕ) : ℕ :=
  soda_weight + empty_can_weight

def total_weight_of_full_cans (num_full_cans weight_per_full_can : ℕ) : ℕ :=
  num_full_cans * weight_per_full_can

def total_weight_of_empty_cans (num_empty_cans empty_can_weight : ℕ) : ℕ :=
  num_empty_cans * empty_can_weight

theorem bridge_must_hold_weight :
  let num_full_cans := 6
  let soda_weight := 12
  let empty_can_weight := 2
  let num_empty_cans := 2
  let weight_per_full_can := weight_of_full_can soda_weight empty_can_weight
  let total_full_cans_weight := total_weight_of_full_cans num_full_cans weight_per_full_can
  let total_empty_cans_weight := total_weight_of_empty_cans num_empty_cans empty_can_weight
  total_full_cans_weight + total_empty_cans_weight = 88 := by
  sorry

end NUMINAMATH_GPT_bridge_must_hold_weight_l2157_215700


namespace NUMINAMATH_GPT_evaluate_expression_l2157_215766

theorem evaluate_expression :
  abs ((4^2 - 8 * (3^2 - 12))^2) - abs (Real.sin (5 * Real.pi / 6) - Real.cos (11 * Real.pi / 3)) = 1600 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2157_215766


namespace NUMINAMATH_GPT_shape_described_by_constant_phi_is_cone_l2157_215779

-- Definition of spherical coordinates
-- (ρ, θ, φ) where ρ is the radial distance,
-- θ is the azimuthal angle, and φ is the polar angle.
structure SphericalCoordinates :=
  (ρ : ℝ)
  (θ : ℝ)
  (φ : ℝ)

-- The condition that φ is equal to a constant d
def satisfies_condition (p : SphericalCoordinates) (d : ℝ) : Prop :=
  p.φ = d

-- The main theorem to prove
theorem shape_described_by_constant_phi_is_cone (d : ℝ) :
  ∃ (S : Set SphericalCoordinates), (∀ p ∈ S, satisfies_condition p d) ∧
  (∀ p, satisfies_condition p d → ∃ ρ θ, p = ⟨ρ, θ, d⟩) ∧
  (∀ ρ θ, ρ > 0 → θ ∈ [0, 2 * Real.pi] → SphericalCoordinates.mk ρ θ d ∈ S) :=
sorry

end NUMINAMATH_GPT_shape_described_by_constant_phi_is_cone_l2157_215779


namespace NUMINAMATH_GPT_arithmetic_sequence_a2_a9_l2157_215719

theorem arithmetic_sequence_a2_a9 (a : ℕ → ℚ) (d : ℚ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_sum : a 5 + a 6 = 12) :
  a 2 + a 9 = 12 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a2_a9_l2157_215719


namespace NUMINAMATH_GPT_find_angle_A_l2157_215717

-- Variables representing angles A and B
variables (A B : ℝ)

-- The conditions of the problem translated into Lean
def angle_relationship := A = 2 * B - 15
def angle_supplementary := A + B = 180

-- The theorem statement we need to prove
theorem find_angle_A (h1 : angle_relationship A B) (h2 : angle_supplementary A B) : A = 115 :=
by { sorry }

end NUMINAMATH_GPT_find_angle_A_l2157_215717


namespace NUMINAMATH_GPT_percentage_increase_240_to_288_l2157_215756

theorem percentage_increase_240_to_288 :
  let initial := 240
  let final := 288
  ((final - initial) / initial) * 100 = 20 := by 
  sorry

end NUMINAMATH_GPT_percentage_increase_240_to_288_l2157_215756


namespace NUMINAMATH_GPT_radius_of_spheres_in_cone_l2157_215724

-- Given Definitions
def cone_base_radius : ℝ := 6
def cone_height : ℝ := 15
def tangent_spheres (r : ℝ) : Prop :=
  r = (12 * Real.sqrt 29) / 29

-- Problem Statement
theorem radius_of_spheres_in_cone :
  ∃ r : ℝ, tangent_spheres r :=
sorry

end NUMINAMATH_GPT_radius_of_spheres_in_cone_l2157_215724


namespace NUMINAMATH_GPT_largest_prime_factor_of_1729_is_19_l2157_215796

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def largest_prime_factor (n : ℕ) (p : ℕ) := is_prime p ∧ p ∣ n ∧ ∀ q : ℕ, is_prime q ∧ q ∣ n → q ≤ p

theorem largest_prime_factor_of_1729_is_19 : largest_prime_factor 1729 19 :=
by
  sorry

end NUMINAMATH_GPT_largest_prime_factor_of_1729_is_19_l2157_215796


namespace NUMINAMATH_GPT_find_value_a2_b2_c2_l2157_215795

variable (a b c p q r : ℝ)
variable (h1 : a * b = p)
variable (h2 : b * c = q)
variable (h3 : c * a = r)
variable (h4 : p ≠ 0)
variable (h5 : q ≠ 0)
variable (h6 : r ≠ 0)

theorem find_value_a2_b2_c2 : a^2 + b^2 + c^2 = 1 :=
by sorry

end NUMINAMATH_GPT_find_value_a2_b2_c2_l2157_215795


namespace NUMINAMATH_GPT_union_of_sets_l2157_215723

def A : Set ℤ := {0, 1}
def B : Set ℤ := {1, 2}

theorem union_of_sets :
  A ∪ B = {0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_union_of_sets_l2157_215723


namespace NUMINAMATH_GPT_volume_of_cube_surface_area_times_l2157_215792

theorem volume_of_cube_surface_area_times (V1 : ℝ) (hV1 : V1 = 8) : 
  ∃ V2, V2 = 24 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_volume_of_cube_surface_area_times_l2157_215792


namespace NUMINAMATH_GPT_comparison_of_negatives_l2157_215750

theorem comparison_of_negatives : -2 < - (3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_comparison_of_negatives_l2157_215750


namespace NUMINAMATH_GPT_convex_over_real_l2157_215752

def f (x : ℝ) : ℝ := x^4 - 2 * x^3 + 36 * x^2 - x + 7

theorem convex_over_real : ∀ x : ℝ, 0 ≤ (12 * x^2 - 12 * x + 72) :=
by sorry

end NUMINAMATH_GPT_convex_over_real_l2157_215752


namespace NUMINAMATH_GPT_selling_price_ratio_l2157_215764

theorem selling_price_ratio (CP SP1 SP2 : ℝ) (h1 : SP1 = CP + 0.5 * CP) (h2 : SP2 = CP + 3 * CP) :
  SP2 / SP1 = 8 / 3 :=
by
  sorry

end NUMINAMATH_GPT_selling_price_ratio_l2157_215764


namespace NUMINAMATH_GPT_tables_needed_l2157_215785

-- Conditions
def n_invited : ℕ := 18
def n_no_show : ℕ := 12
def capacity_per_table : ℕ := 3

-- Calculation of attendees
def n_attendees : ℕ := n_invited - n_no_show

-- Proof for the number of tables needed
theorem tables_needed : (n_attendees / capacity_per_table) = 2 := by
  -- Sorry will be here to show it's incomplete
  sorry

end NUMINAMATH_GPT_tables_needed_l2157_215785


namespace NUMINAMATH_GPT_correct_option_D_l2157_215740

theorem correct_option_D (a : ℝ) : (-a^3)^2 = a^6 :=
sorry

end NUMINAMATH_GPT_correct_option_D_l2157_215740


namespace NUMINAMATH_GPT_initial_cupcakes_l2157_215778

   theorem initial_cupcakes (X : ℕ) (condition : X - 20 + 20 = 26) : X = 26 :=
   by
     sorry
   
end NUMINAMATH_GPT_initial_cupcakes_l2157_215778


namespace NUMINAMATH_GPT_solve_for_n_l2157_215715

theorem solve_for_n (n : ℕ) (h : 2 * n - 5 = 1) : n = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_n_l2157_215715


namespace NUMINAMATH_GPT_remainder_of_polynomial_division_l2157_215775

-- Definitions based on conditions in the problem
def polynomial (x : ℝ) : ℝ := 8 * x^4 - 22 * x^3 + 9 * x^2 + 10 * x - 45

def divisor (x : ℝ) : ℝ := 4 * x - 8

-- Proof statement as per the problem equivalence
theorem remainder_of_polynomial_division : polynomial 2 = -37 := by
  sorry

end NUMINAMATH_GPT_remainder_of_polynomial_division_l2157_215775


namespace NUMINAMATH_GPT_f_increasing_on_interval_l2157_215701

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (x^2, x + 1)
noncomputable def vec_b (x t : ℝ) : ℝ × ℝ := (1 - x, t)

noncomputable def f (x t : ℝ) : ℝ :=
  let (a1, a2) := vec_a x
  let (b1, b2) := vec_b x t
  a1 * b1 + a2 * b2

noncomputable def f_prime (x t : ℝ) : ℝ :=
  2 * x - 3 * x^2 + t

theorem f_increasing_on_interval :
  ∀ t x, -1 < x → x < 1 → (0 ≤ f_prime x t) → (t ≥ 5) :=
sorry

end NUMINAMATH_GPT_f_increasing_on_interval_l2157_215701


namespace NUMINAMATH_GPT_emily_subtracts_99_l2157_215755

theorem emily_subtracts_99 : ∀ (a b : ℕ), (51 * 51 = a + 101) → (49 * 49 = b - 99) → b - 99 = 2401 := by
  intros a b h1 h2
  sorry

end NUMINAMATH_GPT_emily_subtracts_99_l2157_215755


namespace NUMINAMATH_GPT_line_perpendicular_to_plane_implies_parallel_l2157_215741

-- Definitions for lines and planes in space
axiom Line : Type
axiom Plane : Type

-- Relation of perpendicularity between a line and a plane
axiom perp : Line → Plane → Prop

-- Relation of parallelism between two lines
axiom parallel : Line → Line → Prop

-- The theorem to be proved
theorem line_perpendicular_to_plane_implies_parallel (x y : Line) (z : Plane) :
  perp x z → perp y z → parallel x y :=
by sorry

end NUMINAMATH_GPT_line_perpendicular_to_plane_implies_parallel_l2157_215741


namespace NUMINAMATH_GPT_radius_smaller_circle_l2157_215769

theorem radius_smaller_circle (A₁ A₂ A₃ : ℝ) (s : ℝ)
  (h1 : A₁ + A₂ = 12 * Real.pi)
  (h2 : A₃ = (Real.sqrt 3 / 4) * s^2)
  (h3 : 2 * A₂ = A₁ + A₁ + A₂ + A₃) :
  ∃ r : ℝ, r = Real.sqrt (6 - (Real.sqrt 3 / 8) * s^2) := by
  sorry

end NUMINAMATH_GPT_radius_smaller_circle_l2157_215769


namespace NUMINAMATH_GPT_system1_solution_system2_solution_l2157_215794

-- Statement for the Part 1 Equivalent Problem.
theorem system1_solution :
  ∀ (x y : ℤ),
    (x - 3 * y = -10) ∧ (x + y = 6) → (x = 2 ∧ y = 4) :=
by
  intros x y h
  rcases h with ⟨h1, h2⟩
  sorry

-- Statement for the Part 2 Equivalent Problem.
theorem system2_solution :
  ∀ (x y : ℚ),
    (x / 2 - (y - 1) / 3 = 1) ∧ (4 * x - y = 8) → (x = 12 / 5 ∧ y = 8 / 5) :=
by
  intros x y h
  rcases h with ⟨h1, h2⟩
  sorry

end NUMINAMATH_GPT_system1_solution_system2_solution_l2157_215794


namespace NUMINAMATH_GPT_initial_kola_volume_l2157_215753

theorem initial_kola_volume (V : ℝ) (S : ℝ) :
  S = 0.14 * V →
  (S + 3.2) / (V + 20) = 0.14111111111111112 →
  V = 340 :=
by
  intro h_S h_equation
  sorry

end NUMINAMATH_GPT_initial_kola_volume_l2157_215753


namespace NUMINAMATH_GPT_inequality_proof_l2157_215733

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
    (((2 + x) / (1 + x))^2 + ((2 + y) / (1 + y))^2) ≥ 9 / 2 := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l2157_215733


namespace NUMINAMATH_GPT_task1_task2_l2157_215788

-- Define the conditions and the probabilities to be proven

def total_pens := 6
def first_class_pens := 3
def second_class_pens := 2
def third_class_pens := 1

def total_combinations := Nat.choose total_pens 2

def combinations_with_exactly_one_first_class : Nat :=
  (first_class_pens * (total_pens - first_class_pens))

def probability_one_first_class_pen : ℚ :=
  combinations_with_exactly_one_first_class / total_combinations

def combinations_without_any_third_class : Nat :=
  Nat.choose (first_class_pens + second_class_pens) 2

def probability_no_third_class_pen : ℚ :=
  combinations_without_any_third_class / total_combinations

theorem task1 : probability_one_first_class_pen = 3 / 5 := 
  sorry

theorem task2 : probability_no_third_class_pen = 2 / 3 := 
  sorry

end NUMINAMATH_GPT_task1_task2_l2157_215788


namespace NUMINAMATH_GPT_example_one_example_two_l2157_215782

-- We will define natural numbers corresponding to seven, seventy-seven, and seven hundred seventy-seven.
def seven : ℕ := 7
def seventy_seven : ℕ := 77
def seven_hundred_seventy_seven : ℕ := 777

-- We will define both solutions in the form of equalities producing 100.
theorem example_one : (seven_hundred_seventy_seven / seven) - (seventy_seven / seven) = 100 :=
  by sorry

theorem example_two : (seven * seven) + (seven * seven) + (seven / seven) + (seven / seven) = 100 :=
  by sorry

end NUMINAMATH_GPT_example_one_example_two_l2157_215782


namespace NUMINAMATH_GPT_monotonicity_intervals_inequality_condition_l2157_215728

noncomputable def f (x : ℝ) := Real.exp x * (x^2 + 2 * x + 1)

theorem monotonicity_intervals :
  (∀ x ∈ Set.Iio (-3 : ℝ), 0 < (Real.exp x * ((x + 3) * (x + 1)))) ∧
  (∀ x ∈ Set.Ioo (-3 : ℝ) (-1 : ℝ), 0 > (Real.exp x * ((x + 3) * (x + 1)))) ∧
  (∀ x ∈ Set.Ioi (-1 : ℝ), 0 < (Real.exp x * ((x + 3) * (x + 1)))) := sorry

theorem inequality_condition (a : ℝ) : 
  (∀ x > 0, Real.exp x * (x^2 + 2 * x + 1) > a * x^2 + a * x + 1) ↔ a ≤ 3 := sorry

end NUMINAMATH_GPT_monotonicity_intervals_inequality_condition_l2157_215728


namespace NUMINAMATH_GPT_blueberries_in_each_blue_box_l2157_215783

theorem blueberries_in_each_blue_box (S B : ℕ) (h1 : S - B = 12) (h2 : 2 * S = 76) : B = 26 := by
  sorry

end NUMINAMATH_GPT_blueberries_in_each_blue_box_l2157_215783


namespace NUMINAMATH_GPT_solve_for_alpha_l2157_215762

variables (α β γ δ : ℝ)

theorem solve_for_alpha (h : α + β + γ + δ = 360) : α = 360 - β - γ - δ :=
by sorry

end NUMINAMATH_GPT_solve_for_alpha_l2157_215762
