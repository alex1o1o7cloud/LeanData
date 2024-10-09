import Mathlib

namespace additional_time_due_to_leak_l1855_185594

theorem additional_time_due_to_leak 
  (normal_time_per_barrel : ℕ)
  (leak_time_per_barrel : ℕ)
  (barrels : ℕ)
  (normal_duration : normal_time_per_barrel = 3)
  (leak_duration : leak_time_per_barrel = 5)
  (barrels_needed : barrels = 12) :
  (leak_time_per_barrel * barrels - normal_time_per_barrel * barrels) = 24 := 
by
  sorry

end additional_time_due_to_leak_l1855_185594


namespace largest_prime_factor_of_expression_l1855_185525

theorem largest_prime_factor_of_expression :
  ∃ (p : ℕ), Prime p ∧ p > 35 ∧ p > 2 ∧ p ∣ (18^4 + 2 * 18^2 + 1 - 17^4) ∧ ∀ q, Prime q ∧ q ∣ (18^4 + 2 * 18^2 + 1 - 17^4) → q ≤ p :=
by
  sorry

end largest_prime_factor_of_expression_l1855_185525


namespace intersection_points_l1855_185551

-- Define the line equation
def line (x : ℝ) : ℝ := 2 * x - 1

-- Problem statement to be proven
theorem intersection_points :
  (line 0.5 = 0) ∧ (line 0 = -1) :=
by 
  sorry

end intersection_points_l1855_185551


namespace tan_15_degrees_theta_range_valid_max_f_value_l1855_185574

-- Define the dot product condition
def dot_product_condition (AB BC : ℝ) (θ : ℝ) : Prop :=
  AB * BC * (Real.cos θ) = 6

-- Define the sine inequality condition
def sine_inequality_condition (AB BC : ℝ) (θ : ℝ) : Prop :=
  6 * (2 - Real.sqrt 3) ≤ AB * BC * (Real.sin θ) ∧ AB * BC * (Real.sin θ) ≤ 6 * Real.sqrt 3

-- Define the maximum value function
noncomputable def f (θ : ℝ) : ℝ :=
  (1 - Real.sqrt 2 * Real.cos (2 * θ - Real.pi / 4)) / (Real.sin θ)

-- Proof that tan 15 degrees is equal to 2 - sqrt(3)
theorem tan_15_degrees : Real.tan (Real.pi / 12) = 2 - Real.sqrt 3 := 
  by sorry

-- Proof for the range of θ
theorem theta_range_valid (AB BC : ℝ) (θ : ℝ) 
  (h1 : dot_product_condition AB BC θ)
  (h2 : sine_inequality_condition AB BC θ) : 
  (Real.pi / 12) ≤ θ ∧ θ ≤ (Real.pi / 3) := 
  by sorry

-- Proof for the maximum value of the function
theorem max_f_value (θ : ℝ) 
  (h : (Real.pi / 12) ≤ θ ∧ θ ≤ (Real.pi / 3)) : 
  f θ ≤ Real.sqrt 3 - 1 := 
  by sorry

end tan_15_degrees_theta_range_valid_max_f_value_l1855_185574


namespace not_enough_info_sweets_l1855_185577

theorem not_enough_info_sweets
    (S : ℕ)         -- Initial number of sweet cookies.
    (initial_salty : ℕ := 6)  -- Initial number of salty cookies given as 6.
    (eaten_sweets : ℕ := 20)   -- Number of sweet cookies Paco ate.
    (eaten_salty : ℕ := 34)    -- Number of salty cookies Paco ate.
    (diff_eaten : eaten_salty - eaten_sweets = 14) -- Paco ate 14 more salty cookies than sweet cookies.
    : (∃ S', S' = S) → False :=  -- Conclusion: Not enough information to determine initial number of sweet cookies S.
by
  sorry

end not_enough_info_sweets_l1855_185577


namespace num_ways_to_select_five_crayons_including_red_l1855_185539

noncomputable def num_ways_select_five_crayons (total_crayons : ℕ) (selected_crayons : ℕ) (fixed_red_crayon : ℕ) : ℕ :=
  Nat.choose (total_crayons - fixed_red_crayon) selected_crayons

theorem num_ways_to_select_five_crayons_including_red
  (total_crayons : ℕ) 
  (fixed_red_crayon : ℕ)
  (selected_crayons : ℕ)
  (h1 : total_crayons = 15)
  (h2 : fixed_red_crayon = 1)
  (h3 : selected_crayons = 4) : 
  num_ways_select_five_crayons total_crayons selected_crayons fixed_red_crayon = 1001 := by
  sorry

end num_ways_to_select_five_crayons_including_red_l1855_185539


namespace find_base_a_l1855_185580

theorem find_base_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : (if a < 1 then a + a^2 else a^2 + a) = 12) : a = 3 := 
sorry

end find_base_a_l1855_185580


namespace arithmetic_sequences_diff_l1855_185579

theorem arithmetic_sequences_diff
  (a : ℕ → ℤ)
  (b : ℕ → ℤ)
  (d_a d_b : ℤ)
  (ha : ∀ n, a n = 3 + n * d_a)
  (hb : ∀ n, b n = -3 + n * d_b)
  (h : a 19 - b 19 = 16) :
  a 10 - b 10 = 11 := by
    sorry

end arithmetic_sequences_diff_l1855_185579


namespace increase_by_percentage_l1855_185568

theorem increase_by_percentage (x : ℝ) (y : ℝ): x = 90 → y = 0.50 → x + x * y = 135 := 
by
  intro h1 h2
  sorry

end increase_by_percentage_l1855_185568


namespace charges_are_equal_l1855_185587

variable (a : ℝ)  -- original price for both travel agencies

def charge_A (a : ℝ) : ℝ := a + 2 * 0.7 * a
def charge_B (a : ℝ) : ℝ := 3 * 0.8 * a

theorem charges_are_equal : charge_A a = charge_B a :=
by
  sorry

end charges_are_equal_l1855_185587


namespace value_of_a_l1855_185596

theorem value_of_a (a : ℝ) (h : (1 : ℝ)^2 - 2 * (1 : ℝ) + a = 0) : a = 1 := 
by 
  sorry

end value_of_a_l1855_185596


namespace fraction_sum_simplified_l1855_185558

theorem fraction_sum_simplified (a b : ℕ) (h1 : 0.6125 = (a : ℝ) / b) (h2 : Nat.gcd a b = 1) : a + b = 129 :=
sorry

end fraction_sum_simplified_l1855_185558


namespace functional_relationship_inversely_proportional_l1855_185589

-- Definitions based on conditions
def table_data : List (ℝ × ℝ) := [(100, 1.00), (200, 0.50), (400, 0.25), (500, 0.20)]

-- The main conjecture to be proved
theorem functional_relationship_inversely_proportional (y x : ℝ) (h : (x, y) ∈ table_data) : y = 100 / x :=
sorry

end functional_relationship_inversely_proportional_l1855_185589


namespace probability_one_from_harold_and_one_from_marilyn_l1855_185544

-- Define the names and the number of letters in each name
def harold_name_length := 6
def marilyn_name_length := 7

-- Total cards
def total_cards := harold_name_length + marilyn_name_length

-- Probability of drawing one card from Harold's name and one from Marilyn's name
theorem probability_one_from_harold_and_one_from_marilyn :
    (harold_name_length : ℚ) / total_cards * marilyn_name_length / (total_cards - 1) +
    marilyn_name_length / total_cards * harold_name_length / (total_cards - 1) 
    = 7 / 13 := 
by
  sorry

end probability_one_from_harold_and_one_from_marilyn_l1855_185544


namespace compare_squares_l1855_185510

theorem compare_squares : -6 * Real.sqrt 5 < -5 * Real.sqrt 6 := sorry

end compare_squares_l1855_185510


namespace sum_first_10_terms_arith_seq_l1855_185569

theorem sum_first_10_terms_arith_seq (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : a 3 = 5)
  (h2 : a 7 = 13)
  (h3 : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2)
  (h4 : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) :
  S 10 = 100 :=
sorry

end sum_first_10_terms_arith_seq_l1855_185569


namespace product_of_consecutive_integers_between_sqrt_29_l1855_185513

-- Define that \(5 \lt \sqrt{29} \lt 6\)
lemma sqrt_29_bounds : 5 < Real.sqrt 29 ∧ Real.sqrt 29 < 6 :=
sorry

-- Main theorem statement
theorem product_of_consecutive_integers_between_sqrt_29 :
  (∃ (a b : ℤ), 5 < Real.sqrt 29 ∧ Real.sqrt 29 < 6 ∧ a = 5 ∧ b = 6 ∧ a * b = 30) := 
sorry

end product_of_consecutive_integers_between_sqrt_29_l1855_185513


namespace prob_no_distinct_roots_l1855_185517

-- Definition of integers a, b, c between -7 and 7
def valid_range (n : Int) : Prop := -7 ≤ n ∧ n ≤ 7

-- Definition of the discriminant condition for non-distinct real roots
def no_distinct_real_roots (a b c : Int) : Prop := b * b - 4 * a * c ≤ 0

-- Counting total triplets (a, b, c) with valid range
def total_triplets : Int := 15 * 15 * 15

-- Counting valid triplets with no distinct real roots
def valid_triplets : Int := 225 + (3150 / 2) -- 225 when a = 0 and estimation for a ≠ 0

theorem prob_no_distinct_roots : 
  let P := valid_triplets / total_triplets 
  P = (604 / 1125 : Rat) := 
by
  sorry

end prob_no_distinct_roots_l1855_185517


namespace exists_k_such_that_n_eq_k_2010_l1855_185595

theorem exists_k_such_that_n_eq_k_2010 (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n)
  (h : m * n ∣ m ^ 2010 + n ^ 2010 + n) : ∃ k : ℕ, 0 < k ∧ n = k ^ 2010 := by
  sorry

end exists_k_such_that_n_eq_k_2010_l1855_185595


namespace max_vertex_value_in_cube_l1855_185538

def transform_black (v : ℕ) (e1 e2 e3 : ℕ) : ℕ :=
  e1 + e2 + e3

def transform_white (v : ℕ) (d1 d2 d3 : ℕ) : ℕ :=
  d1 + d2 + d3

def max_value_after_transformation (initial_values : Fin 8 → ℕ) : ℕ :=
  -- Combination of transformations and iterations are derived here
  42648

theorem max_vertex_value_in_cube :
  ∀ (initial_values : Fin 8 → ℕ),
  (∀ i, 1 ≤ initial_values i ∧ initial_values i ≤ 8) →
  (∃ (final_value : ℕ), final_value = max_value_after_transformation initial_values) → final_value = 42648 :=
by {
  sorry
}

end max_vertex_value_in_cube_l1855_185538


namespace find_a_l1855_185537

noncomputable def f (x : ℝ) : ℝ := x^2 + 12
noncomputable def g (x : ℝ) : ℝ := x^2 - x - 4

theorem find_a (a : ℝ) (h_pos : a > 0) (h_fga : f (g a) = 12) : a = (1 + Real.sqrt 17) / 2 :=
by
  sorry

end find_a_l1855_185537


namespace largest_expression_l1855_185570

def P : ℕ := 3 * 2024 ^ 2025
def Q : ℕ := 2024 ^ 2025
def R : ℕ := 2023 * 2024 ^ 2024
def S : ℕ := 3 * 2024 ^ 2024
def T : ℕ := 2024 ^ 2024
def U : ℕ := 2024 ^ 2023

theorem largest_expression : 
  (P - Q) > (Q - R) ∧ 
  (P - Q) > (R - S) ∧ 
  (P - Q) > (S - T) ∧ 
  (P - Q) > (T - U) :=
by sorry

end largest_expression_l1855_185570


namespace polygon_sides_l1855_185530

theorem polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 140 * n) : n = 9 :=
sorry

end polygon_sides_l1855_185530


namespace a_minus_c_value_l1855_185532

theorem a_minus_c_value (a b c : ℝ) 
  (h1 : (a + b) / 2 = 110) 
  (h2 : (b + c) / 2 = 150) : 
  a - c = -80 := 
by 
  -- We provide the proof inline with sorry
  sorry

end a_minus_c_value_l1855_185532


namespace bisecting_chord_line_eqn_l1855_185592

theorem bisecting_chord_line_eqn :
  ∀ (x1 y1 x2 y2 : ℝ),
  y1 ^ 2 = 16 * x1 →
  y2 ^ 2 = 16 * x2 →
  (x1 + x2) / 2 = 2 →
  (y1 + y2) / 2 = 1 →
  ∃ (a b c : ℝ), a = 8 ∧ b = -1 ∧ c = -15 ∧
  ∀ (x y : ℝ), y = 8 * x - 15 → a * x + b * y + c = 0 :=
by 
  sorry

end bisecting_chord_line_eqn_l1855_185592


namespace unique_solution_l1855_185523

def my_operation (x y : ℝ) : ℝ := 5 * x - 2 * y + 3 * x * y

theorem unique_solution :
  ∃! y : ℝ, my_operation 4 y = 15 ∧ y = -1/2 :=
by 
  sorry

end unique_solution_l1855_185523


namespace can_pay_without_change_l1855_185550

theorem can_pay_without_change (n : ℕ) (h : n > 7) :
  ∃ (a b : ℕ), 3 * a + 5 * b = n :=
sorry

end can_pay_without_change_l1855_185550


namespace min_children_see_ear_l1855_185535

theorem min_children_see_ear (n : ℕ) : ∃ (k : ℕ), k = n + 2 :=
by
  sorry

end min_children_see_ear_l1855_185535


namespace intersection_A_B_subsets_C_l1855_185556

-- Definition of sets A and B
def A : Set ℤ := {-1, 1, 2}
def B : Set ℤ := {x | 0 ≤ x}

-- Definition of intersection C
def C : Set ℤ := A ∩ B

-- The proof statements
theorem intersection_A_B : C = {1, 2} := 
by sorry

theorem subsets_C : {s | s ⊆ C} = {∅, {1}, {2}, {1, 2}} := 
by sorry

end intersection_A_B_subsets_C_l1855_185556


namespace geometric_sum_formula_l1855_185590

noncomputable def geometric_sequence_sum (n : ℕ) : ℕ :=
  sorry

theorem geometric_sum_formula (a : ℕ → ℕ)
  (h_geom : ∀ n, a (n + 1) = 2 * a n)
  (h_a1_a2 : a 0 + a 1 = 3)
  (h_a1_a2_a3 : a 0 * a 1 * a 2 = 8) :
  geometric_sequence_sum n = 2^n - 1 :=
sorry

end geometric_sum_formula_l1855_185590


namespace german_students_count_l1855_185522

def total_students : ℕ := 45
def both_english_german : ℕ := 12
def only_english : ℕ := 23

theorem german_students_count :
  ∃ G : ℕ, G = 45 - (23 + 12) + 12 :=
sorry

end german_students_count_l1855_185522


namespace find_x_l1855_185591

theorem find_x (x : ℚ) : |x + 3| = |x - 4| → x = 1/2 := 
by 
-- Add appropriate content here
sorry

end find_x_l1855_185591


namespace average_visitors_per_day_l1855_185506

theorem average_visitors_per_day (avg_sunday_visitors : ℕ) (avg_otherday_visitors : ℕ) (days_in_month : ℕ)
  (starts_with_sunday : Bool) (num_sundays : ℕ) (num_otherdays : ℕ)
  (h1 : avg_sunday_visitors = 510)
  (h2 : avg_otherday_visitors = 240)
  (h3 : days_in_month = 30)
  (h4 : starts_with_sunday = true)
  (h5 : num_sundays = 5)
  (h6 : num_otherdays = 25) :
  (num_sundays * avg_sunday_visitors + num_otherdays * avg_otherday_visitors) / days_in_month = 285 :=
by 
  sorry

end average_visitors_per_day_l1855_185506


namespace fraction_eggs_used_for_cupcakes_l1855_185585

theorem fraction_eggs_used_for_cupcakes:
  ∀ (total_eggs crepes_fraction remaining_eggs after_cupcakes_eggs used_for_cupcakes_fraction: ℚ),
  total_eggs = 36 →
  crepes_fraction = 1 / 4 →
  after_cupcakes_eggs = 9 →
  used_for_cupcakes_fraction = 2 / 3 →
  (total_eggs * (1 - crepes_fraction) - after_cupcakes_eggs) / (total_eggs * (1 - crepes_fraction)) = used_for_cupcakes_fraction :=
by
  intros
  sorry

end fraction_eggs_used_for_cupcakes_l1855_185585


namespace value_is_200_l1855_185572

variable (x value : ℝ)
variable (h1 : 0.20 * x = value)
variable (h2 : 1.20 * x = 1200)

theorem value_is_200 : value = 200 :=
by
  sorry

end value_is_200_l1855_185572


namespace parabola_solution_l1855_185599

noncomputable def parabola_coefficients (a b c : ℝ) : Prop :=
  (6 : ℝ) = a * (5 : ℝ)^2 + b * (5 : ℝ) + c ∧
  0 = a * (3 : ℝ)^2 + b * (3 : ℝ) + c

theorem parabola_solution :
  ∃ (a b c : ℝ), parabola_coefficients a b c ∧ (a + b + c = 6) :=
by {
  -- definitions and constraints based on problem conditions
  sorry
}

end parabola_solution_l1855_185599


namespace parabola_tangents_min_area_l1855_185501

noncomputable def parabola_tangents (p : ℝ) : Prop :=
  ∃ (y₀ : ℝ), p > 0 ∧ (2 * Real.sqrt (y₀^2 + 2 * p) = 4)

theorem parabola_tangents_min_area (p : ℝ) : parabola_tangents 2 :=
by
  sorry

end parabola_tangents_min_area_l1855_185501


namespace line_through_points_l1855_185533

theorem line_through_points (x1 y1 x2 y2 : ℝ) :
  (3 * x1 - 4 * y1 - 2 = 0) →
  (3 * x2 - 4 * y2 - 2 = 0) →
  (∀ x y : ℝ, (x = x1) → (y = y1) ∨ (x = x2) → (y = y2) → 3 * x - 4 * y - 2 = 0) :=
by
  sorry

end line_through_points_l1855_185533


namespace no_valid_a_exists_l1855_185528

theorem no_valid_a_exists (a : ℕ) (n : ℕ) (h1 : a > 1) (b := a * (10^n + 1)) :
  ¬ (∃ a : ℕ, b % (a^2) = 0) :=
by {
  sorry -- The actual proof is not required as per instructions.
}

end no_valid_a_exists_l1855_185528


namespace tan_eq_one_over_three_l1855_185516

theorem tan_eq_one_over_three (x : ℝ) (h1 : x ∈ Set.Ioo 0 Real.pi)
  (h2 : Real.cos (2 * x - (Real.pi / 2)) = Real.sin x ^ 2) :
  Real.tan (x - Real.pi / 4) = 1 / 3 := by
  sorry

end tan_eq_one_over_three_l1855_185516


namespace georgia_makes_muffins_l1855_185500

/--
Georgia makes muffins and brings them to her students on the first day of every month.
Her muffin recipe only makes 6 muffins and she has 24 students. 
Prove that Georgia makes 36 batches of muffins in 9 months.
-/
theorem georgia_makes_muffins 
  (muffins_per_batch : ℕ)
  (students : ℕ)
  (months : ℕ) 
  (batches_per_day : ℕ) 
  (total_batches : ℕ)
  (h1 : muffins_per_batch = 6)
  (h2 : students = 24)
  (h3 : months = 9)
  (h4 : batches_per_day = students / muffins_per_batch) : 
  total_batches = months * batches_per_day :=
by
  -- The proof would go here
  sorry

end georgia_makes_muffins_l1855_185500


namespace apple_price_l1855_185557

theorem apple_price :
  ∀ (l q : ℝ), 
    (10 * l = 3.62) →
    (30 * l + 3 * q = 11.67) →
    (30 * l + 6 * q = 12.48) :=
by
  intros l q h₁ h₂
  -- The proof would go here with the steps, but for now we use sorry.
  sorry

end apple_price_l1855_185557


namespace eval_f_at_5_l1855_185578

def f (x : ℝ) : ℝ := 2 * x^7 - 9 * x^6 + 5 * x^5 - 49 * x^4 - 5 * x^3 + 2 * x^2 + x + 1

theorem eval_f_at_5 : f 5 = 56 := 
 by 
   sorry

end eval_f_at_5_l1855_185578


namespace total_vehicles_is_120_l1855_185598

def num_trucks : ℕ := 20
def num_tanks : ℕ := 5 * num_trucks
def total_vehicles : ℕ := num_tanks + num_trucks

theorem total_vehicles_is_120 : total_vehicles = 120 :=
by
  sorry

end total_vehicles_is_120_l1855_185598


namespace g_inv_undefined_at_one_l1855_185561

noncomputable def g (x : ℝ) : ℝ := (x - 5) / (x - 6)

theorem g_inv_undefined_at_one :
  ∀ (x : ℝ), (∃ (y : ℝ), g y = x ∧ ¬ ∃ (z : ℝ), g z = y ∧ g z = 1) ↔ x = 1 :=
by
  sorry

end g_inv_undefined_at_one_l1855_185561


namespace prove_original_sides_l1855_185571

def original_parallelogram_sides (a b : ℕ) : Prop :=
  ∃ k : ℕ, (a, b) = (k * 1, k * 2) ∨ (a, b) = (1, 5) ∨ (a, b) = (4, 5) ∨ (a, b) = (3, 7) ∨ (a, b) = (4, 7) ∨ (a, b) = (3, 8) ∨ (a, b) = (5, 8) ∨ (a, b) = (5, 7) ∨ (a, b) = (2, 7)

theorem prove_original_sides (a b : ℕ) : original_parallelogram_sides a b → (1, 2) = (1, 2) :=
by
  intro h
  sorry

end prove_original_sides_l1855_185571


namespace student_count_l1855_185562

theorem student_count (N : ℕ) (h1 : ∀ W : ℝ, W - 46 = 86 - 40) (h2 : (86 - 46) = 5 * N) : N = 8 :=
sorry

end student_count_l1855_185562


namespace extremum_value_l1855_185575

noncomputable def f (x a b : ℝ) : ℝ := x^3 + 3 * a * x^2 + b * x + a^2

theorem extremum_value (a b : ℝ) (h1 : (3 - 6 * a + b = 0)) (h2 : (-1 + 3 * a - b + a^2 = 0)) :
  a - b = -7 :=
by
  sorry

end extremum_value_l1855_185575


namespace intersection_correct_l1855_185584

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {0, 2, 3, 4}

theorem intersection_correct : A ∩ B = {2, 3} := sorry

end intersection_correct_l1855_185584


namespace households_using_both_brands_l1855_185524

def total : ℕ := 260
def neither : ℕ := 80
def onlyA : ℕ := 60
def onlyB (both : ℕ) : ℕ := 3 * both

theorem households_using_both_brands (both : ℕ) : 80 + 60 + both + onlyB both = 260 → both = 30 :=
by
  intro h
  sorry

end households_using_both_brands_l1855_185524


namespace percentage_decrease_is_25_percent_l1855_185503

noncomputable def percentage_decrease_in_revenue
  (R : ℝ)
  (projected_revenue : ℝ)
  (actual_revenue : ℝ) : ℝ :=
  ((R - actual_revenue) / R) * 100

-- Conditions
def last_year_revenue (R : ℝ) := R
def projected_revenue (R : ℝ) := 1.20 * R
def actual_revenue (R : ℝ) := 0.625 * (1.20 * R)

-- Proof statement
theorem percentage_decrease_is_25_percent (R : ℝ) :
  percentage_decrease_in_revenue R (projected_revenue R) (actual_revenue R) = 25 :=
by
  sorry

end percentage_decrease_is_25_percent_l1855_185503


namespace initial_liquid_A_amount_l1855_185515

noncomputable def initial_amount_of_A (x : ℚ) : ℚ :=
  3 * x

theorem initial_liquid_A_amount {x : ℚ} (h : (3 * x - 3) / (2 * x + 3) = 3 / 5) : initial_amount_of_A (8 / 3) = 8 := by
  sorry

end initial_liquid_A_amount_l1855_185515


namespace machine_production_time_l1855_185545

theorem machine_production_time (x : ℝ) 
  (h1 : 60 / x + 2 = 12) : 
  x = 6 :=
sorry

end machine_production_time_l1855_185545


namespace linear_function_no_third_quadrant_l1855_185536

theorem linear_function_no_third_quadrant (k b : ℝ) (h : ∀ x y : ℝ, x < 0 → y < 0 → y ≠ k * x + b) : k < 0 ∧ 0 ≤ b :=
sorry

end linear_function_no_third_quadrant_l1855_185536


namespace probability_of_selected_cubes_l1855_185548

-- Total number of unit cubes
def total_unit_cubes : ℕ := 125

-- Number of cubes with exactly two blue faces (from edges not corners)
def two_blue_faces : ℕ := 9

-- Number of unpainted unit cubes
def unpainted_cubes : ℕ := 51

-- Calculate total combinations of choosing 2 cubes out of 125
def total_combinations : ℕ := Nat.choose total_unit_cubes 2

-- Calculate favorable outcomes: one cube with 2 blue faces and one unpainted cube
def favorable_outcomes : ℕ := two_blue_faces * unpainted_cubes

-- Calculate probability
def probability : ℚ := favorable_outcomes / total_combinations

-- The theorem we want to prove
theorem probability_of_selected_cubes :
  probability = 3 / 50 :=
by
  -- Show that the probability indeed equals 3/50
  sorry

end probability_of_selected_cubes_l1855_185548


namespace gcd_polynomial_l1855_185547

-- Define conditions
variables (b : ℤ) (k : ℤ)

-- Assume b is an even multiple of 8753
def is_even_multiple_of_8753 (b : ℤ) : Prop := ∃ k : ℤ, b = 2 * 8753 * k

-- Statement to be proven
theorem gcd_polynomial (b : ℤ) (h : is_even_multiple_of_8753 b) :
  Int.gcd (4 * b^2 + 27 * b + 100) (3 * b + 7) = 2 :=
by sorry

end gcd_polynomial_l1855_185547


namespace fraction_identity_l1855_185573

theorem fraction_identity (a b c : ℕ) (h : (a : ℚ) / (36 - a) + (b : ℚ) / (48 - b) + (c : ℚ) / (72 - c) = 9) : 
  4 / (36 - a) + 6 / (48 - b) + 9 / (72 - c) = 13 / 3 := 
by 
  sorry

end fraction_identity_l1855_185573


namespace similar_triangles_XY_length_l1855_185526

-- Defining necessary variables.
variables (PQ QR YZ XY : ℝ) (area_XYZ : ℝ)

-- Given conditions to be used in the proof.
def condition1 : PQ = 8 := sorry
def condition2 : QR = 16 := sorry
def condition3 : YZ = 24 := sorry
def condition4 : area_XYZ = 144 := sorry

-- Statement of the mathematical proof problem to show XY = 12
theorem similar_triangles_XY_length :
  PQ = 8 → QR = 16 → YZ = 24 → area_XYZ = 144 → XY = 12 :=
by
  intros hPQ hQR hYZ hArea
  sorry

end similar_triangles_XY_length_l1855_185526


namespace tan_sum_angle_identity_l1855_185566

theorem tan_sum_angle_identity
  (α β : ℝ)
  (h1 : Real.tan (α + 2 * β) = 2)
  (h2 : Real.tan β = -3) :
  Real.tan (α + β) = -1 := sorry

end tan_sum_angle_identity_l1855_185566


namespace find_share_of_b_l1855_185520

variable (a b c : ℕ)
axiom h1 : a = 3 * b
axiom h2 : b = c + 25
axiom h3 : a + b + c = 645

theorem find_share_of_b : b = 134 := by
  sorry

end find_share_of_b_l1855_185520


namespace line_l_equation_symmetrical_line_equation_l1855_185582

theorem line_l_equation (x y : ℝ) (h₁ : 3 * x + 4 * y - 2 = 0) (h₂ : 2 * x + y + 2 = 0) :
  2 * x + y + 2 = 0 :=
sorry

theorem symmetrical_line_equation (x y : ℝ) :
  (2 * x + y + 2 = 0) → (2 * x + y - 2 = 0) :=
sorry

end line_l_equation_symmetrical_line_equation_l1855_185582


namespace sum_fifth_powers_divisible_by_15_l1855_185559

theorem sum_fifth_powers_divisible_by_15
  (A B C D E : ℤ) 
  (h : A + B + C + D + E = 0) : 
  (A^5 + B^5 + C^5 + D^5 + E^5) % 15 = 0 := 
by 
  sorry

end sum_fifth_powers_divisible_by_15_l1855_185559


namespace ratio_area_octagons_correct_l1855_185564

noncomputable def ratio_area_octagons (r : ℝ) : ℝ :=
  let s := 2 * r * Real.sin (Real.pi / 8)   -- side length of inscribed octagon
  let S := 2 * r * Real.cos (Real.pi / 8)   -- side length of circumscribed octagon
  let ratio_side_lengths := S / s
  let ratio_areas := (ratio_side_lengths)^2
  ratio_areas

theorem ratio_area_octagons_correct (r : ℝ) :
  ratio_area_octagons r = 6 + 4 * Real.sqrt 2 :=
by
  sorry

end ratio_area_octagons_correct_l1855_185564


namespace triangle_A_and_Area_l1855_185518

theorem triangle_A_and_Area :
  ∀ (a b c A B C : ℝ), 
  (b - (1 / 2) * c = a * Real.cos C) 
  → (4 * (b + c) = 3 * b * c) 
  → (a = 2 * Real.sqrt 3)
  → (A = 60) ∧ (1/2 * b * c * Real.sin A = 2 * Real.sqrt 3) :=
by
  intros a b c A B C h1 h2 h3
  sorry

end triangle_A_and_Area_l1855_185518


namespace locus_is_hyperbola_l1855_185534

theorem locus_is_hyperbola
  (x y a θ₁ θ₂ c : ℝ)
  (h1 : (x - a) * Real.cos θ₁ + y * Real.sin θ₁ = a)
  (h2 : (x - a) * Real.cos θ₂ + y * Real.sin θ₂ = a)
  (h3 : Real.tan (θ₁ / 2) - Real.tan (θ₂ / 2) = 2 * c)
  (hc : c > 1) 
  : ∃ k l m : ℝ, k * (x ^ 2) + l * x * y + m * (y ^ 2) = 1 := sorry

end locus_is_hyperbola_l1855_185534


namespace correct_option_given_inequality_l1855_185508

theorem correct_option_given_inequality (a b : ℝ) (h : a > b) : -2 * a < -2 * b :=
sorry

end correct_option_given_inequality_l1855_185508


namespace operation_star_correct_l1855_185511

def op_table (i j : ℕ) : ℕ :=
  if i = 1 then
    if j = 1 then 4 else if j = 2 then 1 else if j = 3 then 2 else if j = 4 then 3 else 0
  else if i = 2 then
    if j = 1 then 1 else if j = 2 then 3 else if j = 3 then 4 else if j = 4 then 2 else 0
  else if i = 3 then
    if j = 1 then 2 else if j = 2 then 4 else if j = 3 then 1 else if j = 4 then 3 else 0
  else if i = 4 then
    if j = 1 then 3 else if j = 2 then 2 else if j = 3 then 3 else if j = 4 then 4 else 0
  else 0

theorem operation_star_correct : op_table (op_table 3 1) (op_table 4 2) = 3 :=
  by sorry

end operation_star_correct_l1855_185511


namespace quadratic_inequality_real_solution_l1855_185527

theorem quadratic_inequality_real_solution (a : ℝ) :
  (∃ x : ℝ, 2*x^2 + (a-1)*x + 1/2 ≤ 0) ↔ (a ≤ -1 ∨ 3 ≤ a) := 
sorry

end quadratic_inequality_real_solution_l1855_185527


namespace quadratic_distinct_real_roots_iff_l1855_185593

theorem quadratic_distinct_real_roots_iff (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (∀ (z : ℝ), z^2 - 2 * (m - 2) * z + m^2 = (z - x) * (z - y))) ↔ m < 1 :=
by
  sorry

end quadratic_distinct_real_roots_iff_l1855_185593


namespace range_of_k_l1855_185588

variables (k : ℝ)

def vector_a (k : ℝ) : ℝ × ℝ := (-k, 4)
def vector_b (k : ℝ) : ℝ × ℝ := (k, k + 3)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem range_of_k (h : 0 < dot_product (vector_a k) (vector_b k)) : 
  -2 < k ∧ k < 0 ∨ 0 < k ∧ k < 6 :=
sorry

end range_of_k_l1855_185588


namespace range_of_a_in_quadratic_l1855_185529

theorem range_of_a_in_quadratic :
  (∃ x1 x2 : ℝ, x1 < -1 ∧ x2 > 1 ∧ x1 ≠ x2 ∧ x1^2 + a * x1 - 2 = 0 ∧ x2^2 + a * x2 - 2 = 0) → -1 < a ∧ a < 1 :=
by
  sorry

end range_of_a_in_quadratic_l1855_185529


namespace max_ab_correct_l1855_185576

noncomputable def max_ab (k : ℝ) (a b: ℝ) : ℝ :=
if k = -3 then 9 else sorry

theorem max_ab_correct (k : ℝ) (a b: ℝ)
  (h1 : (-3 ≤ k ∧ k ≤ 1))
  (h2 : a + b = 2 * k)
  (h3 : a^2 + b^2 = k^2 - 2 * k + 3) :
  max_ab k a b = 9 :=
sorry

end max_ab_correct_l1855_185576


namespace least_subset_gcd_l1855_185514

variable (S : Set ℕ) (f : ℕ → ℤ)
variable (a : ℕ → ℕ)
variable (k : ℕ)

def conditions (S : Set ℕ) (f : ℕ → ℤ) : Prop :=
  ∃ (a : ℕ → ℕ), 
  (∀ i j, i ≠ j → a i < a j) ∧ 
  (S = {i | ∃ n, i = a n ∧ n < 2004}) ∧ 
  (∀ i, f (a i) < 2003) ∧ 
  (∀ i j, f (a i) = f (a j))

theorem least_subset_gcd (h : conditions S f) : k = 1003 :=
  sorry

end least_subset_gcd_l1855_185514


namespace number_of_divisors_125n5_l1855_185543

theorem number_of_divisors_125n5 (n : ℕ) (hn : n > 0)
  (h150 : ∀ m : ℕ, m = 150 * n ^ 4 → (∃ d : ℕ, d * (d + 1) = 150)) :
  ∃ d : ℕ, d = 125 * n ^ 5 ∧ ((13 + 1) * (5 + 1) * (5 + 1) = 504) :=
by
  sorry

end number_of_divisors_125n5_l1855_185543


namespace int_product_negative_max_negatives_l1855_185586

theorem int_product_negative_max_negatives (n : ℤ) (hn : n ≤ 9) (hp : n % 2 = 1) :
  ∃ m : ℤ, n + m = m ∧ m ≥ 0 :=
by
  use 9
  sorry

end int_product_negative_max_negatives_l1855_185586


namespace matrix_power_sub_l1855_185512

section 
variable (A : Matrix (Fin 2) (Fin 2) ℝ)
variable (hA : A = ![![2, 3], ![0, 1]])

theorem matrix_power_sub (A : Matrix (Fin 2) (Fin 2) ℝ)
  (h: A = ![![2, 3], ![0, 1]]) :
  A ^ 20 - 2 * A ^ 19 = ![![0, 3], ![0, -1]] :=
by
  sorry
end

end matrix_power_sub_l1855_185512


namespace power_function_value_l1855_185502

theorem power_function_value (α : ℝ) (h₁ : (2 : ℝ) ^ α = (Real.sqrt 2) / 2) : (9 : ℝ) ^ α = 1 / 3 := 
by
  sorry

end power_function_value_l1855_185502


namespace room_analysis_l1855_185565

-- First person's statements
def statement₁ (n: ℕ) (liars: ℕ) :=
  n ≤ 3 ∧ liars = n

-- Second person's statements
def statement₂ (n: ℕ) (liars: ℕ) :=
  n ≤ 4 ∧ liars < n

-- Third person's statements
def statement₃ (n: ℕ) (liars: ℕ) :=
  n = 5 ∧ liars = 3

theorem room_analysis (n liars : ℕ) :
  (¬ statement₁ n liars) ∧ statement₂ n liars ∧ ¬ statement₃ n liars → (n = 4 ∧ liars = 2) :=
by
  sorry

end room_analysis_l1855_185565


namespace coin_toss_probability_l1855_185521

theorem coin_toss_probability :
  (∀ n : ℕ, 0 ≤ n → n ≤ 10 → (∀ m : ℕ, 0 ≤ m → m = 10 → 
  (∀ k : ℕ, k = 9 → 
  (∀ i : ℕ, 0 ≤ i → i = 10 → ∃ p : ℝ, p = 1/2 → 
  (∃ q : ℝ, q = 1/2 → q = p))))) := 
sorry

end coin_toss_probability_l1855_185521


namespace daughterAgeThreeYearsFromNow_l1855_185531

-- Definitions of constants and conditions
def motherAgeNow := 41
def motherAgeFiveYearsAgo := motherAgeNow - 5
def daughterAgeFiveYearsAgo := motherAgeFiveYearsAgo / 2
def daughterAgeNow := daughterAgeFiveYearsAgo + 5
def daughterAgeInThreeYears := daughterAgeNow + 3

-- Theorem to prove the daughter's age in 3 years given conditions
theorem daughterAgeThreeYearsFromNow :
  motherAgeNow = 41 →
  motherAgeFiveYearsAgo = 2 * daughterAgeFiveYearsAgo →
  daughterAgeInThreeYears = 26 :=
by
  intros h1 h2
  -- Original Lean would have the proof steps here
  sorry

end daughterAgeThreeYearsFromNow_l1855_185531


namespace divisible_by_12_for_all_integral_n_l1855_185541

theorem divisible_by_12_for_all_integral_n (n : ℤ) : 12 ∣ (2 * n ^ 3 - 2 * n) :=
sorry

end divisible_by_12_for_all_integral_n_l1855_185541


namespace percentage_of_silver_in_final_solution_l1855_185546

noncomputable section -- because we deal with real numbers and division

variable (volume_4pct : ℝ) (percentage_4pct : ℝ)
variable (volume_10pct : ℝ) (percentage_10pct : ℝ)

def final_percentage_silver (v4 : ℝ) (p4 : ℝ) (v10 : ℝ) (p10 : ℝ) : ℝ :=
  let total_silver := v4 * p4 + v10 * p10
  let total_volume := v4 + v10
  (total_silver / total_volume) * 100

theorem percentage_of_silver_in_final_solution :
  final_percentage_silver 5 0.04 2.5 0.10 = 6 := by
  sorry

end percentage_of_silver_in_final_solution_l1855_185546


namespace limit_C_of_f_is_2_l1855_185555

variable {f : ℝ → ℝ}
variable {x₀ : ℝ}
variable {f' : ℝ}

noncomputable def differentiable_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ f' : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ h, abs h < δ → abs (f (x + h) - f x - f' * h) / abs (h) < ε

axiom hf_differentiable : differentiable_at f x₀
axiom f'_at_x₀ : f' = 1

theorem limit_C_of_f_is_2 
  (hf_differentiable : differentiable_at f x₀) 
  (h_f'_at_x₀ : f' = 1) : 
  (∀ ε > 0, ∃ δ > 0, ∀ Δx, abs Δx < δ → abs ((f (x₀ + 2 * Δx) - f x₀) / Δx - 2) < ε) :=
sorry

end limit_C_of_f_is_2_l1855_185555


namespace gcd_divisors_remainders_l1855_185567

theorem gcd_divisors_remainders (d : ℕ) :
  (1657 % d = 6) ∧ (2037 % d = 5) → d = 127 :=
by
  sorry

end gcd_divisors_remainders_l1855_185567


namespace fraction_of_smaller_part_l1855_185560

theorem fraction_of_smaller_part (A B : ℕ) (x : ℚ) (h1 : A + B = 66) (h2 : A = 50) (h3 : 0.40 * A = x * B + 10) : x = 5 / 8 :=
by
  sorry

end fraction_of_smaller_part_l1855_185560


namespace no_solution_lines_parallel_l1855_185583

theorem no_solution_lines_parallel (m : ℝ) :
  (∀ t s : ℝ, (1 + 5 * t = 4 - 2 * s) ∧ (-3 + 2 * t = 1 + m * s) → false) ↔ m = -4 / 5 :=
by
  sorry

end no_solution_lines_parallel_l1855_185583


namespace average_speed_l1855_185553

theorem average_speed (d1 d2 t1 t2 : ℝ) (h1 : d1 = 50) (h2 : d2 = 20) (h3 : t1 = 50 / 20) (h4 : t2 = 20 / 40) :
  ((d1 + d2) / (t1 + t2)) = 23.33 := 
  sorry

end average_speed_l1855_185553


namespace apples_per_pie_l1855_185581

-- Definitions of given conditions
def total_apples : ℕ := 75
def handed_out_apples : ℕ := 19
def remaining_apples : ℕ := total_apples - handed_out_apples
def pies_made : ℕ := 7

-- Statement of the problem to be proved
theorem apples_per_pie : remaining_apples / pies_made = 8 := by
  sorry

end apples_per_pie_l1855_185581


namespace andrew_brian_ratio_l1855_185509

-- Definitions based on conditions extracted from the problem
variables (A S B : ℕ)

-- Conditions
def steven_shirts : Prop := S = 72
def brian_shirts : Prop := B = 3
def steven_andrew_relation : Prop := S = 4 * A

-- The goal is to prove the ratio of Andrew's shirts to Brian's shirts is 6
theorem andrew_brian_ratio (A S B : ℕ) 
  (h1 : steven_shirts S) 
  (h2 : brian_shirts B)
  (h3 : steven_andrew_relation A S) :
  A / B = 6 := by
  sorry

end andrew_brian_ratio_l1855_185509


namespace find_inverse_value_l1855_185563

noncomputable def f (x : ℝ) : ℝ := sorry -- f(x) function definition goes here

theorem find_inverse_value :
  (∀ x : ℝ, f (x - 1) = f (x + 3)) →
  (∀ x : ℝ, f (-x) = f x) →
  (∀ x : ℝ, 4 ≤ x ∧ x ≤ 6 → f x = 2^x + 1) →
  f⁻¹ 19 = 3 - 2 * (Real.log 3 / Real.log 2) :=
by
  intros h1 h2 h3
  -- Proof goes here
  sorry

end find_inverse_value_l1855_185563


namespace final_notebooks_l1855_185504

def initial_notebooks : ℕ := 10
def ordered_notebooks : ℕ := 6
def lost_notebooks : ℕ := 2

theorem final_notebooks : initial_notebooks + ordered_notebooks - lost_notebooks = 14 :=
by
  sorry

end final_notebooks_l1855_185504


namespace gymnastics_performance_participation_l1855_185540

def total_people_in_gym_performance (grades : ℕ) (classes_per_grade : ℕ) (students_per_class : ℕ) : ℕ :=
  grades * classes_per_grade * students_per_class

theorem gymnastics_performance_participation :
  total_people_in_gym_performance 3 4 15 = 180 :=
by
  -- This is where the proof would go
  sorry

end gymnastics_performance_participation_l1855_185540


namespace proof_problem_l1855_185549

theorem proof_problem (a b c : ℤ) (h1 : a > 2) (h2 : b < 10) (h3 : c ≥ 0) (h4 : 32 = a + 2 * b + 3 * c) : 
  a = 4 ∧ b = 8 ∧ c = 4 :=
by
  sorry

end proof_problem_l1855_185549


namespace min_colors_needed_l1855_185554

def cell := (ℤ × ℤ)

def rook_distance (c1 c2 : cell) : ℤ :=
  max (abs (c1.1 - c2.1)) (abs (c1.2 - c2.2))

def color (c : cell) : ℤ :=
  (c.1 + c.2) % 4

theorem min_colors_needed : 4 = 4 :=
sorry

end min_colors_needed_l1855_185554


namespace probability_queen_then_spade_l1855_185552

-- Define the size of the deck and the quantities for specific cards
def deck_size : ℕ := 52
def num_queens : ℕ := 4
def num_spades : ℕ := 13

-- Define the probability calculation problem
theorem probability_queen_then_spade :
  (num_queens / deck_size : ℚ) * ((num_spades - 1) / (deck_size - 1) : ℚ) + ((num_queens - 1) / deck_size : ℚ) * (num_spades / (deck_size - 1) : ℚ) = 1 / deck_size :=
by sorry

end probability_queen_then_spade_l1855_185552


namespace total_property_value_l1855_185542

-- Define the given conditions
def price_per_sq_ft_condo := 98
def price_per_sq_ft_barn := 84
def price_per_sq_ft_detached := 102
def price_per_sq_ft_garage := 60
def sq_ft_condo := 2400
def sq_ft_barn := 1200
def sq_ft_detached := 3500
def sq_ft_garage := 480

-- Main statement to prove the total value of the property
theorem total_property_value :
  (price_per_sq_ft_condo * sq_ft_condo + 
   price_per_sq_ft_barn * sq_ft_barn + 
   price_per_sq_ft_detached * sq_ft_detached + 
   price_per_sq_ft_garage * sq_ft_garage = 721800) :=
by
  -- Placeholder for the actual proof
  sorry

end total_property_value_l1855_185542


namespace max_ben_cupcakes_l1855_185519

theorem max_ben_cupcakes (total_cupcakes : ℕ) (ben_cupcakes charles_cupcakes diana_cupcakes : ℕ)
    (h1 : total_cupcakes = 30)
    (h2 : diana_cupcakes = 2 * ben_cupcakes)
    (h3 : charles_cupcakes = diana_cupcakes)
    (h4 : total_cupcakes = ben_cupcakes + charles_cupcakes + diana_cupcakes) :
    ben_cupcakes = 6 :=
by
  -- Proof steps would go here
  sorry

end max_ben_cupcakes_l1855_185519


namespace product_of_faces_and_vertices_of_cube_l1855_185507

def number_of_faces := 6
def number_of_vertices := 8

theorem product_of_faces_and_vertices_of_cube : number_of_faces * number_of_vertices = 48 := 
by 
  sorry

end product_of_faces_and_vertices_of_cube_l1855_185507


namespace extreme_value_at_1_l1855_185505

theorem extreme_value_at_1 (a b : ℝ) (h1 : (deriv (λ x => x^3 + a * x^2 + b * x + a^2) 1 = 0))
(h2 : (1 + a + b + a^2 = 10)) : a + b = -7 := by
  sorry

end extreme_value_at_1_l1855_185505


namespace largest_integral_value_l1855_185597

theorem largest_integral_value (y : ℤ) (h1 : 0 < y) (h2 : (1 : ℚ)/4 < y / 7) (h3 : y / 7 < 7 / 11) : y = 4 :=
sorry

end largest_integral_value_l1855_185597
