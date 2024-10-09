import Mathlib

namespace sum_of_b_values_l27_2745

theorem sum_of_b_values (b1 b2 : ℝ) : 
  (∀ x : ℝ, (9 * x^2 + (b1 + 15) * x + 16 = 0 ∨ 9 * x^2 + (b2 + 15) * x + 16 = 0) ∧ 
           (b1 + 15)^2 - 4 * 9 * 16 = 0 ∧ 
           (b2 + 15)^2 - 4 * 9 * 16 = 0) → 
  (b1 + b2) = -30 := 
sorry

end sum_of_b_values_l27_2745


namespace room_breadth_is_five_l27_2754

theorem room_breadth_is_five 
  (length : ℝ)
  (height : ℝ)
  (bricks_per_square_meter : ℝ)
  (total_bricks : ℝ)
  (H_length : length = 4)
  (H_height : height = 2)
  (H_bricks_per_square_meter : bricks_per_square_meter = 17)
  (H_total_bricks : total_bricks = 340) 
  : ∃ (breadth : ℝ), breadth = 5 :=
by
  -- we leave the proof as sorry for now
  sorry

end room_breadth_is_five_l27_2754


namespace number_of_solid_figures_is_4_l27_2751

def is_solid_figure (shape : String) : Bool :=
  shape = "cone" ∨ shape = "cuboid" ∨ shape = "sphere" ∨ shape = "triangular prism"

def shapes : List String :=
  ["circle", "square", "cone", "cuboid", "line segment", "sphere", "triangular prism", "right-angled triangle"]

def number_of_solid_figures : Nat :=
  (shapes.filter is_solid_figure).length

theorem number_of_solid_figures_is_4 : number_of_solid_figures = 4 :=
  by sorry

end number_of_solid_figures_is_4_l27_2751


namespace profit_percentage_is_correct_l27_2778

noncomputable def shopkeeper_profit_percentage : ℚ :=
  let cost_A : ℚ := 12 * (15/16)
  let cost_B : ℚ := 18 * (47/50)
  let profit_A : ℚ := 12 - cost_A
  let profit_B : ℚ := 18 - cost_B
  let total_profit : ℚ := profit_A + profit_B
  let total_cost : ℚ := cost_A + cost_B
  (total_profit / total_cost) * 100

theorem profit_percentage_is_correct :
  shopkeeper_profit_percentage = 6.5 := by
  sorry

end profit_percentage_is_correct_l27_2778


namespace speed_of_current_l27_2793

theorem speed_of_current (v_b v_c v_d : ℝ) (hd : v_d = 15) 
  (hvd1 : v_b + v_c = v_d) (hvd2 : v_b - v_c = 12) :
  v_c = 1.5 :=
by sorry

end speed_of_current_l27_2793


namespace quadratic_ratio_l27_2772

theorem quadratic_ratio (b c : ℤ) (h : ∀ x : ℤ, x^2 + 1400 * x + 1400 = (x + b) ^ 2 + c) : c / b = -698 :=
sorry

end quadratic_ratio_l27_2772


namespace taxi_fare_distance_l27_2714

theorem taxi_fare_distance (x : ℕ) (h₁ : 8 + 2 * (x - 3) = 20) : x = 9 :=
by {
  sorry
}

end taxi_fare_distance_l27_2714


namespace chase_travel_time_l27_2710

-- Definitions of speeds
def chase_speed (C : ℝ) := C
def cameron_speed (C : ℝ) := 2 * C
def danielle_speed (C : ℝ) := 6 * (cameron_speed C)

-- Time taken by Danielle to cover distance
def time_taken_by_danielle (C : ℝ) := 30  
def distance_travelled (C : ℝ) := (time_taken_by_danielle C) * (danielle_speed C)  -- 180C

-- Speeds on specific stretches
def cameron_bike_speed (C : ℝ) := 0.75 * (cameron_speed C)
def chase_scooter_speed (C : ℝ) := 1.25 * (chase_speed C)

-- Prove the time Chase takes to travel the same distance D
theorem chase_travel_time (C : ℝ) : 
  (distance_travelled C) / (chase_speed C) = 180 := sorry

end chase_travel_time_l27_2710


namespace intersect_empty_range_of_a_union_subsets_range_of_a_l27_2736

variable {x a : ℝ}

def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | (x - 6) * (x + 2) > 0}

theorem intersect_empty_range_of_a (h : A a ∩ B = ∅) : -2 ≤ a ∧ a ≤ 3 :=
by
  sorry

theorem union_subsets_range_of_a (h : A a ∪ B = B) : a < -5 ∨ a > 6 :=
by
  sorry

end intersect_empty_range_of_a_union_subsets_range_of_a_l27_2736


namespace circle_passing_through_points_l27_2774

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l27_2774


namespace proof_problem_l27_2731

def h (x : ℝ) : ℝ := x^2 - 3 * x + 7
def k (x : ℝ) : ℝ := 2 * x + 4

theorem proof_problem : h (k 3) - k (h 3) = 59 := by
  sorry

end proof_problem_l27_2731


namespace sad_children_count_l27_2796

theorem sad_children_count (total_children happy_children neither_happy_nor_sad children sad_children : ℕ)
  (h_total : total_children = 60)
  (h_happy : happy_children = 30)
  (h_neither : neither_happy_nor_sad = 20)
  (boys girls happy_boys sad_girls neither_boys : ℕ)
  (h_boys : boys = 17)
  (h_girls : girls = 43)
  (h_happy_boys : happy_boys = 6)
  (h_sad_girls : sad_girls = 4)
  (h_neither_boys : neither_boys = 5) :
  sad_children = total_children - happy_children - neither_happy_nor_sad :=
by sorry

end sad_children_count_l27_2796


namespace zoo_ticket_problem_l27_2798

def students_6A (total_cost_6A : ℕ) (saved_tickets_6A : ℕ) (ticket_price : ℕ) : ℕ :=
  let paid_tickets := (total_cost_6A / ticket_price)
  (paid_tickets + saved_tickets_6A)

def students_6B (total_cost_6B : ℕ) (total_students_6A : ℕ) (ticket_price : ℕ) : ℕ :=
  let paid_tickets := (total_cost_6B / ticket_price)
  let total_students := paid_tickets + (paid_tickets / 4)
  (total_students - total_students_6A)

theorem zoo_ticket_problem :
  (students_6A 1995 4 105 = 23) ∧
  (students_6B 4410 23 105 = 29) :=
by {
  -- The proof will follow the steps to confirm the calculations and final result
  sorry
}

end zoo_ticket_problem_l27_2798


namespace probability_at_least_one_hit_l27_2725

variable (P₁ P₂ : ℝ)

theorem probability_at_least_one_hit (h₁ : 0 ≤ P₁ ∧ P₁ ≤ 1) (h₂ : 0 ≤ P₂ ∧ P₂ ≤ 1) :
  1 - (1 - P₁) * (1 - P₂) = P₁ + P₂ - P₁ * P₂ :=
by
  sorry

end probability_at_least_one_hit_l27_2725


namespace average_number_of_problems_per_day_l27_2749

theorem average_number_of_problems_per_day (P D : ℕ) (hP : P = 161) (hD : D = 7) : (P / D) = 23 :=
  by sorry

end average_number_of_problems_per_day_l27_2749


namespace mr_willam_land_percentage_over_taxable_land_l27_2790

def total_tax_collected : ℝ := 3840
def tax_paid_by_mr_willam : ℝ := 480
def farm_tax_percentage : ℝ := 0.45

theorem mr_willam_land_percentage_over_taxable_land :
  (tax_paid_by_mr_willam / total_tax_collected) * 100 = 5.625 :=
by
  sorry

end mr_willam_land_percentage_over_taxable_land_l27_2790


namespace simplify_polynomial_l27_2715

variable (x : ℝ)

theorem simplify_polynomial : 
  (2 * x^4 + 3 * x^3 - 5 * x + 6) + (-6 * x^4 - 2 * x^3 + 3 * x^2 + 5 * x - 4) = 
  -4 * x^4 + x^3 + 3 * x^2 + 2 :=
by
  sorry

end simplify_polynomial_l27_2715


namespace factorization_correct_l27_2711

noncomputable def factorize_poly (m n : ℕ) : ℕ := 2 * m * n ^ 2 - 12 * m * n + 18 * m

theorem factorization_correct (m n : ℕ) :
  factorize_poly m n = 2 * m * (n - 3) ^ 2 :=
by
  sorry

end factorization_correct_l27_2711


namespace m_coins_can_collect_k_rubles_l27_2700

theorem m_coins_can_collect_k_rubles
  (a1 a2 a3 a4 a5 a6 a7 m k : ℕ)
  (h1 : a1 + 2 * a2 + 5 * a3 + 10 * a4 + 20 * a5 + 50 * a6 + 100 * a7 = m)
  (h2 : a1 + a2 + a3 + a4 + a5 + a6 + a7 = k) :
  ∃ (b1 b2 b3 b4 b5 b6 b7 : ℕ), 
    100 * (b1 + 2 * b2 + 5 * b3 + 10 * b4 + 20 * b5 + 50 * b6 + 100 * b7) = 100 * k ∧ 
    b1 + b2 + b3 + b4 + b5 + b6 + b7 = m := 
sorry

end m_coins_can_collect_k_rubles_l27_2700


namespace correct_growth_rate_equation_l27_2757

-- Define the conditions
def packages_first_day := 200
def packages_third_day := 242

-- Define the average daily growth rate
variable (x : ℝ)

-- State the theorem to prove
theorem correct_growth_rate_equation :
  packages_first_day * (1 + x)^2 = packages_third_day :=
by
  sorry

end correct_growth_rate_equation_l27_2757


namespace minimum_value_of_a_l27_2782

noncomputable def inequality_valid_for_all_x (a : ℝ) : Prop :=
  ∀ (x : ℝ), 1 < x → x + a * Real.log x - x^a + 1 / Real.exp x ≥ 0

theorem minimum_value_of_a : ∃ a, inequality_valid_for_all_x a ∧ a = -Real.exp 1 := sorry

end minimum_value_of_a_l27_2782


namespace problem_statement_l27_2713

def atOp (a b : ℝ) := a * b ^ (1 / 2)

theorem problem_statement : atOp ((2 * 3) ^ 2) ((3 * 5) ^ 2 / 9) = 180 := by
  sorry

end problem_statement_l27_2713


namespace original_square_perimeter_l27_2760

-- Define the problem statement
theorem original_square_perimeter (P_perimeter : ℕ) (hP : P_perimeter = 56) : 
  ∃ sq_perimeter : ℕ, sq_perimeter = 32 := 
by 
  sorry

end original_square_perimeter_l27_2760


namespace bottles_left_on_shelf_l27_2722

variable (initial_bottles : ℕ)
variable (bottles_jason : ℕ)
variable (bottles_harry : ℕ)

theorem bottles_left_on_shelf (h₁ : initial_bottles = 35) (h₂ : bottles_jason = 5) (h₃ : bottles_harry = bottles_jason + 6) :
  initial_bottles - (bottles_jason + bottles_harry) = 24 := by
  sorry

end bottles_left_on_shelf_l27_2722


namespace first_student_time_l27_2761

-- Define the conditions
def num_students := 4
def avg_last_three := 35
def avg_all := 30
def total_time_all := num_students * avg_all
def total_time_last_three := (num_students - 1) * avg_last_three

-- State the theorem
theorem first_student_time : (total_time_all - total_time_last_three) = 15 :=
by
  -- Proof is skipped
  sorry

end first_student_time_l27_2761


namespace apples_in_basket_l27_2742

noncomputable def total_apples (good_cond: ℕ) (good_ratio: ℝ) := (good_cond : ℝ) / good_ratio

theorem apples_in_basket : total_apples 66 0.88 = 75 :=
by
  sorry

end apples_in_basket_l27_2742


namespace total_glass_area_l27_2750

theorem total_glass_area 
  (len₁ len₂ len₃ wid₁ wid₂ wid₃ : ℕ)
  (h₁ : len₁ = 30) (h₂ : wid₁ = 12)
  (h₃ : len₂ = 30) (h₄ : wid₂ = 12)
  (h₅ : len₃ = 20) (h₆ : wid₃ = 12) :
  (len₁ * wid₁ + len₂ * wid₂ + len₃ * wid₃) = 960 := 
by
  sorry

end total_glass_area_l27_2750


namespace greatest_sum_of_consecutive_integers_product_less_500_l27_2726

theorem greatest_sum_of_consecutive_integers_product_less_500 :
  ∃ n : ℤ, n * (n + 1) < 500 ∧ (n + (n + 1)) = 43 :=
by
  sorry

end greatest_sum_of_consecutive_integers_product_less_500_l27_2726


namespace hyperbola_property_l27_2734

def hyperbola := {x : ℝ // ∃ y : ℝ, x^2 - y^2 / 8 = 1}

def is_on_left_branch (M : hyperbola) : Prop :=
  M.1 < 0

def focus1 : ℝ := -3
def focus2 : ℝ := 3

def distance (a b : ℝ) : ℝ := abs (a - b)

theorem hyperbola_property (M : hyperbola) (hM : is_on_left_branch M) :
  distance M.1 focus1 + distance focus1 focus2 - distance M.1 focus2 = 4 :=
  sorry

end hyperbola_property_l27_2734


namespace roots_quadratic_reciprocal_l27_2752

theorem roots_quadratic_reciprocal (x1 x2 : ℝ) (h1 : x1 + x2 = -8) (h2 : x1 * x2 = 4) :
  (1 / x1) + (1 / x2) = -2 :=
sorry

end roots_quadratic_reciprocal_l27_2752


namespace jaden_toy_cars_l27_2719

theorem jaden_toy_cars :
  let initial : Nat := 14
  let bought : Nat := 28
  let birthday : Nat := 12
  let to_sister : Nat := 8
  let to_friend : Nat := 3
  initial + bought + birthday - to_sister - to_friend = 43 :=
by
  let initial : Nat := 14
  let bought : Nat := 28
  let birthday : Nat := 12
  let to_sister : Nat := 8
  let to_friend : Nat := 3
  show initial + bought + birthday - to_sister - to_friend = 43
  sorry

end jaden_toy_cars_l27_2719


namespace base5_div_l27_2730

-- Definitions for base 5 numbers
def n1 : ℕ := (2 * 125) + (4 * 25) + (3 * 5) + 4  -- 2434_5 in base 10 is 369
def n2 : ℕ := (1 * 25) + (3 * 5) + 2              -- 132_5 in base 10 is 42
def d  : ℕ := (2 * 5) + 1                          -- 21_5 in base 10 is 11

theorem base5_div (res : ℕ) : res = (122 : ℕ) → (n1 + n2) / d = res :=
by sorry

end base5_div_l27_2730


namespace largest_n_for_positive_sum_l27_2789

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

def arithmetic_sum (a d : ℤ) (n : ℕ) : ℤ := n * (2 * a + (n - 1) * d) / 2

theorem largest_n_for_positive_sum (n : ℕ) :
  ∀ (a : ℕ) (S : ℕ → ℤ), (a_1 = 9 ∧ a_5 = 1 ∧ S n > 0) → n = 9 :=
sorry

end largest_n_for_positive_sum_l27_2789


namespace find_larger_number_l27_2747

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 := 
by 
  sorry

end find_larger_number_l27_2747


namespace find_k_l27_2727

theorem find_k (k n m : ℕ) (hk : k > 0) (hn : n > 0) (hm : m > 0) 
  (h : (1 / (n ^ 2 : ℝ) + 1 / (m ^ 2 : ℝ)) = (k : ℝ) / (n ^ 2 + m ^ 2)) : k = 4 :=
sorry

end find_k_l27_2727


namespace probability_of_centrally_symmetric_card_l27_2786

def is_centrally_symmetric (shape : String) : Bool :=
  shape = "parallelogram" ∨ shape = "circle"

theorem probability_of_centrally_symmetric_card :
  let shapes := ["parallelogram", "isosceles_right_triangle", "regular_pentagon", "circle"]
  let total_cards := shapes.length
  let centrally_symmetric_cards := shapes.filter is_centrally_symmetric
  let num_centrally_symmetric := centrally_symmetric_cards.length
  (num_centrally_symmetric : ℚ) / (total_cards : ℚ) = 1 / 2 :=
by
  sorry

end probability_of_centrally_symmetric_card_l27_2786


namespace bananas_added_l27_2779

variable (initial_bananas final_bananas added_bananas : ℕ)

-- Initial condition: There are 2 bananas initially
def initial_bananas_def : Prop := initial_bananas = 2

-- Final condition: There are 9 bananas finally
def final_bananas_def : Prop := final_bananas = 9

-- The number of bananas added to the pile
def added_bananas_def : Prop := final_bananas = initial_bananas + added_bananas

-- Proof statement: Prove that the number of bananas added is 7
theorem bananas_added (h1 : initial_bananas = 2) (h2 : final_bananas = 9) : added_bananas = 7 := by
  sorry

end bananas_added_l27_2779


namespace roots_equal_of_quadratic_eq_zero_l27_2740

theorem roots_equal_of_quadratic_eq_zero (a : ℝ) :
  (∃ x : ℝ, (x^2 - a*x + 1) = 0 ∧ (∀ y : ℝ, (y^2 - a*y + 1) = 0 → y = x)) → (a = 2 ∨ a = -2) :=
by
  sorry

end roots_equal_of_quadratic_eq_zero_l27_2740


namespace arithmetic_sequence_S9_l27_2729

theorem arithmetic_sequence_S9 :
  ∀ {a : ℕ → ℤ} {S : ℕ → ℤ},
  (∀ n : ℕ, S n = (n * (2 * a 1 + (n - 1) * d)) / 2) →
  a 2 = 3 →
  S 4 = 16 →
  S 9 = 81 :=
by
  intro a S h_S h_a2 h_S4
  sorry

end arithmetic_sequence_S9_l27_2729


namespace algebraic_expression_value_l27_2756

-- Definitions based on the conditions
variable {a : ℝ}
axiom root_equation : 2 * a^2 + 3 * a - 4 = 0

-- Definition of the problem: Proving that 2a^2 + 3a equals 4.
theorem algebraic_expression_value : 2 * a^2 + 3 * a = 4 :=
by 
  have h : 2 * a^2 + 3 * a - 4 = 0 := root_equation
  have h' : 2 * a^2 + 3 * a = 4 := by sorry
  exact h'

end algebraic_expression_value_l27_2756


namespace solve_for_x_l27_2765

theorem solve_for_x (x : ℝ) (h : (7 * x + 3) / (x + 5) - 5 / (x + 5) = 2 / (x + 5)) : x = 4 / 7 :=
by
  sorry

end solve_for_x_l27_2765


namespace initial_boys_l27_2766

theorem initial_boys (p : ℝ) (initial_boys : ℝ) (final_boys : ℝ) (final_groupsize : ℝ) : 
  (initial_boys = 0.35 * p) ->
  (final_boys = 0.35 * p - 1) ->
  (final_groupsize = p + 3) ->
  (final_boys / final_groupsize = 0.3) ->
  initial_boys = 13 := 
by
  sorry

end initial_boys_l27_2766


namespace remainder_71_73_div_8_l27_2784

theorem remainder_71_73_div_8 :
  (71 * 73) % 8 = 7 :=
by
  sorry

end remainder_71_73_div_8_l27_2784


namespace travel_time_l27_2709

-- Given conditions
def distance_per_hour : ℤ := 27
def distance_to_sfl : ℤ := 81

-- Theorem statement to prove
theorem travel_time (dph : ℤ) (dts : ℤ) (h1 : dph = distance_per_hour) (h2 : dts = distance_to_sfl) : 
  dts / dph = 3 := 
by
  -- immediately helps execute the Lean statement
  sorry

end travel_time_l27_2709


namespace no_b_satisfies_143b_square_of_integer_l27_2704

theorem no_b_satisfies_143b_square_of_integer :
  ∀ b : ℤ, b > 4 → ¬ ∃ k : ℤ, b^2 + 4 * b + 3 = k^2 :=
by
  intro b hb
  by_contra h
  obtain ⟨k, hk⟩ := h
  have : b^2 + 4 * b + 3 = k ^ 2 := hk
  sorry

end no_b_satisfies_143b_square_of_integer_l27_2704


namespace clara_loses_q_minus_p_l27_2773

def clara_heads_prob : ℚ := 2 / 3
def clara_tails_prob : ℚ := 1 / 3

def ethan_heads_prob : ℚ := 1 / 4
def ethan_tails_prob : ℚ := 3 / 4

def lose_prob_clara : ℚ := clara_heads_prob
def both_tails_prob : ℚ := clara_tails_prob * ethan_tails_prob

noncomputable def total_prob_clara_loses : ℚ :=
  lose_prob_clara + ∑' n : ℕ, (both_tails_prob ^ n) * lose_prob_clara

theorem clara_loses_q_minus_p :
  ∃ (p q : ℕ), Nat.gcd p q = 1 ∧ total_prob_clara_loses = p / q ∧ (q - p = 1) :=
sorry

end clara_loses_q_minus_p_l27_2773


namespace correct_interval_for_monotonic_decrease_l27_2741

noncomputable def f (x : ℝ) : ℝ := |Real.tan (1 / 2 * x - Real.pi / 6)|

theorem correct_interval_for_monotonic_decrease :
  ∀ k : ℤ, ∃ I : Set ℝ,
    I = Set.Ioc (2 * k * Real.pi - 2 * Real.pi / 3) (2 * k * Real.pi + Real.pi / 3) ∧
    ∀ x y, x ∈ I → y ∈ I → x < y → f y < f x :=
sorry

end correct_interval_for_monotonic_decrease_l27_2741


namespace smallest_positive_integer_cube_ends_in_632_l27_2716

theorem smallest_positive_integer_cube_ends_in_632 :
  ∃ n : ℕ, (n > 0) ∧ (n^3 % 1000 = 632) ∧ ∀ m : ℕ, (m > 0) ∧ (m^3 % 1000 = 632) → n ≤ m := 
sorry

end smallest_positive_integer_cube_ends_in_632_l27_2716


namespace find_t_l27_2791

theorem find_t (t : ℝ) :
  (2 * t - 7) * (3 * t - 4) = (3 * t - 9) * (2 * t - 6) →
  t = 26 / 7 := 
by 
  intro h
  sorry

end find_t_l27_2791


namespace find_coordinates_of_M_l27_2771

def point_in_second_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 < 0 ∧ P.2 > 0

def distance_to_x_axis (P : ℝ × ℝ) (d : ℝ) : Prop :=
  abs (P.2) = d

def distance_to_y_axis (P : ℝ × ℝ) (d : ℝ) : Prop :=
  abs (P.1) = d

theorem find_coordinates_of_M :
  ∃ M : ℝ × ℝ, point_in_second_quadrant M ∧ distance_to_x_axis M 5 ∧ distance_to_y_axis M 3 ∧ M = (-3, 5) :=
by
  sorry

end find_coordinates_of_M_l27_2771


namespace n_in_S_implies_n_squared_in_S_l27_2723

-- Definition of the set S
def S : Set ℕ := {n | ∃ a b c d e f : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧ 
                      n - 1 = a^2 + b^2 ∧ n = c^2 + d^2 ∧ n + 1 = e^2 + f^2}

-- The proof goal
theorem n_in_S_implies_n_squared_in_S (n : ℕ) (h : n ∈ S) : n^2 ∈ S :=
by
  sorry

end n_in_S_implies_n_squared_in_S_l27_2723


namespace relationship_between_x_and_y_l27_2795

theorem relationship_between_x_and_y
  (z : ℤ)
  (x : ℝ)
  (y : ℝ)
  (h1 : x = (z^4 + z^3 + z^2 + z + 1) / (z^2 + 1))
  (h2 : y = (z^3 + z^2 + z + 1) / (z^2 + 1)) :
  (y^2 - 2 * y + 2) * (x + y - y^2) - 1 = 0 := 
by
  sorry

end relationship_between_x_and_y_l27_2795


namespace part_I_part_II_l27_2767

variable {a b : ℝ}

theorem part_I (h1 : a * b ≠ 0) (h2 : a * b > 0) :
  b / a + a / b ≥ 2 :=
sorry

theorem part_II (h1 : a * b ≠ 0) (h3 : a * b < 0) :
  abs (b / a + a / b) ≥ 2 :=
sorry

end part_I_part_II_l27_2767


namespace min_cookies_satisfy_conditions_l27_2721

theorem min_cookies_satisfy_conditions : ∃ (b : ℕ), b ≡ 5 [MOD 6] ∧ b ≡ 7 [MOD 8] ∧ b ≡ 8 [MOD 9] ∧ ∀ (b' : ℕ), (b' ≡ 5 [MOD 6] ∧ b' ≡ 7 [MOD 8] ∧ b' ≡ 8 [MOD 9]) → b ≤ b' := 
sorry

end min_cookies_satisfy_conditions_l27_2721


namespace calc_expression_l27_2705

theorem calc_expression (a : ℝ) : 4 * a * a^3 - a^4 = 3 * a^4 := by
  sorry

end calc_expression_l27_2705


namespace days_in_week_l27_2735

theorem days_in_week {F D : ℕ} (h1 : F = 3 + 11) (h2 : F = 2 * D) : D = 7 :=
by
  sorry

end days_in_week_l27_2735


namespace simplification_at_negative_two_l27_2737

noncomputable def simplify_expression (x : ℚ) : ℚ :=
  ((x^2 - 4*x + 4) / (x^2 - 1)) / ((x^2 - 2*x) / (x + 1)) + (1 / (x - 1))

theorem simplification_at_negative_two :
  ∀ x : ℚ, -2 ≤ x ∧ x ≤ 2 ∧ x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 → simplify_expression (-2) = -1 :=
by simp [simplify_expression]; sorry

end simplification_at_negative_two_l27_2737


namespace find_number_of_non_officers_l27_2768

theorem find_number_of_non_officers
  (avg_salary_all : ℝ)
  (avg_salary_officers : ℝ)
  (avg_salary_non_officers : ℝ)
  (num_officers : ℕ) :
  avg_salary_all = 120 ∧
  avg_salary_officers = 450 ∧
  avg_salary_non_officers = 110 ∧
  num_officers = 15 →
  ∃ N : ℕ, (120 * (15 + N) = 450 * 15 + 110 * N) ∧ N = 495 :=
by
  sorry

end find_number_of_non_officers_l27_2768


namespace abs_inequality_solution_l27_2799

theorem abs_inequality_solution (x : ℝ) : (|3 - x| < 4) ↔ (-1 < x ∧ x < 7) :=
by
  sorry

end abs_inequality_solution_l27_2799


namespace sum_possible_values_l27_2739

theorem sum_possible_values (x y : ℝ) (h : x * y - x / y^3 - y / x^3 = 4) :
  (x - 2) * (y - 2) = 4 ∨ (x - 2) * (y - 2) = 0 → (4 + 0 = 4) :=
by
  sorry

end sum_possible_values_l27_2739


namespace polynomial_expansion_sum_constants_l27_2728

theorem polynomial_expansion_sum_constants :
  ∃ (A B C D : ℤ), ((x - 3) * (4 * x ^ 2 + 2 * x - 7) = A * x ^ 3 + B * x ^ 2 + C * x + D) → A + B + C + D = 2 := 
by
  sorry

end polynomial_expansion_sum_constants_l27_2728


namespace largest_of_seven_consecutive_integers_l27_2701

theorem largest_of_seven_consecutive_integers (a : ℕ) (h : a > 0) (sum_eq_77 : (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6) = 77)) :
  a + 6 = 14 :=
by
  sorry

end largest_of_seven_consecutive_integers_l27_2701


namespace correct_yeast_population_change_statement_l27_2763

def yeast_produces_CO2 (aerobic : Bool) : Bool := 
  True

def yeast_unicellular_fungus : Bool := 
  True

def boiling_glucose_solution_purpose : Bool := 
  True

def yeast_facultative_anaerobe : Bool := 
  True

theorem correct_yeast_population_change_statement : 
  (∀ (aerobic : Bool), yeast_produces_CO2 aerobic) →
  yeast_unicellular_fungus →
  boiling_glucose_solution_purpose →
  yeast_facultative_anaerobe →
  "D is correct" = "D is correct" :=
by
  intros
  exact rfl

end correct_yeast_population_change_statement_l27_2763


namespace power_of_power_l27_2787

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := 
by sorry

end power_of_power_l27_2787


namespace cost_of_paving_l27_2717

def length : ℝ := 5.5
def width : ℝ := 4
def rate_per_sq_meter : ℝ := 850

theorem cost_of_paving :
  rate_per_sq_meter * (length * width) = 18700 :=
by
  sorry

end cost_of_paving_l27_2717


namespace integer_coordinates_for_all_vertices_l27_2776

-- Define a three-dimensional vector with integer coordinates
structure Vec3 :=
  (x : ℤ)
  (y : ℤ)
  (z : ℤ)

-- Define a cube with 8 vertices in 3D space
structure Cube :=
  (A1 A2 A3 A4 A1' A2' A3' A4' : Vec3)

-- Assumption: four vertices with integer coordinates that do not lie on the same plane
def has_four_integer_vertices (cube : Cube) : Prop :=
  ∃ (A B C D : Vec3),
    A ∈ [cube.A1, cube.A2, cube.A3, cube.A4, cube.A1', cube.A2', cube.A3', cube.A4'] ∧
    B ∈ [cube.A1, cube.A2, cube.A3, cube.A4, cube.A1', cube.A2', cube.A3', cube.A4'] ∧
    C ∈ [cube.A1, cube.A2, cube.A3, cube.A4, cube.A1', cube.A2', cube.A3', cube.A4'] ∧
    D ∈ [cube.A1, cube.A2, cube.A3, cube.A4, cube.A1', cube.A2', cube.A3', cube.A4'] ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    (C.x - A.x) * (D.y - B.y) ≠ (D.x - B.x) * (C.y - A.y) ∧  -- Ensure not co-planar
    (C.y - A.y) * (D.z - B.z) ≠ (D.y - B.y) * (C.z - A.z)

-- The proof problem: prove all vertices have integer coordinates given the condition
theorem integer_coordinates_for_all_vertices (cube : Cube) (h : has_four_integer_vertices cube) : 
  ∀ v ∈ [cube.A1, cube.A2, cube.A3, cube.A4, cube.A1', cube.A2', cube.A3', cube.A4'], 
    ∃ (v' : Vec3), v = v' := 
  by
  sorry

end integer_coordinates_for_all_vertices_l27_2776


namespace bananas_in_each_group_l27_2724

theorem bananas_in_each_group (total_bananas groups : ℕ) (h1 : total_bananas = 392) (h2 : groups = 196) :
    total_bananas / groups = 2 :=
by
  sorry

end bananas_in_each_group_l27_2724


namespace cylindrical_tank_depth_l27_2781

theorem cylindrical_tank_depth (V : ℝ) (d h : ℝ) (π : ℝ) : 
  V = 1848 ∧ d = 14 ∧ π = Real.pi → h = 12 :=
by
  sorry

end cylindrical_tank_depth_l27_2781


namespace normalize_equation1_normalize_equation2_l27_2748

-- Define the first equation
def equation1 (x y : ℝ) := 2 * x - 3 * y - 10 = 0

-- Define the normalized form of the first equation
def normalized_equation1 (x y : ℝ) := (2 / Real.sqrt 13) * x - (3 / Real.sqrt 13) * y - (10 / Real.sqrt 13) = 0

-- Prove that the normalized form of the first equation is correct
theorem normalize_equation1 (x y : ℝ) (h : equation1 x y) : normalized_equation1 x y := 
sorry

-- Define the second equation
def equation2 (x y : ℝ) := 3 * x + 4 * y = 0

-- Define the normalized form of the second equation
def normalized_equation2 (x y : ℝ) := (3 / 5) * x + (4 / 5) * y = 0

-- Prove that the normalized form of the second equation is correct
theorem normalize_equation2 (x y : ℝ) (h : equation2 x y) : normalized_equation2 x y := 
sorry

end normalize_equation1_normalize_equation2_l27_2748


namespace rowing_trip_time_l27_2764

theorem rowing_trip_time
  (v_0 : ℝ) -- Rowing speed in still water
  (v_c : ℝ) -- Velocity of current
  (d : ℝ) -- Distance to the place
  (h_v0 : v_0 = 10) -- Given condition that rowing speed is 10 kmph
  (h_vc : v_c = 2) -- Given condition that current speed is 2 kmph
  (h_d : d = 144) -- Given condition that distance is 144 km :
  : (d / (v_0 - v_c) + d / (v_0 + v_c)) = 30 := -- Proving the total round trip time is 30 hours
by
  sorry

end rowing_trip_time_l27_2764


namespace initial_pencils_correct_l27_2770

variable (initial_pencils : ℕ)
variable (pencils_added : ℕ := 45)
variable (total_pencils : ℕ := 72)

theorem initial_pencils_correct (h : total_pencils = initial_pencils + pencils_added) : initial_pencils = 27 := by
  sorry

end initial_pencils_correct_l27_2770


namespace ralph_fewer_pictures_l27_2720

-- Define the number of wild animal pictures Ralph and Derrick have.
def ralph_pictures : ℕ := 26
def derrick_pictures : ℕ := 34

-- The main theorem stating that Ralph has 8 fewer pictures than Derrick.
theorem ralph_fewer_pictures : derrick_pictures - ralph_pictures = 8 := by
  -- The proof is omitted, denoted by 'sorry'.
  sorry

end ralph_fewer_pictures_l27_2720


namespace compound_proposition_C_l27_2769

open Real

def p : Prop := ∃ x : ℝ, x - 2 > log x 
def q : Prop := ∀ x : ℝ, sin x < x

theorem compound_proposition_C : p ∧ ¬q :=
by sorry

end compound_proposition_C_l27_2769


namespace daria_needs_to_earn_l27_2783

variable (ticket_cost : ℕ) (current_money : ℕ) (total_tickets : ℕ)

def total_cost (ticket_cost : ℕ) (total_tickets : ℕ) : ℕ :=
  ticket_cost * total_tickets

def money_needed (total_cost : ℕ) (current_money : ℕ) : ℕ :=
  total_cost - current_money

theorem daria_needs_to_earn :
  total_cost 90 4 - 189 = 171 :=
by
  sorry

end daria_needs_to_earn_l27_2783


namespace option_d_correct_l27_2744

theorem option_d_correct (a b : ℝ) : (a - b)^2 = (b - a)^2 := 
by {
  sorry
}

end option_d_correct_l27_2744


namespace complement_supplement_angle_l27_2743

theorem complement_supplement_angle (α : ℝ) : 
  ( 180 - α) = 3 * ( 90 - α ) → α = 45 :=
by 
  sorry

end complement_supplement_angle_l27_2743


namespace min_value_a_3b_9c_l27_2753

theorem min_value_a_3b_9c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 27) : 
  a + 3 * b + 9 * c ≥ 27 := 
sorry

end min_value_a_3b_9c_l27_2753


namespace theater_loss_l27_2708

/-- 
A movie theater has a total capacity of 50 people and charges $8 per ticket.
On a Tuesday night, they only sold 24 tickets. 
Prove that the revenue lost by not selling out is $208.
-/
theorem theater_loss 
  (capacity : ℕ) 
  (price : ℕ) 
  (sold_tickets : ℕ) 
  (h_cap : capacity = 50) 
  (h_price : price = 8) 
  (h_sold : sold_tickets = 24) : 
  capacity * price - sold_tickets * price = 208 :=
by
  sorry

end theater_loss_l27_2708


namespace chord_line_equation_l27_2759

theorem chord_line_equation (x y : ℝ) 
  (ellipse : ∀ (x y : ℝ), x^2 / 36 + y^2 / 9 = 1)
  (bisect_point : x / 2 = 4 ∧ y / 2 = 2) : 
  x + 2 * y - 8 = 0 :=
sorry

end chord_line_equation_l27_2759


namespace four_digit_numbers_with_product_exceeds_10_l27_2703

theorem four_digit_numbers_with_product_exceeds_10 : 
  (∃ count : ℕ, 
    count = (6 * 58 * 10) ∧
    count = 3480) := 
by {
  sorry
}

end four_digit_numbers_with_product_exceeds_10_l27_2703


namespace triangle_side_lengths_l27_2712

open Real

theorem triangle_side_lengths (a b c : ℕ) (R : ℝ)
    (h1 : a * a + 4 * d * d = 2500)
    (h2 : b * b + 4 * e * e = 2500)
    (h3 : R = 12.5)
    (h4 : (2:ℝ) * d ≤ a)
    (h5 : (2:ℝ) * e ≤ b)
    (h6 : a > b)
    (h7 : a ≠ b)
    (h8 : 2 * R = 25) :
    (a, b, c) = (15, 7, 20) := by
  sorry

end triangle_side_lengths_l27_2712


namespace day_of_week_after_45_days_l27_2794

theorem day_of_week_after_45_days (day_of_week : ℕ → String) (birthday_is_tuesday : day_of_week 0 = "Tuesday") : day_of_week 45 = "Friday" :=
by
  sorry

end day_of_week_after_45_days_l27_2794


namespace problem1_problem2_l27_2775

-- Problem 1: Prove the expression
theorem problem1 (a b : ℝ) : 
  2 * a * (a - 2 * b) - (2 * a - b) ^ 2 = -2 * a ^ 2 - b ^ 2 := 
sorry

-- Problem 2: Prove the solution to the equation
theorem problem2 (x : ℝ) (h : (x - 1) ^ 3 - 3 = 3 / 8) : 
  x = 5 / 2 := 
sorry

end problem1_problem2_l27_2775


namespace ScientificNotation_of_45400_l27_2792

theorem ScientificNotation_of_45400 :
  45400 = 4.54 * 10^4 := sorry

end ScientificNotation_of_45400_l27_2792


namespace find_sum_u_v_l27_2762

theorem find_sum_u_v (u v : ℤ) (huv : 0 < v ∧ v < u) (pentagon_area : u^2 + 3 * u * v = 451) : u + v = 21 :=
by 
  sorry

end find_sum_u_v_l27_2762


namespace exists_m_such_that_m_plus_one_pow_zero_eq_one_l27_2797

theorem exists_m_such_that_m_plus_one_pow_zero_eq_one : 
  ∃ m : ℤ, (m + 1)^0 = 1 ∧ m ≠ -1 :=
by
  sorry

end exists_m_such_that_m_plus_one_pow_zero_eq_one_l27_2797


namespace required_cement_l27_2738

def total_material : ℝ := 0.67
def sand : ℝ := 0.17
def dirt : ℝ := 0.33
def cement : ℝ := 0.17

theorem required_cement : cement = total_material - (sand + dirt) := 
by
  sorry

end required_cement_l27_2738


namespace johnny_future_years_l27_2732

theorem johnny_future_years (x : ℕ) (h1 : 8 + x = 2 * (8 - 3)) : x = 2 :=
by
  sorry

end johnny_future_years_l27_2732


namespace monkey_reaches_tree_top_in_hours_l27_2707

-- Definitions based on conditions
def height_of_tree : ℕ := 22
def hop_per_hour : ℕ := 3
def slip_per_hour : ℕ := 2
def effective_climb_per_hour : ℕ := hop_per_hour - slip_per_hour

-- The theorem we want to prove
theorem monkey_reaches_tree_top_in_hours
  (height_of_tree hop_per_hour slip_per_hour : ℕ)
  (h1 : height_of_tree = 22)
  (h2 : hop_per_hour = 3)
  (h3 : slip_per_hour = 2) :
  ∃ t : ℕ, t = 22 ∧ effective_climb_per_hour * (t - 1) + hop_per_hour = height_of_tree := by
  sorry

end monkey_reaches_tree_top_in_hours_l27_2707


namespace sqrt_fraction_addition_l27_2755

theorem sqrt_fraction_addition :
  (Real.sqrt ((25 : ℝ) / 36 + 16 / 9)) = Real.sqrt 89 / 6 := by
  sorry

end sqrt_fraction_addition_l27_2755


namespace slower_time_to_reach_top_l27_2780

def time_for_lola (stories : ℕ) (time_per_story : ℕ) : ℕ :=
  stories * time_per_story

def time_for_tara (stories : ℕ) (time_per_story : ℕ) (stopping_time : ℕ) (num_stops : ℕ) : ℕ :=
  (stories * time_per_story) + (num_stops * stopping_time)

theorem slower_time_to_reach_top (stories : ℕ) (lola_time_per_story : ℕ) (tara_time_per_story : ℕ) 
  (tara_stop_time : ℕ) (tara_num_stops : ℕ) : 
  stories = 20 
  → lola_time_per_story = 10 
  → tara_time_per_story = 8 
  → tara_stop_time = 3
  → tara_num_stops = 18
  → max (time_for_lola stories lola_time_per_story) (time_for_tara stories tara_time_per_story tara_stop_time tara_num_stops) = 214 :=
by sorry

end slower_time_to_reach_top_l27_2780


namespace hall_length_width_difference_l27_2733

theorem hall_length_width_difference : 
  ∃ L W : ℝ, W = (1 / 2) * L ∧ L * W = 450 ∧ L - W = 15 :=
sorry

end hall_length_width_difference_l27_2733


namespace trigonometric_transform_l27_2785

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def h (x : ℝ) : ℝ := f (x - 3)
noncomputable def g (x : ℝ) : ℝ := 3 * h (x / 3)

theorem trigonometric_transform (x : ℝ) : g x = 3 * Real.sin (x / 3 - 3) := by
  sorry

end trigonometric_transform_l27_2785


namespace fraction_Cal_to_Anthony_l27_2706

-- definitions for Mabel, Anthony, Cal, and Jade's transactions
def Mabel_transactions : ℕ := 90
def Anthony_transactions : ℕ := Mabel_transactions + (Mabel_transactions / 10)
def Jade_transactions : ℕ := 85
def Cal_transactions : ℕ := Jade_transactions - 19

-- goal: prove the fraction Cal handled compared to Anthony is 2/3
theorem fraction_Cal_to_Anthony : (Cal_transactions : ℚ) / (Anthony_transactions : ℚ) = 2 / 3 :=
by
  sorry

end fraction_Cal_to_Anthony_l27_2706


namespace polygon_sides_and_diagonals_l27_2718

theorem polygon_sides_and_diagonals (n : ℕ) :
  (180 * (n - 2) = 3 * 360 + 180) → n = 9 ∧ (n - 3 = 6) :=
by
  intro h_sum_angles
  -- This is where you would provide the proof.
  sorry

end polygon_sides_and_diagonals_l27_2718


namespace simplify_expression_l27_2746

variable (x : ℝ)

theorem simplify_expression : 3 * x + 4 * x^3 + 2 - (7 - 3 * x - 4 * x^3) = 8 * x^3 + 6 * x - 5 := 
by 
  sorry

end simplify_expression_l27_2746


namespace jimmy_earnings_l27_2702

theorem jimmy_earnings : 
  let price15 := 15
  let price20 := 20
  let discount := 5
  let sale_price15 := price15 - discount
  let sale_price20 := price20 - discount
  let num_low_worth := 4
  let num_high_worth := 1
  num_low_worth * sale_price15 + num_high_worth * sale_price20 = 55 :=
by
  sorry

end jimmy_earnings_l27_2702


namespace solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_solve_quadratic_4_solve_quadratic_5_solve_quadratic_6_l27_2758

-- Problem 1: 5x² = 40x
theorem solve_quadratic_1 (x : ℝ) : 5 * x^2 = 40 * x ↔ (x = 0 ∨ x = 8) :=
by sorry

-- Problem 2: 25/9 x² = 100
theorem solve_quadratic_2 (x : ℝ) : (25 / 9) * x^2 = 100 ↔ (x = 6 ∨ x = -6) :=
by sorry

-- Problem 3: 10x = x² + 21
theorem solve_quadratic_3 (x : ℝ) : 10 * x = x^2 + 21 ↔ (x = 7 ∨ x = 3) :=
by sorry

-- Problem 4: x² = 12x + 288
theorem solve_quadratic_4 (x : ℝ) : x^2 = 12 * x + 288 ↔ (x = 24 ∨ x = -12) :=
by sorry

-- Problem 5: x² + 20 1/4 = 11 1/4 x
theorem solve_quadratic_5 (x : ℝ) : x^2 + 81 / 4 = 45 / 4 * x ↔ (x = 9 / 4 ∨ x = 9) :=
by sorry

-- Problem 6: 1/12 x² + 7/12 x = 19
theorem solve_quadratic_6 (x : ℝ) : (1 / 12) * x^2 + (7 / 12) * x = 19 ↔ (x = 12 ∨ x = -19) :=
by sorry

end solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_solve_quadratic_4_solve_quadratic_5_solve_quadratic_6_l27_2758


namespace modulus_of_complex_number_l27_2777

/-- Definition of the imaginary unit i defined as the square root of -1 --/
def i : ℂ := Complex.I

/-- Statement that the modulus of z = i (1 - i) equals sqrt(2) --/
theorem modulus_of_complex_number : Complex.abs (i * (1 - i)) = Real.sqrt 2 :=
sorry

end modulus_of_complex_number_l27_2777


namespace sin_pi_plus_alpha_l27_2788

/-- Given that \(\sin \left(\frac{\pi}{2}+\alpha \right) = \frac{3}{5}\)
    and \(\alpha \in (0, \frac{\pi}{2})\),
    prove that \(\sin(\pi + \alpha) = -\frac{4}{5}\). -/
theorem sin_pi_plus_alpha (α : ℝ) (h1 : Real.sin (Real.pi / 2 + α) = 3 / 5)
  (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.sin (Real.pi + α) = -4 / 5 := 
  sorry

end sin_pi_plus_alpha_l27_2788
