import Mathlib

namespace odd_function_sum_l367_36792

noncomputable def f : ℝ → ℝ := sorry

theorem odd_function_sum :
  (∀ x, f x = -f (-x)) ∧ 
  (∀ x y (hx : 3 ≤ x) (hy : y ≤ 7), x < y → f x < f y) ∧ 
  ( ∃ x, 3 ≤ x ∧ x ≤ 6 ∧ f x = 8) ∧ 
  ( ∃ x, 3 ≤ x ∧ x ≤ 6 ∧ f x = -1) →
  (2 * f (-6) + f (-3) = -15) :=
by
  intros
  sorry

end odd_function_sum_l367_36792


namespace find_speed_l367_36730

-- Definitions corresponding to conditions
def JacksSpeed (x : ℝ) : ℝ := x^2 - 7 * x - 12
def JillsDistance (x : ℝ) : ℝ := x^2 - 3 * x - 10
def JillsTime (x : ℝ) : ℝ := x + 2

-- Theorem statement
theorem find_speed (x : ℝ) (hx : x ≠ -2) (h_speed_eq : JacksSpeed x = (JillsDistance x) / (JillsTime x)) : JacksSpeed x = 2 :=
by
  sorry

end find_speed_l367_36730


namespace bread_consumption_l367_36740

-- Definitions using conditions
def members := 4
def slices_snacks := 2
def slices_per_loaf := 12
def total_loaves := 5
def total_days := 3

-- The main theorem to prove
theorem bread_consumption :
  (3 * members * (B + slices_snacks) = total_loaves * slices_per_loaf) → B = 3 :=
by
  intro h
  sorry

end bread_consumption_l367_36740


namespace greatest_triangle_perimeter_l367_36706

theorem greatest_triangle_perimeter :
  ∃ x : ℕ, (x > 4) ∧ (x ≤ 6) ∧ (∀ (y : ℕ), (y > 4) ∧ (y ≤ 6) → 5 * y + 20 = 50) := sorry

end greatest_triangle_perimeter_l367_36706


namespace slope_of_line_between_solutions_l367_36712

theorem slope_of_line_between_solutions (x1 y1 x2 y2 : ℝ) (h1 : 3 / x1 + 4 / y1 = 0) (h2 : 3 / x2 + 4 / y2 = 0) (h3 : x1 ≠ x2) :
  (y2 - y1) / (x2 - x1) = -4 / 3 := 
sorry

end slope_of_line_between_solutions_l367_36712


namespace point_P_through_graph_l367_36700

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 + a^(x - 1)

theorem point_P_through_graph (a : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) : 
  f a 1 = 5 :=
by
  unfold f
  sorry

end point_P_through_graph_l367_36700


namespace time_to_fill_one_barrel_with_leak_l367_36741

-- Define the conditions
def normal_time_per_barrel := 3
def time_to_fill_12_barrels_no_leak := normal_time_per_barrel * 12
def additional_time_due_to_leak := 24
def time_to_fill_12_barrels_with_leak (t : ℕ) := 12 * t

-- Define the theorem
theorem time_to_fill_one_barrel_with_leak :
  ∃ t : ℕ, time_to_fill_12_barrels_with_leak t = time_to_fill_12_barrels_no_leak + additional_time_due_to_leak ∧ t = 5 :=
by {
  use 5, 
  sorry
}

end time_to_fill_one_barrel_with_leak_l367_36741


namespace total_ticket_income_l367_36744

-- All given conditions as definitions/assumptions
def total_seats : ℕ := 200
def children_tickets : ℕ := 60
def adult_ticket_price : ℝ := 3.00
def children_ticket_price : ℝ := 1.50
def adult_tickets : ℕ := total_seats - children_tickets

-- The claim we need to prove
theorem total_ticket_income :
  (adult_tickets * adult_ticket_price + children_tickets * children_ticket_price) = 510.00 :=
by
  -- Placeholder to complete proof later
  sorry

end total_ticket_income_l367_36744


namespace more_karabases_than_barabases_l367_36759

/-- In the fairy-tale land of Perra-Terra, each Karabas is acquainted with nine Barabases, 
    and each Barabas is acquainted with ten Karabases. We aim to prove that there are more Karabases than Barabases. -/
theorem more_karabases_than_barabases (K B : ℕ) (h1 : 9 * K = 10 * B) : K > B := 
by {
    -- Following the conditions and conclusion
    sorry
}

end more_karabases_than_barabases_l367_36759


namespace speedster_convertibles_l367_36716

theorem speedster_convertibles 
  (T : ℕ) 
  (h1 : T > 0)
  (h2 : 30 = (2/3 : ℚ) * T)
  (h3 : ∀ n, n = (1/3 : ℚ) * T → ∃ m, m = (4/5 : ℚ) * n) :
  ∃ m, m = 12 := 
sorry

end speedster_convertibles_l367_36716


namespace solve_for_x_l367_36781

theorem solve_for_x : ∀ (x : ℝ), (2 * x + 3) / 5 = 11 → x = 26 :=
by {
  sorry
}

end solve_for_x_l367_36781


namespace largest_root_of_quadratic_l367_36769

theorem largest_root_of_quadratic :
  ∀ (x : ℝ), x^2 - 9*x - 22 = 0 → x ≤ 11 :=
by
  sorry

end largest_root_of_quadratic_l367_36769


namespace mass_percentage_oxygen_NaBrO3_l367_36704

-- Definitions
def molar_mass_Na : ℝ := 22.99
def molar_mass_Br : ℝ := 79.90
def molar_mass_O : ℝ := 16.00

def molar_mass_NaBrO3 : ℝ := molar_mass_Na + molar_mass_Br + 3 * molar_mass_O

-- Theorem: proof that the mass percentage of oxygen in NaBrO3 is 31.81%
theorem mass_percentage_oxygen_NaBrO3 :
  ((3 * molar_mass_O) / molar_mass_NaBrO3) * 100 = 31.81 := by
  sorry

end mass_percentage_oxygen_NaBrO3_l367_36704


namespace range_of_a_l367_36753

theorem range_of_a (a : ℝ) (h : ¬ ∃ t : ℝ, t^2 - a * t - a < 0) : -4 ≤ a ∧ a ≤ 0 :=
by 
  sorry

end range_of_a_l367_36753


namespace marble_weight_l367_36709

theorem marble_weight (m d : ℝ) : (9 * m = 4 * d) → (3 * d = 36) → (m = 16 / 3) :=
by
  intro h1 h2
  sorry

end marble_weight_l367_36709


namespace number_of_tables_cost_price_l367_36715

theorem number_of_tables_cost_price
  (C S : ℝ)
  (N : ℝ)
  (h1 : N * C = 20 * S)
  (h2 : S = 0.75 * C) :
  N = 15 := by
  -- insert proof here
  sorry

end number_of_tables_cost_price_l367_36715


namespace bankers_discount_is_correct_l367_36767

-- Define the given conditions
def TD := 45   -- True discount in Rs.
def FV := 270  -- Face value in Rs.

-- Calculate Present Value based on the given conditions
def PV := FV - TD

-- Define the formula for Banker's Discount
def BD := TD + (TD ^ 2 / PV)

-- Prove that the Banker's Discount is Rs. 54 given the conditions
theorem bankers_discount_is_correct : BD = 54 :=
by
  -- Steps to prove the theorem can be filled here
  -- Add "sorry" to skip the actual proof
  sorry

end bankers_discount_is_correct_l367_36767


namespace train_length_l367_36777

theorem train_length :
  ∀ (t : ℝ) (v_man : ℝ) (v_train : ℝ),
  t = 41.9966402687785 →
  v_man = 3 →
  v_train = 63 →
  (v_train - v_man) * (5 / 18) * t = 699.94400447975 :=
by
  intros t v_man v_train ht hv_man hv_train
  -- Use the given conditions as definitions
  rw [ht, hv_man, hv_train]
  sorry

end train_length_l367_36777


namespace production_rate_is_constant_l367_36774

def drum_rate := 6 -- drums per day

def days_needed_to_produce (n : ℕ) : ℕ := n / drum_rate

theorem production_rate_is_constant (n : ℕ) : days_needed_to_produce n = n / drum_rate :=
by
  sorry

end production_rate_is_constant_l367_36774


namespace sodium_chloride_formed_l367_36750

section 

-- Definitions based on the conditions
def hydrochloric_acid_moles : ℕ := 2
def sodium_bicarbonate_moles : ℕ := 2

-- Balanced chemical equation represented as a function (1:1 reaction ratio)
def reaction (hcl_moles naHCO3_moles : ℕ) : ℕ := min hcl_moles naHCO3_moles

-- Theorem stating the reaction outcome
theorem sodium_chloride_formed : reaction hydrochloric_acid_moles sodium_bicarbonate_moles = 2 :=
by
  -- Proof is omitted
  sorry

end

end sodium_chloride_formed_l367_36750


namespace circle_equation_l367_36718

theorem circle_equation
  (a b r : ℝ)
  (ha : (4 - a)^2 + (1 - b)^2 = r^2)
  (hb : (2 - a)^2 + (1 - b)^2 = r^2)
  (ht : (b - 1) / (a - 2) = -1) :
  (a = 3) ∧ (b = 0) ∧ (r = 2) :=
by {
  sorry
}

-- Given the above values for a, b, r
def circle_equation_verified : Prop :=
  (∀ (x y : ℝ), ((x - 3)^2 + y^2) = 4)

example : circle_equation_verified :=
by {
  sorry
}

end circle_equation_l367_36718


namespace no_zero_root_l367_36751

theorem no_zero_root (x : ℝ) :
  (¬ (∃ x : ℝ, (4 * x ^ 2 - 3 = 49) ∧ x = 0)) ∧
  (¬ (∃ x : ℝ, (x ^ 2 - x - 20 = 0) ∧ x = 0)) :=
by
  sorry

end no_zero_root_l367_36751


namespace manuscript_typing_cost_l367_36754

-- Defining the conditions as per our problem
def first_time_typing_rate : ℕ := 5 -- $5 per page for first-time typing
def revision_rate : ℕ := 3 -- $3 per page per revision

def num_pages : ℕ := 100 -- total number of pages
def revised_once : ℕ := 30 -- number of pages revised once
def revised_twice : ℕ := 20 -- number of pages revised twice
def no_revision := num_pages - (revised_once + revised_twice) -- pages with no revisions

-- Defining the cost function to calculate the total cost of typing
noncomputable def total_typing_cost : ℕ :=
  (num_pages * first_time_typing_rate) + (revised_once * revision_rate) + (revised_twice * revision_rate * 2)

-- Lean theorem statement to prove the total cost is $710
theorem manuscript_typing_cost :
  total_typing_cost = 710 := by
  sorry

end manuscript_typing_cost_l367_36754


namespace max_value_of_n_l367_36796

theorem max_value_of_n (A B : ℤ) (h1 : A * B = 48) : 
  ∃ n, (∀ n', (∃ A' B', (A' * B' = 48) ∧ (n' = 2 * B' + 3 * A')) → n' ≤ n) ∧ n = 99 :=
by
  sorry

end max_value_of_n_l367_36796


namespace problem_equivalence_l367_36711

theorem problem_equivalence :
  (1 / Real.sin (Real.pi / 18) - Real.sqrt 3 / Real.sin (4 * Real.pi / 18)) = 4 := 
sorry

end problem_equivalence_l367_36711


namespace probability_blackboard_empty_k_l367_36793

-- Define the conditions for the problem
def Ben_blackboard_empty_probability (n : ℕ) : ℚ :=
  if h : n = 2013 then (2 * (2013 / 3) + 1) / 2^(2013 / 3 * 2) else 0 / 1

-- Define the theorem that Ben's blackboard is empty after 2013 flips, and determine k
theorem probability_blackboard_empty_k :
  ∃ (u v k : ℕ), Ben_blackboard_empty_probability 2013 = (2 * u + 1) / (2^k * (2 * v + 1)) ∧ k = 1336 :=
by sorry

end probability_blackboard_empty_k_l367_36793


namespace graveling_cost_is_correct_l367_36772

noncomputable def graveling_cost (lawn_length lawn_breadth road_width cost_per_sqm : ℝ) : ℝ :=
  let road1_area := road_width * lawn_breadth
  let road2_area := road_width * lawn_length
  let intersection_area := road_width * road_width
  let total_area := road1_area + road2_area - intersection_area
  total_area * cost_per_sqm

theorem graveling_cost_is_correct :
  graveling_cost 80 60 10 2 = 2600 := by
  sorry

end graveling_cost_is_correct_l367_36772


namespace triangle_cos_area_l367_36783

/-- In triangle ABC, with angles A, B, and C, opposite sides a, b, and c respectively, given the condition 
    a * cos C = (2 * b - c) * cos A, prove: 
    1. cos A = 1/2
    2. If a = 6 and b + c = 8, then the area of triangle ABC is 7 * sqrt 3 / 3 --/
theorem triangle_cos_area (A B C : ℝ) (a b c : ℝ) (h1 : a * Real.cos C = (2 * b - c) * Real.cos A)
  (h2 : a = 6) (h3 : b + c = 8) :
  Real.cos A = 1 / 2 ∧ ∃ area : ℝ, area = 7 * Real.sqrt 3 / 3 :=
by {
  sorry
}

end triangle_cos_area_l367_36783


namespace car_speed_kmph_l367_36731

noncomputable def speed_of_car (d : ℝ) (t : ℝ) : ℝ :=
  (d / t) * 3.6

theorem car_speed_kmph : speed_of_car 10 0.9999200063994881 = 36000.29 := by
  sorry

end car_speed_kmph_l367_36731


namespace gcd_of_three_numbers_l367_36768

theorem gcd_of_three_numbers : Nat.gcd (Nat.gcd 279 372) 465 = 93 := 
by 
  sorry

end gcd_of_three_numbers_l367_36768


namespace arithmetic_sequence_problem_l367_36719

theorem arithmetic_sequence_problem (a : ℕ → ℤ) (h_arith : ∀ n m, a (n + 1) - a n = a (m + 1) - a m) (h_incr : ∀ n, a (n + 1) > a n) (h_prod : a 4 * a 5 = 13) : a 3 * a 6 = -275 := 
sorry

end arithmetic_sequence_problem_l367_36719


namespace joe_average_test_score_l367_36722

theorem joe_average_test_score 
  (A B C : ℕ) 
  (Hsum : A + B + C = 135) 
  : (A + B + C + 25) / 4 = 40 :=
by
  sorry

end joe_average_test_score_l367_36722


namespace binomial_expansion_example_l367_36791

theorem binomial_expansion_example :
  57^3 + 3 * (57^2) * 4 + 3 * 57 * (4^2) + 4^3 = 226981 :=
by
  -- The proof would go here, using the steps outlined.
  sorry

end binomial_expansion_example_l367_36791


namespace sin_90_deg_l367_36787

theorem sin_90_deg : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_90_deg_l367_36787


namespace log_sum_property_l367_36710

noncomputable def f (a : ℝ) (x : ℝ) := Real.log x / Real.log a
noncomputable def f_inv (a : ℝ) (y : ℝ) := a ^ y

theorem log_sum_property (a : ℝ) (h1 : f_inv a 2 = 9) (h2 : f a 9 = 2) : f a 9 + f a 6 = 1 :=
by
  sorry

end log_sum_property_l367_36710


namespace hyperbola_focal_length_l367_36748

def is_hyperbola (x y a : ℝ) : Prop := (x^2) / (a^2) - (y^2) = 1
def is_perpendicular_asymptote (slope_asymptote slope_line : ℝ) : Prop := slope_asymptote * slope_line = -1

theorem hyperbola_focal_length {a : ℝ} (h1 : is_hyperbola x y a)
  (h2 : is_perpendicular_asymptote (1 / a) (-1)) : 2 * Real.sqrt 2 = 2 * Real.sqrt 2 :=
sorry

end hyperbola_focal_length_l367_36748


namespace total_value_of_item_l367_36797

variable {V : ℝ}

theorem total_value_of_item (h : 0.07 * (V - 1000) = 109.20) : V = 2560 := 
by
  sorry

end total_value_of_item_l367_36797


namespace min_value_of_exponential_l367_36775

theorem min_value_of_exponential (x y : ℝ) (h : x + 2 * y = 1) : 
  2^x + 4^y ≥ 2 * Real.sqrt 2 ∧ 
  (∀ a, (2^x + 4^y = a) → a ≥ 2 * Real.sqrt 2) :=
by
  sorry

end min_value_of_exponential_l367_36775


namespace original_price_of_wand_l367_36707

theorem original_price_of_wand (P : ℝ) (h1 : 8 = P / 8) : P = 64 :=
by sorry

end original_price_of_wand_l367_36707


namespace values_of_a_and_b_l367_36785

theorem values_of_a_and_b (a b : ℝ) : 
  (∀ x : ℝ, (x + a - 2 > 0 ∧ 2 * x - b - 1 < 0) ↔ (0 < x ∧ x < 1)) → (a = 2 ∧ b = 1) :=
by 
  sorry

end values_of_a_and_b_l367_36785


namespace range_of_m_l367_36725

noncomputable def f (x : ℝ) : ℝ :=
  if x < -2 then 3 + 3 * x
  else if x <= 3 then -1
  else x + 5

theorem range_of_m (m : ℝ) (x : ℝ) (hx : f x ≥ 1 / m - 4) :
  m < 0 ∨ m = 1 :=
sorry

end range_of_m_l367_36725


namespace james_driving_speed_l367_36755

theorem james_driving_speed
  (distance : ℝ)
  (total_time : ℝ)
  (stop_time : ℝ)
  (driving_time : ℝ)
  (speed : ℝ)
  (h1 : distance = 360)
  (h2 : total_time = 7)
  (h3 : stop_time = 1)
  (h4 : driving_time = total_time - stop_time)
  (h5 : speed = distance / driving_time) :
  speed = 60 := by
  -- Here you would put the detailed proof.
  sorry

end james_driving_speed_l367_36755


namespace parabola_vertex_expression_l367_36771

theorem parabola_vertex_expression (h k : ℝ) :
  (h = 2 ∧ k = 3) →
  ∃ (a : ℝ), (a ≠ 0) ∧
    (∀ x y : ℝ, y = a * (x - h)^2 + k ↔ y = -(x - 2)^2 + 3) :=
by
  sorry

end parabola_vertex_expression_l367_36771


namespace fido_reach_fraction_simplified_l367_36764

noncomputable def fidoReach (s r : ℝ) : ℝ :=
  let octagonArea := 2 * (1 + Real.sqrt 2) * s^2
  let circleArea := Real.pi * (s / Real.sqrt (2 + Real.sqrt 2))^2
  circleArea / octagonArea

theorem fido_reach_fraction_simplified (s : ℝ) :
  (∃ a b : ℕ, fidoReach s (s / Real.sqrt (2 + Real.sqrt 2)) = (Real.sqrt a / b) * Real.pi ∧ a * b = 16) :=
  sorry

end fido_reach_fraction_simplified_l367_36764


namespace find_a8_a12_l367_36728

noncomputable def geometric_sequence_value_8_12 (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then a 0 else a 0 * q^n

theorem find_a8_a12 (a : ℕ → ℝ) (q : ℝ) (terms_geometric : ∀ n, a n = a 0 * q^n)
  (h2_6 : a 2 + a 6 = 3) (h6_10 : a 6 + a 10 = 12) :
  a 8 + a 12 = 24 :=
by
  sorry

end find_a8_a12_l367_36728


namespace find_sets_l367_36758

open Set

noncomputable def U := ℝ
def A := {x : ℝ | Real.log x / Real.log 2 <= 2}
def B := {x : ℝ | x ≥ 1}

theorem find_sets (x : ℝ) :
  (A = {x : ℝ | -1 ≤ x ∧ x < 3}) ∧
  (B = {x : ℝ | -2 < x ∧ x ≤ 3}) ∧
  (compl A ∩ B = {x : ℝ | (-2 < x ∧ x < -1) ∨ x = 3}) :=
  sorry

end find_sets_l367_36758


namespace parabola_properties_l367_36776

theorem parabola_properties :
  ∀ x : ℝ, (x - 3)^2 + 5 = (x-3)^2 + 5 ∧ 
  (x - 3)^2 + 5 > 0 ∧ 
  (∃ h : ℝ, h = 3 ∧ ∀ x1 x2 : ℝ, (x1 - h)^2 <= (x2 - h)^2) ∧ 
  (∃ h k : ℝ, h = 3 ∧ k = 5) := 
by 
  sorry

end parabola_properties_l367_36776


namespace units_digit_p_plus_5_l367_36799

theorem units_digit_p_plus_5 (p : ℕ) (h1 : p % 2 = 0) (h2 : p % 10 = 6) (h3 : (p^3 % 10) - (p^2 % 10) = 0) : (p + 5) % 10 = 1 :=
by
  sorry

end units_digit_p_plus_5_l367_36799


namespace least_positive_multiple_of_24_gt_450_l367_36736

theorem least_positive_multiple_of_24_gt_450 : 
  ∃ n : ℕ, n > 450 ∧ (∃ k : ℕ, n = 24 * k) → n = 456 :=
by 
  sorry

end least_positive_multiple_of_24_gt_450_l367_36736


namespace geometric_sequence_property_l367_36721

theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geom: ∀ n, a (n + 1) = a n * r) 
  (h_pos: ∀ n, a n > 0)
  (h_root1: a 3 * a 15 = 8)
  (h_root2: a 3 + a 15 = 6) :
  a 1 * a 17 / a 9 = 2 * Real.sqrt 2 :=
by
  sorry

end geometric_sequence_property_l367_36721


namespace Jace_post_break_time_correct_l367_36708

noncomputable def Jace_post_break_time (total_distance : ℝ) (speed : ℝ) (pre_break_time : ℝ) : ℝ :=
  (total_distance - (speed * pre_break_time)) / speed

theorem Jace_post_break_time_correct :
  Jace_post_break_time 780 60 4 = 9 :=
by
  sorry

end Jace_post_break_time_correct_l367_36708


namespace three_digit_number_problem_l367_36779

theorem three_digit_number_problem (c d : ℕ) (h1 : 400 + c*10 + 1 = 786 - (300 + d*10 + 5)) (h2 : (300 + d*10 + 5) % 7 = 0) : c + d = 8 := 
sorry

end three_digit_number_problem_l367_36779


namespace hundredth_term_sequence_l367_36732

def numerators (n : ℕ) : ℕ := 1 + (n - 1) * 2
def denominators (n : ℕ) : ℕ := 2 + (n - 1) * 3

theorem hundredth_term_sequence : numerators 100 / denominators 100 = 199 / 299 := by
  sorry

end hundredth_term_sequence_l367_36732


namespace find_length_of_rod_l367_36784

-- Constants representing the given conditions
def weight_6m_rod : ℝ := 6.1
def length_6m_rod : ℝ := 6
def weight_unknown_rod : ℝ := 12.2

-- Proof statement ensuring the length of the rod that weighs 12.2 kg is 12 meters
theorem find_length_of_rod (L : ℝ) (h : weight_6m_rod / length_6m_rod = weight_unknown_rod / L) : 
  L = 12 := by
  sorry

end find_length_of_rod_l367_36784


namespace gcd_all_abc_plus_cba_l367_36752

noncomputable def gcd_of_abc_cba (a : ℕ) (b : ℕ := 2 * a) (c : ℕ := 3 * a) : ℕ :=
  let abc := 64 * a + 8 * b + c
  let cba := 64 * c + 8 * b + a
  Nat.gcd (abc + cba) 300

theorem gcd_all_abc_plus_cba (a : ℕ) : gcd_of_abc_cba a = 300 :=
  sorry

end gcd_all_abc_plus_cba_l367_36752


namespace cookie_ratio_l367_36739

theorem cookie_ratio (cookies_monday cookies_tuesday cookies_wednesday final_cookies : ℕ)
  (h1 : cookies_monday = 32)
  (h2 : cookies_tuesday = cookies_monday / 2)
  (h3 : final_cookies = 92)
  (h4 : cookies_wednesday = final_cookies + 4 - cookies_monday - cookies_tuesday) :
  cookies_wednesday / cookies_tuesday = 3 :=
by
  sorry

end cookie_ratio_l367_36739


namespace complex_number_quadrant_l367_36724

theorem complex_number_quadrant :
  let z := (2 * Complex.I) / (1 - Complex.I)
  Complex.re z < 0 ∧ Complex.im z > 0 :=
by
  sorry

end complex_number_quadrant_l367_36724


namespace setD_is_empty_l367_36726

-- Definitions of sets A, B, C, D
def setA : Set ℝ := {x | x + 3 = 3}
def setB : Set (ℝ × ℝ) := {(x, y) | y^2 ≠ -x^2}
def setC : Set ℝ := {x | x^2 ≤ 0}
def setD : Set ℝ := {x | x^2 - x + 1 = 0}

-- Theorem stating that set D is the empty set
theorem setD_is_empty : setD = ∅ := 
by 
  sorry

end setD_is_empty_l367_36726


namespace stationary_train_length_l367_36713

-- Definitions
def speed_km_per_h := 72
def speed_m_per_s := speed_km_per_h * (1000 / 3600) -- conversion from km/h to m/s
def time_to_pass_pole := 10 -- in seconds
def time_to_cross_stationary_train := 35 -- in seconds
def speed := 20 -- speed in m/s, 72 km/h = 20 m/s, can be inferred from conversion

-- Length of moving train
def length_of_moving_train := speed * time_to_pass_pole

-- Total distance in crossing stationary train
def total_distance := speed * time_to_cross_stationary_train

-- Length of stationary train
def length_of_stationary_train := total_distance - length_of_moving_train

-- Proof statement
theorem stationary_train_length :
  length_of_stationary_train = 500 := by
  sorry

end stationary_train_length_l367_36713


namespace volume_of_reservoir_proof_relationship_Q_t_proof_min_hourly_drainage_proof_min_time_to_drain_proof_l367_36701

noncomputable def volume_of_reservoir (drain_rate : ℝ) (time_to_drain : ℝ) : ℝ :=
  drain_rate * time_to_drain

theorem volume_of_reservoir_proof :
  volume_of_reservoir 8 6 = 48 :=
by
  sorry

noncomputable def relationship_Q_t (volume : ℝ) (t : ℝ) : ℝ :=
  volume / t

theorem relationship_Q_t_proof :
  ∀ (t : ℝ), relationship_Q_t 48 t = 48 / t :=
by
  intro t
  sorry

noncomputable def min_hourly_drainage (volume : ℝ) (time : ℝ) : ℝ :=
  volume / time

theorem min_hourly_drainage_proof :
  min_hourly_drainage 48 5 = 9.6 :=
by
  sorry

theorem min_time_to_drain_proof :
  ∀ (max_capacity : ℝ), relationship_Q_t 48 max_capacity = 12 → 48 / 12 = 4 :=
by
  intro max_capacity h
  sorry

end volume_of_reservoir_proof_relationship_Q_t_proof_min_hourly_drainage_proof_min_time_to_drain_proof_l367_36701


namespace product_of_a_l367_36798

theorem product_of_a : 
  (∃ a b : ℝ, (3 * a - 5)^2 + (a - 5 - (-2))^2 = (3 * Real.sqrt 13)^2 ∧ 
    (a * b = -8.32)) :=
by 
  sorry

end product_of_a_l367_36798


namespace pyramid_base_length_of_tangent_hemisphere_l367_36782

noncomputable def pyramid_base_side_length (radius height : ℝ) (tangent : ℝ → ℝ → Prop) : ℝ := sorry

theorem pyramid_base_length_of_tangent_hemisphere 
(r h : ℝ) (tangent : ℝ → ℝ → Prop) (tangent_property : ∀ x y, tangent x y → y = 0) 
(h_radius : r = 3) (h_height : h = 9) 
(tangent_conditions : tangent r h → tangent r h) : 
  pyramid_base_side_length r h tangent = 9 :=
sorry

end pyramid_base_length_of_tangent_hemisphere_l367_36782


namespace find_ab_l367_36778

theorem find_ab (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 30) : a * b = 32 :=
by
  -- We will complete the proof in this space
  sorry

end find_ab_l367_36778


namespace flowers_are_55_percent_daisies_l367_36714

noncomputable def percent_daisies (F : ℝ) (yellow : ℝ) (white_daisies : ℝ) (yellow_daisies : ℝ) : ℝ :=
  (yellow_daisies + white_daisies) / F * 100

theorem flowers_are_55_percent_daisies (F : ℝ) (yellow_t : ℝ) (yellow_d : ℝ) (white : ℝ) (white_d : ℝ) :
    yellow_t = 0.5 * yellow →
    yellow_d = yellow - yellow_t →
    white_d = (2 / 3) * white →
    yellow = (7 / 10) * F →
    white = F - yellow →
    percent_daisies F yellow white_d yellow_d = 55 :=
by
  sorry

end flowers_are_55_percent_daisies_l367_36714


namespace inequality_solution_l367_36717

theorem inequality_solution (a b : ℝ) :
  (∀ x : ℝ, (-1/2 < x ∧ x < 2) → (ax^2 + bx + 2 > 0)) →
  a + b = 1 :=
by
  sorry

end inequality_solution_l367_36717


namespace total_time_equiv_l367_36735

-- Define the number of chairs
def chairs := 7

-- Define the number of tables
def tables := 3

-- Define the time spent on each piece of furniture in minutes
def time_per_piece := 4

-- Prove the total time taken to assemble all furniture
theorem total_time_equiv : chairs + tables = 10 ∧ 4 * 10 = 40 := by
  sorry

end total_time_equiv_l367_36735


namespace AdultsNotWearingBlue_l367_36780

theorem AdultsNotWearingBlue (number_of_children : ℕ) (number_of_adults : ℕ) (adults_who_wore_blue : ℕ) :
  number_of_children = 45 → 
  number_of_adults = number_of_children / 3 → 
  adults_who_wore_blue = number_of_adults / 3 → 
  number_of_adults - adults_who_wore_blue = 10 :=
by
  sorry

end AdultsNotWearingBlue_l367_36780


namespace value_of_f3_f10_l367_36794

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom periodic_condition (x : ℝ) : f (x + 4) = f x + f 2
axiom f_at_one : f 1 = 4

theorem value_of_f3_f10 : f 3 + f 10 = 4 := sorry

end value_of_f3_f10_l367_36794


namespace problem_thre_is_15_and_10_percent_l367_36761

theorem problem_thre_is_15_and_10_percent (x y : ℝ) 
  (h1 : 3 = 0.15 * x) 
  (h2 : 3 = 0.10 * y) : 
  x - y = -10 := 
by 
  sorry

end problem_thre_is_15_and_10_percent_l367_36761


namespace remainder_7325_mod_11_l367_36738

theorem remainder_7325_mod_11 : 7325 % 11 = 6 := sorry

end remainder_7325_mod_11_l367_36738


namespace lengths_available_total_cost_l367_36770

def available_lengths := [1, 2, 3, 4, 5, 6]
def pipe_prices := [10, 15, 20, 25, 30, 35]

-- Given conditions
def purchased_pipes := [2, 5]
def target_perimeter_is_even := True

-- Prove: 
theorem lengths_available (x : ℕ) (hx : x ∈ available_lengths) : 
  3 < x ∧ x < 7 → x = 4 ∨ x = 5 ∨ x = 6 := by
  sorry

-- Prove: 
theorem total_cost (p : ℕ) (h : target_perimeter_is_even) : 
  p = 75 := by
  sorry

end lengths_available_total_cost_l367_36770


namespace shifted_function_correct_l367_36763

variable (x : ℝ)

/-- The original function -/
def original_function : ℝ := 3 * x - 4

/-- The function after shifting up by 2 units -/
def shifted_function : ℝ := original_function x + 2

theorem shifted_function_correct :
  shifted_function x = 3 * x - 2 :=
by
  sorry

end shifted_function_correct_l367_36763


namespace eight_div_repeating_three_l367_36747

theorem eight_div_repeating_three : (8 / (1 / 3)) = 24 := by
  sorry

end eight_div_repeating_three_l367_36747


namespace maximum_M_value_l367_36729

noncomputable def max_value_of_M : ℝ :=
  Real.sqrt 2 + 1 

theorem maximum_M_value {x y z : ℝ} (hx : 0 ≤ x) (hx1 : x ≤ 1) (hy : 0 ≤ y) (hy1 : y ≤ 1) (hz : 0 ≤ z) (hz1 : z ≤ 1) :
  Real.sqrt (abs (x - y)) + Real.sqrt (abs (y - z)) + Real.sqrt (abs (z - x)) ≤ max_value_of_M :=
by
  sorry

end maximum_M_value_l367_36729


namespace regular_polygon_sides_l367_36703

theorem regular_polygon_sides (n : ℕ) (h : (n - 2) * 180 = 144 * n) : n = 10 := 
by 
  sorry

end regular_polygon_sides_l367_36703


namespace point_P_in_first_quadrant_l367_36766

def pointInFirstQuadrant (x y : Int) : Prop := x > 0 ∧ y > 0

theorem point_P_in_first_quadrant : pointInFirstQuadrant 2 3 :=
by
  sorry

end point_P_in_first_quadrant_l367_36766


namespace mason_car_nuts_l367_36727

def busy_squirrels_num := 2
def busy_squirrel_nuts_per_day := 30
def sleepy_squirrel_num := 1
def sleepy_squirrel_nuts_per_day := 20
def days := 40

theorem mason_car_nuts : 
  busy_squirrels_num * busy_squirrel_nuts_per_day * days + sleepy_squirrel_nuts_per_day * days = 3200 :=
  by
    sorry

end mason_car_nuts_l367_36727


namespace minimum_value_of_a_l367_36746

def is_prime (n : ℕ) : Prop := sorry  -- Provide the definition of a prime number

def is_perfect_square (n : ℕ) : Prop := sorry  -- Provide the definition of a perfect square

theorem minimum_value_of_a 
  (a b : ℕ) 
  (h1 : is_prime (a - b)) 
  (h2 : is_perfect_square (a * b)) 
  (h3 : a ≥ 2012) : 
  a = 2025 := 
sorry

end minimum_value_of_a_l367_36746


namespace line_tangent_to_circle_l367_36762

theorem line_tangent_to_circle (x y : ℝ) :
  (3 * x - 4 * y + 25 = 0) ∧ (x^2 + y^2 = 25) → (x = -3 ∧ y = 4) :=
by sorry

end line_tangent_to_circle_l367_36762


namespace third_measurement_multiple_of_one_l367_36756

-- Define the lengths in meters
def length1_meter : ℕ := 6
def length2_meter : ℕ := 5

-- Convert lengths to centimeters
def length1_cm := length1_meter * 100
def length2_cm := length2_meter * 100

-- Define that the greatest common divisor (gcd) of lengths in cm is 100 cm
def gcd_length : ℕ := Nat.gcd length1_cm length2_cm

-- Given that the gcd is 100 cm
theorem third_measurement_multiple_of_one
  (h1 : gcd_length = 100) :
  ∃ n : ℕ, n = 1 :=
sorry

end third_measurement_multiple_of_one_l367_36756


namespace focus_of_parabola_y_eq_x_sq_l367_36765

theorem focus_of_parabola_y_eq_x_sq : ∃ (f : ℝ × ℝ), f = (0, 1/4) ∧ (∃ (p : ℝ), p = 1/2 ∧ ∀ x, y = x^2 → y = 2 * p * (0, y).snd) :=
by
  sorry

end focus_of_parabola_y_eq_x_sq_l367_36765


namespace maximum_profit_at_110_l367_36720

noncomputable def profit (x : ℕ) : ℝ := 
if x > 0 ∧ x < 100 then 
  -0.5 * (x : ℝ)^2 + 90 * (x : ℝ) - 600 
else if x ≥ 100 then 
  -2 * (x : ℝ) - 24200 / (x : ℝ) + 4100 
else 
  0 -- To ensure totality, although this won't match the problem's condition that x is always positive

theorem maximum_profit_at_110 :
  ∃ (y_max : ℝ), ∀ (x : ℕ), profit 110 = y_max ∧ (∀ x ≠ 0, profit 110 ≥ profit x) :=
sorry

end maximum_profit_at_110_l367_36720


namespace expected_value_correct_l367_36743

-- Define the probability distribution of the user's score in the first round
noncomputable def first_round_prob (X : ℕ) : ℚ :=
  if X = 3 then 1 / 4
  else if X = 2 then 1 / 2
  else if X = 1 then 1 / 4
  else 0

-- Define the conditional probability of the user's score in the second round given the first round score
noncomputable def second_round_prob (X Y : ℕ) : ℚ :=
  if X = 3 then
    if Y = 2 then 1 / 5
    else if Y = 1 then 4 / 5
    else 0
  else
    if Y = 2 then 1 / 3
    else if Y = 1 then 2 / 3
    else 0

-- Define the total score probability
noncomputable def total_score_prob (X Y : ℕ) : ℚ :=
  first_round_prob X * second_round_prob X Y

-- Compute the expected value of the user's total score
noncomputable def expected_value : ℚ :=
  (5 * (total_score_prob 3 2) +
   4 * (total_score_prob 3 1 + total_score_prob 2 2) +
   3 * (total_score_prob 2 1 + total_score_prob 1 2) +
   2 * (total_score_prob 1 1))

-- The theorem to be proven
theorem expected_value_correct : expected_value = 3.3 := 
by sorry

end expected_value_correct_l367_36743


namespace set_D_is_empty_l367_36789

theorem set_D_is_empty :
  {x : ℝ | x^2 + 2 = 0} = ∅ :=
by {
  sorry
}

end set_D_is_empty_l367_36789


namespace alcohol_percentage_l367_36702

theorem alcohol_percentage (P : ℝ) : 
  (0.10 * 300) + (P / 100 * 450) = 0.22 * 750 → P = 30 :=
by
  intros h
  sorry

end alcohol_percentage_l367_36702


namespace polygon_sides_l367_36773

theorem polygon_sides (x : ℕ) 
  (h1 : 180 * (x - 2) = 3 * 360) 
  : x = 8 := 
by
  sorry

end polygon_sides_l367_36773


namespace largest_five_digit_palindromic_number_l367_36733

def is_five_digit_palindrome (n : ℕ) : Prop := n / 10000 = n % 10 ∧ (n / 1000) % 10 = (n / 10) % 10

def is_four_digit_palindrome (n : ℕ) : Prop := n / 1000 = n % 10 ∧ (n / 100) % 10 = (n / 10) % 10

theorem largest_five_digit_palindromic_number :
  ∃ (abcba deed : ℕ), is_five_digit_palindrome abcba ∧ 10000 ≤ abcba ∧ abcba < 100000 ∧ is_four_digit_palindrome deed ∧ 1000 ≤ deed ∧ deed < 10000 ∧ abcba = 45 * deed ∧ abcba = 59895 :=
by
  sorry

end largest_five_digit_palindromic_number_l367_36733


namespace fencing_cost_l367_36723

noncomputable def diameter : ℝ := 14
noncomputable def cost_per_meter : ℝ := 2.50
noncomputable def pi := Real.pi

noncomputable def circumference (d : ℝ) : ℝ := pi * d

noncomputable def total_cost (c : ℝ) (r : ℝ) : ℝ := r * c

theorem fencing_cost : total_cost (circumference diameter) cost_per_meter = 109.95 := by
  sorry

end fencing_cost_l367_36723


namespace price_of_table_l367_36742

-- Given the conditions:
def chair_table_eq1 (C T : ℝ) : Prop := 2 * C + T = 0.6 * (C + 2 * T)
def chair_table_eq2 (C T : ℝ) : Prop := C + T = 72

-- Prove that the price of one table is $63
theorem price_of_table (C T : ℝ) (h1 : chair_table_eq1 C T) (h2 : chair_table_eq2 C T) : T = 63 := by
  sorry

end price_of_table_l367_36742


namespace find_x_l367_36760

def diamond (a b : ℝ) : ℝ := 3 * a * b - a + b

theorem find_x : ∃ x : ℝ, diamond 3 x = 24 ∧ x = 2.7 :=
by
  sorry

end find_x_l367_36760


namespace eight_girls_circle_least_distance_l367_36786

theorem eight_girls_circle_least_distance :
  let r := 50
  let num_girls := 8
  let total_distance := (8 * (3 * (r * Real.sqrt 2) + 2 * (2 * r)))
  total_distance = 1200 * Real.sqrt 2 + 1600 :=
by
  sorry

end eight_girls_circle_least_distance_l367_36786


namespace sqrt_1708249_eq_1307_l367_36788

theorem sqrt_1708249_eq_1307 :
  ∃ (n : ℕ), n * n = 1708249 ∧ n = 1307 :=
sorry

end sqrt_1708249_eq_1307_l367_36788


namespace find_F_l367_36749

theorem find_F (C : ℝ) (F : ℝ) (h₁ : C = 35) (h₂ : C = 4 / 7 * (F - 40)) : F = 101.25 := by
  sorry

end find_F_l367_36749


namespace f_of_f_3_eq_3_l367_36705

noncomputable def f (x : ℝ) : ℝ :=
if x < 2 then 1 - Real.logb 2 (2 - x) else 2^(1 - x) + 3 / 2

theorem f_of_f_3_eq_3 : f (f 3) = 3 := by
  sorry

end f_of_f_3_eq_3_l367_36705


namespace parabola_constant_unique_l367_36737

theorem parabola_constant_unique (b c : ℝ) :
  (∀ x y : ℝ, (x = 2 ∧ y = 20) → y = x^2 + b * x + c) →
  (∀ x y : ℝ, (x = -2 ∧ y = -4) → y = x^2 + b * x + c) →
  c = 4 :=
by
    sorry

end parabola_constant_unique_l367_36737


namespace expression_range_l367_36745

theorem expression_range (a b c d : ℝ) 
    (ha : 0 ≤ a) (ha' : a ≤ 2)
    (hb : 0 ≤ b) (hb' : b ≤ 2)
    (hc : 0 ≤ c) (hc' : c ≤ 2)
    (hd : 0 ≤ d) (hd' : d ≤ 2) :
  4 + 2 * Real.sqrt 2 ≤ 
    Real.sqrt (a^2 + (2-b)^2) 
    + Real.sqrt (b^2 + (2-c)^2) 
    + Real.sqrt (c^2 + (2-d)^2) 
    + Real.sqrt (d^2 + (2-a)^2) 
  ∧ Real.sqrt (a^2 + (2-b)^2) 
    + Real.sqrt (b^2 + (2-c)^2) 
    + Real.sqrt (c^2 + (2-d)^2) 
    + Real.sqrt (d^2 + (2-a)^2) ≤ 8 := 
sorry

end expression_range_l367_36745


namespace area_hexagon_STUVWX_l367_36734

noncomputable def area_of_hexagon (area_PQR : ℕ) (small_area : ℕ) : ℕ := 
  area_PQR - (3 * small_area)

theorem area_hexagon_STUVWX : 
  let area_PQR := 45
  let small_area := 1 
  ∃ area_hexagon, area_hexagon = 42 := 
by
  let area_PQR := 45
  let small_area := 1
  let area_hexagon := area_of_hexagon area_PQR small_area
  use area_hexagon
  sorry

end area_hexagon_STUVWX_l367_36734


namespace pages_left_to_read_l367_36757

def total_pages : ℕ := 17
def pages_read : ℕ := 11

theorem pages_left_to_read : total_pages - pages_read = 6 := by
  sorry

end pages_left_to_read_l367_36757


namespace picnic_problem_l367_36790

theorem picnic_problem
  (M W C A : ℕ)
  (h1 : M + W + C = 240)
  (h2 : M = W + 80)
  (h3 : A = C + 80)
  (h4 : A = M + W) :
  M = 120 :=
by
  sorry

end picnic_problem_l367_36790


namespace Norbs_age_l367_36795

def guesses : List ℕ := [24, 28, 30, 32, 36, 38, 41, 44, 47, 49]

def is_prime (n : ℕ) : Prop := Nat.Prime n

def two_off_by_one (n : ℕ) (guesses : List ℕ) : Prop := 
  (n - 1 ∈ guesses) ∧ (n + 1 ∈ guesses)

def at_least_half_too_low (n : ℕ) (guesses : List ℕ) : Prop := 
  (guesses.filter (· < n)).length ≥ guesses.length / 2

theorem Norbs_age : 
  ∃ x, is_prime x ∧ two_off_by_one x guesses ∧ at_least_half_too_low x guesses ∧ x = 37 := 
by 
  sorry

end Norbs_age_l367_36795
