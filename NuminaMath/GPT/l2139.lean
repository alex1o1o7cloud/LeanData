import Mathlib

namespace scientific_notation_of_8450_l2139_213959

theorem scientific_notation_of_8450 :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ (8450 : ℝ) = a * 10^n ∧ (a = 8.45) ∧ (n = 3) :=
sorry

end scientific_notation_of_8450_l2139_213959


namespace decimal_representation_prime_has_zeros_l2139_213918

theorem decimal_representation_prime_has_zeros (p : ℕ) [Fact (Nat.Prime p)] : 
  ∃ n : ℕ, n > 0 ∧ ∃ k : ℕ, 10^2002 ∣ p^n * 10^k :=
sorry

end decimal_representation_prime_has_zeros_l2139_213918


namespace plumber_salary_percentage_l2139_213912

def salary_construction_worker : ℕ := 100
def salary_electrician : ℕ := 2 * salary_construction_worker
def total_salary_without_plumber : ℕ := 2 * salary_construction_worker + salary_electrician
def total_labor_cost : ℕ := 650
def salary_plumber : ℕ := total_labor_cost - total_salary_without_plumber
def percentage_salary_plumber_as_construction_worker (x y : ℕ) : ℕ := (x * 100) / y

theorem plumber_salary_percentage :
  percentage_salary_plumber_as_construction_worker salary_plumber salary_construction_worker = 250 :=
by 
  sorry

end plumber_salary_percentage_l2139_213912


namespace triangle_angles_and_type_l2139_213945

theorem triangle_angles_and_type
  (largest_angle : ℝ)
  (smallest_angle : ℝ)
  (middle_angle : ℝ)
  (h1 : largest_angle = 90)
  (h2 : largest_angle = 3 * smallest_angle)
  (h3 : largest_angle + smallest_angle + middle_angle = 180) :
  (largest_angle = 90 ∧ middle_angle = 60 ∧ smallest_angle = 30 ∧ largest_angle = 90) := by
  sorry

end triangle_angles_and_type_l2139_213945


namespace proof_problem_l2139_213961

noncomputable def sqrt_repeated (x : ℕ) (y : ℕ) : ℕ :=
Nat.sqrt x ^ y

theorem proof_problem (x y z : ℕ) :
  (sqrt_repeated x y = z) ↔ 
  ((∃ t : ℕ, x = t^2 ∧ y = 1 ∧ z = t) ∨ (x = 0 ∧ z = 0 ∧ y ≠ 0)) :=
sorry

end proof_problem_l2139_213961


namespace exists_segment_with_points_l2139_213943

theorem exists_segment_with_points (S : Finset ℕ) (n : ℕ) (hS : S.card = 6 * n)
  (hB : ∃ B : Finset ℕ, B ⊆ S ∧ B.card = 4 * n) (hG : ∃ G : Finset ℕ, G ⊆ S ∧ G.card = 2 * n) :
  ∃ t : Finset ℕ, t ⊆ S ∧ t.card = 3 * n ∧ (∃ B' : Finset ℕ, B' ⊆ t ∧ B'.card = 2 * n) ∧ (∃ G' : Finset ℕ, G' ⊆ t ∧ G'.card = n) :=
  sorry

end exists_segment_with_points_l2139_213943


namespace problem_solution_count_l2139_213928

theorem problem_solution_count (n : ℕ) (h1 : (80 * n) ^ 40 > n ^ 80) (h2 : n ^ 80 > 3 ^ 160) : 
  ∃ s : Finset ℕ, s.card = 70 ∧ ∀ x ∈ s, 10 ≤ x ∧ x ≤ 79 :=
by
  sorry

end problem_solution_count_l2139_213928


namespace inequality_transformation_l2139_213954

variable {x y : ℝ}

theorem inequality_transformation (h : x > y) : x + 5 > y + 5 :=
by
  sorry

end inequality_transformation_l2139_213954


namespace age_of_B_l2139_213929

theorem age_of_B (A B : ℕ) (h1 : A + 10 = 2 * (B - 10)) (h2 : A = B + 11) : B = 41 :=
by
  -- Proof not required as per instructions
  sorry

end age_of_B_l2139_213929


namespace sin_arithmetic_sequence_l2139_213947

noncomputable def sin_value (a : ℝ) := Real.sin (a * (Real.pi / 180))

theorem sin_arithmetic_sequence (a : ℝ) : 
  (0 < a) ∧ (a < 360) ∧ (sin_value a + sin_value (3 * a) = 2 * sin_value (2 * a)) ↔ a = 90 ∨ a = 270 :=
by 
  sorry

end sin_arithmetic_sequence_l2139_213947


namespace alice_book_payment_l2139_213926

/--
Alice is in the UK and wants to purchase a book priced at £25.
If one U.S. dollar is equivalent to £0.75, 
then Alice needs to pay 33.33 USD for the book.
-/
theorem alice_book_payment :
  ∀ (price_gbp : ℝ) (conversion_rate : ℝ), 
  price_gbp = 25 → conversion_rate = 0.75 → 
  (price_gbp / conversion_rate) = 33.33 :=
by
  intros price_gbp conversion_rate hprice hrate
  rw [hprice, hrate]
  sorry

end alice_book_payment_l2139_213926


namespace garden_area_l2139_213905

-- Given that the garden is a square with certain properties
variables (s A P : ℕ)

-- Conditions:
-- The perimeter of the square garden is 28 feet
def perimeter_condition : Prop := P = 28

-- The area of the garden is equal to the perimeter plus 21
def area_condition : Prop := A = P + 21

-- The perimeter of a square garden with side length s
def perimeter_def : Prop := P = 4 * s

-- The area of a square garden with side length s
def area_def : Prop := A = s * s

-- Prove that the area A is 49 square feet
theorem garden_area : perimeter_condition P → area_condition P A → perimeter_def s P → area_def s A → A = 49 :=
by 
  sorry

end garden_area_l2139_213905


namespace green_apples_count_l2139_213901

-- Definitions for the conditions
def total_apples : ℕ := 9
def red_apples : ℕ := 7

-- Theorem stating the number of green apples
theorem green_apples_count : total_apples - red_apples = 2 := by
  sorry

end green_apples_count_l2139_213901


namespace dentist_ratio_l2139_213924

-- Conditions
def cost_cleaning : ℕ := 70
def cost_filling : ℕ := 120
def cost_extraction : ℕ := 290

-- Theorem statement
theorem dentist_ratio : (cost_cleaning + 2 * cost_filling + cost_extraction) / cost_filling = 5 := 
by
  -- To be proven
  sorry

end dentist_ratio_l2139_213924


namespace express_x_in_terms_of_y_l2139_213966

variable {x y : ℝ}

theorem express_x_in_terms_of_y (h : 3 * x - 4 * y = 6) : x = (6 + 4 * y) / 3 := 
sorry

end express_x_in_terms_of_y_l2139_213966


namespace cost_price_of_pots_l2139_213995

variable (C : ℝ)

-- Define the conditions
def selling_price (C : ℝ) := 1.25 * C
def total_revenue (selling_price : ℝ) := 150 * selling_price

-- State the main proof goal
theorem cost_price_of_pots (h : total_revenue (selling_price C) = 450) : C = 2.4 := by
  sorry

end cost_price_of_pots_l2139_213995


namespace inequality_solution_l2139_213930

theorem inequality_solution (a : ℝ) (h : 1 < a) : ∀ x : ℝ, a ^ (2 * x + 1) > (1 / a) ^ (2 * x) ↔ x > -1 / 4 :=
by
  sorry

end inequality_solution_l2139_213930


namespace calculate_expression_l2139_213908

theorem calculate_expression : 
  (1 - Real.sqrt 2)^0 + |(2 - Real.sqrt 5)| + (-1)^2022 - (1/3) * Real.sqrt 45 = 0 :=
by
  sorry

end calculate_expression_l2139_213908


namespace distance_from_C_to_B_is_80_l2139_213936

theorem distance_from_C_to_B_is_80
  (x : ℕ)
  (h1 : x = 60)
  (h2 : ∀ (ab cb : ℕ), ab = x → cb = x + 20  → (cb = 80))
  : x + 20 = 80 := by
  sorry

end distance_from_C_to_B_is_80_l2139_213936


namespace Arthur_total_distance_l2139_213949

/-- Arthur walks 8 blocks south and then 10 blocks west. Each block is one-fourth of a mile.
How many miles did Arthur walk in total? -/
theorem Arthur_total_distance (blocks_south : ℕ) (blocks_west : ℕ) (block_length_miles : ℝ) :
  blocks_south = 8 ∧ blocks_west = 10 ∧ block_length_miles = 1/4 →
  (blocks_south + blocks_west) * block_length_miles = 4.5 :=
by
  intro h
  have h1 : blocks_south = 8 := h.1
  have h2 : blocks_west = 10 := h.2.1
  have h3 : block_length_miles = 1 / 4 := h.2.2
  sorry

end Arthur_total_distance_l2139_213949


namespace derivative_at_one_l2139_213960

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.log x

theorem derivative_at_one : (deriv f 1) = 4 := by
  sorry

end derivative_at_one_l2139_213960


namespace dividend_calculation_l2139_213987

theorem dividend_calculation (D : ℝ) (Q : ℝ) (R : ℝ) (Dividend : ℝ) (h1 : D = 47.5) (h2 : Q = 24.3) (h3 : R = 32.4)  :
  Dividend = D * Q + R := by
  rw [h1, h2, h3]
  sorry -- This skips the actual computation proof

end dividend_calculation_l2139_213987


namespace find_y_value_l2139_213927

theorem find_y_value
  (y z : ℝ)
  (h1 : y + z + 175 = 360)
  (h2 : z = y + 10) :
  y = 88 :=
by
  sorry

end find_y_value_l2139_213927


namespace ads_not_blocked_not_interesting_l2139_213942

theorem ads_not_blocked_not_interesting:
  (let A_blocks := 0.75
   let B_blocks := 0.85
   let C_blocks := 0.95
   let A_let_through := 1 - A_blocks
   let B_let_through := 1 - B_blocks
   let C_let_through := 1 - C_blocks
   let all_let_through := A_let_through * B_let_through * C_let_through
   let interesting := 0.15
   let not_interesting := 1 - interesting
   (all_let_through * not_interesting) = 0.00159375) :=
  sorry

end ads_not_blocked_not_interesting_l2139_213942


namespace complex_fraction_equivalence_l2139_213907

/-- The complex number 2 / (1 - i) is equal to 1 + i. -/
theorem complex_fraction_equivalence : (2 : ℂ) / (1 - (I : ℂ)) = 1 + (I : ℂ) := by
  sorry

end complex_fraction_equivalence_l2139_213907


namespace smallest_four_digit_int_equiv_8_mod_9_l2139_213994

theorem smallest_four_digit_int_equiv_8_mod_9 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 9 = 8 ∧ n = 1007 := 
by
  sorry

end smallest_four_digit_int_equiv_8_mod_9_l2139_213994


namespace total_ticket_cost_l2139_213951

theorem total_ticket_cost (V G : ℕ) 
  (h1 : V + G = 320) 
  (h2 : V = G - 276) 
  (price_vip : ℕ := 45) 
  (price_regular : ℕ := 20) : 
  (price_vip * V + price_regular * G = 6950) :=
by sorry

end total_ticket_cost_l2139_213951


namespace flour_more_than_salt_l2139_213962

open Function

-- Definitions based on conditions
def flour_needed : ℕ := 12
def flour_added : ℕ := 2
def salt_needed : ℕ := 7
def salt_added : ℕ := 0

-- Given that these definitions hold, prove the following theorem
theorem flour_more_than_salt : (flour_needed - flour_added) - (salt_needed - salt_added) = 3 :=
by
  -- Here you would include the proof, but as instructed, we skip it with "sorry".
  sorry

end flour_more_than_salt_l2139_213962


namespace additional_fee_per_minute_for_second_plan_l2139_213986

theorem additional_fee_per_minute_for_second_plan :
  (∃ x : ℝ, (22 + 0.13 * 280 = 8 + x * 280) ∧ x = 0.18) :=
sorry

end additional_fee_per_minute_for_second_plan_l2139_213986


namespace beetles_consumed_per_day_l2139_213957

-- Definitions
def bird_eats_beetles (n : Nat) : Nat := 12 * n
def snake_eats_birds (n : Nat) : Nat := 3 * n
def jaguar_eats_snakes (n : Nat) : Nat := 5 * n
def crocodile_eats_jaguars (n : Nat) : Nat := 2 * n

-- Initial values
def initial_jaguars : Nat := 6
def initial_crocodiles : Nat := 30
def net_increase_birds : Nat := 4
def net_increase_snakes : Nat := 2
def net_increase_jaguars : Nat := 1

-- Proof statement
theorem beetles_consumed_per_day : 
  bird_eats_beetles (snake_eats_birds (jaguar_eats_snakes initial_jaguars)) = 1080 := 
by 
  sorry

end beetles_consumed_per_day_l2139_213957


namespace expressions_equal_l2139_213917

theorem expressions_equal {x y z : ℤ} : (x + 2 * y * z = (x + y) * (x + 2 * z)) ↔ (x + y + 2 * z = 1) :=
by
  sorry

end expressions_equal_l2139_213917


namespace marcella_shoes_lost_l2139_213937

theorem marcella_shoes_lost (pairs_initial : ℕ) (pairs_left_max : ℕ) (individuals_initial : ℕ) (individuals_left_max : ℕ) (pairs_initial_eq : pairs_initial = 25) (pairs_left_max_eq : pairs_left_max = 20) (individuals_initial_eq : individuals_initial = pairs_initial * 2) (individuals_left_max_eq : individuals_left_max = pairs_left_max * 2) : (individuals_initial - individuals_left_max) = 10 := 
by
  sorry

end marcella_shoes_lost_l2139_213937


namespace inequality_proof_l2139_213992

theorem inequality_proof
  (a1 a2 a3 b1 b2 b3 : ℝ)
  (ha1 : 0 < a1) (ha2 : 0 < a2) (ha3 : 0 < a3)
  (hb1 : 0 < b1) (hb2 : 0 < b2) (hb3 : 0 < b3) :
  (a1 * b2 + a2 * b1 + a2 * b3 + a3 * b2 + a3 * b1 + a1 * b3)^2 ≥
    4 * (a1 * a2 + a2 * a3 + a3 * a1) * (b1 * b2 + b2 * b3 + b3 * b1) :=
sorry

end inequality_proof_l2139_213992


namespace minimum_cost_peking_opera_l2139_213940

theorem minimum_cost_peking_opera (T p₆ p₁₀ : ℕ) (xₛ yₛ : ℕ) :
  T = 140 ∧ p₆ = 6 ∧ p₁₀ = 10 ∧ xₛ + yₛ = T ∧ yₛ ≥ 2 * xₛ →
  6 * xₛ + 10 * yₛ = 1216 ∧ xₛ = 46 ∧ yₛ = 94 :=
by
   -- Proving this is skipped (left as a sorry)
  sorry

end minimum_cost_peking_opera_l2139_213940


namespace total_nails_needed_l2139_213902

-- Define the conditions
def nails_already_have : ℕ := 247
def nails_found : ℕ := 144
def nails_to_buy : ℕ := 109

-- The statement to prove
theorem total_nails_needed : nails_already_have + nails_found + nails_to_buy = 500 := by
  -- The proof goes here
  sorry

end total_nails_needed_l2139_213902


namespace min_rings_to_connect_all_segments_l2139_213934

-- Define the problem setup
structure ChainSegment where
  rings : Fin 3 → Type

-- Define the number of segments
def num_segments : ℕ := 5

-- Define the minimum number of rings to be opened and rejoined
def min_rings_to_connect (seg : Fin num_segments) : ℕ :=
  3

theorem min_rings_to_connect_all_segments :
  ∀ segs : Fin num_segments,
  (∃ n, n = min_rings_to_connect segs) :=
by
  -- Proof to be provided
  sorry

end min_rings_to_connect_all_segments_l2139_213934


namespace integral_cos_neg_one_l2139_213944

theorem integral_cos_neg_one: 
  ∫ x in (Set.Icc (Real.pi / 2) Real.pi), Real.cos x = -1 :=
by
  sorry

end integral_cos_neg_one_l2139_213944


namespace puppies_given_to_friends_l2139_213980

def original_puppies : ℕ := 8
def current_puppies : ℕ := 4

theorem puppies_given_to_friends : original_puppies - current_puppies = 4 :=
by
  sorry

end puppies_given_to_friends_l2139_213980


namespace perpendicular_lines_l2139_213989

theorem perpendicular_lines (m : ℝ) :
  (m+2)*(m-1) + m*(m-4) = 0 ↔ m = 2 ∨ m = -1/2 :=
by 
  sorry

end perpendicular_lines_l2139_213989


namespace volume_of_tetrahedron_equiv_l2139_213913

noncomputable def volume_tetrahedron (D1 D2 D3 : ℝ) 
  (h1 : D1 = 24) (h2 : D3 = 20) (h3 : D2 = 16) : ℝ :=
  30 * Real.sqrt 6

theorem volume_of_tetrahedron_equiv (D1 D2 D3 : ℝ) 
  (h1 : D1 = 24) (h2 : D3 = 20) (h3 : D2 = 16) :
  volume_tetrahedron D1 D2 D3 h1 h2 h3 = 30 * Real.sqrt 6 :=
  sorry

end volume_of_tetrahedron_equiv_l2139_213913


namespace parallelogram_circumference_l2139_213981

-- Define the lengths of the sides of the parallelogram.
def side1 : ℝ := 18
def side2 : ℝ := 12

-- Define the formula for the circumference (or perimeter) of the parallelogram.
def circumference (a b : ℝ) : ℝ :=
  2 * (a + b)

-- Statement of the proof problem:
theorem parallelogram_circumference : circumference side1 side2 = 60 := 
  by
    sorry

end parallelogram_circumference_l2139_213981


namespace point_between_lines_l2139_213922

theorem point_between_lines (b : ℝ) (h1 : 6 * 5 - 8 * b + 1 < 0) (h2 : 3 * 5 - 4 * b + 5 > 0) : b = 4 :=
  sorry

end point_between_lines_l2139_213922


namespace isosceles_triangle_side_l2139_213909

theorem isosceles_triangle_side (a : ℝ) : 
  (10 - a = 7 ∨ 10 - a = 6) ↔ (a = 3 ∨ a = 4) := 
by sorry

end isosceles_triangle_side_l2139_213909


namespace strip_covers_cube_l2139_213920

   -- Define the given conditions
   def strip_length := 12
   def strip_width := 1
   def cube_edge := 1
   def layers := 2

   -- Define the main statement to be proved
   theorem strip_covers_cube : 
     (strip_length >= 6 * cube_edge / layers) ∧ 
     (strip_width >= cube_edge) ∧ 
     (layers == 2) → 
     true :=
   by
     intro h
     sorry
   
end strip_covers_cube_l2139_213920


namespace michael_can_cover_both_classes_l2139_213973

open Nat

def total_students : ℕ := 30
def german_students : ℕ := 20
def japanese_students : ℕ := 24

-- Calculate the number of students taking both German and Japanese using inclusion-exclusion principle.
def both_students : ℕ := german_students + japanese_students - total_students

-- Calculate the number of students only taking German.
def only_german_students : ℕ := german_students - both_students

-- Calculate the number of students only taking Japanese.
def only_japanese_students : ℕ := japanese_students - both_students

-- Calculate the total number of ways to choose 2 students out of 30.
def total_ways_to_choose_2 : ℕ := (total_students * (total_students - 1)) / 2

-- Calculate the number of ways to choose 2 students only taking German or only taking Japanese.
def undesirable_outcomes : ℕ := (only_german_students * (only_german_students - 1)) / 2 + (only_japanese_students * (only_japanese_students - 1)) / 2

-- Calculate the probability of undesirable outcomes.
def undesirable_probability : ℚ := undesirable_outcomes / total_ways_to_choose_2

-- Calculate the probability Michael can cover both German and Japanese classes.
def desired_probability : ℚ := 1 - undesirable_probability

theorem michael_can_cover_both_classes : desired_probability = 25 / 29 := by sorry

end michael_can_cover_both_classes_l2139_213973


namespace Allyson_age_is_28_l2139_213958

-- Define the conditions of the problem
def Hiram_age : ℕ := 40
def add_12_to_Hiram_age (h_age : ℕ) : ℕ := h_age + 12
def twice_Allyson_age (a_age : ℕ) : ℕ := 2 * a_age
def condition (h_age : ℕ) (a_age : ℕ) : Prop := add_12_to_Hiram_age h_age = twice_Allyson_age a_age - 4

-- Define the theorem to be proven
theorem Allyson_age_is_28 (a_age : ℕ) (h_age : ℕ) (h_condition : condition h_age a_age) (h_hiram : h_age = Hiram_age) : a_age = 28 :=
by sorry

end Allyson_age_is_28_l2139_213958


namespace impossible_event_abs_lt_zero_l2139_213974

theorem impossible_event_abs_lt_zero (a : ℝ) : ¬ (|a| < 0) :=
sorry

end impossible_event_abs_lt_zero_l2139_213974


namespace FindAngleB_FindIncircleRadius_l2139_213968

-- Define the problem setting
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Condition 1: a + c = 2b * sin (C + π / 6)
def Condition1 (T : Triangle) : Prop :=
  T.a + T.c = 2 * T.b * Real.sin (T.C + Real.pi / 6)

-- Condition 2: (b + c) (sin B - sin C) = (a - c) sin A
def Condition2 (T : Triangle) : Prop :=
  (T.b + T.c) * (Real.sin T.B - Real.sin T.C) = (T.a - T.c) * Real.sin T.A

-- Condition 3: (2a - c) cos B = b cos C
def Condition3 (T : Triangle) : Prop :=
  (2 * T.a - T.c) * Real.cos T.B = T.b * Real.cos T.C

-- Given: radius of incircle and dot product of vectors condition
def Given (T : Triangle) (r : ℝ) : Prop :=
  (T.a + T.c = 4 * Real.sqrt 3) ∧
  (2 * T.b * (T.a * T.c * Real.cos T.B - 3 * Real.sqrt 3 / 2) = 6)

-- Proof of B = π / 3
theorem FindAngleB (T : Triangle) :
  (Condition1 T ∨ Condition2 T ∨ Condition3 T) → T.B = Real.pi / 3 := 
sorry

-- Proof for the radius of the incircle
theorem FindIncircleRadius (T : Triangle) (r : ℝ) :
  Given T r → T.B = Real.pi / 3 → r = 1 := 
sorry


end FindAngleB_FindIncircleRadius_l2139_213968


namespace ezekiel_first_day_distance_l2139_213938

noncomputable def distance_first_day (total_distance second_day_distance third_day_distance : ℕ) :=
  total_distance - (second_day_distance + third_day_distance)

theorem ezekiel_first_day_distance:
  ∀ (total_distance second_day_distance third_day_distance : ℕ),
  total_distance = 50 →
  second_day_distance = 25 →
  third_day_distance = 15 →
  distance_first_day total_distance second_day_distance third_day_distance = 10 :=
by
  intros total_distance second_day_distance third_day_distance h1 h2 h3
  sorry

end ezekiel_first_day_distance_l2139_213938


namespace cost_of_individual_roll_l2139_213969

theorem cost_of_individual_roll
  (p : ℕ) (c : ℝ) (s : ℝ) (x : ℝ)
  (hc : c = 9)
  (hp : p = 12)
  (hs : s = 0.25)
  (h : 12 * x = 9 * (1 + s)) :
  x = 0.9375 :=
by
  sorry

end cost_of_individual_roll_l2139_213969


namespace find_third_vertex_l2139_213983

open Real

-- Define the vertices of the triangle
def vertex1 : ℝ × ℝ := (9, 3)
def vertex2 : ℝ × ℝ := (0, 0)

-- Define the conditions
def on_negative_x_axis (p : ℝ × ℝ) : Prop :=
  p.2 = 0 ∧ p.1 < 0

def area_of_triangle (a b c : ℝ × ℝ) : ℝ :=
  0.5 * abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2))

-- Statement of the problem in Lean
theorem find_third_vertex :
  ∃ (vertex3 : ℝ × ℝ), 
    on_negative_x_axis vertex3 ∧ 
    area_of_triangle vertex1 vertex2 vertex3 = 45 ∧
    vertex3 = (-30, 0) :=
sorry

end find_third_vertex_l2139_213983


namespace ten_thousand_points_length_l2139_213925

theorem ten_thousand_points_length (a b : ℝ) (d : ℝ) 
  (h1 : d = a / 99) 
  (h2 : b = 9999 * d) : b = 101 * a := by
  sorry

end ten_thousand_points_length_l2139_213925


namespace find_price_per_package_l2139_213955

theorem find_price_per_package (P : ℝ) :
  (10 * P + 50 * (4/5 * P) = 1340) → (P = 26.80) := by
  intros h
  sorry

end find_price_per_package_l2139_213955


namespace nth_term_l2139_213988

theorem nth_term (b : ℕ → ℝ) (h₀ : b 1 = 1)
  (h_rec : ∀ n ≥ 1, (b (n + 1))^2 = 36 * (b n)^2) : 
  b 50 = 6^49 :=
by
  sorry

end nth_term_l2139_213988


namespace pre_image_of_f_l2139_213903

theorem pre_image_of_f (x y : ℝ) (f : ℝ × ℝ → ℝ × ℝ) 
  (h : f = λ p => (2 * p.1 + p.2, p.1 - 2 * p.2)) :
  f (1, 0) = (2, 1) := by
  sorry

end pre_image_of_f_l2139_213903


namespace tan_alpha_plus_pi_l2139_213977

-- Define the given conditions and prove the desired equality.
theorem tan_alpha_plus_pi 
  (α : ℝ) 
  (hα : 0 < α ∧ α < π) 
  (hcos : Real.cos (π - α) = 1 / 3) : 
  Real.tan (α + π) = -2 * Real.sqrt 2 :=
by
  sorry

end tan_alpha_plus_pi_l2139_213977


namespace proof_problem_l2139_213972

-- Conditions
def in_fourth_quadrant (α : ℝ) : Prop := (α > 3 * Real.pi / 2) ∧ (α < 2 * Real.pi)
def x_coordinate_unit_circle (α : ℝ) : Prop := Real.cos α = 1/3

-- Proof statement
theorem proof_problem (α : ℝ) (h1 : in_fourth_quadrant α) (h2 : x_coordinate_unit_circle α) :
  Real.tan α = -2 * Real.sqrt 2 ∧
  ((Real.sin α)^2 - Real.sqrt 2 * (Real.sin α) * (Real.cos α)) / (1 + (Real.cos α)^2) = 6 / 5 :=
by
  sorry

end proof_problem_l2139_213972


namespace unique_polynomial_l2139_213923

-- Define the conditions
def valid_polynomial (P : ℝ → ℝ) : Prop :=
  ∃ (p : Polynomial ℝ), Polynomial.degree p > 0 ∧ ∀ (z : ℝ), z ≠ 0 → P z = Polynomial.eval z p

-- The main theorem
theorem unique_polynomial (P : ℝ → ℝ) (hP : valid_polynomial P) :
  (∀ (z : ℝ), z ≠ 0 → P z ≠ 0 → P (1/z) ≠ 0 → 
  1 / P z + 1 / P (1 / z) = z + 1 / z) → ∀ x, P x = x :=
by
  sorry

end unique_polynomial_l2139_213923


namespace neon_signs_blink_together_l2139_213956

theorem neon_signs_blink_together :
  Nat.lcm (Nat.lcm (Nat.lcm 7 11) 13) 17 = 17017 :=
by
  sorry

end neon_signs_blink_together_l2139_213956


namespace ratio_father_to_children_after_5_years_l2139_213953

def father's_age := 15
def sum_children_ages := father's_age / 3

def father's_age_after_5_years := father's_age + 5
def sum_children_ages_after_5_years := sum_children_ages + 10

theorem ratio_father_to_children_after_5_years :
  father's_age_after_5_years / sum_children_ages_after_5_years = 4 / 3 := by
  sorry

end ratio_father_to_children_after_5_years_l2139_213953


namespace find_first_number_l2139_213906

def is_lcm (a b l : ℕ) : Prop := l = Nat.lcm a b

theorem find_first_number :
  ∃ (a b : ℕ), (5 * b) = a ∧ (4 * b) = b ∧ is_lcm a b 80 ∧ a = 20 :=
by
  sorry

end find_first_number_l2139_213906


namespace cycle_selling_price_l2139_213998

theorem cycle_selling_price 
  (CP : ℝ) (gain_percent : ℝ) (SP : ℝ) 
  (h1 : CP = 840) 
  (h2 : gain_percent = 45.23809523809524 / 100)
  (h3 : SP = CP * (1 + gain_percent)) :
  SP = 1220 :=
sorry

end cycle_selling_price_l2139_213998


namespace arithmetic_sequence_sum_l2139_213941

noncomputable def a_n (n : ℕ) : ℕ :=
  n

def b_n (n : ℕ) : ℕ :=
  n * 2^n

def S_n (n : ℕ) : ℕ :=
  (n - 1) * 2^(n + 1) + 2

theorem arithmetic_sequence_sum
  (a_n : ℕ → ℕ)
  (b_n : ℕ → ℕ)
  (S_n : ℕ → ℕ)
  (h1 : a_n 1 + a_n 2 + a_n 3 = 6)
  (h2 : a_n 5 = 5)
  (h3 : ∀ n, b_n n = a_n n * 2^(a_n n)) :
  (∀ n, a_n n = n) ∧ (∀ n, S_n n = (n - 1) * 2^(n + 1) + 2) :=
by
  sorry

end arithmetic_sequence_sum_l2139_213941


namespace revenue_function_correct_strategy_not_profitable_l2139_213979

-- Given conditions 
def purchase_price : ℝ := 1
def last_year_price : ℝ := 2
def last_year_sales_volume : ℕ := 10000
def last_year_revenue : ℝ := 20000
def proportionality_constant : ℝ := 4
def increased_sales_volume (x : ℝ) : ℝ := proportionality_constant * (2 - x) ^ 2

-- Questions translated to Lean statements
def revenue_this_year (x : ℝ) : ℝ := 4 * x ^ 3 - 20 * x ^ 2 + 33 * x - 17

theorem revenue_function_correct (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) : 
    revenue_this_year x = 4 * x ^ 3 - 20 * x ^ 2 + 33 * x - 17 :=
by
  sorry

theorem strategy_not_profitable (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) : 
    revenue_this_year x ≤ last_year_revenue :=
by
  sorry

end revenue_function_correct_strategy_not_profitable_l2139_213979


namespace distinct_four_digit_integers_with_digit_product_eight_l2139_213996

theorem distinct_four_digit_integers_with_digit_product_eight : 
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ (∀ (a b c d : ℕ), 10 > a ∧ 10 > b ∧ 10 > c ∧ 10 > d ∧ n = 1000 * a + 100 * b + 10 * c + d ∧ a * b * c * d = 8) ∧ (∃ (count : ℕ), count = 20 ) :=
sorry

end distinct_four_digit_integers_with_digit_product_eight_l2139_213996


namespace probability_greater_than_n_l2139_213935

theorem probability_greater_than_n (n : ℕ) : 
  (1 ≤ n ∧ n ≤ 5) → (∃ k, k = 6 - n - 1 ∧ k / 6 = 1 / 2) → n = 3 := 
by sorry

end probability_greater_than_n_l2139_213935


namespace find_central_angle_l2139_213911

noncomputable def sector := 
  {R : ℝ // R > 0}

noncomputable def central_angle (R : ℝ) : ℝ := 
  (6 - 2 * R) / R

theorem find_central_angle :
  ∃ α : ℝ, (α = 1 ∨ α = 4) ∧ 
  (∃ R : ℝ, 
    (2 * R + α * R = 6) ∧ 
    (1 / 2 * R^2 * α = 2)) := 
by {
  sorry
}

end find_central_angle_l2139_213911


namespace time_to_pass_faster_train_l2139_213910

noncomputable def speed_slower_train_kmph : ℝ := 36
noncomputable def speed_faster_train_kmph : ℝ := 45
noncomputable def length_faster_train_m : ℝ := 225.018
noncomputable def kmph_to_mps_factor : ℝ := 1000 / 3600

noncomputable def relative_speed_mps : ℝ := (speed_slower_train_kmph + speed_faster_train_kmph) * kmph_to_mps_factor

theorem time_to_pass_faster_train : 
  (length_faster_train_m / relative_speed_mps) = 10.001 := 
sorry

end time_to_pass_faster_train_l2139_213910


namespace total_lines_correct_l2139_213900

-- Define the shapes and their corresponding lines
def triangles := 12
def squares := 8
def pentagons := 4
def hexagons := 6
def octagons := 2

def triangle_sides := 3
def square_sides := 4
def pentagon_sides := 5
def hexagon_sides := 6
def octagon_sides := 8

def lines_in_triangles := triangles * triangle_sides
def lines_in_squares := squares * square_sides
def lines_in_pentagons := pentagons * pentagon_sides
def lines_in_hexagons := hexagons * hexagon_sides
def lines_in_octagons := octagons * octagon_sides

def shared_lines_ts := 5
def shared_lines_ph := 3
def shared_lines_ho := 1

def total_lines_triangles := lines_in_triangles - shared_lines_ts
def total_lines_squares := lines_in_squares - shared_lines_ts
def total_lines_pentagons := lines_in_pentagons - shared_lines_ph
def total_lines_hexagons := lines_in_hexagons - shared_lines_ph - shared_lines_ho
def total_lines_octagons := lines_in_octagons - shared_lines_ho

-- The statement to prove
theorem total_lines_correct :
  total_lines_triangles = 31 ∧
  total_lines_squares = 27 ∧
  total_lines_pentagons = 17 ∧
  total_lines_hexagons = 32 ∧
  total_lines_octagons = 15 :=
by sorry

end total_lines_correct_l2139_213900


namespace factor_poly1_factor_poly2_factor_poly3_l2139_213914

-- Define the three polynomial functions.
def poly1 (x : ℝ) : ℝ := 2 * x^4 - 2
def poly2 (x : ℝ) : ℝ := x^4 - 18 * x^2 + 81
def poly3 (y : ℝ) : ℝ := (y^2 - 1)^2 + 11 * (1 - y^2) + 24

-- Formulate the goals: proving that each polynomial equals its respective factored form.
theorem factor_poly1 (x : ℝ) : poly1 x = 2 * (x^2 + 1) * (x + 1) * (x - 1) :=
sorry

theorem factor_poly2 (x : ℝ) : poly2 x = (x + 3)^2 * (x - 3)^2 :=
sorry

theorem factor_poly3 (y : ℝ) : poly3 y = (y + 2) * (y - 2) * (y + 3) * (y - 3) :=
sorry

end factor_poly1_factor_poly2_factor_poly3_l2139_213914


namespace length_of_other_diagonal_l2139_213990

theorem length_of_other_diagonal (d1 d2 : ℝ) (A : ℝ) (h1 : d1 = 15) (h2 : A = 150) : d2 = 20 :=
by
  sorry

end length_of_other_diagonal_l2139_213990


namespace smallest_n_satisfying_mod_cond_l2139_213997

theorem smallest_n_satisfying_mod_cond (n : ℕ) : (15 * n - 3) % 11 = 0 ↔ n = 9 := by
  sorry

end smallest_n_satisfying_mod_cond_l2139_213997


namespace no_solution_to_system_l2139_213984

theorem no_solution_to_system : ∀ (x y : ℝ), ¬ (y^2 - (⌊x⌋ : ℝ)^2 = 2001 ∧ x^2 + (⌊y⌋ : ℝ)^2 = 2001) :=
by sorry

end no_solution_to_system_l2139_213984


namespace equivalent_operation_l2139_213950

theorem equivalent_operation : 
  let initial_op := (5 / 6 : ℝ)
  let multiply_3_2 := (3 / 2 : ℝ)
  (initial_op * multiply_3_2) = (5 / 4 : ℝ) :=
by
  -- setup operations
  let initial_op := (5 / 6 : ℝ)
  let multiply_3_2 := (3 / 2 : ℝ)
  -- state the goal
  have h : (initial_op * multiply_3_2) = (5 / 4 : ℝ) := sorry
  exact h

end equivalent_operation_l2139_213950


namespace corrected_mean_l2139_213931

theorem corrected_mean (mean_incorrect : ℝ) (number_of_observations : ℕ) (wrong_observation correct_observation : ℝ) : 
  mean_incorrect = 36 → 
  number_of_observations = 50 → 
  wrong_observation = 23 → 
  correct_observation = 43 → 
  (mean_incorrect * number_of_observations + (correct_observation - wrong_observation)) / number_of_observations = 36.4 :=
by
  intros h_mean_incorrect h_number_of_observations h_wrong_observation h_correct_observation
  have S_incorrect : ℝ := mean_incorrect * number_of_observations
  have difference : ℝ := correct_observation - wrong_observation
  have S_correct : ℝ := S_incorrect + difference
  have mean_correct : ℝ := S_correct / number_of_observations
  sorry

end corrected_mean_l2139_213931


namespace sum_a2000_inv_a2000_l2139_213982

theorem sum_a2000_inv_a2000 (a : ℂ) (h : a^2 - a + 1 = 0) : a^2000 + 1/(a^2000) = -1 :=
by
    sorry

end sum_a2000_inv_a2000_l2139_213982


namespace range_of_a_for_quadratic_inequality_l2139_213985

theorem range_of_a_for_quadratic_inequality :
  ∀ a : ℝ, (∀ (x : ℝ), 1 ≤ x ∧ x < 5 → x^2 - (a + 1)*x + a ≤ 0) ↔ (4 ≤ a ∧ a < 5) :=
sorry

end range_of_a_for_quadratic_inequality_l2139_213985


namespace proper_polygons_m_lines_l2139_213976

noncomputable def smallest_m := 2

theorem proper_polygons_m_lines (P : Finset (Set (ℝ × ℝ)))
  (properly_placed : ∀ (p1 p2 : Set (ℝ × ℝ)), p1 ∈ P → p2 ∈ P → ∃ l : Set (ℝ × ℝ), (0, 0) ∈ l ∧ ∀ (p : Set (ℝ × ℝ)), p ∈ P → ¬Disjoint l p) :
  ∃ (m : ℕ), m = smallest_m ∧ ∀ (lines : Finset (Set (ℝ × ℝ))), 
    (∀ l ∈ lines, (0, 0) ∈ l) → lines.card = m → ∀ p ∈ P, ∃ l ∈ lines, ¬Disjoint l p := sorry

end proper_polygons_m_lines_l2139_213976


namespace total_cable_cost_l2139_213964

theorem total_cable_cost 
    (num_east_west_streets : ℕ)
    (length_east_west_street : ℕ)
    (num_north_south_streets : ℕ)
    (length_north_south_street : ℕ)
    (cable_multiplier : ℕ)
    (cable_cost_per_mile : ℕ)
    (h1 : num_east_west_streets = 18)
    (h2 : length_east_west_street = 2)
    (h3 : num_north_south_streets = 10)
    (h4 : length_north_south_street = 4)
    (h5 : cable_multiplier = 5)
    (h6 : cable_cost_per_mile = 2000) :
    (num_east_west_streets * length_east_west_street + num_north_south_streets * length_north_south_street) * cable_multiplier * cable_cost_per_mile = 760000 := 
by
    sorry

end total_cable_cost_l2139_213964


namespace roots_sum_reciprocal_squares_l2139_213952

theorem roots_sum_reciprocal_squares (a b c : ℝ) (h1 : a + b + c = 12) (h2 : ab + bc + ca = 20) (h3 : abc = 3) :
  (1 / a ^ 2) + (1 / b ^ 2) + (1 / c ^ 2) = 328 / 9 := 
by
  sorry

end roots_sum_reciprocal_squares_l2139_213952


namespace value_range_of_function_l2139_213991

theorem value_range_of_function :
  ∀ (x : ℝ), -1 ≤ Real.sin x ∧ Real.sin x ≤ 1 → -1 ≤ Real.sin x * Real.sin x - 2 * Real.sin x ∧ Real.sin x * Real.sin x - 2 * Real.sin x ≤ 3 :=
by
  sorry

end value_range_of_function_l2139_213991


namespace solve_system_of_equations_l2139_213933

theorem solve_system_of_equations :
  ∃ x y z : ℚ, 
    (y * z = 3 * y + 2 * z - 8) ∧
    (z * x = 4 * z + 3 * x - 8) ∧
    (x * y = 2 * x + y - 1) ∧
    ((x = 2 ∧ y = 3 ∧ z = 1) ∨ 
     (x = 3 ∧ y = 5 / 2 ∧ z = -1)) := 
by
  sorry

end solve_system_of_equations_l2139_213933


namespace find_x4_y4_l2139_213915

theorem find_x4_y4 (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 15) : x^4 + y^4 = 175 := by
  sorry

end find_x4_y4_l2139_213915


namespace not_perfect_square_l2139_213967

theorem not_perfect_square (n : ℕ) (h : 0 < n) : ¬ ∃ k : ℕ, k * k = 2551 * 543^n - 2008 * 7^n :=
by
  sorry

end not_perfect_square_l2139_213967


namespace tan_gt_neg_one_solution_set_l2139_213971

def tangent_periodic_solution_set (x : ℝ) : Prop :=
  ∃ k : ℤ, k * Real.pi - Real.pi / 4 < x ∧ x < k * Real.pi + Real.pi / 2

theorem tan_gt_neg_one_solution_set (x : ℝ) :
  tangent_periodic_solution_set x ↔ Real.tan x > -1 :=
by
  sorry

end tan_gt_neg_one_solution_set_l2139_213971


namespace upper_limit_of_people_l2139_213975

theorem upper_limit_of_people (P : ℕ) (h1 : 36 = (3 / 8) * P) (h2 : P > 50) (h3 : (5 / 12) * P = 40) : P = 96 :=
by
  sorry

end upper_limit_of_people_l2139_213975


namespace find_c_of_parabola_l2139_213978

theorem find_c_of_parabola (a b c : ℚ) (h_vertex : (5 : ℚ) = a * (3 : ℚ)^2 + b * (3 : ℚ) + c)
    (h_point : (7 : ℚ) = a * (1 : ℚ)^2 + b * (1 : ℚ) + c) :
  c = 19 / 2 :=
by
  sorry

end find_c_of_parabola_l2139_213978


namespace carpet_length_is_9_l2139_213921

noncomputable def carpet_length (width : ℝ) (living_room_area : ℝ) (coverage : ℝ) : ℝ :=
  living_room_area * coverage / width

theorem carpet_length_is_9 (width : ℝ) (living_room_area : ℝ) (coverage : ℝ) (length := carpet_length width living_room_area coverage) :
    width = 4 → living_room_area = 48 → coverage = 0.75 → length = 9 := by
  intros
  sorry

end carpet_length_is_9_l2139_213921


namespace exponent_simplification_l2139_213904

theorem exponent_simplification : (7^3 * (2^5)^3) / (7^2 * 2^(3*3)) = 448 := by
  sorry

end exponent_simplification_l2139_213904


namespace line_equation_l2139_213932

theorem line_equation (P : ℝ × ℝ) (hP : P = (1, 5)) (h1 : ∃ a, a ≠ 0 ∧ (P.1 + P.2 = a)) (h2 : x_intercept = y_intercept) : 
  (∃ a, a ≠ 0 ∧ P = (a, 0) ∧ P = (0, a) → x + y - 6 = 0) ∨ (5*P.1 - P.2 = 0) :=
by
  sorry

end line_equation_l2139_213932


namespace counterexample_disproving_proposition_l2139_213963

theorem counterexample_disproving_proposition (angle1 angle2 : ℝ) (h1 : angle1 + angle2 = 90) (h2 : angle1 = angle2) : False :=
by
  have h_contradiction : angle1 ≠ angle2 := sorry
  exact h_contradiction h2

end counterexample_disproving_proposition_l2139_213963


namespace remainder_calculation_l2139_213939

theorem remainder_calculation 
  (x : ℤ) (y : ℝ)
  (hx : 0 < x)
  (hy : y = 70.00000000000398)
  (hx_div_y : (x : ℝ) / y = 86.1) :
  x % y = 7 :=
by
  sorry

end remainder_calculation_l2139_213939


namespace ball_bounces_height_l2139_213999

theorem ball_bounces_height (initial_height : ℝ) (decay_factor : ℝ) (threshold : ℝ) (n : ℕ) :
  initial_height = 20 →
  decay_factor = 3/4 →
  threshold = 2 →
  n = 9 →
  initial_height * (decay_factor ^ n) < threshold :=
by
  intros
  sorry

end ball_bounces_height_l2139_213999


namespace maximal_k_value_l2139_213948

noncomputable def max_edges (n : ℕ) : ℕ :=
  2 * n - 4
   
theorem maximal_k_value (k n : ℕ) (h1 : n = 2016) (h2 : k ≤ max_edges n) :
  k = 4028 :=
by sorry

end maximal_k_value_l2139_213948


namespace constant_term_of_expansion_l2139_213916

noncomputable def constant_term := 
  (20: ℕ) * (216: ℕ) * (1/27: ℚ) = (160: ℕ)

theorem constant_term_of_expansion : constant_term :=
  by sorry

end constant_term_of_expansion_l2139_213916


namespace find_t_l2139_213970

theorem find_t : ∃ t, ∀ (x y : ℝ), (x, y) = (0, 1) ∨ (x, y) = (-6, -3) → (t, 7) ∈ {p : ℝ × ℝ | ∃ m b, p.2 = m * p.1 + b ∧ ((0, 1) ∈ {p : ℝ × ℝ | p.2 = m * p.1 + b}) ∧ ((-6, -3) ∈ {p : ℝ × ℝ | p.2 = m * p.1 + b}) } → t = 9 :=
by
  sorry

end find_t_l2139_213970


namespace pages_in_first_chapter_l2139_213919

theorem pages_in_first_chapter (x : ℕ) (h1 : x + 43 = 80) : x = 37 :=
by
  sorry

end pages_in_first_chapter_l2139_213919


namespace color_blocks_probability_at_least_one_box_match_l2139_213946

/-- Given Ang, Ben, and Jasmin each having 6 blocks of different colors (red, blue, yellow, white, green, and orange) 
    and they independently place one of their blocks into each of 6 empty boxes, 
    the proof shows that the probability that at least one box receives 3 blocks all of the same color is 1/6. 
    Since 1/6 is equal to the fraction m/n where m=1 and n=6 are relatively prime, thus m+n=7. -/
theorem color_blocks_probability_at_least_one_box_match (p : ℕ × ℕ) (h : p = (1, 6)) : p.1 + p.2 = 7 :=
by {
  sorry
}

end color_blocks_probability_at_least_one_box_match_l2139_213946


namespace average_brown_mms_per_bag_l2139_213993

-- Definitions based on the conditions
def bag1_brown_mm : ℕ := 9
def bag2_brown_mm : ℕ := 12
def bag3_brown_mm : ℕ := 8
def bag4_brown_mm : ℕ := 8
def bag5_brown_mm : ℕ := 3
def number_of_bags : ℕ := 5

-- The proof problem statement
theorem average_brown_mms_per_bag : 
  (bag1_brown_mm + bag2_brown_mm + bag3_brown_mm + bag4_brown_mm + bag5_brown_mm) / number_of_bags = 8 := 
by
  sorry

end average_brown_mms_per_bag_l2139_213993


namespace max_value_expr_l2139_213965

theorem max_value_expr (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (∀ x : ℝ, (a + b)^2 / (a^2 + 2 * a * b + b^2) ≤ x) → 1 ≤ x :=
sorry

end max_value_expr_l2139_213965
