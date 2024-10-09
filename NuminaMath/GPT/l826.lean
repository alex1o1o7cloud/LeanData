import Mathlib

namespace five_letter_sequences_l826_82659

-- Define the quantities of each vowel.
def quantity_vowel_A : Nat := 3
def quantity_vowel_E : Nat := 4
def quantity_vowel_I : Nat := 5
def quantity_vowel_O : Nat := 6
def quantity_vowel_U : Nat := 7

-- Define the number of choices for each letter in a five-letter sequence.
def choices_per_letter : Nat := 5

-- Define the total number of five-letter sequences.
noncomputable def total_sequences : Nat := choices_per_letter ^ 5

-- Prove that the number of five-letter sequences is 3125.
theorem five_letter_sequences : total_sequences = 3125 :=
by sorry

end five_letter_sequences_l826_82659


namespace sum_first_9_terms_arithmetic_sequence_l826_82651

noncomputable def sum_of_first_n_terms (a_1 d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a_1 + (n - 1) * d) / 2

def arithmetic_sequence_term (a_1 d : ℤ) (n : ℕ) : ℤ :=
  a_1 + (n - 1) * d

theorem sum_first_9_terms_arithmetic_sequence :
  ∃ a_1 d : ℤ, (a_1 + arithmetic_sequence_term a_1 d 4 + arithmetic_sequence_term a_1 d 7 = 39) ∧
               (arithmetic_sequence_term a_1 d 3 + arithmetic_sequence_term a_1 d 6 + arithmetic_sequence_term a_1 d 9 = 27) ∧
               (sum_of_first_n_terms a_1 d 9 = 99) :=
by
  sorry

end sum_first_9_terms_arithmetic_sequence_l826_82651


namespace find_rstu_l826_82634

theorem find_rstu (a x y c : ℝ) (r s t u : ℤ) (hc : a^10 * x * y - a^8 * y - a^7 * x = a^6 * (c^3 - 1)) :
  (a^r * x - a^s) * (a^t * y - a^u) = a^6 * c^3 ∧ r * s * t * u = 0 :=
by
  sorry

end find_rstu_l826_82634


namespace complete_square_form_l826_82690

theorem complete_square_form :
  ∀ x : ℝ, (3 * x^2 - 6 * x + 2 = 0) → (x - 1)^2 = (1 / 3) :=
by
  intro x h
  sorry

end complete_square_form_l826_82690


namespace sum_coordinates_of_k_l826_82679

theorem sum_coordinates_of_k :
  ∀ (f k : ℕ → ℕ), (f 4 = 8) → (∀ x, k x = (f x) ^ 3) → (4 + k 4) = 516 :=
by
  intros f k h1 h2
  sorry

end sum_coordinates_of_k_l826_82679


namespace charity_total_cost_l826_82625

theorem charity_total_cost
  (plates : ℕ)
  (rice_cost_per_plate chicken_cost_per_plate : ℕ)
  (h1 : plates = 100)
  (h2 : rice_cost_per_plate = 10)
  (h3 : chicken_cost_per_plate = 40) :
  plates * (rice_cost_per_plate + chicken_cost_per_plate) / 100 = 50 := 
by
  sorry

end charity_total_cost_l826_82625


namespace problem_solution_l826_82662

theorem problem_solution (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2):
    (x ∈ Set.Iio (-2) ∪ Set.Ioo (-2) ((1 - Real.sqrt 129)/8) ∪ Set.Ioo 2 3 ∪ Set.Ioi ((1 + (Real.sqrt 129))/8)) ↔
    (2 * x^2 / (x + 2) ≥ 3 / (x - 2) + 6 / 4) :=
by
  sorry

end problem_solution_l826_82662


namespace charge_per_mile_l826_82676

def rental_fee : ℝ := 20.99
def total_amount_paid : ℝ := 95.74
def miles_driven : ℝ := 299

theorem charge_per_mile :
  (total_amount_paid - rental_fee) / miles_driven = 0.25 := 
sorry

end charge_per_mile_l826_82676


namespace center_and_radius_of_circle_l826_82601

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x + 6 * y + 6 = 0

-- State the theorem
theorem center_and_radius_of_circle :
  (∃ x₀ y₀ r, (∀ x y, circle_eq x y ↔ (x - x₀)^2 + (y - y₀)^2 = r^2) ∧
  x₀ = 1 ∧ y₀ = -3 ∧ r = 2) :=
by
  -- Proof is omitted
  sorry

end center_and_radius_of_circle_l826_82601


namespace vector_equation_proof_l826_82629

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C P : V)

/-- The given condition. -/
def given_condition : Prop :=
  (P - A) + 2 • (P - B) + 3 • (P - C) = 0

/-- The target equality we want to prove. -/
theorem vector_equation_proof (h : given_condition A B C P) :
  P - A = (1 / 3 : ℝ) • (B - A) + (1 / 2 : ℝ) • (C - A) :=
sorry

end vector_equation_proof_l826_82629


namespace fractions_of_120_equals_2_halves_l826_82647

theorem fractions_of_120_equals_2_halves :
  (1 / 6) * (1 / 4) * (1 / 5) * 120 = 2 / 2 := 
by
  sorry

end fractions_of_120_equals_2_halves_l826_82647


namespace cylinder_surface_area_l826_82655

/-- A right cylinder with radius 3 inches and height twice the radius has a total surface area of 54π square inches. -/
theorem cylinder_surface_area (r : ℝ) (h : ℝ) (A_total : ℝ) (π : ℝ) : r = 3 → h = 2 * r → π = Real.pi → A_total = 54 * π :=
by
  sorry

end cylinder_surface_area_l826_82655


namespace probability_third_smallest_is_five_l826_82669

theorem probability_third_smallest_is_five :
  let total_ways := Nat.choose 12 7
  let favorable_ways := (Nat.choose 4 2) * (Nat.choose 7 4)
  let probability := favorable_ways / total_ways
  probability = Rat.ofInt 35 / 132 :=
by
  let total_ways := Nat.choose 12 7
  let favorable_ways := (Nat.choose 4 2) * (Nat.choose 7 4)
  let probability := favorable_ways / total_ways
  show probability = Rat.ofInt 35 / 132
  sorry

end probability_third_smallest_is_five_l826_82669


namespace polygon_sides_l826_82686

theorem polygon_sides (n : ℕ) (c : ℕ) 
  (h₁ : c = n * (n - 3) / 2)
  (h₂ : c = 2 * n) : n = 7 :=
sorry

end polygon_sides_l826_82686


namespace student_count_l826_82649

theorem student_count 
  (initial_avg_height : ℚ)
  (incorrect_height : ℚ)
  (actual_height : ℚ)
  (actual_avg_height : ℚ)
  (n : ℕ)
  (h1 : initial_avg_height = 175)
  (h2 : incorrect_height = 151)
  (h3 : actual_height = 136)
  (h4 : actual_avg_height = 174.5)
  (h5 : n > 0) : n = 30 :=
by
  sorry

end student_count_l826_82649


namespace fraction_inequality_l826_82677

theorem fraction_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (1 / a) + (1 / b) ≥ (4 / (a + b)) :=
by 
-- Skipping the proof using 'sorry'
sorry

end fraction_inequality_l826_82677


namespace application_schemes_eq_l826_82607

noncomputable def number_of_application_schemes (graduates : ℕ) (universities : ℕ) : ℕ :=
  universities ^ graduates

theorem application_schemes_eq : 
  number_of_application_schemes 5 3 = 3 ^ 5 := 
by 
  -- proof goes here
  sorry

end application_schemes_eq_l826_82607


namespace annual_growth_rate_l826_82658

theorem annual_growth_rate (x : ℝ) (h : 2000 * (1 + x) ^ 2 = 2880) : x = 0.2 :=
by sorry

end annual_growth_rate_l826_82658


namespace roger_individual_pouches_per_pack_l826_82617

variable (members : ℕ) (coaches : ℕ) (helpers : ℕ) (packs : ℕ)

-- Given conditions
def total_people (members coaches helpers : ℕ) : ℕ := members + coaches + helpers
def pouches_per_pack (total_people packs : ℕ) : ℕ := total_people / packs

-- Specific values from the problem
def roger_total_people : ℕ := total_people 13 3 2
def roger_packs : ℕ := 3

-- The problem statement to prove:
theorem roger_individual_pouches_per_pack : pouches_per_pack roger_total_people roger_packs = 6 :=
by
  sorry

end roger_individual_pouches_per_pack_l826_82617


namespace circle_equation_bisected_and_tangent_l826_82674

theorem circle_equation_bisected_and_tangent :
  (∃ x0 y0 r : ℝ, x0 = y0 ∧ (x0 + y0 - 2 * r) = 0 ∧ (∀ x y : ℝ, (x - x0)^2 + (y - y0)^2 = r^2 → (x - 1)^2 + (y - 1)^2 = 2)) := sorry

end circle_equation_bisected_and_tangent_l826_82674


namespace carrie_total_sales_l826_82627

theorem carrie_total_sales :
  let tomatoes := 200
  let carrots := 350
  let price_tomato := 1.0
  let price_carrot := 1.50
  (tomatoes * price_tomato + carrots * price_carrot) = 725 := by
  -- let tomatoes := 200
  -- let carrots := 350
  -- let price_tomato := 1.0
  -- let price_carrot := 1.50
  -- show (tomatoes * price_tomato + carrots * price_carrot) = 725
  sorry

end carrie_total_sales_l826_82627


namespace parallelogram_area_correct_l826_82642

noncomputable def parallelogram_area (s1 s2 : ℝ) (a : ℝ) : ℝ :=
s2 * (2 * s2 * Real.sin a)

theorem parallelogram_area_correct (s2 a : ℝ) (h_pos_s2 : 0 < s2) :
  parallelogram_area (2 * s2) s2 a = 2 * s2^2 * Real.sin a :=
by
  unfold parallelogram_area
  sorry

end parallelogram_area_correct_l826_82642


namespace train_boarding_probability_l826_82643

theorem train_boarding_probability :
  (0.5 / 5) = 1 / 10 :=
by sorry

end train_boarding_probability_l826_82643


namespace distinct_real_roots_find_p_l826_82681

theorem distinct_real_roots (p : ℝ) : 
  let f := (fun x => (x - 3) * (x - 2) - p^2)
  let Δ := 1 + 4 * p ^ 2 
  0 < Δ :=
by sorry

theorem find_p (x1 x2 p : ℝ) : 
  (x1 + x2 = 5) → 
  (x1 * x2 = 6 - p^2) → 
  (x1^2 + x2^2 = 3 * x1 * x2) → 
  (p = 1 ∨ p = -1) :=
by sorry

end distinct_real_roots_find_p_l826_82681


namespace digit_150_in_17_div_70_l826_82614

noncomputable def repeating_sequence_170 : List Nat := [2, 4, 2, 8, 5, 7]

theorem digit_150_in_17_div_70 : (repeating_sequence_170.get? (150 % 6 - 1) = some 7) := by
  sorry

end digit_150_in_17_div_70_l826_82614


namespace sufficient_not_necessary_l826_82663

theorem sufficient_not_necessary (x : ℝ) : (x - 1 > 0) → (x^2 - 1 > 0) ∧ ¬((x^2 - 1 > 0) → (x - 1 > 0)) :=
by
  sorry

end sufficient_not_necessary_l826_82663


namespace geometric_sum_eight_terms_l826_82632

theorem geometric_sum_eight_terms :
  let a0 := (1 : ℚ) / 3
  let r := (1 : ℚ) / 4
  let n := 8
  let S_n := a0 * (1 - r^n) / (1 - r)
  S_n = 65535 / 147456 := by
  sorry

end geometric_sum_eight_terms_l826_82632


namespace perp_tangents_l826_82616

theorem perp_tangents (a b : ℝ) (h : a + b = 5) (tangent_perp : ∀ x y : ℝ, x = 1 ∧ y = 1) :
  a / b = 1 / 3 :=
sorry

end perp_tangents_l826_82616


namespace percentage_silver_cars_after_shipment_l826_82613

-- Definitions for conditions
def initialCars : ℕ := 40
def initialSilverPerc : ℝ := 0.15
def newShipmentCars : ℕ := 80
def newShipmentNonSilverPerc : ℝ := 0.30

-- Proof statement that needs to be proven
theorem percentage_silver_cars_after_shipment :
  let initialSilverCars := initialSilverPerc * initialCars
  let newShipmentSilverPerc := 1 - newShipmentNonSilverPerc
  let newShipmentSilverCars := newShipmentSilverPerc * newShipmentCars
  let totalSilverCars := initialSilverCars + newShipmentSilverCars
  let totalCars := initialCars + newShipmentCars
  (totalSilverCars / totalCars) * 100 = 51.67 :=
by
  sorry

end percentage_silver_cars_after_shipment_l826_82613


namespace quadratic_complex_roots_condition_l826_82699

theorem quadratic_complex_roots_condition (a : ℝ) :
  (∀ a, -2 ≤ a ∧ a ≤ 2 → (a^2 < 4)) ∧ 
  ¬(∀ a, (a^2 < 4) → -2 ≤ a ∧ a ≤ 2) :=
by
  sorry

end quadratic_complex_roots_condition_l826_82699


namespace smallest_prime_fifth_term_of_arithmetic_sequence_l826_82694

theorem smallest_prime_fifth_term_of_arithmetic_sequence :
  ∃ (a d : ℕ) (seq : ℕ → ℕ), 
    (∀ n, seq n = a + n * d) ∧ 
    (∀ n < 5, Prime (seq n)) ∧ 
    d = 6 ∧ 
    a = 5 ∧ 
    seq 4 = 29 := by
  sorry

end smallest_prime_fifth_term_of_arithmetic_sequence_l826_82694


namespace greatest_value_l826_82675

theorem greatest_value (y : ℝ) (h : 4 * y^2 + 4 * y + 3 = 1) : (y + 1)^2 = 1/4 :=
sorry

end greatest_value_l826_82675


namespace continuity_of_f_at_2_l826_82602

def f (x : ℝ) := -2 * x^2 - 5

theorem continuity_of_f_at_2 : ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |f x - f 2| < ε :=
by {
  sorry
}

end continuity_of_f_at_2_l826_82602


namespace product_of_slopes_l826_82697

theorem product_of_slopes (m n : ℝ) (φ₁ φ₂ : ℝ) 
  (h1 : ∀ x, y = m * x)
  (h2 : ∀ x, y = n * x)
  (h3 : φ₁ = 2 * φ₂) 
  (h4 : m = 3 * n)
  (h5 : m ≠ 0 ∧ n ≠ 0)
  : m * n = 3 / 5 :=
sorry

end product_of_slopes_l826_82697


namespace committee_formation_l826_82633

/-- Problem statement: In how many ways can a 5-person executive committee be formed if one of the 
members must be the president, given there are 30 members. --/
theorem committee_formation (n : ℕ) (k : ℕ) (h : n = 30) (h2 : k = 5) : 
  (n * Nat.choose (n - 1) (k - 1) = 712530 ) :=
by
  sorry

end committee_formation_l826_82633


namespace find_k_l826_82680

theorem find_k (x y z k : ℝ) (h1 : 8 / (x + y + 1) = k / (x + z + 2)) (h2 : k / (x + z + 2) = 12 / (z - y + 3)) : k = 20 := by
  sorry

end find_k_l826_82680


namespace cubic_expression_l826_82678

theorem cubic_expression (x : ℝ) (hx : x + 1/x = -7) : x^3 + 1/x^3 = -322 :=
by sorry

end cubic_expression_l826_82678


namespace eventually_periodic_sequence_l826_82636

noncomputable def eventually_periodic (a : ℕ → ℕ) : Prop :=
  ∃ N k : ℕ, k > 0 ∧ ∀ m ≥ N, a m = a (m + k)

theorem eventually_periodic_sequence
  (a : ℕ → ℕ)
  (h_pos : ∀ n, a n > 0)
  (h_condition : ∀ n, a n * a (n + 1) = a (n + 2) * a (n + 3)) :
  eventually_periodic a :=
sorry

end eventually_periodic_sequence_l826_82636


namespace max_range_walk_min_range_walk_count_max_range_sequences_l826_82660

variable {a b : ℕ}

-- Condition: a > b
def valid_walk (a b : ℕ) : Prop := a > b

-- Proof that the maximum possible range of the walk is a
theorem max_range_walk (h : valid_walk a b) : 
  (a + b) = a + b := sorry

-- Proof that the minimum possible range of the walk is a - b
theorem min_range_walk (h : valid_walk a b) : 
  (a - b) = a - b := sorry

-- Proof that the number of different sequences with the maximum possible range is b + 1
theorem count_max_range_sequences (h : valid_walk a b) : 
  b + 1 = b + 1 := sorry

end max_range_walk_min_range_walk_count_max_range_sequences_l826_82660


namespace determine_x_l826_82657

theorem determine_x (x : ℝ) (h : (1 / (Real.log x / Real.log 3) + 1 / (Real.log x / Real.log 5) + 1 / (Real.log x / Real.log 6) = 1)) : 
    x = 90 := 
by 
  sorry

end determine_x_l826_82657


namespace quadratic_inequality_empty_solution_set_l826_82673

theorem quadratic_inequality_empty_solution_set (a b c : ℝ) (hₐ : a ≠ 0) :
  (∀ x : ℝ, a * x^2 + b * x + c < 0 → False) ↔ (a > 0 ∧ (b^2 - 4 * a * c) ≤ 0) :=
by 
  sorry

end quadratic_inequality_empty_solution_set_l826_82673


namespace integer_solutions_l826_82689

theorem integer_solutions (x y z : ℤ) : 
  x + y + z = 3 ∧ x^3 + y^3 + z^3 = 3 ↔ 
  (x = 1 ∧ y = 1 ∧ z = 1) ∨
  (x = 4 ∧ y = 4 ∧ z = -5) ∨
  (x = 4 ∧ y = -5 ∧ z = 4) ∨
  (x = -5 ∧ y = 4 ∧ z = 4) := 
sorry

end integer_solutions_l826_82689


namespace sum_of_reciprocals_of_squares_l826_82670

open BigOperators

theorem sum_of_reciprocals_of_squares (n : ℕ) (h : n ≥ 2) :
   (∑ k in Finset.range n, 1 / (k + 1)^2) < (2 * n - 1) / n :=
sorry

end sum_of_reciprocals_of_squares_l826_82670


namespace min_workers_to_make_profit_l826_82692

theorem min_workers_to_make_profit :
  ∃ n : ℕ, 500 + 8 * 15 * n < 124 * n ∧ n = 126 :=
by
  sorry

end min_workers_to_make_profit_l826_82692


namespace arithmetic_expression_equiv_l826_82609

theorem arithmetic_expression_equiv :
  (-1:ℤ)^2009 * (-3) + 1 - 2^2 * 3 + (1 - 2^2) / 3 + (1 - 2 * 3)^2 = 16 := by
  sorry

end arithmetic_expression_equiv_l826_82609


namespace best_choice_to_calculate_89_8_sq_l826_82682

theorem best_choice_to_calculate_89_8_sq 
  (a b c d : ℚ) 
  (h1 : (89 + 0.8)^2 = a) 
  (h2 : (80 + 9.8)^2 = b) 
  (h3 : (90 - 0.2)^2 = c) 
  (h4 : (100 - 10.2)^2 = d) : 
  c = 89.8^2 := by
  sorry

end best_choice_to_calculate_89_8_sq_l826_82682


namespace find_cost_of_fourth_cd_l826_82618

variables (cost1 cost2 cost3 cost4 : ℕ)
variables (h1 : (cost1 + cost2 + cost3) / 3 = 15)
variables (h2 : (cost1 + cost2 + cost3 + cost4) / 4 = 16)

theorem find_cost_of_fourth_cd : cost4 = 19 := 
by 
  sorry

end find_cost_of_fourth_cd_l826_82618


namespace index_difference_l826_82608

theorem index_difference (n f m : ℕ) (h_n : n = 25) (h_f : f = 8) (h_m : m = 25 - 8) :
  (n - f) / n - (n - m) / n = 9 / 25 :=
by
  -- The proof is to be completed here.
  sorry

end index_difference_l826_82608


namespace evaluate_expression_l826_82672

theorem evaluate_expression : -1 ^ 2010 + (-1) ^ 2011 + 1 ^ 2012 - 1 ^ 2013 = -2 :=
by
  -- sorry is added as a placeholder for the proof steps
  sorry

end evaluate_expression_l826_82672


namespace factor_1024_count_l826_82635

theorem factor_1024_count :
  ∃ (n : ℕ), 
  (∀ (a b c : ℕ), (a >= b) → (b >= c) → (2^a * 2^b * 2^c = 1024) → a + b + c = 10) ∧ n = 14 :=
sorry

end factor_1024_count_l826_82635


namespace TrainTravelDays_l826_82603

-- Definition of the problem conditions
def train_start (days: ℕ) : ℕ := 
  if days = 0 then 0 -- no trains to meet on the first day
  else days -- otherwise, meet 'days' number of trains

/-- 
  Prove that if a train comes across 4 trains on its way from Amritsar to Bombay and starts at 9 am, 
  then it takes 5 days for the train to reach its destination.
-/
theorem TrainTravelDays (meet_train_count : ℕ) : meet_train_count = 4 → train_start (meet_train_count) + 1 = 5 :=
by
  intro h
  rw [h]
  sorry

end TrainTravelDays_l826_82603


namespace johns_cycling_speed_needed_l826_82638

theorem johns_cycling_speed_needed 
  (swim_speed : Float := 3)
  (swim_distance : Float := 0.5)
  (run_speed : Float := 8)
  (run_distance : Float := 4)
  (total_time : Float := 3)
  (bike_distance : Float := 20) :
  (bike_distance / (total_time - (swim_distance / swim_speed + run_distance / run_speed))) = 60 / 7 := 
  by
  sorry

end johns_cycling_speed_needed_l826_82638


namespace inequality_always_holds_l826_82621

theorem inequality_always_holds (m : ℝ) : (-6 < m ∧ m ≤ 0) ↔ ∀ x : ℝ, 2 * m * x^2 + m * x - 3 / 4 < 0 := 
sorry

end inequality_always_holds_l826_82621


namespace infinite_indices_exist_l826_82624

theorem infinite_indices_exist (a : ℕ → ℕ) (h_seq : ∀ n, a n < a (n + 1)) :
  ∃ᶠ m in ⊤, ∃ x y h k : ℕ, 0 < h ∧ h < k ∧ k < m ∧ a m = x * a h + y * a k :=
by sorry

end infinite_indices_exist_l826_82624


namespace non_obtuse_triangle_medians_ge_4R_l826_82622

theorem non_obtuse_triangle_medians_ge_4R
  (A B C : Type*)
  (triangle_non_obtuse : ∀ (α β γ : ℝ), α ≤ 90 ∧ β ≤ 90 ∧ γ ≤ 90)
  (m_a m_b m_c : ℝ)
  (R : ℝ)
  (h1 : AO + BO ≤ AM + BM)
  (h2 : AM = 2 * m_a / 3 ∧ BM = 2 * m_b / 3)
  (h3 : AO + BO = 2 * R)
  (h4 : m_c ≥ R) : 
  m_a + m_b + m_c ≥ 4 * R :=
by
  sorry

end non_obtuse_triangle_medians_ge_4R_l826_82622


namespace domain_of_function_l826_82606

theorem domain_of_function :
  ∀ x : ℝ, (x > 0) ∧ (x ≤ 2) ∧ (x ≠ 1) ↔ ∀ x, (∃ y : ℝ, y = (1 / (Real.log x / Real.log 10) + Real.sqrt (2 - x))) :=
by
  sorry

end domain_of_function_l826_82606


namespace tree_age_when_23_feet_l826_82688

theorem tree_age_when_23_feet (initial_age initial_height growth_rate final_height : ℕ) 
(h_initial_age : initial_age = 1)
(h_initial_height : initial_height = 5) 
(h_growth_rate : growth_rate = 3) 
(h_final_height : final_height = 23) : 
initial_age + (final_height - initial_height) / growth_rate = 7 := 
by sorry

end tree_age_when_23_feet_l826_82688


namespace distance_from_neg6_to_origin_l826_82639

theorem distance_from_neg6_to_origin :
  abs (-6) = 6 :=
by
  sorry

end distance_from_neg6_to_origin_l826_82639


namespace arithmetic_geometric_sequence_l826_82628

theorem arithmetic_geometric_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) (q : ℕ)
  (h₀ : ∀ n, a n = 2^(n-1))
  (h₁ : a 1 = 1)
  (h₂ : a 1 + a 2 + a 3 = 7)
  (h₃ : q > 0) :
  (∀ n, a n = 2^(n-1)) ∧ (∀ n, S n = 2^n - 1) :=
by {
  sorry
}

end arithmetic_geometric_sequence_l826_82628


namespace probability_abs_x_le_one_l826_82667

noncomputable def geometric_probability (a b c d : ℝ) : ℝ := (b - a) / (d - c)

theorem probability_abs_x_le_one : 
  ∀ (x : ℝ), x ∈ Set.Icc (-1 : ℝ) 3 →  
  geometric_probability (-1) 1 (-1) 3 = 1 / 2 := 
by
  sorry

end probability_abs_x_le_one_l826_82667


namespace smallest_n_l826_82600

theorem smallest_n {n : ℕ} (h1 : n ≡ 4 [MOD 6]) (h2 : n ≡ 3 [MOD 7]) (h3 : n > 10) : n = 52 :=
sorry

end smallest_n_l826_82600


namespace problem1_problem2_l826_82664

-- Step 1
theorem problem1 (a b c A B C : ℝ) (h1 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) :
  2 * a^2 = b^2 + c^2 := sorry

-- Step 2
theorem problem2 (a b c : ℝ) (h_a : a = 5) (h_cosA : Real.cos A = 25 / 31) 
  (h_conditions : 2 * a^2 = b^2 + c^2 ∧ 2 * b * c = a^2 / Real.cos A) :
  a + b + c = 14 := sorry

end problem1_problem2_l826_82664


namespace range_of_a_l826_82630

theorem range_of_a (a b : ℝ) (h : a - 4 * Real.sqrt b = 2 * Real.sqrt (a - b)) : 
  a ∈ {x | 0 ≤ x} ∧ ((a = 0) ∨ (4 ≤ a ∧ a ≤ 20)) :=
by
  sorry

end range_of_a_l826_82630


namespace mod_2_pow_1000_by_13_l826_82684

theorem mod_2_pow_1000_by_13 :
  (2 ^ 1000) % 13 = 3 := by
  sorry

end mod_2_pow_1000_by_13_l826_82684


namespace shelly_thread_length_l826_82661

theorem shelly_thread_length 
  (threads_per_keychain : ℕ := 12) 
  (friends_in_class : ℕ := 6) 
  (friends_from_clubs := friends_in_class / 2)
  (total_friends := friends_in_class + friends_from_clubs) 
  (total_threads_needed := total_friends * threads_per_keychain) : 
  total_threads_needed = 108 := 
by 
  -- proof skipped
  sorry

end shelly_thread_length_l826_82661


namespace parallel_vectors_l826_82687

theorem parallel_vectors (m : ℝ) (a b : ℝ × ℝ) (h₁ : a = (2, 1)) (h₂ : b = (1, m))
  (h₃ : ∃ k : ℝ, b = k • a) : m = 1 / 2 :=
by 
  sorry

end parallel_vectors_l826_82687


namespace math_expression_evaluation_l826_82620

theorem math_expression_evaluation :
  36 + (120 / 15) + (15 * 19) - 150 - (450 / 9) = 129 :=
by
  sorry

end math_expression_evaluation_l826_82620


namespace odd_nat_composite_iff_exists_a_l826_82648

theorem odd_nat_composite_iff_exists_a (c : ℕ) (h_odd : c % 2 = 1) :
  (∃ a : ℕ, a ≤ c / 3 - 1 ∧ ∃ k : ℕ, (2*a - 1)^2 + 8*c = k^2) ↔
  ∃ d : ℕ, ∃ e : ℕ, d > 1 ∧ e > 1 ∧ d * e = c := 
sorry

end odd_nat_composite_iff_exists_a_l826_82648


namespace flattest_ellipse_is_B_l826_82646

-- Definitions for the given ellipses
def ellipseA : Prop := ∀ (x y : ℝ), (x^2 / 16 + y^2 / 12 = 1)
def ellipseB : Prop := ∀ (x y : ℝ), (x^2 / 4 + y^2 = 1)
def ellipseC : Prop := ∀ (x y : ℝ), (x^2 / 6 + y^2 / 3 = 1)
def ellipseD : Prop := ∀ (x y : ℝ), (x^2 / 9 + y^2 / 8 = 1)

-- The proof to show that ellipseB is the flattest
theorem flattest_ellipse_is_B : ellipseB := by
  sorry

end flattest_ellipse_is_B_l826_82646


namespace circle_center_l826_82691

theorem circle_center (x y : ℝ) : (x - 2)^2 + (y + 1)^2 = 3 → (2, -1) = (2, -1) :=
by
  intro h
  -- Proof omitted
  sorry

end circle_center_l826_82691


namespace find_a_l826_82668

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + a / ((Real.exp (2 * x)) - 1)

theorem find_a : ∃ a : ℝ, (∀ x : ℝ, f a x = -f a (-x)) → a = 2 :=
by
  sorry

end find_a_l826_82668


namespace propositions_correctness_l826_82645

variable {a b c d : ℝ}

theorem propositions_correctness (h0 : a > b) (h1 : c > d) (h2 : c > 0) :
  (a > b ∧ c > d → a + c > b + d) ∧ 
  (a > b ∧ c > d → ¬(a - c > b - d)) ∧ 
  (a > b ∧ c > d → ¬(a * c > b * d)) ∧ 
  (a > b ∧ c > 0 → a * c > b * c) :=
by
  sorry

end propositions_correctness_l826_82645


namespace gemstones_needed_l826_82696

noncomputable def magnets_per_earring : ℕ := 2

noncomputable def buttons_per_earring : ℕ := magnets_per_earring / 2

noncomputable def gemstones_per_earring : ℕ := 3 * buttons_per_earring

noncomputable def sets_of_earrings : ℕ := 4

noncomputable def earrings_per_set : ℕ := 2

noncomputable def total_gemstones : ℕ := sets_of_earrings * earrings_per_set * gemstones_per_earring

theorem gemstones_needed :
  total_gemstones = 24 :=
  by
    sorry

end gemstones_needed_l826_82696


namespace calculate_sequence_sum_l826_82656

noncomputable def sum_arithmetic_sequence (a l d: Int) : Int :=
  let n := ((l - a) / d) + 1
  (n * (a + l)) / 2

theorem calculate_sequence_sum :
  3 * (sum_arithmetic_sequence 45 93 2) + 2 * (sum_arithmetic_sequence (-4) 38 2) = 5923 := by
  sorry

end calculate_sequence_sum_l826_82656


namespace ratio_is_9_l826_82641

-- Define the set of numbers
def set_of_numbers := { x : ℕ | ∃ n, n ≤ 8 ∧ x = 10^n }

-- Define the sum of the geometric series excluding the largest element
def sum_of_others : ℕ := (Finset.range 8).sum (λ n => 10^n)

-- Define the largest element
def largest_element := 10^8

-- Define the ratio of the largest element to the sum of the other elements
def ratio := largest_element / sum_of_others

-- Problem statement: The ratio is 9
theorem ratio_is_9 : ratio = 9 := by
  sorry

end ratio_is_9_l826_82641


namespace inequality_proof_l826_82623

theorem inequality_proof (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a > b) : 
  1 / (a * b^2) > 1 / (a^2 * b) :=
sorry

end inequality_proof_l826_82623


namespace division_expression_result_l826_82666

theorem division_expression_result :
  -1 / (-5) / (-1 / 5) = -1 :=
by sorry

end division_expression_result_l826_82666


namespace largest_k_divides_2_pow_3_pow_m_add_1_l826_82693

theorem largest_k_divides_2_pow_3_pow_m_add_1 (m : ℕ) : 9 ∣ 2^(3^m) + 1 := sorry

end largest_k_divides_2_pow_3_pow_m_add_1_l826_82693


namespace max_sum_of_factors_l826_82644

theorem max_sum_of_factors (x y : ℕ) (h1 : x * y = 48) (h2 : x ≠ y) : x + y ≤ 49 :=
by
  sorry

end max_sum_of_factors_l826_82644


namespace slope_of_line_determined_by_solutions_l826_82631

theorem slope_of_line_determined_by_solutions (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : 3 / x₁ +  4 / y₁ = 0)
  (h₂ : 3 / x₂ + 4 / y₂ = 0) :
  (y₂ - y₁) / (x₂ - x₁) = -4 / 3 :=
sorry

end slope_of_line_determined_by_solutions_l826_82631


namespace fraction_power_multiplication_l826_82612

theorem fraction_power_multiplication :
  ((1 : ℝ) / 3) ^ 4 * ((1 : ℝ) / 5) = ((1 : ℝ) / 405) := by
  sorry

end fraction_power_multiplication_l826_82612


namespace find_b_l826_82654

theorem find_b (a b : ℝ) (h_inv_var : a^2 * Real.sqrt b = k) (h_ab : a * b = 72) (ha3 : a = 3) (hb64 : b = 64) : b = 18 :=
sorry

end find_b_l826_82654


namespace option_C_forms_a_set_l826_82610

-- Definition of the criteria for forming a set
def well_defined (criterion : Prop) : Prop := criterion

-- Criteria for option C: all female students in grade one of Jiu Middle School
def grade_one_students_criteria (is_female : Prop) (is_grade_one_student : Prop) : Prop :=
  is_female ∧ is_grade_one_student

-- Proof statement
theorem option_C_forms_a_set :
  ∀ (is_female : Prop) (is_grade_one_student : Prop), well_defined (grade_one_students_criteria is_female is_grade_one_student) :=
  by sorry

end option_C_forms_a_set_l826_82610


namespace number_of_women_in_preston_after_one_year_l826_82615

def preston_is_25_times_leesburg (preston leesburg : ℕ) : Prop := 
  preston = 25 * leesburg

def leesburg_population : ℕ := 58940

def women_percentage_leesburg : ℕ := 40

def women_percentage_preston : ℕ := 55

def growth_rate_leesburg : ℝ := 0.025

def growth_rate_preston : ℝ := 0.035

theorem number_of_women_in_preston_after_one_year : 
  ∀ (preston leesburg : ℕ), 
  preston_is_25_times_leesburg preston leesburg → 
  leesburg = 58940 → 
  (women_percentage_preston : ℝ) / 100 * (preston * (1 + growth_rate_preston) : ℝ) = 838788 :=
by 
  sorry

end number_of_women_in_preston_after_one_year_l826_82615


namespace two_a_plus_two_d_eq_zero_l826_82619

theorem two_a_plus_two_d_eq_zero
  (a b c d : ℝ)
  (h₀ : a ≠ 0)
  (h₁ : b ≠ 0)
  (h₂ : c ≠ 0)
  (h₃ : d ≠ 0)
  (h₄ : ∀ x : ℝ, (2 * a * ((2 * a * x + b) / (3 * c * x + 2 * d)) + b)
                 / (3 * c * ((2 * a * x + b) / (3 * c * x + 2 * d)) + 2 * d) = x) :
  2 * a + 2 * d = 0 :=
by sorry

end two_a_plus_two_d_eq_zero_l826_82619


namespace files_remaining_l826_82604

def initial_music_files : ℕ := 27
def initial_video_files : ℕ := 42
def initial_doc_files : ℕ := 12
def compression_ratio_music : ℕ := 2
def compression_ratio_video : ℕ := 3
def files_deleted : ℕ := 11

def compressed_music_files : ℕ := initial_music_files * compression_ratio_music
def compressed_video_files : ℕ := initial_video_files * compression_ratio_video
def total_compressed_files : ℕ := compressed_music_files + compressed_video_files + initial_doc_files

theorem files_remaining : total_compressed_files - files_deleted = 181 := by
  -- we skip the proof for now
  sorry

end files_remaining_l826_82604


namespace statement_b_statement_e_l826_82637

-- Statement (B): ∀ x, if x^3 > 0 then x > 0.
theorem statement_b (x : ℝ) : x^3 > 0 → x > 0 := sorry

-- Statement (E): ∀ x, if x < 1 then x^3 < x.
theorem statement_e (x : ℝ) : x < 1 → x^3 < x := sorry

end statement_b_statement_e_l826_82637


namespace nancy_soap_bars_l826_82665

def packs : ℕ := 6
def bars_per_pack : ℕ := 5

theorem nancy_soap_bars : packs * bars_per_pack = 30 := by
  sorry

end nancy_soap_bars_l826_82665


namespace profit_percentage_is_correct_l826_82650

noncomputable def CP : ℝ := 47.50
noncomputable def SP : ℝ := 74.21875
noncomputable def MP : ℝ := SP / 0.8
noncomputable def Profit : ℝ := SP - CP
noncomputable def ProfitPercentage : ℝ := (Profit / CP) * 100

theorem profit_percentage_is_correct : ProfitPercentage = 56.25 := by
  -- Proof steps to be filled in
  sorry

end profit_percentage_is_correct_l826_82650


namespace remaining_pie_l826_82652

theorem remaining_pie (carlos_take: ℝ) (sophia_share : ℝ) (final_remaining : ℝ) :
  carlos_take = 0.6 ∧ sophia_share = (1 - carlos_take) / 4 ∧ final_remaining = (1 - carlos_take) - sophia_share →
  final_remaining = 0.3 :=
by
  intros h
  sorry

end remaining_pie_l826_82652


namespace unique_integer_solution_m_l826_82611

theorem unique_integer_solution_m {m : ℤ} (h : ∀ x : ℤ, |2 * x - m| ≤ 1 → x = 2) : m = 4 := 
sorry

end unique_integer_solution_m_l826_82611


namespace lines_parallel_condition_l826_82698

theorem lines_parallel_condition (a : ℝ) : 
  (a = 1) ↔ (∀ x y : ℝ, (a * x + 2 * y - 1 = 0 → x + (a + 1) * y + 4 = 0)) :=
sorry

end lines_parallel_condition_l826_82698


namespace head_start_ratio_l826_82605

variable (Va Vb L H : ℕ)

-- Conditions
def speed_relation : Prop := Va = (4 * Vb) / 3

-- The head start fraction that makes A and B finish the race at the same time given the speed relation
theorem head_start_ratio (Va Vb L H : ℕ)
  (h1 : speed_relation Va Vb)
  (h2 : L > 0) : (H = L / 4) :=
sorry

end head_start_ratio_l826_82605


namespace exterior_angle_measure_l826_82626

theorem exterior_angle_measure (sum_interior_angles : ℝ) (h : sum_interior_angles = 1260) :
  ∃ (n : ℕ) (d : ℝ), (n - 2) * 180 = sum_interior_angles ∧ d = 360 / n ∧ d = 40 := 
by
  sorry

end exterior_angle_measure_l826_82626


namespace quadrant_of_points_l826_82695

theorem quadrant_of_points (x y : ℝ) (h : |3 * x + 2| + |2 * y - 1| = 0) : 
  ((x < 0) ∧ (y > 0) ∧ (x + 1 > 0) ∧ (y - 2 < 0)) :=
by
  sorry

end quadrant_of_points_l826_82695


namespace max_C_trees_l826_82671

theorem max_C_trees 
  (price_A : ℕ) (price_B : ℕ) (price_C : ℕ) (total_price : ℕ) (total_trees : ℕ)
  (h_price_ratio : 2 * price_B = 2 * price_A ∧ 3 * price_A = 2 * price_C)
  (h_price_A : price_A = 200)
  (h_total_price : total_price = 220120)
  (h_total_trees : total_trees = 1000) :
  ∃ (num_C : ℕ), num_C = 201 ∧ ∀ num_C', num_C' > num_C → 
  total_price < price_A * (total_trees - num_C') + price_C * num_C' :=
by
  sorry

end max_C_trees_l826_82671


namespace balloons_lost_l826_82640

theorem balloons_lost (initial remaining : ℕ) (h_initial : initial = 9) (h_remaining : remaining = 7) : initial - remaining = 2 := by
  sorry

end balloons_lost_l826_82640


namespace g_10_44_l826_82685

def g (x y : ℕ) : ℕ := sorry

axiom g_cond1 (x : ℕ) : g x x = x ^ 2
axiom g_cond2 (x y : ℕ) : g x y = g y x
axiom g_cond3 (x y : ℕ) : (x + y) * g x y = y * g x (x + y)

theorem g_10_44 : g 10 44 = 440 := sorry

end g_10_44_l826_82685


namespace simplify_expression_l826_82653

theorem simplify_expression :
  (∃ (a b c d e f : ℝ), 
    a = (7)^(1/4) ∧ 
    b = (3)^(1/3) ∧ 
    c = (7)^(1/2) ∧ 
    d = (3)^(1/6) ∧ 
    e = (a / b) / (c / d) ∧ 
    f = ((1 / 7)^(1/4)) * ((1 / 3)^(1/6))
    → e = f) :=
by {
  sorry
}

end simplify_expression_l826_82653


namespace factorize_expr_l826_82683

theorem factorize_expr (x y : ℝ) : 3 * x^2 + 6 * x * y + 3 * y^2 = 3 * (x + y)^2 := 
  sorry

end factorize_expr_l826_82683
