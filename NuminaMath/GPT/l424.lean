import Mathlib

namespace solve_for_x_l424_42486

theorem solve_for_x (h : 125 = 5 ^ 3) : ∃ x : ℕ, 125 ^ 4 = 5 ^ x ∧ x = 12 := by
  sorry

end solve_for_x_l424_42486


namespace units_digit_of_expression_l424_42497

theorem units_digit_of_expression :
  (3 * 19 * 1981 - 3^4) % 10 = 6 :=
sorry

end units_digit_of_expression_l424_42497


namespace arithmetic_seq_a8_l424_42463

def is_arith_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_seq_a8
  (a : ℕ → ℤ)
  (h_arith : is_arith_seq a)
  (h_a2 : a 2 = 2)
  (h_a4 : a 4 = 6) :
  a 8 = 14 := sorry

end arithmetic_seq_a8_l424_42463


namespace individual_max_food_l424_42407

/-- Given a minimum number of guests and a total amount of food consumed,
    we want to find the maximum amount of food an individual guest could have consumed. -/
def total_food : ℝ := 319
def min_guests : ℝ := 160
def max_food_per_guest : ℝ := 1.99

theorem individual_max_food :
  total_food / min_guests <= max_food_per_guest := by
  sorry

end individual_max_food_l424_42407


namespace feeding_sequences_count_l424_42423

def num_feeding_sequences (num_pairs : ℕ) : ℕ :=
  num_pairs * num_pairs.pred * num_pairs.pred * num_pairs.pred.pred *
  num_pairs.pred.pred * num_pairs.pred.pred.pred * num_pairs.pred.pred.pred *
  1 * 1

theorem feeding_sequences_count (num_pairs : ℕ) (h : num_pairs = 5) :
  num_feeding_sequences num_pairs = 5760 := 
by
  rw [h]
  unfold num_feeding_sequences
  norm_num
  sorry

end feeding_sequences_count_l424_42423


namespace largest_angle_in_isosceles_triangle_l424_42446

-- Definitions of the conditions from the problem
def isosceles_triangle (A B C : ℕ) : Prop :=
  A = B ∨ B = C ∨ A = C

def angle_opposite_equal_side (θ : ℕ) : Prop :=
  θ = 50

-- The proof problem statement
theorem largest_angle_in_isosceles_triangle (A B C : ℕ) (θ : ℕ)
  : isosceles_triangle A B C → angle_opposite_equal_side θ → ∃ γ, γ = 80 :=
by
  sorry

end largest_angle_in_isosceles_triangle_l424_42446


namespace length_of_arc_l424_42458

theorem length_of_arc (angle_SIT : ℝ) (radius_OS : ℝ) (h1 : angle_SIT = 45) (h2 : radius_OS = 15) :
  arc_length_SIT = 7.5 * Real.pi :=
by
  sorry

end length_of_arc_l424_42458


namespace population_growth_l424_42467

theorem population_growth (P : ℝ) (x : ℝ) (y : ℝ) 
  (h₁ : P = 5.48) 
  (h₂ : y = P * (1 + x / 100)^8) : 
  y = 5.48 * (1 + x / 100)^8 := 
by
  sorry

end population_growth_l424_42467


namespace range_of_a_l424_42487

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x + 4 < 0) ↔ a < -4 ∨ a > 4 :=
by
  sorry

end range_of_a_l424_42487


namespace real_solutions_l424_42455

theorem real_solutions (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 4) (h3 : x ≠ 5) :
  ( (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 3) * (x - 2) * (x - 1) ) / 
  ( (x - 2) * (x - 4) * (x - 5) * (x - 2) ) = 1 
  ↔ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2 :=
by sorry

end real_solutions_l424_42455


namespace roommate_payment_l424_42411

theorem roommate_payment :
  (1100 + 114 + 300) / 2 = 757 := 
by
  sorry

end roommate_payment_l424_42411


namespace perfect_square_transformation_l424_42408

theorem perfect_square_transformation (a : ℤ) :
  (∃ x y : ℤ, x^2 + a = y^2) ↔ 
  ∃ α β : ℤ, α * β = a ∧ (α % 2 = β % 2) ∧ 
  ∃ x y : ℤ, x = (β - α) / 2 ∧ y = (β + α) / 2 :=
by
  sorry

end perfect_square_transformation_l424_42408


namespace iris_jackets_l424_42414

theorem iris_jackets (J : ℕ) (h : 10 * J + 12 + 48 = 90) : J = 3 :=
by
  sorry

end iris_jackets_l424_42414


namespace jello_cost_calculation_l424_42454

-- Conditions as definitions
def jello_per_pound : ℝ := 1.5
def tub_volume_cubic_feet : ℝ := 6
def cubic_foot_to_gallons : ℝ := 7.5
def gallon_weight_pounds : ℝ := 8
def cost_per_tablespoon_jello : ℝ := 0.5

-- Tub total water calculation
def tub_water_gallons (volume_cubic_feet : ℝ) (cubic_foot_to_gallons : ℝ) : ℝ :=
  volume_cubic_feet * cubic_foot_to_gallons

-- Water weight calculation
def water_weight_pounds (water_gallons : ℝ) (gallon_weight_pounds : ℝ) : ℝ :=
  water_gallons * gallon_weight_pounds

-- Jello mix required calculation
def jello_mix_tablespoons (water_pounds : ℝ) (jello_per_pound : ℝ) : ℝ :=
  water_pounds * jello_per_pound

-- Total cost calculation
def total_cost (jello_mix_tablespoons : ℝ) (cost_per_tablespoon_jello : ℝ) : ℝ :=
  jello_mix_tablespoons * cost_per_tablespoon_jello

-- Theorem statement
theorem jello_cost_calculation :
  total_cost (jello_mix_tablespoons (water_weight_pounds (tub_water_gallons tub_volume_cubic_feet cubic_foot_to_gallons) gallon_weight_pounds) jello_per_pound) cost_per_tablespoon_jello = 270 := 
by sorry

end jello_cost_calculation_l424_42454


namespace total_sum_of_money_l424_42488

theorem total_sum_of_money (x : ℝ) (A B C : ℝ) 
  (hA : A = x) 
  (hB : B = 0.65 * x) 
  (hC : C = 0.40 * x) 
  (hC_share : C = 32) :
  A + B + C = 164 := 
  sorry

end total_sum_of_money_l424_42488


namespace shadow_length_of_flagpole_l424_42425

theorem shadow_length_of_flagpole :
  ∀ (S : ℝ), (18 : ℝ) / S = (22 : ℝ) / 55 → S = 45 :=
by
  intro S h
  sorry

end shadow_length_of_flagpole_l424_42425


namespace fabric_ratio_l424_42461

theorem fabric_ratio
  (d_m : ℕ) (d_t : ℕ) (d_w : ℕ) (cost : ℕ) (total_revenue : ℕ) (revenue_monday : ℕ) (revenue_tuesday : ℕ) (revenue_wednesday : ℕ)
  (h_d_m : d_m = 20)
  (h_cost : cost = 2)
  (h_d_w : d_w = d_t / 4)
  (h_total_revenue : total_revenue = 140)
  (h_revenue : revenue_monday + revenue_tuesday + revenue_wednesday = total_revenue)
  (h_r_m : revenue_monday = d_m * cost)
  (h_r_t : revenue_tuesday = d_t * cost) 
  (h_r_w : revenue_wednesday = d_w * cost) :
  (d_t / d_m = 1) :=
by
  sorry

end fabric_ratio_l424_42461


namespace Davante_boys_count_l424_42413

def days_in_week := 7
def friends (days : Nat) := days * 2
def girls := 3
def boys (total_friends girls : Nat) := total_friends - girls

theorem Davante_boys_count :
  boys (friends days_in_week) girls = 11 :=
  by
    sorry

end Davante_boys_count_l424_42413


namespace arrangement_count_l424_42495

def arrangements_with_conditions 
  (boys girls : Nat) 
  (cannot_be_next_to_each_other : Bool) : Nat :=
if cannot_be_next_to_each_other then
  sorry -- The proof will go here
else
  sorry

theorem arrangement_count :
  arrangements_with_conditions 3 2 true = 72 :=
sorry

end arrangement_count_l424_42495


namespace total_votes_cast_l424_42440

theorem total_votes_cast (b_votes c_votes total_votes : ℕ)
  (h1 : b_votes = 48)
  (h2 : c_votes = 35)
  (h3 : b_votes = (4 * total_votes) / 15) :
  total_votes = 180 :=
by
  sorry

end total_votes_cast_l424_42440


namespace trigonometric_identity_l424_42442

variable {α : Real}
variable (h : Real.cos α = -2 / 3)

theorem trigonometric_identity : 
  (Real.cos α = -2 / 3) → 
  (Real.cos (4 * Real.pi - α) * Real.sin (-α) / 
  (Real.sin (Real.pi / 2 + α) * Real.tan (Real.pi - α)) = Real.cos α) :=
by
  intro h
  sorry

end trigonometric_identity_l424_42442


namespace Sam_total_books_l424_42434

/-- Sam's book purchases -/
def Sam_bought_books : Real := 
  let used_adventure_books := 13.0
  let used_mystery_books := 17.0
  let new_crime_books := 15.0
  used_adventure_books + used_mystery_books + new_crime_books

theorem Sam_total_books : Sam_bought_books = 45.0 :=
by
  -- The proof will show that Sam indeed bought 45 books in total
  sorry

end Sam_total_books_l424_42434


namespace base_7_perfect_square_ab2c_l424_42445

-- Define the necessary conditions
def is_base_7_representation_of (n : ℕ) (a b c : ℕ) : Prop :=
  n = a * 7^3 + b * 7^2 + 2 * 7 + c

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Lean statement for the problem
theorem base_7_perfect_square_ab2c (n a b c : ℕ) (h1 : a ≠ 0) (h2 : is_base_7_representation_of n a b c) (h3 : is_perfect_square n) :
  c = 2 ∨ c = 3 ∨ c = 6 :=
  sorry

end base_7_perfect_square_ab2c_l424_42445


namespace cos_even_function_l424_42438

theorem cos_even_function : ∀ x : ℝ, Real.cos (-x) = Real.cos x := 
by 
  sorry

end cos_even_function_l424_42438


namespace bob_total_calories_l424_42471

def total_calories (slices_300 : ℕ) (calories_300 : ℕ) (slices_400 : ℕ) (calories_400 : ℕ) : ℕ :=
  slices_300 * calories_300 + slices_400 * calories_400

theorem bob_total_calories 
  (slices_300 : ℕ := 3)
  (calories_300 : ℕ := 300)
  (slices_400 : ℕ := 4)
  (calories_400 : ℕ := 400) :
  total_calories slices_300 calories_300 slices_400 calories_400 = 2500 := 
by 
  sorry

end bob_total_calories_l424_42471


namespace algebraic_expression_value_l424_42465

theorem algebraic_expression_value (a b c d m : ℝ) (h1 : a + b = 0) (h2 : c * d = 1) (h3 : m ^ 2 = 25) :
  m^2 - 100*a - 99*b - b*c*d + |c*d - 2| = -74 :=
by
  sorry

end algebraic_expression_value_l424_42465


namespace expected_heads_of_fair_coin_l424_42420

noncomputable def expected_heads (n : ℕ) (p : ℝ) : ℝ := n * p

theorem expected_heads_of_fair_coin :
  expected_heads 5 0.5 = 2.5 :=
by
  sorry

end expected_heads_of_fair_coin_l424_42420


namespace sara_initial_black_marbles_l424_42490

-- Define the given conditions
def red_marbles (sara_has : Nat) : Prop := sara_has = 122
def black_marbles_taken_by_fred (fred_took : Nat) : Prop := fred_took = 233
def black_marbles_now (sara_has_now : Nat) : Prop := sara_has_now = 559

-- The proof problem statement
theorem sara_initial_black_marbles
  (sara_has_red : ∀ n : Nat, red_marbles n)
  (fred_took_marbles : ∀ f : Nat, black_marbles_taken_by_fred f)
  (sara_has_now_black : ∀ b : Nat, black_marbles_now b) :
  ∃ b, b = 559 + 233 :=
by
  sorry

end sara_initial_black_marbles_l424_42490


namespace find_larger_number_l424_42448

theorem find_larger_number (x y : ℤ) (h1 : 4 * y = 3 * x) (h2 : y - x = 12) : y = -36 := 
by sorry

end find_larger_number_l424_42448


namespace smallest_N_l424_42430

noncomputable def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n
  
noncomputable def is_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

noncomputable def is_fifth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k ^ 5 = n

theorem smallest_N :
  ∃ N : ℕ, is_square (N / 2) ∧ is_cube (N / 3) ∧ is_fifth_power (N / 5) ∧
  N = 2^15 * 3^10 * 5^6 :=
by
  exists 2^15 * 3^10 * 5^6
  sorry

end smallest_N_l424_42430


namespace solve_for_x_l424_42424

theorem solve_for_x (x : ℝ) (h : (9 + 1/x)^(1/3) = -2) : x = -1/17 :=
by
  sorry

end solve_for_x_l424_42424


namespace weight_conversion_l424_42468

theorem weight_conversion (a b : ℝ) (conversion_rate : ℝ) : a = 3600 → b = 600 → conversion_rate = 1000 → (a - b) / conversion_rate = 3 := 
by
  intros h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  sorry

end weight_conversion_l424_42468


namespace inequality_proof_l424_42418

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y > 2) : (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
sorry

end inequality_proof_l424_42418


namespace find_p_over_q_l424_42464

variables (x y p q : ℚ)

theorem find_p_over_q (h1 : (7 * x + 6 * y) / (x - 2 * y) = 27)
                      (h2 : x / (2 * y) = p / q) :
                      p / q = 3 / 2 :=
sorry

end find_p_over_q_l424_42464


namespace difference_of_squares_l424_42496

theorem difference_of_squares (x : ℤ) (h : x^2 = 1521) : (x + 1) * (x - 1) = 1520 := by
  sorry

end difference_of_squares_l424_42496


namespace geometric_sequence_b_value_l424_42416

noncomputable def b_value (b : ℝ) : Prop :=
  ∃ s : ℝ, 180 * s = b ∧ b * s = 75 / 32 ∧ b > 0

theorem geometric_sequence_b_value (b : ℝ) : b_value b → b = 20.542 :=
by
  sorry

end geometric_sequence_b_value_l424_42416


namespace volume_inside_sphere_outside_cylinder_l424_42404

noncomputable def sphere_radius := 6
noncomputable def cylinder_diameter := 8
noncomputable def sphere_volume := 4/3 * Real.pi * (sphere_radius ^ 3)
noncomputable def cylinder_height := Real.sqrt ((sphere_radius * 2) ^ 2 - (cylinder_diameter) ^ 2)
noncomputable def cylinder_volume := Real.pi * ((cylinder_diameter / 2) ^ 2) * cylinder_height
noncomputable def volume_difference := sphere_volume - cylinder_volume

theorem volume_inside_sphere_outside_cylinder:
  volume_difference = (288 - 64 * Real.sqrt 5) * Real.pi :=
sorry

end volume_inside_sphere_outside_cylinder_l424_42404


namespace books_received_l424_42482

theorem books_received (students : ℕ) (books_per_student : ℕ) (books_fewer : ℕ) (expected_books : ℕ) (received_books : ℕ) :
  students = 20 →
  books_per_student = 15 →
  books_fewer = 6 →
  expected_books = students * books_per_student →
  received_books = expected_books - books_fewer →
  received_books = 294 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end books_received_l424_42482


namespace train_length_l424_42415

theorem train_length (L S : ℝ) 
  (h1 : L = S * 15) 
  (h2 : L + 100 = S * 25) : 
  L = 150 :=
by
  sorry

end train_length_l424_42415


namespace x_less_than_2_necessary_not_sufficient_x_less_than_2_is_necessary_not_sufficient_l424_42419

theorem x_less_than_2_necessary_not_sufficient (x : ℝ) :
  (x^2 - 3 * x + 2 < 0) ↔ (1 < x ∧ x < 2) := sorry

theorem x_less_than_2_is_necessary_not_sufficient : 
  (∀ x : ℝ, x^2 - 3*x + 2 < 0 → x < 2) ∧ 
  (¬ ∀ x : ℝ, x < 2 → x^2 - 3*x + 2 < 0) := sorry

end x_less_than_2_necessary_not_sufficient_x_less_than_2_is_necessary_not_sufficient_l424_42419


namespace determine_c_l424_42492

theorem determine_c (c : ℝ) 
  (h : ∃ a : ℝ, (∀ x : ℝ, x^2 + 200 * x + c = (x + a)^2)) : c = 10000 :=
sorry

end determine_c_l424_42492


namespace principal_amount_l424_42491

-- Define the conditions and required result
theorem principal_amount
  (P R T : ℝ)
  (hR : R = 0.5)
  (h_diff : (P * R * (T + 4) / 100) - (P * R * T / 100) = 40) :
  P = 2000 :=
  sorry

end principal_amount_l424_42491


namespace factorable_iff_some_even_b_l424_42450

open Int

theorem factorable_iff_some_even_b (b : ℤ) :
  (∃ m n p q : ℤ,
    (35 : ℤ) = m * p ∧
    (35 : ℤ) = n * q ∧
    b = m * q + n * p) →
  (∃ k : ℤ, b = 2 * k) :=
by
  sorry

end factorable_iff_some_even_b_l424_42450


namespace polygon_sides_l424_42444

theorem polygon_sides (n : ℕ) (h : 3 * n * (n * (n - 3)) = 300) : n = 10 :=
sorry

end polygon_sides_l424_42444


namespace domain_of_sqrt_function_l424_42439

theorem domain_of_sqrt_function : {x : ℝ | 0 ≤ x ∧ x ≤ 1} = {x : ℝ | 1 - x ≥ 0 ∧ x - Real.sqrt (1 - x) ≥ 0} :=
by
  sorry

end domain_of_sqrt_function_l424_42439


namespace inequality_solution_l424_42402

variable {α : Type*} [LinearOrderedField α]
variable (a b x : α)

theorem inequality_solution (h1 : a < 0) (h2 : b = -a) :
  0 < x ∧ x < 1 ↔ ax^2 + bx > 0 :=
by sorry

end inequality_solution_l424_42402


namespace find_f_2009_l424_42435

noncomputable def f : ℝ → ℝ := sorry

axiom cond1 : ∀ x : ℝ, f x * f (x + 2) = 13
axiom cond2 : f 1 = 2

theorem find_f_2009 : f 2009 = 2 := by
  sorry

end find_f_2009_l424_42435


namespace quadratic_condition_l424_42485

variables {c y1 y2 y3 : ℝ}

/-- Points P1(-1, y1), P2(3, y2), P3(5, y3) are all on the graph of the quadratic function y = -x^2 + 2x + c. --/
def points_on_parabola (y1 y2 y3 c : ℝ) : Prop :=
  y1 = -(-1)^2 + 2*(-1) + c ∧
  y2 = -(3)^2 + 2*(3) + c ∧
  y3 = -(5)^2 + 2*(5) + c

/-- The quadratic function y = -x^2 + 2x + c has an axis of symmetry at x = 1 and opens downwards. --/
theorem quadratic_condition (h : points_on_parabola y1 y2 y3 c) : 
  y1 = y2 ∧ y2 > y3 :=
sorry

end quadratic_condition_l424_42485


namespace initial_pretzels_in_bowl_l424_42466

-- Definitions and conditions
def John_pretzels := 28
def Alan_pretzels := John_pretzels - 9
def Marcus_pretzels := John_pretzels + 12
def Marcus_pretzels_actual := 40

-- The main theorem stating the initial number of pretzels in the bowl
theorem initial_pretzels_in_bowl : 
  Marcus_pretzels = Marcus_pretzels_actual → 
  John_pretzels + Alan_pretzels + Marcus_pretzels = 87 :=
by
  intro h
  sorry -- proof to be filled in

end initial_pretzels_in_bowl_l424_42466


namespace volunteer_comprehensive_score_l424_42401

theorem volunteer_comprehensive_score :
  let written_score := 90
  let trial_score := 94
  let interview_score := 92
  let written_weight := 0.30
  let trial_weight := 0.50
  let interview_weight := 0.20
  (written_score * written_weight + trial_score * trial_weight + interview_score * interview_weight = 92.4) := by
  sorry

end volunteer_comprehensive_score_l424_42401


namespace solve_equation_l424_42406

-- Define the conditions of the problem.
def equation (x : ℝ) : Prop := (5 - x / 3)^(1/3) = -2

-- Define the main theorem to prove that x = 39 is the solution to the equation.
theorem solve_equation : ∃ x : ℝ, equation x ∧ x = 39 :=
by
  existsi 39
  intros
  simp [equation]
  sorry

end solve_equation_l424_42406


namespace compare_negative_fractions_l424_42410

theorem compare_negative_fractions : (- (1 / 3 : ℝ)) < (- (1 / 4 : ℝ)) :=
sorry

end compare_negative_fractions_l424_42410


namespace other_number_more_than_42_l424_42481

theorem other_number_more_than_42 (a b : ℕ) (h1 : a + b = 96) (h2 : a = 42) : b - a = 12 := by
  sorry

end other_number_more_than_42_l424_42481


namespace sphere_volume_increase_factor_l424_42498

theorem sphere_volume_increase_factor (r : Real) : 
  let V_original := (4 / 3) * Real.pi * r^3
  let V_increased := (4 / 3) * Real.pi * (2 * r)^3
  V_increased / V_original = 8 :=
by
  -- Definitions of volumes
  let V_original := (4 / 3) * Real.pi * r^3
  let V_increased := (4 / 3) * Real.pi * (2 * r)^3
  -- Volume ratio
  have h : V_increased / V_original = 8 := sorry
  exact h

end sphere_volume_increase_factor_l424_42498


namespace find_the_number_l424_42433

theorem find_the_number : ∃ x : ℝ, (10 + x + 50) / 3 = (20 + 40 + 6) / 3 + 8 ∧ x = 30 := 
by
  sorry

end find_the_number_l424_42433


namespace range_of_m_l424_42451

def sufficient_condition (x m : ℝ) : Prop :=
  m - 1 < x ∧ x < m + 1

def inequality (x : ℝ) : Prop :=
  x^2 - 2 * x - 3 > 0

theorem range_of_m (m : ℝ) :
  (∀ x, sufficient_condition x m → inequality x) ↔ (m ≤ -2 ∨ m ≥ 4) :=
by 
  sorry

end range_of_m_l424_42451


namespace probability_mask_with_ear_loops_l424_42427

-- Definitions from the conditions
def production_ratio_regular : ℝ := 0.8
def production_ratio_surgical : ℝ := 0.2
def proportion_ear_loops_regular : ℝ := 0.1
def proportion_ear_loops_surgical : ℝ := 0.2

-- Theorem statement based on the translated proof problem
theorem probability_mask_with_ear_loops :
  production_ratio_regular * proportion_ear_loops_regular +
  production_ratio_surgical * proportion_ear_loops_surgical = 0.12 :=
by
  -- Proof omitted
  sorry

end probability_mask_with_ear_loops_l424_42427


namespace allocation_first_grade_places_l424_42483

theorem allocation_first_grade_places (total_students : ℕ)
                                      (ratio_1 : ℕ)
                                      (ratio_2 : ℕ)
                                      (ratio_3 : ℕ)
                                      (total_places : ℕ) :
  total_students = 160 →
  ratio_1 = 6 →
  ratio_2 = 5 →
  ratio_3 = 5 →
  total_places = 160 →
  (total_places * ratio_1) / (ratio_1 + ratio_2 + ratio_3) = 60 :=
sorry

end allocation_first_grade_places_l424_42483


namespace a_squared_plus_b_squared_eq_sqrt_11_l424_42436

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom h_pos_a : a > 0
axiom h_pos_b : b > 0
axiom h_condition : a * b * (a - b) = 1

theorem a_squared_plus_b_squared_eq_sqrt_11 : a^2 + b^2 = Real.sqrt 11 := by
  sorry

end a_squared_plus_b_squared_eq_sqrt_11_l424_42436


namespace missing_dimension_of_carton_l424_42449

theorem missing_dimension_of_carton (x : ℕ) 
  (h1 : 0 < x)
  (h2 : 0 < 48)
  (h3 : 0 < 60)
  (h4 : 0 < 8)
  (h5 : 0 < 6)
  (h6 : 0 < 5)
  (h7 : (x * 48 * 60) / (8 * 6 * 5) = 300) : 
  x = 25 :=
by
  sorry

end missing_dimension_of_carton_l424_42449


namespace part1_part2_l424_42474

open Set

-- Define the sets M and N based on given conditions
def M (a : ℝ) : Set ℝ := { x | (x + a) * (x - 1) ≤ 0 }
def N : Set ℝ := { x | 4 * x^2 - 4 * x - 3 < 0 }

-- Part (1): Prove that if M ∪ N = { x | -2 ≤ x < 3 / 2 }, then a = 2
theorem part1 (a : ℝ) (h : a > 0)
  (h_union : M a ∪ N = { x | -2 ≤ x ∧ x < 3 / 2 }) : a = 2 := by
  sorry

-- Part (2): Prove that if N ∪ (compl (M a)) = univ, then 0 < a ≤ 1/2
theorem part2 (a : ℝ) (h : a > 0)
  (h_union : N ∪ compl (M a) = univ) : 0 < a ∧ a ≤ 1 / 2 := by
  sorry

end part1_part2_l424_42474


namespace average_marks_l424_42480

theorem average_marks
  (M P C : ℕ)
  (h1 : M + P = 70)
  (h2 : C = P + 20) :
  (M + C) / 2 = 45 :=
sorry

end average_marks_l424_42480


namespace log_inequality_l424_42417

theorem log_inequality (a x y : ℝ) (ha : 0 < a) (ha_lt_1 : a < 1) 
(h : x^2 + y = 0) : 
  Real.log (a^x + a^y) / Real.log a ≤ Real.log 2 / Real.log a + 1 / 8 :=
sorry

end log_inequality_l424_42417


namespace Pam_has_740_fruits_l424_42405

/-
Define the given conditions.
-/
def Gerald_apple_bags : ℕ := 5
def apples_per_Gerald_bag : ℕ := 30
def Gerald_orange_bags : ℕ := 4
def oranges_per_Gerald_bag : ℕ := 25

def Pam_apple_bags : ℕ := 6
def apples_per_Pam_bag : ℕ := 3 * apples_per_Gerald_bag
def Pam_orange_bags : ℕ := 4
def oranges_per_Pam_bag : ℕ := 2 * oranges_per_Gerald_bag

/-
Proving the total number of apples and oranges Pam has.
-/
def total_fruits_Pam : ℕ :=
    Pam_apple_bags * apples_per_Pam_bag + Pam_orange_bags * oranges_per_Pam_bag

theorem Pam_has_740_fruits : total_fruits_Pam = 740 := by
  sorry

end Pam_has_740_fruits_l424_42405


namespace symmetric_line_equation_l424_42452

theorem symmetric_line_equation :
  (∃ l : ℝ × ℝ × ℝ, (∀ x₁ y₁ x₂ y₂ : ℝ, 
    x₁ + x₂ = -4 → y₁ + y₂ = 2 → 
    ∃ a b c : ℝ, l = (a, b, c) ∧ x₁ * a + y₁ * b + c = 0 ∧ x₂ * a + y₂ * b + c = 0) → 
  l = (2, -1, 5)) :=
sorry

end symmetric_line_equation_l424_42452


namespace max_students_per_class_l424_42460

theorem max_students_per_class (num_students : ℕ) (seats_per_bus : ℕ) (num_buses : ℕ) (k : ℕ) 
  (h_num_students : num_students = 920) 
  (h_seats_per_bus : seats_per_bus = 71) 
  (h_num_buses : num_buses = 16) 
  (h_class_size_bound : ∀ c, c ≤ k) : 
  k = 17 :=
sorry

end max_students_per_class_l424_42460


namespace larger_number_is_23_l424_42456

theorem larger_number_is_23 (a b : ℕ) (h1 : a + b = 40) (h2 : a - b = 6) : a = 23 := 
by
  sorry

end larger_number_is_23_l424_42456


namespace solution_set_of_inequality_l424_42499

theorem solution_set_of_inequality :
  { x : ℝ | (x - 1) / x ≥ 2 } = { x : ℝ | -1 ≤ x ∧ x < 0 } :=
by
  sorry

end solution_set_of_inequality_l424_42499


namespace value_of_a_l424_42494

theorem value_of_a (a : ℝ) :
  (∀ x, (2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5) → (3 ≤ x ∧ x ≤ 22)) ↔ (6 ≤ a ∧ a ≤ 9) :=
by
  sorry

end value_of_a_l424_42494


namespace problem_1_problem_2_l424_42459

open Real

noncomputable def vec_a (θ : ℝ) : ℝ × ℝ :=
( sin θ, cos θ - 2 * sin θ )

def vec_b : ℝ × ℝ :=
( 1, 2 )

theorem problem_1 (θ : ℝ) (h : (cos θ - 2 * sin θ) / sin θ = 2) : tan θ = 1 / 4 :=
by {
  sorry
}

theorem problem_2 (θ : ℝ) (h1 : sin θ ^ 2 + (cos θ - 2 * sin θ) ^ 2 = 5) (h2 : 0 < θ) (h3 : θ < π) : θ = π / 2 ∨ θ = 3 * π / 4 :=
by {
  sorry
}

end problem_1_problem_2_l424_42459


namespace solution_set_of_inequality_l424_42403

theorem solution_set_of_inequality :
  { x : ℝ | -x^2 + 2*x + 3 ≥ 0 } = { x : ℝ | -1 ≤ x ∧ x ≤ 3 } :=
sorry

end solution_set_of_inequality_l424_42403


namespace triangle_ABC_area_l424_42475

def point := (ℝ × ℝ)

def A : point := (0, 0)
def B : point := (1, 2)
def C : point := (2, 0)

def triangle_area (A B C : point) : ℝ :=
  0.5 * |(A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))|

theorem triangle_ABC_area :
  triangle_area A B C = 2 :=
by
  sorry

end triangle_ABC_area_l424_42475


namespace determine_a_and_b_l424_42469

variable (a b : ℕ)
theorem determine_a_and_b 
  (h1: 0 ≤ a ∧ a ≤ 9) 
  (h2: 0 ≤ b ∧ b ≤ 9)
  (h3: (a + b + 45) % 9 = 0)
  (h4: (b - a) % 11 = 3) : 
  a = 3 ∧ b = 6 :=
sorry

end determine_a_and_b_l424_42469


namespace hospital_staff_l424_42473

-- Define the conditions
variables (d n : ℕ) -- d: number of doctors, n: number of nurses
variables (x : ℕ) -- common multiplier

theorem hospital_staff (h1 : d + n = 456) (h2 : 8 * x = d) (h3 : 11 * x = n) : n = 264 :=
by
  -- noncomputable def only when necessary, skipping the proof with sorry
  sorry

end hospital_staff_l424_42473


namespace area_of_triangle_l424_42484

theorem area_of_triangle (a c : ℝ) (A : ℝ) (h_a : a = 2) (h_c : c = 2 * Real.sqrt 3) (h_A : A = Real.pi / 6) :
  ∃ (area : ℝ), area = 2 * Real.sqrt 3 ∨ area = Real.sqrt 3 :=
by
  sorry

end area_of_triangle_l424_42484


namespace max_xy_is_2_min_y_over_x_plus_4_over_y_is_4_l424_42441

noncomputable def max_xy (x y : ℝ) : ℝ :=
if h : x > 0 ∧ y > 0 ∧ x + 2 * y = 4 then x * y else 0

noncomputable def min_y_over_x_plus_4_over_y (x y : ℝ) : ℝ :=
if h : x > 0 ∧ y > 0 ∧ x + 2 * y = 4 then y / x + 4 / y else 0

theorem max_xy_is_2 : ∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = 4 → max_xy x y = 2 :=
by
  intros x y hx hy hxy
  sorry

theorem min_y_over_x_plus_4_over_y_is_4 : ∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = 4 → min_y_over_x_plus_4_over_y x y = 4 :=
by
  intros x y hx hy hxy
  sorry

end max_xy_is_2_min_y_over_x_plus_4_over_y_is_4_l424_42441


namespace determine_k_l424_42431

theorem determine_k (k : ℝ) : 
  (∀ x : ℝ, (x^2 = 2 * x + k) → (∃ x0 : ℝ, ∀ x : ℝ, (x - x0)^2 = 0)) ↔ k = -1 :=
by 
  sorry

end determine_k_l424_42431


namespace rectangle_width_l424_42421

theorem rectangle_width (width : ℝ) : 
  ∃ w, w = 14 ∧
  (∀ length : ℝ, length = 10 →
  (2 * (length + width) = 3 * 16)) → 
  width = w :=
by
  sorry

end rectangle_width_l424_42421


namespace minimize_q_l424_42426

noncomputable def q (x : ℝ) : ℝ := (x - 5)^2 + (x + 1)^2 - 6

theorem minimize_q : ∃ x : ℝ, q x = 2 :=
by
  sorry

end minimize_q_l424_42426


namespace sum_faces_edges_vertices_triangular_prism_l424_42432

-- Given conditions for triangular prism:
def triangular_prism_faces : Nat := 2 + 3  -- 2 triangular faces and 3 rectangular faces
def triangular_prism_edges : Nat := 3 + 3 + 3  -- 3 top edges, 3 bottom edges, 3 connecting edges
def triangular_prism_vertices : Nat := 3 + 3  -- 3 vertices on the top base, 3 on the bottom base

-- Proof statement for the sum of the faces, edges, and vertices of a triangular prism
theorem sum_faces_edges_vertices_triangular_prism : 
  triangular_prism_faces + triangular_prism_edges + triangular_prism_vertices = 20 := by
  sorry

end sum_faces_edges_vertices_triangular_prism_l424_42432


namespace parabola_expression_correct_area_triangle_ABM_correct_l424_42412

-- Given conditions
def pointA : ℝ × ℝ := (-1, 0)
def pointB : ℝ × ℝ := (3, 0)
def pointC : ℝ × ℝ := (0, 3)

-- Analytical expression of the parabola as y = -x^2 + 2x + 3
def parabola_eqn (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Definition of the vertex M of the parabola (derived from calculations)
def vertexM : ℝ × ℝ := (1, 4)

-- Calculation of distance AB
def distance_AB : ℝ := 4

-- Calculation of area of triangle ABM
def triangle_area_ABM : ℝ := 8

theorem parabola_expression_correct :
  (∀ x y, (y = parabola_eqn x ↔ (parabola_eqn x = y))) ∧
  (parabola_eqn pointC.1 = pointC.2) :=
by
  sorry

theorem area_triangle_ABM_correct :
  (1 / 2 * distance_AB * vertexM.2 = 8) :=
by
  sorry

end parabola_expression_correct_area_triangle_ABM_correct_l424_42412


namespace find_present_worth_l424_42477

noncomputable def present_worth (BG : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
(BG * 100) / (R * ((1 + R/100)^T - 1) - R * T)

theorem find_present_worth : present_worth 36 10 3 = 1161.29 :=
by
  sorry

end find_present_worth_l424_42477


namespace find_m_l424_42479

theorem find_m (m : ℝ) :
  (∃ m : ℝ, ∀ x y : ℝ, x + y - m = 0 ∧ x + (3 - 2 * m) * y = 0 → 
     (m = 1)) := 
sorry

end find_m_l424_42479


namespace solve_for_x_l424_42476

/-- Given condition that 0.75 : x :: 5 : 9 -/
def ratio_condition (x : ℝ) : Prop := 0.75 / x = 5 / 9

theorem solve_for_x (x : ℝ) (h : ratio_condition x) : x = 1.35 := by
  sorry

end solve_for_x_l424_42476


namespace candy_bars_weeks_l424_42437

theorem candy_bars_weeks (buy_per_week : ℕ) (eat_per_4_weeks : ℕ) (saved_candies : ℕ) (weeks_passed : ℕ) :
  (buy_per_week = 2) →
  (eat_per_4_weeks = 1) →
  (saved_candies = 28) →
  (weeks_passed = 4 * (saved_candies / (4 * buy_per_week - eat_per_4_weeks))) →
  weeks_passed = 16 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end candy_bars_weeks_l424_42437


namespace boats_distance_one_minute_before_collision_l424_42447

noncomputable def distance_between_boats_one_minute_before_collision
  (speed_boat1 : ℝ) (speed_boat2 : ℝ) (initial_distance : ℝ) : ℝ :=
  let relative_speed := speed_boat1 + speed_boat2
  let relative_speed_per_minute := relative_speed / 60
  let time_to_collide := initial_distance / relative_speed_per_minute
  let distance_one_minute_before := initial_distance - (relative_speed_per_minute * (time_to_collide - 1))
  distance_one_minute_before

theorem boats_distance_one_minute_before_collision :
  distance_between_boats_one_minute_before_collision 5 21 20 = 0.4333 :=
by
  -- Proof skipped
  sorry

end boats_distance_one_minute_before_collision_l424_42447


namespace daily_pre_promotion_hours_l424_42400

-- Defining conditions
def weekly_additional_hours := 6
def hours_driven_in_two_weeks_after_promotion := 40
def days_in_two_weeks := 14
def hours_added_in_two_weeks := 2 * weekly_additional_hours

-- Math proof problem statement
theorem daily_pre_promotion_hours :
  (hours_driven_in_two_weeks_after_promotion - hours_added_in_two_weeks) / days_in_two_weeks = 2 :=
by
  sorry

end daily_pre_promotion_hours_l424_42400


namespace minimum_value_fraction_l424_42422

noncomputable def log (a x : ℝ) : ℝ := Real.log x / Real.log a

/-- Given that the function f(x) = log_a(4x-3) + 1 (where a > 0 and a ≠ 1) has a fixed point A(m, n), 
if for any positive numbers x and y, mx + ny = 3, 
then the minimum value of 1/(x+1) + 1/y is 1. -/
theorem minimum_value_fraction (a x y : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) (hx : x + y = 3) : 
  (1 / (x + 1) + 1 / y) = 1 := 
sorry

end minimum_value_fraction_l424_42422


namespace weeks_to_fill_moneybox_l424_42478

-- Monica saves $15 every week
def savings_per_week : ℕ := 15

-- Number of cycles Monica repeats
def cycles : ℕ := 5

-- Total amount taken to the bank
def total_savings : ℕ := 4500

-- Prove that the number of weeks it takes for the moneybox to get full is 60
theorem weeks_to_fill_moneybox : ∃ W : ℕ, (cycles * savings_per_week * W = total_savings) ∧ W = 60 := 
by 
  sorry

end weeks_to_fill_moneybox_l424_42478


namespace cannot_determine_letters_afternoon_l424_42443

theorem cannot_determine_letters_afternoon
  (emails_morning : ℕ) (letters_morning : ℕ)
  (emails_afternoon : ℕ) (letters_afternoon : ℕ)
  (h1 : emails_morning = 10)
  (h2 : letters_morning = 12)
  (h3 : emails_afternoon = 3)
  (h4 : emails_morning = emails_afternoon + 7) :
  ¬∃ (letters_afternoon : ℕ), true := 
sorry

end cannot_determine_letters_afternoon_l424_42443


namespace raviraj_distance_home_l424_42428

theorem raviraj_distance_home :
  let origin := (0, 0)
  let after_south := (0, -20)
  let after_west := (-10, -20)
  let after_north := (-10, 0)
  let final_pos := (-30, 0)
  Real.sqrt ((final_pos.1 - origin.1)^2 + (final_pos.2 - origin.2)^2) = 30 :=
by
  sorry

end raviraj_distance_home_l424_42428


namespace total_batteries_produced_l424_42470

def time_to_gather_materials : ℕ := 6 -- in minutes
def time_to_create_battery : ℕ := 9   -- in minutes
def num_robots : ℕ := 10
def total_time : ℕ := 5 * 60 -- in minutes (5 hours * 60 minutes/hour)

theorem total_batteries_produced :
  total_time / (time_to_gather_materials + time_to_create_battery) * num_robots = 200 :=
by
  -- Placeholder for the proof steps
  sorry

end total_batteries_produced_l424_42470


namespace left_handed_like_jazz_l424_42409

theorem left_handed_like_jazz (total_people left_handed like_jazz right_handed_dislike_jazz : ℕ)
    (h1 : total_people = 30)
    (h2 : left_handed = 12)
    (h3 : like_jazz = 20)
    (h4 : right_handed_dislike_jazz = 3)
    (h5 : ∀ p, p = total_people - left_handed ∧ p = total_people - (left_handed + right_handed_dislike_jazz)) :
    ∃ x, x = 5 := by
  sorry

end left_handed_like_jazz_l424_42409


namespace total_visitors_over_two_days_l424_42429

-- Conditions given in the problem statement
def first_day_visitors : ℕ := 583
def second_day_visitors : ℕ := 246

-- The main problem: proving the total number of visitors over the two days
theorem total_visitors_over_two_days : first_day_visitors + second_day_visitors = 829 := by
  -- Proof is omitted
  sorry

end total_visitors_over_two_days_l424_42429


namespace number_of_divisors_of_square_l424_42493

theorem number_of_divisors_of_square {n : ℕ} (h : ∃ p q : ℕ, p ≠ q ∧ Nat.Prime p ∧ Nat.Prime q ∧ n = p * q) : Nat.totient (n^2) = 9 :=
sorry

end number_of_divisors_of_square_l424_42493


namespace triangle_inequality_l424_42489

def can_form_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem triangle_inequality :
  ∃ (a b c : ℕ), 
  ((a = 3 ∧ b = 4 ∧ c = 5) ∧ can_form_triangle a b c) ∧
  ¬ ((a = 1 ∧ b = 2 ∧ c = 3) ∧ can_form_triangle a b c) ∧
  ¬ ((a = 2 ∧ b = 3 ∧ c = 6) ∧ can_form_triangle a b c) ∧
  ¬ ((a = 3 ∧ b = 3 ∧ c = 6) ∧ can_form_triangle a b c) :=
by
  sorry

end triangle_inequality_l424_42489


namespace infinite_series_sum_l424_42472

theorem infinite_series_sum :
  (∑' n : ℕ, (4 * (n + 1) - 3) / 3 ^ (n + 1)) = 13 / 8 :=
by sorry

end infinite_series_sum_l424_42472


namespace smallest_k_for_no_real_roots_l424_42462

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem smallest_k_for_no_real_roots :
  ∃ (k : ℤ), (∀ (x : ℝ), (x * x + 6 * x + 2 * k : ℝ) ≠ 0 ∧ k ≥ 5) :=
by
  sorry

end smallest_k_for_no_real_roots_l424_42462


namespace reading_time_difference_l424_42457

theorem reading_time_difference 
  (xanthia_reading_speed : ℕ) 
  (molly_reading_speed : ℕ) 
  (book_pages : ℕ) 
  (time_conversion_factor : ℕ)
  (hx : xanthia_reading_speed = 150)
  (hm : molly_reading_speed = 75)
  (hp : book_pages = 300)
  (ht : time_conversion_factor = 60) :
  ((book_pages / molly_reading_speed - book_pages / xanthia_reading_speed) * time_conversion_factor = 120) := 
by
  sorry

end reading_time_difference_l424_42457


namespace sum_of_ages_l424_42453

-- Given conditions and definitions
variables (M J : ℝ)

def condition1 : Prop := M = J + 8
def condition2 : Prop := M + 6 = 3 * (J - 3)

-- Proof goal
theorem sum_of_ages (h1 : condition1 M J) (h2 : condition2 M J) : M + J = 31 := 
by sorry

end sum_of_ages_l424_42453
