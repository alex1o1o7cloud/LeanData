import Mathlib

namespace NUMINAMATH_GPT_function_passes_through_fixed_point_l195_19531

noncomputable def f (a : ℝ) (x : ℝ) := 4 + Real.log (x + 1) / Real.log a

theorem function_passes_through_fixed_point (a : ℝ) (h : a > 0 ∧ a ≠ 1) :
  f a 0 = 4 := 
by
  sorry

end NUMINAMATH_GPT_function_passes_through_fixed_point_l195_19531


namespace NUMINAMATH_GPT_max_cards_mod3_l195_19555

theorem max_cards_mod3 (s : Finset ℕ) (h : s = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) : 
  ∃ t ⊆ s, t.card = 6 ∧ (t.prod id) % 3 = 1 := sorry

end NUMINAMATH_GPT_max_cards_mod3_l195_19555


namespace NUMINAMATH_GPT_calories_per_slice_l195_19502

theorem calories_per_slice
  (total_calories : ℕ)
  (portion_eaten : ℕ)
  (percentage_eaten : ℝ)
  (slices_in_cheesecake : ℕ)
  (calories_in_slice : ℕ) :
  total_calories = 2800 →
  percentage_eaten = 0.25 →
  portion_eaten = 2 →
  portion_eaten = percentage_eaten * slices_in_cheesecake →
  calories_in_slice = total_calories / slices_in_cheesecake →
  calories_in_slice = 350 :=
by
  intros
  sorry

end NUMINAMATH_GPT_calories_per_slice_l195_19502


namespace NUMINAMATH_GPT_diameter_of_circle_given_radius_l195_19513

theorem diameter_of_circle_given_radius (radius: ℝ) (h: radius = 7): 
  2 * radius = 14 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_diameter_of_circle_given_radius_l195_19513


namespace NUMINAMATH_GPT_exists_abc_gcd_equation_l195_19565

theorem exists_abc_gcd_equation (n : ℕ) : ∃ a b c : ℤ, n = Int.gcd a b * (c^2 - a*b) + Int.gcd b c * (a^2 - b*c) + Int.gcd c a * (b^2 - c*a) := sorry

end NUMINAMATH_GPT_exists_abc_gcd_equation_l195_19565


namespace NUMINAMATH_GPT_cricket_average_score_l195_19564

theorem cricket_average_score (A : ℝ)
    (h1 : 3 * 30 = 90)
    (h2 : 5 * 26 = 130) :
    2 * A + 90 = 130 → A = 20 :=
by
  intros h
  linarith

end NUMINAMATH_GPT_cricket_average_score_l195_19564


namespace NUMINAMATH_GPT_lina_walk_probability_l195_19587

/-- Total number of gates -/
def num_gates : ℕ := 20

/-- Distance between adjacent gates in feet -/
def gate_distance : ℕ := 50

/-- Maximum distance in feet Lina can walk to be within the desired range -/
def max_walk_distance : ℕ := 200

/-- Number of gates Lina can move within the max walk distance -/
def max_gates_within_distance : ℕ := max_walk_distance / gate_distance

/-- Total possible gate pairs for initial and new gate selection -/
def total_possible_pairs : ℕ := num_gates * (num_gates - 1)

/-- Total number of favorable gate pairs where walking distance is within the allowed range -/
def total_favorable_pairs : ℕ :=
  let edge_favorable (g : ℕ) := if g = 1 ∨ g = num_gates then 4
                                else if g = 2 ∨ g = num_gates - 1 then 5
                                else if g = 3 ∨ g = num_gates - 2 then 6
                                else if g = 4 ∨ g = num_gates - 3 then 7 else 8
  (edge_favorable 1) + (edge_favorable 2) + (edge_favorable 3) +
  (edge_favorable 4) + (num_gates - 8) * 8

/-- Probability that Lina walks 200 feet or less expressed as a reduced fraction -/
def probability_within_distance : ℚ :=
  (total_favorable_pairs : ℚ) / (total_possible_pairs : ℚ)

/-- p and q components of the fraction representing the probability -/
def p := 7
def q := 19

/-- Sum of p and q -/
def p_plus_q : ℕ := p + q

theorem lina_walk_probability : p_plus_q = 26 := by sorry

end NUMINAMATH_GPT_lina_walk_probability_l195_19587


namespace NUMINAMATH_GPT_min_sum_ab_l195_19559

theorem min_sum_ab (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ab : a * b = a + b + 3) : 
  a + b ≥ 6 := 
sorry

end NUMINAMATH_GPT_min_sum_ab_l195_19559


namespace NUMINAMATH_GPT_license_plates_count_correct_l195_19560

/-- Calculate the number of five-character license plates. -/
def count_license_plates : Nat :=
  let num_consonants := 20
  let num_vowels := 6
  let num_digits := 10
  num_consonants^2 * num_vowels^2 * num_digits

theorem license_plates_count_correct :
  count_license_plates = 144000 :=
by
  sorry

end NUMINAMATH_GPT_license_plates_count_correct_l195_19560


namespace NUMINAMATH_GPT_factor_x4_minus_64_l195_19570

theorem factor_x4_minus_64 :
  ∀ x : ℝ, (x^4 - 64 = (x - 2 * Real.sqrt 2) * (x + 2 * Real.sqrt 2) * (x^2 + 8)) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_factor_x4_minus_64_l195_19570


namespace NUMINAMATH_GPT_sum_A_B_C_l195_19524

noncomputable def number_B (A : ℕ) : ℕ := (A * 5) / 2
noncomputable def number_C (B : ℕ) : ℕ := (B * 7) / 4

theorem sum_A_B_C (A B C : ℕ) (h1 : A = 16) (h2 : A * 5 = B * 2) (h3 : B * 7 = C * 4) :
  A + B + C = 126 :=
by
  sorry

end NUMINAMATH_GPT_sum_A_B_C_l195_19524


namespace NUMINAMATH_GPT_total_cost_smore_night_l195_19520

-- Define the costs per item
def cost_graham_cracker : ℝ := 0.10
def cost_marshmallow : ℝ := 0.15
def cost_chocolate : ℝ := 0.25
def cost_caramel_piece : ℝ := 0.20
def cost_toffee_piece : ℝ := 0.05

-- Calculate the cost for each ingredient per S'more
def cost_caramel : ℝ := 2 * cost_caramel_piece
def cost_toffee : ℝ := 4 * cost_toffee_piece

-- Total cost of one S'more
def cost_one_smore : ℝ :=
  cost_graham_cracker + cost_marshmallow + cost_chocolate + cost_caramel + cost_toffee

-- Number of people and S'mores per person
def num_people : ℕ := 8
def smores_per_person : ℕ := 3

-- Total number of S'mores
def total_smores : ℕ := num_people * smores_per_person

-- Total cost of all the S'mores
def total_cost : ℝ := total_smores * cost_one_smore

-- The final statement
theorem total_cost_smore_night : total_cost = 26.40 := 
  sorry

end NUMINAMATH_GPT_total_cost_smore_night_l195_19520


namespace NUMINAMATH_GPT_distribution_ways_5_to_3_l195_19503

noncomputable def num_ways (n m : ℕ) : ℕ :=
  m ^ n

theorem distribution_ways_5_to_3 : num_ways 5 3 = 243 := by
  sorry

end NUMINAMATH_GPT_distribution_ways_5_to_3_l195_19503


namespace NUMINAMATH_GPT_BKINGTON_appears_first_on_eighth_line_l195_19563

-- Define the cycle lengths for letters and digits
def cycle_letters : ℕ := 8
def cycle_digits : ℕ := 4

-- Define the problem statement
theorem BKINGTON_appears_first_on_eighth_line :
  Nat.lcm cycle_letters cycle_digits = 8 := by
  sorry

end NUMINAMATH_GPT_BKINGTON_appears_first_on_eighth_line_l195_19563


namespace NUMINAMATH_GPT_insured_fraction_l195_19534

theorem insured_fraction (premium : ℝ) (rate : ℝ) (insured_value : ℝ) (original_value : ℝ)
  (h₁ : premium = 910)
  (h₂ : rate = 0.013)
  (h₃ : insured_value = premium / rate)
  (h₄ : original_value = 87500) :
  insured_value / original_value = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_insured_fraction_l195_19534


namespace NUMINAMATH_GPT_class_mean_score_l195_19558

theorem class_mean_score:
  ∀ (n: ℕ) (m: ℕ) (a b: ℕ),
  n + m = 50 →
  n * a = 3400 →
  m * b = 750 →
  a = 85 →
  b = 75 →
  (n * a + m * b) / (n + m) = 83 :=
by
  intros n m a b h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_class_mean_score_l195_19558


namespace NUMINAMATH_GPT_total_rainfall_l195_19536

theorem total_rainfall (R1 R2 : ℝ) (h1 : R2 = 1.5 * R1) (h2 : R2 = 15) : R1 + R2 = 25 := 
by
  sorry

end NUMINAMATH_GPT_total_rainfall_l195_19536


namespace NUMINAMATH_GPT_max_cards_l195_19552

def card_cost : ℝ := 0.85
def budget : ℝ := 7.50

theorem max_cards (n : ℕ) : card_cost * n ≤ budget → n ≤ 8 :=
by sorry

end NUMINAMATH_GPT_max_cards_l195_19552


namespace NUMINAMATH_GPT_solve_arithmetic_series_l195_19598

theorem solve_arithmetic_series : 
  25^2 - 23^2 + 21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 338 :=
by sorry

end NUMINAMATH_GPT_solve_arithmetic_series_l195_19598


namespace NUMINAMATH_GPT_find_x_l195_19521

theorem find_x (x : ℝ) (h : x^2 ∈ ({1, 0, x} : Set ℝ)) : x = -1 :=
sorry

end NUMINAMATH_GPT_find_x_l195_19521


namespace NUMINAMATH_GPT_wheels_travel_distance_l195_19577

noncomputable def total_horizontal_distance (R₁ R₂ : ℝ) : ℝ :=
  2 * Real.pi * R₁ + 2 * Real.pi * R₂

theorem wheels_travel_distance (R₁ R₂ : ℝ) (h₁ : R₁ = 2) (h₂ : R₂ = 3) :
  total_horizontal_distance R₁ R₂ = 10 * Real.pi :=
by
  rw [total_horizontal_distance, h₁, h₂]
  sorry

end NUMINAMATH_GPT_wheels_travel_distance_l195_19577


namespace NUMINAMATH_GPT_division_of_polynomial_l195_19545

theorem division_of_polynomial (a : ℤ) : (-28 * a^3) / (7 * a) = -4 * a^2 := by
  sorry

end NUMINAMATH_GPT_division_of_polynomial_l195_19545


namespace NUMINAMATH_GPT_awards_distribution_l195_19588

theorem awards_distribution :
  let num_awards := 6
  let num_students := 3 
  let min_awards_per_student := 2
  (num_awards = 6 ∧ num_students = 3 ∧ min_awards_per_student = 2) →
  ∃ (ways : ℕ), ways = 15 :=
by
  sorry

end NUMINAMATH_GPT_awards_distribution_l195_19588


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l195_19528

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  ((x + 2) * (x - 3) < 0 → |x - 1| < 2) ∧ (¬(|x - 1| < 2 → (x + 2) * (x - 3) < 0)) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l195_19528


namespace NUMINAMATH_GPT_range_cos_2alpha_cos_2beta_l195_19510

variable (α β : ℝ)
variable (h : Real.sin α + Real.cos β = 3 / 2)

theorem range_cos_2alpha_cos_2beta :
  -3/2 ≤ Real.cos (2 * α) + Real.cos (2 * β) ∧ Real.cos (2 * α) + Real.cos (2 * β) ≤ 3/2 :=
sorry

end NUMINAMATH_GPT_range_cos_2alpha_cos_2beta_l195_19510


namespace NUMINAMATH_GPT_total_cost_of_feeding_pets_for_one_week_l195_19527

-- Definitions based on conditions
def turtle_food_per_weight : ℚ := 1 / (1 / 2)
def turtle_weight : ℚ := 30
def turtle_food_qty_per_jar : ℚ := 15
def turtle_food_cost_per_jar : ℚ := 3

def bird_food_per_weight : ℚ := 2
def bird_weight : ℚ := 8
def bird_food_qty_per_bag : ℚ := 40
def bird_food_cost_per_bag : ℚ := 5

def hamster_food_per_weight : ℚ := 1.5 / (1 / 2)
def hamster_weight : ℚ := 3
def hamster_food_qty_per_box : ℚ := 20
def hamster_food_cost_per_box : ℚ := 4

-- Theorem stating the equivalent proof problem
theorem total_cost_of_feeding_pets_for_one_week :
  let turtle_food_needed := (turtle_weight * turtle_food_per_weight)
  let turtle_jars_needed := turtle_food_needed / turtle_food_qty_per_jar
  let turtle_cost := turtle_jars_needed * turtle_food_cost_per_jar
  let bird_food_needed := (bird_weight * bird_food_per_weight)
  let bird_bags_needed := bird_food_needed / bird_food_qty_per_bag
  let bird_cost := if bird_bags_needed < 1 then bird_food_cost_per_bag else bird_bags_needed * bird_food_cost_per_bag
  let hamster_food_needed := (hamster_weight * hamster_food_per_weight)
  let hamster_boxes_needed := hamster_food_needed / hamster_food_qty_per_box
  let hamster_cost := if hamster_boxes_needed < 1 then hamster_food_cost_per_box else hamster_boxes_needed * hamster_food_cost_per_box
  turtle_cost + bird_cost + hamster_cost = 21 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_feeding_pets_for_one_week_l195_19527


namespace NUMINAMATH_GPT_point_on_transformed_graph_l195_19541

variable (f : ℝ → ℝ)

theorem point_on_transformed_graph :
  (f 12 = 10) →
  3 * (19 / 9) = (f (3 * 4)) / 3 + 3 ∧ (4 + 19 / 9 = 55 / 9) :=
by
  sorry

end NUMINAMATH_GPT_point_on_transformed_graph_l195_19541


namespace NUMINAMATH_GPT_c_geq_one_l195_19573

variable {α : Type*} [LinearOrderedField α]

theorem c_geq_one
  (a : ℕ → α)
  (c : α)
  (h1 : ∀ i : ℕ, 0 < i → 0 ≤ a i ∧ a i ≤ c)
  (h2 : ∀ i j : ℕ, 0 < i → 0 < j → i ≠ j → |a i - a j| ≥ 1 / (i + j)) :
  c ≥ 1 :=
sorry

end NUMINAMATH_GPT_c_geq_one_l195_19573


namespace NUMINAMATH_GPT_range_of_a_l195_19511

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a < |x - 4| + |x + 3|) → a < 7 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l195_19511


namespace NUMINAMATH_GPT_odd_function_expression_on_negative_domain_l195_19582

theorem odd_function_expression_on_negative_domain
  (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_pos : ∀ x : ℝ, 0 < x → f x = x * (x - 1))
  (x : ℝ)
  (h_neg : x < 0)
  : f x = x * (x + 1) :=
sorry

end NUMINAMATH_GPT_odd_function_expression_on_negative_domain_l195_19582


namespace NUMINAMATH_GPT_Masc_age_difference_l195_19522

theorem Masc_age_difference (masc_age sam_age : ℕ) (h1 : masc_age + sam_age = 27) (h2 : masc_age = 17) (h3 : sam_age = 10) : masc_age - sam_age = 7 :=
by {
  -- Proof would go here, but it's omitted as per instructions
  sorry
}

end NUMINAMATH_GPT_Masc_age_difference_l195_19522


namespace NUMINAMATH_GPT_find_interest_rate_of_second_part_l195_19540

-- Definitions for the problem
def total_sum : ℚ := 2678
def P2 : ℚ := 1648
def P1 : ℚ := total_sum - P2
def r1 : ℚ := 0.03  -- 3% per annum
def t1 : ℚ := 8     -- 8 years
def I1 : ℚ := P1 * r1 * t1
def t2 : ℚ := 3     -- 3 years

-- Statement to prove
theorem find_interest_rate_of_second_part : ∃ r2 : ℚ, I1 = P2 * r2 * t2 ∧ r2 * 100 = 5 := by
  sorry

end NUMINAMATH_GPT_find_interest_rate_of_second_part_l195_19540


namespace NUMINAMATH_GPT_number_of_mixed_vegetable_plates_l195_19506

theorem number_of_mixed_vegetable_plates :
  ∃ n : ℕ, n * 70 = 1051 - (16 * 6 + 5 * 45 + 6 * 40) ∧ n = 7 :=
by
  sorry

end NUMINAMATH_GPT_number_of_mixed_vegetable_plates_l195_19506


namespace NUMINAMATH_GPT_price_of_ice_cream_bar_is_correct_l195_19538

noncomputable def price_ice_cream_bar (n_ice_cream_bars n_sundaes total_price price_of_sundae price_ice_cream_bar : ℝ) : Prop :=
  n_ice_cream_bars = 125 ∧
  n_sundaes = 125 ∧
  total_price = 225 ∧
  price_of_sundae = 1.2 →
  price_ice_cream_bar = 0.6

theorem price_of_ice_cream_bar_is_correct :
  price_ice_cream_bar 125 125 225 1.2 0.6 :=
by
  sorry

end NUMINAMATH_GPT_price_of_ice_cream_bar_is_correct_l195_19538


namespace NUMINAMATH_GPT_votes_cast_l195_19530

-- Define the conditions as given in the problem.
def total_votes (V : ℕ) := 35 * V / 100 + (35 * V / 100 + 2400) = V

-- The goal is to prove that the number of total votes V equals 8000.
theorem votes_cast : ∃ V : ℕ, total_votes V ∧ V = 8000 :=
by
  sorry -- The proof is not required, only the statement.

end NUMINAMATH_GPT_votes_cast_l195_19530


namespace NUMINAMATH_GPT_value_of_V3_l195_19597

def f (x : ℝ) : ℝ := 3 * x^5 + 8 * x^4 - 3 * x^3 + 5 * x^2 + 12 * x - 6

def horner (a : ℝ) : ℝ :=
  let V0 := 3
  let V1 := V0 * a + 8
  let V2 := V1 * a - 3
  let V3 := V2 * a + 5
  V3

theorem value_of_V3 : horner 2 = 55 :=
  by
    simp [horner]
    sorry

end NUMINAMATH_GPT_value_of_V3_l195_19597


namespace NUMINAMATH_GPT_find_b_l195_19584

theorem find_b (b p : ℝ) 
  (h1 : 3 * p + 15 = 0)
  (h2 : 15 * p + 3 = b) :
  b = -72 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l195_19584


namespace NUMINAMATH_GPT_four_digit_number_l195_19571

-- Definitions of the conditions
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Statement of the theorem
theorem four_digit_number (x y : ℕ) (hx : is_two_digit x) (hy : is_two_digit y) :
    (100 * x + y) = 1000 * x + y := sorry

end NUMINAMATH_GPT_four_digit_number_l195_19571


namespace NUMINAMATH_GPT_proof_N_union_complement_M_eq_235_l195_19557

open Set

theorem proof_N_union_complement_M_eq_235 :
  let U := ({1,2,3,4,5} : Set ℕ)
  let M := ({1, 4} : Set ℕ)
  let N := ({2, 5} : Set ℕ)
  N ∪ (U \ M) = ({2, 3, 5} : Set ℕ) :=
by
  sorry

end NUMINAMATH_GPT_proof_N_union_complement_M_eq_235_l195_19557


namespace NUMINAMATH_GPT_total_units_in_building_l195_19556

theorem total_units_in_building (x y : ℕ) (cost_1_bedroom cost_2_bedroom total_cost : ℕ)
  (h1 : cost_1_bedroom = 360) (h2 : cost_2_bedroom = 450)
  (h3 : total_cost = 4950) (h4 : y = 7) (h5 : total_cost = cost_1_bedroom * x + cost_2_bedroom * y) :
  x + y = 12 :=
sorry

end NUMINAMATH_GPT_total_units_in_building_l195_19556


namespace NUMINAMATH_GPT_right_triangle_side_lengths_l195_19508

theorem right_triangle_side_lengths (a S : ℝ) (b c : ℝ)
  (h1 : S = b + c)
  (h2 : c^2 = a^2 + b^2) :
  b = (S^2 - a^2) / (2 * S) ∧ c = (S^2 + a^2) / (2 * S) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_side_lengths_l195_19508


namespace NUMINAMATH_GPT_prime_gt_three_square_mod_twelve_l195_19509

theorem prime_gt_three_square_mod_twelve (p : ℕ) (h_prime: Prime p) (h_gt_three: p > 3) : (p^2) % 12 = 1 :=
by
  sorry

end NUMINAMATH_GPT_prime_gt_three_square_mod_twelve_l195_19509


namespace NUMINAMATH_GPT_solution_set_of_inequality_l195_19595

theorem solution_set_of_inequality :
  { x : ℝ | 2 * x^2 - x - 3 > 0 } = { x : ℝ | x > 3 / 2 ∨ x < -1 } :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l195_19595


namespace NUMINAMATH_GPT_convert_deg_to_rad_l195_19578

theorem convert_deg_to_rad (deg : ℝ) (π : ℝ) (h : deg = 50) : (deg * (π / 180) = 5 / 18 * π) :=
by
  -- Conditions
  sorry

end NUMINAMATH_GPT_convert_deg_to_rad_l195_19578


namespace NUMINAMATH_GPT_max_points_on_four_coplanar_circles_l195_19537

noncomputable def max_points_on_circles (num_circles : ℕ) (max_intersections : ℕ) : ℕ :=
num_circles * max_intersections

theorem max_points_on_four_coplanar_circles :
  max_points_on_circles 4 2 = 8 := 
sorry

end NUMINAMATH_GPT_max_points_on_four_coplanar_circles_l195_19537


namespace NUMINAMATH_GPT_find_b_l195_19569

theorem find_b (b : ℚ) (H : ∃ x y : ℚ, x = 3 ∧ y = -7 ∧ b * x + (b - 1) * y = b + 3) : 
  b = 4 / 5 := 
by
  sorry

end NUMINAMATH_GPT_find_b_l195_19569


namespace NUMINAMATH_GPT_unicorn_rope_problem_l195_19572

/-
  A unicorn is tethered by a 24-foot golden rope to the base of a sorcerer's cylindrical tower
  whose radius is 10 feet. The rope is attached to the tower at ground level and to the unicorn
  at a height of 6 feet. The unicorn has pulled the rope taut, and the end of the rope is 6 feet
  from the nearest point on the tower.
  The length of the rope that is touching the tower is given as:
  ((96 - sqrt(36)) / 6) feet,
  where 96, 36, and 6 are positive integers, and 6 is prime.
  We need to prove that the sum of these integers is 138.
-/
theorem unicorn_rope_problem : 
  let d := 96
  let e := 36
  let f := 6
  d + e + f = 138 := by
  sorry

end NUMINAMATH_GPT_unicorn_rope_problem_l195_19572


namespace NUMINAMATH_GPT_year_2023_not_lucky_l195_19561

def is_valid_date (month day year : ℕ) : Prop :=
  month * day = year % 100

def is_lucky_year (year : ℕ) : Prop :=
  ∃ month day, month ≤ 12 ∧ day ≤ 31 ∧ is_valid_date month day year

theorem year_2023_not_lucky : ¬ is_lucky_year 2023 :=
by sorry

end NUMINAMATH_GPT_year_2023_not_lucky_l195_19561


namespace NUMINAMATH_GPT_probability_of_winning_pair_l195_19551

-- Conditions: Define the deck composition and the winning pair.
inductive Color
| Red
| Green
| Blue

inductive Label
| A
| B
| C

structure Card :=
(color : Color)
(label : Label)

def deck : List Card :=
  [ {color := Color.Red, label := Label.A},
    {color := Color.Red, label := Label.B},
    {color := Color.Red, label := Label.C},
    {color := Color.Green, label := Label.A},
    {color := Color.Green, label := Label.B},
    {color := Color.Green, label := Label.C},
    {color := Color.Blue, label := Label.A},
    {color := Color.Blue, label := Label.B},
    {color := Color.Blue, label := Label.C} ]

def is_winning_pair (c1 c2 : Card) : Prop :=
  c1.color = c2.color ∨ c1.label = c2.label

-- Question: Prove the probability of drawing a winning pair.
theorem probability_of_winning_pair :
  (∃ (c1 c2 : Card), c1 ∈ deck ∧ c2 ∈ deck ∧ c1 ≠ c2 ∧ is_winning_pair c1 c2) →
  (∃ (c1 c2 : Card), c1 ∈ deck ∧ c2 ∈ deck ∧ c1 ≠ c2) →
  (9 + 9) / 36 = 1 / 2 :=
sorry

end NUMINAMATH_GPT_probability_of_winning_pair_l195_19551


namespace NUMINAMATH_GPT_remainder_7n_mod_4_l195_19585

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_7n_mod_4_l195_19585


namespace NUMINAMATH_GPT_gcd_of_terms_l195_19514

theorem gcd_of_terms (m n : ℕ) : gcd (4 * m^3 * n) (9 * m * n^3) = m * n := 
sorry

end NUMINAMATH_GPT_gcd_of_terms_l195_19514


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l195_19544

theorem hyperbola_eccentricity (a b c : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) 
  (h_eq : b ^ 2 = (5 / 4) * a ^ 2) 
  (h_c : c ^ 2 = a ^ 2 + b ^ 2) : 
  (3 / 2) = c / a :=
by sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l195_19544


namespace NUMINAMATH_GPT_possible_k_values_l195_19566

theorem possible_k_values :
  (∃ k b a c : ℤ, b = 2020 + k ∧ a * (c ^ 2) = (2020 + k) ∧ 
  (k = -404 ∨ k = -1010)) :=
sorry

end NUMINAMATH_GPT_possible_k_values_l195_19566


namespace NUMINAMATH_GPT_problem1_problem2_l195_19525

-- Definitions based on the given conditions
def p (a : ℝ) (x : ℝ) : Prop := a < x ∧ x < 3 * a
def q (x : ℝ) : Prop := x^2 - 5 * x + 6 < 0

-- Problem (1)
theorem problem1 (a x : ℝ) (h : a = 1) (hp : p a x) (hq : q x) : 2 < x ∧ x < 3 := by
  sorry

-- Problem (2)
theorem problem2 (a : ℝ) (h : ∀ x, q x → p a x) : 1 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l195_19525


namespace NUMINAMATH_GPT_part1_part2_l195_19529

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

def p (m : ℝ) : Prop :=
  let Δ := discriminant 1 m 1
  Δ > 0 ∧ -m / 2 < 0

def q (m : ℝ) : Prop :=
  let Δ := discriminant 4 (4 * (m - 2)) 1
  Δ < 0

theorem part1 (m : ℝ) (hp : p m) : m > 2 := 
sorry

theorem part2 (m : ℝ) (h : ¬(p m ∧ q m) ∧ (p m ∨ q m)) : (m ≥ 3) ∨ (1 < m ∧ m ≤ 2) := 
sorry

end NUMINAMATH_GPT_part1_part2_l195_19529


namespace NUMINAMATH_GPT_angle_in_quadrants_l195_19568

theorem angle_in_quadrants (α : ℝ) (hα : 0 < α ∧ α < π / 2) (k : ℤ) :
  (∃ i : ℤ, k = 2 * i + 1 ∧ π < (2 * i + 1) * π + α ∧ (2 * i + 1) * π + α < 3 * π / 2) ∨
  (∃ i : ℤ, k = 2 * i ∧ 0 < 2 * i * π + α ∧ 2 * i * π + α < π / 2) :=
sorry

end NUMINAMATH_GPT_angle_in_quadrants_l195_19568


namespace NUMINAMATH_GPT_holden_master_bath_size_l195_19532

theorem holden_master_bath_size (b n m : ℝ) (h_b : b = 309) (h_n : n = 918) (h : 2 * (b + m) = n) : m = 150 := by
  sorry

end NUMINAMATH_GPT_holden_master_bath_size_l195_19532


namespace NUMINAMATH_GPT_evaluate_log_expression_l195_19542

noncomputable def evaluate_expression (x y : Real) : Real :=
  (Real.log x / Real.log (y ^ 8)) * 
  (Real.log (y ^ 3) / Real.log (x ^ 7)) * 
  (Real.log (x ^ 7) / Real.log (y ^ 3)) * 
  (Real.log (y ^ 8) / Real.log (x ^ 2))

theorem evaluate_log_expression (x y : Real) : 
  evaluate_expression x y = (1 : Real) := sorry

end NUMINAMATH_GPT_evaluate_log_expression_l195_19542


namespace NUMINAMATH_GPT_find_line_AB_l195_19516

noncomputable def equation_of_line_AB : Prop :=
  ∀ (x y : ℝ), ((x-2)^2 + (y-1)^2 = 10) ∧ ((x+6)^2 + (y+3)^2 = 50) → (2*x + y = 0)

theorem find_line_AB : equation_of_line_AB := by
  sorry

end NUMINAMATH_GPT_find_line_AB_l195_19516


namespace NUMINAMATH_GPT_total_students_in_class_l195_19539

-- Definitions of the conditions
def E : ℕ := 55
def T : ℕ := 85
def N : ℕ := 30
def B : ℕ := 20

-- Statement of the theorem to prove the total number of students
theorem total_students_in_class : (E + T - B) + N = 150 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_total_students_in_class_l195_19539


namespace NUMINAMATH_GPT_total_apples_for_bobbing_l195_19550

theorem total_apples_for_bobbing (apples_per_bucket : ℕ) (buckets : ℕ) (total_apples : ℕ) : 
  apples_per_bucket = 9 → buckets = 7 → total_apples = apples_per_bucket * buckets → total_apples = 63 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_total_apples_for_bobbing_l195_19550


namespace NUMINAMATH_GPT_relationship_among_abc_l195_19535

noncomputable def a : ℝ := Real.logb 11 10
noncomputable def b : ℝ := (Real.logb 11 9) ^ 2
noncomputable def c : ℝ := Real.logb 10 11

theorem relationship_among_abc : b < a ∧ a < c :=
  sorry

end NUMINAMATH_GPT_relationship_among_abc_l195_19535


namespace NUMINAMATH_GPT_sixth_employee_salary_l195_19579

-- We define the salaries of the five employees
def salaries : List ℝ := [1000, 2500, 3100, 1500, 2000]

-- The mean of the salaries of these 5 employees and another employee
def mean_salary : ℝ := 2291.67

-- The number of employees
def number_of_employees : ℝ := 6

-- The total salary of the first five employees
def total_salary_5 : ℝ := salaries.sum

-- The total salary based on the given mean and number of employees
def total_salary_all : ℝ := mean_salary * number_of_employees

-- The statement to prove: The salary of the sixth employee
theorem sixth_employee_salary :
  total_salary_all - total_salary_5 = 3650.02 := 
  sorry

end NUMINAMATH_GPT_sixth_employee_salary_l195_19579


namespace NUMINAMATH_GPT_find_f_2016_l195_19567

noncomputable def f : ℝ → ℝ :=
sorry

axiom f_0_eq_2016 : f 0 = 2016

axiom f_x_plus_2_minus_f_x_leq : ∀ x : ℝ, f (x + 2) - f x ≤ 3 * 2 ^ x

axiom f_x_plus_6_minus_f_x_geq : ∀ x : ℝ, f (x + 6) - f x ≥ 63 * 2 ^ x

theorem find_f_2016 : f 2016 = 2015 + 2 ^ 2020 :=
sorry

end NUMINAMATH_GPT_find_f_2016_l195_19567


namespace NUMINAMATH_GPT_proportional_function_range_l195_19518

theorem proportional_function_range (m : ℝ) (h : ∀ x : ℝ, (x < 0 → (1 - m) * x > 0) ∧ (x > 0 → (1 - m) * x < 0)) : m > 1 :=
by sorry

end NUMINAMATH_GPT_proportional_function_range_l195_19518


namespace NUMINAMATH_GPT_negation_of_existential_l195_19599

theorem negation_of_existential (x : ℝ) : ¬(∃ x : ℝ, x^2 - 2 * x + 3 > 0) ↔ ∀ x : ℝ, x^2 - 2 * x + 3 ≤ 0 := 
by
  sorry

end NUMINAMATH_GPT_negation_of_existential_l195_19599


namespace NUMINAMATH_GPT_circle_reflection_l195_19576

/-- The reflection of a point over the line y = -x results in swapping the x and y coordinates 
and changing their signs. Given a circle with center (3, -7), the reflected center should be (7, -3). -/
theorem circle_reflection (x y : ℝ) (h : (x, y) = (3, -7)) : (y, -x) = (7, -3) :=
by
  -- since the problem is stated to skip the proof, we use sorry
  sorry

end NUMINAMATH_GPT_circle_reflection_l195_19576


namespace NUMINAMATH_GPT_part1_part2_l195_19580

def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 1|

theorem part1 : {x : ℝ | f x < 2} = {x : ℝ | -4 < x ∧ x < 2 / 3} :=
by
  sorry

theorem part2 : ∀ a : ℝ, (∃ x : ℝ, f x ≤ a - a^2 / 2) → (-1 ≤ a ∧ a ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l195_19580


namespace NUMINAMATH_GPT_find_n_l195_19500

noncomputable def e : ℝ := Real.exp 1

-- lean cannot compute non-trivial transcendental solutions, this would need numerical methods
theorem find_n (n : ℝ) (x : ℝ) (y : ℝ) (h1 : x = 3) (h2 : y = 27) :
  Real.log n ^ (n / (2 * Real.sqrt (Real.pi + x))) = y :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_find_n_l195_19500


namespace NUMINAMATH_GPT_power_root_l195_19523

noncomputable def x : ℝ := 1024 ^ (1 / 5)

theorem power_root (h : 1024 = 2^10) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_power_root_l195_19523


namespace NUMINAMATH_GPT_at_least_one_greater_than_one_l195_19526

theorem at_least_one_greater_than_one (a b : ℝ) (h : a + b > 2) : a > 1 ∨ b > 1 :=
sorry

end NUMINAMATH_GPT_at_least_one_greater_than_one_l195_19526


namespace NUMINAMATH_GPT_veranda_width_l195_19592

-- Defining the conditions as given in the problem
def room_length : ℝ := 21
def room_width : ℝ := 12
def veranda_area : ℝ := 148

-- The main statement to prove
theorem veranda_width :
  ∃ (w : ℝ), (21 + 2*w) * (12 + 2*w) - 21 * 12 = 148 ∧ w = 2 :=
by
  sorry

end NUMINAMATH_GPT_veranda_width_l195_19592


namespace NUMINAMATH_GPT_tangent_parallel_x_axis_monotonically_increasing_intervals_l195_19548

noncomputable def f (m n : ℝ) (x : ℝ) : ℝ := m * x^3 + n * x^2

theorem tangent_parallel_x_axis (m n : ℝ) (h : m ≠ 0) (h_tangent : 3 * m * (2:ℝ)^2 + 2 * n * (2:ℝ) = 0) :
  n = -3 * m :=
by
  sorry

theorem monotonically_increasing_intervals (m : ℝ) (h : m ≠ 0) : 
  (∀ x : ℝ, 3 * m * x * (x - (2 : ℝ)) > 0 ↔ 
    if m > 0 then x < 0 ∨ 2 < x else 0 < x ∧ x < 2) :=
by
  sorry

end NUMINAMATH_GPT_tangent_parallel_x_axis_monotonically_increasing_intervals_l195_19548


namespace NUMINAMATH_GPT_total_sleep_time_is_correct_l195_19591

-- Define the sleeping patterns of the animals
def cougar_sleep_even_days : ℕ := 4
def cougar_sleep_odd_days : ℕ := 6
def zebra_sleep_more : ℕ := 2

-- Define the distribution of even and odd days in a week
def even_days_in_week : ℕ := 3
def odd_days_in_week : ℕ := 4

-- Define the total weekly sleep time for the cougar
def cougar_total_weekly_sleep : ℕ := 
  (cougar_sleep_even_days * even_days_in_week) + 
  (cougar_sleep_odd_days * odd_days_in_week)

-- Define the total weekly sleep time for the zebra
def zebra_total_weekly_sleep : ℕ := 
  ((cougar_sleep_even_days + zebra_sleep_more) * even_days_in_week) + 
  ((cougar_sleep_odd_days + zebra_sleep_more) * odd_days_in_week)

-- Define the total weekly sleep time for both the cougar and the zebra
def total_weekly_sleep : ℕ := 
  cougar_total_weekly_sleep + zebra_total_weekly_sleep

-- Prove that the total weekly sleep time for both animals is 86 hours
theorem total_sleep_time_is_correct : total_weekly_sleep = 86 :=
by
  -- skipping proof
  sorry

end NUMINAMATH_GPT_total_sleep_time_is_correct_l195_19591


namespace NUMINAMATH_GPT_sam_letters_on_wednesday_l195_19501

/-- Sam's average letters per day. -/
def average_letters_per_day : ℕ := 5

/-- Number of days Sam wrote letters. -/
def number_of_days : ℕ := 2

/-- Letters Sam wrote on Tuesday. -/
def letters_on_tuesday : ℕ := 7

/-- Total letters Sam wrote in two days. -/
def total_letters : ℕ := average_letters_per_day * number_of_days

/-- Letters Sam wrote on Wednesday. -/
def letters_on_wednesday : ℕ := total_letters - letters_on_tuesday

theorem sam_letters_on_wednesday : letters_on_wednesday = 3 :=
by
  -- placeholder proof
  sorry

end NUMINAMATH_GPT_sam_letters_on_wednesday_l195_19501


namespace NUMINAMATH_GPT_find_a_l195_19517

-- Definitions for the problem
def quadratic_distinct_roots (a : ℝ) : Prop :=
  let Δ := a^2 - 16
  Δ > 0

def satisfies_root_equation (x1 x2 : ℝ) : Prop :=
  (x1^2 - (20 / (3 * x2^3)) = x2^2 - (20 / (3 * x1^3)))

-- Main statement of the proof problem
theorem find_a (a x1 x2 : ℝ) (h_quadratic_roots : quadratic_distinct_roots a)
               (h_root_equation : satisfies_root_equation x1 x2)
               (h_vieta_sum : x1 + x2 = -a) (h_vieta_product : x1 * x2 = 4) :
  a = -10 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l195_19517


namespace NUMINAMATH_GPT_range_of_a_plus_3b_l195_19504

theorem range_of_a_plus_3b :
  ∀ (a b : ℝ),
    -1 ≤ a + b ∧ a + b ≤ 1 ∧ 1 ≤ a - 2 * b ∧ a - 2 * b ≤ 3 →
    -11 / 3 ≤ a + 3 * b ∧ a + 3 * b ≤ 7 / 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_plus_3b_l195_19504


namespace NUMINAMATH_GPT_cherie_sparklers_count_l195_19515

-- Conditions
def koby_boxes : ℕ := 2
def koby_sparklers_per_box : ℕ := 3
def koby_whistlers_per_box : ℕ := 5
def cherie_boxes : ℕ := 1
def cherie_whistlers : ℕ := 9
def total_fireworks : ℕ := 33

-- Total number of fireworks Koby has
def koby_total_fireworks : ℕ :=
  koby_boxes * (koby_sparklers_per_box + koby_whistlers_per_box)

-- Total number of fireworks Cherie has
def cherie_total_fireworks : ℕ :=
  total_fireworks - koby_total_fireworks

-- Number of sparklers in Cherie's box
def cherie_sparklers : ℕ :=
  cherie_total_fireworks - cherie_whistlers

-- Proof statement
theorem cherie_sparklers_count : cherie_sparklers = 8 := by
  sorry

end NUMINAMATH_GPT_cherie_sparklers_count_l195_19515


namespace NUMINAMATH_GPT_find_percentage_of_male_students_l195_19596

def percentage_of_male_students (M F : ℝ) : Prop :=
  M + F = 1 ∧ 0.40 * M + 0.60 * F = 0.52

theorem find_percentage_of_male_students (M F : ℝ) (h1 : M + F = 1) (h2 : 0.40 * M + 0.60 * F = 0.52) : M = 0.40 :=
by
  sorry

end NUMINAMATH_GPT_find_percentage_of_male_students_l195_19596


namespace NUMINAMATH_GPT_executiveCommittee_ways_l195_19575

noncomputable def numberOfWaysToFormCommittee (totalMembers : ℕ) (positions : ℕ) : ℕ :=
Nat.choose (totalMembers - 1) (positions - 1)

theorem executiveCommittee_ways : numberOfWaysToFormCommittee 30 5 = 25839 := 
by
  -- skipping the proof as it's not required
  sorry

end NUMINAMATH_GPT_executiveCommittee_ways_l195_19575


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l195_19512

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = 2) : 
  (1 / (x - 3) / (1 / (x^2 - 9)) - x / (x + 1) * ((x^2 + x) / x^2)) = 4 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l195_19512


namespace NUMINAMATH_GPT_binary_to_decimal_and_septal_l195_19553

theorem binary_to_decimal_and_septal :
  let bin : ℕ := 110101
  let dec : ℕ := 53
  let septal : ℕ := 104
  let convert_to_decimal (b : ℕ) : ℕ := 
    (b % 10) * 2^0 + ((b / 10) % 10) * 2^1 + ((b / 100) % 10) * 2^2 + 
    ((b / 1000) % 10) * 2^3 + ((b / 10000) % 10) * 2^4 + ((b / 100000) % 10) * 2^5
  let convert_to_septal (n : ℕ) : ℕ :=
    let rec aux (n : ℕ) (acc : ℕ) (place : ℕ) : ℕ :=
      if n = 0 then acc
      else aux (n / 7) (acc + (n % 7) * place) (place * 10)
    aux n 0 1
  convert_to_decimal bin = dec ∧ convert_to_septal dec = septal :=
by
  sorry

end NUMINAMATH_GPT_binary_to_decimal_and_septal_l195_19553


namespace NUMINAMATH_GPT_tangent_line_at_origin_is_minus_3x_l195_19583

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + (a - 3) * x
noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + (a - 3)

theorem tangent_line_at_origin_is_minus_3x (a : ℝ) (h : ∀ x : ℝ, f_prime a x = f_prime a (-x)) : 
  (f_prime 0 0 = -3) → ∀ x : ℝ, (f a x = -3 * x) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_origin_is_minus_3x_l195_19583


namespace NUMINAMATH_GPT_range_of_y_l195_19574

-- Define the vectors
def a : ℝ × ℝ := (1, -2)
def b (y : ℝ) : ℝ × ℝ := (4, y)

-- Define the vector sum
def a_plus_b (y : ℝ) : ℝ × ℝ := (a.1 + (b y).1, a.2 + (b y).2)

-- Define the dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Prove the angle between a and a + b is acute and y ≠ -8
theorem range_of_y (y : ℝ) :
  (dot_product a (a_plus_b y) > 0) ↔ (y < 4.5 ∧ y ≠ -8) :=
by
  sorry

end NUMINAMATH_GPT_range_of_y_l195_19574


namespace NUMINAMATH_GPT_investment_amount_correct_l195_19589

noncomputable def investment_problem : Prop :=
  let initial_investment_rubles : ℝ := 10000
  let initial_exchange_rate : ℝ := 50
  let annual_return_rate : ℝ := 0.12
  let end_year_exchange_rate : ℝ := 80
  let currency_conversion_commission : ℝ := 0.05
  let broker_profit_commission_rate : ℝ := 0.3

  -- Computations
  let initial_investment_dollars := initial_investment_rubles / initial_exchange_rate
  let profit_dollars := initial_investment_dollars * annual_return_rate
  let total_dollars := initial_investment_dollars + profit_dollars
  let broker_commission_dollars := profit_dollars * broker_profit_commission_rate
  let post_commission_dollars := total_dollars - broker_commission_dollars
  let amount_in_rubles_before_conversion_commission := post_commission_dollars * end_year_exchange_rate
  let conversion_commission := amount_in_rubles_before_conversion_commission * currency_conversion_commission
  let final_amount_rubles := amount_in_rubles_before_conversion_commission - conversion_commission

  -- Proof goal
  final_amount_rubles = 16476.8

theorem investment_amount_correct : investment_problem := by {
  sorry
}

end NUMINAMATH_GPT_investment_amount_correct_l195_19589


namespace NUMINAMATH_GPT_coexistent_pair_example_coexistent_pair_neg_coexistent_pair_find_a_l195_19543

section coexistent_rational_number_pairs

-- Definitions based on the problem conditions:
def coexistent_pair (a b : ℚ) : Prop := a - b = a * b + 1

-- Proof problem 1
theorem coexistent_pair_example : coexistent_pair 3 (1/2) :=
sorry

-- Proof problem 2
theorem coexistent_pair_neg (m n : ℚ) (h : coexistent_pair m n) :
  coexistent_pair (-n) (-m) :=
sorry

-- Proof problem 3
example : ∃ (p q : ℚ), coexistent_pair p q ∧ (p, q) ≠ (2, 1/3) ∧ (p, q) ≠ (5, 2/3) ∧ (p, q) ≠ (3, 1/2) :=
sorry

-- Proof problem 4
theorem coexistent_pair_find_a (a : ℚ) (h : coexistent_pair a 3) :
  a = -2 :=
sorry

end coexistent_rational_number_pairs

end NUMINAMATH_GPT_coexistent_pair_example_coexistent_pair_neg_coexistent_pair_find_a_l195_19543


namespace NUMINAMATH_GPT_smallest_integral_k_no_real_roots_l195_19547

theorem smallest_integral_k_no_real_roots :
  ∃ k : ℤ, (∀ x : ℝ, 2 * x * (k * x - 4) - x^2 + 6 ≠ 0) ∧ 
           (∀ j : ℤ, j < k → (∃ x : ℝ, 2 * x * (j * x - 4) - x^2 + 6 = 0)) ∧
           k = 2 :=
by sorry

end NUMINAMATH_GPT_smallest_integral_k_no_real_roots_l195_19547


namespace NUMINAMATH_GPT_problem_solution_l195_19590

theorem problem_solution (x y : ℝ) (h1 : y = x / (3 * x + 1)) (hx : x ≠ 0) (hy : y ≠ 0) :
    (x - y + 3 * x * y) / (x * y) = 6 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l195_19590


namespace NUMINAMATH_GPT_arithmetic_sequence_properties_l195_19546

-- Defining the arithmetic sequence and the conditions
variable {a : ℕ → ℤ}
variable {d : ℤ}
noncomputable def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = d

-- Given conditions
variable (h1 : a 5 = 10)
variable (h2 : a 1 + a 2 + a 3 = 3)

-- The theorem to prove
theorem arithmetic_sequence_properties :
  is_arithmetic_sequence a d → a 1 = -2 ∧ d = 3 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_properties_l195_19546


namespace NUMINAMATH_GPT_minimum_value_w_l195_19593

theorem minimum_value_w : ∃ (x y : ℝ), ∀ w, w = 3 * x^2 + 3 * y^2 + 9 * x - 6 * y + 30 → w ≥ 20.25 :=
sorry

end NUMINAMATH_GPT_minimum_value_w_l195_19593


namespace NUMINAMATH_GPT_craig_apples_total_l195_19581

-- Defining the conditions
def initial_apples_craig : ℝ := 20.0
def apples_from_eugene : ℝ := 7.0

-- Defining the total number of apples Craig will have
noncomputable def total_apples_craig : ℝ := initial_apples_craig + apples_from_eugene

-- The theorem stating that Craig will have 27.0 apples.
theorem craig_apples_total : total_apples_craig = 27.0 := by
  -- Proof here
  sorry

end NUMINAMATH_GPT_craig_apples_total_l195_19581


namespace NUMINAMATH_GPT_min_x_plus_2y_l195_19507

theorem min_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : 1 / (2 * x + y) + 1 / (y + 1) = 1) : x + 2 * y ≥ (1 / 2) + Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_min_x_plus_2y_l195_19507


namespace NUMINAMATH_GPT_num_best_friends_l195_19554

theorem num_best_friends (total_cards : ℕ) (cards_per_friend : ℕ) (h1 : total_cards = 455) (h2 : cards_per_friend = 91) : total_cards / cards_per_friend = 5 :=
by
  -- We assume the proof is going to be done here
  sorry

end NUMINAMATH_GPT_num_best_friends_l195_19554


namespace NUMINAMATH_GPT_cos_A_eq_sqrt3_div3_of_conditions_l195_19586

noncomputable def given_conditions
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : (Real.sqrt 3 * b - c) * Real.cos A = a * Real.cos C)
  (h2 : a ≠ 0) 
  (h3 : b ≠ 0) 
  (h4 : c ≠ 0) 
  (h5 : A ≠ 0) 
  (h6 : B ≠ 0) 
  (h7 : C ≠ 0) : Prop :=
  (Real.cos A = Real.sqrt 3 / 3)

theorem cos_A_eq_sqrt3_div3_of_conditions
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : (Real.sqrt 3 * b - c) * Real.cos A = a * Real.cos C)
  (h2 : a ≠ 0) 
  (h3 : b ≠ 0) 
  (h4 : c ≠ 0) 
  (h5 : A ≠ 0) 
  (h6 : B ≠ 0) 
  (h7 : C ≠ 0) :
  Real.cos A = Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_GPT_cos_A_eq_sqrt3_div3_of_conditions_l195_19586


namespace NUMINAMATH_GPT_correct_options_l195_19549

-- Definitions for lines l and n
def line_l (a : ℝ) (x y : ℝ) : Prop := (a + 2) * x + a * y - 2 = 0
def line_n (a : ℝ) (x y : ℝ) : Prop := (a - 2) * x + 3 * y - 6 = 0

-- The condition for lines to be parallel, equating the slopes
def parallel_lines (a : ℝ) : Prop := -(a + 2) / a = -(a - 2) / 3

-- The condition that line l passes through the point (1, -1)
def passes_through_point (a : ℝ) : Prop := line_l a 1 (-1)

-- The theorem statement
theorem correct_options (a : ℝ) :
  (parallel_lines a → a = 6 ∨ a = -1) ∧ (passes_through_point a) :=
by
  sorry

end NUMINAMATH_GPT_correct_options_l195_19549


namespace NUMINAMATH_GPT_parabola_directrix_l195_19533

theorem parabola_directrix (x y : ℝ) (h : y = 8 * x^2) : y = -1 / 32 :=
sorry

end NUMINAMATH_GPT_parabola_directrix_l195_19533


namespace NUMINAMATH_GPT_value_of_expression_l195_19594

variable {a b : ℝ}
variables (h1 : ∀ x, 3 * x^2 + 9 * x - 18 = 0 → x = a ∨ x = b)

theorem value_of_expression : (3 * a - 2) * (6 * b - 9) = 27 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l195_19594


namespace NUMINAMATH_GPT_sum_of_coefficients_of_expansion_l195_19519

-- Define a predicate for a term being constant
def is_constant_term (n : ℕ) (term : ℚ) : Prop := 
  term = 0

-- Define the sum of coefficients computation
noncomputable def sum_of_coefficients (n : ℕ) : ℚ := 
  (1 - 3)^n

-- The main statement of the problem in Lean
theorem sum_of_coefficients_of_expansion {n : ℕ} 
  (h : is_constant_term n (2 * n - 10)) : 
  sum_of_coefficients 5 = -32 := 
sorry

end NUMINAMATH_GPT_sum_of_coefficients_of_expansion_l195_19519


namespace NUMINAMATH_GPT_alia_markers_l195_19562

theorem alia_markers (S A a : ℕ) (h1 : S = 60) (h2 : A = S / 3) (h3 : a = 2 * A) : a = 40 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_alia_markers_l195_19562


namespace NUMINAMATH_GPT_parakeet_eats_2_grams_per_day_l195_19505

-- Define the conditions
def parrot_daily : ℕ := 14
def finch_daily (parakeet_daily : ℕ) : ℕ := parakeet_daily / 2
def num_parakeets : ℕ := 3
def num_parrots : ℕ := 2
def num_finches : ℕ := 4
def total_weekly_consumption : ℕ := 266

-- Define the daily consumption equation for all birds
def daily_consumption (parakeet_daily : ℕ) : ℕ :=
  num_parakeets * parakeet_daily + num_parrots * parrot_daily + num_finches * finch_daily parakeet_daily

-- Define the weekly consumption equation
def weekly_consumption (parakeet_daily : ℕ) : ℕ :=
  7 * daily_consumption parakeet_daily

-- State the theorem to prove that each parakeet eats 2 grams per day
theorem parakeet_eats_2_grams_per_day :
  (weekly_consumption 2) = total_weekly_consumption ↔ 2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_parakeet_eats_2_grams_per_day_l195_19505
