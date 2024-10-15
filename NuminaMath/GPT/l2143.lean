import Mathlib

namespace NUMINAMATH_GPT_radius_increase_125_surface_area_l2143_214326

theorem radius_increase_125_surface_area (r r' : ℝ) 
(increase_surface_area : 4 * π * (r'^2) = 2.25 * 4 * π * r^2) : r' = 1.5 * r :=
by 
  sorry

end NUMINAMATH_GPT_radius_increase_125_surface_area_l2143_214326


namespace NUMINAMATH_GPT_slope_of_line_l2143_214398

theorem slope_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : (- (4 : ℝ) / 7) = -4 / 7 :=
by
  -- Sorry for the proof for completeness
  sorry

end NUMINAMATH_GPT_slope_of_line_l2143_214398


namespace NUMINAMATH_GPT_remainder_is_23_l2143_214304

def number_remainder (n : ℤ) : ℤ :=
  n % 36

theorem remainder_is_23 (n : ℤ) (h1 : n % 4 = 3) (h2 : n % 9 = 5) :
  number_remainder n = 23 :=
by
  sorry

end NUMINAMATH_GPT_remainder_is_23_l2143_214304


namespace NUMINAMATH_GPT_expression_eqn_l2143_214340

theorem expression_eqn (a : ℝ) (E : ℝ → ℝ)
  (h₁ : -6 * a^2 = 3 * (E a + 2))
  (h₂ : a = 1) : E a = -2 * a^2 - 2 :=
by
  sorry

end NUMINAMATH_GPT_expression_eqn_l2143_214340


namespace NUMINAMATH_GPT_house_painting_cost_l2143_214355

theorem house_painting_cost :
  let judson_contrib := 500.0
  let kenny_contrib_euros := judson_contrib * 1.2 / 1.1
  let camilo_contrib_pounds := (kenny_contrib_euros * 1.1 + 200.0) / 1.3
  let camilo_contrib_usd := camilo_contrib_pounds * 1.3
  judson_contrib + kenny_contrib_euros * 1.1 + camilo_contrib_usd = 2020.0 := 
by {
  sorry
}

end NUMINAMATH_GPT_house_painting_cost_l2143_214355


namespace NUMINAMATH_GPT_tom_profit_l2143_214338

-- Define the initial conditions
def initial_investment : ℕ := 20 * 3
def revenue_from_selling : ℕ := 10 * 4
def value_of_remaining_shares : ℕ := 10 * 6
def total_amount : ℕ := revenue_from_selling + value_of_remaining_shares

-- We claim that the profit Tom makes is 40 dollars
theorem tom_profit : (total_amount - initial_investment) = 40 := by
  sorry

end NUMINAMATH_GPT_tom_profit_l2143_214338


namespace NUMINAMATH_GPT_monotonic_increasing_intervals_max_min_values_l2143_214363

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (2 * x - Real.pi / 3)

theorem monotonic_increasing_intervals (k : ℤ) :
  ∃ (a b : ℝ), a = k * Real.pi - Real.pi / 12 ∧ b = k * Real.pi + 5 * Real.pi / 12 ∧
    ∀ x₁ x₂ : ℝ, a ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ b → f x₁ ≤ f x₂ :=
sorry

theorem max_min_values : ∃ (xmin xmax : ℝ) (fmin fmax : ℝ),
  xmin = 0 ∧ fmin = f 0 ∧ fmin = - Real.sqrt 3 / 2 ∧
  xmax = 5 * Real.pi / 12 ∧ fmax = f (5 * Real.pi / 12) ∧ fmax = 1 ∧
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 →
    fmin ≤ f x ∧ f x ≤ fmax :=
sorry

end NUMINAMATH_GPT_monotonic_increasing_intervals_max_min_values_l2143_214363


namespace NUMINAMATH_GPT_relationship_among_a_b_c_l2143_214374

noncomputable def a : ℝ := (1 / 2) ^ (3 / 4)
noncomputable def b : ℝ := (3 / 4) ^ (1 / 2)
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem relationship_among_a_b_c : a < b ∧ b < c := 
by
  -- Skipping the proof steps
  sorry

end NUMINAMATH_GPT_relationship_among_a_b_c_l2143_214374


namespace NUMINAMATH_GPT_yellow_lights_count_l2143_214329

theorem yellow_lights_count (total_lights : ℕ) (red_lights : ℕ) (blue_lights : ℕ) (yellow_lights : ℕ) :
  total_lights = 95 → red_lights = 26 → blue_lights = 32 → yellow_lights = total_lights - (red_lights + blue_lights) → yellow_lights = 37 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end NUMINAMATH_GPT_yellow_lights_count_l2143_214329


namespace NUMINAMATH_GPT_complex_division_l2143_214325

theorem complex_division :
  (⟨5, -1⟩ : ℂ) / (⟨1, -1⟩ : ℂ) = (⟨3, 2⟩ : ℂ) :=
sorry

end NUMINAMATH_GPT_complex_division_l2143_214325


namespace NUMINAMATH_GPT_squirrel_spiral_path_height_l2143_214350

-- Define the conditions
def spiralPath (circumference rise totalDistance : ℝ) : Prop :=
  ∃ (numberOfCircuits : ℝ), numberOfCircuits = totalDistance / circumference ∧ numberOfCircuits * rise = totalDistance

-- Define the height of the post proof
theorem squirrel_spiral_path_height : 
  let circumference := 2 -- feet
  let rise := 4 -- feet
  let totalDistance := 8 -- feet
  let height := 16 -- feet
  spiralPath circumference rise totalDistance → height = (totalDistance / circumference) * rise :=
by
  intro h
  sorry

end NUMINAMATH_GPT_squirrel_spiral_path_height_l2143_214350


namespace NUMINAMATH_GPT_pentagon_angles_l2143_214356

def is_point_in_convex_pentagon (O A B C D E : Point) : Prop := sorry
def angle (A B C : Point) : ℝ := sorry -- Assume definition of angle in radians

theorem pentagon_angles (O A B C D E: Point) (hO : is_point_in_convex_pentagon O A B C D E)
  (h1: angle A O B = angle B O C) (h2: angle B O C = angle C O D)
  (h3: angle C O D = angle D O E) (h4: angle D O E = angle E O A) :
  (angle E O A = angle A O B) ∨ (angle E O A + angle A O B = π) :=
sorry

end NUMINAMATH_GPT_pentagon_angles_l2143_214356


namespace NUMINAMATH_GPT_find_a_l2143_214359

theorem find_a (a : ℤ) (h1 : 0 < a) (h2 : a < 13) (h3 : (53^2017 + a) % 13 = 0) : a = 12 :=
sorry

end NUMINAMATH_GPT_find_a_l2143_214359


namespace NUMINAMATH_GPT_quotient_of_even_and_odd_composites_l2143_214308

theorem quotient_of_even_and_odd_composites:
  (4 * 6 * 8 * 10 * 12) / (9 * 15 * 21 * 25 * 27) = 512 / 28525 := by
sorry

end NUMINAMATH_GPT_quotient_of_even_and_odd_composites_l2143_214308


namespace NUMINAMATH_GPT_find_exponent_l2143_214390

theorem find_exponent (n : ℝ) (hn: (3:ℝ)^n = Real.sqrt 3) : n = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_find_exponent_l2143_214390


namespace NUMINAMATH_GPT_solve_natural_numbers_system_l2143_214354

theorem solve_natural_numbers_system :
  ∃ a b c : ℕ, (a^3 - b^3 - c^3 = 3 * a * b * c) ∧ (a^2 = 2 * (a + b + c)) ∧
  ((a = 4 ∧ b = 1 ∧ c = 3) ∨ (a = 4 ∧ b = 2 ∧ c = 2) ∨ (a = 4 ∧ b = 3 ∧ c = 1)) :=
by
  sorry

end NUMINAMATH_GPT_solve_natural_numbers_system_l2143_214354


namespace NUMINAMATH_GPT_angles_sum_eq_l2143_214370

variables {a b c : ℝ} {A B C : ℝ}

theorem angles_sum_eq {a b c : ℝ} {A B C : ℝ}
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : A > 0) (h5 : B > 0) (h6 : C > 0)
  (h7 : A + B + C = π)
  (h8 : (a + c - b) * (a + c + b) = 3 * a * c) :
  A + C = 2 * π / 3 :=
sorry

end NUMINAMATH_GPT_angles_sum_eq_l2143_214370


namespace NUMINAMATH_GPT_scientific_notation_4040000_l2143_214368

theorem scientific_notation_4040000 :
  (4040000 : ℝ) = 4.04 * (10 : ℝ)^6 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_4040000_l2143_214368


namespace NUMINAMATH_GPT_side_length_of_square_l2143_214318

theorem side_length_of_square (d : ℝ) (h : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, s = 2 ∧ d = s * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_side_length_of_square_l2143_214318


namespace NUMINAMATH_GPT_compare_abc_l2143_214365

noncomputable def a := Real.exp (Real.sqrt 2)
noncomputable def b := 2 + Real.sqrt 2
noncomputable def c := Real.log (12 + 6 * Real.sqrt 2)

theorem compare_abc : a > b ∧ b > c :=
by
  sorry

end NUMINAMATH_GPT_compare_abc_l2143_214365


namespace NUMINAMATH_GPT_line_perpendicular_passing_through_point_l2143_214303

theorem line_perpendicular_passing_through_point :
  ∃ (a b c : ℝ), (∀ (x y : ℝ), 2 * x + y - 2 = 0 ↔ a * x + b * y + c = 0) ∧ 
                (a, b) ≠ (0, 0) ∧ 
                (a * -1 + b * 4 + c = 0) ∧ 
                (a * 1/2 + b * (-2) ≠ -4) :=
by { sorry }

end NUMINAMATH_GPT_line_perpendicular_passing_through_point_l2143_214303


namespace NUMINAMATH_GPT_find_f_l2143_214330

theorem find_f {f : ℝ → ℝ} (h : ∀ x : ℝ, f (x - 1) = x^2 - 1) : ∀ x : ℝ, f x = x^2 + 2*x := 
by
  sorry

end NUMINAMATH_GPT_find_f_l2143_214330


namespace NUMINAMATH_GPT_exists_fraction_x_only_and_f_of_1_is_0_l2143_214360

theorem exists_fraction_x_only_and_f_of_1_is_0 : ∃ f : ℚ → ℚ, (∀ x : ℚ, f x = (x - 1) / x) ∧ f 1 = 0 := 
by
  sorry

end NUMINAMATH_GPT_exists_fraction_x_only_and_f_of_1_is_0_l2143_214360


namespace NUMINAMATH_GPT_cost_price_article_l2143_214331

theorem cost_price_article (x : ℝ) (h : 56 - x = x - 42) : x = 49 :=
by sorry

end NUMINAMATH_GPT_cost_price_article_l2143_214331


namespace NUMINAMATH_GPT_zoo_charge_for_child_l2143_214302

theorem zoo_charge_for_child (charge_adult : ℕ) (total_people total_bill children : ℕ) (charge_child : ℕ) : 
  charge_adult = 8 → total_people = 201 → total_bill = 964 → children = 161 → 
  total_bill - (total_people - children) * charge_adult = children * charge_child → 
  charge_child = 4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_zoo_charge_for_child_l2143_214302


namespace NUMINAMATH_GPT_initial_percentage_increase_l2143_214323

variable (P : ℝ) (x : ℝ)

theorem initial_percentage_increase :
  (P * (1 + x / 100) * 1.3 = P * 1.625) → (x = 25) := by
  sorry

end NUMINAMATH_GPT_initial_percentage_increase_l2143_214323


namespace NUMINAMATH_GPT_find_angle_A_l2143_214334

theorem find_angle_A (a b : ℝ) (A B : ℝ) 
  (ha : a = Real.sqrt 2) (hb : b = Real.sqrt 3) (hB : B = Real.pi / 3) :
  A = Real.pi / 4 :=
by
  -- This is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_find_angle_A_l2143_214334


namespace NUMINAMATH_GPT_perimeter_C_l2143_214316

def is_square (n : ℕ) : Prop := n > 0 ∧ ∃ s : ℕ, s * s = n

variable (A B C : ℕ) -- Defining the squares
variable (sA sB sC : ℕ) -- Defining the side lengths

-- Conditions as definitions
axiom square_figures : is_square A ∧ is_square B ∧ is_square C 
axiom perimeter_A : 4 * sA = 20
axiom perimeter_B : 4 * sB = 40
axiom side_length_C : sC = 2 * (sA + sB)

-- The equivalent proof problem statement
theorem perimeter_C : 4 * sC = 120 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_perimeter_C_l2143_214316


namespace NUMINAMATH_GPT_smallest_consecutive_sum_l2143_214367

theorem smallest_consecutive_sum (x : ℤ) (h : x + (x + 1) + (x + 2) = 90) : x = 29 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_consecutive_sum_l2143_214367


namespace NUMINAMATH_GPT_algorithm_comparable_to_euclidean_l2143_214313

-- Define the conditions
def ancient_mathematics_world_leading : Prop := 
  True -- Placeholder representing the historical condition

def song_yuan_algorithm : Prop :=
  True -- Placeholder representing the algorithmic condition

-- The main theorem representing the problem statement
theorem algorithm_comparable_to_euclidean :
  ancient_mathematics_world_leading → song_yuan_algorithm → 
  True :=  -- Placeholder representing that the algorithm is the method of successive subtraction
by 
  intro h1 h2 
  sorry

end NUMINAMATH_GPT_algorithm_comparable_to_euclidean_l2143_214313


namespace NUMINAMATH_GPT_cost_of_largest_pot_equals_229_l2143_214347

-- Define the conditions
variables (total_cost : ℝ) (num_pots : ℕ) (cost_diff : ℝ)

-- Assume given conditions
axiom h1 : num_pots = 6
axiom h2 : total_cost = 8.25
axiom h3 : cost_diff = 0.3

-- Define the function for the cost of the smallest pot and largest pot
noncomputable def smallest_pot_cost : ℝ :=
  (total_cost - (num_pots - 1) * cost_diff) / num_pots

noncomputable def largest_pot_cost : ℝ :=
  smallest_pot_cost total_cost num_pots cost_diff + (num_pots - 1) * cost_diff

-- Prove the cost of the largest pot equals 2.29
theorem cost_of_largest_pot_equals_229 (h1 : num_pots = 6) (h2 : total_cost = 8.25) (h3 : cost_diff = 0.3) :
  largest_pot_cost total_cost num_pots cost_diff = 2.29 :=
  by sorry

end NUMINAMATH_GPT_cost_of_largest_pot_equals_229_l2143_214347


namespace NUMINAMATH_GPT_find_total_values_l2143_214344

theorem find_total_values (n : ℕ) (S : ℝ) 
  (h1 : S / n = 150) 
  (h2 : (S + 25) / n = 151.25) 
  (h3 : 25 = 160 - 135) : n = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_total_values_l2143_214344


namespace NUMINAMATH_GPT_Betty_flies_caught_in_morning_l2143_214388

-- Definitions from the conditions
def total_flies_needed_in_a_week : ℕ := 14
def flies_eaten_per_day : ℕ := 2
def days_in_a_week : ℕ := 7
def flies_caught_in_morning (X : ℕ) : ℕ := X
def flies_caught_in_afternoon : ℕ := 6
def flies_escaped : ℕ := 1
def flies_short : ℕ := 4

-- Given statement in Lean 4
theorem Betty_flies_caught_in_morning (X : ℕ) 
  (h1 : flies_caught_in_morning X + flies_caught_in_afternoon - flies_escaped = total_flies_needed_in_a_week - flies_short) : 
  X = 5 :=
by
  sorry

end NUMINAMATH_GPT_Betty_flies_caught_in_morning_l2143_214388


namespace NUMINAMATH_GPT_sum_ABC_eq_7_base_8_l2143_214328

/-- Lean 4 statement for the problem.

A, B, C: are distinct non-zero digits less than 8 in base 8, and
A B C_8 + B C_8 = A C A_8 holds true.
-/
theorem sum_ABC_eq_7_base_8 :
  ∃ (A B C : ℕ), A < 8 ∧ B < 8 ∧ C < 8 ∧ 
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
  (A * 64 + B * 8 + C) + (B * 8 + C) = A * 64 + C * 8 + A ∧
  A + B + C = 7 :=
by { sorry }

end NUMINAMATH_GPT_sum_ABC_eq_7_base_8_l2143_214328


namespace NUMINAMATH_GPT_valid_word_combinations_l2143_214351

-- Definition of valid_combination based on given conditions
def valid_combination : ℕ :=
  26 * 5 * 26

-- Statement to prove the number of valid four-letter combinations is 3380
theorem valid_word_combinations : valid_combination = 3380 := by
  sorry

end NUMINAMATH_GPT_valid_word_combinations_l2143_214351


namespace NUMINAMATH_GPT_olivia_wallet_after_shopping_l2143_214333

variable (initial_wallet : ℝ := 200) 
variable (groceries : ℝ := 65)
variable (shoes_original_price : ℝ := 75)
variable (shoes_discount_rate : ℝ := 0.15)
variable (belt : ℝ := 25)

theorem olivia_wallet_after_shopping :
  initial_wallet - (groceries + (shoes_original_price - shoes_original_price * shoes_discount_rate) + belt) = 46.25 := by
  sorry

end NUMINAMATH_GPT_olivia_wallet_after_shopping_l2143_214333


namespace NUMINAMATH_GPT_probability_of_event_l2143_214375

-- Definitions for the problem setup

-- Box C and its range
def boxC := {i : ℕ | 1 ≤ i ∧ i ≤ 30}

-- Box D and its range
def boxD := {i : ℕ | 21 ≤ i ∧ i ≤ 50}

-- Condition for a tile from box C being less than 20
def tile_from_C_less_than_20 (i : ℕ) : Prop := i ∈ boxC ∧ i < 20

-- Condition for a tile from box D being odd or greater than 45
def tile_from_D_odd_or_greater_than_45 (i : ℕ) : Prop := i ∈ boxD ∧ (i % 2 = 1 ∨ i > 45)

-- Main statement
theorem probability_of_event :
  (19 / 30 : ℚ) * (17 / 30 : ℚ) = (323 / 900 : ℚ) :=
by sorry

end NUMINAMATH_GPT_probability_of_event_l2143_214375


namespace NUMINAMATH_GPT_balls_is_perfect_square_l2143_214327

open Classical -- Open classical logic for nonconstructive proofs

-- Define a noncomputable function to capture the main proof argument
noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem balls_is_perfect_square {a v : ℕ} (h : (2 * a * v) = (a + v) * (a + v - 1))
  : is_perfect_square (a + v) :=
sorry

end NUMINAMATH_GPT_balls_is_perfect_square_l2143_214327


namespace NUMINAMATH_GPT_correct_propositions_l2143_214372

def line (P: Type) := P → P → Prop  -- A line is a relation between points in a plane

variables (plane1 plane2: Type) -- Define two types representing two planes
variables (P1 P2: plane1) -- Points in plane1
variables (Q1 Q2: plane2) -- Points in plane2

axiom perpendicular_planes : ¬∃ l1 : line plane1, ∀ l2 : line plane2, ¬ (∀ p1 p2, l1 p1 p2 ∧ ∀ q1 q2, l2 q1 q2)

theorem correct_propositions : 3 = 3 := by
  sorry

end NUMINAMATH_GPT_correct_propositions_l2143_214372


namespace NUMINAMATH_GPT_find_math_books_l2143_214366

theorem find_math_books 
  (M H : ℕ)
  (h1 : M + H = 80)
  (h2 : 4 * M + 5 * H = 390) : 
  M = 10 := 
by 
  sorry

end NUMINAMATH_GPT_find_math_books_l2143_214366


namespace NUMINAMATH_GPT_gcf_lcm_problem_l2143_214382

def GCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem gcf_lcm_problem :
  GCF (LCM 9 15) (LCM 10 21) = 15 := by
  sorry

end NUMINAMATH_GPT_gcf_lcm_problem_l2143_214382


namespace NUMINAMATH_GPT_find_number_l2143_214381

theorem find_number (x : ℝ) (h : 50 + 5 * 12 / (x / 3) = 51) : x = 180 := 
by 
  sorry

end NUMINAMATH_GPT_find_number_l2143_214381


namespace NUMINAMATH_GPT_fraction_first_to_second_l2143_214345

def digit_fraction_proof_problem (a b c d : ℕ) (number : ℕ) :=
  number = 1349 ∧
  a = b / 3 ∧
  c = a + b ∧
  d = 3 * b

theorem fraction_first_to_second (a b c d : ℕ) (number : ℕ) :
  digit_fraction_proof_problem a b c d number → a / b = 1 / 3 :=
by
  intro problem
  sorry

end NUMINAMATH_GPT_fraction_first_to_second_l2143_214345


namespace NUMINAMATH_GPT_robin_gum_count_l2143_214300

theorem robin_gum_count (initial_gum : ℝ) (additional_gum : ℝ) (final_gum : ℝ) 
  (h1 : initial_gum = 18.0) (h2 : additional_gum = 44.0) : final_gum = 62.0 :=
by {
  sorry
}

end NUMINAMATH_GPT_robin_gum_count_l2143_214300


namespace NUMINAMATH_GPT_part_I_part_II_l2143_214337

noncomputable def f (x a : ℝ) : ℝ := |2 * x + 1| - |x - a|

-- Problem (I)
theorem part_I (x : ℝ) : 
  (f x 4) > 2 ↔ (x < -7 ∨ x > 5 / 3) :=
sorry

-- Problem (II)
theorem part_II (a : ℝ) :
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 → f x a ≥ |x - 4|) ↔ -1 ≤ a ∧ a ≤ 5 :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l2143_214337


namespace NUMINAMATH_GPT_polynomial_simplification_l2143_214317

theorem polynomial_simplification (x : ℝ) : 
  (x * (x * (2 - x) - 4) + 10) + 1 = -x^4 + 2 * x^3 - 4 * x^2 + 10 * x + 1 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_simplification_l2143_214317


namespace NUMINAMATH_GPT_find_original_price_l2143_214306

def initial_price (P : ℝ) : Prop :=
  let first_discount := P * 0.76
  let second_discount := first_discount * 0.85
  let final_price := second_discount * 1.10
  final_price = 532

theorem find_original_price : ∃ P : ℝ, initial_price P :=
sorry

end NUMINAMATH_GPT_find_original_price_l2143_214306


namespace NUMINAMATH_GPT_cube_inequality_contradiction_l2143_214305

variable {x y : ℝ}

theorem cube_inequality_contradiction (h : x < y) (hne : x^3 ≥ y^3) : false :=
by 
  sorry

end NUMINAMATH_GPT_cube_inequality_contradiction_l2143_214305


namespace NUMINAMATH_GPT_equivalent_problem_l2143_214380

noncomputable def problem_statement : Prop :=
  ∀ (a b c d : ℝ), a ≠ -1 → b ≠ -1 → c ≠ -1 → d ≠ -1 →
  ∀ (ω : ℂ), ω^4 = 1 → ω ≠ 1 →
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 4 / (1 + ω)) →
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2)

#check problem_statement

-- Expected output for type checking without providing the proof
theorem equivalent_problem : problem_statement :=
  sorry

end NUMINAMATH_GPT_equivalent_problem_l2143_214380


namespace NUMINAMATH_GPT_lego_set_cost_l2143_214309

-- Definitions and conditions
def price_per_car := 5
def cars_sold := 3
def action_figures_sold := 2
def total_earnings := 120

-- Derived prices
def price_per_action_figure := 2 * price_per_car
def price_per_board_game := price_per_action_figure + price_per_car

-- Total cost of sold items (cars, action figures, and board game)
def total_cost_of_sold_items := 
  (cars_sold * price_per_car) + 
  (action_figures_sold * price_per_action_figure) + 
  price_per_board_game

-- Cost of Lego set
theorem lego_set_cost : 
  total_earnings - total_cost_of_sold_items = 70 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_lego_set_cost_l2143_214309


namespace NUMINAMATH_GPT_greatest_n_le_5_value_ge_2525_l2143_214393

theorem greatest_n_le_5_value_ge_2525 (n : ℤ) (V : ℤ) 
  (h1 : 101 * n^2 ≤ V) 
  (h2 : ∀ k : ℤ, (101 * k^2 ≤ V) → (k ≤ 5)) : 
  V ≥ 2525 := 
sorry

end NUMINAMATH_GPT_greatest_n_le_5_value_ge_2525_l2143_214393


namespace NUMINAMATH_GPT_peter_drew_8_pictures_l2143_214336

theorem peter_drew_8_pictures : 
  ∃ (P : ℕ), ∀ (Q R : ℕ), Q = P + 20 → R = 5 → R + P + Q = 41 → P = 8 :=
by
  sorry

end NUMINAMATH_GPT_peter_drew_8_pictures_l2143_214336


namespace NUMINAMATH_GPT_nontrivial_solution_exists_l2143_214391

theorem nontrivial_solution_exists
  (a b c : ℝ) :
  (∃ x y z : ℝ, (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧ 
    a * x + b * y + c * z = 0 ∧ 
    b * x + c * y + a * z = 0 ∧ 
    c * x + a * y + b * z = 0) ↔ (a + b + c = 0 ∨ a = b ∧ b = c) := 
sorry

end NUMINAMATH_GPT_nontrivial_solution_exists_l2143_214391


namespace NUMINAMATH_GPT_vendor_sells_50_percent_on_first_day_l2143_214361

variables (A : ℝ) (S : ℝ)

theorem vendor_sells_50_percent_on_first_day 
  (h : 0.2 * A * (1 - S) + 0.5 * A * (1 - S) * 0.8 = 0.3 * A) : S = 0.5 :=
  sorry

end NUMINAMATH_GPT_vendor_sells_50_percent_on_first_day_l2143_214361


namespace NUMINAMATH_GPT_tan_435_eq_2_plus_sqrt3_l2143_214341

open Real

theorem tan_435_eq_2_plus_sqrt3 : tan (435 * (π / 180)) = 2 + sqrt 3 :=
  sorry

end NUMINAMATH_GPT_tan_435_eq_2_plus_sqrt3_l2143_214341


namespace NUMINAMATH_GPT_maximize_sector_area_l2143_214383

noncomputable def max_area_sector_angle (r : ℝ) (l := 36 - 2 * r) (α := l / r) : ℝ :=
  α

theorem maximize_sector_area (h : ∀ r : ℝ, 2 * r + 36 - 2 * r = 36) :
  max_area_sector_angle 9 = 2 :=
by
  sorry

end NUMINAMATH_GPT_maximize_sector_area_l2143_214383


namespace NUMINAMATH_GPT_total_flowers_collected_l2143_214339

/- Definitions for the given conditions -/
def maxFlowers : ℕ := 50
def arwenTulips : ℕ := 20
def arwenRoses : ℕ := 18
def arwenSunflowers : ℕ := 6

def elrondTulips : ℕ := 2 * arwenTulips
def elrondRoses : ℕ := if 3 * arwenRoses + elrondTulips > maxFlowers then maxFlowers - elrondTulips else 3 * arwenRoses

def galadrielTulips : ℕ := if 3 * elrondTulips > maxFlowers then maxFlowers else 3 * elrondTulips
def galadrielRoses : ℕ := if 2 * arwenRoses + galadrielTulips > maxFlowers then maxFlowers - galadrielTulips else 2 * arwenRoses

def galadrielSunflowers : ℕ := 0 -- she didn't pick any sunflowers
def legolasSunflowers : ℕ := arwenSunflowers + galadrielSunflowers
def legolasRemaining : ℕ := maxFlowers - legolasSunflowers
def legolasRosesAndTulips : ℕ := legolasRemaining / 2
def legolasTulips : ℕ := legolasRosesAndTulips
def legolasRoses : ℕ := legolasRosesAndTulips

def arwenTotal : ℕ := arwenTulips + arwenRoses + arwenSunflowers
def elrondTotal : ℕ := elrondTulips + elrondRoses
def galadrielTotal : ℕ := galadrielTulips + galadrielRoses + galadrielSunflowers
def legolasTotal : ℕ := legolasTulips + legolasRoses + legolasSunflowers

def totalFlowers : ℕ := arwenTotal + elrondTotal + galadrielTotal + legolasTotal

theorem total_flowers_collected : totalFlowers = 194 := by
  /- This will be where the proof goes, but we leave it as a placeholder. -/
  sorry

end NUMINAMATH_GPT_total_flowers_collected_l2143_214339


namespace NUMINAMATH_GPT_graph_of_equation_l2143_214399

theorem graph_of_equation (x y : ℝ) : 
  (x - y)^2 = x^2 + y^2 ↔ (x = 0 ∨ y = 0) := 
by 
  sorry

end NUMINAMATH_GPT_graph_of_equation_l2143_214399


namespace NUMINAMATH_GPT_angle_symmetry_l2143_214332

theorem angle_symmetry (α β : ℝ) (hα : 0 < α ∧ α < 2 * Real.pi) (hβ : 0 < β ∧ β < 2 * Real.pi) (h_symm : α = 2 * Real.pi - β) : α + β = 2 * Real.pi := 
by 
  sorry

end NUMINAMATH_GPT_angle_symmetry_l2143_214332


namespace NUMINAMATH_GPT_complement_U_A_is_singleton_one_l2143_214314

-- Define the universe and subset
def U : Set ℝ := Set.Icc 0 1
def A : Set ℝ := Set.Ico 0 1

-- Define the complement of A relative to U
def complement_U_A : Set ℝ := U \ A

-- Theorem statement
theorem complement_U_A_is_singleton_one : complement_U_A = {1} := by
  sorry

end NUMINAMATH_GPT_complement_U_A_is_singleton_one_l2143_214314


namespace NUMINAMATH_GPT_hire_charges_paid_by_B_l2143_214371

theorem hire_charges_paid_by_B (total_cost : ℝ) (hours_A hours_B hours_C : ℝ) (b_payment : ℝ) :
  total_cost = 720 ∧ hours_A = 9 ∧ hours_B = 10 ∧ hours_C = 13 ∧ b_payment = (total_cost / (hours_A + hours_B + hours_C)) * hours_B → b_payment = 225 :=
by
  sorry

end NUMINAMATH_GPT_hire_charges_paid_by_B_l2143_214371


namespace NUMINAMATH_GPT_percentage_games_won_l2143_214349

theorem percentage_games_won 
  (P_first : ℝ)
  (P_remaining : ℝ)
  (total_games : ℕ)
  (H1 : P_first = 0.7)
  (H2 : P_remaining = 0.5)
  (H3 : total_games = 100) :
  True :=
by
  -- To prove the percentage of games won is 70%
  have percentage_won : ℝ := P_first
  have : percentage_won * 100 = 70 := by sorry
  trivial

end NUMINAMATH_GPT_percentage_games_won_l2143_214349


namespace NUMINAMATH_GPT_gift_wrapping_combinations_l2143_214322

theorem gift_wrapping_combinations :
  (10 * 5 * 6 * 2 = 600) :=
by
  sorry

end NUMINAMATH_GPT_gift_wrapping_combinations_l2143_214322


namespace NUMINAMATH_GPT_sum_of_solutions_l2143_214387

theorem sum_of_solutions (s : Finset ℝ) :
  (∀ x ∈ s, |x^2 - 16 * x + 60| = 4) →
  s.sum id = 24 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l2143_214387


namespace NUMINAMATH_GPT_hex_B2F_to_dec_l2143_214352

theorem hex_B2F_to_dec : 
  let A := 10
  let B := 11
  let C := 12
  let D := 13
  let E := 14
  let F := 15
  let base := 16
  let b2f := B * base^2 + 2 * base^1 + F * base^0
  b2f = 2863 :=
by {
  sorry
}

end NUMINAMATH_GPT_hex_B2F_to_dec_l2143_214352


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2143_214364

noncomputable def A : Set ℝ := {x : ℝ | x^2 - 1 ≤ 0}
noncomputable def B : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l2143_214364


namespace NUMINAMATH_GPT_pictures_per_album_l2143_214379

-- Define the conditions
def uploaded_pics_phone : ℕ := 22
def uploaded_pics_camera : ℕ := 2
def num_albums : ℕ := 4

-- Define the total pictures uploaded
def total_pictures : ℕ := uploaded_pics_phone + uploaded_pics_camera

-- Define the target statement as the theorem
theorem pictures_per_album : (total_pictures / num_albums) = 6 := by
  sorry

end NUMINAMATH_GPT_pictures_per_album_l2143_214379


namespace NUMINAMATH_GPT_find_fraction_of_number_l2143_214369

theorem find_fraction_of_number (N : ℚ) (h : (3/10 : ℚ) * N - 8 = 12) :
  (1/5 : ℚ) * N = 40 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_fraction_of_number_l2143_214369


namespace NUMINAMATH_GPT_calculate_expression_l2143_214396

theorem calculate_expression : 
  3 * 995 + 4 * 996 + 5 * 997 + 6 * 998 + 7 * 999 - 4985 * 3 = 9980 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2143_214396


namespace NUMINAMATH_GPT_boxed_meals_solution_count_l2143_214319

theorem boxed_meals_solution_count :
  ∃ n : ℕ, n = 4 ∧ 
  ∃ x y z : ℕ, 
      x + y + z = 22 ∧ 
      10 * x + 8 * y + 5 * z = 183 ∧ 
      x > 0 ∧ y > 0 ∧ z > 0 :=
sorry

end NUMINAMATH_GPT_boxed_meals_solution_count_l2143_214319


namespace NUMINAMATH_GPT_range_of_a_l2143_214397

def solution_set_non_empty (a : ℝ) : Prop :=
  ∃ x : ℝ, |x - 3| + |x - 4| < a

theorem range_of_a (a : ℝ) : solution_set_non_empty a ↔ a > 1 := sorry

end NUMINAMATH_GPT_range_of_a_l2143_214397


namespace NUMINAMATH_GPT_shekar_math_marks_l2143_214324

variable (science socialStudies english biology average : ℕ)

theorem shekar_math_marks 
  (h1 : science = 65)
  (h2 : socialStudies = 82)
  (h3 : english = 67)
  (h4 : biology = 95)
  (h5 : average = 77) :
  ∃ M, average = (science + socialStudies + english + biology + M) / 5 ∧ M = 76 :=
by
  sorry

end NUMINAMATH_GPT_shekar_math_marks_l2143_214324


namespace NUMINAMATH_GPT_find_abc_l2143_214353

theorem find_abc (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : a^3 + b^3 + c^3 = 2001 → (a = 10 ∧ b = 10 ∧ c = 1) ∨ (a = 10 ∧ b = 1 ∧ c = 10) ∨ (a = 1 ∧ b = 10 ∧ c = 10) := 
sorry

end NUMINAMATH_GPT_find_abc_l2143_214353


namespace NUMINAMATH_GPT_christine_amount_l2143_214346

theorem christine_amount (S C : ℕ) 
  (h1 : S + C = 50)
  (h2 : C = S + 30) :
  C = 40 :=
by
  -- Proof goes here.
  -- This part should be filled in to complete the proof.
  sorry

end NUMINAMATH_GPT_christine_amount_l2143_214346


namespace NUMINAMATH_GPT_ratio_of_areas_is_two_thirds_l2143_214310

noncomputable def PQ := 10
noncomputable def PR := 6
noncomputable def QR := 4
noncomputable def r_PQ := PQ / 2
noncomputable def r_PR := PR / 2
noncomputable def r_QR := QR / 2
noncomputable def area_semi_PQ := (1 / 2) * Real.pi * r_PQ^2
noncomputable def area_semi_PR := (1 / 2) * Real.pi * r_PR^2
noncomputable def area_semi_QR := (1 / 2) * Real.pi * r_QR^2
noncomputable def shaded_area := (area_semi_PQ - area_semi_PR) + area_semi_QR
noncomputable def total_area_circle := Real.pi * r_PQ^2
noncomputable def unshaded_area := total_area_circle - shaded_area
noncomputable def ratio := shaded_area / unshaded_area

theorem ratio_of_areas_is_two_thirds : ratio = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_is_two_thirds_l2143_214310


namespace NUMINAMATH_GPT_relationship_f_neg2_f_expr_l2143_214395

noncomputable def f : ℝ → ℝ := sorry  -- f is some function ℝ → ℝ, the exact definition is not provided

axiom even_function : ∀ x : ℝ, f (-x) = f x -- f is an even function
axiom increasing_on_negatives : ∀ x y : ℝ, x < y ∧ y < 0 → f x < f y -- f is increasing on (-∞, 0)

theorem relationship_f_neg2_f_expr (a : ℝ) : f (-2) ≥ f (a^2 - 4 * a + 6) := by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_relationship_f_neg2_f_expr_l2143_214395


namespace NUMINAMATH_GPT_vec_op_not_comm_l2143_214357

open Real

-- Define the operation ⊙
def vec_op (a b: ℝ × ℝ) : ℝ :=
  (a.1 * b.2) - (a.2 * b.1)

-- Define a predicate to check if two vectors are collinear
def collinear (a b: ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

-- Define the proof theorem
theorem vec_op_not_comm (a b: ℝ × ℝ) : vec_op a b ≠ vec_op b a :=
by
  -- The contents of the proof will go here. Insert 'sorry' to skip.
  sorry

end NUMINAMATH_GPT_vec_op_not_comm_l2143_214357


namespace NUMINAMATH_GPT_ingrid_tax_rate_proof_l2143_214389

namespace TaxProblem

-- Define the given conditions
def john_income : ℝ := 56000
def ingrid_income : ℝ := 72000
def combined_income := john_income + ingrid_income

def john_tax_rate : ℝ := 0.30
def combined_tax_rate : ℝ := 0.35625

-- Calculate John's tax
def john_tax := john_tax_rate * john_income

-- Calculate total tax paid
def total_tax_paid := combined_tax_rate * combined_income

-- Calculate Ingrid's tax
def ingrid_tax := total_tax_paid - john_tax

-- Prove Ingrid's tax rate
theorem ingrid_tax_rate_proof (r : ℝ) :
  (ingrid_tax / ingrid_income) * 100 = 40 :=
  by sorry

end TaxProblem

end NUMINAMATH_GPT_ingrid_tax_rate_proof_l2143_214389


namespace NUMINAMATH_GPT_sum_of_values_of_n_l2143_214335

theorem sum_of_values_of_n (n : ℚ) (h : |3 * n - 4| = 6) : 
  (n = 10 / 3 ∨ n = -2 / 3) → (10 / 3 + -2 / 3 = 8 / 3) :=
sorry

end NUMINAMATH_GPT_sum_of_values_of_n_l2143_214335


namespace NUMINAMATH_GPT_alice_bob_meet_l2143_214312

/--
Alice and Bob play a game on a circle divided into 18 equally-spaced points.
Alice moves 7 points clockwise per turn, and Bob moves 13 points counterclockwise.
Prove that they will meet at the same point after 9 turns.
-/
theorem alice_bob_meet : ∃ k : ℕ, k = 9 ∧ (7 * k) % 18 = (18 - 13 * k) % 18 :=
by
  sorry

end NUMINAMATH_GPT_alice_bob_meet_l2143_214312


namespace NUMINAMATH_GPT_higher_selling_price_is_463_l2143_214386

-- Definitions and conditions
def cost_price : ℝ := 400
def selling_price_340 : ℝ := 340
def loss_340 : ℝ := selling_price_340 - cost_price
def gain_percent : ℝ := 0.05
def additional_gain : ℝ := gain_percent * -loss_340
def expected_gain := -loss_340 + additional_gain

-- Theorem to prove that the higher selling price is 463
theorem higher_selling_price_is_463 : ∃ P : ℝ, P = cost_price + expected_gain ∧ P = 463 :=
by
  sorry

end NUMINAMATH_GPT_higher_selling_price_is_463_l2143_214386


namespace NUMINAMATH_GPT_positive_difference_between_solutions_l2143_214373

theorem positive_difference_between_solutions : 
  let f (x : ℝ) := (5 - (x^2 / 3 : ℝ))^(1 / 3 : ℝ)
  let a := 4 * Real.sqrt 6
  let b := -4 * Real.sqrt 6
  |a - b| = 8 * Real.sqrt 6 := 
by 
  sorry

end NUMINAMATH_GPT_positive_difference_between_solutions_l2143_214373


namespace NUMINAMATH_GPT_total_amount_spent_l2143_214384

def speakers : ℝ := 118.54
def new_tires : ℝ := 106.33
def window_tints : ℝ := 85.27
def seat_covers : ℝ := 79.99
def scheduled_maintenance : ℝ := 199.75
def steering_wheel_cover : ℝ := 15.63
def air_fresheners_set : ℝ := 12.96
def car_wash : ℝ := 25.0

theorem total_amount_spent :
  speakers + new_tires + window_tints + seat_covers + scheduled_maintenance + steering_wheel_cover + air_fresheners_set + car_wash = 643.47 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_spent_l2143_214384


namespace NUMINAMATH_GPT_find_ordered_pair_l2143_214315

theorem find_ordered_pair (x y : ℚ) 
  (h1 : 3 * x - 18 * y = 2) 
  (h2 : 4 * y - x = 6) :
  x = -58 / 3 ∧ y = -10 / 3 :=
sorry

end NUMINAMATH_GPT_find_ordered_pair_l2143_214315


namespace NUMINAMATH_GPT_line_passes_through_center_l2143_214320

-- Define the equation of the circle as given in the problem.
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 6*y + 8 = 0

-- Define the center of the circle.
def center_of_circle (x y : ℝ) : Prop := x = 1 ∧ y = -3

-- Define the equation of the line.
def line_equation (x y : ℝ) : Prop := 2*x + y + 1 = 0

-- The theorem to prove.
theorem line_passes_through_center :
  (∃ x y, circle_equation x y ∧ center_of_circle x y) →
  (∃ x y, center_of_circle x y ∧ line_equation x y) :=
by
  sorry

end NUMINAMATH_GPT_line_passes_through_center_l2143_214320


namespace NUMINAMATH_GPT_toy_car_production_l2143_214385

theorem toy_car_production (yesterday today total : ℕ) 
  (hy : yesterday = 60)
  (ht : today = 2 * yesterday) :
  total = yesterday + today :=
by
  sorry

end NUMINAMATH_GPT_toy_car_production_l2143_214385


namespace NUMINAMATH_GPT_ratio_of_adult_to_kid_charge_l2143_214377

variable (A : ℝ)  -- Charge for adults

-- Conditions
def kids_charge : ℝ := 3
def num_kids_per_day : ℝ := 8
def num_adults_per_day : ℝ := 10
def weekly_earnings : ℝ := 588
def days_per_week : ℝ := 7

-- Hypothesis for the relationship between charges and total weekly earnings
def total_weekly_earnings_eq : Prop :=
  days_per_week * (num_kids_per_day * kids_charge + num_adults_per_day * A) = weekly_earnings

-- Statement to be proved
theorem ratio_of_adult_to_kid_charge (h : total_weekly_earnings_eq A) : (A / kids_charge) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_adult_to_kid_charge_l2143_214377


namespace NUMINAMATH_GPT_find_multiple_l2143_214392

theorem find_multiple (x m : ℤ) (hx : x = 13) (h : x + x + 2 * x + m * x = 104) : m = 4 :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_find_multiple_l2143_214392


namespace NUMINAMATH_GPT_apples_purchased_l2143_214342

variable (A : ℕ) -- Let A be the number of kg of apples purchased.

-- Conditions
def cost_of_apples (A : ℕ) : ℕ := 70 * A
def cost_of_mangoes : ℕ := 45 * 9
def total_amount_paid : ℕ := 965

-- Theorem to prove that A == 8
theorem apples_purchased
  (h : cost_of_apples A + cost_of_mangoes = total_amount_paid) :
  A = 8 := by
sorry

end NUMINAMATH_GPT_apples_purchased_l2143_214342


namespace NUMINAMATH_GPT_trapezoid_distances_l2143_214378

-- Define the problem parameters
variables (AB CD AD BC : ℝ)
-- Assume given conditions
axiom h1 : AD > BC
noncomputable def k := AD / BC

-- Formalizing the proof problem in Lean 4
theorem trapezoid_distances (M : Type) (BM AM CM DM : ℝ) :
  BM = AB * BC / (AD - BC) →
  AM = AB * AD / (AD - BC) →
  CM = CD * BC / (AD - BC) →
  DM = CD * AD / (AD - BC) →
  true :=
sorry

end NUMINAMATH_GPT_trapezoid_distances_l2143_214378


namespace NUMINAMATH_GPT_range_x_minus_y_compare_polynomials_l2143_214301

-- Proof Problem 1: Range of x - y
theorem range_x_minus_y (x y : ℝ) (hx : -1 < x ∧ x < 4) (hy : 2 < y ∧ y < 3) : 
  -4 < x - y ∧ x - y < 2 := 
  sorry

-- Proof Problem 2: Comparison of polynomials
theorem compare_polynomials (x : ℝ) : 
  (x - 1) * (x^2 + x + 1) < (x + 1) * (x^2 - x + 1) := 
  sorry

end NUMINAMATH_GPT_range_x_minus_y_compare_polynomials_l2143_214301


namespace NUMINAMATH_GPT_negation_of_P_is_true_l2143_214321

theorem negation_of_P_is_true :
  ¬ (∃ x : ℝ, x^2 + 1 < 2 * x) :=
by sorry

end NUMINAMATH_GPT_negation_of_P_is_true_l2143_214321


namespace NUMINAMATH_GPT_min_total_books_l2143_214311

-- Definitions based on conditions
variables (P C B : ℕ)

-- Condition 1: Ratio of physics to chemistry books is 3:2
def ratio_physics_chemistry := 3 * C = 2 * P

-- Condition 2: Ratio of chemistry to biology books is 4:3
def ratio_chemistry_biology := 4 * B = 3 * C

-- Condition 3: Total number of books is 3003
def total_books := P + C + B = 3003

-- The theorem to prove
theorem min_total_books (h1 : ratio_physics_chemistry P C) (h2 : ratio_chemistry_biology C B) (h3: total_books P C B) :
  3003 = 3003 :=
by
  sorry

end NUMINAMATH_GPT_min_total_books_l2143_214311


namespace NUMINAMATH_GPT_find_number_l2143_214376

theorem find_number (some_number : ℤ) (h : some_number + 9 = 54) : some_number = 45 :=
sorry

end NUMINAMATH_GPT_find_number_l2143_214376


namespace NUMINAMATH_GPT_mental_math_competition_l2143_214348

theorem mental_math_competition :
  -- The number of teams that participated is 4
  (∃ (teams : ℕ) (numbers : List ℕ),
     -- Each team received a number that can be written as 15M + 11m where M is the largest odd divisor
     -- and m is the smallest odd divisor greater than 1.
     teams = 4 ∧ 
     numbers = [528, 880, 1232, 1936] ∧
     ∀ n ∈ numbers,
       ∃ M m, M > 1 ∧ m > 1 ∧
       M % 2 = 1 ∧ m % 2 = 1 ∧
       (∀ d, d ∣ n → (d % 2 = 1 → M ≥ d)) ∧ 
       (∀ d, d ∣ n → (d % 2 = 1 ∧ d > 1 → m ≤ d)) ∧
       n = 15 * M + 11 * m) :=
sorry

end NUMINAMATH_GPT_mental_math_competition_l2143_214348


namespace NUMINAMATH_GPT_problem_statement_l2143_214358

-- Definitions for the conditions in the problem
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p
def has_three_divisors (k : ℕ) : Prop := 
  ∃ p : ℕ, is_prime p ∧ k = p^2

-- Given conditions
def m : ℕ := 3 -- the smallest odd prime
def n : ℕ := 49 -- the largest integer less than 50 with exactly three positive divisors

-- The proof statement
theorem problem_statement : m + n = 52 :=
by sorry

end NUMINAMATH_GPT_problem_statement_l2143_214358


namespace NUMINAMATH_GPT_option_D_correct_l2143_214394

theorem option_D_correct (y : ℝ) : -9 * y^2 + 16 * y^2 = 7 * y^2 :=
by sorry

end NUMINAMATH_GPT_option_D_correct_l2143_214394


namespace NUMINAMATH_GPT_identify_incorrect_propositions_l2143_214362

-- Definitions for parallel lines and planes
def line := Type -- Define a line type
def plane := Type -- Define a plane type
def parallel_to (l1 l2 : line) : Prop := sorry -- Assume a definition for parallel lines
def parallel_to_plane (l : line) (pl : plane) : Prop := sorry -- Assume a definition for a line parallel to a plane
def contained_in (l : line) (pl : plane) : Prop := sorry -- Assume a definition for a line contained in a plane

theorem identify_incorrect_propositions (a b : line) (α : plane) :
  (parallel_to_plane a α ∧ parallel_to_plane b α → ¬parallel_to a b) ∧
  (parallel_to_plane a α ∧ contained_in b α → ¬parallel_to a b) ∧
  (parallel_to a b ∧ contained_in b α → ¬parallel_to_plane a α) ∧
  (parallel_to a b ∧ parallel_to_plane b α → ¬parallel_to_plane a α) :=
by
  sorry -- The proof is not required

end NUMINAMATH_GPT_identify_incorrect_propositions_l2143_214362


namespace NUMINAMATH_GPT_dot_product_of_a_b_l2143_214307

theorem dot_product_of_a_b 
  (a b : ℝ)
  (θ : ℝ)
  (ha : a = 2 * Real.sin (15 * Real.pi / 180))
  (hb : b = 4 * Real.cos (15 * Real.pi / 180))
  (hθ : θ = 30 * Real.pi / 180) :
  (a * b * Real.cos θ) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_dot_product_of_a_b_l2143_214307


namespace NUMINAMATH_GPT_distinct_digits_and_difference_is_945_l2143_214343

theorem distinct_digits_and_difference_is_945 (a b c : ℕ)
  (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_difference : 10 * (100 * a + 10 * b + c) + 2 - (2000 + 100 * a + 10 * b + c) = 945) :
  (100 * a + 10 * b + c) = 327 :=
by
  sorry

end NUMINAMATH_GPT_distinct_digits_and_difference_is_945_l2143_214343
