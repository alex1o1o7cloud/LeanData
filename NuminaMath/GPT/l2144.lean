import Mathlib

namespace NUMINAMATH_GPT_second_number_in_first_set_l2144_214403

theorem second_number_in_first_set :
  ∃ (x : ℝ), (20 + x + 60) / 3 = (10 + 80 + 15) / 3 + 5 ∧ x = 40 :=
by
  use 40
  sorry

end NUMINAMATH_GPT_second_number_in_first_set_l2144_214403


namespace NUMINAMATH_GPT_sum_of_solutions_eq_neg2_l2144_214439

noncomputable def sum_of_real_solutions (a : ℝ) (h : a > 2) : ℝ :=
  -2

theorem sum_of_solutions_eq_neg2 (a : ℝ) (h : a > 2) :
  sum_of_real_solutions a h = -2 := sorry

end NUMINAMATH_GPT_sum_of_solutions_eq_neg2_l2144_214439


namespace NUMINAMATH_GPT_leftmost_three_nonzero_digits_of_arrangements_l2144_214404

-- Definitions based on the conditions
def num_rings := 10
def chosen_rings := 6
def num_fingers := 5

-- Calculate the possible arrangements
def arrangements : ℕ := Nat.choose num_rings chosen_rings * Nat.factorial chosen_rings * Nat.choose (chosen_rings + (num_fingers - 1)) (num_fingers - 1)

-- Find the leftmost three nonzero digits
def leftmost_three_nonzero_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  (digits.reverse.takeWhile (· > 0)).reverse.take 3
  |> List.foldl (· + · * 10) 0
  
-- The main theorem to prove
theorem leftmost_three_nonzero_digits_of_arrangements :
  leftmost_three_nonzero_digits arrangements = 317 :=
by
  sorry

end NUMINAMATH_GPT_leftmost_three_nonzero_digits_of_arrangements_l2144_214404


namespace NUMINAMATH_GPT_fractional_pizza_eaten_after_six_trips_l2144_214489

def pizza_eaten : ℚ := (1/3) * (1 - (2/3)^6) / (1 - 2/3)

theorem fractional_pizza_eaten_after_six_trips : pizza_eaten = 665 / 729 :=
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_fractional_pizza_eaten_after_six_trips_l2144_214489


namespace NUMINAMATH_GPT_grey_pairs_coincide_l2144_214497

theorem grey_pairs_coincide (h₁ : 4 = orange_count / 2) 
                                (h₂ : 6 = green_count / 2)
                                (h₃ : 9 = grey_count / 2)
                                (h₄ : 3 = orange_pairs)
                                (h₅ : 4 = green_pairs)
                                (h₆ : 1 = orange_grey_pairs) :
    grey_pairs = 6 := by
  sorry

noncomputable def half_triangle_counts : (ℕ × ℕ × ℕ) := (4, 6, 9)

noncomputable def triangle_pairs : (ℕ × ℕ × ℕ) := (3, 4, 1)

noncomputable def prove_grey_pairs (orange_count green_count grey_count : ℕ)
                                   (orange_pairs green_pairs orange_grey_pairs : ℕ) : ℕ :=
  sorry

end NUMINAMATH_GPT_grey_pairs_coincide_l2144_214497


namespace NUMINAMATH_GPT_interval_of_decrease_l2144_214450

def quadratic (x : ℝ) := 3 * x^2 - 7 * x + 2

def decreasing_interval (y : ℝ) := y < 2 / 3

theorem interval_of_decrease :
  {x : ℝ | x < (1 / 3)} = {x : ℝ | x < (1 / 3)} :=
by sorry

end NUMINAMATH_GPT_interval_of_decrease_l2144_214450


namespace NUMINAMATH_GPT_length_of_one_side_of_hexagon_l2144_214435

variable (P : ℝ) (n : ℕ)
-- Condition: perimeter P is 60 inches
def hexagon_perimeter_condition : Prop := P = 60
-- Hexagon has six sides
def hexagon_sides_condition : Prop := n = 6
-- The question asks for the side length
noncomputable def side_length_of_hexagon : ℝ := P / n

-- Prove that if a hexagon has a perimeter of 60 inches, then its side length is 10 inches
theorem length_of_one_side_of_hexagon (hP : hexagon_perimeter_condition P) (hn : hexagon_sides_condition n) :
  side_length_of_hexagon P n = 10 := by
  sorry

end NUMINAMATH_GPT_length_of_one_side_of_hexagon_l2144_214435


namespace NUMINAMATH_GPT_systems_solution_l2144_214455

    theorem systems_solution : 
      (∃ x y : ℝ, 2 * x + 5 * y = -26 ∧ 3 * x - 5 * y = 36 ∧ 
                 (∃ a b : ℝ, a * x - b * y = -4 ∧ b * x + a * y = -8 ∧ 
                 (2 * a + b) ^ 2020 = 1)) := 
    by
      sorry
    
end NUMINAMATH_GPT_systems_solution_l2144_214455


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l2144_214419

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, |x| + x^2 ≥ 0)) ↔ (∃ x : ℝ, |x| + x^2 < 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l2144_214419


namespace NUMINAMATH_GPT_total_computers_sold_l2144_214471

theorem total_computers_sold (T : ℕ) (h_half_sales_laptops : 2 * T / 2 = T)
        (h_third_sales_netbooks : 3 * T / 3 = T)
        (h_desktop_sales : T - T / 2 - T / 3 = 12) : T = 72 :=
by
  sorry

end NUMINAMATH_GPT_total_computers_sold_l2144_214471


namespace NUMINAMATH_GPT_solve_inequality_l2144_214414

theorem solve_inequality (k x : ℝ) :
  (x^2 > (k + 1) * x - k) ↔ 
  (if k > 1 then (x < 1 ∨ x > k)
   else if k = 1 then (x ≠ 1)
   else (x < k ∨ x > 1)) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l2144_214414


namespace NUMINAMATH_GPT_square_difference_l2144_214498

theorem square_difference (a b : ℝ) (h : a > b) :
  ∃ c : ℝ, c^2 = a^2 - b^2 :=
by
  sorry

end NUMINAMATH_GPT_square_difference_l2144_214498


namespace NUMINAMATH_GPT_range_of_a_if_distinct_zeros_l2144_214459

theorem range_of_a_if_distinct_zeros (a : ℝ) :
(∀ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁ ∧ (x₁^3 - 3*x₁ + a = 0) ∧ (x₂^3 - 3*x₂ + a = 0) ∧ (x₃^3 - 3*x₃ + a = 0)) → -2 < a ∧ a < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_if_distinct_zeros_l2144_214459


namespace NUMINAMATH_GPT_second_train_length_l2144_214423

theorem second_train_length
  (L1 : ℝ) (V1 : ℝ) (V2 : ℝ) (T : ℝ)
  (h1 : L1 = 300)
  (h2 : V1 = 72 * 1000 / 3600)
  (h3 : V2 = 36 * 1000 / 3600)
  (h4 : T = 79.99360051195904) :
  L1 + (V1 - V2) * T = 799.9360051195904 :=
by
  sorry

end NUMINAMATH_GPT_second_train_length_l2144_214423


namespace NUMINAMATH_GPT_sum_of_ages_l2144_214441

def Maria_age (E : ℕ) : ℕ := E + 7

theorem sum_of_ages (M E : ℕ) (h1 : M = E + 7) (h2 : M + 10 = 3 * (E - 5)) :
  M + E = 39 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_ages_l2144_214441


namespace NUMINAMATH_GPT_digit_sum_is_twelve_l2144_214415

theorem digit_sum_is_twelve (n x y : ℕ) (h1 : n = 10 * x + y) (h2 : 0 ≤ x ∧ x ≤ 9) (h3 : 0 ≤ y ∧ y ≤ 9)
  (h4 : (1 / 2 : ℚ) * n = (1 / 4 : ℚ) * n + 3) : x + y = 12 :=
by
  sorry

end NUMINAMATH_GPT_digit_sum_is_twelve_l2144_214415


namespace NUMINAMATH_GPT_max_ratio_BO_BM_l2144_214469

theorem max_ratio_BO_BM
  (C : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hC : C = (0, -4))
  (hCir : ∃ (P : ℝ × ℝ), (P.1 - 2)^2 + (P.2 - 4)^2 = 1 ∧ A = ((P.1 + C.1) / 2, (P.2 + C.2) / 2))
  (hPar : ∃ (x y : ℝ), B = (x, y) ∧ y^2 = 4 * x) :
  ∃ t, t = (4 * Real.sqrt 7)/7 ∧ t = Real.sqrt ((B.1^2 + 4 * B.1)/((B.1 + 1/2)^2)) := by
  -- Given conditions and definitions
  obtain ⟨P, hP, hA⟩ := hCir
  obtain ⟨x, y, hB⟩ := hPar
  use (4 * Real.sqrt 7) / 7
  sorry

end NUMINAMATH_GPT_max_ratio_BO_BM_l2144_214469


namespace NUMINAMATH_GPT_xy_equals_nine_l2144_214465

theorem xy_equals_nine (x y : ℝ) (h : x * (x + 2 * y) = x ^ 2 + 18) : x * y = 9 :=
by
  sorry

end NUMINAMATH_GPT_xy_equals_nine_l2144_214465


namespace NUMINAMATH_GPT_simplify_vector_eq_l2144_214488

-- Define points A, B, C
variables {A B C O : Type} [AddGroup A]

-- Define vector operations corresponding to overrightarrow.
variables (AB OC OB AC AO BO : A)

-- Conditions in Lean definitions
-- Assuming properties like vector addition and subtraction, and associative properties
def vector_eq : Prop := AB + OC - OB = AC

theorem simplify_vector_eq :
  AB + OC - OB = AC :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_simplify_vector_eq_l2144_214488


namespace NUMINAMATH_GPT_find_principal_l2144_214461

def r : ℝ := 0.03
def t : ℝ := 3
def I (P : ℝ) : ℝ := P - 1820
def simple_interest (P : ℝ) : ℝ := P * r * t

theorem find_principal (P : ℝ) : simple_interest P = I P -> P = 2000 :=
by
  sorry

end NUMINAMATH_GPT_find_principal_l2144_214461


namespace NUMINAMATH_GPT_f_x_plus_1_l2144_214474

-- Given function definition
def f (x : ℝ) := x^2

-- Statement to prove
theorem f_x_plus_1 (x : ℝ) : f (x + 1) = x^2 + 2 * x + 1 := 
by
  rw [f]
  -- This simplifies to:
  -- (x + 1)^2 = x^2 + 2 * x + 1
  sorry

end NUMINAMATH_GPT_f_x_plus_1_l2144_214474


namespace NUMINAMATH_GPT_balls_sum_l2144_214427

theorem balls_sum (m n : ℕ) (h₁ : ∀ a, a ∈ ({m, 8, n} : Finset ℕ)) -- condition: balls are identical except for color
  (h₂ : (8 : ℝ) / (m + 8 + n) = (m + n : ℝ) / (m + 8 + n)) : m + n = 8 :=
sorry

end NUMINAMATH_GPT_balls_sum_l2144_214427


namespace NUMINAMATH_GPT_condition_sufficient_but_not_necessary_condition_not_necessary_combined_condition_l2144_214442

theorem condition_sufficient_but_not_necessary (x y : ℝ) :
  (x^2 + y^2 + 4*x + 3 ≤ 0) → ((x + 4) * (x + 3) ≥ 0) :=
sorry

theorem condition_not_necessary (x y : ℝ) :
  ((x + 4) * (x + 3) ≥ 0) → ¬ (x^2 + y^2 + 4*x + 3 ≤ 0) :=
sorry

-- Combine both into a single statement using conjunction
theorem combined_condition (x y : ℝ) :
  ((x^2 + y^2 + 4*x + 3 ≤ 0) → ((x + 4) * (x + 3) ≥ 0))
  ∧ ((x + 4) * (x + 3) ≥ 0 → ¬(x^2 + y^2 + 4*x + 3 ≤ 0)) :=
sorry

end NUMINAMATH_GPT_condition_sufficient_but_not_necessary_condition_not_necessary_combined_condition_l2144_214442


namespace NUMINAMATH_GPT_martha_flower_cost_l2144_214480

theorem martha_flower_cost :
  let roses_per_centerpiece := 8
  let orchids_per_centerpiece := 2 * roses_per_centerpiece
  let lilies_per_centerpiece := 6
  let centerpieces := 6
  let cost_per_flower := 15
  let total_roses := roses_per_centerpiece * centerpieces
  let total_orchids := orchids_per_centerpiece * centerpieces
  let total_lilies := lilies_per_centerpiece * centerpieces
  let total_flowers := total_roses + total_orchids + total_lilies
  let total_cost := total_flowers * cost_per_flower
  total_cost = 2700 :=
by
  let roses_per_centerpiece := 8
  let orchids_per_centerpiece := 2 * roses_per_centerpiece
  let lilies_per_centerpiece := 6
  let centerpieces := 6
  let cost_per_flower := 15
  let total_roses := roses_per_centerpiece * centerpieces
  let total_orchids := orchids_per_centerpiece * centerpieces
  let total_lilies := lilies_per_centerpiece * centerpieces
  let total_flowers := total_roses + total_orchids + total_lilies
  let total_cost := total_flowers * cost_per_flower
  -- Proof to be added here
  sorry

end NUMINAMATH_GPT_martha_flower_cost_l2144_214480


namespace NUMINAMATH_GPT_value_of_fraction_l2144_214496

theorem value_of_fraction : (121^2 - 112^2) / 9 = 233 := by
  -- use the difference of squares property
  sorry

end NUMINAMATH_GPT_value_of_fraction_l2144_214496


namespace NUMINAMATH_GPT_train_speed_l2144_214483

theorem train_speed (length_train length_bridge time : ℝ) (h_train : length_train = 125) (h_bridge : length_bridge = 250) (h_time : time = 30) :
    (length_train + length_bridge) / time * 3.6 = 45 := by
  sorry

end NUMINAMATH_GPT_train_speed_l2144_214483


namespace NUMINAMATH_GPT_factorization_correct_l2144_214407

theorem factorization_correct (a x y : ℝ) : a * x - a * y = a * (x - y) := by sorry

end NUMINAMATH_GPT_factorization_correct_l2144_214407


namespace NUMINAMATH_GPT_base4_arithmetic_l2144_214467

theorem base4_arithmetic :
  (Nat.ofDigits 4 [2, 3, 1] * Nat.ofDigits 4 [2, 2] / Nat.ofDigits 4 [3]) = Nat.ofDigits 4 [2, 2, 1] := by
sorry

end NUMINAMATH_GPT_base4_arithmetic_l2144_214467


namespace NUMINAMATH_GPT_prove_P_plus_V_eq_zero_l2144_214493

variable (P Q R S T U V : ℤ)

-- Conditions in Lean
def sequence_conditions (P Q R S T U V : ℤ) :=
  S = 7 ∧
  P + Q + R = 27 ∧
  Q + R + S = 27 ∧
  R + S + T = 27 ∧
  S + T + U = 27 ∧
  T + U + V = 27 ∧
  U + V + P = 27

-- Assertion that needs to be proved
theorem prove_P_plus_V_eq_zero (P Q R S T U V : ℤ) (h : sequence_conditions P Q R S T U V) : 
  P + V = 0 := by
  sorry

end NUMINAMATH_GPT_prove_P_plus_V_eq_zero_l2144_214493


namespace NUMINAMATH_GPT_probability_of_shaded_shape_l2144_214458

   def total_shapes : ℕ := 4
   def shaded_shapes : ℕ := 1

   theorem probability_of_shaded_shape : shaded_shapes / total_shapes = 1 / 4 := 
   by
     sorry
   
end NUMINAMATH_GPT_probability_of_shaded_shape_l2144_214458


namespace NUMINAMATH_GPT_arnolds_total_protein_l2144_214445

theorem arnolds_total_protein (collagen_protein_per_two_scoops : ℕ) (protein_per_scoop : ℕ) 
    (steak_protein : ℕ) (scoops_of_collagen : ℕ) (scoops_of_protein : ℕ) :
    collagen_protein_per_two_scoops = 18 →
    protein_per_scoop = 21 →
    steak_protein = 56 →
    scoops_of_collagen = 1 →
    scoops_of_protein = 1 →
    (collagen_protein_per_two_scoops / 2 * scoops_of_collagen + protein_per_scoop * scoops_of_protein + steak_protein = 86) :=
by
  intros hc p s sc sp
  sorry

end NUMINAMATH_GPT_arnolds_total_protein_l2144_214445


namespace NUMINAMATH_GPT_propositions_using_logical_connectives_l2144_214448

-- Define each of the propositions.
def prop1 := "October 1, 2004, is the National Day and also the Mid-Autumn Festival."
def prop2 := "Multiples of 10 are definitely multiples of 5."
def prop3 := "A trapezoid is not a rectangle."
def prop4 := "The solutions to the equation x^2 = 1 are x = ± 1."

-- Define logical connectives usage.
def uses_and (s : String) : Prop := 
  s = "October 1, 2004, is the National Day and also the Mid-Autumn Festival."
def uses_not (s : String) : Prop := 
  s = "A trapezoid is not a rectangle."
def uses_or (s : String) : Prop := 
  s = "The solutions to the equation x^2 = 1 are x = ± 1."

-- The lean theorem stating the propositions that use logical connectives
theorem propositions_using_logical_connectives :
  (uses_and prop1) ∧ (¬ uses_and prop2) ∧ (uses_not prop3) ∧ (uses_or prop4) := 
by
  sorry

end NUMINAMATH_GPT_propositions_using_logical_connectives_l2144_214448


namespace NUMINAMATH_GPT_national_park_sightings_l2144_214492

def january_sightings : ℕ := 26

def february_sightings : ℕ := 3 * january_sightings

def march_sightings : ℕ := february_sightings / 2

def total_sightings : ℕ := january_sightings + february_sightings + march_sightings

theorem national_park_sightings : total_sightings = 143 := by
  sorry

end NUMINAMATH_GPT_national_park_sightings_l2144_214492


namespace NUMINAMATH_GPT_minimum_value_of_f_l2144_214466

noncomputable def f (x : ℝ) : ℝ := 4 * x + 9 / x

theorem minimum_value_of_f : 
  (∀ (x : ℝ), x > 0 → f x ≥ 12) ∧ (∃ (x : ℝ), x > 0 ∧ f x = 12) :=
by {
  sorry
}

end NUMINAMATH_GPT_minimum_value_of_f_l2144_214466


namespace NUMINAMATH_GPT_trevor_eggs_left_l2144_214418

def gertrude_eggs : Nat := 4
def blanche_eggs : Nat := 3
def nancy_eggs : Nat := 2
def martha_eggs : Nat := 2
def dropped_eggs : Nat := 2

theorem trevor_eggs_left : 
  (gertrude_eggs + blanche_eggs + nancy_eggs + martha_eggs - dropped_eggs) = 9 := 
  by sorry

end NUMINAMATH_GPT_trevor_eggs_left_l2144_214418


namespace NUMINAMATH_GPT_shares_difference_l2144_214438

theorem shares_difference (x : ℝ) (h_ratio : 2.5 * x + 3.5 * x + 7.5 * x + 9.8 * x = (23.3 * x))
  (h_difference : 7.5 * x - 3.5 * x = 4500) : 9.8 * x - 2.5 * x = 8212.5 :=
by
  sorry

end NUMINAMATH_GPT_shares_difference_l2144_214438


namespace NUMINAMATH_GPT_find_x_values_l2144_214429

theorem find_x_values (x : ℝ) :
  (2 / (x + 2) + 8 / (x + 4) ≥ 2) ↔ (x ∈ Set.Ici 2 ∨ x ∈ Set.Iic (-4)) := by
sorry

end NUMINAMATH_GPT_find_x_values_l2144_214429


namespace NUMINAMATH_GPT_largest_multiple_of_7_neg_greater_than_neg_150_l2144_214432

theorem largest_multiple_of_7_neg_greater_than_neg_150 : 
  ∃ (k : ℤ), k % 7 = 0 ∧ -k > -150 ∧ (∀ (m : ℤ), m % 7 = 0 ∧ -m > -150 → k ≥ m) ∧ k = 147 :=
by
  sorry

end NUMINAMATH_GPT_largest_multiple_of_7_neg_greater_than_neg_150_l2144_214432


namespace NUMINAMATH_GPT_pow_mod_eq_l2144_214400

theorem pow_mod_eq : (17 ^ 2001) % 23 = 11 := 
by {
  sorry
}

end NUMINAMATH_GPT_pow_mod_eq_l2144_214400


namespace NUMINAMATH_GPT_stock_price_end_of_third_year_l2144_214444

def first_year_price (initial_price : ℝ) (first_year_increase : ℝ) : ℝ :=
  initial_price + (initial_price * first_year_increase)

def second_year_price (price_end_first : ℝ) (second_year_decrease : ℝ) : ℝ :=
  price_end_first - (price_end_first * second_year_decrease)

def third_year_price (price_end_second : ℝ) (third_year_increase : ℝ) : ℝ :=
  price_end_second + (price_end_second * third_year_increase)

theorem stock_price_end_of_third_year :
  ∀ (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) (third_year_increase : ℝ),
    initial_price = 150 →
    first_year_increase = 0.5 →
    second_year_decrease = 0.3 →
    third_year_increase = 0.2 →
    third_year_price (second_year_price (first_year_price initial_price first_year_increase) second_year_decrease) third_year_increase = 189 :=
by
  intros initial_price first_year_increase second_year_decrease third_year_increase
  sorry

end NUMINAMATH_GPT_stock_price_end_of_third_year_l2144_214444


namespace NUMINAMATH_GPT_simplify_and_evaluate_l2144_214462

theorem simplify_and_evaluate :
  let x := 2 * Real.sqrt 3
  (x - Real.sqrt 2) * (x + Real.sqrt 2) + x * (x - 1) = 22 - 2 * Real.sqrt 3 := 
by
  let x := 2 * Real.sqrt 3
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l2144_214462


namespace NUMINAMATH_GPT_students_chose_water_l2144_214446

theorem students_chose_water (total_students : ℕ)
  (h1 : 75 * total_students / 100 = 90)
  (h2 : 25 * total_students / 100 = x) :
  x = 30 := 
sorry

end NUMINAMATH_GPT_students_chose_water_l2144_214446


namespace NUMINAMATH_GPT_wire_divided_into_quarters_l2144_214417

theorem wire_divided_into_quarters
  (l : ℕ) -- length of the wire
  (parts : ℕ) -- number of parts the wire is divided into
  (h_l : l = 28) -- wire is 28 cm long
  (h_parts : parts = 4) -- wire is divided into 4 parts
  : l / parts = 7 := -- each part is 7 cm long
by
  -- use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_wire_divided_into_quarters_l2144_214417


namespace NUMINAMATH_GPT_cos_angle_equiv_370_l2144_214486

open Real

noncomputable def find_correct_n : ℕ :=
  sorry

theorem cos_angle_equiv_370 (n : ℕ) (h : 0 ≤ n ∧ n ≤ 180) : cos (n * π / 180) = cos (370 * π / 180) → n = 10 :=
by
  sorry

end NUMINAMATH_GPT_cos_angle_equiv_370_l2144_214486


namespace NUMINAMATH_GPT_pyramid_volume_l2144_214428

theorem pyramid_volume (S A : ℝ)
  (h_surface : 3 * S = 432)
  (h_half_triangular : A = 0.5 * S) :
  (1 / 3) * S * (12 * Real.sqrt 3) = 288 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_volume_l2144_214428


namespace NUMINAMATH_GPT_a_b_condition_l2144_214479

theorem a_b_condition (a b : ℂ) (h : (a + b) / a = b / (a + b)) :
  (∃ x y : ℂ, x = a ∧ y = b ∧ ((¬ x.im = 0 ∧ y.im = 0) ∨ (x.im = 0 ∧ ¬ y.im = 0) ∨ (¬ x.im = 0 ∧ ¬ y.im = 0))) :=
by
  sorry

end NUMINAMATH_GPT_a_b_condition_l2144_214479


namespace NUMINAMATH_GPT_subtraction_and_multiplication_problem_l2144_214440

theorem subtraction_and_multiplication_problem :
  (5 / 6 - 1 / 3) * 3 / 4 = 3 / 8 :=
by sorry

end NUMINAMATH_GPT_subtraction_and_multiplication_problem_l2144_214440


namespace NUMINAMATH_GPT_total_dogs_l2144_214487

variable (U : Type) [Fintype U]
variable (jump fetch shake : U → Prop)
variable [DecidablePred jump] [DecidablePred fetch] [DecidablePred shake]

theorem total_dogs (h_jump : Fintype.card {u | jump u} = 70)
  (h_jump_and_fetch : Fintype.card {u | jump u ∧ fetch u} = 30)
  (h_fetch : Fintype.card {u | fetch u} = 40)
  (h_fetch_and_shake : Fintype.card {u | fetch u ∧ shake u} = 20)
  (h_shake : Fintype.card {u | shake u} = 50)
  (h_jump_and_shake : Fintype.card {u | jump u ∧ shake u} = 25)
  (h_all_three : Fintype.card {u | jump u ∧ fetch u ∧ shake u} = 15)
  (h_none : Fintype.card {u | ¬jump u ∧ ¬fetch u ∧ ¬shake u} = 15) :
  Fintype.card U = 115 :=
by
  sorry

end NUMINAMATH_GPT_total_dogs_l2144_214487


namespace NUMINAMATH_GPT_compare_cubic_terms_l2144_214410

theorem compare_cubic_terms (a b : ℝ) :
    (a ≥ b → a^3 - b^3 ≥ a * b^2 - a^2 * b) ∧
    (a < b → a^3 - b^3 ≤ a * b^2 - a^2 * b) :=
by sorry

end NUMINAMATH_GPT_compare_cubic_terms_l2144_214410


namespace NUMINAMATH_GPT_ratio_of_Lev_to_Akeno_l2144_214409

theorem ratio_of_Lev_to_Akeno (L : ℤ) (A : ℤ) (Ambrocio : ℤ) :
  A = 2985 ∧ Ambrocio = L - 177 ∧ A = L + Ambrocio + 1172 → L / A = 1 / 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_ratio_of_Lev_to_Akeno_l2144_214409


namespace NUMINAMATH_GPT_find_X_l2144_214460

theorem find_X (X : ℕ) (h1 : 2 + 1 + 3 + X = 3 + 4 + 5) : X = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_X_l2144_214460


namespace NUMINAMATH_GPT_combined_average_age_l2144_214430

-- Definitions based on given conditions
def num_fifth_graders : ℕ := 28
def avg_age_fifth_graders : ℝ := 10
def num_parents : ℕ := 45
def avg_age_parents : ℝ := 40

-- The statement to prove
theorem combined_average_age : (num_fifth_graders * avg_age_fifth_graders + num_parents * avg_age_parents) / (num_fifth_graders + num_parents) = 28.49 :=
  by
  sorry

end NUMINAMATH_GPT_combined_average_age_l2144_214430


namespace NUMINAMATH_GPT_total_distance_l2144_214456

theorem total_distance (D : ℝ) 
  (h1 : 1/4 * (3/8 * D) = 210) : D = 840 := 
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_total_distance_l2144_214456


namespace NUMINAMATH_GPT_lionsAfterOneYear_l2144_214495

-- Definitions based on problem conditions
def initialLions : Nat := 100
def birthRate : Nat := 5
def deathRate : Nat := 1
def monthsInYear : Nat := 12

-- Theorem statement
theorem lionsAfterOneYear :
  initialLions + birthRate * monthsInYear - deathRate * monthsInYear = 148 :=
by
  sorry

end NUMINAMATH_GPT_lionsAfterOneYear_l2144_214495


namespace NUMINAMATH_GPT_initial_candies_equal_twenty_l2144_214411

-- Definitions based on conditions
def friends : ℕ := 6
def candies_per_friend : ℕ := 4
def total_needed_candies : ℕ := friends * candies_per_friend
def additional_candies : ℕ := 4

-- Main statement
theorem initial_candies_equal_twenty :
  (total_needed_candies - additional_candies) = 20 := by
  sorry

end NUMINAMATH_GPT_initial_candies_equal_twenty_l2144_214411


namespace NUMINAMATH_GPT_calculate_blue_candles_l2144_214426

-- Definitions based on identified conditions
def total_candles : Nat := 79
def yellow_candles : Nat := 27
def red_candles : Nat := 14
def blue_candles : Nat := total_candles - (yellow_candles + red_candles)

-- The proof statement
theorem calculate_blue_candles : blue_candles = 38 :=
by
  sorry

end NUMINAMATH_GPT_calculate_blue_candles_l2144_214426


namespace NUMINAMATH_GPT_total_charts_16_l2144_214463

def total_charts_brought (number_of_associate_professors : Int) (number_of_assistant_professors : Int) : Int :=
  number_of_associate_professors * 1 + number_of_assistant_professors * 2

theorem total_charts_16 (A B : Int)
  (h1 : 2 * A + B = 11)
  (h2 : A + B = 9) :
  total_charts_brought A B = 16 :=
by {
  -- the proof will go here
  sorry
}

end NUMINAMATH_GPT_total_charts_16_l2144_214463


namespace NUMINAMATH_GPT_sarah_initial_bake_l2144_214477

theorem sarah_initial_bake (todd_ate : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) 
  (initial_cupcakes : ℕ)
  (h1 : todd_ate = 14)
  (h2 : packages = 3)
  (h3 : cupcakes_per_package = 8)
  (h4 : packages * cupcakes_per_package + todd_ate = initial_cupcakes) :
  initial_cupcakes = 38 :=
by sorry

end NUMINAMATH_GPT_sarah_initial_bake_l2144_214477


namespace NUMINAMATH_GPT_largest_2_digit_number_l2144_214457

theorem largest_2_digit_number:
  ∃ (N: ℕ), N >= 10 ∧ N < 100 ∧ N % 4 = 0 ∧ (∀ k: ℕ, k ≥ 1 → (N^k) % 100 = N % 100) ∧ 
  (∀ M: ℕ, M >= 10 → M < 100 → M % 4 = 0 → (∀ k: ℕ, k ≥ 1 → (M^k) % 100 = M % 100) → N ≥ M) :=
sorry

end NUMINAMATH_GPT_largest_2_digit_number_l2144_214457


namespace NUMINAMATH_GPT_smallest_n_l2144_214422

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, b * k = a

def meets_condition (n : ℕ) : Prop :=
  n > 0 ∧
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ n + 1 ∧ is_divisible (n^2 - n + 1) k ∧
  ∃ l : ℕ, 1 ≤ l ∧ l ≤ n + 1 ∧ ¬ is_divisible (n^2 - n + 1) l

theorem smallest_n : ∃ n : ℕ, meets_condition n ∧ n = 5 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_l2144_214422


namespace NUMINAMATH_GPT_vector_dot_product_l2144_214436

theorem vector_dot_product
  (AB : ℝ × ℝ) (BC : ℝ × ℝ)
  (t : ℝ)
  (hAB : AB = (2, 3))
  (hBC : BC = (3, t))
  (ht : t > 0)
  (hmagnitude : (3^2 + t^2).sqrt = (10:ℝ).sqrt) :
  (AB.1 * (AB.1 + BC.1) + AB.2 * (AB.2 + BC.2) = 22) :=
by
  sorry

end NUMINAMATH_GPT_vector_dot_product_l2144_214436


namespace NUMINAMATH_GPT_find_minuend_l2144_214405

variable (x y : ℕ)

-- Conditions
axiom h1 : x - y = 8008
axiom h2 : x - 10 * y = 88

-- Theorem statement
theorem find_minuend : x = 8888 :=
by
  sorry

end NUMINAMATH_GPT_find_minuend_l2144_214405


namespace NUMINAMATH_GPT_determine_some_number_l2144_214464

theorem determine_some_number (x : ℝ) (n : ℝ) (hx : x = 1.5) (h : (3 + 2 * x)^5 = (1 + n * x)^4) : n = 10 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_determine_some_number_l2144_214464


namespace NUMINAMATH_GPT_find_a1_of_geom_series_l2144_214420

noncomputable def geom_series_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

theorem find_a1_of_geom_series (a₁ q : ℝ) (S : ℕ → ℝ)
  (h1 : S 6 = 9 * S 3)
  (h2 : S 5 = 62)
  (neq1 : q ≠ 1)
  (neqm1 : q ≠ -1) :
  a₁ = 2 :=
by
  have eq1 : S 6 = geom_series_sum a₁ q 6 := sorry
  have eq2 : S 3 = geom_series_sum a₁ q 3 := sorry
  have eq3 : S 5 = geom_series_sum a₁ q 5 := sorry
  sorry

end NUMINAMATH_GPT_find_a1_of_geom_series_l2144_214420


namespace NUMINAMATH_GPT_min_value_geometric_sequence_l2144_214475

-- Definition for conditions and problem setup
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n+1) = r * a n

-- Given data
variable (a : ℕ → ℝ)
variable (h_geom : is_geometric_sequence a)
variable (h_sum : a 2015 + a 2017 = Real.pi)

-- Goal statement
theorem min_value_geometric_sequence :
  ∃ (min_val : ℝ), min_val = (Real.pi^2) / 2 ∧ (
    ∀ a : ℕ → ℝ, 
    is_geometric_sequence a → 
    a 2015 + a 2017 = Real.pi → 
    a 2016 * (a 2014 + a 2018) ≥ (Real.pi^2) / 2
  ) :=
sorry

end NUMINAMATH_GPT_min_value_geometric_sequence_l2144_214475


namespace NUMINAMATH_GPT_cost_of_used_cd_l2144_214476

theorem cost_of_used_cd (N U : ℝ) 
    (h1 : 6 * N + 2 * U = 127.92) 
    (h2 : 3 * N + 8 * U = 133.89) :
    U = 9.99 :=
by 
  sorry

end NUMINAMATH_GPT_cost_of_used_cd_l2144_214476


namespace NUMINAMATH_GPT_abs_diff_of_slopes_l2144_214451

theorem abs_diff_of_slopes (k1 k2 b : ℝ) (h : k1 * k2 < 0) (area_cond : (1 / 2) * 3 * |k1 - k2| * 3 = 9) :
  |k1 - k2| = 2 :=
by
  sorry

end NUMINAMATH_GPT_abs_diff_of_slopes_l2144_214451


namespace NUMINAMATH_GPT_exist_pos_integers_m_n_l2144_214402

def d (n : ℕ) : ℕ :=
  -- Number of divisors of n
  sorry 

theorem exist_pos_integers_m_n :
  ∃ (m n : ℕ), (m > 0) ∧ (n > 0) ∧ (m = 24) ∧ 
  ((∃ (triples : Finset (ℕ × ℕ × ℕ)),
    (∀ (a b c : ℕ), (a, b, c) ∈ triples ↔ (0 < a) ∧ (a < b) ∧ (b < c) ∧ (c ≤ m) ∧ (d (n + a) * d (n + b) * d (n + c)) % (a * b * c) = 0) ∧ 
    (triples.card = 2024))) :=
sorry

end NUMINAMATH_GPT_exist_pos_integers_m_n_l2144_214402


namespace NUMINAMATH_GPT_craig_apples_after_sharing_l2144_214468

-- Defining the initial conditions
def initial_apples_craig : ℕ := 20
def shared_apples : ℕ := 7

-- The proof statement
theorem craig_apples_after_sharing : 
  initial_apples_craig - shared_apples = 13 := 
by
  sorry

end NUMINAMATH_GPT_craig_apples_after_sharing_l2144_214468


namespace NUMINAMATH_GPT_volume_of_solid_l2144_214453

def x_y_relation (x y : ℝ) : Prop := x = (y - 2)^(1/3)
def x1 (x : ℝ) : Prop := x = 1
def y1 (y : ℝ) : Prop := y = 1

theorem volume_of_solid :
  ∀ (x y : ℝ),
    (x_y_relation x y ∧ x1 x ∧ y1 y) →
    ∃ V : ℝ, V = (44 / 7) * Real.pi :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_volume_of_solid_l2144_214453


namespace NUMINAMATH_GPT_track_length_is_450_l2144_214443

theorem track_length_is_450 (x : ℝ) (d₁ : ℝ) (d₂ : ℝ)
  (h₁ : d₁ = 150)
  (h₂ : x - d₁ = 120)
  (h₃ : d₂ = 200)
  (h₄ : ∀ (d₁ d₂ : ℝ) (t₁ t₂ : ℝ), t₁ / t₂ = d₁ / d₂)
  : x = 450 := by
  sorry

end NUMINAMATH_GPT_track_length_is_450_l2144_214443


namespace NUMINAMATH_GPT_proposition_b_proposition_d_l2144_214431

-- Proposition B: For a > 0 and b > 0, if ab = 2, then the minimum value of a + 2b is 4
theorem proposition_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 2) : a + 2 * b ≥ 4 :=
  sorry

-- Proposition D: For a > 0 and b > 0, if a² + b² = 1, then the maximum value of a + b is sqrt(2).
theorem proposition_d (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + b^2 = 1) : a + b ≤ Real.sqrt 2 :=
  sorry

end NUMINAMATH_GPT_proposition_b_proposition_d_l2144_214431


namespace NUMINAMATH_GPT_roots_bounds_if_and_only_if_conditions_l2144_214481

theorem roots_bounds_if_and_only_if_conditions (a b c : ℝ) (h : a > 0) (x1 x2 : ℝ) (hr : ∀ {x : ℝ}, a * x^2 + b * x + c = 0 → x = x1 ∨ x = x2) :
  (|x1| ≤ 1 ∧ |x2| ≤ 1) ↔ (a + b + c ≥ 0 ∧ a - b + c ≥ 0 ∧ a - c ≥ 0) :=
sorry

end NUMINAMATH_GPT_roots_bounds_if_and_only_if_conditions_l2144_214481


namespace NUMINAMATH_GPT_sum_of_final_two_numbers_l2144_214494

theorem sum_of_final_two_numbers (x y T : ℕ) (h : x + y = T) :
  3 * (x + 5) + 3 * (y + 5) = 3 * T + 30 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_final_two_numbers_l2144_214494


namespace NUMINAMATH_GPT_triangle_largest_angle_and_type_l2144_214478

theorem triangle_largest_angle_and_type
  (a b c : ℝ) 
  (h1 : a + b + c = 180)
  (h2 : a = 4 * k) 
  (h3 : b = 3 * k) 
  (h4 : c = 2 * k) 
  (h5 : a ≥ b) 
  (h6 : a ≥ c) : 
  a = 80 ∧ a < 90 ∧ b < 90 ∧ c < 90 := 
by
  -- Replace 'by' with 'sorry' to denote that the proof should go here
  sorry

end NUMINAMATH_GPT_triangle_largest_angle_and_type_l2144_214478


namespace NUMINAMATH_GPT_product_of_two_numbers_l2144_214447

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 21) (h2 : x^2 + y^2 = 527) : x * y = -43 :=
sorry

end NUMINAMATH_GPT_product_of_two_numbers_l2144_214447


namespace NUMINAMATH_GPT_find_ratio_eq_eighty_six_l2144_214412

-- Define the set S
def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 45}

-- Define the sum of the first n natural numbers function
def sum_n_nat (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define our specific scenario setup
def selected_numbers (x y : ℕ) : Prop :=
  x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ x * y = sum_n_nat 45 - (x + y)

-- Prove the resulting ratio condition
theorem find_ratio_eq_eighty_six (x y : ℕ) (h : selected_numbers x y) : 
  x < y → y / x = 86 :=
by
  sorry

end NUMINAMATH_GPT_find_ratio_eq_eighty_six_l2144_214412


namespace NUMINAMATH_GPT_find_m_plus_n_l2144_214485

variable (U : Set ℝ) (A : Set ℝ) (CUA : Set ℝ) (m n : ℝ)
  -- Condition 1: The universal set U is the set of all real numbers
  (hU : U = Set.univ)
  -- Condition 2: A is defined as the set of all x such that (x - 1)(x - m) > 0
  (hA : A = { x : ℝ | (x - 1) * (x - m) > 0 })
  -- Condition 3: The complement of A in U is [-1, -n]
  (hCUA : CUA = { x : ℝ | x ∈ U ∧ x ∉ A } ∧ CUA = Icc (-1) (-n))

theorem find_m_plus_n : m + n = -2 :=
  sorry 

end NUMINAMATH_GPT_find_m_plus_n_l2144_214485


namespace NUMINAMATH_GPT_smallest_z_value_l2144_214416

theorem smallest_z_value :
  ∃ (x z : ℕ), (w = x - 2) ∧ (y = x + 2) ∧ (z = x + 4) ∧ ((x - 2)^3 + x^3 + (x + 2)^3 = (x + 4)^3) ∧ z = 2 := by
  sorry

end NUMINAMATH_GPT_smallest_z_value_l2144_214416


namespace NUMINAMATH_GPT_fraction_eval_l2144_214406

theorem fraction_eval :
  (8 : ℝ) / (4 * 25) = (0.8 : ℝ) / (0.4 * 25) :=
sorry

end NUMINAMATH_GPT_fraction_eval_l2144_214406


namespace NUMINAMATH_GPT_equation_C_is_symmetric_l2144_214424

def symm_y_axis (f : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), f x y ↔ f (-x) y

def equation_A (x y : ℝ) : Prop := x^2 - x + y^2 = 1
def equation_B (x y : ℝ) : Prop := x^2 * y + x * y^2 = 1
def equation_C (x y : ℝ) : Prop := x^2 - y^2 = 1
def equation_D (x y : ℝ) : Prop := x - y = 1

theorem equation_C_is_symmetric : symm_y_axis equation_C :=
by
  sorry

end NUMINAMATH_GPT_equation_C_is_symmetric_l2144_214424


namespace NUMINAMATH_GPT_Cary_walked_miles_round_trip_l2144_214425

theorem Cary_walked_miles_round_trip : ∀ (m : ℕ), 
  150 * m - 200 = 250 → m = 3 := 
by
  intros m h
  sorry

end NUMINAMATH_GPT_Cary_walked_miles_round_trip_l2144_214425


namespace NUMINAMATH_GPT_rectangle_area_l2144_214413

theorem rectangle_area (x y : ℝ) (hx : 3 * y = 7 * x) (hp : 2 * (x + y) = 40) :
  x * y = 84 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l2144_214413


namespace NUMINAMATH_GPT_intersection_points_count_l2144_214434

theorem intersection_points_count
  : (∀ n : ℤ, ∃ (x y : ℝ), (x - ⌊x⌋) ^ 2 + y ^ 2 = 2 * (x - ⌊x⌋) ∨ y = 1 / 3 * x) →
    (∃ count : ℕ, count = 12) :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_count_l2144_214434


namespace NUMINAMATH_GPT_total_waiting_time_difference_l2144_214408

theorem total_waiting_time_difference :
  let n_swings := 6
  let n_slide := 4 * n_swings
  let t_swings := 3.5 * 60
  let t_slide := 45
  let T_swings := n_swings * t_swings
  let T_slide := n_slide * t_slide
  let T_difference := T_swings - T_slide
  T_difference = 180 :=
by
  sorry

end NUMINAMATH_GPT_total_waiting_time_difference_l2144_214408


namespace NUMINAMATH_GPT_quadratic_roots_range_no_real_k_for_reciprocal_l2144_214437

theorem quadratic_roots_range (k : ℝ) (h : 12 * k + 4 > 0) : k > -1 / 3 ∧ k ≠ 0 :=
by
  sorry

theorem no_real_k_for_reciprocal (k : ℝ) : ¬∃ (x1 x2 : ℝ), (kx^2 - 2*(k+1)*x + k-1 = 0) ∧ (1/x1 + 1/x2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_range_no_real_k_for_reciprocal_l2144_214437


namespace NUMINAMATH_GPT_discount_equation_l2144_214449

variable (P₀ P_f x : ℝ)
variable (h₀ : P₀ = 200)
variable (h₁ : P_f = 164)

theorem discount_equation :
  P₀ * (1 - x)^2 = P_f := by
  sorry

end NUMINAMATH_GPT_discount_equation_l2144_214449


namespace NUMINAMATH_GPT_line_equation_l2144_214470

-- Define the point A(2, 1)
def A : ℝ × ℝ := (2, 1)

-- Define the notion of a line with equal intercepts on the coordinates
def line_has_equal_intercepts (c : ℝ) : Prop :=
  ∃ (m b : ℝ), ∀ (x y : ℝ), y = m * x + b ↔ x = y ∧ y = c

-- Define the condition that the line passes through point A
def line_passes_through_A (m b : ℝ) : Prop :=
  A.2 = m * A.1 + b

-- Define the two possible equations for the line
def line_eq1 (x y : ℝ) : Prop :=
  x + y - 3 = 0

def line_eq2 (x y : ℝ) : Prop :=
  2 * x - y = 0

-- Combined conditions in a single theorem
theorem line_equation (m b c x y : ℝ) (h_pass : line_passes_through_A m b) (h_int : line_has_equal_intercepts c) :
  (line_eq1 x y ∨ line_eq2 x y) :=
sorry

end NUMINAMATH_GPT_line_equation_l2144_214470


namespace NUMINAMATH_GPT_sum_of_roots_combined_eq_five_l2144_214401

noncomputable def sum_of_roots_poly1 : ℝ :=
-(-9/3)

noncomputable def sum_of_roots_poly2 : ℝ :=
-(-8/4)

theorem sum_of_roots_combined_eq_five :
  sum_of_roots_poly1 + sum_of_roots_poly2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_combined_eq_five_l2144_214401


namespace NUMINAMATH_GPT_josh_and_fred_age_l2144_214491

theorem josh_and_fred_age
    (a b k : ℕ)
    (h1 : 10 * a + b > 10 * b + a)
    (h2 : 99 * (a^2 - b^2) = k^2)
    (ha : a ≥ 0 ∧ a ≤ 9)
    (hb : b ≥ 0 ∧ b ≤ 9) : 
    10 * a + b = 65 ∧ 
    10 * b + a = 56 := 
sorry

end NUMINAMATH_GPT_josh_and_fred_age_l2144_214491


namespace NUMINAMATH_GPT_quadratic_distinct_zeros_l2144_214499

theorem quadratic_distinct_zeros (m : ℝ) : 
  (x^2 + m * x + (m + 3)) = 0 → 
  (0 < m^2 - 4 * (m + 3)) ↔ (m < -2) ∨ (m > 6) :=
sorry

end NUMINAMATH_GPT_quadratic_distinct_zeros_l2144_214499


namespace NUMINAMATH_GPT_dan_initial_money_l2144_214454

def money_left : ℕ := 3
def cost_candy : ℕ := 2
def initial_money : ℕ := money_left + cost_candy

theorem dan_initial_money :
  initial_money = 5 :=
by
  -- Definitions according to problem
  let money_left := 3
  let cost_candy := 2

  have h : initial_money = money_left + cost_candy := by rfl
  rw [h]

  -- Show the final equivalence
  show 3 + 2 = 5
  rfl

end NUMINAMATH_GPT_dan_initial_money_l2144_214454


namespace NUMINAMATH_GPT_kylie_earrings_l2144_214452

def number_of_necklaces_monday := 10
def number_of_necklaces_tuesday := 2
def number_of_bracelets_wednesday := 5
def beads_per_necklace := 20
def beads_per_bracelet := 10
def beads_per_earring := 5
def total_beads := 325

theorem kylie_earrings : 
    (total_beads - ((number_of_necklaces_monday + number_of_necklaces_tuesday) * beads_per_necklace + number_of_bracelets_wednesday * beads_per_bracelet)) / beads_per_earring = 7 :=
by
    sorry

end NUMINAMATH_GPT_kylie_earrings_l2144_214452


namespace NUMINAMATH_GPT_maximum_cells_covered_at_least_five_times_l2144_214433

theorem maximum_cells_covered_at_least_five_times :
  let areas := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let total_covered := List.sum areas
  let exact_coverage := 1 * 1 + 2 * 2 + 3 * 3 + 4 * 4
  let remaining_coverage := total_covered - exact_coverage
  let max_cells_covered_at_least_five := remaining_coverage / 5
  max_cells_covered_at_least_five = 5 :=
by
  sorry

end NUMINAMATH_GPT_maximum_cells_covered_at_least_five_times_l2144_214433


namespace NUMINAMATH_GPT_complete_square_proof_l2144_214421

def quadratic_eq := ∀ (x : ℝ), x^2 - 6 * x + 5 = 0
def form_completing_square (b c : ℝ) := ∀ (x : ℝ), (x + b)^2 = c

theorem complete_square_proof :
  quadratic_eq → (∃ b c : ℤ, form_completing_square (b : ℝ) (c : ℝ) ∧ b + c = 11) :=
by
  sorry

end NUMINAMATH_GPT_complete_square_proof_l2144_214421


namespace NUMINAMATH_GPT_question_l2144_214490

variable (a : ℝ)

def condition_p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2 * x ≤ a^2 - a - 3

def condition_q (a : ℝ) : Prop := ∀ (x y : ℝ) , x > y → (5 - 2 * a)^x < (5 - 2 * a)^y

theorem question (h1 : condition_p a ∨ condition_q a)
                (h2 : ¬ (condition_p a ∧ condition_q a)) : a = 2 ∨ a ≥ 5 / 2 :=
sorry

end NUMINAMATH_GPT_question_l2144_214490


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2144_214484

-- Definitions from conditions
def arithmetic_sequence (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℤ)
  (h_seq : arithmetic_sequence a)
  (h_a5 : a 5 = 3)
  (h_a6 : a 6 = -2) :
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 3 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2144_214484


namespace NUMINAMATH_GPT_calc_x6_plus_inv_x6_l2144_214482

theorem calc_x6_plus_inv_x6 (x : ℝ) (hx : x + (1 / x) = 7) : x^6 + (1 / x^6) = 103682 := by
  sorry

end NUMINAMATH_GPT_calc_x6_plus_inv_x6_l2144_214482


namespace NUMINAMATH_GPT_steps_taken_l2144_214472

noncomputable def andrewSpeed : ℝ := 1 -- Let Andrew's speed be represented by 1 feet per minute
noncomputable def benSpeed : ℝ := 3 * andrewSpeed -- Ben's speed is 3 times Andrew's speed
noncomputable def totalDistance : ℝ := 21120 -- Distance between the houses in feet
noncomputable def andrewStep : ℝ := 3 -- Each step of Andrew covers 3 feet

theorem steps_taken : (totalDistance / (andrewSpeed + benSpeed)) * andrewSpeed / andrewStep = 1760 := by
  sorry -- proof to be filled in later

end NUMINAMATH_GPT_steps_taken_l2144_214472


namespace NUMINAMATH_GPT_line_equation_45_deg_through_point_l2144_214473

theorem line_equation_45_deg_through_point :
  ∀ (x y : ℝ), 
  (∃ m k: ℝ, m = 1 ∧ k = 5 ∧ y = m * x + k) ∧ (∃ p q : ℝ, p = -2 ∧ q = 3 ∧ y = q ) :=  
  sorry

end NUMINAMATH_GPT_line_equation_45_deg_through_point_l2144_214473
