import Mathlib

namespace jellybeans_left_l68_68614

theorem jellybeans_left :
  let initial_jellybeans := 500
  let total_kindergarten := 10
  let total_firstgrade := 10
  let total_secondgrade := 10
  let sick_kindergarten := 2
  let sick_secondgrade := 3
  let jellybeans_sick_kindergarten := 5
  let jellybeans_sick_secondgrade := 10
  let jellybeans_remaining_kindergarten := 3
  let jellybeans_firstgrade := 5
  let jellybeans_secondgrade_per_firstgrade := 5 / 2 * total_firstgrade
  let consumed_by_sick := sick_kindergarten * jellybeans_sick_kindergarten + sick_secondgrade * jellybeans_sick_secondgrade
  let remaining_kindergarten := total_kindergarten - sick_kindergarten
  let consumed_by_remaining := remaining_kindergarten * jellybeans_remaining_kindergarten + total_firstgrade * jellybeans_firstgrade + total_secondgrade * jellybeans_secondgrade_per_firstgrade
  let total_consumed := consumed_by_sick + consumed_by_remaining
  initial_jellybeans - total_consumed = 176 := by 
  sorry

end jellybeans_left_l68_68614


namespace seq_nat_eq_n_l68_68149

theorem seq_nat_eq_n (a : ℕ → ℕ) (h_inc : ∀ n, a n < a (n + 1))
  (h_le : ∀ n, a n ≤ n + 2020)
  (h_div : ∀ n, a (n + 1) ∣ (n^3 * a n - 1)) :
  ∀ n, a n = n :=
by
  sorry

end seq_nat_eq_n_l68_68149


namespace uphill_flat_road_system_l68_68541

variables {x y : ℝ}

theorem uphill_flat_road_system :
  (3 : ℝ)⁻¹ * x + (4 : ℝ)⁻¹ * y = 70 / 60 ∧
  (4 : ℝ)⁻¹ * y + (5 : ℝ)⁻¹ * x = 54 / 60 :=
sorry

end uphill_flat_road_system_l68_68541


namespace negation_of_proposition_l68_68239

theorem negation_of_proposition :
  (¬ (∀ a b : ℤ, a = 0 → a * b = 0)) ↔ (∃ a b : ℤ, a = 0 ∧ a * b ≠ 0) :=
by
  sorry

end negation_of_proposition_l68_68239


namespace smallest_c_minus_a_l68_68077

theorem smallest_c_minus_a (a b c : ℕ) (h1 : a * b * c = 720) (h2 : a < b) (h3 : b < c) : c - a ≥ 24 :=
sorry

end smallest_c_minus_a_l68_68077


namespace arithmetic_sequence_sum_l68_68698

theorem arithmetic_sequence_sum (S : ℕ → ℝ) (h_arith : ∀ k, S (k + 1) - S k = S 1 - S 0)
  (h_S5 : S 5 = 10) (h_S10 : S 10 = 18) : S 15 = 26 :=
by
  -- Rest of the proof goes here
  sorry

end arithmetic_sequence_sum_l68_68698


namespace elements_author_is_euclid_l68_68459

def author_of_elements := "Euclid"

theorem elements_author_is_euclid : author_of_elements = "Euclid" :=
by
  rfl -- Reflexivity of equality, since author_of_elements is defined to be "Euclid".

end elements_author_is_euclid_l68_68459


namespace least_five_digit_perfect_square_and_cube_l68_68305

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, n = 5^6 ∧ (n >= 10000 ∧ n < 100000) ∧ (∃ k : ℕ, n = k^2) ∧ (∃ m : ℕ, n = m^3) :=
by
  use 15625
  split
  · exact pow_succ 5 6
  split
  · have h1: 10000 ≤ 5^6 := nat.le_of_eq (calc 5^6 = 15625 : rfl)
    have h2: 5^6 < 100000 := nat.lt_of_eq_of_lt (calc 5^6 = 15625 : rfl) (by decide)
    exact ⟨h1, h2⟩
  · use 125
    exact pow_two 125
  · use 25
    exact pow_three 25
  sorry

end least_five_digit_perfect_square_and_cube_l68_68305


namespace hyperbola_eccentricity_is_sqrt2_l68_68608

noncomputable theory

-- Definition of the parabola C1 and its focus F
def parabola (p : ℝ) : ℝ × ℝ → Prop :=
λ (P : ℝ × ℝ), P.1 ^ 2 = 2 * p * P.2

-- Definition of the hyperbola C2 and its foci F1, F2
def hyperbola (a b : ℝ) : ℝ × ℝ → Prop :=
λ (P: ℝ × ℝ), (P.1 ^ 2 / a ^ 2) - (P.2 ^ 2 / b ^ 2) = 1

-- The eccentricity of a hyperbola
def hyperbola_eccentricity (a b : ℝ) : ℝ :=
sqrt (1 + (b / a)^2)

theorem hyperbola_eccentricity_is_sqrt2
    (p a b : ℝ)
    (F P F1 F2 : ℝ × ℝ)
    (hP1 : parabola p P)
    (hP2 : hyperbola a b P)
    (hF : F = (0, p / 2))
    (hF1 : F1 = (c, 0))
    (hF2 : F2 = (-c, 0))
    (hline : P.1 / P.2 = F1.1 / F1.2)
    (hc : c = b^2 / a)
    (hcollinear : ∀ {x y z : ℝ × ℝ}, 
        function.LinearlyIndependent ℝ ![[x, y], [x, z], [y, z]])
    (htangent : ∀ P', tangent parabola p P' = tangent hyperbola a b P')
    : hyperbola_eccentricity a b = sqrt 2 := sorry

end hyperbola_eccentricity_is_sqrt2_l68_68608


namespace books_more_than_movies_l68_68249

theorem books_more_than_movies (books_count movies_count read_books watched_movies : ℕ) 
  (h_books : books_count = 10)
  (h_movies : movies_count = 6)
  (h_read_books : read_books = 10) 
  (h_watched_movies : watched_movies = 6) : 
  read_books - watched_movies = 4 := by
  sorry

end books_more_than_movies_l68_68249


namespace inequality_ge_one_l68_68763

open Nat

variable (p q : ℝ) (m n : ℕ)

def conditions := p ≥ 0 ∧ q ≥ 0 ∧ p + q = 1 ∧ m > 0 ∧ n > 0

theorem inequality_ge_one (h : conditions p q m n) :
  (1 - p^m)^n + (1 - q^n)^m ≥ 1 := 
by sorry

end inequality_ge_one_l68_68763


namespace find_a_l68_68976

theorem find_a (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 119) : a = 59 := 
sorry

end find_a_l68_68976


namespace sum_of_odd_integers_15_to_51_l68_68872

def odd_arithmetic_series_sum (a1 an d : ℤ) (n : ℕ) : ℤ :=
  (n * (a1 + an)) / 2

theorem sum_of_odd_integers_15_to_51 :
  odd_arithmetic_series_sum 15 51 2 19 = 627 :=
by
  sorry

end sum_of_odd_integers_15_to_51_l68_68872


namespace sum_and_ratio_implies_difference_l68_68864

theorem sum_and_ratio_implies_difference (a b : ℚ) (h1 : a + b = 500) (h2 : a / b = 0.8) : b - a = 55.55555555555556 := by
  sorry

end sum_and_ratio_implies_difference_l68_68864


namespace intersecting_line_at_one_point_l68_68228

theorem intersecting_line_at_one_point (k : ℝ) :
  (∃ y : ℝ, k = -3 * y^2 - 4 * y + 7 ∧ 
           ∀ z : ℝ, k = -3 * z^2 - 4 * z + 7 → y = z) ↔ 
  k = 25 / 3 :=
by
  sorry

end intersecting_line_at_one_point_l68_68228


namespace negation_of_P_l68_68599

theorem negation_of_P : ¬(∀ x : ℝ, x^2 + 1 ≥ 2 * x) ↔ ∃ x : ℝ, x^2 + 1 < 2 * x :=
by sorry

end negation_of_P_l68_68599


namespace percentage_increase_after_decrease_l68_68073

theorem percentage_increase_after_decrease (P : ℝ) :
  let P_decreased := 0.70 * P
  let P_final := 1.16 * P
  let x := ((P_final / P_decreased) - 1) * 100
  (P_decreased * (1 + x / 100) = P_final) → x = 65.71 := 
by 
  intros
  let P_decreased := 0.70 * P
  let P_final := 1.16 * P
  let x := ((P_final / P_decreased) - 1) * 100
  have h : (P_decreased * (1 + x / 100) = P_final) := by assumption
  sorry

end percentage_increase_after_decrease_l68_68073


namespace find_a_l68_68934

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 3^x + a / (3^x + 1)

theorem find_a (a : ℝ) : 
  (∀ x : ℝ, 3^x + a / (3^x + 1) ≥ 5) ∧ (∃ x : ℝ, 3^x + a / (3^x + 1) = 5) 
  → a = 9 := 
by 
  intro h
  sorry

end find_a_l68_68934


namespace expression_evaluation_l68_68969

theorem expression_evaluation (a b c d : ℤ) : 
  a / b - c * d^2 = a / (b - c * d^2) :=
sorry

end expression_evaluation_l68_68969


namespace geometric_sequence_a5_l68_68555

theorem geometric_sequence_a5 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_a3 : a 3 = -4) 
  (h_a7 : a 7 = -16) 
  : a 5 = -8 :=
sorry

end geometric_sequence_a5_l68_68555


namespace wall_length_to_height_ratio_l68_68607

theorem wall_length_to_height_ratio (W H L V : ℝ) (h1 : H = 6 * W) (h2 : V = W * H * L) (h3 : W = 4) (h4 : V = 16128) :
  L / H = 7 :=
by
  -- Note: The proof steps are omitted as per the problem's instructions.
  sorry

end wall_length_to_height_ratio_l68_68607


namespace game_A_greater_game_B_l68_68501

-- Defining the probabilities and independence condition
def P_H := 2 / 3
def P_T := 1 / 3
def independent_tosses := true

-- Game A Probability Definition
def P_A := (P_H ^ 3) + (P_T ^ 3)

-- Game B Probability Definition
def P_B := ((P_H ^ 2) + (P_T ^ 2)) ^ 2

-- Statement to be proved
theorem game_A_greater_game_B : P_A = (27:ℚ) / 81 ∧ P_B = (25:ℚ) / 81 ∧ ((27:ℚ) / 81 - (25:ℚ) / 81 = (2:ℚ) / 81) := 
by
  -- P_A has already been computed: 1/3 = 27/81
  -- P_B has already been computed: 25/81
  sorry

end game_A_greater_game_B_l68_68501


namespace minimize_S_l68_68859

theorem minimize_S (n : ℕ) (a : ℕ → ℤ) (h : ∀ n, a n = 3 * n - 23) : n = 7 ↔ ∃ (m : ℕ), (∀ k ≤ m, a k <= 0) ∧ m = 7 :=
by
  sorry

end minimize_S_l68_68859


namespace moles_NaOH_combined_with_HCl_l68_68890

-- Definitions for given conditions
def NaOH : Type := Unit
def HCl : Type := Unit
def NaCl : Type := Unit
def H2O : Type := Unit

def balanced_reaction (nHCl nNaOH nNaCl nH2O : ℕ) : Prop :=
  nHCl = nNaOH ∧ nNaOH = nNaCl ∧ nNaCl = nH2O

def mole_mass_H2O : ℕ := 18

-- Given: certain amount of NaOH combined with 1 mole of HCl
def initial_moles_HCl : ℕ := 1

-- Given: 18 grams of H2O formed
def grams_H2O : ℕ := 18

-- Molar mass of H2O is approximately 18 g/mol, so 18 grams is 1 mole
def moles_H2O : ℕ := grams_H2O / mole_mass_H2O

-- Prove that number of moles of NaOH combined with HCl is 1 mole
theorem moles_NaOH_combined_with_HCl : 
  balanced_reaction initial_moles_HCl 1 1 moles_H2O →
  moles_H2O = 1 →
  1 = 1 :=
by
  intros h1 h2
  sorry

end moles_NaOH_combined_with_HCl_l68_68890


namespace find_smallest_value_l68_68441

noncomputable def smallest_value (a b c d : ℝ) : ℝ := a^2 + b^2 + c^2 + d^2

theorem find_smallest_value (a b c d : ℝ) (h1: a + b = 18)
  (h2: ab + c + d = 85) (h3: ad + bc = 180) (h4: cd = 104) :
  smallest_value a b c d = 484 :=
sorry

end find_smallest_value_l68_68441


namespace polygon_sides_l68_68682

theorem polygon_sides (n : ℕ) (h : n - 3 = 5) : n = 8 :=
by {
  sorry
}

end polygon_sides_l68_68682


namespace Lin_peels_15_potatoes_l68_68165

-- Define the conditions
def total_potatoes : Nat := 60
def homer_rate : Nat := 2 -- potatoes per minute
def christen_rate : Nat := 3 -- potatoes per minute
def lin_rate : Nat := 4 -- potatoes per minute
def christen_join_time : Nat := 6 -- minutes
def lin_join_time : Nat := 9 -- minutes

-- Prove that Lin peels 15 potatoes
theorem Lin_peels_15_potatoes :
  ∃ (lin_potatoes : Nat), lin_potatoes = 15 :=
by
  sorry

end Lin_peels_15_potatoes_l68_68165


namespace least_five_digit_is_15625_l68_68279

noncomputable def least_five_digit_perfect_square_and_cube : ℕ :=
  15625

theorem least_five_digit_is_15625 :
  (10000 ≤ least_five_digit_perfect_square_and_cube) ∧ (least_five_digit_perfect_square_and_cube < 100000) ∧
  (∃ a : ℕ, least_five_digit_perfect_square_and_cube = a ^ 6) :=
by
  sorry

end least_five_digit_is_15625_l68_68279


namespace ratio_apples_simplified_l68_68209

variable (n : ℕ) (m : ℕ) (k : ℕ)
variable (a : n = 45) (b : m = 9) (c : k = 27)

theorem ratio_apples_simplified (n m k : ℕ) (a : n = 45) (b : m = 9) (c : k = 27) : 
  (n / n.gcd m / n.gcd k) = 5 ∧ (m / n.gcd m / n.gcd k) = 1 ∧ (k / n.gcd m / n.gcd k) = 3 := 
by
  sorry

end ratio_apples_simplified_l68_68209


namespace parabola_vector_sum_distance_l68_68195

noncomputable def parabola_focus (x y : ℝ) : Prop := x^2 = 8 * y

noncomputable def on_parabola (x y : ℝ) : Prop := parabola_focus x y

theorem parabola_vector_sum_distance :
  ∀ (A B C : ℝ × ℝ) (F : ℝ × ℝ),
  on_parabola A.1 A.2 ∧ on_parabola B.1 B.2 ∧ on_parabola C.1 C.2 ∧
  F = (0, 2) ∧
  ((A.1 - F.1)^2 + (A.2 - F.2)^2) + ((B.1 - F.1)^2 + (B.2 - F.2)^2) + ((C.1 - F.1)^2 + (C.2 - F.2)^2) = 0
  → (abs ((A.2 + F.2)) + abs ((B.2 + F.2)) + abs ((C.2 + F.2))) = 12 :=
by sorry

end parabola_vector_sum_distance_l68_68195


namespace confidence_95_implies_K2_gt_3_841_l68_68825

-- Conditions
def confidence_no_relationship (K2 : ℝ) : Prop := K2 ≤ 3.841
def confidence_related_95 (K2 : ℝ) : Prop := K2 > 3.841
def confidence_related_99 (K2 : ℝ) : Prop := K2 > 6.635

theorem confidence_95_implies_K2_gt_3_841 (K2 : ℝ) :
  confidence_related_95 K2 ↔ K2 > 3.841 :=
by sorry

end confidence_95_implies_K2_gt_3_841_l68_68825


namespace combined_distance_20_birds_two_seasons_l68_68339

theorem combined_distance_20_birds_two_seasons :
  let distance_jim_to_disney := 50
  let distance_disney_to_london := 60
  let number_of_birds := 20
  (number_of_birds * (distance_jim_to_disney + distance_disney_to_london)) = 2200 := by
  sorry

end combined_distance_20_birds_two_seasons_l68_68339


namespace select_at_least_8_sticks_l68_68405

theorem select_at_least_8_sticks (S : Finset ℕ) (hS : S = (Finset.range 92 \ {0})) :
  ∃ (sticks : Finset ℕ) (h_sticks : sticks.card = 8),
    ∃ (a b c : ℕ) (h_a : a ∈ sticks) (h_b : b ∈ sticks) (h_c : c ∈ sticks),
    (a + b > c) ∧ (b + c > a) ∧ (c + a > b) :=
by
  -- Proof required here
  sorry

end select_at_least_8_sticks_l68_68405


namespace sum_of_powers_of_i_l68_68921

noncomputable def i : Complex := Complex.I

theorem sum_of_powers_of_i :
  (Finset.range 2011).sum (λ n => i^(n+1)) = -1 := by
  sorry

end sum_of_powers_of_i_l68_68921


namespace cave_depth_l68_68464

theorem cave_depth (current_depth remaining_distance : ℕ) (h₁ : current_depth = 849) (h₂ : remaining_distance = 369) :
  current_depth + remaining_distance = 1218 :=
by
  sorry

end cave_depth_l68_68464


namespace least_five_digit_is_15625_l68_68280

noncomputable def least_five_digit_perfect_square_and_cube : ℕ :=
  15625

theorem least_five_digit_is_15625 :
  (10000 ≤ least_five_digit_perfect_square_and_cube) ∧ (least_five_digit_perfect_square_and_cube < 100000) ∧
  (∃ a : ℕ, least_five_digit_perfect_square_and_cube = a ^ 6) :=
by
  sorry

end least_five_digit_is_15625_l68_68280


namespace value_of_k_l68_68820

theorem value_of_k :
  3^1999 - 3^1998 - 3^1997 + 3^1996 = 16 * 3^1996 :=
by sorry

end value_of_k_l68_68820


namespace cost_per_minute_l68_68036

-- Conditions as Lean definitions
def initial_credit : ℝ := 30
def remaining_credit : ℝ := 26.48
def call_duration : ℝ := 22

-- Question: How much does a long distance call cost per minute?

theorem cost_per_minute :
  (initial_credit - remaining_credit) / call_duration = 0.16 := 
by
  sorry

end cost_per_minute_l68_68036


namespace tan_alpha_minus_pi_over_4_l68_68544

theorem tan_alpha_minus_pi_over_4 
  (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2) 
  (h2 : Real.tan (β + π/4) = 3) 
  : Real.tan (α - π/4) = -1 / 7 :=
by
  sorry

end tan_alpha_minus_pi_over_4_l68_68544


namespace car_overtakes_buses_l68_68659

/-- 
  Buses leave the airport every 3 minutes. 
  A bus takes 60 minutes to travel from the airport to the city center. 
  A car takes 35 minutes to travel from the airport to the city center. 
  Prove that the car overtakes 8 buses on its way to the city center excluding the bus it left with.
--/
theorem car_overtakes_buses (arr_bus : ℕ) (arr_car : ℕ) (interval : ℕ) (diff : ℕ) : 
  interval = 3 → arr_bus = 60 → arr_car = 35 → diff = arr_bus - arr_car →
  ∃ n : ℕ, n = diff / interval ∧ n = 8 := by
  sorry

end car_overtakes_buses_l68_68659


namespace total_pencils_correct_l68_68357

-- Define the number of pencils Reeta has
def ReetaPencils : ℕ := 20

-- Define the number of pencils Anika has based on the conditions
def AnikaPencils : ℕ := 2 * ReetaPencils + 4

-- Define the total number of pencils Anika and Reeta have together
def TotalPencils : ℕ := ReetaPencils + AnikaPencils

-- Statement to prove
theorem total_pencils_correct : TotalPencils = 64 :=
by
  sorry

end total_pencils_correct_l68_68357


namespace donuts_percentage_missing_l68_68570

noncomputable def missing_donuts_percentage (initial_donuts : ℕ) (remaining_donuts : ℕ) : ℝ :=
  ((initial_donuts - remaining_donuts : ℕ) : ℝ) / initial_donuts * 100

theorem donuts_percentage_missing
  (h_initial : ℕ := 30)
  (h_remaining : ℕ := 9) :
  missing_donuts_percentage h_initial h_remaining = 70 :=
by
  sorry

end donuts_percentage_missing_l68_68570


namespace midpoint_product_coordinates_l68_68589

theorem midpoint_product_coordinates :
  ∃ (x y : ℝ), (4 : ℝ) = (-2 + x) / 2 ∧ (-3 : ℝ) = (-7 + y) / 2 ∧ x * y = 10 := by
  sorry

end midpoint_product_coordinates_l68_68589


namespace least_five_digit_perfect_square_and_cube_l68_68294

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (n >= 10000 ∧ n < 100000) ∧ (∃ a : ℕ, n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68294


namespace least_value_x_y_z_l68_68826

theorem least_value_x_y_z 
  (x y z : ℕ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (h_eq: 2 * x = 5 * y) 
  (h_eq': 5 * y = 8 * z) : 
  x + y + z = 33 :=
by 
  sorry

end least_value_x_y_z_l68_68826


namespace find_x_l68_68667

noncomputable section

variable (x : ℝ)
def vector_v : ℝ × ℝ := (x, 4)
def vector_w : ℝ × ℝ := (5, 2)
def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let num := (v.1 * w.1 + v.2 * w.2)
  let den := (w.1 * w.1 + w.2 * w.2)
  (num / den * w.1, num / den * w.2)

theorem find_x (h : projection (vector_v x) (vector_w) = (3, 1.2)) : 
  x = 47 / 25 :=
by
  sorry

end find_x_l68_68667


namespace simplify_expr_l68_68995

-- Define the variables a and b as real numbers
variables {a b : ℝ}

-- Define the mathematical expression in the problem
def expr1 : ℝ := (a + 2 * b) / (a + b)
def expr2 : ℝ := (a - b) / (a - 2 * b)
def expr3 : ℝ := (a ^ 2 - b ^ 2) / (a ^ 2 - 4 * a * b + 4 * b ^ 2)
def lhs : ℝ := expr1 - (expr2 / expr3)

-- The simplified expression
def rhs : ℝ := (4 * b) / (a + b)

-- Prove the equivalence under the given conditions
theorem simplify_expr (h₁ : a ≠ -b) (h₂ : a ≠ 2 * b) (h₃ : a ≠ b) : lhs = rhs := by
sorry

end simplify_expr_l68_68995


namespace repair_cost_l68_68507

theorem repair_cost (C : ℝ) (repair_cost : ℝ) (profit : ℝ) (selling_price : ℝ)
  (h1 : repair_cost = 0.10 * C)
  (h2 : profit = 1100)
  (h3 : selling_price = 1.20 * C)
  (h4 : profit = selling_price - C) :
  repair_cost = 550 :=
by
  sorry

end repair_cost_l68_68507


namespace hcf_of_numbers_l68_68220

def lcm_factors (a b l : ℕ) : Prop := lcm a b = l

theorem hcf_of_numbers : 
  ∃ N1 N2,
    max N1 N2 = 600 ∧
    lcm_factors N1 N2 (11 * 12) ∧
    Nat.gcd N1 N2 = 12 :=
by
  sorry

end hcf_of_numbers_l68_68220


namespace least_five_digit_perfect_square_and_cube_l68_68310

/-- 
Prove that the least five-digit whole number that is both a perfect square 
and a perfect cube is 15625.
-/
theorem least_five_digit_perfect_square_and_cube : 
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k : ℕ, n = k^6) ∧ (∀ m : ℕ, (10000 ≤ m ∧ m < 100000) ∧ (∃ l : ℕ, m = l^6) → n ≤ m) :=
begin
  use 15625,
  split,
  { split,
    { norm_num },
    { norm_num } },
  { use 5,
    norm_num,
    intros m hm,
    rcases hm with ⟨⟨hm1, hm2⟩, ⟨l, rfl⟩⟩,
    cases le_or_lt l 5 with hl hl,
    { by_contradiction h,
      have : m < 10000 := by norm_num [nat.pow_succ, nat.pow_succ, hl],
      exact not_le_of_gt this hm1 },
    { have : l ≥ 6 := nat.lt_succ_iff.mp hl,
      have : m ≥ 6^6 := nat.pow_le_pow_of_le_right nat.zero_lt_succ_iff.1 this,
      norm_num at this,
      exact not_lt_of_ge this hm2 } }
end

end least_five_digit_perfect_square_and_cube_l68_68310


namespace probability_same_color_pair_l68_68031

theorem probability_same_color_pair : 
  let total_shoes := 28
  let black_pairs := 8
  let brown_pairs := 4
  let gray_pairs := 2
  total_shoes = 2 * (black_pairs + brown_pairs + gray_pairs) → 
  ∃ (prob : ℚ), prob = 7 / 32 := by
  sorry

end probability_same_color_pair_l68_68031


namespace original_rectangle_area_l68_68067

theorem original_rectangle_area
  (A : ℝ)
  (h1 : ∀ (a : ℝ), a = 2 * A)
  (h2 : 4 * A = 32) : 
  A = 8 := 
by
  sorry

end original_rectangle_area_l68_68067


namespace least_five_digit_perfect_square_and_cube_l68_68287

theorem least_five_digit_perfect_square_and_cube : 
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ (∃ a : ℕ, n = a ^ 6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68287


namespace polynomial_value_l68_68807

theorem polynomial_value (a b : ℝ) (h₁ : a * b = 7) (h₂ : a + b = 2) : a^2 * b + a * b^2 - 20 = -6 :=
by {
  sorry
}

end polynomial_value_l68_68807


namespace winnie_keeps_10_lollipops_l68_68753

def winnie_keep_lollipops : Prop :=
  let cherry := 72
  let wintergreen := 89
  let grape := 23
  let shrimp_cocktail := 316
  let total_lollipops := cherry + wintergreen + grape + shrimp_cocktail
  let friends := 14
  let lollipops_per_friend := total_lollipops / friends
  let winnie_keeps := total_lollipops % friends
  winnie_keeps = 10

theorem winnie_keeps_10_lollipops : winnie_keep_lollipops := by
  sorry

end winnie_keeps_10_lollipops_l68_68753


namespace least_five_digit_perfect_square_cube_l68_68274

theorem least_five_digit_perfect_square_cube :
  ∃ n : Nat, (10000 ≤ n ∧ n < 100000) ∧ (∃ m1 m2 : Nat, n = m1^2 ∧ n = m2^3 ∧ n = 15625) :=
by
  sorry

end least_five_digit_perfect_square_cube_l68_68274


namespace algebraic_expression_value_l68_68145

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 * y = -4) :
  (2 * y - x) ^ 2 - 2 * x + 4 * y - 1 = 23 :=
by
  sorry

end algebraic_expression_value_l68_68145


namespace find_n_l68_68837

open Nat

theorem find_n (n : ℕ) (d : ℕ → ℕ) (h1 : d 1 = 1) (hk : d 6^2 + d 7^2 - 1 = n) :
  n = 1984 ∨ n = 144 :=
by
  sorry

end find_n_l68_68837


namespace bowling_tournament_prize_orders_l68_68024
-- Import necessary Lean library

-- Define the conditions
def match_outcome (num_games : ℕ) : ℕ := 2 ^ num_games

-- Theorem statement
theorem bowling_tournament_prize_orders : match_outcome 5 = 32 := by
  -- This is the statement, proof is not required
  sorry

end bowling_tournament_prize_orders_l68_68024


namespace surface_area_after_removal_l68_68887

theorem surface_area_after_removal :
  let cube_side := 4
  let corner_cube_side := 2
  let original_surface_area := 6 * (cube_side * cube_side)
  (original_surface_area = 96) ->
  (6 * (cube_side * cube_side) - 8 * 3 * (corner_cube_side * corner_cube_side) + 8 * 3 * (corner_cube_side * corner_cube_side) = 96) :=
by
  intros
  sorry

end surface_area_after_removal_l68_68887


namespace maximum_value_x2y_y2z_z2x_l68_68916

theorem maximum_value_x2y_y2z_z2x (x y z : ℝ) (h_sum : x + y + z = 0) (h_squares : x^2 + y^2 + z^2 = 6) :
  x^2 * y + y^2 * z + z^2 * x ≤ 6 :=
sorry

end maximum_value_x2y_y2z_z2x_l68_68916


namespace find_x_l68_68248

theorem find_x (x : ℚ) (h : (3 - x) / (2 - x) - 1 / (x - 2) = 3) : x = 1 := 
  sorry

end find_x_l68_68248


namespace cannot_cut_out_rect_l68_68445

noncomputable def square_area : ℝ := 400
noncomputable def rect_area : ℝ := 300
noncomputable def length_to_width_ratio : ℝ × ℝ := (3, 2)

theorem cannot_cut_out_rect (h1: square_area = 400) (h2: rect_area = 300) (h3: length_to_width_ratio = (3, 2)) : 
  false := sorry

end cannot_cut_out_rect_l68_68445


namespace smallest_circle_covering_region_l68_68161

/-- 
Given the conditions describing the plane region:
1. x ≥ 0
2. y ≥ 0
3. x + 2y - 4 ≤ 0

Prove that the equation of the smallest circle covering this region is (x - 2)² + (y - 1)² = 5.
-/
theorem smallest_circle_covering_region :
  (∀ (x y : ℝ), (x ≥ 0 ∧ y ≥ 0 ∧ x + 2 * y - 4 ≤ 0) → (x - 2)^2 + (y - 1)^2 ≤ 5) :=
sorry

end smallest_circle_covering_region_l68_68161


namespace find_x_l68_68417

noncomputable def a : ℝ × ℝ := (2, 1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (x, 2)
noncomputable def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
noncomputable def scalar_vec_mul (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)
noncomputable def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

theorem find_x (x : ℝ) :
  (vec_add a (b x)).1 * (vec_sub a (scalar_vec_mul 2 (b x))).2 =
  (vec_add a (b x)).2 * (vec_sub a (scalar_vec_mul 2 (b x))).1 →
  x = 4 :=
by sorry

end find_x_l68_68417


namespace find_R_position_l68_68847

theorem find_R_position :
  ∀ (P Q R : ℤ), P = -6 → Q = -1 → Q = (P + R) / 2 → R = 4 :=
by
  intros P Q R hP hQ hQ_halfway
  sorry

end find_R_position_l68_68847


namespace largest_real_number_l68_68392

theorem largest_real_number (x : ℝ) (h : ⌊x⌋ / x = 8 / 9) : x ≤ 63 / 8 :=
sorry

end largest_real_number_l68_68392


namespace possible_amounts_l68_68827

theorem possible_amounts (n : ℕ) : 
  ¬ (∃ x y : ℕ, 3 * x + 5 * y = n) ↔ n = 1 ∨ n = 2 ∨ n = 4 ∨ n = 7 :=
sorry

end possible_amounts_l68_68827


namespace prove_f_x1_minus_f_x2_lt_zero_l68_68683

variable {f : ℝ → ℝ}

-- Define even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Specify that f is decreasing for x < 0
def decreasing_on_negative (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < 0 → y < 0 → x < y → f x > f y

theorem prove_f_x1_minus_f_x2_lt_zero (hx1x2 : |x1| < |x2|)
  (h_even : even_function f)
  (h_decreasing : decreasing_on_negative f) :
  f x1 - f x2 < 0 :=
sorry

end prove_f_x1_minus_f_x2_lt_zero_l68_68683


namespace probability_sibling_pair_l68_68879

-- Define the necessary constants for the problem.
def B : ℕ := 500 -- Number of business students
def L : ℕ := 800 -- Number of law students
def S : ℕ := 30  -- Number of sibling pairs

-- State the theorem representing the mathematical proof problem
theorem probability_sibling_pair :
  (S : ℝ) / (B * L) = 0.000075 := sorry

end probability_sibling_pair_l68_68879


namespace max_sum_small_numbers_l68_68736

-- Definition: A number is either "big" or "small"
def is_big_or_small (n : ℕ) (neighbors : List ℕ) :=
  (n > neighbors.head! ∧ n > neighbors.tail.head!) ∨ (n < neighbors.head! ∧ n < neighbors.tail.head!)

-- Given the conditions
def conditions (circle: List ℕ) : Prop :=
  circle.length = 8 ∧ 
  ∀ i, is_big_or_small (circle.nth_le i sorry) [circle.nth_le ((i-1) % 8) sorry, circle.nth_le ((i+1) % 8) sorry]

-- Proof statement: The maximum possible sum of the small numbers is 13
theorem max_sum_small_numbers : ∀ circle: List ℕ, conditions circle → ∑ i in (Finset.filter (λ n, n = circle.nth_le i sorry ∧ n < circle.nth_le ((i-1) % 8) sorry ∧ n < circle.nth_le ((i+1) % 8) sorry) (Finset.range 8)), circle.nth_le i sorry = 13 :=
by
  -- Here you would generally prove the theorem
  sorry

end max_sum_small_numbers_l68_68736


namespace paolo_coconuts_l68_68990

theorem paolo_coconuts
  (P : ℕ)
  (dante_coconuts : ℕ := 3 * P)
  (dante_sold : ℕ := 10)
  (dante_left : ℕ := 32)
  (h : dante_left + dante_sold = dante_coconuts) : P = 14 :=
by {
  sorry
}

end paolo_coconuts_l68_68990


namespace fill_tank_time_l68_68878

theorem fill_tank_time 
  (tank_capacity : ℕ) (initial_fill : ℕ) (fill_rate : ℝ) 
  (drain_rate1 : ℝ) (drain_rate2 : ℝ) : 
  tank_capacity = 8000 ∧ initial_fill = 4000 ∧ fill_rate = 0.5 ∧ drain_rate1 = 0.25 ∧ drain_rate2 = 0.1667 
  → (initial_fill + fill_rate * t - (drain_rate1 + drain_rate2) * t) = tank_capacity → t = 48 := sorry

end fill_tank_time_l68_68878


namespace part_a_first_player_wins_part_b_first_player_wins_l68_68758

/-- Define the initial state of the game -/
structure GameState :=
(pile1 : Nat) (pile2 : Nat)

/-- Define the moves allowed in Part a) -/
inductive MoveA
| take_from_pile1 : MoveA
| take_from_pile2 : MoveA
| take_from_both  : MoveA

/-- Define the moves allowed in Part b) -/
inductive MoveB
| take_from_pile1 : MoveB
| take_from_pile2 : MoveB
| take_from_both  : MoveB
| transfer_to_pile2 : MoveB

/-- Define what it means for the first player to have a winning strategy in part a) -/
def first_player_wins_a (initial_state : GameState) : Prop := sorry

/-- Define what it means for the first player to have a winning strategy in part b) -/
def first_player_wins_b (initial_state : GameState) : Prop := sorry

/-- Theorem statement for part a) -/
theorem part_a_first_player_wins :
  first_player_wins_a ⟨7, 7⟩ :=
sorry

/-- Theorem statement for part b) -/
theorem part_b_first_player_wins :
  first_player_wins_b ⟨7, 7⟩ :=
sorry

end part_a_first_player_wins_part_b_first_player_wins_l68_68758


namespace part_one_part_two_l68_68549

variable {a b c : ℝ}
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_eq : a^2 + b^2 + 4*c^2 = 3)

theorem part_one (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 + b^2 + 4*c^2 = 3) :
  a + b + 2*c ≤ 3 :=
sorry

theorem part_two (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 + b^2 + 4*c^2 = 3) (h_b_eq_2c : b = 2*c) :
  1/a + 1/c ≥ 3 :=
sorry

end part_one_part_two_l68_68549


namespace number_of_B_students_l68_68428

/-- Let x be the number of students who earn a B. 
    Given the conditions:
    - The number of students who earn an A is 0.5x.
    - The number of students who earn a C is 2x.
    - The number of students who earn a D is 0.3x.
    - The total number of students in the class is 40.
    Prove the number of students who earn a B is 40 / 3.8 = 200 / 19, approximately 11. -/
theorem number_of_B_students (x : ℝ) (h_bA: x * 0.5 + x + x * 2 + x * 0.3 = 40) : 
  x = 40 / 3.8 :=
by 
  sorry

end number_of_B_students_l68_68428


namespace number_of_math_players_l68_68508

theorem number_of_math_players (total_players physics_players both_players : ℕ)
    (h1 : total_players = 25)
    (h2 : physics_players = 15)
    (h3 : both_players = 6)
    (h4 : total_players = physics_players + (total_players - physics_players - (total_players - physics_players - both_players)) + both_players ) :
  total_players - (physics_players - both_players) = 16 :=
sorry

end number_of_math_players_l68_68508


namespace train_overtakes_motorbike_in_80_seconds_l68_68506

-- Definitions of the given conditions
def speed_train_kmph : ℝ := 100
def speed_motorbike_kmph : ℝ := 64
def length_train_m : ℝ := 800.064

-- Definition to convert kmph to m/s
noncomputable def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

-- Relative speed in m/s
noncomputable def relative_speed_mps : ℝ :=
  kmph_to_mps (speed_train_kmph - speed_motorbike_kmph)

-- Time taken for the train to overtake the motorbike
noncomputable def time_to_overtake (distance_m : ℝ) (speed_mps : ℝ) : ℝ :=
  distance_m / speed_mps

-- The statement to be proved
theorem train_overtakes_motorbike_in_80_seconds :
  time_to_overtake length_train_m relative_speed_mps = 80.0064 :=
by
  sorry

end train_overtakes_motorbike_in_80_seconds_l68_68506


namespace sara_height_correct_l68_68730

variable (Roy_height : ℕ)
variable (Joe_height : ℕ)
variable (Sara_height : ℕ)

def problem_conditions (Roy_height Joe_height Sara_height : ℕ) : Prop :=
  Roy_height = 36 ∧
  Joe_height = Roy_height + 3 ∧
  Sara_height = Joe_height + 6

theorem sara_height_correct (Roy_height Joe_height Sara_height : ℕ) :
  problem_conditions Roy_height Joe_height Sara_height → Sara_height = 45 := by
  sorry

end sara_height_correct_l68_68730


namespace combined_cost_price_l68_68751

theorem combined_cost_price :
  let stock1_price := 100
  let stock1_discount := 5 / 100
  let stock1_brokerage := 1.5 / 100
  let stock2_price := 200
  let stock2_discount := 7 / 100
  let stock2_brokerage := 0.75 / 100
  let stock3_price := 300
  let stock3_discount := 3 / 100
  let stock3_brokerage := 1 / 100

  -- Calculated values
  let stock1_discounted_price := stock1_price * (1 - stock1_discount)
  let stock1_total_price := stock1_discounted_price * (1 + stock1_brokerage)
  
  let stock2_discounted_price := stock2_price * (1 - stock2_discount)
  let stock2_total_price := stock2_discounted_price * (1 + stock2_brokerage)
  
  let stock3_discounted_price := stock3_price * (1 - stock3_discount)
  let stock3_total_price := stock3_discounted_price * (1 + stock3_brokerage)
  
  let combined_cost := stock1_total_price + stock2_total_price + stock3_total_price
  combined_cost = 577.73 := sorry

end combined_cost_price_l68_68751


namespace no_real_solutions_l68_68785

theorem no_real_solutions (x : ℝ) : 
  x^(Real.log x / Real.log 2) ≠ x^4 / 256 :=
by
  sorry

end no_real_solutions_l68_68785


namespace minute_hand_position_l68_68868

theorem minute_hand_position (t : ℕ) (h_start : t = 2022) :
  let cycle_minutes := 8
  let net_movement_per_cycle := 2
  let full_cycles := t / cycle_minutes
  let remaining_minutes := t % cycle_minutes
  let full_cycles_movement := full_cycles * net_movement_per_cycle
  let extra_movement := if remaining_minutes <= 5 then remaining_minutes else 5 - (remaining_minutes - 5)
  let total_movement := full_cycles_movement + extra_movement
  (total_movement % 60) = 28 :=
by {
  sorry
}

end minute_hand_position_l68_68868


namespace call_cost_per_minute_l68_68034

-- Definitions (conditions)
def initial_credit : ℝ := 30
def call_duration : ℕ := 22
def remaining_credit : ℝ := 26.48

-- The goal is to prove that the cost per minute of the call is 0.16
theorem call_cost_per_minute :
  (initial_credit - remaining_credit) / call_duration = 0.16 := 
sorry

end call_cost_per_minute_l68_68034


namespace problem_statement_l68_68635

theorem problem_statement : (-0.125 ^ 2006) * (8 ^ 2005) = -0.125 := by
  sorry

end problem_statement_l68_68635


namespace solve_x_l68_68216

noncomputable def solveEquation (a b c d : ℝ) (x : ℝ) : Prop :=
  x = 3 * a * b + 33 * b^2 + 333 * c^3 + 3.33 * (Real.sin d)^4

theorem solve_x :
  solveEquation 2 (-1) 0.5 (Real.pi / 6) 68.833125 :=
by
  sorry

end solve_x_l68_68216


namespace impossible_odd_n_m_l68_68692

theorem impossible_odd_n_m (n m : ℤ) (h : Even (n^2 + m + n * m)) : ¬ (Odd n ∧ Odd m) :=
by
  intro h1
  sorry

end impossible_odd_n_m_l68_68692


namespace find_distance_walker_l68_68102

noncomputable def distance_walked (x t d : ℝ) : Prop :=
  (d = x * t) ∧
  (d = (x + 1) * (3 / 4) * t) ∧
  (d = (x - 1) * (t + 3))

theorem find_distance_walker (x t d : ℝ) (h : distance_walked x t d) : d = 18 := 
sorry

end find_distance_walker_l68_68102


namespace find_ratio_l68_68735

noncomputable def p (x : ℝ) : ℝ := 3 * x * (x - 5)
noncomputable def q (x : ℝ) : ℝ := (x + 2) * (x - 5)

theorem find_ratio : (p 3) / (q 3) = 9 / 5 := by
  sorry

end find_ratio_l68_68735


namespace largest_common_divisor_of_product_l68_68523

theorem largest_common_divisor_of_product (n : ℕ) (h_even : n % 2 = 0) (h_pos : 0 < n) :
  ∃ d : ℕ, d = 105 ∧ ∀ k : ℕ, k = (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13) → d ∣ k :=
by
  sorry

end largest_common_divisor_of_product_l68_68523


namespace range_of_a_for_monotonicity_l68_68009

noncomputable def f (x : ℝ) (a : ℝ) := (Real.sqrt (x^2 + 1)) - a * x

theorem range_of_a_for_monotonicity (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → f x a < f y a) ↔ a ≥ 1 := sorry

end range_of_a_for_monotonicity_l68_68009


namespace inequality_tangents_l68_68684

def f (x : ℝ) (a b : ℝ) : ℝ := x^3 - a * x - b

theorem inequality_tangents (a b : ℝ) (h1 : 0 < a)
  (h2 : ∃ x0 : ℝ, 2 * x0^3 - 3 * a * x0^2 + a^2 + 2 * b = 0): 
  -a^2 / 2 < b ∧ b < f a a b :=
by
  sorry

end inequality_tangents_l68_68684


namespace main_diagonal_squares_second_diagonal_composite_third_diagonal_composite_l68_68512

-- Problem Statement in Lean 4

theorem main_diagonal_squares (k : ℕ) : ∃ m : ℕ, (4 * k * (k + 1) + 1 = m * m) := 
sorry

theorem second_diagonal_composite (k : ℕ) (hk : k ≥ 1) : ∃ a b : ℕ, a ≠ 1 ∧ b ≠ 1 ∧ (4 * (2 * k * (2 * k - 1) - 1) + 1 = a * b) :=
sorry

theorem third_diagonal_composite (k : ℕ) : ∃ a b : ℕ, a ≠ 1 ∧ b ≠ 1 ∧ (4 * ((4 * k + 3) * (4 * k - 1)) + 1 = a * b) :=
sorry

end main_diagonal_squares_second_diagonal_composite_third_diagonal_composite_l68_68512


namespace average_of_remaining_two_numbers_l68_68086

theorem average_of_remaining_two_numbers (a b c d e f : ℝ)
(h_avg_6 : (a + b + c + d + e + f) / 6 = 3.95)
(h_avg_2_1 : (a + b) / 2 = 3.4)
(h_avg_2_2 : (c + d) / 2 = 3.85) :
  (e + f) / 2 = 4.6 := 
sorry

end average_of_remaining_two_numbers_l68_68086


namespace line_parabola_intersection_one_point_l68_68235

theorem line_parabola_intersection_one_point (k : ℝ) :
  (∃ y : ℝ, (-3 * y^2 - 4 * y + 7 = k) ∧ ∀ y1 y2 : ℝ, ( 3 * y1^2 + 4 * y1 + (k - 7) = 0 → 3 * y2^2 + 4 * y2 + (k - 7) = 0 → y1 = y2)) ↔ (k = 25 / 3) :=
by
  sorry

end line_parabola_intersection_one_point_l68_68235


namespace inequality_must_hold_l68_68670

theorem inequality_must_hold (a b c : ℝ) (h : (a / c^2) > (b / c^2)) (hc : c ≠ 0) : a^2 > b^2 :=
sorry

end inequality_must_hold_l68_68670


namespace jordan_time_for_7_miles_l68_68189

noncomputable def time_for_7_miles (jordan_miles : ℕ) (jordan_time : ℤ) : ℤ :=
  jordan_miles * jordan_time 

theorem jordan_time_for_7_miles :
  ∃ jordan_time : ℤ, (time_for_7_miles 7 (16 / 3)) = 112 / 3 :=
by
  sorry

end jordan_time_for_7_miles_l68_68189


namespace fraction_identity_l68_68941

variables (a b : ℚ)
hypothesis (h : a / 5 = b / 3)

theorem fraction_identity : (a - b) / (3 * a) = 2 / 15 := by
  sorry

end fraction_identity_l68_68941


namespace largest_real_number_l68_68393

theorem largest_real_number (x : ℝ) (h : ⌊x⌋ / x = 8 / 9) : x ≤ 63 / 8 :=
sorry

end largest_real_number_l68_68393


namespace multiple_of_sales_total_l68_68192

theorem multiple_of_sales_total
  (A : ℝ)
  (M : ℝ)
  (h : M * A = 0.3125 * (11 * A + M * A)) :
  M = 5 :=
by
  sorry

end multiple_of_sales_total_l68_68192


namespace least_five_digit_perfect_square_and_cube_l68_68259

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ ∃ k : ℕ, k^6 = n ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68259


namespace chess_tournament_solution_l68_68134

def chess_tournament_points (points : List ℝ) : Prop :=
  let andrey := points[0]
  let dima := points[1]
  let vanya := points[2]
  let sasha := points[3]
  andrey = 4 ∧ dima = 3.5 ∧ vanya = 2.5 ∧ sasha = 2

axiom chess_tournament_conditions (points : List ℝ) :
  -- Andrey secured first place, Dima secured second, Vanya secured third, and Sasha secured fourth.
  List.Nodup points ∧
  points.length = 4 ∧
  (∀ p, p ∈ points → p = 4 ∨ p = 3.5 ∨ p = 2.5 ∨ p = 2) ∧
  -- Andrey and Sasha won the same number of games.
  (points[0] ≠ points[1] ∧ points[0] ≠ points[2] ∧ points[0] ≠ points[3] ∧
   points[1] ≠ points[2] ∧ points[1] ≠ points[3] ∧
   points[2] ≠ points[3])

theorem chess_tournament_solution (points : List ℝ) :
  chess_tournament_conditions points → chess_tournament_points points :=
by
  sorry

end chess_tournament_solution_l68_68134


namespace outfits_count_l68_68328

-- Definitions of various clothing counts
def numRedShirts : ℕ := 7
def numGreenShirts : ℕ := 3
def numPants : ℕ := 8
def numBlueShoes : ℕ := 5
def numRedShoes : ℕ := 5
def numGreenHats : ℕ := 10
def numRedHats : ℕ := 6

-- Statement of the theorem based on the problem description
theorem outfits_count :
  (numRedShirts * numPants * numBlueShoes * numGreenHats) + 
  (numGreenShirts * numPants * (numBlueShoes + numRedShoes) * numRedHats) = 4240 := 
by
  -- No proof required, only the statement is needed
  sorry

end outfits_count_l68_68328


namespace volume_units_correct_l68_68793

/-- Definition for the volume of a bottle of coconut juice in milliliters (200 milliliters). -/
def volume_of_coconut_juice := 200 

/-- Definition for the volume of an electric water heater in liters (50 liters). -/
def volume_of_electric_water_heater := 50 

/-- Prove that the volume of a bottle of coconut juice is measured in milliliters (200 milliliters)
    and the volume of an electric water heater is measured in liters (50 liters).
-/
theorem volume_units_correct :
  volume_of_coconut_juice = 200 ∧ volume_of_electric_water_heater = 50 :=
sorry

end volume_units_correct_l68_68793


namespace contrapositive_example_l68_68465

theorem contrapositive_example (x : ℝ) (h : x = 1 → x^2 - 3 * x + 2 = 0) :
  x^2 - 3 * x + 2 ≠ 0 → x ≠ 1 :=
by
  intro h₀
  intro h₁
  have h₂ := h h₁
  contradiction

end contrapositive_example_l68_68465


namespace tan_triple_angle_l68_68947

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end tan_triple_angle_l68_68947


namespace branches_sum_one_main_stem_l68_68378

theorem branches_sum_one_main_stem (x : ℕ) (h : 1 + x + x^2 = 31) : x = 5 :=
by {
  sorry
}

end branches_sum_one_main_stem_l68_68378


namespace xyz_zero_unique_solution_l68_68128

theorem xyz_zero_unique_solution {x y z : ℝ} (h1 : x^2 * y + y^2 * z + z^2 = 0)
                                 (h2 : z^3 + z^2 * y + z * y^3 + x^2 * y = 1 / 4 * (x^4 + y^4)) :
  x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end xyz_zero_unique_solution_l68_68128


namespace g_difference_l68_68439

variable (g : ℝ → ℝ)

-- Condition: g is a linear function
axiom linear_g : ∃ a b : ℝ, ∀ x : ℝ, g x = a * x + b

-- Condition: g(10) - g(4) = 18
axiom g_condition : g 10 - g 4 = 18

theorem g_difference : g 16 - g 4 = 36 :=
by
  sorry

end g_difference_l68_68439


namespace largest_fraction_among_fractions_l68_68909

theorem largest_fraction_among_fractions :
  let A := (2 : ℚ) / 5
  let B := (3 : ℚ) / 7
  let C := (4 : ℚ) / 9
  let D := (3 : ℚ) / 8
  let E := (9 : ℚ) / 20
  (A < E) ∧ (B < E) ∧ (C < E) ∧ (D < E) :=
by
  let A := (2 : ℚ) / 5
  let B := (3 : ℚ) / 7
  let C := (4 : ℚ) / 9
  let D := (3 : ℚ) / 8
  let E := (9 : ℚ) / 20
  sorry

end largest_fraction_among_fractions_l68_68909


namespace slope_angle_correct_l68_68072

def parametric_line (α : ℝ) : Prop :=
  α = 50 * (Real.pi / 180)

theorem slope_angle_correct : ∀ (t : ℝ),
  parametric_line 50 →
  ∀ α : ℝ, α = 140 * (Real.pi / 180) :=
by
  intro t
  intro h
  intro α
  sorry

end slope_angle_correct_l68_68072


namespace A_completes_work_in_18_days_l68_68628

-- Define the conditions
def efficiency_A_twice_B (A B : ℕ → ℕ) : Prop := ∀ w, A w = 2 * B w
def same_work_time (A B C D : ℕ → ℕ) : Prop := 
  ∀ w t, A w + B w = C w + D w ∧ C t = 1 / 20 ∧ D t = 1 / 30

-- Define the key quantity to be proven
theorem A_completes_work_in_18_days (A B C D : ℕ → ℕ) 
  (h1 : efficiency_A_twice_B A B) 
  (h2 : same_work_time A B C D) : 
  ∀ w, A w = 1 / 18 :=
sorry

end A_completes_work_in_18_days_l68_68628


namespace least_possible_product_of_primes_l68_68078

-- Define a prime predicate for a number greater than 20
def is_prime_over_20 (p : Nat) : Prop := Nat.Prime p ∧ p > 20

-- Define the two primes
def prime1 := 23
def prime2 := 29

-- Given the conditions, prove the least possible product of two distinct primes greater than 20 is 667
theorem least_possible_product_of_primes :
  ∃ p1 p2 : Nat, is_prime_over_20 p1 ∧ is_prime_over_20 p2 ∧ p1 ≠ p2 ∧ (p1 * p2 = 667) :=
by
  -- Theorem statement without proof
  existsi (prime1)
  existsi (prime2)
  have h1 : is_prime_over_20 prime1 := by sorry
  have h2 : is_prime_over_20 prime2 := by sorry
  have h3 : prime1 ≠ prime2 := by sorry
  have h4 : prime1 * prime2 = 667 := by sorry
  exact ⟨h1, h2, h3, h4⟩

end least_possible_product_of_primes_l68_68078


namespace inequality_proof_l68_68155

theorem inequality_proof (a b : ℝ) (h₀ : a > b) (h₁ : b > 0) :
  (a^2 - b^2) / (a^2 + b^2) > (a - b) / (a + b) :=
by 
  sorry

end inequality_proof_l68_68155


namespace branches_on_one_stem_l68_68376

theorem branches_on_one_stem (x : ℕ) (h : 1 + x + x^2 = 31) : x = 5 :=
by {
  sorry
}

end branches_on_one_stem_l68_68376


namespace rhombus_perimeter_l68_68063

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) : 
  ∃ p : ℝ, p = 16 * Real.sqrt 13 := by
sorry

end rhombus_perimeter_l68_68063


namespace least_five_digit_perfect_square_and_cube_l68_68293

theorem least_five_digit_perfect_square_and_cube : 
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ (∃ a : ℕ, n = a ^ 6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68293


namespace xyz_square_sum_l68_68822

theorem xyz_square_sum {x y z a b c d : ℝ} (h1 : x * y = a) (h2 : x * z = b) (h3 : y * z = c) (h4 : x + y + z = d) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0):
  x^2 + y^2 + z^2 = d^2 - 2 * (a + b + c) :=
sorry

end xyz_square_sum_l68_68822


namespace product_of_solutions_abs_eq_l68_68182

theorem product_of_solutions_abs_eq (x : ℝ) (h : |x - 5| + 4 = 7) : x * (if x = 8 then 2 else 8) = 16 :=
by {
  sorry
}

end product_of_solutions_abs_eq_l68_68182


namespace sin_and_tan_inequality_l68_68453

theorem sin_and_tan_inequality (n : ℕ) (hn : 0 < n) :
  2 * Real.sin (1 / n) + Real.tan (1 / n) > 3 / n :=
sorry

end sin_and_tan_inequality_l68_68453


namespace courses_chosen_by_students_l68_68830

theorem courses_chosen_by_students :
  let num_courses := 4
  let num_students := 3
  let choices_per_student := 2
  let total_ways := (Nat.choose num_courses choices_per_student) ^ num_students
  let cases_two_courses_not_chosen := Nat.choose num_courses 2
  let cases_one_course_not_chosen := 
    num_courses * ((Nat.choose (num_courses - 1) choices_per_student) ^ num_students - Nat.choose (num_courses - 1) (choices_per_student - 1) ^ num_students)
  total_ways - cases_two_courses_not_chosen - cases_one_course_not_chosen = 114 := by
  sorry

end courses_chosen_by_students_l68_68830


namespace total_pencils_correct_l68_68358

-- Define the number of pencils Reeta has
def ReetaPencils : ℕ := 20

-- Define the number of pencils Anika has based on the conditions
def AnikaPencils : ℕ := 2 * ReetaPencils + 4

-- Define the total number of pencils Anika and Reeta have together
def TotalPencils : ℕ := ReetaPencils + AnikaPencils

-- Statement to prove
theorem total_pencils_correct : TotalPencils = 64 :=
by
  sorry

end total_pencils_correct_l68_68358


namespace chess_tournament_scores_l68_68139

def points (name : String) := Real

def total_points : Real := 12

variables (A D V S : Real)
variable (total_games : ℕ := 12)

axiom different_scores : A ≠ D ∧ A ≠ V ∧ A ≠ S ∧ D ≠ V ∧ D ≠ S ∧ V ≠ S

axiom ranking : A > D ∧ D > V ∧ V > S

axiom equal_wins (A S : Real) : (A = 2 * win_points) ∧ (S = 2 * win_points)

axiom total_points_constraint : A + D + V + S = total_points

theorem chess_tournament_scores :
  A = 4 ∧ D = 3.5 ∧ V = 2.5 ∧ S = 2 :=
by 
  sorry

end chess_tournament_scores_l68_68139


namespace find_m_l68_68925

def is_ellipse (x y m : ℝ) : Prop :=
  (x^2 / (m + 1) + y^2 / m = 1)

def has_eccentricity (e : ℝ) (m : ℝ) : Prop :=
  e = Real.sqrt (1 - m / (m + 1))

theorem find_m (m : ℝ) (h_m : m > 0) (h_ellipse : ∀ x y, is_ellipse x y m) (h_eccentricity : has_eccentricity (1 / 2) m) : m = 3 :=
by
  sorry

end find_m_l68_68925


namespace quadratic_graph_above_x_axis_l68_68637

theorem quadratic_graph_above_x_axis (a b c : ℝ) :
  ¬ ((b^2 - 4*a*c < 0) ↔ ∀ x : ℝ, a*x^2 + b*x + c > 0) :=
sorry

end quadratic_graph_above_x_axis_l68_68637


namespace cost_per_minute_l68_68035

-- Conditions as Lean definitions
def initial_credit : ℝ := 30
def remaining_credit : ℝ := 26.48
def call_duration : ℝ := 22

-- Question: How much does a long distance call cost per minute?

theorem cost_per_minute :
  (initial_credit - remaining_credit) / call_duration = 0.16 := 
by
  sorry

end cost_per_minute_l68_68035


namespace trapezoid_height_l68_68225

theorem trapezoid_height (BC AD AB CD h : ℝ) (hBC : BC = 4) (hAD : AD = 25) (hAB : AB = 20) (hCD : CD = 13) :
  h = 12 :=
by
  sorry

end trapezoid_height_l68_68225


namespace michael_truck_meet_once_l68_68595

-- Michael's walking speed.
def michael_speed := 4 -- feet per second

-- Distance between trash pails.
def pail_distance := 100 -- feet

-- Truck's speed.
def truck_speed := 8 -- feet per second

-- Time truck stops at each pail.
def truck_stop_time := 20 -- seconds

-- Prove how many times Michael and the truck will meet given the initial condition.
theorem michael_truck_meet_once :
  ∃ n : ℕ, michael_truck_meet_count == 1 :=
sorry

end michael_truck_meet_once_l68_68595


namespace total_price_of_basic_computer_and_printer_l68_68866

-- Definitions for the conditions
def basic_computer_price := 2000
def enhanced_computer_price (C : ℕ) := C + 500
def printer_price (C : ℕ) (P : ℕ) := 1/6 * (C + 500 + P)

-- The proof problem statement
theorem total_price_of_basic_computer_and_printer (C P : ℕ) 
  (h1 : C = 2000)
  (h2 : printer_price C P = P) : 
  C + P = 2500 :=
sorry

end total_price_of_basic_computer_and_printer_l68_68866


namespace max_perfect_squares_sequence_l68_68674

def seq (a₀ : ℕ) : ℕ → ℕ
| 0     => a₀
| (n+1) => (seq n) ^ 5 + 487

theorem max_perfect_squares_sequence (m : ℕ) : m = 9 → 
(∀ n : ℕ, (∃ k : ℕ, seq m n = k^2) → n ≤ 1) := 
sorry

end max_perfect_squares_sequence_l68_68674


namespace solve_equation_real_l68_68456

theorem solve_equation_real (x : ℝ) (h : (x ^ 2 - x + 1) * (3 * x ^ 2 - 10 * x + 3) = 20 * x ^ 2) :
    x = (5 + Real.sqrt 21) / 2 ∨ x = (5 - Real.sqrt 21) / 2 :=
by
  sorry

end solve_equation_real_l68_68456


namespace least_five_digit_perfect_square_and_cube_l68_68314

/-- 
Prove that the least five-digit whole number that is both a perfect square 
and a perfect cube is 15625.
-/
theorem least_five_digit_perfect_square_and_cube : 
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k : ℕ, n = k^6) ∧ (∀ m : ℕ, (10000 ≤ m ∧ m < 100000) ∧ (∃ l : ℕ, m = l^6) → n ≤ m) :=
begin
  use 15625,
  split,
  { split,
    { norm_num },
    { norm_num } },
  { use 5,
    norm_num,
    intros m hm,
    rcases hm with ⟨⟨hm1, hm2⟩, ⟨l, rfl⟩⟩,
    cases le_or_lt l 5 with hl hl,
    { by_contradiction h,
      have : m < 10000 := by norm_num [nat.pow_succ, nat.pow_succ, hl],
      exact not_le_of_gt this hm1 },
    { have : l ≥ 6 := nat.lt_succ_iff.mp hl,
      have : m ≥ 6^6 := nat.pow_le_pow_of_le_right nat.zero_lt_succ_iff.1 this,
      norm_num at this,
      exact not_lt_of_ge this hm2 } }
end

end least_five_digit_perfect_square_and_cube_l68_68314


namespace correctOptionOnlyC_l68_68485

-- Definitions for the transformations
def isTransformA (a b : ℝ) : Prop := (a ≠ 0) → (b ≠ 0) → (b / a = (b^2) / (a^2)) 
def isTransformB (a b : ℝ) : Prop := (a ≠ 0) → (b ≠ 0) → (b / a = (b + 1) / (a + 1))
def isTransformC (a b : ℝ) : Prop := (a ≠ 0) → (b / a = (a * b) / (a^2))
def isTransformD (a b : ℝ) : Prop := (a ≠ 0) → ((-b + 1) / a = -(b + 1) / a)

-- Main theorem to assert the correctness of the transformations
theorem correctOptionOnlyC (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) : 
  ¬isTransformA a b ∧ ¬isTransformB a b ∧ isTransformC a b ∧ ¬isTransformD a b :=
by
  sorry

end correctOptionOnlyC_l68_68485


namespace numerator_multiple_of_prime_l68_68240

theorem numerator_multiple_of_prime (n : ℕ) (hp : Nat.Prime (3 * n + 1)) :
  (2 * n - 1) % (3 * n + 1) = 0 :=
sorry

end numerator_multiple_of_prime_l68_68240


namespace least_five_digit_perfect_square_and_cube_l68_68295

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (n >= 10000 ∧ n < 100000) ∧ (∃ a : ℕ, n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68295


namespace buckets_needed_l68_68474

variable {C : ℝ} (hC : C > 0)

theorem buckets_needed (h : 42 * C = 42 * C) : 
  (42 * C) / ((2 / 5) * C) = 105 :=
by
  sorry

end buckets_needed_l68_68474


namespace num_congruent_mod_7_count_mod_7_eq_22_l68_68168

theorem num_congruent_mod_7 (n : ℕ) :
  (1 ≤ n ∧ n ≤ 150 ∧ n % 7 = 1) → ∃ k, 0 ≤ k ∧ k ≤ 21 ∧ n = 7 * k + 1 :=
sorry

theorem count_mod_7_eq_22 : 
  (∃ n_set : Finset ℕ, 
    (∀ n ∈ n_set, 1 ≤ n ∧ n ≤ 150 ∧ n % 7 = 1) ∧ 
    Finset.card n_set = 22) :=
sorry

end num_congruent_mod_7_count_mod_7_eq_22_l68_68168


namespace bus_passengers_l68_68460

def passengers_after_first_stop := 7

def passengers_after_second_stop := passengers_after_first_stop - 3 + 5

def passengers_after_third_stop := passengers_after_second_stop - 2 + 4

theorem bus_passengers (passengers_after_first_stop passengers_after_second_stop passengers_after_third_stop : ℕ) : passengers_after_third_stop = 11 :=
by
  sorry

end bus_passengers_l68_68460


namespace true_discount_different_time_l68_68016

theorem true_discount_different_time (FV TD_initial TD_different : ℝ) (r : ℝ) (initial_time different_time : ℝ) 
  (h1 : r = initial_time / different_time)
  (h2 : FV = 110)
  (h3 : TD_initial = 10)
  (h4 : initial_time / different_time = 1 / 2) :
  TD_different = 2 * TD_initial :=
by
  sorry

end true_discount_different_time_l68_68016


namespace mila_social_media_time_week_l68_68381

theorem mila_social_media_time_week
  (hours_per_day_on_phone : ℕ)
  (half_on_social_media : ℕ)
  (days_in_week : ℕ)
  (h1 : hours_per_day_on_phone = 6)
  (h2 : half_on_social_media = hours_per_day_on_phone / 2)
  (h3 : days_in_week = 7) : 
  half_on_social_media * days_in_week = 21 := 
by
  rw [h2, h3]
  norm_num
  exact h1.symm ▸ rfl

end mila_social_media_time_week_l68_68381


namespace sequence_b_n_l68_68516

theorem sequence_b_n (b : ℕ → ℝ) 
  (h1 : b 1 = 3)
  (h2 : ∀ n ≥ 1, (b (n + 1))^3 = 27 * (b n)^3) :
  b 50 = 3^50 :=
sorry

end sequence_b_n_l68_68516


namespace chess_tournament_points_distribution_l68_68132

noncomputable def points_distribution (Andrey Dima Vanya Sasha : ℝ) : Prop :=
  ∃ (p_a p_d p_v p_s : ℝ), 
    p_a ≠ p_d ∧ p_d ≠ p_v ∧ p_v ≠ p_s ∧ p_a ≠ p_v ∧ p_a ≠ p_s ∧ p_d ≠ p_s ∧
    p_a + p_d + p_v + p_s = 12 ∧ -- Total points sum
    p_a > p_d ∧ p_d > p_v ∧ p_v > p_s ∧ -- Order of points
    Andrey = p_a ∧ Dima = p_d ∧ Vanya = p_v ∧ Sasha = p_s ∧
    Andrey - (Sasha - 2) = 2 -- Andrey and Sasha won the same number of games

theorem chess_tournament_points_distribution :
  points_distribution 4 3.5 2.5 2 :=
sorry

end chess_tournament_points_distribution_l68_68132


namespace positive_difference_is_correct_l68_68899

/-- Angela's compounded interest parameters -/
def angela_initial_deposit : ℝ := 9000
def angela_interest_rate : ℝ := 0.08
def years : ℕ := 25

/-- Bob's simple interest parameters -/
def bob_initial_deposit : ℝ := 11000
def bob_interest_rate : ℝ := 0.09

/-- Compound interest calculation for Angela -/
def angela_balance : ℝ := angela_initial_deposit * (1 + angela_interest_rate) ^ years

/-- Simple interest calculation for Bob -/
def bob_balance : ℝ := bob_initial_deposit * (1 + bob_interest_rate * years)

/-- Difference calculation -/
def balance_difference : ℝ := angela_balance - bob_balance

/-- The positive difference between their balances to the nearest dollar -/
theorem positive_difference_is_correct :
  abs (round balance_difference) = 25890 :=
by
  sorry

end positive_difference_is_correct_l68_68899


namespace percentage_of_men_l68_68967

variable (M : ℝ)

theorem percentage_of_men (h1 : 0.20 * M + 0.40 * (1 - M) = 0.33) : 
  M = 0.35 :=
sorry

end percentage_of_men_l68_68967


namespace prove_AF_eq_l68_68179

-- Definitions
variables {A B C E F : Type*}
variables [Field A] [Field B] [Field C] [Field E] [Field F]

-- Conditions
def triangle_ABC (AB AC : ℝ) (h : AB > AC) : Prop := true

def external_bisector (angleA : ℝ) (circumcircle_meets : ℝ) : Prop := true

def foot_perpendicular (E AB : ℝ) : Prop := true

-- Theorem statement
theorem prove_AF_eq (AB AC AF : ℝ) (h_triangle : triangle_ABC AB AC (by sorry))
  (h_external_bisector : external_bisector (by sorry) (by sorry))
  (h_foot_perpendicular : foot_perpendicular (by sorry) AB) :
  2 * AF = AB - AC := by
  sorry

end prove_AF_eq_l68_68179


namespace arithmetic_sequence_a5_eq_6_l68_68153

variable {a_n : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_a5_eq_6 (h_arith : is_arithmetic_sequence a_n) (h_sum : a_n 2 + a_n 8 = 12) : a_n 5 = 6 :=
by
  sorry

end arithmetic_sequence_a5_eq_6_l68_68153


namespace c_a_plus_c_b_geq_a_a_plus_b_b_l68_68931

theorem c_a_plus_c_b_geq_a_a_plus_b_b (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  (c : ℚ) (h : c = (a^(a+1) + b^(b+1)) / (a^a + b^b)) :
  c^a + c^b ≥ a^a + b^b :=
sorry

end c_a_plus_c_b_geq_a_a_plus_b_b_l68_68931


namespace hexagon_circle_radius_l68_68499

theorem hexagon_circle_radius (r : ℝ) :
  let side_length := 3
  let probability := (1 : ℝ) / 3
  (probability = 1 / 3) →
  r = 12 * Real.sqrt 3 / (Real.sqrt 6 - Real.sqrt 2) :=
by
  -- Begin proof here
  sorry

end hexagon_circle_radius_l68_68499


namespace fraction_addition_l68_68781

theorem fraction_addition : (3 / 4 : ℚ) + (5 / 6) = 19 / 12 :=
by
  sorry

end fraction_addition_l68_68781


namespace bowling_ball_weight_l68_68130

theorem bowling_ball_weight (b k : ℝ) (h1 : 5 * b = 3 * k) (h2 : 4 * k = 120) : b = 18 :=
by
  sorry

end bowling_ball_weight_l68_68130


namespace least_five_digit_perfect_square_and_cube_l68_68292

theorem least_five_digit_perfect_square_and_cube : 
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ (∃ a : ℕ, n = a ^ 6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68292


namespace graph_passes_through_point_l68_68606

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 1) + 3

theorem graph_passes_through_point (a : ℝ) : f a 1 = 4 := by
  sorry

end graph_passes_through_point_l68_68606


namespace least_five_digit_perfect_square_and_cube_l68_68325

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k1 : ℕ, n = k1 ^ 2) ∧ (∃ k2 : ℕ, n = k2 ^ 3) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68325


namespace purple_tile_cost_correct_l68_68187

-- Definitions of given conditions
def turquoise_cost_per_tile : ℕ := 13
def wall1_area : ℕ := 5 * 8
def wall2_area : ℕ := 7 * 8
def total_area : ℕ := wall1_area + wall2_area
def tiles_per_square_foot : ℕ := 4
def total_tiles_needed : ℕ := total_area * tiles_per_square_foot
def turquoise_total_cost : ℕ := total_tiles_needed * turquoise_cost_per_tile
def savings : ℕ := 768
def purple_total_cost : ℕ := turquoise_total_cost - savings
def purple_cost_per_tile : ℕ := 11

-- Theorem stating the problem
theorem purple_tile_cost_correct :
  purple_total_cost / total_tiles_needed = purple_cost_per_tile :=
sorry

end purple_tile_cost_correct_l68_68187


namespace shortest_distance_to_y_axis_is_3_l68_68247

-- Define the parabola and the fixed length of the line segment
def parabola (x y : ℝ) := y^2 = 8 * x
def fixed_length (A B : ℝ × ℝ) := (A.1 - B.1)^2 + (A.2 - B.2)^2 = 100

-- Define the midpoint of AB
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the shortest distance from P to the y-axis
def mid_distance_to_y_axis (P : ℝ × ℝ) : ℝ := abs P.1

-- The theorem statement
theorem shortest_distance_to_y_axis_is_3 (A B P : ℝ × ℝ) 
  (hA : parabola A.1 A.2) 
  (hB : parabola B.1 B.2) 
  (hAB : fixed_length A B) 
  (hP : P = midpoint A B) :
  mid_distance_to_y_axis P = 3 :=
sorry

end shortest_distance_to_y_axis_is_3_l68_68247


namespace arithmetic_seq_sum_div_fifth_term_l68_68740

open Int

/-- The sequence {a_n} is an arithmetic sequence with a non-zero common difference,
    given that a₂ + a₆ = a₈, prove that S₅ / a₅ = 3. -/
theorem arithmetic_seq_sum_div_fifth_term
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_nonzero : d ≠ 0)
  (h_condition : a 2 + a 6 = a 8) :
  ((5 * a 1 + 10 * d) / (a 1 + 4 * d) : ℚ) = 3 := 
by
  sorry

end arithmetic_seq_sum_div_fifth_term_l68_68740


namespace felipe_building_time_l68_68527

theorem felipe_building_time
  (F E : ℕ)
  (combined_time_without_breaks : ℕ)
  (felipe_time_fraction : F = E / 2)
  (combined_time_condition : F + E = 90)
  (felipe_break : ℕ)
  (emilio_break : ℕ)
  (felipe_break_is_6_months : felipe_break = 6)
  (emilio_break_is_double_felipe : emilio_break = 2 * felipe_break) :
  F + felipe_break = 36 := by
  sorry

end felipe_building_time_l68_68527


namespace coin_flips_probability_l68_68574

section 

-- Definition for the probability of heads in a single flip
def prob_heads : ℚ := 1 / 2

-- Definition for flipping the coin 5 times and getting heads on the first 4 flips and tails on the last flip
def prob_specific_sequence (n : ℕ) (k : ℕ) : ℚ := (prob_heads) ^ k * (prob_heads) ^ (n - k)

-- The main theorem which states the probability of the desired outcome
theorem coin_flips_probability : 
  prob_specific_sequence 5 4 = 1 / 32 :=
sorry

end

end coin_flips_probability_l68_68574


namespace a_squared_plus_b_squared_eq_zero_implies_a_eq_zero_and_b_eq_zero_l68_68452

-- Mathematical condition: a^2 + b^2 = 0
variable {a b : ℝ}

-- Mathematical statement to be proven
theorem a_squared_plus_b_squared_eq_zero_implies_a_eq_zero_and_b_eq_zero 
  (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
sorry  -- proof yet to be provided

end a_squared_plus_b_squared_eq_zero_implies_a_eq_zero_and_b_eq_zero_l68_68452


namespace triangle_construction_l68_68352

-- Given: A triangle ABC where the lengths of the sides AB and AC are known
variables (A B C : Point)
variables (c b : ℝ) [Fact (0 < c)] [Fact (0 < b)]
variable (triangle_ABC : Triangle A B C)
variable (hAB : dist A B = c)
variable (hAC : dist A C = b)

-- To prove: The internal and external angle bisectors of ∠BAC are equal
theorem triangle_construction
    (h_bisectors_equal : internal_and_external_angle_bisectors_equal A B C) :
    internal_angle_bisector_length A B C = external_angle_bisector_length A B C := by
  sorry

end triangle_construction_l68_68352


namespace normal_distribution_prob_l68_68160

open MeasureTheory

noncomputable def normal_cdf : ℝ → ℝ := sorry -- Placeholder for the standard normal CDF

variable {σ : ℝ} (σ_pos : 0 < σ)

theorem normal_distribution_prob (h : normal_cdf (1 / σ) = 2 / 3) :
  normal_cdf (1 * σ⁻¹) = 2 / 3 →
  (Π (σ > 0), normal_cdf (-1 / σ) = 1 / 3) :=
sorry

end normal_distribution_prob_l68_68160


namespace interest_rate_second_part_l68_68851

theorem interest_rate_second_part 
    (total_investment : ℝ) 
    (annual_interest : ℝ) 
    (P1 : ℝ) 
    (rate1 : ℝ) 
    (P2 : ℝ)
    (rate2 : ℝ) : 
    total_investment = 3600 → 
    annual_interest = 144 → 
    P1 = 1800 → 
    rate1 = 3 → 
    P2 = total_investment - P1 → 
    (annual_interest - (P1 * rate1 / 100)) = (P2 * rate2 / 100) →
    rate2 = 5 :=
by 
  intros total_investment_eq annual_interest_eq P1_eq rate1_eq P2_eq interest_eq
  sorry

end interest_rate_second_part_l68_68851


namespace rahul_salary_l68_68920

variable (X : ℝ)

def house_rent_deduction (salary : ℝ) : ℝ := salary * 0.8
def education_expense (remaining_after_rent : ℝ) : ℝ := remaining_after_rent * 0.9
def clothing_expense (remaining_after_education : ℝ) : ℝ := remaining_after_education * 0.9

theorem rahul_salary : (X * 0.8 * 0.9 * 0.9 = 1377) → X = 2125 :=
by
  intros h
  sorry

end rahul_salary_l68_68920


namespace cyclist_speed_ratio_l68_68620

-- Define the conditions
def speeds_towards_each_other (v1 v2 : ℚ) : Prop :=
  v1 + v2 = 25

def speeds_apart_with_offset (v1 v2 : ℚ) : Prop :=
  v1 - v2 = 10 / 3

-- The proof problem to show the required ratio of speeds
theorem cyclist_speed_ratio (v1 v2 : ℚ) (h1 : speeds_towards_each_other v1 v2) (h2 : speeds_apart_with_offset v1 v2) :
  v1 / v2 = 17 / 13 :=
sorry

end cyclist_speed_ratio_l68_68620


namespace increasing_interval_f_l68_68557

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - (Real.pi / 6))

theorem increasing_interval_f : ∃ a b : ℝ, a < b ∧ 
  (∀ x y : ℝ, a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b ∧ x < y → f x < f y) ∧
  (a = - (Real.pi / 6)) ∧ (b = (Real.pi / 3)) :=
by
  sorry

end increasing_interval_f_l68_68557


namespace linear_function_quadrant_l68_68583

theorem linear_function_quadrant (x y : ℝ) : 
  y = 2 * x - 3 → ¬ ((x < 0 ∧ y > 0)) := 
sorry

end linear_function_quadrant_l68_68583


namespace line_parabola_intersection_one_point_l68_68237

theorem line_parabola_intersection_one_point (k : ℝ) :
  (∃ y : ℝ, (-3 * y^2 - 4 * y + 7 = k) ∧ ∀ y1 y2 : ℝ, ( 3 * y1^2 + 4 * y1 + (k - 7) = 0 → 3 * y2^2 + 4 * y2 + (k - 7) = 0 → y1 = y2)) ↔ (k = 25 / 3) :=
by
  sorry

end line_parabola_intersection_one_point_l68_68237


namespace quadratic_neq_l68_68694

theorem quadratic_neq (m : ℝ) : (m-2) ≠ 0 ↔ m ≠ 2 :=
sorry

end quadratic_neq_l68_68694


namespace inverse_of_matrix_l68_68385

open Matrix

def mat : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![4, 9], ![2, 5]]

def inv_mat : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![5/2, -9/2], ![-1, 2]]

theorem inverse_of_matrix :
  ∃ (inv : Matrix (Fin 2) (Fin 2) ℚ), 
    inv * mat = 1 ∧ mat * inv = 1 :=
  ⟨inv_mat, by
    -- Providing the proof steps here is beyond the scope
    sorry⟩

end inverse_of_matrix_l68_68385


namespace line_through_point_bisected_by_hyperbola_l68_68796

theorem line_through_point_bisected_by_hyperbola :
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (a * 3 + b * (-1) + c = 0) ∧
  (∀ x y : ℝ, (x^2 / 4 - y^2 = 1) → (a * x + b * y + c = 0)) ↔ (a = 3 ∧ b = 4 ∧ c = -5) :=
by
  sorry

end line_through_point_bisected_by_hyperbola_l68_68796


namespace triangle_length_product_square_l68_68454

theorem triangle_length_product_square 
  (a1 : ℝ) (b1 : ℝ) (c1 : ℝ) (a2 : ℝ) (b2 : ℝ) (c2 : ℝ) 
  (h1 : a1 * b1 / 2 = 3)
  (h2 : a2 * b2 / 2 = 4)
  (h3 : a1 = a2)
  (h4 : c1 = 2 * c2) 
  (h5 : c1^2 = a1^2 + b1^2)
  (h6 : c2^2 = a2^2 + b2^2) :
  (b1 * b2)^2 = (2304 / 25 : ℝ) :=
by
  sorry

end triangle_length_product_square_l68_68454


namespace largest_real_number_l68_68390

theorem largest_real_number (x : ℝ) (h : (⌊x⌋ : ℝ) / x = 8 / 9) : x = 63 / 8 := sorry

end largest_real_number_l68_68390


namespace slopes_hyperbola_l68_68413

theorem slopes_hyperbola 
  (x y : ℝ)
  (M : ℝ × ℝ) 
  (t m : ℝ) 
  (h_point_M_on_line: M = (9 / 5, t))
  (h_hyperbola : ∀ t: ℝ, (16 * m^2 - 9) * t^2 + 160 * m * t + 256 = 0)
  (k1 k2 k3 : ℝ)
  (h_k2 : k2 = -5 * t / 16) :
  k1 + k3 = 2 * k2 :=
sorry

end slopes_hyperbola_l68_68413


namespace vasya_max_consecutive_liked_numbers_l68_68871

def is_liked_by_vasya (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), d ≠ 0 → n % d = 0

theorem vasya_max_consecutive_liked_numbers : 
  ∃ (seq : ℕ → ℕ), 
    (∀ n, seq n = n ∧ is_liked_by_vasya (seq n)) ∧
    (∀ m, seq m + 1 < seq (m + 1)) ∧ seq 12 - seq 0 + 1 = 13 :=
sorry

end vasya_max_consecutive_liked_numbers_l68_68871


namespace A_can_finish_remaining_work_in_4_days_l68_68641

theorem A_can_finish_remaining_work_in_4_days
  (A_days : ℕ) (B_days : ℕ) (B_worked_days : ℕ) : 
  A_days = 12 → B_days = 15 → B_worked_days = 10 → 
  (4 * (1 / A_days) = 1 / 3 - B_worked_days * (1 / B_days)) :=
by
  intros hA hB hBwork
  sorry

end A_can_finish_remaining_work_in_4_days_l68_68641


namespace Dan_running_speed_is_10_l68_68477

noncomputable def running_speed
  (d : ℕ)
  (S : ℕ)
  (avg : ℚ) : ℚ :=
  let total_distance := 2 * d
  let total_time := d / (avg * 60) 
  let swim_time := d / S
  let run_time := total_time - swim_time
  total_distance / run_time

theorem Dan_running_speed_is_10
  (d S : ℕ)
  (avg : ℚ)
  (h1 : d = 4)
  (h2 : S = 6)
  (h3 : avg = 0.125) :
  running_speed d S (avg * 60) = 10 := by 
  sorry

end Dan_running_speed_is_10_l68_68477


namespace f_at_7_l68_68438

noncomputable def f (x : ℝ) (a b c d : ℝ) := a * x^7 + b * x^5 + c * x^3 + d * x + 5

theorem f_at_7 (a b c d : ℝ) (h : f (-7) a b c d = -7) : f 7 a b c d = 17 := 
by
  sorry

end f_at_7_l68_68438


namespace cost_price_equals_720_l68_68823

theorem cost_price_equals_720 (C : ℝ) :
  (0.27 * C - 0.12 * C = 108) → (C = 720) :=
by
  sorry

end cost_price_equals_720_l68_68823


namespace star_calculation_l68_68116

-- Define the operation '*' via the given table
def star_table : Matrix (Fin 5) (Fin 5) (Fin 5) :=
  ![
    ![0, 1, 2, 3, 4],
    ![1, 0, 4, 2, 3],
    ![2, 3, 1, 4, 0],
    ![3, 4, 0, 1, 2],
    ![4, 2, 3, 0, 1]
  ]

def star (a b : Fin 5) : Fin 5 := star_table a b

-- Prove (3 * 5) * (2 * 4) = 3
theorem star_calculation : star (star 2 4) (star 4 1) = 2 := by
  sorry

end star_calculation_l68_68116


namespace lines_perpendicular_l68_68728

-- Define the conditions: lines not parallel to the coordinate planes 
-- (which translates to k_1 and k_2 not being infinite, but we can code it directly as a statement on the product being -1)
variable {k1 k2 l1 l2 : ℝ} 

-- Define the theorem statement 
theorem lines_perpendicular (hk : k1 * k2 = -1) : 
  ∀ (x : ℝ), (k1 ≠ 0) ∧ (k2 ≠ 0) → 
  (∀ (y1 y2 : ℝ), y1 = k1 * x + l1 → y2 = k2 * x + l2 → 
  (k1 * k2 = -1)) :=
sorry

end lines_perpendicular_l68_68728


namespace subtraction_of_decimals_l68_68252

theorem subtraction_of_decimals : 58.3 - 0.45 = 57.85 := by
  sorry

end subtraction_of_decimals_l68_68252


namespace range_of_a_in_third_quadrant_l68_68180

def pointInThirdQuadrant (x y : ℝ) := x < 0 ∧ y < 0

theorem range_of_a_in_third_quadrant (a : ℝ) (M : ℝ × ℝ) 
  (hM : M = (-1, a-1)) (hThirdQuad : pointInThirdQuadrant M.1 M.2) : 
  a < 1 :=
by
  sorry

end range_of_a_in_third_quadrant_l68_68180


namespace part1_part2_l68_68838

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- Part (1): Given m = 4, prove A ∪ B = {x | -2 ≤ x ∧ x ≤ 7}
theorem part1 : A ∪ B 4 = {x | -2 ≤ x ∧ x ≤ 7} :=
by
  sorry

-- Part (2): Given B ⊆ A, prove m ∈ (-∞, 3]
theorem part2 {m : ℝ} (h : B m ⊆ A) : m ∈ Set.Iic 3 :=
by
  sorry

end part1_part2_l68_68838


namespace num_children_got_off_l68_68494

-- Define the original number of children on the bus
def original_children : ℕ := 43

-- Define the number of children left after some got off the bus
def children_left : ℕ := 21

-- Define the number of children who got off the bus as the difference between original_children and children_left
def children_got_off : ℕ := original_children - children_left

-- State the theorem that the number of children who got off the bus is 22
theorem num_children_got_off : children_got_off = 22 :=
by
  -- Proof steps would go here, but are omitted
  sorry

end num_children_got_off_l68_68494


namespace transistors_in_2002_transistors_in_2010_l68_68653

-- Definitions based on the conditions
def mooresLawDoubling (initial_transistors : ℕ) (years : ℕ) : ℕ :=
  initial_transistors * 2^(years / 2)

-- Conditions
def initial_transistors := 2000000
def year_1992 := 1992
def year_2002 := 2002
def year_2010 := 2010

-- Questions translated into proof targets
theorem transistors_in_2002 : mooresLawDoubling initial_transistors (year_2002 - year_1992) = 64000000 := by
  sorry

theorem transistors_in_2010 : mooresLawDoubling (mooresLawDoubling initial_transistors (year_2002 - year_1992)) (year_2010 - year_2002) = 1024000000 := by
  sorry

end transistors_in_2002_transistors_in_2010_l68_68653


namespace lever_equilibrium_min_force_l68_68098

noncomputable def lever_minimum_force (F L : ℝ) : Prop :=
  (F * L = 49 + 2 * (L^2))

theorem lever_equilibrium_min_force : ∃ F : ℝ, ∃ L : ℝ, L = 7 → lever_minimum_force F L :=
by
  sorry

end lever_equilibrium_min_force_l68_68098


namespace sum_six_terms_l68_68415

variable (S : ℕ → ℝ)
variable (n : ℕ)
variable (S_2 S_4 S_6 : ℝ)

-- Given conditions
axiom sum_two_terms : S 2 = 4
axiom sum_four_terms : S 4 = 16

-- Problem statement
theorem sum_six_terms : S 6 = 52 :=
by
  -- Insert the proof here
  sorry

end sum_six_terms_l68_68415


namespace find_smallest_c_l68_68656

theorem find_smallest_c (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
    (graph_eq : ∀ x, (a * Real.sin (b * x + c) + d) = 5 → x = (π / 6))
    (amplitude_eq : a = 3) : c = π / 2 :=
sorry

end find_smallest_c_l68_68656


namespace line_equation_l68_68584

def line_through (A B : ℝ × ℝ) (x y : ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let m := (y₂ - y₁) / (x₂ - x₁)
  y - y₁ = m * (x - x₁)

noncomputable def is_trisection_point (A B QR : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (qx, qy) := QR
  (qx = (2 * x₂ + x₁) / 3 ∧ qy = (2 * y₂ + y₁) / 3) ∨
  (qx = (x₂ + 2 * x₁) / 3 ∧ qy = (y₂ + 2 * y₁) / 3)

theorem line_equation (A B P Q : ℝ × ℝ)
  (hA : A = (3, 4))
  (hB : B = (-4, 5))
  (hP : is_trisection_point B A P)
  (hQ : is_trisection_point B A Q) :
  line_through A P 1 3 ∨ line_through A P 2 1 → 
  (line_through A P 3 4 → P = (1, 3)) ∧ 
  (line_through A P 2 1 → P = (2, 1)) ∧ 
  (line_through A P x y → x - 4 * y + 13 = 0) := 
by 
  sorry

end line_equation_l68_68584


namespace train_speed_problem_l68_68478

theorem train_speed_problem (l1 l2 : ℝ) (v2 : ℝ) (t : ℝ) (v1 : ℝ) :
  l1 = 120 → l2 = 280 → v2 = 30 → t = 19.99840012798976 →
  0.4 / (t / 3600) = v1 + v2 → v1 = 42 :=
by
  intros hl1 hl2 hv2 ht hrel
  rw [hl1, hl2, hv2, ht] at *
  sorry

end train_speed_problem_l68_68478


namespace acute_angle_inclination_range_l68_68576

/-- 
For the line passing through points P(1-a, 1+a) and Q(3, 2a), 
prove that the range of the real number a such that the line has an acute angle of inclination is (-∞, 1) ∪ (1, 4).
-/
theorem acute_angle_inclination_range (a : ℝ) : 
  (a < 1 ∨ (1 < a ∧ a < 4)) ↔ (0 < (a - 1) / (4 - a)) :=
sorry

end acute_angle_inclination_range_l68_68576


namespace valentine_day_spending_l68_68724

structure DogTreatsConfig where
  heart_biscuits_count_A : Nat
  puppy_boots_count_A : Nat
  small_toy_count_A : Nat
  heart_biscuits_count_B : Nat
  puppy_boots_count_B : Nat
  large_toy_count_B : Nat
  heart_biscuit_price : Nat
  puppy_boots_price : Nat
  small_toy_price : Nat
  large_toy_price : Nat
  heart_biscuits_discount : Float
  large_toy_discount : Float

def treats_config : DogTreatsConfig :=
  { heart_biscuits_count_A := 5
    puppy_boots_count_A := 1
    small_toy_count_A := 1
    heart_biscuits_count_B := 7
    puppy_boots_count_B := 2
    large_toy_count_B := 1
    heart_biscuit_price := 2
    puppy_boots_price := 15
    small_toy_price := 10
    large_toy_price := 20
    heart_biscuits_discount := 0.20
    large_toy_discount := 0.15 }

def total_discounted_amount_spent (cfg : DogTreatsConfig) : Float :=
  let heart_biscuits_total_cost := (cfg.heart_biscuits_count_A + cfg.heart_biscuits_count_B) * cfg.heart_biscuit_price
  let puppy_boots_total_cost := (cfg.puppy_boots_count_A * cfg.puppy_boots_price) + (cfg.puppy_boots_count_B * cfg.puppy_boots_price)
  let small_toy_total_cost := cfg.small_toy_count_A * cfg.small_toy_price
  let large_toy_total_cost := cfg.large_toy_count_B * cfg.large_toy_price
  let total_cost_without_discount := Float.ofNat (heart_biscuits_total_cost + puppy_boots_total_cost + small_toy_total_cost + large_toy_total_cost)
  let heart_biscuits_discount_amount := cfg.heart_biscuits_discount * Float.ofNat heart_biscuits_total_cost
  let large_toy_discount_amount := cfg.large_toy_discount * Float.ofNat large_toy_total_cost
  let total_discount_amount := heart_biscuits_discount_amount + large_toy_discount_amount
  total_cost_without_discount - total_discount_amount

theorem valentine_day_spending : total_discounted_amount_spent treats_config = 91.20 := by
  sorry

end valentine_day_spending_l68_68724


namespace original_area_of_doubled_rectangle_l68_68064

theorem original_area_of_doubled_rectangle (A_new : ℝ) (h : A_new = 32) :
  ∃ A : ℝ, A * 4 = A_new ∧ A = 8 :=
by {
  use 8,
  split,
  { norm_num, exact h.symm },
  { rfl }
}

end original_area_of_doubled_rectangle_l68_68064


namespace necessary_but_not_sufficient_condition_l68_68221

-- Definitions
def represents_ellipse (m n : ℝ) (x y : ℝ) : Prop := 
  (x^2 / m + y^2 / n = 1)

-- Main theorem statement
theorem necessary_but_not_sufficient_condition 
    (m n x y : ℝ) (h_mn_pos : m * n > 0) :
    (represents_ellipse m n x y) → 
    (m ≠ n ∧ m > 0 ∧ n > 0 ∧ represents_ellipse m n x y) → 
    (m * n > 0) ∧ ¬(
    ∀ m n : ℝ, (m ≠ n ∧ m > 0 ∧ n > 0) →
    represents_ellipse m n x y
    ) :=
by
  sorry

end necessary_but_not_sufficient_condition_l68_68221


namespace not_on_graph_ln_l68_68577

theorem not_on_graph_ln {a b : ℝ} (h : b = Real.log a) : ¬ (1 + b = Real.log (a + Real.exp 1)) :=
by
  sorry

end not_on_graph_ln_l68_68577


namespace polynomial_solution_l68_68384

open Polynomial
open Real

theorem polynomial_solution (P : Polynomial ℝ) (h : ∀ x : ℝ, |x| ≤ 1 → P.eval (x * sqrt 2) = P.eval (x + sqrt (1 - x^2))) :
  ∃ U : Polynomial ℝ, P = (U.comp (Polynomial.C (1/4) - 2 * X^2 + 5 * X^4 - 4 * X^6 + X^8)) :=
sorry

end polynomial_solution_l68_68384


namespace correct_statement_is_c_l68_68625

-- Definitions corresponding to conditions
def lateral_surface_of_cone_unfolds_into_isosceles_triangle : Prop :=
  false -- This is false because it unfolds into a sector.

def prism_with_two_congruent_bases_other_faces_rectangles : Prop :=
  false -- This is false because the bases are congruent and parallel, and all other faces are parallelograms.

def frustum_complemented_with_pyramid_forms_new_pyramid : Prop :=
  true -- This is true, as explained in the solution.

def point_on_lateral_surface_of_truncated_cone_has_countless_generatrices : Prop :=
  false -- This is false because there is exactly one generatrix through such a point.

-- The main proof statement
theorem correct_statement_is_c :
  ¬lateral_surface_of_cone_unfolds_into_isosceles_triangle ∧
  ¬prism_with_two_congruent_bases_other_faces_rectangles ∧
  frustum_complemented_with_pyramid_forms_new_pyramid ∧
  ¬point_on_lateral_surface_of_truncated_cone_has_countless_generatrices :=
by
  -- The proof involves evaluating all the conditions above.
  sorry

end correct_statement_is_c_l68_68625


namespace least_five_digit_is_15625_l68_68282

noncomputable def least_five_digit_perfect_square_and_cube : ℕ :=
  15625

theorem least_five_digit_is_15625 :
  (10000 ≤ least_five_digit_perfect_square_and_cube) ∧ (least_five_digit_perfect_square_and_cube < 100000) ∧
  (∃ a : ℕ, least_five_digit_perfect_square_and_cube = a ^ 6) :=
by
  sorry

end least_five_digit_is_15625_l68_68282


namespace range_of_m_l68_68720

theorem range_of_m (α β m : ℝ) (hαβ : 0 < α ∧ α < 1 ∧ 1 < β ∧ β < 2)
  (h_eq : ∀ x, x^2 - 2*(m-1)*x + (m-1) = 0 ↔ (x = α ∨ x = β)) :
  2 < m ∧ m < 7 / 3 := by
  sorry

end range_of_m_l68_68720


namespace total_price_correct_l68_68738

-- Define the initial price, reduction, and the number of boxes
def initial_price : ℝ := 104
def price_reduction : ℝ := 24
def number_of_boxes : ℕ := 20

-- Define the new price as initial price minus the reduction
def new_price := initial_price - price_reduction

-- Define the total price as the new price times the number of boxes
def total_price := (number_of_boxes : ℝ) * new_price

-- The goal is to prove the total price equals 1600
theorem total_price_correct : total_price = 1600 := by
  sorry

end total_price_correct_l68_68738


namespace segments_do_not_intersect_l68_68849

noncomputable def check_intersection (AP PB BQ QC CR RD DS SA : ℚ) : Bool :=
  (AP / PB) * (BQ / QC) * (CR / RD) * (DS / SA) = 1

theorem segments_do_not_intersect :
  let AP := (3 : ℚ)
  let PB := (6 : ℚ)
  let BQ := (2 : ℚ)
  let QC := (4 : ℚ)
  let CR := (1 : ℚ)
  let RD := (5 : ℚ)
  let DS := (4 : ℚ)
  let SA := (6 : ℚ)
  ¬ check_intersection AP PB BQ QC CR RD DS SA :=
by sorry

end segments_do_not_intersect_l68_68849


namespace ellipse_sum_l68_68716

noncomputable def h : ℝ := 3
noncomputable def k : ℝ := 0
noncomputable def a : ℝ := 5
noncomputable def b : ℝ := Real.sqrt 21
noncomputable def F_1 : (ℝ × ℝ) := (1, 0)
noncomputable def F_2 : (ℝ × ℝ) := (5, 0)

theorem ellipse_sum :
  (F_1 = (1, 0)) → 
  (F_2 = (5, 0)) →
  (∀ P : (ℝ × ℝ), (Real.sqrt ((P.1 - F_1.1)^2 + (P.2 - F_1.2)^2) + Real.sqrt ((P.1 - F_2.1)^2 + (P.2 - F_2.2)^2) = 10)) →
  (h + k + a + b = 8 + Real.sqrt 21) :=
by
  intros
  sorry

end ellipse_sum_l68_68716


namespace votes_for_candidate_a_l68_68023

theorem votes_for_candidate_a :
  let total_votes : ℝ := 560000
  let percentage_invalid : ℝ := 0.15
  let percentage_candidate_a : ℝ := 0.85
  let valid_votes := (1 - percentage_invalid) * total_votes
  let votes_candidate_a := percentage_candidate_a * valid_votes
  votes_candidate_a = 404600 :=
by
  sorry

end votes_for_candidate_a_l68_68023


namespace perpendicular_bisector_eq_l68_68467

theorem perpendicular_bisector_eq
  (circle1 : ∀ x y : ℝ, x^2 + y^2 - 4 * x + 6 * y = 0)
  (circle2 : ∀ x y : ℝ, x^2 + y^2 - 6 * x = 0):
  ∃ (a b c : ℝ), a = 3 ∧ b = -1 ∧ c = -9 ∧ (∀ (x y : ℝ), a * x + b * y + c = 0) :=
sorry

end perpendicular_bisector_eq_l68_68467


namespace least_five_digit_perfect_square_and_cube_l68_68255

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ ∃ k : ℕ, k^6 = n ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68255


namespace trapezoid_height_l68_68602

-- Definitions of the problem conditions
def is_isosceles_trapezoid (a b : ℝ) : Prop :=
  ∃ (AB CD BM CN h : ℝ), a = 24 ∧ b = 10 ∧ AB = 25 ∧ CD = 25 ∧ BM = h ∧ CN = h ∧
  BM ^ 2 + ((24 - 10) / 2) ^ 2 = AB ^ 2

-- The theorem to prove
theorem trapezoid_height (a b : ℝ) (h : ℝ) 
  (H : is_isosceles_trapezoid a b) : h = 24 :=
sorry

end trapezoid_height_l68_68602


namespace curve_intersection_l68_68816

theorem curve_intersection (a m : ℝ) (a_pos : 0 < a) :
  (∀ x y : ℝ, 
     (x^2 / a^2 + y^2 = 1) ∧ (y^2 = 2 * (x + m)) 
     → 
     (1 / 2 * (a^2 + 1) = m) ∨ (-a < m ∧ m <= a))
  ∨ (a >= 1 → -a < m ∧ m < a) := 
sorry

end curve_intersection_l68_68816


namespace symmetric_line_equation_l68_68070

theorem symmetric_line_equation (x y : ℝ) :
  (∃ x y : ℝ, 3 * x + 4 * y = 2) →
  (4 * x + 3 * y = 2) :=
by
  intros h
  sorry

end symmetric_line_equation_l68_68070


namespace find_number_l68_68874

theorem find_number (n : ℕ) (h : 2 * 2 + n = 6) : n = 2 := by
  sorry

end find_number_l68_68874


namespace students_taking_only_science_l68_68901

theorem students_taking_only_science (total_students : ℕ) (students_science : ℕ) (students_math : ℕ)
  (h1 : total_students = 120) (h2 : students_science = 80) (h3 : students_math = 75) :
  (students_science - (students_science + students_math - total_students)) = 45 :=
by
  sorry

end students_taking_only_science_l68_68901


namespace intersection_A_B_l68_68194

def A : Set ℝ := { x | x + 1 > 0 }
def B : Set ℝ := { x | x < 0 }

theorem intersection_A_B :
  A ∩ B = { x | -1 < x ∧ x < 0 } :=
sorry

end intersection_A_B_l68_68194


namespace length_BC_fraction_AD_l68_68206

theorem length_BC_fraction_AD {A B C D : Type} {AB BD AC CD AD BC : ℕ} 
  (h1 : AB = 4 * BD) (h2 : AC = 9 * CD) (h3 : AD = AB + BD) (h4 : AD = AC + CD)
  (h5 : B ≠ A) (h6 : C ≠ A) (h7 : A ≠ D) : BC = AD / 10 :=
by
  sorry

end length_BC_fraction_AD_l68_68206


namespace bus_passenger_count_l68_68463

-- Definitions for conditions
def initial_passengers : ℕ := 0
def passengers_first_stop (initial : ℕ) : ℕ := initial + 7
def passengers_second_stop (after_first : ℕ) : ℕ := after_first - 3 + 5
def passengers_third_stop (after_second : ℕ) : ℕ := after_second - 2 + 4

-- Statement we want to prove
theorem bus_passenger_count : 
  passengers_third_stop (passengers_second_stop (passengers_first_stop initial_passengers)) = 11 :=
by
  -- proof would go here
  sorry

end bus_passenger_count_l68_68463


namespace find_k_intersects_parabola_at_one_point_l68_68232

theorem find_k_intersects_parabola_at_one_point :
  ∃ k : ℝ, (∀ y : ℝ, -3 * y^2 - 4 * y + 7 = k ↔ y = (-4 / (2 * 3))) →
    k = 25 / 3 :=
by sorry

end find_k_intersects_parabola_at_one_point_l68_68232


namespace combined_stripes_is_22_l68_68984

-- Definition of stripes per shoe for each person based on the conditions
def stripes_per_shoe_Olga : ℕ := 3
def stripes_per_shoe_Rick : ℕ := stripes_per_shoe_Olga - 1
def stripes_per_shoe_Hortense : ℕ := stripes_per_shoe_Olga * 2

-- The total combined number of stripes on all shoes for Olga, Rick, and Hortense
def total_stripes : ℕ := 2 * (stripes_per_shoe_Olga + stripes_per_shoe_Rick + stripes_per_shoe_Hortense)

-- The statement to prove that the total number of stripes on all their shoes is 22
theorem combined_stripes_is_22 : total_stripes = 22 :=
by
  sorry

end combined_stripes_is_22_l68_68984


namespace downstream_speed_l68_68775

def V_u : ℝ := 26
def V_m : ℝ := 28
def V_s : ℝ := V_m - V_u
def V_d : ℝ := V_m + V_s

theorem downstream_speed : V_d = 30 := by
  sorry

end downstream_speed_l68_68775


namespace arithmetic_sequence_first_term_and_common_difference_l68_68933

def a_n (n : ℕ) : ℕ := 2 * n + 5

theorem arithmetic_sequence_first_term_and_common_difference :
  a_n 1 = 7 ∧ ∀ n : ℕ, a_n (n + 1) - a_n n = 2 := by
  sorry

end arithmetic_sequence_first_term_and_common_difference_l68_68933


namespace pqrsum_eq_neg209_l68_68717

noncomputable def Q (z : ℂ) (p q r : ℝ) : ℂ :=
  z^3 + (p : ℂ) * z^2 + (q : ℂ) * z + (r : ℂ)

theorem pqrsum_eq_neg209 (u p q r : ℂ) (i : ℂ) (hu : u.im ≠ 0) (huj : i^2 = -1)
  (hroots : (Q (u + 2 * i) p q r) = 0 ∧ (Q (u + 7 * i) p q r) = 0 ∧ (Q (2 * u - 5) p q r) = 0)
  (hreals : p.im = 0 ∧ q.im = 0 ∧ r.im = 0) :
  p + q + r = -209 :=
sorry

end pqrsum_eq_neg209_l68_68717


namespace tangent_parallel_l68_68245

-- Define the curve function
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve function
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Define the slope of the line 4x - y - 1 = 0, which is 4
def line_slope : ℝ := 4

-- The main theorem statement
theorem tangent_parallel (a b : ℝ) (h1 : f a = b) (h2 : f' a = line_slope) :
  (a = 1 ∧ b = 0) ∨ (a = -1 ∧ b = -4) :=
sorry

end tangent_parallel_l68_68245


namespace circle_Q_radius_l68_68364

theorem circle_Q_radius
  (radius_P : ℝ := 2)
  (radius_S : ℝ := 4)
  (u v : ℝ)
  (h1: (2 + v)^2 = (2 + u)^2 + v^2)
  (h2: (4 - v)^2 = u^2 + v^2)
  (h3: v = u + u^2 / 2)
  (h4: v = 2 - u^2 / 4) :
  v = 16 / 9 :=
by
  /- Proof goes here. -/
  sorry

end circle_Q_radius_l68_68364


namespace call_cost_per_minute_l68_68033

-- Definitions (conditions)
def initial_credit : ℝ := 30
def call_duration : ℕ := 22
def remaining_credit : ℝ := 26.48

-- The goal is to prove that the cost per minute of the call is 0.16
theorem call_cost_per_minute :
  (initial_credit - remaining_credit) / call_duration = 0.16 := 
sorry

end call_cost_per_minute_l68_68033


namespace largest_x_l68_68397

-- Conditions setup
def largest_x_satisfying_condition : ℝ :=
  let x : ℝ := 63 / 8 in x

theorem largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 8 / 9) : 
  x = largest_x_satisfying_condition :=
by
  sorry

end largest_x_l68_68397


namespace solution_proof_l68_68215

noncomputable def proof_problem : Prop :=
  ∀ (x : ℝ), x ≠ 1 → (1 - 1 / (x - 1) = 2 * x / (1 - x)) → x = 2 / 3

theorem solution_proof : proof_problem := 
by
  sorry

end solution_proof_l68_68215


namespace rationalize_denominator_l68_68207

theorem rationalize_denominator (cbrt : ℝ → ℝ) (h₁ : cbrt 81 = 3 * cbrt 3) :
  1 / (cbrt 3 + cbrt 81) = cbrt 9 / 12 :=
sorry

end rationalize_denominator_l68_68207


namespace sequence_fraction_l68_68407

-- Definitions for arithmetic and geometric sequences
def isArithmeticSeq (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def isGeometricSeq (a b c : ℝ) :=
  b^2 = a * c

-- Given conditions
variables {a : ℕ → ℝ} {d : ℝ}

-- a is an arithmetic sequence with common difference d ≠ 0
axiom h1 : isArithmeticSeq a d
axiom h2 : d ≠ 0

-- a_2, a_3, a_9 form a geometric sequence
axiom h3 : isGeometricSeq (a 2) (a 3) (a 9)

-- Goal: prove the value of the given expression
theorem sequence_fraction {a : ℕ → ℝ} {d : ℝ} (h1 : isArithmeticSeq a d) (h2 : d ≠ 0) (h3 : isGeometricSeq (a 2) (a 3) (a 9)) :
  (a 2 + a 3 + a 4) / (a 4 + a 5 + a 6) = 3 / 8 :=
by
  sorry

end sequence_fraction_l68_68407


namespace discontinuity_conditions_l68_68621

variable {α : Type*} [TopologicalSpace α] {β : Type*} [TopologicalSpace β]

def is_discontinuous_at (f : α → β) (M₀ : α) : Prop :=
¬ContinuousAt f M₀

theorem discontinuity_conditions (f : α → β) (M₀ : α) :
  is_discontinuous_at f M₀ ↔
  (¬∃ U : Set α, M₀ ∈ U ∧ ∀ x ∈ U, M₀ ≠ x → f x ≠ f M₀) ∨
  (∃ U : Set α, M₀ ∈ U ∧ ∀ x ∈ U, x ≠ M₀ → ¬ContinuousAt f x) ∨
  (∃ L, Tendsto f (𝓝 M₀) (𝓝 L) ∧ f M₀ ≠ L) ∨
  (∃ C : Set α, M₀ ∈ C ∧ ∀ M ∈ C, ¬ContinuousAt f M) := sorry

end discontinuity_conditions_l68_68621


namespace prism_volume_is_25_l68_68333

noncomputable def triangle_area (a b : ℝ) : ℝ := (1 / 2) * a * b

noncomputable def prism_volume (base_area height : ℝ) : ℝ := base_area * height

theorem prism_volume_is_25 :
  let a := Real.sqrt 5
  let base_area := triangle_area a a
  let volume := prism_volume base_area 10
  volume = 25 :=
by
  intros
  sorry

end prism_volume_is_25_l68_68333


namespace janet_earned_1390_in_interest_l68_68835

def janets_total_interest (total_investment investment_at_10_rate investment_at_10_interest investment_at_1_rate remaining_investment remaining_investment_interest : ℝ) : ℝ :=
    investment_at_10_interest + remaining_investment_interest

theorem janet_earned_1390_in_interest :
  janets_total_interest 31000 12000 0.10 (12000 * 0.10) 0.01 (19000 * 0.01) = 1390 :=
by
  sorry

end janet_earned_1390_in_interest_l68_68835


namespace linear_equation_a_is_the_only_one_l68_68875

-- Definitions for each equation
def equation_a (x y : ℝ) : Prop := x + y = 2
def equation_b (x : ℝ) : Prop := x + 1 = -10
def equation_c (x y : ℝ) : Prop := x - 1/y = 6
def equation_d (x y : ℝ) : Prop := x^2 = 2 * y

-- Proof that equation_a is the only linear equation with two variables
theorem linear_equation_a_is_the_only_one (x y : ℝ) : 
  equation_a x y ∧ ¬equation_b x ∧ ¬(∃ y, equation_c x y) ∧ ¬(∃ y, equation_d x y) :=
by
  sorry

end linear_equation_a_is_the_only_one_l68_68875


namespace exists_primes_sum_2024_with_one_gt_1000_l68_68115

open Nat

-- Definition of primality
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Conditions given in the problem
def sum_primes_eq_2024 (p q : ℕ) : Prop :=
  p + q = 2024 ∧ is_prime p ∧ is_prime q

def at_least_one_gt_1000 (p q : ℕ) : Prop :=
  p > 1000 ∨ q > 1000

-- The theorem to be proved
theorem exists_primes_sum_2024_with_one_gt_1000 :
  ∃ (p q : ℕ), sum_primes_eq_2024 p q ∧ at_least_one_gt_1000 p q :=
sorry

end exists_primes_sum_2024_with_one_gt_1000_l68_68115


namespace find_unknown_number_l68_68334

theorem find_unknown_number (x : ℤ) (h : (20 + 40 + 60) / 3 = 9 + (10 + 70 + x) / 3) : x = 13 :=
by
  sorry

end find_unknown_number_l68_68334


namespace mean_of_sequence_l68_68087

def mean (s : List ℕ) : ℚ := (s.sum : ℚ) / s.length

theorem mean_of_sequence :
  mean [1^2, 2^2, 3^2, 4^2, 5^2, 6^2, 7^2, 2] = 17.75 := by
sorry

end mean_of_sequence_l68_68087


namespace least_five_digit_perfect_square_and_cube_l68_68288

theorem least_five_digit_perfect_square_and_cube : 
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ (∃ a : ℕ, n = a ^ 6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68288


namespace quadratic_unique_real_root_l68_68177

theorem quadratic_unique_real_root (m : ℝ) :
  (∀ x : ℝ, x^2 + 6 * m * x + 2 * m = 0 → ∃! r : ℝ, x = r) → m = 2/9 :=
by
  sorry

end quadratic_unique_real_root_l68_68177


namespace quadratic_single_root_pos_value_l68_68174

theorem quadratic_single_root_pos_value (m : ℝ) (h1 : (6 * m)^2 - 4 * 1 * 2 * m = 0) : m = 2 / 9 :=
sorry

end quadratic_single_root_pos_value_l68_68174


namespace least_five_digit_perfect_square_and_cube_l68_68304

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, n = 5^6 ∧ (n >= 10000 ∧ n < 100000) ∧ (∃ k : ℕ, n = k^2) ∧ (∃ m : ℕ, n = m^3) :=
by
  use 15625
  split
  · exact pow_succ 5 6
  split
  · have h1: 10000 ≤ 5^6 := nat.le_of_eq (calc 5^6 = 15625 : rfl)
    have h2: 5^6 < 100000 := nat.lt_of_eq_of_lt (calc 5^6 = 15625 : rfl) (by decide)
    exact ⟨h1, h2⟩
  · use 125
    exact pow_two 125
  · use 25
    exact pow_three 25
  sorry

end least_five_digit_perfect_square_and_cube_l68_68304


namespace range_of_sum_of_products_l68_68924

theorem range_of_sum_of_products (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c)
  (h_sum : a + b + c = (Real.sqrt 3) / 2) :
  0 < (a * b + b * c + c * a) ∧ (a * b + b * c + c * a) ≤ 1 / 4 :=
by
  sorry

end range_of_sum_of_products_l68_68924


namespace sin_2A_cos_C_l68_68926

theorem sin_2A (A B : ℝ) (h1 : Real.sin A = 3 / 5) (h2 : Real.cos B = -5 / 13) : 
  Real.sin (2 * A) = 24 / 25 :=
sorry

theorem cos_C (A B C : ℝ) (h1 : Real.sin A = 3 / 5) (h2 : Real.cos B = -5 / 13) 
  (h3 : ∀ x y z : ℝ, x + y + z = π) :
  Real.cos C = 56 / 65 :=
sorry

end sin_2A_cos_C_l68_68926


namespace log_base_9_of_x_cubed_is_3_l68_68853

theorem log_base_9_of_x_cubed_is_3 
  (x : Real) 
  (hx : x = 9.000000000000002) : 
  Real.logb 9 (x^3) = 3 := 
by 
  sorry

end log_base_9_of_x_cubed_is_3_l68_68853


namespace domain_composite_function_l68_68411

theorem domain_composite_function (f : ℝ → ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 3 → f x = y) →
  (∀ x, 1 ≤ x ∧ x ≤ 2 → f (2^x - 1) = y) :=
by
  sorry

end domain_composite_function_l68_68411


namespace verify_solution_l68_68697

-- Definition of the condition about Pascal's Triangle rows.
def pascals_triangle_contains_odd (n : ℕ) : ℕ → Prop
| 0 => true
| k => ∀ (m : ℕ) (hm : m ≤ k), ((n.choose m) % 2) = 1

-- Define the proposition we wish to prove.
def problem_statement_as_proof : Prop :=
  (Finset.filter (λ n, pascals_triangle_contains_odd n n)
                 (Finset.range 20)).card = 1

-- Create the main theorem to verify our claim.
theorem verify_solution : problem_statement_as_proof := by
  sorry

end verify_solution_l68_68697


namespace art_collection_area_l68_68368

theorem art_collection_area :
  let square_paintings := 3 * (6 * 6)
  let small_paintings := 4 * (2 * 3)
  let large_painting := 1 * (10 * 15)
  square_paintings + small_paintings + large_painting = 282 := by
  sorry

end art_collection_area_l68_68368


namespace measure_8_liters_with_buckets_l68_68564

theorem measure_8_liters_with_buckets (capacity_B10 capacity_B6 : ℕ) (B10_target : ℕ) (B10_initial B6_initial : ℕ) : 
  capacity_B10 = 10 ∧ capacity_B6 = 6 ∧ B10_target = 8 ∧ B10_initial = 0 ∧ B6_initial = 0 →
  ∃ (B10 B6 : ℕ), B10 = 8 ∧ (B10 ≥ 0 ∧ B10 ≤ capacity_B10) ∧ (B6 ≥ 0 ∧ B6 ≤ capacity_B6) :=
by
  sorry

end measure_8_liters_with_buckets_l68_68564


namespace pyramid_top_block_l68_68354

theorem pyramid_top_block (a b c d e : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : a ≠ e)
                         (h5 : b ≠ c) (h6 : b ≠ d) (h7 : b ≠ e) (h8 : c ≠ d) (h9 : c ≠ e) (h10 : d ≠ e)
                         (h : a * b ^ 4 * c ^ 6 * d ^ 4 * e = 140026320) : 
                         (a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 7 ∧ e = 5) ∨ 
                         (a = 1 ∧ b = 7 ∧ c = 3 ∧ d = 2 ∧ e = 5) ∨ 
                         (a = 5 ∧ b = 2 ∧ c = 3 ∧ d = 7 ∧ e = 1) ∨ 
                         (a = 5 ∧ b = 7 ∧ c = 3 ∧ d = 2 ∧ e = 1) := 
sorry

end pyramid_top_block_l68_68354


namespace min_value_condition_l68_68554

noncomputable def poly_min_value (a b : ℝ) : ℝ := a^2 + b^2

theorem min_value_condition (a b : ℝ) (h: ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : 
  ∃ a b : ℝ, poly_min_value a b = 4 := 
by sorry

end min_value_condition_l68_68554


namespace g_g_g_g_of_2_eq_242_l68_68042

def g (x : ℕ) : ℕ :=
  if x % 3 = 0 then x / 3 else 3 * x + 2

theorem g_g_g_g_of_2_eq_242 : g (g (g (g 2))) = 242 :=
by
  sorry

end g_g_g_g_of_2_eq_242_l68_68042


namespace lally_internet_days_l68_68631

-- Definitions based on the conditions
def cost_per_day : ℝ := 0.5
def debt_limit : ℝ := 5
def initial_payment : ℝ := 7
def initial_balance : ℝ := 0

-- Proof problem statement
theorem lally_internet_days : ∀ (d : ℕ), 
  (initial_balance + initial_payment - cost_per_day * d ≤ debt_limit) -> (d = 14) :=
sorry

end lally_internet_days_l68_68631


namespace professors_seating_l68_68251

/-- There are 13 chairs arranged in a single row. Four professors (Alpha, Beta, Gamma, and Delta) and 
nine students need to sit such that each professor is seated between at least one student. Professors 
cannot occupy the first or last position. Prove that the number of ways in which the four professors 
can choose their chairs is 3024. -/
theorem professors_seating : 
  ∃ c : ℕ, 
  (c = (∑ x in (Finset.range 13).powerset.filter (λ s, 
    s.card = 4 ∧ ∀ x ∈ s, 2 ≤ x ∧ x ≤ 11 ∧ 
    (∀ p q ∈ s, p ≠ q → (p + 1 < q ∨ q + 1 < p)) ), 
    1) * factoral 4)) ∧ c = 3024 := 
by
  sorry

end professors_seating_l68_68251


namespace find_n_l68_68632

theorem find_n (n : ℕ) 
  (hM : ∀ M, M = n - 7 → 1 ≤ M)
  (hA : ∀ A, A = n - 2 → 1 ≤ A)
  (hT : ∀ M A, M = n - 7 → A = n - 2 → M + A < n) :
  n = 8 :=
by
  sorry

end find_n_l68_68632


namespace least_five_digit_perfect_square_and_cube_l68_68322

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k1 : ℕ, n = k1 ^ 2) ∧ (∃ k2 : ℕ, n = k2 ^ 3) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68322


namespace remainder_of_trailing_zeroes_in_factorials_product_l68_68979

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def product_factorials (n : ℕ) : ℕ :=
  (List.range (n + 1)).foldr (λ x acc => acc * factorial x) 1 

def trailing_zeroes (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n / 5 + trailing_zeroes (n / 5))

def trailing_zeroes_in_product (n : ℕ) : ℕ :=
  (List.range (n + 1)).foldr (λ x acc => acc + trailing_zeroes x) 0 

theorem remainder_of_trailing_zeroes_in_factorials_product :
  let N := trailing_zeroes_in_product 150
  N % 500 = 45 :=
by
  sorry

end remainder_of_trailing_zeroes_in_factorials_product_l68_68979


namespace time_to_fill_pool_l68_68722

theorem time_to_fill_pool :
  ∀ (total_volume : ℝ) (filling_rate : ℝ) (leaking_rate : ℝ),
  total_volume = 60 →
  filling_rate = 1.6 →
  leaking_rate = 0.1 →
  (total_volume / (filling_rate - leaking_rate)) = 40 :=
by
  intros total_volume filling_rate leaking_rate hv hf hl
  rw [hv, hf, hl]
  sorry

end time_to_fill_pool_l68_68722


namespace one_number_is_zero_l68_68615

variable {a b c : ℤ}
variable (cards : Fin 30 → ℤ)

theorem one_number_is_zero (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c)
    (h_cards : ∀ i : Fin 30, cards i = a ∨ cards i = b ∨ cards i = c)
    (h_sum_zero : ∀ (S : Finset (Fin 30)) (hS : S.card = 5),
        ∃ T : Finset (Fin 30), T.card = 5 ∧ (S ∪ T).sum cards = 0) :
    b = 0 := 
sorry

end one_number_is_zero_l68_68615


namespace journey_duration_l68_68888

theorem journey_duration
  (distance : ℕ) (speed : ℕ) (h1 : distance = 48) (h2 : speed = 8) :
  distance / speed = 6 := 
by
  sorry

end journey_duration_l68_68888


namespace tan_subtraction_formula_l68_68573

theorem tan_subtraction_formula 
  (α β : ℝ) (hα : Real.tan α = 3) (hβ : Real.tan β = 2) :
  Real.tan (α - β) = 1 / 7 := 
by
  sorry

end tan_subtraction_formula_l68_68573


namespace land_area_l68_68444

theorem land_area (x : ℝ) (h : (70 * x - 800) / 1.2 * 1.6 + 800 = 80 * x) : x = 20 :=
by
  sorry

end land_area_l68_68444


namespace least_five_digit_perfect_square_and_cube_l68_68260

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ ∃ k : ℕ, k^6 = n ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68260


namespace parabola_translation_l68_68619

-- Define the initial equation of the parabola
def initial_parabola (x : ℝ) : ℝ := x^2 - 2

-- Define the transformation: translate one unit to the right
def translate_right (x : ℝ) : ℝ := initial_parabola (x - 1)

-- Define the transformation: move up three units
def move_up (y : ℝ) : ℝ := y + 3

-- Define the resulting equation after the transformations
def resulting_parabola (x : ℝ) : ℝ := move_up (translate_right x)

-- Define the target equation
def target_parabola (x : ℝ) : ℝ := (x - 1)^2 + 1

-- Formalize the proof problem
theorem parabola_translation :
  ∀ x : ℝ, resulting_parabola x = target_parabola x :=
by
  -- Proof steps go here
  sorry

end parabola_translation_l68_68619


namespace inequality_proof_l68_68764

noncomputable theory

variables {p q : ℝ}
variables {m n : ℕ}

-- Define the conditions
def conditions (p q : ℝ) (m n : ℕ) : Prop :=
  p ≥ 0 ∧ q ≥ 0 ∧ p + q = 1 ∧ m > 0 ∧ n > 0

-- Define the statement to prove
theorem inequality_proof (p q : ℝ) (m n : ℕ) (h : conditions p q m n) :
  (1 - p^m)^n + (1 - q^n)^m ≥ 1 :=
sorry

end inequality_proof_l68_68764


namespace tan_triple_angle_l68_68952

variable θ : ℝ
variable h : Real.tan θ = 3

theorem tan_triple_angle (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 :=
sorry

end tan_triple_angle_l68_68952


namespace total_pencils_correct_l68_68359

def reeta_pencils : Nat := 20
def anika_pencils : Nat := 2 * reeta_pencils + 4
def total_pencils : Nat := reeta_pencils + anika_pencils

theorem total_pencils_correct : total_pencils = 64 :=
by
  sorry

end total_pencils_correct_l68_68359


namespace scores_are_correct_l68_68138

variable
  Andrey_score : ℝ
  Dima_score : ℝ
  Vanya_score : ℝ
  Sasha_score : ℝ

-- Conditions
axiom andrey_first : Andrey_score > Dima_score ∧ Andrey_score > Vanya_score ∧ Andrey_score > Sasha_score
axiom dima_second : Dima_score > Vanya_score ∧ Dima_score > Sasha_score
axiom vanya_third : Vanya_score > Sasha_score
axiom unique_scores : Andrey_score ≠ Dima_score ∧ Andrey_score ≠ Vanya_score ∧ Andrey_score ≠ Sasha_score ∧ Dima_score ≠ Vanya_score ∧ Dima_score ≠ Sasha_score ∧ Vanya_score ≠ Sasha_score
axiom total_points : Andrey_score + Dima_score + Vanya_score + Sasha_score = 12
axiom andrey_sasha_wins : Andrey_score = 4 ∧ Sasha_score = 2

-- Conclusion
theorem scores_are_correct :
  Andrey_score = 4 ∧ Dima_score = 3.5 ∧ Vanya_score = 2.5 ∧ Sasha_score = 2 :=
  sorry

end scores_are_correct_l68_68138


namespace fred_washing_cars_l68_68836

theorem fred_washing_cars :
  ∀ (initial_amount final_amount money_made : ℕ),
  initial_amount = 23 →
  final_amount = 86 →
  money_made = final_amount - initial_amount →
  money_made = 63 := by
    intros initial_amount final_amount money_made h_initial h_final h_calc
    rw [h_initial, h_final] at h_calc
    exact h_calc

end fred_washing_cars_l68_68836


namespace symmetry_about_origin_l68_68971

def Point : Type := ℝ × ℝ

def A : Point := (2, -1)
def B : Point := (-2, 1)

theorem symmetry_about_origin (A B : Point) : A = (2, -1) ∧ B = (-2, 1) → B = (-A.1, -A.2) :=
by
  sorry

end symmetry_about_origin_l68_68971


namespace johns_burritos_l68_68434

-- Definitions based on conditions:
def initial_burritos : Nat := 3 * 20
def burritos_given_away : Nat := initial_burritos / 3
def burritos_after_giving_away : Nat := initial_burritos - burritos_given_away
def burritos_eaten : Nat := 3 * 10
def burritos_left : Nat := burritos_after_giving_away - burritos_eaten

-- The theorem we need to prove:
theorem johns_burritos : burritos_left = 10 := by
  sorry

end johns_burritos_l68_68434


namespace boat_speed_still_water_l68_68092

def effective_upstream_speed (b c : ℝ) : ℝ := b - c
def effective_downstream_speed (b c : ℝ) : ℝ := b + c

theorem boat_speed_still_water :
  ∃ b c : ℝ, effective_upstream_speed b c = 9 ∧ effective_downstream_speed b c = 15 ∧ b = 12 :=
by {
  sorry
}

end boat_speed_still_water_l68_68092


namespace scores_fraction_difference_l68_68617

theorem scores_fraction_difference (y : ℕ) (white_ratio : ℕ) (black_ratio : ℕ) (total : ℕ) 
(h1 : white_ratio = 7) (h2 : black_ratio = 6) (h3 : total = 78) 
(h4 : y = white_ratio + black_ratio) : 
  ((white_ratio * total / y) - (black_ratio * total / y)) / total = 1 / 13 :=
by
 sorry

end scores_fraction_difference_l68_68617


namespace greatest_n_divides_l68_68810

theorem greatest_n_divides (m : ℕ) (hm : 0 < m) : 
  ∃ n : ℕ, (n = m^4 - m^2 + m) ∧ (m^2 + n) ∣ (n^2 + m) := 
by {
  sorry
}

end greatest_n_divides_l68_68810


namespace psychologist_charge_difference_l68_68769

-- Define the variables and conditions
variables (F A : ℝ)
axiom cond1 : F + 4 * A = 250
axiom cond2 : F + A = 115

theorem psychologist_charge_difference : F - A = 25 :=
by
  -- conditions are already stated as axioms, we'll just provide the target theorem
  sorry

end psychologist_charge_difference_l68_68769


namespace branches_on_one_stem_l68_68377

theorem branches_on_one_stem (x : ℕ) (h : 1 + x + x^2 = 31) : x = 5 :=
by {
  sorry
}

end branches_on_one_stem_l68_68377


namespace least_five_digit_perfect_square_and_cube_l68_68309

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, n = 5^6 ∧ (n >= 10000 ∧ n < 100000) ∧ (∃ k : ℕ, n = k^2) ∧ (∃ m : ℕ, n = m^3) :=
by
  use 15625
  split
  · exact pow_succ 5 6
  split
  · have h1: 10000 ≤ 5^6 := nat.le_of_eq (calc 5^6 = 15625 : rfl)
    have h2: 5^6 < 100000 := nat.lt_of_eq_of_lt (calc 5^6 = 15625 : rfl) (by decide)
    exact ⟨h1, h2⟩
  · use 125
    exact pow_two 125
  · use 25
    exact pow_three 25
  sorry

end least_five_digit_perfect_square_and_cube_l68_68309


namespace trajectory_is_parabola_l68_68173

def distance_to_line (p : ℝ × ℝ) (a : ℝ) : ℝ :=
|p.1 - a|

noncomputable def distance_to_point (p q : ℝ × ℝ) : ℝ :=
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def parabola_condition (P : ℝ × ℝ) : Prop :=
distance_to_line P (-1) + 1 = distance_to_point P (2, 0)

theorem trajectory_is_parabola : ∀ (P : ℝ × ℝ), parabola_condition P ↔
(P.1 + 1)^2 = (Real.sqrt ((P.1 - 2)^2 + P.2^2))^2 := 
by 
  sorry

end trajectory_is_parabola_l68_68173


namespace cos_of_acute_angle_l68_68003

theorem cos_of_acute_angle (θ : ℝ) (hθ1 : 0 < θ ∧ θ < π / 2) (hθ2 : Real.sin θ = 1 / 3) :
  Real.cos θ = 2 * Real.sqrt 2 / 3 :=
by
  -- The proof steps will be filled here
  sorry

end cos_of_acute_angle_l68_68003


namespace least_five_digit_perfect_square_and_cube_l68_68290

theorem least_five_digit_perfect_square_and_cube : 
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ (∃ a : ℕ, n = a ^ 6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68290


namespace total_books_l68_68587

theorem total_books (joan_books : ℕ) (tom_books : ℕ) (h1 : joan_books = 10) (h2 : tom_books = 38) : joan_books + tom_books = 48 :=
by
  -- insert proof here
  sorry

end total_books_l68_68587


namespace distance_earth_sun_l68_68862

theorem distance_earth_sun (speed_of_light : ℝ) (time_to_earth: ℝ) 
(h1 : speed_of_light = 3 * 10^8) 
(h2 : time_to_earth = 5 * 10^2) :
  speed_of_light * time_to_earth = 1.5 * 10^11 := 
by 
  -- proof steps can be filled here
  sorry

end distance_earth_sun_l68_68862


namespace Kyle_is_25_l68_68710

variable (Tyson_age : ℕ := 20)
variable (Frederick_age : ℕ := 2 * Tyson_age)
variable (Julian_age : ℕ := Frederick_age - 20)
variable (Kyle_age : ℕ := Julian_age + 5)

theorem Kyle_is_25 : Kyle_age = 25 := by
  sorry

end Kyle_is_25_l68_68710


namespace least_five_digit_is_15625_l68_68283

noncomputable def least_five_digit_perfect_square_and_cube : ℕ :=
  15625

theorem least_five_digit_is_15625 :
  (10000 ≤ least_five_digit_perfect_square_and_cube) ∧ (least_five_digit_perfect_square_and_cube < 100000) ∧
  (∃ a : ℕ, least_five_digit_perfect_square_and_cube = a ^ 6) :=
by
  sorry

end least_five_digit_is_15625_l68_68283


namespace boys_down_slide_l68_68090

theorem boys_down_slide (boys_1 boys_2 : ℕ) (h : boys_1 = 22) (h' : boys_2 = 13) : boys_1 + boys_2 = 35 := by
  sorry

end boys_down_slide_l68_68090


namespace mark_reading_time_l68_68723

variable (x y : ℕ)

theorem mark_reading_time (x y : ℕ) : 
  7 * x + y = 7 * x + y :=
by
  sorry

end mark_reading_time_l68_68723


namespace bottles_more_than_apples_l68_68773

def bottles_regular : ℕ := 72
def bottles_diet : ℕ := 32
def apples : ℕ := 78

def total_bottles : ℕ := bottles_regular + bottles_diet

theorem bottles_more_than_apples : (total_bottles - apples) = 26 := by
  sorry

end bottles_more_than_apples_l68_68773


namespace find_cosine_l68_68147
open Real

noncomputable def alpha (α : ℝ) : Prop := 0 < α ∧ α < π / 2 ∧ sin α = 3 / 5

theorem find_cosine (α : ℝ) (h : alpha α) :
  cos (π - α / 2) = - (3 * sqrt 10) / 10 :=
by sorry

end find_cosine_l68_68147


namespace number_of_men_in_first_group_l68_68426

-- Definitions for the conditions
def rate_of_work (men : ℕ) (length : ℕ) (days : ℕ) : ℕ :=
  length / days / men

def work_rate_first_group (M : ℕ) : ℕ :=
  rate_of_work M 48 2

def work_rate_second_group : ℕ :=
  rate_of_work 2 36 3

theorem number_of_men_in_first_group (M : ℕ) 
  (h₁ : work_rate_first_group M = 24)
  (h₂ : work_rate_second_group = 12) :
  M = 4 :=
  sorry

end number_of_men_in_first_group_l68_68426


namespace grace_walks_distance_l68_68688

theorem grace_walks_distance
  (south_blocks west_blocks : ℕ)
  (block_length_in_miles : ℚ)
  (h_south_blocks : south_blocks = 4)
  (h_west_blocks : west_blocks = 8)
  (h_block_length : block_length_in_miles = 1 / 4)
  : ((south_blocks + west_blocks) * block_length_in_miles = 3) :=
by 
  sorry

end grace_walks_distance_l68_68688


namespace fourth_number_is_8_l68_68856

theorem fourth_number_is_8 (a b c : ℕ) (mean : ℕ) (h_mean : mean = 20) (h_a : a = 12) (h_b : b = 24) (h_c : c = 36) :
  ∃ d : ℕ, mean * 4 = a + b + c + d ∧ (∃ x : ℕ, d = x^2) ∧ d = 8 := by
sorry

end fourth_number_is_8_l68_68856


namespace jessica_cut_21_roses_l68_68616

def initial_roses : ℕ := 2
def thrown_roses : ℕ := 4
def final_roses : ℕ := 23

theorem jessica_cut_21_roses : (final_roses - initial_roses) = 21 :=
by
  sorry

end jessica_cut_21_roses_l68_68616


namespace find_special_numbers_l68_68193

/-- Define the sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

/-- Define the main statement to be proved -/
theorem find_special_numbers :
  { n : ℕ | sum_of_digits n * (sum_of_digits n - 1) = n - 1 } = {1, 13, 43, 91, 157} :=
by
  sorry

end find_special_numbers_l68_68193


namespace cos_B_find_b_l68_68198

theorem cos_B (a b c : ℝ) (h1 : 2 * b = a + c) (h2 : 7 * a = 3 * c) :
  Real.cos (Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) = 11 / 14 := by
  sorry

theorem find_b (a b c : ℝ) (h1 : 2 * b = a + c) (h2 : 7 * a = 3 * c)
  (area : ℝ := 15 * Real.sqrt 3 / 4)
  (h3 : (1/2) * a * c * Real.sin (Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) = area) :
  b = 5 := by
  sorry

end cos_B_find_b_l68_68198


namespace riley_pawns_lost_l68_68709

theorem riley_pawns_lost (initial_pawns : ℕ) (kennedy_lost : ℕ) (total_pawns_left : ℕ)
  (kennedy_initial_pawns : ℕ) (riley_initial_pawns : ℕ) : 
  kennedy_initial_pawns = initial_pawns ∧
  riley_initial_pawns = initial_pawns ∧
  kennedy_lost = 4 ∧
  total_pawns_left = 11 →
  riley_initial_pawns - (total_pawns_left - (kennedy_initial_pawns - kennedy_lost)) = 1 :=
by
  sorry

end riley_pawns_lost_l68_68709


namespace remove_green_balls_l68_68093

theorem remove_green_balls (total_balls green_balls yellow_balls x : ℕ) 
  (h1 : total_balls = 600)
  (h2 : green_balls = 420)
  (h3 : yellow_balls = 180)
  (h4 : green_balls = 70 * total_balls / 100)
  (h5 : yellow_balls = total_balls - green_balls)
  (h6 : (green_balls - x) = 60 * (total_balls - x) / 100) :
  x = 150 := 
by {
  -- sorry placeholder for proof.
  sorry
}

end remove_green_balls_l68_68093


namespace find_number_of_real_solutions_l68_68536

noncomputable def f (x : ℝ) : ℝ := 
  (∑ k in finset.range 1 (50 + 1), (k : ℝ) / (x - k))

theorem find_number_of_real_solutions (f : ℝ → ℝ) (hx : ∀ x, f x = ∑ k in finset.range 50, (k + 1 : ℝ) / (x - (k + 1))) :
  ∃ n, n = 51 := 
begin
  sorry
end

end find_number_of_real_solutions_l68_68536


namespace interval_k_is_40_l68_68475

def total_students := 1200
def sample_size := 30

theorem interval_k_is_40 : (total_students / sample_size) = 40 :=
by
  sorry

end interval_k_is_40_l68_68475


namespace quadratic_real_roots_quadratic_product_of_roots_l68_68737

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, x^2 - 2 * m * x + m^2 + m - 3 = 0) ↔ m ≤ 3 := by
{
  sorry
}

theorem quadratic_product_of_roots (m : ℝ) (α β : ℝ) :
  α * β = 17 ∧ α^2 - 2 * m * α + m^2 + m - 3 = 0 ∧ β^2 - 2 * m * β + m^2 + m - 3 = 0 →
  m = -5 := by
{
  sorry
}

end quadratic_real_roots_quadratic_product_of_roots_l68_68737


namespace bus_speed_including_stoppages_l68_68665

theorem bus_speed_including_stoppages
  (speed_without_stoppages : ℝ)
  (stoppage_time : ℝ)
  (remaining_time_ratio : ℝ)
  (h1 : speed_without_stoppages = 12)
  (h2 : stoppage_time = 0.5)
  (h3 : remaining_time_ratio = 1 - stoppage_time) :
  (speed_without_stoppages * remaining_time_ratio) = 6 := 
by
  sorry

end bus_speed_including_stoppages_l68_68665


namespace cistern_emptying_l68_68893

theorem cistern_emptying (h: (3 / 4) / 12 = 1 / 16) : (8 * (1 / 16) = 1 / 2) :=
by sorry

end cistern_emptying_l68_68893


namespace find_x_l68_68542

theorem find_x (a b x : ℝ) (h1 : 2^a = x) (h2 : 3^b = x)
    (h3 : 1 / a + 1 / b = 1) : x = 6 :=
sorry

end find_x_l68_68542


namespace tan_triple_angle_l68_68949

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end tan_triple_angle_l68_68949


namespace rectangle_area_correct_l68_68750

-- Definitions of side lengths
def sideOne : ℝ := 5.9
def sideTwo : ℝ := 3

-- Definition of the area calculation for a rectangle
def rectangleArea (a b : ℝ) : ℝ :=
  a * b

-- The main theorem stating the area is as calculated
theorem rectangle_area_correct :
  rectangleArea sideOne sideTwo = 17.7 := by
  sorry

end rectangle_area_correct_l68_68750


namespace rhombus_area_l68_68531

theorem rhombus_area (R r : ℝ) : 
  ∃ A : ℝ, A = (8 * R^3 * r^3) / ((R^2 + r^2)^2) :=
by
  sorry

end rhombus_area_l68_68531


namespace remainder_when_add_13_l68_68424

theorem remainder_when_add_13 (x : ℤ) (h : x % 82 = 5) : (x + 13) % 41 = 18 :=
sorry

end remainder_when_add_13_l68_68424


namespace pond_length_l68_68582

-- Define the dimensions and volume of the pond
def pond_width : ℝ := 15
def pond_depth : ℝ := 5
def pond_volume : ℝ := 1500

-- Define the length variable
variable (L : ℝ)

-- State that the volume relationship holds and L is the length we're solving for
theorem pond_length :
  pond_volume = L * pond_width * pond_depth → L = 20 :=
by
  sorry

end pond_length_l68_68582


namespace abc_value_l68_68592

variables (a b c : ℂ)

theorem abc_value :
  (a * b + 4 * b = -16) →
  (b * c + 4 * c = -16) →
  (c * a + 4 * a = -16) →
  a * b * c = 64 :=
by
  intros h1 h2 h3
  sorry

end abc_value_l68_68592


namespace angle_in_fourth_quadrant_l68_68169
-- Import the necessary library

-- Definition of the conditions
def pos_cos (α : ℝ) := cos α > 0
def neg_tan (α : ℝ) := tan α < 0

-- Theorem statement
theorem angle_in_fourth_quadrant (α : ℝ) 
  (hcos : pos_cos α) 
  (htan : neg_tan α) : 
  ∃ n : ℤ, (π / 2 + n * π < α) ∧ (α < π + n * π) :=
sorry

end angle_in_fourth_quadrant_l68_68169


namespace product_of_powers_l68_68791

theorem product_of_powers (x y : ℕ) (h1 : x = 2) (h2 : y = 3) :
  ((x ^ 1 + y ^ 1) * (x ^ 2 + y ^ 2) * (x ^ 4 + y ^ 4) * 
   (x ^ 8 + y ^ 8) * (x ^ 16 + y ^ 16) * (x ^ 32 + y ^ 32) * 
   (x ^ 64 + y ^ 64)) = y ^ 128 - x ^ 128 :=
by
  rw [h1, h2]
  -- We would proceed with the proof here, but it's not needed per instructions.
  sorry

end product_of_powers_l68_68791


namespace building_height_l68_68832

-- We start by defining the heights of the stories.
def first_story_height : ℕ := 12
def additional_height_per_story : ℕ := 3
def number_of_stories : ℕ := 20
def first_ten_stories : ℕ := 10
def remaining_stories : ℕ := number_of_stories - first_ten_stories

-- Now we define what it means for the total height of the building to be 270 feet.
theorem building_height :
  first_ten_stories * first_story_height + remaining_stories * (first_story_height + additional_height_per_story) = 270 := by
  sorry

end building_height_l68_68832


namespace interior_surface_area_is_correct_l68_68057

-- Define the original dimensions of the rectangular sheet
def original_length : ℕ := 40
def original_width : ℕ := 50

-- Define the side length of the square corners
def corner_side : ℕ := 10

-- Define the area of the original sheet
def area_original : ℕ := original_length * original_width

-- Define the area of one square corner
def area_corner : ℕ := corner_side * corner_side

-- Define the total area removed by all four corners
def area_removed : ℕ := 4 * area_corner

-- Define the remaining area after the corners are removed
def area_remaining : ℕ := area_original - area_removed

-- The theorem to be proved
theorem interior_surface_area_is_correct : area_remaining = 1600 := by
  sorry

end interior_surface_area_is_correct_l68_68057


namespace scores_are_correct_l68_68137

variable
  Andrey_score : ℝ
  Dima_score : ℝ
  Vanya_score : ℝ
  Sasha_score : ℝ

-- Conditions
axiom andrey_first : Andrey_score > Dima_score ∧ Andrey_score > Vanya_score ∧ Andrey_score > Sasha_score
axiom dima_second : Dima_score > Vanya_score ∧ Dima_score > Sasha_score
axiom vanya_third : Vanya_score > Sasha_score
axiom unique_scores : Andrey_score ≠ Dima_score ∧ Andrey_score ≠ Vanya_score ∧ Andrey_score ≠ Sasha_score ∧ Dima_score ≠ Vanya_score ∧ Dima_score ≠ Sasha_score ∧ Vanya_score ≠ Sasha_score
axiom total_points : Andrey_score + Dima_score + Vanya_score + Sasha_score = 12
axiom andrey_sasha_wins : Andrey_score = 4 ∧ Sasha_score = 2

-- Conclusion
theorem scores_are_correct :
  Andrey_score = 4 ∧ Dima_score = 3.5 ∧ Vanya_score = 2.5 ∧ Sasha_score = 2 :=
  sorry

end scores_are_correct_l68_68137


namespace tan_sum_identity_l68_68539

noncomputable def tan_25 := Real.tan (Real.pi / 180 * 25)
noncomputable def tan_35 := Real.tan (Real.pi / 180 * 35)
noncomputable def sqrt_3 := Real.sqrt 3

theorem tan_sum_identity :
  tan_25 + tan_35 + sqrt_3 * tan_25 * tan_35 = 1 :=
by
  sorry

end tan_sum_identity_l68_68539


namespace present_age_of_son_l68_68629

theorem present_age_of_son (S F : ℕ)
  (h1 : F = S + 24)
  (h2 : F + 2 = 2 * (S + 2)) :
  S = 22 :=
by {
  -- The proof is omitted, as per instructions.
  sorry
}

end present_age_of_son_l68_68629


namespace tan_triple_angle_l68_68951

variable θ : ℝ
variable h : Real.tan θ = 3

theorem tan_triple_angle (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 :=
sorry

end tan_triple_angle_l68_68951


namespace chimney_bricks_l68_68658

theorem chimney_bricks (x : ℕ) 
  (h1 : Brenda_rate = x / 8) 
  (h2 : Brandon_rate = x / 12) 
  (h3 : Brian_rate = x / 16) 
  (h4 : effective_combined_rate = (Brenda_rate + Brandon_rate + Brian_rate) - 15) 
  (h5 : total_time = 4) :
  (4 * effective_combined_rate) = x := 
  sorry

end chimney_bricks_l68_68658


namespace angle_of_inclination_of_tangent_at_point_l68_68121

theorem angle_of_inclination_of_tangent_at_point :
  ∀ (x y : ℝ), y = (1 / 3) * x^3 - 2 → 
  ((∃ (x0: ℝ), x0 = 1) ∧ (∃ (y0: ℝ), y0 = -5/3)) → 
  ∃ θ : ℝ, θ = 45 := by
  sorry

end angle_of_inclination_of_tangent_at_point_l68_68121


namespace sum_of_N_values_eq_neg_one_l68_68171

theorem sum_of_N_values_eq_neg_one (R : ℝ) :
  ∀ (N : ℝ), N ≠ 0 ∧ (N + N^2 - 5 / N = R) →
  (∃ N₁ N₂ N₃ : ℝ, N₁ + N₂ + N₃ = -1 ∧ N₁ ≠ 0 ∧ N₂ ≠ 0 ∧ N₃ ≠ 0) :=
by
  sorry

end sum_of_N_values_eq_neg_one_l68_68171


namespace man_age_twice_son_age_l68_68502

-- Definitions based on conditions
def son_age : ℕ := 20
def man_age : ℕ := son_age + 22

-- Definition of the main statement to be proven
theorem man_age_twice_son_age (Y : ℕ) : man_age + Y = 2 * (son_age + Y) → Y = 2 :=
by sorry

end man_age_twice_son_age_l68_68502


namespace linear_equation_conditions_l68_68018

theorem linear_equation_conditions (m n : ℤ) :
  (∀ x y : ℝ, 4 * x^(m - n) - 5 * y^(m + n) = 6 → 
    m - n = 1 ∧ m + n = 1) →
  m = 1 ∧ n = 0 :=
by
  sorry

end linear_equation_conditions_l68_68018


namespace Cherie_boxes_l68_68191

theorem Cherie_boxes (x : ℕ) :
  (2 * 8 + x * (8 + 9) = 33) → x = 1 :=
by
  intros h
  have h_eq : 16 + 17 * x = 33 := by simp [mul_add, mul_comm, h]
  linarith

end Cherie_boxes_l68_68191


namespace turtles_on_lonely_island_l68_68562

theorem turtles_on_lonely_island (T : ℕ) (h1 : 60 = 2 * T + 10) : T = 25 := 
by sorry

end turtles_on_lonely_island_l68_68562


namespace sample_avg_std_dev_xy_l68_68414

theorem sample_avg_std_dev_xy {x y : ℝ} (h1 : (4 + 5 + 6 + x + y) / 5 = 5)
  (h2 : (( (4 - 5)^2 + (5 - 5)^2 + (6 - 5)^2 + (x - 5)^2 + (y - 5)^2 ) / 5) = 2) : x * y = 21 :=
by
  sorry

end sample_avg_std_dev_xy_l68_68414


namespace measure_8_liters_with_buckets_l68_68563

theorem measure_8_liters_with_buckets (capacity_B10 capacity_B6 : ℕ) (B10_target : ℕ) (B10_initial B6_initial : ℕ) : 
  capacity_B10 = 10 ∧ capacity_B6 = 6 ∧ B10_target = 8 ∧ B10_initial = 0 ∧ B6_initial = 0 →
  ∃ (B10 B6 : ℕ), B10 = 8 ∧ (B10 ≥ 0 ∧ B10 ≤ capacity_B10) ∧ (B6 ≥ 0 ∧ B6 ≤ capacity_B6) :=
by
  sorry

end measure_8_liters_with_buckets_l68_68563


namespace train_speed_is_72_l68_68652

def distance : ℕ := 24
def time_minutes : ℕ := 20
def time_hours : ℚ := time_minutes / 60
def speed := distance / time_hours

theorem train_speed_is_72 :
  speed = 72 := by
  sorry

end train_speed_is_72_l68_68652


namespace ellipse_equation_line_equation_l68_68355
-- Import the necessary libraries

-- Problem (I): The equation of the ellipse
theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (hA : (1 : ℝ) / a^2 + (9 / 4 : ℝ) / b^2 = 1)
  (h_ecc : b^2 = (3 / 4 : ℝ) * a^2) : 
  (a^2 = 4 ∧ b^2 = 3) :=
by
  sorry

-- Problem (II): The equation of the line
theorem line_equation (k : ℝ) (h_area : (12 * Real.sqrt (2 : ℝ)) / 7 = 12 * abs k / (4 * k^2 + 3)) : 
  k = 1 ∨ k = -1 :=
by
  sorry

end ellipse_equation_line_equation_l68_68355


namespace min_nSn_l68_68811

theorem min_nSn 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (m : ℕ)
  (h1 : m ≥ 2)
  (h2 : S (m-1) = -2) 
  (h3 : S m = 0) 
  (h4 : S (m+1) = 3) : 
  ∃ n : ℕ, n * S n = -9 :=
by {
  sorry
}

end min_nSn_l68_68811


namespace cricket_target_runs_l68_68022

theorem cricket_target_runs 
  (run_rate1 : ℝ) (run_rate2 : ℝ) (overs : ℕ)
  (h1 : run_rate1 = 5.4) (h2 : run_rate2 = 10.6) (h3 : overs = 25) :
  (run_rate1 * overs + run_rate2 * overs = 400) :=
by sorry

end cricket_target_runs_l68_68022


namespace game_winner_Aerith_first_game_winner_Bob_first_l68_68897

-- Conditions: row of 20 squares, players take turns crossing out one square,
-- game ends when there are two squares left, Aerith wins if two remaining squares
-- are adjacent, Bob wins if they are not adjacent.

-- Definition of the game and winning conditions
inductive Player
| Aerith
| Bob

-- Function to determine the winner given the initial player
def winning_strategy (initial_player : Player) : Player :=
  match initial_player with
  | Player.Aerith => Player.Bob  -- Bob wins if Aerith goes first
  | Player.Bob    => Player.Aerith  -- Aerith wins if Bob goes first

-- Statement to prove
theorem game_winner_Aerith_first : 
  winning_strategy Player.Aerith = Player.Bob :=
by 
  sorry -- Proof is to be done

theorem game_winner_Bob_first :
  winning_strategy Player.Bob = Player.Aerith :=
by
  sorry -- Proof is to be done

end game_winner_Aerith_first_game_winner_Bob_first_l68_68897


namespace sixth_number_is_eight_l68_68970

/- 
  The conditions are:
  1. The sequence is an increasing list of consecutive integers.
  2. The 3rd and 4th numbers add up to 11.
  We need to prove that the 6th number is 8.
-/

theorem sixth_number_is_eight (n : ℕ) (h : n + (n + 1) = 11) : (n + 3) = 8 :=
by
  sorry

end sixth_number_is_eight_l68_68970


namespace factorize_poly1_factorize_poly2_factorize_poly3_factorize_poly4_l68_68126

-- Statements corresponding to the given problems

-- Theorem for 1)
theorem factorize_poly1 (a : ℤ) : 
  (a^7 + a^5 + 1) = (a^2 + a + 1) * (a^5 - a^4 + a^3 - a + 1) := 
by sorry

-- Theorem for 2)
theorem factorize_poly2 (a b : ℤ) : 
  (a^5 + a*b^4 + b^5) = (a + b) * (a^4 - a^3*b + a^2*b^2 - a*b^3 + b^4) := 
by sorry

-- Theorem for 3)
theorem factorize_poly3 (a : ℤ) : 
  (a^7 - 1) = (a - 1) * (a^6 + a^5 + a^4 + a^3 + a^2 + a + 1) := 
by sorry

-- Theorem for 4)
theorem factorize_poly4 (a x : ℤ) : 
  (2 * a^3 - a * x^2 - x^3) = (a - x) * (2 * a^2 + 2 * a * x + x^2) := 
by sorry

end factorize_poly1_factorize_poly2_factorize_poly3_factorize_poly4_l68_68126


namespace at_least_one_equals_a_l68_68600

theorem at_least_one_equals_a (x y z a : ℝ) (hx_ne_0 : x ≠ 0) (hy_ne_0 : y ≠ 0) (hz_ne_0 : z ≠ 0) (ha_ne_0 : a ≠ 0)
  (h1 : x + y + z = a) (h2 : 1/x + 1/y + 1/z = 1/a) : x = a ∨ y = a ∨ z = a :=
  sorry

end at_least_one_equals_a_l68_68600


namespace sum_two_integers_l68_68469

theorem sum_two_integers (a b : ℤ) (h1 : a = 17) (h2 : b = 19) : a + b = 36 := by
  sorry

end sum_two_integers_l68_68469


namespace ball_distribution_l68_68691

theorem ball_distribution :
  let n := 6 in
  let b := 3 in
  let ways := (choose 6 6) + (choose 6 5) + (choose 6 4 * choose 2 2) + (choose 6 3 * choose 3 2 * choose 1 1) in
  ways = 82 :=
by
  sorry

end ball_distribution_l68_68691


namespace cost_per_trip_l68_68449

theorem cost_per_trip
  (pass_cost : ℝ)
  (oldest_trips : ℕ)
  (youngest_trips : ℕ)
  (h_pass_cost : pass_cost = 100.0)
  (h_oldest_trips : oldest_trips = 35)
  (h_youngest_trips : youngest_trips = 15) :
  (2 * pass_cost) / (oldest_trips + youngest_trips) = 4.0 :=
by
  sorry

end cost_per_trip_l68_68449


namespace geometric_to_arithmetic_l68_68151

theorem geometric_to_arithmetic {a1 a2 a3 a4 q : ℝ}
  (hq : q ≠ 1)
  (geom_seq : a2 = a1 * q ∧ a3 = a1 * q^2 ∧ a4 = a1 * q^3)
  (arith_seq : (2 * a3 = a1 + a4 ∨ 2 * a2 = a1 + a4)) :
  q = (1 + Real.sqrt 5) / 2 ∨ q = (-1 + Real.sqrt 5) / 2 :=
by
  sorry

end geometric_to_arithmetic_l68_68151


namespace find_a12_a12_value_l68_68675

variable (a : ℕ → ℝ)

-- Given conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

axiom h1 : a 6 + a 10 = 16
axiom h2 : a 4 = 1

-- Theorem to prove
theorem find_a12 : a 6 + a 10 = a 4 + a 12 := by
  -- Place for the proof
  sorry

theorem a12_value : (∃ a12, a 6 + a 10 = 16 ∧ a 4 = 1 ∧ a 6 + a 10 = a 4 + a12) → a 12 = 15 :=
by
  -- Place for the proof
  sorry

end find_a12_a12_value_l68_68675


namespace condition_for_diff_of_roots_l68_68795

/-- Statement: For a quadratic equation of the form x^2 + px + q = 0, if the difference of the roots is a, then the condition a^2 - p^2 = -4q holds. -/
theorem condition_for_diff_of_roots (p q a : ℝ) (h : ∀ x : ℝ, x^2 + p * x + q = 0 → ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ x1 - x2 = a) :
  a^2 - p^2 = -4 * q :=
sorry

end condition_for_diff_of_roots_l68_68795


namespace find_y_l68_68605

theorem find_y (y : ℝ) (h : 9 * y^2 + 36 * y^2 + 9 * y^2 = 1300) : 
  y = Real.sqrt 1300 / Real.sqrt 54 :=
by 
  sorry

end find_y_l68_68605


namespace john_ratio_amounts_l68_68706

/-- John gets $30 from his grandpa and some multiple of that amount from his grandma. 
He got $120 from the two grandparents. What is the ratio of the amount he got from 
his grandma to the amount he got from his grandpa? --/
theorem john_ratio_amounts (amount_grandpa amount_total : ℝ) (multiple : ℝ) :
  amount_grandpa = 30 → amount_total = 120 →
  amount_total = amount_grandpa + multiple * amount_grandpa →
  multiple = 3 :=
by
  intros h1 h2 h3
  sorry

end john_ratio_amounts_l68_68706


namespace oldest_child_age_l68_68601

theorem oldest_child_age 
  (avg_age : ℕ) (child1 : ℕ) (child2 : ℕ) (child3 : ℕ) (child4 : ℕ)
  (h_avg : avg_age = 8) 
  (h_child1 : child1 = 5) 
  (h_child2 : child2 = 7) 
  (h_child3 : child3 = 10)
  (h_avg_eq : (child1 + child2 + child3 + child4) / 4 = avg_age) :
  child4 = 10 := 
by 
  sorry

end oldest_child_age_l68_68601


namespace age_of_teacher_l68_68491

/-- Given that the average age of 23 students is 22 years, and the average age increases
by 1 year when the teacher's age is included, prove that the teacher's age is 46 years. -/
theorem age_of_teacher (n : ℕ) (s_avg : ℕ) (new_avg : ℕ) (teacher_age : ℕ) :
  n = 23 →
  s_avg = 22 →
  new_avg = s_avg + 1 →
  teacher_age = new_avg * (n + 1) - s_avg * n →
  teacher_age = 46 :=
by
  intros h_n h_s_avg h_new_avg h_teacher_age
  sorry

end age_of_teacher_l68_68491


namespace ted_age_solution_l68_68902

theorem ted_age_solution (t s : ℝ) (h1 : t = 3 * s - 10) (h2 : t + s = 60) : t = 42.5 :=
by {
  sorry
}

end ted_age_solution_l68_68902


namespace exponentiated_value_l68_68922

theorem exponentiated_value (x a b : ℝ) (h1 : x^a = 2) (h2 : x^b = 3) : x^(3 * a + b) = 24 := by
  sorry

end exponentiated_value_l68_68922


namespace john_burritos_left_l68_68433

theorem john_burritos_left : 
  ∀ (boxes : ℕ) (burritos_per_box : ℕ) (given_away_fraction : ℚ) (eaten_per_day : ℕ) (days : ℕ),
  boxes = 3 → 
  burritos_per_box = 20 →
  given_away_fraction = 1 / 3 →
  eaten_per_day = 3 →
  days = 10 →
  let initial_burritos := boxes * burritos_per_box in
  let given_away_burritos := given_away_fraction * initial_burritos in
  let after_giving_away := initial_burritos - given_away_burritos in
  let eaten_burritos := eaten_per_day * days in
  let final_burritos := after_giving_away - eaten_burritos in
  final_burritos = 10 := 
by 
  intros,
  sorry

end john_burritos_left_l68_68433


namespace terrell_total_distance_l68_68219

theorem terrell_total_distance (saturday_distance sunday_distance : ℝ) (h_saturday : saturday_distance = 8.2) (h_sunday : sunday_distance = 1.6) :
  saturday_distance + sunday_distance = 9.8 :=
by
  rw [h_saturday, h_sunday]
  -- sorry
  norm_num

end terrell_total_distance_l68_68219


namespace xiaohongs_mother_deposit_l68_68754

theorem xiaohongs_mother_deposit (x : ℝ) :
  x + x * 3.69 / 100 * 3 * (1 - 20 / 100) = 5442.8 :=
by
  sorry

end xiaohongs_mother_deposit_l68_68754


namespace areaOfTangencyTriangle_l68_68848

noncomputable def semiPerimeter (a b c : ℝ) : ℝ :=
  (a + b + c) / 2

noncomputable def areaABC (a b c : ℝ) : ℝ :=
  let p := semiPerimeter a b c
  Real.sqrt (p * (p - a) * (p - b) * (p - c))

noncomputable def excircleRadius (a b c : ℝ) : ℝ :=
  let S := areaABC a b c
  let p := semiPerimeter a b c
  S / (p - a)

theorem areaOfTangencyTriangle (a b c R : ℝ) :
  let p := semiPerimeter a b c
  let S := areaABC a b c
  let ra := excircleRadius a b c
  (S * (ra / (2 * R))) = (S ^ 2 / (2 * R * (p - a))) :=
by
  let p := semiPerimeter a b c
  let S := areaABC a b c
  let ra := excircleRadius a b c
  sorry

end areaOfTangencyTriangle_l68_68848


namespace least_five_digit_perfect_square_cube_l68_68270

theorem least_five_digit_perfect_square_cube :
  ∃ n : Nat, (10000 ≤ n ∧ n < 100000) ∧ (∃ m1 m2 : Nat, n = m1^2 ∧ n = m2^3 ∧ n = 15625) :=
by
  sorry

end least_five_digit_perfect_square_cube_l68_68270


namespace pints_in_two_liters_l68_68410

theorem pints_in_two_liters (p : ℝ) (h : p = 1.575 / 0.75) : 2 * p = 4.2 := 
sorry

end pints_in_two_liters_l68_68410


namespace sin_add_cos_l68_68742

theorem sin_add_cos (s72 c18 c72 s18 : ℝ) (h1 : s72 = Real.sin (72 * Real.pi / 180)) (h2 : c18 = Real.cos (18 * Real.pi / 180)) (h3 : c72 = Real.cos (72 * Real.pi / 180)) (h4 : s18 = Real.sin (18 * Real.pi / 180)) :
  s72 * c18 + c72 * s18 = 1 :=
by 
  sorry

end sin_add_cos_l68_68742


namespace mike_hours_per_day_l68_68047

theorem mike_hours_per_day (total_hours : ℕ) (total_days : ℕ) (h_total_hours : total_hours = 15) (h_total_days : total_days = 5) : (total_hours / total_days) = 3 := by
  sorry

end mike_hours_per_day_l68_68047


namespace Kyle_is_25_l68_68713

-- Definitions based on the conditions
def Tyson_age : Nat := 20
def Frederick_age : Nat := 2 * Tyson_age
def Julian_age : Nat := Frederick_age - 20
def Kyle_age : Nat := Julian_age + 5

-- The theorem to prove
theorem Kyle_is_25 : Kyle_age = 25 := by
  sorry

end Kyle_is_25_l68_68713


namespace incorrect_statements_l68_68486

open Function

theorem incorrect_statements (a : ℝ) (x y x₁ y₁ x₂ y₂ k : ℝ) : 
  ¬ (a = -1 ↔ (∀ x y, a^2 * x - y + 1 = 0 ∧ x - a * y - 2 = 0 → (a = -1 ∨ a = 0))) ∧ 
  ¬ (∀ x y (x₁ y₁ x₂ y₂ : ℝ), (∃ (m : ℝ), (y - y₁) = m * (x - x₁) ∧ (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)) → 
    ((y - y₁) / (y₂ - y₁) = (x - x₁) / (x₂ - x₁))) :=
sorry

end incorrect_statements_l68_68486


namespace least_five_digit_perfect_square_and_cube_l68_68311

/-- 
Prove that the least five-digit whole number that is both a perfect square 
and a perfect cube is 15625.
-/
theorem least_five_digit_perfect_square_and_cube : 
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k : ℕ, n = k^6) ∧ (∀ m : ℕ, (10000 ≤ m ∧ m < 100000) ∧ (∃ l : ℕ, m = l^6) → n ≤ m) :=
begin
  use 15625,
  split,
  { split,
    { norm_num },
    { norm_num } },
  { use 5,
    norm_num,
    intros m hm,
    rcases hm with ⟨⟨hm1, hm2⟩, ⟨l, rfl⟩⟩,
    cases le_or_lt l 5 with hl hl,
    { by_contradiction h,
      have : m < 10000 := by norm_num [nat.pow_succ, nat.pow_succ, hl],
      exact not_le_of_gt this hm1 },
    { have : l ≥ 6 := nat.lt_succ_iff.mp hl,
      have : m ≥ 6^6 := nat.pow_le_pow_of_le_right nat.zero_lt_succ_iff.1 this,
      norm_num at this,
      exact not_lt_of_ge this hm2 } }
end

end least_five_digit_perfect_square_and_cube_l68_68311


namespace product_positive_l68_68598

variables {x y : ℝ}

noncomputable def non_zero (z : ℝ) := z ≠ 0

theorem product_positive (hx : non_zero x) (hy : non_zero y) 
(h1 : x^2 - x > y^2) (h2 : y^2 - y > x^2) : x * y > 0 :=
by
  sorry

end product_positive_l68_68598


namespace find_a_l68_68515

noncomputable def ab (a b : ℝ) : ℝ := 3 * a - 2 * b^2

theorem find_a {a : ℝ} : ab a 6 = -3 → a = 23 :=
by
  sorry

end find_a_l68_68515


namespace find_first_number_l68_68855

theorem find_first_number (HCF LCM number2 number1 : ℕ) 
    (hcf_condition : HCF = 12) 
    (lcm_condition : LCM = 396) 
    (number2_condition : number2 = 198) 
    (number1_condition : number1 * number2 = HCF * LCM) : 
    number1 = 24 := 
by 
    sorry

end find_first_number_l68_68855


namespace abs_diff_squares_eq_300_l68_68079

theorem abs_diff_squares_eq_300 : 
  let a := (103 : ℚ) / 2 
  let b := (97 : ℚ) / 2
  |a^2 - b^2| = 300 := 
by
  let a := (103 : ℚ) / 2 
  let b := (97 : ℚ) / 2
  sorry

end abs_diff_squares_eq_300_l68_68079


namespace seating_arrangement_l68_68380

theorem seating_arrangement : 
  ∃ x y z : ℕ, 
  7 * x + 8 * y + 9 * z = 65 ∧ z = 1 ∧ x + y + z = r :=
sorry

end seating_arrangement_l68_68380


namespace chess_tournament_scores_l68_68142

theorem chess_tournament_scores :
    ∃ (A D V S : ℝ),
    A ≠ D ∧ A ≠ V ∧ A ≠ S ∧ D ≠ V ∧ D ≠ S ∧ V ≠ S ∧
    A = 4 ∧ D = 3.5 ∧ V = 2.5 ∧ S = 2 ∧
    A > D ∧ D > V ∧ V > S ∧
    (∃ (wins_A wins_S : ℕ), wins_A = wins_S) :=
begin
    sorry
end

end chess_tournament_scores_l68_68142


namespace share_per_person_in_dollars_l68_68218

-- Definitions based on conditions
def total_cost_euros : ℝ := 25 * 10^9  -- 25 billion Euros
def number_of_people : ℝ := 300 * 10^6  -- 300 million people
def exchange_rate : ℝ := 1.2  -- 1 Euro = 1.2 dollars

-- To prove
theorem share_per_person_in_dollars : (total_cost_euros * exchange_rate) / number_of_people = 100 := 
by 
  sorry

end share_per_person_in_dollars_l68_68218


namespace green_more_than_blue_l68_68489

theorem green_more_than_blue (B Y G : Nat) (h1 : B + Y + G = 108) (h2 : B * 7 = Y * 3) (h3 : B * 8 = G * 3) : G - B = 30 := by
  sorry

end green_more_than_blue_l68_68489


namespace square_area_with_circles_l68_68505

theorem square_area_with_circles (r : ℝ) (h : r = 8) : (2 * (2 * r))^2 = 1024 := 
by 
  sorry

end square_area_with_circles_l68_68505


namespace C1_Cartesian_equation_C2_Cartesian_equation_m_value_when_C2_passes_through_P_l68_68025

noncomputable def parametric_C1 (α : ℝ) : ℝ × ℝ := (2 + Real.cos α, 4 + Real.sin α)

noncomputable def polar_C2 (ρ θ m : ℝ) : ℝ := ρ * (Real.cos θ - m * Real.sin θ) + 1

theorem C1_Cartesian_equation :
  ∀ (x y : ℝ), (∃ α : ℝ, parametric_C1 α = (x, y)) ↔ (x - 2)^2 + (y - 4)^2 = 1 := sorry

theorem C2_Cartesian_equation :
  ∀ (x y m : ℝ), (∃ ρ θ : ℝ, polar_C2 ρ θ m = 0 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ)
  ↔ x - m * y + 1 = 0 := sorry

def closest_point_on_C1_to_x_axis : ℝ × ℝ := (2, 3)

theorem m_value_when_C2_passes_through_P :
  ∃ (m : ℝ), x - m * y + 1 = 0 ∧ x = 2 ∧ y = 3 ∧ m = 1 := sorry

end C1_Cartesian_equation_C2_Cartesian_equation_m_value_when_C2_passes_through_P_l68_68025


namespace least_five_digit_is_15625_l68_68285

noncomputable def least_five_digit_perfect_square_and_cube : ℕ :=
  15625

theorem least_five_digit_is_15625 :
  (10000 ≤ least_five_digit_perfect_square_and_cube) ∧ (least_five_digit_perfect_square_and_cube < 100000) ∧
  (∃ a : ℕ, least_five_digit_perfect_square_and_cube = a ^ 6) :=
by
  sorry

end least_five_digit_is_15625_l68_68285


namespace subtract_base3_sum_eq_result_l68_68731

theorem subtract_base3_sum_eq_result :
  let a := 10 -- interpreted as 10_3
  let b := 1101 -- interpreted as 1101_3
  let c := 2102 -- interpreted as 2102_3
  let d := 212 -- interpreted as 212_3
  let sum := 1210 -- interpreted as the base 3 sum of a + b + c
  let result := 1101 -- interpreted as the final base 3 result
  sum - d = result :=
by sorry

end subtract_base3_sum_eq_result_l68_68731


namespace problem_statement_l68_68005

noncomputable def f : ℝ → ℝ := sorry

theorem problem_statement (h1 : ∀ x : ℝ, f (x + 2016) = f (-x + 2016))
    (h2 : ∀ x1 x2 : ℝ, 2016 ≤ x1 ∧ 2016 ≤ x2 ∧ x1 ≠ x2 → (f x2 - f x1) / (x2 - x1) < 0) :
    f 2019 < f 2014 ∧ f 2014 < f 2017 :=
sorry

end problem_statement_l68_68005


namespace union_of_sets_l68_68559

theorem union_of_sets (A B : Set ℕ) (hA : A = {1, 2}) (hB : B = {2, 3}) : A ∪ B = {1, 2, 3} := by
  sorry

end union_of_sets_l68_68559


namespace interior_angle_of_regular_polygon_l68_68107

theorem interior_angle_of_regular_polygon (n : ℕ) (h_diagonals : n * (n - 3) / 2 = n) :
    n = 5 ∧ (5 - 2) * 180 / 5 = 108 := by
  sorry

end interior_angle_of_regular_polygon_l68_68107


namespace linear_equation_with_two_variables_is_A_l68_68876

-- Define the equations
def equation_A := ∀ (x y : ℝ), x + y = 2
def equation_B := ∀ (x y : ℝ), x + 1 = -10
def equation_C := ∀ (x y : ℝ), x - 1 / y = 6
def equation_D := ∀ (x y : ℝ), x^2 = 2 * y

-- Define the question as a theorem
theorem linear_equation_with_two_variables_is_A :
  (∃ (x y : ℝ), equation_A x y)
  ∧ ¬(∃ (x y : ℝ), equation_B x y)
  ∧ ¬(∃ (x y : ℝ), equation_C x y)
  ∧ ¬(∃ (x y : ℝ), equation_D x y) := by
sorry

end linear_equation_with_two_variables_is_A_l68_68876


namespace numLinesTangentToCircles_eq_2_l68_68205

noncomputable def lineTangents (A B : Point) (dAB rA rB : ℝ) : ℕ :=
  if dAB < rA + rB then 2 else 0

theorem numLinesTangentToCircles_eq_2
  (A B : Point) (dAB rA rB : ℝ)
  (hAB : dAB = 4) (hA : rA = 3) (hB : rB = 2) :
  lineTangents A B dAB rA rB = 2 := by
  sorry

end numLinesTangentToCircles_eq_2_l68_68205


namespace least_value_of_a_l68_68914

theorem least_value_of_a (a : ℝ) (h : a^2 - 12 * a + 35 ≤ 0) : 5 ≤ a :=
by {
  sorry
}

end least_value_of_a_l68_68914


namespace find_pairs_l68_68520

-- Define the problem conditions
def equation (n k : ℕ) : Prop := nat.factorial n + n = n ^ k

-- Define the positive integer property
def positive (n : ℕ) : Prop := n > 0

-- State the goal of the theorem
theorem find_pairs : ∀ (n k : ℕ), positive n → positive k → equation n k ↔ 
  (n = 2 ∧ k = 2) ∨ (n = 3 ∧ k = 2) ∨ (n = 5 ∧ k = 3) :=
by
  intros n k hn hk
  sorry

end find_pairs_l68_68520


namespace min_sum_one_over_xy_l68_68470

theorem min_sum_one_over_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 6) : 
  ∃ c, (∀ x y, (x > 0) → (y > 0) → (x + y = 6) → (c ≤ (1/x + 1/y))) ∧ (c = 2 / 3) :=
by 
  sorry

end min_sum_one_over_xy_l68_68470


namespace number_of_girls_l68_68500

theorem number_of_girls
  (total_boys : ℕ)
  (total_boys_eq : total_boys = 10)
  (fraction_girls_reading : ℚ)
  (fraction_girls_reading_eq : fraction_girls_reading = 5/6)
  (fraction_boys_reading : ℚ)
  (fraction_boys_reading_eq : fraction_boys_reading = 4/5)
  (total_not_reading : ℕ)
  (total_not_reading_eq : total_not_reading = 4)
  (G : ℝ)
  (remaining_girls_reading : (1 - fraction_girls_reading) * G = 2)
  (remaining_boys_not_reading : (1 - fraction_boys_reading) * total_boys = 2)
  (remaining_total_not_reading : 2 + 2 = total_not_reading)
  : G = 12 :=
by
  sorry

end number_of_girls_l68_68500


namespace cost_of_five_dozen_l68_68705

noncomputable def price_per_dozen (total_cost : ℝ) (num_dozen : ℕ) : ℝ :=
  total_cost / num_dozen

noncomputable def total_cost (price_per_dozen : ℝ) (num_dozen : ℕ) : ℝ :=
  price_per_dozen * num_dozen

theorem cost_of_five_dozen (total_cost_threedozens : ℝ := 28.20) (num_threedozens : ℕ := 3) (num_fivedozens : ℕ := 5) :
  total_cost (price_per_dozen total_cost_threedozens num_threedozens) num_fivedozens = 47.00 :=
  by sorry

end cost_of_five_dozen_l68_68705


namespace yellow_marbles_l68_68032

-- Define the conditions from a)
variables (total_marbles red blue green yellow : ℕ)
variables (h1 : total_marbles = 110)
variables (h2 : red = 8)
variables (h3 : blue = 4 * red)
variables (h4 : green = 2 * blue)
variables (h5 : yellow = total_marbles - (red + blue + green))

-- Prove the question in c)
theorem yellow_marbles : yellow = 6 :=
by
  -- Proof will be inserted here
  sorry

end yellow_marbles_l68_68032


namespace number_of_workers_who_read_all_three_books_l68_68560

theorem number_of_workers_who_read_all_three_books
  (W S K A SK SA KA SKA N : ℝ)
  (hW : W = 75)
  (hS : S = 1 / 2 * W)
  (hK : K = 1 / 4 * W)
  (hA : A = 1 / 5 * W)
  (hSK : SK = 2 * SKA)
  (hN : N = S - (SK + SA + SKA) - 1)
  (hTotal : S + K + A - (SK + SA + KA - SKA) + N = W) :
  SKA = 6 :=
by
  -- The proof steps are omitted
  sorry

end number_of_workers_who_read_all_three_books_l68_68560


namespace candles_to_new_five_oz_l68_68689

theorem candles_to_new_five_oz 
  (h_wax_percent: ℝ)
  (h_candles_20oz_count: ℕ) 
  (h_candles_5oz_count: ℕ) 
  (h_candles_1oz_count: ℕ) 
  (h_candles_20oz_wax: ℝ) 
  (h_candles_5oz_wax: ℝ)
  (h_candles_1oz_wax: ℝ):
  h_wax_percent = 0.10 →
  h_candles_20oz_count = 5 →
  h_candles_5oz_count = 5 → 
  h_candles_1oz_count = 25 →
  h_candles_20oz_wax = 20 →
  h_candles_5oz_wax = 5 →
  h_candles_1oz_wax = 1 →
  (h_wax_percent * h_candles_20oz_wax * h_candles_20oz_count + 
   h_wax_percent * h_candles_5oz_wax * h_candles_5oz_count + 
   h_wax_percent * h_candles_1oz_wax * h_candles_1oz_count) / 5 = 3 :=
by
  sorry

end candles_to_new_five_oz_l68_68689


namespace least_five_digit_whole_number_is_perfect_square_and_cube_l68_68267

theorem least_five_digit_whole_number_is_perfect_square_and_cube :
  ∃ (n : ℕ), (10000 ≤ n ∧ n < 100000) ∧ (∃ (a : ℕ), n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_whole_number_is_perfect_square_and_cube_l68_68267


namespace number_of_sets_B_l68_68678

theorem number_of_sets_B (A : Set ℕ) (hA : A = {1, 2}) :
    ∃ (n : ℕ), n = 4 ∧ (∀ B : Set ℕ, A ∪ B = {1, 2} → B ⊆ A) := sorry

end number_of_sets_B_l68_68678


namespace skipping_ropes_l68_68488

theorem skipping_ropes (length1 length2 : ℕ) (h1 : length1 = 18) (h2 : length2 = 24) :
  ∃ (max_length : ℕ) (num_ropes : ℕ),
    max_length = Nat.gcd length1 length2 ∧
    max_length = 6 ∧
    num_ropes = length1 / max_length + length2 / max_length ∧
    num_ropes = 7 :=
by
  have max_length : ℕ := Nat.gcd length1 length2
  have num_ropes : ℕ := length1 / max_length + length2 / max_length
  use max_length, num_ropes
  sorry

end skipping_ropes_l68_68488


namespace monotonic_intervals_of_f_min_value_of_h_l68_68813

open Real

noncomputable def f (x : ℝ) := exp x * sin x
noncomputable def g (x : ℝ) := x * exp x
noncomputable def h (x : ℝ) := f x / g x

theorem monotonic_intervals_of_f (k : ℤ) : 
  let incr_intervals := [2 * k * π - π / 4, 2 * k * π + 3 * π / 4]
  let decr_intervals := [2 * k * π - 5 * π / 4, 2 * k * π - π / 4]
  f' x ≥ 0 ↔ x ∈ incr_intervals ∧ f' x < 0 ↔ x ∈ decr_intervals := 
sorry

theorem min_value_of_h : 
  ∀ x ∈ Ioo 0 (π / 2), h x ≥ (2 / π) :=
sorry

end monotonic_intervals_of_f_min_value_of_h_l68_68813


namespace median_ratio_within_bounds_l68_68409

def median_ratio_limits (α : ℝ) (hα : 0 < α ∧ α < π) : Prop :=
  ∀ (s_c s_b : ℝ), s_b = 1 → (1 / 2) ≤ (s_c / s_b) ∧ (s_c / s_b) ≤ 2

theorem median_ratio_within_bounds (α : ℝ) (hα : 0 < α ∧ α < π) : 
  median_ratio_limits α hα :=
by
  sorry

end median_ratio_within_bounds_l68_68409


namespace least_possible_value_l68_68026

theorem least_possible_value (y q p : ℝ) (h1: 5 < y) (h2: y < 7)
  (hq: q = 7) (hp: p = 5) : q - p = 2 :=
by
  sorry

end least_possible_value_l68_68026


namespace xy_sum_of_squares_l68_68146

theorem xy_sum_of_squares (x y : ℝ) (h1 : x - y = 5) (h2 : -x * y = 4) : x^2 + y^2 = 17 := 
sorry

end xy_sum_of_squares_l68_68146


namespace books_at_end_of_year_l68_68046

def init_books : ℕ := 72
def monthly_books : ℕ := 12 -- 1 book each month for 12 months
def books_bought1 : ℕ := 5
def books_bought2 : ℕ := 2
def books_gift1 : ℕ := 1
def books_gift2 : ℕ := 4
def books_donated : ℕ := 12
def books_sold : ℕ := 3

theorem books_at_end_of_year :
  init_books + monthly_books + books_bought1 + books_bought2 + books_gift1 + books_gift2 - books_donated - books_sold = 81 :=
by
  sorry

end books_at_end_of_year_l68_68046


namespace calories_per_person_l68_68431

-- Definitions based on the conditions from a)
def oranges : ℕ := 5
def pieces_per_orange : ℕ := 8
def people : ℕ := 4
def calories_per_orange : ℝ := 80

-- Theorem based on the equivalent proof problem
theorem calories_per_person : 
    ((oranges * pieces_per_orange) / people) / pieces_per_orange * calories_per_orange = 100 := 
by
  sorry

end calories_per_person_l68_68431


namespace part1_part2_l68_68550

variable (a b c : ℝ)

-- Conditions
axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : c > 0
axiom h4 : a^2 + b^2 + 4*c^2 = 3

-- Part 1: Prove that a + b + 2c ≤ 3
theorem part1 : a + b + 2*c ≤ 3 := sorry

-- Part 2: Given b = 2c, prove that 1/a + 1/c ≥ 3
axiom h5 : b = 2*c
theorem part2 : 1/a + 1/c ≥ 3 := sorry

end part1_part2_l68_68550


namespace least_five_digit_perfect_square_cube_l68_68271

theorem least_five_digit_perfect_square_cube :
  ∃ n : Nat, (10000 ≤ n ∧ n < 100000) ∧ (∃ m1 m2 : Nat, n = m1^2 ∧ n = m2^3 ∧ n = 15625) :=
by
  sorry

end least_five_digit_perfect_square_cube_l68_68271


namespace value_of_f_5_l68_68083

theorem value_of_f_5 (f : ℕ → ℕ) (y : ℕ)
  (h1 : ∀ x, f x = 2 * x^2 + y)
  (h2 : f 2 = 20) : f 5 = 62 :=
sorry

end value_of_f_5_l68_68083


namespace wire_length_unique_l68_68473

noncomputable def distance_increment := (5 / 3)

theorem wire_length_unique (d L : ℝ) 
  (h1 : L = 25 * d) 
  (h2 : L = 24 * (d + distance_increment)) :
  L = 1000 := by
  sorry

end wire_length_unique_l68_68473


namespace chapter_page_difference_l68_68341

/-- The first chapter of a book has 37 pages -/
def first_chapter_pages : Nat := 37

/-- The second chapter of a book has 80 pages -/
def second_chapter_pages : Nat := 80

/-- Prove the difference in the number of pages between the second and the first chapter is 43 -/
theorem chapter_page_difference : (second_chapter_pages - first_chapter_pages) = 43 := by
  sorry

end chapter_page_difference_l68_68341


namespace least_sum_of_exponents_l68_68490

theorem least_sum_of_exponents (a b c d e : ℕ) (h : ℕ) (h_divisors : 225 ∣ h ∧ 216 ∣ h ∧ 847 ∣ h)
  (h_form : h = (2 ^ a) * (3 ^ b) * (5 ^ c) * (7 ^ d) * (11 ^ e)) : 
  a + b + c + d + e = 10 :=
sorry

end least_sum_of_exponents_l68_68490


namespace residue_625_mod_17_l68_68373

theorem residue_625_mod_17 : 625 % 17 = 13 :=
by
  sorry

end residue_625_mod_17_l68_68373


namespace circles_intersect_and_inequality_l68_68747

variable {R r d : ℝ}

theorem circles_intersect_and_inequality (hR : R > r) (h_intersect: R - r < d ∧ d < R + r) : R - r < d ∧ d < R + r :=
by
  exact h_intersect

end circles_intersect_and_inequality_l68_68747


namespace product_multiple_of_3_probability_l68_68744

theorem product_multiple_of_3_probability :
  let s : Finset ℕ := {1, 2, 3, 4, 5, 6}
  let total_choices := (Finset.card s).choose 3
  let non_multiples_of_3 := {1, 2, 4, 5}
  let non_multiples_choices := (Finset.card non_multiples_of_3).choose 3
  (1 - non_multiples_choices / total_choices) = 4 / 5 :=
by
  let s : Finset ℕ := {1, 2, 3, 4, 5, 6}
  let total_choices := (Finset.card s).choose 3
  let non_multiples_of_3 := {1, 2, 4, 5}
  let non_multiples_choices := (Finset.card non_multiples_of_3).choose 3
  have h : (1 - non_multiples_choices / total_choices) = 4 / 5 := sorry
  exact h

end product_multiple_of_3_probability_l68_68744


namespace vasya_wins_l68_68204

/-
  Petya and Vasya are playing a game where initially there are 2022 boxes, 
  each containing exactly one matchstick. In one move, a player can transfer 
  all matchsticks from one non-empty box to another non-empty box. They take turns, 
  with Petya starting first. The winner is the one who, after their move, has 
  at least half of all the matchsticks in one box for the first time. 

  We want to prove that Vasya will win the game with the optimal strategy.
-/

theorem vasya_wins : true :=
  sorry -- placeholder for the actual proof

end vasya_wins_l68_68204


namespace cos_pi_over_6_minus_2alpha_l68_68197

open Real

noncomputable def tan_plus_pi_over_6 (α : ℝ) := tan (α + π / 6) = 2

theorem cos_pi_over_6_minus_2alpha (α : ℝ) 
  (h1 : π < α ∧ α < 2 * π) 
  (h2 : tan_plus_pi_over_6 α) : 
  cos (π / 6 - 2 * α) = 4 / 5 :=
sorry

end cos_pi_over_6_minus_2alpha_l68_68197


namespace quadratic_unique_real_root_l68_68176

theorem quadratic_unique_real_root (m : ℝ) :
  (∀ x : ℝ, x^2 + 6 * m * x + 2 * m = 0 → ∃! r : ℝ, x = r) → m = 2/9 :=
by
  sorry

end quadratic_unique_real_root_l68_68176


namespace minimum_workers_in_team_A_l68_68004

variable (a b c : ℤ)

theorem minimum_workers_in_team_A (h1 : b + 90 = 2 * (a - 90))
                               (h2 : a + c = 6 * (b - c)) :
  ∃ a ≥ 148, a = 153 :=
by
  sorry

end minimum_workers_in_team_A_l68_68004


namespace convert_500_to_base5_l68_68117

def base10_to_base5 (n : ℕ) : ℕ :=
  -- A function to convert base 10 to base 5 would be defined here
  sorry

theorem convert_500_to_base5 : base10_to_base5 500 = 4000 := 
by 
  -- The actual proof would go here
  sorry

end convert_500_to_base5_l68_68117


namespace remainder_mod_5_l68_68242

theorem remainder_mod_5 :
  let a := 1492
  let b := 1776
  let c := 1812
  let d := 1996
  (a * b * c * d) % 5 = 4 :=
by
  sorry

end remainder_mod_5_l68_68242


namespace least_five_digit_perfect_square_and_cube_l68_68324

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k1 : ℕ, n = k1 ^ 2) ∧ (∃ k2 : ℕ, n = k2 ^ 3) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68324


namespace least_number_remainder_seven_exists_l68_68326

theorem least_number_remainder_seven_exists :
  ∃ x : ℕ, x ≡ 7 [MOD 11] ∧ x ≡ 7 [MOD 17] ∧ x ≡ 7 [MOD 21] ∧ x ≡ 7 [MOD 29] ∧ x ≡ 7 [MOD 35] ∧ 
           x ≡ 1547 [MOD Nat.lcm 11 (Nat.lcm 17 (Nat.lcm 21 (Nat.lcm 29 35)))] :=
  sorry

end least_number_remainder_seven_exists_l68_68326


namespace three_layers_rug_area_l68_68634

theorem three_layers_rug_area :
  ∀ (A B C D E : ℝ),
    A + B + C = 212 →
    (A + B + C) - D - 2 * E = 140 →
    D = 24 →
    E = 24 :=
by
  intros A B C D E h1 h2 h3
  sorry

end three_layers_rug_area_l68_68634


namespace original_price_l68_68101

theorem original_price (P : ℝ) (h : P * 0.80 = 960) : P = 1200 :=
sorry

end original_price_l68_68101


namespace train_speed_kmph_l68_68351

noncomputable def speed_of_train
  (train_length : ℝ) (bridge_cross_time : ℝ) (total_length : ℝ) : ℝ :=
  (total_length / bridge_cross_time) * 3.6

theorem train_speed_kmph
  (train_length : ℝ := 130) 
  (bridge_cross_time : ℝ := 30) 
  (total_length : ℝ := 245) : 
  speed_of_train train_length bridge_cross_time total_length = 29.4 := by
  sorry

end train_speed_kmph_l68_68351


namespace train_speeds_l68_68726

noncomputable def c1 : ℝ := sorry  -- speed of the passenger train in km/min
noncomputable def c2 : ℝ := sorry  -- speed of the freight train in km/min
noncomputable def c3 : ℝ := sorry  -- speed of the express train in km/min

def conditions : Prop :=
  (5 / c1 + 5 / c2 = 15) ∧
  (5 / c2 + 5 / c3 = 11) ∧
  (c2 ≤ c1) ∧
  (c3 ≤ 2.5)

-- The theorem to be proved
theorem train_speeds :
  conditions →
  (40 / 60 ≤ c1 ∧ c1 ≤ 50 / 60) ∧ 
  (100 / 3 / 60 ≤ c2 ∧ c2 ≤ 40 / 60) ∧ 
  (600 / 7 / 60 ≤ c3 ∧ c3 ≤ 150 / 60) :=
sorry

end train_speeds_l68_68726


namespace oldest_sibling_age_difference_l68_68611

theorem oldest_sibling_age_difference 
  (D : ℝ) 
  (avg_age : ℝ) 
  (hD : D = 25.75) 
  (h_avg : avg_age = 30) :
  ∃ A : ℝ, (A - D ≥ 17) :=
by
  sorry

end oldest_sibling_age_difference_l68_68611


namespace car_win_probability_l68_68829

noncomputable def P (n : ℕ) : ℚ := 1 / n

theorem car_win_probability :
  let P_x := 1 / 7
  let P_y := 1 / 3
  let P_z := 1 / 5
  P_x + P_y + P_z = 71 / 105 :=
by
  sorry

end car_win_probability_l68_68829


namespace art_collection_area_l68_68367

theorem art_collection_area :
  let square_paintings := 3 * (6 * 6)
  let small_paintings := 4 * (2 * 3)
  let large_painting := 1 * (10 * 15)
  square_paintings + small_paintings + large_painting = 282 := by
  sorry

end art_collection_area_l68_68367


namespace limes_left_l68_68513

-- Define constants
def num_limes_initial : ℕ := 9
def num_limes_given : ℕ := 4

-- Theorem to be proved
theorem limes_left : num_limes_initial - num_limes_given = 5 :=
by
  sorry

end limes_left_l68_68513


namespace isosceles_trapezoid_ratio_ab_cd_l68_68588

theorem isosceles_trapezoid_ratio_ab_cd (AB CD : ℝ) (P : ℝ → ℝ → Prop)
  (area1 area2 area3 area4 : ℝ)
  (h1 : AB > CD)
  (h2 : area1 = 5)
  (h3 : area2 = 7)
  (h4 : area3 = 3)
  (h5 : area4 = 9) :
  AB / CD = 1 + 2 * Real.sqrt 2 :=
sorry

end isosceles_trapezoid_ratio_ab_cd_l68_68588


namespace linear_function_decreasing_iff_l68_68122

-- Define the conditions
def linear_function (m b x : ℝ) : ℝ := m * x + b

-- Define the condition for decreasing function
def is_decreasing (f : ℝ → ℝ) := ∀ x1 x2 : ℝ, x1 < x2 → f x1 ≥ f x2

-- The theorem to prove
theorem linear_function_decreasing_iff (m b : ℝ) :
  (is_decreasing (linear_function m b)) ↔ (m < 0) :=
by
  sorry

end linear_function_decreasing_iff_l68_68122


namespace distribution_X_company_receives_rewards_l68_68780

noncomputable def binom_p := 4
noncomputable def binom_q := 1/2

def xi : ℕ → Prop := {n | n ≤ 2}

def P_xi_lt_3 := 11 / 16
def P_xi_geq_3 := 5 / 16

def p_1 := 1
def p_n (n : ℕ) := ∑ i in (Finset.range n), (xi i)
def p_geq_half (n : ℕ) := (1 / 2) * (3 / 8)^(n-1) + (1 / 2) > 1 / 2

theorem distribution_X :
  ∀ (X : ℕ),
  (X = 1 → P_xi_geq_3 * P_xi_lt_3 = 55 / 256) ∧
  (X = 2 → (P_xi_lt_3 * P_xi_geq_3 + P_xi_geq_3 * P_xi_geq_3 = 5 / 16)) ∧
  (X = 3 → P_xi_lt_3 * P_xi_lt_3 = 121 / 256) :=
by sorry

theorem company_receives_rewards :
  ∀ n : ℕ, n > 0 → p_geq_half n :=
by sorry

end distribution_X_company_receives_rewards_l68_68780


namespace double_acute_angle_l68_68546

theorem double_acute_angle (θ : ℝ) (h : 0 < θ ∧ θ < 90) : 0 < 2 * θ ∧ 2 * θ < 180 :=
by
  sorry

end double_acute_angle_l68_68546


namespace yogurt_calories_per_ounce_l68_68756

variable (calories_strawberries_per_unit : ℕ)
variable (calories_yogurt_total : ℕ)
variable (calories_total : ℕ)
variable (strawberries_count : ℕ)
variable (yogurt_ounces_count : ℕ)

theorem yogurt_calories_per_ounce (h1: strawberries_count = 12)
                                   (h2: yogurt_ounces_count = 6)
                                   (h3: calories_strawberries_per_unit = 4)
                                   (h4: calories_total = 150)
                                   (h5: calories_yogurt_total = calories_total - strawberries_count * calories_strawberries_per_unit):
                                   calories_yogurt_total / yogurt_ounces_count = 17 :=
by
  -- We conjecture that this is correct based on given conditions.
  sorry

end yogurt_calories_per_ounce_l68_68756


namespace stratified_sampling_third_year_students_l68_68580

theorem stratified_sampling_third_year_students :
  let total_students := 900
  let first_year_students := 300
  let second_year_students := 200
  let third_year_students := 400
  let sample_size := 45
  let sampling_ratio := (sample_size : ℚ) / (total_students : ℚ)
  (third_year_students : ℚ) * sampling_ratio = 20 :=
by 
  let total_students := 900
  let first_year_students := 300
  let second_year_students := 200
  let third_year_students := 400
  let sample_size := 45
  let sampling_ratio := (sample_size : ℚ) / (total_students : ℚ)
  show (third_year_students : ℚ) * sampling_ratio = 20
  sorry

end stratified_sampling_third_year_students_l68_68580


namespace banana_cream_pie_correct_slice_l68_68427

def total_students := 45
def strawberry_pie_preference := 15
def pecan_pie_preference := 10
def pumpkin_pie_preference := 9

noncomputable def banana_cream_pie_slice_degrees : ℝ :=
  let remaining_students := total_students - strawberry_pie_preference - pecan_pie_preference - pumpkin_pie_preference
  let students_per_preference := remaining_students / 2
  (students_per_preference / total_students) * 360

theorem banana_cream_pie_correct_slice :
  banana_cream_pie_slice_degrees = 44 := by
  sorry

end banana_cream_pie_correct_slice_l68_68427


namespace largest_real_number_l68_68394

theorem largest_real_number (x : ℝ) (h : ⌊x⌋ / x = 8 / 9) : x ≤ 63 / 8 :=
sorry

end largest_real_number_l68_68394


namespace building_height_270_l68_68834

theorem building_height_270 :
  ∀ (total_stories first_partition_height additional_height_per_story : ℕ), 
  total_stories = 20 → 
  first_partition_height = 12 → 
  additional_height_per_story = 3 →
  let first_partition_stories := 10 in
  let remaining_partition_stories := total_stories - first_partition_stories in
  let first_partition_total_height := first_partition_stories * first_partition_height in
  let remaining_story_height := first_partition_height + additional_height_per_story in
  let remaining_partition_total_height := remaining_partition_stories * remaining_story_height in
  first_partition_total_height + remaining_partition_total_height = 270 :=
by
  intros total_stories first_partition_height additional_height_per_story h_total_stories h_first_height h_additional_height
  let first_partition_stories := 10
  let remaining_partition_stories := total_stories - first_partition_stories
  let first_partition_total_height := first_partition_stories * first_partition_height
  let remaining_story_height := first_partition_height + additional_height_per_story
  let remaining_partition_total_height := remaining_partition_stories * remaining_story_height
  have h_total_height : first_partition_total_height + remaining_partition_total_height = 270 := sorry
  exact h_total_height

end building_height_270_l68_68834


namespace total_parallelograms_in_grid_l68_68525

theorem total_parallelograms_in_grid (n : ℕ) : 
  ∃ p : ℕ, p = 3 * Nat.choose (n + 2) 4 :=
sorry

end total_parallelograms_in_grid_l68_68525


namespace probability_AB_hired_l68_68425

theorem probability_AB_hired (A B C D E : Type)
  (h : Finset {A, B, C, D, E} → Finset UnorderedTriple A, B, C, D, E)
  (p : ∀ x ∈ {A, B, C, D, E}, 1 / 5)
  : (∀ s ∈ Finset.powerset_len 3 ({A, B, C, D, E} : Finset Type), 
  P(s.contains A ∨ s.contains B) = (9 / 10))
:=
  sorry

end probability_AB_hired_l68_68425


namespace matrix_problem_l68_68591

variable (A B : Matrix (Fin 2) (Fin 2) ℝ)
variable (I : Matrix (Fin 2) (Fin 2) ℝ)

theorem matrix_problem 
  (h1 : A + B = A * B)
  (h2 : A * B = !![2, 1; 4, 3]) :
  B * A = !![2, 1; 4, 3] :=
sorry

end matrix_problem_l68_68591


namespace split_coins_l68_68184

theorem split_coins (p n d q : ℕ) (hp : p % 5 = 0) 
  (h_total : p + 5 * n + 10 * d + 25 * q = 10000) :
  ∃ (p1 n1 d1 q1 p2 n2 d2 q2 : ℕ),
    (p1 + 5 * n1 + 10 * d1 + 25 * q1 = 5000) ∧
    (p2 + 5 * n2 + 10 * d2 + 25 * q2 = 5000) ∧
    (p = p1 + p2) ∧ (n = n1 + n2) ∧ (d = d1 + d2) ∧ (q = q1 + q2) :=
sorry

end split_coins_l68_68184


namespace correct_transformation_l68_68484

theorem correct_transformation (a b : ℝ) (h : a ≠ 0) : (b / a = (a * b) / (a ^ 2)) :=
begin
  sorry
end

end correct_transformation_l68_68484


namespace f_g_2_eq_256_l68_68961

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 3 * x^2 + 4

theorem f_g_2_eq_256 : f (g 2) = 256 := by
  sorry

end f_g_2_eq_256_l68_68961


namespace sin_half_cos_half_l68_68804

theorem sin_half_cos_half (θ : ℝ) (h_cos: Real.cos θ = -3/5) (h_range: Real.pi < θ ∧ θ < 3/2 * Real.pi) :
  Real.sin (θ/2) + Real.cos (θ/2) = Real.sqrt 5 / 5 := 
by 
  sorry

end sin_half_cos_half_l68_68804


namespace division_quotient_difference_l68_68913

theorem division_quotient_difference :
  (32.5 / 1.3) - (60.8 / 7.6) = 17 :=
by
  sorry

end division_quotient_difference_l68_68913


namespace Adam_final_amount_l68_68896

def initial_amount : ℝ := 5.25
def spent_on_game : ℝ := 2.30
def spent_on_snacks : ℝ := 1.75
def found_dollar : ℝ := 1.00
def allowance : ℝ := 5.50

theorem Adam_final_amount :
  (initial_amount - spent_on_game - spent_on_snacks + found_dollar + allowance) = 7.70 :=
by
  sorry

end Adam_final_amount_l68_68896


namespace cost_of_article_l68_68015

noncomputable def find_cost_of_article (C G : ℝ) (h1 : C + G = 240) (h2 : C + 1.12 * G = 320) : Prop :=
  C = 168.57

theorem cost_of_article (C G : ℝ) (h1 : C + G = 240) (h2 : C + 1.12 * G = 320) : 
  find_cost_of_article C G h1 h2 :=
by
  sorry

end cost_of_article_l68_68015


namespace least_five_digit_perfect_square_and_cube_l68_68316

/-- 
Prove that the least five-digit whole number that is both a perfect square 
and a perfect cube is 15625.
-/
theorem least_five_digit_perfect_square_and_cube : 
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k : ℕ, n = k^6) ∧ (∀ m : ℕ, (10000 ≤ m ∧ m < 100000) ∧ (∃ l : ℕ, m = l^6) → n ≤ m) :=
begin
  use 15625,
  split,
  { split,
    { norm_num },
    { norm_num } },
  { use 5,
    norm_num,
    intros m hm,
    rcases hm with ⟨⟨hm1, hm2⟩, ⟨l, rfl⟩⟩,
    cases le_or_lt l 5 with hl hl,
    { by_contradiction h,
      have : m < 10000 := by norm_num [nat.pow_succ, nat.pow_succ, hl],
      exact not_le_of_gt this hm1 },
    { have : l ≥ 6 := nat.lt_succ_iff.mp hl,
      have : m ≥ 6^6 := nat.pow_le_pow_of_le_right nat.zero_lt_succ_iff.1 this,
      norm_num at this,
      exact not_lt_of_ge this hm2 } }
end

end least_five_digit_perfect_square_and_cube_l68_68316


namespace least_number_subtracted_l68_68401

theorem least_number_subtracted (n : ℕ) (x : ℕ) (h_n : n = 4273981567) (h_x : x = 17) : 
  (n - x) % 25 = 0 := by
  sorry

end least_number_subtracted_l68_68401


namespace at_least_one_non_zero_l68_68238

theorem at_least_one_non_zero (a b : ℝ) : a^2 + b^2 > 0 ↔ (a ≠ 0 ∨ b ≠ 0) :=
by sorry

end at_least_one_non_zero_l68_68238


namespace person_B_reads_more_than_A_l68_68846

-- Assuming people are identifiers for Person A and Person B.
def pages_read_A (days : ℕ) (daily_read : ℕ) : ℕ := days * daily_read

def pages_read_B (days : ℕ) (daily_read : ℕ) (rest_cycle : ℕ) : ℕ := 
  let full_cycles := days / rest_cycle
  let remainder_days := days % rest_cycle
  let active_days := days - full_cycles
  active_days * daily_read

-- Given conditions
def daily_read_A := 8
def daily_read_B := 13
def rest_cycle_B := 3
def total_days := 7

-- The main theorem to prove
theorem person_B_reads_more_than_A : 
  (pages_read_B total_days daily_read_B rest_cycle_B) - (pages_read_A total_days daily_read_A) = 9 :=
by
  sorry

end person_B_reads_more_than_A_l68_68846


namespace new_team_average_weight_l68_68762

theorem new_team_average_weight :
  let original_team_weight := 7 * 94
  let new_players_weight := 110 + 60
  let new_total_weight := original_team_weight + new_players_weight
  let new_player_count := 9
  (new_total_weight / new_player_count) = 92 :=
by
  let original_team_weight := 7 * 94
  let new_players_weight := 110 + 60
  let new_total_weight := original_team_weight + new_players_weight
  let new_player_count := 9
  sorry

end new_team_average_weight_l68_68762


namespace max_non_overlapping_areas_l68_68643

theorem max_non_overlapping_areas (n : ℕ) (h : 0 < n) :
  ∃ k : ℕ, k = 4 * n + 1 :=
sorry

end max_non_overlapping_areas_l68_68643


namespace total_valid_votes_l68_68699

theorem total_valid_votes (V : ℝ) (h1 : 0.70 * V - 0.30 * V = 176) : V = 440 :=
by sorry

end total_valid_votes_l68_68699


namespace tan_subtraction_formula_l68_68572

theorem tan_subtraction_formula 
  (α β : ℝ) (hα : Real.tan α = 3) (hβ : Real.tan β = 2) :
  Real.tan (α - β) = 1 / 7 := 
by
  sorry

end tan_subtraction_formula_l68_68572


namespace tan_triple_angle_l68_68954

variable θ : ℝ
variable h : Real.tan θ = 3

theorem tan_triple_angle (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 :=
sorry

end tan_triple_angle_l68_68954


namespace apples_preference_count_l68_68831

theorem apples_preference_count (total_people : ℕ) (total_angle : ℝ) (apple_angle : ℝ) 
  (h_total_people : total_people = 530) 
  (h_total_angle : total_angle = 360) 
  (h_apple_angle : apple_angle = 285) : 
  round ((total_people : ℝ) * (apple_angle / total_angle)) = 419 := 
by 
  sorry

end apples_preference_count_l68_68831


namespace berry_circle_properties_l68_68340

theorem berry_circle_properties :
  ∃ r : ℝ, (∀ x y : ℝ, x^2 + y^2 - 12 = 2 * x + 4 * y → r = Real.sqrt 17)
    ∧ (π * Real.sqrt 17 ^ 2 > 30) :=
by
  sorry

end berry_circle_properties_l68_68340


namespace simple_interest_rate_l68_68575

theorem simple_interest_rate (P R: ℝ) (T : ℝ) (hT : T = 8) (h : 2 * P = P + (P * R * T) / 100) : R = 12.5 :=
by
  -- Placeholder for proof steps
  sorry

end simple_interest_rate_l68_68575


namespace eq_has_one_integral_root_l68_68069

theorem eq_has_one_integral_root :
  ∀ x : ℝ, (x - (9 / (x - 5)) = 4 - (9 / (x-5))) → x = 4 := by
  intros x h
  sorry

end eq_has_one_integral_root_l68_68069


namespace cost_per_person_l68_68457

def total_cost : ℕ := 30000  -- Cost in million dollars
def num_people : ℕ := 300    -- Number of people in million

theorem cost_per_person : total_cost / num_people = 100 :=
by
  sorry

end cost_per_person_l68_68457


namespace surface_area_of_cube_l68_68604

theorem surface_area_of_cube (edge : ℝ) (h : edge = 5) : 6 * (edge * edge) = 150 := by
  have h_square : edge * edge = 25 := by
    rw [h]
    norm_num
  rw [h_square]
  norm_num

end surface_area_of_cube_l68_68604


namespace total_students_in_school_l68_68579

theorem total_students_in_school (C1 C2 C3 C4 C5 : ℕ) 
  (h1 : C1 = 23)
  (h2 : C2 = C1 - 2)
  (h3 : C3 = C2 - 2)
  (h4 : C4 = C3 - 2)
  (h5 : C5 = C4 - 2)
  : C1 + C2 + C3 + C4 + C5 = 95 := 
by 
  -- proof details skipped with sorry
  sorry

end total_students_in_school_l68_68579


namespace probability_exactly_one_of_A_or_B_selected_l68_68540

-- We define a set of four people
inductive Person
| A | B | C | D

open Person

-- Define the event of selecting exactly one of A and B
def exactlyOneOfABSelected (s : set (Person × Person)) : Prop :=
  (⟨A, C⟩ ∈ s ∨ ⟨A, D⟩ ∈ s ∨ ⟨B, C⟩ ∈ s ∨ ⟨B, D⟩ ∈ s) ∧
  (⟨A, B⟩ ∉ s ∧ ⟨C, D⟩ ∉ s)

-- Define the universal set of all two-person combinations
def allCombinations : set (Person × Person) :=
  {⟨A, B⟩, ⟨A, C⟩, ⟨A, D⟩, ⟨B, C⟩, ⟨B, D⟩, ⟨C, D⟩}

theorem probability_exactly_one_of_A_or_B_selected : 
  (∃ (s : set (Person × Person)), exactlyOneOfABSelected s) → 
  (s.card = 4 / 6) :=
sorry

end probability_exactly_one_of_A_or_B_selected_l68_68540


namespace jim_anne_mary_paul_report_time_l68_68748

def typing_rate_jim := 1 / 12
def typing_rate_anne := 1 / 20
def combined_typing_rate := typing_rate_jim + typing_rate_anne
def typing_time := 1 / combined_typing_rate

def editing_rate_mary := 1 / 30
def editing_rate_paul := 1 / 10
def combined_editing_rate := editing_rate_mary + editing_rate_paul
def editing_time := 1 / combined_editing_rate

theorem jim_anne_mary_paul_report_time : 
  typing_time + editing_time = 15 := by
  sorry

end jim_anne_mary_paul_report_time_l68_68748


namespace solution_to_diff_eq_l68_68055

variable {C : ℝ} {x y : ℝ}

-- Defining the differential equation conditions
def differential_eq (x y : ℝ) : Prop :=
  2 * x * y * Real.log y * (deriv fun _ => x) + (x^2 + y^2 * Real.sqrt (y^2 + 1)) * (deriv fun _ => y) = 0

-- Defining the candidate solution
def candidate_solution (x y : ℝ) (C : ℝ) : Prop :=
  3 * x^2 * Real.log y + Real.sqrt ((y^2 + 1)^3) = C

-- Proof problem statement
theorem solution_to_diff_eq (x y : ℝ) (C : ℝ) :
  differential_eq x y → candidate_solution x y C :=
sorry

end solution_to_diff_eq_l68_68055


namespace census_entirety_is_population_l68_68733

-- Define the options as a type
inductive CensusOptions
| Part
| Whole
| Individual
| Population

-- Define the condition: the entire object under investigation in a census
def entirety_of_objects_under_investigation : CensusOptions := CensusOptions.Population

-- Prove that the entirety of objects under investigation in a census is called Population
theorem census_entirety_is_population :
  entirety_of_objects_under_investigation = CensusOptions.Population :=
sorry

end census_entirety_is_population_l68_68733


namespace monthly_incomes_l68_68743

theorem monthly_incomes (a b c d e : ℕ) : 
  a + b = 8100 ∧ 
  b + c = 10500 ∧ 
  a + c = 8400 ∧
  (a + b + d) / 3 = 4800 ∧
  (c + d + e) / 3 = 6000 ∧
  (b + a + e) / 3 = 4500 → 
  (a = 3000 ∧ b = 5100 ∧ c = 5400 ∧ d = 6300 ∧ e = 5400) :=
by sorry

end monthly_incomes_l68_68743


namespace find_k_intersects_parabola_at_one_point_l68_68233

theorem find_k_intersects_parabola_at_one_point :
  ∃ k : ℝ, (∀ y : ℝ, -3 * y^2 - 4 * y + 7 = k ↔ y = (-4 / (2 * 3))) →
    k = 25 / 3 :=
by sorry

end find_k_intersects_parabola_at_one_point_l68_68233


namespace equilateral_is_peculiar_rt_triangle_is_peculiar_peculiar_rt_triangle_ratio_l68_68208

-- Definition of a peculiar triangle.
def is_peculiar_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = 2 * c^2

-- Problem 1: Proving an equilateral triangle is a peculiar triangle
theorem equilateral_is_peculiar (a : ℝ) : is_peculiar_triangle a a a :=
sorry

-- Problem 2: Proving the case when b is the hypotenuse in Rt△ABC makes it peculiar
theorem rt_triangle_is_peculiar (a b c : ℝ) (ha : a = 5 * Real.sqrt 2) (hc : c = 10) : 
  is_peculiar_triangle a b c ↔ b = Real.sqrt (c^2 + a^2) :=
sorry

-- Problem 3: Proving the ratio of the sides in a peculiar right triangle is 1 : √2 : √3
theorem peculiar_rt_triangle_ratio (a b c : ℝ) (hc : c^2 = a^2 + b^2) (hpeculiar : is_peculiar_triangle a c b) :
  (b = Real.sqrt 2 * a) ∧ (c = Real.sqrt 3 * a) :=
sorry

end equilateral_is_peculiar_rt_triangle_is_peculiar_peculiar_rt_triangle_ratio_l68_68208


namespace find_n_l68_68013

theorem find_n (n : ℕ) (h : 12^(4 * n) = (1/12)^(n - 30)) : n = 6 := 
by {
  sorry 
}

end find_n_l68_68013


namespace cubes_closed_under_multiplication_l68_68906

-- Define the set of cubes of positive integers
def is_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

-- Define the multiplication operation on the set of cubes
def cube_mult_closed : Prop :=
  ∀ x y : ℕ, is_cube x → is_cube y → is_cube (x * y)

-- The statement we want to prove
theorem cubes_closed_under_multiplication : cube_mult_closed :=
sorry

end cubes_closed_under_multiplication_l68_68906


namespace discounted_price_of_russian_doll_l68_68787

theorem discounted_price_of_russian_doll (original_price : ℕ) (number_of_dolls_original : ℕ) (number_of_dolls_discounted : ℕ) (discounted_price : ℕ) :
  original_price = 4 →
  number_of_dolls_original = 15 →
  number_of_dolls_discounted = 20 →
  (number_of_dolls_original * original_price) = 60 →
  (number_of_dolls_discounted * discounted_price) = 60 →
  discounted_price = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end discounted_price_of_russian_doll_l68_68787


namespace binkie_gemstones_l68_68118

noncomputable def gemstones_solution : ℕ :=
sorry

theorem binkie_gemstones : ∀ (Binkie Frankie Spaatz Whiskers Snowball : ℕ),
  Spaatz = 1 ∧
  Whiskers = Spaatz + 3 ∧
  Snowball = 2 * Whiskers ∧ 
  Snowball % 2 = 0 ∧
  Whiskers % 2 = 0 ∧
  Spaatz = (1 / 2 * Frankie) - 2 ∧
  Binkie = 4 * Frankie ∧
  Binkie + Frankie + Spaatz + Whiskers + Snowball <= 50 →
  Binkie = 24 :=
sorry

end binkie_gemstones_l68_68118


namespace rod_cut_l68_68777

theorem rod_cut (x : ℕ) (h : 3 * x + 5 * x + 7 * x = 120) : 3 * x = 24 :=
by
  sorry

end rod_cut_l68_68777


namespace least_five_digit_whole_number_is_perfect_square_and_cube_l68_68268

theorem least_five_digit_whole_number_is_perfect_square_and_cube :
  ∃ (n : ℕ), (10000 ≤ n ∧ n < 100000) ∧ (∃ (a : ℕ), n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_whole_number_is_perfect_square_and_cube_l68_68268


namespace unique_intersection_point_l68_68231

theorem unique_intersection_point (k : ℝ) :
x = k ->
∃ x : ℝ, x = -3*y^2 - 4*y + 7 -> ∃ k : ℝ, k = 25/3 -> y = 0 -> x = k

end unique_intersection_point_l68_68231


namespace pair1_equivalent_pair2_non_equivalent_pair3_equivalent_pair4_equivalent_pair5_non_equivalent_pair6_equivalent_l68_68900

theorem pair1_equivalent (x : ℝ) : (x^2 + 5 * x < 4) ↔ (x^2 + 5 * x + 3 * x < 4 + 3 * x) :=
sorry

theorem pair2_non_equivalent (x : ℝ) (hx : x ≠ 0) : (x^2 + 5 * x < 4) ↔ (x^2 + 5 * x + 1 / x < 4 + 1 / x) :=
sorry

theorem pair3_equivalent (x : ℝ) (hx : x ≥ 3) : (x ≥ 3) ↔ (x * (x + 5)^2 ≥ 3 * (x + 5)^2) :=
sorry

theorem pair4_equivalent (x : ℝ) (hx : x ≥ 3) : (x ≥ 3) ↔ (x * (x - 5)^2 ≥ 3 * (x - 5)^2) :=
sorry

theorem pair5_non_equivalent (x : ℝ) (hx : x ≠ -1) : (x + 3 > 0) ↔ ( (x + 3) * (x + 1) / (x + 1) > 0) :=
sorry

theorem pair6_equivalent (x : ℝ) (hx : x ≠ -2) : (x - 3 > 0) ↔ ( (x + 2) * (x - 3) / (x + 2) > 0) :=
sorry

end pair1_equivalent_pair2_non_equivalent_pair3_equivalent_pair4_equivalent_pair5_non_equivalent_pair6_equivalent_l68_68900


namespace reserved_fraction_l68_68447

variable (initial_oranges : ℕ) (sold_fraction : ℚ) (rotten_oranges : ℕ) (leftover_oranges : ℕ)
variable (f : ℚ)

def mrSalazarFractionReserved (initial_oranges : ℕ) (sold_fraction : ℚ) (rotten_oranges : ℕ) (leftover_oranges : ℕ) : ℚ :=
  1 - (leftover_oranges + rotten_oranges) * sold_fraction / initial_oranges

theorem reserved_fraction (h1 : initial_oranges = 84) (h2 : sold_fraction = 3 / 7) (h3 : rotten_oranges = 4) (h4 : leftover_oranges = 32) :
  (mrSalazarFractionReserved initial_oranges sold_fraction rotten_oranges leftover_oranges) = 1 / 4 :=
  by
    -- Proof is omitted
    sorry

end reserved_fraction_l68_68447


namespace boat_cannot_complete_round_trip_l68_68889

theorem boat_cannot_complete_round_trip
  (speed_still_water : ℝ)
  (speed_current : ℝ)
  (distance : ℝ)
  (total_time : ℝ)
  (speed_still_water_pos : speed_still_water > 0)
  (speed_current_nonneg : speed_current ≥ 0)
  (distance_pos : distance > 0)
  (total_time_pos : total_time > 0) :
  let speed_downstream := speed_still_water + speed_current
  let speed_upstream := speed_still_water - speed_current
  let time_downstream := distance / speed_downstream
  let time_upstream := distance / speed_upstream
  let total_trip_time := time_downstream + time_upstream
  total_trip_time > total_time :=
by {
  -- Proof goes here
  sorry
}

end boat_cannot_complete_round_trip_l68_68889


namespace trigonometric_identity_l68_68113

noncomputable def cos190 := Real.cos (190 * Real.pi / 180)
noncomputable def sin290 := Real.sin (290 * Real.pi / 180)
noncomputable def cos40 := Real.cos (40 * Real.pi / 180)
noncomputable def tan10 := Real.tan (10 * Real.pi / 180)

theorem trigonometric_identity :
  (cos190 * (1 + Real.sqrt 3 * tan10)) / (sin290 * Real.sqrt (1 - cos40)) = 2 * Real.sqrt 2 :=
by
  sorry

end trigonometric_identity_l68_68113


namespace two_digit_number_is_54_l68_68110

theorem two_digit_number_is_54 
    (n : ℕ) 
    (h1 : 10 ≤ n ∧ n < 100) 
    (h2 : n % 2 = 0) 
    (h3 : ∃ (a b : ℕ), a * b = 20 ∧ 10 * a + b = n) : 
    n = 54 := 
by
  sorry

end two_digit_number_is_54_l68_68110


namespace remainder_of_polynomial_l68_68537

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x + 4

-- Define the main theorem stating the remainder when f(x) is divided by (x - 1) is 6
theorem remainder_of_polynomial : f 1 = 6 := 
by 
  sorry

end remainder_of_polynomial_l68_68537


namespace find_a_b_extreme_points_l68_68008

noncomputable def f (a b x : ℝ) : ℝ := x^3 - 3 * a * x + b

theorem find_a_b (a b : ℝ) (h₁ : a ≠ 0) (h₂ : deriv (f a b) 2 = 0) (h₃ : f a b 2 = 8) : 
  a = 4 ∧ b = 24 :=
by
  sorry

noncomputable def f_deriv (a x : ℝ) : ℝ := 3 * x^2 - 3 * a

theorem extreme_points (a : ℝ) (h₁ : a > 0) : 
  (∃ x: ℝ, f_deriv a x = 0 ∧ 
      ((x = -Real.sqrt a ∧ f a 24 x = 40) ∨ 
       (x = Real.sqrt a ∧ f a 24 x = 16))) := 
by
  sorry

end find_a_b_extreme_points_l68_68008


namespace tan_three_theta_l68_68943

theorem tan_three_theta (θ : Real) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end tan_three_theta_l68_68943


namespace power_boat_travel_time_l68_68106

theorem power_boat_travel_time {r p t : ℝ} (h1 : r > 0) (h2 : p > 0) 
  (h3 : (p + r) * t + (p - r) * (9 - t) = 9 * r) : t = 4.5 :=
by
  sorry

end power_boat_travel_time_l68_68106


namespace pirate_ship_minimum_speed_l68_68348

noncomputable def minimum_speed (initial_distance : ℝ) (caravel_speed : ℝ) (caravel_direction : ℝ) : ℝ :=
  let caravel_velocity_x := -caravel_speed * Real.cos caravel_direction
  let caravel_velocity_y := -caravel_speed * Real.sin caravel_direction
  let t := initial_distance / (caravel_speed * (1 + Real.sqrt 3))
  let v_p := Real.sqrt ((initial_distance / t - caravel_velocity_x)^2 + (caravel_velocity_y)^2)
  v_p

theorem pirate_ship_minimum_speed : 
  minimum_speed 10 12 (Real.pi / 3) = 6 * Real.sqrt 6 :=
by
  sorry

end pirate_ship_minimum_speed_l68_68348


namespace pamela_spilled_sugar_l68_68989

theorem pamela_spilled_sugar 
  (original_amount : ℝ)
  (amount_left : ℝ)
  (h1 : original_amount = 9.8)
  (h2 : amount_left = 4.6)
  : original_amount - amount_left = 5.2 :=
by 
  sorry

end pamela_spilled_sugar_l68_68989


namespace balloons_remaining_proof_l68_68655

-- The initial number of balloons the clown has
def initial_balloons : ℕ := 3 * 12

-- The number of boys who buy balloons
def boys : ℕ := 3

-- The number of girls who buy balloons
def girls : ℕ := 12

-- The total number of children buying balloons
def total_children : ℕ := boys + girls

-- The remaining number of balloons after sales
def remaining_balloons (initial : ℕ) (sold : ℕ) : ℕ := initial - sold

-- Problem statement: Proof that the remaining balloons are 21 given the conditions
theorem balloons_remaining_proof : remaining_balloons initial_balloons total_children = 21 := sorry

end balloons_remaining_proof_l68_68655


namespace ElaCollected13Pounds_l68_68437

def KimberleyCollection : ℕ := 10
def HoustonCollection : ℕ := 12
def TotalCollection : ℕ := 35

def ElaCollection : ℕ := TotalCollection - KimberleyCollection - HoustonCollection

theorem ElaCollected13Pounds : ElaCollection = 13 := sorry

end ElaCollected13Pounds_l68_68437


namespace fraction_of_l68_68479

theorem fraction_of (a b : ℚ) (h_a : a = 3/4) (h_b : b = 1/6) : b / a = 2/9 :=
by
  sorry

end fraction_of_l68_68479


namespace range_of_a_l68_68006

open Set

theorem range_of_a (a x : ℝ) (h : x^2 - 2 * x + 1 - a^2 < 0) (h2 : 0 < x) (h3 : x < 4) :
  a < -3 ∨ a > 3 :=
sorry

end range_of_a_l68_68006


namespace least_five_digit_perfect_square_and_cube_l68_68308

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, n = 5^6 ∧ (n >= 10000 ∧ n < 100000) ∧ (∃ k : ℕ, n = k^2) ∧ (∃ m : ℕ, n = m^3) :=
by
  use 15625
  split
  · exact pow_succ 5 6
  split
  · have h1: 10000 ≤ 5^6 := nat.le_of_eq (calc 5^6 = 15625 : rfl)
    have h2: 5^6 < 100000 := nat.lt_of_eq_of_lt (calc 5^6 = 15625 : rfl) (by decide)
    exact ⟨h1, h2⟩
  · use 125
    exact pow_two 125
  · use 25
    exact pow_three 25
  sorry

end least_five_digit_perfect_square_and_cube_l68_68308


namespace determine_list_price_l68_68353

theorem determine_list_price (x : ℝ) :
  0.12 * (x - 15) = 0.15 * (x - 25) → x = 65 :=
by 
  sorry

end determine_list_price_l68_68353


namespace polygon_sides_l68_68578

theorem polygon_sides
  (n : ℕ)
  (h1 : 180 * (n - 2) - (2 * (2790 / (n - 1)) - 20) = 2790) :
  n = 18 := sorry

end polygon_sides_l68_68578


namespace expected_number_of_digits_l68_68446

noncomputable def expectedNumberDigits : ℝ :=
  let oneDigitProbability := (9 : ℝ) / 16
  let twoDigitProbability := (7 : ℝ) / 16
  (oneDigitProbability * 1) + (twoDigitProbability * 2)

theorem expected_number_of_digits :
  expectedNumberDigits = 1.4375 := by
  sorry

end expected_number_of_digits_l68_68446


namespace part1_correct_part2_correct_l68_68476

-- Definitions for conditions
def total_students := 200
def likes_employment := 140
def dislikes_employment := 60
def p_likes : ℚ := likes_employment / total_students

def male_likes := 60
def male_dislikes := 40
def female_likes := 80
def female_dislikes := 20
def n := total_students
def alpha := 0.005
def chi_squared_critical_value := 7.879

-- Part 1: Estimate the probability of selecting at least 2 students who like employment
def probability_at_least_2_of_3 : ℚ :=
  3 * ((7/10) ^ 2) * (3/10) + ((7/10) ^ 3)

-- Proof goal for Part 1
theorem part1_correct : probability_at_least_2_of_3 = 98 / 125 := by
  sorry

-- Part 2: Chi-squared test for independence between intention and gender
def a := male_likes
def b := male_dislikes
def c := female_likes
def d := female_dislikes
def chi_squared_statistic : ℚ :=
  (n * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Proof goal for Part 2
theorem part2_correct : chi_squared_statistic = 200 / 21 ∧ 200 / 21 > chi_squared_critical_value := by
  sorry

end part1_correct_part2_correct_l68_68476


namespace probability_of_same_color_l68_68639

-- Definitions for the conditions
def red_marbles := 6
def white_marbles := 8
def blue_marbles := 9
def total_marbles := red_marbles + white_marbles + blue_marbles

def total_draws := 4

-- Definition capturing the probability calculations
def probability_same_color := 
  ((6 * 5 * 4 / (23 * 22 * 21)) + 
   (8 * 7 * 6 / (23 * 22 * 21)) + 
   (9 * 8 * 7 / (23 * 22 * 21)))

-- Translate the problem statement into a Lean 4 statement
theorem probability_of_same_color:
  probability_same_color = (160 / 1771) := sorry

end probability_of_same_color_l68_68639


namespace multiplication_difference_l68_68782

theorem multiplication_difference :
  672 * 673 * 674 - 671 * 673 * 675 = 2019 := by
  sorry

end multiplication_difference_l68_68782


namespace least_five_digit_whole_number_is_perfect_square_and_cube_l68_68269

theorem least_five_digit_whole_number_is_perfect_square_and_cube :
  ∃ (n : ℕ), (10000 ≤ n ∧ n < 100000) ∧ (∃ (a : ℕ), n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_whole_number_is_perfect_square_and_cube_l68_68269


namespace least_five_digit_perfect_square_and_cube_l68_68321

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k1 : ℕ, n = k1 ^ 2) ∧ (∃ k2 : ℕ, n = k2 ^ 3) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68321


namespace part_one_part_two_l68_68548

variable {a b c : ℝ}
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_eq : a^2 + b^2 + 4*c^2 = 3)

theorem part_one (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 + b^2 + 4*c^2 = 3) :
  a + b + 2*c ≤ 3 :=
sorry

theorem part_two (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 + b^2 + 4*c^2 = 3) (h_b_eq_2c : b = 2*c) :
  1/a + 1/c ≥ 3 :=
sorry

end part_one_part_two_l68_68548


namespace molecular_weight_CO_l68_68481

theorem molecular_weight_CO :
  let atomic_weight_C := 12.01
  let atomic_weight_O := 16.00
  let molecular_weight := atomic_weight_C + atomic_weight_O
  molecular_weight = 28.01 := 
by
  sorry

end molecular_weight_CO_l68_68481


namespace range_of_a_l68_68594

theorem range_of_a 
  (a b x1 x2 x3 x4 : ℝ)
  (h1 : a ≠ 0)
  (h2 : a^2 ≠ 0)
  (hx1 : a * x1^2 + b * x1 + 1 = 0) 
  (hx2 : a * x2^2 + b * x2 + 1 = 0) 
  (hx3 : a^2 * x3^2 + b * x3 + 1 = 0) 
  (hx4 : a^2 * x4^2 + b * x4 + 1 = 0)
  (h_order : x3 < x1 ∧ x1 < x2 ∧ x2 < x4) :
  0 < a ∧ a < 1 :=
sorry

end range_of_a_l68_68594


namespace right_triangle_ratio_l68_68349

theorem right_triangle_ratio (a b c r s : ℝ) (h : a / b = 2 / 5)
  (h_c : c^2 = a^2 + b^2)
  (h_r : r = a^2 / c)
  (h_s : s = b^2 / c) :
  r / s = 4 / 25 := by
  sorry

end right_triangle_ratio_l68_68349


namespace five_points_plane_distance_gt3_five_points_space_not_necessarily_gt3_l68_68676

/-
Problem (a): Given five points on a plane, where the distance between any two points is greater than 2. 
             Prove that there exists a distance between some two of them that is greater than 3.
-/
theorem five_points_plane_distance_gt3 (P : Fin 5 → ℝ × ℝ) 
    (h : ∀ i j : Fin 5, i ≠ j → dist (P i) (P j) > 2) : 
    ∃ i j : Fin 5, i ≠ j ∧ dist (P i) (P j) > 3 :=
sorry

/-
Problem (b): Given five points in space, where the distance between any two points is greater than 2. 
             Prove that it is not necessarily true that there exists a distance between some two of them that is greater than 3.
-/
theorem five_points_space_not_necessarily_gt3 (P : Fin 5 → ℝ × ℝ × ℝ) 
    (h : ∀ i j : Fin 5, i ≠ j → dist (P i) (P j) > 2) : 
    ¬ ∃ i j : Fin 5, i ≠ j ∧ dist (P i) (P j) > 3 :=
sorry

end five_points_plane_distance_gt3_five_points_space_not_necessarily_gt3_l68_68676


namespace tan_3theta_eq_9_13_l68_68958

open Real

noncomputable def tan3theta (θ : ℝ) (h : tan θ = 3) : Prop :=
  tan (3 * θ) = (9 / 13)

theorem tan_3theta_eq_9_13 (θ : ℝ) (h : tan θ = 3) : tan3theta θ h :=
by
  sorry

end tan_3theta_eq_9_13_l68_68958


namespace least_five_digit_whole_number_is_perfect_square_and_cube_l68_68266

theorem least_five_digit_whole_number_is_perfect_square_and_cube :
  ∃ (n : ℕ), (10000 ≤ n ∧ n < 100000) ∧ (∃ (a : ℕ), n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_whole_number_is_perfect_square_and_cube_l68_68266


namespace area_increase_percentage_l68_68964

variable (r : ℝ) (π : ℝ := Real.pi)

theorem area_increase_percentage (h₁ : r > 0) (h₂ : π > 0) : 
  let new_radius := 2.5 * r
  let original_area := π * r^2
  let new_area := π * (new_radius)^2
  (new_area - original_area) / original_area * 100 = 525 := 
by
  let new_radius := 2.5 * r
  let original_area := π * r^2
  let new_area := π * (new_radius)^2
  sorry

end area_increase_percentage_l68_68964


namespace least_five_digit_perfect_square_cube_l68_68272

theorem least_five_digit_perfect_square_cube :
  ∃ n : Nat, (10000 ≤ n ∧ n < 100000) ∧ (∃ m1 m2 : Nat, n = m1^2 ∧ n = m2^3 ∧ n = 15625) :=
by
  sorry

end least_five_digit_perfect_square_cube_l68_68272


namespace house_cats_initial_l68_68104

def initial_house_cats (S A T H : ℝ) : Prop :=
  S + H + A = T

theorem house_cats_initial (S A T H : ℝ) (h1 : S = 13.0) (h2 : A = 10.0) (h3 : T = 28) :
  initial_house_cats S A T H ↔ H = 5 := by
sorry

end house_cats_initial_l68_68104


namespace find_c1_minus_c2_l68_68815

-- Define the conditions of the problem
variables (c1 c2 : ℝ)
variables (x y : ℝ)
variables (h1 : (2 : ℝ) * x + 3 * y = c1)
variables (h2 : (3 : ℝ) * x + 2 * y = c2)
variables (sol_x : x = 2)
variables (sol_y : y = 1)

-- Define the theorem to be proven
theorem find_c1_minus_c2 : c1 - c2 = -1 := 
by
  sorry

end find_c1_minus_c2_l68_68815


namespace largest_x_63_over_8_l68_68388

theorem largest_x_63_over_8 (x : ℝ) (h1 : ⌊x⌋ / x = 8 / 9) : x = 63 / 8 :=
by
  sorry

end largest_x_63_over_8_l68_68388


namespace boxes_neither_markers_nor_crayons_l68_68526

theorem boxes_neither_markers_nor_crayons (total boxes_markers boxes_crayons boxes_both: ℕ)
  (htotal : total = 15)
  (hmarkers : boxes_markers = 9)
  (hcrayons : boxes_crayons = 4)
  (hboth : boxes_both = 5) :
  total - (boxes_markers + boxes_crayons - boxes_both) = 7 := by
  sorry

end boxes_neither_markers_nor_crayons_l68_68526


namespace three_numbers_difference_of_two_primes_l68_68167

def is_prime (n : ℕ) : Prop := Nat.Prime n
def difference_of_two_primes (a : ℕ) : Prop := ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ a = p2 - p1
def sequence (n : ℕ) : ℕ := 10 * n + 1

theorem three_numbers_difference_of_two_primes :
  ∃ n1 n2 n3 : ℕ, difference_of_two_primes (sequence n1) ∧ difference_of_two_primes (sequence n2) ∧ difference_of_two_primes (sequence n3) ∧ n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3 :=
  sorry

end three_numbers_difference_of_two_primes_l68_68167


namespace has_three_zeros_iff_b_lt_neg3_l68_68817

def f (x b : ℝ) : ℝ := x^3 - b * x^2 - 4

theorem has_three_zeros_iff_b_lt_neg3 (b : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, f x₁ b = 0 ∧ f x₂ b = 0 ∧ f x₃ b = 0 ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ↔ b < -3 := 
sorry

end has_three_zeros_iff_b_lt_neg3_l68_68817


namespace sum_of_squares_base_6_l68_68661

def to_base (n b : ℕ) : ℕ := sorry

theorem sum_of_squares_base_6 :
  let squares := (List.range 12).map (λ x => x.succ ^ 2);
  let squares_base6 := squares.map (λ x => to_base x 6);
  (squares_base6.sum) = to_base 10515 6 :=
by sorry

end sum_of_squares_base_6_l68_68661


namespace rational_decomposition_of_angle_l68_68923

theorem rational_decomposition_of_angle 
  (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2)
  (h_sin_α : ∃ a : ℚ, real.sin α = a)
  (h_cos_α : ∃ b : ℚ, real.cos α = b) : 
  ∃ α1 α2 : ℝ, 
    0 < α1 ∧ α1 < π / 2 ∧ 
    0 < α2 ∧ α2 < π / 2 ∧ 
    α = α1 + α2 ∧ 
    (∃ a1 a2 b1 b2 : ℚ, 
      real.sin α1 = a1 ∧ real.sin α2 = a2 ∧ 
      real.cos α1 = b1 ∧ real.cos α2 = b2) :=
sorry

end rational_decomposition_of_angle_l68_68923


namespace find_first_term_l68_68590

theorem find_first_term (S_n : ℕ → ℝ) (a d : ℝ) (n : ℕ) (h₁ : ∀ n > 0, S_n n = n * (2 * a + (n - 1) * d) / 2)
  (h₂ : d = 3) (h₃ : ∃ c, ∀ n > 0, S_n (3 * n) / S_n n = c) : a = 3 / 2 :=
by
  sorry

end find_first_term_l68_68590


namespace least_five_digit_perfect_square_and_cube_l68_68302

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, n = 5^6 ∧ (n >= 10000 ∧ n < 100000) ∧ (∃ k : ℕ, n = k^2) ∧ (∃ m : ℕ, n = m^3) :=
by
  use 15625
  split
  · exact pow_succ 5 6
  split
  · have h1: 10000 ≤ 5^6 := nat.le_of_eq (calc 5^6 = 15625 : rfl)
    have h2: 5^6 < 100000 := nat.lt_of_eq_of_lt (calc 5^6 = 15625 : rfl) (by decide)
    exact ⟨h1, h2⟩
  · use 125
    exact pow_two 125
  · use 25
    exact pow_three 25
  sorry

end least_five_digit_perfect_square_and_cube_l68_68302


namespace pairs_satisfy_inequality_l68_68440

section inequality_problem

variables (a b : ℝ)

-- Conditions
variable (hb1 : b ≠ -1)
variable (hb2 : b ≠ 0)

-- Inequalities to check
def inequality (a b : ℝ) : Prop :=
  (1 + a) ^ 2 / (1 + b) ≤ 1 + a ^ 2 / b

-- Main theorem
theorem pairs_satisfy_inequality :
  (b > 0 ∨ b < -1 → ∀ a, a ≠ b → inequality a b) ∧
  (∀ a, a ≠ -1 ∧ a ≠ 0 → inequality a a) :=
by
  sorry

end inequality_problem

end pairs_satisfy_inequality_l68_68440


namespace problem_1_problem_2_l68_68696

theorem problem_1 (A B C : ℝ) (h_cond : (abs (B - A)) * (abs (C - A)) * (Real.cos A) = 3 * (abs (A - B)) * (abs (C - B)) * (Real.cos B)) : 
  (Real.tan B = 3 * Real.tan A) := 
sorry

theorem problem_2 (A B C : ℝ) (h_cosC : Real.cos C = Real.sqrt 5 / 5) (h_tanB : Real.tan B = 3 * Real.tan A) : 
  (A = Real.pi / 4) := 
sorry

end problem_1_problem_2_l68_68696


namespace least_five_digit_perfect_square_and_cube_l68_68297

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (n >= 10000 ∧ n < 100000) ∧ (∃ a : ℕ, n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68297


namespace seven_digit_number_insertion_l68_68761

theorem seven_digit_number_insertion (num : ℕ) (h : num = 52115) : (∃ (count : ℕ), count = 21) :=
by 
  sorry

end seven_digit_number_insertion_l68_68761


namespace find_x_and_y_l68_68930

variables (x y : ℝ)

def arithmetic_mean_condition : Prop := (8 + 15 + x + y + 22 + 30) / 6 = 15
def relationship_condition : Prop := y = x + 6

theorem find_x_and_y (h1 : arithmetic_mean_condition x y) (h2 : relationship_condition x y) : 
  x = 4.5 ∧ y = 10.5 :=
by
  sorry

end find_x_and_y_l68_68930


namespace simplify_polynomial_simplify_expression_l68_68214

-- Problem 1:
theorem simplify_polynomial (x : ℝ) : 
  2 * x^3 - 4 * x^2 - 3 * x - 2 * x^2 - x^3 + 5 * x - 7 = x^3 - 6 * x^2 + 2 * x - 7 := 
by
  sorry

-- Problem 2:
theorem simplify_expression (m n : ℝ) (A B : ℝ) (hA : A = 2 * m^2 - m * n) (hB : B = m^2 + 2 * m * n - 5) : 
  4 * A - 2 * B = 6 * m^2 - 8 * m * n + 10 := 
by
  sorry

end simplify_polynomial_simplify_expression_l68_68214


namespace ratio_of_seconds_l68_68609

theorem ratio_of_seconds (x : ℕ) :
  (12 : ℕ) / 8 = x / 240 → x = 360 :=
by
  sorry

end ratio_of_seconds_l68_68609


namespace staff_price_l68_68331

theorem staff_price (d : ℝ) : (d - 0.55 * d) / 2 = 0.225 * d := by
  sorry

end staff_price_l68_68331


namespace no_integer_solution_l68_68518

theorem no_integer_solution :
  ∀ (x : ℤ), ¬ (x^2 + 3 < 2 * x) :=
by
  intro x
  sorry

end no_integer_solution_l68_68518


namespace gcd_lcm_product_l68_68917

theorem gcd_lcm_product (a b : ℕ) (ha : a = 100) (hb : b = 120) :
  Nat.gcd a b * Nat.lcm a b = 12000 := by
  sorry

end gcd_lcm_product_l68_68917


namespace largest_x_63_over_8_l68_68386

theorem largest_x_63_over_8 (x : ℝ) (h1 : ⌊x⌋ / x = 8 / 9) : x = 63 / 8 :=
by
  sorry

end largest_x_63_over_8_l68_68386


namespace meaningful_fraction_l68_68963

theorem meaningful_fraction (x : ℝ) : (x + 5 ≠ 0) → (x ≠ -5) :=
by
  sorry

end meaningful_fraction_l68_68963


namespace find_point_P_coordinates_l68_68181

noncomputable def coordinates_of_point (x y : ℝ) : Prop :=
  y > 0 ∧ x < 0 ∧ abs x = 4 ∧ abs y = 4

theorem find_point_P_coordinates : ∃ (x y : ℝ), coordinates_of_point x y ∧ (x, y) = (-4, 4) :=
by
  sorry

end find_point_P_coordinates_l68_68181


namespace solve_inequality_system_l68_68854

theorem solve_inequality_system (x : ℝ) (h1 : x - 2 ≤ 0) (h2 : (x - 1) / 2 < x) : -1 < x ∧ x ≤ 2 := 
sorry

end solve_inequality_system_l68_68854


namespace calculate_p_p_l68_68442

def p (x y : ℤ) : ℤ :=
  if x ≥ 0 ∧ y ≥ 0 then x + y
  else if x < 0 ∧ y < 0 then x - 2*y
  else if x ≥ 0 ∧ y < 0 then x^2 + y^2
  else 3*x + y

theorem calculate_p_p : p (p 2 (-3)) (p (-4) 1) = 290 :=
by {
  -- required statement of proof problem
  sorry
}

end calculate_p_p_l68_68442


namespace squares_count_correct_l68_68374

-- Assuming basic setup and coordinate system.
def is_valid_point (x y : ℕ) : Prop :=
  x ≤ 8 ∧ y ≤ 8

-- Checking if a point (a, b) in the triangle as described.
def is_in_triangle (a b : ℕ) : Prop :=
  0 ≤ b ∧ b ≤ a ∧ a ≤ 4

-- Function derived from the solution detailing the number of such squares.
def count_squares (a b : ℕ) : ℕ :=
  -- Placeholder to represent the derived formula - to be replaced with actual derivation function
  (9 - a + b) * (a + b + 1) - 1

-- Statement to prove
theorem squares_count_correct (a b : ℕ) (h : is_in_triangle a b) :
  ∃ n, n = count_squares a b := 
sorry

end squares_count_correct_l68_68374


namespace find_value_of_x_squared_plus_one_over_x_squared_l68_68423

theorem find_value_of_x_squared_plus_one_over_x_squared (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 :=
by 
  sorry

end find_value_of_x_squared_plus_one_over_x_squared_l68_68423


namespace least_five_digit_perfect_square_and_cube_l68_68291

theorem least_five_digit_perfect_square_and_cube : 
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ (∃ a : ℕ, n = a ^ 6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68291


namespace find_pairs_l68_68519

theorem find_pairs (n k : ℕ) (h_pos_n : 0 < n) (h_cond : n! + n = n ^ k) : 
  (n = 2 ∧ k = 2) ∨ (n = 3 ∧ k = 2) ∨ (n = 5 ∧ k = 3) := 
by 
  sorry

end find_pairs_l68_68519


namespace developer_break_even_price_l68_68344

theorem developer_break_even_price :
  let acres := 4
  let cost_per_acre := 1863
  let total_cost := acres * cost_per_acre
  let num_lots := 9
  let cost_per_lot := total_cost / num_lots
  cost_per_lot = 828 :=
by {
  sorry  -- This is where the proof would go.
} 

end developer_break_even_price_l68_68344


namespace tan_triple_angle_l68_68948

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end tan_triple_angle_l68_68948


namespace investment_calculation_l68_68347

theorem investment_calculation
  (face_value : ℝ)
  (market_price : ℝ)
  (rate_of_dividend : ℝ)
  (annual_income : ℝ)
  (h1 : face_value = 10)
  (h2 : market_price = 8.25)
  (h3 : rate_of_dividend = 12)
  (h4 : annual_income = 648) :
  ∃ investment : ℝ, investment = 4455 :=
by
  sorry

end investment_calculation_l68_68347


namespace number_of_digits_in_product_l68_68660

open Nat

noncomputable def num_digits (n : ℕ) : ℕ :=
if n = 0 then 1 else Nat.log 10 n + 1

def compute_product : ℕ := 234567 * 123^3

theorem number_of_digits_in_product : num_digits compute_product = 13 := by 
  sorry

end number_of_digits_in_product_l68_68660


namespace foci_on_x_axis_l68_68007

theorem foci_on_x_axis (k : ℝ) : (∃ a b : ℝ, ∀ x y : ℝ, (x^2)/(3 - k) + (y^2)/(1 + k) = 1) ↔ -1 < k ∧ k < 1 :=
by
  sorry

end foci_on_x_axis_l68_68007


namespace triangle_area_is_4_l68_68080

-- Define the lines
def line1 (x : ℝ) : ℝ := 4
def line2 (x : ℝ) : ℝ := 2 + x
def line3 (x : ℝ) : ℝ := 2 - x

-- Define intersection points
def intersection1 : ℝ × ℝ := (2, 4)
def intersection2 : ℝ × ℝ := (-2, 4)
def intersection3 : ℝ × ℝ := (0, 2)

-- Function to calculate the area of a triangle using its vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1))

-- Statement of the proof problem
theorem triangle_area_is_4 :
  ∀ A B C : ℝ × ℝ, A = intersection1 → B = intersection2 → C = intersection3 →
  triangle_area A B C = 4 := by
  sorry

end triangle_area_is_4_l68_68080


namespace number_of_three_leaf_clovers_l68_68715

theorem number_of_three_leaf_clovers (total_leaves : ℕ) (three_leaf_clover : ℕ) (four_leaf_clover : ℕ) (n : ℕ)
  (h1 : total_leaves = 40) (h2 : three_leaf_clover = 3) (h3 : four_leaf_clover = 4) (h4: total_leaves = 3 * n + 4) :
  n = 12 :=
by
  sorry

end number_of_three_leaf_clovers_l68_68715


namespace optimal_cookies_l68_68991

-- Define the initial state and the game's rules
def initial_blackboard : List Int := List.replicate 2020 1

def erase_two (l : List Int) (x y : Int) : List Int :=
  l.erase x |>.erase y

def write_back (l : List Int) (n : Int) : List Int :=
  n :: l

-- Define termination conditions
def game_ends_condition1 (l : List Int) : Prop :=
  ∃ x ∈ l, x > l.sum - x

def game_ends_condition2 (l : List Int) : Prop :=
  l = List.replicate (l.length) 0

def game_ends (l : List Int) : Prop :=
  game_ends_condition1 l ∨ game_ends_condition2 l

-- Define the number of cookies given to Player A
def cookies (l : List Int) : Int :=
  l.length

-- Prove that if both players play optimally, Player A receives 7 cookies
theorem optimal_cookies : cookies (initial_blackboard) = 7 :=
  sorry

end optimal_cookies_l68_68991


namespace least_five_digit_perfect_square_and_cube_l68_68318

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k1 : ℕ, n = k1 ^ 2) ∧ (∃ k2 : ℕ, n = k2 ^ 3) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68318


namespace quadratic_roots_inequality_solution_set_l68_68636

-- Problem 1 statement
theorem quadratic_roots : 
  (∀ x : ℝ, x^2 - 4 * x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) := 
by
  sorry

-- Problem 2 statement
theorem inequality_solution_set :
  (∀ x : ℝ, (x - 2 * (x - 1) ≤ 1 ∧ (1 + x) / 3 > x - 1) ↔ -1 ≤ x ∧ x < 2) :=
by
  sorry

end quadratic_roots_inequality_solution_set_l68_68636


namespace quadratic_equation_has_real_root_l68_68210

theorem quadratic_equation_has_real_root
  (a c m n : ℝ) :
  ∃ x : ℝ, c * x^2 + m * x - a = 0 ∨ ∃ y : ℝ, a * y^2 + n * y + c = 0 :=
by
  -- Proof omitted
  sorry

end quadratic_equation_has_real_root_l68_68210


namespace faster_train_length_l68_68882

noncomputable def length_of_faster_train 
    (speed_train_1_kmph : ℤ) 
    (speed_train_2_kmph : ℤ) 
    (time_seconds : ℤ) : ℤ := 
    (speed_train_1_kmph + speed_train_2_kmph) * 1000 / 3600 * time_seconds

theorem faster_train_length 
    (speed_train_1_kmph : ℤ)
    (speed_train_2_kmph : ℤ)
    (time_seconds : ℤ)
    (h1 : speed_train_1_kmph = 36)
    (h2 : speed_train_2_kmph = 45)
    (h3 : time_seconds = 12) :
    length_of_faster_train speed_train_1_kmph speed_train_2_kmph time_seconds = 270 :=
by
    sorry

end faster_train_length_l68_68882


namespace product_of_last_two_digits_l68_68327

theorem product_of_last_two_digits (A B : ℕ) (h1 : B = 0 ∨ B = 5) (h2 : A + B = 12) : A * B = 35 :=
by {
  -- proof omitted
  sorry
}

end product_of_last_two_digits_l68_68327


namespace simplify_polynomial_l68_68213

theorem simplify_polynomial : 
  ∀ (x : ℝ), 
    (2 * x + 1) ^ 5 - 5 * (2 * x + 1) ^ 4 + 10 * (2 * x + 1) ^ 3 - 10 * (2 * x + 1) ^ 2 + 5 * (2 * x + 1) - 1 
    = 32 * x ^ 5 := 
by sorry

end simplify_polynomial_l68_68213


namespace sin_squared_not_periodic_l68_68703

noncomputable def sin_squared (x : ℝ) : ℝ := Real.sin (x^2)

theorem sin_squared_not_periodic : 
  ¬ (∃ T > 0, ∀ x ∈ Set.univ, sin_squared (x + T) = sin_squared x) := 
sorry

end sin_squared_not_periodic_l68_68703


namespace calculate_c_l68_68962

-- Define the given equation as a hypothesis
theorem calculate_c (a b k c : ℝ) (h : (1 / (k * a) - 1 / (k * b) = 1 / c)) :
  c = k * a * b / (b - a) :=
by
  sorry

end calculate_c_l68_68962


namespace least_five_digit_perfect_square_and_cube_l68_68301

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (n >= 10000 ∧ n < 100000) ∧ (∃ a : ℕ, n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68301


namespace trigonometric_identity_l68_68027

variable (A B C a b c : ℝ)
variable (h_triangle : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
variable (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_sum_angles : A + B + C = π)
variable (h_condition : (c / b) + (b / c) = (5 * Real.cos A) / 2)

theorem trigonometric_identity 
  (h_triangle_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) 
  (h_sides_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum_angles_eq : A + B + C = π) 
  (h_given : (c / b) + (b / c) = (5 * Real.cos A) / 2) : 
  (Real.tan A / Real.tan B) + (Real.tan A / Real.tan C) = 1/2 :=
by
  sorry

end trigonometric_identity_l68_68027


namespace find_x_l68_68217

theorem find_x (x y z : ℝ) (h1 : x^2 / y = 4) (h2 : y^2 / z = 9) (h3 : z^2 / x = 16) : x = 4 :=
sorry

end find_x_l68_68217


namespace pet_store_animals_l68_68610

theorem pet_store_animals (cats dogs birds : ℕ) 
    (ratio_cats_dogs_birds : 2 * birds = 4 * cats ∧ 3 * cats = 2 * dogs) 
    (num_cats : cats = 20) : dogs = 30 ∧ birds = 40 :=
by 
  -- This is where the proof would go, but we can skip it for this problem statement.
  sorry

end pet_store_animals_l68_68610


namespace math_problem_l68_68695

theorem math_problem (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x^2) = 23 :=
sorry

end math_problem_l68_68695


namespace expand_and_simplify_l68_68792

theorem expand_and_simplify (x : ℝ) : (x - 3) * (x + 7) + x = x^2 + 5 * x - 21 := 
by 
  sorry

end expand_and_simplify_l68_68792


namespace find_k_intersects_parabola_at_one_point_l68_68234

theorem find_k_intersects_parabola_at_one_point :
  ∃ k : ℝ, (∀ y : ℝ, -3 * y^2 - 4 * y + 7 = k ↔ y = (-4 / (2 * 3))) →
    k = 25 / 3 :=
by sorry

end find_k_intersects_parabola_at_one_point_l68_68234


namespace factor_expression_l68_68783

noncomputable def expression (x : ℝ) : ℝ := (15 * x^3 + 80 * x - 5) - (-4 * x^3 + 4 * x - 5)

theorem factor_expression (x : ℝ) : expression x = 19 * x * (x^2 + 4) := 
by 
  sorry

end factor_expression_l68_68783


namespace arithmetic_sequence_general_formula_l68_68812

theorem arithmetic_sequence_general_formula
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h2 : a 4 - a 2 = 4)
  (h3 : S 3 = 9)
  : ∀ n : ℕ, a n = 2 * n - 1 := 
sorry

end arithmetic_sequence_general_formula_l68_68812


namespace maximal_product_sum_l68_68052

theorem maximal_product_sum : 
  ∃ (k m : ℕ), 
  k = 671 ∧ 
  m = 2 ∧ 
  2017 = 3 * k + 2 * m ∧ 
  ∀ a b : ℕ, a + b = 2017 ∧ (a < k ∨ b < m) → a * b ≤ 3 * k * 2 * m
:= 
sorry

end maximal_product_sum_l68_68052


namespace remainder_when_divided_by_seven_l68_68801

theorem remainder_when_divided_by_seven (n : ℕ) (h₁ : n^3 ≡ 3 [MOD 7]) (h₂ : n^4 ≡ 2 [MOD 7]) : 
  n ≡ 6 [MOD 7] :=
sorry

end remainder_when_divided_by_seven_l68_68801


namespace jen_age_difference_l68_68028

-- Definitions as conditions given in the problem
def son_present_age := 16
def jen_present_age := 41

-- The statement to be proved
theorem jen_age_difference :
  3 * son_present_age - jen_present_age = 7 :=
by
  sorry

end jen_age_difference_l68_68028


namespace sum_weights_greater_than_2p_l68_68170

variables (p x y l l' : ℝ)

-- Conditions
axiom balance1 : x * l = p * l'
axiom balance2 : y * l' = p * l

-- The statement to prove
theorem sum_weights_greater_than_2p : x + y > 2 * p :=
by
  sorry

end sum_weights_greater_than_2p_l68_68170


namespace division_of_decimals_l68_68114

theorem division_of_decimals : 0.18 / 0.003 = 60 :=
by
  sorry

end division_of_decimals_l68_68114


namespace percentage_decrease_after_raise_l68_68243

theorem percentage_decrease_after_raise
  (original_salary : ℝ) (final_salary : ℝ) (initial_raise_percent : ℝ)
  (initial_salary_raised : original_salary * (1 + initial_raise_percent / 100) = 5500): 
  original_salary = 5000 -> final_salary = 5225 -> initial_raise_percent = 10 ->
  ∃ (percentage_decrease : ℝ),
    final_salary = original_salary * (1 + initial_raise_percent / 100) * (1 - percentage_decrease / 100)
    ∧ percentage_decrease = 5 := by
  intros h1 h2 h3
  use 5
  rw [h1, h2, h3]
  simp
  sorry

end percentage_decrease_after_raise_l68_68243


namespace sqrt_9025_squared_l68_68904

-- Define the square root function and its properties
noncomputable def sqrt (x : ℕ) : ℕ := sorry

axiom sqrt_def (n : ℕ) (hn : 0 ≤ n) : (sqrt n) ^ 2 = n

-- Prove the specific case
theorem sqrt_9025_squared : (sqrt 9025) ^ 2 = 9025 :=
sorry

end sqrt_9025_squared_l68_68904


namespace construct_right_triangle_l68_68907

theorem construct_right_triangle (hypotenuse : ℝ) (ε : ℝ) (h_positive : 0 < ε) (h_less_than_ninety : ε < 90) :
    ∃ α β : ℝ, α + β = 90 ∧ α - β = ε ∧ 45 < α ∧ α < 90 :=
by
  sorry

end construct_right_triangle_l68_68907


namespace tan_three_theta_l68_68944

theorem tan_three_theta (θ : Real) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end tan_three_theta_l68_68944


namespace least_five_digit_perfect_square_and_cube_l68_68319

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k1 : ℕ, n = k1 ^ 2) ∧ (∃ k2 : ℕ, n = k2 ^ 3) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68319


namespace max_candy_received_l68_68095

theorem max_candy_received (students : ℕ) (candies : ℕ) (min_candy_per_student : ℕ) 
    (h_students : students = 40) (h_candies : candies = 200) (h_min_candy : min_candy_per_student = 2) :
    ∃ max_candy : ℕ, max_candy = 122 := by
  sorry

end max_candy_received_l68_68095


namespace gcd_m_n_l68_68980

namespace GCDProof

def m : ℕ := 33333333
def n : ℕ := 666666666

theorem gcd_m_n : gcd m n = 2 := 
  sorry

end GCDProof

end gcd_m_n_l68_68980


namespace scientific_notation_of_distance_l68_68059

theorem scientific_notation_of_distance :
  ∃ (n : ℝ), n = 384000 ∧ 384000 = n * 10^5 :=
sorry

end scientific_notation_of_distance_l68_68059


namespace complement_problem_l68_68937

open Set

variable (U A : Set ℕ)

def complement (U A : Set ℕ) : Set ℕ := U \ A

theorem complement_problem
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 3}) :
  complement U A = {2, 4, 5} :=
by
  rw [complement, hU, hA]
  sorry

end complement_problem_l68_68937


namespace minimize_a_l68_68039

open Polynomial

noncomputable def smallest_possible_value_a (P : ℤ[X]) (a : ℤ) : Prop :=
  (∀ x : ℤ, x ∈ {1, 2, 3, 4} → P.eval x = a) ∧
  (∀ x : ℤ, x ∈ {-1, -2, -3, -4} → P.eval x = -a) ∧
  (∀ n : ℤ, 0 < n → a = n → ∃ Q : ℤ[X], P = (X - 1) * (X - 2) * (X - 3) * (X - 4) * Q + C a)

theorem minimize_a (P : ℤ[X]) (a : ℤ) (h1 : ∀ (x : ℤ), x ∈ {1, 2, 3, 4} → P.eval x = a)
  (h2 : ∀ (x : ℤ), x ∈ {-1, -2, -3, -4} → P.eval x = -a) :
  ∃ n, n = 1680 ∧ a = n :=
begin
  sorry
end

end minimize_a_l68_68039


namespace common_chord_eq_l68_68164

-- Define the equations of the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 10*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 2*y - 40 = 0

-- Define the statement to prove
theorem common_chord_eq (x y : ℝ) : circle1 x y ∧ circle2 x y → 2*x + y - 5 = 0 :=
sorry

end common_chord_eq_l68_68164


namespace ball_color_problem_l68_68869

theorem ball_color_problem
  (n : ℕ)
  (h₀ : ∀ i : ℕ, i ≤ 49 → ∃ r : ℕ, r = 49 ∧ i = 50) 
  (h₁ : ∀ i : ℕ, i > 49 → ∃ r : ℕ, r = 49 + 7 * (i - 50) / 8 ∧ i = n)
  (h₂ : 90 ≤ (49 + (7 * (n - 50) / 8)) * 10 / n) :
  n ≤ 210 := 
sorry

end ball_color_problem_l68_68869


namespace inequality_not_always_true_l68_68156

-- Declare the variables and conditions
variables {a b c : ℝ}

-- Given conditions
axiom h1 : a < b 
axiom h2 : b < c 
axiom h3 : a * c < 0

-- Statement of the problem
theorem inequality_not_always_true : ¬ (∀ a b c, (a < b ∧ b < c ∧ a * c < 0) → (c^2 / a < b^2 / a)) :=
by { sorry }

end inequality_not_always_true_l68_68156


namespace product_of_two_numbers_l68_68865

theorem product_of_two_numbers (x y : ℝ) 
  (h₁ : x + y = 50) 
  (h₂ : x - y = 6) : 
  x * y = 616 := 
by
  sorry

end product_of_two_numbers_l68_68865


namespace max_gold_coins_l68_68050

variables (planks : ℕ)
          (windmill_planks windmill_gold : ℕ)
          (steamboat_planks steamboat_gold : ℕ)
          (airplane_planks airplane_gold : ℕ)

theorem max_gold_coins (h_planks: planks = 130)
                       (h_windmill: windmill_planks = 5 ∧ windmill_gold = 6)
                       (h_steamboat: steamboat_planks = 7 ∧ steamboat_gold = 8)
                       (h_airplane: airplane_planks = 14 ∧ airplane_gold = 19) :
  ∃ (gold : ℕ), gold = 172 :=
by
  sorry

end max_gold_coins_l68_68050


namespace integer_part_divisible_by_112_l68_68199

def is_odd (n : ℕ) : Prop := n % 2 = 1
def not_divisible_by_3 (n : ℕ) : Prop := n % 3 ≠ 0

theorem integer_part_divisible_by_112
  (m : ℕ) (hm_pos : 0 < m) (hm_odd : is_odd m) (hm_not_div3 : not_divisible_by_3 m) :
  ∃ n : ℤ, 112 * n = 4^m - (2 + Real.sqrt 2)^m - (2 - Real.sqrt 2)^m :=
by
  sorry

end integer_part_divisible_by_112_l68_68199


namespace bulbs_needed_l68_68597

theorem bulbs_needed (M : ℕ) (hM : M = 12) : 
  let large := 2 * M in
  let small := M + 10 in
  let bulbs_medium := 2 * M in
  let bulbs_large := 3 * large in
  let bulbs_small := small in
  bulbs_medium + bulbs_large + bulbs_small = 118 :=
by
  sorry

end bulbs_needed_l68_68597


namespace cards_left_l68_68977

variable (initialCards : ℕ) (givenCards : ℕ) (remainingCards : ℕ)

def JasonInitialCards := 13
def CardsGivenAway := 9

theorem cards_left : initialCards = JasonInitialCards → givenCards = CardsGivenAway → remainingCards = initialCards - givenCards → remainingCards = 4 :=
by
  intros
  subst_vars
  sorry

end cards_left_l68_68977


namespace function_relationship_l68_68673

variable {A B : Type} [Nonempty A] [Nonempty B]
variable (f : A → B) 

def domain (f : A → B) : Set A := {a | ∃ b, f a = b}
def range (f : A → B) : Set B := {b | ∃ a, f a = b}

theorem function_relationship (M : Set A) (N : Set B) (hM : M = Set.univ)
                              (hN : N = range f) : M = Set.univ ∧ N ⊆ Set.univ :=
  sorry

end function_relationship_l68_68673


namespace total_cans_collected_l68_68768

-- Definitions based on conditions
def cans_LaDonna : ℕ := 25
def cans_Prikya : ℕ := 2 * cans_LaDonna
def cans_Yoki : ℕ := 10

-- Theorem statement
theorem total_cans_collected : 
  cans_LaDonna + cans_Prikya + cans_Yoki = 85 :=
by
  -- The proof is not required, inserting sorry to complete the statement
  sorry

end total_cans_collected_l68_68768


namespace find_b_age_l68_68084

theorem find_b_age (a b : ℕ) (h1 : a + 10 = 2 * (b - 10)) (h2 : a = b + 9) : b = 39 :=
sorry

end find_b_age_l68_68084


namespace max_value_sum_seq_l68_68183

theorem max_value_sum_seq : 
  ∃ a1 a2 a3 a4 : ℝ, 
    a1 = 0 ∧ 
    |a2| = |a1 - 1| ∧ 
    |a3| = |a2 - 1| ∧ 
    |a4| = |a3 - 1| ∧ 
    a1 + a2 + a3 + a4 = 2 := 
by 
  sorry

end max_value_sum_seq_l68_68183


namespace find_divisor_for_multiple_l68_68528

theorem find_divisor_for_multiple (d : ℕ) :
  (∃ k : ℕ, k * d % 1821 = 710 ∧ k * d % 24 = 13 ∧ k * d = 3024) →
  d = 23 :=
by
  intros h
  sorry

end find_divisor_for_multiple_l68_68528


namespace measure_8_liters_possible_l68_68566

-- Define the types for buckets
structure Bucket :=
  (capacity : ℕ)
  (water : ℕ := 0)

-- Initial state with a 10-liter bucket and a 6-liter bucket, both empty
def B10_init := Bucket.mk 10 0
def B6_init := Bucket.mk 6 0

-- Define a function to check if we can measure 8 liters in B10
def can_measure_8_liters (B10 B6 : Bucket) : Prop :=
  (B10.water = 8 ∧ B10.capacity = 10 ∧ B6.capacity = 6)

-- The statement to prove there exists a sequence of operations to measure 8 liters in B10
theorem measure_8_liters_possible : ∃ (B10 B6 : Bucket), can_measure_8_liters B10 B6 :=
by
  -- Proof omitted
  sorry

end measure_8_liters_possible_l68_68566


namespace maddox_more_profit_than_theo_l68_68200

-- Definitions (conditions)
def cost_per_camera : ℕ := 20
def num_cameras : ℕ := 3
def total_cost : ℕ := num_cameras * cost_per_camera

def maddox_selling_price_per_camera : ℕ := 28
def theo_selling_price_per_camera : ℕ := 23

-- Total selling price
def maddox_total_selling_price : ℕ := num_cameras * maddox_selling_price_per_camera
def theo_total_selling_price : ℕ := num_cameras * theo_selling_price_per_camera

-- Profits
def maddox_profit : ℕ := maddox_total_selling_price - total_cost
def theo_profit : ℕ := theo_total_selling_price - total_cost

-- Proof Statement
theorem maddox_more_profit_than_theo : maddox_profit - theo_profit = 15 := by
  sorry

end maddox_more_profit_than_theo_l68_68200


namespace ordered_triple_l68_68040

theorem ordered_triple (a b c : ℝ) (h1 : 4 < a) (h2 : 4 < b) (h3 : 4 < c) 
  (h_eq : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 45) 
  : (a, b, c) = (12, 10, 8) :=
  sorry

end ordered_triple_l68_68040


namespace hike_on_saturday_l68_68058

-- Define the conditions
variables (x : Real) -- distance hiked on Saturday
variables (y : Real) -- distance hiked on Sunday
variables (z : Real) -- total distance hiked

-- Define given values
def hiked_on_sunday : Real := 1.6
def total_hiked : Real := 9.8

-- The hypothesis: y + x = z
axiom hike_total : y + x = z

theorem hike_on_saturday : x = 8.2 :=
by
  sorry

end hike_on_saturday_l68_68058


namespace sum_of_reciprocal_transformed_roots_l68_68784

-- Define the polynomial f
def f (x : ℝ) : ℝ := 15 * x^3 - 35 * x^2 + 20 * x - 2

-- Define the condition that the roots are distinct real numbers between 0 and 1
def is_root (f : ℝ → ℝ) (x : ℝ) : Prop := f x = 0
def roots_between_0_and_1 (a b c : ℝ) : Prop := 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  0 < a ∧ a < 1 ∧ 
  0 < b ∧ b < 1 ∧ 
  0 < c ∧ c < 1 ∧
  is_root f a ∧ is_root f b ∧ is_root f c

-- The theorem representing the proof problem
theorem sum_of_reciprocal_transformed_roots (a b c : ℝ) 
  (h : roots_between_0_and_1 a b c) :
  (1/(1-a)) + (1/(1-b)) + (1/(1-c)) = 2/3 :=
by
  sorry

end sum_of_reciprocal_transformed_roots_l68_68784


namespace find_unknown_rate_l68_68650

-- Define the known quantities
def num_blankets1 := 4
def price1 := 100

def num_blankets2 := 5
def price2 := 150

def num_blankets3 := 3
def price3 := 200

def num_blankets4 := 6
def price4 := 75

def num_blankets_unknown := 2

def avg_price := 150
def total_blankets := num_blankets1 + num_blankets2 + num_blankets3 + num_blankets4 + num_blankets_unknown -- 20 blankets in total

-- Hypotheses
def total_known_cost := num_blankets1 * price1 + num_blankets2 * price2 + num_blankets3 * price3 + num_blankets4 * price4
-- 2200 Rs.

def total_cost := total_blankets * avg_price -- 3000 Rs.

theorem find_unknown_rate :
  (total_cost - total_known_cost) / num_blankets_unknown = 400 :=
by sorry

end find_unknown_rate_l68_68650


namespace contributions_before_john_l68_68960

theorem contributions_before_john
  (A : ℝ) (n : ℕ)
  (h1 : 1.5 * A = 75)
  (h2 : (n * A + 150) / (n + 1) = 75) :
  n = 3 :=
by
  sorry

end contributions_before_john_l68_68960


namespace sets_relationship_l68_68372

def M : Set ℤ := {x : ℤ | ∃ k : ℤ, x = 3 * k - 2}
def P : Set ℤ := {x : ℤ | ∃ n : ℤ, x = 3 * n + 1}
def S : Set ℤ := {x : ℤ | ∃ m : ℤ, x = 6 * m + 1}

theorem sets_relationship : S ⊆ P ∧ M = P := by
  sorry

end sets_relationship_l68_68372


namespace circle_line_tangent_l68_68019

theorem circle_line_tangent (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = 4 * m ∧ x + y = 2 * m) ↔ m = 2 :=
sorry

end circle_line_tangent_l68_68019


namespace competition_scores_l68_68096

theorem competition_scores (n d : ℕ) (h_n : 1 < n)
  (h_total_score : d * (n * (n + 1)) / 2 = 26 * n) :
  (n, d) = (3, 13) ∨ (n, d) = (12, 4) ∨ (n, d) = (25, 2) :=
by
  sorry

end competition_scores_l68_68096


namespace repayment_correct_l68_68771

noncomputable def repayment_amount (a γ : ℝ) : ℝ :=
  a * γ * (1 + γ) ^ 5 / ((1 + γ) ^ 5 - 1)

theorem repayment_correct (a γ : ℝ) (γ_pos : γ > 0) : 
  repayment_amount a γ = a * γ * (1 + γ) ^ 5 / ((1 + γ) ^ 5 - 1) :=
by
   sorry

end repayment_correct_l68_68771


namespace largest_real_number_l68_68398

theorem largest_real_number (x : ℝ) (h : (⌊x⌋ / x) = (8 / 9)) : x ≤ 63 / 8 :=
by
  sorry

end largest_real_number_l68_68398


namespace max_x4_y6_l68_68842

noncomputable def maximum_product (x y : ℝ) := x^4 * y^6

theorem max_x4_y6 (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 100) :
  maximum_product x y ≤ maximum_product 40 60 := sorry

end max_x4_y6_l68_68842


namespace students_neither_play_football_nor_cricket_l68_68727

theorem students_neither_play_football_nor_cricket
  (total_students football_players cricket_players both_players : ℕ)
  (h_total : total_students = 470)
  (h_football : football_players = 325)
  (h_cricket : cricket_players = 175)
  (h_both : both_players = 80) :
  (total_students - (football_players + cricket_players - both_players)) = 50 :=
by
  sorry

end students_neither_play_football_nor_cricket_l68_68727


namespace iris_to_tulip_ratio_l68_68704

theorem iris_to_tulip_ratio (earnings_per_bulb : ℚ)
  (tulip_bulbs daffodil_bulbs crocus_ratio total_earnings : ℕ)
  (iris_bulbs : ℕ) (h0 : earnings_per_bulb = 0.50)
  (h1 : tulip_bulbs = 20) (h2 : daffodil_bulbs = 30)
  (h3 : crocus_ratio = 3) (h4 : total_earnings = 75)
  (h5 : total_earnings = earnings_per_bulb * (tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_ratio * daffodil_bulbs))
  : iris_bulbs = 10 → tulip_bulbs = 20 → (iris_bulbs : ℚ) / (tulip_bulbs : ℚ) = 1 / 2 :=
by {
  intros; sorry
}

end iris_to_tulip_ratio_l68_68704


namespace lines_perpendicular_l68_68603

noncomputable def is_perpendicular (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

theorem lines_perpendicular {m : ℝ} :
  is_perpendicular (m + 2) (1 - m) (m - 1) (2 * m + 3) ↔ m = 1 :=
by
  sorry

end lines_perpendicular_l68_68603


namespace least_five_digit_perfect_square_and_cube_l68_68303

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, n = 5^6 ∧ (n >= 10000 ∧ n < 100000) ∧ (∃ k : ℕ, n = k^2) ∧ (∃ m : ℕ, n = m^3) :=
by
  use 15625
  split
  · exact pow_succ 5 6
  split
  · have h1: 10000 ≤ 5^6 := nat.le_of_eq (calc 5^6 = 15625 : rfl)
    have h2: 5^6 < 100000 := nat.lt_of_eq_of_lt (calc 5^6 = 15625 : rfl) (by decide)
    exact ⟨h1, h2⟩
  · use 125
    exact pow_two 125
  · use 25
    exact pow_three 25
  sorry

end least_five_digit_perfect_square_and_cube_l68_68303


namespace at_most_n_zeros_l68_68504

-- Definitions of conditions
variables {α : Type*} [Inhabited α]

/-- Define the structure of the sheet of numbers with the given properties -/
structure sheet :=
(n : ℕ)
(val : ℕ → ℤ)

-- Assuming infinite sheet and the properties
variable (s : sheet)

-- Predicate for a row having only positive integers
def all_positive (r : ℕ → ℤ) : Prop := ∀ i, r i > 0

-- Define the initial row R which has all positive integers
variable {R : ℕ → ℤ}

-- Statement that each element in the row below is sum of element above and to the left
def below_sum (r R : ℕ → ℤ) (n : ℕ) : Prop := ∀ i, r i = R i + (if i = 0 then 0 else R (i - 1))

-- Variable for the row n below R
variable {Rn : ℕ → ℤ}

-- Main theorem statement
theorem at_most_n_zeros (n : ℕ) (hr : all_positive R) (hs : below_sum R Rn n) : 
  ∃ k ≤ n, Rn k = 0 ∨ Rn k > 0 := sorry

end at_most_n_zeros_l68_68504


namespace probability_both_cards_are_diamonds_l68_68108

-- Conditions definitions
def total_cards : ℕ := 52
def diamonds_in_deck : ℕ := 13
def two_draws : ℕ := 2

-- Calculation definitions
def total_possible_outcomes : ℕ := (total_cards * (total_cards - 1)) / two_draws
def favorable_outcomes : ℕ := (diamonds_in_deck * (diamonds_in_deck - 1)) / two_draws

-- Definition of the probability asked in the question
def probability_both_diamonds : ℚ := favorable_outcomes / total_possible_outcomes

theorem probability_both_cards_are_diamonds :
  probability_both_diamonds = 1 / 17 := 
sorry

end probability_both_cards_are_diamonds_l68_68108


namespace combined_stripes_eq_22_l68_68985

def stripes_olga_per_shoe : ℕ := 3
def shoes_per_person : ℕ := 2
def stripes_olga_total : ℕ := stripes_olga_per_shoe * shoes_per_person

def stripes_rick_per_shoe : ℕ := stripes_olga_per_shoe - 1
def stripes_rick_total : ℕ := stripes_rick_per_shoe * shoes_per_person

def stripes_hortense_per_shoe : ℕ := stripes_olga_per_shoe * 2
def stripes_hortense_total : ℕ := stripes_hortense_per_shoe * shoes_per_person

def total_stripes : ℕ := stripes_olga_total + stripes_rick_total + stripes_hortense_total

theorem combined_stripes_eq_22 : total_stripes = 22 := by
  sorry

end combined_stripes_eq_22_l68_68985


namespace rhombus_perimeter_l68_68061

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) :
  let side_length := real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  in 4 * side_length = 16 * real.sqrt 13 :=
by
  let d1_half := d1 / 2
  let d2_half := d2 / 2
  have h3 : d1_half = 12 := by sorry
  have h4 : d2_half = 8 := by sorry
  let side_length := real.sqrt (d1_half ^ 2 + d2_half ^ 2)
  have h5 : side_length = real.sqrt 208 := by sorry
  have h6 : real.sqrt 208 = 4 * real.sqrt 13 := by sorry
  show 4 * side_length = 16 * real.sqrt 13
  from by
    rw [h6]
    rfl

end rhombus_perimeter_l68_68061


namespace measure_8_liters_possible_l68_68565

-- Define the types for buckets
structure Bucket :=
  (capacity : ℕ)
  (water : ℕ := 0)

-- Initial state with a 10-liter bucket and a 6-liter bucket, both empty
def B10_init := Bucket.mk 10 0
def B6_init := Bucket.mk 6 0

-- Define a function to check if we can measure 8 liters in B10
def can_measure_8_liters (B10 B6 : Bucket) : Prop :=
  (B10.water = 8 ∧ B10.capacity = 10 ∧ B6.capacity = 6)

-- The statement to prove there exists a sequence of operations to measure 8 liters in B10
theorem measure_8_liters_possible : ∃ (B10 B6 : Bucket), can_measure_8_liters B10 B6 :=
by
  -- Proof omitted
  sorry

end measure_8_liters_possible_l68_68565


namespace find_multiple_of_q_l68_68966

-- Definitions of x and y
def x (k q : ℤ) : ℤ := 55 + k * q
def y (q : ℤ) : ℤ := 4 * q + 41

-- The proof statement
theorem find_multiple_of_q (k : ℤ) : x k 7 = y 7 → k = 2 := by
  sorry

end find_multiple_of_q_l68_68966


namespace carlos_initial_blocks_l68_68363

theorem carlos_initial_blocks (g : ℕ) (l : ℕ) (total : ℕ) : g = 21 → l = 37 → total = g + l → total = 58 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end carlos_initial_blocks_l68_68363


namespace num_ints_between_sqrt2_and_sqrt32_l68_68166

theorem num_ints_between_sqrt2_and_sqrt32 : 
  ∃ n : ℕ, n = 4 ∧ 
  (∀ k : ℤ, (2 ≤ k) ∧ (k ≤ 5)) :=
by
  sorry

end num_ints_between_sqrt2_and_sqrt32_l68_68166


namespace sin_sum_less_than_zero_l68_68404

noncomputable def is_acute_triangle (α β γ : ℝ) : Prop :=
  α + β + γ = Real.pi ∧ 0 < α ∧ α < Real.pi / 2 ∧ 0 < β ∧ β < Real.pi / 2 ∧ 0 < γ ∧ γ < Real.pi / 2

theorem sin_sum_less_than_zero (n : ℕ) :
  (∀ (α β γ : ℝ), is_acute_triangle α β γ → (Real.sin (n * α) + Real.sin (n * β) + Real.sin (n * γ) < 0)) ↔ n = 4 :=
by
  sorry

end sin_sum_less_than_zero_l68_68404


namespace friend_balloon_count_l68_68329

theorem friend_balloon_count (you_balloons friend_balloons : ℕ) (h1 : you_balloons = 7) (h2 : you_balloons = friend_balloons + 2) : friend_balloons = 5 :=
by
  sorry

end friend_balloon_count_l68_68329


namespace derivative_bound_l68_68408

theorem derivative_bound {a b : ℝ} (f : ℝ → ℝ) (h_diff : ∀ x ∈ set.Icc a b, deriv f x ∧ deriv (deriv f) x)
  (h_ab : b ≥ a + 2)
  (h_f_bound : ∀ x ∈ set.Icc a b, abs (f x) ≤ 1)
  (h_f''_bound : ∀ x ∈ set.Icc a b, abs (deriv (deriv f) x) ≤ 1) :
  ∀ x ∈ set.Icc a b, abs (deriv f x) ≤ 2 :=
by
  sorry

end derivative_bound_l68_68408


namespace line_points_k_l68_68099

noncomputable def k : ℝ := 8

theorem line_points_k (k : ℝ) : 
  (∀ k : ℝ, ∃ b : ℝ, b = (10 - k) / (5 - 5) ∧
  ∀ b, b = (-k) / (20 - 5) → k = 8) :=
  by
  sorry

end line_points_k_l68_68099


namespace farmer_sowed_correct_amount_l68_68345

def initial_buckets : ℝ := 8.75
def final_buckets : ℝ := 6
def buckets_sowed : ℝ := initial_buckets - final_buckets

theorem farmer_sowed_correct_amount : buckets_sowed = 2.75 :=
by {
  sorry
}

end farmer_sowed_correct_amount_l68_68345


namespace a_share_is_1400_l68_68627

-- Definitions for the conditions
def investment_A : ℕ := 7000
def investment_B : ℕ := 11000
def investment_C : ℕ := 18000
def share_B : ℕ := 2200

-- Definition for the ratios
def ratio_A : ℚ := investment_A / 1000
def ratio_B : ℚ := investment_B / 1000
def ratio_C : ℚ := investment_C / 1000

-- Sum of ratios
def sum_ratios : ℚ := ratio_A + ratio_B + ratio_C

-- Total profit P can be deduced from B's share
def total_profit : ℚ := share_B * sum_ratios / ratio_B

-- Goal: Prove that A's share is $1400
def share_A : ℚ := ratio_A * total_profit / sum_ratios

theorem a_share_is_1400 : share_A = 1400 :=
sorry

end a_share_is_1400_l68_68627


namespace find_non_negative_integer_solutions_l68_68383

theorem find_non_negative_integer_solutions :
  ∃ (x y z w : ℕ), 2 ^ x * 3 ^ y - 5 ^ z * 7 ^ w = 1 ∧
  ((x = 1 ∧ y = 0 ∧ z = 0 ∧ w = 0) ∨
   (x = 3 ∧ y = 0 ∧ z = 0 ∧ w = 1) ∨
   (x = 1 ∧ y = 1 ∧ z = 1 ∧ w = 0) ∨
   (x = 2 ∧ y = 2 ∧ z = 1 ∧ w = 1)) := by
  sorry

end find_non_negative_integer_solutions_l68_68383


namespace tan_3theta_eq_9_13_l68_68955

open Real

noncomputable def tan3theta (θ : ℝ) (h : tan θ = 3) : Prop :=
  tan (3 * θ) = (9 / 13)

theorem tan_3theta_eq_9_13 (θ : ℝ) (h : tan θ = 3) : tan3theta θ h :=
by
  sorry

end tan_3theta_eq_9_13_l68_68955


namespace fn_simplified_l68_68788

open BigOperators

def a : ℕ → ℤ
| 0        := 0
| 1        := 0 
| 2        := 1
| (n + 3)  := (n + 3 : ℤ) / 2 * a (n + 2) + (n + 3 : ℤ) * (n + 2) / 2 * a (n + 1) 
              + (-1 : ℤ) ^ (n + 3) * (1 - (n + 3) / 2)

def f (n : ℕ) : ℤ :=
∑ k in finset.range n, (k + 1) * nat.choose n k * a (n - k)

theorem fn_simplified (n : ℕ) : 
  f n = 2 * n.factorial - n - 1 := 
sorry

end fn_simplified_l68_68788


namespace largest_common_term_l68_68732

theorem largest_common_term (n m : ℕ) (k : ℕ) (a : ℕ) 
  (h1 : a = 7 + 7 * n) 
  (h2 : a = 8 + 12 * m) 
  (h3 : 56 + 84 * k < 500) : a = 476 :=
  sorry

end largest_common_term_l68_68732


namespace number_of_BMWs_sold_l68_68097

theorem number_of_BMWs_sold (total_cars_sold : ℕ)
  (percent_Ford percent_Nissan percent_Chevrolet : ℕ)
  (h_total : total_cars_sold = 300)
  (h_percent_Ford : percent_Ford = 18)
  (h_percent_Nissan : percent_Nissan = 25)
  (h_percent_Chevrolet : percent_Chevrolet = 20) :
  (300 * (100 - (percent_Ford + percent_Nissan + percent_Chevrolet)) / 100) = 111 :=
by
  -- We assert that the calculated number of BMWs is 111
  sorry

end number_of_BMWs_sold_l68_68097


namespace total_students_after_new_classes_l68_68458

def initial_classes : ℕ := 15
def students_per_class : ℕ := 20
def new_classes : ℕ := 5

theorem total_students_after_new_classes :
  initial_classes * students_per_class + new_classes * students_per_class = 400 :=
by
  sorry

end total_students_after_new_classes_l68_68458


namespace range_of_a_l68_68687

variable (a : ℝ)

def A (a : ℝ) : Set ℝ := {x : ℝ | (a * x - 1) / (x - a) < 0}

theorem range_of_a (h1 : 2 ∈ A a) (h2 : 3 ∉ A a) : (1 / 3 : ℝ) ≤ a ∧ a < 1 / 2 ∨ 2 < a ∧ a ≤ 3 :=
by
  sorry

end range_of_a_l68_68687


namespace remainder_6_pow_23_mod_5_l68_68129

theorem remainder_6_pow_23_mod_5 : (6 ^ 23) % 5 = 1 := 
by {
  sorry
}

end remainder_6_pow_23_mod_5_l68_68129


namespace least_five_digit_perfect_square_and_cube_l68_68320

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k1 : ℕ, n = k1 ^ 2) ∧ (∃ k2 : ℕ, n = k2 ^ 3) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68320


namespace value_of_a_l68_68421

/-- Given that 0.5% of a is 85 paise, prove that the value of a is 170 rupees. --/
theorem value_of_a (a : ℝ) (h : 0.005 * a = 85) : a = 170 := 
  sorry

end value_of_a_l68_68421


namespace largest_x_63_over_8_l68_68387

theorem largest_x_63_over_8 (x : ℝ) (h1 : ⌊x⌋ / x = 8 / 9) : x = 63 / 8 :=
by
  sorry

end largest_x_63_over_8_l68_68387


namespace no_rational_root_l68_68912

theorem no_rational_root (x : ℚ) : 3 * x^4 - 2 * x^3 - 8 * x^2 + x + 1 ≠ 0 := 
by
  sorry

end no_rational_root_l68_68912


namespace pizza_varieties_l68_68630

-- Definition of the problem conditions
def base_flavors : ℕ := 4
def topping_options : ℕ := 4  -- No toppings, extra cheese, mushrooms, both

-- The math proof problem statement
theorem pizza_varieties : base_flavors * topping_options = 16 := by 
  sorry

end pizza_varieties_l68_68630


namespace chess_tournament_points_distribution_l68_68131

noncomputable def points_distribution (Andrey Dima Vanya Sasha : ℝ) : Prop :=
  ∃ (p_a p_d p_v p_s : ℝ), 
    p_a ≠ p_d ∧ p_d ≠ p_v ∧ p_v ≠ p_s ∧ p_a ≠ p_v ∧ p_a ≠ p_s ∧ p_d ≠ p_s ∧
    p_a + p_d + p_v + p_s = 12 ∧ -- Total points sum
    p_a > p_d ∧ p_d > p_v ∧ p_v > p_s ∧ -- Order of points
    Andrey = p_a ∧ Dima = p_d ∧ Vanya = p_v ∧ Sasha = p_s ∧
    Andrey - (Sasha - 2) = 2 -- Andrey and Sasha won the same number of games

theorem chess_tournament_points_distribution :
  points_distribution 4 3.5 2.5 2 :=
sorry

end chess_tournament_points_distribution_l68_68131


namespace simplify_expression_l68_68483

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = (yz + xz + xy) / (xyz * (x + y + z)) :=
by
  sorry

end simplify_expression_l68_68483


namespace natalie_bushes_needed_l68_68124

theorem natalie_bushes_needed (b c p : ℕ) 
  (h1 : ∀ b, b * 10 = c) 
  (h2 : ∀ c, c * 2 = p)
  (target_p : p = 36) :
  ∃ b, b * 10 ≥ 72 :=
by
  sorry

end natalie_bushes_needed_l68_68124


namespace find_k_l68_68543

-- Define the vectors a, b, and c
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (0, 1)

-- Define the vector c involving variable k
variables (k : ℝ)
def vec_c : ℝ × ℝ := (k, -2)

-- Define the combined vector (a + 2b)
def combined_vec : ℝ × ℝ := (vec_a.1 + 2 * vec_b.1, vec_a.2 + 2 * vec_b.2)

-- Define the dot product function
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- State the theorem to prove
theorem find_k (h : dot_product combined_vec (vec_c k) = 0) : k = 8 :=
by sorry

end find_k_l68_68543


namespace percentage_difference_y_less_than_z_l68_68892

-- Define the variables and the conditions
variables (x y z : ℝ)
variables (h₁ : x = 12 * y)
variables (h₂ : z = 1.2 * x)

-- Define the theorem statement
theorem percentage_difference_y_less_than_z (h₁ : x = 12 * y) (h₂ : z = 1.2 * x) :
  ((z - y) / z) * 100 = 93.06 := by
  sorry

end percentage_difference_y_less_than_z_l68_68892


namespace least_five_digit_is_15625_l68_68284

noncomputable def least_five_digit_perfect_square_and_cube : ℕ :=
  15625

theorem least_five_digit_is_15625 :
  (10000 ≤ least_five_digit_perfect_square_and_cube) ∧ (least_five_digit_perfect_square_and_cube < 100000) ∧
  (∃ a : ℕ, least_five_digit_perfect_square_and_cube = a ^ 6) :=
by
  sorry

end least_five_digit_is_15625_l68_68284


namespace cos_alpha_value_l68_68929

theorem cos_alpha_value (α : ℝ) (h₀ : 0 < α ∧ α < 90) (h₁ : Real.sin (α - 45) = - (Real.sqrt 2 / 10)) : 
  Real.cos α = 4 / 5 := 
sorry

end cos_alpha_value_l68_68929


namespace smallest_four_digit_divisible_by_3_and_8_l68_68081

theorem smallest_four_digit_divisible_by_3_and_8 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 3 = 0 ∧ n % 8 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 3 = 0 ∧ m % 8 = 0 → n ≤ m := by
  sorry

end smallest_four_digit_divisible_by_3_and_8_l68_68081


namespace no_integer_solution_for_large_n_l68_68993

theorem no_integer_solution_for_large_n (n : ℕ) (m : ℤ) (h : n ≥ 11) : ¬(m^2 + 2 * 3^n = m * (2^(n+1) - 1)) :=
sorry

end no_integer_solution_for_large_n_l68_68993


namespace percentage_is_26_53_l68_68779

noncomputable def percentage_employees_with_six_years_or_more (y: ℝ) : ℝ :=
  let total_employees := 10*y + 4*y + 6*y + 5*y + 8*y + 3*y + 5*y + 4*y + 2*y + 2*y
  let employees_with_six_years_or_more := 5*y + 4*y + 2*y + 2*y
  (employees_with_six_years_or_more / total_employees) * 100

theorem percentage_is_26_53 (y: ℝ) (hy: y ≠ 0): percentage_employees_with_six_years_or_more y = 26.53 :=
by
  sorry

end percentage_is_26_53_l68_68779


namespace impossible_to_reduce_time_l68_68356

def current_speed := 60 -- speed in km/h
def time_per_km (v : ℕ) : ℕ := 60 / v -- 60 minutes divided by speed in km/h gives time per km in minutes

theorem impossible_to_reduce_time (v : ℕ) (h : v = current_speed) : time_per_km v = 1 → ¬(time_per_km v - 1 = 0) :=
by
  intros h1 h2
  sorry

end impossible_to_reduce_time_l68_68356


namespace number_of_zeros_is_one_l68_68861

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 3 * x

theorem number_of_zeros_is_one : 
  ∃! x : ℝ, f x = 0 :=
sorry

end number_of_zeros_is_one_l68_68861


namespace part1_part2_l68_68552

variable (a b c : ℝ)

open Classical

noncomputable theory

-- Defining the conditions
def cond_positive_numbers : Prop := (0 < a) ∧ (0 < b) ∧ (0 < c)
def cond_main_equation : Prop := a^2 + b^2 + 4*c^2 = 3
def cond_b_eq_2c : Prop := b = 2*c

-- Statement for part (1)
theorem part1 (h1 : cond_positive_numbers a b c) (h2 : cond_main_equation a b c) :
  a + b + 2*c ≤ 3 := sorry

-- Statement for part (2)
theorem part2 (h1 : cond_positive_numbers a b c) (h2 : cond_main_equation a b c) (h3 : cond_b_eq_2c b c) :
  (1 / a) + (1 / c) ≥ 3 := sorry

end part1_part2_l68_68552


namespace chess_tournament_distribution_l68_68144

theorem chess_tournament_distribution 
    (students : List String)
    (games_played : Nat)
    (scores : List ℝ)
    (points_per_game : List ℝ)
    (unique_scores : ∀ (x y : ℝ), x ≠ y → scores.contains x → scores.contains y → x ≠ y)
    (first_place : String)
    (second_place : String)
    (third_place : String)
    (fourth_place : String)
    (andrey_wins_equal_sasha : ℝ)
    (total_points : ℝ)
    : 
    students = ["Andrey", "Vanya", "Dima", "Sasha"] ∧
    games_played = 6 ∧
    points_per_game = [1, 0.5, 0] ∧
    first_place = "Andrey" ∧
    second_place = "Dima" ∧
    third_place = "Vanya" ∧
    fourth_place = "Sasha" ∧
    scores = [4, 3.5, 2.5, 2] ∧
    andrey_wins_equal_sasha = 2 ∧
    total_points = 12 := 
sorry

end chess_tournament_distribution_l68_68144


namespace find_missing_number_l68_68125

theorem find_missing_number (x : ℤ) : x + 64 = 16 → x = -48 := by
  intro h
  linarith

end find_missing_number_l68_68125


namespace find_a_sq_plus_b_sq_l68_68932

theorem find_a_sq_plus_b_sq (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 10) :
  a^2 + b^2 = 29 := by
  sorry

end find_a_sq_plus_b_sq_l68_68932


namespace area_of_isosceles_right_triangle_l68_68666

def is_isosceles_right_triangle (a b c : ℝ) : Prop :=
  (a = b) ∧ (a^2 + b^2 = c^2)

theorem area_of_isosceles_right_triangle (a : ℝ) (hypotenuse : ℝ) (h_isosceles : is_isosceles_right_triangle a a hypotenuse) (h_hypotenuse : hypotenuse = 6) :
  (1 / 2) * a * a = 9 :=
by
  sorry

end area_of_isosceles_right_triangle_l68_68666


namespace line_parabola_intersection_one_point_l68_68236

theorem line_parabola_intersection_one_point (k : ℝ) :
  (∃ y : ℝ, (-3 * y^2 - 4 * y + 7 = k) ∧ ∀ y1 y2 : ℝ, ( 3 * y1^2 + 4 * y1 + (k - 7) = 0 → 3 * y2^2 + 4 * y2 + (k - 7) = 0 → y1 = y2)) ↔ (k = 25 / 3) :=
by
  sorry

end line_parabola_intersection_one_point_l68_68236


namespace construct_rectangle_l68_68662

-- Define the essential properties of the rectangles
structure Rectangle where
  length : ℕ
  width : ℕ 

-- Define the given rectangles
def r1 : Rectangle := ⟨7, 1⟩
def r2 : Rectangle := ⟨6, 1⟩
def r3 : Rectangle := ⟨5, 1⟩
def r4 : Rectangle := ⟨4, 1⟩
def r5 : Rectangle := ⟨3, 1⟩
def r6 : Rectangle := ⟨2, 1⟩
def s  : Rectangle := ⟨1, 1⟩

-- Hypothesis for condition that length of each side of resulting rectangle should be > 1
def validSide (rect : Rectangle) : Prop :=
  rect.length > 1 ∧ rect.width > 1

-- The proof statement
theorem construct_rectangle : 
  (∃ rect1 rect2 rect3 rect4 : Rectangle, 
      rect1 = ⟨7, 1⟩ ∧ rect2 = ⟨6, 1⟩ ∧ rect3 = ⟨5, 1⟩ ∧ rect4 = ⟨4, 1⟩) →
  (∃ rect5 rect6 : Rectangle, 
      rect5 = ⟨3, 1⟩ ∧ rect6 = ⟨2, 1⟩) →
  (∃ square : Rectangle, 
      square = ⟨1, 1⟩) →
  (∃ compositeRect : Rectangle, 
      compositeRect.length = 7 ∧ 
      compositeRect.width = 4 ∧ 
      validSide compositeRect) :=
sorry

end construct_rectangle_l68_68662


namespace least_five_digit_perfect_square_and_cube_l68_68323

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k1 : ℕ, n = k1 ^ 2) ∧ (∃ k2 : ℕ, n = k2 ^ 3) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68323


namespace differentiate_and_evaluate_l68_68928

theorem differentiate_and_evaluate (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) (x : ℝ) :
  (2*x - 1)^6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 →
  a_1 + 2*a_2 + 3*a_3 + 4*a_4 + 5*a_5 + 6*a_6 = 12 :=
sorry

end differentiate_and_evaluate_l68_68928


namespace area_of_shaded_region_l68_68585

open Real

-- Define points and squares
structure Point (α : Type*) := (x : α) (y : α)

def A := Point.mk 0 12 -- top-left corner of large square
def G := Point.mk 0 0  -- bottom-left corner of large square
def F := Point.mk 4 0  -- bottom-right corner of small square
def E := Point.mk 4 4  -- top-right corner of small square
def C := Point.mk 12 0 -- bottom-right corner of large square
def D := Point.mk 3 0  -- intersection of AF extended with the bottom edge

-- Define the length of sides
def side_small_square : ℝ := 4
def side_large_square : ℝ := 12

-- Areas calculation
def area_square (side : ℝ) : ℝ := side * side

def area_triangle (base height : ℝ) : ℝ := 0.5 * base * height

-- Theorem statement
theorem area_of_shaded_region : area_square side_small_square - area_triangle 3 side_small_square = 10 :=
by
  rw [area_square, area_triangle]
  -- Plug in values: 4^2 - 0.5 * 3 * 4
  norm_num
  sorry

end area_of_shaded_region_l68_68585


namespace find_x_l68_68857

theorem find_x (x : ℝ) (h : (20 + 30 + 40 + x) / 4 = 35) : x = 50 := by
  sorry

end find_x_l68_68857


namespace percentage_of_boys_answered_neither_l68_68821

theorem percentage_of_boys_answered_neither (P_A P_B P_A_and_B : ℝ) (hP_A : P_A = 0.75) (hP_B : P_B = 0.55) (hP_A_and_B : P_A_and_B = 0.50) :
  1 - (P_A + P_B - P_A_and_B) = 0.20 :=
by
  sorry

end percentage_of_boys_answered_neither_l68_68821


namespace part1_part2_l68_68551

variable (a b c : ℝ)

-- Conditions
axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : c > 0
axiom h4 : a^2 + b^2 + 4*c^2 = 3

-- Part 1: Prove that a + b + 2c ≤ 3
theorem part1 : a + b + 2*c ≤ 3 := sorry

-- Part 2: Given b = 2c, prove that 1/a + 1/c ≥ 3
axiom h5 : b = 2*c
theorem part2 : 1/a + 1/c ≥ 3 := sorry

end part1_part2_l68_68551


namespace least_positive_integer_reducible_fraction_l68_68535

theorem least_positive_integer_reducible_fraction :
  ∃ n : ℕ, n > 0 ∧ gcd (n - 17) (7 * n + 4) > 1 ∧ (∀ m : ℕ, m > 0 ∧ gcd (m - 17) (7 * m + 4) > 1 → n ≤ m) :=
by sorry

end least_positive_integer_reducible_fraction_l68_68535


namespace area_of_rectangle_l68_68521

noncomputable def length := 44.4
noncomputable def width := 29.6

theorem area_of_rectangle (h1 : width = 2 / 3 * length) (h2 : 2 * (length + width) = 148) : 
  (length * width) = 1314.24 := 
by 
  sorry

end area_of_rectangle_l68_68521


namespace total_cost_correct_l68_68330

-- Conditions given in the problem.
def net_profit : ℝ := 44
def gross_revenue : ℝ := 47
def lemonades_sold : ℝ := 50
def babysitting_income : ℝ := 31

def cost_per_lemon : ℝ := 0.20
def cost_per_sugar : ℝ := 0.15
def cost_per_ice : ℝ := 0.05

def one_time_cost_sunhat : ℝ := 10

-- Definition of variable cost per lemonade.
def variable_cost_per_lemonade : ℝ := cost_per_lemon + cost_per_sugar + cost_per_ice

-- Definition of total variable cost for all lemonades sold.
def total_variable_cost : ℝ := lemonades_sold * variable_cost_per_lemonade

-- Final total cost to operate the lemonade stand.
def total_cost : ℝ := total_variable_cost + one_time_cost_sunhat

-- The proof statement that total cost is equal to $30.
theorem total_cost_correct : total_cost = 30 := by
  sorry

end total_cost_correct_l68_68330


namespace find_c_l68_68448

theorem find_c (b c : ℤ) (H : (b - 4) / (2 * b + 42) = c / 6) : c = 2 := 
sorry

end find_c_l68_68448


namespace mary_final_books_l68_68044

-- Initial number of books
def initial_books : ℕ := 72

-- Books received each month from book club for 12 months
def books_from_club : ℕ := 12 * 1

-- Books bought from different sources
def books_from_bookstore : ℕ := 5
def books_from_yard_sales : ℕ := 2

-- Books received as gifts
def books_from_daughter : ℕ := 1
def books_from_mother : ℕ := 4

-- Books gotten rid of
def books_donated : ℕ := 12
def books_sold : ℕ := 3

-- Final calculation
theorem mary_final_books : 
  initial_books + books_from_club + books_from_bookstore + books_from_yard_sales + books_from_daughter + books_from_mother - (books_donated + books_sold) = 81 :=
  by sorry

end mary_final_books_l68_68044


namespace heat_released_is_1824_l68_68362

def ΔH_f_NH3 : ℝ := -46  -- Enthalpy of formation of NH3 in kJ/mol
def ΔH_f_H2SO4 : ℝ := -814  -- Enthalpy of formation of H2SO4 in kJ/mol
def ΔH_f_NH4SO4 : ℝ := -909  -- Enthalpy of formation of (NH4)2SO4 in kJ/mol

def ΔH_rxn : ℝ :=
  2 * ΔH_f_NH4SO4 - (2 * ΔH_f_NH3 + ΔH_f_H2SO4)  -- Reaction enthalpy change

def heat_released : ℝ := 2 * ΔH_rxn  -- Heat released for 4 moles of NH3

theorem heat_released_is_1824 : heat_released = -1824 :=
by
  -- Theorem statement for proving heat released is 1824 kJ
  sorry

end heat_released_is_1824_l68_68362


namespace least_five_digit_perfect_square_and_cube_l68_68299

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (n >= 10000 ∧ n < 100000) ∧ (∃ a : ℕ, n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68299


namespace least_five_digit_whole_number_is_perfect_square_and_cube_l68_68264

theorem least_five_digit_whole_number_is_perfect_square_and_cube :
  ∃ (n : ℕ), (10000 ≤ n ∧ n < 100000) ∧ (∃ (a : ℕ), n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_whole_number_is_perfect_square_and_cube_l68_68264


namespace missy_yells_at_obedient_dog_12_times_l68_68201

theorem missy_yells_at_obedient_dog_12_times (x : ℕ) (h : x + 4 * x = 60) : x = 12 :=
by
  -- Proof steps can be filled in here
  sorry

end missy_yells_at_obedient_dog_12_times_l68_68201


namespace mass_percentage_oxygen_NaBrO3_l68_68371

-- Definitions
def molar_mass_Na : ℝ := 22.99
def molar_mass_Br : ℝ := 79.90
def molar_mass_O : ℝ := 16.00

def molar_mass_NaBrO3 : ℝ := molar_mass_Na + molar_mass_Br + 3 * molar_mass_O

-- Theorem: proof that the mass percentage of oxygen in NaBrO3 is 31.81%
theorem mass_percentage_oxygen_NaBrO3 :
  ((3 * molar_mass_O) / molar_mass_NaBrO3) * 100 = 31.81 := by
  sorry

end mass_percentage_oxygen_NaBrO3_l68_68371


namespace fathers_age_l68_68647

variable (S F : ℕ)
variable (h1 : F = 3 * S)
variable (h2 : F + 15 = 2 * (S + 15))

theorem fathers_age : F = 45 :=
by
  -- the proof steps would go here
  sorry

end fathers_age_l68_68647


namespace statement_A_l68_68905

theorem statement_A (x : ℝ) (h : x > 1) : x^2 > x := 
by
  sorry

end statement_A_l68_68905


namespace work_last_duration_l68_68337

theorem work_last_duration
  (work_rate_x : ℚ := 1 / 20)
  (work_rate_y : ℚ := 1 / 12)
  (days_x_worked_alone : ℚ := 4)
  (combined_work_rate : ℚ := work_rate_x + work_rate_y)
  (remaining_work : ℚ := 1 - days_x_worked_alone * work_rate_x) :
  (remaining_work / combined_work_rate + days_x_worked_alone = 10) :=
by
  sorry

end work_last_duration_l68_68337


namespace first_part_amount_l68_68729

-- Given Definitions
def total_amount : ℝ := 3200
def interest_rate_part1 : ℝ := 0.03
def interest_rate_part2 : ℝ := 0.05
def total_interest : ℝ := 144

-- The problem to be proven
theorem first_part_amount : 
  ∃ (x : ℝ), 0.03 * x + 0.05 * (3200 - x) = 144 ∧ x = 800 :=
by
  sorry

end first_part_amount_l68_68729


namespace traci_flour_l68_68745

variable (HarrisFlour : ℕ) (cakeFlour : ℕ) (cakesEach : ℕ)

theorem traci_flour (HarrisFlour := 400) (cakeFlour := 100) (cakesEach := 9) :
  ∃ (TraciFlour : ℕ), 
  (cakesEach * 2 * cakeFlour) - HarrisFlour = TraciFlour ∧ 
  TraciFlour = 1400 :=
by
  have totalCakes : ℕ := cakesEach * 2
  have totalFlourNeeded : ℕ := totalCakes * cakeFlour
  have TraciFlour := totalFlourNeeded - HarrisFlour
  exact ⟨TraciFlour, rfl, rfl⟩

end traci_flour_l68_68745


namespace functional_equation_solution_l68_68529

open Function

theorem functional_equation_solution :
  ∀ (f g : ℚ → ℚ), 
    (∀ x y : ℚ, f (g x + g y) = f (g x) + y ∧ g (f x + f y) = g (f x) + y) →
    (∃ a b : ℚ, (ab = 1) ∧ (∀ x : ℚ, f x = a * x) ∧ (∀ x : ℚ, g x = b * x)) :=
by
  intros f g h
  sorry

end functional_equation_solution_l68_68529


namespace find_13_real_coins_find_15_real_coins_cannot_find_17_real_coins_l68_68492

noncomputable def board : Type := (Fin 5) × (Fin 5)

def is_counterfeit (c1 : board) (c2 : board) : Prop :=
  (c1.1 = c2.1 ∧ (c1.2 = c2.2 + 1 ∨ c1.2 + 1 = c2.2)) ∨
  (c1.2 = c2.2 ∧ (c1.1 = c2.1 + 1 ∨ c1.1 + 1 = c2.1))

theorem find_13_real_coins (coins : board → ℝ) (c1 c2 : board) :
  (coins c1 < coins (0,0) ∧ coins c2 < coins (0,0)) ∧ is_counterfeit c1 c2 →
  ∃ C : Finset board, C.card = 13 ∧ ∀ c ∈ C, coins c = coins (0,0) :=
sorry

theorem find_15_real_coins (coins : board → ℝ) (c1 c2 : board) :
  (coins c1 < coins (0,0) ∧ coins c2 < coins (0,0)) ∧ is_counterfeit c1 c2 →
  ∃ C : Finset board, C.card = 15 ∧ ∀ c ∈ C, coins c = coins (0,0) :=
sorry

theorem cannot_find_17_real_coins (coins : board → ℝ) (c1 c2 : board) :
  (coins c1 < coins (0,0) ∧ coins c2 < coins (0,0)) ∧ is_counterfeit c1 c2 →
  ¬ (∃ C : Finset board, C.card = 17 ∧ ∀ c ∈ C, coins c = coins (0,0)) :=
sorry

end find_13_real_coins_find_15_real_coins_cannot_find_17_real_coins_l68_68492


namespace minute_hand_position_l68_68867

theorem minute_hand_position (minutes: ℕ) : (minutes ≡ 28 [% 60]) :=
  -- Define the cycle behavior and prove the end result
  let full_cycle_minutes := 8
  let forward_movement := 5
  let backward_movement := 3
  let net_movement_per_cycle := forward_movement - backward_movement
  let number_of_cycles := minutes / full_cycle_minutes
  let remaining_minutes := minutes % full_cycle_minutes
  let total_forward_movement := number_of_cycles * net_movement_per_cycle + 
    if remaining_minutes >= forward_movement 
    then forward_movement - remaining_minutes + backward_movement 
    else remaining_minutes
  in total_forward_movement ≡ 28 [% 60]

end minute_hand_position_l68_68867


namespace gcd_poly_l68_68002

theorem gcd_poly (b : ℤ) (h : ∃ k : ℤ, b = 17 * (2 * k + 1)) : 
  Int.gcd (4 * b ^ 2 + 63 * b + 144) (2 * b + 7) = 1 := 
by 
  sorry

end gcd_poly_l68_68002


namespace cindy_added_pens_l68_68626

-- Define the initial number of pens
def initial_pens : ℕ := 5

-- Define the number of pens given by Mike
def pens_from_mike : ℕ := 20

-- Define the number of pens given to Sharon
def pens_given_to_sharon : ℕ := 10

-- Define the final number of pens
def final_pens : ℕ := 40

-- Formulate the theorem regarding the pens added by Cindy
theorem cindy_added_pens :
  final_pens = initial_pens + pens_from_mike - pens_given_to_sharon + 25 :=
by
  sorry

end cindy_added_pens_l68_68626


namespace smallest_k_remainder_1_l68_68622

theorem smallest_k_remainder_1
  (k : ℤ) : 
  (k > 1) ∧ (k % 13 = 1) ∧ (k % 8 = 1) ∧ (k % 4 = 1)
  ↔ k = 105 :=
by
  sorry

end smallest_k_remainder_1_l68_68622


namespace Kyle_is_25_l68_68711

variable (Tyson_age : ℕ := 20)
variable (Frederick_age : ℕ := 2 * Tyson_age)
variable (Julian_age : ℕ := Frederick_age - 20)
variable (Kyle_age : ℕ := Julian_age + 5)

theorem Kyle_is_25 : Kyle_age = 25 := by
  sorry

end Kyle_is_25_l68_68711


namespace bus_passenger_count_l68_68462

-- Definitions for conditions
def initial_passengers : ℕ := 0
def passengers_first_stop (initial : ℕ) : ℕ := initial + 7
def passengers_second_stop (after_first : ℕ) : ℕ := after_first - 3 + 5
def passengers_third_stop (after_second : ℕ) : ℕ := after_second - 2 + 4

-- Statement we want to prove
theorem bus_passenger_count : 
  passengers_third_stop (passengers_second_stop (passengers_first_stop initial_passengers)) = 11 :=
by
  -- proof would go here
  sorry

end bus_passenger_count_l68_68462


namespace chess_tournament_solution_l68_68133

def chess_tournament_points (points : List ℝ) : Prop :=
  let andrey := points[0]
  let dima := points[1]
  let vanya := points[2]
  let sasha := points[3]
  andrey = 4 ∧ dima = 3.5 ∧ vanya = 2.5 ∧ sasha = 2

axiom chess_tournament_conditions (points : List ℝ) :
  -- Andrey secured first place, Dima secured second, Vanya secured third, and Sasha secured fourth.
  List.Nodup points ∧
  points.length = 4 ∧
  (∀ p, p ∈ points → p = 4 ∨ p = 3.5 ∨ p = 2.5 ∨ p = 2) ∧
  -- Andrey and Sasha won the same number of games.
  (points[0] ≠ points[1] ∧ points[0] ≠ points[2] ∧ points[0] ≠ points[3] ∧
   points[1] ≠ points[2] ∧ points[1] ≠ points[3] ∧
   points[2] ≠ points[3])

theorem chess_tournament_solution (points : List ℝ) :
  chess_tournament_conditions points → chess_tournament_points points :=
by
  sorry

end chess_tournament_solution_l68_68133


namespace least_five_digit_whole_number_is_perfect_square_and_cube_l68_68262

theorem least_five_digit_whole_number_is_perfect_square_and_cube :
  ∃ (n : ℕ), (10000 ≤ n ∧ n < 100000) ∧ (∃ (a : ℕ), n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_whole_number_is_perfect_square_and_cube_l68_68262


namespace profit_percentage_l68_68342

theorem profit_percentage (SP CP : ℝ) (hs : SP = 270) (hc : CP = 225) : 
  ((SP - CP) / CP) * 100 = 20 :=
by
  rw [hs, hc]
  sorry  -- The proof will go here

end profit_percentage_l68_68342


namespace measure_8_liters_with_two_buckets_l68_68568

def bucket_is_empty (B : ℕ) : Prop :=
  B = 0

def bucket_has_capacity (B : ℕ) (c : ℕ) : Prop :=
  B ≤ c

def fill_bucket (B : ℕ) (c : ℕ) : ℕ :=
  c

def empty_bucket (B : ℕ) : ℕ :=
  0

def pour_bucket (B1 B2 : ℕ) (c1 c2 : ℕ) : (ℕ × ℕ) :=
  if B1 + B2 <= c2 then (0, B1 + B2)
  else (B1 - (c2 - B2), c2)

theorem measure_8_liters_with_two_buckets (B10 B6 : ℕ) (c10 c6 : ℕ) :
  bucket_has_capacity B10 c10 ∧ bucket_has_capacity B6 c6 ∧
  c10 = 10 ∧ c6 = 6 →
  ∃ B10' B6', B10' = 8 ∧ B6' ≤ 6 :=
by
  intros h
  have h1 : ∃ B1, bucket_is_empty B1,
    from ⟨0, rfl⟩
  let B10 := fill_bucket 0 c10
  let ⟨B10, B6⟩ := pour_bucket B10 0 c10 c6
  let B6 := empty_bucket B6
  let ⟨B10, B6⟩ := pour_bucket B10 B6 c10 c6
  let B10 := fill_bucket B10 c10
  let ⟨B10, B6⟩ := pour_bucket B10 B6 c10 c6
  exact ⟨B10, B6, rfl, le_refl 6⟩

end measure_8_liters_with_two_buckets_l68_68568


namespace cuboid_surface_area_l68_68863

-- Definition of the problem with given conditions and the statement we need to prove.
theorem cuboid_surface_area (h l w: ℝ) (H1: 4 * (2 * h) + 4 * (2 * h) + 4 * h = 100)
                            (H2: l = 2 * h)
                            (H3: w = 2 * h) :
                            (2 * (l * w + l * h + w * h) = 400) :=
by
  sorry

end cuboid_surface_area_l68_68863


namespace sum_tens_ones_digits_of_6_pow_15_l68_68752

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_tens_ones_digits_of_6_pow_15 :
  tens_digit (6^15) + ones_digit (6^15) = 13 :=
by
  -- we simplify 6^15 mod 100
  have h : (6^15 : Zmod 100) = 76 := sorry
  
  -- tens digit of 76 is 7
  have tens : tens_digit (6^15) = 7 := by
    rw h
    exact rfl
  
  -- ones digit of 76 is 6
  have ones : ones_digit (6^15) = 6 := by
    rw h
    exact rfl
  
  -- sum of tens and ones digits is 13
  rw [tens, ones]
  exact rfl

sorry

end sum_tens_ones_digits_of_6_pow_15_l68_68752


namespace number_of_girls_in_school_l68_68881

theorem number_of_girls_in_school
  (total_students : ℕ)
  (avg_age_boys avg_age_girls avg_age_school : ℝ)
  (B G : ℕ)
  (h1 : total_students = 640)
  (h2 : avg_age_boys = 12)
  (h3 : avg_age_girls = 11)
  (h4 : avg_age_school = 11.75)
  (h5 : B + G = total_students)
  (h6 : (avg_age_boys * B + avg_age_girls * G = avg_age_school * total_students)) :
  G = 160 :=
by
  sorry

end number_of_girls_in_school_l68_68881


namespace compute_x_plus_y_l68_68152

theorem compute_x_plus_y :
    ∃ (x y : ℕ), 4 * y = 7 * 84 ∧ 4 * 63 = 7 * x ∧ x + y = 183 :=
by
  sorry

end compute_x_plus_y_l68_68152


namespace least_five_digit_perfect_square_and_cube_l68_68300

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (n >= 10000 ∧ n < 100000) ∧ (∃ a : ℕ, n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68300


namespace ratio_girls_to_boys_l68_68336

-- Definitions of the conditions
def numGirls : ℕ := 10
def numBoys : ℕ := 20

-- Statement of the proof problem
theorem ratio_girls_to_boys : (numGirls / Nat.gcd numGirls numBoys) = 1 ∧ (numBoys / Nat.gcd numGirls numBoys) = 2 :=
by
  sorry

end ratio_girls_to_boys_l68_68336


namespace remainder_when_divided_l68_68981

theorem remainder_when_divided (a b : ℕ) (n m : ℤ) (H1 : a ≡ 64 [MOD 70]) (H2 : b ≡ 99 [MOD 105]) : (a + b) % 35 = 23 :=
by
  sorry

end remainder_when_divided_l68_68981


namespace profit_percentage_l68_68759

theorem profit_percentage (CP SP : ℝ) (h : 18 * CP = 16 * SP) : 
  (SP - CP) / CP * 100 = 12.5 := by
sorry

end profit_percentage_l68_68759


namespace find_a_l68_68973

theorem find_a (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 119) : a = 59 :=
sorry

end find_a_l68_68973


namespace least_five_digit_perfect_square_and_cube_l68_68289

theorem least_five_digit_perfect_square_and_cube : 
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ (∃ a : ℕ, n = a ^ 6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68289


namespace john_burritos_left_l68_68435

theorem john_burritos_left : 
  let total_boxes := 3 
  let burritos_per_box := 20
  let total_burritos := total_boxes * burritos_per_box
  let burritos_given_away := total_burritos / 3
  let burritos_left_after_giving := total_burritos - burritos_given_away
  let burritos_eaten_per_day := 3
  let days := 10
  let total_burritos_eaten := burritos_eaten_per_day * days
  let burritos_left := burritos_left_after_giving - total_burritos_eaten
  in burritos_left = 10 := by
  let total_boxes := 3 
  let burritos_per_box := 20
  let total_burritos := total_boxes * burritos_per_box
  let burritos_given_away := total_burritos / 3
  let burritos_left_after_giving := total_burritos - burritos_given_away
  let burritos_eaten_per_day := 3
  let days := 10
  let total_burritos_eaten := burritos_eaten_per_day * days
  let burritos_left := burritos_left_after_giving - total_burritos_eaten
  have h : total_burritos = 60 := by rfl
  have h1 : burritos_given_away = 20 := by sorry
  have h2 : burritos_left_after_giving = 40 := by sorry
  have h3 : total_burritos_eaten = 30 := by sorry
  have h4 : burritos_left = 10 := by sorry
  exact h4 -- Concluding that burritos_left = 10

end john_burritos_left_l68_68435


namespace least_five_digit_perfect_square_and_cube_l68_68286

theorem least_five_digit_perfect_square_and_cube : 
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ (∃ a : ℕ, n = a ^ 6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68286


namespace work_days_B_works_l68_68496

theorem work_days_B_works (x : ℕ) (A_work_rate B_work_rate : ℚ) (A_remaining_days : ℕ) (total_work : ℚ) :
  A_work_rate = (1 / 12) ∧
  B_work_rate = (1 / 15) ∧
  A_remaining_days = 4 ∧
  total_work = 1 →
  x * B_work_rate + A_remaining_days * A_work_rate = total_work →
  x = 10 :=
sorry

end work_days_B_works_l68_68496


namespace breakfast_calories_l68_68436

variable (B : ℝ) 

def lunch_calories := 1.25 * B
def dinner_calories := 2.5 * B
def shakes_calories := 900
def total_calories := 3275

theorem breakfast_calories:
  (B + lunch_calories B + dinner_calories B + shakes_calories = total_calories) → B = 500 :=
by
  sorry

end breakfast_calories_l68_68436


namespace least_five_digit_perfect_square_and_cube_l68_68258

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ ∃ k : ℕ, k^6 = n ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68258


namespace resulting_parabola_is_correct_l68_68618

-- Conditions
def initial_parabola (x : ℝ) : ℝ := x^2 - 2

def translate_right (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ :=
  λ x, f (x - a)

def translate_up (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ :=
  λ y, f y + b

-- The equivalent proof problem
theorem resulting_parabola_is_correct :
  (translate_up (translate_right initial_parabola 1) 3) = (λ x, (x - 1)^2 + 1) :=
sorry

end resulting_parabola_is_correct_l68_68618


namespace min_value_fraction_l68_68808

theorem min_value_fraction (x y : ℝ) (h₁ : x + y = 4) (h₂ : x > y) (h₃ : y > 0) : (∃ z : ℝ, z = (2 / (x - y)) + (1 / y) ∧ z = 2) :=
by
  sorry

end min_value_fraction_l68_68808


namespace workload_increase_l68_68581

theorem workload_increase (a b c d p : ℕ) (h : p ≠ 0) :
  let total_workload := a + b + c + d
  let workload_per_worker := total_workload / p
  let absent_workers := p / 4
  let remaining_workers := p - absent_workers
  let workload_per_remaining_worker := total_workload / (3 * p / 4)
  workload_per_remaining_worker = (a + b + c + d) * 4 / (3 * p) :=
by
  sorry

end workload_increase_l68_68581


namespace more_girls_than_boys_l68_68250

theorem more_girls_than_boys (total_kids girls boys : ℕ) (h1 : total_kids = 34) (h2 : girls = 28) (h3 : total_kids = girls + boys) : girls - boys = 22 :=
by
  -- Proof placeholder
  sorry

end more_girls_than_boys_l68_68250


namespace sufficient_not_necessary_l68_68765

theorem sufficient_not_necessary (b c: ℝ) : (c < 0) → ∃ x y : ℝ, x^2 + b * x + c = 0 ∧ y^2 + b * y + c = 0 :=
by
  sorry

end sufficient_not_necessary_l68_68765


namespace rhombus_perimeter_l68_68062

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) : 
  let s := real.sqrt ((d1 / 2)^2 + (d2 / 2)^2)
  in 4 * s = 16 * real.sqrt 13 := 
by {
  let a := d1 / 2,
  let b := d2 / 2,
  have h_a : a = 12 := by { rw h1, norm_num },
  have h_b : b = 8 := by { rw h2, norm_num },
  have h_s : s = real.sqrt (a^2 + b^2) := by refl,
  have h_s_val : s = 4 * real.sqrt 13 := by {
    rw [h_a, h_b], 
    norm_num,
    simp [real.sqrt_mul (show 16 > 0, by norm_num), show 13 > 0, by norm_num]
  },
  rw h_s_val,
  norm_num
}

end rhombus_perimeter_l68_68062


namespace hall_length_width_difference_l68_68335

theorem hall_length_width_difference (L W : ℝ) 
(h1 : W = 1 / 2 * L) 
(h2 : L * W = 200) : L - W = 10 := 
by 
  sorry

end hall_length_width_difference_l68_68335


namespace find_a_l68_68412

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 4 * x^2 + 3 * x
noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 8 * x + 3

theorem find_a (a : ℝ) : f_prime a 1 = 2 → a = -3 := by
  intros h
  -- skipping the proof, as it is not required
  sorry

end find_a_l68_68412


namespace factors_of_m_multiples_of_200_l68_68419

theorem factors_of_m_multiples_of_200 (m : ℕ) (h : m = 2^12 * 3^10 * 5^9) : 
  (∃ k, 200 * k ≤ m ∧ ∃ a b c, k = 2^a * 3^b * 5^c ∧ 3 ≤ a ∧ a ≤ 12 ∧ 2 ≤ c ∧ c ≤ 9 ∧ 0 ≤ b ∧ b ≤ 10) := 
by sorry

end factors_of_m_multiples_of_200_l68_68419


namespace converse_inverse_l68_68163

-- Define the properties
def is_parallelogram (polygon : Type) : Prop := sorry -- needs definitions about polygons
def has_two_pairs_of_parallel_sides (polygon : Type) : Prop := sorry -- needs definitions about polygons

-- The given condition
axiom parallelogram_implies_parallel_sides (polygon : Type) :
  is_parallelogram polygon → has_two_pairs_of_parallel_sides polygon

-- Proof of the converse:
theorem converse (polygon : Type) :
  has_two_pairs_of_parallel_sides polygon → is_parallelogram polygon := sorry

-- Proof of the inverse:
theorem inverse (polygon : Type) :
  ¬is_parallelogram polygon → ¬has_two_pairs_of_parallel_sides polygon := sorry

end converse_inverse_l68_68163


namespace total_revenue_is_correct_l68_68586

-- Joan decided to sell all of her old books.
-- She had 33 books in total.
-- She sold 15 books at $4 each.
-- She sold 6 books at $7 each.
-- The rest of the books were sold at $10 each.
-- We need to prove that the total revenue is $222.

def totalBooks := 33
def booksAt4 := 15
def priceAt4 := 4
def booksAt7 := 6
def priceAt7 := 7
def priceAt10 := 10
def remainingBooks := totalBooks - (booksAt4 + booksAt7)
def revenueAt4 := booksAt4 * priceAt4
def revenueAt7 := booksAt7 * priceAt7
def revenueAt10 := remainingBooks * priceAt10
def totalRevenue := revenueAt4 + revenueAt7 + revenueAt10

theorem total_revenue_is_correct : totalRevenue = 222 := by
  sorry

end total_revenue_is_correct_l68_68586


namespace modulus_remainder_l68_68799

theorem modulus_remainder (n : ℕ) 
  (h1 : n^3 % 7 = 3) 
  (h2 : n^4 % 7 = 2) : 
  n % 7 = 6 :=
by
  sorry

end modulus_remainder_l68_68799


namespace k1_k2_ratio_l68_68839

theorem k1_k2_ratio (a b k k1 k2 : ℝ)
  (h1 : a^2 * k - (k - 1) * a + 5 = 0)
  (h2 : b^2 * k - (k - 1) * b + 5 = 0)
  (h3 : (a / b) + (b / a) = 4/5)
  (h4 : k1^2 - 16 * k1 + 1 = 0)
  (h5 : k2^2 - 16 * k2 + 1 = 0) :
  (k1 / k2) + (k2 / k1) = 254 := by
  sorry

end k1_k2_ratio_l68_68839


namespace allocation_methods_count_l68_68103

def number_of_allocation_methods (doctors nurses : ℕ) (hospitals : ℕ) (nurseA nurseB : ℕ) :=
  if (doctors = 3) ∧ (nurses = 6) ∧ (hospitals = 3) ∧ (nurseA = 1) ∧ (nurseB = 1) then 684 else 0

theorem allocation_methods_count :
  number_of_allocation_methods 3 6 3 2 2 = 684 :=
by
  sorry

end allocation_methods_count_l68_68103


namespace sequence_inequality_l68_68739

open Real

def seq (F : ℕ → ℝ) : Prop :=
  F 1 = 1 ∧ F 2 = 2 ∧ ∀ n ≥ 1, F (n + 2) = F (n + 1) + F n

theorem sequence_inequality (F : ℕ → ℝ) (h : seq F) (n : ℕ) : 
  sqrt (F (n+1))^(1/(n:ℝ)) ≥ 1 + 1 / sqrt (F n)^(1/(n:ℝ)) :=
sorry

end sequence_inequality_l68_68739


namespace find_y_l68_68806

variable (a b c x : ℝ) (p q r y : ℝ)
variable (log : ℝ → ℝ) -- represents the logarithm function

-- Conditions as hypotheses
axiom log_eq : (log a) / p = (log b) / q
axiom log_eq' : (log b) / q = (log c) / r
axiom log_eq'' : (log c) / r = log x
axiom x_ne_one : x ≠ 1
axiom eq_exp : (b^3) / (a^2 * c) = x^y

-- Statement to be proven
theorem find_y : y = 3 * q - 2 * p - r := by
  sorry

end find_y_l68_68806


namespace arith_sqrt_abs_neg_nine_l68_68884

theorem arith_sqrt_abs_neg_nine : Real.sqrt (abs (-9)) = 3 := by
  sorry

end arith_sqrt_abs_neg_nine_l68_68884


namespace evaluate_expression_l68_68693

-- Defining the primary condition
def condition (x : ℝ) : Prop := x > 3

-- Definition of the expression we need to evaluate
def expression (x : ℝ) : ℝ := abs (1 - abs (x - 3))

-- Stating the theorem
theorem evaluate_expression (x : ℝ) (h : condition x) : expression x = abs (4 - x) := 
by 
  -- Since the problem only asks for the statement, the proof is left as sorry.
  sorry

end evaluate_expression_l68_68693


namespace find_D_l68_68000

noncomputable def point := (ℝ × ℝ)

def vector_add (u v : point) : point := (u.1 + v.1, u.2 + v.2)
def vector_sub (u v : point) : point := (u.1 - v.1, u.2 - v.2)
def scalar_multiplication (k : ℝ) (u : point) : point := (k * u.1, k * u.2)

namespace GeometryProblem

def A : point := (2, 3)
def B : point := (-1, 5)

def D : point := 
  let AB := vector_sub B A
  vector_add A (scalar_multiplication 3 AB)

theorem find_D : D = (-7, 9) := by
  sorry

end GeometryProblem

end find_D_l68_68000


namespace part1_part2_l68_68553

variable (a b c : ℝ)

open Classical

noncomputable theory

-- Defining the conditions
def cond_positive_numbers : Prop := (0 < a) ∧ (0 < b) ∧ (0 < c)
def cond_main_equation : Prop := a^2 + b^2 + 4*c^2 = 3
def cond_b_eq_2c : Prop := b = 2*c

-- Statement for part (1)
theorem part1 (h1 : cond_positive_numbers a b c) (h2 : cond_main_equation a b c) :
  a + b + 2*c ≤ 3 := sorry

-- Statement for part (2)
theorem part2 (h1 : cond_positive_numbers a b c) (h2 : cond_main_equation a b c) (h3 : cond_b_eq_2c b c) :
  (1 / a) + (1 / c) ≥ 3 := sorry

end part1_part2_l68_68553


namespace number_of_sides_of_polygon_l68_68556

theorem number_of_sides_of_polygon (n : ℕ) (h : (n - 2) * 180 = 900) : n = 7 := 
sorry

end number_of_sides_of_polygon_l68_68556


namespace range_of_m_l68_68936

theorem range_of_m (m : ℝ) :
  (∀ x: ℝ, |x| + |x - 1| > m) ∨ (∀ x y, x < y → (5 - 2 * m)^x ≤ (5 - 2 * m)^y) 
  → ¬ ((∀ x: ℝ, |x| + |x - 1| > m) ∧ (∀ x y, x < y → (5 - 2 * m)^x ≤ (5 - 2 * m)^y)) 
  ↔ (1 ≤ m ∧ m < 2) :=
by
  sorry

end range_of_m_l68_68936


namespace least_five_digit_perfect_square_and_cube_l68_68317

/-- 
Prove that the least five-digit whole number that is both a perfect square 
and a perfect cube is 15625.
-/
theorem least_five_digit_perfect_square_and_cube : 
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k : ℕ, n = k^6) ∧ (∀ m : ℕ, (10000 ≤ m ∧ m < 100000) ∧ (∃ l : ℕ, m = l^6) → n ≤ m) :=
begin
  use 15625,
  split,
  { split,
    { norm_num },
    { norm_num } },
  { use 5,
    norm_num,
    intros m hm,
    rcases hm with ⟨⟨hm1, hm2⟩, ⟨l, rfl⟩⟩,
    cases le_or_lt l 5 with hl hl,
    { by_contradiction h,
      have : m < 10000 := by norm_num [nat.pow_succ, nat.pow_succ, hl],
      exact not_le_of_gt this hm1 },
    { have : l ≥ 6 := nat.lt_succ_iff.mp hl,
      have : m ≥ 6^6 := nat.pow_le_pow_of_le_right nat.zero_lt_succ_iff.1 this,
      norm_num at this,
      exact not_lt_of_ge this hm2 } }
end

end least_five_digit_perfect_square_and_cube_l68_68317


namespace real_solutions_to_system_l68_68366

theorem real_solutions_to_system :
  ∃ (s : Finset (ℝ × ℝ × ℝ × ℝ)), 
    (∀ (x y z w : ℝ), 
    (x = z + w + 2*z*w*x) ∧ 
    (y = w + x + 2*w*x*y) ∧ 
    (z = x + y + 2*x*y*z) ∧ 
    (w = y + z + 2*y*z*w) ↔ 
    (x, y, z, w) ∈ s) ∧
    (s.card = 15) :=
sorry

end real_solutions_to_system_l68_68366


namespace find_angle_sum_l68_68406

theorem find_angle_sum
  {α β : ℝ}
  (hα_acute : 0 < α ∧ α < π / 2)
  (hβ_acute : 0 < β ∧ β < π / 2)
  (h_tan_α : Real.tan α = 1 / 3)
  (h_cos_β : Real.cos β = 3 / 5) :
  α + 2 * β = π - Real.arctan (13 / 9) :=
sorry

end find_angle_sum_l68_68406


namespace frustum_small_cone_height_is_correct_l68_68522

noncomputable def frustum_small_cone_height (altitude : ℝ) 
                                             (lower_base_area : ℝ) 
                                             (upper_base_area : ℝ) : ℝ :=
  let r1 := Real.sqrt (lower_base_area / Real.pi)
  let r2 := Real.sqrt (upper_base_area / Real.pi)
  let H := 2 * altitude
  altitude

theorem frustum_small_cone_height_is_correct 
  (altitude : ℝ)
  (lower_base_area : ℝ)
  (upper_base_area : ℝ)
  (h1 : altitude = 16)
  (h2 : lower_base_area = 196 * Real.pi)
  (h3 : upper_base_area = 49 * Real.pi ) : 
  frustum_small_cone_height altitude lower_base_area upper_base_area = 16 := by
  sorry

end frustum_small_cone_height_is_correct_l68_68522


namespace seq_le_n_squared_l68_68041

theorem seq_le_n_squared (a : ℕ → ℕ) (h_increasing : ∀ n, a n < a (n + 1))
  (h_positive : ∀ n, 0 < a n)
  (h_property : ∀ t, ∃ i j, t = a i ∨ t = a i + a j) :
  ∀ n, a n ≤ n^2 :=
by {
  sorry
}

end seq_le_n_squared_l68_68041


namespace Norine_retire_age_l68_68048

theorem Norine_retire_age:
  ∀ (A W : ℕ),
    (A = 50) →
    (W = 19) →
    (A + W = 85) →
    (A = 50 + 8) :=
by
  intros A W hA hW hAW
  sorry

end Norine_retire_age_l68_68048


namespace least_five_digit_perfect_square_and_cube_l68_68261

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ ∃ k : ℕ, k^6 = n ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68261


namespace tax_free_amount_l68_68109

theorem tax_free_amount (X : ℝ) (total_value : ℝ) (tax_paid : ℝ) (tax_rate : ℝ) 
(h1 : total_value = 1720) 
(h2 : tax_paid = 134.4) 
(h3 : tax_rate = 0.12) 
(h4 : tax_paid = tax_rate * (total_value - X)) 
: X = 600 := 
sorry

end tax_free_amount_l68_68109


namespace unique_intersection_point_l68_68229

theorem unique_intersection_point (k : ℝ) :
x = k ->
∃ x : ℝ, x = -3*y^2 - 4*y + 7 -> ∃ k : ℝ, k = 25/3 -> y = 0 -> x = k

end unique_intersection_point_l68_68229


namespace consecutive_integer_quadratic_l68_68894

theorem consecutive_integer_quadratic :
  ∃ (a b c : ℤ) (x₁ x₂ : ℤ),
  (a * x₁ ^ 2 + b * x₁ + c = 0 ∧ a * x₂ ^ 2 + b * x₂ + c = 0) ∧
  (a = 2 ∧ b = 0 ∧ c = -2) ∨ (a = -2 ∧ b = 0 ∧ c = 2) := sorry

end consecutive_integer_quadratic_l68_68894


namespace simplest_form_eq_a_l68_68244

theorem simplest_form_eq_a (a : ℝ) (h : a ≠ 1) : 1 - (1 / (1 + (a / (1 - a)))) = a :=
by sorry

end simplest_form_eq_a_l68_68244


namespace quadratic_single_root_pos_value_l68_68175

theorem quadratic_single_root_pos_value (m : ℝ) (h1 : (6 * m)^2 - 4 * 1 * 2 * m = 0) : m = 2 / 9 :=
sorry

end quadratic_single_root_pos_value_l68_68175


namespace cost_price_eq_l68_68789

variables (x : ℝ)

def f (x : ℝ) : ℝ := x * (1 + 0.30)
def g (x : ℝ) : ℝ := f x * 0.80

theorem cost_price_eq (h : g x = 2080) : x * (1 + 0.30) * 0.80 = 2080 :=
by sorry

end cost_price_eq_l68_68789


namespace unique_set_property_l68_68870

theorem unique_set_property (a b c : ℕ) (h1: 1 < a) (h2: 1 < b) (h3: 1 < c) 
    (gcd_ab_c: (Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd a c = 1))
    (property_abc: (a * b) % c = (a * c) % b ∧ (a * c) % b = (b * c) % a) : 
    (a = 2 ∧ b = 3 ∧ c = 5) ∨ 
    (a = 2 ∧ b = 5 ∧ c = 3) ∨ 
    (a = 3 ∧ b = 2 ∧ c = 5) ∨ 
    (a = 3 ∧ b = 5 ∧ c = 2) ∨ 
    (a = 5 ∧ b = 2 ∧ c = 3) ∨ 
    (a = 5 ∧ b = 3 ∧ c = 2) := sorry

end unique_set_property_l68_68870


namespace total_area_calculations_l68_68123

noncomputable def total_area_in_hectares : ℝ :=
  let sections := 5
  let area_per_section := 60
  let conversion_factor_acre_to_hectare := 0.404686
  sections * area_per_section * conversion_factor_acre_to_hectare

noncomputable def total_area_in_square_meters : ℝ :=
  let conversion_factor_hectare_to_square_meter := 10000
  total_area_in_hectares * conversion_factor_hectare_to_square_meter

theorem total_area_calculations :
  total_area_in_hectares = 121.4058 ∧ total_area_in_square_meters = 1214058 := by
  sorry

end total_area_calculations_l68_68123


namespace probability_of_even_product_l68_68786

-- Each die has faces numbered from 1 to 8.
def faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Calculate the number of outcomes where the product of two rolls is even.
def num_even_product_outcomes : ℕ := (64 - 16)

-- Calculate the total number of outcomes when two eight-sided dice are rolled.
def total_outcomes : ℕ := 64

-- The probability that the product is even.
def probability_even_product : ℚ := num_even_product_outcomes / total_outcomes

theorem probability_of_even_product :
  probability_even_product = 3 / 4 :=
  by
    sorry

end probability_of_even_product_l68_68786


namespace simplify_expression_l68_68852

theorem simplify_expression (m n : ℝ) (h : m ≠ 0) : 
  (m^(4/3) - 27 * m^(1/3) * n) / 
  (m^(2/3) + 3 * (m * n)^(1/3) + 9 * n^(2/3)) / 
  (1 - 3 * (n / m)^(1/3)) - 
  (m^2)^(1/3) = 0 := 
sorry

end simplify_expression_l68_68852


namespace john_total_distance_l68_68707

-- Define the conditions
def john_speed_alone : ℝ := 4 -- miles per hour
def john_speed_with_dog : ℝ := 6 -- miles per hour
def time_with_dog : ℝ := 0.5 -- hours
def time_alone : ℝ := 0.5 -- hours

-- Calculate distances based on conditions and prove the total distance
theorem john_total_distance : 
  john_speed_with_dog * time_with_dog + john_speed_alone * time_alone = 5 := 
by 
  calc
    john_speed_with_dog * time_with_dog + john_speed_alone * time_alone
    = 6 * 0.5 + 4 * 0.5 : by sorry
    ... = 3 + 2 : by sorry
    ... = 5 : by sorry

end john_total_distance_l68_68707


namespace least_five_digit_is_15625_l68_68278

noncomputable def least_five_digit_perfect_square_and_cube : ℕ :=
  15625

theorem least_five_digit_is_15625 :
  (10000 ≤ least_five_digit_perfect_square_and_cube) ∧ (least_five_digit_perfect_square_and_cube < 100000) ∧
  (∃ a : ℕ, least_five_digit_perfect_square_and_cube = a ^ 6) :=
by
  sorry

end least_five_digit_is_15625_l68_68278


namespace M_intersect_N_equals_M_l68_68162

-- Define the sets M and N
def M := { x : ℝ | x^2 - 3 * x + 2 = 0 }
def N := { x : ℝ | x * (x - 1) * (x - 2) = 0 }

-- The theorem we want to prove
theorem M_intersect_N_equals_M : M ∩ N = M := 
by 
  sorry

end M_intersect_N_equals_M_l68_68162


namespace isosceles_triangle_l68_68992

theorem isosceles_triangle
  (a b c : ℝ)
  (α β γ : ℝ)
  (h1 : a + b = Real.tan (γ / 2) * (a * Real.tan α + b * Real.tan β)) :
  α = β ∨ α = γ ∨ β = γ :=
sorry

end isosceles_triangle_l68_68992


namespace solution_valid_l68_68999

noncomputable def verify_solution (x : ℝ) : Prop :=
  (Real.arcsin (3 * x) + Real.arccos (2 * x) = Real.pi / 4) ∧
  (|2 * x| ≤ 1) ∧
  (|3 * x| ≤ 1)

theorem solution_valid (x : ℝ) :
  verify_solution x ↔ (x = 1 / Real.sqrt (11 - 2 * Real.sqrt 2) ∨ x = -(1 / Real.sqrt (11 - 2 * Real.sqrt 2))) :=
by {
  sorry
}

end solution_valid_l68_68999


namespace least_five_digit_perfect_square_and_cube_l68_68296

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (n >= 10000 ∧ n < 100000) ∧ (∃ a : ℕ, n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68296


namespace yellow_tint_percentage_new_mixture_l68_68891

def original_volume : ℝ := 40
def yellow_tint_percentage : ℝ := 0.35
def additional_yellow_tint : ℝ := 10
def new_volume : ℝ := original_volume + additional_yellow_tint
def original_yellow_tint : ℝ := yellow_tint_percentage * original_volume
def new_yellow_tint : ℝ := original_yellow_tint + additional_yellow_tint

theorem yellow_tint_percentage_new_mixture : 
  (new_yellow_tint / new_volume) * 100 = 48 := 
by
  sorry

end yellow_tint_percentage_new_mixture_l68_68891


namespace tangent_line_eq_l68_68533

def f (x : ℝ) : ℝ := x^3 + 4 * x + 5

theorem tangent_line_eq (x y : ℝ) (h : (x, y) = (1, 10)) : 
  (7 * x - y + 3 = 0) :=
sorry

end tangent_line_eq_l68_68533


namespace problem1_problem2_l68_68510

theorem problem1 : 3 * Real.sqrt 3 - Real.sqrt 8 + Real.sqrt 2 - Real.sqrt 27 = -Real.sqrt 2 := 
by sorry

theorem problem2 : (Real.sqrt 5 - Real.sqrt 3) * (Real.sqrt 5 + Real.sqrt 3) = 2 := 
by sorry

end problem1_problem2_l68_68510


namespace tan_triple_angle_l68_68950

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end tan_triple_angle_l68_68950


namespace min_sum_of_dimensions_l68_68068

/-- A theorem to find the minimum possible sum of the three dimensions of a rectangular box 
with given volume 1729 inch³ and positive integer dimensions. -/
theorem min_sum_of_dimensions (x y z : ℕ) (h1 : x * y * z = 1729) : x + y + z ≥ 39 :=
by
  sorry

end min_sum_of_dimensions_l68_68068


namespace no_integers_p_and_q_l68_68375

theorem no_integers_p_and_q (p q : ℤ) : ¬(∀ x : ℤ, 3 ∣ (x^2 + p * x + q)) :=
by
  sorry

end no_integers_p_and_q_l68_68375


namespace determine_k_l68_68663

theorem determine_k (k : ℤ) : (∀ n : ℤ, gcd (4 * n + 1) (k * n + 1) = 1) ↔ 
  (∃ m : ℕ, k = 4 + 2 ^ m ∨ k = 4 - 2 ^ m) :=
by
  sorry

end determine_k_l68_68663


namespace geometric_sequence_ninth_tenth_term_sum_l68_68968

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q^n

theorem geometric_sequence_ninth_tenth_term_sum (a₁ q : ℝ)
  (h1 : a₁ + a₁ * q = 2)
  (h5 : a₁ * q^4 + a₁ * q^5 = 4) :
  geometric_sequence a₁ q 8 + geometric_sequence a₁ q 9 = 8 :=
by
  sorry

end geometric_sequence_ninth_tenth_term_sum_l68_68968


namespace total_pencils_correct_l68_68360

def reeta_pencils : Nat := 20
def anika_pencils : Nat := 2 * reeta_pencils + 4
def total_pencils : Nat := reeta_pencils + anika_pencils

theorem total_pencils_correct : total_pencils = 64 :=
by
  sorry

end total_pencils_correct_l68_68360


namespace books_at_end_of_year_l68_68045

def init_books : ℕ := 72
def monthly_books : ℕ := 12 -- 1 book each month for 12 months
def books_bought1 : ℕ := 5
def books_bought2 : ℕ := 2
def books_gift1 : ℕ := 1
def books_gift2 : ℕ := 4
def books_donated : ℕ := 12
def books_sold : ℕ := 3

theorem books_at_end_of_year :
  init_books + monthly_books + books_bought1 + books_bought2 + books_gift1 + books_gift2 - books_donated - books_sold = 81 :=
by
  sorry

end books_at_end_of_year_l68_68045


namespace range_of_b_l68_68672

open Real

theorem range_of_b (b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 4 → abs (y - (x + b)) = 1) ↔ -sqrt 2 < b ∧ b < sqrt 2 := 
by sorry

end range_of_b_l68_68672


namespace initial_time_is_11_55_l68_68644

-- Definitions for the conditions
variable (X : ℕ) (Y : ℕ)

def initial_time_shown_by_clock (X Y : ℕ) : Prop :=
  (5 * (18 - X) = 35) ∧ (Y = 60 - 5)

theorem initial_time_is_11_55 (h : initial_time_shown_by_clock X Y) : (X = 11) ∧ (Y = 55) :=
sorry

end initial_time_is_11_55_l68_68644


namespace find_acute_angle_x_l68_68718

def a_parallel_b (x : ℝ) : Prop :=
  let a := (Real.sin x, 3 / 4)
  let b := (1 / 3, 1 / 2 * Real.cos x)
  b.1 * a.2 = a.1 * b.2

theorem find_acute_angle_x (x : ℝ) (h : a_parallel_b x) : x = Real.pi / 4 :=
by
  sorry

end find_acute_angle_x_l68_68718


namespace chess_tournament_points_l68_68136

theorem chess_tournament_points
  (points : String → ℝ)
  (total_points : points "Andrey" + points "Dima" + points "Vanya" + points "Sasha" = 12)
  (distinct_points : 
    points "Andrey" ≠ points "Dima" ∧ 
    points "Andrey" ≠ points "Vanya" ∧ 
    points "Andrey" ≠ points "Sasha" ∧ 
    points "Dima" ≠ points "Vanya" ∧ 
    points "Dima" ≠ points "Sasha" ∧ 
    points "Vanya" ≠ points "Sasha")
  (order : 
    points "Andrey" > points "Dima" ∧ 
    points "Dima" > points "Vanya" ∧ 
    points "Vanya" > points "Sasha")
  (same_wins :
    let games_won (student : String) := (points student - 3) / 0.5 in
    games_won "Andrey" = games_won "Sasha") :
  points "Andrey" = 4 ∧ points "Dima" = 3.5 ∧ points "Vanya" = 2.5 ∧ points "Sasha" = 2 :=
by
  sorry

end chess_tournament_points_l68_68136


namespace expected_value_of_fair_8_sided_die_l68_68253

-- Define the outcomes of the fair 8-sided die
def outcomes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

-- Define the probability of each outcome for a fair die
def prob (n : ℕ) : ℚ := 1 / 8

-- Calculate the expected value of the outcomes
noncomputable def expected_value : ℚ :=
  (outcomes.map (λ x => prob x * x)).sum

-- State the theorem that the expected value is 4.5
theorem expected_value_of_fair_8_sided_die : expected_value = 4.5 :=
  sorry

end expected_value_of_fair_8_sided_die_l68_68253


namespace graph_of_equation_is_two_lines_l68_68120

theorem graph_of_equation_is_two_lines :
  ∀ x y : ℝ, x^2 - 16*y^2 - 8*x + 16 = 0 ↔ (x = 4 + 4*y ∨ x = 4 - 4*y) :=
by
  sorry

end graph_of_equation_is_two_lines_l68_68120


namespace find_expression_value_l68_68154

variable (a b : ℝ)

theorem find_expression_value (h : a - 2 * b = 7) : 6 - 2 * a + 4 * b = -8 := by
  sorry

end find_expression_value_l68_68154


namespace paige_folders_l68_68988

def initial_files : Nat := 135
def deleted_files : Nat := 27
def files_per_folder : Rat := 8.5
def folders_rounded_up (files_left : Nat) (per_folder : Rat) : Nat :=
  (Rat.ceil (Rat.ofInt files_left / per_folder)).toNat

theorem paige_folders :
  folders_rounded_up (initial_files - deleted_files) files_per_folder = 13 :=
by
  sorry

end paige_folders_l68_68988


namespace vitya_knows_answers_29_attempts_vitya_knows_answers_24_attempts_l68_68089

/-- The test consists of 30 questions, each with two possible answers (one correct and one incorrect). 
    Vitya can proceed in such a way that he can guarantee to know all the correct answers no later than:
    (a) after the 29th attempt (and answer all questions correctly on the 30th attempt)
    (b) after the 24th attempt (and answer all questions correctly on the 25th attempt)
    - Vitya initially does not know any of the answers.
    - The test is always the same.
-/
def vitya_test (k : Nat) : Prop :=
  k = 30 ∧ (∀ (attempts : Fin 30 → Bool), attempts 30 = attempts 29 ∧ attempts 30)

theorem vitya_knows_answers_29_attempts :
  vitya_test 30 :=
by 
  sorry

theorem vitya_knows_answers_24_attempts :
  vitya_test 25 :=
by 
  sorry

end vitya_knows_answers_29_attempts_vitya_knows_answers_24_attempts_l68_68089


namespace smallest_z_is_14_l68_68702

-- Define the consecutive even integers and the equation.
def w (k : ℕ) := 2 * k
def x (k : ℕ) := 2 * k + 2
def y (k : ℕ) := 2 * k + 4
def z (k : ℕ) := 2 * k + 6

theorem smallest_z_is_14 : ∃ k : ℕ, z k = 14 ∧ w k ^ 3 + x k ^ 3 + y k ^ 3 = z k ^ 3 :=
by sorry

end smallest_z_is_14_l68_68702


namespace savings_percentage_correct_l68_68770

def coat_price : ℝ := 120
def hat_price : ℝ := 30
def gloves_price : ℝ := 50

def coat_discount : ℝ := 0.20
def hat_discount : ℝ := 0.40
def gloves_discount : ℝ := 0.30

def original_total : ℝ := coat_price + hat_price + gloves_price
def coat_savings : ℝ := coat_price * coat_discount
def hat_savings : ℝ := hat_price * hat_discount
def gloves_savings : ℝ := gloves_price * gloves_discount
def total_savings : ℝ := coat_savings + hat_savings + gloves_savings

theorem savings_percentage_correct :
  (total_savings / original_total) * 100 = 25.5 := by
  sorry

end savings_percentage_correct_l68_68770


namespace tan_triple_angle_l68_68953

variable θ : ℝ
variable h : Real.tan θ = 3

theorem tan_triple_angle (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 :=
sorry

end tan_triple_angle_l68_68953


namespace calculate_expression_l68_68903

/-
We need to prove that the value of 18 * 36 + 54 * 18 + 18 * 9 is equal to 1782.
-/

theorem calculate_expression : (18 * 36 + 54 * 18 + 18 * 9 = 1782) :=
by
  have a1 : Int := 18 * 36
  have a2 : Int := 54 * 18
  have a3 : Int := 18 * 9
  sorry

end calculate_expression_l68_68903


namespace n_squared_divisible_by_144_l68_68172

theorem n_squared_divisible_by_144
  (n : ℕ)
  (h1 : 0 < n)
  (h2 : ∀ d : ℕ, d > 1 → d ∣ n → d ≤ 12) :
  144 ∣ n^2 :=
by
  sorry

end n_squared_divisible_by_144_l68_68172


namespace fraction_of_yard_occupied_by_flower_beds_l68_68774

theorem fraction_of_yard_occupied_by_flower_beds :
  let leg_length := (36 - 26) / 3
  let triangle_area := (1 / 2) * leg_length^2
  let total_flower_bed_area := 3 * triangle_area
  let yard_area := 36 * 6
  (total_flower_bed_area / yard_area) = 25 / 324
  := by
  let leg_length := (36 - 26) / 3
  let triangle_area := (1 / 2) * leg_length^2
  let total_flower_bed_area := 3 * triangle_area
  let yard_area := 36 * 6
  have h1 : leg_length = 10 / 3 := by sorry
  have h2 : triangle_area = (1 / 2) * (10 / 3)^2 := by sorry
  have h3 : total_flower_bed_area = 3 * ((1 / 2) * (10 / 3)^2) := by sorry
  have h4 : yard_area = 216 := by sorry
  have h5 : total_flower_bed_area / yard_area = 25 / 324 := by sorry
  exact h5

end fraction_of_yard_occupied_by_flower_beds_l68_68774


namespace least_five_digit_perfect_square_cube_l68_68273

theorem least_five_digit_perfect_square_cube :
  ∃ n : Nat, (10000 ≤ n ∧ n < 100000) ∧ (∃ m1 m2 : Nat, n = m1^2 ∧ n = m2^3 ∧ n = 15625) :=
by
  sorry

end least_five_digit_perfect_square_cube_l68_68273


namespace find_a_b_max_min_values_l68_68686

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  (1/3) * x^3 + a * x^2 + b * x

noncomputable def f' (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^2 + 2 * a * x + b

theorem find_a_b (a b : ℝ) :
  f' (-3) a b = 0 ∧ f (-3) a b = 9 → a = 1 ∧ b = -3 :=
  by sorry

theorem max_min_values (a b : ℝ) (h₁ : a = 1) (h₂ : b = -3):
  ∀ x ∈ Set.Icc (-3 : ℝ) 3, f x a b ≥ -5 / 3 ∧ f x a b ≤ 9 :=
  by sorry

end find_a_b_max_min_values_l68_68686


namespace pentadecagon_diagonals_l68_68418

def numberOfDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem pentadecagon_diagonals : numberOfDiagonals 15 = 90 :=
by
  sorry

end pentadecagon_diagonals_l68_68418


namespace limo_cost_is_correct_l68_68185

def prom_tickets_cost : ℕ := 2 * 100
def dinner_cost : ℕ := 120
def dinner_tip : ℕ := (30 * dinner_cost) / 100
def total_cost_before_limo : ℕ := prom_tickets_cost + dinner_cost + dinner_tip
def total_cost : ℕ := 836
def limo_hours : ℕ := 6
def limo_total_cost : ℕ := total_cost - total_cost_before_limo
def limo_cost_per_hour : ℕ := limo_total_cost / limo_hours

theorem limo_cost_is_correct : limo_cost_per_hour = 80 := 
by
  sorry

end limo_cost_is_correct_l68_68185


namespace valid_passwords_count_l68_68669

def total_passwords : Nat := 10 ^ 5
def restricted_passwords : Nat := 10

theorem valid_passwords_count : total_passwords - restricted_passwords = 99990 := by
  sorry

end valid_passwords_count_l68_68669


namespace fair_split_adjustment_l68_68714

theorem fair_split_adjustment
    (A B : ℝ)
    (h : A < B)
    (d1 d2 d3 : ℝ)
    (h1 : d1 = 120)
    (h2 : d2 = 150)
    (h3 : d3 = 180)
    (bernardo_pays_twice : ∀ D, (2 : ℝ) * D = d1 + d2 + d3) :
    (B - A) / 2 - 75 = ((d1 + d2 + d3) - 450) / 2 - (A - (d1 + d2 + d3) / 3) :=
by
  sorry

end fair_split_adjustment_l68_68714


namespace shortest_distance_to_y_axis_l68_68246

noncomputable def parabola : set (ℝ × ℝ) := {p | ∃ x, p = (x, sqrt (8 * x)) ∨ p = (x, -sqrt (8 * x))}

theorem shortest_distance_to_y_axis :
  ∀ (A B : ℝ × ℝ), 
  (A ∈ parabola) → (B ∈ parabola) → 
  (let d := dist A B in d = 10) →
  let P := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in
  abs P.1 = 3 :=
by
  intros A B hA hB hAB P
  sorry

end shortest_distance_to_y_axis_l68_68246


namespace bus_passengers_l68_68461

def passengers_after_first_stop := 7

def passengers_after_second_stop := passengers_after_first_stop - 3 + 5

def passengers_after_third_stop := passengers_after_second_stop - 2 + 4

theorem bus_passengers (passengers_after_first_stop passengers_after_second_stop passengers_after_third_stop : ℕ) : passengers_after_third_stop = 11 :=
by
  sorry

end bus_passengers_l68_68461


namespace coffee_mix_price_l68_68346

theorem coffee_mix_price 
  (P : ℝ)
  (pound_2nd : ℝ := 2.45)
  (total_pounds : ℝ := 18)
  (final_price_per_pound : ℝ := 2.30)
  (pounds_each_kind : ℝ := 9) :
  9 * P + 9 * pound_2nd = total_pounds * final_price_per_pound →
  P = 2.15 :=
by
  intros h
  sorry

end coffee_mix_price_l68_68346


namespace largest_real_number_l68_68391

theorem largest_real_number (x : ℝ) (h : (⌊x⌋ : ℝ) / x = 8 / 9) : x = 63 / 8 := sorry

end largest_real_number_l68_68391


namespace max_cables_to_ensure_communication_l68_68772

theorem max_cables_to_ensure_communication
    (A B : ℕ) (n : ℕ) 
    (hA : A = 16) (hB : B = 12) (hn : n = 28) :
    (A * B ≤ 192) ∧ (A * B = 192) :=
by
  sorry

end max_cables_to_ensure_communication_l68_68772


namespace brand_z_percentage_correct_l68_68654

noncomputable def percentage_of_brand_z (capacity : ℝ := 1) (brand_z1 : ℝ := 1) (brand_x1 : ℝ := 0) 
(brand_z2 : ℝ := 1/4) (brand_x2 : ℝ := 3/4) (brand_z3 : ℝ := 5/8) (brand_x3 : ℝ := 3/8) 
(brand_z4 : ℝ := 5/16) (brand_x4 : ℝ := 11/16) : ℝ :=
    (brand_z4 / (brand_z4 + brand_x4)) * 100

theorem brand_z_percentage_correct : percentage_of_brand_z = 31.25 := by
  sorry

end brand_z_percentage_correct_l68_68654


namespace Maria_waist_size_correct_l68_68668

noncomputable def waist_size_mm (waist_size_in : ℕ) (mm_per_ft : ℝ) (in_per_ft : ℕ) : ℝ :=
  (waist_size_in : ℝ) / (in_per_ft : ℝ) * mm_per_ft

theorem Maria_waist_size_correct :
  let waist_size_in := 27
  let mm_per_ft := 305
  let in_per_ft := 12
  waist_size_mm waist_size_in mm_per_ft in_per_ft = 686.3 :=
by
  sorry

end Maria_waist_size_correct_l68_68668


namespace ab_inequality_smaller_than_fourth_sum_l68_68524

theorem ab_inequality_smaller_than_fourth_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a * b) / (a + b + 2 * c) + (b * c) / (b + c + 2 * a) + (c * a) / (c + a + 2 * b) ≤ (1 / 4) * (a + b + c) := 
by
  sorry

end ab_inequality_smaller_than_fourth_sum_l68_68524


namespace ant_food_cost_l68_68725

-- Definitions for the conditions
def number_of_ants : ℕ := 400
def food_per_ant : ℕ := 2
def job_charge : ℕ := 5
def leaf_charge : ℕ := 1 / 100 -- 1 penny is 1 cent which is 0.01 dollars
def leaves_raked : ℕ := 6000
def jobs_completed : ℕ := 4

-- Compute the total money earned from jobs
def money_from_jobs : ℕ := jobs_completed * job_charge

-- Compute the total money earned from raking leaves
def money_from_leaves : ℕ := leaves_raked * leaf_charge

-- Compute the total money earned
def total_money_earned : ℕ := money_from_jobs + money_from_leaves

-- Compute the total ounces of food needed
def total_food_needed : ℕ := number_of_ants * food_per_ant

-- Calculate the cost per ounce of food
def cost_per_ounce : ℕ := total_money_earned / total_food_needed

theorem ant_food_cost :
  cost_per_ounce = 1 / 10 := sorry

end ant_food_cost_l68_68725


namespace max_min_sums_l68_68471

def P (x y : ℤ) := x^2 + y^2 = 50

theorem max_min_sums : 
  ∃ (x₁ y₁ x₂ y₂ : ℤ), P x₁ y₁ ∧ P x₂ y₂ ∧ 
    (x₁ + y₁ = 8) ∧ (x₂ + y₂ = -8) :=
by
  sorry

end max_min_sums_l68_68471


namespace frustum_slant_height_l68_68105

-- The setup: we are given specific conditions for a frustum resulting from cutting a cone
variable {r : ℝ} -- represents the radius of the upper base of the frustum
variable {h : ℝ} -- represents the slant height of the frustum
variable {h_removed : ℝ} -- represents the slant height of the removed cone

-- The given conditions
def upper_base_radius : ℝ := r
def lower_base_radius : ℝ := 4 * r
def slant_height_removed_cone : ℝ := 3

-- The proportion derived from similar triangles
def proportion (h r : ℝ) := (h / (4 * r)) = ((h + 3) / (5 * r))

-- The main statement: proving the slant height of the frustum is 9 cm
theorem frustum_slant_height (r : ℝ) (h : ℝ) (hr : proportion h r) : h = 9 :=
sorry

end frustum_slant_height_l68_68105


namespace root_equation_l68_68157

variables (m : ℝ)

theorem root_equation {m : ℝ} (h : m^2 - 2 * m - 3 = 0) : m^2 - 2 * m + 2023 = 2026 :=
by {
  sorry 
}

end root_equation_l68_68157


namespace least_five_digit_perfect_square_cube_l68_68275

theorem least_five_digit_perfect_square_cube :
  ∃ n : Nat, (10000 ≤ n ∧ n < 100000) ∧ (∃ m1 m2 : Nat, n = m1^2 ∧ n = m2^3 ∧ n = 15625) :=
by
  sorry

end least_five_digit_perfect_square_cube_l68_68275


namespace range_of_a_l68_68802

noncomputable def f (a x : ℝ) : ℝ := a * x ^ 2 - 1

def is_fixed_point (a x : ℝ) : Prop := f a x = x

def is_stable_point (a x : ℝ) : Prop := f a (f a x) = x

def are_equal_sets (a : ℝ) : Prop :=
  {x : ℝ | is_fixed_point a x} = {x : ℝ | is_stable_point a x}

theorem range_of_a (a : ℝ) (h : are_equal_sets a) : - (1 / 4) ≤ a ∧ a ≤ 3 / 4 := 
by
  sorry

end range_of_a_l68_68802


namespace no_solution_inequalities_l68_68416

theorem no_solution_inequalities (a : ℝ) :
  (¬ ∃ x : ℝ, x > 1 ∧ x < a - 1) → a ≤ 2 :=
by
  intro h
  sorry

end no_solution_inequalities_l68_68416


namespace incorrect_option_A_l68_68082

theorem incorrect_option_A (x y : ℝ) :
  ¬(5 * x + y / 2 = (5 * x + y) / 2) :=
by sorry

end incorrect_option_A_l68_68082


namespace total_rowing_campers_l68_68638

theorem total_rowing_campers (morning_rowing afternoon_rowing : ℕ) : 
  morning_rowing = 13 -> 
  afternoon_rowing = 21 -> 
  morning_rowing + afternoon_rowing = 34 :=
by
  sorry

end total_rowing_campers_l68_68638


namespace chess_tournament_points_l68_68135

theorem chess_tournament_points
  (points : String → ℝ)
  (total_points : points "Andrey" + points "Dima" + points "Vanya" + points "Sasha" = 12)
  (distinct_points : 
    points "Andrey" ≠ points "Dima" ∧ 
    points "Andrey" ≠ points "Vanya" ∧ 
    points "Andrey" ≠ points "Sasha" ∧ 
    points "Dima" ≠ points "Vanya" ∧ 
    points "Dima" ≠ points "Sasha" ∧ 
    points "Vanya" ≠ points "Sasha")
  (order : 
    points "Andrey" > points "Dima" ∧ 
    points "Dima" > points "Vanya" ∧ 
    points "Vanya" > points "Sasha")
  (same_wins :
    let games_won (student : String) := (points student - 3) / 0.5 in
    games_won "Andrey" = games_won "Sasha") :
  points "Andrey" = 4 ∧ points "Dima" = 3.5 ∧ points "Vanya" = 2.5 ∧ points "Sasha" = 2 :=
by
  sorry

end chess_tournament_points_l68_68135


namespace largest_error_in_circle_area_l68_68190

theorem largest_error_in_circle_area (d : ℝ) (error_percent : ℝ) (A : ℝ) : 
  d = 30 → error_percent = 0.3 → A = (Real.pi * (d / 2)^2) → 
  let d_min := d * (1 - error_percent),
      d_max := d * (1 + error_percent),
      A_min := Real.pi * (d_min / 2)^2,
      A_max := Real.pi * (d_max / 2)^2 in
  max ((A - A_min) / A * 100) ((A_max - A) / A * 100) = 69 :=
sorry

end largest_error_in_circle_area_l68_68190


namespace mutually_exclusive_A_B_head_l68_68054

variables (A_head B_head B_end : Prop)

def mut_exclusive (P Q : Prop) : Prop := ¬(P ∧ Q)

theorem mutually_exclusive_A_B_head (A_head B_head : Prop) :
  mut_exclusive A_head B_head :=
sorry

end mutually_exclusive_A_B_head_l68_68054


namespace total_bulbs_is_118_l68_68596

-- Define the number of medium lights
def medium_lights : Nat := 12

-- Define the number of large and small lights based on the given conditions
def large_lights : Nat := 2 * medium_lights
def small_lights : Nat := medium_lights + 10

-- Define the number of bulbs required for each type of light
def bulbs_needed_for_medium : Nat := 2 * medium_lights
def bulbs_needed_for_large : Nat := 3 * large_lights
def bulbs_needed_for_small : Nat := 1 * small_lights

-- Define the total number of bulbs needed
def total_bulbs_needed : Nat := bulbs_needed_for_medium + bulbs_needed_for_large + bulbs_needed_for_small

-- The theorem that represents the proof problem
theorem total_bulbs_is_118 : total_bulbs_needed = 118 := by 
  sorry

end total_bulbs_is_118_l68_68596


namespace pump_without_leak_time_l68_68651

variables (P : ℝ) (effective_rate_with_leak : ℝ) (leak_rate : ℝ)
variable (pump_filling_time : ℝ)

-- Define the conditions
def conditions :=
  effective_rate_with_leak = 3/7 ∧
  leak_rate = 1/14 ∧
  pump_filling_time = P

-- Define the theorem
theorem pump_without_leak_time (h : conditions P effective_rate_with_leak leak_rate pump_filling_time) : 
  P = 2 :=
sorry

end pump_without_leak_time_l68_68651


namespace reach_14_from_458_l68_68338

def double (n : ℕ) : ℕ :=
  n * 2

def erase_last_digit (n : ℕ) : ℕ :=
  n / 10

def can_reach (start target : ℕ) (ops : List (ℕ → ℕ)) : Prop :=
  ∃ seq : List (ℕ → ℕ), seq = ops ∧
    seq.foldl (fun acc f => f acc) start = target

-- The proof problem statement
theorem reach_14_from_458 : can_reach 458 14 [double, erase_last_digit, double, double, erase_last_digit, double, double, erase_last_digit] :=
  sorry

end reach_14_from_458_l68_68338


namespace pages_same_units_digit_l68_68767

theorem pages_same_units_digit (n : ℕ) (H : n = 63) : 
  ∃ (count : ℕ), count = 13 ∧ ∀ x : ℕ, (1 ≤ x ∧ x ≤ n) → 
  (((x % 10) = ((n + 1 - x) % 10)) → (x = 2 ∨ x = 7 ∨ x = 12 ∨ x = 17 ∨ x = 22 ∨ x = 27 ∨ x = 32 ∨ x = 37 ∨ x = 42 ∨ x = 47 ∨ x = 52 ∨ x = 57 ∨ x = 62)) :=
by
  sorry

end pages_same_units_digit_l68_68767


namespace rectangle_ratio_width_length_l68_68701

variable (w : ℝ)

theorem rectangle_ratio_width_length (h1 : w + 8 + w + 8 = 24) : 
  w / 8 = 1 / 2 :=
by
  sorry

end rectangle_ratio_width_length_l68_68701


namespace sum_of_cubes_eq_zero_l68_68965

theorem sum_of_cubes_eq_zero (a b : ℝ) (h1 : a + b = 0) (h2 : a * b = -4) : a^3 + b^3 = 0 :=
sorry

end sum_of_cubes_eq_zero_l68_68965


namespace no_positive_integer_satisfies_inequality_l68_68690

theorem no_positive_integer_satisfies_inequality :
  ∀ x : ℕ, 0 < x → ¬ (15 < -3 * (x : ℤ) + 18) := by
  sorry

end no_positive_integer_satisfies_inequality_l68_68690


namespace art_collection_total_area_l68_68370

-- Define the dimensions and quantities of the paintings
def square_painting_side := 6
def small_painting_width := 2
def small_painting_height := 3
def large_painting_width := 10
def large_painting_height := 15

def num_square_paintings := 3
def num_small_paintings := 4
def num_large_paintings := 1

-- Define areas of individual paintings
def square_painting_area := square_painting_side * square_painting_side
def small_painting_area := small_painting_width * small_painting_height
def large_painting_area := large_painting_width * large_painting_height

-- Define the total area calculation
def total_area :=
  num_square_paintings * square_painting_area +
  num_small_paintings * small_painting_area +
  num_large_paintings * large_painting_area

-- The theorem statement
theorem art_collection_total_area : total_area = 282 := by
  sorry

end art_collection_total_area_l68_68370


namespace frank_candy_bags_l68_68919

theorem frank_candy_bags (total_candies : ℕ) (candies_per_bag : ℕ) (bags : ℕ) 
  (h1 : total_candies = 22) (h2 : candies_per_bag = 11) : bags = 2 :=
by
  sorry

end frank_candy_bags_l68_68919


namespace find_K_values_l68_68797

-- Define summation of first K natural numbers
def sum_natural_numbers (K : ℕ) : ℕ :=
  K * (K + 1) / 2

-- Define the main problem conditions
theorem find_K_values (K N : ℕ) (hN_positive : N > 0) (hN_bound : N < 150) (h_sum_eq : sum_natural_numbers K = 3 * N^2) :
  K = 2 ∨ K = 12 ∨ K = 61 :=
  sorry

end find_K_values_l68_68797


namespace competition_participants_l68_68241

theorem competition_participants (n : ℕ) :
    (100 < n ∧ n < 200) ∧
    (n % 4 = 2) ∧
    (n % 5 = 2) ∧
    (n % 6 = 2)
    → (n = 122 ∨ n = 182) :=
by
  intro h
  sorry

end competition_participants_l68_68241


namespace percentage_increase_of_soda_l68_68402

variable (C S x : ℝ)

theorem percentage_increase_of_soda
  (h1 : 1.25 * C = 10)
  (h2 : S + x * S = 12)
  (h3 : C + S = 16) :
  x = 0.5 :=
sorry

end percentage_increase_of_soda_l68_68402


namespace ratio_is_five_to_three_l68_68472

variable (g b : ℕ)

def girls_more_than_boys : Prop := g - b = 6
def total_pupils : Prop := g + b = 24
def ratio_girls_to_boys : ℚ := g / b

theorem ratio_is_five_to_three (h1 : girls_more_than_boys g b) (h2 : total_pupils g b) : ratio_girls_to_boys g b = 5 / 3 := by
  sorry

end ratio_is_five_to_three_l68_68472


namespace subset_neg1_of_leq3_l68_68571

theorem subset_neg1_of_leq3 :
  {x | x = -1} ⊆ {x | x ≤ 3} :=
sorry

end subset_neg1_of_leq3_l68_68571


namespace largest_x_l68_68396

-- Conditions setup
def largest_x_satisfying_condition : ℝ :=
  let x : ℝ := 63 / 8 in x

theorem largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 8 / 9) : 
  x = largest_x_satisfying_condition :=
by
  sorry

end largest_x_l68_68396


namespace tan_3theta_eq_9_13_l68_68957

open Real

noncomputable def tan3theta (θ : ℝ) (h : tan θ = 3) : Prop :=
  tan (3 * θ) = (9 / 13)

theorem tan_3theta_eq_9_13 (θ : ℝ) (h : tan θ = 3) : tan3theta θ h :=
by
  sorry

end tan_3theta_eq_9_13_l68_68957


namespace part_a_l68_68493

theorem part_a 
  (x y u v : ℝ) 
  (h1 : x + y = u + v) 
  (h2 : x^2 + y^2 = u^2 + v^2) : 
  ∀ n : ℕ, x^n + y^n = u^n + v^n := 
by sorry

end part_a_l68_68493


namespace solve_inequality_l68_68538

theorem solve_inequality (x : ℝ) : 6 - x - 2 * x^2 < 0 ↔ x < -2 ∨ x > 3 / 2 := sorry

end solve_inequality_l68_68538


namespace range_of_a_l68_68545

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : 3 * a ≥ 1) (h3 : 4 * a ≤ 3 / 2) : 
  (1 / 3) ≤ a ∧ a ≤ (3 / 8) :=
by
  sorry

end range_of_a_l68_68545


namespace least_five_digit_whole_number_is_perfect_square_and_cube_l68_68263

theorem least_five_digit_whole_number_is_perfect_square_and_cube :
  ∃ (n : ℕ), (10000 ≤ n ∧ n < 100000) ∧ (∃ (a : ℕ), n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_whole_number_is_perfect_square_and_cube_l68_68263


namespace problem_mod_1000_l68_68196

noncomputable def M : ℕ := Nat.choose 18 9

theorem problem_mod_1000 : M % 1000 = 620 := by
  sorry

end problem_mod_1000_l68_68196


namespace sara_total_quarters_l68_68053

def initial_quarters : ℝ := 783.0
def given_quarters : ℝ := 271.0

theorem sara_total_quarters : initial_quarters + given_quarters = 1054.0 := 
by
  sorry

end sara_total_quarters_l68_68053


namespace positive_integers_N_segment_condition_l68_68818

theorem positive_integers_N_segment_condition (N : ℕ) (x : ℕ) (n : ℕ)
  (h1 : 10 ≤ N ∧ N ≤ 10^20)
  (h2 : N = x * (10^n - 1) / 9) (h3 : 1 ≤ n ∧ n ≤ 20) : 
  N + 1 = (x + 1) * (9 + 1)^n ∧ x < 10 :=
by {
  sorry
}

end positive_integers_N_segment_condition_l68_68818


namespace compute_sin_product_l68_68365

theorem compute_sin_product : 
  (1 - Real.sin (Real.pi / 12)) *
  (1 - Real.sin (5 * Real.pi / 12)) *
  (1 - Real.sin (7 * Real.pi / 12)) *
  (1 - Real.sin (11 * Real.pi / 12)) = 
  (1 / 16) :=
by
  sorry

end compute_sin_product_l68_68365


namespace remainder_when_divided_by_seven_l68_68800

theorem remainder_when_divided_by_seven (n : ℕ) (h₁ : n^3 ≡ 3 [MOD 7]) (h₂ : n^4 ≡ 2 [MOD 7]) : 
  n ≡ 6 [MOD 7] :=
sorry

end remainder_when_divided_by_seven_l68_68800


namespace solution_to_equation_l68_68778

theorem solution_to_equation : 
    (∃ x : ℤ, (x = 2 ∨ x = -2 ∨ x = 1 ∨ x = -1) ∧ (2 * x - 3 = -1)) → x = 1 :=
by
  sorry

end solution_to_equation_l68_68778


namespace find_a_l68_68975

theorem find_a (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 119) : a = 59 := 
sorry

end find_a_l68_68975


namespace train_distance_l68_68343

def fuel_efficiency := 5 / 2 
def coal_remaining := 160
def expected_distance := 400

theorem train_distance : fuel_efficiency * coal_remaining = expected_distance := 
by
  sorry

end train_distance_l68_68343


namespace branches_sum_one_main_stem_l68_68379

theorem branches_sum_one_main_stem (x : ℕ) (h : 1 + x + x^2 = 31) : x = 5 :=
by {
  sorry
}

end branches_sum_one_main_stem_l68_68379


namespace polynomial_factorization_example_l68_68844

open Polynomial

theorem polynomial_factorization_example
  (a_5 a_4 a_3 a_2 a_1 a_0 : ℤ) (hf : ∀ i ∈ [a_5, a_4, a_3, a_2, a_1, a_0], |i| ≤ 4)
  (b_3 b_2 b_1 b_0 : ℤ) (hg : ∀ i ∈ [b_3, b_2, b_1, b_0], |i| ≤ 1)
  (c_2 c_1 c_0 : ℤ) (hh : ∀ i ∈ [c_2, c_1, c_0], |i| ≤ 1)
  (h : (C a_5 * X^5 + C a_4 * X^4 + C a_3 * X^3 + C a_2 * X^2 + C a_1 * X + C a_0).eval 10 =
       ((C b_3 * X^3 + C b_2 * X^2 + C b_1 * X + C b_0) * (C c_2 * X^2 + C c_1 * X + C c_0)).eval 10) :
  (C a_5 * X^5 + C a_4 * X^4 + C a_3 * X^3 + C a_2 * X^2 + C a_1 * X + C a_0) =
  (C b_3 * X^3 + C b_2 * X^2 + C b_1 * X + C b_0) * (C c_2 * X^2 + C c_1 * X + C c_0) :=
sorry

end polynomial_factorization_example_l68_68844


namespace diamonds_in_G_15_l68_68858

/-- Define the number of diamonds in G_n -/
def diamonds_in_G (n : ℕ) : ℕ :=
  if n < 3 then 1
  else 3 * n ^ 2 - 3 * n + 1

/-- Theorem to prove the number of diamonds in G_15 is 631 -/
theorem diamonds_in_G_15 : diamonds_in_G 15 = 631 :=
by
  -- The proof is omitted
  sorry

end diamonds_in_G_15_l68_68858


namespace domain_of_c_is_all_real_l68_68530

theorem domain_of_c_is_all_real (a : ℝ) :
  (∀ x : ℝ, -3 * x^2 - 3 * x + a ≠ 0) ↔ a < -3 / 4 :=
by
  sorry

end domain_of_c_is_all_real_l68_68530


namespace least_five_digit_perfect_square_and_cube_l68_68312

/-- 
Prove that the least five-digit whole number that is both a perfect square 
and a perfect cube is 15625.
-/
theorem least_five_digit_perfect_square_and_cube : 
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k : ℕ, n = k^6) ∧ (∀ m : ℕ, (10000 ≤ m ∧ m < 100000) ∧ (∃ l : ℕ, m = l^6) → n ≤ m) :=
begin
  use 15625,
  split,
  { split,
    { norm_num },
    { norm_num } },
  { use 5,
    norm_num,
    intros m hm,
    rcases hm with ⟨⟨hm1, hm2⟩, ⟨l, rfl⟩⟩,
    cases le_or_lt l 5 with hl hl,
    { by_contradiction h,
      have : m < 10000 := by norm_num [nat.pow_succ, nat.pow_succ, hl],
      exact not_le_of_gt this hm1 },
    { have : l ≥ 6 := nat.lt_succ_iff.mp hl,
      have : m ≥ 6^6 := nat.pow_le_pow_of_le_right nat.zero_lt_succ_iff.1 this,
      norm_num at this,
      exact not_lt_of_ge this hm2 } }
end

end least_five_digit_perfect_square_and_cube_l68_68312


namespace turtles_on_lonely_island_l68_68561

theorem turtles_on_lonely_island (T : ℕ) (h1 : 60 = 2 * T + 10) : T = 25 := 
by sorry

end turtles_on_lonely_island_l68_68561


namespace chess_tournament_scores_l68_68140

def points (name : String) := Real

def total_points : Real := 12

variables (A D V S : Real)
variable (total_games : ℕ := 12)

axiom different_scores : A ≠ D ∧ A ≠ V ∧ A ≠ S ∧ D ≠ V ∧ D ≠ S ∧ V ≠ S

axiom ranking : A > D ∧ D > V ∧ V > S

axiom equal_wins (A S : Real) : (A = 2 * win_points) ∧ (S = 2 * win_points)

axiom total_points_constraint : A + D + V + S = total_points

theorem chess_tournament_scores :
  A = 4 ∧ D = 3.5 ∧ V = 2.5 ∧ S = 2 :=
by 
  sorry

end chess_tournament_scores_l68_68140


namespace trajectory_of_M_l68_68100

open Real

-- Define the endpoints A and B
variable {A B M : Real × Real}

-- Given conditions
def segment_length (A B : Real × Real) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 25

def on_axes (A B : Real × Real) : Prop :=
  A.2 = 0 ∧ B.1 = 0

def point_m_relationship (A B M : Real × Real) : Prop :=
  let AM := (M.1 - A.1, M.2 - A.2)
  let MB := (M.1 - B.1, M.2 - B.2)
  AM.1 = (2 / 3) * MB.1 ∧ AM.2 = (2 / 3) * MB.2 ∧
  (M.1 - A.1)^2 + (M.2 - A.2)^2 = 4

theorem trajectory_of_M (A B M : Real × Real)
  (h1 : segment_length A B)
  (h2 : on_axes A B)
  (h3 : point_m_relationship A B M) :
  (M.1^2 / 9) + (M.2^2 / 4) = 1 :=
sorry

end trajectory_of_M_l68_68100


namespace total_wheels_in_neighborhood_l68_68188

def cars_in_Jordan_driveway := 2
def wheels_per_car := 4
def spare_wheel := 1
def bikes_with_2_wheels := 3
def wheels_per_bike := 2
def bike_missing_rear_wheel := 1
def bike_with_training_wheel := 2 + 1
def trash_can_wheels := 2
def tricycle_wheels := 3
def wheelchair_main_wheels := 2
def wheelchair_small_wheels := 2
def wagon_wheels := 4
def roller_skates_total_wheels := 4
def roller_skates_missing_wheel := 1

def pickup_truck_wheels := 4
def boat_trailer_wheels := 2
def motorcycle_wheels := 2
def atv_wheels := 4

theorem total_wheels_in_neighborhood :
  (cars_in_Jordan_driveway * wheels_per_car + spare_wheel + bikes_with_2_wheels * wheels_per_bike + bike_missing_rear_wheel + bike_with_training_wheel + trash_can_wheels + tricycle_wheels + wheelchair_main_wheels + wheelchair_small_wheels + wagon_wheels + (roller_skates_total_wheels - roller_skates_missing_wheel)) +
  (pickup_truck_wheels + boat_trailer_wheels + motorcycle_wheels + atv_wheels) = 47 := by
  sorry

end total_wheels_in_neighborhood_l68_68188


namespace theresa_sons_count_l68_68613

theorem theresa_sons_count (total_meat_left : ℕ) (meat_per_plate : ℕ) (frac_left : ℚ) (s : ℕ) :
  total_meat_left = meat_per_plate ∧ meat_per_plate * frac_left * s = 3 → s = 9 :=
by sorry

end theresa_sons_count_l68_68613


namespace triangle_type_l68_68972

-- Definitions given in the problem
def is_not_equal (a : ℝ) (b : ℝ) : Prop := a ≠ b
def log_eq (b x : ℝ) : Prop := Real.log x = Real.log 4 / Real.log b + Real.log (4 * x - 4) / Real.log b

-- Main theorem stating the type of triangle ABC
theorem triangle_type (a b c A B C : ℝ) (h_b_ne_1 : is_not_equal b 1) (h_C_over_A_root : log_eq b (C / A)) (h_sin_B_over_sin_A_root : log_eq b (Real.sin B / Real.sin A)) : (B = 90) ∧ (A ≠ C) :=
by
  sorry

end triangle_type_l68_68972


namespace lamp_turn_off_count_l68_68076

theorem lamp_turn_off_count : 
  ∀ (n : ℕ), n = 10 →
  (∀ (k i : ℕ), 1 ≤ i ∧ i ≤ k ∧ k < 10 ∧ k - i ≥ 2 → ∀ (off : Finset ℕ),
  off ⊆ Finset.range 10 ∧ off.card = 3 ∧ ¬ 0 ∈ off ∧ ¬ (n - 1) ∈ off →
  (∀ (x y ∈ off), x ≠ y → |x - y| > 1) → 
  (off.card.choose 3 = 20)) :=
by {
  sorry
}

end lamp_turn_off_count_l68_68076


namespace combined_stripes_is_22_l68_68983

-- Definition of stripes per shoe for each person based on the conditions
def stripes_per_shoe_Olga : ℕ := 3
def stripes_per_shoe_Rick : ℕ := stripes_per_shoe_Olga - 1
def stripes_per_shoe_Hortense : ℕ := stripes_per_shoe_Olga * 2

-- The total combined number of stripes on all shoes for Olga, Rick, and Hortense
def total_stripes : ℕ := 2 * (stripes_per_shoe_Olga + stripes_per_shoe_Rick + stripes_per_shoe_Hortense)

-- The statement to prove that the total number of stripes on all their shoes is 22
theorem combined_stripes_is_22 : total_stripes = 22 :=
by
  sorry

end combined_stripes_is_22_l68_68983


namespace middle_number_of_consecutive_numbers_sum_of_squares_eq_2030_l68_68468

theorem middle_number_of_consecutive_numbers_sum_of_squares_eq_2030 :
  ∃ n : ℕ, n^2 + (n+1)^2 + (n+2)^2 = 2030 ∧ (n + 1) = 26 :=
by sorry

end middle_number_of_consecutive_numbers_sum_of_squares_eq_2030_l68_68468


namespace lisa_score_is_85_l68_68030

def score_formula (c w : ℕ) : ℕ := 30 + 4 * c - w

theorem lisa_score_is_85 (c w : ℕ) 
  (score_equality : 85 = score_formula c w)
  (non_neg_w : w ≥ 0)
  (total_questions : c + w ≤ 30) :
  (c = 14 ∧ w = 1) :=
by
  sorry

end lisa_score_is_85_l68_68030


namespace fraction_identity_l68_68942

variable (a b : ℚ) (h : a / b = 2 / 3)

theorem fraction_identity : a / (a - b) = -2 :=
by
  sorry

end fraction_identity_l68_68942


namespace volume_is_correct_l68_68898

def volume_of_box (x : ℝ) : ℝ :=
  (14 - 2 * x) * (10 - 2 * x) * x

theorem volume_is_correct (x : ℝ) :
  volume_of_box x = 140 * x - 48 * x^2 + 4 * x^3 :=
by
  sorry

end volume_is_correct_l68_68898


namespace least_five_digit_perfect_square_and_cube_l68_68306

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, n = 5^6 ∧ (n >= 10000 ∧ n < 100000) ∧ (∃ k : ℕ, n = k^2) ∧ (∃ m : ℕ, n = m^3) :=
by
  use 15625
  split
  · exact pow_succ 5 6
  split
  · have h1: 10000 ≤ 5^6 := nat.le_of_eq (calc 5^6 = 15625 : rfl)
    have h2: 5^6 < 100000 := nat.lt_of_eq_of_lt (calc 5^6 = 15625 : rfl) (by decide)
    exact ⟨h1, h2⟩
  · use 125
    exact pow_two 125
  · use 25
    exact pow_three 25
  sorry

end least_five_digit_perfect_square_and_cube_l68_68306


namespace sales_volume_increase_30_units_every_5_yuan_initial_sales_volume_750_units_daily_sales_volume_at_540_yuan_l68_68074

def price_reduction_table : List (ℕ × ℕ) := 
  [(5, 780), (10, 810), (15, 840), (20, 870), (25, 900), (30, 930), (35, 960)]

theorem sales_volume_increase_30_units_every_5_yuan :
  ∀ reduction volume1 volume2, (reduction + 5, volume1) ∈ price_reduction_table →
  (reduction + 10, volume2) ∈ price_reduction_table → volume2 - volume1 = 30 := sorry

theorem initial_sales_volume_750_units :
  (5, 780) ∈ price_reduction_table → (10, 810) ∈ price_reduction_table →
  (0, 750) ∉ price_reduction_table → 780 - 30 = 750 := sorry

theorem daily_sales_volume_at_540_yuan :
  ∀ P₀ P₁ volume, P₀ = 600 → P₁ = 540 → 
  (5, 780) ∈ price_reduction_table → (10, 810) ∈ price_reduction_table →
  (15, 840) ∈ price_reduction_table → (20, 870) ∈ price_reduction_table →
  (25, 900) ∈ price_reduction_table → (30, 930) ∈ price_reduction_table →
  (35, 960) ∈ price_reduction_table →
  volume = 750 + (P₀ - P₁) / 5 * 30 → volume = 1110 := sorry

end sales_volume_increase_30_units_every_5_yuan_initial_sales_volume_750_units_daily_sales_volume_at_540_yuan_l68_68074


namespace complex_square_simplification_l68_68994

theorem complex_square_simplification (i : ℂ) (h : i^2 = -1) : (4 - 3 * i)^2 = 7 - 24 * i :=
by {
  sorry
}

end complex_square_simplification_l68_68994


namespace pqrs_product_l68_68158

noncomputable def P : ℝ := Real.sqrt 2012 + Real.sqrt 2013
noncomputable def Q : ℝ := -Real.sqrt 2012 - Real.sqrt 2013
noncomputable def R : ℝ := Real.sqrt 2012 - Real.sqrt 2013
noncomputable def S : ℝ := Real.sqrt 2013 - Real.sqrt 2012

theorem pqrs_product : P * Q * R * S = 1 := 
by 
  sorry

end pqrs_product_l68_68158


namespace quadratic_rewriting_l68_68987

theorem quadratic_rewriting (d e : ℤ) (f : ℤ) : 
  (16 * x^2 - 40 * x - 24) = (d * x + e)^2 + f → 
  d^2 = 16 → 
  2 * d * e = -40 → 
  d * e = -20 := 
by
  intros h1 h2 h3
  sorry

end quadratic_rewriting_l68_68987


namespace crease_length_l68_68776

noncomputable def length_of_crease (theta : ℝ) : ℝ :=
  8 * Real.sin theta

theorem crease_length (theta : ℝ) (hθ : 0 ≤ theta ∧ theta ≤ π / 2) : 
  length_of_crease theta = 8 * Real.sin theta :=
by sorry

end crease_length_l68_68776


namespace combined_stripes_eq_22_l68_68986

def stripes_olga_per_shoe : ℕ := 3
def shoes_per_person : ℕ := 2
def stripes_olga_total : ℕ := stripes_olga_per_shoe * shoes_per_person

def stripes_rick_per_shoe : ℕ := stripes_olga_per_shoe - 1
def stripes_rick_total : ℕ := stripes_rick_per_shoe * shoes_per_person

def stripes_hortense_per_shoe : ℕ := stripes_olga_per_shoe * 2
def stripes_hortense_total : ℕ := stripes_hortense_per_shoe * shoes_per_person

def total_stripes : ℕ := stripes_olga_total + stripes_rick_total + stripes_hortense_total

theorem combined_stripes_eq_22 : total_stripes = 22 := by
  sorry

end combined_stripes_eq_22_l68_68986


namespace rhombus_perimeter_l68_68060

theorem rhombus_perimeter (d1 d2 : ℝ) (h_d1 : d1 = 24) (h_d2 : d2 = 16) : 
  let side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  in 4 * side = 16 * Real.sqrt 13 :=
by
  sorry

end rhombus_perimeter_l68_68060


namespace tan_three_theta_l68_68945

theorem tan_three_theta (θ : Real) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end tan_three_theta_l68_68945


namespace sequence_divisibility_count_l68_68940

theorem sequence_divisibility_count :
  ∀ (f : ℕ → ℕ), (∀ n, n ≥ 2 → f n = 10^n - 1) → 
  (∃ count, count = 504 ∧ ∀ i, 2 ≤ i ∧ i ≤ 2023 → (101 ∣ f i ↔ i % 4 = 0)) :=
by { sorry }

end sequence_divisibility_count_l68_68940


namespace bird_families_flew_away_for_winter_l68_68877

def bird_families_africa : ℕ := 38
def bird_families_asia : ℕ := 80
def total_bird_families_flew_away : ℕ := bird_families_africa + bird_families_asia

theorem bird_families_flew_away_for_winter : total_bird_families_flew_away = 118 := by
  -- proof goes here (not required)
  sorry

end bird_families_flew_away_for_winter_l68_68877


namespace cos_alpha_value_l68_68001

theorem cos_alpha_value
  (a : ℝ) (h1 : π < a ∧ a < 3 * π / 2)
  (h2 : Real.tan a = 2) :
  Real.cos a = - (Real.sqrt 5) / 5 :=
sorry

end cos_alpha_value_l68_68001


namespace chess_tournament_distribution_l68_68143

theorem chess_tournament_distribution 
    (students : List String)
    (games_played : Nat)
    (scores : List ℝ)
    (points_per_game : List ℝ)
    (unique_scores : ∀ (x y : ℝ), x ≠ y → scores.contains x → scores.contains y → x ≠ y)
    (first_place : String)
    (second_place : String)
    (third_place : String)
    (fourth_place : String)
    (andrey_wins_equal_sasha : ℝ)
    (total_points : ℝ)
    : 
    students = ["Andrey", "Vanya", "Dima", "Sasha"] ∧
    games_played = 6 ∧
    points_per_game = [1, 0.5, 0] ∧
    first_place = "Andrey" ∧
    second_place = "Dima" ∧
    third_place = "Vanya" ∧
    fourth_place = "Sasha" ∧
    scores = [4, 3.5, 2.5, 2] ∧
    andrey_wins_equal_sasha = 2 ∧
    total_points = 12 := 
sorry

end chess_tournament_distribution_l68_68143


namespace first_month_sale_l68_68648

theorem first_month_sale (sales_2 : ℕ) (sales_3 : ℕ) (sales_4 : ℕ) (sales_5 : ℕ) (sales_6 : ℕ) (average_sale : ℕ) (total_months : ℕ)
  (H_sales_2 : sales_2 = 6927)
  (H_sales_3 : sales_3 = 6855)
  (H_sales_4 : sales_4 = 7230)
  (H_sales_5 : sales_5 = 6562)
  (H_sales_6 : sales_6 = 5591)
  (H_average_sale : average_sale = 6600)
  (H_total_months : total_months = 6) :
  ∃ (sale_1 : ℕ), sale_1 = 6435 :=
by
  -- placeholder for the proof
  sorry

end first_month_sale_l68_68648


namespace calculate_sum_and_double_l68_68873

theorem calculate_sum_and_double :
  2 * (1324 + 4231 + 3124 + 2413) = 22184 :=
by
  sorry

end calculate_sum_and_double_l68_68873


namespace friends_receive_pens_l68_68978

-- Define the given conditions
def packs_kendra : ℕ := 4
def packs_tony : ℕ := 2
def pens_per_pack : ℕ := 3
def pens_kept_per_person : ℕ := 2

-- Define the proof problem
theorem friends_receive_pens :
  (packs_kendra * pens_per_pack + packs_tony * pens_per_pack - (pens_kept_per_person * 2)) = 14 :=
by sorry

end friends_receive_pens_l68_68978


namespace original_rectangle_area_l68_68066

theorem original_rectangle_area
  (A : ℝ)
  (h1 : ∀ (a : ℝ), a = 2 * A)
  (h2 : 4 * A = 32) : 
  A = 8 := 
by
  sorry

end original_rectangle_area_l68_68066


namespace num_of_B_sets_l68_68558

def A : Set ℕ := {1, 2}

theorem num_of_B_sets (S : Set ℕ) (A : Set ℕ) (h : A = {1, 2}) (h1 : ∀ B : Set ℕ, A ∪ B = S) : 
  ∃ n : ℕ, n = 4 ∧ (∀ B : Set ℕ, B ⊆ {1, 2} → S = {1, 2}) :=
by {
  sorry
}

end num_of_B_sets_l68_68558


namespace domain_of_f_l68_68908

noncomputable def f (x : ℝ) : ℝ := (Real.log (2 * x - 1)) / Real.sqrt (x + 1)

theorem domain_of_f :
  {x : ℝ | 2 * x - 1 > 0 ∧ x + 1 ≥ 0} = {x : ℝ | x > 1/2} :=
by
  sorry

end domain_of_f_l68_68908


namespace train_cross_pole_time_l68_68760

variable (L : Real) (V : Real)

theorem train_cross_pole_time (hL : L = 110) (hV : V = 144) : 
  (110 / (144 * 1000 / 3600) = 2.75) := 
by
  sorry

end train_cross_pole_time_l68_68760


namespace sum_f_eq_28743_l68_68127

def f (n : ℕ) : ℕ := 4 * n ^ 3 - 6 * n ^ 2 + 4 * n + 13

theorem sum_f_eq_28743 : (Finset.range 13).sum (λ n => f (n + 1)) = 28743 :=
by
  -- Placeholder for actual proof
  sorry

end sum_f_eq_28743_l68_68127


namespace intersecting_line_at_one_point_l68_68227

theorem intersecting_line_at_one_point (k : ℝ) :
  (∃ y : ℝ, k = -3 * y^2 - 4 * y + 7 ∧ 
           ∀ z : ℝ, k = -3 * z^2 - 4 * z + 7 → y = z) ↔ 
  k = 25 / 3 :=
by
  sorry

end intersecting_line_at_one_point_l68_68227


namespace directrix_of_parabola_l68_68466

theorem directrix_of_parabola :
  ∀ (x y : ℝ), y = - (1 / 8) * x^2 → y = 2 :=
by
  sorry

end directrix_of_parabola_l68_68466


namespace minor_axis_length_l68_68915

theorem minor_axis_length (h : ∀ x y : ℝ, x^2 / 4 + y^2 / 36 = 1) : 
  ∃ b : ℝ, b = 2 ∧ 2 * b = 4 :=
by
  sorry

end minor_axis_length_l68_68915


namespace largest_c_such_that_neg5_in_range_l68_68534

theorem largest_c_such_that_neg5_in_range :
  ∃ (c : ℝ), (∀ x : ℝ, x^2 + 5 * x + c = -5) → c = 5 / 4 :=
sorry

end largest_c_such_that_neg5_in_range_l68_68534


namespace final_surface_area_l68_68886

theorem final_surface_area 
  (original_cube_volume : ℕ)
  (small_cube_volume : ℕ)
  (remaining_cubes : ℕ)
  (removed_cubes : ℕ)
  (per_face_expose_area : ℕ)
  (initial_surface_area_per_cube : ℕ)
  (total_cubes : ℕ)
  (shared_internal_faces_area : ℕ)
  (final_surface_area : ℕ) :
  original_cube_volume = 12 * 12 * 12 →
  small_cube_volume = 3 * 3 * 3 →
  total_cubes = 64 →
  removed_cubes = 14 →
  remaining_cubes = total_cubes - removed_cubes →
  initial_surface_area_per_cube = 6 * 3 * 3 →
  per_face_expose_area = 6 * 4 →
  final_surface_area = remaining_cubes * (initial_surface_area_per_cube + per_face_expose_area) - shared_internal_faces_area →
  (remaining_cubes * (initial_surface_area_per_cube + per_face_expose_area) - shared_internal_faces_area) = 2820 :=
sorry

end final_surface_area_l68_68886


namespace original_square_area_l68_68681

-- Definitions based on the given problem conditions
variable (s : ℝ) (A : ℝ)
def is_square (s : ℝ) : Prop := s > 0
def oblique_projection (s : ℝ) (A : ℝ) : Prop :=
  (A = s^2 ∨ A = 4^2) ∧ s = 4

-- The theorem statement based on the problem question and correct answer
theorem original_square_area :
  is_square s →
  oblique_projection s A →
  ∃ A, A = 16 ∨ A = 64 := 
sorry

end original_square_area_l68_68681


namespace tan_three_theta_l68_68946

theorem tan_three_theta (θ : Real) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end tan_three_theta_l68_68946


namespace largest_real_number_l68_68389

theorem largest_real_number (x : ℝ) (h : (⌊x⌋ : ℝ) / x = 8 / 9) : x = 63 / 8 := sorry

end largest_real_number_l68_68389


namespace math_olympiad_problem_l68_68509

theorem math_olympiad_problem (students : Fin 11 → Finset (Fin n)) (h_solved : ∀ i, (students i).card = 3)
  (h_distinct : ∀ i j, i ≠ j → ∃ p, p ∈ students i ∧ p ∉ students j) : 
  6 ≤ n := 
sorry

end math_olympiad_problem_l68_68509


namespace problem1_problem2a_problem2b_l68_68455

noncomputable def x : ℝ := Real.sqrt 6 - Real.sqrt 2
noncomputable def a : ℝ := Real.sqrt 3 + Real.sqrt 2
noncomputable def b : ℝ := Real.sqrt 3 - Real.sqrt 2

theorem problem1 : x * (Real.sqrt 6 - x) + (x + Real.sqrt 5) * (x - Real.sqrt 5) = 1 - 2 * Real.sqrt 3 := 
by
  sorry

theorem problem2a : a - b = 2 * Real.sqrt 2 := 
by 
  sorry

theorem problem2b : a^2 - 2 * a * b + b^2 = 8 := 
by 
  sorry

end problem1_problem2a_problem2b_l68_68455


namespace range_of_m_l68_68935

theorem range_of_m (m : ℝ) : (∀ x : ℝ, m * x ^ 2 - m * x + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
sorry

end range_of_m_l68_68935


namespace conic_sections_are_parabolas_l68_68223

theorem conic_sections_are_parabolas (x y : ℝ) :
  y^6 - 9*x^6 = 3*y^3 - 1 → ∃ k : ℝ, (y^3 - 1 = k * 3 * x^3 ∨ y^3 = -k * 3 * x^3 + 1) := by
  sorry

end conic_sections_are_parabolas_l68_68223


namespace chess_tournament_scores_l68_68141

theorem chess_tournament_scores :
    ∃ (A D V S : ℝ),
    A ≠ D ∧ A ≠ V ∧ A ≠ S ∧ D ≠ V ∧ D ≠ S ∧ V ≠ S ∧
    A = 4 ∧ D = 3.5 ∧ V = 2.5 ∧ S = 2 ∧
    A > D ∧ D > V ∧ V > S ∧
    (∃ (wins_A wins_S : ℕ), wins_A = wins_S) :=
begin
    sorry
end

end chess_tournament_scores_l68_68141


namespace find_first_term_geometric_sequence_l68_68918

theorem find_first_term_geometric_sequence 
  (a b c : ℚ) 
  (h₁ : b = a * 4) 
  (h₂ : 36 = a * 4^2) 
  (h₃ : c = a * 4^3) 
  (h₄ : 144 = a * 4^4) : 
  a = 9 / 4 :=
sorry

end find_first_term_geometric_sequence_l68_68918


namespace find_algebraic_expression_value_l68_68805

theorem find_algebraic_expression_value (x : ℝ) (h : 3 * x^2 + 5 * x + 1 = 0) : 
  (x + 2) ^ 2 + x * (2 * x + 1) = 3 := 
by 
  -- Proof steps go here
  sorry

end find_algebraic_expression_value_l68_68805


namespace unique_intersection_point_l68_68230

theorem unique_intersection_point (k : ℝ) :
x = k ->
∃ x : ℝ, x = -3*y^2 - 4*y + 7 -> ∃ k : ℝ, k = 25/3 -> y = 0 -> x = k

end unique_intersection_point_l68_68230


namespace evaluate_cyclotomic_sum_l68_68910

theorem evaluate_cyclotomic_sum : 
  (Complex.I ^ 1520 + Complex.I ^ 1521 + Complex.I ^ 1522 + Complex.I ^ 1523 + Complex.I ^ 1524 = 2) :=
by sorry

end evaluate_cyclotomic_sum_l68_68910


namespace value_of_percent_l68_68959

theorem value_of_percent (x : ℝ) (h : 0.50 * x = 200) : 0.40 * x = 160 :=
sorry

end value_of_percent_l68_68959


namespace least_five_digit_whole_number_is_perfect_square_and_cube_l68_68265

theorem least_five_digit_whole_number_is_perfect_square_and_cube :
  ∃ (n : ℕ), (10000 ≤ n ∧ n < 100000) ∧ (∃ (a : ℕ), n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_whole_number_is_perfect_square_and_cube_l68_68265


namespace simplify_expression_l68_68997

variable {a b : ℚ}

theorem simplify_expression (h1 : a + b ≠ 0) (h2 : a - 2b ≠ 0) (h3 : a^2 - 4a * b + 4b^2 ≠ 0) :
    (a + 2b) / (a + b) - (a - b) / (a - 2b) / ((a^2 - b^2) / (a^2 - 4a * b + 4b^2)) = 4 * b / (a + b) :=
by
  sorry

end simplify_expression_l68_68997


namespace fraction_of_girls_correct_l68_68112

-- Define the total number of students in each school
def total_greenwood : ℕ := 300
def total_maplewood : ℕ := 240

-- Define the ratios of boys to girls
def ratio_boys_girls_greenwood := (3, 2)
def ratio_boys_girls_maplewood := (3, 4)

-- Define the number of boys and girls at Greenwood Middle School
def boys_greenwood (x : ℕ) : ℕ := 3 * x
def girls_greenwood (x : ℕ) : ℕ := 2 * x

-- Define the number of boys and girls at Maplewood Middle School
def boys_maplewood (y : ℕ) : ℕ := 3 * y
def girls_maplewood (y : ℕ) : ℕ := 4 * y

-- Define the total fractions
def total_girls (x y : ℕ) : ℚ := (girls_greenwood x + girls_maplewood y)
def total_students : ℚ := (total_greenwood + total_maplewood)

-- Main theorem to prove the fraction of girls at the event
theorem fraction_of_girls_correct (x y : ℕ)
  (h1 : 5 * x = total_greenwood)
  (h2 : 7 * y = total_maplewood) :
  (total_girls x y) / total_students = 5 / 7 :=
by
  sorry

end fraction_of_girls_correct_l68_68112


namespace domain_of_p_l68_68532

def is_domain_of_p (x : ℝ) : Prop := x > 5

theorem domain_of_p :
  {x : ℝ | ∃ y : ℝ, y = 5*x + 2 ∧ ∃ z : ℝ, z = 2*x - 10 ∧
    z ≥ 0 ∧ z ≠ 0 ∧ p = 5*x + 2} = {x : ℝ | x > 5} :=
by
  sorry

end domain_of_p_l68_68532


namespace simplify_expression_l68_68996

variable {a b : ℝ}

theorem simplify_expression (h1 : a ≠ -b) (h2 : a ≠ 2b) (h3 : a ≠ b) :
  (a + 2 * b) / (a + b) - (a - b) / (a - 2 * b) / ((a^2 - b^2) / (a^2 - 4 * a * b + 4 * b^2)) = 4 * b / (a + b) :=
by
  sorry

end simplify_expression_l68_68996


namespace min_value_expression_l68_68843

theorem min_value_expression (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_cond : x * y * z = 1/2) :
  x^3 + 4 * x * y + 16 * y^3 + 8 * y * z + 3 * z^3 ≥ 18 :=
sorry

end min_value_expression_l68_68843


namespace least_five_digit_perfect_square_and_cube_l68_68254

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ ∃ k : ℕ, k^6 = n ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68254


namespace car_speed_l68_68757

/-- Given a car covers a distance of 624 km in 2 3/5 hours,
    prove that the speed of the car is 240 km/h. -/
theorem car_speed (distance : ℝ) (time : ℝ)
  (h_distance : distance = 624)
  (h_time : time = 13 / 5) :
  (distance / time) = 240 :=
by
  sorry

end car_speed_l68_68757


namespace sum_of_roots_of_quadratic_l68_68420

noncomputable def x1_x2_roots_properties : Prop :=
  ∃ x₁ x₂ : ℝ, (x₁ + x₂ = 3) ∧ (x₁ * x₂ = -4)

theorem sum_of_roots_of_quadratic :
  ∃ x₁ x₂ : ℝ, (x₁ * x₂ = -4) → (x₁ + x₂ = 3) :=
by
  sorry

end sum_of_roots_of_quadratic_l68_68420


namespace least_five_digit_perfect_square_cube_l68_68277

theorem least_five_digit_perfect_square_cube :
  ∃ n : Nat, (10000 ≤ n ∧ n < 100000) ∧ (∃ m1 m2 : Nat, n = m1^2 ∧ n = m2^3 ∧ n = 15625) :=
by
  sorry

end least_five_digit_perfect_square_cube_l68_68277


namespace intersection_complement_l68_68010

open Set

-- Define sets A and B as provided in the conditions
def A : Set ℝ := {x | x ≤ 3}
def B : Set ℝ := {x | x < 2}

-- Define the theorem to prove the question is equal to the answer given the conditions
theorem intersection_complement : (A ∩ compl B) = {x | 2 ≤ x ∧ x ≤ 3} := by
  sorry

end intersection_complement_l68_68010


namespace cyclist_speed_ratio_l68_68749

variables (k r t v1 v2 : ℝ)
variable (h1 : v1 = 2 * v2) -- Condition 5

-- When traveling in the same direction, relative speed is v1 - v2 and they cover 2k miles in 3r hours
variable (h2 : 2 * k = (v1 - v2) * 3 * r)

-- When traveling in opposite directions, relative speed is v1 + v2 and they pass each other in 2t hours
variable (h3 : 2 * k = (v1 + v2) * 2 * t)

theorem cyclist_speed_ratio (h1 : v1 = 2 * v2) (h2 : 2 * k = (v1 - v2) * 3 * r) (h3 : 2 * k = (v1 + v2) * 2 * t) :
  v1 / v2 = 2 :=
sorry

end cyclist_speed_ratio_l68_68749


namespace drive_time_from_city_B_to_city_A_l68_68755

theorem drive_time_from_city_B_to_city_A
  (t : ℝ)
  (round_trip_distance : ℝ := 360)
  (saved_time_per_trip : ℝ := 0.5)
  (average_speed : ℝ := 80) :
  (80 * ((3 + t) - 2 * 0.5)) = 360 → t = 2.5 :=
by
  intro h
  sorry

end drive_time_from_city_B_to_city_A_l68_68755


namespace prime_half_sum_l68_68037

theorem prime_half_sum
  (a b c : ℕ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h1 : Nat.Prime (a.factorial + b + c))
  (h2 : Nat.Prime (b.factorial + c + a))
  (h3 : Nat.Prime (c.factorial + a + b)) :
  Nat.Prime ((a + b + c + 1) / 2) := 
sorry

end prime_half_sum_l68_68037


namespace neg_P_4_of_P_implication_and_neg_P_5_l68_68148

variable (P : ℕ → Prop)

theorem neg_P_4_of_P_implication_and_neg_P_5
  (h1 : ∀ k : ℕ, 0 < k → (P k → P (k+1)))
  (h2 : ¬ P 5) :
  ¬ P 4 :=
by
  sorry

end neg_P_4_of_P_implication_and_neg_P_5_l68_68148


namespace find_value_l68_68814

theorem find_value 
  (a b c d e f : ℚ)
  (h1 : a / b = 1 / 2)
  (h2 : c / d = 1 / 2)
  (h3 : e / f = 1 / 2)
  (h4 : 3 * b - 2 * d + f ≠ 0) : 
  (3 * a - 2 * c + e) / (3 * b - 2 * d + f) = 1 / 2 := 
by
  sorry

end find_value_l68_68814


namespace almonds_weight_l68_68498

def nuts_mixture (almonds_ratio walnuts_ratio total_weight : ℚ) : ℚ :=
  let total_parts := almonds_ratio + walnuts_ratio
  let weight_per_part := total_weight / total_parts
  let weight_almonds := weight_per_part * almonds_ratio
  weight_almonds

theorem almonds_weight (total_weight : ℚ) (h1 : total_weight = 140) : nuts_mixture 5 1 total_weight = 116.67 :=
by
  sorry

end almonds_weight_l68_68498


namespace least_five_digit_perfect_square_and_cube_l68_68313

/-- 
Prove that the least five-digit whole number that is both a perfect square 
and a perfect cube is 15625.
-/
theorem least_five_digit_perfect_square_and_cube : 
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k : ℕ, n = k^6) ∧ (∀ m : ℕ, (10000 ≤ m ∧ m < 100000) ∧ (∃ l : ℕ, m = l^6) → n ≤ m) :=
begin
  use 15625,
  split,
  { split,
    { norm_num },
    { norm_num } },
  { use 5,
    norm_num,
    intros m hm,
    rcases hm with ⟨⟨hm1, hm2⟩, ⟨l, rfl⟩⟩,
    cases le_or_lt l 5 with hl hl,
    { by_contradiction h,
      have : m < 10000 := by norm_num [nat.pow_succ, nat.pow_succ, hl],
      exact not_le_of_gt this hm1 },
    { have : l ≥ 6 := nat.lt_succ_iff.mp hl,
      have : m ≥ 6^6 := nat.pow_le_pow_of_le_right nat.zero_lt_succ_iff.1 this,
      norm_num at this,
      exact not_lt_of_ge this hm2 } }
end

end least_five_digit_perfect_square_and_cube_l68_68313


namespace simplify_nested_fourth_roots_l68_68790

variable (M : ℝ)
variable (hM : M > 1)

theorem simplify_nested_fourth_roots : 
  (M^(1/4) * (M^(1/4) * (M^(1/4) * M)^(1/4))^(1/4))^(1/4) = M^(21/64) := by
  sorry

end simplify_nested_fourth_roots_l68_68790


namespace mixing_ratios_indeterminacy_l68_68487

theorem mixing_ratios_indeterminacy (x : ℝ) (a b : ℝ) (h1 : a + b = 50) (h2 : 0.40 * a + (x / 100) * b = 25) : False :=
sorry

end mixing_ratios_indeterminacy_l68_68487


namespace smallest_positive_integer_l68_68624

theorem smallest_positive_integer (m n : ℤ) : ∃ m n : ℤ, 3003 * m + 60606 * n = 273 :=
sorry

end smallest_positive_integer_l68_68624


namespace length_of_square_side_l68_68640

noncomputable def speed_km_per_hr_to_m_per_s (speed : ℝ) : ℝ :=
  speed * 1000 / 3600

noncomputable def perimeter_of_square (side_length : ℝ) : ℝ :=
  4 * side_length

theorem length_of_square_side
  (time_seconds : ℝ)
  (speed_km_per_hr : ℝ)
  (distance_m : ℝ)
  (side_length : ℝ)
  (h1 : time_seconds = 72)
  (h2 : speed_km_per_hr = 10)
  (h3 : distance_m = speed_km_per_hr_to_m_per_s speed_km_per_hr * time_seconds)
  (h4 : distance_m = perimeter_of_square side_length) :
  side_length = 50 :=
sorry

end length_of_square_side_l68_68640


namespace option_D_correct_l68_68938

-- Defining the types for lines and planes
variables {Line Plane : Type}

-- Defining what's needed for perpendicularity and parallelism
variables (perp : Line → Plane → Prop)
variables (subset : Line → Plane → Prop)
variables (parallel : Line → Line → Prop)
variables (perp_planes : Plane → Plane → Prop)

-- Main theorem statement
theorem option_D_correct (a b : Line) (α β : Plane) :
  perp a α → subset b β → parallel a b → perp_planes α β :=
by
  sorry

end option_D_correct_l68_68938


namespace simplify_and_evaluate_expression_l68_68211

variable (x y : ℝ)
variable (h1 : x = 1)
variable (h2 : y = Real.sqrt 2)

theorem simplify_and_evaluate_expression : 
  (x + 2 * y) ^ 2 - x * (x + 4 * y) + (1 - y) * (1 + y) = 7 := by
  sorry

end simplify_and_evaluate_expression_l68_68211


namespace find_g7_l68_68840

noncomputable def g (x : ℝ) (a b c d : ℝ) : ℝ := a * x ^ 7 + b * x ^ 3 + d * x ^ 2 + c * x - 8

theorem find_g7 (a b c d : ℝ) (h : g (-7) a b c d = 3) (h_d : d = 0) : g 7 a b c d = -19 :=
by
  simp [g, h, h_d]
  sorry

end find_g7_l68_68840


namespace sin_arith_seq_l68_68547

theorem sin_arith_seq (a : ℕ → ℝ) (d : ℝ)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_sum : a 1 + a 5 + a 9 = 5 * Real.pi) :
  Real.sin (a 2 + a 8) = - (Real.sqrt 3) / 2 :=
sorry

end sin_arith_seq_l68_68547


namespace total_hiking_distance_l68_68049

def saturday_distance : ℝ := 8.2
def sunday_distance : ℝ := 1.6
def total_distance (saturday_distance sunday_distance : ℝ) : ℝ := saturday_distance + sunday_distance

theorem total_hiking_distance :
  total_distance saturday_distance sunday_distance = 9.8 :=
by
  -- The proof is omitted
  sorry

end total_hiking_distance_l68_68049


namespace complement_of_P_in_U_l68_68011

open Set

theorem complement_of_P_in_U :
  let U := ℝ
  let P := {x : ℝ | x^2 ≤ 1}
  ∁ U P = {x : ℝ | x < -1} ∪ {x : ℝ | x > 1} :=
by
  sorry

end complement_of_P_in_U_l68_68011


namespace expected_value_min_of_subset_l68_68480

noncomputable def expected_value_min (n r : ℕ) (h : 1 ≤ r ∧ r ≤ n) : ℚ :=
  (n + 1) / (r + 1)

theorem expected_value_min_of_subset (n r : ℕ) (h : 1 ≤ r ∧ r ≤ n) : 
  expected_value_min n r h = (n + 1) / (r + 1) :=
sorry

end expected_value_min_of_subset_l68_68480


namespace nth_equation_l68_68203

theorem nth_equation (n : ℕ) (hn : n ≠ 0) : 
  (↑n + 2) / ↑n - 2 / (↑n + 2) = ((↑n + 2)^2 + ↑n^2) / (↑n * (↑n + 2)) - 1 :=
by
  sorry

end nth_equation_l68_68203


namespace find_a_l68_68974

theorem find_a (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 119) : a = 59 :=
sorry

end find_a_l68_68974


namespace intersection_P_compl_M_l68_68443

-- Define universal set U
def U : Set ℤ := Set.univ

-- Define set M
def M : Set ℤ := {1, 2}

-- Define set P
def P : Set ℤ := {-2, -1, 0, 1, 2}

-- Define the complement of M in U
def M_compl : Set ℤ := { x | x ∉ M }

-- Define the intersection of P and the complement of M
def P_inter_M_compl : Set ℤ := P ∩ M_compl

-- The theorem we want to prove
theorem intersection_P_compl_M : P_inter_M_compl = {-2, -1, 0} := 
by {
  sorry
}

end intersection_P_compl_M_l68_68443


namespace seeds_germination_percentage_l68_68403

theorem seeds_germination_percentage :
  ∀ (total_seeds first_plot_seeds second_plot_seeds germinated_percentage_total germinated_percentage_second_plot germinated_seeds_total germinated_seeds_second_plot germinated_seeds_first_plot x : ℕ),
    total_seeds = 300 + 200 → 
    germinated_percentage_second_plot = 35 → 
    germinated_percentage_total = 32 → 
    second_plot_seeds = 200 → 
    germinated_seeds_second_plot = (germinated_percentage_second_plot * second_plot_seeds) / 100 → 
    germinated_seeds_total = (germinated_percentage_total * total_seeds) / 100 → 
    germinated_seeds_first_plot = germinated_seeds_total - germinated_seeds_second_plot → 
    x = 30 → 
    x = (germinated_seeds_first_plot * 100) / 300 → 
    x = 30 :=
  by 
    intros total_seeds first_plot_seeds second_plot_seeds germinated_percentage_total germinated_percentage_second_plot germinated_seeds_total germinated_seeds_second_plot germinated_seeds_first_plot x
    sorry

end seeds_germination_percentage_l68_68403


namespace compute_expression_l68_68119

-- Define the operation a Δ b
def Delta (a b : ℝ) : ℝ := a^2 - 2 * b

theorem compute_expression :
  let x := 3 ^ (Delta 4 10)
  let y := 4 ^ (Delta 2 3)
  Delta x y = ( -819.125 / 6561) :=
by 
  sorry

end compute_expression_l68_68119


namespace probability_of_stock_price_increase_l68_68511

namespace StockPriceProbability

variables (P_A P_B P_C P_D_given_A P_D_given_B P_D_given_C : ℝ)

def P_D : ℝ := P_A * P_D_given_A + P_B * P_D_given_B + P_C * P_D_given_C

theorem probability_of_stock_price_increase :
    P_A = 0.6 → P_B = 0.3 → P_C = 0.1 → 
    P_D_given_A = 0.7 → P_D_given_B = 0.2 → P_D_given_C = 0.1 → 
    P_D P_A P_B P_C P_D_given_A P_D_given_B P_D_given_C = 0.49 :=
by intros h₁ h₂ h₃ h₄ h₅ h₆; sorry

end StockPriceProbability

end probability_of_stock_price_increase_l68_68511


namespace original_area_of_doubled_rectangle_l68_68065

theorem original_area_of_doubled_rectangle (A_new : ℝ) (h : A_new = 32) :
  ∃ A : ℝ, A * 4 = A_new ∧ A = 8 :=
by {
  use 8,
  split,
  { norm_num, exact h.symm },
  { rfl }
}

end original_area_of_doubled_rectangle_l68_68065


namespace measure_8_liters_with_two_buckets_l68_68567

def bucket_is_empty (B : ℕ) : Prop :=
  B = 0

def bucket_has_capacity (B : ℕ) (c : ℕ) : Prop :=
  B ≤ c

def fill_bucket (B : ℕ) (c : ℕ) : ℕ :=
  c

def empty_bucket (B : ℕ) : ℕ :=
  0

def pour_bucket (B1 B2 : ℕ) (c1 c2 : ℕ) : (ℕ × ℕ) :=
  if B1 + B2 <= c2 then (0, B1 + B2)
  else (B1 - (c2 - B2), c2)

theorem measure_8_liters_with_two_buckets (B10 B6 : ℕ) (c10 c6 : ℕ) :
  bucket_has_capacity B10 c10 ∧ bucket_has_capacity B6 c6 ∧
  c10 = 10 ∧ c6 = 6 →
  ∃ B10' B6', B10' = 8 ∧ B6' ≤ 6 :=
by
  intros h
  have h1 : ∃ B1, bucket_is_empty B1,
    from ⟨0, rfl⟩
  let B10 := fill_bucket 0 c10
  let ⟨B10, B6⟩ := pour_bucket B10 0 c10 c6
  let B6 := empty_bucket B6
  let ⟨B10, B6⟩ := pour_bucket B10 B6 c10 c6
  let B10 := fill_bucket B10 c10
  let ⟨B10, B6⟩ := pour_bucket B10 B6 c10 c6
  exact ⟨B10, B6, rfl, le_refl 6⟩

end measure_8_liters_with_two_buckets_l68_68567


namespace probability_division_integer_l68_68224

-- Definitions of sets and conditions
def R : Finset ℤ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def K : Finset ℤ := {3, 4, 5, 6, 7, 8, 9}

-- Lean definition of the proof problem
theorem probability_division_integer :
  let valid_pairs := Finset.filter (λ (rk : ℤ × ℤ), rk.2 ∣ rk.1) (R.product K) in
  (valid_pairs.card : ℚ) / (R.card * K.card) = 1 / 7 := by
{
  -- Sorry placeholder for the proof
  sorry
}

end probability_division_integer_l68_68224


namespace least_five_digit_perfect_square_and_cube_l68_68315

/-- 
Prove that the least five-digit whole number that is both a perfect square 
and a perfect cube is 15625.
-/
theorem least_five_digit_perfect_square_and_cube : 
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k : ℕ, n = k^6) ∧ (∀ m : ℕ, (10000 ≤ m ∧ m < 100000) ∧ (∃ l : ℕ, m = l^6) → n ≤ m) :=
begin
  use 15625,
  split,
  { split,
    { norm_num },
    { norm_num } },
  { use 5,
    norm_num,
    intros m hm,
    rcases hm with ⟨⟨hm1, hm2⟩, ⟨l, rfl⟩⟩,
    cases le_or_lt l 5 with hl hl,
    { by_contradiction h,
      have : m < 10000 := by norm_num [nat.pow_succ, nat.pow_succ, hl],
      exact not_le_of_gt this hm1 },
    { have : l ≥ 6 := nat.lt_succ_iff.mp hl,
      have : m ≥ 6^6 := nat.pow_le_pow_of_le_right nat.zero_lt_succ_iff.1 this,
      norm_num at this,
      exact not_lt_of_ge this hm2 } }
end

end least_five_digit_perfect_square_and_cube_l68_68315


namespace ice_cream_bar_price_l68_68642

theorem ice_cream_bar_price 
  (num_bars num_sundaes : ℕ)
  (total_cost : ℝ)
  (sundae_price ice_cream_bar_price : ℝ)
  (h1 : num_bars = 125)
  (h2 : num_sundaes = 125)
  (h3 : total_cost = 250.00)
  (h4 : sundae_price = 1.40)
  (total_price_condition : num_bars * ice_cream_bar_price + num_sundaes * sundae_price = total_cost) :
  ice_cream_bar_price = 0.60 :=
sorry

end ice_cream_bar_price_l68_68642


namespace second_route_time_l68_68649

-- Defining time for the first route with all green lights
def R_green : ℕ := 10

-- Defining the additional time added by each red light
def per_red_light : ℕ := 3

-- Defining total time for the first route with all red lights
def R_red : ℕ := R_green + 3 * per_red_light

-- Defining the second route time plus the difference
def S : ℕ := R_red - 5

theorem second_route_time : S = 14 := by
  sorry

end second_route_time_l68_68649


namespace minimum_value_of_xy_l68_68671

theorem minimum_value_of_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : xy = 64 :=
sorry

end minimum_value_of_xy_l68_68671


namespace relationship_between_sums_l68_68677

-- Conditions: four distinct positive integers
variables {a b c d : ℕ}
-- additional conditions: positive integers
variables (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) (d_pos : d > 0)

-- Condition: a is the largest and d is the smallest
variables (a_largest : a > b ∧ a > c ∧ a > d)
variables (d_smallest : d < b ∧ d < c ∧ d < a)

-- Condition: a / b = c / d
variables (ratio_condition : a * d = b * c)

theorem relationship_between_sums :
  a + d > b + c :=
sorry

end relationship_between_sums_l68_68677


namespace class_average_weight_l68_68075

theorem class_average_weight :
  (24 * 40 + 16 * 35 + 18 * 42 + 22 * 38) / (24 + 16 + 18 + 22) = 38.9 :=
by
  -- skipped proof
  sorry

end class_average_weight_l68_68075


namespace largest_real_number_l68_68399

theorem largest_real_number (x : ℝ) (h : (⌊x⌋ / x) = (8 / 9)) : x ≤ 63 / 8 :=
by
  sorry

end largest_real_number_l68_68399


namespace sin_add_arctan_arcsin_l68_68382

theorem sin_add_arctan_arcsin :
  let a := Real.arcsin (4 / 5)
  let b := Real.arctan 3
  (Real.sin a = 4 / 5) →
  (Real.tan b = 3) →
  Real.sin (a + b) = (13 * Real.sqrt 10) / 50 :=
by
  intros _ _
  sorry

end sin_add_arctan_arcsin_l68_68382


namespace probability_of_winning_second_lawsuit_l68_68111

theorem probability_of_winning_second_lawsuit
  (P_W1 P_L1 P_W2 P_L2 : ℝ)
  (h1 : P_W1 = 0.30)
  (h2 : P_L1 = 0.70)
  (h3 : P_L1 * P_L2 = P_W1 * P_W2 + 0.20)
  (h4 : P_L2 = 1 - P_W2) :
  P_W2 = 0.50 :=
by
  sorry

end probability_of_winning_second_lawsuit_l68_68111


namespace calories_per_person_l68_68430

open Nat

theorem calories_per_person :
  ∀ (oranges people pieces_per_orange calories_per_orange : ℕ),
    oranges = 5 →
    pieces_per_orange = 8 →
    people = 4 →
    calories_per_orange = 80 →
    (oranges * pieces_per_orange) / people * (calories_per_orange / pieces_per_orange) = 100 :=
by
  intros oranges people pieces_per_orange calories_per_orange
  assume h_oranges h_pieces_per_orange h_people h_calories_per_orange
  rw [h_oranges, h_pieces_per_orange, h_people, h_calories_per_orange]
  norm_num
  sorry

end calories_per_person_l68_68430


namespace sum_four_least_tau_equals_eight_l68_68038

def tau (n : ℕ) : ℕ := n.divisors.card

theorem sum_four_least_tau_equals_eight :
  ∃ n1 n2 n3 n4 : ℕ, 
    tau n1 + tau (n1 + 1) = 8 ∧ 
    tau n2 + tau (n2 + 1) = 8 ∧
    tau n3 + tau (n3 + 1) = 8 ∧
    tau n4 + tau (n4 + 1) = 8 ∧
    n1 + n2 + n3 + n4 = 80 := 
sorry

end sum_four_least_tau_equals_eight_l68_68038


namespace boys_without_notebooks_l68_68020

/-
Given that:
1. There are 16 boys in Ms. Green's history class.
2. 20 students overall brought their notebooks to class.
3. 11 of the students who brought notebooks are girls.

Prove that the number of boys who did not bring their notebooks is 7.
-/

theorem boys_without_notebooks (total_boys : ℕ) (total_notebooks : ℕ) (girls_with_notebooks : ℕ)
  (hb : total_boys = 16) (hn : total_notebooks = 20) (hg : girls_with_notebooks = 11) : 
  (total_boys - (total_notebooks - girls_with_notebooks) = 7) :=
by
  sorry

end boys_without_notebooks_l68_68020


namespace smallest_possible_n_l68_68841

theorem smallest_possible_n (n : ℕ) (h_pos: n > 0)
  (h_int: (1/3 : ℚ) + 1/4 + 1/9 + 1/n = (1:ℚ)) : 
  n = 18 :=
sorry

end smallest_possible_n_l68_68841


namespace smallest_non_10_multiple_abundant_l68_68664

def is_abundant (n : ℕ) : Prop :=
  (∑ d in Nat.properDivisors n, d) > n

def not_multiple_of_10 (n : ℕ) : Prop :=
  ¬ (10 ∣ n)

theorem smallest_non_10_multiple_abundant :
  ∀ n : ℕ, is_abundant n ∧ not_multiple_of_10 n → n = 12 :=
begin
  assume n,
  intros h,
  sorry
end

end smallest_non_10_multiple_abundant_l68_68664


namespace polynomial_divisibility_l68_68517

theorem polynomial_divisibility (m : ℤ) : (3 * (-2)^2 + 5 * (-2) + m = 0) ↔ (m = -2) :=
by
  sorry

end polynomial_divisibility_l68_68517


namespace find_z_l68_68721

theorem find_z (x y z w : ℚ) (h1 : x > y) (h2 : y > z) 
    (h3 : x - 1 = y) (h4 : y - 1 = z) 
    (h5 : w = 5 * x / 3) (h6 : w^2 = x * z) 
    (h7 : 2 * x + 3 * y + 3 * z = 5 * y + 11) : 
    z = 3 :=
by
  -- proof will be inserted here
  sorry

end find_z_l68_68721


namespace fraction_of_girls_at_dance_l68_68612

theorem fraction_of_girls_at_dance :
  (270 : ℚ) * (4 / 9) + (180 : ℚ) * (5 / 9) = 220 ∧
  450 = 450 →
  (220 : ℚ) / 450 = 22 / 45 := 
by sorry

end fraction_of_girls_at_dance_l68_68612


namespace least_five_digit_perfect_square_and_cube_l68_68256

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ ∃ k : ℕ, k^6 = n ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68256


namespace necessary_not_sufficient_condition_l68_68911

theorem necessary_not_sufficient_condition (x : ℝ) : 
  x^2 - 2 * x - 3 < 0 → -2 < x ∧ x < 3 :=
by  
  sorry

end necessary_not_sufficient_condition_l68_68911


namespace factorial_units_digit_l68_68017

theorem factorial_units_digit (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (hba : a < b) : 
  ¬ (∃ k : ℕ, (b! - a!) % 10 = 7) := 
sorry

end factorial_units_digit_l68_68017


namespace inequality1_inequality2_l68_68451

variables (Γ B P : ℕ)

def convex_polyhedron : Prop :=
  Γ - B + P = 2

theorem inequality1 (h : convex_polyhedron Γ B P) : 
  3 * Γ ≥ 6 + P :=
sorry

theorem inequality2 (h : convex_polyhedron Γ B P) : 
  3 * B ≥ 6 + P :=
sorry

end inequality1_inequality2_l68_68451


namespace Kyle_is_25_l68_68712

-- Definitions based on the conditions
def Tyson_age : Nat := 20
def Frederick_age : Nat := 2 * Tyson_age
def Julian_age : Nat := Frederick_age - 20
def Kyle_age : Nat := Julian_age + 5

-- The theorem to prove
theorem Kyle_is_25 : Kyle_age = 25 := by
  sorry

end Kyle_is_25_l68_68712


namespace building_height_l68_68833

theorem building_height :
  ∀ (n1 n2: ℕ) (h1 h2: ℕ),
  n1 = 10 → n2 = 10 → h1 = 12 → h2 = h1 + 3 →
  (n1 * h1 + n2 * h2) = 270 := 
by {
  intros n1 n2 h1 h2 h1_eq h2_eq h3_eq h4_eq,
  rw [h1_eq, h2_eq, h3_eq, h4_eq],
  simp,
  sorry
}

end building_height_l68_68833


namespace least_five_digit_perfect_square_cube_l68_68276

theorem least_five_digit_perfect_square_cube :
  ∃ n : Nat, (10000 ≤ n ∧ n < 100000) ∧ (∃ m1 m2 : Nat, n = m1^2 ∧ n = m2^3 ∧ n = 15625) :=
by
  sorry

end least_five_digit_perfect_square_cube_l68_68276


namespace find_p_l68_68332

theorem find_p (m n p : ℝ) 
  (h1 : m = 3 * n + 5) 
  (h2 : m + 2 = 3 * (n + p) + 5) : p = 2 / 3 :=
by
  sorry

end find_p_l68_68332


namespace divide_rope_length_l68_68350

-- Definitions of variables based on the problem conditions
def rope_length : ℚ := 8 / 15
def num_parts : ℕ := 3

-- Theorem statement
theorem divide_rope_length :
  (1 / num_parts = (1 : ℚ) / 3) ∧ (rope_length * (1 / num_parts) = 8 / 45) :=
by
  sorry

end divide_rope_length_l68_68350


namespace geometric_progressions_sum_eq_l68_68741

variable {a q b : ℝ}
variable {n : ℕ}
variable (h1 : q ≠ 1)

/-- The given statement in Lean 4 -/
theorem geometric_progressions_sum_eq (h : a * (q^(3*n) - 1) / (q - 1) = b * (q^(3*n) - 1) / (q^3 - 1)) : 
  b = a * (1 + q + q^2) := 
by
  sorry

end geometric_progressions_sum_eq_l68_68741


namespace simplify_and_evaluate_expression_l68_68212

variable (x y : ℝ)
variable (h1 : x = 1)
variable (h2 : y = Real.sqrt 2)

theorem simplify_and_evaluate_expression : 
  (x + 2 * y) ^ 2 - x * (x + 4 * y) + (1 - y) * (1 + y) = 7 := by
  sorry

end simplify_and_evaluate_expression_l68_68212


namespace off_road_vehicle_cost_l68_68429

theorem off_road_vehicle_cost
  (dirt_bike_count : ℕ) (dirt_bike_cost : ℕ)
  (off_road_vehicle_count : ℕ) (register_cost : ℕ)
  (total_cost : ℕ) (off_road_vehicle_cost : ℕ) :
  dirt_bike_count = 3 → dirt_bike_cost = 150 →
  off_road_vehicle_count = 4 → register_cost = 25 →
  total_cost = 1825 →
  3 * dirt_bike_cost + 4 * off_road_vehicle_cost + 7 * register_cost = total_cost →
  off_road_vehicle_cost = 300 := by
  intros h1 h2 h3 h4 h5 h6
  sorry

end off_road_vehicle_cost_l68_68429


namespace min_value_of_expr_l68_68824

noncomputable def min_value (x y : ℝ) : ℝ :=
  (4 * x^2) / (y + 1) + (y^2) / (2*x + 2)

theorem min_value_of_expr : 
  ∀ (x y : ℝ), (0 < x) → (0 < y) → (2 * x + y = 2) →
  min_value x y = 4 / 5 :=
by
  intros x y hx hy hxy
  sorry

end min_value_of_expr_l68_68824


namespace graph_t_intersects_x_axis_exists_integer_a_with_integer_points_on_x_axis_intersection_l68_68202

open Real

def function_y (a x : ℝ) : ℝ := (4 * a + 2) * x^2 + (9 - 6 * a) * x - 4 * a + 4

theorem graph_t_intersects_x_axis (a : ℝ) : ∃ x : ℝ, function_y a x = 0 :=
by sorry

theorem exists_integer_a_with_integer_points_on_x_axis_intersection :
  ∃ (a : ℤ), 
  (∀ x : ℝ, (function_y a x = 0) → ∃ (x_int : ℤ), x = x_int) ∧ 
  (a = -2 ∨ a = -1 ∨ a = 0 ∨ a = 1) :=
by sorry

end graph_t_intersects_x_axis_exists_integer_a_with_integer_points_on_x_axis_intersection_l68_68202


namespace john_text_messages_l68_68029

/-- John decides to get a new phone number and it ends up being a recycled number. 
    He used to get some text messages a day. 
    Now he is getting 55 text messages a day, 
    and he is getting 245 text messages per week that are not intended for him. 
    How many text messages a day did he used to get?
-/
theorem john_text_messages (m : ℕ) (h1 : 55 = m + 35) (h2 : 245 = 7 * 35) : m = 20 := 
by 
  sorry

end john_text_messages_l68_68029


namespace vans_needed_l68_68514

-- Given Conditions
def van_capacity : ℕ := 4
def students : ℕ := 2
def adults : ℕ := 6
def total_people : ℕ := students + adults

-- Theorem to prove
theorem vans_needed : total_people / van_capacity = 2 :=
by
  -- Proof will be added here
  sorry

end vans_needed_l68_68514


namespace Ben_hits_7_l68_68998

def regions : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
def Alice_score : ℕ := 18
def Ben_score : ℕ := 13
def Cindy_score : ℕ := 19
def Dave_score : ℕ := 16
def Ellen_score : ℕ := 20
def Frank_score : ℕ := 5

def hit_score (name : String) (region1 region2 : ℕ) (score : ℕ) : Prop :=
  region1 ∈ regions ∧ region2 ∈ regions ∧ region1 ≠ region2 ∧ region1 + region2 = score

theorem Ben_hits_7 :
  ∃ r1 r2, hit_score "Ben" r1 r2 Ben_score ∧ (r1 = 7 ∨ r2 = 7) :=
sorry

end Ben_hits_7_l68_68998


namespace james_driving_speed_l68_68186

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

end james_driving_speed_l68_68186


namespace work_duration_l68_68088

theorem work_duration (work_rate_x work_rate_y : ℚ) (time_x : ℕ) (total_work : ℚ) :
  work_rate_x = (1 / 20) → 
  work_rate_y = (1 / 12) → 
  time_x = 4 → 
  total_work = 1 →
  ((time_x * work_rate_x) + ((total_work - (time_x * work_rate_x)) / (work_rate_x + work_rate_y))) = 10 := 
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end work_duration_l68_68088


namespace anya_hairs_wanted_more_l68_68361

def anya_initial_number_of_hairs : ℕ := 0 -- for simplicity, assume she starts with 0 hairs
def hairs_lost_washing : ℕ := 32
def hairs_lost_brushing : ℕ := hairs_lost_washing / 2
def total_hairs_lost : ℕ := hairs_lost_washing + hairs_lost_brushing
def hairs_to_grow_back : ℕ := 49

theorem anya_hairs_wanted_more : total_hairs_lost + hairs_to_grow_back = 97 :=
by
  sorry

end anya_hairs_wanted_more_l68_68361


namespace mary_final_books_l68_68043

-- Initial number of books
def initial_books : ℕ := 72

-- Books received each month from book club for 12 months
def books_from_club : ℕ := 12 * 1

-- Books bought from different sources
def books_from_bookstore : ℕ := 5
def books_from_yard_sales : ℕ := 2

-- Books received as gifts
def books_from_daughter : ℕ := 1
def books_from_mother : ℕ := 4

-- Books gotten rid of
def books_donated : ℕ := 12
def books_sold : ℕ := 3

-- Final calculation
theorem mary_final_books : 
  initial_books + books_from_club + books_from_bookstore + books_from_yard_sales + books_from_daughter + books_from_mother - (books_donated + books_sold) = 81 :=
  by sorry

end mary_final_books_l68_68043


namespace min_class_size_l68_68828

theorem min_class_size (x : ℕ) (h : 50 ≤ 5 * x + 2) : 52 ≤ 5 * x + 2 :=
by
  sorry

end min_class_size_l68_68828


namespace least_five_digit_is_15625_l68_68281

noncomputable def least_five_digit_perfect_square_and_cube : ℕ :=
  15625

theorem least_five_digit_is_15625 :
  (10000 ≤ least_five_digit_perfect_square_and_cube) ∧ (least_five_digit_perfect_square_and_cube < 100000) ∧
  (∃ a : ℕ, least_five_digit_perfect_square_and_cube = a ^ 6) :=
by
  sorry

end least_five_digit_is_15625_l68_68281


namespace max_product_partition_l68_68809

theorem max_product_partition (k n : ℕ) (hkn : k ≥ n) 
  (q r : ℕ) (hqr : k = n * q + r) (h_r : 0 ≤ r ∧ r < n) : 
  ∃ (F : ℕ → ℕ), F k = q^(n-r) * (q+1)^r :=
by
  sorry

end max_product_partition_l68_68809


namespace solve_inequality_l68_68685

noncomputable def f (x : ℝ) : ℝ := -x^2 + 2 * x

theorem solve_inequality {x : ℝ} (hx : 0 < x) : 
  f (Real.log x / Real.log 2) < f 2 ↔ (0 < x ∧ x < 1) ∨ (4 < x) :=
by
sorry

end solve_inequality_l68_68685


namespace lcm_hcf_product_l68_68633

theorem lcm_hcf_product (lcm hcf a b : ℕ) (hlcm : lcm = 2310) (hhcf : hcf = 30) (ha : a = 330) (eq : lcm * hcf = a * b) : b = 210 :=
by {
  sorry
}

end lcm_hcf_product_l68_68633


namespace largest_real_number_l68_68400

theorem largest_real_number (x : ℝ) (h : (⌊x⌋ / x) = (8 / 9)) : x ≤ 63 / 8 :=
by
  sorry

end largest_real_number_l68_68400


namespace sophie_marble_exchange_l68_68056

theorem sophie_marble_exchange (sophie_initial_marbles joe_initial_marbles : ℕ) 
  (final_ratio : ℕ) (sophie_gives_joe : ℕ) : 
  sophie_initial_marbles = 120 → joe_initial_marbles = 19 → final_ratio = 3 → 
  (120 - sophie_gives_joe = 3 * (19 + sophie_gives_joe)) → sophie_gives_joe = 16 := 
by
  intros h1 h2 h3 h4
  sorry

end sophie_marble_exchange_l68_68056


namespace sphere_diagonal_property_l68_68495

variable {A B C D : ℝ}

-- conditions provided
variable (radius : ℝ) (x y z : ℝ)
variable (h_radius : radius = 1)
variable (h_non_coplanar : ¬(is_coplanar A B C D))
variable (h_AB_CD : dist A B = x ∧ dist C D = x)
variable (h_BC_DA : dist B C = y ∧ dist D A = y)
variable (h_CA_BD : dist C A = z ∧ dist B D = z)

theorem sphere_diagonal_property :
  x^2 + y^2 + z^2 = 8 := 
sorry

end sphere_diagonal_property_l68_68495


namespace largest_value_of_b_l68_68719

theorem largest_value_of_b (b : ℚ) (h : (2 * b + 5) * (b - 1) = 6 * b) : b = 5 / 2 :=
by
  sorry

end largest_value_of_b_l68_68719


namespace traci_flour_brought_l68_68746

-- Definitions based on the conditions
def harris_flour : ℕ := 400
def flour_per_cake : ℕ := 100
def cakes_each : ℕ := 9

-- Proving the amount of flour Traci brought
theorem traci_flour_brought :
  (cakes_each * flour_per_cake) - harris_flour = 500 :=
by
  sorry

end traci_flour_brought_l68_68746


namespace avg_speed_l68_68860

noncomputable def jane_total_distance : ℝ := 120
noncomputable def time_period_hours : ℝ := 7

theorem avg_speed :
  jane_total_distance / time_period_hours = (120 / 7 : ℝ):=
by
  sorry

end avg_speed_l68_68860


namespace only_zero_solution_l68_68051

theorem only_zero_solution (a b c n : ℤ) (h_gcd : Int.gcd (Int.gcd (Int.gcd a b) c) n = 1)
  (h_eq : 6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 :=
sorry

end only_zero_solution_l68_68051


namespace puppy_ratios_l68_68646

theorem puppy_ratios :
  ∀(total_puppies : ℕ)(golden_retriever_females golden_retriever_males : ℕ)
   (labrador_females labrador_males : ℕ)(poodle_females poodle_males : ℕ)
   (beagle_females beagle_males : ℕ),
  total_puppies = golden_retriever_females + golden_retriever_males +
                  labrador_females + labrador_males +
                  poodle_females + poodle_males +
                  beagle_females + beagle_males →
  golden_retriever_females = 2 →
  golden_retriever_males = 4 →
  labrador_females = 1 →
  labrador_males = 3 →
  poodle_females = 3 →
  poodle_males = 2 →
  beagle_females = 1 →
  beagle_males = 2 →
  (golden_retriever_females / golden_retriever_males = 1 / 2) ∧
  (labrador_females / labrador_males = 1 / 3) ∧
  (poodle_females / poodle_males = 3 / 2) ∧
  (beagle_females / beagle_males = 1 / 2) ∧
  (7 / 11 = (golden_retriever_females + labrador_females + poodle_females + beagle_females) / 
            (golden_retriever_males + labrador_males + poodle_males + beagle_males)) :=
by intros;
   sorry

end puppy_ratios_l68_68646


namespace power_of_two_ends_with_identical_digits_l68_68794

theorem power_of_two_ends_with_identical_digits : ∃ (k : ℕ), k ≥ 10 ∧ (∀ (x y : ℕ), 2^k = 1000 * x + 111 * y → y = 8 → (2^k % 1000 = 888)) :=
by sorry

end power_of_two_ends_with_identical_digits_l68_68794


namespace least_five_digit_perfect_square_and_cube_l68_68307

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, n = 5^6 ∧ (n >= 10000 ∧ n < 100000) ∧ (∃ k : ℕ, n = k^2) ∧ (∃ m : ℕ, n = m^3) :=
by
  use 15625
  split
  · exact pow_succ 5 6
  split
  · have h1: 10000 ≤ 5^6 := nat.le_of_eq (calc 5^6 = 15625 : rfl)
    have h2: 5^6 < 100000 := nat.lt_of_eq_of_lt (calc 5^6 = 15625 : rfl) (by decide)
    exact ⟨h1, h2⟩
  · use 125
    exact pow_two 125
  · use 25
    exact pow_three 25
  sorry

end least_five_digit_perfect_square_and_cube_l68_68307


namespace art_collection_total_area_l68_68369

-- Define the dimensions and quantities of the paintings
def square_painting_side := 6
def small_painting_width := 2
def small_painting_height := 3
def large_painting_width := 10
def large_painting_height := 15

def num_square_paintings := 3
def num_small_paintings := 4
def num_large_paintings := 1

-- Define areas of individual paintings
def square_painting_area := square_painting_side * square_painting_side
def small_painting_area := small_painting_width * small_painting_height
def large_painting_area := large_painting_width * large_painting_height

-- Define the total area calculation
def total_area :=
  num_square_paintings * square_painting_area +
  num_small_paintings * small_painting_area +
  num_large_paintings * large_painting_area

-- The theorem statement
theorem art_collection_total_area : total_area = 282 := by
  sorry

end art_collection_total_area_l68_68369


namespace find_principal_amount_l68_68482

variable {P R T : ℝ} -- variables for principal, rate, and time
variable (H1: R = 25)
variable (H2: T = 2)
variable (H3: (P * (0.5625) - P * (0.5)) = 225)

theorem find_principal_amount
    (H1 : R = 25)
    (H2 : T = 2)
    (H3 : (P * 0.0625) = 225) : 
    P = 3600 := 
  sorry

end find_principal_amount_l68_68482


namespace initial_marbles_l68_68939

theorem initial_marbles (M : ℝ) (h0 : 0.2 * M + 0.35 * (0.8 * M) + 130 = M) : M = 250 :=
by
  sorry

end initial_marbles_l68_68939


namespace cos_alpha_plus_pi_div_4_value_l68_68679

noncomputable def cos_alpha_plus_pi_div_4 (α : ℝ) (h1 : π / 2 < α ∧ α < π) (h2 : Real.sin (α - 3 * π / 4) = 3 / 5) : Real :=
  Real.cos (α + π / 4)

theorem cos_alpha_plus_pi_div_4_value (α : ℝ) (h1 : π / 2 < α ∧ α < π) (h2 : Real.sin (α - 3 * π / 4) = 3 / 5) :
  cos_alpha_plus_pi_div_4 α h1 h2 = -4 / 5 :=
sorry

end cos_alpha_plus_pi_div_4_value_l68_68679


namespace platform_length_l68_68091

theorem platform_length
  (train_length : ℕ)
  (time_pole : ℕ)
  (time_platform : ℕ)
  (h_train_length : train_length = 300)
  (h_time_pole : time_pole = 18)
  (h_time_platform : time_platform = 39) :
  ∃ (platform_length : ℕ), platform_length = 350 :=
by
  sorry

end platform_length_l68_68091


namespace train_length_is_400_l68_68895

-- Define the conditions
def time := 40 -- seconds
def speed_kmh := 36 -- km/h

-- Conversion factor from km/h to m/s
def kmh_to_ms (v : ℕ) := (v * 5) / 18

def speed_ms := kmh_to_ms speed_kmh -- convert speed to m/s

-- Definition of length of the train using the given conditions
def train_length := speed_ms * time

-- Theorem to prove the length of the train is 400 meters
theorem train_length_is_400 : train_length = 400 := by
  sorry

end train_length_is_400_l68_68895


namespace exactly_one_valid_N_l68_68569

def four_digit_number (N : ℕ) : Prop := 1000 ≤ N ∧ N < 10000

def condition (N x a : ℕ) : Prop := 
  N = 1000 * a + x ∧ x = N / 7

theorem exactly_one_valid_N : 
  ∃! N : ℕ, ∃ x a : ℕ, four_digit_number N ∧ condition N x a :=
sorry

end exactly_one_valid_N_l68_68569


namespace area_quadrilateral_ABCDE_correct_l68_68700

noncomputable def area_quadrilateral_ABCDE (AM NM AN BN BO OC CP CD EP DE : ℝ) : ℝ :=
  (0.5 * AM * NM * Real.sqrt 2) + (0.5 * BN * BO) + (0.5 * OC * CP * Real.sqrt 2) - (0.5 * DE * EP)

theorem area_quadrilateral_ABCDE_correct :
  ∀ (AM NM AN BN BO OC CP CD EP DE : ℝ),
    DE = 12 ∧ 
    AM = 36 ∧ 
    NM = 36 ∧ 
    AN = 36 * Real.sqrt 2 ∧
    BN = 36 * Real.sqrt 2 - 36 ∧
    BO = 36 ∧
    OC = 36 ∧
    CP = 36 * Real.sqrt 2 ∧
    CD = 24 ∧
    EP = 24
    → area_quadrilateral_ABCDE AM NM AN BN BO OC CP CD EP DE = 2311.2 * Real.sqrt 2 + 504 :=
by intro AM NM AN BN BO OC CP CD EP DE h;
   cases h;
   sorry

end area_quadrilateral_ABCDE_correct_l68_68700


namespace largest_x_l68_68395

-- Conditions setup
def largest_x_satisfying_condition : ℝ :=
  let x : ℝ := 63 / 8 in x

theorem largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 8 / 9) : 
  x = largest_x_satisfying_condition :=
by
  sorry

end largest_x_l68_68395


namespace symmetric_line_equation_l68_68222

theorem symmetric_line_equation (x y : ℝ) (h₁ : x + y + 1 = 0) : (2 - x) + (4 - y) - 7 = 0 :=
by
  sorry

end symmetric_line_equation_l68_68222


namespace factorization_sum_l68_68734

theorem factorization_sum (a b c : ℤ) 
  (h1 : ∀ x : ℤ, x^2 + 9 * x + 20 = (x + a) * (x + b))
  (h2 : ∀ x : ℤ, x^2 + 7 * x - 60 = (x + b) * (x - c)) :
  a + b + c = 21 :=
by
  sorry

end factorization_sum_l68_68734


namespace least_five_digit_perfect_square_and_cube_l68_68257

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ ∃ k : ℕ, k^6 = n ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68257


namespace actual_average_height_l68_68178

theorem actual_average_height (average_height : ℝ) (num_students : ℕ)
  (incorrect_heights actual_heights : Fin 3 → ℝ)
  (h_avg : average_height = 165)
  (h_num : num_students = 50)
  (h_incorrect : incorrect_heights 0 = 150 ∧ incorrect_heights 1 = 175 ∧ incorrect_heights 2 = 190)
  (h_actual : actual_heights 0 = 135 ∧ actual_heights 1 = 170 ∧ actual_heights 2 = 185) :
  (average_height * num_students 
   - (incorrect_heights 0 + incorrect_heights 1 + incorrect_heights 2) 
   + (actual_heights 0 + actual_heights 1 + actual_heights 2))
   / num_students = 164.5 :=
by
  -- proof steps here
  sorry

end actual_average_height_l68_68178


namespace dave_trips_l68_68883

/-- Dave can only carry 9 trays at a time. -/
def trays_per_trip := 9

/-- Number of trays Dave has to pick up from one table. -/
def trays_from_table1 := 17

/-- Number of trays Dave has to pick up from another table. -/
def trays_from_table2 := 55

/-- Total number of trays Dave has to pick up. -/
def total_trays := trays_from_table1 + trays_from_table2

/-- The number of trips Dave will make. -/
def number_of_trips := total_trays / trays_per_trip

theorem dave_trips :
  number_of_trips = 8 :=
sorry

end dave_trips_l68_68883


namespace tank_never_fills_l68_68450

structure Pipe :=
(rate1 : ℕ) (rate2 : ℕ)

def net_flow (pA pB pC pD : Pipe) (time1 time2 : ℕ) : ℤ :=
  let fillA := pA.rate1 * time1 + pA.rate2 * time2
  let fillB := pB.rate1 * time1 + pB.rate2 * time2
  let drainC := pC.rate1 * time1 + pC.rate2 * time2
  let drainD := pD.rate1 * (time1 + time2)
  (fillA + fillB) - (drainC + drainD)

theorem tank_never_fills (pA pB pC pD : Pipe) (time1 time2 : ℕ)
  (hA : pA = Pipe.mk 40 20) (hB : pB = Pipe.mk 20 40) 
  (hC : pC = Pipe.mk 20 40) (hD : pD = Pipe.mk 30 30) 
  (hTime : time1 = 30 ∧ time2 = 30): 
  net_flow pA pB pC pD time1 time2 = 0 := by
  sorry

end tank_never_fills_l68_68450


namespace modulus_remainder_l68_68798

theorem modulus_remainder (n : ℕ) 
  (h1 : n^3 % 7 = 3) 
  (h2 : n^4 % 7 = 2) : 
  n % 7 = 6 :=
by
  sorry

end modulus_remainder_l68_68798


namespace purchasing_plans_and_optimal_plan_l68_68094

def company_time := 10
def model_A_cost := 60000
def model_B_cost := 40000
def model_A_production := 15
def model_B_production := 10
def budget := 440000
def production_capacity := 102

theorem purchasing_plans_and_optimal_plan (x y : ℕ) (h1 : x + y = company_time) (h2 : model_A_cost * x + model_B_cost * y ≤ budget) :
  (x = 0 ∧ y = 10) ∨ (x = 1 ∧ y = 9) ∨ (x = 2 ∧ y = 8) ∧ (x = 1 ∧ y = 9) :=
by 
  sorry

end purchasing_plans_and_optimal_plan_l68_68094


namespace billy_scores_two_points_each_round_l68_68657

def billy_old_score := 725
def billy_rounds := 363
def billy_target_score := billy_old_score + 1
def billy_points_per_round := billy_target_score / billy_rounds

theorem billy_scores_two_points_each_round :
  billy_points_per_round = 2 := by
  sorry

end billy_scores_two_points_each_round_l68_68657


namespace find_N_l68_68422

theorem find_N (N : ℤ) :
  (10 + 11 + 12) / 3 = (2010 + 2011 + 2012 + N) / 4 → N = -5989 :=
by
  sorry

end find_N_l68_68422


namespace greatest_possible_sum_of_visible_numbers_l68_68803

theorem greatest_possible_sum_of_visible_numbers :
  ∀ (numbers : ℕ → ℕ) (Cubes : Fin 4 → ℤ), 
  (numbers 0 = 1) → (numbers 1 = 3) → (numbers 2 = 9) → (numbers 3 = 27) → (numbers 4 = 81) → (numbers 5 = 243) →
  (Cubes 0 = (16 - 2) * (243 + 81 + 27 + 9 + 3)) → 
  (Cubes 1 = (16 - 2) * (243 + 81 + 27 + 9 + 3)) →
  (Cubes 2 = (16 - 2) * (243 + 81 + 27 + 9 + 3)) ->
  (Cubes 3 = 16 * (243 + 81 + 27 + 9 + 3)) ->
  (Cubes 0 + Cubes 1 + Cubes 2 + Cubes 3 = 1452) :=
by 
  sorry

end greatest_possible_sum_of_visible_numbers_l68_68803


namespace record_expenditure_l68_68819

theorem record_expenditure (income_recording : ℤ) (expenditure_amount : ℤ) (h : income_recording = 20) : -expenditure_amount = -50 :=
by sorry

end record_expenditure_l68_68819


namespace total_distance_run_l68_68766

-- Given conditions
def number_of_students : Nat := 18
def distance_per_student : Nat := 106

-- Prove that the total distance run by the students equals 1908 meters.
theorem total_distance_run : number_of_students * distance_per_student = 1908 := by
  sorry

end total_distance_run_l68_68766


namespace least_five_digit_perfect_square_and_cube_l68_68298

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (n >= 10000 ∧ n < 100000) ∧ (∃ a : ℕ, n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l68_68298


namespace sum_is_3600_l68_68880

variables (P R T : ℝ)
variables (CI SI : ℝ)

theorem sum_is_3600
  (hR : R = 10)
  (hT : T = 2)
  (hCI : CI = P * (1 + R / 100) ^ T - P)
  (hSI : SI = P * R * T / 100)
  (h_diff : CI - SI = 36) :
  P = 3600 :=
sorry

end sum_is_3600_l68_68880


namespace smallest_spherical_triangle_angle_l68_68071

-- Define the conditions
def is_ratio (a b c : ℕ) : Prop := a = 4 ∧ b = 5 ∧ c = 6
def sum_of_angles (α β γ : ℕ) : Prop := α + β + γ = 270

-- Define the problem statement
theorem smallest_spherical_triangle_angle 
  (a b c α β γ : ℕ)
  (h1 : is_ratio a b c)
  (h2 : sum_of_angles (a * α) (b * β) (c * γ)) :
  a * α = 72 := 
sorry

end smallest_spherical_triangle_angle_l68_68071


namespace value_of_M_l68_68014

theorem value_of_M (M : ℝ) (H : 0.25 * M = 0.55 * 1500) : M = 3300 := 
by
  sorry

end value_of_M_l68_68014


namespace jessica_seashells_l68_68432

theorem jessica_seashells (joan jessica total : ℕ) (h1 : joan = 6) (h2 : total = 14) (h3 : total = joan + jessica) : jessica = 8 :=
by
  -- proof steps would go here
  sorry

end jessica_seashells_l68_68432


namespace cone_lateral_area_l68_68159

/--
Given that the radius of the base of a cone is 3 cm and the slant height is 6 cm,
prove that the lateral area of this cone is 18π cm².
-/
theorem cone_lateral_area {r l : ℝ} (h_radius : r = 3) (h_slant_height : l = 6) :
  (π * r * l) = 18 * π :=
by
  have h1 : r = 3 := h_radius
  have h2 : l = 6 := h_slant_height
  rw [h1, h2]
  norm_num
  sorry

end cone_lateral_area_l68_68159


namespace part1_l68_68927

noncomputable def P : Set ℝ := {x | (1 / 2) ≤ x ∧ x ≤ 1}
noncomputable def Q (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}
def U : Set ℝ := Set.univ
noncomputable def complement_P : Set ℝ := {x | x < (1 / 2)} ∪ {x | x > 1}

theorem part1 (a : ℝ) (h : a = 1) : 
  (complement_P ∩ Q a) = {x | 1 < x ∧ x ≤ 2} :=
sorry

end part1_l68_68927


namespace intersecting_line_at_one_point_l68_68226

theorem intersecting_line_at_one_point (k : ℝ) :
  (∃ y : ℝ, k = -3 * y^2 - 4 * y + 7 ∧ 
           ∀ z : ℝ, k = -3 * z^2 - 4 * z + 7 → y = z) ↔ 
  k = 25 / 3 :=
by
  sorry

end intersecting_line_at_one_point_l68_68226


namespace tan_3theta_eq_9_13_l68_68956

open Real

noncomputable def tan3theta (θ : ℝ) (h : tan θ = 3) : Prop :=
  tan (3 * θ) = (9 / 13)

theorem tan_3theta_eq_9_13 (θ : ℝ) (h : tan θ = 3) : tan3theta θ h :=
by
  sorry

end tan_3theta_eq_9_13_l68_68956


namespace combined_resistance_parallel_l68_68085

theorem combined_resistance_parallel (x y : ℝ) (r : ℝ) (hx : x = 3) (hy : y = 5) 
  (h : 1 / r = 1 / x + 1 / y) : r = 15 / 8 :=
by
  sorry

end combined_resistance_parallel_l68_68085


namespace find_A_plus_B_plus_C_plus_D_l68_68850

noncomputable def A : ℤ := -7
noncomputable def B : ℕ := 8
noncomputable def C : ℤ := 21
noncomputable def D : ℕ := 1

def conditions_satisfied : Prop :=
  D > 0 ∧
  ¬∃ p : ℕ, Nat.Prime p ∧ p^2 ∣ B ∧ p ≠ 1 ∧ p ≠ B ∧ p ≥ 2 ∧
  Int.gcd A (Int.gcd C (Int.ofNat D)) = 1

theorem find_A_plus_B_plus_C_plus_D : conditions_satisfied → A + B + C + D = 23 :=
by
  intro h
  sorry

end find_A_plus_B_plus_C_plus_D_l68_68850


namespace potion_kits_needed_l68_68012

-- Definitions
def num_spellbooks := 5
def cost_spellbook_gold := 5
def cost_potion_kit_silver := 20
def num_owls := 1
def cost_owl_gold := 28
def silver_per_gold := 9
def total_silver := 537

-- Prove that Harry needs to buy 3 potion kits.
def Harry_needs_to_buy : Prop :=
  let cost_spellbooks_silver := num_spellbooks * cost_spellbook_gold * silver_per_gold
  let cost_owl_silver := num_owls * cost_owl_gold * silver_per_gold
  let total_cost_silver := cost_spellbooks_silver + cost_owl_silver
  let remaining_silver := total_silver - total_cost_silver
  let num_potion_kits := remaining_silver / cost_potion_kit_silver
  num_potion_kits = 3

theorem potion_kits_needed : Harry_needs_to_buy :=
  sorry

end potion_kits_needed_l68_68012


namespace check_conditions_l68_68150

noncomputable def arithmetic_sequence (a d : ℤ) (n : ℕ) := a + (n - 1) * d

noncomputable def sum_of_first_n_terms (a d : ℤ) (n : ℕ) := n * a + (n * (n - 1) / 2) * d

theorem check_conditions {a d : ℤ}
  (S6 S7 S5 : ℤ)
  (h1 : S6 = sum_of_first_n_terms a d 6)
  (h2 : S7 = sum_of_first_n_terms a d 7)
  (h3 : S5 = sum_of_first_n_terms a d 5)
  (h : S6 > S7 ∧ S7 > S5) :
  d < 0 ∧
  sum_of_first_n_terms a d 11 > 0 ∧
  sum_of_first_n_terms a d 13 < 0 ∧
  sum_of_first_n_terms a d 9 > sum_of_first_n_terms a d 3 := 
sorry

end check_conditions_l68_68150


namespace car_trip_distance_l68_68497

theorem car_trip_distance (speed_first_car speed_second_car : ℝ) (time_first_car time_second_car distance_first_car distance_second_car : ℝ) 
  (h_speed_first : speed_first_car = 30)
  (h_time_first : time_first_car = 1.5)
  (h_speed_second : speed_second_car = 60)
  (h_time_second : time_second_car = 1.3333)
  (h_distance_first : distance_first_car = speed_first_car * time_first_car)
  (h_distance_second : distance_second_car = speed_second_car * time_second_car) :
  distance_first_car = 45 :=
by
  sorry

end car_trip_distance_l68_68497


namespace mika_jogging_speed_l68_68982

theorem mika_jogging_speed 
  (s : ℝ)  -- Mika's constant jogging speed in meters per second.
  (r : ℝ)  -- Radius of the inner semicircle.
  (L : ℝ)  -- Length of each straight section.
  (h1 : 8 > 0) -- Overall width of the track is 8 meters.
  (h2 : (2 * L + 2 * π * (r + 8)) / s = (2 * L + 2 * π * r) / s + 48) -- Time difference equation.
  : s = π / 3 := 
sorry

end mika_jogging_speed_l68_68982


namespace base_b_representation_1987_l68_68680

theorem base_b_representation_1987 (x y z b : ℕ) (h1 : x + y + z = 25) (h2 : x ≥ 1)
  (h3 : 1987 = x * b^2 + y * b + z) (h4 : 12 < b) (h5 : b < 45) :
  x = 5 ∧ y = 9 ∧ z = 11 ∧ b = 19 :=
sorry

end base_b_representation_1987_l68_68680


namespace cyclist_time_no_wind_l68_68645

theorem cyclist_time_no_wind (v w : ℝ) 
    (h1 : v + w = 1 / 3) 
    (h2 : v - w = 1 / 4) : 
    1 / v = 24 / 7 := 
by
  sorry

end cyclist_time_no_wind_l68_68645


namespace find_z_l68_68845

theorem find_z (x y z : ℝ) 
  (h1 : y = 2 * x + 3) 
  (h2 : x + 1 / x = 3.5 + (Real.sin (z * Real.exp (-z)))) :
  z = x^2 + 1 / x^2 := 
sorry

end find_z_l68_68845


namespace selection_count_l68_68885

def choose (n k : ℕ) : ℕ := -- Binomial coefficient definition
  if h : 0 ≤ k ∧ k ≤ n then
    Nat.choose n k
  else
    0

theorem selection_count : choose 9 5 - choose 6 5 = 120 := by
  sorry

end selection_count_l68_68885


namespace units_digit_of_k3_plus_5k_l68_68593

def k : ℕ := 2024^2 + 3^2024

theorem units_digit_of_k3_plus_5k (k := 2024^2 + 3^2024) : 
  ((k^3 + 5^k) % 10) = 8 := 
by 
  sorry

end units_digit_of_k3_plus_5k_l68_68593


namespace similar_triangle_shortest_side_l68_68503

theorem similar_triangle_shortest_side (a b c: ℝ) (d e f: ℝ) :
  a = 21 ∧ b = 20 ∧ c = 29 ∧ d = 87 ∧ c^2 = a^2 + b^2 ∧ d / c = 3 → e = 60 :=
by
  sorry

end similar_triangle_shortest_side_l68_68503


namespace kanul_initial_amount_l68_68708

-- Definition based on the problem conditions
def spent_on_raw_materials : ℝ := 3000
def spent_on_machinery : ℝ := 2000
def spent_on_labor : ℝ := 1000
def percent_spent : ℝ := 0.15

-- Definition of the total amount initially had by Kanul
def total_amount_initial (X : ℝ) : Prop :=
  spent_on_raw_materials + spent_on_machinery + percent_spent * X + spent_on_labor = X

-- Theorem stating the conclusion based on the given conditions
theorem kanul_initial_amount : ∃ X : ℝ, total_amount_initial X ∧ X = 7058.82 :=
by {
  sorry
}

end kanul_initial_amount_l68_68708


namespace average_age_of_students_is_14_l68_68021

noncomputable def average_age_of_students (student_count : ℕ) (teacher_age : ℕ) (combined_avg_age : ℕ) : ℕ :=
  let total_people := student_count + 1
  let total_combined_age := total_people * combined_avg_age
  let total_student_age := total_combined_age - teacher_age
  total_student_age / student_count

theorem average_age_of_students_is_14 :
  average_age_of_students 50 65 15 = 14 :=
by
  sorry

end average_age_of_students_is_14_l68_68021


namespace smallest_positive_n_l68_68623

theorem smallest_positive_n (n : ℕ) : n > 0 → (3 * n ≡ 1367 [MOD 26]) → n = 5 :=
by
  intros _ _
  sorry

end smallest_positive_n_l68_68623
