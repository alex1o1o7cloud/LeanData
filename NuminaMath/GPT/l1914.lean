import Mathlib

namespace barry_wand_trick_l1914_191460

theorem barry_wand_trick (n : ℕ) (h : (n + 3 : ℝ) / 3 = 50) : n = 147 := by
  sorry

end barry_wand_trick_l1914_191460


namespace unique_pair_natural_numbers_l1914_191479

theorem unique_pair_natural_numbers (a b : ℕ) :
  (∀ n : ℕ, ∃ c : ℕ, a ^ n + b ^ n = c ^ (n + 1)) → (a = 2 ∧ b = 2) :=
by
  sorry

end unique_pair_natural_numbers_l1914_191479


namespace find_smallest_x_l1914_191417

def smallest_x_divisible (y : ℕ) : ℕ :=
  if y = 11 then 257 else 0

theorem find_smallest_x : 
  smallest_x_divisible 11 = 257 ∧ 
  ∃ k : ℕ, 264 * k - 7 = 257 :=
by
  sorry

end find_smallest_x_l1914_191417


namespace speed_of_man_in_still_water_l1914_191400

theorem speed_of_man_in_still_water
  (v_m v_s : ℝ)
  (h1 : v_m + v_s = 4)
  (h2 : v_m - v_s = 2) :
  v_m = 3 := 
by sorry

end speed_of_man_in_still_water_l1914_191400


namespace solve_for_x_l1914_191428

theorem solve_for_x (x : ℝ) : (5 : ℝ)^(x + 6) = (625 : ℝ)^x → x = 2 :=
by
  sorry

end solve_for_x_l1914_191428


namespace sum_of_roots_l1914_191415

theorem sum_of_roots (x₁ x₂ b : ℝ) (h : ∀ x : ℝ, x^2 - 2 * x + b = 0 → x = x₁ ∨ x = x₂) :
  x₁ + x₂ = 2 :=
sorry

end sum_of_roots_l1914_191415


namespace smallest_positive_integer_l1914_191477

theorem smallest_positive_integer :
  ∃ x : ℕ, x > 0 ∧ x % 5 = 2 ∧ x % 3 = 1 ∧ x % 7 = 3 ∧ ∀ y : ℕ, y > 0 ∧ y % 5 = 2 ∧ y % 3 = 1 ∧ y % 7 = 3 → x ≤ y :=
by
  sorry

end smallest_positive_integer_l1914_191477


namespace number_of_classes_l1914_191419

theorem number_of_classes (total_basketballs classes_basketballs : ℕ) (h1 : total_basketballs = 54) (h2 : classes_basketballs = 7) : total_basketballs / classes_basketballs = 7 := by
  sorry

end number_of_classes_l1914_191419


namespace find_f_of_9_l1914_191459

variable (f : ℝ → ℝ)

-- Conditions
axiom functional_eq : ∀ x y : ℝ, f (x + y) = f x * f y
axiom f_of_3 : f 3 = 4

-- Theorem statement to prove
theorem find_f_of_9 : f 9 = 64 := by
  sorry

end find_f_of_9_l1914_191459


namespace min_sum_of_inverses_l1914_191431

theorem min_sum_of_inverses 
  (x y z p q r : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
  (h_sum : x + y + z + p + q + r = 10) :
  (1 / x + 9 / y + 4 / z + 25 / p + 16 / q + 36 / r) = 44.1 :=
sorry

end min_sum_of_inverses_l1914_191431


namespace all_possible_triples_l1914_191445

theorem all_possible_triples (x y : ℕ) (z : ℤ) (hz : z % 2 = 1)
                            (h : x.factorial + y.factorial = 8 * z + 2017) :
                            (x = 1 ∧ y = 4 ∧ z = -249) ∨
                            (x = 4 ∧ y = 1 ∧ z = -249) ∨
                            (x = 1 ∧ y = 5 ∧ z = -237) ∨
                            (x = 5 ∧ y = 1 ∧ z = -237) := 
  sorry

end all_possible_triples_l1914_191445


namespace ratio_of_men_to_women_l1914_191421

def num_cannoneers : ℕ := 63
def num_people : ℕ := 378
def num_women (C : ℕ) : ℕ := 2 * C
def num_men (total : ℕ) (women : ℕ) : ℕ := total - women

theorem ratio_of_men_to_women : 
  let C := num_cannoneers
  let total := num_people
  let W := num_women C
  let M := num_men total W
  M / W = 2 :=
by
  sorry

end ratio_of_men_to_women_l1914_191421


namespace pens_bought_is_17_l1914_191427

def number_of_pens_bought (C S : ℝ) (bought_pens : ℝ) : Prop :=
  (bought_pens * C = 12 * S) ∧ (0.4 = (S - C) / C)

theorem pens_bought_is_17 (C S : ℝ) (bought_pens : ℝ) 
  (h1 : bought_pens * C = 12 * S)
  (h2 : 0.4 = (S - C) / C) :
  bought_pens = 17 :=
sorry

end pens_bought_is_17_l1914_191427


namespace decagon_area_bisection_ratio_l1914_191453

theorem decagon_area_bisection_ratio
  (decagon_area : ℝ := 12)
  (below_PQ_area : ℝ := 6)
  (trapezoid_area : ℝ := 4)
  (b1 : ℝ := 3)
  (b2 : ℝ := 6)
  (h : ℝ := 8/9)
  (XQ : ℝ := 4)
  (QY : ℝ := 2) :
  (XQ / QY = 2) :=
by
  sorry

end decagon_area_bisection_ratio_l1914_191453


namespace units_digit_l1914_191463

noncomputable def C := 20 + Real.sqrt 153
noncomputable def D := 20 - Real.sqrt 153

theorem units_digit (h : ∀ n ≥ 1, 20 ^ n % 10 = 0) :
  (C ^ 12 + D ^ 12) % 10 = 0 :=
by
  -- Proof will be provided based on the outlined solution
  sorry

end units_digit_l1914_191463


namespace parallelogram_area_288_l1914_191482

/-- A statement of the area of a given parallelogram -/
theorem parallelogram_area_288 
  (AB BC : ℝ)
  (hAB : AB = 24)
  (hBC : BC = 30)
  (height_from_A_to_DC : ℝ)
  (h_height : height_from_A_to_DC = 12)
  (is_parallelogram : true) :
  AB * height_from_A_to_DC = 288 :=
by
  -- We are focusing only on stating the theorem; the proof is not required.
  sorry

end parallelogram_area_288_l1914_191482


namespace gcd_8885_4514_5246_l1914_191439

theorem gcd_8885_4514_5246 : Nat.gcd (Nat.gcd 8885 4514) 5246 = 1 :=
sorry

end gcd_8885_4514_5246_l1914_191439


namespace find_certain_number_l1914_191462

theorem find_certain_number (x y : ℝ)
  (h1 : (28 + x + 42 + y + 104) / 5 = 62)
  (h2 : (48 + 62 + 98 + 124 + x) / 5 = 78) :
  y = 78 :=
by
  sorry

end find_certain_number_l1914_191462


namespace multiplication_example_l1914_191483

theorem multiplication_example : 28 * (9 + 2 - 5) * 3 = 504 := by 
  sorry

end multiplication_example_l1914_191483


namespace decimal_equiv_of_fraction_l1914_191499

theorem decimal_equiv_of_fraction : (1 / 5) ^ 2 = 0.04 := by
  sorry

end decimal_equiv_of_fraction_l1914_191499


namespace square_of_rational_l1914_191449

theorem square_of_rational (b : ℚ) : b^2 = b * b :=
sorry

end square_of_rational_l1914_191449


namespace solve_quadratic_eq_solve_cubic_eq_l1914_191404

-- Statement for the first equation
theorem solve_quadratic_eq (x : ℝ) : 9 * x^2 - 25 = 0 ↔ x = 5 / 3 ∨ x = -5 / 3 :=
by sorry

-- Statement for the second equation
theorem solve_cubic_eq (x : ℝ) : (x + 1)^3 - 27 = 0 ↔ x = 2 :=
by sorry

end solve_quadratic_eq_solve_cubic_eq_l1914_191404


namespace total_oranges_in_stack_l1914_191410

-- Definitions based on the given conditions
def base_layer_oranges : Nat := 5 * 8
def second_layer_oranges : Nat := 4 * 7
def third_layer_oranges : Nat := 3 * 6
def fourth_layer_oranges : Nat := 2 * 5
def fifth_layer_oranges : Nat := 1 * 4

-- Theorem statement equivalent to the math problem
theorem total_oranges_in_stack : base_layer_oranges + second_layer_oranges + third_layer_oranges + fourth_layer_oranges + fifth_layer_oranges = 100 :=
by
  sorry

end total_oranges_in_stack_l1914_191410


namespace no_perpendicular_hatching_other_than_cube_l1914_191461

def is_convex_polyhedron (P : Polyhedron) : Prop :=
  -- Definition of a convex polyhedron
  sorry

def number_of_faces (P : Polyhedron) : ℕ :=
  -- Function returning the number of faces of polyhedron P
  sorry

def hatching_perpendicular (P : Polyhedron) : Prop :=
  -- Definition that checks if the hatching on adjacent faces of P is perpendicular
  sorry

theorem no_perpendicular_hatching_other_than_cube :
  ∀ (P : Polyhedron), is_convex_polyhedron P ∧ number_of_faces P ≠ 6 → ¬hatching_perpendicular P :=
by
  sorry

end no_perpendicular_hatching_other_than_cube_l1914_191461


namespace smallest_angle_in_convex_polygon_l1914_191452

theorem smallest_angle_in_convex_polygon :
  ∀ (n : ℕ) (angles : ℕ → ℕ) (d : ℕ), n = 25 → (∀ i, 1 ≤ i ∧ i ≤ n → angles i = 166 - 1 * (13 - i)) 
  → 1 ≤ d ∧ d ≤ 1 → (angles 1 = 154) := 
by
  sorry

end smallest_angle_in_convex_polygon_l1914_191452


namespace find_y_value_l1914_191438

theorem find_y_value (a y : ℕ) (h1 : (15^2) * y^3 / 256 = a) (h2 : a = 450) : y = 8 := 
by 
  sorry

end find_y_value_l1914_191438


namespace symmetric_points_a_minus_b_l1914_191411

theorem symmetric_points_a_minus_b (a b : ℝ) 
  (h1 : a = -5) 
  (h2 : b = -1) :
  a - b = -4 := 
sorry

end symmetric_points_a_minus_b_l1914_191411


namespace decimal_multiplication_l1914_191443

theorem decimal_multiplication (h : 268 * 74 = 19832) : 2.68 * 0.74 = 1.9832 :=
by sorry

end decimal_multiplication_l1914_191443


namespace probability_red_or_white_l1914_191446

-- Define the total number of marbles and the counts of blue and red marbles.
def total_marbles : Nat := 60
def blue_marbles : Nat := 5
def red_marbles : Nat := 9

-- Define the remainder to calculate white marbles.
def white_marbles : Nat := total_marbles - (blue_marbles + red_marbles)

-- Lean proof statement to show the probability of selecting a red or white marble.
theorem probability_red_or_white :
  (red_marbles + white_marbles) / total_marbles = 11 / 12 :=
by
  sorry

end probability_red_or_white_l1914_191446


namespace arithmetic_sequence_sum_l1914_191444

variable {a : ℕ → ℝ}

-- Definition of an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d

-- Definition of the fourth term condition
def a4_condition (a : ℕ → ℝ) : Prop :=
  a 4 = 2 - a 3

-- Definition of the sum of the first 6 terms
def sum_first_six_terms (a : ℕ → ℝ) : ℝ :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6

-- Proof statement
theorem arithmetic_sequence_sum :
  is_arithmetic_sequence a →
  a4_condition a →
  sum_first_six_terms a = 6 :=
by
  sorry

end arithmetic_sequence_sum_l1914_191444


namespace factor_expression_l1914_191488

variable {R : Type*} [CommRing R]

theorem factor_expression (a b c : R) :
    a^4 * (b^2 - c^2) + b^4 * (c^2 - a^2) + c^4 * (a^2 - b^2) =
    (a - b) * (b - c) * (c - a) * ((a + b) * a^2 * b^2 + (b + c) * b^2 * c^2 + (a + c) * c^2 * a) :=
sorry

end factor_expression_l1914_191488


namespace balloon_arrangements_l1914_191471

theorem balloon_arrangements : 
  let n := 7
  let k1 := 2
  let k2 := 2
  (Nat.factorial n) / (Nat.factorial k1 * Nat.factorial k2) = 1260 := 
by
  let n := 7
  let k1 := 2
  let k2 := 2
  sorry

end balloon_arrangements_l1914_191471


namespace number_of_hardbacks_l1914_191402

theorem number_of_hardbacks (H P : ℕ) (books total_books selections : ℕ) (comb : ℕ → ℕ → ℕ) :
  total_books = 8 →
  P = 2 →
  comb total_books 3 - comb H 3 = 36 →
  H = 6 :=
by sorry

end number_of_hardbacks_l1914_191402


namespace maximum_obtuse_dihedral_angles_l1914_191414

-- condition: define what a tetrahedron is and its properties
structure Tetrahedron :=
  (edges : Fin 6 → ℝ)   -- represents the 6 edges
  (dihedral_angles : Fin 6 → ℝ) -- represents the 6 dihedral angles

-- Define obtuse angle in degrees
def is_obtuse (angle : ℝ) : Prop := angle > 90 ∧ angle < 180

-- Theorem statement
theorem maximum_obtuse_dihedral_angles (T : Tetrahedron) : 
  (∃ count : ℕ, count = 3 ∧ (∀ i, is_obtuse (T.dihedral_angles i) → count <= 3)) := sorry

end maximum_obtuse_dihedral_angles_l1914_191414


namespace direction_vectors_of_line_l1914_191406

theorem direction_vectors_of_line : 
  ∃ v : ℝ × ℝ, (3 * v.1 - 4 * v.2 = 0) ∧ (v = (1, 3/4) ∨ v = (4, 3)) :=
by
  sorry

end direction_vectors_of_line_l1914_191406


namespace acquaintances_unique_l1914_191412

theorem acquaintances_unique (N : ℕ) : ∃ acquaintances : ℕ → ℕ, 
  (∀ i j k : ℕ, i < N → j < N → k < N → i ≠ j → j ≠ k → i ≠ k → 
    acquaintances i ≠ acquaintances j ∨ acquaintances j ≠ acquaintances k ∨ acquaintances i ≠ acquaintances k) :=
sorry

end acquaintances_unique_l1914_191412


namespace gcd_gx_x_eq_one_l1914_191409

   variable (x : ℤ)
   variable (hx : ∃ k : ℤ, x = 34567 * k)

   def g (x : ℤ) : ℤ := (3 * x + 4) * (8 * x + 3) * (15 * x + 11) * (x + 15)

   theorem gcd_gx_x_eq_one : Int.gcd (g x) x = 1 :=
   by 
     sorry
   
end gcd_gx_x_eq_one_l1914_191409


namespace blocks_used_l1914_191401

theorem blocks_used (initial_blocks used_blocks : ℕ) (h_initial : initial_blocks = 78) (h_left : initial_blocks - used_blocks = 59) : used_blocks = 19 := by
  sorry

end blocks_used_l1914_191401


namespace value_of_abc_l1914_191469

-- Conditions
def cond1 (a b : ℤ) : Prop := ∀ x : ℤ, x^2 + 19 * x + 88 = (x + a) * (x + b)
def cond2 (b c : ℤ) : Prop := ∀ x : ℤ, x^2 - 23 * x + 132 = (x - b) * (x - c)

-- Theorem statement
theorem value_of_abc (a b c : ℤ) (h₁ : cond1 a b) (h₂ : cond2 b c) : a + b + c = 31 :=
sorry

end value_of_abc_l1914_191469


namespace latin_student_sophomore_probability_l1914_191487

variable (F S J SE : ℕ) -- freshmen, sophomores, juniors, seniors total
variable (FL SL JL SEL : ℕ) -- freshmen, sophomores, juniors, seniors taking latin
variable (p : ℚ) -- probability fraction
variable (m n : ℕ) -- relatively prime integers

-- Let the total number of students be 100 for simplicity in percentage calculations
-- Let us encode the given conditions
def conditions := 
  F = 40 ∧ 
  S = 30 ∧ 
  J = 20 ∧ 
  SE = 10 ∧ 
  FL = 40 ∧ 
  SL = S * 80 / 100 ∧ 
  JL = J * 50 / 100 ∧ 
  SEL = SE * 20 / 100

-- The probability calculation
def probability_sophomore (SL : ℕ) (FL SL JL SEL : ℕ) : ℚ := SL / (FL + SL + JL + SEL)

-- Target probability as a rational number
def target_probability := (6 : ℚ) / 19

theorem latin_student_sophomore_probability : 
  conditions F S J SE FL SL JL SEL → 
  probability_sophomore SL FL SL JL SEL = target_probability ∧ 
  m + n = 25 := 
by 
  sorry

end latin_student_sophomore_probability_l1914_191487


namespace decorate_eggs_time_calculation_l1914_191474

/-- Definition of Mia's and Billy's egg decorating rates, total number of eggs to be decorated, and the calculated time when working together --/
def MiaRate : ℕ := 24
def BillyRate : ℕ := 10
def totalEggs : ℕ := 170
def combinedRate : ℕ := MiaRate + BillyRate

theorem decorate_eggs_time_calculation :
  (totalEggs / combinedRate) = 5 := by
  sorry

end decorate_eggs_time_calculation_l1914_191474


namespace third_year_increment_l1914_191465

-- Define the conditions
def total_payments : ℕ := 96
def first_year_cost : ℕ := 20
def second_year_cost : ℕ := first_year_cost + 2
def third_year_cost (x : ℕ) : ℕ := second_year_cost + x
def fourth_year_cost (x : ℕ) : ℕ := third_year_cost x + 4

-- The main proof statement
theorem third_year_increment (x : ℕ) 
  (H : first_year_cost + second_year_cost + third_year_cost x + fourth_year_cost x = total_payments) :
  x = 2 :=
sorry

end third_year_increment_l1914_191465


namespace expand_polynomial_l1914_191495

theorem expand_polynomial :
  (3 * x ^ 2 - 4 * x + 3) * (-2 * x ^ 2 + 3 * x - 4) = -6 * x ^ 4 + 17 * x ^ 3 - 30 * x ^ 2 + 25 * x - 12 :=
by
  sorry

end expand_polynomial_l1914_191495


namespace ab_is_zero_l1914_191426

theorem ab_is_zero (a b : ℝ) (h₁ : a + b = 5) (h₂ : a^3 + b^3 = 125) : a * b = 0 :=
by
  -- Begin proof here
  sorry

end ab_is_zero_l1914_191426


namespace divides_if_not_divisible_by_4_l1914_191416

theorem divides_if_not_divisible_by_4 (n : ℕ) :
  (¬ (4 ∣ n)) → (5 ∣ (1^n + 2^n + 3^n + 4^n)) :=
by sorry

end divides_if_not_divisible_by_4_l1914_191416


namespace lemons_needed_l1914_191468

theorem lemons_needed (lemons32 : ℕ) (lemons4 : ℕ) (h1 : lemons32 = 24) (h2 : (24 : ℕ) / 32 = (lemons4 : ℕ) / 4) : lemons4 = 3 := 
sorry

end lemons_needed_l1914_191468


namespace max_tickets_jane_can_buy_l1914_191451

def ticket_price : ℝ := 15.75
def processing_fee : ℝ := 1.25
def jane_money : ℝ := 150.00

theorem max_tickets_jane_can_buy : ⌊jane_money / (ticket_price + processing_fee)⌋ = 8 := 
by
  sorry

end max_tickets_jane_can_buy_l1914_191451


namespace probability_of_specific_sequence_l1914_191455

def probFirstDiamond : ℚ := 13 / 52
def probSecondSpadeGivenFirstDiamond : ℚ := 13 / 51
def probThirdHeartGivenDiamondSpade : ℚ := 13 / 50

def combinedProbability : ℚ :=
  probFirstDiamond * probSecondSpadeGivenFirstDiamond * probThirdHeartGivenDiamondSpade

theorem probability_of_specific_sequence :
  combinedProbability = 2197 / 132600 := by
  sorry

end probability_of_specific_sequence_l1914_191455


namespace students_no_A_l1914_191480

theorem students_no_A
  (total_students : ℕ)
  (A_in_history : ℕ)
  (A_in_math : ℕ)
  (A_in_science : ℕ)
  (A_in_history_and_math : ℕ)
  (A_in_history_and_science : ℕ)
  (A_in_math_and_science : ℕ)
  (A_in_all_three : ℕ)
  (h_total_students : total_students = 40)
  (h_A_in_history : A_in_history = 10)
  (h_A_in_math : A_in_math = 15)
  (h_A_in_science : A_in_science = 8)
  (h_A_in_history_and_math : A_in_history_and_math = 5)
  (h_A_in_history_and_science : A_in_history_and_science = 3)
  (h_A_in_math_and_science : A_in_math_and_science = 4)
  (h_A_in_all_three : A_in_all_three = 2) :
  total_students - (A_in_history + A_in_math + A_in_science 
    - A_in_history_and_math - A_in_history_and_science - A_in_math_and_science 
    + A_in_all_three) = 17 := 
sorry

end students_no_A_l1914_191480


namespace triple_nested_application_l1914_191489

def g (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 1 else 2 * n + 3

theorem triple_nested_application : g (g (g 3)) = 49 := by
  sorry

end triple_nested_application_l1914_191489


namespace sequence_value_G_50_l1914_191430

theorem sequence_value_G_50 :
  ∀ G : ℕ → ℚ, (∀ n : ℕ, G (n + 1) = (3 * G n + 1) / 3) ∧ G 1 = 3 → G 50 = 152 / 3 :=
by
  intros
  sorry

end sequence_value_G_50_l1914_191430


namespace right_triangle_geo_seq_ratio_l1914_191470

theorem right_triangle_geo_seq_ratio (l r : ℝ) (ht : 0 < l)
  (hr : 1 < r) (hgeo : l^2 + (l * r)^2 = (l * r^2)^2) :
  (l * r^2) / l = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end right_triangle_geo_seq_ratio_l1914_191470


namespace min_value_c_l1914_191432

-- Define the problem using Lean
theorem min_value_c 
    (a b c d e : ℕ)
    (h1 : a + 1 = b) 
    (h2 : b + 1 = c)
    (h3 : c + 1 = d)
    (h4 : d + 1 = e)
    (h5 : ∃ n : ℕ, 5 * c = n ^ 3)
    (h6 : ∃ m : ℕ, 3 * c = m ^ 2) : 
    c = 675 := 
sorry

end min_value_c_l1914_191432


namespace Mark_owes_total_l1914_191467

noncomputable def base_fine : ℕ := 50

def additional_fine (speed_over_limit : ℕ) : ℕ :=
  let first_10 := min speed_over_limit 10 * 2
  let next_5 := min (speed_over_limit - 10) 5 * 3
  let next_10 := min (speed_over_limit - 15) 10 * 5
  let remaining := max (speed_over_limit - 25) 0 * 6
  first_10 + next_5 + next_10 + remaining

noncomputable def total_fine (base : ℕ) (additional : ℕ) (school_zone : Bool) : ℕ :=
  let fine := base + additional
  if school_zone then fine * 2 else fine

def court_costs : ℕ := 350

noncomputable def processing_fee (fine : ℕ) : ℕ := fine / 10

def lawyer_fees (hourly_rate : ℕ) (hours : ℕ) : ℕ := hourly_rate * hours

theorem Mark_owes_total :
  let speed_over_limit := 45
  let base := base_fine
  let additional := additional_fine speed_over_limit
  let school_zone := true
  let fine := total_fine base additional school_zone
  let total_fine_with_costs := fine + court_costs
  let processing := processing_fee total_fine_with_costs
  let lawyer := lawyer_fees 100 4
  let total := total_fine_with_costs + processing + lawyer
  total = 1346 := sorry

end Mark_owes_total_l1914_191467


namespace number_of_mappings_A_to_B_number_of_mappings_B_to_A_l1914_191423

theorem number_of_mappings_A_to_B (A B : Finset ℕ) (hA : A.card = 5) (hB : B.card = 4) :
  (B.card ^ A.card) = 4^5 :=
by sorry

theorem number_of_mappings_B_to_A (A B : Finset ℕ) (hA : A.card = 5) (hB : B.card = 4) :
  (A.card ^ B.card) = 5^4 :=
by sorry

end number_of_mappings_A_to_B_number_of_mappings_B_to_A_l1914_191423


namespace angle_degrees_l1914_191466

-- Define the conditions
def sides_parallel (θ₁ θ₂ : ℝ) : Prop :=
  θ₁ = θ₂ ∨ (θ₁ + θ₂ = 180)

def angle_relation (θ₁ θ₂ : ℝ) : Prop :=
  θ₁ = 3 * θ₂ - 20 ∨ θ₂ = 3 * θ₁ - 20

-- Statement of the problem
theorem angle_degrees (θ₁ θ₂ : ℝ) (h_parallel : sides_parallel θ₁ θ₂) (h_relation : angle_relation θ₁ θ₂) :
  (θ₁ = 10 ∧ θ₂ = 10) ∨ (θ₁ = 50 ∧ θ₂ = 130) ∨ (θ₁ = 130 ∧ θ₂ = 50) ∨ θ₁ + θ₂ = 180 ∧ (θ₁ = 3 * θ₂ - 20 ∨ θ₂ = 3 * θ₁ - 20) :=
by sorry

end angle_degrees_l1914_191466


namespace find_length_of_MN_l1914_191435

theorem find_length_of_MN (A B C M N : ℝ × ℝ)
  (AB AC : ℝ) (M_midpoint : M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (N_midpoint : N = ((A.1 + C.1) / 2, (A.2 + C.2) / 2))
  (length_AB : abs (B.1 - A.1) + abs (B.2 - A.2) = 15)
  (length_AC : abs (C.1 - A.1) + abs (C.2 - A.2) = 20) :
  abs (N.1 - M.1) + abs (N.2 - M.2) = 40 / 3 := sorry

end find_length_of_MN_l1914_191435


namespace quadratic_equation_solution_l1914_191433

-- We want to prove that for the conditions given, the only possible value of m is 3
theorem quadratic_equation_solution (m : ℤ) (h1 : m^2 - 7 = 2) (h2 : m + 3 ≠ 0) : m = 3 :=
sorry

end quadratic_equation_solution_l1914_191433


namespace pie_contest_l1914_191447

def first_student_pie := 7 / 6
def second_student_pie := 4 / 3
def third_student_eats_from_first := 1 / 2
def third_student_eats_from_second := 1 / 3

theorem pie_contest :
  (first_student_pie - third_student_eats_from_first = 2 / 3) ∧
  (second_student_pie - third_student_eats_from_second = 1) ∧
  (third_student_eats_from_first + third_student_eats_from_second = 5 / 6) :=
by
  sorry

end pie_contest_l1914_191447


namespace discount_double_time_l1914_191450

theorem discount_double_time (TD FV : ℝ) (h1 : TD = 10) (h2 : FV = 110) : 
  2 * TD = 20 :=
by
  sorry

end discount_double_time_l1914_191450


namespace rubber_duck_charity_fundraiser_l1914_191420

noncomputable def charity_raised (price_small price_medium price_large : ℕ) 
(bulk_discount_threshold_small bulk_discount_threshold_medium bulk_discount_threshold_large : ℕ)
(bulk_discount_rate_small bulk_discount_rate_medium bulk_discount_rate_large : ℝ)
(tax_rate_small tax_rate_medium tax_rate_large : ℝ)
(sold_small sold_medium sold_large : ℕ) : ℝ :=
  let cost_small := price_small * sold_small
  let cost_medium := price_medium * sold_medium
  let cost_large := price_large * sold_large

  let discount_small := if sold_small >= bulk_discount_threshold_small then 
                          (bulk_discount_rate_small * cost_small) else 0
  let discount_medium := if sold_medium >= bulk_discount_threshold_medium then 
                          (bulk_discount_rate_medium * cost_medium) else 0
  let discount_large := if sold_large >= bulk_discount_threshold_large then 
                          (bulk_discount_rate_large * cost_large) else 0

  let after_discount_small := cost_small - discount_small
  let after_discount_medium := cost_medium - discount_medium
  let after_discount_large := cost_large - discount_large

  let tax_small := tax_rate_small * after_discount_small
  let tax_medium := tax_rate_medium * after_discount_medium
  let tax_large := tax_rate_large * after_discount_large

  let total_small := after_discount_small + tax_small
  let total_medium := after_discount_medium + tax_medium
  let total_large := after_discount_large + tax_large

  total_small + total_medium + total_large

theorem rubber_duck_charity_fundraiser :
  charity_raised 2 3 5 10 15 20 0.1 0.15 0.2
  0.05 0.07 0.09 150 221 185 = 1693.10 :=
by 
  -- implementation of math corresponding to problem's solution
  sorry

end rubber_duck_charity_fundraiser_l1914_191420


namespace minimum_disks_needed_l1914_191493

theorem minimum_disks_needed :
  ∀ (n_files : ℕ) (disk_space : ℝ) (mb_files_1 : ℕ) (size_file_1 : ℝ) (mb_files_2 : ℕ) (size_file_2 : ℝ) (remaining_files : ℕ) (size_remaining_files : ℝ),
    n_files = 30 →
    disk_space = 1.5 →
    mb_files_1 = 4 →
    size_file_1 = 1.0 →
    mb_files_2 = 10 →
    size_file_2 = 0.6 →
    remaining_files = 16 →
    size_remaining_files = 0.5 →
    ∃ (min_disks : ℕ), min_disks = 13 :=
by
  sorry

end minimum_disks_needed_l1914_191493


namespace leaves_problem_l1914_191407

noncomputable def leaves_dropped_last_day (L : ℕ) (n : ℕ) : ℕ :=
  L - n * (L / 10)

theorem leaves_problem (L : ℕ) (n : ℕ) (h1 : L = 340) (h2 : leaves_dropped_last_day L n = 204) :
  n = 4 :=
by {
  sorry
}

end leaves_problem_l1914_191407


namespace people_sharing_pizzas_l1914_191405

-- Definitions based on conditions
def number_of_pizzas : ℝ := 21.0
def slices_per_person : ℝ := 2.625
def slices_per_pizza : ℝ := 8.0

-- Theorem to prove the number of people
theorem people_sharing_pizzas : (number_of_pizzas * slices_per_pizza) / slices_per_person = 64 :=
by
  sorry

end people_sharing_pizzas_l1914_191405


namespace claudia_has_three_25_cent_coins_l1914_191408

def number_of_coins (x y z : ℕ) := x + y + z = 15
def number_of_combinations (x y : ℕ) := 4 * x + 3 * y = 51

theorem claudia_has_three_25_cent_coins (x y z : ℕ) 
  (h1: number_of_coins x y z) 
  (h2: number_of_combinations x y): 
  z = 3 := 
by 
sorry

end claudia_has_three_25_cent_coins_l1914_191408


namespace henry_income_percent_increase_l1914_191429

theorem henry_income_percent_increase :
  let original_income : ℝ := 120
  let new_income : ℝ := 180
  let increase := new_income - original_income
  let percent_increase := (increase / original_income) * 100
  percent_increase = 50 :=
by
  sorry

end henry_income_percent_increase_l1914_191429


namespace quadratic_congruence_solution_l1914_191418

theorem quadratic_congruence_solution (p : ℕ) (hp : Nat.Prime p) : 
  ∃ n : ℕ, 6 * n^2 + 5 * n + 1 ≡ 0 [MOD p] := 
sorry

end quadratic_congruence_solution_l1914_191418


namespace find_side_b_l1914_191492

-- Given the side and angle conditions in the triangle
variable (A B C : ℝ)
variable (a b c : ℝ)
variable (S : ℝ) 

-- Conditions provided in the problem
axiom side_a (h : a = 1) : True
axiom angle_B (h : B = Real.pi / 4) : True  -- 45 degrees in radians
axiom area_triangle (h : S = 2) : True

-- Final proof statement
theorem find_side_b (h₁ : a = 1) (h₂ : B = Real.pi / 4) (h₃ : S = 2) : 
  b = 5 := sorry

end find_side_b_l1914_191492


namespace find_biology_marks_l1914_191441

theorem find_biology_marks (english math physics chemistry : ℕ) (avg_marks : ℕ) (biology : ℕ)
  (h_english : english = 86) (h_math : math = 89) (h_physics : physics = 82)
  (h_chemistry : chemistry = 87) (h_avg_marks : avg_marks = 85) :
  (english + math + physics + chemistry + biology) = avg_marks * 5 →
  biology = 81 :=
by
  sorry

end find_biology_marks_l1914_191441


namespace common_chord_eq_l1914_191478

theorem common_chord_eq (x y : ℝ) :
  x^2 + y^2 + 2*x = 0 →
  x^2 + y^2 - 4*y = 0 →
  x + 2*y = 0 :=
by
  intros h1 h2
  sorry

end common_chord_eq_l1914_191478


namespace dynamic_load_L_value_l1914_191494

theorem dynamic_load_L_value (T H : ℝ) (hT : T = 3) (hH : H = 6) : 
  (L : ℝ) = (50 * T^3) / (H^3) -> L = 6.25 := 
by 
  sorry 

end dynamic_load_L_value_l1914_191494


namespace ratio_sum_of_square_lengths_equals_68_l1914_191491

theorem ratio_sum_of_square_lengths_equals_68 (a b c : ℕ) 
  (h1 : (∃ (r : ℝ), r = 50 / 98) → a = 5 ∧ b = 14 ∧ c = 49) :
  a + b + c = 68 :=
by
  sorry -- Proof is not required

end ratio_sum_of_square_lengths_equals_68_l1914_191491


namespace triangle_angle_A_l1914_191464

theorem triangle_angle_A (A B a b : ℝ) (h1 : b = 2 * a) (h2 : B = A + 60) : A = 30 :=
by sorry

end triangle_angle_A_l1914_191464


namespace num_solutions_eq_40_l1914_191458

theorem num_solutions_eq_40 : 
  ∀ (n : ℕ), 
  (∃ seq : ℕ → ℕ, seq 1 = 4 ∧ (∀ k : ℕ, 1 ≤ k → seq (k + 1) = seq k + 4) ∧ seq 10 = 40) :=
by
  sorry

end num_solutions_eq_40_l1914_191458


namespace smallest_w_l1914_191403

theorem smallest_w (w : ℕ) (h1 : Nat.gcd 1452 w = 1) (h2 : 2 ∣ w ∧ 3 ∣ w ∧ 13 ∣ w) :
  (∃ (w : ℕ), 2^4 ∣ 1452 * w ∧ 3^3 ∣ 1452 * w ∧ 13^3 ∣ 1452 * w ∧ w > 0) ∧
  ∀ (w' : ℕ), (2^4 ∣ 1452 * w' ∧ 3^3 ∣ 1452 * w' ∧ 13^3 ∣ 1452 * w' ∧ w' > 0) → w ≤ w' :=
  sorry

end smallest_w_l1914_191403


namespace find_y_l1914_191496

theorem find_y {x y : ℝ} (hx : (8 : ℝ) = (1/4 : ℝ) * x) (hy : (y : ℝ) = (1/4 : ℝ) * (20 : ℝ)) (hprod : x * y = 160) : y = 5 :=
by {
  sorry
}

end find_y_l1914_191496


namespace unique_solution_condition_l1914_191497

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + c = d * x + 4) ↔ d ≠ 4 :=
by
  sorry

end unique_solution_condition_l1914_191497


namespace oldest_sister_clothing_l1914_191473

-- Define the initial conditions
def Nicole_initial := 10
def First_sister := Nicole_initial / 2
def Next_sister := Nicole_initial + 2
def Nicole_end := 36

-- Define the proof statement
theorem oldest_sister_clothing : 
    (First_sister + Next_sister + Nicole_initial + x = Nicole_end) → x = 9 :=
by
  sorry

end oldest_sister_clothing_l1914_191473


namespace expand_product_correct_l1914_191485

noncomputable def expand_product (x : ℝ) : ℝ :=
  (3 / 7) * (7 / x^2 + 6 * x^3 - 2)

theorem expand_product_correct (x : ℝ) (h : x ≠ 0) :
  expand_product x = (3 / x^2) + (18 * x^3 / 7) - (6 / 7) := by
  unfold expand_product
  -- The proof will go here
  sorry

end expand_product_correct_l1914_191485


namespace min_sum_ab_max_product_ab_l1914_191481

theorem min_sum_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) : a + b ≥ 2 :=
by
  sorry

theorem max_product_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : a * b ≤ 1 / 4 :=
by
  sorry

end min_sum_ab_max_product_ab_l1914_191481


namespace sum_of_series_l1914_191434

open BigOperators

-- Define the sequence a(n) = 2 / (n * (n + 3))
def a (n : ℕ) : ℚ := 2 / (n * (n + 3))

-- Prove the sum of the first 20 terms of sequence a equals 10 / 9.
theorem sum_of_series : (∑ n in Finset.range 20, a (n + 1)) = 10 / 9 := by
  sorry

end sum_of_series_l1914_191434


namespace root_ratios_equal_l1914_191475

theorem root_ratios_equal (a : ℝ) (ha : 0 < a)
  (hroots : ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁^3 + 1 = a * x₁ ∧ x₂^3 + 1 = a * x₂ ∧ x₂ / x₁ = 2018) :
  ∃ y₁ y₂ : ℝ, 0 < y₁ ∧ 0 < y₂ ∧ y₁^3 + 1 = a * y₁^2 ∧ y₂^3 + 1 = a * y₂^2 ∧ y₂ / y₁ = 2018 :=
sorry

end root_ratios_equal_l1914_191475


namespace total_number_of_eggs_l1914_191422

theorem total_number_of_eggs 
  (cartons : ℕ) 
  (eggs_per_carton_length : ℕ) 
  (eggs_per_carton_width : ℕ)
  (egg_position_from_front : ℕ)
  (egg_position_from_back : ℕ)
  (egg_position_from_left : ℕ)
  (egg_position_from_right : ℕ) :
  cartons = 28 →
  egg_position_from_front = 14 →
  egg_position_from_back = 20 →
  egg_position_from_left = 3 →
  egg_position_from_right = 2 →
  eggs_per_carton_length = egg_position_from_front + egg_position_from_back - 1 →
  eggs_per_carton_width = egg_position_from_left + egg_position_from_right - 1 →
  cartons * (eggs_per_carton_length * eggs_per_carton_width) = 3696 := 
  by 
  intros
  sorry

end total_number_of_eggs_l1914_191422


namespace max_n_consecutive_sum_2014_l1914_191472

theorem max_n_consecutive_sum_2014 : 
  ∃ (k n : ℕ), (2 * k + n - 1) * n = 4028 ∧ n = 53 ∧ k > 0 := sorry

end max_n_consecutive_sum_2014_l1914_191472


namespace cinco_de_mayo_day_days_between_feb_14_and_may_5_l1914_191442

theorem cinco_de_mayo_day {
  feb_14_is_tuesday : ∃ n : ℕ, n % 7 = 2
}: 
∃ n : ℕ, n % 7 = 5 := sorry

theorem days_between_feb_14_and_may_5: 
  ∃ d : ℕ, 
  d = 81 := sorry

end cinco_de_mayo_day_days_between_feb_14_and_may_5_l1914_191442


namespace jam_cost_l1914_191425

theorem jam_cost (N B J H : ℕ) (h1 : N > 1) (h2 : N * (3 * B + 6 * J + 2 * H) = 342) :
  6 * N * J = 270 := 
sorry

end jam_cost_l1914_191425


namespace difference_of_numbers_l1914_191457

-- Definitions for the digits and the numbers formed
def digits : List ℕ := [5, 3, 1, 4]

def largestNumber : ℕ := 5431
def leastNumber : ℕ := 1345

-- The problem statement
theorem difference_of_numbers (digits : List ℕ) (n_largest n_least : ℕ) :
  n_largest = 5431 ∧ n_least = 1345 → (n_largest - n_least) = 4086 :=
by
  sorry

end difference_of_numbers_l1914_191457


namespace find_y_l1914_191413

theorem find_y (x y : ℝ) : x - y = 8 ∧ x + y = 14 → y = 3 := by
  sorry

end find_y_l1914_191413


namespace intercept_sum_l1914_191440

theorem intercept_sum (x0 y0 : ℕ) (h1 : x0 < 17) (h2 : y0 < 17)
  (hx : 7 * x0 ≡ 2 [MOD 17]) (hy : 3 * y0 ≡ 15 [MOD 17]) : x0 + y0 = 17 :=
sorry

end intercept_sum_l1914_191440


namespace purely_periodic_period_le_T_l1914_191448

theorem purely_periodic_period_le_T {a b : ℚ} (T : ℕ) 
  (ha : ∃ m, a = m / (10^T - 1)) 
  (hb : ∃ n, b = n / (10^T - 1)) :
  (∃ T₁, T₁ ≤ T ∧ ∃ p, a = p / (10^T₁ - 1)) ∧ 
  (∃ T₂, T₂ ≤ T ∧ ∃ q, b = q / (10^T₂ - 1)) := 
sorry

end purely_periodic_period_le_T_l1914_191448


namespace octahedron_has_eulerian_circuit_cube_has_no_eulerian_circuit_l1914_191456

-- Part (a) - Octahedron
/- 
A connected graph representing an octahedron. 
Each vertex has a degree of 4, making the graph Eulerian.
-/
theorem octahedron_has_eulerian_circuit : 
  ∃ circuit : List (ℕ × ℕ), 
    (∀ (u v : ℕ), List.elem (u, v) circuit ↔ List.elem (v, u) circuit) ∧
    (∃ start, ∀ v ∈ circuit, v = start) :=
sorry

-- Part (b) - Cube
/- 
A connected graph representing a cube.
Each vertex has a degree of 3, making it impossible for the graph to be Eulerian.
-/
theorem cube_has_no_eulerian_circuit : 
  ¬ ∃ (circuit : List (ℕ × ℕ)), 
    (∀ (u v : ℕ), List.elem (u, v) circuit ↔ List.elem (v, u) circuit) ∧
    (∃ start, ∀ v ∈ circuit, v = start) :=
sorry

end octahedron_has_eulerian_circuit_cube_has_no_eulerian_circuit_l1914_191456


namespace range_of_a_l1914_191484

variable {a : ℝ}

def A := Set.Ioo (-1 : ℝ) 1
def B (a : ℝ) := Set.Ioo a (a + 1)

theorem range_of_a :
  B a ⊆ A ↔ (-1 : ℝ) ≤ a ∧ a ≤ 0 :=
by
  sorry

end range_of_a_l1914_191484


namespace necessary_and_sufficient_l1914_191486

theorem necessary_and_sufficient (a b : ℝ) : a > b ↔ a * |a| > b * |b| := sorry

end necessary_and_sufficient_l1914_191486


namespace ellipse_equation_is_standard_form_l1914_191424

theorem ellipse_equation_is_standard_form (m n : ℝ) (h_m_pos : m > 0) (h_n_pos : n > 0) (h_mn_neq : m ≠ n) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧ (∀ x y : ℝ, mx^2 + ny^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

end ellipse_equation_is_standard_form_l1914_191424


namespace area_proof_l1914_191490

-- Define the problem conditions
variables (l w : ℕ)
def length_is_four_times_width : Prop := l = 4 * w
def perimeter_is_200 : Prop := 2 * l + 2 * w = 200

-- Define the target to prove
def area_of_rectangle : Prop := (l * w = 1600)


-- Lean 4 statement to prove the area given the conditions
theorem area_proof (h1 : length_is_four_times_width l w) (h2 : perimeter_is_200 l w) : area_of_rectangle l w := 
  sorry

end area_proof_l1914_191490


namespace complex_purely_imaginary_l1914_191498

theorem complex_purely_imaginary (a : ℂ) (h1 : a^2 - 3 * a + 2 = 0) (h2 : a - 1 ≠ 0) : a = 2 :=
sorry

end complex_purely_imaginary_l1914_191498


namespace negation_equiv_exists_l1914_191437

theorem negation_equiv_exists : 
  ¬ (∀ x : ℝ, x^2 + 1 > 0) ↔ ∃ x_0 : ℝ, x_0^2 + 1 ≤ 0 := 
by 
  sorry

end negation_equiv_exists_l1914_191437


namespace least_positive_multiple_24_gt_450_l1914_191476

theorem least_positive_multiple_24_gt_450 : ∃ n : ℕ, n > 450 ∧ n % 24 = 0 ∧ n = 456 :=
by
  use 456
  sorry

end least_positive_multiple_24_gt_450_l1914_191476


namespace fraction_subtraction_simplified_l1914_191436

theorem fraction_subtraction_simplified : (8 / 19 - 5 / 57) = (1 / 3) := by
  sorry

end fraction_subtraction_simplified_l1914_191436


namespace right_triangle_leg_length_l1914_191454

theorem right_triangle_leg_length
  (A : ℝ)
  (b h : ℝ)
  (hA : A = 800)
  (hb : b = 40)
  (h_area : A = (1 / 2) * b * h) :
  h = 40 :=
by
  sorry

end right_triangle_leg_length_l1914_191454
