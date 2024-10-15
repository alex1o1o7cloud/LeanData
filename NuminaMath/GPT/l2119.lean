import Mathlib

namespace NUMINAMATH_GPT_product_of_geometric_terms_l2119_211912

noncomputable def arithmeticSeq (a1 d : ℕ) (n : ℕ) : ℕ :=
  a1 + (n - 1) * d

noncomputable def geometricSeq (b1 r : ℕ) (n : ℕ) : ℕ :=
  b1 * r^(n - 1)

theorem product_of_geometric_terms :
  ∃ (a1 d b1 r : ℕ),
    (3 * a1 - (arithmeticSeq a1 d 8)^2 + 3 * (arithmeticSeq a1 d 15) = 0) ∧ 
    (arithmeticSeq a1 d 8 = geometricSeq b1 r 10) ∧ 
    (geometricSeq b1 r 3 * geometricSeq b1 r 17 = 36) :=
sorry

end NUMINAMATH_GPT_product_of_geometric_terms_l2119_211912


namespace NUMINAMATH_GPT_non_neg_int_solutions_eq_10_l2119_211947

theorem non_neg_int_solutions_eq_10 :
  ∃ n : ℕ, n = 55 ∧
  (∃ (x y z : ℕ), x + y + z = 10) :=
by
  sorry

end NUMINAMATH_GPT_non_neg_int_solutions_eq_10_l2119_211947


namespace NUMINAMATH_GPT_sum_of_decimals_l2119_211993

theorem sum_of_decimals :
  (2 / 100 : ℝ) + (5 / 1000) + (8 / 10000) + (6 / 100000) = 0.02586 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_decimals_l2119_211993


namespace NUMINAMATH_GPT_quadratic_root_difference_l2119_211902

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

noncomputable def root_difference (a b c : ℝ) : ℝ :=
  (Real.sqrt (discriminant a b c)) / a

theorem quadratic_root_difference :
  root_difference (3 + 2 * Real.sqrt 2) (5 + Real.sqrt 2) (-4) = Real.sqrt (177 - 122 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_difference_l2119_211902


namespace NUMINAMATH_GPT_factor_72x3_minus_252x7_l2119_211945

theorem factor_72x3_minus_252x7 (x : ℝ) : (72 * x^3 - 252 * x^7) = (36 * x^3 * (2 - 7 * x^4)) :=
by
  sorry

end NUMINAMATH_GPT_factor_72x3_minus_252x7_l2119_211945


namespace NUMINAMATH_GPT_abs_sub_nonneg_l2119_211903

theorem abs_sub_nonneg (a : ℝ) : |a| - a ≥ 0 :=
sorry

end NUMINAMATH_GPT_abs_sub_nonneg_l2119_211903


namespace NUMINAMATH_GPT_EF_length_proof_l2119_211953

noncomputable def length_BD (AB BC : ℝ) : ℝ := Real.sqrt (AB^2 + BC^2)

noncomputable def length_EF (BD AB BC : ℝ) : ℝ :=
  let BE := BD * AB / BD
  let BF := BD * BC / AB
  BE + BF

theorem EF_length_proof : 
  ∀ (AB BC : ℝ), AB = 4 ∧ BC = 3 →
  length_EF (length_BD AB BC) AB BC = 125 / 12 :=
by
  intros AB BC h
  rw [length_BD, length_EF]
  simp
  rw [Real.sqrt_eq_rpow]
  simp
  sorry

end NUMINAMATH_GPT_EF_length_proof_l2119_211953


namespace NUMINAMATH_GPT_area_of_tangents_l2119_211966

def radius := 3
def segment_length := 6

theorem area_of_tangents (r : ℝ) (l : ℝ) (h1 : r = radius) (h2 : l = segment_length) :
  let R := r * Real.sqrt 2 
  let annulus_area := π * (R ^ 2) - π * (r ^ 2)
  annulus_area = 9 * π :=
by
  sorry

end NUMINAMATH_GPT_area_of_tangents_l2119_211966


namespace NUMINAMATH_GPT_car_speed_ratio_to_pedestrian_speed_l2119_211940

noncomputable def ratio_of_speeds (length_pedestrian_car: ℝ) (length_bridge: ℝ) (speed_pedestrian: ℝ) (speed_car: ℝ) : ℝ :=
  speed_car / speed_pedestrian

theorem car_speed_ratio_to_pedestrian_speed
  (L : ℝ)  -- length of the bridge
  (vc vp : ℝ)  -- speeds of car and pedestrian respectively
  (h1 : (2 / 5) * L + (3 / 5) * vp = L)
  (h2 : (3 / 5) * L = vc * (L / (5 * vp))) :
  ratio_of_speeds L L vp vc = 5 :=
by
  sorry

end NUMINAMATH_GPT_car_speed_ratio_to_pedestrian_speed_l2119_211940


namespace NUMINAMATH_GPT_find_unit_prices_and_evaluate_discount_schemes_l2119_211987

theorem find_unit_prices_and_evaluate_discount_schemes :
  ∃ (x y : ℝ),
    40 * x + 100 * y = 280 ∧
    30 * x + 200 * y = 260 ∧
    x = 6 ∧
    y = 0.4 ∧
    (∀ m : ℝ, m > 200 → 
      (50 * 6 + 0.4 * (m - 50) < 50 * 6 + 0.4 * 200 + 0.4 * 0.8 * (m - 200) ↔ m < 450) ∧
      (50 * 6 + 0.4 * (m - 50) = 50 * 6 + 0.4 * 200 + 0.4 * 0.8 * (m - 200) ↔ m = 450) ∧
      (50 * 6 + 0.4 * (m - 50) > 50 * 6 + 0.4 * 200 + 0.4 * 0.8 * (m - 200) ↔ m > 450)) :=
sorry

end NUMINAMATH_GPT_find_unit_prices_and_evaluate_discount_schemes_l2119_211987


namespace NUMINAMATH_GPT_unique_cell_50_distance_l2119_211905

-- Define the distance between two cells
def kingDistance (p1 p2 : ℤ × ℤ) : ℤ :=
  max (abs (p1.1 - p2.1)) (abs (p1.2 - p2.2))

-- A condition stating three cells with specific distances
variables (A B C : ℤ × ℤ) (hAB : kingDistance A B = 100) (hBC : kingDistance B C = 100) (hCA : kingDistance C A = 100)

-- A proposition to prove there is exactly one cell at a distance of 50 from all three given cells
theorem unique_cell_50_distance : ∃! D : ℤ × ℤ, kingDistance D A = 50 ∧ kingDistance D B = 50 ∧ kingDistance D C = 50 :=
sorry

end NUMINAMATH_GPT_unique_cell_50_distance_l2119_211905


namespace NUMINAMATH_GPT_find_coordinates_of_P_l2119_211965

theorem find_coordinates_of_P : 
  ∃ P: ℝ × ℝ, 
  (∃ θ: ℝ, 0 ≤ θ ∧ θ ≤ π ∧ P = (3 * Real.cos θ, 4 * Real.sin θ)) ∧ 
  ∃ m: ℝ, m = 1 ∧ P.fst = P.snd ∧ P = (12/5, 12/5) :=
by {
  sorry -- Proof is omitted as per instruction
}

end NUMINAMATH_GPT_find_coordinates_of_P_l2119_211965


namespace NUMINAMATH_GPT_problem_a_problem_b_problem_c_l2119_211920

variable (a b : ℝ)

theorem problem_a {a b : ℝ} (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem problem_b {a b : ℝ} (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem problem_c {a b : ℝ} (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

end NUMINAMATH_GPT_problem_a_problem_b_problem_c_l2119_211920


namespace NUMINAMATH_GPT_white_animals_count_l2119_211982

-- Definitions
def total : ℕ := 13
def black : ℕ := 6
def white : ℕ := total - black

-- Theorem stating the number of white animals
theorem white_animals_count : white = 7 :=
by {
  -- The proof would go here, but we'll use sorry to skip it.
  sorry
}

end NUMINAMATH_GPT_white_animals_count_l2119_211982


namespace NUMINAMATH_GPT_quotient_when_divided_by_44_is_3_l2119_211975

/-
A number, when divided by 44, gives a certain quotient and 0 as remainder.
When dividing the same number by 30, the remainder is 18.
Prove that the quotient in the first division is 3.
-/

theorem quotient_when_divided_by_44_is_3 (N : ℕ) (Q : ℕ) (P : ℕ) 
  (h1 : N % 44 = 0)
  (h2 : N % 30 = 18) :
  N = 44 * Q →
  Q = 3 := 
by
  -- since no proof is required, we use sorry
  sorry

end NUMINAMATH_GPT_quotient_when_divided_by_44_is_3_l2119_211975


namespace NUMINAMATH_GPT_defective_pens_l2119_211942

theorem defective_pens :
  ∃ D N : ℕ, (N + D = 9) ∧ (N / 9 * (N - 1) / 8 = 5 / 12) ∧ (D = 3) :=
by
  sorry

end NUMINAMATH_GPT_defective_pens_l2119_211942


namespace NUMINAMATH_GPT_student_correct_answers_l2119_211938

theorem student_correct_answers (c w : ℕ) 
  (h1 : c + w = 60)
  (h2 : 4 * c - w = 120) : 
  c = 36 :=
sorry

end NUMINAMATH_GPT_student_correct_answers_l2119_211938


namespace NUMINAMATH_GPT_discount_percentage_is_correct_l2119_211984

noncomputable def cost_prices := [540, 660, 780]
noncomputable def markup_percentages := [0.15, 0.20, 0.25]
noncomputable def selling_prices := [496.80, 600, 750]

noncomputable def marked_price (cost : ℝ) (markup : ℝ) : ℝ := cost + (markup * cost)

noncomputable def total_marked_price : ℝ := 
  (marked_price 540 0.15) + (marked_price 660 0.20) + (marked_price 780 0.25)

noncomputable def total_selling_price : ℝ := 496.80 + 600 + 750

noncomputable def overall_discount_percentage : ℝ :=
  ((total_marked_price - total_selling_price) / total_marked_price) * 100

theorem discount_percentage_is_correct : overall_discount_percentage = 22.65 :=
by
  sorry

end NUMINAMATH_GPT_discount_percentage_is_correct_l2119_211984


namespace NUMINAMATH_GPT_village_population_l2119_211977

noncomputable def number_of_people_in_village
  (vampire_drains_per_week : ℕ)
  (werewolf_eats_per_week : ℕ)
  (weeks : ℕ) : ℕ :=
  let drained := vampire_drains_per_week * weeks
  let eaten := werewolf_eats_per_week * weeks
  drained + eaten

theorem village_population :
  number_of_people_in_village 3 5 9 = 72 := by
  sorry

end NUMINAMATH_GPT_village_population_l2119_211977


namespace NUMINAMATH_GPT_power_mod_equiv_l2119_211925

theorem power_mod_equiv :
  7 ^ 145 % 12 = 7 % 12 :=
by
  -- Here the solution would go
  sorry

end NUMINAMATH_GPT_power_mod_equiv_l2119_211925


namespace NUMINAMATH_GPT_factor_expression_l2119_211998

theorem factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := 
by 
  sorry

end NUMINAMATH_GPT_factor_expression_l2119_211998


namespace NUMINAMATH_GPT_volume_percentage_correct_l2119_211980

-- Define the initial conditions
def box_length := 8
def box_width := 6
def box_height := 12
def cube_side := 3

-- Calculate the number of cubes along each dimension
def num_cubes_length := box_length / cube_side
def num_cubes_width := box_width / cube_side
def num_cubes_height := box_height / cube_side

-- Calculate volumes
def volume_cube := cube_side ^ 3
def volume_box := box_length * box_width * box_height
def volume_cubes := (num_cubes_length * num_cubes_width * num_cubes_height) * volume_cube

-- Prove the percentage calculation
theorem volume_percentage_correct : (volume_cubes.toFloat / volume_box.toFloat) * 100 = 75 := by
  sorry

end NUMINAMATH_GPT_volume_percentage_correct_l2119_211980


namespace NUMINAMATH_GPT_consecutive_even_sum_l2119_211999

theorem consecutive_even_sum (N S : ℤ) (m : ℤ) 
  (hk : 2 * m + 1 > 0) -- k is the number of consecutive even numbers, which is odd
  (h_sum : (2 * m + 1) * N = S) -- The condition of the sum
  (h_even : N % 2 = 0) -- The middle number is even
  : (∃ k : ℤ, k = 2 * m + 1 ∧ k > 0 ∧ (k * N / 2) = S/2 ) := 
  sorry

end NUMINAMATH_GPT_consecutive_even_sum_l2119_211999


namespace NUMINAMATH_GPT_shanna_tomato_ratio_l2119_211941

-- Define the initial conditions
def initial_tomato_plants : ℕ := 6
def initial_eggplant_plants : ℕ := 2
def initial_pepper_plants : ℕ := 4
def pepper_plants_died : ℕ := 1
def vegetables_per_plant : ℕ := 7
def total_vegetables_harvested : ℕ := 56

-- Define the number of tomato plants that died
def tomato_plants_died (total_vegetables : ℕ) (veg_per_plant : ℕ) (initial_tomato : ℕ) 
  (initial_eggplant : ℕ) (initial_pepper : ℕ) (pepper_died : ℕ) : ℕ :=
  let surviving_plants := total_vegetables / veg_per_plant
  let surviving_pepper := initial_pepper - pepper_died
  let surviving_tomato := surviving_plants - (initial_eggplant + surviving_pepper)
  initial_tomato - surviving_tomato

-- Define the ratio
def ratio_tomato_plants_died_to_initial (tomato_died : ℕ) (initial_tomato : ℕ) : ℚ :=
  (tomato_died : ℚ) / (initial_tomato : ℚ)

theorem shanna_tomato_ratio :
  ratio_tomato_plants_died_to_initial (tomato_plants_died total_vegetables_harvested vegetables_per_plant 
    initial_tomato_plants initial_eggplant_plants initial_pepper_plants pepper_plants_died) initial_tomato_plants 
  = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_shanna_tomato_ratio_l2119_211941


namespace NUMINAMATH_GPT_power_sum_divisible_by_5_l2119_211962

theorem power_sum_divisible_by_5 (n : ℕ) : (2^(4*n + 1) + 3^(4*n + 1)) % 5 = 0 :=
by
  sorry

end NUMINAMATH_GPT_power_sum_divisible_by_5_l2119_211962


namespace NUMINAMATH_GPT_sqrt_sum_ineq_l2119_211910

open Real

theorem sqrt_sum_ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a^2 + b^2 + c^2 = 1) :
  sqrt (1 - a^2) + sqrt (1 - b^2) + sqrt (1 - c^2) + a + b + c > 3 := by
  sorry

end NUMINAMATH_GPT_sqrt_sum_ineq_l2119_211910


namespace NUMINAMATH_GPT_find_b_l2119_211926

-- Define the variables involved
variables (a b : ℝ)

-- Conditions provided in the problem
def condition_1 : Prop := 2 * a + 1 = 1
def condition_2 : Prop := b + a = 3

-- Theorem statement to prove b = 3 given the conditions
theorem find_b (h1 : condition_1 a) (h2 : condition_2 a b) : b = 3 := by
  sorry

end NUMINAMATH_GPT_find_b_l2119_211926


namespace NUMINAMATH_GPT_polynomial_product_expansion_l2119_211946

theorem polynomial_product_expansion (x : ℝ) : (x^2 + 3 * x + 3) * (x^2 - 3 * x + 3) = x^4 - 3 * x^2 + 9 := 
by sorry

end NUMINAMATH_GPT_polynomial_product_expansion_l2119_211946


namespace NUMINAMATH_GPT_area_ADC_calculation_l2119_211963

-- Definitions and assumptions
variables (BD DC : ℝ)
variables (area_ABD area_ADC : ℝ)

-- Given conditions
axiom ratio_BD_DC : BD / DC = 2 / 5
axiom area_ABD_given : area_ABD = 40

-- The theorem to prove
theorem area_ADC_calculation (h1 : BD / DC = 2 / 5) (h2 : area_ABD = 40) :
  area_ADC = 100 :=
sorry

end NUMINAMATH_GPT_area_ADC_calculation_l2119_211963


namespace NUMINAMATH_GPT_consecutive_odd_integers_sum_l2119_211988

theorem consecutive_odd_integers_sum (x : ℤ) (h : x + (x + 4) = 134) : x + (x + 2) + (x + 4) = 201 := 
by sorry

end NUMINAMATH_GPT_consecutive_odd_integers_sum_l2119_211988


namespace NUMINAMATH_GPT_sum_of_first_ten_terms_l2119_211906

theorem sum_of_first_ten_terms (a : ℕ → ℝ)
  (h1 : a 3 ^ 2 + a 8 ^ 2 + 2 * a 3 * a 8 = 9)
  (h2 : ∀ n, a n < 0) :
  (5 * (a 3 + a 8) = -15) :=
sorry

end NUMINAMATH_GPT_sum_of_first_ten_terms_l2119_211906


namespace NUMINAMATH_GPT_range_of_reciprocals_l2119_211935

theorem range_of_reciprocals (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h_neq : a ≠ b) (h_sum : a + b = 1) :
  4 < (1 / a + 1 / b) :=
sorry

end NUMINAMATH_GPT_range_of_reciprocals_l2119_211935


namespace NUMINAMATH_GPT_no_closed_loop_after_replacement_l2119_211944

theorem no_closed_loop_after_replacement (N M : ℕ) 
  (h1 : N = M) 
  (h2 : (N + M) % 4 = 0) :
  ¬((N - 1) - (M + 1)) % 4 = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_closed_loop_after_replacement_l2119_211944


namespace NUMINAMATH_GPT_Jo_has_least_l2119_211908

variable (Money : Type) 
variable (Bo Coe Flo Jo Moe Zoe : Money)
variable [LT Money] [LE Money] -- Money type is an ordered type with less than and less than or equal relations.

-- Conditions
axiom h1 : Jo < Flo 
axiom h2 : Flo < Bo
axiom h3 : Jo < Moe
axiom h4 : Moe < Bo
axiom h5 : Bo < Coe
axiom h6 : Coe < Zoe

-- The main statement to prove that Jo has the least money.
theorem Jo_has_least (h1 : Jo < Flo) (h2 : Flo < Bo) (h3 : Jo < Moe) (h4 : Moe < Bo) (h5 : Bo < Coe) (h6 : Coe < Zoe) : 
  ∀ x, x = Jo ∨ x = Bo ∨ x = Flo ∨ x = Moe ∨ x = Coe ∨ x = Zoe → Jo ≤ x :=
by
  -- Proof is skipped using sorry
  sorry

end NUMINAMATH_GPT_Jo_has_least_l2119_211908


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l2119_211978

-- Define the data for the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := (y - 1)^2 / 16 - (x + 2)^2 / 25 = 1

-- Define the two equations for the asymptotes
def asymptote1 (x y : ℝ) : Prop := y = 4 / 5 * x + 13 / 5
def asymptote2 (x y : ℝ) : Prop := y = -4 / 5 * x + 13 / 5

-- Theorem stating that the given asymptotes are correct for the hyperbola
theorem hyperbola_asymptotes : 
  (∀ x y : ℝ, hyperbola_eq x y → (asymptote1 x y ∨ asymptote2 x y)) := 
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l2119_211978


namespace NUMINAMATH_GPT_previous_painting_price_l2119_211979

-- Define the amount received for the most recent painting
def recentPainting (p : ℕ) := 5 * p - 1000

-- Define the target amount
def target := 44000

-- State that the target amount is achieved by the prescribed function
theorem previous_painting_price : recentPainting 9000 = target :=
by
  sorry

end NUMINAMATH_GPT_previous_painting_price_l2119_211979


namespace NUMINAMATH_GPT_height_previous_year_l2119_211932

theorem height_previous_year (current_height : ℝ) (growth_rate : ℝ) (previous_height : ℝ) 
  (h1 : current_height = 126)
  (h2 : growth_rate = 0.05) 
  (h3 : current_height = 1.05 * previous_height) : 
  previous_height = 120 :=
sorry

end NUMINAMATH_GPT_height_previous_year_l2119_211932


namespace NUMINAMATH_GPT_school_population_proof_l2119_211911

variables (x y z: ℕ)
variable (B: ℕ := (50 * y) / 100)

theorem school_population_proof (h1 : 162 = (x * B) / 100)
                               (h2 : B = (50 * y) / 100)
                               (h3 : z = 100 - 50) :
  z = 50 :=
  sorry

end NUMINAMATH_GPT_school_population_proof_l2119_211911


namespace NUMINAMATH_GPT_cost_per_candy_bar_l2119_211904

-- Define the conditions as hypotheses
variables (candy_bars_total : ℕ) (candy_bars_paid_by_dave : ℕ) (amount_paid_by_john : ℝ)
-- Assume the given values
axiom total_candy_bars : candy_bars_total = 20
axiom candy_bars_by_dave : candy_bars_paid_by_dave = 6
axiom paid_by_john : amount_paid_by_john = 21

-- Define the proof problem
theorem cost_per_candy_bar :
  (amount_paid_by_john / (candy_bars_total - candy_bars_paid_by_dave) = 1.50) :=
by
  sorry

end NUMINAMATH_GPT_cost_per_candy_bar_l2119_211904


namespace NUMINAMATH_GPT_increase_factor_is_46_8_l2119_211985

-- Definitions for the conditions
def old_plates : ℕ := 26^3 * 10^3
def new_plates_type_A : ℕ := 26^2 * 10^4
def new_plates_type_B : ℕ := 26^4 * 10^2
def average_new_plates := (new_plates_type_A + new_plates_type_B) / 2

-- The Lean 4 statement to prove that the increase factor is 46.8
theorem increase_factor_is_46_8 :
  (average_new_plates : ℚ) / (old_plates : ℚ) = 46.8 := by
  sorry

end NUMINAMATH_GPT_increase_factor_is_46_8_l2119_211985


namespace NUMINAMATH_GPT_total_children_correct_l2119_211964

def blocks : ℕ := 9
def children_per_block : ℕ := 6
def total_children : ℕ := blocks * children_per_block

theorem total_children_correct : total_children = 54 := by
  sorry

end NUMINAMATH_GPT_total_children_correct_l2119_211964


namespace NUMINAMATH_GPT_inverse_function_domain_l2119_211915

noncomputable def f (x : ℝ) : ℝ := -3 + Real.log (x - 1) / Real.log 2

theorem inverse_function_domain :
  ∀ x : ℝ, x ≥ 5 → ∃ y : ℝ, f x = y ∧ y ≥ -1 :=
by
  intro x hx
  use f x
  sorry

end NUMINAMATH_GPT_inverse_function_domain_l2119_211915


namespace NUMINAMATH_GPT_afternoon_more_than_evening_l2119_211943

def campers_in_morning : Nat := 33
def campers_in_afternoon : Nat := 34
def campers_in_evening : Nat := 10

theorem afternoon_more_than_evening : campers_in_afternoon - campers_in_evening = 24 := by
  sorry

end NUMINAMATH_GPT_afternoon_more_than_evening_l2119_211943


namespace NUMINAMATH_GPT_cannot_cut_square_into_7_rectangles_l2119_211930

theorem cannot_cut_square_into_7_rectangles (a : ℝ) :
  ¬ ∃ (x : ℝ), 7 * 2 * x ^ 2 = a ^ 2 ∧ 
    ∀ (i : ℕ), 0 ≤ i → i < 7 → (∃ (rect : ℝ × ℝ), rect.1 = x ∧ rect.2 = 2 * x ) :=
by
  sorry

end NUMINAMATH_GPT_cannot_cut_square_into_7_rectangles_l2119_211930


namespace NUMINAMATH_GPT_water_channel_area_l2119_211917

-- Define the given conditions
def top_width := 14
def bottom_width := 8
def depth := 70

-- The area formula for a trapezium given the top width, bottom width, and height
def trapezium_area (a b h : ℕ) : ℕ :=
  (a + b) * h / 2

-- The main theorem stating the area of the trapezium
theorem water_channel_area : 
  trapezium_area top_width bottom_width depth = 770 := by
  -- Proof can be completed here
  sorry

end NUMINAMATH_GPT_water_channel_area_l2119_211917


namespace NUMINAMATH_GPT_inv_203_mod_301_exists_l2119_211971

theorem inv_203_mod_301_exists : ∃ b : ℤ, 203 * b % 301 = 1 := sorry

end NUMINAMATH_GPT_inv_203_mod_301_exists_l2119_211971


namespace NUMINAMATH_GPT_time_to_reach_6400ft_is_200min_l2119_211936

noncomputable def time_to_reach_ship (depth : ℕ) (rate : ℕ) : ℕ :=
  depth / rate

theorem time_to_reach_6400ft_is_200min :
  time_to_reach_ship 6400 32 = 200 := by
  sorry

end NUMINAMATH_GPT_time_to_reach_6400ft_is_200min_l2119_211936


namespace NUMINAMATH_GPT_buffaloes_number_l2119_211934

theorem buffaloes_number (B D : ℕ) 
  (h : 4 * B + 2 * D = 2 * (B + D) + 24) : 
  B = 12 :=
sorry

end NUMINAMATH_GPT_buffaloes_number_l2119_211934


namespace NUMINAMATH_GPT_smallest_positive_shift_l2119_211981

noncomputable def g : ℝ → ℝ := sorry

theorem smallest_positive_shift
  (H1 : ∀ x, g (x - 20) = g x) : 
  ∃ a > 0, (∀ x, g ((x - a) / 10) = g (x / 10)) ∧ a = 200 :=
sorry

end NUMINAMATH_GPT_smallest_positive_shift_l2119_211981


namespace NUMINAMATH_GPT_pens_each_student_gets_now_l2119_211928

-- Define conditions
def red_pens_per_student := 62
def black_pens_per_student := 43
def num_students := 3
def pens_taken_first_month := 37
def pens_taken_second_month := 41

-- Define total pens bought and remaining pens after each month
def total_pens := num_students * (red_pens_per_student + black_pens_per_student)
def remaining_pens_after_first_month := total_pens - pens_taken_first_month
def remaining_pens_after_second_month := remaining_pens_after_first_month - pens_taken_second_month

-- Theorem statement
theorem pens_each_student_gets_now :
  (remaining_pens_after_second_month / num_students) = 79 :=
by
  sorry

end NUMINAMATH_GPT_pens_each_student_gets_now_l2119_211928


namespace NUMINAMATH_GPT_sample_size_is_five_l2119_211900

def population := 100
def sample (n : ℕ) := n ≤ population
def sample_size (n : ℕ) := n

theorem sample_size_is_five (n : ℕ) (h : sample 5) : sample_size 5 = 5 :=
by
  sorry

end NUMINAMATH_GPT_sample_size_is_five_l2119_211900


namespace NUMINAMATH_GPT_water_left_in_bucket_l2119_211969

theorem water_left_in_bucket (initial_amount poured_amount : ℝ) (h1 : initial_amount = 0.8) (h2 : poured_amount = 0.2) : initial_amount - poured_amount = 0.6 := by
  sorry

end NUMINAMATH_GPT_water_left_in_bucket_l2119_211969


namespace NUMINAMATH_GPT_train_speed_in_kph_l2119_211991

noncomputable def speed_of_train (jogger_speed_kph : ℝ) (gap_m : ℝ) (train_length_m : ℝ) (time_s : ℝ) : ℝ :=
let jogger_speed_mps := jogger_speed_kph * (1000 / 3600)
let total_distance_m := gap_m + train_length_m
let speed_mps := total_distance_m / time_s
speed_mps * (3600 / 1000)

theorem train_speed_in_kph :
  speed_of_train 9 240 120 36 = 36 := 
by
  sorry

end NUMINAMATH_GPT_train_speed_in_kph_l2119_211991


namespace NUMINAMATH_GPT_tan_315_eq_neg1_l2119_211989

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by 
  sorry

end NUMINAMATH_GPT_tan_315_eq_neg1_l2119_211989


namespace NUMINAMATH_GPT_four_pow_2024_mod_11_l2119_211901

theorem four_pow_2024_mod_11 : (4 ^ 2024) % 11 = 3 :=
by
  sorry

end NUMINAMATH_GPT_four_pow_2024_mod_11_l2119_211901


namespace NUMINAMATH_GPT_sum_arithmetic_sequence_min_value_l2119_211983

theorem sum_arithmetic_sequence_min_value (a d : ℤ) 
  (S : ℕ → ℤ) 
  (H1 : S 8 ≤ 6) 
  (H2 : S 11 ≥ 27)
  (H_Sn : ∀ n, S n = n * a + (n * (n - 1) / 2) * d) : 
  S 19 ≥ 133 :=
by
  sorry

end NUMINAMATH_GPT_sum_arithmetic_sequence_min_value_l2119_211983


namespace NUMINAMATH_GPT_working_capacity_ratio_l2119_211939

theorem working_capacity_ratio (team_p_engineers : ℕ) (team_q_engineers : ℕ) (team_p_days : ℕ) (team_q_days : ℕ) :
  team_p_engineers = 20 → team_q_engineers = 16 → team_p_days = 32 → team_q_days = 30 →
  (team_p_days / team_q_days) = (16:ℤ) / (15:ℤ) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_working_capacity_ratio_l2119_211939


namespace NUMINAMATH_GPT_find_range_t_l2119_211924

def sequence_increasing (n : ℕ) (t : ℝ) : Prop :=
  (2 * (n + 1) + t^2 - 8) / (n + 1 + t) > (2 * n + t^2 - 8) / (n + t)

theorem find_range_t (t : ℝ) (h : ∀ n : ℕ, sequence_increasing n t) : 
  -1 < t ∧ t < 4 :=
sorry

end NUMINAMATH_GPT_find_range_t_l2119_211924


namespace NUMINAMATH_GPT_Michael_needs_more_money_l2119_211922

def money_Michael_has : ℕ := 50
def cake_cost : ℕ := 20
def bouquet_cost : ℕ := 36
def balloons_cost : ℕ := 5

def total_cost : ℕ := cake_cost + bouquet_cost + balloons_cost
def money_needed : ℕ := total_cost - money_Michael_has

theorem Michael_needs_more_money : money_needed = 11 :=
by
  sorry

end NUMINAMATH_GPT_Michael_needs_more_money_l2119_211922


namespace NUMINAMATH_GPT_sum_of_a5_a6_l2119_211970

variable (a : ℕ → ℕ)

def S (n : ℕ) : ℕ :=
  n ^ 2 + 2

theorem sum_of_a5_a6 :
  a 5 + a 6 = S 6 - S 4 := by
  sorry

end NUMINAMATH_GPT_sum_of_a5_a6_l2119_211970


namespace NUMINAMATH_GPT_jill_total_phone_time_l2119_211916

def phone_time : ℕ → ℕ
| 0 => 5
| (n + 1) => 2 * phone_time n

theorem jill_total_phone_time (n : ℕ) (h : n = 4) : 
  phone_time 0 + phone_time 1 + phone_time 2 + phone_time 3 + phone_time 4 = 155 :=
by
  cases h
  sorry

end NUMINAMATH_GPT_jill_total_phone_time_l2119_211916


namespace NUMINAMATH_GPT_limit_of_sequence_l2119_211927

theorem limit_of_sequence {ε : ℝ} (hε : ε > 0) : 
  ∃ (N : ℝ), ∀ (n : ℝ), n > N → |(2 * n^3) / (n^3 - 2) - 2| < ε :=
by
  sorry

end NUMINAMATH_GPT_limit_of_sequence_l2119_211927


namespace NUMINAMATH_GPT_combined_market_value_two_years_later_l2119_211951

theorem combined_market_value_two_years_later:
  let P_A := 8000
  let P_B := 10000
  let P_C := 12000
  let r_A := 0.20
  let r_B := 0.15
  let r_C := 0.10

  let V_A_year_1 := P_A - r_A * P_A
  let V_A_year_2 := V_A_year_1 - r_A * P_A
  let V_B_year_1 := P_B - r_B * P_B
  let V_B_year_2 := V_B_year_1 - r_B * P_B
  let V_C_year_1 := P_C - r_C * P_C
  let V_C_year_2 := V_C_year_1 - r_C * P_C

  V_A_year_2 + V_B_year_2 + V_C_year_2 = 21400 :=
by
  sorry

end NUMINAMATH_GPT_combined_market_value_two_years_later_l2119_211951


namespace NUMINAMATH_GPT_perpendicular_lines_l2119_211954

theorem perpendicular_lines (a : ℝ) :
  (if a ≠ 0 then a^2 ≠ 0 else true) ∧ (a^2 * a + (-1/a) * 2 = -1) → (a = 2 ∨ a = 0) :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_l2119_211954


namespace NUMINAMATH_GPT_project_assignment_l2119_211950

open Nat

def binom (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem project_assignment :
  let A := 3
  let B := 1
  let C := 2
  let D := 2
  let total_projects := 8
  A + B + C + D = total_projects →
  (binom 8 3) * (binom 5 1) * (binom 4 2) * (binom 2 2) = 1680 :=
by
  intros
  sorry

end NUMINAMATH_GPT_project_assignment_l2119_211950


namespace NUMINAMATH_GPT_greatest_integer_less_than_PS_l2119_211973

theorem greatest_integer_less_than_PS :
  ∀ (PQ PS : ℝ), PQ = 150 → 
  PS = 150 * Real.sqrt 3 → 
  (⌊PS⌋ = 259) := 
by
  intros PQ PS hPQ hPS
  sorry

end NUMINAMATH_GPT_greatest_integer_less_than_PS_l2119_211973


namespace NUMINAMATH_GPT_craig_total_distance_l2119_211918

-- Define the distances Craig walked
def dist_school_to_david : ℝ := 0.27
def dist_david_to_home : ℝ := 0.73

-- Prove the total distance walked
theorem craig_total_distance : dist_school_to_david + dist_david_to_home = 1.00 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_craig_total_distance_l2119_211918


namespace NUMINAMATH_GPT_contrapositive_equivalence_l2119_211974

theorem contrapositive_equivalence (P Q : Prop) : (P → Q) ↔ (¬ Q → ¬ P) :=
by sorry

end NUMINAMATH_GPT_contrapositive_equivalence_l2119_211974


namespace NUMINAMATH_GPT_matrix_B6_eq_sB_plus_tI_l2119_211921

noncomputable section

open Matrix

def B : Matrix (Fin 2) (Fin 2) ℤ :=
  !![1, -1;
     4, 2]

theorem matrix_B6_eq_sB_plus_tI :
  ∃ s t : ℤ, B^6 = s • B + t • (1 : Matrix (Fin 2) (Fin 2) ℤ) := by
  have B2_eq : B^2 = -3 • B :=
    -- Matrix multiplication and scalar multiplication
    sorry
  use 81, 0
  have B4_eq : B^4 = 9 • B^2 := by
    rw [B2_eq]
    -- Calculation steps for B^4 equation
    sorry
  have B6_eq : B^6 = B^4 * B^2 := by
    rw [B4_eq, B2_eq]
    -- Calculation steps for B^6 final equation
    sorry
  rw [B6_eq]
  -- Final steps to show (81 • B + 0 • I = 81 • B)
  sorry

end NUMINAMATH_GPT_matrix_B6_eq_sB_plus_tI_l2119_211921


namespace NUMINAMATH_GPT_max_distance_line_l2119_211990

noncomputable def equation_of_line (x y : ℝ) : ℝ := x + 2 * y - 5

theorem max_distance_line (x y : ℝ) : 
  equation_of_line 1 2 = 0 ∧ 
  (∀ (a b c : ℝ), c ≠ 0 → (x = 1 ∧ y = 2 → equation_of_line x y = 0)) ∧ 
  (∀ (L : ℝ → ℝ → ℝ), L 1 2 = 0 → (L = equation_of_line)) :=
sorry

end NUMINAMATH_GPT_max_distance_line_l2119_211990


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l2119_211995

theorem arithmetic_sequence_problem
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : a 1 + 3 * a 8 + a 15 = 120)
  : 2 * a 9 - a 10 = 24 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l2119_211995


namespace NUMINAMATH_GPT_gcd_g_150_151_l2119_211907

def g (x : ℤ) : ℤ := x^2 - 2*x + 3020

theorem gcd_g_150_151 : Int.gcd (g 150) (g 151) = 1 :=
  by
  sorry

end NUMINAMATH_GPT_gcd_g_150_151_l2119_211907


namespace NUMINAMATH_GPT_remainder_sum_mod_11_l2119_211949

theorem remainder_sum_mod_11 :
  (72501 + 72502 + 72503 + 72504 + 72505 + 72506 + 72507 + 72508 + 72509 + 72510) % 11 = 5 :=
by
  sorry

end NUMINAMATH_GPT_remainder_sum_mod_11_l2119_211949


namespace NUMINAMATH_GPT_production_cost_decrease_l2119_211957

theorem production_cost_decrease (x : ℝ) :
  let initial_production_cost := 50
  let initial_selling_price := 65
  let first_quarter_decrease := 0.10
  let second_quarter_increase := 0.05
  let final_selling_price := initial_selling_price * (1 - first_quarter_decrease) * (1 + second_quarter_increase)
  let original_profit := initial_selling_price - initial_production_cost
  let final_production_cost := initial_production_cost * (1 - x) ^ 2
  (final_selling_price - final_production_cost) = original_profit :=
by
  sorry

end NUMINAMATH_GPT_production_cost_decrease_l2119_211957


namespace NUMINAMATH_GPT_B_equals_1_2_3_l2119_211929

def A : Set ℝ := { x | x^2 ≤ 4 }
def B : Set ℕ := { x | x > 0 ∧ (x - 1:ℝ) ∈ A }

theorem B_equals_1_2_3 : B = {1, 2, 3} :=
by
  sorry

end NUMINAMATH_GPT_B_equals_1_2_3_l2119_211929


namespace NUMINAMATH_GPT_avg_score_false_iff_unequal_ints_l2119_211972

variable {a b m n : ℕ}

theorem avg_score_false_iff_unequal_ints 
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (m_pos : 0 < m) 
  (n_pos : 0 < n) 
  (m_neq_n : m ≠ n) : 
  (∃ a b, (ma + nb) / (m + n) = (a + b)/2) ↔ a ≠ b := 
sorry

end NUMINAMATH_GPT_avg_score_false_iff_unequal_ints_l2119_211972


namespace NUMINAMATH_GPT_smallest_integer_value_l2119_211986

theorem smallest_integer_value (y : ℤ) (h : 7 - 3 * y < -8) : y ≥ 6 :=
sorry

end NUMINAMATH_GPT_smallest_integer_value_l2119_211986


namespace NUMINAMATH_GPT_field_trip_students_l2119_211952

theorem field_trip_students 
  (seats_per_bus : ℕ) 
  (buses_needed : ℕ) 
  (total_students : ℕ) 
  (h1 : seats_per_bus = 2) 
  (h2 : buses_needed = 7) 
  (h3 : total_students = seats_per_bus * buses_needed) : 
  total_students = 14 :=
by 
  rw [h1, h2] at h3
  assumption

end NUMINAMATH_GPT_field_trip_students_l2119_211952


namespace NUMINAMATH_GPT_y_coord_intersection_with_y_axis_l2119_211955

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 + 11

-- Define the point P
def P : ℝ × ℝ := (1, curve 1)

-- Define the derivative of the curve
def derivative (x : ℝ) : ℝ := 3 * x^2

-- Define the tangent line at point P (1, 12)
def tangent_line (x : ℝ) : ℝ := 3 * (x - 1) + 12

-- Proof statement
theorem y_coord_intersection_with_y_axis : 
  tangent_line 0 = 9 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_y_coord_intersection_with_y_axis_l2119_211955


namespace NUMINAMATH_GPT_min_reciprocal_sum_l2119_211967

theorem min_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy_sum : x + y = 12) (hxy_neq : x ≠ y) : 
  ∃ c : ℝ, c = 1 / 3 ∧ (1 / x + 1 / y ≥ c) :=
sorry

end NUMINAMATH_GPT_min_reciprocal_sum_l2119_211967


namespace NUMINAMATH_GPT_integer_pairs_satisfying_equation_l2119_211961

theorem integer_pairs_satisfying_equation:
  ∀ (a b : ℕ), a ≥ 1 → b ≥ 1 → a^(b^2) = b^a ↔ (a, b) = (1, 1) ∨ (a, b) = (16, 2) ∨ (a, b) = (27, 3) :=
by
  sorry

end NUMINAMATH_GPT_integer_pairs_satisfying_equation_l2119_211961


namespace NUMINAMATH_GPT_greatest_y_l2119_211914

theorem greatest_y (x y : ℤ) (h : x * y + 3 * x + 2 * y = -9) : y ≤ -2 :=
by {
  sorry
}

end NUMINAMATH_GPT_greatest_y_l2119_211914


namespace NUMINAMATH_GPT_inequality_am_gm_cauchy_schwarz_equality_iff_l2119_211948

theorem inequality_am_gm_cauchy_schwarz 
  (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) :=
sorry

theorem equality_iff (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^2 + b^2 + c^2 + d^2)^2 = (a + b) * (b + c) * (c + d) * (d + a) 
  ↔ a = b ∧ b = c ∧ c = d :=
sorry

end NUMINAMATH_GPT_inequality_am_gm_cauchy_schwarz_equality_iff_l2119_211948


namespace NUMINAMATH_GPT_problem_statement_l2119_211919

variables {A B x y a : ℝ}

theorem problem_statement (h1 : 1/A = 1 - (1 - x) / y)
                          (h2 : 1/B = 1 - y / (1 - x))
                          (h3 : x = (1 - a) / (1 - 1/a))
                          (h4 : y = 1 - 1/x)
                          (h5 : a ≠ 1) (h6 : a ≠ -1) : 
                          A + B = 1 :=
sorry

end NUMINAMATH_GPT_problem_statement_l2119_211919


namespace NUMINAMATH_GPT_multiplication_trick_l2119_211933

theorem multiplication_trick (a b c : ℕ) (h : b + c = 10) :
  (10 * a + b) * (10 * a + c) = 100 * a * (a + 1) + b * c :=
by
  sorry

end NUMINAMATH_GPT_multiplication_trick_l2119_211933


namespace NUMINAMATH_GPT_min_jugs_needed_to_fill_container_l2119_211960

def min_jugs_to_fill (jug_capacity container_capacity : ℕ) : ℕ :=
  Nat.ceil (container_capacity / jug_capacity)

theorem min_jugs_needed_to_fill_container :
  min_jugs_to_fill 16 200 = 13 :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_min_jugs_needed_to_fill_container_l2119_211960


namespace NUMINAMATH_GPT_time_for_b_l2119_211923

theorem time_for_b (A B C : ℚ) (H1 : A + B + C = 1/4) (H2 : A = 1/12) (H3 : C = 1/18) : B = 1/9 :=
by {
  sorry
}

end NUMINAMATH_GPT_time_for_b_l2119_211923


namespace NUMINAMATH_GPT_y_pow_x_eq_x_pow_y_l2119_211956

open Real

noncomputable def x (n : ℕ) : ℝ := (1 + 1 / n) ^ n
noncomputable def y (n : ℕ) : ℝ := (1 + 1 / n) ^ (n + 1)

theorem y_pow_x_eq_x_pow_y (n : ℕ) (hn : 0 < n) : (y n) ^ (x n) = (x n) ^ (y n) :=
by
  sorry

end NUMINAMATH_GPT_y_pow_x_eq_x_pow_y_l2119_211956


namespace NUMINAMATH_GPT_find_sequence_index_l2119_211976

theorem find_sequence_index (a : ℕ → ℕ) 
  (h₁ : a 1 = 1) 
  (h₂ : ∀ n, a (n + 1) - 3 = a n)
  (h₃ : ∃ n, a n = 2023) : ∃ n, a n = 2023 ∧ n = 675 := 
by 
  sorry

end NUMINAMATH_GPT_find_sequence_index_l2119_211976


namespace NUMINAMATH_GPT_john_text_messages_per_day_l2119_211909

theorem john_text_messages_per_day (m n : ℕ) (h1 : m = 20) (h2 : n = 245) : 
  m + n / 7 = 55 :=
by
  sorry

end NUMINAMATH_GPT_john_text_messages_per_day_l2119_211909


namespace NUMINAMATH_GPT_sum_of_areas_of_two_squares_l2119_211958

theorem sum_of_areas_of_two_squares (a b : ℕ) (h1 : a = 8) (h2 : b = 10) :
  a * a + b * b = 164 := by
  sorry

end NUMINAMATH_GPT_sum_of_areas_of_two_squares_l2119_211958


namespace NUMINAMATH_GPT_exists_nat_not_in_geom_progressions_l2119_211996

theorem exists_nat_not_in_geom_progressions
  (progressions : Fin 5 → ℕ → ℕ)
  (is_geometric : ∀ i : Fin 5, ∃ a q : ℕ, ∀ n : ℕ, progressions i n = a * q^n) :
  ∃ n : ℕ, ∀ i : Fin 5, ∀ m : ℕ, progressions i m ≠ n :=
by
  sorry

end NUMINAMATH_GPT_exists_nat_not_in_geom_progressions_l2119_211996


namespace NUMINAMATH_GPT_starting_number_is_100_l2119_211937

theorem starting_number_is_100 (n : ℕ) (h1 : 100 ≤ n ∧ n ≤ 1000) (h2 : ∃ k : ℕ, k = 10 ∧ n = 1000 - (k - 1) * 100) :
  n = 100 := by
  sorry

end NUMINAMATH_GPT_starting_number_is_100_l2119_211937


namespace NUMINAMATH_GPT_value_of_expression_l2119_211997

def expression (x y z : ℤ) : ℤ :=
  x^2 + y^2 - z^2 + 2 * x * y + x * y * z

theorem value_of_expression (x y z : ℤ) (h1 : x = 2) (h2 : y = -3) (h3 : z = 1) : 
  expression x y z = -7 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2119_211997


namespace NUMINAMATH_GPT_k_value_function_range_l2119_211931

noncomputable def f : ℝ → ℝ := λ x => Real.log x + x

def is_k_value_function (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∃ (a b : ℝ), a < b ∧ (∀ x, a ≤ x ∧ x ≤ b → (f x = k * x)) ∧ (k > 0)

theorem k_value_function_range :
  ∀ (f : ℝ → ℝ), (∀ x, f x = Real.log x + x) →
  (∃ (k : ℝ), is_k_value_function f k) →
  1 < k ∧ k < 1 + (1 / Real.exp 1) :=
by
  sorry

end NUMINAMATH_GPT_k_value_function_range_l2119_211931


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l2119_211992

-- Define the sets M and N
def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | x^2 - 2*x < 0}

-- The proof statement
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | 1 < x ∧ x < 2} := 
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l2119_211992


namespace NUMINAMATH_GPT_cost_to_paint_floor_l2119_211968

-- Define the conditions
def length_more_than_breadth_by_200_percent (L B : ℝ) : Prop :=
L = 3 * B

def length_of_floor := 23
def cost_per_sq_meter := 3

-- Prove the cost to paint the floor
theorem cost_to_paint_floor (B : ℝ) (L : ℝ) 
    (h1: length_more_than_breadth_by_200_percent L B) (h2: L = length_of_floor) 
    (rate: ℝ) (h3: rate = cost_per_sq_meter) :
    rate * (L * B) = 529.23 :=
by
  -- intermediate steps would go here
  sorry

end NUMINAMATH_GPT_cost_to_paint_floor_l2119_211968


namespace NUMINAMATH_GPT_maximum_value_expression_l2119_211913

noncomputable def expression (s t : ℝ) := -2 * s^2 + 24 * s + 3 * t - 38

theorem maximum_value_expression : ∀ (s : ℝ), expression s 4 ≤ 46 :=
by sorry

end NUMINAMATH_GPT_maximum_value_expression_l2119_211913


namespace NUMINAMATH_GPT_sequence_problem_l2119_211959

-- Given sequence
variable (P Q R S T U V : ℤ)

-- Given conditions
variable (hR : R = 7)
variable (hPQ : P + Q + R = 21)
variable (hQS : Q + R + S = 21)
variable (hST : R + S + T = 21)
variable (hTU : S + T + U = 21)
variable (hUV : T + U + V = 21)

theorem sequence_problem : P + V = 14 := by
  sorry

end NUMINAMATH_GPT_sequence_problem_l2119_211959


namespace NUMINAMATH_GPT_number_of_passed_candidates_l2119_211994

variables (P F : ℕ) (h1 : P + F = 100)
          (h2 : P * 70 + F * 20 = 100 * 50)
          (h3 : ∀ p, p = P → 70 * p = 70 * P)
          (h4 : ∀ f, f = F → 20 * f = 20 * F)

theorem number_of_passed_candidates (P F : ℕ) (h1 : P + F = 100) 
                                    (h2 : P * 70 + F * 20 = 100 * 50) 
                                    (h3 : ∀ p, p = P → 70 * p = 70 * P) 
                                    (h4 : ∀ f, f = F → 20 * f = 20 * F) : 
  P = 60 :=
sorry

end NUMINAMATH_GPT_number_of_passed_candidates_l2119_211994
