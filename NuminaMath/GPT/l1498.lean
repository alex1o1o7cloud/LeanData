import Mathlib

namespace NUMINAMATH_GPT_complement_intersection_eq_l1498_149868

open Set

def P : Set ℝ := { x | x^2 - 2 * x ≥ 0 }
def Q : Set ℝ := { x | 1 < x ∧ x ≤ 2 }

theorem complement_intersection_eq :
  (compl P) ∩ Q = { x : ℝ | 1 < x ∧ x < 2 } :=
sorry

end NUMINAMATH_GPT_complement_intersection_eq_l1498_149868


namespace NUMINAMATH_GPT_simplify_expr_l1498_149896

theorem simplify_expr : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expr_l1498_149896


namespace NUMINAMATH_GPT_f_2015_2016_l1498_149830

noncomputable def f : ℤ → ℤ := sorry

theorem f_2015_2016 (h1 : ∀ x, f (-x) = -f x) (h2 : ∀ x, f (x + 2) = -f x) (h3 : f 1 = 2) :
  f 2015 + f 2016 = -2 :=
sorry

end NUMINAMATH_GPT_f_2015_2016_l1498_149830


namespace NUMINAMATH_GPT_maximize_perimeter_OIH_l1498_149854

/-- In triangle ABC, given certain angles and side lengths, prove that
    angle ABC = 70° maximizes the perimeter of triangle OIH, where O, I,
    and H are the circumcenter, incenter, and orthocenter of triangle ABC. -/
theorem maximize_perimeter_OIH 
  (A : ℝ) (B : ℝ) (C : ℝ)
  (BC : ℝ) (AB : ℝ) (AC : ℝ)
  (BOC : ℝ) (BIC : ℝ) (BHC : ℝ) :
  A = 75 ∧ BC = 2 ∧ AB ≥ AC ∧
  BOC = 150 ∧ BIC = 127.5 ∧ BHC = 105 → 
  B = 70 :=
by
  sorry

end NUMINAMATH_GPT_maximize_perimeter_OIH_l1498_149854


namespace NUMINAMATH_GPT_space_diagonal_of_prism_l1498_149831

theorem space_diagonal_of_prism (l w h : ℝ) (hl : l = 2) (hw : w = 3) (hh : h = 4) :
  (l ^ 2 + w ^ 2 + h ^ 2).sqrt = Real.sqrt 29 :=
by
  rw [hl, hw, hh]
  sorry

end NUMINAMATH_GPT_space_diagonal_of_prism_l1498_149831


namespace NUMINAMATH_GPT_parabola_intersects_x_axis_l1498_149825

theorem parabola_intersects_x_axis 
  (a c : ℝ) 
  (h : ∃ x : ℝ, x = 1 ∧ (a * x^2 + x + c = 0)) : 
  a + c = -1 :=
sorry

end NUMINAMATH_GPT_parabola_intersects_x_axis_l1498_149825


namespace NUMINAMATH_GPT_find_f_2_l1498_149840

-- Condition: f(x + 1) = x^2 - 2x
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Statement to prove
theorem find_f_2 : f 2 = -1 := by
  sorry

end NUMINAMATH_GPT_find_f_2_l1498_149840


namespace NUMINAMATH_GPT_value_of_power_l1498_149843

theorem value_of_power (a : ℝ) (m n k : ℕ) (h1 : a ^ m = 2) (h2 : a ^ n = 4) (h3 : a ^ k = 32) : 
  a ^ (3 * m + 2 * n - k) = 4 := 
by sorry

end NUMINAMATH_GPT_value_of_power_l1498_149843


namespace NUMINAMATH_GPT_change_in_spiders_l1498_149862

theorem change_in_spiders 
  (x a y b : ℤ) 
  (h1 : x + a = 20) 
  (h2 : y + b = 23) 
  (h3 : x - b = 5) :
  y - a = 8 := 
by
  sorry

end NUMINAMATH_GPT_change_in_spiders_l1498_149862


namespace NUMINAMATH_GPT_shari_total_distance_l1498_149801

theorem shari_total_distance (speed : ℝ) (time_1 : ℝ) (rest : ℝ) (time_2 : ℝ) (distance : ℝ) :
  speed = 4 ∧ time_1 = 2 ∧ rest = 0.5 ∧ time_2 = 1 ∧ distance = speed * time_1 + speed * time_2 → distance = 12 :=
by
  sorry

end NUMINAMATH_GPT_shari_total_distance_l1498_149801


namespace NUMINAMATH_GPT_quadratic_eq_solutions_l1498_149821

theorem quadratic_eq_solutions : ∀ x : ℝ, 2 * x^2 - 5 * x + 3 = 0 ↔ x = 3 / 2 ∨ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_eq_solutions_l1498_149821


namespace NUMINAMATH_GPT_sum_pos_implies_one_pos_l1498_149876

theorem sum_pos_implies_one_pos (a b : ℝ) (h : a + b > 0) : a > 0 ∨ b > 0 := 
sorry

end NUMINAMATH_GPT_sum_pos_implies_one_pos_l1498_149876


namespace NUMINAMATH_GPT_bryce_received_raisins_l1498_149842

theorem bryce_received_raisins
  (C B : ℕ)
  (h1 : B = C + 8)
  (h2 : C = B / 3) :
  B = 12 :=
by sorry

end NUMINAMATH_GPT_bryce_received_raisins_l1498_149842


namespace NUMINAMATH_GPT_find_f_neg_one_l1498_149870

noncomputable def f (x : ℝ) : ℝ := 
  if 0 ≤ x then x^2 + 2 * x else - ( (x^2) + (2 * x))

theorem find_f_neg_one : 
  f (-1) = -3 :=
by 
  sorry

end NUMINAMATH_GPT_find_f_neg_one_l1498_149870


namespace NUMINAMATH_GPT_part_I_part_II_l1498_149888

open Real

def f (x m n : ℝ) := abs (x - m) + abs (x + n)

theorem part_I (m n M : ℝ) (h1 : m + n = 9) (h2 : ∀ x : ℝ, f x m n ≥ M) : M ≤ 9 := 
sorry

theorem part_II (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 9) : (a + b) * (a^3 + b^3) ≥ 81 := 
sorry

end NUMINAMATH_GPT_part_I_part_II_l1498_149888


namespace NUMINAMATH_GPT_floor_sum_proof_l1498_149847

noncomputable def floor_sum (x y z w : ℝ) : ℝ :=
  x + y + z + w

theorem floor_sum_proof
  (x y z w : ℝ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y)
  (hz_pos : 0 < z)
  (hw_pos : 0 < w)
  (h1 : x^2 + y^2 = 2010)
  (h2 : z^2 + w^2 = 2010)
  (h3 : x * z = 1008)
  (h4 : y * w = 1008) :
  ⌊floor_sum x y z w⌋ = 126 :=
by
  sorry

end NUMINAMATH_GPT_floor_sum_proof_l1498_149847


namespace NUMINAMATH_GPT_find_c_plus_inv_b_l1498_149817

theorem find_c_plus_inv_b (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h1 : a * b * c = 1) (h2 : a + 1 / c = 7) (h3 : b + 1 / a = 35) :
  c + 1 / b = 11 / 61 :=
by
  sorry

end NUMINAMATH_GPT_find_c_plus_inv_b_l1498_149817


namespace NUMINAMATH_GPT_correct_calculation_l1498_149851

def calculation_is_correct (a b x y : ℝ) : Prop :=
  (3 * x^2 * y - 2 * y * x^2 = x^2 * y)

theorem correct_calculation :
  ∀ (a b x y : ℝ), calculation_is_correct a b x y :=
by
  intros a b x y
  sorry

end NUMINAMATH_GPT_correct_calculation_l1498_149851


namespace NUMINAMATH_GPT_value_of_y_l1498_149898

theorem value_of_y (y : ℤ) (h : (2010 + y)^2 = y^2) : y = -1005 :=
sorry

end NUMINAMATH_GPT_value_of_y_l1498_149898


namespace NUMINAMATH_GPT_hash_op_correct_l1498_149877

-- Definition of the custom operation #
def hash_op (a b : ℕ) : ℕ := a * b - b + b ^ 2

-- The theorem to prove that 3 # 8 = 80
theorem hash_op_correct : hash_op 3 8 = 80 :=
by
  sorry

end NUMINAMATH_GPT_hash_op_correct_l1498_149877


namespace NUMINAMATH_GPT_angle_A_size_max_area_triangle_l1498_149806

open Real

variable {A B C a b c : ℝ}

-- Part 1: Prove the size of angle A given the conditions
theorem angle_A_size (h1 : (2 * c - b) / a = cos B / cos A) :
  A = π / 3 :=
sorry

-- Part 2: Prove the maximum area of triangle ABC
theorem max_area_triangle (h2 : a = 2 * sqrt 5) :
  ∃ (S : ℝ), S = 5 * sqrt 3 ∧ ∀ (b c : ℝ), S ≤ 1/2 * b * c * sin (π / 3) :=
sorry

end NUMINAMATH_GPT_angle_A_size_max_area_triangle_l1498_149806


namespace NUMINAMATH_GPT_xiao_ming_second_half_time_l1498_149803

theorem xiao_ming_second_half_time :
  ∀ (total_distance : ℕ) (speed1 : ℕ) (speed2 : ℕ), 
    total_distance = 360 →
    speed1 = 5 →
    speed2 = 4 →
    let t_total := total_distance / (speed1 + speed2) * 2
    let half_distance := total_distance / 2
    let t2 := half_distance / speed2
    half_distance / speed2 + (half_distance / speed1) = 44 :=
sorry

end NUMINAMATH_GPT_xiao_ming_second_half_time_l1498_149803


namespace NUMINAMATH_GPT_minimum_shift_value_l1498_149829

theorem minimum_shift_value
    (m : ℝ) 
    (h1 : m > 0) :
    (∃ (k : ℤ), m = k * π - π / 3 ∧ k > 0) → (m = (2 * π) / 3) :=
sorry

end NUMINAMATH_GPT_minimum_shift_value_l1498_149829


namespace NUMINAMATH_GPT_money_last_weeks_l1498_149822

-- Conditions
def money_from_mowing : ℕ := 14
def money_from_weeding : ℕ := 31
def weekly_spending : ℕ := 5

-- Total money made
def total_money : ℕ := money_from_mowing + money_from_weeding

-- Expected result
def expected_weeks : ℕ := 9

-- Prove the number of weeks the money will last Jerry
theorem money_last_weeks : (total_money / weekly_spending) = expected_weeks :=
by
  sorry

end NUMINAMATH_GPT_money_last_weeks_l1498_149822


namespace NUMINAMATH_GPT_percentage_markup_l1498_149884

theorem percentage_markup (CP SP : ℕ) (hCP : CP = 800) (hSP : SP = 1000) :
  let Markup := SP - CP
  let PercentageMarkup := (Markup : ℚ) / CP * 100
  PercentageMarkup = 25 := by
  sorry

end NUMINAMATH_GPT_percentage_markup_l1498_149884


namespace NUMINAMATH_GPT_find_value_of_sum_of_squares_l1498_149846

theorem find_value_of_sum_of_squares
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : a + b + c = 0)
  (h5 : a^4 + b^4 + c^4 = a^6 + b^6 + c^6) :
  a^2 + b^2 + c^2 = 6 / 5 := by
  sorry

end NUMINAMATH_GPT_find_value_of_sum_of_squares_l1498_149846


namespace NUMINAMATH_GPT_roots_polynomial_value_l1498_149897

theorem roots_polynomial_value (a b c : ℝ) 
  (h1 : a + b + c = 15)
  (h2 : a * b + b * c + c * a = 25)
  (h3 : a * b * c = 12) :
  (2 + a) * (2 + b) * (2 + c) = 130 := 
by
  sorry

end NUMINAMATH_GPT_roots_polynomial_value_l1498_149897


namespace NUMINAMATH_GPT_sam_found_seashells_l1498_149889

def seashells_given : Nat := 18
def seashells_left : Nat := 17
def seashells_found : Nat := seashells_given + seashells_left

theorem sam_found_seashells : seashells_found = 35 := by
  sorry

end NUMINAMATH_GPT_sam_found_seashells_l1498_149889


namespace NUMINAMATH_GPT_eval_expr_ceil_floor_l1498_149869

theorem eval_expr_ceil_floor (x y : ℚ) (h1 : x = 7 / 3) (h2 : y = -7 / 3) :
  (⌈x⌉ + ⌊y⌋ = 0) :=
sorry

end NUMINAMATH_GPT_eval_expr_ceil_floor_l1498_149869


namespace NUMINAMATH_GPT_mangoes_combined_l1498_149867

variable (Alexis Dilan Ashley : ℕ)

theorem mangoes_combined :
  (Alexis = 60) → (Alexis = 4 * (Dilan + Ashley)) → (Alexis + Dilan + Ashley = 75) := 
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_mangoes_combined_l1498_149867


namespace NUMINAMATH_GPT_rattlesnakes_count_l1498_149844

-- Definitions
def total_snakes : ℕ := 200
def boa_constrictors : ℕ := 40
def pythons : ℕ := 3 * boa_constrictors
def rattlesnakes : ℕ := total_snakes - (boa_constrictors + pythons)

-- Theorem to prove
theorem rattlesnakes_count : rattlesnakes = 40 := by
  -- provide proof here
  sorry

end NUMINAMATH_GPT_rattlesnakes_count_l1498_149844


namespace NUMINAMATH_GPT_min_cards_to_guarantee_four_same_suit_l1498_149863

theorem min_cards_to_guarantee_four_same_suit (n : ℕ) (suits : Fin n) (cards_per_suit : ℕ) (total_cards : ℕ)
  (h1 : n = 4) (h2 : cards_per_suit = 13) : total_cards ≥ 13 :=
by
  sorry

end NUMINAMATH_GPT_min_cards_to_guarantee_four_same_suit_l1498_149863


namespace NUMINAMATH_GPT_arithmetic_sequence_general_term_l1498_149836

theorem arithmetic_sequence_general_term (a : ℕ → ℤ) (d : ℤ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h_increasing : d > 0)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = a 2 ^ 2 - 4) :
  ∀ n, a n = 2 * n - 1 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_term_l1498_149836


namespace NUMINAMATH_GPT_max_b_in_box_l1498_149859

theorem max_b_in_box (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b ≤ 12 := 
by
  sorry

end NUMINAMATH_GPT_max_b_in_box_l1498_149859


namespace NUMINAMATH_GPT_cuboid_surface_area_l1498_149811

-- Define the given conditions
def cuboid (a b c : ℝ) := 2 * (a + b + c)

-- Given areas of distinct sides
def area_face_1 : ℝ := 4
def area_face_2 : ℝ := 3
def area_face_3 : ℝ := 6

-- Prove the total surface area of the cuboid
theorem cuboid_surface_area : cuboid area_face_1 area_face_2 area_face_3 = 26 :=
by
  sorry

end NUMINAMATH_GPT_cuboid_surface_area_l1498_149811


namespace NUMINAMATH_GPT_thabo_books_l1498_149820

/-- Thabo's book count puzzle -/
theorem thabo_books (H P F : ℕ) (h1 : P = H + 20) (h2 : F = 2 * P) (h3 : H + P + F = 200) : H = 35 :=
by
  -- sorry is used to skip the proof, only state the theorem.
  sorry

end NUMINAMATH_GPT_thabo_books_l1498_149820


namespace NUMINAMATH_GPT_find_y_l1498_149886

theorem find_y (steps distance : ℕ) (total_steps : ℕ) (marking_step : ℕ)
  (h1 : total_steps = 8)
  (h2 : distance = 48)
  (h3 : marking_step = 6) :
  steps = distance / total_steps * marking_step → steps = 36 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_y_l1498_149886


namespace NUMINAMATH_GPT_infinite_rational_points_on_circle_l1498_149824

noncomputable def exists_infinitely_many_rational_points_on_circle : Prop :=
  ∃ f : ℚ → ℚ × ℚ, (∀ m : ℚ, (f m).1 ^ 2 + (f m).2 ^ 2 = 1) ∧ 
                   (∀ x y : ℚ, ∃ m : ℚ, (x, y) = f m)

theorem infinite_rational_points_on_circle :
  ∃ (f : ℚ → ℚ × ℚ), (∀ m : ℚ, (f m).1 ^ 2 + (f m).2 ^ 2 = 1) ∧ 
                     (∀ x y : ℚ, ∃ m : ℚ, (x, y) = f m) := sorry

end NUMINAMATH_GPT_infinite_rational_points_on_circle_l1498_149824


namespace NUMINAMATH_GPT_probability_outside_circle_is_7_over_9_l1498_149874

noncomputable def probability_point_outside_circle :
    ℚ :=
sorry

theorem probability_outside_circle_is_7_over_9 :
    probability_point_outside_circle = 7 / 9 :=
sorry

end NUMINAMATH_GPT_probability_outside_circle_is_7_over_9_l1498_149874


namespace NUMINAMATH_GPT_ship_speed_in_still_water_l1498_149845

theorem ship_speed_in_still_water 
  (distance : ℝ) 
  (time : ℝ) 
  (current_speed : ℝ) 
  (x : ℝ) 
  (h1 : distance = 36)
  (h2 : time = 6)
  (h3 : current_speed = 3) 
  (h4 : (18 / (x + 3) + 18 / (x - 3) = 6)) 
  : x = 3 + 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_ship_speed_in_still_water_l1498_149845


namespace NUMINAMATH_GPT_percent_is_250_l1498_149856

def part : ℕ := 150
def whole : ℕ := 60
def percent := (part : ℚ) / (whole : ℚ) * 100

theorem percent_is_250 : percent = 250 := 
by 
  sorry

end NUMINAMATH_GPT_percent_is_250_l1498_149856


namespace NUMINAMATH_GPT_prove_ordered_pair_l1498_149858

noncomputable def p : ℝ → ℝ := sorry
noncomputable def q : ℝ → ℝ := sorry

theorem prove_ordered_pair (h1 : p 0 = -24) (h2 : q 0 = 30) (h3 : ∀ x : ℝ, p (q x) = q (p x)) : (p 3, q 6) = (3, -24) := 
sorry

end NUMINAMATH_GPT_prove_ordered_pair_l1498_149858


namespace NUMINAMATH_GPT_fraction_dropped_l1498_149852

theorem fraction_dropped (f : ℝ) 
  (h1 : 0 ≤ f ∧ f ≤ 1) 
  (initial_passengers : ℝ) 
  (final_passenger_count : ℝ)
  (first_pickup : ℝ)
  (second_pickup : ℝ) 
  (first_drop_factor : ℝ)
  (second_drop_factor : ℕ):
  initial_passengers = 270 →
  final_passenger_count = 242 →
  first_pickup = 280 →
  second_pickup = 12 →
  first_drop_factor = f →
  second_drop_factor = 2 →
  ((initial_passengers - initial_passengers * first_drop_factor) + first_pickup) / second_drop_factor + second_pickup = final_passenger_count →
  f = 1 / 3 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_fraction_dropped_l1498_149852


namespace NUMINAMATH_GPT_average_points_per_player_l1498_149860

theorem average_points_per_player (Lefty_points Righty_points OtherTeammate_points : ℕ)
  (hL : Lefty_points = 20)
  (hR : Righty_points = Lefty_points / 2)
  (hO : OtherTeammate_points = 6 * Righty_points) :
  (Lefty_points + Righty_points + OtherTeammate_points) / 3 = 30 :=
by
  sorry

end NUMINAMATH_GPT_average_points_per_player_l1498_149860


namespace NUMINAMATH_GPT_meaningful_sqrt_domain_l1498_149895

theorem meaningful_sqrt_domain (x : ℝ) : (x - 2 ≥ 0) ↔ (x ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_meaningful_sqrt_domain_l1498_149895


namespace NUMINAMATH_GPT_find_common_difference_l1498_149892

variable (a : ℕ → ℤ)  -- define the arithmetic sequence as a function from ℕ to ℤ
variable (d : ℤ)      -- define the common difference

-- Define the conditions
def conditions := (a 5 = 10) ∧ (a 12 = 31)

-- Define the formula for the nth term of the arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) (n : ℕ) := a 1 + d * (n - 1)

-- Prove that the common difference d is 3 given the conditions
theorem find_common_difference (h : conditions a) : d = 3 :=
sorry

end NUMINAMATH_GPT_find_common_difference_l1498_149892


namespace NUMINAMATH_GPT_find_number_l1498_149871

theorem find_number :
  ∃ x : ℝ, (x - 1.9) * 1.5 + 32 / 2.5 = 20 ∧ x = 13.9 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1498_149871


namespace NUMINAMATH_GPT_average_of_values_l1498_149832

theorem average_of_values (z : ℝ) : 
  (0 + 1 + 2 + 4 + 8 + 32 : ℝ) * z / (6 : ℝ) = 47 * z / 6 :=
by
  sorry

end NUMINAMATH_GPT_average_of_values_l1498_149832


namespace NUMINAMATH_GPT_gcd_of_2475_and_7350_is_225_l1498_149880

-- Definitions and conditions based on the factorization of the given numbers
def factor_2475 := (5^2 * 3^2 * 11)
def factor_7350 := (2 * 3^2 * 5^2 * 7)

-- Proof problem: showing the GCD of 2475 and 7350 is 225
theorem gcd_of_2475_and_7350_is_225 : Nat.gcd 2475 7350 = 225 :=
by
  -- Formal proof would go here
  sorry

end NUMINAMATH_GPT_gcd_of_2475_and_7350_is_225_l1498_149880


namespace NUMINAMATH_GPT_quadrilateral_area_l1498_149835

theorem quadrilateral_area 
  (AB BC DC : ℝ)
  (hAB_perp_BC : true)
  (hDC_perp_BC : true)
  (hAB_eq : AB = 8)
  (hDC_eq : DC = 3)
  (hBC_eq : BC = 10) : 
  (1 / 2 * (AB + DC) * BC = 55) :=
by 
  sorry

end NUMINAMATH_GPT_quadrilateral_area_l1498_149835


namespace NUMINAMATH_GPT_part_I_part_II_l1498_149819

open Set

variable (a b : ℝ)

theorem part_I (A : Set ℝ) (B : Set ℝ) (hA_def : A = { x | a * x^2 + b * x + 1 = 0 })
  (hB_def : B = { -1, 1 }) (hB_sub_A : B ⊆ A) : a = -1 :=
  sorry

theorem part_II (A : Set ℝ) (B : Set ℝ) (hA_def : A = { x | a * x^2 + b * x + 1 = 0 })
  (hB_def : B = { -1, 1 }) (hA_inter_B_nonempty : A ∩ B ≠ ∅) : a^2 - b^2 + 2 * a = -1 :=
  sorry

end NUMINAMATH_GPT_part_I_part_II_l1498_149819


namespace NUMINAMATH_GPT_solve_x_l1498_149873

theorem solve_x (x : ℝ) (h1 : 8 * x^2 + 7 * x - 1 = 0) (h2 : 24 * x^2 + 53 * x - 7 = 0) : x = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_solve_x_l1498_149873


namespace NUMINAMATH_GPT_unobstructed_sight_l1498_149864

-- Define the curve C as y = 2x^2
def curve (x : ℝ) : ℝ := 2 * x^2

-- Define point A and point B
def pointA : ℝ × ℝ := (0, -2)
def pointB (a : ℝ) : ℝ × ℝ := (3, a)

-- Statement of the problem
theorem unobstructed_sight {a : ℝ} (h : ∀ x : ℝ, 0 ≤ x → x ≤ 3 → 4 * x - 2 ≥ 2 * x^2) : a < 10 :=
sorry

end NUMINAMATH_GPT_unobstructed_sight_l1498_149864


namespace NUMINAMATH_GPT_train_cross_pole_in_time_l1498_149810

noncomputable def time_to_cross_pole (length : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  length / speed_ms

theorem train_cross_pole_in_time :
  time_to_cross_pole 100 126 = 100 / (126 * (1000 / 3600)) :=
by
  -- this will unfold the calculation step-by-step
  unfold time_to_cross_pole
  sorry

end NUMINAMATH_GPT_train_cross_pole_in_time_l1498_149810


namespace NUMINAMATH_GPT_percentage_disliked_by_both_l1498_149883

theorem percentage_disliked_by_both 
  (total_comic_books : ℕ) 
  (percentage_females_like : ℕ) 
  (comic_books_males_like : ℕ) :
  total_comic_books = 300 →
  percentage_females_like = 30 →
  comic_books_males_like = 120 →
  ((total_comic_books - (total_comic_books * percentage_females_like / 100) - comic_books_males_like) * 100 / total_comic_books) = 30 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_percentage_disliked_by_both_l1498_149883


namespace NUMINAMATH_GPT_find_values_of_a_and_b_find_square_root_l1498_149893

-- Define the conditions
def condition1 (a b : ℤ) : Prop := (2 * b - 2 * a)^3 = -8
def condition2 (a b : ℤ) : Prop := (4 * a + 3 * b)^2 = 9

-- State the problem to prove the values of a and b
theorem find_values_of_a_and_b (a b : ℤ) (h1 : condition1 a b) (h2 : condition2 a b) : 
  a = 3 ∧ b = -1 :=
sorry

-- State the problem to prove the square root of 5a - b
theorem find_square_root (a b : ℤ) (h1 : condition1 a b) (h2 : condition2 a b) (ha : a = 3) (hb : b = -1) :
  ∃ x : ℤ, x^2 = 5 * a - b ∧ (x = 4 ∨ x = -4) :=
sorry

end NUMINAMATH_GPT_find_values_of_a_and_b_find_square_root_l1498_149893


namespace NUMINAMATH_GPT_number_of_people_is_ten_l1498_149814

-- Define the total number of Skittles and the number of Skittles per person.
def total_skittles : ℕ := 20
def skittles_per_person : ℕ := 2

-- Define the number of people as the total Skittles divided by the Skittles per person.
def number_of_people : ℕ := total_skittles / skittles_per_person

-- Theorem stating that the number of people is 10.
theorem number_of_people_is_ten : number_of_people = 10 := sorry

end NUMINAMATH_GPT_number_of_people_is_ten_l1498_149814


namespace NUMINAMATH_GPT_harmonic_mean_closest_to_2_l1498_149865

theorem harmonic_mean_closest_to_2 (a : ℝ) (b : ℝ) (h₁ : a = 1) (h₂ : b = 4032) : 
  abs ((2 * a * b) / (a + b) - 2) < 1 :=
by
  rw [h₁, h₂]
  -- The rest of the proof follows from here, skipped with sorry
  sorry

end NUMINAMATH_GPT_harmonic_mean_closest_to_2_l1498_149865


namespace NUMINAMATH_GPT_vasya_made_a_mistake_l1498_149861

theorem vasya_made_a_mistake (A B V G D E : ℕ)
  (h1 : A ≠ B)
  (h2 : V ≠ G)
  (h3 : (10 * A + B) * (10 * V + G) = 1000 * D + 100 * D + 10 * E + E)
  (h4 : ∀ {X Y : ℕ}, X ≠ Y → D ≠ E) :
  False :=
by
  -- Proof goes here (skipped)
  sorry

end NUMINAMATH_GPT_vasya_made_a_mistake_l1498_149861


namespace NUMINAMATH_GPT_value_of_t_plus_k_l1498_149881

noncomputable def f (x t : ℝ) : ℝ := x^3 + (t - 1) * x^2 - 1

theorem value_of_t_plus_k (k t : ℝ)
  (h1 : k ≠ 0)
  (h2 : ∀ x, f x t = 2 * x - 1)
  (h3 : ∃ x₁ x₂, f x₁ t = 2 * x₁ - 1 ∧ f x₂ t = 2 * x₂ - 1) :
  t + k = 7 :=
sorry

end NUMINAMATH_GPT_value_of_t_plus_k_l1498_149881


namespace NUMINAMATH_GPT_gain_percentage_l1498_149872

theorem gain_percentage (SP1 SP2 CP: ℝ) (h1 : SP1 = 102) (h2 : SP2 = 144) (h3 : SP1 = CP - 0.15 * CP) :
  ((SP2 - CP) / CP) * 100 = 20 := by
sorry

end NUMINAMATH_GPT_gain_percentage_l1498_149872


namespace NUMINAMATH_GPT_tan_sum_identity_l1498_149841

theorem tan_sum_identity : (1 + Real.tan (Real.pi / 180)) * (1 + Real.tan (44 * Real.pi / 180)) = 2 := 
by sorry

end NUMINAMATH_GPT_tan_sum_identity_l1498_149841


namespace NUMINAMATH_GPT_perfect_square_expression_l1498_149879

theorem perfect_square_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  ∃ m : ℕ, m^2 = (2 * l - n - k) * (2 * l - n + k) / 2 :=
by 
  sorry

end NUMINAMATH_GPT_perfect_square_expression_l1498_149879


namespace NUMINAMATH_GPT_calculate_a_plus_b_l1498_149899

noncomputable def f (a b x : ℝ) : ℝ := a * x + b
noncomputable def g (x : ℝ) : ℝ := 3 * x - 7

theorem calculate_a_plus_b (a b : ℝ) (h : ∀ x : ℝ, g (f a b x) = 4 * x + 6) : a + b = 17 / 3 :=
by
  sorry

end NUMINAMATH_GPT_calculate_a_plus_b_l1498_149899


namespace NUMINAMATH_GPT_simplify_and_rationalize_l1498_149866

theorem simplify_and_rationalize :
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_rationalize_l1498_149866


namespace NUMINAMATH_GPT_least_number_of_marbles_l1498_149855

def divisible_by (n : ℕ) (d : ℕ) : Prop := n % d = 0

theorem least_number_of_marbles 
  (n : ℕ)
  (h3 : divisible_by n 3)
  (h4 : divisible_by n 4)
  (h5 : divisible_by n 5)
  (h7 : divisible_by n 7)
  (h8 : divisible_by n 8) :
  n = 840 :=
sorry

end NUMINAMATH_GPT_least_number_of_marbles_l1498_149855


namespace NUMINAMATH_GPT_technician_percent_round_trip_l1498_149828

noncomputable def round_trip_percentage_completed (D : ℝ) : ℝ :=
  let total_round_trip := 2 * D
  let distance_completed := D + 0.10 * D
  (distance_completed / total_round_trip) * 100

theorem technician_percent_round_trip (D : ℝ) (h : D > 0) : 
  round_trip_percentage_completed D = 55 := 
by 
  sorry

end NUMINAMATH_GPT_technician_percent_round_trip_l1498_149828


namespace NUMINAMATH_GPT_answer_is_correct_l1498_149848

-- We define the prime checking function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

-- We define the set of candidates satisfying initial prime condition
def candidates : Set ℕ := {A | is_prime A ∧ A < 100 
                                   ∧ is_prime (A + 10) 
                                   ∧ is_prime (A - 20)
                                   ∧ is_prime (A + 30) 
                                   ∧ is_prime (A + 60) 
                                   ∧ is_prime (A + 70)}

-- The explicit set of valid answers
def valid_answers : Set ℕ := {37, 43, 79}

-- The statement that we need to prove
theorem answer_is_correct : candidates = valid_answers := 
sorry

end NUMINAMATH_GPT_answer_is_correct_l1498_149848


namespace NUMINAMATH_GPT_endangered_species_count_l1498_149850

section BirdsSanctuary

-- Define the given conditions
def pairs_per_species : ℕ := 7
def total_pairs : ℕ := 203

-- Define the result to be proved
theorem endangered_species_count : total_pairs / pairs_per_species = 29 := by
  sorry

end BirdsSanctuary

end NUMINAMATH_GPT_endangered_species_count_l1498_149850


namespace NUMINAMATH_GPT_find_value_of_a2_b2_c2_l1498_149887

variable {a b c : ℝ}

theorem find_value_of_a2_b2_c2
  (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h2 : a + b + c = 0)
  (h3 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6 / 5 := 
sorry

end NUMINAMATH_GPT_find_value_of_a2_b2_c2_l1498_149887


namespace NUMINAMATH_GPT_sequence_tuple_l1498_149816

/-- Prove the unique solution to the system of equations derived from the sequence pattern. -/
theorem sequence_tuple (x y : ℝ) (h1 : x + y = 9) (h2 : x - y = 7) : (x, y) = (8, 1) :=
by
  sorry

end NUMINAMATH_GPT_sequence_tuple_l1498_149816


namespace NUMINAMATH_GPT_jane_rejected_percentage_l1498_149878

theorem jane_rejected_percentage (P : ℕ) (John_rejected : ℤ) (Jane_inspected_rejected : ℤ) :
  John_rejected = 7 * P ∧
  Jane_inspected_rejected = 5 * P ∧
  (John_rejected + Jane_inspected_rejected) = 75 * P → 
  Jane_inspected_rejected = P  :=
by sorry

end NUMINAMATH_GPT_jane_rejected_percentage_l1498_149878


namespace NUMINAMATH_GPT_small_drinking_glasses_count_l1498_149853

theorem small_drinking_glasses_count :
  ∀ (large_jelly_beans_per_large_glass small_jelly_beans_per_small_glass total_jelly_beans : ℕ),
  (large_jelly_beans_per_large_glass = 50) →
  (small_jelly_beans_per_small_glass = large_jelly_beans_per_large_glass / 2) →
  (5 * large_jelly_beans_per_large_glass + n * small_jelly_beans_per_small_glass = total_jelly_beans) →
  (total_jelly_beans = 325) →
  n = 3 := by
  sorry

end NUMINAMATH_GPT_small_drinking_glasses_count_l1498_149853


namespace NUMINAMATH_GPT_solve_inequality_l1498_149800

theorem solve_inequality (k : ℝ) :
  (∀ (x : ℝ), (k + 2) * x > k + 2 → x < 1) → k = -3 :=
  by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1498_149800


namespace NUMINAMATH_GPT_margarita_jumps_farther_l1498_149805

-- Definitions for conditions
def RiccianaTotalDistance := 24
def RiccianaRunDistance := 20
def RiccianaJumpDistance := 4

def MargaritaRunDistance := 18
def MargaritaJumpDistance := 2 * RiccianaJumpDistance - 1

-- Theorem statement to prove the question
theorem margarita_jumps_farther :
  (MargaritaRunDistance + MargaritaJumpDistance) - RiccianaTotalDistance = 1 :=
by
  -- Proof will be written here
  sorry

end NUMINAMATH_GPT_margarita_jumps_farther_l1498_149805


namespace NUMINAMATH_GPT_find_congruence_l1498_149834

theorem find_congruence (x : ℤ) (h : 4 * x + 9 ≡ 3 [ZMOD 17]) : 3 * x + 12 ≡ 16 [ZMOD 17] :=
sorry

end NUMINAMATH_GPT_find_congruence_l1498_149834


namespace NUMINAMATH_GPT_negation_of_exists_implies_forall_l1498_149809

theorem negation_of_exists_implies_forall :
  (¬ ∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2 * x + 2 > 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_implies_forall_l1498_149809


namespace NUMINAMATH_GPT_max_pies_without_ingredients_l1498_149894

theorem max_pies_without_ingredients
  (total_pies chocolate_pies berries_pies cinnamon_pies poppy_seeds_pies : ℕ)
  (h1 : total_pies = 60)
  (h2 : chocolate_pies = 1 / 3 * total_pies)
  (h3 : berries_pies = 3 / 5 * total_pies)
  (h4 : cinnamon_pies = 1 / 2 * total_pies)
  (h5 : poppy_seeds_pies = 1 / 5 * total_pies) : 
  total_pies - max chocolate_pies (max berries_pies (max cinnamon_pies poppy_seeds_pies)) = 24 := 
by
  sorry

end NUMINAMATH_GPT_max_pies_without_ingredients_l1498_149894


namespace NUMINAMATH_GPT_ten_yuan_notes_count_l1498_149838

theorem ten_yuan_notes_count (total_notes : ℕ) (total_change : ℕ) (item_cost : ℕ) (change_given : ℕ → ℕ → ℕ) (is_ten_yuan_notes : ℕ → Prop) :
    total_notes = 16 →
    total_change = 95 →
    item_cost = 5 →
    change_given 10 5 = total_change →
    (∃ x y : ℕ, x + y = total_notes ∧ 10 * x + 5 * y = total_change ∧ is_ten_yuan_notes x) → is_ten_yuan_notes 3 :=
by
  sorry

end NUMINAMATH_GPT_ten_yuan_notes_count_l1498_149838


namespace NUMINAMATH_GPT_upstream_speed_l1498_149802

theorem upstream_speed (Vm Vdownstream Vupstream Vs : ℝ) 
  (h1 : Vm = 50) 
  (h2 : Vdownstream = 55) 
  (h3 : Vdownstream = Vm + Vs) 
  (h4 : Vupstream = Vm - Vs) : 
  Vupstream = 45 :=
by
  sorry

end NUMINAMATH_GPT_upstream_speed_l1498_149802


namespace NUMINAMATH_GPT_observations_number_l1498_149839

theorem observations_number 
  (mean : ℚ)
  (wrong_obs corrected_obs : ℚ)
  (new_mean : ℚ)
  (n : ℚ)
  (initial_mean : mean = 36)
  (wrong_obs_taken : wrong_obs = 23)
  (corrected_obs_value : corrected_obs = 34)
  (corrected_mean : new_mean = 36.5) :
  (n * mean + (corrected_obs - wrong_obs) = n * new_mean) → 
  n = 22 :=
by
  sorry

end NUMINAMATH_GPT_observations_number_l1498_149839


namespace NUMINAMATH_GPT_function_decreasing_in_interval_l1498_149837

theorem function_decreasing_in_interval :
  ∀ (x1 x2 : ℝ), (0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2) → 
  (x1 - x2) * ((1 / x1 - x1) - (1 / x2 - x2)) < 0 :=
by
  intros x1 x2 hx
  sorry

end NUMINAMATH_GPT_function_decreasing_in_interval_l1498_149837


namespace NUMINAMATH_GPT_power_function_properties_l1498_149815

theorem power_function_properties (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x ^ a) (h2 : f 2 = Real.sqrt 2) : 
  a = 1 / 2 ∧ ∀ x, 0 ≤ x → f x ≤ f (x + 1) :=
by
  sorry

end NUMINAMATH_GPT_power_function_properties_l1498_149815


namespace NUMINAMATH_GPT_tan_theta_point_l1498_149808

open Real

theorem tan_theta_point :
  ∀ θ : ℝ,
  ∃ (x y : ℝ), x = -sqrt 3 / 2 ∧ y = 1 / 2 ∧ (tan θ) = y / x → (tan θ) = -sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_theta_point_l1498_149808


namespace NUMINAMATH_GPT_count_valid_arrays_l1498_149849

-- Define the integer array condition
def valid_array (x1 x2 x3 x4 : ℕ) : Prop :=
  0 < x1 ∧ x1 ≤ x2 ∧ x2 < x3 ∧ x3 ≤ x4 ∧ x4 < 7

-- State the theorem that proves the number of valid arrays is 70
theorem count_valid_arrays : ∃ (n : ℕ), n = 70 ∧ 
    ∀ (x1 x2 x3 x4 : ℕ), valid_array x1 x2 x3 x4 -> ∃ (n : ℕ), n = 70 :=
by
  -- The proof can be filled in later
  sorry

end NUMINAMATH_GPT_count_valid_arrays_l1498_149849


namespace NUMINAMATH_GPT_domain_of_f1_x2_l1498_149813

theorem domain_of_f1_x2 (f : ℝ → ℝ) : 
  (∀ x, -1 ≤ x ∧ x ≤ 2 → ∃ y, y = f x) → 
  (∀ x, -Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2 → ∃ y, y = f (1 - x^2)) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f1_x2_l1498_149813


namespace NUMINAMATH_GPT_cookies_per_bag_l1498_149891

theorem cookies_per_bag (n_bags : ℕ) (total_cookies : ℕ) (n_candies : ℕ) (h_bags : n_bags = 26) (h_cookies : total_cookies = 52) (h_candies : n_candies = 15) : (total_cookies / n_bags) = 2 :=
by sorry

end NUMINAMATH_GPT_cookies_per_bag_l1498_149891


namespace NUMINAMATH_GPT_bookstore_discount_l1498_149875

noncomputable def discount_percentage (total_spent : ℝ) (over_22 : List ℝ) (under_20 : List ℝ) : ℝ :=
  let disc_over_22 := over_22.map (fun p => p * (1 - 0.30))
  let total_over_22 := disc_over_22.sum
  let total_with_under_20 := total_over_22 + 21
  let total_under_20 := under_20.sum
  let discount_received := total_spent - total_with_under_20
  let discount_percentage := (total_under_20 - discount_received) / total_under_20 * 100
  discount_percentage

theorem bookstore_discount :
  discount_percentage 95 [25.00, 35.00] [18.00, 12.00, 10.00] = 20 := by
  sorry

end NUMINAMATH_GPT_bookstore_discount_l1498_149875


namespace NUMINAMATH_GPT_sum_of_not_visible_faces_l1498_149827

-- Define the sum of the numbers on the faces of one die
def die_sum : ℕ := 21

-- List of visible numbers on the dice
def visible_faces_sum : ℕ := 4 + 3 + 2 + 5 + 1 + 3 + 1

-- Define the total sum of the numbers on the faces of three dice
def total_sum : ℕ := die_sum * 3

-- Statement to prove the sum of not-visible faces equals 44
theorem sum_of_not_visible_faces : 
  total_sum - visible_faces_sum = 44 :=
sorry

end NUMINAMATH_GPT_sum_of_not_visible_faces_l1498_149827


namespace NUMINAMATH_GPT_ratio_of_average_speeds_l1498_149812

-- Define the conditions as constants
def distance_ab : ℕ := 510
def distance_ac : ℕ := 300
def time_eddy : ℕ := 3
def time_freddy : ℕ := 4

-- Define the speeds
def speed_eddy := distance_ab / time_eddy
def speed_freddy := distance_ac / time_freddy

-- The ratio calculation and verification function
def speed_ratio (a b : ℕ) : ℕ × ℕ := (a / Nat.gcd a b, b / Nat.gcd a b)

-- Define the main theorem to be proved
theorem ratio_of_average_speeds : speed_ratio speed_eddy speed_freddy = (34, 15) := by
  sorry

end NUMINAMATH_GPT_ratio_of_average_speeds_l1498_149812


namespace NUMINAMATH_GPT_min_value_343_l1498_149885

noncomputable def min_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : ℝ :=
  (a^2 + 5*a + 2) * (b^2 + 5*b + 2) * (c^2 + 5*c + 2) / (a * b * c)

theorem min_value_343 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  min_value a b c ha hb hc = 343 :=
sorry

end NUMINAMATH_GPT_min_value_343_l1498_149885


namespace NUMINAMATH_GPT_hexagon_pillar_height_l1498_149823

noncomputable def height_of_pillar_at_vertex_F (s : ℝ) (hA hB hC : ℝ) (A : ℝ × ℝ) : ℝ :=
  10

theorem hexagon_pillar_height :
  ∀ (s hA hB hC : ℝ) (A : ℝ × ℝ),
  s = 8 ∧ hA = 15 ∧ hB = 10 ∧ hC = 12 ∧ A = (3, 3 * Real.sqrt 3) →
  height_of_pillar_at_vertex_F s hA hB hC A = 10 := by
  sorry

end NUMINAMATH_GPT_hexagon_pillar_height_l1498_149823


namespace NUMINAMATH_GPT_Bill_original_profit_percentage_l1498_149857

theorem Bill_original_profit_percentage 
  (S : ℝ) 
  (h_S : S = 879.9999999999993) 
  (h_cond : ∀ (P : ℝ), 1.17 * P = S + 56) :
  ∃ (profit_percentage : ℝ), profit_percentage = 10 := 
by
  sorry

end NUMINAMATH_GPT_Bill_original_profit_percentage_l1498_149857


namespace NUMINAMATH_GPT_cab_drivers_income_on_third_day_l1498_149804

theorem cab_drivers_income_on_third_day
  (day1 day2 day4 day5 avg_income n_days : ℝ)
  (h_day1 : day1 = 600)
  (h_day2 : day2 = 250)
  (h_day4 : day4 = 400)
  (h_day5 : day5 = 800)
  (h_avg_income : avg_income = 500)
  (h_n_days : n_days = 5) :
  ∃ day3 : ℝ, (day1 + day2 + day3 + day4 + day5) / n_days = avg_income ∧ day3 = 450 :=
by
  sorry

end NUMINAMATH_GPT_cab_drivers_income_on_third_day_l1498_149804


namespace NUMINAMATH_GPT_positive_when_x_negative_l1498_149826

theorem positive_when_x_negative (x : ℝ) (h : x < 0) : (x / |x|)^2 > 0 := by
  sorry

end NUMINAMATH_GPT_positive_when_x_negative_l1498_149826


namespace NUMINAMATH_GPT_trains_distance_apart_l1498_149807

-- Define the initial conditions
def cattle_train_speed : ℝ := 56
def diesel_train_speed : ℝ := cattle_train_speed - 33
def cattle_train_time : ℝ := 6 + 12
def diesel_train_time : ℝ := 12

-- Calculate distances
def cattle_train_distance : ℝ := cattle_train_speed * cattle_train_time
def diesel_train_distance : ℝ := diesel_train_speed * diesel_train_time

-- Define total distance apart
def distance_apart : ℝ := cattle_train_distance + diesel_train_distance

-- The theorem to prove
theorem trains_distance_apart :
  distance_apart = 1284 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_trains_distance_apart_l1498_149807


namespace NUMINAMATH_GPT_mean_score_calculation_l1498_149818

noncomputable def class_mean_score (total_students students_1 mean_score_1 students_2 mean_score_2 : ℕ) : ℚ :=
  ((students_1 * mean_score_1 + students_2 * mean_score_2) : ℚ) / total_students

theorem mean_score_calculation :
  class_mean_score 60 54 76 6 82 = 76.6 := 
sorry

end NUMINAMATH_GPT_mean_score_calculation_l1498_149818


namespace NUMINAMATH_GPT_quadratic_no_real_roots_l1498_149833

open Real

theorem quadratic_no_real_roots
  (p q a b c : ℝ)
  (hpq : p ≠ q)
  (hpositive_p : 0 < p)
  (hpositive_q : 0 < q)
  (hpositive_a : 0 < a)
  (hpositive_b : 0 < b)
  (hpositive_c : 0 < c)
  (h_geo_sequence : a^2 = p * q)
  (h_ari_sequence : b + c = p + q) :
  (a^2 - b * c) < 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_l1498_149833


namespace NUMINAMATH_GPT_negation_of_p_is_neg_p_l1498_149890

-- Define the proposition p
def p : Prop := ∀ x : ℝ, x > 3 → x^3 - 27 > 0

-- Define the negation of proposition p
def neg_p : Prop := ∃ x : ℝ, x > 3 ∧ x^3 - 27 ≤ 0

-- The Lean statement that proves the problem
theorem negation_of_p_is_neg_p : ¬ p ↔ neg_p := by
  sorry

end NUMINAMATH_GPT_negation_of_p_is_neg_p_l1498_149890


namespace NUMINAMATH_GPT_kitty_cleaning_time_l1498_149882

theorem kitty_cleaning_time
    (picking_up_toys : ℕ := 5)
    (vacuuming : ℕ := 20)
    (dusting_furniture : ℕ := 10)
    (total_time_4_weeks : ℕ := 200)
    (weeks : ℕ := 4)
    : (total_time_4_weeks - weeks * (picking_up_toys + vacuuming + dusting_furniture)) / weeks = 15 := by
    sorry

end NUMINAMATH_GPT_kitty_cleaning_time_l1498_149882
