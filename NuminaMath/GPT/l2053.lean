import Mathlib

namespace NUMINAMATH_GPT_megan_markers_final_count_l2053_205379

theorem megan_markers_final_count :
  let initial_markers := 217
  let robert_gave := 109
  let sarah_took := 35
  let teacher_multiplier := 3
  let final_markers := (initial_markers + robert_gave - sarah_took) * (1 + teacher_multiplier) / 2
  final_markers = 582 :=
by
  let initial_markers := 217
  let robert_gave := 109
  let sarah_took := 35
  let teacher_multiplier := 3
  let final_markers := (initial_markers + robert_gave - sarah_took) * (1 + teacher_multiplier) / 2
  have h : final_markers = 582 := sorry
  exact h

end NUMINAMATH_GPT_megan_markers_final_count_l2053_205379


namespace NUMINAMATH_GPT_probability_product_positive_correct_l2053_205323

noncomputable def probability_product_positive : ℚ :=
  let length_total := 45
  let length_negative := 30
  let length_positive := 15
  let prob_negative := (length_negative : ℚ) / length_total
  let prob_positive := (length_positive : ℚ) / length_total
  let prob_product_positive := prob_negative^2 + prob_positive^2
  prob_product_positive

theorem probability_product_positive_correct :
  probability_product_positive = 5 / 9 :=
by
  sorry

end NUMINAMATH_GPT_probability_product_positive_correct_l2053_205323


namespace NUMINAMATH_GPT_right_triangle_side_length_l2053_205320

theorem right_triangle_side_length (r f : ℝ) (h : f < 2 * r) :
  let c := 2 * r
  let a := (f / (4 * c)) * (f + Real.sqrt (f^2 + 8 * c^2))
  a = (f / (4 * (2 * r))) * (f + Real.sqrt (f^2 + 8 * (2 * r)^2)) :=
by
  let c := 2 * r
  let a := (f / (4 * c)) * (f + Real.sqrt (f^2 + 8 * c^2))
  have acalc : a = (f / (4 * (2 * r))) * (f + Real.sqrt (f^2 + 8 * (2 * r)^2)) := by sorry
  exact acalc

end NUMINAMATH_GPT_right_triangle_side_length_l2053_205320


namespace NUMINAMATH_GPT_fixed_point_1_3_l2053_205333

noncomputable def fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : Prop :=
  (f (1) = 3) where f x := a^(x-1) + 2

theorem fixed_point_1_3 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : fixed_point a h1 h2 :=
by
  unfold fixed_point
  sorry

end NUMINAMATH_GPT_fixed_point_1_3_l2053_205333


namespace NUMINAMATH_GPT_operation_1_even_if_input_even_operation_2_even_if_input_even_operation_3_even_if_input_even_operation_4_not_always_even_if_input_even_operation_5_not_always_even_if_input_even_l2053_205389

-- Define what an even integer is
def is_even (a : ℤ) : Prop := ∃ k : ℤ, a = 2 * k

-- Define the operations
def add_four (a : ℤ) := a + 4
def subtract_six (a : ℤ) := a - 6
def multiply_by_eight (a : ℤ) := a * 8
def divide_by_two_add_two (a : ℤ) := a / 2 + 2
def average_with_ten (a : ℤ) := (a + 10) / 2

-- The proof statements
theorem operation_1_even_if_input_even (a : ℤ) (h : is_even a) : is_even (add_four a) := sorry
theorem operation_2_even_if_input_even (a : ℤ) (h : is_even a) : is_even (subtract_six a) := sorry
theorem operation_3_even_if_input_even (a : ℤ) (h : is_even a) : is_even (multiply_by_eight a) := sorry
theorem operation_4_not_always_even_if_input_even (a : ℤ) (h : is_even a) : ¬ is_even (divide_by_two_add_two a) := sorry
theorem operation_5_not_always_even_if_input_even (a : ℤ) (h : is_even a) : ¬ is_even (average_with_ten a) := sorry

end NUMINAMATH_GPT_operation_1_even_if_input_even_operation_2_even_if_input_even_operation_3_even_if_input_even_operation_4_not_always_even_if_input_even_operation_5_not_always_even_if_input_even_l2053_205389


namespace NUMINAMATH_GPT_arrangement_books_l2053_205311

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem arrangement_books : combination 9 4 = 126 := by
  sorry

end NUMINAMATH_GPT_arrangement_books_l2053_205311


namespace NUMINAMATH_GPT_cost_difference_of_dolls_proof_l2053_205329

-- Define constants
def cost_large_doll : ℝ := 7
def total_spent : ℝ := 350
def additional_dolls : ℝ := 20

-- Define the function for the cost of small dolls
def cost_small_doll (S : ℝ) : Prop :=
  total_spent / S = total_spent / cost_large_doll + additional_dolls

-- The statement given the conditions and solving for the difference in cost
theorem cost_difference_of_dolls_proof : 
  ∃ S, cost_small_doll S ∧ (cost_large_doll - S = 2) :=
by
  sorry

end NUMINAMATH_GPT_cost_difference_of_dolls_proof_l2053_205329


namespace NUMINAMATH_GPT_arithmetic_sequence_next_term_perfect_square_sequence_next_term_l2053_205381

theorem arithmetic_sequence_next_term (a : ℕ → ℕ) (n : ℕ) (h₀ : a 0 = 0) (h₁ : ∀ n, a (n + 1) = a n + 3) :
  a 5 = 15 :=
by sorry

theorem perfect_square_sequence_next_term (b : ℕ → ℕ) (k : ℕ) (h₀ : ∀ k, b k = (k + 1) * (k + 1)) :
  b 5 = 36 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_next_term_perfect_square_sequence_next_term_l2053_205381


namespace NUMINAMATH_GPT_people_who_own_neither_l2053_205350

theorem people_who_own_neither (total_people cat_owners cat_and_dog_owners dog_owners non_cat_dog_owners: ℕ)
        (h1: total_people = 522)
        (h2: 20 * cat_and_dog_owners = cat_owners)
        (h3: 7 * dog_owners = 10 * (dog_owners + cat_and_dog_owners))
        (h4: 2 * non_cat_dog_owners = (non_cat_dog_owners + dog_owners)):
    non_cat_dog_owners = 126 := 
by
  sorry

end NUMINAMATH_GPT_people_who_own_neither_l2053_205350


namespace NUMINAMATH_GPT_calculate_green_paint_l2053_205352

theorem calculate_green_paint {green white : ℕ} (ratio_white_to_green : 5 * green = 3 * white) (use_white_paint : white = 15) : green = 9 :=
by
  sorry

end NUMINAMATH_GPT_calculate_green_paint_l2053_205352


namespace NUMINAMATH_GPT_airplane_average_speed_l2053_205359

theorem airplane_average_speed :
  ∃ v : ℝ, 
  (1140 = 12 * (0.9 * v) + 26 * (1.2 * v)) ∧ 
  v = 27.14 := 
by
  sorry

end NUMINAMATH_GPT_airplane_average_speed_l2053_205359


namespace NUMINAMATH_GPT_stockholm_to_uppsala_distance_l2053_205345

-- Definitions based on conditions
def map_distance_cm : ℝ := 3
def scale_cm_to_km : ℝ := 80

-- Theorem statement based on the question and correct answer
theorem stockholm_to_uppsala_distance : 
  (map_distance_cm * scale_cm_to_km = 240) :=
by 
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_stockholm_to_uppsala_distance_l2053_205345


namespace NUMINAMATH_GPT_bianca_points_per_bag_l2053_205321

theorem bianca_points_per_bag (total_bags : ℕ) (not_recycled : ℕ) (total_points : ℕ) 
  (h1 : total_bags = 17) 
  (h2 : not_recycled = 8) 
  (h3 : total_points = 45) : 
  total_points / (total_bags - not_recycled) = 5 :=
by
  sorry 

end NUMINAMATH_GPT_bianca_points_per_bag_l2053_205321


namespace NUMINAMATH_GPT_renu_suma_combined_work_days_l2053_205342

theorem renu_suma_combined_work_days :
  (1 / (1 / 8 + 1 / 4.8)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_renu_suma_combined_work_days_l2053_205342


namespace NUMINAMATH_GPT_number_of_terms_arithmetic_sequence_l2053_205344

theorem number_of_terms_arithmetic_sequence :
  ∀ (a d l : ℤ), a = -36 → d = 6 → l = 66 → ∃ n, l = a + (n-1) * d ∧ n = 18 :=
by
  intros a d l ha hd hl
  exists 18
  rw [ha, hd, hl]
  sorry

end NUMINAMATH_GPT_number_of_terms_arithmetic_sequence_l2053_205344


namespace NUMINAMATH_GPT_average_wage_per_day_l2053_205353

theorem average_wage_per_day :
  let num_male := 20
  let num_female := 15
  let num_child := 5
  let wage_male := 35
  let wage_female := 20
  let wage_child := 8
  let total_wages := (num_male * wage_male) + (num_female * wage_female) + (num_child * wage_child)
  let total_workers := num_male + num_female + num_child
  total_wages / total_workers = 26 := by
  sorry

end NUMINAMATH_GPT_average_wage_per_day_l2053_205353


namespace NUMINAMATH_GPT_total_area_of_strips_l2053_205332

def strip1_length := 12
def strip1_width := 1
def strip2_length := 8
def strip2_width := 2
def num_strips1 := 2
def num_strips2 := 2
def overlap_area_per_strip := 2
def num_overlaps := 4
def total_area_covered := 48

theorem total_area_of_strips : 
  num_strips1 * (strip1_length * strip1_width) + 
  num_strips2 * (strip2_length * strip2_width) - 
  num_overlaps * overlap_area_per_strip = total_area_covered := sorry

end NUMINAMATH_GPT_total_area_of_strips_l2053_205332


namespace NUMINAMATH_GPT_quadrilateral_area_l2053_205319

theorem quadrilateral_area {d o1 o2 : ℝ} (hd : d = 15) (ho1 : o1 = 6) (ho2 : o2 = 4) :
  (d * (o1 + o2)) / 2 = 75 := by
  sorry

end NUMINAMATH_GPT_quadrilateral_area_l2053_205319


namespace NUMINAMATH_GPT_sum_of_prime_factors_2310_l2053_205322

def is_prime (n : Nat) : Prop :=
  2 ≤ n ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def prime_factors_sum (n : Nat) : Nat :=
  (List.filter Nat.Prime (Nat.factors n)).sum

theorem sum_of_prime_factors_2310 :
  prime_factors_sum 2310 = 28 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_prime_factors_2310_l2053_205322


namespace NUMINAMATH_GPT_mass_of_man_is_correct_l2053_205373

-- Definitions for conditions
def length_of_boat : ℝ := 3
def breadth_of_boat : ℝ := 2
def sinking_depth : ℝ := 0.012
def density_of_water : ℝ := 1000

-- Volume of water displaced
def volume_displaced := length_of_boat * breadth_of_boat * sinking_depth

-- Mass of the man
def mass_of_man := density_of_water * volume_displaced

-- Prove that the mass of the man is 72 kg
theorem mass_of_man_is_correct : mass_of_man = 72 := by
  sorry

end NUMINAMATH_GPT_mass_of_man_is_correct_l2053_205373


namespace NUMINAMATH_GPT_simplify_division_l2053_205354

theorem simplify_division (x : ℝ) : 2 * x^8 / x^4 = 2 * x^4 := 
by sorry

end NUMINAMATH_GPT_simplify_division_l2053_205354


namespace NUMINAMATH_GPT_inequality_proof_l2053_205328

theorem inequality_proof (x y : ℝ) (hx : x > -1) (hy : y > -1) (hxy : x + y = 1) : 
    (x / (y + 1) + y / (x + 1) ≥ 2 / 3) := 
  sorry

end NUMINAMATH_GPT_inequality_proof_l2053_205328


namespace NUMINAMATH_GPT_printer_fraction_l2053_205336

noncomputable def basic_computer_price : ℝ := 2000
noncomputable def total_basic_price : ℝ := 2500
noncomputable def printer_price : ℝ := total_basic_price - basic_computer_price -- inferred as 500

noncomputable def enhanced_computer_price : ℝ := basic_computer_price + 500
noncomputable def total_enhanced_price : ℝ := enhanced_computer_price + printer_price -- inferred as 3000

theorem printer_fraction  (h1 : basic_computer_price + printer_price = total_basic_price)
                          (h2 : basic_computer_price = 2000)
                          (h3 : enhanced_computer_price = basic_computer_price + 500) :
  printer_price / total_enhanced_price = 1 / 6 :=
  sorry

end NUMINAMATH_GPT_printer_fraction_l2053_205336


namespace NUMINAMATH_GPT_set_intersection_complement_l2053_205356

-- Define the sets
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 3, 4, 5}
def B : Set ℕ := {2, 3, 6, 7}

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := {x | x ∈ U ∧ ¬ x ∈ A}

-- Define the intersection of B and complement_U_A
def B_inter_complement_U_A : Set ℕ := B ∩ complement_U_A

-- The statement to prove: B ∩ complement_U_A = {6, 7}
theorem set_intersection_complement :
  B_inter_complement_U_A = {6, 7} := by sorry

end NUMINAMATH_GPT_set_intersection_complement_l2053_205356


namespace NUMINAMATH_GPT_mary_jenny_red_marble_ratio_l2053_205377

def mary_red_marble := 30  -- Given that Mary collects the same as Jenny.
def jenny_red_marble := 30 -- Given
def jenny_blue_marble := 25 -- Given
def anie_red_marble := mary_red_marble + 20 -- Anie's red marbles count
def anie_blue_marble := 2 * jenny_blue_marble -- Anie's blue marbles count
def mary_blue_marble := anie_blue_marble / 2 -- Mary's blue marbles count

theorem mary_jenny_red_marble_ratio : 
  mary_red_marble / jenny_red_marble = 1 :=
by
  sorry

end NUMINAMATH_GPT_mary_jenny_red_marble_ratio_l2053_205377


namespace NUMINAMATH_GPT_height_of_cone_l2053_205314

theorem height_of_cone (e : ℝ) (bA : ℝ) (v : ℝ) :
  e = 6 ∧ bA = 54 ∧ v = e^3 → ∃ h : ℝ, (1/3) * bA * h = v ∧ h = 12 := by
  sorry

end NUMINAMATH_GPT_height_of_cone_l2053_205314


namespace NUMINAMATH_GPT_arithmetic_expression_l2053_205368

theorem arithmetic_expression :
  4 * 6 * 8 + 24 / 4 - 2^3 = 190 := by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_l2053_205368


namespace NUMINAMATH_GPT_find_value_of_s_l2053_205393

theorem find_value_of_s
  (a b c w s p : ℕ)
  (h₁ : a + b = w)
  (h₂ : w + c = s)
  (h₃ : s + a = p)
  (h₄ : b + c + p = 16) :
  s = 8 :=
sorry

end NUMINAMATH_GPT_find_value_of_s_l2053_205393


namespace NUMINAMATH_GPT_squirrel_walnut_count_l2053_205330

-- Lean 4 statement
theorem squirrel_walnut_count :
  let initial_boy_walnuts := 12
  let gathered_walnuts := 6
  let dropped_walnuts := 1
  let initial_girl_walnuts := 0
  let brought_walnuts := 5
  let eaten_walnuts := 2
  (initial_boy_walnuts + gathered_walnuts - dropped_walnuts + initial_girl_walnuts + brought_walnuts - eaten_walnuts) = 20 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_squirrel_walnut_count_l2053_205330


namespace NUMINAMATH_GPT_triangle_interior_angle_leq_60_l2053_205317

theorem triangle_interior_angle_leq_60 (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (angle_sum : A + B + C = 180)
  (all_gt_60 : A > 60 ∧ B > 60 ∧ C > 60) :
  false :=
by
  sorry

end NUMINAMATH_GPT_triangle_interior_angle_leq_60_l2053_205317


namespace NUMINAMATH_GPT_correct_proportion_l2053_205362

theorem correct_proportion {a b c x y : ℝ} 
  (h1 : x + y = b)
  (h2 : x * c = y * a) :
  y / a = b / (a + c) :=
sorry

end NUMINAMATH_GPT_correct_proportion_l2053_205362


namespace NUMINAMATH_GPT_polygon_sides_sum_l2053_205398

theorem polygon_sides_sum (triangle_hexagon_sum : ℕ) (triangle_sides : ℕ) (hexagon_sides : ℕ) 
  (h1 : triangle_hexagon_sum = 1260) 
  (h2 : triangle_sides = 3) 
  (h3 : hexagon_sides = 6) 
  (convex : ∀ n, 3 <= n) : 
  triangle_sides + hexagon_sides + 4 = 13 :=
by 
  sorry

end NUMINAMATH_GPT_polygon_sides_sum_l2053_205398


namespace NUMINAMATH_GPT_bags_sold_on_Thursday_l2053_205390

theorem bags_sold_on_Thursday 
    (total_bags : ℕ) (sold_Monday : ℕ) (sold_Tuesday : ℕ) (sold_Wednesday : ℕ) (sold_Friday : ℕ) (percent_not_sold : ℕ) :
    total_bags = 600 →
    sold_Monday = 25 →
    sold_Tuesday = 70 →
    sold_Wednesday = 100 →
    sold_Friday = 145 →
    percent_not_sold = 25 →
    ∃ (sold_Thursday : ℕ), sold_Thursday = 110 :=
by
  sorry

end NUMINAMATH_GPT_bags_sold_on_Thursday_l2053_205390


namespace NUMINAMATH_GPT_mass_percentage_O_in_N2O3_l2053_205351

variable (m_N : ℝ := 14.01)  -- Molar mass of nitrogen (N) in g/mol
variable (m_O : ℝ := 16.00)  -- Molar mass of oxygen (O) in g/mol
variable (n_N : ℕ := 2)      -- Number of nitrogen (N) atoms in N2O3
variable (n_O : ℕ := 3)      -- Number of oxygen (O) atoms in N2O3

theorem mass_percentage_O_in_N2O3 :
  let molar_mass_N2O3 := (n_N * m_N) + (n_O * m_O)
  let mass_O_in_N2O3 := n_O * m_O
  let percentage_O := (mass_O_in_N2O3 / molar_mass_N2O3) * 100
  percentage_O = 63.15 :=
by
  -- Formal proof here
  sorry

end NUMINAMATH_GPT_mass_percentage_O_in_N2O3_l2053_205351


namespace NUMINAMATH_GPT_min_value_of_sum_l2053_205347

theorem min_value_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 4 / x + 1 / y = 1) : x + y = 9 :=
by
  -- sorry used to skip the proof
  sorry

end NUMINAMATH_GPT_min_value_of_sum_l2053_205347


namespace NUMINAMATH_GPT_find_A_B_l2053_205370

theorem find_A_B :
  ∀ (A B : ℝ), (∀ (x : ℝ), 1 < x → ⌊1 / (A * x + B / x)⌋ = 1 / (A * ⌊x⌋ + B / ⌊x⌋)) →
  (A = 0) ∧ (B = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_A_B_l2053_205370


namespace NUMINAMATH_GPT_beth_total_crayons_l2053_205372

theorem beth_total_crayons (packs : ℕ) (crayons_per_pack : ℕ) (extra_crayons : ℕ) 
  (h1 : packs = 8) (h2 : crayons_per_pack = 20) (h3 : extra_crayons = 15) :
  packs * crayons_per_pack + extra_crayons = 175 :=
by
  sorry

end NUMINAMATH_GPT_beth_total_crayons_l2053_205372


namespace NUMINAMATH_GPT_sandy_grew_watermelons_l2053_205318

-- Definitions for the conditions
def jason_grew_watermelons : ℕ := 37
def total_watermelons : ℕ := 48

-- Define what we want to prove
theorem sandy_grew_watermelons : total_watermelons - jason_grew_watermelons = 11 := by
  sorry

end NUMINAMATH_GPT_sandy_grew_watermelons_l2053_205318


namespace NUMINAMATH_GPT_determine_angle_A_l2053_205371

-- Given a triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively
def sin_rule_condition (a b c A B C : ℝ) : Prop :=
  (a + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C

-- The proof statement
theorem determine_angle_A (a b c A B C : ℝ) (h : sin_rule_condition a b c A B C) : A = π / 3 :=
  sorry

end NUMINAMATH_GPT_determine_angle_A_l2053_205371


namespace NUMINAMATH_GPT_replace_asterisk_l2053_205391

theorem replace_asterisk (x : ℕ) (h : (42 / 21) * (42 / x) = 1) : x = 84 := by
  sorry

end NUMINAMATH_GPT_replace_asterisk_l2053_205391


namespace NUMINAMATH_GPT_remainder_when_s_div_6_is_5_l2053_205306

theorem remainder_when_s_div_6_is_5 (s t : ℕ) (h1 : s > t) (Rs Rt : ℕ) (h2 : s % 6 = Rs) (h3 : t % 6 = Rt) (h4 : (s - t) % 6 = 5) : Rs = 5 := 
by
  sorry

end NUMINAMATH_GPT_remainder_when_s_div_6_is_5_l2053_205306


namespace NUMINAMATH_GPT_part_I_part_II_l2053_205367

def f (x : ℝ) (m : ℕ) : ℝ := |x - m| + |x|

theorem part_I (m : ℕ) (hm : m = 1) : ∃ x : ℝ, f x m < 2 :=
by sorry

theorem part_II (α β : ℝ) (hα : 1 < α) (hβ : 1 < β) (h : f α 1 + f β 1 = 2) :
  (4 / α) + (1 / β) ≥ 9 / 2 :=
by sorry

end NUMINAMATH_GPT_part_I_part_II_l2053_205367


namespace NUMINAMATH_GPT_area_of_stripe_l2053_205326

def cylindrical_tank.diameter : ℝ := 40
def cylindrical_tank.height : ℝ := 100
def green_stripe.width : ℝ := 4
def green_stripe.revolutions : ℝ := 3

theorem area_of_stripe :
  let diameter := cylindrical_tank.diameter
  let height := cylindrical_tank.height
  let width := green_stripe.width
  let revolutions := green_stripe.revolutions
  let circumference := Real.pi * diameter
  let length := revolutions * circumference
  let area := length * width
  area = 480 * Real.pi := by
  sorry

end NUMINAMATH_GPT_area_of_stripe_l2053_205326


namespace NUMINAMATH_GPT_resistor_value_l2053_205366

/-- Two resistors with resistance R are connected in series to a DC voltage source U.
    An ideal voltmeter connected in parallel to one resistor shows a reading of 10V.
    The voltmeter is then replaced by an ideal ammeter, which shows a reading of 10A.
    Prove that the resistance R of each resistor is 2Ω. -/
theorem resistor_value (R U U_v I_A : ℝ)
  (hU_v : U_v = 10)
  (hI_A : I_A = 10)
  (hU : U = 2 * U_v)
  (hU_total : U = R * I_A) : R = 2 :=
by
  sorry

end NUMINAMATH_GPT_resistor_value_l2053_205366


namespace NUMINAMATH_GPT_exist_two_numbers_with_GCD_and_LCM_l2053_205340

def GCD (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem exist_two_numbers_with_GCD_and_LCM :
  ∃ A B : ℕ, GCD A B = 21 ∧ LCM A B = 3969 ∧ ((A = 21 ∧ B = 3969) ∨ (A = 147 ∧ B = 567)) :=
by
  sorry

end NUMINAMATH_GPT_exist_two_numbers_with_GCD_and_LCM_l2053_205340


namespace NUMINAMATH_GPT_sufficient_and_necessary_condition_l2053_205305

theorem sufficient_and_necessary_condition {a : ℝ} :
  (∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ a ≥ 4 :=
sorry

end NUMINAMATH_GPT_sufficient_and_necessary_condition_l2053_205305


namespace NUMINAMATH_GPT_calculate_z_l2053_205399

-- Given conditions
def equally_spaced : Prop := true -- assume equally spaced markings do exist
def total_distance : ℕ := 35
def number_of_steps : ℕ := 7
def step_length : ℕ := total_distance / number_of_steps
def starting_point : ℕ := 10
def steps_forward : ℕ := 4

-- Theorem to prove
theorem calculate_z (h1 : equally_spaced)
(h2 : step_length = 5)
: starting_point + (steps_forward * step_length) = 30 :=
by sorry

end NUMINAMATH_GPT_calculate_z_l2053_205399


namespace NUMINAMATH_GPT_value_of_expression_l2053_205312

def delta (a b : ℕ) : ℕ := a * a - b

theorem value_of_expression :
  delta (5 ^ (delta 6 17)) (2 ^ (delta 7 11)) = 5 ^ 38 - 2 ^ 38 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2053_205312


namespace NUMINAMATH_GPT_transition_term_l2053_205349

theorem transition_term (k : ℕ) : (2 * k + 2) + (2 * k + 3) = (2 * (k + 1) + 1) + (2 * k + 2) :=
by
  sorry

end NUMINAMATH_GPT_transition_term_l2053_205349


namespace NUMINAMATH_GPT_gas_tank_size_l2053_205358

-- Conditions from part a)
def advertised_mileage : ℕ := 35
def actual_mileage : ℕ := 31
def total_miles_driven : ℕ := 372

-- Question and the correct answer in the context of conditions
theorem gas_tank_size (h1 : actual_mileage = advertised_mileage - 4) 
                      (h2 : total_miles_driven = 372) 
                      : total_miles_driven / actual_mileage = 12 := 
by sorry

end NUMINAMATH_GPT_gas_tank_size_l2053_205358


namespace NUMINAMATH_GPT_problem_l2053_205361

variable (g : ℝ → ℝ)
variables (x y : ℝ)

noncomputable def cond1 : Prop := ∀ x y : ℝ, 0 < x → 0 < y → g (x^2 * y) = g x / y^2
noncomputable def cond2 : Prop := g 800 = 4

-- The statement to be proved
theorem problem (h1 : cond1 g) (h2 : cond2 g) : g 7200 = 4 / 81 :=
by
  sorry

end NUMINAMATH_GPT_problem_l2053_205361


namespace NUMINAMATH_GPT_arithmetic_sequence_sixtieth_term_l2053_205308

theorem arithmetic_sequence_sixtieth_term (a₁ a₂₁ a₆₀ d : ℕ) 
  (h1 : a₁ = 7)
  (h2 : a₂₁ = 47)
  (h3 : a₂₁ = a₁ + 20 * d) : 
  a₆₀ = a₁ + 59 * d := 
  by
  have HD : d = 2 := by 
    rw [h1] at h3
    rw [h2] at h3
    linarith
  rw [HD]
  rw [h1]
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sixtieth_term_l2053_205308


namespace NUMINAMATH_GPT_max_value_of_expression_l2053_205374

open Real

theorem max_value_of_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x + y + z = 1) : 
  x^4 * y^2 * z ≤ 1024 / 7^7 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l2053_205374


namespace NUMINAMATH_GPT_connie_correct_answer_l2053_205346

theorem connie_correct_answer 
  (x : ℝ) 
  (h1 : 2 * x = 80) 
  (correct_ans : ℝ := x / 3) :
  correct_ans = 40 / 3 :=
by
  sorry

end NUMINAMATH_GPT_connie_correct_answer_l2053_205346


namespace NUMINAMATH_GPT_global_school_math_students_l2053_205375

theorem global_school_math_students (n : ℕ) (h1 : n < 600) (h2 : n % 28 = 27) (h3 : n % 26 = 20) : n = 615 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_global_school_math_students_l2053_205375


namespace NUMINAMATH_GPT_ap_80th_term_l2053_205327

/--
If the sum of the first 20 terms of an arithmetic progression is 200,
and the sum of the first 60 terms is 180, then the 80th term is -573/40.
-/
theorem ap_80th_term (S : ℤ → ℚ) (a d : ℚ)
  (h1 : S 20 = 200)
  (h2 : S 60 = 180)
  (hS : ∀ n, S n = n / 2 * (2 * a + (n - 1) * d)) :
  a + 79 * d = -573 / 40 :=
by {
  sorry
}

end NUMINAMATH_GPT_ap_80th_term_l2053_205327


namespace NUMINAMATH_GPT_bert_bought_300_stamps_l2053_205392

theorem bert_bought_300_stamps (x : ℝ) 
(H1 : x / 2 + x = 450) : x = 300 :=
by
  sorry

end NUMINAMATH_GPT_bert_bought_300_stamps_l2053_205392


namespace NUMINAMATH_GPT_Taso_riddles_correct_l2053_205378

-- Definitions based on given conditions
def Josh_riddles : ℕ := 8
def Ivory_riddles : ℕ := Josh_riddles + 4
def Taso_riddles : ℕ := 2 * Ivory_riddles

-- The theorem to prove
theorem Taso_riddles_correct : Taso_riddles = 24 := by
  sorry

end NUMINAMATH_GPT_Taso_riddles_correct_l2053_205378


namespace NUMINAMATH_GPT_non_powers_of_a_meet_condition_l2053_205310

-- Definitions used directly from the conditions detailed in the problem:
def Sa (a x : ℕ) : ℕ := sorry -- S_{a}(x): sum of the digits of x in base a
def Fa (a x : ℕ) : ℕ := sorry -- F_{a}(x): number of digits of x in base a
def fa (a x : ℕ) : ℕ := sorry -- f_{a}(x): position of the first non-zero digit from the right in base a

theorem non_powers_of_a_meet_condition (a M : ℕ) (h₁: a > 1) (h₂ : M ≥ 2020) :
  ∀ n : ℕ, (n > 0) → (∀ k : ℕ, (k > 0) → (Sa a (k * n) = Sa a n ∧ Fa a (k * n) - fa a (k * n) > M)) ↔ (∃ α : ℕ, n = a ^ α) :=
sorry

end NUMINAMATH_GPT_non_powers_of_a_meet_condition_l2053_205310


namespace NUMINAMATH_GPT_total_toothpicks_correct_l2053_205384

-- Define the number of vertical lines and toothpicks in them
def num_vertical_lines : ℕ := 41
def num_toothpicks_per_vertical_line : ℕ := 20
def vertical_toothpicks : ℕ := num_vertical_lines * num_toothpicks_per_vertical_line

-- Define the number of horizontal lines and toothpicks in them
def num_horizontal_lines : ℕ := 21
def num_toothpicks_per_horizontal_line : ℕ := 40
def horizontal_toothpicks : ℕ := num_horizontal_lines * num_toothpicks_per_horizontal_line

-- Define the dimensions of the triangle
def triangle_base : ℕ := 20
def triangle_height : ℕ := 20
def triangle_hypotenuse : ℕ := 29 -- approximated

-- Total toothpicks in the triangle
def triangle_toothpicks : ℕ := triangle_height + triangle_hypotenuse

-- Total toothpicks used in the structure
def total_toothpicks : ℕ := vertical_toothpicks + horizontal_toothpicks + triangle_toothpicks

-- Theorem to prove the total number of toothpicks used is 1709
theorem total_toothpicks_correct : total_toothpicks = 1709 := by
  sorry

end NUMINAMATH_GPT_total_toothpicks_correct_l2053_205384


namespace NUMINAMATH_GPT_impossible_coins_l2053_205339

theorem impossible_coins (p1 p2 : ℝ) : 
  ¬ ((1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * p2 = p1 * (1 - p2) + p2 * (1 - p1)) :=
by
  sorry

end NUMINAMATH_GPT_impossible_coins_l2053_205339


namespace NUMINAMATH_GPT_max_coconuts_needed_l2053_205307

theorem max_coconuts_needed (goats : ℕ) (coconuts_per_crab : ℕ) (crabs_per_goat : ℕ) 
  (final_goats : ℕ) : 
  goats = 19 ∧ coconuts_per_crab = 3 ∧ crabs_per_goat = 6 →
  ∃ coconuts, coconuts = 342 :=
by
  sorry

end NUMINAMATH_GPT_max_coconuts_needed_l2053_205307


namespace NUMINAMATH_GPT_polygon_sides_l2053_205341

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1080) : n = 8 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l2053_205341


namespace NUMINAMATH_GPT_complex_number_quadrant_l2053_205331

noncomputable def complex_quadrant : ℂ → String
| z => if z.re > 0 ∧ z.im > 0 then "First quadrant"
      else if z.re < 0 ∧ z.im > 0 then "Second quadrant"
      else if z.re < 0 ∧ z.im < 0 then "Third quadrant"
      else if z.re > 0 ∧ z.im < 0 then "Fourth quadrant"
      else "On the axis"

theorem complex_number_quadrant (z : ℂ) (h : z = (5 : ℂ) / (2 + I)) : complex_quadrant z = "Fourth quadrant" :=
by
  sorry

end NUMINAMATH_GPT_complex_number_quadrant_l2053_205331


namespace NUMINAMATH_GPT_remainder_is_zero_l2053_205335

theorem remainder_is_zero :
  (86 * 87 * 88 * 89 * 90 * 91 * 92) % 7 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_is_zero_l2053_205335


namespace NUMINAMATH_GPT_find_d_l2053_205304

noncomputable def area_triangle (base height : ℝ) : ℝ := (1 / 2) * base * height

theorem find_d (d : ℝ) (h₁ : 0 ≤ d ∧ d ≤ 2) (h₂ : 6 - ((1 / 2) * (2 - d) * 2) = 2 * ((1 / 2) * (2 - d) * 2)) : 
  d = 0 :=
sorry

end NUMINAMATH_GPT_find_d_l2053_205304


namespace NUMINAMATH_GPT_find_a_l2053_205338

noncomputable def unique_quad_solution (a : ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1^2 - a * x1 + a = 1 → x2^2 - a * x2 + a = 1 → x1 = x2

theorem find_a (a : ℝ) (h : unique_quad_solution a) : a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_l2053_205338


namespace NUMINAMATH_GPT_wage_difference_l2053_205394

noncomputable def manager_wage : ℝ := 8.50
noncomputable def dishwasher_wage : ℝ := manager_wage / 2
noncomputable def chef_wage : ℝ := dishwasher_wage * 1.2

theorem wage_difference : manager_wage - chef_wage = 3.40 := 
by
  sorry

end NUMINAMATH_GPT_wage_difference_l2053_205394


namespace NUMINAMATH_GPT_find_L_l2053_205386

theorem find_L (RI G SP T M N : ℝ) (h1 : RI + G + SP = 50) (h2 : RI + T + M = 63) (h3 : G + T + SP = 25) 
(h4 : SP + M = 13) (h5 : M + RI = 48) (h6 : N = 1) :
  ∃ L : ℝ, L * M * T + SP * RI * N * G = 2023 ∧ L = 341 / 40 := 
by
  sorry

end NUMINAMATH_GPT_find_L_l2053_205386


namespace NUMINAMATH_GPT_abs_neg_one_div_three_l2053_205316

open Real

theorem abs_neg_one_div_three : abs (-1 / 3) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_abs_neg_one_div_three_l2053_205316


namespace NUMINAMATH_GPT_geometric_sequence_strictly_increasing_iff_l2053_205337

noncomputable def geometric_sequence (a_1 q : ℝ) (n : ℕ) : ℝ :=
  a_1 * q^(n-1)

theorem geometric_sequence_strictly_increasing_iff (a_1 q : ℝ) :
  (∀ n : ℕ, geometric_sequence a_1 q (n+2) > geometric_sequence a_1 q n) ↔ 
  (∀ n : ℕ, geometric_sequence a_1 q (n+1) > geometric_sequence a_1 q n) := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_strictly_increasing_iff_l2053_205337


namespace NUMINAMATH_GPT_find_taller_tree_height_l2053_205300

-- Define the known variables and conditions
variables (H : ℕ) (ratio : ℚ) (difference : ℕ)

-- Specify the conditions from the problem
def taller_tree_height (H difference : ℕ) := H
def shorter_tree_height (H difference : ℕ) := H - difference
def height_ratio (H : ℕ) (ratio : ℚ) (difference : ℕ) :=
  (shorter_tree_height H difference : ℚ) / (taller_tree_height H difference : ℚ) = ratio

-- Prove the height of the taller tree given the conditions
theorem find_taller_tree_height (H : ℕ) (h_ratio : height_ratio H (2/3) 20) : 
  taller_tree_height H 20 = 60 :=
  sorry

end NUMINAMATH_GPT_find_taller_tree_height_l2053_205300


namespace NUMINAMATH_GPT_koala_fiber_l2053_205364

theorem koala_fiber (absorption_percent: ℝ) (absorbed_fiber: ℝ) (total_fiber: ℝ) 
  (h1: absorption_percent = 0.25) 
  (h2: absorbed_fiber = 10.5) 
  (h3: absorbed_fiber = absorption_percent * total_fiber) : 
  total_fiber = 42 :=
by
  rw [h1, h2] at h3
  have h : 10.5 = 0.25 * total_fiber := h3
  sorry

end NUMINAMATH_GPT_koala_fiber_l2053_205364


namespace NUMINAMATH_GPT_cost_of_dowels_l2053_205385

variable (V S : ℝ)

theorem cost_of_dowels 
  (hV : V = 7)
  (h_eq : 0.85 * (V + S) = V + 0.5 * S) :
  S = 3 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_dowels_l2053_205385


namespace NUMINAMATH_GPT_product_of_solutions_l2053_205303

theorem product_of_solutions (x : ℝ) (hx : |x - 5| - 5 = 0) :
  ∃ a b : ℝ, (|a - 5| - 5 = 0 ∧ |b - 5| - 5 = 0) ∧ a * b = 0 := by
  sorry

end NUMINAMATH_GPT_product_of_solutions_l2053_205303


namespace NUMINAMATH_GPT_monomial_exponents_l2053_205325

theorem monomial_exponents (m n : ℕ) 
  (h1 : m + 1 = 3)
  (h2 : n - 1 = 3) : 
  m^n = 16 := by
  sorry

end NUMINAMATH_GPT_monomial_exponents_l2053_205325


namespace NUMINAMATH_GPT_avg_salary_officers_correct_l2053_205324

def total_employees := 465
def avg_salary_employees := 120
def non_officers := 450
def avg_salary_non_officers := 110
def officers := 15

theorem avg_salary_officers_correct : (15 * 420) = ((total_employees * avg_salary_employees) - (non_officers * avg_salary_non_officers)) := by
  sorry

end NUMINAMATH_GPT_avg_salary_officers_correct_l2053_205324


namespace NUMINAMATH_GPT_height_percentage_increase_l2053_205365

theorem height_percentage_increase (B A : ℝ) (h : A = B - 0.3 * B) : 
  ((B - A) / A) * 100 = 42.857 :=
by
  sorry

end NUMINAMATH_GPT_height_percentage_increase_l2053_205365


namespace NUMINAMATH_GPT_product_ge_one_l2053_205369

variable (a b : ℝ)
variable (x1 x2 x3 x4 x5 : ℝ)

theorem product_ge_one
  (ha : 0 < a)
  (hb : 0 < b)
  (h_ab : a + b = 1)
  (hx1 : 0 < x1)
  (hx2 : 0 < x2)
  (hx3 : 0 < x3)
  (hx4 : 0 < x4)
  (hx5 : 0 < x5)
  (h_prod_xs : x1 * x2 * x3 * x4 * x5 = 1) :
  (a * x1 + b) * (a * x2 + b) * (a * x3 + b) * (a * x4 + b) * (a * x5 + b) ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_product_ge_one_l2053_205369


namespace NUMINAMATH_GPT_positive_difference_volumes_l2053_205380

open Real

noncomputable def charlies_height := 12
noncomputable def charlies_circumference := 10
noncomputable def danas_height := 8
noncomputable def danas_circumference := 10

theorem positive_difference_volumes (hC : ℝ := charlies_height) (CC : ℝ := charlies_circumference)
                                   (hD : ℝ := danas_height) (CD : ℝ := danas_circumference) :
    (π * (π * ((CD / (2 * π)) ^ 2) * hD - π * ((CC / (2 * π)) ^ 2) * hC)) = 100 :=
by
  have rC := CC / (2 * π)
  have VC := π * (rC ^ 2) * hC
  have rD := CD / (2 * π)
  have VD := π * (rD ^ 2) * hD
  sorry

end NUMINAMATH_GPT_positive_difference_volumes_l2053_205380


namespace NUMINAMATH_GPT_find_f_2010_l2053_205348

open Nat

variable (f : ℕ → ℕ)

axiom strictly_increasing : ∀ m n : ℕ, m < n → f m < f n

axiom function_condition : ∀ n : ℕ, f (f n) = 3 * n

theorem find_f_2010 : f 2010 = 3015 := sorry

end NUMINAMATH_GPT_find_f_2010_l2053_205348


namespace NUMINAMATH_GPT_sum_of_consecutive_odds_mod_16_l2053_205301

theorem sum_of_consecutive_odds_mod_16 :
  (12001 + 12003 + 12005 + 12007 + 12009 + 12011 + 12013) % 16 = 1 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_odds_mod_16_l2053_205301


namespace NUMINAMATH_GPT_circle_center_radius_l2053_205360

theorem circle_center_radius (x y : ℝ) :
  x^2 + y^2 - 4 * x + 2 * y - 4 = 0 ↔ (x - 2)^2 + (y + 1)^2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_circle_center_radius_l2053_205360


namespace NUMINAMATH_GPT_trig_problem_l2053_205383

theorem trig_problem (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 4 :=
by
  sorry

end NUMINAMATH_GPT_trig_problem_l2053_205383


namespace NUMINAMATH_GPT_part1_part2_l2053_205313

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + |2 * x + 3|

theorem part1 (x : ℝ) : f x ≥ 6 ↔ x ≥ 1 ∨ x ≤ -2 := by
  sorry

theorem part2 (a b : ℝ) (m : ℝ) 
  (a_pos : a > 0) (b_pos : b > 0) 
  (fmin : m = 4) 
  (condition : 2 * a * b + a + 2 * b = m) : 
  a + 2 * b = 2 * Real.sqrt 5 - 2 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l2053_205313


namespace NUMINAMATH_GPT_weight_of_new_person_l2053_205343

-- Definition of the problem
def average_weight_increases (W : ℝ) (N : ℝ) : Prop :=
  let increase := 2.5
  W - 45 + N = W + 8 * increase

-- The main statement we need to prove
theorem weight_of_new_person (W : ℝ) : ∃ N, average_weight_increases W N ∧ N = 65 := 
by
  use 65
  unfold average_weight_increases
  sorry

end NUMINAMATH_GPT_weight_of_new_person_l2053_205343


namespace NUMINAMATH_GPT_sin_270_eq_neg_one_l2053_205302

theorem sin_270_eq_neg_one : Real.sin (270 * Real.pi / 180) = -1 := 
by
  sorry

end NUMINAMATH_GPT_sin_270_eq_neg_one_l2053_205302


namespace NUMINAMATH_GPT_proof_no_solution_l2053_205309

noncomputable def no_solution (a b : ℕ) : Prop :=
  2 * a^2 + 1 ≠ 4 * b^2

theorem proof_no_solution (a b : ℕ) : no_solution a b := by
  sorry

end NUMINAMATH_GPT_proof_no_solution_l2053_205309


namespace NUMINAMATH_GPT_birds_problem_l2053_205388

-- Define the initial number of birds and the total number of birds as given conditions.
def initial_birds : ℕ := 2
def total_birds : ℕ := 6

-- Define the number of new birds that came to join.
def new_birds : ℕ := total_birds - initial_birds

-- State the theorem to be proved, asserting that the number of new birds is 4.
theorem birds_problem : new_birds = 4 := 
by
  -- required proof goes here
  sorry

end NUMINAMATH_GPT_birds_problem_l2053_205388


namespace NUMINAMATH_GPT_correct_statements_l2053_205382

variable (a_1 a_2 b_1 b_2 : ℝ)

def ellipse1 := ∀ x y : ℝ, x^2 / a_1^2 + y^2 / b_1^2 = 1
def ellipse2 := ∀ x y : ℝ, x^2 / a_2^2 + y^2 / b_2^2 = 1

axiom a1_pos : a_1 > 0
axiom b1_pos : b_1 > 0
axiom a2_gt_b2_pos : a_2 > b_2 ∧ b_2 > 0
axiom same_foci : a_1^2 - b_1^2 = a_2^2 - b_2^2
axiom a1_gt_a2 : a_1 > a_2

theorem correct_statements : 
  (¬(∃ x y, (x^2 / a_1^2 + y^2 / b_1^2 = 1) ∧ (x^2 / a_2^2 + y^2 / b_2^2 = 1))) ∧ 
  (a_1^2 - a_2^2 = b_1^2 - b_2^2) :=
by 
  sorry

end NUMINAMATH_GPT_correct_statements_l2053_205382


namespace NUMINAMATH_GPT_solution_form_l2053_205334

noncomputable def required_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) ≤ (x * f y + y * f x) / 2

theorem solution_form (f : ℝ → ℝ) (h : ∀ x : ℝ, 0 < x → 0 < f x) : required_function f → ∃ a : ℝ, 0 < a ∧ ∀ x : ℝ, 0 < x → f x = a * x :=
by
  intros
  sorry

end NUMINAMATH_GPT_solution_form_l2053_205334


namespace NUMINAMATH_GPT_scientists_nobel_greater_than_not_nobel_by_three_l2053_205363

-- Definitions of the given conditions
def total_scientists := 50
def wolf_prize_laureates := 31
def nobel_prize_laureates := 25
def wolf_and_nobel_laureates := 14

-- Derived quantities
def no_wolf_prize := total_scientists - wolf_prize_laureates
def only_wolf_prize := wolf_prize_laureates - wolf_and_nobel_laureates
def only_nobel_prize := nobel_prize_laureates - wolf_and_nobel_laureates
def nobel_no_wolf := only_nobel_prize
def no_wolf_no_nobel := no_wolf_prize - nobel_no_wolf
def difference := nobel_no_wolf - no_wolf_no_nobel

-- The theorem to be proved
theorem scientists_nobel_greater_than_not_nobel_by_three :
  difference = 3 := 
sorry

end NUMINAMATH_GPT_scientists_nobel_greater_than_not_nobel_by_three_l2053_205363


namespace NUMINAMATH_GPT_khalil_dogs_l2053_205355

theorem khalil_dogs (D : ℕ) (cost_dog cost_cat : ℕ) (num_cats total_cost : ℕ) 
  (h1 : cost_dog = 60)
  (h2 : cost_cat = 40)
  (h3 : num_cats = 60)
  (h4 : total_cost = 3600) :
  (num_cats * cost_cat + D * cost_dog = total_cost) → D = 20 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_khalil_dogs_l2053_205355


namespace NUMINAMATH_GPT_no_nat_x_y_square_l2053_205387

theorem no_nat_x_y_square (x y : ℕ) : ¬(∃ a b : ℕ, x^2 + y = a^2 ∧ y^2 + x = b^2) := 
by 
  sorry

end NUMINAMATH_GPT_no_nat_x_y_square_l2053_205387


namespace NUMINAMATH_GPT_gcd_of_720_120_168_is_24_l2053_205396

theorem gcd_of_720_120_168_is_24 : Int.gcd (Int.gcd 720 120) 168 = 24 := 
by sorry

end NUMINAMATH_GPT_gcd_of_720_120_168_is_24_l2053_205396


namespace NUMINAMATH_GPT_area_of_shaded_region_l2053_205395

noncomputable def shaded_area (length_in_feet : ℝ) (diameter : ℝ) : ℝ :=
  let length_in_inches := length_in_feet * 12
  let radius := diameter / 2
  let num_semicircles := length_in_inches / diameter
  let num_full_circles := num_semicircles / 2
  let area := num_full_circles * (radius ^ 2 * Real.pi)
  area

theorem area_of_shaded_region : shaded_area 1.5 3 = 13.5 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_area_of_shaded_region_l2053_205395


namespace NUMINAMATH_GPT_positive_difference_of_two_numbers_l2053_205357

theorem positive_difference_of_two_numbers :
  ∃ x y : ℚ, x + y = 40 ∧ 3 * y - 4 * x = 20 ∧ y - x = 80 / 7 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_of_two_numbers_l2053_205357


namespace NUMINAMATH_GPT_area_of_square_field_l2053_205397

theorem area_of_square_field (x : ℝ) 
  (h₁ : 1.10 * (4 * x - 2) = 732.6) : 
  x = 167 → x ^ 2 = 27889 := by
  sorry

end NUMINAMATH_GPT_area_of_square_field_l2053_205397


namespace NUMINAMATH_GPT_intersection_l2053_205376

namespace Proof

def A := {x : ℝ | 0 ≤ x ∧ x ≤ 6}
def B := {x : ℝ | 3 * x^2 + x - 8 ≤ 0}

theorem intersection (x : ℝ) : x ∈ A ∩ B ↔ 0 ≤ x ∧ x ≤ (4:ℝ)/3 := 
by 
  sorry  -- proof placeholder

end Proof

end NUMINAMATH_GPT_intersection_l2053_205376


namespace NUMINAMATH_GPT_simplify_expression_l2053_205315

noncomputable def term1 : ℝ := 3 / (Real.sqrt 2 + 2)
noncomputable def term2 : ℝ := 4 / (Real.sqrt 5 - 2)
noncomputable def simplifiedExpression : ℝ := 1 / (term1 + term2)
noncomputable def finalExpression : ℝ := 1 / (11 + 4 * Real.sqrt 5 - 3 * Real.sqrt 2 / 2)

theorem simplify_expression : simplifiedExpression = finalExpression := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2053_205315
