import Mathlib

namespace parabola_origin_l106_106060

theorem parabola_origin (x y c : ℝ) (h : y = x^2 - 2 * x + c - 4) (h0 : (0, 0) = (x, y)) : c = 4 :=
by
  sorry

end parabola_origin_l106_106060


namespace Morgan_first_SAT_score_l106_106399

variable (S : ℝ) -- Morgan's first SAT score
variable (improved_score : ℝ := 1100) -- Improved score on second attempt
variable (improvement_rate : ℝ := 0.10) -- Improvement rate

theorem Morgan_first_SAT_score:
  improved_score = S * (1 + improvement_rate) → S = 1000 := 
by 
  sorry

end Morgan_first_SAT_score_l106_106399


namespace Zlatoust_to_Miass_distance_l106_106766

theorem Zlatoust_to_Miass_distance
  (x g k m : ℝ)
  (H1 : (x + 18) / k = (x - 18) / m)
  (H2 : (x + 25) / k = (x - 25) / g)
  (H3 : (x + 8) / m = (x - 8) / g) :
  x = 60 :=
sorry

end Zlatoust_to_Miass_distance_l106_106766


namespace chemistry_textbook_weight_l106_106775

theorem chemistry_textbook_weight (G C : ℝ) (h1 : G = 0.62) (h2 : C = G + 6.5) : C = 7.12 :=
by
  sorry

end chemistry_textbook_weight_l106_106775


namespace p_and_q_necessary_but_not_sufficient_l106_106627

theorem p_and_q_necessary_but_not_sufficient (p q : Prop) :
  ((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q)) := 
by 
  sorry

end p_and_q_necessary_but_not_sufficient_l106_106627


namespace amount_of_loan_l106_106620

theorem amount_of_loan (P R T SI : ℝ) (hR : R = 6) (hT : T = 6) (hSI : SI = 432) :
  SI = (P * R * T) / 100 → P = 1200 :=
by
  intro h
  sorry

end amount_of_loan_l106_106620


namespace moles_of_BeOH2_l106_106402

-- Definitions based on the given conditions
def balanced_chemical_equation (xBe2C xH2O xBeOH2 xCH4 : ℕ) : Prop :=
  xBe2C = 1 ∧ xH2O = 4 ∧ xBeOH2 = 2 ∧ xCH4 = 1

def initial_conditions (yBe2C yH2O : ℕ) : Prop :=
  yBe2C = 1 ∧ yH2O = 4

-- Lean statement to prove the number of moles of Beryllium hydroxide formed
theorem moles_of_BeOH2 (xBe2C xH2O xBeOH2 xCH4 yBe2C yH2O : ℕ) (h1 : balanced_chemical_equation xBe2C xH2O xBeOH2 xCH4) (h2 : initial_conditions yBe2C yH2O) :
  xBeOH2 = 2 :=
by
  sorry

end moles_of_BeOH2_l106_106402


namespace cone_height_l106_106753

theorem cone_height (l : ℝ) (A : ℝ) (h : ℝ) (r : ℝ) 
  (h_slant_height : l = 13)
  (h_lateral_area : A = 65 * π)
  (h_radius : r = 5)
  (h_height_formula : h = Real.sqrt (l^2 - r^2)) : 
  h = 12 := 
by 
  sorry

end cone_height_l106_106753


namespace recycling_drive_target_l106_106268

-- Define the collection totals for each section
def section_collections_first_week : List ℝ := [260, 290, 250, 270, 300, 310, 280, 265]

-- Compute total collection for the first week
def total_first_week (collections: List ℝ) : ℝ := collections.sum

-- Compute collection for the second week with a 10% increase
def second_week_increase (collection: ℝ) : ℝ := collection * 1.10
def total_second_week (collections: List ℝ) : ℝ := (collections.map second_week_increase).sum

-- Compute collection for the third week with a 30% increase from the second week
def third_week_increase (collection: ℝ) : ℝ := collection * 1.30
def total_third_week (collections: List ℝ) : ℝ := (collections.map (second_week_increase)).sum * 1.30

-- Total target collection is the sum of collections for three weeks
def target (collections: List ℝ) : ℝ := total_first_week collections + total_second_week collections + total_third_week collections

-- Main theorem to prove
theorem recycling_drive_target : target section_collections_first_week = 7854.25 :=
by
  sorry -- skipping the proof

end recycling_drive_target_l106_106268


namespace energy_stick_difference_l106_106783

variable (B D : ℕ)

theorem energy_stick_difference (h1 : B = D + 17) : 
  let B' := B - 3
  let D' := D + 3
  D' < B' →
  (B' - D') = 11 :=
by
  sorry

end energy_stick_difference_l106_106783


namespace total_shaded_area_l106_106857

theorem total_shaded_area (S T : ℝ) (h1 : 16 / S = 4) (h2 : S / T = 4) : 
    S^2 + 16 * T^2 = 32 := 
by {
    sorry
}

end total_shaded_area_l106_106857


namespace part1_part2_l106_106150

-- Defining the function f(x) and the given conditions
def f (x a : ℝ) := x^2 - a * x + 2 * a - 2

-- Given conditions
variables (a : ℝ)
axiom f_condition : ∀ (x : ℝ), f (2 + x) a * f (2 - x) a = 4
axiom a_gt_0 : a > 0
axiom fx_bounds : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 4 → 1 ≤ f x a ∧ f x a ≤ 3

-- To prove (part 1)
theorem part1 (h : f 2 a + f 3 a = 6) : a = 2 := sorry

-- To prove (part 2)
theorem part2 : (4 - (2 * Real.sqrt 6) / 3) ≤ a ∧ a ≤ 5 / 2 := sorry

end part1_part2_l106_106150


namespace units_digit_sum_l106_106804

def base8_to_base10 (n : Nat) : Nat :=
  let units := n % 10
  let tens := (n / 10) % 10
  tens * 8 + units

theorem units_digit_sum (n1 n2 : Nat) (h1 : n1 = 45) (h2 : n2 = 67) : ((base8_to_base10 n1) + (base8_to_base10 n2)) % 8 = 4 := by
  sorry

end units_digit_sum_l106_106804


namespace stock_percentage_change_l106_106536

theorem stock_percentage_change :
  let initial_value := 100
  let value_after_first_day := initial_value * (1 - 0.25)
  let value_after_second_day := value_after_first_day * (1 + 0.35)
  let final_value := value_after_second_day * (1 - 0.15)
  let overall_percentage_change := ((final_value - initial_value) / initial_value) * 100
  overall_percentage_change = -13.9375 := 
by
  sorry

end stock_percentage_change_l106_106536


namespace num_isosceles_right_triangles_in_ellipse_l106_106871

theorem num_isosceles_right_triangles_in_ellipse
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (ellipse_eq : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 ↔ ∃ t : ℝ, (x, y) = (a * Real.cos t, b * Real.sin t))
  :
  (∃ n : ℕ,
    (n = 3 ∧ a > Real.sqrt 3 * b) ∨
    (n = 1 ∧ (b < a ∧ a ≤ Real.sqrt 3 * b))
  ) :=
sorry

end num_isosceles_right_triangles_in_ellipse_l106_106871


namespace only_integer_solution_l106_106330

theorem only_integer_solution (a b c d : ℤ) (h : a^2 + b^2 = 3 * (c^2 + d^2)) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := 
by
  sorry

end only_integer_solution_l106_106330


namespace middle_school_students_count_l106_106328

def split_equally (m h : ℕ) : Prop := m = h
def percent_middle (M m : ℕ) : Prop := m = M / 5
def percent_high (H h : ℕ) : Prop := h = 3 * H / 10
def total_students (M H : ℕ) : Prop := M + H = 50
def number_of_middle_school_students (M: ℕ) := M

theorem middle_school_students_count (M H m h : ℕ) 
  (hm_eq : split_equally m h) 
  (hm_percent : percent_middle M m) 
  (hh_percent : percent_high H h) 
  (htotal : total_students M H) : 
  number_of_middle_school_students M = 30 :=
by
  sorry

end middle_school_students_count_l106_106328


namespace box_count_neither_markers_nor_erasers_l106_106948

-- Define the conditions as parameters.
def total_boxes : ℕ := 15
def markers_count : ℕ := 10
def erasers_count : ℕ := 5
def both_count : ℕ := 4

-- State the theorem to be proven in Lean 4.
theorem box_count_neither_markers_nor_erasers : 
  total_boxes - (markers_count + erasers_count - both_count) = 4 := 
sorry

end box_count_neither_markers_nor_erasers_l106_106948


namespace packet_weight_l106_106052

theorem packet_weight :
  ∀ (num_packets : ℕ) (total_weight_kg : ℕ), 
  num_packets = 20 → total_weight_kg = 2 →
  (total_weight_kg * 1000) / num_packets = 100 := by
  intro num_packets total_weight_kg h1 h2
  sorry

end packet_weight_l106_106052


namespace complement_U_A_l106_106676

-- Define the sets U and A
def U : Set Int := {-1, 0, 1, 2}
def A : Set Int := {-1, 1, 2}

-- State the theorem
theorem complement_U_A :
  U \ A = {0} :=
sorry

end complement_U_A_l106_106676


namespace counting_numbers_dividing_48_with_remainder_7_l106_106580

theorem counting_numbers_dividing_48_with_remainder_7 :
  ∃ (S : Finset ℕ), S.card = 5 ∧ ∀ n ∈ S, n > 7 ∧ 48 % n = 0 :=
by
  sorry

end counting_numbers_dividing_48_with_remainder_7_l106_106580


namespace snowflake_stamps_count_l106_106155

theorem snowflake_stamps_count (S : ℕ) (truck_stamps : ℕ) (rose_stamps : ℕ) :
  truck_stamps = S + 9 →
  rose_stamps = S + 9 - 13 →
  S + truck_stamps + rose_stamps = 38 →
  S = 11 :=
by
  intros h1 h2 h3
  sorry

end snowflake_stamps_count_l106_106155


namespace claire_speed_l106_106619

def distance := 2067
def time := 39

def speed (d : ℕ) (t : ℕ) : ℕ := d / t

theorem claire_speed : speed distance time = 53 := by
  sorry

end claire_speed_l106_106619


namespace quadratic_equality_l106_106361

theorem quadratic_equality (x : ℝ) 
  (h : 14*x + 5 - 21*x^2 = -2) : 
  6*x^2 - 4*x + 5 = 7 := 
by
  sorry

end quadratic_equality_l106_106361


namespace travel_time_is_correct_l106_106767

-- Define the conditions
def speed : ℕ := 60 -- Speed in km/h
def distance : ℕ := 120 -- Distance between points A and B in km

-- Time calculation from A to B 
def time_AB : ℕ := distance / speed

-- Time calculation from B to A (since speed and distance are the same)
def time_BA : ℕ := distance / speed

-- Total time calculation
def total_time : ℕ := time_AB + time_BA

-- The proper statement to prove
theorem travel_time_is_correct : total_time = 4 := by
  -- Additional steps and arguments would go here
  -- skipping proof
  sorry

end travel_time_is_correct_l106_106767


namespace angle_C_is_100_l106_106101

-- Define the initial measures in the equilateral triangle
def initial_angle (A B C : ℕ) (h_equilateral : A = B ∧ B = C ∧ C = 60) : ℕ := C

-- Definition to capture the increase in angle C
def increased_angle (C : ℕ) : ℕ := C + 40

-- Now, we need to state the theorem assuming the given conditions
theorem angle_C_is_100
  (A B C : ℕ)
  (h_equilateral : A = 60 ∧ B = 60 ∧ C = 60)
  (h_increase : C = 60 + 40)
  : C = 100 := 
sorry

end angle_C_is_100_l106_106101


namespace final_student_count_l106_106990

def initial_students := 150
def students_joined := 30
def students_left := 15

theorem final_student_count : initial_students + students_joined - students_left = 165 := by
  sorry

end final_student_count_l106_106990


namespace buying_pets_l106_106963

theorem buying_pets {puppies kittens hamsters birds : ℕ} :
(∃ pets : ℕ, pets = 12 * 8 * 10 * 5 * 4 * 3 * 2) ∧ 
puppies = 12 ∧ kittens = 8 ∧ hamsters = 10 ∧ birds = 5 → 
12 * 8 * 10 * 5 * 4 * 3 * 2 = 115200 :=
by
  intros h
  sorry

end buying_pets_l106_106963


namespace appropriate_sampling_method_l106_106583

theorem appropriate_sampling_method
  (total_students : ℕ)
  (male_students : ℕ)
  (female_students : ℕ)
  (survey_size : ℕ)
  (diff_interests : Prop)
  (h1 : total_students = 1000)
  (h2 : male_students = 500)
  (h3 : female_students = 500)
  (h4 : survey_size = 100)
  (h5 : diff_interests) : 
  sampling_method = "stratified sampling" :=
by
  sorry

end appropriate_sampling_method_l106_106583


namespace total_floor_area_covered_l106_106378

theorem total_floor_area_covered (combined_area : ℝ) (area_two_layers : ℝ) (area_three_layers : ℝ) : 
  combined_area = 200 → 
  area_two_layers = 22 → 
  area_three_layers = 19 → 
  (combined_area - (area_two_layers + 2 * area_three_layers)) = 140 := 
by
  sorry

end total_floor_area_covered_l106_106378


namespace number_of_zeros_of_g_is_4_l106_106367

noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then x + 1/x else Real.log x

noncomputable def g (x : ℝ) : ℝ := 
  f (f x + 2) + 2

theorem number_of_zeros_of_g_is_4 : 
  ∃ S : Finset ℝ, S.card = 4 ∧ ∀ x ∈ S, g x = 0 :=
sorry

end number_of_zeros_of_g_is_4_l106_106367


namespace part_a_possible_final_number_l106_106158

theorem part_a_possible_final_number :
  ∃ (n : ℕ), n = 97 ∧ 
  (∃ f : {x // x ≠ 0} → ℕ → ℕ, 
    f ⟨1, by decide⟩ 0 = 1 ∧ 
    f ⟨2, by decide⟩ 1 = 2 ∧ 
    f ⟨4, by decide⟩ 2 = 4 ∧ 
    f ⟨8, by decide⟩ 3 = 8 ∧ 
    f ⟨16, by decide⟩ 4 = 16 ∧ 
    f ⟨32, by decide⟩ 5 = 32 ∧ 
    f ⟨64, by decide⟩ 6 = 64 ∧ 
    f ⟨128, by decide⟩ 7 = 128 ∧ 
    ∀ i j : {x // x ≠ 0}, f i j = (f i j - f i j)) := sorry

end part_a_possible_final_number_l106_106158


namespace problem_statement_l106_106454

theorem problem_statement (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (D : ℕ) (M : ℕ) (h_gcd : D = Nat.gcd (Nat.gcd a b) c) (h_lcm : M = Nat.lcm (Nat.lcm a b) c) :
  ((D * M = a * b * c) ∧ ((Nat.gcd a b = 1) ∧ (Nat.gcd b c = 1) ∧ (Nat.gcd a c = 1) → (D * M = a * b * c))) :=
by sorry

end problem_statement_l106_106454


namespace harry_blue_weights_l106_106680

theorem harry_blue_weights (B : ℕ) 
  (h1 : 2 * B + 17 = 25) : B = 4 :=
by {
  -- proof code here
  sorry
}

end harry_blue_weights_l106_106680


namespace education_expenses_l106_106082

theorem education_expenses (rent milk groceries petrol miscellaneous savings total_salary education : ℝ) 
  (h_rent : rent = 5000)
  (h_milk : milk = 1500)
  (h_groceries : groceries = 4500)
  (h_petrol : petrol = 2000)
  (h_miscellaneous : miscellaneous = 6100)
  (h_savings : savings = 2400)
  (h_saving_percentage : savings = 0.10 * total_salary)
  (h_total_salary : total_salary = savings / 0.10)
  (h_total_expenses : total_salary - savings = rent + milk + groceries + petrol + miscellaneous + education) :
  education = 2500 :=
by
  sorry

end education_expenses_l106_106082


namespace isosceles_base_length_l106_106538

theorem isosceles_base_length :
  ∀ (equilateral_perimeter isosceles_perimeter side_length base_length : ℕ), 
  equilateral_perimeter = 60 →  -- Condition: Perimeter of the equilateral triangle is 60
  isosceles_perimeter = 45 →    -- Condition: Perimeter of the isosceles triangle is 45
  side_length = equilateral_perimeter / 3 →   -- Condition: Each side of the equilateral triangle
  isosceles_perimeter = side_length + side_length + base_length  -- Condition: Perimeter relation in isosceles triangle
  → base_length = 5  -- Result: The base length of the isosceles triangle is 5
:= 
sorry

end isosceles_base_length_l106_106538


namespace marks_difference_is_140_l106_106919

noncomputable def marks_difference (P C M : ℕ) : ℕ :=
  (P + C + M) - P

theorem marks_difference_is_140 (P C M : ℕ) (h1 : (C + M) / 2 = 70) :
  marks_difference P C M = 140 := by
  sorry

end marks_difference_is_140_l106_106919


namespace square_diff_problem_l106_106746

theorem square_diff_problem
  (x : ℤ)
  (h : x^2 = 9801) :
  (x + 3) * (x - 3) = 9792 := 
by
  -- proof would go here
  sorry

end square_diff_problem_l106_106746


namespace union_sets_l106_106426

def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {x | ∃ a ∈ M, x = 3 * a}

theorem union_sets : M ∪ N = {0, 1, 3, 9} := by
  sorry

end union_sets_l106_106426


namespace problem_false_proposition_l106_106242

def p : Prop := ∀ x : ℝ, |x| = x ↔ x > 0

def q : Prop := (¬ ∃ x : ℝ, x^2 - x > 0) ↔ ∀ x : ℝ, x^2 - x ≤ 0

theorem problem_false_proposition : ¬ (p ∧ q) :=
by
  sorry

end problem_false_proposition_l106_106242


namespace proof_l106_106297

open Set

variable (U M P : Set ℕ)

noncomputable def prob_statement : Prop :=
  let C_U (A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}
  U = {1,2,3,4,5,6,7,8} ∧ M = {2,3,4} ∧ P = {1,3,6} ∧ C_U (M ∪ P) = {5,7,8}

theorem proof : prob_statement {1,2,3,4,5,6,7,8} {2,3,4} {1,3,6} :=
by
  sorry

end proof_l106_106297


namespace problem_1_problem_2_l106_106392

def f (x : ℝ) : ℝ := |2 * x - 1|

theorem problem_1 : {x : ℝ | f x > 2} = {x : ℝ | x < -1 / 2 ∨ x > 3 / 2} := sorry

theorem problem_2 (m : ℝ) : (∀ x : ℝ, f x + |2 * (x + 3)| - 4 > m * x) → m ≤ -11 := sorry

end problem_1_problem_2_l106_106392


namespace cyclic_quadrilateral_eq_l106_106565

theorem cyclic_quadrilateral_eq (A B C D : ℝ) (AB AD BC DC : ℝ)
  (h1 : AB = AD) (h2 : based_on_laws_of_cosines) : AC ^ 2 = BC * DC + AB ^ 2 :=
sorry

end cyclic_quadrilateral_eq_l106_106565


namespace cylinder_radius_exists_l106_106576

theorem cylinder_radius_exists (r h : ℕ) (pr : r ≥ 1) :
  (π * ↑r ^ 2 * ↑h = 2 * π * ↑r * (↑h + ↑r)) ↔
  (r = 3 ∨ r = 4 ∨ r = 6) :=
by
  sorry

end cylinder_radius_exists_l106_106576


namespace problem_statement_l106_106076

variable (a b c : ℝ)

theorem problem_statement (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + b^2 + c^2 = 1 / 2) :
  (1 - a^2 + c^2) / (c * (a + 2 * b)) + 
  (1 - b^2 + a^2) / (a * (b + 2 * c)) + 
  (1 - c^2 + b^2) / (b * (c + 2 * a)) ≥ 6 :=
sorry

end problem_statement_l106_106076


namespace number_of_pups_in_second_round_l106_106913

-- Define the conditions
variable (initialMice : Nat := 8)
variable (firstRoundPupsPerMouse : Nat := 6)
variable (secondRoundEatenPupsPerMouse : Nat := 2)
variable (finalMice : Nat := 280)

-- Define the proof problem
theorem number_of_pups_in_second_round (P : Nat) :
  initialMice + initialMice * firstRoundPupsPerMouse = 56 → 
  56 + 56 * P - 56 * secondRoundEatenPupsPerMouse = finalMice →
  P = 6 := by
  intros h1 h2
  sorry

end number_of_pups_in_second_round_l106_106913


namespace second_number_is_22_l106_106757

theorem second_number_is_22 (x second_number : ℕ) : 
  (x + second_number = 33) → 
  (second_number = 2 * x) → 
  second_number = 22 :=
by
  intros h_sum h_double
  sorry

end second_number_is_22_l106_106757


namespace minimum_cards_to_draw_to_ensure_2_of_each_suit_l106_106809

noncomputable def min_cards_to_draw {total_cards : ℕ} {suit_count : ℕ} {cards_per_suit : ℕ} {joker_count : ℕ}
  (h_total : total_cards = 54)
  (h_suits : suit_count = 4)
  (h_cards_per_suit : cards_per_suit = 13)
  (h_jokers : joker_count = 2) : ℕ :=
  43

theorem minimum_cards_to_draw_to_ensure_2_of_each_suit 
  (total_cards suit_count cards_per_suit joker_count : ℕ)
  (h_total : total_cards = 54) 
  (h_suits : suit_count = 4)
  (h_cards_per_suit : cards_per_suit = 13)
  (h_jokers : joker_count = 2) : 
  min_cards_to_draw h_total h_suits h_cards_per_suit h_jokers = 43 :=
  by
  sorry

end minimum_cards_to_draw_to_ensure_2_of_each_suit_l106_106809


namespace new_average_is_10_5_l106_106668

-- define the conditions
def average_of_eight_numbers (numbers : List ℝ) : Prop :=
  numbers.length = 8 ∧ (numbers.sum / 8) = 8

def add_four_to_five_numbers (numbers : List ℝ) (new_numbers : List ℝ) : Prop :=
  new_numbers = (numbers.take 5).map (λ x => x + 4) ++ numbers.drop 5

-- state the theorem
theorem new_average_is_10_5 (numbers new_numbers : List ℝ) 
  (h1 : average_of_eight_numbers numbers)
  (h2 : add_four_to_five_numbers numbers new_numbers) :
  (new_numbers.sum / 8) = 10.5 := 
by 
  sorry

end new_average_is_10_5_l106_106668


namespace min_even_number_for_2015_moves_l106_106717

theorem min_even_number_for_2015_moves (N : ℕ) (hN : N ≥ 2) :
  ∃ k : ℕ, N = 2 ^ k ∧ 2 ^ k ≥ 2 ∧ k ≥ 4030 :=
sorry

end min_even_number_for_2015_moves_l106_106717


namespace problem_solution_l106_106843

theorem problem_solution (N : ℚ) (h : (4/5) * (3/8) * N = 24) : 2.5 * N = 200 :=
by {
  sorry
}

end problem_solution_l106_106843


namespace cube_number_sum_is_102_l106_106589

noncomputable def sum_of_cube_numbers (n1 n2 n3 n4 n5 n6 : ℕ) : ℕ := n1 + n2 + n3 + n4 + n5 + n6

theorem cube_number_sum_is_102 : 
  ∃ (n1 n2 n3 n4 n5 n6 : ℕ),
    n1 = 12 ∧ 
    n2 = n1 + 2 ∧ 
    n3 = n2 + 2 ∧ 
    n4 = n3 + 2 ∧ 
    n5 = n4 + 2 ∧ 
    n6 = n5 + 2 ∧ 
    ((n1 + n6 = n2 + n5) ∧ (n1 + n6 = n3 + n4)) ∧ 
    sum_of_cube_numbers n1 n2 n3 n4 n5 n6 = 102 :=
by
  sorry

end cube_number_sum_is_102_l106_106589


namespace largest_value_of_m_exists_l106_106118

theorem largest_value_of_m_exists (a b c : ℝ) (h₁ : a + b + c = 12) (h₂ : a * b + b * c + c * a = 30) (h₃ : 0 < a) (h₄ : 0 < b) (h₅ : 0 < c) : 
  ∃ m : ℝ, (m = min (a * b) (min (b * c) (c * a))) ∧ (m = 2) := sorry

end largest_value_of_m_exists_l106_106118


namespace sufficient_but_not_necessary_condition_l106_106241

variable (a b : ℝ)

theorem sufficient_but_not_necessary_condition (h : |b| + a < 0) : b^2 < a^2 :=
  sorry

end sufficient_but_not_necessary_condition_l106_106241


namespace determine_k_l106_106213

noncomputable def p (x y : ℝ) : ℝ := x^2 - y^2
noncomputable def q (x y : ℝ) : ℝ := Real.log (x - y)

def m (k : ℝ) : ℝ := 2 * k
def w (n : ℝ) : ℝ := n + 1

theorem determine_k (k : ℝ) (c : ℝ → ℝ → ℝ) (v : ℝ → ℝ → ℝ) (n : ℝ) :
  p 32 6 = k * c 32 6 ∧
  p 45 10 = m k * c 45 10 ∧
  q 15 5 = n * v 15 5 ∧
  q 28 7 = w n * v 28 7 →
  k = 1925 / 1976 :=
by
  sorry

end determine_k_l106_106213


namespace polynomial_factorization_l106_106115

variable (x y : ℝ)

theorem polynomial_factorization (m : ℝ) :
  (∃ (a b : ℝ), 6 * x^2 - 5 * x * y - 4 * y^2 - 11 * x + 22 * y + m = (3 * x - 4 * y + a) * (2 * x + y + b)) →
  m = -10 :=
sorry

end polynomial_factorization_l106_106115


namespace height_of_circular_segment_l106_106522

theorem height_of_circular_segment (d a : ℝ) (h : ℝ) :
  (h = (d - Real.sqrt (d^2 - a^2)) / 2) ↔ 
  ((a / 2)^2 + (d / 2 - h)^2 = (d / 2)^2) :=
sorry

end height_of_circular_segment_l106_106522


namespace least_positive_integer_divisible_by_5_to_15_l106_106665

def is_divisible_by_all (n : ℕ) (l : List ℕ) : Prop :=
  ∀ m ∈ l, m ∣ n

theorem least_positive_integer_divisible_by_5_to_15 :
  ∃ n : ℕ, n > 0 ∧ is_divisible_by_all n [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] ∧
  ∀ m : ℕ, m > 0 ∧ is_divisible_by_all m [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] → n ≤ m ∧ n = 360360 :=
by
  sorry

end least_positive_integer_divisible_by_5_to_15_l106_106665


namespace interval_comparison_l106_106959

theorem interval_comparison (x : ℝ) :
  ((x - 1) * (x + 3) < 0) → ¬((x + 1) * (x - 3) < 0) ∧ ¬((x + 1) * (x - 3) < 0 → (x - 1) * (x + 3) < 0) :=
by
  sorry

end interval_comparison_l106_106959


namespace smallest_prime_after_five_consecutive_nonprimes_l106_106986

theorem smallest_prime_after_five_consecutive_nonprimes :
  ∃ p : ℕ, Nat.Prime p ∧ 
          (∀ n : ℕ, n < p → ¬ (n ≥ 24 ∧ n < 29 ∧ ¬ Nat.Prime n)) ∧
          p = 29 :=
by
  sorry

end smallest_prime_after_five_consecutive_nonprimes_l106_106986


namespace min_value_a_sq_plus_b_sq_l106_106810

theorem min_value_a_sq_plus_b_sq (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (a - 1)^3 + (b - 1)^3 ≥ 3 * (2 - a - b)) : 
  ∃ (m : ℝ), m = 2 ∧ (∀ x y, x > 0 → y > 0 → (x - 1)^3 + (y - 1)^3 ≥ 3 * (2 - x - y) → x^2 + y^2 ≥ m) :=
by
  sorry

end min_value_a_sq_plus_b_sq_l106_106810


namespace negation_correct_l106_106147

-- Definitions needed from the conditions:
def is_positive (m : ℝ) : Prop := m > 0
def square (m : ℝ) : ℝ := m * m

-- The original proposition
def original_proposition (m : ℝ) : Prop := is_positive m → square m > 0

-- The negation of the proposition
def negated_proposition (m : ℝ) : Prop := ¬is_positive m → ¬(square m > 0)

-- The theorem to prove that the negated proposition is the negation of the original proposition
theorem negation_correct (m : ℝ) : (original_proposition m) ↔ (negated_proposition m) :=
by
  sorry

end negation_correct_l106_106147


namespace slope_of_tangent_line_at_1_1_l106_106005

theorem slope_of_tangent_line_at_1_1 : 
  ∃ f' : ℝ → ℝ, (∀ x, f' x = 3 * x^2) ∧ (f' 1 = 3) :=
by
  sorry

end slope_of_tangent_line_at_1_1_l106_106005


namespace arithmetic_geometric_sequence_l106_106023

theorem arithmetic_geometric_sequence (a : ℕ → ℤ) (d : ℤ) (h_arith : ∀ n, a (n + 1) = a n + d) 
  (h_common_diff : d = 2) (h_geom : a 2 ^ 2 = a 1 * a 5) : 
  a 2 = 3 :=
by
  sorry

end arithmetic_geometric_sequence_l106_106023


namespace ellipse_sum_l106_106674

theorem ellipse_sum (h k a b : ℝ) (h_val : h = 3) (k_val : k = -5) (a_val : a = 6) (b_val : b = 2) : h + k + a + b = 6 :=
by
  rw [h_val, k_val, a_val, b_val]
  norm_num

end ellipse_sum_l106_106674


namespace factorize_difference_of_squares_l106_106192

theorem factorize_difference_of_squares (x : ℝ) : 9 - 4 * x^2 = (3 - 2 * x) * (3 + 2 * x) :=
sorry

end factorize_difference_of_squares_l106_106192


namespace least_possible_sum_l106_106435

theorem least_possible_sum (p q : ℕ) (hp : 1 < p) (hq : 1 < q) (h : 17 * (p + 1) = 21 * (q + 1)) : p + q = 5 :=
sorry

end least_possible_sum_l106_106435


namespace raghu_investment_l106_106211

-- Define the conditions as Lean definitions
def invest_raghu : Real := sorry
def invest_trishul := 0.90 * invest_raghu
def invest_vishal := 1.10 * invest_trishul
def invest_chandni := 1.15 * invest_vishal
def total_investment := invest_raghu + invest_trishul + invest_vishal + invest_chandni

-- State the proof problem
theorem raghu_investment (h : total_investment = 10700) : invest_raghu = 2656.25 :=
by
  sorry

end raghu_investment_l106_106211


namespace arithmetic_sequence_a1_a9_l106_106085

theorem arithmetic_sequence_a1_a9 
  (a : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_sum_456 : a 4 + a 5 + a 6 = 36) : 
  a 1 + a 9 = 24 := 
sorry

end arithmetic_sequence_a1_a9_l106_106085


namespace perimeter_of_square_l106_106827

theorem perimeter_of_square (a : ℤ) (h : a * a = 36) : 4 * a = 24 := 
by
  sorry

end perimeter_of_square_l106_106827


namespace optimal_addition_amount_l106_106829

def optimal_material_range := {x : ℝ | 100 ≤ x ∧ x ≤ 200}

def second_trial_amounts := {x : ℝ | x = 138.2 ∨ x = 161.8}

theorem optimal_addition_amount (
  h1 : ∀ x ∈ optimal_material_range, x ∈ second_trial_amounts
  ) :
  138.2 ∈ second_trial_amounts ∧ 161.8 ∈ second_trial_amounts :=
by
  sorry

end optimal_addition_amount_l106_106829


namespace tan_alpha_value_l106_106078

theorem tan_alpha_value (α : ℝ) (h : 0 < α ∧ α < π / 2) 
  (h_tan2α : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α) ) :
  Real.tan α = Real.sqrt 15 / 15 :=
sorry

end tan_alpha_value_l106_106078


namespace ladder_rungs_count_l106_106282

theorem ladder_rungs_count :
  ∃ (n : ℕ), ∀ (start mid : ℕ),
    start = n / 2 →
    mid = ((start + 5 - 7) + 8 + 7) →
    mid = n →
    n = 27 :=
by
  sorry

end ladder_rungs_count_l106_106282


namespace possible_combinations_of_scores_l106_106087

theorem possible_combinations_of_scores 
    (scores : Set ℕ := {0, 3, 5})
    (total_scores : ℕ := 32)
    (teams : ℕ := 3)
    : (∃ (number_of_combinations : ℕ), number_of_combinations = 255) := by
  sorry

end possible_combinations_of_scores_l106_106087


namespace tickets_distribution_correct_l106_106191

def tickets_distribution (tickets programs : nat) (A_tickets_min : nat) : nat :=
sorry

theorem tickets_distribution_correct :
  tickets_distribution 6 4 3 = 17 :=
by
  sorry

end tickets_distribution_correct_l106_106191


namespace part1_part2_l106_106278

noncomputable def f (x a : ℝ) : ℝ := |x - 1| + |2 * x + a|

theorem part1 (x : ℝ) : f x 1 + |x - 1| ≥ 3 :=
  sorry

theorem part2 (a : ℝ) (h : ∃ x : ℝ, f x a = 2) : a = 2 ∨ a = -6 :=
  sorry

end part1_part2_l106_106278


namespace count_similar_divisors_l106_106468

def is_integrally_similar_divisible (a b c : ℕ) : Prop :=
  ∃ x y z : ℕ, a * c = b * z ∧
  x ≤ y ∧ y ≤ z ∧
  b = 2023 ∧ a * c = 2023^2

theorem count_similar_divisors (b : ℕ) (hb : b = 2023) :
  ∃ (n : ℕ), n = 7 ∧ 
    (∀ (a c : ℕ), a ≤ b ∧ b ≤ c → is_integrally_similar_divisible a b c) :=
by
  sorry

end count_similar_divisors_l106_106468


namespace initial_money_l106_106288

theorem initial_money (M : ℝ)
  (clothes : M * (1 / 3) = M - M * (2 / 3))
  (food : (M - M * (1 / 3)) * (1 / 5) = (M - M * (1 / 3)) - ((M - M * (1 / 3)) * (4 / 5)))
  (travel : ((M - M * (1 / 3)) - ((M - M * (1 / 3)) * (1 / 5))) * (1 / 4) = ((M - M * (1 / 3)) - ((M - M * (1 / 3)) * (1 / 5))) - (((M - M * (1 / 3)) - ((M - M * (1 / 3)) * (1 / 5)))) * (3 / 4))
  (left : ((M - M * (1 / 3)) - (((M - M * (1 / 3)) - ((M - M * (1 / 3)) * (1 / 5))) * (1 / 4))) = 400)
  : M = 1000 := 
sorry

end initial_money_l106_106288


namespace max_value_of_expression_l106_106869

noncomputable def max_expression_value (a b c : ℝ) : ℝ :=
  (1 / ((1 - a^2) * (1 - b^2) * (1 - c^2))) + (1 / ((1 + a^2) * (1 + b^2) * (1 + c^2)))

theorem max_value_of_expression (a b c : ℝ) (ha : -1 < a ∧ a < 1) (hb : -1 < b ∧ b < 1) (hc : -1 < c ∧ c < 1) :
  max_expression_value a b c ≤ 2 :=
by sorry

end max_value_of_expression_l106_106869


namespace no_nat_num_divisible_l106_106592

open Nat

theorem no_nat_num_divisible : ¬ ∃ n : ℕ, (n^2 + 6 * n + 2019) % 100 = 0 := sorry

end no_nat_num_divisible_l106_106592


namespace average_is_0_1667X_plus_3_l106_106362

noncomputable def average_of_three_numbers (X Y Z : ℝ) : ℝ := (X + Y + Z) / 3

theorem average_is_0_1667X_plus_3 (X Y Z : ℝ) 
  (h1 : 2001 * Z - 4002 * X = 8008) 
  (h2 : 2001 * Y + 5005 * X = 10010) : 
  average_of_three_numbers X Y Z = 0.1667 * X + 3 := 
sorry

end average_is_0_1667X_plus_3_l106_106362


namespace simplify_eval_l106_106731

theorem simplify_eval (a : ℝ) (h : a = Real.sqrt 3 / 3) : (a + 1) ^ 2 + a * (1 - a) = Real.sqrt 3 + 1 := 
by
  sorry

end simplify_eval_l106_106731


namespace problem_statement_l106_106761

theorem problem_statement (x : ℝ) (h₀ : x > 0) (n : ℕ) (hn : n > 0) :
  (x + (n^n : ℝ) / x^n) ≥ (n + 1) :=
sorry

end problem_statement_l106_106761


namespace mul_digits_example_l106_106149

theorem mul_digits_example (A B C D : ℕ) (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D)
  (h4 : B ≠ C) (h5 : B ≠ D) (h6 : C ≠ D) (h7 : C = 2) (h8 : D = 5) : A + B = 2 := by
  sorry

end mul_digits_example_l106_106149


namespace f1_neither_even_nor_odd_f2_min_value_l106_106904

noncomputable def f1 (x : ℝ) : ℝ :=
  x^2 + abs (x - 2) - 1

theorem f1_neither_even_nor_odd : ¬(∀ x : ℝ, f1 x = f1 (-x)) ∧ ¬(∀ x : ℝ, f1 x = -f1 (-x)) :=
sorry

noncomputable def f2 (x a : ℝ) : ℝ :=
  x^2 + abs (x - a) + 1

theorem f2_min_value (a : ℝ) :
  (if a < -1/2 then (∃ x, f2 x a = 3/4 - a)
  else if -1/2 ≤ a ∧ a ≤ 1/2 then (∃ x, f2 x a = a^2 + 1)
  else (∃ x, f2 x a = 3/4 + a)) :=
sorry

end f1_neither_even_nor_odd_f2_min_value_l106_106904


namespace twice_x_minus_3_l106_106863

theorem twice_x_minus_3 (x : ℝ) : (2 * x) - 3 = 2 * x - 3 := 
by 
  -- This proof is trivial and we can assert equality directly
  sorry

end twice_x_minus_3_l106_106863


namespace find_c_plus_d_l106_106765

variables {a b c d : ℝ}

theorem find_c_plus_d (h1 : a + b = 16) (h2 : b + c = 9) (h3 : a + d = 10) : c + d = 3 :=
by
  sorry

end find_c_plus_d_l106_106765


namespace max_value_point_l106_106971

noncomputable def f (x : ℝ) : ℝ := x + Real.cos (2 * x)

theorem max_value_point : ∃ x ∈ Set.Ioo 0 Real.pi, (∀ y ∈ Set.Ioo 0 Real.pi, f x ≥ f y) ∧ x = Real.pi / 12 :=
by sorry

end max_value_point_l106_106971


namespace minimize_time_theta_l106_106046

theorem minimize_time_theta (α θ : ℝ) (h1 : 0 < α) (h2 : α < 90) (h3 : θ = α / 2) : 
  θ = α / 2 :=
by
  sorry

end minimize_time_theta_l106_106046


namespace zack_travel_countries_l106_106764

theorem zack_travel_countries (G J P Z : ℕ) 
  (hG : G = 6)
  (hJ : J = G / 2)
  (hP : P = 3 * J)
  (hZ : Z = 2 * P) :
  Z = 18 := by
  sorry

end zack_travel_countries_l106_106764


namespace shoveling_time_l106_106567

theorem shoveling_time :
  let kevin_time := 12
  let dave_time := 8
  let john_time := 6
  let allison_time := 4
  let kevin_rate := 1 / kevin_time
  let dave_rate := 1 / dave_time
  let john_rate := 1 / john_time
  let allison_rate := 1 / allison_time
  let combined_rate := kevin_rate + dave_rate + john_rate + allison_rate
  let total_minutes := 60
  let combined_rate_per_minute := combined_rate / total_minutes
  (1 / combined_rate_per_minute = 96) := 
  sorry

end shoveling_time_l106_106567


namespace calculate_correct_subtraction_l106_106154

theorem calculate_correct_subtraction (x : ℤ) (h : x - 63 = 24) : x - 36 = 51 :=
by
  sorry

end calculate_correct_subtraction_l106_106154


namespace f_of_1789_l106_106151

-- Definitions as per conditions
def f : ℕ → ℕ := sorry -- This will be the function definition satisfying the conditions

axiom f_f_n (n : ℕ) (h : n > 0) : f (f n) = 4 * n + 9
axiom f_2_k (k : ℕ) : f (2^k) = 2^(k+1) + 3

-- Prove f(1789) = 3581 given the conditions.
theorem f_of_1789 : f 1789 = 3581 := 
sorry

end f_of_1789_l106_106151


namespace area_of_sector_l106_106577

theorem area_of_sector
  (θ : ℝ) (l : ℝ) (r : ℝ := l / θ)
  (h1 : θ = 2)
  (h2 : l = 4) :
  1 / 2 * r^2 * θ = 4 :=
by
  sorry

end area_of_sector_l106_106577


namespace minimum_value_of_expression_l106_106562

noncomputable def expression (x y : ℝ) : ℝ :=
  (5 * x^2 + 8 * x * y + 5 * y^2 - 14 * x - 10 * y + 30) /
  (4 - x^2 - 10 * x * y - 25 * y^2) ^ (7 / 2)

theorem minimum_value_of_expression : 
  ∃ x y : ℝ, expression x y = 5/32 :=
by
  sorry

end minimum_value_of_expression_l106_106562


namespace find_n_l106_106391

-- Define the arithmetic series sums
def s1 (n : ℕ) : ℕ := (5 * n^2 + 5 * n) / 2
def s2 (n : ℕ) : ℕ := n^2 + n

-- The theorem to be proved
theorem find_n : ∃ n : ℕ, s1 n + s2 n = 156 ∧ n = 7 :=
by
  sorry

end find_n_l106_106391


namespace cylindrical_coordinates_of_point_l106_106235

noncomputable def cylindrical_coordinates (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x = -r then Real.pi else 0 -- From the step if cos θ = -1
  (r, θ, z)

theorem cylindrical_coordinates_of_point :
  cylindrical_coordinates (-5) 0 (-8) = (5, Real.pi, -8) :=
by
  -- placeholder for the actual proof
  sorry

end cylindrical_coordinates_of_point_l106_106235


namespace sum_remainders_eq_two_l106_106024

theorem sum_remainders_eq_two (a b c : ℤ) (h_a : a % 24 = 10) (h_b : b % 24 = 4) (h_c : c % 24 = 12) :
  (a + b + c) % 24 = 2 :=
by
  sorry

end sum_remainders_eq_two_l106_106024


namespace sequence_contains_infinite_squares_l106_106250

theorem sequence_contains_infinite_squares :
  ∃ f : ℕ → ℕ, ∀ m : ℕ, ∃ n : ℕ, f (n + m) * f (n + m) = 1 + 17 * (n + m) ^ 2 :=
sorry

end sequence_contains_infinite_squares_l106_106250


namespace solve_money_conditions_l106_106622

theorem solve_money_conditions 
  (a b : ℝ)
  (h1 : b - 4 * a < 78)
  (h2 : 6 * a - b = 36) :
  a < 57 ∧ b > -36 :=
sorry

end solve_money_conditions_l106_106622


namespace fruit_seller_profit_l106_106353

theorem fruit_seller_profit 
  (SP : ℝ) (Loss_Percentage : ℝ) (New_SP : ℝ) (Profit_Percentage : ℝ) 
  (h1: SP = 8) 
  (h2: Loss_Percentage = 20) 
  (h3: New_SP = 10.5) 
  (h4: Profit_Percentage = 5) :
  ((New_SP - (SP / (1 - (Loss_Percentage / 100.0))) / (SP / (1 - (Loss_Percentage / 100.0)))) * 100) = Profit_Percentage := 
sorry

end fruit_seller_profit_l106_106353


namespace problem_statement_l106_106703

open Real

variable (a b c : ℝ)

theorem problem_statement
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h_cond : a + b + c + a * b * c = 4) :
  (1 + a / b + c * a) * (1 + b / c + a * b) * (1 + c / a + b * c) ≥ 27 := 
by
  sorry

end problem_statement_l106_106703


namespace temp_difference_l106_106153

theorem temp_difference
  (temp_beijing : ℤ) 
  (temp_hangzhou : ℤ) 
  (h_beijing : temp_beijing = -10) 
  (h_hangzhou : temp_hangzhou = -1) : 
  temp_beijing - temp_hangzhou = -9 := 
by 
  rw [h_beijing, h_hangzhou] 
  sorry

end temp_difference_l106_106153


namespace man_receives_total_amount_l106_106074
noncomputable def total_amount_received : ℝ := 
  let itemA_price := 1300
  let itemB_price := 750
  let itemC_price := 1800
  
  let itemA_loss := 0.20 * itemA_price
  let itemB_loss := 0.15 * itemB_price
  let itemC_loss := 0.10 * itemC_price

  let itemA_selling_price := itemA_price - itemA_loss
  let itemB_selling_price := itemB_price - itemB_loss
  let itemC_selling_price := itemC_price - itemC_loss

  let vat_rate := 0.12
  let itemA_vat := vat_rate * itemA_selling_price
  let itemB_vat := vat_rate * itemB_selling_price
  let itemC_vat := vat_rate * itemC_selling_price

  let final_itemA := itemA_selling_price + itemA_vat
  let final_itemB := itemB_selling_price + itemB_vat
  let final_itemC := itemC_selling_price + itemC_vat

  final_itemA + final_itemB + final_itemC

theorem man_receives_total_amount :
  total_amount_received = 3693.2 := by
  sorry

end man_receives_total_amount_l106_106074


namespace problem_l106_106320

variable (x y : ℝ)

theorem problem (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 :=
by
  sorry

end problem_l106_106320


namespace fifth_power_last_digit_l106_106524

theorem fifth_power_last_digit (n : ℕ) : 
  (n % 10)^5 % 10 = n % 10 :=
by sorry

end fifth_power_last_digit_l106_106524


namespace intersection_complement_l106_106663

open Set

theorem intersection_complement (U A B : Set ℕ) (hU : U = {x | x ≤ 6}) (hA : A = {1, 3, 5}) (hB : B = {4, 5, 6}) :
  (U \ A) ∩ B = {4, 6} := by
  sorry

end intersection_complement_l106_106663


namespace production_time_l106_106316

variable (a m : ℝ) -- Define a and m as real numbers

-- State the problem as a theorem in Lean
theorem production_time : (a / m) * 200 = 200 * (a / m) := by
  sorry

end production_time_l106_106316


namespace volume_of_triangular_pyramid_l106_106957

variable (a b : ℝ)

noncomputable def volume_of_pyramid (a b : ℝ) : ℝ :=
  (a * b / 12) * Real.sqrt (3 * b ^ 2 - a ^ 2)

theorem volume_of_triangular_pyramid (a b : ℝ) :
  volume_of_pyramid a b = (a * b / 12) * Real.sqrt (3 * b ^ 2 - a ^ 2) :=
by
  sorry

end volume_of_triangular_pyramid_l106_106957


namespace sara_wrapping_paper_l106_106968

theorem sara_wrapping_paper (s : ℚ) (l : ℚ) (total : ℚ) : 
  total = 3 / 8 → 
  l = 2 * s →
  4 * s + 2 * l = total → 
  s = 3 / 64 :=
by
  intros h1 h2 h3
  sorry

end sara_wrapping_paper_l106_106968


namespace total_wet_surface_area_l106_106317

def cistern_length (L : ℝ) := L = 5
def cistern_width (W : ℝ) := W = 4
def water_depth (D : ℝ) := D = 1.25

theorem total_wet_surface_area (L W D A : ℝ) 
  (hL : cistern_length L) 
  (hW : cistern_width W) 
  (hD : water_depth D) :
  A = 42.5 :=
by
  subst hL
  subst hW
  subst hD
  sorry

end total_wet_surface_area_l106_106317


namespace number_of_books_in_shipment_l106_106747

theorem number_of_books_in_shipment
  (T : ℕ)                   -- The total number of books
  (displayed_ratio : ℚ)     -- Fraction of books displayed
  (remaining_books : ℕ)     -- Number of books in the storeroom
  (h1 : displayed_ratio = 0.3)
  (h2 : remaining_books = 210)
  (h3 : (1 - displayed_ratio) * T = remaining_books) :
  T = 300 := 
by
  -- Add your proof here
  sorry

end number_of_books_in_shipment_l106_106747


namespace pancakes_eaten_by_older_is_12_l106_106114

/-- Pancake problem conditions -/
def initial_pancakes : ℕ := 19
def final_pancakes : ℕ := 11
def younger_eats_per_cycle : ℕ := 1
def older_eats_per_cycle : ℕ := 3
def grandma_bakes_per_cycle : ℕ := 2
def net_reduction_per_cycle := younger_eats_per_cycle + older_eats_per_cycle - grandma_bakes_per_cycle
def total_pancakes_eaten_by_older (cycles : ℕ) := older_eats_per_cycle * cycles

/-- Calculate the cycles based on net reduction -/
def cycles : ℕ := (initial_pancakes - final_pancakes) / net_reduction_per_cycle

/-- Prove the number of pancakes the older grandchild eats is 12 based on given conditions --/
theorem pancakes_eaten_by_older_is_12 : total_pancakes_eaten_by_older cycles = 12 := by
  sorry

end pancakes_eaten_by_older_is_12_l106_106114


namespace find_b_l106_106880

theorem find_b 
  (a b : ℚ)
  (h_root : (1 + Real.sqrt 5) ^ 3 + a * (1 + Real.sqrt 5) ^ 2 + b * (1 + Real.sqrt 5) - 60 = 0) :
  b = 26 :=
sorry

end find_b_l106_106880


namespace sufficient_not_necessary_range_l106_106088

theorem sufficient_not_necessary_range (a : ℝ) (h : ∀ x : ℝ, x > 2 → x^2 > a ∧ ¬(x^2 > a → x > 2)) : a ≤ 4 :=
by
  sorry

end sufficient_not_necessary_range_l106_106088


namespace solution_l106_106260

noncomputable def problem : Prop :=
  let num_apprentices := 200
  let num_junior := 20
  let num_intermediate := 60
  let num_senior := 60
  let num_technician := 40
  let num_senior_technician := 20
  let total_technician := num_technician + num_senior_technician
  let sampling_ratio := 10 / num_apprentices
  
  -- Number of technicians (including both technician and senior technicians) in the exchange group
  let num_technicians_selected := total_technician * sampling_ratio

  -- Probability Distribution of X
  let P_X_0 := 7 / 24
  let P_X_1 := 21 / 40
  let P_X_2 := 7 / 40
  let P_X_3 := 1 / 120

  -- Expected value of X
  let E_X := (0 * P_X_0) + (1 * P_X_1) + (2 * P_X_2) + (3 * P_X_3)
  E_X = 9 / 10

theorem solution : problem :=
  sorry

end solution_l106_106260


namespace min_value_of_expr_l106_106443

theorem min_value_of_expr (a : ℝ) (ha : a > 1) : a + a^2 / (a - 1) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_of_expr_l106_106443


namespace find_positive_integer_solutions_l106_106750

theorem find_positive_integer_solutions (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  3 ^ x + 4 ^ y = 5 ^ z → x = 2 ∧ y = 2 ∧ z = 2 :=
by
  sorry

end find_positive_integer_solutions_l106_106750


namespace number_of_machines_l106_106039

def machine_problem : Prop :=
  ∃ (m : ℕ), (6 * 42) = 6 * 36 ∧ m = 7

theorem number_of_machines : machine_problem :=
  sorry

end number_of_machines_l106_106039


namespace full_time_score_l106_106646

variables (x : ℕ)

def half_time_score_visitors := 14
def half_time_score_home := 9
def visitors_full_time_score := half_time_score_visitors + x
def home_full_time_score := half_time_score_home + 2 * x
def home_team_win_by_one := home_full_time_score = visitors_full_time_score + 1

theorem full_time_score 
  (h : home_team_win_by_one) : 
  visitors_full_time_score = 20 ∧ home_full_time_score = 21 :=
by
  sorry

end full_time_score_l106_106646


namespace chris_money_before_birthday_l106_106349

variables {x : ℕ} -- Assuming we are working with natural numbers (non-negative integers)

-- Conditions
def grandmother_money : ℕ := 25
def aunt_uncle_money : ℕ := 20
def parents_money : ℕ := 75
def total_money_now : ℕ := 279

-- Question
theorem chris_money_before_birthday : x = total_money_now - (grandmother_money + aunt_uncle_money + parents_money) :=
by
  sorry

end chris_money_before_birthday_l106_106349


namespace solve_quadratic_eq_l106_106276

theorem solve_quadratic_eq (x : ℝ) (h : x^2 + 2 * x - 15 = 0) : x = 3 ∨ x = -5 :=
by {
  sorry
}

end solve_quadratic_eq_l106_106276


namespace value_of_a_minus_b_l106_106877

theorem value_of_a_minus_b 
  (a b : ℤ) 
  (x y : ℤ)
  (h1 : x = -2)
  (h2 : y = 1)
  (h3 : a * x + b * y = 1)
  (h4 : b * x + a * y = 7) : 
  a - b = 2 :=
by
  sorry

end value_of_a_minus_b_l106_106877


namespace largest_possible_e_l106_106539

noncomputable def diameter := (2 : ℝ)
noncomputable def PX := (4 / 5 : ℝ)
noncomputable def PY := (3 / 4 : ℝ)
noncomputable def e := (41 - 16 * Real.sqrt 25 : ℝ)
noncomputable def u := 41
noncomputable def v := 16
noncomputable def w := 25

theorem largest_possible_e (P Q X Y Z R S : Real) (d : diameter = 2)
  (PX_len : P - X = 4/5) (PY_len : P - Y = 3/4)
  (e_def : e = 41 - 16 * Real.sqrt 25)
  : u + v + w = 82 :=
by
  sorry

end largest_possible_e_l106_106539


namespace calen_pencils_loss_l106_106690

theorem calen_pencils_loss
  (P_Candy : ℕ)
  (P_Caleb : ℕ)
  (P_Calen_original : ℕ)
  (P_Calen_after_loss : ℕ)
  (h1 : P_Candy = 9)
  (h2 : P_Caleb = 2 * P_Candy - 3)
  (h3 : P_Calen_original = P_Caleb + 5)
  (h4 : P_Calen_after_loss = 10) :
  P_Calen_original - P_Calen_after_loss = 10 := 
sorry

end calen_pencils_loss_l106_106690


namespace ratio_alan_to_ben_l106_106272

theorem ratio_alan_to_ben (A B L : ℕ) (hA : A = 48) (hL : L = 36) (hB : B = L / 3) : A / B = 4 := by
  sorry

end ratio_alan_to_ben_l106_106272


namespace tiffany_bags_difference_l106_106226

theorem tiffany_bags_difference : 
  ∀ (monday_bags next_day_bags : ℕ), monday_bags = 7 → next_day_bags = 12 → next_day_bags - monday_bags = 5 := 
by
  intros monday_bags next_day_bags h1 h2
  sorry

end tiffany_bags_difference_l106_106226


namespace factorize_polynomial_l106_106479

def p (a b : ℝ) : ℝ := a^2 - b^2 + 2 * a + 1

theorem factorize_polynomial (a b : ℝ) : 
  p a b = (a + 1 + b) * (a + 1 - b) :=
by
  sorry

end factorize_polynomial_l106_106479


namespace classrooms_students_guinea_pigs_difference_l106_106962

theorem classrooms_students_guinea_pigs_difference :
  let students_per_classroom := 22
  let guinea_pigs_per_classroom := 3
  let number_of_classrooms := 5
  let total_students := students_per_classroom * number_of_classrooms
  let total_guinea_pigs := guinea_pigs_per_classroom * number_of_classrooms
  total_students - total_guinea_pigs = 95 :=
  by
    sorry

end classrooms_students_guinea_pigs_difference_l106_106962


namespace find_FC_l106_106872

theorem find_FC 
(DC CB AD ED FC : ℝ)
(h1 : DC = 7) 
(h2 : CB = 8) 
(h3 : AB = (1 / 4) * AD)
(h4 : ED = (4 / 5) * AD) : 
FC = 10.4 :=
sorry

end find_FC_l106_106872


namespace distance_from_Asheville_to_Darlington_l106_106031

theorem distance_from_Asheville_to_Darlington (BC AC BD AD : ℝ) 
(h0 : BC = 12) 
(h1 : BC = (1/3) * AC) 
(h2 : BC = (1/4) * BD) :
AD = 72 :=
sorry

end distance_from_Asheville_to_Darlington_l106_106031


namespace speed_of_man_rowing_upstream_l106_106315

-- Define conditions
def V_m : ℝ := 20 -- speed of the man in still water (kmph)
def V_downstream : ℝ := 25 -- speed of the man rowing downstream (kmph)
def V_s : ℝ := V_downstream - V_m -- calculate the speed of the stream

-- Define the theorem to prove the speed of the man rowing upstream
theorem speed_of_man_rowing_upstream 
  (V_m : ℝ) (V_downstream : ℝ) (V_s : ℝ := V_downstream - V_m) : 
  V_upstream = V_m - V_s :=
by
  sorry

end speed_of_man_rowing_upstream_l106_106315


namespace find_a_l106_106469

variable (a b c : ℚ)

theorem find_a (h1 : a + b + c = 150) (h2 : a - 3 = b + 4) (h3 : b + 4 = 4 * c) : 
  a = 631 / 9 :=
by
  sorry

end find_a_l106_106469


namespace find_d_l106_106358

theorem find_d (d : ℚ) (h : ∀ x : ℚ, 4*x^3 + 17*x^2 + d*x + 28 = 0 → x = -4/3) : d = 155 / 9 :=
sorry

end find_d_l106_106358


namespace cara_total_bread_l106_106682

variable (L B : ℕ)  -- Let L and B be the amount of bread for lunch and breakfast, respectively

theorem cara_total_bread :
  (dinner = 240) → 
  (dinner = 8 * L) → 
  (dinner = 6 * B) → 
  (total_bread = dinner + L + B) → 
  total_bread = 310 :=
by
  intros
  -- Here you'd begin your proof, implementing each given condition
  sorry

end cara_total_bread_l106_106682


namespace general_term_l106_106222

noncomputable def F (n : ℕ) : ℝ :=
  1 / (Real.sqrt 5) * (((1 + Real.sqrt 5) / 2)^(n-2) - ((1 - Real.sqrt 5) / 2)^(n-2))

noncomputable def a : ℕ → ℝ
| 0 => 1
| 1 => 5
| n+2 => a (n+1) * a n / Real.sqrt ((a (n+1))^2 + (a n)^2 + 1)

theorem general_term (n : ℕ) :
  a n = (2^(F (n+2)) * 13^(F (n+1)) * 5^(-2 * F (n+1)) - 1)^(1/2) := sorry

end general_term_l106_106222


namespace solve_for_x_l106_106415

theorem solve_for_x (x y : ℚ) (h1 : 3 * x - 2 * y = 8) (h2 : x + 3 * y = 7) : x = 38 / 11 :=
by
  sorry

end solve_for_x_l106_106415


namespace train_length_l106_106059

theorem train_length (L : ℝ) : (L + 200) / 15 = (L + 300) / 20 → L = 100 :=
by
  intro h
  -- Skipping the proof steps
  sorry

end train_length_l106_106059


namespace term_217_is_61st_l106_106163

variables {a_n : ℕ → ℝ}

def arithmetic_sequence (a_n : ℕ → ℝ) (a_15 a_45 : ℝ) : Prop :=
  ∃ (a₁ d : ℝ), (∀ n, a_n n = a₁ + (n - 1) * d) ∧ a_n 15 = a_15 ∧ a_n 45 = a_45

theorem term_217_is_61st (h : arithmetic_sequence a_n 33 153) : a_n 61 = 217 := sorry

end term_217_is_61st_l106_106163


namespace work_efficiency_ratio_l106_106096
noncomputable section

variable (A_eff B_eff : ℚ)

-- Conditions
def efficient_together (A_eff B_eff : ℚ) : Prop := A_eff + B_eff = 1 / 12
def efficient_alone (A_eff : ℚ) : Prop := A_eff = 1 / 16

-- Theorem to prove
theorem work_efficiency_ratio (A_eff B_eff : ℚ) (h1 : efficient_together A_eff B_eff) (h2 : efficient_alone A_eff) : A_eff / B_eff = 3 := by
  sorry

end work_efficiency_ratio_l106_106096


namespace fly_dist_ceiling_eq_sqrt255_l106_106603

noncomputable def fly_distance_from_ceiling : ℝ :=
  let x := 3
  let y := 5
  let d := 17
  let z := Real.sqrt (d^2 - (x^2 + y^2))
  z

theorem fly_dist_ceiling_eq_sqrt255 :
  fly_distance_from_ceiling = Real.sqrt 255 :=
by
  sorry

end fly_dist_ceiling_eq_sqrt255_l106_106603


namespace derivative_of_even_function_is_odd_l106_106610

variables {R : Type*}

-- Definitions and Conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem derivative_of_even_function_is_odd (f g : ℝ → ℝ) (h1 : even_function f) (h2 : ∀ x, deriv f x = g x) : odd_function g :=
sorry

end derivative_of_even_function_is_odd_l106_106610


namespace female_participation_fraction_l106_106465

noncomputable def fraction_of_females (males_last_year : ℕ) (females_last_year : ℕ) : ℚ :=
  let males_this_year := (1.10 * males_last_year : ℚ)
  let females_this_year := (1.25 * females_last_year : ℚ)
  females_this_year / (males_this_year + females_this_year)

theorem female_participation_fraction
  (males_last_year : ℕ) (participation_increase : ℚ)
  (males_increase : ℚ) (females_increase : ℚ)
  (h_males_last_year : males_last_year = 30)
  (h_participation_increase : participation_increase = 1.15)
  (h_males_increase : males_increase = 1.10)
  (h_females_increase : females_increase = 1.25)
  (h_females_last_year : females_last_year = 15) :
  fraction_of_females males_last_year females_last_year = 19 / 52 := by
  sorry

end female_participation_fraction_l106_106465


namespace ratio_P_K_is_2_l106_106212

theorem ratio_P_K_is_2 (P K M : ℝ) (r : ℝ)
  (h1: P + K + M = 153)
  (h2: P = r * K)
  (h3: P = (1/3) * M)
  (h4: M = K + 85) : r = 2 :=
  sorry

end ratio_P_K_is_2_l106_106212


namespace initial_people_in_gym_l106_106923

variable (W A : ℕ)

theorem initial_people_in_gym (W A : ℕ) (h : W + A + 5 + 2 - 3 - 4 + 2 = 20) : W + A = 18 := by
  sorry

end initial_people_in_gym_l106_106923


namespace stratified_sampling_l106_106016

theorem stratified_sampling (n : ℕ) (h_ratio : 2 + 3 + 5 = 10) (h_sample : (5 : ℚ) / 10 = 150 / n) : n = 300 :=
by
  sorry

end stratified_sampling_l106_106016


namespace non_degenerate_ellipse_condition_l106_106081

theorem non_degenerate_ellipse_condition (k : ℝ) :
  (∃ (x y : ℝ), x^2 + 9 * y^2 - 6 * x + 18 * y = k) → k > -9 :=
by
  sorry

end non_degenerate_ellipse_condition_l106_106081


namespace parabola_focus_coincides_ellipse_focus_l106_106507

theorem parabola_focus_coincides_ellipse_focus (p : ℝ) :
  (∃ F : ℝ × ℝ, F = (2, 0) ∧ ∀ x y : ℝ, y^2 = 2 * p * x <-> x = p / 2)
  → p = 4 := 
by
  sorry 

end parabola_focus_coincides_ellipse_focus_l106_106507


namespace find_P_l106_106897

theorem find_P (P : ℕ) (h : P^2 + P = 30) : P = 5 :=
sorry

end find_P_l106_106897


namespace part1_solution_set_part2_range_of_a_l106_106489

open Real

-- For part (1)
theorem part1_solution_set (x a : ℝ) (h : a = 3) : |2 * x - a| + a ≤ 6 ↔ 0 ≤ x ∧ x ≤ 3 := 
by {
  sorry
}

-- For part (2)
theorem part2_range_of_a (f g : ℝ → ℝ) (hf : ∀ x, f x = |2 * x - a| + a) (hg : ∀ x, g x = |2 * x - 3|) :
  (∀ x, f x + g x ≥ 5) ↔ a ≥ 11 / 3 :=
by {
  sorry
}

end part1_solution_set_part2_range_of_a_l106_106489


namespace quadratic_real_roots_iff_l106_106587

theorem quadratic_real_roots_iff (α : ℝ) : (∃ x : ℝ, x^2 - 2 * x + α = 0) ↔ α ≤ 1 :=
by
  sorry

end quadratic_real_roots_iff_l106_106587


namespace mustard_found_at_second_table_l106_106064

variables (total_mustard first_table third_table second_table : ℝ)

def mustard_found (total_mustard first_table third_table : ℝ) := total_mustard - (first_table + third_table)

theorem mustard_found_at_second_table
    (h_total : total_mustard = 0.88)
    (h_first : first_table = 0.25)
    (h_third : third_table = 0.38) :
    mustard_found total_mustard first_table third_table = 0.25 :=
by
    rw [mustard_found, h_total, h_first, h_third]
    simp
    sorry

end mustard_found_at_second_table_l106_106064


namespace inequality_of_f_on_angles_l106_106687

noncomputable def f : ℝ → ℝ := sorry -- Define f as a noncomputable function

-- Stating the properties of the function f
axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodic_function : ∀ x : ℝ, f (x + 1) = -f x
axiom decreasing_interval : ∀ x y : ℝ, (-3 ≤ x ∧ x < y ∧ y ≤ -2) → f x > f y

-- Stating the properties of the angles α and β
variables (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) (hαβ : α ≠ β)

-- The proof statement we want to prove
theorem inequality_of_f_on_angles : f (Real.sin α) > f (Real.cos β) :=
sorry -- The proof is omitted

end inequality_of_f_on_angles_l106_106687


namespace inverse_proportion_quadrants_l106_106292

theorem inverse_proportion_quadrants (k : ℝ) :
  (∀ (x : ℝ), x ≠ 0 → ((x > 0 → (k - 3) / x > 0) ∧ (x < 0 → (k - 3) / x < 0))) → k > 3 :=
by
  intros h
  sorry

end inverse_proportion_quadrants_l106_106292


namespace units_digit_two_pow_2010_l106_106017

-- Conditions from part a)
def two_power_units_digit (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 6
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | _ => 0 -- This case will not occur due to modulo operation

-- Question translated to a proof problem
theorem units_digit_two_pow_2010 : (two_power_units_digit 2010) = 4 :=
by 
  -- Proof would go here
  sorry

end units_digit_two_pow_2010_l106_106017


namespace factorize_expression_l106_106217

theorem factorize_expression (x : ℝ) : 2 * x^3 - 8 * x^2 + 8 * x = 2 * x * (x - 2) ^ 2 := 
sorry

end factorize_expression_l106_106217


namespace z_sum_of_squares_eq_101_l106_106360

open Complex

noncomputable def z_distances_sum_of_squares (z : ℂ) (h : abs (z - (3 + -3 * I)) = 3) : ℝ :=
  abs (z - (1 + 1 * I)) ^ 2 + abs (z - (5 - 5 * I)) ^ 2

theorem z_sum_of_squares_eq_101 (z : ℂ) (h : abs (z - (3 + -3 * I)) = 3) : 
  z_distances_sum_of_squares z h = 101 :=
by
  sorry

end z_sum_of_squares_eq_101_l106_106360


namespace number_of_routes_from_P_to_Q_is_3_l106_106798

-- Definitions of the nodes and paths
inductive Node
| P | Q | R | S | T | U | V
deriving DecidableEq, Repr

-- Definition of paths between nodes based on given conditions
def leads_to : Node → Node → Prop
| Node.P, Node.R => True
| Node.P, Node.S => True
| Node.R, Node.T => True
| Node.R, Node.U => True
| Node.S, Node.Q => True
| Node.T, Node.Q => True
| Node.U, Node.V => True
| Node.V, Node.Q => True
| _, _ => False

-- Proof statement: the number of different routes from P to Q
theorem number_of_routes_from_P_to_Q_is_3 : 
  ∃ (n : ℕ), n = 3 ∧ (∀ (route_count : ℕ), route_count = n → 
  ((leads_to Node.P Node.R ∧ leads_to Node.R Node.T ∧ leads_to Node.T Node.Q) ∨ 
   (leads_to Node.P Node.R ∧ leads_to Node.R Node.U ∧ leads_to Node.U Node.V ∧ leads_to Node.V Node.Q) ∨
   (leads_to Node.P Node.S ∧ leads_to Node.S Node.Q))) :=
by
  -- Placeholder proof
  sorry

end number_of_routes_from_P_to_Q_is_3_l106_106798


namespace digit_sum_divisible_by_9_l106_106700

theorem digit_sum_divisible_by_9 (n : ℕ) (h : n < 10) : 
  (8 + 6 + 5 + n + 7 + 4 + 3 + 2) % 9 = 0 ↔ n = 1 := 
by sorry 

end digit_sum_divisible_by_9_l106_106700


namespace sam_age_l106_106924

theorem sam_age (drew_current_age : ℕ) (drew_future_age : ℕ) (sam_future_age : ℕ) : 
  (drew_current_age = 12) → 
  (drew_future_age = drew_current_age + 5) → 
  (sam_future_age = 3 * drew_future_age) → 
  (sam_future_age - 5 = 46) := 
by sorry

end sam_age_l106_106924


namespace haley_cider_pints_l106_106785

noncomputable def apples_per_farmhand_per_hour := 240
noncomputable def working_hours := 5
noncomputable def total_farmhands := 6

noncomputable def golden_delicious_per_pint := 20
noncomputable def pink_lady_per_pint := 40
noncomputable def golden_delicious_ratio := 1
noncomputable def pink_lady_ratio := 2

noncomputable def total_apples := total_farmhands * apples_per_farmhand_per_hour * working_hours
noncomputable def total_parts := golden_delicious_ratio + pink_lady_ratio

noncomputable def golden_delicious_apples := total_apples / total_parts
noncomputable def pink_lady_apples := golden_delicious_apples * pink_lady_ratio

noncomputable def pints_golden_delicious := golden_delicious_apples / golden_delicious_per_pint
noncomputable def pints_pink_lady := pink_lady_apples / pink_lady_per_pint

theorem haley_cider_pints : 
  total_apples = 7200 → 
  golden_delicious_apples = 2400 → 
  pink_lady_apples = 4800 → 
  pints_golden_delicious = 120 → 
  pints_pink_lady = 120 → 
  pints_golden_delicious = pints_pink_lady →
  pints_golden_delicious = 120 :=
by
  sorry

end haley_cider_pints_l106_106785


namespace no_common_root_l106_106849

variables {R : Type*} [OrderedRing R]

def f (x m n : R) := x^2 + m*x + n
def p (x k l : R) := x^2 + k*x + l

theorem no_common_root (k m n l : R) (h1 : k > m) (h2 : m > n) (h3 : n > l) (h4 : l > 0) :
  ¬ ∃ x : R, (f x m n = 0 ∧ p x k l = 0) :=
by
  sorry

end no_common_root_l106_106849


namespace opposite_of_pi_eq_neg_pi_l106_106034

theorem opposite_of_pi_eq_neg_pi (π : Real) (h : π = Real.pi) : -π = -Real.pi :=
by sorry

end opposite_of_pi_eq_neg_pi_l106_106034


namespace total_voters_l106_106105

-- Definitions
def number_of_voters_first_hour (x : ℕ) := x
def percentage_october_22 (x : ℕ) := 35 * x / 100
def percentage_october_29 (x : ℕ) := 65 * x / 100
def additional_voters_october_22 := 80
def final_percentage_october_29 (total_votes : ℕ) := 45 * total_votes / 100

-- Statement
theorem total_voters (x : ℕ) (h1 : percentage_october_22 x + additional_voters_october_22 = 35 * (x + additional_voters_october_22) / 100)
                      (h2 : percentage_october_29 x = 65 * x / 100)
                      (h3 : final_percentage_october_29 (x + additional_voters_october_22) = 45 * (x + additional_voters_october_22) / 100):
  x + additional_voters_october_22 = 260 := 
sorry

end total_voters_l106_106105


namespace correct_A_correct_B_intersection_A_B_complement_B_l106_106671

noncomputable def A : Set ℝ := {x : ℝ | 2 ≤ x ∧ x ≤ 3}
noncomputable def B : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 4}

theorem correct_A : A = {x : ℝ | 2 ≤ x ∧ x ≤ 3} :=
by
  sorry

theorem correct_B : B = {x : ℝ | 1 ≤ x ∧ x ≤ 4} :=
by
  sorry

theorem intersection_A_B : (A ∩ B) = {x : ℝ | 2 ≤ x ∧ x ≤ 3} :=
by
  sorry

theorem complement_B : (Bᶜ) = {x : ℝ | x < 1 ∨ x > 4} :=
by
  sorry

end correct_A_correct_B_intersection_A_B_complement_B_l106_106671


namespace valid_domain_of_x_l106_106386

theorem valid_domain_of_x (x : ℝ) : 
  (x + 1 ≥ 0 ∧ x ≠ 0) ↔ (x ≥ -1 ∧ x ≠ 0) :=
by sorry

end valid_domain_of_x_l106_106386


namespace susan_more_cats_than_bob_l106_106027

-- Given problem: Initial and transaction conditions
def susan_initial_cats : ℕ := 21
def bob_initial_cats : ℕ := 3
def susan_additional_cats : ℕ := 5
def bob_additional_cats : ℕ := 7
def susan_gives_bob_cats : ℕ := 4

-- Declaration to find the difference between Susan's and Bob's cats
def final_susan_cats (initial : ℕ) (additional : ℕ) (given : ℕ) : ℕ := initial + additional - given
def final_bob_cats (initial : ℕ) (additional : ℕ) (received : ℕ) : ℕ := initial + additional + received

-- The proof statement which we need to show
theorem susan_more_cats_than_bob : 
  final_susan_cats susan_initial_cats susan_additional_cats susan_gives_bob_cats - 
  final_bob_cats bob_initial_cats bob_additional_cats susan_gives_bob_cats = 8 := by
  sorry

end susan_more_cats_than_bob_l106_106027


namespace distinct_real_roots_l106_106306

noncomputable def g (x d : ℝ) : ℝ := x^2 + 4*x + d

theorem distinct_real_roots (d : ℝ) :
  (∃! x : ℝ, g (g x d) d = 0) ↔ d = 0 :=
sorry

end distinct_real_roots_l106_106306


namespace minimum_distance_sum_squared_l106_106749

variable (P : ℝ × ℝ)
variable (F₁ F₂ : ℝ × ℝ)

def on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x^2 / 4) + y^2 = 1

def distance_squared (A B : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  (x1 - x2) ^ 2 + (y1 - y2) ^ 2

theorem minimum_distance_sum_squared
  (hP : on_ellipse P)
  (hF1 : F₁ = (2, 0) ∨ F₁ = (-2, 0)) -- Assuming standard position of foci
  (hF2 : F₂ = (2, 0) ∨ F₂ = (-2, 0)) :
  ∃ P : ℝ × ℝ, on_ellipse P ∧ F₁ ≠ F₂ → distance_squared P F₁ + distance_squared P F₂ = 8 :=
by
  sorry

end minimum_distance_sum_squared_l106_106749


namespace proof_question_1_l106_106491

noncomputable def question_1 (x : ℝ) : ℝ :=
  (Real.sin (2 * x) + 2 * (Real.sin x)^2) / (1 - Real.tan x)

theorem proof_question_1 :
  ∀ x : ℝ, (Real.cos (π / 4 + x) = 3 / 5) →
  (17 * π / 12 < x ∧ x < 7 * π / 4) →
  question_1 x = -9 / 20 :=
by
  intros x h1 h2
  sorry

end proof_question_1_l106_106491


namespace sufficient_but_not_necessary_l106_106408

theorem sufficient_but_not_necessary (a b : ℝ) : (ab >= 2) -> a^2 + b^2 >= 4 ∧ ∃ a b : ℝ, a^2 + b^2 >= 4 ∧ ab < 2 := by
  sorry

end sufficient_but_not_necessary_l106_106408


namespace find_g_at_4_l106_106699

def g (x : ℝ) : ℝ := sorry

theorem find_g_at_4 (h : ∀ x : ℝ, g x + 3 * g (1 - x) = 4 * x ^ 2) : g 4 = 5.5 :=
by
  sorry

end find_g_at_4_l106_106699


namespace find_a_b_f_inequality_l106_106695

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x^2

-- a == 1 and b == 1 from the given conditions
theorem find_a_b (e : ℝ) (h_e : e = Real.exp 1) (b : ℝ) (a : ℝ) 
  (h_tangent : ∀ x, f x a = (e - 2) * x + b → a = 1 ∧ b = 1) : a = 1 ∧ b = 1 :=
sorry

-- prove f(x) > x^2 + 4x - 14 for x >= 0
theorem f_inequality (e : ℝ) (h_e : e = Real.exp 1) :
  ∀ x : ℝ, 0 ≤ x → f x 1 > x^2 + 4 * x - 14 :=
sorry

end find_a_b_f_inequality_l106_106695


namespace consecutive_product_even_product_divisible_by_6_l106_106179

theorem consecutive_product_even (n : ℕ) : ∃ k, n * (n + 1) = 2 * k := 
sorry

theorem product_divisible_by_6 (n : ℕ) : 6 ∣ (n * (n + 1) * (2 * n + 1)) :=
sorry

end consecutive_product_even_product_divisible_by_6_l106_106179


namespace number_of_initial_cans_l106_106144

theorem number_of_initial_cans (n : ℕ) (T : ℝ)
  (h1 : T = n * 36.5)
  (h2 : T - (2 * 49.5) = (n - 2) * 30) :
  n = 6 :=
sorry

end number_of_initial_cans_l106_106144


namespace period_of_f_minimum_value_zero_then_a_eq_one_maximum_value_of_f_axis_of_symmetry_l106_106376

noncomputable def f (x a : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x) ^ 2 + a

theorem period_of_f : ∀ a : ℝ, ∀ x : ℝ, f (x + π) a = f x a := 
by sorry

theorem minimum_value_zero_then_a_eq_one : (∀ x : ℝ, f x a ≥ 0) → a = 1 := 
by sorry

theorem maximum_value_of_f : a = 1 → (∀ x : ℝ, f x 1 ≤ 4) :=
by sorry

theorem axis_of_symmetry : a = 1 → ∃ k : ℤ, ∀ x : ℝ, 2 * x + π / 6 = k * π + π / 2 ↔ f x 1 = f 0 1 :=
by sorry

end period_of_f_minimum_value_zero_then_a_eq_one_maximum_value_of_f_axis_of_symmetry_l106_106376


namespace pots_on_each_shelf_l106_106806

variable (x : ℕ)
variable (h1 : 4 * 3 * x = 60)

theorem pots_on_each_shelf : x = 5 := by
  -- proof will go here
  sorry

end pots_on_each_shelf_l106_106806


namespace cone_volume_l106_106331

theorem cone_volume (r h l : ℝ) (π := Real.pi)
  (slant_height : l = 5)
  (lateral_area : π * r * l = 20 * π) :
  (1 / 3) * π * r^2 * h = 16 * π :=
by
  -- Definitions based on conditions
  let slant_height_definition := slant_height
  let lateral_area_definition := lateral_area
  
  -- Need actual proof steps which are omitted using sorry
  sorry

end cone_volume_l106_106331


namespace intersection_S_T_eq_T_l106_106198

noncomputable def S : Set ℝ := { y | ∃ x : ℝ, y = 3^x }
noncomputable def T : Set ℝ := { y | ∃ x : ℝ, y = x^2 + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l106_106198


namespace rectangle_x_satisfy_l106_106322

theorem rectangle_x_satisfy (x : ℝ) (h1 : 3 * x = 3 * x) (h2 : x + 5 = x + 5) (h3 : (3 * x) * (x + 5) = 2 * (3 * x) + 2 * (x + 5)) : x = 1 :=
sorry

end rectangle_x_satisfy_l106_106322


namespace cupboard_cost_price_l106_106356

theorem cupboard_cost_price (C : ℝ) 
  (h1 : ∀ C₀, C = C₀ → C₀ * 0.88 + 1500 = C₀ * 1.12) :
  C = 6250 := by
  sorry

end cupboard_cost_price_l106_106356


namespace largest_three_digit_multiple_of_17_l106_106648

theorem largest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n ≤ 999) ∧ (n ≥ 100) ∧ (17 ∣ n) ∧ ∀ m, ((m > n) ∧ (m ≤ 999) ∧ (m ≥ 100) ∧ (17 ∣ m)) → false :=
by {
  sorry
}

end largest_three_digit_multiple_of_17_l106_106648


namespace value_of_T_l106_106596

theorem value_of_T (T : ℝ) (h : (1 / 3) * (1 / 6) * T = (1 / 4) * (1 / 8) * 120) : T = 67.5 :=
sorry

end value_of_T_l106_106596


namespace carol_can_invite_friends_l106_106228

-- Definitions based on the problem's conditions
def invitations_per_pack := 9
def packs_bought := 5

-- Required proof statement
theorem carol_can_invite_friends :
  invitations_per_pack * packs_bought = 45 :=
by
  sorry

end carol_can_invite_friends_l106_106228


namespace percentage_of_birth_in_june_l106_106955

theorem percentage_of_birth_in_june (total_scientists: ℕ) (born_in_june: ℕ) (h_total: total_scientists = 150) (h_june: born_in_june = 15) : (born_in_june * 100 / total_scientists) = 10 := 
by 
  sorry

end percentage_of_birth_in_june_l106_106955


namespace sandy_nickels_remaining_l106_106824

def original_nickels : ℕ := 31
def nickels_borrowed : ℕ := 20

theorem sandy_nickels_remaining : (original_nickels - nickels_borrowed) = 11 :=
by
  sorry

end sandy_nickels_remaining_l106_106824


namespace simplify_expression_l106_106737

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (1 - x / (x + 1)) / ((x ^ 2 - 1) / (x ^ 2 + 2 * x + 1)) = Real.sqrt 2 / 2 :=
by
  rw [h]
  sorry

end simplify_expression_l106_106737


namespace smallest_possible_value_l106_106970

theorem smallest_possible_value 
  (a : ℂ)
  (h : 8 * a^2 + 6 * a + 2 = 0) :
  ∃ z : ℂ, z = 3 * a + 1 ∧ z.re = -1 / 8 :=
by
  sorry

end smallest_possible_value_l106_106970


namespace largest_sum_fraction_l106_106561

theorem largest_sum_fraction :
  max 
    ((1/3) + (1/2))
    (max 
      ((1/3) + (1/5))
      (max 
        ((1/3) + (1/6))
        (max 
          ((1/3) + (1/9))
          ((1/3) + (1/10))
        )
      )
    ) = 5/6 :=
by sorry

end largest_sum_fraction_l106_106561


namespace area_of_bounded_curve_is_64_pi_l106_106523

noncomputable def bounded_curve_area : Real :=
  let curve_eq (x y : ℝ) : Prop := (2 * x + 3 * y + 5) ^ 2 + (x + 2 * y - 3) ^ 2 = 64
  let S : Real := 64 * Real.pi
  S

theorem area_of_bounded_curve_is_64_pi : bounded_curve_area = 64 * Real.pi := 
by
  sorry

end area_of_bounded_curve_is_64_pi_l106_106523


namespace bowling_ball_weight_l106_106686

noncomputable def weight_of_one_bowling_ball : ℕ := 20

theorem bowling_ball_weight (b c : ℕ) (h1 : 10 * b = 5 * c) (h2 : 3 * c = 120) : b = weight_of_one_bowling_ball := by
  sorry

end bowling_ball_weight_l106_106686


namespace union_A_B_l106_106134

def A : Set ℕ := {2, 3}
def B : Set ℕ := {1, 2}

theorem union_A_B : A ∪ B = {1, 2, 3} := 
by
  sorry

end union_A_B_l106_106134


namespace rope_cut_probability_l106_106915

theorem rope_cut_probability (L : ℝ) (cut_position : ℝ) (P : ℝ) :
  L = 4 → (∀ cut_position, 0 ≤ cut_position ∧ cut_position ≤ L →
  (cut_position ≥ 1.5 ∧ (L - cut_position) ≥ 1.5)) → P = 1 / 4 :=
by
  intros hL hcut
  sorry

end rope_cut_probability_l106_106915


namespace arithmetic_sequence_property_l106_106631

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_property (h1 : is_arithmetic_sequence a)
  (h2 : a 2 + a 4 + a 6 + a 8 + a 10 = 80) :
  a 7 - (1/2) * a 8 = 8 :=
sorry

end arithmetic_sequence_property_l106_106631


namespace cost_of_one_dozen_pens_l106_106097

theorem cost_of_one_dozen_pens
  (x : ℝ)
  (hx : 20 * x = 150) :
  12 * 5 * (150 / 20) = 450 :=
by
  sorry

end cost_of_one_dozen_pens_l106_106097


namespace farm_area_l106_106012

theorem farm_area (length width area : ℝ) 
  (h1 : length = 0.6) 
  (h2 : width = 3 * length) 
  (h3 : area = length * width) : 
  area = 1.08 := 
by 
  sorry

end farm_area_l106_106012


namespace colored_paper_distribution_l106_106879

theorem colored_paper_distribution (F M : ℕ) (h1 : F + M = 24) (h2 : M = 2 * F) (total_sheets : ℕ) (distributed_sheets : total_sheets = 48) : 
  (48 / F) = 6 := by
  sorry

end colored_paper_distribution_l106_106879


namespace kernels_popped_in_first_bag_l106_106497

theorem kernels_popped_in_first_bag :
  ∀ (x : ℕ), 
    (total_kernels : ℕ := 75 + 50 + 100) →
    (total_popped : ℕ := x + 42 + 82) →
    (average_percentage_popped : ℚ := 82) →
    ((total_popped : ℚ) / total_kernels) * 100 = average_percentage_popped →
    x = 61 :=
by
  sorry

end kernels_popped_in_first_bag_l106_106497


namespace arithmetic_sequence_diff_l106_106021

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given condition for the arithmetic sequence
def condition (a : ℕ → ℝ) : Prop := 
  a 2 + a 4 + a 6 + a 8 + a 10 = 80

-- Definition of the common difference
def common_difference (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- The proof problem statement in Lean 4
theorem arithmetic_sequence_diff (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a → condition a → common_difference a d → a 7 - a 8 = -d :=
by
  intros _ _ _
  -- Proof will be conducted here
  sorry

end arithmetic_sequence_diff_l106_106021


namespace johns_profit_l106_106026

noncomputable def earnings : ℕ := 30000
noncomputable def purchase_price : ℕ := 18000
noncomputable def trade_in_value : ℕ := 6000

noncomputable def depreciation : ℕ := purchase_price - trade_in_value
noncomputable def profit : ℕ := earnings - depreciation

theorem johns_profit : profit = 18000 := by
  sorry

end johns_profit_l106_106026


namespace largest_multiple_of_8_less_than_100_l106_106778

theorem largest_multiple_of_8_less_than_100 : ∃ (n : ℕ), n < 100 ∧ 8 ∣ n ∧ ∀ (m : ℕ), m < 100 ∧ 8 ∣ m → m ≤ n :=
sorry

end largest_multiple_of_8_less_than_100_l106_106778


namespace sum_of_fractions_l106_106424

theorem sum_of_fractions :
  (1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) + 1 / (7 * 8) + 1 / (8 * 9)) = (2 / 9) :=
by
  sorry

end sum_of_fractions_l106_106424


namespace translation_4_units_upwards_l106_106447

theorem translation_4_units_upwards (M N : ℝ × ℝ) (hx : M.1 = N.1) (hy_diff : N.2 - M.2 = 4) :
  N = (M.1, M.2 + 4) :=
by
  sorry

end translation_4_units_upwards_l106_106447


namespace molecular_weight_single_mole_l106_106724

theorem molecular_weight_single_mole :
  (∀ (w_7m C6H8O7 : ℝ), w_7m = 1344 → (w_7m / 7) = 192) :=
by
  intros w_7m C6H8O7 h
  sorry

end molecular_weight_single_mole_l106_106724


namespace printingTime_l106_106395

def printerSpeed : ℝ := 23
def pauseTime : ℝ := 2
def totalPages : ℝ := 350

theorem printingTime : (totalPages / printerSpeed) + ((totalPages / 50 - 1) * pauseTime) = 27 := by 
  sorry

end printingTime_l106_106395


namespace number_of_yellow_balloons_l106_106740

-- Define the problem
theorem number_of_yellow_balloons :
  ∃ (Y B : ℕ), 
  B = Y + 1762 ∧ 
  Y + B = 10 * 859 ∧ 
  Y = 3414 :=
by
  -- Proof is skipped, so we use sorry
  sorry

end number_of_yellow_balloons_l106_106740


namespace complement_angle_l106_106834

theorem complement_angle (A : ℝ) (hA : A = 35) : 90 - A = 55 := by
  sorry

end complement_angle_l106_106834


namespace gcd_three_numbers_4557_1953_5115_l106_106286

theorem gcd_three_numbers_4557_1953_5115 : Nat.gcd (Nat.gcd 4557 1953) 5115 = 93 := 
by 
  sorry

end gcd_three_numbers_4557_1953_5115_l106_106286


namespace remainder_of_x_div_9_is_8_l106_106943

variable (x y r : ℕ)
variable (r_lt_9 : r < 9)
variable (h1 : x = 9 * y + r)
variable (h2 : 2 * x = 14 * y + 1)
variable (h3 : 5 * y - x = 3)

theorem remainder_of_x_div_9_is_8 : r = 8 := by
  sorry

end remainder_of_x_div_9_is_8_l106_106943


namespace solve_for_x_l106_106844

theorem solve_for_x (x y : ℤ) (h1 : x + y = 20) (h2 : x - y = 10) : x = 15 := by
  sorry

end solve_for_x_l106_106844


namespace min_questions_to_determine_number_l106_106770

theorem min_questions_to_determine_number : 
  ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 50) → 
  ∃ (q : ℕ), q = 15 ∧ 
  ∀ (primes : ℕ → Prop), 
  (∀ p, primes p → Nat.Prime p ∧ p ≤ 50) → 
  (∀ p, primes p → (n % p = 0 ↔ p ∣ n)) → 
  (∃ m, (∀ k, k < m → primes k → k ∣ n)) :=
sorry

end min_questions_to_determine_number_l106_106770


namespace sum_is_zero_l106_106735

noncomputable def z : ℂ := Complex.cos (3 * Real.pi / 8) + Complex.sin (3 * Real.pi / 8) * Complex.I

theorem sum_is_zero (hz : z^8 = 1) (hz1 : z ≠ 1) :
  (z / (1 + z^3)) + (z^2 / (1 + z^6)) + (z^4 / (1 + z^12)) = 0 :=
by
  sorry

end sum_is_zero_l106_106735


namespace luke_initial_money_l106_106397

def initial_amount (X : ℤ) : Prop :=
  let spent := 11
  let received := 21
  let current_amount := 58
  X - spent + received = current_amount

theorem luke_initial_money : ∃ (X : ℤ), initial_amount X ∧ X = 48 :=
by
  sorry

end luke_initial_money_l106_106397


namespace sufficient_not_necessary_condition_l106_106197

theorem sufficient_not_necessary_condition (x : ℝ) : (x^2 - 2 * x < 0) → (|x - 1| < 2) ∧ ¬( (|x - 1| < 2) → (x^2 - 2 * x < 0)) :=
by sorry

end sufficient_not_necessary_condition_l106_106197


namespace min_distance_sum_l106_106473

open Real EuclideanGeometry

-- Define the parabola y^2 = 4x
noncomputable def parabola (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1 

-- Define the fixed point M
def M : ℝ × ℝ := (2, 3)

-- Define the line l: x = -1
def line_l (P : ℝ × ℝ) : ℝ := abs (P.1 + 1)

-- Define the distance from point P to point M
def distance_to_M (P : ℝ × ℝ) : ℝ := dist P M

-- Define the distance from point P to line l
def distance_to_line (P : ℝ × ℝ) := line_l P 

-- Define the sum of distances
def sum_of_distances (P : ℝ × ℝ) : ℝ := distance_to_M P + distance_to_line P

-- Prove the minimum value of the sum of distances
theorem min_distance_sum : ∃ P, parabola P ∧ sum_of_distances P = sqrt 10 := sorry

end min_distance_sum_l106_106473


namespace solve_for_y_l106_106866

theorem solve_for_y (y : ℕ) (h : 9^y = 3^14) : y = 7 := 
by
  sorry

end solve_for_y_l106_106866


namespace min_value_reciprocal_sum_l106_106066

open Real

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 20) :
  (1 / a + 1 / b) ≥ 1 / 5 :=
by 
  sorry

end min_value_reciprocal_sum_l106_106066


namespace estimate_sqrt_interval_l106_106364

theorem estimate_sqrt_interval : 4 < 2 * Real.sqrt 5 ∧ 2 * Real.sqrt 5 < 5 :=
by
  sorry

end estimate_sqrt_interval_l106_106364


namespace flat_fee_l106_106859

theorem flat_fee (f n : ℝ) 
  (h1 : f + 3 * n = 205) 
  (h2 : f + 6 * n = 350) : 
  f = 60 := 
by
  sorry

end flat_fee_l106_106859


namespace central_angle_is_2_radians_l106_106895

namespace CircleAngle

def radius : ℝ := 2
def arc_length : ℝ := 4

theorem central_angle_is_2_radians : arc_length / radius = 2 := by
  sorry

end CircleAngle

end central_angle_is_2_radians_l106_106895


namespace trapezoid_diagonals_perpendicular_iff_geometric_mean_l106_106307

structure Trapezoid :=
(a b c d e f : ℝ) -- lengths of sides a, b, c, d, and diagonals e, f.
(right_angle : d^2 = a^2 + c^2) -- Condition that makes it a right-angled trapezoid.

theorem trapezoid_diagonals_perpendicular_iff_geometric_mean (T : Trapezoid) :
  (T.e * T.e + T.f * T.f = T.a * T.a + T.b * T.b + T.c * T.c + T.d * T.d) ↔ 
  (T.d * T.d = T.a * T.c) := 
sorry

end trapezoid_diagonals_perpendicular_iff_geometric_mean_l106_106307


namespace math_problem_l106_106266

variable {x y : ℝ}
variable (hx : x ≠ 0) (hy : y ≠ 0) (h := y^2 - 1 / x^2 ≠ 0) (h₁ := x^2 * y^2 ≠ 1)

theorem math_problem :
  (x^2 - 1 / y^2) / (y^2 - 1 / x^2) = x^2 / y^2 :=
sorry

end math_problem_l106_106266


namespace games_bought_at_garage_sale_l106_106974

-- Definitions based on conditions
def games_from_friend : ℕ := 2
def defective_games : ℕ := 2
def good_games : ℕ := 2

-- Prove the number of games bought at the garage sale equals 2
theorem games_bought_at_garage_sale (G : ℕ) 
  (h : games_from_friend + G - defective_games = good_games) : G = 2 :=
by 
  -- use the given information and work out the proof here
  sorry

end games_bought_at_garage_sale_l106_106974


namespace distance_to_x_axis_l106_106979

def point_P : ℝ × ℝ := (2, -3)

theorem distance_to_x_axis : abs (point_P.snd) = 3 := by
  sorry

end distance_to_x_axis_l106_106979


namespace largest_circle_area_215_l106_106694

theorem largest_circle_area_215
  (length width : ℝ)
  (h1 : length = 16)
  (h2 : width = 10)
  (P : ℝ := 2 * (length + width))
  (C : ℝ := P)
  (r : ℝ := C / (2 * Real.pi))
  (A : ℝ := Real.pi * r^2) :
  round A = 215 := by sorry

end largest_circle_area_215_l106_106694


namespace value_of_A_l106_106091

theorem value_of_A 
  (H M A T E: ℤ)
  (H_value: H = 10)
  (MATH_value: M + A + T + H = 35)
  (TEAM_value: T + E + A + M = 42)
  (MEET_value: M + 2*E + T = 38) : 
  A = 21 := 
by 
  sorry

end value_of_A_l106_106091


namespace inequality_proof_l106_106100

theorem inequality_proof (a b : ℝ) (h1 : a ≠ b) (h2 : a < 0) : a < 2 * b - b^2 / a := 
by
  -- mathematical proof goes here
  sorry

end inequality_proof_l106_106100


namespace number_of_zeros_of_f_is_3_l106_106532

def f (x : ℝ) : ℝ := x^3 - 64 * x

theorem number_of_zeros_of_f_is_3 : ∃ x1 x2 x3, (f x1 = 0) ∧ (f x2 = 0) ∧ (f x3 = 0) ∧ (x1 ≠ x2) ∧ (x2 ≠ x3) ∧ (x1 ≠ x3) :=
by
  sorry

end number_of_zeros_of_f_is_3_l106_106532


namespace find_a_l106_106329

noncomputable def collinear (A B C : ℝ × ℝ) : Prop :=
  (B.2 - A.2) * (C.1 - A.1) = (C.2 - A.2) * (B.1 - A.1)

theorem find_a (a : ℕ) (h : collinear (a, 0) (0, a + 4) (1, 3)) : a = 4 :=
by
  sorry

end find_a_l106_106329


namespace inequality_proof_l106_106667

theorem inequality_proof 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (3 / (a^3 + b^3 + c^3)) ≤ 
  ((1 / (a^3 + b^3 + abc)) + (1 / (b^3 + c^3 + abc)) + (1 / (c^3 + a^3 + abc))) ∧ 
  ((1 / (a^3 + b^3 + abc)) + (1 / (b^3 + c^3 + abc)) + (1 / (c^3 + a^3 + abc)) ≤ (1 / (abc))) := 
sorry

end inequality_proof_l106_106667


namespace mayoral_election_l106_106195

theorem mayoral_election :
  ∀ (X Y Z : ℕ), (X = Y + (Y / 2)) → (Y = Z - (2 * Z / 5)) → (Z = 25000) → X = 22500 :=
by
  intros X Y Z h1 h2 h3
  -- Proof here, not necessary for the task
  sorry

end mayoral_election_l106_106195


namespace arithmetic_sequence_sum_l106_106569

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  a 1 + a 2 = -1 →
  a 3 = 4 →
  (a 1 + 2 * d = 4) →
  ∀ n, a n = a 1 + (n - 1) * d →
  a 4 + a 5 = 17 :=
by
  intros h1 h2 h3 h4
  sorry

end arithmetic_sequence_sum_l106_106569


namespace two_digit_integers_satisfy_R_n_eq_R_n_plus_2_l106_106701

def R (n : ℕ) : ℕ := 
  let remainders := List.range' 2 11 |>.map (λ k => n % k)
  remainders.sum

theorem two_digit_integers_satisfy_R_n_eq_R_n_plus_2 :
  let two_digit_numbers := List.range' 10 89
  (two_digit_numbers.filter (λ n => R n = R (n + 2))).length = 2 := 
by
  sorry

end two_digit_integers_satisfy_R_n_eq_R_n_plus_2_l106_106701


namespace inscribed_circle_radius_l106_106057

variable (AB AC BC s K r : ℝ)
variable (AB_eq AC_eq BC_eq : AB = AC ∧ AC = 8 ∧ BC = 7)
variable (s_eq : s = (AB + AC + BC) / 2)
variable (K_eq : K = Real.sqrt (s * (s - AB) * (s - AC) * (s - BC)))
variable (r_eq : r * s = K)

/-- Prove that the radius of the inscribed circle is 23.75 / 11.5 given the conditions of the triangle --/
theorem inscribed_circle_radius :
  AB = 8 → AC = 8 → BC = 7 → 
  s = (AB + AC + BC) / 2 → 
  K = Real.sqrt (s * (s - AB) * (s - AC) * (s - BC)) →
  r * s = K →
  r = (23.75 / 11.5) :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end inscribed_circle_radius_l106_106057


namespace no_integer_solution_l106_106828

theorem no_integer_solution (y : ℤ) : ¬ (-3 * y ≥ y + 9 ∧ 2 * y ≥ 14 ∧ -4 * y ≥ 2 * y + 21) :=
sorry

end no_integer_solution_l106_106828


namespace pentagon_perimeter_l106_106013

-- Problem statement: Given an irregular pentagon with specified side lengths,
-- prove that its perimeter is equal to 52.9 cm.

theorem pentagon_perimeter 
  (a b c d e : ℝ)
  (h1 : a = 5.2)
  (h2 : b = 10.3)
  (h3 : c = 15.8)
  (h4 : d = 8.7)
  (h5 : e = 12.9) 
  : a + b + c + d + e = 52.9 := 
by
  sorry

end pentagon_perimeter_l106_106013


namespace jill_spent_on_other_items_l106_106632

theorem jill_spent_on_other_items {T : ℝ} (h₁ : T > 0)
    (h₁ : 0.5 * T + 0.2 * T + O * T / 100 = T)
    (h₂ : 0.04 * 0.5 * T = 0.02 * T)
    (h₃ : 0 * 0.2 * T = 0)
    (h₄ : 0.08 * O * T / 100 = 0.0008 * O * T)
    (h₅ : 0.044 * T = 0.02 * T + 0 + 0.0008 * O * T) :
  O = 30 := 
sorry

end jill_spent_on_other_items_l106_106632


namespace num_pairs_satisfying_inequality_l106_106743

theorem num_pairs_satisfying_inequality : 
  ∃ (s : Nat), s = 204 ∧ ∀ (m n : ℕ), m > 0 → n > 0 → m^2 + n < 50 → s = 204 :=
by
  sorry

end num_pairs_satisfying_inequality_l106_106743


namespace dice_roll_probability_is_correct_l106_106862

/-- Define the probability calculation based on conditions of the problem. --/
def dice_rolls_probability_diff_by_two (successful_outcomes : ℕ) (total_outcomes : ℕ) : ℚ :=
  successful_outcomes / total_outcomes

/-- Given the problem conditions, there are 8 successful outcomes and 36 total outcomes. --/
theorem dice_roll_probability_is_correct :
  dice_rolls_probability_diff_by_two 8 36 = 2 / 9 :=
by
  sorry

end dice_roll_probability_is_correct_l106_106862


namespace gcd_lcm_sum_l106_106261

-- Define the given numbers
def a1 := 54
def b1 := 24
def a2 := 48
def b2 := 18

-- Define the GCD and LCM functions in Lean
def gcd_ab := Nat.gcd a1 b1
def lcm_cd := Nat.lcm a2 b2

-- Define the final sum
def final_sum := gcd_ab + lcm_cd

-- State the equality that represents the problem
theorem gcd_lcm_sum : final_sum = 150 := by
  sorry

end gcd_lcm_sum_l106_106261


namespace tan_tan_lt_half_l106_106706

noncomputable def tan_half (x : ℝ) : ℝ := Real.tan (x / 2)

theorem tan_tan_lt_half (a b c α β : ℝ) (h1: a + b < 3 * c) (h2: tan_half α * tan_half β = (a + b - c) / (a + b + c)) :
  tan_half α * tan_half β < 1 / 2 := 
sorry

end tan_tan_lt_half_l106_106706


namespace find_3x2y2_l106_106068

theorem find_3x2y2 (x y : ℤ) 
  (h1 : y^2 + 3 * x^2 * y^2 = 30 * x^2 + 517) : 
  3 * x^2 * y^2 = 588 := by
  sorry

end find_3x2y2_l106_106068


namespace track_length_l106_106090

theorem track_length (x : ℝ) : 
  (∃ B S : ℝ, B + S = x ∧ S = (x / 2 - 75) ∧ B = 75 ∧ S + 100 = x / 2 + 25 ∧ B = x / 2 - 50 ∧ B / S = (x / 2 - 50) / 100) → 
  x = 220 :=
by
  sorry

end track_length_l106_106090


namespace part_one_part_two_l106_106911

variable (a b x k : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + 1

theorem part_one:
  f a b (-1) = 0 → f a b x = x^2 + 2 * x + 1 :=
by
  sorry

theorem part_two:
  (∀ x, -3 ≤ x ∧ x ≤ -1 → f 1 2 x > x + k) ↔ k < 1 :=
by
  sorry

end part_one_part_two_l106_106911


namespace meet_time_opposite_directions_catch_up_time_same_direction_l106_106529

def length_of_track := 440
def speed_A := 5
def speed_B := 6

theorem meet_time_opposite_directions :
  (length_of_track / (speed_A + speed_B)) = 40 :=
by
  sorry

theorem catch_up_time_same_direction :
  (length_of_track / (speed_B - speed_A)) = 440 :=
by
  sorry

end meet_time_opposite_directions_catch_up_time_same_direction_l106_106529


namespace race_dead_heat_l106_106741

theorem race_dead_heat (va vb D : ℝ) (hva_vb : va = (15 / 16) * vb) (dist_a : D = D) (dist_b : D = (15 / 16) * D) (race_finish : D / va = (15 / 16) * D / vb) :
  va / vb = 15 / 16 :=
by sorry

end race_dead_heat_l106_106741


namespace digit_ends_with_l106_106789

theorem digit_ends_with (z : ℕ) (h : z = 1 ∨ z = 3 ∨ z = 7 ∨ z = 9) :
  ∀ (k : ℕ), k ≥ 1 → ∃ (n : ℕ), n ≥ 1 ∧ (∃ m : ℕ, (n ^ 9) % (10 ^ k) = z * (10 ^ m)) :=
by
  sorry

end digit_ends_with_l106_106789


namespace max_stamps_l106_106831

theorem max_stamps (price_per_stamp : ℕ) (total_cents : ℕ) (h1 : price_per_stamp = 45) (h2 : total_cents = 5000) : 
  ∃ n : ℕ, n ≤ total_cents / price_per_stamp ∧ n = 111 :=
by
  sorry

end max_stamps_l106_106831


namespace avg_growth_rate_equation_l106_106291

/-- This theorem formalizes the problem of finding the equation for the average growth rate of working hours.
    Given that the average working hours in the first week are 40 hours and in the third week are 48.4 hours,
    we need to show that the equation for the growth rate \(x\) satisfies \( 40(1 + x)^2 = 48.4 \). -/
theorem avg_growth_rate_equation (x : ℝ) (first_week_hours third_week_hours : ℝ) 
  (h1: first_week_hours = 40) (h2: third_week_hours = 48.4) :
  40 * (1 + x) ^ 2 = 48.4 :=
sorry

end avg_growth_rate_equation_l106_106291


namespace discount_percentage_is_30_l106_106170

theorem discount_percentage_is_30 
  (price_per_pant : ℝ) (num_of_pants : ℕ)
  (price_per_sock : ℝ) (num_of_socks : ℕ)
  (total_spend_after_discount : ℝ)
  (original_pants_price := num_of_pants * price_per_pant)
  (original_socks_price := num_of_socks * price_per_sock)
  (original_total_price := original_pants_price + original_socks_price)
  (discount_amount := original_total_price - total_spend_after_discount)
  (discount_percentage := (discount_amount / original_total_price) * 100) :
  (price_per_pant = 110) ∧ 
  (num_of_pants = 4) ∧ 
  (price_per_sock = 60) ∧ 
  (num_of_socks = 2) ∧ 
  (total_spend_after_discount = 392) →
  discount_percentage = 30 := by
  sorry

end discount_percentage_is_30_l106_106170


namespace gcd_lcm_8951_4267_l106_106404

theorem gcd_lcm_8951_4267 :
  gcd 8951 4267 = 1 ∧ lcm 8951 4267 = 38212917 :=
by
  sorry

end gcd_lcm_8951_4267_l106_106404


namespace lesser_number_is_32_l106_106296

variable (x y : ℕ)

theorem lesser_number_is_32 (h1 : y = 2 * x) (h2 : x + y = 96) : x = 32 := 
sorry

end lesser_number_is_32_l106_106296


namespace evaluate_polynomial_at_2_l106_106914

theorem evaluate_polynomial_at_2 : (2^4 + 2^3 + 2^2 + 2 + 1) = 31 := 
by 
  sorry

end evaluate_polynomial_at_2_l106_106914


namespace history_books_count_l106_106484

theorem history_books_count :
  ∃ (total_books reading_books math_books science_books history_books : ℕ),
    total_books = 10 ∧
    reading_books = (2 * total_books) / 5 ∧
    math_books = (3 * total_books) / 10 ∧
    science_books = math_books - 1 ∧
    history_books = total_books - (reading_books + math_books + science_books) ∧
    history_books = 1 :=
by
  sorry

end history_books_count_l106_106484


namespace magazines_in_third_pile_l106_106807

-- Define the number of magazines in each pile.
def pile1 := 3
def pile2 := 4
def pile4 := 9
def pile5 := 13

-- Define the differences between the piles.
def diff2_1 := pile2 - pile1  -- Difference between second and first pile
def diff4_2 := pile4 - pile2  -- Difference between fourth and second pile

-- Assume the pattern continues with differences increasing by 4.
def diff3_2 := diff2_1 + 4    -- Difference between third and second pile

-- Define the number of magazines in the third pile.
def pile3 := pile2 + diff3_2

-- Theorem stating the number of magazines in the third pile.
theorem magazines_in_third_pile : pile3 = 9 := by sorry

end magazines_in_third_pile_l106_106807


namespace candy_left_l106_106830

-- Definitions according to the conditions
def initialCandy : ℕ := 15
def candyGivenToHaley : ℕ := 6

-- Theorem statement formalizing the proof problem
theorem candy_left (c : ℕ) (h₁ : c = initialCandy - candyGivenToHaley) : c = 9 :=
by
  -- The proof is omitted as instructed.
  sorry

end candy_left_l106_106830


namespace valid_numbers_l106_106411

-- Define the conditions for three-digit numbers
def isThreeDigitNumber (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000

-- Define the splitting cases and the required property
def satisfiesFirstCase (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a ≠ 0 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧
  n = 100 * a + 10 * b + c ∧
  3 * ((10 * a + b) * c) = n

def satisfiesSecondCase (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a ≠ 0 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧
  n = 100 * a + 10 * b + c ∧
  3 * (a * (10 * b + c)) = n

-- Define the main proposition
def validThreeDigitNumber (n : ℕ) : Prop :=
  isThreeDigitNumber n ∧ (satisfiesFirstCase n ∨ satisfiesSecondCase n)

-- The theorem statement which we need to prove
theorem valid_numbers : ∀ n : ℕ, validThreeDigitNumber n ↔ n = 150 ∨ n = 240 ∨ n = 735 :=
by
  sorry

end valid_numbers_l106_106411


namespace mul_large_numbers_l106_106132

theorem mul_large_numbers : 300000 * 300000 * 3 = 270000000000 := by
  sorry

end mul_large_numbers_l106_106132


namespace sqrt_170569_sqrt_175561_l106_106449

theorem sqrt_170569 : Nat.sqrt 170569 = 413 := 
by 
  sorry 

theorem sqrt_175561 : Nat.sqrt 175561 = 419 := 
by 
  sorry

end sqrt_170569_sqrt_175561_l106_106449


namespace investigate_local_extrema_l106_106763

noncomputable def f (x1 x2 : ℝ) : ℝ :=
  3 * x1^2 * x2 - x1^3 - (4 / 3) * x2^3

def is_local_maximum (f : ℝ → ℝ → ℝ) (c : ℝ × ℝ) : Prop :=
  ∃ ε > 0, ∀ (x y : ℝ × ℝ), dist x c < ε → f x.1 x.2 ≤ f c.1 c.2

def is_saddle_point (f : ℝ → ℝ → ℝ) (c : ℝ × ℝ) : Prop :=
  ∃ ε > 0, ∃ (x1 y1 x2 y2 : ℝ × ℝ),
    dist x1 c < ε ∧ dist y1 c < ε ∧ dist x2 c < ε ∧ dist y2 c < ε ∧
    (f x1.1 x1.2 > f c.1 c.2 ∧ f y1.1 y1.2 < f c.1 c.2) ∧
    (f x2.1 x2.2 < f c.1 c.2 ∧ f y2.1 y2.2 > f c.1 c.2)

theorem investigate_local_extrema :
  is_local_maximum f (6, 3) ∧ is_saddle_point f (0, 0) :=
sorry

end investigate_local_extrema_l106_106763


namespace symmetric_line_eq_l106_106446

theorem symmetric_line_eq (x y : ℝ) (h : 2 * x - y = 0) : 2 * x + y = 0 :=
sorry

end symmetric_line_eq_l106_106446


namespace find_k_l106_106083

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x ^ 2 + (k - 1) * x + 3

theorem find_k (k : ℝ) (h : ∀ x, f k x = f k (-x)) : k = 1 :=
by
  sorry

end find_k_l106_106083


namespace rationalize_denominator_l106_106520

noncomputable def cube_root (x : ℝ) := x^(1/3)

theorem rationalize_denominator (a b : ℝ) (h : cube_root 27 = 3) : 
  1 / (cube_root 3 + cube_root 27) = (3 - cube_root 3) / (9 - 3 * cube_root 3) :=
by
  sorry

end rationalize_denominator_l106_106520


namespace max_side_of_triangle_l106_106784

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l106_106784


namespace calculate_expression_l106_106289

theorem calculate_expression : (3.15 * 2.5) - 1.75 = 6.125 := 
by
  -- The proof is omitted, indicated by sorry
  sorry

end calculate_expression_l106_106289


namespace calculate_triple_hash_l106_106515

def hash (N : ℝ) : ℝ := 0.5 * N - 2

theorem calculate_triple_hash : hash (hash (hash 100)) = 9 := by
  sorry

end calculate_triple_hash_l106_106515


namespace one_belt_one_road_l106_106774

theorem one_belt_one_road (m n : ℝ) :
  (∀ x y : ℝ, y = x^2 - 2 * x + n ↔ (x, y) ∈ { p : ℝ × ℝ | p.1 = 0 ∧ p.2 = 1 }) →
  (∀ x y : ℝ, y = m * x + 1 ↔ (x, y) ∈ { q : ℝ × ℝ | q.1 = 0 ∧ q.2 = 1 }) →
  (∀ x y : ℝ, y = x^2 - 2 * x + 1 → y = 0) →
  m = -1 ∧ n = 1 :=
by
  intros h1 h2 h3
  sorry

end one_belt_one_road_l106_106774


namespace find_square_sum_l106_106018

theorem find_square_sum (x y : ℝ) (h1 : 2 * x * (x + y) = 54) (h2 : 3 * y * (x + y) = 81) : (x + y) ^ 2 = 135 :=
sorry

end find_square_sum_l106_106018


namespace number_of_bedrooms_l106_106431

-- Conditions
def battery_life : ℕ := 10
def vacuum_time_per_room : ℕ := 4
def num_initial_rooms : ℕ := 2 -- kitchen and living room
def num_charges : ℕ := 2

-- Computation of total vacuuming time
def total_vacuuming_time : ℕ := battery_life * (num_charges + 1)

-- Computation of remaining time for bedrooms
def time_for_bedrooms : ℕ := total_vacuuming_time - (vacuum_time_per_room * num_initial_rooms)

-- Proof problem: Prove number of bedrooms
theorem number_of_bedrooms (B : ℕ) (h : B = time_for_bedrooms / vacuum_time_per_room) : B = 5 := by 
  sorry

end number_of_bedrooms_l106_106431


namespace first_lock_stall_time_eq_21_l106_106312

-- Definitions of time taken by locks
def firstLockTime : ℕ := 21 -- This will be proven at the end

variables {x : ℕ} -- time for the first lock
variables (secondLockTime : ℕ) (bothLocksTime : ℕ)

-- Conditions given in the problem
axiom lock_relation : secondLockTime = 3 * x - 3
axiom second_lock_time : secondLockTime = 60
axiom combined_locks_time : bothLocksTime = 300

-- Question: Prove that the first lock time is 21 minutes
theorem first_lock_stall_time_eq_21 :
  (bothLocksTime = 5 * secondLockTime) ∧ (secondLockTime = 60) ∧ (bothLocksTime = 300) → x = 21 :=
sorry

end first_lock_stall_time_eq_21_l106_106312


namespace remainder_mod_7_l106_106117

theorem remainder_mod_7 : (4 * 6^24 + 3^48) % 7 = 5 := by
  sorry

end remainder_mod_7_l106_106117


namespace find_truck_weight_l106_106873

variable (T Tr : ℝ)

def weight_condition_1 : Prop := T + Tr = 7000
def weight_condition_2 : Prop := Tr = 0.5 * T - 200

theorem find_truck_weight (h1 : weight_condition_1 T Tr) 
                           (h2 : weight_condition_2 T Tr) : 
  T = 4800 :=
sorry

end find_truck_weight_l106_106873


namespace max_f_value_inequality_m_n_l106_106383

section
variable (x : ℝ)

def f (x : ℝ) := abs (x - 1) - 2 * abs (x + 1)

theorem max_f_value : ∃ k, (∀ x : ℝ, f x ≤ k) ∧ (∃ x₀ : ℝ, f x₀ = k) ∧ k = 2 := 
by sorry

theorem inequality_m_n (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : 1 / m + 1 / (2 * n) = 2) :
  m + 2 * n ≥ 2 :=
by sorry

end

end max_f_value_inequality_m_n_l106_106383


namespace mildred_oranges_l106_106172

theorem mildred_oranges (original after given : ℕ) (h1 : original = 77) (h2 : after = 79) (h3 : given = after - original) : given = 2 :=
by
  sorry

end mildred_oranges_l106_106172


namespace ratio_Florence_Rene_l106_106930

theorem ratio_Florence_Rene :
  ∀ (I F R : ℕ), R = 300 → F = k * R → I = 1/3 * (F + R + I) → F + R + I = 1650 → F / R = 3 / 2 := 
by 
  sorry

end ratio_Florence_Rene_l106_106930


namespace find_rate_l106_106355

noncomputable def national_bank_interest_rate (total_income: ℚ) (investment_national: ℚ) (investment_additional: ℚ) (additional_rate: ℚ) (total_investment_rate: ℚ): ℚ :=
  (total_income - (investment_additional * additional_rate)) / investment_national

theorem find_rate (total_income: ℚ) (investment_national: ℚ) (investment_additional: ℚ) (additional_rate: ℚ) (total_investment_rate: ℚ) (correct_rate: ℚ):
  investment_national = 2400 → investment_additional = 600 → additional_rate = 0.10 → total_investment_rate = 0.06 → total_income = total_investment_rate * (investment_national + investment_additional) → correct_rate = 0.05 → national_bank_interest_rate total_income investment_national investment_additional additional_rate total_investment_rate = correct_rate :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end find_rate_l106_106355


namespace odd_and_monotonically_decreasing_l106_106476

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≥ f y

theorem odd_and_monotonically_decreasing :
  is_odd (fun x : ℝ => -x^3) ∧ is_monotonically_decreasing (fun x : ℝ => -x^3) :=
by
  sorry

end odd_and_monotonically_decreasing_l106_106476


namespace find_a_and_an_l106_106249

-- Given Sequences
def S (n : ℕ) (a : ℝ) : ℝ := 3^n - a

def is_geometric_sequence (a_n : ℕ → ℝ) : Prop := ∃ a1 q, q ≠ 1 ∧ ∀ n, a_n n = a1 * q^n

-- The main statement
theorem find_a_and_an (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (a : ℝ) :
  (∀ n, S_n n = 3^n - a) ∧ is_geometric_sequence a_n →
  ∃ a, a = 1 ∧ ∀ n, a_n n = 2 * 3^(n-1) :=
by
  sorry

end find_a_and_an_l106_106249


namespace final_price_after_discounts_l106_106725

theorem final_price_after_discounts (original_price : ℝ)
  (first_discount_pct : ℝ) (second_discount_pct : ℝ) (third_discount_pct : ℝ) :
  original_price = 200 → 
  first_discount_pct = 0.40 → 
  second_discount_pct = 0.20 → 
  third_discount_pct = 0.10 → 
  (original_price * (1 - first_discount_pct) * (1 - second_discount_pct) * (1 - third_discount_pct) = 86.40) := 
by
  intros
  sorry

end final_price_after_discounts_l106_106725


namespace equipment_total_cost_l106_106477

-- Definition of costs for each item of equipment
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.20
def socks_cost : ℝ := 6.80

-- Number of players
def num_players : ℕ := 16

-- Total cost for one player
def total_cost_one_player : ℝ := jersey_cost + shorts_cost + socks_cost

-- Total cost for all players
def total_cost_all_players : ℝ := total_cost_one_player * num_players

-- Theorem to prove
theorem equipment_total_cost : total_cost_all_players = 752 := by
  sorry

end equipment_total_cost_l106_106477


namespace interest_rate_is_4_percent_l106_106601

variable (P A n : ℝ)
variable (r : ℝ)
variable (n_pos : n ≠ 0)

-- Define the conditions
def principal : ℝ := P
def amount_after_n_years : ℝ := A
def years : ℝ := n
def interest_rate : ℝ := r

-- The compound interest formula
def compound_interest (P A r : ℝ) (n : ℝ) : Prop :=
  A = P * (1 + r) ^ n

-- The Lean theorem statement
theorem interest_rate_is_4_percent
  (P_val : principal = 7500)
  (A_val : amount_after_n_years = 8112)
  (n_val : years = 2)
  (h : compound_interest P A r n) :
  r = 0.04 :=
sorry

end interest_rate_is_4_percent_l106_106601


namespace part1_part2_part3_l106_106575

-- Definition of the given expression
def expr (a b : ℝ) (x : ℝ) : ℝ := (a * x^2 + b * x + 2) - (5 * x^2 + 3 * x)

-- Condition 1: Given final result 2x^2 - 4x + 2
def target_expr1 (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 2

-- Condition 2: Given values for a and b by Student B
def student_b_expr (x : ℝ) : ℝ := (5 * x^2 - 3 * x + 2) - (5 * x^2 + 3 * x)

-- Condition 3: Result independent of x
def target_expr3 : ℝ := 2

-- Prove conditions and answers
theorem part1 (a b : ℝ) : (∀ x : ℝ, expr a b x = target_expr1 x) → a = 7 ∧ b = -1 :=
sorry

theorem part2 : (∀ x : ℝ, student_b_expr x = -6 * x + 2) :=
sorry

theorem part3 (a b : ℝ) : (∀ x : ℝ, expr a b x = 2) → a = 5 ∧ b = 3 :=
sorry

end part1_part2_part3_l106_106575


namespace find_sum_on_si_l106_106271

noncomputable def sum_invested_on_si (r1 r2 r3 : ℝ) (years_si: ℕ) (ci_rate: ℝ) (principal_ci: ℝ) (years_ci: ℕ) (times_compounded: ℕ) :=
  let ci_rate_period := ci_rate / times_compounded
  let amount_ci := principal_ci * (1 + ci_rate_period / 1)^(years_ci * times_compounded)
  let ci := amount_ci - principal_ci
  let si := ci / 2
  let total_si_rate := r1 / 100 + r2 / 100 + r3 / 100
  let principle_si := si / total_si_rate
  principle_si

theorem find_sum_on_si :
  sum_invested_on_si 0.05 0.06 0.07 3 0.10 4000 2 2 = 2394.51 :=
by
  sorry

end find_sum_on_si_l106_106271


namespace cube_root_sum_is_integer_iff_l106_106629

theorem cube_root_sum_is_integer_iff (n m : ℤ) (hn : n = m * (m^2 + 3) / 2) :
  ∃ (k : ℤ), (n + Real.sqrt (n^2 + 1))^(1/3) + (n - Real.sqrt (n^2 + 1))^(1/3) = k :=
by
  sorry

end cube_root_sum_is_integer_iff_l106_106629


namespace solve_trig_equation_l106_106779

open Real

theorem solve_trig_equation (k : ℕ) :
    (∀ x, 8.459 * cos x^2 * cos (x^2) * (tan (x^2) + 2 * tan x) + tan x^3 * (1 - sin (x^2)^2) * (2 - tan x * tan (x^2)) = 0) ↔
    (∃ k : ℕ, x = -1 + sqrt (π * k + 1) ∨ x = -1 - sqrt (π * k + 1)) :=
sorry

end solve_trig_equation_l106_106779


namespace grid_permutation_exists_l106_106944

theorem grid_permutation_exists (n : ℕ) (grid : Fin n → Fin n → ℤ) 
  (cond1 : ∀ i : Fin n, ∃ unique j : Fin n, grid i j = 1)
  (cond2 : ∀ i : Fin n, ∃ unique j : Fin n, grid i j = -1)
  (cond3 : ∀ j : Fin n, ∃ unique i : Fin n, grid i j = 1)
  (cond4 : ∀ j : Fin n, ∃ unique i : Fin n, grid i j = -1)
  (cond5 : ∀ i j, grid i j = 0 ∨ grid i j = 1 ∨ grid i j = -1) :
  ∃ (perm_rows perm_cols : Fin n → Fin n),
    (∀ i j, grid (perm_rows i) (perm_cols j) = -grid i j) :=
by
  -- Proof goes here
  sorry

end grid_permutation_exists_l106_106944


namespace find_number_l106_106916

theorem find_number (x : ℤ) (h : 3 * x + 4 = 19) : x = 5 :=
by {
  sorry
}

end find_number_l106_106916


namespace grunters_win_all_6_games_l106_106124

-- Define the probability of the Grunters winning a single game
def probability_win_single_game : ℚ := 3 / 5

-- Define the number of games
def number_of_games : ℕ := 6

-- Calculate the probability of winning all games (all games are independent)
def probability_win_all_games (p : ℚ) (n : ℕ) : ℚ := p ^ n

-- Prove that the probability of the Grunters winning all 6 games is exactly 729/15625
theorem grunters_win_all_6_games :
  probability_win_all_games probability_win_single_game number_of_games = 729 / 15625 :=
by
  sorry

end grunters_win_all_6_games_l106_106124


namespace max_perimeter_of_rectangle_with_area_36_l106_106106

theorem max_perimeter_of_rectangle_with_area_36 :
  ∃ l w : ℕ, l * w = 36 ∧ (∀ l' w' : ℕ, l' * w' = 36 → 2 * (l + w) ≥ 2 * (l' + w')) ∧ 2 * (l + w) = 74 := 
sorry

end max_perimeter_of_rectangle_with_area_36_l106_106106


namespace total_cost_of_shirts_is_24_l106_106953

-- Definitions based on conditions
def cost_first_shirt : ℕ := 15
def cost_difference : ℕ := 6
def cost_second_shirt : ℕ := cost_first_shirt - cost_difference

-- The proof problem statement: Calculate total cost given the conditions
theorem total_cost_of_shirts_is_24 : cost_first_shirt + cost_second_shshirt = 24 :=
by
  sorry

end total_cost_of_shirts_is_24_l106_106953


namespace ratio_speed_car_speed_bike_l106_106160

def speed_of_tractor := 575 / 23
def speed_of_bike := 2 * speed_of_tractor
def speed_of_car := 540 / 6
def ratio := speed_of_car / speed_of_bike

theorem ratio_speed_car_speed_bike : ratio = 9 / 5 := by
  sorry

end ratio_speed_car_speed_bike_l106_106160


namespace calculate_seven_a_sq_minus_four_a_sq_l106_106246

variable (a : ℝ)

theorem calculate_seven_a_sq_minus_four_a_sq : 7 * a^2 - 4 * a^2 = 3 * a^2 := 
by
  sorry

end calculate_seven_a_sq_minus_four_a_sq_l106_106246


namespace find_min_n_l106_106881

theorem find_min_n (n k : ℕ) (h : 14 * n = k^2) : n = 14 := sorry

end find_min_n_l106_106881


namespace least_m_lcm_l106_106554

theorem least_m_lcm (m : ℕ) (h : m > 0) : Nat.lcm 15 m = Nat.lcm 42 m → m = 70 := by
  sorry

end least_m_lcm_l106_106554


namespace three_digit_number_is_112_l106_106884

theorem three_digit_number_is_112 (a b c : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 1 ≤ c ∧ c ≤ 9) (h4 : 100 * a + 10 * b + c = 56 * c) :
  100 * a + 10 * b + c = 112 :=
by sorry

end three_digit_number_is_112_l106_106884


namespace area_covered_by_three_layers_l106_106159

theorem area_covered_by_three_layers (A B C : ℕ) (total_wallpaper : ℕ := 300)
  (wall_area : ℕ := 180) (two_layer_coverage : ℕ := 30) :
  A + 2 * B + 3 * C = total_wallpaper ∧ B + C = total_wallpaper - wall_area ∧ B = two_layer_coverage → 
  C = 90 :=
by
  sorry

end area_covered_by_three_layers_l106_106159


namespace amber_age_l106_106845

theorem amber_age 
  (a g : ℕ)
  (h1 : g = 15 * a)
  (h2 : g - a = 70) :
  a = 5 :=
by
  sorry

end amber_age_l106_106845


namespace crayons_given_to_friends_l106_106113

def initial_crayons : ℕ := 440
def lost_crayons : ℕ := 106
def remaining_crayons : ℕ := 223

theorem crayons_given_to_friends :
  initial_crayons - remaining_crayons - lost_crayons = 111 := 
by
  sorry

end crayons_given_to_friends_l106_106113


namespace mother_hubbard_children_l106_106903

theorem mother_hubbard_children :
  (∃ c : ℕ, (2 / 3 : ℚ) = c * (1 / 12 : ℚ)) → c = 8 :=
by
  sorry

end mother_hubbard_children_l106_106903


namespace total_votes_l106_106239

-- Define the conditions
variables (V : ℝ) (votes_second_candidate : ℝ) (percent_second_candidate : ℝ)
variables (h1 : votes_second_candidate = 240)
variables (h2 : percent_second_candidate = 0.30)

-- Statement: The total number of votes is 800 given the conditions.
theorem total_votes (h : percent_second_candidate * V = votes_second_candidate) : V = 800 :=
sorry

end total_votes_l106_106239


namespace c_ge_one_l106_106892

theorem c_ge_one (a b : ℕ) (c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : (a + 1) / (b + c) = b / a) : c ≥ 1 := 
sorry

end c_ge_one_l106_106892


namespace Q_2_plus_Q_neg2_l106_106466

variable {k : ℝ}

noncomputable def Q (x : ℝ) : ℝ := 0 -- Placeholder definition, real polynomial will be defined in proof.

theorem Q_2_plus_Q_neg2 (hQ0 : Q 0 = 2 * k)
  (hQ1 : Q 1 = 3 * k)
  (hQ_minus1 : Q (-1) = 4 * k) :
  Q 2 + Q (-2) = 16 * k :=
sorry

end Q_2_plus_Q_neg2_l106_106466


namespace range_of_x_l106_106168

theorem range_of_x (x : ℝ) (h : 2 * x - 4 ≥ 0) : x ≥ 2 :=
sorry

end range_of_x_l106_106168


namespace find_number_l106_106594

-- Define the condition that k is a non-negative integer
def is_nonnegative_int (k : ℕ) : Prop := k ≥ 0

-- Define the condition that 18^k is a divisor of the number n
def is_divisor (n k : ℕ) : Prop := 18^k ∣ n

-- The main theorem statement
theorem find_number (n k : ℕ) (h_nonneg : is_nonnegative_int k) (h_eq : 6^k - k^6 = 1) (h_div : is_divisor n k) : n = 1 :=
  sorry

end find_number_l106_106594


namespace triangle_ratio_l106_106488

variables (A B C : ℝ) (a b c : ℝ)

theorem triangle_ratio (h_cosB : Real.cos B = 4/5)
    (h_a : a = 5)
    (h_area : 1/2 * a * c * Real.sin B = 12) :
    (a + c) / (Real.sin A + Real.sin C) = 25 / 3 :=
sorry

end triangle_ratio_l106_106488


namespace constant_term_expansion_l106_106833

noncomputable def binom : ℕ → ℕ → ℕ
| n, k => if h : k ≤ n then Nat.choose n k else 0

theorem constant_term_expansion :
    ∀ x: ℂ, (x ≠ 0) → ∃ term: ℂ, 
    term = (-1 : ℂ) * binom 6 4 ∧ term = -15 := 
by
  intros x hx
  use (-1 : ℂ) * binom 6 4
  constructor
  · rfl
  · sorry

end constant_term_expansion_l106_106833


namespace circus_tickets_l106_106273

variable (L U : ℕ)

theorem circus_tickets (h1 : L + U = 80) (h2 : 30 * L + 20 * U = 2100) : L = 50 :=
by
  sorry

end circus_tickets_l106_106273


namespace intersection_M_N_l106_106148

def M (x : ℝ) : Prop := abs (x - 1) ≥ 2

def N (x : ℝ) : Prop := x^2 - 4 * x ≥ 0

def P (x : ℝ) : Prop := x ≤ -1 ∨ x ≥ 4

theorem intersection_M_N (x : ℝ) : (M x ∧ N x) → P x :=
by
  sorry

end intersection_M_N_l106_106148


namespace tilted_rectangle_l106_106655

theorem tilted_rectangle (VWYZ : Type) (YW ZV : ℝ) (ZY VW : ℝ) (W_above_horizontal : ℝ) (Z_height : ℝ) (x : ℝ) :
  YW = 100 → ZV = 100 → ZY = 150 → VW = 150 → W_above_horizontal = 20 → Z_height = (100 + x) →
  x = 67 :=
by
  sorry

end tilted_rectangle_l106_106655


namespace difference_between_max_and_min_change_l106_106553

-- Define percentages as fractions for Lean
def initial_yes : ℚ := 60 / 100
def initial_no : ℚ := 40 / 100
def final_yes : ℚ := 80 / 100
def final_no : ℚ := 20 / 100
def new_students : ℚ := 10 / 100

-- Define the minimum and maximum possible values of changes (in percentage as a fraction)
def min_change : ℚ := 10 / 100
def max_change : ℚ := 50 / 100

-- The theorem we need to prove
theorem difference_between_max_and_min_change : (max_change - min_change) = 40 / 100 :=
by
  sorry

end difference_between_max_and_min_change_l106_106553


namespace tom_mowing_lawn_l106_106233

theorem tom_mowing_lawn (hours_to_mow : ℕ) (time_worked : ℕ) (fraction_mowed_per_hour : ℚ) : 
  (hours_to_mow = 6) → 
  (time_worked = 3) → 
  (fraction_mowed_per_hour = (1 : ℚ) / hours_to_mow) → 
  (1 - (time_worked * fraction_mowed_per_hour) = (1 : ℚ) / 2) :=
by
  intros h1 h2 h3
  sorry

end tom_mowing_lawn_l106_106233


namespace median_and_mode_of_successful_shots_l106_106593

theorem median_and_mode_of_successful_shots :
  let shots := [3, 6, 4, 6, 4, 3, 6, 5, 7]
  let sorted_shots := [3, 3, 4, 4, 5, 6, 6, 6, 7]
  let median := sorted_shots[4]  -- 4 is the index for the 5th element (0-based indexing)
  let mode := 6  -- determined by the number that appears most frequently
  median = 5 ∧ mode = 6 :=
by
  sorry

end median_and_mode_of_successful_shots_l106_106593


namespace problem_inequality_l106_106997

theorem problem_inequality (x y z : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hz : 1 ≤ z) :
  (x^2 - 2*x + 2) * (y^2 - 2*y + 2) * (z^2 - 2*z + 2) ≤ (x*y*z)^2 - 2*(x*y*z) + 2 := sorry

end problem_inequality_l106_106997


namespace quotient_of_division_l106_106799

theorem quotient_of_division (dividend divisor remainder quotient : ℕ)
  (h_dividend : dividend = 15)
  (h_divisor : divisor = 3)
  (h_remainder : remainder = 3)
  (h_relation : dividend = divisor * quotient + remainder) :
  quotient = 4 :=
by sorry

end quotient_of_division_l106_106799


namespace roots_of_quadratic_eq_l106_106390

theorem roots_of_quadratic_eq : ∀ x : ℝ, (x^2 = 9) → (x = 3 ∨ x = -3) :=
by
  sorry

end roots_of_quadratic_eq_l106_106390


namespace trajectory_midpoint_l106_106977

theorem trajectory_midpoint {x y : ℝ} (hx : 2 * y + 1 = 2 * (2 * x)^2 + 1) :
  y = 4 * x^2 := 
by sorry

end trajectory_midpoint_l106_106977


namespace math_problem_l106_106371

variable (f : ℝ → ℝ)

-- Conditions
axiom condition1 : f 1 = 1
axiom condition2 : ∀ x y : ℝ, f (x + y) + f (x - y) = f x * f y

-- Proof goals
theorem math_problem :
  (f 0 = 2) ∧
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x : ℝ, f (x + 6) = f x) :=
by 
  sorry

end math_problem_l106_106371


namespace factorize_expression_l106_106679

theorem factorize_expression (a b : ℝ) : 
  a^2 * b + 2 * a * b^2 + b^3 = b * (a + b)^2 :=
by {
  sorry
}

end factorize_expression_l106_106679


namespace smallest_product_not_factor_of_48_l106_106281

theorem smallest_product_not_factor_of_48 (a b : ℕ) (h1 : a ≠ b) (h2 : a ∣ 48) (h3 : b ∣ 48) (h4 : ¬ (a * b ∣ 48)) : a * b = 32 :=
sorry

end smallest_product_not_factor_of_48_l106_106281


namespace deleted_files_l106_106304

variable {initial_files : ℕ}
variable {files_per_folder : ℕ}
variable {folders : ℕ}

noncomputable def files_deleted (initial_files files_in_folders : ℕ) : ℕ :=
  initial_files - files_in_folders

theorem deleted_files (h1 : initial_files = 27) (h2 : files_per_folder = 6) (h3 : folders = 3) :
  files_deleted initial_files (files_per_folder * folders) = 9 :=
by
  sorry

end deleted_files_l106_106304


namespace ratio_white_to_remaining_l106_106368

def total_beans : ℕ := 572

def red_beans (total : ℕ) : ℕ := total / 4

def remaining_beans_after_red (total : ℕ) (red : ℕ) : ℕ := total - red

def green_beans : ℕ := 143

def remaining_beans_after_green (remaining : ℕ) (green : ℕ) : ℕ := remaining - green

def white_beans (remaining : ℕ) : ℕ := remaining / 2

theorem ratio_white_to_remaining (total : ℕ) (red : ℕ) (remaining : ℕ) (green : ℕ) (white : ℕ) 
  (H_total : total = 572)
  (H_red : red = red_beans total)
  (H_remaining : remaining = remaining_beans_after_red total red)
  (H_green : green = 143)
  (H_remaining_after_green : remaining_beans_after_green remaining green = white)
  (H_white : white = white_beans remaining) :
  (white : ℚ) / (remaining : ℚ) = (1 : ℚ) / 2 := 
by sorry

end ratio_white_to_remaining_l106_106368


namespace solve_diff_l106_106771

-- Definitions based on conditions
def equation (e y : ℝ) : Prop := y^2 + e^2 = 3 * e * y + 1

theorem solve_diff (e a b : ℝ) (h1 : equation e a) (h2 : equation e b) (h3 : a ≠ b) : 
  |a - b| = Real.sqrt (5 * e^2 + 4) := 
sorry

end solve_diff_l106_106771


namespace find_prime_powers_l106_106513

open Nat

theorem find_prime_powers (p x y : ℕ) (hp : p.Prime) (hx : 0 < x) (hy : 0 < y) :
  p^x = y^3 + 1 ↔
  (p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2) :=
sorry

end find_prime_powers_l106_106513


namespace jane_buys_4_bagels_l106_106138

theorem jane_buys_4_bagels (b m : ℕ) (h1 : b + m = 7) (h2 : (80 * b + 60 * m) % 100 = 0) : b = 4 := 
by sorry

end jane_buys_4_bagels_l106_106138


namespace evaluate_g_at_neg_one_l106_106499

def g (x : ℝ) : ℝ := 5 * x^3 - 7 * x^2 - 3 * x + 9

theorem evaluate_g_at_neg_one : g (-1) = 7 :=
by 
  -- lean proof here
  sorry

end evaluate_g_at_neg_one_l106_106499


namespace arithmetic_square_root_l106_106533

theorem arithmetic_square_root (n : ℝ) (h : (-5)^2 = n) : Real.sqrt n = 5 :=
by
  sorry

end arithmetic_square_root_l106_106533


namespace book_arrangement_l106_106814

theorem book_arrangement : (Nat.choose 7 3 = 35) :=
by
  sorry

end book_arrangement_l106_106814


namespace triangle_inequality_l106_106727

theorem triangle_inequality 
(a b c : ℝ) (α β γ : ℝ)
(h_t : a + b > c ∧ a + c > b ∧ b + c > a)
(h_opposite : 0 < α ∧ α < π ∧ 0 < β ∧ β < π ∧ 0 < γ ∧ γ < π ∧ α + β + γ = π) :
  a * α + b * β + c * γ ≥ a * β + b * γ + c * α :=
sorry

end triangle_inequality_l106_106727


namespace inequality_solution_l106_106894

theorem inequality_solution (x : ℝ) (h : x ≠ 2 ∧ x ≠ -2) :
    (x^2 - 9) / (x^2 - 4) > 0 ↔ (x < -3 ∨ x > 3) := by
  sorry

end inequality_solution_l106_106894


namespace remainder_of_division_l106_106382

theorem remainder_of_division :
  Nat.mod 4536 32 = 24 :=
sorry

end remainder_of_division_l106_106382


namespace circles_intersect_l106_106403

section PositionalRelationshipCircles

-- Define the first circle O1 with center (1, 0) and radius 1
def Circle1 (p : ℝ × ℝ) : Prop := (p.1 - 1)^2 + p.2^2 = 1

-- Define the second circle O2 with center (0, 3) and radius 3
def Circle2 (p : ℝ × ℝ) : Prop := p.1^2 + (p.2 - 3)^2 = 9

-- Prove that the positional relationship between Circle1 and Circle2 is intersecting
theorem circles_intersect : 
  ∃ p : ℝ × ℝ, Circle1 p ∧ Circle2 p :=
sorry

end PositionalRelationshipCircles

end circles_intersect_l106_106403


namespace find_k_l106_106588

theorem find_k (x k : ℝ) (h1 : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) (h2 : k ≠ 0) : k = 6 :=
by
  sorry

end find_k_l106_106588


namespace parallel_vectors_m_eq_neg3_l106_106875

theorem parallel_vectors_m_eq_neg3 : 
  ∀ m : ℝ, (∀ (a b : ℝ × ℝ), a = (1, -2) → b = (1 + m, 1 - m) → a.1 * b.2 - a.2 * b.1 = 0) → m = -3 :=
by
  intros m h_par
  specialize h_par (1, -2) (1 + m, 1 - m) rfl rfl
  -- We need to show m = -3
  sorry

end parallel_vectors_m_eq_neg3_l106_106875


namespace max_value_f_l106_106931

noncomputable def op_add (a b : ℝ) : ℝ :=
if a >= b then a else b^2

noncomputable def f (x : ℝ) : ℝ :=
(op_add 1 x) + (op_add 2 x)

theorem max_value_f :
  ∃ x ∈ Set.Icc (-2 : ℝ) 3, ∀ y ∈ Set.Icc (-2 : ℝ) 3, f y ≤ f x := 
sorry

end max_value_f_l106_106931


namespace painting_cost_conversion_l106_106205

def paintingCostInCNY (paintingCostNAD : ℕ) (usd_to_nad : ℕ) (usd_to_cny : ℕ) : ℕ :=
  paintingCostNAD * (1 / usd_to_nad) * usd_to_cny

theorem painting_cost_conversion :
  (paintingCostInCNY 105 7 6 = 90) :=
by
  sorry

end painting_cost_conversion_l106_106205


namespace simplify_expression_l106_106086

variable (x y : ℝ)

theorem simplify_expression : (x^2 + x * y) / (x * y) * (y^2 / (x + y)) = y := by
  sorry

end simplify_expression_l106_106086


namespace area_of_given_triangle_is_32_l106_106350

noncomputable def area_of_triangle : ℕ :=
  let A := (-8, 0)
  let B := (0, 8)
  let C := (0, 0)
  1 / 2 * (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2) : ℤ).natAbs

theorem area_of_given_triangle_is_32 : area_of_triangle = 32 := 
  sorry

end area_of_given_triangle_is_32_l106_106350


namespace fraction_n_p_l106_106325

theorem fraction_n_p (m n p : ℝ) (r1 r2 : ℝ)
  (h1 : r1 * r2 = m)
  (h2 : -(r1 + r2) = p)
  (h3 : m ≠ 0)
  (h4 : n ≠ 0)
  (h5 : p ≠ 0)
  (h6 : m = - (r1 + r2) / 2)
  (h7 : n = r1 * r2 / 4) :
  n / p = 1 / 8 :=
by
  sorry

end fraction_n_p_l106_106325


namespace intersection_A_B_l106_106483

namespace SetTheory

open Set

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {0, 1} :=
by
  sorry

end SetTheory

end intersection_A_B_l106_106483


namespace document_completion_time_l106_106531

-- Define the typing rates for different typists
def fast_typist_rate := 1 / 4
def slow_typist_rate := 1 / 9
def additional_typist_rate := 1 / 4

-- Define the number of typists
def num_fast_typists := 2
def num_slow_typists := 3
def num_additional_typists := 2

-- Define the distraction time loss per typist every 30 minutes
def distraction_loss := 1 / 6

-- Define the combined rate without distractions
def combined_rate : ℚ :=
  (num_fast_typists * fast_typist_rate) +
  (num_slow_typists * slow_typist_rate) +
  (num_additional_typists * additional_typist_rate)

-- Define the distraction rate loss per hour (two distractions per hour)
def distraction_rate_loss_per_hour := 2 * distraction_loss

-- Define the effective combined rate considering distractions
def effective_combined_rate : ℚ := combined_rate - distraction_rate_loss_per_hour

-- Prove that the document is completed in 1 hour with the effective rate
theorem document_completion_time :
  effective_combined_rate = 1 :=
sorry

end document_completion_time_l106_106531


namespace distance_between_points_l106_106131

noncomputable def distance (A B : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := A
  let (x2, y2, z2) := B
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2)

theorem distance_between_points :
  distance (-3, 4, 0) (2, -1, 6) = Real.sqrt 86 :=
by
  sorry

end distance_between_points_l106_106131


namespace find_abc_value_l106_106128

open Real

/- Defining the conditions -/
variables (a b c : ℝ)
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
variables (h4 : a * (b + c) = 156) (h5 : b * (c + a) = 168) (h6 : c * (a + b) = 176)

/- Prove the value of abc -/
theorem find_abc_value :
  a * b * c = 754 :=
sorry

end find_abc_value_l106_106128


namespace probability_event_in_single_trial_l106_106609

theorem probability_event_in_single_trial (p : ℝ) 
  (h1 : 0 ≤ p ∧ p ≤ 1) 
  (h2 : (1 - p)^4 = 16 / 81) : 
  p = 1 / 3 :=
sorry

end probability_event_in_single_trial_l106_106609


namespace charlie_first_week_usage_l106_106640

noncomputable def data_used_week1 : ℕ :=
  let data_plan := 8
  let week2_usage := 3
  let week3_usage := 5
  let week4_usage := 10
  let total_extra_cost := 120
  let cost_per_gb_extra := 10
  let total_data_used := data_plan + (total_extra_cost / cost_per_gb_extra)
  let total_data_week_2_3_4 := week2_usage + week3_usage + week4_usage
  total_data_used - total_data_week_2_3_4

theorem charlie_first_week_usage : data_used_week1 = 2 :=
by
  sorry

end charlie_first_week_usage_l106_106640


namespace compute_expression_l106_106262

noncomputable def quadratic_roots (p q : ℝ) (α β γ δ : ℝ) : Prop :=
  (α * β = -2) ∧ (α + β = -p) ∧ (γ * δ = -2) ∧ (γ + δ = -q)

theorem compute_expression (p q α β γ δ : ℝ) 
  (h₁ : quadratic_roots p q α β γ δ) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = -(p^2 - q^2) :=
by
  -- We will provide the proof here
  sorry

end compute_expression_l106_106262


namespace problem_statement_l106_106801

def Delta (a b : ℝ) : ℝ := a^2 - b

theorem problem_statement : Delta (2 ^ (Delta 5 8)) (4 ^ (Delta 2 7)) = 17179869183.984375 := by
  sorry

end problem_statement_l106_106801


namespace men_days_proof_l106_106634

noncomputable def time_to_complete (m d e r : ℕ) : ℕ :=
  (m * d) / (e * (m + r))

theorem men_days_proof (m d e r t : ℕ) (h1 : d = (m * d) / (m * e))
  (h2 : t = (m * d) / (e * (m + r))) :
  t = (m * d) / (e * (m + r)) :=
by
  -- The proof would go here
  sorry

end men_days_proof_l106_106634


namespace triangle_area_l106_106638

theorem triangle_area (a b c : ℝ) (h1 : a = 15) (h2 : b = 36) (h3 : c = 39) (h4 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 270 :=
by
  sorry

end triangle_area_l106_106638


namespace woman_stop_time_l106_106169

-- Conditions
def man_speed := 5 -- in miles per hour
def woman_speed := 15 -- in miles per hour
def wait_time := 4 -- in minutes
def man_speed_mpm : ℚ := man_speed * (1 / 60) -- convert to miles per minute
def distance_covered := man_speed_mpm * wait_time

-- Definition of the relative speed between the woman and the man
def relative_speed := woman_speed - man_speed
def relative_speed_mpm : ℚ := relative_speed * (1 / 60) -- convert to miles per minute

-- The Proof statement
theorem woman_stop_time :
  (distance_covered / relative_speed_mpm) = 2 :=
by
  sorry

end woman_stop_time_l106_106169


namespace share_of_B_l106_106310

noncomputable def problem_statement (A B C : ℝ) : Prop :=
  A + B + C = 595 ∧ A = (2/3) * B ∧ B = (1/4) * C

theorem share_of_B (A B C : ℝ) (h : problem_statement A B C) : B = 105 :=
by
  -- Proof omitted
  sorry

end share_of_B_l106_106310


namespace calc_pairs_count_l106_106126

theorem calc_pairs_count :
  ∃! (ab : ℤ × ℤ), (ab.1 + ab.2 = ab.1 * ab.2) :=
by
  sorry

end calc_pairs_count_l106_106126


namespace minimum_equilateral_triangles_l106_106495

theorem minimum_equilateral_triangles (side_small : ℝ) (side_large : ℝ)
  (h_small : side_small = 1) (h_large : side_large = 15) :
  225 = (side_large / side_small)^2 :=
by
  -- Proof is skipped.
  sorry

end minimum_equilateral_triangles_l106_106495


namespace who_finished_in_7th_place_l106_106270

theorem who_finished_in_7th_place:
  ∀ (Alex Ben Charlie David Ethan : ℕ),
  (Ethan + 4 = Alex) →
  (David + 1 = Ben) →
  (Charlie = Ben + 3) →
  (Alex = Ben + 2) →
  (Ethan + 2 = David) →
  (Ben = 5) →
  Alex = 7 :=
by
  intros Alex Ben Charlie David Ethan h1 h2 h3 h4 h5 h6
  sorry

end who_finished_in_7th_place_l106_106270


namespace expenses_notation_l106_106860

theorem expenses_notation (income expense : ℤ) (h_income : income = 6) (h_expense : -expense = income) : expense = -4 := 
by
  sorry

end expenses_notation_l106_106860


namespace max_surface_area_of_cut_l106_106787

noncomputable def max_sum_surface_areas (l w h : ℝ) : ℝ :=
  if l = 5 ∧ w = 4 ∧ h = 3 then 144 else 0

theorem max_surface_area_of_cut (l w h : ℝ) (h_l : l = 5) (h_w : w = 4) (h_h : h = 3) : 
  max_sum_surface_areas l w h = 144 :=
by 
  rw [max_sum_surface_areas, if_pos]
  exact ⟨h_l, h_w, h_h⟩

end max_surface_area_of_cut_l106_106787


namespace area_of_triangle_formed_by_medians_l106_106608

variable {a b c m_a m_b m_c Δ Δ': ℝ}

-- Conditions from the problem
axiom rel_sum_of_squares : m_a^2 + m_b^2 + m_c^2 = (3 / 4) * (a^2 + b^2 + c^2)
axiom rel_fourth_powers : m_a^4 + m_b^4 + m_c^4 = (9 / 16) * (a^4 + b^4 + c^4)

-- Statement of the problem to prove
theorem area_of_triangle_formed_by_medians :
  Δ' = (3 / 4) * Δ := sorry

end area_of_triangle_formed_by_medians_l106_106608


namespace average_price_per_bottle_l106_106287

/-
  Given:
  * Number of large bottles: 1300
  * Price per large bottle: 1.89
  * Number of small bottles: 750
  * Price per small bottle: 1.38
  
  Prove:
  The approximate average price per bottle is 1.70
-/
theorem average_price_per_bottle : 
  let num_large_bottles := 1300
  let price_per_large_bottle := 1.89
  let num_small_bottles := 750
  let price_per_small_bottle := 1.38
  let total_cost_large_bottles := num_large_bottles * price_per_large_bottle
  let total_cost_small_bottles := num_small_bottles * price_per_small_bottle
  let total_number_bottles := num_large_bottles + num_small_bottles
  let overall_total_cost := total_cost_large_bottles + total_cost_small_bottles
  let average_price := overall_total_cost / total_number_bottles
  average_price = 1.70 :=
by
  sorry

end average_price_per_bottle_l106_106287


namespace extreme_values_of_f_max_min_values_on_interval_l106_106188

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (Real.exp x)

theorem extreme_values_of_f : 
  (∃ x_max : ℝ, f x_max = 2 / Real.exp 1 ∧ ∀ x : ℝ, f x ≤ 2 / Real.exp 1) :=
sorry

theorem max_min_values_on_interval : 
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, 
    (f 1 = 2 / Real.exp 1 ∧ ∀ x : ℝ, x ∈ Set.Icc (1/2) 2 → f x ≤ 2 / Real.exp 1)
     ∧ (f 2 = 4 / (Real.exp 2) ∧ ∀ x ∈ Set.Icc (1/2 : ℝ) 2, 4 / (Real.exp 2) ≤ f x)) :=
sorry

end extreme_values_of_f_max_min_values_on_interval_l106_106188


namespace digit_B_divisible_by_9_l106_106709

theorem digit_B_divisible_by_9 (B : ℕ) (h1 : B ≤ 9) (h2 : (4 + B + B + 1 + 3) % 9 = 0) : B = 5 := 
by {
  /- Proof omitted -/
  sorry
}

end digit_B_divisible_by_9_l106_106709


namespace sequence_general_formula_l106_106876

theorem sequence_general_formula (a : ℕ → ℕ) 
    (h₀ : a 1 = 3) 
    (h : ∀ n : ℕ, a (n + 1) = 2 * a n + 1) : 
    ∀ n : ℕ, a n = 2^(n+1) - 1 :=
by 
  sorry

end sequence_general_formula_l106_106876


namespace part_a_part_b_part_c_l106_106453

variable (p : ℕ) (k : ℕ)

theorem part_a (hp : Prime p) (h : p = 4 * k + 1) :
  ∃ x : ℤ, (x^2 + 1) % p = 0 :=
by
  sorry

theorem part_b (hp : Prime p) (h : p = 4 * k + 1)
  (x : ℤ) (r1 r2 s1 s2 : ℕ)
  (hr1 : 0 ≤ r1) (hr2 : 0 ≤ r2) (hr1_lt : r1 < Nat.sqrt p) (hr2_lt : r2 < Nat.sqrt p)
  (hs1 : 0 ≤ s1) (hs2 : 0 ≤ s2) (hs1_lt : s1 < Nat.sqrt p) (hs2_lt : s2 < Nat.sqrt p)
  (hneq : (r1, s1) ≠ (r2, s2)) :
  ∃ (r1 r2 s1 s2 : ℕ), (r1 * x + s1) % p = (r2 * x + s2) % p :=
by
  sorry

theorem part_c (hp : Prime p) (h : p = 4 * k + 1)
  (x : ℤ) (r1 r2 s1 s2 : ℕ)
  (hr1 : 0 ≤ r1) (hr2 : 0 ≤ r2) (hr1_lt : r1 < Nat.sqrt p) (hr2_lt : r2 < Nat.sqrt p)
  (hs1 : 0 ≤ s1) (hs2 : 0 ≤ s2) (hs1_lt : s1 < Nat.sqrt p) (hs2_lt : s2 < Nat.sqrt p)
  (hneq : (r1, s1) ≠ (r2, s2)):
  p = (Int.ofNat (r1 - r2))^2 + (Int.ofNat (s1 - s2))^2 :=
by
  sorry

end part_a_part_b_part_c_l106_106453


namespace complement_union_l106_106365

def A := { x : ℤ | ∃ k : ℤ, x = 3 * k + 1 }
def B := { x : ℤ | ∃ k : ℤ, x = 3 * k + 2 }
def U := ℤ

theorem complement_union :
  { x : ℤ | x ∉ (A ∪ B) } = { x : ℤ | ∃ k : ℤ, x = 3 * k } := 
by 
  sorry

end complement_union_l106_106365


namespace rectangle_area_l106_106470

theorem rectangle_area (L B : ℕ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 226) : L * B = 3060 := by
  sorry

end rectangle_area_l106_106470


namespace complementary_angle_measure_l106_106385

theorem complementary_angle_measure (A S C : ℝ) (h1 : A = 45) (h2 : A + S = 180) (h3 : A + C = 90) (h4 : S = 3 * C) : C = 45 :=
by
  sorry

end complementary_angle_measure_l106_106385


namespace degree_of_monomial_l106_106710

def degree (m : String) : Nat :=  -- Placeholder type, replace with appropriate type that represents a monomial
  sorry  -- Logic to compute the degree would go here, if required for full implementation

theorem degree_of_monomial : degree "-(3/5) * a * b^2" = 3 := by
  sorry

end degree_of_monomial_l106_106710


namespace arithmetic_sequence_ratio_l106_106054

open Nat

noncomputable def S (n : ℕ) : ℝ := n^2
noncomputable def T (n : ℕ) : ℝ := n * (2 * n + 3)

theorem arithmetic_sequence_ratio 
  (h : ∀ n : ℕ, (2 * n + 3) * S n = n * T n) : 
  (S 5 - S 4) / (T 6 - T 5) = 9 / 25 := by
  sorry

end arithmetic_sequence_ratio_l106_106054


namespace James_age_l106_106685

-- Defining variables
variables (James John Tim : ℕ)
variables (h1 : James + 12 = John)
variables (h2 : Tim + 5 = 2 * John)
variables (h3 : Tim = 79)

-- Statement to prove James' age
theorem James_age : James = 25 :=
by {
  sorry
}

end James_age_l106_106685


namespace juice_packs_in_box_l106_106418

theorem juice_packs_in_box 
  (W_box L_box H_box W_juice_pack L_juice_pack H_juice_pack : ℕ)
  (hW_box : W_box = 24) (hL_box : L_box = 15) (hH_box : H_box = 28)
  (hW_juice_pack : W_juice_pack = 4) (hL_juice_pack : L_juice_pack = 5) (hH_juice_pack : H_juice_pack = 7) : 
  (W_box * L_box * H_box) / (W_juice_pack * L_juice_pack * H_juice_pack) = 72 :=
by
  sorry

end juice_packs_in_box_l106_106418


namespace distance_between_first_and_last_tree_l106_106510

theorem distance_between_first_and_last_tree
  (n : ℕ) (d_1_5 : ℝ) (h1 : n = 8) (h2 : d_1_5 = 100) :
  let interval_distance := d_1_5 / 4
  let total_intervals := n - 1
  let total_distance := interval_distance * total_intervals
  total_distance = 175 :=
by
  sorry

end distance_between_first_and_last_tree_l106_106510


namespace solve_linear_equation_l106_106556

theorem solve_linear_equation (a b x : ℝ) (h : a - b = 0) (ha : a ≠ 0) : ax + b = 0 ↔ x = -1 :=
by sorry

end solve_linear_equation_l106_106556


namespace percentage_paid_to_X_l106_106980

theorem percentage_paid_to_X (X Y : ℝ) (h1 : X + Y = 880) (h2 : Y = 400) : 
  ((X / Y) * 100) = 120 :=
by
  sorry

end percentage_paid_to_X_l106_106980


namespace remainder_of_N_l106_106698

-- Definition of the sequence constraints
def valid_sequence (a : ℕ → ℕ) : Prop :=
  a 0 = 1 ∧ (∀ i, a i < 512) ∧ (∀ k, 1 ≤ k → k ≤ 9 → ∃ m, 0 ≤ m ∧ m ≤ k - 1 ∧ ((a k - 2 * a m) * (a k - 2 * a m - 1) = 0))

-- Defining N as the number of sequences that are valid.
noncomputable def N : ℕ :=
  Nat.factorial 10 - 2^9

-- The goal is to prove that N mod 1000 is 288
theorem remainder_of_N : N % 1000 = 288 :=
  sorry

end remainder_of_N_l106_106698


namespace vector_dot_product_l106_106434

-- Definitions of the vectors
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-1, 2)

-- Definition of the dot product for 2D vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Main statement to prove
theorem vector_dot_product :
  dot_product (a.1 + b.1, a.2 + b.2) (a.1 - b.1, a.2 - b.2) = 0 :=
by
  sorry

end vector_dot_product_l106_106434


namespace tank_filling_time_l106_106808

theorem tank_filling_time
  (T : ℕ) (Rₐ R_b R_c : ℕ) (C : ℕ)
  (hRₐ : Rₐ = 40) (hR_b : R_b = 30) (hR_c : R_c = 20) (hC : C = 950)
  (h_cycle : T = 1 + 1 + 1) : 
  T * (C / (Rₐ + R_b - R_c)) - 1 = 56 :=
by
  sorry

end tank_filling_time_l106_106808


namespace least_integer_to_multiple_of_3_l106_106861

theorem least_integer_to_multiple_of_3 : ∃ n : ℕ, n > 0 ∧ (527 + n) % 3 = 0 ∧ ∀ m : ℕ, m > 0 → (527 + m) % 3 = 0 → m ≥ n :=
sorry

end least_integer_to_multiple_of_3_l106_106861


namespace Suma_can_complete_in_6_days_l106_106089

-- Define the rates for Renu and their combined rate
def Renu_rate := (1 : ℚ) / 6
def Combined_rate := (1 : ℚ) / 3

-- Define Suma's time to complete the work alone
def Suma_days := 6

-- defining the work rate Suma is required to achieve given the known rates and combined rate
def Suma_rate := Combined_rate - Renu_rate

-- Require to prove 
theorem Suma_can_complete_in_6_days : (1 / Suma_rate) = Suma_days :=
by
  -- Using the definitions provided and some basic algebra to prove the theorem 
  sorry

end Suma_can_complete_in_6_days_l106_106089


namespace find_bicycle_speed_l106_106793

def distanceAB := 40 -- Distance from A to B in km
def speed_walk := 6 -- Speed of the walking tourist in km/h
def distance_ahead := 5 -- Distance by which the second tourist is ahead initially in km
def speed_car := 24 -- Speed of the car in km/h
def meeting_time := 2 -- Time after departure when they meet in hours

theorem find_bicycle_speed (v : ℝ) : 
  (distanceAB = 40 ∧ speed_walk = 6 ∧ distance_ahead = 5 ∧ speed_car = 24 ∧ meeting_time = 2) →
  (v = 9) :=
by 
sorry

end find_bicycle_speed_l106_106793


namespace triangle_perimeter_l106_106826

def triangle_side_lengths : ℕ × ℕ × ℕ := (10, 6, 7)

def perimeter (a b c : ℕ) : ℕ := a + b + c

theorem triangle_perimeter (a b c : ℕ) (h : (a, b, c) = triangle_side_lengths) : 
  perimeter a b c = 23 := by
  -- We formulate the statement and leave the proof for later
  sorry

end triangle_perimeter_l106_106826


namespace fish_filets_total_l106_106956

def fish_caught_by_ben : ℕ := 4
def fish_caught_by_judy : ℕ := 1
def fish_caught_by_billy : ℕ := 3
def fish_caught_by_jim : ℕ := 2
def fish_caught_by_susie : ℕ := 5
def fish_thrown_back : ℕ := 3
def filets_per_fish : ℕ := 2

theorem fish_filets_total : 
  (fish_caught_by_ben + fish_caught_by_judy + fish_caught_by_billy + fish_caught_by_jim + fish_caught_by_susie - fish_thrown_back) * filets_per_fish = 24 := 
by
  sorry

end fish_filets_total_l106_106956


namespace sum_of_real_solutions_l106_106982

theorem sum_of_real_solutions (x : ℝ) (h : (x^2 + 2*x + 3)^( (x^2 + 2*x + 3)^( (x^2 + 2*x + 3) )) = 2012) : 
  ∃ (x1 x2 : ℝ), (x1 + x2 = -2) ∧ (x1^2 + 2*x1 + 3 = x2^2 + 2*x2 + 3 ∧ x2^2 + 2*x2 + 3 = x^2 + 2*x + 3) := 
by
  sorry

end sum_of_real_solutions_l106_106982


namespace cards_left_l106_106008

variable (initialCards : ℕ) (givenCards : ℕ) (remainingCards : ℕ)

def JasonInitialCards := 13
def CardsGivenAway := 9

theorem cards_left : initialCards = JasonInitialCards → givenCards = CardsGivenAway → remainingCards = initialCards - givenCards → remainingCards = 4 :=
by
  intros
  subst_vars
  sorry

end cards_left_l106_106008


namespace no_integer_triplets_satisfying_eq_l106_106604

theorem no_integer_triplets_satisfying_eq (x y z : ℤ) : 3 * x^2 + 7 * y^2 ≠ z^4 := 
by {
  sorry
}

end no_integer_triplets_satisfying_eq_l106_106604


namespace cubic_roots_relations_l106_106363

theorem cubic_roots_relations 
    (a b c d : ℚ) 
    (x1 x2 x3 : ℚ) 
    (h : a ≠ 0)
    (hroots : a * x1^3 + b * x1^2 + c * x1 + d = 0 
      ∧ a * x2^3 + b * x2^2 + c * x2 + d = 0 
      ∧ a * x3^3 + b * x3^2 + c * x3 + d = 0) 
    :
    (x1 + x2 + x3 = -b / a) 
    ∧ (x1 * x2 + x1 * x3 + x2 * x3 = c / a) 
    ∧ (x1 * x2 * x3 = -d / a) := 
sorry

end cubic_roots_relations_l106_106363


namespace area_of_isosceles_triangle_PQR_l106_106972

noncomputable def area_of_triangle (P Q R : ℝ) (PQ PR QR PS QS SR : ℝ) : Prop :=
PQ = 17 ∧ PR = 17 ∧ QR = 16 ∧ PS = 15 ∧ QS = 8 ∧ SR = 8 →
(1 / 2) * QR * PS = 120

theorem area_of_isosceles_triangle_PQR :
  ∀ (P Q R : ℝ), 
  ∀ (PQ PR QR PS QS SR : ℝ), 
  PQ = 17 → PR = 17 → QR = 16 → PS = 15 → QS = 8 → SR = 8 →
  area_of_triangle P Q R PQ PR QR PS QS SR := 
by
  intros P Q R PQ PR QR PS QS SR hPQ hPR hQR hPS hQS hSR
  unfold area_of_triangle
  simp [hPQ, hPR, hQR, hPS, hQS, hSR]
  sorry

end area_of_isosceles_triangle_PQR_l106_106972


namespace extreme_values_l106_106954

noncomputable def f (x : ℝ) := (1/3) * x^3 - 4 * x + 6

theorem extreme_values :
  (∃ x : ℝ, f x = 34/3 ∧ (x = -2 ∨ x = 4)) ∧
  (∃ x : ℝ, f x = 2/3 ∧ x = 2) ∧
  (∀ x ∈ Set.Icc (-3 : ℝ) 4, f x ≤ 34/3 ∧ 2/3 ≤ f x) :=
by
  sorry

end extreme_values_l106_106954


namespace min_slope_at_a_half_l106_106544

theorem min_slope_at_a_half (a : ℝ) (h : 0 < a) :
  (∀ b : ℝ, 0 < b → 4 * b + 1 / b ≥ 4) → (4 * a + 1 / a = 4) → a = 1 / 2 :=
by
  sorry

end min_slope_at_a_half_l106_106544


namespace intersect_A_B_l106_106696

def A : Set ℝ := {x | 1/x < 1}
def B : Set ℝ := {-1, 0, 1, 2}
def intersection_result : Set ℝ := {-1, 2}

theorem intersect_A_B : A ∩ B = intersection_result :=
by
  sorry

end intersect_A_B_l106_106696


namespace passengers_landed_in_virginia_l106_106900

theorem passengers_landed_in_virginia
  (P_start : ℕ) (D_Texas : ℕ) (C_Texas : ℕ) (D_NC : ℕ) (C_NC : ℕ) (C : ℕ)
  (hP_start : P_start = 124)
  (hD_Texas : D_Texas = 58)
  (hC_Texas : C_Texas = 24)
  (hD_NC : D_NC = 47)
  (hC_NC : C_NC = 14)
  (hC : C = 10) :
  P_start - D_Texas + C_Texas - D_NC + C_NC + C = 67 := by
  sorry

end passengers_landed_in_virginia_l106_106900


namespace find_x_l106_106340

theorem find_x (x : ℕ) (a : ℕ) (h₁: a = 450) (h₂: (15^x * 8^3) / 256 = a) : x = 2 :=
by
  sorry

end find_x_l106_106340


namespace new_concentration_of_mixture_l106_106625

theorem new_concentration_of_mixture
  (v1_cap : ℝ) (v1_alcohol_percent : ℝ)
  (v2_cap : ℝ) (v2_alcohol_percent : ℝ)
  (new_vessel_cap : ℝ) (poured_liquid : ℝ)
  (filled_water : ℝ) :
  v1_cap = 2 →
  v1_alcohol_percent = 0.25 →
  v2_cap = 6 →
  v2_alcohol_percent = 0.50 →
  new_vessel_cap = 10 →
  poured_liquid = 8 →
  filled_water = (new_vessel_cap - poured_liquid) →
  ((v1_cap * v1_alcohol_percent + v2_cap * v2_alcohol_percent) / new_vessel_cap) = 0.35 :=
by
  intros v1_h v1_per_h v2_h v2_per_h v_new_h poured_h filled_h
  sorry

end new_concentration_of_mixture_l106_106625


namespace pool_capacity_l106_106038

theorem pool_capacity (hose_rate leak_rate : ℝ) (fill_time : ℝ) (net_rate := hose_rate - leak_rate) (total_water := net_rate * fill_time) :
  hose_rate = 1.6 → 
  leak_rate = 0.1 → 
  fill_time = 40 → 
  total_water = 60 := by
  intros
  sorry

end pool_capacity_l106_106038


namespace sum_of_sides_is_seven_l106_106303

def triangle_sides : ℕ := 3
def quadrilateral_sides : ℕ := 4
def sum_of_sides : ℕ := triangle_sides + quadrilateral_sides

theorem sum_of_sides_is_seven : sum_of_sides = 7 :=
by
  sorry

end sum_of_sides_is_seven_l106_106303


namespace perfect_square_trinomial_l106_106574

-- Define the conditions
theorem perfect_square_trinomial (k : ℤ) : 
  ∃ (a b : ℤ), (a^2 = 1 ∧ b^2 = 16 ∧ (x^2 + k * x * y + 16 * y^2 = (a * x + b * y)^2)) ↔ (k = 8 ∨ k = -8) :=
by
  sorry

end perfect_square_trinomial_l106_106574


namespace quad_function_one_zero_l106_106116

theorem quad_function_one_zero (m : ℝ) :
  (∃ x : ℝ, m * x^2 - 6 * x + 1 = 0 ∧ (∀ x1 x2 : ℝ, m * x1^2 - 6 * x1 + 1 = 0 ∧ m * x2^2 - 6 * x2 + 1 = 0 → x1 = x2)) ↔ (m = 0 ∨ m = 9) :=
by
  sorry

end quad_function_one_zero_l106_106116


namespace find_unsuitable_activity_l106_106669

-- Definitions based on the conditions
def suitable_for_questionnaire (activity : String) : Prop :=
  activity = "D: The radiation produced by various mobile phones during use"

-- Question transformed into a statement to prove in Lean
theorem find_unsuitable_activity :
  suitable_for_questionnaire "D: The radiation produced by various mobile phones during use" :=
by
  sorry

end find_unsuitable_activity_l106_106669


namespace find_c_l106_106352

   variable {a b c : ℝ}
   
   theorem find_c (h1 : 4 * a - 3 * b + c = 0)
     (h2 : (a - 1)^2 + (b - 1)^2 = 4) :
     c = 9 ∨ c = -11 := 
   by
     sorry
   
end find_c_l106_106352


namespace smaller_number_is_five_l106_106102

theorem smaller_number_is_five (x y : ℕ) (h1 : x + y = 18) (h2 : x - y = 8) : y = 5 :=
sorry

end smaller_number_is_five_l106_106102


namespace one_statement_is_true_l106_106548

theorem one_statement_is_true :
  ∃ (S1 S2 S3 S4 S5 : Prop),
    ((S1 ↔ (¬S1 ∧ S2 ∧ S3 ∧ S4 ∧ S5)) ∧
     (S2 ↔ (¬S1 ∧ ¬S2 ∧ S3 ∧ S4 ∧ ¬S5)) ∧
     (S3 ↔ (¬S1 ∧ S2 ∧ ¬S3 ∧ S4 ∧ ¬S5)) ∧
     (S4 ↔ (¬S1 ∧ ¬S2 ∧ ¬S3 ∧ S4 ∧ ¬S5)) ∧
     (S5 ↔ (¬S1 ∧ ¬S2 ∧ ¬S3 ∧ ¬S4 ∧ ¬S5))) ∧
    (S2) ∧ (¬S1) ∧ (¬S3) ∧ (¬S4) ∧ (¬S5) :=
by
  -- Proof goes here
  sorry

end one_statement_is_true_l106_106548


namespace max_cookies_Andy_eats_l106_106251

theorem max_cookies_Andy_eats (cookies_total : ℕ) (h_cookies_total : cookies_total = 30) 
  (exists_pos_a : ∃ a : ℕ, a > 0 ∧ 3 * a = 30 - a ∧ (∃ k : ℕ, 3 * a = k ∧ ∃ m : ℕ, a = m)) 
  : ∃ max_a : ℕ, max_a ≤ 7 ∧ 3 * max_a < cookies_total ∧ 3 * max_a ∣ cookies_total ∧ max_a = 6 :=
by
  sorry

end max_cookies_Andy_eats_l106_106251


namespace tan_105_eq_neg2_sub_sqrt3_l106_106547

-- Define the main theorem to be proven
theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l106_106547


namespace bugs_meet_on_diagonal_l106_106238

noncomputable def isosceles_trapezoid (A B C D : Type) : Prop :=
  ∃ (AB CD : ℝ), (AB > CD) ∧ (AB = AB) ∧ (CD = CD)

noncomputable def same_speeds (speed1 speed2 : ℝ) : Prop :=
  speed1 = speed2

noncomputable def opposite_directions (path1 path2 : ℝ → ℝ) (diagonal_length : ℝ) : Prop :=
  ∀ t, path1 t = diagonal_length - path2 t

noncomputable def bugs_meet (A B C D : Type) (path1 path2 : ℝ → ℝ) (T : ℝ) : Prop :=
  ∃ t ≤ T, path1 t = path2 t

theorem bugs_meet_on_diagonal :
  ∀ (A B C D : Type) (speed : ℝ) (path1 path2 : ℝ → ℝ) (diagonal_length cycle_period : ℝ),
  isosceles_trapezoid A B C D →
  same_speeds speed speed →
  (∀ t, 0 ≤ t → t ≤ cycle_period) →
  opposite_directions path1 path2 diagonal_length →
  bugs_meet A B C D path1 path2 cycle_period :=
by
  intros
  sorry

end bugs_meet_on_diagonal_l106_106238


namespace smallest_six_digit_negative_integer_congruent_to_five_mod_17_l106_106677

theorem smallest_six_digit_negative_integer_congruent_to_five_mod_17 :
  ∃ x : ℤ, x < -100000 ∧ x ≥ -999999 ∧ x % 17 = 5 ∧ x = -100011 :=
by
  sorry

end smallest_six_digit_negative_integer_congruent_to_five_mod_17_l106_106677


namespace maximize_angle_distance_l106_106152

noncomputable def f (x : ℝ) : ℝ :=
  40 * x / (x * x + 500)

theorem maximize_angle_distance :
  ∃ x : ℝ, x = 10 * Real.sqrt 5 ∧ ∀ y : ℝ, y ≠ x → f y < f x :=
sorry

end maximize_angle_distance_l106_106152


namespace ralph_socks_l106_106728

theorem ralph_socks
  (x y w z : ℕ)
  (h1 : x + y + w + z = 15)
  (h2 : x + 2 * y + 3 * w + 4 * z = 36)
  (hx : x ≥ 1) (hy : y ≥ 1) (hw : w ≥ 1) (hz : z ≥ 1) :
  x = 5 :=
sorry

end ralph_socks_l106_106728


namespace trees_planted_in_garden_l106_106848

theorem trees_planted_in_garden (yard_length : ℕ) (tree_distance : ℕ) (h₁ : yard_length = 500) (h₂ : tree_distance = 20) :
  ((yard_length / tree_distance) + 1) = 26 :=
by
  -- The proof goes here
  sorry

end trees_planted_in_garden_l106_106848


namespace bombardment_deaths_l106_106928

variable (initial_population final_population : ℕ)
variable (fear_factor death_percentage : ℝ)

theorem bombardment_deaths (h1 : initial_population = 4200)
                           (h2 : final_population = 3213)
                           (h3 : fear_factor = 0.15)
                           (h4 : ∃ x, death_percentage = x / 100 ∧ 
                                       4200 - (x / 100) * 4200 - fear_factor * (4200 - (x / 100) * 4200) = 3213) :
                           death_percentage = 0.1 :=
by
  sorry

end bombardment_deaths_l106_106928


namespace ramsey_example_l106_106584

theorem ramsey_example (P : Fin 10 → Fin 10 → Prop) :
  (∀ (i j k : Fin 10), i ≠ j → i ≠ k → j ≠ k → ¬(¬P i j ∧ ¬P j k ∧ ¬P k i))
  ∨ (∀ (i j k : Fin 10), i ≠ j → i ≠ k → j ≠ k → ¬(P i j ∧ P j k ∧ P k i)) →
  (∃ (i j k l : Fin 10), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ (P i j ∧ P j k ∧ P k l ∧ P i k ∧ P j l ∧ P i l))
  ∨ (∃ (i j k l : Fin 10), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ (¬P i j ∧ ¬P j k ∧ ¬P k l ∧ ¬P i k ∧ ¬P j l ∧ ¬P i l)) :=
by
  sorry

end ramsey_example_l106_106584


namespace total_candy_l106_106656

/-- Bobby ate 26 pieces of candy initially. -/
def initial_candy : ℕ := 26

/-- Bobby ate 17 more pieces of candy thereafter. -/
def more_candy : ℕ := 17

/-- Prove that the total number of pieces of candy Bobby ate is 43. -/
theorem total_candy : initial_candy + more_candy = 43 := by
  -- The total number of candies should be 26 + 17 which is 43
  sorry

end total_candy_l106_106656


namespace seven_divides_n_l106_106283

theorem seven_divides_n (n : ℕ) (h1 : n ≥ 2) (h2 : n ∣ 3^n + 4^n) : 7 ∣ n :=
sorry

end seven_divides_n_l106_106283


namespace number_of_factors_and_perfect_square_factors_l106_106908

open Nat

-- Define the number 1320 and its prime factorization.
def n : ℕ := 1320
def prime_factors : List (ℕ × ℕ) := [(2, 2), (3, 1), (5, 1), (11, 1)]

-- Define a function to count factors.
def count_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc ⟨_, exp⟩ => acc * (exp + 1)) 1

-- Define a function to count perfect square factors.
def count_perfect_square_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc ⟨prime, exp⟩ => acc * (if exp % 2 == 0 then exp / 2 + 1 else 1)) 1

theorem number_of_factors_and_perfect_square_factors :
  count_factors prime_factors = 24 ∧ count_perfect_square_factors prime_factors = 2 :=
by
  sorry

end number_of_factors_and_perfect_square_factors_l106_106908


namespace equation_has_exactly_one_solution_l106_106056

theorem equation_has_exactly_one_solution (m : ℝ) : 
  (m ∈ { -1 } ∪ Set.Ioo (-1/2 : ℝ) (1/0) ) ↔ ∃ (x : ℝ), 2 * Real.sqrt (1 - m * (x + 2)) = x + 4 :=
sorry

end equation_has_exactly_one_solution_l106_106056


namespace simplify_and_evaluate_expr_l106_106693

theorem simplify_and_evaluate_expr (a b : ℕ) (h₁ : a = 2) (h₂ : b = 2023) : 
  (a + b)^2 + b * (a - b) - 3 * a * b = 4 := by
  sorry

end simplify_and_evaluate_expr_l106_106693


namespace correct_value_l106_106161

theorem correct_value (x : ℝ) (h : x / 3.6 = 2.5) : (x * 3.6) / 2 = 16.2 :=
by {
  -- Proof would go here
  sorry
}

end correct_value_l106_106161


namespace binom_1000_1000_and_999_l106_106678

theorem binom_1000_1000_and_999 :
  (Nat.choose 1000 1000 = 1) ∧ (Nat.choose 1000 999 = 1000) :=
by
  sorry

end binom_1000_1000_and_999_l106_106678


namespace sandy_phone_bill_expense_l106_106949

def sandy_age_now (kim_age : ℕ) : ℕ := 3 * (kim_age + 2) - 2

def sandy_phone_bill (sandy_age : ℕ) : ℕ := 10 * sandy_age

theorem sandy_phone_bill_expense
  (kim_age : ℕ)
  (kim_age_condition : kim_age = 10)
  : sandy_phone_bill (sandy_age_now kim_age) = 340 := by
  sorry

end sandy_phone_bill_expense_l106_106949


namespace problem_statement_l106_106722

noncomputable def f : ℝ → ℝ := sorry

axiom condition1 : ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * (f x ^ 2 - f x * f y + f y ^ 2)

theorem problem_statement : ∀ x : ℝ, f (1996 * x) = 1996 * f x :=
by 
  sorry

end problem_statement_l106_106722


namespace rotate_90deg_l106_106014

def Shape := Type

structure Figure :=
(triangle : Shape)
(circle : Shape)
(square : Shape)
(pentagon : Shape)

def rotated_position (fig : Figure) : Figure :=
{ triangle := fig.circle,
  circle := fig.square,
  square := fig.pentagon,
  pentagon := fig.triangle }

theorem rotate_90deg (fig : Figure) :
  rotated_position fig = { triangle := fig.circle,
                           circle := fig.square,
                           square := fig.pentagon,
                           pentagon := fig.triangle } :=
by {
  sorry
}

end rotate_90deg_l106_106014


namespace lucas_raspberry_candies_l106_106165

-- Define the problem conditions and the question
theorem lucas_raspberry_candies :
  ∃ (r l : ℕ), (r = 3 * l) ∧ ((r - 5) = 4 * (l - 5)) ∧ (r = 45) :=
by
  sorry

end lucas_raspberry_candies_l106_106165


namespace derivative_at_3_l106_106237

def f (x : ℝ) : ℝ := x^2

theorem derivative_at_3 : deriv f 3 = 6 := by
  sorry

end derivative_at_3_l106_106237


namespace find_sin_2a_l106_106098

noncomputable def problem_statement (a : ℝ) : Prop :=
a ∈ Set.Ioo (Real.pi / 2) Real.pi ∧
3 * Real.cos (2 * a) = Real.sqrt 2 * Real.sin ((Real.pi / 4) - a)

theorem find_sin_2a (a : ℝ) (h : problem_statement a) : Real.sin (2 * a) = -8 / 9 :=
sorry

end find_sin_2a_l106_106098


namespace estimate_sqrt_expr_l106_106070

theorem estimate_sqrt_expr :
  2 < (Real.sqrt 3 * (Real.sqrt 10 - Real.sqrt 3)) ∧ 
  (Real.sqrt 3 * (Real.sqrt 10 - Real.sqrt 3)) < 3 := 
sorry

end estimate_sqrt_expr_l106_106070


namespace salary_of_b_l106_106788

theorem salary_of_b (S_A S_B : ℝ)
  (h1 : S_A + S_B = 14000)
  (h2 : 0.20 * S_A = 0.15 * S_B) :
  S_B = 8000 :=
by
  sorry

end salary_of_b_l106_106788


namespace sequence_an_l106_106976

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

-- Conditions
axiom S_formula (n : ℕ) (h₁ : n > 0) : S n = 2 * a n - 2

-- Proof goal
theorem sequence_an (n : ℕ) (h₁ : n > 0) : a n = 2 ^ n := by
  sorry

end sequence_an_l106_106976


namespace max_abc_value_l106_106543

theorem max_abc_value (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_equation : a * b + c = (a + c) * (b + c))
  (h_sum : a + b + c = 2) : abc ≤ 1/27 :=
by sorry

end max_abc_value_l106_106543


namespace smallest_integral_value_k_l106_106344

-- Define the discriminant of the quadratic equation
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := 3 * x * (k * x - 5) - x^2 + 4

-- Define the condition for the quadratic equation having no real roots
def no_real_roots (k : ℝ) : Prop :=
  let a := 3 * k - 1
  let b := -15
  let c := 4
  discriminant a b c < 0

-- The Lean 4 statement to find the smallest integral value of k such that the quadratic has no real roots
theorem smallest_integral_value_k : ∃ (k : ℤ), no_real_roots k ∧ (∀ (m : ℤ), no_real_roots m → k ≤ m) :=
  sorry

end smallest_integral_value_k_l106_106344


namespace arithmetic_geometric_sequence_a4_value_l106_106020

theorem arithmetic_geometric_sequence_a4_value 
  (a : ℕ → ℝ)
  (h1 : a 1 + a 3 = 10)
  (h2 : a 4 + a 6 = 5 / 4) : 
  a 4 = 1 := 
sorry

end arithmetic_geometric_sequence_a4_value_l106_106020


namespace determine_b2050_l106_106585

theorem determine_b2050 (b : ℕ → ℝ)
  (h₀ : ∀ n ≥ 2, b n = b (n - 1) * b (n + 1))
  (h₁ : b 1 = 3 + Real.sqrt 2)
  (h₂ : b 2021 = 7 + 2 * Real.sqrt 2) :
  b 2050 = (7 - 2 * Real.sqrt 2) / 41 := 
sorry

end determine_b2050_l106_106585


namespace exactly_one_even_contradiction_assumption_l106_106689

variable (a b c : ℕ)

def exactly_one_even (a b c : ℕ) : Prop :=
  (a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 1) ∨
  (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1) ∨
  (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 0)

def conclusion (a b c : ℕ) : Prop :=
  (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1) ∨
  (a % 2 = 0 ∧ b % 2 = 0) ∨
  (b % 2 = 0 ∧ c % 2 = 0) ∨
  (c % 2 = 0 ∧ a % 2 = 0)

theorem exactly_one_even_contradiction_assumption :
    exactly_one_even a b c ↔ ¬ conclusion a b c :=
by
  sorry

end exactly_one_even_contradiction_assumption_l106_106689


namespace total_amount_spent_l106_106439
-- Since we need broader imports, we include the whole Mathlib library

-- Definition of the prices of each CD and the quantity purchased
def price_the_life_journey : ℕ := 100
def price_a_day_a_life : ℕ := 50
def price_when_you_rescind : ℕ := 85
def quantity_purchased : ℕ := 3

-- Tactic to calculate the total amount spent
theorem total_amount_spent : (price_the_life_journey * quantity_purchased) + 
                             (price_a_day_a_life * quantity_purchased) + 
                             (price_when_you_rescind * quantity_purchased) 
                             = 705 := by
  sorry

end total_amount_spent_l106_106439


namespace total_cost_of_barbed_wire_l106_106094

noncomputable def cost_of_barbed_wire : ℝ :=
  let area : ℝ := 3136
  let side_length : ℝ := Real.sqrt area
  let perimeter_without_gates : ℝ := 4 * side_length - 2 * 1
  let rate_per_meter : ℝ := 1.10
  perimeter_without_gates * rate_per_meter

theorem total_cost_of_barbed_wire :
  cost_of_barbed_wire = 244.20 :=
sorry

end total_cost_of_barbed_wire_l106_106094


namespace certain_number_is_8000_l106_106551

theorem certain_number_is_8000 (x : ℕ) (h : x / 10 - x / 2000 = 796) : x = 8000 :=
sorry

end certain_number_is_8000_l106_106551


namespace ploughing_solution_l106_106572

/-- Definition representing the problem of A and B ploughing the field together and alone --/
noncomputable def ploughing_problem : Prop :=
  ∃ (A : ℝ), (A > 0) ∧ (1 / A + 1 / 30 = 1 / 10) ∧ A = 15

theorem ploughing_solution : ploughing_problem :=
  by sorry

end ploughing_solution_l106_106572


namespace opposite_of_neg_two_l106_106902

theorem opposite_of_neg_two : -(-2) = 2 :=
by
  sorry

end opposite_of_neg_two_l106_106902


namespace total_votes_l106_106705

theorem total_votes (votes_brenda : ℕ) (total_votes : ℕ) 
  (h1 : votes_brenda = 50) 
  (h2 : votes_brenda = (1/4 : ℚ) * total_votes) : 
  total_votes = 200 :=
by 
  sorry

end total_votes_l106_106705


namespace largest_polygon_area_l106_106825

variable (area : ℕ → ℝ)

def polygon_A_area : ℝ := 6
def polygon_B_area : ℝ := 3 + 4 * 0.5
def polygon_C_area : ℝ := 4 + 5 * 0.5
def polygon_D_area : ℝ := 7
def polygon_E_area : ℝ := 2 + 6 * 0.5

theorem largest_polygon_area : polygon_D_area = max (max (max polygon_A_area polygon_B_area) polygon_C_area) polygon_E_area :=
by
  sorry

end largest_polygon_area_l106_106825


namespace girls_more_than_boys_by_155_l106_106219

def number_of_girls : Real := 542.0
def number_of_boys : Real := 387.0
def difference : Real := number_of_girls - number_of_boys

theorem girls_more_than_boys_by_155 :
  difference = 155.0 := 
by
  sorry

end girls_more_than_boys_by_155_l106_106219


namespace tom_has_18_apples_l106_106617

-- Definitions based on conditions
def phillip_apples : ℕ := 40
def ben_apples : ℕ := phillip_apples + 8
def tom_apples : ℕ := (3 * ben_apples) / 8

-- Theorem stating Tom has 18 apples given the conditions
theorem tom_has_18_apples : tom_apples = 18 :=
sorry

end tom_has_18_apples_l106_106617


namespace triangle_sin_ratio_cos_side_l106_106231

noncomputable section

variables (A B C a b c : ℝ)
variables (h1 : a + b + c = 5)
variables (h2 : Real.cos B = 1 / 4)
variables (h3 : Real.cos A - 2 * Real.cos C = (2 * c - a) / b * Real.cos B)

theorem triangle_sin_ratio_cos_side :
  (Real.sin C / Real.sin A = 2) ∧ (b = 2) :=
  sorry

end triangle_sin_ratio_cos_side_l106_106231


namespace find_ab_value_l106_106185

-- Definitions from the conditions
def ellipse_eq (a b : ℝ) : Prop := b^2 - a^2 = 25
def hyperbola_eq (a b : ℝ) : Prop := a^2 + b^2 = 49

-- Main theorem statement
theorem find_ab_value {a b : ℝ} (h_ellipse : ellipse_eq a b) (h_hyperbola : hyperbola_eq a b) : 
  |a * b| = 2 * Real.sqrt 111 :=
by
  -- Proof goes here
  sorry

end find_ab_value_l106_106185


namespace find_rainy_days_l106_106164

theorem find_rainy_days 
  (n d T H P R : ℤ) 
  (h1 : R + (d - R) = d)
  (h2 : 3 * (d - R) = T)
  (h3 : n * R = H)
  (h4 : T = H + P)
  (hd : 1 ≤ d ∧ d ≤ 31)
  (hR_range : 0 ≤ R ∧ R ≤ d) :
  R = (3 * d - P) / (n + 3) :=
sorry

end find_rainy_days_l106_106164


namespace remainder_of_2356912_div_8_l106_106335

theorem remainder_of_2356912_div_8 : 912 % 8 = 0 := 
by 
  sorry

end remainder_of_2356912_div_8_l106_106335


namespace combined_weight_l106_106053

-- Define the main proof problem
theorem combined_weight (student_weight : ℝ) (sister_weight : ℝ) :
  (student_weight - 5 = 2 * sister_weight) ∧ (student_weight = 79) → (student_weight + sister_weight = 116) :=
by
  sorry

end combined_weight_l106_106053


namespace correct_calculation_l106_106500

-- Definition of the expressions in the problem
def exprA (a : ℝ) : Prop := 2 * a^2 + a^3 = 3 * a^5
def exprB (x y : ℝ) : Prop := ((-3 * x^2 * y)^2 / (x * y) = 9 * x^5 * y^3)
def exprC (b : ℝ) : Prop := (2 * b^2)^3 = 8 * b^6
def exprD (x : ℝ) : Prop := (2 * x * 3 * x^5 = 6 * x^5)

-- The proof problem
theorem correct_calculation (a x y b : ℝ) : exprC b ∧ ¬ exprA a ∧ ¬ exprB x y ∧ ¬ exprD x :=
by {
  sorry
}

end correct_calculation_l106_106500


namespace sum_of_b_is_negative_twelve_l106_106256

-- Conditions: the quadratic equation and its property having exactly one solution
def quadratic_equation (b : ℝ) : Prop :=
  ∀ x : ℝ, 3 * x^2 + b * x + 6 * x + 10 = 0

-- Statement to prove: sum of the values of b is -12, 
-- given the condition that the equation has exactly one solution
theorem sum_of_b_is_negative_twelve :
  ∀ b1 b2 : ℝ, (quadratic_equation b1 ∧ quadratic_equation b2) ∧
  (∀ x : ℝ, 3 * x^2 + (b1 + 6) * x + 10 = 0 ∧ 3 * x^2 + (b2 + 6) * x + 10 = 0) ∧
  (∀ b : ℝ, b = b1 ∨ b = b2) →
  b1 + b2 = -12 :=
by
  sorry

end sum_of_b_is_negative_twelve_l106_106256


namespace cost_of_children_ticket_l106_106647

theorem cost_of_children_ticket (total_cost : ℝ) (cost_adult_ticket : ℝ) (num_total_tickets : ℕ) (num_adult_tickets : ℕ) (cost_children_ticket : ℝ) :
  total_cost = 119 ∧ cost_adult_ticket = 21 ∧ num_total_tickets = 7 ∧ num_adult_tickets = 4 -> cost_children_ticket = 11.67 :=
by
  intros h
  sorry

end cost_of_children_ticket_l106_106647


namespace pyramid_prism_sum_l106_106552

-- Definitions based on conditions
structure Prism :=
  (vertices : ℕ)
  (edges : ℕ)
  (faces : ℕ)

-- The initial cylindrical-prism object
noncomputable def initial_prism : Prism :=
  { vertices := 8,
    edges := 10,
    faces := 5 }

-- Structure for Pyramid Addition
structure PyramidAddition :=
  (new_vertices : ℕ)
  (new_edges : ℕ)
  (new_faces : ℕ)

noncomputable def pyramid_addition : PyramidAddition := 
  { new_vertices := 1,
    new_edges := 4,
    new_faces := 4 }

-- Function to add pyramid to the prism
noncomputable def add_pyramid (prism : Prism) (pyramid : PyramidAddition) : Prism :=
  { vertices := prism.vertices + pyramid.new_vertices,
    edges := prism.edges + pyramid.new_edges,
    faces := prism.faces - 1 + pyramid.new_faces }

-- The resulting prism after adding the pyramid
noncomputable def resulting_prism := add_pyramid initial_prism pyramid_addition

-- Proof problem statement
theorem pyramid_prism_sum : 
  resulting_prism.vertices + resulting_prism.edges + resulting_prism.faces = 31 :=
by sorry

end pyramid_prism_sum_l106_106552


namespace jakes_weight_l106_106937

theorem jakes_weight (J S B : ℝ) 
  (h1 : 0.8 * J = 2 * S)
  (h2 : J + S = 168)
  (h3 : B = 1.25 * (J + S))
  (h4 : J + S + B = 221) : 
  J = 120 :=
by
  sorry

end jakes_weight_l106_106937


namespace city_rentals_cost_per_mile_l106_106333

-- The parameters provided in the problem
def safety_base_rate : ℝ := 21.95
def safety_per_mile_rate : ℝ := 0.19
def city_base_rate : ℝ := 18.95
def miles_driven : ℝ := 150.0

-- The cost expressions based on the conditions
def safety_total_cost (miles: ℝ) : ℝ := safety_base_rate + safety_per_mile_rate * miles
def city_total_cost (miles: ℝ) (city_per_mile_rate: ℝ) : ℝ := city_base_rate + city_per_mile_rate * miles

-- The cost equality condition for 150 miles
def cost_condition : Prop :=
  safety_total_cost miles_driven = city_total_cost miles_driven 0.21

-- Prove that the cost per mile for City Rentals is 0.21 dollars
theorem city_rentals_cost_per_mile : cost_condition :=
by
  -- Start the proof
  sorry

end city_rentals_cost_per_mile_l106_106333


namespace green_marble_prob_l106_106025

-- Problem constants
def total_marbles : ℕ := 84
def prob_white : ℚ := 1 / 4
def prob_red_or_blue : ℚ := 0.4642857142857143

-- Defining the individual variables for the counts
variable (W R B G : ℕ)

-- Conditions
axiom total_marbles_eq : W + R + B + G = total_marbles
axiom prob_white_eq : (W : ℚ) / total_marbles = prob_white
axiom prob_red_or_blue_eq : (R + B : ℚ) / total_marbles = prob_red_or_blue

-- Proving the probability of drawing a green marble
theorem green_marble_prob :
  (G : ℚ) / total_marbles = 2 / 7 :=
by
  sorry  -- Proof is not required and thus omitted

end green_marble_prob_l106_106025


namespace group9_40_41_right_angled_l106_106437

theorem group9_40_41_right_angled :
  ¬ (∃ a b c : ℝ, a = 3 ∧ b = 4 ∧ c = 7 ∧ a^2 + b^2 = c^2) ∧
  ¬ (∃ a b c : ℝ, a = 1/3 ∧ b = 1/4 ∧ c = 1/5 ∧ a^2 + b^2 = c^2) ∧
  ¬ (∃ a b c : ℝ, a = 4 ∧ b = 6 ∧ c = 8 ∧ a^2 + b^2 = c^2) ∧
  (∃ a b c : ℝ, a = 9 ∧ b = 40 ∧ c = 41 ∧ a^2 + b^2 = c^2) :=
by
  sorry

end group9_40_41_right_angled_l106_106437


namespace ending_number_divisible_by_9_l106_106011

theorem ending_number_divisible_by_9 (E : ℕ) 
  (h1 : ∀ n, 10 ≤ n → n ≤ E → n % 9 = 0 → ∃ m ≥ 1, n = 18 + 9 * (m - 1)) 
  (h2 : (E - 18) / 9 + 1 = 111110) : 
  E = 999999 :=
by
  sorry

end ending_number_divisible_by_9_l106_106011


namespace trig_identity_nec_but_not_suff_l106_106683

open Real

theorem trig_identity_nec_but_not_suff (α β : ℝ) (k : ℤ) :
  (α + β = 2 * k * π + π / 6) → (sin α * cos β + cos α * sin β = 1 / 2) := by
  sorry

end trig_identity_nec_but_not_suff_l106_106683


namespace stratified_sampling_workshops_l106_106313

theorem stratified_sampling_workshops (units_A units_B units_C sample_B n : ℕ) 
(hA : units_A = 96) 
(hB : units_B = 84) 
(hC : units_C = 60) 
(hSample_B : sample_B = 7) 
(hn : (sample_B : ℚ) / n = (units_B : ℚ) / (units_A + units_B + units_C)) : 
  n = 70 :=
  by
  sorry

end stratified_sampling_workshops_l106_106313


namespace axis_center_symmetry_sine_shifted_l106_106480
  noncomputable def axis_of_symmetry (k : ℤ) : ℝ := 3 * Real.pi / 4 + k * Real.pi

  noncomputable def center_of_symmetry (k : ℤ) : ℝ × ℝ := (Real.pi / 4 + k * Real.pi, 0)

  theorem axis_center_symmetry_sine_shifted :
    ∀ (k : ℤ),
    ∃ x y : ℝ,
      (x = axis_of_symmetry k) ∧ (y = 0) ∧ (y, 0) = center_of_symmetry k := 
  sorry
  
end axis_center_symmetry_sine_shifted_l106_106480


namespace train_complete_time_l106_106505

noncomputable def train_time_proof : Prop :=
  ∃ (t_x : ℕ) (v_x : ℝ) (v_y : ℝ),
    v_y = 140 / 3 ∧
    t_x = 140 / v_x ∧
    (∃ t : ℝ, 
      t * v_x = 60.00000000000001 ∧
      t * v_y = 140 - 60.00000000000001) ∧
    t_x = 4

theorem train_complete_time : train_time_proof := by
  sorry

end train_complete_time_l106_106505


namespace total_animals_is_63_l106_106796

def zoo_animals (penguins polar_bears total : ℕ) : Prop :=
  (penguins = 21) ∧
  (polar_bears = 2 * penguins) ∧
  (total = penguins + polar_bears)

theorem total_animals_is_63 :
  ∃ (penguins polar_bears total : ℕ), zoo_animals penguins polar_bears total ∧ total = 63 :=
by {
  sorry
}

end total_animals_is_63_l106_106796


namespace lily_pad_half_coverage_l106_106612

-- Define the conditions in Lean
def doubles_daily (size: ℕ → ℕ) : Prop :=
  ∀ n : ℕ, size (n + 1) = 2 * size n

def covers_entire_lake (size: ℕ → ℕ) (total_size: ℕ) : Prop :=
  size 34 = total_size

-- The main statement to prove
theorem lily_pad_half_coverage (size : ℕ → ℕ) (total_size : ℕ) 
  (h1 : doubles_daily size) 
  (h2 : covers_entire_lake size total_size) : 
  size 33 = total_size / 2 :=
sorry

end lily_pad_half_coverage_l106_106612


namespace problem1_problem2_l106_106643

-- First problem
theorem problem1 :
  2 * Real.sin (Real.pi / 3) - 3 * Real.tan (Real.pi / 6) - (-1 / 3) ^ 0 + (-1) ^ 2023 = -2 :=
by
  sorry

-- Second problem
theorem problem2 :
  abs (1 - Real.sqrt 2) - Real.sqrt 12 + (1 / 3) ^ (-1 : ℤ) - 2 * Real.cos (Real.pi / 4) = 2 - 2 * Real.sqrt 3 :=
by
  sorry

end problem1_problem2_l106_106643


namespace slope_of_line_determined_by_solutions_l106_106661

theorem slope_of_line_determined_by_solutions :
  ∀ (x1 x2 y1 y2 : ℝ), 
  (4 / x1 + 6 / y1 = 0) → (4 / x2 + 6 / y2 = 0) →
  (y2 - y1) / (x2 - x1) = -3 / 2 :=
by
  intros x1 x2 y1 y2 h1 h2
  -- Proof steps go here
  sorry

end slope_of_line_determined_by_solutions_l106_106661


namespace slower_train_speed_l106_106754

theorem slower_train_speed (length_train : ℕ) (speed_fast : ℕ) (time_seconds : ℕ) (distance_meters : ℕ): 
  (length_train = 150) → 
  (speed_fast = 46) → 
  (time_seconds = 108) → 
  (distance_meters = 300) → 
  (distance_meters = (speed_fast - speed_slow) * 5 / 18 * time_seconds) → 
  speed_slow = 36 := 
by
    intros h1 h2 h3 h4 h5
    sorry

end slower_train_speed_l106_106754


namespace complex_number_second_quadrant_l106_106615

theorem complex_number_second_quadrant 
  : (2 + 3 * Complex.I) / (1 - Complex.I) ∈ { z : Complex | z.re < 0 ∧ z.im > 0 } := 
by
  sorry

end complex_number_second_quadrant_l106_106615


namespace dogwood_tree_count_l106_106836

def initial_dogwoods : ℕ := 34
def additional_dogwoods : ℕ := 49
def total_dogwoods : ℕ := initial_dogwoods + additional_dogwoods

theorem dogwood_tree_count :
  total_dogwoods = 83 :=
by
  -- omitted proof
  sorry

end dogwood_tree_count_l106_106836


namespace two_digit_factors_of_2_pow_18_minus_1_l106_106140

-- Define the main problem statement: 
-- How many two-digit factors does 2^18 - 1 have?

theorem two_digit_factors_of_2_pow_18_minus_1 : 
  ∃ n : ℕ, n = 5 ∧ ∀ f : ℕ, (f ∣ (2^18 - 1) ∧ 10 ≤ f ∧ f < 100) ↔ (f ∣ (2^18 - 1) ∧ 10 ≤ f ∧ f < 100 ∧ ∃ k : ℕ, (2^18 - 1) = k * f) :=
by sorry

end two_digit_factors_of_2_pow_18_minus_1_l106_106140


namespace max_f_value_l106_106526

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 + Real.sqrt 3 * Real.cos x - 3 / 4

theorem max_f_value : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 1 ∧ ∃ x₀ ∈ Set.Icc 0 (Real.pi / 2), f x₀ = 1 :=
by
  sorry

end max_f_value_l106_106526


namespace y_intercept_of_line_l106_106111

theorem y_intercept_of_line :
  ∃ y : ℝ, (∃ x : ℝ, x = 0 ∧ 2 * x - 3 * y = 6) ∧ y = -2 :=
sorry

end y_intercept_of_line_l106_106111


namespace f_diff_l106_106800

def f (n : ℕ) : ℚ := (1 / 3 : ℚ) * n * (n + 1) * (n + 2)

theorem f_diff (r : ℕ) : f r - f (r - 1) = r * (r + 1) := 
by {
  -- proof goes here
  sorry
}

end f_diff_l106_106800


namespace gcd_min_value_l106_106851

theorem gcd_min_value {a b c : ℕ} (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
  (gcd_ab : Nat.gcd a b = 210) (gcd_ac : Nat.gcd a c = 770) : Nat.gcd b c = 10 :=
sorry

end gcd_min_value_l106_106851


namespace matilda_fathers_chocolate_bars_l106_106428

/-- Matilda had 20 chocolate bars and shared them evenly amongst herself and her 4 sisters.
    When her father got home, he was upset that they did not put aside any chocolates for him.
    They felt bad, so they each gave up half of their chocolate bars for their father.
    Their father then gave 3 chocolate bars to their mother and ate some.
    Matilda's father had 5 chocolate bars left.
    Prove that Matilda's father ate 2 chocolate bars. -/
theorem matilda_fathers_chocolate_bars:
  ∀ (total_chocolates initial_people chocolates_per_person given_to_father chocolates_left chocolates_eaten: ℕ ),
    total_chocolates = 20 →
    initial_people = 5 →
    chocolates_per_person = total_chocolates / initial_people →
    given_to_father = (chocolates_per_person / 2) * initial_people →
    chocolates_left = given_to_father - 3 →
    chocolates_left - 5 = chocolates_eaten →
    chocolates_eaten = 2 :=
by
  intros
  sorry

end matilda_fathers_chocolate_bars_l106_106428


namespace ab_is_4_l106_106925

noncomputable def ab_value (a b : ℝ) : ℝ :=
  8 / (0.5 * (8 / a) * (8 / b))

theorem ab_is_4 (a b : ℝ) (ha : a > 0) (hb : b > 0) (area_condition : ab_value a b = 8) : a * b = 4 :=
  by
  sorry

end ab_is_4_l106_106925


namespace valid_outfit_combinations_l106_106688

theorem valid_outfit_combinations (shirts pants hats shoes : ℕ) (colors : ℕ) 
  (h₁ : shirts = 6) (h₂ : pants = 6) (h₃ : hats = 6) (h₄ : shoes = 6) (h₅ : colors = 6) :
  ∀ (valid_combinations : ℕ),
  (valid_combinations = colors * (colors - 1) * (colors - 2) * (colors - 3)) → valid_combinations = 360 := 
by
  intros valid_combinations h_valid_combinations
  sorry

end valid_outfit_combinations_l106_106688


namespace num_Q_polynomials_l106_106952

noncomputable def P (x : ℝ) : ℝ := (x - 1) * (x - 4) * (x - 5)

#check Exists

theorem num_Q_polynomials :
  ∃ (Q : Polynomial ℝ), 
  (∃ (R : Polynomial ℝ), R.degree = 3 ∧ P (Q.eval x) = P x * R.eval x) ∧
  Q.degree = 2 ∧ (Q.coeff 1 = 6) ∧ (∃ (n : ℕ), n = 22) :=
sorry

end num_Q_polynomials_l106_106952


namespace all_numbers_positive_l106_106792

theorem all_numbers_positive (n : ℕ) (a : Fin (2 * n + 1) → ℝ) 
  (h : ∀ S : Finset (Fin (2 * n + 1)), 
        S.card = n + 1 → 
        S.sum a > (Finset.univ \ S).sum a) : 
  ∀ i, 0 < a i :=
by
  sorry

end all_numbers_positive_l106_106792


namespace sequence_difference_l106_106323

theorem sequence_difference (a : ℕ → ℤ) (h : ∀ n, a (n + 1) = a n + n) : a 2017 - a 2016 = 2016 :=
sorry

end sequence_difference_l106_106323


namespace cricket_run_rate_l106_106472

theorem cricket_run_rate (run_rate_first_10_overs : ℝ) (target : ℝ) (overs_first_phase : ℕ) (overs_remaining : ℕ) :
  run_rate_first_10_overs = 4.6 → target = 282 → overs_first_phase = 10 → overs_remaining = 40 →
  (target - run_rate_first_10_overs * overs_first_phase) / overs_remaining = 5.9 :=
by
  intros
  sorry

end cricket_run_rate_l106_106472


namespace least_number_of_tablets_l106_106555

theorem least_number_of_tablets (tablets_A : ℕ) (tablets_B : ℕ) (hA : tablets_A = 10) (hB : tablets_B = 13) :
  ∃ n, ((tablets_A ≤ 10 → n ≥ tablets_A + 2) ∧ (tablets_B ≤ 13 → n ≥ tablets_B + 2)) ∧ n = 12 :=
by
  sorry

end least_number_of_tablets_l106_106555


namespace statistical_measure_mode_l106_106502

theorem statistical_measure_mode (fav_dishes : List ℕ) :
  (∀ measure, (measure = "most frequently occurring value" → measure = "Mode")) :=
by
  intro measure
  intro h
  sorry

end statistical_measure_mode_l106_106502


namespace num_repeating_decimals_between_1_and_20_l106_106253

def is_repeating_decimal (a b : ℕ) : Prop :=
  ∀ p q : ℕ, ¬ b = 2^p * 5^q

theorem num_repeating_decimals_between_1_and_20 :
  ∃ (cnt : ℕ), cnt = 20 ∧
  ∀ n : ℕ, (1 ≤ n ∧ n ≤ 20) → is_repeating_decimal n 18 := 
by
  sorry

end num_repeating_decimals_between_1_and_20_l106_106253


namespace intersection_points_x_axis_vertex_on_line_inequality_c_l106_106899

section
variable {r : ℝ}
def quadratic_function (x m : ℝ) : ℝ := -0.5 * (x - 2*m)^2 + 3 - m

theorem intersection_points_x_axis (m : ℝ) (h : m = 2) : 
  ∃ x1 x2 : ℝ, quadratic_function x1 m = 0 ∧ quadratic_function x2 m = 0 ∧ x1 ≠ x2 :=
by
  sorry

theorem vertex_on_line (m : ℝ) (h : true) : 
  ∀ m : ℝ, (2*m, 3-m) ∈ {p : ℝ × ℝ | p.2 = -0.5 * p.1 + 3} :=
by
  sorry

theorem inequality_c (a c m : ℝ) (hP : quadratic_function (a+1) m = c) (hQ : quadratic_function ((4*m-5)+a) m = c) : 
  c ≤ 13/8 :=
by
  sorry
end

end intersection_points_x_axis_vertex_on_line_inequality_c_l106_106899


namespace pirate_flag_minimal_pieces_l106_106736

theorem pirate_flag_minimal_pieces (original_stripes : ℕ) (desired_stripes : ℕ) (cuts_needed : ℕ) : 
  original_stripes = 12 →
  desired_stripes = 10 →
  cuts_needed = 1 →
  ∃ pieces : ℕ, pieces = 2 ∧ 
  (∀ (top_stripes bottom_stripes: ℕ), top_stripes + bottom_stripes = original_stripes → top_stripes = desired_stripes → 
   pieces = 1 + (if bottom_stripes = original_stripes - desired_stripes then 1 else 0)) :=
by intros;
   sorry

end pirate_flag_minimal_pieces_l106_106736


namespace female_officers_count_l106_106504

theorem female_officers_count
  (total_on_duty : ℕ)
  (on_duty_females : ℕ)
  (total_female_officers : ℕ)
  (h1 : total_on_duty = 240)
  (h2 : on_duty_females = total_on_duty / 2)
  (h3 : on_duty_females = (40 * total_female_officers) / 100) : 
  total_female_officers = 300 := 
by
  sorry

end female_officers_count_l106_106504


namespace expression_independent_of_a_l106_106141

theorem expression_independent_of_a (a : ℝ) :
  7 + a - (8 * a - (a + 5 - (4 - 6 * a))) = 8 :=
by sorry

end expression_independent_of_a_l106_106141


namespace P_neither_l106_106207

-- Definition of probabilities according to given conditions
def P_A : ℝ := 0.63      -- Probability of answering the first question correctly
def P_B : ℝ := 0.50      -- Probability of answering the second question correctly
def P_A_and_B : ℝ := 0.33  -- Probability of answering both questions correctly

-- Theorem to prove the probability of answering neither of the questions correctly
theorem P_neither : (1 - (P_A + P_B - P_A_and_B)) = 0.20 := by
  sorry

end P_neither_l106_106207


namespace central_angle_of_sector_l106_106739

noncomputable def central_angle (radius perimeter: ℝ) : ℝ :=
  ((perimeter - 2 * radius) / (2 * Real.pi * radius)) * 360

theorem central_angle_of_sector :
  central_angle 28 144 = 180.21 :=
by
  simp [central_angle]
  sorry

end central_angle_of_sector_l106_106739


namespace circle_through_focus_l106_106530

open Real

-- Define the parabola as a set of points
def is_on_parabola (P : ℝ × ℝ) : Prop :=
  (P.2 - 3) ^ 2 = 8 * (P.1 - 2)

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (4, 3)

-- Define the circle with center P and radius the distance from P to the y-axis
def is_tangent_circle (P : ℝ × ℝ) (C : ℝ × ℝ) : Prop :=
  (P.1 ^ 2 + (P.2 - 3) ^ 2 = (C.1) ^ 2 + (C.2) ^ 2 ∧ C = (4, 3))

-- The main theorem
theorem circle_through_focus (P : ℝ × ℝ) 
  (hP_on_parabola : is_on_parabola P) 
  (hP_tangent_circle : is_tangent_circle P (4, 3)) :
  is_tangent_circle P (4, 3) :=
by sorry

end circle_through_focus_l106_106530


namespace correct_calculation_l106_106718

theorem correct_calculation (x : ℝ) (h : 5.46 - x = 3.97) : 5.46 + x = 6.95 := by
  sorry

end correct_calculation_l106_106718


namespace yogurt_cost_l106_106346

-- Define the conditions given in the problem
def total_cost_ice_cream : ℕ := 20 * 6
def spent_difference : ℕ := 118

theorem yogurt_cost (y : ℕ) 
  (h1 : total_cost_ice_cream = 2 * y + spent_difference) : 
  y = 1 :=
  sorry

end yogurt_cost_l106_106346


namespace number_of_sections_l106_106748

theorem number_of_sections (pieces_per_section : ℕ) (cost_per_piece : ℕ) (total_cost : ℕ)
  (h1 : pieces_per_section = 30)
  (h2 : cost_per_piece = 2)
  (h3 : total_cost = 480) :
  total_cost / (pieces_per_section * cost_per_piece) = 8 := by
  sorry

end number_of_sections_l106_106748


namespace find_custom_operator_result_l106_106000

def custom_operator (a b : ℝ) : ℝ := 4 * a + 3 * b

theorem find_custom_operator_result :
  custom_operator 2 5 = 23 :=
by
  sorry

end find_custom_operator_result_l106_106000


namespace supplement_of_complement_is_125_l106_106999

-- Definition of the initial angle
def initial_angle : ℝ := 35

-- Definition of the complement of the angle
def complement_angle (θ : ℝ) : ℝ := 90 - θ

-- Definition of the supplement of an angle
def supplement_angle (θ : ℝ) : ℝ := 180 - θ

-- Main theorem statement
theorem supplement_of_complement_is_125 : 
  supplement_angle (complement_angle initial_angle) = 125 := 
by
  sorry

end supplement_of_complement_is_125_l106_106999


namespace max_marks_are_700_l106_106964

/-- 
A student has to obtain 33% of the total marks to pass.
The student got 175 marks and failed by 56 marks.
Prove that the maximum marks are 700.
-/
theorem max_marks_are_700 (M : ℝ) (h1 : 0.33 * M = 175 + 56) : M = 700 :=
sorry

end max_marks_are_700_l106_106964


namespace max_min_values_l106_106633

theorem max_min_values (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) : 
  1 ≤ x^2 + 2*y^2 + 3*z^2 ∧ x^2 + 2*y^2 + 3*z^2 ≤ 3 := by
  sorry

end max_min_values_l106_106633


namespace radical_product_l106_106509

theorem radical_product :
  (64 ^ (1 / 3) * 16 ^ (1 / 4) * 64 ^ (1 / 6) = 16) :=
by
  sorry

end radical_product_l106_106509


namespace avg_height_and_variance_correct_l106_106471

noncomputable def avg_height_and_variance
  (total_students : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (boys_avg_height : ℕ)
  (boys_variance : ℕ)
  (girls_avg_height : ℕ)
  (girls_variance : ℕ) : (ℕ × ℕ) := 
  let total_students := 300
  let boys := 180
  let girls := 120
  let boys_avg_height := 170
  let boys_variance := 14
  let girls_avg_height := 160
  let girls_variance := 24
  let avg_height := (boys * boys_avg_height + girls * girls_avg_height) / total_students 
  let variance := (boys * (boys_variance + (boys_avg_height - avg_height) ^ 2) 
                    + girls * (girls_variance + (girls_avg_height - avg_height) ^ 2)) / total_students
  (avg_height, variance)

theorem avg_height_and_variance_correct:
   avg_height_and_variance 300 180 120 170 14 160 24 = (166, 42) := 
  by {
    sorry
  }

end avg_height_and_variance_correct_l106_106471


namespace Emilee_earns_25_l106_106628

variable (Terrence Jermaine Emilee : ℕ)
variable (h1 : Terrence = 30)
variable (h2 : Jermaine = Terrence + 5)
variable (h3 : Jermaine + Terrence + Emilee = 90)

theorem Emilee_earns_25 : Emilee = 25 := by
  -- Insert the proof here
  sorry

end Emilee_earns_25_l106_106628


namespace blocks_for_sculpture_l106_106984

noncomputable def volume_block := 8 * 3 * 1
noncomputable def radius_cylinder := 3
noncomputable def height_cylinder := 8
noncomputable def volume_cylinder := Real.pi * radius_cylinder^2 * height_cylinder
noncomputable def blocks_needed := Nat.ceil (volume_cylinder / volume_block)

theorem blocks_for_sculpture : blocks_needed = 10 := by
  sorry

end blocks_for_sculpture_l106_106984


namespace triangle_expression_value_l106_106983

theorem triangle_expression_value :
  ∀ (A B C : ℝ) (a b c : ℝ),
  A = 60 ∧ b = 1 ∧ (1 / 2) * b * c * (Real.sin A) = Real.sqrt 3 →
  (a + 2 * b - 3 * c) / (Real.sin A + 2 * Real.sin B - 3 * Real.sin C) = 2 * (Real.sqrt 39) / 3 :=
by
  intro A B C a b c
  rintro ⟨hA, hb, h_area⟩
  sorry

end triangle_expression_value_l106_106983


namespace track_length_l106_106773

theorem track_length (x : ℝ) 
  (h1 : ∀ {d1 d2 : ℝ}, (d1 + d2 = x / 2) → (d1 = 120) → d2 = x / 2 - 120)
  (h2 : ∀ {d1 d2 : ℝ}, (d1 = x / 2 - 120 + 170) → (d1 = x / 2 + 50))
  (h3 : ∀ {d3 : ℝ}, (d3 = 3 * x / 2 - 170)) :
  x = 418 :=
by
  sorry

end track_length_l106_106773


namespace angle_of_inclination_l106_106910

noncomputable def line_slope (a b : ℝ) : ℝ := 1  -- The slope of the line y = x + 1 is 1
noncomputable def inclination_angle (m : ℝ) : ℝ := Real.arctan m -- angle of inclination is arctan of the slope

theorem angle_of_inclination (θ : ℝ) : 
  inclination_angle (line_slope 1 1) = 45 :=
by
  sorry

end angle_of_inclination_l106_106910


namespace marcy_fewer_tickets_l106_106077

theorem marcy_fewer_tickets (A M : ℕ) (h1 : A = 26) (h2 : M = 5 * A) (h3 : A + M = 150) : M - A = 104 :=
by
  sorry

end marcy_fewer_tickets_l106_106077


namespace no_integer_solutions_l106_106035

theorem no_integer_solutions (x y : ℤ) : x^3 + 4 * x^2 + x ≠ 18 * y^3 + 18 * y^2 + 6 * y + 3 := 
by 
  sorry

end no_integer_solutions_l106_106035


namespace vector_parallel_example_l106_106762

theorem vector_parallel_example 
  (a : ℝ × ℝ) 
  (b : ℝ × ℝ)
  (ha : a = (2, 1)) 
  (hb : b = (4, 2))
  (h_parallel : ∃ k : ℝ, b = (k * a.1, k * a.2)) : 
  3 • a + 2 • b = (14, 7) := 
by
  sorry

end vector_parallel_example_l106_106762


namespace determine_n_l106_106381

theorem determine_n (n : ℕ) (h : 3^n = 27 * 81^3 / 9^4) : n = 7 := by
  sorry

end determine_n_l106_106381


namespace roots_of_equation_l106_106981

theorem roots_of_equation (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∀ x : ℝ, a^2 * (x - b) / (a - b) * (x - c) / (a - c) + b^2 * (x - a) / (b - a) * (x - c) / (b - c) + c^2 * (x - a) / (c - a) * (x - b) / (c - b) = x^2 :=
by
  intros
  sorry

end roots_of_equation_l106_106981


namespace polynomial_necessary_but_not_sufficient_l106_106080

-- Definitions
def polynomial_condition (x : ℝ) : Prop := x^2 - 3 * x + 2 = 0

def specific_value : ℝ := 1

-- Theorem statement
theorem polynomial_necessary_but_not_sufficient :
  (polynomial_condition specific_value ∧ ¬ ∀ x, polynomial_condition x -> x = specific_value) :=
by
  sorry

end polynomial_necessary_but_not_sufficient_l106_106080


namespace open_box_volume_l106_106302

theorem open_box_volume (l w s : ℕ) (h1 : l = 50)
  (h2 : w = 36) (h3 : s = 8) : (l - 2 * s) * (w - 2 * s) * s = 5440 :=
by {
  sorry
}

end open_box_volume_l106_106302


namespace Anya_loss_games_l106_106407

noncomputable def game_indices_Anya_lost (Anya Bella Valya Galya Dasha : ℕ) : Set ℕ :=
  {4, 8, 12, 16}

theorem Anya_loss_games
  (Anya Bella Valya Galya Dasha total_games : ℕ)
  (h1 : Anya = 4)
  (h2 : Bella = 6)
  (h3 : Valya = 7)
  (h4 : Galya = 10)
  (h5 : Dasha = 11)
  (h6 : total_games = (Anya + Bella + Valya + Galya + Dasha) / 2)
  (h7 : total_games = 19) :
  game_indices_Anya_lost Anya Bella Valya Galya Dasha = {4, 8, 12, 16} :=
  by sorry

end Anya_loss_games_l106_106407


namespace tan_sin_div_l106_106958

theorem tan_sin_div (tan_30_sq : Real := (Real.sin 30 / Real.cos 30) ^ 2)
                    (sin_30_sq : Real := (Real.sin 30) ^ 2):
                    tan_30_sq = (1/3) → sin_30_sq = (1/4) → 
                    (tan_30_sq - sin_30_sq) / (tan_30_sq * sin_30_sq) = 1 :=
by
  sorry

end tan_sin_div_l106_106958


namespace numberOfWaysToChooseLeadershipStructure_correct_l106_106614

noncomputable def numberOfWaysToChooseLeadershipStructure : ℕ :=
  12 * 11 * 10 * Nat.choose 9 3 * Nat.choose 6 3

theorem numberOfWaysToChooseLeadershipStructure_correct :
  numberOfWaysToChooseLeadershipStructure = 221760 :=
by
  simp [numberOfWaysToChooseLeadershipStructure]
  -- Add detailed simplification/proof steps here if required
  sorry

end numberOfWaysToChooseLeadershipStructure_correct_l106_106614


namespace part_one_part_two_l106_106929
-- Import the Mathlib library for necessary definitions and theorems.

-- Define the conditions as hypotheses.
variables {a b c : ℝ} (h : a + b + c = 3) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- Part (1): State the inequality involving sums of reciprocals.
theorem part_one : (1 / (a + b) + 1 / (b + c) + 1 / (c + a)) ≥ 3 / 2 := 
by
  sorry

-- Part (2): Define the range for m in terms of the inequality condition.
theorem part_two : ∃m: ℝ, (∀a b c : ℝ, a + b + c = 3 → 0 < a → 0 < b → 0 < c → (-x^2 + m*x + 2 ≤ a^2 + b^2 + c^2)) ↔ (-2 ≤ m) ∧ (m ≤ 2) :=
by 
  sorry

end part_one_part_two_l106_106929


namespace apples_needed_per_month_l106_106691

theorem apples_needed_per_month (chandler_apples_per_week : ℕ) (lucy_apples_per_week : ℕ) (weeks_per_month : ℕ)
  (h1 : chandler_apples_per_week = 23)
  (h2 : lucy_apples_per_week = 19)
  (h3 : weeks_per_month = 4) :
  (chandler_apples_per_week + lucy_apples_per_week) * weeks_per_month = 168 :=
by
  sorry

end apples_needed_per_month_l106_106691


namespace _l106_106795

-- Define the notion of opposite (additive inverse) of a number
def opposite (n : Int) : Int :=
  -n

-- State the theorem that the opposite of -5 is 5
example : opposite (-5) = 5 := by
  -- Skipping the proof with sorry
  sorry

end _l106_106795


namespace knives_more_than_forks_l106_106756

variable (F K S T : ℕ)
variable (x : ℕ)

-- Initial conditions
def initial_conditions : Prop :=
  (F = 6) ∧ 
  (K = F + x) ∧ 
  (S = 2 * K) ∧
  (T = F / 2)

-- Total cutlery added
def total_cutlery_added : Prop :=
  (F + 2) + (K + 2) + (S + 2) + (T + 2) = 62

-- Prove that x = 9
theorem knives_more_than_forks :
  initial_conditions F K S T x →
  total_cutlery_added F K S T →
  x = 9 := 
by
  sorry

end knives_more_than_forks_l106_106756


namespace minimum_a_plus_2b_no_a_b_such_that_l106_106639

noncomputable def minimum_value (a b : ℝ) :=
  a + 2 * b

theorem minimum_a_plus_2b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a * b = a + 2 * b + 3) : 
  minimum_value a b ≥ 6 :=
sorry

theorem no_a_b_such_that (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a * b = a + 2 * b + 3) : 
  a^2 + 4 * b^2 ≠ 17 :=
sorry

end minimum_a_plus_2b_no_a_b_such_that_l106_106639


namespace solution_set_of_x_squared_gt_x_l106_106180

theorem solution_set_of_x_squared_gt_x :
  { x : ℝ | x^2 > x } = { x : ℝ | x < 0 } ∪ { x : ℝ | x > 1 } :=
by
  sorry

end solution_set_of_x_squared_gt_x_l106_106180


namespace shaded_area_is_one_third_l106_106716

noncomputable def fractional_shaded_area : ℕ → ℚ
| 0 => 1 / 4
| n + 1 => (1 / 4) * fractional_shaded_area n

theorem shaded_area_is_one_third : (∑' n, fractional_shaded_area n) = 1 / 3 := 
sorry

end shaded_area_is_one_third_l106_106716


namespace vector_dot_product_l106_106460

variables (a b : ℝ × ℝ)
variables (h1 : a = (1, 2))
variables (h2 : a - (1 / 5) • b = (-2, 1))

theorem vector_dot_product : (a.1 * b.1 + a.2 * b.2) = 25 :=
by
  sorry

end vector_dot_product_l106_106460


namespace triangle_side_ratio_l106_106508

theorem triangle_side_ratio (a b c: ℝ) (A B C: ℝ) (h1: b * Real.cos C + c * Real.cos B = 2 * b) :
  a / b = 2 :=
sorry

end triangle_side_ratio_l106_106508


namespace probability_intersection_l106_106069

variable (p : ℝ)

def P_A : ℝ := 1 - (1 - p)^6
def P_B : ℝ := 1 - (1 - p)^6
def P_AuB : ℝ := 1 - (1 - 2 * p)^6
def P_AiB : ℝ := P_A p + P_B p - P_AuB p

theorem probability_intersection :
  P_AiB p = 1 - 2 * (1 - p)^6 + (1 - 2 * p)^6 := by
  sorry

end probability_intersection_l106_106069


namespace correctOptionOnlyC_l106_106815

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

end correctOptionOnlyC_l106_106815


namespace find_least_N_exists_l106_106092

theorem find_least_N_exists (N : ℕ) :
  (∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℕ), 
    N = (a₁ + 2) * (b₁ + 2) * (c₁ + 2) - 8 ∧ 
    N + 1 = (a₂ + 2) * (b₂ + 2) * (c₂ + 2) - 8) ∧
  N = 55 := 
sorry

end find_least_N_exists_l106_106092


namespace g_symmetry_solutions_l106_106218

noncomputable def g : ℝ → ℝ := sorry

theorem g_symmetry_solutions (g_def: ∀ (x : ℝ), x ≠ 0 → g x + 3 * g (1 / x) = 6 * x^2) :
  ∀ (x : ℝ), g x = g (-x) → x = 1 ∨ x = -1 :=
by
  sorry

end g_symmetry_solutions_l106_106218


namespace prob_no_decrease_white_in_A_is_correct_l106_106777

-- Define the conditions of the problem
def bagA_white : ℕ := 3
def bagA_black : ℕ := 5
def bagB_white : ℕ := 4
def bagB_black : ℕ := 6

-- Define the probabilities involved
def prob_draw_black_from_A : ℚ := 5 / 8
def prob_draw_white_from_A : ℚ := 3 / 8
def prob_put_white_back_into_A_conditioned_on_white_drawn : ℚ := 5 / 11

-- Calculate the combined probability
def prob_no_decrease_white_in_A : ℚ := prob_draw_black_from_A + prob_draw_white_from_A * prob_put_white_back_into_A_conditioned_on_white_drawn

-- Prove the probability is as expected
theorem prob_no_decrease_white_in_A_is_correct : prob_no_decrease_white_in_A = 35 / 44 := by
  sorry

end prob_no_decrease_white_in_A_is_correct_l106_106777


namespace emma_age_proof_l106_106135

theorem emma_age_proof (Inez Zack Jose Emma : ℕ)
  (hJose : Jose = 20)
  (hZack : Zack = Jose + 4)
  (hInez : Inez = Zack - 12)
  (hEmma : Emma = Jose + 5) :
  Emma = 25 :=
by
  sorry

end emma_age_proof_l106_106135


namespace annie_milkshakes_l106_106820

theorem annie_milkshakes
  (A : ℕ) (C_hamburger : ℕ) (C_milkshake : ℕ) (H : ℕ) (L : ℕ)
  (initial_money : A = 120)
  (hamburger_cost : C_hamburger = 4)
  (milkshake_cost : C_milkshake = 3)
  (hamburgers_bought : H = 8)
  (money_left : L = 70) :
  ∃ (M : ℕ), A - H * C_hamburger - M * C_milkshake = L ∧ M = 6 :=
by
  sorry

end annie_milkshakes_l106_106820


namespace max_value_frac_inv_sum_l106_106901

theorem max_value_frac_inv_sum (x y : ℝ) (a b : ℝ) (h1 : 1 < a) (h2 : 1 < b)
  (h3 : a^x = 6) (h4 : b^y = 6) (h5 : a + b = 2 * Real.sqrt 6) :
  ∃ m, m = 1 ∧ (∀ x y a b, (1 < a) → (1 < b) → (a^x = 6) → (b^y = 6) → (a + b = 2 * Real.sqrt 6) → 
  (∃ n, (n = (1/x + 1/y)) → n ≤ m)) :=
by
  sorry

end max_value_frac_inv_sum_l106_106901


namespace determine_treasures_possible_l106_106073

structure Subject :=
  (is_knight : Prop)
  (is_liar : Prop)
  (is_normal : Prop)

def island_has_treasures : Prop := sorry

def can_determine_treasures (A B C : Subject) (at_most_one_normal : Bool) : Prop :=
  if at_most_one_normal then
    ∃ (question : (Subject → Prop)),
      (∀ response1, ∃ (question2 : (Subject → Prop)),
        (∀ response2, island_has_treasures ↔ (response1 ∧ response2)))
  else
    false

theorem determine_treasures_possible (A B C : Subject) (at_most_one_normal : Bool) :
  at_most_one_normal = true → can_determine_treasures A B C at_most_one_normal :=
by
  intro h
  sorry

end determine_treasures_possible_l106_106073


namespace game_winner_l106_106095

theorem game_winner (m n : ℕ) (hm : 1 < m) (hn : 1 < n) : 
  (mn % 2 = 1 → first_player_wins) ∧ (mn % 2 = 0 → second_player_wins) :=
sorry

end game_winner_l106_106095


namespace find_a9_l106_106918

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- conditions
def is_arithmetic_sequence := ∀ n : ℕ, a (n + 1) = a n + d
def given_condition1 := a 5 + a 7 = 16
def given_condition2 := a 3 = 4

-- theorem
theorem find_a9 (h1 : is_arithmetic_sequence a d) (h2 : given_condition1 a) (h3 : given_condition2 a) :
  a 9 = 12 :=
sorry

end find_a9_l106_106918


namespace sum_of_coefficients_l106_106652

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) 
  (h : (1 + 2*x)^7 = a + a₁*(1 - x) + a₂*(1 - x)^2 + a₃*(1 - x)^3 + a₄*(1 - x)^4 + a₅*(1 - x)^5 + a₆*(1 - x)^6 + a₇*(1 - x)^7) :
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 1 :=
by 
  sorry

end sum_of_coefficients_l106_106652


namespace minimize_expression_pos_int_l106_106867

theorem minimize_expression_pos_int (n : ℕ) (hn : 0 < n) : 
  (∀ m : ℕ, 0 < m → (m / 3 + 27 / m : ℝ) ≥ (9 / 3 + 27 / 9)) :=
sorry

end minimize_expression_pos_int_l106_106867


namespace lottery_not_guaranteed_to_win_l106_106935

theorem lottery_not_guaranteed_to_win (total_tickets : ℕ) (winning_rate : ℚ) (num_purchased : ℕ) :
  total_tickets = 100000 ∧ winning_rate = 1 / 1000 ∧ num_purchased = 2000 → 
  ∃ (outcome : ℕ), outcome = 0 := by
  sorry

end lottery_not_guaranteed_to_win_l106_106935


namespace problem_statement_l106_106406

theorem problem_statement (x : ℝ) (h₀ : 0 < x) (h₁ : x + 1 / x ≥ 2) (h₂ : x + 4 / x ^ 2 ≥ 3) (h₃ : x + 27 / x ^ 3 ≥ 4) :
  ∀ a : ℝ, (x + a / x ^ 4 ≥ 5) → a = 4 ^ 4 := 
by 
  sorry

end problem_statement_l106_106406


namespace star_value_l106_106852

variable (a b : ℤ)
noncomputable def star (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 : ℚ) / b

theorem star_value
  (h1 : a + b = 11)
  (h2 : a * b = 24)
  (h3 : a ≠ 0)
  (h4 : b ≠ 0) :
  star a b = 11 / 24 := by
  sorry

end star_value_l106_106852


namespace triangle_is_right_l106_106194

variable {a b c : ℝ}

theorem triangle_is_right
  (h : a^3 + (Real.sqrt 2 / 4) * b^3 + (Real.sqrt 3 / 9) * c^3 - (Real.sqrt 6 / 2) * a * b * c = 0) :
  (a * a + b * b = c * c) :=
sorry

end triangle_is_right_l106_106194


namespace number_of_pairs_101_l106_106336

theorem number_of_pairs_101 :
  (∃ n : ℕ, (∀ a b : ℕ, (a > 0) → (b > 0) → (a + b = 101) → (b > a) → (n = 50))) :=
sorry

end number_of_pairs_101_l106_106336


namespace max_squares_covered_l106_106133

theorem max_squares_covered 
    (board_square_side : ℝ) 
    (card_side : ℝ) 
    (n : ℕ) 
    (h1 : board_square_side = 1) 
    (h2 : card_side = 2) 
    (h3 : ∀ x y : ℝ, (x*x + y*y ≤ card_side*card_side) → card_side*card_side ≤ 4) :
    n ≤ 9 := sorry

end max_squares_covered_l106_106133


namespace tom_age_difference_l106_106560

/-- 
Tom Johnson's age is some years less than twice as old as his sister.
The sum of their ages is 14 years.
Tom's age is 9 years.
Prove that the number of years less Tom's age is than twice his sister's age is 1 year. 
-/ 
theorem tom_age_difference (T S : ℕ) 
  (h₁ : T = 9) 
  (h₂ : T + S = 14) : 
  2 * S - T = 1 := 
by 
  sorry

end tom_age_difference_l106_106560


namespace total_birds_on_fence_l106_106692

-- Definitions for the problem conditions
def initial_birds : ℕ := 12
def new_birds : ℕ := 8

-- Theorem to state that the total number of birds on the fence is 20
theorem total_birds_on_fence : initial_birds + new_birds = 20 :=
by
  -- Skip the proof as required
  sorry

end total_birds_on_fence_l106_106692


namespace reciprocal_of_neg_2023_l106_106670

-- Define the number and its proposed reciprocal
def x : ℤ := -2023
def r : ℚ := -1 / 2023

-- State the theorem that the reciprocal of x is r
theorem reciprocal_of_neg_2023: x * r = 1 := by
  sorry

end reciprocal_of_neg_2023_l106_106670


namespace graph_of_direct_proportion_is_line_l106_106318

-- Define the direct proportion function
def direct_proportion (k : ℝ) (x : ℝ) : ℝ :=
  k * x

-- State the theorem to prove the graph of this function is a straight line
theorem graph_of_direct_proportion_is_line (k : ℝ) :
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x : ℝ, direct_proportion k x = a * x + b ∧ b = 0 := 
by 
  sorry

end graph_of_direct_proportion_is_line_l106_106318


namespace part1_part2_l106_106905

noncomputable def f (x : Real) : Real :=
  2 * (Real.sin (Real.pi / 4 + x))^2 - Real.sqrt 3 * Real.cos (2 * x) - 1

noncomputable def h (x t : Real) : Real :=
  f (x + t)

theorem part1 (t : Real) (ht : 0 < t ∧ t < Real.pi / 2) :
  (h (-Real.pi / 6) t = 0) → t = Real.pi / 3 :=
sorry

theorem part2 (A B C : Real) (hA : 0 < A ∧ A < Real.pi / 2) (hA1 : h A (Real.pi / 3) = 1) :
  1 < ((Real.sqrt 3 - 1) * Real.sin B + Real.sqrt 2 * Real.sin (Real.pi / 2 - B)) ∧
  ((Real.sqrt 3 - 1) * Real.sin B + Real.sqrt 2 * Real.sin (Real.pi / 2 - B)) ≤ 2 :=
sorry

end part1_part2_l106_106905


namespace discount_percentage_l106_106387

theorem discount_percentage (M C S : ℝ) (hC : C = 0.64 * M) (hS : S = C * 1.28125) :
  ((M - S) / M) * 100 = 18.08 := 
by
  sorry

end discount_percentage_l106_106387


namespace platform_length_is_correct_l106_106171

noncomputable def length_of_platform (train1_speed_kmph : ℕ) (train2_speed_kmph : ℕ) (cross_time_s : ℕ) (platform_time_s : ℕ) : ℕ :=
  let train1_speed_mps := train1_speed_kmph * 5 / 18
  let train2_speed_mps := train2_speed_kmph * 5 / 18
  let relative_speed := train1_speed_mps + train2_speed_mps
  let total_distance := relative_speed * cross_time_s
  let train1_length := 2 * total_distance / 3
  let platform_length := train1_speed_mps * platform_time_s
  platform_length

theorem platform_length_is_correct : length_of_platform 48 42 12 45 = 600 :=
by
  sorry

end platform_length_is_correct_l106_106171


namespace marks_in_physics_l106_106300

section
variables (P C M B CS : ℕ)

-- Given conditions
def condition_1 : Prop := P + C + M + B + CS = 375
def condition_2 : Prop := P + M + B = 255
def condition_3 : Prop := P + C + CS = 210

-- Prove that P = 90
theorem marks_in_physics : condition_1 P C M B CS → condition_2 P M B → condition_3 P C CS → P = 90 :=
by sorry
end

end marks_in_physics_l106_106300


namespace joshua_finishes_after_malcolm_l106_106570

-- Definitions based on conditions.
def malcolm_speed : ℕ := 6 -- Malcolm's speed in minutes per mile
def joshua_speed : ℕ := 8 -- Joshua's speed in minutes per mile
def race_distance : ℕ := 10 -- Race distance in miles

-- Theorem: How many minutes after Malcolm crosses the finish line will Joshua cross the finish line?
theorem joshua_finishes_after_malcolm :
  (joshua_speed * race_distance) - (malcolm_speed * race_distance) = 20 :=
by
  -- sorry is a placeholder for the proof
  sorry

end joshua_finishes_after_malcolm_l106_106570


namespace not_sum_six_odd_squares_l106_106450

-- Definition stating that a number is odd.
def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

-- Given that the square of any odd number is 1 modulo 8.
lemma odd_square_mod_eight (n : ℕ) (h : is_odd n) : (n^2) % 8 = 1 :=
sorry

-- Main theorem stating that 1986 cannot be the sum of six squares of odd numbers.
theorem not_sum_six_odd_squares : ¬ ∃ n1 n2 n3 n4 n5 n6 : ℕ, 
    is_odd n1 ∧ is_odd n2 ∧ is_odd n3 ∧ is_odd n4 ∧ is_odd n5 ∧ is_odd n6 ∧
    n1^2 + n2^2 + n3^2 + n4^2 + n5^2 + n6^2 = 1986 :=
sorry

end not_sum_six_odd_squares_l106_106450


namespace amoeba_count_after_one_week_l106_106072

/-- An amoeba is placed in a puddle and splits into three amoebas on the same day. Each subsequent
    day, every amoeba in the puddle splits into three new amoebas. -/
theorem amoeba_count_after_one_week : 
  let initial_amoebas := 1
  let daily_split := 3
  let days := 7
  (initial_amoebas * (daily_split ^ days)) = 2187 :=
by
  sorry

end amoeba_count_after_one_week_l106_106072


namespace correct_solution_l106_106347

theorem correct_solution : 
  ∀ (x y a b : ℚ), (a = 1) → (b = 1 / 2) → 
  (a * x + y = 2) → (2 * x - b * y = 1) → 
  (x = 4 / 5 ∧ y = 6 / 5) := 
by
  intros x y a b ha hb h1 h2
  sorry

end correct_solution_l106_106347


namespace total_cost_paper_plates_and_cups_l106_106176

theorem total_cost_paper_plates_and_cups :
  ∀ (P C : ℝ), (20 * P + 40 * C = 1.20) → (100 * P + 200 * C = 6.00) := by
  intros P C h
  sorry

end total_cost_paper_plates_and_cups_l106_106176


namespace probability_of_one_defective_l106_106137

theorem probability_of_one_defective :
  (2 : ℕ) ≤ 5 → (0 : ℕ) ≤ 2 → (0 : ℕ) ≤ 3 →
  let total_outcomes := Nat.choose 5 2
  let favorable_outcomes := Nat.choose 3 1 * Nat.choose 2 1
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  probability = (3 / 5 : ℚ) :=
by
  intros h1 h2 h3
  let total_outcomes := Nat.choose 5 2
  let favorable_outcomes := Nat.choose 3 1 * Nat.choose 2 1
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  have : total_outcomes = 10 := by sorry
  have : favorable_outcomes = 6 := by sorry
  have : probability = (6 / 10 : ℚ) := by sorry
  have : (6 / 10 : ℚ) = (3 / 5 : ℚ) := by sorry
  exact this

end probability_of_one_defective_l106_106137


namespace cities_below_50000_l106_106501

theorem cities_below_50000 (p1 p2 : ℝ) (h1 : p1 = 20) (h2: p2 = 65) :
  p1 + p2 = 85 := 
  by sorry

end cities_below_50000_l106_106501


namespace task_completion_choice_l106_106883

theorem task_completion_choice (A B : ℕ) (hA : A = 3) (hB : B = 5) : A + B = 8 := by
  sorry

end task_completion_choice_l106_106883


namespace carpet_area_l106_106794

-- Definitions
def Rectangle1 (length1 width1 : ℕ) : Prop :=
  length1 = 12 ∧ width1 = 9

def Rectangle2 (length2 width2 : ℕ) : Prop :=
  length2 = 6 ∧ width2 = 9

def feet_to_yards (feet : ℕ) : ℕ :=
  feet / 3

-- Statement to prove
theorem carpet_area (length1 width1 length2 width2 : ℕ) (h1 : Rectangle1 length1 width1) (h2 : Rectangle2 length2 width2) :
  feet_to_yards (length1 * width1) / 3 + feet_to_yards (length2 * width2) / 3 = 18 :=
by
  sorry

end carpet_area_l106_106794


namespace primes_up_to_floor_implies_all_primes_l106_106319

/-- Define the function f. -/
def f (x p : ℕ) : ℕ := x^2 + x + p

/-- Define the initial prime condition. -/
def primes_up_to_floor_sqrt_p_over_3 (p : ℕ) : Prop :=
  ∀ x, x ≤ Nat.floor (Nat.sqrt (p / 3)) → Nat.Prime (f x p)

/-- Define the property we want to prove. -/
def all_primes_up_to_p_minus_2 (p : ℕ) : Prop :=
  ∀ x, x ≤ p - 2 → Nat.Prime (f x p)

/-- The main theorem statement. -/
theorem primes_up_to_floor_implies_all_primes
  (p : ℕ) (h : primes_up_to_floor_sqrt_p_over_3 p) : all_primes_up_to_p_minus_2 p :=
sorry

end primes_up_to_floor_implies_all_primes_l106_106319


namespace average_speed_return_trip_l106_106417

/--
A train travels from Albany to Syracuse, a distance of 120 miles,
at an average rate of 50 miles per hour. The train then continues
to Rochester, which is 90 miles from Syracuse, before returning
to Albany. On its way to Rochester, the train's average speed is
60 miles per hour. Finally, the train travels back to Albany from
Rochester, with the total travel time of the train, including all
three legs of the journey, being 9 hours and 15 minutes. What was
the average rate of speed of the train on the return trip from
Rochester to Albany?
-/
theorem average_speed_return_trip :
  let dist_Albany_Syracuse := 120 -- miles
  let speed_Albany_Syracuse := 50 -- miles per hour
  let dist_Syracuse_Rochester := 90 -- miles
  let speed_Syracuse_Rochester := 60 -- miles per hour
  let total_travel_time := 9.25 -- hours (9 hours 15 minutes)
  let time_Albany_Syracuse := dist_Albany_Syracuse / speed_Albany_Syracuse
  let time_Syracuse_Rochester := dist_Syracuse_Rochester / speed_Syracuse_Rochester
  let total_time_so_far := time_Albany_Syracuse + time_Syracuse_Rochester
  let time_return_trip := total_travel_time - total_time_so_far
  let dist_return_trip := dist_Albany_Syracuse + dist_Syracuse_Rochester
  let average_speed_return := dist_return_trip / time_return_trip
  average_speed_return = 39.25 :=
by
  -- sorry placeholder for the actual proof
  sorry

end average_speed_return_trip_l106_106417


namespace min_shots_to_hit_terrorist_l106_106769

theorem min_shots_to_hit_terrorist : ∀ terrorist_position : ℕ, (1 ≤ terrorist_position ∧ terrorist_position ≤ 10) →
  ∃ shots : ℕ, shots ≥ 6 ∧ (∀ move : ℕ, (shots - move) ≥ 1 → (terrorist_position + move ≤ 10 → terrorist_position % 2 = move % 2)) :=
by
  sorry

end min_shots_to_hit_terrorist_l106_106769


namespace math_problem_l106_106244

noncomputable def problem_statement : Prop :=
  ∃ b c : ℝ, 
  (∀ x : ℝ, (x^2 - b * x + c < 0) ↔ (-3 < x ∧ x < 2)) ∧ 
  (b + c = -7)

theorem math_problem : problem_statement := 
by
  sorry

end math_problem_l106_106244


namespace increase_in_average_weight_l106_106308

variable (A : ℝ)

theorem increase_in_average_weight (h1 : ∀ (A : ℝ), 4 * A - 65 + 71 = 4 * (A + 1.5)) :
  (71 - 65) / 4 = 1.5 :=
by
  sorry

end increase_in_average_weight_l106_106308


namespace puppies_brought_in_correct_l106_106760

-- Define the initial number of puppies in the shelter
def initial_puppies: Nat := 2

-- Define the number of puppies adopted per day
def puppies_adopted_per_day: Nat := 4

-- Define the number of days over which the puppies are adopted
def adoption_days: Nat := 9

-- Define the total number of puppies adopted after the given days
def total_puppies_adopted: Nat := puppies_adopted_per_day * adoption_days

-- Define the number of puppies brought in
def puppies_brought_in: Nat := total_puppies_adopted - initial_puppies

-- Prove that the number of puppies brought in is 34
theorem puppies_brought_in_correct: puppies_brought_in = 34 := by
  -- proof omitted, filled with sorry to skip the proof
  sorry

end puppies_brought_in_correct_l106_106760


namespace min_value_of_sum_l106_106063

theorem min_value_of_sum (a b : ℝ) (h1 : a > 1) (h2 : b > 0) (h3 : a + b = 2) : 
  ∃ x : ℝ, x = (1 / (a - 1) + 1 / b) ∧ x = 4 :=
by
  sorry

end min_value_of_sum_l106_106063


namespace diagonal_crosses_768_unit_cubes_l106_106874

-- Defining the dimensions of the rectangular prism
def a : ℕ := 150
def b : ℕ := 324
def c : ℕ := 375

-- Computing the gcd values
def gcd_ab : ℕ := Nat.gcd a b
def gcd_ac : ℕ := Nat.gcd a c
def gcd_bc : ℕ := Nat.gcd b c
def gcd_abc : ℕ := Nat.gcd (Nat.gcd a b) c

-- Using the formula to compute the number of unit cubes the diagonal intersects
def num_unit_cubes : ℕ := a + b + c - gcd_ab - gcd_ac - gcd_bc + gcd_abc

-- Stating the theorem to prove
theorem diagonal_crosses_768_unit_cubes : num_unit_cubes = 768 := by
  sorry

end diagonal_crosses_768_unit_cubes_l106_106874


namespace range_m_l106_106917

variable {x m : ℝ}

theorem range_m (h1 : m / (1 - x) - 2 / (x - 1) = 1) (h2 : x ≥ 0) (h3 : x ≠ 1) : m ≤ -1 ∧ m ≠ -2 := 
sorry

end range_m_l106_106917


namespace cost_price_per_meter_l106_106041

namespace ClothCost

theorem cost_price_per_meter (selling_price_total : ℝ) (meters_sold : ℕ) (loss_per_meter : ℝ) : 
  selling_price_total = 18000 → 
  meters_sold = 300 → 
  loss_per_meter = 5 →
  (selling_price_total / meters_sold) + loss_per_meter = 65 := 
by
  intros hsp hms hloss
  sorry

end ClothCost

end cost_price_per_meter_l106_106041


namespace tan_theta_value_l106_106463

theorem tan_theta_value (θ k : ℝ) 
  (h1 : Real.sin θ = (k + 1) / (k - 3)) 
  (h2 : Real.cos θ = (k - 1) / (k - 3)) 
  (h3 : (Real.sin θ ≠ 0) ∧ (Real.cos θ ≠ 0)) : 
  Real.tan θ = 3 / 4 := 
sorry

end tan_theta_value_l106_106463


namespace jogging_problem_l106_106037

theorem jogging_problem (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : ¬ ∃ p : ℕ, Prime p ∧ p^2 ∣ z) : 
  (x - y * Real.sqrt z) = 60 - 30 * Real.sqrt 2 → x + y + z = 92 :=
by
  intro h5
  have h6 : (60 - (60 - 30 * Real.sqrt 2))^2 = 1800 :=
    by sorry
  sorry

end jogging_problem_l106_106037


namespace solve_quadratic_eq_l106_106462

theorem solve_quadratic_eq (x : ℝ) :
  (3 * (2 * x + 1) = (2 * x + 1)^2) →
  (x = -1/2 ∨ x = 1) :=
by
  sorry

end solve_quadratic_eq_l106_106462


namespace part1_part2_l106_106566

def A (x : ℝ) : Prop := -2 < x ∧ x < 10
def B (x a : ℝ) : Prop := (x ≥ 1 + a ∨ x ≤ 1 - a) ∧ a > 0
def p (x : ℝ) : Prop := A x
def q (x a : ℝ) : Prop := B x a

theorem part1 (a : ℝ) (hA : ∀ x, A x → ¬ B x a) : a ≥ 9 :=
sorry

theorem part2 (a : ℝ) (hSuff : ∀ x, (x ≥ 10 ∨ x ≤ -2) → B x a) (hNotNec : ∃ x, ¬ (x ≥ 10 ∨ x ≤ -2) ∧ B x a) : 0 < a ∧ a ≤ 3 :=
sorry

end part1_part2_l106_106566


namespace profit_percent_l106_106812

-- Definitions based on the conditions in the problem
def marked_price_per_pen := ℝ
def total_pens := 52
def cost_equivalent_pens := 46
def discount_percentage := 1 / 100

-- Values calculated from conditions
def cost_price (P : ℝ) := cost_equivalent_pens * P
def selling_price_per_pen (P : ℝ) := P * (1 - discount_percentage)
def total_selling_price (P : ℝ) := total_pens * selling_price_per_pen P

-- The proof statement
theorem profit_percent (P : ℝ) (hP : P > 0) :
  ((total_selling_price P - cost_price P) / (cost_price P)) * 100 = 11.91 := by
    sorry

end profit_percent_l106_106812


namespace neutralization_reaction_l106_106855

/-- When combining 2 moles of CH3COOH and 2 moles of NaOH, 2 moles of H2O are formed
    given the balanced chemical reaction CH3COOH + NaOH → CH3COONa + H2O 
    with a molar ratio of 1:1:1 (CH3COOH:NaOH:H2O). -/
theorem neutralization_reaction
  (mCH3COOH : ℕ) (mNaOH : ℕ) :
  (mCH3COOH = 2) → (mNaOH = 2) → (mCH3COOH = mNaOH) →
  ∃ mH2O : ℕ, mH2O = 2 :=
by intros; existsi 2; sorry

end neutralization_reaction_l106_106855


namespace determine_subtracted_number_l106_106624

theorem determine_subtracted_number (x y : ℤ) (h1 : x = 40) (h2 : 7 * x - y = 130) : y = 150 :=
by sorry

end determine_subtracted_number_l106_106624


namespace no_integer_roots_l106_106215

theorem no_integer_roots (a b : ℤ) : ¬∃ u : ℤ, u^2 + 3 * a * u + 3 * (2 - b^2) = 0 :=
by
  sorry

end no_integer_roots_l106_106215


namespace Skylar_chickens_less_than_triple_Colten_l106_106229

def chickens_count (S Q C : ℕ) : Prop := 
  Q + S + C = 383 ∧ 
  Q = 2 * S + 25 ∧ 
  C = 37

theorem Skylar_chickens_less_than_triple_Colten (S Q C : ℕ) 
  (h : chickens_count S Q C) : (3 * C - S = 4) := 
sorry

end Skylar_chickens_less_than_triple_Colten_l106_106229


namespace total_wheels_at_station_l106_106432

/--
There are 4 trains at a train station.
Each train has 4 carriages.
Each carriage has 3 rows of wheels.
Each row of wheels has 5 wheels.
The total number of wheels at the train station is 240.
-/
theorem total_wheels_at_station : 
    let number_of_trains := 4
    let carriages_per_train := 4
    let rows_per_carriage := 3
    let wheels_per_row := 5
    number_of_trains * carriages_per_train * rows_per_carriage * wheels_per_row = 240 := 
by
    sorry

end total_wheels_at_station_l106_106432


namespace student_community_arrangements_l106_106458

theorem student_community_arrangements :
  (3 ^ 4) = 81 :=
by
  sorry

end student_community_arrangements_l106_106458


namespace parabola_directrix_l106_106940

theorem parabola_directrix (x y : ℝ) (h_eqn : y = -3 * x^2 + 6 * x - 5) :
  y = -23 / 12 :=
sorry

end parabola_directrix_l106_106940


namespace length_of_AD_in_parallelogram_l106_106173

theorem length_of_AD_in_parallelogram
  (x : ℝ)
  (AB BC CD : ℝ)
  (AB_eq : AB = x + 3)
  (BC_eq : BC = x - 4)
  (CD_eq : CD = 16)
  (parallelogram_ABCD : AB = CD ∧ AD = BC) :
  AD = 9 := by
sorry

end length_of_AD_in_parallelogram_l106_106173


namespace simplify_fraction_product_l106_106492

theorem simplify_fraction_product :
  4 * (18 / 5) * (35 / -63) * (8 / 14) = - (32 / 7) :=
by sorry

end simplify_fraction_product_l106_106492


namespace total_daisies_l106_106790

-- Define the initial conditions
def white_daisies : Nat := 6
def pink_daisies : Nat := 9 * white_daisies
def red_daisies : Nat := 4 * pink_daisies - 3

-- The main theorem stating that the total number of daisies is 273
theorem total_daisies : white_daisies + pink_daisies + red_daisies = 273 := by
  -- The proof is left as an exercise
  sorry

end total_daisies_l106_106790


namespace find_AD_find_a_rhombus_l106_106200

variable (a : ℝ) (AB AD : ℝ)

-- Problem 1: Given AB = 2, find AD
theorem find_AD (h1 : AB = 2)
    (h_quad : ∀ x, x^2 - (a-4)*x + (a-1) = 0 → x = AB ∨ x = AD) : AD = 5 := sorry

-- Problem 2: Find the value of a such that ABCD is a rhombus
theorem find_a_rhombus (h_quad : ∀ x, x^2 - (a-4)*x + (a-1) = 0 → x = 2 → AB = AD → x = a ∨ AB = AD → x = 10) :
    a = 10 := sorry

end find_AD_find_a_rhombus_l106_106200


namespace remainder_div_13_l106_106029

theorem remainder_div_13 {k : ℤ} (N : ℤ) (h : N = 39 * k + 20) : N % 13 = 7 :=
by
  sorry

end remainder_div_13_l106_106029


namespace inequality_transformation_l106_106802

theorem inequality_transformation (x y a : ℝ) (hxy : x < y) (ha : a < 1) : x + a < y + 1 := by
  sorry

end inequality_transformation_l106_106802


namespace probability_of_selecting_one_second_class_product_l106_106481

def total_products : ℕ := 100
def first_class_products : ℕ := 90
def second_class_products : ℕ := 10
def selected_products : ℕ := 3
def exactly_one_second_class_probability : ℚ :=
  (Nat.choose first_class_products 2 * Nat.choose second_class_products 1) / Nat.choose total_products selected_products

theorem probability_of_selecting_one_second_class_product :
  exactly_one_second_class_probability = 0.25 := 
  sorry

end probability_of_selecting_one_second_class_product_l106_106481


namespace arithmetic_seq_sum_l106_106517

theorem arithmetic_seq_sum (a : ℕ → ℤ) (d : ℤ) 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 3 + a 7 = 38) : 
  a 2 + a 4 + a 6 + a 8 = 76 :=
by 
  sorry

end arithmetic_seq_sum_l106_106517


namespace max_b_no_lattice_points_line_l106_106973

theorem max_b_no_lattice_points_line (b : ℝ) (h : ∀ (m : ℝ), 0 < m ∧ m < b → ∀ (x : ℤ), 0 < (x : ℝ) ∧ (x : ℝ) ≤ 150 → ¬∃ (y : ℤ), y = m * x + 5) :
  b ≤ 1 / 151 :=
by sorry

end max_b_no_lattice_points_line_l106_106973


namespace inequality_holds_for_all_x_l106_106186

theorem inequality_holds_for_all_x (m : ℝ) : (∀ x : ℝ, (m^2 + 4*m - 5)*x^2 - 4*(m - 1)*x + 3 > 0) ↔ (1 ≤ m ∧ m < 19) :=
by {
  sorry
}

end inequality_holds_for_all_x_l106_106186


namespace find_b_minus_a_l106_106878

/-- Proof to find the value of b - a given the inequality conditions on x.
    The conditions are:
    1. x - a < 1
    2. x + b > 2
    3. 0 < x < 4
    We need to show that b - a = -1.
-/
theorem find_b_minus_a (a b x : ℝ) 
  (h1 : x - a < 1) 
  (h2 : x + b > 2) 
  (h3 : 0 < x) 
  (h4 : x < 4) 
  : b - a = -1 := 
sorry

end find_b_minus_a_l106_106878


namespace angle_in_second_quadrant_l106_106518

theorem angle_in_second_quadrant (α : ℝ) (h1 : Real.cos α < 0) (h2 : Real.sin α > 0) : 
    α ∈ Set.Ioo (π / 2) π := 
    sorry

end angle_in_second_quadrant_l106_106518


namespace paisa_per_rupee_z_gets_l106_106534

theorem paisa_per_rupee_z_gets
  (y_share : ℝ)
  (y_per_x_paisa : ℝ)
  (total_amount : ℝ)
  (x_share : ℝ)
  (z_share : ℝ)
  (paisa_per_rupee : ℝ)
  (h1 : y_share = 36)
  (h2 : y_per_x_paisa = 0.45)
  (h3 : total_amount = 140)
  (h4 : x_share = y_share / y_per_x_paisa)
  (h5 : z_share = total_amount - (x_share + y_share))
  (h6 : paisa_per_rupee = (z_share / x_share) * 100) :
  paisa_per_rupee = 30 :=
by
  sorry

end paisa_per_rupee_z_gets_l106_106534


namespace fishing_ratio_l106_106573

variables (B C : ℝ)
variable (brian_per_trip : ℝ)
variable (chris_per_trip : ℝ)

-- Given conditions
def conditions : Prop :=
  C = 10 ∧
  brian_per_trip = 400 ∧
  chris_per_trip = 400 * (5 / 3) ∧
  B * brian_per_trip + 10 * chris_per_trip = 13600

-- The ratio of the number of times Brian goes fishing to the number of times Chris goes fishing
def ratio_correct : Prop :=
  B / C = 26 / 15

theorem fishing_ratio (h : conditions B C brian_per_trip chris_per_trip) : ratio_correct B C :=
by
  sorry

end fishing_ratio_l106_106573


namespace probability_one_white_ball_conditional_probability_P_B_given_A_l106_106506

-- Definitions for Problem 1
def red_balls : Nat := 4
def white_balls : Nat := 2
def total_balls : Nat := red_balls + white_balls

def C (n k : ℕ) : ℕ := n.choose k

theorem probability_one_white_ball :
  (C 2 1 * C 4 2 : ℚ) / C 6 3 = 3 / 5 :=
by sorry

-- Definitions for Problem 2
def total_after_first_draw : Nat := total_balls - 1
def remaining_red_balls : Nat := red_balls - 1

theorem conditional_probability_P_B_given_A :
  (remaining_red_balls : ℚ) / total_after_first_draw = 3 / 5 :=
by sorry

end probability_one_white_ball_conditional_probability_P_B_given_A_l106_106506


namespace men_entered_count_l106_106927

variable (M W x : ℕ)

noncomputable def initial_ratio : Prop := M = 4 * W / 5
noncomputable def men_entered : Prop := M + x = 14
noncomputable def women_double : Prop := 2 * (W - 3) = 14

theorem men_entered_count (M W x : ℕ) (h1 : initial_ratio M W) (h2 : men_entered M x) (h3 : women_double W) : x = 6 := by
  sorry

end men_entered_count_l106_106927


namespace nat_triple_solution_l106_106712

theorem nat_triple_solution (x y n : ℕ) :
  (x! + y!) / n! = 3^n ↔ (x = 1 ∧ y = 2 ∧ n = 1) ∨ (x = 2 ∧ y = 1 ∧ n = 1) := 
by
  sorry

end nat_triple_solution_l106_106712


namespace fewer_miles_per_gallon_city_l106_106224

-- Define the given conditions.
def miles_per_tankful_highway : ℕ := 420
def miles_per_tankful_city : ℕ := 336
def miles_per_gallon_city : ℕ := 24

-- Define the question as a theorem that proves how many fewer miles per gallon in the city compared to the highway.
theorem fewer_miles_per_gallon_city (G : ℕ) (hG : G = miles_per_tankful_city / miles_per_gallon_city) :
  miles_per_tankful_highway / G - miles_per_gallon_city = 6 :=
by
  -- The proof will be provided here.
  sorry

end fewer_miles_per_gallon_city_l106_106224


namespace max_weak_quartets_120_l106_106201

noncomputable def max_weak_quartets (n : ℕ) : ℕ :=
  -- Placeholder definition to represent the maximum weak quartets
  sorry  -- To be replaced with the actual mathematical definition

theorem max_weak_quartets_120 : max_weak_quartets 120 = 4769280 := by
  sorry

end max_weak_quartets_120_l106_106201


namespace correct_choice_C_l106_106079

theorem correct_choice_C (x : ℝ) : x^2 ≥ x - 1 := 
sorry

end correct_choice_C_l106_106079


namespace phoenix_flight_l106_106359

theorem phoenix_flight : ∃ n : ℕ, 3 ^ n > 6560 ∧ ∀ m < n, 3 ^ m ≤ 6560 :=
by sorry

end phoenix_flight_l106_106359


namespace find_area_MOI_l106_106309

noncomputable def incenter_coords (a b c : ℝ) (A B C : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((a * A.1 + b * B.1 + c * C.1) / (a + b + c), (a * A.2 + b * B.2 + c * C.2) / (a + b + c))

noncomputable def shoelace_area (P Q R : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

theorem find_area_MOI :
  let A := (0, 0)
  let B := (8, 0)
  let C := (0, 17)
  let O := (4, 8.5)
  let I := incenter_coords 8 15 17 A B C
  let M := (6.25, 6.25)
  shoelace_area M O I = 25.78125 :=
by
  sorry

end find_area_MOI_l106_106309


namespace geometric_sequence_ratio_l106_106891

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (n : ℕ)

-- Definitions based on given conditions
def is_geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Main statement
theorem geometric_sequence_ratio :
  is_geometric_seq a q →
  q = -1/3 →
  (a 1 + a 3 + a 5 + a 7) / (a 2 + a 4 + a 6 + a 8) = -3 :=
by
  intros
  sorry

end geometric_sequence_ratio_l106_106891


namespace simplify_eval_l106_106474

variable (x : ℝ)
def expr := 8 * x^2 - (x - 2) * (3 * x + 1) - 2 * (x + 1) * (x - 1)

theorem simplify_eval (h : x = -2) : expr x = 6 := by
  sorry

end simplify_eval_l106_106474


namespace intersection_points_are_integers_l106_106022

theorem intersection_points_are_integers :
  ∀ (a b : Fin 2021 → ℕ), Function.Injective a → Function.Injective b →
  ∀ i j, i ≠ j → 
  ∃ x : ℤ, (∃ y : ℚ, y = (a i : ℚ) / (x + (b i : ℚ))) ∧ 
           (∃ y : ℚ, y = (a j : ℚ) / (x + (b j : ℚ))) := 
sorry

end intersection_points_are_integers_l106_106022


namespace price_per_unit_l106_106370

theorem price_per_unit (x y : ℝ) 
    (h1 : 2 * x + 3 * y = 690) 
    (h2 : x + 4 * y = 720) : 
    x = 120 ∧ y = 150 := 
by 
    sorry

end price_per_unit_l106_106370


namespace rectangle_ratio_width_length_l106_106868

variable (w : ℝ)

theorem rectangle_ratio_width_length (h1 : w + 8 + w + 8 = 24) : 
  w / 8 = 1 / 2 :=
by
  sorry

end rectangle_ratio_width_length_l106_106868


namespace moving_circle_fixed_point_coordinates_l106_106745

theorem moving_circle_fixed_point_coordinates (m x y : Real) :
    (∀ m : ℝ, x^2 + y^2 - 2 * m * x - 4 * m * y + 6 * m - 2 = 0) →
    (x = 1 ∧ y = 1 ∨ x = 1 / 5 ∧ y = 7 / 5) :=
  by
    sorry

end moving_circle_fixed_point_coordinates_l106_106745


namespace lucy_current_fish_l106_106673

-- Definitions based on conditions in the problem
def total_fish : ℕ := 280
def fish_needed_to_buy : ℕ := 68

-- Proving the number of fish Lucy currently has
theorem lucy_current_fish : total_fish - fish_needed_to_buy = 212 :=
by
  sorry

end lucy_current_fish_l106_106673


namespace sum_of_roots_l106_106498

theorem sum_of_roots {a b c d : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
    (h1 : c + d = -a) (h2 : c * d = b) (h3 : a + b = -c) (h4 : a * b = d) : 
    a + b + c + d = -2 := 
by
  sorry

end sum_of_roots_l106_106498


namespace units_digit_of_516n_divisible_by_12_l106_106279

theorem units_digit_of_516n_divisible_by_12 (n : ℕ) (h₀ : n ≤ 9) :
  (516 * 10 + n) % 12 = 0 ↔ n = 0 ∨ n = 4 :=
by 
  sorry

end units_digit_of_516n_divisible_by_12_l106_106279


namespace intermediate_circle_radius_l106_106889

theorem intermediate_circle_radius (r1 r3: ℝ) (h1: r1 = 5) (h2: r3 = 13) 
  (h3: π * r1 ^ 2 = π * r3 ^ 2 - π * r2 ^ 2) : r2 = 12 := sorry


end intermediate_circle_radius_l106_106889


namespace smallest_interesting_number_l106_106885

def is_perfect_square (x : ℕ) : Prop :=
∃ y : ℕ, y * y = x

def is_perfect_cube (x : ℕ) : Prop :=
∃ y : ℕ, y * y * y = x

theorem smallest_interesting_number (n : ℕ) :
    is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n) → n = 1800 :=
by {
  sorry
}

end smallest_interesting_number_l106_106885


namespace purple_sequins_each_row_l106_106430

theorem purple_sequins_each_row (x : ℕ) : 
  (6 * 8) + (9 * 6) + (5 * x) = 162 → x = 12 :=
by 
  sorry

end purple_sequins_each_row_l106_106430


namespace possible_values_of_k_l106_106664

theorem possible_values_of_k (k : ℕ) (N : ℕ) (h₁ : (k * (k + 1)) / 2 = N^2) (h₂ : N < 100) :
  k = 1 ∨ k = 8 ∨ k = 49 :=
sorry

end possible_values_of_k_l106_106664


namespace bill_picked_apples_l106_106245

-- Definitions from conditions
def children := 2
def apples_per_child_per_teacher := 3
def favorite_teachers := 2
def apples_per_pie := 10
def pies_baked := 2
def apples_left := 24

-- Number of apples given to teachers
def apples_for_teachers := children * apples_per_child_per_teacher * favorite_teachers

-- Number of apples used for pies
def apples_for_pies := pies_baked * apples_per_pie

-- The final theorem to be stated
theorem bill_picked_apples :
  apples_for_teachers + apples_for_pies + apples_left = 56 := 
sorry

end bill_picked_apples_l106_106245


namespace exists_ellipse_l106_106107

theorem exists_ellipse (a : ℝ) : ∃ a : ℝ, ∀ x y : ℝ, (x^2 + y^2 / a = 1) → a > 0 ∧ a ≠ 1 := 
by 
  sorry

end exists_ellipse_l106_106107


namespace sum_of_cubes_roots_poly_l106_106178

theorem sum_of_cubes_roots_poly :
  (∀ (a b c : ℂ), (a^3 - 2*a^2 + 2*a - 3 = 0) ∧ (b^3 - 2*b^2 + 2*b - 3 = 0) ∧ (c^3 - 2*c^2 + 2*c - 3 = 0) → 
  a^3 + b^3 + c^3 = 5) :=
by
  sorry

end sum_of_cubes_roots_poly_l106_106178


namespace compare_neg_fractions_l106_106145

theorem compare_neg_fractions : (-3 / 4) > (-5 / 6) :=
sorry

end compare_neg_fractions_l106_106145


namespace total_cookies_after_three_days_l106_106267

-- Define the initial conditions
def cookies_baked_monday : ℕ := 32
def cookies_baked_tuesday : ℕ := cookies_baked_monday / 2
def cookies_baked_wednesday_before : ℕ := cookies_baked_tuesday * 3
def brother_ate : ℕ := 4

-- Define the total cookies before brother ate any
def total_cookies_before : ℕ := cookies_baked_monday + cookies_baked_tuesday + cookies_baked_wednesday_before

-- Define the total cookies after brother ate some
def total_cookies_after : ℕ := total_cookies_before - brother_ate

-- The proof statement
theorem total_cookies_after_three_days : total_cookies_after = 92 := by
  -- Here, we would provide the proof, but we add sorry for now to compile successfully.
  sorry

end total_cookies_after_three_days_l106_106267


namespace complex_number_in_first_quadrant_l106_106174

def is_in_first_quadrant (z : ℂ) : Prop :=
  0 < z.re ∧ 0 < z.im

theorem complex_number_in_first_quadrant (z : ℂ) (h : 0 < z.re ∧ 0 < z.im) : is_in_first_quadrant z :=
by sorry

end complex_number_in_first_quadrant_l106_106174


namespace smallest_x_l106_106348

noncomputable def digitSum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_x (x a : ℕ) (h1 : a = 100 * x + 4950)
  (h2 : digitSum a = 50) :
  x = 99950 :=
by sorry

end smallest_x_l106_106348


namespace rth_term_arithmetic_progression_l106_106644

-- Define the sum of the first n terms of the arithmetic progression
def S (n : ℕ) : ℕ := 5 * n + 4 * n^3

-- Define the r-th term of the arithmetic progression
def a (r : ℕ) : ℕ := S r - S (r - 1)

-- Theorem stating the r-th term of the arithmetic progression
theorem rth_term_arithmetic_progression (r : ℕ) : a r = 12 * r^2 - 12 * r + 9 := by
  sorry

end rth_term_arithmetic_progression_l106_106644


namespace part1_part2_l106_106032

def f (x a b : ℝ) : ℝ := |x - a| + |x - b|

theorem part1 (a b c x : ℝ) (h1 : |a - b| > c) : f x a b > c :=
  by sorry

theorem part2 (a : ℝ) (h1 : ∃ (x : ℝ), f x a 1 < 2 - |a - 2|) : 1/2 < a ∧ a < 5/2 :=
  by sorry

end part1_part2_l106_106032


namespace k_range_correct_l106_106181

noncomputable def k_range (k : ℝ) : Prop :=
  (∀ x : ℝ, ¬ (x ^ 2 + k * x + 9 / 4 = 0)) ∧
  (∀ x : ℝ, k * x ^ 2 + k * x + 1 > 0) ∧
  ((∃ x : ℝ, ¬ (x ^ 2 + k * x + 9 / 4 = 0)) ∨
   (∃ x : ℝ, k * x ^ 2 + k * x + 1 > 0)) ∧
  ¬ ((∃ x : ℝ, ¬ (x ^ 2 + k * x + 9 / 4 = 0)) ∧
    (∃ x : ℝ, k * x ^ 2 + k * x + 1 > 0))

theorem k_range_correct (k : ℝ) : k_range k ↔ (-3 < k ∧ k < 0) ∨ (3 ≤ k ∧ k < 4) :=
sorry

end k_range_correct_l106_106181


namespace isosceles_triangle_perimeter_l106_106637

/-- Given an isosceles triangle with one side length of 3 cm and another side length of 5 cm,
    its perimeter is either 11 cm or 13 cm. -/
theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 3) (h2 : b = 5) : 
  (∃ c : ℝ, (c = 3 ∨ c = 5) ∧ (2 * a + b = 11 ∨ 2 * b + a = 13)) :=
by
  sorry

end isosceles_triangle_perimeter_l106_106637


namespace travel_agency_choice_l106_106599

noncomputable def y₁ (x : ℝ) : ℝ := 350 * x + 1000

noncomputable def y₂ (x : ℝ) : ℝ := 400 * x + 800

theorem travel_agency_choice (x : ℝ) (h : 0 < x) :
  (x < 4 → y₁ x > y₂ x) ∧ 
  (x = 4 → y₁ x = y₂ x) ∧ 
  (x > 4 → y₁ x < y₂ x) :=
by {
  sorry
}

end travel_agency_choice_l106_106599


namespace total_wages_l106_106486

theorem total_wages (A_days B_days : ℝ) (A_wages : ℝ) (W : ℝ) 
  (h1 : A_days = 10)
  (h2 : B_days = 15)
  (h3 : A_wages = 2100) :
  W = 3500 :=
by sorry

end total_wages_l106_106486


namespace pure_imaginary_value_l106_106907

theorem pure_imaginary_value (a : ℝ) : (z = (0 : ℝ) + (a^2 + 2 * a - 3) * I) → (a = 0 ∨ a = -2) :=
by
  sorry

end pure_imaginary_value_l106_106907


namespace helen_hand_washing_time_l106_106187

theorem helen_hand_washing_time :
  (52 / 4) * 30 / 60 = 6.5 := by
  sorry

end helen_hand_washing_time_l106_106187


namespace repave_today_l106_106377

theorem repave_today (total_repaved : ℕ) (repaved_before_today : ℕ) (repaved_today : ℕ) :
  total_repaved = 4938 → repaved_before_today = 4133 → repaved_today = total_repaved - repaved_before_today → repaved_today = 805 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3


end repave_today_l106_106377


namespace bookstore_purchase_prices_equal_l106_106366

variable (x : ℝ)

theorem bookstore_purchase_prices_equal
  (h1 : 500 > 0)
  (h2 : 700 > 0)
  (h3 : x > 0)
  (h4 : x + 4 > 0)
  (h5 : ∃ p₁ p₂ : ℝ, p₁ = 500 / x ∧ p₂ = 700 / (x + 4) ∧ p₁ = p₂) :
  500 / x = 700 / (x + 4) :=
by
  sorry

end bookstore_purchase_prices_equal_l106_106366


namespace scientific_notation_of_3930_billion_l106_106112

theorem scientific_notation_of_3930_billion :
  (3930 * 10^9) = 3.93 * 10^12 :=
sorry

end scientific_notation_of_3930_billion_l106_106112


namespace irrational_implies_irrational_l106_106503

-- Define irrational number proposition
def is_irrational (x : ℝ) : Prop := ¬ ∃ (q : ℚ), x = q

-- Define the main proposition to prove
theorem irrational_implies_irrational (a : ℝ) : is_irrational (a - 2) → is_irrational a :=
by
  sorry

end irrational_implies_irrational_l106_106503


namespace find_a_2b_l106_106842

theorem find_a_2b 
  (a b : ℤ) 
  (h1 : a * b = -150) 
  (h2 : a + b = -23) : 
  a + 2 * b = -55 :=
sorry

end find_a_2b_l106_106842


namespace solve_for_x_l106_106525

def f (x : ℝ) : ℝ := x^2 + x - 1

theorem solve_for_x (x : ℝ) (h : f x = 5) : x = 2 ∨ x = -3 := 
by {
  sorry
}

end solve_for_x_l106_106525


namespace divides_or_l106_106373

-- Definitions
variables {m n : ℕ} -- using natural numbers (non-negative integers) for simplicity in Lean

-- Hypothesis: m ∨ n + m ∧ n = m + n
theorem divides_or (h : Nat.lcm m n + Nat.gcd m n = m + n) : m ∣ n ∨ n ∣ m :=
sorry

end divides_or_l106_106373


namespace vertex_set_is_parabola_l106_106103

variables (a c k : ℝ) (ha : a > 0) (hc : c > 0) (hk : k ≠ 0)

theorem vertex_set_is_parabola :
  ∃ (f : ℝ → ℝ), (∀ t : ℝ, f t = (-k^2 / (4 * a)) * t^2 + c) :=
sorry

end vertex_set_is_parabola_l106_106103


namespace max_min_value_x_eq_1_l106_106630

noncomputable def f (x k : ℝ) : ℝ := x^2 - 2 * (2 * k - 1) * x + 3 * k^2 - 2 * k + 6

theorem max_min_value_x_eq_1 :
  ∀ (k : ℝ), (∀ x : ℝ, ∃ m : ℝ, f x k = m → k = 1 → m = 6) → (∃ x : ℝ, x = 1) :=
by
  sorry

end max_min_value_x_eq_1_l106_106630


namespace inequality_problem_l106_106733

variable {a b c : ℝ}

theorem inequality_problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a / Real.sqrt (a^2 + 8 * b * c)) + (b / Real.sqrt (b^2 + 8 * c * a)) + (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
sorry

end inequality_problem_l106_106733


namespace find_a_minus_b_l106_106819

theorem find_a_minus_b (a b x y : ℤ)
  (h_x : x = 1)
  (h_y : y = 1)
  (h1 : a * x + b * y = 2)
  (h2 : x - b * y = 3) :
  a - b = 6 := by
  subst h_x
  subst h_y
  simp at h1 h2
  have h_b: b = -2 := by linarith
  have h_a: a = 4 := by linarith
  rw [h_a, h_b]
  norm_num

end find_a_minus_b_l106_106819


namespace find_smallest_subtract_l106_106985

-- Definitions for multiples
def is_mul_2 (n : ℕ) : Prop := 2 ∣ n
def is_mul_3 (n : ℕ) : Prop := 3 ∣ n
def is_mul_5 (n : ℕ) : Prop := 5 ∣ n

-- Statement of the problem
theorem find_smallest_subtract (x : ℕ) :
  (is_mul_2 (134 - x)) ∧ (is_mul_3 (134 - x)) ∧ (is_mul_5 (134 - x)) → x = 14 :=
by
  sorry

end find_smallest_subtract_l106_106985


namespace problem_l106_106595

-- Step 1: Define the transformation functions
def rotate_90_counterclockwise (h k x y : ℝ) : ℝ × ℝ :=
  (h - (y - k), k + (x - h))

def reflect_y_eq_x (x y : ℝ) : ℝ × ℝ :=
  (y, x)

-- Step 2: Define the given problem condition
theorem problem (a b : ℝ) :
  rotate_90_counterclockwise 2 3 (reflect_y_eq_x 5 1).fst (reflect_y_eq_x 5 1).snd = (a, b) →
  b - a = 0 :=
by
  intro h
  sorry

end problem_l106_106595


namespace number_of_cakes_sold_l106_106277

namespace Bakery

variables (cakes pastries sold_cakes sold_pastries : ℕ)

-- Defining the conditions
def pastries_sold := 154
def more_pastries_than_cakes := 76

-- Defining the problem statement
theorem number_of_cakes_sold (h1 : sold_pastries = pastries_sold) 
                             (h2 : sold_pastries = sold_cakes + more_pastries_than_cakes) : 
                             sold_cakes = 78 :=
by {
  sorry
}

end Bakery

end number_of_cakes_sold_l106_106277


namespace intersection_non_empty_l106_106621

open Set

def M (a : ℤ) : Set ℤ := {a, 0}
def N : Set ℤ := {x | 2 * x^2 - 5 * x < 0}

theorem intersection_non_empty (a : ℤ) (h : (M a) ∩ N ≠ ∅) : a = 1 ∨ a = 2 := 
sorry

end intersection_non_empty_l106_106621


namespace range_of_a_l106_106156

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) → 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l106_106156


namespace petya_max_margin_l106_106582

def max_margin_votes (total_votes P1 P2 V1 V2: ℕ) : ℕ := P1 + P2 - (V1 + V2)

theorem petya_max_margin 
  (P1 P2 V1 V2: ℕ)
  (H1: P1 = V1 + 9) 
  (H2: V2 = P2 + 9) 
  (H3: P1 + P2 + V1 + V2 = 27) 
  (H_win: P1 + P2 > V1 + V2) : 
  max_margin_votes 27 P1 P2 V1 V2 = 9 :=
by
  sorry

end petya_max_margin_l106_106582


namespace value_of_y_l106_106252

theorem value_of_y : (∃ y : ℝ, (1 / 3 - 1 / 4 = 4 / y) ∧ y = 48) :=
by
  sorry

end value_of_y_l106_106252


namespace sin_double_angle_l106_106742

theorem sin_double_angle (α : ℝ) (h : Real.tan α = 2) : Real.sin (2 * α) = 4 / 5 :=
by sorry

end sin_double_angle_l106_106742


namespace cookies_per_child_l106_106681

def num_adults : ℕ := 4
def num_children : ℕ := 6
def cookies_jar1 : ℕ := 240
def cookies_jar2 : ℕ := 360
def cookies_jar3 : ℕ := 480

def fraction_eaten_jar1 : ℚ := 1 / 4
def fraction_eaten_jar2 : ℚ := 1 / 3
def fraction_eaten_jar3 : ℚ := 1 / 5

theorem cookies_per_child :
  let eaten_jar1 := fraction_eaten_jar1 * cookies_jar1
  let eaten_jar2 := fraction_eaten_jar2 * cookies_jar2
  let eaten_jar3 := fraction_eaten_jar3 * cookies_jar3
  let remaining_jar1 := cookies_jar1 - eaten_jar1
  let remaining_jar2 := cookies_jar2 - eaten_jar2
  let remaining_jar3 := cookies_jar3 - eaten_jar3
  let total_remaining_cookies := remaining_jar1 + remaining_jar2 + remaining_jar3
  let cookies_each_child := total_remaining_cookies / num_children
  cookies_each_child = 134 := by
  sorry

end cookies_per_child_l106_106681


namespace range_of_m_l106_106423

theorem range_of_m (x m : ℝ) (h1 : (m - 1) / (x + 1) = 1) (h2 : x < 0) : m < 2 ∧ m ≠ 1 :=
by
  sorry

end range_of_m_l106_106423


namespace two_equal_sum_partition_three_equal_sum_partition_l106_106265

-- Definition 1: Sum of the set X_n
def sum_X_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Definition 2: Equivalences for partitioning X_n into two equal sum parts
def partition_two_equal_sum (n : ℕ) : Prop :=
  (n % 4 = 0 ∨ n % 4 = 3) ↔ ∃ (A B : Finset ℕ), A ∪ B = Finset.range n ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id

-- Definition 3: Equivalences for partitioning X_n into three equal sum parts
def partition_three_equal_sum (n : ℕ) : Prop :=
  (n % 3 ≠ 1) ↔ ∃ (A B C : Finset ℕ), A ∪ B ∪ C = Finset.range n ∧ (A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅) ∧ A.sum id = B.sum id ∧ B.sum id = C.sum id

-- Main theorem statements
theorem two_equal_sum_partition (n : ℕ) : partition_two_equal_sum n :=
  sorry

theorem three_equal_sum_partition (n : ℕ) : partition_three_equal_sum n :=
  sorry

end two_equal_sum_partition_three_equal_sum_partition_l106_106265


namespace phil_won_more_games_than_charlie_l106_106341

theorem phil_won_more_games_than_charlie :
  ∀ (P D C Ph : ℕ),
  (P = D + 5) → (C = D - 2) → (Ph = 12) → (P = Ph + 4) →
  Ph - C = 3 :=
by
  intros P D C Ph hP hC hPh hPPh
  sorry

end phil_won_more_games_than_charlie_l106_106341


namespace problem_l106_106549

theorem problem (f : ℝ → ℝ) (h : ∀ x, f (Real.cos x) = Real.cos (17 * x)) (x : ℝ) :
  f (Real.cos x) ^ 2 + f (Real.sin x) ^ 2 = 1 :=
sorry

end problem_l106_106549


namespace ratio_part_to_whole_l106_106396

/-- One part of one third of two fifth of a number is 17, and 40% of that number is 204. 
Prove that the ratio of the part to the whole number is 1:30. -/
theorem ratio_part_to_whole 
  (N : ℝ)
  (h1 : (1 / 1) * (1 / 3) * (2 / 5) * N = 17) 
  (h2 : 0.40 * N = 204) : 
  17 / N = 1 / 30 :=
  sorry

end ratio_part_to_whole_l106_106396


namespace find_ABC_sum_l106_106841

theorem find_ABC_sum (A B C : ℤ) (h : ∀ x : ℤ, x = -3 ∨ x = 0 ∨ x = 4 → x^3 + A * x^2 + B * x + C = 0) : 
  A + B + C = -13 := 
by 
  sorry

end find_ABC_sum_l106_106841


namespace jessica_has_three_dozens_of_red_marbles_l106_106840

-- Define the number of red marbles Sandy has
def sandy_red_marbles : ℕ := 144

-- Define the relationship between Sandy's and Jessica's red marbles
def relationship (jessica_red_marbles : ℕ) : Prop :=
  sandy_red_marbles = 4 * jessica_red_marbles

-- Define the question to find out how many dozens of red marbles Jessica has
def jessica_dozens (jessica_red_marbles : ℕ) := jessica_red_marbles / 12

-- Theorem stating that given the conditions, Jessica has 3 dozens of red marbles
theorem jessica_has_three_dozens_of_red_marbles (jessica_red_marbles : ℕ)
  (h : relationship jessica_red_marbles) : jessica_dozens jessica_red_marbles = 3 :=
by
  -- The proof is omitted
  sorry

end jessica_has_three_dozens_of_red_marbles_l106_106840


namespace weight_of_10m_l106_106537

-- Defining the proportional weight conditions
variable (weight_of_rod : ℝ → ℝ)

-- Conditional facts about the weight function
axiom weight_proportional : ∀ (length1 length2 : ℝ), length1 ≠ 0 → length2 ≠ 0 → 
  weight_of_rod length1 / length1 = weight_of_rod length2 / length2
axiom weight_of_6m : weight_of_rod 6 = 14.04

-- Theorem stating the weight of a 10m rod
theorem weight_of_10m : weight_of_rod 10 = 23.4 := 
sorry

end weight_of_10m_l106_106537


namespace sin_double_alpha_zero_l106_106394

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem sin_double_alpha_zero (α : ℝ) (h : f α = 1) : Real.sin (2 * α) = 0 :=
by 
  -- Proof would go here, but we're using sorry
  sorry

end sin_double_alpha_zero_l106_106394


namespace Jaron_prize_points_l106_106049

def points_bunnies (bunnies: Nat) (points_per_bunny: Nat) : Nat :=
  bunnies * points_per_bunny

def points_snickers (snickers: Nat) (points_per_snicker: Nat) : Nat :=
  snickers * points_per_snicker

def total_points (bunny_points: Nat) (snicker_points: Nat) : Nat :=
  bunny_points + snicker_points

theorem Jaron_prize_points :
  let bunnies := 8
  let points_per_bunny := 100
  let snickers := 48
  let points_per_snicker := 25
  let bunny_points := points_bunnies bunnies points_per_bunny
  let snicker_points := points_snickers snickers points_per_snicker
  total_points bunny_points snicker_points = 2000 := 
by
  sorry

end Jaron_prize_points_l106_106049


namespace solution_set_abs_inequality_l106_106657

theorem solution_set_abs_inequality :
  { x : ℝ | |x - 2| - |2 * x - 1| > 0 } = { x | -1 < x ∧ x < 1 } :=
by
  sorry

end solution_set_abs_inequality_l106_106657


namespace sampling_method_is_systematic_l106_106768

-- Define the conditions
structure Grade where
  num_classes : Nat
  students_per_class : Nat
  required_student_num : Nat

-- Define our specific problem's conditions
def problem_conditions : Grade :=
  { num_classes := 12, students_per_class := 50, required_student_num := 14 }

-- State the theorem
theorem sampling_method_is_systematic (G : Grade) (h1 : G.num_classes = 12) (h2 : G.students_per_class = 50) (h3 : G.required_student_num = 14) : 
  "Systematic sampling" = "Systematic sampling" :=
by
  sorry

end sampling_method_is_systematic_l106_106768


namespace collinear_iff_real_simple_ratio_l106_106206

theorem collinear_iff_real_simple_ratio (a b c : ℂ) : (∃ k : ℝ, a = k * b + (1 - k) * c) ↔ ∃ r : ℝ, (a - b) / (a - c) = r :=
sorry

end collinear_iff_real_simple_ratio_l106_106206


namespace drum_wife_leopard_cost_l106_106516

-- Definitions
variables (x y z : ℤ)

def system1 := 2 * x + 3 * y + z = 111
def system2 := 3 * x + 4 * y - 2 * z = -8
def even_condition := z % 2 = 0

theorem drum_wife_leopard_cost:
  system1 x y z ∧ system2 x y z ∧ even_condition z →
  x = 20 ∧ y = 9 ∧ z = 44 :=
by
  intro h
  -- Full proof can be provided here
  sorry

end drum_wife_leopard_cost_l106_106516


namespace half_radius_of_circle_y_l106_106920

theorem half_radius_of_circle_y 
  (r_x r_y : ℝ) 
  (h₁ : π * r_x^2 = π * r_y^2) 
  (h₂ : 2 * π * r_x = 14 * π) :
  r_y / 2 = 3.5 :=
by {
  sorry
}

end half_radius_of_circle_y_l106_106920


namespace incorrect_regression_statement_incorrect_statement_proof_l106_106436

-- Define the regression equation and the statement about y and x
def regression_equation (x : ℝ) : ℝ := 3 - 5 * x

-- Proof statement: given the regression equation, show that when x increases by one unit, y decreases by 5 units on average
theorem incorrect_regression_statement : 
  (regression_equation (x + 1) = regression_equation x + (-5)) :=
by sorry

-- Proof statement: prove that the statement "when the variable x increases by one unit, y increases by 5 units on average" is incorrect
theorem incorrect_statement_proof :
  ¬ (regression_equation (x + 1) = regression_equation x + 5) :=
by sorry  

end incorrect_regression_statement_incorrect_statement_proof_l106_106436


namespace coefficient_of_quadratic_polynomial_l106_106003

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem coefficient_of_quadratic_polynomial (a b c : ℝ) (h : a > 0) :
  |f a b c 1| = 2 ∧ |f a b c 2| = 2 ∧ |f a b c 3| = 2 →
  (a = 4 ∧ b = -16 ∧ c = 14) ∨ (a = 2 ∧ b = -6 ∧ c = 2) ∨ (a = 2 ∧ b = -10 ∧ c = 10) :=
by
  sorry

end coefficient_of_quadratic_polynomial_l106_106003


namespace ratio_of_kids_in_morning_to_total_soccer_l106_106752

-- Define the known conditions
def total_kids_in_camp : ℕ := 2000
def kids_going_to_soccer_camp : ℕ := total_kids_in_camp / 2
def kids_going_to_soccer_camp_in_afternoon : ℕ := 750
def kids_going_to_soccer_camp_in_morning : ℕ := kids_going_to_soccer_camp - kids_going_to_soccer_camp_in_afternoon

-- Define the conclusion to be proven
theorem ratio_of_kids_in_morning_to_total_soccer :
  (kids_going_to_soccer_camp_in_morning : ℚ) / (kids_going_to_soccer_camp : ℚ) = 1 / 4 :=
by
  sorry

end ratio_of_kids_in_morning_to_total_soccer_l106_106752


namespace coal_removal_date_l106_106369

theorem coal_removal_date (m n : ℝ) (h1 : 0 < m) (h2 : 0 < n)
  (h3 : 25 * m + 9 * n = 0.5)
  (h4 : ∃ z : ℝ,  z * (n + m) = 0.5)
  (h5 : ∀ z : ℝ, z = 12 → (16 + z) * m = (9 + z) * n):
  ∃ t : ℝ, t = 28 := 
by 
{
  sorry
}

end coal_removal_date_l106_106369


namespace gcd_1029_1437_5649_l106_106280

theorem gcd_1029_1437_5649 : Nat.gcd (Nat.gcd 1029 1437) 5649 = 3 := by
  sorry

end gcd_1029_1437_5649_l106_106280


namespace function_is_monotonically_increasing_l106_106987

theorem function_is_monotonically_increasing (a : ℝ) :
  (∀ x : ℝ, (x^2 + 2*x + a) ≥ 0) ↔ (1 ≤ a) := 
sorry

end function_is_monotonically_increasing_l106_106987


namespace a_alone_days_l106_106558

theorem a_alone_days 
  (B_days : ℕ)
  (B_days_eq : B_days = 8)
  (C_payment : ℝ)
  (C_payment_eq : C_payment = 450)
  (total_payment : ℝ)
  (total_payment_eq : total_payment = 3600)
  (combined_days : ℕ)
  (combined_days_eq : combined_days = 3)
  (combined_rate_eq : (1 / A + 1 / B_days + C = 1 / combined_days)) 
  (rate_proportion : (1 / A) / (1 / B_days) = 7 / 1) 
  : A = 56 :=
sorry

end a_alone_days_l106_106558


namespace circle_area_in_square_centimeters_l106_106870

theorem circle_area_in_square_centimeters (d_meters : ℤ) (h : d_meters = 8) :
  ∃ (A : ℤ), A = 160000 * Real.pi ∧ 
  A = π * (d_meters / 2) ^ 2 * 10000 :=
by
  sorry

end circle_area_in_square_centimeters_l106_106870


namespace solve_for_x_l106_106514

theorem solve_for_x (x : ℚ) (h : (x - 75) / 4 = (5 - 3 * x) / 7) : x = 545 / 19 :=
sorry

end solve_for_x_l106_106514


namespace initial_friends_online_l106_106409

theorem initial_friends_online (F : ℕ) 
  (h1 : 8 + F = 13) 
  (h2 : 6 * F = 30) : 
  F = 5 :=
by
  sorry

end initial_friends_online_l106_106409


namespace solution_set_of_inequality_l106_106414

-- We define the inequality condition
def inequality (x : ℝ) : Prop := (x - 3) * (x + 2) < 0

-- We need to state that for all real numbers x, iff x satisfies the inequality,
-- then x must be within the interval (-2, 3).
theorem solution_set_of_inequality :
  ∀ x : ℝ, inequality x ↔ -2 < x ∧ x < 3 :=
by {
   sorry
}

end solution_set_of_inequality_l106_106414


namespace music_students_count_l106_106338

open Nat

theorem music_students_count (total_students : ℕ) (art_students : ℕ) (both_music_art : ℕ) 
      (neither_music_art : ℕ) (M : ℕ) :
    total_students = 500 →
    art_students = 10 →
    both_music_art = 10 →
    neither_music_art = 470 →
    (total_students - neither_music_art) = 30 →
    (M + (art_students - both_music_art)) = 30 →
    M = 30 :=
by
  intros h_total h_art h_both h_neither h_music_art_total h_music_count
  sorry

end music_students_count_l106_106338


namespace find_brick_length_l106_106301

-- Definitions of dimensions
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6
def wall_length : ℝ := 750
def wall_height : ℝ := 600
def wall_thickness : ℝ := 22.5
def num_bricks : ℝ := 6000

-- Volume calculations
def volume_wall : ℝ := wall_length * wall_height * wall_thickness
def volume_brick (x : ℝ) : ℝ := x * brick_width * brick_height

-- Statement of the problem
theorem find_brick_length (length_of_brick : ℝ) :
  volume_wall = num_bricks * volume_brick length_of_brick → length_of_brick = 25 :=
by
  simp [volume_wall, volume_brick, num_bricks, brick_width, brick_height, wall_length, wall_height, wall_thickness]
  intro h 
  sorry

end find_brick_length_l106_106301


namespace sum_of_a_and_b_is_two_l106_106448

variable (a b : ℝ)
variable (h_a_nonzero : a ≠ 0)
variable (h_fn_passes_through_point : (a * 1^2 + b * 1 - 1) = 1)

theorem sum_of_a_and_b_is_two : a + b = 2 := 
by
  sorry

end sum_of_a_and_b_is_two_l106_106448


namespace original_rectangle_perimeter_difference_is_multiple_of_7_relationship_for_seamless_combination_l106_106221

-- Condition declarations
variables (x y : ℕ)
variables (hx : x > 0) (hy : y > 0)
variables (h_sum : x + y = 25)
variables (h_diff : (x + 5) * (y + 5) - (x - 2) * (y - 2) = 196)

-- Lean 4 statements for the proof problem
theorem original_rectangle_perimeter (x y : ℕ) (hx : x > 0) (hy : y > 0) (h_sum : x + y = 25) : 2 * (x + y) = 50 := by
  sorry

theorem difference_is_multiple_of_7 (x y : ℕ) (hx : x > 0) (hy : y > 0) 
  (h_diff : (x + 5) * (y + 5) - (x - 2) * (y - 2) = 196) : 7 ∣ ((x + 5) * (y + 5) - (x - 2) * (y - 2)) := by
  sorry

theorem relationship_for_seamless_combination (x y : ℕ) (h_sum : x + y = 25) (h_seamless : (x+5)*(y+5) = (x*(y+5))) : x = y + 5 := by
  sorry

end original_rectangle_perimeter_difference_is_multiple_of_7_relationship_for_seamless_combination_l106_106221


namespace total_players_l106_106429

theorem total_players (kabaddi : ℕ) (only_kho_kho : ℕ) (both_games : ℕ) 
  (h_kabaddi : kabaddi = 10) (h_only_kho_kho : only_kho_kho = 15) 
  (h_both_games : both_games = 5) : (kabaddi - both_games) + only_kho_kho + both_games = 25 :=
by
  sorry

end total_players_l106_106429


namespace sum_of_excluded_numbers_l106_106232

theorem sum_of_excluded_numbers (S : ℕ) (X : ℕ) (n m : ℕ) (averageN : ℕ) (averageM : ℕ)
  (h1 : S = 34 * 8) 
  (h2 : n = 8) 
  (h3 : m = 6) 
  (h4 : averageN = 34) 
  (h5 : averageM = 29) 
  (hS : S = n * averageN) 
  (hX : S - X = m * averageM) : 
  X = 98 := by
  sorry

end sum_of_excluded_numbers_l106_106232


namespace factor_expression_l106_106490

theorem factor_expression (b : ℤ) : 52 * b ^ 2 + 208 * b = 52 * b * (b + 4) := 
by {
  sorry
}

end factor_expression_l106_106490


namespace negation_proposition_l106_106557

theorem negation_proposition (a b c : ℝ) : 
  (¬ (a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3)) ↔ (a + b + c ≠ 3 → a^2 + b^2 + c^2 < 3) := 
by
  -- proof goes here
  sorry

end negation_proposition_l106_106557


namespace prod_eq_one_l106_106326

noncomputable def is_parity_equal (A : Finset ℝ) (a : ℝ) : Prop :=
  (A.filter (fun x => x > a)).card % 2 = (A.filter (fun x => x < 1/a)).card % 2

theorem prod_eq_one
  (A : Finset ℝ)
  (hA : ∀ (a : ℝ), 0 < a → is_parity_equal A a)
  (hA_pos : ∀ x ∈ A, 0 < x) :
  A.prod id = 1 :=
sorry

end prod_eq_one_l106_106326


namespace angle_BC₁_plane_BBD₁D_l106_106912

-- Define all the necessary components of the cube and its geometry
variables {A B C D A₁ B₁ C₁ D₁ : ℝ} -- placeholders for points, represented by real coordinates

def is_cube (A B C D A₁ B₁ C₁ D₁ : ℝ) : Prop := sorry -- Define the cube property (this would need a proper definition)

def space_diagonal (B C₁ : ℝ) : Prop := sorry -- Define the property of being a space diagonal

def plane (B B₁ D₁ D : ℝ) : Prop := sorry -- Define a plane through these points (again needs a definition)

-- Define the angle between a line and a plane
def angle_between_line_and_plane (BC₁ B B₁ D₁ D : ℝ) : ℝ := sorry -- Define angle calculation (requires more context)

-- The proof statement, which is currently not proven (contains 'sorry')
theorem angle_BC₁_plane_BBD₁D (s : ℝ):
  is_cube A B C D A₁ B₁ C₁ D₁ →
  space_diagonal B C₁ →
  plane B B₁ D₁ D →
  angle_between_line_and_plane B C₁ B₁ D₁ D = π / 6 :=
sorry

end angle_BC₁_plane_BBD₁D_l106_106912


namespace double_neg_eq_pos_l106_106065

theorem double_neg_eq_pos (n : ℤ) : -(-n) = n := by
  sorry

example : -(-2023) = 2023 :=
  double_neg_eq_pos 2023

end double_neg_eq_pos_l106_106065


namespace ratio_of_area_l106_106033

-- Define the sides of the triangles
def sides_GHI := (6, 8, 10)
def sides_JKL := (9, 12, 15)

-- Define the function to calculate the area of a triangle given its sides as a Pythagorean triple
def area_of_right_triangle (a b : Nat) : Nat :=
  (a * b) / 2

-- Calculate the areas
def area_GHI := area_of_right_triangle 6 8
def area_JKL := area_of_right_triangle 9 12

-- Calculate the ratio of the areas
def ratio_of_areas := area_GHI * 2 / area_JKL * 3

-- Statement to prove
theorem ratio_of_area (sides_GHI sides_JKL : (Nat × Nat × Nat)) :
  (ratio_of_areas = (4 : ℚ) / 9) :=
sorry

end ratio_of_area_l106_106033


namespace like_terms_exponents_equal_l106_106422

theorem like_terms_exponents_equal (a b : ℤ) :
  (∀ x y : ℝ, 2 * x^a * y^2 = -3 * x^3 * y^(b+3) → a = 3 ∧ b = -1) :=
by
  sorry

end like_terms_exponents_equal_l106_106422


namespace sum_of_triangulars_15_to_20_l106_106887

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_of_triangulars_15_to_20 : 
  (triangular_number 15 + triangular_number 16 + triangular_number 17 + triangular_number 18 + triangular_number 19 + triangular_number 20) = 980 :=
by
  sorry

end sum_of_triangulars_15_to_20_l106_106887


namespace workshop_total_workers_l106_106950

theorem workshop_total_workers
  (avg_salary_per_head : ℕ)
  (num_technicians num_managers num_apprentices total_workers : ℕ)
  (avg_tech_salary avg_mgr_salary avg_appr_salary : ℕ) 
  (h1 : avg_salary_per_head = 700)
  (h2 : num_technicians = 5)
  (h3 : num_managers = 3)
  (h4 : avg_tech_salary = 800)
  (h5 : avg_mgr_salary = 1200)
  (h6 : avg_appr_salary = 650)
  (h7 : total_workers = num_technicians + num_managers + num_apprentices)
  : total_workers = 48 := 
sorry

end workshop_total_workers_l106_106950


namespace remainder_when_divided_by_9_l106_106487

variable (k : ℕ)

theorem remainder_when_divided_by_9 :
  (∃ k, k % 5 = 2 ∧ k % 6 = 3 ∧ k % 8 = 7 ∧ k < 100) →
  k % 9 = 6 :=
sorry

end remainder_when_divided_by_9_l106_106487


namespace decreasing_power_function_on_interval_l106_106933

noncomputable def power_function (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 + m - 3)

theorem decreasing_power_function_on_interval (m : ℝ) :
  (∀ x : ℝ, (0 < x) -> power_function m x < 0) ↔ m = -1 := 
by 
  sorry

end decreasing_power_function_on_interval_l106_106933


namespace count_primes_1021_eq_one_l106_106946

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_primes_1021_eq_one :
  (∃ n : ℕ, 3 ≤ n ∧ is_prime (n^3 + 2*n + 1) ∧
  ∀ m : ℕ, (3 ≤ m ∧ m ≠ n) → ¬ is_prime (m^3 + 2*m + 1)) :=
sorry

end count_primes_1021_eq_one_l106_106946


namespace max_mow_time_l106_106425

-- Define the conditions
def timeToMow (x : ℕ) : Prop := 
  let timeToFertilize := 2 * x
  x + timeToFertilize = 120

-- State the theorem
theorem max_mow_time (x : ℕ) (h : timeToMow x) : x = 40 := by
  sorry

end max_mow_time_l106_106425


namespace find_domain_l106_106225

noncomputable def domain (x : ℝ) : Prop :=
  (2 * x + 1 ≥ 0) ∧ (3 - 4 * x ≥ 0)

theorem find_domain :
  {x : ℝ | domain x} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 3/4} :=
by
  sorry

end find_domain_l106_106225


namespace smaller_two_digit_product_l106_106339

theorem smaller_two_digit_product (a b : ℕ) (h1: 10 ≤ a) (h2: a ≤ 99) (h3: 10 ≤ b) (h4: b ≤ 99) (h_prod: a * b = 4536) : a = 54 ∨ b = 54 :=
by sorry

end smaller_two_digit_product_l106_106339


namespace jake_more_peaches_than_jill_l106_106002

theorem jake_more_peaches_than_jill :
  let steven_peaches := 14
  let jake_peaches := steven_peaches - 6
  let jill_peaches := 5
  jake_peaches - jill_peaches = 3 :=
by
  let steven_peaches := 14
  let jake_peaches := steven_peaches - 6
  let jill_peaches := 5
  sorry

end jake_more_peaches_than_jill_l106_106002


namespace point_on_x_axis_l106_106738

theorem point_on_x_axis (m : ℤ) (hx : 2 + m = 0) : (m - 3, 2 + m) = (-5, 0) :=
by sorry

end point_on_x_axis_l106_106738


namespace roses_in_december_l106_106776

theorem roses_in_december (rOct rNov rJan rFeb : ℕ) 
  (hOct : rOct = 108)
  (hNov : rNov = 120)
  (hJan : rJan = 144)
  (hFeb : rFeb = 156)
  (pattern : (rNov - rOct = 12 ∨ rNov - rOct = 24) ∧ 
             (rJan - rNov = 12 ∨ rJan - rNov = 24) ∧
             (rFeb - rJan = 12 ∨ rFeb - rJan = 24) ∧ 
             (∀ m n, (m - n = 12 ∨ m - n = 24) → 
               ((rNov - rOct) ≠ (rJan - rNov) ↔ 
               (rJan - rNov) ≠ (rFeb - rJan)))) : 
  ∃ rDec : ℕ, rDec = 132 := 
by {
  sorry
}

end roses_in_december_l106_106776


namespace smallest_lcm_not_multiple_of_25_l106_106210

theorem smallest_lcm_not_multiple_of_25 (n : ℕ) (h1 : n % 36 = 0) (h2 : n % 45 = 0) (h3 : n % 25 ≠ 0) : n = 180 := 
by 
  sorry

end smallest_lcm_not_multiple_of_25_l106_106210


namespace regular_ngon_on_parallel_lines_l106_106581

theorem regular_ngon_on_parallel_lines (n : ℕ) : 
  (∃ f : ℝ → ℝ, (∀ m : ℕ, ∃ k : ℕ, f (m * (360 / n)) = k * (360 / n))) ↔
  n = 3 ∨ n = 4 ∨ n = 6 := 
sorry

end regular_ngon_on_parallel_lines_l106_106581


namespace broken_pieces_correct_l106_106129

variable (pieces_transported : ℕ)
variable (shipping_cost_per_piece : ℝ)
variable (compensation_per_broken_piece : ℝ)
variable (total_profit : ℝ)
variable (broken_pieces : ℕ)

def logistics_profit (pieces_transported : ℕ) (shipping_cost_per_piece : ℝ) 
                     (compensation_per_broken_piece : ℝ) (broken_pieces : ℕ) : ℝ :=
  shipping_cost_per_piece * (pieces_transported - broken_pieces) - compensation_per_broken_piece * broken_pieces

theorem broken_pieces_correct :
  pieces_transported = 2000 →
  shipping_cost_per_piece = 0.2 →
  compensation_per_broken_piece = 2.3 →
  total_profit = 390 →
  logistics_profit pieces_transported shipping_cost_per_piece compensation_per_broken_piece broken_pieces = total_profit →
  broken_pieces = 4 :=
by
  intros
  sorry

end broken_pieces_correct_l106_106129


namespace bulbs_in_bathroom_and_kitchen_l106_106136

theorem bulbs_in_bathroom_and_kitchen
  (bedroom_bulbs : Nat)
  (basement_bulbs : Nat)
  (garage_bulbs : Nat)
  (bulbs_per_pack : Nat)
  (packs_needed : Nat)
  (total_bulbs : Nat)
  (H1 : bedroom_bulbs = 2)
  (H2 : basement_bulbs = 4)
  (H3 : garage_bulbs = basement_bulbs / 2)
  (H4 : bulbs_per_pack = 2)
  (H5 : packs_needed = 6)
  (H6 : total_bulbs = packs_needed * bulbs_per_pack) :
  (total_bulbs - (bedroom_bulbs + basement_bulbs + garage_bulbs) = 4) :=
by
  sorry

end bulbs_in_bathroom_and_kitchen_l106_106136


namespace max_m_value_inequality_abc_for_sum_l106_106084

-- Define the mathematical conditions and the proof problem.

theorem max_m_value (x m : ℝ) (h1 : |x - 2| - |x + 3| ≥ |m + 1|) :
  m ≤ 4 :=
sorry

theorem inequality_abc_for_sum (a b c : ℝ) (habc_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum_eq_M : a + 2 * b + c = 4) :
  (1 / (a + b)) + (1 / (b + c)) ≥ 1 :=
sorry

end max_m_value_inequality_abc_for_sum_l106_106084


namespace total_points_scored_l106_106189

theorem total_points_scored (m2 m3 m1 o2 o3 o1 : ℕ) 
  (H1 : m2 = 25) 
  (H2 : m3 = 8) 
  (H3 : m1 = 10) 
  (H4 : o2 = 2 * m2) 
  (H5 : o3 = m3 / 2) 
  (H6 : o1 = m1 / 2) : 
  (2 * m2 + 3 * m3 + m1) + (2 * o2 + 3 * o3 + o1) = 201 := 
by
  sorry

end total_points_scored_l106_106189


namespace digit_y_in_base_7_divisible_by_19_l106_106607

def base7_to_decimal (a b c d : ℕ) : ℕ := a * 7^3 + b * 7^2 + c * 7 + d

theorem digit_y_in_base_7_divisible_by_19 (y : ℕ) (hy : y < 7) :
  (∃ k : ℕ, base7_to_decimal 5 2 y 3 = 19 * k) ↔ y = 8 :=
by {
  sorry
}

end digit_y_in_base_7_divisible_by_19_l106_106607


namespace given_polynomial_l106_106922

noncomputable def f (x : ℝ) := x^3 - 2

theorem given_polynomial (x : ℝ) : 
  8 * f (x^3) - x^6 * f (2 * x) - 2 * f (x^2) + 12 = 0 :=
by
  sorry

end given_polynomial_l106_106922


namespace circle_sector_radius_l106_106263

theorem circle_sector_radius (r : ℝ) :
  (2 * r + (r * (Real.pi / 3)) = 144) → r = 432 / (6 + Real.pi) := by
  sorry

end circle_sector_radius_l106_106263


namespace number_of_small_triangles_l106_106050

noncomputable def area_of_large_triangle (hypotenuse_large : ℝ) : ℝ :=
  let leg := hypotenuse_large / Real.sqrt 2
  (1 / 2) * (leg * leg)

noncomputable def area_of_small_triangle (hypotenuse_small : ℝ) : ℝ :=
  let leg := hypotenuse_small / Real.sqrt 2
  (1 / 2) * (leg * leg)

theorem number_of_small_triangles (hypotenuse_large : ℝ) (hypotenuse_small : ℝ) :
  hypotenuse_large = 14 → hypotenuse_small = 2 →
  let number_of_triangles := (area_of_large_triangle hypotenuse_large) / (area_of_small_triangle hypotenuse_small)
  number_of_triangles = 49 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end number_of_small_triangles_l106_106050


namespace trigonometric_identity_l106_106839

theorem trigonometric_identity :
  (1 / 2 - (Real.cos (15 * Real.pi / 180)) ^ 2) = - (Real.sqrt 3 / 4) :=
by
  sorry

end trigonometric_identity_l106_106839


namespace figure_can_be_rearranged_to_square_l106_106166

def can_form_square (n : ℕ) : Prop :=
  let s := Nat.sqrt n
  s * s = n

theorem figure_can_be_rearranged_to_square (n : ℕ) :
  (∃ a b c : ℕ, a + b + c = n) → (can_form_square n) → (n % 1 = 0) :=
by
  intros _ _
  sorry

end figure_can_be_rearranged_to_square_l106_106166


namespace find_a6_geometric_sequence_l106_106295

noncomputable def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

theorem find_a6_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (h1 : geom_seq a q) (h2 : a 4 = 7) (h3 : a 8 = 63) : 
  a 6 = 21 :=
sorry

end find_a6_geometric_sequence_l106_106295


namespace solve_problems_l106_106864

variable (initial_problems : ℕ) 
variable (additional_problems : ℕ)

theorem solve_problems
  (h1 : initial_problems = 12) 
  (h2 : additional_problems = 7) : 
  initial_problems + additional_problems = 19 := 
by 
  sorry

end solve_problems_l106_106864


namespace roots_equation_l106_106978

theorem roots_equation (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : a * 4^3 + b * 4^2 + c * 4 + d = 0) (h₃ : a * (-3)^3 + b * (-3)^2 + c * (-3) + d = 0) :
  (b + c) / a = -13 :=
by
  sorry

end roots_equation_l106_106978


namespace john_total_distance_l106_106019

theorem john_total_distance :
  let s₁ : ℝ := 45       -- Speed for the first part (mph)
  let t₁ : ℝ := 2        -- Time for the first part (hours)
  let s₂ : ℝ := 50       -- Speed for the second part (mph)
  let t₂ : ℝ := 3        -- Time for the second part (hours)
  let d₁ : ℝ := s₁ * t₁ -- Distance for the first part
  let d₂ : ℝ := s₂ * t₂ -- Distance for the second part
  d₁ + d₂ = 240          -- Total distance
:= by
  sorry

end john_total_distance_l106_106019


namespace candle_blow_out_l106_106993

-- Definitions related to the problem.
def funnel := true -- Simplified representation of the funnel
def candle_lit := true -- Simplified representation of the lit candle
def airflow_concentration (align: Bool) : Prop :=
if align then true -- Airflow intersects the flame correctly
else false -- Airflow does not intersect the flame correctly

theorem candle_blow_out (align : Bool) : funnel ∧ candle_lit ∧ airflow_concentration align → align := sorry

end candle_blow_out_l106_106993


namespace minimal_face_sum_of_larger_cube_l106_106618

-- Definitions
def num_small_cubes : ℕ := 27
def num_faces_per_cube : ℕ := 6

-- The goal: Prove the minimal sum of the integers shown on the faces of the larger cube
theorem minimal_face_sum_of_larger_cube (min_sum : ℤ) 
    (H : min_sum = 90) :
    min_sum = 90 :=
by {
  sorry
}

end minimal_face_sum_of_larger_cube_l106_106618


namespace pow_div_pow_eq_l106_106926

theorem pow_div_pow_eq :
  (3^12) / (27^2) = 729 :=
by
  -- We'll use the provided conditions and proof outline
  -- 1. 27 = 3^3
  -- 2. (a^b)^c = a^{bc}
  -- 3. a^b \div a^c = a^{b-c}
  sorry

end pow_div_pow_eq_l106_106926


namespace sqrt_meaningful_l106_106947

theorem sqrt_meaningful (x : ℝ) : x + 1 >= 0 ↔ (∃ y : ℝ, y * y = x + 1) := by
  sorry

end sqrt_meaningful_l106_106947


namespace unique_solution_values_l106_106856

theorem unique_solution_values (x y a : ℝ) :
  (∀ x y a, x^2 + y^2 + 2 * x ≤ 1 ∧ x - y + a = 0) → (a = -1 ∨ a = 3) :=
by
  intro h
  sorry

end unique_solution_values_l106_106856


namespace circle_chord_length_equal_l106_106259

def equation_of_circle (D E F : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0

def distances_equal (D E F : ℝ) : Prop :=
  (D^2 ≠ E^2 ∧ E^2 > 4 * F) → 
  (∀ x y : ℝ, (x^2 + y^2 + D * x + E * y + F = 0) → (x = -D/2) ∧ (y = -E/2) → (abs x = abs y))

theorem circle_chord_length_equal (D E F : ℝ) (h : D^2 ≠ E^2 ∧ E^2 > 4 * F) :
  distances_equal D E F :=
by
  sorry

end circle_chord_length_equal_l106_106259


namespace express_y_in_terms_of_x_l106_106393

theorem express_y_in_terms_of_x (x y : ℝ) (h : y - 2 * x = 6) : y = 2 * x + 6 :=
by
  sorry

end express_y_in_terms_of_x_l106_106393


namespace pumpkins_at_other_orchard_l106_106611

-- Defining the initial conditions
def sunshine_pumpkins : ℕ := 54
def other_orchard_pumpkins : ℕ := 14

-- Equation provided in the problem
def condition_equation (P : ℕ) : Prop := 54 = 3 * P + 12

-- Proving the main statement using the conditions
theorem pumpkins_at_other_orchard : condition_equation other_orchard_pumpkins :=
by
  unfold condition_equation
  sorry -- To be completed with the proof

end pumpkins_at_other_orchard_l106_106611


namespace center_of_circle_l106_106010

theorem center_of_circle (A B : ℝ × ℝ) (hA : A = (2, -3)) (hB : B = (10, 5)) :
    (A.1 + B.1) / 2 = 6 ∧ (A.2 + B.2) / 2 = 1 :=
by
  sorry

end center_of_circle_l106_106010


namespace find_a_plus_b_l106_106324

variables (a b c d x : ℝ)

def conditions (a b c d x : ℝ) : Prop :=
  (a + b = x) ∧
  (b + c = 9) ∧
  (c + d = 3) ∧
  (a + d = 5)

theorem find_a_plus_b (a b c d x : ℝ) (h : conditions a b c d x) : a + b = 11 :=
by
  have h1 : a + b = x := h.1
  have h2 : b + c = 9 := h.2.1
  have h3 : c + d = 3 := h.2.2.1
  have h4 : a + d = 5 := h.2.2.2
  sorry

end find_a_plus_b_l106_106324


namespace vector_t_perpendicular_l106_106675

theorem vector_t_perpendicular (t : ℝ) :
  let a := (2, 4)
  let b := (-1, 1)
  let c := (2 + t, 4 - t)
  b.1 * c.1 + b.2 * c.2 = 0 → t = 1 := by
  sorry

end vector_t_perpendicular_l106_106675


namespace parabola_unique_intersection_x_axis_l106_106511

theorem parabola_unique_intersection_x_axis (m : ℝ) :
  (∃ x : ℝ, x^2 - 6*x + m = 0 ∧ ∀ y, y^2 - 6*y + m = 0 → y = x) → m = 9 :=
by
  sorry

end parabola_unique_intersection_x_axis_l106_106511


namespace cary_ivy_removal_days_correct_l106_106969

noncomputable def cary_ivy_removal_days (initial_ivy : ℕ) (ivy_removed_per_day : ℕ) (ivy_growth_per_night : ℕ) : ℕ :=
  initial_ivy / (ivy_removed_per_day - ivy_growth_per_night)

theorem cary_ivy_removal_days_correct :
  cary_ivy_removal_days 40 6 2 = 10 :=
by
  -- The body of the proof is omitted; it will be filled with the actual proof.
  sorry

end cary_ivy_removal_days_correct_l106_106969


namespace convex_pentagons_l106_106351

theorem convex_pentagons (P : Finset ℝ) (h : P.card = 15) : 
  (P.card.choose 5) = 3003 := 
by
  sorry

end convex_pentagons_l106_106351


namespace avg_of_six_is_3_9_l106_106726

noncomputable def avg_of_six_numbers 
  (avg1 avg2 avg3 : ℝ) 
  (h1 : avg1 = 3.4) 
  (h2 : avg2 = 3.85) 
  (h3 : avg3 = 4.45) : ℝ :=
  (2 * avg1 + 2 * avg2 + 2 * avg3) / 6

theorem avg_of_six_is_3_9 
  (avg1 avg2 avg3 : ℝ) 
  (h1 : avg1 = 3.4) 
  (h2 : avg2 = 3.85) 
  (h3 : avg3 = 4.45) : 
  avg_of_six_numbers avg1 avg2 avg3 h1 h2 h3 = 3.9 := 
by {
  sorry
}

end avg_of_six_is_3_9_l106_106726


namespace investment_difference_l106_106467

theorem investment_difference (x y z : ℕ) 
  (h1 : x + (x + y) + (x + 2 * y) = 9000)
  (h2 : (z / 9000) = (800 / 1800)) 
  (h3 : z = x + 2 * y) :
  y = 1000 := 
by
  -- omitted proof steps
  sorry

end investment_difference_l106_106467


namespace least_side_is_8_l106_106545

-- Define the sides of the right triangle
variables (a b : ℝ) (h : a = 8) (k : b = 15)

-- Define a predicate for the least possible length of the third side
def least_possible_third_side (c : ℝ) : Prop :=
  (c = 8) ∨ (c = 15) ∨ (c = 17)

theorem least_side_is_8 (c : ℝ) (hc : least_possible_third_side c) : c = 8 :=
by
  sorry

end least_side_is_8_l106_106545


namespace initial_maple_trees_l106_106275

theorem initial_maple_trees
  (initial_maple_trees : ℕ)
  (to_be_planted : ℕ)
  (final_maple_trees : ℕ)
  (h1 : to_be_planted = 9)
  (h2 : final_maple_trees = 11) :
  initial_maple_trees + to_be_planted = final_maple_trees → initial_maple_trees = 2 := 
by 
  sorry

end initial_maple_trees_l106_106275


namespace fraction_pow_zero_l106_106040

theorem fraction_pow_zero :
  let a := 7632148
  let b := -172836429
  (a / b ≠ 0) → (a / b)^0 = 1 := by
  sorry

end fraction_pow_zero_l106_106040


namespace distinct_natural_primes_l106_106708

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem distinct_natural_primes :
  ∃ (a b c d : ℕ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 5 ∧
  is_prime (a * b + c * d) ∧
  is_prime (a * c + b * d) ∧
  is_prime (a * d + b * c) := by
  sorry

end distinct_natural_primes_l106_106708


namespace largest_prime_inequality_l106_106636

def largest_prime_divisor (n : Nat) : Nat :=
  sorry  -- Placeholder to avoid distractions in problem statement

theorem largest_prime_inequality (q : Nat) (h_q_prime : Prime q) (hq_odd : q % 2 = 1) :
    ∃ k : Nat, k > 0 ∧ largest_prime_divisor (q^(2^k) - 1) < q ∧ q < largest_prime_divisor (q^(2^k) + 1) :=
sorry

end largest_prime_inequality_l106_106636


namespace probability_no_coinciding_sides_l106_106122

theorem probability_no_coinciding_sides :
  let total_triangles := Nat.choose 10 3
  let unfavorable_outcomes := 60 + 10
  let favorable_outcomes := total_triangles - unfavorable_outcomes
  favorable_outcomes / total_triangles = 5 / 12 := by
  sorry

end probability_no_coinciding_sides_l106_106122


namespace cos_beta_value_l106_106704

open Real

theorem cos_beta_value (α β : ℝ) (h1 : sin α = sqrt 5 / 5) (h2 : sin (α - β) = - sqrt 10 / 10) (h3 : 0 < α ∧ α < π / 2) (h4 : 0 < β ∧ β < π / 2) : cos β = sqrt 2 / 2 :=
by
sorry

end cos_beta_value_l106_106704


namespace largest_gcd_l106_106305

theorem largest_gcd (a b : ℕ) (h : a + b = 1008) : ∃ d, d = gcd a b ∧ (∀ d', d' = gcd a b → d' ≤ d) ∧ d = 504 :=
by
  sorry

end largest_gcd_l106_106305


namespace total_distance_covered_l106_106715

theorem total_distance_covered :
  let speed_fox := 50       -- km/h
  let speed_rabbit := 60    -- km/h
  let speed_deer := 80      -- km/h
  let time_hours := 2       -- hours
  let distance_fox := speed_fox * time_hours
  let distance_rabbit := speed_rabbit * time_hours
  let distance_deer := speed_deer * time_hours
  distance_fox + distance_rabbit + distance_deer = 380 := by
sorry

end total_distance_covered_l106_106715


namespace solve_base_6_addition_l106_106290

variables (X Y k : ℕ)

theorem solve_base_6_addition (h1 : Y + 3 = X) (h2 : ∃ k, X + 5 = 2 + 6 * k) : X + Y = 3 :=
sorry

end solve_base_6_addition_l106_106290


namespace sum_of_solutions_eq_zero_l106_106075

theorem sum_of_solutions_eq_zero :
  ∀ x : ℝ, (-π ≤ x ∧ x ≤ 3 * π ∧ (1 / Real.sin x + 1 / Real.cos x = 4))
  → x = 0 := sorry

end sum_of_solutions_eq_zero_l106_106075


namespace max_length_cos_theta_l106_106006

def domain (x y : ℝ) : Prop := (x^2 + (y - 1)^2 ≤ 1 ∧ x ≥ (Real.sqrt 2 / 3))

theorem max_length_cos_theta :
  (∃ x y : ℝ, domain x y ∧ ∀ θ : ℝ, (0 < θ ∧ θ < (Real.pi / 2)) → θ = Real.arctan (Real.sqrt 2) → 
  (Real.cos θ = Real.sqrt 3 / 3)) := sorry

end max_length_cos_theta_l106_106006


namespace tan_sum_pi_over_4_sin_cos_fraction_l106_106193

open Real

variable (α : ℝ)

axiom tan_α_eq_2 : tan α = 2

theorem tan_sum_pi_over_4 (α : ℝ) (h : tan α = 2) : tan (α + π / 4) = -3 :=
sorry

theorem sin_cos_fraction (α : ℝ) (h : tan α = 2) : (sin α + cos α) / (sin α - cos α) = 3 :=
sorry

end tan_sum_pi_over_4_sin_cos_fraction_l106_106193


namespace infinite_seq_condition_l106_106311

theorem infinite_seq_condition (x : ℕ → ℕ) (n m : ℕ) : 
  (∀ i, x i = 0 → x (i + m) = 1) → 
  (∀ i, x i = 1 → x (i + n) = 0) → 
  ∃ d p q : ℕ, n = 2^d * p ∧ m = 2^d * q ∧ p % 2 = 1 ∧ q % 2 = 1  :=
by 
  intros h1 h2 
  sorry

end infinite_seq_condition_l106_106311


namespace geom_inequality_l106_106157

noncomputable def geom_seq_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
if q = 1 then n * a1 else a1 * (1 - q^n) / (1 - q)

theorem geom_inequality (a1 q : ℝ) (h_q : q ≠ 0) :
  (a1 * (a1 * q^2)) > 0 :=
by
  sorry

end geom_inequality_l106_106157


namespace complex_fraction_sum_l106_106196

theorem complex_fraction_sum :
  let a := (1 : ℂ)
  let b := (0 : ℂ)
  (a + b) = 1 :=
by
  sorry

end complex_fraction_sum_l106_106196


namespace quadratic_points_order_l106_106099

theorem quadratic_points_order (y1 y2 y3 : ℝ) :
  (y1 = -2 * (1:ℝ) ^ 2 + 4) →
  (y2 = -2 * (2:ℝ) ^ 2 + 4) →
  (y3 = -2 * (-3:ℝ) ^ 2 + 4) →
  y1 > y2 ∧ y2 > y3 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end quadratic_points_order_l106_106099


namespace Will_Had_28_Bottles_l106_106838

-- Definitions based on conditions
-- Let days be the number of days water lasted (4 days)
def days : ℕ := 4

-- Let bottles_per_day be the number of bottles Will drank each day (7 bottles/day)
def bottles_per_day : ℕ := 7

-- Correct answer defined as total number of bottles (28 bottles)
def total_bottles : ℕ := 28

-- The proof statement to show that the total number of bottles is equal to 28
theorem Will_Had_28_Bottles :
  (bottles_per_day * days = total_bottles) :=
by
  sorry

end Will_Had_28_Bottles_l106_106838


namespace log_10_7_eqn_l106_106528

variables (p q : ℝ)
noncomputable def log_base (a b : ℝ) : ℝ := (Real.log b) / (Real.log a)

theorem log_10_7_eqn (h1 : log_base 4 5 = p) (h2 : log_base 5 7 = q) : 
  log_base 10 7 = (2 * p * q) / (2 * p + 1) :=
by 
  sorry

end log_10_7_eqn_l106_106528


namespace number_of_integer_values_of_a_l106_106662

theorem number_of_integer_values_of_a (a : ℤ) : 
  (∃ x : ℤ, x^2 + a * x + 9 * a = 0) ↔ 
  (∃ (a_values : Finset ℤ), a_values.card = 6 ∧ ∀ a ∈ a_values, ∃ x : ℤ, x^2 + a * x + 9 * a = 0) :=
by
  sorry

end number_of_integer_values_of_a_l106_106662


namespace values_of_y_satisfy_quadratic_l106_106803

theorem values_of_y_satisfy_quadratic :
  (∃ (x y : ℝ), 3 * x^2 + 4 * x + 7 * y + 2 = 0 ∧ 3 * x + 2 * y + 4 = 0) →
  (∃ (y : ℝ), 4 * y^2 + 29 * y + 6 = 0) :=
by sorry

end values_of_y_satisfy_quadratic_l106_106803


namespace randy_quiz_score_l106_106623

theorem randy_quiz_score (q1 q2 q3 q5 : ℕ) (q4 : ℕ) :
  q1 = 90 → q2 = 98 → q3 = 94 → q5 = 96 → (q1 + q2 + q3 + q4 + q5) / 5 = 94 → q4 = 92 :=
by
  intros h1 h2 h3 h5 h_avg
  sorry

end randy_quiz_score_l106_106623


namespace solution_set_non_empty_implies_a_gt_1_l106_106653

theorem solution_set_non_empty_implies_a_gt_1 (a : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 := 
  sorry

end solution_set_non_empty_implies_a_gt_1_l106_106653


namespace general_term_formula_l106_106455

noncomputable def Sn (n : ℕ) (a : ℝ) : ℝ := 3^n + a
noncomputable def an (n : ℕ) : ℝ := 2 * 3^(n-1)

theorem general_term_formula {a : ℝ} (n : ℕ) (h : Sn n a = 3^n + a) :
  Sn n a - Sn (n-1) a = an n :=
sorry

end general_term_formula_l106_106455


namespace expected_interval_is_correct_l106_106209

-- Define the travel times via northern and southern routes
def travel_time_north : ℝ := 17
def travel_time_south : ℝ := 11

-- Define the average time difference between train arrivals
noncomputable def avg_time_diff : ℝ := 1.25

-- The average time difference for traveling from home to work versus work to home
noncomputable def time_diff_home_to_work : ℝ := 1

-- Define the expected interval between trains
noncomputable def expected_interval_between_trains := 3

-- Proof problem statement
theorem expected_interval_is_correct :
  ∃ (T : ℝ), (T = expected_interval_between_trains)
  → (travel_time_north - travel_time_south + 2 * avg_time_diff = time_diff_home_to_work)
  → (T = 3) := 
by
  use 3 
  intro h1 h2
  sorry

end expected_interval_is_correct_l106_106209


namespace find_number_l106_106001

theorem find_number (x : ℕ) (h : 3 * x = 33) : x = 11 :=
sorry

end find_number_l106_106001


namespace flour_for_each_cupcake_l106_106067

noncomputable def flour_per_cupcake (total_flour remaining_flour cake_flour_per_cake cake_price cupcake_price total_revenue num_cakes num_cupcakes : ℝ) : ℝ :=
  remaining_flour / num_cupcakes

theorem flour_for_each_cupcake :
  ∀ (total_flour remaining_flour cake_flour_per_cake cake_price cupcake_price total_revenue num_cakes num_cupcakes : ℝ),
    total_flour = 6 →
    remaining_flour = 2 →
    cake_flour_per_cake = 0.5 →
    cake_price = 2.5 →
    cupcake_price = 1 →
    total_revenue = 30 →
    num_cakes = 4 / 0.5 →
    num_cupcakes = 10 →
    flour_per_cupcake total_flour remaining_flour cake_flour_per_cake cake_price cupcake_price total_revenue num_cakes num_cupcakes = 0.2 :=
by intros; sorry

end flour_for_each_cupcake_l106_106067


namespace galaxy_destruction_probability_l106_106684

theorem galaxy_destruction_probability :
  let m := 45853
  let n := 65536
  m + n = 111389 :=
by
  sorry

end galaxy_destruction_probability_l106_106684


namespace sum_of_faces_edges_vertices_rectangular_prism_l106_106120

theorem sum_of_faces_edges_vertices_rectangular_prism :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 :=
by
  let faces := 6
  let edges := 12
  let vertices := 8
  sorry

end sum_of_faces_edges_vertices_rectangular_prism_l106_106120


namespace geom_seq_proof_l106_106938

noncomputable def geom_seq (a q : ℝ) (n : ℕ) : ℝ :=
  a * q^(n - 1)

variables {a q : ℝ}

theorem geom_seq_proof (h1 : geom_seq a q 7 = 4) (h2 : geom_seq a q 5 + geom_seq a q 9 = 10) :
  geom_seq a q 3 + geom_seq a q 11 = 17 :=
by
  sorry

end geom_seq_proof_l106_106938


namespace minimum_value_of_quadratic_l106_106274

theorem minimum_value_of_quadratic (x : ℝ) : ∃ (y : ℝ), (∀ x : ℝ, y ≤ x^2 + 2) ∧ (y = 2) :=
by
  sorry

end minimum_value_of_quadratic_l106_106274


namespace Diamond_evaluation_l106_106405

-- Redefine the operation Diamond
def Diamond (a b : ℕ) : ℕ := a * b^3 - b^2 + 1

-- Statement of the proof
theorem Diamond_evaluation : (Diamond 3 2) = 21 := by
  sorry

end Diamond_evaluation_l106_106405


namespace total_number_of_cookies_l106_106613

open Nat -- Open the natural numbers namespace to work with natural number operations

def n_bags : Nat := 7
def cookies_per_bag : Nat := 2
def total_cookies : Nat := n_bags * cookies_per_bag

theorem total_number_of_cookies : total_cookies = 14 := by
  sorry

end total_number_of_cookies_l106_106613


namespace value_of_expression_l106_106451

theorem value_of_expression : 1 + 2 + 3 - 4 + 5 + 6 + 7 - 8 + 9 + 10 + 11 - 12 = 30 :=
by
  sorry

end value_of_expression_l106_106451


namespace total_short_trees_after_planting_l106_106457

def initial_short_trees : ℕ := 31
def planted_short_trees : ℕ := 64

theorem total_short_trees_after_planting : initial_short_trees + planted_short_trees = 95 := by
  sorry

end total_short_trees_after_planting_l106_106457


namespace find_b_plus_m_l106_106284

def line1 (m : ℝ) (x : ℝ) : ℝ := m * x + 7
def line2 (b : ℝ) (x : ℝ) : ℝ := 4 * x + b

theorem find_b_plus_m :
  ∃ (m b : ℝ), line1 m 8 = 11 ∧ line2 b 8 = 11 ∧ b + m = -20.5 :=
sorry

end find_b_plus_m_l106_106284


namespace f_eq_f_inv_l106_106542

noncomputable def f (x : ℝ) : ℝ := 3 * x - 7

noncomputable def f_inv (x : ℝ) : ℝ := (x + 7) / 3

theorem f_eq_f_inv (x : ℝ) : f x = f_inv x ↔ x = 3.5 := by
  sorry

end f_eq_f_inv_l106_106542


namespace balloon_total_l106_106723

def total_balloons (joan_balloons melanie_balloons : ℕ) : ℕ :=
  joan_balloons + melanie_balloons

theorem balloon_total :
  total_balloons 40 41 = 81 :=
by
  sorry

end balloon_total_l106_106723


namespace compare_y1_y2_l106_106666

theorem compare_y1_y2 (m y1 y2 : ℝ) 
  (h1 : y1 = (-1)^2 - 2*(-1) + m) 
  (h2 : y2 = 2^2 - 2*2 + m) : 
  y1 > y2 := 
sorry

end compare_y1_y2_l106_106666


namespace expression_f_range_a_l106_106942

noncomputable def f (x : ℝ) : ℝ :=
if h : -1 ≤ x ∧ x ≤ 1 then x^3
else if h : 1 ≤ x ∧ x < 3 then -(x-2)^3
else (x-4)^3

theorem expression_f (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 5) :
  f x =
    if h : 1 ≤ x ∧ x < 3 then -(x-2)^3
    else (x-4)^3 :=
by sorry

theorem range_a (a : ℝ) : 
  (∃ x, f x > a) ↔ a < 1 :=
by sorry

end expression_f_range_a_l106_106942


namespace exactly_one_passes_l106_106294

theorem exactly_one_passes (P_A P_B : ℚ) (hA : P_A = 3 / 5) (hB : P_B = 1 / 3) : 
  (1 - P_A) * P_B + P_A * (1 - P_B) = 8 / 15 :=
by
  -- skipping the proof as per requirement
  sorry

end exactly_one_passes_l106_106294


namespace range_of_m_l106_106626

noncomputable def f (x : ℝ) : ℝ := 2^x + Real.log x / Real.log 2

theorem range_of_m :
  (∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → 2^x - Real.log x / Real.log (1/2) + m ≤ 0) →
  m ≤ -5 :=
sorry

end range_of_m_l106_106626


namespace min_holiday_days_l106_106427

theorem min_holiday_days 
  (rained_days : ℕ) 
  (sunny_mornings : ℕ)
  (sunny_afternoons : ℕ) 
  (condition1 : rained_days = 7) 
  (condition2 : sunny_mornings = 5) 
  (condition3 : sunny_afternoons = 6) :
  ∃ (days : ℕ), days = 9 :=
by
  -- The specific steps of the proof are omitted as per the instructions
  sorry

end min_holiday_days_l106_106427


namespace measure_of_one_interior_angle_of_regular_octagon_l106_106961

-- Define the conditions for the problem
def num_sides : ℕ := 8
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Statement of the problem
theorem measure_of_one_interior_angle_of_regular_octagon : 
  (sum_of_interior_angles num_sides) / num_sides = 135 :=
by
  -- Conditions are defined above
  -- Proof steps are omitted
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l106_106961


namespace coconut_grove_l106_106898

theorem coconut_grove (x Y : ℕ) (h1 : 3 * x ≠ 0) (h2 : (x+3) * 60 + x * Y + (x-3) * 180 = 3 * x * 100) (hx : x = 6) : Y = 120 :=
by 
  sorry

end coconut_grove_l106_106898


namespace length_of_each_piece_l106_106208

theorem length_of_each_piece (rod_length : ℝ) (num_pieces : ℕ) (h₁ : rod_length = 42.5) (h₂ : num_pieces = 50) : (rod_length / num_pieces * 100) = 85 := 
by 
  sorry

end length_of_each_piece_l106_106208


namespace shaded_square_area_l106_106445

noncomputable def Pythagorean_area (a b c : ℕ) (area_a area_b area_c : ℕ) : Prop :=
  area_a = a^2 ∧ area_b = b^2 ∧ area_c = c^2 ∧ a^2 + b^2 = c^2

theorem shaded_square_area 
  (area1 area2 area3 : ℕ)
  (area_unmarked : ℕ)
  (h1 : area1 = 5)
  (h2 : area2 = 8)
  (h3 : area3 = 32)
  (h_unmarked: area_unmarked = area2 + area3)
  (h_shaded : area1 + area_unmarked = 45) :
  area1 + area_unmarked = 45 :=
by
  exact h_shaded

end shaded_square_area_l106_106445


namespace find_circle_equation_l106_106921

-- Define the conditions and problem
def circle_standard_equation (p1 p2 : ℝ × ℝ) (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (xc, yc) := center
  (x2 - xc)^2 + (y2 - yc)^2 = radius^2

-- Define the conditions as given in the problem
def point_on_circle : Prop := circle_standard_equation (2, 0) (2, 2) (2, 2) 2

-- The main theorem to prove that the standard equation of the circle holds
theorem find_circle_equation : 
  point_on_circle →
  ∃ h k r, h = 2 ∧ k = 2 ∧ r = 2 ∧ (x - h)^2 + (y - k)^2 = r^2 :=
by
  sorry

end find_circle_equation_l106_106921


namespace initial_mixture_volume_l106_106459

theorem initial_mixture_volume (x : ℝ) (hx1 : 0.10 * x + 10 = 0.28 * (x + 10)) : x = 40 :=
by
  sorry

end initial_mixture_volume_l106_106459


namespace remainder_of_largest_divided_by_next_largest_l106_106721

/-
  Conditions:
  Let a = 10, b = 11, c = 12, d = 13.
  The largest number is d (13) and the next largest number is c (12).

  Question:
  What is the remainder when the largest number is divided by the next largest number?

  Answer:
  The remainder is 1.
-/

theorem remainder_of_largest_divided_by_next_largest :
  let a := 10 
  let b := 11
  let c := 12
  let d := 13
  d % c = 1 :=
by
  sorry

end remainder_of_largest_divided_by_next_largest_l106_106721


namespace regular_polygon_sides_l106_106334

theorem regular_polygon_sides (n : ℕ) (h1 : ∃ a : ℝ, a = 120 ∧ ∀ i < n, 120 = a) : n = 6 :=
by
  sorry

end regular_polygon_sides_l106_106334


namespace time_to_coffee_shop_is_18_l106_106379

variable (cycle_constant_pace : Prop)
variable (time_cycle_library : ℕ)
variable (distance_cycle_library : ℕ)
variable (distance_to_coffee_shop : ℕ)

theorem time_to_coffee_shop_is_18
  (h_const_pace : cycle_constant_pace)
  (h_time_library : time_cycle_library = 30)
  (h_distance_library : distance_cycle_library = 5)
  (h_distance_coffee : distance_to_coffee_shop = 3)
  : (30 / 5) * 3 = 18 :=
by
  sorry

end time_to_coffee_shop_is_18_l106_106379


namespace rational_eq_reciprocal_l106_106007

theorem rational_eq_reciprocal (x : ℚ) (h : x = 1 / x) : x = 1 ∨ x = -1 :=
by {
  sorry
}

end rational_eq_reciprocal_l106_106007


namespace tangent_fraction_15_degrees_l106_106635

theorem tangent_fraction_15_degrees : (1 + Real.tan (Real.pi / 12 )) / (1 - Real.tan (Real.pi / 12)) = Real.sqrt 3 :=
by
  sorry

end tangent_fraction_15_degrees_l106_106635


namespace max_area_triangle_l106_106343

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

noncomputable def line_eq (x y : ℝ) : Prop := 2 * Real.sqrt 2 * x - y - 1 = 0

theorem max_area_triangle (x1 y1 x2 y2 xp yp : ℝ) (h1 : circle_eq x1 y1) (h2 : circle_eq x2 y2) (h3 : circle_eq xp yp)
  (h4 : line_eq x1 y1) (h5 : line_eq x2 y2) (h6 : (xp, yp) ≠ (x1, y1)) (h7 : (xp, yp) ≠ (x2, y2)) :
  ∃ S : ℝ, S = 10 * Real.sqrt 5 / 9 :=
by
  sorry

end max_area_triangle_l106_106343


namespace average_of_r_s_t_l106_106729

theorem average_of_r_s_t (r s t : ℝ) (h : (5/4) * (r + s + t) = 20) : (r + s + t) / 3 = 16 / 3 :=
by
  sorry

end average_of_r_s_t_l106_106729


namespace problem_1_problem_2_l106_106216

variable {a : ℕ → ℝ}
variable (n : ℕ)

-- Conditions of the problem
def seq_positive : ∀ (k : ℕ), a k > 0 := sorry
def a1 : a 1 = 1 := sorry
def recurrence (n : ℕ) : a (n + 1) = (a n + 1) / (12 * a n) := sorry

-- Proofs to be provided
theorem problem_1 : ∀ n : ℕ, a (2 * n + 1) < a (2 * n - 1) := 
by 
  apply sorry 

theorem problem_2 : ∀ n : ℕ, 1 / 6 ≤ a n ∧ a n ≤ 1 := 
by 
  apply sorry 

end problem_1_problem_2_l106_106216


namespace white_truck_chance_l106_106293

-- Definitions from conditions
def trucks : ℕ := 50
def cars : ℕ := 40
def vans : ℕ := 30

def red_trucks : ℕ := 50 / 2
def black_trucks : ℕ := (20 * 50) / 100

-- The remaining percentage (30%) of trucks is assumed to be white.
def white_trucks : ℕ := (30 * 50) / 100

def total_vehicles : ℕ := trucks + cars + vans

-- Given
def percentage_white_truck : ℕ := (white_trucks * 100) / total_vehicles

-- Theorem that proves the problem statement
theorem white_truck_chance : percentage_white_truck = 13 := 
by
  -- Proof will be written here (currently stubbed)
  sorry

end white_truck_chance_l106_106293


namespace distance_between_Sasha_and_Koyla_is_19m_l106_106720

-- Defining variables for speeds
variables (v_S v_L v_K : ℝ)
-- Additional conditions
variables (h1 : ∃ (t : ℝ), t > 0 ∧ 100 = v_S * t) -- Sasha finishes the race in time t
variables (h2 : 90 = v_L * (100 / v_S)) -- Lyosha is 10 meters behind when Sasha finishes
variables (h3 : v_K = 0.9 * v_L) -- Kolya's speed is 0.9 times Lyosha's speed

theorem distance_between_Sasha_and_Koyla_is_19m :
  ∀ (v_S v_L v_K : ℝ), (h1 : ∃ t > 0, 100 = v_S * t) → (h2 : 90 = v_L * (100 / v_S)) → (h3 : v_K = 0.9 * v_L)  →
  (100 - (0.81 * 100)) = 19 :=
by
  intros v_S v_L v_K h1 h2 h3
  sorry

end distance_between_Sasha_and_Koyla_is_19m_l106_106720


namespace value_at_zero_eq_sixteen_l106_106945

-- Define the polynomial P(x)
def P (x : ℚ) : ℚ := x ^ 4 - 20 * x ^ 2 + 16

-- Theorem stating the value of P(0)
theorem value_at_zero_eq_sixteen :
  P 0 = 16 :=
by
-- We know the polynomial P(x) is x^4 - 20x^2 + 16
-- When x = 0, P(0) = 0^4 - 20 * 0^2 + 16 = 16
sorry

end value_at_zero_eq_sixteen_l106_106945


namespace man_distance_from_start_l106_106119

noncomputable def distance_from_start (west_distance north_distance : ℝ) : ℝ :=
  Real.sqrt (west_distance^2 + north_distance^2)

theorem man_distance_from_start :
  distance_from_start 10 10 = Real.sqrt 200 :=
by
  sorry

end man_distance_from_start_l106_106119


namespace rabbit_wins_race_l106_106835

theorem rabbit_wins_race :
  ∀ (rabbit_speed1 rabbit_speed2 snail_speed rest_time total_distance : ℕ)
  (rabbit_time1 rabbit_time2 : ℚ),
  rabbit_speed1 = 20 →
  rabbit_speed2 = 30 →
  snail_speed = 2 →
  rest_time = 3 →
  total_distance = 100 →
  rabbit_time1 = (30 : ℚ) / rabbit_speed1 →
  rabbit_time2 = (70 : ℚ) / rabbit_speed2 →
  (rabbit_time1 + rest_time + rabbit_time2 < total_distance / snail_speed) :=
by
  intros
  sorry

end rabbit_wins_race_l106_106835


namespace primes_pos_int_solutions_l106_106182

theorem primes_pos_int_solutions 
  (p : ℕ) [hp : Fact (Nat.Prime p)] (a b : ℕ) (h1 : ∃ k : ℤ, (4 * a + p : ℤ) + k * (4 * b + p : ℤ) = b * k * a)
  (h2 : ∃ m : ℤ, (a^2 : ℤ) + m * (b^2 : ℤ) = b * m * a) : a = b ∨ a = b * p :=
  sorry

end primes_pos_int_solutions_l106_106182


namespace percent_fair_hair_l106_106236

theorem percent_fair_hair (total_employees : ℕ) (total_women_fair_hair : ℕ)
  (percent_fair_haired_women : ℕ) (percent_women_fair_hair : ℕ)
  (h1 : total_women_fair_hair = (total_employees * percent_women_fair_hair) / 100)
  (h2 : percent_fair_haired_women * total_women_fair_hair = total_employees * 10) :
  (25 * total_employees = 100 * total_women_fair_hair) :=
by {
  sorry
}

end percent_fair_hair_l106_106236


namespace top_angle_is_70_l106_106550

theorem top_angle_is_70
  (sum_angles : ℝ)
  (left_angle : ℝ)
  (right_angle : ℝ)
  (top_angle : ℝ)
  (h1 : sum_angles = 250)
  (h2 : left_angle = 2 * right_angle)
  (h3 : right_angle = 60)
  (h4 : sum_angles = left_angle + right_angle + top_angle) :
  top_angle = 70 :=
by
  sorry

end top_angle_is_70_l106_106550


namespace polynomial_is_perfect_cube_l106_106578

theorem polynomial_is_perfect_cube (p q n : ℚ) :
  (∃ a : ℚ, x^3 + p * x^2 + q * x + n = (x + a)^3) ↔ (q = p^2 / 3 ∧ n = p^3 / 27) :=
by sorry

end polynomial_is_perfect_cube_l106_106578


namespace result_of_4_times_3_l106_106177

def operation (a b : ℕ) : ℕ :=
  a^2 + a * Nat.factorial b - b^2

theorem result_of_4_times_3 : operation 4 3 = 31 := by
  sorry

end result_of_4_times_3_l106_106177


namespace increment_in_displacement_l106_106388

variable (d : ℝ)

def equation_of_motion (t : ℝ) : ℝ := 2 * t^2

theorem increment_in_displacement:
  let t1 := 2
  let t2 := 2 + d
  let s1 := equation_of_motion t1
  let s2 := equation_of_motion t2
  s2 - s1 = 8 * d + 2 * d^2 := by
  sorry

end increment_in_displacement_l106_106388


namespace study_tour_buses_l106_106058

variable (x : ℕ) (num_people : ℕ)

def seats_A := 45
def seats_B := 60
def extra_people := 30
def fewer_B := 6

theorem study_tour_buses (h : seats_A * x + extra_people = seats_B * (x - fewer_B)) : 
  x = 26 ∧ (seats_A * 26 + extra_people = 1200) := 
  sorry

end study_tour_buses_l106_106058


namespace total_pieces_of_gum_l106_106527

def packages : ℕ := 12
def pieces_per_package : ℕ := 20

theorem total_pieces_of_gum : packages * pieces_per_package = 240 :=
by
  -- proof is skipped
  sorry

end total_pieces_of_gum_l106_106527


namespace competition_score_difference_l106_106995

theorem competition_score_difference :
  let perc_60 := 0.20
  let perc_75 := 0.25
  let perc_85 := 0.15
  let perc_90 := 0.30
  let perc_95 := 0.10
  let mean := (perc_60 * 60) + (perc_75 * 75) + (perc_85 * 85) + (perc_90 * 90) + (perc_95 * 95)
  let median := 85
  (median - mean = 5) := by
sorry

end competition_score_difference_l106_106995


namespace order_of_abc_l106_106991

noncomputable def a : ℝ := Real.log 3 / Real.log 4
noncomputable def b : ℝ := Real.log 4 / Real.log 3
noncomputable def c : ℝ := Real.log (4/3) / Real.log (3/4)

theorem order_of_abc : b > a ∧ a > c := by
  sorry

end order_of_abc_l106_106991


namespace garden_area_increase_l106_106847

theorem garden_area_increase :
  let length_rect := 60
  let width_rect := 20
  let area_rect := length_rect * width_rect
  
  let perimeter := 2 * (length_rect + width_rect)
  
  let side_square := perimeter / 4
  let area_square := side_square * side_square

  area_square - area_rect = 400 := by
    sorry

end garden_area_increase_l106_106847


namespace a_squared_divisible_by_b_l106_106183

theorem a_squared_divisible_by_b (a b : ℕ) (h1 : a < 1000) (h2 : b > 0) 
    (h3 : ∃ k, a ^ 21 = b ^ 10 * k) : ∃ m, a ^ 2 = b * m := 
by
  sorry

end a_squared_divisible_by_b_l106_106183


namespace sequence_bounds_l106_106805

theorem sequence_bounds :
    ∀ (a : ℕ → ℝ), a 0 = 5 → (∀ n : ℕ, a (n + 1) = a n + 1 / a n) → 45 < a 1000 ∧ a 1000 < 45.1 :=
by
  intros a h0 h_rec
  sorry

end sequence_bounds_l106_106805


namespace find_f_neg2_l106_106421

noncomputable def f (x : ℝ) : ℝ := -2 * (x + 1) + 1

theorem find_f_neg2 : f (-2) = 3 := by
  sorry

end find_f_neg2_l106_106421


namespace translated_parabola_eq_l106_106298

-- Define the original parabola
def orig_parabola (x : ℝ) : ℝ := -2 * x^2

-- Define the translation function
def translate_upwards (f : ℝ → ℝ) (dy : ℝ) : (ℝ → ℝ) :=
  fun x => f x + dy

-- Define the translated parabola
def translated_parabola := translate_upwards orig_parabola 3

-- State the theorem
theorem translated_parabola_eq:
  translated_parabola = (fun x : ℝ => -2 * x^2 + 3) :=
by
  sorry

end translated_parabola_eq_l106_106298


namespace part1_part2_part3_l106_106541

noncomputable def p1_cost (t : ℕ) : ℕ := 
  if t <= 150 then 58 else 58 + 25 * (t - 150) / 100

noncomputable def p2_cost (t : ℕ) (a : ℕ) : ℕ := 
  if t <= 350 then 88 else 88 + a * (t - 350)

-- Part 1: Prove the costs for 260 minutes
theorem part1 : p1_cost 260 = 855 / 10 ∧ p2_cost 260 30 = 88 :=
by 
  sorry

-- Part 2: Prove the existence of t for given a
theorem part2 (t : ℕ) : (a = 30) → (∃ t, p1_cost t = p2_cost t a) :=
by 
  sorry

-- Part 3: Prove a=45 and the range for which Plan 1 is cheaper
theorem part3 : 
  (a = 45) ↔ (p1_cost 450 = p2_cost 450 a) ∧ (∀ t, (0 ≤ t ∧ t < 270) ∨ (t > 450) → p1_cost t < p2_cost t 45 ) :=
by 
  sorry

end part1_part2_part3_l106_106541


namespace culture_growth_l106_106125

/-- Define the initial conditions and growth rates of the bacterial culture -/
def initial_cells : ℕ := 5

def growth_rate1 : ℕ := 3
def growth_rate2 : ℕ := 2

def cycle_duration : ℕ := 3
def first_phase_duration : ℕ := 6
def second_phase_duration : ℕ := 6

def total_duration : ℕ := 12

/-- Define the hypothesis that calculates the number of cells at any point in time based on the given rules -/
theorem culture_growth : 
    (initial_cells * growth_rate1^ (first_phase_duration / cycle_duration) 
    * growth_rate2^ (second_phase_duration / cycle_duration)) = 180 := 
sorry

end culture_growth_l106_106125


namespace parabola_p_q_r_sum_l106_106998

noncomputable def parabola_vertex (p q r : ℝ) (x_vertex y_vertex : ℝ) :=
  ∀ (x : ℝ), p * (x - x_vertex) ^ 2 + y_vertex = p * x ^ 2 + q * x + r

theorem parabola_p_q_r_sum
  (p q r : ℝ)
  (vertex_x vertex_y : ℝ)
  (hx_vertex : vertex_x = 3)
  (hy_vertex : vertex_y = 10)
  (h_vertex : parabola_vertex p q r vertex_x vertex_y)
  (h_contains : p * (0 - 3) ^ 2 + 10 = 7) :
  p + q + r = 23 / 3 :=
sorry

end parabola_p_q_r_sum_l106_106998


namespace ordinary_eq_from_param_eq_l106_106485

theorem ordinary_eq_from_param_eq (α : ℝ) :
  (∃ (x y : ℝ), x = 3 * Real.cos α + 1 ∧ y = - Real.cos α → x + 3 * y - 1 = 0 ∧ (-2 ≤ x ∧ x ≤ 4)) := 
sorry

end ordinary_eq_from_param_eq_l106_106485


namespace first_bakery_sacks_per_week_l106_106045

theorem first_bakery_sacks_per_week (x : ℕ) 
    (H1 : 4 * x + 4 * 4 + 4 * 12 = 72) : x = 2 :=
by 
  -- we will provide the proof here if needed
  sorry

end first_bakery_sacks_per_week_l106_106045


namespace segment_length_l106_106009
noncomputable def cube_root27 : ℝ := 3

theorem segment_length : ∀ (x : ℝ), (|x - cube_root27| = 4) → ∃ (a b : ℝ), (a = cube_root27 + 4) ∧ (b = cube_root27 - 4) ∧ |a - b| = 8 :=
by
  sorry

end segment_length_l106_106009


namespace largest_divisor_of_Pn_for_even_n_l106_106697

def P (n : ℕ) : ℕ := 
  (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9)

theorem largest_divisor_of_Pn_for_even_n : 
  ∀ (n : ℕ), (0 < n ∧ n % 2 = 0) → ∃ d, d = 15 ∧ d ∣ P n :=
by
  intro n h
  sorry

end largest_divisor_of_Pn_for_even_n_l106_106697


namespace daryl_max_crate_weight_l106_106975

variable (crates : ℕ) (weight_nails : ℕ) (bags_nails : ℕ)
variable (weight_hammers : ℕ) (bags_hammers : ℕ) (weight_planks : ℕ)
variable (bags_planks : ℕ) (weight_left_out : ℕ)

def max_weight_per_crate (total_weight: ℕ) (total_crates: ℕ) : ℕ :=
  total_weight / total_crates

-- State the problem in Lean
theorem daryl_max_crate_weight
  (h1 : crates = 15) 
  (h2 : bags_nails = 4) 
  (h3 : weight_nails = 5)
  (h4 : bags_hammers = 12) 
  (h5 : weight_hammers = 5) 
  (h6 : bags_planks = 10) 
  (h7 : weight_planks = 30) 
  (h8 : weight_left_out = 80):
  max_weight_per_crate ((bags_nails * weight_nails + bags_hammers * weight_hammers + bags_planks * weight_planks) - weight_left_out) crates = 20 :=
  by sorry

end daryl_max_crate_weight_l106_106975


namespace general_term_formaula_sum_of_seq_b_l106_106672

noncomputable def seq_a (n : ℕ) := 2 * n + 1

noncomputable def seq_b (n : ℕ) := 1 / ((seq_a n)^2 - 1)

noncomputable def sum_seq_a (n : ℕ) := (Finset.range n).sum seq_a

noncomputable def sum_seq_b (n : ℕ) := (Finset.range n).sum seq_b

theorem general_term_formaula (n : ℕ) (h : n > 0) :
  (∀ n, (seq_a n) > 0) ∧ (∀ n, (seq_a n)^2 + 2 * (seq_a n) = 4 * (sum_seq_a n) + 3) →
  seq_a n = 2 * n + 1 :=
by
  intros
  sorry

theorem sum_of_seq_b (n : ℕ) (h : n > 0) :
  (∀ n, (seq_a n) > 0) ∧ (∀ n, (seq_a n)^2 + 2 * (seq_a n) = 4 * (sum_seq_a n) + 3) →
  sum_seq_b n = n / (4 * (n + 1)) :=
by
  intros
  sorry

end general_term_formaula_sum_of_seq_b_l106_106672


namespace paul_prays_more_than_bruce_l106_106965

-- Conditions as definitions in Lean 4
def prayers_per_day_paul := 20
def prayers_per_sunday_paul := 2 * prayers_per_day_paul
def prayers_per_day_bruce := prayers_per_day_paul / 2
def prayers_per_sunday_bruce := 2 * prayers_per_sunday_paul

def weekly_prayers_paul := 6 * prayers_per_day_paul + prayers_per_sunday_paul
def weekly_prayers_bruce := 6 * prayers_per_day_bruce + prayers_per_sunday_bruce

-- Statement of the proof problem
theorem paul_prays_more_than_bruce :
  (weekly_prayers_paul - weekly_prayers_bruce) = 20 := by
  sorry

end paul_prays_more_than_bruce_l106_106965


namespace problem_condition_neither_sufficient_nor_necessary_l106_106988

theorem problem_condition_neither_sufficient_nor_necessary 
  (m n : ℕ) (hm : m > 0) (hn : n > 0) (a b : ℝ) :
  (a > b → a^(m + n) + b^(m + n) > a^n * b^m + a^m * b^n) ∧
  (a^(m + n) + b^(m + n) > a^n * b^m + a^m * b^n → a > b) = false :=
by sorry

end problem_condition_neither_sufficient_nor_necessary_l106_106988


namespace axis_of_symmetry_l106_106967

-- Define the condition for the parabola equation
def parabola_equation (x y : ℝ) : Prop :=
  x = -4 * y^2

-- Define the statement that needs to be proven
theorem axis_of_symmetry (x : ℝ) (y : ℝ) (h : parabola_equation x y) : x = 1 / 16 :=
  sorry

end axis_of_symmetry_l106_106967


namespace average_percent_score_is_77_l106_106220

def numberOfStudents : ℕ := 100

def percentage_counts : List (ℕ × ℕ) :=
[(100, 7), (90, 18), (80, 35), (70, 25), (60, 10), (50, 3), (40, 2)]

noncomputable def average_score (counts : List (ℕ × ℕ)) : ℚ :=
  (counts.foldl (λ acc p => acc + (p.1 * p.2)) 0 : ℚ) / numberOfStudents

theorem average_percent_score_is_77 : average_score percentage_counts = 77 := by
  sorry

end average_percent_score_is_77_l106_106220


namespace total_number_of_students_l106_106755

-- Statement translating the problem conditions and conclusion
theorem total_number_of_students (rank_from_right rank_from_left total : ℕ) 
  (h_right : rank_from_right = 13) 
  (h_left : rank_from_left = 8) 
  (total_eq : total = rank_from_right + rank_from_left - 1) : 
  total = 20 := 
by 
  -- Proof is skipped
  sorry

end total_number_of_students_l106_106755


namespace original_price_second_store_l106_106932

-- Definitions of the conditions
def price_first_store : ℝ := 950
def discount_first_store : ℝ := 0.06
def discount_second_store : ℝ := 0.05
def price_difference : ℝ := 19

-- Define the discounted price function
def discounted_price (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

-- State the main theorem
theorem original_price_second_store :
  ∃ P : ℝ, 
    (discounted_price price_first_store discount_first_store - discounted_price P discount_second_store = price_difference) ∧ 
    P = 960 :=
by
  sorry

end original_price_second_store_l106_106932


namespace even_iff_a_zero_max_value_f_l106_106227

noncomputable def f (x a : ℝ) : ℝ := -x^2 + |x - a| + a + 1

theorem even_iff_a_zero (a : ℝ) : (∀ x, f x a = f (-x) a) ↔ a = 0 :=
by {
  -- Proof is omitted
  sorry
}

theorem max_value_f (a : ℝ) : 
  ∃ max_val : ℝ, 
    ( 
      (-1/2 < a ∧ a ≤ 0 ∧ max_val = 5/4) ∨ 
      (0 < a ∧ a < 1/2 ∧ max_val = 5/4 + 2*a) ∨ 
      ((a ≤ -1/2 ∨ a ≥ 1/2) ∧ max_val = -a^2 + a + 1)
    ) :=
by {
  -- Proof is omitted
  sorry
}

end even_iff_a_zero_max_value_f_l106_106227


namespace necessary_but_not_sufficient_l106_106521

def p (x : ℝ) : Prop := x ^ 2 = 3 * x + 4
def q (x : ℝ) : Prop := x = Real.sqrt (3 * x + 4)

theorem necessary_but_not_sufficient (x : ℝ) : (p x → q x) ∧ ¬ (q x → p x) := by
  sorry

end necessary_but_not_sufficient_l106_106521


namespace geometric_sequence_b_value_l106_106732

theorem geometric_sequence_b_value (r b : ℝ) (h1 : 120 * r = b) (h2 : b * r = 27 / 16) (hb_pos : b > 0) : b = 15 :=
sorry

end geometric_sequence_b_value_l106_106732


namespace length_of_AB_l106_106818

-- Define the distances given as conditions
def AC : ℝ := 5
def BD : ℝ := 6
def CD : ℝ := 3

-- Define the linear relationship of points A, B, C, D on the line
def points_on_line_in_order := true -- This is just a placeholder

-- Main theorem to prove
theorem length_of_AB : AB = 2 :=
by
  -- Apply the conditions and the linear relationships
  have BC : ℝ := BD - CD
  have AB : ℝ := AC - BC
  -- This would contain the actual proof using steps, but we skip it here
  sorry

end length_of_AB_l106_106818


namespace three_hundred_thousand_times_three_hundred_thousand_minus_one_million_l106_106314

theorem three_hundred_thousand_times_three_hundred_thousand_minus_one_million :
  (300000 * 300000) - 1000000 = 89990000000 := by
  sorry 

end three_hundred_thousand_times_three_hundred_thousand_minus_one_million_l106_106314


namespace repeating_decimal_to_fraction_l106_106586

noncomputable def repeating_decimal_sum (x y z : ℚ) : ℚ := x + y + z

theorem repeating_decimal_to_fraction :
  let x := 4 / 33
  let y := 34 / 999
  let z := 567 / 99999
  repeating_decimal_sum x y z = 134255 / 32929667 := by
  -- proofs are omitted
  sorry

end repeating_decimal_to_fraction_l106_106586


namespace alpha_plus_beta_l106_106886

theorem alpha_plus_beta (α β : ℝ) 
  (hα : 0 < α ∧ α < Real.pi / 2) 
  (hβ : 0 < β ∧ β < Real.pi / 2)
  (h_sin_alpha : Real.sin α = Real.sqrt 10 / 10)
  (h_cos_beta : Real.cos β = 2 * Real.sqrt 5 / 5) :
  α + β = Real.pi / 4 :=
sorry

end alpha_plus_beta_l106_106886


namespace parallel_lines_parallel_lines_solution_l106_106143

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 6 = 0) ∧ (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0) → a = -1 ∨ a = 2 :=
sorry

theorem parallel_lines_solution (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 6 = 0) ∧ (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0) ∧ 
  ((a = -1 → ∀ x y : ℝ, a * x + 2 * y + 6 = 0 ∧ (x + (a - 1) * y + (a^2 - 1) = 0)) ∨ 
  (a = 2 → ∀ x y : ℝ, a * x + 2 * y + 6 = 0 ∧ (x + (a - 1) * y + (a^2 - 1) = 0))) :=
sorry

end parallel_lines_parallel_lines_solution_l106_106143


namespace hazel_sold_18_cups_to_kids_l106_106797

theorem hazel_sold_18_cups_to_kids:
  ∀ (total_cups cups_sold_construction crew_remaining cups_sold_kids cups_given_away last_cup: ℕ),
     total_cups = 56 →
     cups_sold_construction = 28 →
     crew_remaining = total_cups - cups_sold_construction →
     last_cup = 1 →
     crew_remaining = cups_sold_kids + (cups_sold_kids / 2) + last_cup →
     cups_sold_kids = 18 :=
by
  intros total_cups cups_sold_construction crew_remaining cups_sold_kids cups_given_away last_cup h_total h_construction h_remaining h_last h_equation
  sorry

end hazel_sold_18_cups_to_kids_l106_106797


namespace washer_and_dryer_proof_l106_106061

noncomputable def washer_and_dryer_problem : Prop :=
  ∃ (price_of_washer price_of_dryer : ℕ),
    price_of_washer + price_of_dryer = 600 ∧
    (∃ (k : ℕ), price_of_washer = k * price_of_dryer) ∧
    price_of_dryer = 150 ∧
    price_of_washer / price_of_dryer = 3

theorem washer_and_dryer_proof : washer_and_dryer_problem :=
sorry

end washer_and_dryer_proof_l106_106061


namespace percentage_earth_fresh_water_l106_106110

theorem percentage_earth_fresh_water :
  let portion_land := 3 / 10
  let portion_water := 1 - portion_land
  let percent_salt_water := 97 / 100
  let percent_fresh_water := 1 - percent_salt_water
  100 * (portion_water * percent_fresh_water) = 2.1 :=
by
  sorry

end percentage_earth_fresh_water_l106_106110


namespace Thabo_harcdover_nonfiction_books_l106_106175

theorem Thabo_harcdover_nonfiction_books 
  (H P F : ℕ)
  (h1 : P = H + 20)
  (h2 : F = 2 * P)
  (h3 : H + P + F = 180) : 
  H = 30 :=
by
  sorry

end Thabo_harcdover_nonfiction_books_l106_106175


namespace smallest_q_difference_l106_106939

theorem smallest_q_difference (p q : ℕ) (hp_pos : 0 < p) (hq_pos : 0 < q)
  (h_fraction1 : 3 * q < 5 * p)
  (h_fraction2 : 5 * p < 6 * q)
  (h_smallest : ∀ r s : ℕ, 0 < s → 3 * s < 5 * r → 5 * r < 6 * s → q ≤ s) :
  q - p = 3 :=
by
  sorry

end smallest_q_difference_l106_106939


namespace farmer_cows_more_than_goats_l106_106782

-- Definitions of the variables
variables (C P G x : ℕ)

-- Conditions given in the problem
def twice_as_many_pigs_as_cows : Prop := P = 2 * C
def more_cows_than_goats : Prop := C = G + x
def goats_count : Prop := G = 11
def total_animals : Prop := C + P + G = 56

-- The theorem to prove
theorem farmer_cows_more_than_goats
  (h1 : twice_as_many_pigs_as_cows C P)
  (h2 : more_cows_than_goats C G x)
  (h3 : goats_count G)
  (h4 : total_animals C P G) :
  C - G = 4 :=
sorry

end farmer_cows_more_than_goats_l106_106782


namespace partial_fraction_decomposition_l106_106043

theorem partial_fraction_decomposition (x : ℝ) :
  (5 * x - 3) / (x^2 - 5 * x - 14) = (32 / 9) / (x - 7) + (13 / 9) / (x + 2) := by
  sorry

end partial_fraction_decomposition_l106_106043


namespace proposition_false_at_4_l106_106444

open Nat

def prop (n : ℕ) : Prop := sorry -- the actual proposition is not specified, so we use sorry

theorem proposition_false_at_4 :
  (∀ k : ℕ, k > 0 → (prop k → prop (k + 1))) →
  ¬ prop 5 →
  ¬ prop 4 :=
by
  intros h_induction h_proposition_false_at_5
  sorry

end proposition_false_at_4_l106_106444


namespace coplanar_iff_m_eq_neg_8_l106_106823

variable {V : Type} [AddCommGroup V] [Module ℝ V]
variable (A B C D : V)
variable (m : ℝ)

theorem coplanar_iff_m_eq_neg_8 
  (h : 4 • A - 3 • B + 7 • C + m • D = 0) : m = -8 ↔ ∃ a b c d : ℝ, a + b + c + d = 0 ∧ a • A + b • B + c • C + d • D = 0 :=
by
  sorry

end coplanar_iff_m_eq_neg_8_l106_106823


namespace find_t_squared_l106_106992
noncomputable section

-- Definitions of the given conditions
def hyperbola_opens_vertically (x y : ℝ) : Prop :=
  (y^2 / 4 - 5 * x^2 / 16 = 1)

-- Statement of the problem
theorem find_t_squared (t : ℝ) 
  (h1 : hyperbola_opens_vertically 4 (-3))
  (h2 : hyperbola_opens_vertically 0 (-2))
  (h3 : hyperbola_opens_vertically 2 t) : 
  t^2 = 8 := 
sorry -- Proof is omitted, it's just the statement

end find_t_squared_l106_106992


namespace distance_symmetric_parabola_l106_106162

open Real

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def parabola (x : ℝ) : ℝ := 3 - x^2

theorem distance_symmetric_parabola (A B : ℝ × ℝ) 
  (hA : A.2 = parabola A.1) 
  (hB : B.2 = parabola B.1)
  (hSym : A.1 + A.2 = 0 ∧ B.1 + B.2 = 0) 
  (hDistinct : A ≠ B) :
  distance A B = 3 * sqrt 2 :=
by
  sorry

end distance_symmetric_parabola_l106_106162


namespace problem_solution_l106_106028

theorem problem_solution :
  (∀ (p q : ℚ), 
    (∀ (x : ℚ), (x + 3 * p) * (x^2 - x + (1 / 3) * q) = x^3 + (3 * p - 1) * x^2 + ((1 / 3) * q - 3 * p) * x + p * q) →
    (3 * p - 1 = 0) →
    ((1 / 3) * q - 3 * p = 0) →
    p = 1 / 3 ∧ q = 3)
  ∧ ((1 / 3) ^ 2020 * 3 ^ 2021 = 3) :=
by
  sorry

end problem_solution_l106_106028


namespace complement_union_M_N_eq_set_l106_106568

open Set

-- Define the universe U
def U : Set (ℝ × ℝ) := { p | True }

-- Define the set M
def M : Set (ℝ × ℝ) := { p | (p.snd - 3) / (p.fst - 2) ≠ 1 }

-- Define the set N
def N : Set (ℝ × ℝ) := { p | p.snd ≠ p.fst + 1 }

-- Define the complement of M ∪ N in U
def complement_MN : Set (ℝ × ℝ) := compl (M ∪ N)

theorem complement_union_M_N_eq_set : complement_MN = { (2, 3) } :=
  sorry

end complement_union_M_N_eq_set_l106_106568


namespace convert_spherical_to_rectangular_l106_106540

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : (ℝ × ℝ × ℝ) :=
  (ρ * Real.sin φ * Real.cos θ,
   ρ * Real.sin φ * Real.sin θ,
   ρ * Real.cos φ)

theorem convert_spherical_to_rectangular : spherical_to_rectangular 5 (Real.pi / 2) (Real.pi / 3) = 
  (0, 5 * Real.sqrt 3 / 2, 5 / 2) :=
by
  sorry

end convert_spherical_to_rectangular_l106_106540


namespace instantaneous_rate_of_change_at_x1_l106_106332

open Real

noncomputable def f (x : ℝ) : ℝ := (1/3)*x^3 - x^2 + 8

theorem instantaneous_rate_of_change_at_x1 : deriv f 1 = -1 := by
  sorry

end instantaneous_rate_of_change_at_x1_l106_106332


namespace price_ratio_l106_106410

-- Definitions based on the provided conditions
variables (x y : ℕ) -- number of ballpoint pens and gel pens respectively
variables (b g T : ℝ) -- price of ballpoint pen, gel pen, and total amount paid respectively

-- The two given conditions
def cond1 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * g = 4 * (x * b + y * g)

def cond2 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * b = (x * b + y * g) / 2

-- The goal to prove
theorem price_ratio (x y : ℕ) (b g T : ℝ) (h1 : cond1 x y b g T) (h2 : cond2 x y b g T) : 
  g = 8 * b :=
sorry

end price_ratio_l106_106410


namespace cody_marbles_l106_106269

theorem cody_marbles (M : ℕ) (h1 : M / 3 + 5 + 7 = M) : M = 18 :=
by
  have h2 : 3 * M / 3 + 3 * 5 + 3 * 7 = 3 * M := by sorry
  have h3 : 3 * M / 3 = M := by sorry
  have h4 : 3 * 7 = 21 := by sorry
  have h5 : M + 15 + 21 = 3 * M := by sorry
  have h6 : M = 18 := by sorry
  exact h6

end cody_marbles_l106_106269


namespace cubic_sum_identity_l106_106478

theorem cubic_sum_identity
  (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : ab + ac + bc = -3)
  (h3 : abc = 9) :
  a^3 + b^3 + c^3 = 22 :=
by
  sorry

end cubic_sum_identity_l106_106478


namespace fg_at_2_l106_106989

def f (x : ℝ) : ℝ := x^3
def g (x : ℝ) : ℝ := 2*x + 5

theorem fg_at_2 : f (g 2) = 729 := by
  sorry

end fg_at_2_l106_106989


namespace volume_relation_l106_106535

theorem volume_relation 
  (r h : ℝ) 
  (heightC_eq_three_times_radiusD : h = 3 * r)
  (radiusC_eq_heightD : r = h)
  (volumeD_eq_three_times_volumeC : ∀ (π : ℝ), 3 * (π * h^2 * r) = π * r^2 * h) :
  3 = (3 : ℝ) := 
by
  sorry

end volume_relation_l106_106535


namespace fifteen_quarters_twenty_dimes_equal_five_quarters_n_dimes_l106_106597

theorem fifteen_quarters_twenty_dimes_equal_five_quarters_n_dimes :
  (15 * 25 + 20 * 10 = 5 * 25 + n * 10) -> n = 45 :=
by
  sorry

end fifteen_quarters_twenty_dimes_equal_five_quarters_n_dimes_l106_106597


namespace share_y_is_18_l106_106042

-- Definitions from conditions
def total_amount := 70
def ratio_x := 100
def ratio_y := 45
def ratio_z := 30
def total_ratio := ratio_x + ratio_y + ratio_z
def part_value := total_amount / total_ratio
def share_y := ratio_y * part_value

-- Statement to be proved
theorem share_y_is_18 : share_y = 18 :=
by
  -- Placeholder for the proof
  sorry

end share_y_is_18_l106_106042


namespace bases_with_final_digit_one_in_360_l106_106337

theorem bases_with_final_digit_one_in_360 (b : ℕ) (h : 2 ≤ b ∧ b ≤ 9) : ¬(b ∣ 359) :=
by
  sorry

end bases_with_final_digit_one_in_360_l106_106337


namespace number_of_small_branches_l106_106257

-- Define the number of small branches grown by each branch as a variable
variable (x : ℕ)

-- Define the total number of main stems, branches, and small branches
def total := 1 + x + x * x

theorem number_of_small_branches (h : total x = 91) : x = 9 :=
by
  -- Proof is not required as per instructions
  sorry

end number_of_small_branches_l106_106257


namespace debby_remaining_pictures_l106_106858

variable (zoo_pictures : ℕ) (museum_pictures : ℕ) (deleted_pictures : ℕ)

def initial_pictures (zoo_pictures museum_pictures : ℕ) : ℕ :=
  zoo_pictures + museum_pictures

def remaining_pictures (zoo_pictures museum_pictures deleted_pictures : ℕ) : ℕ :=
  (initial_pictures zoo_pictures museum_pictures) - deleted_pictures

theorem debby_remaining_pictures :
  remaining_pictures 24 12 14 = 22 :=
by
  sorry

end debby_remaining_pictures_l106_106858


namespace stock_worth_is_100_l106_106357

-- Define the number of puppies and kittens
def num_puppies : ℕ := 2
def num_kittens : ℕ := 4

-- Define the cost per puppy and kitten
def cost_per_puppy : ℕ := 20
def cost_per_kitten : ℕ := 15

-- Define the total stock worth function
def stock_worth (num_puppies num_kittens cost_per_puppy cost_per_kitten : ℕ) : ℕ :=
  (num_puppies * cost_per_puppy) + (num_kittens * cost_per_kitten)

-- The theorem to prove that the stock worth is $100
theorem stock_worth_is_100 :
  stock_worth num_puppies num_kittens cost_per_puppy cost_per_kitten = 100 :=
by
  sorry

end stock_worth_is_100_l106_106357


namespace minimum_value_proof_l106_106641

noncomputable def minimum_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) : ℝ :=
  (x + y) / (x * y * z)

theorem minimum_value_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) : 
  minimum_value x y z hx hy hz h ≥ 4 :=
sorry

end minimum_value_proof_l106_106641


namespace probability_first_prize_l106_106893

-- Define the total number of tickets
def total_tickets : ℕ := 150

-- Define the number of first prizes
def first_prizes : ℕ := 5

-- Define the probability calculation as a theorem
theorem probability_first_prize : (first_prizes : ℚ) / total_tickets = 1 / 30 := 
by sorry  -- Placeholder for the proof

end probability_first_prize_l106_106893


namespace water_left_after_four_hours_l106_106342

def initial_water : ℕ := 40
def water_loss_per_hour : ℕ := 2
def added_water_hour3 : ℕ := 1
def added_water_hour4 : ℕ := 3
def total_hours : ℕ := 4

theorem water_left_after_four_hours :
  initial_water - (water_loss_per_hour * total_hours) + (added_water_hour3 + added_water_hour4) = 36 := by
  sorry

end water_left_after_four_hours_l106_106342


namespace merchant_discount_l106_106247

-- Definitions based on conditions
def original_price : ℝ := 1
def increased_price : ℝ := original_price * 1.2
def final_price : ℝ := increased_price * 0.8
def actual_discount : ℝ := original_price - final_price

-- The theorem to be proved
theorem merchant_discount : actual_discount = 0.04 :=
by
  -- Proof goes here
  sorry

end merchant_discount_l106_106247


namespace point_on_graph_l106_106650

def f (x : ℝ) : ℝ := -2 * x + 3

theorem point_on_graph (x y : ℝ) : 
  ( (x = 1 ∧ y = 1) ↔ y = f x ) :=
by 
  sorry

end point_on_graph_l106_106650


namespace initial_green_hard_hats_l106_106285

noncomputable def initial_pink_hard_hats : ℕ := 26
noncomputable def initial_yellow_hard_hats : ℕ := 24
noncomputable def carl_taken_pink_hard_hats : ℕ := 4
noncomputable def john_taken_pink_hard_hats : ℕ := 6
noncomputable def john_taken_green_hard_hats (G : ℕ) : ℕ := 2 * john_taken_pink_hard_hats
noncomputable def remaining_pink_hard_hats : ℕ := initial_pink_hard_hats - carl_taken_pink_hard_hats - john_taken_pink_hard_hats
noncomputable def total_remaining_hard_hats (G : ℕ) : ℕ := remaining_pink_hard_hats + (G - john_taken_green_hard_hats G) + initial_yellow_hard_hats

theorem initial_green_hard_hats (G : ℕ) :
  total_remaining_hard_hats G = 43 ↔ G = 15 := by
  sorry

end initial_green_hard_hats_l106_106285


namespace car_mass_nearest_pound_l106_106400

def mass_of_car_kg : ℝ := 1500
def kg_to_pounds : ℝ := 0.4536

theorem car_mass_nearest_pound :
  (↑(Int.floor ((mass_of_car_kg / kg_to_pounds) + 0.5))) = 3307 :=
by
  sorry

end car_mass_nearest_pound_l106_106400


namespace electronics_weight_l106_106649

-- Define the initial conditions and the solution we want to prove.
theorem electronics_weight (B C E : ℕ) (k : ℕ) 
  (h1 : B = 7 * k) 
  (h2 : C = 4 * k) 
  (h3 : E = 3 * k) 
  (h4 : (B : ℚ) / (C - 8 : ℚ) = 2 * (B : ℚ) / (C : ℚ)) :
  E = 12 := 
sorry

end electronics_weight_l106_106649


namespace negation_proposition_l106_106571

theorem negation_proposition :
  ∃ (a : ℝ) (n : ℕ), n > 0 ∧ a ≠ n ∧ a * n = 2 * n :=
sorry

end negation_proposition_l106_106571


namespace find_a_l106_106146

noncomputable def f (x : ℝ) : ℝ := x + 1 + Real.log x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 1

theorem find_a (a : ℝ) : 
  (∃ x : ℝ, g a x = 2 * x) ∧ (deriv f 1 = 2) ∧ f 1 = 2 → a = 4 :=
by
  -- Math proof goes here
  sorry

end find_a_l106_106146


namespace aaron_ate_more_apples_l106_106606

-- Define the number of apples eaten by Aaron and Zeb
def apples_eaten_by_aaron : ℕ := 6
def apples_eaten_by_zeb : ℕ := 1

-- Theorem to prove the difference in apples eaten
theorem aaron_ate_more_apples :
  apples_eaten_by_aaron - apples_eaten_by_zeb = 5 :=
by
  sorry

end aaron_ate_more_apples_l106_106606


namespace ivan_travel_time_l106_106996

theorem ivan_travel_time (d V_I V_P : ℕ) (h1 : d = 3 * V_I * 40)
  (h2 : ∀ t, t = d / V_P + 10) : 
  (d / V_I = 75) :=
by
  sorry

end ivan_travel_time_l106_106996


namespace sin_2A_cos_C_l106_106109

theorem sin_2A (A B : ℝ) (h1 : Real.sin A = 3 / 5) (h2 : Real.cos B = -5 / 13) : 
  Real.sin (2 * A) = 24 / 25 :=
sorry

theorem cos_C (A B C : ℝ) (h1 : Real.sin A = 3 / 5) (h2 : Real.cos B = -5 / 13) 
  (h3 : ∀ x y z : ℝ, x + y + z = π) :
  Real.cos C = 56 / 65 :=
sorry

end sin_2A_cos_C_l106_106109


namespace sale_in_third_month_l106_106854

def grocer_sales (s1 s2 s4 s5 s6 : ℕ) (average : ℕ) (num_months : ℕ) (total_sales : ℕ) : Prop :=
  s1 = 5266 ∧ s2 = 5768 ∧ s4 = 5678 ∧ s5 = 6029 ∧ s6 = 4937 ∧ average = 5600 ∧ num_months = 6 ∧ total_sales = average * num_months

theorem sale_in_third_month
  (s1 s2 s4 s5 s6 total_sales : ℕ)
  (h : grocer_sales s1 s2 s4 s5 s6 5600 6 total_sales) :
  ∃ s3 : ℕ, total_sales - (s1 + s2 + s4 + s5 + s6) = s3 ∧ s3 = 5922 := 
by {
  sorry
}

end sale_in_third_month_l106_106854


namespace car_speed_second_hour_l106_106559

/-- The speed of the car in the first hour is 85 km/h, the average speed is 65 km/h over 2 hours,
proving that the speed of the car in the second hour is 45 km/h. -/
theorem car_speed_second_hour (v1 : ℕ) (v_avg : ℕ) (t : ℕ) (d1 : ℕ) (d2 : ℕ) 
  (h1 : v1 = 85) (h2 : v_avg = 65) (h3 : t = 2) (h4 : d1 = v1 * 1) (h5 : d2 = (v_avg * t) - d1) :
  d2 = 45 :=
sorry

end car_speed_second_hour_l106_106559


namespace system_of_equations_solution_l106_106865

theorem system_of_equations_solution (x y z : ℝ) 
  (h : ∀ (n : ℕ), x * (1 - 1 / 2^(n : ℝ)) + y * (1 - 1 / 2^(n+1 : ℝ)) + z * (1 - 1 / 2^(n+2 : ℝ)) = 0) : 
  y = -3 * x ∧ z = 2 * x :=
sorry

end system_of_equations_solution_l106_106865


namespace new_oranges_added_l106_106401
-- Import the necessary library

-- Define the constants and conditions
def initial_oranges : ℕ := 5
def thrown_away : ℕ := 2
def total_oranges_now : ℕ := 31

-- Define new_oranges as the variable we want to prove
def new_oranges (x : ℕ) : Prop := x = 28

-- The theorem to prove how many new oranges were added
theorem new_oranges_added :
  ∃ (x : ℕ), new_oranges x ∧ total_oranges_now = initial_oranges - thrown_away + x :=
by
  sorry

end new_oranges_added_l106_106401


namespace conic_curve_eccentricity_l106_106494

theorem conic_curve_eccentricity (m : ℝ) 
    (h1 : ∃ k, k ≠ 0 ∧ 1 * k = m ∧ m * k = 4)
    (h2 : m = -2) : ∃ e : ℝ, e = Real.sqrt 3 :=
by
  sorry

end conic_curve_eccentricity_l106_106494


namespace find_x_minus_y_l106_106202

theorem find_x_minus_y {x y z : ℤ} (h1 : x - (y + z) = 5) (h2 : x - y + z = -1) : x - y = 2 :=
by
  sorry

end find_x_minus_y_l106_106202


namespace largest_integer_le_zero_l106_106714

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem largest_integer_le_zero (x k : ℝ) (h1 : f x = 0) (h2 : 2 < x) (h3 : x < 3) : k ≤ x ∧ k = 2 :=
by
  sorry

end largest_integer_le_zero_l106_106714


namespace percentage_returned_l106_106909

theorem percentage_returned (R : ℕ) (S : ℕ) (total : ℕ) (least_on_lot : ℕ) (max_rented : ℕ)
  (h1 : total = 20) (h2 : least_on_lot = 10) (h3 : max_rented = 20) (h4 : R = 20) (h5 : S ≥ 10) :
  (S / R) * 100 ≥ 50 := sorry

end percentage_returned_l106_106909


namespace sum_of_ages_l106_106030

-- Definitions from the problem conditions
def Maria_age : ℕ := 14
def age_difference_between_Jose_and_Maria : ℕ := 12
def Jose_age : ℕ := Maria_age + age_difference_between_Jose_and_Maria

-- To be proven: sum of their ages is 40
theorem sum_of_ages : Maria_age + Jose_age = 40 :=
by
  -- skip the proof
  sorry

end sum_of_ages_l106_106030


namespace negation_of_p_l106_106579

   -- Define the proposition p as an existential quantification
   def p : Prop := ∃ x₀ : ℝ, x₀^2 + 2 * x₀ + 3 > 0

   -- State the theorem that negation of p is a universal quantification
   theorem negation_of_p : ¬ p ↔ ∀ x : ℝ, x^2 + 2*x + 3 ≤ 0 :=
   by sorry
   
end negation_of_p_l106_106579


namespace bagels_count_l106_106816

def total_items : ℕ := 90
def bread_rolls : ℕ := 49
def croissants : ℕ := 19

def bagels : ℕ := total_items - (bread_rolls + croissants)

theorem bagels_count : bagels = 22 :=
by
  sorry

end bagels_count_l106_106816


namespace abc_sum_l106_106380

theorem abc_sum (a b c : ℤ) 
  (h1 : ∀ x : ℤ, (x + a) * (x + b) = x^2 + 21 * x + 110)
  (h2 : ∀ x : ℤ, (x - b) * (x - c) = x^2 - 19 * x + 88) : 
  a + b + c = 29 := 
by
  sorry

end abc_sum_l106_106380


namespace gcd_12a_18b_l106_106130

theorem gcd_12a_18b (a b : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a.gcd b = 15) : (12 * a).gcd (18 * b) = 90 :=
by sorry

end gcd_12a_18b_l106_106130


namespace find_principal_6400_l106_106167

theorem find_principal_6400 (CI SI P : ℝ) (R T : ℝ) 
  (hR : R = 5) (hT : T = 2) 
  (hSI : SI = P * R * T / 100) 
  (hCI : CI = P * (1 + R / 100) ^ T - P) 
  (hDiff : CI - SI = 16) : 
  P = 6400 := 
by 
  sorry

end find_principal_6400_l106_106167


namespace tammy_trees_l106_106951

-- Define the conditions as Lean definitions and the final statement to prove
theorem tammy_trees :
  (∀ (days : ℕ) (earnings : ℕ) (pricePerPack : ℕ) (orangesPerPack : ℕ) (orangesPerTree : ℕ),
    days = 21 →
    earnings = 840 →
    pricePerPack = 2 →
    orangesPerPack = 6 →
    orangesPerTree = 12 →
    (earnings / days) / (pricePerPack / orangesPerPack) / orangesPerTree = 10) :=
by
  intros days earnings pricePerPack orangesPerPack orangesPerTree
  sorry

end tammy_trees_l106_106951


namespace farmer_sowed_buckets_l106_106702

-- Define the initial and final buckets of seeds
def initial_buckets : ℝ := 8.75
def final_buckets : ℝ := 6.00

-- The goal: prove the number of buckets sowed is 2.75
theorem farmer_sowed_buckets : initial_buckets - final_buckets = 2.75 := by
  sorry

end farmer_sowed_buckets_l106_106702


namespace algebraic_expression_value_l106_106203

-- Define the problem conditions and the final proof statement.
theorem algebraic_expression_value : 
  (∀ m n : ℚ, (2 * m - 1 = 0) → (1 / 2 * n - 2 * m = 0) → m ^ 2023 * n ^ 2022 = 1 / 2) :=
by
  sorry

end algebraic_expression_value_l106_106203


namespace only_setB_is_proportional_l106_106817

-- Definitions for the line segments
def setA := (3, 4, 5, 6)
def setB := (5, 15, 2, 6)
def setC := (4, 8, 3, 5)
def setD := (8, 4, 1, 3)

-- Definition to check if a set of line segments is proportional
def is_proportional (s : ℕ × ℕ × ℕ × ℕ) : Prop :=
  let (a, b, c, d) := s
  a * d = b * c

-- Theorem proving that the only proportional set is set B
theorem only_setB_is_proportional :
  is_proportional setA = false ∧
  is_proportional setB = true ∧
  is_proportional setC = false ∧
  is_proportional setD = false :=
by
  sorry

end only_setB_is_proportional_l106_106817


namespace hypotenuse_length_l106_106602

theorem hypotenuse_length (a b c : ℝ) (h : a^2 + b^2 + c^2 = 2500) (h_right : c^2 = a^2 + b^2) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l106_106602


namespace at_least_one_zero_of_product_zero_l106_106327

theorem at_least_one_zero_of_product_zero (a b c : ℝ) (h : a * b * c = 0) : a = 0 ∨ b = 0 ∨ c = 0 := by
  sorry

end at_least_one_zero_of_product_zero_l106_106327


namespace fewer_students_played_thursday_l106_106243

variable (w t : ℕ)

theorem fewer_students_played_thursday (h1 : w = 37) (h2 : w + t = 65) : w - t = 9 :=
by
  sorry

end fewer_students_played_thursday_l106_106243


namespace child_admission_charge_l106_106758

-- Given conditions
variables (A C : ℝ) (T : ℝ := 3.25) (n : ℕ := 3)

-- Admission charge for an adult
def admission_charge_adult : ℝ := 1

-- Admission charge for a child
def admission_charge_child (C : ℝ) : ℝ := C

-- Total cost paid by adult with 3 children
def total_cost (A C : ℝ) (n : ℕ) : ℝ := A + n * C

-- The proof statement
theorem child_admission_charge (C : ℝ) : total_cost 1 C 3 = 3.25 -> C = 0.75 :=
by
  sorry

end child_admission_charge_l106_106758


namespace common_ratio_of_geometric_sequence_l106_106711

-- Define positive geometric sequence a_n with common ratio q
def geometric_sequence (a q : ℝ) (n : ℕ) : ℝ := a * q^n

-- Define the relevant conditions
variable {a q : ℝ}
variable (h1 : a * q^4 + 2 * a * q^2 * q^6 + a * q^4 * q^8 = 16)
variable (h2 : (a * q^4 + a * q^8) / 2 = 4)
variable (pos_q : q > 0)

-- Define the goal: proving the common ratio q is sqrt(2)
theorem common_ratio_of_geometric_sequence : q = Real.sqrt 2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l106_106711


namespace solution_of_equation_l106_106384

theorem solution_of_equation (m : ℝ) :
  (∃ x : ℝ, x = (4 - 3 * m) / 2 ∧ x > 0) ↔ m < 4 / 3 ∧ m ≠ 2 / 3 :=
by
  sorry

end solution_of_equation_l106_106384


namespace students_chocolate_milk_l106_106255

-- Definitions based on the problem conditions
def students_strawberry_milk : ℕ := 15
def students_regular_milk : ℕ := 3
def total_milks_taken : ℕ := 20

-- The proof goal
theorem students_chocolate_milk : total_milks_taken - (students_strawberry_milk + students_regular_milk) = 2 := by
  -- The proof steps will go here (not required as per instructions)
  sorry

end students_chocolate_milk_l106_106255


namespace minimum_expenses_for_Nikifor_to_win_maximum_F_value_l106_106345

noncomputable def number_of_voters := 35
noncomputable def sellable_voters := 14 -- 40% of 35
noncomputable def preference_voters := 21 -- 60% of 35
noncomputable def minimum_votes_to_win := 18 -- 50% of 35 + 1
noncomputable def cost_per_vote := 9

def vote_supply_function (P : ℕ) : ℕ :=
  if P = 0 then 10
  else if 1 ≤ P ∧ P ≤ 14 then 10 + P
  else 24


theorem minimum_expenses_for_Nikifor_to_win :
  ∃ P : ℕ, P * cost_per_vote = 162 ∧ vote_supply_function P ≥ minimum_votes_to_win := 
sorry

theorem maximum_F_value (F : ℕ) : 
  F = 3 :=
sorry

end minimum_expenses_for_Nikifor_to_win_maximum_F_value_l106_106345


namespace rate_of_current_l106_106512

theorem rate_of_current
  (D U R : ℝ)
  (hD : D = 45)
  (hU : U = 23)
  (hR : R = 34)
  : (D - R = 11) ∧ (R - U = 11) :=
by
  sorry

end rate_of_current_l106_106512


namespace tree_height_l106_106837

theorem tree_height (BR MH MB MR TB : ℝ)
  (h_cond1 : BR = 5)
  (h_cond2 : MH = 1.8)
  (h_cond3 : MB = 1)
  (h_cond4 : MR = BR - MB)
  (h_sim : TB / BR = MH / MR)
  : TB = 2.25 :=
by sorry

end tree_height_l106_106837


namespace milton_zoology_books_l106_106660

variable (Z : ℕ)
variable (total_books botany_books : ℕ)

theorem milton_zoology_books (h1 : total_books = 960)
    (h2 : botany_books = 7 * Z)
    (h3 : total_books = Z + botany_books) :
    Z = 120 := by
  sorry

end milton_zoology_books_l106_106660


namespace count_distribution_schemes_l106_106744

theorem count_distribution_schemes :
  let total_pieces := 7
  let pieces_A_B := 2 + 2
  let remaining_pieces := total_pieces - pieces_A_B
  let communities := 5

  -- Number of ways to distribute 7 pieces of equipment such that communities A and B receive at least 2 pieces each
  let ways_one_community := 5
  let ways_two_communities := 20  -- 2 * (choose 5 2)
  let ways_three_communities := 10  -- (choose 5 3)

  ways_one_community + ways_two_communities + ways_three_communities = 35 :=
by
  -- The actual proof steps are omitted here.
  sorry

end count_distribution_schemes_l106_106744


namespace function_above_x_axis_l106_106015

noncomputable def quadratic_function (a x : ℝ) := (a^2 - 3 * a + 2) * x^2 + (a - 1) * x + 2

theorem function_above_x_axis (a : ℝ) :
  (∀ x : ℝ, quadratic_function a x > 0) ↔ (a > 15 / 7 ∨ a ≤ 1) :=
by {
  sorry
}

end function_above_x_axis_l106_106015


namespace total_rent_of_pasture_l106_106882

theorem total_rent_of_pasture 
  (oxen_A : ℕ) (months_A : ℕ) (oxen_B : ℕ) (months_B : ℕ)
  (oxen_C : ℕ) (months_C : ℕ) (share_C : ℕ) (total_rent : ℕ) :
  oxen_A = 10 →
  months_A = 7 →
  oxen_B = 12 →
  months_B = 5 →
  oxen_C = 15 →
  months_C = 3 →
  share_C = 72 →
  total_rent = 280 :=
by
  intros hA1 hA2 hB1 hB2 hC1 hC2 hC3
  sorry

end total_rent_of_pasture_l106_106882


namespace simplify_fraction_l106_106832

theorem simplify_fraction (h1 : 222 = 2 * 3 * 37) (h2 : 8888 = 8 * 11 * 101) :
  (222 / 8888) * 22 = 1 / 2 :=
by
  sorry

end simplify_fraction_l106_106832


namespace function_periodic_l106_106104

open Real

def periodic (f : ℝ → ℝ) := ∃ T > 0, ∀ x, f (x + T) = f x

theorem function_periodic (a : ℚ) (b d c : ℝ) (f : ℝ → ℝ) 
    (hf : ∀ x : ℝ, f (x + ↑a + b) - f (x + b) = c * (x + 2 * ↑a + ⌊x⌋ - 2 * ⌊x + ↑a⌋ - ⌊b⌋) + d) : 
    periodic f :=
sorry

end function_periodic_l106_106104


namespace range_of_a_l106_106374

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a * x^2 + 2 * a * x - 4 < 0) : -4 < a ∧ a ≤ 0 := 
sorry

end range_of_a_l106_106374


namespace car_speed_l106_106354

theorem car_speed (distance time : ℝ) (h1 : distance = 300) (h2 : time = 5) : distance / time = 60 := by
  have h : distance / time = 300 / 5 := by
    rw [h1, h2]
  norm_num at h
  exact h

end car_speed_l106_106354


namespace cryptarithm_solution_l106_106934

theorem cryptarithm_solution :
  ∃ A B C D E F G H J : ℕ,
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ J ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ J ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ J ∧
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ J ∧
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ J ∧
  F ≠ G ∧ F ≠ H ∧ F ≠ J ∧
  G ≠ H ∧ G ≠ J ∧
  H ≠ J ∧
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧ F < 10 ∧ G < 10 ∧ H < 10 ∧ J < 10 ∧
  (10 * A + B) * (10 * C + A) = 100 * D + 10 * E + B ∧
  (10 * F + C) - (10 * D + G) = D ∧
  (10 * E + G) + (10 * H + J) = 100 * A + 10 * A + G ∧
  A = 1 ∧ B = 7 ∧ C = 2 ∧ D = 3 ∧ E = 5 ∧ F = 4 ∧ G = 9 ∧ H = 6 ∧ J = 0 :=
by
  sorry

end cryptarithm_solution_l106_106934


namespace distance_between_C_and_A_l106_106772

theorem distance_between_C_and_A 
    (A B C : Type)
    (d_AB : ℝ) (d_BC : ℝ)
    (h1 : d_AB = 8)
    (h2 : d_BC = 10) :
    ∃ x : ℝ, 2 ≤ x ∧ x ≤ 18 ∧ ¬ (∃ y : ℝ, y = x) :=
sorry

end distance_between_C_and_A_l106_106772


namespace unique_rectangles_perimeter_sum_correct_l106_106821

def unique_rectangle_sum_of_perimeters : ℕ :=
  let possible_pairs := [(4, 12), (6, 6)]
  let perimeters := possible_pairs.map (λ (p : ℕ × ℕ) => 2 * (p.1 + p.2))
  perimeters.sum

theorem unique_rectangles_perimeter_sum_correct : unique_rectangle_sum_of_perimeters = 56 :=
  by 
  -- skipping actual proof
  sorry

end unique_rectangles_perimeter_sum_correct_l106_106821


namespace train_speed_l106_106108

theorem train_speed (lt_train : ℝ) (lt_bridge : ℝ) (time_cross : ℝ) (total_speed_kmph : ℝ) :
  lt_train = 150 ∧ lt_bridge = 225 ∧ time_cross = 30 ∧ total_speed_kmph = (375 / 30) * 3.6 → 
  total_speed_kmph = 45 := 
by
  sorry

end train_speed_l106_106108


namespace trailing_zeros_300_factorial_l106_106375

-- Definition to count the number of times a prime factor divides the factorial of n
def count_factors (n : ℕ) (p : ℕ) : ℕ :=
  Nat.div (n / p) 1 + Nat.div (n / p^2) 1 + Nat.div (n / p^3) 1 + Nat.div (n / p^4) 1

-- Theorem stating the number of trailing zeros in 300! is 74
theorem trailing_zeros_300_factorial : count_factors 300 5 = 74 := by
  sorry

end trailing_zeros_300_factorial_l106_106375


namespace solution_set_for_inequality_l106_106199

theorem solution_set_for_inequality : 
  { x : ℝ | x * (x - 1) < 2 } = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

end solution_set_for_inequality_l106_106199


namespace inequality_solution_l106_106051

theorem inequality_solution (x : ℝ) :
  (1 / (x^2 + 1) > 4 / x + 19 / 10) ↔ x ∈ Set.Ioo (-2 : ℝ) (0 : ℝ) :=
sorry

end inequality_solution_l106_106051


namespace find_f_11_5_l106_106642

-- Definitions based on the conditions.
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def periodic_with_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

def functional_eqn (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 3) = -1 / f x

def f_defined_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, -3 ≤ x ∧ x ≤ -2 → f x = 2 * x

-- The main theorem to prove.
theorem find_f_11_5 (f : ℝ → ℝ) :
  is_even_function f →
  functional_eqn f →
  f_defined_on_interval f →
  periodic_with_period f 6 →
  f 11.5 = 1 / 5 :=
  by
    intros h_even h_fun_eqn h_interval h_periodic
    sorry  -- proof goes here

end find_f_11_5_l106_106642


namespace variance_transformed_list_l106_106004

noncomputable def stddev (xs : List ℝ) : ℝ := sorry
noncomputable def variance (xs : List ℝ) : ℝ := sorry

theorem variance_transformed_list :
  ∀ (a_1 a_2 a_3 a_4 a_5 : ℝ),
  stddev [a_1, a_2, a_3, a_4, a_5] = 2 →
  variance [3 * a_1 - 2, 3 * a_2 - 2, 3 * a_3 - 2, 3 * a_4 - 2, 3 * a_5 - 2] = 36 :=
by
  intros
  sorry

end variance_transformed_list_l106_106004


namespace noemi_initial_amount_l106_106846

theorem noemi_initial_amount : 
  ∀ (rouletteLoss blackjackLoss pokerLoss baccaratLoss remainingAmount initialAmount : ℕ), 
    rouletteLoss = 600 → 
    blackjackLoss = 800 → 
    pokerLoss = 400 → 
    baccaratLoss = 700 → 
    remainingAmount = 1500 → 
    initialAmount = rouletteLoss + blackjackLoss + pokerLoss + baccaratLoss + remainingAmount →
    initialAmount = 4000 :=
by
  intros rouletteLoss blackjackLoss pokerLoss baccaratLoss remainingAmount initialAmount
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5] at h6
  exact h6

end noemi_initial_amount_l106_106846


namespace find_a1_l106_106591

-- Define the sequence
def seq (a : ℕ → ℝ) := ∀ n : ℕ, 0 < n → a n = (1/2) * a (n + 1)

-- Given conditions
def a3_value (a : ℕ → ℝ) := a 3 = 12

-- Theorem statement
theorem find_a1 (a : ℕ → ℝ) (h_seq : seq a) (h_a3 : a3_value a) : a 1 = 3 :=
by
  sorry

end find_a1_l106_106591


namespace min_value_3x_plus_4y_l106_106048

variable (x y : ℝ)

theorem min_value_3x_plus_4y (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = x * y) : 
  3 * x + 4 * y ≥ 25 :=
sorry

end min_value_3x_plus_4y_l106_106048


namespace find_speed_l106_106123

noncomputable def circumference := 15 / 5280 -- miles
noncomputable def increased_speed (r : ℝ) := r + 5 -- miles per hour
noncomputable def reduced_time (t : ℝ) := t - 1 / 10800 -- hours
noncomputable def original_distance (r t : ℝ) := r * t
noncomputable def new_distance (r t : ℝ) := increased_speed r * reduced_time t

theorem find_speed (r t : ℝ) (h1 : original_distance r t = circumference) 
(h2 : new_distance r t = circumference) : r = 13.5 := by
  sorry

end find_speed_l106_106123


namespace tangent_line_circle_l106_106413

theorem tangent_line_circle (m : ℝ) (h : ∀ x y : ℝ,  (x + y + m = 0) → (x^2 + y^2 = m) → m = 2) : m = 2 :=
sorry

end tangent_line_circle_l106_106413


namespace find_larger_number_l106_106127

variable (L S : ℕ)

theorem find_larger_number 
  (h1 : L - S = 1355) 
  (h2 : L = 6 * S + 15) : 
  L = 1623 := 
sorry

end find_larger_number_l106_106127


namespace functional_equation_solution_l106_106654

noncomputable def func_form (f : ℝ → ℝ) : Prop :=
  ∃ α β : ℝ, (α = 1 ∨ α = -1 ∨ α = 0) ∧ (∀ x, f x = α * x + β ∨ f x = α * x ^ 3 + β)

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ a b c : ℝ, (f a - f b) * (f b - f c) * (f c - f a) = f (a * b ^ 2 + b * c ^ 2 + c * a ^ 2) - f (a ^ 2 * b + b ^ 2 * c + c ^ 2 * a)) →
  func_form f :=
sorry

end functional_equation_solution_l106_106654


namespace x_intercept_of_line_l106_106822

theorem x_intercept_of_line : ∀ x y : ℝ, 2 * x + 3 * y = 6 → y = 0 → x = 3 :=
by
  intros x y h_line h_y_zero
  sorry

end x_intercept_of_line_l106_106822


namespace intersection_point_l106_106906

variable (x y : ℝ)

-- Definitions given by the conditions
def line1 (x y : ℝ) := 3 * y = -2 * x + 6
def line2 (x y : ℝ) := -2 * y = 6 * x + 4

-- The theorem we want to prove
theorem intersection_point : ∃ x y : ℝ, line1 x y ∧ line2 x y ∧ x = -12/7 ∧ y = 22/7 := 
sorry

end intersection_point_l106_106906


namespace abs_div_inequality_l106_106464

theorem abs_div_inequality (x : ℝ) : 
  (|-((x+1)/x)| > (x+1)/x) ↔ (-1 < x ∧ x < 0) :=
sorry

end abs_div_inequality_l106_106464


namespace simplify_expression_l106_106093

theorem simplify_expression :
  (4 + 5) * (4 ^ 2 + 5 ^ 2) * (4 ^ 4 + 5 ^ 4) * (4 ^ 8 + 5 ^ 8) * (4 ^ 16 + 5 ^ 16) * (4 ^ 32 + 5 ^ 32) * (4 ^ 64 + 5 ^ 64) = 5 ^ 128 - 4 ^ 128 :=
by sorry

end simplify_expression_l106_106093


namespace price_difference_is_correct_l106_106248

noncomputable def total_cost : ℝ := 70.93
noncomputable def cost_of_pants : ℝ := 34.0
noncomputable def cost_of_belt : ℝ := total_cost - cost_of_pants
noncomputable def price_difference : ℝ := cost_of_belt - cost_of_pants

theorem price_difference_is_correct :
  price_difference = 2.93 := by
  sorry

end price_difference_is_correct_l106_106248


namespace find_fx_l106_106234

theorem find_fx (f : ℝ → ℝ) 
  (h1 : ∀ x > 0, f (-x) = -(2 * x - 3)) 
  (h2 : ∀ x < 0, -f x = f (-x)) :
  ∀ x < 0, f x = 2 * x + 3 :=
by
  sorry

end find_fx_l106_106234


namespace express_as_scientific_notation_l106_106598

-- Definitions
def billion : ℝ := 10^9
def amount : ℝ := 850 * billion

-- Statement
theorem express_as_scientific_notation : amount = 8.5 * 10^11 :=
by
  sorry

end express_as_scientific_notation_l106_106598


namespace convex_polygon_sides_l106_106811

theorem convex_polygon_sides (S : ℝ) (n : ℕ) (a₁ a₂ a₃ a₄ : ℝ) 
    (h₁ : S = 4320) 
    (h₂ : a₁ = 120) 
    (h₃ : a₂ = 120) 
    (h₄ : a₃ = 120) 
    (h₅ : a₄ = 120) 
    (h_sum : S = 180 * (n - 2)) :
    n = 26 :=
by
  sorry

end convex_polygon_sides_l106_106811


namespace total_project_hours_l106_106047

def research_hours : ℕ := 10
def proposal_hours : ℕ := 2
def report_hours_left : ℕ := 8

theorem total_project_hours :
  research_hours + proposal_hours + report_hours_left = 20 := 
  sorry

end total_project_hours_l106_106047


namespace materials_total_order_l106_106398

theorem materials_total_order :
  let concrete := 0.16666666666666666
  let bricks := 0.16666666666666666
  let stone := 0.5
  concrete + bricks + stone = 0.8333333333333332 :=
by
  sorry

end materials_total_order_l106_106398


namespace product_of_four_consecutive_integers_divisible_by_twelve_l106_106659

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l106_106659


namespace average_birth_rate_l106_106240

theorem average_birth_rate (B : ℕ) (death_rate : ℕ) (net_increase : ℕ) (seconds_per_day : ℕ) 
  (two_sec_intervals : ℕ) (H1 : death_rate = 2) (H2 : net_increase = 86400) (H3 : seconds_per_day = 86400) 
  (H4 : two_sec_intervals = seconds_per_day / 2) 
  (H5 : net_increase = (B - death_rate) * two_sec_intervals) : B = 4 := 
by 
  sorry

end average_birth_rate_l106_106240


namespace subset_A_implies_a_subset_B_implies_range_a_l106_106420

variable (a : ℝ)

def A : Set ℝ := {x | x^2 - 2*x - 8 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + a^2 - 12 = 0}

theorem subset_A_implies_a (h : A ⊆ B a) : a = -2 := 
sorry

theorem subset_B_implies_range_a (h : B a ⊆ A) : a >= 4 ∨ a < -4 ∨ a = -2 := 
sorry

end subset_A_implies_a_subset_B_implies_range_a_l106_106420


namespace base_length_of_isosceles_l106_106600

-- Define the lengths of the sides and the perimeter of the triangle.
def side_length1 : ℝ := 10
def side_length2 : ℝ := 10
def perimeter : ℝ := 35

-- Define the problem statement to prove the length of the base.
theorem base_length_of_isosceles (b : ℝ) 
  (h1 : side_length1 = 10) 
  (h2 : side_length2 = 10) 
  (h3 : perimeter = 35) : b = 15 :=
by
  -- Skip the proof.
  sorry

end base_length_of_isosceles_l106_106600


namespace eventually_composite_appending_threes_l106_106546

theorem eventually_composite_appending_threes (n : ℕ) :
  ∃ n' : ℕ, n' = 10 * n + 3 ∧ ∃ k : ℕ, k > 0 ∧ (3 * k + 3) % 7 ≠ 1 ∧ (3 * k + 3) % 7 ≠ 2 ∧ (3 * k + 3) % 7 ≠ 3 ∧
  (3 * k + 3) % 7 ≠ 5 ∧ (3 * k + 3) % 7 ≠ 6 :=
sorry

end eventually_composite_appending_threes_l106_106546


namespace determine_fake_coin_l106_106264

theorem determine_fake_coin (N : ℕ) : 
  (∃ (n : ℕ), N = 2 * n + 2) ↔ (∃ (n : ℕ), N = 2 * n + 2) := by 
  sorry

end determine_fake_coin_l106_106264


namespace possible_values_of_a_l106_106751

variables {a b k : ℤ}

def sum_distances (a : ℤ) (k : ℤ) : ℤ :=
  (a - k).natAbs + (a - (k + 1)).natAbs + (a - (k + 2)).natAbs +
  (a - (k + 3)).natAbs + (a - (k + 4)).natAbs + (a - (k + 5)).natAbs +
  (a - (k + 6)).natAbs + (a - (k + 7)).natAbs + (a - (k + 8)).natAbs +
  (a - (k + 9)).natAbs + (a - (k + 10)).natAbs

theorem possible_values_of_a :
  sum_distances a k = 902 →
  sum_distances b k = 374 →
  a + b = 98 →
  a = 25 ∨ a = 107 ∨ a = -9 :=
sorry

end possible_values_of_a_l106_106751


namespace units_digit_sum_base8_l106_106493

theorem units_digit_sum_base8 : 
  let n1 := 53 
  let n2 := 64 
  let sum_base8 := n1 + n2 
  (sum_base8 % 8) = 7 := 
by 
  sorry

end units_digit_sum_base8_l106_106493


namespace correct_equation_by_moving_digit_l106_106590

theorem correct_equation_by_moving_digit :
  (10^2 - 1 = 99) → (101 = 102 - 1) :=
by
  intro h
  sorry

end correct_equation_by_moving_digit_l106_106590


namespace total_eyes_in_extended_family_l106_106062

def mom_eyes := 1
def dad_eyes := 3
def kids_eyes := 3 * 4
def moms_previous_child_eyes := 5
def dads_previous_children_eyes := 6 + 2
def dads_ex_wife_eyes := 1
def dads_ex_wifes_new_partner_eyes := 7
def child_of_ex_wife_and_partner_eyes := 8

theorem total_eyes_in_extended_family :
  mom_eyes + dad_eyes + kids_eyes + moms_previous_child_eyes + dads_previous_children_eyes +
  dads_ex_wife_eyes + dads_ex_wifes_new_partner_eyes + child_of_ex_wife_and_partner_eyes = 45 :=
by
  -- add proof here
  sorry

end total_eyes_in_extended_family_l106_106062


namespace expression_value_l106_106452

theorem expression_value : 2013 * (2015 / 2014) + 2014 * (2016 / 2015) + (4029 / (2014 * 2015)) = 4029 :=
by
  sorry

end expression_value_l106_106452


namespace compound_interest_interest_l106_106719

theorem compound_interest_interest :
  let P := 2000
  let r := 0.05
  let n := 5
  let A := P * (1 + r)^n
  let interest := A - P
  interest = 552.56 := by
  sorry

end compound_interest_interest_l106_106719


namespace john_saves_1200_yearly_l106_106299

noncomputable def former_rent_per_month (sq_ft_cost : ℝ) (sq_ft : ℝ) : ℝ :=
  sq_ft_cost * sq_ft

noncomputable def new_rent_per_month (total_cost : ℝ) (roommates : ℝ) : ℝ :=
  total_cost / roommates

noncomputable def monthly_savings (former_rent : ℝ) (new_rent : ℝ) : ℝ :=
  former_rent - new_rent

noncomputable def annual_savings (monthly_savings : ℝ) : ℝ :=
  monthly_savings * 12

theorem john_saves_1200_yearly :
  let former_rent := former_rent_per_month 2 750
  let new_rent := new_rent_per_month 2800 2
  let monthly_savings := monthly_savings former_rent new_rent
  annual_savings monthly_savings = 1200 := 
by 
  sorry

end john_saves_1200_yearly_l106_106299


namespace arithmetic_sequence_sum_l106_106519

theorem arithmetic_sequence_sum {a : ℕ → ℤ} (S : ℕ → ℤ) 
  (h₀ : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2)
  (h₁ : S 9 = 27) :
  (a 4 + a 6) = 6 :=
sorry

end arithmetic_sequence_sum_l106_106519


namespace angie_age_problem_l106_106786

theorem angie_age_problem (a certain_number : ℕ) 
  (h1 : 2 * 8 + certain_number = 20) : 
  certain_number = 4 :=
by 
  sorry

end angie_age_problem_l106_106786


namespace right_triangle_of_ratio_and_right_angle_l106_106813

-- Define the sides and the right angle condition based on the problem conditions
variable (x : ℝ) (hx : 0 < x)

-- Variables for the sides in the given ratio
def a := 3 * x
def b := 4 * x
def c := 5 * x

-- The proposition we need to prove
theorem right_triangle_of_ratio_and_right_angle (h : a^2 + b^2 = c^2) : a^2 + b^2 = c^2 :=
by sorry  -- Proof not required as per instructions

end right_triangle_of_ratio_and_right_angle_l106_106813


namespace revenue_times_l106_106780

noncomputable def revenue_ratio (D : ℝ) : ℝ :=
  let revenue_Nov := (2 / 5) * D
  let revenue_Jan := (1 / 3) * revenue_Nov
  let average := (revenue_Nov + revenue_Jan) / 2
  D / average

theorem revenue_times (D : ℝ) (hD : D ≠ 0) : revenue_ratio D = 3.75 :=
by
  -- skipped proof
  sorry

end revenue_times_l106_106780


namespace correct_sampling_methods_l106_106044

-- Define conditions for the sampling problems
structure SamplingProblem where
  scenario: String
  samplingMethod: String

-- Define the three scenarios
def firstScenario : SamplingProblem :=
  { scenario := "Draw 5 bottles from 15 bottles of drinks for food hygiene inspection", samplingMethod := "Simple random sampling" }

def secondScenario : SamplingProblem :=
  { scenario := "Sample 20 staff members from 240 staff members in a middle school", samplingMethod := "Stratified sampling" }

def thirdScenario : SamplingProblem :=
  { scenario := "Select 25 audience members from a full science and technology report hall", samplingMethod := "Systematic sampling" }

-- Main theorem combining all conditions and proving the correct answer
theorem correct_sampling_methods :
  (firstScenario.samplingMethod = "Simple random sampling") ∧
  (secondScenario.samplingMethod = "Stratified sampling") ∧
  (thirdScenario.samplingMethod = "Systematic sampling") :=
by
  sorry -- Proof is omitted

end correct_sampling_methods_l106_106044


namespace cosine_identity_l106_106853

theorem cosine_identity (α : ℝ) (h : Real.sin (Real.pi / 6 + α) = Real.sqrt 3 / 2) :
  Real.cos (Real.pi / 3 - α) = Real.sqrt 3 / 2 := 
by
  sorry

end cosine_identity_l106_106853


namespace overall_average_score_l106_106651

theorem overall_average_score 
  (M : ℝ) (E : ℝ) (m e : ℝ)
  (hM : M = 82)
  (hE : E = 75)
  (hRatio : m / e = 5 / 3) :
  (M * m + E * e) / (m + e) = 79.375 := 
by
  sorry

end overall_average_score_l106_106651


namespace total_fires_l106_106645

theorem total_fires (Doug_fires Kai_fires Eli_fires : ℕ)
  (h1 : Doug_fires = 20)
  (h2 : Kai_fires = 3 * Doug_fires)
  (h3 : Eli_fires = Kai_fires / 2) :
  Doug_fires + Kai_fires + Eli_fires = 110 :=
by
  sorry

end total_fires_l106_106645


namespace dave_total_time_l106_106190

variable (W J : ℕ)

-- Given conditions
def time_walked := W = 9
def ratio := J / W = 4 / 3

-- Statement to prove
theorem dave_total_time (time_walked : time_walked W) (ratio : ratio J W) : W + J = 21 := 
by
  sorry

end dave_total_time_l106_106190


namespace negation_of_p_l106_106372

open Classical

variable (p : Prop)

theorem negation_of_p (h : ∀ x : ℝ, x^3 + 2 < 0) : 
  ∃ x : ℝ, x^3 + 2 ≥ 0 :=
by
  sorry

end negation_of_p_l106_106372


namespace cost_price_of_cloth_l106_106258

-- Definitions for conditions
def sellingPrice (totalMeters : ℕ) : ℕ := 8500
def profitPerMeter : ℕ := 15
def totalMeters : ℕ := 85

-- Proof statement with conditions and expected proof
theorem cost_price_of_cloth : 
  (sellingPrice totalMeters) = 8500 -> 
  profitPerMeter = 15 -> 
  totalMeters = 85 -> 
  (8500 - (profitPerMeter * totalMeters)) / totalMeters = 85 := 
by 
  sorry

end cost_price_of_cloth_l106_106258


namespace ratio_A_to_B_l106_106440

noncomputable def A_annual_income : ℝ := 436800.0000000001
noncomputable def B_increase_rate : ℝ := 0.12
noncomputable def C_monthly_income : ℝ := 13000

noncomputable def A_monthly_income : ℝ := A_annual_income / 12
noncomputable def B_monthly_income : ℝ := C_monthly_income + (B_increase_rate * C_monthly_income)

theorem ratio_A_to_B :
  ((A_monthly_income / 80) : ℝ) = 455 ∧
  ((B_monthly_income / 80) : ℝ) = 182 :=
by
  sorry

end ratio_A_to_B_l106_106440


namespace at_least_one_not_less_than_one_l106_106564

theorem at_least_one_not_less_than_one (x : ℝ) (a b c : ℝ) 
  (ha : a = x^2 + 1/2) 
  (hb : b = 2 - x) 
  (hc : c = x^2 - x + 1) : 
  (1 ≤ a) ∨ (1 ≤ b) ∨ (1 ≤ c) := 
sorry

end at_least_one_not_less_than_one_l106_106564


namespace jacob_writing_speed_ratio_l106_106759

theorem jacob_writing_speed_ratio (N : ℕ) (J : ℕ) (hN : N = 25) (h1 : J + N = 75) : J / N = 2 :=
by {
  sorry
}

end jacob_writing_speed_ratio_l106_106759


namespace factorize_polynomial_l106_106890

theorem factorize_polynomial (c : ℝ) :
  (x : ℝ) → (x - 1) * (x - 3) = x^2 - 4 * x + c → c = 3 :=
by 
  sorry

end factorize_polynomial_l106_106890


namespace remainder_when_dividing_sum_l106_106966

theorem remainder_when_dividing_sum (k m : ℤ) (c d : ℤ) (h1 : c = 60 * k + 47) (h2 : d = 42 * m + 17) :
  (c + d) % 21 = 1 :=
by
  sorry

end remainder_when_dividing_sum_l106_106966


namespace method_is_systematic_sampling_l106_106456

-- Define the conditions
def rows : ℕ := 25
def seats_per_row : ℕ := 20
def filled_auditorium : Prop := True
def seat_numbered_15_sampled : Prop := True
def interval : ℕ := 20

-- Define the concept of systematic sampling
def systematic_sampling (rows seats_per_row interval : ℕ) : Prop :=
  (rows > 0 ∧ seats_per_row > 0 ∧ interval > 0 ∧ (interval = seats_per_row))

-- State the problem in terms of proving that the sampling method is systematic
theorem method_is_systematic_sampling :
  filled_auditorium → seat_numbered_15_sampled → systematic_sampling rows seats_per_row interval :=
by
  intros h1 h2
  -- Assume that the proof goes here
  sorry

end method_is_systematic_sampling_l106_106456


namespace sum_possible_x_l106_106055

noncomputable def sum_of_x (x : ℝ) : ℝ :=
  let lst : List ℝ := [1, 2, 5, 2, 3, 2, x]
  let mean := (1 + 2 + 5 + 2 + 3 + 2 + x) / 7
  let median := 2
  let mode := 2
  if lst = List.reverse lst ∧ mean ≠ mode then
    mean
  else 
    0

theorem sum_possible_x : sum_of_x 1 + sum_of_x 5 = 6 :=
by 
  sorry

end sum_possible_x_l106_106055


namespace larry_substitution_l106_106616

theorem larry_substitution (a b c d e : ℤ)
  (h_a : a = 2)
  (h_b : b = 5)
  (h_c : c = 3)
  (h_d : d = 4)
  (h_expr1 : a + b - c - d * e = 4 - 4 * e)
  (h_expr2 : a + (b - (c - (d * e))) = 4 + 4 * e) :
  e = 0 :=
by
  sorry

end larry_substitution_l106_106616


namespace S_4n_l106_106707

variable {a : ℕ → ℕ}
variable (S : ℕ → ℝ)
variable (n : ℕ)
variable (r : ℝ)
variable (a1 : ℝ)

-- Conditions
axiom geometric_sequence : ∀ n, a (n + 1) = a n * r
axiom positive_terms : ∀ n, 0 < a n
axiom sum_n : S n = a1 * (1 - r^n) / (1 - r)
axiom sum_3n : S (3 * n) = 14
axiom sum_n_value : S n = 2

-- Theorem
theorem S_4n : S (4 * n) = 30 :=
sorry

end S_4n_l106_106707


namespace value_of_a7_l106_106791

-- Define the geometric sequence and its properties
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- Define the conditions of the problem
variables (a : ℕ → ℝ) (h_geom : is_geometric_sequence a) (h_pos : ∀ n : ℕ, a n > 0) (h_product : a 3 * a 11 = 16)

-- Conjecture that we aim to prove
theorem value_of_a7 : a 7 = 4 :=
by {
  sorry
}

end value_of_a7_l106_106791


namespace multiplication_of_decimals_l106_106071

theorem multiplication_of_decimals : (0.4 * 0.75 = 0.30) := by
  sorry

end multiplication_of_decimals_l106_106071


namespace rhombus_new_perimeter_l106_106416

theorem rhombus_new_perimeter (d1 d2 : ℝ) (scale : ℝ) (h_d1 : d1 = 10) (h_d2 : d2 = 24) (h_scale : scale = 0.5) : 
  4 * (scale * (Real.sqrt ((d1/2)^2 + (d2/2)^2))) = 26 := 
by
  sorry

end rhombus_new_perimeter_l106_106416


namespace beanie_babies_total_l106_106142

theorem beanie_babies_total
  (Lori_beanie_babies : ℕ) (Sydney_beanie_babies : ℕ)
  (h1 : Lori_beanie_babies = 15 * Sydney_beanie_babies)
  (h2 : Lori_beanie_babies = 300) :
  Lori_beanie_babies + Sydney_beanie_babies = 320 :=
sorry

end beanie_babies_total_l106_106142


namespace salad_dressing_vinegar_percentage_l106_106419

-- Define the initial conditions
def percentage_in_vinegar_in_Q : ℝ := 10
def percentage_of_vinegar_in_combined : ℝ := 12
def percentage_of_dressing_P_in_combined : ℝ := 0.10
def percentage_of_dressing_Q_in_combined : ℝ := 0.90
def percentage_of_vinegar_in_P (V : ℝ) : ℝ := V

-- The statement to prove
theorem salad_dressing_vinegar_percentage (V : ℝ) 
  (hQ : percentage_in_vinegar_in_Q = 10)
  (hCombined : percentage_of_vinegar_in_combined = 12)
  (hP_combined : percentage_of_dressing_P_in_combined = 0.10)
  (hQ_combined : percentage_of_dressing_Q_in_combined = 0.90)
  (hV_combined : 0.10 * percentage_of_vinegar_in_P V + 0.90 * percentage_in_vinegar_in_Q = 12) :
  V = 30 :=
by 
  sorry

end salad_dressing_vinegar_percentage_l106_106419


namespace rhombus_area_l106_106204

-- Define the lengths of the diagonals
def d1 : ℝ := 25
def d2 : ℝ := 30

-- Statement to prove that the area of the rhombus is 375 square centimeters
theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 25) (h2 : d2 = 30) : 
  (d1 * d2) / 2 = 375 := by
  -- Proof to be provided
  sorry

end rhombus_area_l106_106204


namespace multiple_of_regular_rate_is_1_5_l106_106321

-- Definitions
def hourly_rate := 5.50
def regular_hours := 7.5
def total_hours := 10.5
def total_earnings := 66.0
def excess_hours := total_hours - regular_hours
def regular_earnings := regular_hours * hourly_rate
def excess_earnings := total_earnings - regular_earnings
def rate_per_excess_hour := excess_earnings / excess_hours
def multiple_of_regular_rate := rate_per_excess_hour / hourly_rate

-- Statement of the problem
theorem multiple_of_regular_rate_is_1_5 : multiple_of_regular_rate = 1.5 :=
by
  -- Note: The proof is not required, hence sorry is used.
  sorry

end multiple_of_regular_rate_is_1_5_l106_106321


namespace initial_birds_l106_106254

-- Given conditions
def number_birds_initial (x : ℕ) : Prop :=
  ∃ (y : ℕ), y = 4 ∧ (x + y = 6)

-- Proof statement
theorem initial_birds : ∃ x : ℕ, number_birds_initial x ↔ x = 2 :=
by {
  sorry
}

end initial_birds_l106_106254


namespace f_2019_eq_2019_l106_106850

def f : ℝ → ℝ := sorry

axiom f_pos : ∀ x, x > 0 → f x > 0
axiom f_one : f 1 = 1
axiom f_eq : ∀ a b : ℝ, f (a + b) * (f a + f b) = 2 * f a * f b + a^2 + b^2

theorem f_2019_eq_2019 : f 2019 = 2019 :=
by sorry

end f_2019_eq_2019_l106_106850


namespace students_catching_up_on_homework_l106_106184

theorem students_catching_up_on_homework
  (total_students : ℕ)
  (half_doing_silent_reading : ℕ)
  (third_playing_board_games : ℕ)
  (remain_catching_up_homework : ℕ) :
  total_students = 24 →
  half_doing_silent_reading = total_students / 2 →
  third_playing_board_games = total_students / 3 →
  remain_catching_up_homework = total_students - (half_doing_silent_reading + third_playing_board_games) →
  remain_catching_up_homework = 4 :=
by
  intros h_total h_half h_third h_remain
  sorry

end students_catching_up_on_homework_l106_106184


namespace exists_arithmetic_progression_with_sum_zero_l106_106036

theorem exists_arithmetic_progression_with_sum_zero : 
  ∃ (a d : Int) (n : Int), n > 0 ∧ (n * (2 * a + (n - 1) * d)) = 0 :=
by 
  sorry

end exists_arithmetic_progression_with_sum_zero_l106_106036


namespace lyle_notebook_cost_l106_106713

theorem lyle_notebook_cost (pen_cost : ℝ) (notebook_multiplier : ℝ) (num_notebooks : ℕ) 
  (h_pen_cost : pen_cost = 1.50) (h_notebook_mul : notebook_multiplier = 3) 
  (h_num_notebooks : num_notebooks = 4) :
  (pen_cost * notebook_multiplier) * num_notebooks = 18 := 
  by
  sorry

end lyle_notebook_cost_l106_106713


namespace value_of_b_add_c_l106_106433

variables {a b c d : ℝ}

theorem value_of_b_add_c (h1 : a + b = 5) (h2 : c + d = 3) (h3 : a + d = 2) : b + c = 6 :=
sorry

end value_of_b_add_c_l106_106433


namespace slices_per_sandwich_l106_106223

theorem slices_per_sandwich (total_sandwiches : ℕ) (total_slices : ℕ) (h1 : total_sandwiches = 5) (h2 : total_slices = 15) :
  total_slices / total_sandwiches = 3 :=
by sorry

end slices_per_sandwich_l106_106223


namespace John_pushup_count_l106_106781

-- Definitions arising from conditions
def Zachary_pushups : ℕ := 51
def David_pushups : ℕ := Zachary_pushups + 22
def John_pushups : ℕ := David_pushups - 4

-- Theorem statement
theorem John_pushup_count : John_pushups = 69 := 
by 
  sorry

end John_pushup_count_l106_106781


namespace largest_digit_M_l106_106475

-- Define the conditions as Lean types
def digit_sum_divisible_by_3 (M : ℕ) := (4 + 5 + 6 + 7 + M) % 3 = 0
def even_digit (M : ℕ) := M % 2 = 0

-- Define the problem statement in Lean
theorem largest_digit_M (M : ℕ) (h : even_digit M ∧ digit_sum_divisible_by_3 M) : M ≤ 8 ∧ (∀ N : ℕ, even_digit N ∧ digit_sum_divisible_by_3 N → N ≤ M) :=
sorry

end largest_digit_M_l106_106475


namespace tino_jellybeans_l106_106658

variable (Tino Lee Arnold : ℕ)

-- Conditions
variable (h1 : Tino = Lee + 24)
variable (h2 : Arnold = Lee / 2)
variable (h3 : Arnold = 5)

-- Prove Tino has 34 jellybeans
theorem tino_jellybeans : Tino = 34 :=
by
  sorry

end tino_jellybeans_l106_106658


namespace find_f2_l106_106412

noncomputable def f (a b x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem find_f2 (a b : ℝ) (h : f a b (-2) = 10) : f a b 2 = -26 :=
by
  sorry

end find_f2_l106_106412


namespace sabina_loan_l106_106482

-- Define the conditions
def tuition_per_year : ℕ := 30000
def living_expenses_per_year : ℕ := 12000
def duration : ℕ := 4
def sabina_savings : ℕ := 10000
def grant_first_two_years_percent : ℕ := 40
def grant_last_two_years_percent : ℕ := 30
def scholarship_percent : ℕ := 20

-- Calculate total tuition for 4 years
def total_tuition : ℕ := tuition_per_year * duration

-- Calculate total living expenses for 4 years
def total_living_expenses : ℕ := living_expenses_per_year * duration

-- Calculate total cost
def total_cost : ℕ := total_tuition + total_living_expenses

-- Calculate grant coverage
def grant_first_two_years : ℕ := (grant_first_two_years_percent * tuition_per_year / 100) * 2
def grant_last_two_years : ℕ := (grant_last_two_years_percent * tuition_per_year / 100) * 2
def total_grant_coverage : ℕ := grant_first_two_years + grant_last_two_years

-- Calculate scholarship savings
def annual_scholarship_savings : ℕ := living_expenses_per_year * scholarship_percent / 100
def total_scholarship_savings : ℕ := annual_scholarship_savings * (duration - 1)

-- Calculate total reductions
def total_reductions : ℕ := total_grant_coverage + total_scholarship_savings + sabina_savings

-- Calculate the total loan needed
def total_loan_needed : ℕ := total_cost - total_reductions

theorem sabina_loan : total_loan_needed = 108800 := by
  sorry

end sabina_loan_l106_106482


namespace addition_comm_subtraction_compare_multiplication_comm_subtraction_greater_l106_106230

theorem addition_comm (a b : ℕ) : a + b = b + a :=
by sorry

theorem subtraction_compare {a b c : ℕ} (h1 : a < b) (h2 : c = 28) : 56 - c < 65 - c :=
by sorry

theorem multiplication_comm (a b : ℕ) : a * b = b * a :=
by sorry

theorem subtraction_greater {a b c : ℕ} (h1 : a - b = 18) (h2 : a - c = 27) (h3 : 32 = b) (h4 : 23 = c) : a - b > a - c :=
by sorry

end addition_comm_subtraction_compare_multiplication_comm_subtraction_greater_l106_106230


namespace factory_output_increase_l106_106214

theorem factory_output_increase (x : ℝ) (h : (1 + x / 100) ^ 4 = 4) : x = 41.4 :=
by
  -- Given (1 + x / 100) ^ 4 = 4
  sorry

end factory_output_increase_l106_106214


namespace geom_seq_common_ratio_q_l106_106936

-- Define the geometric sequence
def geom_seq (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)

-- State the theorem
theorem geom_seq_common_ratio_q {a₁ q : ℝ} :
  (a₁ = 2) → (geom_seq a₁ q 4 = 16) → (q = 2) :=
by
  intros h₁ h₂
  sorry

end geom_seq_common_ratio_q_l106_106936


namespace valid_number_of_m_values_l106_106389

theorem valid_number_of_m_values : 
  (∃ m : ℕ, 2 ≤ m ∧ m ∣ 420 ∧ 2 ≤ (420 / m)) ∧ ∀ m, 2 ≤ m ∧ m ∣ 420 ∧ 2 ≤ (420 / m) → m > 1  → 
  ∃ n : ℕ, n = 22 :=
by
  sorry

end valid_number_of_m_values_l106_106389


namespace proof_problem_l106_106896

def U : Set ℝ := {x | True}
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | x ≤ -1}

theorem proof_problem :
  ((A ∩ {x | x > -1}) ∪ (B ∩ {x | x ≤ 0})) = {x | x > 0 ∨ x ≤ -1} :=
by 
  sorry

end proof_problem_l106_106896


namespace rhombus_perimeter_l106_106496

-- Define the conditions for the rhombus
variable (d1 d2 : ℝ) (a b s : ℝ)

-- State the condition that the diagonals of a rhombus measure 24 cm and 10 cm
def diagonal_condition := (d1 = 24) ∧ (d2 = 10)

-- State the Pythagorean theorem for the lengths of half-diagonals
def pythagorean_theorem := a^2 + b^2 = s^2

-- State the relationship of diagonals bisecting each other at right angles
def bisect_condition := (a = d1 / 2) ∧ (b = d2 / 2)

-- State the definition of the perimeter for a rhombus
def perimeter (s : ℝ) : ℝ := 4 * s

-- The theorem we want to prove
theorem rhombus_perimeter : diagonal_condition d1 d2 →
                            bisect_condition d1 d2 a b →
                            pythagorean_theorem a b s →
                            perimeter s = 52 :=
by
  intros h1 h2 h3
  -- Proof would go here, but it is omitted
  sorry

end rhombus_perimeter_l106_106496


namespace flour_needed_for_one_batch_l106_106121

theorem flour_needed_for_one_batch (F : ℝ) (h1 : 8 * F + 8 * 1.5 = 44) : F = 4 := 
by
    sorry

end flour_needed_for_one_batch_l106_106121


namespace fiona_shirt_number_l106_106438

def is_two_digit_prime (n : ℕ) : Prop := 
  (n ≥ 10 ∧ n < 100 ∧ Nat.Prime n)

theorem fiona_shirt_number (d e f : ℕ) 
  (h1 : is_two_digit_prime d)
  (h2 : is_two_digit_prime e)
  (h3 : is_two_digit_prime f)
  (h4 : e + f = 36)
  (h5 : d + e = 30)
  (h6 : d + f = 32) : 
  f = 19 := 
sorry

end fiona_shirt_number_l106_106438


namespace lloyd_excess_rate_multiple_l106_106605

theorem lloyd_excess_rate_multiple :
  let h_regular := 7.5
  let r := 4.00
  let h_total := 10.5
  let e_total := 48
  let e_regular := h_regular * r
  let excess_hours := h_total - h_regular
  let e_excess := e_total - e_regular
  let m := e_excess / (excess_hours * r)
  m = 1.5 :=
by
  sorry

end lloyd_excess_rate_multiple_l106_106605


namespace smallest_n_for_g4_l106_106994

def g (n : ℕ) : ℕ :=
  (List.range (n + 1)).foldl (λ acc a => acc + (List.range (n + 1)).countP (λ b => a * a + b * b = n)) 0

theorem smallest_n_for_g4 : ∃ n : ℕ, g n = 4 ∧ 
  (∀ m : ℕ, m < n → g m ≠ 4) :=
by
  use 65
  -- Proof would go here
  sorry

end smallest_n_for_g4_l106_106994


namespace original_curve_equation_l106_106139

theorem original_curve_equation (x y : ℝ) (θ : ℝ) (hθ : θ = π / 4)
  (h : (∃ P : ℝ × ℝ, P = (x, y) ∧ (∃ P' : ℝ × ℝ, P' = (x * Real.cos θ - y * Real.sin θ, x * Real.sin θ + y * Real.cos θ) ∧ ((P'.fst)^2 - (P'.snd)^2 = 2)))) :
  x * y = -1 :=
sorry

end original_curve_equation_l106_106139


namespace find_k_range_l106_106960

open Nat

def a_n (n : ℕ) : ℕ := 2^ (5 - n)

def b_n (n : ℕ) (k : ℤ) : ℤ := n + k

def c_n (n : ℕ) (k : ℤ) : ℤ :=
if (a_n n : ℤ) ≤ (b_n n k) then b_n n k else a_n n

theorem find_k_range : 
  (∀ n ∈ { m : ℕ | m > 0 }, c_n 5 = a_n 5 ∧ c_n 5 ≤ c_n n) → 
  (∃ k : ℤ, -5 ≤ k ∧ k ≤ -3) :=
by
  sorry

end find_k_range_l106_106960


namespace find_length_AX_l106_106888

theorem find_length_AX 
  (A B C X : Type)
  (BC BX AC : ℝ)
  (h_BC : BC = 36)
  (h_BX : BX = 30)
  (h_AC : AC = 27)
  (h_bisector : ∃ (x : ℝ), x = BX / BC ∧ x = AX / AC ) :
  ∃ AX : ℝ, AX = 22.5 := 
sorry

end find_length_AX_l106_106888


namespace lowest_price_for_butter_l106_106441

def cost_single_package : ℝ := 7.0
def cost_8oz_package : ℝ := 4.0
def cost_4oz_package : ℝ := 2.0
def discount : ℝ := 0.5

theorem lowest_price_for_butter : 
  min cost_single_package (cost_8oz_package + 2 * (cost_4oz_package * discount)) = 6.0 :=
by
  sorry

end lowest_price_for_butter_l106_106441


namespace problem_l106_106734

noncomputable def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem problem (f : ℝ → ℝ) (h : isOddFunction f) : 
  f (-2012) + f (-2011) + f 0 + f 2011 + f 2012 = 0 :=
by
  sorry

end problem_l106_106734


namespace algebraic_expression_value_l106_106442

theorem algebraic_expression_value (x : ℝ) (h : x^2 + x - 1 = 0) : x^3 + 2 * x^2 - 7 = -6 :=
by
  sorry

end algebraic_expression_value_l106_106442


namespace equation_solution_unique_l106_106730

theorem equation_solution_unique (x y : ℤ) : 
  x^4 = y^2 + 2*y + 2 ↔ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = -1) :=
by
  sorry

end equation_solution_unique_l106_106730


namespace initial_cars_l106_106941

theorem initial_cars (X : ℕ) : (X - 13 + (13 + 5) = 85) → (X = 80) :=
by
  sorry

end initial_cars_l106_106941


namespace find_X_l106_106563

theorem find_X 
  (X Y : ℕ)
  (h1 : 6 + X = 13)
  (h2 : Y = 7) :
  X = 7 := by
  sorry

end find_X_l106_106563


namespace identify_incorrect_proposition_l106_106461

-- Definitions based on problem conditions
def propositionA : Prop :=
  (∀ x : ℝ, (x^2 - 3*x + 2 = 0 → x = 1) ↔ (x ≠ 1 → x^2 - 3*x + 2 ≠ 0))

def propositionB : Prop :=
  (¬ (∃ x : ℝ, x^2 + x + 1 = 0) ↔ ∀ x : ℝ, x^2 + x + 1 ≠ 0)

def propositionD (x : ℝ) : Prop :=
  (x > 2 → x^2 - 3*x + 2 > 0) ∧ (¬(x > 2) → ¬(x^2 - 3*x + 2 > 0))

-- Proposition C is given to be incorrect in the problem
def propositionC (p q : Prop) : Prop := ¬ (p ∧ q) → ¬p ∧ ¬q

theorem identify_incorrect_proposition (p q : Prop) : 
  (propositionA ∧ propositionB ∧ (∀ x : ℝ, propositionD x)) → 
  ¬ (propositionC p q) :=
by
  intros
  -- We know proposition C is false based on the problem's solution
  sorry

end identify_incorrect_proposition_l106_106461
