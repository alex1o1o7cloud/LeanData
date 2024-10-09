import Mathlib

namespace initial_ratio_milk_water_l1594_159494

theorem initial_ratio_milk_water (M W : ℕ) 
  (h1 : M + W = 45) 
  (h2 : M = 3 * (W + 3)) 
  : M / W = 4 := 
sorry

end initial_ratio_milk_water_l1594_159494


namespace fraction_expression_evaluation_l1594_159481

theorem fraction_expression_evaluation : 
  (1/4 - 1/6) / (1/3 - 1/4) = 1 := 
by
  sorry

end fraction_expression_evaluation_l1594_159481


namespace min_value_a_plus_b_l1594_159404

open Real

theorem min_value_a_plus_b (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h : 1 / a + 2 / b = 1) :
  a + b = 3 + 2 * sqrt 2 :=
sorry

end min_value_a_plus_b_l1594_159404


namespace binom_8_2_eq_28_l1594_159468

open Nat

theorem binom_8_2_eq_28 : Nat.choose 8 2 = 28 := by
  sorry

end binom_8_2_eq_28_l1594_159468


namespace student_difference_l1594_159437

theorem student_difference 
  (C1 : ℕ) (x : ℕ)
  (hC1 : C1 = 25)
  (h_total : C1 + (C1 - x) + (C1 - 2 * x) + (C1 - 3 * x) + (C1 - 4 * x) = 105) : 
  x = 2 := 
by
  sorry

end student_difference_l1594_159437


namespace probability_of_square_or_circle_is_seven_tenths_l1594_159484

-- Define the total number of figures
def total_figures : ℕ := 10

-- Define the number of squares
def num_squares : ℕ := 4

-- Define the number of circles
def num_circles : ℕ := 3

-- The number of squares or circles
def num_squares_or_circles : ℕ := num_squares + num_circles

-- The probability of selecting a square or a circle
def probability_square_or_circle : ℚ := num_squares_or_circles / total_figures

-- The theorem stating the required proof
theorem probability_of_square_or_circle_is_seven_tenths :
  probability_square_or_circle = 7/10 :=
sorry -- proof goes here

end probability_of_square_or_circle_is_seven_tenths_l1594_159484


namespace maximize_profit_l1594_159492

noncomputable def profit (x : ℕ) : ℝ :=
  let price := (180 + 10 * x : ℝ)
  let rooms_occupied := (50 - x : ℝ)
  let expenses := 20
  (price - expenses) * rooms_occupied

theorem maximize_profit :
  ∃ x : ℕ, profit x = profit 17 → (180 + 10 * x) = 350 :=
by
  use 17
  sorry

end maximize_profit_l1594_159492


namespace triplet_D_sum_not_one_l1594_159485

def triplet_sum_not_equal_to_one : Prop :=
  (1.2 + -0.2 + 0.0 ≠ 1)

theorem triplet_D_sum_not_one : triplet_sum_not_equal_to_one := 
  by
    sorry

end triplet_D_sum_not_one_l1594_159485


namespace average_books_collected_per_day_l1594_159455

theorem average_books_collected_per_day :
  let n := 7
  let a := 12
  let d := 12
  let S_n := (n * (2 * a + (n - 1) * d)) / 2
  S_n / n = 48 :=
by
  let n := 7
  let a := 12
  let d := 12
  let S_n := (n * (2 * a + (n - 1) * d)) / 2
  show S_n / n = 48
  sorry

end average_books_collected_per_day_l1594_159455


namespace new_average_doubled_marks_l1594_159445

theorem new_average_doubled_marks (n : ℕ) (avg : ℕ) (h_n : n = 11) (h_avg : avg = 36) :
  (2 * avg * n) / n = 72 :=
by
  sorry

end new_average_doubled_marks_l1594_159445


namespace translate_parabola_l1594_159487

noncomputable def f (x : ℝ) : ℝ := 3 * x^2

noncomputable def g (x : ℝ) : ℝ := 3 * (x - 1)^2 - 4

theorem translate_parabola (x : ℝ) : g x = 3 * (x - 1)^2 - 4 :=
by {
  -- proof would go here
  sorry
}

end translate_parabola_l1594_159487


namespace geometric_mean_a_b_l1594_159422

theorem geometric_mean_a_b : ∀ (a b : ℝ), a > 0 → b > 0 → Real.sqrt 3 = Real.sqrt (3^a * 3^b) → a + b = 1 :=
by
  intros a b ha hb hgeo
  sorry

end geometric_mean_a_b_l1594_159422


namespace students_not_in_any_subject_l1594_159471

theorem students_not_in_any_subject (total_students mathematics_students chemistry_students biology_students
  mathematics_chemistry_students chemistry_biology_students mathematics_biology_students all_three_students: ℕ)
  (h_total: total_students = 120) 
  (h_m: mathematics_students = 70)
  (h_c: chemistry_students = 50)
  (h_b: biology_students = 40)
  (h_mc: mathematics_chemistry_students = 30)
  (h_cb: chemistry_biology_students = 20)
  (h_mb: mathematics_biology_students = 10)
  (h_all: all_three_students = 5) :
  total_students - ((mathematics_students - mathematics_chemistry_students - mathematics_biology_students + all_three_students) +
    (chemistry_students - chemistry_biology_students - mathematics_chemistry_students + all_three_students) +
    (biology_students - chemistry_biology_students - mathematics_biology_students + all_three_students) +
    (mathematics_chemistry_students + chemistry_biology_students + mathematics_biology_students - 2 * all_three_students)) = 20 :=
by sorry

end students_not_in_any_subject_l1594_159471


namespace find_a_squared_plus_b_squared_l1594_159454

theorem find_a_squared_plus_b_squared (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 1) : a^2 + b^2 = 7 := 
by
  sorry

end find_a_squared_plus_b_squared_l1594_159454


namespace fraction_of_pianists_got_in_l1594_159473

-- Define the conditions
def flutes_got_in (f : ℕ) := f = 16
def clarinets_got_in (c : ℕ) := c = 15
def trumpets_got_in (t : ℕ) := t = 20
def total_band_members (total : ℕ) := total = 53
def total_pianists (p : ℕ) := p = 20

-- The main statement we want to prove
theorem fraction_of_pianists_got_in : 
  ∃ (pi : ℕ), 
    flutes_got_in 16 ∧ 
    clarinets_got_in 15 ∧ 
    trumpets_got_in 20 ∧ 
    total_band_members 53 ∧ 
    total_pianists 20 ∧ 
    pi / 20 = 1 / 10 := 
  sorry

end fraction_of_pianists_got_in_l1594_159473


namespace correct_calculation_l1594_159438

variable (a : ℕ)

theorem correct_calculation : 
  ¬(a + a = a^2) ∧ ¬(a^3 * a = a^3) ∧ ¬(a^8 / a^2 = a^4) ∧ ((a^3)^2 = a^6) := 
by
  sorry

end correct_calculation_l1594_159438


namespace cookout_ratio_l1594_159412

theorem cookout_ratio (K_2004 K_2005 : ℕ) (h1 : K_2004 = 60) (h2 : (2 / 3) * K_2005 = 20) :
  K_2005 / K_2004 = 1 / 2 :=
by sorry

end cookout_ratio_l1594_159412


namespace encore_songs_l1594_159439

-- Definitions corresponding to the conditions
def repertoire_size : ℕ := 30
def first_set_songs : ℕ := 5
def second_set_songs : ℕ := 7
def average_songs_per_set_3_and_4 : ℕ := 8

-- The statement to prove
theorem encore_songs : (repertoire_size - (first_set_songs + second_set_songs)) - (2 * average_songs_per_set_3_and_4) = 2 := by
  sorry

end encore_songs_l1594_159439


namespace bananas_left_l1594_159475

theorem bananas_left (dozen_bananas : ℕ) (eaten_bananas : ℕ) (h1 : dozen_bananas = 12) (h2 : eaten_bananas = 2) : dozen_bananas - eaten_bananas = 10 :=
sorry

end bananas_left_l1594_159475


namespace find_abc_sum_l1594_159417

noncomputable def x := Real.sqrt ((Real.sqrt 105) / 2 + 7 / 2)

theorem find_abc_sum :
  ∃ (a b c : ℕ), a + b + c = 5824 ∧
  x ^ 100 = 3 * x ^ 98 + 15 * x ^ 96 + 12 * x ^ 94 - x ^ 50 + a * x ^ 46 + b * x ^ 44 + c * x ^ 40 :=
  sorry

end find_abc_sum_l1594_159417


namespace smallest_value_geq_4_l1594_159474

noncomputable def smallest_value (a b c d : ℝ) : ℝ :=
  (a + b + c + d) * ((1 / (a + b + d)) + (1 / (a + c + d)) + (1 / (b + c + d)))

theorem smallest_value_geq_4 (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  smallest_value a b c d ≥ 4 :=
by
  sorry

end smallest_value_geq_4_l1594_159474


namespace geometric_sequence_common_ratio_l1594_159440

theorem geometric_sequence_common_ratio (a₁ : ℕ) (S₃ : ℕ) (q : ℤ) 
  (h₁ : a₁ = 2) (h₂ : S₃ = 6) : 
  (q = 1 ∨ q = -2) :=
by
  sorry

end geometric_sequence_common_ratio_l1594_159440


namespace total_candies_l1594_159426

-- Condition definitions
def lindaCandies : ℕ := 34
def chloeCandies : ℕ := 28

-- Proof statement to show their total candies
theorem total_candies : lindaCandies + chloeCandies = 62 := 
by
  sorry

end total_candies_l1594_159426


namespace sum_of_vertices_l1594_159406

theorem sum_of_vertices (rect_verts: Nat) (pent_verts: Nat) (h1: rect_verts = 4) (h2: pent_verts = 5) : rect_verts + pent_verts = 9 :=
by
  sorry

end sum_of_vertices_l1594_159406


namespace area_of_larger_square_l1594_159451

theorem area_of_larger_square (side_length : ℕ) (num_squares : ℕ)
  (h₁ : side_length = 2)
  (h₂ : num_squares = 8) : 
  (num_squares * side_length^2) = 32 :=
by
  sorry

end area_of_larger_square_l1594_159451


namespace arrangements_count_l1594_159453

-- Definitions of students and grades
inductive Student : Type
| A | B | C | D | E | F
deriving DecidableEq

inductive Grade : Type
| first | second | third
deriving DecidableEq

-- A function to count valid arrangements
def valid_arrangements (assignments : Student → Grade) : Bool :=
  assignments Student.A = Grade.first ∧
  assignments Student.B ≠ Grade.third ∧
  assignments Student.C ≠ Grade.third ∧
  (assignments Student.A = Grade.first) ∧
  ((assignments Student.B = Grade.second ∧ assignments Student.C = Grade.second ∧ 
    (assignments Student.D ≠ Grade.first ∨ assignments Student.E ≠ Grade.first ∨ assignments Student.F ≠ Grade.first)) ∨
   ((assignments Student.B ≠ Grade.second ∨ assignments Student.C ≠ Grade.second) ∧ 
    (assignments Student.B ≠ Grade.first ∨ assignments Student.C ≠ Grade.first)))

theorem arrangements_count : 
  ∃ (count : ℕ), count = 9 ∧
  count = (Nat.card { assign : Student → Grade // valid_arrangements assign } : ℕ) := sorry

end arrangements_count_l1594_159453


namespace min_value_expression_l1594_159402

open Real

theorem min_value_expression (x y z: ℝ) (h1: 0 < x) (h2: 0 < y) (h3: 0 < z)
    (h4: (x / y + y / z + z / x) + (y / x + z / y + x / z) = 10):
    (x / y + y / z + z / x) * (y / x + z / y + x / z) = 25 :=
by
  sorry

end min_value_expression_l1594_159402


namespace intersection_of_A_and_B_l1594_159483

def setA : Set ℝ := { x | (x - 3) * (x + 1) ≥ 0 }
def setB : Set ℝ := { x | x < -4/5 }

theorem intersection_of_A_and_B : setA ∩ setB = { x | x ≤ -1 } :=
  sorry

end intersection_of_A_and_B_l1594_159483


namespace average_visitors_per_day_l1594_159447

theorem average_visitors_per_day
  (sunday_visitors : ℕ := 540)
  (other_days_visitors : ℕ := 240)
  (days_in_month : ℕ := 30)
  (first_day_is_sunday : Bool := true)
  (result : ℕ := 290) :
  let num_sundays := 5
  let num_other_days := days_in_month - num_sundays
  let total_visitors := num_sundays * sunday_visitors + num_other_days * other_days_visitors
  let average_visitors := total_visitors / days_in_month
  average_visitors = result :=
by
  sorry

end average_visitors_per_day_l1594_159447


namespace odd_function_f_l1594_159425

noncomputable def f : ℝ → ℝ := sorry

theorem odd_function_f (f_odd : ∀ x : ℝ, f (-x) = - f x)
                       (f_lt_0 : ∀ x : ℝ, x < 0 → f x = x * (x - 1)) :
  ∀ x : ℝ, x > 0 → f x = - x * (x + 1) :=
by
  sorry

end odd_function_f_l1594_159425


namespace polygon_angle_multiple_l1594_159477

theorem polygon_angle_multiple (m : ℕ) (h : m ≥ 3) : 
  (∃ k : ℕ, (2 * m - 2) * 180 = k * ((m - 2) * 180)) ↔ (m = 3 ∨ m = 4) :=
by sorry

end polygon_angle_multiple_l1594_159477


namespace three_gorges_dam_capacity_scientific_notation_l1594_159427

theorem three_gorges_dam_capacity_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), (16780000 : ℝ) = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.678 ∧ n = 7 :=
by
  sorry

end three_gorges_dam_capacity_scientific_notation_l1594_159427


namespace total_baseball_cards_l1594_159420

/-- 
Given that you have 5 friends and each friend gets 91 baseball cards, 
prove that the total number of baseball cards you have is 455.
-/
def baseball_cards (f c : Nat) (t : Nat) : Prop :=
  (t = f * c)

theorem total_baseball_cards:
  ∀ (f c t : Nat), f = 5 → c = 91 → t = 455 → baseball_cards f c t :=
by
  intros f c t hf hc ht
  sorry

end total_baseball_cards_l1594_159420


namespace highest_percentage_without_car_l1594_159418

noncomputable def percentage_without_car (total_percentage : ℝ) (car_percentage : ℝ) : ℝ :=
  total_percentage - total_percentage * car_percentage / 100

theorem highest_percentage_without_car :
  let A_total := 30
  let A_with_car := 25
  let B_total := 50
  let B_with_car := 15
  let C_total := 20
  let C_with_car := 35

  percentage_without_car A_total A_with_car = 22.5 /\
  percentage_without_car B_total B_with_car = 42.5 /\
  percentage_without_car C_total C_with_car = 13 /\
  percentage_without_car B_total B_with_car = max (percentage_without_car A_total A_with_car) (max (percentage_without_car B_total B_with_car) (percentage_without_car C_total C_with_car)) :=
by
  sorry

end highest_percentage_without_car_l1594_159418


namespace min_value_inequality_l1594_159463

theorem min_value_inequality (x y : ℝ) (h1 : x^2 + y^2 = 3) (h2 : |x| ≠ |y|) :
  ∃ (m : ℝ), m = (1 / (2*x + y)^2 + 4 / (x - 2*y)^2) ∧ m = 3 / 5 :=
by
  sorry

end min_value_inequality_l1594_159463


namespace continuous_stripe_probability_l1594_159448

-- Define the conditions of the tetrahedron and stripe orientations
def tetrahedron_faces : ℕ := 4
def stripe_orientations_per_face : ℕ := 2
def total_stripe_combinations : ℕ := stripe_orientations_per_face ^ tetrahedron_faces
def favorable_stripe_combinations : ℕ := 2 -- Clockwise and Counterclockwise combinations for a continuous stripe

-- Define the probability calculation
def probability_of_continuous_stripe : ℚ :=
  favorable_stripe_combinations / total_stripe_combinations

-- Theorem statement
theorem continuous_stripe_probability : probability_of_continuous_stripe = 1 / 8 :=
by
  -- The proof is omitted for brevity
  sorry

end continuous_stripe_probability_l1594_159448


namespace min_value_of_expression_l1594_159491

theorem min_value_of_expression (x y : ℤ) (h : 4 * x + 5 * y = 7) : ∃ k : ℤ, 
  5 * Int.natAbs (3 + 5 * k) - 3 * Int.natAbs (-1 - 4 * k) = 1 :=
sorry

end min_value_of_expression_l1594_159491


namespace egg_whites_per_cake_l1594_159416

-- Define the conversion ratio between tablespoons of aquafaba and egg whites
def tablespoons_per_egg_white : ℕ := 2

-- Define the total amount of aquafaba used for two cakes
def total_tablespoons_for_two_cakes : ℕ := 32

-- Define the number of cakes
def number_of_cakes : ℕ := 2

-- Prove the number of egg whites needed per cake
theorem egg_whites_per_cake :
  (total_tablespoons_for_two_cakes / tablespoons_per_egg_white) / number_of_cakes = 8 := by
  sorry

end egg_whites_per_cake_l1594_159416


namespace problem_statement_l1594_159429

variable {x a : Real}

theorem problem_statement (h1 : x < a) (h2 : a < 0) : x^2 > a * x ∧ a * x > a^2 := 
sorry

end problem_statement_l1594_159429


namespace lucille_house_difference_l1594_159419

def height_lucille : ℕ := 80
def height_neighbor1 : ℕ := 70
def height_neighbor2 : ℕ := 99

def average_height (h1 h2 h3 : ℕ) : ℕ := (h1 + h2 + h3) / 3

def difference (h_average h_actual : ℕ) : ℕ := h_average - h_actual

theorem lucille_house_difference :
  difference (average_height height_lucille height_neighbor1 height_neighbor2) height_lucille = 3 :=
by
  unfold difference
  unfold average_height
  sorry

end lucille_house_difference_l1594_159419


namespace trace_bag_weight_l1594_159434

-- Define the weights of Gordon's bags
def gordon_bag1_weight : ℕ := 3
def gordon_bag2_weight : ℕ := 7

-- Define the number of Trace's bags
def trace_num_bags : ℕ := 5

-- Define what we are trying to prove: the weight of one of Trace's shopping bags
theorem trace_bag_weight :
  (gordon_bag1_weight + gordon_bag2_weight) = (trace_num_bags * 2) :=
by
  sorry

end trace_bag_weight_l1594_159434


namespace sandy_marks_loss_l1594_159472

theorem sandy_marks_loss (n m c p : ℕ) (h1 : n = 30) (h2 : m = 65) (h3 : c = 25) (h4 : p = 3) :
  ∃ x : ℕ, (c * p - m) / (n - c) = x ∧ x = 2 := by
  sorry

end sandy_marks_loss_l1594_159472


namespace chromium_atoms_in_compound_l1594_159443

-- Definitions of given conditions
def hydrogen_atoms : Nat := 2
def oxygen_atoms : Nat := 4
def compound_molecular_weight : ℝ := 118
def hydrogen_atomic_weight : ℝ := 1
def chromium_atomic_weight : ℝ := 52
def oxygen_atomic_weight : ℝ := 16

-- Problem statement to find the number of Chromium atoms
theorem chromium_atoms_in_compound (hydrogen_atoms : Nat) (oxygen_atoms : Nat) (compound_molecular_weight : ℝ)
    (hydrogen_atomic_weight : ℝ) (chromium_atomic_weight : ℝ) (oxygen_atomic_weight : ℝ) :
  hydrogen_atoms * hydrogen_atomic_weight + 
  oxygen_atoms * oxygen_atomic_weight + 
  chromium_atomic_weight = compound_molecular_weight → 
  chromium_atomic_weight = 52 :=
by
  sorry

end chromium_atoms_in_compound_l1594_159443


namespace find_larger_number_l1594_159460

variable (x y : ℕ)

theorem find_larger_number (h1 : 4 * y = 5 * x) (h2 : y - x = 10) : y = 50 := 
by 
  sorry

end find_larger_number_l1594_159460


namespace log_relation_l1594_159446

theorem log_relation (a b c: ℝ) (h₁: a = (Real.log 2) / 2) (h₂: b = (Real.log 3) / 3) (h₃: c = (Real.log 5) / 5) : c < a ∧ a < b :=
by
  sorry

end log_relation_l1594_159446


namespace factorization_of_w4_minus_81_l1594_159409

theorem factorization_of_w4_minus_81 (w : ℝ) : 
  (w^4 - 81) = (w - 3) * (w + 3) * (w^2 + 9) :=
by sorry

end factorization_of_w4_minus_81_l1594_159409


namespace min_value_of_sum_of_powers_l1594_159470

theorem min_value_of_sum_of_powers (x y : ℝ) (h : x + 3 * y = 1) : 
  2^x + 8^y ≥ 2 * Real.sqrt 2 :=
by
  sorry

end min_value_of_sum_of_powers_l1594_159470


namespace triangle_angle_C_l1594_159493

theorem triangle_angle_C (A B C : ℝ) (h1 : A = 86) (h2 : B = 3 * C + 22) (h3 : A + B + C = 180) : C = 18 :=
by
  sorry

end triangle_angle_C_l1594_159493


namespace sid_money_left_after_purchases_l1594_159431

theorem sid_money_left_after_purchases : 
  ∀ (original_money money_spent_on_computer money_spent_on_snacks half_of_original_money money_left final_more_than_half),
  original_money = 48 → 
  money_spent_on_computer = 12 → 
  money_spent_on_snacks = 8 →
  half_of_original_money = original_money / 2 → 
  money_left = original_money - (money_spent_on_computer + money_spent_on_snacks) → 
  final_more_than_half = money_left - half_of_original_money →
  final_more_than_half = 4 := 
by
  intros original_money money_spent_on_computer money_spent_on_snacks half_of_original_money money_left final_more_than_half
  intros h1 h2 h3 h4 h5 h6
  sorry

end sid_money_left_after_purchases_l1594_159431


namespace find_q_from_min_y_l1594_159435

variables (a p q m : ℝ)
variable (a_nonzero : a ≠ 0)
variable (min_y : ∀ x : ℝ, a*x^2 + p*x + q ≥ m)

theorem find_q_from_min_y :
  q = m + p^2 / (4 * a) :=
sorry

end find_q_from_min_y_l1594_159435


namespace trapezoid_total_area_l1594_159423

/-- 
Given a trapezoid with side lengths 4, 6, 8, and 10, where sides 4 and 8 are used as parallel bases, 
prove that the total area of the trapezoid in all possible configurations is 48√2.
-/
theorem trapezoid_total_area : 
  let a := 4
  let b := 8
  let c := 6
  let d := 10
  let h := 4 * Real.sqrt 2
  let Area := (1 / 2) * (a + b) * h
  (Area + Area) = 48 * Real.sqrt 2 :=
by 
  sorry

end trapezoid_total_area_l1594_159423


namespace miniature_tower_height_l1594_159469

theorem miniature_tower_height
  (actual_height : ℝ)
  (actual_volume : ℝ)
  (miniature_volume : ℝ)
  (actual_height_eq : actual_height = 60)
  (actual_volume_eq : actual_volume = 200000)
  (miniature_volume_eq : miniature_volume = 0.2) :
  ∃ (miniature_height : ℝ), miniature_height = 0.6 :=
by
  sorry

end miniature_tower_height_l1594_159469


namespace necessary_but_not_sufficient_l1594_159466

variables {a b : ℝ}

theorem necessary_but_not_sufficient (h : a > 0) (h₁ : a > b) (h₂ : a⁻¹ > b⁻¹) : 
  b < 0 :=
sorry

end necessary_but_not_sufficient_l1594_159466


namespace common_roots_product_sum_l1594_159461

theorem common_roots_product_sum (C D u v w t p q r : ℝ) (huvw : u^3 + C * u - 20 = 0) (hvw : v^3 + C * v - 20 = 0)
  (hw: w^3 + C * w - 20 = 0) (hut: t^3 + D * t^2 - 40 = 0) (hvw: v^3 + D * v^2 - 40 = 0) 
  (hu: u^3 + D * u^2 - 40 = 0) (h1: u + v + w = 0) (h2: u * v * w = 20) 
  (h3: u * v + u * t + v * t = 0) (h4: u * v * t = 40) :
  p = 4 → q = 3 → r = 5 → p + q + r = 12 :=
by sorry

end common_roots_product_sum_l1594_159461


namespace factorize_expression_l1594_159408

-- The primary goal is to prove that -2xy^2 + 4xy - 2x = -2x(y - 1)^2
theorem factorize_expression (x y : ℝ) : 
  -2 * x * y^2 + 4 * x * y - 2 * x = -2 * x * (y - 1)^2 := 
by 
  sorry

end factorize_expression_l1594_159408


namespace age_of_child_l1594_159490

theorem age_of_child (H W C : ℕ) (h1 : (H + W) / 2 = 23) (h2 : (H + 5 + W + 5 + C) / 3 = 19) : C = 1 := by
  sorry

end age_of_child_l1594_159490


namespace ratio_of_sequence_l1594_159411

variables (a b c : ℝ)

-- Condition 1: arithmetic sequence
def arithmetic_sequence : Prop := 2 * b = a + c

-- Condition 2: geometric sequence
def geometric_sequence : Prop := c^2 = a * b

-- Theorem stating the ratio of a:b:c
theorem ratio_of_sequence (h1 : arithmetic_sequence a b c) (h2 : geometric_sequence a b c) : 
  (a = 4 * b) ∧ (c = -2 * b) :=
sorry

end ratio_of_sequence_l1594_159411


namespace count_4x4_increasing_arrays_l1594_159495

-- Define the notion of a 4x4 grid that satisfies the given conditions
def isInIncreasingOrder (matrix : (Fin 4) → (Fin 4) → Nat) : Prop :=
  (∀ i j : Fin 4, i < 3 -> matrix i j < matrix (i+1) j) ∧
  (∀ i j : Fin 4, j < 3 -> matrix i j < matrix i (j+1))

def validGrid (matrix : (Fin 4) → (Fin 4) → Nat) : Prop :=
  (∀ i j : Fin 4, 1 ≤ matrix i j ∧ matrix i j ≤ 16) ∧ isInIncreasingOrder matrix

noncomputable def countValidGrids : ℕ :=
  sorry

theorem count_4x4_increasing_arrays : countValidGrids = 13824 :=
  sorry

end count_4x4_increasing_arrays_l1594_159495


namespace time_to_pass_platform_is_correct_l1594_159499

noncomputable def train_length : ℝ := 250 -- meters
noncomputable def time_to_pass_pole : ℝ := 10 -- seconds
noncomputable def time_to_pass_platform : ℝ := 60 -- seconds

-- Speed of the train
noncomputable def train_speed := train_length / time_to_pass_pole -- meters/second

-- Length of the platform
noncomputable def platform_length := train_speed * time_to_pass_platform - train_length -- meters

-- Proving the time to pass the platform is 50 seconds
theorem time_to_pass_platform_is_correct : 
  (platform_length / train_speed) = 50 :=
by
  sorry

end time_to_pass_platform_is_correct_l1594_159499


namespace number_of_distinct_values_l1594_159415

theorem number_of_distinct_values (n : ℕ) (mode_count : ℕ) (second_count : ℕ) (total_count : ℕ) 
    (h1 : n = 3000) (h2 : mode_count = 15) (h3 : second_count = 14) : 
    (n - mode_count - second_count) / 13 + 2 ≥ 232 :=
by 
  sorry

end number_of_distinct_values_l1594_159415


namespace integer_solution_existence_l1594_159450

theorem integer_solution_existence : ∃ (x y : ℤ), 2 * x + y - 1 = 0 :=
by
  use 1
  use -1
  sorry

end integer_solution_existence_l1594_159450


namespace deceased_member_income_l1594_159496

theorem deceased_member_income
  (initial_income_4_members : ℕ)
  (initial_members : ℕ := 4)
  (initial_average_income : ℕ := 840)
  (final_income_3_members : ℕ)
  (remaining_members : ℕ := 3)
  (final_average_income : ℕ := 650)
  (total_income_initial : initial_income_4_members = initial_average_income * initial_members)
  (total_income_final : final_income_3_members = final_average_income * remaining_members)
  (income_deceased : ℕ) :
  income_deceased = initial_income_4_members - final_income_3_members :=
by
  -- sorry indicates this part of the proof is left as an exercise
  sorry

end deceased_member_income_l1594_159496


namespace inequality_for_positive_reals_l1594_159430

theorem inequality_for_positive_reals (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) (k : ℕ) (h_k : 2 ≤ k) :
  (a^k / (a + b) + b^k / (b + c) + c^k / (c + a) ≥ 3 / 2) :=
by
  intros
  sorry

end inequality_for_positive_reals_l1594_159430


namespace find_number_l1594_159400

theorem find_number (x : ℝ) (h : 0.80 * 40 = (4/5) * x + 16) : x = 20 :=
by sorry

end find_number_l1594_159400


namespace tan_sum_formula_l1594_159482

theorem tan_sum_formula {A B : ℝ} (hA : A = 55) (hB : B = 65) (h1 : Real.tan (A + B) = Real.tan 120) 
    (h2 : Real.tan 120 = -Real.sqrt 3) :
    Real.tan 55 + Real.tan 65 - Real.sqrt 3 * Real.tan 55 * Real.tan 65 = -Real.sqrt 3 := 
by
  sorry

end tan_sum_formula_l1594_159482


namespace possible_values_of_expression_l1594_159433

theorem possible_values_of_expression (x y : ℝ) (hxy : x + 2 * y = 2) (hx_pos : x > 0) (hy_pos : y > 0) :
  ∃ v, v = 21 / 4 ∧ (1 / x + 2 / y) = v :=
sorry

end possible_values_of_expression_l1594_159433


namespace find_y_solution_l1594_159441

variable (y : ℚ)

theorem find_y_solution (h : (y^2 - 12*y + 32) / (y - 2) + (3*y^2 + 11*y - 14) / (3*y - 1) = -5) : 
    y = -17/6 :=
by
  sorry

end find_y_solution_l1594_159441


namespace tree_count_in_yard_l1594_159407

-- Definitions from conditions
def yard_length : ℕ := 350
def tree_distance : ℕ := 14

-- Statement of the theorem
theorem tree_count_in_yard : (yard_length / tree_distance) + 1 = 26 := by
  sorry

end tree_count_in_yard_l1594_159407


namespace tip_percentage_l1594_159480

theorem tip_percentage
  (total_amount_paid : ℝ)
  (price_of_food : ℝ)
  (sales_tax_rate : ℝ)
  (total_amount : ℝ)
  (tip_percentage : ℝ)
  (h1 : total_amount_paid = 184.80)
  (h2 : price_of_food = 140)
  (h3 : sales_tax_rate = 0.10)
  (h4 : total_amount = price_of_food + (price_of_food * sales_tax_rate))
  (h5 : tip_percentage = ((total_amount_paid - total_amount) / total_amount) * 100) :
  tip_percentage = 20 := sorry

end tip_percentage_l1594_159480


namespace zero_of_my_function_l1594_159456

-- Define the function y = e^(2x) - 1
noncomputable def my_function (x : ℝ) : ℝ :=
  Real.exp (2 * x) - 1

-- Statement that the zero of the function is at x = 0
theorem zero_of_my_function : my_function 0 = 0 :=
by sorry

end zero_of_my_function_l1594_159456


namespace cos_double_angle_of_tan_half_l1594_159457

theorem cos_double_angle_of_tan_half (α : ℝ) (h : Real.tan α = 1 / 2) :
  Real.cos (2 * α) = 3 / 5 :=
sorry

end cos_double_angle_of_tan_half_l1594_159457


namespace max_coins_as_pleases_max_coins_equally_distributed_l1594_159442

-- Part a
theorem max_coins_as_pleases {N : ℕ} (N_warriors : N = 33) (total_coins : ℕ := 240) : 
  ∃ k : ℕ, k ≤ N ∧ (∃ remaining_coins : ℕ, remaining_coins ≤ total_coins ∧ remaining_coins = 31) := 
by
  sorry

-- Part b
theorem max_coins_equally_distributed {N : ℕ} (N_warriors : N = 33) (total_coins : ℕ := 240) : 
  ∃ k : ℕ, k ≤ N ∧ (∃ remaining_coins : ℕ, remaining_coins ≤ total_coins ∧ remaining_coins = 30) := 
by
  sorry

end max_coins_as_pleases_max_coins_equally_distributed_l1594_159442


namespace g_at_1_l1594_159486

variable (g : ℝ → ℝ)

theorem g_at_1 (h : ∀ x : ℝ, g (2 * x - 5) = 3 * x + 9) : g 1 = 18 := by
  sorry

end g_at_1_l1594_159486


namespace tan_alpha_eq_neg2_complex_expression_eq_neg5_l1594_159403

variables (α : ℝ)
variables (h_sin : Real.sin α = - (2 * Real.sqrt 5) / 5)
variables (h_tan_neg : Real.tan α < 0)

theorem tan_alpha_eq_neg2 :
  Real.tan α = -2 :=
sorry

theorem complex_expression_eq_neg5 :
  (2 * Real.sin (α + Real.pi) + Real.cos (2 * Real.pi - α)) /
  (Real.cos (α - Real.pi / 2) - Real.sin (3 * Real.pi / 2 + α)) = -5 :=
sorry

end tan_alpha_eq_neg2_complex_expression_eq_neg5_l1594_159403


namespace total_money_spent_l1594_159497

/-- 
John buys a gaming PC for $1200.
He decides to replace the video card in it.
He sells the old card for $300 and buys a new one for $500.
Prove total money spent on the computer after counting the savings from selling the old card is $1400.
-/
theorem total_money_spent (initial_cost : ℕ) (sale_price_old_card : ℕ) (price_new_card : ℕ) : 
  (initial_cost = 1200) → (sale_price_old_card = 300) → (price_new_card = 500) → 
  (initial_cost + (price_new_card - sale_price_old_card) = 1400) :=
by 
  intros
  sorry

end total_money_spent_l1594_159497


namespace quadratic_y_axis_intersection_l1594_159405

theorem quadratic_y_axis_intersection :
  (∃ y, (y = (0 - 1) ^ 2 + 2) ∧ (0, y) = (0, 3)) :=
sorry

end quadratic_y_axis_intersection_l1594_159405


namespace polynomial_use_square_of_binomial_form_l1594_159458

theorem polynomial_use_square_of_binomial_form (a b x y : ℝ) :
  (1 + x) * (x + 1) = (x + 1) ^ 2 ∧ 
  (2 * a + b) * (b - 2 * a) = b^2 - 4 * a^2 ∧ 
  (-a + b) * (a - b) = - (a - b)^2 ∧ 
  (x^2 - y) * (y^2 + x) ≠ (x + y)^2 :=
by 
  sorry

end polynomial_use_square_of_binomial_form_l1594_159458


namespace eval_expr_l1594_159488

theorem eval_expr :
  - (18 / 3 * 8 - 48 + 4 * 6) = -24 := by
  sorry

end eval_expr_l1594_159488


namespace train_length_72kmphr_9sec_180m_l1594_159414

/-- Given speed in km/hr and time in seconds, calculate the length of the train in meters -/
theorem train_length_72kmphr_9sec_180m : ∀ (speed_kmph : ℕ) (time_sec : ℕ),
  speed_kmph = 72 → time_sec = 9 → 
  (speed_kmph * 1000 / 3600) * time_sec = 180 :=
by
  intros speed_kmph time_sec h1 h2
  sorry

end train_length_72kmphr_9sec_180m_l1594_159414


namespace numbers_distance_one_neg_two_l1594_159401

theorem numbers_distance_one_neg_two (x : ℝ) (h : abs (x + 2) = 1) : x = -1 ∨ x = -3 := 
sorry

end numbers_distance_one_neg_two_l1594_159401


namespace minimum_value_of_f_l1594_159489

noncomputable def f (x : ℝ) : ℝ := (x^2 - 3) * Real.exp x

theorem minimum_value_of_f : ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ f x = -2 * Real.exp 1 :=
by
  sorry

end minimum_value_of_f_l1594_159489


namespace div_val_is_2_l1594_159410

theorem div_val_is_2 (x : ℤ) (h : 5 * x = 100) : x / 10 = 2 :=
by 
  sorry

end div_val_is_2_l1594_159410


namespace problem_proof_l1594_159465

theorem problem_proof (N : ℤ) (h : N / 5 = 4) : ((N - 10) * 3) - 18 = 12 :=
by
  -- proof goes here
  sorry

end problem_proof_l1594_159465


namespace probability_between_lines_l1594_159467

def line_l (x : ℝ) : ℝ := -2 * x + 8
def line_m (x : ℝ) : ℝ := -3 * x + 9

theorem probability_between_lines 
  (h1 : ∀ x > 0, line_l x ≥ 0) 
  (h2 : ∀ x > 0, line_m x ≥ 0) 
  (h3 : ∀ x > 0, line_l x < line_m x ∨ line_m x ≤ 0) : 
  (1 / 16 : ℝ) * 100 = 0.16 :=
by
  sorry

end probability_between_lines_l1594_159467


namespace minimum_value_frac_sum_l1594_159462

-- Define the statement problem C and proof outline skipping the steps
theorem minimum_value_frac_sum (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_sum : a + b = 2) :
  (1 / a + 1 / b) ≥ 2 :=
by
  -- Proof is to be constructed here
  sorry

end minimum_value_frac_sum_l1594_159462


namespace binary_to_base5_conversion_l1594_159449

theorem binary_to_base5_conversion : ∀ (b : ℕ), b = 1101 → (13 : ℕ) % 5 = 3 ∧ (13 / 5) % 5 = 2 → b = 1101 → (1101 : ℕ) = 13 → 13 = 23 :=
by
  sorry

end binary_to_base5_conversion_l1594_159449


namespace maximum_of_function_l1594_159413

theorem maximum_of_function :
  ∃ x y : ℝ, 
    (1/3 ≤ x ∧ x ≤ 2/5 ∧ 1/4 ≤ y ∧ y ≤ 5/12) ∧ 
    (∀ x' y' : ℝ, 1/3 ≤ x' ∧ x' ≤ 2/5 ∧ 1/4 ≤ y' ∧ y' ≤ 5/12 → 
                (xy / (x^2 + y^2) ≤ x' * y' / (x'^2 + y'^2))) ∧ 
    (xy / (x^2 + y^2) = 20 / 41) := 
sorry

end maximum_of_function_l1594_159413


namespace find_angle_B_l1594_159432

noncomputable def triangle_sides_and_angles 
(a b c : ℝ) (A B C : ℝ) : Prop :=
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

noncomputable def vectors_parallel 
(A B C a b c : ℝ) : Prop :=
  (Real.sin B - Real.sin A) / Real.sin C = (Real.sqrt 3 * a + c) / (a + b)

theorem find_angle_B (A B C a b c : ℝ)
  (h_triangle : triangle_sides_and_angles a b c A B C)
  (h_parallel : vectors_parallel A B C a b c) :
  B = 5 * Real.pi / 6 :=
sorry

end find_angle_B_l1594_159432


namespace greatest_xy_value_l1594_159464

theorem greatest_xy_value (x y : ℕ) (h1 : 7 * x + 4 * y = 140) (h2 : x > 0) (h3 : y > 0) : 
  xy ≤ 112 :=
by
  sorry

end greatest_xy_value_l1594_159464


namespace pizza_consumption_order_l1594_159498

noncomputable def amount_eaten (fraction: ℚ) (total: ℚ) := fraction * total

theorem pizza_consumption_order :
  let total := 1
  let samuel := (1 / 6 : ℚ)
  let teresa := (2 / 5 : ℚ)
  let uma := (1 / 4 : ℚ)
  let victor := total - (samuel + teresa + uma)
  let samuel_eaten := amount_eaten samuel 60
  let teresa_eaten := amount_eaten teresa 60
  let uma_eaten := amount_eaten uma 60
  let victor_eaten := amount_eaten victor 60
  (teresa_eaten > uma_eaten) 
  ∧ (uma_eaten > victor_eaten) 
  ∧ (victor_eaten > samuel_eaten) := 
by
  sorry

end pizza_consumption_order_l1594_159498


namespace most_reasonable_sampling_method_l1594_159452

-- Definitions based on the conditions in the problem:
def area_divided_into_200_plots : Prop := true
def plan_randomly_select_20_plots : Prop := true
def large_difference_in_plant_coverage : Prop := true
def goal_representative_sample_accurate_estimate : Prop := true

-- Main theorem statement
theorem most_reasonable_sampling_method
  (h1 : area_divided_into_200_plots)
  (h2 : plan_randomly_select_20_plots)
  (h3 : large_difference_in_plant_coverage)
  (h4 : goal_representative_sample_accurate_estimate) :
  Stratified_sampling := 
sorry

end most_reasonable_sampling_method_l1594_159452


namespace range_of_k_intersecting_hyperbola_l1594_159479

theorem range_of_k_intersecting_hyperbola :
  (∀ b : ℝ, ∃ x y : ℝ, y = k * x + b ∧ x^2 - 2 * y^2 = 1) →
  -Real.sqrt 2 / 2 < k ∧ k < Real.sqrt 2 / 2 :=
sorry

end range_of_k_intersecting_hyperbola_l1594_159479


namespace number_of_red_balloons_l1594_159478

-- Definitions for conditions
def balloons_total : ℕ := 85
def at_least_one_red (red blue : ℕ) : Prop := red ≥ 1 ∧ red + blue = balloons_total
def every_pair_has_blue (red blue : ℕ) : Prop := ∀ r1 r2, r1 < red → r2 < red → red = 1

-- Theorem to be proved
theorem number_of_red_balloons (red blue : ℕ) 
  (total : red + blue = balloons_total)
  (at_least_one : at_least_one_red red blue)
  (pair_condition : every_pair_has_blue red blue) : red = 1 :=
sorry

end number_of_red_balloons_l1594_159478


namespace Ron_eats_24_pickle_slices_l1594_159421

theorem Ron_eats_24_pickle_slices : 
  ∀ (pickle_slices_Sammy Tammy Ron : ℕ), 
    pickle_slices_Sammy = 15 → 
    Tammy = 2 * pickle_slices_Sammy → 
    Ron = Tammy - (20 * Tammy / 100) → 
    Ron = 24 := by
  intros pickle_slices_Sammy Tammy Ron h_sammy h_tammy h_ron
  sorry

end Ron_eats_24_pickle_slices_l1594_159421


namespace triangle_angle_side_cases_l1594_159476

theorem triangle_angle_side_cases
  (b c : ℝ) (B : ℝ)
  (hb : b = 3)
  (hc : c = 3 * Real.sqrt 3)
  (hB : B = Real.pi / 6) :
  (∃ A C a, A = Real.pi / 2 ∧ C = Real.pi / 3 ∧ a = Real.sqrt 21) ∨
  (∃ A C a, A = Real.pi / 6 ∧ C = 2 * Real.pi / 3 ∧ a = 3) :=
by
  sorry

end triangle_angle_side_cases_l1594_159476


namespace derivative_at_one_l1594_159436

noncomputable def f (x : ℝ) : ℝ := x / (x - 2)

theorem derivative_at_one : deriv f 1 = -2 :=
by 
  -- Here we would provide the proof that f'(1) = -2
  sorry

end derivative_at_one_l1594_159436


namespace reciprocal_difference_decreases_l1594_159428

theorem reciprocal_difference_decreases (n : ℕ) (hn : n > 0) : 
  (1 / (n : ℝ) - 1 / (n + 1 : ℝ)) < (1 / (n * n : ℝ)) :=
by 
  sorry

end reciprocal_difference_decreases_l1594_159428


namespace compare_inequalities_l1594_159424

theorem compare_inequalities (a b c π : ℝ) (h1 : a > π) (h2 : π > b) (h3 : b > 1) (h4 : 1 > c) (h5 : c > 0) 
  (x := a^(1 / π)) (y := Real.log b / Real.log π) (z := Real.log π / Real.log c) : x > y ∧ y > z := 
sorry

end compare_inequalities_l1594_159424


namespace intersection_point_not_on_x_3_l1594_159459

noncomputable def f (x : ℝ) : ℝ := (x^2 - 8*x + 15) / (3*x - 6)
noncomputable def g (x : ℝ) : ℝ := (-1/3 * x^2 + 6*x - 6) / (x - 2)

theorem intersection_point_not_on_x_3 : 
  ∃ x y : ℝ, (x ≠ 3) ∧ (f x = g x) ∧ (y = f x) ∧ (x = 11/3 ∧ y = -11/3) :=
by
  sorry

end intersection_point_not_on_x_3_l1594_159459


namespace fill_time_with_conditions_l1594_159444

-- Define rates as constants
def pipeA_rate := 1 / 10
def pipeB_rate := 1 / 6
def pipeC_rate := 1 / 5
def tarp_factor := 1 / 2
def leak_rate := 1 / 15

-- Define effective fill rate taking into account the tarp and leak
def effective_fill_rate := ((pipeA_rate + pipeB_rate + pipeC_rate) * tarp_factor) - leak_rate

-- Define the required time to fill the pool
def required_time := 1 / effective_fill_rate

theorem fill_time_with_conditions :
  required_time = 6 :=
by
  sorry

end fill_time_with_conditions_l1594_159444
