import Mathlib

namespace alex_needs_packs_of_buns_l2415_241546

-- Definitions (conditions)
def guests : ℕ := 10
def burgers_per_guest : ℕ := 3
def meat_eating_guests : ℕ := guests - 1
def bread_eating_ratios : ℕ := meat_eating_guests - 1
def buns_per_pack : ℕ := 8

-- Theorem (question == answer)
theorem alex_needs_packs_of_buns : 
  (burgers_per_guest * meat_eating_guests - burgers_per_guest) / buns_per_pack = 3 := by
  sorry

end alex_needs_packs_of_buns_l2415_241546


namespace how_many_bottles_did_maria_drink_l2415_241533

-- Define the conditions as variables and constants.
variable (x : ℕ)
def initial_bottles : ℕ := 14
def bought_bottles : ℕ := 45
def total_bottles_after_drinking_and_buying : ℕ := 51

-- The goal is to prove that Maria drank 8 bottles of water.
theorem how_many_bottles_did_maria_drink (h : initial_bottles - x + bought_bottles = total_bottles_after_drinking_and_buying) : x = 8 :=
by
  sorry

end how_many_bottles_did_maria_drink_l2415_241533


namespace elastic_band_radius_increase_l2415_241590

theorem elastic_band_radius_increase 
  (C1 C2 : ℝ) 
  (hC1 : C1 = 40) 
  (hC2 : C2 = 80) 
  (hC1_def : C1 = 2 * π * r1) 
  (hC2_def : C2 = 2 * π * r2) :
  r2 - r1 = 20 / π :=
by
  sorry

end elastic_band_radius_increase_l2415_241590


namespace solve_cos_theta_l2415_241513

def cos_theta_proof (v1 v2 : ℝ × ℝ) (θ : ℝ) : Prop :=
  let dot_product := (v1.1 * v2.1 + v1.2 * v2.2)
  let norm_v1 := Real.sqrt (v1.1 ^ 2 + v1.2 ^ 2)
  let norm_v2 := Real.sqrt (v2.1 ^ 2 + v2.2 ^ 2)
  let cos_theta := dot_product / (norm_v1 * norm_v2)
  cos_theta = 43 / Real.sqrt 2173

theorem solve_cos_theta :
  cos_theta_proof (4, 5) (2, 7) (43 / Real.sqrt 2173) :=
by
  sorry

end solve_cos_theta_l2415_241513


namespace product_mod_25_l2415_241550

theorem product_mod_25 (m : ℕ) (h : 0 ≤ m ∧ m < 25) : 
  43 * 67 * 92 % 25 = 2 :=
by
  sorry

end product_mod_25_l2415_241550


namespace Vanya_433_sum_l2415_241575

theorem Vanya_433_sum : 
  ∃ (A B : ℕ), 
  A + B = 91 
  ∧ (3 * A + 7 * B = 433) 
  ∧ (∃ (subsetA subsetB : Finset ℕ),
      (∀ x ∈ subsetA, x ∈ Finset.range (13 + 1))
      ∧ (∀ x ∈ subsetB, x ∈ Finset.range (13 + 1))
      ∧ subsetA ∩ subsetB = ∅
      ∧ subsetA ∪ subsetB = Finset.range (13 + 1)
      ∧ subsetA.card = 5
      ∧ subsetA.sum id = A
      ∧ subsetB.sum id = B) :=
by
  sorry

end Vanya_433_sum_l2415_241575


namespace lcm_36_105_l2415_241548

theorem lcm_36_105 : Int.lcm 36 105 = 1260 := by
  sorry

end lcm_36_105_l2415_241548


namespace solve_inequality_l2415_241566

theorem solve_inequality (x : ℝ) :
  (x - 5) / (x - 3)^2 < 0 ↔ x ∈ (Set.Iio 3 ∪ Set.Ioo 3 5) :=
by
  sorry

end solve_inequality_l2415_241566


namespace investment_rate_l2415_241505

theorem investment_rate (total : ℝ) (invested_at_3_percent : ℝ) (rate_3_percent : ℝ) 
                        (invested_at_5_percent : ℝ) (rate_5_percent : ℝ) 
                        (desired_income : ℝ) (remaining : ℝ) (additional_income : ℝ) (r : ℝ) : 
  total = 12000 ∧ 
  invested_at_3_percent = 5000 ∧ 
  rate_3_percent = 0.03 ∧ 
  invested_at_5_percent = 4000 ∧ 
  rate_5_percent = 0.05 ∧ 
  desired_income = 600 ∧ 
  remaining = total - invested_at_3_percent - invested_at_5_percent ∧ 
  additional_income = desired_income - (invested_at_3_percent * rate_3_percent + invested_at_5_percent * rate_5_percent) ∧ 
  r = (additional_income / remaining) * 100 → 
  r = 8.33 := 
by
  sorry

end investment_rate_l2415_241505


namespace evaluate_expression_l2415_241588

theorem evaluate_expression (a : ℚ) (h : a = 4 / 3) : (6 * a^2 - 8 * a + 3) * (3 * a - 4) = 0 :=
by
  rw [h]
  sorry

end evaluate_expression_l2415_241588


namespace simple_interest_l2415_241565

/-- Given:
    - Principal (P) = Rs. 80325
    - Rate (R) = 1% per annum
    - Time (T) = 5 years
    Prove that the total simple interest earned (SI) is Rs. 4016.25.
-/
theorem simple_interest (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ)
  (hP : P = 80325)
  (hR : R = 1)
  (hT : T = 5)
  (hSI : SI = P * R * T / 100) :
  SI = 4016.25 :=
by
  sorry

end simple_interest_l2415_241565


namespace smallest_n_l2415_241532

theorem smallest_n (n : ℕ) (k : ℕ) (a m : ℕ) 
  (h1 : 0 ≤ k)
  (h2 : k < n)
  (h3 : a ≡ k [MOD n])
  (h4 : m > 0) :
  (∀ a m, (∃ k, a = n * k + 5) -> (a^2 - 3*a + 1) ∣ (a^m + 3^m) → false) 
  → n = 11 := sorry

end smallest_n_l2415_241532


namespace dodecagon_diagonals_l2415_241504

def numDiagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem dodecagon_diagonals : numDiagonals 12 = 54 := by
  sorry

end dodecagon_diagonals_l2415_241504


namespace smallest_denominator_fraction_l2415_241549

theorem smallest_denominator_fraction 
  (p q : ℕ) (hp : 0 < p) (hq : 0 < q) 
  (h1 : 99 / 100 < p / q) 
  (h2 : p / q < 100 / 101) :
  p = 199 ∧ q = 201 := 
by 
  sorry

end smallest_denominator_fraction_l2415_241549


namespace total_balls_without_holes_l2415_241500

theorem total_balls_without_holes 
  (soccer_balls : ℕ) (soccer_balls_with_hole : ℕ)
  (basketballs : ℕ) (basketballs_with_hole : ℕ)
  (h1 : soccer_balls = 40)
  (h2 : soccer_balls_with_hole = 30)
  (h3 : basketballs = 15)
  (h4 : basketballs_with_hole = 7) :
  soccer_balls - soccer_balls_with_hole + (basketballs - basketballs_with_hole) = 18 := by
  sorry

end total_balls_without_holes_l2415_241500


namespace total_students_l2415_241543

theorem total_students (m d : ℕ) 
  (H1: 30 < m + d ∧ m + d < 40)
  (H2: ∃ r, r = 3 * m ∧ r = 5 * d) : 
  m + d = 32 := 
by
  sorry

end total_students_l2415_241543


namespace nearest_edge_of_picture_l2415_241570

theorem nearest_edge_of_picture
    (wall_width : ℝ) (picture_width : ℝ) (offset : ℝ) (x : ℝ)
    (hw : wall_width = 25) (hp : picture_width = 5) (ho : offset = 2) :
    x + (picture_width / 2) + offset = wall_width / 2 →
    x = 8 :=
by
  intros h
  sorry

end nearest_edge_of_picture_l2415_241570


namespace adult_tickets_count_l2415_241597

theorem adult_tickets_count (A C : ℕ) (h1 : A + C = 7) (h2 : 21 * A + 14 * C = 119) : A = 3 :=
sorry

end adult_tickets_count_l2415_241597


namespace arithmetic_sequence_third_term_l2415_241591

theorem arithmetic_sequence_third_term
  (a d : ℤ)
  (h_fifteenth_term : a + 14 * d = 15)
  (h_sixteenth_term : a + 15 * d = 21) :
  a + 2 * d = -57 :=
by
  sorry

end arithmetic_sequence_third_term_l2415_241591


namespace circle_problem_l2415_241539

theorem circle_problem 
  (x y : ℝ)
  (h : x^2 + 8*x - 10*y = 10 - y^2 + 6*x) :
  let a := -1
  let b := 5
  let r := 6
  a + b + r = 10 :=
by sorry

end circle_problem_l2415_241539


namespace eggs_distribution_l2415_241525

theorem eggs_distribution
  (total_eggs : ℕ)
  (eggs_per_adult : ℕ)
  (num_adults : ℕ)
  (num_girls : ℕ)
  (num_boys : ℕ)
  (eggs_per_girl : ℕ)
  (total_eggs_def : total_eggs = 3 * 12)
  (eggs_per_adult_def : eggs_per_adult = 3)
  (num_adults_def : num_adults = 3)
  (num_girls_def : num_girls = 7)
  (num_boys_def : num_boys = 10)
  (eggs_per_girl_def : eggs_per_girl = 1) :
  ∃ eggs_per_boy : ℕ, eggs_per_boy - eggs_per_girl = 1 :=
by {
  sorry
}

end eggs_distribution_l2415_241525


namespace triangle_side_split_l2415_241522

theorem triangle_side_split
  (PQ QR PR : ℝ)  -- Triangle sides
  (PS SR : ℝ)     -- Segments of PR divided by angle bisector
  (h_ratio : PQ / QR = 3 / 4)
  (h_sum : PR = 15)
  (h_PS_SR : PS / SR = 3 / 4)
  (h_PR_split : PS + SR = PR) :
  SR = 60 / 7 :=
by
  sorry

end triangle_side_split_l2415_241522


namespace tan_of_cos_alpha_l2415_241562

open Real

theorem tan_of_cos_alpha (α : ℝ) (h1 : cos α = 3 / 5) (h2 : -π < α ∧ α < 0) : tan α = -4 / 3 :=
sorry

end tan_of_cos_alpha_l2415_241562


namespace log_base_2_y_l2415_241502

theorem log_base_2_y (y : ℝ) (h : y = (Real.log 3 / Real.log 9) ^ Real.log 27 / Real.log 3) : 
  Real.log y = -3 :=
by
  sorry

end log_base_2_y_l2415_241502


namespace treasure_chest_total_value_l2415_241509

def base7_to_base10 (n : Nat) : Nat :=
  let rec convert (n acc base : Nat) : Nat :=
    if n = 0 then acc
    else convert (n / 10) (acc + (n % 10) * base) (base * 7)
  convert n 0 1

theorem treasure_chest_total_value :
  base7_to_base10 5346 + base7_to_base10 6521 + base7_to_base10 320 = 4305 :=
by
  sorry

end treasure_chest_total_value_l2415_241509


namespace deductive_reasoning_correctness_l2415_241573

theorem deductive_reasoning_correctness (major_premise minor_premise form_of_reasoning correct : Prop) 
  (h : major_premise ∧ minor_premise ∧ form_of_reasoning) : correct :=
  sorry

end deductive_reasoning_correctness_l2415_241573


namespace area_of_ABCD_l2415_241581

theorem area_of_ABCD (x : ℕ) (h1 : 0 < x)
  (h2 : 10 * x = 160) : 4 * x ^ 2 = 1024 := by
  sorry

end area_of_ABCD_l2415_241581


namespace _l2415_241506

-- Here we define our conditions

def parabola (x y : ℝ) := y^2 = 8 * x

def focus : ℝ × ℝ := (2, 0)

def point_on_parabola (y : ℝ) : ℝ × ℝ := (4, y)

example (y_P : ℝ) (hP : parabola 4 y_P) :
  dist (point_on_parabola y_P) focus = 6 := by
  -- Since we only need the theorem statement, we finish with sorry
  sorry

end _l2415_241506


namespace sum_of_interior_edges_l2415_241530

def frame_width : ℝ := 1
def outer_length : ℝ := 5
def frame_area : ℝ := 18
def inner_length1 : ℝ := outer_length - 2 * frame_width

/-- Given conditions and required to prove:
1. The frame is made of one-inch-wide pieces of wood.
2. The area of just the frame is 18 square inches.
3. One of the outer edges of the frame is 5 inches long.
Prove: The sum of the lengths of the four interior edges is 14 inches.
-/
theorem sum_of_interior_edges (inner_length2 : ℝ) 
  (h1 : (outer_length * (inner_length2 + 2) - inner_length1 * inner_length2) = frame_area)
  (h2 : (inner_length2 - 2) / 2 = 1) : 
  inner_length1 + inner_length1 + inner_length2 + inner_length2 = 14 :=
by
  sorry

end sum_of_interior_edges_l2415_241530


namespace order_magnitudes_ln_subtraction_l2415_241537

noncomputable def ln (x : ℝ) : ℝ := Real.log x -- Assuming the natural logarithm definition for real numbers

theorem order_magnitudes_ln_subtraction :
  (ln (3/2) - (3/2)) > (ln 3 - 3) ∧ 
  (ln 3 - 3) > (ln π - π) :=
sorry

end order_magnitudes_ln_subtraction_l2415_241537


namespace proof_problem_l2415_241571

def a : ℕ := 5^2
def b : ℕ := a^4

theorem proof_problem : b = 390625 := 
by 
  sorry

end proof_problem_l2415_241571


namespace ralph_tv_hours_l2415_241516

theorem ralph_tv_hours :
  (4 * 5 + 6 * 2) = 32 :=
by
  sorry

end ralph_tv_hours_l2415_241516


namespace power_identity_l2415_241511

theorem power_identity (x a b : ℝ) (h1 : x^a = 2) (h2 : x^b = 3) : x^(3 * a + 2 * b) = 72 := 
  sorry

end power_identity_l2415_241511


namespace find_a_for_symmetric_and_parallel_lines_l2415_241544

theorem find_a_for_symmetric_and_parallel_lines :
  ∃ (a : ℝ), (∀ (x y : ℝ), y = a * x + 3 ↔ x = a * y + 3) ∧ (∀ (x y : ℝ), x + 2 * y - 1 = 0 ↔ x = a * y + 3) ∧ ∃ (a : ℝ), a = -2 := 
sorry

end find_a_for_symmetric_and_parallel_lines_l2415_241544


namespace resistor_problem_l2415_241540

theorem resistor_problem (R : ℝ)
  (initial_resistance : ℝ := 3 * R)
  (parallel_resistance : ℝ := R / 3)
  (resistance_change : ℝ := initial_resistance - parallel_resistance)
  (condition : resistance_change = 10) : 
  R = 3.75 := by
  sorry

end resistor_problem_l2415_241540


namespace plates_not_adj_l2415_241586

def num_ways_arrange_plates (blue red green orange : ℕ) (no_adj : Bool) : ℕ :=
  -- assuming this function calculates the desired number of arrangements
  sorry

theorem plates_not_adj (h : num_ways_arrange_plates 6 2 2 1 true = 1568) : 
  num_ways_arrange_plates 6 2 2 1 true = 1568 :=
  by exact h -- using the hypothesis directly for the theorem statement

end plates_not_adj_l2415_241586


namespace combined_height_after_1_year_l2415_241520

def initial_heights : ℕ := 200 + 150 + 250
def spring_and_summer_growth_A : ℕ := (6 * 4 / 2) * 50
def spring_and_summer_growth_B : ℕ := (6 * 4 / 3) * 70
def spring_and_summer_growth_C : ℕ := (6 * 4 / 4) * 90
def autumn_and_winter_growth_A : ℕ := (6 * 4 / 2) * 25
def autumn_and_winter_growth_B : ℕ := (6 * 4 / 3) * 35
def autumn_and_winter_growth_C : ℕ := (6 * 4 / 4) * 45

def total_growth_A : ℕ := spring_and_summer_growth_A + autumn_and_winter_growth_A
def total_growth_B : ℕ := spring_and_summer_growth_B + autumn_and_winter_growth_B
def total_growth_C : ℕ := spring_and_summer_growth_C + autumn_and_winter_growth_C

def total_growth : ℕ := total_growth_A + total_growth_B + total_growth_C

def combined_height : ℕ := initial_heights + total_growth

theorem combined_height_after_1_year : combined_height = 3150 := by
  sorry

end combined_height_after_1_year_l2415_241520


namespace recurring_decimal_as_fraction_l2415_241576

theorem recurring_decimal_as_fraction :
  0.53 + (247 / 999) * 0.001 = 53171 / 99900 :=
by
  sorry

end recurring_decimal_as_fraction_l2415_241576


namespace even_marked_squares_9x9_l2415_241598

open Nat

theorem even_marked_squares_9x9 :
  let n := 9
  let total_squares := n * n
  let odd_rows_columns := [1, 3, 5, 7, 9]
  let odd_squares := odd_rows_columns.length * odd_rows_columns.length
  total_squares - odd_squares = 56 :=
by
  sorry

end even_marked_squares_9x9_l2415_241598


namespace abs_neg_five_l2415_241559

theorem abs_neg_five : abs (-5) = 5 :=
by
  sorry

end abs_neg_five_l2415_241559


namespace clock_ticks_12_times_l2415_241547

theorem clock_ticks_12_times (t1 t2 : ℕ) (d1 d2 : ℕ) (h1 : t1 = 6) (h2 : d1 = 40) (h3 : d2 = 88) : t2 = 12 := by
  sorry

end clock_ticks_12_times_l2415_241547


namespace find_cupcakes_l2415_241507

def total_students : ℕ := 20
def treats_per_student : ℕ := 4
def cookies : ℕ := 20
def brownies : ℕ := 35
def total_treats : ℕ := total_students * treats_per_student
def cupcakes : ℕ := total_treats - (cookies + brownies)

theorem find_cupcakes : cupcakes = 25 := by
  sorry

end find_cupcakes_l2415_241507


namespace remaining_hours_needed_l2415_241517

noncomputable
def hours_needed_to_finish (x : ℚ) : Prop :=
  (1/5 : ℚ) * (2 + x) + (1/8 : ℚ) * x = 1

theorem remaining_hours_needed :
  ∃ x : ℚ, hours_needed_to_finish x ∧ x = 24/13 :=
by
  use 24/13
  sorry

end remaining_hours_needed_l2415_241517


namespace DanAgeIs12_l2415_241574

def DanPresentAge (x : ℕ) : Prop :=
  (x + 18 = 5 * (x - 6))

theorem DanAgeIs12 : ∃ x : ℕ, DanPresentAge x ∧ x = 12 :=
by
  use 12
  unfold DanPresentAge
  sorry

end DanAgeIs12_l2415_241574


namespace distance_point_to_vertical_line_l2415_241515

/-- The distance from a point to a vertical line equals the absolute difference in the x-coordinates. -/
theorem distance_point_to_vertical_line (x1 y1 x2 : ℝ) (h_line : x2 = -2) (h_point : (x1, y1) = (1, 2)) :
  abs (x1 - x2) = 3 :=
by
  -- Place proof here
  sorry

end distance_point_to_vertical_line_l2415_241515


namespace green_apples_more_than_red_apples_l2415_241563

theorem green_apples_more_than_red_apples 
    (total_apples : ℕ)
    (red_apples : ℕ)
    (total_apples_eq : total_apples = 44)
    (red_apples_eq : red_apples = 16) :
    (total_apples - red_apples) - red_apples = 12 :=
by
  sorry

end green_apples_more_than_red_apples_l2415_241563


namespace problem_statement_l2415_241524

noncomputable def alpha := 3 + Real.sqrt 8
noncomputable def beta := 3 - Real.sqrt 8
noncomputable def x := alpha ^ 1000
noncomputable def n := Int.floor x
noncomputable def f := x - n

theorem problem_statement : x * (1 - f) = 1 :=
by sorry

end problem_statement_l2415_241524


namespace ice_cream_amount_l2415_241589

/-- Given: 
    Amount of ice cream eaten on Friday night: 3.25 pints
    Total amount of ice cream eaten over both nights: 3.5 pints
    Prove: 
    Amount of ice cream eaten on Saturday night = 0.25 pints -/
theorem ice_cream_amount (friday_night saturday_night total : ℝ) (h_friday : friday_night = 3.25) (h_total : total = 3.5) : 
  saturday_night = total - friday_night → saturday_night = 0.25 :=
by
  intro h
  rw [h_total, h_friday] at h
  simp [h]
  sorry

end ice_cream_amount_l2415_241589


namespace product_of_last_two_digits_l2415_241593

theorem product_of_last_two_digits (A B : ℕ) (h1 : A + B = 14) (h2 : B = 0 ∨ B = 5) : A * B = 45 :=
sorry

end product_of_last_two_digits_l2415_241593


namespace grasshopper_jump_distance_l2415_241512

variable (F G M : ℕ) -- F for frog's jump, G for grasshopper's jump, M for mouse's jump

theorem grasshopper_jump_distance (h1 : F = G + 39) (h2 : M = F - 94) (h3 : F = 58) : G = 19 := 
by
  sorry

end grasshopper_jump_distance_l2415_241512


namespace find_d_minus_c_l2415_241528

theorem find_d_minus_c (c d x : ℝ) (h : c ≤ 3 * x - 2 ∧ 3 * x - 2 ≤ d) : (d - c = 45) :=
  sorry

end find_d_minus_c_l2415_241528


namespace necessary_condition_not_sufficient_condition_l2415_241561

variable (a b : ℝ)
def isPureImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0
def proposition_p (a : ℝ) : Prop := a = 0

theorem necessary_condition (a b : ℝ) (z : ℂ) (h : z = ⟨a, b⟩) : isPureImaginary z → proposition_p a := sorry

theorem not_sufficient_condition (a b : ℝ) (z : ℂ) (h : z = ⟨a, b⟩) : proposition_p a → ¬isPureImaginary z := sorry

end necessary_condition_not_sufficient_condition_l2415_241561


namespace Mike_siblings_l2415_241531

-- Define the types for EyeColor, HairColor and Sport
inductive EyeColor
| Blue
| Green

inductive HairColor
| Black
| Blonde

inductive Sport
| Soccer
| Basketball

-- Define the Child structure
structure Child where
  name : String
  eyeColor : EyeColor
  hairColor : HairColor
  favoriteSport : Sport

-- Define all the children based on the given conditions
def Lily : Child := { name := "Lily", eyeColor := EyeColor.Green, hairColor := HairColor.Black, favoriteSport := Sport.Soccer }
def Mike : Child := { name := "Mike", eyeColor := EyeColor.Blue, hairColor := HairColor.Blonde, favoriteSport := Sport.Basketball }
def Oliver : Child := { name := "Oliver", eyeColor := EyeColor.Blue, hairColor := HairColor.Black, favoriteSport := Sport.Soccer }
def Emma : Child := { name := "Emma", eyeColor := EyeColor.Green, hairColor := HairColor.Blonde, favoriteSport := Sport.Basketball }
def Jacob : Child := { name := "Jacob", eyeColor := EyeColor.Blue, hairColor := HairColor.Blonde, favoriteSport := Sport.Soccer }
def Sophia : Child := { name := "Sophia", eyeColor := EyeColor.Green, hairColor := HairColor.Blonde, favoriteSport := Sport.Soccer }

-- Siblings relation
def areSiblings (c1 c2 c3 : Child) : Prop :=
  (c1.eyeColor = c2.eyeColor ∨ c1.hairColor = c2.hairColor ∨ c1.favoriteSport = c2.favoriteSport) ∧
  (c1.eyeColor = c3.eyeColor ∨ c1.hairColor = c3.hairColor ∨ c1.favoriteSport = c3.favoriteSport) ∧
  (c2.eyeColor = c3.eyeColor ∨ c2.hairColor = c3.hairColor ∨ c2.favoriteSport = c3.favoriteSport)

-- The proof statement
theorem Mike_siblings : areSiblings Mike Emma Jacob := by
  -- Proof must be provided here
  sorry

end Mike_siblings_l2415_241531


namespace donation_ratio_l2415_241518

theorem donation_ratio (D1 : ℝ) (D1_value : D1 = 10)
  (total_donation : D1 + D1 * 2 + D1 * 4 + D1 * 8 + D1 * 16 = 310) : 
  2 = 2 :=
by
  sorry

end donation_ratio_l2415_241518


namespace find_age_of_D_l2415_241526

theorem find_age_of_D
(Eq1 : a + b + c + d = 108)
(Eq2 : a - b = 12)
(Eq3 : c - (a - 34) = 3 * (d - (a - 34)))
: d = 13 := 
sorry

end find_age_of_D_l2415_241526


namespace geometric_sequence_a3_l2415_241577

theorem geometric_sequence_a3 (a : ℕ → ℝ) (r : ℝ)
  (h1 : a 1 = 1)
  (h5 : a 5 = 4)
  (geo_seq : ∀ n, a n = a 1 * r ^ (n - 1)) :
  a 3 = 2 :=
by
  sorry

end geometric_sequence_a3_l2415_241577


namespace diamond_value_l2415_241564

def diamond (x y : ℕ) : ℕ := 4 * x + 6 * y

theorem diamond_value : diamond 3 4 = 36 := by
  -- Given condition: x ♢ y = 4x + 6y
  -- To prove: (diamond 3 4) = 36
  sorry

end diamond_value_l2415_241564


namespace expected_number_of_heads_after_flips_l2415_241552

theorem expected_number_of_heads_after_flips :
  let p_heads_after_tosses : ℚ := (1/3) + (2/9) + (4/27) + (8/81)
  let expected_heads : ℚ := 100 * p_heads_after_tosses
  expected_heads = 6500 / 81 :=
by
  let p_heads_after_tosses : ℚ := (1/3) + (2/9) + (4/27) + (8/81)
  let expected_heads : ℚ := 100 * p_heads_after_tosses
  show expected_heads = (6500 / 81)
  sorry

end expected_number_of_heads_after_flips_l2415_241552


namespace part_I_solution_set_part_II_range_of_a_l2415_241555

-- Definitions
def f (x : ℝ) (a : ℝ) := |x - 1| + |a * x + 1|
def g (x : ℝ) := |x + 1| + 2

-- Part I: Prove the solution set of the inequality f(x) < 2 when a = 1/2
theorem part_I_solution_set (x : ℝ) : f x (1/2 : ℝ) < 2 ↔ 0 < x ∧ x < (4/3 : ℝ) :=
sorry
  
-- Part II: Prove the range of a such that (0, 1] ⊆ {x | f x a ≤ g x}
theorem part_II_range_of_a (a : ℝ) : (∀ x, 0 < x ∧ x ≤ 1 → f x a ≤ g x) ↔ -5 ≤ a ∧ a ≤ 3 :=
sorry

end part_I_solution_set_part_II_range_of_a_l2415_241555


namespace find_m_l2415_241595

theorem find_m (a b m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 ^ a = m) (h4 : 3 ^ b = m) (h5 : 2 * a * b = a + b) : m = Real.sqrt 6 :=
sorry

end find_m_l2415_241595


namespace perpendicular_lines_m_l2415_241585

theorem perpendicular_lines_m (m : ℝ) : 
    (∀ x y : ℝ, x - 2 * y + 5 = 0 → 
                2 * x + m * y - 6 = 0 → 
                (1 / 2) * (-2 / m) = -1) → 
    m = 1 :=
by
  intros
  -- proof goes here
  sorry

end perpendicular_lines_m_l2415_241585


namespace factorize_problem1_factorize_problem2_l2415_241594

-- Problem 1: Prove that 6p^3q - 10p^2 == 2p^2 * (3pq - 5)
theorem factorize_problem1 (p q : ℝ) : 
    6 * p^3 * q - 10 * p^2 = 2 * p^2 * (3 * p * q - 5) := 
by 
    sorry

-- Problem 2: Prove that a^4 - 8a^2 + 16 == (a-2)^2 * (a+2)^2
theorem factorize_problem2 (a : ℝ) : 
    a^4 - 8 * a^2 + 16 = (a - 2)^2 * (a + 2)^2 := 
by 
    sorry

end factorize_problem1_factorize_problem2_l2415_241594


namespace determinant_matrix_3x3_l2415_241551

theorem determinant_matrix_3x3 :
  Matrix.det ![![3, 1, -2], ![8, 5, -4], ![1, 3, 6]] = 140 :=
by
  sorry

end determinant_matrix_3x3_l2415_241551


namespace herring_invariant_l2415_241592

/--
A circle is divided into six sectors. Each sector contains one herring. 
In one move, you can move any two herrings in adjacent sectors moving them in opposite directions.
Prove that it is impossible to gather all herrings into one sector using these operations.
-/
theorem herring_invariant (herring : Fin 6 → Bool) :
  ¬ ∃ i : Fin 6, ∀ j : Fin 6, herring j = herring i := 
sorry

end herring_invariant_l2415_241592


namespace guests_accommodation_l2415_241508

open Nat

theorem guests_accommodation :
  let guests := 15
  let rooms := 4
  (4 ^ 15 - 4 * 3 ^ 15 + 6 * 2 ^ 15 - 4 = 4 ^ 15 - 4 * 3 ^ 15 + 6 * 2 ^ 15 - 4) :=
by
  sorry

end guests_accommodation_l2415_241508


namespace probability_prime_or_odd_ball_l2415_241596

def isPrime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def isOdd (n : ℕ) : Prop :=
  n % 2 = 1

def isPrimeOrOdd (n : ℕ) : Prop :=
  isPrime n ∨ isOdd n

theorem probability_prime_or_odd_ball :
  (1+2+3+5+7)/8 = 5/8 := by
  sorry

end probability_prime_or_odd_ball_l2415_241596


namespace ratio_books_purchased_l2415_241503

-- Definitions based on the conditions
def books_last_year : ℕ := 50
def books_before_purchase : ℕ := 100
def books_now : ℕ := 300

-- Let x be the multiple of the books purchased this year
def multiple_books_purchased_this_year (x : ℕ) : Prop :=
  books_now = books_before_purchase + books_last_year + books_last_year * x

-- Prove the ratio is 3:1
theorem ratio_books_purchased (x : ℕ) (h : multiple_books_purchased_this_year x) : x = 3 :=
  by sorry

end ratio_books_purchased_l2415_241503


namespace Clarissa_photos_needed_l2415_241514

theorem Clarissa_photos_needed :
  (7 + 10 + 9 <= 40) → 40 - (7 + 10 + 9) = 14 :=
by
  sorry

end Clarissa_photos_needed_l2415_241514


namespace B_contribution_l2415_241578

theorem B_contribution (A_capital : ℝ) (A_time : ℝ) (B_time : ℝ) (total_profit : ℝ) (A_profit_share : ℝ) (B_contributed : ℝ) :
  A_capital * A_time / (A_capital * A_time + B_contributed * B_time) = A_profit_share / total_profit →
  B_contributed = 6000 :=
by
  intro h
  sorry

end B_contribution_l2415_241578


namespace math_problem_l2415_241538

theorem math_problem
  (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
  (h : a / (b - c) + b / (c - a) + c / (a - b) = 1) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 1 / (b - c) + 1 / (c - a) + 1 / (a - b) :=
  sorry

end math_problem_l2415_241538


namespace multiple_of_5_multiple_of_10_not_multiple_of_20_not_multiple_of_40_l2415_241579

def x : ℤ := 50 + 100 + 140 + 180 + 320 + 400 + 5000

theorem multiple_of_5 : x % 5 = 0 := by 
  sorry

theorem multiple_of_10 : x % 10 = 0 := by 
  sorry

theorem not_multiple_of_20 : x % 20 ≠ 0 := by 
  sorry

theorem not_multiple_of_40 : x % 40 ≠ 0 := by 
  sorry

end multiple_of_5_multiple_of_10_not_multiple_of_20_not_multiple_of_40_l2415_241579


namespace product_sum_correct_l2415_241587

def product_sum_eq : Prop :=
  let a := 4 * 10^6
  let b := 8 * 10^6
  (a * b + 2 * 10^13) = 5.2 * 10^13

theorem product_sum_correct : product_sum_eq :=
by
  sorry

end product_sum_correct_l2415_241587


namespace find_a_l2415_241557

def operation (a b : ℤ) : ℤ := 2 * a - b * b

theorem find_a (a : ℤ) : operation a 3 = 15 → a = 12 := by
  sorry

end find_a_l2415_241557


namespace polyhedron_equation_l2415_241572

variables (V E F H T : ℕ)

-- Euler's formula for convex polyhedra
axiom euler_formula : V - E + F = 2
-- Number of faces is 50, and each face is either a triangle or a hexagon
axiom faces_count : F = 50
-- At each vertex, 3 triangles and 2 hexagons meet
axiom triangles_meeting : T = 3
axiom hexagons_meeting : H = 2

-- Prove that 100H + 10T + V = 230
theorem polyhedron_equation : 100 * H + 10 * T + V = 230 :=
  sorry

end polyhedron_equation_l2415_241572


namespace value_of_a_l2415_241560

def star (a b : ℤ) : ℤ := 3 * a - b^3

theorem value_of_a : ∃ a : ℤ, star a 3 = 63 ∧ a = 30 := by
  sorry

end value_of_a_l2415_241560


namespace total_disks_in_bag_l2415_241541

/-- Given that the number of blue disks b, yellow disks y, and green disks g are in the ratio 3:7:8,
    and there are 30 more green disks than blue disks (g = b + 30),
    prove that the total number of disks is 108. -/
theorem total_disks_in_bag (b y g : ℕ) (h1 : 3 * y = 7 * b) (h2 : 8 * y = 7 * g) (h3 : g = b + 30) :
  b + y + g = 108 := by
  sorry

end total_disks_in_bag_l2415_241541


namespace value_of_a8_l2415_241545

variable (a : ℕ → ℝ) (a_1 : a 1 = 2) (common_sum : ℝ) (h_sum : common_sum = 5)
variable (equal_sum_sequence : ∀ n, a (n + 1) + a n = common_sum)

theorem value_of_a8 : a 8 = 3 :=
sorry

end value_of_a8_l2415_241545


namespace snowdrift_ratio_l2415_241534

theorem snowdrift_ratio
  (depth_first_day : ℕ := 20)
  (depth_second_day : ℕ)
  (h1 : depth_second_day + 24 = 34)
  (h2 : depth_second_day = 10) :
  depth_second_day / depth_first_day = 1 / 2 := by
  sorry

end snowdrift_ratio_l2415_241534


namespace units_digit_of_sum_is_three_l2415_241583

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def units_digit (n : ℕ) : ℕ :=
  n % 10

def sum_of_factorials : ℕ :=
  (List.range 10).map factorial |>.sum

def power_of_ten (n : ℕ) : ℕ :=
  10^n

theorem units_digit_of_sum_is_three : 
  units_digit (sum_of_factorials + power_of_ten 3) = 3 := by
  sorry

end units_digit_of_sum_is_three_l2415_241583


namespace linear_equation_value_l2415_241529

-- Define the conditions of the equation
def equation_is_linear (m : ℝ) : Prop :=
  |m| = 1 ∧ m - 1 ≠ 0

-- Prove the equivalence statement
theorem linear_equation_value (m : ℝ) (h : equation_is_linear m) : m = -1 := 
sorry

end linear_equation_value_l2415_241529


namespace number_of_M_subsets_l2415_241535

def P : Set ℤ := {0, 1, 2}
def Q : Set ℤ := {0, 2, 4}

theorem number_of_M_subsets (M : Set ℤ) (hP : M ⊆ P) (hQ : M ⊆ Q) : 
  ∃ n : ℕ, n = 4 :=
by
  sorry

end number_of_M_subsets_l2415_241535


namespace joel_average_speed_l2415_241580

theorem joel_average_speed :
  let start_time := (8, 50)
  let end_time := (14, 35)
  let total_distance := 234
  let total_time := (14 - 8) + (35 - 50) / 60
  ∀ start_time end_time total_distance,
    (start_time = (8, 50)) →
    (end_time = (14, 35)) →
    total_distance = 234 →
    (total_time = (14 - 8) + (35 - 50) / 60) →
    total_distance / total_time = 41 :=
by
  sorry

end joel_average_speed_l2415_241580


namespace power_function_condition_direct_proportionality_condition_inverse_proportionality_condition_l2415_241599

theorem power_function_condition (m : ℝ) : m^2 + 2 * m = 1 ↔ m = -1 + Real.sqrt 2 ∨ m = -1 - Real.sqrt 2 :=
by sorry

theorem direct_proportionality_condition (m : ℝ) : (m^2 + m - 1 = 1 ∧ m^2 + 3 * m ≠ 0) ↔ m = 1 :=
by sorry

theorem inverse_proportionality_condition (m : ℝ) : (m^2 + m - 1 = -1 ∧ m^2 + 3 * m ≠ 0) ↔ m = -1 :=
by sorry

end power_function_condition_direct_proportionality_condition_inverse_proportionality_condition_l2415_241599


namespace binomial_10_2_equals_45_l2415_241523

open Nat

theorem binomial_10_2_equals_45 : Nat.choose 10 2 = 45 := 
by
  sorry

end binomial_10_2_equals_45_l2415_241523


namespace average_of_four_l2415_241569

variable {r s t u : ℝ}

theorem average_of_four (h : (5 / 2) * (r + s + t + u) = 20) : (r + s + t + u) / 4 = 2 := 
by 
  sorry

end average_of_four_l2415_241569


namespace sum_of_powers_twice_square_l2415_241553

theorem sum_of_powers_twice_square (x y : ℤ) : 
  ∃ z : ℤ, x^4 + y^4 + (x + y)^4 = 2 * z^2 := by
  let z := x^2 + x * y + y^2
  use z
  sorry

end sum_of_powers_twice_square_l2415_241553


namespace inequality_solution_inequality_solution_b_monotonic_increasing_monotonic_decreasing_l2415_241501

variable (a : ℝ) (x : ℝ)
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.sqrt (x^2 + 1) - a * x

-- Part (1)
theorem inequality_solution (a : ℝ) (h1 : 0 < a ∧ a < 1) : (0 ≤ x ∧ x ≤ 2*a / (1 - a^2)) → (f x a ≤ 1) :=
sorry

theorem inequality_solution_b (a : ℝ) (h2 : a ≥ 1) : (0 ≤ x) → (f x a ≤ 1) :=
sorry

-- Part (2)
theorem monotonic_increasing (a : ℝ) (h3 : a ≤ 0) (x1 x2 : ℝ) (hx : 0 ≤ x1 ∧ 0 ≤ x2 ∧ x1 < x2) : f x1 a ≤ f x2 a :=
sorry

theorem monotonic_decreasing (a : ℝ) (h4 : a ≥ 1) (x1 x2 : ℝ) (hx : 0 ≤ x1 ∧ 0 ≤ x2 ∧ x1 < x2) : f x1 a ≥ f x2 a :=
sorry

end inequality_solution_inequality_solution_b_monotonic_increasing_monotonic_decreasing_l2415_241501


namespace orange_ribbons_count_l2415_241556

variable (total_ribbons : ℕ)
variable (orange_ribbons : ℚ)

-- Definitions of the given conditions
def yellow_fraction := (1 : ℚ) / 4
def purple_fraction := (1 : ℚ) / 3
def orange_fraction := (1 : ℚ) / 6
def black_ribbons := 40
def black_fraction := (1 : ℚ) / 4

-- Using the given and derived conditions
theorem orange_ribbons_count
  (hy : yellow_fraction = 1 / 4)
  (hp : purple_fraction = 1 / 3)
  (ho : orange_fraction = 1 / 6)
  (hb : black_ribbons = 40)
  (hbf : black_fraction = 1 / 4)
  (total_eq : total_ribbons = black_ribbons * 4) :
  orange_ribbons = total_ribbons * orange_fraction := by
  -- Proof omitted
  sorry

end orange_ribbons_count_l2415_241556


namespace tan_alpha_implies_fraction_l2415_241521

theorem tan_alpha_implies_fraction (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 / 3 :=
sorry

end tan_alpha_implies_fraction_l2415_241521


namespace initial_men_checking_exam_papers_l2415_241527

theorem initial_men_checking_exam_papers :
  ∀ (M : ℕ),
  (M * 8 * 5 = (1/2 : ℝ) * (2 * 20 * 8)) → M = 4 :=
by
  sorry

end initial_men_checking_exam_papers_l2415_241527


namespace length_of_AB_l2415_241567

theorem length_of_AB
  (height h : ℝ)
  (AB CD : ℝ)
  (ratio_AB_ADC : (1/2 * AB * h) / (1/2 * CD * h) = 5/4)
  (sum_AB_CD : AB + CD = 300) :
  AB = 166.67 :=
by
  -- The proof goes here.
  sorry

end length_of_AB_l2415_241567


namespace A_and_C_together_2_hours_l2415_241584

theorem A_and_C_together_2_hours (A_rate B_rate C_rate : ℝ) (hA : A_rate = 1 / 5)
  (hBC : B_rate + C_rate = 1 / 3) (hB : B_rate = 1 / 30) : A_rate + C_rate = 1 / 2 := 
by
  sorry

end A_and_C_together_2_hours_l2415_241584


namespace smallest_y_square_factor_l2415_241542

theorem smallest_y_square_factor (y n : ℕ) (h₀ : y = 10) 
  (h₁ : ∀ m : ℕ, ∃ k : ℕ, k * k = m * y)
  (h₂ : ∀ (y' : ℕ), (∀ m : ℕ, ∃ k : ℕ, k * k = m * y') → y ≤ y') : 
  n = 10 :=
by sorry

end smallest_y_square_factor_l2415_241542


namespace range_of_x_l2415_241519

theorem range_of_x (a : ℕ → ℝ) (x : ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_a1 : a 1 = 1)
  (h_condition : ∀ n, a (n + 1)^2 + a n^2 < (5 / 2) * a (n + 1) * a n)
  (h_a2 : a 2 = 3 / 2)
  (h_a3 : a 3 = x)
  (h_a4 : a 4 = 4) : 2 < x ∧ x < 3 := by
  sorry

end range_of_x_l2415_241519


namespace first_machine_copies_per_minute_l2415_241554

theorem first_machine_copies_per_minute
    (x : ℕ)
    (h1 : ∀ (x : ℕ), 30 * x + 30 * 55 = 2850) :
  x = 40 :=
by
  sorry

end first_machine_copies_per_minute_l2415_241554


namespace zamena_inequalities_l2415_241582

theorem zamena_inequalities :
  ∃ (M E H A : ℕ), 
    M ≠ E ∧ M ≠ H ∧ M ≠ A ∧ 
    E ≠ H ∧ E ≠ A ∧ H ≠ A ∧
    (∃ Z, Z = 5 ∧ -- Assume Z is 5 based on the conclusion
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    1 ≤ A ∧ A ≤ 5 ∧
    3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A ∧
    -- ZAMENA evaluates to 541234
    Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234) :=
sorry

end zamena_inequalities_l2415_241582


namespace average_of_numbers_not_1380_l2415_241558

def numbers : List ℤ := [1200, 1300, 1400, 1520, 1530, 1200]

theorem average_of_numbers_not_1380 :
  let s := numbers.sum
  let n := numbers.length
  n > 0 → (s / n : ℚ) ≠ 1380 := by
  sorry

end average_of_numbers_not_1380_l2415_241558


namespace greatest_negative_root_l2415_241568

noncomputable def sine (x : ℝ) : ℝ := Real.sin (Real.pi * x)
noncomputable def cosine (x : ℝ) : ℝ := Real.cos (2 * Real.pi * x)

theorem greatest_negative_root :
  ∀ (x : ℝ), (x < 0 ∧ (sine x - cosine x) / ((sine x + 1)^2 + (Real.cos (Real.pi * x))^2) = 0) → 
    x ≤ -7/6 :=
by
  sorry

end greatest_negative_root_l2415_241568


namespace chocolate_bar_cost_l2415_241510

-- Definitions based on the conditions given in the problem.
def total_bars : ℕ := 7
def remaining_bars : ℕ := 4
def total_money : ℚ := 9
def bars_sold : ℕ := total_bars - remaining_bars
def cost_per_bar := total_money / bars_sold

-- The theorem that needs to be proven.
theorem chocolate_bar_cost : cost_per_bar = 3 := by
  -- proof placeholder
  sorry

end chocolate_bar_cost_l2415_241510


namespace Sara_Jim_equal_savings_l2415_241536

theorem Sara_Jim_equal_savings:
  ∃ (w : ℕ), (∃ (sara_saved jim_saved : ℕ),
  sara_saved = 4100 + 10 * w ∧
  jim_saved = 15 * w ∧
  sara_saved = jim_saved) → w = 820 :=
by
  sorry

end Sara_Jim_equal_savings_l2415_241536
