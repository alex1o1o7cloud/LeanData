import Mathlib

namespace cotton_equals_iron_l1461_146143

theorem cotton_equals_iron (cotton_weight : ℝ) (iron_weight : ℝ)
  (h_cotton : cotton_weight = 1)
  (h_iron : iron_weight = 4) :
  (4 / 5) * cotton_weight = (1 / 5) * iron_weight :=
by
  rw [h_cotton, h_iron]
  simp
  sorry

end cotton_equals_iron_l1461_146143


namespace LCM_of_36_and_220_l1461_146157

theorem LCM_of_36_and_220:
  let A := 36
  let B := 220
  let productAB := A * B
  let HCF := 4
  let LCM := (A * B) / HCF
  LCM = 1980 := 
by
  sorry

end LCM_of_36_and_220_l1461_146157


namespace total_red_cards_l1461_146144

theorem total_red_cards (num_standard_decks : ℕ) (num_special_decks : ℕ)
  (red_standard_deck : ℕ) (additional_red_special_deck : ℕ)
  (total_decks : ℕ) (h1 : num_standard_decks = 5)
  (h2 : num_special_decks = 10)
  (h3 : red_standard_deck = 26)
  (h4 : additional_red_special_deck = 4)
  (h5 : total_decks = num_standard_decks + num_special_decks) :
  num_standard_decks * red_standard_deck +
  num_special_decks * (red_standard_deck + additional_red_special_deck) = 430 := by
  -- Proof is omitted.
  sorry

end total_red_cards_l1461_146144


namespace chord_segments_division_l1461_146172

-- Definitions based on the conditions
variables (R OM : ℝ) (AB : ℝ)
-- Setting the values as the problem provides 
def radius : ℝ := 15
def distance_from_center : ℝ := 13
def chord_length : ℝ := 18

-- Formulate the problem statement as a theorem
theorem chord_segments_division :
  ∃ (AM MB : ℝ), AM = 14 ∧ MB = 4 :=
by
  let CB := chord_length / 2
  let OC := Real.sqrt (radius^2 - CB^2)
  let MC := Real.sqrt (distance_from_center^2 - OC^2)
  let AM := CB + MC
  let MB := CB - MC
  use AM, MB
  sorry

end chord_segments_division_l1461_146172


namespace ratio_of_fuji_trees_l1461_146194

variable (F T : ℕ) -- Declaring F as number of pure Fuji trees, T as total number of trees
variables (C : ℕ) -- Declaring C as number of cross-pollinated trees 

theorem ratio_of_fuji_trees 
  (h1: 10 * C = T) 
  (h2: F + C = 221) 
  (h3: T = F + 39 + C) : 
  F * 52 = 39 * T := 
sorry

end ratio_of_fuji_trees_l1461_146194


namespace smallest_positive_n_l1461_146169

theorem smallest_positive_n (n : ℕ) : n > 0 → (3 * n ≡ 1367 [MOD 26]) → n = 5 :=
by
  intros _ _
  sorry

end smallest_positive_n_l1461_146169


namespace jason_initial_money_l1461_146116

theorem jason_initial_money (M : ℝ) 
  (h1 : M - (M / 4 + 10 + (2 / 5 * (3 / 4 * M - 10) + 8)) = 130) : 
  M = 320 :=
by
  sorry

end jason_initial_money_l1461_146116


namespace log_27_3_l1461_146154

noncomputable def log_base (a b : ℝ) : ℝ := Real.log a / Real.log b

theorem log_27_3 :
  log_base 3 27 = 1 / 3 := by
  sorry

end log_27_3_l1461_146154


namespace fuel_ethanol_problem_l1461_146158

theorem fuel_ethanol_problem (x : ℝ) (h : 0.12 * x + 0.16 * (200 - x) = 28) : x = 100 := 
by
  sorry

end fuel_ethanol_problem_l1461_146158


namespace cost_of_each_art_book_l1461_146173

-- Define the conditions
def total_cost : ℕ := 30
def cost_per_math_and_science_book : ℕ := 3
def num_math_books : ℕ := 2
def num_art_books : ℕ := 3
def num_science_books : ℕ := 6

-- The proof problem statement
theorem cost_of_each_art_book :
  (total_cost - (num_math_books * cost_per_math_and_science_book + num_science_books * cost_per_math_and_science_book)) / num_art_books = 2 :=
by
  sorry -- proof goes here,

end cost_of_each_art_book_l1461_146173


namespace ratio_of_area_l1461_146133

noncomputable def area_ratio (l w r : ℝ) : ℝ :=
  if h1 : 2 * l + 2 * w = 2 * Real.pi * r 
  ∧ l = 2 * w then 
    (l * w) / (Real.pi * r ^ 2) 
  else 
    0

theorem ratio_of_area (l w r : ℝ) 
  (h1 : 2 * l + 2 * w = 2 * Real.pi * r) 
  (h2 : l = 2 * w) :
  area_ratio l w r = 2 * Real.pi / 9 :=
by
  unfold area_ratio
  simp [h1, h2]
  sorry

end ratio_of_area_l1461_146133


namespace quadratic_m_condition_l1461_146132

theorem quadratic_m_condition (m : ℝ) (h_eq : (m - 2) * x ^ (m ^ 2 - 2) - m * x + 1 = 0) (h_pow : m ^ 2 - 2 = 2) :
  m = -2 :=
by sorry

end quadratic_m_condition_l1461_146132


namespace candidate_lost_by_votes_l1461_146182

theorem candidate_lost_by_votes :
  let candidate_votes := (31 / 100) * 6450
  let rival_votes := (69 / 100) * 6450
  candidate_votes <= 6450 ∧ rival_votes <= 6450 ∧ rival_votes - candidate_votes = 2451 :=
by
  let candidate_votes := (31 / 100) * 6450
  let rival_votes := (69 / 100) * 6450
  have h1: candidate_votes <= 6450 := sorry
  have h2: rival_votes <= 6450 := sorry
  have h3: rival_votes - candidate_votes = 2451 := sorry
  exact ⟨h1, h2, h3⟩

end candidate_lost_by_votes_l1461_146182


namespace proportion_of_second_prize_winners_l1461_146153

-- conditions
variables (A B C : ℝ) -- A, B, and C represent the proportions of first, second, and third prize winners respectively.
variables (h1 : A + B = 3 / 4)
variables (h2 : B + C = 2 / 3)

-- statement
theorem proportion_of_second_prize_winners : B = 5 / 12 :=
by
  sorry

end proportion_of_second_prize_winners_l1461_146153


namespace problem_solution_l1461_146130

theorem problem_solution : (3106 - 2935 + 17)^2 / 121 = 292 := by
  sorry

end problem_solution_l1461_146130


namespace complex_quadrant_l1461_146198

open Complex

theorem complex_quadrant (z : ℂ) (h : z = (2 - I) / (2 + I)) : 
  z.re > 0 ∧ z.im < 0 := 
by
  sorry

end complex_quadrant_l1461_146198


namespace find_Y_length_l1461_146145

theorem find_Y_length (Y : ℝ) : 
  (3 + 2 + 3 + 4 + Y = 7 + 4 + 2) → Y = 1 :=
by
  intro h
  sorry

end find_Y_length_l1461_146145


namespace obtuse_triangle_side_range_l1461_146104

theorem obtuse_triangle_side_range {a : ℝ} (h1 : a > 3) (h2 : (a - 3)^2 < 36) : 3 < a ∧ a < 9 := 
by
  sorry

end obtuse_triangle_side_range_l1461_146104


namespace angles_on_axes_correct_l1461_146134

-- Definitions for angles whose terminal sides lie on x-axis and y-axis.
def angles_on_x_axis (α : ℝ) : Prop := ∃ k : ℤ, α = k * Real.pi
def angles_on_y_axis (α : ℝ) : Prop := ∃ k : ℤ, α = k * Real.pi + Real.pi / 2

-- Combined definition for angles on the coordinate axes using Lean notation
def angles_on_axes (α : ℝ) : Prop := ∃ n : ℤ, α = n * (Real.pi / 2)

-- Theorem stating that angles on the coordinate axes are of the form nπ/2.
theorem angles_on_axes_correct : ∀ α : ℝ, (angles_on_x_axis α ∨ angles_on_y_axis α) ↔ angles_on_axes α := 
sorry -- Proof is omitted.

end angles_on_axes_correct_l1461_146134


namespace sum_of_coordinates_l1461_146167

theorem sum_of_coordinates {g h : ℝ → ℝ} 
  (h₁ : g 4 = 5)
  (h₂ : ∀ x, h x = (g x)^2) :
  4 + h 4 = 29 := by
  sorry

end sum_of_coordinates_l1461_146167


namespace ratio_b_a_4_l1461_146187

theorem ratio_b_a_4 (a b : ℚ) (h1 : b / a = 4) (h2 : b = 15 - 6 * a) : a = 3 / 2 :=
by
  sorry

end ratio_b_a_4_l1461_146187


namespace disjoint_subsets_less_elements_l1461_146124

open Nat

theorem disjoint_subsets_less_elements (m : ℕ) (A B : Finset ℕ) (hA : A ⊆ Finset.range (m + 1))
  (hB : B ⊆ Finset.range (m + 1)) (h_disjoint : Disjoint A B)
  (h_sum : A.sum id = B.sum id) : ↑(A.card) < m / Real.sqrt 2 ∧ ↑(B.card) < m / Real.sqrt 2 := 
sorry

end disjoint_subsets_less_elements_l1461_146124


namespace fewer_cans_collected_today_than_yesterday_l1461_146150

theorem fewer_cans_collected_today_than_yesterday :
  let sarah_yesterday := 50
  let lara_yesterday := sarah_yesterday + 30
  let sarah_today := 40
  let lara_today := 70
  let total_yesterday := sarah_yesterday + lara_yesterday
  let total_today := sarah_today + lara_today
  total_yesterday - total_today = 20 :=
by
  sorry

end fewer_cans_collected_today_than_yesterday_l1461_146150


namespace shopkeeper_gain_percentage_l1461_146185

noncomputable def gain_percentage (false_weight: ℕ) (true_weight: ℕ) : ℝ :=
  (↑(true_weight - false_weight) / ↑false_weight) * 100

theorem shopkeeper_gain_percentage :
  gain_percentage 960 1000 = 4.166666666666667 := 
sorry

end shopkeeper_gain_percentage_l1461_146185


namespace largest_common_term_in_range_l1461_146186

theorem largest_common_term_in_range :
  ∃ (a : ℕ), a < 150 ∧ (∃ (n : ℕ), a = 3 + 8 * n) ∧ (∃ (n : ℕ), a = 5 + 9 * n) ∧ a = 131 :=
by
  sorry

end largest_common_term_in_range_l1461_146186


namespace primes_less_than_200_with_ones_digit_3_l1461_146139

theorem primes_less_than_200_with_ones_digit_3 : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, Prime n ∧ n < 200 ∧ n % 10 = 3) ∧ S.card = 12 := 
by
  sorry

end primes_less_than_200_with_ones_digit_3_l1461_146139


namespace paint_house_18_women_4_days_l1461_146164

theorem paint_house_18_women_4_days :
  (∀ (m1 m2 : ℕ) (d1 d2 : ℕ), m1 * d1 = m2 * d2) →
  (12 * 6 = 72) →
  (72 = 18 * d) →
  d = 4.0 :=
by
  sorry

end paint_house_18_women_4_days_l1461_146164


namespace possible_values_of_n_l1461_146110

theorem possible_values_of_n (n : ℕ) (h_pos : 0 < n) (h_prime_n : Nat.Prime n) (h_prime_double_sub1 : Nat.Prime (2 * n - 1)) (h_prime_quad_sub1 : Nat.Prime (4 * n - 1)) :
  n = 2 ∨ n = 3 :=
by
  sorry

end possible_values_of_n_l1461_146110


namespace greatest_two_digit_multiple_of_17_is_85_l1461_146138

theorem greatest_two_digit_multiple_of_17_is_85 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ m % 17 = 0) → m ≤ n) ∧ n = 85 :=
sorry

end greatest_two_digit_multiple_of_17_is_85_l1461_146138


namespace stratified_sampling_l1461_146180

-- Definitions of the classes and their student counts
def class1_students : Nat := 54
def class2_students : Nat := 42

-- Definition of total students to be sampled
def total_sampled_students : Nat := 16

-- Definition of the number of students to be selected from each class
def students_selected_from_class1 : Nat := 9
def students_selected_from_class2 : Nat := 7

-- The proof problem
theorem stratified_sampling :
  students_selected_from_class1 + students_selected_from_class2 = total_sampled_students ∧ 
  students_selected_from_class1 * (class2_students + class1_students) = class1_students * total_sampled_students :=
by
  sorry

end stratified_sampling_l1461_146180


namespace dog_food_bags_needed_l1461_146159

theorem dog_food_bags_needed
  (cup_weight: ℝ)
  (dogs: ℕ)
  (cups_per_day: ℕ)
  (days_in_month: ℕ)
  (bag_weight: ℝ)
  (hcw: cup_weight = 1/4)
  (hd: dogs = 2)
  (hcd: cups_per_day = 6 * 2)
  (hdm: days_in_month = 30)
  (hbw: bag_weight = 20) :
  (dogs * cups_per_day * days_in_month * cup_weight) / bag_weight = 9 :=
by
  sorry

end dog_food_bags_needed_l1461_146159


namespace second_yellow_probability_l1461_146175

-- Define the conditions in Lean
def BagA : Type := {marble : Int // marble ≥ 0}
def BagB : Type := {marble : Int // marble ≥ 0}
def BagC : Type := {marble : Int // marble ≥ 0}
def BagD : Type := {marble : Int // marble ≥ 0}

noncomputable def marbles_in_A := 4 + 5 + 2
noncomputable def marbles_in_B := 7 + 5
noncomputable def marbles_in_C := 3 + 7
noncomputable def marbles_in_D := 8 + 2

-- Probabilities of drawing specific colors from Bag A
noncomputable def prob_white_A := 4 / 11
noncomputable def prob_black_A := 5 / 11
noncomputable def prob_red_A := 2 / 11

-- Probabilities of drawing a yellow marble from Bags B, C and D
noncomputable def prob_yellow_B := 7 / 12
noncomputable def prob_yellow_C := 3 / 10
noncomputable def prob_yellow_D := 8 / 10

-- Expected probability that the second marble is yellow
noncomputable def prob_second_yellow : ℚ :=
  (prob_white_A * prob_yellow_B) + (prob_black_A * prob_yellow_C) + (prob_red_A * prob_yellow_D)

/-- Prove that the total probability the second marble drawn is yellow is 163/330. -/
theorem second_yellow_probability :
  prob_second_yellow = 163 / 330 := sorry

end second_yellow_probability_l1461_146175


namespace quadratic_completing_square_l1461_146127

theorem quadratic_completing_square :
  ∃ (a b c : ℚ), a = 12 ∧ b = 6 ∧ c = 1296 ∧ 12 + 6 + 1296 = 1314 ∧
  (12 * (x + b)^2 + c = 12 * x^2 + 144 * x + 1728) :=
by
  sorry

end quadratic_completing_square_l1461_146127


namespace proposition_5_l1461_146163

/-! 
  Proposition 5: If there are four points A, B, C, D in a plane, 
  then the vector addition relation: \overrightarrow{AC} + \overrightarrow{BD} = \overrightarrow{BC} + \overrightarrow{AD} must hold.
--/

variables {A B C D : Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]
variables (AC BD BC AD : A)

-- Theorem Statement in Lean 4
theorem proposition_5 (AC BD BC AD : A)
  : AC + BD = BC + AD := by
  -- Proof by congruence and equality, will add actual steps here
  sorry

end proposition_5_l1461_146163


namespace find_2a_2b_2c_2d_l1461_146111

open Int

theorem find_2a_2b_2c_2d (a b c d : ℤ) 
  (h1 : a - b + c = 7) 
  (h2 : b - c + d = 8) 
  (h3 : c - d + a = 4) 
  (h4 : d - a + b = 1) : 
  2*a + 2*b + 2*c + 2*d = 20 := 
sorry

end find_2a_2b_2c_2d_l1461_146111


namespace inverse_negation_l1461_146193

theorem inverse_negation :
  (∀ x : ℝ, x ≥ 3 → x < 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ ¬ (x < 3)) :=
by
  sorry

end inverse_negation_l1461_146193


namespace cost_per_pack_l1461_146176

theorem cost_per_pack (total_bill : ℕ) (change_given : ℕ) (packs : ℕ) (total_cost := total_bill - change_given) (cost_per_pack := total_cost / packs) 
  (h1 : total_bill = 20) 
  (h2 : change_given = 11) 
  (h3 : packs = 3) : 
  cost_per_pack = 3 := by
  sorry

end cost_per_pack_l1461_146176


namespace Alan_total_cost_is_84_l1461_146148

theorem Alan_total_cost_is_84 :
  let D := 2 * 12
  let A := 12
  let cost_other := 2 * D + A
  let M := 0.4 * cost_other
  2 * D + A + M = 84 := by
    sorry

end Alan_total_cost_is_84_l1461_146148


namespace power_mod_residue_l1461_146192

theorem power_mod_residue (n : ℕ) (h : n = 1234) : (7^n) % 19 = 9 := by
  sorry

end power_mod_residue_l1461_146192


namespace part_a_part_b_l1461_146101

-- the conditions
variables (r R x : ℝ) (h_rltR : r < R)
variables (h_x : x = (R - r) / 2)
variables (h1 : 0 < x)
variables (h12_circles : ∀ i : ℕ, i ∈ Finset.range 12 → ∃ c_i : ℝ × ℝ, True)  -- Informal way to note 12 circles of radius x are placed

-- prove each part
theorem part_a (r R : ℝ) (h_rltR : r < R) : x = (R - r) / 2 :=
sorry

theorem part_b (r R : ℝ) (h_rltR : r < R) (h_x : x = (R - r) / 2) :
  (R / r) = (4 + Real.sqrt 6 - Real.sqrt 2) / (4 - Real.sqrt 6 + Real.sqrt 2) :=
sorry

end part_a_part_b_l1461_146101


namespace fewest_toothpicks_proof_l1461_146166

noncomputable def fewest_toothpicks_to_remove (total_toothpicks : ℕ) (additional_row_and_column : ℕ) (triangles : ℕ) (upward_triangles : ℕ) (downward_triangles : ℕ) (max_destroyed_per_toothpick : ℕ) (horizontal_toothpicks : ℕ) : ℕ :=
  horizontal_toothpicks

theorem fewest_toothpicks_proof 
  (total_toothpicks : ℕ := 40) 
  (additional_row_and_column : ℕ := 1) 
  (triangles : ℕ := 35) 
  (upward_triangles : ℕ := 15) 
  (downward_triangles : ℕ := 10)
  (max_destroyed_per_toothpick : ℕ := 1)
  (horizontal_toothpicks : ℕ := 15) :
  fewest_toothpicks_to_remove total_toothpicks additional_row_and_column triangles upward_triangles downward_triangles max_destroyed_per_toothpick horizontal_toothpicks = 15 := 
by 
  sorry

end fewest_toothpicks_proof_l1461_146166


namespace extracellular_proof_l1461_146191

-- Define the components
def component1 : Set String := {"Na＋", "antibodies", "plasma proteins"}
def component2 : Set String := {"Hemoglobin", "O2", "glucose"}
def component3 : Set String := {"glucose", "CO2", "insulin"}
def component4 : Set String := {"Hormones", "neurotransmitter vesicles", "amino acids"}

-- Define the properties of being a part of the extracellular fluid
def is_extracellular (x : Set String) : Prop :=
  x = component1 ∨ x = component3

-- State the theorem to prove
theorem extracellular_proof : is_extracellular component1 ∧ ¬is_extracellular component2 ∧ is_extracellular component3 ∧ ¬is_extracellular component4 :=
by
  sorry

end extracellular_proof_l1461_146191


namespace nearest_integer_x_sub_y_l1461_146168

theorem nearest_integer_x_sub_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : |x| - y = 4) 
  (h2 : |x| * y - x^3 = 1) : 
  abs (x - y - 4) < 1 :=
sorry

end nearest_integer_x_sub_y_l1461_146168


namespace cost_per_slice_in_cents_l1461_146171

def loaves : ℕ := 3
def slices_per_loaf : ℕ := 20
def total_payment : ℕ := 2 * 20
def change : ℕ := 16
def total_cost : ℕ := total_payment - change
def total_slices : ℕ := loaves * slices_per_loaf

theorem cost_per_slice_in_cents :
  (total_cost : ℕ) * 100 / total_slices = 40 :=
by
  sorry

end cost_per_slice_in_cents_l1461_146171


namespace option_d_correct_factorization_l1461_146128

theorem option_d_correct_factorization (x : ℝ) : 
  -8 * x ^ 2 + 8 * x - 2 = -2 * (2 * x - 1) ^ 2 :=
by 
  sorry

end option_d_correct_factorization_l1461_146128


namespace parallel_segments_have_equal_slopes_l1461_146190

theorem parallel_segments_have_equal_slopes
  (A B X : ℝ × ℝ)
  (Y : ℝ × ℝ)
  (hA : A = (-5, -1))
  (hB : B = (2, -8))
  (hX : X = (2, 10))
  (hY1 : Y.1 = 20)
  (h_parallel : (B.2 - A.2) / (B.1 - A.1) = (Y.2 - X.2) / (Y.1 - X.1)) :
  Y.2 = -8 :=
by
  sorry

end parallel_segments_have_equal_slopes_l1461_146190


namespace hourly_wage_increase_is_10_percent_l1461_146109

theorem hourly_wage_increase_is_10_percent :
  ∀ (H W : ℝ), 
    ∀ (H' : ℝ), H' = H * (1 - 0.09090909090909092) →
    (H * W = H' * W') →
    (W' = (100 * W) / 90) := by
  sorry

end hourly_wage_increase_is_10_percent_l1461_146109


namespace digit_five_occurrences_l1461_146126

/-- 
  Define that a 24-hour digital clock display shows times containing at least one 
  occurrence of the digit '5' a total of 450 times in a 24-hour period.
--/
def contains_digit_five (n : Nat) : Prop := 
  n / 10 = 5 ∨ n % 10 = 5

def count_times_with_digit_five : Nat :=
  let hours_with_five := 2 * 60  -- 05:00-05:59 and 15:00-15:59, each hour has 60 minutes
  let remaining_hours := 22 * 15 -- 22 hours, each hour has 15 minutes
  hours_with_five + remaining_hours

theorem digit_five_occurrences : count_times_with_digit_five = 450 := by
  sorry

end digit_five_occurrences_l1461_146126


namespace length_QF_l1461_146122

noncomputable def parabola (x y : ℝ) : Prop := y^2 = 8 * x

def focus : ℝ × ℝ := (2, 0)

def directrix (x y : ℝ) : Prop := x = 1 -- Directrix of the given parabola

def point_on_directrix (P : ℝ × ℝ) : Prop := directrix P.1 P.2

def point_on_parabola (Q : ℝ × ℝ) : Prop := parabola Q.1 Q.2

def point_on_line_PF (P F Q : ℝ × ℝ) : Prop :=
  ∃ m : ℝ, m ≠ 0 ∧ (Q.2 = m * (Q.1 - F.1) + F.2) ∧ point_on_parabola Q

def vector_equality (F P Q : ℝ × ℝ) : Prop :=
  (4 * (Q.1 - F.1), 4 * (Q.2 - F.2)) = (P.1 - F.1, P.2 - F.2)

theorem length_QF 
  (P Q : ℝ × ℝ)
  (hPd : point_on_directrix P)
  (hPQ : point_on_line_PF P focus Q)
  (hVec : vector_equality focus P Q) : 
  dist Q focus = 3 :=
by
  sorry

end length_QF_l1461_146122


namespace twelfth_term_l1461_146183

noncomputable def a (n : ℕ) : ℕ := 
  if n = 0 then 0 
  else (n * (n + 2)) - ((n - 1) * (n + 1))

theorem twelfth_term : a 12 = 25 :=
by sorry

end twelfth_term_l1461_146183


namespace cakes_served_yesterday_l1461_146141

theorem cakes_served_yesterday (lunch_cakes dinner_cakes total_cakes served_yesterday : ℕ)
  (h1 : lunch_cakes = 5)
  (h2 : dinner_cakes = 6)
  (h3 : total_cakes = 14)
  (h4 : total_cakes = lunch_cakes + dinner_cakes + served_yesterday) :
  served_yesterday = 3 := 
by 
  sorry

end cakes_served_yesterday_l1461_146141


namespace eventually_periodic_sequence_l1461_146123

theorem eventually_periodic_sequence
  (a : ℕ → ℕ)
  (h1 : ∀ n m : ℕ, 0 < n → 0 < m → a (n + 2 * m) ∣ (a n + a (n + m)))
  : ∃ N d : ℕ, 0 < N ∧ 0 < d ∧ ∀ n > N, a n = a (n + d) :=
sorry

end eventually_periodic_sequence_l1461_146123


namespace arithmetic_sequence_problem_l1461_146135

theorem arithmetic_sequence_problem
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (a1 : ℝ)
  (d : ℝ)
  (h1 : d = 2)
  (h2 : ∀ n : ℕ, a n = a1 + (n - 1) * d)
  (h3 :  ∀ n : ℕ, S n = (n * (2 * a1 + (n - 1) * d)) / 2)
  (h4 : S 6 = 3 * S 3) :
  a 9 = 20 :=
by sorry

end arithmetic_sequence_problem_l1461_146135


namespace least_money_Moe_l1461_146179

theorem least_money_Moe (Bo Coe Flo Jo Moe Zoe : ℝ)
  (H1 : Flo > Jo) 
  (H2 : Flo > Bo) 
  (H3 : Bo > Zoe) 
  (H4 : Coe > Zoe) 
  (H5 : Jo > Zoe) 
  (H6 : Bo > Jo) 
  (H7 : Zoe > Moe) : 
  (Moe < Bo) ∧ (Moe < Coe) ∧ (Moe < Flo) ∧ (Moe < Jo) ∧ (Moe < Zoe) :=
by
  sorry

end least_money_Moe_l1461_146179


namespace eval_expr_l1461_146197

theorem eval_expr : 3^2 * 4 * 6^3 * Nat.factorial 7 = 39191040 := by
  -- the proof will be filled in here
  sorry

end eval_expr_l1461_146197


namespace find_a_plus_b_l1461_146156

-- Define the constants and conditions
variables (a b c : ℤ)
variables (a_cond : 0 ≤ a ∧ a < 5) (b_cond : 0 ≤ b ∧ b < 13)
variables (frac_decomp : (1 : ℚ) / 2015 = (a : ℚ) / 5 + (b : ℚ) / 13 + (c : ℚ) / 31)

-- State the theorem
theorem find_a_plus_b (a b c : ℤ) (a_cond : 0 ≤ a ∧ a < 5) (b_cond : 0 ≤ b ∧ b < 13) (frac_decomp : (1 : ℚ) / 2015 = (a : ℚ) / 5 + (b : ℚ) / 13 + (c : ℚ) / 31) :
  a + b = 14 := 
sorry

end find_a_plus_b_l1461_146156


namespace rectangle_area_l1461_146170

theorem rectangle_area (side_length : ℝ) (rect_width : ℝ) (rect_length : ℝ) : 
  side_length^2 = 64 → 
  rect_width = side_length →
  rect_length = 3 * rect_width →
  rect_width * rect_length = 192 := 
by
  intros h1 h2 h3
  sorry

end rectangle_area_l1461_146170


namespace tablet_battery_life_l1461_146188

noncomputable def battery_life_remaining
  (no_use_life : ℝ) (use_life : ℝ) (total_on_time : ℝ) (use_time : ℝ) : ℝ :=
  let no_use_consumption_rate := 1 / no_use_life
  let use_consumption_rate := 1 / use_life
  let no_use_time := total_on_time - use_time
  let total_battery_used := no_use_time * no_use_consumption_rate + use_time * use_consumption_rate
  let remaining_battery := 1 - total_battery_used
  remaining_battery / no_use_consumption_rate

theorem tablet_battery_life (no_use_life : ℝ) (use_life : ℝ) (total_on_time : ℝ) (use_time : ℝ) :
  battery_life_remaining no_use_life use_life total_on_time use_time = 6 :=
by
  -- The proof will go here, we use sorry for now to skip the proof step.
  sorry

end tablet_battery_life_l1461_146188


namespace sandy_correct_value_t_l1461_146117

theorem sandy_correct_value_t (p q r s : ℕ) (t : ℕ) 
  (hp : p = 2) (hq : q = 4) (hr : r = 6) (hs : s = 8)
  (expr1 : p + q - r + s - t = p + (q - (r + (s - t)))) :
  t = 8 := 
by
  sorry

end sandy_correct_value_t_l1461_146117


namespace tangent_line_value_of_a_l1461_146147

theorem tangent_line_value_of_a (a : ℝ) :
  (∃ (m : ℝ), (2 * m - 1 = a * m + Real.log m) ∧ (a + 1 / m = 2)) → a = 1 :=
by 
sorry

end tangent_line_value_of_a_l1461_146147


namespace find_number_l1461_146152

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 8) : x = 4 :=
by
  sorry

end find_number_l1461_146152


namespace problem_part1_problem_part2_l1461_146149

theorem problem_part1 (α : ℝ) (h : Real.tan α = -2) :
    (3 * Real.sin α + 2 * Real.cos α) / (5 * Real.cos α - Real.sin α) = -4 / 7 := 
    sorry

theorem problem_part2 (α : ℝ) (h : Real.tan α = -2) :
    3 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = -5 := 
    sorry

end problem_part1_problem_part2_l1461_146149


namespace total_opaque_stackings_l1461_146181

-- Define the glass pane and its rotation
inductive Rotation
| deg_0 | deg_90 | deg_180 | deg_270
deriving DecidableEq, Repr

-- The property of opacity for a stack of glass panes
def isOpaque (stack : List (List Rotation)) : Bool :=
  -- The implementation of this part depends on the specific condition in the problem
  -- and here is abstracted out for the problem statement.
  sorry

-- The main problem stating the required number of ways
theorem total_opaque_stackings : ∃ (n : ℕ), n = 7200 :=
  sorry

end total_opaque_stackings_l1461_146181


namespace sin_squared_value_l1461_146129

theorem sin_squared_value (x : ℝ) (h : Real.tan x = 1 / 2) : 
  Real.sin (π / 4 + x) ^ 2 = 9 / 10 :=
by
  -- Proof part, skipped.
  sorry

end sin_squared_value_l1461_146129


namespace fraction_of_succeeding_number_l1461_146131

theorem fraction_of_succeeding_number (N : ℝ) (hN : N = 24.000000000000004) :
  ∃ f : ℝ, (1 / 4) * N > f * (N + 1) + 1 ∧ f = 0.2 :=
by
  sorry

end fraction_of_succeeding_number_l1461_146131


namespace fifth_term_arithmetic_sequence_l1461_146162

variable (x y : ℝ)

def a1 := x + 2 * y^2
def a2 := x - 2 * y^2
def a3 := x + 3 * y
def a4 := x - 4 * y
def d := a2 - a1

theorem fifth_term_arithmetic_sequence : y = -1/2 → 
  x - 10 * y^2 - 4 * y^2 = x - 7/2 := by
  sorry

end fifth_term_arithmetic_sequence_l1461_146162


namespace opposite_of_neg_three_l1461_146196

theorem opposite_of_neg_three : -(-3) = 3 := by
  sorry

end opposite_of_neg_three_l1461_146196


namespace son_l1461_146146

variable (S M : ℕ)

theorem son's_age
  (h1 : M = S + 24)
  (h2 : M + 2 = 2 * (S + 2))
  : S = 22 :=
sorry

end son_l1461_146146


namespace quadratic_root_range_l1461_146114

theorem quadratic_root_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, (x₁ > 0) ∧ (x₂ < 0) ∧ (x₁^2 + 2 * (a - 1) * x₁ + 2 * a + 6 = 0) ∧ (x₂^2 + 2 * (a - 1) * x₂ + 2 * a + 6 = 0)) → a < -3 :=
by
  sorry

end quadratic_root_range_l1461_146114


namespace jo_thinking_greatest_integer_l1461_146136

theorem jo_thinking_greatest_integer :
  ∃ n : ℕ, n < 150 ∧ 
           (∃ k : ℤ, n = 9 * k - 2) ∧ 
           (∃ m : ℤ, n = 11 * m - 4) ∧ 
           (∀ N : ℕ, (N < 150 ∧ 
                      (∃ K : ℤ, N = 9 * K - 2) ∧ 
                      (∃ M : ℤ, N = 11 * M - 4)) → N ≤ n) 
:= by
  sorry

end jo_thinking_greatest_integer_l1461_146136


namespace roger_final_money_l1461_146102

variable (initial_money : ℕ)
variable (spent_money : ℕ)
variable (received_money : ℕ)

theorem roger_final_money (h1 : initial_money = 45) (h2 : spent_money = 20) (h3 : received_money = 46) :
  (initial_money - spent_money + received_money) = 71 :=
by
  sorry

end roger_final_money_l1461_146102


namespace orangeade_price_l1461_146199

theorem orangeade_price (O W : ℝ) (h1 : O = W) (price_day1 : ℝ) (price_day2 : ℝ) 
    (volume_day1 : ℝ) (volume_day2 : ℝ) (revenue_day1 : ℝ) (revenue_day2 : ℝ) : 
    volume_day1 = 2 * O ∧ volume_day2 = 3 * O ∧ revenue_day1 = revenue_day2 ∧ price_day1 = 0.82 
    → price_day2 = 0.55 :=
by
    intros
    sorry

end orangeade_price_l1461_146199


namespace smallest_positive_m_integral_solutions_l1461_146118

theorem smallest_positive_m_integral_solutions (m : ℕ) :
  (∃ (x y : ℤ), 10 * x * x - m * x + 660 = 0 ∧ 10 * y * y - m * y + 660 = 0 ∧ x ≠ y)
  → m = 170 := sorry

end smallest_positive_m_integral_solutions_l1461_146118


namespace initial_welders_count_l1461_146106

theorem initial_welders_count (W : ℕ) (h1: (1 + 16 * (W - 9) / W = 8)) : W = 16 :=
by {
  sorry
}

end initial_welders_count_l1461_146106


namespace three_digit_numbers_with_2_without_4_l1461_146100

theorem three_digit_numbers_with_2_without_4 : 
  ∃ n : Nat, n = 200 ∧
  (∀ x : Nat, 100 ≤ x ∧ x ≤ 999 → 
      (∃ d1 d2 d3,
        d1 ≠ 0 ∧ 
        x = d1 * 100 + d2 * 10 + d3 ∧ 
        (d1 ≠ 4 ∧ d2 ≠ 4 ∧ d3 ≠ 4) ∧
        (d1 = 2 ∨ d2 = 2 ∨ d3 = 2))) :=
sorry

end three_digit_numbers_with_2_without_4_l1461_146100


namespace two_digit_decimal_bounds_l1461_146174

def is_approximate (original approx : ℝ) : Prop :=
  abs (original - approx) < 0.05

theorem two_digit_decimal_bounds :
  ∃ max min : ℝ, is_approximate 15.6 max ∧ max = 15.64 ∧ is_approximate 15.6 min ∧ min = 15.55 :=
by
  sorry

end two_digit_decimal_bounds_l1461_146174


namespace burger_cost_l1461_146115

theorem burger_cost (b s : ℕ) (h1 : 3 * b + 2 * s = 385) (h2 : 2 * b + 3 * s = 360) : b = 87 :=
sorry

end burger_cost_l1461_146115


namespace Q_transform_l1461_146108

def rotate_180_clockwise (p q : ℝ × ℝ) : ℝ × ℝ :=
  let (px, py) := p
  let (qx, qy) := q
  (2 * px - qx, 2 * py - qy)

def reflect_y_equals_x (p : ℝ × ℝ) : ℝ × ℝ :=
  let (px, py) := p
  (py, px)

def Q := (8, -11) -- from the reverse transformations

theorem Q_transform (c d : ℝ) :
  (reflect_y_equals_x (rotate_180_clockwise (2, -3) (c, d)) = (5, -4)) → (d - c = -19) :=
by sorry

end Q_transform_l1461_146108


namespace three_pair_probability_l1461_146189

theorem three_pair_probability :
  let total_combinations := Nat.choose 52 5
  let three_pair_combinations := 13 * 4 * 12 * 4
  total_combinations = 2598960 ∧ three_pair_combinations = 2496 →
  (three_pair_combinations : ℚ) / total_combinations = 2496 / 2598960 :=
by
  -- Definitions and computations can be added here if necessary
  sorry

end three_pair_probability_l1461_146189


namespace common_ratio_of_geometric_sequence_l1461_146120

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ) (d : ℝ) (h1 : d ≠ 0) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + d)
  (h3 : a 1 + 4 * d = (a 0 + 16 * d) * (a 0 + 4 * d) / a 0 ) :
  (a 1 + 4 * d) / a 0 = 3 :=
by
  sorry

end common_ratio_of_geometric_sequence_l1461_146120


namespace impossible_even_n_m_if_n3_plus_m3_is_odd_l1461_146161

theorem impossible_even_n_m_if_n3_plus_m3_is_odd
  (n m : ℤ) (h : (n^3 + m^3) % 2 = 1) : ¬((n % 2 = 0) ∧ (m % 2 = 0)) := by
  sorry

end impossible_even_n_m_if_n3_plus_m3_is_odd_l1461_146161


namespace election_winner_votes_l1461_146155

theorem election_winner_votes (V : ℝ) (h1 : 0.62 * V - 0.38 * V = 360) :
  0.62 * V = 930 :=
by {
  sorry
}

end election_winner_votes_l1461_146155


namespace smallest_n_l1461_146177

variable {a : ℕ → ℝ} -- the arithmetic sequence
noncomputable def d := a 2 - a 1  -- common difference

variable {S : ℕ → ℝ}  -- sum of the first n terms

-- conditions
axiom cond1 : a 66 < 0
axiom cond2 : a 67 > 0
axiom cond3 : a 67 > abs (a 66)

-- sum of the first n terms of the arithmetic sequence
noncomputable def sum_n (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

theorem smallest_n (n : ℕ) : S n > 0 → n = 132 :=
by
  sorry

end smallest_n_l1461_146177


namespace least_prime_factor_of_5_to_the_3_minus_5_to_the_2_l1461_146107

theorem least_prime_factor_of_5_to_the_3_minus_5_to_the_2 : 
  Nat.minFac (5^3 - 5^2) = 2 := by
  sorry

end least_prime_factor_of_5_to_the_3_minus_5_to_the_2_l1461_146107


namespace remainder_of_7_pow_145_mod_12_l1461_146165

theorem remainder_of_7_pow_145_mod_12 : (7 ^ 145) % 12 = 7 :=
by
  sorry

end remainder_of_7_pow_145_mod_12_l1461_146165


namespace cost_D_to_E_l1461_146195

def distance_DF (DF DE EF : ℝ) : Prop :=
  DE^2 = DF^2 + EF^2

def cost_to_fly (distance : ℝ) (per_kilometer_cost booking_fee : ℝ) : ℝ :=
  distance * per_kilometer_cost + booking_fee

noncomputable def total_cost_to_fly_from_D_to_E : ℝ :=
  let DE := 3750 -- Distance from D to E (km)
  let booking_fee := 120 -- Booking fee in dollars
  let per_kilometer_cost := 0.12 -- Cost per kilometer in dollars
  cost_to_fly DE per_kilometer_cost booking_fee

theorem cost_D_to_E : total_cost_to_fly_from_D_to_E = 570 := by
  sorry

end cost_D_to_E_l1461_146195


namespace missing_number_is_correct_l1461_146113

theorem missing_number_is_correct (mean : ℝ) (observed_numbers : List ℝ) (total_obs : ℕ) (x : ℝ) :
  mean = 14.2 →
  observed_numbers = [8, 13, 21, 7, 23] →
  total_obs = 6 →
  (mean * total_obs = x + observed_numbers.sum) →
  x = 13.2 :=
by
  intros h_mean h_obs h_total h_sum
  sorry

end missing_number_is_correct_l1461_146113


namespace gambler_final_amount_l1461_146140

theorem gambler_final_amount :
  let initial_money := 100
  let win_multiplier := (3/2 : ℚ)
  let loss_multiplier := (1/2 : ℚ)
  let final_multiplier := (win_multiplier * loss_multiplier)^4
  let final_amount := initial_money * final_multiplier
  final_amount = (8100 / 256) :=
by
  sorry

end gambler_final_amount_l1461_146140


namespace find_multiplier_l1461_146112

theorem find_multiplier (x n : ℤ) (h : 2 * n + 20 = x * n - 4) (hn : n = 4) : x = 8 :=
by
  sorry

end find_multiplier_l1461_146112


namespace scarlet_savings_l1461_146125

theorem scarlet_savings :
  ∀ (initial_savings cost_of_earrings cost_of_necklace amount_left : ℕ),
    initial_savings = 80 →
    cost_of_earrings = 23 →
    cost_of_necklace = 48 →
    amount_left = initial_savings - (cost_of_earrings + cost_of_necklace) →
    amount_left = 9 :=
by
  intros initial_savings cost_of_earrings cost_of_necklace amount_left h_is h_earrings h_necklace h_left
  rw [h_is, h_earrings, h_necklace] at h_left
  exact h_left

end scarlet_savings_l1461_146125


namespace modulus_of_z_l1461_146137

noncomputable def z : ℂ := sorry
def condition (z : ℂ) : Prop := z * (1 - Complex.I) = 2 * Complex.I

theorem modulus_of_z (hz : condition z) : Complex.abs z = Real.sqrt 2 := sorry

end modulus_of_z_l1461_146137


namespace countSumPairs_correct_l1461_146142

def countSumPairs (n : ℕ) : ℕ :=
  n / 2

theorem countSumPairs_correct (n : ℕ) : countSumPairs n = n / 2 := by
  sorry

end countSumPairs_correct_l1461_146142


namespace math_problem_l1461_146160
noncomputable def sum_of_terms (a b c d : ℕ) : ℕ := a + b + c + d

theorem math_problem
  (x y : ℝ)
  (h₁ : x + y = 5)
  (h₂ : 5 * x * y = 7) :
  ∃ a b c d : ℕ, 
  x = (a + b * Real.sqrt c) / d ∧
  a = 25 ∧ b = 1 ∧ c = 485 ∧ d = 10 ∧ sum_of_terms a b c d = 521 := by
sorry

end math_problem_l1461_146160


namespace unique_integer_sequence_l1461_146103

theorem unique_integer_sequence :
  ∃ a : ℕ → ℤ, a 1 = 1 ∧ a 2 > 1 ∧ ∀ n ≥ 1, (a (n + 1))^3 + 1 = a n * a (n + 2) :=
sorry

end unique_integer_sequence_l1461_146103


namespace solve_system_of_inequalities_l1461_146105

theorem solve_system_of_inequalities {x : ℝ} :
  (|x^2 + 5 * x| < 6) ∧ (|x + 1| ≤ 1) ↔ (0 ≤ x ∧ x < 2) ∨ (4 < x ∧ x ≤ 6) :=
by
  sorry

end solve_system_of_inequalities_l1461_146105


namespace complement_union_eq_l1461_146151

open Set

-- Define the universe and sets P and Q
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 3, 5}
def Q : Set ℕ := {1, 2, 4}

-- State the theorem
theorem complement_union_eq :
  ((U \ P) ∪ Q) = {1, 2, 4, 6} := by
  sorry

end complement_union_eq_l1461_146151


namespace luke_fish_fillets_l1461_146119

theorem luke_fish_fillets : 
  (∃ (catch_rate : ℕ) (days : ℕ) (fillets_per_fish : ℕ), catch_rate = 2 ∧ days = 30 ∧ fillets_per_fish = 2 → 
  (catch_rate * days * fillets_per_fish = 120)) :=
by
  sorry

end luke_fish_fillets_l1461_146119


namespace kosher_clients_count_l1461_146184

def T := 30
def V := 7
def VK := 3
def Neither := 18

theorem kosher_clients_count (K : ℕ) : T - Neither = V + K - VK → K = 8 :=
by
  intro h
  sorry

end kosher_clients_count_l1461_146184


namespace eqn_of_line_through_intersection_parallel_eqn_of_line_perpendicular_distance_l1461_146178

-- Proof 1: Line through intersection and parallel
theorem eqn_of_line_through_intersection_parallel :
  ∃ k : ℝ, (9 : ℝ) * (x: ℝ) + (18: ℝ) * (y: ℝ) - 4 = 0 ∧
           (∀ x y : ℝ, (2 * x + 3 * y - 5 = 0) → (7 * x + 15 * y + 1 = 0) → (x + 2 * y + k = 0)) :=
sorry

-- Proof 2: Line perpendicular and specific distance from origin
theorem eqn_of_line_perpendicular_distance :
  ∃ k : ℝ, (∃ m : ℝ, (k = 30 ∨ k = -30) ∧ (4 * (x: ℝ) - 3 * (y: ℝ) + m = 0 ∧ (∃ d : ℝ, d = 6 ∧ (|m| / (4 ^ 2 + (-3) ^ 2).sqrt) = d))) :=
sorry

end eqn_of_line_through_intersection_parallel_eqn_of_line_perpendicular_distance_l1461_146178


namespace annie_ride_miles_l1461_146121

noncomputable def annie_ride_distance : ℕ := 14

theorem annie_ride_miles
  (mike_base_rate : ℝ := 2.5)
  (mike_per_mile_rate : ℝ := 0.25)
  (mike_miles : ℕ := 34)
  (annie_base_rate : ℝ := 2.5)
  (annie_bridge_toll : ℝ := 5.0)
  (annie_per_mile_rate : ℝ := 0.25)
  (annie_miles : ℕ := annie_ride_distance)
  (mike_cost : ℝ := mike_base_rate + mike_per_mile_rate * mike_miles)
  (annie_cost : ℝ := annie_base_rate + annie_bridge_toll + annie_per_mile_rate * annie_miles) :
  mike_cost = annie_cost → annie_miles = 14 := 
by
  sorry

end annie_ride_miles_l1461_146121
