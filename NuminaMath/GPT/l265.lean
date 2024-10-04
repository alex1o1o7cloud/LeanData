import Mathlib

namespace community_members_after_five_years_l265_265071

theorem community_members_after_five_years:
  ∀ (a : ℕ → ℕ),
  a 0 = 20 →
  (∀ k : ℕ, a (k + 1) = 4 * a k - 15) →
  a 5 = 15365 :=
by
  intros a h₀ h₁
  sorry

end community_members_after_five_years_l265_265071


namespace platform_length_correct_l265_265780

noncomputable def length_of_platform (train_length: ℝ) (train_speed_kmh: ℝ) (crossing_time: ℝ): ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

theorem platform_length_correct :
  length_of_platform 250 55 50.395968322534195 ≈ 520.00 :=
begin
  -- We setup the conditions
  let train_length := 250
  let train_speed_kmh := 55
  let crossing_time := 50.395968322534195

  -- Calculate the speed in m/s
  let train_speed_ms := train_speed_kmh * 1000 / 3600

  -- Calculate the total distance
  let total_distance := train_speed_ms * crossing_time

  -- Determine the platform length
  let platform_length := total_distance - train_length

  -- Assert that the platform length is approximately 520
  have : abs (platform_length - 520) < 0.0001, by sorry,

  exact this,
end

end platform_length_correct_l265_265780


namespace xy_problem_l265_265306

theorem xy_problem (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 :=
by
  sorry

end xy_problem_l265_265306


namespace product_b2_b7_l265_265869

def is_increasing_arithmetic_sequence (bs : ℕ → ℤ) :=
  ∀ n m : ℕ, n < m → bs n < bs m

def arithmetic_sequence (bs : ℕ → ℤ) (d : ℤ) :=
  ∀ n : ℕ, bs (n + 1) - bs n = d

theorem product_b2_b7 (bs : ℕ → ℤ) (d : ℤ) (h_incr : is_increasing_arithmetic_sequence bs)
    (h_arith : arithmetic_sequence bs d)
    (h_prod : bs 4 * bs 5 = 10) :
    bs 2 * bs 7 = -224 ∨ bs 2 * bs 7 = -44 :=
by
  sorry

end product_b2_b7_l265_265869


namespace janet_final_lives_l265_265007

-- Given conditions
def initial_lives : ℕ := 47
def lives_lost_in_game : ℕ := 23
def points_collected : ℕ := 1840
def lives_per_100_points : ℕ := 2
def penalty_per_200_points : ℕ := 1

-- Definitions based on conditions
def remaining_lives_after_game : ℕ := initial_lives - lives_lost_in_game
def lives_earned_from_points : ℕ := (points_collected / 100) * lives_per_100_points
def lives_lost_due_to_penalties : ℕ := points_collected / 200

-- Theorem statement
theorem janet_final_lives : remaining_lives_after_game + lives_earned_from_points - lives_lost_due_to_penalties = 51 :=
by
  sorry

end janet_final_lives_l265_265007


namespace sum_in_range_l265_265614

theorem sum_in_range :
  let a := (27 : ℚ) / 8
  let b := (22 : ℚ) / 5
  let c := (67 : ℚ) / 11
  13 < a + b + c ∧ a + b + c < 14 :=
by
  sorry

end sum_in_range_l265_265614


namespace xy_problem_l265_265308

theorem xy_problem (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 :=
by
  sorry

end xy_problem_l265_265308


namespace min_cuts_to_one_meter_pieces_l265_265618

theorem min_cuts_to_one_meter_pieces (x y : ℕ) (hx : x + y = 30) (hl : 3 * x + 4 * y = 100) : (2 * x + 3 * y) = 70 := 
by sorry

end min_cuts_to_one_meter_pieces_l265_265618


namespace find_common_difference_l265_265552

section
variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

-- Condition: S_n represents the sum of the first n terms of the arithmetic sequence {a_n}
def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop := 
  S n = (n * (2 * a 1 + (n - 1) * d)) / 2

-- Condition: 2S_3 = 3S_2 + 6
def arithmetic_condition (S : ℕ → ℝ) : Prop :=
  2 * S 3 = 3 * S 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem find_common_difference 
  (h₁ : sum_arithmetic_sequence S a 2)
  (h₂ : sum_arithmetic_sequence S a 3)
  (h₃ : arithmetic_condition S) :
  d = 2 :=
sorry
end

end find_common_difference_l265_265552


namespace christina_age_half_in_five_years_l265_265807

theorem christina_age_half_in_five_years (C Y : ℕ) 
  (h1 : C + 5 = Y / 2)
  (h2 : 21 = 3 * C / 5) :
  Y = 80 :=
sorry

end christina_age_half_in_five_years_l265_265807


namespace system_inequalities_1_system_inequalities_2_l265_265609

theorem system_inequalities_1 (x: ℝ):
  (4 * (x + 1) ≤ 7 * x + 10) → (x - 5 < (x - 8)/3) → (-2 ≤ x ∧ x < 7 / 2) :=
by
  intros h1 h2
  sorry

theorem system_inequalities_2 (x: ℝ):
  (x - 3 * (x - 2) ≥ 4) → ((2 * x - 1) / 5 ≥ (x + 1) / 2) → (x ≤ -7) :=
by
  intros h1 h2
  sorry

end system_inequalities_1_system_inequalities_2_l265_265609


namespace alan_tickets_l265_265912

variables (A M : ℕ)

def condition1 := A + M = 150
def condition2 := M = 5 * A - 6

theorem alan_tickets : A = 26 :=
by
  have h1 : condition1 A M := sorry
  have h2 : condition2 A M := sorry
  sorry

end alan_tickets_l265_265912


namespace fraction_distance_walked_by_first_class_l265_265450

namespace CulturalCenterProblem

def walking_speed : ℝ := 4
def bus_speed_with_students : ℝ := 40
def bus_speed_empty : ℝ := 60

theorem fraction_distance_walked_by_first_class :
  ∃ (x : ℝ), 
    (x / walking_speed) = ((1 - x) / bus_speed_with_students) + ((1 - 2 * x) / bus_speed_empty)
    ∧ x = 5 / 37 :=
by
  sorry

end CulturalCenterProblem

end fraction_distance_walked_by_first_class_l265_265450


namespace arithmetic_sequence_term_l265_265499

theorem arithmetic_sequence_term (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : S 4 = 6)
    (h2 : 2 * (a 3) - (a 2) = 6)
    (h_sum : ∀ n, S n = n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))) : 
  a 1 = -3 := 
by sorry

end arithmetic_sequence_term_l265_265499


namespace prove_scientific_notation_l265_265221

def scientific_notation_correct : Prop :=
  340000 = 3.4 * (10 ^ 5)

theorem prove_scientific_notation : scientific_notation_correct :=
  by
    sorry

end prove_scientific_notation_l265_265221


namespace ratatouille_cost_per_quart_l265_265744

def eggplants := 88 * 0.22
def zucchini := 60.8 * 0.15
def tomatoes := 73.6 * 0.25
def onions := 43.2 * 0.07
def basil := (16 / 4) * 2.70
def bell_peppers := 12 * 0.20

def total_cost := eggplants + zucchini + tomatoes + onions + basil + bell_peppers
def yield := 4.5

def cost_per_quart := total_cost / yield

theorem ratatouille_cost_per_quart : cost_per_quart = 14.02 := 
by
  unfold cost_per_quart total_cost eggplants zucchini tomatoes onions basil bell_peppers 
  sorry

end ratatouille_cost_per_quart_l265_265744


namespace polynomial_factorization_l265_265893

-- Definitions used in the conditions
def given_polynomial (a b c : ℝ) : ℝ :=
  a^3 * (b^2 - c^2) + b^3 * (c^2 - a^2) + c^3 * (a^2 - b^2)

def p (a b c : ℝ) : ℝ := -(a * b + a * c + b * c)

-- The Lean 4 statement to be proved
theorem polynomial_factorization (a b c : ℝ) :
  given_polynomial a b c = (a - b) * (b - c) * (c - a) * p a b c :=
by
  sorry

end polynomial_factorization_l265_265893


namespace min_value_of_sum_squares_l265_265872

noncomputable def min_value_sum_squares 
  (y1 y2 y3 : ℝ) (h1 : 2 * y1 + 3 * y2 + 4 * y3 = 120) 
  (h2 : 0 < y1) (h3 : 0 < y2) (h4 : 0 < y3) : ℝ :=
  y1^2 + y2^2 + y3^2

theorem min_value_of_sum_squares 
  (y1 y2 y3 : ℝ) (h1 : 2 * y1 + 3 * y2 + 4 * y3 = 120) 
  (h2 : 0 < y1) (h3 : 0 < y2) (h4 : 0 < y3) : 
  min_value_sum_squares y1 y2 y3 h1 h2 h3 h4 = 14400 / 29 := 
sorry

end min_value_of_sum_squares_l265_265872


namespace larger_interior_angle_trapezoid_pavilion_l265_265326

theorem larger_interior_angle_trapezoid_pavilion :
  let n := 12
  let central_angle := 360 / n
  let smaller_angle := 180 - (central_angle / 2)
  let larger_angle := 180 - smaller_angle
  larger_angle = 97.5 :=
by
  sorry

end larger_interior_angle_trapezoid_pavilion_l265_265326


namespace greatest_two_digit_multiple_of_17_l265_265175

theorem greatest_two_digit_multiple_of_17 : ∃ n, (n < 100) ∧ (n % 17 = 0) ∧ (∀ m, (m < 100) ∧ (m % 17 = 0) → m ≤ n) := by
  let multiples := [17, 34, 51, 68, 85]
  have ht : ∀ m, m ∈ multiples → (m < 100) ∧ (m % 17 = 0) := by
    intros m hm
    cases hm with
    | inl h => exact ⟨by norm_num, by norm_num⟩
    | inr hm' =>
      cases hm' with
      | inl h => exact ⟨by norm_num, by norm_num⟩
      | inr hm'' =>
        cases hm'' with
        | inl h => exact ⟨by norm_num, by norm_num⟩
        | inr hm''' =>
          cases hm''' with
          | inl h => exact ⟨by norm_num, by norm_num⟩
          | inr hm'''' =>
            cases hm'''' with
            | inl h => exact ⟨by norm_num, by norm_num⟩
            | inr hm => cases hm
  
  use 85
  split
  { norm_num },
  split
  { exact (by norm_num : 85 % 17 = 0) },
  intro m
  intro h
  exact list.minimum_mem multiples h

end greatest_two_digit_multiple_of_17_l265_265175


namespace sale_price_for_50_percent_profit_l265_265896

theorem sale_price_for_50_percent_profit
  (C L: ℝ)
  (h1: 892 - C = C - L)
  (h2: 1005 = 1.5 * C) :
  1.5 * C = 1005 :=
by
  sorry

end sale_price_for_50_percent_profit_l265_265896


namespace speed_of_boat_in_still_water_l265_265613

variable (x : ℝ)

theorem speed_of_boat_in_still_water (h : 10 = (x + 5) * 0.4) : x = 20 :=
sorry

end speed_of_boat_in_still_water_l265_265613


namespace triangle_side_length_mod_l265_265287

theorem triangle_side_length_mod {a d x : ℕ} 
  (h_equilateral : ∃ (a : ℕ), 3 * a = 1 + d + x)
  (h_triangle : ∀ {a d x : ℕ}, 1 + d > x ∧ 1 + x > d ∧ d + x > 1)
  : d % 3 = 1 :=
by
  sorry

end triangle_side_length_mod_l265_265287


namespace head_ninth_flip_probability_l265_265683

noncomputable def fair_coin_flip (n : ℕ) : Probability := 
  if n % 2 = 0 then 1 / 2 else 1 / 2

theorem head_ninth_flip_probability :
  P(fair_coin_flip 9 = 1 / 2) :=
sorry

end head_ninth_flip_probability_l265_265683


namespace johns_beef_order_l265_265130

theorem johns_beef_order (B : ℕ)
  (h1 : 8 * B + 6 * (2 * B) = 14000) :
  B = 1000 :=
by
  sorry

end johns_beef_order_l265_265130


namespace factorization_problem_l265_265997

theorem factorization_problem (x : ℝ) :
  (x^4 + x^2 - 4) * (x^4 + x^2 + 3) + 10 =
  (x^2 + x + 1) * (x^2 - x + 1) * (x^2 + 2) * (x + 1) * (x - 1) :=
sorry

end factorization_problem_l265_265997


namespace profit_percentage_is_25_percent_l265_265952

noncomputable def costPrice : ℝ := 47.50
noncomputable def markedPrice : ℝ := 64.54
noncomputable def discountRate : ℝ := 0.08

noncomputable def discountAmount : ℝ := discountRate * markedPrice
noncomputable def sellingPrice : ℝ := markedPrice - discountAmount
noncomputable def profit : ℝ := sellingPrice - costPrice
noncomputable def profitPercentage : ℝ := (profit / costPrice) * 100

theorem profit_percentage_is_25_percent :
  profitPercentage = 25 := by
  sorry

end profit_percentage_is_25_percent_l265_265952


namespace correct_operation_only_l265_265630

theorem correct_operation_only (a b x y : ℝ) : 
  (2 + Real.sqrt 2 ≠ 2 * Real.sqrt 2) ∧ 
  (4 * x^2 * y - x^2 * y ≠ 3) ∧ 
  ((a + b)^2 ≠ a^2 + b^2) ∧
  ((ab)^3 = a^3 * b^3) := 
by 
  sorry

end correct_operation_only_l265_265630


namespace factorize_expression_l265_265486

-- Variables x and y are real numbers
variables (x y : ℝ)

-- Theorem statement
theorem factorize_expression : 3 * x^2 - 12 * y^2 = 3 * (x - 2 * y) * (x + 2 * y) :=
sorry

end factorize_expression_l265_265486


namespace radius_of_circle_l265_265403

theorem radius_of_circle :
  ∀ (r : ℝ), (π * r^2 = 2.5 * 2 * π * r) → r = 5 :=
by sorry

end radius_of_circle_l265_265403


namespace find_common_difference_l265_265553

section
variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

-- Condition: S_n represents the sum of the first n terms of the arithmetic sequence {a_n}
def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop := 
  S n = (n * (2 * a 1 + (n - 1) * d)) / 2

-- Condition: 2S_3 = 3S_2 + 6
def arithmetic_condition (S : ℕ → ℝ) : Prop :=
  2 * S 3 = 3 * S 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem find_common_difference 
  (h₁ : sum_arithmetic_sequence S a 2)
  (h₂ : sum_arithmetic_sequence S a 3)
  (h₃ : arithmetic_condition S) :
  d = 2 :=
sorry
end

end find_common_difference_l265_265553


namespace brick_width_is_correct_l265_265070

-- Defining conditions
def wall_length : ℝ := 200 -- wall length in cm
def wall_width : ℝ := 300 -- wall width in cm
def wall_height : ℝ := 2   -- wall height in cm
def brick_length : ℝ := 25 -- brick length in cm
def brick_height : ℝ := 6  -- brick height in cm
def num_bricks : ℝ := 72.72727272727273

-- Total volume of wall
def vol_wall : ℝ := wall_length * wall_width * wall_height

-- Volume of one brick
def vol_brick (width : ℝ) : ℝ := brick_length * width * brick_height

-- Proof statement
theorem brick_width_is_correct : ∃ width : ℝ, vol_wall = vol_brick width * num_bricks ∧ width = 11 :=
by
  sorry

end brick_width_is_correct_l265_265070


namespace retailer_overhead_expenses_l265_265664

theorem retailer_overhead_expenses (purchase_price selling_price profit_percent : ℝ) (overhead_expenses : ℝ) 
  (h1 : purchase_price = 225) 
  (h2 : selling_price = 300) 
  (h3 : profit_percent = 25) 
  (h4 : selling_price = (purchase_price + overhead_expenses) * (1 + profit_percent / 100)) : 
  overhead_expenses = 15 := 
by
  sorry

end retailer_overhead_expenses_l265_265664


namespace number_of_members_l265_265648

theorem number_of_members (n h : ℕ) (h1 : n * n * h = 362525) : n = 5 :=
sorry

end number_of_members_l265_265648


namespace arithmetic_sequence_common_difference_l265_265561

theorem arithmetic_sequence_common_difference (a_1 d : ℝ) (S : ℕ → ℝ) 
    (h1 : S 2 = 2 * a_1 + d)
    (h2 : S 3 = 3 * a_1 + 3 * d)
    (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 := 
by
  sorry

end arithmetic_sequence_common_difference_l265_265561


namespace good_students_count_l265_265364

theorem good_students_count (E B : ℕ) 
    (h1 : E + B = 25) 
    (h2 : ∀ (e ∈ finset.range 5), B > 12) 
    (h3 : ∀ (e ∈ finset.range 20), B = 3 * (E - 1)) :
    E = 5 ∨ E = 7 :=
by
  sorry

end good_students_count_l265_265364


namespace remainder_division_l265_265768

theorem remainder_division (x : ℕ) (dividend : ℕ) (divisor : ℕ) (correct_remainder : ℕ) 
    (h1 : dividend = 2^202 + 202) 
    (h2 : divisor = 2^101 + 2^51 + 1) 
    (h3 : correct_remainder = 201) :
    dividend % divisor = correct_remainder := 
by 
    rw [h1, h2, h3]
    sorry

end remainder_division_l265_265768


namespace good_students_options_l265_265344

variables (E B : ℕ)

-- Define the condition that the class has 25 students
def total_students : Prop := E + B = 25

-- Define the condition given by the first group of students
def first_group_condition : Prop := B > 12

-- Define the condition given by the second group of students
def second_group_condition : Prop := B = 3 * (E - 1)

-- Define the problem statement
theorem good_students_options (E B : ℕ) :
  total_students E B → first_group_condition B → second_group_condition E B → (E = 5 ∨ E = 7) :=
by
  intros h1 h2 h3
  sorry

end good_students_options_l265_265344


namespace find_y_l265_265491

theorem find_y : ∃ y : ℚ, y + 2/3 = 1/4 - (2/5) * 2 ∧ y = -511/420 :=
by
  sorry

end find_y_l265_265491


namespace john_labor_cost_l265_265376

def plank_per_tree : ℕ := 25
def table_cost : ℕ := 300
def profit : ℕ := 12000
def trees_chopped : ℕ := 30
def planks_per_table : ℕ := 15
def total_table_revenue := (trees_chopped * plank_per_tree / planks_per_table) * table_cost
def labor_cost := total_table_revenue - profit

theorem john_labor_cost :
  labor_cost = 3000 :=
by
  sorry

end john_labor_cost_l265_265376


namespace arithmetic_sequence_common_difference_l265_265546

theorem arithmetic_sequence_common_difference 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : S 3 = a 1 + a 2 + a 3)
  (h2 : S 2 = a 1 + a 2)
  (h3 : a 2 = a 1 + d)
  (h4 : a 3 = a 1 + 2 * d)
  (h5 : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l265_265546


namespace largest_possible_sum_l265_265948

-- Define whole numbers
def whole_numbers : Set ℕ := Set.univ

-- Define the given conditions
variables (a b : ℕ)
axiom h1 : a ∈ whole_numbers
axiom h2 : b ∈ whole_numbers
axiom h3 : a * b = 48

-- Prove the largest sum condition
theorem largest_possible_sum : a + b ≤ 49 :=
sorry

end largest_possible_sum_l265_265948


namespace sequence_term_2023_l265_265833

theorem sequence_term_2023 
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ) 
  (h : ∀ n, 2 * S n = a n * (a n + 1)) : 
  a 2023 = 2023 :=
sorry

end sequence_term_2023_l265_265833


namespace greatest_two_digit_multiple_of_17_l265_265173

theorem greatest_two_digit_multiple_of_17 : ∀ n, n ∈ finset.range 100 → n % 17 = 0 → n ≤ 85 →
  (∀ m, m ∈ finset.range 100 → m % 17 = 0 → m > n → m ≥ 100) →
  n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l265_265173


namespace residue_of_neg_2035_mod_47_l265_265817

theorem residue_of_neg_2035_mod_47 : (-2035 : ℤ) % 47 = 33 := 
by
  sorry

end residue_of_neg_2035_mod_47_l265_265817


namespace lcm_gcf_ratio_120_504_l265_265436

theorem lcm_gcf_ratio_120_504 : 
  let a := 120
  let b := 504
  (Int.lcm a b) / (Int.gcd a b) = 105 := by
  sorry

end lcm_gcf_ratio_120_504_l265_265436


namespace initially_working_machines_l265_265787

theorem initially_working_machines (N R x : ℝ) 
  (h1 : N * R = x / 3) 
  (h2 : 45 * R = x / 2) : 
  N = 30 := by
  sorry

end initially_working_machines_l265_265787


namespace blue_lipstick_count_l265_265604

def total_students : Nat := 200

def colored_lipstick_students (total : Nat) : Nat :=
  total / 2

def red_lipstick_students (colored : Nat) : Nat :=
  colored / 4

def blue_lipstick_students (red : Nat) : Nat :=
  red / 5

theorem blue_lipstick_count :
  blue_lipstick_students (red_lipstick_students (colored_lipstick_students total_students)) = 5 := 
sorry

end blue_lipstick_count_l265_265604


namespace mr_smith_markers_l265_265387

theorem mr_smith_markers :
  ∀ (initial_markers : ℕ) (total_markers : ℕ) (markers_per_box : ℕ) 
  (number_of_boxes : ℕ),
  initial_markers = 32 → 
  total_markers = 86 → 
  markers_per_box = 9 → 
  number_of_boxes = (total_markers - initial_markers) / markers_per_box →
  number_of_boxes = 6 :=
by
  intros initial_markers total_markers markers_per_box number_of_boxes h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃] at h₄
  simp only [Nat.sub] at h₄
  exact h₄

end mr_smith_markers_l265_265387


namespace good_students_count_l265_265348

noncomputable def student_count := 25

def is_good_student (s : Nat) := s ≤ student_count

def is_troublemaker (s : Nat) := s ≤ student_count

def always_tell_truth (s : Nat) := is_good_student s

def always_lie (s : Nat) := is_troublemaker s

def condition1 (E B : Nat) := E + B = student_count

def condition2 := ∀ (x : Nat), x ≤ 5 → is_good_student x → 
  ∃ (B : Nat), B > 24 / 2

def condition3 := ∀ (x : Nat), x ≤ 20 → is_troublemaker x → 
  ∃ (B E : Nat), B = 3 * (E - 1) ∧ E + B = student_count

theorem good_students_count :
  ∃ (E : Nat), condition1 E (student_count - E) ∧ condition2 ∧ condition3 :=
sorry  -- the actual proof is not required

end good_students_count_l265_265348


namespace jack_finishes_in_16_days_l265_265006

noncomputable def pages_in_book : ℕ := 285
noncomputable def weekday_reading_rate : ℕ := 23
noncomputable def weekend_reading_rate : ℕ := 35
noncomputable def weekdays_per_week : ℕ := 5
noncomputable def weekends_per_week : ℕ := 2
noncomputable def weekday_skipped : ℕ := 1
noncomputable def weekend_skipped : ℕ := 1

noncomputable def pages_per_week : ℕ :=
  (weekdays_per_week - weekday_skipped) * weekday_reading_rate + 
  (weekends_per_week - weekend_skipped) * weekend_reading_rate

noncomputable def weeks_needed : ℕ :=
  pages_in_book / pages_per_week

noncomputable def pages_left_after_weeks : ℕ :=
  pages_in_book % pages_per_week

noncomputable def extra_days_needed (pages_left : ℕ) : ℕ :=
  if pages_left > weekend_reading_rate then 2
  else if pages_left > weekday_reading_rate then 2
  else 1

noncomputable def total_days_needed : ℕ :=
  weeks_needed * 7 + extra_days_needed (pages_left_after_weeks)

theorem jack_finishes_in_16_days : total_days_needed = 16 := by
  sorry

end jack_finishes_in_16_days_l265_265006


namespace each_girl_brought_2_cups_l265_265427

-- Definitions of the conditions
def total_students : ℕ := 30
def boys : ℕ := 10
def total_cups : ℕ := 90
def cups_per_boy : ℕ := 5
def girls : ℕ := total_students - boys

def total_cups_by_boys : ℕ := boys * cups_per_boy
def total_cups_by_girls : ℕ := total_cups - total_cups_by_boys
def cups_per_girl : ℕ := total_cups_by_girls / girls

-- The statement with the correct answer
theorem each_girl_brought_2_cups (
  h1 : total_students = 30,
  h2 : boys = 10,
  h3 : total_cups = 90,
  h4 : cups_per_boy = 5,
  h5 : total_cups_by_boys = boys * cups_per_boy,
  h6 : total_cups_by_girls = total_cups - total_cups_by_boys,
  h7 : cups_per_girl = total_cups_by_girls / girls
) : cups_per_girl = 2 := 
sorry

end each_girl_brought_2_cups_l265_265427


namespace gary_money_left_l265_265691

theorem gary_money_left (initial_amount spent_amount remaining_amount : ℕ)
  (h1 : initial_amount = 73)
  (h2 : spent_amount = 55)
  (h3 : remaining_amount = 18) : initial_amount - spent_amount = remaining_amount := 
by 
  sorry

end gary_money_left_l265_265691


namespace length_of_PB_l265_265662

theorem length_of_PB 
  (AB BC : ℝ) 
  (PA PD PC PB : ℝ)
  (h1 : AB = 2 * BC) 
  (h2 : PA = 5) 
  (h3 : PD = 12) 
  (h4 : PC = 13) 
  (h5 : PA^2 + PB^2 = (AB^2 + BC^2) / 5) -- derived from question
  (h6 : PB^2 = ((2 * BC)^2) - PA^2) : 
  PB = 10.5 :=
by 
  -- We would insert proof steps here (not required as per instructions)
  sorry

end length_of_PB_l265_265662


namespace half_of_original_amount_l265_265521

theorem half_of_original_amount (h : ∃ (m : ℚ), (4/7 : ℚ) * m = 24) : 
  ∃ (half_m : ℚ), half_m = 21 :=
by
  obtain ⟨m, hm⟩ := h
  have original := m
  have half_orig := (1/2 : ℚ) * original
  have target := (7/4 : ℚ) * 24 / 2
  use half_orig
  rw [←hm]
  have fact : (4 / 7) * original * (7 / 4) = original := by sorry
  have eq1 : (7 / 4) * 24 = original := eq.trans (mul_eq_mul_right_iff.mpr (oreq_of_ne_zero (by norm_num)) (by norm_num) hm.symm)
  have eq2 := eq.trans eq1 div_eq_div_right_iff nonzero_of_ne_zero (by norm_num)
  rw [eq2] at this
  exact sorry


end half_of_original_amount_l265_265521


namespace prove_common_difference_l265_265575

variable {a1 d : ℤ}
variable (S : ℕ → ℤ)
variable (arith_seq : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
def sum_first_n (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Define the condition given in the problem
def problem_condition : Prop := 2 * sum_first_n 3 = 3 * sum_first_n 2 + 6

-- Define that the common difference d is 2
def d_value_is_2 : Prop := d = 2

theorem prove_common_difference (h : problem_condition) : d_value_is_2 :=
by
  sorry

end prove_common_difference_l265_265575


namespace solve_for_k_l265_265829

noncomputable def base_k_representation (k : ℕ) (r : ℚ) := 
  ∑' n : ℕ, r.num * k^(-n*r.denom) / (1 - k^(-n*r.denom))

theorem solve_for_k (k : ℕ) (h_positive : k > 0) 
  (h_representation : base_k_representation k (2 * k^1 + 3 * k^2) = 7 / 51) : 
  k = 16 :=
by
  sorry

end solve_for_k_l265_265829


namespace temperature_reading_l265_265750

theorem temperature_reading (scale_min scale_max : ℝ) (arrow : ℝ) (h1 : scale_min = -6.0) (h2 : scale_max = -5.5) (h3 : scale_min < arrow) (h4 : arrow < scale_max) : arrow = -5.7 :=
sorry

end temperature_reading_l265_265750


namespace max_length_interval_l265_265697

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := ((m ^ 2 + m) * x - 1) / (m ^ 2 * x)

theorem max_length_interval (a b m : ℝ) (h1 : m ≠ 0) (h2 : ∀ x, f m x = x → x ∈ Set.Icc a b) :
  |b - a| = (2 * Real.sqrt 3) / 3 := sorry

end max_length_interval_l265_265697


namespace sumsquare_properties_l265_265964

theorem sumsquare_properties {a b c d e f g h i : ℕ} (hc1 : a + b + c = d + e + f) 
(hc2 : d + e + f = g + h + i) 
(hc3 : a + e + i = d + e + f) 
(hc4 : c + e + g = d + e + f) : 
∃ m : ℕ, m % 3 = 0 ∧ (a ≤ (2 * m / 3 - 1)) ∧ (b ≤ (2 * m / 3 - 1)) ∧ (c ≤ (2 * m / 3 - 1)) ∧ (d ≤ (2 * m / 3 - 1)) ∧ (e ≤ (2 * m / 3 - 1)) ∧ (f ≤ (2 * m / 3 - 1)) ∧ (g ≤ (2 * m / 3 - 1)) ∧ (h ≤ (2 * m / 3 - 1)) ∧ (i ≤ (2 * m / 3 - 1)) := 
by {
  sorry
}

end sumsquare_properties_l265_265964


namespace good_students_options_l265_265342

variables (E B : ℕ)

-- Define the condition that the class has 25 students
def total_students : Prop := E + B = 25

-- Define the condition given by the first group of students
def first_group_condition : Prop := B > 12

-- Define the condition given by the second group of students
def second_group_condition : Prop := B = 3 * (E - 1)

-- Define the problem statement
theorem good_students_options (E B : ℕ) :
  total_students E B → first_group_condition B → second_group_condition E B → (E = 5 ∨ E = 7) :=
by
  intros h1 h2 h3
  sorry

end good_students_options_l265_265342


namespace sum_of_interior_angles_pentagon_l265_265615

theorem sum_of_interior_angles_pentagon : 
  let n := 5 in
  180 * (n - 2) = 540 :=
  by
  -- Introducing the variable n for the number of sides
  let n := 5
  -- Using the given formula to calculate the sum of the interior angles
  have h : 180 * (n - 2) = 540 := by sorry
  exact h

end sum_of_interior_angles_pentagon_l265_265615


namespace equalized_distance_l265_265847

noncomputable def wall_width : ℝ := 320 -- wall width in centimeters
noncomputable def poster_count : ℕ := 6 -- number of posters
noncomputable def poster_width : ℝ := 30 -- width of each poster in centimeters
noncomputable def equal_distance : ℝ := 20 -- equal distance in centimeters to be proven

theorem equalized_distance :
  let total_posters_width := poster_count * poster_width
  let remaining_space := wall_width - total_posters_width
  let number_of_spaces := poster_count + 1
  remaining_space / number_of_spaces = equal_distance :=
by {
  sorry
}

end equalized_distance_l265_265847


namespace students_spend_185_minutes_in_timeout_l265_265020

variable (tR tF tS t_total : ℕ)

-- Conditions
def running_timeouts : ℕ := 5
def food_timeouts : ℕ := 5 * running_timeouts - 1
def swearing_timeouts : ℕ := food_timeouts / 3
def total_timeouts : ℕ := running_timeouts + food_timeouts + swearing_timeouts
def timeout_duration : ℕ := 5

-- Total time spent in time-out
def total_timeout_minutes : ℕ := total_timeouts * timeout_duration

theorem students_spend_185_minutes_in_timeout :
  total_timeout_minutes = 185 :=
by
  -- The answer is directly given by the conditions and the correct answer identified.
  sorry

end students_spend_185_minutes_in_timeout_l265_265020


namespace solve_X_l265_265995

def diamond (X Y : ℝ) : ℝ := 4 * X - 3 * Y + 2

theorem solve_X :
  (∃ X : ℝ, diamond X 6 = 35) ↔ (X = 51 / 4) := by
  sorry

end solve_X_l265_265995


namespace intersecting_lines_solution_l265_265404

theorem intersecting_lines_solution (x y b : ℝ) 
  (h₁ : y = 2 * x - 5)
  (h₂ : y = 3 * x + b)
  (hP : x = 1 ∧ y = -3) : 
  b = -6 ∧ x = 1 ∧ y = -3 := by
  sorry

end intersecting_lines_solution_l265_265404


namespace necessary_condition_l265_265433

variable (P Q : Prop)

/-- If the presence of the dragon city's flying general implies that
    the horses of the Hu people will not cross the Yin Mountains,
    then "not letting the horses of the Hu people cross the Yin Mountains"
    is a necessary condition for the presence of the dragon city's flying general. -/
theorem necessary_condition (h : P → Q) : ¬Q → ¬P :=
by sorry

end necessary_condition_l265_265433


namespace final_price_of_pencil_l265_265046

-- Define the initial constants
def initialCost : ℝ := 4.00
def christmasDiscount : ℝ := 0.63
def seasonalDiscountRate : ℝ := 0.07
def finalDiscountRate : ℝ := 0.05
def taxRate : ℝ := 0.065

-- Define the steps of the problem concisely
def priceAfterChristmasDiscount := initialCost - christmasDiscount
def priceAfterSeasonalDiscount := priceAfterChristmasDiscount * (1 - seasonalDiscountRate)
def priceAfterFinalDiscount := priceAfterSeasonalDiscount * (1 - finalDiscountRate)
def finalPrice := priceAfterFinalDiscount * (1 + taxRate)

-- The theorem to be proven
theorem final_price_of_pencil :
  abs (finalPrice - 3.17) < 0.01 := by
  sorry

end final_price_of_pencil_l265_265046


namespace tom_gave_8_boxes_l265_265913

-- Define the given conditions and the question in terms of variables
variables (total_boxes : ℕ) (pieces_per_box : ℕ) (pieces_left : ℕ) (boxes_given : ℕ)

-- Specify the actual values for the given problem
def tom_initial_pieces := total_boxes * pieces_per_box
def pieces_given := tom_initial_pieces - pieces_left
def calculated_boxes_given := pieces_given / pieces_per_box

-- Prove the number of boxes Tom gave to his little brother
theorem tom_gave_8_boxes
  (h1 : total_boxes = 14)
  (h2 : pieces_per_box = 3)
  (h3 : pieces_left = 18)
  (h4 : calculated_boxes_given = boxes_given) :
  boxes_given = 8 :=
by
  sorry

end tom_gave_8_boxes_l265_265913


namespace common_difference_of_arithmetic_sequence_l265_265530

variable {a₁ d : ℕ}
def S (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem common_difference_of_arithmetic_sequence (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l265_265530


namespace dot_product_of_PA_PB_l265_265114

theorem dot_product_of_PA_PB
  (A B P: ℝ × ℝ)
  (h_circle : ∀ (x y : ℝ), x ^ 2 + y ^ 2 + 4 * x - 5 = 0 → (x, y) = A ∨ (x, y) = B)
  (h_midpoint : (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 1)
  (h_x_axis_intersect : P.2 = 0 ∧ (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) = -5) :
  (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) = -5 :=
sorry

end dot_product_of_PA_PB_l265_265114


namespace factorize_expression_l265_265484

-- Variables x and y are real numbers
variables (x y : ℝ)

-- Theorem statement
theorem factorize_expression : 3 * x^2 - 12 * y^2 = 3 * (x - 2 * y) * (x + 2 * y) :=
sorry

end factorize_expression_l265_265484


namespace particle_path_count_l265_265661

def lattice_path_count (n : ℕ) : ℕ :=
sorry -- Placeholder for the actual combinatorial function

theorem particle_path_count : lattice_path_count 7 = sorry :=
sorry -- Placeholder for the actual count

end particle_path_count_l265_265661


namespace melissa_games_played_l265_265601

theorem melissa_games_played (total_points : ℕ) (points_per_game : ℕ) (num_games : ℕ) 
  (h1 : total_points = 81) 
  (h2 : points_per_game = 27) 
  (h3 : num_games = total_points / points_per_game) : 
  num_games = 3 :=
by
  -- Proof goes here
  sorry

end melissa_games_played_l265_265601


namespace find_g_l265_265262

-- Define given functions and terms
def f1 (x : ℝ) := 7 * x^4 - 4 * x^3 + 2 * x - 5
def f2 (x : ℝ) := 5 * x^3 - 3 * x^2 + 4 * x - 1
def g (x : ℝ) := -7 * x^4 + 9 * x^3 - 3 * x^2 + 2 * x + 4

-- Theorem to prove that g(x) satisfies the given condition
theorem find_g : ∀ x : ℝ, f1 x + g x = f2 x :=
by 
  -- Alternatively: Proof is required here
  sorry

end find_g_l265_265262


namespace chocolate_bars_gigantic_box_l265_265957

def large_boxes : ℕ := 50
def medium_boxes : ℕ := 25
def small_boxes : ℕ := 10
def chocolate_bars_per_small_box : ℕ := 45

theorem chocolate_bars_gigantic_box : 
  large_boxes * medium_boxes * small_boxes * chocolate_bars_per_small_box = 562500 :=
by
  sorry

end chocolate_bars_gigantic_box_l265_265957


namespace price_increase_decrease_l265_265963

theorem price_increase_decrease (P : ℝ) (x : ℝ) (h : P > 0) :
  (P * (1 + x / 100) * (1 - x / 100) = 0.64 * P) → (x = 60) :=
by
  sorry

end price_increase_decrease_l265_265963


namespace find_common_difference_l265_265554

section
variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

-- Condition: S_n represents the sum of the first n terms of the arithmetic sequence {a_n}
def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop := 
  S n = (n * (2 * a 1 + (n - 1) * d)) / 2

-- Condition: 2S_3 = 3S_2 + 6
def arithmetic_condition (S : ℕ → ℝ) : Prop :=
  2 * S 3 = 3 * S 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem find_common_difference 
  (h₁ : sum_arithmetic_sequence S a 2)
  (h₂ : sum_arithmetic_sequence S a 3)
  (h₃ : arithmetic_condition S) :
  d = 2 :=
sorry
end

end find_common_difference_l265_265554


namespace probability_product_positive_is_5_div_9_l265_265928

noncomputable def probability_positive_product : ℚ :=
  let interval := Set.Icc (-30 : ℝ) 15
  let length_interval := 45
  let length_neg := 30
  let length_pos := 15
  let prob_neg := (length_neg : ℚ) / length_interval
  let prob_pos := (length_pos : ℚ) / length_interval
  let prob_product_pos := prob_neg^2 + prob_pos^2
  prob_product_pos

theorem probability_product_positive_is_5_div_9 :
  probability_positive_product = 5 / 9 :=
by
  sorry

end probability_product_positive_is_5_div_9_l265_265928


namespace magnus_score_in_third_game_l265_265734

-- Definitions and conditions
def game1_score (a b: ℕ) : Prop := a = b + 2
def game2_score (b c: ℕ) : Prop := b = (a + c) / 2
def unique_scores (s: Finset ℕ) : Prop := s.card = 6

def problem_conditions (m1 m2 m3 v1 v2 v3: ℕ) : Prop :=
  -- Positive integer scores and unique across games
  0 < m1 ∧ 0 < m2 ∧ 0 < m3 ∧ 0 < v1 ∧ 0 < v2 ∧ 0 < v3 ∧
  unique_scores {m1, m2, m3, v1, v2, v3} ∧
  -- Win and scoring conditions
  (v1 ≥ 25 ∨ v2 ≥ 25 ∨ v3 = 25) ∧
  ((v1 = 25 → m1 ≤ 23) ∧ (v2 = 25 → m2 ≤ 23)) ∧
  ((v1 > 25 → m1 = v1 - 2) ∧ (v2 > 25 → m2 = v2 - 2)) ∧
  (v1 ≠ v2) ∧ (v2 ≠ v3) ∧ (v1 ≠ v3) ∧
  -- Specific conditions
  v3 = 25 ∧
  -- Each player's second game score is the average of their 1st and 3rd game scores
  game1_score m1 m2 ∧ game2_score m2 m3 ∧
  game1_score v1 v2 ∧ game2_score v2 v3

-- Proving Magnus's score in the 3rd game equals 19 given conditions.
theorem magnus_score_in_third_game (m1 m2 m3 v1 v2 v3: ℕ) :
  problem_conditions m1 m2 m3 v1 v2 v3 → m3 = 19 :=
by
  sorry

end magnus_score_in_third_game_l265_265734


namespace sequence_term_l265_265902

theorem sequence_term (S : ℕ → ℕ) (a : ℕ → ℕ) (n : ℕ) (hn : n > 0)
  (hSn : ∀ n, S n = n^2)
  (hrec : ∀ n, n > 1 → a n = S n - S (n-1)) :
  a n = 2 * n - 1 := by
  -- Base case
  cases n with
  | zero => contradiction  -- n > 0 implies n ≠ 0
  | succ n' =>
    cases n' with
    | zero => sorry  -- When n = 0 + 1 = 1, we need to show a 1 = 2 * 1 - 1 = 1 based on given conditions
    | succ k => sorry -- When n = k + 1, we use the provided recursive relation to prove the statement

end sequence_term_l265_265902


namespace rogers_parents_paid_percentage_l265_265883

variables 
  (house_cost : ℝ)
  (down_payment_percentage : ℝ)
  (remaining_balance_owed : ℝ)
  (down_payment : ℝ := down_payment_percentage * house_cost)
  (remaining_balance_after_down : ℝ := house_cost - down_payment)
  (parents_payment : ℝ := remaining_balance_after_down - remaining_balance_owed)
  (percentage_paid_by_parents : ℝ := (parents_payment / remaining_balance_after_down) * 100)

theorem rogers_parents_paid_percentage
  (h1 : house_cost = 100000)
  (h2 : down_payment_percentage = 0.20)
  (h3 : remaining_balance_owed = 56000) :
  percentage_paid_by_parents = 30 :=
sorry

end rogers_parents_paid_percentage_l265_265883


namespace math_problem_l265_265500

open Real

theorem math_problem (α : ℝ) (h₁ : 0 < α) (h₂ : α < π / 2) (h₃ : cos (2 * π - α) - sin (π - α) = - sqrt 5 / 5) :
  (sin α + cos α = 3 * sqrt 5 / 5) ∧
  (cos (3 * π / 2 + α) ^ 2 + 2 * cos α * cos (π / 2 - α)) / (1 + sin (π / 2 - α) ^ 2) = 4 / 3 :=
by
  sorry

end math_problem_l265_265500


namespace range_of_m_l265_265497

-- Definitions given in the problem
def p (x : ℝ) : Prop := x < -2 ∨ x > 10
def q (x m : ℝ) : Prop := x^2 - 2*x - (m^2 - 1) ≥ 0
def neg_q_sufficient_for_neg_p : Prop :=
  ∀ {x m : ℝ}, (1 - m < x ∧ x < 1 + m) → (-2 ≤ x ∧ x ≤ 10)

-- The statement to prove
theorem range_of_m (m : ℝ) (h1 : m > 0) (h2 : 1 - m ≥ -2) (h3 : 1 + m ≤ 10) :
  0 < m ∧ m ≤ 3 :=
by
  sorry

end range_of_m_l265_265497


namespace ellipse_equation_with_foci_l265_265866

theorem ellipse_equation_with_foci (M N P : ℝ × ℝ)
  (area_triangle : Real) (tan_M tan_N : ℝ)
  (h₁ : area_triangle = 1)
  (h₂ : tan_M = 1 / 2)
  (h₃ : tan_N = -2) :
  ∃ (a b : ℝ), (4 * x^2) / (15 : ℝ) + y^2 / (3 : ℝ) = 1 :=
by
  -- Definitions to meet given conditions would be here
  sorry

end ellipse_equation_with_foci_l265_265866


namespace common_difference_of_arithmetic_sequence_l265_265535

noncomputable def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range n, a i

def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_cond : 2 * S a 3 = 3 * S a 2 + 6) :
  ∃ d : ℝ, d = 2 := sorry

end common_difference_of_arithmetic_sequence_l265_265535


namespace penny_exceeded_by_32_l265_265003

def bulk_price : ℤ := 5
def min_spend_before_tax : ℤ := 40
def tax_per_pound : ℤ := 1
def penny_payment : ℤ := 240

def total_cost_per_pound : ℤ := bulk_price + tax_per_pound

def min_pounds_for_min_spend : ℤ := min_spend_before_tax / bulk_price

def total_pounds_penny_bought : ℤ := penny_payment / total_cost_per_pound

def pounds_exceeded : ℤ := total_pounds_penny_bought - min_pounds_for_min_spend

theorem penny_exceeded_by_32 : pounds_exceeded = 32 := by
  sorry

end penny_exceeded_by_32_l265_265003


namespace solve_x_squared_plus_y_squared_l265_265300

-- Variables
variables {x y : ℝ}

-- Conditions
def cond1 : (x + y)^2 = 36 := sorry
def cond2 : x * y = 8 := sorry

-- Theorem stating the problem's equivalent proof
theorem solve_x_squared_plus_y_squared : x^2 + y^2 = 20 := sorry

end solve_x_squared_plus_y_squared_l265_265300


namespace common_difference_of_arithmetic_sequence_l265_265567

variable (a1 d : ℤ)
def S : ℕ → ℤ
| 0     => 0
| (n+1) => S n + (a1 + n * d)

theorem common_difference_of_arithmetic_sequence
  (h : 2 * S a1 d 3 = 3 * S a1 d 2 + 6) :
  d = 2 :=
  sorry

end common_difference_of_arithmetic_sequence_l265_265567


namespace largest_three_digit_geometric_sequence_with_8_l265_265055

theorem largest_three_digit_geometric_sequence_with_8 :
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ n = 842 ∧ (∃ (a b c : ℕ), n = 100*a + 10*b + c ∧ a = 8 ∧ (a * c = b^2) ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) ) :=
by
  sorry

end largest_three_digit_geometric_sequence_with_8_l265_265055


namespace number_of_candidates_l265_265890

-- Definitions for the given conditions
def total_marks : ℝ := 2000
def average_marks : ℝ := 40

-- Theorem to prove the number of candidates
theorem number_of_candidates : total_marks / average_marks = 50 := by
  sorry

end number_of_candidates_l265_265890


namespace return_kittens_due_to_rehoming_problems_l265_265597

def num_breeding_rabbits : Nat := 10
def kittens_first_spring : Nat := num_breeding_rabbits * num_breeding_rabbits
def kittens_adopted_first_spring : Nat := kittens_first_spring / 2
def kittens_second_spring : Nat := 60
def kittens_adopted_second_spring : Nat := 4
def total_rabbits : Nat := 121

def non_breeding_rabbits_from_first_spring : Nat :=
  total_rabbits - num_breeding_rabbits - kittens_second_spring

def kittens_returned_to_lola : Prop :=
  non_breeding_rabbits_from_first_spring - kittens_adopted_first_spring = 1

theorem return_kittens_due_to_rehoming_problems : kittens_returned_to_lola :=
sorry

end return_kittens_due_to_rehoming_problems_l265_265597


namespace product_of_random_numbers_greater_zero_l265_265934

noncomputable def random_product_positive_probability : ℝ := 
  let interval_length := 45
  let neg_interval_length := 30
  let pos_interval_length := 15
  let prob_neg := (neg_interval_length : ℝ) / interval_length
  let prob_pos := (pos_interval_length : ℝ) / interval_length
  prob_pos * prob_pos + prob_neg * prob_neg

-- Prove that the probability that the product of two randomly selected numbers
-- from the interval [-30, 15] is greater than zero is 5/9.
theorem product_of_random_numbers_greater_zero : 
  random_product_positive_probability = 5 / 9 := by
  sorry

end product_of_random_numbers_greater_zero_l265_265934


namespace calculate_expression_l265_265803

theorem calculate_expression : 103^3 - 3 * 103^2 + 3 * 103 - 1 = 1061208 := by
  sorry

end calculate_expression_l265_265803


namespace neg_prop_p_l265_265835

def prop_p (x : ℝ) : Prop := x ≥ 0 → Real.log (x^2 + 1) ≥ 0

theorem neg_prop_p : (¬ (∀ x ≥ 0, Real.log (x^2 + 1) ≥ 0)) ↔ (∃ x ≥ 0, Real.log (x^2 + 1) < 0) := by
  sorry

end neg_prop_p_l265_265835


namespace probability_two_green_in_four_l265_265236

def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def bag_marbles := 12
def green_marbles := 5
def blue_marbles := 3
def yellow_marbles := 4
def total_picked := 4
def green_picked := 2
def remaining_marbles := bag_marbles - green_marbles
def non_green_picked := total_picked - green_picked

theorem probability_two_green_in_four : 
  (choose green_marbles green_picked * choose remaining_marbles non_green_picked : ℚ) / (choose bag_marbles total_picked) = 14 / 33 := by
  sorry

end probability_two_green_in_four_l265_265236


namespace greatest_two_digit_multiple_of_17_l265_265203

theorem greatest_two_digit_multiple_of_17 : ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (17 ∣ n) ∧ n = 85 :=
by {
    use 85,
    split,
    { exact ⟨le_of_eq rfl, lt_of_le_of_ne (nat.le_of_eq rfl) (ne.symm (dec_trivial))⟩ },
    split,
    { exact dvd.intro 5 rfl },
    { refl }
}

end greatest_two_digit_multiple_of_17_l265_265203


namespace andrew_stickers_now_l265_265598

-- Defining the conditions
def total_stickers : Nat := 1500
def ratio_susan : Nat := 1
def ratio_andrew : Nat := 1
def ratio_sam : Nat := 3
def total_ratio : Nat := ratio_susan + ratio_andrew + ratio_sam
def part : Nat := total_stickers / total_ratio
def susan_share : Nat := ratio_susan * part
def andrew_share_initial : Nat := ratio_andrew * part
def sam_share : Nat := ratio_sam * part
def sam_to_andrew : Nat := (2 * sam_share) / 3

-- Andrew's final stickers count
def andrew_share_final : Nat :=
  andrew_share_initial + sam_to_andrew

-- The theorem to prove
theorem andrew_stickers_now : andrew_share_final = 900 :=
by
  -- Proof would go here
  sorry

end andrew_stickers_now_l265_265598


namespace new_boxes_of_markers_l265_265389

theorem new_boxes_of_markers (initial_markers new_markers_per_box total_markers : ℕ) 
    (h_initial : initial_markers = 32) 
    (h_new_markers_per_box : new_markers_per_box = 9)
    (h_total : total_markers = 86) :
  (total_markers - initial_markers) / new_markers_per_box = 6 := 
by
  sorry

end new_boxes_of_markers_l265_265389


namespace alice_cranes_ratio_alice_cranes_l265_265456

theorem alice_cranes {A : ℕ} (h1 : A + (1/5 : ℝ) * (1000 - A) + 400 = 1000) :
  A = 500 := by
  sorry

theorem ratio_alice_cranes :
  (500 : ℝ) / 1000 = 1 / 2 := by
  sorry

end alice_cranes_ratio_alice_cranes_l265_265456


namespace greatest_two_digit_multiple_of_17_l265_265197

noncomputable def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem greatest_two_digit_multiple_of_17 : 
    ∃ n : ℕ, is_two_digit n ∧ (∃ k : ℕ, n = 17 * k) ∧
    ∀ m : ℕ, is_two_digit m → (∃ k : ℕ, m = 17 * k) → m ≤ n := 
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l265_265197


namespace max_airlines_l265_265716

-- Definitions for the conditions
-- There are 200 cities
def num_cities : ℕ := 200

-- Calculate the total number of city pairs
def num_city_pairs (n : ℕ) : ℕ := (n * (n - 1)) / 2

def total_city_pairs : ℕ := num_city_pairs num_cities

-- Minimum spanning tree concept
def min_flights_per_airline (n : ℕ) : ℕ := n - 1

def total_flights_required : ℕ := num_cities * min_flights_per_airline num_cities

-- Claim: Maximum number of airlines
theorem max_airlines (n : ℕ) (h : n = 200) : ∃ m : ℕ, m = (total_city_pairs / (min_flights_per_airline n)) ∧ m = 100 :=
by sorry

end max_airlines_l265_265716


namespace mixture_correct_l265_265370

def water_amount : ℚ := (3/5) * 20
def vinegar_amount : ℚ := (5/6) * 18
def mixture_amount : ℚ := water_amount + vinegar_amount

theorem mixture_correct : mixture_amount = 27 := 
by
  -- Here goes the proof steps
  sorry

end mixture_correct_l265_265370


namespace modulus_of_power_of_complex_l265_265688

theorem modulus_of_power_of_complex (z : ℂ) (n : ℕ) : 
  |(2 + 1*I)^8| = 625 :=
by
  sorry

end modulus_of_power_of_complex_l265_265688


namespace arithmetic_sequence_common_difference_l265_265540

variable {a₁ d : ℕ}
variable S : ℕ → ℕ

-- Definitions of the sums S₂ and S₃ in an arithmetic sequence
def S₂ : ℕ := a₁ + (a₁ + d)
def S₃ : ℕ := a₁ + (a₁ + d) + (a₁ + 2 * d)

theorem arithmetic_sequence_common_difference (h : 2 * S₃ = 3 * S₂ + 6) : d = 2 :=
by
  -- Skip the proof.
  sorry

end arithmetic_sequence_common_difference_l265_265540


namespace intersection_complement_l265_265695

def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | x ≥ 2}

theorem intersection_complement :
  A ∩ (compl B) = {x | 0 < x ∧ x < 2} := by
  sorry

end intersection_complement_l265_265695


namespace total_pixels_correct_l265_265795

-- Define the monitor's dimensions and pixel density as given conditions
def width_inches : ℕ := 21
def height_inches : ℕ := 12
def pixels_per_inch : ℕ := 100

-- Define the width and height in pixels based on the given conditions
def width_pixels : ℕ := width_inches * pixels_per_inch
def height_pixels : ℕ := height_inches * pixels_per_inch

-- State the objective: proving the total number of pixels on the monitor
theorem total_pixels_correct : width_pixels * height_pixels = 2520000 := by
  sorry

end total_pixels_correct_l265_265795


namespace unique_integer_solution_l265_265595

theorem unique_integer_solution (a b : ℤ) : 
  ∀ x₁ x₂ : ℤ, (x₁ - a) * (x₁ - b) * (x₁ - 3) + 1 = 0 ∧ (x₂ - a) * (x₂ - b) * (x₂ - 3) + 1 = 0 → x₁ = x₂ :=
by
  sorry

end unique_integer_solution_l265_265595


namespace daily_rate_first_week_l265_265161

-- Definitions from given conditions
variable (x : ℝ) (h1 : ∀ y : ℝ, 0 ≤ y)
def cost_first_week := 7 * x
def additional_days_cost := 16 * 14
def total_cost := cost_first_week + additional_days_cost

-- Theorem to solve the problem
theorem daily_rate_first_week (h : total_cost = 350) : x = 18 :=
sorry

end daily_rate_first_week_l265_265161


namespace polygon_sides_l265_265960

theorem polygon_sides (n : ℕ) : 
  (∃ D, D = 104) ∧ (D = (n - 1) * (n - 4) / 2)  → n = 17 :=
by
  sorry

end polygon_sides_l265_265960


namespace angle_A_is_correct_l265_265713

-- Define the given conditions and the main theorem.
theorem angle_A_is_correct (A : ℝ) (m n : ℝ × ℝ) 
  (h_m : m = (Real.sin (A / 2), Real.cos (A / 2)))
  (h_n : n = (Real.cos (A / 2), -Real.cos (A / 2)))
  (h_eq : 2 * ((Prod.fst m * Prod.fst n) + (Prod.snd m * Prod.snd n)) + (Real.sqrt ((Prod.fst m)^2 + (Prod.snd m)^2)) = Real.sqrt 2 / 2) 
  : A = 5 * Real.pi / 12 := by
  sorry

end angle_A_is_correct_l265_265713


namespace sampling_method_is_stratified_l265_265075

/-- There are 500 boys and 400 girls in the high school senior year.
The total population consists of 900 students.
A random sample of 25 boys and 20 girls was taken.
Prove that the sampling method used is stratified sampling method. -/
theorem sampling_method_is_stratified :
    let boys := 500
    let girls := 400
    let total_students := 900
    let sample_boys := 25
    let sample_girls := 20
    let sampling_method := "Stratified sampling"
    sample_boys < boys ∧ sample_girls < girls → sampling_method = "Stratified sampling"
:=
sorry

end sampling_method_is_stratified_l265_265075


namespace triangle_isosceles_l265_265526

-- Definitions involved: Triangle, Circumcircle, Angle Bisector, Isosceles Triangle
universe u

structure Triangle (α : Type u) :=
  (A B C : α)

structure Circumcircle (α : Type u) :=
  (triangle : Triangle α)

structure AngleBisector (α : Type u) :=
  (A : α)
  (triangle : Triangle α)

def IsoscelesTriangle {α : Type u} (P Q R : α) : Prop :=
  ∃ (p₁ p₂ p₃ : α), (p₁ = P ∧ p₂ = Q ∧ p₃ = R) ∧
                  ((∃ θ₁ θ₂, θ₁ + θ₂ = 90) → (∃ θ₃ θ₂, θ₃ + θ₂ = 90))

theorem triangle_isosceles {α : Type u} (T : Triangle α) (S : α)
  (h1 : Circumcircle α) (h2 : AngleBisector α) :
  IsoscelesTriangle T.B T.C S :=
by
  sorry

end triangle_isosceles_l265_265526


namespace kho_kho_only_l265_265950

theorem kho_kho_only (K H B total : ℕ) (h1 : K + B = 10) (h2 : B = 5) (h3 : K + H + B = 25) : H = 15 :=
by {
  sorry
}

end kho_kho_only_l265_265950


namespace problem_statement_l265_265293

-- Define the conditions and the goal
theorem problem_statement {x y : ℝ} 
  (h1 : (x + y)^2 = 36)
  (h2 : x * y = 8) : 
  x^2 + y^2 = 20 := 
by
  sorry

end problem_statement_l265_265293


namespace playground_area_l265_265164

open Real

theorem playground_area (l w : ℝ) (h1 : 2*l + 2*w = 100) (h2 : l = 2*w) : l * w = 5000 / 9 :=
by
  sorry

end playground_area_l265_265164


namespace Niko_total_profit_l265_265031

-- Definitions based on conditions
def cost_per_pair : ℕ := 2
def total_pairs : ℕ := 9
def profit_margin_4_pairs : ℚ := 0.25
def profit_per_other_pair : ℚ := 0.2
def pairs_with_margin : ℕ := 4
def pairs_with_fixed_profit : ℕ := 5

-- Calculations based on definitions
def total_cost : ℚ := total_pairs * cost_per_pair
def profit_on_margin_pairs : ℚ := pairs_with_margin * (profit_margin_4_pairs * cost_per_pair)
def profit_on_fixed_profit_pairs : ℚ := pairs_with_fixed_profit * profit_per_other_pair
def total_profit : ℚ := profit_on_margin_pairs + profit_on_fixed_profit_pairs

-- Statement to prove
theorem Niko_total_profit : total_profit = 3 := by
  sorry

end Niko_total_profit_l265_265031


namespace age_problem_lean4_l265_265515

/-
Conditions:
1. Mr. Bernard's age in eight years will be 60.
2. Luke's age in eight years will be 28.
3. Sarah's age in eight years will be 48.
4. The sum of their ages in eight years will be 136.

Question (translated to proof problem):
Prove that 10 years less than the average age of all three of them is approximately 35.33.

The Lean 4 statement below formalizes this:
-/

theorem age_problem_lean4 :
  let bernard_age := 60
  let luke_age := 28
  let sarah_age := 48
  let total_age := bernard_age + luke_age + sarah_age
  total_age = 136 → ((total_age / 3.0) - 10.0 = 35.33) :=
by
  intros
  sorry

end age_problem_lean4_l265_265515


namespace annie_job_time_l265_265257

noncomputable def annie_time : ℝ :=
  let dan_time := 15
  let dan_rate := 1 / dan_time
  let dan_hours := 6
  let fraction_done_by_dan := dan_rate * dan_hours
  let fraction_left_for_annie := 1 - fraction_done_by_dan
  let annie_work_remaining := fraction_left_for_annie
  let annie_hours := 6
  let annie_rate := annie_work_remaining / annie_hours
  let annie_time := 1 / annie_rate 
  annie_time

theorem annie_job_time :
  annie_time = 3.6 := 
sorry

end annie_job_time_l265_265257


namespace correct_calculation_l265_265628

variable (a : ℝ)

theorem correct_calculation : (2 * a ^ 3) ^ 3 = 8 * a ^ 9 :=
by sorry

end correct_calculation_l265_265628


namespace good_students_l265_265334

theorem good_students (E B : ℕ) (h1 : E + B = 25) (h2 : 12 < B) (h3 : B = 3 * (E - 1)) :
  E = 5 ∨ E = 7 :=
by 
  sorry

end good_students_l265_265334


namespace projections_on_hypotenuse_l265_265860

variables {a b c p q : ℝ}
variables {ρa ρb : ℝ}

-- Given conditions
variable (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
variable (h2 : a < b)
variable (h3 : p = a * a / c)
variable (h4 : q = b * b / c)
variable (h5 : ρa = (a * (b + c - a)) / (a + b + c))
variable (h6 : ρb = (b * (a + c - b)) / (a + b + c))

-- Proof goal
theorem projections_on_hypotenuse 
  (h_right_triangle: a^2 + b^2 = c^2) : p < ρa ∧ q > ρb :=
by
  sorry

end projections_on_hypotenuse_l265_265860


namespace polynomial_terms_equal_l265_265410

theorem polynomial_terms_equal (p q : ℝ) (hp : 0 < p) (hq : 0 < q) (h : p + q = 1) :
  (9 * p^8 * q = 36 * p^7 * q^2) → p = 4 / 5 :=
by
  sorry

end polynomial_terms_equal_l265_265410


namespace find_x_l265_265374

structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point :=
  ⟨Q.x - P.x, Q.y - P.y⟩

def parallel (v w : Point) : Prop :=
  v.x * w.y = v.y * w.x

theorem find_x (A B C : Point) (hA : A = ⟨0, -3⟩) (hB : B = ⟨3, 3⟩) (hC : C = ⟨x, -1⟩) (h_parallel : parallel (vector A B) (vector A C)) : x = 1 := 
by
  sorry

end find_x_l265_265374


namespace mushroom_pickers_at_least_50_l265_265043

-- Given conditions
variables (a : Fin 7 → ℕ) -- Each picker collects a different number of mushrooms.
variables (distinct : ∀ i j, i ≠ j → a i ≠ a j)
variable (total_mushrooms : (Finset.univ.sum a) = 100)

-- The proof that at least three of the pickers collected at least 50 mushrooms together
theorem mushroom_pickers_at_least_50 (a : Fin 7 → ℕ) (distinct : ∀ i j, i ≠ j → a i ≠ a j)
    (total_mushrooms : (Finset.univ.sum a) = 100) :
    ∃ i j k : Fin 7, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ (a i + a j + a k) ≥ 50 :=
sorry

end mushroom_pickers_at_least_50_l265_265043


namespace intersecting_lines_sum_l265_265407

theorem intersecting_lines_sum (a b : ℝ) (h1 : 2 = (1/3) * 4 + a) (h2 : 4 = (1/3) * 2 + b) : a + b = 4 :=
sorry

end intersecting_lines_sum_l265_265407


namespace vehicle_value_last_year_l265_265168

theorem vehicle_value_last_year (value_this_year : ℝ) (ratio : ℝ) (value_this_year_cond : value_this_year = 16000) (ratio_cond : ratio = 0.8) :
  ∃ (value_last_year : ℝ), value_this_year = ratio * value_last_year ∧ value_last_year = 20000 :=
by
  use 20000
  sorry

end vehicle_value_last_year_l265_265168


namespace max_value_of_f_period_of_f_not_monotonically_increasing_incorrect_zeros_l265_265503

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x

theorem max_value_of_f : ∃ x, (f x) = 1/2 :=
sorry

theorem period_of_f : ∀ x, f (x + π) = f x :=
sorry

theorem not_monotonically_increasing : ¬ ∀ x y, 0 < x ∧ x < y ∧ y < π/2 → f x < f y :=
sorry

theorem incorrect_zeros : ∃ x y z, (0 ≤ x ∧ x < y ∧ y < z ∧ z ≤ π) ∧ (f x = 0 ∧ f y = 0 ∧ f z = 0) :=
sorry

end max_value_of_f_period_of_f_not_monotonically_increasing_incorrect_zeros_l265_265503


namespace martha_initial_blocks_l265_265151

theorem martha_initial_blocks (final_blocks : ℕ) (found_blocks : ℕ) (initial_blocks : ℕ) : 
  final_blocks = initial_blocks + found_blocks → 
  final_blocks = 84 →
  found_blocks = 80 → 
  initial_blocks = 4 :=
by
  intros h1 h2 h3
  sorry

end martha_initial_blocks_l265_265151


namespace n_squared_plus_inverse_squared_plus_four_eq_102_l265_265849

theorem n_squared_plus_inverse_squared_plus_four_eq_102 (n : ℝ) (h : n + 1 / n = 10) :
    n^2 + 1 / n^2 + 4 = 102 :=
by sorry

end n_squared_plus_inverse_squared_plus_four_eq_102_l265_265849


namespace number_of_possible_heights_is_680_l265_265431

noncomputable def total_possible_heights : Nat :=
  let base_height := 200 * 3
  let max_additional_height := 200 * (20 - 3)
  let min_height := base_height
  let max_height := base_height + max_additional_height
  let number_of_possible_heights := (max_height - min_height) / 5 + 1
  number_of_possible_heights

theorem number_of_possible_heights_is_680 : total_possible_heights = 680 := by
  sorry

end number_of_possible_heights_is_680_l265_265431


namespace problem1_problem2_l265_265281

noncomputable def f (x : Real) : Real := 
  let a := (2 * Real.cos x, Real.sqrt 3 * Real.sin (2 * x))
  let b := (Real.cos x, 1)
  a.1 * b.1 + a.2 * b.2

theorem problem1 (x : Real) : 
  ∃ k : Int, - Real.pi / 3 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 6 + k * Real.pi :=
  sorry

theorem problem2 (A B C a b c : Real)
  (h1 : a = Real.sqrt 7)
  (h2 : Real.sin B = 2 * Real.sin C)
  (h3 : f A = 2)
  : (∃ area : Real, area = (7 * Real.sqrt 3) / 6) :=
  sorry

end problem1_problem2_l265_265281


namespace find_T_b_plus_T_neg_b_l265_265102

noncomputable def T (r : ℝ) : ℝ := 15 / (1 - r)

theorem find_T_b_plus_T_neg_b (b : ℝ) (h1 : -1 < b) (h2 : b < 1) (h3 : T b * T (-b) = 3600) :
  T b + T (-b) = 480 :=
sorry

end find_T_b_plus_T_neg_b_l265_265102


namespace turtle_feeding_cost_l265_265156

def cost_to_feed_turtles (turtle_weight: ℝ) (food_per_half_pound: ℝ) (jar_capacity: ℝ) (jar_cost: ℝ) : ℝ :=
  let total_food := turtle_weight * (food_per_half_pound / 0.5)
  let total_jars := total_food / jar_capacity
  total_jars * jar_cost

theorem turtle_feeding_cost :
  cost_to_feed_turtles 30 1 15 2 = 8 :=
by
  sorry

end turtle_feeding_cost_l265_265156


namespace circle_radius_l265_265412

theorem circle_radius (x y : ℝ) : (x^2 - 4 * x + y^2 - 21 = 0) → (∃ r : ℝ, r = 5) :=
by
  sorry

end circle_radius_l265_265412


namespace value_of_x_l265_265437

theorem value_of_x (x y z w : ℕ) (h1 : x = y + 7) (h2 : y = z + 12) (h3 : z = w + 25) (h4 : w = 90) : x = 134 :=
by
  sorry

end value_of_x_l265_265437


namespace trapezoid_area_calculation_l265_265623

noncomputable def trapezoid_area : ℝ :=
  let y1 := 20
  let y2 := 10
  let x1 := y1 / 2
  let x2 := y2 / 2
  let base1 := x1
  let base2 := x2
  let height := y1 - y2
  (base1 + base2) * height / 2

theorem trapezoid_area_calculation :
  let y1 := 20
  let y2 := 10
  let x1 := y1 / 2
  let x2 := y2 / 2
  let base1 := x1
  let base2 := x2
  let height := y1 - y2
  (base1 + base2) * height / 2 = 75 := 
by
  -- Validation of the translation to Lean 4. Proof steps are omitted.
  sorry

end trapezoid_area_calculation_l265_265623


namespace inequality_product_geq_two_power_n_equality_condition_l265_265137

open Real BigOperators

noncomputable def is_solution (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i ∧ a i = 1

theorem inequality_product_geq_two_power_n (a : ℕ → ℝ) (n : ℕ)
  (h1 : ( ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i))
  (h2 : ∑ i in Finset.range n, a (i + 1) = n) :
  (∏ i in Finset.range n, (1 + 1 / a (i + 1))) ≥ 2 ^ n :=
sorry

theorem equality_condition (a : ℕ → ℝ) (n : ℕ)
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i)
  (h2 : ∑ i in Finset.range n, a (i + 1) = n):
  (∏ i in Finset.range n, (1 + 1 / a (i + 1))) = 2 ^ n ↔ is_solution a n :=
sorry

end inequality_product_geq_two_power_n_equality_condition_l265_265137


namespace stratified_sampling_correct_l265_265072

variables (total_employees senior_employees mid_level_employees junior_employees sample_size : ℕ)
          (sampling_ratio : ℚ)
          (senior_sample mid_sample junior_sample : ℕ)

-- Conditions
def company_conditions := 
  total_employees = 450 ∧ 
  senior_employees = 45 ∧ 
  mid_level_employees = 135 ∧ 
  junior_employees = 270 ∧ 
  sample_size = 30 ∧ 
  sampling_ratio = 1 / 15

-- Proof goal
theorem stratified_sampling_correct : 
  company_conditions total_employees senior_employees mid_level_employees junior_employees sample_size sampling_ratio →
  senior_sample = senior_employees * sampling_ratio ∧ 
  mid_sample = mid_level_employees * sampling_ratio ∧ 
  junior_sample = junior_employees * sampling_ratio ∧
  senior_sample + mid_sample + junior_sample = sample_size :=
by sorry

end stratified_sampling_correct_l265_265072


namespace large_block_dimension_ratio_l265_265645

theorem large_block_dimension_ratio
  (V_normal V_large : ℝ) 
  (k : ℝ)
  (h1 : V_normal = 4)
  (h2 : V_large = 32) 
  (h3 : V_large = k^3 * V_normal) :
  k = 2 := by
  sorry

end large_block_dimension_ratio_l265_265645


namespace cistern_fill_time_l265_265920

-- Define the problem conditions
def pipe_p_fill_time : ℕ := 10
def pipe_q_fill_time : ℕ := 15
def joint_filling_time : ℕ := 2
def remaining_fill_time : ℕ := 10 -- This is the answer we need to prove

-- Prove that the remaining fill time is equal to 10 minutes
theorem cistern_fill_time :
  (joint_filling_time * (1 / pipe_p_fill_time + 1 / pipe_q_fill_time) + (remaining_fill_time / pipe_q_fill_time)) = 1 :=
sorry

end cistern_fill_time_l265_265920


namespace correct_answer_l265_265941

variables (A B : polynomial ℝ) (a : ℝ)

theorem correct_answer (hB : B = 3 * a^2 - 5 * a - 7) (hMistake : A - 2 * B = -2 * a^2 + 3 * a + 6) :
  A + 2 * B = 10 * a^2 - 17 * a - 22 :=
by
  sorry

end correct_answer_l265_265941


namespace alan_tickets_l265_265909

theorem alan_tickets (a m : ℕ) (h1 : a + m = 150) (h2 : m = 5 * a - 6) : a = 26 :=
by
  sorry

end alan_tickets_l265_265909


namespace greatest_two_digit_multiple_of_17_l265_265181

theorem greatest_two_digit_multiple_of_17 : ∃ (n : ℕ), n < 100 ∧ 10 ≤ n ∧ 17 ∣ n ∧ ∀ m : ℕ, m < 100 → 10 ≤ m → 17 ∣ m → m ≤ n :=
begin
  use 85,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m hm1 hm2 hm3,
    by_contradiction hmn,
    have h : m > 85 := not_le.mp hmn,
    linear_combination h * 17,
    sorry,
  },
end

end greatest_two_digit_multiple_of_17_l265_265181


namespace good_students_l265_265358

theorem good_students (G T : ℕ) (h1 : G + T = 25) (h2 : T > 12) (h3 : T = 3 * (G - 1)) :
  G = 5 ∨ G = 7 :=
by
  sorry

end good_students_l265_265358


namespace find_a2023_l265_265831

noncomputable def a_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  (∀ n, 2 * S n = a n * (a n + 1))

theorem find_a2023 (a S : ℕ → ℕ) 
  (h_seq : a_sequence a S)
  (h_pos : ∀ n, 0 < a n) 
  : a 2023 = 2023 :=
sorry

end find_a2023_l265_265831


namespace comparison_of_a_b_c_l265_265133

theorem comparison_of_a_b_c : 
  let a := (1/3)^(2/5)
  let b := 2^(4/3)
  let c := Real.logb 2 (1/3)
  c < a ∧ a < b :=
by
  sorry

end comparison_of_a_b_c_l265_265133


namespace largest_even_of_sum_140_l265_265051

theorem largest_even_of_sum_140 :
  ∃ (n : ℕ), 2 * n + 2 * (n + 1) + 2 * (n + 2) + 2 * (n + 3) = 140 ∧ 2 * (n + 3) = 38 :=
by
  sorry

end largest_even_of_sum_140_l265_265051


namespace each_girl_brought_2_cups_l265_265426

-- Definitions of the conditions
def total_students : ℕ := 30
def boys : ℕ := 10
def total_cups : ℕ := 90
def cups_per_boy : ℕ := 5
def girls : ℕ := total_students - boys

def total_cups_by_boys : ℕ := boys * cups_per_boy
def total_cups_by_girls : ℕ := total_cups - total_cups_by_boys
def cups_per_girl : ℕ := total_cups_by_girls / girls

-- The statement with the correct answer
theorem each_girl_brought_2_cups (
  h1 : total_students = 30,
  h2 : boys = 10,
  h3 : total_cups = 90,
  h4 : cups_per_boy = 5,
  h5 : total_cups_by_boys = boys * cups_per_boy,
  h6 : total_cups_by_girls = total_cups - total_cups_by_boys,
  h7 : cups_per_girl = total_cups_by_girls / girls
) : cups_per_girl = 2 := 
sorry

end each_girl_brought_2_cups_l265_265426


namespace find_common_difference_l265_265581

-- Define the arithmetic sequence and the sum of the first n terms
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
def sum_of_first_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ := ∑ k in finset.range n, arithmetic_sequence a₁ d (k + 1)

-- Given condition
def condition (a₁ d : ℝ) : Prop := 
  2 * sum_of_first_n_terms a₁ d 3 = 3 * sum_of_first_n_terms a₁ d 2 + 6

-- The proof statement
theorem find_common_difference (a₁ d : ℝ) (h : condition a₁ d) : d = 2 :=
by
  sorry

end find_common_difference_l265_265581


namespace arithmetic_sequence_common_difference_l265_265538

variable {a₁ d : ℕ}
variable S : ℕ → ℕ

-- Definitions of the sums S₂ and S₃ in an arithmetic sequence
def S₂ : ℕ := a₁ + (a₁ + d)
def S₃ : ℕ := a₁ + (a₁ + d) + (a₁ + 2 * d)

theorem arithmetic_sequence_common_difference (h : 2 * S₃ = 3 * S₂ + 6) : d = 2 :=
by
  -- Skip the proof.
  sorry

end arithmetic_sequence_common_difference_l265_265538


namespace upper_limit_b_l265_265709

theorem upper_limit_b (a b : ℤ) (h1 : 6 < a) (h2 : a < 17) (h3 : 3 < b) (h4 : (a : ℚ) / b ≤ 3.75) : b ≤ 4 := by
  sorry

end upper_limit_b_l265_265709


namespace arithmetic_sequence_common_difference_l265_265545

theorem arithmetic_sequence_common_difference 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : S 3 = a 1 + a 2 + a 3)
  (h2 : S 2 = a 1 + a 2)
  (h3 : a 2 = a 1 + d)
  (h4 : a 3 = a 1 + 2 * d)
  (h5 : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l265_265545


namespace minimum_k_coloring_l265_265490

theorem minimum_k_coloring (f : ℕ → ℕ) (color : ℕ → Fin 3) :
  (∀ n m : ℕ, color n = color m → f (n + m) = f n + f m) →
  (∃ n m : ℕ, f (n + m) ≠ f n + f m) →
  ∃! k : ℕ, k = 3 :=
by
  sorry

end minimum_k_coloring_l265_265490


namespace log_inequality_l265_265269

theorem log_inequality {a x : ℝ} (h1 : 0 < x) (h2 : x < 1) (h3 : a > 0) (h4 : a ≠ 1) : 
  abs (Real.logb a (1 - x)) > abs (Real.logb a (1 + x)) :=
sorry

end log_inequality_l265_265269


namespace factorial_sum_mod_30_l265_265094

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def sum_of_factorials (n : Nat) : Nat :=
  (List.range (n + 1)).map factorial |>.sum

def remainder_when_divided_by (m k : Nat) : Nat :=
  m % k

theorem factorial_sum_mod_30 : remainder_when_divided_by (sum_of_factorials 100) 30 = 3 :=
by
  sorry

end factorial_sum_mod_30_l265_265094


namespace division_problem_l265_265760

theorem division_problem (D d q r : ℕ) 
  (h1 : D + d + q + r = 205)
  (h2 : q = d) :
  D = 174 ∧ d = 13 :=
by {
  sorry
}

end division_problem_l265_265760


namespace tyler_meal_choices_l265_265937

-- Define the total number of different meals Tyler can choose given the conditions.
theorem tyler_meal_choices : 
    (3 * (Nat.choose 5 3) * 4 * 4 = 480) := 
by
    -- Using the built-in combination function and the fact that meat, dessert, and drink choices are directly multiplied.
    sorry

end tyler_meal_choices_l265_265937


namespace Emily_age_is_23_l265_265766

variable (UncleBob Daniel Emily Zoe : ℕ)

-- Conditions
axiom h1 : UncleBob = 54
axiom h2 : Daniel = UncleBob / 2
axiom h3 : Emily = Daniel - 4
axiom h4 : Emily = 2 * Zoe / 3

-- Question: Prove that Emily's age is 23
theorem Emily_age_is_23 : Emily = 23 :=
by
  sorry

end Emily_age_is_23_l265_265766


namespace undefined_count_expression_l265_265493

theorem undefined_count_expression : 
  let expr (x : ℝ) := (x^2 - 16) / ((x^2 + 3*x - 10) * (x - 4))
  ∃ u v w : ℝ, (u = 2 ∨ v = -5 ∨ w = 4) ∧
  (u ≠ v ∧ u ≠ w ∧ v ≠ w) :=
by
  sorry

end undefined_count_expression_l265_265493


namespace recurring_fraction_difference_l265_265976

theorem recurring_fraction_difference :
  let x := (36 / 99 : ℚ)
  let y := (36 / 100 : ℚ)
  x - y = (1 / 275 : ℚ) :=
by
  sorry

end recurring_fraction_difference_l265_265976


namespace sandy_correct_value_t_l265_265397

theorem sandy_correct_value_t (p q r s : ℕ) (t : ℕ) 
  (hp : p = 2) (hq : q = 4) (hr : r = 6) (hs : s = 8)
  (expr1 : p + q - r + s - t = p + (q - (r + (s - t)))) :
  t = 8 := 
by
  sorry

end sandy_correct_value_t_l265_265397


namespace selling_price_equivalence_l265_265050

noncomputable def cost_price_25_profit : ℝ := 1750 / 1.25
def selling_price_profit := 1520
def selling_price_loss := 1280

theorem selling_price_equivalence
  (cp : ℝ)
  (h1 : cp = cost_price_25_profit)
  (h2 : cp = 1400) :
  (selling_price_profit - cp = cp - selling_price_loss) → (selling_price_loss = 1280) := 
  by
  unfold cost_price_25_profit at h1
  simp [h1] at h2
  sorry

end selling_price_equivalence_l265_265050


namespace rain_probability_l265_265411

/-
Theorem: Given that the probability it will rain on Monday is 40%
and the probability it will rain on Tuesday is 30%, and the probability of
rain on a given day is independent of the weather on any other day,
the probability it will rain on both Monday and Tuesday is 12%.
-/
theorem rain_probability (p_monday : ℝ) (p_tuesday : ℝ) (independent : Prop) :
  p_monday = 0.4 ∧ p_tuesday = 0.3 ∧ independent → (p_monday * p_tuesday) * 100 = 12 :=
by sorry

end rain_probability_l265_265411


namespace circle_intersection_l265_265103

theorem circle_intersection (m : ℝ) :
  (x^2 + y^2 - 2*m*x + m^2 - 4 = 0 ∧ x^2 + y^2 + 2*x - 4*m*y + 4*m^2 - 8 = 0) →
  (-12/5 < m ∧ m < -2/5) ∨ (0 < m ∧ m < 2) :=
by sorry

end circle_intersection_l265_265103


namespace triangle_side_relation_l265_265703

theorem triangle_side_relation (a b c : ℝ) (h1 : a^2 - 16 * b^2 - c^2 + 6 * a * b + 10 * b * c = 0) (h2 : a + b > c) :
  a + c = 2 * b := 
sorry

end triangle_side_relation_l265_265703


namespace chess_tournament_games_l265_265640

def num_games (n : Nat) : Nat := n * (n - 1) * 2

theorem chess_tournament_games : num_games 7 = 84 :=
by
  sorry

end chess_tournament_games_l265_265640


namespace common_difference_of_arithmetic_sequence_l265_265569

variable (a1 d : ℤ)
def S : ℕ → ℤ
| 0     => 0
| (n+1) => S n + (a1 + n * d)

theorem common_difference_of_arithmetic_sequence
  (h : 2 * S a1 d 3 = 3 * S a1 d 2 + 6) :
  d = 2 :=
  sorry

end common_difference_of_arithmetic_sequence_l265_265569


namespace jamie_catches_bus_probability_l265_265237

noncomputable def probability_jamie_catches_bus : ℝ :=
  let total_area := 120 * 120
  let overlap_area := 20 * 100
  overlap_area / total_area

theorem jamie_catches_bus_probability :
  probability_jamie_catches_bus = (5 / 36) :=
by
  sorry

end jamie_catches_bus_probability_l265_265237


namespace cary_net_calorie_deficit_is_250_l265_265987

-- Define the conditions
def miles_walked : ℕ := 3
def candy_bar_calories : ℕ := 200
def calories_per_mile : ℕ := 150

-- Define the function to calculate total calories burned
def total_calories_burned (miles : ℕ) (calories_per_mile : ℕ) : ℕ :=
  miles * calories_per_mile

-- Define the function to calculate net calorie deficit
def net_calorie_deficit (total_calories : ℕ) (candy_calories : ℕ) : ℕ :=
  total_calories - candy_calories

-- The statement to be proven
theorem cary_net_calorie_deficit_is_250 :
  net_calorie_deficit (total_calories_burned miles_walked calories_per_mile) candy_bar_calories = 250 :=
  by sorry

end cary_net_calorie_deficit_is_250_l265_265987


namespace rupert_jumps_more_l265_265040

theorem rupert_jumps_more (Ronald_jumps Rupert_jumps total_jumps : ℕ)
  (h1 : Ronald_jumps = 157)
  (h2 : total_jumps = 243)
  (h3 : Rupert_jumps + Ronald_jumps = total_jumps) :
  Rupert_jumps - Ronald_jumps = 86 :=
by
  sorry

end rupert_jumps_more_l265_265040


namespace solve_x_squared_plus_y_squared_l265_265299

-- Variables
variables {x y : ℝ}

-- Conditions
def cond1 : (x + y)^2 = 36 := sorry
def cond2 : x * y = 8 := sorry

-- Theorem stating the problem's equivalent proof
theorem solve_x_squared_plus_y_squared : x^2 + y^2 = 20 := sorry

end solve_x_squared_plus_y_squared_l265_265299


namespace penny_exceeded_by_32_l265_265004

def bulk_price : ℤ := 5
def min_spend_before_tax : ℤ := 40
def tax_per_pound : ℤ := 1
def penny_payment : ℤ := 240

def total_cost_per_pound : ℤ := bulk_price + tax_per_pound

def min_pounds_for_min_spend : ℤ := min_spend_before_tax / bulk_price

def total_pounds_penny_bought : ℤ := penny_payment / total_cost_per_pound

def pounds_exceeded : ℤ := total_pounds_penny_bought - min_pounds_for_min_spend

theorem penny_exceeded_by_32 : pounds_exceeded = 32 := by
  sorry

end penny_exceeded_by_32_l265_265004


namespace find_savings_l265_265163

noncomputable def savings (income expenditure : ℕ) : ℕ :=
  income - expenditure

theorem find_savings (I E : ℕ) (h_ratio : I = 9 * E) (h_income : I = 18000) : savings I E = 2000 :=
by
  sorry

end find_savings_l265_265163


namespace radio_loss_percentage_l265_265227

theorem radio_loss_percentage (cost_price selling_price : ℕ) (h1 : cost_price = 1500) (h2 : selling_price = 1305) : 
  (cost_price - selling_price) * 100 / cost_price = 13 := by
  sorry

end radio_loss_percentage_l265_265227


namespace product_positive_probability_l265_265929

theorem product_positive_probability :
  let interval := set.Icc (-30 : ℝ) 15 in
  let prob_neg := (30 : ℝ) / 45 in
  let prob_pos := (15 : ℝ) / 45 in
  let prob_product_neg := 2 * (prob_neg * prob_pos) in
  let prob_product_pos := (prob_neg ^ 2) + (prob_pos ^ 2) in
  (prob_product_pos = 5 / 9) :=
by
  sorry

end product_positive_probability_l265_265929


namespace log_condition_iff_l265_265277

variables (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) (h1 : a ≠ 1) (h2 : b ≠ 1) (h3 : x ≠ 1)

theorem log_condition_iff (h : 4 * (Real.logb a x)^2 + 3 * (Real.logb b x)^2 = 8 * (Real.logb a x) * (Real.logb b x)) :
  a = b ^ 2 :=
sorry

end log_condition_iff_l265_265277


namespace initial_games_l265_265324

theorem initial_games (X : ℕ) (h1 : X - 68 + 47 = 74) : X = 95 :=
by
  sorry

end initial_games_l265_265324


namespace good_students_count_l265_265365

theorem good_students_count (E B : ℕ) 
    (h1 : E + B = 25) 
    (h2 : ∀ (e ∈ finset.range 5), B > 12) 
    (h3 : ∀ (e ∈ finset.range 20), B = 3 * (E - 1)) :
    E = 5 ∨ E = 7 :=
by
  sorry

end good_students_count_l265_265365


namespace molecular_weight_correct_l265_265056

-- Define the atomic weights
def atomic_weight_Cu : ℝ := 63.546
def atomic_weight_C : ℝ := 12.011
def atomic_weight_O : ℝ := 15.999

-- Define the number of atoms in the compound
def num_atoms_Cu : ℕ := 1
def num_atoms_C : ℕ := 1
def num_atoms_O : ℕ := 3

-- Define the molecular weight calculation
def molecular_weight : ℝ :=
  num_atoms_Cu * atomic_weight_Cu + 
  num_atoms_C * atomic_weight_C + 
  num_atoms_O * atomic_weight_O

-- Prove the molecular weight of the compound
theorem molecular_weight_correct : molecular_weight = 123.554 :=
by
  sorry

end molecular_weight_correct_l265_265056


namespace fifth_equation_l265_265737

theorem fifth_equation
: 1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2 :=
by
  sorry

end fifth_equation_l265_265737


namespace greatest_two_digit_multiple_of_17_l265_265214

theorem greatest_two_digit_multiple_of_17 : ∃ m : ℕ, (10 ≤ m) ∧ (m ≤ 99) ∧ (17 ∣ m) ∧ (∀ n : ℕ, (10 ≤ n) ∧ (n ≤ 99) ∧ (17 ∣ n) → n ≤ m) ∧ m = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l265_265214


namespace max_value_of_xy_expression_l265_265596

theorem max_value_of_xy_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 4 * x + 3 * y < 60) : 
  xy * (60 - 4 * x - 3 * y) ≤ 2000 / 3 := 
sorry

end max_value_of_xy_expression_l265_265596


namespace bus_initial_count_l265_265970

theorem bus_initial_count (x : ℕ) (got_off : ℕ) (remained : ℕ) (h1 : got_off = 47) (h2 : remained = 43) (h3 : x - got_off = remained) : x = 90 :=
by
  rw [h1, h2] at h3
  sorry

end bus_initial_count_l265_265970


namespace f_periodic_zeros_in_interval_zeros_count_and_sum_l265_265874

-- Define the function and the conditions here
variable {f : ℝ → ℝ} 

axiom f_symmetry1 : ∀ x : ℝ, f (3 + x) = f (3 - x)
axiom f_symmetry2 : ∀ x : ℝ, f (8 + x) = f (8 - x)
axiom f_values : f 1 = 0 ∧ f 5 = 0 ∧ f 7 = 0

-- Problem (1): Prove the function is periodic with period 10
theorem f_periodic : ∃ T : ℝ, T = 10 ∧ ∀ x : ℝ, f (x + T) = f x := 
by sorry

-- Problem (2): Find all zeros of f(x) on [-10, 0]
theorem zeros_in_interval :
  (f (-1) = 0) ∧ (f (-3) = 0) ∧ (f (-5) = 0) ∧ (f (-9) = 0) :=
by sorry

-- Problem (3): Determine the number and sum of zeros in the interval [-2012, 2012]
theorem zeros_count_and_sum :
  (num_zeros : ℕ), (num_zeros = 1610) ∧ (sum_zeros : ℝ), (sum_zeros = 0) := 
by sorry

end f_periodic_zeros_in_interval_zeros_count_and_sum_l265_265874


namespace parallel_line_eq_perpendicular_line_eq_l265_265649

-- Define the conditions: A line passing through (1, -4) and the given line equation 2x + 3y + 5 = 0
def passes_through (x y : ℝ) (a b c : ℝ) : Prop := a * x + b * y + c = 0

-- Define the theorem statements for parallel and perpendicular lines
theorem parallel_line_eq (m : ℝ) :
  passes_through 1 (-4) 2 3 m → m = 10 := 
sorry

theorem perpendicular_line_eq (n : ℝ) :
  passes_through 1 (-4) 3 (-2) (-n) → n = 11 :=
sorry

end parallel_line_eq_perpendicular_line_eq_l265_265649


namespace games_bought_at_garage_sale_l265_265731

theorem games_bought_at_garage_sale (G : ℕ)
  (h1 : 2 + G - 2  = 2) :
  G = 2 :=
by {
  sorry
}

end games_bought_at_garage_sale_l265_265731


namespace inequality_and_equality_condition_l265_265727

theorem inequality_and_equality_condition (x : ℝ) (hx : x > 0) : 
  x + 1 / x ≥ 2 ∧ (x + 1 / x = 2 ↔ x = 1) :=
  sorry

end inequality_and_equality_condition_l265_265727


namespace greatest_two_digit_multiple_of_17_l265_265171

theorem greatest_two_digit_multiple_of_17 : ∀ n, n ∈ finset.range 100 → n % 17 = 0 → n ≤ 85 →
  (∀ m, m ∈ finset.range 100 → m % 17 = 0 → m > n → m ≥ 100) →
  n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l265_265171


namespace proof_problem_l265_265707

theorem proof_problem (f g g_inv : ℝ → ℝ) (hinv : ∀ x, f (x ^ 4 - 1) = g x)
  (hginv : ∀ y, g (g_inv y) = y) (h : ∀ y, f (g_inv y) = g (g_inv y)) :
  g_inv (f 15) = 2 :=
by
  sorry

end proof_problem_l265_265707


namespace percentage_difference_l265_265123

variable (x y z : ℝ)

theorem percentage_difference (h1 : y = 1.75 * x) (h2 : z = 0.60 * y) :
  (1 - x / z) * 100 = 4.76 :=
by
  sorry

end percentage_difference_l265_265123


namespace perimeter_of_isosceles_triangle_l265_265867

theorem perimeter_of_isosceles_triangle (a b : ℕ) (h_isosceles : (a = 3 ∧ b = 4) ∨ (a = 4 ∧ b = 3)) :
  ∃ p : ℕ, p = 10 ∨ p = 11 :=
by
  sorry

end perimeter_of_isosceles_triangle_l265_265867


namespace diagonal_intersection_probability_decagon_l265_265233

noncomputable def probability_diagonal_intersection_in_decagon : ℚ :=
  let vertices := 10
  let total_diagonals := (vertices * (vertices - 3)) / 2
  let total_pairs_of_diagonals := total_diagonals * (total_diagonals - 1) / 2
  let total_intersecting_pairs := (vertices * (vertices - 1) * (vertices - 2) * (vertices - 3)) / 24
  total_intersecting_pairs / total_pairs_of_diagonals

theorem diagonal_intersection_probability_decagon (h : probability_diagonal_intersection_in_decagon = 42 / 119) : 
  probability_diagonal_intersection_in_decagon = 42 / 119 :=
sorry

end diagonal_intersection_probability_decagon_l265_265233


namespace min_value_of_parabola_l265_265627

theorem min_value_of_parabola : ∃ x : ℝ, ∀ y : ℝ, y = 3 * x^2 - 18 * x + 244 → y = 217 := by
  sorry

end min_value_of_parabola_l265_265627


namespace area_of_smaller_part_l265_265792

noncomputable def average (a b : ℝ) : ℝ :=
  (a + b) / 2

theorem area_of_smaller_part:
  ∃ A B : ℝ, A + B = 900 ∧ (B - A) = (1 / 5) * average A B ∧ A = 405 :=
by
  sorry

end area_of_smaller_part_l265_265792


namespace pizza_slices_l265_265877

theorem pizza_slices (P T S : ℕ) (h1 : P = 2) (h2 : T = 16) : S = 8 :=
by
  -- to be filled in
  sorry

end pizza_slices_l265_265877


namespace arithmetic_common_difference_l265_265586

-- Defining the arithmetic sequence sum function S
def S (a₁ a₂ a₃ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if n = 3 then a₁ + a₂ + a₃
  else if n = 2 then a₁ + a₂
  else 0

-- The Lean statement for the problem
theorem arithmetic_common_difference (a₁ a₂ a₃ d : ℝ) 
  (h1 : a₂ = a₁ + d) 
  (h2 : a₃ = a₂ + d)
  (h3 : 2 * S a₁ a₂ a₃ d 3 = 3 * S a₁ a₂ a₃ d 2 + 6) :
  d = 2 :=
by
  sorry

end arithmetic_common_difference_l265_265586


namespace greatest_two_digit_multiple_of_17_l265_265201

theorem greatest_two_digit_multiple_of_17 : ∃ N, N = 85 ∧ 
  (∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n → n ≤ 85) :=
by 
  sorry

end greatest_two_digit_multiple_of_17_l265_265201


namespace decagon_diagonals_intersection_probability_l265_265235

def isRegularDecagon : Prop :=
  ∃ decagon : ℕ, decagon = 10  -- A regular decagon has 10 sides

def chosen_diagonals (n : ℕ) : ℕ :=
  (Nat.choose n 2) - n   -- Number of diagonals in an n-sided polygon =
                          -- number of pairs of vertices - n sides

noncomputable def probability_intersection : ℚ :=
  let total_diagonals := chosen_diagonals 10
  let number_of_ways_to_pick_four := Nat.choose 10 4
  (number_of_ways_to_pick_four * 2) / (total_diagonals * (total_diagonals - 1) / 2)

theorem decagon_diagonals_intersection_probability :
  isRegularDecagon → probability_intersection = 42 / 119 :=
sorry

end decagon_diagonals_intersection_probability_l265_265235


namespace geometric_sequence_a7_eq_64_l265_265140

open Nat

theorem geometric_sequence_a7_eq_64 (a : ℕ → ℕ) (h1 : a 1 = 1) (hrec : ∀ n : ℕ, a (n + 1) = 2 * a n) : a 7 = 64 := by
  sorry

end geometric_sequence_a7_eq_64_l265_265140


namespace arithmetic_sequence_c_d_sum_l265_265413

theorem arithmetic_sequence_c_d_sum (c d : ℕ) 
  (h1 : 10 - 3 = 7) 
  (h2 : 17 - 10 = 7) 
  (h3 : c = 17 + 7) 
  (h4 : d = c + 7) 
  (h5 : d + 7 = 38) :
  c + d = 55 := 
by
  sorry

end arithmetic_sequence_c_d_sum_l265_265413


namespace good_students_options_l265_265346

variables (E B : ℕ)

-- Define the condition that the class has 25 students
def total_students : Prop := E + B = 25

-- Define the condition given by the first group of students
def first_group_condition : Prop := B > 12

-- Define the condition given by the second group of students
def second_group_condition : Prop := B = 3 * (E - 1)

-- Define the problem statement
theorem good_students_options (E B : ℕ) :
  total_students E B → first_group_condition B → second_group_condition E B → (E = 5 ∨ E = 7) :=
by
  intros h1 h2 h3
  sorry

end good_students_options_l265_265346


namespace arithmetic_progression_infinite_squares_l265_265457

theorem arithmetic_progression_infinite_squares (a d : ℤ) (k : ℤ) (hk : a = k^2) :
  ∃ infinite_set_of_squares : set ℤ, (∀ n ∈ infinite_set_of_squares, ∃ m : ℤ, m^2 = a + n * d) ∧ infinite infinite_set_of_squares :=
  sorry

end arithmetic_progression_infinite_squares_l265_265457


namespace each_girl_brought_2_cups_l265_265429

-- Here we define all the given conditions
def total_students : ℕ := 30
def num_boys : ℕ := 10
def cups_per_boy : ℕ := 5
def total_cups : ℕ := 90

-- Define the conditions as Lean definitions
def num_girls : ℕ := total_students - num_boys -- From condition 3
def cups_by_boys : ℕ := num_boys * cups_per_boy
def cups_by_girls : ℕ := total_cups - cups_by_boys

-- Define the final question and expected answer
def cups_per_girl : ℕ := cups_by_girls / num_girls

-- Final problem statement to prove
theorem each_girl_brought_2_cups :
  cups_per_girl = 2 :=
by
  have h1 : num_girls = 20 := by sorry
  have h2 : cups_by_boys = 50 := by sorry
  have h3 : cups_by_girls = 40 := by sorry
  have h4 : cups_per_girl = 2 := by sorry
  exact h4

end each_girl_brought_2_cups_l265_265429


namespace common_difference_of_arithmetic_sequence_l265_265528

variable {a₁ d : ℕ}
def S (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem common_difference_of_arithmetic_sequence (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l265_265528


namespace prime_pairs_satisfying_conditions_l265_265824

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def satisfies_conditions (p q : ℕ) : Prop :=
  (7 * p + 1) % q = 0 ∧ (7 * q + 1) % p = 0

theorem prime_pairs_satisfying_conditions :
  { (p, q) | is_prime p ∧ is_prime q ∧ satisfies_conditions p q } = {(2, 3), (2, 5), (3, 11)} := 
sorry

end prime_pairs_satisfying_conditions_l265_265824


namespace probability_product_positive_is_5_div_9_l265_265925

noncomputable def probability_positive_product : ℚ :=
  let interval := Set.Icc (-30 : ℝ) 15
  let length_interval := 45
  let length_neg := 30
  let length_pos := 15
  let prob_neg := (length_neg : ℚ) / length_interval
  let prob_pos := (length_pos : ℚ) / length_interval
  let prob_product_pos := prob_neg^2 + prob_pos^2
  prob_product_pos

theorem probability_product_positive_is_5_div_9 :
  probability_positive_product = 5 / 9 :=
by
  sorry

end probability_product_positive_is_5_div_9_l265_265925


namespace josh_money_left_l265_265008

def initial_amount : ℝ := 9
def spent_on_drink : ℝ := 1.75
def spent_on_item : ℝ := 1.25

theorem josh_money_left : initial_amount - (spent_on_drink + spent_on_item) = 6 := by
  sorry

end josh_money_left_l265_265008


namespace probability_of_mathematics_letter_l265_265317

-- Definitions for the problem
def english_alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}

def mathematics_letters : Finset Char := {'M', 'A', 'T', 'H', 'E', 'I', 'C', 'S'}

-- Set the total number of letters in the English alphabet
def total_letters := english_alphabet.card

-- Set the number of unique letters in 'MATHEMATICS'
def mathematics_unique_letters := mathematics_letters.card

-- Statement of the Lean theorem
theorem probability_of_mathematics_letter : (mathematics_unique_letters : ℚ) / total_letters = 4 / 13 :=
by
  sorry

end probability_of_mathematics_letter_l265_265317


namespace part1_part2_l265_265322

theorem part1 (A B : ℝ) (c : ℝ) (cos_A : ℝ) (tan_half_B_add_cot_half_B: ℝ) 
  (h1: cos_A = 5 / 13) 
  (h2: tan_half_B_add_cot_half_B = 10 / 3) 
  (pos_c: c = 21) :
  ∃ (cos_AB: ℝ), cos_AB = 56 / 65 :=
by {
  sorry
}

theorem part2 (A B : ℝ) (c : ℝ) (cos_A : ℝ) (tan_half_B_add_cot_half_B: ℝ) 
  (h1: cos_A = 5 / 13) 
  (h2: tan_half_B_add_cot_half_B = 10 / 3) 
  (pos_c: c = 21) :
  ∃ (area: ℝ), area = 126 :=
by {
  sorry
}

end part1_part2_l265_265322


namespace cyclists_meet_at_starting_point_l265_265944

/--
Given a circular track of length 1200 meters, and three cyclists with speeds of 36 kmph, 54 kmph, and 72 kmph,
prove that all three cyclists will meet at the starting point for the first time after 4 minutes.
-/
theorem cyclists_meet_at_starting_point :
  let track_length := 1200
  let speed_a_kmph := 36
  let speed_b_kmph := 54
  let speed_c_kmph := 72
  
  let speed_a_m_per_min := speed_a_kmph * 1000 / 60
  let speed_b_m_per_min := speed_b_kmph * 1000 / 60
  let speed_c_m_per_min := speed_c_kmph * 1000 / 60
  
  let time_a := track_length / speed_a_m_per_min
  let time_b := track_length / speed_b_m_per_min
  let time_c := track_length / speed_c_m_per_min
  
  let lcm := (2 : ℚ)

  (time_a = 2) ∧ (time_b = 4 / 3) ∧ (time_c = 1) → 
  ∀ t, t = lcm * 3 → t = 12 / 3 → t = 4 :=
by
  sorry

end cyclists_meet_at_starting_point_l265_265944


namespace recurring_decimal_exceeds_by_fraction_l265_265974

theorem recurring_decimal_exceeds_by_fraction : 
  let y := (36 : ℚ) / 99
  let x := (36 : ℚ) / 100
  ((4 : ℚ) / 11) - x = (4 : ℚ) / 1100 :=
by
  sorry

end recurring_decimal_exceeds_by_fraction_l265_265974


namespace american_literature_marks_l265_265144

variable (History HomeEconomics PhysicalEducation Art AverageMarks NumberOfSubjects TotalMarks KnownMarks : ℕ)
variable (A : ℕ)

axiom marks_history : History = 75
axiom marks_home_economics : HomeEconomics = 52
axiom marks_physical_education : PhysicalEducation = 68
axiom marks_art : Art = 89
axiom average_marks : AverageMarks = 70
axiom number_of_subjects : NumberOfSubjects = 5

def total_marks (AverageMarks NumberOfSubjects : ℕ) : ℕ := AverageMarks * NumberOfSubjects

def known_marks (History HomeEconomics PhysicalEducation Art : ℕ) : ℕ := History + HomeEconomics + PhysicalEducation + Art

axiom total_marks_eq : TotalMarks = total_marks AverageMarks NumberOfSubjects
axiom known_marks_eq : KnownMarks = known_marks History HomeEconomics PhysicalEducation Art

theorem american_literature_marks :
  A = TotalMarks - KnownMarks := by
  sorry

end american_literature_marks_l265_265144


namespace complement_union_correct_l265_265504

open Set

theorem complement_union_correct :
  let P : Set ℕ := { x | x * (x - 3) ≥ 0 }
  let Q : Set ℕ := {2, 4}
  (compl P) ∪ Q = {1, 2, 4} :=
by
  let P : Set ℕ := { x | x * (x - 3) ≥ 0 }
  let Q : Set ℕ := {2, 4}
  have h : (compl P) ∪ Q = {1, 2, 4} := sorry
  exact h

end complement_union_correct_l265_265504


namespace greatest_two_digit_multiple_of_17_l265_265193

theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n < 100 ∧ n ≥ 10 ∧ 17 ∣ n ∧ ∀ m, m < 100 ∧ m ≥ 10 ∧ 17 ∣ m → m ≤ 85 :=
by
  use 85
  -- Prove conditions follow sorry
  sorry

end greatest_two_digit_multiple_of_17_l265_265193


namespace solve_x_squared_plus_y_squared_l265_265297

-- Variables
variables {x y : ℝ}

-- Conditions
def cond1 : (x + y)^2 = 36 := sorry
def cond2 : x * y = 8 := sorry

-- Theorem stating the problem's equivalent proof
theorem solve_x_squared_plus_y_squared : x^2 + y^2 = 20 := sorry

end solve_x_squared_plus_y_squared_l265_265297


namespace arithmetic_sequence_c_d_sum_l265_265415

theorem arithmetic_sequence_c_d_sum (c d : ℕ) 
  (h1 : 10 - 3 = 7) 
  (h2 : 17 - 10 = 7) 
  (h3 : c = 17 + 7) 
  (h4 : d = c + 7) 
  (h5 : d + 7 = 38) :
  c + d = 55 := 
by
  sorry

end arithmetic_sequence_c_d_sum_l265_265415


namespace abs_sum_lt_abs_l265_265110

theorem abs_sum_lt_abs (a b : ℝ) (h : a * b < 0) : |a + b| < |a| + |b| :=
sorry

end abs_sum_lt_abs_l265_265110


namespace work_problem_l265_265942

/--
Given:
1. A and B together can finish the work in 16 days.
2. B alone can finish the work in 48 days.
To Prove:
A alone can finish the work in 24 days.
-/
theorem work_problem (a b : ℕ)
  (h1 : a + b = 16)
  (h2 : b = 48) :
  a = 24 := 
sorry

end work_problem_l265_265942


namespace carlos_blocks_l265_265468

theorem carlos_blocks (initial_blocks : ℕ) (blocks_given : ℕ) (remaining_blocks : ℕ) 
  (h1 : initial_blocks = 58) (h2 : blocks_given = 21) : remaining_blocks = 37 :=
by
  sorry

end carlos_blocks_l265_265468


namespace find_P_7_l265_265138

-- This definition represents the polynomial P as stated in the problem
def P (x : ℝ) : ℝ :=
  (3 * x^4 - 30 * x^3 + a * x^2 + b * x + c) *
  (4 * x^4 - 84 * x^3 + d * x^2 + e * x + f)

-- The given conditions and question is to find P(7)
theorem find_P_7 (a b c d e f : ℝ)
  (h_roots : (polynomial.map polynomial.algebra_map (3 * polynomial.X^4 - 30 * polynomial.X^3 + a * polynomial.X^2 + b * polynomial.X + c) *
              polynomial.map polynomial.algebra_map (4 * polynomial.X^4 - 84 * polynomial.X^3 + d * polynomial.X^2 + e * polynomial.X + f)).roots = {2, 3, 4, 5, 5}) :
  P 7 = 86400 := 
sorry

end find_P_7_l265_265138


namespace median_of_consecutive_integers_l265_265778

theorem median_of_consecutive_integers (a n : ℤ) (N : ℕ) (h1 : (a + (n - 1)) + (a + (N - n)) = 110) : 
  (2 * a + N - 1) / 2 = 55 := 
by {
  -- The proof goes here.
  sorry
}

end median_of_consecutive_integers_l265_265778


namespace place_two_after_three_digit_number_l265_265120

theorem place_two_after_three_digit_number (h t u : ℕ) 
  (Hh : h < 10) (Ht : t < 10) (Hu : u < 10) : 
  (100 * h + 10 * t + u) * 10 + 2 = 1000 * h + 100 * t + 10 * u + 2 := 
by
  sorry

end place_two_after_three_digit_number_l265_265120


namespace find_min_y_l265_265873

theorem find_min_y (x y : ℕ) (hx : x = y + 8) 
    (h : Nat.gcd ((x^3 + y^3) / (x + y)) (x * y) = 16) : 
    y = 4 :=
sorry

end find_min_y_l265_265873


namespace product_of_random_numbers_greater_zero_l265_265935

noncomputable def random_product_positive_probability : ℝ := 
  let interval_length := 45
  let neg_interval_length := 30
  let pos_interval_length := 15
  let prob_neg := (neg_interval_length : ℝ) / interval_length
  let prob_pos := (pos_interval_length : ℝ) / interval_length
  prob_pos * prob_pos + prob_neg * prob_neg

-- Prove that the probability that the product of two randomly selected numbers
-- from the interval [-30, 15] is greater than zero is 5/9.
theorem product_of_random_numbers_greater_zero : 
  random_product_positive_probability = 5 / 9 := by
  sorry

end product_of_random_numbers_greater_zero_l265_265935


namespace combined_weight_l265_265243

variables (G D C : ℝ)

def grandmother_weight (G D C : ℝ) := G + D + C = 150
def daughter_weight (D : ℝ) := D = 42
def child_weight (G C : ℝ) := C = 1/5 * G

theorem combined_weight (G D C : ℝ) (h1 : grandmother_weight G D C) (h2 : daughter_weight D) (h3 : child_weight G C) : D + C = 60 :=
by
  sorry

end combined_weight_l265_265243


namespace greatest_two_digit_multiple_of_17_l265_265176

theorem greatest_two_digit_multiple_of_17 : ∃ n, (n < 100) ∧ (n % 17 = 0) ∧ (∀ m, (m < 100) ∧ (m % 17 = 0) → m ≤ n) := by
  let multiples := [17, 34, 51, 68, 85]
  have ht : ∀ m, m ∈ multiples → (m < 100) ∧ (m % 17 = 0) := by
    intros m hm
    cases hm with
    | inl h => exact ⟨by norm_num, by norm_num⟩
    | inr hm' =>
      cases hm' with
      | inl h => exact ⟨by norm_num, by norm_num⟩
      | inr hm'' =>
        cases hm'' with
        | inl h => exact ⟨by norm_num, by norm_num⟩
        | inr hm''' =>
          cases hm''' with
          | inl h => exact ⟨by norm_num, by norm_num⟩
          | inr hm'''' =>
            cases hm'''' with
            | inl h => exact ⟨by norm_num, by norm_num⟩
            | inr hm => cases hm
  
  use 85
  split
  { norm_num },
  split
  { exact (by norm_num : 85 % 17 = 0) },
  intro m
  intro h
  exact list.minimum_mem multiples h

end greatest_two_digit_multiple_of_17_l265_265176


namespace good_students_count_l265_265362

theorem good_students_count (E B : ℕ) 
    (h1 : E + B = 25) 
    (h2 : ∀ (e ∈ finset.range 5), B > 12) 
    (h3 : ∀ (e ∈ finset.range 20), B = 3 * (E - 1)) :
    E = 5 ∨ E = 7 :=
by
  sorry

end good_students_count_l265_265362


namespace natural_pairs_l265_265685

theorem natural_pairs (x y : ℕ) : 2^(2 * x + 1) + 2^x + 1 = y^2 ↔ (x = 0 ∧ y = 2) ∨ (x = 4 ∧ y = 23) :=
by sorry

end natural_pairs_l265_265685


namespace value_of_expression_l265_265314

theorem value_of_expression (m : ℝ) (h : m^2 - m - 110 = 0) : (m - 1)^2 + m = 111 := by
  sorry

end value_of_expression_l265_265314


namespace find_number_of_good_students_l265_265331

variable {G T : ℕ}

def is_good_or_troublemaker (G T : ℕ) : Prop :=
  G + T = 25 ∧ 
  (∀ s ∈ finset.range 5, T > 12) ∧ 
  (∀ s ∈ finset.range 20, T = 3 * (G - 1))

-- There are 25 students in total. 
-- Each student is either a good student or a troublemaker.
-- Good students always tell the truth, while troublemakers always lie.
-- Five students make the statement: "If I transfer to another class, more than half of the remaining students will be troublemakers."
-- The remaining 20 students make the statement: "If I transfer to another class, the number of troublemakers among the remaining students will be three times the number of good students."

theorem find_number_of_good_students (G T : ℕ) (h : is_good_or_troublemaker G T) :
  (G = 5 ∨ G = 7) :=
sorry

end find_number_of_good_students_l265_265331


namespace sufficient_but_not_necessary_condition_l265_265016

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x^2 - 1 = 0 → x^3 - x = 0) ∧ ¬ (x^3 - x = 0 → x^2 - 1 = 0) := by
  sorry

end sufficient_but_not_necessary_condition_l265_265016


namespace quiz_score_of_dropped_student_l265_265065

theorem quiz_score_of_dropped_student (avg16 : ℝ) (avg15 : ℝ) (num_students : ℝ) (dropped_students : ℝ) (x : ℝ)
  (h1 : avg16 = 60.5) (h2 : avg15 = 64) (h3 : num_students = 16) (h4 : dropped_students = 1) :
  x = 60.5 * 16 - 64 * 15 :=
by
  sorry

end quiz_score_of_dropped_student_l265_265065


namespace cost_to_feed_turtles_l265_265155

theorem cost_to_feed_turtles :
  (∀ (weight : ℚ), weight > 0 → (food_per_half_pound_ounces = 1) →
    (total_weight_pounds = 30) →
    (jar_contents_ounces = 15 ∧ jar_cost = 2) →
    (total_cost_dollars = 8)) :=
by
  sorry

variables
  (food_per_half_pound_ounces : ℚ := 1)
  (total_weight_pounds : ℚ := 30)
  (jar_contents_ounces : ℚ := 15)
  (jar_cost : ℚ := 2)
  (total_cost_dollars : ℚ := 8)

-- Assuming all variables needed to state the theorem exist and are meaningful
/-!
  Given:
  - Each turtle needs 1 ounce of food per 1/2 pound of body weight
  - Total turtles' weight is 30 pounds
  - Each jar of food contains 15 ounces and costs $2

  Prove:
  - The total cost to feed Sylvie's turtles is $8
-/

end cost_to_feed_turtles_l265_265155


namespace cake_cost_is_20_l265_265602

-- Define the given conditions
def total_budget : ℕ := 50
def additional_needed : ℕ := 11
def bouquet_cost : ℕ := 36
def balloons_cost : ℕ := 5

-- Define the derived conditions
def total_cost : ℕ := total_budget + additional_needed
def combined_bouquet_balloons_cost : ℕ := bouquet_cost + balloons_cost
def cake_cost : ℕ := total_cost - combined_bouquet_balloons_cost

-- The theorem to be proved
theorem cake_cost_is_20 : cake_cost = 20 :=
by
  -- proof steps are not required
  sorry

end cake_cost_is_20_l265_265602


namespace tom_total_calories_l265_265917

-- Define the conditions
def c_weight : ℕ := 1
def c_calories_per_pound : ℕ := 51
def b_weight : ℕ := 2 * c_weight
def b_calories_per_pound : ℕ := c_calories_per_pound / 3

-- Define the total calories
def total_calories : ℕ := (c_weight * c_calories_per_pound) + (b_weight * b_calories_per_pound)

-- Prove the total calories Tom eats
theorem tom_total_calories : total_calories = 85 := by
  sorry

end tom_total_calories_l265_265917


namespace percentage_is_60_l265_265954

-- Definitions based on the conditions
def fraction_value (x : ℕ) : ℕ := x / 3
def percentage_less_value (x p : ℕ) : ℕ := x - (p * x) / 100

-- Lean statement based on the mathematically equivalent proof problem
theorem percentage_is_60 : ∀ (x p : ℕ), x = 180 → fraction_value x = 60 → percentage_less_value 60 p = 24 → p = 60 :=
by
  intros x p H1 H2 H3
  -- Proof is not required, so we use sorry
  sorry

end percentage_is_60_l265_265954


namespace diagonal_intersection_probability_decagon_l265_265232

noncomputable def probability_diagonal_intersection_in_decagon : ℚ :=
  let vertices := 10
  let total_diagonals := (vertices * (vertices - 3)) / 2
  let total_pairs_of_diagonals := total_diagonals * (total_diagonals - 1) / 2
  let total_intersecting_pairs := (vertices * (vertices - 1) * (vertices - 2) * (vertices - 3)) / 24
  total_intersecting_pairs / total_pairs_of_diagonals

theorem diagonal_intersection_probability_decagon (h : probability_diagonal_intersection_in_decagon = 42 / 119) : 
  probability_diagonal_intersection_in_decagon = 42 / 119 :=
sorry

end diagonal_intersection_probability_decagon_l265_265232


namespace good_students_l265_265361

theorem good_students (G T : ℕ) (h1 : G + T = 25) (h2 : T > 12) (h3 : T = 3 * (G - 1)) :
  G = 5 ∨ G = 7 :=
by
  sorry

end good_students_l265_265361


namespace tangents_intersect_on_line_l265_265391

theorem tangents_intersect_on_line (a : ℝ) (x y : ℝ) (hx : 8 * a = 1) (hx_line : x - y = 5) (hx_point : x = 3) (hy_point : y = -2) : 
  x - y = 5 :=
by
  sorry -- Proof to be completed

end tangents_intersect_on_line_l265_265391


namespace positive_integer_solutions_l265_265488

theorem positive_integer_solutions (n m : ℕ) (h : n > 0 ∧ m > 0) : 
  (n + 1) * m = n! + 1 ↔ (n = 1 ∧ m = 1) ∨ (n = 2 ∧ m = 1) ∨ (n = 4 ∧ m = 5) := by
  sorry

end positive_integer_solutions_l265_265488


namespace probability_fourth_roll_six_l265_265990

noncomputable def fair_die_prob : ℚ := 1 / 6
noncomputable def biased_die_prob : ℚ := 3 / 4
noncomputable def biased_die_other_face_prob : ℚ := 1 / 20
noncomputable def prior_prob : ℚ := 1 / 2

def p := 41
def q := 67

theorem probability_fourth_roll_six (p q : ℕ) (h1 : fair_die_prob = 1 / 6) (h2 : biased_die_prob = 3 / 4) (h3 : prior_prob = 1 / 2) :
  p + q = 108 :=
sorry

end probability_fourth_roll_six_l265_265990


namespace remainder_div_357_l265_265738

theorem remainder_div_357 (N : ℤ) (h : N % 17 = 2) : N % 357 = 2 :=
sorry

end remainder_div_357_l265_265738


namespace wheel_distance_l265_265799

noncomputable def diameter : ℝ := 9
noncomputable def revolutions : ℝ := 18.683651804670912
noncomputable def pi_approx : ℝ := 3.14159
noncomputable def circumference (d : ℝ) : ℝ := pi_approx * d
noncomputable def distance (r : ℝ) (c : ℝ) : ℝ := r * c

theorem wheel_distance : distance revolutions (circumference diameter) = 528.219 :=
by
  unfold distance circumference diameter revolutions pi_approx
  -- Here we would perform the calculation and show that the result is approximately 528.219
  sorry

end wheel_distance_l265_265799


namespace cary_net_calorie_deficit_is_250_l265_265988

-- Define the conditions
def miles_walked : ℕ := 3
def candy_bar_calories : ℕ := 200
def calories_per_mile : ℕ := 150

-- Define the function to calculate total calories burned
def total_calories_burned (miles : ℕ) (calories_per_mile : ℕ) : ℕ :=
  miles * calories_per_mile

-- Define the function to calculate net calorie deficit
def net_calorie_deficit (total_calories : ℕ) (candy_calories : ℕ) : ℕ :=
  total_calories - candy_calories

-- The statement to be proven
theorem cary_net_calorie_deficit_is_250 :
  net_calorie_deficit (total_calories_burned miles_walked calories_per_mile) candy_bar_calories = 250 :=
  by sorry

end cary_net_calorie_deficit_is_250_l265_265988


namespace factorization_correct_l265_265478

noncomputable def factorize_poly (m n : ℕ) : ℕ := 2 * m * n ^ 2 - 12 * m * n + 18 * m

theorem factorization_correct (m n : ℕ) :
  factorize_poly m n = 2 * m * (n - 3) ^ 2 :=
by
  sorry

end factorization_correct_l265_265478


namespace good_students_l265_265357

theorem good_students (G T : ℕ) (h1 : G + T = 25) (h2 : T > 12) (h3 : T = 3 * (G - 1)) :
  G = 5 ∨ G = 7 :=
by
  sorry

end good_students_l265_265357


namespace f_at_2_lt_e6_l265_265276

variable (f : ℝ → ℝ)

-- Specify the conditions
axiom derivable_f : Differentiable ℝ f
axiom condition_3f_gt_fpp : ∀ x : ℝ, 3 * f x > (deriv (deriv f)) x
axiom f_at_1 : f 1 = Real.exp 3

-- Conclusion to prove
theorem f_at_2_lt_e6 : f 2 < Real.exp 6 :=
sorry

end f_at_2_lt_e6_l265_265276


namespace solution_set_of_inequality_l265_265166

theorem solution_set_of_inequality (x : ℝ) : (1 / |x - 1| ≥ 1) ↔ (0 ≤ x ∧ x < 1) ∨ (1 < x ∧ x ≤ 2) :=
by
  sorry

end solution_set_of_inequality_l265_265166


namespace lines_intersect_at_l265_265169

def Line1 (t : ℝ) : ℝ × ℝ :=
  let x := 1 + 3 * t
  let y := 2 - t
  (x, y)

def Line2 (u : ℝ) : ℝ × ℝ :=
  let x := -1 + 4 * u
  let y := 4 + 3 * u
  (x, y)

theorem lines_intersect_at :
  ∃ t u : ℝ, Line1 t = Line2 u ∧
             Line1 t = (-53 / 17, 56 / 17) :=
by
  sorry

end lines_intersect_at_l265_265169


namespace maximum_value_l265_265593

def expression (A B C : ℕ) : ℕ := A * B * C + A * B + B * C + C * A

theorem maximum_value (A B C : ℕ) 
  (h1 : A + B + C = 15) : 
  expression A B C ≤ 200 :=
sorry

end maximum_value_l265_265593


namespace frac_pow_zero_l265_265435

def frac := 123456789 / (-987654321 : ℤ)

theorem frac_pow_zero : frac ^ 0 = 1 :=
by sorry

end frac_pow_zero_l265_265435


namespace mary_initial_flour_l265_265026

theorem mary_initial_flour (F_total F_add F_initial : ℕ) 
  (h_total : F_total = 9)
  (h_add : F_add = 6)
  (h_initial : F_initial = F_total - F_add) :
  F_initial = 3 :=
sorry

end mary_initial_flour_l265_265026


namespace min_a_plus_b_l265_265014

variable (a b : ℝ)
variable (ha_pos : a > 0)
variable (hb_pos : b > 0)
variable (h1 : a^2 - 12 * b ≥ 0)
variable (h2 : 9 * b^2 - 4 * a ≥ 0)

theorem min_a_plus_b (a b : ℝ) (ha_pos : a > 0) (hb_pos : b > 0)
  (h1 : a^2 - 12 * b ≥ 0) (h2 : 9 * b^2 - 4 * a ≥ 0) :
  a + b = 3.3442 := 
sorry

end min_a_plus_b_l265_265014


namespace moon_arrangements_l265_265814

theorem moon_arrangements :
  (∃ (MOON : Finset (List Char)), 
    {w : List Char |
      w ∈ MOON ∧ w = ['M', 'O', 'O', 'N']}.card = 12) :=
sorry

end moon_arrangements_l265_265814


namespace max_min_value_x_eq_1_l265_265751

noncomputable def f (x k : ℝ) : ℝ := x^2 - 2 * (2 * k - 1) * x + 3 * k^2 - 2 * k + 6

theorem max_min_value_x_eq_1 :
  ∀ (k : ℝ), (∀ x : ℝ, ∃ m : ℝ, f x k = m → k = 1 → m = 6) → (∃ x : ℝ, x = 1) :=
by
  sorry

end max_min_value_x_eq_1_l265_265751


namespace greatest_two_digit_multiple_of_17_l265_265198

noncomputable def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem greatest_two_digit_multiple_of_17 : 
    ∃ n : ℕ, is_two_digit n ∧ (∃ k : ℕ, n = 17 * k) ∧
    ∀ m : ℕ, is_two_digit m → (∃ k : ℕ, m = 17 * k) → m ≤ n := 
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l265_265198


namespace good_students_l265_265335

theorem good_students (E B : ℕ) (h1 : E + B = 25) (h2 : 12 < B) (h3 : B = 3 * (E - 1)) :
  E = 5 ∨ E = 7 :=
by 
  sorry

end good_students_l265_265335


namespace total_notes_l265_265240

theorem total_notes (total_money : ℕ) (fifty_notes : ℕ) (fifty_value : ℕ) (fivehundred_value : ℕ) (fivehundred_notes : ℕ) :
  total_money = 10350 →
  fifty_notes = 57 →
  fifty_value = 50 →
  fivehundred_value = 500 →
  57 * 50 + fivehundred_notes * 500 = 10350 →
  fifty_notes + fivehundred_notes = 72 :=
by
  intros h_total_money h_fifty_notes h_fifty_value h_fivehundred_value h_equation
  sorry

end total_notes_l265_265240


namespace greatest_two_digit_multiple_of_17_l265_265182

theorem greatest_two_digit_multiple_of_17 : ∃ (n : ℕ), n < 100 ∧ 10 ≤ n ∧ 17 ∣ n ∧ ∀ m : ℕ, m < 100 → 10 ≤ m → 17 ∣ m → m ≤ n :=
begin
  use 85,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m hm1 hm2 hm3,
    by_contradiction hmn,
    have h : m > 85 := not_le.mp hmn,
    linear_combination h * 17,
    sorry,
  },
end

end greatest_two_digit_multiple_of_17_l265_265182


namespace find_d_value_l265_265871

/-- Let d be an odd prime number. If 89 - (d+3)^2 is the square of an integer, then d = 5. -/
theorem find_d_value (d : ℕ) (h₁ : Nat.Prime d) (h₂ : Odd d) (h₃ : ∃ m : ℤ, 89 - (d + 3)^2 = m^2) : d = 5 := 
by
  sorry

end find_d_value_l265_265871


namespace composite_prime_fraction_l265_265736

theorem composite_prime_fraction :
  let P1 : ℕ := 4 * 6 * 8 * 9 * 10 * 12 * 14 * 15
  let P2 : ℕ := 16 * 18 * 20 * 21 * 22 * 24 * 25 * 26
  let first_prime : ℕ := 2
  let second_prime : ℕ := 3
  (P1 + first_prime) / (P2 + second_prime) =
    (4 * 6 * 8 * 9 * 10 * 12 * 14 * 15 + 2) / (16 * 18 * 20 * 21 * 22 * 24 * 25 * 26 + 3) := by
  sorry

end composite_prime_fraction_l265_265736


namespace range_of_g_l265_265098

noncomputable def g (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctan (2 * x)

theorem range_of_g : Set.Icc (-1.1071) 1.1071 = Set.image g (Set.Icc (-1:ℝ) 1) := by
  sorry

end range_of_g_l265_265098


namespace students_who_wore_blue_lipstick_l265_265606

theorem students_who_wore_blue_lipstick (total_students : ℕ) (h1 : total_students = 200) : 
  ∃ blue_lipstick_students : ℕ, blue_lipstick_students = 5 :=
by
  have colored_lipstick_students := total_students / 2
  have red_lipstick_students := colored_lipstick_students / 4
  let blue_lipstick_students := red_lipstick_students / 5
  have h2 : blue_lipstick_students = 5 :=
    calc blue_lipstick_students
          = (total_students / 2) / 4 / 5 : by sorry -- detailed calculation steps omitted
  use blue_lipstick_students
  exact h2

end students_who_wore_blue_lipstick_l265_265606


namespace solution_set_of_inequality_l265_265100

theorem solution_set_of_inequality : { x : ℝ | 0 < x ∧ x < 2 } = { x : ℝ | 0 < x ∧ x < 2 } :=
by
  sorry

end solution_set_of_inequality_l265_265100


namespace probability_product_positive_of_independent_selection_l265_265922

theorem probability_product_positive_of_independent_selection :
  let I := set.Icc (-30 : ℝ) (15 : ℝ)
  let P := (λ (x y : ℝ), x ∈ I ∧ y ∈ I ∧ x * y > 0)
  (Prob { x : ℝ × ℝ | P x.1 x.2 } :
    ProbabilitySpace (I × I)) = 5 / 9 :=
by
  sorry

end probability_product_positive_of_independent_selection_l265_265922


namespace girl_boy_lineup_probability_l265_265793

theorem girl_boy_lineup_probability :
  let total_configurations := Nat.choose 20 9
  let valid_case1 := Nat.choose 14 9
  let valid_subcases := 6 * Nat.choose 13 8
  let valid_configurations := valid_case1 + valid_subcases
  (valid_configurations : ℚ) / total_configurations = 0.058 :=
by
  let total_configurations := Nat.choose 20 9
  let valid_case1 := Nat.choose 14 9
  let valid_subcases := 6 * Nat.choose 13 8
  let valid_configurations := valid_case1 + valid_subcases
  have h : (valid_configurations : ℚ) / total_configurations = 0.058 := sorry
  exact h

end girl_boy_lineup_probability_l265_265793


namespace cat_and_mouse_positions_after_317_moves_l265_265516

-- Define the conditions of the problem
def cat_positions : List String := ["Top Left", "Top Right", "Bottom Right", "Bottom Left"]
def mouse_positions : List String := ["Top Left", "Top Middle", "Top Right", "Right Middle", "Bottom Right", "Bottom Middle", "Bottom Left", "Left Middle"]

-- Calculate the position of the cat after n moves
def cat_position_after_moves (n : Nat) : String :=
  cat_positions.get! (n % 4)

-- Calculate the position of the mouse after n moves
def mouse_position_after_moves (n : Nat) : String :=
  mouse_positions.get! (n % 8)

-- Prove the final positions of the cat and mouse after 317 moves
theorem cat_and_mouse_positions_after_317_moves :
  cat_position_after_moves 317 = "Top Left" ∧ mouse_position_after_moves 317 = "Bottom Middle" :=
by
  sorry

end cat_and_mouse_positions_after_317_moves_l265_265516


namespace basketball_team_girls_l265_265783

theorem basketball_team_girls (B G : ℕ) 
  (h1 : B + G = 30) 
  (h2 : B + (1 / 3) * G = 18) : 
  G = 18 :=
by
  have h3 : G - (1 / 3) * G = 30 - 18 := by sorry
  have h4 : (2 / 3) * G = 12 := by sorry
  have h5 : G = 12 * (3 / 2) := by sorry
  have h6 : G = 18 := by sorry
  exact h6

end basketball_team_girls_l265_265783


namespace soccer_ball_price_l265_265127

theorem soccer_ball_price 
  (B S V : ℕ) 
  (h1 : (B + S + V) / 3 = 36)
  (h2 : B = V + 10)
  (h3 : S = V + 8) : 
  S = 38 := 
by 
  sorry

end soccer_ball_price_l265_265127


namespace inequality_solution_l265_265812

noncomputable def condition (x : ℝ) : Prop :=
  2 * Real.cos x ≤ Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))
  ∧ Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)) ≤ Real.sqrt 2

theorem inequality_solution (x : ℝ) (h₀ : 0 ≤ x ∧ x ≤ 2 * Real.pi) (h₁ : condition x) :
  Real.cos x ≤ Real.sqrt (2:ℝ) / 2 ∧ x ∈ [Real.pi/4, 7 * Real.pi/4] := sorry

end inequality_solution_l265_265812


namespace arithmetic_sequence_common_difference_l265_265557

theorem arithmetic_sequence_common_difference (a_1 d : ℝ) (S : ℕ → ℝ) 
    (h1 : S 2 = 2 * a_1 + d)
    (h2 : S 3 = 3 * a_1 + 3 * d)
    (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 := 
by
  sorry

end arithmetic_sequence_common_difference_l265_265557


namespace total_pixels_l265_265794

theorem total_pixels (width height dpi : ℕ) (h_width : width = 21) (h_height : height = 12) (h_dpi : dpi = 100) :
  width * dpi * height * dpi = 2520000 := 
by
  rw [h_width, h_height, h_dpi]
  simp
  sorry

end total_pixels_l265_265794


namespace annual_interest_approx_l265_265806

noncomputable def P : ℝ := 10000
noncomputable def r : ℝ := 0.05
noncomputable def t : ℝ := 1
noncomputable def e : ℝ := Real.exp 1

theorem annual_interest_approx :
  let A := P * Real.exp (r * t)
  let interest := A - P
  abs (interest - 512.71) < 0.01 := sorry

end annual_interest_approx_l265_265806


namespace gas_consumption_100_l265_265715

noncomputable def gas_consumption (x : ℝ) : Prop :=
  60 * 1 + (x - 60) * 1.5 = 1.2 * x

theorem gas_consumption_100 (x : ℝ) (h : gas_consumption x) : x = 100 := 
by {
  sorry
}

end gas_consumption_100_l265_265715


namespace greatest_two_digit_multiple_of_17_l265_265184

theorem greatest_two_digit_multiple_of_17 : ∃ x : ℕ, x < 100 ∧ x ≥ 10 ∧ x % 17 = 0 ∧ ∀ y : ℕ, y < 100 ∧ y ≥ 10 ∧ y % 17 = 0 → y ≤ x :=
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l265_265184


namespace Jeff_pays_when_picking_up_l265_265129

-- Definition of the conditions
def deposit_rate : ℝ := 0.10
def increase_rate : ℝ := 0.40
def last_year_cost : ℝ := 250
def this_year_cost : ℝ := last_year_cost * (1 + increase_rate)
def deposit : ℝ := this_year_cost * deposit_rate

-- Lean statement of the proof
theorem Jeff_pays_when_picking_up : this_year_cost - deposit = 315 := by
  sorry

end Jeff_pays_when_picking_up_l265_265129


namespace product_of_random_numbers_greater_zero_l265_265933

noncomputable def random_product_positive_probability : ℝ := 
  let interval_length := 45
  let neg_interval_length := 30
  let pos_interval_length := 15
  let prob_neg := (neg_interval_length : ℝ) / interval_length
  let prob_pos := (pos_interval_length : ℝ) / interval_length
  prob_pos * prob_pos + prob_neg * prob_neg

-- Prove that the probability that the product of two randomly selected numbers
-- from the interval [-30, 15] is greater than zero is 5/9.
theorem product_of_random_numbers_greater_zero : 
  random_product_positive_probability = 5 / 9 := by
  sorry

end product_of_random_numbers_greater_zero_l265_265933


namespace common_difference_of_arithmetic_sequence_l265_265568

variable (a1 d : ℤ)
def S : ℕ → ℤ
| 0     => 0
| (n+1) => S n + (a1 + n * d)

theorem common_difference_of_arithmetic_sequence
  (h : 2 * S a1 d 3 = 3 * S a1 d 2 + 6) :
  d = 2 :=
  sorry

end common_difference_of_arithmetic_sequence_l265_265568


namespace find_rate_percent_l265_265222

theorem find_rate_percent (P : ℝ) (r : ℝ) (A1 A2 : ℝ) (t1 t2 : ℕ)
  (h1 : A1 = P * (1 + r)^t1) (h2 : A2 = P * (1 + r)^t2) (hA1 : A1 = 2420) (hA2 : A2 = 3146) (ht1 : t1 = 2) (ht2 : t2 = 3) :
  r = 0.2992 :=
by
  sorry

end find_rate_percent_l265_265222


namespace mr_smith_markers_l265_265386

theorem mr_smith_markers :
  ∀ (initial_markers : ℕ) (total_markers : ℕ) (markers_per_box : ℕ) 
  (number_of_boxes : ℕ),
  initial_markers = 32 → 
  total_markers = 86 → 
  markers_per_box = 9 → 
  number_of_boxes = (total_markers - initial_markers) / markers_per_box →
  number_of_boxes = 6 :=
by
  intros initial_markers total_markers markers_per_box number_of_boxes h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃] at h₄
  simp only [Nat.sub] at h₄
  exact h₄

end mr_smith_markers_l265_265386


namespace mode_of_scores_is_97_l265_265042

open List

def scores : List ℕ := [
  65, 65,
  71, 73, 73, 76,
  80, 80, 84, 84, 88, 88, 88,
  92, 92, 95, 97, 97, 97, 97,
  101, 101, 101, 104, 106,
  110, 110, 110
]

theorem mode_of_scores_is_97 : ∃ m, m = 97 ∧ (∀ x, count x scores ≤ count 97 scores) :=
by
  sorry

end mode_of_scores_is_97_l265_265042


namespace percent_democrats_l265_265714

/-- The percentage of registered voters in the city who are democrats and republicans -/
def D : ℝ := sorry -- Percent of democrats
def R : ℝ := sorry -- Percent of republicans

-- Given conditions
axiom H1 : D + R = 100
axiom H2 : 0.65 * D + 0.20 * R = 47

-- Statement to prove
theorem percent_democrats : D = 60 :=
by
  sorry

end percent_democrats_l265_265714


namespace each_girl_brought_2_cups_l265_265428

-- Here we define all the given conditions
def total_students : ℕ := 30
def num_boys : ℕ := 10
def cups_per_boy : ℕ := 5
def total_cups : ℕ := 90

-- Define the conditions as Lean definitions
def num_girls : ℕ := total_students - num_boys -- From condition 3
def cups_by_boys : ℕ := num_boys * cups_per_boy
def cups_by_girls : ℕ := total_cups - cups_by_boys

-- Define the final question and expected answer
def cups_per_girl : ℕ := cups_by_girls / num_girls

-- Final problem statement to prove
theorem each_girl_brought_2_cups :
  cups_per_girl = 2 :=
by
  have h1 : num_girls = 20 := by sorry
  have h2 : cups_by_boys = 50 := by sorry
  have h3 : cups_by_girls = 40 := by sorry
  have h4 : cups_per_girl = 2 := by sorry
  exact h4

end each_girl_brought_2_cups_l265_265428


namespace rods_in_one_mile_l265_265275

-- Define the conditions as assumptions in Lean

-- 1. 1 mile = 8 furlongs
def mile_to_furlong : ℕ := 8

-- 2. 1 furlong = 220 paces
def furlong_to_pace : ℕ := 220

-- 3. 1 pace = 0.2 rods
def pace_to_rod : ℝ := 0.2

-- Define the statement to be proven
theorem rods_in_one_mile : (mile_to_furlong * furlong_to_pace * pace_to_rod) = 352 := by
  sorry

end rods_in_one_mile_l265_265275


namespace mulberry_sales_l265_265636

theorem mulberry_sales (x : ℝ) (p : ℝ) (h1 : 3000 = x * p)
    (h2 : 150 * (p * 1.4) + (x - 150) * (p * 0.8) - 3000 = 750) :
    x = 200 := by sorry

end mulberry_sales_l265_265636


namespace a_101_mod_49_l265_265380

def a (n : ℕ) : ℕ := 5 ^ n + 9 ^ n

theorem a_101_mod_49 : (a 101) % 49 = 0 :=
by
  -- proof to be filled here
  sorry

end a_101_mod_49_l265_265380


namespace number_of_goats_l265_265951

theorem number_of_goats (C G : ℕ) 
  (h1 : C = 2) 
  (h2 : ∀ G : ℕ, 460 * C + 60 * G = 1400) 
  (h3 : 460 = 460) 
  (h4 : 60 = 60) : 
  G = 8 :=
by
  sorry

end number_of_goats_l265_265951


namespace sum_arithmetic_seq_l265_265979

theorem sum_arithmetic_seq (a d n : ℕ) :
  a = 2 → d = 2 → a + (n - 1) * d = 20 → (n / 2) * (a + (a + (n - 1) * d)) = 110 :=
by sorry

end sum_arithmetic_seq_l265_265979


namespace maximum_reflections_l265_265958

theorem maximum_reflections (θ : ℕ) (h : θ = 10) (max_angle : ℕ) (h_max : max_angle = 180) : 
∃ n : ℕ, n ≤ max_angle / θ ∧ n = 18 := by
  sorry

end maximum_reflections_l265_265958


namespace cups_per_girl_l265_265424

noncomputable def numStudents := 30
noncomputable def numBoys := 10
noncomputable def numCupsByBoys := numBoys * 5
noncomputable def totalCups := 90
noncomputable def numGirls := 2 * numBoys
noncomputable def numCupsByGirls := totalCups - numCupsByBoys

theorem cups_per_girl : (numCupsByGirls / numGirls) = 2 := by
  sorry

end cups_per_girl_l265_265424


namespace find_P_l265_265444

theorem find_P (P Q R S : ℕ) (h1: P ≠ Q) (h2: R ≠ S) (h3: P * Q = 72) (h4: R * S = 72) (h5: P - Q = R + S) :
  P = 18 := 
  sorry

end find_P_l265_265444


namespace inequality_3var_l265_265594

theorem inequality_3var (x y z : ℝ) (h₁ : 0 ≤ x) (h₂ : 0 ≤ y) (h₃ : 0 ≤ z) (h₄ : x * y + y * z + z * x = 1) : 
    1 / (x + y) + 1 / (y + z) + 1 / (z + x) ≥ 5 / 2 :=
sorry

end inequality_3var_l265_265594


namespace common_difference_of_arithmetic_sequence_l265_265570

variable (a1 d : ℤ)
def S : ℕ → ℤ
| 0     => 0
| (n+1) => S n + (a1 + n * d)

theorem common_difference_of_arithmetic_sequence
  (h : 2 * S a1 d 3 = 3 * S a1 d 2 + 6) :
  d = 2 :=
  sorry

end common_difference_of_arithmetic_sequence_l265_265570


namespace jane_savings_l265_265789

-- Given conditions
def cost_pair_1 : ℕ := 50
def cost_pair_2 : ℕ := 40

def promotion_A (cost1 cost2 : ℕ) : ℕ :=
  cost1 + cost2 / 2

def promotion_B (cost1 cost2 : ℕ) : ℕ :=
  cost1 + (cost2 - 15)

-- Define the savings calculation
def savings (promoA promoB : ℕ) : ℕ :=
  promoB - promoA

-- Specify the theorem to prove
theorem jane_savings :
  savings (promotion_A cost_pair_1 cost_pair_2) (promotion_B cost_pair_1 cost_pair_2) = 5 := 
by
  sorry

end jane_savings_l265_265789


namespace max_m_eq_half_l265_265841

noncomputable def f (x m : ℝ) : ℝ := (1/2) * x^2 + m * x + m * Real.log x

theorem max_m_eq_half :
  ∃ m : ℝ, (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (∀ x1 x2 : ℝ, (1 ≤ x1 ∧ x1 ≤ 2) → (1 ≤ x2 ∧ x2 ≤ 2) → 
  x1 < x2 → |f x1 m - f x2 m| < x2^2 - x1^2)) ∧ m = 1/2 :=
sorry

end max_m_eq_half_l265_265841


namespace fraction_irreducible_l265_265394

theorem fraction_irreducible (n : ℕ) : Nat.gcd (12 * n + 1) (30 * n + 2) = 1 :=
sorry

end fraction_irreducible_l265_265394


namespace average_total_goals_l265_265677

theorem average_total_goals (carter_avg shelby_avg judah_avg total_avg : ℕ) 
    (h1: carter_avg = 4) 
    (h2: shelby_avg = carter_avg / 2)
    (h3: judah_avg = 2 * shelby_avg - 3) 
    (h4: total_avg = carter_avg + shelby_avg + judah_avg) :
  total_avg = 7 :=
by
  sorry

end average_total_goals_l265_265677


namespace umbrellas_problem_l265_265242

theorem umbrellas_problem :
  ∃ (b r : ℕ), b = 36 ∧ r = 27 ∧ 
  b = (45 + r) / 2 ∧ 
  r = (45 + b) / 3 :=
by sorry

end umbrellas_problem_l265_265242


namespace turtle_feeding_cost_l265_265158

theorem turtle_feeding_cost (total_weight_pounds : ℝ) (food_per_half_pound : ℝ)
  (jar_food_ounces : ℝ) (jar_cost_dollars : ℝ) (total_cost : ℝ) : 
  total_weight_pounds = 30 →
  food_per_half_pound = 1 →
  jar_food_ounces = 15 →
  jar_cost_dollars = 2 →
  total_cost = 8 :=
by
  intros h_weight h_food h_jar_ounces h_jar_cost
  sorry

end turtle_feeding_cost_l265_265158


namespace magician_trick_l265_265650

theorem magician_trick (coins : Fin 11 → Bool) 
  (exists_pair : ∃ i : Fin 11, coins i = coins ((i + 1) % 11)) 
  (assistant_left_uncovered : Fin 11) 
  (uncovered_coin_same_face : coins assistant_left_uncovered = coins (some (exists_pair))) :
  ∃ j : Fin 11, j ≠ assistant_left_uncovered ∧ coins j = coins assistant_left_uncovered := 
  sorry

end magician_trick_l265_265650


namespace jenny_money_l265_265523

theorem jenny_money (x : ℝ) (h : (4 / 7) * x = 24) : (x / 2) = 21 := 
sorry

end jenny_money_l265_265523


namespace bridge_length_l265_265667

theorem bridge_length (train_length : ℕ) (train_cross_bridge_time : ℕ) (train_cross_lamp_time : ℕ) (bridge_length : ℕ) :
  train_length = 600 →
  train_cross_bridge_time = 70 →
  train_cross_lamp_time = 20 →
  bridge_length = 1500 :=
by
  intro h1 h2 h3
  sorry

end bridge_length_l265_265667


namespace sum_of_sequence_l265_265494

def a (n : ℕ) : ℕ := 2 * n + 1 + 2^n

def S (n : ℕ) : ℕ := (Finset.range n).sum (λ k => a (k + 1))

theorem sum_of_sequence (n : ℕ) : S n = n^2 + 2 * n + 2^(n + 1) - 2 := 
by 
  sorry

end sum_of_sequence_l265_265494


namespace mixture_correct_l265_265369

def water_amount : ℚ := (3/5) * 20
def vinegar_amount : ℚ := (5/6) * 18
def mixture_amount : ℚ := water_amount + vinegar_amount

theorem mixture_correct : mixture_amount = 27 := 
by
  -- Here goes the proof steps
  sorry

end mixture_correct_l265_265369


namespace greatest_two_digit_multiple_of_17_l265_265194

theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n < 100 ∧ n ≥ 10 ∧ 17 ∣ n ∧ ∀ m, m < 100 ∧ m ≥ 10 ∧ 17 ∣ m → m ≤ 85 :=
by
  use 85
  -- Prove conditions follow sorry
  sorry

end greatest_two_digit_multiple_of_17_l265_265194


namespace prove_common_difference_l265_265573

variable {a1 d : ℤ}
variable (S : ℕ → ℤ)
variable (arith_seq : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
def sum_first_n (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Define the condition given in the problem
def problem_condition : Prop := 2 * sum_first_n 3 = 3 * sum_first_n 2 + 6

-- Define that the common difference d is 2
def d_value_is_2 : Prop := d = 2

theorem prove_common_difference (h : problem_condition) : d_value_is_2 :=
by
  sorry

end prove_common_difference_l265_265573


namespace hex_A08_to_decimal_l265_265256

noncomputable def hex_A := 10
noncomputable def hex_A08_base_10 : ℕ :=
  (hex_A * 16^2) + (0 * 16^1) + (8 * 16^0)

theorem hex_A08_to_decimal :
  hex_A08_base_10 = 2568 :=
by
  sorry

end hex_A08_to_decimal_l265_265256


namespace bulb_standard_probability_l265_265671

noncomputable def prob_A 
  (P_H1 : ℝ) (P_H2 : ℝ) (P_A_given_H1 : ℝ) (P_A_given_H2 : ℝ) :=
  P_A_given_H1 * P_H1 + P_A_given_H2 * P_H2

theorem bulb_standard_probability 
  (P_H1 : ℝ := 0.6) (P_H2 : ℝ := 0.4) 
  (P_A_given_H1 : ℝ := 0.95) (P_A_given_H2 : ℝ := 0.85) :
  prob_A P_H1 P_H2 P_A_given_H1 P_A_given_H2 = 0.91 :=
by
  sorry

end bulb_standard_probability_l265_265671


namespace angle_same_terminal_side_315_l265_265160

theorem angle_same_terminal_side_315 (k : ℤ) : ∃ α, α = k * 360 + 315 ∧ α = -45 :=
by
  use -45
  sorry

end angle_same_terminal_side_315_l265_265160


namespace part1_part2_l265_265670

section
  variable {x a : ℝ}

  def f (x a : ℝ) := |x - a| + 3 * x

  theorem part1 (h : a = 1) : 
    (∀ x, f x a ≥ 3 * x + 2 ↔ (x ≥ 3 ∨ x ≤ -1)) :=
    sorry

  theorem part2 : 
    (∀ x, (f x a) ≤ 0 ↔ (x ≤ -1)) → a = 2 :=
    sorry
end

end part1_part2_l265_265670


namespace arithmetic_sequence_common_difference_l265_265542

theorem arithmetic_sequence_common_difference 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : S 3 = a 1 + a 2 + a 3)
  (h2 : S 2 = a 1 + a 2)
  (h3 : a 2 = a 1 + d)
  (h4 : a 3 = a 1 + 2 * d)
  (h5 : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l265_265542


namespace sum_alternating_series_l265_265088

theorem sum_alternating_series :
  (Finset.sum (Finset.range 2023) (λ k => (-1)^(k + 1))) = -1 := 
by
  sorry

end sum_alternating_series_l265_265088


namespace sum_of_fractions_decimal_equivalence_l265_265261

theorem sum_of_fractions :
  (2 / 15 : ℚ) + (4 / 20) + (5 / 45) = 4 / 9 := 
sorry

theorem decimal_equivalence :
  (4 / 9 : ℚ) = 0.444 := 
sorry

end sum_of_fractions_decimal_equivalence_l265_265261


namespace law_of_cosines_l265_265637

theorem law_of_cosines (a b c : ℝ) (A : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ A ≥ 0 ∧ A ≤ π) :
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A :=
sorry

end law_of_cosines_l265_265637


namespace ivan_total_pay_l265_265864

theorem ivan_total_pay (cost_per_card : ℕ) (number_of_cards : ℕ) (discount_per_card : ℕ) :
  cost_per_card = 12 → number_of_cards = 10 → discount_per_card = 2 →
  (number_of_cards * (cost_per_card - discount_per_card)) = 100 :=
by
  intro h1 h2 h3
  sorry

end ivan_total_pay_l265_265864


namespace plates_count_l265_265452

theorem plates_count (n : ℕ)
  (h1 : 500 < n)
  (h2 : n < 600)
  (h3 : n % 10 = 7)
  (h4 : n % 12 = 7) : n = 547 :=
sorry

end plates_count_l265_265452


namespace cone_to_sphere_ratio_l265_265961

-- Prove the ratio of the cone's altitude to its base radius
theorem cone_to_sphere_ratio (r h : ℝ) (h_r_pos : 0 < r) 
  (vol_cone : ℝ) (vol_sphere : ℝ) 
  (hyp_vol_relation : vol_cone = (1 / 3) * vol_sphere)
  (vol_sphere_def : vol_sphere = (4 / 3) * π * r^3)
  (vol_cone_def : vol_cone = (1 / 3) * π * r^2 * h) :
  h / r = 4 / 3 :=
by
  sorry

end cone_to_sphere_ratio_l265_265961


namespace find_common_difference_l265_265579

-- Define the arithmetic sequence and the sum of the first n terms
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
def sum_of_first_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ := ∑ k in finset.range n, arithmetic_sequence a₁ d (k + 1)

-- Given condition
def condition (a₁ d : ℝ) : Prop := 
  2 * sum_of_first_n_terms a₁ d 3 = 3 * sum_of_first_n_terms a₁ d 2 + 6

-- The proof statement
theorem find_common_difference (a₁ d : ℝ) (h : condition a₁ d) : d = 2 :=
by
  sorry

end find_common_difference_l265_265579


namespace ara_current_height_l265_265460

variable (h : ℝ)  -- Original height of both Shea and Ara
variable (sheas_growth_rate : ℝ := 0.20)  -- Shea's growth rate (20%)
variable (sheas_current_height : ℝ := 60)  -- Shea's current height
variable (aras_growth_rate : ℝ := 0.5)  -- Ara's growth rate in terms of Shea's growth

theorem ara_current_height : 
  h * (1 + sheas_growth_rate) = sheas_current_height →
  (h + (sheas_current_height - h) * aras_growth_rate) = 55 :=
  by
    sorry

end ara_current_height_l265_265460


namespace tori_original_height_l265_265054

-- Definitions for given conditions
def current_height : ℝ := 7.26
def height_gained : ℝ := 2.86

-- Theorem statement
theorem tori_original_height : current_height - height_gained = 4.40 :=
by sorry

end tori_original_height_l265_265054


namespace good_students_options_l265_265345

variables (E B : ℕ)

-- Define the condition that the class has 25 students
def total_students : Prop := E + B = 25

-- Define the condition given by the first group of students
def first_group_condition : Prop := B > 12

-- Define the condition given by the second group of students
def second_group_condition : Prop := B = 3 * (E - 1)

-- Define the problem statement
theorem good_students_options (E B : ℕ) :
  total_students E B → first_group_condition B → second_group_condition E B → (E = 5 ∨ E = 7) :=
by
  intros h1 h2 h3
  sorry

end good_students_options_l265_265345


namespace arrangement_problem_l265_265761

def numWaysToArrangeParticipants : ℕ := 90

theorem arrangement_problem :
  ∃ (boys : ℕ) (girls : ℕ) (select_boys : ℕ → ℕ) (select_girls : ℕ → ℕ)
    (arrange : ℕ × ℕ × ℕ → ℕ),
  boys = 3 ∧ girls = 5 ∧
  select_boys boys = 3 ∧ select_girls girls = 5 ∧ 
  arrange (select_boys boys, select_girls girls, 2) = numWaysToArrangeParticipants :=
by
  sorry

end arrangement_problem_l265_265761


namespace avg_goals_per_game_l265_265679

def carter_goals_per_game := 4
def shelby_goals_per_game := carter_goals_per_game / 2
def judah_goals_per_game := (2 * shelby_goals_per_game) - 3
def average_total_goals_team := carter_goals_per_game + shelby_goals_per_game + judah_goals_per_game

theorem avg_goals_per_game : average_total_goals_team = 7 :=
by
  -- Proof would go here
  sorry

end avg_goals_per_game_l265_265679


namespace kitchen_chairs_count_l265_265955

-- Define the conditions
def total_chairs : ℕ := 9
def living_room_chairs : ℕ := 3

-- Prove the number of kitchen chairs
theorem kitchen_chairs_count : total_chairs - living_room_chairs = 6 := by
  -- Proof goes here
  sorry

end kitchen_chairs_count_l265_265955


namespace polygon_interior_angle_sum_l265_265617

theorem polygon_interior_angle_sum (n : ℕ) (hn : 3 ≤ n) :
  (n - 2) * 180 + 180 = 2007 → n = 13 := by
  sorry

end polygon_interior_angle_sum_l265_265617


namespace common_difference_l265_265565

variable (a1 d : ℤ)
variable (S : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def sum_first_n_terms (n : ℕ) : ℤ :=
  n * a1 + d * (n * (n - 1) / 2)

-- Condition: 2 * S 3 = 3 * S 2 + 6
axiom cond : 2 * sum_first_n_terms 3 = 3 * sum_first_n_terms 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem common_difference : d = 2 :=
by
  sorry

end common_difference_l265_265565


namespace angle_of_inclination_l265_265852

theorem angle_of_inclination (A B : ℝ × ℝ) (hA : A = (2, 5)) (hB : B = (4, 3)) : 
  ∃ θ : ℝ, θ = (3 * Real.pi) / 4 ∧ (∃ k : ℝ, k = (A.2 - B.2) / (A.1 - B.1) ∧ Real.tan θ = k) :=
by
  sorry

end angle_of_inclination_l265_265852


namespace equivalent_expression_l265_265402

theorem equivalent_expression (m n : ℕ) (P Q : ℕ) (hP : P = 3^m) (hQ : Q = 5^n) :
  15^(m + n) = P * Q :=
by
  sorry

end equivalent_expression_l265_265402


namespace smallest_n_interval_l265_265759

theorem smallest_n_interval :
  ∃ n : ℕ, (∃ x : ℤ, ⌊10 ^ n / x⌋ = 2006) ∧ 7 ≤ n ∧ n ≤ 12 :=
sorry

end smallest_n_interval_l265_265759


namespace only_solution_l265_265686

def phi : ℕ → ℕ := sorry  -- Euler's totient function
def d : ℕ → ℕ := sorry    -- Divisor function

theorem only_solution (n : ℕ) (h1 : n ∣ (phi n)^(d n) + 1) (h2 : ¬ d n ^ 5 ∣ n ^ (phi n) - 1) : n = 2 :=
sorry

end only_solution_l265_265686


namespace book_weight_l265_265643

theorem book_weight (total_weight : ℕ) (num_books : ℕ) (each_book_weight : ℕ) 
  (h1 : total_weight = 42) (h2 : num_books = 14) :
  each_book_weight = total_weight / num_books :=
by
  sorry

end book_weight_l265_265643


namespace sin_alpha_beta_eq_l265_265836

theorem sin_alpha_beta_eq 
  (α β : ℝ) 
  (h1 : π / 4 < α) (h2 : α < 3 * π / 4)
  (h3 : 0 < β) (h4 : β < π / 4)
  (h5: Real.sin (α + π / 4) = 3 / 5)
  (h6: Real.cos (π / 4 + β) = 5 / 13) :
  Real.sin (α + β) = 56 / 65 :=
sorry

end sin_alpha_beta_eq_l265_265836


namespace students_spend_185_minutes_in_timeout_l265_265021

variable (tR tF tS t_total : ℕ)

-- Conditions
def running_timeouts : ℕ := 5
def food_timeouts : ℕ := 5 * running_timeouts - 1
def swearing_timeouts : ℕ := food_timeouts / 3
def total_timeouts : ℕ := running_timeouts + food_timeouts + swearing_timeouts
def timeout_duration : ℕ := 5

-- Total time spent in time-out
def total_timeout_minutes : ℕ := total_timeouts * timeout_duration

theorem students_spend_185_minutes_in_timeout :
  total_timeout_minutes = 185 :=
by
  -- The answer is directly given by the conditions and the correct answer identified.
  sorry

end students_spend_185_minutes_in_timeout_l265_265021


namespace calculate_total_marks_l265_265372

theorem calculate_total_marks 
  (total_questions : ℕ) 
  (correct_answers : ℕ) 
  (marks_per_correct : ℕ) 
  (marks_per_wrong : ℤ) 
  (total_attempted : total_questions = 60) 
  (correct_attempted : correct_answers = 44)
  (marks_per_correct_is_4 : marks_per_correct = 4)
  (marks_per_wrong_is_neg1 : marks_per_wrong = -1) : 
  total_questions * marks_per_correct - (total_questions - correct_answers) * (abs marks_per_wrong) = 160 := 
by 
  sorry

end calculate_total_marks_l265_265372


namespace tom_calories_l265_265916

theorem tom_calories :
  let carrot_pounds := 1
  let broccoli_pounds := 2 * carrot_pounds
  let carrot_calories_per_pound := 51
  let broccoli_calories_per_pound := carrot_calories_per_pound / 3
  let total_carrot_calories := carrot_pounds * carrot_calories_per_pound
  let total_broccoli_calories := broccoli_pounds * broccoli_calories_per_pound
  let total_calories := total_carrot_calories + total_broccoli_calories
  total_calories = 85 :=
by
  sorry

end tom_calories_l265_265916


namespace remainder_of_division_l265_265052

theorem remainder_of_division (x y R : ℕ) 
  (h1 : y = 1782)
  (h2 : y - x = 1500)
  (h3 : y = 6 * x + R) :
  R = 90 :=
by
  sorry

end remainder_of_division_l265_265052


namespace oldest_daily_cheese_l265_265525

-- Given conditions
def days_per_week : ℕ := 5
def weeks : ℕ := 4
def youngest_daily : ℕ := 1
def cheeses_per_pack : ℕ := 30
def packs_needed : ℕ := 2

-- Derived conditions
def total_days : ℕ := days_per_week * weeks
def total_cheeses : ℕ := packs_needed * cheeses_per_pack
def youngest_total_cheeses : ℕ := youngest_daily * total_days
def oldest_total_cheeses : ℕ := total_cheeses - youngest_total_cheeses

-- Prove that the oldest child wants 2 string cheeses per day
theorem oldest_daily_cheese : oldest_total_cheeses / total_days = 2 := by
  sorry

end oldest_daily_cheese_l265_265525


namespace milk_production_l265_265969

theorem milk_production (a b c d e : ℕ) (f g : ℝ) (hf : f = 0.8) (hg : g = 1.1) :
  ((d : ℝ) * e * g * (b : ℝ) / (a * c)) = 1.1 * b * d * e / (a * c) := by
  sorry

end milk_production_l265_265969


namespace probability_product_positive_of_independent_selection_l265_265923

theorem probability_product_positive_of_independent_selection :
  let I := set.Icc (-30 : ℝ) (15 : ℝ)
  let P := (λ (x y : ℝ), x ∈ I ∧ y ∈ I ∧ x * y > 0)
  (Prob { x : ℝ × ℝ | P x.1 x.2 } :
    ProbabilitySpace (I × I)) = 5 / 9 :=
by
  sorry

end probability_product_positive_of_independent_selection_l265_265923


namespace basketball_probability_third_shot_l265_265641

theorem basketball_probability_third_shot
  (p1 : ℚ) (p2_given_made1 : ℚ) (p2_given_missed1 : ℚ) (p3_given_made2 : ℚ) (p3_given_missed2 : ℚ) :
  p1 = 2 / 3 → p2_given_made1 = 2 / 3 → p2_given_missed1 = 1 / 3 → p3_given_made2 = 2 / 3 → p3_given_missed2 = 2 / 3 →
  (p1 * p2_given_made1 * p3_given_made2 + p1 * p2_given_missed1 * p3_given_misseds2 + 
   (1 - p1) * p2_given_made1 * p3_given_made2 + (1 - p1) * p2_given_missed1 * p3_given_missed2) = 14 / 27 :=
by
  sorry

end basketball_probability_third_shot_l265_265641


namespace product_positive_probability_l265_265930

theorem product_positive_probability :
  let interval := set.Icc (-30 : ℝ) 15 in
  let prob_neg := (30 : ℝ) / 45 in
  let prob_pos := (15 : ℝ) / 45 in
  let prob_product_neg := 2 * (prob_neg * prob_pos) in
  let prob_product_pos := (prob_neg ^ 2) + (prob_pos ^ 2) in
  (prob_product_pos = 5 / 9) :=
by
  sorry

end product_positive_probability_l265_265930


namespace good_students_options_l265_265343

variables (E B : ℕ)

-- Define the condition that the class has 25 students
def total_students : Prop := E + B = 25

-- Define the condition given by the first group of students
def first_group_condition : Prop := B > 12

-- Define the condition given by the second group of students
def second_group_condition : Prop := B = 3 * (E - 1)

-- Define the problem statement
theorem good_students_options (E B : ℕ) :
  total_students E B → first_group_condition B → second_group_condition E B → (E = 5 ∨ E = 7) :=
by
  intros h1 h2 h3
  sorry

end good_students_options_l265_265343


namespace number_of_good_students_is_5_or_7_l265_265354

-- Definitions based on the conditions
def total_students : ℕ := 25
def number_of_good_students (G : ℕ) (T : ℕ) := G + T = total_students
def first_group_condition (T : ℕ) := T > 12
def second_group_condition (G : ℕ) (T : ℕ) := T = 3 * (G - 1)

-- Problem statement in Lean 4:
theorem number_of_good_students_is_5_or_7 (G T : ℕ) :
  number_of_good_students G T ∧ first_group_condition T ∧ second_group_condition G T → G = 5 ∨ G = 7 :=
by
  sorry

end number_of_good_students_is_5_or_7_l265_265354


namespace martin_less_than_43_l265_265603

variable (C K M : ℕ)

-- Conditions
def campbell_correct := C = 35
def kelsey_correct := K = C + 8
def martin_fewer := M < K

-- Conclusion we want to prove
theorem martin_less_than_43 (h1 : campbell_correct C) (h2 : kelsey_correct C K) (h3 : martin_fewer K M) : M < 43 := 
by {
  sorry
}

end martin_less_than_43_l265_265603


namespace find_number_of_good_students_l265_265328

variable {G T : ℕ}

def is_good_or_troublemaker (G T : ℕ) : Prop :=
  G + T = 25 ∧ 
  (∀ s ∈ finset.range 5, T > 12) ∧ 
  (∀ s ∈ finset.range 20, T = 3 * (G - 1))

-- There are 25 students in total. 
-- Each student is either a good student or a troublemaker.
-- Good students always tell the truth, while troublemakers always lie.
-- Five students make the statement: "If I transfer to another class, more than half of the remaining students will be troublemakers."
-- The remaining 20 students make the statement: "If I transfer to another class, the number of troublemakers among the remaining students will be three times the number of good students."

theorem find_number_of_good_students (G T : ℕ) (h : is_good_or_troublemaker G T) :
  (G = 5 ∨ G = 7) :=
sorry

end find_number_of_good_students_l265_265328


namespace linear_combination_of_matrices_l265_265826

variable (A B : Matrix (Fin 3) (Fin 3) ℤ) 

def matrixA : Matrix (Fin 3) (Fin 3) ℤ := 
  ![
    ![2, -4, 0],
    ![-1, 5, 1],
    ![0, 3, -7]
  ]

def matrixB : Matrix (Fin 3) (Fin 3) ℤ := 
  ![
    ![4, -1, -2],
    ![0, -3, 5],
    ![2, 0, -4]
  ]

theorem linear_combination_of_matrices :
  3 • matrixA - 2 • matrixB = 
  ![
    ![-2, -10, 4],
    ![-3, 21, -7],
    ![-4, 9, -13]
  ] :=
sorry

end linear_combination_of_matrices_l265_265826


namespace units_digit_4659_pow_157_l265_265983

theorem units_digit_4659_pow_157 : 
  (4659^157) % 10 = 9 := 
by 
  sorry

end units_digit_4659_pow_157_l265_265983


namespace combined_age_of_four_siblings_l265_265668

theorem combined_age_of_four_siblings :
  let aaron_age := 15
  let sister_age := 3 * aaron_age
  let henry_age := 4 * sister_age
  let alice_age := aaron_age - 2
  aaron_age + sister_age + henry_age + alice_age = 253 :=
by
  let aaron_age := 15
  let sister_age := 3 * aaron_age
  let henry_age := 4 * sister_age
  let alice_age := aaron_age - 2
  have h1 : aaron_age + sister_age + henry_age + alice_age = 15 + 3 * 15 + 4 * (3 * 15) + (15 - 2) := by sorry
  have h2 : 15 + 3 * 15 + 4 * (3 * 15) + (15 - 2) = 253 := by sorry
  exact h1.trans h2

end combined_age_of_four_siblings_l265_265668


namespace total_weight_of_13_gold_bars_l265_265165

theorem total_weight_of_13_gold_bars
    (C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 : ℝ)
    (w12 w13 w23 w45 w67 w89 w1011 w1213 : ℝ)
    (h1 : w12 = C1 + C2)
    (h2 : w13 = C1 + C3)
    (h3 : w23 = C2 + C3)
    (h4 : w45 = C4 + C5)
    (h5 : w67 = C6 + C7)
    (h6 : w89 = C8 + C9)
    (h7 : w1011 = C10 + C11)
    (h8 : w1213 = C12 + C13) :
    C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8 + C9 + C10 + C11 + C12 + C13 = 
    (C1 + C2 + C3) + (C4 + C5) + (C6 + C7) + (C8 + C9) + (C10 + C11) + (C12 + C13) := 
  by
  sorry

end total_weight_of_13_gold_bars_l265_265165


namespace min_right_triangle_side_l265_265758

theorem min_right_triangle_side (s : ℕ) : 
  (7^2 + 24^2 = s^2 ∧ 7 + 24 > s ∧ 24 + s > 7 ∧ 7 + s > 24) → s = 25 :=
by
  intro h
  sorry

end min_right_triangle_side_l265_265758


namespace permutation_sum_eq_744_l265_265693

open Nat

theorem permutation_sum_eq_744 (n : ℕ) (h1 : n ≠ 0) (h2 : n + 3 ≤ 2 * n) (h3 : n + 1 ≤ 4) :
  choose (2 * n) (n + 3) + choose 4 (n + 1) = 744 := by
  sorry

end permutation_sum_eq_744_l265_265693


namespace common_difference_of_arithmetic_sequence_l265_265536

noncomputable def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range n, a i

def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_cond : 2 * S a 3 = 3 * S a 2 + 6) :
  ∃ d : ℝ, d = 2 := sorry

end common_difference_of_arithmetic_sequence_l265_265536


namespace correct_calculation_is_A_l265_265217

theorem correct_calculation_is_A : (1 + (-2)) = -1 :=
by 
  sorry

end correct_calculation_is_A_l265_265217


namespace diagonals_of_polygon_l265_265610

theorem diagonals_of_polygon (f : ℕ → ℕ) (k : ℕ) (h_k : k ≥ 3) : f (k + 1) = f k + (k - 1) :=
sorry

end diagonals_of_polygon_l265_265610


namespace quadratic_expression_value_l265_265905

theorem quadratic_expression_value (x₁ x₂ : ℝ) (h₁ : x₁^2 - 3 * x₁ + 1 = 0) (h₂ : x₂^2 - 3 * x₂ + 1 = 0) :
  x₁^2 + 3 * x₂ + x₁ * x₂ - 2 = 7 :=
by
  sorry

end quadratic_expression_value_l265_265905


namespace turtle_feeding_cost_l265_265157

def cost_to_feed_turtles (turtle_weight: ℝ) (food_per_half_pound: ℝ) (jar_capacity: ℝ) (jar_cost: ℝ) : ℝ :=
  let total_food := turtle_weight * (food_per_half_pound / 0.5)
  let total_jars := total_food / jar_capacity
  total_jars * jar_cost

theorem turtle_feeding_cost :
  cost_to_feed_turtles 30 1 15 2 = 8 :=
by
  sorry

end turtle_feeding_cost_l265_265157


namespace alan_tickets_l265_265910

theorem alan_tickets (a m : ℕ) (h1 : a + m = 150) (h2 : m = 5 * a - 6) : a = 26 :=
by
  sorry

end alan_tickets_l265_265910


namespace jennifer_sweets_l265_265520

theorem jennifer_sweets :
  let green_sweets := 212
  let blue_sweets := 310
  let yellow_sweets := 502
  let total_sweets := green_sweets + blue_sweets + yellow_sweets
  let number_of_people := 4
  total_sweets / number_of_people = 256 := 
by
  sorry

end jennifer_sweets_l265_265520


namespace smallest_K_222_multiple_of_198_l265_265278

theorem smallest_K_222_multiple_of_198 :
  ∀ K : ℕ, (∃ x : ℕ, x = 2 * (10^K - 1) / 9 ∧ x % 198 = 0) → K = 18 :=
by
  sorry

end smallest_K_222_multiple_of_198_l265_265278


namespace geometric_sequence_ratio_l265_265749
-- Lean 4 Code

noncomputable def a_n (a : ℝ) (q : ℝ) (n : ℕ) : ℝ := a * q ^ n

theorem geometric_sequence_ratio
  (a : ℝ)
  (q : ℝ)
  (h_pos : a > 0)
  (h_q_neq_1 : q ≠ 1)
  (h_arith_seq : 2 * a_n a q 4 = a_n a q 2 + a_n a q 5)
  : (a_n a q 2 + a_n a q 3) / (a_n a q 3 + a_n a q 4) = (Real.sqrt 5 - 1) / 2 :=
by {
  sorry
}

end geometric_sequence_ratio_l265_265749


namespace integer_satisfies_inequality_l265_265772

theorem integer_satisfies_inequality (n : ℤ) : 
  (3 : ℚ) / 10 < n / 20 ∧ n / 20 < 2 / 5 → n = 7 :=
sorry

end integer_satisfies_inequality_l265_265772


namespace ivan_total_pay_l265_265865

theorem ivan_total_pay (cost_per_card : ℕ) (number_of_cards : ℕ) (discount_per_card : ℕ) :
  cost_per_card = 12 → number_of_cards = 10 → discount_per_card = 2 →
  (number_of_cards * (cost_per_card - discount_per_card)) = 100 :=
by
  intro h1 h2 h3
  sorry

end ivan_total_pay_l265_265865


namespace intersection_point_exists_l265_265284

def line_l (x y : ℝ) : Prop := 2 * x + y = 10
def line_l_prime (x y : ℝ) : Prop := x - 2 * y + 10 = 0
def passes_through (x y : ℝ) (p : ℝ × ℝ) : Prop := p.2 = y ∧ 2 * p.1 - 10 = x

theorem intersection_point_exists :
  ∃ p : ℝ × ℝ, line_l p.1 p.2 ∧ line_l_prime p.1 p.2 ∧ passes_through p.1 p.2 (-10, 0) :=
sorry

end intersection_point_exists_l265_265284


namespace peony_total_count_l265_265608

theorem peony_total_count (n : ℕ) (x : ℕ) (total_sample : ℕ) (single_sample : ℕ) (double_sample : ℕ) (thousand_sample : ℕ) (extra_thousand : ℕ)
    (h1 : thousand_sample > single_sample)
    (h2 : thousand_sample - single_sample = extra_thousand)
    (h3 : total_sample = single_sample + double_sample + thousand_sample)
    (h4 : total_sample = 12)
    (h5 : single_sample = 4)
    (h6 : double_sample = 2)
    (h7 : thousand_sample = 6)
    (h8 : extra_thousand = 30) :
    n = 180 :=
by 
  sorry

end peony_total_count_l265_265608


namespace cannot_form_triangle_l265_265060

theorem cannot_form_triangle (a b c : ℕ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  ¬ ∃ a b c : ℕ, (a, b, c) = (1, 2, 3) := 
  sorry

end cannot_form_triangle_l265_265060


namespace problem_statements_l265_265061

theorem problem_statements :
  let S1 := ∀ (x : ℤ) (k : ℤ), x = 2 * k + 1 → (x % 2 = 1)
  let S2 := (∀ (x : ℝ), x > 2 → x > 1) 
            ∧ (∀ (x : ℝ), x > 1 → (x ≥ 2 ∨ x < 2)) 
  let S3 := ∀ (x : ℝ), ¬(∃ (x : ℝ), ∃ (y : ℝ), y = x^2 + 1 ∧ x = y)
  let S4 := ¬(∀ (x : ℝ), x > 1 → x^2 - x > 0) → (∃ (x : ℝ), x > 1 ∧ x^2 - x ≤ 0)
  (S1 ∧ S2 ∧ S3 ∧ ¬S4) := by
    sorry

end problem_statements_l265_265061


namespace solve_problem_l265_265035

-- Definitions based on conditions
def salty_cookies_eaten : ℕ := 28
def sweet_cookies_eaten : ℕ := 15

-- Problem statement
theorem solve_problem : salty_cookies_eaten - sweet_cookies_eaten = 13 := by
  sorry

end solve_problem_l265_265035


namespace minimize_distance_l265_265432

-- Definitions of points and lines in the Euclidean plane
structure Point : Type :=
(x : ℝ)
(y : ℝ)

-- Line is defined by a point and a direction vector
structure Line : Type :=
(point : Point)
(direction : Point)

-- Distance function between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

-- Given conditions
variables (a b : Line) -- lines a and b
variables (A1 A2 : Point) -- positions of point A on line a
variables (B1 B2 : Point) -- positions of point B on line b

-- Hypotheses about uniform motion along the lines
def moves_uniformly (A1 A2 : Point) (a : Line) (B1 B2 : Point) (b : Line) : Prop :=
  ∀ t : ℝ, ∃ (At Bt : Point), 
  At.x = A1.x + t * (A2.x - A1.x) ∧ At.y = A1.y + t * (A2.y - A1.y) ∧
  Bt.x = B1.x + t * (B2.x - B1.x) ∧ Bt.y = B1.y + t * (B2.y - B1.y) ∧
  ∀ s : ℝ, At.x + s * (a.direction.x) = Bt.x + s * (b.direction.x) ∧
           At.y + s * (a.direction.y) = Bt.y + s * (b.direction.y)

-- Problem statement: Prove the existence of points such that AB is minimized
theorem minimize_distance (a b : Line) (A1 A2 B1 B2 : Point) (h : moves_uniformly A1 A2 a B1 B2 b) : 
  ∃ (A B : Point), distance A B = Real.sqrt ((A2.x - B2.x) ^ 2 + (A2.y - B2.y) ^ 2) ∧ distance A B ≤ distance A1 B1 ∧ distance A B ≤ distance A2 B2 :=
sorry

end minimize_distance_l265_265432


namespace megan_dials_correct_number_probability_l265_265382

-- Define the set of possible first three digits
def first_three_digits : Finset ℕ := {296, 299, 295}

-- Define the set of possible last five digits
def last_five_digits : Finset (Finset ℕ) := {Finset.singleton 0, Finset.singleton 1, Finset.singleton 6, Finset.singleton 7, Finset.singleton 8}

-- The total number of possible phone numbers that Megan can dial
def total_possible_numbers : ℕ := (first_three_digits.card) * (5!)

-- The probability that Megan dials Fatima's correct number
def probability_correct_number : ℚ := 1 / total_possible_numbers

theorem megan_dials_correct_number_probability :
  probability_correct_number = 1 / 360 :=
by
  sorry

end megan_dials_correct_number_probability_l265_265382


namespace fibonacci_series_sum_l265_265012

noncomputable def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib (n + 1) + fib n

theorem fibonacci_series_sum :
  (∑' n, (fib n : ℝ) / 7^n) = (49 : ℝ) / 287 := 
by
  sorry

end fibonacci_series_sum_l265_265012


namespace revenue_decrease_1_percent_l265_265904

variable (T C : ℝ)  -- Assumption: T and C are real numbers representing the original tax and consumption

noncomputable def original_revenue : ℝ := T * C
noncomputable def new_tax_rate : ℝ := T * 0.90
noncomputable def new_consumption : ℝ := C * 1.10
noncomputable def new_revenue : ℝ := new_tax_rate T * new_consumption C

theorem revenue_decrease_1_percent :
  new_revenue T C = 0.99 * original_revenue T C := by
  sorry

end revenue_decrease_1_percent_l265_265904


namespace max_soap_boxes_l265_265074

theorem max_soap_boxes 
  (base_width base_length top_width top_length height soap_width soap_length soap_height max_weight soap_weight : ℝ)
  (h_base_dims : base_width = 25)
  (h_base_len : base_length = 42)
  (h_top_width : top_width = 20)
  (h_top_length : top_length = 35)
  (h_height : height = 60)
  (h_soap_width : soap_width = 7)
  (h_soap_length : soap_length = 6)
  (h_soap_height : soap_height = 10)
  (h_max_weight : max_weight = 150)
  (h_soap_weight : soap_weight = 3) :
  (50 = 
    min 
      (⌊top_width / soap_width⌋ * ⌊top_length / soap_length⌋ * ⌊height / soap_height⌋)
      (⌊max_weight / soap_weight⌋)) := by sorry

end max_soap_boxes_l265_265074


namespace triangle_right_angled_l265_265507

theorem triangle_right_angled (A B C : ℝ) (h : A + B + C = 180) (h_ratio : A = 1 * x ∧ B = 2 * x ∧ C = 3 * x) :
  C = 90 :=
by {
  sorry
}

end triangle_right_angled_l265_265507


namespace find_k_l265_265692

theorem find_k (m n k : ℝ) 
  (h1 : 3^m = k) 
  (h2 : 5^n = k) 
  (h3 : 1/m + 1/n = 2) : 
  k = Real.sqrt 15 := 
sorry

end find_k_l265_265692


namespace aqua_park_earnings_l265_265081

def admission_fee : ℕ := 12
def tour_fee : ℕ := 6
def meal_fee : ℕ := 10
def souvenir_fee : ℕ := 8

def group1_admission_count : ℕ := 10
def group1_tour_count : ℕ := 10
def group1_meal_count : ℕ := 10
def group1_souvenir_count : ℕ := 10
def group1_discount : ℚ := 0.10

def group2_admission_count : ℕ := 15
def group2_meal_count : ℕ := 15
def group2_meal_discount : ℚ := 0.05

def group3_admission_count : ℕ := 8
def group3_tour_count : ℕ := 8
def group3_souvenir_count : ℕ := 8

-- total cost for group 1 before discount
def group1_total_before_discount : ℕ := 
  (group1_admission_count * admission_fee) +
  (group1_tour_count * tour_fee) +
  (group1_meal_count * meal_fee) +
  (group1_souvenir_count * souvenir_fee)

-- group 1 total cost after discount
def group1_total_after_discount : ℚ :=
  group1_total_before_discount * (1 - group1_discount)

-- total cost for group 2 before discount
def group2_admission_total_before_discount : ℕ := 
  group2_admission_count * admission_fee
def group2_meal_total_before_discount : ℕ := 
  group2_meal_count * meal_fee

-- group 2 total cost after discount
def group2_meal_total_after_discount : ℚ :=
  group2_meal_total_before_discount * (1 - group2_meal_discount)
def group2_total_after_discount : ℚ :=
  group2_admission_total_before_discount + group2_meal_total_after_discount

-- total cost for group 3 before discount
def group3_total_before_discount : ℕ := 
  (group3_admission_count * admission_fee) +
  (group3_tour_count * tour_fee) +
  (group3_souvenir_count * souvenir_fee)

-- group 3 total cost after discount (no discount applied)
def group3_total_after_discount : ℕ := group3_total_before_discount

-- total earnings from all groups
def total_earnings : ℚ :=
  group1_total_after_discount +
  group2_total_after_discount +
  group3_total_after_discount

theorem aqua_park_earnings : total_earnings = 854.50 := by
  sorry

end aqua_park_earnings_l265_265081


namespace volume_of_regular_triangular_pyramid_l265_265492

noncomputable def regular_triangular_pyramid_volume (h : ℝ) : ℝ :=
  (h^3 * Real.sqrt 3) / 2

theorem volume_of_regular_triangular_pyramid (h : ℝ) :
  regular_triangular_pyramid_volume h = (h^3 * Real.sqrt 3) / 2 :=
by
  sorry

end volume_of_regular_triangular_pyramid_l265_265492


namespace no_integer_solution_l265_265626

theorem no_integer_solution :
  ¬(∃ x : ℤ, 7 - 3 * (x^2 - 2) > 19) :=
by
  sorry

end no_integer_solution_l265_265626


namespace greatest_two_digit_multiple_of_17_l265_265178

theorem greatest_two_digit_multiple_of_17 : ∃ n, (n < 100) ∧ (n % 17 = 0) ∧ (∀ m, (m < 100) ∧ (m % 17 = 0) → m ≤ n) := by
  let multiples := [17, 34, 51, 68, 85]
  have ht : ∀ m, m ∈ multiples → (m < 100) ∧ (m % 17 = 0) := by
    intros m hm
    cases hm with
    | inl h => exact ⟨by norm_num, by norm_num⟩
    | inr hm' =>
      cases hm' with
      | inl h => exact ⟨by norm_num, by norm_num⟩
      | inr hm'' =>
        cases hm'' with
        | inl h => exact ⟨by norm_num, by norm_num⟩
        | inr hm''' =>
          cases hm''' with
          | inl h => exact ⟨by norm_num, by norm_num⟩
          | inr hm'''' =>
            cases hm'''' with
            | inl h => exact ⟨by norm_num, by norm_num⟩
            | inr hm => cases hm
  
  use 85
  split
  { norm_num },
  split
  { exact (by norm_num : 85 % 17 = 0) },
  intro m
  intro h
  exact list.minimum_mem multiples h

end greatest_two_digit_multiple_of_17_l265_265178


namespace hens_count_l265_265956

theorem hens_count (H R : ℕ) (h₁ : H = 9 * R - 5) (h₂ : H + R = 75) : H = 67 :=
by {
  sorry
}

end hens_count_l265_265956


namespace side_face_area_l265_265994

noncomputable def box_lengths (l w h : ℕ) : Prop :=
  (w * h = (1 / 2) * l * w ∧
   l * w = (3 / 2) * l * h ∧
   l * w * h = 5184 ∧
   2 * (l + h) = (6 / 5) * 2 * (l + w))

theorem side_face_area :
  ∃ (l w h : ℕ), box_lengths l w h ∧ l * h = 384 := by
  sorry

end side_face_area_l265_265994


namespace eval_expression_l265_265095

theorem eval_expression :
  (2^2003 * 3^2005) / (6^2004) = 3 / 2 := by
  sorry

end eval_expression_l265_265095


namespace sum_difference_l265_265624

noncomputable def sum_arith_seq (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_difference :
  let S_even := sum_arith_seq 2 2 1001
  let S_odd := sum_arith_seq 1 2 1002
  S_odd - S_even = 1002 :=
by
  sorry

end sum_difference_l265_265624


namespace mary_needs_to_add_l265_265142

-- Define the conditions
def total_flour_required : ℕ := 7
def flour_already_added : ℕ := 2

-- Define the statement that corresponds to the mathematical equivalent proof problem
theorem mary_needs_to_add :
  total_flour_required - flour_already_added = 5 :=
by
  sorry

end mary_needs_to_add_l265_265142


namespace no_real_b_for_inequality_l265_265475

theorem no_real_b_for_inequality (b : ℝ) :
  (∃ x : ℝ, |x^2 + 3*b*x + 4*b| ≤ 5 ∧ (∀ y : ℝ, |y^2 + 3*b*y + 4*b| ≤ 5 → y = x)) → false :=
by
  sorry

end no_real_b_for_inequality_l265_265475


namespace sum_of_transformed_numbers_l265_265252

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) : 3 * (a - 5) + 3 * (b - 5) = 3 * S - 30 :=
by
  sorry

end sum_of_transformed_numbers_l265_265252


namespace focus_of_parabola_l265_265690

theorem focus_of_parabola (a : ℝ) (h : ℝ) (k : ℝ) (x y : ℝ) :
  (∀ x, y = a * (x - h) ^ 2 + k) →
  a = -2 ∧ h = 0 ∧ k = 4 →
  (0, y - (1 / (4 * a))) = (0, 31 / 8) := by
  sorry

end focus_of_parabola_l265_265690


namespace find_a_l265_265696

open Set

noncomputable def A : Set ℝ := {x | x^2 - 2 * x - 8 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 + a * x + a^2 - 12 = 0}

theorem find_a (a : ℝ) : (A ∪ (B a) = A) ↔ (a = -2 ∨ a ≥ 4 ∨ a < -4) := by
  sorry

end find_a_l265_265696


namespace max_handshakes_without_cycles_l265_265639

open BigOperators

theorem max_handshakes_without_cycles :
  ∀ n : ℕ, n = 20 → ∑ i in Finset.range (n - 1), i = 190 :=
by intros;
   sorry

end max_handshakes_without_cycles_l265_265639


namespace part1_part2_part3_l265_265116

namespace Problem

-- Definitions and conditions for problem 1
def f (m x : ℝ) : ℝ := (m + 1) * x^2 - (m - 1) * x + (m - 1)

theorem part1 (m : ℝ) :
  (∀ x : ℝ, f m x < 0) ↔ m < -5/3 := sorry

-- Definitions and conditions for problem 2
theorem part2 (m : ℝ) (h : m < 0) :
  ((-1 < m ∧ m < 0) → ∀ x : ℝ, x ≤ 1 ∨ x ≥ 1 / (m + 1)) ∧
  (m = -1 → ∀ x : ℝ, x ≤ 1) ∧
  (m < -1 → ∀ x : ℝ, 1 / (m + 1) ≤ x ∧ x ≤ 1) := sorry

-- Definitions and conditions for problem 3
theorem part3 (m : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f m x ≥ x^2 + 2 * x) ↔ m ≥ (2 * Real.sqrt 3) / 3 + 1 := sorry

end Problem

end part1_part2_part3_l265_265116


namespace niko_total_profit_l265_265033

def pairs_of_socks : Nat := 9
def cost_per_pair : ℝ := 2
def profit_percentage_first_four : ℝ := 0.25
def profit_per_pair_remaining_five : ℝ := 0.2

theorem niko_total_profit :
  let total_profit_first_four := 4 * (cost_per_pair * profit_percentage_first_four)
  let total_profit_remaining_five := 5 * profit_per_pair_remaining_five
  let total_profit := total_profit_first_four + total_profit_remaining_five
  total_profit = 3 := by
  sorry

end niko_total_profit_l265_265033


namespace no_valid_solutions_l265_265400

theorem no_valid_solutions (x : ℝ) (h : x ≠ 1) : 
  ¬(3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) :=
sorry

end no_valid_solutions_l265_265400


namespace find_sum_u_v_l265_265968

theorem find_sum_u_v (u v : ℤ) (huv : 0 < v ∧ v < u) (pentagon_area : u^2 + 3 * u * v = 451) : u + v = 21 :=
by 
  sorry

end find_sum_u_v_l265_265968


namespace movie_box_office_growth_l265_265754

theorem movie_box_office_growth 
  (x : ℝ) 
  (r₁ r₃ : ℝ) 
  (h₁ : r₁ = 1) 
  (h₃ : r₃ = 2.4) 
  (growth : r₃ = (1 + x) ^ 2) : 
  (1 + x) ^ 2 = 2.4 :=
by sorry

end movie_box_office_growth_l265_265754


namespace packet_b_average_height_l265_265143

theorem packet_b_average_height (x y R_A R_B H_A H_B : ℝ)
  (h_RA : R_A = 2 * x + y)
  (h_RB : R_B = 3 * x - y)
  (h_x : x = 10)
  (h_y : y = 6)
  (h_HA : H_A = 192)
  (h_20percent : H_A = H_B + 0.20 * H_B) :
  H_B = 160 := 
sorry

end packet_b_average_height_l265_265143


namespace sum_is_eighteen_or_twentyseven_l265_265067

theorem sum_is_eighteen_or_twentyseven :
  ∀ (A B C D E I J K L M : ℕ),
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ I ∧ A ≠ J ∧ A ≠ K ∧ A ≠ L ∧ A ≠ M ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ I ∧ B ≠ J ∧ B ≠ K ∧ B ≠ L ∧ B ≠ M ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ I ∧ C ≠ J ∧ C ≠ K ∧ C ≠ L ∧ C ≠ M ∧
  D ≠ E ∧ D ≠ I ∧ D ≠ J ∧ D ≠ K ∧ D ≠ L ∧ D ≠ M ∧
  E ≠ I ∧ E ≠ J ∧ E ≠ K ∧ E ≠ L ∧ E ≠ M ∧
  I ≠ J ∧ I ≠ K ∧ I ≠ L ∧ I ≠ M ∧
  J ≠ K ∧ J ≠ L ∧ J ≠ M ∧
  K ≠ L ∧ K ≠ M ∧
  L ≠ M ∧
  (0 < I) ∧ (0 < J) ∧ (0 < K) ∧ (0 < L) ∧ (0 < M) ∧
  A + B + C + D + E + I + J + K + L + M = 45 ∧
  (I + J + K + L + M) % 10 = 0 →
  A + B + C + D + E + (I + J + K + L + M) / 10 = 18 ∨
  A + B + C + D + E + (I + J + K + L + M) / 10 = 27 :=
by
  intros
  sorry

end sum_is_eighteen_or_twentyseven_l265_265067


namespace point_Q_representation_l265_265875

-- Definitions
variables {C D Q : Type} [AddCommGroup C] [AddCommGroup D] [AddCommGroup Q] [Module ℝ C] [Module ℝ D] [Module ℝ Q]
variable (CQ : ℝ)
variable (QD : ℝ)
variable (r s : ℝ)

-- Given condition: ratio CQ:QD = 7:2
axiom CQ_QD_ratio : CQ / QD = 7 / 2

-- Proof goal: the affine combination representation of the point Q
theorem point_Q_representation : CQ / (CQ + QD) = 7 / 9 ∧ QD / (CQ + QD) = 2 / 9 :=
sorry

end point_Q_representation_l265_265875


namespace smith_boxes_l265_265385

theorem smith_boxes (initial_markers : ℕ) (markers_per_box : ℕ) (total_markers : ℕ)
  (h1 : initial_markers = 32) (h2 : markers_per_box = 9) (h3 : total_markers = 86) :
  total_markers - initial_markers = 6 * markers_per_box :=
by
  -- We state our assumptions explicitly for better clarity
  have h_total : total_markers = 86 := h3
  have h_initial : initial_markers = 32 := h1
  have h_box : markers_per_box = 9 := h2
  sorry

end smith_boxes_l265_265385


namespace good_students_l265_265359

theorem good_students (G T : ℕ) (h1 : G + T = 25) (h2 : T > 12) (h3 : T = 3 * (G - 1)) :
  G = 5 ∨ G = 7 :=
by
  sorry

end good_students_l265_265359


namespace total_number_of_people_l265_265124

theorem total_number_of_people (L F LF N T : ℕ) (hL : L = 13) (hF : F = 15) (hLF : LF = 9) (hN : N = 6) : 
  T = (L + F - LF) + N → T = 25 :=
by
  intros h
  rw [hL, hF, hLF, hN] at h
  exact h

end total_number_of_people_l265_265124


namespace find_k_l265_265752

noncomputable def y (k x : ℝ) : ℝ := k / x

theorem find_k (k : ℝ) (h₁ : k ≠ 0) (h₂ : 1 ≤ 3) 
  (h₃ : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → x = 1 ∨ x = 3) 
  (h₄ : |y k 1 - y k 3| = 4) : k = 6 ∨ k = -6 :=
  sorry

end find_k_l265_265752


namespace q_true_or_false_l265_265321

variable (p q : Prop)

theorem q_true_or_false (h1 : ¬ (p ∧ q)) (h2 : ¬ p) : q ∨ ¬ q :=
by
  sorry

end q_true_or_false_l265_265321


namespace sequence_term_2023_l265_265832

theorem sequence_term_2023 
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ) 
  (h : ∀ n, 2 * S n = a n * (a n + 1)) : 
  a 2023 = 2023 :=
sorry

end sequence_term_2023_l265_265832


namespace man_l265_265775

-- Define the speeds and values given in the problem conditions
def man_speed_with_current : ℝ := 15
def speed_of_current : ℝ := 2.5
def man_speed_against_current : ℝ := 10

-- Define the man's speed in still water as a variable
def man_speed_in_still_water : ℝ := man_speed_with_current - speed_of_current

-- The theorem we need to prove
theorem man's_speed_against_current_is_correct :
  (man_speed_in_still_water - speed_of_current = man_speed_against_current) :=
by
  -- Placeholder for proof
  sorry

end man_l265_265775


namespace common_difference_arithmetic_sequence_l265_265547

theorem common_difference_arithmetic_sequence
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (d : ℕ) 
  (h₁ : ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2) -- sum formula for arithmetic sequence
  (h₂ : 2 * S 3 = 3 * S 2 + 6) : 
  d = 2 := 
sorry

end common_difference_arithmetic_sequence_l265_265547


namespace isosceles_triangle_equal_sides_length_l265_265459

noncomputable def equal_side_length_isosceles_triangle (base median : ℝ) (vertex_angle_deg : ℝ) : ℝ :=
  if base = 36 ∧ median = 15 ∧ vertex_angle_deg = 60 then 3 * Real.sqrt 191 else 0

theorem isosceles_triangle_equal_sides_length:
  equal_side_length_isosceles_triangle 36 15 60 = 3 * Real.sqrt 191 :=
by
  sorry

end isosceles_triangle_equal_sides_length_l265_265459


namespace mary_initial_nickels_l265_265024

variable {x : ℕ}

theorem mary_initial_nickels (h : x + 5 = 12) : x = 7 := by
  sorry

end mary_initial_nickels_l265_265024


namespace number_of_triangles_l265_265392

theorem number_of_triangles (points_AB points_BC points_AC : ℕ)
                            (hAB : points_AB = 12)
                            (hBC : points_BC = 9)
                            (hAC : points_AC = 10) :
    let total_points := points_AB + points_BC + points_AC
    let total_combinations := Nat.choose total_points 3
    let degenerate_AB := Nat.choose points_AB 3
    let degenerate_BC := Nat.choose points_BC 3
    let degenerate_AC := Nat.choose points_AC 3
    let valid_triangles := total_combinations - (degenerate_AB + degenerate_BC + degenerate_AC)
    valid_triangles = 4071 :=
by
  sorry

end number_of_triangles_l265_265392


namespace mean_score_l265_265062

theorem mean_score (mu sigma : ℝ) 
  (h1 : 86 = mu - 7 * sigma) 
  (h2 : 90 = mu + 3 * sigma) :
  mu = 88.8 :=
by
  -- skipping the proof
  sorry

end mean_score_l265_265062


namespace sum_of_acute_angles_pi_over_2_l265_265430

open Real

theorem sum_of_acute_angles_pi_over_2
  {α β : ℝ} (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  (h : sin α * sin α + sin β * sin β = sin (α + β)) :
  α + β = π / 2 :=
sorry

end sum_of_acute_angles_pi_over_2_l265_265430


namespace point_on_circle_x_value_l265_265719

/-
In the xy-plane, the segment with endpoints (-3,0) and (21,0) is the diameter of a circle.
If the point (x,12) is on the circle, then x = 9.
-/
theorem point_on_circle_x_value :
  let c := (9, 0) -- center of the circle
  let r := 12 -- radius of the circle
  let circle := {p | (p.1 - 9)^2 + p.2^2 = 144} -- equation of the circle
  ∀ x : Real, (x, 12) ∈ circle → x = 9 :=
by
  intros
  sorry

end point_on_circle_x_value_l265_265719


namespace max_sum_x1_x2_x3_l265_265017

theorem max_sum_x1_x2_x3 : 
  ∀ (x1 x2 x3 x4 x5 x6 x7 : ℕ), 
    x1 < x2 → x2 < x3 → x3 < x4 → x4 < x5 → x5 < x6 → x6 < x7 →
    x1 + x2 + x3 + x4 + x5 + x6 + x7 = 159 →
    x1 + x2 + x3 = 61 :=
by
  intros x1 x2 x3 x4 x5 x6 x7 h1 h2 h3 h4 h5 h6 h_sum
  sorry

end max_sum_x1_x2_x3_l265_265017


namespace greatest_two_digit_multiple_of_17_is_85_l265_265210

theorem greatest_two_digit_multiple_of_17_is_85 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ m % 17 = 0) → m ≤ n) ∧ n = 85 :=
sorry

end greatest_two_digit_multiple_of_17_is_85_l265_265210


namespace total_amount_paid_l265_265862

def original_price_per_card : Int := 12
def discount_per_card : Int := 2
def number_of_cards : Int := 10

theorem total_amount_paid :
  original_price_per_card - discount_per_card * number_of_cards = 100 :=
by
  sorry

end total_amount_paid_l265_265862


namespace part1_part2_l265_265134

open Real

def f (t x : ℝ) : ℝ := x^2 - (t + 1) * x + t

theorem part1 : 
  ∃ sol_set : Set ℝ, 
    ({x : ℝ | f 3 x > 0} = sol_set) ∧ 
    (sol_set = {x : ℝ | x < 1} ∪ {x : ℝ | x > 3}) :=
by
  use {x : ℝ | x < 1} ∪ {x : ℝ | x > 3}
  sorry

theorem part2 :
  (∀ x : ℝ, f t x ≥ 0) ↔ (t = 1) :=
by
  split
  sorry
  sorry

end part1_part2_l265_265134


namespace slower_plane_speed_l265_265620

-- Let's define the initial conditions and state the theorem in Lean 4
theorem slower_plane_speed 
    (x : ℕ) -- speed of the slower plane
    (h1 : x + 2*x = 900) : -- based on the total distance after 3 hours
    x = 300 :=
by
    -- Proof goes here
    sorry

end slower_plane_speed_l265_265620


namespace flowers_to_embroider_l265_265085

-- Defining constants based on the problem conditions
def stitches_per_minute : ℕ := 4
def stitches_per_flower : ℕ := 60
def stitches_per_unicorn : ℕ := 180
def stitches_per_godzilla : ℕ := 800
def num_unicorns : ℕ := 3
def num_godzillas : ℕ := 1
def total_minutes : ℕ := 1085

-- Theorem statement to prove the number of flowers Carolyn wants to embroider
theorem flowers_to_embroider : 
  (total_minutes * stitches_per_minute - (num_godzillas * stitches_per_godzilla + num_unicorns * stitches_per_unicorn)) / stitches_per_flower = 50 :=
by
  sorry

end flowers_to_embroider_l265_265085


namespace prove_sum_13_l265_265422

open Finset

variables (a_1 d : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Define the arithmetic sequence a_n
def arithmetic_sequence (a_1 d : ℝ) (n : ℕ) : ℝ :=
  a_1 + n * d

-- Define the sum of the first n terms of the arithmetic sequence S_n
def sum_of_first_n_terms (a_1 d : ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (2 * a_1 + (n - 1) * d)

lemma collinear_condition (a_1 d : ℝ) :
  a 2 + a 7 + a 12 = 1 :=
sorry

lemma S_13_value (a_1 d : ℝ) :
  S 13 = 13 * (arithmetic_sequence a_1 d 6) :=
sorry

theorem prove_sum_13 (a_1 d : ℝ) (h1 : S = sum_of_first_n_terms a_1 d)
  (h2 : collinear_condition a_1 d) : 
  S 13 = 13 / 3 :=
begin
  sorry
end

end prove_sum_13_l265_265422


namespace arithmetic_series_sum_l265_265982

theorem arithmetic_series_sum : 
  let a := 2 in 
  let d := 2 in 
  let n := 10 in 
  let l := 20 in 
  (a + l) * n / 2 = 110 := 
by
  sorry

end arithmetic_series_sum_l265_265982


namespace games_bought_at_garage_sale_l265_265733

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

end games_bought_at_garage_sale_l265_265733


namespace correct_computation_gives_l265_265705

variable (x : ℝ)

theorem correct_computation_gives :
  ((3 * x - 12) / 6 = 60) → ((x / 3) + 12 = 160 / 3) :=
by
  sorry

end correct_computation_gives_l265_265705


namespace find_P_l265_265443

theorem find_P (P Q R S : ℕ) (h1: P ≠ Q) (h2: R ≠ S) (h3: P * Q = 72) (h4: R * S = 72) (h5: P - Q = R + S) :
  P = 18 := 
  sorry

end find_P_l265_265443


namespace multiplication_division_l265_265224

theorem multiplication_division:
  (213 * 16 = 3408) → (1.6 * 2.13 = 3.408) :=
by
  sorry

end multiplication_division_l265_265224


namespace area_of_rectangle_ABCD_l265_265663

-- Definitions based on conditions
def side_length_smaller_square := 2
def area_smaller_square := side_length_smaller_square ^ 2
def side_length_larger_square := 3 * side_length_smaller_square
def area_larger_square := side_length_larger_square ^ 2
def area_rect_ABCD := 2 * area_smaller_square + area_larger_square

-- Lean theorem statement for the proof problem
theorem area_of_rectangle_ABCD : area_rect_ABCD = 44 := by
  sorry

end area_of_rectangle_ABCD_l265_265663


namespace caterer_preparations_l265_265673

theorem caterer_preparations :
  let b_guests := 84
  let a_guests := (2/3) * b_guests
  let total_guests := b_guests + a_guests
  let extra_plates := 10
  let total_plates := total_guests + extra_plates

  let cherry_tomatoes_per_plate := 5
  let regular_asparagus_per_plate := 8
  let vegetarian_asparagus_per_plate := 6
  let larger_asparagus_per_plate := 12
  let larger_asparagus_portion_guests := 0.1 * total_plates

  let blueberries_per_plate := 15
  let raspberries_per_plate := 8
  let blackberries_per_plate := 10

  let cherry_tomatoes_needed := cherry_tomatoes_per_plate * total_plates

  let regular_portion_guests := 0.9 * total_plates
  let regular_asparagus_needed := regular_asparagus_per_plate * regular_portion_guests
  let larger_asparagus_needed := larger_asparagus_per_plate * larger_asparagus_portion_guests
  let asparagus_needed := regular_asparagus_needed + larger_asparagus_needed

  let blueberries_needed := blueberries_per_plate * total_plates
  let raspberries_needed := raspberries_per_plate * total_plates
  let blackberries_needed := blackberries_per_plate * total_plates

  cherry_tomatoes_needed = 750 ∧
  asparagus_needed = 1260 ∧
  blueberries_needed = 2250 ∧
  raspberries_needed = 1200 ∧
  blackberries_needed = 1500 :=
by
  -- Proof goes here
  sorry

end caterer_preparations_l265_265673


namespace arithmetic_common_difference_l265_265584

-- Defining the arithmetic sequence sum function S
def S (a₁ a₂ a₃ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if n = 3 then a₁ + a₂ + a₃
  else if n = 2 then a₁ + a₂
  else 0

-- The Lean statement for the problem
theorem arithmetic_common_difference (a₁ a₂ a₃ d : ℝ) 
  (h1 : a₂ = a₁ + d) 
  (h2 : a₃ = a₂ + d)
  (h3 : 2 * S a₁ a₂ a₃ d 3 = 3 * S a₁ a₂ a₃ d 2 + 6) :
  d = 2 :=
by
  sorry

end arithmetic_common_difference_l265_265584


namespace penny_purchase_exceeded_minimum_spend_l265_265002

theorem penny_purchase_exceeded_minimum_spend :
  let bulk_price_per_pound := 5
  let minimum_spend := 40
  let tax_per_pound := 1
  let total_paid := 240
  let total_cost_per_pound := bulk_price_per_pound + tax_per_pound
  let pounds_purchased := total_paid / total_cost_per_pound
  let minimum_pounds_to_spend := minimum_spend / bulk_price_per_pound
  pounds_purchased - minimum_pounds_to_spend = 32 :=
by
  -- The proof is omitted here as per the instructions.
  sorry

end penny_purchase_exceeded_minimum_spend_l265_265002


namespace bottle_and_beverage_weight_l265_265647

theorem bottle_and_beverage_weight 
  (B : ℝ)  -- Weight of the bottle in kilograms
  (x : ℝ)  -- Original weight of the beverage in kilograms
  (h1 : B + 2 * x = 5)  -- Condition: double the beverage weight total
  (h2 : B + 4 * x = 9)  -- Condition: quadruple the beverage weight total
: x = 2 ∧ B = 1 := 
by
  sorry

end bottle_and_beverage_weight_l265_265647


namespace relationship_bx_x2_a2_l265_265495

theorem relationship_bx_x2_a2 {a b x : ℝ} (h1 : b < x) (h2 : x < a) (h3 : 0 < a) (h4 : 0 < b) : 
  b * x < x^2 ∧ x^2 < a^2 :=
by sorry

end relationship_bx_x2_a2_l265_265495


namespace calc_total_push_ups_correct_l265_265464

-- Definitions based on conditions
def sets : ℕ := 9
def push_ups_per_set : ℕ := 12
def reduced_push_ups : ℕ := 8

-- Calculate total push-ups considering the reduction in the ninth set
def total_push_ups (sets : ℕ) (push_ups_per_set : ℕ) (reduced_push_ups : ℕ) : ℕ :=
  (sets - 1) * push_ups_per_set + (push_ups_per_set - reduced_push_ups)

-- Theorem statement
theorem calc_total_push_ups_correct :
  total_push_ups sets push_ups_per_set reduced_push_ups = 100 :=
by
  sorry

end calc_total_push_ups_correct_l265_265464


namespace problem_statement_l265_265296

-- Define the conditions and the goal
theorem problem_statement {x y : ℝ} 
  (h1 : (x + y)^2 = 36)
  (h2 : x * y = 8) : 
  x^2 + y^2 = 20 := 
by
  sorry

end problem_statement_l265_265296


namespace find_q_l265_265311

theorem find_q (p q : ℚ) (h1 : 5 * p + 6 * q = 17) (h2 : 6 * p + 5 * q = 20) : q = 2 / 11 :=
by
  sorry

end find_q_l265_265311


namespace find_a_l265_265698

theorem find_a (a : ℝ) (α : ℝ) (h1 : ∃ (y : ℝ), (a, y) = (a, -2))
(h2 : Real.tan (π + α) = 1 / 3) : a = -6 :=
sorry

end find_a_l265_265698


namespace range_of_a_l265_265712

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) ↔ (-2 * Real.sqrt 2 ≤ a ∧ a ≤ 2 * Real.sqrt 2) :=
sorry

end range_of_a_l265_265712


namespace inverse_of_a_gt_b_implies_a_cubed_gt_b_cubed_l265_265756

theorem inverse_of_a_gt_b_implies_a_cubed_gt_b_cubed:
  (∀ a b : ℝ, a > b → a^3 > b^3) → (∀ a b : ℝ, a^3 > b^3 → a > b) :=
  by
  sorry

end inverse_of_a_gt_b_implies_a_cubed_gt_b_cubed_l265_265756


namespace convert_mps_to_kmph_l265_265776

theorem convert_mps_to_kmph (v_mps : ℝ) (c : ℝ) (h_c : c = 3.6) (h_v_mps : v_mps = 20) : (v_mps * c = 72) :=
by
  rw [h_v_mps, h_c]
  sorry

end convert_mps_to_kmph_l265_265776


namespace abscissa_of_tangent_point_is_2_l265_265813

noncomputable def f (x : ℝ) : ℝ := (x^2) / 4 - 3 * Real.log x

noncomputable def f' (x : ℝ) : ℝ := (1/2) * x - 3 / x

theorem abscissa_of_tangent_point_is_2 : 
  ∃ x0 : ℝ, f' x0 = -1/2 ∧ x0 = 2 :=
by
  sorry

end abscissa_of_tangent_point_is_2_l265_265813


namespace frog_jumps_further_l265_265162

-- Given conditions
def grasshopper_jump : ℕ := 9 -- The grasshopper jumped 9 inches
def frog_jump : ℕ := 12 -- The frog jumped 12 inches

-- Proof statement
theorem frog_jumps_further : frog_jump - grasshopper_jump = 3 := by
  sorry

end frog_jumps_further_l265_265162


namespace convert_300_degree_to_radian_l265_265681

theorem convert_300_degree_to_radian : (300 : ℝ) * π / 180 = 5 * π / 3 :=
by
  sorry

end convert_300_degree_to_radian_l265_265681


namespace geometric_series_first_term_l265_265082

theorem geometric_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) (h_ratio : r = 1/4) (h_sum : S = 80) (h_series : S = a / (1 - r)) :
  a = 60 :=
by
  sorry

end geometric_series_first_term_l265_265082


namespace number_of_good_students_is_5_or_7_l265_265353

-- Definitions based on the conditions
def total_students : ℕ := 25
def number_of_good_students (G : ℕ) (T : ℕ) := G + T = total_students
def first_group_condition (T : ℕ) := T > 12
def second_group_condition (G : ℕ) (T : ℕ) := T = 3 * (G - 1)

-- Problem statement in Lean 4:
theorem number_of_good_students_is_5_or_7 (G T : ℕ) :
  number_of_good_students G T ∧ first_group_condition T ∧ second_group_condition G T → G = 5 ∨ G = 7 :=
by
  sorry

end number_of_good_students_is_5_or_7_l265_265353


namespace speed_calculation_l265_265798

def distance := 600 -- in meters
def time := 2 -- in minutes

def distance_km := distance / 1000 -- converting meters to kilometers
def time_hr := time / 60 -- converting minutes to hours

theorem speed_calculation : (distance_km / time_hr = 18) :=
 by
  sorry

end speed_calculation_l265_265798


namespace exists_geom_prog_first_100_integers_and_not_after_l265_265258

/-- 
  There exists an increasing geometric progression where the first 100 terms 
  are integers, but all subsequent terms are not integers.
-/
theorem exists_geom_prog_first_100_integers_and_not_after :
  ∃ (a₁ : ℕ) (q : ℚ), (∀ n ≤ 100, (a₁ * q ^ (n - 1)) ∈ ℤ) ∧ 
                       (∀ n > 100, ¬ (a₁ * q ^ (n - 1)) ∈ ℤ) ∧ 
                       (∀ m₁ m₂, m₁ < m₂ → a₁ * q ^ (m₁ - 1) < a₁ * q ^ (m₂ - 1)) := 
begin
  let a₁ := (2 : ℕ) ^ 99,
  let q : ℚ := 3 / 2,
  have h1 : ∀ n ≤ 100, ((a₁ * q ^ (n - 1)) ∈ ℤ),
  { intros n hn,
    have h2 : a₁ * q ^ (n - 1) = 2 ^ (100 - n) * 3 ^ (n - 1),
    { rw [mul_comm, ← pow_sub _ _ nat.le_of_lt_succ hn], 
      field_simp, ring },
    exact_mod_cast h2.symm },
  have h3 : ∀ n > 100, ¬ (a₁ * q ^ (n - 1) ∈ ℤ),
  { intros n hn,
    have h2 : a₁ * q ^ (n - 1) = 3 ^ (n - 1) / 2 ^ (n - 100),
    { 
      rw [mul_comm, ← pow_sub],
      field_simp, ring,
      exact nat.sub_pos_of_lt hn, }, 
    rw [h2],
    intro h4,
    have : (3 ^ (n - 1) : ℚ) = (2 ^ (n - 100) : ℚ) * (k : ℚ),
    { exact_mod_cast h4 },
    have h5 : 2 ∣ 3 ^ (n - 1),
    { norm_num at this, },
    have h6 := nat.prime_dvd_prime_pow nat.prime_two,
    exact nat.not_prime_three.
  },
  have h7: ∀ m₁ m₂, m₁ < m₂ → a₁ * q ^ (m₁ - 1) < a₁ * q ^ (m₂ - 1),
  { 
    intros m₁ m₂ hm,
    suffices h : q > 0, { exact mul_pos a₁.mpr (pow_pos h _), },
    norm_num.
  },

  exact ⟨a₁, q, h1, h3, h7⟩,
end

end exists_geom_prog_first_100_integers_and_not_after_l265_265258


namespace greatest_two_digit_multiple_of_17_is_85_l265_265209

theorem greatest_two_digit_multiple_of_17_is_85 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ m % 17 = 0) → m ≤ n) ∧ n = 85 :=
sorry

end greatest_two_digit_multiple_of_17_is_85_l265_265209


namespace volume_set_points_sum_l265_265472

-- Defining the problem conditions
def rectangular_parallelepiped_length : ℝ := 5
def rectangular_parallelepiped_width : ℝ := 6
def rectangular_parallelepiped_height : ℝ := 7
def unit_extension : ℝ := 1

-- Defining what we need to prove
theorem volume_set_points_sum :
  let V_box : ℝ := rectangular_parallelepiped_length * rectangular_parallelepiped_width * rectangular_parallelepiped_height
  let V_ext : ℝ := 2 * (unit_extension * rectangular_parallelepiped_length * rectangular_parallelepiped_width 
                  + unit_extension * rectangular_parallelepiped_length * rectangular_parallelepiped_height 
                  + unit_extension * rectangular_parallelepiped_width * rectangular_parallelepiped_height)
  let V_cyl : ℝ := 18 * π
  let V_sph : ℝ := (4 / 3) * π
  let V_total : ℝ := V_box + V_ext + V_cyl + V_sph
  let m : ℕ := 1272
  let n : ℕ := 58
  let p : ℕ := 3
  V_total = (m : ℝ) + (n : ℝ) * π / (p : ℝ) ∧ (m + n + p = 1333)
  := by
  sorry

end volume_set_points_sum_l265_265472


namespace remainder_5n_div_3_l265_265770

theorem remainder_5n_div_3 (n : ℤ) (h : n % 3 = 2) : (5 * n) % 3 = 1 := by
  sorry

end remainder_5n_div_3_l265_265770


namespace greatest_two_digit_multiple_of_17_l265_265204

theorem greatest_two_digit_multiple_of_17 : ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (17 ∣ n) ∧ n = 85 :=
by {
    use 85,
    split,
    { exact ⟨le_of_eq rfl, lt_of_le_of_ne (nat.le_of_eq rfl) (ne.symm (dec_trivial))⟩ },
    split,
    { exact dvd.intro 5 rfl },
    { refl }
}

end greatest_two_digit_multiple_of_17_l265_265204


namespace magician_trick_l265_265659

def coin (a : Fin 11 → Bool) : Prop :=
  ∃ i : Fin 11, a i = a (i + 1) % 11

theorem magician_trick (a : Fin 11 → Bool) 
(H : ∃ i : Fin 11, a i = a (i + 1) % 11) : 
  ∃ j : Fin 11, j ≠ 0 ∧ a j = a 0 :=
sorry

end magician_trick_l265_265659


namespace complement_intersection_l265_265842

open Set

variable {R : Type} [LinearOrderedField R]

def P : Set R := {x | x^2 - 2*x ≥ 0}
def Q : Set R := {x | 1 < x ∧ x ≤ 3}

theorem complement_intersection : (compl P ∩ Q) = {x : R | 1 < x ∧ x < 2} := by
  sorry

end complement_intersection_l265_265842


namespace xy_problem_l265_265307

theorem xy_problem (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 :=
by
  sorry

end xy_problem_l265_265307


namespace josh_money_left_l265_265009

def initial_amount : ℝ := 9
def spent_on_drink : ℝ := 1.75
def spent_on_item : ℝ := 1.25

theorem josh_money_left : initial_amount - (spent_on_drink + spent_on_item) = 6 := by
  sorry

end josh_money_left_l265_265009


namespace find_number_of_good_students_l265_265329

variable {G T : ℕ}

def is_good_or_troublemaker (G T : ℕ) : Prop :=
  G + T = 25 ∧ 
  (∀ s ∈ finset.range 5, T > 12) ∧ 
  (∀ s ∈ finset.range 20, T = 3 * (G - 1))

-- There are 25 students in total. 
-- Each student is either a good student or a troublemaker.
-- Good students always tell the truth, while troublemakers always lie.
-- Five students make the statement: "If I transfer to another class, more than half of the remaining students will be troublemakers."
-- The remaining 20 students make the statement: "If I transfer to another class, the number of troublemakers among the remaining students will be three times the number of good students."

theorem find_number_of_good_students (G T : ℕ) (h : is_good_or_troublemaker G T) :
  (G = 5 ∨ G = 7) :=
sorry

end find_number_of_good_students_l265_265329


namespace f_one_f_a_f_f_a_l265_265496

noncomputable def f (x : ℝ) : ℝ := 2 * x + 3

theorem f_one : f 1 = 5 := by
  sorry

theorem f_a (a : ℝ) : f a = 2 * a + 3 := by
  sorry

theorem f_f_a (a : ℝ) : f (f a) = 4 * a + 9 := by
  sorry

end f_one_f_a_f_f_a_l265_265496


namespace smallest_value_a2_b2_c2_l265_265728

theorem smallest_value_a2_b2_c2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 2 * a + 3 * b + 4 * c = 120) : 
  a^2 + b^2 + c^2 ≥ 14400 / 29 :=
by sorry

end smallest_value_a2_b2_c2_l265_265728


namespace perimeter_of_triangle_eq_28_l265_265895

-- Definitions of conditions
variables (p : ℝ)
def inradius : ℝ := 2.0
def area : ℝ := 28

-- Main theorem statement
theorem perimeter_of_triangle_eq_28 : p = 28 :=
  by
  -- The proof is omitted
  sorry

end perimeter_of_triangle_eq_28_l265_265895


namespace roots_numerically_equal_opposite_signs_l265_265706

theorem roots_numerically_equal_opposite_signs
  (a b c : ℝ) (k : ℝ)
  (h : (∃ x : ℝ, x^2 - (b+1) * x ≠ 0) →
    ∃ x : ℝ, x ≠ 0 ∧ x ∈ {x : ℝ | (k+2)*(x^2 - (b+1)*x) = (k-2)*((a+1)*x - c)} ∧ -x ∈ {x : ℝ | (k+2)*(x^2 - (b+1)*x) = (k-2)*((a+1)*x - c)}) :
  k = (-2 * (b - a)) / (b + a + 2) :=
by
  sorry

end roots_numerically_equal_opposite_signs_l265_265706


namespace remainder_8437_by_9_l265_265769

theorem remainder_8437_by_9 : 8437 % 9 = 4 :=
by
  -- proof goes here
  sorry

end remainder_8437_by_9_l265_265769


namespace modulus_of_complex_raised_to_eight_l265_265689

-- Define the complex number 2 + i in Lean
def z : Complex := Complex.mk 2 1

-- State the proof problem with conditions
theorem modulus_of_complex_raised_to_eight : Complex.abs (z ^ 8) = 625 := by
  sorry

end modulus_of_complex_raised_to_eight_l265_265689


namespace sum_arithmetic_sequence_l265_265980

theorem sum_arithmetic_sequence : ∀ (a d l : ℕ), 
  (d = 2) → (a = 2) → (l = 20) → 
  ∃ (n : ℕ), (l = a + (n - 1) * d) ∧ 
  (∑ k in Finset.range n, (a + k * d)) = 110 :=
by
  intros a d l h_d h_a h_l
  use 10
  split
  · sorry
  · sorry

end sum_arithmetic_sequence_l265_265980


namespace arithmetic_common_difference_l265_265585

-- Defining the arithmetic sequence sum function S
def S (a₁ a₂ a₃ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if n = 3 then a₁ + a₂ + a₃
  else if n = 2 then a₁ + a₂
  else 0

-- The Lean statement for the problem
theorem arithmetic_common_difference (a₁ a₂ a₃ d : ℝ) 
  (h1 : a₂ = a₁ + d) 
  (h2 : a₃ = a₂ + d)
  (h3 : 2 * S a₁ a₂ a₃ d 3 = 3 * S a₁ a₂ a₃ d 2 + 6) :
  d = 2 :=
by
  sorry

end arithmetic_common_difference_l265_265585


namespace remainder_when_divided_by_23_l265_265771

theorem remainder_when_divided_by_23 (y : ℕ) (h : y % 276 = 42) : y % 23 = 19 := by
  sorry

end remainder_when_divided_by_23_l265_265771


namespace green_pen_count_l265_265857

theorem green_pen_count 
  (blue_pens green_pens : ℕ)
  (h_ratio : blue_pens = 5 * green_pens / 3)
  (h_blue_pens : blue_pens = 20)
  : green_pens = 12 :=
by
  sorry

end green_pen_count_l265_265857


namespace wall_building_time_l265_265886

theorem wall_building_time (n t : ℕ) (h1 : n * t = 48) (h2 : n = 4) : t = 12 :=
by
  -- appropriate proof steps would go here
  sorry

end wall_building_time_l265_265886


namespace toaster_sales_promotion_l265_265810

theorem toaster_sales_promotion :
  ∀ (p : ℕ) (c₁ c₂ : ℕ) (k : ℕ), 
    (c₁ = 600 ∧ p = 15 ∧ k = p * c₁) ∧ 
    (c₂ = 450 ∧ (p * c₂ = k) ) ∧ 
    (p' = p * 11 / 10) →
    p' = 22 :=
by 
  sorry

end toaster_sales_promotion_l265_265810


namespace arithmetic_sequence_common_difference_l265_265560

theorem arithmetic_sequence_common_difference (a_1 d : ℝ) (S : ℕ → ℝ) 
    (h1 : S 2 = 2 * a_1 + d)
    (h2 : S 3 = 3 * a_1 + 3 * d)
    (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 := 
by
  sorry

end arithmetic_sequence_common_difference_l265_265560


namespace range_of_a_l265_265282

noncomputable def piecewiseFunc (x : ℝ) (a : ℝ) : ℝ :=
  if x < Real.exp 1 then -x^3 + x^2 else a * Real.log x

theorem range_of_a :
  ∃a : ℝ, (∃ P Q : ℝ × ℝ,
    let O := (0, 0) in
    P.1 * Q.1 = 0 ∧
    (-P.1^2 + piecewiseFunc P.1 a * (Q.1^3 + Q.1^2)) = 0 ∧
    ((P.1 + Q.1) / 2 = 0)) ↔ (0 < a ∧ a ≤ 1 / (Real.exp 1 + 1)) :=
sorry

end range_of_a_l265_265282


namespace composite_number_l265_265743

theorem composite_number (n : ℕ) (h : n ≥ 1) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = (10 ^ n + 1) * (10 ^ (n + 1) - 1) / 9 :=
by sorry

end composite_number_l265_265743


namespace triangle_perimeter_sqrt_l265_265423

theorem triangle_perimeter_sqrt :
  let a := Real.sqrt 8
  let b := Real.sqrt 18
  let c := Real.sqrt 32
  a + b + c = 9 * Real.sqrt 2 :=
by
  sorry

end triangle_perimeter_sqrt_l265_265423


namespace largest_integer_of_four_l265_265747

theorem largest_integer_of_four (A B C D : ℤ)
  (h_diff: A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_order: A < B ∧ B < C ∧ C < D)
  (h_avg: (A + B + C + D) / 4 = 74)
  (h_A_min: A ≥ 29) : D = 206 :=
by
  sorry

end largest_integer_of_four_l265_265747


namespace probability_product_positive_is_5_div_9_l265_265926

noncomputable def probability_positive_product : ℚ :=
  let interval := Set.Icc (-30 : ℝ) 15
  let length_interval := 45
  let length_neg := 30
  let length_pos := 15
  let prob_neg := (length_neg : ℚ) / length_interval
  let prob_pos := (length_pos : ℚ) / length_interval
  let prob_product_pos := prob_neg^2 + prob_pos^2
  prob_product_pos

theorem probability_product_positive_is_5_div_9 :
  probability_positive_product = 5 / 9 :=
by
  sorry

end probability_product_positive_is_5_div_9_l265_265926


namespace penny_purchase_exceeded_minimum_spend_l265_265001

theorem penny_purchase_exceeded_minimum_spend :
  let bulk_price_per_pound := 5
  let minimum_spend := 40
  let tax_per_pound := 1
  let total_paid := 240
  let total_cost_per_pound := bulk_price_per_pound + tax_per_pound
  let pounds_purchased := total_paid / total_cost_per_pound
  let minimum_pounds_to_spend := minimum_spend / bulk_price_per_pound
  pounds_purchased - minimum_pounds_to_spend = 32 :=
by
  -- The proof is omitted here as per the instructions.
  sorry

end penny_purchase_exceeded_minimum_spend_l265_265001


namespace class_proof_l265_265341

def class_problem : Prop :=
  let total_students := 25
  let (students : Type) := {s // s ∈ Fin total_students}
  let good students (s : students) : Prop := sorry -- Condition Placeholder
  let troublemakers (s : students) : Prop := sorry -- Condition Placeholder
  have all_good_or_trouble : ∀ s, good students s ∨ troublemakers s,
    from sorry,
  have lh_trouble : ∀ (a b c d e : students), (good students a ∧ good students b ∧ good students c ∧ good students d ∧ good students e) →
    ∃ (x : ℕ), x > ((total_students - 1) / 2) ∧
    (λ rem_trouble : students - {a, b, c, d, e}, B : ℕ, B = x) ∧
    (λ rem_good : students - {a, b, c, d, e}, G : ℕ, G ∈ rem_good), from sorry,
  have th_triple_ratio : ∀ s, (students - {s}).card = 24 →
    ∃ (x y : ℕ), (troublemakers x ∧ good students y) ∧ (3*y = x), from sorry,
  ∃ (num_good_students : ℕ), (num_good_students = 5 ∨ num_good_students = 7)
  
theorem class_proof : class_problem :=
  sorry

end class_proof_l265_265341


namespace problem1_problem2_l265_265779

-- Problem 1: (-3xy)² * 4x² = 36x⁴y²
theorem problem1 (x y : ℝ) : ((-3 * x * y) ^ 2) * (4 * x ^ 2) = 36 * x ^ 4 * y ^ 2 := by
  sorry

-- Problem 2: (x + 2)(2x - 3) = 2x² + x - 6
theorem problem2 (x : ℝ) : (x + 2) * (2 * x - 3) = 2 * x ^ 2 + x - 6 := by
  sorry

end problem1_problem2_l265_265779


namespace common_difference_of_arithmetic_sequence_l265_265571

variable (a1 d : ℤ)
def S : ℕ → ℤ
| 0     => 0
| (n+1) => S n + (a1 + n * d)

theorem common_difference_of_arithmetic_sequence
  (h : 2 * S a1 d 3 = 3 * S a1 d 2 + 6) :
  d = 2 :=
  sorry

end common_difference_of_arithmetic_sequence_l265_265571


namespace trees_after_planting_l265_265729

variable (x : ℕ)

theorem trees_after_planting (x : ℕ) : 
  let additional_trees := 12
  let days := 12 / 2
  let trees_removed := days * 3
  x + additional_trees - trees_removed = x - 6 :=
by
  let additional_trees := 12
  let days := 12 / 2
  let trees_removed := days * 3
  sorry

end trees_after_planting_l265_265729


namespace find_common_difference_l265_265580

-- Define the arithmetic sequence and the sum of the first n terms
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
def sum_of_first_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ := ∑ k in finset.range n, arithmetic_sequence a₁ d (k + 1)

-- Given condition
def condition (a₁ d : ℝ) : Prop := 
  2 * sum_of_first_n_terms a₁ d 3 = 3 * sum_of_first_n_terms a₁ d 2 + 6

-- The proof statement
theorem find_common_difference (a₁ d : ℝ) (h : condition a₁ d) : d = 2 :=
by
  sorry

end find_common_difference_l265_265580


namespace buying_ways_l265_265906

theorem buying_ways (students : ℕ) (choices : ℕ) (at_least_one_pencil : ℕ) : 
  students = 4 ∧ choices = 2 ∧ at_least_one_pencil = 1 → 
  (choices^students - 1) = 15 :=
by
  sorry

end buying_ways_l265_265906


namespace first_complete_row_cover_l265_265079

def is_shaded_square (n : ℕ) : ℕ := n ^ 2

def row_number (square_number : ℕ) : ℕ :=
  (square_number + 9) / 10 -- ceiling of square_number / 10

theorem first_complete_row_cover : ∃ n, ∀ r : ℕ, 1 ≤ r ∧ r ≤ 10 → ∃ k : ℕ, is_shaded_square k ≤ n ∧ row_number (is_shaded_square k) = r :=
by
  use 100
  intros r h
  sorry

end first_complete_row_cover_l265_265079


namespace product_of_random_numbers_greater_zero_l265_265936

noncomputable def random_product_positive_probability : ℝ := 
  let interval_length := 45
  let neg_interval_length := 30
  let pos_interval_length := 15
  let prob_neg := (neg_interval_length : ℝ) / interval_length
  let prob_pos := (pos_interval_length : ℝ) / interval_length
  prob_pos * prob_pos + prob_neg * prob_neg

-- Prove that the probability that the product of two randomly selected numbers
-- from the interval [-30, 15] is greater than zero is 5/9.
theorem product_of_random_numbers_greater_zero : 
  random_product_positive_probability = 5 / 9 := by
  sorry

end product_of_random_numbers_greater_zero_l265_265936


namespace probability_of_three_black_balls_l265_265068

def total_ball_count : ℕ := 4 + 8

def white_ball_count : ℕ := 4

def black_ball_count : ℕ := 8

def total_combinations : ℕ := Nat.choose total_ball_count 3

def black_combinations : ℕ := Nat.choose black_ball_count 3

def probability_three_black : ℚ := black_combinations / total_combinations

theorem probability_of_three_black_balls : 
  probability_three_black = 14 / 55 := 
sorry

end probability_of_three_black_balls_l265_265068


namespace average_of_tenths_and_thousandths_l265_265947

theorem average_of_tenths_and_thousandths :
  (0.4 + 0.005) / 2 = 0.2025 :=
by
  -- We skip the proof here
  sorry

end average_of_tenths_and_thousandths_l265_265947


namespace cups_per_girl_l265_265425

noncomputable def numStudents := 30
noncomputable def numBoys := 10
noncomputable def numCupsByBoys := numBoys * 5
noncomputable def totalCups := 90
noncomputable def numGirls := 2 * numBoys
noncomputable def numCupsByGirls := totalCups - numCupsByBoys

theorem cups_per_girl : (numCupsByGirls / numGirls) = 2 := by
  sorry

end cups_per_girl_l265_265425


namespace proof_problem_l265_265274

theorem proof_problem :
  (∃ x : ℝ, x - 1 ≥ Real.log x) ∧ (¬ ∀ x ∈ Ioo 0 Real.pi, Real.sin x + 1 / Real.sin x > 2) := sorry

end proof_problem_l265_265274


namespace cary_net_calorie_deficit_is_250_l265_265989

-- Define the conditions
def miles_walked : ℕ := 3
def candy_bar_calories : ℕ := 200
def calories_per_mile : ℕ := 150

-- Define the function to calculate total calories burned
def total_calories_burned (miles : ℕ) (calories_per_mile : ℕ) : ℕ :=
  miles * calories_per_mile

-- Define the function to calculate net calorie deficit
def net_calorie_deficit (total_calories : ℕ) (candy_calories : ℕ) : ℕ :=
  total_calories - candy_calories

-- The statement to be proven
theorem cary_net_calorie_deficit_is_250 :
  net_calorie_deficit (total_calories_burned miles_walked calories_per_mile) candy_bar_calories = 250 :=
  by sorry

end cary_net_calorie_deficit_is_250_l265_265989


namespace geom_seq_sum_problem_l265_265720

noncomputable def geom_sum_first_n_terms (a₁ q : ℕ) (n : ℕ) : ℕ :=
  a₁ * (1 - q^n) / (1 - q)

noncomputable def geom_sum_specific_terms (a₃ q : ℕ) (n m : ℕ) : ℕ :=
  a₃ * ((1 - (q^m) ^ n) / (1 - q^m))

theorem geom_seq_sum_problem :
  ∀ (a₁ q S₈₇ : ℕ),
  q = 2 →
  S₈₇ = 140 →
  geom_sum_first_n_terms a₁ q 87 = S₈₇ →
  ∃ a₃, a₃ = ((q * q) * a₁) →
  geom_sum_specific_terms a₃ q 29 3 = 80 := 
by
  intros a₁ q S₈₇ hq₁ hS₈₇ hsum
  -- Further proof would go here
  sorry

end geom_seq_sum_problem_l265_265720


namespace mary_initial_nickels_l265_265025

variable {x : ℕ}

theorem mary_initial_nickels (h : x + 5 = 12) : x = 7 := by
  sorry

end mary_initial_nickels_l265_265025


namespace number_of_good_students_is_5_or_7_l265_265356

-- Definitions based on the conditions
def total_students : ℕ := 25
def number_of_good_students (G : ℕ) (T : ℕ) := G + T = total_students
def first_group_condition (T : ℕ) := T > 12
def second_group_condition (G : ℕ) (T : ℕ) := T = 3 * (G - 1)

-- Problem statement in Lean 4:
theorem number_of_good_students_is_5_or_7 (G T : ℕ) :
  number_of_good_students G T ∧ first_group_condition T ∧ second_group_condition G T → G = 5 ∨ G = 7 :=
by
  sorry

end number_of_good_students_is_5_or_7_l265_265356


namespace greatest_two_digit_multiple_of_17_l265_265188

-- Define the range of two-digit numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate for being a multiple of 17
def is_multiple_of_17 (n : ℕ) := ∃ k : ℕ, n = 17 * k

-- The specific problem to prove
theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n ∈ two_digit_numbers ∧ is_multiple_of_17(n) ∧ 
  (∀ m : ℕ, m ∈ two_digit_numbers ∧ is_multiple_of_17(m) → n ≥ m) ∧ n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l265_265188


namespace arithmetic_sequence_a6_l265_265279

theorem arithmetic_sequence_a6 {a : ℕ → ℤ}
  (h1 : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m)
  (h2 : a 2 + a 8 = 16)
  (h3 : a 4 = 6) :
  a 6 = 10 :=
by
  sorry

end arithmetic_sequence_a6_l265_265279


namespace min_value_expression_l265_265112

theorem min_value_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 1) :
  3 ≤ (1 / (x + y)) + ((x + y) / z) :=
by
  sorry

end min_value_expression_l265_265112


namespace multiply_3_6_and_0_3_l265_265998

theorem multiply_3_6_and_0_3 : 3.6 * 0.3 = 1.08 :=
by
  sorry

end multiply_3_6_and_0_3_l265_265998


namespace calculate_g_inv_l265_265135

noncomputable def g : ℤ → ℤ := sorry
noncomputable def g_inv : ℤ → ℤ := sorry

axiom g_inv_eq : ∀ x, g (g_inv x) = x

axiom cond1 : g (-1) = 2
axiom cond2 : g (0) = 3
axiom cond3 : g (1) = 6

theorem calculate_g_inv : 
  g_inv (g_inv 6 - g_inv 2) = -1 := 
by
  -- The proof goes here
  sorry

end calculate_g_inv_l265_265135


namespace int_to_fourth_power_l265_265118

theorem int_to_fourth_power:
  3^4 * 9^8 = 243^4 :=
by 
  sorry

end int_to_fourth_power_l265_265118


namespace quadratic_inequality_false_iff_l265_265757

theorem quadratic_inequality_false_iff (a : ℝ) :
  (¬ ∃ x : ℝ, 2 * x^2 - 3 * a * x + 9 < 0) ↔ (-2 * Real.sqrt 2 ≤ a ∧ a ≤ 2 * Real.sqrt 2) :=
by sorry

end quadratic_inequality_false_iff_l265_265757


namespace greatest_two_digit_multiple_of_17_l265_265187

-- Define the range of two-digit numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate for being a multiple of 17
def is_multiple_of_17 (n : ℕ) := ∃ k : ℕ, n = 17 * k

-- The specific problem to prove
theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n ∈ two_digit_numbers ∧ is_multiple_of_17(n) ∧ 
  (∀ m : ℕ, m ∈ two_digit_numbers ∧ is_multiple_of_17(m) → n ≥ m) ∧ n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l265_265187


namespace magician_assistant_trick_l265_265653

/-- A coin can be either heads or tails. -/
inductive Coin
| heads : Coin
| tails : Coin

/-- Given a cyclic arrangement of 11 coins, there exists at least one pair of adjacent coins with the same face. -/
theorem magician_assistant_trick (coins : Fin 11 → Coin) : 
  ∃ i : Fin 11, coins i = coins (i + 1) := 
by
  sorry

end magician_assistant_trick_l265_265653


namespace first_positive_term_is_7_l265_265838

-- Define the conditions and the sequence
def a1 : ℚ := -1
def d : ℚ := 1 / 5

-- Define the general term of the sequence
def a_n (n : ℕ) : ℚ := a1 + (n - 1) * d

-- Define the proposition that the 7th term is the first positive term
theorem first_positive_term_is_7 :
  ∀ n : ℕ, (0 < a_n n) → (7 <= n) :=
by
  intro n h
  sorry

end first_positive_term_is_7_l265_265838


namespace molecular_weight_correct_l265_265216

-- Define atomic weights
def atomic_weight_aluminium : Float := 26.98
def atomic_weight_oxygen : Float := 16.00
def atomic_weight_hydrogen : Float := 1.01
def atomic_weight_silicon : Float := 28.09
def atomic_weight_nitrogen : Float := 14.01

-- Define the number of each atom in the compound
def num_aluminium : Nat := 2
def num_oxygen : Nat := 6
def num_hydrogen : Nat := 3
def num_silicon : Nat := 2
def num_nitrogen : Nat := 4

-- Calculate the expected molecular weight
def expected_molecular_weight : Float :=
  (2 * atomic_weight_aluminium) + 
  (6 * atomic_weight_oxygen) + 
  (3 * atomic_weight_hydrogen) + 
  (2 * atomic_weight_silicon) + 
  (4 * atomic_weight_nitrogen)

-- Prove that the expected molecular weight is 265.21 amu
theorem molecular_weight_correct : expected_molecular_weight = 265.21 :=
by
  sorry

end molecular_weight_correct_l265_265216


namespace males_watch_tvxy_l265_265246

-- Defining the conditions
def total_watch := 160
def females_watch := 75
def males_dont_watch := 83
def total_dont_watch := 120

-- Proving that the number of males who watch TVXY equals 85
theorem males_watch_tvxy : (total_watch - females_watch) = 85 :=
by sorry

end males_watch_tvxy_l265_265246


namespace arithmetic_sequence_sum_l265_265420

theorem arithmetic_sequence_sum (c d : ℕ) 
  (h1 : ∀ (a1 a2 a3 a4 a5 a6 : ℕ), a1 = 3 → a2 = 10 → a3 = 17 → a6 = 38 → (a2 - a1 = a3 - a2) → (a3 - a2 = c - a3) → (c - a3 = d - c) → (d - c = a6 - d)) : 
  c + d = 55 := 
by 
  sorry

end arithmetic_sequence_sum_l265_265420


namespace common_difference_is_two_l265_265589

variable {a₁ a₂ a₃ S₃ S₂ : ℕ}
variable (d : ℕ)

-- Given condition
axiom H : 2 * S₃ = 3 * S₂ + 6

-- Definitions based on arithmetic sequence properties
def S₂ := a₁ + a₂
def S₃ := a₁ + a₂ + a₃
def a₂ := a₁ + d
def a₃ := a₁ + 2 * d

theorem common_difference_is_two : d = 2 := 
by 
  sorry

end common_difference_is_two_l265_265589


namespace arithmetic_sequence_common_difference_l265_265539

variable {a₁ d : ℕ}
variable S : ℕ → ℕ

-- Definitions of the sums S₂ and S₃ in an arithmetic sequence
def S₂ : ℕ := a₁ + (a₁ + d)
def S₃ : ℕ := a₁ + (a₁ + d) + (a₁ + 2 * d)

theorem arithmetic_sequence_common_difference (h : 2 * S₃ = 3 * S₂ + 6) : d = 2 :=
by
  -- Skip the proof.
  sorry

end arithmetic_sequence_common_difference_l265_265539


namespace water_parts_in_solution_l265_265665

def lemonade_syrup_parts : ℝ := 7
def target_percentage : ℝ := 0.30
def adjusted_parts : ℝ := 2.1428571428571423

-- Original equation: L = 0.30 * (L + W)
-- Substitute L = 7 for the particular instance.
-- Therefore, 7 = 0.30 * (7 + W)

theorem water_parts_in_solution (W : ℝ) : 
  (7 = 0.30 * (7 + W)) → 
  W = 16.333333333333332 := 
by
  sorry

end water_parts_in_solution_l265_265665


namespace greatest_two_digit_multiple_of_17_l265_265183

theorem greatest_two_digit_multiple_of_17 : ∃ x : ℕ, x < 100 ∧ x ≥ 10 ∧ x % 17 = 0 ∧ ∀ y : ℕ, y < 100 ∧ y ≥ 10 ∧ y % 17 = 0 → y ≤ x :=
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l265_265183


namespace main_theorem_l265_265091

noncomputable def phase_trajectories_are_spirals
  (x : ℝ → ℝ)
  (v : ℝ → ℝ := λ t, deriv x t)
  (d2x_dt2 : ℝ → ℝ := λ t, deriv (deriv x) t) :
  Prop :=
  ∀ t : ℝ,
    d2x_dt2 t - v t + x t = 0 →
    ∃ k : ℝ, v t = x t / (1 - k)

theorem main_theorem {x : ℝ → ℝ}
  (h : phase_trajectories_are_spirals x) :
  ∀ t : ℝ, ∃ k : ℝ, (deriv x t) = x t / (1 - k) :=
sorry

end main_theorem_l265_265091


namespace dress_designs_count_l265_265646

theorem dress_designs_count :
  let colors := 5
  let patterns := 4
  let sizes := 3
  colors * patterns * sizes = 60 :=
by
  let colors := 5
  let patterns := 4
  let sizes := 3
  have h : colors * patterns * sizes = 60 := by norm_num
  exact h

end dress_designs_count_l265_265646


namespace recurring_decimal_exceeds_by_fraction_l265_265975

theorem recurring_decimal_exceeds_by_fraction : 
  let y := (36 : ℚ) / 99
  let x := (36 : ℚ) / 100
  ((4 : ℚ) / 11) - x = (4 : ℚ) / 1100 :=
by
  sorry

end recurring_decimal_exceeds_by_fraction_l265_265975


namespace problem_part1_problem_part2_l265_265505

-- Definitions of the vectors
def a (x : ℝ) : ℝ × ℝ := (1, 2 * x)
def b (x : ℝ) : ℝ × ℝ := (x, 3)
def c : ℝ × ℝ := (-2, 0)

-- Definitions for vector operations
def add_vec (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def parallel (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1
def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

noncomputable def part1 (x : ℝ) : Prop := parallel (add_vec (a x) (scalar_mul 2 (b x))) (add_vec (scalar_mul 2 (a x)) (scalar_mul (-1) c))

noncomputable def part2 (x : ℝ) : Prop := perpendicular (add_vec (a x) (scalar_mul 2 (b x))) (add_vec (scalar_mul 2 (a x)) (scalar_mul (-1) c))

theorem problem_part1 : part1 2 ∧ part1 (-3 / 2) := sorry

theorem problem_part2 : part2 ((-4 + Real.sqrt 14) / 2) ∧ part2 ((-4 - Real.sqrt 14) / 2) := sorry

end problem_part1_problem_part2_l265_265505


namespace cows_and_goats_sum_l265_265371

theorem cows_and_goats_sum (x y z : ℕ) 
  (h1 : x + y + z = 12) 
  (h2 : 4 * x + 2 * y + 4 * z = 18 + 2 * (x + y + z)) 
  : x + z = 9 := by 
  sorry

end cows_and_goats_sum_l265_265371


namespace greatest_two_digit_multiple_of_17_l265_265189

-- Define the range of two-digit numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate for being a multiple of 17
def is_multiple_of_17 (n : ℕ) := ∃ k : ℕ, n = 17 * k

-- The specific problem to prove
theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n ∈ two_digit_numbers ∧ is_multiple_of_17(n) ∧ 
  (∀ m : ℕ, m ∈ two_digit_numbers ∧ is_multiple_of_17(m) → n ≥ m) ∧ n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l265_265189


namespace parametric_to_standard_l265_265048

theorem parametric_to_standard (t : ℝ) : 
  (x = (2 + 3 * t) / (1 + t)) ∧ (y = (1 - 2 * t) / (1 + t)) → (3 * x + y - 7 = 0) ∧ (x ≠ 3) := 
by 
  sorry

end parametric_to_standard_l265_265048


namespace arcsin_one_half_eq_pi_over_six_arccos_one_half_eq_pi_over_three_l265_265992

theorem arcsin_one_half_eq_pi_over_six : Real.arcsin (1/2) = Real.pi/6 :=
by 
  sorry

theorem arccos_one_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi/3 :=
by 
  sorry

end arcsin_one_half_eq_pi_over_six_arccos_one_half_eq_pi_over_three_l265_265992


namespace greatest_two_digit_multiple_of_17_l265_265185

theorem greatest_two_digit_multiple_of_17 : ∃ x : ℕ, x < 100 ∧ x ≥ 10 ∧ x % 17 = 0 ∧ ∀ y : ℕ, y < 100 ∧ y ≥ 10 ∧ y % 17 = 0 → y ≤ x :=
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l265_265185


namespace max_value_of_expression_l265_265132

theorem max_value_of_expression (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + 2 * b + 3 * c = 1) :
    (1 / (2 * a + b) + 1 / (2 * b + c) + 1 / (2 * c + a) ≤ 7) :=
sorry

end max_value_of_expression_l265_265132


namespace find_value_l265_265839

theorem find_value
  (x a y b z c : ℝ)
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 3) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 16 :=
by 
  sorry

end find_value_l265_265839


namespace decagon_diagonals_intersection_probability_l265_265234

def isRegularDecagon : Prop :=
  ∃ decagon : ℕ, decagon = 10  -- A regular decagon has 10 sides

def chosen_diagonals (n : ℕ) : ℕ :=
  (Nat.choose n 2) - n   -- Number of diagonals in an n-sided polygon =
                          -- number of pairs of vertices - n sides

noncomputable def probability_intersection : ℚ :=
  let total_diagonals := chosen_diagonals 10
  let number_of_ways_to_pick_four := Nat.choose 10 4
  (number_of_ways_to_pick_four * 2) / (total_diagonals * (total_diagonals - 1) / 2)

theorem decagon_diagonals_intersection_probability :
  isRegularDecagon → probability_intersection = 42 / 119 :=
sorry

end decagon_diagonals_intersection_probability_l265_265234


namespace vasya_gift_ways_l265_265739

theorem vasya_gift_ways :
  let cars := 7
  let constructor_sets := 5
  (cars * constructor_sets) + (Nat.choose cars 2) + (Nat.choose constructor_sets 2) = 66 :=
by
  let cars := 7
  let constructor_sets := 5
  sorry

end vasya_gift_ways_l265_265739


namespace common_difference_of_arithmetic_sequence_l265_265534

noncomputable def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range n, a i

def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_cond : 2 * S a 3 = 3 * S a 2 + 6) :
  ∃ d : ℝ, d = 2 := sorry

end common_difference_of_arithmetic_sequence_l265_265534


namespace part1_part2_l265_265819

-- Assuming x is a real number
variable (x : ℝ) (a : ℝ)

theorem part1 : ∀ a : ℝ, (∀ x : ℝ, ¬ (| x - 4 | + | 3 - x | < a)) → a ≤ 1 :=
by sorry

theorem part2 : ∀ a : ℝ, (∃ x : ℝ, | x - 4 | + | 3 - x | < a) → a > 1 :=
by sorry

end part1_part2_l265_265819


namespace train_pass_bridge_in_36_seconds_l265_265223

def train_length : ℝ := 360 -- meters
def bridge_length : ℝ := 140 -- meters
def train_speed_kmh : ℝ := 50 -- km/h

noncomputable def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600) -- m/s
noncomputable def total_distance : ℝ := train_length + bridge_length -- meters
noncomputable def passing_time : ℝ := total_distance / train_speed_ms -- seconds

theorem train_pass_bridge_in_36_seconds :
  passing_time = 36 := 
sorry

end train_pass_bridge_in_36_seconds_l265_265223


namespace diagonals_intersection_probability_l265_265230

theorem diagonals_intersection_probability (decagon : Polygon) (h_regular : decagon.is_regular ∧ decagon.num_sides = 10) :
  probability_intersection_inside decagon = 42 / 119 := 
sorry

end diagonals_intersection_probability_l265_265230


namespace consecutive_sum_impossible_l265_265064

theorem consecutive_sum_impossible (n : ℕ) :
  (¬ (∃ (a b : ℕ), a < b ∧ n = (b - a + 1) * (a + b) / 2)) ↔ ∃ s : ℕ, n = 2 ^ s :=
sorry

end consecutive_sum_impossible_l265_265064


namespace greatest_two_digit_multiple_of_17_l265_265186

theorem greatest_two_digit_multiple_of_17 : ∃ x : ℕ, x < 100 ∧ x ≥ 10 ∧ x % 17 = 0 ∧ ∀ y : ℕ, y < 100 ∧ y ≥ 10 ∧ y % 17 = 0 → y ≤ x :=
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l265_265186


namespace problem_l265_265699

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (1 / 2) * a * x^2 - (x - 1) * Real.exp x

theorem problem (a : ℝ) :
  (∀ x1 x2 x3 : ℝ, 0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ x2 ∧ x2 ≤ 1 ∧ 0 ≤ x3 ∧ x3 ≤ 1 →
                  f a x1 + f a x2 ≥ f a x3) →
  1 ≤ a ∧ a ≤ 4 :=
sorry

end problem_l265_265699


namespace charley_pencils_lost_l265_265991

theorem charley_pencils_lost :
  ∃ x : ℕ, (30 - x - (1/3 : ℝ) * (30 - x) = 16) ∧ x = 6 :=
by
  -- Since x must be an integer and the equations naturally produce whole numbers,
  -- we work within the context of natural numbers, then cast to real as needed.
  use 6
  -- Express the main condition in terms of x
  have h: (30 - 6 - (1/3 : ℝ) * (30 - 6) = 16) := by sorry
  exact ⟨h, rfl⟩

end charley_pencils_lost_l265_265991


namespace range_of_g_l265_265099

noncomputable def g (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctan (2 * x)

theorem range_of_g : ∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), 
  ∃ r ∈ set.Icc (Real.pi / 2 - Real.arctan 2) (Real.pi / 2 + Real.arctan 2), g x = r :=
by
  sorry

end range_of_g_l265_265099


namespace part_I_5_continuous_part_I_6_not_continuous_part_II_min_k_for_8_continuous_part_III_min_k_for_20_continuous_l265_265266

def is_continuous_representable (m : ℕ) (Q : List ℤ) : Prop :=
  ∀ n ∈ (List.range (m + 1)).tail, ∃ (sublist : List ℤ), sublist ≠ [] ∧ sublist ∈ Q.sublists' ∧ sublist.sum = n

theorem part_I_5_continuous :
  is_continuous_representable 5 [2, 1, 4] :=
sorry

theorem part_I_6_not_continuous :
  ¬is_continuous_representable 6 [2, 1, 4] :=
sorry

theorem part_II_min_k_for_8_continuous (Q : List ℤ) :
  is_continuous_representable 8 Q → Q.length ≥ 4 :=
sorry

theorem part_III_min_k_for_20_continuous (Q : List ℤ) 
  (h : is_continuous_representable 20 Q) (h_sum : Q.sum < 20) :
  Q.length ≥ 7 :=
sorry

end part_I_5_continuous_part_I_6_not_continuous_part_II_min_k_for_8_continuous_part_III_min_k_for_20_continuous_l265_265266


namespace prove_common_difference_l265_265572

variable {a1 d : ℤ}
variable (S : ℕ → ℤ)
variable (arith_seq : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
def sum_first_n (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Define the condition given in the problem
def problem_condition : Prop := 2 * sum_first_n 3 = 3 * sum_first_n 2 + 6

-- Define that the common difference d is 2
def d_value_is_2 : Prop := d = 2

theorem prove_common_difference (h : problem_condition) : d_value_is_2 :=
by
  sorry

end prove_common_difference_l265_265572


namespace inverse_sum_l265_265870

def f (x : ℝ) : ℝ := x * |x|

theorem inverse_sum (h1 : ∃ x : ℝ, f x = 9) (h2 : ∃ x : ℝ, f x = -81) :
  ∃ a b: ℝ, f a = 9 ∧ f b = -81 ∧ a + b = -6 :=
by
  sorry

end inverse_sum_l265_265870


namespace initial_concentration_alcohol_l265_265454

theorem initial_concentration_alcohol (x : ℝ) 
    (h1 : 0 ≤ x ∧ x ≤ 100)
    (h2 : 0.44 * 10 = (x / 100) * 2 + 3.6) :
    x = 40 :=
sorry

end initial_concentration_alcohol_l265_265454


namespace last_10_digits_repeat_periodically_l265_265680

theorem last_10_digits_repeat_periodically :
  ∃ (p : ℕ) (n₀ : ℕ), p = 4 * 10^9 ∧ n₀ = 10 ∧ 
  ∀ n, (2^(n + p) % 10^10 = 2^n % 10^10) :=
by sorry

end last_10_digits_repeat_periodically_l265_265680


namespace father_age_l265_265439

theorem father_age (M F : ℕ) 
  (h1 : M = 2 * F / 5) 
  (h2 : M + 10 = (F + 10) / 2) : F = 50 :=
sorry

end father_age_l265_265439


namespace algebraic_expression_l265_265096

-- Definition for the problem expressed in Lean
def number_one_less_than_three_times (a : ℝ) : ℝ :=
  3 * a - 1

-- Theorem stating the proof problem
theorem algebraic_expression (a : ℝ) : number_one_less_than_three_times a = 3 * a - 1 :=
by
  -- Proof steps would go here; omitted as per instructions
  sorry

end algebraic_expression_l265_265096


namespace no_solution_range_has_solution_range_l265_265820

open Real

theorem no_solution_range (a : ℝ) : (∀ x, ¬ (|x - 4| + |3 - x| < a)) ↔ a ≤ 1 := 
sorry

theorem has_solution_range (a : ℝ) : (∃ x, |x - 4| + |3 - x| < a) ↔ 1 < a :=
sorry

end no_solution_range_has_solution_range_l265_265820


namespace max_value_expression_l265_265013

noncomputable def a (φ : ℝ) : ℝ := 3 * Real.cos φ
noncomputable def b (φ : ℝ) : ℝ := 3 * Real.sin φ

theorem max_value_expression (φ θ : ℝ) : 
  ∃ c : ℝ, c = 3 * Real.cos (θ - φ) ∧ c ≤ 3 := by
  sorry

end max_value_expression_l265_265013


namespace coordinates_of_E_l265_265373

theorem coordinates_of_E :
  let A := (-2, 1)
  let B := (1, 4)
  let C := (4, -3)
  let ratio_AB := (1, 2)
  let ratio_CE_ED := (1, 4)
  let D := ( (ratio_AB.1 * B.1 + ratio_AB.2 * A.1) / (ratio_AB.1 + ratio_AB.2),
             (ratio_AB.1 * B.2 + ratio_AB.2 * A.2) / (ratio_AB.1 + ratio_AB.2) )
  let E := ( (ratio_CE_ED.1 * C.1 - ratio_CE_ED.2 * D.1) / (ratio_CE_ED.1 - ratio_CE_ED.2),
             (ratio_CE_ED.1 * C.2 - ratio_CE_ED.2 * D.2) / (ratio_CE_ED.1 - ratio_CE_ED.2) )
  E = (-8 / 3, 11 / 3) := by
  sorry

end coordinates_of_E_l265_265373


namespace solve_for_z_l265_265887

theorem solve_for_z (z : ℂ) (i : ℂ) (h : i^2 = -1) : 3 + 2 * i * z = 5 - 3 * i * z → z = - (2 * i) / 5 :=
by
  intro h_equation
  -- Proof steps will be provided here.
  sorry

end solve_for_z_l265_265887


namespace lcm_inequality_l265_265393

theorem lcm_inequality (k m n : ℕ) (hk : 0 < k) (hm : 0 < m) (hn : 0 < n) : 
  Nat.lcm k m * Nat.lcm m n * Nat.lcm n k ≥ Nat.lcm (Nat.lcm k m) n ^ 2 :=
by sorry

end lcm_inequality_l265_265393


namespace total_time_is_correct_l265_265023

-- Definitions based on conditions
def timeouts_for_running : ℕ := 5
def timeouts_for_throwing_food : ℕ := 5 * timeouts_for_running - 1
def timeouts_for_swearing : ℕ := timeouts_for_throwing_food / 3

-- Definition for total time-outs
def total_timeouts : ℕ := timeouts_for_running + timeouts_for_throwing_food + timeouts_for_swearing
-- Each time-out is 5 minutes
def timeout_duration : ℕ := 5

-- Total time in minutes
def total_time_in_minutes : ℕ := total_timeouts * timeout_duration

-- The proof statement
theorem total_time_is_correct : total_time_in_minutes = 185 := by
  sorry

end total_time_is_correct_l265_265023


namespace probability_of_sequence_l265_265053

theorem probability_of_sequence :
  let total_cards := 52
  let face_cards := 12
  let hearts := 13
  let first_card_face_prob := (face_cards : ℝ) / total_cards
  let second_card_heart_prob := (10 : ℝ) / (total_cards - 1)
  let third_card_face_prob := (11 : ℝ) / (total_cards - 2)
  let total_prob := first_card_face_prob * second_card_heart_prob * third_card_face_prob
  total_prob = 1 / 100.455 :=
by
  sorry

end probability_of_sequence_l265_265053


namespace arithmetic_sequence_sum_l265_265418

theorem arithmetic_sequence_sum (c d : ℤ) (h₁ : d = 10 - 3)
    (h₂ : c = 17 + (10 - 3)) (h₃ : d = 24 + (10 - 3)) :
    c + d = 55 :=
sorry

end arithmetic_sequence_sum_l265_265418


namespace car_speed_l265_265644

variable (fuel_efficiency : ℝ) (fuel_decrease_gallons : ℝ) (time_hours : ℝ) 
          (gallons_to_liters : ℝ) (kilometers_to_miles : ℝ)
          (car_speed_mph : ℝ)

-- Conditions given in the problem
def fuelEfficiency : ℝ := 40 -- km per liter
def fuelDecreaseGallons : ℝ := 3.9 -- gallons
def timeHours : ℝ := 5.7 -- hours
def gallonsToLiters : ℝ := 3.8 -- liters per gallon
def kilometersToMiles : ℝ := 1.6 -- km per mile

theorem car_speed (fuel_efficiency fuelDecreaseGallons timeHours gallonsToLiters kilometersToMiles : ℝ) : 
  let fuelDecreaseLiters := fuelDecreaseGallons * gallonsToLiters
  let distanceKm := fuelDecreaseLiters * fuel_efficiency
  let distanceMiles := distanceKm / kilometersToMiles
  let averageSpeed := distanceMiles / timeHours
  averageSpeed = 65 := sorry

end car_speed_l265_265644


namespace average_percentage_difference_in_tail_sizes_l265_265260

-- Definitions for the number of segments in each type of rattlesnake
def segments_eastern : ℕ := 6
def segments_western : ℕ := 8
def segments_southern : ℕ := 7
def segments_northern : ℕ := 9

-- Definition for percentage difference function
def percentage_difference (a : ℕ) (b : ℕ) : ℚ := ((b - a : ℚ) / b) * 100

-- Theorem statement to prove the average percentage difference
theorem average_percentage_difference_in_tail_sizes :
  (percentage_difference segments_eastern segments_western +
   percentage_difference segments_southern segments_western +
   percentage_difference segments_northern segments_western) / 3 = 16.67 := 
sorry

end average_percentage_difference_in_tail_sizes_l265_265260


namespace expected_heads_value_in_cents_l265_265076

open ProbabilityTheory

-- Define the coins and their respective values
def penny_value := 1
def nickel_value := 5
def half_dollar_value := 50
def dollar_value := 100

-- Define the probability of landing heads for each coin
def heads_prob := 1 / 2

-- Define the expected value function
noncomputable def expected_value_of_heads : ℝ :=
  heads_prob * (penny_value + nickel_value + half_dollar_value + dollar_value)

theorem expected_heads_value_in_cents : expected_value_of_heads = 78 := by
  sorry

end expected_heads_value_in_cents_l265_265076


namespace quotient_correct_l265_265489

noncomputable def find_quotient (z : ℚ) : ℚ :=
  let dividend := (5 * z ^ 5 - 3 * z ^ 4 + 6 * z ^ 3 - 8 * z ^ 2 + 9 * z - 4)
  let divisor := (4 * z ^ 2 + 5 * z + 3)
  let quotient := ((5 / 4) * z ^ 3 - (47 / 16) * z ^ 2 + (257 / 64) * z - (1547 / 256))
  quotient

theorem quotient_correct (z : ℚ) :
  find_quotient z = ((5 / 4) * z ^ 3 - (47 / 16) * z ^ 2 + (257 / 64) * z - (1547 / 256)) :=
by
  sorry

end quotient_correct_l265_265489


namespace good_students_l265_265332

theorem good_students (E B : ℕ) (h1 : E + B = 25) (h2 : 12 < B) (h3 : B = 3 * (E - 1)) :
  E = 5 ∨ E = 7 :=
by 
  sorry

end good_students_l265_265332


namespace greatest_two_digit_multiple_of_17_l265_265174

theorem greatest_two_digit_multiple_of_17 : ∀ n, n ∈ finset.range 100 → n % 17 = 0 → n ≤ 85 →
  (∀ m, m ∈ finset.range 100 → m % 17 = 0 → m > n → m ≥ 100) →
  n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l265_265174


namespace bones_received_on_sunday_l265_265131

-- Definitions based on the conditions
def initial_bones : ℕ := 50
def bones_eaten : ℕ := initial_bones / 2
def bones_left_after_saturday : ℕ := initial_bones - bones_eaten
def total_bones_after_sunday : ℕ := 35

-- The theorem to prove how many bones received on Sunday
theorem bones_received_on_sunday : 
  (total_bones_after_sunday - bones_left_after_saturday = 10) :=
by
  -- proof will be filled in here
  sorry

end bones_received_on_sunday_l265_265131


namespace fraction_of_time_spent_covering_initial_distance_l265_265973

variables (D T : ℝ) (h1 : T = ((2 / 3) * D) / 80 + ((1 / 3) * D) / 40)

theorem fraction_of_time_spent_covering_initial_distance (h1 : T = ((2 / 3) * D) / 80 + ((1 / 3) * D) / 40) :
  ((2 / 3) * D / 80) / T = 1 / 2 :=
by
  sorry

end fraction_of_time_spent_covering_initial_distance_l265_265973


namespace orthocenter_circumcircle_reflection_l265_265501

noncomputable def reflection_point (A B P : Point) : Point :=
  -- Reflect P over the midpoint of segment AB. Placeholder definition, adjust accordingly.
  sorry 

theorem orthocenter_circumcircle_reflection
  (O : Point) (A B C P: Point)
  (Γ : Circle)
  (H : Point) 
  (A₁ B₁ C₁ A₂ B₂ C₂ : Point)
  (hΓA : Γ.radius = dist O A)
  (hΓB : Γ.radius = dist O B)
  (hΓC : Γ.radius = dist O C)
  (hA₁ : ∃ rayAP : Ray, Line_through (O) (A₁) ∧ intersects (rayAP) (Γ) ∧ A₁ ∈ rayAP)
  (hB₁ : ∃ rayBP : Ray, Line_through (O) (B₁) ∧ intersects (rayBP) (Γ) ∧ B₁ ∈ rayBP)
  (hC₁ : ∃ rayCP : Ray, Line_through (O) (C₁) ∧ intersects (rayCP) (Γ) ∧ C₁ ∈ rayCP)
  (hReflectA1 : A₂ = reflection_point B C A₁)
  (hReflectB1 : B₂ = reflection_point C A B₁)
  (hReflectC1 : C₂ = reflection_point A B C₁)
  (hH : ∃ h : H, H = orthocenter A B C) :
  Concyclic (H) (A₂ B₂ C₂) :=
  sorry

end orthocenter_circumcircle_reflection_l265_265501


namespace range_of_a_in_circle_l265_265115

theorem range_of_a_in_circle (a : ℝ) : 
  ((1 - a)^2 + (1 + a)^2 < 4) ↔ (-1 < a ∧ a < 1) :=
by
  sorry

end range_of_a_in_circle_l265_265115


namespace common_difference_l265_265564

variable (a1 d : ℤ)
variable (S : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def sum_first_n_terms (n : ℕ) : ℤ :=
  n * a1 + d * (n * (n - 1) / 2)

-- Condition: 2 * S 3 = 3 * S 2 + 6
axiom cond : 2 * sum_first_n_terms 3 = 3 * sum_first_n_terms 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem common_difference : d = 2 :=
by
  sorry

end common_difference_l265_265564


namespace vasya_gift_choices_l265_265741

theorem vasya_gift_choices :
  let cars := 7
  let construction_sets := 5
  (cars * construction_sets + Nat.choose cars 2 + Nat.choose construction_sets 2) = 66 :=
by
  sorry

end vasya_gift_choices_l265_265741


namespace frac_ab_eq_five_thirds_l265_265312

theorem frac_ab_eq_five_thirds (a b : ℝ) (hb : b ≠ 0) (h : (a - b) / b = 2 / 3) : a / b = 5 / 3 :=
by
  sorry

end frac_ab_eq_five_thirds_l265_265312


namespace find_a_parallel_l265_265320

-- Define the lines
def line1 (a : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  (a + 1) * x + 2 * y = 2

def line2 (a : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  x + a * y = 1

-- Define the parallel condition
def are_parallel (a : ℝ) : Prop :=
  ∀ x y : ℝ, line1 a x y → line2 a x y

-- The theorem stating our problem
theorem find_a_parallel (a : ℝ) : are_parallel a → a = -2 :=
by
  sorry

end find_a_parallel_l265_265320


namespace intersection_points_vertex_of_function_value_of_m_shift_l265_265285

noncomputable def quadratic_function (x m : ℝ) : ℝ :=
  (x - m) ^ 2 - 2 * (x - m)

theorem intersection_points (m : ℝ) : 
  ∃ x, quadratic_function x m = 0 ↔ x = m ∨ x = m + 2 := 
by
  sorry

theorem vertex_of_function (m : ℝ) : 
  ∃ x y, y = quadratic_function x m 
  ∧ x = m + 1 ∧ y = -1 := 
by
  sorry

theorem value_of_m_shift (m : ℝ) :
  (m - 2 = 0) → m = 2 :=
by
  sorry

end intersection_points_vertex_of_function_value_of_m_shift_l265_265285


namespace tom_calories_l265_265914

theorem tom_calories :
  let carrot_pounds := 1
  let broccoli_pounds := 2 * carrot_pounds
  let carrot_calories_per_pound := 51
  let broccoli_calories_per_pound := carrot_calories_per_pound / 3
  let total_carrot_calories := carrot_pounds * carrot_calories_per_pound
  let total_broccoli_calories := broccoli_pounds * broccoli_calories_per_pound
  let total_calories := total_carrot_calories + total_broccoli_calories
  total_calories = 85 :=
by
  sorry

end tom_calories_l265_265914


namespace largest_three_digit_number_l265_265215

theorem largest_three_digit_number :
  ∃ (n : ℕ), (n < 1000) ∧ (n % 7 = 1) ∧ (n % 8 = 4) ∧ (∀ (m : ℕ), (m < 1000) ∧ (m % 7 = 1) ∧ (m % 8 = 4) → m ≤ n) :=
sorry

end largest_three_digit_number_l265_265215


namespace determine_omega_phi_l265_265139

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem determine_omega_phi (ω φ : ℝ) (x : ℝ)
  (h₁ : 0 < ω) (h₂ : |φ| < Real.pi)
  (h₃ : f ω φ (5 * Real.pi / 8) = 2)
  (h₄ : f ω φ (11 * Real.pi / 8) = 0)
  (h₅ : (2 * Real.pi / ω) > 2 * Real.pi) :
  ω = 2 / 3 ∧ φ = Real.pi / 12 :=
sorry

end determine_omega_phi_l265_265139


namespace tablespoons_in_half_cup_l265_265907

theorem tablespoons_in_half_cup
    (grains_per_cup : ℕ)
    (half_cup : ℕ)
    (tbsp_to_tsp : ℕ)
    (grains_per_tsp : ℕ)
    (h1 : grains_per_cup = 480)
    (h2 : half_cup = grains_per_cup / 2)
    (h3 : tbsp_to_tsp = 3)
    (h4 : grains_per_tsp = 10) :
    (half_cup / (tbsp_to_tsp * grains_per_tsp) = 8) :=
by
  sorry

end tablespoons_in_half_cup_l265_265907


namespace beta_still_water_speed_l265_265764

-- Definitions that are used in the conditions
def alpha_speed_still_water : ℝ := 56 
def beta_speed_still_water : ℝ := 52  
def water_current_speed : ℝ := 4

-- The main theorem statement 
theorem beta_still_water_speed : β_speed_still_water = 61 := 
  sorry -- the proof goes here

end beta_still_water_speed_l265_265764


namespace bicycle_count_l265_265220

theorem bicycle_count (B T : ℕ) (hT : T = 20) (h_wheels : 2 * B + 3 * T = 160) : B = 50 :=
by
  sorry

end bicycle_count_l265_265220


namespace common_difference_arithmetic_sequence_l265_265551

theorem common_difference_arithmetic_sequence
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (d : ℕ) 
  (h₁ : ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2) -- sum formula for arithmetic sequence
  (h₂ : 2 * S 3 = 3 * S 2 + 6) : 
  d = 2 := 
sorry

end common_difference_arithmetic_sequence_l265_265551


namespace B1F_base16_to_base10_is_2847_l265_265474

theorem B1F_base16_to_base10_is_2847 : 
  let B := 11
  let one := 1
  let F := 15
  let base := 16
  B * base^2 + one * base^1 + F * base^0 = 2847 := 
by
  sorry

end B1F_base16_to_base10_is_2847_l265_265474


namespace price_increase_equivalence_l265_265005

theorem price_increase_equivalence (P : ℝ) : 
  let increase_35 := P * 1.35
  let increase_40 := increase_35 * 1.40
  let increase_20 := increase_40 * 1.20
  let final_increase := increase_20
  final_increase = P * 2.268 :=
by
  -- proof skipped
  sorry

end price_increase_equivalence_l265_265005


namespace fraction_computation_l265_265434

theorem fraction_computation : (2 / 3) * (3 / 4 * 40) = 20 := 
by
  -- The proof will go here, for now we use sorry to skip the proof.
  sorry

end fraction_computation_l265_265434


namespace victor_wins_ratio_l265_265622

theorem victor_wins_ratio (victor_wins friend_wins : ℕ) (hvw : victor_wins = 36) (fw : friend_wins = 20) : (victor_wins : ℚ) / friend_wins = 9 / 5 :=
by
  sorry

end victor_wins_ratio_l265_265622


namespace city_population_correct_l265_265254

variable (C G : ℕ)

theorem city_population_correct :
  (C - G = 119666) ∧ (C + G = 845640) → (C = 482653) := by
  intro h
  have h1 : C - G = 119666 := h.1
  have h2 : C + G = 845640 := h.2
  sorry

end city_population_correct_l265_265254


namespace usable_field_area_l265_265711

open Float

def breadth_of_field (P : ℕ) (extra_length : ℕ) := (P / 2 - extra_length) / 2

def length_of_field (b : ℕ) (extra_length : ℕ) := b + extra_length

def effective_length (l : ℕ) (obstacle_length : ℕ) := l - obstacle_length

def effective_breadth (b : ℕ) (obstacle_breadth : ℕ) := b - obstacle_breadth

def field_area (length : ℕ) (breadth : ℕ) := length * breadth 

theorem usable_field_area : 
  ∀ (P extra_length obstacle_length obstacle_breadth : ℕ), 
  P = 540 -> extra_length = 30 -> obstacle_length = 10 -> obstacle_breadth = 5 -> 
  field_area (effective_length (length_of_field (breadth_of_field P extra_length) extra_length) obstacle_length) (effective_breadth (breadth_of_field P extra_length) obstacle_breadth) = 16100 := by
  sorry

end usable_field_area_l265_265711


namespace reciprocal_of_abs_neg_two_l265_265898

theorem reciprocal_of_abs_neg_two : 1 / |(-2: ℤ)| = (1 / 2: ℚ) := by
  sorry

end reciprocal_of_abs_neg_two_l265_265898


namespace not_possible_perimeter_l265_265405

theorem not_possible_perimeter :
  ∀ (x : ℝ), 13 < x ∧ x < 37 → ¬ (37 + x = 50) :=
by
  intros x h
  sorry

end not_possible_perimeter_l265_265405


namespace ratio_2006_to_2005_l265_265323

-- Conditions
def kids_in_2004 : ℕ := 60
def kids_in_2005 : ℕ := kids_in_2004 / 2
def kids_in_2006 : ℕ := 20

-- The statement to prove
theorem ratio_2006_to_2005 : 
  (kids_in_2006 : ℚ) / kids_in_2005 = 2 / 3 :=
sorry

end ratio_2006_to_2005_l265_265323


namespace tom_total_calories_l265_265918

-- Define the conditions
def c_weight : ℕ := 1
def c_calories_per_pound : ℕ := 51
def b_weight : ℕ := 2 * c_weight
def b_calories_per_pound : ℕ := c_calories_per_pound / 3

-- Define the total calories
def total_calories : ℕ := (c_weight * c_calories_per_pound) + (b_weight * b_calories_per_pound)

-- Prove the total calories Tom eats
theorem tom_total_calories : total_calories = 85 := by
  sorry

end tom_total_calories_l265_265918


namespace valid_cone_from_sector_l265_265218

-- Given conditions
def sector_angle : ℝ := 300
def circle_radius : ℝ := 15

-- Definition of correct option E
def base_radius_E : ℝ := 12
def slant_height_E : ℝ := 15

theorem valid_cone_from_sector :
  ( (sector_angle / 360) * (2 * Real.pi * circle_radius) = 25 * Real.pi ) ∧
  (slant_height_E = circle_radius) ∧
  (base_radius_E = 12) ∧
  (15^2 = 12^2 + 9^2) :=
by
  -- This theorem states that given sector angle and circle radius, the valid option is E
  sorry

end valid_cone_from_sector_l265_265218


namespace k_starts_at_10_l265_265633

variable (V_k V_l : ℝ)
variable (t_k t_l : ℝ)

-- Conditions
axiom k_faster_than_l : V_k = 1.5 * V_l
axiom l_speed : V_l = 50
axiom l_start_time : t_l = 9
axiom meet_time : t_k + 3 = 12
axiom distance_apart : V_l * 3 + V_k * (12 - t_k) = 300

-- Proof goal
theorem k_starts_at_10 : t_k = 10 :=
by
  sorry

end k_starts_at_10_l265_265633


namespace magician_can_identify_matching_coin_l265_265656

-- Define the types and conditions
variable (coins : Fin 11 → Bool) -- Each coin is either heads (true) or tails (false).
variable (uncovered_index : Fin 11) -- The index of the initially uncovered coin.

-- Define the property to be proved: the magician can point to an adjacent coin with the same face as the uncovered one.
theorem magician_can_identify_matching_coin (h : ∃ i : Fin 11, coins i = coins (i + 1) % 11) :
  ∃ j : Fin 11, (j ≠ uncovered_index) ∧ coins j = coins uncovered_index :=
  sorry

end magician_can_identify_matching_coin_l265_265656


namespace largest_root_of_equation_l265_265938

theorem largest_root_of_equation : ∃ (x : ℝ), (x - 37)^2 - 169 = 0 ∧ ∀ y, (y - 37)^2 - 169 = 0 → y ≤ x :=
by
  sorry

end largest_root_of_equation_l265_265938


namespace barycentric_identity_l265_265940

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

noncomputable def barycentric (α β γ : ℝ) (a b c : V) : V := 
  α • a + β • b + γ • c

theorem barycentric_identity 
  (A B C X : V) 
  (α β γ : ℝ)
  (h : α + β + γ = 1)
  (hXA : X = barycentric α β γ A B C) :
  X - A = β • (B - A) + γ • (C - A) :=
by
  sorry

end barycentric_identity_l265_265940


namespace arccos_cos_10_l265_265470

theorem arccos_cos_10 : Real.arccos (Real.cos 10) = 2 := by
  sorry

end arccos_cos_10_l265_265470


namespace count_defective_pens_l265_265513

theorem count_defective_pens
  (total_pens : ℕ) (prob_non_defective : ℚ)
  (h1 : total_pens = 12)
  (h2 : prob_non_defective = 0.5454545454545454) :
  ∃ (D : ℕ), D = 1 := by
  sorry

end count_defective_pens_l265_265513


namespace items_in_bags_l265_265463

def calculateWaysToPlaceItems (n_items : ℕ) (n_bags : ℕ) : ℕ :=
  sorry

theorem items_in_bags :
  calculateWaysToPlaceItems 5 3 = 41 :=
by sorry

end items_in_bags_l265_265463


namespace good_students_count_l265_265350

noncomputable def student_count := 25

def is_good_student (s : Nat) := s ≤ student_count

def is_troublemaker (s : Nat) := s ≤ student_count

def always_tell_truth (s : Nat) := is_good_student s

def always_lie (s : Nat) := is_troublemaker s

def condition1 (E B : Nat) := E + B = student_count

def condition2 := ∀ (x : Nat), x ≤ 5 → is_good_student x → 
  ∃ (B : Nat), B > 24 / 2

def condition3 := ∀ (x : Nat), x ≤ 20 → is_troublemaker x → 
  ∃ (B E : Nat), B = 3 * (E - 1) ∧ E + B = student_count

theorem good_students_count :
  ∃ (E : Nat), condition1 E (student_count - E) ∧ condition2 ∧ condition3 :=
sorry  -- the actual proof is not required

end good_students_count_l265_265350


namespace evaluate_F_2_f_3_l265_265090

def f (a : ℤ) : ℤ := a^2 - 1

def F (a b : ℤ) : ℤ := b^3 - a

theorem evaluate_F_2_f_3 : F 2 (f 3) = 510 := by
  sorry

end evaluate_F_2_f_3_l265_265090


namespace train_crossing_time_l265_265965

-- Defining basic conditions
def train_length : ℕ := 150
def platform_length : ℕ := 100
def time_to_cross_post : ℕ := 15

-- The time it takes for the train to cross the platform
theorem train_crossing_time :
  (train_length + platform_length) / (train_length / time_to_cross_post) = 25 := 
sorry

end train_crossing_time_l265_265965


namespace solve_x_squared_plus_y_squared_l265_265298

-- Variables
variables {x y : ℝ}

-- Conditions
def cond1 : (x + y)^2 = 36 := sorry
def cond2 : x * y = 8 := sorry

-- Theorem stating the problem's equivalent proof
theorem solve_x_squared_plus_y_squared : x^2 + y^2 = 20 := sorry

end solve_x_squared_plus_y_squared_l265_265298


namespace grace_earnings_september_l265_265288

def charge_small_lawn_per_hour := 6
def charge_large_lawn_per_hour := 10
def charge_pull_small_weeds_per_hour := 11
def charge_pull_large_weeds_per_hour := 15
def charge_small_mulch_per_hour := 9
def charge_large_mulch_per_hour := 13

def hours_small_lawn := 20
def hours_large_lawn := 43
def hours_small_weeds := 4
def hours_large_weeds := 5
def hours_small_mulch := 6
def hours_large_mulch := 4

def earnings_small_lawn := hours_small_lawn * charge_small_lawn_per_hour
def earnings_large_lawn := hours_large_lawn * charge_large_lawn_per_hour
def earnings_small_weeds := hours_small_weeds * charge_pull_small_weeds_per_hour
def earnings_large_weeds := hours_large_weeds * charge_pull_large_weeds_per_hour
def earnings_small_mulch := hours_small_mulch * charge_small_mulch_per_hour
def earnings_large_mulch := hours_large_mulch * charge_large_mulch_per_hour

def total_earnings : ℕ :=
  earnings_small_lawn + earnings_large_lawn + earnings_small_weeds + earnings_large_weeds +
  earnings_small_mulch + earnings_large_mulch

theorem grace_earnings_september : total_earnings = 775 :=
by
  sorry

end grace_earnings_september_l265_265288


namespace arithmetic_sequence_sum_l265_265416

theorem arithmetic_sequence_sum (c d : ℤ) (h₁ : d = 10 - 3)
    (h₂ : c = 17 + (10 - 3)) (h₃ : d = 24 + (10 - 3)) :
    c + d = 55 :=
sorry

end arithmetic_sequence_sum_l265_265416


namespace finalStoresAtEndOf2020_l265_265611

def initialStores : ℕ := 23
def storesOpened2019 : ℕ := 5
def storesClosed2019 : ℕ := 2
def storesOpened2020 : ℕ := 10
def storesClosed2020 : ℕ := 6

theorem finalStoresAtEndOf2020 : initialStores + (storesOpened2019 - storesClosed2019) + (storesOpened2020 - storesClosed2020) = 30 :=
by
  sorry

end finalStoresAtEndOf2020_l265_265611


namespace find_P_l265_265442

theorem find_P (P Q R S : ℕ) (h1 : P ≠ Q) (h2 : P ≠ R) (h3 : P ≠ S) (h4 : Q ≠ R) (h5 : Q ≠ S) (h6 : R ≠ S)
  (h7 : P > 0) (h8 : Q > 0) (h9 : R > 0) (h10 : S > 0)
  (hPQ : P * Q = 72) (hRS : R * S = 72) (hDiff : P - Q = R + S) : P = 12 :=
by
  sorry

end find_P_l265_265442


namespace eventually_composite_appending_threes_l265_265763

theorem eventually_composite_appending_threes (n : ℕ) :
  ∃ n' : ℕ, n' = 10 * n + 3 ∧ ∃ k : ℕ, k > 0 ∧ (3 * k + 3) % 7 ≠ 1 ∧ (3 * k + 3) % 7 ≠ 2 ∧ (3 * k + 3) % 7 ≠ 3 ∧
  (3 * k + 3) % 7 ≠ 5 ∧ (3 * k + 3) % 7 ≠ 6 :=
sorry

end eventually_composite_appending_threes_l265_265763


namespace pet_store_cages_l265_265635

theorem pet_store_cages (initial_puppies sold_puppies puppies_per_cage : ℕ) (h₁ : initial_puppies = 78)
(h₂ : sold_puppies = 30) (h₃ : puppies_per_cage = 8) : (initial_puppies - sold_puppies) / puppies_per_cage = 6 :=
by
  -- assumptions: initial_puppies = 78, sold_puppies = 30, puppies_per_cage = 8
  -- goal: (initial_puppies - sold_puppies) / puppies_per_cage = 6
  sorry

end pet_store_cages_l265_265635


namespace common_difference_arithmetic_sequence_l265_265550

theorem common_difference_arithmetic_sequence
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (d : ℕ) 
  (h₁ : ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2) -- sum formula for arithmetic sequence
  (h₂ : 2 * S 3 = 3 * S 2 + 6) : 
  d = 2 := 
sorry

end common_difference_arithmetic_sequence_l265_265550


namespace find_general_term_l265_265272

theorem find_general_term (S a : ℕ → ℤ) (n : ℕ) (h_sum : S n = 2 * a n + 1) : a n = -2 * n - 1 := sorry

end find_general_term_l265_265272


namespace uv_square_l265_265119

theorem uv_square (u v : ℝ) (h1 : u * (u + v) = 50) (h2 : v * (u + v) = 100) : (u + v)^2 = 150 := by
  sorry

end uv_square_l265_265119


namespace solve_x_squared_plus_y_squared_l265_265302

-- Variables
variables {x y : ℝ}

-- Conditions
def cond1 : (x + y)^2 = 36 := sorry
def cond2 : x * y = 8 := sorry

-- Theorem stating the problem's equivalent proof
theorem solve_x_squared_plus_y_squared : x^2 + y^2 = 20 := sorry

end solve_x_squared_plus_y_squared_l265_265302


namespace sqrt_equation_l265_265145

theorem sqrt_equation (n : ℕ) (h : 0 < n) : 
  Real.sqrt (1 + 1 / (n^2 : ℝ) + 1 / ((n+1)^2 : ℝ)) = 1 + 1 / (n * (n + 1) : ℝ) :=
sorry

end sqrt_equation_l265_265145


namespace inequality_not_true_l265_265268

variable (a b c : ℝ)

theorem inequality_not_true (h : a < b) : ¬ (-3 * a < -3 * b) :=
by
  sorry

end inequality_not_true_l265_265268


namespace proof_q_values_proof_q_comparison_l265_265971

-- Definitions of the conditions given.
def q : ℝ → ℝ := 
  sorry -- The definition is not required to be constructed, as we are only focusing on the conditions given.

-- Conditions
axiom cond1 : q 2 = 5
axiom cond2 : q 1.5 = 3

-- Statements to prove
theorem proof_q_values : (q 2 = 5) ∧ (q 1.5 = 3) := 
  by sorry

theorem proof_q_comparison : q 2 > q 1.5 :=
  by sorry

end proof_q_values_proof_q_comparison_l265_265971


namespace total_plums_correct_l265_265876

/-- Each picked number of plums. -/
def melanie_picked := 4
def dan_picked := 9
def sally_picked := 3
def ben_picked := 2 * (melanie_picked + dan_picked)
def sally_ate := 2

/-- The total number of plums picked in the end. -/
def total_plums_picked :=
  melanie_picked + dan_picked + sally_picked + ben_picked - sally_ate

theorem total_plums_correct : total_plums_picked = 40 := by
  sorry

end total_plums_correct_l265_265876


namespace triangle_perimeter_l265_265822

theorem triangle_perimeter (P₁ P₂ P₃ : ℝ) (hP₁ : P₁ = 12) (hP₂ : P₂ = 14) (hP₃ : P₃ = 16) : 
  P₁ + P₂ + P₃ = 42 := by
  sorry

end triangle_perimeter_l265_265822


namespace four_digit_div_by_99_then_sum_div_by_18_l265_265037

/-- 
If a whole number with at most four digits is divisible by 99, then 
the sum of its digits is divisible by 18. 
-/
theorem four_digit_div_by_99_then_sum_div_by_18 (n : ℕ) (h1 : n < 10000) (h2 : 99 ∣ n) : 
  18 ∣ (n.digits 10).sum := 
sorry

end four_digit_div_by_99_then_sum_div_by_18_l265_265037


namespace complement_union_example_l265_265286

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {1, 3, 4}

-- Define set B
def B : Set ℕ := {2, 4}

-- State the theorem we want to prove
theorem complement_union_example : (U \ A) ∪ B = {2, 4, 5} :=
by
  sorry

end complement_union_example_l265_265286


namespace volume_of_tetrahedron_l265_265126

-- Define the setup of tetrahedron D-ABC
def tetrahedron_volume (V : ℝ) : Prop :=
  ∃ (DA : ℝ) (A B C D : ℝ × ℝ × ℝ), 
  A = (0, 0, 0) ∧ 
  B = (2, 0, 0) ∧ 
  C = (1, Real.sqrt 3, 0) ∧
  D = (1, Real.sqrt 3/3, DA) ∧
  DA = 2 * Real.sqrt 3 ∧
  ∃ tan_dihedral : ℝ, tan_dihedral = 2 ∧
  V = 2

-- The statement to prove the volume is indeed 2 given the conditions.
theorem volume_of_tetrahedron : ∃ V, tetrahedron_volume V :=
by 
  sorry

end volume_of_tetrahedron_l265_265126


namespace sum_of_A_B_C_l265_265476

theorem sum_of_A_B_C (A B C : ℕ) (h_pos_A : 0 < A) (h_pos_B : 0 < B) (h_pos_C : 0 < C) (h_rel_prime : Nat.gcd A (Nat.gcd B C) = 1) (h_eq : A * Real.log 3 / Real.log 180 + B * Real.log 5 / Real.log 180 = C) : A + B + C = 4 :=
sorry

end sum_of_A_B_C_l265_265476


namespace no_solution_a4_plus_6_eq_b3_mod_13_l265_265398

theorem no_solution_a4_plus_6_eq_b3_mod_13 :
  ¬ ∃ (a b : ℤ), (a^4 + 6) % 13 = b^3 % 13 :=
by
  sorry

end no_solution_a4_plus_6_eq_b3_mod_13_l265_265398


namespace total_wood_gathered_l265_265469

def pieces_per_sack := 20
def number_of_sacks := 4

theorem total_wood_gathered : pieces_per_sack * number_of_sacks = 80 := 
by 
  sorry

end total_wood_gathered_l265_265469


namespace part_a_part_b_l265_265449

noncomputable def withdraw_rubles_after_one_year
  (initial_deposit : ℤ) (initial_rate : ℤ) (annual_yield : ℚ)
  (final_rate : ℤ) (conversion_commission : ℚ) (broker_commission : ℚ) : ℚ :=
  let deposit_in_dollars := initial_deposit / initial_rate
  let interest_earned := deposit_in_dollars * annual_yield
  let total_in_dollars := deposit_in_dollars + interest_earned
  let broker_fee := interest_earned * broker_commission
  let amount_after_fee := total_in_dollars - broker_fee
  let total_in_rubles := amount_after_fee * final_rate
  let conversion_fee := total_in_rubles * conversion_commission
  total_in_rubles - conversion_fee

theorem part_a
  (initial_deposit : ℤ) (initial_rate : ℤ) (annual_yield : ℚ)
  (final_rate : ℤ) (conversion_commission : ℚ) (broker_commission : ℚ) :
  withdraw_rubles_after_one_year initial_deposit initial_rate annual_yield final_rate conversion_commission broker_commission =
  16476.8 := sorry

def effective_yield (initial_rubles final_rubles : ℚ) : ℚ :=
  (final_rubles / initial_rubles - 1) * 100

theorem part_b
  (initial_deposit : ℤ) (final_rubles : ℚ) :
  effective_yield initial_deposit final_rubles = 64.77 := sorry

end part_a_part_b_l265_265449


namespace solve_for_x_l265_265101

theorem solve_for_x (x : ℕ) (h : x + 1 = 4) : x = 3 :=
by
  sorry

end solve_for_x_l265_265101


namespace maximum_m_value_l265_265111

theorem maximum_m_value (a : ℕ → ℤ) (m : ℕ) :
  (∀ n, a (n + 1) - a n = 3) →
  a 3 = -2 →
  (∀ k : ℕ, k ≥ 4 → (3 * k - 8) * (3 * k - 5) / (3 * k - 11) ≥ 3 * m - 11) →
  m ≤ 9 :=
by
  sorry

end maximum_m_value_l265_265111


namespace turtle_feeding_cost_l265_265159

theorem turtle_feeding_cost (total_weight_pounds : ℝ) (food_per_half_pound : ℝ)
  (jar_food_ounces : ℝ) (jar_cost_dollars : ℝ) (total_cost : ℝ) : 
  total_weight_pounds = 30 →
  food_per_half_pound = 1 →
  jar_food_ounces = 15 →
  jar_cost_dollars = 2 →
  total_cost = 8 :=
by
  intros h_weight h_food h_jar_ounces h_jar_cost
  sorry

end turtle_feeding_cost_l265_265159


namespace hypotenuse_length_l265_265438

theorem hypotenuse_length (a b c : ℕ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : a^2 + b^2 = c^2) : c = 13 :=
by
  -- proof
  sorry

end hypotenuse_length_l265_265438


namespace kevin_initial_cards_l265_265723

theorem kevin_initial_cards (found_cards : ℕ) (total_cards : ℕ) (h1 : found_cards = 47) (h2 : total_cards = 54) : total_cards - found_cards = 7 := 
by
  rw [h1, h2]
  norm_num
  done

end kevin_initial_cards_l265_265723


namespace projectiles_meet_in_84_minutes_l265_265765

theorem projectiles_meet_in_84_minutes :
  ∀ (d v₁ v₂ : ℝ), d = 1386 → v₁ = 445 → v₂ = 545 → (20 : ℝ) = 20 → 
  ((1386 / (445 + 545) / 60) * 60 * 60 = 84) :=
by
  intros d v₁ v₂ h_d h_v₁ h_v₂ h_wind
  sorry

end projectiles_meet_in_84_minutes_l265_265765


namespace min_k_period_at_least_15_l265_265879

theorem min_k_period_at_least_15 (a b : ℚ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
    (h_period_a : ∃ m, a = m / (10^30 - 1))
    (h_period_b : ∃ n, b = n / (10^30 - 1))
    (h_period_ab : ∃ p, (a - b) = p / (10^30 - 1) ∧ 10^15 + 1 ∣ p) :
    ∃ k : ℕ, k = 6 ∧ (∃ q, (a + k * b) = q / (10^30 - 1) ∧ 10^15 + 1 ∣ q) :=
sorry

end min_k_period_at_least_15_l265_265879


namespace second_set_parallel_lines_l265_265708

theorem second_set_parallel_lines (n : ℕ) (h1 : 5 * (n - 1) = 420) : n = 85 :=
by sorry

end second_set_parallel_lines_l265_265708


namespace no_function_satisfies_inequality_l265_265229

theorem no_function_satisfies_inequality (f : ℝ → ℝ) :
  ¬ ∀ x y : ℝ, (f x + f y) / 2 ≥ f ((x + y) / 2) + |x - y| :=
sorry

end no_function_satisfies_inequality_l265_265229


namespace find_k_l265_265721

theorem find_k (k : ℝ) (h_line : ∀ x y : ℝ, 3 * x + 5 * y + k = 0)
    (h_sum_intercepts : - (k / 3) - (k / 5) = 16) : k = -30 := by
  sorry

end find_k_l265_265721


namespace bags_filled_l265_265878

def bags_filled_on_certain_day (x : ℕ) : Prop :=
  let bags := x + 3
  let total_cans := 8 * bags
  total_cans = 72

theorem bags_filled {x : ℕ} (h : bags_filled_on_certain_day x) : x = 6 :=
  sorry

end bags_filled_l265_265878


namespace circumference_of_circle_of_given_area_l265_265170

theorem circumference_of_circle_of_given_area (A : ℝ) (h : A = 225 * Real.pi) : 
  ∃ C : ℝ, C = 2 * Real.pi * 15 :=
by
  let r := 15
  let C := 2 * Real.pi * r
  use C
  sorry

end circumference_of_circle_of_given_area_l265_265170


namespace product_positive_probability_l265_265932

theorem product_positive_probability :
  let interval := set.Icc (-30 : ℝ) 15 in
  let prob_neg := (30 : ℝ) / 45 in
  let prob_pos := (15 : ℝ) / 45 in
  let prob_product_neg := 2 * (prob_neg * prob_pos) in
  let prob_product_pos := (prob_neg ^ 2) + (prob_pos ^ 2) in
  (prob_product_pos = 5 / 9) :=
by
  sorry

end product_positive_probability_l265_265932


namespace number_of_lizards_l265_265447

theorem number_of_lizards (total_geckos : ℕ) (insects_per_gecko : ℕ) (total_insects_eaten : ℕ) (insects_per_lizard : ℕ) 
  (gecko_total_insects : total_geckos * insects_per_gecko = 5 * 6) (lizard_insects: insects_per_lizard = 2 * insects_per_gecko)
  (total_insects : total_insects_eaten = 66) : 
  (total_insects_eaten - total_geckos * insects_per_gecko) / insects_per_lizard = 3 :=
by 
  sorry

end number_of_lizards_l265_265447


namespace donuts_eaten_on_monday_l265_265083

theorem donuts_eaten_on_monday (D : ℕ) (h1 : D + D / 2 + 4 * D = 49) : 
  D = 9 :=
sorry

end donuts_eaten_on_monday_l265_265083


namespace smallest_candies_value_l265_265249

def smallest_valid_n := ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n % 9 = 2 ∧ n % 7 = 5 ∧ ∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ m % 9 = 2 ∧ m % 7 = 5 → n ≤ m

theorem smallest_candies_value : ∃ n : ℕ, smallest_valid_n ∧ n = 101 := 
by {
  sorry  
}

end smallest_candies_value_l265_265249


namespace probability_at_most_3_heads_in_12_flips_l265_265446

theorem probability_at_most_3_heads_in_12_flips :
  let favorable := Nat.choose 12 0 + Nat.choose 12 1 + Nat.choose 12 2 + Nat.choose 12 3
  let total := 2^12
  (favorable : ℝ) / total = 299 / 4096 :=
by
  sorry

end probability_at_most_3_heads_in_12_flips_l265_265446


namespace students_who_wore_blue_lipstick_l265_265607

theorem students_who_wore_blue_lipstick (total_students : ℕ) (h1 : total_students = 200) : 
  ∃ blue_lipstick_students : ℕ, blue_lipstick_students = 5 :=
by
  have colored_lipstick_students := total_students / 2
  have red_lipstick_students := colored_lipstick_students / 4
  let blue_lipstick_students := red_lipstick_students / 5
  have h2 : blue_lipstick_students = 5 :=
    calc blue_lipstick_students
          = (total_students / 2) / 4 / 5 : by sorry -- detailed calculation steps omitted
  use blue_lipstick_students
  exact h2

end students_who_wore_blue_lipstick_l265_265607


namespace max_angle_between_tangents_l265_265502

open Real

-- Define the equation of the circle
def circle_A (x y : ℝ) : Prop := (x - 3) ^ 2 + y ^ 2 = 2

-- Define the equation of the parabola
def parabola_C (x y : ℝ) : Prop := y ^ 2 = 4 * x

-- Define the statement we want to prove
theorem max_angle_between_tangents :
  ∃ (P : ℝ × ℝ), parabola_C P.1 P.2 ∧
    (∀ (A : ℝ × ℝ), circle_A A.1 A.2 → (angle_between_tangents P A ≤ 60)) :=
sorry

end max_angle_between_tangents_l265_265502


namespace find_certain_number_l265_265786

theorem find_certain_number (x : ℝ) (h : ((7 * (x + 5)) / 5) - 5 = 33) : x = 22 :=
by
  sorry

end find_certain_number_l265_265786


namespace tom_total_calories_l265_265919

-- Define the conditions
def c_weight : ℕ := 1
def c_calories_per_pound : ℕ := 51
def b_weight : ℕ := 2 * c_weight
def b_calories_per_pound : ℕ := c_calories_per_pound / 3

-- Define the total calories
def total_calories : ℕ := (c_weight * c_calories_per_pound) + (b_weight * b_calories_per_pound)

-- Prove the total calories Tom eats
theorem tom_total_calories : total_calories = 85 := by
  sorry

end tom_total_calories_l265_265919


namespace cone_volume_difference_l265_265238

theorem cone_volume_difference (H R : ℝ) : ΔV = (1/12) * Real.pi * R^2 * H := 
sorry

end cone_volume_difference_l265_265238


namespace max_value_ab_bc_cd_l265_265015

theorem max_value_ab_bc_cd (a b c d : ℝ) (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0)
  (h_sum : a + b + c + d = 120) : ab + bc + cd ≤ 3600 :=
by {
  sorry
}

end max_value_ab_bc_cd_l265_265015


namespace exactly_one_solves_l265_265036

-- Define the independent probabilities for person A and person B
variables (p₁ p₂ : ℝ)

-- Assume probabilities are between 0 and 1 inclusive
axiom h1 : 0 ≤ p₁ ∧ p₁ ≤ 1
axiom h2 : 0 ≤ p₂ ∧ p₂ ≤ 1

theorem exactly_one_solves : (p₁ * (1 - p₂) + p₂ * (1 - p₁)) = (p₁ * (1 - p₂) + p₂ * (1 - p₁)) := 
by sorry

end exactly_one_solves_l265_265036


namespace quadratic_equation_coefficients_l265_265518

theorem quadratic_equation_coefficients :
  ∃ (a b c : ℤ), a = 2 ∧ b = 1 ∧ c = -5 ∧ (∀ x : ℤ, 2 * x^2 + x - 5 = a * x^2 + b * x + c) :=
begin
  sorry
end

end quadratic_equation_coefficients_l265_265518


namespace find_y_l265_265511

theorem find_y (y : ℝ) : 3 + 1 / (2 - y) = 2 * (1 / (2 - y)) → y = 5 / 3 := 
by
  sorry

end find_y_l265_265511


namespace ratio_mara_janet_l265_265128

variables {B J M : ℕ}

/-- Janet has 9 cards more than Brenda --/
def janet_cards (B : ℕ) : ℕ := B + 9

/-- Mara has 40 cards less than 150 --/
def mara_cards : ℕ := 150 - 40

/-- They have a total of 211 cards --/
axiom total_cards_eq (B : ℕ) : B + janet_cards B + mara_cards = 211

/-- Mara has a multiple of Janet's number of cards --/
axiom multiples_cards (J M : ℕ) : J * 2 = M

theorem ratio_mara_janet (B J M : ℕ) (h1 : janet_cards B = J)
  (h2 : mara_cards = M) (h3 : J * 2 = M) :
  (M / J : ℕ) = 2 :=
sorry

end ratio_mara_janet_l265_265128


namespace factorize_expression_l265_265485

-- Variables x and y are real numbers
variables (x y : ℝ)

-- Theorem statement
theorem factorize_expression : 3 * x^2 - 12 * y^2 = 3 * (x - 2 * y) * (x + 2 * y) :=
sorry

end factorize_expression_l265_265485


namespace quadratic_part_of_equation_l265_265901

theorem quadratic_part_of_equation (x: ℝ) :
  (x^2 - 8*x + 21 = |x - 5| + 4) → (x^2 - 8*x + 21) = x^2 - 8*x + 21 :=
by
  intros h
  sorry

end quadratic_part_of_equation_l265_265901


namespace loan_amount_needed_l265_265396

-- Define the total cost of tuition.
def total_tuition : ℝ := 30000

-- Define the amount Sabina has saved.
def savings : ℝ := 10000

-- Define the grant coverage rate.
def grant_coverage_rate : ℝ := 0.4

-- Define the remainder of the tuition after using savings.
def remaining_tuition : ℝ := total_tuition - savings

-- Define the amount covered by the grant.
def grant_amount : ℝ := grant_coverage_rate * remaining_tuition

-- Define the loan amount Sabina needs to apply for.
noncomputable def loan_amount : ℝ := remaining_tuition - grant_amount

-- State the theorem to prove the loan amount needed.
theorem loan_amount_needed : loan_amount = 12000 := by
  sorry

end loan_amount_needed_l265_265396


namespace Niko_total_profit_l265_265030

-- Definitions based on conditions
def cost_per_pair : ℕ := 2
def total_pairs : ℕ := 9
def profit_margin_4_pairs : ℚ := 0.25
def profit_per_other_pair : ℚ := 0.2
def pairs_with_margin : ℕ := 4
def pairs_with_fixed_profit : ℕ := 5

-- Calculations based on definitions
def total_cost : ℚ := total_pairs * cost_per_pair
def profit_on_margin_pairs : ℚ := pairs_with_margin * (profit_margin_4_pairs * cost_per_pair)
def profit_on_fixed_profit_pairs : ℚ := pairs_with_fixed_profit * profit_per_other_pair
def total_profit : ℚ := profit_on_margin_pairs + profit_on_fixed_profit_pairs

-- Statement to prove
theorem Niko_total_profit : total_profit = 3 := by
  sorry

end Niko_total_profit_l265_265030


namespace company_production_average_l265_265632

theorem company_production_average (n : ℕ) 
  (h1 : (50 * n) / n = 50) 
  (h2 : (50 * n + 105) / (n + 1) = 55) :
  n = 10 :=
sorry

end company_production_average_l265_265632


namespace probability_of_stopping_after_2nd_shot_l265_265259

-- Definitions based on the conditions
def shootingProbability : ℚ := 2 / 3

noncomputable def scoring (n : ℕ) : ℕ := 12 - n

def stopShootingProbabilityAfterNthShot (n : ℕ) (probOfShooting : ℚ) : ℚ :=
  if n = 2 then (1 / 3) * (2 / 3) * sorry -- Note: Here, filling in the remaining calculation steps according to problem logic.
  else sorry -- placeholder for other cases

theorem probability_of_stopping_after_2nd_shot :
  stopShootingProbabilityAfterNthShot 2 shootingProbability = 8 / 729 :=
by
  sorry

end probability_of_stopping_after_2nd_shot_l265_265259


namespace arithmetic_sequence_sum_l265_265417

theorem arithmetic_sequence_sum (c d : ℤ) (h₁ : d = 10 - 3)
    (h₂ : c = 17 + (10 - 3)) (h₃ : d = 24 + (10 - 3)) :
    c + d = 55 :=
sorry

end arithmetic_sequence_sum_l265_265417


namespace sum_of_angles_l265_265466

theorem sum_of_angles (θ₁ θ₂ θ₃ θ₄ : ℝ)
  (h₁ : θ₁ = 67.5) (h₂ : θ₂ = 157.5) (h₃ : θ₃ = 247.5) (h₄ : θ₄ = 337.5) :
  θ₁ + θ₂ + θ₃ + θ₄ = 810 :=
by
  -- These parameters are used only to align with provided conditions
  let r₁ := 1
  let r₂ := r₁
  let r₃ := r₁
  let r₄ := r₁
  have z₁ := r₁ * (Complex.cos θ₁ + Complex.sin θ₁ * Complex.I)
  have z₂ := r₂ * (Complex.cos θ₂ + Complex.sin θ₂ * Complex.I)
  have z₃ := r₃ * (Complex.cos θ₃ + Complex.sin θ₃ * Complex.I)
  have z₄ := r₄ * (Complex.cos θ₄ + Complex.sin θ₄ * Complex.I)
  sorry

end sum_of_angles_l265_265466


namespace circle_radius_k_l265_265105

theorem circle_radius_k (k : ℝ) : (∃ x y : ℝ, (x^2 + 14*x + y^2 + 8*y - k = 0) ∧ ((x + 7)^2 + (y + 4)^2 = 100)) → k = 35 :=
by
  sorry

end circle_radius_k_l265_265105


namespace mo_rainy_days_last_week_l265_265063

theorem mo_rainy_days_last_week (R NR n : ℕ) (h1 : n * R + 4 * NR = 26) (h2 : 4 * NR - n * R = 14) (h3 : R + NR = 7) : R = 2 :=
sorry

end mo_rainy_days_last_week_l265_265063


namespace sodas_per_pack_l265_265519

theorem sodas_per_pack 
  (packs : ℕ) (initial_sodas : ℕ) (days_in_a_week : ℕ) (sodas_per_day : ℕ) 
  (total_sodas_consumed : ℕ) (sodas_per_pack : ℕ) :
  packs = 5 →
  initial_sodas = 10 →
  days_in_a_week = 7 →
  sodas_per_day = 10 →
  total_sodas_consumed = 70 →
  total_sodas_consumed - initial_sodas = packs * sodas_per_pack →
  sodas_per_pack = 12 :=
by
  intros hpacks hinitial hsodas hdaws htpd htcs
  sorry

end sodas_per_pack_l265_265519


namespace knight_liar_grouping_l265_265066

noncomputable def can_be_partitioned_into_knight_liar_groups (n m : ℕ) (h1 : n ≥ 2) (h2 : ∃ k : ℕ, 1 ≤ k ∧ k < n) : Prop :=
  ∃ t : ℕ, n = (m + 1) * t

-- Show that if the company has n people, where n ≥ 2, and there exists at least one knight,
-- then n can be partitioned into groups where each group contains 1 knight and m liars.
theorem knight_liar_grouping (n m : ℕ) (h1 : n ≥ 2) (h2 : ∃ k : ℕ, 1 ≤ k ∧ k < n) : can_be_partitioned_into_knight_liar_groups n m h1 h2 :=
sorry

end knight_liar_grouping_l265_265066


namespace Toph_caught_12_fish_l265_265800

-- Define the number of fish each person caught
def Aang_fish : ℕ := 7
def Sokka_fish : ℕ := 5
def average_fish : ℕ := 8
def num_people : ℕ := 3

-- The total number of fish based on the average
def total_fish : ℕ := average_fish * num_people

-- Define the number of fish Toph caught
def Toph_fish : ℕ := total_fish - Aang_fish - Sokka_fish

-- Prove that Toph caught the correct number of fish
theorem Toph_caught_12_fish : Toph_fish = 12 := sorry

end Toph_caught_12_fish_l265_265800


namespace find_m_l265_265802

noncomputable def first_series_sum : ℝ := 
  let a1 : ℝ := 18
  let a2 : ℝ := 6
  let r : ℝ := a2 / a1
  a1 / (1 - r)

noncomputable def second_series_sum (m : ℝ) : ℝ := 
  let b1 : ℝ := 18
  let b2 : ℝ := 6 + m
  let s : ℝ := b2 / b1
  b1 / (1 - s)

theorem find_m : 
  (3 : ℝ) * first_series_sum = second_series_sum m → m = 8 := 
by 
  sorry

end find_m_l265_265802


namespace hyperbola_foci_difference_l265_265113

noncomputable def hyperbola_foci_distance (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) (a : ℝ) : ℝ :=
  |dist P F₁ - dist P F₂|

theorem hyperbola_foci_difference (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) : 
  (P.1 ^ 2 - P.2 ^ 2 = 4) ∧ (P.1 < 0) → (hyperbola_foci_distance P F₁ F₂ 2 = -4) :=
by
  intros h
  sorry

end hyperbola_foci_difference_l265_265113


namespace xy_problem_l265_265310

theorem xy_problem (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 :=
by
  sorry

end xy_problem_l265_265310


namespace arithmetic_common_difference_l265_265583

-- Defining the arithmetic sequence sum function S
def S (a₁ a₂ a₃ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if n = 3 then a₁ + a₂ + a₃
  else if n = 2 then a₁ + a₂
  else 0

-- The Lean statement for the problem
theorem arithmetic_common_difference (a₁ a₂ a₃ d : ℝ) 
  (h1 : a₂ = a₁ + d) 
  (h2 : a₃ = a₂ + d)
  (h3 : 2 * S a₁ a₂ a₃ d 3 = 3 * S a₁ a₂ a₃ d 2 + 6) :
  d = 2 :=
by
  sorry

end arithmetic_common_difference_l265_265583


namespace find_average_after_17th_inning_l265_265448

def initial_average_after_16_inns (A : ℕ) : Prop :=
  let total_runs := 16 * A
  let new_total_runs := total_runs + 87
  let new_average := new_total_runs / 17
  new_average = A + 4

def runs_in_17th_inning := 87

noncomputable def average_after_17th_inning (A : ℕ) : Prop :=
  A + 4 = 23

theorem find_average_after_17th_inning (A : ℕ) :
  initial_average_after_16_inns A →
  average_after_17th_inning A :=
  sorry

end find_average_after_17th_inning_l265_265448


namespace arithmetic_sequence_common_difference_l265_265558

theorem arithmetic_sequence_common_difference (a_1 d : ℝ) (S : ℕ → ℝ) 
    (h1 : S 2 = 2 * a_1 + d)
    (h2 : S 3 = 3 * a_1 + 3 * d)
    (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 := 
by
  sorry

end arithmetic_sequence_common_difference_l265_265558


namespace sequence_identical_l265_265885

noncomputable def a (n : ℕ) : ℝ :=
  (1 / (2 * Real.sqrt 3)) * ((2 + Real.sqrt 3)^n - (2 - Real.sqrt 3)^n)

theorem sequence_identical (n : ℕ) :
  a (n + 1) = (a n + a (n + 2)) / 4 :=
by
  sorry

end sequence_identical_l265_265885


namespace simplify_expression_l265_265153

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^2 + 2 * b - 4) - 2 * b^2 = 9 * b^3 + 4 * b^2 - 12 * b :=
by sorry

end simplify_expression_l265_265153


namespace arithmetic_sequence_sum_l265_265978

theorem arithmetic_sequence_sum :
  let sequence := list.range (20 / 2) in
  let sum := sequence.map (λ n, 2 * (n + 1)).sum in
  sum = 110 :=
by
  -- Define the sequence as the arithmetic series
  let sequence := list.range (20 / 2)
  -- Calculate the sum of the arithmetic sequence
  let sum := sequence.map (λ n, 2 * (n + 1)).sum
  -- Check the sum
  have : sum = 110 := sorry
  exact this

end arithmetic_sequence_sum_l265_265978


namespace class_proof_l265_265340

def class_problem : Prop :=
  let total_students := 25
  let (students : Type) := {s // s ∈ Fin total_students}
  let good students (s : students) : Prop := sorry -- Condition Placeholder
  let troublemakers (s : students) : Prop := sorry -- Condition Placeholder
  have all_good_or_trouble : ∀ s, good students s ∨ troublemakers s,
    from sorry,
  have lh_trouble : ∀ (a b c d e : students), (good students a ∧ good students b ∧ good students c ∧ good students d ∧ good students e) →
    ∃ (x : ℕ), x > ((total_students - 1) / 2) ∧
    (λ rem_trouble : students - {a, b, c, d, e}, B : ℕ, B = x) ∧
    (λ rem_good : students - {a, b, c, d, e}, G : ℕ, G ∈ rem_good), from sorry,
  have th_triple_ratio : ∀ s, (students - {s}).card = 24 →
    ∃ (x y : ℕ), (troublemakers x ∧ good students y) ∧ (3*y = x), from sorry,
  ∃ (num_good_students : ℕ), (num_good_students = 5 ∨ num_good_students = 7)
  
theorem class_proof : class_problem :=
  sorry

end class_proof_l265_265340


namespace market_trips_l265_265167

theorem market_trips (d_school_round: ℝ) (d_market_round: ℝ) (num_school_trips_per_day: ℕ) (num_school_days_per_week: ℕ) (total_week_mileage: ℝ) :
  d_school_round = 5 →
  d_market_round = 4 →
  num_school_trips_per_day = 2 →
  num_school_days_per_week = 4 →
  total_week_mileage = 44 →
  (total_week_mileage - (d_school_round * num_school_trips_per_day * num_school_days_per_week)) / d_market_round = 1 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end market_trips_l265_265167


namespace john_initial_money_l265_265378

variable (X S : ℕ)
variable (L : ℕ := 500)
variable (cond1 : L = S - 600)
variable (cond2 : X = S + L)

theorem john_initial_money : X = 1600 :=
by
  sorry

end john_initial_money_l265_265378


namespace unique_solution_p_l265_265993

theorem unique_solution_p (p : ℚ) :
  (∀ x : ℝ, (2 * x + 3) / (p * x - 2) = x) ↔ p = -4 / 3 := sorry

end unique_solution_p_l265_265993


namespace total_amount_paid_l265_265863

def original_price_per_card : Int := 12
def discount_per_card : Int := 2
def number_of_cards : Int := 10

theorem total_amount_paid :
  original_price_per_card - discount_per_card * number_of_cards = 100 :=
by
  sorry

end total_amount_paid_l265_265863


namespace melted_mixture_weight_l265_265634

-- Let Zinc and Copper be real numbers representing their respective weights in kilograms.
variables (Zinc Copper: ℝ)
-- Assume the ratio of Zinc to Copper is 9:11.
axiom ratio_zinc_copper : Zinc / Copper = 9 / 11
-- Assume 26.1kg of Zinc has been used.
axiom zinc_value : Zinc = 26.1

-- Define the total weight of the melted mixture.
def total_weight := Zinc + Copper

-- We state the theorem to prove that the total weight of the mixture equals 58kg.
theorem melted_mixture_weight : total_weight Zinc Copper = 58 :=
by
  sorry

end melted_mixture_weight_l265_265634


namespace kevin_started_with_cards_l265_265722

-- The definitions corresponding to the conditions in the problem
def ended_with : Nat := 54
def found_cards : Nat := 47
def started_with (ended_with found_cards : Nat) : Nat := ended_with - found_cards

-- The Lean statement for the proof problem itself
theorem kevin_started_with_cards : started_with ended_with found_cards = 7 := by
  sorry

end kevin_started_with_cards_l265_265722


namespace factorize_expression_l265_265483

-- Variables x and y are real numbers
variables (x y : ℝ)

-- Theorem statement
theorem factorize_expression : 3 * x^2 - 12 * y^2 = 3 * (x - 2 * y) * (x + 2 * y) :=
sorry

end factorize_expression_l265_265483


namespace hypotenuse_length_l265_265861

variable (a b c : ℝ)

-- Given conditions
theorem hypotenuse_length (h1 : b = 3 * a) 
                          (h2 : a^2 + b^2 + c^2 = 500) 
                          (h3 : c^2 = a^2 + b^2) : 
                          c = 5 * Real.sqrt 10 := 
by 
  sorry

end hypotenuse_length_l265_265861


namespace round_trip_time_l265_265047

theorem round_trip_time (current_speed : ℝ) (boat_speed_still : ℝ) (distance_upstream : ℝ) (total_time : ℝ) :
  current_speed = 4 → 
  boat_speed_still = 18 → 
  distance_upstream = 85.56 →
  total_time = 10 :=
by
  intros h_current h_boat h_distance
  sorry

end round_trip_time_l265_265047


namespace secant_length_l265_265108

theorem secant_length
  (A B C D E : ℝ)
  (AB : A - B = 7)
  (BC : B - C = 7)
  (AD : A - D = 10)
  (pos : A > E ∧ D > E):
  E - D = 0.2 :=
by
  sorry

end secant_length_l265_265108


namespace greatest_two_digit_multiple_of_17_l265_265206

theorem greatest_two_digit_multiple_of_17 : ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (17 ∣ n) ∧ n = 85 :=
by {
    use 85,
    split,
    { exact ⟨le_of_eq rfl, lt_of_le_of_ne (nat.le_of_eq rfl) (ne.symm (dec_trivial))⟩ },
    split,
    { exact dvd.intro 5 rfl },
    { refl }
}

end greatest_two_digit_multiple_of_17_l265_265206


namespace class_proof_l265_265338

def class_problem : Prop :=
  let total_students := 25
  let (students : Type) := {s // s ∈ Fin total_students}
  let good students (s : students) : Prop := sorry -- Condition Placeholder
  let troublemakers (s : students) : Prop := sorry -- Condition Placeholder
  have all_good_or_trouble : ∀ s, good students s ∨ troublemakers s,
    from sorry,
  have lh_trouble : ∀ (a b c d e : students), (good students a ∧ good students b ∧ good students c ∧ good students d ∧ good students e) →
    ∃ (x : ℕ), x > ((total_students - 1) / 2) ∧
    (λ rem_trouble : students - {a, b, c, d, e}, B : ℕ, B = x) ∧
    (λ rem_good : students - {a, b, c, d, e}, G : ℕ, G ∈ rem_good), from sorry,
  have th_triple_ratio : ∀ s, (students - {s}).card = 24 →
    ∃ (x y : ℕ), (troublemakers x ∧ good students y) ∧ (3*y = x), from sorry,
  ∃ (num_good_students : ℕ), (num_good_students = 5 ∨ num_good_students = 7)
  
theorem class_proof : class_problem :=
  sorry

end class_proof_l265_265338


namespace xy_problem_l265_265309

theorem xy_problem (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 :=
by
  sorry

end xy_problem_l265_265309


namespace arithmetic_sequence_sum_l265_265273

theorem arithmetic_sequence_sum {a : ℕ → ℤ} (S : ℕ → ℤ) 
  (h₀ : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2)
  (h₁ : S 9 = 27) :
  (a 4 + a 6) = 6 :=
sorry

end arithmetic_sequence_sum_l265_265273


namespace parabola_constant_c_l265_265660

theorem parabola_constant_c (b c : ℝ): 
  (∀ x : ℝ, y = x^2 + b * x + c) ∧ 
  (10 = 2^2 + b * 2 + c) ∧ 
  (31 = 4^2 + b * 4 + c) → 
  c = -3 :=
by
  sorry

end parabola_constant_c_l265_265660


namespace prob1_prob2_l265_265467

variables (x y a b c : ℝ)

-- Proof for the first problem
theorem prob1 :
  3 * x^2 * (-3 * x * y)^2 - x^2 * (x^2 * y^2 - 2 * x) = 26 * x^4 * y^2 + 2 * x^3 := 
sorry

-- Proof for the second problem
theorem prob2 :
  -2 * (-a^2 * b * c)^2 * (1 / 2) * a * (b * c)^3 - (-a * b * c)^3 * (-a * b * c)^2 = 0 :=
sorry

end prob1_prob2_l265_265467


namespace mixture_volume_correct_l265_265368

-- Define the input values
def water_volume : ℕ := 20
def vinegar_volume : ℕ := 18
def water_ratio : ℚ := 3/5
def vinegar_ratio : ℚ := 5/6

-- Calculate the mixture volume
def mixture_volume : ℚ :=
  (water_volume * water_ratio) + (vinegar_volume * vinegar_ratio)

-- Define the expected result
def expected_mixture_volume : ℚ := 27

-- State the theorem
theorem mixture_volume_correct : mixture_volume = expected_mixture_volume := by
  sorry

end mixture_volume_correct_l265_265368


namespace total_cantaloupes_l265_265264

def cantaloupes (fred : ℕ) (tim : ℕ) := fred + tim

theorem total_cantaloupes : cantaloupes 38 44 = 82 := by
  sorry

end total_cantaloupes_l265_265264


namespace number_of_good_students_is_5_or_7_l265_265355

-- Definitions based on the conditions
def total_students : ℕ := 25
def number_of_good_students (G : ℕ) (T : ℕ) := G + T = total_students
def first_group_condition (T : ℕ) := T > 12
def second_group_condition (G : ℕ) (T : ℕ) := T = 3 * (G - 1)

-- Problem statement in Lean 4:
theorem number_of_good_students_is_5_or_7 (G T : ℕ) :
  number_of_good_students G T ∧ first_group_condition T ∧ second_group_condition G T → G = 5 ∨ G = 7 :=
by
  sorry

end number_of_good_students_is_5_or_7_l265_265355


namespace calculation_proof_l265_265675

theorem calculation_proof : 441 + 2 * 21 * 7 + 49 = 784 := by
  sorry

end calculation_proof_l265_265675


namespace garden_path_width_l265_265073

theorem garden_path_width (R r : ℝ) (h : 2 * Real.pi * R - 2 * Real.pi * r = 20 * Real.pi) : R - r = 10 :=
by
  sorry

end garden_path_width_l265_265073


namespace total_books_to_read_l265_265039

theorem total_books_to_read (books_per_week : ℕ) (weeks : ℕ) (total_books : ℕ) 
  (h1 : books_per_week = 6) 
  (h2 : weeks = 5) 
  (h3 : total_books = books_per_week * weeks) : 
  total_books = 30 :=
by
  rw [h1, h2] at h3
  exact h3

end total_books_to_read_l265_265039


namespace exam_room_selection_l265_265325

theorem exam_room_selection (rooms : List ℕ) (n : ℕ) 
    (fifth_room_selected : 5 ∈ rooms) (twentyfirst_room_selected : 21 ∈ rooms) :
    rooms = [5, 13, 21, 29, 37, 45, 53, 61] → 
    37 ∈ rooms ∧ 53 ∈ rooms :=
by
  sorry

end exam_room_selection_l265_265325


namespace simplify_and_evaluate_l265_265044

def a : Int := 1
def b : Int := -2

theorem simplify_and_evaluate :
  ((a * b - 3 * a^2) - 2 * b^2 - 5 * a * b - (a^2 - 2 * a * b)) = -8 := by
  sorry

end simplify_and_evaluate_l265_265044


namespace sara_wrapping_paper_l265_265041

theorem sara_wrapping_paper (s : ℚ) (l : ℚ) (total : ℚ) : 
  total = 3 / 8 → 
  l = 2 * s →
  4 * s + 2 * l = total → 
  s = 3 / 64 :=
by
  intros h1 h2 h3
  sorry

end sara_wrapping_paper_l265_265041


namespace andrew_total_stickers_l265_265599

theorem andrew_total_stickers
  (total_stickers : ℕ)
  (ratio_susan: ℕ)
  (ratio_andrew: ℕ)
  (ratio_sam: ℕ)
  (sam_share_ratio: 2/3)
  (initial_andrew_share: ℕ)
  (initial_sam_share: ℕ) :
  total_stickers = 1500 →
  ratio_susan = 1 →
  ratio_andrew = 1 →
  ratio_sam = 3 →
  initial_andrew_share = (total_stickers / (ratio_susan + ratio_andrew + ratio_sam)) * ratio_andrew →
  initial_sam_share = (total_stickers / (ratio_susan + ratio_andrew + ratio_sam)) * ratio_sam →
  (ratio_andrew = (initial_andrew_share + initial_sam_share * sam_share_ratio)) →
  initial_andrew_share + initial_sam_share * sam_share_ratio = 900 :=
by
  intros _ _ _ _ _ _
  sorry

end andrew_total_stickers_l265_265599


namespace greatest_two_digit_multiple_of_17_l265_265200

theorem greatest_two_digit_multiple_of_17 : ∃ N, N = 85 ∧ 
  (∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n → n ≤ 85) :=
by 
  sorry

end greatest_two_digit_multiple_of_17_l265_265200


namespace jenny_money_l265_265524

theorem jenny_money (x : ℝ) (h : (4 / 7) * x = 24) : (x / 2) = 21 := 
sorry

end jenny_money_l265_265524


namespace find_a_and_b_l265_265837

theorem find_a_and_b (a b : ℤ) (h : ∀ x : ℝ, x ≤ 0 → (a*x + 2)*(x^2 + 2*b) ≤ 0) : a = 1 ∧ b = -2 := 
by 
  -- Proof steps would go here, but they are omitted as per instructions.
  sorry

end find_a_and_b_l265_265837


namespace equations_not_equivalent_l265_265251

theorem equations_not_equivalent :
  (∀ x, (2 * (x - 10) / (x^2 - 13 * x + 30) = 1 ↔ x = 5)) ∧ 
  (∃ x, x ≠ 5 ∧ (x^2 - 15 * x + 50 = 0)) :=
sorry

end equations_not_equivalent_l265_265251


namespace diagonals_intersection_probability_l265_265231

theorem diagonals_intersection_probability (decagon : Polygon) (h_regular : decagon.is_regular ∧ decagon.num_sides = 10) :
  probability_intersection_inside decagon = 42 / 119 := 
sorry

end diagonals_intersection_probability_l265_265231


namespace games_bought_at_garage_sale_l265_265732

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

end games_bought_at_garage_sale_l265_265732


namespace evie_gave_2_shells_to_brother_l265_265687

def daily_shells : ℕ := 10
def days : ℕ := 6
def remaining_shells : ℕ := 58

def total_shells : ℕ := daily_shells * days
def shells_given : ℕ := total_shells - remaining_shells

theorem evie_gave_2_shells_to_brother :
  shells_given = 2 :=
by
  sorry

end evie_gave_2_shells_to_brother_l265_265687


namespace find_common_difference_l265_265578

-- Define the arithmetic sequence and the sum of the first n terms
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
def sum_of_first_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ := ∑ k in finset.range n, arithmetic_sequence a₁ d (k + 1)

-- Given condition
def condition (a₁ d : ℝ) : Prop := 
  2 * sum_of_first_n_terms a₁ d 3 = 3 * sum_of_first_n_terms a₁ d 2 + 6

-- The proof statement
theorem find_common_difference (a₁ d : ℝ) (h : condition a₁ d) : d = 2 :=
by
  sorry

end find_common_difference_l265_265578


namespace second_sample_correct_l265_265455

def total_samples : ℕ := 7341
def first_sample : ℕ := 4221
def second_sample : ℕ := total_samples - first_sample

theorem second_sample_correct : second_sample = 3120 :=
by
  sorry

end second_sample_correct_l265_265455


namespace find_a_l265_265315

-- Given conditions
variables (x y z a : ℤ)

def conditions : Prop :=
  (x - 10) * (y - a) * (z - 2) = 1000 ∧
  ∃ (x y z : ℤ), x + y + z = 7

theorem find_a (x y z : ℤ) (h : conditions x y z 1) : a = 1 := 
  by
    sorry

end find_a_l265_265315


namespace greatest_two_digit_multiple_of_17_l265_265196

noncomputable def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem greatest_two_digit_multiple_of_17 : 
    ∃ n : ℕ, is_two_digit n ∧ (∃ k : ℕ, n = 17 * k) ∧
    ∀ m : ℕ, is_two_digit m → (∃ k : ℕ, m = 17 * k) → m ≤ n := 
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l265_265196


namespace green_shirt_pairs_l265_265859

theorem green_shirt_pairs (blue_shirts green_shirts total_pairs blue_blue_pairs : ℕ) 
(h1 : blue_shirts = 68) 
(h2 : green_shirts = 82) 
(h3 : total_pairs = 75) 
(h4 : blue_blue_pairs = 30) 
: (green_shirts - (blue_shirts - 2 * blue_blue_pairs)) / 2 = 37 := 
by 
  -- This is where the proof would be written, but we use sorry to skip it.
  sorry

end green_shirt_pairs_l265_265859


namespace photos_on_last_page_l265_265845

noncomputable def total_photos : ℕ := 10 * 35 * 4
noncomputable def photos_per_page_after_reorganization : ℕ := 8
noncomputable def total_pages_needed : ℕ := (total_photos + photos_per_page_after_reorganization - 1) / photos_per_page_after_reorganization
noncomputable def pages_filled_in_first_6_albums : ℕ := 6 * 35
noncomputable def last_page_photos : ℕ := if total_pages_needed ≤ pages_filled_in_first_6_albums then 0 else total_photos % photos_per_page_after_reorganization

theorem photos_on_last_page : last_page_photos = 0 :=
by
  sorry

end photos_on_last_page_l265_265845


namespace probability_white_ball_l265_265782

theorem probability_white_ball (num_black_balls num_white_balls : ℕ) 
  (black_balls : num_black_balls = 6) 
  (white_balls : num_white_balls = 5) : 
  (num_white_balls / (num_black_balls + num_white_balls) : ℚ) = 5 / 11 :=
by
  sorry

end probability_white_ball_l265_265782


namespace grandparents_to_parents_ratio_l265_265972

-- Definitions corresponding to the conditions
def wallet_cost : ℕ := 100
def betty_half_money : ℕ := wallet_cost / 2
def parents_contribution : ℕ := 15
def betty_needs_more : ℕ := 5
def grandparents_contribution : ℕ := 95 - (betty_half_money + parents_contribution)

-- The mathematical statement for the proof
theorem grandparents_to_parents_ratio :
  grandparents_contribution / parents_contribution = 2 := by
  sorry

end grandparents_to_parents_ratio_l265_265972


namespace avg_height_trees_l265_265642

-- Assuming heights are defined as h1, h2, ..., h7 with known h2
noncomputable def avgHeight (h1 h2 h3 h4 h5 h6 h7 : ℝ) : ℝ := 
  (h1 + h2 + h3 + h4 + h5 + h6 + h7) / 7

theorem avg_height_trees :
  ∃ (h1 h3 h4 h5 h6 h7 : ℝ), 
    h2 = 15 ∧ 
    (h1 = 2 * h2 ∨ h1 = 3 * h2) ∧
    (h3 = h2 / 3 ∨ h3 = h2 / 2) ∧
    (h4 = 2 * h3 ∨ h4 = 3 * h3 ∨ h4 = h3 / 2 ∨ h4 = h3 / 3) ∧
    (h5 = 2 * h4 ∨ h5 = 3 * h4 ∨ h5 = h4 / 2 ∨ h5 = h4 / 3) ∧
    (h6 = 2 * h5 ∨ h6 = 3 * h5 ∨ h6 = h5 / 2 ∨ h6 = h5 / 3) ∧
    (h7 = 2 * h6 ∨ h7 = 3 * h6 ∨ h7 = h6 / 2 ∨ h7 = h6 / 3) ∧
    avgHeight h1 h2 h3 h4 h5 h6 h7 = 26.4 :=
by
  sorry

end avg_height_trees_l265_265642


namespace least_clock_equiv_square_l265_265146

def clock_equiv (h k : ℕ) : Prop := (h - k) % 24 = 0

theorem least_clock_equiv_square : ∃ (h : ℕ), h > 6 ∧ (h^2) % 24 = h % 24 ∧ (∀ (k : ℕ), k > 6 ∧ clock_equiv k (k^2) → h ≤ k) :=
sorry

end least_clock_equiv_square_l265_265146


namespace common_difference_is_two_l265_265591

variable {a₁ a₂ a₃ S₃ S₂ : ℕ}
variable (d : ℕ)

-- Given condition
axiom H : 2 * S₃ = 3 * S₂ + 6

-- Definitions based on arithmetic sequence properties
def S₂ := a₁ + a₂
def S₃ := a₁ + a₂ + a₃
def a₂ := a₁ + d
def a₃ := a₁ + 2 * d

theorem common_difference_is_two : d = 2 := 
by 
  sorry

end common_difference_is_two_l265_265591


namespace greatest_two_digit_multiple_of_17_l265_265213

theorem greatest_two_digit_multiple_of_17 : ∃ m : ℕ, (10 ≤ m) ∧ (m ≤ 99) ∧ (17 ∣ m) ∧ (∀ n : ℕ, (10 ≤ n) ∧ (n ≤ 99) ∧ (17 ∣ n) → n ≤ m) ∧ m = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l265_265213


namespace greatest_two_digit_multiple_of_17_l265_265211

theorem greatest_two_digit_multiple_of_17 : ∃ m : ℕ, (10 ≤ m) ∧ (m ≤ 99) ∧ (17 ∣ m) ∧ (∀ n : ℕ, (10 ≤ n) ∧ (n ≤ 99) ∧ (17 ∣ n) → n ≤ m) ∧ m = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l265_265211


namespace expression_never_prime_l265_265059

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem expression_never_prime (p : ℕ) (hp : is_prime p) : ¬ is_prime (p^2 + 20) := sorry

end expression_never_prime_l265_265059


namespace initial_volume_of_mixture_l265_265959

theorem initial_volume_of_mixture (M W : ℕ) (h1 : 2 * M = 3 * W) (h2 : 4 * M = 3 * (W + 46)) : M + W = 115 := 
sorry

end initial_volume_of_mixture_l265_265959


namespace probability_non_defective_pens_l265_265225

theorem probability_non_defective_pens :
  let total_pens := 12
  let defective_pens := 6
  let non_defective_pens := total_pens - defective_pens
  let probability_first_non_defective := non_defective_pens / total_pens
  let probability_second_non_defective := (non_defective_pens - 1) / (total_pens - 1)
  (probability_first_non_defective * probability_second_non_defective = 5 / 22) :=
by
  rfl

end probability_non_defective_pens_l265_265225


namespace equal_red_B_black_C_l265_265856

theorem equal_red_B_black_C (a : ℕ) (h_even : a % 2 = 0) :
  ∃ (x y k j l i : ℕ), x + y = a ∧ y + i + j = a ∧ i + k = y ∧ k + j = x ∧ i = k := 
  sorry

end equal_red_B_black_C_l265_265856


namespace fraction_identity_l265_265092

def at_op (a b : ℤ) : ℤ := a * b - 3 * b ^ 2
def hash_op (a b : ℤ) : ℤ := a + 2 * b - 2 * a * b ^ 2

theorem fraction_identity : (at_op 8 3) / (hash_op 8 3) = 3 / 130 := by
  sorry

end fraction_identity_l265_265092


namespace class_proof_l265_265339

def class_problem : Prop :=
  let total_students := 25
  let (students : Type) := {s // s ∈ Fin total_students}
  let good students (s : students) : Prop := sorry -- Condition Placeholder
  let troublemakers (s : students) : Prop := sorry -- Condition Placeholder
  have all_good_or_trouble : ∀ s, good students s ∨ troublemakers s,
    from sorry,
  have lh_trouble : ∀ (a b c d e : students), (good students a ∧ good students b ∧ good students c ∧ good students d ∧ good students e) →
    ∃ (x : ℕ), x > ((total_students - 1) / 2) ∧
    (λ rem_trouble : students - {a, b, c, d, e}, B : ℕ, B = x) ∧
    (λ rem_good : students - {a, b, c, d, e}, G : ℕ, G ∈ rem_good), from sorry,
  have th_triple_ratio : ∀ s, (students - {s}).card = 24 →
    ∃ (x y : ℕ), (troublemakers x ∧ good students y) ∧ (3*y = x), from sorry,
  ∃ (num_good_students : ℕ), (num_good_students = 5 ∨ num_good_students = 7)
  
theorem class_proof : class_problem :=
  sorry

end class_proof_l265_265339


namespace net_calorie_deficit_l265_265984

-- Define the conditions as constants.
def total_distance : ℕ := 3
def calories_burned_per_mile : ℕ := 150
def calories_in_candy_bar : ℕ := 200

-- Prove the net calorie deficit.
theorem net_calorie_deficit : total_distance * calories_burned_per_mile - calories_in_candy_bar = 250 := by
  sorry

end net_calorie_deficit_l265_265984


namespace smallest_omega_l265_265283

theorem smallest_omega (ω : ℝ) (h_pos : ω > 0) :
  (∃ k : ℤ, ω = 6 * k) ∧ (∀ k : ℤ, k > 0 → ω = 6 * k → ω = 6) :=
by sorry

end smallest_omega_l265_265283


namespace vasya_gift_choices_l265_265742

theorem vasya_gift_choices :
  let cars := 7
  let construction_sets := 5
  (cars * construction_sets + Nat.choose cars 2 + Nat.choose construction_sets 2) = 66 :=
by
  sorry

end vasya_gift_choices_l265_265742


namespace ellipse_meets_sine_more_than_8_points_l265_265669

noncomputable def ellipse_intersects_sine_curve_more_than_8_times (a b : ℝ) (h k : ℝ) :=
  ∃ p : ℕ, p > 8 ∧ 
  ∃ (x y : ℝ), 
    (∃ (i : ℕ), y = Real.sin x ∧ 
    (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1)

theorem ellipse_meets_sine_more_than_8_points : 
  ∀ (a b h k : ℝ), ellipse_intersects_sine_curve_more_than_8_times a b h k := 
by sorry

end ellipse_meets_sine_more_than_8_points_l265_265669


namespace find_f_neg_6_l265_265239

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  if x ≥ 0 then Real.log (x + 2) / Real.log 2 + (a - 1) * x + b else -(Real.log (-x + 2) / Real.log 2 + (a - 1) * -x + b)

theorem find_f_neg_6 (a b : ℝ) (h1 : ∀ x : ℝ, f x a b = -f (-x) a b) 
                     (h2 : ∀ x : ℝ, x ≥ 0 → f x a b = Real.log (x + 2) / Real.log 2 + (a - 1) * x + b)
                     (h3 : f 2 a b = -1) : f (-6) 0 (-1) = 4 :=
by
  sorry

end find_f_neg_6_l265_265239


namespace total_rope_length_l265_265253

theorem total_rope_length 
  (longer_side : ℕ) (shorter_side : ℕ) 
  (h1 : longer_side = 28) (h2 : shorter_side = 22) : 
  2 * longer_side + 2 * shorter_side = 100 := by
  sorry

end total_rope_length_l265_265253


namespace neg_p_l265_265897

variable (x : ℝ)

def p : Prop := ∃ x_0 : ℝ, x_0^2 + x_0 + 2 ≤ 0

theorem neg_p : ¬p ↔ ∀ x : ℝ, x^2 + x + 2 > 0 := by
  sorry

end neg_p_l265_265897


namespace sabina_loan_amount_l265_265395

-- Definitions corresponding to the problem conditions
def tuition_cost : ℕ := 30000
def sabina_savings : ℕ := 10000
def grant_percentage : ℚ := 0.40

-- Proof statement to show the amount of loan required
theorem sabina_loan_amount : ∀ (tuition_cost sabina_savings : ℕ) (grant_percentage : ℚ), 
  sabina_savings < tuition_cost →
  let remainder := tuition_cost - sabina_savings in
  let grant_amount := grant_percentage * remainder in
  let loan_amount := remainder - grant_amount in
  loan_amount = 12000 :=
by 
  intros tuition_cost sabina_savings grant_percentage h_savings_lt;
  let remainder := tuition_cost - sabina_savings;
  let grant_amount := grant_percentage * ↑remainder;
  let loan_amount := remainder - grant_amount.to_nat;
  sorry

end sabina_loan_amount_l265_265395


namespace passes_through_1_1_l265_265313

theorem passes_through_1_1 (a : ℝ) (h_pos : a > 0) (h_ne : a ≠ 1) : (1, 1) ∈ {p : ℝ × ℝ | ∃ x : ℝ, p = (x, a^ (x - 1))} :=
by
  -- proof not required
  sorry

end passes_through_1_1_l265_265313


namespace degree_le_three_l265_265097

theorem degree_le_three
  (d : ℕ)
  (P : Polynomial ℤ)
  (hdeg : P.degree = d)
  (hP : ∃ (S : Finset ℤ), (S.card ≥ d + 1) ∧ ∀ m ∈ S, |P.eval m| = 1) :
  d ≤ 3 := 
sorry

end degree_le_three_l265_265097


namespace find_interest_rate_l265_265019

def interest_rate_borrowed (p_borrowed: ℝ) (p_lent: ℝ) (time: ℝ) (rate_lent: ℝ) (gain: ℝ) (r: ℝ) : Prop :=
  let interest_from_ramu := p_lent * rate_lent * time / 100
  let interest_to_anwar := p_borrowed * r * time / 100
  gain = interest_from_ramu - interest_to_anwar

theorem find_interest_rate :
  interest_rate_borrowed 3900 5655 3 9 824.85 5.95 := sorry

end find_interest_rate_l265_265019


namespace union_sets_l265_265701

def setA : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def setB : Set ℝ := { x | 0 < x ∧ x < 3 }

theorem union_sets :
  (setA ∪ setB) = { x | -1 ≤ x ∧ x < 3 } :=
sorry

end union_sets_l265_265701


namespace ticket_cost_l265_265451

theorem ticket_cost
    (rows : ℕ) (seats_per_row : ℕ)
    (fraction_sold : ℚ) (total_earnings : ℚ)
    (N : ℕ := rows * seats_per_row)
    (S : ℚ := fraction_sold * N)
    (C : ℚ := total_earnings / S)
    (h1 : rows = 20) (h2 : seats_per_row = 10)
    (h3 : fraction_sold = 3 / 4) (h4 : total_earnings = 1500) :
    C = 10 :=
by
  sorry

end ticket_cost_l265_265451


namespace half_of_original_amount_l265_265522

theorem half_of_original_amount (h : ∃ (m : ℚ), (4/7 : ℚ) * m = 24) : 
  ∃ (half_m : ℚ), half_m = 21 :=
by
  obtain ⟨m, hm⟩ := h
  have original := m
  have half_orig := (1/2 : ℚ) * original
  have target := (7/4 : ℚ) * 24 / 2
  use half_orig
  rw [←hm]
  have fact : (4 / 7) * original * (7 / 4) = original := by sorry
  have eq1 : (7 / 4) * 24 = original := eq.trans (mul_eq_mul_right_iff.mpr (oreq_of_ne_zero (by norm_num)) (by norm_num) hm.symm)
  have eq2 := eq.trans eq1 div_eq_div_right_iff nonzero_of_ne_zero (by norm_num)
  rw [eq2] at this
  exact sorry


end half_of_original_amount_l265_265522


namespace arithmetic_sequence_sum_l265_265419

theorem arithmetic_sequence_sum (c d : ℕ) 
  (h1 : ∀ (a1 a2 a3 a4 a5 a6 : ℕ), a1 = 3 → a2 = 10 → a3 = 17 → a6 = 38 → (a2 - a1 = a3 - a2) → (a3 - a2 = c - a3) → (c - a3 = d - c) → (d - c = a6 - d)) : 
  c + d = 55 := 
by 
  sorry

end arithmetic_sequence_sum_l265_265419


namespace arithmetic_sequence_common_difference_l265_265543

theorem arithmetic_sequence_common_difference 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : S 3 = a 1 + a 2 + a 3)
  (h2 : S 2 = a 1 + a 2)
  (h3 : a 2 = a 1 + d)
  (h4 : a 3 = a 1 + 2 * d)
  (h5 : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l265_265543


namespace proof_problem_l265_265136

noncomputable def question (a b c : ℝ) : ℝ := 
  (a ^ 2 * b ^ 2) / ((a ^ 2 + b * c) * (b ^ 2 + a * c)) +
  (a ^ 2 * c ^ 2) / ((a ^ 2 + b * c) * (c ^ 2 + a * b)) +
  (b ^ 2 * c ^ 2) / ((b ^ 2 + a * c) * (c ^ 2 + a * b))

theorem proof_problem (a b c : ℝ) (h : a ≠ 0) (h1 : b ≠ 0) (h2 : c ≠ 0) 
  (h3 : a ^ 2 + b ^ 2 + c ^ 2 = a * b + b * c + c * a ) : 
  question a b c = 1 := 
by 
  sorry

end proof_problem_l265_265136


namespace factorize_expression_l265_265480

theorem factorize_expression (x y : ℝ) : 3 * x^2 - 12 * y^2 = 3 * (x - 2 * y) * (x + 2 * y) := by
  sorry

end factorize_expression_l265_265480


namespace recurring_fraction_difference_l265_265977

theorem recurring_fraction_difference :
  let x := (36 / 99 : ℚ)
  let y := (36 / 100 : ℚ)
  x - y = (1 / 275 : ℚ) :=
by
  sorry

end recurring_fraction_difference_l265_265977


namespace circle_radius_10_l265_265106

theorem circle_radius_10 (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 14 * x + y^2 + 8 * y - k = 0 → (x + 7) ^ 2 + (y + 4) ^ 2 = 100) ↔ (k = 35) :=
begin
  sorry
end

end circle_radius_10_l265_265106


namespace max_consecutive_sum_le_1000_l265_265767

theorem max_consecutive_sum_le_1000 : 
  ∃ n : ℕ, (∀ m : ℕ, m ≤ n → (m * (m + 1)) / 2 < 1000) ∧ ¬∃ n' : ℕ, n < n' ∧ (n' * (n' + 1)) / 2 < 1000 :=
sorry

end max_consecutive_sum_le_1000_l265_265767


namespace no_valid_transformation_l265_265228

theorem no_valid_transformation :
  ¬ ∃ (n1 n2 n3 n4 : ℤ),
    2 * n1 + n2 - 2 * n3 - n4 = 27 ∧
    -n1 + 2 * n2 + n3 - 2 * n4 = -27 :=
by
  sorry

end no_valid_transformation_l265_265228


namespace solution_set_of_inequality_l265_265827

theorem solution_set_of_inequality (a : ℝ) (h : a < 0) :
  {x : ℝ | x^2 - 2 * a * x - 3 * a^2 < 0} = {x | 3 * a < x ∧ x < -a} :=
sorry

end solution_set_of_inequality_l265_265827


namespace father_age_l265_265791

theorem father_age : 
  ∀ (S F : ℕ), (S - 5 = 11) ∧ (F - S = S) → F = 32 := 
by
  intros S F h
  -- Use the conditions to derive further equations and steps
  sorry

end father_age_l265_265791


namespace decrease_in_B_share_l265_265884

theorem decrease_in_B_share (a b c : ℝ) (x : ℝ) 
  (h1 : c = 495)
  (h2 : a + b + c = 1010)
  (h3 : (a - 25) / 3 = (b - x) / 2)
  (h4 : (a - 25) / 3 = (c - 15) / 5) :
  x = 10 :=
by
  sorry

end decrease_in_B_share_l265_265884


namespace rectangle_area_l265_265077

theorem rectangle_area (x : ℝ) (w : ℝ) (h_diag : (3 * w) ^ 2 + w ^ 2 = x ^ 2) : 
  3 * w ^ 2 = (3 / 10) * x ^ 2 :=
by
  sorry

end rectangle_area_l265_265077


namespace min_value_a_plus_8b_min_value_a_plus_8b_min_l265_265109

theorem min_value_a_plus_8b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a * b = a + 2 * b) :
  a + 8 * b ≥ 9 :=
by sorry

-- The minimum value is 9 (achievable at specific values of a and b)
theorem min_value_a_plus_8b_min (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a * b = a + 2 * b) :
  ∃ a b, a > 0 ∧ b > 0 ∧ 2 * a * b = a + 2 * b ∧ a + 8 * b = 9 :=
by sorry

end min_value_a_plus_8b_min_value_a_plus_8b_min_l265_265109


namespace sandbox_width_l265_265962

theorem sandbox_width :
  ∀ (length area width : ℕ), length = 312 → area = 45552 →
  area = length * width → width = 146 :=
by
  intros length area width h_length h_area h_eq
  sorry

end sandbox_width_l265_265962


namespace boat_and_current_speed_boat_and_current_speed_general_log_drift_time_l265_265949

-- Problem 1: Specific case
theorem boat_and_current_speed (x y : ℝ) 
  (h1 : 3 * (x + y) = 75) 
  (h2 : 5 * (x - y) = 75) : 
  x = 20 ∧ y = 5 := 
sorry

-- Problem 2: General case
theorem boat_and_current_speed_general (x y : ℝ) (a b S : ℝ) 
  (h1 : a * (x + y) = S) 
  (h2 : b * (x - y) = S) : 
  x = (a + b) * S / (2 * a * b) ∧ 
  y = (b - a) * S / (2 * a * b) := 
sorry

theorem log_drift_time (y S a b : ℝ)
  (h_y : y = (b - a) * S / (2 * a * b)) : 
  S / y = 2 * a * b / (b - a) := 
sorry

end boat_and_current_speed_boat_and_current_speed_general_log_drift_time_l265_265949


namespace desired_gain_percentage_l265_265674

theorem desired_gain_percentage (cp16 sp16 cp12881355932203391 sp12881355932203391 : ℝ) :
  sp16 = 1 →
  sp16 = 0.95 * cp16 →
  sp12881355932203391 = 1 →
  cp12881355932203391 = (12.881355932203391 / 16) * cp16 →
  (sp12881355932203391 - cp12881355932203391) / cp12881355932203391 * 100 = 18.75 :=
by sorry

end desired_gain_percentage_l265_265674


namespace vasya_gift_ways_l265_265740

theorem vasya_gift_ways :
  let cars := 7
  let constructor_sets := 5
  (cars * constructor_sets) + (Nat.choose cars 2) + (Nat.choose constructor_sets 2) = 66 :=
by
  let cars := 7
  let constructor_sets := 5
  sorry

end vasya_gift_ways_l265_265740


namespace intersection_A_B_intersection_CR_A_B_l265_265702

noncomputable def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 7}
noncomputable def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}
noncomputable def CR_A : Set ℝ := {x : ℝ | x < 3} ∪ {x : ℝ | 7 ≤ x}

theorem intersection_A_B :
  A ∩ B = {x : ℝ | 3 ≤ x ∧ x < 7} :=
by
  sorry

theorem intersection_CR_A_B :
  CR_A ∩ B = ({x : ℝ | 2 < x ∧ x < 3} ∪ {x : ℝ | 7 ≤ x ∧ x < 10}) :=
by
  sorry

end intersection_A_B_intersection_CR_A_B_l265_265702


namespace greatest_two_digit_multiple_of_17_l265_265179

theorem greatest_two_digit_multiple_of_17 : ∃ (n : ℕ), n < 100 ∧ 10 ≤ n ∧ 17 ∣ n ∧ ∀ m : ℕ, m < 100 → 10 ≤ m → 17 ∣ m → m ≤ n :=
begin
  use 85,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m hm1 hm2 hm3,
    by_contradiction hmn,
    have h : m > 85 := not_le.mp hmn,
    linear_combination h * 17,
    sorry,
  },
end

end greatest_two_digit_multiple_of_17_l265_265179


namespace dot_product_theorem_l265_265844

open Real

namespace VectorProof

-- Define the vectors m and n
def m := (2, 5)
def n (t : ℝ) := (-5, t)

-- Define the condition that m is perpendicular to n
def perpendicular (t : ℝ) : Prop := (2 * -5) + (5 * t) = 0

-- Function to calculate the dot product
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Define the vectors m+n and m-2n
def vector_add (t : ℝ) : ℝ × ℝ := (m.1 + (n t).1, m.2 + (n t).2)
def vector_sub (t : ℝ) : ℝ × ℝ := (m.1 - 2 * (n t).1, m.2 - 2 * (n t).2)

-- The theorem to prove
theorem dot_product_theorem : ∀ (t : ℝ), perpendicular t → dot_product (vector_add t) (vector_sub t) = -29 :=
by
  intros t ht
  sorry

end VectorProof

end dot_product_theorem_l265_265844


namespace inverse_of_p_l265_265853

variables {p q r : Prop}

theorem inverse_of_p (m n : Prop) (hp : p = (m → n)) (hq : q = (¬m → ¬n)) (hr : r = (n → m)) : r = p ∧ r = (n → m) :=
by
  sorry

end inverse_of_p_l265_265853


namespace puppies_brought_in_l265_265244

open Nat

theorem puppies_brought_in (orig_puppies adopt_rate days total_adopted brought_in_puppies : ℕ) 
  (h_orig : orig_puppies = 3)
  (h_adopt_rate : adopt_rate = 3)
  (h_days : days = 2)
  (h_total_adopted : total_adopted = adopt_rate * days)
  (h_equation : total_adopted = orig_puppies + brought_in_puppies) :
  brought_in_puppies = 3 :=
by
  sorry

end puppies_brought_in_l265_265244


namespace horner_evaluation_l265_265621

def f (x : ℝ) := x^5 + 3 * x^4 - 5 * x^3 + 7 * x^2 - 9 * x + 11

theorem horner_evaluation : f 4 = 1559 := by
  sorry

end horner_evaluation_l265_265621


namespace good_students_l265_265333

theorem good_students (E B : ℕ) (h1 : E + B = 25) (h2 : 12 < B) (h3 : B = 3 * (E - 1)) :
  E = 5 ∨ E = 7 :=
by 
  sorry

end good_students_l265_265333


namespace common_difference_arithmetic_sequence_l265_265549

theorem common_difference_arithmetic_sequence
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (d : ℕ) 
  (h₁ : ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2) -- sum formula for arithmetic sequence
  (h₂ : 2 * S 3 = 3 * S 2 + 6) : 
  d = 2 := 
sorry

end common_difference_arithmetic_sequence_l265_265549


namespace no_solution_inequality_l265_265710

theorem no_solution_inequality (m : ℝ) : ¬(∃ x : ℝ, 2 * x - 1 > 1 ∧ x < m) → m ≤ 1 :=
by
  intro h
  sorry

end no_solution_inequality_l265_265710


namespace evaluate_f_l265_265318

def f (x : ℝ) : ℝ := x^5 + 3 * x^3 + 4 * x

theorem evaluate_f (h : f 3 - f (-3) = 672) : True :=
by
  sorry

end evaluate_f_l265_265318


namespace product_positive_probability_l265_265931

theorem product_positive_probability :
  let interval := set.Icc (-30 : ℝ) 15 in
  let prob_neg := (30 : ℝ) / 45 in
  let prob_pos := (15 : ℝ) / 45 in
  let prob_product_neg := 2 * (prob_neg * prob_pos) in
  let prob_product_pos := (prob_neg ^ 2) + (prob_pos ^ 2) in
  (prob_product_pos = 5 / 9) :=
by
  sorry

end product_positive_probability_l265_265931


namespace arrange_moon_l265_265815

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def ways_to_arrange_moon : ℕ :=
  let total_letters := 4
  let repeated_O_count := 2
  factorial total_letters / factorial repeated_O_count

theorem arrange_moon : ways_to_arrange_moon = 12 := 
by {
  sorry -- Proof is omitted as instructed
}

end arrange_moon_l265_265815


namespace series_inequality_l265_265149

open BigOperators

theorem series_inequality :
  (∑ k in Finset.range 2012, (1 / (((k + 1) * Real.sqrt k) + (k * Real.sqrt (k + 1))))) > 0.97 :=
sorry

end series_inequality_l265_265149


namespace common_difference_of_arithmetic_sequence_l265_265532

noncomputable def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range n, a i

def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_cond : 2 * S a 3 = 3 * S a 2 + 6) :
  ∃ d : ℝ, d = 2 := sorry

end common_difference_of_arithmetic_sequence_l265_265532


namespace problem_statement_l265_265290

-- Define the conditions and the goal
theorem problem_statement {x y : ℝ} 
  (h1 : (x + y)^2 = 36)
  (h2 : x * y = 8) : 
  x^2 + y^2 = 20 := 
by
  sorry

end problem_statement_l265_265290


namespace find_common_difference_l265_265577

-- Define the arithmetic sequence and the sum of the first n terms
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
def sum_of_first_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ := ∑ k in finset.range n, arithmetic_sequence a₁ d (k + 1)

-- Given condition
def condition (a₁ d : ℝ) : Prop := 
  2 * sum_of_first_n_terms a₁ d 3 = 3 * sum_of_first_n_terms a₁ d 2 + 6

-- The proof statement
theorem find_common_difference (a₁ d : ℝ) (h : condition a₁ d) : d = 2 :=
by
  sorry

end find_common_difference_l265_265577


namespace soda_choosers_l265_265512

-- Definitions based on conditions
def total_people := 600
def soda_angle := 108
def full_circle := 360

-- Statement to prove the number of people who referred to soft drinks as "Soda"
theorem soda_choosers : total_people * (soda_angle / full_circle) = 180 :=
by
  sorry

end soda_choosers_l265_265512


namespace fourth_root_of_25000000_eq_70_7_l265_265087

theorem fourth_root_of_25000000_eq_70_7 :
  Real.sqrt (Real.sqrt 25000000) = 70.7 :=
sorry

end fourth_root_of_25000000_eq_70_7_l265_265087


namespace arithmetic_sequence_common_difference_l265_265544

theorem arithmetic_sequence_common_difference 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : S 3 = a 1 + a 2 + a 3)
  (h2 : S 2 = a 1 + a 2)
  (h3 : a 2 = a 1 + d)
  (h4 : a 3 = a 1 + 2 * d)
  (h5 : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l265_265544


namespace greatest_two_digit_multiple_of_17_l265_265202

theorem greatest_two_digit_multiple_of_17 : ∃ N, N = 85 ∧ 
  (∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n → n ≤ 85) :=
by 
  sorry

end greatest_two_digit_multiple_of_17_l265_265202


namespace magician_assistant_strategy_l265_265654

-- Define the possible states of a coin
inductive CoinState | heads | tails

-- Define the circle of coins
def coinCircle := Fin 11 → CoinState

-- The strategy ensures there are adjacent coins with the same state
theorem magician_assistant_strategy (c : coinCircle) :
  ∃ i : Fin 11, c i = c ((i + 1) % 11) := by sorry

end magician_assistant_strategy_l265_265654


namespace common_difference_l265_265562

variable (a1 d : ℤ)
variable (S : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def sum_first_n_terms (n : ℕ) : ℤ :=
  n * a1 + d * (n * (n - 1) / 2)

-- Condition: 2 * S 3 = 3 * S 2 + 6
axiom cond : 2 * sum_first_n_terms 3 = 3 * sum_first_n_terms 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem common_difference : d = 2 :=
by
  sorry

end common_difference_l265_265562


namespace games_bought_at_garage_sale_l265_265730

theorem games_bought_at_garage_sale (G : ℕ)
  (h1 : 2 + G - 2  = 2) :
  G = 2 :=
by {
  sorry
}

end games_bought_at_garage_sale_l265_265730


namespace daps_equivalent_to_dips_l265_265316

-- Definitions from conditions
def daps (n : ℕ) : ℕ := n
def dops (n : ℕ) : ℕ := n
def dips (n : ℕ) : ℕ := n

-- Given conditions
def equivalence_daps_dops : daps 8 = dops 6 := sorry
def equivalence_dops_dips : dops 3 = dips 11 := sorry

-- Proof problem
theorem daps_equivalent_to_dips (n : ℕ) (h1 : daps 8 = dops 6) (h2 : dops 3 = dips 11) : daps 24 = dips 66 :=
sorry

end daps_equivalent_to_dips_l265_265316


namespace cristina_pace_is_4_l265_265029

-- Definitions of the conditions
def head_start : ℝ := 36
def nicky_pace : ℝ := 3
def time : ℝ := 36

-- Definition of the distance Nicky runs
def distance_nicky_runs : ℝ := nicky_pace * time

-- Definition of the total distance Cristina ran to catch up
def distance_cristina_runs : ℝ := distance_nicky_runs + head_start

-- Lean 4 theorem statement to prove Cristina's pace
theorem cristina_pace_is_4 :
  (distance_cristina_runs / time) = 4 := 
by sorry

end cristina_pace_is_4_l265_265029


namespace greatest_two_digit_multiple_of_17_l265_265172

theorem greatest_two_digit_multiple_of_17 : ∀ n, n ∈ finset.range 100 → n % 17 = 0 → n ≤ 85 →
  (∀ m, m ∈ finset.range 100 → m % 17 = 0 → m > n → m ≥ 100) →
  n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l265_265172


namespace greatest_two_digit_multiple_of_17_l265_265192

theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n < 100 ∧ n ≥ 10 ∧ 17 ∣ n ∧ ∀ m, m < 100 ∧ m ≥ 10 ∧ 17 ∣ m → m ≤ 85 :=
by
  use 85
  -- Prove conditions follow sorry
  sorry

end greatest_two_digit_multiple_of_17_l265_265192


namespace probability_product_positive_is_5_div_9_l265_265927

noncomputable def probability_positive_product : ℚ :=
  let interval := Set.Icc (-30 : ℝ) 15
  let length_interval := 45
  let length_neg := 30
  let length_pos := 15
  let prob_neg := (length_neg : ℚ) / length_interval
  let prob_pos := (length_pos : ℚ) / length_interval
  let prob_product_pos := prob_neg^2 + prob_pos^2
  prob_product_pos

theorem probability_product_positive_is_5_div_9 :
  probability_positive_product = 5 / 9 :=
by
  sorry

end probability_product_positive_is_5_div_9_l265_265927


namespace xy_problem_l265_265304

theorem xy_problem (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 :=
by
  sorry

end xy_problem_l265_265304


namespace magician_trick_l265_265658

def coin (a : Fin 11 → Bool) : Prop :=
  ∃ i : Fin 11, a i = a (i + 1) % 11

theorem magician_trick (a : Fin 11 → Bool) 
(H : ∃ i : Fin 11, a i = a (i + 1) % 11) : 
  ∃ j : Fin 11, j ≠ 0 ∧ a j = a 0 :=
sorry

end magician_trick_l265_265658


namespace prove_product_of_b_l265_265684

noncomputable def g (x b : ℝ) := b / (5 * x - 7)

noncomputable def g_inv (y b : ℝ) := (b + 7 * y) / (5 * y)

theorem prove_product_of_b (b1 b2 : ℝ) (h1 : g 3 b1 = g_inv (b1 + 2) b1) (h2 : g 3 b2 = g_inv (b2 + 2) b2) :
  b1 * b2 = -22.39 := by
  sorry

end prove_product_of_b_l265_265684


namespace integer_part_of_result_is_40_l265_265753

noncomputable def numerator : ℝ := 0.1 + 1.2 + 2.3 + 3.4 + 4.5 + 5.6 + 6.7 + 7.8 + 8.9
noncomputable def denominator : ℝ := 0.01 + 0.03 + 0.05 + 0.07 + 0.09 + 0.11 + 0.13 + 0.15 + 0.17 + 0.19
noncomputable def result : ℝ := numerator / denominator

theorem integer_part_of_result_is_40 : ⌊result⌋ = 40 := 
by
  -- proof goes here
  sorry

end integer_part_of_result_is_40_l265_265753


namespace avg_goals_per_game_l265_265678

def carter_goals_per_game := 4
def shelby_goals_per_game := carter_goals_per_game / 2
def judah_goals_per_game := (2 * shelby_goals_per_game) - 3
def average_total_goals_team := carter_goals_per_game + shelby_goals_per_game + judah_goals_per_game

theorem avg_goals_per_game : average_total_goals_team = 7 :=
by
  -- Proof would go here
  sorry

end avg_goals_per_game_l265_265678


namespace factorial_equation_solution_unique_l265_265825

theorem factorial_equation_solution_unique :
  ∀ a b c : ℕ, (0 < a ∧ 0 < b ∧ 0 < c) →
  (a.factorial * b.factorial = a.factorial + b.factorial + c.factorial) →
  (a = 3 ∧ b = 3 ∧ c = 4) := 
by
  intros a b c h_positive h_eq
  sorry

end factorial_equation_solution_unique_l265_265825


namespace average_gas_mileage_round_trip_l265_265080

theorem average_gas_mileage_round_trip
  (d : ℝ) (ms mr : ℝ)
  (h1 : d = 150)
  (h2 : ms = 35)
  (h3 : mr = 15) :
  (2 * d) / ((d / ms) + (d / mr)) = 21 :=
by
  sorry

end average_gas_mileage_round_trip_l265_265080


namespace find_number_of_good_students_l265_265330

variable {G T : ℕ}

def is_good_or_troublemaker (G T : ℕ) : Prop :=
  G + T = 25 ∧ 
  (∀ s ∈ finset.range 5, T > 12) ∧ 
  (∀ s ∈ finset.range 20, T = 3 * (G - 1))

-- There are 25 students in total. 
-- Each student is either a good student or a troublemaker.
-- Good students always tell the truth, while troublemakers always lie.
-- Five students make the statement: "If I transfer to another class, more than half of the remaining students will be troublemakers."
-- The remaining 20 students make the statement: "If I transfer to another class, the number of troublemakers among the remaining students will be three times the number of good students."

theorem find_number_of_good_students (G T : ℕ) (h : is_good_or_troublemaker G T) :
  (G = 5 ∨ G = 7) :=
sorry

end find_number_of_good_students_l265_265330


namespace common_difference_arithmetic_sequence_l265_265548

theorem common_difference_arithmetic_sequence
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (d : ℕ) 
  (h₁ : ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2) -- sum formula for arithmetic sequence
  (h₂ : 2 * S 3 = 3 * S 2 + 6) : 
  d = 2 := 
sorry

end common_difference_arithmetic_sequence_l265_265548


namespace class_proof_l265_265337

def class_problem : Prop :=
  let total_students := 25
  let (students : Type) := {s // s ∈ Fin total_students}
  let good students (s : students) : Prop := sorry -- Condition Placeholder
  let troublemakers (s : students) : Prop := sorry -- Condition Placeholder
  have all_good_or_trouble : ∀ s, good students s ∨ troublemakers s,
    from sorry,
  have lh_trouble : ∀ (a b c d e : students), (good students a ∧ good students b ∧ good students c ∧ good students d ∧ good students e) →
    ∃ (x : ℕ), x > ((total_students - 1) / 2) ∧
    (λ rem_trouble : students - {a, b, c, d, e}, B : ℕ, B = x) ∧
    (λ rem_good : students - {a, b, c, d, e}, G : ℕ, G ∈ rem_good), from sorry,
  have th_triple_ratio : ∀ s, (students - {s}).card = 24 →
    ∃ (x y : ℕ), (troublemakers x ∧ good students y) ∧ (3*y = x), from sorry,
  ∃ (num_good_students : ℕ), (num_good_students = 5 ∨ num_good_students = 7)
  
theorem class_proof : class_problem :=
  sorry

end class_proof_l265_265337


namespace no_same_last_four_digits_of_powers_of_five_and_six_l265_265818

theorem no_same_last_four_digits_of_powers_of_five_and_six : 
  ¬ ∃ (n m : ℕ), 0 < n ∧ 0 < m ∧ (5 ^ n % 10000 = 6 ^ m % 10000) := 
by 
  sorry

end no_same_last_four_digits_of_powers_of_five_and_six_l265_265818


namespace fish_population_estimate_l265_265226

theorem fish_population_estimate
  (N : ℕ) 
  (tagged_initial : ℕ)
  (caught_again : ℕ)
  (tagged_again : ℕ)
  (h1 : tagged_initial = 60)
  (h2 : caught_again = 60)
  (h3 : tagged_again = 2)
  (h4 : (tagged_initial : ℚ) / N = (tagged_again : ℚ) / caught_again) :
  N = 1800 :=
by
  sorry

end fish_population_estimate_l265_265226


namespace system_of_equations_solution_l265_265889

theorem system_of_equations_solution :
  ∃ (x1 x2 x3 x4 x5 : ℝ),
  (x1 + 2 * x2 + 2 * x3 + 2 * x4 + 2 * x5 = 1) ∧
  (x1 + 3 * x2 + 4 * x3 + 4 * x4 + 4 * x5 = 2) ∧
  (x1 + 3 * x2 + 5 * x3 + 6 * x4 + 6 * x5 = 3) ∧
  (x1 + 3 * x2 + 5 * x3 + 7 * x4 + 8 * x5 = 4) ∧
  (x1 + 3 * x2 + 5 * x3 + 7 * x4 + 9 * x5 = 5) ∧
  (x1 = 1) ∧ (x2 = -1) ∧ (x3 = 1) ∧ (x4 = -1) ∧ (x5 = 1) := by
sorry

end system_of_equations_solution_l265_265889


namespace company_stores_l265_265788

theorem company_stores (total_uniforms : ℕ) (uniforms_per_store : ℕ) 
  (h1 : total_uniforms = 121) (h2 : uniforms_per_store = 4) : 
  total_uniforms / uniforms_per_store = 30 :=
by
  sorry

end company_stores_l265_265788


namespace new_boxes_of_markers_l265_265388

theorem new_boxes_of_markers (initial_markers new_markers_per_box total_markers : ℕ) 
    (h_initial : initial_markers = 32) 
    (h_new_markers_per_box : new_markers_per_box = 9)
    (h_total : total_markers = 86) :
  (total_markers - initial_markers) / new_markers_per_box = 6 := 
by
  sorry

end new_boxes_of_markers_l265_265388


namespace esther_commute_distance_l265_265996

theorem esther_commute_distance (D : ℕ) :
  (D / 45 + D / 30 = 1) → D = 18 :=
by
  sorry

end esther_commute_distance_l265_265996


namespace problem1_problem2_l265_265805

variable {a b : ℝ}

theorem problem1 (ha : a ≠ 0) (hb : b ≠ 0) :
  (4 * b^3 / a) / (2 * b / a^2) = 2 * a * b^2 :=
by 
  sorry

theorem problem2 (ha : a ≠ b) :
  (a^2 / (a - b)) + (b^2 / (a - b)) - (2 * a * b / (a - b)) = a - b :=
by 
  sorry

end problem1_problem2_l265_265805


namespace average_of_a_and_b_l265_265748

theorem average_of_a_and_b (a b c M : ℝ)
  (h1 : (a + b) / 2 = M)
  (h2 : (b + c) / 2 = 180)
  (h3 : a - c = 200) : 
  M = 280 :=
sorry

end average_of_a_and_b_l265_265748


namespace probability_product_positive_of_independent_selection_l265_265924

theorem probability_product_positive_of_independent_selection :
  let I := set.Icc (-30 : ℝ) (15 : ℝ)
  let P := (λ (x y : ℝ), x ∈ I ∧ y ∈ I ∧ x * y > 0)
  (Prob { x : ℝ × ℝ | P x.1 x.2 } :
    ProbabilitySpace (I × I)) = 5 / 9 :=
by
  sorry

end probability_product_positive_of_independent_selection_l265_265924


namespace largest_of_four_integers_l265_265107

theorem largest_of_four_integers (n : ℤ) (h1 : n % 2 = 0) (h2 : (n+2) % 2 = 0) (h3 : (n+4) % 2 = 0) (h4 : (n+6) % 2 = 0) (h : n * (n+2) * (n+4) * (n+6) = 6720) : max (max (max n (n+2)) (n+4)) (n+6) = 14 := 
sorry

end largest_of_four_integers_l265_265107


namespace shortest_altitude_l265_265514

theorem shortest_altitude (a b c : ℝ) (h : a = 9 ∧ b = 12 ∧ c = 15) (h_right : a^2 + b^2 = c^2) : 
  ∃ x : ℝ, x = 7.2 ∧ (1/2) * c * x = (1/2) * a * b := 
by
  sorry

end shortest_altitude_l265_265514


namespace range_of_x_range_of_a_l265_265725

-- Problem (1) representation
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (m x : ℝ) : Prop := 1 < m ∧ m < 2 ∧ x = (1 / 2)^(m - 1)

theorem range_of_x (x : ℝ) :
  (∀ m, 1 < m ∧ m < 2 → x = (1 / 2)^(m - 1)) ∧ p (1/4) x →
  1/2 < x ∧ x < 3/4 :=
sorry

-- Problem (2) representation
theorem range_of_a (a : ℝ) :
  (∀ m, 1 < m ∧ m < 2 → ∀ x, x = (1 / 2)^(m - 1) → p a x) →
  1/3 ≤ a ∧ a ≤ 1/2 :=
sorry

end range_of_x_range_of_a_l265_265725


namespace common_difference_l265_265563

variable (a1 d : ℤ)
variable (S : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def sum_first_n_terms (n : ℕ) : ℤ :=
  n * a1 + d * (n * (n - 1) / 2)

-- Condition: 2 * S 3 = 3 * S 2 + 6
axiom cond : 2 * sum_first_n_terms 3 = 3 * sum_first_n_terms 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem common_difference : d = 2 :=
by
  sorry

end common_difference_l265_265563


namespace magician_trick_l265_265651

theorem magician_trick (coins : Fin 11 → Bool) 
  (exists_pair : ∃ i : Fin 11, coins i = coins ((i + 1) % 11)) 
  (assistant_left_uncovered : Fin 11) 
  (uncovered_coin_same_face : coins assistant_left_uncovered = coins (some (exists_pair))) :
  ∃ j : Fin 11, j ≠ assistant_left_uncovered ∧ coins j = coins assistant_left_uncovered := 
  sorry

end magician_trick_l265_265651


namespace arithmetic_sequence_common_difference_l265_265541

variable {a₁ d : ℕ}
variable S : ℕ → ℕ

-- Definitions of the sums S₂ and S₃ in an arithmetic sequence
def S₂ : ℕ := a₁ + (a₁ + d)
def S₃ : ℕ := a₁ + (a₁ + d) + (a₁ + 2 * d)

theorem arithmetic_sequence_common_difference (h : 2 * S₃ = 3 * S₂ + 6) : d = 2 :=
by
  -- Skip the proof.
  sorry

end arithmetic_sequence_common_difference_l265_265541


namespace common_difference_of_arithmetic_sequence_l265_265533

noncomputable def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range n, a i

def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_cond : 2 * S a 3 = 3 * S a 2 + 6) :
  ∃ d : ℝ, d = 2 := sorry

end common_difference_of_arithmetic_sequence_l265_265533


namespace exponentiation_and_multiplication_of_fractions_l265_265808

-- Let's define the required fractions
def a : ℚ := 3 / 4
def b : ℚ := 1 / 5

-- Define the expected result
def expected_result : ℚ := 81 / 1280

-- State the theorem to prove
theorem exponentiation_and_multiplication_of_fractions : (a^4) * b = expected_result := by 
  sorry

end exponentiation_and_multiplication_of_fractions_l265_265808


namespace find_other_endpoint_l265_265408

theorem find_other_endpoint (x y : ℝ) : 
  (∃ x1 y1 x2 y2 : ℝ, (x1 + x2)/2 = 2 ∧ (y1 + y2)/2 = 3 ∧ x1 = -1 ∧ y1 = 7 ∧ x2 = x ∧ y2 = y) → (x = 5 ∧ y = -1) :=
by
  sorry

end find_other_endpoint_l265_265408


namespace find_a2023_l265_265830

noncomputable def a_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  (∀ n, 2 * S n = a n * (a n + 1))

theorem find_a2023 (a S : ℕ → ℕ) 
  (h_seq : a_sequence a S)
  (h_pos : ∀ n, 0 < a n) 
  : a 2023 = 2023 :=
sorry

end find_a2023_l265_265830


namespace grade_assignment_ways_l265_265245

theorem grade_assignment_ways : (4^12 = 16777216) := 
by 
  sorry

end grade_assignment_ways_l265_265245


namespace flyDistanceCeiling_l265_265717

variable (P : ℝ × ℝ × ℝ)
variable (x : ℝ)
variable (y : ℝ)
variable (z : ℝ)

-- Defining the conditions
def isAtRightAngles (P : ℝ × ℝ × ℝ) : Prop :=
  P = (0, 0, 0)

def distanceFromWall1 (x : ℝ) : Prop :=
  x = 2

def distanceFromWall2 (y : ℝ) : Prop :=
  y = 5

def distanceFromPointP (x y z : ℝ) : Prop :=
  7 = Real.sqrt (x^2 + y^2 + z^2)

-- Proving the distance from the ceiling
theorem flyDistanceCeiling (P : ℝ × ℝ × ℝ) (x y z : ℝ) :
  isAtRightAngles P →
  distanceFromWall1 x →
  distanceFromWall2 y →
  distanceFromPointP x y z →
  z = 2 * Real.sqrt 5 := 
sorry

end flyDistanceCeiling_l265_265717


namespace find_common_difference_l265_265555

section
variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

-- Condition: S_n represents the sum of the first n terms of the arithmetic sequence {a_n}
def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop := 
  S n = (n * (2 * a 1 + (n - 1) * d)) / 2

-- Condition: 2S_3 = 3S_2 + 6
def arithmetic_condition (S : ℕ → ℝ) : Prop :=
  2 * S 3 = 3 * S 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem find_common_difference 
  (h₁ : sum_arithmetic_sequence S a 2)
  (h₂ : sum_arithmetic_sequence S a 3)
  (h₃ : arithmetic_condition S) :
  d = 2 :=
sorry
end

end find_common_difference_l265_265555


namespace circle_diameter_point_x_l265_265718

-- Define the endpoints of the circle's diameter
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (21, 0)

-- Define the point (x, 12)
def P_x : ℝ → ℝ × ℝ := λ x, (x, 12)

-- Mathematical statement to prove: for a point on the circle with diameter endpoints (-3,0) and (21,0), 
--   and y-coordinate 12, the x-coordinate must be 9.
theorem circle_diameter_point_x (x : ℝ) :
  let C := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in  -- center of the circle
  let r := (Mathlib.Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) / 2 in  -- radius of the circle
  let D := (9, 0) in  -- center is (9, 0)
  let R := 12 in  -- radius is 12
  (P_x x).fst = 9 := by
  -- stating that the point (x,12) lies on the circle with the center (9,0) and radius 12
  sorry

end circle_diameter_point_x_l265_265718


namespace niko_total_profit_l265_265032

def pairs_of_socks : Nat := 9
def cost_per_pair : ℝ := 2
def profit_percentage_first_four : ℝ := 0.25
def profit_per_pair_remaining_five : ℝ := 0.2

theorem niko_total_profit :
  let total_profit_first_four := 4 * (cost_per_pair * profit_percentage_first_four)
  let total_profit_remaining_five := 5 * profit_per_pair_remaining_five
  let total_profit := total_profit_first_four + total_profit_remaining_five
  total_profit = 3 := by
  sorry

end niko_total_profit_l265_265032


namespace tangent_line_at_point_l265_265117

def f (x : ℝ) : ℝ := x^3 + x - 16

def f' (x : ℝ) : ℝ := 3*x^2 + 1

def tangent_line (x : ℝ) (f'val : ℝ) (p_x p_y : ℝ) : ℝ := f'val * (x - p_x) + p_y

theorem tangent_line_at_point (x y : ℝ) (h : x = 2 ∧ y = -6 ∧ f 2 = -6) : 
  ∃ a b c : ℝ, a*x + b*y + c = 0 ∧ a = 13 ∧ b = -1 ∧ c = -32 :=
by
  use 13, -1, -32
  sorry

end tangent_line_at_point_l265_265117


namespace solve_x_squared_plus_y_squared_l265_265303

-- Variables
variables {x y : ℝ}

-- Conditions
def cond1 : (x + y)^2 = 36 := sorry
def cond2 : x * y = 8 := sorry

-- Theorem stating the problem's equivalent proof
theorem solve_x_squared_plus_y_squared : x^2 + y^2 = 20 := sorry

end solve_x_squared_plus_y_squared_l265_265303


namespace x1_x2_eq_e2_l265_265850

variable (x1 x2 : ℝ)

-- Conditions
def condition1 : Prop := x1 * Real.exp x1 = Real.exp 2
def condition2 : Prop := x2 * Real.log x2 = Real.exp 2

-- The proof problem
theorem x1_x2_eq_e2 (hx1 : condition1 x1) (hx2 : condition2 x2) : x1 * x2 = Real.exp 2 := 
sorry

end x1_x2_eq_e2_l265_265850


namespace proof_problem_l265_265672

def intelligentFailRate (r1 r2 r3 : ℚ) : ℚ :=
  1 - r1 * r2 * r3

def phi (p : ℚ) : ℚ :=
  30 * p * (1 - p)^29

def derivativePhi (p : ℚ) : ℚ :=
  30 * (1 - p)^28 * (1 - 30 * p)

def qualifiedPassRate (intelligentPassRate comprehensivePassRate : ℚ) : ℚ :=
  intelligentPassRate * comprehensivePassRate

theorem proof_problem :
  let r1 := (99 : ℚ) / 100
  let r2 := (98 : ℚ) / 99
  let r3 := (97 : ℚ) / 98
  let p0 := (1 : ℚ) / 30
  let comprehensivePassRate := 1 - p0
  let qualifiedRate := qualifiedPassRate (r1 * r2 * r3) comprehensivePassRate
  (intelligentFailRate r1 r2 r3 = 3 / 100) ∧
  (derivativePhi p0 = 0) ∧
  (qualifiedRate < 96 / 100) :=
by
  sorry

end proof_problem_l265_265672


namespace solve_for_x_l265_265506

theorem solve_for_x (b x : ℝ) (h1 : b > 1) (h2 : x > 0)
    (h3 : (4 * x) ^ (Real.log 4 / Real.log b) = (6 * x) ^ (Real.log 6 / Real.log b)) :
    x = 1 / 6 :=
by
  sorry

end solve_for_x_l265_265506


namespace smallest_counterexample_is_14_l265_265473

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_not_prime (n : ℕ) : Prop := ¬Prime n
def smallest_counterexample (n : ℕ) : Prop :=
  is_even n ∧ is_not_prime n ∧ is_not_prime (n + 2) ∧ ∀ m, is_even m ∧ is_not_prime m ∧ is_not_prime (m + 2) → n ≤ m

theorem smallest_counterexample_is_14 : smallest_counterexample 14 :=
by
  sorry

end smallest_counterexample_is_14_l265_265473


namespace car_travel_distance_l265_265953

theorem car_travel_distance:
  (∃ r, r = 3 / 4 ∧ ∀ t, t = 2 → ((r * 60) * t = 90)) :=
by
  sorry

end car_travel_distance_l265_265953


namespace good_students_count_l265_265363

theorem good_students_count (E B : ℕ) 
    (h1 : E + B = 25) 
    (h2 : ∀ (e ∈ finset.range 5), B > 12) 
    (h3 : ∀ (e ∈ finset.range 20), B = 3 * (E - 1)) :
    E = 5 ∨ E = 7 :=
by
  sorry

end good_students_count_l265_265363


namespace wheat_flour_used_l265_265631

-- Conditions and definitions
def total_flour_used : ℝ := 0.3
def white_flour_used : ℝ := 0.1

-- Statement of the problem
theorem wheat_flour_used : 
  (total_flour_used - white_flour_used) = 0.2 :=
by
  sorry

end wheat_flour_used_l265_265631


namespace josh_money_left_l265_265010

theorem josh_money_left (initial_amount : ℝ) (first_spend : ℝ) (second_spend : ℝ) 
  (h1 : initial_amount = 9) 
  (h2 : first_spend = 1.75) 
  (h3 : second_spend = 1.25) : 
  initial_amount - first_spend - second_spend = 6 := 
by 
  sorry

end josh_money_left_l265_265010


namespace last_popsicle_melts_32_times_faster_l265_265069

theorem last_popsicle_melts_32_times_faster (t : ℕ) : 
  let time_first := t
  let time_sixth := t / 2^5
  (time_first / time_sixth) = 32 :=
by
  sorry

end last_popsicle_melts_32_times_faster_l265_265069


namespace common_difference_is_two_l265_265590

variable {a₁ a₂ a₃ S₃ S₂ : ℕ}
variable (d : ℕ)

-- Given condition
axiom H : 2 * S₃ = 3 * S₂ + 6

-- Definitions based on arithmetic sequence properties
def S₂ := a₁ + a₂
def S₃ := a₁ + a₂ + a₃
def a₂ := a₁ + d
def a₃ := a₁ + 2 * d

theorem common_difference_is_two : d = 2 := 
by 
  sorry

end common_difference_is_two_l265_265590


namespace greatest_two_digit_multiple_of_17_l265_265195

noncomputable def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem greatest_two_digit_multiple_of_17 : 
    ∃ n : ℕ, is_two_digit n ∧ (∃ k : ℕ, n = 17 * k) ∧
    ∀ m : ℕ, is_two_digit m → (∃ k : ℕ, m = 17 * k) → m ≤ n := 
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l265_265195


namespace probability_at_most_3_heads_l265_265445

theorem probability_at_most_3_heads (n : ℕ) (h : n = 12) :
  let total_outcomes := 2^12,
      favorable_outcomes := (∑ k in finset.Icc 0 3, nat.choose 12 k)
  in favorable_outcomes / total_outcomes = 299 / 4096 :=
by {
  sorry
}

end probability_at_most_3_heads_l265_265445


namespace incorrect_transformation_l265_265267

theorem incorrect_transformation (a b : ℕ) (h : a ≠ 0 ∧ b ≠ 0) (h_eq : a / 2 = b / 3) :
  (∃ k : ℕ, 2 * a = 3 * b → false) ∧ 
  (a / b = 2 / 3) ∧ 
  (b / a = 3 / 2) ∧
  (3 * a = 2 * b) :=
by
  sorry

end incorrect_transformation_l265_265267


namespace find_B_l265_265967

variable {A B C D : ℕ}

-- Condition 1: The first dig site (A) was dated 352 years more recent than the second dig site (B)
axiom h1 : A = B + 352

-- Condition 2: The third dig site (C) was dated 3700 years older than the first dig site (A)
axiom h2 : C = A - 3700

-- Condition 3: The fourth dig site (D) was twice as old as the third dig site (C)
axiom h3 : D = 2 * C

-- Condition 4: The age difference between the second dig site (B) and the third dig site (C) was four times the difference between the fourth dig site (D) and the first dig site (A)
axiom h4 : B - C = 4 * (D - A)

-- Condition 5: The fourth dig site is dated 8400 BC.
axiom h5 : D = 8400

-- Prove the question
theorem find_B : B = 7548 :=
by
  sorry

end find_B_l265_265967


namespace modulus_of_z_l265_265271

open Complex

theorem modulus_of_z (z : ℂ) (h : z^2 = (3/4 : ℝ) - I) : abs z = Real.sqrt 5 / 2 := 
  sorry

end modulus_of_z_l265_265271


namespace probability_X_greater_than_2_l265_265508

noncomputable def probability_distribution (i : ℕ) : ℝ :=
  if h : 1 ≤ i ∧ i ≤ 4 then i / 10 else 0

theorem probability_X_greater_than_2 :
  (probability_distribution 3 + probability_distribution 4) = 0.7 := by 
  sorry

end probability_X_greater_than_2_l265_265508


namespace floor_inequality_solution_set_l265_265868

/-- Let ⌊x⌋ denote the greatest integer less than or equal to x.
    Prove that the solution set of the inequality ⌊x⌋² - 5⌊x⌋ - 36 ≤ 0 is {x | -4 ≤ x < 10}. -/
theorem floor_inequality_solution_set (x : ℝ) :
  (⌊x⌋^2 - 5 * ⌊x⌋ - 36 ≤ 0) ↔ -4 ≤ x ∧ x < 10 := by
    sorry

end floor_inequality_solution_set_l265_265868


namespace coef_linear_term_expansion_l265_265000

theorem coef_linear_term_expansion : 
  let f := (1 : ℚ) + X
  let g := (X + 1/X) * f^5
  coeff g 1 = 11 := by
  sorry

end coef_linear_term_expansion_l265_265000


namespace problem_statement_l265_265840

open Real

noncomputable def f (ω varphi : ℝ) (x : ℝ) := 2 * sin (ω * x + varphi)

theorem problem_statement (ω varphi : ℝ) (x1 x2 : ℝ) (hω_pos : ω > 0) (hvarphi_abs : abs varphi < π / 2)
    (hf0 : f ω varphi 0 = -1) (hmonotonic : ∀ x y, π / 18 < x ∧ x < y ∧ y < π / 3 → f ω varphi x < f ω varphi y)
    (hshift : ∀ x, f ω varphi (x + π) = f ω varphi x)
    (hx1x2_interval : -17 * π / 12 < x1 ∧ x1 < -2 * π / 3 ∧ -17 * π / 12 < x2 ∧ x2 < -2 * π / 3 ∧ x1 ≠ x2)
    (heq_fx : f ω varphi x1 = f ω varphi x2) :
    f ω varphi (x1 + x2) = -1 :=
sorry

end problem_statement_l265_265840


namespace magician_can_identify_matching_coin_l265_265657

-- Define the types and conditions
variable (coins : Fin 11 → Bool) -- Each coin is either heads (true) or tails (false).
variable (uncovered_index : Fin 11) -- The index of the initially uncovered coin.

-- Define the property to be proved: the magician can point to an adjacent coin with the same face as the uncovered one.
theorem magician_can_identify_matching_coin (h : ∃ i : Fin 11, coins i = coins (i + 1) % 11) :
  ∃ j : Fin 11, (j ≠ uncovered_index) ∧ coins j = coins uncovered_index :=
  sorry

end magician_can_identify_matching_coin_l265_265657


namespace average_total_goals_l265_265676

theorem average_total_goals (carter_avg shelby_avg judah_avg total_avg : ℕ) 
    (h1: carter_avg = 4) 
    (h2: shelby_avg = carter_avg / 2)
    (h3: judah_avg = 2 * shelby_avg - 3) 
    (h4: total_avg = carter_avg + shelby_avg + judah_avg) :
  total_avg = 7 :=
by
  sorry

end average_total_goals_l265_265676


namespace smallest_number_among_given_l265_265966

theorem smallest_number_among_given :
  ∀ (a b c d : ℚ), a = -2 → b = -5/2 → c = 0 → d = 1/5 →
  (min (min (min a b) c) d) = b :=
by
  intros a b c d ha hb hc hd
  rw [ha, hb, hc, hd]
  sorry

end smallest_number_among_given_l265_265966


namespace trig_identity_example_l265_265804

open Real -- Using the Real namespace for trigonometric functions

theorem trig_identity_example :
  sin (135 * π / 180) * cos (-15 * π / 180) + cos (225 * π / 180) * sin (15 * π / 180) = 1 / 2 :=
by 
  -- sorry to skip the proof steps
  sorry

end trig_identity_example_l265_265804


namespace gas_usage_correct_l265_265383

def starting_gas : ℝ := 0.5
def ending_gas : ℝ := 0.16666666666666666

theorem gas_usage_correct : starting_gas - ending_gas = 0.33333333333333334 := by
  sorry

end gas_usage_correct_l265_265383


namespace principal_amount_l265_265122

theorem principal_amount (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) 
  (h1 : R = 4) 
  (h2 : T = 5) 
  (h3 : SI = P - 1920) 
  (h4 : SI = (P * R * T) / 100) : 
  P = 2400 := 
by 
  sorry

end principal_amount_l265_265122


namespace moon_arrangement_l265_265816

theorem moon_arrangement : 
  let M_count := 1
  let O_count := 2
  let N_count := 1
  let total_letters := 4
  ∑ perm : List Nat, perm.permutations.length = total_letters! // (M_count! * O_count! * N_count!) :=
  12
 :=
begin
  -- Definitions from the condition
  have M_count := 1,
  have O_count := 2,
  have N_count := 1,
  have total_letters := 4,
  
  -- Applying formulas for permutation counts
  let num_unique_arrangements := total_letters.factorial / (M_count.factorial * O_count.factorial * N_count.factorial),
  show num_unique_arrangements = 12 from sorry
end

end moon_arrangement_l265_265816


namespace target_annual_revenue_l265_265078

-- Given conditions as definitions
def monthly_sales : ℕ := 4000
def additional_sales : ℕ := 1000

-- The proof problem in Lean statement form
theorem target_annual_revenue : (monthly_sales + additional_sales) * 12 = 60000 := by
  sorry

end target_annual_revenue_l265_265078


namespace common_difference_l265_265566

variable (a1 d : ℤ)
variable (S : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def sum_first_n_terms (n : ℕ) : ℤ :=
  n * a1 + d * (n * (n - 1) / 2)

-- Condition: 2 * S 3 = 3 * S 2 + 6
axiom cond : 2 * sum_first_n_terms 3 = 3 * sum_first_n_terms 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem common_difference : d = 2 :=
by
  sorry

end common_difference_l265_265566


namespace sin_function_props_l265_265700

theorem sin_function_props (A ω m : ℝ) 
  (hA : A > 0) (hω : ω > 0) 
  (h_max : ∀ x, A * sin (ω * x + π / 6) + m ≤ 3) 
  (h_min : ∀ x, A * sin (ω * x + π / 6) + m ≥ -5)
  (h_sym : ∀ x, A * sin (ω * (x + π / (2 * ω)) + π / 6) + m = A * sin (ω * x + π / 6) + m) :
  A = 4 ∧ ω = 2 ∧ m = -1 :=
by
  sorry

end sin_function_props_l265_265700


namespace initial_saltwater_amount_l265_265784

variable (x y : ℝ)
variable (h1 : 0.04 * x = (x - y) * 0.1)
variable (h2 : ((x - y) * 0.1 + 300 * 0.04) / (x - y + 300) = 0.064)

theorem initial_saltwater_amount : x = 500 :=
by
  sorry

end initial_saltwater_amount_l265_265784


namespace total_heads_is_46_l265_265241

noncomputable def total_heads (hens cows : ℕ) : ℕ :=
  hens + cows

def num_feet_hens (num_hens : ℕ) : ℕ :=
  2 * num_hens

def num_cows (total_feet feet_hens_per_cow feet_cow_per_cow : ℕ) : ℕ :=
  (total_feet - feet_hens_per_cow) / feet_cow_per_cow

theorem total_heads_is_46 (num_hens : ℕ) (total_feet : ℕ)
  (hen_feet cow_feet hen_head cow_head : ℕ)
  (num_heads : ℕ) :
  num_hens = 24 →
  total_feet = 136 →
  hen_feet = 2 →
  cow_feet = 4 →
  hen_head = 1 →
  cow_head = 1 →
  num_heads = total_heads num_hens (num_cows total_feet (num_feet_hens num_hens) cow_feet) →
  num_heads = 46 :=
by
  intros
  sorry

end total_heads_is_46_l265_265241


namespace factorize_expression_l265_265479

theorem factorize_expression (x y : ℝ) : 3 * x^2 - 12 * y^2 = 3 * (x - 2 * y) * (x + 2 * y) := by
  sorry

end factorize_expression_l265_265479


namespace gcd_pow_diff_l265_265625

theorem gcd_pow_diff :
  gcd (2 ^ 2100 - 1) (2 ^ 2091 - 1) = 511 := 
sorry

end gcd_pow_diff_l265_265625


namespace Alex_failing_implies_not_all_hw_on_time_l265_265858

-- Definitions based on the conditions provided
variable (Alex_submits_all_hw_on_time : Prop)
variable (Alex_passes_course : Prop)

-- Given condition: Submitting all homework assignments implies passing the course
axiom Mrs_Thompson_statement : Alex_submits_all_hw_on_time → Alex_passes_course

-- The problem: Prove that if Alex failed the course, then he did not submit all homework assignments on time
theorem Alex_failing_implies_not_all_hw_on_time (h : ¬Alex_passes_course) : ¬Alex_submits_all_hw_on_time :=
  by
  sorry

end Alex_failing_implies_not_all_hw_on_time_l265_265858


namespace solve_x_squared_plus_y_squared_l265_265301

-- Variables
variables {x y : ℝ}

-- Conditions
def cond1 : (x + y)^2 = 36 := sorry
def cond2 : x * y = 8 := sorry

-- Theorem stating the problem's equivalent proof
theorem solve_x_squared_plus_y_squared : x^2 + y^2 = 20 := sorry

end solve_x_squared_plus_y_squared_l265_265301


namespace smallest_n_for_at_least_64_candies_l265_265790

theorem smallest_n_for_at_least_64_candies :
  ∃ n : ℕ, (n > 0) ∧ (n * (n + 1) / 2 ≥ 64) ∧ (∀ m : ℕ, (m > 0) ∧ (m * (m + 1) / 2 ≥ 64) → n ≤ m) := 
sorry

end smallest_n_for_at_least_64_candies_l265_265790


namespace simplify_expression_l265_265399

variable (x y : ℝ)

theorem simplify_expression:
  3*x^2 - 3*(2*x^2 + 4*y) + 2*(x^2 - y) = -x^2 - 14*y := 
by 
  sorry

end simplify_expression_l265_265399


namespace total_time_is_correct_l265_265022

-- Definitions based on conditions
def timeouts_for_running : ℕ := 5
def timeouts_for_throwing_food : ℕ := 5 * timeouts_for_running - 1
def timeouts_for_swearing : ℕ := timeouts_for_throwing_food / 3

-- Definition for total time-outs
def total_timeouts : ℕ := timeouts_for_running + timeouts_for_throwing_food + timeouts_for_swearing
-- Each time-out is 5 minutes
def timeout_duration : ℕ := 5

-- Total time in minutes
def total_time_in_minutes : ℕ := total_timeouts * timeout_duration

-- The proof statement
theorem total_time_is_correct : total_time_in_minutes = 185 := by
  sorry

end total_time_is_correct_l265_265022


namespace xy_problem_l265_265305

theorem xy_problem (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 :=
by
  sorry

end xy_problem_l265_265305


namespace greatest_two_digit_multiple_of_17_is_85_l265_265208

theorem greatest_two_digit_multiple_of_17_is_85 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ m % 17 = 0) → m ≤ n) ∧ n = 85 :=
sorry

end greatest_two_digit_multiple_of_17_is_85_l265_265208


namespace find_base_k_l265_265828

theorem find_base_k (k : ℕ) (hk : 0 < k) (h : 7/51 = (2 * k + 3) / (k^2 - 1)) : k = 16 :=
sorry

end find_base_k_l265_265828


namespace initial_bees_l265_265908

theorem initial_bees (B : ℕ) (h : B + 8 = 24) : B = 16 := 
by {
  sorry
}

end initial_bees_l265_265908


namespace complex_modulus_inequality_l265_265881

theorem complex_modulus_inequality (z : ℂ) : (‖z‖ ^ 2 + 2 * ‖z - 1‖) ≥ 1 :=
by
  sorry

end complex_modulus_inequality_l265_265881


namespace problem_statement_l265_265294

-- Define the conditions and the goal
theorem problem_statement {x y : ℝ} 
  (h1 : (x + y)^2 = 36)
  (h2 : x * y = 8) : 
  x^2 + y^2 = 20 := 
by
  sorry

end problem_statement_l265_265294


namespace smith_boxes_l265_265384

theorem smith_boxes (initial_markers : ℕ) (markers_per_box : ℕ) (total_markers : ℕ)
  (h1 : initial_markers = 32) (h2 : markers_per_box = 9) (h3 : total_markers = 86) :
  total_markers - initial_markers = 6 * markers_per_box :=
by
  -- We state our assumptions explicitly for better clarity
  have h_total : total_markers = 86 := h3
  have h_initial : initial_markers = 32 := h1
  have h_box : markers_per_box = 9 := h2
  sorry

end smith_boxes_l265_265384


namespace exists_n_prime_factors_m_exp_n_plus_n_exp_m_l265_265498

theorem exists_n_prime_factors_m_exp_n_plus_n_exp_m (m k : ℕ) (hm : m > 0) (hm_odd : m % 2 = 1) (hk : k > 0) :
  ∃ n : ℕ, n > 0 ∧ (∃ primes : Finset ℕ, primes.card ≥ k ∧ ∀ p ∈ primes, p.Prime ∧ p ∣ m ^ n + n ^ m) := 
sorry

end exists_n_prime_factors_m_exp_n_plus_n_exp_m_l265_265498


namespace initial_butterfly_count_l265_265821

theorem initial_butterfly_count (n : ℕ) (h : (2 / 3 : ℚ) * n = 6) : n = 9 :=
sorry

end initial_butterfly_count_l265_265821


namespace f_is_odd_range_of_x_l265_265093

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ + f x₂
axiom f_3 : f 3 = 1
axiom f_increase_nonneg : ∀ x₁ x₂ : ℝ, (0 ≤ x₁ ∧ x₁ ≤ x₂) → f x₁ ≤ f x₂
axiom f_lt_2 : ∀ x : ℝ, f (x - 1) < 2

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x :=
sorry

theorem range_of_x : {x : ℝ | f (x - 1) < 2} =
{s : ℝ | sorry } :=
sorry

end f_is_odd_range_of_x_l265_265093


namespace percentage_reduction_l265_265797

theorem percentage_reduction (y x z p q : ℝ) (hy : y ≠ 0) (h1 : x = y - 10) (h2 : z = y - 20) :
  p = 1000 / y ∧ q = 2000 / y := by
  sorry

end percentage_reduction_l265_265797


namespace add_fractions_l265_265487

theorem add_fractions : (2 / 3 : ℚ) + (7 / 8) = 37 / 24 := 
by sorry

end add_fractions_l265_265487


namespace average_price_of_remaining_packets_l265_265735

variables (initial_avg_price : ℕ) (initial_packets : ℕ) (returned_packets : ℕ) (returned_avg_price : ℕ)

def total_initial_cost := initial_avg_price * initial_packets
def total_returned_cost := returned_avg_price * returned_packets
def remaining_packets := initial_packets - returned_packets
def total_remaining_cost := total_initial_cost initial_avg_price initial_packets - total_returned_cost returned_avg_price returned_packets
def remaining_avg_price := total_remaining_cost initial_avg_price initial_packets returned_avg_price returned_packets / remaining_packets initial_packets returned_packets

theorem average_price_of_remaining_packets :
  initial_avg_price = 20 →
  initial_packets = 5 →
  returned_packets = 2 →
  returned_avg_price = 32 →
  remaining_avg_price initial_avg_price initial_packets returned_avg_price returned_packets = 12
:=
by
  intros h1 h2 h3 h4
  rw [remaining_avg_price, total_remaining_cost, total_initial_cost, total_returned_cost]
  norm_num [h1, h2, h3, h4]
  sorry

end average_price_of_remaining_packets_l265_265735


namespace cost_difference_is_120_l265_265265

-- Define the monthly costs and duration
def rent_monthly_cost : ℕ := 20
def buy_monthly_cost : ℕ := 30
def months_in_a_year : ℕ := 12

-- Annual cost definitions
def annual_rent_cost : ℕ := rent_monthly_cost * months_in_a_year
def annual_buy_cost : ℕ := buy_monthly_cost * months_in_a_year

-- The main theorem to prove the difference in annual cost is $120
theorem cost_difference_is_120 : annual_buy_cost - annual_rent_cost = 120 := by
  sorry

end cost_difference_is_120_l265_265265


namespace find_Q_l265_265125

theorem find_Q (m n Q p : ℝ) (h1 : m = 6 * n + 5)
    (h2 : p = 0.3333333333333333)
    (h3 : m + Q = 6 * (n + p) + 5) : Q = 2 := 
by
  sorry

end find_Q_l265_265125


namespace factorize_expression_l265_265482

theorem factorize_expression (x y : ℝ) : 3 * x^2 - 12 * y^2 = 3 * (x - 2 * y) * (x + 2 * y) := by
  sorry

end factorize_expression_l265_265482


namespace arithmetic_sequence_common_difference_l265_265559

theorem arithmetic_sequence_common_difference (a_1 d : ℝ) (S : ℕ → ℝ) 
    (h1 : S 2 = 2 * a_1 + d)
    (h2 : S 3 = 3 * a_1 + 3 * d)
    (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 := 
by
  sorry

end arithmetic_sequence_common_difference_l265_265559


namespace factorize_expression_l265_265481

theorem factorize_expression (x y : ℝ) : 3 * x^2 - 12 * y^2 = 3 * (x - 2 * y) * (x + 2 * y) := by
  sorry

end factorize_expression_l265_265481


namespace alan_tickets_l265_265911

variables (A M : ℕ)

def condition1 := A + M = 150
def condition2 := M = 5 * A - 6

theorem alan_tickets : A = 26 :=
by
  have h1 : condition1 A M := sorry
  have h2 : condition2 A M := sorry
  sorry

end alan_tickets_l265_265911


namespace remainder_1234_5678_9012_div_5_l265_265939

theorem remainder_1234_5678_9012_div_5 : (1234 * 5678 * 9012) % 5 = 4 := by
  sorry

end remainder_1234_5678_9012_div_5_l265_265939


namespace problem_statement_l265_265292

-- Define the conditions and the goal
theorem problem_statement {x y : ℝ} 
  (h1 : (x + y)^2 = 36)
  (h2 : x * y = 8) : 
  x^2 + y^2 = 20 := 
by
  sorry

end problem_statement_l265_265292


namespace max_value_fraction_l265_265121

theorem max_value_fraction (a b : ℝ)
  (h1 : a + b - 2 ≥ 0)
  (h2 : b - a - 1 ≤ 0)
  (h3 : a ≤ 1) :
  (a ≠ 0) → (b ≠ 0) →
  ∃ m, m = (a + 2 * b) / (2 * a + b) ∧ m ≤ 7 / 5 :=
by
  sorry

end max_value_fraction_l265_265121


namespace induction_problem_l265_265880

open BigOperators

theorem induction_problem (n : ℕ) (h : n > 0) :
  ∑ i in Finset.range n, (i + 1) * (i + 2) * (i + 3) = (n * (n + 1) * (n + 2) * (n + 3)) / 4 := 
by
  sorry

end induction_problem_l265_265880


namespace find_h3_l265_265471

noncomputable def h (x : ℝ) : ℝ :=
  ((x + 1) * (x^2 + 1) * (x^3 + 1) * (x^9 + 1) - 1) / (x^(3^3 - 1) - 1)

theorem find_h3 : h 3 = 3 := by
  sorry

end find_h3_l265_265471


namespace probability_product_positive_of_independent_selection_l265_265921

theorem probability_product_positive_of_independent_selection :
  let I := set.Icc (-30 : ℝ) (15 : ℝ)
  let P := (λ (x y : ℝ), x ∈ I ∧ y ∈ I ∧ x * y > 0)
  (Prob { x : ℝ × ℝ | P x.1 x.2 } :
    ProbabilitySpace (I × I)) = 5 / 9 :=
by
  sorry

end probability_product_positive_of_independent_selection_l265_265921


namespace log_inequalities_l265_265638

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 2 / Real.log 3
noncomputable def c : ℝ := Real.log (Real.log 2 / Real.log 3) / Real.log 2

theorem log_inequalities : c < b ∧ b < a :=
  sorry

end log_inequalities_l265_265638


namespace sum_of_interior_angles_of_pentagon_l265_265616

theorem sum_of_interior_angles_of_pentagon :
  let n := 5
  let angleSum := 180 * (n - 2)
  angleSum = 540 :=
by
  sorry

end sum_of_interior_angles_of_pentagon_l265_265616


namespace correct_exponentiation_l265_265058

theorem correct_exponentiation : ∀ (x : ℝ), (x^(4/5))^(5/4) = x :=
by
  intro x
  sorry

end correct_exponentiation_l265_265058


namespace initial_calculated_average_was_23_l265_265891

theorem initial_calculated_average_was_23 (S : ℕ) (incorrect_sum : ℕ) (n : ℕ)
  (correct_sum : ℕ) (correct_average : ℕ) (wrong_read : ℕ) (correct_read : ℕ) :
  (n = 10) →
  (wrong_read = 26) →
  (correct_read = 36) →
  (correct_average = 24) →
  (correct_sum = n * correct_average) →
  (incorrect_sum = correct_sum - correct_read + wrong_read) →
  S = incorrect_sum →
  S / n = 23 :=
by
  intros
  sorry

end initial_calculated_average_was_23_l265_265891


namespace magician_assistant_trick_l265_265652

/-- A coin can be either heads or tails. -/
inductive Coin
| heads : Coin
| tails : Coin

/-- Given a cyclic arrangement of 11 coins, there exists at least one pair of adjacent coins with the same face. -/
theorem magician_assistant_trick (coins : Fin 11 → Coin) : 
  ∃ i : Fin 11, coins i = coins (i + 1) := 
by
  sorry

end magician_assistant_trick_l265_265652


namespace coeffs_of_quadratic_eq_l265_265517

theorem coeffs_of_quadratic_eq :
  ∃ a b c : ℤ, (2 * x^2 + x - 5 = 0) → (a = 2 ∧ b = 1 ∧ c = -5) :=
by
  sorry

end coeffs_of_quadratic_eq_l265_265517


namespace frozen_yogurt_price_l265_265745

variable (F G S : ℝ) -- Define the variables F, G, S as real numbers

-- Define the conditions given in the problem
variable (h1 : 5 * F + 2 * G + 5 * S = 55)
variable (h2 : S = 5)
variable (h3 : G = 1 / 2 * F)

-- State the proof goal
theorem frozen_yogurt_price : F = 5 :=
by
  sorry

end frozen_yogurt_price_l265_265745


namespace greatest_two_digit_multiple_of_17_is_85_l265_265207

theorem greatest_two_digit_multiple_of_17_is_85 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ m % 17 = 0) → m ≤ n) ∧ n = 85 :=
sorry

end greatest_two_digit_multiple_of_17_is_85_l265_265207


namespace train_speed_l265_265943

theorem train_speed 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (crossing_time : ℝ) 
  (h_train_length : train_length = 400) 
  (h_bridge_length : bridge_length = 300) 
  (h_crossing_time : crossing_time = 45) : 
  (train_length + bridge_length) / crossing_time = 700 / 45 := 
  by
    rw [h_train_length, h_bridge_length, h_crossing_time]
    sorry

end train_speed_l265_265943


namespace greatest_two_digit_multiple_of_17_l265_265190

-- Define the range of two-digit numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate for being a multiple of 17
def is_multiple_of_17 (n : ℕ) := ∃ k : ℕ, n = 17 * k

-- The specific problem to prove
theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n ∈ two_digit_numbers ∧ is_multiple_of_17(n) ∧ 
  (∀ m : ℕ, m ∈ two_digit_numbers ∧ is_multiple_of_17(m) → n ≥ m) ∧ n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l265_265190


namespace highest_mean_possible_l265_265855

def max_arithmetic_mean (g : Matrix (Fin 3) (Fin 3) ℕ) : ℚ := 
  let mean (a b c d : ℕ) : ℚ := (a + b + c + d : ℚ) / 4
  let circles := [
    mean (g 0 0) (g 0 1) (g 1 0) (g 1 1),
    mean (g 0 1) (g 0 2) (g 1 1) (g 1 2),
    mean (g 1 0) (g 1 1) (g 2 0) (g 2 1),
    mean (g 1 1) (g 1 2) (g 2 1) (g 2 2)
  ]
  (circles.sum / 4)

theorem highest_mean_possible :
  ∃ g : Matrix (Fin 3) (Fin 3) ℕ, 
  (∀ i j, 1 ≤ g i j ∧ g i j ≤ 9) ∧ 
  max_arithmetic_mean g = 6.125 :=
by
  sorry

end highest_mean_possible_l265_265855


namespace area_of_triangle_from_squares_l265_265280

theorem area_of_triangle_from_squares :
  ∃ (a b c : ℕ), (a = 15 ∧ b = 15 ∧ c = 6 ∧ (1/2 : ℚ) * a * c = 45) :=
by
  let a := 15
  let b := 15
  let c := 6
  have h1 : (1/2 : ℚ) * a * c = 45 := sorry
  exact ⟨a, b, c, ⟨rfl, rfl, rfl, h1⟩⟩

end area_of_triangle_from_squares_l265_265280


namespace arithmetic_sequence_sum_l265_265421

theorem arithmetic_sequence_sum (c d : ℕ) 
  (h1 : ∀ (a1 a2 a3 a4 a5 a6 : ℕ), a1 = 3 → a2 = 10 → a3 = 17 → a6 = 38 → (a2 - a1 = a3 - a2) → (a3 - a2 = c - a3) → (c - a3 = d - c) → (d - c = a6 - d)) : 
  c + d = 55 := 
by 
  sorry

end arithmetic_sequence_sum_l265_265421


namespace mixture_volume_correct_l265_265367

-- Define the input values
def water_volume : ℕ := 20
def vinegar_volume : ℕ := 18
def water_ratio : ℚ := 3/5
def vinegar_ratio : ℚ := 5/6

-- Calculate the mixture volume
def mixture_volume : ℚ :=
  (water_volume * water_ratio) + (vinegar_volume * vinegar_ratio)

-- Define the expected result
def expected_mixture_volume : ℚ := 27

-- State the theorem
theorem mixture_volume_correct : mixture_volume = expected_mixture_volume := by
  sorry

end mixture_volume_correct_l265_265367


namespace power_function_not_origin_l265_265509

theorem power_function_not_origin (m : ℝ) 
  (h1 : m^2 - 3 * m + 3 = 1) 
  (h2 : m^2 - m - 2 ≤ 0) : 
  m = 1 ∨ m = 2 :=
sorry

end power_function_not_origin_l265_265509


namespace min_value_of_sum_inverse_l265_265406

theorem min_value_of_sum_inverse (m n : ℝ) 
  (H1 : ∃ (x y : ℝ), (x + y - 1 = 0 ∧ 3 * x - y - 7 = 0) ∧ (mx + y + n = 0))
  (H2 : mn > 0) : 
  ∃ k : ℝ, k = 8 ∧ ∀ (m n : ℝ), mn > 0 → (2 * m + n = 1) → 1 / m + 2 / n ≥ k :=
by
  sorry

end min_value_of_sum_inverse_l265_265406


namespace trigonometric_identity_simplification_l265_265851

open Real

theorem trigonometric_identity_simplification (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 4) :
  (sqrt (1 - 2 * sin (3 * π - θ) * sin (π / 2 + θ)) = cos θ - sin θ) :=
sorry

end trigonometric_identity_simplification_l265_265851


namespace good_students_l265_265336

theorem good_students (E B : ℕ) (h1 : E + B = 25) (h2 : 12 < B) (h3 : B = 3 * (E - 1)) :
  E = 5 ∨ E = 7 :=
by 
  sorry

end good_students_l265_265336


namespace problem_statement_l265_265291

-- Define the conditions and the goal
theorem problem_statement {x y : ℝ} 
  (h1 : (x + y)^2 = 36)
  (h2 : x * y = 8) : 
  x^2 + y^2 = 20 := 
by
  sorry

end problem_statement_l265_265291


namespace tenth_term_geometric_sequence_l265_265809

theorem tenth_term_geometric_sequence :
  let a := (8 : ℚ)
  let r := (-2 / 3 : ℚ)
  a * r^9 = -4096 / 19683 :=
by
  sorry

end tenth_term_geometric_sequence_l265_265809


namespace greatest_two_digit_multiple_of_17_l265_265199

theorem greatest_two_digit_multiple_of_17 : ∃ N, N = 85 ∧ 
  (∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n → n ≤ 85) :=
by 
  sorry

end greatest_two_digit_multiple_of_17_l265_265199


namespace cos_75_eq_sqrt6_sub_sqrt2_div_4_l265_265086

theorem cos_75_eq_sqrt6_sub_sqrt2_div_4 :
  Real.cos (75 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := sorry

end cos_75_eq_sqrt6_sub_sqrt2_div_4_l265_265086


namespace find_large_number_l265_265946

theorem find_large_number (L S : ℕ) 
  (h1 : L - S = 1335) 
  (h2 : L = 6 * S + 15) : 
  L = 1599 := 
by 
  -- proof omitted
  sorry

end find_large_number_l265_265946


namespace no_possible_numbering_for_equal_sidesum_l265_265148

theorem no_possible_numbering_for_equal_sidesum (O : Point) (A : Fin 10 → Point) 
  (side_numbers : (Fin 10) → ℕ) (segment_numbers : (Fin 10) → ℕ) : 
  ¬ ∃ (side_segment_sum_equal : Fin 10 → ℕ) (sum_equal : ℕ),
    (∀ i, side_segment_sum_equal i = side_numbers i + segment_numbers i) ∧ 
    (∀ i, side_segment_sum_equal i = sum_equal) := 
sorry

end no_possible_numbering_for_equal_sidesum_l265_265148


namespace find_breadth_of_rectangle_l265_265409

noncomputable def breadth_of_rectangle (s : ℝ) (π_approx : ℝ := 3.14) : ℝ :=
2 * s - 22

theorem find_breadth_of_rectangle (b s : ℝ) (π_approx : ℝ := 3.14) :
  4 * s = 2 * (22 + b) →
  π_approx * s / 2 + s = 29.85 →
  b = 1.22 :=
by
  intros h1 h2
  sorry

end find_breadth_of_rectangle_l265_265409


namespace true_discount_is_52_l265_265440

/-- The banker's gain on a bill due 3 years hence at 15% per annum is Rs. 23.4. -/
def BG : ℝ := 23.4

/-- The rate of interest per annum is 15%. -/
def R : ℝ := 15

/-- The time in years is 3. -/
def T : ℝ := 3

/-- The true discount is Rs. 52. -/
theorem true_discount_is_52 : BG * 100 / (R * T) = 52 :=
by
  -- Placeholder for proof. This needs proper calculation.
  sorry

end true_discount_is_52_l265_265440


namespace base_price_lowered_percentage_l265_265796

theorem base_price_lowered_percentage (P : ℝ) (new_price final_price : ℝ) (x : ℝ)
    (h1 : new_price = P - (x / 100) * P)
    (h2 : final_price = 0.9 * new_price)
    (h3 : final_price = P - (14.5 / 100) * P) :
    x = 5 :=
  sorry

end base_price_lowered_percentage_l265_265796


namespace abs_inequality_solution_l265_265899

theorem abs_inequality_solution (x : ℝ) (h : |x - 4| ≤ 6) : -2 ≤ x ∧ x ≤ 10 := 
sorry

end abs_inequality_solution_l265_265899


namespace good_students_count_l265_265347

noncomputable def student_count := 25

def is_good_student (s : Nat) := s ≤ student_count

def is_troublemaker (s : Nat) := s ≤ student_count

def always_tell_truth (s : Nat) := is_good_student s

def always_lie (s : Nat) := is_troublemaker s

def condition1 (E B : Nat) := E + B = student_count

def condition2 := ∀ (x : Nat), x ≤ 5 → is_good_student x → 
  ∃ (B : Nat), B > 24 / 2

def condition3 := ∀ (x : Nat), x ≤ 20 → is_troublemaker x → 
  ∃ (B E : Nat), B = 3 * (E - 1) ∧ E + B = student_count

theorem good_students_count :
  ∃ (E : Nat), condition1 E (student_count - E) ∧ condition2 ∧ condition3 :=
sorry  -- the actual proof is not required

end good_students_count_l265_265347


namespace quadratic_roots_and_signs_l265_265888

theorem quadratic_roots_and_signs :
  (∃ x1 x2 : ℝ, (x1^2 - 13*x1 + 40 = 0) ∧ (x2^2 - 13*x2 + 40 = 0) ∧ x1 = 5 ∧ x2 = 8 ∧ 0 < x1 ∧ 0 < x2) :=
by
  sorry

end quadratic_roots_and_signs_l265_265888


namespace find_P_l265_265441

theorem find_P (P Q R S : ℕ) (h1 : P ≠ Q) (h2 : P ≠ R) (h3 : P ≠ S) (h4 : Q ≠ R) (h5 : Q ≠ S) (h6 : R ≠ S)
  (h7 : P > 0) (h8 : Q > 0) (h9 : R > 0) (h10 : S > 0)
  (hPQ : P * Q = 72) (hRS : R * S = 72) (hDiff : P - Q = R + S) : P = 12 :=
by
  sorry

end find_P_l265_265441


namespace least_sum_possible_l265_265777

theorem least_sum_possible (x y z w k : ℕ) (hpos : 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w) 
  (hx : 4 * x = k) (hy : 5 * y = k) (hz : 6 * z = k) (hw : 7 * w = k) :
  x + y + z + w = 319 := 
  sorry

end least_sum_possible_l265_265777


namespace number_of_subsets_l265_265724

theorem number_of_subsets (m n s : ℤ) (hm : m ≥ 1) (hn : n ≥ 1) 
  (coprime_mn : Int.gcd m n = 1) : 
  (∃ (A : Finset ℤ), A.card = m ∧ (A.Sum id) % n = s % n) → 
    (∃ x : ℤ, x = (Nat.choose (m + n - 1).toNat m.toNat) / n) := 
sorry

end number_of_subsets_l265_265724


namespace convex_pentagon_angle_greater_than_36_l265_265038

theorem convex_pentagon_angle_greater_than_36
  (α γ : ℝ)
  (h_sum : 5 * α + 10 * γ = 3 * Real.pi)
  (h_convex : ∀ i : Fin 5, (α + i.val * γ < Real.pi)) :
  α > Real.pi / 5 :=
sorry

end convex_pentagon_angle_greater_than_36_l265_265038


namespace team_total_score_is_correct_l265_265453

-- Define the total number of team members
def total_members : ℕ := 30

-- Define the number of members who didn't show up
def members_absent : ℕ := 8

-- Define the score per member
def score_per_member : ℕ := 4

-- Define the points deducted per incorrect answer
def points_per_incorrect_answer : ℕ := 2

-- Define the total number of incorrect answers
def total_incorrect_answers : ℕ := 6

-- Define the bonus multiplier
def bonus_multiplier : ℝ := 1.5

-- Define the total score calculation
def total_score_calculation (total_members : ℕ) (members_absent : ℕ) (score_per_member : ℕ)
  (points_per_incorrect_answer : ℕ) (total_incorrect_answers : ℕ) (bonus_multiplier : ℝ) : ℝ :=
  let members_present := total_members - members_absent
  let initial_score := members_present * score_per_member
  let total_deductions := total_incorrect_answers * points_per_incorrect_answer
  let final_score := initial_score - total_deductions
  final_score * bonus_multiplier

-- Prove that the total score is 114 points
theorem team_total_score_is_correct : total_score_calculation total_members members_absent score_per_member
  points_per_incorrect_answer total_incorrect_answers bonus_multiplier = 114 :=
by
  sorry

end team_total_score_is_correct_l265_265453


namespace common_difference_of_arithmetic_sequence_l265_265529

variable {a₁ d : ℕ}
def S (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem common_difference_of_arithmetic_sequence (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l265_265529


namespace find_m_n_difference_l265_265848

theorem find_m_n_difference (x y m n : ℤ)
  (hx : x = 2)
  (hy : y = -3)
  (hm : x + y = m)
  (hn : 2 * x - y = n) :
  m - n = -8 :=
by {
  sorry
}

end find_m_n_difference_l265_265848


namespace largest_angle_of_triangle_l265_265903

theorem largest_angle_of_triangle (A B C : ℝ) :
  A + B + C = 180 ∧ A + B = 126 ∧ abs (A - B) = 45 → max A (max B C) = 85.5 :=
by sorry

end largest_angle_of_triangle_l265_265903


namespace ratio_of_numbers_l265_265263

theorem ratio_of_numbers (x : ℝ) (h_sum : x + 3.5 = 14) : x / 3.5 = 3 :=
by
  sorry

end ratio_of_numbers_l265_265263


namespace maggie_sold_2_subscriptions_to_neighbor_l265_265141

-- Definition of the problem conditions
def maggie_pays_per_subscription : Int := 5
def maggie_subscriptions_to_parents : Int := 4
def maggie_subscriptions_to_grandfather : Int := 1
def maggie_earned_total : Int := 55

-- Define the function to be proven
def subscriptions_sold_to_neighbor (x : Int) : Prop :=
  maggie_pays_per_subscription * (maggie_subscriptions_to_parents + maggie_subscriptions_to_grandfather + x + 2*x) = maggie_earned_total

-- The statement we need to prove
theorem maggie_sold_2_subscriptions_to_neighbor :
  subscriptions_sold_to_neighbor 2 :=
sorry

end maggie_sold_2_subscriptions_to_neighbor_l265_265141


namespace sum_of_coefficients_l265_265379

theorem sum_of_coefficients (a b c : ℝ) (w : ℂ) (h_roots : ∃ w : ℂ, (∃ i : ℂ, i^2 = -1) ∧ 
  (x + ax^2 + bx + c)^3 = (w + 3*im)* (w + 9*im)*(2*w - 4)) :
  a + b + c = -136 :=
sorry

end sum_of_coefficients_l265_265379


namespace problem_statement_l265_265295

-- Define the conditions and the goal
theorem problem_statement {x y : ℝ} 
  (h1 : (x + y)^2 = 36)
  (h2 : x * y = 8) : 
  x^2 + y^2 = 20 := 
by
  sorry

end problem_statement_l265_265295


namespace amy_total_tickets_l265_265462

def amy_initial_tickets : ℕ := 33
def amy_additional_tickets : ℕ := 21

theorem amy_total_tickets : amy_initial_tickets + amy_additional_tickets = 54 := by
  sorry

end amy_total_tickets_l265_265462


namespace man_speed_l265_265781

theorem man_speed (L T V_t V_m : ℝ) (hL : L = 400) (hT : T = 35.99712023038157) (hVt : V_t = 46 * 1000 / 3600) (hE : L = (V_t - V_m) * T) : V_m = 1.666666666666684 :=
by
  sorry

end man_speed_l265_265781


namespace umbrella_cost_l265_265377

theorem umbrella_cost (number_of_umbrellas : Nat) (total_cost : Nat) (h1 : number_of_umbrellas = 3) (h2 : total_cost = 24) :
  (total_cost / number_of_umbrellas) = 8 :=
by
  -- The proof will go here
  sorry

end umbrella_cost_l265_265377


namespace infinite_squares_in_ap_l265_265458

theorem infinite_squares_in_ap
    (a d : ℤ)
    (h : ∃ n : ℤ, a^2 = a + n * d) :
    ∀ N : ℕ, ∃ m : ℤ, ∃ k : ℕ, k > N ∧ m^2 = a + k * d :=
by
  sorry

end infinite_squares_in_ap_l265_265458


namespace good_students_count_l265_265366

theorem good_students_count (E B : ℕ) 
    (h1 : E + B = 25) 
    (h2 : ∀ (e ∈ finset.range 5), B > 12) 
    (h3 : ∀ (e ∈ finset.range 20), B = 3 * (E - 1)) :
    E = 5 ∨ E = 7 :=
by
  sorry

end good_students_count_l265_265366


namespace equation_one_solution_equation_two_solution_l265_265045

variables (x : ℝ)

theorem equation_one_solution (h : 2 * (x + 3) = 5 * x) : x = 2 :=
sorry

theorem equation_two_solution (h : (x - 3) / 0.5 - (x + 4) / 0.2 = 1.6) : x = -9.2 :=
sorry

end equation_one_solution_equation_two_solution_l265_265045


namespace eccentricity_of_ellipse_l265_265892

theorem eccentricity_of_ellipse : 
  ∀ (a b c e : ℝ), a^2 = 16 → b^2 = 8 → c^2 = a^2 - b^2 → e = c / a → e = (Real.sqrt 2) / 2 := 
by 
  intros a b c e ha hb hc he
  sorry

end eccentricity_of_ellipse_l265_265892


namespace greatest_two_digit_multiple_of_17_l265_265177

theorem greatest_two_digit_multiple_of_17 : ∃ n, (n < 100) ∧ (n % 17 = 0) ∧ (∀ m, (m < 100) ∧ (m % 17 = 0) → m ≤ n) := by
  let multiples := [17, 34, 51, 68, 85]
  have ht : ∀ m, m ∈ multiples → (m < 100) ∧ (m % 17 = 0) := by
    intros m hm
    cases hm with
    | inl h => exact ⟨by norm_num, by norm_num⟩
    | inr hm' =>
      cases hm' with
      | inl h => exact ⟨by norm_num, by norm_num⟩
      | inr hm'' =>
        cases hm'' with
        | inl h => exact ⟨by norm_num, by norm_num⟩
        | inr hm''' =>
          cases hm''' with
          | inl h => exact ⟨by norm_num, by norm_num⟩
          | inr hm'''' =>
            cases hm'''' with
            | inl h => exact ⟨by norm_num, by norm_num⟩
            | inr hm => cases hm
  
  use 85
  split
  { norm_num },
  split
  { exact (by norm_num : 85 % 17 = 0) },
  intro m
  intro h
  exact list.minimum_mem multiples h

end greatest_two_digit_multiple_of_17_l265_265177


namespace infinite_triangles_with_conditions_l265_265289

theorem infinite_triangles_with_conditions :
  ∃ (A : ℝ) (B : ℝ) (C : ℝ), 
  (A > 0) ∧ (B > 0) ∧ (C > 0) ∧ (B - A = 2) ∧ (C = 4) ∧ 
  (Δ > 0) := sorry

end infinite_triangles_with_conditions_l265_265289


namespace arithmetic_sequence_c_d_sum_l265_265414

theorem arithmetic_sequence_c_d_sum (c d : ℕ) 
  (h1 : 10 - 3 = 7) 
  (h2 : 17 - 10 = 7) 
  (h3 : c = 17 + 7) 
  (h4 : d = c + 7) 
  (h5 : d + 7 = 38) :
  c + d = 55 := 
by
  sorry

end arithmetic_sequence_c_d_sum_l265_265414


namespace angle_A_is_pi_over_4_l265_265834

theorem angle_A_is_pi_over_4
  (A B C : ℝ)
  (a b c : ℝ)
  (h : a^2 = b^2 + c^2 - 2 * b * c * Real.sin A) :
  A = Real.pi / 4 :=
  sorry

end angle_A_is_pi_over_4_l265_265834


namespace total_oranges_and_apples_l265_265666

-- Given conditions as definitions
def bags_with_5_oranges_and_7_apples (m : ℕ) : ℕ × ℕ :=
  (5 * m + 1, 7 * m)

def bags_with_9_oranges_and_7_apples (n : ℕ) : ℕ × ℕ :=
  (9 * n, 7 * n + 21)

theorem total_oranges_and_apples (m n : ℕ) (k : ℕ) 
  (h1 : (5 * m + 1, 7 * m) = (9 * n, 7 * n + 21)) 
  (h2 : 4 * n ≡ 1 [MOD 5]) : 85 = 36 + 49 :=
by
  sorry

end total_oranges_and_apples_l265_265666


namespace nancy_threw_out_2_carrots_l265_265027

theorem nancy_threw_out_2_carrots :
  ∀ (x : ℕ), 12 - x + 21 = 31 → x = 2 :=
by
  sorry

end nancy_threw_out_2_carrots_l265_265027


namespace solve_inequality_l265_265746

theorem solve_inequality (a : ℝ) : 
  {x : ℝ | x^2 - (a + 2) * x + 2 * a > 0} = 
  (if a > 2 then {x | x < 2 ∨ x > a}
   else if a = 2 then {x | x ≠ 2}
   else {x | x < a ∨ x > 2}) :=
sorry

end solve_inequality_l265_265746


namespace triangle_third_side_l265_265510

theorem triangle_third_side (x : ℝ) (h1 : x > 2) (h2 : x < 6) : x = 5 :=
sorry

end triangle_third_side_l265_265510


namespace percentage_defective_l265_265250

theorem percentage_defective (examined rejected : ℚ) (h1 : examined = 66.67) (h2 : rejected = 10) :
  (rejected / examined) * 100 = 15 := by
  sorry

end percentage_defective_l265_265250


namespace common_difference_is_two_l265_265587

variable {a₁ a₂ a₃ S₃ S₂ : ℕ}
variable (d : ℕ)

-- Given condition
axiom H : 2 * S₃ = 3 * S₂ + 6

-- Definitions based on arithmetic sequence properties
def S₂ := a₁ + a₂
def S₃ := a₁ + a₂ + a₃
def a₂ := a₁ + d
def a₃ := a₁ + 2 * d

theorem common_difference_is_two : d = 2 := 
by 
  sorry

end common_difference_is_two_l265_265587


namespace a_horses_is_18_l265_265945

-- Definitions of given conditions
def total_cost : ℕ := 435
def b_share : ℕ := 180
def horses_b : ℕ := 16
def months_b : ℕ := 9
def cost_b : ℕ := horses_b * months_b

def horses_c : ℕ := 18
def months_c : ℕ := 6
def cost_c : ℕ := horses_c * months_c

def total_cost_eq (x : ℕ) : Prop :=
  x * 8 + cost_b + cost_c = total_cost

-- Statement of the proof problem
theorem a_horses_is_18 (x : ℕ) : total_cost_eq x → x = 18 := 
sorry

end a_horses_is_18_l265_265945


namespace common_difference_of_arithmetic_sequence_l265_265527

variable {a₁ d : ℕ}
def S (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem common_difference_of_arithmetic_sequence (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l265_265527


namespace proof_sum_q_p_x_l265_265843

def p (x : ℝ) : ℝ := |x| - 3
def q (x : ℝ) : ℝ := -|x|

-- define the list of x values
def x_values : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

-- define q_p_x to apply q to p of each x
def q_p_x : List ℝ := x_values.map (λ x => q (p x))

-- define the sum of q(p(x)) for given x values
def sum_q_p_x : ℝ := q_p_x.sum

theorem proof_sum_q_p_x : sum_q_p_x = -15 := by
  -- steps of solution
  sorry

end proof_sum_q_p_x_l265_265843


namespace net_calorie_deficit_l265_265986

-- Define the conditions as constants.
def total_distance : ℕ := 3
def calories_burned_per_mile : ℕ := 150
def calories_in_candy_bar : ℕ := 200

-- Prove the net calorie deficit.
theorem net_calorie_deficit : total_distance * calories_burned_per_mile - calories_in_candy_bar = 250 := by
  sorry

end net_calorie_deficit_l265_265986


namespace sum_of_primes_is_prime_l265_265755

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

theorem sum_of_primes_is_prime (P Q : ℕ) :
  is_prime P → is_prime Q → is_prime (P - Q) → is_prime (P + Q) →
  ∃ n : ℕ, n = P + Q + (P - Q) + (P + Q) ∧ is_prime n := by
  sorry

end sum_of_primes_is_prime_l265_265755


namespace blue_lipstick_count_l265_265605

def total_students : Nat := 200

def colored_lipstick_students (total : Nat) : Nat :=
  total / 2

def red_lipstick_students (colored : Nat) : Nat :=
  colored / 4

def blue_lipstick_students (red : Nat) : Nat :=
  red / 5

theorem blue_lipstick_count :
  blue_lipstick_students (red_lipstick_students (colored_lipstick_students total_students)) = 5 := 
sorry

end blue_lipstick_count_l265_265605


namespace arithmetic_sequence_sum_l265_265981

-- Define the arithmetic sequence properties
def is_arithmetic_sequence (seq : ℕ → ℕ) :=
  ∀ n : ℕ, seq (n + 1) = seq n + 2

-- Define the arithmetic sequence in question
def sequence : ℕ → ℕ
| 0       := 2
| (n + 1) := sequence n + 2

-- Check that our sequence matches the properties of an arithmetic sequence
lemma sequence_is_arithmetic : is_arithmetic_sequence sequence :=
by intros n; simp [sequence]

-- Define the sum of the first n terms of the sequence
def sum_n_terms (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, sequence i

-- State the main theorem to be proven: the sum of the first 10 terms is 110
theorem arithmetic_sequence_sum : sum_n_terms 10 = 110 :=
sorry

end arithmetic_sequence_sum_l265_265981


namespace max_min_sum_l265_265255

noncomputable def f : ℝ → ℝ := sorry

-- Define the interval and properties of the function f
def within_interval (x : ℝ) : Prop := -2016 ≤ x ∧ x ≤ 2016
def functional_eq (x1 x2 : ℝ) : Prop := f (x1 + x2) = f x1 + f x2 - 2016
def less_than_2016_proof (x : ℝ) : Prop := x > 0 → f x < 2016

-- Define the minimum and maximum values of the function f
def M : ℝ := sorry
def N : ℝ := sorry

-- Prove that M + N = 4032 given the properties and conditions
theorem max_min_sum : 
  (∀ x1 x2, within_interval x1 → within_interval x2 → functional_eq x1 x2) →
  (∀ x, x > 0 → less_than_2016_proof x) →
  M + N = 4032 :=
by {
  -- Define the formal proof here, placeholder for actual proof
  sorry
}

end max_min_sum_l265_265255


namespace greatest_two_digit_multiple_of_17_l265_265205

theorem greatest_two_digit_multiple_of_17 : ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (17 ∣ n) ∧ n = 85 :=
by {
    use 85,
    split,
    { exact ⟨le_of_eq rfl, lt_of_le_of_ne (nat.le_of_eq rfl) (ne.symm (dec_trivial))⟩ },
    split,
    { exact dvd.intro 5 rfl },
    { refl }
}

end greatest_two_digit_multiple_of_17_l265_265205


namespace greatest_two_digit_multiple_of_17_l265_265180

theorem greatest_two_digit_multiple_of_17 : ∃ (n : ℕ), n < 100 ∧ 10 ≤ n ∧ 17 ∣ n ∧ ∀ m : ℕ, m < 100 → 10 ≤ m → 17 ∣ m → m ≤ n :=
begin
  use 85,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m hm1 hm2 hm3,
    by_contradiction hmn,
    have h : m > 85 := not_le.mp hmn,
    linear_combination h * 17,
    sorry,
  },
end

end greatest_two_digit_multiple_of_17_l265_265180


namespace correct_equation_l265_265773

theorem correct_equation :
  (2 * Real.sqrt 2) / (Real.sqrt 2) = 2 :=
by
  -- Proof goes here
  sorry

end correct_equation_l265_265773


namespace area_square_hypotenuse_l265_265084

theorem area_square_hypotenuse 
(a : ℝ) 
(h1 : ∀ a: ℝ,  ∃ YZ: ℝ, YZ = a + 3) 
(h2: ∀ XY: ℝ, ∃ total_area: ℝ, XY^2 + XY * (XY + 3) + (2 * XY^2 + 6 * XY + 9) = 450) :
  ∃ XZ: ℝ, (2 * a^2 + 6 * a + 9 = XZ) → XZ = 201 := by
  sorry

end area_square_hypotenuse_l265_265084


namespace greatest_two_digit_multiple_of_17_l265_265212

theorem greatest_two_digit_multiple_of_17 : ∃ m : ℕ, (10 ≤ m) ∧ (m ≤ 99) ∧ (17 ∣ m) ∧ (∀ n : ℕ, (10 ≤ n) ∧ (n ≤ 99) ∧ (17 ∣ n) → n ≤ m) ∧ m = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l265_265212


namespace min_cells_marked_l265_265390

theorem min_cells_marked (grid_size : ℕ) (triomino_size : ℕ) (total_cells : ℕ) : 
  grid_size = 5 ∧ triomino_size = 3 ∧ total_cells = grid_size * grid_size → ∃ m, m = 9 :=
by
  intros h
  -- Placeholder for detailed proof steps
  sorry

end min_cells_marked_l265_265390


namespace a_eq_b_if_b2_ab_1_divides_a2_ab_1_l265_265726

theorem a_eq_b_if_b2_ab_1_divides_a2_ab_1 (a b : ℕ) (ha_pos : a > 0) (hb_pos : b > 0)
  (h : b^2 + a * b + 1 ∣ a^2 + a * b + 1) : a = b :=
by
  sorry

end a_eq_b_if_b2_ab_1_divides_a2_ab_1_l265_265726


namespace sum_pos_integers_9_l265_265319

theorem sum_pos_integers_9 (x y z : ℕ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h_eq : 30 / 7 = x + 1 / (y + 1 / z)) : x + y + z = 9 :=
sorry

end sum_pos_integers_9_l265_265319


namespace geometric_series_sum_example_l265_265089

def geometric_series_sum (a r : ℤ) (n : ℕ) : ℤ :=
  a * ((r ^ n - 1) / (r - 1))

theorem geometric_series_sum_example : geometric_series_sum 2 (-3) 8 = -3280 :=
by
  sorry

end geometric_series_sum_example_l265_265089


namespace net_calorie_deficit_l265_265985

-- Define the conditions as constants.
def total_distance : ℕ := 3
def calories_burned_per_mile : ℕ := 150
def calories_in_candy_bar : ℕ := 200

-- Prove the net calorie deficit.
theorem net_calorie_deficit : total_distance * calories_burned_per_mile - calories_in_candy_bar = 250 := by
  sorry

end net_calorie_deficit_l265_265985


namespace no_consecutive_squares_l265_265104

open Nat

-- Define a function to get the n-th prime number
def prime (n : ℕ) : ℕ := sorry -- Use an actual function or sequence that generates prime numbers, this is a placeholder.

-- Define the sequence S_n, the sum of the first n prime numbers
def S : ℕ → ℕ
| 0       => 0
| (n + 1) => S n + prime (n + 1)

-- Define a predicate to check if a number is a perfect square
def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

-- The theorem that no two consecutive terms S_{n-1} and S_n can both be perfect squares
theorem no_consecutive_squares (n : ℕ) : ¬ (is_square (S n) ∧ is_square (S (n + 1))) :=
by
  sorry

end no_consecutive_squares_l265_265104


namespace number_of_blue_butterflies_l265_265381

theorem number_of_blue_butterflies 
  (total_butterflies : ℕ)
  (B Y : ℕ)
  (H1 : total_butterflies = 11)
  (H2 : B = 2 * Y)
  (H3 : total_butterflies = B + Y + 5) : B = 4 := 
sorry

end number_of_blue_butterflies_l265_265381


namespace greatest_two_digit_multiple_of_17_l265_265191

theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n < 100 ∧ n ≥ 10 ∧ 17 ∣ n ∧ ∀ m, m < 100 ∧ m ≥ 10 ∧ 17 ∣ m → m ≤ 85 :=
by
  use 85
  -- Prove conditions follow sorry
  sorry

end greatest_two_digit_multiple_of_17_l265_265191


namespace prove_common_difference_l265_265576

variable {a1 d : ℤ}
variable (S : ℕ → ℤ)
variable (arith_seq : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
def sum_first_n (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Define the condition given in the problem
def problem_condition : Prop := 2 * sum_first_n 3 = 3 * sum_first_n 2 + 6

-- Define that the common difference d is 2
def d_value_is_2 : Prop := d = 2

theorem prove_common_difference (h : problem_condition) : d_value_is_2 :=
by
  sorry

end prove_common_difference_l265_265576


namespace final_hair_length_l265_265882

-- Define the initial conditions and the expected final result.
def initial_hair_length : ℕ := 14
def hair_growth (x : ℕ) : ℕ := x
def hair_cut : ℕ := 20

-- Prove that the final hair length is x - 6.
theorem final_hair_length (x : ℕ) : initial_hair_length + hair_growth x - hair_cut = x - 6 :=
by
  sorry

end final_hair_length_l265_265882


namespace chord_length_is_sqrt_6_l265_265694

open Real

def circle_eq (x y : ℝ) := x^2 + y^2 + 4 * x - 4 * y + 6 = 0

def line_eq (k x y : ℝ) := k * x + y + 4 = 0

def point_on_line (k : ℝ) := (0, k)

def line_m (k x y : ℝ) := y = x + k

noncomputable def chord_length (k : ℝ) : ℝ := 
  let cx := -2
  let cy := 2
  let d := abs (cx + cy + k) / sqrt 2 in
  2 * sqrt (2 - d^2)

theorem chord_length_is_sqrt_6 (k : ℝ) 
  (h_line_symm : ∀ x y : ℝ, line_eq k x y → circle_eq x y)
  (h_point_on_line : (0, k) = point_on_line k) :
  chord_length k = sqrt 6 := by
  sorry

end chord_length_is_sqrt_6_l265_265694


namespace find_common_difference_l265_265556

section
variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

-- Condition: S_n represents the sum of the first n terms of the arithmetic sequence {a_n}
def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop := 
  S n = (n * (2 * a 1 + (n - 1) * d)) / 2

-- Condition: 2S_3 = 3S_2 + 6
def arithmetic_condition (S : ℕ → ℝ) : Prop :=
  2 * S 3 = 3 * S 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem find_common_difference 
  (h₁ : sum_arithmetic_sequence S a 2)
  (h₂ : sum_arithmetic_sequence S a 3)
  (h₃ : arithmetic_condition S) :
  d = 2 :=
sorry
end

end find_common_difference_l265_265556


namespace part_a_part_b_l265_265147

open Matrix

-- Definition of the board size
def board_size : ℕ := 7

-- A configuration of Xs on a board is a matrix of size 7x7 with boolean entries
def Board := Matrix (Fin board_size) (Fin board_size) Bool

-- Part (a): A specific configuration where no 4 cells form a rectangle
def no_castles_config : Board :=
 λ r c, 
   (r.val, c.val) ∈ [
     (0, 0), (0, 1), (0, 3), 
     (1, 1), (1, 2), (1, 4), 
     (2, 2), (2, 3), (2, 5), 
     (3, 3), (3, 4), (3, 6), 
     (4, 0), (4, 4), (4, 5), 
     (5, 1), (5, 5), (5, 6), 
     (6, 0), (6, 2), (6, 6)]

-- Definition to check if four given points form a rectangle
def forms_rectangle (a b c d : (Fin board_size) × (Fin board_size)) : Prop :=
  (a.1 = b.1 ∧ c.1 = d.1 ∧ a.2 = c.2 ∧ b.2 = d.2) ∨
  (a.1 = c.1 ∧ b.1 = d.1 ∧ a.2 = b.2 ∧ c.2 = d.2) ∨
  (a.1 = d.1 ∧ b.1 = c.1 ∧ a.2 = b.2 ∧ d.2 = c.2)

-- Part (a) theorem
theorem part_a : 
  ¬ ∃ (a b c d : (Fin board_size) × (Fin board_size)), 
    no_castles_config a.1 a.2 ∧ no_castles_config b.1 b.2 ∧ 
    no_castles_config c.1 c.2 ∧ no_castles_config d.1 d.2 ∧ 
    forms_rectangle a b c d := 
by
  sorry

-- Part (b): For any configuration of 22 marked cells, there is at least one rectangle
theorem part_b (marked_cells : Fin 22 → (Fin board_size) × (Fin board_size)) :
  ∃ (a b c d : (Fin board_size) × (Fin board_size)),
    (∃ i, marked_cells i = a) ∧ (∃ i, marked_cells i = b) ∧
    (∃ i, marked_cells i = c) ∧ (∃ i, marked_cells i = d) ∧
    forms_rectangle a b c d := 
by
  sorry

end part_a_part_b_l265_265147


namespace possible_sets_C_l265_265477

def M : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

def is_partition (A B C : Set ℕ) : Prop :=
  A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧ A ∪ B ∪ C = M

def conditions (A B C : Set ℕ) : Prop :=
  is_partition A B C ∧ (∃ (a1 a2 a3 a4 b1 b2 b3 b4 c1 c2 c3 c4 : ℕ), 
    A = {a1, a2, a3, a4} ∧
    B = {b1, b2, b3, b4} ∧
    C = {c1, c2, c3, c4} ∧
    c1 < c2 ∧ c2 < c3 ∧ c3 < c4 ∧
    a1 + b1 = c1 ∧ a2 + b2 = c2 ∧ a3 + b3 = c3 ∧ a4 + b4 = c4)

theorem possible_sets_C (A B C : Set ℕ) (h : conditions A B C) :
  C = {8, 9, 10, 12} ∨ C = {7, 9, 11, 12} ∨ C = {6, 10, 11, 12} :=
sorry

end possible_sets_C_l265_265477


namespace TruckCapacities_RentalPlanExists_MinimumRentalCost_l265_265785

-- Problem 1
theorem TruckCapacities (x y : ℕ) (h1: 2 * x + y = 10) (h2: x + 2 * y = 11) :
  x = 3 ∧ y = 4 :=
by
  sorry

-- Problem 2
theorem RentalPlanExists (a b : ℕ) (h: 3 * a + 4 * b = 31) :
  (a = 9 ∧ b = 1) ∨ (a = 5 ∧ b = 4) ∨ (a = 1 ∧ b = 7) :=
by
  sorry

-- Problem 3
theorem MinimumRentalCost (a b : ℕ) (h1: 3 * a + 4 * b = 31) 
  (h2: 100 * a + 120 * b = 940) :
  ∃ a b, a = 1 ∧ b = 7 :=
by
  sorry

end TruckCapacities_RentalPlanExists_MinimumRentalCost_l265_265785


namespace crosswalk_red_light_wait_l265_265461

theorem crosswalk_red_light_wait :
  let red_light_duration := 40
  let wait_time_requirement := 15
  let favorable_duration := red_light_duration - wait_time_requirement
  (favorable_duration : ℝ) / red_light_duration = (5 : ℝ) / 8 :=
by
  sorry

end crosswalk_red_light_wait_l265_265461


namespace common_difference_is_two_l265_265588

variable {a₁ a₂ a₃ S₃ S₂ : ℕ}
variable (d : ℕ)

-- Given condition
axiom H : 2 * S₃ = 3 * S₂ + 6

-- Definitions based on arithmetic sequence properties
def S₂ := a₁ + a₂
def S₃ := a₁ + a₂ + a₃
def a₂ := a₁ + d
def a₃ := a₁ + 2 * d

theorem common_difference_is_two : d = 2 := 
by 
  sorry

end common_difference_is_two_l265_265588


namespace correct_operation_l265_265629

theorem correct_operation : 
  ¬(2 + sqrt 2 = 2 * sqrt 2) ∧
  ¬(4 * x^2 * y - x^2 * y = 3) ∧
  ¬((a + b)^2 = a^2 + b^2) ∧
  ((ab)^3 = a^3 * b^3) :=
by sorry

end correct_operation_l265_265629


namespace rectangular_to_polar_coordinates_l265_265682

theorem rectangular_to_polar_coordinates :
  ∃ r θ, (r > 0) ∧ (0 ≤ θ ∧ θ < 2 * Real.pi) ∧ (r, θ) = (5, 7 * Real.pi / 4) :=
by
  sorry

end rectangular_to_polar_coordinates_l265_265682


namespace josh_money_left_l265_265011

theorem josh_money_left (initial_amount : ℝ) (first_spend : ℝ) (second_spend : ℝ) 
  (h1 : initial_amount = 9) 
  (h2 : first_spend = 1.75) 
  (h3 : second_spend = 1.25) : 
  initial_amount - first_spend - second_spend = 6 := 
by 
  sorry

end josh_money_left_l265_265011


namespace L_shape_area_and_perimeter_l265_265465

def rectangle1_length := 0.5
def rectangle1_width := 0.3
def rectangle2_length := 0.2
def rectangle2_width := 0.5

def area_rectangle1 := rectangle1_length * rectangle1_width
def area_rectangle2 := rectangle2_length * rectangle2_width
def total_area := area_rectangle1 + area_rectangle2

def perimeter_L_shape := rectangle1_length + rectangle1_width + rectangle1_width + rectangle2_length + rectangle2_length + rectangle2_width

theorem L_shape_area_and_perimeter :
  total_area = 0.25 ∧ perimeter_L_shape = 2.0 :=
by
  sorry

end L_shape_area_and_perimeter_l265_265465


namespace emily_jumps_75_seconds_l265_265219

/-- Emily jumps 52 times in 60 seconds, maintaining the same rate.
    Prove that she jumps 65 times in 75 seconds. -/
theorem emily_jumps_75_seconds (
    jumps_per_60_seconds : ℚ := 52 / 60
) : (75 * jumps_per_60_seconds = 65) :=
by
  -- Simplify the jumps per 60 seconds rate
  have rate : ℚ := 13 / 15

  -- Calculate the jumps in 75 seconds
  have jumps_75 : ℚ := 75 * rate

  -- Conclude that the number of jumps in 75 seconds is 65
  show jumps_75 = 65, by
    calc
      jumps_75 = 75 * (13 / 15) : by rw [rate]
      ... = (75 / 15) * 13 : by rw [mul_div_assoc]
      ... = 5 * 13 : by norm_num
      ... = 65 : by norm_num

end emily_jumps_75_seconds_l265_265219


namespace amount_spent_on_milk_l265_265801

-- Define conditions
def monthly_salary (S : ℝ) := 0.10 * S = 1800
def rent := 5000
def groceries := 4500
def education := 2500
def petrol := 2000
def miscellaneous := 700
def total_expenses (S : ℝ) := S - 1800
def known_expenses := rent + groceries + education + petrol + miscellaneous

-- Define the proof problem
theorem amount_spent_on_milk (S : ℝ) (milk : ℝ) :
  monthly_salary S →
  total_expenses S = known_expenses + milk →
  milk = 1500 :=
by
  sorry

end amount_spent_on_milk_l265_265801


namespace max_chips_can_be_removed_l265_265034

theorem max_chips_can_be_removed (initial_chips : (Fin 10) × (Fin 10) → ℕ) 
  (condition : ∀ i j, initial_chips (i, j) = 1) : 
    ∃ removed_chips : ℕ, removed_chips = 90 :=
by
  sorry

end max_chips_can_be_removed_l265_265034


namespace patriots_won_games_l265_265894

theorem patriots_won_games (C P M S T E : ℕ) 
  (hC : C > 25)
  (hPC : P > C)
  (hMP : M > P)
  (hSC : S > C)
  (hSP : S < P)
  (hTE : T > E) : 
  P = 35 :=
sorry

end patriots_won_games_l265_265894


namespace data_a_value_l265_265612

theorem data_a_value (a b c : ℕ) (h1 : a + b = c) (h2 : b = 3 * a) (h3 : a + b + c = 96) : a = 12 :=
by
  sorry

end data_a_value_l265_265612


namespace tom_calories_l265_265915

theorem tom_calories :
  let carrot_pounds := 1
  let broccoli_pounds := 2 * carrot_pounds
  let carrot_calories_per_pound := 51
  let broccoli_calories_per_pound := carrot_calories_per_pound / 3
  let total_carrot_calories := carrot_pounds * carrot_calories_per_pound
  let total_broccoli_calories := broccoli_pounds * broccoli_calories_per_pound
  let total_calories := total_carrot_calories + total_broccoli_calories
  total_calories = 85 :=
by
  sorry

end tom_calories_l265_265915


namespace sam_total_pennies_l265_265152

def a : ℕ := 98
def b : ℕ := 93

theorem sam_total_pennies : a + b = 191 :=
by
  sorry

end sam_total_pennies_l265_265152


namespace number_of_bricks_needed_l265_265704

theorem number_of_bricks_needed :
  ∀ (brick_length brick_width brick_height wall_length wall_height wall_width : ℝ),
  brick_length = 25 → 
  brick_width = 11.25 → 
  brick_height = 6 → 
  wall_length = 750 → 
  wall_height = 600 → 
  wall_width = 22.5 → 
  (wall_length * wall_height * wall_width) / (brick_length * brick_width * brick_height) = 6000 :=
by
  intros brick_length brick_width brick_height wall_length wall_height wall_width
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end number_of_bricks_needed_l265_265704


namespace good_students_count_l265_265351

noncomputable def student_count := 25

def is_good_student (s : Nat) := s ≤ student_count

def is_troublemaker (s : Nat) := s ≤ student_count

def always_tell_truth (s : Nat) := is_good_student s

def always_lie (s : Nat) := is_troublemaker s

def condition1 (E B : Nat) := E + B = student_count

def condition2 := ∀ (x : Nat), x ≤ 5 → is_good_student x → 
  ∃ (B : Nat), B > 24 / 2

def condition3 := ∀ (x : Nat), x ≤ 20 → is_troublemaker x → 
  ∃ (B E : Nat), B = 3 * (E - 1) ∧ E + B = student_count

theorem good_students_count :
  ∃ (E : Nat), condition1 E (student_count - E) ∧ condition2 ∧ condition3 :=
sorry  -- the actual proof is not required

end good_students_count_l265_265351


namespace partial_fraction_product_l265_265049

theorem partial_fraction_product :
  ∃ (A B C : ℚ), 
  (∀ x : ℚ, x ≠ 1 ∧ x ≠ -3 ∧ x ≠ 4 → 
    (x^2 - 4) / (x^3 + x^2 - 11 * x - 13) = A / (x - 1) + B / (x + 3) + C / (x - 4)) ∧
  A * B * C = 5 / 196 :=
sorry

end partial_fraction_product_l265_265049


namespace smallest_multiple_of_3_l265_265900

theorem smallest_multiple_of_3 (a : ℕ) (h : ∀ i j : ℕ, i < 6 → j < 6 → 3 * (a + i) = 3 * (a + 10 + j) → a = 50) : 3 * a = 150 :=
by
  sorry

end smallest_multiple_of_3_l265_265900


namespace magician_assistant_strategy_l265_265655

-- Define the possible states of a coin
inductive CoinState | heads | tails

-- Define the circle of coins
def coinCircle := Fin 11 → CoinState

-- The strategy ensures there are adjacent coins with the same state
theorem magician_assistant_strategy (c : coinCircle) :
  ∃ i : Fin 11, c i = c ((i + 1) % 11) := by sorry

end magician_assistant_strategy_l265_265655


namespace ways_to_change_12_dollars_into_nickels_and_quarters_l265_265592

theorem ways_to_change_12_dollars_into_nickels_and_quarters :
  ∃ n q : ℕ, 5 * n + 25 * q = 1200 ∧ n > 0 ∧ q > 0 ∧ ∀ q', (q' ≥ 1 ∧ q' ≤ 47) ↔ (n = 240 - 5 * q') :=
by
  sorry

end ways_to_change_12_dollars_into_nickels_and_quarters_l265_265592


namespace Al_initial_portion_l265_265248

theorem Al_initial_portion (a b c : ℕ) 
  (h1 : a + b + c = 1200) 
  (h2 : a - 150 + 2 * b + 3 * c = 1800) 
  (h3 : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  a = 550 :=
by {
  sorry
}

end Al_initial_portion_l265_265248


namespace adam_and_simon_time_to_be_80_miles_apart_l265_265247

theorem adam_and_simon_time_to_be_80_miles_apart :
  ∃ x : ℝ, (10 * x)^2 + (8 * x)^2 = 80^2 ∧ x = 6.25 :=
by
  sorry

end adam_and_simon_time_to_be_80_miles_apart_l265_265247


namespace common_difference_of_arithmetic_sequence_l265_265531

variable {a₁ d : ℕ}
def S (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem common_difference_of_arithmetic_sequence (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l265_265531


namespace arithmetic_sequence_common_difference_l265_265537

variable {a₁ d : ℕ}
variable S : ℕ → ℕ

-- Definitions of the sums S₂ and S₃ in an arithmetic sequence
def S₂ : ℕ := a₁ + (a₁ + d)
def S₃ : ℕ := a₁ + (a₁ + d) + (a₁ + 2 * d)

theorem arithmetic_sequence_common_difference (h : 2 * S₃ = 3 * S₂ + 6) : d = 2 :=
by
  -- Skip the proof.
  sorry

end arithmetic_sequence_common_difference_l265_265537


namespace find_certain_number_l265_265401

theorem find_certain_number : ∃ x : ℕ, (((x - 50) / 4) * 3 + 28 = 73) → x = 110 :=
by
  sorry

end find_certain_number_l265_265401


namespace onions_on_scale_l265_265762

theorem onions_on_scale (N : ℕ) (W_total : ℕ) (W_removed : ℕ) (avg_remaining : ℕ) (avg_removed : ℕ) :
  W_total = 7680 →
  W_removed = 5 * 206 →
  avg_remaining = 190 →
  avg_removed = 206 →
  N = 40 :=
by
  sorry

end onions_on_scale_l265_265762


namespace integer_solutions_count_l265_265846

theorem integer_solutions_count :
  (∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 15 ∧
    ∀ (pair : ℕ × ℕ), pair ∈ pairs ↔ (∃ x y, pair = (x, y) ∧ (Nat.sqrt x + Nat.sqrt y = 14))) :=
by
  sorry

end integer_solutions_count_l265_265846


namespace quadratic_inequality_solution_l265_265854

theorem quadratic_inequality_solution (a b c : ℝ) (h : a < 0) 
  (h_sol : ∀ x, ax^2 + bx + c > 0 ↔ x > -2 ∧ x < 1) :
  ∀ x, ax^2 + (a + b) * x + c - a < 0 ↔ x < -3 ∨ x > 1 := 
sorry

end quadratic_inequality_solution_l265_265854


namespace max_value_min_4x_y_4y_x2_5y2_l265_265018

theorem max_value_min_4x_y_4y_x2_5y2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ t, t = min (4 * x + y) (4 * y / (x^2 + 5 * y^2)) ∧ t ≤ 2 :=
by
  sorry

end max_value_min_4x_y_4y_x2_5y2_l265_265018


namespace smallest_possible_n_l265_265150

theorem smallest_possible_n (n : ℕ) (h1 : n % 6 = 4) (h2 : n % 7 = 3) (h3 : n > 20) : n = 52 := 
sorry

end smallest_possible_n_l265_265150


namespace recipe_sugar_amount_l265_265600

theorem recipe_sugar_amount (F_total F_added F_additional F_needed S : ℕ)
  (h1 : F_total = 9)
  (h2 : F_added = 2)
  (h3 : F_additional = S + 1)
  (h4 : F_needed = F_total - F_added)
  (h5 : F_needed = F_additional) :
  S = 6 := 
sorry

end recipe_sugar_amount_l265_265600


namespace part1_part2_l265_265270

noncomputable def f (x a : ℝ) : ℝ := |x + a|
noncomputable def g (x : ℝ) : ℝ := |x + 3| - x

theorem part1 (x : ℝ) : f x 1 < g x → x < 2 :=
sorry

theorem part2 (a : ℝ) : (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x a < g x) → -2 < a ∧ a < 2 :=
sorry

end part1_part2_l265_265270


namespace product_of_two_numbers_l265_265619

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y ≠ 0) 
  (h2 : (x + y) / (x - y) = 7)
  (h3 : xy = 24 * (x - y)) : xy = 48 := 
sorry

end product_of_two_numbers_l265_265619


namespace find_number_of_good_students_l265_265327

variable {G T : ℕ}

def is_good_or_troublemaker (G T : ℕ) : Prop :=
  G + T = 25 ∧ 
  (∀ s ∈ finset.range 5, T > 12) ∧ 
  (∀ s ∈ finset.range 20, T = 3 * (G - 1))

-- There are 25 students in total. 
-- Each student is either a good student or a troublemaker.
-- Good students always tell the truth, while troublemakers always lie.
-- Five students make the statement: "If I transfer to another class, more than half of the remaining students will be troublemakers."
-- The remaining 20 students make the statement: "If I transfer to another class, the number of troublemakers among the remaining students will be three times the number of good students."

theorem find_number_of_good_students (G T : ℕ) (h : is_good_or_troublemaker G T) :
  (G = 5 ∨ G = 7) :=
sorry

end find_number_of_good_students_l265_265327


namespace largest_n_rational_sqrt_l265_265999

theorem largest_n_rational_sqrt : ∃ n : ℕ, 
  (∀ k l : ℤ, k = Int.natAbs (Int.sqrt (n - 100)) ∧ l = Int.natAbs (Int.sqrt (n + 100)) → 
  k + l = 100) ∧ 
  (n = 2501) :=
by
  sorry

end largest_n_rational_sqrt_l265_265999


namespace ratio_jordana_jennifer_10_years_l265_265375

-- Let's define the necessary terms and conditions:
def Jennifer_future_age := 30
def Jordana_current_age := 80
def years := 10

-- Define the ratio of ages function:
noncomputable def ratio_of_ages (future_age_jen : ℕ) (current_age_jord : ℕ) (yrs : ℕ) : ℚ :=
  (current_age_jord + yrs) / future_age_jen

-- The statement we need to prove:
theorem ratio_jordana_jennifer_10_years :
  ratio_of_ages Jennifer_future_age Jordana_current_age years = 3 := by
  sorry

end ratio_jordana_jennifer_10_years_l265_265375


namespace price_per_vanilla_cookie_l265_265774

theorem price_per_vanilla_cookie (P : ℝ) (h1 : 220 + 70 * P = 360) : P = 2 := 
by 
  sorry

end price_per_vanilla_cookie_l265_265774


namespace good_students_l265_265360

theorem good_students (G T : ℕ) (h1 : G + T = 25) (h2 : T > 12) (h3 : T = 3 * (G - 1)) :
  G = 5 ∨ G = 7 :=
by
  sorry

end good_students_l265_265360


namespace good_students_count_l265_265349

noncomputable def student_count := 25

def is_good_student (s : Nat) := s ≤ student_count

def is_troublemaker (s : Nat) := s ≤ student_count

def always_tell_truth (s : Nat) := is_good_student s

def always_lie (s : Nat) := is_troublemaker s

def condition1 (E B : Nat) := E + B = student_count

def condition2 := ∀ (x : Nat), x ≤ 5 → is_good_student x → 
  ∃ (B : Nat), B > 24 / 2

def condition3 := ∀ (x : Nat), x ≤ 20 → is_troublemaker x → 
  ∃ (B E : Nat), B = 3 * (E - 1) ∧ E + B = student_count

theorem good_students_count :
  ∃ (E : Nat), condition1 E (student_count - E) ∧ condition2 ∧ condition3 :=
sorry  -- the actual proof is not required

end good_students_count_l265_265349


namespace arithmetic_evaluation_l265_265823

theorem arithmetic_evaluation : (64 / 0.08) - 2.5 = 797.5 :=
by
  sorry

end arithmetic_evaluation_l265_265823


namespace probability_of_rolling_2_or_4_l265_265057

theorem probability_of_rolling_2_or_4 (fair : ℕ) (sides : fin 6) : 
  (2/6 : ℚ) = (1/3 : ℚ) := 
by 
  sorry

end probability_of_rolling_2_or_4_l265_265057


namespace A_3_2_eq_29_l265_265811

-- Define the recursive function A(m, n).
def A : Nat → Nat → Nat
| 0, n => n + 1
| (m + 1), 0 => A m 1
| (m + 1), (n + 1) => A m (A (m + 1) n)

-- Prove that A(3, 2) = 29
theorem A_3_2_eq_29 : A 3 2 = 29 := by 
  sorry

end A_3_2_eq_29_l265_265811


namespace prove_common_difference_l265_265574

variable {a1 d : ℤ}
variable (S : ℕ → ℤ)
variable (arith_seq : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
def sum_first_n (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Define the condition given in the problem
def problem_condition : Prop := 2 * sum_first_n 3 = 3 * sum_first_n 2 + 6

-- Define that the common difference d is 2
def d_value_is_2 : Prop := d = 2

theorem prove_common_difference (h : problem_condition) : d_value_is_2 :=
by
  sorry

end prove_common_difference_l265_265574


namespace arithmetic_common_difference_l265_265582

-- Defining the arithmetic sequence sum function S
def S (a₁ a₂ a₃ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if n = 3 then a₁ + a₂ + a₃
  else if n = 2 then a₁ + a₂
  else 0

-- The Lean statement for the problem
theorem arithmetic_common_difference (a₁ a₂ a₃ d : ℝ) 
  (h1 : a₂ = a₁ + d) 
  (h2 : a₃ = a₂ + d)
  (h3 : 2 * S a₁ a₂ a₃ d 3 = 3 * S a₁ a₂ a₃ d 2 + 6) :
  d = 2 :=
by
  sorry

end arithmetic_common_difference_l265_265582


namespace number_of_good_students_is_5_or_7_l265_265352

-- Definitions based on the conditions
def total_students : ℕ := 25
def number_of_good_students (G : ℕ) (T : ℕ) := G + T = total_students
def first_group_condition (T : ℕ) := T > 12
def second_group_condition (G : ℕ) (T : ℕ) := T = 3 * (G - 1)

-- Problem statement in Lean 4:
theorem number_of_good_students_is_5_or_7 (G T : ℕ) :
  number_of_good_students G T ∧ first_group_condition T ∧ second_group_condition G T → G = 5 ∨ G = 7 :=
by
  sorry

end number_of_good_students_is_5_or_7_l265_265352


namespace probability_of_specific_roll_l265_265028

noncomputable def probability_event : ℚ :=
  let favorable_outcomes_first_die := 3 -- 1, 2, 3
  let total_outcomes_die := 8
  let probability_first_die := favorable_outcomes_first_die / total_outcomes_die
  
  let favorable_outcomes_second_die := 4 -- 5, 6, 7, 8
  let probability_second_die := favorable_outcomes_second_die / total_outcomes_die
  
  probability_first_die * probability_second_die

theorem probability_of_specific_roll :
  probability_event = 3 / 16 := 
  by
    sorry

end probability_of_specific_roll_l265_265028


namespace cost_to_feed_turtles_l265_265154

theorem cost_to_feed_turtles :
  (∀ (weight : ℚ), weight > 0 → (food_per_half_pound_ounces = 1) →
    (total_weight_pounds = 30) →
    (jar_contents_ounces = 15 ∧ jar_cost = 2) →
    (total_cost_dollars = 8)) :=
by
  sorry

variables
  (food_per_half_pound_ounces : ℚ := 1)
  (total_weight_pounds : ℚ := 30)
  (jar_contents_ounces : ℚ := 15)
  (jar_cost : ℚ := 2)
  (total_cost_dollars : ℚ := 8)

-- Assuming all variables needed to state the theorem exist and are meaningful
/-!
  Given:
  - Each turtle needs 1 ounce of food per 1/2 pound of body weight
  - Total turtles' weight is 30 pounds
  - Each jar of food contains 15 ounces and costs $2

  Prove:
  - The total cost to feed Sylvie's turtles is $8
-/

end cost_to_feed_turtles_l265_265154
