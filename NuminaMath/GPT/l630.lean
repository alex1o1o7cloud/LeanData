import Mathlib

namespace pairs_satisfied_condition_l630_63096

def set_A : Set ℕ := {1, 2, 3, 4, 5, 6, 10, 11, 12, 15, 20, 22, 30, 33, 44, 55, 60, 66, 110, 132, 165, 220, 330, 660}
def set_B : Set ℕ := {1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72}

def is_valid_pair (a b : ℕ) := a ∈ set_A ∧ b ∈ set_B ∧ (a - b = 4)

def valid_pairs : Set (ℕ × ℕ) := 
  {(6, 2), (10, 6), (12, 8), (22, 18)}

theorem pairs_satisfied_condition :
  { (a, b) | is_valid_pair a b } = valid_pairs := 
sorry

end pairs_satisfied_condition_l630_63096


namespace tan_45_deg_eq_one_l630_63011

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l630_63011


namespace c_work_rate_l630_63043

noncomputable def work_rate (days : ℕ) : ℝ := 1 / days

theorem c_work_rate (A B C: ℝ) 
  (h1 : A + B = work_rate 28) 
  (h2 : A + B + C = work_rate 21) : C = work_rate 84 := by
  -- Proof will go here
  sorry

end c_work_rate_l630_63043


namespace geometric_sequence_a5_l630_63087

-- Definitions based on the conditions:
variable {a : ℕ → ℝ} -- the sequence {a_n}
variable (q : ℝ) -- the common ratio of the geometric sequence

-- The sequence is geometric and terms are given:
axiom seq_geom (n m : ℕ) : a n = a 0 * q ^ n
axiom a_3_is_neg4 : a 3 = -4
axiom a_7_is_neg16 : a 7 = -16

-- The specific theorem we are proving:
theorem geometric_sequence_a5 :
  a 5 = -8 :=
by {
  sorry
}

end geometric_sequence_a5_l630_63087


namespace garden_perimeter_equals_104_l630_63034

theorem garden_perimeter_equals_104 :
  let playground_length := 16
  let playground_width := 12
  let playground_area := playground_length * playground_width
  let garden_width := 4
  let garden_length := playground_area / garden_width
  let garden_perimeter := 2 * garden_length + 2 * garden_width
  playground_area = 192 ∧ garden_perimeter = 104 :=
by {
  -- Declarations
  let playground_length := 16
  let playground_width := 12
  let playground_area := playground_length * playground_width
  let garden_width := 4
  let garden_length := playground_area / garden_width
  let garden_perimeter := 2 * garden_length + 2 * garden_width

  -- Assertions
  have area_playground : playground_area = 192 := by sorry
  have perimeter_garden : garden_perimeter = 104 := by sorry

  -- Conclusion
  exact ⟨area_playground, perimeter_garden⟩
}

end garden_perimeter_equals_104_l630_63034


namespace point_not_in_plane_l630_63016

def is_in_plane (p0 : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) (p : ℝ × ℝ × ℝ) : Prop :=
  let (x0, y0, z0) := p0
  let (nx, ny, nz) := n
  let (x, y, z) := p
  (nx * (x - x0) + ny * (y - y0) + nz * (z - z0)) = 0

theorem point_not_in_plane :
  ¬ is_in_plane (1, 2, 3) (1, 1, 1) (-2, 5, 4) :=
by
  sorry

end point_not_in_plane_l630_63016


namespace compute_100p_plus_q_l630_63012

theorem compute_100p_plus_q
  (p q : ℝ)
  (h1 : ∀ x : ℝ, (x + p) * (x + q) * (x + 15) = 0 → 
                  x ≠ -4 → x ≠ -15 → x ≠ -p → x ≠ -q)
  (h2 : ∀ x : ℝ, (x + 2 * p) * (x + 4) * (x + 9) = 0 → 
                  x ≠ -q → x ≠ -15 → (x = -4 ∨ x = -9))
  : 100 * p + q = -191 := 
sorry

end compute_100p_plus_q_l630_63012


namespace range_of_angle_A_l630_63025

theorem range_of_angle_A (a b : ℝ) (A : ℝ) (h₁ : a = 2) (h₂ : b = 2 * Real.sqrt 2) 
  (h_triangle : 0 < A ∧ A ≤ Real.pi / 4) :
  (0 < A ∧ A ≤ Real.pi / 4) :=
by
  sorry

end range_of_angle_A_l630_63025


namespace expenditure_ratio_l630_63022

def ratio_of_incomes (I1 I2 : ℕ) : Prop := I1 / I2 = 5 / 4
def savings (I E : ℕ) : ℕ := I - E
def ratio_of_expenditures (E1 E2 : ℕ) : Prop := E1 / E2 = 3 / 2

theorem expenditure_ratio (I1 I2 E1 E2 : ℕ) 
  (I1_income : I1 = 5500)
  (income_ratio : ratio_of_incomes I1 I2)
  (savings_equal : savings I1 E1 = 2200 ∧ savings I2 E2 = 2200)
  : ratio_of_expenditures E1 E2 :=
by 
  sorry

end expenditure_ratio_l630_63022


namespace probability_sum_7_is_1_over_3_l630_63065

def odd_die : Set ℕ := {1, 3, 5}
def even_die : Set ℕ := {2, 4, 6}

noncomputable def total_outcomes : ℕ := 6 * 6

noncomputable def favorable_outcomes : ℕ := 4 + 4 + 4

noncomputable def probability_sum_7 : ℚ := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

theorem probability_sum_7_is_1_over_3 :
  probability_sum_7 = 1 / 3 :=
by
  sorry

end probability_sum_7_is_1_over_3_l630_63065


namespace find_k_l630_63015

theorem find_k (a : ℕ → ℕ) (h₀ : a 1 = 2) (h₁ : ∀ m n, a (m + n) = a m * a n) (hk : a (k + 1) = 1024) : k = 9 := 
sorry

end find_k_l630_63015


namespace positive_difference_of_squares_l630_63024

theorem positive_difference_of_squares (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 18) : a^2 - b^2 = 1080 :=
by
  sorry

end positive_difference_of_squares_l630_63024


namespace sum_first_5_terms_l630_63070

variable {a : ℕ → ℝ}
variable (h : 2 * a 2 = a 1 + 3)

theorem sum_first_5_terms (a : ℕ → ℝ) (h : 2 * a 2 = a 1 + 3) : 
  (a 1 + a 2 + a 3 + a 4 + a 5) = 15 :=
sorry

end sum_first_5_terms_l630_63070


namespace line_intersects_y_axis_at_point_l630_63095

def line_intersects_y_axis (x1 y1 x2 y2 : ℚ) : Prop :=
  ∃ c : ℚ, ∀ x : ℚ, y1 + (y2 - y1) / (x2 - x1) * (x - x1) = (y2 - y1) / (x2 - x1) * x + c

theorem line_intersects_y_axis_at_point :
  line_intersects_y_axis 3 21 (-9) (-6) :=
  sorry

end line_intersects_y_axis_at_point_l630_63095


namespace exists_three_digit_number_cube_ends_in_777_l630_63050

theorem exists_three_digit_number_cube_ends_in_777 :
  ∃ x : ℤ, 100 ≤ x ∧ x < 1000 ∧ x^3 % 1000 = 777 := 
sorry

end exists_three_digit_number_cube_ends_in_777_l630_63050


namespace sum_of_numbers_eq_answer_l630_63030

open Real

noncomputable def sum_of_numbers (x y : ℝ) : ℝ := x + y

theorem sum_of_numbers_eq_answer (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 16) (h4 : (1 / x) = 3 * (1 / y)) :
  sum_of_numbers x y = 16 * Real.sqrt 3 / 3 := 
sorry

end sum_of_numbers_eq_answer_l630_63030


namespace x_plus_y_value_l630_63014

theorem x_plus_y_value (x y : ℕ) (h1 : 2^x = 8^(y + 1)) (h2 : 9^y = 3^(x - 9)) : x + y = 27 :=
by
  sorry

end x_plus_y_value_l630_63014


namespace sufficient_condition_l630_63005

variable {α : Type*} (A B : Set α)

theorem sufficient_condition (h : A ⊆ B) (x : α) : x ∈ A → x ∈ B :=
by
  sorry

end sufficient_condition_l630_63005


namespace polygon_exterior_angle_l630_63059

theorem polygon_exterior_angle (n : ℕ) (h : 36 = 360 / n) : n = 10 :=
sorry

end polygon_exterior_angle_l630_63059


namespace complement_of_angle_l630_63031

theorem complement_of_angle (x : ℝ) (h : 90 - x = 3 * x + 10) : x = 20 := by
  sorry

end complement_of_angle_l630_63031


namespace perpendicular_line_eq_l630_63052

theorem perpendicular_line_eq :
  ∃ (A B C : ℝ), (A * 0 + B * 4 + C = 0) ∧ (A = 3) ∧ (B = 1) ∧ (C = -4) ∧ (3 * 1 + 1 * -3 = 0) :=
sorry

end perpendicular_line_eq_l630_63052


namespace flower_bed_profit_l630_63047

theorem flower_bed_profit (x : ℤ) :
  (3 + x) * (10 - x) = 40 :=
sorry

end flower_bed_profit_l630_63047


namespace average_hamburgers_per_day_l630_63035

def total_hamburgers : ℕ := 63
def days_in_week : ℕ := 7
def average_per_day : ℕ := total_hamburgers / days_in_week

theorem average_hamburgers_per_day : average_per_day = 9 := by
  sorry

end average_hamburgers_per_day_l630_63035


namespace fraction_of_third_is_eighth_l630_63072

theorem fraction_of_third_is_eighth : (1 / 8) / (1 / 3) = 3 / 8 := by
  sorry

end fraction_of_third_is_eighth_l630_63072


namespace determine_set_B_l630_63078
open Set

/-- Given problem conditions and goal in Lean 4 -/
theorem determine_set_B (U A B : Set ℕ) (hU : U = { x | x < 10 } )
  (hA_inter_compl_B : A ∩ (U \ B) = {1, 3, 5, 7, 9} ) :
  B = {2, 4, 6, 8} :=
by
  sorry

end determine_set_B_l630_63078


namespace roots_expression_value_l630_63026

theorem roots_expression_value (x1 x2 : ℝ) (h1 : x1 + x2 = 5) (h2 : x1 * x2 = 2) :
  2 * x1 - x1 * x2 + 2 * x2 = 8 :=
by
  sorry

end roots_expression_value_l630_63026


namespace valid_assignments_count_l630_63076

noncomputable def validAssignments : Nat := sorry

theorem valid_assignments_count : validAssignments = 4 := 
by {
  sorry
}

end valid_assignments_count_l630_63076


namespace find_fg3_l630_63036

def f (x : ℝ) : ℝ := x^2 + 1
def g (x : ℝ) : ℝ := x^2 - 4*x + 4

theorem find_fg3 : f (g 3) = 2 := by
  sorry

end find_fg3_l630_63036


namespace molar_mass_of_compound_l630_63069

variable (total_weight : ℝ) (num_moles : ℝ)

theorem molar_mass_of_compound (h1 : total_weight = 2352) (h2 : num_moles = 8) :
    total_weight / num_moles = 294 :=
by
  rw [h1, h2]
  norm_num

end molar_mass_of_compound_l630_63069


namespace num_ways_placing_2015_bishops_l630_63010

-- Define the concept of placing bishops on a 2 x n chessboard without mutual attacks
def max_bishops (n : ℕ) : ℕ := n

-- Define the calculation of the number of ways to place these bishops
def num_ways_to_place_bishops (n : ℕ) : ℕ := 2 ^ n

-- The proof statement for our specific problem
theorem num_ways_placing_2015_bishops :
  num_ways_to_place_bishops 2015 = 2 ^ 2015 :=
by
  sorry

end num_ways_placing_2015_bishops_l630_63010


namespace total_photos_newspaper_l630_63063

theorem total_photos_newspaper (pages1 pages2 photos_per_page1 photos_per_page2 : ℕ)
  (h1 : pages1 = 12) (h2 : photos_per_page1 = 2)
  (h3 : pages2 = 9) (h4 : photos_per_page2 = 3) :
  (pages1 * photos_per_page1) + (pages2 * photos_per_page2) = 51 :=
by
  sorry

end total_photos_newspaper_l630_63063


namespace cards_in_center_pile_l630_63056

/-- Represents the number of cards in each pile initially. -/
def initial_cards (x : ℕ) : Prop := x ≥ 2

/-- Represents the state of the piles after step 2. -/
def step2 (x : ℕ) (left center right : ℕ) : Prop :=
  left = x - 2 ∧ center = x + 2 ∧ right = x

/-- Represents the state of the piles after step 3. -/
def step3 (x : ℕ) (left center right : ℕ) : Prop :=
  left = x - 2 ∧ center = x + 3 ∧ right = x - 1

/-- Represents the state of the piles after step 4. -/
def step4 (x : ℕ) (left center : ℕ) : Prop :=
  left = 2 * x - 4 ∧ center = 5

/-- Prove that after performing all steps, the number of cards in the center pile is 5. -/
theorem cards_in_center_pile (x : ℕ) :
  initial_cards x →
  (∃ l₁ c₁ r₁, step2 x l₁ c₁ r₁) →
  (∃ l₂ c₂ r₂, step3 x l₂ c₂ r₂) →
  (∃ l₃ c₃, step4 x l₃ c₃) →
  ∃ (center_final : ℕ), center_final = 5 :=
by
  sorry

end cards_in_center_pile_l630_63056


namespace deductive_reasoning_not_always_correct_l630_63060

theorem deductive_reasoning_not_always_correct (P: Prop) (Q: Prop) 
    (h1: (P → Q) → (P → Q)) :
    (¬ (∀ P Q : Prop, (P → Q) → Q → Q)) :=
sorry

end deductive_reasoning_not_always_correct_l630_63060


namespace train_crosses_platform_in_39_seconds_l630_63020

-- Definitions based on the problem's conditions
def train_length : ℕ := 450
def time_to_cross_signal : ℕ := 18
def platform_length : ℕ := 525

-- The speed of the train
def train_speed : ℕ := train_length / time_to_cross_signal

-- The total distance the train has to cover
def total_distance : ℕ := train_length + platform_length

-- The time it takes for the train to cross the platform
def time_to_cross_platform : ℕ := total_distance / train_speed

-- The theorem we need to prove
theorem train_crosses_platform_in_39_seconds :
  time_to_cross_platform = 39 := by
  sorry

end train_crosses_platform_in_39_seconds_l630_63020


namespace convert_to_scientific_notation_l630_63046

theorem convert_to_scientific_notation (N : ℕ) (h : 2184300000 = 2184.3 * 10^6) : 
    (2184300000 : ℝ) = 2.1843 * 10^7 :=
by 
  sorry

end convert_to_scientific_notation_l630_63046


namespace points_for_correct_answer_l630_63084

theorem points_for_correct_answer
  (x y a b : ℕ)
  (hx : x - y = 7)
  (hsum : a + b = 43)
  (hw_score : a * x - b * (20 - x) = 328)
  (hz_score : a * y - b * (20 - y) = 27) :
  a = 25 := 
sorry

end points_for_correct_answer_l630_63084


namespace product_of_repeating_decimal_l630_63080

   -- Definitions
   def repeating_decimal : ℚ := 456 / 999  -- 0.\overline{456}

   -- Problem Statement
   theorem product_of_repeating_decimal (t : ℚ) (h : t = repeating_decimal) : (t * 7) = 1064 / 333 :=
   by
     sorry
   
end product_of_repeating_decimal_l630_63080


namespace sin_alpha_beta_l630_63068

theorem sin_alpha_beta (a b c α β : Real) (h₁ : a * Real.cos α + b * Real.sin α + c = 0)
  (h₂ : a * Real.cos β + b * Real.sin β + c = 0) (h₃ : 0 < α) (h₄ : α < β) (h₅ : β < π) :
  Real.sin (α + β) = (2 * a * b) / (a^2 + b^2) :=
by 
  sorry

end sin_alpha_beta_l630_63068


namespace units_digit_product_l630_63071

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_product (a b c : ℕ) :
  units_digit a = 7 → units_digit b = 3 → units_digit c = 9 →
  units_digit ((a * b) * c) = 9 :=
by
  intros h1 h2 h3
  sorry

end units_digit_product_l630_63071


namespace percentage_of_360_is_120_l630_63045

theorem percentage_of_360_is_120 (part whole : ℝ) (h1 : part = 120) (h2 : whole = 360) : 
  ((part / whole) * 100 = 33.33) :=
by
  sorry

end percentage_of_360_is_120_l630_63045


namespace identical_lines_unique_pair_l630_63028

theorem identical_lines_unique_pair :
  ∃! (a b : ℚ), 2 * (0 : ℚ) + a * (0 : ℚ) + 10 = 0 ∧ b * (0 : ℚ) - 3 * (0 : ℚ) - 15 = 0 ∧ 
  (-2 / a = b / 3) ∧ (-10 / a = 5) :=
by {
  -- Given equations in slope-intercept form:
  -- y = -2 / a * x - 10 / a
  -- y = b / 3 * x + 5
  -- Slope and intercept comparison leads to equations:
  -- -2 / a = b / 3
  -- -10 / a = 5
  sorry
}

end identical_lines_unique_pair_l630_63028


namespace tourism_revenue_scientific_notation_l630_63098

theorem tourism_revenue_scientific_notation:
  (12.41 * 10^9) = (1.241 * 10^9) := 
sorry

end tourism_revenue_scientific_notation_l630_63098


namespace find_largest_square_area_l630_63057

def area_of_largest_square (XY YZ XZ : ℝ) (sum_of_areas : ℝ) (right_angle : Prop) : Prop :=
  sum_of_areas = XY^2 + YZ^2 + XZ^2 + 4 * YZ^2 ∧  -- sum of areas condition
  right_angle ∧                                    -- right angle condition
  XZ^2 = XY^2 + YZ^2 ∧                             -- Pythagorean theorem
  sum_of_areas = 650 ∧                             -- total area condition
  XY = YZ                                          -- assumption for simplified solving.

theorem find_largest_square_area (XY YZ XZ : ℝ) (sum_of_areas : ℝ):
  area_of_largest_square XY YZ XZ sum_of_areas (90 = 90) → 2 * XY^2 + 5 * YZ^2 = 650 → XZ^2 = 216.67 :=
sorry

end find_largest_square_area_l630_63057


namespace cube_and_reciprocal_l630_63049

theorem cube_and_reciprocal (m : ℝ) (hm : m + 1/m = 10) : m^3 + 1/m^3 = 970 := 
by
  sorry

end cube_and_reciprocal_l630_63049


namespace product_of_solutions_abs_eq_l630_63092

theorem product_of_solutions_abs_eq (x : ℝ) :
  (∃ x1 x2 : ℝ, |6 * x1 + 2| + 5 = 47 ∧ |6 * x2 + 2| + 5 = 47 ∧ x ≠ x1 ∧ x ≠ x2 ∧ x1 * x2 = -440 / 9) :=
by
  sorry

end product_of_solutions_abs_eq_l630_63092


namespace not_perfect_square_l630_63058

theorem not_perfect_square (a : ℤ) : ¬ (∃ x : ℤ, a^2 + 4 = x^2) := 
sorry

end not_perfect_square_l630_63058


namespace profit_bicycle_l630_63051

theorem profit_bicycle (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 650) 
  (h2 : x + 2 * y = 350) : 
  x = 150 ∧ y = 100 :=
by 
  sorry

end profit_bicycle_l630_63051


namespace sequence_value_l630_63038

theorem sequence_value (a : ℕ → ℤ) (h1 : ∀ p q : ℕ, 0 < p → 0 < q → a (p + q) = a p + a q)
                       (h2 : a 2 = -6) : a 10 = -30 :=
by
  sorry

end sequence_value_l630_63038


namespace triangle_angles_sum_l630_63017

theorem triangle_angles_sum (x : ℝ) (h : 40 + 3 * x + (x + 10) = 180) : x = 32.5 := by
  sorry

end triangle_angles_sum_l630_63017


namespace problem1_l630_63013

theorem problem1 : 
  ∀ a b : ℤ, a = 1 → b = -3 → (a - b)^2 - 2 * a * (a + 3 * b) + (a + 2 * b) * (a - 2 * b) = -3 :=
by
  intros a b h1 h2
  rw [h1, h2]
  sorry

end problem1_l630_63013


namespace parallel_lines_l630_63033

open Real -- Open the real number namespace

/-- Definition of line l1 --/
def line_l1 (a : ℝ) (x y : ℝ) := a * x + 2 * y - 1 = 0

/-- Definition of line l2 --/
def line_l2 (a : ℝ) (x y : ℝ) := x + (a + 1) * y + 4 = 0

/-- The proof statement --/
theorem parallel_lines (a : ℝ) : (a = 1) → (line_l1 a x y) → (line_l2 a x y) := 
sorry

end parallel_lines_l630_63033


namespace continuous_sum_m_l630_63053

noncomputable def g : ℝ → ℝ → ℝ
| x, m => if x < m then x^2 + 4 else 3 * x + 6

theorem continuous_sum_m :
  ∀ m1 m2 : ℝ, (∀ m : ℝ, (g m m1 = g m m2) → g m (m1 + m2) = g m m1 + g m m2) →
  m1 + m2 = 3 :=
sorry

end continuous_sum_m_l630_63053


namespace car_rental_cost_l630_63075

variable (x : ℝ)

theorem car_rental_cost (h : 65 + 0.40 * 325 = x * 325) : x = 0.60 :=
by 
  sorry

end car_rental_cost_l630_63075


namespace triangle_area_546_l630_63082

theorem triangle_area_546 :
  ∀ (a b c : ℕ), a = 13 ∧ b = 84 ∧ c = 85 ∧ a^2 + b^2 = c^2 →
  (1 / 2 : ℝ) * (a * b) = 546 :=
by
  intro a b c
  intro h
  sorry

end triangle_area_546_l630_63082


namespace sum_terms_a1_a17_l630_63044

theorem sum_terms_a1_a17 (S : ℕ → ℤ) (a : ℕ → ℤ)
  (hS : ∀ n, S n = n^2 - 2 * n - 1)
  (ha : ∀ n, a n = if n = 1 then S 1 else S n - S (n - 1)) :
  a 1 + a 17 = 29 :=
sorry

end sum_terms_a1_a17_l630_63044


namespace prob_CD_l630_63091

variable (P : String → ℚ)
variable (x : ℚ)

axiom probA : P "A" = 1 / 3
axiom probB : P "B" = 1 / 4
axiom probC : P "C" = 2 * x
axiom probD : P "D" = x
axiom sumProb : P "A" + P "B" + P "C" + P "D" = 1

theorem prob_CD :
  P "D" = 5 / 36 ∧ P "C" = 5 / 18 := by
  sorry

end prob_CD_l630_63091


namespace ellipsoid_center_and_axes_sum_l630_63066

theorem ellipsoid_center_and_axes_sum :
  let x₀ := -2
  let y₀ := 3
  let z₀ := 1
  let A := 6
  let B := 4
  let C := 2
  x₀ + y₀ + z₀ + A + B + C = 14 := 
by
  sorry

end ellipsoid_center_and_axes_sum_l630_63066


namespace hiker_speed_correct_l630_63042

variable (hikerSpeed : ℝ)
variable (cyclistSpeed : ℝ := 15)
variable (cyclistTravelTime : ℝ := 5 / 60)  -- Converted 5 minutes to hours
variable (hikerCatchUpTime : ℝ := 13.75 / 60)  -- Converted 13.75 minutes to hours
variable (cyclistDistance : ℝ := cyclistSpeed * cyclistTravelTime)

theorem hiker_speed_correct :
  (hikerSpeed * hikerCatchUpTime = cyclistDistance) →
  hikerSpeed = 60 / 11 :=
by
  intro hiker_eq_cyclist_distance
  sorry

end hiker_speed_correct_l630_63042


namespace students_in_line_l630_63029

theorem students_in_line (T N : ℕ) (hT : T = 1) (h_btw : N = T + 4) (h_behind: ∃ k, k = 8) : T + (N - T) + 1 + 8 = 13 :=
by
  sorry

end students_in_line_l630_63029


namespace natalie_blueberry_bushes_l630_63037

-- Definitions of the conditions
def bushes_yield_containers (bushes containers : ℕ) : Prop :=
  containers = bushes * 7

def containers_exchange_zucchinis (containers zucchinis : ℕ) : Prop :=
  zucchinis = containers * 3 / 7

-- Theorem statement
theorem natalie_blueberry_bushes (zucchinis_needed : ℕ) (zucchinis_per_trade containers_per_trade bushes_per_container : ℕ) 
  (h1 : zucchinis_per_trade = 3) (h2 : containers_per_trade = 7) (h3 : bushes_per_container = 7) 
  (h4 : zucchinis_needed = 63) : 
  ∃ bushes_needed : ℕ, bushes_needed = 21 := 
by
  sorry

end natalie_blueberry_bushes_l630_63037


namespace integral_solution_l630_63006

noncomputable def integral_expression : Real → Real :=
  fun x => (1 + (x ^ (3 / 4))) ^ (4 / 5) / (x ^ (47 / 20))

theorem integral_solution :
  ∫ (x : Real), integral_expression x = - (20 / 27) * ((1 + (x ^ (3 / 4)) / (x ^ (3 / 4))) ^ (9 / 5)) + C := 
by 
  sorry

end integral_solution_l630_63006


namespace tetrahedron_edge_length_correct_l630_63001

noncomputable def radius := Real.sqrt 2
noncomputable def center_to_center_distance := 2 * radius
noncomputable def tetrahedron_edge_length := center_to_center_distance

theorem tetrahedron_edge_length_correct :
  tetrahedron_edge_length = 2 * Real.sqrt 2 := by
  sorry

end tetrahedron_edge_length_correct_l630_63001


namespace john_weekly_earnings_after_raise_l630_63055

theorem john_weekly_earnings_after_raise (original_earnings : ℝ) (raise_percentage : ℝ) (raise_amount new_earnings : ℝ) 
  (h1 : original_earnings = 50) (h2 : raise_percentage = 60) (h3 : raise_amount = (raise_percentage / 100) * original_earnings) 
  (h4 : new_earnings = original_earnings + raise_amount) : 
  new_earnings = 80 := 
by sorry

end john_weekly_earnings_after_raise_l630_63055


namespace max_books_borrowed_l630_63021

theorem max_books_borrowed (total_students : ℕ) (no_books : ℕ) (one_book : ℕ)
  (two_books : ℕ) (at_least_three_books : ℕ) (avg_books_per_student : ℕ) :
  total_students = 35 →
  no_books = 2 →
  one_book = 12 →
  two_books = 10 →
  avg_books_per_student = 2 →
  total_students - (no_books + one_book + two_books) = at_least_three_books →
  ∃ max_books_borrowed_by_individual, max_books_borrowed_by_individual = 8 :=
by
  intros h_total_students h_no_books h_one_book h_two_books h_avg_books_per_student h_remaining_students
  -- Skipping the proof steps
  sorry

end max_books_borrowed_l630_63021


namespace ice_cream_cone_cost_l630_63094

theorem ice_cream_cone_cost (total_sales : ℝ) (free_cones_given : ℕ) (cost_per_cone : ℝ) 
  (customers_per_group : ℕ) (cones_sold_per_group : ℕ) 
  (h1 : total_sales = 100)
  (h2: free_cones_given = 10)
  (h3: customers_per_group = 6)
  (h4: cones_sold_per_group = 5) :
  cost_per_cone = 2 := sorry

end ice_cream_cone_cost_l630_63094


namespace remainder_of_prime_division_l630_63090

theorem remainder_of_prime_division
  (p : ℕ) (hp : Nat.Prime p)
  (r : ℕ) (hr : r = p % 210) 
  (hcomp : ¬ Nat.Prime r)
  (hsum : ∃ a b : ℕ, r = a^2 + b^2) : 
  r = 169 := 
sorry

end remainder_of_prime_division_l630_63090


namespace soda_cost_l630_63083

theorem soda_cost (total_cost sandwich_price : ℝ) (num_sandwiches num_sodas : ℕ) (total : total_cost = 8.38)
  (sandwich_cost : sandwich_price = 2.45) (total_sandwiches : num_sandwiches = 2) (total_sodas : num_sodas = 4) :
  ((total_cost - (num_sandwiches * sandwich_price)) / num_sodas) = 0.87 :=
by
  sorry

end soda_cost_l630_63083


namespace scientific_notation_correct_l630_63048

noncomputable def scientific_notation (x : ℕ) : Prop :=
  x = 3010000000 → 3.01 * (10 ^ 9) = 3.01 * (10 ^ 9)

theorem scientific_notation_correct : 
  scientific_notation 3010000000 :=
by
  intros h
  sorry

end scientific_notation_correct_l630_63048


namespace cistern_width_l630_63074

theorem cistern_width (l d A : ℝ) (h_l: l = 5) (h_d: d = 1.25) (h_A: A = 42.5) :
  ∃ w : ℝ, 5 * w + 2 * (1.25 * 5) + 2 * (1.25 * w) = 42.5 ∧ w = 4 :=
by
  use 4
  sorry

end cistern_width_l630_63074


namespace temperature_difference_l630_63027

theorem temperature_difference : 
  let beijing_temp := -6
  let changtai_temp := 15
  changtai_temp - beijing_temp = 21 := 
by
  -- Let the given temperatures
  let beijing_temp := -6
  let changtai_temp := 15
  -- Perform the subtraction and define the expected equality
  show changtai_temp - beijing_temp = 21
  -- Preliminary proof placeholder
  sorry

end temperature_difference_l630_63027


namespace proposition_3_proposition_4_l630_63085

variable {Line Plane : Type} -- Introduce the types for lines and planes
variable (m n : Line) (α β : Plane) -- Introduce specific lines and planes

-- Define parallel and perpendicular relations
variables {parallel : Line → Plane → Prop} {perpendicular : Line → Plane → Prop}
variables {parallel_line : Line → Line → Prop} {perpendicular_line : Line → Line → Prop}
variables {parallel_plane : Plane → Plane → Prop} {perpendicular_plane : Plane → Plane → Prop}

-- Define subset: a line n is in a plane α
variable {subset : Line → Plane → Prop}

-- Hypotheses for propositions 3 and 4
axiom prop3_hyp1 : perpendicular m α
axiom prop3_hyp2 : parallel_line m n
axiom prop3_hyp3 : parallel_plane α β

axiom prop4_hyp1 : perpendicular_line m n
axiom prop4_hyp2 : perpendicular m α
axiom prop4_hyp3 : perpendicular n β

theorem proposition_3 (h1 : perpendicular m α) (h2 : parallel_line m n) (h3 : parallel_plane α β) : perpendicular n β := sorry

theorem proposition_4 (h1 : perpendicular_line m n) (h2 : perpendicular m α) (h3 : perpendicular n β) : perpendicular_plane α β := sorry

end proposition_3_proposition_4_l630_63085


namespace find_other_number_l630_63062

theorem find_other_number (LCM : ℕ) (HCF : ℕ) (n1 : ℕ) (n2 : ℕ) 
  (h_lcm : LCM = 2310) (h_hcf : HCF = 26) (h_n1 : n1 = 210) :
  n2 = 286 :=
by
  sorry

end find_other_number_l630_63062


namespace add_one_gt_add_one_l630_63086

theorem add_one_gt_add_one (a b c : ℝ) (h : a > b) : (a + c) > (b + c) :=
sorry

end add_one_gt_add_one_l630_63086


namespace relationship_a_b_c_l630_63041

noncomputable def a : ℝ := Real.sin (Real.pi / 16)
noncomputable def b : ℝ := 0.25
noncomputable def c : ℝ := 2 * Real.log 2 - Real.log 3

theorem relationship_a_b_c : a < b ∧ b < c :=
by
  sorry

end relationship_a_b_c_l630_63041


namespace cos_arith_prog_impossible_l630_63089

theorem cos_arith_prog_impossible
  (x y z : ℝ)
  (sin_arith_prog : 2 * Real.sin y = Real.sin x + Real.sin z) :
  ¬ (2 * Real.cos y = Real.cos x + Real.cos z) :=
by
  sorry

end cos_arith_prog_impossible_l630_63089


namespace has_only_one_zero_point_l630_63077

noncomputable def f (x a : ℝ) := (x - 1) * Real.exp x + (a / 2) * x^2

theorem has_only_one_zero_point (a : ℝ) (h : -Real.exp 1 ≤ a ∧ a ≤ 0) :
  ∃! x : ℝ, f x a = 0 :=
sorry

end has_only_one_zero_point_l630_63077


namespace john_bought_two_dozens_l630_63000

theorem john_bought_two_dozens (x : ℕ) (h₁ : 21 + 3 = x * 12) : x = 2 :=
by {
    -- Placeholder for skipping the proof since it's not required.
    sorry
}

end john_bought_two_dozens_l630_63000


namespace train_pass_bridge_in_50_seconds_l630_63007

noncomputable def time_to_pass_bridge (length_train length_bridge : ℕ) (speed_kmh : ℕ) : ℕ :=
  let total_distance := length_train + length_bridge
  let speed_ms := (speed_kmh * 1000) / 3600
  total_distance / speed_ms

theorem train_pass_bridge_in_50_seconds :
  time_to_pass_bridge 485 140 45 = 50 :=
by
  sorry

end train_pass_bridge_in_50_seconds_l630_63007


namespace Mairead_triathlon_l630_63099

noncomputable def convert_km_to_miles (km: Float) : Float :=
  0.621371 * km

noncomputable def convert_yards_to_miles (yd: Float) : Float :=
  0.000568182 * yd

noncomputable def convert_feet_to_miles (ft: Float) : Float :=
  0.000189394 * ft

noncomputable def total_distance_in_miles := 
  let run_distance_km := 40.0
  let run_distance_miles := convert_km_to_miles run_distance_km
  let walk_distance_miles := 3.0/5.0 * run_distance_miles
  let jog_distance_yd := 5.0 * (walk_distance_miles * 1760.0)
  let jog_distance_miles := convert_yards_to_miles jog_distance_yd
  let bike_distance_ft := 3.0 * (jog_distance_miles * 5280.0)
  let bike_distance_miles := convert_feet_to_miles bike_distance_ft
  let swim_distance_miles := 2.5
  run_distance_miles + walk_distance_miles + jog_distance_miles + bike_distance_miles + swim_distance_miles

theorem Mairead_triathlon:
  total_distance_in_miles = 340.449562 ∧
  (convert_km_to_miles 40.0) / 10.0 = 2.485484 ∧
  (3.0/5.0 * (convert_km_to_miles 40.0)) / 10.0 = 1.4912904 ∧
  (convert_yards_to_miles (5.0 * (3.0/5.0 * (convert_km_to_miles 40.0) * 1760.0))) / 10.0 = 7.45454544 ∧
  (convert_feet_to_miles (3.0 * (convert_yards_to_miles (5.0 * (3.0/5.0 * (convert_km_to_miles 40.0) * 1760.0)) * 5280.0))) / 10.0 = 22.36363636 ∧
  2.5 / 10.0 = 0.25 := sorry

end Mairead_triathlon_l630_63099


namespace isolate_urea_decomposing_bacteria_valid_option_l630_63004

variable (KH2PO4 Na2HPO4 MgSO4_7H2O urea glucose agar water : Type)
variable (urea_decomposing_bacteria : Type)
variable (CarbonSource : Type → Prop)
variable (NitrogenSource : Type → Prop)
variable (InorganicSalt : Type → Prop)
variable (bacteria_can_synthesize_urease : urea_decomposing_bacteria → Prop)

axiom KH2PO4_is_inorganic_salt : InorganicSalt KH2PO4
axiom Na2HPO4_is_inorganic_salt : InorganicSalt Na2HPO4
axiom MgSO4_7H2O_is_inorganic_salt : InorganicSalt MgSO4_7H2O
axiom urea_is_nitrogen_source : NitrogenSource urea

theorem isolate_urea_decomposing_bacteria_valid_option :
  (InorganicSalt KH2PO4) ∧
  (InorganicSalt Na2HPO4) ∧
  (InorganicSalt MgSO4_7H2O) ∧
  (NitrogenSource urea) ∧
  (CarbonSource glucose) → (∃ bacteria : urea_decomposing_bacteria, bacteria_can_synthesize_urease bacteria) := sorry

end isolate_urea_decomposing_bacteria_valid_option_l630_63004


namespace problem_solve_l630_63039

theorem problem_solve (x y : ℝ) (h1 : x ≠ y) (h2 : x / y + (x + 6 * y) / (y + 6 * x) = 3) : 
    x / y = (8 + Real.sqrt 46) / 6 := 
  sorry

end problem_solve_l630_63039


namespace speed_of_stream_l630_63073

theorem speed_of_stream (v : ℝ) (h1 : 22 > 0) (h2 : 8 > 0) (h3 : 216 = (22 + v) * 8) : v = 5 := 
by 
  sorry

end speed_of_stream_l630_63073


namespace infinite_solutions_or_no_solutions_l630_63097

theorem infinite_solutions_or_no_solutions (a b : ℚ) :
  (∃ (x y : ℚ), a * x^2 + b * y^2 = 1) →
  (∀ (k : ℚ), a * k^2 + b ≠ 0 → ∃ (x_k y_k : ℚ), a * x_k^2 + b * y_k^2 = 1) :=
by
  intro h_sol h_k
  sorry

end infinite_solutions_or_no_solutions_l630_63097


namespace student_walking_time_l630_63054

-- Define the conditions
def total_time_walking_and_bus : ℕ := 90  -- Total time walking to school and taking the bus back home
def total_time_bus_both_ways : ℕ := 30 -- Total time taking the bus both ways

-- Calculate the time taken for walking both ways
def time_bus_one_way : ℕ := total_time_bus_both_ways / 2
def time_walking_one_way : ℕ := total_time_walking_and_bus - time_bus_one_way
def total_time_walking_both_ways : ℕ := 2 * time_walking_one_way

-- State the theorem to be proved
theorem student_walking_time :
  total_time_walking_both_ways = 150 := by
  sorry

end student_walking_time_l630_63054


namespace car_average_speed_l630_63079

-- Define the given conditions
def total_time_hours : ℕ := 5
def total_distance_miles : ℕ := 200

-- Define the average speed calculation
def average_speed (distance time : ℕ) : ℕ :=
  distance / time

-- State the theorem to be proved
theorem car_average_speed :
  average_speed total_distance_miles total_time_hours = 40 :=
by
  sorry

end car_average_speed_l630_63079


namespace compute_expression_l630_63002

theorem compute_expression : 1004^2 - 996^2 - 1000^2 + 1000^2 = 16000 := 
by sorry

end compute_expression_l630_63002


namespace sum_of_cubes_1998_l630_63023

theorem sum_of_cubes_1998 : 1998 = 334^3 + 332^3 + (-333)^3 + (-333)^3 := by
  sorry

end sum_of_cubes_1998_l630_63023


namespace triangle_b_value_triangle_area_value_l630_63009

noncomputable def triangle_b (a : ℝ) (cosA : ℝ) : ℝ :=
  let sinA := Real.sqrt (1 - cosA^2)
  let sinB := cosA
  (a * sinB) / sinA

noncomputable def triangle_area (a b c : ℝ) (sinC : ℝ) : ℝ :=
  0.5 * a * b * sinC

-- Given conditions
variable (A B : ℝ) (a : ℝ := 3) (cosA : ℝ := Real.sqrt 6 / 3) (B := A + Real.pi / 2)

-- The assertions to prove
theorem triangle_b_value :
  triangle_b a cosA = 3 * Real.sqrt 2 :=
sorry

theorem triangle_area_value :
  triangle_area 3 (3 * Real.sqrt 2) 1 (1 / 3) = (3 * Real.sqrt 2) / 2 :=
sorry

end triangle_b_value_triangle_area_value_l630_63009


namespace total_pages_proof_l630_63064

/-
Conditions:
1. Johnny's essay has 150 words.
2. Madeline's essay is double the length of Johnny's essay.
3. Timothy's essay has 30 more words than Madeline's essay.
4. One page contains 260 words.

Question:
Prove that the total number of pages do Johnny, Madeline, and Timothy's essays fill is 5.
-/

def johnny_words : ℕ := 150
def words_per_page : ℕ := 260

def madeline_words : ℕ := 2 * johnny_words
def timothy_words : ℕ := madeline_words + 30

def pages (words : ℕ) : ℕ := (words + words_per_page - 1) / words_per_page  -- division rounding up

def johnny_pages : ℕ := pages johnny_words
def madeline_pages : ℕ := pages madeline_words
def timothy_pages : ℕ := pages timothy_words

def total_pages : ℕ := johnny_pages + madeline_pages + timothy_pages

theorem total_pages_proof : total_pages = 5 :=
by sorry

end total_pages_proof_l630_63064


namespace smallest_integer_y_l630_63081

theorem smallest_integer_y (y : ℤ) (h: y < 3 * y - 15) : y ≥ 8 :=
by sorry

end smallest_integer_y_l630_63081


namespace original_price_of_sarees_l630_63093
open Real

theorem original_price_of_sarees (P : ℝ) (h : 0.70 * 0.80 * P = 224) : P = 400 :=
sorry

end original_price_of_sarees_l630_63093


namespace aquarium_pufferfish_problem_l630_63061

/-- Define the problem constants and equations -/
theorem aquarium_pufferfish_problem :
  ∃ (P S : ℕ), S = 5 * P ∧ S + P = 90 ∧ P = 15 :=
by
  sorry

end aquarium_pufferfish_problem_l630_63061


namespace probability_C_l630_63032

-- Variables representing the probabilities of each region
variables (P_A P_B P_C P_D P_E : ℚ)

-- Given conditions
def conditions := P_A = 3/10 ∧ P_B = 1/4 ∧ P_D = 1/5 ∧ P_E = 1/10 ∧ P_A + P_B + P_C + P_D + P_E = 1

-- The statement to prove
theorem probability_C (h : conditions P_A P_B P_C P_D P_E) : P_C = 3/20 := 
by
  sorry

end probability_C_l630_63032


namespace total_number_of_coins_l630_63088

-- Definitions and conditions
def num_coins_25c := 17
def num_coins_10c := 17

-- Statement to prove
theorem total_number_of_coins : num_coins_25c + num_coins_10c = 34 := by
  sorry

end total_number_of_coins_l630_63088


namespace solve_problem_l630_63067

-- Define the polynomial g(x) as given in the problem
def g (p q r s t : ℝ) (x : ℝ) : ℝ := p * x^4 + q * x^3 + r * x^2 + s * x + t

-- Define the condition given in the problem
def condition (p q r s t : ℝ) : Prop := g p q r s t (-2) = -4

-- State the theorem to be proved
theorem solve_problem (p q r s t : ℝ) (h : condition p q r s t) :
  16 * p - 8 * q + 4 * r - 2 * s + t = 4 :=
by
  -- Proof is omitted
  sorry

end solve_problem_l630_63067


namespace max_m_l630_63003

noncomputable def f (x a : ℝ) : ℝ := 2 ^ |x + a|

theorem max_m (a m : ℝ) (H1 : ∀ x, f (3 + x) a = f (3 - x) a) 
(H2 : ∀ x y, x ≤ y → y ≤ m → f x a ≥ f y a) : 
  m = 3 :=
by
  sorry

end max_m_l630_63003


namespace find_positive_integers_l630_63019

theorem find_positive_integers (n : ℕ) (h_pos : n > 0) : 
  (∃ d : ℕ, ∀ k : ℕ, 6^n + 1 = d * (10^k - 1) / 9 → d = 7) → 
  n = 1 ∨ n = 5 :=
sorry

end find_positive_integers_l630_63019


namespace simplify_fraction_l630_63040

theorem simplify_fraction :
  (175 / 1225) * 25 = 25 / 7 :=
by
  -- Code to indicate proof steps would go here.
  sorry

end simplify_fraction_l630_63040


namespace sum_of_pairwise_rel_prime_integers_l630_63018

def is_pairwise_rel_prime (a b c : ℕ) : Prop :=
  Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd a c = 1

theorem sum_of_pairwise_rel_prime_integers 
  (a b c : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) 
  (h_prod : a * b * c = 343000) (h_rel_prime : is_pairwise_rel_prime a b c) : 
  a + b + c = 476 := 
sorry

end sum_of_pairwise_rel_prime_integers_l630_63018


namespace sequence_problems_l630_63008
open Nat

-- Define the arithmetic sequence conditions
def arith_seq_condition_1 (a : ℕ → ℤ) : Prop :=
  a 2 + a 7 = -23

def arith_seq_condition_2 (a : ℕ → ℤ) : Prop :=
  a 3 + a 8 = -29

-- Define the geometric sequence condition
def geom_seq_condition (a b : ℕ → ℤ) (c : ℤ) : Prop :=
  ∀ n, a n + b n = c^(n - 1)

-- Define the arithmetic sequence formula
def arith_seq_formula (a : ℕ → ℤ) : Prop :=
  ∀ n, a n = -3 * n + 2

-- Define the sum of the first n terms of the sequence b_n
def sum_b_n (b : ℕ → ℤ) (S_n : ℕ → ℤ) (c : ℤ) : Prop :=
  (c = 1 → ∀ n, S_n n = (3 * n^2 + n) / 2) ∧
  (c ≠ 1 → ∀ n, S_n n = (n * (3 * n - 1)) / 2 + ((1 - c^n) / (1 - c)))

-- Define the main theorem
theorem sequence_problems (a b : ℕ → ℤ) (c : ℤ) (S_n : ℕ → ℤ) :
  arith_seq_condition_1 a →
  arith_seq_condition_2 a →
  geom_seq_condition a b c →
  arith_seq_formula a ∧ sum_b_n b S_n c :=
by
  -- Proofs for the conditions to the formula
  sorry

end sequence_problems_l630_63008
