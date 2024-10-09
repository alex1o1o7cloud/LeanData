import Mathlib

namespace common_ratio_of_sequence_l1237_123743

variable {a : ℕ → ℝ}
variable {d : ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) (n1 n2 n3 : ℕ) : Prop :=
  a n2 = a n1 * r ∧ a n3 = a n1 * r^2

theorem common_ratio_of_sequence {a : ℕ → ℝ} {d : ℝ}
  (h_arith : arithmetic_sequence a d)
  (h_geom : geometric_sequence a ((a 2)/(a 1)) 2 3 6) :
  ((a 3) / (a 2)) = 3 ∨ ((a 3) / (a 2)) = 1 :=
sorry

end common_ratio_of_sequence_l1237_123743


namespace find_original_selling_price_l1237_123745

variable (SP : ℝ)
variable (CP : ℝ := 10000)
variable (discounted_SP : ℝ := 0.9 * SP)
variable (profit : ℝ := 0.08 * CP)

theorem find_original_selling_price :
  discounted_SP = CP + profit → SP = 12000 := by
sorry

end find_original_selling_price_l1237_123745


namespace length_PQ_l1237_123732

theorem length_PQ (AB BC CA AH : ℝ) (P Q : ℝ) : 
  AB = 7 → BC = 8 → CA = 9 → 
  AH = 3 * Real.sqrt 5 → 
  PQ = AQ - AP → 
  AQ = 7 * (Real.sqrt 5) / 3 → 
  AP = 9 * (Real.sqrt 5) / 5 → 
  PQ = Real.sqrt 5 * 8 / 15 :=
by
  intros hAB hBC hCA hAH hPQ hAQ hAP
  sorry

end length_PQ_l1237_123732


namespace find_a_b_l1237_123725

theorem find_a_b :
  ∃ a b : ℝ, 
    (a = -4) ∧ (b = -9) ∧
    (∀ x : ℝ, |8 * x + 9| < 7 ↔ a * x^2 + b * x - 2 > 0) := 
sorry

end find_a_b_l1237_123725


namespace measure_of_one_interior_angle_of_regular_octagon_l1237_123781

theorem measure_of_one_interior_angle_of_regular_octagon 
  (n : Nat) (h_n : n = 8) : 
  (one_interior_angle : ℝ) = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l1237_123781


namespace sum_of_two_numbers_l1237_123773

theorem sum_of_two_numbers (x y : ℕ) (h1 : y = 2 * x - 3) (h2 : x = 14) : x + y = 39 :=
by
  sorry

end sum_of_two_numbers_l1237_123773


namespace day_of_week_299th_day_2004_l1237_123705

noncomputable def day_of_week (day: ℕ): ℕ := day % 7

theorem day_of_week_299th_day_2004 : 
  ∀ (d: ℕ), day_of_week d = 3 → d = 45 → day_of_week 299 = 5 :=
by
  sorry

end day_of_week_299th_day_2004_l1237_123705


namespace quadratic_has_two_roots_l1237_123792

variables {R : Type*} [LinearOrderedField R]

theorem quadratic_has_two_roots (a1 a2 a3 b1 b2 b3 : R) 
  (h1 : a1 * a2 * a3 = b1 * b2 * b3) (h2 : a1 * a2 * a3 > 1) : 
  (4 * a1^2 - 4 * b1 > 0) ∨ (4 * a2^2 - 4 * b2 > 0) ∨ (4 * a3^2 - 4 * b3 > 0) :=
sorry

end quadratic_has_two_roots_l1237_123792


namespace units_digit_of_square_ne_2_l1237_123708

theorem units_digit_of_square_ne_2 (n : ℕ) : (n * n) % 10 ≠ 2 :=
sorry

end units_digit_of_square_ne_2_l1237_123708


namespace suitable_survey_method_l1237_123795

-- Definitions based on conditions
def large_population (n : ℕ) : Prop := n > 10000  -- Example threshold for large population
def impractical_comprehensive_survey : Prop := true  -- Given in condition

-- The statement of the problem
theorem suitable_survey_method (n : ℕ) (h1 : large_population n) (h2 : impractical_comprehensive_survey) : 
  ∃ method : String, method = "sampling survey" :=
sorry

end suitable_survey_method_l1237_123795


namespace average_age_increase_l1237_123741

theorem average_age_increase 
    (num_students : ℕ) (avg_age_students : ℕ) (age_staff : ℕ)
    (H1: num_students = 32)
    (H2: avg_age_students = 16)
    (H3: age_staff = 49) : 
    ((num_students * avg_age_students + age_staff) / (num_students + 1) - avg_age_students = 1) :=
by
  sorry

end average_age_increase_l1237_123741


namespace people_in_room_l1237_123727

theorem people_in_room (total_people total_chairs : ℕ) (h1 : (2/3 : ℚ) * total_chairs = 1/2 * total_people)
  (h2 : total_chairs - (2/3 : ℚ) * total_chairs = 8) : total_people = 32 := 
by
  sorry

end people_in_room_l1237_123727


namespace exams_in_fourth_year_l1237_123763

variable (a b c d e : ℕ)

theorem exams_in_fourth_year:
  a + b + c + d + e = 31 ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e = 3 * a → d = 8 := by
  sorry

end exams_in_fourth_year_l1237_123763


namespace beer_drawing_time_l1237_123751

theorem beer_drawing_time :
  let rate_A := 1 / 5
  let rate_C := 1 / 4
  let combined_rate := 9 / 20
  let extra_beer := 12
  let total_drawn := 48
  let t := total_drawn / combined_rate
  t = 48 * 20 / 9 :=
by {
  sorry -- proof not required
}

end beer_drawing_time_l1237_123751


namespace more_crayons_given_to_Lea_than_Mae_l1237_123728

-- Define the initial number of crayons
def initial_crayons : ℕ := 4 * 8

-- Condition: Nori gave 5 crayons to Mae
def crayons_given_to_Mae : ℕ := 5

-- Condition: Nori has 15 crayons left after giving some to Lea
def crayons_left_after_giving_to_Lea : ℕ := 15

-- Define the number of crayons after giving to Mae
def crayons_after_giving_to_Mae : ℕ := initial_crayons - crayons_given_to_Mae

-- Define the number of crayons given to Lea
def crayons_given_to_Lea : ℕ := crayons_after_giving_to_Mae - crayons_left_after_giving_to_Lea

-- Prove the number of more crayons given to Lea than Mae
theorem more_crayons_given_to_Lea_than_Mae : (crayons_given_to_Lea - crayons_given_to_Mae) = 7 := by
  sorry

end more_crayons_given_to_Lea_than_Mae_l1237_123728


namespace choir_member_count_l1237_123760

theorem choir_member_count (n : ℕ) : 
  (n ≡ 4 [MOD 7]) ∧ 
  (n ≡ 8 [MOD 6]) ∧ 
  (50 ≤ n ∧ n ≤ 200) 
  ↔ 
  (n = 60 ∨ n = 102 ∨ n = 144 ∨ n = 186) := 
by 
  sorry

end choir_member_count_l1237_123760


namespace parabola_directrix_l1237_123761

theorem parabola_directrix (x y : ℝ) :
    x^2 = - (1 / 4) * y → y = - (1 / 16) :=
by
  sorry

end parabola_directrix_l1237_123761


namespace cos_double_angle_l1237_123790

open Real

theorem cos_double_angle (α : Real) (h : tan α = 3) : cos (2 * α) = -4/5 :=
  sorry

end cos_double_angle_l1237_123790


namespace all_children_receive_candy_iff_power_of_two_l1237_123759

theorem all_children_receive_candy_iff_power_of_two (n : ℕ) : 
  (∀ (k : ℕ), k < n → ∃ (m : ℕ), (m * (m + 1) / 2) % n = k) ↔ ∃ (k : ℕ), n = 2^k :=
by sorry

end all_children_receive_candy_iff_power_of_two_l1237_123759


namespace complement_A_correct_l1237_123707

-- Define the universal set U
def U : Set ℝ := { x | x ≥ 1 ∨ x ≤ -1 }

-- Define the set A
def A : Set ℝ := { x | 1 < x ∧ x ≤ 2 }

-- Define the complement of A in U
def complement_A_in_U : Set ℝ := { x | x ≤ -1 ∨ x = 1 ∨ x > 2 }

-- Prove that the complement of A in U is as defined
theorem complement_A_correct : (U \ A) = complement_A_in_U := by
  sorry

end complement_A_correct_l1237_123707


namespace correct_statement_l1237_123719

theorem correct_statement : ∀ (a b : ℝ), ((a ≠ b ∧ ¬(a < b ∧ ∀ x, (x = a ∨ x = b) → x = a ∨ x = b)) ∧
                                            ¬(∀ p q : ℝ, p = q → p = q) ∧
                                            ¬(∀ a : ℝ, |a| = -a → a < 0) ∧
                                            ¬(∀ a b : ℝ, (a ≠ 0 ∧ b ≠ 0 ∧ (a = -b)) → (a / b = -1))) :=
by sorry

-- Explanation of conditions:
-- a  ≠ b ensures two distinct points
-- ¬(a < b ∧ ∀ x, (x = a ∨ x = b) → x is between a and b) incorrectly rephrased as shortest distance as a line segment
-- ¬(∀ p q : ℝ, p = q → p = q) is not directly used, a minimum to refute the concept as required.
-- |a| = -a → a < 0 reinterpreted as a ≤ 0 but incorrectly stated as < 0 explicitly refuted
-- ¬(∀ a b : ℝ, a ≠ 0 and/or b ≠ 0 maintained where a / b not strictly required/misinterpreted)

end correct_statement_l1237_123719


namespace solve_eq1_solve_eq2_solve_eq3_solve_eq4_l1237_123721

theorem solve_eq1 (x : ℝ) : (3 * x + 2) ^ 2 = 25 ↔ (x = 1 ∨ x = -7 / 3) := by
  sorry

theorem solve_eq2 (x : ℝ) : 3 * x ^ 2 - 1 = 4 * x ↔ (x = (2 + Real.sqrt 7) / 3 ∨ x = (2 - Real.sqrt 7) / 3) := by
  sorry

theorem solve_eq3 (x : ℝ) : (2 * x - 1) ^ 2 = 3 * (2 * x + 1) ↔ (x = -1 / 2 ∨ x = 1) := by
  sorry

theorem solve_eq4 (x : ℝ) : x ^ 2 - 7 * x + 10 = 0 ↔ (x = 5 ∨ x = 2) := by
  sorry

end solve_eq1_solve_eq2_solve_eq3_solve_eq4_l1237_123721


namespace area_of_circle_B_l1237_123762

theorem area_of_circle_B (rA rB : ℝ) (h : π * rA^2 = 16 * π) (h1 : rB = 2 * rA) : π * rB^2 = 64 * π :=
by
  sorry

end area_of_circle_B_l1237_123762


namespace eccentricity_of_hyperbola_is_e_l1237_123788

-- Definitions and given conditions
variable (a b c : ℝ)
variable (h_a_pos : a > 0) (h_b_pos : b > 0)
variable (h_hyperbola : ∀ x y : ℝ, (x^2)/(a^2) - (y^2)/(b^2) = 1)
variable (h_left_focus : ∀ F : ℝ × ℝ, F = (-c, 0))
variable (h_circle : ∀ E : ℝ × ℝ, E.1^2 + E.2^2 = a^2)
variable (h_parabola : ∀ P : ℝ × ℝ, P.2^2 = 4*c*P.1)
variable (h_midpoint : ∀ E P F : ℝ × ℝ, E = (F.1 + P.1) / 2 ∧ E.2 = (F.2 + P.2) / 2)

-- The statement to be proved
theorem eccentricity_of_hyperbola_is_e :
    ∃ e : ℝ, e = (Real.sqrt 5 + 1) / 2 :=
sorry

end eccentricity_of_hyperbola_is_e_l1237_123788


namespace exist_directed_graph_two_step_l1237_123703

theorem exist_directed_graph_two_step {n : ℕ} (h : n > 4) :
  ∃ G : SimpleGraph (Fin n), 
    (∀ u v : Fin n, u ≠ v → 
      (G.Adj u v ∨ (∃ w : Fin n, u ≠ w ∧ w ≠ v ∧ G.Adj u w ∧ G.Adj w v))) :=
sorry

end exist_directed_graph_two_step_l1237_123703


namespace purchase_price_l1237_123711

theorem purchase_price (P : ℝ)
  (down_payment : ℝ) (monthly_payment : ℝ) (number_of_payments : ℝ)
  (interest_rate : ℝ) (total_paid : ℝ)
  (h1 : down_payment = 12)
  (h2 : monthly_payment = 10)
  (h3 : number_of_payments = 12)
  (h4 : interest_rate = 0.10714285714285714)
  (h5 : total_paid = 132) :
  P = 132 / 1.1071428571428572 :=
by
  sorry

end purchase_price_l1237_123711


namespace find_ordered_pair_l1237_123799

theorem find_ordered_pair (x y : ℝ) 
  (h1 : x + y = (7 - x) + (7 - y)) 
  (h2 : x - y = (x - 2) + (y - 2)) : 
  (x = 5 ∧ y = 2) :=
by
  sorry

end find_ordered_pair_l1237_123799


namespace problem1_l1237_123720

variable (x : ℝ)

theorem problem1 : 5 * x^2 * x^4 + x^8 / (-x)^2 = 6 * x^6 :=
  sorry

end problem1_l1237_123720


namespace stickers_remaining_l1237_123755

theorem stickers_remaining (total_stickers : ℕ) (front_page_stickers : ℕ) (other_pages_stickers : ℕ) (num_other_pages : ℕ) (remaining_stickers : ℕ)
  (h0 : total_stickers = 89)
  (h1 : front_page_stickers = 3)
  (h2 : other_pages_stickers = 7)
  (h3 : num_other_pages = 6)
  (h4 : remaining_stickers = total_stickers - (front_page_stickers + other_pages_stickers * num_other_pages)) :
  remaining_stickers = 44 :=
by
  sorry

end stickers_remaining_l1237_123755


namespace solve_system_l1237_123772

theorem solve_system (x y z : ℝ) (h1 : x + y + z = 6) (h2 : x * y + y * z + z * x = 11) (h3 : x * y * z = 6) :
  (x = 1 ∧ y = 2 ∧ z = 3) ∨ (x = 1 ∧ y = 3 ∧ z = 2) ∨ (x = 2 ∧ y = 1 ∧ z = 3) ∨ 
  (x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 1 ∧ z = 2) ∨ (x = 3 ∧ y = 2 ∧ z = 1) :=
sorry

end solve_system_l1237_123772


namespace sub_eight_l1237_123768

theorem sub_eight (x y : ℝ) (h1 : 3 = 0.15 * x) (h2 : 3 = 0.25 * y) : x - y = 8 :=
by
  sorry

end sub_eight_l1237_123768


namespace julia_paid_for_puppy_l1237_123769

theorem julia_paid_for_puppy :
  let dog_food := 20
  let treat := 2.5
  let treats := 2 * treat
  let toys := 15
  let crate := 20
  let bed := 20
  let collar_leash := 15
  let discount_rate := 0.20
  let total_before_discount := dog_food + treats + toys + crate + bed + collar_leash
  let discount := discount_rate * total_before_discount
  let total_after_discount := total_before_discount - discount
  let total_spent := 96
  total_spent - total_after_discount = 20 := 
by 
  sorry

end julia_paid_for_puppy_l1237_123769


namespace binomial_coeff_sum_l1237_123752

theorem binomial_coeff_sum :
  ∀ a b : ℝ, 15 * a^4 * b^2 = 135 ∧ 6 * a^5 * b = -18 →
  (a + b) ^ 6 = 64 :=
by
  intros a b h
  sorry

end binomial_coeff_sum_l1237_123752


namespace halfway_between_l1237_123733

-- Definitions based on given conditions
def a : ℚ := 1 / 7
def b : ℚ := 1 / 9

-- Theorem that needs to be proved
theorem halfway_between (h : True) : (a + b) / 2 = 8 / 63 := by
  sorry

end halfway_between_l1237_123733


namespace roots_reciprocal_sum_l1237_123724

theorem roots_reciprocal_sum
  (a b c : ℂ)
  (h : Polynomial.roots (Polynomial.C 1 + Polynomial.X - Polynomial.C 1 * Polynomial.X^2 + Polynomial.C 1 * Polynomial.X^3) = {a, b, c}) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = -2 :=
by
  sorry

end roots_reciprocal_sum_l1237_123724


namespace actual_time_of_storm_l1237_123765

theorem actual_time_of_storm
  (malfunctioned_hours_tens_digit : ℕ)
  (malfunctioned_hours_units_digit : ℕ)
  (malfunctioned_minutes_tens_digit : ℕ)
  (malfunctioned_minutes_units_digit : ℕ)
  (original_time : ℕ × ℕ)
  (hours_tens_digit : ℕ := 2)
  (hours_units_digit : ℕ := 0)
  (minutes_tens_digit : ℕ := 0)
  (minutes_units_digit : ℕ := 9) :
  (malfunctioned_hours_tens_digit = hours_tens_digit + 1 ∨ malfunctioned_hours_tens_digit = hours_tens_digit - 1) →
  (malfunctioned_hours_units_digit = hours_units_digit + 1 ∨ malfunctioned_hours_units_digit = hours_units_digit - 1) →
  (malfunctioned_minutes_tens_digit = minutes_tens_digit + 1 ∨ malfunctioned_minutes_tens_digit = minutes_tens_digit - 1) →
  (malfunctioned_minutes_units_digit = minutes_units_digit + 1 ∨ malfunctioned_minutes_units_digit = minutes_units_digit - 1) →
  original_time = (11, 18) :=
by
  sorry

end actual_time_of_storm_l1237_123765


namespace distance_A_beats_B_l1237_123784

theorem distance_A_beats_B
  (time_A time_B : ℝ)
  (dist : ℝ)
  (time_A_eq : time_A = 198)
  (time_B_eq : time_B = 220)
  (dist_eq : dist = 3) :
  (dist / time_A) * time_B - dist = 333 / 1000 :=
by
  sorry

end distance_A_beats_B_l1237_123784


namespace solve_a₃_l1237_123796

noncomputable def geom_seq (a₁ a₅ a₃ : ℝ) : Prop :=
a₁ = 1 / 9 ∧ a₅ = 9 ∧ a₁ * a₅ = a₃^2

theorem solve_a₃ : ∃ a₃ : ℝ, geom_seq (1/9) 9 a₃ ∧ a₃ = 1 :=
by
  sorry

end solve_a₃_l1237_123796


namespace find_constants_l1237_123737

noncomputable def f (x : ℕ) (a c : ℕ) : ℝ :=
  if x < a then c / Real.sqrt x else c / Real.sqrt a

theorem find_constants (a c : ℕ) (h₁ : f 4 a c = 30) (h₂ : f a a c = 5) : 
  c = 60 ∧ a = 144 := 
by
  sorry

end find_constants_l1237_123737


namespace knowledge_competition_score_l1237_123735

theorem knowledge_competition_score (x : ℕ) (hx : x ≤ 20) : 5 * x - (20 - x) ≥ 88 :=
  sorry

end knowledge_competition_score_l1237_123735


namespace tan_product_ge_sqrt2_l1237_123786

variable {α β γ : ℝ}

theorem tan_product_ge_sqrt2 (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) (hγ : 0 < γ ∧ γ < π / 2) 
  (h : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) : 
  Real.tan α * Real.tan β * Real.tan γ ≥ 2 * Real.sqrt 2 := 
by
  sorry

end tan_product_ge_sqrt2_l1237_123786


namespace years_ago_l1237_123701

theorem years_ago (M D X : ℕ) (hM : M = 41) (hD : D = 23) 
  (h_eq : M - X = 2 * (D - X)) : X = 5 := by 
  sorry

end years_ago_l1237_123701


namespace bmw_length_l1237_123715

theorem bmw_length : 
  let horiz1 : ℝ := 2 -- Length of each horizontal segment in 'B'
  let horiz2 : ℝ := 2 -- Length of each horizontal segment in 'B'
  let vert1  : ℝ := 2 -- Length of each vertical segment in 'B'
  let vert2  : ℝ := 2 -- Length of each vertical segment in 'B'
  let vert3  : ℝ := 2 -- Length of each vertical segment in 'M'
  let vert4  : ℝ := 2 -- Length of each vertical segment in 'M'
  let vert5  : ℝ := 2 -- Length of each vertical segment in 'W'
  let diag1  : ℝ := Real.sqrt 2 -- Length of each diagonal segment in 'W'
  let diag2  : ℝ := Real.sqrt 2 -- Length of each diagonal segment in 'W'
  (horiz1 + horiz2 + vert1 + vert2 + vert3 + vert4 + vert5 + diag1 + diag2) = 14 + 2 * Real.sqrt 2 :=
by
  sorry

end bmw_length_l1237_123715


namespace abs_equality_holds_if_interval_l1237_123739

noncomputable def quadratic_abs_equality (x : ℝ) : Prop :=
  |x^2 - 8 * x + 12| = x^2 - 8 * x + 12

theorem abs_equality_holds_if_interval (x : ℝ) :
  quadratic_abs_equality x ↔ (x ≤ 2 ∨ x ≥ 6) :=
by
  sorry

end abs_equality_holds_if_interval_l1237_123739


namespace max_checkers_on_board_l1237_123793

-- Define the size of the board.
def board_size : ℕ := 8

-- Define the max number of checkers per row/column.
def max_checkers_per_line : ℕ := 3

-- Define the conditions of the board.
structure BoardConfiguration :=
  (rows : Fin board_size → Fin (max_checkers_per_line + 1))
  (columns : Fin board_size → Fin (max_checkers_per_line + 1))
  (valid : ∀ (i : Fin board_size), rows i ≤ max_checkers_per_line ∧ columns i ≤ max_checkers_per_line)

-- Define the function to calculate the total number of checkers.
def total_checkers (config : BoardConfiguration) : ℕ :=
  Finset.univ.sum (λ i => config.rows i + config.columns i)

-- The theorem which states that the maximum number of checkers is 30.
theorem max_checkers_on_board : ∃ (config : BoardConfiguration), total_checkers config = 30 :=
  sorry

end max_checkers_on_board_l1237_123793


namespace avg_weight_section_b_l1237_123748

/-- Definition of the average weight of section B based on given conditions --/
theorem avg_weight_section_b :
  let W_A := 50
  let W_class := 54.285714285714285
  let num_A := 40
  let num_B := 30
  let total_class_weight := (num_A + num_B) * W_class
  let total_A_weight := num_A * W_A
  let total_B_weight := total_class_weight - total_A_weight
  let W_B := total_B_weight / num_B
  W_B = 60 :=
by
  sorry

end avg_weight_section_b_l1237_123748


namespace factorize_m_square_minus_4m_l1237_123758

theorem factorize_m_square_minus_4m (m : ℝ) : m^2 - 4 * m = m * (m - 4) :=
by
  sorry

end factorize_m_square_minus_4m_l1237_123758


namespace cost_of_fencing_per_meter_in_cents_l1237_123776

-- Definitions for the conditions
def ratio_length_width : ℕ := 3
def ratio_width_length : ℕ := 2
def total_area : ℕ := 3750
def total_fencing_cost : ℕ := 175

-- Main theorem statement with proof omitted
theorem cost_of_fencing_per_meter_in_cents :
  (ratio_length_width = 3) →
  (ratio_width_length = 2) →
  (total_area = 3750) →
  (total_fencing_cost = 175) →
  ∃ (cost_per_meter_in_cents : ℕ), cost_per_meter_in_cents = 70 :=
by
  intros h1 h2 h3 h4
  sorry

end cost_of_fencing_per_meter_in_cents_l1237_123776


namespace sum_of_coefficients_l1237_123702

theorem sum_of_coefficients :
  (∃ a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℤ,
    (1 - 2 * x)^9 = a_9 * x^9 + a_8 * x^8 + a_7 * x^7 + a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0) →
    a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 = -2 :=
by
  sorry

end sum_of_coefficients_l1237_123702


namespace arithmetic_sequence_ninth_term_eq_l1237_123779

theorem arithmetic_sequence_ninth_term_eq :
  let a_1 := (3 : ℚ) / 8
  let a_17 := (2 : ℚ) / 3
  let a_9 := (a_1 + a_17) / 2
  a_9 = (25 : ℚ) / 48 := by
  let a_1 := (3 : ℚ) / 8
  let a_17 := (2 : ℚ) / 3
  let a_9 := (a_1 + a_17) / 2
  sorry

end arithmetic_sequence_ninth_term_eq_l1237_123779


namespace sqrt_meaningful_range_iff_l1237_123746

noncomputable def sqrt_meaningful_range (x : ℝ) : Prop :=
  (∃ r : ℝ, r ≥ 0 ∧ r * r = x - 2023)

theorem sqrt_meaningful_range_iff {x : ℝ} : sqrt_meaningful_range x ↔ x ≥ 2023 :=
by
  sorry

end sqrt_meaningful_range_iff_l1237_123746


namespace mandy_quarters_l1237_123767

theorem mandy_quarters (q : ℕ) : 
  40 < q ∧ q < 400 ∧ 
  q % 6 = 2 ∧ 
  q % 7 = 2 ∧ 
  q % 8 = 2 →
  (q = 170 ∨ q = 338) :=
by
  intro h
  sorry

end mandy_quarters_l1237_123767


namespace inequality_solution_l1237_123778

theorem inequality_solution
  (f : ℝ → ℝ)
  (h_deriv : ∀ x : ℝ, deriv f x > 2 * f x)
  (h_value : f (1/2) = Real.exp 1)
  (x : ℝ)
  (h_pos : 0 < x) :
  f (Real.log x) < x^2 ↔ x < Real.exp (1/2) :=
sorry

end inequality_solution_l1237_123778


namespace goods_train_speed_l1237_123754

theorem goods_train_speed:
  let speed_mans_train := 100   -- in km/h
  let length_goods_train := 280 -- in meters
  let passing_time := 9         -- in seconds
  ∃ speed_goods_train: ℝ, 
  (speed_mans_train + speed_goods_train) * (5 / 18) * passing_time = length_goods_train ↔ speed_goods_train = 12 :=
by
  sorry

end goods_train_speed_l1237_123754


namespace km_to_m_is_750_l1237_123713

-- Define 1 kilometer equals 5 hectometers
def km_to_hm := 5

-- Define 1 hectometer equals 10 dekameters
def hm_to_dam := 10

-- Define 1 dekameter equals 15 meters
def dam_to_m := 15

-- Theorem stating that the number of meters in one kilometer is 750
theorem km_to_m_is_750 : 1 * km_to_hm * hm_to_dam * dam_to_m = 750 :=
by 
  -- Proof goes here
  sorry

end km_to_m_is_750_l1237_123713


namespace students_in_class_l1237_123749

theorem students_in_class
  (S : ℕ)
  (h1 : S / 3 * 4 / 3 = 12) :
  S = 36 := 
sorry

end students_in_class_l1237_123749


namespace largest_n_for_triangle_property_l1237_123785

-- Define the triangle property for a set
def triangle_property (S : Set ℕ) : Prop :=
  ∀ {a b c : ℕ}, a ∈ S → b ∈ S → c ∈ S → a < b → b < c → a + b > c

-- Define the smallest subset that violates the triangle property
def violating_subset : Set ℕ := {5, 6, 11, 17, 28, 45, 73, 118, 191, 309}

-- Define the set of consecutive integers from 5 to n
def consecutive_integers (n : ℕ) : Set ℕ := {x : ℕ | 5 ≤ x ∧ x ≤ n}

-- The theorem we want to prove
theorem largest_n_for_triangle_property : ∀ (S : Set ℕ), S = consecutive_integers 308 → triangle_property S := sorry

end largest_n_for_triangle_property_l1237_123785


namespace verify_magic_square_l1237_123764

-- Define the grid as a 3x3 matrix
def magic_square := Matrix (Fin 3) (Fin 3) ℕ

-- Conditions for the magic square
def is_magic_square (m : magic_square) : Prop :=
  (∀ i : Fin 3, (m i 0) + (m i 1) + (m i 2) = 15) ∧
  (∀ j : Fin 3, (m 0 j) + (m 1 j) + (m 2 j) = 15) ∧
  ((m 0 0) + (m 1 1) + (m 2 2) = 15) ∧
  ((m 0 2) + (m 1 1) + (m 2 0) = 15)

-- Given specific filled numbers in the grid
def given_filled_values (m : magic_square) : Prop :=
  (m 0 1 = 5) ∧
  (m 1 0 = 2) ∧
  (m 2 2 = 8)

-- The complete grid based on the solution
def completed_magic_square : magic_square :=
  ![![4, 9, 2], ![3, 5, 7], ![8, 1, 6]]

-- The main theorem to prove
theorem verify_magic_square : 
  is_magic_square completed_magic_square ∧ 
  given_filled_values completed_magic_square := 
by 
  sorry

end verify_magic_square_l1237_123764


namespace problem1_part1_problem1_part2_problem2_l1237_123757

open Set

-- Definitions for sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 2}
def U : Set ℝ := univ

-- Part (1) of the problem
theorem problem1_part1 : A ∩ B = {x | 2 < x ∧ x ≤ 3} :=
sorry

theorem problem1_part2 : A ∪ (U \ B) = {x | x ≤ 3} :=
sorry

-- Definitions for set C
def C (a : ℝ) : Set ℝ := {x | 1 < x ∧ x < a}

-- Part (2) of the problem
theorem problem2 (a : ℝ) (h : C a ⊆ A) : 1 < a ∧ a ≤ 3 :=
sorry

end problem1_part1_problem1_part2_problem2_l1237_123757


namespace evaluate_square_difference_l1237_123780

theorem evaluate_square_difference:
  let a := 70
  let b := 30
  (a^2 - b^2) = 4000 :=
by
  sorry

end evaluate_square_difference_l1237_123780


namespace at_least_one_genuine_l1237_123704

theorem at_least_one_genuine (batch : Finset ℕ) 
  (h_batch_size : batch.card = 12) 
  (genuine_items : Finset ℕ)
  (h_genuine_size : genuine_items.card = 10)
  (defective_items : Finset ℕ)
  (h_defective_size : defective_items.card = 2)
  (h_disjoint : genuine_items ∩ defective_items = ∅)
  (drawn_items : Finset ℕ)
  (h_draw_size : drawn_items.card = 3)
  (h_subset : drawn_items ⊆ batch)
  (h_union : genuine_items ∪ defective_items = batch) :
  (∃ (x : ℕ), x ∈ drawn_items ∧ x ∈ genuine_items) :=
sorry

end at_least_one_genuine_l1237_123704


namespace calc_miscellaneous_collective_expenses_l1237_123738

def individual_needed_amount : ℕ := 450
def additional_needed_amount : ℕ := 475
def total_students : ℕ := 6
def first_day_amount : ℕ := 600
def second_day_amount : ℕ := 900
def third_day_amount : ℕ := 400
def days : ℕ := 4

def total_individual_goal : ℕ := individual_needed_amount + additional_needed_amount
def total_students_goal : ℕ := total_individual_goal * total_students
def total_first_3_days : ℕ := first_day_amount + second_day_amount + third_day_amount
def total_next_4_days : ℕ := (total_first_3_days / 2) * days
def total_raised : ℕ := total_first_3_days + total_next_4_days

def miscellaneous_collective_expenses : ℕ := total_raised - total_students_goal

theorem calc_miscellaneous_collective_expenses : miscellaneous_collective_expenses = 150 := by
  sorry

end calc_miscellaneous_collective_expenses_l1237_123738


namespace rhyme_around_3_7_l1237_123794

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def rhymes_around (p q m : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p ≠ q ∧ ((p < m ∧ q > m ∧ q - m = m - p) ∨ (p > m ∧ q < m ∧ p - m = m - q))

theorem rhyme_around_3_7 : ∃ m : ℕ, rhymes_around 3 7 m ∧ m = 5 :=
by
  sorry

end rhyme_around_3_7_l1237_123794


namespace cylinder_volume_l1237_123775

theorem cylinder_volume (r l : ℝ) (h1 : r = 1) (h2 : l = 2 * r) : 
  ∃ V : ℝ, V = 2 * Real.pi := 
by 
  sorry

end cylinder_volume_l1237_123775


namespace casey_stays_for_n_months_l1237_123714

-- Definitions based on conditions.
def weekly_cost : ℕ := 280
def monthly_cost : ℕ := 1000
def weeks_per_month : ℕ := 4
def total_savings : ℕ := 360

-- Calculate monthly cost when paying weekly.
def monthly_cost_weekly := weekly_cost * weeks_per_month

-- Calculate savings per month when paying monthly instead of weekly.
def savings_per_month := monthly_cost_weekly - monthly_cost

-- Define the problem statement.
theorem casey_stays_for_n_months :
  (total_savings / savings_per_month) = 3 := by
  -- Proof is omitted.
  sorry

end casey_stays_for_n_months_l1237_123714


namespace calculate_lego_set_cost_l1237_123734

variable (total_revenue_after_tax : ℝ) (little_cars_base_price : ℝ)
  (discount_rate : ℝ) (tax_rate : ℝ) (num_little_cars : ℕ)
  (num_action_figures : ℕ) (num_board_games : ℕ)
  (lego_set_cost_before_tax : ℝ)

theorem calculate_lego_set_cost :
  total_revenue_after_tax = 136.50 →
  little_cars_base_price = 5 →
  discount_rate = 0.10 →
  tax_rate = 0.05 →
  num_little_cars = 3 →
  num_action_figures = 2 →
  num_board_games = 1 →
  lego_set_cost_before_tax = 85 :=
by
  sorry

end calculate_lego_set_cost_l1237_123734


namespace jane_uses_40_ribbons_l1237_123791

theorem jane_uses_40_ribbons :
  (∀ dresses1 dresses2 ribbons_per_dress, 
  dresses1 = 2 * 7 ∧ 
  dresses2 = 3 * 2 → 
  ribbons_per_dress = 2 → 
  (dresses1 + dresses2) * ribbons_per_dress = 40)
:= 
by 
  sorry

end jane_uses_40_ribbons_l1237_123791


namespace arrange_in_circle_l1237_123747

open Nat

noncomputable def smallest_n := 70

theorem arrange_in_circle (n : ℕ) (h : n = 70) :
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ n →
    (∀ j : ℕ, 1 ≤ j ∧ j ≤ 40 → k > ((k + j) % n)) ∨
    (∀ p : ℕ, 1 ≤ p ∧ p ≤ 30 → k < ((k + p) % n))) :=
by
  sorry

end arrange_in_circle_l1237_123747


namespace robert_time_to_complete_l1237_123740

noncomputable def time_to_complete_semicircle_path (length_mile : ℝ) (width_feet : ℝ) (speed_mph : ℝ) (mile_to_feet : ℝ) : ℝ :=
  let diameter_mile := width_feet / mile_to_feet
  let radius_mile := diameter_mile / 2
  let circumference_mile := 2 * Real.pi * radius_mile
  let semicircle_length_mile := circumference_mile / 2
  semicircle_length_mile / speed_mph

theorem robert_time_to_complete :
  time_to_complete_semicircle_path 1 40 5 5280 = Real.pi / 10 :=
by
  sorry

end robert_time_to_complete_l1237_123740


namespace coordinates_provided_l1237_123730

-- Define the coordinates of point P in the Cartesian coordinate system
structure Point where
  x : ℝ
  y : ℝ

-- Define the point P with its given coordinates
def P : Point := {x := 3, y := -5}

-- Lean 4 statement for the proof problem
theorem coordinates_provided : (P.x, P.y) = (3, -5) := by
  -- Proof not provided
  sorry

end coordinates_provided_l1237_123730


namespace num_values_x_satisfying_l1237_123710

theorem num_values_x_satisfying (
  f : ℝ → ℝ → ℝ)
  (cos : ℝ → ℝ)
  (sin : ℝ → ℝ)
  (x : ℝ)
  (h_eq : ∀ x, f (cos x) (sin x) = 2 ↔ (cos x) ^ 2 + 3 * (sin x) ^ 2 = 2)
  (h_interval : ∀ x, -20 < x ∧ x < 90)
  (h_cos_sin : ∀ x, cos x = cos (x) ∧ sin x = sin (x)) :
  ∃ n, n = 70 := sorry

end num_values_x_satisfying_l1237_123710


namespace min_sum_of_2x2_grid_l1237_123770

theorem min_sum_of_2x2_grid (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
(h_sum : a * b + c * d + a * c + b * d = 2015) : a + b + c + d = 88 :=
sorry

end min_sum_of_2x2_grid_l1237_123770


namespace sin_A_plus_B_eq_max_area_eq_l1237_123700

-- Conditions for problem 1 and 2
variables (A B C a b c : ℝ)
variable (h_A_B_C : A + B + C = Real.pi)
variable (h_sin_C_div_2 : Real.sin (C / 2) = 2 * Real.sqrt 2 / 3)

noncomputable def sin_A_plus_B := Real.sin (A + B)

-- Problem 1: Prove that sin(A + B) = 4 * sqrt 2 / 9
theorem sin_A_plus_B_eq : sin_A_plus_B A B = 4 * Real.sqrt 2 / 9 :=
by sorry

-- Adding additional conditions for problem 2
variable (h_a_b_sum : a + b = 2 * Real.sqrt 2)

noncomputable def area (a b C : ℝ) := (1 / 2) * a * b * (2 * Real.sin (C / 2) * (Real.cos (C / 2)))

-- Problem 2: Prove that the maximum value of the area S of triangle ABC is 4 * sqrt 2 / 9
theorem max_area_eq : ∃ S, S = area a b C ∧ S ≤ 4 * Real.sqrt 2 / 9 :=
by sorry

end sin_A_plus_B_eq_max_area_eq_l1237_123700


namespace smallest_ratio_l1237_123783

theorem smallest_ratio (r s : ℤ) (h1 : 3 * r ≥ 2 * s - 3) (h2 : 4 * s ≥ r + 12) : 
  (∃ r s, (r : ℚ) / s = 1 / 2) :=
by 
  sorry

end smallest_ratio_l1237_123783


namespace tilly_total_profit_l1237_123729

theorem tilly_total_profit :
  let bags_sold := 100
  let selling_price_per_bag := 10
  let buying_price_per_bag := 7
  let profit_per_bag := selling_price_per_bag - buying_price_per_bag
  let total_profit := bags_sold * profit_per_bag
  total_profit = 300 :=
by
  let bags_sold := 100
  let selling_price_per_bag := 10
  let buying_price_per_bag := 7
  let profit_per_bag := selling_price_per_bag - buying_price_per_bag
  let total_profit := bags_sold * profit_per_bag
  sorry

end tilly_total_profit_l1237_123729


namespace line_intersects_ellipse_if_and_only_if_l1237_123774

theorem line_intersects_ellipse_if_and_only_if (k : ℝ) (m : ℝ) :
  (∀ x, ∃ y, y = k * x + 1 ∧ (x^2 / 5 + y^2 / m = 1)) ↔ (m ≥ 1 ∧ m ≠ 5) := 
sorry

end line_intersects_ellipse_if_and_only_if_l1237_123774


namespace rectangle_length_l1237_123712

-- Define a structure for the rectangle.
structure Rectangle where
  breadth : ℝ
  length : ℝ
  area : ℝ

-- Define the given conditions.
def givenConditions (r : Rectangle) : Prop :=
  r.length = 3 * r.breadth ∧ r.area = 6075

-- State the theorem.
theorem rectangle_length (r : Rectangle) (h : givenConditions r) : r.length = 135 :=
by
  sorry

end rectangle_length_l1237_123712


namespace find_omitted_angle_l1237_123798

-- Definitions and conditions
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

def omitted_angle (calculated_sum actual_sum : ℝ) : ℝ :=
  actual_sum - calculated_sum

-- The theorem to be proven
theorem find_omitted_angle (n : ℕ) (h₁ : 1958 + 22 = sum_of_interior_angles n) :
  omitted_angle 1958 (sum_of_interior_angles n) = 22 :=
by
  sorry

end find_omitted_angle_l1237_123798


namespace sum_of_cubes_of_consecutive_integers_div_by_9_l1237_123718

theorem sum_of_cubes_of_consecutive_integers_div_by_9 (x : ℤ) : 
  let a := (x - 1) ^ 3
  let b := x ^ 3
  let c := (x + 1) ^ 3
  (a + b + c) % 9 = 0 :=
by
  sorry

end sum_of_cubes_of_consecutive_integers_div_by_9_l1237_123718


namespace distance_from_yz_plane_l1237_123726

theorem distance_from_yz_plane (x z : ℝ) : 
  (abs (-6) = (abs x) / 2) → abs x = 12 :=
by
  sorry

end distance_from_yz_plane_l1237_123726


namespace min_combined_number_of_horses_and_ponies_l1237_123736

theorem min_combined_number_of_horses_and_ponies :
  ∃ P H : ℕ, H = P + 4 ∧ (∃ k : ℕ, k = (3 * P) / 10 ∧ k = 16 * (3 * P) / (16 * 10) ∧ H + P = 36) :=
sorry

end min_combined_number_of_horses_and_ponies_l1237_123736


namespace symmetric_points_origin_l1237_123787

theorem symmetric_points_origin {a b : ℝ} (h₁ : a = -(-4)) (h₂ : b = -(3)) : a - b = 7 :=
by 
  -- since this is a statement template, the proof is omitted
  sorry

end symmetric_points_origin_l1237_123787


namespace moli_initial_payment_l1237_123771

variable (R C S M : ℕ)

-- Conditions
def condition1 : Prop := 3 * R + 7 * C + 1 * S = M
def condition2 : Prop := 4 * R + 10 * C + 1 * S = 164
def condition3 : Prop := 1 * R + 1 * C + 1 * S = 32

theorem moli_initial_payment : condition1 R C S M ∧ condition2 R C S ∧ condition3 R C S → M = 120 := by
  sorry

end moli_initial_payment_l1237_123771


namespace count_possible_P_l1237_123706

-- Define the distinct digits with initial conditions
def digits : Type := {n // n ≥ 0 ∧ n ≤ 9}

-- Define the parameters P, Q, R, S as distinct digits
variables (P Q R S : digits)

-- Define the condition that P, Q, R, S are distinct.
def distinct (P Q R S : digits) : Prop := 
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S

-- Assertion conditions based on a valid subtraction layout
def valid_subtraction (P Q R S : digits) : Prop :=
  Q.val - P.val = S.val ∧ (P.val - R.val = P.val) ∧ (P.val - Q.val = S.val)

-- Prove that there are exactly 9 possible values for P.
theorem count_possible_P : ∃ n : ℕ, n = 9 ∧ ∀ P Q R S : digits, distinct P Q R S → valid_subtraction P Q R S → n = 9 :=
by sorry

end count_possible_P_l1237_123706


namespace ellipse_major_axis_length_l1237_123782

-- Given conditions
variable (radius : ℝ) (h_radius : radius = 2)
variable (minor_axis : ℝ) (h_minor_axis : minor_axis = 2 * radius)
variable (major_axis : ℝ) (h_major_axis : major_axis = 1.4 * minor_axis)

-- Proof problem statement
theorem ellipse_major_axis_length : major_axis = 5.6 :=
by
  sorry

end ellipse_major_axis_length_l1237_123782


namespace cubic_inequality_l1237_123717

theorem cubic_inequality (p q x : ℝ) (h : x^3 + p * x + q = 0) : 4 * q * x ≤ p^2 :=
sorry

end cubic_inequality_l1237_123717


namespace train_speed_km_per_hr_l1237_123789

theorem train_speed_km_per_hr 
  (length : ℝ) 
  (time : ℝ) 
  (h_length : length = 150) 
  (h_time : time = 9.99920006399488) : 
  length / time * 3.6 = 54.00287976961843 :=
by
  sorry

end train_speed_km_per_hr_l1237_123789


namespace smaller_screen_diagonal_l1237_123756

/-- The area of a 20-inch square screen is 38 square inches greater than the area
    of a smaller square screen. Prove that the length of the diagonal of the smaller screen is 18 inches. -/
theorem smaller_screen_diagonal (x : ℝ) (d : ℝ) (A₁ A₂ : ℝ)
  (h₀ : d = x * Real.sqrt 2)
  (h₁ : A₁ = 20 * Real.sqrt 2 * 20 * Real.sqrt 2)
  (h₂ : A₂ = x * x)
  (h₃ : A₁ = A₂ + 38) :
  d = 18 :=
by
  sorry

end smaller_screen_diagonal_l1237_123756


namespace marissa_tied_boxes_l1237_123744

def Total_ribbon : ℝ := 4.5
def Leftover_ribbon : ℝ := 1
def Ribbon_per_box : ℝ := 0.7

theorem marissa_tied_boxes : (Total_ribbon - Leftover_ribbon) / Ribbon_per_box = 5 := by
  sorry

end marissa_tied_boxes_l1237_123744


namespace circle_radius_l1237_123750

theorem circle_radius : ∀ (x y : ℝ), x^2 + 10*x + y^2 - 8*y + 25 = 0 → False := sorry

end circle_radius_l1237_123750


namespace projectile_height_at_time_l1237_123731

theorem projectile_height_at_time
  (y : ℝ)
  (t : ℝ)
  (h_eq : y = -16 * t ^ 2 + 64 * t) :
  ∃ t₀ : ℝ, t₀ = 3 ∧ y = 49 :=
by sorry

end projectile_height_at_time_l1237_123731


namespace choose_4_out_of_10_l1237_123753

theorem choose_4_out_of_10 : Nat.choose 10 4 = 210 := by
  sorry

end choose_4_out_of_10_l1237_123753


namespace largest_part_of_proportional_division_l1237_123777

theorem largest_part_of_proportional_division (sum : ℚ) (a b c largest : ℚ) 
  (prop1 prop2 prop3 : ℚ) 
  (h1 : sum = 156)
  (h2 : prop1 = 2)
  (h3 : prop2 = 1 / 2)
  (h4 : prop3 = 1 / 4)
  (h5 : sum = a + b + c)
  (h6 : a / prop1 = b / prop2 ∧ b / prop2 = c / prop3)
  (h7 : largest = max a (max b c)) :
  largest = 112 + 8 / 11 :=
by
  sorry

end largest_part_of_proportional_division_l1237_123777


namespace evaluate_expression_l1237_123716

theorem evaluate_expression : 
  let expr := (15 / 8) ^ 2
  let ceil_expr := Nat.ceil expr
  let mult_expr := ceil_expr * (21 / 5)
  Nat.floor mult_expr = 16 := by
  sorry

end evaluate_expression_l1237_123716


namespace walking_ring_width_l1237_123709

theorem walking_ring_width (r₁ r₂ : ℝ) (h : 2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 20 * Real.pi) :
  r₁ - r₂ = 10 :=
by
  sorry

end walking_ring_width_l1237_123709


namespace afternoon_to_morning_ratio_l1237_123797

theorem afternoon_to_morning_ratio
  (A : ℕ) (M : ℕ)
  (h1 : A = 340)
  (h2 : A + M = 510) :
  A / M = 2 :=
by
  sorry

end afternoon_to_morning_ratio_l1237_123797


namespace trapezoid_problem_l1237_123723

theorem trapezoid_problem (b h x : ℝ) 
  (hb : b > 0)
  (hh : h > 0)
  (h_ratio : (b + 90) / (b + 30) = 3 / 4)
  (h_x_def : x = 150 * (h / (x - 90) - 90))
  (hx2 : x^2 = 26100) :
  ⌊x^2 / 120⌋ = 217 := sorry

end trapezoid_problem_l1237_123723


namespace inequality_proof_l1237_123722

variable {A B C a b c r : ℝ}

theorem inequality_proof (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hr : 0 < r) :
  (A + a + B + b) / (A + a + B + b + c + r) + (B + b + C + c) / (B + b + C + c + a + r) > (C + c + A + a) / (C + c + A + a + b + r) := 
    sorry

end inequality_proof_l1237_123722


namespace male_salmon_count_l1237_123766

theorem male_salmon_count (total_salmon : ℕ) (female_salmon : ℕ) (male_salmon : ℕ) 
  (h1 : total_salmon = 971639) 
  (h2 : female_salmon = 259378) 
  (h3 : male_salmon = total_salmon - female_salmon) : 
  male_salmon = 712261 :=
by
  sorry

end male_salmon_count_l1237_123766


namespace brownies_pieces_l1237_123742

theorem brownies_pieces (pan_length pan_width piece_length piece_width : ℕ)
  (h_pan_dims : pan_length = 15) (h_pan_width : pan_width = 25)
  (h_piece_length : piece_length = 3) (h_piece_width : piece_width = 5) :
  (pan_length * pan_width) / (piece_length * piece_width) = 25 :=
by
  sorry

end brownies_pieces_l1237_123742
