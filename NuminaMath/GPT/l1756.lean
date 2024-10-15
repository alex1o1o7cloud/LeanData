import Mathlib

namespace NUMINAMATH_GPT_nina_money_l1756_175601

theorem nina_money (W : ℝ) (P: ℝ) (Q : ℝ) 
  (h1 : P = 6 * W)
  (h2 : Q = 8 * (W - 1))
  (h3 : P = Q) 
  : P = 24 := 
by 
  sorry

end NUMINAMATH_GPT_nina_money_l1756_175601


namespace NUMINAMATH_GPT_distinct_real_roots_a1_l1756_175613

theorem distinct_real_roots_a1 {x : ℝ} :
  ∀ a : ℝ, a = 1 →
    ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a * x1^2 + (1 - a) * x1 - 1 = 0) ∧ (a * x2^2 + (1 - a) * x2 - 1 = 0) :=
by sorry

end NUMINAMATH_GPT_distinct_real_roots_a1_l1756_175613


namespace NUMINAMATH_GPT_avg_visitors_on_sundays_l1756_175696

theorem avg_visitors_on_sundays (avg_other_days : ℕ) (avg_month : ℕ) (days_in_month sundays other_days : ℕ) (total_month_visitors : ℕ) (total_other_days_visitors : ℕ) (S : ℕ):
  avg_other_days = 240 →
  avg_month = 285 →
  days_in_month = 30 →
  sundays = 5 →
  other_days = 25 →
  total_month_visitors = avg_month * days_in_month →
  total_other_days_visitors = avg_other_days * other_days →
  5 * S + total_other_days_visitors = total_month_visitors →
  S = 510 :=
by
  intros _
          _
          _
          _
          _
          _
          _
          h
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_avg_visitors_on_sundays_l1756_175696


namespace NUMINAMATH_GPT_prize_distribution_l1756_175636

theorem prize_distribution : 
  ∃ (n1 n2 n3 : ℕ), -- The number of 1st, 2nd, and 3rd prize winners
  n1 + n2 + n3 = 7 ∧ -- Total number of winners is 7
  n1 * 800 + n2 * 700 + n3 * 300 = 4200 ∧ -- Total prize money distributed is $4200
  n1 = 1 ∧ -- Number of 1st prize winners
  n2 = 4 ∧ -- Number of 2nd prize winners
  n3 = 2 -- Number of 3rd prize winners
:= sorry

end NUMINAMATH_GPT_prize_distribution_l1756_175636


namespace NUMINAMATH_GPT_circle_line_distance_difference_l1756_175645

/-- We define the given circle and line and prove the difference between maximum and minimum distances
    from any point on the circle to the line is 5√2. -/
theorem circle_line_distance_difference :
  (∀ (x y : ℝ), x^2 + y^2 - 4*x - 4*y - 10 = 0) →
  (∀ (x y : ℝ), x + y - 8 = 0) →
  ∃ (d : ℝ), d = 5 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_circle_line_distance_difference_l1756_175645


namespace NUMINAMATH_GPT_largest_c_for_range_of_f_l1756_175699

def has_real_roots (a b c : ℝ) : Prop :=
  b * b - 4 * a * c ≥ 0

theorem largest_c_for_range_of_f (c : ℝ) :
  (∃ x : ℝ, x^2 + 3 * x + c = 7) ↔ c ≤ 37 / 4 := by
  sorry

end NUMINAMATH_GPT_largest_c_for_range_of_f_l1756_175699


namespace NUMINAMATH_GPT_range_of_m_l1756_175621

def p (m : ℝ) : Prop :=
  ∀ (x : ℝ), 2^x - m + 1 > 0

def q (m : ℝ) : Prop :=
  5 - 2 * m > 1

theorem range_of_m (m : ℝ) (hpq : p m ∧ q m) : m ≤ 1 := sorry

end NUMINAMATH_GPT_range_of_m_l1756_175621


namespace NUMINAMATH_GPT_abc_inequality_l1756_175615

theorem abc_inequality 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h_abc : a * b * c = 1) : 
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := 
by
  sorry

end NUMINAMATH_GPT_abc_inequality_l1756_175615


namespace NUMINAMATH_GPT_circle_area_l1756_175684

-- Condition: Given the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 10 * x + 4 * y + 20 = 0

-- Theorem: The area enclosed by the given circle equation is 9π
theorem circle_area : ∀ x y : ℝ, circle_eq x y → ∃ A : ℝ, A = 9 * Real.pi :=
by
  intros
  sorry

end NUMINAMATH_GPT_circle_area_l1756_175684


namespace NUMINAMATH_GPT_positive_divisors_3k1_ge_3k_minus_1_l1756_175660

theorem positive_divisors_3k1_ge_3k_minus_1 (n : ℕ) (h : 0 < n) :
  (∃ k : ℕ, (3 * k + 1) ∣ n) → (∃ k : ℕ, ¬ (3 * k - 1) ∣ n) :=
  sorry

end NUMINAMATH_GPT_positive_divisors_3k1_ge_3k_minus_1_l1756_175660


namespace NUMINAMATH_GPT_not_perfect_square_for_n_greater_than_11_l1756_175638

theorem not_perfect_square_for_n_greater_than_11 (n : ℤ) (h1 : n > 11) :
  ∀ m : ℤ, n^2 - 19 * n + 89 ≠ m^2 :=
sorry

end NUMINAMATH_GPT_not_perfect_square_for_n_greater_than_11_l1756_175638


namespace NUMINAMATH_GPT_busy_squirrels_count_l1756_175678

variable (B : ℕ)
variable (busy_squirrel_nuts_per_day : ℕ := 30)
variable (sleepy_squirrel_nuts_per_day : ℕ := 20)
variable (days : ℕ := 40)
variable (total_nuts : ℕ := 3200)

theorem busy_squirrels_count : busy_squirrel_nuts_per_day * days * B + sleepy_squirrel_nuts_per_day * days = total_nuts → B = 2 := by
  sorry

end NUMINAMATH_GPT_busy_squirrels_count_l1756_175678


namespace NUMINAMATH_GPT_square_area_inscribed_in_parabola_l1756_175679

def parabola (x : ℝ) : ℝ := x^2 - 10*x + 21

theorem square_area_inscribed_in_parabola :
  ∃ s : ℝ, s = (-1 + Real.sqrt 5) ∧ (2 * s)^2 = 24 - 8 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_square_area_inscribed_in_parabola_l1756_175679


namespace NUMINAMATH_GPT_tic_tac_toe_tie_probability_l1756_175633

theorem tic_tac_toe_tie_probability (john_wins martha_wins : ℚ) 
  (hj : john_wins = 4 / 9) 
  (hm : martha_wins = 5 / 12) : 
  1 - (john_wins + martha_wins) = 5 / 36 := 
by {
  /- insert proof here -/
  sorry
}

end NUMINAMATH_GPT_tic_tac_toe_tie_probability_l1756_175633


namespace NUMINAMATH_GPT_non_sophomores_is_75_percent_l1756_175606

def students_not_sophomores_percentage (total_students : ℕ) 
                                       (percent_juniors : ℚ)
                                       (num_seniors : ℕ)
                                       (freshmen_more_than_sophomores : ℕ) : ℚ :=
  let num_juniors := total_students * percent_juniors 
  let s := (total_students - num_juniors - num_seniors - freshmen_more_than_sophomores) / 2
  let f := s + freshmen_more_than_sophomores
  let non_sophomores := total_students - s
  (non_sophomores / total_students) * 100

theorem non_sophomores_is_75_percent : students_not_sophomores_percentage 800 0.28 160 16 = 75 := by
  sorry

end NUMINAMATH_GPT_non_sophomores_is_75_percent_l1756_175606


namespace NUMINAMATH_GPT_find_k_series_sum_l1756_175689

theorem find_k_series_sum :
  (∃ k : ℝ, 5 + ∑' n : ℕ, ((5 + (n + 1) * k) / 5^n.succ) = 10) →
  k = 12 :=
sorry

end NUMINAMATH_GPT_find_k_series_sum_l1756_175689


namespace NUMINAMATH_GPT_same_sign_iff_product_positive_different_sign_iff_product_negative_l1756_175622

variable (a b : ℝ)

theorem same_sign_iff_product_positive :
  ((a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0)) ↔ (a * b > 0) :=
sorry

theorem different_sign_iff_product_negative :
  ((a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0)) ↔ (a * b < 0) :=
sorry

end NUMINAMATH_GPT_same_sign_iff_product_positive_different_sign_iff_product_negative_l1756_175622


namespace NUMINAMATH_GPT_determine_real_pairs_l1756_175687

theorem determine_real_pairs (a b : ℝ) :
  (∀ n : ℕ+, a * ⌊ b * n ⌋ = b * ⌊ a * n ⌋) →
  (∃ c : ℝ, (a = 0 ∧ b = c) ∨ (a = c ∧ b = 0) ∨ (a = c ∧ b = c) ∨ (∃ k l : ℤ, a = k ∧ b = l)) :=
by
  sorry

end NUMINAMATH_GPT_determine_real_pairs_l1756_175687


namespace NUMINAMATH_GPT_ratio_between_house_and_park_l1756_175603

theorem ratio_between_house_and_park (w x y : ℝ) (hw : w > 0) (hx : x > 0) (hy : y > 0)
    (h : y / w = x / w + (x + y) / (10 * w)) : x / y = 9 / 11 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_between_house_and_park_l1756_175603


namespace NUMINAMATH_GPT_find_smaller_number_l1756_175644

theorem find_smaller_number (a b : ℕ) (h1 : a + b = 18) (h2 : a * b = 45) : a = 3 ∨ b = 3 := 
by 
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_find_smaller_number_l1756_175644


namespace NUMINAMATH_GPT_range_of_3x_plus_2y_l1756_175683

theorem range_of_3x_plus_2y (x y : ℝ) (hx : 1 ≤ x ∧ x ≤ 3) (hy : -1 ≤ y ∧ y ≤ 4) :
  1 ≤ 3 * x + 2 * y ∧ 3 * x + 2 * y ≤ 17 :=
sorry

end NUMINAMATH_GPT_range_of_3x_plus_2y_l1756_175683


namespace NUMINAMATH_GPT_number_of_cakes_sold_l1756_175654

-- Definitions based on the conditions provided
def cakes_made : ℕ := 173
def cakes_bought : ℕ := 103
def cakes_left : ℕ := 190

-- Calculate the initial total number of cakes
def initial_cakes : ℕ := cakes_made + cakes_bought

-- Calculate the number of cakes sold
def cakes_sold : ℕ := initial_cakes - cakes_left

-- The proof statement
theorem number_of_cakes_sold : cakes_sold = 86 :=
by
  unfold cakes_sold initial_cakes cakes_left cakes_bought cakes_made
  rfl

end NUMINAMATH_GPT_number_of_cakes_sold_l1756_175654


namespace NUMINAMATH_GPT_maria_original_number_25_3_l1756_175607

theorem maria_original_number_25_3 (x : ℚ) 
  (h : ((3 * (x + 3) - 4) / 3) = 10) : 
  x = 25 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_maria_original_number_25_3_l1756_175607


namespace NUMINAMATH_GPT_combined_age_of_siblings_l1756_175617

theorem combined_age_of_siblings (a s h : ℕ) (h1 : a = 15) (h2 : s = 3 * a) (h3 : h = 4 * s) : a + s + h = 240 :=
by
  sorry

end NUMINAMATH_GPT_combined_age_of_siblings_l1756_175617


namespace NUMINAMATH_GPT_M_subset_P_l1756_175643

universe u

-- Definitions of the sets
def U : Set ℝ := {x : ℝ | True}
def M : Set ℝ := {x : ℝ | x > 1}
def P : Set ℝ := {x : ℝ | x^2 > 1}

-- Proof statement
theorem M_subset_P : M ⊆ P := by
  sorry

end NUMINAMATH_GPT_M_subset_P_l1756_175643


namespace NUMINAMATH_GPT_scientific_notation_conversion_l1756_175648

theorem scientific_notation_conversion :
  216000 = 2.16 * 10^5 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_conversion_l1756_175648


namespace NUMINAMATH_GPT_four_digit_div_90_count_l1756_175631

theorem four_digit_div_90_count :
  ∃ n : ℕ, n = 10 ∧ (∀ (ab : ℕ), 10 ≤ ab ∧ ab ≤ 99 → ab % 9 = 0 → 
  (10 * ab + 90) % 90 = 0 ∧ 1000 ≤ 10 * ab + 90 ∧ 10 * ab + 90 < 10000) :=
sorry

end NUMINAMATH_GPT_four_digit_div_90_count_l1756_175631


namespace NUMINAMATH_GPT_solve_diff_eq_l1756_175691

def solution_of_diff_eq (x y : ℝ) (y' : ℝ → ℝ) : Prop :=
  (x + y) * y' x = 1

def initial_condition (y x : ℝ) : Prop :=
  y = 0 ∧ x = -1

theorem solve_diff_eq (x : ℝ) (y : ℝ) (y' : ℝ → ℝ) (h1 : initial_condition y x) (h2 : solution_of_diff_eq x y y') :
  y = -(x + 1) :=
by 
  sorry

end NUMINAMATH_GPT_solve_diff_eq_l1756_175691


namespace NUMINAMATH_GPT_treasure_chest_l1756_175675

theorem treasure_chest (n : ℕ) 
  (h1 : n % 8 = 2)
  (h2 : n % 7 = 6)
  (h3 : ∀ m : ℕ, (m % 8 = 2 → m % 7 = 6 → m ≥ n)) :
  n % 9 = 7 :=
sorry

end NUMINAMATH_GPT_treasure_chest_l1756_175675


namespace NUMINAMATH_GPT_solution_set_for_inequality_l1756_175618

theorem solution_set_for_inequality :
  {x : ℝ | -x^2 + 4 * x + 5 < 0} = {x : ℝ | x > 5 ∨ x < -1} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_for_inequality_l1756_175618


namespace NUMINAMATH_GPT_find_x_l1756_175682

theorem find_x (x : ℝ) (h : 65 + 5 * 12 / (x / 3) = 66) : x = 180 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1756_175682


namespace NUMINAMATH_GPT_proof_negation_l1756_175652

-- Definitions of rational and real numbers
def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b
def is_irrational (x : ℝ) : Prop := ¬ is_rational x

-- Proposition stating the existence of an irrational number that is rational
def original_proposition : Prop :=
  ∃ x : ℝ, is_irrational x ∧ is_rational x

-- Negation of the original proposition
def negated_proposition : Prop :=
  ∀ x : ℝ, is_irrational x → ¬ is_rational x

theorem proof_negation : ¬ original_proposition = negated_proposition := 
sorry

end NUMINAMATH_GPT_proof_negation_l1756_175652


namespace NUMINAMATH_GPT_min_flowers_for_bouquets_l1756_175624

open Classical

noncomputable def minimum_flowers (types : ℕ) (flowers_for_bouquet : ℕ) (bouquets : ℕ) : ℕ := 
  sorry

theorem min_flowers_for_bouquets : minimum_flowers 6 5 10 = 70 := 
  sorry

end NUMINAMATH_GPT_min_flowers_for_bouquets_l1756_175624


namespace NUMINAMATH_GPT_work_done_on_gas_in_process_1_2_l1756_175616

variables (V₁ V₂ V₃ V₄ A₁₂ A₃₄ T n R : ℝ)

-- Both processes 1-2 and 3-4 are isothermal.
def is_isothermal_process := true -- Placeholder

-- Volumes relationship: for any given pressure, the volume in process 1-2 is exactly twice the volume in process 3-4.
def volumes_relation (V₁ V₂ V₃ V₄ : ℝ) : Prop :=
  V₁ = 2 * V₃ ∧ V₂ = 2 * V₄

-- Work done on a gas during an isothermal process can be represented as: A = 2 * A₃₄
def work_relation (A₁₂ A₃₄ : ℝ) : Prop :=
  A₁₂ = 2 * A₃₄

theorem work_done_on_gas_in_process_1_2
  (h_iso : is_isothermal_process)
  (h_vol : volumes_relation V₁ V₂ V₃ V₄)
  (h_work : work_relation A₁₂ A₃₄) :
  A₁₂ = 2 * A₃₄ :=
by 
  sorry

end NUMINAMATH_GPT_work_done_on_gas_in_process_1_2_l1756_175616


namespace NUMINAMATH_GPT_measure_of_angle_XPM_l1756_175663

-- Definitions based on given conditions
variables (X Y Z L M N P : Type)
variables (a b c : ℝ) -- Angles are represented in degrees
variables [DecidableEq X] [DecidableEq Y] [DecidableEq Z]

-- Triangle XYZ with angle bisectors XL, YM, and ZN meeting at incenter P
-- Given angle XYZ in degrees
def angle_XYZ : ℝ := 46

-- Incenter angle properties
axiom angle_bisector_XL (angle_XYP : ℝ) : angle_XYP = angle_XYZ / 2
axiom angle_bisector_YM (angle_YXP : ℝ) : ∃ (angle_YXZ : ℝ), angle_YXP = angle_YXZ / 2

-- The proposition we need to prove
theorem measure_of_angle_XPM : ∃ (angle_XPM : ℝ), angle_XPM = 67 := 
by {
  sorry
}

end NUMINAMATH_GPT_measure_of_angle_XPM_l1756_175663


namespace NUMINAMATH_GPT_find_function_l1756_175639

def satisfies_condition (f : ℕ+ → ℕ+) :=
  ∀ a b : ℕ+, f a + b ∣ a^2 + f a * f b

theorem find_function :
  ∀ f : ℕ+ → ℕ+, satisfies_condition f → (∀ a : ℕ+, f a = a) :=
by
  intros f h
  sorry

end NUMINAMATH_GPT_find_function_l1756_175639


namespace NUMINAMATH_GPT_tan_plus_pi_over_4_l1756_175637

variable (θ : ℝ)

-- Define the conditions
def condition_θ_interval : Prop := θ ∈ Set.Ioo (Real.pi / 2) Real.pi
def condition_sin_θ : Prop := Real.sin θ = 3 / 5

-- Define the theorem to be proved
theorem tan_plus_pi_over_4 (h1 : condition_θ_interval θ) (h2 : condition_sin_θ θ) :
  Real.tan (θ + Real.pi / 4) = 7 :=
sorry

end NUMINAMATH_GPT_tan_plus_pi_over_4_l1756_175637


namespace NUMINAMATH_GPT_k_value_l1756_175623

theorem k_value (k : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → |k * x - 4| ≤ 2) → k = 2 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_k_value_l1756_175623


namespace NUMINAMATH_GPT_length_of_paving_stone_l1756_175672

theorem length_of_paving_stone (courtyard_length courtyard_width : ℝ)
  (num_paving_stones : ℕ) (paving_stone_width : ℝ) (total_area : ℝ)
  (paving_stone_length : ℝ) : 
  courtyard_length = 70 ∧ courtyard_width = 16.5 ∧ num_paving_stones = 231 ∧ paving_stone_width = 2 ∧ total_area = courtyard_length * courtyard_width ∧ total_area = num_paving_stones * paving_stone_length * paving_stone_width → 
  paving_stone_length = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_length_of_paving_stone_l1756_175672


namespace NUMINAMATH_GPT_triangle_inscribed_circle_area_l1756_175604

noncomputable def circle_radius (circumference : ℝ) : ℝ :=
  circumference / (2 * Real.pi)

noncomputable def triangle_area (r : ℝ) : ℝ :=
  (1 / 2) * r^2 * (Real.sin (Real.pi / 2) + Real.sin (2 * Real.pi / 3) + Real.sin (5 * Real.pi / 6))

theorem triangle_inscribed_circle_area (a b c : ℝ) (h : a + b + c = 24) :
  ∀ (r : ℝ) (h_r : r = circle_radius 24),
  triangle_area r = 72 / Real.pi^2 * (Real.sqrt 3 + 1) :=
by
  intro r h_r
  rw [h_r, circle_radius, triangle_area]
  sorry

end NUMINAMATH_GPT_triangle_inscribed_circle_area_l1756_175604


namespace NUMINAMATH_GPT_simplify_polynomial_l1756_175651

theorem simplify_polynomial (y : ℝ) :
  (3 * y - 2) * (5 * y ^ 12 + 3 * y ^ 11 + y ^ 10 + 2 * y ^ 9) =
  15 * y ^ 13 - y ^ 12 - 3 * y ^ 11 + 4 * y ^ 10 - 4 * y ^ 9 := 
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l1756_175651


namespace NUMINAMATH_GPT_petes_original_number_l1756_175665

theorem petes_original_number (x : ℤ) (h : 5 * (3 * x - 6) = 195) : x = 15 :=
sorry

end NUMINAMATH_GPT_petes_original_number_l1756_175665


namespace NUMINAMATH_GPT_N_subset_M_l1756_175625

open Set

def M : Set (ℝ × ℝ) := { p | ∃ x, p = (x, 2*x + 1) }
def N : Set (ℝ × ℝ) := { p | ∃ x, p = (x, -x^2) }

theorem N_subset_M : N ⊆ M :=
by
  sorry

end NUMINAMATH_GPT_N_subset_M_l1756_175625


namespace NUMINAMATH_GPT_inequality_proof_l1756_175620

theorem inequality_proof (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a > 0) (h3 : b > 0) (h4 : c > 0) :
  (a / (2 * a + 1) + b / (3 * b + 1) + c / (6 * c + 1)) ≤ 1 / 2 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1756_175620


namespace NUMINAMATH_GPT_linear_equation_solution_l1756_175677

theorem linear_equation_solution (a : ℝ) (x y : ℝ) 
    (h : (a - 2) * x^(|a| - 1) + 3 * y = 1) 
    (h1 : ∀ (x y : ℝ), (a - 2) ≠ 0)
    (h2 : |a| - 1 = 1) : a = -2 :=
by
  sorry

end NUMINAMATH_GPT_linear_equation_solution_l1756_175677


namespace NUMINAMATH_GPT_part_a_part_b_l1756_175668

theorem part_a (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : 16 * x * y * z = (x + y)^2 * (x + z)^2) : x + y + z ≤ 4 :=
sorry

theorem part_b : ∃ (S : Set (ℚ × ℚ × ℚ)), S.Countable ∧
  (∀ (x y z : ℚ), (x, y, z) ∈ S → 0 < x ∧ 0 < y ∧ 0 < z ∧ 16 * x * y * z = (x + y)^2 * (x + z)^2 ∧ x + y + z = 4) ∧ 
  Infinite S :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l1756_175668


namespace NUMINAMATH_GPT_cost_of_each_nose_spray_l1756_175674

def total_nose_sprays : ℕ := 10
def total_cost : ℝ := 15
def buy_one_get_one_free : Bool := true

theorem cost_of_each_nose_spray :
  buy_one_get_one_free = true →
  total_nose_sprays = 10 →
  total_cost = 15 →
  (total_cost / (total_nose_sprays / 2)) = 3 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_cost_of_each_nose_spray_l1756_175674


namespace NUMINAMATH_GPT_unique_solution_exists_l1756_175605

theorem unique_solution_exists (a x y z : ℝ) 
  (h1 : z = a * (x + 2 * y + 5 / 2)) 
  (h2 : x^2 + y^2 + 2 * x - y + a * (x + 2 * y + 5 / 2) = 0) :
  a = 1 → x = -3 / 2 ∧ y = -1 / 2 ∧ z = 0 := 
by
  sorry

end NUMINAMATH_GPT_unique_solution_exists_l1756_175605


namespace NUMINAMATH_GPT_remaining_money_l1756_175659

-- Defining the conditions
def orchids_qty := 20
def price_per_orchid := 50
def chinese_money_plants_qty := 15
def price_per_plant := 25
def worker_qty := 2
def salary_per_worker := 40
def cost_of_pots := 150

-- Earnings and expenses calculations
def earnings_from_orchids := orchids_qty * price_per_orchid
def earnings_from_plants := chinese_money_plants_qty * price_per_plant
def total_earnings := earnings_from_orchids + earnings_from_plants

def worker_expenses := worker_qty * salary_per_worker
def total_expenses := worker_expenses + cost_of_pots

-- The proof problem
theorem remaining_money : total_earnings - total_expenses = 1145 :=
by
  sorry

end NUMINAMATH_GPT_remaining_money_l1756_175659


namespace NUMINAMATH_GPT_machine_parts_probabilities_l1756_175686

-- Define the yield rates for the two machines
def yield_rate_A : ℝ := 0.8
def yield_rate_B : ℝ := 0.9

-- Define the probabilities of defectiveness for each machine
def defective_probability_A := 1 - yield_rate_A
def defective_probability_B := 1 - yield_rate_B

theorem machine_parts_probabilities :
  (defective_probability_A * defective_probability_B = 0.02) ∧
  (((yield_rate_A * defective_probability_B) + (defective_probability_A * yield_rate_B)) = 0.26) ∧
  (defective_probability_A * defective_probability_B + (1 - (defective_probability_A * defective_probability_B)) = 1) :=
by
  sorry

end NUMINAMATH_GPT_machine_parts_probabilities_l1756_175686


namespace NUMINAMATH_GPT_smallest_four_digit_palindrome_div7_eq_1661_l1756_175695

theorem smallest_four_digit_palindrome_div7_eq_1661 :
  ∃ (A B : ℕ), (A == 1 ∨ A == 3 ∨ A == 5 ∨ A == 7 ∨ A == 9) ∧
  (1000 ≤ 1100 * A + 11 * B ∧ 1100 * A + 11 * B < 10000) ∧
  (1100 * A + 11 * B) % 7 = 0 ∧
  (1100 * A + 11 * B) = 1661 :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_palindrome_div7_eq_1661_l1756_175695


namespace NUMINAMATH_GPT_principal_amount_l1756_175626

theorem principal_amount (P : ℝ) (SI : ℝ) (R : ℝ) (T : ℝ)
  (h1 : SI = (P * R * T) / 100)
  (h2 : SI = 640)
  (h3 : R = 8)
  (h4 : T = 2) :
  P = 4000 :=
sorry

end NUMINAMATH_GPT_principal_amount_l1756_175626


namespace NUMINAMATH_GPT_smallest_x_for_multiple_l1756_175697

theorem smallest_x_for_multiple 
  (x : ℕ) (h₁ : ∀ m : ℕ, 450 * x = 800 * m) 
  (h₂ : ∀ y : ℕ, (∀ m : ℕ, 450 * y = 800 * m) → x ≤ y) : 
  x = 16 := 
sorry

end NUMINAMATH_GPT_smallest_x_for_multiple_l1756_175697


namespace NUMINAMATH_GPT_find_m_values_l1756_175692

theorem find_m_values (m : ℝ) : 
  (∃ A B : ℝ × ℝ, A = (2, 2) ∧ B = (m, 0) ∧ 
   ∃ r R : ℝ, r = 1 ∧ R = 3 ∧ 
   ∃ d : ℝ, d = abs (dist A B) ∧ d = (R + r)) →
  (m = 2 - 2 * Real.sqrt 3 ∨ m = 2 + 2 * Real.sqrt 3) := 
sorry

end NUMINAMATH_GPT_find_m_values_l1756_175692


namespace NUMINAMATH_GPT_find_c_l1756_175667

theorem find_c (a b c : ℤ) (h1 : a + b * c = 2017) (h2 : b + c * a = 8) :
  c = -6 ∨ c = 0 ∨ c = 2 ∨ c = 8 :=
by 
  sorry

end NUMINAMATH_GPT_find_c_l1756_175667


namespace NUMINAMATH_GPT_Jessie_lost_7_kilograms_l1756_175642

def Jessie_previous_weight : ℕ := 74
def Jessie_current_weight : ℕ := 67
def Jessie_weight_lost : ℕ := Jessie_previous_weight - Jessie_current_weight

theorem Jessie_lost_7_kilograms : Jessie_weight_lost = 7 :=
by
  sorry

end NUMINAMATH_GPT_Jessie_lost_7_kilograms_l1756_175642


namespace NUMINAMATH_GPT_dune_buggy_speed_l1756_175610

theorem dune_buggy_speed (S : ℝ) :
  (1/3 * S + 1/3 * (S + 12) + 1/3 * (S - 18) = 58) → S = 60 :=
by
  sorry

end NUMINAMATH_GPT_dune_buggy_speed_l1756_175610


namespace NUMINAMATH_GPT_smallest_positive_integer_congruence_l1756_175698

theorem smallest_positive_integer_congruence :
  ∃ x : ℕ, 0 < x ∧ x < 17 ∧ (3 * x ≡ 14 [MOD 17]) := sorry

end NUMINAMATH_GPT_smallest_positive_integer_congruence_l1756_175698


namespace NUMINAMATH_GPT_christian_age_in_years_l1756_175650

theorem christian_age_in_years (B C x : ℕ) (h1 : C = 2 * B) (h2 : B + x = 40) (h3 : C + x = 72) :
    x = 8 := 
sorry

end NUMINAMATH_GPT_christian_age_in_years_l1756_175650


namespace NUMINAMATH_GPT_train_length_is_correct_l1756_175602

noncomputable def speed_km_per_hr := 60
noncomputable def time_seconds := 15
noncomputable def speed_m_per_s : ℝ := (60 * 1000) / 3600
noncomputable def expected_length : ℝ := 250.05

theorem train_length_is_correct : (speed_m_per_s * time_seconds) = expected_length := by
  sorry

end NUMINAMATH_GPT_train_length_is_correct_l1756_175602


namespace NUMINAMATH_GPT_john_finances_l1756_175685

theorem john_finances :
  let total_first_year := 10000
  let tuition_percent := 0.4
  let room_board_percent := 0.35
  let textbook_transport_percent := 0.25
  let tuition_increase := 0.06
  let room_board_increase := 0.03
  let aid_first_year := 0.25
  let aid_increase := 0.02

  let tuition_first_year := total_first_year * tuition_percent
  let room_board_first_year := total_first_year * room_board_percent
  let textbook_transport_first_year := total_first_year * textbook_transport_percent

  let tuition_second_year := tuition_first_year * (1 + tuition_increase)
  let room_board_second_year := room_board_first_year * (1 + room_board_increase)
  let financial_aid_second_year := tuition_second_year * (aid_first_year + aid_increase)

  let tuition_third_year := tuition_second_year * (1 + tuition_increase)
  let room_board_third_year := room_board_second_year * (1 + room_board_increase)
  let financial_aid_third_year := tuition_third_year * (aid_first_year + 2 * aid_increase)

  let total_cost_first_year := 
      (tuition_first_year - tuition_first_year * aid_first_year) +
      room_board_first_year + 
      textbook_transport_first_year

  let total_cost_second_year :=
      (tuition_second_year - financial_aid_second_year) +
      room_board_second_year +
      textbook_transport_first_year

  let total_cost_third_year :=
      (tuition_third_year - financial_aid_third_year) +
      room_board_third_year +
      textbook_transport_first_year

  total_cost_first_year = 9000 ∧
  total_cost_second_year = 9200.20 ∧
  total_cost_third_year = 9404.17 := 
by
  sorry

end NUMINAMATH_GPT_john_finances_l1756_175685


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1756_175680

theorem solution_set_of_inequality :
  {x : ℝ | |x + 1| - |x - 5| < 4} = {x : ℝ | x < 4} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1756_175680


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l1756_175628

-- Proof problem 1: Prove that under the condition 6x - 4 = 3x + 2, x = 2
theorem solve_eq1 : ∀ x : ℝ, 6 * x - 4 = 3 * x + 2 → x = 2 :=
by
  intro x
  intro h
  sorry

-- Proof problem 2: Prove that under the condition (x / 4) - (3 / 5) = (x + 1) / 2, x = -22/5
theorem solve_eq2 : ∀ x : ℝ, (x / 4) - (3 / 5) = (x + 1) / 2 → x = -(22 / 5) :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l1756_175628


namespace NUMINAMATH_GPT_carol_age_l1756_175649

theorem carol_age (B C : ℕ) (h1 : B + C = 66) (h2 : C = 3 * B + 2) : C = 50 :=
sorry

end NUMINAMATH_GPT_carol_age_l1756_175649


namespace NUMINAMATH_GPT_power_minus_self_even_l1756_175693

theorem power_minus_self_even (a n : ℕ) (ha : 0 < a) (hn : 0 < n) : Even (a^n - a) := by
  sorry

end NUMINAMATH_GPT_power_minus_self_even_l1756_175693


namespace NUMINAMATH_GPT_average_difference_l1756_175666

theorem average_difference (F1 L1 F2 L2 : ℤ) (H1 : F1 = 200) (H2 : L1 = 400) (H3 : F2 = 100) (H4 : L2 = 200) :
  (F1 + L1) / 2 - (F2 + L2) / 2 = 150 := 
by 
  sorry

end NUMINAMATH_GPT_average_difference_l1756_175666


namespace NUMINAMATH_GPT_problem_l1756_175619

-- Conditions
def a_n (n : ℕ) : ℚ := (1/3)^(n-1)

def b_n (n : ℕ) : ℚ := n * (1/3)^n

-- Sums over the first n terms
def S_n (n : ℕ) : ℚ := (3/2) - (1/2) * (1/3)^n

def T_n (n : ℕ) : ℚ := (3/4) - (1/4) * (1/3)^n - (n/2) * (1/3)^n

-- Problem: Prove T_n < S_n / 2
theorem problem (n : ℕ) : T_n n < S_n n / 2 :=
by sorry

end NUMINAMATH_GPT_problem_l1756_175619


namespace NUMINAMATH_GPT_probability_of_events_l1756_175612

noncomputable def total_types : ℕ := 8

noncomputable def fever_reducing_types : ℕ := 3

noncomputable def cough_suppressing_types : ℕ := 5

noncomputable def total_ways_to_choose_two : ℕ := Nat.choose total_types 2

noncomputable def event_A_ways : ℕ := total_ways_to_choose_two - Nat.choose cough_suppressing_types 2

noncomputable def P_A : ℚ := event_A_ways / total_ways_to_choose_two

noncomputable def event_B_ways : ℕ := fever_reducing_types * cough_suppressing_types

noncomputable def P_B_given_A : ℚ := event_B_ways / event_A_ways

theorem probability_of_events :
  P_A = 9 / 14 ∧ P_B_given_A = 5 / 6 := by
  sorry

end NUMINAMATH_GPT_probability_of_events_l1756_175612


namespace NUMINAMATH_GPT_david_reading_time_l1756_175641

theorem david_reading_time
  (total_time : ℕ)
  (math_time : ℕ)
  (spelling_time : ℕ)
  (reading_time : ℕ)
  (h1 : total_time = 60)
  (h2 : math_time = 15)
  (h3 : spelling_time = 18)
  (h4 : reading_time = total_time - (math_time + spelling_time)) :
  reading_time = 27 := 
by {
  sorry
}

end NUMINAMATH_GPT_david_reading_time_l1756_175641


namespace NUMINAMATH_GPT_expression_equals_one_l1756_175647

def evaluate_expression : ℕ :=
  3 * (3 * (3 * (3 * (3 - 2 * 1) - 2 * 1) - 2 * 1) - 2 * 1) - 2 * 1

theorem expression_equals_one : evaluate_expression = 1 := by
  sorry

end NUMINAMATH_GPT_expression_equals_one_l1756_175647


namespace NUMINAMATH_GPT_repayment_days_least_integer_l1756_175655

theorem repayment_days_least_integer:
  ∀ (x : ℤ), (20 + 2 * x ≥ 60) → (x ≥ 20) :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_repayment_days_least_integer_l1756_175655


namespace NUMINAMATH_GPT_percent_red_prob_l1756_175658

-- Define the conditions
def initial_red := 2
def initial_blue := 4
def additional_red := 2
def additional_blue := 2
def total_balloons := initial_red + initial_blue + additional_red + additional_blue
def total_red := initial_red + additional_red

-- State the theorem
theorem percent_red_prob : (total_red.toFloat / total_balloons.toFloat) * 100 = 40 :=
by
  sorry

end NUMINAMATH_GPT_percent_red_prob_l1756_175658


namespace NUMINAMATH_GPT_f_const_one_l1756_175671

-- Mathematical Translation of the Definitions
variable (f g h : ℕ → ℕ)

-- Given conditions
axiom h_injective : Function.Injective h
axiom g_surjective : Function.Surjective g
axiom f_eq : ∀ n, f n = g n - h n + 1

-- Theorem to Prove
theorem f_const_one : ∀ n, f n = 1 :=
by
  sorry

end NUMINAMATH_GPT_f_const_one_l1756_175671


namespace NUMINAMATH_GPT_students_transferred_l1756_175669

theorem students_transferred (initial_students : ℝ) (students_left : ℝ) (end_students : ℝ) :
  initial_students = 42.0 →
  students_left = 4.0 →
  end_students = 28.0 →
  initial_students - students_left - end_students = 10.0 :=
by
  intros
  sorry

end NUMINAMATH_GPT_students_transferred_l1756_175669


namespace NUMINAMATH_GPT_first_number_Harold_says_l1756_175676

/-
  Define each student's sequence of numbers.
  - Alice skips every 4th number.
  - Barbara says numbers that Alice didn't say, skipping every 4th in her sequence.
  - Subsequent students follow the same rule.
  - Harold picks the smallest prime number not said by any student.
-/

def is_skipped_by_Alice (n : Nat) : Prop :=
  n % 4 ≠ 0

def is_skipped_by_Barbara (n : Nat) : Prop :=
  is_skipped_by_Alice n ∧ (n / 4) % 4 ≠ 3

def is_skipped_by_Candice (n : Nat) : Prop :=
  is_skipped_by_Barbara n ∧ (n / 16) % 4 ≠ 3

def is_skipped_by_Debbie (n : Nat) : Prop :=
  is_skipped_by_Candice n ∧ (n / 64) % 4 ≠ 3

def is_skipped_by_Eliza (n : Nat) : Prop :=
  is_skipped_by_Debbie n ∧ (n / 256) % 4 ≠ 3

def is_skipped_by_Fatima (n : Nat) : Prop :=
  is_skipped_by_Eliza n ∧ (n / 1024) % 4 ≠ 3

def is_skipped_by_Grace (n : Nat) : Prop :=
  is_skipped_by_Fatima n

def is_skipped_by_anyone (n : Nat) : Prop :=
  ¬ is_skipped_by_Alice n ∨ ¬ is_skipped_by_Barbara n ∨ ¬ is_skipped_by_Candice n ∨
  ¬ is_skipped_by_Debbie n ∨ ¬ is_skipped_by_Eliza n ∨ ¬ is_skipped_by_Fatima n ∨
  ¬ is_skipped_by_Grace n

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ (m : Nat), m ∣ n → m = 1 ∨ m = n

theorem first_number_Harold_says : ∃ n : Nat, is_prime n ∧ ¬ is_skipped_by_anyone n ∧ n = 11 := by
  sorry

end NUMINAMATH_GPT_first_number_Harold_says_l1756_175676


namespace NUMINAMATH_GPT_road_construction_equation_l1756_175630

theorem road_construction_equation (x : ℝ) (hx : x > 0) :
  (9 / x) - (12 / (x + 1)) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_road_construction_equation_l1756_175630


namespace NUMINAMATH_GPT_minimum_yellow_marbles_l1756_175609

theorem minimum_yellow_marbles :
  ∀ (n y : ℕ), 
  (3 ∣ n) ∧ (4 ∣ n) ∧ 
  (9 + y + 2 * y ≤ n) ∧ 
  (n = n / 3 + n / 4 + 9 + y + 2 * y) → 
  y = 4 :=
by
  sorry

end NUMINAMATH_GPT_minimum_yellow_marbles_l1756_175609


namespace NUMINAMATH_GPT_polygon_eq_quadrilateral_l1756_175646

theorem polygon_eq_quadrilateral (n : ℕ) (h : (n - 2) * 180 = 360) : n = 4 := 
sorry

end NUMINAMATH_GPT_polygon_eq_quadrilateral_l1756_175646


namespace NUMINAMATH_GPT_find_desired_expression_l1756_175690

variable (y : ℝ)

theorem find_desired_expression
  (h : y + Real.sqrt (y^2 - 4) + (1 / (y - Real.sqrt (y^2 - 4))) = 12) :
  y^2 + Real.sqrt (y^4 - 4) + (1 / (y^2 - Real.sqrt (y^4 - 4))) = 200 / 9 :=
sorry

end NUMINAMATH_GPT_find_desired_expression_l1756_175690


namespace NUMINAMATH_GPT_dot_product_of_a_and_b_is_correct_l1756_175640

-- Define vectors a and b
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (2, -1)

-- Define dot product for ℝ × ℝ vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Theorem statement (proof can be omitted with sorry)
theorem dot_product_of_a_and_b_is_correct : dot_product a b = -4 :=
by
  -- proof goes here, omitted for now
  sorry

end NUMINAMATH_GPT_dot_product_of_a_and_b_is_correct_l1756_175640


namespace NUMINAMATH_GPT_lisa_flight_distance_l1756_175681

-- Define the given speed and time
def speed : ℝ := 32
def time : ℝ := 8

-- Define the distance formula
def distance (v : ℝ) (t : ℝ) : ℝ := v * t

-- State the theorem to be proved
theorem lisa_flight_distance : distance speed time = 256 := by
sorry

end NUMINAMATH_GPT_lisa_flight_distance_l1756_175681


namespace NUMINAMATH_GPT_largest_gcd_sum_1089_l1756_175611

theorem largest_gcd_sum_1089 (c d : ℕ) (h₁ : 0 < c) (h₂ : 0 < d) (h₃ : c + d = 1089) : ∃ k, k = Nat.gcd c d ∧ k = 363 :=
by
  sorry

end NUMINAMATH_GPT_largest_gcd_sum_1089_l1756_175611


namespace NUMINAMATH_GPT_intersection_M_N_l1756_175662

open Set

-- Definitions from conditions
def M : Set ℤ := {-1, 1, 2}
def N : Set ℤ := {x | x < 1}

-- Proof statement
theorem intersection_M_N : M ∩ N = {-1} := 
by sorry

end NUMINAMATH_GPT_intersection_M_N_l1756_175662


namespace NUMINAMATH_GPT_right_triangle_excircle_incircle_l1756_175661

theorem right_triangle_excircle_incircle (a b c r r_a : ℝ) (h : a^2 + b^2 = c^2) :
  (r = (a + b - c) / 2) → (r_a = (b + c - a) / 2) → r_a = 2 * r :=
by
  intros hr hra
  sorry

end NUMINAMATH_GPT_right_triangle_excircle_incircle_l1756_175661


namespace NUMINAMATH_GPT_tennis_tournament_cycle_l1756_175664

noncomputable def exists_cycle_of_three_players (P : Type) [Fintype P] (G : P → P → Bool) : Prop :=
  (∃ (a b c : P), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ G a b ∧ G b c ∧ G c a)

theorem tennis_tournament_cycle (P : Type) [Fintype P] (n : ℕ) (hp : 3 ≤ n) 
  (G : P → P → Bool) (H : ∀ a b : P, a ≠ b → (G a b ∨ G b a))
  (Hw : ∀ a : P, ∃ b : P, a ≠ b ∧ G a b) : exists_cycle_of_three_players P G :=
by 
  sorry

end NUMINAMATH_GPT_tennis_tournament_cycle_l1756_175664


namespace NUMINAMATH_GPT_coeffs_divisible_by_5_l1756_175614

theorem coeffs_divisible_by_5
  (a b c d : ℤ)
  (h1 : a + b + c + d ≡ 0 [ZMOD 5])
  (h2 : -a + b - c + d ≡ 0 [ZMOD 5])
  (h3 : 8 * a + 4 * b + 2 * c + d ≡ 0 [ZMOD 5])
  (h4 : d ≡ 0 [ZMOD 5]) :
  a ≡ 0 [ZMOD 5] ∧ b ≡ 0 [ZMOD 5] ∧ c ≡ 0 [ZMOD 5] ∧ d ≡ 0 [ZMOD 5] :=
sorry

end NUMINAMATH_GPT_coeffs_divisible_by_5_l1756_175614


namespace NUMINAMATH_GPT_Elizabeth_lost_bottles_l1756_175608

theorem Elizabeth_lost_bottles :
  ∃ (L : ℕ), (10 - L - 1) * 3 = 21 ∧ L = 2 := by
  sorry

end NUMINAMATH_GPT_Elizabeth_lost_bottles_l1756_175608


namespace NUMINAMATH_GPT_Grace_pool_water_capacity_l1756_175688

theorem Grace_pool_water_capacity :
  let rate1 := 50 -- gallons per hour of the first hose
  let rate2 := 70 -- gallons per hour of the second hose
  let hours1 := 3 -- hours the first hose was used alone
  let hours2 := 2 -- hours both hoses were used together
  let water1 := rate1 * hours1 -- water from the first hose in the first period
  let water2 := rate2 * hours2 -- water from the second hose in the second period
  let water3 := rate1 * hours2 -- water from the first hose in the second period
  let total_water := water1 + water2 + water3 -- total water in the pool
  total_water = 390 :=
by
  sorry

end NUMINAMATH_GPT_Grace_pool_water_capacity_l1756_175688


namespace NUMINAMATH_GPT_find_positive_integer_pair_l1756_175657

noncomputable def quadratic_has_rational_solutions (d : ℤ) : Prop :=
  ∃ x : ℚ, 7 * x^2 + 13 * x + d = 0

theorem find_positive_integer_pair :
  ∃ (d1 d2 : ℕ), 
  d1 > 0 ∧ d2 > 0 ∧ 
  quadratic_has_rational_solutions d1 ∧ quadratic_has_rational_solutions d2 ∧ 
  d1 * d2 = 2 := 
sorry -- Proof left as an exercise

end NUMINAMATH_GPT_find_positive_integer_pair_l1756_175657


namespace NUMINAMATH_GPT_not_universally_better_l1756_175634

-- Definitions based on the implicitly given conditions
def can_show_quantity (chart : Type) : Prop := sorry
def can_reflect_changes (chart : Type) : Prop := sorry

-- Definitions of bar charts and line charts
inductive BarChart
| mk : BarChart

inductive LineChart
| mk : LineChart

-- Assumptions based on characteristics of the charts
axiom bar_chart_shows_quantity : can_show_quantity BarChart 
axiom line_chart_shows_quantity : can_show_quantity LineChart 
axiom line_chart_reflects_changes : can_reflect_changes LineChart 

-- Proof problem statement
theorem not_universally_better : ¬(∀ (c1 c2 : Type), can_show_quantity c1 → can_reflect_changes c1 → ¬can_show_quantity c2 → ¬can_reflect_changes c2) :=
  sorry

end NUMINAMATH_GPT_not_universally_better_l1756_175634


namespace NUMINAMATH_GPT_scientific_notation_of_0_0000000005_l1756_175656

theorem scientific_notation_of_0_0000000005 : 0.0000000005 = 5 * 10^(-10) :=
by {
  sorry
}

end NUMINAMATH_GPT_scientific_notation_of_0_0000000005_l1756_175656


namespace NUMINAMATH_GPT_min_value_expr_l1756_175627

-- Definition of the expression given a real constant k
def expr (k : ℝ) (x y : ℝ) : ℝ := 9 * x^2 - 8 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 6 * y + 9

-- The proof problem statement
theorem min_value_expr (k : ℝ) (h : k = 2 / 9) : ∃ x y : ℝ, expr k x y = 1 ∧ ∀ x y : ℝ, expr k x y ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expr_l1756_175627


namespace NUMINAMATH_GPT_gcd_calculation_l1756_175670

def gcd_36_45_495 : ℕ :=
  Int.gcd 36 (Int.gcd 45 495)

theorem gcd_calculation : gcd_36_45_495 = 9 := by
  sorry

end NUMINAMATH_GPT_gcd_calculation_l1756_175670


namespace NUMINAMATH_GPT_average_growth_rate_of_second_brand_l1756_175629

theorem average_growth_rate_of_second_brand 
  (init1 : ℝ) (rate1 : ℝ) (init2 : ℝ) (t : ℝ) (r : ℝ)
  (h1 : init1 = 4.9) (h2 : rate1 = 0.275) (h3 : init2 = 2.5) (h4 : t = 5.647)
  (h_eq : init1 + rate1 * t = init2 + r * t) : 
  r = 0.7 :=
by 
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_average_growth_rate_of_second_brand_l1756_175629


namespace NUMINAMATH_GPT_smallest_sector_angle_l1756_175635

def arithmetic_sequence_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem smallest_sector_angle 
  (a : ℕ) (d : ℕ) (n : ℕ := 15) (sum_angles : ℕ := 360) 
  (angles_arith_seq : arithmetic_sequence_sum a d n = sum_angles) 
  (h_poses : ∀ m : ℕ, arithmetic_sequence_sum a d m = sum_angles -> m = n) 
  : a = 3 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_sector_angle_l1756_175635


namespace NUMINAMATH_GPT_ratio_Y_to_Z_l1756_175632

variables (X Y Z : ℕ)

def population_relation1 (X Y : ℕ) : Prop := X = 3 * Y
def population_relation2 (X Z : ℕ) : Prop := X = 6 * Z

theorem ratio_Y_to_Z (h1 : population_relation1 X Y) (h2 : population_relation2 X Z) : Y / Z = 2 :=
  sorry

end NUMINAMATH_GPT_ratio_Y_to_Z_l1756_175632


namespace NUMINAMATH_GPT_pos_real_ineq_l1756_175653

theorem pos_real_ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^a * b^b * c^c ≥ (a * b * c)^((a + b + c)/3) :=
by 
  sorry

end NUMINAMATH_GPT_pos_real_ineq_l1756_175653


namespace NUMINAMATH_GPT_problem_a_problem_b_l1756_175673

-- Problem (a)
theorem problem_a (n : ℕ) (hn : n > 0) :
  ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x^(n-1) + y^n = z^(n+1) :=
sorry

-- Problem (b)
theorem problem_b (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (coprime_ab : Nat.gcd a b = 1)
  (coprime_ac_or_bc : Nat.gcd c a = 1 ∨ Nat.gcd c b = 1) :
  ∃ᶠ x : ℕ in Filter.atTop, ∃ (y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x^a + y^b = z^c :=
sorry

end NUMINAMATH_GPT_problem_a_problem_b_l1756_175673


namespace NUMINAMATH_GPT_number_of_ways_to_represent_5030_l1756_175694

theorem number_of_ways_to_represent_5030 :
  let even := {x : ℕ | x % 2 = 0}
  let in_range := {x : ℕ | x ≤ 98}
  let valid_b := even ∩ in_range
  ∃ (M : ℕ), M = 150 ∧ ∀ (b3 b2 b1 b0 : ℕ), 
    b3 ∈ valid_b ∧ b2 ∈ valid_b ∧ b1 ∈ valid_b ∧ b0 ∈ valid_b →
    5030 = b3 * 10 ^ 3 + b2 * 10 ^ 2 + b1 * 10 + b0 → 
    M = 150 :=
  sorry

end NUMINAMATH_GPT_number_of_ways_to_represent_5030_l1756_175694


namespace NUMINAMATH_GPT_point_to_focus_distance_l1756_175600

def parabola : Set (ℝ × ℝ) := { p | p.2^2 = 4 * p.1 }

def point_P : ℝ × ℝ := (3, 2) -- Since y^2 = 4*3 hence y = ±2 and we choose one of the (3, 2) or (3, -2)

def focus_F : ℝ × ℝ := (1, 0) -- Focus of y^2 = 4x is (1, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem point_to_focus_distance : distance point_P focus_F = 4 := by
  sorry -- Proof goes here

end NUMINAMATH_GPT_point_to_focus_distance_l1756_175600
