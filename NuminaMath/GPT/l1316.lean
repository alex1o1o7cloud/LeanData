import Mathlib

namespace other_root_of_quadratic_l1316_131651

theorem other_root_of_quadratic (a b c : ℚ) (x₁ x₂ : ℚ) :
  a ≠ 0 →
  x₁ = 4 / 9 →
  (a * x₁^2 + b * x₁ + c = 0) →
  (a = 81) →
  (b = -145) →
  (c = 64) →
  x₂ = -16 / 9
:=
sorry

end other_root_of_quadratic_l1316_131651


namespace ferry_routes_ratio_l1316_131669

theorem ferry_routes_ratio :
  ∀ (D_P D_Q : ℝ) (speed_P time_P speed_Q time_Q : ℝ),
  speed_P = 8 →
  time_P = 3 →
  speed_Q = speed_P + 4 →
  time_Q = time_P + 1 →
  D_P = speed_P * time_P →
  D_Q = speed_Q * time_Q →
  D_Q / D_P = 2 :=
by sorry

end ferry_routes_ratio_l1316_131669


namespace johns_share_is_1100_l1316_131614

def total_amount : ℕ := 6600
def ratio_john : ℕ := 2
def ratio_jose : ℕ := 4
def ratio_binoy : ℕ := 6
def total_parts : ℕ := ratio_john + ratio_jose + ratio_binoy
def value_per_part : ℚ := total_amount / total_parts
def amount_received_by_john : ℚ := value_per_part * ratio_john

theorem johns_share_is_1100 : amount_received_by_john = 1100 := by
  sorry

end johns_share_is_1100_l1316_131614


namespace largest_six_consecutive_composites_less_than_40_l1316_131696

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) := ¬ is_prime n ∧ n > 1

theorem largest_six_consecutive_composites_less_than_40 :
  ∃ (seq : ℕ → ℕ) (i : ℕ),
    (∀ j : ℕ, j < 6 → is_composite (seq (i + j))) ∧ 
    (seq i < 40) ∧ 
    (seq (i+1) < 40) ∧ 
    (seq (i+2) < 40) ∧ 
    (seq (i+3) < 40) ∧ 
    (seq (i+4) < 40) ∧ 
    (seq (i+5) < 40) ∧ 
    seq (i+5) = 30 
:= sorry

end largest_six_consecutive_composites_less_than_40_l1316_131696


namespace second_hand_degree_per_minute_l1316_131617

theorem second_hand_degree_per_minute :
  (∀ (t : ℝ), t = 60 → 360 / t = 6) :=
by
  intro t
  intro ht
  rw [ht]
  norm_num

end second_hand_degree_per_minute_l1316_131617


namespace integer_roots_of_quadratic_l1316_131694

theorem integer_roots_of_quadratic (a : ℚ) :
  (∃ x₁ x₂ : ℤ, 
    a * x₁ * x₁ + (a + 1) * x₁ + (a - 1) = 0 ∧ 
    a * x₂ * x₂ + (a + 1) * x₂ + (a - 1) = 0 ∧ 
    x₁ ≠ x₂) ↔ 
      a = 0 ∨ a = -1/7 ∨ a = 1 :=
by
  sorry

end integer_roots_of_quadratic_l1316_131694


namespace valid_x_values_l1316_131691

noncomputable def valid_triangle_sides (x : ℕ) : Prop :=
  8 + 11 > x + 3 ∧ 8 + (x + 3) > 11 ∧ 11 + (x + 3) > 8

theorem valid_x_values :
  {x : ℕ | valid_triangle_sides x ∧ x > 0} = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15} :=
by
  sorry

end valid_x_values_l1316_131691


namespace avg_width_is_3_5_l1316_131650

def book_widths : List ℚ := [4, (3/4), 1.25, 3, 2, 7, 5.5]

noncomputable def average (l : List ℚ) : ℚ :=
  l.sum / l.length

theorem avg_width_is_3_5 : average book_widths = 23.5 / 7 :=
by
  sorry

end avg_width_is_3_5_l1316_131650


namespace withdrawal_amount_in_2008_l1316_131611

noncomputable def total_withdrawal (a : ℕ) (p : ℝ) : ℝ :=
  (a / p) * ((1 + p) - (1 + p)^8)

theorem withdrawal_amount_in_2008 (a : ℕ) (p : ℝ) (h_pos : 0 < p) (h_neg_one_lt : -1 < p) :
  total_withdrawal a p = (a / p) * ((1 + p) - (1 + p)^8) :=
by
  -- Conditions
  -- Starting from May 10th, 2001, multiple annual deposits.
  -- Annual interest rate p > 0 and p > -1.
  sorry

end withdrawal_amount_in_2008_l1316_131611


namespace range_x_range_a_l1316_131673

variable {x a : ℝ}
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x - 3) * (2 - x) ≥ 0

-- (1) If a = 1, find the range of x for which p ∧ q is true.
theorem range_x (h : a = 1) : 2 ≤ x ∧ x < 3 ↔ p 1 x ∧ q x := sorry

-- (2) If ¬p is a necessary but not sufficient condition for ¬q, find the range of real number a.
theorem range_a : (¬p a x → ¬q x) → (∃ a : ℝ, 1 < a ∧ a < 2) := sorry

end range_x_range_a_l1316_131673


namespace exists_parallel_line_l1316_131699

variable (P : ℝ × ℝ)
variable (g : ℝ × ℝ)
variable (in_first_quadrant : 0 < P.1 ∧ 0 < P.2)
variable (parallel_to_second_projection_plane : ∃ c : ℝ, g = (c, 0))

theorem exists_parallel_line (P : ℝ × ℝ) (g : ℝ × ℝ) (in_first_quadrant : 0 < P.1 ∧ 0 < P.2)
  (parallel_to_second_projection_plane : ∃ c : ℝ, g = (c, 0)) :
  ∃ a : ℝ × ℝ, (∃ d : ℝ, g = (d, 0)) ∧ (a = P) :=
sorry

end exists_parallel_line_l1316_131699


namespace inequality_solution_non_negative_integer_solutions_l1316_131692

theorem inequality_solution (x : ℝ) :
  (x - 2) / 2 ≤ (7 - x) / 3 → x ≤ 4 :=
by
  sorry

theorem non_negative_integer_solutions :
  { n : ℤ | n ≥ 0 ∧ n ≤ 4 } = {0, 1, 2, 3, 4} :=
by
  sorry

end inequality_solution_non_negative_integer_solutions_l1316_131692


namespace percentage_loss_l1316_131628

theorem percentage_loss (CP SP : ℝ) (h₁ : CP = 1400) (h₂ : SP = 1232) :
  ((CP - SP) / CP) * 100 = 12 :=
by
  sorry

end percentage_loss_l1316_131628


namespace number_of_trees_l1316_131674

theorem number_of_trees (n : ℕ) (diff : ℕ) (count1 : ℕ) (count2 : ℕ) (timur1 : ℕ) (alexander1 : ℕ) (timur2 : ℕ) (alexander2 : ℕ) : 
  diff = alexander1 - timur1 ∧
  count1 = timur2 + (alexander2 - timur1) ∧
  n = count1 + diff →
  n = 118 :=
by
  sorry

end number_of_trees_l1316_131674


namespace cost_per_mile_sunshine_is_018_l1316_131689

theorem cost_per_mile_sunshine_is_018 :
  ∀ (x : ℝ) (daily_rate_sunshine daily_rate_city cost_per_mile_city : ℝ),
  daily_rate_sunshine = 17.99 →
  daily_rate_city = 18.95 →
  cost_per_mile_city = 0.16 →
  (daily_rate_sunshine + 48 * x = daily_rate_city + cost_per_mile_city * 48) →
  x = 0.18 :=
by
  intros x daily_rate_sunshine daily_rate_city cost_per_mile_city
  intros h1 h2 h3 h4
  sorry

end cost_per_mile_sunshine_is_018_l1316_131689


namespace prove_ab_ge_5_l1316_131618

theorem prove_ab_ge_5 (a b c : ℕ) (h : ∀ x, x * (a * x) = b * x + c → 0 ≤ x ∧ x ≤ 1) : 5 ≤ a ∧ 5 ≤ b := 
sorry

end prove_ab_ge_5_l1316_131618


namespace arithmetic_geometric_sequence_l1316_131697

theorem arithmetic_geometric_sequence (a : ℕ → ℤ) (h1 : ∀ n, a (n + 1) = a n + 3)
    (h2 : (a 1 + 3) * (a 1 + 21) = (a 1 + 9) ^ 2) : a 3 = 12 :=
by 
  sorry

end arithmetic_geometric_sequence_l1316_131697


namespace proof_problem_l1316_131620

-- Define sets A and B according to the given conditions
def A : Set ℝ := { x | x ≥ -1 }
def B : Set ℝ := { x | x > 2 }
def complement_B : Set ℝ := { x | ¬ (x > 2) }  -- Complement of B

-- Remaining intersection expression
def intersect_expr : Set ℝ := { x | x ≥ -1 ∧ x ≤ 2 }

-- Statement to prove
theorem proof_problem : (A ∩ complement_B) = intersect_expr :=
sorry

end proof_problem_l1316_131620


namespace coffee_ratio_correct_l1316_131600

noncomputable def ratio_of_guests (cups_weak : ℕ) (cups_strong : ℕ) (tablespoons_weak : ℕ) (tablespoons_strong : ℕ) (total_tablespoons : ℕ) : ℤ :=
  if (cups_weak * tablespoons_weak + cups_strong * tablespoons_strong = total_tablespoons) then
    (cups_weak * tablespoons_weak / gcd (cups_weak * tablespoons_weak) (cups_strong * tablespoons_strong)) /
    (cups_strong * tablespoons_strong / gcd (cups_weak * tablespoons_weak) (cups_strong * tablespoons_strong))
  else 0

theorem coffee_ratio_correct :
  ratio_of_guests 12 12 1 2 36 = 1 / 2 :=
by
  sorry

end coffee_ratio_correct_l1316_131600


namespace pencils_per_box_l1316_131661

-- Variables and Definitions based on the problem conditions
def num_boxes : ℕ := 10
def pencils_kept : ℕ := 10
def friends : ℕ := 5
def pencils_per_friend : ℕ := 8

-- Theorem to prove the solution
theorem pencils_per_box (pencils_total : ℕ)
  (h1 : pencils_total = pencils_kept + (friends * pencils_per_friend))
  (h2 : pencils_total = num_boxes * (pencils_total / num_boxes)) :
  (pencils_total / num_boxes) = 5 :=
sorry

end pencils_per_box_l1316_131661


namespace quadratic_roots_correct_l1316_131658

theorem quadratic_roots_correct (x : ℝ) : (x^2 = 2 * x) ↔ (x = 0 ∨ x = 2) := 
by
  sorry

end quadratic_roots_correct_l1316_131658


namespace find_daily_rate_second_company_l1316_131665

def daily_rate_second_company (x : ℝ) : Prop :=
  let total_cost_1 := 21.95 + 0.19 * 150
  let total_cost_2 := x + 0.21 * 150
  total_cost_1 = total_cost_2

theorem find_daily_rate_second_company : daily_rate_second_company 18.95 :=
  by
  unfold daily_rate_second_company
  sorry

end find_daily_rate_second_company_l1316_131665


namespace find_f_of_2_l1316_131612

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x - b

theorem find_f_of_2 (a b : ℝ) (h_pos : 0 < a)
  (h1 : ∀ x : ℝ, a * f x a b - b = 4 * x - 3)
  : f 2 a b = 3 := 
sorry

end find_f_of_2_l1316_131612


namespace average_percentage_l1316_131670

theorem average_percentage (n1 n2 : ℕ) (s1 s2 : ℕ)
  (h1 : n1 = 15) (h2 : s1 = 80) (h3 : n2 = 10) (h4 : s2 = 90) :
  (n1 * s1 + n2 * s2) / (n1 + n2) = 84 :=
by
  sorry

end average_percentage_l1316_131670


namespace minimum_value_expression_l1316_131649

noncomputable def expr (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2 - 2*x - 2*y + 2) + 
  Real.sqrt (x^2 + y^2 - 2*x + 4*y + 2*Real.sqrt 3*y + 8 + 4*Real.sqrt 3) +
  Real.sqrt (x^2 + y^2 + 8*x + 4*Real.sqrt 3*x - 4*y + 32 + 16*Real.sqrt 3)

theorem minimum_value_expression : (∃ x y : ℝ, expr x y = 3*Real.sqrt 6 + 4*Real.sqrt 2) :=
sorry

end minimum_value_expression_l1316_131649


namespace sibling_age_difference_l1316_131608

theorem sibling_age_difference (Y : ℝ) (Y_eq : Y = 25.75) (avg_age_eq : (Y + (Y + 3) + (Y + 6) + (Y + x)) / 4 = 30) : (Y + 6) - Y = 6 :=
by
  sorry

end sibling_age_difference_l1316_131608


namespace travel_cost_is_correct_l1316_131601

-- Definitions of the conditions
def lawn_length : ℝ := 80
def lawn_breadth : ℝ := 60
def road_width : ℝ := 15
def cost_per_sq_m : ℝ := 3

-- Areas of individual roads
def area_road_length := road_width * lawn_breadth
def area_road_breadth := road_width * lawn_length
def intersection_area := road_width * road_width

-- Adjusted area for roads discounting intersection area
def total_area_roads := area_road_length + area_road_breadth - intersection_area

-- Total cost of traveling the roads
def total_cost := total_area_roads * cost_per_sq_m

theorem travel_cost_is_correct : total_cost = 5625 := by
  sorry

end travel_cost_is_correct_l1316_131601


namespace non_swimmers_play_soccer_percentage_l1316_131634

theorem non_swimmers_play_soccer_percentage (N : ℕ) (hN_pos : 0 < N)
 (h1 : (0.7 * N : ℝ) = x)
 (h2 : (0.5 * N : ℝ) = y)
 (h3 : (0.6 * x : ℝ) = z)
 : (0.56 * y = 0.28 * N) := 
 sorry

end non_swimmers_play_soccer_percentage_l1316_131634


namespace vertical_line_division_l1316_131666

theorem vertical_line_division (A B C : ℝ × ℝ)
    (hA : A = (0, 2)) (hB : B = (0, 0)) (hC : C = (6, 0))
    (a : ℝ) (h_area_half : 1 / 2 * 6 * 2 / 2 = 3) :
    a = 3 :=
sorry

end vertical_line_division_l1316_131666


namespace value_of_z_plus_one_over_y_l1316_131668

theorem value_of_z_plus_one_over_y
  (x y z : ℝ)
  (h1 : 0 < x) 
  (h2 : 0 < y) 
  (h3 : 0 < z)
  (h4 : x * y * z = 1)
  (h5 : x + 1 / z = 3)
  (h6 : y + 1 / x = 31) :
  z + 1 / y = 9 / 23 :=
by
  sorry

end value_of_z_plus_one_over_y_l1316_131668


namespace reduced_price_after_discount_l1316_131606

theorem reduced_price_after_discount (P R : ℝ) (h1 : R = 0.8 * P) (h2 : 1500 / R - 1500 / P = 10) :
  R = 30 := 
by
  sorry

end reduced_price_after_discount_l1316_131606


namespace percentage_carnations_l1316_131648

variable (F : ℕ)
variable (H1 : F ≠ 0) -- Non-zero flowers
variable (H2 : ∀ (y : ℕ), 5 * y = F → 2 * y ≠ 0) -- Two fifths of the pink flowers are roses.
variable (H3 : ∀ (z : ℕ), 7 * z = 3 * (F - F / 2 - F / 5) → 6 * z ≠ 0) -- Six sevenths of the red flowers are carnations.
variable (H4 : ∀ (w : ℕ), 5 * w = F → w ≠ 0) -- One fifth of the flowers are yellow tulips.
variable (H5 : 2 * F / 2 = F) -- Half of the flowers are pink.
variable (H6 : ∀ (c : ℕ), 10 * c = F → c ≠ 0) -- Total flowers in multiple of 10

theorem percentage_carnations :
  (exists (pc rc : ℕ), 70 * (pc + rc) = 55 * F) :=
sorry

end percentage_carnations_l1316_131648


namespace probability_of_same_suit_l1316_131616

-- Definitions for the conditions
def total_cards : ℕ := 52
def suits : ℕ := 4
def cards_per_suit : ℕ := 13
def total_draws : ℕ := 2

-- Definition of factorial for binomial coefficient calculation
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Binomial coefficient calculation
def binomial_coeff (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Calculation of the probability
def prob_same_suit : ℚ :=
  let ways_to_choose_2_cards_from_52 := binomial_coeff total_cards total_draws
  let ways_to_choose_2_cards_per_suit := binomial_coeff cards_per_suit total_draws
  let total_ways_to_choose_2_same_suit := suits * ways_to_choose_2_cards_per_suit
  total_ways_to_choose_2_same_suit / ways_to_choose_2_cards_from_52

theorem probability_of_same_suit :
  prob_same_suit = 4 / 17 :=
by
  sorry

end probability_of_same_suit_l1316_131616


namespace sum_of_three_integers_l1316_131641

def three_positive_integers (x y z : ℕ) : Prop :=
  x + y = 2003 ∧ y - z = 1000

theorem sum_of_three_integers (x y z : ℕ) (h1 : x + y = 2003) (h2 : y - z = 1000) (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) : 
  x + y + z = 2004 := 
by 
  sorry

end sum_of_three_integers_l1316_131641


namespace find_eccentricity_find_equation_l1316_131625

open Real

-- Conditions for the first question
def is_ellipse (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

def are_focus (a b : ℝ) (F1 F2 : ℝ × ℝ) : Prop :=
  F1 = ( - sqrt (a^2 - b^2), 0) ∧ F2 = (sqrt (a^2 - b^2), 0)

def arithmetic_sequence (a b : ℝ) (A B : ℝ × ℝ) (F1 : ℝ × ℝ) : Prop :=
  let dist_AF1 := abs (A.1 - F1.1)
  let dist_BF1 := abs (B.1 - F1.1)
  let dist_AB := sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  (dist_AF1 + dist_AB + dist_BF1 = 4 * a) ∧
  (dist_AF1 + dist_BF1 = 2 * dist_AB)

-- Proof statement for the eccentricity
theorem find_eccentricity (a b : ℝ) (F1 F2 A B : ℝ × ℝ)
  (h1 : a > b) (h2 : b > 0) (h3 : is_ellipse a b)
  (h4 : are_focus a b F1 F2)
  (h5 : arithmetic_sequence a b A B F1) :
  ∃ e : ℝ, e = sqrt 2 / 2 :=
sorry

-- Conditions for the second question
def geometric_property (a b : ℝ) (A B P : ℝ × ℝ) : Prop :=
  ∀ x y : ℝ, P = (0, -1) → 
             (x^2 / a^2) + (y^2 / b^2) = 1 → 
             abs ((P.1 - A.1)^2 + (P.2 - A.2)^2) = 
             abs ((P.1 - B.1)^2 + (P.2 - B.2)^2)

-- Proof statement for the equation of the ellipse
theorem find_equation (a b : ℝ) (A B P : ℝ × ℝ)
  (h1 : a = 3 * sqrt 2) (h2 : b = 3) (h3 : P = (0, -1))
  (h4 : is_ellipse a b) (h5 : geometric_property a b A B P) :
  ∃ E : Prop, E = ((x : ℝ) * 2 / 18 + (y : ℝ) * 2 / 9 = 1) :=
sorry

end find_eccentricity_find_equation_l1316_131625


namespace intersection_M_N_l1316_131613

def is_M (x : ℝ) : Prop := x^2 + x - 6 < 0
def is_N (x : ℝ) : Prop := abs (x - 1) <= 2

theorem intersection_M_N : {x : ℝ | is_M x} ∩ {x : ℝ | is_N x} = {x : ℝ | -1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_M_N_l1316_131613


namespace sequence_squared_l1316_131604

theorem sequence_squared (a : ℕ → ℕ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n ≥ 2, a n = a (n - 1) + 2 * (n - 1)) 
  : ∀ n, a n = n^2 := 
by
  sorry

end sequence_squared_l1316_131604


namespace absent_children_count_l1316_131638

theorem absent_children_count : ∀ (total_children present_children absent_children bananas : ℕ), 
  total_children = 260 → 
  bananas = 2 * total_children → 
  bananas = 4 * present_children → 
  present_children + absent_children = total_children →
  absent_children = 130 :=
by
  intros total_children present_children absent_children bananas h1 h2 h3 h4
  sorry

end absent_children_count_l1316_131638


namespace river_flow_rate_l1316_131636

theorem river_flow_rate
  (depth width volume_per_minute : ℝ)
  (h1 : depth = 2)
  (h2 : width = 45)
  (h3 : volume_per_minute = 6000) :
  (volume_per_minute / (depth * width)) * (1 / 1000) * 60 = 4.0002 :=
by
  -- Sorry is used to skip the proof.
  sorry

end river_flow_rate_l1316_131636


namespace sum_series_and_convergence_l1316_131653

theorem sum_series_and_convergence (x : ℝ) (h : -1 < x ∧ x < 1) :
  ∑' n, (n + 6) * x^(7 * n) = (6 - 5 * x^7) / (1 - x^7)^2 :=
by
  sorry

end sum_series_and_convergence_l1316_131653


namespace inequality_power_cubed_l1316_131633

theorem inequality_power_cubed
  (x y a : ℝ)
  (h_condition : (0 < a ∧ a < 1) ∧ a ^ x < a ^ y) : x^3 > y^3 :=
by {
  sorry
}

end inequality_power_cubed_l1316_131633


namespace prob_iff_eq_l1316_131667

noncomputable def A (m : ℝ) : Set ℝ := { x | x^2 + m * x + 2 ≥ 0 ∧ x ≥ 0 }
noncomputable def B (m : ℝ) : Set ℝ := { y | ∃ x, x ∈ A m ∧ y = Real.sqrt (x^2 + m * x + 2) }

theorem prob_iff_eq (m : ℝ) : (A m = { y | ∃ x, x ^ 2 + m * x + 2 = y ^ 2 ∧ x ≥ 0 } ↔ m = -2 * Real.sqrt 2) :=
by
  sorry

end prob_iff_eq_l1316_131667


namespace unique_solution_quadratic_l1316_131659

theorem unique_solution_quadratic (x : ℚ) (b : ℚ) (h_b_nonzero : b ≠ 0) (h_discriminant_zero : 625 - 36 * b = 0) : 
  (b = 625 / 36) ∧ (x = -18 / 25) → b * x^2 + 25 * x + 9 = 0 :=
by 
  -- We assume b = 625 / 36 and x = -18 / 25
  rintro ⟨hb, hx⟩
  -- Substitute b and x into the quadratic equation and simplify
  rw [hb, hx]
  -- Show the left-hand side evaluates to zero
  sorry

end unique_solution_quadratic_l1316_131659


namespace correct_equation_for_annual_consumption_l1316_131647

-- Definitions based on the problem conditions
-- average_monthly_consumption_first_half is the average monthly electricity consumption in the first half of the year, assumed to be x
def average_monthly_consumption_first_half (x : ℝ) := x

-- average_monthly_consumption_second_half is the average monthly consumption in the second half of the year, i.e., x - 2000
def average_monthly_consumption_second_half (x : ℝ) := x - 2000

-- total_annual_consumption is the total annual electricity consumption which is 150000 kWh
def total_annual_consumption (x : ℝ) := 6 * average_monthly_consumption_first_half x + 6 * average_monthly_consumption_second_half x

-- The main theorem statement which we need to prove
theorem correct_equation_for_annual_consumption (x : ℝ) : total_annual_consumption x = 150000 :=
by
  -- equation derivation
  sorry

end correct_equation_for_annual_consumption_l1316_131647


namespace value_of_4_Y_3_l1316_131626

def Y (a b : ℕ) : ℕ := (2 * a ^ 2 - 3 * a * b + b ^ 2) ^ 2

theorem value_of_4_Y_3 : Y 4 3 = 25 := by
  sorry

end value_of_4_Y_3_l1316_131626


namespace find_some_number_l1316_131695

-- Definitions of symbol replacements
def replacement_minus (a b : Nat) := a + b
def replacement_plus (a b : Nat) := a * b
def replacement_times (a b : Nat) := a / b
def replacement_div (a b : Nat) := a - b

-- The transformed equation using the replacements
def transformed_equation (some_number : Nat) :=
  replacement_minus
    some_number
    (replacement_div
      (replacement_plus 9 (replacement_times 8 3))
      25) = 5

theorem find_some_number : ∃ some_number : Nat, transformed_equation some_number ∧ some_number = 6 :=
by
  exists 6
  unfold transformed_equation
  unfold replacement_minus replacement_plus replacement_times replacement_div
  sorry

end find_some_number_l1316_131695


namespace fruit_trees_l1316_131656

theorem fruit_trees (total_streets : ℕ) 
  (fruit_trees_every_other : total_streets % 2 = 0) 
  (equal_fruit_trees : ∀ n : ℕ, 3 * n = total_streets / 2) : 
  ∃ n : ℕ, n = total_streets / 6 :=
by
  sorry

end fruit_trees_l1316_131656


namespace find_n_l1316_131690

theorem find_n : ∃ n : ℕ, n < 200 ∧ ∃ k : ℕ, n^2 + (n + 1)^2 = k^2 ∧ (n = 3 ∨ n = 20 ∨ n = 119) := 
by
  sorry

end find_n_l1316_131690


namespace contingency_fund_l1316_131683

theorem contingency_fund:
  let d := 240
  let cp := d * (1.0 / 3)
  let lc := d * (1.0 / 2)
  let r := d - cp - lc
  let lp := r * (1.0 / 4)
  let cf := r - lp
  cf = 30 := 
by
  sorry

end contingency_fund_l1316_131683


namespace integrality_condition_l1316_131682

noncomputable def binom (n k : ℕ) : ℕ := 
  n.choose k

theorem integrality_condition (n k : ℕ) (h : 1 ≤ k) (h1 : k < n) (h2 : (k + 1) ∣ (n^2 - 3*k^2 - 2)) : 
  ∃ m : ℕ, m = (n^2 - 3*k^2 - 2) / (k + 1) ∧ (m * binom n k) % 1 = 0 :=
sorry

end integrality_condition_l1316_131682


namespace different_books_read_l1316_131624

theorem different_books_read (t_books d_books b_books td_same all_same : ℕ)
  (h_t_books : t_books = 23)
  (h_d_books : d_books = 12)
  (h_b_books : b_books = 17)
  (h_td_same : td_same = 3)
  (h_all_same : all_same = 1) : 
  t_books + d_books + b_books - (td_same + all_same) = 48 := 
by
  sorry

end different_books_read_l1316_131624


namespace real_solutions_l1316_131688

theorem real_solutions (x : ℝ) :
  (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) + 1 / ((x - 5) * (x - 6)) = 1 / 12) ↔ (x = 12 ∨ x = -4) :=
by
  sorry

end real_solutions_l1316_131688


namespace intersection_of_A_and_B_l1316_131652

open Set

-- Definition of set A
def A : Set ℤ := {1, 2, 3}

-- Definition of set B
def B : Set ℤ := {x | x < -1 ∨ 0 < x ∧ x < 2}

-- The theorem to prove A ∩ B = {1}
theorem intersection_of_A_and_B : A ∩ B = {1} := by
  -- Proof logic here
  sorry

end intersection_of_A_and_B_l1316_131652


namespace gcd_217_155_l1316_131645

theorem gcd_217_155 : Nat.gcd 217 155 = 1 := by
  sorry

end gcd_217_155_l1316_131645


namespace option_A_option_B_option_D_l1316_131605

-- Definitions of sequences
def arithmetic_seq (a_1 : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a_1 + n * d

def geometric_seq (b_1 : ℤ) (q : ℤ) (n : ℕ) : ℤ :=
  b_1 * q ^ n

-- Option A: Prove that there exist d and q such that a_n = b_n
theorem option_A : ∃ (d q : ℤ), ∀ (a_1 b_1 : ℤ) (n : ℕ), 
  (arithmetic_seq a_1 d n = geometric_seq b_1 q n) := sorry

-- Sum of the first n terms of an arithmetic sequence
def sum_arithmetic_seq (a_1 : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a_1 + (n - 1) * d) / 2

-- Option B: Prove the differences form an arithmetic sequence
theorem option_B (a_1 : ℤ) (d : ℤ) :
  ∀ n k : ℕ, k > 0 → 
  (sum_arithmetic_seq a_1 d ((k + 1) * n) - sum_arithmetic_seq a_1 d (k * n) =
   (sum_arithmetic_seq a_1 d n + k * n * n * d)) := sorry

-- Option D: Prove there exist real numbers A and a such that A * a^a_n = b_n
theorem option_D (a_1 : ℤ) (d : ℤ) (b_1 : ℤ) (q : ℤ) :
  ∀ n : ℕ, b_1 > 0 → q > 0 → 
  ∃ A a : ℝ, A * a^ (arithmetic_seq a_1 d n) = (geometric_seq b_1 q n) := sorry

end option_A_option_B_option_D_l1316_131605


namespace participants_won_more_than_lost_l1316_131637

-- Define the conditions given in the problem
def total_participants := 64
def rounds := 6

-- Define a function that calculates the number of participants reaching a given round
def participants_after_round (n : Nat) (r : Nat) : Nat :=
  n / (2 ^ r)

-- The theorem we need to prove
theorem participants_won_more_than_lost :
  participants_after_round total_participants 2 = 16 :=
by 
  -- Provide a placeholder for the proof
  sorry

end participants_won_more_than_lost_l1316_131637


namespace evaluate_expression_l1316_131640

theorem evaluate_expression (x : ℕ) (h : x = 5) : 2 * x ^ 2 + 5 = 55 := by
  sorry

end evaluate_expression_l1316_131640


namespace reservoir_solution_l1316_131607

theorem reservoir_solution (x y z : ℝ) :
  8 * (1 / x - 1 / y) = 1 →
  24 * (1 / x - 1 / y - 1 / z) = 1 →
  8 * (1 / y + 1 / z) = 1 →
  x = 8 ∧ y = 24 ∧ z = 12 :=
by
  intros h1 h2 h3
  sorry

end reservoir_solution_l1316_131607


namespace find_x_range_l1316_131632

theorem find_x_range (x : ℝ) (h1 : 1 / x < 3) (h2 : 1 / x > -2) (h3 : 2 * x - 5 > 0) : x > 5 / 2 :=
by
  sorry

end find_x_range_l1316_131632


namespace mars_mission_cost_per_person_l1316_131602

theorem mars_mission_cost_per_person
  (total_cost : ℕ) (number_of_people : ℕ)
  (h1 : total_cost = 50000000000) (h2 : number_of_people = 500000000) :
  (total_cost / number_of_people) = 100 := 
by
  sorry

end mars_mission_cost_per_person_l1316_131602


namespace upper_limit_b_l1316_131663

theorem upper_limit_b (a b : ℤ) (h1 : 6 < a) (h2 : a < 17) (h3 : 3 < b) (h4 : (a : ℚ) / b ≤ 3.75) : b ≤ 4 := by
  sorry

end upper_limit_b_l1316_131663


namespace finance_to_manufacturing_ratio_l1316_131635

theorem finance_to_manufacturing_ratio : 
    let finance_angle := 72
    let manufacturing_angle := 108
    (finance_angle:ℕ) / (Nat.gcd finance_angle manufacturing_angle) = 2 ∧ 
    (manufacturing_angle:ℕ) / (Nat.gcd finance_angle manufacturing_angle) = 3 := 
by 
    sorry

end finance_to_manufacturing_ratio_l1316_131635


namespace average_temperature_l1316_131631

theorem average_temperature (temps : List ℕ) (temps_eq : temps = [40, 47, 45, 41, 39]) :
  (temps.sum : ℚ) / temps.length = 42.4 :=
by
  sorry

end average_temperature_l1316_131631


namespace proper_subset_count_of_set_l1316_131680

theorem proper_subset_count_of_set (s : Finset ℕ) (h : s = {1, 2, 3}) : s.powerset.card - 1 = 7 := by
  sorry

end proper_subset_count_of_set_l1316_131680


namespace naomi_number_of_ways_to_1000_l1316_131672

-- Define the initial condition and operations

def start : ℕ := 2

def add1 (n : ℕ) : ℕ := n + 1

def square (n : ℕ) : ℕ := n * n

-- Define a proposition that counts the number of ways to reach 1000 from 2 using these operations
def count_ways (start target : ℕ) : ℕ := sorry  -- We'll need a complex function to literally count the paths, but we'll abstract this here.

-- Theorem stating the number of ways to reach 1000
theorem naomi_number_of_ways_to_1000 : count_ways start 1000 = 128 := 
sorry

end naomi_number_of_ways_to_1000_l1316_131672


namespace intersection_of_A_and_B_l1316_131677

open Finset

def A : Finset ℤ := {-2, -1, 0, 1, 2}
def B : Finset ℤ := {1, 2, 3}

theorem intersection_of_A_and_B :
  A ∩ B = {1, 2} :=
by sorry

end intersection_of_A_and_B_l1316_131677


namespace leo_current_weight_l1316_131642

theorem leo_current_weight (L K : ℝ) 
  (h1 : L + 10 = 1.5 * K) 
  (h2 : L + K = 140) : 
  L = 80 :=
by 
  sorry

end leo_current_weight_l1316_131642


namespace four_friends_total_fish_l1316_131671

-- Define the number of fish each friend has based on the conditions
def micah_fish : ℕ := 7
def kenneth_fish : ℕ := 3 * micah_fish
def matthias_fish : ℕ := kenneth_fish - 15
def total_three_boys_fish : ℕ := micah_fish + kenneth_fish + matthias_fish
def gabrielle_fish : ℕ := 2 * total_three_boys_fish
def total_fish : ℕ := micah_fish + kenneth_fish + matthias_fish + gabrielle_fish

-- The proof goal
theorem four_friends_total_fish : total_fish = 102 :=
by
  -- We assume the proof steps are correct and leave the proof part as sorry
  sorry

end four_friends_total_fish_l1316_131671


namespace pie_chart_probability_l1316_131679

theorem pie_chart_probability
  (P_W P_X P_Z : ℚ)
  (h_W : P_W = 1/4)
  (h_X : P_X = 1/3)
  (h_Z : P_Z = 1/6) :
  1 - P_W - P_X - P_Z = 1/4 :=
by
  -- The detailed proof steps are omitted as per the requirement.
  sorry

end pie_chart_probability_l1316_131679


namespace proof_problem_l1316_131685

noncomputable def p : ℝ := -5 / 3
noncomputable def q : ℝ := -1

def A (p : ℝ) : Set ℝ := {x | 2 * x^2 + 3 * p * x + 2 = 0}
def B (q : ℝ) : Set ℝ := {x | 2 * x^2 + x + q = 0}

theorem proof_problem (h : (A p ∩ B q) = {1 / 2}) :
    p = -5 / 3 ∧ q = -1 ∧ (A p ∪ B q) = {-1, 1 / 2, 2} := by
  sorry

end proof_problem_l1316_131685


namespace power_multiplication_l1316_131678

variable (a : ℝ)

theorem power_multiplication : (-a)^3 * a^2 = -a^5 := 
sorry

end power_multiplication_l1316_131678


namespace part_a_avg_area_difference_part_b_prob_same_area_part_c_expected_value_difference_l1316_131655

-- Part (a)
theorem part_a_avg_area_difference : 
  let zahid_avg := (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2) / 6
  let yana_avg := (21 / 6)^2
  zahid_avg - yana_avg = 35 / 12 := sorry

-- Part (b)
theorem part_b_prob_same_area :
  let prob_zahid_min n := (13 - 2 * n) / 36
  let prob_same_area := (1 / 36) * ((11 / 36) + (9 / 36) + (7 / 36) + (5 / 36) + (3 / 36) + (1 / 36))
  prob_same_area = 1 / 24 := sorry

-- Part (c)
theorem part_c_expected_value_difference :
  let yana_avg := 49 / 4
  let zahid_avg := (11 / 36 * 1^2 + 9 / 36 * 2^2 + 7 / 36 * 3^2 + 5 / 36 * 4^2 + 3 / 36 * 5^2 + 1 / 36 * 6^2)
  (yana_avg - zahid_avg) = 35 / 9 := sorry

end part_a_avg_area_difference_part_b_prob_same_area_part_c_expected_value_difference_l1316_131655


namespace intersection_line_l1316_131646

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 3*x - y = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + y = 0

-- Define the line that we need to prove as the intersection
def line (x y : ℝ) : Prop := x - 2*y = 0

-- The theorem to prove
theorem intersection_line (x y : ℝ) : circle1 x y ∧ circle2 x y → line x y :=
by
  sorry

end intersection_line_l1316_131646


namespace percentage_ethanol_in_fuel_B_l1316_131643

-- Definitions from the conditions
def tank_capacity : ℝ := 218
def ethanol_percentage_fuel_A : ℝ := 0.12
def total_ethanol : ℝ := 30
def volume_of_fuel_A : ℝ := 122

-- Expression to calculate ethanol in Fuel A
def ethanol_in_fuel_A : ℝ := ethanol_percentage_fuel_A * volume_of_fuel_A

-- The remaining ethanol in Fuel B = Total ethanol - Ethanol in Fuel A
def ethanol_in_fuel_B : ℝ := total_ethanol - ethanol_in_fuel_A

-- The volume of fuel B used to fill the tank
def volume_of_fuel_B : ℝ := tank_capacity - volume_of_fuel_A

-- Statement to prove:
theorem percentage_ethanol_in_fuel_B : (ethanol_in_fuel_B / volume_of_fuel_B) * 100 = 16 :=
sorry

end percentage_ethanol_in_fuel_B_l1316_131643


namespace solution_set_l1316_131664

open Nat

def is_solution (a b c : ℕ) : Prop :=
  a ^ (b + 20) * (c - 1) = c ^ (b + 21) - 1

theorem solution_set (a b c : ℕ) : 
  (is_solution a b c) ↔ ((c = 0 ∧ a = 1) ∨ (c = 1)) := 
sorry

end solution_set_l1316_131664


namespace periodicity_iff_condition_l1316_131627

-- Define the given conditions
variable (f : ℝ → ℝ)
variable (h_even : ∀ x, f (-x) = f x)

-- State the problem
theorem periodicity_iff_condition :
  (∀ x, f (1 - x) = f (1 + x)) ↔ (∀ x, f (x + 2) = f x) :=
sorry

end periodicity_iff_condition_l1316_131627


namespace find_circle_radius_l1316_131657

/-- Eight congruent copies of the parabola y = x^2 are arranged in the plane so that each vertex 
is tangent to a circle, and each parabola is tangent to its two neighbors at an angle of 45°.
Find the radius of the circle. -/

theorem find_circle_radius
  (r : ℝ)
  (h_tangent_to_circle : ∀ (x : ℝ), (x^2 + r) = x → x^2 - x + r = 0)
  (h_single_tangent_point : ∀ (x : ℝ), (x^2 - x + r = 0) → ((1 : ℝ)^2 - 4 * 1 * r = 0)) :
  r = 1/4 :=
by
  -- the proof would go here
  sorry

end find_circle_radius_l1316_131657


namespace jacob_fraction_of_phoebe_age_l1316_131675

-- Definitions
def Rehana_current_age := 25
def Rehana_future_age (years : Nat) := Rehana_current_age + years
def Phoebe_future_age (years : Nat) := (Rehana_future_age years) / 3
def Phoebe_current_age := Phoebe_future_age 5 - 5
def Jacob_age := 3
def fraction_of_Phoebe_age := Jacob_age / Phoebe_current_age

-- Theorem statement
theorem jacob_fraction_of_phoebe_age :
  fraction_of_Phoebe_age = 3 / 5 :=
  sorry

end jacob_fraction_of_phoebe_age_l1316_131675


namespace parking_savings_l1316_131681

theorem parking_savings (weekly_cost : ℕ) (monthly_cost : ℕ) (weeks_in_year : ℕ) (months_in_year : ℕ)
  (h_weekly_cost : weekly_cost = 10)
  (h_monthly_cost : monthly_cost = 42)
  (h_weeks_in_year : weeks_in_year = 52)
  (h_months_in_year : months_in_year = 12) :
  weekly_cost * weeks_in_year - monthly_cost * months_in_year = 16 := 
by
  sorry

end parking_savings_l1316_131681


namespace radius_of_sphere_with_surface_area_4pi_l1316_131623

noncomputable def sphere_radius (surface_area: ℝ) : ℝ :=
  sorry

theorem radius_of_sphere_with_surface_area_4pi :
  sphere_radius (4 * Real.pi) = 1 :=
by
  sorry

end radius_of_sphere_with_surface_area_4pi_l1316_131623


namespace expand_product_l1316_131622

theorem expand_product (y : ℝ) : 3 * (y - 6) * (y + 9) = 3 * y^2 + 9 * y - 162 :=
by sorry

end expand_product_l1316_131622


namespace melissa_work_hours_l1316_131693

theorem melissa_work_hours (total_fabric : ℕ) (fabric_per_dress : ℕ) (hours_per_dress : ℕ) (total_num_dresses : ℕ) (total_hours : ℕ) 
  (h1 : total_fabric = 56) (h2 : fabric_per_dress = 4) (h3 : hours_per_dress = 3) : 
  total_hours = (total_fabric / fabric_per_dress) * hours_per_dress := by
  sorry

end melissa_work_hours_l1316_131693


namespace longer_diagonal_of_rhombus_l1316_131662

theorem longer_diagonal_of_rhombus (d1 d2 : ℝ) (area : ℝ) (h₁ : d1 = 12) (h₂ : area = 120) :
  d2 = 20 :=
by
  sorry

end longer_diagonal_of_rhombus_l1316_131662


namespace value_of_expression_l1316_131630

theorem value_of_expression 
  (x1 x2 x3 x4 x5 x6 x7 : ℝ)
  (h1 : x1 + 9*x2 + 25*x3 + 49*x4 + 81*x5 + 121*x6 + 169*x7 = 2)
  (h2 : 9*x1 + 25*x2 + 49*x3 + 81*x4 + 121*x5 + 169*x6 + 225*x7 = 24)
  (h3 : 25*x1 + 49*x2 + 81*x3 + 121*x4 + 169*x5 + 225*x6 + 289*x7 = 246) : 
  49*x1 + 81*x2 + 121*x3 + 169*x4 + 225*x5 + 289*x6 + 361*x7 = 668 := 
sorry

end value_of_expression_l1316_131630


namespace symmetry_about_origin_l1316_131654

noncomputable def f (x : ℝ) : ℝ := x * Real.log (-x)
noncomputable def g (x : ℝ) : ℝ := x * Real.log x

theorem symmetry_about_origin :
  ∀ x : ℝ, f (-x) = -g (-x) :=
by
  sorry

end symmetry_about_origin_l1316_131654


namespace future_skyscraper_climb_proof_l1316_131629

variable {H_f H_c H_fut : ℝ}

theorem future_skyscraper_climb_proof
  (H_f : ℝ)
  (H_c : ℝ := 3 * H_f)
  (H_fut : ℝ := 1.25 * H_c)
  (T_f : ℝ := 1) :
  (H_fut * T_f / H_f) > 2 * T_f :=
by
  -- specific calculations would go here
  sorry

end future_skyscraper_climb_proof_l1316_131629


namespace isosceles_triangle_perimeter_l1316_131684

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 6 ∨ a = 9) (h2 : b = 6 ∨ b = 9) (h : a ≠ b) : (a * 2 + b = 21 ∨ a * 2 + b = 24) :=
by
  sorry

end isosceles_triangle_perimeter_l1316_131684


namespace find_square_number_divisible_by_9_between_40_and_90_l1316_131609

theorem find_square_number_divisible_by_9_between_40_and_90 :
  ∃ x : ℕ, (∃ n : ℕ, x = n^2) ∧ (9 ∣ x) ∧ 40 < x ∧ x < 90 ∧ x = 81 :=
by
  sorry

end find_square_number_divisible_by_9_between_40_and_90_l1316_131609


namespace smaller_circle_radius_l1316_131615

theorem smaller_circle_radius (A1 A2 : ℝ) 
  (h1 : A1 + 2 * A2 = 25 * Real.pi) 
  (h2 : ∃ d : ℝ, A1 + d = A2 ∧ A2 + d = A1 + 2 * A2) : 
  ∃ r : ℝ, r^2 = 5 ∧ Real.pi * r^2 = A1 :=
by
  sorry

end smaller_circle_radius_l1316_131615


namespace determine_a_perpendicular_l1316_131610

theorem determine_a_perpendicular 
  (a : ℝ)
  (h1 : 2 * x + 3 * y + 5 = 0)
  (h2 : a * x + 3 * y - 4 = 0) 
  (h_perpendicular : ∀ x y, (2 * x + 3 * y + 5 = 0) → ∀ x y, (a * x + 3 * y - 4 = 0) → (-(2 : ℝ) / (3 : ℝ)) * (-(a : ℝ) / (3 : ℝ)) = -1) :
  a = -9 / 2 :=
by
  sorry

end determine_a_perpendicular_l1316_131610


namespace rectangle_area_l1316_131660

variable (x y : ℕ)

theorem rectangle_area
  (h1 : (x + 3) * (y - 1) = x * y)
  (h2 : (x - 3) * (y + 2) = x * y) :
  x * y = 36 :=
by
  -- Proof omitted
  sorry

end rectangle_area_l1316_131660


namespace negative_integer_solution_l1316_131639

theorem negative_integer_solution (M : ℤ) (h1 : 2 * M^2 + M = 12) (h2 : M < 0) : M = -4 :=
sorry

end negative_integer_solution_l1316_131639


namespace negation_of_exists_proposition_l1316_131676

theorem negation_of_exists_proposition :
  (¬ ∃ x : ℝ, x^2 - x + 1 = 0) ↔ ∀ x : ℝ, x^2 - x + 1 ≠ 0 :=
sorry

end negation_of_exists_proposition_l1316_131676


namespace pipe_ratio_l1316_131619

theorem pipe_ratio (A B : ℝ) (hA : A = 1 / 12) (hAB : A + B = 1 / 3) : B / A = 3 := by
  sorry

end pipe_ratio_l1316_131619


namespace exceeds_500_bacteria_l1316_131621

noncomputable def bacteria_count (n : Nat) : Nat :=
  4 * 3^n

theorem exceeds_500_bacteria (n : Nat) (h : 4 * 3^n > 500) : n ≥ 6 :=
by
  sorry

end exceeds_500_bacteria_l1316_131621


namespace common_divisors_count_48_80_l1316_131644

noncomputable def prime_factors_48 : Nat -> Prop
| n => n = 48

noncomputable def prime_factors_80 : Nat -> Prop
| n => n = 80

theorem common_divisors_count_48_80 :
  let gcd_48_80 := 2^4
  let divisors_of_gcd := [1, 2, 4, 8, 16]
  prime_factors_48 48 ∧ prime_factors_80 80 →
  List.length divisors_of_gcd = 5 :=
by
  intros
  sorry

end common_divisors_count_48_80_l1316_131644


namespace empty_seats_after_second_stop_l1316_131603

-- Definitions for the conditions described in the problem
def bus_seats : Nat := 23 * 4
def initial_people : Nat := 16
def first_stop_people_on : Nat := 15
def first_stop_people_off : Nat := 3
def second_stop_people_on : Nat := 17
def second_stop_people_off : Nat := 10

-- The theorem statement proving the number of empty seats
theorem empty_seats_after_second_stop : 
  (bus_seats - (initial_people + first_stop_people_on - first_stop_people_off + second_stop_people_on - second_stop_people_off)) = 57 :=
by
  sorry

end empty_seats_after_second_stop_l1316_131603


namespace ax_by_power5_l1316_131687

-- Define the real numbers a, b, x, and y
variables (a b x y : ℝ)

-- Define the conditions as assumptions
axiom axiom1 : a * x + b * y = 3
axiom axiom2 : a * x^2 + b * y^2 = 7
axiom axiom3 : a * x^3 + b * y^3 = 16
axiom axiom4 : a * x^4 + b * y^4 = 42

-- State the theorem to prove ax^5 + by^5 = 20
theorem ax_by_power5 : a * x^5 + b * y^5 = 20 :=
  sorry

end ax_by_power5_l1316_131687


namespace surface_area_is_correct_volume_is_approximately_correct_l1316_131686

noncomputable def surface_area_of_CXYZ (height : ℝ) (side_length : ℝ) : ℝ :=
  let area_CZX_CZY := 48
  let area_CXY := 9 * Real.sqrt 3
  let area_XYZ := 9 * Real.sqrt 15
  2 * area_CZX_CZY + area_CXY + area_XYZ

theorem surface_area_is_correct (height : ℝ) (side_length : ℝ) (h : height = 24) (s : side_length = 18) :
  surface_area_of_CXYZ height side_length = 96 + 9 * Real.sqrt 3 + 9 * Real.sqrt 15 :=
by
  sorry

noncomputable def volume_of_CXYZ (height : ℝ ) (side_length : ℝ) : ℝ :=
  -- Placeholder for the volume calculation approximation method.
  486

theorem volume_is_approximately_correct
  (height : ℝ) (side_length : ℝ) (h : height = 24) (s : side_length = 18) :
  volume_of_CXYZ height side_length = 486 :=
by
  sorry

end surface_area_is_correct_volume_is_approximately_correct_l1316_131686


namespace functional_equation_l1316_131698

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation (f_add : ∀ x y : ℝ, f (x + y) = f x + f y) (f_two : f 2 = 4) : f 1 = 2 :=
sorry

end functional_equation_l1316_131698
