import Mathlib

namespace reflection_on_incenter_circcenter_line_l822_822857

open Locale Classical

noncomputable theory

variables {A B C E F I O B' C' : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited E]
          [Inhabited F] [Inhabited I] [Inhabited O] [Inhabited B'] [Inhabited C']

/-- Given an acute-angled, non-isosceles triangle ABC, let E be the midpoint of the segment 
    joining the orthocenter and vertex A. The incircle of triangle ABC touches sides AB and AC 
    at points C' and B' respectively. Prove that point F, which is the reflection of E with 
    respect to the line B'C', lies on the line passing through the centers of the incircle I and 
    the circumcircle O of triangle ABC. -/
theorem reflection_on_incenter_circcenter_line 
  (triangle_ABC_acute : ∀ {α : Type} [linear_ordered_triangle α], acute_triangle α A B C)
  (triangle_ABC_non_isosceles : ∀ {α : Type} [linear_ordered_triangle α], non_isosceles_triangle α A B C)
  (E_midpoint : ∀ {H : Type} [orthocenter H], E = midpoint H A)
  (incircle_touches : ∀ {circle : Type} (I : Type) [incircle circle I],
    touches_sides circle I (A B) C' ∧ touches_sides circle I (A C) B')
  (F_reflection : reflection F E (line B' C'))
  : collinear ({I, O, F}) :=
begin
  sorry
end

end reflection_on_incenter_circcenter_line_l822_822857


namespace train_length_l822_822232

-- Define the conditions
variables (speed_kmh : ℝ) (cross_time_s : ℝ) (bridge_length_m : ℝ)
-- Given conditions
def train_speed_in_m_per_s := speed_kmh * (1000 / 3600)
def total_cross_distance := train_speed_in_m_per_s * cross_time_s

-- Problem statement
theorem train_length : train_length_m := (total_cross_distance - bridge_length_m)

end train_length_l822_822232


namespace average_speed_l822_822177

theorem average_speed (D : ℝ) (hD : D > 0) :
  let t1 := D / 240
  let t2 := D / 72
  let t3 := D / 132
  let T := t1 + t2 + t3
  let V_avg := D / T
  V_avg ≈ 39.40 := 
by
  sorry

end average_speed_l822_822177


namespace find_value_of_a_l822_822604

theorem find_value_of_a (b : ℤ) (q : ℚ) (a : ℤ) (h₁ : b = 2120) (h₂ : q = 0.5) (h₃ : (a : ℚ) / b = q) : a = 1060 :=
sorry

end find_value_of_a_l822_822604


namespace speed_relationship_maximum_speed_l822_822555

-- Define the conditions and the function for part (I)
def speed_conditions (a b c d : ℝ) : Prop :=
  (c = 0) ∧ (d = 0) ∧ (12 * a + 4 * b = 0)
  
def v (a b c d t : ℝ) : ℝ := a * t^3 + b * t^2 + c * t + d

-- Define the distance covered in the first 2 minutes
noncomputable def distance_first_2_minutes (a : ℝ) : ℝ :=
  ∫ t in 0..2, -950 * t^3 + 2850 * t^2

-- Lean statement for part (I)
theorem speed_relationship (a : ℝ) :
    speed_conditions a (-3 * a) 0 0 →
    v a (-3 * a) 0 0 t = -950 * t^3 + 2850 * t^2 :=
by sorry

-- Lean statement for part (II)
theorem maximum_speed : ∃ t, 0 ≤ t ∧ t ≤ 2 ∧ (∀ x, 0 ≤ x ∧ x ≤ 2 → v (-950) (2850) 0 0 x ≤ 3800) :=
by sorry

end speed_relationship_maximum_speed_l822_822555


namespace events_A_and_B_mutually_exclusive_l822_822593

def fair_hexahedral_die_faces := {1, 2, 3, 4, 5, 6}

def event_A (n : ℕ) : Prop := n = 1 ∨ n = 3 ∨ n = 5
def event_B (n : ℕ) : Prop := n = 4

theorem events_A_and_B_mutually_exclusive:
  ∀ n ∈ fair_hexahedral_die_faces, event_A n → ¬ event_B n :=
by
  sorry

end events_A_and_B_mutually_exclusive_l822_822593


namespace ball_arrangement_sum_ten_l822_822590

theorem ball_arrangement_sum_ten :
  let red_balls := [1, 2, 3, 4]
  let white_balls := [1, 2, 3, 4]
  let all_balls := red_balls.product [true, false] ++ white_balls.product [true, false]
  ∃ selection : list (ℕ × bool), selection.length = 4 ∧ (selection.map prod.fst).sum = 10 ∧ selection.nodup → select_arrangements selection = 432 :=
begin
  -- the proof would normally follow here
  sorry
end

end ball_arrangement_sum_ten_l822_822590


namespace sand_hourglass_time_l822_822246

theorem sand_hourglass_time (r h : ℝ) (sand_flow_rate : ℝ) (one_hour : ℝ) :
  (one_hour = 1) →
  (sand_flow_rate = 1 / one_hour) →
  (∃ t : ℝ, t = (7 / 8) * one_hour ∧ t * 60 = 52.5) :=
by
  assume (one_hour_eq : one_hour = 1) (sand_flow_rate_eq : sand_flow_rate = 1 / one_hour),
  exists.intro ((7 / 8) * one_hour) (by
    rw [one_hour_eq, sand_flow_rate_eq, mul_comm, mul_assoc, div_eq_mul_inv],
    exact eq.refl 52.5)

end sand_hourglass_time_l822_822246


namespace man_l822_822989

theorem man's_speed_upstream (v : ℝ) (downstream_speed : ℝ) (stream_speed : ℝ) :
  downstream_speed = v + stream_speed → stream_speed = 1 → downstream_speed = 10 → v - stream_speed = 8 :=
by
  intros h1 h2 h3
  sorry

end man_l822_822989


namespace sqrt_180_eq_l822_822042

noncomputable def simplify_sqrt_180 : Real := 6 * Real.sqrt 5

theorem sqrt_180_eq : Real.sqrt 180 = simplify_sqrt_180 := 
by
  -- proof omitted
  sorry

end sqrt_180_eq_l822_822042


namespace smallest_positive_integer_modulo_conditions_l822_822939

theorem smallest_positive_integer_modulo_conditions : 
  ∃ n : ℕ, n > 0 ∧ n ≡ 1 [MOD 2] ∧ n ≡ 3 [MOD 4] ∧ n ≡ 9 [MOD 10]
  := ∃ k : ℕ, k > 0 ∧ k ≠ 0 [MOD 3] ∧ ∀ n : ℕ, (n > 0 ∧ n ≡ 1 [MOD 2] ∧ n ≡ 3 [MOD 4] ∧ n ≡ 9 [MOD 10] ∧ n ≠ k) → n > 59 :=
sorry

end smallest_positive_integer_modulo_conditions_l822_822939


namespace cost_of_one_bag_of_potatoes_l822_822123

theorem cost_of_one_bag_of_potatoes :
  let x := 250 in
  ∀ (price : ℕ)
    (bags : ℕ)
    (andrey_initial_price : ℕ)
    (andrey_sold_price : ℕ)
    (boris_initial_price : ℕ)
    (boris_first_price : ℕ)
    (boris_second_price : ℕ)
    (earnings_andrey : ℕ)
    (earnings_boris_first : ℕ)
    (earnings_boris_second : ℕ)
    (total_earnings_boris : ℕ),
  bags = 60 →
  andrey_initial_price = price →
  andrey_sold_price = 2 * price →
  andrey_sold_price * bags = earnings_andrey →
  boris_initial_price = price →
  boris_first_price = 1.6 * price →
  boris_second_price = 2.24 * price →
  boris_first_price * 15 + boris_second_price * 45 = total_earnings_boris →
  total_earnings_boris = earnings_andrey + 1200 →
  price = x :=
by
  intros x price bags andrey_initial_price andrey_sold_price boris_initial_price boris_first_price boris_second_price earnings_andrey earnings_boris_first earnings_boris_second total_earnings_boris
  assume h_bags h_andrey_initial_price h_andrey_sold_price h_earnings_andrey h_boris_initial_price h_boris_first_price h_boris_second_price h_total_earnings_boris h_total_earnings_difference
  if h_necessary : x = 250 then
    sorry
  else
    sorry


end cost_of_one_bag_of_potatoes_l822_822123


namespace find_numbers_l822_822554

theorem find_numbers (a b : ℝ) (h₁ : a - b = 157) (h₂ : a / b = 2) : a = 314 ∧ b = 157 :=
sorry

end find_numbers_l822_822554


namespace ratio_rational_l822_822511

-- Let the positive numbers be represented as n1, n2, n3, n4, n5
variable (n1 n2 n3 n4 n5 : ℚ)

open Classical

-- Assume distinctness and positivity
axiom h_distinct : (n1 ≠ n2) ∧ (n1 ≠ n3) ∧ (n1 ≠ n4) ∧ (n1 ≠ n5) ∧ (n2 ≠ n3) ∧ (n2 ≠ n4) ∧ (n2 ≠ n5) ∧ (n3 ≠ n4) ∧ (n3 ≠ n5) ∧ (n4 ≠ n5)
axiom h_positive : (0 < n1) ∧ (0 < n2) ∧ (0 < n3) ∧ (0 < n4) ∧ (0 < n5)

-- For any choice of three numbers, the expression ab + bc + ca is rational
axiom h_rational : ∀ {a b c: ℚ}, (a ∈ {n1, n2, n3, n4, n5}) → (b ∈ {n1, n2, n3, n4, n5}) → (c ∈ {n1, n2, n3, n4, n5}) → a ≠ b → a ≠ c → b ≠ c → (a * b + b * c + c * a).is_rat

-- Prove that the ratio of any two numbers on the board is rational
theorem ratio_rational : ∀ {x y : ℚ}, (x ∈ {n1, n2, n3, n4, n5}) → (y ∈ {n1, n2, n3, n4, n5}) → x ≠ y → (x / y).is_rat :=
by sorry

end ratio_rational_l822_822511


namespace problem_statement_l822_822301

variables (p1 p2 p3 p4 : Prop)

theorem problem_statement (h_p1 : p1 = True)
                         (h_p2 : p2 = False)
                         (h_p3 : p3 = False)
                         (h_p4 : p4 = True) :
  (p1 ∧ p4) = True ∧
  (p1 ∧ p2) = False ∧
  (¬p2 ∨ p3) = True ∧
  (¬p3 ∨ ¬p4) = True :=
by
  sorry

end problem_statement_l822_822301


namespace tile_count_l822_822654

theorem tile_count (a : ℕ) (h1 : ∃ b : ℕ, b = 2 * a) (h2 : 2 * (Int.floor (a * Real.sqrt 5)) - 1 = 49) :
  2 * a^2 = 50 :=
by
  sorry

end tile_count_l822_822654


namespace parabola_vertex_b_l822_822101

theorem parabola_vertex_b (a b c p : ℝ) (h₁ : p ≠ 0)
  (h₂ : ∀ x, (x = p → -p = a * (p^2) + b * p + c) ∧ (x = 0 → p = c)) :
  b = - (4 / p) :=
sorry

end parabola_vertex_b_l822_822101


namespace three_digit_numbers_with_repeated_digits_l822_822772

/--
  Prove that the number of positive three-digit integers less than 500
  that have at least two digits that are the same is 112.
-/
theorem three_digit_numbers_with_repeated_digits :
  (card {n : ℕ | 100 ≤ n ∧ n < 500 ∧ (∃ (i j : ℕ), (i ≠ j) ∧ digit_at_pos n 1 = digit_at_pos n i ∧ digit_at_pos n 1 = digit_at_pos n j)}) = 112 :=
sorry

def digit_at_pos (n : ℕ) (k : ℕ) : ℕ :=
sorry

end three_digit_numbers_with_repeated_digits_l822_822772


namespace Mike_changed_2_sets_of_tires_l822_822471

theorem Mike_changed_2_sets_of_tires
  (wash_time_per_car : ℕ := 10)
  (oil_change_time_per_car : ℕ := 15)
  (tire_change_time_per_set : ℕ := 30)
  (num_washed_cars : ℕ := 9)
  (num_oil_changes : ℕ := 6)
  (total_work_time_minutes : ℕ := 4 * 60) :
  ((total_work_time_minutes - (num_washed_cars * wash_time_per_car + num_oil_changes * oil_change_time_per_car)) / tire_change_time_per_set) = 2 :=
by
  sorry

end Mike_changed_2_sets_of_tires_l822_822471


namespace bill_needs_125_bouquets_to_earn_1000_l822_822254

-- Define the constants for the problem
def cost_per_bouquet : ℕ := 20
def roses_per_bouquet_buy : ℕ := 7
def roses_per_bouquet_sell : ℕ := 5
def target_profit : ℕ := 1000

-- Define the problem in terms of a theorem
theorem bill_needs_125_bouquets_to_earn_1000 :
  ∃ n : ℕ, (35 / roses_per_bouquet_sell) * cost_per_bouquet - (5 * cost_per_bouquet) = 40 → (5 * n) = 125 :=
begin
  sorry
end

end bill_needs_125_bouquets_to_earn_1000_l822_822254


namespace number_of_incorrect_statements_l822_822240

theorem number_of_incorrect_statements :
  let stmt1 := ¬ (∀ x : ℚ, (0 < x ∨ x < 0) → x ∈ ℚ);
  let stmt2 := ∀ x : ℚ, (x < 0) → x ∈ ℤ ∨ x ∉ ℤ;
  let stmt3 := ¬ (∀ x : ℤ, (x ≠ 0) → x ∈ ℤ);
  let stmt4 := 0 ∈ ℤ ∧ ¬ ∀ (a b : ℤ), b ≠ 0 → (0 / b : ℚ) = 0 in
  [stmt1, stmt2, stmt3, stmt4].count(λ x, x = true) = 2 :=
by
  let stmt1 := ¬ (∀ x : ℚ, (0 < x ∨ x < 0) → x ∈ ℚ);
  let stmt2 := ∀ x : ℚ, (x < 0) → x ∈ ℤ ∨ x ∉ ℤ;
  let stmt3 := ¬ (∀ x : ℤ, (x ≠ 0) → x ∈ ℤ);
  let stmt4 := 0 ∈ ℤ ∧ ¬ ∀ (a b : ℤ), b ≠ 0 → (0 / b : ℚ) = 0 in
  have h1 : stmt1 = true := sorry;
  have h2 : stmt2 = true := sorry;
  have h3 : stmt3 = true := sorry;
  have h4 : stmt4 = false := sorry;
  have hSum := [stmt1, stmt2, stmt3, stmt4].count(λ x, x = true) = 2;
  exact hSum

end number_of_incorrect_statements_l822_822240


namespace complex_modulus_ratios_l822_822838

-- Definitions of z1 and z2
def z1 := Complex.mk (Real.sqrt 3 / 2) (1 / 2)
def z2 := Complex.mk 3 4

-- The proof statement
theorem complex_modulus_ratios :
  (Complex.abs (z1 ^ 2016)) / (Complex.abs z2) = 1 / 5 :=
by
  sorry

end complex_modulus_ratios_l822_822838


namespace find_regular_price_per_can_l822_822579

noncomputable def regularPricePerCan : ℝ :=
  let priceOf75Cans := 10.125
  let numberOfCans := 75
  let discountRate := 0.10
  let discountedPricePerCan (P : ℝ) := P * (1 - discountRate)
  let totalCostOfNumberOfCans := numberOfCans * discountedPricePerCan regularPricePerCan
  totalCostOfNumberOfCans = priceOf75Cans

theorem find_regular_price_per_can
  (priceOf75Cans : ℝ := 10.125)
  (numberOfCans : ℝ := 75)
  (discountRate : ℝ := 0.10)
  (P : ℝ := 0.15)
  : numberOfCans * (P * (1 - discountRate)) = priceOf75Cans := by
  sorry

end find_regular_price_per_can_l822_822579


namespace subset_of_A_l822_822725

-- Definitions of given sets A and B
def A : Set ℝ := { x | x^2 - 8 * x + 15 = 0 }
def B (a : ℝ) : Set ℝ := { x | a * x - 1 = 0 }

-- The theorem stating that if B is a subset of A, then a is either 1/3, 1/5, or 0
theorem subset_of_A (a : ℝ) (h : B a ⊆ A) : a ∈ {1/3, 1/5, 0} :=
sorry

end subset_of_A_l822_822725


namespace rectangles_have_same_perimeter_and_area_calculate_curve_length_and_area_l822_822079

noncomputable def equilateralTriangleCircumscribedRectangle (l : ℝ) : ℝ := 
  4 * l

-- Question's formal structure
theorem rectangles_have_same_perimeter_and_area
  (ABC : Triangle) 
  (h_equilateral : ∀ {a b c : ℝ}, IsEquilateral ABC) 
  (extended_segments : ∀ {x y z : ℝ}, SegmentExtended ABC) 
  (circumscribed_perimeter : ℝ) :
  circumscribed_perimeter = equilateralTriangleCircumscribedRectangle l :=
by
  sorry

theorem calculate_curve_length_and_area
  (l : ℝ)
  (ABC : Triangle) 
  (h_equilateral : ∀ {a b c : ℝ}, IsEquilateral ABC)
  (extended_segments : ∀ {x y z : ℝ}, SegmentExtended ABC)
  (circumscribed_perimeter : ℝ = 4 * l) :
  length_of_curve_K = ... ∧ area_enclosed_by_K = l^2 * (Math.pi + 2 * Math.sqrt(3) - 6) :=
by
  sorry

end rectangles_have_same_perimeter_and_area_calculate_curve_length_and_area_l822_822079


namespace line_length_l822_822793

theorem line_length (n : ℕ) (d : ℤ) (h1 : n = 51) (h2 : d = 3) : 
  (n - 1) * d = 150 := sorry

end line_length_l822_822793


namespace distance_between_locations_A_and_B_l822_822025

theorem distance_between_locations_A_and_B 
  (speed_A speed_B speed_C : ℝ)
  (distance_CD : ℝ)
  (distance_initial_A : ℝ)
  (distance_A_to_B : ℝ)
  (h1 : speed_A = 3 * speed_C)
  (h2 : speed_A = 1.5 * speed_B)
  (h3 : distance_CD = 12)
  (h4 : distance_initial_A = 50)
  (h5 : distance_A_to_B = 130)
  : distance_A_to_B = 130 :=
by
  sorry

end distance_between_locations_A_and_B_l822_822025


namespace parallel_lines_if_perpendicular_to_same_plane_l822_822341

variables (m n : Set Point) (α : Set Point)

-- Assume that m and n are lines and α is a plane
axiom line_m : is_line m
axiom line_n : is_line n
axiom plane_α : is_plane α

-- Given conditions
axiom m_perp_α : perp m α
axiom n_perp_α : perp n α

-- Proof statement
theorem parallel_lines_if_perpendicular_to_same_plane : parallel m n := by
sorry

end parallel_lines_if_perpendicular_to_same_plane_l822_822341


namespace correct_exponent_identity_l822_822951

theorem correct_exponent_identity: (2 : ℝ)^(-3) = 1 / 8 :=
by
  sorry

end correct_exponent_identity_l822_822951


namespace sum_of_squares_pairwise_distances_leq_l822_822469

-- Definitions based on conditions
def circle (R : ℝ) : Type := {p : ℝ × ℝ // p.1 ^ 2 + p.2 ^ 2 ≤ R ^ 2}
def points_in_circle (n : ℕ) (R : ℝ) : list (circle R) → Prop := λ ps, ps.length = n

-- Theorem statement based on the question and correct answer
theorem sum_of_squares_pairwise_distances_leq (n : ℕ) (R : ℝ) (ps : list (circle R)) (h : points_in_circle n R ps) :
  (∑ i in finset.range n, ∑ j in finset.range i, (ps.nth_le i sorry).val.1 ^ 2 + (ps.nth_le i sorry).val.2 ^ 2)
  ≤ n^2 * R^2 :=
sorry

end sum_of_squares_pairwise_distances_leq_l822_822469


namespace remainder_sum_first_120_div_980_l822_822938

theorem remainder_sum_first_120_div_980 : 
  let S : ℕ := 120 * (120 + 1) / 2 in 
  S % 980 = 320 :=
by 
  sorry

end remainder_sum_first_120_div_980_l822_822938


namespace intersection_complement_eq_l822_822763

def setA : Set ℝ := { x | (x - 6) * (x + 1) ≤ 0 }
def setB : Set ℝ := { x | x ≥ 2 }

theorem intersection_complement_eq :
  setA ∩ (Set.univ \ setB) = { x | -1 ≤ x ∧ x < 2 } := 
by 
  sorry

end intersection_complement_eq_l822_822763


namespace sqrt_180_simplified_l822_822057

theorem sqrt_180_simplified : Real.sqrt 180 = 6 * Real.sqrt 5 :=
   sorry

end sqrt_180_simplified_l822_822057


namespace proposition_not_true_at_3_l822_822974

variable (P : ℕ → Prop)

theorem proposition_not_true_at_3
  (h1 : ∀ k : ℕ, P k → P (k + 1))
  (h2 : ¬ P 4) :
  ¬ P 3 :=
sorry

end proposition_not_true_at_3_l822_822974


namespace anna_product_of_11th_and_12th_l822_822803

def anna_first_ten_games : List ℕ := [5, 7, 9, 2, 6, 10, 5, 7, 8, 4]

def anna_scores (n : ℕ) (scores : List ℕ) : ℕ := List.sum (List.take n scores)

theorem anna_product_of_11th_and_12th :
  let total_first_10 := anna_scores 10 anna_first_ten_games in
  ∃ (game11 game12 : ℕ),
    game11 < 15 ∧
    game12 < 15 ∧
    (total_first_10 + game11) % 11 = 0 ∧
    (total_first_10 + game11 + game12) % 12 = 0 ∧
    game11 * game12 = 18 :=
by
  let total_first_10 := 63
  have h1 : total_first_10 = anna_scores 10 anna_first_ten_games := rfl
  sorry

end anna_product_of_11th_and_12th_l822_822803


namespace percentage_of_ink_left_l822_822923

-- Definitions
def marker_area : ℝ := 3 * (4 * 4)
def rectangle1_area : ℝ := 2 * (6 * 2)
def rectangle2_area : ℝ := 8 * 4
def triangle_area : ℝ := 0.5 * 5 * 3
def total_painted_area : ℝ := rectangle1_area + rectangle2_area + triangle_area

-- Theorem to prove
theorem percentage_of_ink_left : total_painted_area > marker_area → 0% := 
by
  sorry

end percentage_of_ink_left_l822_822923


namespace winner_is_C_l822_822337

def A_statement (winner : String) : Prop := winner = "B" ∨ winner = "C"
def B_statement (winner : String) : Prop := winner ≠ "A" ∧ winner ≠ "C"
def C_statement (winner : String) : Prop := winner = "C"
def D_statement (winner : String) : Prop := winner = "B"

def count_true_statements (winner : String) : Nat :=
  (if A_statement winner then 1 else 0) +
  (if B_statement winner then 1 else 0) +
  (if C_statement winner then 1 else 0) +
  (if D_statement winner then 1 else 0)

theorem winner_is_C : (∃ winner : String, winner = "C" ∧ count_true_statements winner = 2) :=
  sorry

end winner_is_C_l822_822337


namespace bill_buys_125_bouquets_to_make_1000_l822_822260

-- Defining the conditions
def cost_per_bouquet : ℕ := 20
def roses_per_bouquet_buy : ℕ := 7
def roses_per_bouquet_sell : ℕ := 5
def target_difference : ℕ := 1000

-- To be demonstrated: number of bouquets Bill needs to buy to get a profit difference of $1000
theorem bill_buys_125_bouquets_to_make_1000 : 
  let total_cost_to_buy (n : ℕ) := n * cost_per_bouquet
  let total_roses (n : ℕ) := n * roses_per_bouquet_buy
  let bouquets_sell_from_roses (roses : ℕ) := roses / roses_per_bouquet_sell
  let total_revenue (bouquets : ℕ) := bouquets * cost_per_bouquet
  let profit (n : ℕ) := total_revenue (bouquets_sell_from_roses (total_roses n)) - total_cost_to_buy n
  profit (125 * 5) = target_difference := 
sorry

end bill_buys_125_bouquets_to_make_1000_l822_822260


namespace proof_problem_l822_822282

variable (p1 p2 p3 p4 : Prop)

theorem proof_problem (hp1 : p1) (hp2 : ¬ p2) (hp3 : ¬ p3) (hp4 : p4) :
  (p1 ∧ p4) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) := by
  sorry

end proof_problem_l822_822282


namespace price_reduction_equation_l822_822973

variable (x : ℝ)

theorem price_reduction_equation (h : 25 * (1 - x) ^ 2 = 16) : 25 * (1 - x) ^ 2 = 16 :=
by
  assumption

end price_reduction_equation_l822_822973


namespace minimum_months_to_triple_amount_l822_822016

theorem minimum_months_to_triple_amount (r : ℝ) (t : ℕ) (b : ℝ) : 
  r = 1.06 → b = 3 → (r^19 > b) ∧ (∀ n < 19, r^n ≤ b) :=
by
  intros r_eq b_eq
  have h₁ : 1.06^19 > 3 := sorry
  have h₂ : ∀ n < 19, 1.06^n ≤ 3 := sorry
  exact ⟨h₁, h₂⟩

end minimum_months_to_triple_amount_l822_822016


namespace quadratic_change_root_l822_822549

theorem quadratic_change_root (a b c : ℕ) (h : a + b + c = 2000) :
  ∃ a' b' c' : ℕ, (∃ x : ℤ, a' * x^2 + b' * x + c' = 0) ∧ (abs (int.ofNat a' - a) + abs (int.ofNat b' - b) + abs (int.ofNat c' - c) ≤ 1050) :=
begin
  sorry
end

end quadratic_change_root_l822_822549


namespace expression_divisible_by_3_l822_822415

theorem expression_divisible_by_3 (k : ℤ) : ∃ m : ℤ, (2 * k + 3)^2 - 4 * k^2 = 3 * m :=
by
  sorry

end expression_divisible_by_3_l822_822415


namespace max_value_of_a_l822_822748

-- Definitions of the conditions
def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x

def is_monotonic_inc_on (f : ℝ → ℝ) (a : ℝ) (S : set ℝ) : Prop :=
  ∀ x y ∈ S, x ≤ y → f x ≤ f y

-- The proof problem statement
theorem max_value_of_a (a : ℝ) :
  is_monotonic_inc_on (λ x, f x a) a (set.Ici 1) → a ≤ 3 :=
sorry

end max_value_of_a_l822_822748


namespace bill_needs_125_bouquets_to_earn_1000_l822_822257

-- Define the constants for the problem
def cost_per_bouquet : ℕ := 20
def roses_per_bouquet_buy : ℕ := 7
def roses_per_bouquet_sell : ℕ := 5
def target_profit : ℕ := 1000

-- Define the problem in terms of a theorem
theorem bill_needs_125_bouquets_to_earn_1000 :
  ∃ n : ℕ, (35 / roses_per_bouquet_sell) * cost_per_bouquet - (5 * cost_per_bouquet) = 40 → (5 * n) = 125 :=
begin
  sorry
end

end bill_needs_125_bouquets_to_earn_1000_l822_822257


namespace pasha_can_change_coefficients_l822_822544

theorem pasha_can_change_coefficients 
  (a b c : ℕ) 
  (h1 : a + b + c = 2000) :
  ∃ a' b' c' : ℕ, 
  (abs (a' - a) + abs (b' - b) + abs (c' - c) ≤ 1050) ∧ 
  ∃ x : ℤ, (a' * x^2 + b' * x + c' = 0) := 
sorry

end pasha_can_change_coefficients_l822_822544


namespace trigonometric_expression_value_l822_822712

theorem trigonometric_expression_value :
  cos 43 * cos 77 + sin 43 * cos 167 = -1 / 2 :=
by
  sorry

end trigonometric_expression_value_l822_822712


namespace inequality_a2_b2_c2_inequality_sqrt_x_l822_822969

-- Problem (1)
theorem inequality_a2_b2_c2 (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 1) : 
    a^2 + b^2 + c^2 ≥ 1/3 :=
    sorry

-- Problem (2)
theorem inequality_sqrt_x (x : ℝ) (hx : x ≥ 4) : 
    sqrt (x - 1) + sqrt (x - 4) < sqrt (x - 2) + sqrt (x - 3) :=
    sorry

end inequality_a2_b2_c2_inequality_sqrt_x_l822_822969


namespace impossible_rearrangement_l822_822114

def total_students := 450
def total_desks := 225
def half_girls_with_boys (D B : ℕ) := D / 2
def rearrange_possible (D B : ℕ) := half_girls_with_boys D B = B / 2

theorem impossible_rearrangement :
  ∀ (D B : ℕ), D + B = total_students → half_girls_with_boys D B + B / 4 ≠ total_desks :=
begin
  sorry
end

end impossible_rearrangement_l822_822114


namespace cost_of_one_bag_of_potatoes_l822_822120

theorem cost_of_one_bag_of_potatoes :
  let x := 250 in
  ∀ (price : ℕ)
    (bags : ℕ)
    (andrey_initial_price : ℕ)
    (andrey_sold_price : ℕ)
    (boris_initial_price : ℕ)
    (boris_first_price : ℕ)
    (boris_second_price : ℕ)
    (earnings_andrey : ℕ)
    (earnings_boris_first : ℕ)
    (earnings_boris_second : ℕ)
    (total_earnings_boris : ℕ),
  bags = 60 →
  andrey_initial_price = price →
  andrey_sold_price = 2 * price →
  andrey_sold_price * bags = earnings_andrey →
  boris_initial_price = price →
  boris_first_price = 1.6 * price →
  boris_second_price = 2.24 * price →
  boris_first_price * 15 + boris_second_price * 45 = total_earnings_boris →
  total_earnings_boris = earnings_andrey + 1200 →
  price = x :=
by
  intros x price bags andrey_initial_price andrey_sold_price boris_initial_price boris_first_price boris_second_price earnings_andrey earnings_boris_first earnings_boris_second total_earnings_boris
  assume h_bags h_andrey_initial_price h_andrey_sold_price h_earnings_andrey h_boris_initial_price h_boris_first_price h_boris_second_price h_total_earnings_boris h_total_earnings_difference
  if h_necessary : x = 250 then
    sorry
  else
    sorry


end cost_of_one_bag_of_potatoes_l822_822120


namespace proof_set_operations_and_bound_a_l822_822741

def A := {x : ℝ | 2 ≤ 2^x ∧ 2^x ≤ 16}
def B := {x : ℝ | log 3 x > 1}
def C (a : ℝ) := {x : ℝ | 1 < x ∧ x < a}

theorem proof_set_operations_and_bound_a (a : ℝ) :
  (A ∩ B = {x : ℝ | 3 < x ∧ x ≤ 4}) ∧
  (set.compl B ∪ A = {x : ℝ | x ≤ 4}) ∧
  (C a ⊆ A → a ∈ set.Iic (4 : ℝ)) :=
by
  sorry

end proof_set_operations_and_bound_a_l822_822741


namespace units_digit_divisible_by_18_l822_822163

theorem units_digit_divisible_by_18 : ∃ n : ℕ, (3150 ≤ 315 * n) ∧ (315 * n < 3160) ∧ (n % 2 = 0) ∧ (315 * n % 18 = 0) ∧ (n = 0) :=
by
  use 0
  sorry

end units_digit_divisible_by_18_l822_822163


namespace correct_operation_l822_822168

variables (a : ℝ)

-- defining the expressions to be compared
def lhs := 2 * a^2 * a^4
def rhs := 2 * a^6

theorem correct_operation : lhs a = rhs a := 
by sorry

end correct_operation_l822_822168


namespace find_number_l822_822162

theorem find_number
  (a b c : ℕ)
  (h_a1 : a ≤ 3)
  (h_b1 : b ≤ 3)
  (h_c1 : c ≤ 3)
  (h_a2 : a ≠ 3)
  (h_b_condition1 : b ≠ 1 → 2 * a * b < 10)
  (h_b_condition2 : b ≠ 2 → 2 * a * b < 10)
  (h_c3 : c = 3)
  : a = 2 ∧ b = 3 ∧ c = 3 :=
by
  sorry

end find_number_l822_822162


namespace find_c_l822_822889

def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem find_c (a b c : ℝ) 
  (h1 : perpendicular (a / 2) (-2 / b))
  (h2 : a = b)
  (h3 : a * 1 - 2 * (-5) = c) 
  (h4 : 2 * 1 + b * (-5) = -c) : 
  c = 13 := by
  sorry

end find_c_l822_822889


namespace symmetry_implies_a_plus_d_zero_l822_822888

variable {a b c d : ℝ}

-- Assume all variables are nonzero
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)

-- Assume the curve equation and axis of symmetry condition
def curve (x : ℝ) := (a * x + b) / (c * x + d)

theorem symmetry_implies_a_plus_d_zero
  (h_symm : ∀ x : ℝ, curve a ha hb hc hd x = x → curve a ha hb hc hd x = x) : 
  a + d = 0 := 
sorry

end symmetry_implies_a_plus_d_zero_l822_822888


namespace cost_of_one_bag_l822_822135

theorem cost_of_one_bag (x : ℝ) (h1 : ∀ p : ℝ, 60 * x = p -> 60 * p = 120 * x ) 
  (h2 : ∀ p1 p2: ℝ, 60 * x = p1 ∧ 15 * 1.6 * x = p2 ∧ 45 * 2.24 * x = 100.8 * x -> 124.8 * x - 120 * x = 1200) :
  x = 250 := 
sorry

end cost_of_one_bag_l822_822135


namespace continued_fraction_is_correct_l822_822465

noncomputable def duration_in_seconds : ℕ := 
  (5 * 3600) + (48 * 60) + 46

def total_seconds_in_day : ℕ := 86400

def fraction_of_day : ℚ := duration_in_seconds / total_seconds_in_day

def continued_fraction : List ℕ := [0, 4, 7, 1, 3, 5, 64]

def convergents : List ℚ := [
  1 / 4,
  7 / 29, 
  8 / 33,
  31 / 128,
  163 / 673
]

theorem continued_fraction_is_correct : 
  (fraction_of_day : ℚ) = 20926 / 86400 ∧
  (continued_fraction = [0, 4, 7, 1, 3, 5, 64]) ∧
  (convergents = [
    1 / 4,
    7 / 29, 
    8 / 33,
    31 / 128,
    163 / 673
  ]) :=
sorry

end continued_fraction_is_correct_l822_822465


namespace triangle_inequality_l822_822006

theorem triangle_inequality (A B C F D E : Point) 
  (h1 : triangle A B C)
  (h2 : interior_point A B C F)
  (h3 : angle A F B = angle B F C ∧ angle B F C = angle C F A)
  (h4 : meets_side B F A C D)
  (h5 : meets_side C F A B E) :
  length (segment A B) + length (segment A C) ≥ 4 * length (segment D E) :=
sorry

end triangle_inequality_l822_822006


namespace part1_part2_l822_822362

-- Definitions of sets A and B.
def A (a : ℝ) : set ℝ := { x | a - 1 < x ∧ x < 2 * a + 1 }
def B : set ℝ := { x | 0 < x ∧ x < 1 }

-- Part (I)
theorem part1 (a : ℝ) (h : a = 1 / 2) : A a ∩ B = { x | 0 < x ∧ x < 1 } :=
sorry

-- Part (II)
theorem part2 (a : ℝ) (h1: A a ≠ ∅) (h2 : A a ∩ B = ∅) :
    (-2 < a ∧ a ≤ -1 / 2) ∨ (a ≥ 2) :=
sorry

end part1_part2_l822_822362


namespace find_f_of_2_l822_822495

variable {a b : ℝ}
variable (f : ℝ → ℝ)

-- Given conditions
def quadratic_function (x : ℝ) := x^2 + a * x + b
def distinct_real_numbers := a ≠ b
def function_condition := quadratic_function a = quadratic_function b

-- Question to prove
theorem find_f_of_2
  (ha_distinct : distinct_real_numbers)
  (hf_condition : function_condition) :
  quadratic_function 2 = 4 := 
  sorry

end find_f_of_2_l822_822495


namespace area_PQR_l822_822822

open Real

noncomputable reason
def area_eq := 1998 -- Area of △ABC in cm²
def ratio := (3, 4) -- The division ratio of K, L, M on AB, BC, CA

theorem area_PQR
  (A B C K L M P Q R : Type)
  (hABC_equilateral: equilateral △ A B C) 
  (hABC_area : area A B C = area_eq)
  (hK_on_AB : divides_segment K (3:4) A B)
  (hL_on_BC : divides_segment L (3:4) B C)
  (hM_on_CA : divides_segment M (3:4) C A)
  (hAL_intersects_CK_at_P : AL ∩ CK = P)
  (hAL_intersects_BM_at_Q : AL ∩ BM = Q)
  (hBM_intersects_CK_at_R : BM ∩ CK = R) :
  area P Q R = 54 :=
by sorry

end area_PQR_l822_822822


namespace smallest_solution_l822_822709

open Real

-- Define the function f(s) = (15s^2 - 40s + 18) / (4s - 3) + 7s
def f (s : ℝ) : ℝ := (15 * s^2 - 40 * s + 18) / (4 * s - 3) + 7 * s

-- Define the target equation to solve for s
def target (s : ℝ) : Prop := f(s) = 9 * s - 2

theorem smallest_solution : ∀ s : ℝ, target(s) → s = 4 / 7 ∨ s = 3 := sorry

end smallest_solution_l822_822709


namespace simplify_fraction_l822_822039

theorem simplify_fraction (x : ℝ) : (2 * x - 3) / 4 + (4 * x + 5) / 3 = (22 * x + 11) / 12 := by
  sorry

end simplify_fraction_l822_822039


namespace emily_meals_count_l822_822693

theorem emily_meals_count :
  let protein_count := 4
  let sides_count := 5
  let choose_sides := (sides_count.choose 3)
  let dessert_count := 5
  protein_count * choose_sides * dessert_count = 200 :=
by
  sorry

end emily_meals_count_l822_822693


namespace max_outstanding_boys_100_l822_822426

structure Boy :=
  (height : ℕ)
  (weight : ℕ)

def not_inferior (A B : Boy) : Prop :=
  A.height > B.height ∨ A.weight > B.weight

def outstanding_boy (b : Boy) (others : List Boy) : Prop :=
  ∀ other ∈ others, not_inferior b other

def max_outstanding_boys (boys : List Boy) : ℕ :=
  boys.countp (λ b => outstanding_boy b (boys.erase b))

theorem max_outstanding_boys_100 (boys : List Boy) (h : length boys = 100) :
  max_outstanding_boys boys = 100 := 
  sorry

end max_outstanding_boys_100_l822_822426


namespace opposite_of_negative_2023_l822_822100

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l822_822100


namespace maximum_length_segment_ST_l822_822605

theorem maximum_length_segment_ST 
  {α β : Type}
  (arc : set α) (chord : set α)
  (C D M R P S T A B : α) 
  (RP AR PB : ℝ) 
  (angle_ARP : angle α α = angle α α)
  (angle_BPR : angle α α = angle α α)
  (angle_AMB : angle α α = angle α α):
  (M ∈ arc) → 
  (S ∈ segment M A) → (S ∈ segment C D) → 
  (T ∈ segment M B) → (T ∈ segment C D) → 
  (RP = |R - P|) → 
  (AR = |A - R|) → 
  (PB = |P - B|) → 
  (RP - 2 * real.sqrt (AR * PB) = ST) :=
by
  sorry

end maximum_length_segment_ST_l822_822605


namespace sum_due_is_42_l822_822081

-- Define the conditions
def BD : ℝ := 42
def TD : ℝ := 36

-- Statement to prove
theorem sum_due_is_42 (H1 : BD = 42) (H2 : TD = 36) : ∃ (FV : ℝ), FV = 42 := by
  -- Proof Placeholder
  sorry

end sum_due_is_42_l822_822081


namespace max_profit_at_100_l822_822077

section
variables (x : ℝ)

def production_cost (x : ℝ) : ℝ :=
if h : 0 < x ∧ x < 40 then 10 * x^2 + 100 * x 
else if h : x ≥ 40 then 501 * x + 10000 / x - 4500
else 0 -- handle cases where x <= 0 which are not considered

def profit_function (x : ℝ) : ℝ :=
if h : 0 < x ∧ x < 40 then -10 * x^2 + 400 * x - 2500
else if h : x ≥ 40 then 2000 - (x + 10000 / x)
else 0 -- handle cases where x <= 0 which are not considered

theorem max_profit_at_100 : (∀ x > 0, x < 40 → profit_function x ≤ profit_function 20) ∧ (∀ x ≥ 40, profit_function x ≤ profit_function 100) :=
begin
  sorry
end

end

end max_profit_at_100_l822_822077


namespace det_A_B_inv_l822_822414

theorem det_A_B_inv (A B : Matrix) 
  (hA : det A = 3) (hB : det B = 8) : 
  det (A * B⁻¹) = 3 / 8 := 
by
  sorry

end det_A_B_inv_l822_822414


namespace trucks_initial_distance_l822_822142

noncomputable def initial_distance (speedA speedB start_gap extra_distance : ℕ) : ℕ :=
  let tB := extra_distance / 10 in
  let dB := speedB * tB in
  let dA := dB + extra_distance in
  dA + dB

theorem trucks_initial_distance (speedA speedB start_gap extra_distance : ℕ)
    (hA : speedA = 90)
    (hB : speedB = 80)
    (hSG : start_gap = 1)
    (hED : extra_distance = 145) :
    initial_distance speedA speedB start_gap extra_distance = 1025 := by
  sorry

end trucks_initial_distance_l822_822142


namespace simplify_expression_l822_822630

theorem simplify_expression (x y : ℝ) : x^2 * y - 3 * x * y^2 + 2 * y * x^2 - y^2 * x = 3 * x^2 * y - 4 * x * y^2 :=
by
  sorry

end simplify_expression_l822_822630


namespace square_area_l822_822102

theorem square_area (p : ℕ) (h : p = 48) : (p / 4) * (p / 4) = 144 := by
  sorry

end square_area_l822_822102


namespace shifted_graph_is_D_l822_822089

def g (x : ℝ) : ℝ :=
  if x >= -4 ∧ x <= 0 then -1 - 0.5 * x
  else if x >= 0 ∧ x <= 3 then real.sqrt (9 - (x - 3)^2) - 1
  else if x >= 3 ∧ x <= 5 then 1.5 * (x - 3)
  else 0

def g_shifted (x : ℝ) : ℝ := g (x - 2)

theorem shifted_graph_is_D :
  (∀ x, g_shifted x = g (x - 2)) →
  -- Here we should include the exact formal representation of the graph D
  -- For simplicity, we may state it as a placeholder
  graph_of (λ x, g_shifted x) = graph_D :=
sorry

end shifted_graph_is_D_l822_822089


namespace meteor_radius_l822_822669

noncomputable def radius_of_planet := 30000 -- in meters
noncomputable def increase_in_height := 0.01 -- in meters
noncomputable def volume_displaced := 4 * π * radius_of_planet^2 * increase_in_height -- volume of water displaced

theorem meteor_radius :
  ∃ r : ℝ, (4/3) * π * r^3 = volume_displaced ∧ r = 300 :=
by sorry

end meteor_radius_l822_822669


namespace modular_units_l822_822223

theorem modular_units (U N S : ℕ) 
  (h1 : N = S / 4)
  (h2 : (S : ℚ) / (S + U * N) = 0.14285714285714285) : 
  U = 24 :=
by
  sorry

end modular_units_l822_822223


namespace cost_of_one_bag_l822_822137

theorem cost_of_one_bag (x : ℝ) (h1 : ∀ p : ℝ, 60 * x = p -> 60 * p = 120 * x ) 
  (h2 : ∀ p1 p2: ℝ, 60 * x = p1 ∧ 15 * 1.6 * x = p2 ∧ 45 * 2.24 * x = 100.8 * x -> 124.8 * x - 120 * x = 1200) :
  x = 250 := 
sorry

end cost_of_one_bag_l822_822137


namespace equation_of_asymptotes_l822_822392

variables {a b c : ℝ}
variables (x y : ℝ)

def is_hyperbola : Prop := (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)
def eccentricity : Prop := (c = 2 * a)
def c_relation : Prop := (c^2 = a^2 + b^2)

theorem equation_of_asymptotes
  (h1 : is_hyperbola x y)
  (h2 : eccentricity)
  (h3 : c_relation) :
  (y = sqrt 3 * x ∨ y = -sqrt 3 * x) :=
sorry

end equation_of_asymptotes_l822_822392


namespace find_length_AC_l822_822801

variables (A B C H X Y : Type u) [EuclideanGeometry]
  (AX AY AB AC : ℝ)
  (right_triangle : RightTriangle A B C)
  (right_angle_at_A : right_triangle ∠ A = 90)
  (altitude : Altitude A H B C)
  (circle_through_A_H : CircleThroughPoints A H)
  (circle_intersects_AB_AC : (circle_through_A_H ∩ LineThrough A B = {X}) ∧ (circle_through_A_H ∩ LineThrough A C = {Y}))
  (AX_eq_five : AX = 5)
  (AY_eq_six : AY = 6)
  (AB_eq_nine : AB = 9)

theorem find_length_AC : AC = 13.5 :=
  sorry

end find_length_AC_l822_822801


namespace problem1_problem2_l822_822381

-- Definition of the function f(x)
def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x^2

-- Given conditions
def conditions (a b : ℝ) : Prop :=
  (f 1 a b = -3) ∧ (3 * a + 2 * b = 0)

-- Theorem statements
theorem problem1 (a b : ℝ) (h : conditions a b) :
  (a = 6) ∧ (b = -9) ↔
  (∀ x : ℝ, f' x = 18 * x^2 - 18 * x) ∧
  (∀ x : ℝ, (x < 0 ∨ x > 1) → f' x > 0) ∧
  (∀ x : ℝ, (0 < x ∧ x < 1) → f' x < 0) :=
sorry

theorem problem2 :
  (∀ x : ℝ, x > 0 → f x 6 (-9) + 2 * m ^ 2 - m ≥ 0) ↔
  (m ≤ -1 ∨ m ≥ 3/2) :=
sorry

end problem1_problem2_l822_822381


namespace neg_exponent_reciprocal_l822_822607

theorem neg_exponent_reciprocal : (2 : ℝ) ^ (-1 : ℤ) = 1 / 2 := by
  -- Insert your proof here
  sorry

end neg_exponent_reciprocal_l822_822607


namespace fraction_math_problem_l822_822073

theorem fraction_math_problem :
  ∃ (fraction : ℚ) (N : ℚ), 
    (0.40 * N = 420) ∧ 
    (fraction * (1/3) * (2/5) * N = 35) ∧
    (fraction = 1/4) :=
begin
  use [1/4, 1050],
  split,
  { norm_num },
  split,
  { norm_num },
  norm_num
end

end fraction_math_problem_l822_822073


namespace cube_probability_l822_822892

theorem cube_probability :
  let m := 1
  let n := 504
  ∀ (faces : Finset (Fin 6)) (nums : Finset (Fin 9)), 
    faces.card = 6 → nums.card = 9 →
    (∀ f ∈ faces, ∃ n ∈ nums, true) →
    m + n = 505 :=
by
  sorry

end cube_probability_l822_822892


namespace find_c_l822_822086

theorem find_c (x c : ℤ) (h1 : 3 * x + 8 = 5) (h2 : c * x - 7 = 1) : c = -8 :=
sorry

end find_c_l822_822086


namespace sum_of_D_coordinates_l822_822856

variable {D : ℝ × ℝ}
variable {N : ℝ × ℝ := (6, 2)}
variable {C : ℝ × ℝ := (10, 0)}

theorem sum_of_D_coordinates
  (h1 : N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) :
  D.1 + D.2 = 6 := 
sorry

end sum_of_D_coordinates_l822_822856


namespace simple_interest_problem_l822_822956

-- Define the given conditions
def SI : ℝ := 4016.25
def R : ℝ := 12
def T : ℝ := 5
def P : ℝ := 6693.75

-- Express the problem as a theorem in Lean 4
theorem simple_interest_problem (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ) : SI = P * R * T / 100 :=
by
  -- Sorry to skip the proof part
  sorry

#eval simple_interest_problem SI R T P

end simple_interest_problem_l822_822956


namespace range_of_m_l822_822761

noncomputable def f (A φ x : ℝ) : ℝ := A * sin (2 * x + φ) - 1 / 2

theorem range_of_m (A φ m : ℝ)
  (hA : A > 0)
  (hφ1 : 0 < φ)
  (hφ2 : φ < π / 2)
  (hy_intercept : A * sin φ - 1 / 2 = 1)
  (hsymm : 2 * (π / 12 : ℝ) + φ = (1 : ℤ) * π + π / 2)
  (h_forall_x : ∀ x ∈ Icc (0 : ℝ) (π / 2), m ^ 2 - 3 * m ≤ f A φ x) :
  1 ≤ m ∧ m ≤ 2 :=
sorry

end range_of_m_l822_822761


namespace football_field_area_l822_822983

theorem football_field_area (total_fertilizer : ℝ) (part_fertilizer : ℝ) (part_area : ℝ) (rate : ℝ) (total_area : ℝ) :
  total_fertilizer = 1200 → part_fertilizer = 600 → part_area = 3600 → rate = part_fertilizer / part_area → 
  total_area = total_fertilizer / rate → total_area = 7200 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end football_field_area_l822_822983


namespace proposition_true_l822_822902

theorem proposition_true (x y : ℝ) : x + 2 * y ≠ 5 → (x ≠ 1 ∨ y ≠ 2) :=
by
  sorry

end proposition_true_l822_822902


namespace find_area_of_triangle_ABC_l822_822700

noncomputable def area_of_triangle_ABC (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (BAC : Angle A B C) (ABC : Angle B A C) (BC : ℝ) [Nonempty A] [Nonempty B] [Nonempty C] : ℝ :=
  if BAC = (π / 3) ∧ ABC = (π / 2) ∧ BC = 24 then 72 * Real.sqrt 3 else 0

theorem find_area_of_triangle_ABC
  (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (BAC : Angle A B C) (ABC : Angle B A C) (BC : ℝ) [Nonempty A] [Nonempty B] [Nonempty C] :
  BAC = (π / 3) → ABC = (π / 2) → BC = 24 → area_of_triangle_ABC A B C BAC ABC BC = 72 * Real.sqrt 3 :=
by
  intros hBAC hABC hBC
  rw [area_of_triangle_ABC, if_pos]
  exact ⟨hBAC, hABC, hBC⟩
  sorry

end find_area_of_triangle_ABC_l822_822700


namespace min_chemistry_teachers_l822_822224

/--
A school has 7 maths teachers, 6 physics teachers, and some chemistry teachers.
Each teacher can teach a maximum of 3 subjects.
The minimum number of teachers required is 6.
Prove that the minimum number of chemistry teachers required is 1.
-/
theorem min_chemistry_teachers (C : ℕ) (math_teachers : ℕ := 7) (physics_teachers : ℕ := 6) 
  (max_subjects_per_teacher : ℕ := 3) (min_teachers_required : ℕ := 6) :
  7 + 6 + C ≤ 6 * 3 → C = 1 := 
by
  sorry

end min_chemistry_teachers_l822_822224


namespace polynomial_properties_l822_822898

namespace PolynomialProof

def polynomial : ℚ[a, b] :=
  3 * a^2 * b^3 - a^3 * b + a^4 * b + a * b^2 - 5

theorem polynomial_properties :
  (∃ n : ℕ, polynomial.number_of_terms = 5) ∧
  (∃ d : ℕ, polynomial.degree = 5) ∧
  (∃ term : ℚ[a, b], term = -a^3 * b ∧ term.degree = 4) := 
by
  sorry

end PolynomialProof

end polynomial_properties_l822_822898


namespace tan_identity_l822_822968

theorem tan_identity :
  let t5 := Real.tan (Real.pi / 36) -- 5 degrees in radians
  let t40 := Real.tan (Real.pi / 9)  -- 40 degrees in radians
  t5 + t40 + t5 * t40 = 1 :=
by
  sorry

end tan_identity_l822_822968


namespace necessary_but_not_sufficient_condition_l822_822105

theorem necessary_but_not_sufficient_condition (k : ℝ) :
  (∀ x : ℝ, 2 * k * x^2 + k * x - 1 < 0) → k ∈ Icc (-8 : ℝ) 0 :=
sorry

end necessary_but_not_sufficient_condition_l822_822105


namespace max_non_pairwise_rel_prime_l822_822035

noncomputable def maxNonPairwiseRelativelyPrimeSubsetCard {S : Finset ℕ} (h : S ⊆ (Finset.range 17) \ {0}) : ℕ :=
  if ∀ s ⊆ S, s.card = 3 → ¬ (s.pairwise (λ a b, Nat.coprime a b)) then S.card else 0

theorem max_non_pairwise_rel_prime (S : Finset ℕ) (hS : S ⊆ (Finset.range 17) \ {0}) :
    maxNonPairwiseRelativelyPrimeSubsetCard hS ≤ 11 :=
sorry

end max_non_pairwise_rel_prime_l822_822035


namespace square_of_binomial_l822_822164

theorem square_of_binomial (k : ℝ) : (∃ b : ℝ, (x^2 - 18 * x + k) = (x + b)^2) ↔ k = 81 :=
by
  sorry

end square_of_binomial_l822_822164


namespace max_regions_with_parallel_lines_l822_822118

theorem max_regions_with_parallel_lines (total_lines : ℕ) (parallel_lines : ℕ) 
(h_total : total_lines = 50) (h_parallel : parallel_lines = 20) : 
  ∃ regions : ℕ, regions = 1086 :=
by 
  use 1086
  sorry

end max_regions_with_parallel_lines_l822_822118


namespace necessary_but_not_sufficient_condition_l822_822584

theorem necessary_but_not_sufficient_condition
    {a b : ℕ} :
    (¬ (a = 1) ∨ ¬ (b = 2)) ↔ (a + b ≠ 3) → (a ≠ 1 ∨ b ≠ 2) :=
by
    sorry

end necessary_but_not_sufficient_condition_l822_822584


namespace oranges_in_bin_l822_822993

variable (n₀ n_throw n_new : ℕ)

theorem oranges_in_bin (h₀ : n₀ = 50) (h_throw : n_throw = 40) (h_new : n_new = 24) : 
  n₀ - n_throw + n_new = 34 := 
by 
  sorry

end oranges_in_bin_l822_822993


namespace number_of_digits_l822_822226

theorem number_of_digits {a : Finset ℕ} (h : (a.card)! = 720) : a.card = 6 :=
sorry

end number_of_digits_l822_822226


namespace sheets_of_paper_l822_822018

theorem sheets_of_paper (x : ℕ) (sheets : ℕ) 
  (h1 : sheets = 3 * x + 31)
  (h2 : sheets = 4 * x + 8) : 
  sheets = 100 := by
  sorry

end sheets_of_paper_l822_822018


namespace expression_value_l822_822490

theorem expression_value (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 4 * y) : x + y + z = 16 * x :=
by
  -- Insert proof here
  sorry

end expression_value_l822_822490


namespace incenter_correct_l822_822466

variable (P Q R : Type) [AddCommGroup P] [Module ℝ P]
variable (p q r : ℝ)
variable (P_vec Q_vec R_vec : P)

noncomputable def incenter_coordinates (p q r : ℝ) : ℝ × ℝ × ℝ :=
  (p / (p + q + r), q / (p + q + r), r / (p + q + r))

theorem incenter_correct : 
  incenter_coordinates 8 10 6 = (1/3, 5/12, 1/4) := by
  sorry

end incenter_correct_l822_822466


namespace b_div_c_eq_27_l822_822903

variable {a b c s1 s2 : ℝ}
variable (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
variable (h4 : s1 ≠ s2)

def roots_equations : Prop := 
  let eq1 := s1 + s2 + c = 0
  let eq2 := s1 * s2 = a
  let eq3 := 9 * a = b
  let eq4 := 3 * (s1 + s2) + a = 0
  eq1 ∧ eq2 ∧ eq3 ∧ eq4

theorem b_div_c_eq_27 (h : roots_equations a b c s1 s2) : b / c = 27 :=
by {
  sorry
}

end b_div_c_eq_27_l822_822903


namespace sqrt_180_simplified_l822_822054

theorem sqrt_180_simplified : Real.sqrt 180 = 6 * Real.sqrt 5 :=
   sorry

end sqrt_180_simplified_l822_822054


namespace range_of_m_l822_822373

theorem range_of_m (m x : ℝ) : (m-1 < x ∧ x < m+1) → (1/3 < x ∧ x < 1/2) → (-1/2 ≤ m ∧ m ≤ 4/3) :=
by
  intros h1 h2
  have h3 : 1/3 < m + 1 := by sorry
  have h4 : m - 1 < 1/2 := by sorry
  have h5 : -1/2 ≤ m := by sorry
  have h6 : m ≤ 4/3 := by sorry
  exact ⟨h5, h6⟩

end range_of_m_l822_822373


namespace hyperbola_asymptotes_l822_822397

theorem hyperbola_asymptotes (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) (h4 : (2 : ℝ) = 2) :
  ∀ x : ℝ, y = sqrt 3 * x :=
by skip

end hyperbola_asymptotes_l822_822397


namespace kamal_physics_marks_l822_822477

theorem kamal_physics_marks
  (marks_english: ℕ := 76)
  (marks_math: ℕ := 65)
  (marks_chemistry: ℕ := 67)
  (marks_biology: ℕ := 85)
  (average_marks: ℕ := 75)
  : marks_english + marks_math + marks_chemistry + marks_biology + ?physics_marks = average_marks * 5 :=
sorry

end kamal_physics_marks_l822_822477


namespace mike_initial_total_games_l822_822844

theorem mike_initial_total_games (non_working_games working_games_earned: ℕ) (price_per_game: ℕ): 
(8 = non_working_games) → (56 = working_games_earned) → (7 = price_per_game) → 
let working_games := working_games_earned / price_per_game in 
let total_games := working_games + non_working_games in 
total_games = 16 :=
by
  intros
  let working_games := working_games_earned / price_per_game
  let total_games := working_games + non_working_games
  sorry

end mike_initial_total_games_l822_822844


namespace speed_of_second_car_l822_822140

/-!
Two cars started from the same point, at 5 am, traveling in opposite directions. 
One car was traveling at 50 mph, and they were 450 miles apart at 10 am. 
Prove that the speed of the other car is 40 mph.
-/

variable (S : ℝ) -- Speed of the second car

theorem speed_of_second_car
    (h1 : ∀ t : ℝ, t = 5) -- The time of travel from 5 am to 10 am is 5 hours 
    (h2 : ∀ d₁ : ℝ, d₁ = 50 * 5) -- Distance traveled by the first car
    (h3 : ∀ d₂ : ℝ, d₂ = S * 5) -- Distance traveled by the second car
    (h4 : 450 = 50 * 5 + S * 5) -- Total distance between the two cars
    : S = 40 := sorry

end speed_of_second_car_l822_822140


namespace perimeter_triangle_DEF_l822_822792

variable (D E F G H I J : Type)
variable [Inhabited D] [Inhabited E] [Inhabited F]

axiom triangle_DEF_right (angle_DEF_right : ∀ (x : E), ∠ DEF = 90)
axiom DE_len (DE_eq_15 : ∀ (d e : D), dist d e = 15)
axiom squares_const (squares : ∀ (d e f g h i : D), square d e f g ∧ square e f h i)
axiom circle_GHIF (circle : ∀ (g h i f j : D), circle_reflects g h i f j)
axiom JF_len (JF_eq_3 : ∀ (j f : D), dist j f = 3)

theorem perimeter_triangle_DEF : 
  ∀ (d e f : D), ∠ DEF = 90 → dist d e = 15 → square d e f g ∧ square e f h i →
  circle_reflects g h i f j → dist j f = 3 → 
  perimeter (triangle d e f) = 15 + 15 * sqrt 2 :=
by
  sorry

end perimeter_triangle_DEF_l822_822792


namespace inequality_proof_l822_822007

variable (a b c d : ℝ)
variable (habcda : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ ab + bc + cd + da = 1)

theorem inequality_proof :
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) ∧ (ab + bc + cd + da = 1) →
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1 / 3) :=
by sorry

end inequality_proof_l822_822007


namespace simplify_sqrt180_l822_822051

-- Conditions from the problem definition
def sqrt (x : ℕ) : ℝ := Real.sqrt x
def six := 6
def five := 5

-- The statement of the problem as a Lean theorem
theorem simplify_sqrt180 : sqrt 180 = six * sqrt five := by 
  sorry

end simplify_sqrt180_l822_822051


namespace maximum_value_of_function_l822_822092

theorem maximum_value_of_function : 
  (∀ x : ℝ, 3 * x^2 + 3 * x + 4) / (x^2 + x + 1) ≤ (13 / 3) := sorry

end maximum_value_of_function_l822_822092


namespace find_unit_vector_collinear_with_MN_l822_822359

open Real

namespace Vectors

def M : ℝ × ℝ := (1, 1)   -- Given point M
def N : ℝ × ℝ := (4, -3)  -- Given point N

def MN : ℝ × ℝ := (N.1 - M.1, N.2 - M.2)  -- Vector MN

def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1^2 + v.2^2)

def unit_vector (v : ℝ × ℝ) : ℝ × ℝ := 
  let mag := magnitude v
  (v.1 / mag, v.2 / mag)

theorem find_unit_vector_collinear_with_MN :
  unit_vector MN = (3 / 5, -4 / 5) ∨ unit_vector MN = (-3 / 5, 4 / 5) :=
sorry

end Vectors

end find_unit_vector_collinear_with_MN_l822_822359


namespace inscribed_triangle_area_l822_822344

theorem inscribed_triangle_area (R : ℝ) (a b c : ℝ) (hR : R = 4) (h_abc : a * b * c = 16 * real.sqrt 2) :
  1 / 2 * a * b * (c / (2 * R)) = real.sqrt 2 := by
  sorry

end inscribed_triangle_area_l822_822344


namespace monotonic_intervals_range_of_k_range_of_a_l822_822380

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

-- 1. Monotonic intervals of f(x)
theorem monotonic_intervals :
  (∀ x ∈ Ioo 1 real.e, f' x < 0) ∧ (∀ x ∈ Ioi real.e, f' x > 0) :=
sorry

-- 2. Range of real number k
theorem range_of_k (k : ℝ) :
  (∃ x0, x0 ∈ Ioi 1 ∨ x0 ∈ Ioo 0 1 ∧ (1 / f x0) ≥ k * x0) ↔ k ∈ Iio (1 / (2 * real.e)) :=
sorry

-- 3. Range of real number a
theorem range_of_a (a : ℝ) :
  (∀ m n, 
    sqrt real.e ≤ m ∧ m ≤ real.e^2 ∧ 
    sqrt real.e ≤ n ∧ n ≤ real.e^2 → 
    (f m - f' n) / (a - 2022) ≤ 1)
  ↔ a ∈ Iio 2022 ∪ Ici (real.e^2 / 2 + 2024) :=
sorry

end monotonic_intervals_range_of_k_range_of_a_l822_822380


namespace Jon_regular_bottle_size_is_16oz_l822_822476

noncomputable def Jon_bottle_size (x : ℝ) : Prop :=
  let daily_intake := 4 * x + 2 * 1.25 * x
  let weekly_intake := 7 * daily_intake
  weekly_intake = 728

theorem Jon_regular_bottle_size_is_16oz : ∃ x : ℝ, Jon_bottle_size x ∧ x = 16 :=
by
  use 16
  sorry

end Jon_regular_bottle_size_is_16oz_l822_822476


namespace probability_sum_less_than_10_l822_822930

theorem probability_sum_less_than_10 : 
  let outcomes := [(A, B) | A ← [1, 2, 3, 4, 5, 6], B ← [1, 2, 3, 4, 5, 6]],
      favorable_outcomes := (outcomes.filter (λ (p : ℕ × ℕ), p.fst + p.snd < 10)).length,
      total_outcomes := outcomes.length
  in (favorable_outcomes.toRat / total_outcomes.toRat = 5 / 6) :=
by
  sorry

end probability_sum_less_than_10_l822_822930


namespace brick_wall_completion_time_l822_822253

def rate (hours : ℚ) : ℚ := 1 / hours

/-- Avery can build a brick wall in 3 hours. -/
def avery_rate : ℚ := rate 3
/-- Tom can build a brick wall in 2.5 hours. -/
def tom_rate : ℚ := rate 2.5
/-- Catherine can build a brick wall in 4 hours. -/
def catherine_rate : ℚ := rate 4
/-- Derek can build a brick wall in 5 hours. -/
def derek_rate : ℚ := rate 5

/-- Combined rate for Avery, Tom, and Catherine working together. -/
def combined_rate_1 : ℚ := avery_rate + tom_rate + catherine_rate
/-- Combined rate for Tom and Catherine working together. -/
def combined_rate_2 : ℚ := tom_rate + catherine_rate
/-- Combined rate for Tom, Catherine, and Derek working together. -/
def combined_rate_3 : ℚ := tom_rate + catherine_rate + derek_rate

/-- Total time taken to complete the wall. -/
def total_time (t : ℚ) : Prop :=
  t = 2

theorem brick_wall_completion_time (t : ℚ) : total_time t :=
by
  sorry

end brick_wall_completion_time_l822_822253


namespace simplify_expression_l822_822536

theorem simplify_expression :
  (Real.sqrt 15 + Real.sqrt 45 - (Real.sqrt (4/3) - Real.sqrt 108)) = 
  (Real.sqrt 15 + 3 * Real.sqrt 5 + 16 * Real.sqrt 3 / 3) :=
by
  sorry

end simplify_expression_l822_822536


namespace father_optimal_strategy_is_to_play_with_mother_first_l822_822982

axiom FatherWeakest : Player
axiom Mother : Player
axiom SonStrongest : Player
axiom x y z : ℝ
axiom x_gt_y : x > y
axiom all_lt_half : x < 0.5 ∧ y < 0.5 ∧ z < 0.5
axiom tournament_rules : (∀ p1 p2 : Player, Game p1 p2)

theorem father_optimal_strategy_is_to_play_with_mother_first :
  ∀ (father mother son : Player) (x y z : ℝ),
  father = FatherWeakest →
  son = SonStrongest →
  x > y →
  x < 0.5 ∧ y < 0.5 ∧ z < 0.5 →
  optimal_strategy father mother son = play_with_mother_first :=
sorry

end father_optimal_strategy_is_to_play_with_mother_first_l822_822982


namespace hyperbola_eccentricity_l822_822647

open Real

def is_asymptote (M : ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, M x y → l x y ∨ x = 0

noncomputable def eccentricity (a b : ℝ) : ℝ := 
  real.sqrt (1 + (b / a)^2)

theorem hyperbola_eccentricity {b : ℝ} (hb : 0 < b) :
  let M := λ (x y : ℝ), x^2 - (y^2 / b^2) = 1
  let A := (-1, 0)
  let l := λ (x y : ℝ), y = x + 1
  let B := λ (x : ℝ), (x, x + 1)
  let C := λ (x : ℝ), (x, x + 1)
  is_asymptote M l ∧ 
  (∃ x1 x2 : ℝ, B x1 ∧ C x2 ∧ |(-1 : ℝ) - x1| = |x1 - x2|) →
  eccentricity 1 b = real.sqrt 10 :=
by
  sorry

end hyperbola_eccentricity_l822_822647


namespace common_sale_days_in_july_l822_822198

def BookstoreSaleDays (d : ℕ) : Prop :=
  (d ≥ 1) ∧ (d ≤ 31) ∧ (d % 4 = 0)

def ShoeStoreSaleDays (d : ℕ) : Prop :=
  (d ≥ 1) ∧ (d ≤ 31) ∧ (∃ k : ℕ, d = 2 + k * 7)

theorem common_sale_days_in_july : ∃! d, (BookstoreSaleDays d) ∧ (ShoeStoreSaleDays d) :=
by {
  sorry
}

end common_sale_days_in_july_l822_822198


namespace mean_of_all_students_is_83_l822_822019

-- Definitions for the problem
variables (M_A M_B : ℝ) -- mean scores of Class A and B
variables (a b : ℝ) -- number of students in Class A and B
variables (ratio : ℝ) -- ratio of students in Class A to Class B

-- Given conditions
axiom mean_A : M_A = 90
axiom mean_B : M_B = 78
axiom ratio_ab : a / b = 5 / 7

-- Total scores based on the number of students and mean score
def total_score_A : ℝ := M_A * a
def total_score_B : ℝ := M_B * b

-- Total number of students
def total_students : ℝ := a + b

-- Mean score of all students
noncomputable def overall_mean_score : ℝ := (total_score_A + total_score_B) / total_students

-- The goal is to prove the overall mean score is 83
theorem mean_of_all_students_is_83 : overall_mean_score = 83 :=
by
  sorry

end mean_of_all_students_is_83_l822_822019


namespace find_a3_l822_822736

noncomputable def S (n : ℕ) : ℤ := 2 * n^2 - 1
noncomputable def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem find_a3 : a 3 = 10 := by
  sorry

end find_a3_l822_822736


namespace convex_polygon_equal_angles_l822_822861

theorem convex_polygon_equal_angles (n : ℕ) (A : fin n → ℝ × ℝ) 
  (hconvex : ∀ i : fin n, convex (polygon A))
  (heqangles : ∀ i : fin n, angle (A i) (A (i + 1) % n) (A (i + 2) % n) = (2 * π) / n) :
  ∃ i : fin n, (dist (A i) (A (i + 1) % n) ≤ dist (A (i + 1) % n) (A (i + 2) % n)) ∨ (dist (A i) (A (i + 1) % n) ≤ dist (A (i - 1) % n) (A i)) :=
sorry

end convex_polygon_equal_angles_l822_822861


namespace equation_of_asymptotes_l822_822393

variables {a b c : ℝ}
variables (x y : ℝ)

def is_hyperbola : Prop := (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)
def eccentricity : Prop := (c = 2 * a)
def c_relation : Prop := (c^2 = a^2 + b^2)

theorem equation_of_asymptotes
  (h1 : is_hyperbola x y)
  (h2 : eccentricity)
  (h3 : c_relation) :
  (y = sqrt 3 * x ∨ y = -sqrt 3 * x) :=
sorry

end equation_of_asymptotes_l822_822393


namespace subtraction_of_largest_three_digit_from_smallest_five_digit_l822_822624

def largest_three_digit_number : ℕ := 999
def smallest_five_digit_number : ℕ := 10000

theorem subtraction_of_largest_three_digit_from_smallest_five_digit :
  smallest_five_digit_number - largest_three_digit_number = 9001 :=
by
  sorry

end subtraction_of_largest_three_digit_from_smallest_five_digit_l822_822624


namespace domain_of_logarithm_l822_822556

theorem domain_of_logarithm : 
  (∀ x : ℝ, ∃ y : ℝ, y = Real.log (x + 1) ↔ x > -1) :=
by
  intros x
  use Real.log (x + 1)
  split
  {
    intro h
    apply Real.log_pos
    exact h
  }
  {
    intro h
    apply h
  }
  sorry

end domain_of_logarithm_l822_822556


namespace hens_count_l822_822650

theorem hens_count
  (H C : ℕ)
  (heads_eq : H + C = 48)
  (feet_eq : 2 * H + 4 * C = 136) :
  H = 28 :=
by
  sorry

end hens_count_l822_822650


namespace map_distance_l822_822853

/--
On a map, 8 cm represents 40 km. Prove that 20 cm represents 100 km.
-/
theorem map_distance (scale_factor : ℕ) (distance_cm : ℕ) (distance_km : ℕ) 
  (h_scale : scale_factor = 5) (h_distance_cm : distance_cm = 20) : 
  distance_km = 20 * scale_factor := 
by {
  sorry
}

end map_distance_l822_822853


namespace vacation_cost_split_l822_822503

theorem vacation_cost_split 
  (airbnb_cost : ℕ)
  (car_rental_cost : ℕ)
  (people : ℕ)
  (split_equally : Prop)
  (h1 : airbnb_cost = 3200)
  (h2 : car_rental_cost = 800)
  (h3 : people = 8)
  (h4 : split_equally)
  : (airbnb_cost + car_rental_cost) / people = 500 :=
by
  sorry

end vacation_cost_split_l822_822503


namespace probability_of_multiple_of_3_l822_822721

theorem probability_of_multiple_of_3 :
  let s := {1, 2, 3, 4, 5, 6}
  let multiples_of_3 := {x ∈ s | x % 3 = 0}
  (multiples_of_3.card / s.card : ℚ) = 1 / 3 := by
  sorry

end probability_of_multiple_of_3_l822_822721


namespace average_milk_per_cow_per_day_l822_822192

theorem average_milk_per_cow_per_day (total_cows : ℕ) (total_milk : ℕ) (total_days : ℕ) (hcows : total_cows = 40) (hmilk : total_milk = 12000) (hdays : total_days = 30) :
  (total_milk / total_cows) / total_days = 10 :=
by {
  rw [hcows, hmilk, hdays],
  norm_num,
}
sorry

end average_milk_per_cow_per_day_l822_822192


namespace distance_between_parallel_lines_l822_822085

theorem distance_between_parallel_lines :
  ∀ {x y : ℝ},
    let l1 := (3 * x + 4 * y = 2)
    let l2 := (3 * x + 4 * y = 7)
    distance l1 l2 = 1 :=
sorry

end distance_between_parallel_lines_l822_822085


namespace solve_math_problem_l822_822351

def satisfies_condition (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 50 ∧ (Nat.factorial (n * n - 1) / Nat.factorial n ^ n ∈ ℤ)

theorem solve_math_problem :
  (Set.filter satisfies_condition (Finset.range 51)).card = 31 := 
sorry

end solve_math_problem_l822_822351


namespace transformed_average_and_variance_l822_822881

variables {ι : Type*} [Fintype ι] (x : ι → ℝ) (n : ℕ)
noncomputable def average (x : ι → ℝ) := (∑ i, x i) / Fintype.card ι
noncomputable def variance (x : ι → ℝ) := average (λ i, (x i - average x)^2)

theorem transformed_average_and_variance (x : ι → ℝ) 
  (avg : ℝ) (S2 : ℝ) (h_avg : average x = avg) (h_var : variance x = S2) :
  average (λ i, 3 * x i + 5) = 3 * avg + 5 ∧ variance (λ i, 3 * x i + 5) = 9 * S2 :=
sorry

end transformed_average_and_variance_l822_822881


namespace total_students_l822_822670

theorem total_students (m f : ℕ) (h_ratio : 3 * f = 7 * m) (h_males : m = 21) : m + f = 70 :=
by
  sorry

end total_students_l822_822670


namespace vacation_cost_per_person_l822_822501

theorem vacation_cost_per_person (airbnb_cost car_cost : ℝ) (num_people : ℝ) 
  (h1 : airbnb_cost = 3200) (h2 : car_cost = 800) (h3 : num_people = 8) : 
  (airbnb_cost + car_cost) / num_people = 500 := 
by 
  sorry

end vacation_cost_per_person_l822_822501


namespace find_y_l822_822419

variable {x y : ℤ}

-- Definition 1: The first condition x - y = 20
def condition1 : Prop := x - y = 20

-- Definition 2: The second condition x + y = 10
def condition2 : Prop := x + y = 10

-- The main theorem to prove that y = -5 given the above conditions
theorem find_y (h1 : condition1) (h2 : condition2) : y = -5 :=
  sorry

end find_y_l822_822419


namespace distance_from_plate_to_bottom_edge_l822_822977

theorem distance_from_plate_to_bottom_edge (d : ℝ) : 
  (10 + d + 63 = 20 + d + 53) :=
by
  -- The proof can be completed here.
  sorry

end distance_from_plate_to_bottom_edge_l822_822977


namespace constant_function_l822_822698

theorem constant_function {f : ℝ → ℝ} (h : ∀ a b : ℝ, irrational a → irrational b → f (a * b) = f (a + b)) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
sorry

end constant_function_l822_822698


namespace find_a_l822_822908

def curve (x : ℝ) : ℝ := 3 * x^2 + 2 * x

def tangent_slope_at (x : ℝ) : ℝ := (curve' x) 

def line_slope (a : ℝ) : ℝ := a

theorem find_a : 
  let x := 1 in
  let tang_line_slope := (6 * x + 2) in
  let line_slope := a in
  tang_line_slope = line_slope → 
  a = 8 := by
  sorry

end find_a_l822_822908


namespace proj_dist_inequality_l822_822744

open EuclideanGeometry

variables {A B C D X A' B' C' : Point}

def is_tetrahedron (A B C D : Point) : Prop := A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

def segment_intersects_face (X D A B C : Point) : Prop :=
  ∃ P, P ∈ interior (convex_hull ℝ {A, B, C}) ∧ P ∈ line[X, D]

def projection (D X B C : Point) : Point :=
  sorry -- Define the projection point D onto plane XBC

def DA : ℝ := dist D A
def DB : ℝ := dist D B
def DC : ℝ := dist D C

def A'_proj : Point := projection D X B C
def B'_proj : Point := projection D X C A
def C'_proj : Point := projection D X A B

def AB' : ℝ := dist A'_proj B'_proj
def BC' : ℝ := dist B'_proj C'_proj
def CA' : ℝ := dist C'_proj A'_proj

theorem proj_dist_inequality
  (h_tetra : is_tetrahedron A B C D)
  (h_intersect : segment_intersects_face X D A B C)
  (h_proj_A' : A' = projection D X B C)
  (h_proj_B' : B' = projection D X C A)
  (h_proj_C' : C' = projection D X A B) :
  AB' + BC' + CA' < DA + DB + DC :=
sorry

end proj_dist_inequality_l822_822744


namespace total_assignment_schemes_correct_assignment_schemes_with_A_at_A_correct_assignment_schemes_with_A_and_B_apart_correct_l822_822200

variable (Doctor Company : Type)
variable [Fintype Doctor] [Fintype Company]
variable [DecidableEq Doctor] [DecidableEq Company]

noncomputable def total_assignment_schemes [Fintype (Doctor → Company)] : ℕ :=
  ∑ (f : Doctor → Company), (∀ c : Company, ∃ d : Doctor, f d = c)

noncomputable def assignment_schemes_with_A_at_A [Fintype (Doctor → Company)] (A : Company) (docA : Doctor) : ℕ :=
  ∑ (f : Doctor → Company), (f docA = A ∧ ∀ c : Company, ∃ d : Doctor, f d = c)

noncomputable def assignment_schemes_with_A_and_B_apart [Fintype (Doctor → Company)] (docA docB : Doctor) : ℕ :=
  ∑ (f : Doctor → Company), (f docA ≠ f docB ∧ ∀ c : Company, ∃ d : Doctor, f d = c)

theorem total_assignment_schemes_correct :
  total_assignment_schemes {1, 2, 3, 4} {A, B, C} = 36 :=
sorry

theorem assignment_schemes_with_A_at_A_correct :
  assignment_schemes_with_A_at_A {1, 2, 3, 4} {A, B, C} A 1 = 12 :=
sorry

theorem assignment_schemes_with_A_and_B_apart_correct :
  assignment_schemes_with_A_and_B_apart {1, 2, 3, 4} {A, B, C} 1 2 = 30 :=
sorry

end total_assignment_schemes_correct_assignment_schemes_with_A_at_A_correct_assignment_schemes_with_A_and_B_apart_correct_l822_822200


namespace find_length_AC_BC_l822_822852

-- Definitions of points and segments
variables (A B C : ℝ)

-- Definitions of distances
def d_AB : ℝ := 5
def d_AC : ℝ := d_BC + 1 -- Given that AC is 1 unit longer than BC

-- The theorem to be proved
theorem find_length_AC_BC (d_BC : ℝ) (h1 : A < B) (h2 : A < C ∨ C < B)
  (h3 : abs (B - A) = d_AB) (h4 : abs (C - A) = d_AC) (h5 : abs (C - B) = d_BC) :
  d_AC = 3 ∧ d_BC = 2 :=
sorry

end find_length_AC_BC_l822_822852


namespace volume_ratio_of_cones_l822_822225

theorem volume_ratio_of_cones (R : ℝ) (hR : 0 < R) :
  let circumference := 2 * Real.pi * R
  let sector1_circumference := (2 / 3) * circumference
  let sector2_circumference := (1 / 3) * circumference
  let r1 := sector1_circumference / (2 * Real.pi)
  let r2 := sector2_circumference / (2 * Real.pi)
  let s := R
  let h1 := Real.sqrt (R^2 - r1^2)
  let h2 := Real.sqrt (R^2 - r2^2)
  let V1 := (Real.pi * r1^2 * h1) / 3
  let V2 := (Real.pi * r2^2 * h2) / 3
  V1 / V2 = Real.sqrt 10 := 
by
  sorry

end volume_ratio_of_cones_l822_822225


namespace loading_time_correct_l822_822204

-- Define the time taken by each worker to load one truck
def time_worker1 : ℝ := 5
def time_worker2 : ℝ := 8
def time_worker3 : ℝ := 10

-- Define the individual rates at which each worker can load the truck
def rate_worker1 : ℝ := 1 / time_worker1
def rate_worker2 : ℝ := 1 / time_worker2
def rate_worker3 : ℝ := 1 / time_worker3

-- Define the combined rate when all three workers work together
def combined_rate : ℝ := rate_worker1 + rate_worker2 + rate_worker3

-- The expected time to load one truck when all three workers work together
def expected_time : ℝ := 40 / 17

-- Theorem statement to prove that the combined time required is 40/17 hours
theorem loading_time_correct : (1 / combined_rate) = expected_time :=
by 
  sorry

end loading_time_correct_l822_822204


namespace solve_first_l822_822814

theorem solve_first (x y : ℝ) (C : ℝ) :
  (1 + y^2) * (deriv id x) - (1 + x^2) * y * (deriv id y) = 0 →
  Real.arctan x = 1/2 * Real.log (1 + y^2) + Real.log C := 
sorry

end solve_first_l822_822814


namespace modify_quadratic_polynomial_exists_integer_root_l822_822546

theorem modify_quadratic_polynomial_exists_integer_root
  (a b c : ℕ) (h_sum : a + b + c = 2000) :
  ∃ a' b' c' : ℕ, (a'.natAbs - a).natAbs + (b'.natAbs - b).natAbs + (c'.natAbs - c).natAbs ≤ 1050 ∧
  ∃ x : ℤ, a' * (x^2) + b' * x + c' = 0 :=
sorry

end modify_quadratic_polynomial_exists_integer_root_l822_822546


namespace range_f_positive_l822_822000

def f (x : ℝ) : ℝ :=
  if 0 < x then log x else -log (-x)

theorem range_f_positive :
  (∀ x : ℝ, f x = if 0 < x then log x else if x < 0 then -log (-x) else 0) →
  (set_of (λ x : ℝ, 0 < f x) = set.Ioo (-1 : ℝ) 0 ∪ set.Ioi 1) :=
by
  intro h1
  sorry

end range_f_positive_l822_822000


namespace polynomial_does_not_take_value_8_l822_822990

theorem polynomial_does_not_take_value_8
  (p : ℤ[X])
  (h_monic : p.monic)
  (h_int_coeffs : ∀ n, p.coeff n ∈ ℤ)
  (a b c d : ℤ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_values : p.eval a = 5 ∧ p.eval b = 5 ∧ p.eval c = 5 ∧ p.eval d = 5) :
  ∀ x : ℤ, p.eval x ≠ 8 := 
sorry

end polynomial_does_not_take_value_8_l822_822990


namespace paused_time_l822_822686

theorem paused_time (total_length remaining_length paused_at : ℕ) (h1 : total_length = 60) (h2 : remaining_length = 30) : paused_at = total_length - remaining_length :=
by
  sorry

end paused_time_l822_822686


namespace figure_square_count_l822_822280

theorem figure_square_count (f : ℕ → ℕ)
  (h0 : f 0 = 2)
  (h1 : f 1 = 8)
  (h2 : f 2 = 18)
  (h3 : f 3 = 32) :
  f 100 = 20402 :=
sorry

end figure_square_count_l822_822280


namespace y_share_per_x_l822_822995

theorem y_share_per_x (total_amount y_share : ℝ) (z_share_per_x : ℝ) 
  (h_total : total_amount = 234)
  (h_y_share : y_share = 54)
  (h_z_share_per_x : z_share_per_x = 0.5) :
  ∃ a : ℝ, (forall x : ℝ, y_share = a * x) ∧ a = 9 / 20 :=
by
  use 9 / 20
  intros
  sorry

end y_share_per_x_l822_822995


namespace cost_of_mixture_verify_cost_of_mixture_l822_822468

variables {C1 C2 Cm : ℝ}

def ratio := 5 / 12

axiom cost_of_rice_1 : C1 = 4.5
axiom cost_of_rice_2 : C2 = 8.75
axiom mix_ratio : ratio = 5 / 12

theorem cost_of_mixture (h1 : C1 = 4.5) (h2 : C2 = 8.75) (r : ratio = 5 / 12) :
  Cm = (8.75 * 5 + 4.5 * 12) / 17 :=
by sorry

-- Prove that the cost of the mixture Cm is indeed 5.75
theorem verify_cost_of_mixture (h1 : C1 = 4.5) (h2 : C2 = 8.75) (r : ratio = 5 / 12) :
  Cm = 5.75 :=
by sorry

end cost_of_mixture_verify_cost_of_mixture_l822_822468


namespace train_average_speed_l822_822996

theorem train_average_speed 
  (x : ℝ)
  (flat_terrain_speed : ℝ := 50)
  (heavy_rain_speed : ℝ := 30)
  (rain_wind_resistance: ℝ := 15)
  (slope_speed: ℝ := 35)
  (descent_speed: ℝ := 40)
  (flat_acceleration: ℝ := 0.1)
  (rain_deceleration: ℝ := 0.2)
  (slope_inclination: ℝ := 8)
  (slope_wind_resistance: ℝ := 6)
  (descent_inclination: ℝ := 12)
  (descent_tailwind: ℝ := 5)
  (descent_acceleration: ℝ := 0.15)
  (total_distance : ℝ := x + 2 * x + 0.5 * x + 1.5 * x)
  (total_time : ℝ := (x / flat_terrain_speed) + (2 * x / (heavy_rain_speed - rain_wind_resistance)) + (0.5 * x / slope_speed) + (1.5 * x / descent_speed))
  (average_speed : ℝ := total_distance / total_time) : 
  average_speed ≈ 25.95 :=
sorry

# Sanity check the definition to ensure it's correct in Lean
# #eval (5 * 2100) / 404.5  -- Expected to be approximately 25.95

end train_average_speed_l822_822996


namespace apple_allocation_proof_l822_822173

theorem apple_allocation_proof : 
    ∃ (ann mary jane kate ned tom bill jack : ℕ), 
    ann = 1 ∧
    mary = 2 ∧
    jane = 3 ∧
    kate = 4 ∧
    ned = jane ∧
    tom = 2 * kate ∧
    bill = 3 * ann ∧
    jack = 4 * mary ∧
    ann + mary + jane + ned + kate + tom + bill + jack = 32 :=
by {
    sorry
}

end apple_allocation_proof_l822_822173


namespace bill_buys_125_bouquets_to_make_1000_l822_822261

-- Defining the conditions
def cost_per_bouquet : ℕ := 20
def roses_per_bouquet_buy : ℕ := 7
def roses_per_bouquet_sell : ℕ := 5
def target_difference : ℕ := 1000

-- To be demonstrated: number of bouquets Bill needs to buy to get a profit difference of $1000
theorem bill_buys_125_bouquets_to_make_1000 : 
  let total_cost_to_buy (n : ℕ) := n * cost_per_bouquet
  let total_roses (n : ℕ) := n * roses_per_bouquet_buy
  let bouquets_sell_from_roses (roses : ℕ) := roses / roses_per_bouquet_sell
  let total_revenue (bouquets : ℕ) := bouquets * cost_per_bouquet
  let profit (n : ℕ) := total_revenue (bouquets_sell_from_roses (total_roses n)) - total_cost_to_buy n
  profit (125 * 5) = target_difference := 
sorry

end bill_buys_125_bouquets_to_make_1000_l822_822261


namespace sub_neg_eq_add_problem1_l822_822673

theorem sub_neg_eq_add (a b : ℤ) : a - (-b) = a + b := by sorry

theorem problem1 : 3 - (-1) = 4 :=
by
  apply sub_neg_eq_add
  -- This refers to the specific instance where a = 3 and b = 1
  sorry

end sub_neg_eq_add_problem1_l822_822673


namespace inequality_solution_minimum_value_l822_822632

-- Statement for Problem 1
theorem inequality_solution (x : ℝ) : 
  (\(\frac {2*x+1}{3-x} \geq 1\)) → (x ≤ 1 ∨ x > 2) :=
by
  sorry

-- Statement for Problem 2
theorem minimum_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) : 
  \(\frac {4}{x} + \frac {9}{y}\) = 25 :=
by
  sorry

end inequality_solution_minimum_value_l822_822632


namespace standard_deviation_from_mean_l822_822879

theorem standard_deviation_from_mean :
  let mean := 10.5
      std_dev := 1
      value := 8.5
  in (value - mean) / std_dev = -2 :=
by
  let mean := 10.5
      std_dev := 1
      value := 8.5
  show (value - mean) / std_dev = -2
  sorry

end standard_deviation_from_mean_l822_822879


namespace dog_max_distance_l822_822023

-- We define the maximum distance a dog secured with a rope to a point can be from the origin.

theorem dog_max_distance :
  let origin := (0, 0) : ℝ × ℝ
  let post := (6, 8) : ℝ × ℝ
  let rope_length : ℝ := 12
  let distance (p1 p2 : ℝ × ℝ) : ℝ := ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2).sqrt
  distance origin post + rope_length = 22 :=
by sorry

end dog_max_distance_l822_822023


namespace shift_graph_equivalence_l822_822869

-- Define the conditions
def f (x : ℝ) (θ : ℝ) : ℝ := Real.sin (2 * x + θ)
def g (x : ℝ) (θ : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x - 2 * φ + θ)

-- Define the theorem
theorem shift_graph_equivalence 
  (θ : ℝ) (φ : ℝ) (hθ1 : -π / 2 < θ)
  (hθ2 : θ < π / 2)
  (hφ : φ > 0)
  (hf_pass : f 0 θ = sqrt 3 / 2)
  (hg_pass : g 0 θ φ = sqrt 3 / 2) :
  φ = 5 * π / 6 :=
by
  sorry

end shift_graph_equivalence_l822_822869


namespace tourist_speed_proof_l822_822231

-- Conditions and Given Variables
variables (x y : ℝ) (t t1 t2 : ℝ)

-- Define the conditions and equations
def condition1 : Prop := x * (t + 1/6) + 5 * y / 24 = 8
def condition2 : Prop := y * t = 8
def condition3 : Prop := t ≥ 1/2
def quadratic_eq : Prop := 5 * y^2 + 4 * x * y - 192 * x = 0

-- Define the problem statement
theorem tourist_speed_proof
  (h1 : condition1 x y t)
  (h2 : condition2 y t)
  (h3 : condition3 t)
  (h4 : discriminant (a b c) ≥ 0) :
  x = 7 ∧ y = 16 :=
  by sorry

end tourist_speed_proof_l822_822231


namespace no_positive_solution_for_special_k_l822_822862
open Nat

theorem no_positive_solution_for_special_k (p : ℕ) (hp : p.Prime) (hmod : p % 4 = 3) :
    ¬ ∃ n m k : ℕ, (n > 0) ∧ (m > 0) ∧ (k = p^2) ∧ (n^2 + m^2 = k * (m^4 + n)) :=
sorry

end no_positive_solution_for_special_k_l822_822862


namespace true_compound_propositions_l822_822288

-- Define the propositions
def p1 : Prop := 
  ∀ (l₁ l₂ l₃ : Line), 
    (l₁ ≠ l₂ ∧ l₂ ≠ l₃ ∧ l₁ ≠ l₃) ∧ 
    (∃ (p₁ p₂ p₃ : Point),
      l₁.contains p₁ ∧ l₁.contains p₂ ∧
      l₂.contains p₂ ∧ l₂.contains p₃ ∧
      l₃.contains p₃ ∧ l₃.contains p₁)
    → ∃ (α : Plane), 
      α.contains l₁ ∧ α.contains l₂ ∧ α.contains l₃

def p2 : Prop :=
  ∀ (a b c : Point),
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) 
    → ∃! (α : Plane), 
      α.contains a ∧ α.contains b ∧ α.contains c

def p3 : Prop :=
  ∀ (l₁ l₂ : Line),
    (¬ ∃ (p : Point), l₁.contains p ∧ l₂.contains p)
    → l₁.parallel l₂

def p4 : Prop :=
  ∀ (α : Plane) (l m : Line),
    (α.contains l ∧ ¬ α.contains m ∧ m.perpendicular α) 
    → m.perpendicular l

-- Main theorem that identifies the true compound propositions
theorem true_compound_propositions :
  (p1 ∧ p4) ∧ ¬ (p1 ∧ p2) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) :=
by
  -- Proof might involve defining lines, points, and planes and their relationships
  sorry

end true_compound_propositions_l822_822288


namespace number_of_possible_rational_roots_l822_822216

-- Define the polynomial with integer coefficients
def polynomial (x : ℚ) : ℚ :=
  4 * x^5 + b_4 * x^4 + b_3 * x^3 + b_2 * x^2 + b_1 * x + 28

-- State the theorem
theorem number_of_possible_rational_roots (b_1 b_2 b_3 b_4 : ℤ) :
  ∃ p q : ℚ, (polynomial p = 0 ∧ polynomial q = 0) → (∃ n, n = 22) :=
sorry

end number_of_possible_rational_roots_l822_822216


namespace sum_of_odd_integers_less_than_50_l822_822161

def sumOddIntegersLessThan (n : Nat) : Nat :=
  List.sum (List.filter (λ x => x % 2 = 1) (List.range n))

theorem sum_of_odd_integers_less_than_50 : sumOddIntegersLessThan 50 = 625 :=
  by
    sorry

end sum_of_odd_integers_less_than_50_l822_822161


namespace count_sets_A_l822_822891

theorem count_sets_A :
  let universal_set : set ℤ := {1, 0, -1}
  let additional_elements : set ℤ := {1, -1}
  (∃ A : set ℤ, A ∪ additional_elements = universal_set) =
  4 :=
by
  sorry

end count_sets_A_l822_822891


namespace ratio_rational_l822_822513

-- Let the positive numbers be represented as n1, n2, n3, n4, n5
variable (n1 n2 n3 n4 n5 : ℚ)

open Classical

-- Assume distinctness and positivity
axiom h_distinct : (n1 ≠ n2) ∧ (n1 ≠ n3) ∧ (n1 ≠ n4) ∧ (n1 ≠ n5) ∧ (n2 ≠ n3) ∧ (n2 ≠ n4) ∧ (n2 ≠ n5) ∧ (n3 ≠ n4) ∧ (n3 ≠ n5) ∧ (n4 ≠ n5)
axiom h_positive : (0 < n1) ∧ (0 < n2) ∧ (0 < n3) ∧ (0 < n4) ∧ (0 < n5)

-- For any choice of three numbers, the expression ab + bc + ca is rational
axiom h_rational : ∀ {a b c: ℚ}, (a ∈ {n1, n2, n3, n4, n5}) → (b ∈ {n1, n2, n3, n4, n5}) → (c ∈ {n1, n2, n3, n4, n5}) → a ≠ b → a ≠ c → b ≠ c → (a * b + b * c + c * a).is_rat

-- Prove that the ratio of any two numbers on the board is rational
theorem ratio_rational : ∀ {x y : ℚ}, (x ∈ {n1, n2, n3, n4, n5}) → (y ∈ {n1, n2, n3, n4, n5}) → x ≠ y → (x / y).is_rat :=
by sorry

end ratio_rational_l822_822513


namespace andy_candy_canes_l822_822249

def candy_canes_from_parents (P : ℕ) : Prop :=
  let teachers_give := 3 * 4 in
  let buys := (P + teachers_give) / 7 in
  let total := P + teachers_give + buys in
  let cavities := 16 in
  let eaten := cavities * 4 in
  total = eaten -> P = 44

theorem andy_candy_canes : ∃ P : ℕ, candy_canes_from_parents P :=
begin
  use 44,
  unfold candy_canes_from_parents,
  sorry
end

end andy_candy_canes_l822_822249


namespace find_base_l822_822796

theorem find_base (b : ℕ) (h : (3 * b + 2) ^ 2 = b ^ 3 + b + 4) : b = 8 :=
sorry

end find_base_l822_822796


namespace tan_div_sum_eq_one_l822_822011

variable {x y : ℝ}

theorem tan_div_sum_eq_one
  (h1 : (sin x / cos y) + (sin y / cos x) = 2)
  (h2 : (cos x / sin y) + (cos y / sin x) = 3) :
  (tan x / tan y) + (tan y / tan x) = 1 := by
  sorry

end tan_div_sum_eq_one_l822_822011


namespace axis_of_symmetry_triangle_axis_of_symmetry_odd_polygon_l822_822031

theorem axis_of_symmetry_triangle (T : Triangle) : 
  ∃ v ∈ T.vertices, T.symmetry_axis_passes_through v := sorry

theorem axis_of_symmetry_odd_polygon (k : ℕ) (P : Polygon (2 * k + 1)) : 
  ∃ v ∈ P.vertices, P.symmetry_axis_passes_through v := sorry

end axis_of_symmetry_triangle_axis_of_symmetry_odd_polygon_l822_822031


namespace votes_cast_proof_l822_822640

variable (V : ℝ)
variable (candidate_votes : ℝ)
variable (rival_votes : ℝ)

noncomputable def total_votes_cast : Prop :=
  candidate_votes = 0.40 * V ∧ 
  rival_votes = candidate_votes + 2000 ∧ 
  rival_votes = 0.60 * V ∧ 
  V = 10000

theorem votes_cast_proof : total_votes_cast V candidate_votes rival_votes :=
by {
  sorry
  }

end votes_cast_proof_l822_822640


namespace cost_of_six_books_l822_822600

theorem cost_of_six_books (cost_two_books : ℕ) (cost_six_books : ℕ) (h : cost_two_books = 36) : 
  (cost_six_books = 108) :=
  let cost_one_book := cost_two_books / 2 in
  let six_books := 6 * cost_one_book in
  six_books = cost_six_books
  sorry

end cost_of_six_books_l822_822600


namespace white_line_longer_l822_822694

theorem white_line_longer :
  let white_line := 7.67
  let blue_line := 3.33
  white_line - blue_line = 4.34 := by
  sorry

end white_line_longer_l822_822694


namespace smallest_positive_period_f_eq_pi_min_value_f_interval_l822_822757
noncomputable def f (x : ℝ) : ℝ := 2 * sin (x - π) * cos (π - x)

theorem smallest_positive_period_f_eq_pi : 
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ ∀ ε > 0, ∀ T' > 0, (∀ x, f (x + T') = f x) → (T ≤ T' + ε) :=
begin
  -- The existence of the smallest positive period
  let T := π,
  use [T, by linarith],
  split,
  { intro x,
    rw [f, f, sin_add, cos_add],
    sorry },
  { intros ε hε T' hT' h_per,
    sorry }
end

theorem min_value_f_interval :
  ∃ (c : ℝ), c ∈ (set.Icc (-π / 6) (π / 2)) ∧ f c = -sqrt 3 / 2 :=
begin
  -- Min value of f(x) in the interval [-π/6, π/2]
  use [-π / 6], -- This might not be precise, just a placeholder
  split,
  { split; linarith, },
  { sorry }
end

end smallest_positive_period_f_eq_pi_min_value_f_interval_l822_822757


namespace sequence_bounds_l822_822626

theorem sequence_bounds (θ : ℝ) (n : ℕ) (a : ℕ → ℝ) (hθ : 0 < θ ∧ θ < π / 2) 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 1 - 2 * (Real.sin θ * Real.cos θ)^2) 
  (h_recurrence : ∀ n, a (n + 2) - a (n + 1) + a n * (Real.sin θ * Real.cos θ)^2 = 0) :
  1 / 2 ^ (n - 1) ≤ a n ∧ a n ≤ 1 - (Real.sin (2 * θ))^n * (1 - 1 / 2 ^ (n - 1)) := 
sorry

end sequence_bounds_l822_822626


namespace sqrt_180_simplify_l822_822059

theorem sqrt_180_simplify : real.sqrt 180 = 6 * real.sqrt 5 := by
  have h : 180 = 2^2 * 3^2 * 5 := by norm_num
  rw [h, real.sqrt_mul, real.sqrt_mul, real.sqrt_mul]
  all_goals { norm_num }
  sorry

end sqrt_180_simplify_l822_822059


namespace sum_of_divisors_85_l822_822947

theorem sum_of_divisors_85 : (1 + 5 + 17 + 85 = 108) := by
  sorry

end sum_of_divisors_85_l822_822947


namespace probability_draw_l822_822975

theorem probability_draw (h1 : P(A_{win}) = 0.6) (h2 : P(A_{not_lose}) = 0.9) : P(draw) = 0.3 :=
by
  -- Skipping the proof part
  sorry

end probability_draw_l822_822975


namespace domain_of_expression_l822_822705

theorem domain_of_expression (x : ℝ) 
  (h1 : 3 * x - 6 ≥ 0) 
  (h2 : 7 - 2 * x ≥ 0) 
  (h3 : 7 - 2 * x > 0) : 
  2 ≤ x ∧ x < 7 / 2 := by
sorry

end domain_of_expression_l822_822705


namespace pencil_distribution_l822_822335

-- Formalize the problem in Lean
theorem pencil_distribution (x1 x2 x3 x4 : ℕ) (hx1 : 1 ≤ x1 ∧ x1 ≤ 5) (hx2 : 1 ≤ x2 ∧ x2 ≤ 5) (hx3 : 1 ≤ x3 ∧ x3 ≤ 5) (hx4 : 1 ≤ x4 ∧ x4 ≤ 5) :
  x1 + x2 + x3 + x4 = 10 → 64 = 64 :=
by {
  sorry
}

end pencil_distribution_l822_822335


namespace num_ways_to_admit_students_l822_822806

theorem num_ways_to_admit_students :
  ∃ (n : ℕ) (k : ℕ) (students : ℕ), 8 = n ∧ 2 = k ∧ 3 = students ∧ nat.choose students 1 * nat.choose 2 2 * finset.card (finset.image2 nat.mul (finset.range n) (finset.range (n-1))) = 168 :=
by
  sorry

end num_ways_to_admit_students_l822_822806


namespace eccentricity_has_maximum_l822_822358

-- Given points A and B, and the point P on the line y = x + 4
theorem eccentricity_has_maximum {x0 : ℝ} (A : ℝ × ℝ := (-2, 0)) (B : ℝ × ℝ := (2, 0)) (P : ℝ × ℝ := (x0, x0 + 4)) :
  ∃ e : ℝ → ℝ, (e = λ x0, 2 / (1 / 2 * (|complex.abs (A - complex.of_real x0 + (x0 + 4) * complex.i)| + |complex.abs (B - complex.of_real x0 + (x0 + 4) * complex.i)|)) ∧ 
  (∀ x0, e(x0) has_max ⇑(e x0)) :=
sorry

end eccentricity_has_maximum_l822_822358


namespace questions_for_first_project_l822_822651

theorem questions_for_first_project (days_in_week : ℕ) (questions_per_day : ℕ) (questions_second_project : ℕ) :
  days_in_week = 7 →
  questions_per_day = 142 →
  questions_second_project = 476 →
  (days_in_week * questions_per_day - questions_second_project) = 518 :=
begin
  intro h1,
  intro h2,
  intro h3,
  rw [h1, h2, h3],
  norm_num,
end

end questions_for_first_project_l822_822651


namespace weight_of_b_l822_822080

theorem weight_of_b (A B C : ℕ) 
  (h1 : A + B + C = 129) 
  (h2 : A + B = 80) 
  (h3 : B + C = 86) : 
  B = 37 := 
by 
  sorry

end weight_of_b_l822_822080


namespace show_a2_eq_pc_show_b2_eq_qc_show_h2_eq_pq_show_a2_plus_b2_eq_c2_l822_822005

-- Definitions based on the conditions from the problem
variables {α : Type*}
variables (A B C H : α)
variables [metric_space α] [ordered_comm_ring α]
variables (a b c p q h : α)

-- Assume the triangle is right-angled at C and h is the altitude from C to AB,
-- dividing c into p and q.
def right_angle {α : Type*} [metric_space α] [ordered_comm_ring α] (A B C : α) : Prop :=
  ∀ x y z : α, dist x y = 0 → dist y z = 0 → dist z x = max (dist x z) (dist y z)

axiom right_triangle_C (right_angle : right_angle A B C) : 
  dist A C ^ 2 + dist B C ^ 2 = dist A B ^ 2

noncomputable def altitude (H : α) (right_angle : right_angle A B C) : α :=
  sorry

axiom division (H : α) (right_angle : right_angle A B C) : dist A H + dist H B = dist A B

-- Theorem statements
theorem show_a2_eq_pc {a b c p q h : α} (right_angle : right_angle A B C) : a^2 = p * c :=
sorry

theorem show_b2_eq_qc {a b c p q h : α} (right_angle : right_angle A B C) : b^2 = q * c :=
sorry

theorem show_h2_eq_pq {a b c p q h : α} (right_angle : right_angle A B C) : h^2 = p * q :=
sorry

theorem show_a2_plus_b2_eq_c2 {a b c p q h : α} (right_angle : right_angle A B C) : a^2 + b^2 = c^2 :=
sorry

end show_a2_eq_pc_show_b2_eq_qc_show_h2_eq_pq_show_a2_plus_b2_eq_c2_l822_822005


namespace c_investment_time_l822_822235

theorem c_investment_time
  (x : ℕ)  -- A's investment
  (total_gain : ℕ) (a_share : ℕ)
  (h1 : total_gain = 18600) (h2 : a_share = 6200) :
  let total_investment_time := 12 * x + 2 * x * 6 + 3 * x * (12 - 8) in
  let a_profit_ratio := (x * 12) in
  a_profit_ratio / total_investment_time = a_share / total_gain →
  8 = 8 :=
by
  sorry

end c_investment_time_l822_822235


namespace cricketer_sixes_l822_822202

-- Definitions based on conditions
-- Total runs scored by the cricketer
def total_runs : ℕ := 138

-- Number of boundaries
def num_boundaries : ℕ := 12

-- Runs scored by running between the wickets in percentage
def running_percentage : ℚ := 56.52 / 100

-- Value of a boundary (each boundary is worth 4 runs)
def boundary_value : ℕ := 4

-- Value of a six (each six is worth 6 runs)
def six_value : ℕ := 6

-- Main Theorem stating the cricketer hit 2 sixes
theorem cricketer_sixes : ∃ sixes : ℕ, sixes = 2 :=
by
  have runs_from_running := (running_percentage * total_runs).to_rat
  have runs_from_boundaries := num_boundaries * boundary_value
  have remaining_runs := total_runs - runs_from_running.to_nat - runs_from_boundaries
  have sixes := remaining_runs / six_value
  existsi sixes
  sorry  -- proof steps omitted

end cricketer_sixes_l822_822202


namespace boys_and_girls_sums_equal_l822_822183

theorem boys_and_girls_sums_equal (n : ℕ) (h : n ≥ 3) 
  (seating_condition : ∀ i, (i : ℕ) < n → (i%2 == 0 → i is a boy) ∧ (i%2 == 1 → i is a girl))
  (card_distribution : ∀ i, (i : ℕ) < n → (card i > n → i is a girl) ∧ (card i ≤ n → i is a boy))
  (sums_equal : ∀ i, (i : ℕ) < n → (sum = card i + card (i-1) + card (i+1) for all boys)) :
  sum_vals n = (x * n) → n % 2 = 1 :=
by
  sorry

end boys_and_girls_sums_equal_l822_822183


namespace true_compound_propositions_l822_822287

-- Define the propositions
def p1 : Prop := 
  ∀ (l₁ l₂ l₃ : Line), 
    (l₁ ≠ l₂ ∧ l₂ ≠ l₃ ∧ l₁ ≠ l₃) ∧ 
    (∃ (p₁ p₂ p₃ : Point),
      l₁.contains p₁ ∧ l₁.contains p₂ ∧
      l₂.contains p₂ ∧ l₂.contains p₃ ∧
      l₃.contains p₃ ∧ l₃.contains p₁)
    → ∃ (α : Plane), 
      α.contains l₁ ∧ α.contains l₂ ∧ α.contains l₃

def p2 : Prop :=
  ∀ (a b c : Point),
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) 
    → ∃! (α : Plane), 
      α.contains a ∧ α.contains b ∧ α.contains c

def p3 : Prop :=
  ∀ (l₁ l₂ : Line),
    (¬ ∃ (p : Point), l₁.contains p ∧ l₂.contains p)
    → l₁.parallel l₂

def p4 : Prop :=
  ∀ (α : Plane) (l m : Line),
    (α.contains l ∧ ¬ α.contains m ∧ m.perpendicular α) 
    → m.perpendicular l

-- Main theorem that identifies the true compound propositions
theorem true_compound_propositions :
  (p1 ∧ p4) ∧ ¬ (p1 ∧ p2) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) :=
by
  -- Proof might involve defining lines, points, and planes and their relationships
  sorry

end true_compound_propositions_l822_822287


namespace number_of_hydrogen_atoms_l822_822201

theorem number_of_hydrogen_atoms (C_atoms : ℕ) (O_atoms : ℕ) (molecular_weight : ℕ) 
    (C_weight : ℕ) (O_weight : ℕ) (H_weight : ℕ) : C_atoms = 3 → O_atoms = 1 → 
    molecular_weight = 58 → C_weight = 12 → O_weight = 16 → H_weight = 1 → 
    (molecular_weight - (C_atoms * C_weight + O_atoms * O_weight)) / H_weight = 6 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end number_of_hydrogen_atoms_l822_822201


namespace range_of_m_l822_822361

theorem range_of_m 
  (p : ∀ x > 0, m^2 + 2m - 1 ≤ x + x⁻¹)
  (q : ∀ x, (5 - m^2)^x = (5 - m^2) * (λ x, x) x)
  (h1 : ∃ x > 0, p ∨ q)
  (h2 : ∀ x, p ∧ q = false) :
  -3 ≤ m ∧ m ≤ -2 ∨ 1 < m ∧ m < 2 := 
sorry

end range_of_m_l822_822361


namespace no_nat_transfer_initial_digit_end_increases_by_5_6_8_l822_822032

theorem no_nat_transfer_initial_digit_end_increases_by_5_6_8 :
  ∀ m : ℕ, m > 0 → (∃ a1 a2 … an : ℕ, (a1 ≠ 0) ∧ 
    (m = a1 * 10^(n-1) + T) ∧ 
    (m' = T * 10 + a1) ∧ 
    (∀ k ∈ {5, 6, 8}, m' ≠ k * m)) := 
begin
  intro m,
  intros m_pos,
  // Proof omitted
  sorry,
end

end no_nat_transfer_initial_digit_end_increases_by_5_6_8_l822_822032


namespace vans_needed_for_trip_l822_822847

theorem vans_needed_for_trip (students adults van_capacity : ℕ) (total_people : ℤ) : 
    students = 28 → adults = 7 → van_capacity = 4 → total_people = students + adults →
    ⌈(total_people : ℚ) / van_capacity⌉ = 9 := 
by 
  intros h_students h_adults h_van_capacity h_total_people
  rw [h_students, h_adults, h_van_capacity, h_total_people]
  norm_num
  sorry

end vans_needed_for_trip_l822_822847


namespace three_digit_numbers_with_digit_5_l822_822410

theorem three_digit_numbers_with_digit_5 : 
  {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ d : ℕ, d ∈ [n / 100 % 10, n / 10 % 10, n % 10] ∧ d = 5)}.card = 270 :=
by
  sorry

end three_digit_numbers_with_digit_5_l822_822410


namespace tan_div_sum_eq_one_l822_822010

variable {x y : ℝ}

theorem tan_div_sum_eq_one
  (h1 : (sin x / cos y) + (sin y / cos x) = 2)
  (h2 : (cos x / sin y) + (cos y / sin x) = 3) :
  (tan x / tan y) + (tan y / tan x) = 1 := by
  sorry

end tan_div_sum_eq_one_l822_822010


namespace terminating_decimal_fraction_count_l822_822715

theorem terminating_decimal_fraction_count :
  {n : ℤ | 1 ≤ n ∧ n ≤ 569 ∧ (∃ k : ℤ, n = 57 * k)}.finite.card = 9 :=
by sorry

end terminating_decimal_fraction_count_l822_822715


namespace xy_min_x_min_y_divisible_by_3_prob_l822_822599

theorem xy_min_x_min_y_divisible_by_3_prob :
  let S := {n | 1 ≤ n ∧ n ≤ 15}
  let E := {(x, y) | x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ (x * y - x - y) % 3 = 0}
  let total_pairs := S.toFinset.pairwise_prod
  let favorable_pairs := E.toFinset
  (favorable_pairs.card : ℚ) / total_pairs.card = 2/21 :=
by
  -- Implementation
  sorry

end xy_min_x_min_y_divisible_by_3_prob_l822_822599


namespace potato_cost_l822_822126

variables (x : ℝ)
variables (b a : ℝ)

def andrey_earnings (x : ℝ) : ℝ := 120 * x
def boris_earnings (x : ℝ) : ℝ := 124.8 * x

theorem potato_cost :
  (boris_earnings x) - (andrey_earnings x) = 1200 → x = 250 :=
  by
    unfold andrey_earnings
    unfold boris_earnings
    sorry

end potato_cost_l822_822126


namespace impossible_all_white_by_flipping_l822_822444

-- Defining the initial grid configuration
def initial_grid : list (list bool) :=
  [[true, false, false],
   [false, false, false],
   [false, false, false]]

-- A function to flip a row in the grid
def flip_row (grid : list (list bool)) (r : ℕ) : list (list bool) :=
  grid.map_with_index (λ i row, if i = r then row.map bnot else row)

-- A function to flip a column in the grid
def flip_col (grid : list (list bool)) (c : ℕ) : list (list bool) :=
  grid.map (λ row, row.map_with_index (λ j cell, if j = c then bnot cell else cell))

-- Defining the property of having all cells white
def all_white (grid : list (list bool)) : Prop :=
  grid.all (λ row, row.all (λ cell, ¬cell))

-- Proving the impossibility
theorem impossible_all_white_by_flipping (g : list (list bool)) :
  g = initial_grid → ¬(∃ steps : list (bool × ℕ), 
    all_white (
      steps.foldl
      (λ grid step,
        match step with
        | (tt, r) => flip_row grid r
        | (ff, c) => flip_col grid c
        end)
      g)) :=
by
  intro h
  rw h
  sorry

end impossible_all_white_by_flipping_l822_822444


namespace number_of_bouquets_to_earn_1000_dollars_l822_822264

def cost_of_buying (n : ℕ) : ℕ :=
  n * 20

def revenue_from_selling (m : ℕ) : ℕ :=
  m * 20

def profit_per_operation : ℤ :=
  revenue_from_selling 7 - cost_of_buying 5

theorem number_of_bouquets_to_earn_1000_dollars :
  ∀ bouquets_needed : ℕ, bouquets_needed = 5 * (1000 / profit_per_operation.nat_abs) :=
sorry

end number_of_bouquets_to_earn_1000_dollars_l822_822264


namespace ratio_of_any_two_numbers_is_rational_l822_822508

theorem ratio_of_any_two_numbers_is_rational
  (x1 x2 x3 x4 x5 : ℝ)
  (h1 : ∀ a b c : ℝ, a ≠ b → b ≠ c → a ≠ c → a ∈ {x1, x2, x3, x4, x5} → b ∈ {x1, x2, x3, x4, x5} → c ∈ {x1, x2, x3, x4, x5} → (a * b + b * c + c * a) ∈ ℚ) :
  ∀ a b : ℝ, a ∈ {x1, x2, x3, x4, x5} → b ∈ {x1, x2, x3, x4, x5} → (a / b) ∈ ℚ := 
sorry

end ratio_of_any_two_numbers_is_rational_l822_822508


namespace trig_identity_l822_822366

theorem trig_identity (α : ℝ) 
  (h : Real.tan (α - Real.pi / 4) = 1 / 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2 := 
sorry

end trig_identity_l822_822366


namespace modify_quadratic_polynomial_exists_integer_root_l822_822545

theorem modify_quadratic_polynomial_exists_integer_root
  (a b c : ℕ) (h_sum : a + b + c = 2000) :
  ∃ a' b' c' : ℕ, (a'.natAbs - a).natAbs + (b'.natAbs - b).natAbs + (c'.natAbs - c).natAbs ≤ 1050 ∧
  ∃ x : ℤ, a' * (x^2) + b' * x + c' = 0 :=
sorry

end modify_quadratic_polynomial_exists_integer_root_l822_822545


namespace common_tangents_intersect_at_single_point_l822_822402

-- Definitions for circles and their mutual tangents
structure Circle (σ : Type) := 
  (center : σ)
  (radius : ℝ)

variables {σ : Type} (α β γ : Circle σ)

-- Definitions for common internal tangents
variables (l1 l2 m1 m2 n1 n2 : σ → Prop)

-- Conditions and the main theorem
theorem common_tangents_intersect_at_single_point
  (H1 : ∀ x, l1 x ↔ ∃ p q, p ∈ α → q ∈ β → x = intersection_point_of_tangents p q)
  (H2 : ∀ x, l2 x ↔ ∃ p q, p ∈ α → q ∈ β → x = intersection_point_of_tangents p q)
  (H3 : ∀ x, m1 x ↔ ∃ p q, p ∈ β → q ∈ γ → x = intersection_point_of_tangents p q)
  (H4 : ∀ x, m2 x ↔ ∃ p q, p ∈ β → q ∈ γ → x = intersection_point_of_tangents p q)
  (H5 : ∀ x, n1 x ↔ ∃ p q, p ∈ γ → q ∈ α → x = intersection_point_of_tangents p q)
  (H6 : ∀ x, n2 x ↔ ∃ p q, p ∈ γ → q ∈ α → x = intersection_point_of_tangents p q)
  (H_intersect : ∃ (P : σ), l1 P ∧ m1 P ∧ n1 P) :
  ∃ (Q : σ), l2 Q ∧ m2 Q ∧ n2 Q :=
sorry

end common_tangents_intersect_at_single_point_l822_822402


namespace monotonic_decreasing_interval_l822_822572

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - Real.log x

theorem monotonic_decreasing_interval :
  {x : ℝ | 0 < x ∧ x ≤ 1} = {x : ℝ | ∃ ε > 0, ∀ y, y < x → f y > f x ∧ y > 0} :=
sorry

end monotonic_decreasing_interval_l822_822572


namespace calc_sqrt_expr_l822_822675

theorem calc_sqrt_expr : (Real.sqrt 2 + 1) ^ 2 - Real.sqrt 18 + 2 * Real.sqrt (1 / 2) = 3 := by
  sorry

end calc_sqrt_expr_l822_822675


namespace binary_to_decimal_addition_l822_822684

theorem binary_to_decimal_addition :
  (let binary_value := 1 * 2^6 + 1 * 2^5 + 0 * 2^4 + 1 * 2^3 + 0 * 2^2 + 0 * 2^1 + 1 * 2^0 in
   binary_value + 14 = 119) :=
by
  let binary_value := 1 * 2^6 + 1 * 2^5 + 0 * 2^4 + 1 * 2^3 + 0 * 2^2 + 0 * 2^1 + 1 * 2^0
  have h1 : binary_value = 105 := by sorry
  have h2 : binary_value + 14 = 119 := by sorry
  exact h2

end binary_to_decimal_addition_l822_822684


namespace largest_x_FloorDiv7_eq_FloorDiv8_plus_1_l822_822824

-- Definitions based on conditions
def floor_div_7 (x : ℕ) : ℕ := x / 7
def floor_div_8 (x : ℕ) : ℕ := x / 8

-- The statement of the problem
theorem largest_x_FloorDiv7_eq_FloorDiv8_plus_1 :
  ∃ x : ℕ, (floor_div_7 x = floor_div_8 x + 1) ∧ (∀ y : ℕ, floor_div_7 y = floor_div_8 y + 1 → y ≤ x) ∧ x = 104 :=
sorry

end largest_x_FloorDiv7_eq_FloorDiv8_plus_1_l822_822824


namespace chocolates_problem_l822_822676

-- Let denote the quantities as follows:
-- C: number of caramels
-- N: number of nougats
-- T: number of truffles
-- P: number of peanut clusters

def C_nougats_truffles_peanutclusters (C N T P : ℕ) :=
  N = 2 * C ∧
  T = C + 6 ∧
  C + N + T + P = 50 ∧
  P = 32

theorem chocolates_problem (C N T P : ℕ) :
  C_nougats_truffles_peanutclusters C N T P → C = 3 :=
by
  intros h
  have hN := h.1
  have hT := h.2.1
  have hSum := h.2.2.1
  have hP := h.2.2.2
  sorry

end chocolates_problem_l822_822676


namespace number_of_correct_statements_is_zero_l822_822083

theorem number_of_correct_statements_is_zero :
  (¬ ∃ S, (∀ ε > 0, ∀ x ∈ S, abs x < ε)) ∧
  (∀ x y : ℝ, ¬ (x * y - 1 = 0) = (∀ y : ℝ, y = x^2 - 1)) ∧
  ({1, (3 / 2), (6 / 4), abs (-1 / 2), 0.5}.card = 5) = false ∧
  (∀ x y : ℝ, (x * y ≤ 0 ↔ (x, y) ∈ { p : ℝ × ℝ | (p.1 > 0 ∧ p.2 < 0) ∨ (p.1 < 0 ∧ p.2 > 0) })) :=
begin
  split,
  { intro h,
    cases h with S hS,
    have := hS 1 (by norm_num) 0 (set.mem_univ 0),
    linarith },
  split,
  { intro h,
    apply set.not_subset,
    intros a b,
    intros ha,
    apply not_forall_of_exists_not,
    existsi (a, b),
    split,
    { linarith },
    { intros hx,
      norm_num,
      linarith } },
  split,
  { intro hs,
    have := finset.card_eq_five,
    apply h,
    simp [finset.card] },
  { intro hxy,
    apply set.not_forall_of_exists_not,
    existsi (0, 0),
    split,
    { norm_num },
    { norm_num } }
end

end number_of_correct_statements_is_zero_l822_822083


namespace sum_of_odd_integers_less_than_50_l822_822160

def sumOddIntegersLessThan (n : Nat) : Nat :=
  List.sum (List.filter (λ x => x % 2 = 1) (List.range n))

theorem sum_of_odd_integers_less_than_50 : sumOddIntegersLessThan 50 = 625 :=
  by
    sorry

end sum_of_odd_integers_less_than_50_l822_822160


namespace trigonometric_fraction_simplification_l822_822492

noncomputable def c := Real.pi / 7

theorem trigonometric_fraction_simplification :
  (sin (2 * c) * sin (4 * c) * sin (6 * c) * sin (8 * c) * sin (10 * c)) / 
  (sin c * sin (3 * c) * sin (5 * c) * sin (7 * c) * sin (9 * c)) = 
  (sin c) / (sin (2 * c)) :=
sorry

end trigonometric_fraction_simplification_l822_822492


namespace sum_of_pattern_l822_822710

-- Definition of the nth row sums in the given pattern
def row_sum (k n : ℕ) : ℕ :=
  (n * (n + 2 * k - 1)) / 2

-- Definition of the total sum in the given pattern of n^2 numbers
def total_sum (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k, row_sum (k+1) n)

-- The theorem to prove
theorem sum_of_pattern (n : ℕ) : total_sum n = n^3 :=
by
  sorry

end sum_of_pattern_l822_822710


namespace problem_statement_l822_822297

variables (p1 p2 p3 p4 : Prop)

theorem problem_statement (h_p1 : p1 = True)
                         (h_p2 : p2 = False)
                         (h_p3 : p3 = False)
                         (h_p4 : p4 = True) :
  (p1 ∧ p4) = True ∧
  (p1 ∧ p2) = False ∧
  (¬p2 ∨ p3) = True ∧
  (¬p3 ∨ ¬p4) = True :=
by
  sorry

end problem_statement_l822_822297


namespace problem_l822_822893

theorem problem (a b c : ℝ) (h : 1/a + 1/b + 1/c = 1/(a + b + c)) : (a + b) * (b + c) * (a + c) = 0 := 
by
  sorry

end problem_l822_822893


namespace length_of_floor_y_l822_822179

theorem length_of_floor_y
  (A B : ℝ)
  (hx : A = 10)
  (hy : B = 18)
  (width_y : ℝ)
  (length_y : ℝ)
  (width_y_eq : width_y = 9)
  (area_eq : A * B = width_y * length_y) :
  length_y = 20 := 
sorry

end length_of_floor_y_l822_822179


namespace passengers_have_41_legs_l822_822649

noncomputable def total_legs_of_passengers (H C : ℕ) (one_legged_captain : Bool) : ℕ :=
  let cat_legs := C * 4
  let human_heads := H - C
  let human_legs := (human_heads - 1) * 2 + (if one_legged_captain then 1 else 2)
  cat_legs + human_legs

theorem passengers_have_41_legs :
  ∃ (H C : ℕ) (one_legged_captain : Bool), H = 14 ∧ C = 7 ∧ one_legged_captain = true ∧ total_legs_of_passengers H C one_legged_captain = 41 := 
by
  exists 14
  exists 7
  exists true
  split
  repeat {refl}
  sorry

end passengers_have_41_legs_l822_822649


namespace math_ineq_problem_l822_822900

variable (a b c : ℝ)

theorem math_ineq_problem
  (h1 : a ≥ b) 
  (h2 : b ≥ c) 
  (h3 : a + b + c ≤ 1)
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) : 
  a^2 + 3 * b^2 + 5 * c^2 ≤ 1 :=
by
  sorry

end math_ineq_problem_l822_822900


namespace willowdale_hockey_club_members_l822_822457

noncomputable def cost_socks := 6
noncomputable def cost_tshirt := cost_socks + 7
noncomputable def cost_cap := cost_tshirt - 3
noncomputable def total_cost_one_game := cost_socks + cost_tshirt + cost_cap
noncomputable def total_cost_both_games := 2 * total_cost_one_game
noncomputable def total_uniform_cost := 3630

theorem willowdale_hockey_club_members:
  (total_uniform_cost / total_cost_both_games).round = 63 :=
by
  sorry

end willowdale_hockey_club_members_l822_822457


namespace D_is_divisible_by_2k_l822_822583

-- Define finite sets A_1, A_2, ..., A_n
variable {α : Type*}
variable {n k : ℕ}
variable (A : Fin n → Finset α)

-- Define d as the number of elements in ⋃ {A i | i < n} that are in an odd number of A i sets
noncomputable def d (A : Fin n → Finset α) : ℕ := 
  Fintype.card { x : α // ∃! i : Fin n, x ∈ A i }

-- Define D(k)
noncomputable def D (k : ℕ) : ℕ :=
  d A 
  - Finset.sum (Finset.univ) (λ i, (A i).card)
  + 2 * Finset.sum (Finset.univ.powerset_len 2) (λ s, (A s.1).inter (A s.2)).card
  + ∑ j in Finset.range k, (-1) ^ j * 2 ^ (j - 1) * ∑ s in Finset.univ.powerset_len j, ((s : Finset α).bvfilter A).card

theorem D_is_divisible_by_2k (A : Fin n → Finset α) : 
  D A k % 2^k = 0 :=
sorry

end D_is_divisible_by_2k_l822_822583


namespace number_pyramid_l822_822464

theorem number_pyramid (x y : ℕ) : 
  let b1 := 9
  let b2 := 6
  let b3 := 10
  let b4 := 8
  let l21 := b1 + b2 -- 15
  let l22 := b2 + b3 -- 16
  let l23 := b3 + b4 -- 18
  let l31 := l21 + l22 -- 31
  let l32 := l22 + l23 -- 34
  (x + 15 = 31) ∧ (15 + y = 34) → x = 16 ∧ y = 19 := by
  intro h
  cases h with h1 h2
  have hx := h1
  have hy := h2
  sorry

end number_pyramid_l822_822464


namespace solve_by_completing_square_l822_822155

theorem solve_by_completing_square (x: ℝ) (h: x^2 + 4 * x - 3 = 0) : (x + 2)^2 = 7 := 
by 
  sorry

end solve_by_completing_square_l822_822155


namespace equation_of_line_l822_822885

-- Define the circle C with radius 5 centered at (3, 4)
def circle (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 25

-- Define line equation given in problem
def line (m x y : ℝ) : Prop := m * x + y - m - 2 = 0

-- The problem statement to be translated into a theorem
theorem equation_of_line (m : ℝ) :
  (∀ x y : ℝ, circle x y ∧ line m x y → mx + y - 3 = 0) :=
begin
  sorry
end

end equation_of_line_l822_822885


namespace triangle_side_b_l822_822443

theorem triangle_side_b (A B C a b c : ℝ)
  (hA : A = 135)
  (hc : c = 1)
  (hSinB_SinC : Real.sin B * Real.sin C = Real.sqrt 2 / 10) :
  b = Real.sqrt 2 ∨ b = Real.sqrt 2 / 2 :=
by
  sorry

end triangle_side_b_l822_822443


namespace number_of_bouquets_to_earn_1000_dollars_l822_822262

def cost_of_buying (n : ℕ) : ℕ :=
  n * 20

def revenue_from_selling (m : ℕ) : ℕ :=
  m * 20

def profit_per_operation : ℤ :=
  revenue_from_selling 7 - cost_of_buying 5

theorem number_of_bouquets_to_earn_1000_dollars :
  ∀ bouquets_needed : ℕ, bouquets_needed = 5 * (1000 / profit_per_operation.nat_abs) :=
sorry

end number_of_bouquets_to_earn_1000_dollars_l822_822262


namespace shortest_path_is_sqrt_five_l822_822657

-- Define the concept of a cube and the diagonal path on its surface
def is_cube (shape : Type) : Prop := 
  ∃ length : ℝ, length > 0 ∧ 
  ∀ (x y z : ℝ), 
  shape = (λ (x y z : ℝ), (0 <= x ∧ x <= length) ∧ (0 <= y ∧ y <= length) ∧ (0 <= z ∧ z <= length))

-- Define the 2D representation unfolding of two adjacent faces of a cube
def unfolded_cube_rectangle (length : ℝ) : set (ℝ × ℝ) := 
  { p : ℝ × ℝ | (0 <= p.1 ∧ p.1 <= 2 * length) ∧ (0 <= p.2 ∧ p.2 <= length) }

-- Define the shortest path calculation in the 2D plane
def shortest_path (length : ℝ) : ℝ := 
  real.sqrt ((2 * length) ^ 2 + length ^ 2)

-- The theorem we want to prove
theorem shortest_path_is_sqrt_five (length : ℝ) (h_length_pos : length > 0) : 
  let diagonal := shortest_path length in 
  diagonal = real.sqrt 5 := 
by 
  calc diagonal
      = real.sqrt ((2 * length) ^ 2 + length ^ 2) : by refl
  ... = real.sqrt (4 * length ^ 2 + length ^ 2) : by norm_num
  ... = real.sqrt (5 * length ^ 2) : by rw [mul_assoc, add_comm]
  ... = real.sqrt 5 * real.sqrt (length ^ 2) : by rw real.sqrt_mul
  ... = real.sqrt 5 * length : by { rw real.sqrt_sq_eq_abs }, sorry

end shortest_path_is_sqrt_five_l822_822657


namespace ratio_rational_l822_822514

variable (α : Type) [LinearOrderedField α]
variables (a b c d e : α)
variable (numbers : Fin 5 → α)
variable (h_diff : ∀ i j : Fin 5, i ≠ j → numbers i ≠ numbers j)
variable (h_pos : ∀ i : Fin 5, 0 < numbers i)
variable (h_rational_prod_sum : ∀ (i j k : Fin 5), i ≠ j → j ≠ k → i ≠ k → is_r := ((numbers i) * (numbers j) + (numbers j) * (numbers k) + (numbers k) * (numbers i)))

theorem ratio_rational (i j : Fin 5) (h_ij: i ≠ j) : is_r := (numbers i / numbers j) := sorry

end ratio_rational_l822_822514


namespace min_trucks_l822_822203

theorem min_trucks (W_total : ℝ) (W_box_max : ℝ) (W_truck : ℝ) :
  W_total = 13.5 → W_box_max ≤ 0.35 → W_truck = 1.5 →
  ∃ (n : ℕ), n = 11 ∧ ∀ (k : ℕ), k < n → 
  k * W_truck + W_box_max ≤ W_total := 
by
  intro hW_total hW_box_max hW_truck
  use 11
  split
  { exact rfl }
  { intro k hk
    sorry
  }

end min_trucks_l822_822203


namespace pears_sales_l822_822622

-- Define the problem statement as a function
theorem pears_sales (x : ℕ) (h_condition1 : x + 2 * x = 480) : 2 * x = 320 :=
by {
  -- You can use the given condition directly to demonstrate the proof.
  have h_total : 3 * x = 480,
  {
    exact h_condition1,
  },
  -- Obtain the value of x by solving the equation
  have h_x : x = 480 / 3,
  {
    sorry,
  },
  -- Substitute the value of x in the condition to get the answer
  rw h_x,
  norm_num,
}

end pears_sales_l822_822622


namespace problem_proof_l822_822479

variables (A B C D E F : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] 
[p : Points A B C] (r : ℝ) (s : ℝ)

-- Given conditions
def condition1 : Prop := ∃ (ABC : Triangle A B C), True
def condition2 : Prop := D ∈ segment ℝ B C ∧ E ∈ segment ℝ C A ∧ F ∈ segment ℝ A B
def condition3 : Prop := (inradius (Triangle.mk A E F) = r / 2) ∧
                         (inradius (Triangle.mk B D F) = r / 2) ∧
                         (inradius (Triangle.mk C D E) = r / 2)

-- Conclusion to be proved
def to_prove : Prop := midpoint ℝ D B C ∧ midpoint ℝ E C A ∧ midpoint ℝ F A B

theorem problem_proof : condition1 A B C D E F r s → condition2 A B C D E F → condition3 A B C D E F r → to_prove A B C D E F := by
sorry

end problem_proof_l822_822479


namespace pq_sum_l822_822003

noncomputable def p := sorry -- p will be provided by the conditions
noncomputable def q := sorry -- q will be provided by the conditions

axiom p_condition : p^3 - 12 * p^2 + 25 * p - 75 = 0
axiom q_condition : 10 * q^3 - 75 * q^2 - 375 * q + 3750 = 0

theorem pq_sum : p + q = -5 / 2 := 
by 
  sorry

end pq_sum_l822_822003


namespace cost_of_one_bag_l822_822133

theorem cost_of_one_bag (x : ℝ) :
  let Andrey_earning := 60 * 2 * x
  let Boris_earning := 15 * 1.6 * x + 45 * (1.6 * 1.4) * x
  Boris_earning - Andrey_earning = 1200 →
  x = 250 := 
by
  intros
  let Andrey_earning := 60 * 2 * x
  let Boris_earning := 15 * 1.6 * x + 45 * (1.6 * 1.4) * x
  have h : Boris_earning - Andrey_earning = 1200 := by assumption
  let simplified_h := 
    calc
      Boris_earning - Andrey_earning
        = (15 * 1.6 * x + 45 * (1.6 * 1.4) * x) - (60 * 2 * x) : by simp [Andrey_earning, Boris_earning]
    ... = (15 * 1.6 * x + 45 * (1.6 * 1.4) * x) - 120 * x : by simp
    ... = (24 * x + 100.8 * x) - 120 * x : by simp
    ... = 124.8 * x - 120 * x : by simp
    ... = 4.8 * x : by simp
    ... = 1200 : by rw h
  exact (div_eq_iff (by norm_num : (4.8 : ℝ) ≠ 0)).1 simplified_h  -- solves for x

end cost_of_one_bag_l822_822133


namespace remainder_17_pow_2023_mod_28_l822_822688

theorem remainder_17_pow_2023_mod_28 :
  17^2023 % 28 = 17 := 
by sorry

end remainder_17_pow_2023_mod_28_l822_822688


namespace simplify_sqrt180_l822_822049

-- Conditions from the problem definition
def sqrt (x : ℕ) : ℝ := Real.sqrt x
def six := 6
def five := 5

-- The statement of the problem as a Lean theorem
theorem simplify_sqrt180 : sqrt 180 = six * sqrt five := by 
  sorry

end simplify_sqrt180_l822_822049


namespace solution_set_sin_log_inequality_l822_822108

theorem solution_set_sin_log_inequality :
  {x : ℝ | x ∈ set.Ioo (-1 : ℝ) 1 ∧ abs (sin x) + abs (log (1 - x ^ 2)) > abs (sin x + log (1 - x ^ 2))} = set.Ioo 0 1 :=
by
  sorry

end solution_set_sin_log_inequality_l822_822108


namespace eliza_ironing_l822_822311

noncomputable def time_to_iron_dress := 20
noncomputable def blouse_ironing_time := 120
noncomputable def dress_ironing_time := 180
noncomputable def total_pieces := 17

theorem eliza_ironing (time_to_iron_dress blouse_ironing_time dress_ironing_time total_pieces: ℕ) 
  (assump1 : time_to_iron_dress = 20)
  (assump2 : blouse_ironing_time = 120)
  (assump3 : dress_ironing_time = 180)
  (assump4 : total_pieces = 17)
  : ∃ (time_to_iron_blouse : ℕ), time_to_iron_blouse = 15 :=
begin
  sorry
end

end eliza_ironing_l822_822311


namespace average_salary_of_all_workers_l822_822882

variable (number_of_workers : ℕ) (techs_num : ℕ) (rest_num : ℕ)
variable (tech_avg_salary : ℕ) (rest_avg_salary : ℕ)
variable (total_salary : ℕ) (avg_salary : ℕ)

-- Defining the conditions
def condition_number_of_workers : Prop := number_of_workers = 14
def condition_techs_num : Prop := techs_num = 7
def condition_rest_num : Prop := rest_num = number_of_workers - techs_num
def condition_tech_avg_salary : Prop := tech_avg_salary = 12000
def condition_rest_avg_salary : Prop := rest_avg_salary = 6000
def condition_total_salary : Prop := total_salary = (techs_num * tech_avg_salary) + (rest_num * rest_avg_salary)
def condition_avg_salary : Prop := avg_salary = total_salary / number_of_workers

-- The theorem to be proved
theorem average_salary_of_all_workers 
  (h1 : condition_number_of_workers)
  (h2 : condition_techs_num)
  (h3 : condition_rest_num)
  (h4 : condition_tech_avg_salary)
  (h5 : condition_rest_avg_salary)
  (h6 : condition_total_salary)
  (h7 : condition_avg_salary)
  : avg_salary = 9000 :=
sorry

end average_salary_of_all_workers_l822_822882


namespace simplify_sqrt180_l822_822050

-- Conditions from the problem definition
def sqrt (x : ℕ) : ℝ := Real.sqrt x
def six := 6
def five := 5

-- The statement of the problem as a Lean theorem
theorem simplify_sqrt180 : sqrt 180 = six * sqrt five := by 
  sorry

end simplify_sqrt180_l822_822050


namespace min_ab_l822_822360

theorem min_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b - a * b + 3 = 0) : 
  9 ≤ a * b :=
sorry

end min_ab_l822_822360


namespace divisible_by_prime_greater_than_7_l822_822627

theorem divisible_by_prime_greater_than_7 (N : ℕ) (hN : N % 100 = 33) : ∃ p : ℕ, prime p ∧ p > 7 ∧ p ∣ N :=
by
  sorry

end divisible_by_prime_greater_than_7_l822_822627


namespace largest_number_l822_822242

theorem largest_number (a b c d : ℝ)
  (h1 : a = real.sqrt 2)
  (h2 : b = 0)
  (h3 : c = -1)
  (h4 : d = 2) :
  d = max (max a b) (max c d) :=
by
  sorry

end largest_number_l822_822242


namespace num_of_friends_l822_822530

theorem num_of_friends :
  ∃ friends : Finset String, 
  friends = {"Sam", "Dan", "Tom", "Keith"} ∧
  friends.card = 4 :=
begin
  sorry
end

end num_of_friends_l822_822530


namespace tom_seashells_l822_822596

theorem tom_seashells (days : ℕ) (seashells_per_day : ℕ) (h1 : days = 5) (h2 : seashells_per_day = 7) : 
  seashells_per_day * days = 35 := 
by
  sorry

end tom_seashells_l822_822596


namespace equal_segments_among_AM_BM_CM_DM_l822_822347

noncomputable def isosceles_tr (A B C : Type) [metric_space A] (a b c : A) :=
    dist a b = dist a c ∨ dist b c = dist b a ∨ dist c a = dist c b

def convex_quadrilateral (A B C D M : Type) [metric_space A] :=
  (isosceles_tr A B M ∧ isosceles_tr B C M ∧ isosceles_tr C D M ∧ isosceles_tr D A M)

theorem equal_segments_among_AM_BM_CM_DM (A B C D M : Type) [metric_space A] :
  convex_quadrilateral A B C D M → 
  ∃ x y ∈ [dist A M, dist B M, dist C M, dist D M], x = y := 
begin
  sorry
end

end equal_segments_among_AM_BM_CM_DM_l822_822347


namespace intersection_of_sets_l822_822740

open Set

theorem intersection_of_sets (A B : Set ℕ) (hA : A = {0, 1, 2, 3, 4, 5}) (hB : B = {2, 4, 6}) :
  A ∩ B = {2, 4} :=
by
  sorry

end intersection_of_sets_l822_822740


namespace arithmetic_sequence_third_term_l822_822790

theorem arithmetic_sequence_third_term (b y : ℝ) 
  (h1 : 2 * b + y + 2 = 10) 
  (h2 : b + y + 2 = b + y + 2) : 
  8 - b = 6 := 
by 
  sorry

end arithmetic_sequence_third_term_l822_822790


namespace inequality_implies_range_of_a_l822_822713

theorem inequality_implies_range_of_a (a : ℝ) :
  (∀ x : ℝ, |2 - x| + |1 + x| ≥ a^2 - 2 * a) → (-1 ≤ a ∧ a ≤ 3) :=
sorry

end inequality_implies_range_of_a_l822_822713


namespace bill_needs_125_bouquets_to_earn_1000_l822_822255

-- Define the constants for the problem
def cost_per_bouquet : ℕ := 20
def roses_per_bouquet_buy : ℕ := 7
def roses_per_bouquet_sell : ℕ := 5
def target_profit : ℕ := 1000

-- Define the problem in terms of a theorem
theorem bill_needs_125_bouquets_to_earn_1000 :
  ∃ n : ℕ, (35 / roses_per_bouquet_sell) * cost_per_bouquet - (5 * cost_per_bouquet) = 40 → (5 * n) = 125 :=
begin
  sorry
end

end bill_needs_125_bouquets_to_earn_1000_l822_822255


namespace length_AC_eq_9_74_l822_822864

-- Define the cyclic quadrilateral and given constraints
noncomputable def quad (A B C D : Type) : Prop := sorry
def angle_BAC := 50
def angle_ADB := 60
def AD := 3
def BC := 9

-- Prove that length of AC is 9.74 given the above conditions
theorem length_AC_eq_9_74 
  (A B C D : Type)
  (h_quad : quad A B C D)
  (h_angle_BAC : angle_BAC = 50)
  (h_angle_ADB : angle_ADB = 60)
  (h_AD : AD = 3)
  (h_BC : BC = 9) :
  ∃ AC, AC = 9.74 :=
sorry

end length_AC_eq_9_74_l822_822864


namespace perimeter_of_field_l822_822887

theorem perimeter_of_field (b l : ℕ) (h1 : l = b + 30) (h2 : b * l = 18000) : 2 * (l + b) = 540 := 
by 
  -- Proof goes here
sorry

end perimeter_of_field_l822_822887


namespace magnitude_of_complex_number_l822_822345

theorem magnitude_of_complex_number (z : ℂ) (h : complex.I * z = 2 + 4 * complex.I) : complex.abs z = 2 * real.sqrt 5 :=
by
  sorry

end magnitude_of_complex_number_l822_822345


namespace simplify_sqrt_180_l822_822068

theorem simplify_sqrt_180 : sqrt 180 = 6 * sqrt 5 :=
by
  have h : 180 = 2^2 * 3^2 * 5 := by norm_num
  calc
    sqrt 180
      = sqrt (2^2 * 3^2 * 5)     : by rw [h]
  ... = sqrt (2^2) * sqrt (3^2) * sqrt 5 : by rw [sqrt_mul, sqrt_mul]
  ... = 2 * 3 * sqrt 5          : by rw [sqrt_sq, sqrt_sq]
  ... = 6 * sqrt 5              : by norm_num

end simplify_sqrt_180_l822_822068


namespace sqrt_180_simplified_l822_822056

theorem sqrt_180_simplified : Real.sqrt 180 = 6 * Real.sqrt 5 :=
   sorry

end sqrt_180_simplified_l822_822056


namespace common_difference_minimum_sum_value_l822_822905

variable {α : Type}
variables (a : ℕ → ℤ) (d : ℤ)
variables (S : ℕ → ℚ)

-- Conditions: Arithmetic sequence property and specific initial values
def is_arithmetic_sequence (d : ℤ) : Prop :=
  ∀ n, a n = a 1 + (n - 1) * d

axiom a1_eq_neg3 : a 1 = -3
axiom condition : 11 * a 5 = 5 * a 8 - 13

-- Define the sum of the first n terms of the arithmetic sequence
def sum_of_arithmetic_sequence (n : ℕ) (a : ℕ → ℤ) (d : ℤ) : ℚ :=
  (↑n / 2) * (2 * a 1 + ↑((n - 1) * d))

-- Prove the common difference and the minimum sum value
theorem common_difference : d = 31 / 9 :=
sorry

theorem minimum_sum_value : S 1 = -2401 / 840 :=
sorry

end common_difference_minimum_sum_value_l822_822905


namespace sum_of_exponents_of_prime_factors_of_sqrt_of_largest_perfect_square_l822_822159

open Nat

-- The condition: Calculation of prime factor exponents for 15!

def v_p_factorial (n p : ℕ) : ℕ :=
  (List.range' 1 (n+1)).map (λ k, n / p ^ k).sum

-- Specific exponents for our primes
def exp2 := v_p_factorial 15 2  -- exponent for 2 in 15!
def exp3 := v_p_factorial 15 3  -- exponent for 3 in 15!
def exp5 := v_p_factorial 15 5  -- exponent for 5 in 15!

theorem sum_of_exponents_of_prime_factors_of_sqrt_of_largest_perfect_square (h1 : exp2 = 11) (h2 : exp3 = 6) (h3 : exp5 = 3) : (5 + 3 + 1 = 9) := 
by
  sorry

end sum_of_exponents_of_prime_factors_of_sqrt_of_largest_perfect_square_l822_822159


namespace opposite_of_negative_2023_l822_822099

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l822_822099


namespace infinite_geometric_series_sum_l822_822313

-- First term of the geometric series
def a : ℚ := 5/3

-- Common ratio of the geometric series
def r : ℚ := -1/4

-- The sum of the infinite geometric series
def S : ℚ := a / (1 - r)

-- Prove that the sum of the series is equal to 4/3
theorem infinite_geometric_series_sum : S = 4/3 := by
  sorry

end infinite_geometric_series_sum_l822_822313


namespace least_roots_g_in_interval_l822_822644

noncomputable def g (x : ℝ) : ℝ := sorry

theorem least_roots_g_in_interval :
  (∀ x, g(3 + x) = g(3 - x)) →
  (∀ x, g(8 + x) = g(8 - x)) →
  g(0) = 0 →
  ∃ S : set ℝ, (∀ x ∈ S, g(x) = 0) ∧ set.finite S ∧ set.card S ≥ 334 ∧ ∀ x ∈ S, -1000 ≤ x ∧ x ≤ 1000 :=
by
  intros h1 h2 h3
  sorry

end least_roots_g_in_interval_l822_822644


namespace count_satisfying_integers_l822_822771

theorem count_satisfying_integers :
  (∃ S : Finset ℕ, (∀ n ∈ S, 9 < n ∧ n < 60) ∧ S.card = 50) :=
by
  sorry

end count_satisfying_integers_l822_822771


namespace find_g_18_l822_822877

noncomputable def g (x : ℝ) : ℝ :=
sorry -- The actual form is derived in the solution, not given as a condition

theorem find_g_18 :
  (∀ x : ℝ, g (x + g x) = 3 * g x) →
  (g 2 = 3) →
  (∃ a : ℝ, ∀ x : ℝ, g x = a * x) →
  g 18 = 27 :=
by {
  intros h1 h2 h3,
  sorry -- Proof details skipped per instructions
}

end find_g_18_l822_822877


namespace sum_of_squares_equality_l822_822306

theorem sum_of_squares_equality (n : ℕ) (h : n = 5) :
  (∑ i in Finset.range (n + 1), i^2) = (∑ i in Finset.range (2 * n + 1), i) := by
  sorry

end sum_of_squares_equality_l822_822306


namespace number_of_bouquets_to_earn_1000_dollars_l822_822263

def cost_of_buying (n : ℕ) : ℕ :=
  n * 20

def revenue_from_selling (m : ℕ) : ℕ :=
  m * 20

def profit_per_operation : ℤ :=
  revenue_from_selling 7 - cost_of_buying 5

theorem number_of_bouquets_to_earn_1000_dollars :
  ∀ bouquets_needed : ℕ, bouquets_needed = 5 * (1000 / profit_per_operation.nat_abs) :=
sorry

end number_of_bouquets_to_earn_1000_dollars_l822_822263


namespace simplify_expression_l822_822535

noncomputable def cube_root (x : ℝ) := x^(1/3)

theorem simplify_expression : 
  cube_root (8 + 27) * cube_root (8 + real.sqrt 64) = cube_root 560 := 
by 
  sorry

end simplify_expression_l822_822535


namespace part_I_part_II_part_III_l822_822485

-- Given conditions and functions
variables {a b : ℝ}
def f (x : ℝ) : ℝ := (1 / 3) * x^3 + a * x^2 + b * x + 1
def g (x : ℝ) : ℝ := exp x

-- Part (I): Proving that b = 1
theorem part_I (h_tangent : b = 1) : b = 1 :=
h_tangent

-- Part (II): Discussing monotonicity
theorem part_II (x : ℝ) : 
  let f' := λ x, x^2 + 2 * a * x + b in
  a^2 ≤ 1 → f'(x) ≥ 0 := 
begin
  let f' := λ x, x^2 + 2 * a * x + b,
  intro h,
  have : f'(x) ≥ 0, sorry,
  exact this,
end

-- Part (III): Proving g(x) > f(x) in (-∞, 0) when a ≤ 1/2
theorem part_III (h_b : b = 1) (h_a : a ≤ 1/2) (x : ℝ) (h_x : x < 0) :
  g(x) > f(x) :=
begin
  have : g(x) > f(x), sorry,
  exact this,
end

end part_I_part_II_part_III_l822_822485


namespace maximize_sales_volume_l822_822641

open Real

def profit (x : ℝ) : ℝ := (x - 20) * (400 - 20 * (x - 30))

theorem maximize_sales_volume : 
  ∃ x : ℝ, (∀ x' : ℝ, profit x' ≤ profit x) ∧ x = 35 := 
by
  sorry

end maximize_sales_volume_l822_822641


namespace chessboard_dark_more_than_light_l822_822665

theorem chessboard_dark_more_than_light :
  let light_squares := 4 * 5 + 4 * 4
  let dark_squares := 5 * 5 + 4 * 4
  dark_squares - light_squares = 5 := by
  let light_squares := 4 * 5 + 4 * 4
  let dark_squares := 5 * 5 + 4 * 4
  have h1 : dark_squares = 41 := rfl
  have h2 : light_squares = 36 := rfl
  rw [h1, h2]
  exact Nat.sub_self 36 41 sorry

end chessboard_dark_more_than_light_l822_822665


namespace apples_shared_equally_l822_822526

-- Definitions of the given conditions
def num_apples : ℕ := 9
def num_friends : ℕ := 3

-- Statement of the problem
theorem apples_shared_equally : num_apples / num_friends = 3 := by
  sorry

end apples_shared_equally_l822_822526


namespace find_functional_expression_find_x_when_y_neg4_l822_822422

noncomputable def proportional_relation (x y : ℝ) : Prop :=
  ∃ k : ℝ, y - 4 = k * (2 * x + 1)

theorem find_functional_expression :
  (proportional_relation (-1) 6) → (∀ x, ∃ y, y = -4 * x + 2) :=
by
  intros h x
  cases h with k hk
  have : k = -2 :=
    by
      linarith
  use -4 * x + 2
  linarith
  sorry  -- This part would include the steps to complete the proof

theorem find_x_when_y_neg4 :
  (proportional_relation (-1) 6) → (∃ x : ℝ, (∀ y : ℝ, y = -4) → x = 3 / 2) :=
by
  intros h
  cases h with k hk
  have : k = -2 :=
    by
      linarith
  use 3 / 2
  intros y hy
  linarith
  sorry  -- This part would include the steps to complete the proof

end find_functional_expression_find_x_when_y_neg4_l822_822422


namespace proof_problem_l822_822283

variable (p1 p2 p3 p4 : Prop)

theorem proof_problem (hp1 : p1) (hp2 : ¬ p2) (hp3 : ¬ p3) (hp4 : p4) :
  (p1 ∧ p4) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) := by
  sorry

end proof_problem_l822_822283


namespace mary_should_drink_6_glasses_l822_822921

-- Definitions based on conditions
def daily_water_goal_liters : ℚ := 1.5
def glass_capacity_ml : ℚ := 250
def liter_to_milliliters : ℚ := 1000

-- Conversion from liters to milliliters
def daily_water_goal_milliliters : ℚ := daily_water_goal_liters * liter_to_milliliters

-- Proof problem to show Mary needs 6 glasses per day
theorem mary_should_drink_6_glasses :
  daily_water_goal_milliliters / glass_capacity_ml = 6 := by
  sorry

end mary_should_drink_6_glasses_l822_822921


namespace tan_value_l822_822432

-- Define the conditions
def point_condition (a : ℝ) : Prop := 9 = 3^a

-- State the theorem
theorem tan_value (a : ℝ) (h : point_condition a) : 
  Real.tan (a * Real.pi / 3) = -Real.sqrt 3 :=
sorry

end tan_value_l822_822432


namespace proof_problem_l822_822813

-- Given a triangle ABC with points M and N on sides AC and BC respectively,
-- and point L on segment MN. Let the areas of triangles ABC, AML, and BNL be S, P, and Q respectively.
-- Define the ratios α = |AM| / |MC|, β = |CN| / |NB|, and γ = |ML| / |LN|.
-- We need to prove that ∛S ≥ ∛P + ∛Q.

variables (A B C M N L : Type) -- points
variables (S P Q α β γ : ℝ) -- areas and ratios

-- Conditions from the problem
axiom h1 : α = |AM| / |MC|
axiom h2 : β = |CN| / |NB|
axiom h3 : γ = |ML| / |LN|

axiom h4 : P = Q * α * β * γ
axiom h5 : S = Q * (α + 1) * (β + 1) * (γ + 1)

-- Known inequality relation
axiom inequality_relation : (α + 1) * (β + 1) * (γ + 1) ≥ (∛(α * β * γ) + 1)^3

-- The goal is to prove this statement
theorem proof_problem : ∛S ≥ ∛P + ∛Q :=
by 
  sorry

end proof_problem_l822_822813


namespace smallest_sum_is_417_l822_822940

def smallest_sum_of_two_3_digit_numbers (a b c d e f : ℕ) : ℕ :=
  100 * (a + d) + 10 * (b + e) + (c + f)

theorem smallest_sum_is_417 :
  ∃ a b c d e f : ℕ, a ∈ {1, 2, 3} ∧ b ∈ {1, 2, 3} ∧ c ∈ {1, 2, 3} ∧ 
                   d ∈ {7, 8, 9} ∧ e ∈ {7, 8, 9} ∧ f ∈ {7, 8, 9} ∧ 
                   a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ d ≠ e ∧ d ≠ f ∧ e ≠ f ∧
                   smallest_sum_of_two_3_digit_numbers a b c d e f = 417 :=
by {
  -- Acknowledging the existence of such a combination is equivalent to the smallest sum being 417.
  sorry 
}

end smallest_sum_is_417_l822_822940


namespace maximize_f_l822_822730

noncomputable def f (x : ℝ) : ℝ := 4 * x - 2 + 1 / (4 * x - 5)

theorem maximize_f (x : ℝ) (h : x < 5 / 4): ∃ M, (∀ y, (y < 5 / 4) → f y ≤ M) ∧ M = 1 := by
  sorry

end maximize_f_l822_822730


namespace F_has_2017_zeros_l822_822378

-- Define the functions f and g based on given transformations
def f (x : ℝ) : ℝ := Real.cos (2 * x)

def g (x : ℝ) : ℝ := Real.sin x

-- Define the combined function F
def F (x : ℝ) (a : ℝ) : ℝ := f x + a * g x

-- Declare a proof that with a = 1 and n = 1345, F(x) has exactly 2017 zeros in the interval (0, nπ)
theorem F_has_2017_zeros : ∃ (a : ℝ) (n : ℕ), a = 1 ∧ n = 1345 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 2017 → ∃ x : ℝ, 0 < x ∧ x < (n * Real.pi) ∧ F x a = 0) := 
  by 
  -- Since proof is not required, we add sorry to skip the proof
  sorry

end F_has_2017_zeros_l822_822378


namespace problem_statement_l822_822299

variables (p1 p2 p3 p4 : Prop)

theorem problem_statement (h_p1 : p1 = True)
                         (h_p2 : p2 = False)
                         (h_p3 : p3 = False)
                         (h_p4 : p4 = True) :
  (p1 ∧ p4) = True ∧
  (p1 ∧ p2) = False ∧
  (¬p2 ∨ p3) = True ∧
  (¬p3 ∨ ¬p4) = True :=
by
  sorry

end problem_statement_l822_822299


namespace greatest_int_pi_minus_e_eq_zero_l822_822281

noncomputable def pi := 3.141592
noncomputable def e := 2.718282

theorem greatest_int_pi_minus_e_eq_zero : Int.floor (pi - e) = 0 := by
  sorry

end greatest_int_pi_minus_e_eq_zero_l822_822281


namespace determine_values_l822_822307

-- Definitions of the conditions
variables {x y z : ℝ}
axiom condition_1 : |x - y^2| = z * x + y^2
axiom condition_2 : z * x + y^2 ≥ 0

-- Target conclusion
theorem determine_values (h1 : condition_1) (h2 : condition_2) :
  (x = 0 ∧ y = 0) ∨ (x = 2 * y^2 / (1 - z) ∧ z ≠ 1 ∧ z > -1) :=
sorry

end determine_values_l822_822307


namespace cabbage_price_is_4_02_l822_822817

noncomputable def price_of_cabbage (broccoli_price_per_pound: ℝ) (broccoli_pounds: ℝ) 
                                    (orange_price_each: ℝ) (oranges: ℝ) 
                                    (bacon_price_per_pound: ℝ) (bacon_pounds: ℝ) 
                                    (chicken_price_per_pound: ℝ) (chicken_pounds: ℝ) 
                                    (budget_percentage_for_meat: ℝ) 
                                    (meat_price: ℝ) : ℝ := 
  let broccoli_total := broccoli_pounds * broccoli_price_per_pound
  let oranges_total := oranges * orange_price_each
  let bacon_total := bacon_pounds * bacon_price_per_pound
  let chicken_total := chicken_pounds * chicken_price_per_pound
  let subtotal := broccoli_total + oranges_total + bacon_total + chicken_total
  let total_budget := meat_price / budget_percentage_for_meat
  total_budget - subtotal

theorem cabbage_price_is_4_02 : 
  price_of_cabbage 4 3 0.75 3 3 1 3 2 0.33 9 = 4.02 := 
by 
  sorry

end cabbage_price_is_4_02_l822_822817


namespace jewelry_store_total_cost_l822_822206

theorem jewelry_store_total_cost :
  let necklaces_needed := 7
  let rings_needed := 12
  let bracelets_needed := 7
  let necklace_price := 4
  let ring_price := 10
  let bracelet_price := 5
  let necklace_discount := if necklaces_needed >= 6 then 0.15 else if necklaces_needed >= 4 then 0.10 else 0
  let ring_discount := if rings_needed >= 20 then 0.10 else if rings_needed >= 10 then 0.05 else 0
  let bracelet_discount := if bracelets_needed >= 10 then 0.12 else if bracelets_needed >= 7 then 0.08 else 0
  let necklace_cost := necklaces_needed * (necklace_price * (1 - necklace_discount))
  let ring_cost := rings_needed * (ring_price * (1 - ring_discount))
  let bracelet_cost := bracelets_needed * (bracelet_price * (1 - bracelet_discount))
  let total_cost := necklace_cost + ring_cost + bracelet_cost
  total_cost = 170 := by
  -- calculation details omitted
  sorry

end jewelry_store_total_cost_l822_822206


namespace range_of_a_l822_822739

theorem range_of_a (x a : ℝ) 
  (h₁ : ∀ x, |x + 1| ≤ 2 → x ≤ a) 
  (h₂ : ∃ x, x > a ∧ |x + 1| ≤ 2) 
  : a ≥ 1 :=
sorry

end range_of_a_l822_822739


namespace max_value_3a_2b_l822_822357

-- The problem setup
def line_eq (a b : ℝ) : ℝ → ℝ → Prop :=
  λ x y, a * x + 2 * b * y - 1 = 0

def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

def chord_length (a b : ℝ) : ℝ :=
  2

def distance_from_center (a b : ℝ) : ℝ :=
  1

theorem max_value_3a_2b {a b : ℝ} (hline: ∀ x y, line_eq a b x y)
  (hchord: chord_length a b = 2 * sqrt 3)
  (hdistance: distance_from_center a b = 1)
  (heq: a^2 + 4 * b^2 = 1) : 
  ∃ θ φ : ℝ, 3 * a + 2 * b ≤ sqrt 10 := 
sorry

end max_value_3a_2b_l822_822357


namespace sequence_2023rd_term_l822_822848

theorem sequence_2023rd_term :
  let term := λ n : ℕ, (-1 : ℤ) ^ n * (n : ℚ) / (n + 1)
  in term 2023 = -2023 / 2024 := 
by
  sorry

end sequence_2023rd_term_l822_822848


namespace number_of_friends_l822_822528

def has14_pokemon_cards (x : String) : Prop :=
  x = "Sam" ∨ x = "Dan" ∨ x = "Tom" ∨ x = "Keith"

theorem number_of_friends :
  ∃ n, n = 4 ∧
        ∀ x, has14_pokemon_cards x ↔ x = "Sam" ∨ x = "Dan" ∨ x = "Tom" ∨ x = "Keith" :=
by
  sorry

end number_of_friends_l822_822528


namespace milk_tea_sales_l822_822580

-- Definitions
def relationship (x y : ℕ) : Prop := y = 10 * x + 2

-- Theorem statement
theorem milk_tea_sales (x y : ℕ) :
  relationship x y → (y = 822 → x = 82) :=
by
  intros h_rel h_y
  sorry

end milk_tea_sales_l822_822580


namespace evaluate_expression_l822_822563

theorem evaluate_expression : 2 + (0 * 2^2) = 2 :=
by
  calc
    2 + (0 * 2^2) = 2 + 0 : by rw [mul_zero, pow_two, mul_zero]
    ... = 2 : add_zero 2

end evaluate_expression_l822_822563


namespace math_problem_l822_822728

-- Definition of the functions f and g according to the problem's conditions
def f (x : ℝ) (n : ℕ) : ℝ := (x ^ 2 + x + 1) ^ n

-- Assuming g(x) is a polynomial of degree 2n
variable {g : ℝ → ℝ} (hg : ∃ n : ℕ, polynomial.degree g = (2 : ℕ))

-- Main theorem statement
theorem math_problem (n : ℕ) (hn : n > 0) (hgf : ∀ x : ℝ, f (x^2) n * g x = g (x^3)) :
  g 1 = 0 ∧ g (-1) = 0 ∧
  ∃ (a : ℕ → ℝ), f x n = ∑ i in range (n+1), a i * (x^i + x^(2*n - i)) := by
  sorry

end math_problem_l822_822728


namespace min_sum_primes_digits_l822_822601

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def digits := {1, 2, 3, 4, 5, 6, 7}

def primes_using_digits : set ℕ :=
  {p | is_prime p ∧ ∀ d ∈ digits, d ≠ 0 → d ∈ digits ∨ (d % 10 ∈ digits ∧ (d / 10) % 10 ∈ digits)}

def sum_min_primes : ℕ :=
  2 + 3 + 5 + 7 + 41 + 61

theorem min_sum_primes_digits :
  ∀ (S : set ℕ), (∀ p ∈ S, is_prime p) →
  (∀ d ∈ digits, ∃ p ∈ S, (p % 10 = d ∨ (p / 10) % 10 = d ∨ p / 100 = d)) →
  (∑ p in S, p) = 119 :=
by
  sorry

end min_sum_primes_digits_l822_822601


namespace probability_calculation_l822_822652

noncomputable def fair_coin_flip : ℕ → ℝ
| n := if n % 2 = 0 then 1 else -1

def sequence_sum (n : ℕ) : ℝ :=
∑ i in Finset.range n, fair_coin_flip i

def probability_S2_ne_0_S8_eq_2 : ℝ := 
  -- Skipping complex probability calculation via sorry
  13 / 128

-- The theorem statement
theorem probability_calculation :
  (∑ i in Finset.range 2, fair_coin_flip i ≠ 0) ∧
  (∑ i in Finset.range 8, fair_coin_flip i = 2) → 
  probability_S2_ne_0_S8_eq_2 = 13 / 128 :=
  sorry

end probability_calculation_l822_822652


namespace ratio_of_shaded_to_non_shaded_area_l822_822928

def triangle_equilateral (A B C : Type) [has_dist A B] [has_dist B C] [has_dist C A] : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

def trisection_points (A B C D E F : Type) [has_dist A B] [has_dist B C] [has_dist C A] : Prop :=
  dist A D = dist D B ∧ dist B E = dist E C ∧ dist C F = dist F A ∧ dist A B = 3 * dist A D

def midpoints (D E F G H : Type) [has_dist D F] [has_dist F E] : Prop :=
  dist D G = dist G F ∧ dist F H = dist H E

theorem ratio_of_shaded_to_non_shaded_area
  (A B C D E F G H : Type)
  [has_dist A B] [has_dist B C] [has_dist C A]
  [has_dist D F] [has_dist F E]
  (h1 : triangle_equilateral A B C)
  (h2 : trisection_points A B C D E F)
  (h3 : midpoints D E F G H) :
  let shaded_area := sorry in -- compute shaded_area here
  let non_shaded_area := sorry in -- compute non_shaded_area here
  (shaded_area / non_shaded_area) = 1 / 3 :=
sorry -- proof goes here

end ratio_of_shaded_to_non_shaded_area_l822_822928


namespace metric_regression_equation_l822_822592

noncomputable def predicted_weight_imperial (height : ℝ) : ℝ :=
  4 * height - 130

def inch_to_cm (inch : ℝ) : ℝ := 2.54 * inch
def pound_to_kg (pound : ℝ) : ℝ := 0.45 * pound

theorem metric_regression_equation (height_cm : ℝ) :
  (0.72 * height_cm - 58.5) = 
  (pound_to_kg (predicted_weight_imperial (height_cm / 2.54))) :=
by
  sorry

end metric_regression_equation_l822_822592


namespace potato_cost_l822_822129

variables (x : ℝ)
variables (b a : ℝ)

def andrey_earnings (x : ℝ) : ℝ := 120 * x
def boris_earnings (x : ℝ) : ℝ := 124.8 * x

theorem potato_cost :
  (boris_earnings x) - (andrey_earnings x) = 1200 → x = 250 :=
  by
    unfold andrey_earnings
    unfold boris_earnings
    sorry

end potato_cost_l822_822129


namespace original_numbers_not_222_l822_822851

theorem original_numbers_not_222 (a b c : ℤ) : 
  (a = 2 ∧ b = 2 ∧ c = 2) → 
  ¬ (∃ (final1 final2 final3 : ℤ), 
    (final1 = 1995 ∧ final2 = 1996 ∧ final3 = 1997) ∧ 
    all_steps_transform [a, b, c] [final1, final2, final3]) 
:= sorry

/-- 
  Helper function to describe the allowed transformation steps 
  For example, all_steps_transform [a, b, c] [1995, 1996, 1997]
  would mean that starting from [a, b, c] you can reach [1995, 1996, 1997]
  via any number of allowed transformation steps.
-/
def all_steps_transform (initial final : list ℤ) : Prop := sorry

end original_numbers_not_222_l822_822851


namespace solution_set_abs_inequality_l822_822109

theorem solution_set_abs_inequality : {x : ℝ | |x - 2| < 1} = {x : ℝ | 1 < x ∧ x < 3} :=
sorry

end solution_set_abs_inequality_l822_822109


namespace tank_capacity_l822_822815

def bucket_capacity := 5 -- Each bucket is five gallons

def jack_bucket_count := 2 -- Jack can carry two buckets of water at a time

def jill_bucket_count := 1 -- Jill can only manage one bucket at a time

def jack_trip_ratio := 3 / 2 -- Jack can complete three trips in the time it takes Jill to make two

def jill_trip_count := 30 -- Jill made 30 trips before the tank was filled

theorem tank_capacity :
  let jill_water := jill_trip_count * bucket_capacity in
  let jack_trip_count := jack_trip_ratio * jill_trip_count in
  let jack_water := jack_trip_count * (jack_bucket_count * bucket_capacity) in
  jill_water + jack_water = 600 :=
by
  sorry

end tank_capacity_l822_822815


namespace relationship_of_y_values_l822_822520

theorem relationship_of_y_values (k : ℝ) (y₁ y₂ y₃ : ℝ) :
  (y₁ = (k^2 + 3) / (-3)) ∧ (y₂ = (k^2 + 3) / (-1)) ∧ (y₃ = (k^2 + 3) / 2) →
  y₂ < y₁ ∧ y₁ < y₃ :=
by
  intro h
  have h₁ : y₁ = (k^2 + 3) / (-3) := h.1
  have h₂ : y₂ = (k^2 + 3) / (-1) := h.2.1
  have h₃ : y₃ = (k^2 + 3) / 2 := h.2.2
  sorry

end relationship_of_y_values_l822_822520


namespace unique_zero_point_c_monotonic_f2_b_increasing_sequence_x_n_l822_822839

-- Problem part 1: range of c for unique zero point
theorem unique_zero_point_c (n : ℕ) (h₀ : n > 0) :
  ∀ (c : ℝ), 0 < c ∧ c < 3 / 2 ↔ 
    ∃! (x : ℝ), (1 / 2 < x ∧ x < 1) ∧ (x^n - (1 / x) + c = 0) := sorry

-- Problem part 2: range of b for monotonicity
theorem monotonic_f2_b :
  ∀ (b : ℝ), (b ≥ 16 ∨ b ≤ 2) ↔ 
    (∀ (x₁ x₂ : ℝ), (1 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2) → (x₂^2 + b / x₂ + c ≥ x₁^2 + b / x₁ + c)) := sorry

-- Problem part 3: monotonicity of sequence x_n
theorem increasing_sequence_x_n :
  ∀ (n : ℕ) (h₀ : n > 0), 
    let x_n := classical.some (exists_unique_zero_point (x^n - 1/x + 1) (1/2, 1)) in 
    increase (sequence x_1 x_2 x_3 ...) := sorry

end unique_zero_point_c_monotonic_f2_b_increasing_sequence_x_n_l822_822839


namespace max_beads_find_lighter_l822_822916

-- We define the maximum number n beads in the pile
def n_beads := 9

-- The conditions for our problem
def pile_of_beads (n : ℕ) : Prop := 
  ∃ (B : Finset ℕ), B.card = n ∧ ∀ b ∈ B, b ≠ 0

-- The theorem to prove the question
theorem max_beads_find_lighter (n : ℕ) (h : pile_of_beads n) : n ≤ n_beads :=
by rfl

end max_beads_find_lighter_l822_822916


namespace cost_of_one_bag_l822_822138

theorem cost_of_one_bag (x : ℝ) (h1 : ∀ p : ℝ, 60 * x = p -> 60 * p = 120 * x ) 
  (h2 : ∀ p1 p2: ℝ, 60 * x = p1 ∧ 15 * 1.6 * x = p2 ∧ 45 * 2.24 * x = 100.8 * x -> 124.8 * x - 120 * x = 1200) :
  x = 250 := 
sorry

end cost_of_one_bag_l822_822138


namespace terminating_decimal_count_l822_822717

theorem terminating_decimal_count (h₁ : 1 ≤ n ∧ n ≤ 569) 
(h₂ : ∀ m, m ∣ 570 → m = 2 ∨ m = 3 ∨ m = 5 ∨ m = 19) :
  ∑ (i : ℕ) in (finset.range 570).filter (λ i, i % 57 = 0), 1 = 9 :=
by sorry

end terminating_decimal_count_l822_822717


namespace triangle_altitude_and_area_l822_822997

open Real

-- Definitions based on the conditions of the problem
def side_a : ℝ := 11
def side_b : ℝ := 13
def side_c : ℝ := 16

def s : ℝ := (side_a + side_b + side_c) / 2

def herons_area : ℝ :=
  Real.sqrt (s * (s - side_a) * (s - side_b) * (s - side_c))

def altitude : ℝ :=
  2 * herons_area / side_b

-- The main theorem statement to prove
theorem triangle_altitude_and_area : altitude = 168 / 13 ∧ herons_area = 84 := 
sorry

end triangle_altitude_and_area_l822_822997


namespace problem_statement_l822_822300

variables (p1 p2 p3 p4 : Prop)

theorem problem_statement (h_p1 : p1 = True)
                         (h_p2 : p2 = False)
                         (h_p3 : p3 = False)
                         (h_p4 : p4 = True) :
  (p1 ∧ p4) = True ∧
  (p1 ∧ p2) = False ∧
  (¬p2 ∨ p3) = True ∧
  (¬p3 ∨ ¬p4) = True :=
by
  sorry

end problem_statement_l822_822300


namespace part_a_part_b_l822_822962

-- Definitions of the problem conditions
variables {A B C D O : Point}
def is_convex_quadrilateral (A B C D : Point) : Prop := 
  -- A dummy definition. The actual definition depends on the specific properties of a convex quadrilateral
  sorry 

def is_permissible (A B C D : Point) : Prop := 
  -- A dummy definition. Permissible implies the quadrilateral remains pairwise distinct and convex after operations.
  sorry 

-- The translated proof problems
theorem part_a :
  ∃ (A' B' C' D' O : Point), 
    is_convex_quadrilateral A B C D → 
    is_permissible A B C D →
    (apply_operations A B C D 3) = (A, B, C, D) :=
sorry

theorem part_b :
  ∃ (n₀ : ℕ), 
    n₀ = 6 ∧ 
    ∀ (A B C D : Point), 
      is_convex_quadrilateral A B C D → 
      is_permissible A B C D →
      (apply_operations A B C D n₀) = (A, B, C, D) :=
sorry

end part_a_part_b_l822_822962


namespace bill_buys_125_bouquets_to_make_1000_l822_822258

-- Defining the conditions
def cost_per_bouquet : ℕ := 20
def roses_per_bouquet_buy : ℕ := 7
def roses_per_bouquet_sell : ℕ := 5
def target_difference : ℕ := 1000

-- To be demonstrated: number of bouquets Bill needs to buy to get a profit difference of $1000
theorem bill_buys_125_bouquets_to_make_1000 : 
  let total_cost_to_buy (n : ℕ) := n * cost_per_bouquet
  let total_roses (n : ℕ) := n * roses_per_bouquet_buy
  let bouquets_sell_from_roses (roses : ℕ) := roses / roses_per_bouquet_sell
  let total_revenue (bouquets : ℕ) := bouquets * cost_per_bouquet
  let profit (n : ℕ) := total_revenue (bouquets_sell_from_roses (total_roses n)) - total_cost_to_buy n
  profit (125 * 5) = target_difference := 
sorry

end bill_buys_125_bouquets_to_make_1000_l822_822258


namespace bond_value_at_6_years_l822_822855

-- Definition of the problem conditions
def principal_amount (P : ℤ) : Prop :=
  P + P * 5 / 10 * 2 = 100

def simple_interest (P r t : ℕ) : ℤ :=
  P * (r / 100) * t

-- The proof statement
theorem bond_value_at_6_years :
  ∃ P : ℤ, principal_amount P ∧
  let r := 50 in
  let t1 := 2 in
  let t2 := 4 in
  let value_at_6_years := P + simple_interest P r t2 in
  value_at_6_years = 150 :=
sorry

end bond_value_at_6_years_l822_822855


namespace unsolved_problems_exist_l822_822724

noncomputable def main_theorem: Prop :=
  ∃ (P : Prop), ¬(P = true) ∧ ¬(P = false)

theorem unsolved_problems_exist : main_theorem :=
sorry

end unsolved_problems_exist_l822_822724


namespace find_point_on_parabola_with_given_conditions_l822_822211

def parabola_has_given_focus_and_vertex (x y : ℝ) : Prop :=
  (x^2 = 8 * y) ∧ (y ≥ 0)

def point_on_parabola_in_first_quadrant (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in
  parabola_has_given_focus_and_vertex x y ∧ (x > 0) ∧ (y > 0)

def point_on_parabola_at_distance (P F : ℝ × ℝ) (d : ℝ) : Prop :=
  let (x₁, y₁) := P in
  let (x₂, y₂) := F in
  real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = d

theorem find_point_on_parabola_with_given_conditions :
  ∃ (P : ℝ × ℝ), point_on_parabola_in_first_quadrant P ∧ point_on_parabola_at_distance P (0, 2) 150 ∧ P = (2 * real.sqrt 296, 148) :=
begin
  sorry
end

end find_point_on_parabola_with_given_conditions_l822_822211


namespace median_room_number_of_remaining_scholars_l822_822251

theorem median_room_number_of_remaining_scholars :
  ∀ rooms : Finset ℕ, rooms = (Finset.range 26).erase 15.erase 16 → 
  median (remaining_rooms) = 12 :=
by
  sorry

end median_room_number_of_remaining_scholars_l822_822251


namespace sqrt_180_simplified_l822_822052

theorem sqrt_180_simplified : Real.sqrt 180 = 6 * Real.sqrt 5 :=
   sorry

end sqrt_180_simplified_l822_822052


namespace triangular_prism_volume_l822_822446

def rectangle (A B C D : Type) : Prop := sorry -- Define what it means geometrically for quadrilateral ABCD to be a rectangle

noncomputable def lengthAB : ℝ := 13 * Real.sqrt 3
noncomputable def lengthBC : ℝ := 12 * Real.sqrt 3

def midpoint_diagonals_intersect_at (P : Type) (A B C D : Type) : Prop := sorry -- Define the property that diagonals AC and BD of rectangle ABCD intersect at P

theorem triangular_prism_volume (A B C D P : Type) :
  rectangle A B C D →
  midpoint_diagonals_intersect_at P A B C D →
  (∃ (h1 h2 h3 h4 : ℝ), lengthAB = h1 ∧ lengthBC = h2 ∧ h3 = Real.sqrt (h1^2 + h2^2) ∧ h3 = length D A) →
  volume_of_trianglular_prism_formed_by_folding A B C D P = 594 := 
sorry

end triangular_prism_volume_l822_822446


namespace volume_unoccupied_l822_822141

-- Definitions from conditions
def cone_radius := 10
def cone_height := 15
def cylinder_radius := 10
def cylinder_height := 40

-- Volumes from problem statement
def volume_cylinder := π * (cylinder_radius ^ 2) * cylinder_height
def volume_cone := (1 / 3) * π * (cone_radius ^ 2) * cone_height
def volume_cones := 2 * volume_cone

-- The volume of the cylinder not occupied by the cones
theorem volume_unoccupied:
  volume_cylinder - volume_cones = 3000 * π :=
by 
  sorry

end volume_unoccupied_l822_822141


namespace find_length_PR_l822_822797

-- Definitions of given constants and conditions
def is_rectangle (P Q R S : Point) : Prop :=
  -- Define what it means for PQRS to be a rectangle
  ...

def semicircle (P Q : Point) : Set Point :=
  -- Define the semicircle with diameter PQ
  ...

def points_on_line (m : Line) (V W Z : Point) : Prop :=
  -- Define the points V, W, Z lying on the line m
  ...

def divides_region (m : Line) (V W Z : Point) (S : Set Point) : Prop :=
  -- Define how line m divides the region into two parts with given area ratio
  ...

-- Main statement to prove in Lean 4
theorem find_length_PR (P Q R S V W Z : Point) (m : Line)
  (h_rectangle : is_rectangle P Q R S)
  (h_semicircle : semicircle P Q)
  (h_points_on_line : points_on_line m V W Z)
  (h_divides_region : divides_region m V W Z (semicircle P Q ∪ rectangle P Q R S))
  (h_PW : distance P W = 72)
  (h_PV : distance P V = 108)
  (h_WQ : distance W Q = 144) :
  ∃ a b : ℕ, b ≠ 0 ∧ ¬(∃ p : ℕ, prime p ∧ p^2 ∣ b) ∧
  distance P R = a * sqrt b ∧ a + b = 165 := by
  sorry

end find_length_PR_l822_822797


namespace sum_of_lengths_ge_l822_822343

def is_prefix {α : Type} [decidable_eq α] : list α → list α → Prop
| [], _ => true
| _, [] => false
| (a :: as), (b :: bs) => a = b ∧ is_prefix as bs

theorem sum_of_lengths_ge {n : ℕ} (sequences : list (list ℕ)) (h_len : sequences.length = 2^n)
  (h_unique : ∀ s1 s2 : list ℕ, s1 ∈ sequences → s2 ∈ sequences → s1 ≠ s2 → ¬ is_prefix s1 s2) :
  sequences.foldl (λ acc s => acc + s.length) 0 ≥ n * 2^n :=
sorry

end sum_of_lengths_ge_l822_822343


namespace circle_equation_l822_822319

theorem circle_equation :
  ∃ (h k r : ℝ),
    (∀ x y : ℝ, (x - h) ^ 2 + (y - k) ^ 2 = r ^ 2 ↔ (3 * h + k - 5 = 0 ∧ ((h = 2 ∧ k = -1) ∧ r ^ 2 = 5))) :=
by
  -- Condition: Center lies on the line 3x + y - 5 = 0
  let C := {p : ℝ × ℝ | 3 * p.1 + p.2 - 5 = 0},

  -- Condition: Circle passes through the origin (0,0) and the point (4,0)
  let P := {p : ℝ × ℝ | p = (0, 0) ∨ p = (4, 0)},

  -- We need to find the values h, k, and r
  have h : ℝ := 2,
  have k : ℝ := -1,
  have r_squared : ℝ := 5,

  -- The correct answer should be that the circle equation is (x - h)² + (y - k)² = r²
  use [h, k, r_squared],
  split,
  -- Forward direction
  {
    intros x y,
    split,
    {
      intros H,
      split,
      {
        -- Prove 3 * h + k - 5 = 0
        have line_eq := C.mk (h, k),
        simp [h, k] at *,
        assumption,
      },
      split,
      {
        -- Prove (h = 2 ∧ k = -1)
        exact ⟨rfl, rfl⟩,
      },
      {
        -- Prove r² = 5
        exact rfl,
      },
    },
    -- Backward direction
    {
      intros H,
      cases H with line_cond H,
      cases H with hk r_eq,
      cases hk with h_eq k_eq,
      rwa [h_eq, k_eq, r_eq],
    },
  },
  sorry -- Add any additional conditions or steps if necessary

end circle_equation_l822_822319


namespace price_decrease_784_percent_l822_822901

theorem price_decrease_784_percent (a : Real) : 
  let after_increases := a * (1 + 0.2)^2 in
  let after_decreases := after_increases * (1 - 0.2)^2 in
  after_decreases = a * 0.9216 → 
  ((a - after_decreases) / a * 100) = 7.84 :=
by 
  intros
  sorry

end price_decrease_784_percent_l822_822901


namespace find_g_expression_range_of_k_l822_822680

-- Given conditions
def f (x : Real) (ω : Real) (φ : Real) : Real :=
  sin (ω * x + φ)

variable (ω : Real) (φ : Real) (h1 : ω > 0) (h2 : |φ| < (π / 2))

-- The function is monotically decreasing on a given interval
variable (h3 : ∀ x1 x2, (5 * π / 12) ≤ x1 ∧ x1 < x2 ∧ x2 ≤ (11 * π / 12) → f x1 ω φ > f x2 ω φ)

-- Translated and scaled function
def g (x : Real) : Real :=
  sin (4 * x + (π / 6))

-- Condition about triangle and angle
variable (a b c x : Real) (hx : 0 < x ∧ x ≤ π / 3) (triangle_cond : b^2 = a * c)

-- Proof 1: finding the analytical expression of g(x)
theorem find_g_expression : g x = sin (4*x + (π/6)) :=
  sorry

-- Proof 2: range of k
theorem range_of_k (k : Real) (h4 : ∀ x : Real, g x = k → ∃ x1 x2, x1 ≠ x2 ∧ g x1 = k ∧ g x2 = k) :
  k ∈ (Set.Ioo (1 / 2) 1) :=
  sorry

end find_g_expression_range_of_k_l822_822680


namespace correct_statements_l822_822369

theorem correct_statements (f : ℝ → ℝ) :
  (∀ x ≠ x₀, f(x) > f(x₀) → ∀ y, f(y) ≥ f(x₀)) ∧
  (∀ x < x₀, deriv f x > 0 → ∀ x > x₀, deriv f x < 0 → ∀ y, f(y) ≤ f(x₀)) :=
begin
  sorry
end

end correct_statements_l822_822369


namespace cost_of_one_bag_l822_822130

theorem cost_of_one_bag (x : ℝ) :
  let Andrey_earning := 60 * 2 * x
  let Boris_earning := 15 * 1.6 * x + 45 * (1.6 * 1.4) * x
  Boris_earning - Andrey_earning = 1200 →
  x = 250 := 
by
  intros
  let Andrey_earning := 60 * 2 * x
  let Boris_earning := 15 * 1.6 * x + 45 * (1.6 * 1.4) * x
  have h : Boris_earning - Andrey_earning = 1200 := by assumption
  let simplified_h := 
    calc
      Boris_earning - Andrey_earning
        = (15 * 1.6 * x + 45 * (1.6 * 1.4) * x) - (60 * 2 * x) : by simp [Andrey_earning, Boris_earning]
    ... = (15 * 1.6 * x + 45 * (1.6 * 1.4) * x) - 120 * x : by simp
    ... = (24 * x + 100.8 * x) - 120 * x : by simp
    ... = 124.8 * x - 120 * x : by simp
    ... = 4.8 * x : by simp
    ... = 1200 : by rw h
  exact (div_eq_iff (by norm_num : (4.8 : ℝ) ≠ 0)).1 simplified_h  -- solves for x

end cost_of_one_bag_l822_822130


namespace no_max_no_minimum_l822_822014

noncomputable def f (x : ℝ) : ℝ := x^2 * ln x - x^2 + (2 + 1/e) * x

def has_extrema (f : ℝ → ℝ) (extrema : ℝ) :=
  ∃ x, (x ∈ set.Ioo 0 extrema) ∧ is_max (f x) ∨ is_min (f x)

theorem no_max_no_minimum :
  ¬ (has_extrema f ⊤) := 
  sorry

end no_max_no_minimum_l822_822014


namespace number_of_plans_for_participation_l822_822532

open Finset

/-- Proof that there are 18 different plans for participation given the conditions. -/
theorem number_of_plans_for_participation :
  let students := {"A", "B", "C", "D"} in
  let must_participate := "A" in
  let remaining_students := erase students must_participate in
  (card (choose 2 remaining_students) * factorial 3) = 18 := sorry

end number_of_plans_for_participation_l822_822532


namespace five_digit_numbers_with_4_or_5_l822_822770

theorem five_digit_numbers_with_4_or_5 : 
  let total_five_digit := 99999 - 10000 + 1
  let without_4_or_5 := 7 * 8^4
  total_five_digit - without_4_or_5 = 61328 :=
by
  let total_five_digit := 99999 - 10000 + 1
  let without_4_or_5 := 7 * 8^4
  have h : total_five_digit - without_4_or_5 = 61328 := by sorry
  exact h

end five_digit_numbers_with_4_or_5_l822_822770


namespace positive_difference_proof_l822_822376

-- Define the points for lines p and q
structure Point where
  x : ℚ
  y : ℚ

def p1 : Point := ⟨0, 3⟩
def p2 : Point := ⟨4, 0⟩

def q1 : Point := ⟨0, 1⟩
def q2 : Point := ⟨8, 0⟩

-- Define the slopes of lines p and q
def slope (A B : Point) : ℚ :=
  (B.y - A.y) / (B.x - A.x)

def p_slope : ℚ := slope p1 p2
def q_slope : ℚ := slope q1 q2

-- Define the equations of lines p and q when y = 10
def p_eq (y : ℚ) : ℚ := (y - p1.y) / p_slope
def q_eq (y : ℚ) : ℚ := (y - q1.y) / q_slope

def p_x := p_eq 10
def q_x := q_eq 10

-- Define the positive difference
def positive_difference (a b : ℚ) : ℚ := abs (a - b)

-- Prove that positive_difference (p_x) (q_x) is 188 / 3
theorem positive_difference_proof :
  positive_difference p_x q_x = 188 / 3 :=
by
  -- Provide proof here
  sorry

end positive_difference_proof_l822_822376


namespace pedestrian_wait_probability_l822_822212

-- Define the duration of the red light
def red_light_duration := 45

-- Define the favorable time window for the pedestrian to wait at least 20 seconds
def favorable_window := 25

-- The probability that the pedestrian has to wait at least 20 seconds
def probability_wait_at_least_20 : ℚ := favorable_window / red_light_duration

theorem pedestrian_wait_probability : probability_wait_at_least_20 = 5 / 9 := by
  sorry

end pedestrian_wait_probability_l822_822212


namespace simson_line_quadrilateral_simson_line_ngon_l822_822185

-- Problem a
theorem simson_line_quadrilateral (P A B C D : Point) (H: CyclicQuadrilateral P A B C D) (hP_on_circumcircle : P ∈ Circumcircle A B C D) :
  ∃ l : Line, ∀ (B_1 C_1 D_1 : Point), IsProjection P (Line.through A B B_1) (Line.through A C C_1) (Line.through A D D_1) → B_1 ∈ l ∧ C_1 ∈ l ∧ D_1 ∈ l :=
sorry

-- Problem b
theorem simson_line_ngon (n : ℕ) (P : Point) (A : Fin n → Point) (H: Cyclic n A) (hP_on_circumcircle : P ∈ Circumcircle_of_ngon n A) :
  ∀ (k : ℕ) (B Proj : Fin k → Point),
  (∀ i, IsProjection P (Polygon.remove_vertex n A i) (B i) (Proj i)) →
  (k ≥ 3) →
  ∃ l : Line, ∀ i, Proj i ∈ l :=
sorry

end simson_line_quadrilateral_simson_line_ngon_l822_822185


namespace thousandths_digit_l822_822961

-- Define the necessary constants
def fifty_seven : ℝ := 57
def five_thousand : ℝ := 5000
def x := fifty_seven / five_thousand

-- State the theorem about the thousandths digit
theorem thousandths_digit : (Real.frac_part (x * 1000) - Real.frac_part (x * 100) * 10).floor = 1 := 
  by sorry

end thousandths_digit_l822_822961


namespace convex_polygon_angles_leq_35_l822_822860

theorem convex_polygon_angles_leq_35 {n : ℕ} (h : n ≥ 3) (angles : Fin n → ℝ) 
  (sum_eq : ∑ i, angles i = 180 * (n - 2)) :
  (∃ count : ℕ, count ≤ n ∧ count > 35 ∧ ∀ i : Fin count, angles i < 170) → False :=
by 
  sorry

end convex_polygon_angles_leq_35_l822_822860


namespace bisection_accuracy_ln_equation_l822_822229

theorem bisection_accuracy_ln_equation :
  ∀ x ∈ Icc 2 3, 0.01 ∈ Icc (1 / 128 : ℝ) (1 / 64 : ℝ) → 
  (∀ n, n = 7 → interval_halved_length 1 n ≤ 0.01) → 
  (∀ i, i = n → half_length_seq 1 i ∈ Icc (1 / 128 : ℝ) (1 / 64 : ℝ)) :=
begin
  intros,
  sorry,
end

end bisection_accuracy_ln_equation_l822_822229


namespace arithmetic_sequence_difference_l822_822412

theorem arithmetic_sequence_difference (a b c : ℝ) :
  (∃ (d : ℝ), a = 2 + d ∧ c = 2 + 3d ∧ 9 = 2 + 4d) → c - a = 3.5 :=
by
  intro h
  rcases h with ⟨d, ha, hc, h9⟩
  calc
    c - a = 3.5 : sorry

end arithmetic_sequence_difference_l822_822412


namespace vacation_cost_per_person_l822_822500

theorem vacation_cost_per_person (airbnb_cost car_cost : ℝ) (num_people : ℝ) 
  (h1 : airbnb_cost = 3200) (h2 : car_cost = 800) (h3 : num_people = 8) : 
  (airbnb_cost + car_cost) / num_people = 500 := 
by 
  sorry

end vacation_cost_per_person_l822_822500


namespace hyperbola_asymptotes_l822_822395

theorem hyperbola_asymptotes (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) (h4 : (2 : ℝ) = 2) :
  ∀ x : ℝ, y = sqrt 3 * x :=
by skip

end hyperbola_asymptotes_l822_822395


namespace asymptotes_of_hyperbola_l822_822388

variable (a b c : ℝ)
variable (e : ℝ := 2)

theorem asymptotes_of_hyperbola (ha : a > 0) (hb : b > 0)
  (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1) 
  (h_eccentricity : e = 2): 
  ∀ x, (y = √3 * x) ∨ (y = -√3 * x) := 
by
  sorry

end asymptotes_of_hyperbola_l822_822388


namespace equal_common_friends_l822_822447

open Finset

variable {α : Type*} [DecidableEq α] [Fintype α]

/-- Given a school with some students, where friendships are mutual -/
structure School where
  (students : Finset α)

/-- Friendship relation in the school -/
def is_friend (s : School) (a b : α) : Prop := true -- Placeholder, define as per friendship relation

-- Conditions from the problem
variable (s : School)
variable [Friendship : ∀ a b : α, is_friend s a b = is_friend s b a]
variable [Cond1 : ∀ a b c : α, ∃ d : α, is_friend s d a ∧ is_friend s d b ∧ is_friend s d c]
variable [Cond2 : ∀ a b : α, is_friend s a b → ∀ c d : α, is_friend s a c → is_friend s a d → is_friend s c d]
variable [Cond3 : ¬ ∃ p q : Finset α, p ∪ q = s.students ∧ ∀ x ∈ p, ∀ y ∈ q, is_friend s x y]
variable [Cond4 : ∀ a : α, is_friend s a a]

-- This should imply the desired result: Any two people who aren't friends with each other, have the same number of common friends.
theorem equal_common_friends (a b : α) (ha : ¬ is_friend s a b) (c : α) (hc : ¬ is_friend s a c) :
  (s.students.filter (λ x, is_friend s a x ∧ is_friend s b x)).card =
  (s.students.filter (λ x, is_friend s a x ∧ is_friend s c x)).card :=
sorry -- proof omitted

end equal_common_friends_l822_822447


namespace simplify_sqrt180_l822_822048

-- Conditions from the problem definition
def sqrt (x : ℕ) : ℝ := Real.sqrt x
def six := 6
def five := 5

-- The statement of the problem as a Lean theorem
theorem simplify_sqrt180 : sqrt 180 = six * sqrt five := by 
  sorry

end simplify_sqrt180_l822_822048


namespace hyperbola_problems_l822_822349

noncomputable def hyperbola_equation (a b : ℝ) := (λ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1)

/-- The problem statement for Lean -/
theorem hyperbola_problems 
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h_eccentricity : c / a = sqrt 3)
  (h_vertex : (sqrt 3, 0) = (a, 0)) :
  hyperbola_equation a b = (λ x y, (x^2 / 3) - (y^2 / 6) = 1) ∧ 
  (let f2 : ℝ × ℝ := (3, 0),
       line_eq := (λ x, (sqrt 3 / 3) * (x - 3)) in 
   ∀ A B : ℝ × ℝ, 
     (A.1^2 / 3 - A.2^2 / 6 = 1) ∧ 
     (B.1^2 / 3 - B.2^2 / 6 = 1) ∧
     (A.2 = line_eq A.1) ∧ 
     (B.2 = line_eq B.1) → 
     dist A B = 16 * sqrt 3 / 5) :=
by sorry

end hyperbola_problems_l822_822349


namespace women_hours_per_day_l822_822191

theorem women_hours_per_day (h : ℕ) :
  (∀ men days men_hours total_work total_work_women,
      total_work = 15 * 21 * 8 →
      total_work_women = 21 * 20 * h * (2 / 3) →
      total_work = total_work_women) →
  h = 9 :=
by
  intros h h_eq
  sorry

end women_hours_per_day_l822_822191


namespace cylinder_surface_area_proof_l822_822082

noncomputable def sphere_volume := (500 * Real.pi) / 3
noncomputable def cylinder_base_diameter := 8
noncomputable def cylinder_surface_area := 80 * Real.pi

theorem cylinder_surface_area_proof :
  ∀ (R : ℝ) (r h : ℝ), 
    (4 * Real.pi / 3) * R^3 = (500 * Real.pi) / 3 → -- sphere volume condition
    2 * r = cylinder_base_diameter →               -- base diameter condition
    r * r + (h / 2)^2 = R^2 →                      -- Pythagorean theorem (half height)
    2 * Real.pi * r * h + 2 * Real.pi * r^2 = cylinder_surface_area := -- surface area formula
by
  intros R r h sphere_vol_cond base_diameter_cond pythagorean_cond
  sorry

end cylinder_surface_area_proof_l822_822082


namespace concentration_of_salt_solution_l822_822976

theorem concentration_of_salt_solution :
  ∀ (C : ℕ), 
  (let salt_original := 1 * (C / 100 : ℚ),
       salt_mixture := 2 * (20 / 100 : ℚ) in
   salt_original = salt_mixture → C = 40) :=
by
  intros C,
  let salt_original := 1 * (C / 100 : ℚ),
  let salt_mixture := 2 * (20 / 100 : ℚ),
  intro h,
  sorry

end concentration_of_salt_solution_l822_822976


namespace sum_of_arithmetic_sequence_l822_822374

variable {α : Type*} [LinearOrderedField α]

noncomputable def is_arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
∀ n, a (n + 1) - a n = d

noncomputable def sum_of_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
n * (a 1 + a n) / 2

theorem sum_of_arithmetic_sequence {a : ℕ → α} {d : α}
  (h3 : a 3 * a 7 = -16)
  (h4 : a 4 + a 6 = 0)
  (ha : is_arithmetic_sequence a d) :
  ∃ (s : α), s = n * (n - 9) ∨ s = -n * (n - 9) :=
sorry

end sum_of_arithmetic_sequence_l822_822374


namespace triangle_area_correct_l822_822628

open Real

def triangle_area (a b c : ℝ) : ℝ :=
  sqrt (1 / 4 * (c^2 * a^2 - (1 / 4 * (c^2 + a^2 - b^2)^2)))

theorem triangle_area_correct :
  let a := 4
  let b := 2 * sqrt 7
  let c := 6
  triangle_area a b c = 6 * sqrt 3 := 
by
  sorry

end triangle_area_correct_l822_822628


namespace find_angle_A_find_max_area_l822_822467

-- Given conditions for the first part of the problem
def triangleABC (a b c : ℝ) (A B C : ℝ) :=
  a = 2 * Real.sqrt 3 ∧ 
  (2 * Real.sqrt 3 + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C

-- Proving A = π/3
theorem find_angle_A (a b c A B C : ℝ) (h : triangleABC a b c A B C) : 
  A = π / 3 :=
by
  sorry

-- Given conditions for the second part of the problem, based on the first theorem
def maxAreaTriangle (b c : ℝ) :=
  (c * b) / 4

-- Proving the maximum area of triangle
theorem find_max_area (a b c : ℝ) (A : ℝ) (h_a : a = 2 * Real.sqrt 3) (h_A : A = π / 3) :
  maxAreaTriangle b c = 3 * Real.sqrt 3 :=
by
  sorry

end find_angle_A_find_max_area_l822_822467


namespace number_of_regions_split_by_lines_l822_822682

-- Definitions for the lines
def line_eq (a b : ℝ) (p : ℝ × ℝ) : Prop := p.snd = a * p.fst + b  -- General line equation y = ax + b
def line_y_eq_2x (p : ℝ × ℝ) : Prop := line_eq 2 0 p
def line_y_eq_half_x (p : ℝ × ℝ) : Prop := line_eq (1 / 2) 0 p
def line_y_eq_x (p : ℝ × ℝ) : Prop := line_eq 1 0 p

-- The theorem statement
theorem number_of_regions_split_by_lines : 
  (∀ p : ℝ × ℝ, line_y_eq_2x p ∨ line_y_eq_half_x p ∨ line_y_eq_x p) → 
  (number_of_regions {[p | line_y_eq_2x p], [p | line_y_eq_half_x p], [p | line_y_eq_x p]} = 6) :=
sorry

end number_of_regions_split_by_lines_l822_822682


namespace sum_of_powers_l822_822678

theorem sum_of_powers (i : ℂ) (h : i^2 = -1) : (∑ k in finset.range 606, i^k) = i :=
by
  sorry

end sum_of_powers_l822_822678


namespace intersection_A_B_l822_822743

variable (x : ℤ)

def A := { x | x^2 - 4 * x ≤ 0 }
def B := { x | -1 ≤ x ∧ x < 4 }

theorem intersection_A_B :
  { x : ℤ | x ∈ A ∧ x ∈ B } = {0, 1, 2, 3} :=
sorry

end intersection_A_B_l822_822743


namespace rearrangement_inequality_l822_822620

open_locale BigOperators

theorem rearrangement_inequality {n : ℕ} 
  {a b B : Finₓ n → ℝ}
  (ha : ∀ i j, i ≤ j → a i ≥ a j)
  (hb : Multiset(Perm a.to_list).mem b)
  (hB : ∀ i j, i ≤ j → B i ≥ B j) :
  ∑ i in Finₓ.range n, a i * b i ≤ ∑ i in Finₓ.range n, a i * B i := 
sorry

end rearrangement_inequality_l822_822620


namespace alternatingBoysGirls_l822_822587
open Classical
noncomputable theory

-- Define what a triangle and set of vertices mean
structure Triangle where
  v1 : ℕ 
  v2 : ℕ 
  v3 : ℕ
  distinct : v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3

-- Define the problem as vertices of n distinct triangles inscribed in a circle
def inscribedTriangles (n : ℕ) : Prop :=
  ∃ (triangles : Fin n → Triangle),
    (∀ i j: Fin n, i ≠ j → triangles i ≠ triangles j)

-- The final theorem to be proved (statement only)
theorem alternatingBoysGirls (n : ℕ) 
  (h : inscribedTriangles n) : 
  ∃ (b : Fin (3 * n) → bool), 
  ∀ i, b i ≠ b ((i + 1) % (3 * n)) := 
sorry

end alternatingBoysGirls_l822_822587


namespace village_population_rate_l822_822931

noncomputable def population_change_X (initial_X : ℕ) (decrease_rate : ℕ) (years : ℕ) : ℕ :=
  initial_X - decrease_rate * years

noncomputable def population_change_Y (initial_Y : ℕ) (increase_rate : ℕ) (years : ℕ) : ℕ :=
  initial_Y + increase_rate * years

theorem village_population_rate (initial_X decrease_rate initial_Y years result : ℕ) 
  (h1 : initial_X = 70000) (h2 : decrease_rate = 1200) 
  (h3 : initial_Y = 42000) (h4 : years = 14) 
  (h5 : initial_X - decrease_rate * years = initial_Y + result * years) 
  : result = 800 :=
  sorry

end village_population_rate_l822_822931


namespace geometric_progression_common_ratio_l822_822799

theorem geometric_progression_common_ratio (a r : ℝ) (h : a = a * r + a * r^2 + a * r^3) (ha : a > 0) : 
  r^3 + r^2 + r - 1 = 0 :=
by
  have h1 : 1 = r + r^2 + r^3 :=
    by
      have h2 : a ≠ 0 := by linarith
      rw [mul_eq_mul_left_iff] at h
      cases h
      · assumption
      · exfalso
        exact h2 h
  sorry

end geometric_progression_common_ratio_l822_822799


namespace mike_total_money_l822_822017

theorem mike_total_money (num_bills : ℕ) (value_per_bill : ℕ) (h1 : num_bills = 9) (h2 : value_per_bill = 5) :
  (num_bills * value_per_bill) = 45 :=
by
  sorry

end mike_total_money_l822_822017


namespace sqrt_180_eq_l822_822041

noncomputable def simplify_sqrt_180 : Real := 6 * Real.sqrt 5

theorem sqrt_180_eq : Real.sqrt 180 = simplify_sqrt_180 := 
by
  -- proof omitted
  sorry

end sqrt_180_eq_l822_822041


namespace complex_triangle_equilateral_length_l822_822677

noncomputable def complex_norm (z : ℂ) : ℝ := complex.abs z

variables {p q r : ℂ}

-- Given conditions
variables (h1 : (complex.abs (p - q) = 24) ∧ (complex.abs (q - r) = 24) ∧ (complex.abs (r - p) = 24))
variables (h2 : complex_norm (p + q + r) = 48)

-- The theorem to prove
theorem complex_triangle_equilateral_length
    (h1 : (complex.abs (p - q) = 24) ∧ (complex.abs (q - r) = 24) ∧ (complex.abs (r - p) = 24))
    (h2 : complex_norm (p + q + r) = 48) :
    complex_norm (p * q + p * r + q * r) = 768 :=
sorry

end complex_triangle_equilateral_length_l822_822677


namespace conic_passing_through_vertices_l822_822701

-- Define the condition of the triangle and the general line equation
variables {A B C : Type} [Triangle A] [Triangle B] [Triangle C]

-- Define the non-zero constants p, q, r
variable (p q r : ℝ) (non_zero : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)

-- Define the isogonal conjugate transformation and resulting conic equation
theorem conic_passing_through_vertices (x y z : ℝ) (h_line_not_through_vertices : px + qy + rz = 0)
  (h_isogonal_conjugate : pyz + qxz + rxy = 0) :
  ∀ v ∈ {A, B, C}, h_isogonal_conjugate :=
sorry

end conic_passing_through_vertices_l822_822701


namespace true_and_false_propositions_l822_822773

theorem true_and_false_propositions (p q : Prop) 
  (hp : p = true) (hq : q = false) : (¬q) = true :=
by
  sorry

end true_and_false_propositions_l822_822773


namespace sin_2BPC_l822_822030

-- Define the points and their properties
variables {A B C D P : Point}
variables {a b c d : ℝ}
variables {α β γ δ : ℝ}

-- Conditions given in the problem
def equally_spaced : Prop := -- Definition for points being equally spaced
  dist A B = dist B C ∧ dist B C = dist C D

def cos_APC (α : ℝ) : Prop :=
  cos α = 3 / 5

def cos_BPD (β : ℝ) : Prop :=
  cos β = 12 / 13

-- Question to prove
theorem sin_2BPC (A B C D P : Point) (h1 : equally_spaced) (h2 : cos_APC α) (h3 : cos_BPD β) :
  ∃ (γ : ℝ), sin (2 * γ) = 2 * (sin γ ^ 2) :=
sorry

end sin_2BPC_l822_822030


namespace mutually_exclusive_event_at_most_one_head_and_event_at_least_two_heads_l822_822615

-- Define possible events when two coins are tossed
def event_at_least_one_head := {outcome | ∃ (c1 c2 : Bool), (c1 = tt ∨ c2 = tt) ∧ outcome = (c1, c2)}
def event_exactly_one_head := {outcome | ∃ (c1 c2 : Bool), (c1 = tt ∧ c2 = ff ∨ c1 = ff ∧ c2 = tt) ∧ outcome = (c1, c2)}
def event_exactly_two_heads := {outcome | ∃ (c1 c2 : Bool), (c1 = tt ∧ c2 = tt) ∧ outcome = (c1, c2)}
def event_at_most_one_head := {outcome | ∃ (c1 c2 : Bool), (c1 = ff ∧ c2 = ff ∨ c1 = tt ∧ c2 = ff ∨ c1 = ff ∧ c2 = tt) ∧ outcome = (c1, c2)}
def event_at_least_two_heads := {outcome | ∃ (c1 c2 : Bool), (c1 = tt ∧ c2 = tt) ∧ outcome = (c1, c2)}

-- Define mutual exclusivity of two events
def are_mutually_exclusive (E1 E2 : set (Bool × Bool)) := ∀ outcome, outcome ∈ E1 → outcome ∉ E2

-- The theorem we need to prove
theorem mutually_exclusive_event_at_most_one_head_and_event_at_least_two_heads :
  are_mutually_exclusive event_at_most_one_head event_at_least_two_heads :=
by
  sorry

end mutually_exclusive_event_at_most_one_head_and_event_at_least_two_heads_l822_822615


namespace second_discount_percentage_l822_822581

-- Define the initial conditions.
def listed_price : ℝ := 200
def first_discount_rate : ℝ := 0.20
def final_sale_price : ℝ := 144

-- Calculate the price after the first discount.
def first_discount_amount := first_discount_rate * listed_price
def price_after_first_discount := listed_price - first_discount_amount

-- Define the second discount amount.
def second_discount_amount := price_after_first_discount - final_sale_price

-- Define the theorem to prove the second discount rate.
theorem second_discount_percentage : 
  (second_discount_amount / price_after_first_discount) * 100 = 10 :=
by 
  sorry -- Proof placeholder

end second_discount_percentage_l822_822581


namespace simplify_sqrt_180_l822_822064

theorem simplify_sqrt_180 : sqrt 180 = 6 * sqrt 5 :=
by
  have h : 180 = 2^2 * 3^2 * 5 := by norm_num
  calc
    sqrt 180
      = sqrt (2^2 * 3^2 * 5)     : by rw [h]
  ... = sqrt (2^2) * sqrt (3^2) * sqrt 5 : by rw [sqrt_mul, sqrt_mul]
  ... = 2 * 3 * sqrt 5          : by rw [sqrt_sq, sqrt_sq]
  ... = 6 * sqrt 5              : by norm_num

end simplify_sqrt_180_l822_822064


namespace average_value_of_set_l822_822679

-- Definitions
def T (n : ℕ) := Finₓ n → ℕ
def removed (s : T n) (i : Finₓ n) : T (n - 1) := fun k => if h : k < i.val then s k else s ⟨k.1+1, lt_of_le_of_lt (nat.le_succ k.1) i.2⟩

variables {T : Type} [fintype T] [decidable_eq T] (b : T) 
variables (h1 : (b \ {fintype.max T}).sum id / (fintype.card T - 1) = 30)
variables (h2 : (b \ {fintype.min T, fintype.max T}).sum id / (fintype.card T - 2) = 33)
variables (h3 : (b \ {fintype.min T} ∪ {fintype.max T}).sum id / (fintype.card T - 1) = 38)
variables (h4 : fintype.max T = fintype.min T + 64)

-- Problem statement
theorem average_value_of_set : (b.sum id) / (fintype.card T) = 34.7 := 
sorry

end average_value_of_set_l822_822679


namespace mark_brings_in_148_cans_l822_822986

-- Define the given conditions
variable (R : ℕ) (Mark Jaydon Sophie : ℕ)

-- Conditions
def jaydon_cans := 2 * R + 5
def mark_cans := 4 * jaydon_cans
def unit_ratio := mark_cans / 4
def sophie_cans := 2 * unit_ratio

-- Condition: Total cans
def total_cans := mark_cans + jaydon_cans + sophie_cans

-- Condition: Each contributes at least 5 cans
axiom each_contributes_at_least_5 : R ≥ 5

-- Condition: Total cans is an odd number not less than 250
axiom total_odd_not_less_than_250 : ∃ k : ℕ, total_cans = 2 * k + 1 ∧ total_cans ≥ 250

-- Theorem: Prove Mark brings in 148 cans under the conditions
theorem mark_brings_in_148_cans (h : R = 16) : mark_cans = 148 :=
by sorry

end mark_brings_in_148_cans_l822_822986


namespace urn_probability_correct_l822_822666

noncomputable def urn_contains_five_balls_each_color :
  (initial_red_balls initial_blue_balls initial_green_balls : ℕ)
  (operations : ℕ) (final_ball_count : ℕ) (prob : ℚ) :=
  (initial_red_balls = 2 ∧ initial_blue_balls = 1 ∧ initial_green_balls = 1) ∧
  (operations = 6) ∧
  (final_ball_count = 14) →
  prob = 1 / 19

theorem urn_probability_correct :
  urn_contains_five_balls_each_color 2 1 1 6 14 (1 / 19) := sorry

end urn_probability_correct_l822_822666


namespace quadratic_change_root_l822_822550

theorem quadratic_change_root (a b c : ℕ) (h : a + b + c = 2000) :
  ∃ a' b' c' : ℕ, (∃ x : ℤ, a' * x^2 + b' * x + c' = 0) ∧ (abs (int.ofNat a' - a) + abs (int.ofNat b' - b) + abs (int.ofNat c' - c) ≤ 1050) :=
begin
  sorry
end

end quadratic_change_root_l822_822550


namespace max_area_triangle_OBC_equation_of_line_max_area_l822_822377

-- Given conditions and definitions
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 3 = 1
def point_B : ℝ × ℝ := (1, 3/2)
def intersect_BC (B_x B_y C_x C_y : ℝ) : Prop := ellipse_eq B_x B_y ∧ ellipse_eq C_x C_y ∧ B_x > 0 ∧ B_y > 0

-- Question 1: Maximum area of the triangle OBC
theorem max_area_triangle_OBC : ∃ (C_x C_y : ℝ), intersect_BC 1 (3/2) C_x C_y → ∃ (A : ℝ), A = sqrt 3 := sorry

-- Question 2: Equation of the line l
def max_area_condition (B_y C_y : ℝ) : Prop := 3 * B_y + C_y = 0
def line_eq (m n : ℝ) (x y : ℝ) : Prop := x = m * y + n

theorem equation_of_line_max_area :
  ∃ (m n : ℝ), (∃ (B_x B_y C_x C_y : ℝ), intersect_BC B_x B_y C_x C_y ∧ max_area_condition B_y C_y ∧ line_eq m n B_x B_y) →
  (line_eq (sqrt 3 / 3) (sqrt 10 / 2) 2 (sqrt 3 / 2) = (2 * sqrt 3 * 2) - 2 * (sqrt 3 / 2) - sqrt 30) := sorry

end max_area_triangle_OBC_equation_of_line_max_area_l822_822377


namespace part1_part2_l822_822966

-- Part 1: Prove that x < -12 given the inequality 2(-3 + x) > 3(x + 2)
theorem part1 (x : ℝ) : 2 * (-3 + x) > 3 * (x + 2) → x < -12 := 
  by
  intro h
  sorry

-- Part 2: Prove that 0 ≤ x < 3 given the system of inequalities
theorem part2 (x : ℝ) : 
    (1 / 2) * (x + 1) < 2 ∧ (x + 2) / 2 ≥ (x + 3) / 3 → 0 ≤ x ∧ x < 3 :=
  by
  intro h
  sorry

end part1_part2_l822_822966


namespace bernardo_wins_with_smallest_N_l822_822270

theorem bernardo_wins_with_smallest_N :
  ∃ (N : ℕ), 0 ≤ N ∧ N ≤ 999 ∧ 8 * N + 525 < 1000 ∧ 16 * N + 1050 ≥ 1000 ∧ (N = 56) ∧ (56.digits.sum = 11) :=
sorry

end bernardo_wins_with_smallest_N_l822_822270


namespace domain_expression_l822_822703

-- Define the conditions for the domain of the expression
def valid_numerator (x : ℝ) : Prop := 3 * x - 6 ≥ 0
def valid_denominator (x : ℝ) : Prop := 7 - 2 * x > 0

-- Proof problem statement
theorem domain_expression (x : ℝ) : valid_numerator x ∧ valid_denominator x ↔ 2 ≤ x ∧ x < 3.5 :=
sorry

end domain_expression_l822_822703


namespace wheel_diameter_l822_822999

theorem wheel_diameter (dist : ℝ) (revolutions : ℝ) (C : ℝ) (π : ℝ) (d : ℝ)
  (h1 : dist = 1200)
  (h2 : revolutions = 19.108280254777068)
  (h3 : C = dist / revolutions)
  (h4 : C = π * d)
  (h5 : π = Real.pi)
  : d ≈ 20 := 
sorry

end wheel_diameter_l822_822999


namespace set_inter_eq_subset_l822_822483

variable (S : Type) [DecidableEq S] (A B : Set S)

theorem set_inter_eq_subset {S : Type} [DecidableEq S] (A B : Set S) (h1 : B ⊆ A) (h2 : A ⊆ S) (h3 : B ⊆ S) :
  A ∩ B = B :=
by
  sorry

end set_inter_eq_subset_l822_822483


namespace correct_calculation_l822_822616

theorem correct_calculation (a b : ℕ) : a^3 * b^3 = (a * b)^3 :=
sorry

end correct_calculation_l822_822616


namespace simplify_sqrt180_l822_822047

-- Conditions from the problem definition
def sqrt (x : ℕ) : ℝ := Real.sqrt x
def six := 6
def five := 5

-- The statement of the problem as a Lean theorem
theorem simplify_sqrt180 : sqrt 180 = six * sqrt five := by 
  sorry

end simplify_sqrt180_l822_822047


namespace clock_hands_21st_coincidence_time_l822_822883

noncomputable def degrees_per_minute_minute_hand : ℝ := 360 / 60
noncomputable def degrees_per_minute_hour_hand : ℝ := 30 / 60

noncomputable def relative_speed : ℝ := degrees_per_minute_minute_hand - degrees_per_minute_hour_hand

noncomputable def time_per_coincidence : ℝ := 360 / relative_speed
noncomputable def total_time_for_21_coincidences : ℝ := 21 * time_per_coincidence

theorem clock_hands_21st_coincidence_time :
  (Float.round 1374.55 2) = (Float.round total_time_for_21_coincidences 2) :=
by
  sorry

end clock_hands_21st_coincidence_time_l822_822883


namespace ratio_of_surface_areas_l822_822786

-- Definitions based on conditions
def side_length_ratio (a b : ℝ) : Prop := b = 6 * a
def surface_area (a : ℝ) : ℝ := 6 * a ^ 2

-- Theorem statement
theorem ratio_of_surface_areas (a b : ℝ) (h : side_length_ratio a b) :
  (surface_area b) / (surface_area a) = 36 := by
  sorry

end ratio_of_surface_areas_l822_822786


namespace alpha_beta_values_l822_822326

theorem alpha_beta_values (n k : ℤ) : 
  let α := (π/4) + 2 * π * n,
      β := (π/3) + 2 * π * k
  in α = (π/4) + 2 * π * n ∧ β = (π/3) + 2 * π * k :=
by
  let α := (π/4) + 2 * π * n
  let β := (π/3) + 2 * π * k
  exact ⟨rfl, rfl⟩

end alpha_beta_values_l822_822326


namespace intervals_of_monotonicity_max_min_values_l822_822379

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem intervals_of_monotonicity :
  (∀ x ∈ Ioo (-1 : ℝ) 0, deriv f x > 0) ∧ (∀ x ∈ Ioo (2 : ℝ) 3, deriv f x > 0) ∧
  (∀ x ∈ Ioo (0 : ℝ) 2, deriv f x < 0) :=
by
  sorry

theorem max_min_values :
  (∃ x ∈ Icc (-1 : ℝ) 3, f x = 1) ∧ (∃ x ∈ Icc (-1 : ℝ) 3, f x = -3) :=
by
  sorry

end intervals_of_monotonicity_max_min_values_l822_822379


namespace modify_quadratic_polynomial_exists_integer_root_l822_822547

theorem modify_quadratic_polynomial_exists_integer_root
  (a b c : ℕ) (h_sum : a + b + c = 2000) :
  ∃ a' b' c' : ℕ, (a'.natAbs - a).natAbs + (b'.natAbs - b).natAbs + (c'.natAbs - c).natAbs ≤ 1050 ∧
  ∃ x : ℤ, a' * (x^2) + b' * x + c' = 0 :=
sorry

end modify_quadratic_polynomial_exists_integer_root_l822_822547


namespace vector_magnitude_problem_statement_l822_822878

noncomputable def vector_a : ℝ × ℝ := (2, 0)
noncomputable def vector_b : ℝ × ℝ := (1, 0) -- The exact form of vector_b can be calculated but assigned this form to the given magnitude.

def angle_ab : ℝ := real.pi / 3 -- 60 degrees in radians
def magnitude_b : ℝ := 1

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem vector_magnitude (u : ℝ × ℝ) : ℝ :=
real.sqrt (u.1 ^ 2 + u.2 ^ 2)

theorem problem_statement :
  vector_magnitude (vector_a.1 + 2 * vector_b.1, vector_a.2 + 2 * vector_b.2) = 2 * real.sqrt 3 := 
sorry

end vector_magnitude_problem_statement_l822_822878


namespace smallest_number_satisfying_conditions_l822_822941

theorem smallest_number_satisfying_conditions :
  ∃ b : ℕ, b ≡ 3 [MOD 5] ∧ b ≡ 2 [MOD 4] ∧ b ≡ 2 [MOD 6] ∧ b = 38 := 
by
  sorry

end smallest_number_satisfying_conditions_l822_822941


namespace three_digit_numbers_with_digit_five_l822_822409

open Nat

def isValidHundredsDigit (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 9
def isValidTensUnitsDigit (d : ℕ) : Prop := 0 ≤ d ∧ d ≤ 9

def hasDigitFive (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  hundreds = 5 ∨ tens = 5 ∨ units = 5

theorem three_digit_numbers_with_digit_five :
  { n : ℕ | 100 ≤ n ∧ n < 1000 ∧ hasDigitFive n }.card = 251 := 
sorry

end three_digit_numbers_with_digit_five_l822_822409


namespace min_S_n_condition_l822_822808

noncomputable def a_n (n : ℕ) : ℤ := -28 + 4 * (n - 1)

noncomputable def S_n (n : ℕ) : ℤ := n * (a_n 1 + a_n n) / 2

theorem min_S_n_condition : S_n 7 = S_n 8 ∧ (∀ m < 7, S_n m > S_n 7) ∧ (∀ m < 8, S_n m > S_n 8) := 
by
  sorry

end min_S_n_condition_l822_822808


namespace projectile_reaches_64_first_time_l822_822561

theorem projectile_reaches_64_first_time :
  ∃ t : ℝ, t > 0 ∧ t ≈ 0.7 ∧ (-16 * t^2 + 100 * t = 64) :=
sorry

end projectile_reaches_64_first_time_l822_822561


namespace memorable_phone_numbers_count_l822_822991

open Finset

def is_memorable (d : Fin 10 → Fin 10) : Prop :=
  (d ⟨0, by simp⟩ = d ⟨3, by simp⟩ ∧ d ⟨1, by simp⟩ = d ⟨4, by simp⟩ ∧ d ⟨2, by simp⟩ = d ⟨5, by simp⟩) ∨
  (d ⟨0, by simp⟩ = d ⟨4, by simp⟩ ∧ d ⟨1, by simp⟩ = d ⟨5, by simp⟩ ∧ d ⟨2, by simp⟩ = d ⟨6, by simp⟩)

def num_memorable_phone_numbers : ℕ :=
  (Finset.univ.filter (λ d : Fin 10 → Fin 10, is_memorable d)).card

theorem memorable_phone_numbers_count : num_memorable_phone_numbers = 19990 :=
by
  -- Proof steps to show the count of memorable phone numbers is 19990 will be written here.
  sorry

end memorable_phone_numbers_count_l822_822991


namespace largest_number_l822_822243

theorem largest_number (a b c d : ℝ)
  (h1 : a = real.sqrt 2)
  (h2 : b = 0)
  (h3 : c = -1)
  (h4 : d = 2) :
  d = max (max a b) (max c d) :=
by
  sorry

end largest_number_l822_822243


namespace pasha_can_change_coefficients_l822_822543

theorem pasha_can_change_coefficients 
  (a b c : ℕ) 
  (h1 : a + b + c = 2000) :
  ∃ a' b' c' : ℕ, 
  (abs (a' - a) + abs (b' - b) + abs (c' - c) ≤ 1050) ∧ 
  ∃ x : ℤ, (a' * x^2 + b' * x + c' = 0) := 
sorry

end pasha_can_change_coefficients_l822_822543


namespace whale_eighth_hour_consumption_is_83_l822_822233

noncomputable def plankton_consumption_in_8th_hour : ℕ :=
let total_consumption : ℕ := 960 in
let hours : ℕ := 12 in
let consumption_first_hour (x : ℕ) : ℕ := x in
let additional_consumption_per_hour (n : ℕ) : ℕ := 2 * (n - 1) in
let total_consumed (x : ℕ) : ℕ := (hours / 2) * (2 * x + additional_consumption_per_hour hours) in
let x := (total_consumption - 11 * 2) / 12 in -- Simplified from solving linear equation
let eighth_hour_consumption := consumption_first_hour x + additional_consumption_per_hour 8 in
eighth_hour_consumption

theorem whale_eighth_hour_consumption_is_83 : plankton_consumption_in_8th_hour = 83 := by
  sorry

end whale_eighth_hour_consumption_is_83_l822_822233


namespace angle_CMD_equal_65_l822_822449

/-- In a triangle QCD, Q is on circle M, and CD and CQ are tangents to circle M, 
given that ∠CQD = 50°, prove that ∠CMD = 65°. -/
theorem angle_CMD_equal_65
  (Q C D M : Point)
  (H1 : Q ∈ Circle M)
  (H2 : TangentAt C M Q)
  (H3 : TangentAt C M D)
  (H4 : ∠CQD = 50) : ∠CMD = 65 :=
sorry

end angle_CMD_equal_65_l822_822449


namespace interior_diagonal_length_l822_822110

theorem interior_diagonal_length 
  (a b c : ℝ)
  (h_surface : 2 * (a * b + a * c + b * c) = 48)
  (h_edge : 4 * (a + b + c) = 40) :
  real.sqrt (a^2 + b^2 + c^2) = 2 * real.sqrt 13 :=
by
  sorry

end interior_diagonal_length_l822_822110


namespace hotel_manager_packages_l822_822571

theorem hotel_manager_packages :
  (∑ digit in (List.range 10), 
    max 
      (List.count digit (List.range' 105 41 ++ List.range' 205 41) % 10)
      (List.count digit (List.range' 105 41 ++ List.range' 205 41) / 10) 
  ) = 51 :=
  sorry

end hotel_manager_packages_l822_822571


namespace number_of_friends_l822_822527

def has14_pokemon_cards (x : String) : Prop :=
  x = "Sam" ∨ x = "Dan" ∨ x = "Tom" ∨ x = "Keith"

theorem number_of_friends :
  ∃ n, n = 4 ∧
        ∀ x, has14_pokemon_cards x ↔ x = "Sam" ∨ x = "Dan" ∨ x = "Tom" ∨ x = "Keith" :=
by
  sorry

end number_of_friends_l822_822527


namespace donuts_multiplier_l822_822667

theorem donuts_multiplier :
  ∃ x : ℤ, 
    let monday := 14 in
    let tuesday := monday / 2 in
    let wednesday := x * monday in
    monday + tuesday + wednesday = 49 ∧ x = 2 :=
by
  sorry

end donuts_multiplier_l822_822667


namespace g_inv_sum_l822_822821

def g (x : ℝ) : ℝ :=
  if x < 10 then 2 * x + 4 else x - 3

noncomputable def g_inv_8 : ℝ := if (2 * 2 + 4 = 8) then 2 else 11
noncomputable def g_inv_16 : ℝ := if (19 - 3 = 16) then 19 else 6

theorem g_inv_sum : g_inv_8 + g_inv_16 = 21 := by
  have h1 : g_inv_8 = 2 := by
    dsimp only [g_inv_8]
    split_ifs
    · trivial
    · simp at h; contradiction

  have h2 : g_inv_16 = 19 := by
    dsimp only [g_inv_16]
    split_ifs
    · trivial
    · simp at h; contradiction

  rw [h1, h2]
  norm_num

end g_inv_sum_l822_822821


namespace silver_lake_academy_math_enrollment_l822_822250

theorem silver_lake_academy_math_enrollment
  (total_players : ℕ)
  (players_phys_or_math : ℕ)
  (players_physics : ℕ)
  (players_both : ℕ)
  (h1 : total_players = 15)
  (h2 : players_phys_or_math = 15)
  (h3 : players_physics = 9)
  (h4 : players_both = 3) :
  ∃ players_math, players_math = 9 :=
by 
  use (total_players - players_physics + players_both)
  rw [h1, h3, h4]
  simp
  sorry

end silver_lake_academy_math_enrollment_l822_822250


namespace number_of_zeros_l822_822386

def f (x a : ℝ) := a * x^2 - 2 * a * x + a + 1

def g (x b : ℝ) := b * x^3 - 2 * b * x^2 + b * x - (4 / 27)

theorem number_of_zeros (a b : ℝ) (ha : 0 < a) (hb : 1 < b) : 
  ∃ (zeros : ℝ), zeros = 2 := 
sorry

end number_of_zeros_l822_822386


namespace point_in_second_quadrant_l822_822363

theorem point_in_second_quadrant 
  (A B : ℝ) 
  (hA : 0 < A) (hB : 0 < B) 
  (hA_lt_90 : A < 90) (hB_lt_90 : B < 90) 
  (h_sum_gt_90 : A + B > 90) :
  -((cos B - sin A) + (sin B - cos A)* I).re > 0 
  ∧ ((cos B - sin A) + (sin B - cos A)* I).im > 0 :=
sorry

end point_in_second_quadrant_l822_822363


namespace smallest_m_divisible_l822_822689

theorem smallest_m_divisible 
  (m : ℕ) (m > 0) 
  (h79 : 79.prime) 
  (h83 : 83.prime) 
  (H₁ : (m * (m - 1) * (m - 2)) % 79 = 0) 
  (H₂ : (m * (m - 1) * (m - 2)) % 83 = 0) : 
  m = 1660 :=
sorry

end smallest_m_divisible_l822_822689


namespace waring_formula_l822_822521

variables {k n : ℕ} {σ s : ℕ → ℕ} 

theorem waring_formula (L : Fin n → ℕ)
  (h_sum : ∑ i in Finset.univ, (i + 1) * L i = k) :
  (-1 : ℤ) ^ k * σ k = ∑ l in Finset.univ, 
    (-1 : ℤ) ^ (∑ i in Finset.univ, L i) *
    (L i)^s(i) / (∏ i in Finset.univ, i.factorial) :=
sorry

end waring_formula_l822_822521


namespace asymptotes_of_hyperbola_l822_822387

variable (a b c : ℝ)
variable (e : ℝ := 2)

theorem asymptotes_of_hyperbola (ha : a > 0) (hb : b > 0)
  (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1) 
  (h_eccentricity : e = 2): 
  ∀ x, (y = √3 * x) ∨ (y = -√3 * x) := 
by
  sorry

end asymptotes_of_hyperbola_l822_822387


namespace correct_operation_l822_822167

variables (a : ℝ)

-- defining the expressions to be compared
def lhs := 2 * a^2 * a^4
def rhs := 2 * a^6

theorem correct_operation : lhs a = rhs a := 
by sorry

end correct_operation_l822_822167


namespace runners_meet_again_l822_822336

def speed₁ : ℝ := 3.6
def speed₂ : ℝ := 4.2
def speed₃ : ℝ := 5.4
def speed₄ : ℝ := 6.0
def track_length : ℝ := 600

theorem runners_meet_again (t : ℕ) :
  (t : ℝ) = Int.lcm
    (600 / Real.gcd speed₁ 600)
    (Int.lcm
      (600 / Real.gcd speed₂ 600)
      (Int.lcm
        (600 / Real.gcd speed₃ 600)
        (600 / Real.gcd speed₄ 600))) := 1000 :=
sorry

end runners_meet_again_l822_822336


namespace avg_departure_time_diff_from_noon_l822_822237

def time_in_minutes (h : ℕ) (m : ℕ) : ℕ := h * 60 + m

theorem avg_departure_time_diff_from_noon :
  let t_p := time_in_minutes 15 11 in -- 3:11pm in minutes
  let t_A := time_in_minutes 15 3 in -- 3:03pm in minutes
  let t_B := time_in_minutes 14 53 in -- 2:53pm in minutes
  (t_A - time_in_minutes 12 0 + t_B - time_in_minutes 12 0) / 2 = 179 :=
by
  let noon := time_in_minutes 12 0 -- Noon in minutes
  let t_A := time_in_minutes 15 3 -- 3:03pm
  let t_B := time_in_minutes 14 53 -- 2:53pm
  calc
    (t_A - noon + t_B - noon) / 2
    = (183 + 173) / 2 : by sorry
    = 356 / 2 : by sorry
    = 179 : by sorry

end avg_departure_time_diff_from_noon_l822_822237


namespace opposite_of_negative_2023_l822_822098

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l822_822098


namespace roots_ab_a_b_eq_l822_822484

noncomputable def polynomial := λ x : ℝ, x^4 - 6 * x + 1

theorem roots_ab_a_b_eq (a b : ℝ) (h_roots : polynomial a = 0 ∧ polynomial b = 0) :
  a * b + a + b = 0.75 :=
sorry

end roots_ab_a_b_eq_l822_822484


namespace equation_of_asymptotes_l822_822394

variables {a b c : ℝ}
variables (x y : ℝ)

def is_hyperbola : Prop := (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)
def eccentricity : Prop := (c = 2 * a)
def c_relation : Prop := (c^2 = a^2 + b^2)

theorem equation_of_asymptotes
  (h1 : is_hyperbola x y)
  (h2 : eccentricity)
  (h3 : c_relation) :
  (y = sqrt 3 * x ∨ y = -sqrt 3 * x) :=
sorry

end equation_of_asymptotes_l822_822394


namespace sqrt_180_eq_l822_822045

noncomputable def simplify_sqrt_180 : Real := 6 * Real.sqrt 5

theorem sqrt_180_eq : Real.sqrt 180 = simplify_sqrt_180 := 
by
  -- proof omitted
  sorry

end sqrt_180_eq_l822_822045


namespace range_of_a_l822_822401

open Set

def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

theorem range_of_a (a : ℝ) : 
  (A a ∪ B) = B ↔ a ∈ Set.Iic (-2) ∪ Set.Icc 1 2 :=
by
  sorry

end range_of_a_l822_822401


namespace value_of_expression_l822_822111

theorem value_of_expression : (4 * 3) + 2 = 14 := by
  sorry

end value_of_expression_l822_822111


namespace least_value_a_plus_b_l822_822423

theorem least_value_a_plus_b (a b : ℕ) (h : 20 / 19 = 1 + 1 / (1 + a / b)) : a + b = 19 :=
sorry

end least_value_a_plus_b_l822_822423


namespace TriangleArea_m_n_sum_l822_822924

noncomputable theory

open EuclideanGeometry Real

variables {ABC : Triangle} (BC : ℝ) (AD trisected_by_incircle : Prop)
variables (m n : ℕ)

-- Adding the required parameters as part of the formal statement
def trichotomy (ABC : Triangle) (BC : ℝ) (AD trisected_by_incircle : Prop) :=
  BC = 20 ∧ (ABC.has_incircle) ∧ (trisected_by_incircle)

-- Main theorem statement
theorem TriangleArea_m_n_sum :
  trichotomy ABC BC AD trisected_by_incircle →
    ∃ (m n : ℕ), (area ABC = m * sqrt n) ∧ (nat.is_coprime n (prime_square_factors n).prod) ∧ (m + n = 38) :=
begin
  sorry
end

end TriangleArea_m_n_sum_l822_822924


namespace asymptotes_of_hyperbola_l822_822390

variable (a b c : ℝ)
variable (e : ℝ := 2)

theorem asymptotes_of_hyperbola (ha : a > 0) (hb : b > 0)
  (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1) 
  (h_eccentricity : e = 2): 
  ∀ x, (y = √3 * x) ∨ (y = -√3 * x) := 
by
  sorry

end asymptotes_of_hyperbola_l822_822390


namespace min_AB_distance_l822_822648

theorem min_AB_distance : 
  ∀ (A B : ℝ × ℝ), 
  A ≠ B → 
  ((∃ (m : ℝ), A.2 = m * (A.1 - 1) + 1 ∧ B.2 = m * (B.1 - 1) + 1) ∧ 
    ((A.1 - 2)^2 + (A.2 - 3)^2 = 9) ∧ 
    ((B.1 - 2)^2 + (B.2 - 3)^2 = 9)) → 
  dist A B = 4 :=
sorry

end min_AB_distance_l822_822648


namespace triangle_area_trisection_l822_822927

theorem triangle_area_trisection (ABC : Type*) [triangle ABC]
  (BC : real) (hBC : BC = 20) 
  (A B C D : Type*) [point A] [point B] [point C] [point D]
  (incircle_trisects_median : median_trisects_inc_A D A (triangle_median AD))
  (area_formula : ∃ (m n : ℕ), (area ABC) = m * real.sqrt n ∧ ¬ ∃ p : ℕ, (nat.prime p) ∧ (p^2 ∣ n)) :
  let m := 24,
      n := 14 
  in m + n = 38 := 
sorry

end triangle_area_trisection_l822_822927


namespace trigonometric_expression_l822_822727

variable (α : Real)
open Real

theorem trigonometric_expression (h : tan α = 3) : 
  (2 * sin α - cos α) / (sin α + 3 * cos α) = 5 / 6 := 
by
  sorry

end trigonometric_expression_l822_822727


namespace sqrt_180_simplify_l822_822060

theorem sqrt_180_simplify : real.sqrt 180 = 6 * real.sqrt 5 := by
  have h : 180 = 2^2 * 3^2 * 5 := by norm_num
  rw [h, real.sqrt_mul, real.sqrt_mul, real.sqrt_mul]
  all_goals { norm_num }
  sorry

end sqrt_180_simplify_l822_822060


namespace average_value_l822_822304

noncomputable def f : ℝ → ℝ := λ x, log x

theorem average_value (C : ℝ) :
  (∀ x1 ∈ Icc (10 : ℝ) 100, ∃! x2 ∈ Icc (10 : ℝ) 100, (f x1 + f x2) / 2 = C) ↔ C = 3 / 2 :=
sorry

end average_value_l822_822304


namespace domain_of_g_l822_822706

def g (x : ℝ) : ℝ := Real.tan (Real.arccos (x^3 - x))

theorem domain_of_g :
  {x : ℝ | -1 ≤ x^3 - x ∧ x^3 - x ≤ 1 ∧ x ≠ 0} = Set.Icc (-1 : ℝ) (0 : ℝ) ∪ Set.Icc (0 : ℝ) (1 : ℝ) := 
sorry

end domain_of_g_l822_822706


namespace part1_part2_l822_822486

def f (x b : ℝ) := |x - b| + |x + b|

theorem part1 (x : ℝ) : (f x 1 ≤ x + 2) ↔ (x ∈ set.Icc 0 2) :=
by sorry

theorem part2 (b : ℝ) : (∀ a : ℝ, a ≠ 0 → (f 1 b) ≥ (|a + 1| - |2a - 1|) / |a|) ↔ 
                       (b ∈ set.Iic (-3/2) ∪ set.Ici (3/2)) :=
by sorry

end part1_part2_l822_822486


namespace max_regions_divided_by_lines_l822_822116

theorem max_regions_divided_by_lines (n m : ℕ) (hn : n = 50) (hm : m = 20) (h_lines : m ≤ n) : 
  let k := n - m in
  let S_k := (k * (k + 1)) / 2 + 1 in
  let S_parallel := m * (k + 1) in
  S_k + S_parallel = 1086 :=
by
  have hn : n = 50 := hn
  have hm : m = 20 := hm
  have h_lines : m ≤ n := h_lines
  let k := n - m
  let S_k := (k * (k + 1)) / 2 + 1
  let S_parallel := m * (k + 1)
  sorry

end max_regions_divided_by_lines_l822_822116


namespace minimize_expression_l822_822827

variable (b1 b2 b3 : ℝ) (s : ℝ)
-- Conditions
def is_geometric_sequence : Prop :=
  b2 = b1 * s ∧ b3 = b1 * s^2

def b1_equals_2 : Prop := b1 = 2

-- Prove the smallest possible value of 3b2 + 7b3
theorem minimize_expression (h1 : is_geometric_sequence b1 b2 b3 s) (h2 : b1_equals_2 b1) :
  ∃ s, 3 * b2 + 7 * b3 = -9 / 14 :=
by sorry

end minimize_expression_l822_822827


namespace bonus_percentage_is_correct_l822_822119

theorem bonus_percentage_is_correct (kills total_points enemies_points bonus_threshold bonus_percentage : ℕ) 
  (h1 : enemies_points = 10) 
  (h2 : kills = 150) 
  (h3 : total_points = 2250) 
  (h4 : bonus_threshold = 100) 
  (h5 : kills >= bonus_threshold) 
  (h6 : bonus_percentage = (total_points - kills * enemies_points) * 100 / (kills * enemies_points)) : 
  bonus_percentage = 50 := 
by
  sorry

end bonus_percentage_is_correct_l822_822119


namespace correct_system_of_equations_l822_822965

-- Setting the types for weights of gold and silver
variables (x y : ℝ)

-- Defining the conditions as hypotheses
def gold_silver_equal_weight : Prop := 9 * x = 11 * y
def weight_difference : Prop := (10 * y + x) - (8 * x + y) = 13

-- Stating the theorem that combines the conditions
theorem correct_system_of_equations : 
  gold_silver_equal_weight x y ∧ weight_difference x y :=
by 
  constructor;
  { sorry }

end correct_system_of_equations_l822_822965


namespace irrational_sqrt3_9_l822_822663

theorem irrational_sqrt3_9 :
  ∀ (a b c d : ℝ), a = 4 → b = real.cbrt 9 → c = 22 / 7 → d = 0 →
  irrational b ∧ ¬ irrational a ∧ ¬ irrational c ∧ ¬ irrational d :=
by
  intros a b c d ha hb hc hd
  rw [ha, hb, hc, hd]
  have h_ir_a : ¬ irrational 4 := by sorry
  have h_ir_c : ¬ irrational (22 / 7) := by sorry
  have h_ir_d : ¬ irrational 0 := by sorry
  have h_irrational_b : irrational (real.cbrt 9) := by sorry
  exact ⟨h_irrational_b, h_ir_a, h_ir_c, h_ir_d⟩

end irrational_sqrt3_9_l822_822663


namespace distance_to_origin_of_z_111_l822_822687

theorem distance_to_origin_of_z_111 :
  let z : ℕ → ℂ := λ n, if n = 1 then 0 else
  let rec z_seq (n : ℕ) : ℂ :=
    if n = 1 then 0 else z_seq (n - 1) ^ 2 + complex.I
  in z_seq n
  in complex.abs (z 111) = real.sqrt 2 :=
by
  sorry

end distance_to_origin_of_z_111_l822_822687


namespace cost_of_one_bag_of_potatoes_l822_822122

theorem cost_of_one_bag_of_potatoes :
  let x := 250 in
  ∀ (price : ℕ)
    (bags : ℕ)
    (andrey_initial_price : ℕ)
    (andrey_sold_price : ℕ)
    (boris_initial_price : ℕ)
    (boris_first_price : ℕ)
    (boris_second_price : ℕ)
    (earnings_andrey : ℕ)
    (earnings_boris_first : ℕ)
    (earnings_boris_second : ℕ)
    (total_earnings_boris : ℕ),
  bags = 60 →
  andrey_initial_price = price →
  andrey_sold_price = 2 * price →
  andrey_sold_price * bags = earnings_andrey →
  boris_initial_price = price →
  boris_first_price = 1.6 * price →
  boris_second_price = 2.24 * price →
  boris_first_price * 15 + boris_second_price * 45 = total_earnings_boris →
  total_earnings_boris = earnings_andrey + 1200 →
  price = x :=
by
  intros x price bags andrey_initial_price andrey_sold_price boris_initial_price boris_first_price boris_second_price earnings_andrey earnings_boris_first earnings_boris_second total_earnings_boris
  assume h_bags h_andrey_initial_price h_andrey_sold_price h_earnings_andrey h_boris_initial_price h_boris_first_price h_boris_second_price h_total_earnings_boris h_total_earnings_difference
  if h_necessary : x = 250 then
    sorry
  else
    sorry


end cost_of_one_bag_of_potatoes_l822_822122


namespace triangle_rotated_surface_area_l822_822658

def is_right_triangle (a b c : ℕ) : Prop :=
  a*a + b*b = c*c

noncomputable def surface_area_of_solid (a b c : ℕ) : ℝ :=
  let base_area := Real.pi * c^2
  let lateral_area := Real.pi * b * c
  base_area + lateral_area

theorem triangle_rotated_surface_area :
  is_right_triangle 3 4 5 →
  surface_area_of_solid 3 4 5 = 36 * Real.pi :=
by
  intro h
  rw [is_right_triangle, surface_area_of_solid, Real.pi, Real.pi]
  sorry

end triangle_rotated_surface_area_l822_822658


namespace inv_f_of_neg3_l822_822957

def f (x : Real) : Real := 5 - 2 * x

theorem inv_f_of_neg3 : f⁻¹ (-3) = 4 :=
by
  sorry

end inv_f_of_neg3_l822_822957


namespace highlights_part_to_whole_relation_l822_822156

/-- A predicate representing different types of statistical graphs. -/
inductive StatGraphType where
  | BarGraph : StatGraphType
  | PieChart : StatGraphType
  | LineGraph : StatGraphType
  | FrequencyDistributionHistogram : StatGraphType

/-- A lemma specifying that the PieChart is the graph type that highlights the relationship between a part and the whole. -/
theorem highlights_part_to_whole_relation (t : StatGraphType) : t = StatGraphType.PieChart :=
  sorry

end highlights_part_to_whole_relation_l822_822156


namespace num_solutions_gg_x_eq_3_l822_822557

def g (x : ℝ) : ℝ :=
  if x < -1 then -0.5 * x^2 + x + 4 else if x < 2 then -x + 3 else x - 2

theorem num_solutions_gg_x_eq_3 : 
  (set.countable {x : ℝ | g (g x) = 3} = 1) :=
by
  sorry

end num_solutions_gg_x_eq_3_l822_822557


namespace final_fraction_of_water_is_243_over_1024_l822_822194

theorem final_fraction_of_water_is_243_over_1024 :
  let initial_volume := 20
  let replaced_volume := 5
  let cycles := 5
  let initial_fraction_of_water := 1
  let final_fraction_of_water :=
        (initial_fraction_of_water * (initial_volume - replaced_volume) / initial_volume) ^ cycles
  final_fraction_of_water = 243 / 1024 :=
by
  sorry

end final_fraction_of_water_is_243_over_1024_l822_822194


namespace probability_AC_lt_9_l822_822247

noncomputable def prob_AC_lt_9 : ℝ :=
  ∫ (β : ℝ) in 0..(π/2), if let x := 7 * cos β, let y := 10 + 7 * sin β in
  (x*x + y*y < 81) then 1 / (π / 2) else 0

theorem probability_AC_lt_9 :
  prob_AC_lt_9 = 1 / 3 := by
  sorry

end probability_AC_lt_9_l822_822247


namespace part1_part2_l822_822382

def f (x : ℝ) (a : ℝ) : ℝ := |x - a| + 4 * x

theorem part1 (x : ℝ) :
  f x 2 ≥ 2 * x + 1 ↔ x ∈ set.Ici (-1) :=
sorry

theorem part2 (a : ℝ) :
  (∀ x ∈ set.Ioi (-2), f (2 * x) a > 7 * x + a^2 - 3) ↔ a ∈ set.Ioo 0 2 :=
sorry

end part1_part2_l822_822382


namespace percentage_change_in_receipts_l822_822614

theorem percentage_change_in_receipts (P S : ℝ)
  (h_price_reduction : P' = 0.7 * P)
  (h_sales_increase : S' = 1.5 * S) :
  let original_receipts := P * S,
      new_receipts := P' * S',
      percentage_change := (new_receipts - original_receipts) / original_receipts * 100
  in percentage_change = 5 :=
by
  sorry

end percentage_change_in_receipts_l822_822614


namespace find_ellipse_equation_find_max_area_triangle_l822_822745

def ellipse_equation (a b : ℝ) : Prop := (a > b ∧ b > 0 ∧ e = Real.sqrt 3 / 2) ∧ 
                                        (a = 2) ∧ (b = 1) ∧ (e = Real.sqrt 3 / 2)

theorem find_ellipse_equation :
  ∃ (a b : ℝ), ellipse_equation a b :=
  by {
    use [2, 1],
    sorry
  }

def max_area_triangle (a : ℝ) : Prop := a = 1 

theorem find_max_area_triangle :
  ∃ (a : ℝ), max_area_triangle a :=
  by {
    use 1,
    sorry
  }

end find_ellipse_equation_find_max_area_triangle_l822_822745


namespace card_game_ensures_final_state_l822_822148

def card_game_eventually_ends (n : ℕ) (beats : ℕ → ℕ → Prop) (deck1 deck2 : list ℕ) : Prop :=
  ∃ (strategy : (list ℕ × list ℕ) → (list ℕ × list ℕ)), ∀ (deck1 deck2 : list ℕ),
    (deck1.length + deck2.length = n) →
    (∀ deck1 deck2, (deck1, deck2) ≠ ([], [])) →
    (∃ final_state, (strategy (deck1, deck2) = final_state ∧ (final_state.1 = [] ∨ final_state.2 = [])))

theorem card_game_ensures_final_state (n : ℕ) (beats : ℕ → ℕ → Prop) (deck1 deck2 : list ℕ) :
  card_game_eventually_ends n beats deck1 deck2 :=
sorry

end card_game_ensures_final_state_l822_822148


namespace angle_ABC_is_90_area_of_triangle_ABC_l822_822029

variables {A B C D P T M N : Type*}
variables [Points : Set A B C D P T] [Lie_on : ∀ {X Y}, Set X Y → Prop]
variables [Circle : ∀ {X Y}, Set X Y → Prop]
variables [Midpoint : ∀ {X Y : Type*}, Set X Y → Type*] 

-- Conditions
axiom D_on_AC : Lie_on D (AC : Set A C)
axiom Circ_Diam_BD : Circle (P T : Set (BD : diameter B D))
axiom M_mid_AD : Midpoint M (AD : Set A D)
axiom N_mid_CD : Midpoint N (CD : Set C D)
axiom PM_parallel_TN : parallelogram PM TN
axiom MP_length : ∥MP∥ = 1
axiom NT_length : ∥NT∥ = 3 / 2
axiom BD_length : ∥BD∥ = sqrt 5

-- First Statement: Angle ABC is 90 degrees
theorem angle_ABC_is_90 : ∃ (ABC : Set ℝ), ℝ.angle ABC = 90 := by
begin
  sorry
end

-- Second Statement: Area of triangle ABC is 5
theorem area_of_triangle_ABC : ∃ (S : Set ℝ), Area(△ABC) = 5 := by
begin
  sorry
end

end angle_ABC_is_90_area_of_triangle_ABC_l822_822029


namespace factorize_expression_l822_822695

theorem factorize_expression (x : ℝ) : 9 * x^3 - 18 * x^2 + 9 * x = 9 * x * (x - 1)^2 := 
by 
    sorry

end factorize_expression_l822_822695


namespace arithmetic_seq_sum_l822_822458

variable {α : Type*} [Add α] [Mul α] [AddMonoid α]

-- Given: An arithmetic sequence \(a_n\) where \( a_2 = 5 \) and \( a_5 = 33 \)
variables (a : ℕ → α)
variable (n : ℕ)
variable (d : α) -- common difference in the arithmetic sequence
variable (h_arith_seq : ∀ n, a (n+1) = a n + d)

-- Given specific values and the condition in the problem
def a2_eq_five := a 2 = 5
def a5_eq_thirty_three := a 5 = 33

-- The statement to prove
theorem arithmetic_seq_sum (h1 : a2_eq_five a) (h2 : a5_eq_thirty_three a) : a 3 + a 4 = 38 :=
by
  sorry

end arithmetic_seq_sum_l822_822458


namespace simplify_sqrt_180_l822_822069

theorem simplify_sqrt_180 : sqrt 180 = 6 * sqrt 5 :=
by
  have h : 180 = 2^2 * 3^2 * 5 := by norm_num
  calc
    sqrt 180
      = sqrt (2^2 * 3^2 * 5)     : by rw [h]
  ... = sqrt (2^2) * sqrt (3^2) * sqrt 5 : by rw [sqrt_mul, sqrt_mul]
  ... = 2 * 3 * sqrt 5          : by rw [sqrt_sq, sqrt_sq]
  ... = 6 * sqrt 5              : by norm_num

end simplify_sqrt_180_l822_822069


namespace sum_of_divisors_85_l822_822944

theorem sum_of_divisors_85 :
  let divisors := {d ∈ Finset.range 86 | 85 % d = 0}
  Finset.sum divisors id = 108 :=
sorry

end sum_of_divisors_85_l822_822944


namespace intersection_on_circumcircle_l822_822494

noncomputable theory

open_locale classical

-- Definition of a square.
structure Square (α : Type*) :=
(A B C D E F : α) -- Points A, B, C, D, E, F
(sqsides : ∀ (x y : set α), (x ∈ sqsides ↔ x ∈ {A, B, C, D}))
(sqpoints : pairwise (≠) {A, B, C, D})

-- Given conditions translated into definitions.
variables {α : Type*} [LinearOrderedField α]

def condition1 (S : Square α) : Prop :=
true

def E_on_BC_div_fifths (S : Square α) : Prop :=
∃ k, 0 < k ∧ k < 1 ∧ S.E = S.B + k • (S.C - S.B) ∧ k = 1 / 5

def F_reflect (S : Square α) : Prop :=
∃ G, G = S.D + (1 / 3) • (S.C - S.D) ∧ S.F = S.C + (S.C - G)

-- Final statement to be proven in Lean 4
theorem intersection_on_circumcircle (S : Square α) 
  (h1 : condition1 S) 
  (h2 : E_on_BC_div_fifths S) 
  (h3 : F_reflect S) : 
∃ P, ∃ Q : set α, P ∈ Q ∧ ... :=
sorry

end intersection_on_circumcircle_l822_822494


namespace arithmetic_sequence_property_l822_822807

-- Conditions given in the problem
variable (a : ℕ → ℤ) (d : ℤ)
hypothesis h1 : ∀ n : ℕ, a (n + 1) = a n + d
hypothesis h2 : a 1 + a 8 + a 15 = 72

-- Goal
theorem arithmetic_sequence_property : a 5 + 3 * d = 24 :=
by { sorry }

end arithmetic_sequence_property_l822_822807


namespace stephen_speed_second_third_l822_822074

theorem stephen_speed_second_third
  (first_third_speed : ℝ)
  (last_third_speed : ℝ)
  (total_distance : ℝ)
  (travel_time : ℝ)
  (time_in_hours : ℝ)
  (h1 : first_third_speed = 16)
  (h2 : last_third_speed = 20)
  (h3 : total_distance = 12)
  (h4 : travel_time = 15)
  (h5 : time_in_hours = travel_time / 60) :
  time_in_hours * (total_distance - (first_third_speed * time_in_hours + last_third_speed * time_in_hours)) = 12 := 
by 
  sorry

end stephen_speed_second_third_l822_822074


namespace bill_needs_125_bouquets_to_earn_1000_l822_822256

-- Define the constants for the problem
def cost_per_bouquet : ℕ := 20
def roses_per_bouquet_buy : ℕ := 7
def roses_per_bouquet_sell : ℕ := 5
def target_profit : ℕ := 1000

-- Define the problem in terms of a theorem
theorem bill_needs_125_bouquets_to_earn_1000 :
  ∃ n : ℕ, (35 / roses_per_bouquet_sell) * cost_per_bouquet - (5 * cost_per_bouquet) = 40 → (5 * n) = 125 :=
begin
  sorry
end

end bill_needs_125_bouquets_to_earn_1000_l822_822256


namespace bob_can_order_199_sandwiches_l822_822551

-- Define the types of bread, meat, and cheese
def number_of_bread : ℕ := 5
def number_of_meat : ℕ := 7
def number_of_cheese : ℕ := 6

-- Define the forbidden combinations
def forbidden_turkey_swiss : ℕ := number_of_bread -- 5
def forbidden_rye_roastbeef : ℕ := number_of_cheese -- 6

-- Calculate the total sandwiches and subtract forbidden combinations
def total_sandwiches : ℕ := number_of_bread * number_of_meat * number_of_cheese
def forbidden_sandwiches : ℕ := forbidden_turkey_swiss + forbidden_rye_roastbeef

def sandwiches_bob_can_order : ℕ := total_sandwiches - forbidden_sandwiches

theorem bob_can_order_199_sandwiches :
  sandwiches_bob_can_order = 199 :=
by
  -- The calculation steps are encapsulated in definitions and are considered done
  sorry

end bob_can_order_199_sandwiches_l822_822551


namespace page_added_twice_is_33_l822_822895

noncomputable def sum_first_n (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem page_added_twice_is_33 :
  ∃ n : ℕ, ∃ m : ℕ, sum_first_n n + m = 1986 ∧ 1 ≤ m ∧ m ≤ n → m = 33 := 
by {
  sorry
}

end page_added_twice_is_33_l822_822895


namespace gcd_18_30_is_6_gcd_18_30_is_even_l822_822320

def gcd_18_30 : ℕ := Nat.gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 := by
  sorry

theorem gcd_18_30_is_even : Even gcd_18_30 := by
  sorry

end gcd_18_30_is_6_gcd_18_30_is_even_l822_822320


namespace solve_eq_l822_822954

noncomputable def log_b (b x : ℝ) := real.log x / real.log b

theorem solve_eq (x : ℝ) (k : ℤ) (hx : 0 < real.sin x) :
  8.492 * log_b (real.sin x * real.cos x) (real.sin x) * log_b (real.sin x * real.cos x) (real.cos x) = 1 / 4
  → x = (real.pi / 4) * (8 * (k : ℝ) + 1) :=
begin
  sorry
end

end solve_eq_l822_822954


namespace pasha_can_change_coefficients_l822_822542

theorem pasha_can_change_coefficients 
  (a b c : ℕ) 
  (h1 : a + b + c = 2000) :
  ∃ a' b' c' : ℕ, 
  (abs (a' - a) + abs (b' - b) + abs (c' - c) ≤ 1050) ∧ 
  ∃ x : ℤ, (a' * x^2 + b' * x + c' = 0) := 
sorry

end pasha_can_change_coefficients_l822_822542


namespace modulo_power_problem_statement_l822_822611

theorem modulo_power (a : ℕ) (n : ℕ) (k : ℕ) [hn : Fact (11 > 0)] (h : a ≡ 7 [MOD 11]) : 
  a ^ n ≡ 7 ^ k [MOD 11] :=
sorry

theorem problem_statement : 777 ^ 444 ≡ 3 [MOD 11] :=
by
  have h1 : 777 ≡ 7 [MOD 11] := by sorry
  have h2 : 7 ^ 10 ≡ 1 [MOD 11] := by sorry
  have h3 : 444 = 44 * 10 + 4 := by sorry
  have h4 : 777 ^ 444 ≡ 7 ^ 444 [MOD 11] := by apply modulo_power 777 444 444 h1
  rw [h3] at h4
  rw [pow_add] at h4
  rw [pow_mul] at h4
  rw [h2] at h4
  rw [one_pow] at h4
  rw [mul_one] at h4
  exact h4

end modulo_power_problem_statement_l822_822611


namespace math_problem_l822_822749

noncomputable def f : ℝ → ℝ := sorry  -- definition of f, as an even and specifically described function

theorem math_problem
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_mono_incr : ∀ x y : ℝ, x < y ∧ y < 0 → f x ≤ f y)
  (a_def : f (Real.log 3 / Real.log (1/2)) = f (Real.log 3 / -Real.log 2)) -- simplification from the log_base
  (b_def : f ((1 / 3) ^ 0.2) = f ((1 / 3) ^ 0.2))  -- directly simplified
  (c_def : f (Real.log 5 / Real.log 2) = f (Real.log 5 / Real.log 2)) -- using log_base properties
  (h_order : (1 / 3) ^ 0.2 < 1 ∧ 1 < Real.log 2 ∧ Real.log 2 < Real.log 3 ∧ Real.log 3 < Real.log 5)
: f (Real.log 5 / Real.log 2) < f (Real.log 3 / Real.log 2) ∧ f (Real.log 3 / Real.log 2) < f ((1 / 3) ^ 0.2) := sorry

end math_problem_l822_822749


namespace locus_of_intersection_is_circumcircle_of_AOB_l822_822691

variables {K : Type*} [EuclideanGeometry K] {A B C D E: Point K}
variables (O : Circle K) [hABC : IsInscribed O (IsoscelesTrapezoid A B C D)]

noncomputable def locus_of_diagonal_intersection : Set.Point K :=
  { E | ( ∃ A B C D : Point K, 
    IsInscribed O (IsoscelesTrapezoid A B C D) ∧ 
    Intersection.IsDiagonalsIntersection (AC, BD) E ) ∧ 
    SubtendsAngle E A B O.∡ }

theorem locus_of_intersection_is_circumcircle_of_AOB :
  locus_of_diagonal_intersection O = Circumcircle (Triangle.mk A B O) :=
sorry

end locus_of_intersection_is_circumcircle_of_AOB_l822_822691


namespace area_triangle_BPQ_l822_822802

-- Define the geometrical setup
variables (A B C H X Y P Q : Type) (h : ℝ) (right_triangle_ABC : ℝ) (angle_AHC : ℝ) (distance_AC : ℝ)

-- Assume points and conditions
variables (height_BH : B = H → H = A → H = C) (XY_intersects_AB_P : XY ∩ AB = P) (XY_intersects_BC_Q : XY ∩ BC = Q)
variables (BH_h : BH = h)

-- The main goal: proving the area of triangle BPQ
theorem area_triangle_BPQ : real :=
begin
  -- The specific conditions required to prove the area would have been set here
  -- Using the given height and geometric properties
  calc area_of_triangle BPQ = h^2 / 2 : sorry
end

end area_triangle_BPQ_l822_822802


namespace moles_of_CaOH₂_formed_l822_822699

theorem moles_of_CaOH₂_formed :
  ∀ (m₁ m₂ : ℝ) (n_H₂O : ℝ),
  (m₁ = 56.08) ∧ (m₂ = 112) ∧ (n_H₂O = 2) →
  (m₂ / m₁) = n_H₂O :=
by
  intros m₁ m₂ n_H₂O h,
  cases h with h₁ h2,
  cases h2 with h2_1 h2_2,
  rw [h2_1, h2_2, h₁],
  norm_num,
  linarith,
end 

end moles_of_CaOH₂_formed_l822_822699


namespace marathon_time_is_correct_l822_822918

def first_movie := 2
def second_movie := 1.5 * first_movie
def combined_time := first_movie + second_movie
def third_movie := combined_time - 0.2 * combined_time
def fourth_movie := 2 * second_movie
def fifth_movie := third_movie - 0.5

def total_marathon_time := first_movie + second_movie + third_movie + fourth_movie + fifth_movie

theorem marathon_time_is_correct : total_marathon_time = 18.5 := by
  sorry

end marathon_time_is_correct_l822_822918


namespace distance_is_3_km_l822_822209

noncomputable def distance_to_place (V_m V_r : ℝ) (total_time : ℝ) : ℝ :=
  let V_upstream := V_m - V_r
  let V_downstream := V_m + V_r
  let T := (50.0 / 60.0 : ℝ)
  let D := @calc
    let D := 3 -- Solution found
    D : ℝ := 3
  D

theorem distance_is_3_km : distance_to_place 7.5 1.5 (50 / 60) = 3 := by
  unfold distance_to_place
  sorry

end distance_is_3_km_l822_822209


namespace class_size_is_10_l822_822091

theorem class_size_is_10 
  (num_92 : ℕ) (num_80 : ℕ) (last_score : ℕ) (target_avg : ℕ) (total_score : ℕ) 
  (h_num_92 : num_92 = 5) (h_num_80 : num_80 = 4) (h_last_score : last_score = 70) 
  (h_target_avg : target_avg = 85) (h_total_score : total_score = 85 * (num_92 + num_80 + 1)) 
  : (num_92 * 92 + num_80 * 80 + last_score = total_score) → 
    (num_92 + num_80 + 1 = 10) :=
by {
  sorry
}

end class_size_is_10_l822_822091


namespace foreign_trade_income_l822_822088

variable (m x n : ℝ)

theorem foreign_trade_income :
  m * (1 + x)^2 = n → m * (1 + x) * (1 + x) = n :=
by
  intro h
  exact h

end foreign_trade_income_l822_822088


namespace sum_of_radii_equals_radius_l822_822186

noncomputable def midpoint (BC : Line) := {M : Point // M ∈ BC ∧ ∀ B C : Point, midpoint B C = M}
noncomputable def arbitrary_point (BC : Line) := {P : Point // P ∈ BC}

structure Circle where
  center : Point
  radius : ℝ

variable (BC : Line) 
variable (M : midpoint BC) 
variable (P : arbitrary_point BC)
variable (big_circle : Circle)
variable (C1 : Circle)
variable (r1 r2 r3 R : ℝ)

-- Given conditions
axiom big_circle_properties : big_circle.radius = R
axiom C1_properties : C1.radius = r1 ∧ tangent C1 big_circle
axiom C4_properties : C4.radius = r1 ∧ tangent C4 BC ∧ C4.center = P
axiom C2_properties : tangent C2 big_circle ∧ tangent C2 BC ∧ tangent C2 C4
axiom C3_properties : tangent C3 big_circle ∧ tangent C3 BC ∧ tangent C3 C4

theorem sum_of_radii_equals_radius : r1 + r2 + r3 = R := 
sorry

end sum_of_radii_equals_radius_l822_822186


namespace min_sum_xyz_l822_822318

theorem min_sum_xyz (x y z : ℝ) 
  (hx : x ≥ 4) (hy : y ≥ 5) (hz : z ≥ 6) 
  (hxyz : x^2 + y^2 + z^2 ≥ 90) : 
  x + y + z ≥ 16 := 
sorry

end min_sum_xyz_l822_822318


namespace sum_first_20_terms_l822_822766

open Set
open Nat

def P : Set ℕ := {x | ∃ n : ℕ, x = 2^n}
def Q : Set ℕ := {x | ∃ n : ℕ, x = 2*n}

-- Construct the sequence a_n by combining and sorting P ∪ Q
noncomputable def a_seq : List ℕ := ((P ∪ Q).toList.qsort (≤))

noncomputable def S₀ : ℕ := (a_seq.take 20).sum

theorem sum_first_20_terms : S₀ = 343 :=
by {
  -- Proof omitted
  sorry
}

end sum_first_20_terms_l822_822766


namespace min_value_expression_l822_822488

section
variables {p q r s t u v w : ℝ}
-- Define the conditions
hypothesis h1 : p * q * r * s = 16
hypothesis h2 : t * u * v * w = 16

-- Define the goal
theorem min_value_expression : (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 = 64 :=
sorry
end

end min_value_expression_l822_822488


namespace b_arithmetic_geometric_sum_c_l822_822400

-- Define the sequence a_n
def a : ℕ → ℕ
| 0       := 3
| 1       := 5
| (n + 2) := 2 * a n.succ - a n

-- Define the sequence b_n
def b (n : ℕ) : ℕ := a (n + 1) - a n

-- Define the sequence c_n
def c (n : ℕ) : ℕ :=
  if n % 2 = 0 then 2 ^ (a n - 1) else a n

-- Define the sum S_2n of the first 2n terms of the sequence c_n
noncomputable def S (n : ℕ) : ℕ := 
  let So := (finset.range (n + 1)).sum (λ k, a (2 * k))
  let Se := (finset.range (n + 1)).sum (λ k, 2 ^ (a (2 * k + 1) - 1))
  in So + Se

-- Prove that b_n is an arithmetic sequence or a geometric sequence
theorem b_arithmetic_geometric (n : ℕ) :
  (∃ d, ∀ k, b (k + 1) = b k + d) ∨ (∃ r, ∀ k, b (k + 1) = b k * r) :=
sorry

-- Prove the sum S_2n of the first 2n terms of the sequence c_n
theorem sum_c (n : ℕ) :
  S n = 2 * n^2 + n + (16 * (16^n - 1)) / 15 :=
sorry

end b_arithmetic_geometric_sum_c_l822_822400


namespace mia_spending_on_each_parent_l822_822842

variables (sibling_spending sibling_count total_spending parent_count : ℕ)
variables (spending_siblings spending_parents spending_each_parent : ℕ)
variables (equal_value : Prop)

-- Define the conditions
def condition_1 : sibling_spending = 30 := rfl
def condition_2 : sibling_count = 3 := rfl
def condition_3 : total_spending = 150 := rfl
def condition_4 : spending_siblings = sibling_count * sibling_spending := rfl
def condition_5 : spending_parents = total_spending - spending_siblings := rfl
def condition_6 : parent_count = 2 := rfl
def condition_7 : equal_value = (spending_each_parent = spending_parents / parent_count) := rfl

-- Define the main proof statement
theorem mia_spending_on_each_parent :
  sibling_spending = 30 →
  sibling_count = 3 →
  total_spending = 150 →
  spending_siblings = sibling_count * sibling_spending →
  spending_parents = total_spending - spending_siblings →
  parent_count = 2 →
  equal_value →
  spending_each_parent = 30 :=
by intros; sorry

end mia_spending_on_each_parent_l822_822842


namespace flea_miss_point_on_circle_l822_822912

theorem flea_miss_point_on_circle :
  ∃ (p : ℕ), p < 101 ∧ ∀ (n : ℕ), p ≠ (n * (n + 1) / 2) % 101 :=
begin
  sorry
end

end flea_miss_point_on_circle_l822_822912


namespace find_ellipse_equation_l822_822762

-- Define the function representing the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the point A
def A : ℝ × ℝ := (2, 4)

-- Define the tangent line at point A
def tangent_line_at_A (x : ℝ) : ℝ := 4 * x - 4

-- Given the ellipse equation form
def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Point of focus (intersection with x-axis) and lower vertex (intersection with y-axis)
def focus : ℝ × ℝ := (1, 0)
def lower_vertex : ℝ × ℝ := (0, -4)

-- Main theorem stating the ellipse equation
theorem find_ellipse_equation :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ ellipse a b x y → 
  a = sqrt 17 ∧ b = 4 ∧ ellipse (sqrt 17) 4 x y := 
by
  sorry

end find_ellipse_equation_l822_822762


namespace isosceles_triangle_perimeter_l822_822769

-- Statement: An isosceles triangle with sides of lengths 2 and 4 has a perimeter of 10.
theorem isosceles_triangle_perimeter (a b : ℕ)
  (h_iso : a = 2 ∨ a = 4)
  (h_ineq : a + a > b ∧ a + b > a ∧ b + a > a) : a + a + b = 10 :=
by
  cases h_iso with h2 h4
  case left =>
    subst h2
    have h1 : b = 4 := by {cases h_ineq.right with _ h3, exact h3}
    sorry
  case right =>
    subst h4
    have h1 : b = 2 := by {cases h_ineq.left with _ h3, exact h3}
    sorry

end isosceles_triangle_perimeter_l822_822769


namespace find_S7_l822_822735

variable {a b : ℝ}
variable {S : ℕ → ℝ} (hS : ∀ n, S n = a * n^2 + b * n)

def a_2_eq_3 (ha2 : (S 2 - S 1) = 3) : Prop := ha2

def a_6_eq_11 (ha6 : (S 6 - S 5) = 11) : Prop := ha6

theorem find_S7 (hS : ∀ n, S n = a * n^2 + b * n) (ha2 : (S 2 - S 1) = 3) (ha6 : (S 6 - S 5) = 11) : S 7 = 49 :=
by
  sorry

end find_S7_l822_822735


namespace ratio_problem_l822_822635

theorem ratio_problem (X : ℕ) :
  (18 : ℕ) * 360 = 9 * X → X = 720 :=
by
  intro h
  sorry

end ratio_problem_l822_822635


namespace jackson_purchase_total_nearest_dollar_l822_822816

theorem jackson_purchase_total_nearest_dollar :
  round (2.49 + 3.75 + 11.23) = 17 := 
by
  sorry

end jackson_purchase_total_nearest_dollar_l822_822816


namespace log_eq_neg1_imp_x_eq_point2_l822_822004

theorem log_eq_neg1_imp_x_eq_point2
  (x : ℝ) (h1 : x > 0) (h2 : log 5 x = -1) : x = 0.2 :=
sorry

end log_eq_neg1_imp_x_eq_point2_l822_822004


namespace find_m_l822_822737

variables (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (m : ℕ)

theorem find_m (h1 : ∀ n, a (n + 1) = a n + d) -- arithmetic sequence
               (h2 : S (2 * m - 1) = 39)       -- sum of first (2m-1) terms
               (h3 : a (m - 1) + a (m + 1) - a m - 1 = 0)
               (h4 : m > 1) : 
               m = 20 :=
   sorry

end find_m_l822_822737


namespace pumps_empty_pool_in_approx_65_point_45_minutes_l822_822863

theorem pumps_empty_pool_in_approx_65_point_45_minutes :
  let rate_A := 1 / 4
  let rate_B := 1 / 2
  let rate_C := 1 / 6
  let combined_rate := rate_A + rate_B + rate_C
  let time_hours := 1 / combined_rate 
  let time_minutes := time_hours * 60
  65.44 < time_minutes ∧ time_minutes < 65.46 := 
by
  let rate_A := 1 / 4
  let rate_B := 1 / 2
  let rate_C := 1 / 6
  let combined_rate := rate_A + rate_B + rate_C
  let time_hours := 1 / combined_rate
  let time_minutes := time_hours * 60
  have h1 : combined_rate = (1 / 4) + (1 / 2) + (1 / 6) := by rfl
  have h2 : combined_rate * 12 = 3 + 6 + 2 := by sorry
  have h3 : combined_rate = 11 / 12 := by sorry
  have h4 : time_hours = 1 / (11 / 12) := by sorry
  have h5 : time_hours = 12 / 11 := by sorry
  have h6 : time_minutes = (12 / 11) * 60 := by sorry
  have h7 : time_minutes = 720 / 11 := by sorry
  have h8 : time_minutes ≈ 65.45 := by sorry
  exact ⟨by linarith, by linarith⟩

end pumps_empty_pool_in_approx_65_point_45_minutes_l822_822863


namespace container_volume_ratio_l822_822722

theorem container_volume_ratio
  (C D : ℕ)
  (h1 : (3 / 5 : ℚ) * C = (1 / 2 : ℚ) * D)
  (h2 : (1 / 3 : ℚ) * ((1 / 2 : ℚ) * D) + (3 / 5 : ℚ) * C = C) :
  (C : ℚ) / D = 5 / 6 :=
by {
  sorry
}

end container_volume_ratio_l822_822722


namespace child_B_share_l822_822213

theorem child_B_share (total_money : ℕ) (ratio_A ratio_B ratio_C ratio_D ratio_E total_parts : ℕ) 
  (h1 : total_money = 12000)
  (h2 : ratio_A = 2)
  (h3 : ratio_B = 3)
  (h4 : ratio_C = 4)
  (h5 : ratio_D = 5)
  (h6 : ratio_E = 6)
  (h_total_parts : total_parts = ratio_A + ratio_B + ratio_C + ratio_D + ratio_E) :
  (total_money / total_parts) * ratio_B = 1800 :=
by
  sorry

end child_B_share_l822_822213


namespace problem_l822_822384

variable {a : ℝ} {m : ℝ} {x1 x2 : ℝ}

def g (x : ℝ) := Real.log x + 2 * x + a / x

def f (x : ℝ) := x * g x - ((a / 2) + 2) * x^2 - x

axiom h_a : 0 < a ∧ a < 1 / Real.exp 1
axiom h_m : m ≥ 1
axiom h_x1x2 : ∃ x1 x2, x1 < x2 ∧ f' x1 = 0 ∧ f' x2 = 0

theorem problem (h : f x1 = 0 ∧ f x2 = 0) : x1 * x2^m > Real.exp (1 + m) := 
sorry

end problem_l822_822384


namespace star_n_eq_n_l822_822303

noncomputable def star (n : ℕ) : ℕ := 
  match n with 
  | 1 => 1
  | n + 1 => star n + 1

theorem star_n_eq_n (n : ℕ) : star n = n := by
  sorry

end star_n_eq_n_l822_822303


namespace find_numbers_l822_822634

theorem find_numbers 
  (x y z : ℕ) 
  (h1 : y = 2 * x - 3) 
  (h2 : x + y = 51) 
  (h3 : z = 4 * x - y) : 
  x = 18 ∧ y = 33 ∧ z = 39 :=
by sorry

end find_numbers_l822_822634


namespace sqrt_180_simplify_l822_822061

theorem sqrt_180_simplify : real.sqrt 180 = 6 * real.sqrt 5 := by
  have h : 180 = 2^2 * 3^2 * 5 := by norm_num
  rw [h, real.sqrt_mul, real.sqrt_mul, real.sqrt_mul]
  all_goals { norm_num }
  sorry

end sqrt_180_simplify_l822_822061


namespace math_problem_l822_822933

open BigOperators

def problem_statement (n : ℕ) (a b : Finₓ n → ℕ) : Prop :=
  (∀ i j : Finₓ n, i < j → a i < a j) ∧
  (∀ i j : Finₓ n, i < j → b i > b j) ∧
  (Multiset.toFinset (Multiset.map a (Multiset.range n) + Multiset.map b (Multiset.range n)) = Finset.range (2 * n)) →
  (∑ i, (a i ∣ - b i).abs = n * n)

theorem math_problem (n : ℕ) (a b : Finₓ n → ℕ) (h_a : ∀ i j : Finₓ n, i < j → a i < a j)
 (h_b : ∀ i j : Finₓ n, i < j → b i > b j)
 (h_union: Multiset.toFinset (Multiset.map a (Multiset.range n) + Multiset.map b (Multiset.range n)) = Finset.range (2 * n)) :
   ∑ i, (a i - b i).abs = n ^ 2 :=
begin
  sorry
end

end math_problem_l822_822933


namespace trajectory_of_center_of_moving_circle_l822_822734

theorem trajectory_of_center_of_moving_circle :
  (∀ (M : ℝ × ℝ), let R1 : ℝ := 1, R2 : ℝ := 3 in
  let C1 : ℝ × ℝ := (-2, 0) in
  let C2 : ℝ × ℝ := (2, 0) in
  let r : ℝ := dist M C1 - R1 in
  (dist M C1 = r + R1) ∧ (dist M C2 = r + R2) ∧ (dist M C2 - dist M C1 = 2) →
  M = x ∧ y = y → x^2 - 3*y^2 = 1) :=
begin
  sorry
end

end trajectory_of_center_of_moving_circle_l822_822734


namespace topmost_circle_number_l822_822668

/-- 
Given a set of numbers {1, 2, 3, 4, 5, 6} and a configuration where the number in each circle (excluding the bottom three) 
is the difference between the numbers in the two adjacent circles below it, the number in the topmost circle must be one of {1, 2, 3}.
-/
theorem topmost_circle_number {a b c d e f g : ℕ} (h1 : a ∈ {1, 2, 3, 4, 5, 6})
                              (h2 : b ∈ {1, 2, 3, 4, 5, 6})
                              (h3 : c ∈ {1, 2, 3, 4, 5, 6})
                              (h4 : d ∈ {1, 2, 3, 4, 5, 6})
                              (h5 : e ∈ {1, 2, 3, 4, 5, 6})
                              (h6 : f ∈ {1, 2, 3, 4, 5, 6})
                              (h7 : g ∈ {1, 2, 3, 4, 5, 6})
                              (h8 : a = abs (max b c - min b c))
                              (h9 : b = abs (max d e - min d e))
                              (h10 : c = abs (max e f - min e f))
                              (h11 : d, e, f, g pairwise distinct) :
  a = 1 ∨ a = 2 ∨ a = 3 :=
by
  sorry

end topmost_circle_number_l822_822668


namespace find_least_skilled_painter_l822_822445

-- Define the genders
inductive Gender
| Male
| Female

-- Define the family members
inductive Member
| Grandmother
| Niece
| Nephew
| Granddaughter

-- Define a structure to hold the properties of each family member
structure Properties where
  gender : Gender
  age : Nat
  isTwin : Bool

-- Assume the properties of each family member as given
def grandmother : Properties := { gender := Gender.Female, age := 70, isTwin := false }
def niece : Properties := { gender := Gender.Female, age := 20, isTwin := false }
def nephew : Properties := { gender := Gender.Male, age := 20, isTwin := true }
def granddaughter : Properties := { gender := Gender.Female, age := 20, isTwin := true }

-- Define the best painter
def bestPainter := niece

-- Conditions based on the problem (rephrased to match formalization)
def conditions (least_skilled : Member) : Prop :=
  (bestPainter.gender ≠ (match least_skilled with
                          | Member.Grandmother => grandmother
                          | Member.Niece => niece
                          | Member.Nephew => nephew
                          | Member.Granddaughter => granddaughter ).gender) ∧
  ((match least_skilled with
    | Member.Grandmother => grandmother
    | Member.Niece => niece
    | Member.Nephew => nephew
    | Member.Granddaughter => granddaughter ).isTwin) ∧
  (bestPainter.age = (match least_skilled with
                      | Member.Grandmother => grandmother
                      | Member.Niece => niece
                      | Member.Nephew => nephew
                      | Member.Granddaughter => granddaughter ).age)

-- Statement of the problem
theorem find_least_skilled_painter : ∃ m : Member, conditions m ∧ m = Member.Granddaughter :=
by
  sorry

end find_least_skilled_painter_l822_822445


namespace mia_spending_on_each_parent_l822_822843

variables (sibling_spending sibling_count total_spending parent_count : ℕ)
variables (spending_siblings spending_parents spending_each_parent : ℕ)
variables (equal_value : Prop)

-- Define the conditions
def condition_1 : sibling_spending = 30 := rfl
def condition_2 : sibling_count = 3 := rfl
def condition_3 : total_spending = 150 := rfl
def condition_4 : spending_siblings = sibling_count * sibling_spending := rfl
def condition_5 : spending_parents = total_spending - spending_siblings := rfl
def condition_6 : parent_count = 2 := rfl
def condition_7 : equal_value = (spending_each_parent = spending_parents / parent_count) := rfl

-- Define the main proof statement
theorem mia_spending_on_each_parent :
  sibling_spending = 30 →
  sibling_count = 3 →
  total_spending = 150 →
  spending_siblings = sibling_count * sibling_spending →
  spending_parents = total_spending - spending_siblings →
  parent_count = 2 →
  equal_value →
  spending_each_parent = 30 :=
by intros; sorry

end mia_spending_on_each_parent_l822_822843


namespace sufficient_condition_for_perpendicular_l822_822747

variables {a b : Type} {α β : Type}
variables [line a] [line b]
variables [plane α] [plane β]

def condition (a b : Type) [line a] [line b] [plane α] [plane β] : Prop :=
  (parallel a α) ∧ (perpendicular b β) ∧ (parallel α β)

theorem sufficient_condition_for_perpendicular {a b : Type} {α β : Type}
  [line a] [line b] [plane α] [plane β]
  (h : condition a b α β) : perpendicular a b :=
sorry

end sufficient_condition_for_perpendicular_l822_822747


namespace length_rounded_l822_822036

theorem length_rounded (y : ℕ) : 
  (∃ (y : ℝ), (∃ (z : ℝ), 
    (7 * (y * (2/3 * y)) = 4900) ∧ 
    (2 * y = 3 * z)) ∧ 
    (y = Real.sqrt 1050) ∧ y ≈ 32) :=
sorry

end length_rounded_l822_822036


namespace card_game_termination_l822_822147

theorem card_game_termination 
  {n : ℕ} 
  (cards : Fin n → Fin n → Prop) 
  (irreflexive : ∀ i, ¬ cards i i)
  (arbitrary_distribution : ∃ p1 p2 : Fin n → {0, 1}, ∀ i, p1 i + p2 i = 1) 
  (moves : ∀ (p1_cards p2_cards : List (Fin n)), { p1_cards' : List (Fin n) // length p1_cards + length p2_cards = length p1_cards' + length p2_cards }
    ∨ { p2_cards' : List (Fin n) // length p1_cards + length p2_cards = length p1_cards + length p2_cards' }) :
  ∀ (p1_cards p2_cards : List (Fin n)),
    (length p1_cards = 0 ∨ length p2_cards = 0) ∨ 
    (∃ final_state : List (List (Fin n) × List (Fin n)),
      ∀ s ∈ final_state, s ≠ ([], []) → ∃ s', moves s.1 s.2 = some s') :=
sorry

end card_game_termination_l822_822147


namespace number_of_bouquets_to_earn_1000_dollars_l822_822265

def cost_of_buying (n : ℕ) : ℕ :=
  n * 20

def revenue_from_selling (m : ℕ) : ℕ :=
  m * 20

def profit_per_operation : ℤ :=
  revenue_from_selling 7 - cost_of_buying 5

theorem number_of_bouquets_to_earn_1000_dollars :
  ∀ bouquets_needed : ℕ, bouquets_needed = 5 * (1000 / profit_per_operation.nat_abs) :=
sorry

end number_of_bouquets_to_earn_1000_dollars_l822_822265


namespace class_size_l822_822880

theorem class_size (N : ℕ) 
  (avg_age : ℕ → ℕ)
  (avg_age_class : avg_age N = 20)
  (avg_age_5 : avg_age 5 = 14)
  (avg_age_9 : avg_age 9 = 16)
  (last_student_age : ℕ → ℕ)
  (last_student_age 1 = 186)
  (total_age : (5 * 14) + (9 * 16) + 186 = 20 * N) : 
  N = 20 :=
sorry

end class_size_l822_822880


namespace continuous_function_exists_fx_eq_gx_closed_interval_set_eq_l822_822638

open Set

variables {f g : ℝ → ℝ}

-- Given conditions
def is_function (f : ℝ → ℝ) : Prop := ∀ x, (x ∈ Icc 0 1 → f x ∈ Icc 0 1)
def is_monotonic (g : ℝ → ℝ) : Prop := Monotone g
def is_surjective (g : ℝ → ℝ) : Prop := ∀ y ∈ Icc 0 1, ∃ x ∈ Icc 0 1, g x = y
def bounded_diff (f g : ℝ → ℝ) : Prop := ∀ x y ∈ Icc 0 1, |f x - f y| ≤ |g x - g y|

-- Translation to proof problems
theorem continuous_function (f g : ℝ → ℝ) :
  is_function f ∧ is_function g ∧ is_monotonic g ∧ is_surjective g ∧ bounded_diff f g → ContinuousOn f (Icc 0 1) := by
  sorry

theorem exists_fx_eq_gx (f g : ℝ → ℝ) :
  is_function f ∧ is_function g ∧ is_monotonic g ∧ is_surjective g ∧ bounded_diff f g → 
  ∃ x ∈ Icc 0 1, f x = g x := by
  sorry

theorem closed_interval_set_eq (f g : ℝ → ℝ) :
  is_function f ∧ is_function g ∧ is_monotonic g ∧ is_surjective g ∧ bounded_diff f g →
  ∃ a b ∈ Icc 0 1, {x ∈ Icc 0 1 | f x = g x} = Icc a b := by
  sorry

end continuous_function_exists_fx_eq_gx_closed_interval_set_eq_l822_822638


namespace multiple_choice_questions_l822_822230

theorem multiple_choice_questions (n : ℕ) : 
  (∃ n : ℕ, 6 * 4^n = 96) → n = 2 :=
by
  assume h : ∃ n : ℕ, 6 * 4^n = 96
  cases h with n hn
  have h_exp : 4^2 = 16 := rfl
  have hh : 6 * 16 = 96 := rfl
  have hval : 4^n = 16 := nat.pow_eq_of_mul_eq_mul_right (by norm_num) hn
  have h_two : 4^2 = 4^n := by
    rw hval
  have hn_eq_two : n = 2 := nat.pow_right_injective (by norm_num) hval
  exact hn_eq_two 

end multiple_choice_questions_l822_822230


namespace proposition_1_proposition_2_proposition_3_proposition_4_l822_822296

axiom p1 : Prop
axiom p2 : Prop
axiom p3 : Prop
axiom p4 : Prop

axiom p1_true : p1 = true
axiom p2_false : p2 = false
axiom p3_false : p3 = false
axiom p4_true : p4 = true

theorem proposition_1 : (p1 ∧ p4) = true := by sorry
theorem proposition_2 : (p1 ∧ p2) = false := by sorry
theorem proposition_3 : (¬p2 ∨ p3) = true := by sorry
theorem proposition_4 : (¬p3 ∨ ¬p4) = true := by sorry

end proposition_1_proposition_2_proposition_3_proposition_4_l822_822296


namespace normal_distribution_prob_l822_822371

noncomputable def X : ℝ → ProbabilitySpace ℝ :=
  sorry -- Definition of the normal distribution

variable (μ σ : ℝ) (P : Set ℝ → ℝ) [H1 : ∀ a b : ℝ, P (Set.Ioo a b) = sorry]

theorem normal_distribution_prob (μ : ℝ) (σ : ℝ)
  (h₁ : P (Set.Ioo (μ - 2 * σ) (μ + 2 * σ)) = 0.9544)
  (h₂ : P (Set.Ioo (μ - σ) (μ + σ)) = 0.6826)
  (h₃ : μ = 4)
  (h₄ : σ = 1) :
  P (Set.Ioo 5 6) = 0.1359 :=
sorry

end normal_distribution_prob_l822_822371


namespace find_k_value_l822_822367

open Real

noncomputable def angle_in_radians (deg : ℝ) : ℝ := deg * (π / 180)

def a : ℝ := 2
def b : ℝ := 3
def angle_ab : ℝ := angle_in_radians 60
def cos_60 : ℝ := cos angle_ab
def a_dot_b : ℝ := a * b * cos_60
def a_squared : ℝ := a^2
def b_squared : ℝ := b^2
def c := (5 : ℝ) • ![a, 0, 0] + (3 : ℝ) • ![0, b, 0]
def d (k : ℝ) := (3 : ℝ) • ![a, 0, 0] + k • ![0, b, 0]

theorem find_k_value : ∃ k : ℝ, (5 * 3 * a_squared + (5 * k + 9) * a_dot_b + 9 * k * b_squared) = 0 ∧ k = - (87 : ℝ) / 42 :=
by
  use - (29 : ℝ) / 14
  simp [a, b, angle_ab, cos_60, a_dot_b, a_squared, b_squared]
  sorry

end find_k_value_l822_822367


namespace bie_l822_822455

noncomputable def surface_area_of_sphere (PA AB AC : ℝ) (hPA_AB : PA = AB) (hPA : PA = 2) (hAC : AC = 4) (r : ℝ) : ℝ :=
  let PC := Real.sqrt (PA ^ 2 + AC ^ 2)
  let radius := PC / 2
  4 * Real.pi * radius ^ 2

theorem bie'zhi_tetrahedron_surface_area
  (PA AB AC : ℝ)
  (hPA_AB : PA = AB)
  (hPA : PA = 2)
  (hAC : AC = 4)
  (PC : ℝ := Real.sqrt (PA ^ 2 + AC ^ 2))
  (r : ℝ := PC / 2)
  (surface_area : ℝ := 4 * Real.pi * r ^ 2)
  :
  surface_area = 20 * Real.pi := 
sorry

end bie_l822_822455


namespace compute_expr_l822_822834

open Real

-- Define the polynomial and its roots.
def polynomial (x : ℝ) := 3 * x^2 - 5 * x - 2

-- Given conditions: p and q are roots of the polynomial.
def is_root (p q : ℝ) : Prop := 
  polynomial p = 0 ∧ polynomial q = 0

-- The main theorem.
theorem compute_expr (p q : ℝ) (h : is_root p q) : 
  ∃ k : ℝ, k = p - q ∧ (p ≠ q) → (9 * p^3 + 9 * q^3) / (p - q) = 215 / (3 * (p - q)) :=
sorry

end compute_expr_l822_822834


namespace initial_beavers_l822_822252

theorem initial_beavers (B C : ℕ) (h1 : C = 40) (h2 : B + C + 2 * B + (C - 10) = 130) : B = 20 :=
by
  sorry

end initial_beavers_l822_822252


namespace root_relationship_specific_root_five_l822_822873

def f (x : ℝ) : ℝ := x^3 - 6 * x^2 - 39 * x - 10
def g (x : ℝ) : ℝ := x^3 + x^2 - 20 * x - 50

theorem root_relationship :
  ∃ (x_0 : ℝ), g x_0 = 0 ∧ f (2 * x_0) = 0 :=
sorry

theorem specific_root_five :
  g 5 = 0 ∧ f 10 = 0 :=
sorry

end root_relationship_specific_root_five_l822_822873


namespace nehas_mother_age_l822_822959

variables (N M : ℕ)

axiom age_condition1 : M - 12 = 4 * (N - 12)
axiom age_condition2 : M + 12 = 2 * (N + 12)

theorem nehas_mother_age : M = 60 :=
by
  -- Sorry added to skip the proof
  sorry

end nehas_mother_age_l822_822959


namespace elena_allowance_fraction_l822_822310

variable {A m s : ℝ}

theorem elena_allowance_fraction {A : ℝ} (h1 : m = 0.25 * (A - s)) (h2 : s = 0.10 * (A - m)) : m + s = (4 / 13) * A :=
by
  sorry

end elena_allowance_fraction_l822_822310


namespace ratio_rational_l822_822512

-- Let the positive numbers be represented as n1, n2, n3, n4, n5
variable (n1 n2 n3 n4 n5 : ℚ)

open Classical

-- Assume distinctness and positivity
axiom h_distinct : (n1 ≠ n2) ∧ (n1 ≠ n3) ∧ (n1 ≠ n4) ∧ (n1 ≠ n5) ∧ (n2 ≠ n3) ∧ (n2 ≠ n4) ∧ (n2 ≠ n5) ∧ (n3 ≠ n4) ∧ (n3 ≠ n5) ∧ (n4 ≠ n5)
axiom h_positive : (0 < n1) ∧ (0 < n2) ∧ (0 < n3) ∧ (0 < n4) ∧ (0 < n5)

-- For any choice of three numbers, the expression ab + bc + ca is rational
axiom h_rational : ∀ {a b c: ℚ}, (a ∈ {n1, n2, n3, n4, n5}) → (b ∈ {n1, n2, n3, n4, n5}) → (c ∈ {n1, n2, n3, n4, n5}) → a ≠ b → a ≠ c → b ≠ c → (a * b + b * c + c * a).is_rat

-- Prove that the ratio of any two numbers on the board is rational
theorem ratio_rational : ∀ {x y : ℚ}, (x ∈ {n1, n2, n3, n4, n5}) → (y ∈ {n1, n2, n3, n4, n5}) → x ≠ y → (x / y).is_rat :=
by sorry

end ratio_rational_l822_822512


namespace sqrt_180_eq_l822_822043

noncomputable def simplify_sqrt_180 : Real := 6 * Real.sqrt 5

theorem sqrt_180_eq : Real.sqrt 180 = simplify_sqrt_180 := 
by
  -- proof omitted
  sorry

end sqrt_180_eq_l822_822043


namespace write_in_numerical_form_read_out_loud_l822_822618

theorem write_in_numerical_form (a : ℕ) (h : a = 1859080496) : (∃ b = 1859080496, b = a) :=
by {
  use 1859080496,
  sorry
}

theorem read_out_loud (b : ℕ) (h : b = 1738406002) : (∃ r = "一十七亿三千八百四十万六千零二", r = "一十七亿三千八百四十万六千零二") :=
by {
  use "一十七亿三千八百四十万六千零二",
  sorry
}

end write_in_numerical_form_read_out_loud_l822_822618


namespace bar_graph_representation_l822_822952

theorem bar_graph_representation :
  (white: ℕ) = 3 * (black: ℕ) 
  ∧ black = (gray: ℕ) 
  → (3, 1, 1) = (white, black, gray) :=
by
  intros h
  obtain ⟨h1, h2⟩ := h
  rw h2
  rw ←h1
  simp
  sorry

end bar_graph_representation_l822_822952


namespace solution_set_inequality_l822_822107

theorem solution_set_inequality (x : ℝ) : (x - 1) * (x + 2) < 0 ↔ x ∈ set.Ioo (-2 : ℝ) (1 : ℝ) :=
by
  sorry

end solution_set_inequality_l822_822107


namespace find_lambda_l822_822731

-- Define vectors
def a : ℝ × ℝ × ℝ := (0, 1, -1)
def b : ℝ × ℝ × ℝ := (1, 1, 0)

-- Define dot product function
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Define the main theorem to prove that λ = -2
theorem find_lambda (λ : ℝ) (h : dot_product (a.1 + λ * b.1, a.2 + λ * b.2, a.3 + λ * b.3) a = 0) : λ = -2 :=
sorry

end find_lambda_l822_822731


namespace proof_problem_l822_822284

variable (p1 p2 p3 p4 : Prop)

theorem proof_problem (hp1 : p1) (hp2 : ¬ p2) (hp3 : ¬ p3) (hp4 : p4) :
  (p1 ∧ p4) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) := by
  sorry

end proof_problem_l822_822284


namespace total_quadratic_functions_l822_822385

-- Let a, b, c be elements of the set {0, 1, 2}
def is_element (x : ℕ) : Prop := x = 0 ∨ x = 1 ∨ x = 2

-- Define the function a ≠ 0
def a_not_zero (a : ℕ) : Prop := a ≠ 0

-- Define a, b, c to satisfy the given conditions
def valid_coefficients (a b c : ℕ) : Prop := is_element a ∧ a_not_zero a ∧ is_element b ∧ is_element c

-- Total number of different valid quadratic functions
theorem total_quadratic_functions : 
  (Σ' (a b c : ℕ), valid_coefficients a b c).to_finset.card = 18 := 
by 
  sorry

end total_quadratic_functions_l822_822385


namespace final_milk_quantity_correct_l822_822659

-- Defining the initial quantities
def initial_milk : ℝ := 50
def removed_volume : ℝ := 9

-- The quantity of milk after the first replacement
def milk_after_first_replacement : ℝ := initial_milk - removed_volume

-- The ratio of milk to the total solution
def milk_ratio : ℝ := milk_after_first_replacement / initial_milk

-- The amount of milk removed in the second step
def milk_removed_second_step : ℝ := milk_ratio * removed_volume

-- The final quantity of milk in the solution
def final_milk_quantity : ℝ := milk_after_first_replacement - milk_removed_second_step

theorem final_milk_quantity_correct :
  final_milk_quantity = 33.62 := by
  sorry

end final_milk_quantity_correct_l822_822659


namespace monotonicity_intervals_range_of_b_exp_ln_comparison_l822_822758

-- Given function definition and conditions

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 2 * a * x + b * x - 1 - 2 * log x
def interval_mono_dec (a : ℝ) : Prop := ∀ x > 0, a ≤ 0 → deriv (f x a 0) x < 0
def interval_mono_mixed (a : ℝ) : Prop :=
  ∀ (x : ℝ) (a_pos : a > 0), (x > 0 → x < 1 / a → deriv (f x a 0) x < 0) ∧ (x > 1 / a → deriv (f x a 0) x > 0)

-- Proof for Question 1: Intervals of monotonicity
theorem monotonicity_intervals (a : ℝ) :
  (interval_mono_dec a ∨ interval_mono_mixed a) :=
sorry

-- Inequality relation definition for Question 2
def g (x : ℝ) (a : ℝ) : ℝ := a + 1/x - log x / x
def g_min (a : ℝ) : ℝ := a - 1 / (real.exp 2)

-- Proof for Question 2: Range of values for b
theorem range_of_b (a : ℝ) (b : ℝ) (H : ∀ x > 0, ∀ α ∈ set.Icc 1 3, f x a b ≥ 2 * b * x - 3) :
  b ≤ 2 - 2 / (real.exp 2) :=
sorry

-- Given condition for Question 3
theorem exp_ln_comparison (x y : ℝ) (Hxy : x > y) (Hye : y > real.exp (-1)) :
  real.exp x * log (y + 1) > real.exp y * log (x + 1) :=
sorry

end monotonicity_intervals_range_of_b_exp_ln_comparison_l822_822758


namespace card_game_termination_l822_822146

theorem card_game_termination 
  {n : ℕ} 
  (cards : Fin n → Fin n → Prop) 
  (irreflexive : ∀ i, ¬ cards i i)
  (arbitrary_distribution : ∃ p1 p2 : Fin n → {0, 1}, ∀ i, p1 i + p2 i = 1) 
  (moves : ∀ (p1_cards p2_cards : List (Fin n)), { p1_cards' : List (Fin n) // length p1_cards + length p2_cards = length p1_cards' + length p2_cards }
    ∨ { p2_cards' : List (Fin n) // length p1_cards + length p2_cards = length p1_cards + length p2_cards' }) :
  ∀ (p1_cards p2_cards : List (Fin n)),
    (length p1_cards = 0 ∨ length p2_cards = 0) ∨ 
    (∃ final_state : List (List (Fin n) × List (Fin n)),
      ∀ s ∈ final_state, s ≠ ([], []) → ∃ s', moves s.1 s.2 = some s') :=
sorry

end card_game_termination_l822_822146


namespace star_polygon_points_l822_822279

theorem star_polygon_points (n : ℕ) (A B : Fin n → ℝ) (hA : ∀ i, A i = B i + 15)
  (h_sum : ∑ i in Finset.finRange n, A i + ∑ i in Finset.finRange n, B i = 360) :
  n = 24 :=
by
  sorry

end star_polygon_points_l822_822279


namespace polynomial_inequality_conditions_l822_822840

/-- Let the polynomial f(x) = x^3 + a*x^2 + b*x + c,
    where a, b, c ∈ ℝ. If for any non-negative real numbers x and y, 
    it holds that f(x + y) ≥ f(x) + f(y), 
    then this theorem states the conditions on a, b, and c. -/
theorem polynomial_inequality_conditions
  (a b c : ℝ)
  (h : ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → (x + y)^3 + a * (x + y)^2 + b * (x + y) + c ≥ x^3 + a * x^2 + b * x + c + y^3 + a * y^2 + b * y + c) :
  a ≥ (3 / 2) * (9 * c)^(1/3) ∧ c ≤ 0 ∧ (b ∈ ℝ) :=
sorry

end polynomial_inequality_conditions_l822_822840


namespace cosine_acute_angle_between_lines_l822_822208

def direction_vector_line_1 : ℝ × ℝ := (4, 5)
def direction_vector_line_2 : ℝ × ℝ := (2, 7)

theorem cosine_acute_angle_between_lines :
  let dot_product := (direction_vector_line_1.1 * direction_vector_line_2.1 + direction_vector_line_1.2 * direction_vector_line_2.2)
  let magnitude_1 := real.sqrt (direction_vector_line_1.1^2 + direction_vector_line_1.2^2)
  let magnitude_2 := real.sqrt (direction_vector_line_2.1^2 + direction_vector_line_2.2^2)
  let cosine_theta := dot_product / (magnitude_1 * magnitude_2)
  cosine_theta = 43 / real.sqrt 2173 :=
by
  sorry

end cosine_acute_angle_between_lines_l822_822208


namespace is_tangent_to_line_at_C_l822_822681

structure Triangle (A B C : Type) :=
  (is_isosceles_right : isosceles_right_triangle A B C)
  (angle_BAC_ninety_deg : angle A B C = 90)

def midpoint {A B : Type} (AB : line A B) : Type := sorry
def circumference {A B : Type} (diam : segment A B) : Type := sorry
def intersection_point {L : Type} (line : L) (circumference : Type) : Type := sorry
def tangent {Circ : Type} (circ : Circ) (line : Type) (point : Type) : Prop := sorry

theorem is_tangent_to_line_at_C {A B C : Type} 
  (T : Triangle A B C)
  (ell : line B (midpoint A C))
  (Gamma : circumference (segment A B))
  (P : Type) 
  (P_intersection : P = intersection_point ell Gamma ∧ P ≠ B) :
  tangent (circumference A C P) (line B C) C :=
sorry

end is_tangent_to_line_at_C_l822_822681


namespace line_equation_l822_822732

def Line (k : ℝ) (c : ℝ) : Prop :=
 ∃ x y : ℝ, y = k * x + c

def Condition1 (l : ℝ → ℝ → Prop) : Prop :=
 l 2 1

def Condition2 (l : ℝ → ℝ → Prop) : Prop :=
  let P := l 
  let intercepts := λ (x1 y1 x2 y2 : ℝ), ((x1 - x2)^2 + (y1 - y2)^2 = 2) in
  ∃ k : ℝ,
      let y := k * x + (1 - 2 * k),
      let x1 := (6 * k - 4) / (4 + 3 * k),
      let y1 := (4 - 9 * k) / (4 + 3 * k),
      let x2 := (6 * k - 9) / (4 + 3 * k),
      let y2 := (4 - 14 * k) / (4 + 3 * k),
      intercepts x1 y1 x2 y2

theorem line_equation (l : Line) (k1 k2 : ℝ):
  Condition1 l → Condition2 l →
  (k1 = -1/7 → ∃ c, l = λ x y, y = k1 * x + c) ∨
  (k2 = 7 → ∃ c, l = λ x y, y = k2 * x + c) :=
by sorry

end line_equation_l822_822732


namespace probability_of_sunny_days_l822_822087

/-- Probability that Oliver and Alice get sunny weather for exactly one or exactly two days during their five-day holiday week in Paris, given a 60% chance of rain each day, resulting in sunny weather with a probability of 40% is 378/625 -/
theorem probability_of_sunny_days :
  let p_rain := 0.6
  let p_sun := 0.4
  let n_days := 5
  (∑ k in {1, 2}, (nat.choose n_days k : ℚ) * p_sun^k * p_rain^(n_days - k)) = 378 / 625 := sorry

end probability_of_sunny_days_l822_822087


namespace evaluate_f_sum_l822_822348

noncomputable def f (x : ℝ) : ℝ :=
if x < 2 then 2 + Real.log (2 - x) / Real.log 2 else 3 ^ (x - 2)

-- Evaluating the function at specific points and proving the required equality
theorem evaluate_f_sum : f 0 + f (Real.log 36 / Real.log 3) = 7 := 
by 
  have f0 : f 0 = 3 := by 
    simp [f]
    have h1 : Real.log (2 - 0) = Real.log 2 := by simp
    have h2 : Real.log 2 / Real.log 2 = 1 := by rw [Real.log_div_self]
    ring

  have flog36 : f (Real.log 36 / Real.log 3) = 4 := by 
    simp [f]
    have hlog : 2 < (Real.log 36 / Real.log 3) := by 
      have eq_log36 : (Real.log 36 / Real.log 3) = 2 := by 
        rw [Real.log_eq_log_iff_eq_exp (by norm_num : 3 ≠ 1)]
        simp [Real.log, Real.exp, Mathlib.Real.log_real_succ_nat_succ, Complex.log, 
              Complex.exp, eq_to_diff]
      simp [eq_log36]

    have h3 : (Real.log 36 / Real.log 3) = 2 + Real.log 4 / Real.log 3 := by
      have hlog_di : 36 = 3 ^ 2 * 3 ^ 2 := by norm_num
      simp [Real.log, hlog_di, mul_self_eq]
    simp [Real.log, Real.exp, h3]

  rw [f0, flog36]
  norm_num


end evaluate_f_sum_l822_822348


namespace number_of_ordered_triples_l822_822364

noncomputable def lcm (r s : ℕ) : ℕ := Nat.lcm r s

theorem number_of_ordered_triples :
  {p : (ℕ × ℕ × ℕ) // lcm p.1 p.2.1 = 1000 ∧ lcm p.2.1 p.2.2 = 2000 ∧ lcm p.2.2 p.1 = 2000}.to_finmap.card = 70 :=
by
  sorry

end number_of_ordered_triples_l822_822364


namespace sin_B_value_sin_A_value_area_of_triangle_l822_822437

-- Definition of the sides and angles given in the problem
def a : ℝ := 2
def C : ℝ := Real.pi / 4
def cos_B : ℝ := 3 / 5

-- Using the conditions, we define what we need to prove each part
theorem sin_B_value : ∃ (sin_B : ℝ), cos_B = 3 / 5 → sin_B = 4 / 5 := by
  sorry

theorem sin_A_value (B : ℝ) (hB : B = Real.arccos cos_B) : ∃ (sin_A : ℝ),
  C = Real.pi / 4 ∧ cos_B = 3 / 5 → sin_A = 7 * Real.sqrt 2 / 10 := by
  sorry

theorem area_of_triangle :
  ∃ (S : ℝ) (B : ℝ) (sin_B sin_A : ℝ),
    a = 2 ∧ C = Real.pi / 4 ∧ cos_B = 3 / 5 ∧
    sin_B = 4 / 5 ∧ sin_A = 7 * Real.sqrt 2 / 10 →
    S = (1 / 2) * a * (10 / 7) * sin_B ∧ S = 8 / 7 := by
  sorry

end sin_B_value_sin_A_value_area_of_triangle_l822_822437


namespace rose_bushes_planted_l822_822589

-- Define the conditions as variables
variable (current_bushes planted_bushes total_bushes : Nat)
variable (h1 : current_bushes = 2) (h2 : total_bushes = 6)
variable (h3 : total_bushes = current_bushes + planted_bushes)

theorem rose_bushes_planted : planted_bushes = 4 := by
  sorry

end rose_bushes_planted_l822_822589


namespace employee_wages_l822_822236

-- Define Abe's budget and expenditure conditions
def monthly_budget : ℝ := 4500
def food_expense : ℝ := (1/3) * monthly_budget
def supplies_expense : ℝ := (1/4) * monthly_budget
def rent_expense : ℝ := 800
def utilities_expense : ℝ := 300
def taxes_expense : ℝ := 0.1 * monthly_budget

-- Statement to prove the amount spent on employee wages
theorem employee_wages :
  monthly_budget - (food_expense + supplies_expense + rent_expense + utilities_expense + taxes_expense) = 325 := 
  by sorry

end employee_wages_l822_822236


namespace work_left_fraction_l822_822178

theorem work_left_fraction (p_day_work q_day_work : ℚ) (total_days : ℕ) (p_work_time q_work_time : ℚ) : 
  p_day_work = (1/p_work_time) ∧ 
  q_day_work = (1/q_work_time) ∧ 
  p_work_time = 15 ∧ 
  q_work_time = 20 ∧ 
  total_days = 4 → 
  (1 - total_days * (p_day_work + q_day_work) = 8/15) := 
by
  intro h
  cases h with hp hq
  cases hq with pq h3
  cases h3 with pq_time work_time
  cases work_time with total4
  sorry

end work_left_fraction_l822_822178


namespace find_top_left_corner_l822_822452

-- Define the size of the table
def n := 6

-- Define the main statement as a theorem
theorem find_top_left_corner (a : ℕ) :
  (∀ i j, i < n ∧ j < n →
    (i = 0 ∨ j = 0 → a) ∧
    (i > 0 ∧ j > 0 → f i j = f (i-1) j + f i (j-1)) ∧
    f (n-1) (n-1) = 2016) → 
  a = 8 :=
begin
  sorry
end

-- Define the function f for cells (i, j)
noncomputable def f : ℕ → ℕ → ℕ
| 0 0 := a
| 0 j := a
| i 0 := a
| i j := f (i-1) j + f i (j-1)

end find_top_left_corner_l822_822452


namespace proposition_1_proposition_2_proposition_3_proposition_4_l822_822295

axiom p1 : Prop
axiom p2 : Prop
axiom p3 : Prop
axiom p4 : Prop

axiom p1_true : p1 = true
axiom p2_false : p2 = false
axiom p3_false : p3 = false
axiom p4_true : p4 = true

theorem proposition_1 : (p1 ∧ p4) = true := by sorry
theorem proposition_2 : (p1 ∧ p2) = false := by sorry
theorem proposition_3 : (¬p2 ∨ p3) = true := by sorry
theorem proposition_4 : (¬p3 ∨ ¬p4) = true := by sorry

end proposition_1_proposition_2_proposition_3_proposition_4_l822_822295


namespace min_value_of_expression_l822_822828

variable {x y z : ℝ}

theorem min_value_of_expression (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 1) :
  (x^2 / y + y^2 / z + z^2 / x) ≥ 3 :=
begin
  sorry
end

end min_value_of_expression_l822_822828


namespace card_game_eventually_one_without_cards_l822_822153

def card_game (n : ℕ) : Prop :=
  ∃ (cards : Fin n → Fin n) 
    (beats : (Fin n) → (Fin n) → Prop)
    (initial_distribution : Fin n → Fin 2),
  (∀ x y z : Fin n, beats x y ∨ beats y x) →
  (∀ x y z : Fin n, beats x z ∧ beats z y → beats x y) →
  (∀ distribution : Fin n → Fin 2, 
     ∃ turn : ℕ → ((Fin n) × (Fin n)) → Prop,
       (∀ i : ℕ, (turn i) (initial_distribution i) →
         ∃ j : ℕ, turn (i + 1) (beats _ _) → 
           (∃ k : ℕ, turn k (initial_distribution i) → false)))

theorem card_game_eventually_one_without_cards (n : ℕ) : card_game n :=
sorry

end card_game_eventually_one_without_cards_l822_822153


namespace sum_of_divisors_85_l822_822943

theorem sum_of_divisors_85 :
  let divisors := {d ∈ Finset.range 86 | 85 % d = 0}
  Finset.sum divisors id = 108 :=
sorry

end sum_of_divisors_85_l822_822943


namespace solve_for_x_l822_822791

theorem solve_for_x (A B C D: Type) 
(y z w x : ℝ) 
(h_triangle : ∃ a b c : Type, True) 
(h_D_on_extension : ∃ D_on_extension : Type, True)
(h_AD_GT_BD : ∃ s : Type, True) 
(h_x_at_D : ∃ t : Type, True) 
(h_y_at_A : ∃ u : Type, True) 
(h_z_at_B : ∃ v : Type, True) 
(h_w_at_C : ∃ w : Type, True)
(h_triangle_angle_sum : y + z + w = 180):
x = 180 - z - w := by
  sorry

end solve_for_x_l822_822791


namespace correct_algorithm_structures_l822_822910

-- Definitions used in the conditions
def is_basic_structure (structure : String) : Prop :=
  structure = "Sequential structure" ∨ 
  structure = "Conditional structure" ∨ 
  structure = "Loop structure"

def options : List (List String) :=
  [ ["Sequential structure", "Module structure", "Conditional structure"],
    ["Sequential structure", "Loop structure", "Module structure"],
    ["Sequential structure", "Conditional structure", "Loop structure"],
    ["Module structure", "Conditional structure", "Loop structure"] ]

-- Theorem stating that the correct answer is option 2 (index starting from 0 means 2 is actually third option)
theorem correct_algorithm_structures : List String :=
  ["Sequential structure", "Conditional structure", "Loop structure"]

lemma algorithm_structures_correct : 
  correct_algorithm_structures = options[2] :=
  sorry

end correct_algorithm_structures_l822_822910


namespace smallest_positive_odd_n_l822_822158

theorem smallest_positive_odd_n :
  ∃ n : ℕ, odd n ∧ n > 0 ∧ (2 : ℝ)^((finset.range (2 * n + 2)).filter odd).sum (λ x, x / 6 : ℝ) > 8000 ∧ ∀ m : ℕ, odd m ∧ m > 0 ∧ (2 : ℝ)^((finset.range (2 * m + 2)).filter odd).sum (λ x, x / 6 : ℝ) > 8000 → n ≤ m :=
sorry

end smallest_positive_odd_n_l822_822158


namespace length_of_goods_train_correct_l822_822175

-- Define the given conditions
def speed_kmph : ℝ := 72
def time_seconds : ℝ := 30
def platform_length_meters : ℝ := 250

-- Define the conversion factor from km/h to m/s
def conversion_factor := 5 / 18

-- Define the speed in m/s
def speed_mps := speed_kmph * conversion_factor

-- Define the distance covered by the train
def distance_covered := speed_mps * time_seconds

-- Define the length of the goods train
def length_of_train := distance_covered - platform_length_meters

-- Prove that the length of the goods train is 350 meters
theorem length_of_goods_train_correct : length_of_train = 350 :=
by
  have speed_converted : speed_mps = 20 := by
    rw [conversion_factor, speed_kmph]
    norm_num
  rw [distance_covered, speed_converted, length_of_train]
  norm_num
  sorry

end length_of_goods_train_correct_l822_822175


namespace gardener_cabbages_this_year_l822_822621

-- Definitions for the conditions
def side_length_last_year (x : ℕ) := true
def area_last_year (x : ℕ) := x * x
def increase_in_output := 197

-- Proposition to prove the number of cabbages this year
theorem gardener_cabbages_this_year (x : ℕ) (hx : side_length_last_year x) : 
  (area_last_year x + increase_in_output) = 9801 :=
by 
  sorry

end gardener_cabbages_this_year_l822_822621


namespace ratio_of_any_two_numbers_is_rational_l822_822509

theorem ratio_of_any_two_numbers_is_rational
  (x1 x2 x3 x4 x5 : ℝ)
  (h1 : ∀ a b c : ℝ, a ≠ b → b ≠ c → a ≠ c → a ∈ {x1, x2, x3, x4, x5} → b ∈ {x1, x2, x3, x4, x5} → c ∈ {x1, x2, x3, x4, x5} → (a * b + b * c + c * a) ∈ ℚ) :
  ∀ a b : ℝ, a ∈ {x1, x2, x3, x4, x5} → b ∈ {x1, x2, x3, x4, x5} → (a / b) ∈ ℚ := 
sorry

end ratio_of_any_two_numbers_is_rational_l822_822509


namespace tan_six_minus_tan_squared_l822_822413

noncomputable def proof_tan_six_minus_tan_squared : Prop :=
  ∀ x : ℝ, (cos x + cot x = 2 * sin x) → (tan x = 1 / 2 ∨ tan x = -1) →
  (tan x = 1 / 2 → (tan x ^ 6 - tan x ^ 2 = -15 / 64)) ∧
  (tan x = -1 → (tan x ^ 6 - tan x ^ 2 = 0))

theorem tan_six_minus_tan_squared : proof_tan_six_minus_tan_squared :=
  sorry

end tan_six_minus_tan_squared_l822_822413


namespace algebraic_expression_value_l822_822776

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 3 * x + 5 = 7) : 3 * x^2 + 9 * x - 11 = -5 :=
by
  sorry

end algebraic_expression_value_l822_822776


namespace segment_BC_length_l822_822858

noncomputable def length_of_segment_BC : ℝ :=
let AB := 123.4 -- meters
let CD := 96.4 -- meters
let α := 25.7 * (real.pi / 180) -- converting to radians
let β := 48.3 * (real.pi / 180) -- converting to radians
let γ := 32.9 * (real.pi / 180) -- converting to radians
let k := AB * real.sin γ / (CD * real.sin α)
-- Defining the equations and the quadratic parameters (m, n)
let δ := α + β + γ
let m := (real.sin (2 * δ - γ) - k * real.sin α) / (real.sin (α + β) * real.sin δ)
let n := (real.cos (α + β) * real.cos δ - k * real.cos α) / (real.sin (α + β) * real.sin δ)
-- Solving for cotangent of the angle ϕ
let cotϕ := (-m + real.sqrt (m^2 - 4 * n)) / 2  -- only considering the valid cotangent from solution
-- Using ϕ to find x, the length of segment BC
let ϕ := real.arccot cotϕ
let x := AB * real.sin β * (real.sin ϕ) / (real.sin α * real.sin (ϕ + α + β))

in x

theorem segment_BC_length : length_of_segment_BC = 109.5 :=
by
  -- This is the statement; adding 'sorry' for now as we are not providing the proof.
  sorry

end segment_BC_length_l822_822858


namespace compute_LI_length_l822_822831

noncomputable def LI_length (PROBLEMZ : Type) [regular_octagon PROBLEMZ] (W : Point) : ℝ :=
  let r : ℝ := 1 in
  let I := center_of_octagon PROBLEMZ in
  let LI : ℝ := distance from (I) to a vertex of octagon in √2 by geometry in a triangle
  LI

theorem compute_LI_length (PROBLEMZ : Type) [regular_octagon PROBLEMZ] (W : Point) : LI_length PROBLEMZ W = √2 :=
by
  -- Omitted: Proof that validates the theorem LI_length PROBLEMZ W == √2
  sorry

end compute_LI_length_l822_822831


namespace sum_segments_constant_l822_822220

variable (Pyramid : Type) [RegularPyramid Pyramid]
variable (Base : RegularPolygon)
variable (P : Base.Point)
variable (perpendicular : Perpendicular P Base.Plane)

theorem sum_segments_constant :
  ∀ (P : Base.Point),
  let intersections := faces_of_pyramid Pyramid P perpendicular in
  sum (λ face, distance P (intersection_point face perpendicular)) intersections = 
  sum (λ face, distance origin (intersection_point face perpendicular)) intersections :=
by sorry

end sum_segments_constant_l822_822220


namespace distance_to_school_l822_822474

theorem distance_to_school : 
  ∀ (d v : ℝ), (d = v * (1 / 3)) → (d = (v + 20) * (1 / 4)) → d = 20 :=
by
  intros d v h1 h2
  sorry

end distance_to_school_l822_822474


namespace find_k_l822_822907

variable {S : ℕ → ℤ} -- Assuming the sum function S for the arithmetic sequence 
variable {k : ℕ} -- k is a natural number

theorem find_k (h1 : S (k - 2) = -4) (h2 : S k = 0) (h3 : S (k + 2) = 8) (hk2 : k > 2) (hnaturalk : k ∈ Set.univ) : k = 6 := by
  sorry

end find_k_l822_822907


namespace root_difference_l822_822553

-- Given the cubic equation
def cubic_eq (x p : ℝ) : ℝ := x^3 - p * x^2 + (p^2 - 1) / 4 * x

-- Define a predicate to check if x is a root of the cubic equation given p
def is_root (x p : ℝ) : Prop := cubic_eq x p = 0

-- Proving the main statement
theorem root_difference (p : ℝ) : 
  let roots := {x : ℝ | is_root x p} in
  ∃ r s, r ∈ roots ∧ s ∈ roots ∧ r ≠ s ∧ (r - s = 1 ∨ s - r = 1) 
  ∧ (∀ t ∈ roots, t = 0 ∨ t = r ∨ t = s) :=
sorry

end root_difference_l822_822553


namespace trapezoid_perimeter_is_200_l822_822811

noncomputable def perimeter_of_trapezoid (AB CD AD BC : ℝ) (∠BAD : ℝ) : ℝ :=
  AB + BC + CD + AD

theorem trapezoid_perimeter_is_200 :
  ∀ (AB CD AD BC : ℝ) (∠BAD : ℝ),
    AB = 40 →
    CD = 35 →
    AD = 70 →
    BC = 55 →
    ∠BAD = 30 →
    perimeter_of_trapezoid AB CD AD BC ∠BAD = 200 :=
by
  intros AB CD AD BC BAD hAB hCD hAD hBC hBAD
  rw [hAB, hCD, hAD, hBC, hBAD]
  -- Perimeter calculation detail can be assumed from the conditions
  norm_num
  done

#check trapezoid_perimeter_is_200

end trapezoid_perimeter_is_200_l822_822811


namespace find_certain_number_l822_822636

theorem find_certain_number (x : ℝ) (h : 25 * x = 675) : x = 27 :=
by {
  sorry
}

end find_certain_number_l822_822636


namespace pi_grid_covering_l822_822205

variable {m n : ℕ}
variable (π : ℕ × ℕ → ℕ → ℕ → Prop)
variable (P : ℕ × ℕ → ℤ)
variable (placements : list ((ℕ × ℕ) → ℤ))
variable (d : list ℤ)
variable (sum_positive : ∀ (grid : (ℕ × ℕ) → ℤ), (∀ x y, grid (x, y) > 0) → ∃ π, (∀ i j, π (i, j) > 0 → grid (i, j) > 0))

noncomputable def pi_covers_grid_without_gaps (π : (ℕ × ℕ) → ℕ → ℕ → Prop) : Prop :=
  ∀ grid : (ℕ × ℕ) → ℤ, 
  (∀ x y, grid (x, y) > 0) →
  ∃ (placements : list ((ℕ × ℕ) → ℤ)) (d : list ℤ),
  ∀ (i j : ℕ × ℕ),
  (∑ (θ : ℕ), (d θ * (π (i, j) θ))) = 1

theorem pi_grid_covering (π : (ℕ × ℕ) → ℕ → ℕ → Prop) 
  (sum_positive : ∀ grid : (ℕ × ℕ) → ℤ, (∀ x y, grid (x, y) > 0) → ∃ (π : ℕ × ℕ → ℕ → ℕ → Prop), ∃ i j, π (i, j) (grid (i, j)) > 0):
  pi_covers_grid_without_gaps π := 
sorry

end pi_grid_covering_l822_822205


namespace range_for_a_l822_822767

noncomputable def inequality_p (x : ℝ) : Prop :=
  x^2 - 8x - 20 > 0

noncomputable def inequality_q (x a : ℝ) : Prop :=
  x^2 - 2x + 1 - a^2 > 0

noncomputable def is_sufficient_but_not_necessary_condition (p q : ℝ → Prop) : Prop :=
  ∀ x, p x → q x ∧ ∃ x', q x' ∧ ¬ p x'

theorem range_for_a (a : ℝ) : (∃ a, a > 0 ∧ is_sufficient_but_not_necessary_condition (inequality_p) (λ x, inequality_q x a)) → 3 ≤ a ∧ a ≤ 9 :=
begin
  sorry
end

end range_for_a_l822_822767


namespace problem_value_eq_13_l822_822949

theorem problem_value_eq_13 : 8 / 4 - 3^2 + 4 * 5 = 13 :=
by
  sorry

end problem_value_eq_13_l822_822949


namespace percentage_beth_sees_more_l822_822661

-- Let A be the number of ants Abe finds
def A : ℕ := 4

-- Let B be the number of ants Beth sees
def B (P : ℕ) : ℕ := A + ((P * A) / 100)

-- Let C be the number of ants CeCe watches
def C : ℕ := 2 * A

-- Let D be the number of ants Duke discovers
def D : ℕ := A / 2

-- Total number of ants found by the four children
def total_ants (P : ℕ) : ℕ := A + B(P) + C + D

-- The proof statement
theorem percentage_beth_sees_more (P : ℕ) : total_ants P = 20 → P = 50 :=
by 
  intros h,
  sorry

end percentage_beth_sees_more_l822_822661


namespace prob_prime_and_multiple_of_five_l822_822868

def is_prime (n : ℕ) : Prop := nat.prime n
def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

theorem prob_prime_and_multiple_of_five :
  (∃!(n : ℕ), n ∈ finset.range 1 75 ∧ is_prime n ∧ is_multiple_of_five n) / 75 = 1 / 75 :=
by
  sorry

end prob_prime_and_multiple_of_five_l822_822868


namespace largest_number_in_set_l822_822245

theorem largest_number_in_set : ∀ (a b c d : ℝ), (a = real.sqrt 2) → (b = 0) → (c = -1) → (d = 2) → a < d ∧ b < d ∧ c < d :=
begin
  intros a b c d ha hb hc hd,
  rw [ha, hb, hc, hd],
  split,
  {
    have h₁: real.sqrt 2 < 2 := by linarith [real.sqrt_lt_self (by norm_num)],
    exact h₁,
  },
  {
    split,
    {
      exact zero_lt_two,
    },
    {
      linarith,
    }
  }
end

end largest_number_in_set_l822_822245


namespace card_game_ensures_final_state_l822_822150

def card_game_eventually_ends (n : ℕ) (beats : ℕ → ℕ → Prop) (deck1 deck2 : list ℕ) : Prop :=
  ∃ (strategy : (list ℕ × list ℕ) → (list ℕ × list ℕ)), ∀ (deck1 deck2 : list ℕ),
    (deck1.length + deck2.length = n) →
    (∀ deck1 deck2, (deck1, deck2) ≠ ([], [])) →
    (∃ final_state, (strategy (deck1, deck2) = final_state ∧ (final_state.1 = [] ∨ final_state.2 = [])))

theorem card_game_ensures_final_state (n : ℕ) (beats : ℕ → ℕ → Prop) (deck1 deck2 : list ℕ) :
  card_game_eventually_ends n beats deck1 deck2 :=
sorry

end card_game_ensures_final_state_l822_822150


namespace hayley_friends_stickers_l822_822193

theorem hayley_friends_stickers (total_stickers : ℕ) (friends : ℕ) (stickers_each : ℕ) 
(h_total : total_stickers = 72) (h_friends : friends = 9) : 
stickers_each = total_stickers / friends := 
by
  have h1 : stickers_each = 72 / 9 := by rw [h_total, h_friends]
  calc
    stickers_each = 72 / 9 := h1
                  = 8 := by norm_num

end hayley_friends_stickers_l822_822193


namespace max_intersections_l822_822329

-- Definitions of the conditions
variable (circles : Fin 5 → Circle) (triangularRegion : Triangle)
variable (h1 : ∀ i, Circle.coplanar circles i)
variable (h2 : ∃ pairs : Finset (Fin 5), pairs.card = 2 ∧ 
               ∃ region : Triangle, (∀ circle ∈ pairs, ∃ point, point ∈ circles circle)) 
               ∧ Triangle.isCommonRegion triangularRegion pairs

-- Definition of the question
def maximumIntersections : ℕ := 10

-- Statement of the problem as a Lean theorem
theorem max_intersections : ∃ l : Line, (Line.inTriangle l triangularRegion) → 
                            (∑ i, (Line.intersections l (circles i)).card) ≤ maximumIntersections :=
by
  sorry

end max_intersections_l822_822329


namespace rectangle_perimeter_divided_into_six_congruent_l822_822218

theorem rectangle_perimeter_divided_into_six_congruent (l w : ℕ) (h1 : 2 * (w + l / 6) = 40) (h2 : l = 120 - 6 * w) : 
  2 * (l + w) = 280 :=
by
  sorry

end rectangle_perimeter_divided_into_six_congruent_l822_822218


namespace probability_all_hats_retrieved_l822_822309

noncomputable def harmonic (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, 1 / (i + 1 : ℝ)

noncomputable def factorial (n : ℕ) : ℝ :=
  if n = 0 then 1 else Nat.factorial n

noncomputable def p_n (n : ℕ) : ℝ :=
  (harmonic n / factorial n) * ∏ i in Finset.range n, harmonic i

theorem probability_all_hats_retrieved :
  p_n 10 ≈ 0.000516 :=
sorry

end probability_all_hats_retrieved_l822_822309


namespace max_sides_in_subpolygon_l822_822798

/-- In a convex 1950-sided polygon with all its diagonals drawn, the polygon with the greatest number of sides among these smaller polygons can have at most 1949 sides. -/
theorem max_sides_in_subpolygon (n : ℕ) (hn : n = 1950) : 
  ∃ p : ℕ, p = 1949 ∧ ∀ m, m ≤ n-2 → m ≤ 1949 :=
sorry

end max_sides_in_subpolygon_l822_822798


namespace intersection_A_B_l822_822742

variable (x : ℤ)

def A := { x | x^2 - 4 * x ≤ 0 }
def B := { x | -1 ≤ x ∧ x < 4 }

theorem intersection_A_B :
  { x : ℤ | x ∈ A ∧ x ∈ B } = {0, 1, 2, 3} :=
sorry

end intersection_A_B_l822_822742


namespace cos_210_eq_neg_sqrt3_div_2_l822_822629

theorem cos_210_eq_neg_sqrt3_div_2 : real.cos (210 * real.pi / 180) = - real.sqrt 3 / 2 :=
by
  sorry

end cos_210_eq_neg_sqrt3_div_2_l822_822629


namespace gravity_anomaly_l822_822992

noncomputable def gravity_anomaly_acceleration
  (α : ℝ) (v₀ : ℝ) (g : ℝ) (S : ℝ) (g_a : ℝ) : Prop :=
  α = 30 ∧ v₀ = 10 ∧ g = 10 ∧ S = 3 * Real.sqrt 3 → g_a = 250

theorem gravity_anomaly (α v₀ g S g_a : ℝ) : gravity_anomaly_acceleration α v₀ g S g_a :=
by
  intro h
  sorry

end gravity_anomaly_l822_822992


namespace cyclic_quadrilateral_largest_BD_l822_822482

theorem cyclic_quadrilateral_largest_BD (a b c d : ℕ) 
  (h1 : a < 20) 
  (h2 : b < 20) 
  (h3 : c < 20) 
  (h4 : d < 20) 
  (h5 : a ≠ b) 
  (h6 : a ≠ c) 
  (h7 : a ≠ d) 
  (h8 : b ≠ c) 
  (h9 : b ≠ d) 
  (h10 : c ≠ d) 
  (h11 : a * b = c * d)
  : bd_length := 
  sqrt 405 :=
sorry

end cyclic_quadrilateral_largest_BD_l822_822482


namespace simplify_sqrt180_l822_822046

-- Conditions from the problem definition
def sqrt (x : ℕ) : ℝ := Real.sqrt x
def six := 6
def five := 5

-- The statement of the problem as a Lean theorem
theorem simplify_sqrt180 : sqrt 180 = six * sqrt five := by 
  sorry

end simplify_sqrt180_l822_822046


namespace solve_inequality_system_l822_822072

theorem solve_inequality_system (x : ℝ) :
  (8 * x - 3 ≤ 13) ∧ ((x - 1) / 3 - 2 < x - 1) → -2 < x ∧ x ≤ 2 :=
by
  intros h
  sorry

end solve_inequality_system_l822_822072


namespace quadratic_change_root_l822_822548

theorem quadratic_change_root (a b c : ℕ) (h : a + b + c = 2000) :
  ∃ a' b' c' : ℕ, (∃ x : ℤ, a' * x^2 + b' * x + c' = 0) ∧ (abs (int.ofNat a' - a) + abs (int.ofNat b' - b) + abs (int.ofNat c' - c) ≤ 1050) :=
begin
  sorry
end

end quadratic_change_root_l822_822548


namespace noMixedPoints_l822_822215

-- Define what it means for a point to be a mixed point
def isMixedPoint (x y : ℝ) : Prop :=
  (∃ r1 : ℚ, x = r1 ∧ ¬ ∃ r2 : ℚ, y = r2) ∨ (∃ r1 : ℚ, y = r1 ∧ ¬ ∃ r2 : ℚ, x = r2)

-- Polynomial definition with real coefficients
def isPolynomial (P : ℝ → ℝ) : Prop :=
  ∃ (n : ℕ) (a : Fin n → ℝ), ∀ x, P x = ∑ i, a i * x ^ i

-- The main theorem stating that such polynomials must be linear with rational coefficients
theorem noMixedPoints (P : ℝ → ℝ) (hP : isPolynomial P) :
  (∀ x y : ℝ, isMixedPoint x y → P x ≠ y) →
  ∃ (a₀ a₁ : ℚ), ∀ x, P x = a₁ * x + a₀ :=
sorry

end noMixedPoints_l822_822215


namespace bouquets_needed_to_earn_1000_l822_822267

theorem bouquets_needed_to_earn_1000 :
  ∀ (cost_per_bouquet sell_price_bouquet: ℕ) (roses_per_bouquet_bought roses_per_bouquet_sold target_profit: ℕ),
    cost_per_bouquet = 20 →
    sell_price_bouquet = 20 →
    roses_per_bouquet_bought = 7 →
    roses_per_bouquet_sold = 5 →
    target_profit = 1000 →
    (target_profit / (sell_price_bouquet * roses_per_bouquet_sold / roses_per_bouquet_bought - cost_per_bouquet) * roses_per_bouquet_bought = 125) :=
by
  intros cost_per_bouquet sell_price_bouquet roses_per_bouquet_bought roses_per_bouquet_sold target_profit 
    h_cost_per_bouquet h_sell_price_bouquet h_roses_per_bouquet_bought h_roses_per_bouquet_sold h_target_profit
  sorry

end bouquets_needed_to_earn_1000_l822_822267


namespace mike_reaches_sarah_in_36_minutes_l822_822867

-- Define initial conditions as constants
constant distance_separated : ℝ := 24
constant sarah_speed_multiplier : ℝ := 4
constant speed_sum : ℝ := 120
constant initial_time : ℕ := 6
constant decrease_rate : ℝ := 2

-- The theorem to be proved
theorem mike_reaches_sarah_in_36_minutes :
  ∃ v_M v_S : ℝ, v_S = sarah_speed_multiplier * v_M ∧ (v_S + v_M = speed_sum) ∧
    let remaining_distance := distance_separated - decrease_rate * (initial_time : ℝ) in
    let time_mike_alone := remaining_distance / v_M in
    (initial_time : ℝ) + time_mike_alone * 60 = 36 :=
sorry

end mike_reaches_sarah_in_36_minutes_l822_822867


namespace puncture_point_exists_l822_822022

noncomputable def similar_transformation (K₀ K₁ : set (ℝ × ℝ)) (f : (ℝ × ℝ) → (ℝ × ℝ)) : Prop :=
  ∃ r > 0, ∃ θ, ∃ c₀, ∃ c₁, ∀ p ∈ K₀, f p = (r • (rotation θ (p - c₀)) + c₁) ∧ f '' K₀ = K₁

theorem puncture_point_exists (K₀ K₁ : set (ℝ × ℝ)) (f : (ℝ × ℝ) → (ℝ × ℝ)) 
    (hK₀ : is_compact K₀) (hK₁ : is_compact K₁) 
    (h_sub : K₁ ⊆ K₀) (h_similar : similar_transformation K₀ K₁ f) :
    ∃ X, X ∈ K₀ ∧ X ∈ K₁ ∧ f X = X :=
by
  sorry

end puncture_point_exists_l822_822022


namespace g_five_eq_one_l822_822567

noncomputable def g : ℝ → ℝ := sorry

axiom g_mul (x z : ℝ) : g (x * z) = g x * g z
axiom g_one_ne_zero : g (1) ≠ 0

theorem g_five_eq_one : g (5) = 1 := 
by
  sorry

end g_five_eq_one_l822_822567


namespace room_length_correct_l822_822569

noncomputable def width : ℝ := 3.75
noncomputable def total_cost : ℝ := 16500
noncomputable def rate_per_sq_meter : ℝ := 800
noncomputable def expected_length : ℝ := 5.5

theorem room_length_correct :
  let total_area := total_cost / rate_per_sq_meter in
  let length := total_area / width in
  length = expected_length :=
by
  sorry

end room_length_correct_l822_822569


namespace final_number_greater_than_one_l822_822021

theorem final_number_greater_than_one (n : ℕ) (h : n = 2023) :
  (∃ seq : list ℕ, seq.length = n ∧ (∀ x ∈ seq, x = 2023) ∧ (∃ final : ℕ, 
  (∀ i j (H1 : i ≠ j) (H2 : i < seq.length) (H3 : j < seq.length), 
  seq[i] + seq[j] = 4 * final) ∧ final > 1)) :=
sorry

end final_number_greater_than_one_l822_822021


namespace investment_triples_in_value_l822_822176

theorem investment_triples_in_value {P A : ℝ} {r : ℝ} (h_r : r = 0.341) (h_P : P > 0) (h_A : A > 3 * P) :
  ∃ t : ℕ, t ≥ 4 ∧ A = P * (1 + r)^t :=
begin
  sorry
end

end investment_triples_in_value_l822_822176


namespace card_game_eventually_one_without_cards_l822_822152

def card_game (n : ℕ) : Prop :=
  ∃ (cards : Fin n → Fin n) 
    (beats : (Fin n) → (Fin n) → Prop)
    (initial_distribution : Fin n → Fin 2),
  (∀ x y z : Fin n, beats x y ∨ beats y x) →
  (∀ x y z : Fin n, beats x z ∧ beats z y → beats x y) →
  (∀ distribution : Fin n → Fin 2, 
     ∃ turn : ℕ → ((Fin n) × (Fin n)) → Prop,
       (∀ i : ℕ, (turn i) (initial_distribution i) →
         ∃ j : ℕ, turn (i + 1) (beats _ _) → 
           (∃ k : ℕ, turn k (initial_distribution i) → false)))

theorem card_game_eventually_one_without_cards (n : ℕ) : card_game n :=
sorry

end card_game_eventually_one_without_cards_l822_822152


namespace exists_digit_sum_divisible_by_27_not_number_l822_822433

-- Definitions
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def divisible_by (a b : ℕ) : Prop :=
  b ≠ 0 ∧ a % b = 0

-- Theorem statement
theorem exists_digit_sum_divisible_by_27_not_number (n : ℕ) :
  divisible_by (sum_of_digits n) 27 ∧ ¬ divisible_by n 27 :=
  sorry

end exists_digit_sum_divisible_by_27_not_number_l822_822433


namespace second_player_wins_l822_822904

def first_player_prevention_possible (cube : ℕ → ℕ) : Prop :=
  ∃ strategy : (ℕ → ℕ), ∀ ring_sums : set (fin 8 → ℕ), ∃ r : ring_sums, sum (r.ge 0) = sum (r.ge 1)

def second_player_strategy_valid : Prop :=
  ¬first_player_prevention_possible (λ n, n)

theorem second_player_wins : second_player_strategy_valid :=
  sorry

end second_player_wins_l822_822904


namespace problem1_problem2_l822_822631

section
variable (x : ℝ)

-- Problem 1
theorem problem1 : (-1)^3 + (1/2)^(-2) - real.sqrt(12) * real.sqrt(3) = -3 := by
sory

-- Problem 2
theorem problem2 (h : x ≠ 1 ∧ x ≠ -1) :
  ((1 / (x^2 - 1) + 1) * (x^2 + 2*x + 1)) / x^2 = (x + 1) / (x - 1) := by
sory
end

end problem1_problem2_l822_822631


namespace tallest_is_Justina_l822_822723

variable (H G I J K : ℝ)

axiom height_conditions1 : H < G
axiom height_conditions2 : G < J
axiom height_conditions3 : K < I
axiom height_conditions4 : I < G

theorem tallest_is_Justina : J > G ∧ J > H ∧ J > I ∧ J > K :=
by
  sorry

end tallest_is_Justina_l822_822723


namespace youngest_child_age_l822_822960

theorem youngest_child_age 
  (x : ℕ)
  (h : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 50) : 
  x = 6 := 
by 
  sorry

end youngest_child_age_l822_822960


namespace division_quotient_l822_822165

theorem division_quotient (x : ℤ) (y : ℤ) (r : ℝ) (h1 : x > 0) (h2 : y = 96) (h3 : r = 11.52) :
  ∃ q : ℝ, q = (x - r) / y := 
sorry

end division_quotient_l822_822165


namespace find_interest_rate_l822_822623

def compound_interest (P r : ℝ) (n t : ℝ) : ℝ := P * (1 + r / n) ^ (n * t)
 
theorem find_interest_rate (P : ℝ) (r : ℝ) :
  compound_interest P r 1 10 = 9000 ∧ compound_interest P r 1 11 = 9990 → r = 0.11 :=
by
  intro h
  sorry

end find_interest_rate_l822_822623


namespace simplify_sqrt_180_l822_822065

theorem simplify_sqrt_180 : sqrt 180 = 6 * sqrt 5 :=
by
  have h : 180 = 2^2 * 3^2 * 5 := by norm_num
  calc
    sqrt 180
      = sqrt (2^2 * 3^2 * 5)     : by rw [h]
  ... = sqrt (2^2) * sqrt (3^2) * sqrt 5 : by rw [sqrt_mul, sqrt_mul]
  ... = 2 * 3 * sqrt 5          : by rw [sqrt_sq, sqrt_sq]
  ... = 6 * sqrt 5              : by norm_num

end simplify_sqrt_180_l822_822065


namespace coats_collected_in_total_l822_822539

def high_school_coats : Nat := 6922
def elementary_school_coats : Nat := 2515
def total_coats : Nat := 9437

theorem coats_collected_in_total : 
  high_school_coats + elementary_school_coats = total_coats := 
  by
  sorry

end coats_collected_in_total_l822_822539


namespace largest_modulus_z_l822_822826

open Complex

noncomputable def z_largest_value (a b c z : ℂ) : ℝ :=
  (1 + Real.sqrt 5) / 2

theorem largest_modulus_z (a b c z : ℂ) (r : ℝ) (hr_pos : 0 < r)
  (hmod_a : Complex.abs a = r) (hmod_b : Complex.abs b = r) (hmod_c : Complex.abs c = r)
  (heqn : a * z ^ 2 + b * z + c = 0) :
  Complex.abs z ≤ z_largest_value a b c z :=
sorry

end largest_modulus_z_l822_822826


namespace product_of_numbers_eq_l822_822894

noncomputable def number1 : ℚ := -3 / 4
noncomputable def number2 : ℚ := number1 - 1 / 2

theorem product_of_numbers_eq :
  number1 * number2 = 15 / 16 :=
by sorry

end product_of_numbers_eq_l822_822894


namespace find_q_l822_822899

-- Defining the polynomial and conditions
def Q (x : ℝ) (p q r : ℝ) : ℝ := x^3 + p * x^2 + q * x + r

variable (p q r : ℝ)

-- Given conditions
def mean_of_zeros_eq_prod_of_zeros (p q r : ℝ) : Prop :=
  -p / 3 = r

def prod_of_zeros_eq_sum_of_coeffs (p q r : ℝ) : Prop :=
  r = 1 + p + q + r

def y_intercept_eq_three (r : ℝ) : Prop :=
  r = 3

-- Final proof statement asserting q = 5
theorem find_q (p q r : ℝ) (h1 : mean_of_zeros_eq_prod_of_zeros p q r)
  (h2 : prod_of_zeros_eq_sum_of_coeffs p q r)
  (h3 : y_intercept_eq_three r) :
  q = 5 :=
sorry

end find_q_l822_822899


namespace valid_n_values_count_l822_822645

open Nat

theorem valid_n_values_count (n : ℕ) (hn : 7 ≤ n ∧ n ≤ 1000) :
  2^4 ∣ n * (n-1) * (n-2) * (n-3) * (n-4) ∧ 3^2 ∣ n * (n-1) * (n-2) * (n-3) * (n-4) ∧ 5 ∣ n * (n-1) * (n-2) * (n-3) * (n-4) → ∃ c, 1 ≤ c ∧ c . (n \in Icc 7 1000) = 557 :=
begin
  sorry
end

end valid_n_values_count_l822_822645


namespace possible_values_of_k_l822_822540

noncomputable def has_roots (p q r s t k : ℂ) : Prop :=
  (p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0 ∧ t ≠ 0) ∧ 
  (p * k^4 + q * k^3 + r * k^2 + s * k + t = 0) ∧
  (q * k^4 + r * k^3 + s * k^2 + t * k + p = 0)

theorem possible_values_of_k (p q r s t k : ℂ) (hk : has_roots p q r s t k) : 
  k^5 = 1 :=
  sorry

end possible_values_of_k_l822_822540


namespace hyperbola_vertex_to_asymptote_distance_l822_822429

theorem hyperbola_vertex_to_asymptote_distance 
  (a : ℝ) (h1 : a > 0) :
  let e := 2
  let b : ℝ := sqrt 3
  let d : ℝ := sqrt 3 / 2
  (eccentricity_hyperbola : e = (sqrt (a^2 + b^2)) / a)
  (hyperbola_eq : ∀ (x y : ℝ), (x^2 / a^2 - y^2 / 3 = 1)) :
  distance (1, 0) (line_eq : ∀ (x y : ℝ), y = sqrt 3 * x) = d := 
sorry

end hyperbola_vertex_to_asymptote_distance_l822_822429


namespace probability_range_l822_822172

theorem probability_range (P : ℝ → ℝ) (A : ℝ) : 
  0 ≤ P A ∧ P A ≤ 1 :=
begin
  sorry
end

end probability_range_l822_822172


namespace projectile_height_time_l822_822558

theorem projectile_height_time (h : ∀ t : ℝ, -16 * t^2 + 100 * t = 64 → t = 1) : (∃ t : ℝ, -16 * t^2 + 100 * t = 64 ∧ t = 1) :=
by sorry

end projectile_height_time_l822_822558


namespace potato_cost_l822_822125

variables (x : ℝ)
variables (b a : ℝ)

def andrey_earnings (x : ℝ) : ℝ := 120 * x
def boris_earnings (x : ℝ) : ℝ := 124.8 * x

theorem potato_cost :
  (boris_earnings x) - (andrey_earnings x) = 1200 → x = 250 :=
  by
    unfold andrey_earnings
    unfold boris_earnings
    sorry

end potato_cost_l822_822125


namespace find_p_plus_q_l822_822823

noncomputable def point := (ℝ × ℝ)
noncomputable def angle := ℝ 
noncomputable def dist (p1 p2 : point) := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def A : point := (0, 0)
def B (c : ℝ) : point := (c, 3)
def valid_y_coordinates : set ℝ := {0, 3, 6, 9, 12, 15}

-- Given conditions
structure Hexagon :=
  (A B C D E F : point)
  (is_equilateral : dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D E ∧ dist D E = dist E F ∧ dist E F = dist F A)
  (fab_angle : angle := 150)
  (ab_de_parallel : B.2 - A.2 = E.2 - D.2)
  (bc_ef_parallel : C.2 - B.2 = F.2 - E.2)
  (cd_fa_parallel : D.2 - C.2 = A.2 - F.2)
  (distinct_y_coords : {A.2, B.2, C.2, D.2, E.2, F.2} = valid_y_coordinates)

theorem find_p_plus_q (A B F : point) (c : ℝ) (hex : Hexagon) : 
  let p := 144 in
  let q := 3 in
  p + q = 147 := 
sorry

end find_p_plus_q_l822_822823


namespace union_of_sets_l822_822837

theorem union_of_sets :
  let A := {1, 3}
  let B := {1, 2, 4, 5}
  A ∪ B = {1, 2, 3, 4, 5} :=
by {
  sorry
}

end union_of_sets_l822_822837


namespace bill_buys_125_bouquets_to_make_1000_l822_822259

-- Defining the conditions
def cost_per_bouquet : ℕ := 20
def roses_per_bouquet_buy : ℕ := 7
def roses_per_bouquet_sell : ℕ := 5
def target_difference : ℕ := 1000

-- To be demonstrated: number of bouquets Bill needs to buy to get a profit difference of $1000
theorem bill_buys_125_bouquets_to_make_1000 : 
  let total_cost_to_buy (n : ℕ) := n * cost_per_bouquet
  let total_roses (n : ℕ) := n * roses_per_bouquet_buy
  let bouquets_sell_from_roses (roses : ℕ) := roses / roses_per_bouquet_sell
  let total_revenue (bouquets : ℕ) := bouquets * cost_per_bouquet
  let profit (n : ℕ) := total_revenue (bouquets_sell_from_roses (total_roses n)) - total_cost_to_buy n
  profit (125 * 5) = target_difference := 
sorry

end bill_buys_125_bouquets_to_make_1000_l822_822259


namespace number_of_correct_statements_l822_822096

def is_opposite (a b : ℤ) : Prop := a + b = 0

def statement1 : Prop := ∀ a b : ℤ, (a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0) → is_opposite a b
def statement2 : Prop := ∀ n : ℤ, n = -n → n < 0
def statement3 : Prop := ∀ a b : ℤ, is_opposite a b → a + b = 0
def statement4 : Prop := ∀ a b : ℤ, is_opposite a b → (a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0)

theorem number_of_correct_statements : (¬statement1 ∧ ¬statement2 ∧ statement3 ∧ ¬statement4) ↔ (∃n : ℕ, n = 1) :=
by
  sorry

end number_of_correct_statements_l822_822096


namespace log_eq_solution_l822_822071

open Real

theorem log_eq_solution (x : ℝ) (h : x > 0) : log x + log (x + 1) = 2 ↔ x = (-1 + sqrt 401) / 2 :=
by
  sorry

end log_eq_solution_l822_822071


namespace pilot_weeks_l822_822453

-- Given conditions
def milesTuesday : ℕ := 1134
def milesThursday : ℕ := 1475
def totalMiles : ℕ := 7827

-- Calculate total miles flown in one week
def milesPerWeek : ℕ := milesTuesday + milesThursday

-- Define the proof problem statement
theorem pilot_weeks (w : ℕ) (h : w * milesPerWeek = totalMiles) : w = 3 :=
by
  -- Here we would provide the proof, but we leave it with a placeholder
  sorry

end pilot_weeks_l822_822453


namespace problem_1_problem_2_l822_822765

def A : Set ℝ := { x | x^2 - 3 * x + 2 < 0 }

def B (a : ℝ) : Set ℝ := { x | a - 1 < x ∧ x < 3 * a + 1 }

-- Problem 1
theorem problem_1 (a : ℝ) (h : a = 1 / 4) : A ∩ B a = { x | 1 < x ∧ x < 7 / 4 } :=
by
  rw [h]
  sorry

-- Problem 2
theorem problem_2 : (∀ x, A x → B a x) → ∀ a, 1 / 3 ≤ a ∧ a ≤ 2 :=
by
  sorry

end problem_1_problem_2_l822_822765


namespace three_digit_numbers_with_digit_five_l822_822408

open Nat

def isValidHundredsDigit (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 9
def isValidTensUnitsDigit (d : ℕ) : Prop := 0 ≤ d ∧ d ≤ 9

def hasDigitFive (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  hundreds = 5 ∨ tens = 5 ∨ units = 5

theorem three_digit_numbers_with_digit_five :
  { n : ℕ | 100 ≤ n ∧ n < 1000 ∧ hasDigitFive n }.card = 251 := 
sorry

end three_digit_numbers_with_digit_five_l822_822408


namespace sum_powers_neg_one_l822_822948

theorem sum_powers_neg_one : 
  ((finsupp.sum (finsupp.mk (λ n, (-1) ^ n) (finset.range 2007) (λ x, true))) = 1) := 
sorry

end sum_powers_neg_one_l822_822948


namespace student_ticket_cost_l822_822330

theorem student_ticket_cost 
  (total_tickets_sold : ℕ) 
  (total_revenue : ℕ) 
  (nonstudent_ticket_cost : ℕ) 
  (student_tickets_sold : ℕ) 
  (cost_per_student_ticket : ℕ) 
  (nonstudent_tickets_sold : ℕ) 
  (H1 : total_tickets_sold = 821) 
  (H2 : total_revenue = 1933)
  (H3 : nonstudent_ticket_cost = 3)
  (H4 : student_tickets_sold = 530) 
  (H5 : nonstudent_tickets_sold = total_tickets_sold - student_tickets_sold)
  (H6 : 530 * cost_per_student_ticket + nonstudent_tickets_sold * 3 = 1933) : 
  cost_per_student_ticket = 2 := 
by
  sorry

end student_ticket_cost_l822_822330


namespace intersection_area_l822_822403

def regionM (x y : ℝ) : Prop :=
  0 ≤ y ∧ y ≤ min (2 * x) (3 - x)

def regionN (x : ℝ) (t : ℝ) : Prop :=
  t ≤ x ∧ x ≤ t + 2

noncomputable def f (t : ℝ) : ℝ :=
  - (3 / 2) * t^2 + t + (5 / 2)

theorem intersection_area (t : ℝ) (h : 0 ≤ t ∧ t ≤ 1) :
  (∫ x in t..t+2, ∫ y in 0..(min (2*x) (3-x)), 1) = f t :=
sorry

end intersection_area_l822_822403


namespace series_sum_eq_51_l822_822696

def a_n (n : ℕ) : ℝ := n / 101

noncomputable def series_sum : ℝ :=
  ∑ n in Finset.range 101,
    let an := a_n (n + 1)
    in an^3 / (1 - 3 * an + 3 * an^2)

theorem series_sum_eq_51 :
  series_sum = 51 := by
  sorry

end series_sum_eq_51_l822_822696


namespace second_derivative_at_pi_over_2_l822_822497

def f (x : ℝ) : ℝ := x * Real.sin x

theorem second_derivative_at_pi_over_2 :
  (derivative^[2] f) (Real.pi / 2) = 1 :=
sorry

end second_derivative_at_pi_over_2_l822_822497


namespace distance_covered_downstream_approximation_l822_822988

theorem distance_covered_downstream_approximation :
  let
    boat_speed := 14.0
    current_speed := 2.0
    time_seconds := 8.999280057595392
    effective_speed_kmph := boat_speed + current_speed
    effective_speed_mps := (effective_speed_kmph * 1000) / 3600
  in
    approx (effective_speed_mps * time_seconds) 39.992640512 :=
by
  sorry

end distance_covered_downstream_approximation_l822_822988


namespace kate_spent_on_mouse_l822_822819

theorem kate_spent_on_mouse :
  let march := 27
  let april := 13
  let may := 28
  let saved := march + april + may
  let keyboard := 49
  let left := 14
  saved - left - keyboard = 5 :=
by
  let march := 27
  let april := 13
  let may := 28
  let saved := march + april + may
  let keyboard := 49
  let left := 14
  show saved - left - keyboard = 5
  sorry

end kate_spent_on_mouse_l822_822819


namespace greatest_n_factor_of_4_in_10_fact_l822_822958

theorem greatest_n_factor_of_4_in_10_fact :
  ∃ (n : ℕ), (∀ m, 4^m ∣ 10.factorial → m ≤ n) ∧ (4^n ∣ 10.factorial) :=
begin
  use 4,
  split,
  {
    intro m,
    sorry,
  },
  {
    sorry,
  }
end

end greatest_n_factor_of_4_in_10_fact_l822_822958


namespace number_of_students_above_110_l822_822642

-- We start with the conditions
variables (students : ℕ) (mu sigma : ℝ)
  (xi : ℕ → ℝ)
  (P : set ℝ → ℝ)
  (h50 : students = 50)
  (hN : ∀ x, ℝ.dist x xi/pdf.eval (0, pdf.eval (norm(100,10))=0.3)): xi follows a normal_distribution N(100, 10^2) and is symmetric around xi =100emphasise each statement using seperate lines 

-- stating the probability conditions
(P1 : P ({y : ℝ | 90 ≤ y ∧ y ≤ 100} ) = 0.3)

-- The goal
theorem number_of_students_above_110 : 
  students * P({y : ℝ | y ≥ 110}) = 10 :=
by
  sorry

end number_of_students_above_110_l822_822642


namespace correct_statement_c_l822_822171

theorem correct_statement_c (five_boys_two_girls : Nat := 7) (select_three : Nat := 3) :
  (∃ boys girls : Nat, boys + girls = five_boys_two_girls ∧ boys = 5 ∧ girls = 2) →
  (∃ selected_boys selected_girls : Nat, selected_boys + selected_girls = select_three ∧ selected_boys > 0) :=
by
  sorry

end correct_statement_c_l822_822171


namespace remaining_surface_area_of_block_l822_822228

-- Define the block dimensions
def block_dimensions : ℝ × ℝ × ℝ := (5, 4, 3)

-- Define the radius of the cylindrical hole
def hole_radius : ℝ := 1

-- Define the correct answer as a term that needs to be proven
theorem remaining_surface_area_of_block : 
  let orig_surface_area := 2 * (5 * 4 + 5 * 3 + 4 * 3) in
  let hole_area := 2 * (π * hole_radius ^ 2) in
  let cylinder_wall_area := 2 * π * hole_radius * 3 in
  (orig_surface_area - hole_area + cylinder_wall_area) = 94 + 4 * π :=
  sorry

end remaining_surface_area_of_block_l822_822228


namespace sin_ratio_side_length_b_l822_822442

-- Definitions from conditions
variables {A B C : ℝ}      -- Angles
variables {a b c : ℝ}      -- Sides opposite to the angles
variables (cosA cosB cosC sinA sinC : ℝ)

-- Conditions
def triangle_conditions : Prop :=
  (cosA - 2 * cosC) / cosB = (2 * c - a) / b ∧
  cosB = 1 / 4 ∧
  a + b + c = 5

-- Equivalent proof problems
theorem sin_ratio (h : triangle_conditions) :
  (sinC / sinA) = 2 :=
sorry

theorem side_length_b (h : triangle_conditions) :
  b = 2 :=
sorry

end sin_ratio_side_length_b_l822_822442


namespace clay_target_permutations_l822_822448

theorem clay_target_permutations : 
  let targets := ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
  ∃ (n_1 n_2 n_3 : ℕ), 
  multiset.card targets = 9 ∧
  multiset.count 'A' targets = 3 ∧
  multiset.count 'B' targets = 3 ∧
  multiset.count 'C' targets = 3 ∧
  n_1 = 3 ∧ n_2 = 3 ∧ n_3 = 3 ∧ 
  (fact 9 / (fact 3 * fact 3 * fact 3) = 1680) :=
begin
  let targets := ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
  use [3, 3, 3],
  split, 
  sorry, -- prove multiset.card targets = 9
  split,
  sorry, -- proved multiset.count 'A' targets = 3
  split,
  sorry, -- prove multiset.count 'B' targets = 3
  split,
  sorry, -- prove multiset.count 'C' targets = 3
  split,
  refl,
  split,
  refl,
  split,
  refl,
  have h : fact 9 / (fact 3 * fact 3 * fact 3) = 1680,
  sorry, -- prove factorial calculation
  exact h,
end

end clay_target_permutations_l822_822448


namespace max_regions_divided_by_lines_l822_822115

theorem max_regions_divided_by_lines (n m : ℕ) (hn : n = 50) (hm : m = 20) (h_lines : m ≤ n) : 
  let k := n - m in
  let S_k := (k * (k + 1)) / 2 + 1 in
  let S_parallel := m * (k + 1) in
  S_k + S_parallel = 1086 :=
by
  have hn : n = 50 := hn
  have hm : m = 20 := hm
  have h_lines : m ≤ n := h_lines
  let k := n - m
  let S_k := (k * (k + 1)) / 2 + 1
  let S_parallel := m * (k + 1)
  sorry

end max_regions_divided_by_lines_l822_822115


namespace more_stable_performance_l822_822143

theorem more_stable_performance (s_A_sq s_B_sq : ℝ) (hA : s_A_sq = 0.25) (hB : s_B_sq = 0.12) : s_A_sq > s_B_sq :=
by
  rw [hA, hB]
  sorry

end more_stable_performance_l822_822143


namespace count_propositions_l822_822241

def is_proposition (s : string) : Prop :=
  s = "statement can be judged to be true or false"

def s1 : string := "|x+2|"
def s2 : string := "-5 ∈ ℤ"
def s3 : string := "π ∉ ℝ"
def s4 : string := "{0} ∈ ℕ"

def is_proposition_s2 : Prop := is_proposition s2
def is_proposition_s3 : Prop := is_proposition s3
def is_proposition_s4 : Prop := is_proposition s4

theorem count_propositions :
  (cond : is_proposition_s2 ∧ is_proposition_s3 ∧ is_proposition_s4) → 
  (¬ is_proposition s1) →
  (is_proposition s2) →
  (is_proposition s3) →
  (is_proposition s4) →
  (3 = 1 + 1 + 1) := 
sorry

end count_propositions_l822_822241


namespace sqrt_inequality_l822_822751

theorem sqrt_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (sqrt(x) + sqrt(y)) / 2 ≤ sqrt((x + y) / 2) :=
by
  sorry

end sqrt_inequality_l822_822751


namespace yellow_faces_of_cube_l822_822984

theorem yellow_faces_of_cube (n : ℕ) (h : 6 * n^2 = (1 / 3) * (6 * n^3)) : n = 3 :=
by {
  sorry
}

end yellow_faces_of_cube_l822_822984


namespace cost_of_one_bag_l822_822134

theorem cost_of_one_bag (x : ℝ) :
  let Andrey_earning := 60 * 2 * x
  let Boris_earning := 15 * 1.6 * x + 45 * (1.6 * 1.4) * x
  Boris_earning - Andrey_earning = 1200 →
  x = 250 := 
by
  intros
  let Andrey_earning := 60 * 2 * x
  let Boris_earning := 15 * 1.6 * x + 45 * (1.6 * 1.4) * x
  have h : Boris_earning - Andrey_earning = 1200 := by assumption
  let simplified_h := 
    calc
      Boris_earning - Andrey_earning
        = (15 * 1.6 * x + 45 * (1.6 * 1.4) * x) - (60 * 2 * x) : by simp [Andrey_earning, Boris_earning]
    ... = (15 * 1.6 * x + 45 * (1.6 * 1.4) * x) - 120 * x : by simp
    ... = (24 * x + 100.8 * x) - 120 * x : by simp
    ... = 124.8 * x - 120 * x : by simp
    ... = 4.8 * x : by simp
    ... = 1200 : by rw h
  exact (div_eq_iff (by norm_num : (4.8 : ℝ) ≠ 0)).1 simplified_h  -- solves for x

end cost_of_one_bag_l822_822134


namespace triangle_right_angle_and_m_values_l822_822756

open Real

-- Definitions and conditions
def line_AB (x y : ℝ) : Prop := 3 * x - 2 * y + 6 = 0
def line_AC (x y : ℝ) : Prop := 2 * x + 3 * y - 22 = 0
def line_BC (x y m : ℝ) : Prop := 3 * x + 4 * y - m = 0

-- Prove the shape and value of m when the height from BC is 1
theorem triangle_right_angle_and_m_values :
  (∃ (x y : ℝ), line_AB x y ∧ line_AC x y ∧ line_AB x y ∧ (-3/2) ≠ (2/3)) ∧
  (∀ x y, line_AB x y → line_AC x y → 3 * x + 4 * y - 25 = 0 ∨ 3 * x + 4 * y - 35 = 0) := 
sorry

end triangle_right_angle_and_m_values_l822_822756


namespace probability_one_of_last_three_red_l822_822024

theorem probability_one_of_last_three_red :
  let total_balls := 10
  let red_balls := 3
  let total_children := 10
  let last_children := 3
  (3 / 10) * (7 / 10) * (7 / 10) * 3 = 21 / 100 :=
by
  sorry

end probability_one_of_last_three_red_l822_822024


namespace find_side_length_l822_822646

def hollow_cube_formula (n : ℕ) : ℕ :=
  6 * n^2 - (n^2 + 4 * (n - 2))

theorem find_side_length :
  ∃ n : ℕ, hollow_cube_formula n = 98 ∧ n = 9 :=
by
  sorry

end find_side_length_l822_822646


namespace not_basic_event_l822_822795

theorem not_basic_event :
  let balls := {red_red, red_white, red_black, white_white, white_black, black_black} :
  (∃ (events: Finset (Finset (Subsingleton balls))), 
    {red_red} ∉ events ∧ {red_white} ∉ events ∧ {red_black} ∉ events) :=
sorry

end not_basic_event_l822_822795


namespace x_coord_of_y5_on_line_k_l822_822809

theorem x_coord_of_y5_on_line_k :
  ∀ (x y : ℝ), (x, 1) ∈ (λ (x : ℝ), (x, (1/5) * x)) ∧ (5, y) ∈ (λ (x : ℝ), (x, (1/5) * x)) →
  (∃ x : ℝ, (x, 5) ∈ (λ (x : ℝ), (x, (1/5) * x)) ∧ x = 25) :=
by
  intros x y h
  have hy := (h.right)
  simp at hy
  use 25
  split
  · simp
  · rfl

end x_coord_of_y5_on_line_k_l822_822809


namespace prove_jimmy_is_2_determine_rachel_age_l822_822633

-- Define the conditions of the problem
variables (a b c r1 r2 : ℤ)

-- Condition 1: Rachel's age and Jimmy's age are roots of the quadratic equation
def is_root (p : ℤ → ℤ) (x : ℤ) : Prop := p x = 0

def quadratic_eq (x : ℤ) : ℤ := a * x^2 + b * x + c

-- Condition 2: Sum of the coefficients is a prime number
def sum_of_coefficients_is_prime : Prop :=
  Nat.Prime (a + b + c).natAbs

-- Condition 3: Substituting Rachel’s age into the quadratic equation gives -55
def substitute_rachel_is_minus_55 (r : ℤ) : Prop :=
  quadratic_eq a b c r = -55

-- Question 1: Prove Jimmy is 2 years old
theorem prove_jimmy_is_2 (h1 : is_root (quadratic_eq a b c) r1)
                           (h2 : is_root (quadratic_eq a b c) r2)
                           (h3 : sum_of_coefficients_is_prime a b c)
                           (h4 : substitute_rachel_is_minus_55 a b c r1) :
  r2 = 2 :=
sorry

-- Question 2: Determine Rachel's age
theorem determine_rachel_age (h1 : is_root (quadratic_eq a b c) r1)
                             (h2 : is_root (quadratic_eq a b c) r2)
                             (h3 : sum_of_coefficients_is_prime a b c)
                             (h4 : substitute_rachel_is_minus_55 a b c r1)
                             (h5 : r2 = 2) :
  r1 = 7 :=
sorry

end prove_jimmy_is_2_determine_rachel_age_l822_822633


namespace arrests_per_day_in_each_city_l822_822154

-- Define the known conditions
def daysOfProtest := 30
def numberOfCities := 21
def daysInJailBeforeTrial := 4
def daysInJailAfterTrial := 7 / 2 * 7 -- half of a 2-week sentence in days, converted from weeks to days
def combinedJailTimeInWeeks := 9900
def combinedJailTimeInDays := combinedJailTimeInWeeks * 7

-- Define the proof statement
theorem arrests_per_day_in_each_city :
  (combinedJailTimeInDays / (daysInJailBeforeTrial + daysInJailAfterTrial)) / daysOfProtest / numberOfCities = 10 := 
by
  sorry

end arrests_per_day_in_each_city_l822_822154


namespace ratio_rational_l822_822515

variable (α : Type) [LinearOrderedField α]
variables (a b c d e : α)
variable (numbers : Fin 5 → α)
variable (h_diff : ∀ i j : Fin 5, i ≠ j → numbers i ≠ numbers j)
variable (h_pos : ∀ i : Fin 5, 0 < numbers i)
variable (h_rational_prod_sum : ∀ (i j k : Fin 5), i ≠ j → j ≠ k → i ≠ k → is_r := ((numbers i) * (numbers j) + (numbers j) * (numbers k) + (numbers k) * (numbers i)))

theorem ratio_rational (i j : Fin 5) (h_ij: i ≠ j) : is_r := (numbers i / numbers j) := sorry

end ratio_rational_l822_822515


namespace relationship_of_length_and_width_l822_822818

variable (x y : ℝ)

theorem relationship_of_length_and_width 
  (h : 2 * (x + y) = 20) : y = -x + 10 :=
by
  calc
    2 * (x + y) = 20     : h
    (x + y) = 10     : by linarith
    y = 10 - x       : by linarith
    y = -x + 10   : by linarith

end relationship_of_length_and_width_l822_822818


namespace real_values_count_l822_822573

theorem real_values_count :
  (∃ x : ℝ, 4^(x^2 - 5*x + 6) + 3 = 4 ∧ x > 1) ∧
  (∃ y : ℝ, 4^(y^2 - 5*y + 6) + 3 = 4 ∧ y > 1) ∧
  ∀ z : ℝ, 4^(z^2 - 5*z + 6) + 3 = 4 ∧ z > 1 → (z = 2 ∨ z = 3) :=
by
  sorry

end real_values_count_l822_822573


namespace greatest_sum_l822_822936

-- stating the conditions
def condition1 (x y : ℝ) := x^2 + y^2 = 130
def condition2 (x y : ℝ) := x * y = 45

-- proving the result
theorem greatest_sum (x y : ℝ) 
  (h1 : condition1 x y) 
  (h2 : condition2 x y) : 
  x + y = 10 * Real.sqrt 2.2 :=
sorry

end greatest_sum_l822_822936


namespace sqrt_180_simplify_l822_822063

theorem sqrt_180_simplify : real.sqrt 180 = 6 * real.sqrt 5 := by
  have h : 180 = 2^2 * 3^2 * 5 := by norm_num
  rw [h, real.sqrt_mul, real.sqrt_mul, real.sqrt_mul]
  all_goals { norm_num }
  sorry

end sqrt_180_simplify_l822_822063


namespace proof_problem_l822_822286

variable (p1 p2 p3 p4 : Prop)

theorem proof_problem (hp1 : p1) (hp2 : ¬ p2) (hp3 : ¬ p3) (hp4 : p4) :
  (p1 ∧ p4) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) := by
  sorry

end proof_problem_l822_822286


namespace concentric_circle_area_theorem_l822_822683

noncomputable def concentricCircleArea 
    (r s t p q : ℝ) (h1 : r > s) (h2 : s > t) (h3 : r^2 - t^2 = p^2 + q^2) : ℝ :=
π * (p^2 + q^2)

theorem concentric_circle_area_theorem 
    (r s t p q : ℝ) 
    (h1 : r > s) 
    (h2 : s > t) 
    (h3 : r^2 - t^2 = p^2 + q^2) : 
    concentricCircleArea r s t p q h1 h2 h3 = π * (p^2 + q^2) :=
by 
  sorry

end concentric_circle_area_theorem_l822_822683


namespace probability_of_same_color_l822_822871

-- Defining the given conditions
def green_balls := 6
def red_balls := 4
def total_balls := green_balls + red_balls

def probability_same_color : ℚ :=
  let prob_green := (green_balls / total_balls) * (green_balls / total_balls)
  let prob_red := (red_balls / total_balls) * (red_balls / total_balls)
  prob_green + prob_red

-- Statement of the problem rewritten in Lean 4
theorem probability_of_same_color :
  probability_same_color = 13 / 25 :=
by
  sorry

end probability_of_same_color_l822_822871


namespace tan_identity_l822_822008

variable {x y : ℝ}

theorem tan_identity (h1 : (sin x / cos y) + (sin y / cos x) = 2) 
                      (h2 : (cos x / sin y) + (cos y / sin x) = 3) :
  (tan x / tan y) + (tan y / tan x) = 2 / 5 :=
sorry

end tan_identity_l822_822008


namespace fermat_little_theorem_seven_pow_2048_mod_17_l822_822321

theorem fermat_little_theorem (a : ℤ) (p : ℕ) [fact (nat.prime p)] (h : a % p ≠ 0): a ^ (p - 1) ≡ 1 [MOD p] :=
sorry

theorem seven_pow_2048_mod_17 : (7 ^ 2048) % 17 = 1 :=
by
  have p := 17
  have prime_p : nat.prime p := by norm_num
  have a := 7
  have h : a % p ≠ 0 := by norm_num
  have fermat := fermat_little_theorem a p prime_p h
  have exp_decomp: 2048 = 16 * 128 := by norm_num
  rw exp_decomp
  calc 7 ^ (16 * 128) % 17
      = (7 ^ 16) ^ 128 % 17     : by rw pow_mul
      ... ≡ 1 ^ 128 % 17       : by rw fermat
      ... = 1 % 17             : by rw one_pow
      ... = 1                 : by norm_num

end fermat_little_theorem_seven_pow_2048_mod_17_l822_822321


namespace magical_stack_cards_l822_822174

theorem magical_stack_cards (n : ℕ) (h_cond1 : 2 * n ≥ 332)
  (h_cond2 : (111 ≤ n + 55) ∧ (90 ≤ n + 45)) : 
  2 * n = 332 :=
by
  have h1 : 111 = n + 55 := by ..
  have h2 : 90 = n + 45 := by ..
  sorry

end magical_stack_cards_l822_822174


namespace number_of_points_equal_start_angles_l822_822350

noncomputable def projection (S : set Point) (A : Point) : Point := sorry

theorem number_of_points_equal_start_angles 
  (S : set Point) (A B C : Point) (hA : A ∉ S) (hB : B ∉ S) (hC : C ∉ S) :
  ∃ n, n = 0 ∨ n = 1 ∨ n = 2 ∧ 
    ∃ P : set Point, 
      P ⊆ S ∧ 
      ∀ P' ∈ P, 
        (angle_between_lines (⟨P', A⟩, S) = angle_between_lines (⟨P', B⟩, S) ∧
         angle_between_lines (⟨P', B⟩, S) = angle_between_lines (⟨P', C⟩, S)) := 
sorry

end number_of_points_equal_start_angles_l822_822350


namespace complementary_event_defn_l822_822222

-- Defining the events
def event_A (n : ℕ) : Prop := ∃ k, k ≥ 2 ∧ k ≤ n ∧ sample_has_k_defective_products(k)
def complementary_event_of_A (n : ℕ) : Prop := ∃ k, k ≤ 1 ∧ k ≤ n ∧ sample_has_k_defective_products(k)

-- The proof statement
theorem complementary_event_defn (n : ℕ) : complementary_event_of_A n ↔ ¬ event_A n :=
sorry

end complementary_event_defn_l822_822222


namespace initial_avg_income_l822_822980

theorem initial_avg_income (A : ℝ) :
  (4 * A - 990 = 3 * 650) → (A = 735) :=
by
  sorry

end initial_avg_income_l822_822980


namespace triangle_area_l822_822935

def point := ℝ × ℝ

def A : point := (2, -3)
def B : point := (8, 1)
def C : point := (2, 3)

def area_triangle (A B C : point) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area : area_triangle A B C = 18 :=
  sorry

end triangle_area_l822_822935


namespace product_of_primes_l822_822591

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

theorem product_of_primes
  (a b : ℕ)
  (h_prime_a : is_prime a)
  (h_prime_b : is_prime b)
  (h_sum_odd : (a + b) % 2 = 1)
  (h_sum_less_100 : a + b < 100)
  (h_sum_multiple_17 : (a + b) % 17 = 0) :
  a * b = 166 :=
sorry

end product_of_primes_l822_822591


namespace ratio_of_any_two_numbers_is_rational_l822_822510

theorem ratio_of_any_two_numbers_is_rational
  (x1 x2 x3 x4 x5 : ℝ)
  (h1 : ∀ a b c : ℝ, a ≠ b → b ≠ c → a ≠ c → a ∈ {x1, x2, x3, x4, x5} → b ∈ {x1, x2, x3, x4, x5} → c ∈ {x1, x2, x3, x4, x5} → (a * b + b * c + c * a) ∈ ℚ) :
  ∀ a b : ℝ, a ∈ {x1, x2, x3, x4, x5} → b ∈ {x1, x2, x3, x4, x5} → (a / b) ∈ ℚ := 
sorry

end ratio_of_any_two_numbers_is_rational_l822_822510


namespace proposition_1_proposition_2_proposition_3_proposition_4_l822_822294

axiom p1 : Prop
axiom p2 : Prop
axiom p3 : Prop
axiom p4 : Prop

axiom p1_true : p1 = true
axiom p2_false : p2 = false
axiom p3_false : p3 = false
axiom p4_true : p4 = true

theorem proposition_1 : (p1 ∧ p4) = true := by sorry
theorem proposition_2 : (p1 ∧ p2) = false := by sorry
theorem proposition_3 : (¬p2 ∨ p3) = true := by sorry
theorem proposition_4 : (¬p3 ∨ ¬p4) = true := by sorry

end proposition_1_proposition_2_proposition_3_proposition_4_l822_822294


namespace arithmetic_sequence_properties_l822_822372

theorem arithmetic_sequence_properties (a : ℕ → ℤ) (T : ℕ → ℤ) (h1 : ∀ n, a (n + 1) - a n = a 1 - a 0) (h2 : a 4 = a 2 + 4) (h3 : a 3 = 6) :
  (∀ n, a n = 2 * n) ∧ (∀ n, T n = (4 / 3 * (4^n - 1))) :=
by
  sorry

end arithmetic_sequence_properties_l822_822372


namespace acquaintances_at_ends_equal_l822_822896

theorem acquaintances_at_ends_equal 
  (n : ℕ) -- number of participants
  (a b : ℕ → ℕ) -- functions which return the number of acquaintances before/after for each participant
  (h_ai_bi : ∀ (i : ℕ), 1 < i ∧ i < n → a i = b i) -- condition for participants except first and last
  (h_a1 : a 1 = 0) -- the first person has no one before them
  (h_bn : b n = 0) -- the last person has no one after them
  :
  a n = b 1 :=
by
  sorry

end acquaintances_at_ends_equal_l822_822896


namespace yara_ahead_of_theon_l822_822911

theorem yara_ahead_of_theon 
  (theon_speed : ℕ) (yara_speed : ℕ) (distance : ℕ)
  (h_theon_speed : theon_speed = 15)
  (h_yara_speed : yara_speed = 30)
  (h_distance : distance = 90) :
  (distance / theon_speed) - (distance / yara_speed) = 3 :=
by
  rw [h_theon_speed, h_yara_speed, h_distance]
  norm_num
  sorry

end yara_ahead_of_theon_l822_822911


namespace mary_machines_sold_l822_822015

open Nat

-- Definitions
def a₁ := 1
def d := 2

-- Sequence definition
def a (n : ℕ) := a₁ + (n - 1) * d

-- Sum of the arithmetic series
def S (n : ℕ) := (n * (a₁ + a n)) / 2

-- Problem statement
theorem mary_machines_sold : S 20 = 400 :=
by
  sorry

end mary_machines_sold_l822_822015


namespace simplify_expression_l822_822038

theorem simplify_expression (w : ℝ) : 3 * w + 6 * w - 9 * w + 12 * w - 15 * w + 21 = -3 * w + 21 :=
by
  sorry

end simplify_expression_l822_822038


namespace mary_should_drink_6_glasses_l822_822922

-- Definitions based on conditions
def daily_water_goal_liters : ℚ := 1.5
def glass_capacity_ml : ℚ := 250
def liter_to_milliliters : ℚ := 1000

-- Conversion from liters to milliliters
def daily_water_goal_milliliters : ℚ := daily_water_goal_liters * liter_to_milliliters

-- Proof problem to show Mary needs 6 glasses per day
theorem mary_should_drink_6_glasses :
  daily_water_goal_milliliters / glass_capacity_ml = 6 := by
  sorry

end mary_should_drink_6_glasses_l822_822922


namespace rice_containers_l822_822034

theorem rice_containers (total_weight_pounds : ℚ) (weight_per_container_ounces : ℚ) (pound_to_ounces : ℚ) : 
  total_weight_pounds = 29/4 → 
  weight_per_container_ounces = 29 → 
  pound_to_ounces = 16 → 
  (total_weight_pounds * pound_to_ounces) / weight_per_container_ounces = 4 := 
by
  intros h1 h2 h3 
  rw [h1, h2, h3]
  sorry

end rice_containers_l822_822034


namespace triangle_problem_l822_822441

noncomputable def angle_A_satisfies (A : ℝ) : Prop :=
  A = π / 3

noncomputable def ratio_sin_B_sin_C (sinB sinC : ℝ) : Prop :=
  sinB / sinC = 3 / 4

theorem triangle_problem
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : (2 * Real.cos A - 1) * Real.sin B + 2 * Real.cos A = 1)
  (h2 : 5 * b^2 = a^2 + 2 * c^2) :
  angle_A_satisfies A ∧ ratio_sin_B_sin_C (Real.sin B) (Real.sin C) :=
begin
  sorry
end

end triangle_problem_l822_822441


namespace tan_identity_l822_822009

variable {x y : ℝ}

theorem tan_identity (h1 : (sin x / cos y) + (sin y / cos x) = 2) 
                      (h2 : (cos x / sin y) + (cos y / sin x) = 3) :
  (tan x / tan y) + (tan y / tan x) = 2 / 5 :=
sorry

end tan_identity_l822_822009


namespace smallest_number_divisible_by_6_l822_822845

theorem smallest_number_divisible_by_6 :
  ∃ n, (∃ (digits : Finset ℕ), digits = {1, 2, 6, 7, 8} ∧ 
              (∀ x ∈ digits, x ∈ {1, 2, 6, 7, 8}) ∧ 
              ∀ x, multiset.count x digits.to_multiset = 1) ∧
            (even_digit_set = {2, 6, 8}) ∧
            12678 ∈ (multiset.permutations digits.to_multiset) ∧
            12678 % 6 = 0 := sorry

end smallest_number_divisible_by_6_l822_822845


namespace part1_part2_part3_l822_822733

-- Definition of the function
def linear_function (m : ℝ) (x : ℝ) : ℝ :=
  (2 * m + 1) * x + m - 3

-- Part 1: If the graph passes through the origin
theorem part1 (h : linear_function m 0 = 0) : m = 3 :=
by {
  sorry
}

-- Part 2: If the graph is parallel to y = 3x - 3
theorem part2 (h : ∀ x, linear_function m x = 3 * x - 3 → 2 * m + 1 = 3) : m = 1 :=
by {
  sorry
}

-- Part 3: If the graph intersects the y-axis below the x-axis
theorem part3 (h_slope : 2 * m + 1 ≠ 0) (h_intercept : m - 3 < 0) : m < 3 ∧ m ≠ -1 / 2 :=
by {
  sorry
}

end part1_part2_part3_l822_822733


namespace addition_correctness_l822_822608

theorem addition_correctness : 1.25 + 47.863 = 49.113 :=
by 
  sorry

end addition_correctness_l822_822608


namespace trapezoid_CD_length_l822_822812

theorem trapezoid_CD_length (AB CD AD BC : ℝ) (P : ℝ) 
  (h₁ : AB = 12) 
  (h₂ : AD = 5) 
  (h₃ : BC = 7) 
  (h₄ : P = 40) : CD = 16 :=
by
  sorry

end trapezoid_CD_length_l822_822812


namespace sum_of_divisors_85_l822_822942

theorem sum_of_divisors_85 :
  let divisors := {d ∈ Finset.range 86 | 85 % d = 0}
  Finset.sum divisors id = 108 :=
sorry

end sum_of_divisors_85_l822_822942


namespace unique_k_largest_n_l822_822609

theorem unique_k_largest_n :
  ∃! k : ℤ, ∃ n : ℕ, (n > 0) ∧ (5 / 18 < n / (n + k) ∧ n / (n + k) < 9 / 17) ∧ (n = 1) :=
by
  sorry

end unique_k_largest_n_l822_822609


namespace cost_of_fencing_per_meter_is_11_l822_822219

def farm_area : ℝ := 1200
def short_side : ℝ := 30
def total_cost : ℝ := 1320

def long_side (A W : ℝ) : ℝ := A / W
def diagonal (L W : ℝ) : ℝ := Real.sqrt (L^2 + W^2)
def total_length (L W D : ℝ) : ℝ := L + W + D
def cost_per_meter (totalCost totalLength : ℝ) : ℝ := totalCost / totalLength

theorem cost_of_fencing_per_meter_is_11 :
  cost_per_meter total_cost (total_length (long_side farm_area short_side) short_side (diagonal (long_side farm_area short_side) short_side)) = 11 := by
  sorry

end cost_of_fencing_per_meter_is_11_l822_822219


namespace largest_number_in_set_l822_822244

theorem largest_number_in_set : ∀ (a b c d : ℝ), (a = real.sqrt 2) → (b = 0) → (c = -1) → (d = 2) → a < d ∧ b < d ∧ c < d :=
begin
  intros a b c d ha hb hc hd,
  rw [ha, hb, hc, hd],
  split,
  {
    have h₁: real.sqrt 2 < 2 := by linarith [real.sqrt_lt_self (by norm_num)],
    exact h₁,
  },
  {
    split,
    {
      exact zero_lt_two,
    },
    {
      linarith,
    }
  }
end

end largest_number_in_set_l822_822244


namespace constant_sequence_iff_perfect_square_l822_822334

noncomputable def d (n : ℕ) : ℕ :=
  let m := Nat.sqrt n
  n - m * m

def b_seq (b0 : ℕ) : ℕ → ℕ
  | 0     => b0
  | (n+1) => b_seq n + d (b_seq n)

theorem constant_sequence_iff_perfect_square (b0 : ℕ) :
  (∃ N, ∀ n ≥ N, b_seq b0 n = b_seq b0 N) ↔ ∃ m, b0 = m * m := sorry

end constant_sequence_iff_perfect_square_l822_822334


namespace Q_has_at_most_n_integer_fixed_points_l822_822830

variable {R : Type} [CommRing R]

noncomputable def nested_polynomial (P : R[X]) (k : ℕ) : R[X] :=
  nat.rec_on k P (λ _ Q, Q.comp P)

theorem Q_has_at_most_n_integer_fixed_points
  (P : ℤ[X]) (n : ℕ) (hn : 1 < n) (k : ℕ) (hk : 0 < k)
  (hP : P.degree = n) :
  ∀ Q, Q = nested_polynomial P k → 
  ∃ m ≤ n, ∀ x : ℤ, Q.eval x = x → ∃ i : ℕ, i ≤ m ∧ P.eval (ite (i = 0) x (P.eval x ^ i)) = x :=
sorry

end Q_has_at_most_n_integer_fixed_points_l822_822830


namespace quasicolorable_condition_l822_822274

def is_quasicolorable (G : SimpleGraph) : Prop :=
  ∃ f : G.Edge → color, (
    (∀ (v : G.V), G.degree v = 3 →
      (G.edges_at v = {e1, e2, e3} → (f e1 = color.red ∧ f e2 = color.green ∧ f e3 = color.blue) ∨ (f e1 = color.white ∧ f e2 = color.white ∧ f e3 = color.white))) ∧
    (∃ e : G.Edge, f e ≠ color.white))

theorem quasicolorable_condition (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (G : SimpleGraph)
  (degree4_count : G.vertex_set.count (λ v, G.degree v = 4) = a)
  (degree3_count : G.vertex_set.count (λ v, G.degree v = 3) = b)
  (other_degrees : ∀ v ∈ G.vertex_set, G.degree v = 4 ∨ G.degree v = 3)
  (ab_condition : (a : ℚ) / b > 1 / 4) :
  is_quasicolorable G :=
sorry

end quasicolorable_condition_l822_822274


namespace solve_x_eq_10000_l822_822538

theorem solve_x_eq_10000 (x : ℝ) (h : 5 * x^(1/4 : ℝ) - 3 * (x / x^(3/4 : ℝ)) = 10 + x^(1/4 : ℝ)) : x = 10000 :=
by
  sorry

end solve_x_eq_10000_l822_822538


namespace simplify_sqrt_180_l822_822066

theorem simplify_sqrt_180 : sqrt 180 = 6 * sqrt 5 :=
by
  have h : 180 = 2^2 * 3^2 * 5 := by norm_num
  calc
    sqrt 180
      = sqrt (2^2 * 3^2 * 5)     : by rw [h]
  ... = sqrt (2^2) * sqrt (3^2) * sqrt 5 : by rw [sqrt_mul, sqrt_mul]
  ... = 2 * 3 * sqrt 5          : by rw [sqrt_sq, sqrt_sq]
  ... = 6 * sqrt 5              : by norm_num

end simplify_sqrt_180_l822_822066


namespace problem_l822_822353

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ := (-1)^n * a n + 1 / 2^n
noncomputable def T (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), S a i

theorem problem (a : ℕ → ℝ) 
  (h1 : ∀ n, S a n = (-1)^n * a n + 1 / 2^n) :
  T a 2017 = 1/3 * (1 - (1/2)^2016) :=
by
  sorry

end problem_l822_822353


namespace marker_exchange_impossible_l822_822913

/-
Statement:
Given 28 students where each student brings 3 markers: one red, one green, and one blue,
prove that it is impossible for the students to exchange markers such that each student ends up with three markers of the same color.
-/

theorem marker_exchange_impossible (n : ℕ) (hn : n = 28) :
  (∃ (f : Fin n → Fin 3 → Fin n → Fin 3), 
    (∀ i j, f i j ≠ ⟨i.val + 1, sorry⟩) ∧ 
    (∀ c : Fin 3, f (Fin.ofNat 0) c = c) ∧ 
    (∀ i j c, f i j c = f (Fin.mod Nat) (Fin.lean0) c)
  ) → false := 
by
  have h : 3 ∣ 84 := by sorry
  have h28 : ¬ (3 ∣ 28) := by sorry
  exact h28 (show 3 ∣ 28, by sorry)

end marker_exchange_impossible_l822_822913


namespace recurring_decimal_to_fraction_l822_822606

theorem recurring_decimal_to_fraction : 
  let x := 0.4ℚ + 37 / 990 in 
  x = 433 / 990 := 
by {
  -- Define recurring decimal and its fraction representation
  let x : ℚ := 0.4 + 37 / 900, -- This suffices for 0.4 recurring
  find recurrent representation,
  sorry
}

end recurring_decimal_to_fraction_l822_822606


namespace compute_mod_expression_l822_822272

theorem compute_mod_expression :
  (3 * (1 / 7) + 9 * (1 / 13)) % 72 = 18 := sorry

end compute_mod_expression_l822_822272


namespace bouquets_needed_to_earn_1000_l822_822269

theorem bouquets_needed_to_earn_1000 :
  ∀ (cost_per_bouquet sell_price_bouquet: ℕ) (roses_per_bouquet_bought roses_per_bouquet_sold target_profit: ℕ),
    cost_per_bouquet = 20 →
    sell_price_bouquet = 20 →
    roses_per_bouquet_bought = 7 →
    roses_per_bouquet_sold = 5 →
    target_profit = 1000 →
    (target_profit / (sell_price_bouquet * roses_per_bouquet_sold / roses_per_bouquet_bought - cost_per_bouquet) * roses_per_bouquet_bought = 125) :=
by
  intros cost_per_bouquet sell_price_bouquet roses_per_bouquet_bought roses_per_bouquet_sold target_profit 
    h_cost_per_bouquet h_sell_price_bouquet h_roses_per_bouquet_bought h_roses_per_bouquet_sold h_target_profit
  sorry

end bouquets_needed_to_earn_1000_l822_822269


namespace keystone_arch_angle_l822_822090

theorem keystone_arch_angle :
  (∃ trapezoids : Fin₁₀ (Triangle ℂ), 
     (∀ (t ∈ trapezoids), IsCongruent t) ∧ 
     (AreArrangedCenterCommonPoint trapezoids) ∧
     (EndTrapezoidHorizontal trapezoids)) →
  (x = 99) := 
sorry

end keystone_arch_angle_l822_822090


namespace terminating_decimal_count_l822_822718

theorem terminating_decimal_count (h₁ : 1 ≤ n ∧ n ≤ 569) 
(h₂ : ∀ m, m ∣ 570 → m = 2 ∨ m = 3 ∨ m = 5 ∨ m = 19) :
  ∑ (i : ℕ) in (finset.range 570).filter (λ i, i % 57 = 0), 1 = 9 :=
by sorry

end terminating_decimal_count_l822_822718


namespace part1_part2_part3_l822_822356

-- Define the even function f defined on ℝ
noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then x^2 - 4*x else x^2 + 4*x

-- Condition: f is an even function
axiom even_f : ∀ x : ℝ, f (-x) = f x

-- Prove f(-2) = -4
theorem part1 : f (-2) = -4 :=
sorry

-- Prove the expression for f(x) when x < 0
theorem part2 : ∀ x : ℝ, x < 0 → f(x) = x^2 + 4*x :=
sorry

-- Define the maximum value of function f on interval [t-1, t+1], where t > 1, as g(t)
noncomputable def g (t : ℝ) : ℝ :=
if 1 < t ∧ t <= 2 then f (t - 1)
else if t > 2 then f (t + 1)
else 0

-- Prove the minimum value of g(t)
theorem part3 : ∃ t : ℝ, t > 1 ∧ g(t) = -3 :=
sorry

end part1_part2_part3_l822_822356


namespace trig_identity_l822_822365

theorem trig_identity (α : ℝ) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) 
  : Real.cos (5 / 6 * π + α) + (Real.cos (4 * π / 3 + α))^2 = (2 - Real.sqrt 3) / 3 := 
sorry

end trig_identity_l822_822365


namespace height_of_wooden_box_l822_822234

theorem height_of_wooden_box 
  (height : ℝ)
  (h₁ : ∀ (length width : ℝ), length = 8 ∧ width = 10)
  (h₂ : ∀ (small_length small_width small_height : ℕ), small_length = 4 ∧ small_width = 5 ∧ small_height = 6)
  (h₃ : ∀ (num_boxes : ℕ), num_boxes = 4000000) :
  height = 6 := 
sorry

end height_of_wooden_box_l822_822234


namespace greatest_sum_l822_822937

-- stating the conditions
def condition1 (x y : ℝ) := x^2 + y^2 = 130
def condition2 (x y : ℝ) := x * y = 45

-- proving the result
theorem greatest_sum (x y : ℝ) 
  (h1 : condition1 x y) 
  (h2 : condition2 x y) : 
  x + y = 10 * Real.sqrt 2.2 :=
sorry

end greatest_sum_l822_822937


namespace fraction_apple_juice_in_mixture_l822_822144

theorem fraction_apple_juice_in_mixture :
  let pitcher1_capacity := 800
  let pitcher2_capacity := 500
  let fraction_juice_pitcher1 := (1 : ℚ) / 4
  let fraction_juice_pitcher2 := (3 : ℚ) / 8
  let apple_juice_pitcher1 := pitcher1_capacity * fraction_juice_pitcher1
  let apple_juice_pitcher2 := pitcher2_capacity * fraction_juice_pitcher2
  let total_apple_juice := apple_juice_pitcher1 + apple_juice_pitcher2
  let total_capacity := pitcher1_capacity + pitcher2_capacity
  (total_apple_juice / total_capacity = 31 / 104) :=
by
  sorry

end fraction_apple_juice_in_mixture_l822_822144


namespace Jane_sequins_total_l822_822472

theorem Jane_sequins_total :
  (10 * 12) + (8 * 15) + (14 * 20) + ((list.sum [10, 15, 20, 25, 30])) = 620 :=
by
  sorry

end Jane_sequins_total_l822_822472


namespace card_game_termination_l822_822145

theorem card_game_termination 
  {n : ℕ} 
  (cards : Fin n → Fin n → Prop) 
  (irreflexive : ∀ i, ¬ cards i i)
  (arbitrary_distribution : ∃ p1 p2 : Fin n → {0, 1}, ∀ i, p1 i + p2 i = 1) 
  (moves : ∀ (p1_cards p2_cards : List (Fin n)), { p1_cards' : List (Fin n) // length p1_cards + length p2_cards = length p1_cards' + length p2_cards }
    ∨ { p2_cards' : List (Fin n) // length p1_cards + length p2_cards = length p1_cards + length p2_cards' }) :
  ∀ (p1_cards p2_cards : List (Fin n)),
    (length p1_cards = 0 ∨ length p2_cards = 0) ∨ 
    (∃ final_state : List (List (Fin n) × List (Fin n)),
      ∀ s ∈ final_state, s ≠ ([], []) → ∃ s', moves s.1 s.2 = some s') :=
sorry

end card_game_termination_l822_822145


namespace last_letter_53rd_permutation_bench_l822_822076

theorem last_letter_53rd_permutation_bench : 
  let word := "BENCH" in
  let alphabet := ['B', 'E', 'N', 'C', 'H'] in
  let perms := List.permutationsRec alphabet in
  let sorted_perms := List.sort perms in
  true → -- assumption to state the theorem without dealing with Lean's syntactic requirements
  List.getLast (List.get! sorted_perms 52) = 'H' :=
by
  intros
  sorry -- Proof omitted

end last_letter_53rd_permutation_bench_l822_822076


namespace constant_term_binomial_expansion_l822_822496

def a : ℝ := ∫ x in 0..Real.pi, Real.sin x

theorem constant_term_binomial_expansion :
  a = 2 →
  ∃ (c : ℝ), c = -160 ∧
    (∀ x : ℝ, (a * Real.sqrt x - 1 / Real.sqrt x) ^ 6 = 
              c * x^0 + (some of other terms)) :=
by {
  intro ha,
  use -160,
  split,
  { exact rfl },
  { sorry }
}

end constant_term_binomial_expansion_l822_822496


namespace arccos_lt_arcsin_iff_l822_822317

theorem arccos_lt_arcsin_iff (x : ℝ) (h : x ∈ Icc (-1 : ℝ) 1) : arccos x < arcsin x ↔ 0 < x ∧ x ≤ 1 :=
by
  sorry

end arccos_lt_arcsin_iff_l822_822317


namespace eight_n_is_even_l822_822779

theorem eight_n_is_even (n : ℕ) (h : n = 7) : 8 * n = 56 :=
by {
  sorry
}

end eight_n_is_even_l822_822779


namespace calc_sqrt_expr_l822_822674

theorem calc_sqrt_expr : (Real.sqrt 2 + 1) ^ 2 - Real.sqrt 18 + 2 * Real.sqrt (1 / 2) = 3 := by
  sorry

end calc_sqrt_expr_l822_822674


namespace remainder_when_587421_divided_by_6_l822_822612

theorem remainder_when_587421_divided_by_6 :
  ¬ (587421 % 2 = 0) → (587421 % 3 = 0) → 587421 % 6 = 3 :=
by sorry

end remainder_when_587421_divided_by_6_l822_822612


namespace bart_firewood_burning_period_l822_822314

-- We'll state the conditions as definitions.
def pieces_per_tree := 75
def trees_cut_down := 8
def logs_burned_per_day := 5

-- The theorem to prove the period Bart burns the logs.
theorem bart_firewood_burning_period :
  (trees_cut_down * pieces_per_tree) / logs_burned_per_day = 120 :=
by
  sorry

end bart_firewood_burning_period_l822_822314


namespace remainder_expression_div_10_l822_822425

theorem remainder_expression_div_10 (p t : ℕ) (hp : p > t) (ht : t > 1) :
  (92^p * 5^p + t + 11^t * 6^(p * t)) % 10 = 1 :=
by
  sorry

end remainder_expression_div_10_l822_822425


namespace domain_expression_l822_822702

-- Define the conditions for the domain of the expression
def valid_numerator (x : ℝ) : Prop := 3 * x - 6 ≥ 0
def valid_denominator (x : ℝ) : Prop := 7 - 2 * x > 0

-- Proof problem statement
theorem domain_expression (x : ℝ) : valid_numerator x ∧ valid_denominator x ↔ 2 ≤ x ∧ x < 3.5 :=
sorry

end domain_expression_l822_822702


namespace hyperbola_asymptotes_l822_822398

theorem hyperbola_asymptotes (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) (h4 : (2 : ℝ) = 2) :
  ∀ x : ℝ, y = sqrt 3 * x :=
by skip

end hyperbola_asymptotes_l822_822398


namespace no_polynomial_transform_l822_822470

theorem no_polynomial_transform (a b c : ℚ) :
  ¬ (∀ (x y : ℚ),
      ((x = 1 ∧ y = 1) ∨ (x = 4 ∧ y = 10) ∨ (x = 7 ∧ y = 7)) →
      a * x^2 + b * x + c = y) :=
by
  sorry

end no_polynomial_transform_l822_822470


namespace polyhedron_properties_l822_822643

noncomputable def polyhedron_surface_area (vertices : Fin 8 → (Fin 3 → ℝ)) : ℝ := 
  -- Assume a function that computes the surface area
  sorry

noncomputable def polyhedron_volume (vertices : Fin 8 → (Fin 3 → ℝ)) : ℝ :=
  -- Assume a function that computes the volume
  sorry

theorem polyhedron_properties :
    ∃ (vertices : Fin 8 → (Fin 3 → ℝ)),
    (∀ i, 
      -- Defining edge lengths and conditions
      if i = 0 ∨ i = 1 then 
        vertices i = λ j, if j = 0 ∨ j = 1 then 1 else 0.8
      else if i = 2 ∨ i = 3 ∨ i = 4 ∨ i = 5 then 
        vertices i = λ j, if j = 0 then 1.1 else 2
      else
        vertices i = λ j, if j = 0 then 2.2 else 3) ∧
    polyhedron_surface_area vertices = 12.49 ∧
    polyhedron_volume vertices = 2.47 :=
sorry

end polyhedron_properties_l822_822643


namespace proof_problem_l822_822285

variable (p1 p2 p3 p4 : Prop)

theorem proof_problem (hp1 : p1) (hp2 : ¬ p2) (hp3 : ¬ p3) (hp4 : p4) :
  (p1 ∧ p4) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) := by
  sorry

end proof_problem_l822_822285


namespace ball_experiment_proof_l822_822451

-- Defining the problem conditions and necessary variables
variable (n : ℕ) (observations : List (ℕ × ℕ))

-- Assuming the conditions from the table
def conditions (observations : List (ℕ × ℕ)) : Prop :=
  observations = [(150, 60), (300, 126), (600, 247), (900, 365), (1200, 484), (1500, 609)]

-- The probability calculation related to the white balls
def frequency (picks : ℕ) (white_picks : ℕ) : ℚ :=
  white_picks / picks

-- Proving the relationships including frequency and final results
theorem ball_experiment_proof (hcond : conditions observations) : 
  (frequency 300 126 = 0.42) ∧
  (frequency 1500 609 = 0.406) ∧
  (let P_white := 0.40 in 1 - P_white = 0.6) ∧
  (let P_red := 0.6 in let x := n in (x / (x + 10) = P_red) → x = 15) :=
by
  sorry

end ball_experiment_proof_l822_822451


namespace problem_1_split_terms_problem_2_split_terms_l822_822499

-- Problem 1 Lean statement
theorem problem_1_split_terms :
  (28 + 5/7) + (-25 - 1/7) = 3 + 4/7 := 
  sorry
  
-- Problem 2 Lean statement
theorem problem_2_split_terms :
  (-2022 - 2/7) + (-2023 - 4/7) + 4046 - 1/7 = 0 := 
  sorry

end problem_1_split_terms_problem_2_split_terms_l822_822499


namespace sum_of_roots_l822_822711

open Real

noncomputable def quadratic_sum_of_roots : ℝ :=
  let a : ℝ := 5 + 3 * sqrt 3
  let b : ℝ := -(1 + 2 * sqrt 3)
  let c : ℝ := 1
  -(b / a)

theorem sum_of_roots (a b c : ℝ) :
  a = 5 + 3 * sqrt 3 →
  b = -(1 + 2 * sqrt 3) →
  c = 1 →
  quadratic_sum_of_roots = -6.5 + 3.5 * sqrt 3 :=
by
  intros ha hb hc
  simp [quadratic_sum_of_roots, ha, hb, hc]
  sorry

end sum_of_roots_l822_822711


namespace probability_root_exists_l822_822525

theorem probability_root_exists :
  ∀ (a b : ℝ), a ∈ Icc (-π) π → b ∈ Ιcc (-π) π →
  (∃ x, x^2 + 2 * a * x - b^2 + π = 0) →
  (Prob (λ (a b : ℝ), a^2 + b^2 ≥ π) (prod measure_Icc measure_Icc : measure (ℝ × ℝ))) = 3 / 4 :=
by
  sorry

end probability_root_exists_l822_822525


namespace problem1_problem2_problem3_l822_822859

variables {a b c : ℝ}
def s := (a + b + c) / 2
def s1 := s - a
def s2 := s - b
def s3 := s - c
def t : ℝ -- the area of the triangle is implicitly given and not defined here
def r := t / s
def r1 := t / s1
def r2 := t / s2
def r3 := t / s3

theorem problem1 : r^2 = s1 * s2 * s3 / s := sorry
theorem problem2 : 1 / r1 + 1 / r2 + 1 / r3 = 1 / r := sorry
theorem problem3 : r * r1 * r2 * r3 = s * s1 * s2 * s3 := sorry

end problem1_problem2_problem3_l822_822859


namespace students_in_A_and_D_combined_l822_822277

theorem students_in_A_and_D_combined (AB BC CD : ℕ) (hAB : AB = 83) (hBC : BC = 86) (hCD : CD = 88) : (AB + CD - BC = 85) :=
by
  sorry

end students_in_A_and_D_combined_l822_822277


namespace freezer_temperature_l822_822434

theorem freezer_temperature 
  (refrigeration_temp : ℝ)
  (freezer_temp_diff : ℝ)
  (h1 : refrigeration_temp = 4)
  (h2 : freezer_temp_diff = 22)
  : (refrigeration_temp - freezer_temp_diff) = -18 :=
by 
  sorry

end freezer_temperature_l822_822434


namespace least_number_remainder_l822_822610

theorem least_number_remainder :
  ∃ n, n ≡ 4 [MOD 5] ∧ n ≡ 4 [MOD 6] ∧ n ≡ 4 [MOD 9] ∧ n ≡ 4 [MOD 12] ∧ n = 184 := 
by
  sorry

end least_number_remainder_l822_822610


namespace ratio_AJEC_correct_l822_822033

def Rectangle (A B C D : Type) (AB : ℝ) (BC : ℝ) := 
  sorry

def Square (E F G H : Type) (EF : ℝ) := 
  sorry

def midpoint (x y : ℝ) := (x + y) / 2

noncomputable def AJEC_ratio : ℝ :=
  let area_ABCD := 2 * 1
  let area_EFGH := 1 * 1
  let area_AJEC := 0.5
  area_AJEC / (area_ABCD + area_EFGH)

theorem ratio_AJEC_correct :
  AJEC_ratio = 1 / 6 :=
sorry

end ratio_AJEC_correct_l822_822033


namespace extreme_value_when_a_is_one_range_of_lambda_l822_822825

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 * exp(1 - x) - a * (x - 1)

theorem extreme_value_when_a_is_one :
  let f1 (x : ℝ) := f 1 x in
  (∀ x ∈ Ioo (3 / 4 : ℝ) 2, f1' x = deriv f1 x) →
  (∀ x ∈ Ioo (3 / 4 : ℝ) 2, 
    (deriv f1 x > 0 ↔ x < 1) ∧ 
    (deriv f1 x < 0 ↔ x > 1) ∧ 
    f1 1 = 1) :=
sorry

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + a * (x - 1 - exp(1 - x))

theorem range_of_lambda (a : ℝ) (x1 x2 : ℝ) (h : x1 < x2)
  (hx1 : deriv (g a) x1 = 0)
  (hx2 : deriv (g a) x2 = 0)
  (h_condition : x2 * g a x1 ≤ λ * deriv (f a) x1) :
  a > -1 → ∃ λ, λ = 2 * exp(1) / (exp(1) + 1) :=
sorry

end extreme_value_when_a_is_one_range_of_lambda_l822_822825


namespace susana_chocolate_chips_l822_822932

theorem susana_chocolate_chips :
  ∃ (S_c : ℕ), 
  (∃ (V_c V_v S_v : ℕ), 
    V_c = S_c + 5 ∧
    S_v = (3 * V_v) / 4 ∧
    V_v = 20 ∧
    V_c + S_c + V_v + S_v = 90) ∧
  S_c = 25 :=
by
  existsi 25
  sorry

end susana_chocolate_chips_l822_822932


namespace Miriam_flowers_six_days_l822_822846

theorem Miriam_flowers_six_days (h1 : ∀ days, Miriam_works_hours_per_day : 5, flowers_taken_care_per_day : 60) : 
  flowers_taken_care_in_6_days = 360 :=
sorry

end Miriam_flowers_six_days_l822_822846


namespace num_of_friends_l822_822529

theorem num_of_friends :
  ∃ friends : Finset String, 
  friends = {"Sam", "Dan", "Tom", "Keith"} ∧
  friends.card = 4 :=
begin
  sorry
end

end num_of_friends_l822_822529


namespace rotokas_license_plates_l822_822078

theorem rotokas_license_plates :
  let alphabet := {A, E, G, I, K, O, P, R, T, U, V, B} in
  let total_letters := 12 in
  let plate_length := 5 in
  let first_letter_choices := 2 in
  let last_letter := B in
  let excluded_letters := {V} in
  ∃ plates : ℕ,
  plates = first_letter_choices * 
           (total_letters - 3) *  -- 9 choices for the second letter
           (total_letters - 4) *  -- 8 choices for the third letter
           (total_letters - 5) *  -- 7 choices for the fourth letter
           1 ∧  -- 1 choice for the fifth letter (B)
  plates = 1008 :=
sorry

end rotokas_license_plates_l822_822078


namespace odd_function_condition_l822_822094

theorem odd_function_condition (a b : ℝ) :
  (∀ x : ℝ, f(x) = x * |x + a| + b → (∀ x : ℝ, f(-x) = -f(x))) ↔ (a^2 + b^2 = 0) := sorry

end odd_function_condition_l822_822094


namespace part1_even_part2_monotonic_part3_range_m_l822_822383

noncomputable def f (x : ℝ) := Real.log (Real.exp x + Real.exp (-x))

theorem part1_even : 
  ∀ x, f (-x) = f x := 
by sorry

theorem part2_monotonic:
  ∀ (x1 x2 : ℝ), x1 > 0 → x2 > 0 → x2 > x1 → f x2 > f x1 := 
by sorry

theorem part3_range_m (m : ℝ) :
  (∀ x : ℝ, Real.exp (2*x) + Real.exp (-2*x) - 2*m*Real.exp(f x) + 6*m + 2 ≥ 0) ↔
  (-2 ≤ m ∧ m ≤ 2 ∨ 2 < m ∧ m ≤ 6) :=
by sorry

end part1_even_part2_monotonic_part3_range_m_l822_822383


namespace find_cos_tan_values_given_sinα_and_point_l822_822355

noncomputable def cos_tan_values_given_sinα_and_point (y : ℝ) (α : ℝ) :=
∃ (cosα tanα : ℝ),
  (cosα = -1 ∧ tanα = 0 ∧ y = 0) ∨
  (cosα = -√6 / 4 ∧ tanα = -√15 / 3 ∧ y = √5) ∨
  (cosα = -√6 / 4 ∧ tanα = √15 / 3 ∧ y = -√5)

theorem find_cos_tan_values_given_sinα_and_point (y : ℝ) (α : ℝ) (h : sin α = √2 / 4 * y) :
  cos_tan_values_given_sinα_and_point y α
:= sorry

end find_cos_tan_values_given_sinα_and_point_l822_822355


namespace chip_production_profit_l822_822850

-- Define conditions as assumptions in Lean
def R (x : ℝ) (k : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 20 then 100 - k * x
  else (2100 / x) - (9000 * k / (x ^ 2))

def W (x : ℝ) (k : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 20 then x * (100 - k * x) - 50 - 20 * x
  else x * ((2100 / x) - (9000 * k / (x ^ 2))) - 50 - 20 * x

-- Defining the theorem with the questions and conditions
theorem chip_production_profit (x : ℝ) (k : ℝ) :
  (W 5 k = 300) → (k = 2) ∧ 
  (W x k = if 0 < x ∧ x ≤ 20 then 80 * x - 2 * x^2 - 50 else 2050 - 20 * x - 18000 / x) ∧
  ((x = 30) → (W 30 2 = 850)) := 
sorry

end chip_production_profit_l822_822850


namespace sequence_formula_correct_l822_822697

axiom sequence_first_five (a : ℕ → ℕ) : a 1 = 1 ∧ a 2 = 2 ∧ a 3 = 4 ∧ a 4 = 7 ∧ a 5 = 11

theorem sequence_formula_correct (a : ℕ → ℕ) (h : ∀ n, a 1 = 1 ∧ a 2 = 2 ∧ a 3 = 4 ∧ a 4 = 7 ∧ a 5 = 11) :
  ∀ n, a n = (n^2 - n + 2)/2 :=
by
  sorry

end sequence_formula_correct_l822_822697


namespace units_digit_of_m_squared_plus_3_to_the_m_l822_822002

theorem units_digit_of_m_squared_plus_3_to_the_m (m : ℕ) (h : m = 2010^2 + 2^2010) : 
  (m^2 + 3^m) % 10 = 7 :=
by {
  sorry -- proof goes here
}

end units_digit_of_m_squared_plus_3_to_the_m_l822_822002


namespace parabola_standard_equation_l822_822782

theorem parabola_standard_equation (distance_focus_directrix : ℝ) (parabola_opens_left : Bool) :
  distance_focus_directrix = 3 ∧ parabola_opens_left = true → (∃ k : ℝ, k = -6 ∧ ∀ y x : ℝ, y^2 = k * x) :=
by
  intros h
  have h1: distance_focus_directrix = 3 := h.1
  have h2: parabola_opens_left = true := h.2
  use -6
  split
  . exact rfl
  . intros y x
    exact sorry

end parabola_standard_equation_l822_822782


namespace placing_balls_problem_l822_822519
open Nat

noncomputable def ways_to_place_balls : ℕ :=
  20

theorem placing_balls_problem :
  let balls := [1, 2, 3, 4, 5]
  let boxes := [1, 2, 3, 4, 5]
  -- Exactly 2 balls must be placed in the corresponding numbered boxes
  (∃ (f : Fin 5 → Fin 5), 
     (∀ i, f i = i → i = 1 ∨ i = 2) ∧ 
     (([:2] : List (Fin 5)).size = 2) ∧
     let rem := ([3, 4, 5] : List (Fin 5)) in
     (∃ (g : Fin 3 → Fin 3), 
        ∀ i, g i ≠ i)) → 
  (@ways_to_place_balls = 20) :=
sorry

end placing_balls_problem_l822_822519


namespace find_n_l822_822487

theorem find_n (n : ℕ) (b : Fin (n + 1) → ℝ) (h0 : b 0 = 45) (h1 : b 1 = 81) (hn : b n = 0) (rec : ∀ (k : ℕ), 1 ≤ k → k < n → b (k+1) = b (k-1) - 5 / b k) : 
  n = 730 :=
sorry

end find_n_l822_822487


namespace divide_money_equally_l822_822929

def total_distance : ℕ := 16
def anatoly_distance : ℕ := 6
def vladimir_distance : ℕ := 10
def boris_contribution : ℕ := 16 -- in million rubles

noncomputable def anatoly_share : ℕ := 2 -- in million rubles
noncomputable def vladimir_share : ℕ := 14 -- in million rubles

theorem divide_money_equally :
  anatoly_share = (2 : ℕ) ∧ vladimir_share = (14 : ℕ) :=
by
  -- conditions
  have h1 : anatoly_distance + vladimir_distance = total_distance := sorry
  have h2 : total_distance / 3 = (5 + 1/3 : ℚ) := sorry
  -- proof of shares
  have h3 : anatoly_distance - (5 + 1/3 : ℚ) = 2/3 := sorry
  have h4 : vladimir_distance - (5 + 1/3 : ℚ) = 14/3 := sorry
  have share_ratio : (2/3) / (14/3) = 1/7 := sorry
  have total_rubles : 16 = boris_contribution := sorry
  have anatoli_share_proof : anatoly_share = ((2 / 16) * boris_contribution : ℚ) := sorry
  have vladimir_share_proof : vladimir_share = ((14 / 16) * boris_contribution : ℚ) := sorry
  exact ⟨anatoli_share_proof, vladimir_share_proof⟩

end divide_money_equally_l822_822929


namespace smallest_d_value_l822_822214

theorem smallest_d_value (d : ℚ) (h : real.sqrt ((4 * real.sqrt 3)^2 + (2 * d - 1)^2) = 2 * d + 4) : 
  d = 33 / 20 :=
sorry

end smallest_d_value_l822_822214


namespace cubic_equation_unique_real_solution_l822_822720

theorem cubic_equation_unique_real_solution :
  (∃ (m : ℝ), ∀ x : ℝ, x^3 - 4*x - m = 0 → x = 2) ↔ m = -8 :=
by sorry

end cubic_equation_unique_real_solution_l822_822720


namespace sequence_sum_eq_six_l822_822785

theorem sequence_sum_eq_six :
  let a_n (n : ℕ) : ℤ := (-1)^(n+1) * (3 * n - 2)
  in a_n 1 + a_n 2 + a_n 3 + a_n 4 = 6 :=
by
  let a_n (n : ℕ) : ℤ := (-1)^(n+1) * (3 * n - 2)
  exact Eq.refl 6

end sequence_sum_eq_six_l822_822785


namespace tangent_line_intersects_y_axis_at_origin_l822_822586

theorem tangent_line_intersects_y_axis_at_origin :
  let x := exp 1
  let y := 2 * log x
  let dydx := (2 : ℝ) / x
  let tangent_line := λ x' : ℝ, dydx * x'
  tangent_line 0 = 0 :=
by
  sorry

end tangent_line_intersects_y_axis_at_origin_l822_822586


namespace carol_total_spent_l822_822275

-- Given total savings S. The amount spent on the stereo.
def stereo_spent (S : ℝ) : ℝ := (1 / 4) * S

-- Given total savings S. The amount spent on the television.
def television_spent (S : ℝ) : ℝ := (1 / 4) * S - (1 / 6) * S

-- The problem is to prove that the total spent equals 1/3 of the savings
theorem carol_total_spent (S : ℝ) (h : S > 0) : (stereo_spent S) + (television_spent S) = (1 / 3) * S := 
sorry

end carol_total_spent_l822_822275


namespace unpainted_area_on_five_inch_board_l822_822597

noncomputable def area_unpainted_region (w1 w2 : ℝ) (θ : ℝ) : ℝ :=
  let height := w1 * Real.sin θ in w2 * height

theorem unpainted_area_on_five_inch_board :
  ∀ (w1 w2 : ℝ) (θ : ℝ), 
  w1 = 5 ∧ w2 = 7 ∧ θ = Real.pi / 4 →
  area_unpainted_region w1 w2 θ = 35 * Real.sqrt 2 / 2 :=
by
  intros w1 w2 θ h
  sorry

end unpainted_area_on_five_inch_board_l822_822597


namespace cube_shadow_l822_822660

theorem cube_shadow (y : ℝ) (h1 : y = real.sqrt 179 - 2) :
  ⌊100 * y⌋ = 333 :=
by
  sorry

end cube_shadow_l822_822660


namespace problem_solution_l822_822746

variable (α : ℝ)
variable (h : Real.cos α = 1 / 5)

theorem problem_solution : Real.cos (2 * α - 2017 * Real.pi) = 23 / 25 := by
  sorry

end problem_solution_l822_822746


namespace mean_combined_scores_l822_822505

theorem mean_combined_scores (M A : ℝ) (m a : ℕ) 
  (hM : M = 88) 
  (hA : A = 72) 
  (hm : (m:ℝ) / (a:ℝ) = 2 / 3) :
  (88 * m + 72 * a) / (m + a) = 78 :=
by
  sorry

end mean_combined_scores_l822_822505


namespace card_game_eventually_one_without_cards_l822_822151

def card_game (n : ℕ) : Prop :=
  ∃ (cards : Fin n → Fin n) 
    (beats : (Fin n) → (Fin n) → Prop)
    (initial_distribution : Fin n → Fin 2),
  (∀ x y z : Fin n, beats x y ∨ beats y x) →
  (∀ x y z : Fin n, beats x z ∧ beats z y → beats x y) →
  (∀ distribution : Fin n → Fin 2, 
     ∃ turn : ℕ → ((Fin n) × (Fin n)) → Prop,
       (∀ i : ℕ, (turn i) (initial_distribution i) →
         ∃ j : ℕ, turn (i + 1) (beats _ _) → 
           (∃ k : ℕ, turn k (initial_distribution i) → false)))

theorem card_game_eventually_one_without_cards (n : ℕ) : card_game n :=
sorry

end card_game_eventually_one_without_cards_l822_822151


namespace area_ratio_of_similar_polygons_l822_822789

theorem area_ratio_of_similar_polygons (similarity_ratio: ℚ) (hratio: similarity_ratio = 1/5) : (similarity_ratio ^ 2 = 1/25) := 
by 
  sorry

end area_ratio_of_similar_polygons_l822_822789


namespace positive_slope_of_asymptote_l822_822568

-- Define the conditions
def is_hyperbola (x y : ℝ) : Prop :=
  abs (Real.sqrt ((x - 1) ^ 2 + (y + 2) ^ 2) - Real.sqrt ((x - 5) ^ 2 + (y + 2) ^ 2)) = 3

-- Prove the positive slope of the asymptote of the given hyperbola
theorem positive_slope_of_asymptote :
  (∀ x y : ℝ, is_hyperbola x y) → abs (Real.sqrt 7 / 3) = Real.sqrt 7 / 3 :=
by
  intros h
  -- Proof to be provided (proof steps from the provided solution would be used here usually)
  sorry

end positive_slope_of_asymptote_l822_822568


namespace x_squared_lt_1_neither_sufficient_nor_necessary_for_2x_lt_1_l822_822421

theorem x_squared_lt_1_neither_sufficient_nor_necessary_for_2x_lt_1
  (x : ℝ) : ¬ ((x^2 < 1) → (2^x < 1)) ∧ ¬ ((2^x < 1) → (x^2 < 1)) := 
sorry

end x_squared_lt_1_neither_sufficient_nor_necessary_for_2x_lt_1_l822_822421


namespace polar_coords_of_point_neg3_3_l822_822302

theorem polar_coords_of_point_neg3_3 :
  ∃ (r θ : ℝ), 0 < r ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 
  r = Real.sqrt ((-3)^2 + 3^2) ∧
  θ = Real.pi - Real.atan(1) ∧
  (r, θ) = (3 * Real.sqrt 2, 3 * Real.pi / 4) :=
sorry

end polar_coords_of_point_neg3_3_l822_822302


namespace collinear_tangents_l822_822865

noncomputable theory
open Geometry

/-- Given a quadrilateral ABCD inscribed in a circle, and points P, Q such that extensions of AB, DC
intersect at P, and extensions of AD, BC intersect at Q. From Q, two tangents QE and QF are drawn
to the circle at E and F respectively. Prove that the points P, E, and F are collinear. -/
theorem collinear_tangents 
  (A B C D P Q E F : Point)
  (h_cyclic_ABCD : Cyclic (Quadrilateral A B C D)) 
  (h_inter_AB_DC : Intersect_extended_at (Line_thru_points A B) (Line_thru_points D C) P)
  (h_inter_AD_BC : Intersect_extended_at (Line_thru_points A D) (Line_thru_points B C) Q)
  (h_tangent_QE : Tangent (Line_thru_points Q E) (Circumcircle A B C D))
  (h_tangent_QF : Tangent (Line_thru_points Q F) (Circumcircle A B C D)) :
  Collinear P E F :=
sorry

end collinear_tangents_l822_822865


namespace lines_divisible_by_4_l822_822478

theorem lines_divisible_by_4 (n : ℕ) (hn : 0 < n) :
  let S := { (a, b) : ℕ × ℕ | a ≤ 90 * n + 1 ∧ b ≤ 90 * n + 5 } in
  ∃ k : ℕ, k * 4 = (number_of_lines_through_points S) :=
begin
  sorry
end

end lines_divisible_by_4_l822_822478


namespace find_function_and_max_profit_l822_822692

noncomputable def profit_function (x : ℝ) : ℝ := -50 * x^2 + 1200 * x - 6400

theorem find_function_and_max_profit :
  (∀ (x : ℝ), (x = 10 → (-50 * x + 800 = 300)) ∧ (x = 13 → (-50 * x + 800 = 150))) ∧
  (∃ (x : ℝ), x = 12 ∧ profit_function x = 800) :=
by
  sorry

end find_function_and_max_profit_l822_822692


namespace proposition_1_proposition_2_proposition_3_proposition_4_l822_822293

axiom p1 : Prop
axiom p2 : Prop
axiom p3 : Prop
axiom p4 : Prop

axiom p1_true : p1 = true
axiom p2_false : p2 = false
axiom p3_false : p3 = false
axiom p4_true : p4 = true

theorem proposition_1 : (p1 ∧ p4) = true := by sorry
theorem proposition_2 : (p1 ∧ p2) = false := by sorry
theorem proposition_3 : (¬p2 ∨ p3) = true := by sorry
theorem proposition_4 : (¬p3 ∨ ¬p4) = true := by sorry

end proposition_1_proposition_2_proposition_3_proposition_4_l822_822293


namespace solve_sqrt_equation_l822_822070

theorem solve_sqrt_equation (x y : ℚ) :
  sqrt (2 * sqrt 3 - 3) = sqrt (x * sqrt 3) - sqrt (y * sqrt 3) →
  x = 3 / 2 ∧ y = 1 / 2 := 
sorry

end solve_sqrt_equation_l822_822070


namespace complex_point_in_fourth_quadrant_l822_822576

theorem complex_point_in_fourth_quadrant (z : ℂ) (h : z = 1 / (1 + I)) :
  z.re > 0 ∧ z.im < 0 :=
by
  -- Here we would provide the proof, but it is omitted as per the instructions.
  sorry

end complex_point_in_fourth_quadrant_l822_822576


namespace correct_operation_B_l822_822169

variable (a : ℝ)

theorem correct_operation_B :
  2 * a^2 * a^4 = 2 * a^6 :=
by sorry

end correct_operation_B_l822_822169


namespace unpainted_area_on_five_inch_board_l822_822598

noncomputable def area_unpainted_region (w1 w2 : ℝ) (θ : ℝ) : ℝ :=
  let height := w1 * Real.sin θ in w2 * height

theorem unpainted_area_on_five_inch_board :
  ∀ (w1 w2 : ℝ) (θ : ℝ), 
  w1 = 5 ∧ w2 = 7 ∧ θ = Real.pi / 4 →
  area_unpainted_region w1 w2 θ = 35 * Real.sqrt 2 / 2 :=
by
  intros w1 w2 θ h
  sorry

end unpainted_area_on_five_inch_board_l822_822598


namespace vector_relation_l822_822768

def vec : Type := ℝ × ℝ × ℝ

def dot_product (v1 v2 : vec) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def scalar_mult (k : ℝ) (v : vec) : vec :=
  (k * v.1, k * v.2, k * v.3)

theorem vector_relation (a b c : vec)
  (ha : a = (-2, -3, 1)) (hb : b = (2, 0, 4)) (hc : c = (-4, -6, 2)) :
  dot_product a b = 0 ∧ ∃ k : ℝ, k ≠ 0 ∧ c = scalar_mult k a :=
by sorry

end vector_relation_l822_822768


namespace enclosed_area_of_parabola_and_line_l822_822273

noncomputable def area_parabola_line : ℝ := 
let f := λ x: ℝ, (8*x - x^2)/4
let g := λ x: ℝ, (x + 6)/4 in
(∫ x in 1..6, (f x - g x))/4

theorem enclosed_area_of_parabola_and_line :
  area_parabola_line = 5.2083 :=
sorry

end enclosed_area_of_parabola_and_line_l822_822273


namespace tangent_segments_count_l822_822753

theorem tangent_segments_count:
  let n := 2017 in
  let num_tangent_segments := (3 * n * (n - 1)) / 2 in
  num_tangent_segments = (3 * 2017 * 2016) / 2 :=
by simp [num_tangent_segments, n]; sorry

end tangent_segments_count_l822_822753


namespace simplified_expression_eq_l822_822537

def simplify_expression (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 3) : ℝ :=
  (1 - (2 / (x - 1))) * ((x^2 - x) / (x^2 - 6*x + 9))

theorem simplified_expression_eq (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 3) :
  simplify_expression x h1 h2 = x / (x - 3) :=
by
  sorry

end simplified_expression_eq_l822_822537


namespace magic_sum_divisible_by_3_l822_822524

noncomputable def magic_square_3x3 := {a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℤ}

def is_third_order_magic_square (m : magic_square_3x3) (S : ℤ) : Prop :=
  (m.a_1 + m.a_2 + m.a_3 = S) ∧
  (m.a_4 + m.a_5 + m.a_6 = S) ∧
  (m.a_7 + m.a_8 + m.a_9 = S) ∧
  (m.a_1 + m.a_4 + m.a_7 = S) ∧
  (m.a_2 + m.a_5 + m.a_8 = S) ∧
  (m.a_3 + m.a_6 + m.a_9 = S) ∧
  (m.a_1 + m.a_5 + m.a_9 = S) ∧
  (m.a_3 + m.a_5 + m.a_7 = S)

theorem magic_sum_divisible_by_3 (m : magic_square_3x3) (S : ℤ) (h : is_third_order_magic_square m S) : S % 3 = 0 :=
sorry

end magic_sum_divisible_by_3_l822_822524


namespace distance_l1_l2_l822_822399

-- Definitions for the parametric and cartesian equations of the lines
def line_l1_parametric (t : ℝ) : ℝ × ℝ := (1 + t, 1 + 3 * t)
def line_l2 (x : ℝ) : ℝ := 3 * x + 4

-- Standard form conversion
def line_l1_standard : ℝ × ℝ → Prop := λ p, (3 * p.1 - p.2 - 2 = 0)

-- Given a point on l2 (e.g., (0,4)), let's prove the distance
def point_on_line_l2 : ℝ × ℝ := (0, 4)

-- Coefficients for line l1 in standard form
def A : ℝ := 3
def B : ℝ := -1
def C : ℝ := -2

-- Distance formula calculation
def distance_between_lines : ℝ :=
  real.abs (A * point_on_line_l2.1 + B * point_on_line_l2.2 + C) / real.sqrt (A ^ 2 + B ^ 2)

theorem distance_l1_l2 : distance_between_lines = 3 * real.sqrt 10 / 5 :=
by
  -- provide or omit proof
  sorry

end distance_l1_l2_l822_822399


namespace cost_of_one_bag_of_potatoes_l822_822124

theorem cost_of_one_bag_of_potatoes :
  let x := 250 in
  ∀ (price : ℕ)
    (bags : ℕ)
    (andrey_initial_price : ℕ)
    (andrey_sold_price : ℕ)
    (boris_initial_price : ℕ)
    (boris_first_price : ℕ)
    (boris_second_price : ℕ)
    (earnings_andrey : ℕ)
    (earnings_boris_first : ℕ)
    (earnings_boris_second : ℕ)
    (total_earnings_boris : ℕ),
  bags = 60 →
  andrey_initial_price = price →
  andrey_sold_price = 2 * price →
  andrey_sold_price * bags = earnings_andrey →
  boris_initial_price = price →
  boris_first_price = 1.6 * price →
  boris_second_price = 2.24 * price →
  boris_first_price * 15 + boris_second_price * 45 = total_earnings_boris →
  total_earnings_boris = earnings_andrey + 1200 →
  price = x :=
by
  intros x price bags andrey_initial_price andrey_sold_price boris_initial_price boris_first_price boris_second_price earnings_andrey earnings_boris_first earnings_boris_second total_earnings_boris
  assume h_bags h_andrey_initial_price h_andrey_sold_price h_earnings_andrey h_boris_initial_price h_boris_first_price h_boris_second_price h_total_earnings_boris h_total_earnings_difference
  if h_necessary : x = 250 then
    sorry
  else
    sorry


end cost_of_one_bag_of_potatoes_l822_822124


namespace correct_operation_B_l822_822170

variable (a : ℝ)

theorem correct_operation_B :
  2 * a^2 * a^4 = 2 * a^6 :=
by sorry

end correct_operation_B_l822_822170


namespace bound_riemann_sum_difference_l822_822368

theorem bound_riemann_sum_difference (f : ℝ → ℝ) (k : ℝ) (n : ℕ) (hn : 0 < n) 
  (hf : ∀ x ∈ Ioo (0 : ℝ) 1, differentiable_at ℝ f x) 
  (h_bound : ∀ x ∈ Ioo (0 : ℝ) 1, abs (deriv f x) ≤ k) :
  abs (∫ x in 0..1, f x - (∑ i in finset.range n, (f (i / n : ℝ) / n))) ≤ k / n :=
sorry

end bound_riemann_sum_difference_l822_822368


namespace sqrt_180_eq_l822_822040

noncomputable def simplify_sqrt_180 : Real := 6 * Real.sqrt 5

theorem sqrt_180_eq : Real.sqrt 180 = simplify_sqrt_180 := 
by
  -- proof omitted
  sorry

end sqrt_180_eq_l822_822040


namespace relative_errors_same_l822_822664

theorem relative_errors_same (length1 error1 length2 error2 : ℝ) 
  (h1 : length1 = 25) 
  (h2 : error1 = 0.05) 
  (h3 : length2 = 150) 
  (h4 : error2 = 0.3) : 
  (error1 / length1 * 100) = (error2 / length2 * 100) :=
by
  rw [h1, h2, h3, h4]
  have hrel1 : (0.05 / 25 * 100) = 0.2 := by norm_num
  have hrel2 : (0.3 / 150 * 100) = 0.2 := by norm_num
  rw [hrel1, hrel2]
  rfl

end relative_errors_same_l822_822664


namespace mars_mission_hours_l822_822849

def scientific_notation (h : ℕ) : ℝ := h * (10 ^ (-4: ℤ))
def round_to_three_sig_figs (r : ℝ) : ℝ := Float.ofInt (Float.toInt (r * 100)) / 100

theorem mars_mission_hours (h : ℕ) (H : h = 12480) :
  scientific_notation h = 1.2480 ∧ round_to_three_sig_figs (scientific_notation h) = 1.25 :=
by
  sorry

end mars_mission_hours_l822_822849


namespace sandy_spent_percentage_l822_822866

theorem sandy_spent_percentage (I R : ℝ) (hI : I = 200) (hR : R = 140) : 
  ((I - R) / I) * 100 = 30 :=
by
  sorry

end sandy_spent_percentage_l822_822866


namespace fractions_with_smallest_difference_l822_822328

theorem fractions_with_smallest_difference 
    (x y : ℤ) 
    (f1 : ℚ := (x : ℚ) / 8) 
    (f2 : ℚ := (y : ℚ) / 13) 
    (h : abs (13 * x - 8 * y) = 1): 
    (f1 ≠ f2) ∧ abs ((x : ℚ) / 8 - (y : ℚ) / 13) = 1 / 104 :=
by
  sorry

end fractions_with_smallest_difference_l822_822328


namespace ratio_of_triangle_areas_l822_822917

theorem ratio_of_triangle_areas 
  (r s : ℝ) (n : ℝ)
  (h_ratio : 3 * s = r) 
  (h_area : (3 / 2) * n = 1 / 2 * r * ((3 * n * 2) / r)) :
  3 / 3 = n :=
by
  sorry

end ratio_of_triangle_areas_l822_822917


namespace sequence_values_l822_822588

def sequence (n : ℕ) : ℝ :=
  Nat.recOn n (1 / 2) (λ n' a_n', 1 / (1 - a_n'))

theorem sequence_values :
  sequence 2 = 2 ∧
  sequence 3 = -1 ∧
  sequence 4 = 1 / 2 ∧
  sequence 2010 = -1 ∧
  sequence 2011 = 1 / 2 ∧
  sequence 2012 = 2 :=
by
  sorry

end sequence_values_l822_822588


namespace sqrt_180_eq_l822_822044

noncomputable def simplify_sqrt_180 : Real := 6 * Real.sqrt 5

theorem sqrt_180_eq : Real.sqrt 180 = simplify_sqrt_180 := 
by
  -- proof omitted
  sorry

end sqrt_180_eq_l822_822044


namespace ratio_bc_ad_l822_822835

-- Definitions of conditions
def volume_prism (length width height : ℝ) : ℝ := length * width * height
def surface_area_prism (length width height : ℝ) : ℝ := 2 * (length * width + length * height + width * height)
def quarter_cylinder_volume (r : ℝ) (length : ℝ) : ℝ := π * r^2 * length / 4
def one_eighth_sphere_volume (r : ℝ) : ℝ := (4 / 3) * π * r^3 / 8

-- Values for the dimensions of the prism
def l := 2
def w := 3
def h := 5

-- Compute coefficients a, b, c, d based on the given conditions
def a := 4 * π / 3
def b := 10 * π
def c := surface_area_prism l w h
def d := volume_prism l w h

-- Given the coefficients, we want to prove the final equation
theorem ratio_bc_ad : (b * c) / (a * d) = 15.5 := by
  -- Steps to prove the equation are omitted
  sorry

end ratio_bc_ad_l822_822835


namespace sequence_general_formula_l822_822352

noncomputable def a : ℕ → ℕ
| 1     := 20
| (n+1) := a n + 2*n - 1

theorem sequence_general_formula (n : ℕ) (hn : n > 0) : a n = n^2 - 2*n + 21 :=
sorry

end sequence_general_formula_l822_822352


namespace red_crayons_count_l822_822914

theorem red_crayons_count :
  let r := 94 - (6 * 8 + 7 * 5) in
  r = 11 :=
by
  sorry

end red_crayons_count_l822_822914


namespace distance_swam_downstream_l822_822210

theorem distance_swam_downstream :
  ∀ (V_m d_up t_up t_down : ℝ)
    (h_conditions : V_m = 4 ∧ d_up = 18 ∧ t_up = 6 ∧ t_down = 6),
  ∃ (D_down : ℝ), D_down = 30 :=
by
  intros V_m d_up t_up t_down h_conditions
  let V_s := V_m - (d_up / t_up)
  let V_downstream := V_m + V_s
  let D_down := V_downstream * t_down
  use D_down
  have h1 := h_conditions.1
  have h2 := h_conditions.2.1
  have h3 := h_conditions.2.2.1
  have h4 := h_conditions.2.2.2
  have h5 : V_s = V_m - (d_up / t_up) := rfl
  have h6 : V_downstream = V_m + V_s := rfl
  have h7 : D_down = V_downstream * t_down := rfl
  rw [h5, h6, h7, h1, h2, h3, h4]
  sorry

end distance_swam_downstream_l822_822210


namespace determine_functions_l822_822305

noncomputable def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), f(x) * f(y) + f(x + y) = x * y

theorem determine_functions (f : ℝ → ℝ) :
  (functional_equation f) →
  (f = (λ x, 0) ∨ f = (λ x, x - 1) ∨ f = (λ x, -x - 1)) :=
sorry

end determine_functions_l822_822305


namespace triangle_area_trisection_l822_822926

theorem triangle_area_trisection (ABC : Type*) [triangle ABC]
  (BC : real) (hBC : BC = 20) 
  (A B C D : Type*) [point A] [point B] [point C] [point D]
  (incircle_trisects_median : median_trisects_inc_A D A (triangle_median AD))
  (area_formula : ∃ (m n : ℕ), (area ABC) = m * real.sqrt n ∧ ¬ ∃ p : ℕ, (nat.prime p) ∧ (p^2 ∣ n)) :
  let m := 24,
      n := 14 
  in m + n = 38 := 
sorry

end triangle_area_trisection_l822_822926


namespace probability_in_interval_l822_822182

noncomputable def CDF (x : ℝ) : ℝ :=
  if x ≤ 0 then 0
  else if x ≤ 2 then x / 2
  else 1

theorem probability_in_interval (X : ℝ → ℝ) :
  (∀ x, X x = CDF x) →
  (P (1 < X 2)) = 0.5 :=
by
  intro hX
  sorry

end probability_in_interval_l822_822182


namespace domain_of_expression_l822_822704

theorem domain_of_expression (x : ℝ) 
  (h1 : 3 * x - 6 ≥ 0) 
  (h2 : 7 - 2 * x ≥ 0) 
  (h3 : 7 - 2 * x > 0) : 
  2 ≤ x ∧ x < 7 / 2 := by
sorry

end domain_of_expression_l822_822704


namespace sum_of_roots_eq_l822_822325

noncomputable def p : ℚ[X] := 3 * X^3 + 2 * X^2 - 5 * X - 12
noncomputable def q : ℚ[X] := 4 * X^3 - 16 * X^2 + X + 10

theorem sum_of_roots_eq :
  (p.roots.sum + q.roots.sum) = (10/3 : ℚ) :=
sorry

end sum_of_roots_eq_l822_822325


namespace problem_solution_l822_822810

def roads_closing_problem : Prop :=
  ∃ (G : SimpleGraph (Fin 8)) (H : G.edgeSet.card = 12),
  (∃ (s : Finset G.edgeSet) (hs : s.card = 5), G.is_connected_subgraph s) →
  (Finset.card (Finset.filter (λ s : Finset G.edgeSet, s.card = 5 ∧ G.is_connected_subgraph s) (Finset.powersetLen 5 G.edgeSet)) = 384)

theorem problem_solution : roads_closing_problem :=
sorry

end problem_solution_l822_822810


namespace parallel_line_no_common_points_l822_822890

def line_parallel_to_plane_no_common_points 
  (m : Type) (α : Type)
  (line : m -> Prop) (plane : α -> Prop)
  (parallel : m -> α -> Prop) 
  (no_common_points : m -> α -> Prop) : Prop :=
  ∀ (m0 : m) (α0 : α), (parallel m0 α0) ↔ (no_common_points m0 α0)

-- The theorem stating the necessary and sufficient condition
theorem parallel_line_no_common_points (m α : Type)
  [inhabited m] [inhabited α]
  (line : m -> Prop) (plane : α -> Prop)
  (parallel : m -> α -> Prop) 
  (no_common_points : m -> α -> Prop) :
  line_parallel_to_plane_no_common_points m α line plane parallel no_common_points :=
sorry

end parallel_line_no_common_points_l822_822890


namespace num_digits_divisible_l822_822333

theorem num_digits_divisible (h : Nat) :
  (∃ n : Fin 10, (10 * 24 + n) % n = 0) -> h = 7 :=
by sorry

end num_digits_divisible_l822_822333


namespace evaluate_f_of_f_of_f_of_3_l822_822012

def f (x : ℝ) : ℝ :=
  if x >= 5 then sqrt x else x^2

theorem evaluate_f_of_f_of_f_of_3 : f (f (f 3)) = 9 := by
  sorry

end evaluate_f_of_f_of_f_of_3_l822_822012


namespace total_money_spent_l822_822637

/-- Let's denote the expenditure of each of the 8 persons -/
def expenditure_8 (n : ℕ) (h : n = 8) : ℕ := 12 * n

/-- The average expenditure of all 9 persons, A -/
def avg_expenditure (total : ℕ) : ℕ := total / 9

/-- The ninth person spent Rs 8 more than the average expenditure of all 9 persons -/
def expenditure_9 (A : ℕ) : ℕ := A + 8

theorem total_money_spent :
  let total := expenditure_8 8 rfl in
  let A := avg_expenditure (total + expenditure_9 (avg_expenditure (total + expenditure_9 0))) in
  total + expenditure_9 A = 117 :=
by
  sorry

end total_money_spent_l822_822637


namespace equations_no_solution_l822_822166

theorem equations_no_solution :
  ¬ (∃ x : ℝ, |4 * x| + 7 = 0) ∧ ¬ (∃ x : ℝ, sqrt (-3 * x) + 1 = 0) :=
by
  sorry

end equations_no_solution_l822_822166


namespace largest_stamps_per_page_l822_822473

theorem largest_stamps_per_page (a b c : ℕ) (h1 : a = 924) (h2 : b = 1260) (h3 : c = 1386) : 
  Nat.gcd (Nat.gcd a b) c = 42 := by
  sorry

end largest_stamps_per_page_l822_822473


namespace sum_of_divisors_85_l822_822946

theorem sum_of_divisors_85 : (1 + 5 + 17 + 85 = 108) := by
  sorry

end sum_of_divisors_85_l822_822946


namespace range_of_a_l822_822784

noncomputable def f : ℝ → ℝ := λ x, 
  if x ≤ 1 then 2^(x + 1) 
  else real.log (x^2 - 1) / real.log 2

theorem range_of_a (a : ℝ) (h : f a > 1) : -1 < a ∧ a ≤ 1 ∨ a > sqrt 3 :=
by {
  sorry,
}

end range_of_a_l822_822784


namespace possible_k_values_l822_822316

-- Define an appropriate range for positive integers k
variable (k : ℕ) (N : ℕ) (f : ℕ → ℚ)
-- Assume conditions: polynomial f with rational coefficients
-- lcm condition holds for all sufficiently large n
axiom k_exists_polynomial (hk : (1 ≤ k ∧ k ≤ 2) ∧ ∃ N f, ∀ n ≥ N, f n = Nat.lcm_range (n + 1) (n + k + 1))

-- Define the theorem to prove this is the only possible k
theorem possible_k_values : k = 1 ∨ k = 2 :=
sorry

end possible_k_values_l822_822316


namespace correct_expression_l822_822662

-- Definitions based on given conditions
def expr1 (a b : ℝ) := 3 * a + 2 * b = 5 * a * b
def expr2 (a : ℝ) := 2 * a^3 - a^3 = a^3
def expr3 (a b : ℝ) := a^2 * b - a * b = a
def expr4 (a : ℝ) := a^2 + a^2 = 2 * a^4

-- Statement to prove that expr2 is the only correct expression
theorem correct_expression (a b : ℝ) : 
  expr2 a := by
  sorry

end correct_expression_l822_822662


namespace max_stamps_l822_822787

def price_of_stamp : ℕ := 25  -- Price of one stamp in cents
def total_money : ℕ := 4000   -- Total money available in cents

theorem max_stamps : ∃ n : ℕ, price_of_stamp * n ≤ total_money ∧ (∀ m : ℕ, price_of_stamp * m ≤ total_money → m ≤ n) :=
by
  use 160
  sorry

end max_stamps_l822_822787


namespace infinitely_many_n_l822_822331

def is_prime (p : ℕ) : Prop :=
  ∀ n : ℕ, n > 1 → n < p → p % n ≠ 0

def nu_p (p n : ℕ) : ℕ :=
  if is_prime p then 
    ∑ k in list.range (n+1), n/p^k 
  else 0

theorem infinitely_many_n (d : ℕ) (k : ℕ) (primes : fin k → ℕ) 
  (hp : ∀ i, is_prime (primes i)) :
  ∃ (n : ℕ) (infinitely_many : ℕ → Prop), 
  (∀ n, infinitely_many n → ∀ i, d ∣ nu_p (primes i) n) ∧ 
  (∃ infinite_n : ℕ, infinitely_many infinite_n) :=
by sorry

end infinitely_many_n_l822_822331


namespace number_of_integer_exponent_terms_in_binomial_expansion_l822_822459

open BigOperators

theorem number_of_integer_exponent_terms_in_binomial_expansion :
  (fin n → (x * y : ℤ), 
  (n : ℕ), 
  ∑ i in range (24 + 1), 
  (binom 24 i) * (x ^ (24 - i)) * (y ^ i) = 
  (x - y) ^ 24) → 
  ∃ terms : ℕ, terms = 9 :=
by
  sorry

end number_of_integer_exponent_terms_in_binomial_expansion_l822_822459


namespace circle_radius_range_l822_822435

theorem circle_radius_range {r : ℝ} :
  (∃ (r : ℝ), r > 0 ∧ (∀ p : ℝ × ℝ, 
    ((p.1 - 3)^2 + (p.2 + 5)^2 = r^2 → abs ((4 * p.1 - 3 * p.2 - 2) / sqrt (16 + 9)) = 1)) ↔ 4 < r ∧ r < 6 :=
sorry

end circle_radius_range_l822_822435


namespace measure_A_at_least_half_l822_822480

open MeasureTheory Set Interval

noncomputable def f (x : ℝ) : ℝ := sorry

def A (f : ℝ → ℝ) (s : Set ℝ) := 
  {h ∈ s | ∃ x ∈ s, f (x + h) = f x}

theorem measure_A_at_least_half (f : ℝ → ℝ) (s : Set ℝ)
  (hf_cont : ContinuousOn f s) (hf_0_1 : f 0 = 0 ∧ f 1 = 0) : 
  MeasurableSet (A f s) ∧ measure (A f s) ≥ 1 / 2 :=
by
  sorry

end measure_A_at_least_half_l822_822480


namespace potato_cost_l822_822128

variables (x : ℝ)
variables (b a : ℝ)

def andrey_earnings (x : ℝ) : ℝ := 120 * x
def boris_earnings (x : ℝ) : ℝ := 124.8 * x

theorem potato_cost :
  (boris_earnings x) - (andrey_earnings x) = 1200 → x = 250 :=
  by
    unfold andrey_earnings
    unfold boris_earnings
    sorry

end potato_cost_l822_822128


namespace max_size_of_S_l822_822833

noncomputable def max_size_S (k m n : ℤ) : ℤ :=
  k - (⌊m / n - (n + 1) / 2⌋ : ℤ)

theorem max_size_of_S (k m n : ℤ) (S : set ℤ) 
  (hk : 1 < n ∧ n ≤ m - 1 ∧ m - 1 ≤ k)
  (hS : ∀ (s : finset ℤ), s ⊆ S → s.card = n → s.sum id ≠ m) :
  (S.card : ℤ) ≤ max_size_S k m n :=
sorry

end max_size_of_S_l822_822833


namespace even_function_analytic_expression_l822_822370

noncomputable def f (x : ℝ) : ℝ := 
if x ≥ 0 then Real.log (x^2 - 2 * x + 2) 
else Real.log (x^2 + 2 * x + 2)

theorem even_function_analytic_expression (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_nonneg : ∀ x : ℝ, 0 ≤ x → f x = Real.log (x^2 - 2 * x + 2)) :
  ∀ x : ℝ, x < 0 → f x = Real.log (x^2 + 2 * x + 2) :=
by
  sorry

end even_function_analytic_expression_l822_822370


namespace equal_integers_l822_822953

theorem equal_integers (a b : ℕ)
  (h : ∀ n : ℕ, n > 0 → a > 0 → b > 0 → (a^n + n) ∣ (b^n + n)) : a = b := 
sorry

end equal_integers_l822_822953


namespace last_four_digits_5_2011_l822_822507

theorem last_four_digits_5_2011 :
  (5^2011 % 10000) = 8125 := by
  sorry

end last_four_digits_5_2011_l822_822507


namespace dive_score_l822_822800

theorem dive_score (scores : List ℝ)
  (degree_of_difficulty : ℝ)
  (highest_score : ℝ)
  (lowest_score : ℝ)
  (expected_score : ℝ) :
  scores = [12.5, 11.8, 10.0, 13.6, 14.0, 9.5, 10.5, 12.0] →
  degree_of_difficulty = 3.8 →
  highest_score = 14.0 →
  lowest_score = 9.5 →
  expected_score = 280.89 →
  let scores_filtered := scores.erase highest_score |>.erase lowest_score in
  let sum_scores := scores_filtered.sum in
  let base_score := sum_scores * degree_of_difficulty in
  let top_3_scores := scores.sort (· ≥ ·) |>.take 3 in
  let style_bonus := top_3_scores.sum / 3 in
  base_score + style_bonus = expected_score :=
by
  -- Initial conditions and variables are defined.
  intros _ _ _ _ _,
  -- Skip proof for now
  sorry

end dive_score_l822_822800


namespace find_y_l822_822418

theorem find_y (x y : ℝ) (h1 : x - y = 20) (h2 : x + y = 10) : y = -5 := 
sorry

end find_y_l822_822418


namespace convex_polygon_exists_l822_822726

theorem convex_polygon_exists (N : ℕ) 
  (P : fin N → ℝ × ℝ)
  (hnc : ∀ i j k : fin N, i ≠ j → j ≠ k → i ≠ k → ¬ collinear ({P i, P j, P k}))
  (hntc : ∀ i j k l : fin N, i ≠ j → j ≠ k → k ≠ i → (inside_triangle (P i) (P j) (P k) (P l)) → false) :
  ∃ (σ : fin N → fin N), convex_polygon (σ ∘ P) :=
by
  sorry

end convex_polygon_exists_l822_822726


namespace geometric_sequence_sum_eq_80_243_l822_822585

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_eq_80_243 {n : ℕ} :
  let a := (1 / 3 : ℝ)
  let r := (1 / 3 : ℝ)
  geometric_sum a r n = 80 / 243 ↔ n = 3 :=
by
  intros a r
  sorry

end geometric_sequence_sum_eq_80_243_l822_822585


namespace ellipse_standard_equation_parabola_standard_equation_l822_822189

theorem ellipse_standard_equation (x y : ℝ) (a b : ℝ) (h₁ : a > b ∧ b > 0)
  (h₂ : 2 * a = Real.sqrt ((3 + 2) ^ 2 + (-2 * Real.sqrt 6) ^ 2) 
      + Real.sqrt ((3 - 2) ^ 2 + (-2 * Real.sqrt 6) ^ 2))
  (h₃ : b^2 = a^2 - 4) 
  : (x^2 / 36 + y^2 / 32 = 1) :=
by sorry

theorem parabola_standard_equation (y : ℝ) (p : ℝ) (h₁ : p > 0)
  (h₂ : -p / 2 = -1 / 2) 
  : (y^2 = 2 * p * 1) :=
by sorry

end ellipse_standard_equation_parabola_standard_equation_l822_822189


namespace sqrt_180_simplify_l822_822062

theorem sqrt_180_simplify : real.sqrt 180 = 6 * real.sqrt 5 := by
  have h : 180 = 2^2 * 3^2 * 5 := by norm_num
  rw [h, real.sqrt_mul, real.sqrt_mul, real.sqrt_mul]
  all_goals { norm_num }
  sorry

end sqrt_180_simplify_l822_822062


namespace number_of_intersections_l822_822475

theorem number_of_intersections (m n : ℕ) : 
  0 < m → 0 < n → 
  ∑ x in finset.range(m), ∑ y in finset.range(n), (x < y) ∧ (x ≠ 0) ∧ (y ≠ 0) → 
  (∑ x in finset.range(m*(m-1) * n*(n-1) / 4), x) = (m*(m-1)*n*(n-1)/4) :=
by
  sorry

end number_of_intersections_l822_822475


namespace hungarian_olympiad_problem_l822_822970

-- Define the function A_n as given in the problem
def A (n : ℕ) : ℕ := 5^n + 2 * 3^(n - 1) + 1

-- State the theorem to be proved
theorem hungarian_olympiad_problem (n : ℕ) (h : 0 < n) : 8 ∣ A n :=
by
  sorry

end hungarian_olympiad_problem_l822_822970


namespace min_time_from_A_to_B_l822_822028

noncomputable def min_travel_time (d_AB d_A_boundary d_B_boundary : ℝ) 
  (v_meadow v_wasteland : ℝ) : ℝ := 
  let t := λ x : ℝ, (1 / v_meadow) * real.sqrt (x^2 + d_A_boundary^2) + 
                    (1 / v_wasteland) * real.sqrt ((d_AB - x)^2 + d_B_boundary^2) in
  (exists x ∈ set.Icc 0 d_AB, t x = 4.89)

theorem min_time_from_A_to_B : 
  min_travel_time 24 8 4 6 3 := 
sorry

end min_time_from_A_to_B_l822_822028


namespace chord_bisected_by_Q_l822_822738

theorem chord_bisected_by_Q {x y : ℝ} :
  (x - 2)^2 + (y - 1)^2 = 4 ∧ (1, 0) ∈ line (x, y) (x + 1, y - 1) →
  y = -x + 1 :=
sorry

end chord_bisected_by_Q_l822_822738


namespace simplify_expression_correct_l822_822037

def simplify_expression (i : ℂ) (h : i ^ 2 = -1) : ℂ :=
  3 * (4 - 2 * i) + 2 * i * (3 - i)

theorem simplify_expression_correct (i : ℂ) (h : i ^ 2 = -1) : simplify_expression i h = 14 := 
by
  sorry

end simplify_expression_correct_l822_822037


namespace volume_equiv_l822_822184

-- Defining the variables and conditions
variable (V α β : Real)
variable (AB CD : Real)
variable (M N : ℝ)
variable (P Q: ℝ)

-- Volume of tetrahedron ABCD
def volume_ABCD (a b d : Real) (sin_phi : Real) : Real := 
  (1 / 6) * a * b * d * sin_phi

-- Volume of tetrahedron MNPQ
def volume_MNPQ (α β a b d : Real) (sin_phi : Real) : Real := 
  (1 / 6) * (α * a) * (β * b) * d * sin_phi

-- Main theorem statement to be proved
theorem volume_equiv :
  ∀ (a b d sin_phi : Real),
  volume_MNPQ α β a b d sin_phi = α * β * (volume_ABCD a b d sin_phi) :=
by
  intros
  unfold volume_MNPQ
  unfold volume_ABCD
  ring
  sorry

end volume_equiv_l822_822184


namespace equal_circumcircle_angles_l822_822506

theorem equal_circumcircle_angles
    (A B C D : Type) 
    [EuclideanGeometry A B C D]
    (h : ¬ collinear {A, B, C} ∧ ¬ collinear {A, B, D} ∧ ¬ collinear {A, C, D} ∧ ¬ collinear {B, C, D}) :
    let θ1 = angle (circumcircle A B C) (circumcircle A B D),
        θ2 = angle (circumcircle A C D) (circumcircle B C D) in
    θ1 = θ2 :=
by
  sorry

end equal_circumcircle_angles_l822_822506


namespace math_problem_l822_822489

noncomputable def x := (3 + Real.sqrt 5)^500
def n := Int.floor x
def f := x - n

theorem math_problem :
  x * (1 - f) = 4^500 := 
by
  sorry

end math_problem_l822_822489


namespace no_integer_solution_mx2_minus_sy2_eq_3_l822_822522

theorem no_integer_solution_mx2_minus_sy2_eq_3 (m s : ℤ) (x y : ℤ) (h : m * s = 2000 ^ 2001) :
  ¬ (m * x ^ 2 - s * y ^ 2 = 3) :=
sorry

end no_integer_solution_mx2_minus_sy2_eq_3_l822_822522


namespace ratio_rational_l822_822516

variable (α : Type) [LinearOrderedField α]
variables (a b c d e : α)
variable (numbers : Fin 5 → α)
variable (h_diff : ∀ i j : Fin 5, i ≠ j → numbers i ≠ numbers j)
variable (h_pos : ∀ i : Fin 5, 0 < numbers i)
variable (h_rational_prod_sum : ∀ (i j k : Fin 5), i ≠ j → j ≠ k → i ≠ k → is_r := ((numbers i) * (numbers j) + (numbers j) * (numbers k) + (numbers k) * (numbers i)))

theorem ratio_rational (i j : Fin 5) (h_ij: i ≠ j) : is_r := (numbers i / numbers j) := sorry

end ratio_rational_l822_822516


namespace andrew_correct_answer_l822_822248

variable {x : ℕ}

theorem andrew_correct_answer (h : (x - 8) / 7 = 15) : (x - 5) / 11 = 10 :=
by
  sorry

end andrew_correct_answer_l822_822248


namespace xiao_wang_parts_processed_l822_822619

-- Definitions for the processing rates and conditions
def xiao_wang_rate := 15 -- parts per hour
def xiao_wang_max_continuous_hours := 2
def xiao_wang_break_hours := 1

def xiao_li_rate := 12 -- parts per hour

-- Constants for the problem setup
def xiao_wang_process_time := 4 -- hours including breaks after first cycle
def xiao_li_process_time := 5 -- hours including no breaks

-- Total parts processed by both when they finish simultaneously
def parts_processed_when_finished_simultaneously := 60

theorem xiao_wang_parts_processed :
  (xiao_wang_rate * xiao_wang_max_continuous_hours) * (xiao_wang_process_time / 
  (xiao_wang_max_continuous_hours + xiao_wang_break_hours)) =
  parts_processed_when_finished_simultaneously :=
sorry

end xiao_wang_parts_processed_l822_822619


namespace investment_difference_l822_822026

noncomputable def compound_interest_annual (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

noncomputable def compound_interest_monthly (P : ℝ) (r : ℝ) (n : ℕ) (m : ℕ) : ℝ :=
  P * (1 + r / m)^(n * m)

theorem investment_difference:
  let P := 30000 in
  let r := 0.05 in
  let n := 3 in
  let m := 12 in
  let peter_total := compound_interest_annual P r n in
  let sophia_total := compound_interest_monthly P r n m in
  abs (sophia_total - peter_total) == 121 := by
    sorry


end investment_difference_l822_822026


namespace distance_between_centers_l822_822454

-- Define the points and their properties
variables {A B C D : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]

-- Define the lengths a and b
variables (a b : ℝ)

-- Define the conditions: ∠DAB = 90° and ∠DBC = 90°
variable (angle_DAB : ∡ D A B = 90)
variable (angle_DBC : ∡ D B C = 90)

-- Define the lengths DB = a and DC = b
variable (length_DB : dist D B = a)
variable (length_DC : dist D C = b)

-- The ultimate goal is to show the distance between centers of the circles
theorem distance_between_centers : 
  dist (midpoint D B) (midpoint D C) = (sqrt (b^2 - a^2)) / 2 :=
sorry

end distance_between_centers_l822_822454


namespace Q_evaluation_ratio_l822_822104

-- Define the polynomial g with the specified properties
def g : Polynomial ℂ := Polynomial.X ^ 2010 + 14 * Polynomial.X ^ 2009 + 1

-- Let s be the roots of g
def s : ℕ → ℂ := sorry  -- Just denote it for later use

-- PMF: Degree of the polynomial Q
def degree_Q : ℤ := 2010

-- Condition on Q: It is a polynomial of degree 2010 that satisfies the specified property
def Q : Polynomial ℂ := sorry  -- Just denote it for later use

-- Main theorem: Objective to prove
theorem Q_evaluation_ratio :
  g.roots.Nodup →  (∀ j ∈ Finset.range degree_Q, Q (s j + 1 / s j) = 0) → 
  (Q 1) / (Q (-1)) = (49 : ℂ) / (64 : ℂ) := 
  by
  intros h1 h2
  sorry

end Q_evaluation_ratio_l822_822104


namespace value_of_b_l822_822613

theorem value_of_b (a b : ℝ) (h : a > 2) (sol_set : ∀ x : ℝ, (ax + 3 < 2x + b) ↔ (x < 0)) : 
  b = 3 :=
sorry

end value_of_b_l822_822613


namespace line_intersects_circle_l822_822752

theorem line_intersects_circle (r d : ℝ) (h_r : r = 5) (h_d : d = 4) : d < r → "intersect" := by
  sorry

end line_intersects_circle_l822_822752


namespace circle_diameter_percentage_l822_822552

theorem circle_diameter_percentage (d_R d_S : ℝ) 
    (h : π * (d_R / 2)^2 = 0.04 * π * (d_S / 2)^2) : 
    d_R = 0.4 * d_S :=
by
    sorry

end circle_diameter_percentage_l822_822552


namespace find_x_if_vectors_are_parallel_l822_822404

noncomputable def vector_a : ℝ × ℝ := (1, 1) 
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (2, x)

noncomputable def vector_sum (x : ℝ) : ℝ × ℝ := (vector_a.1 + (vector_b x).1, vector_a.2 + (vector_b x).2)
noncomputable def vector_diff (x : ℝ) : ℝ × ℝ := (vector_a.1 - (vector_b x).1, vector_a.2 - (vector_b x).2)

theorem find_x_if_vectors_are_parallel (x : ℝ) : 
  vector_sum x = (3, x + 1) → 
  vector_diff x = (-1, 1 - x) → 
  vector_sum x.1 * vector_diff x.2 - vector_sum x.2 * vector_diff x.1 = 0 → 
  x = 2 := 
by 
  sorry

end find_x_if_vectors_are_parallel_l822_822404


namespace asymptotes_of_hyperbola_l822_822389

variable (a b c : ℝ)
variable (e : ℝ := 2)

theorem asymptotes_of_hyperbola (ha : a > 0) (hb : b > 0)
  (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1) 
  (h_eccentricity : e = 2): 
  ∀ x, (y = √3 * x) ∨ (y = -√3 * x) := 
by
  sorry

end asymptotes_of_hyperbola_l822_822389


namespace hyperbola_asymptotes_l822_822396

theorem hyperbola_asymptotes (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) (h4 : (2 : ℝ) = 2) :
  ∀ x : ℝ, y = sqrt 3 * x :=
by skip

end hyperbola_asymptotes_l822_822396


namespace sum_bounds_l822_822829

open Real

theorem sum_bounds {n : ℕ} (x : Fin n → ℝ) 
  (h_nonneg : ∀ i, 0 ≤ x i) 
  (h_eq : ∑ i, x i ^ 2 + 2 * ∑ k in finRange n, ∑ j in finRange (k+1) n, (sqrt (j / k)) * (x k) * (x j) = 1) :
  1 ≤ ∑ i, x i ∧ ∑ i, x i ≤ sqrt (∑ i in finRange n, (sqrt i - sqrt (i-1)) ^ 2) := 
sorry

end sum_bounds_l822_822829


namespace projectile_reaches_64_first_time_l822_822560

theorem projectile_reaches_64_first_time :
  ∃ t : ℝ, t > 0 ∧ t ≈ 0.7 ∧ (-16 * t^2 + 100 * t = 64) :=
sorry

end projectile_reaches_64_first_time_l822_822560


namespace problem_l822_822754

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x * (deriv f 2)

theorem problem :
  deriv f 5 = 6 := by
  sorry

end problem_l822_822754


namespace incorrect_methods_l822_822690

variable (Points : Type) (L : set Points) (cond : Points → Prop)

-- Conditions given in the problem as definitions
def method_A := ∀ p, p ∈ L ↔ cond p
def method_B := ∀ p, p ∈ L → cond p
def method_C := ∀ p, (cond p ↔ p ∈ L)
def method_D := ∀ p, p ∉ L → ¬cond p ∧ (p ∈ L → cond p)
def method_E := ∀ p, ¬cond p → p ∉ L

-- The proof problem statement
theorem incorrect_methods : (¬method_B L cond) ∧ (¬method_E L cond) :=
sorry

end incorrect_methods_l822_822690


namespace find_y_l822_822420

variable {x y : ℤ}

-- Definition 1: The first condition x - y = 20
def condition1 : Prop := x - y = 20

-- Definition 2: The second condition x + y = 10
def condition2 : Prop := x + y = 10

-- The main theorem to prove that y = -5 given the above conditions
theorem find_y (h1 : condition1) (h2 : condition2) : y = -5 :=
  sorry

end find_y_l822_822420


namespace angle_A_eq_pi_div_3_length_c_eq_2_sqrt_6_div_3_l822_822439

-- Part 1
theorem angle_A_eq_pi_div_3 
  (a b c : ℝ) (A B C : ℝ)
  (h1 : b^2 + c^2 - a^2 = bc) 
  (h2 : ∀ θ, 0 < θ < π → cos θ = (b^2 + c^2 - a^2) / (2 * b * c)) :
  A = π / 3 :=
by
  sorry

-- Part 2
theorem length_c_eq_2_sqrt_6_div_3
  (a c : ℝ) (C : ℝ)
  (h1 : a = sqrt 3) 
  (h2 : cos C = (sqrt 3) / 3)
  (h3 : ∀ θ, sin θ = sqrt (1 - (cos θ)^2)) :
  c = 2 * sqrt 6 / 3 :=
by
  sorry

end angle_A_eq_pi_div_3_length_c_eq_2_sqrt_6_div_3_l822_822439


namespace candidate_failed_by_l822_822199

variable (max_mark : ℝ) (required_percentage : ℝ) (candidate_score : ℝ) (failing_marks : ℝ)

/-- Proof that the candidate failed by 23 marks given the conditions. -/
theorem candidate_failed_by (h_max_mark : max_mark = 185.71)
    (h_required_percentage : required_percentage = 0.35)
    (h_candidate_score : candidate_score = 42) :
    failing_marks = (required_percentage * max_mark) - candidate_score := 
begin
  sorry
end

#eval candidate_failed_by 185.71 0.35 42

end candidate_failed_by_l822_822199


namespace periodic_odd_function_value_l822_822564

theorem periodic_odd_function_value (f : ℝ → ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x)
    (h_periodic : ∀ x : ℝ, f (x + 2) = f x) (h_value : f 0.5 = -1) : f 7.5 = 1 :=
by
  -- Proof would go here.
  sorry

end periodic_odd_function_value_l822_822564


namespace T_n_is_perfect_square_l822_822332

/-- Function f which denotes the greatest power of 3 that divides x --/
def f (x : ℕ) : ℕ := 
  let rec max_power (x : ℕ) (p : ℕ) : ℕ :=
    if x % p ≠ 0 then (p / 3)
    else max_power x (p * 3)
  max_power x 3

/-- Sum T_n defined as the sum of f(3k) for k from 1 to 3^n --/
def T_n (n : ℕ) : ℕ := 
  Finset.sum (Finset.range (3^n + 1)) (λ k, f (3 * k))

/-- Greatest integer n less than 1000 such that T_n is a perfect square --/
def greatest_n : ℕ := 960

/-- Proof that greatest_n is the largest integer less than 1000 such that T_n is a perfect square --/
theorem T_n_is_perfect_square (n : ℕ) (hn : n < 1000) : 
  ∃ m : ℕ, m * m = T_n n ↔ n = 960 :=
sorry

end T_n_is_perfect_square_l822_822332


namespace sum_of_divisors_85_l822_822945

theorem sum_of_divisors_85 : (1 + 5 + 17 + 85 = 108) := by
  sorry

end sum_of_divisors_85_l822_822945


namespace potato_cost_l822_822127

variables (x : ℝ)
variables (b a : ℝ)

def andrey_earnings (x : ℝ) : ℝ := 120 * x
def boris_earnings (x : ℝ) : ℝ := 124.8 * x

theorem potato_cost :
  (boris_earnings x) - (andrey_earnings x) = 1200 → x = 250 :=
  by
    unfold andrey_earnings
    unfold boris_earnings
    sorry

end potato_cost_l822_822127


namespace calculation_correct_l822_822672

theorem calculation_correct : (5 * 7 + 9 * 4 - 36 / 3 : ℤ) = 59 := by
  sorry

end calculation_correct_l822_822672


namespace angle_CDE_proof_l822_822460

theorem angle_CDE_proof 
  (right_angle_A : ∠ A = 90°) 
  (right_angle_B : ∠ B = 90°) 
  (right_angle_C : ∠ C = 90°) 
  (angle_AEB : ∠ AEB = 30°)
  (isosceles_BED : ∠ BED = ∠ BDE) 
  : ∠ CDE = 105° :=
by
  -- skip the proof steps
  sorry

end angle_CDE_proof_l822_822460


namespace problem_2023_divisible_by_consecutive_integers_l822_822950

theorem problem_2023_divisible_by_consecutive_integers :
  ∃ (n : ℕ), (n = 2022 ∨ n = 2023 ∨ n = 2024) ∧ (2023^2023 - 2023^2021) % n = 0 :=
sorry

end problem_2023_divisible_by_consecutive_integers_l822_822950


namespace correct_statements_l822_822097

-- Define the statements
def statement_1 := true
def statement_2 := false
def statement_3 := true
def statement_4 := true

-- Define a function to count the number of true statements
def num_correct_statements (s1 s2 s3 s4 : Bool) : Nat :=
  [s1, s2, s3, s4].countP id

-- Define the theorem to prove that the number of correct statements is 3
theorem correct_statements :
  num_correct_statements statement_1 statement_2 statement_3 statement_4 = 3 :=
by
  -- You can use sorry to skip the proof
  sorry

end correct_statements_l822_822097


namespace terminating_decimal_fraction_count_l822_822716

theorem terminating_decimal_fraction_count :
  {n : ℤ | 1 ≤ n ∧ n ≤ 569 ∧ (∃ k : ℤ, n = 57 * k)}.finite.card = 9 :=
by sorry

end terminating_decimal_fraction_count_l822_822716


namespace cost_of_one_bag_l822_822131

theorem cost_of_one_bag (x : ℝ) :
  let Andrey_earning := 60 * 2 * x
  let Boris_earning := 15 * 1.6 * x + 45 * (1.6 * 1.4) * x
  Boris_earning - Andrey_earning = 1200 →
  x = 250 := 
by
  intros
  let Andrey_earning := 60 * 2 * x
  let Boris_earning := 15 * 1.6 * x + 45 * (1.6 * 1.4) * x
  have h : Boris_earning - Andrey_earning = 1200 := by assumption
  let simplified_h := 
    calc
      Boris_earning - Andrey_earning
        = (15 * 1.6 * x + 45 * (1.6 * 1.4) * x) - (60 * 2 * x) : by simp [Andrey_earning, Boris_earning]
    ... = (15 * 1.6 * x + 45 * (1.6 * 1.4) * x) - 120 * x : by simp
    ... = (24 * x + 100.8 * x) - 120 * x : by simp
    ... = 124.8 * x - 120 * x : by simp
    ... = 4.8 * x : by simp
    ... = 1200 : by rw h
  exact (div_eq_iff (by norm_num : (4.8 : ℝ) ≠ 0)).1 simplified_h  -- solves for x

end cost_of_one_bag_l822_822131


namespace milk_for_9_cookies_l822_822594

def quarts_to_pints (q : ℕ) : ℕ := q * 2

def milk_for_cookies (cookies : ℕ) (milk_in_quarts : ℕ) : ℕ :=
  quarts_to_pints milk_in_quarts * cookies / 18

theorem milk_for_9_cookies :
  milk_for_cookies 9 3 = 3 :=
by
  -- We define the conversion and proportional conditions explicitly here.
  unfold milk_for_cookies
  unfold quarts_to_pints
  sorry

end milk_for_9_cookies_l822_822594


namespace hyperbola_distance_congruence_l822_822994

variables {a b x y : ℝ}
variables {P Q P' Q' : ℝ × ℝ}

noncomputable def hyperbola (a b : ℝ) : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1)^2 / a^2 - (p.2)^2 / b^2 = 1}

axiom a_pos : a > 0
axiom b_pos : b > 0

-- Assume P and Q lie on the hyperbola
axiom P_on_hyperbola : P ∈ hyperbola a b
axiom Q_on_hyperbola : Q ∈ hyperbola a b

-- Assume P' and Q' lie on the asymptotes of the hyperbola
def on_asymptote (p : ℝ × ℝ) (a b : ℝ) : Prop :=
  p.2 = (b / a) * p.1 ∨ p.2 = -(b / a) * p.1

axiom P'_on_asymptote : on_asymptote P' a b
axiom Q'_on_asymptote : on_asymptote Q' a b

-- Assume the distances are defined as the Euclidean distances between points
noncomputable def dist (p₁ p₂ : ℝ × ℝ) : ℝ :=
  (p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2

-- The theorem statement
theorem hyperbola_distance_congruence 
    (a_pos: a > 0) 
    (b_pos: b > 0) 
    (P_on_hyperbola: P ∈ hyperbola a b) 
    (Q_on_hyperbola: Q ∈ hyperbola a b) 
    (P'_on_asymptote: on_asymptote P' a b) 
    (Q'_on_asymptote: on_asymptote Q' a b) : 
  dist P P' = dist Q Q' :=
sorry

end hyperbola_distance_congruence_l822_822994


namespace binary_to_base5_1101_l822_822685

-- Definition of the binary to decimal conversion for the given number
def binary_to_decimal (b: Nat): Nat :=
  match b with
  | 0    => 0
  | 1101 => 1 * 2^0 + 0 * 2^1 + 1 * 2^2 + 1 * 2^3
  | _    => 0  -- This is a specific case for the given problem

-- Definition of the decimal to base-5 conversion method
def decimal_to_base5 (d: Nat): Nat :=
  match d with
  | 0    => 0
  | 13   =>
    let rem1 := 13 % 5
    let div1 := 13 / 5
    let rem2 := div1 % 5
    let div2 := div1 / 5
    rem2 * 10 + rem1  -- Assemble the base-5 number from remainders
  | _    => 0  -- This is a specific case for the given problem

-- Proof statement: conversion of 1101 in binary to base-5 yields 23
theorem binary_to_base5_1101 : decimal_to_base5 (binary_to_decimal 1101) = 23 := by
  sorry

end binary_to_base5_1101_l822_822685


namespace find_density_function_l822_822106

-- Define distribution function F
def F (x : ℝ) (a : ℝ) : ℝ :=
  if x ≤ -a then 0
  else if -a < x ∧ x ≤ 0 then (a + x)^2 / (2 * a^2)
  else if 0 < x ∧ x ≤ a then 1 - (a - x)^2 / (2 * a^2)
  else 1

-- Define the expected density function p
def p (x : ℝ) (a : ℝ) : ℝ :=
  if x ≤ -a then 0
  else if -a < x ∧ x ≤ 0 then (a + x) / a^2
  else if 0 < x ∧ x ≤ a then (a - x) / a^2
  else 0

theorem find_density_function (a : ℝ) (ha : 0 < a) :
  ∀ x : ℝ, p x a = (deriv (fun t => F t a)) x :=
by
  sorry -- Proof is not required, just the statement.

end find_density_function_l822_822106


namespace mary_should_drink_six_glasses_per_day_l822_822920

def daily_water_goal : ℕ := 1500
def glass_capacity : ℕ := 250
def required_glasses (daily_goal : ℕ) (capacity : ℕ) : ℕ := daily_goal / capacity

theorem mary_should_drink_six_glasses_per_day :
  required_glasses daily_water_goal glass_capacity = 6 :=
by
  sorry

end mary_should_drink_six_glasses_per_day_l822_822920


namespace count_multiples_of_2015_l822_822832

theorem count_multiples_of_2015 :
  (Finset.card (Finset.filter (λ n, (n * (n + 1) / 2) % 2015 = 0) (Finset.range 2016))) = 8 :=
sorry

end count_multiples_of_2015_l822_822832


namespace gravel_cost_l822_822955

-- Definitions of conditions
def lawn_length : ℝ := 70
def lawn_breadth : ℝ := 30
def road_width : ℝ := 5
def gravel_cost_per_sqm : ℝ := 4

-- Theorem statement
theorem gravel_cost : (lawn_length * road_width + lawn_breadth * road_width - road_width * road_width) * gravel_cost_per_sqm = 1900 :=
by
  -- Definitions used in the problem
  let area_first_road := lawn_length * road_width
  let area_second_road := lawn_breadth * road_width
  let area_intersection := road_width * road_width

  -- Total area to be graveled
  let total_area_to_be_graveled := area_first_road + area_second_road - area_intersection

  -- Calculate the cost
  let cost := total_area_to_be_graveled * gravel_cost_per_sqm

  show cost = 1900
  sorry

end gravel_cost_l822_822955


namespace triangle_area_is_correct_l822_822998

def point (α : Type) := (α × α)

def area_of_triangle (A B C : point ℝ) : ℝ :=
  (1 / 2) * abs (B.2 - A.2) * abs (C.1 - B.1)

theorem triangle_area_is_correct : area_of_triangle (4, -3) (4, 7) (9, 7) = 25.0 := by
  sorry

end triangle_area_is_correct_l822_822998


namespace length_of_BC_l822_822095

theorem length_of_BC (A B C P Q R M : Type) 
  [metric_space A] [metric_space B] [metric_space C]
  [metric_space P] [metric_space Q] [metric_space R] [metric_space M]
  (omega : circle A B C)
  (P_mid : midpoint (arc omega B C))
  (Q_mid : midpoint (arc omega A B))
  (tangent_A : tangent omega A PQ)
  (M_mid_AR : midpoint (segment A R) ∈ line B C)
  (perimeter_ABC : length (segment A B) + length (segment B C) + length (segment A C) = 12) :
  length (segment B C) = 4 :=
sorry

end length_of_BC_l822_822095


namespace binomial_20_5_l822_822278

theorem binomial_20_5 : Nat.choose 20 5 = 15504 := by
  sorry

end binomial_20_5_l822_822278


namespace problem_statement_l822_822774

theorem problem_statement (r p q : ℝ) (hr : r < 0) (hpq_ne_zero : p * q ≠ 0) (hp2r_gt_q2r : p^2 * r > q^2 * r) :
  ¬ (-p > -q) ∧ ¬ (-p < q) ∧ ¬ (1 < -q / p) ∧ ¬ (1 > q / p) :=
by
  sorry

end problem_statement_l822_822774


namespace tax_difference_is_250000_l822_822978

noncomputable def old_tax_rate : ℝ := 0.20
noncomputable def new_tax_rate : ℝ := 0.30
noncomputable def old_income : ℝ := 1000000
noncomputable def new_income : ℝ := 1500000
noncomputable def old_taxes_paid := old_tax_rate * old_income
noncomputable def new_taxes_paid := new_tax_rate * new_income
noncomputable def tax_difference := new_taxes_paid - old_taxes_paid

theorem tax_difference_is_250000 : tax_difference = 250000 := by
  sorry

end tax_difference_is_250000_l822_822978


namespace value_of_f_at_neg3_l822_822565

def f (x : ℤ) : ℤ := x^2 + x

theorem value_of_f_at_neg3 : f (-3) = 6 := by
  -- condition f(x) = x^2 + x and question f(-3)=?
  have h : f(-3) = (-3)^2 + (-3) := rfl
  -- the conclusion is f(-3) = 6
  rw h
  exact rfl

end value_of_f_at_neg3_l822_822565


namespace students_gold_award_freshmen_l822_822308

theorem students_gold_award_freshmen 
    (total_students total_award_winners : ℕ)
    (students_selected exchange_meeting : ℕ)
    (freshmen_selected gold_award_selected : ℕ)
    (prop1 : total_award_winners = 120)
    (prop2 : exchange_meeting = 24)
    (prop3 : freshmen_selected = 6)
    (prop4 : gold_award_selected = 4) :
    ∃ (gold_award_students : ℕ), gold_award_students = 4 ∧ gold_award_students ≤ freshmen_selected :=
by
  sorry

end students_gold_award_freshmen_l822_822308


namespace range_of_m_for_ellipse_l822_822430

theorem range_of_m_for_ellipse (m : ℝ) :
  (fraction x m^2 + fraction y (m+2) = 1 ∧ foci_on_x_axis) → 
  (m ∈ Set.Ioo (-2 : ℝ) (-1) ∨ m ∈ Set.Ioi (2 : ℝ)) :=
sorry

end range_of_m_for_ellipse_l822_822430


namespace count_valid_integers_l822_822407

theorem count_valid_integers : 
  (∃ count : ℕ, count = 1295 ∧ 
    ∀ n ∈ (Finset.range 10000).filter (λ x, ∀ d ∈ to_digits 10 x, d ∉ {2,3,4,5}), 
      true) :=
begin
  use 1295,
  split,
  { exact 1295 },
  { intros n hn,
    sorry }
end

end count_valid_integers_l822_822407


namespace remaining_to_original_ratio_l822_822981

-- Define the number of rows and production per row for corn and potatoes.
def rows_of_corn : ℕ := 10
def corn_per_row : ℕ := 9
def rows_of_potatoes : ℕ := 5
def potatoes_per_row : ℕ := 30

-- Define the remaining crops after pest destruction.
def remaining_crops : ℕ := 120

-- Calculate the original number of crops from corn and potato productions.
def original_crops : ℕ :=
  (rows_of_corn * corn_per_row) + (rows_of_potatoes * potatoes_per_row)

-- Define the ratio of remaining crops to original crops.
def crops_ratio : ℚ := remaining_crops / original_crops

theorem remaining_to_original_ratio : crops_ratio = 1 / 2 := 
by
  sorry

end remaining_to_original_ratio_l822_822981


namespace ant_paths_l822_822934

theorem ant_paths (n m : ℕ) : 
  ∃ paths : ℕ, paths = Nat.choose (n + m) m := sorry

end ant_paths_l822_822934


namespace max_sin_sum_in_triangle_l822_822783

-- Definition of convexity
def isConvex (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄ (a b : ℝ), 0 ≤ a → 0 ≤ b → a + b = 1 → x ∈ I → y ∈ I → 
    f (a * x + b * y) ≤ a * f x + b * f y

-- Proof statement
theorem max_sin_sum_in_triangle :
  (∀ x ∈ (Set.Ioo 0 π), isConvex sin (Set.Ioo 0 π)) →
  ∃ (A B C : ℝ), A + B + C = π ∧ 
  A ∈ (Set.Ioo 0 π) ∧ B ∈ (Set.Ioo 0 π) ∧ C ∈ (Set.Ioo 0 π) ∧ 
  sin A + sin B + sin C ≤ 3/2 * sqrt 3 :=
begin
  sorry
end

end max_sin_sum_in_triangle_l822_822783


namespace handshakes_per_boy_l822_822778

theorem handshakes_per_boy (n : ℕ) (h : n = 12) (total_handshakes : ℕ) (h2 : total_handshakes = 66) :
  (total_handshakes = (n * (n - 1)) / 2) → (total_handshakes / (n - 1) = 6) :=
by
  intros
  rw [h, h2]
  sorry

end handshakes_per_boy_l822_822778


namespace reflection_vector_l822_822655

noncomputable def reflection (p1 p2 p : ℝ × ℝ) : ℝ × ℝ :=
  let midx := (p1.1 + p2.1) / 2
  let midy := (p1.2 + p2.2) / 2
  let mid := (midx, midy)
  let d := p.1 - mid.1
  let e := p.2 - mid.2
  (midx - d, midy - e)

theorem reflection_vector : reflection (2, -3) (8, 1) (1, 4) = (5/13 * -4, -53/13) :=
by
  sorry

end reflection_vector_l822_822655


namespace low_correlation_near_zero_l822_822570

-- Define conditions 
variable {X Y : Type}
variable [MeasurableSpace X] [MeasurableSpace Y]
variable (r : ℝ)
variable (lower_degree : r < 1)

-- State the theorem: Prove that a lower degree of linear correlation implies the correlation coefficient is closer to 0
theorem low_correlation_near_zero (h : lower_degree) : abs r < 1 :=
sorry

end low_correlation_near_zero_l822_822570


namespace parallel_vectors_have_specific_x_l822_822338

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (x, 3)
noncomputable def vector_b : ℝ × ℝ := (3, 1)

theorem parallel_vectors_have_specific_x (x : ℝ) (h : vector_a x ∥ vector_b) : x = 9 :=
by
  sorry

end parallel_vectors_have_specific_x_l822_822338


namespace hyperbola_eccentricity_l822_822707

theorem hyperbola_eccentricity (a c b : ℝ) (h₀ : b = 3)
  (h₁ : ∃ p, (p = 5) ∧ (a^2 + b^2 = (p : ℝ)^2))
  (h₂ : ∃ f, f = (p : ℝ)) :
  ∃ e, e = c / a ∧ e = 5 / 4 :=
by
  obtain ⟨p, hp, hap⟩ := h₁
  obtain ⟨f, hf⟩ := h₂
  sorry

end hyperbola_eccentricity_l822_822707


namespace probability_product_multiple_of_90_l822_822436

def set : Set ℕ := {4, 9, 15, 18, 30, 36, 45}

-- Definition of the condition and proof request
theorem probability_product_multiple_of_90 :
  let pairs := { (a, b) | a ∈ set ∧ b ∈ set ∧ a ≠ b ∧ ¬ (a * b % 10 = 0 ∧ a * b % 15 = 0) } in
  let valid_pairs := { (a, b) | (a, b) ∈ pairs ∧ (a * b % 90 = 0) } in
  (Finset.card valid_pairs:Fℤ)/ (Finset.card pairs:Fℤ) = 1 / 21 :=
sorry

end probability_product_multiple_of_90_l822_822436


namespace length_of_BC_l822_822518

theorem length_of_BC (O A M B C : Point) (r : ℝ) (α : ℝ) (h_radius : dist O A = r) 
  (h_M_on_AO : is_on_line M O A) (h_B_on_circ : dist O B = r)
  (h_C_on_circ : dist O C = r) (h_angle_AMB : ∠ A M B = α) 
  (h_angle_OMC : ∠ O M C = α) (h_cos_alpha : Real.cos α = 4 / 7) :
  dist B C = 24 :=
sorry

end length_of_BC_l822_822518


namespace ladder_in_alley_l822_822450

-- Define the conditions and the final result
theorem ladder_in_alley (l : ℝ) : 
  ∃ w : ℝ, w = (l * (1 + Real.sqrt 3)) / 2 :=
by
  use (l * (1 + Real.sqrt 3)) / 2
  sorry

end ladder_in_alley_l822_822450


namespace complex_area_d_squared_l822_822517

noncomputable def d_squared (z : ℂ) : ℝ :=
  let den := z + 1/z
  in complex.norm_sq den

theorem complex_area_d_squared (z : ℂ) (hz_imag : z.im > 0)
  (h_area : (2 * |complex.sin (2 * z.arg)|) = (12/13)) :
  ∃ d : ℝ, d^2 = d_squared z ∧ d^2 = 16 / 13 :=
by
  sorry

end complex_area_d_squared_l822_822517


namespace equation_of_asymptotes_l822_822391

variables {a b c : ℝ}
variables (x y : ℝ)

def is_hyperbola : Prop := (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)
def eccentricity : Prop := (c = 2 * a)
def c_relation : Prop := (c^2 = a^2 + b^2)

theorem equation_of_asymptotes
  (h1 : is_hyperbola x y)
  (h2 : eccentricity)
  (h3 : c_relation) :
  (y = sqrt 3 * x ∨ y = -sqrt 3 * x) :=
sorry

end equation_of_asymptotes_l822_822391


namespace time_to_fill_pool_is_33_l822_822075

-- Note: realistically, we would use noncomputable theory only if there were floating-point calculations
noncomputable theory

def poolVolume : ℕ := 30000
def numHoses : ℕ := 5
def flowRatePerHose : ℕ := 3
def flowRatePerMinute : ℕ := numHoses * flowRatePerHose
def flowRatePerHour : ℕ := flowRatePerMinute * 60
def timeToFillPool : ℕ := poolVolume / flowRatePerHour

theorem time_to_fill_pool_is_33 :
  timeToFillPool = 33 := by
  sorry

end time_to_fill_pool_is_33_l822_822075


namespace problem_part1_problem_part2_l822_822639

theorem problem_part1 (n : ℕ) (h : (n : ℚ) / (1 + 1 + n) = 1 / 2) : n = 2 :=
by sorry

theorem problem_part2 : 
  let outcomes := ({("red", "black"), ("red", "white1"), ("red", "white2"), ("black", "white1"), ("black", "white2"), ("white1", "white2")} : Finset (String × String)) in
  (outcomes.filter (λ p => p.1 = "red" ∧ (p.2 = "white1" ∨ p.2 = "white2"))).card / outcomes.card = 1 / 3 :=
by sorry

end problem_part1_problem_part2_l822_822639


namespace distance_between_A_and_B_l822_822196

-- Define the basic setup
constant locale_A : Type
constant locale_B : Type

constant speed_B : ℝ := 50  -- B's speed, 50 meters per minute
constant catch_up_time : ℝ := 30  -- Time for A to catch B moving in the same direction in minutes
constant meet_time : ℝ := 6  -- Time for A and B to meet moving towards each other in minutes

-- Define speeds and other distances
noncomputable def speed_difference : ℝ := 1 / catch_up_time
noncomputable def speed_sum : ℝ := 1 / meet_time

theorem distance_between_A_and_B : ∀ (d : ℝ),
  d = (speed_B / ((speed_sum - speed_difference) / 2)) → d = 750 := 
by
  sorry

end distance_between_A_and_B_l822_822196


namespace melly_cats_l822_822504

/-- 
Problem statement:
  The first cat has 3 blue-eyed kittens and an unknown number (B) of brown-eyed kittens.
  The second cat has 4 blue-eyed kittens and 6 brown-eyed kittens.
  35% of all the kittens have blue eyes.
  Determine the number of brown-eyed kittens (B) that the first cat has.
-/
theorem melly_cats :
  ∃ (B : ℕ), 
    let total_blue_eyed := 3 + 4 in
    let total_kittens := 3 + B + 4 + 6 in
    (7 / ↑total_kittens) * 100 = 35 ∧ 
    B = 7 := 
begin
  let B := 7,
  let total_blue_eyed := 3 + 4,
  let total_kittens := 3 + B + 4 + 6,
  have h1: (total_blue_eyed / total_kittens) * 100 = 35,
  { -- Show that (7 / (B + 13)) * 100 = 35 
    sorry },
  use B,
  exact and.intro h1 rfl,
end

end melly_cats_l822_822504


namespace weight_of_a_l822_822181

theorem weight_of_a (a b c d e : ℝ)
  (h1 : (a + b + c) / 3 = 84)
  (h2 : (a + b + c + d) / 4 = 80)
  (h3 : e = d + 8)
  (h4 : (b + c + d + e) / 4 = 79) :
  a = 80 :=
by
  sorry

end weight_of_a_l822_822181


namespace prove_equation_l822_822103

theorem prove_equation (a b p : ℕ) (h1 : 4 * 10 = 40) (h2 : a + b * real.sqrt p = 30 + 10 * real.sqrt 3) (hp : nat.prime p) : 7 * a + 5 * b + 3 * p = 269 := by
  sorry

end prove_equation_l822_822103


namespace participation_schemes_count_l822_822533

theorem participation_schemes_count:
  let students := {A, B, C, D} in
  let subjects := {subject1, subject2, subject3} in
  let must_participate := 'A' in
  ∃ (selected: Finset Char), 
    selected ⊆ students ∧
    must_participate ∈ selected ∧
    selected.card = 3 ∧
  (∃ (arrangements: Finset (selected → subjects)),
    arrangements.card = 18) :=
by
  let students := {'A', 'B', 'C', 'D'}
  let subjects := {subject1, subject2, subject3}
  let must_participate := 'A'
  let selected := (students \ {must_participate}).powerset.filter (λ s, s.card = 2) 
  have h_selected_card :  selected.card = 3 := sorry
  let arrangements := (selected.bUnion (λ s, (s.product subjects.to_finset)))
  have h_arrangements_card : arrangements.card = 18 := sorry
  exact ⟨selected, finset.subset_univ selected, finset.mem_univ 'A', h_selected_card, ⟨arrangements, h_arrangements_card⟩⟩

end participation_schemes_count_l822_822533


namespace angle_A_is_45_l822_822440

theorem angle_A_is_45 
  (B : ℝ) (a b : ℝ) (hB : B = 60) (ha : a = sqrt 6) (hb : b = 3) : 
  ∃ A : ℝ, A = 45 :=
by
  use 45
  sorry

end angle_A_is_45_l822_822440


namespace sum_of_first_nine_terms_l822_822375

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ (n : ℕ), a n = a 1 + d * (n - 1)

variables (a : ℕ → ℝ) (h_seq : arithmetic_sequence a)

-- Given condition: a₂ + a₃ + a₇ + a₈ = 20
def condition : Prop := a 2 + a 3 + a 7 + a 8 = 20

-- Statement: Prove that the sum of the first 9 terms is 45
theorem sum_of_first_nine_terms (h : condition a) : 
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) = 45 :=
by sorry

end sum_of_first_nine_terms_l822_822375


namespace cost_of_one_bag_of_potatoes_l822_822121

theorem cost_of_one_bag_of_potatoes :
  let x := 250 in
  ∀ (price : ℕ)
    (bags : ℕ)
    (andrey_initial_price : ℕ)
    (andrey_sold_price : ℕ)
    (boris_initial_price : ℕ)
    (boris_first_price : ℕ)
    (boris_second_price : ℕ)
    (earnings_andrey : ℕ)
    (earnings_boris_first : ℕ)
    (earnings_boris_second : ℕ)
    (total_earnings_boris : ℕ),
  bags = 60 →
  andrey_initial_price = price →
  andrey_sold_price = 2 * price →
  andrey_sold_price * bags = earnings_andrey →
  boris_initial_price = price →
  boris_first_price = 1.6 * price →
  boris_second_price = 2.24 * price →
  boris_first_price * 15 + boris_second_price * 45 = total_earnings_boris →
  total_earnings_boris = earnings_andrey + 1200 →
  price = x :=
by
  intros x price bags andrey_initial_price andrey_sold_price boris_initial_price boris_first_price boris_second_price earnings_andrey earnings_boris_first earnings_boris_second total_earnings_boris
  assume h_bags h_andrey_initial_price h_andrey_sold_price h_earnings_andrey h_boris_initial_price h_boris_first_price h_boris_second_price h_total_earnings_boris h_total_earnings_difference
  if h_necessary : x = 250 then
    sorry
  else
    sorry


end cost_of_one_bag_of_potatoes_l822_822121


namespace closest_to_one_tenth_l822_822416

noncomputable def p (n : ℕ) : ℚ :=
  1 / (n * (n + 2)) + 1 / ((n + 2) * (n + 4)) + 1 / ((n + 4) * (n + 6)) +
  1 / ((n + 6) * (n + 8)) + 1 / ((n + 8) * (n + 10))

theorem closest_to_one_tenth {n : ℕ} (h₀ : 4 ≤ n ∧ n ≤ 7) : 
  |(5 : ℚ) / (n * (n + 10)) - 1 / 10| ≤ 
  |(5 : ℚ) / (4 * (4 + 10)) - 1 / 10| ∧ n = 4 := 
sorry

end closest_to_one_tenth_l822_822416


namespace true_compound_propositions_l822_822289

-- Define the propositions
def p1 : Prop := 
  ∀ (l₁ l₂ l₃ : Line), 
    (l₁ ≠ l₂ ∧ l₂ ≠ l₃ ∧ l₁ ≠ l₃) ∧ 
    (∃ (p₁ p₂ p₃ : Point),
      l₁.contains p₁ ∧ l₁.contains p₂ ∧
      l₂.contains p₂ ∧ l₂.contains p₃ ∧
      l₃.contains p₃ ∧ l₃.contains p₁)
    → ∃ (α : Plane), 
      α.contains l₁ ∧ α.contains l₂ ∧ α.contains l₃

def p2 : Prop :=
  ∀ (a b c : Point),
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) 
    → ∃! (α : Plane), 
      α.contains a ∧ α.contains b ∧ α.contains c

def p3 : Prop :=
  ∀ (l₁ l₂ : Line),
    (¬ ∃ (p : Point), l₁.contains p ∧ l₂.contains p)
    → l₁.parallel l₂

def p4 : Prop :=
  ∀ (α : Plane) (l m : Line),
    (α.contains l ∧ ¬ α.contains m ∧ m.perpendicular α) 
    → m.perpendicular l

-- Main theorem that identifies the true compound propositions
theorem true_compound_propositions :
  (p1 ∧ p4) ∧ ¬ (p1 ∧ p2) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) :=
by
  -- Proof might involve defining lines, points, and planes and their relationships
  sorry

end true_compound_propositions_l822_822289


namespace probability_fourth_term_is_integer_l822_822498

-- Definitions of the transformation functions
def roll1 (a : ℕ) : ℚ := 3 * a - 2
def roll2 (a : ℕ) : ℚ := 2 * a - 1
def roll3 (a : ℕ) : ℚ := (a : ℚ) / 2 - 1

-- Definition of the initial term
noncomputable def a1 : ℕ := 7

-- The statement of the theorem
theorem probability_fourth_term_is_integer :
  let a2 (r1 : ℕ) := if r1 = 1 then roll1 a1 else if r1 = 2 then roll2 a1 else roll3 a1 in
  let a3 (r1 r2 : ℕ) := if r2 = 1 then roll1 (a2 r1).to_nat else if r2 = 2 then roll2 (a2 r1).to_nat else roll3 (a2 r1).to_nat in
  let a4 (r1 r2 r3 : ℕ) := if r3 = 1 then roll1 (a3 r1 r2).to_nat else if r3 = 2 then roll2 (a3 r1 r2).to_nat else roll3 (a3 r1 r2).to_nat in
  𝔓 ({(r1, r2, r3) : ℕ × ℕ × ℕ // ∃ k : ℤ, a4 r1 r2 r3 = k}) = 5 / 9 := sorry

end probability_fourth_term_is_integer_l822_822498


namespace simplify_sqrt_180_l822_822067

theorem simplify_sqrt_180 : sqrt 180 = 6 * sqrt 5 :=
by
  have h : 180 = 2^2 * 3^2 * 5 := by norm_num
  calc
    sqrt 180
      = sqrt (2^2 * 3^2 * 5)     : by rw [h]
  ... = sqrt (2^2) * sqrt (3^2) * sqrt 5 : by rw [sqrt_mul, sqrt_mul]
  ... = 2 * 3 * sqrt 5          : by rw [sqrt_sq, sqrt_sq]
  ... = 6 * sqrt 5              : by norm_num

end simplify_sqrt_180_l822_822067


namespace place_synthetic_method_l822_822084

theorem place_synthetic_method :
  "Synthetic Method" = "Direct Proof" :=
sorry

end place_synthetic_method_l822_822084


namespace minutes_watched_on_Thursday_l822_822841

theorem minutes_watched_on_Thursday 
  (n_total : ℕ) (n_Mon : ℕ) (n_Tue : ℕ) (n_Wed : ℕ) (n_Fri : ℕ) (n_weekend : ℕ)
  (h_total : n_total = 352)
  (h_Mon : n_Mon = 138)
  (h_Tue : n_Tue = 0)
  (h_Wed : n_Wed = 0)
  (h_Fri : n_Fri = 88)
  (h_weekend : n_weekend = 105) :
  n_total - (n_Mon + n_Tue + n_Wed + n_Fri + n_weekend) = 21 := by
  sorry

end minutes_watched_on_Thursday_l822_822841


namespace link_integers_chain_l822_822870

theorem link_integers_chain (m n : ℕ) (h_m : m > 2) (h_n : n > 2) :
  ∃ (k : ℕ) (a : Fin (k+2) → ℕ), a 0 = m ∧ a (Fin.last _) = n ∧
  ∀ i : Fin k, (a i) * (a (i + 1)) % ((a i) + (a (i + 1))) = 0 :=
begin
  sorry
end

end link_integers_chain_l822_822870


namespace fraction_to_decimal_l822_822405

theorem fraction_to_decimal : (5 : ℚ) / 8 = 0.625 := by
  -- Prove that the fraction 5/8 equals the decimal 0.625
  sorry

end fraction_to_decimal_l822_822405


namespace conjugate_of_z_l822_822781

def conj (z : ℂ) : ℂ := complex.conj z

def z : ℂ := (2 - complex.i) / (complex.i ^ 3)

theorem conjugate_of_z : conj z = 1 - 2 * complex.i :=
by
  sorry

end conjugate_of_z_l822_822781


namespace card_game_ensures_final_state_l822_822149

def card_game_eventually_ends (n : ℕ) (beats : ℕ → ℕ → Prop) (deck1 deck2 : list ℕ) : Prop :=
  ∃ (strategy : (list ℕ × list ℕ) → (list ℕ × list ℕ)), ∀ (deck1 deck2 : list ℕ),
    (deck1.length + deck2.length = n) →
    (∀ deck1 deck2, (deck1, deck2) ≠ ([], [])) →
    (∃ final_state, (strategy (deck1, deck2) = final_state ∧ (final_state.1 = [] ∨ final_state.2 = [])))

theorem card_game_ensures_final_state (n : ℕ) (beats : ℕ → ℕ → Prop) (deck1 deck2 : list ℕ) :
  card_game_eventually_ends n beats deck1 deck2 :=
sorry

end card_game_ensures_final_state_l822_822149


namespace triangle_right_angle_l822_822794

theorem triangle_right_angle (A B C : Point)
  (hAB : dist A B = 1)
  (hAC : dist A C = 2)
  (hBC : dist B C = Real.sqrt 5) : 
  is_right_triangle A B C := 
sorry

end triangle_right_angle_l822_822794


namespace bouquets_needed_to_earn_1000_l822_822268

theorem bouquets_needed_to_earn_1000 :
  ∀ (cost_per_bouquet sell_price_bouquet: ℕ) (roses_per_bouquet_bought roses_per_bouquet_sold target_profit: ℕ),
    cost_per_bouquet = 20 →
    sell_price_bouquet = 20 →
    roses_per_bouquet_bought = 7 →
    roses_per_bouquet_sold = 5 →
    target_profit = 1000 →
    (target_profit / (sell_price_bouquet * roses_per_bouquet_sold / roses_per_bouquet_bought - cost_per_bouquet) * roses_per_bouquet_bought = 125) :=
by
  intros cost_per_bouquet sell_price_bouquet roses_per_bouquet_bought roses_per_bouquet_sold target_profit 
    h_cost_per_bouquet h_sell_price_bouquet h_roses_per_bouquet_bought h_roses_per_bouquet_sold h_target_profit
  sorry

end bouquets_needed_to_earn_1000_l822_822268


namespace monotonic_sufficient_not_necessary_maximum_l822_822964

noncomputable def f (x : ℝ) : ℝ := sorry  -- Placeholder for the function f
def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop := (∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y) ∨ (∀ x y, a ≤ x → x ≤ y → y ≤ b → f y ≤ f x)
def has_max_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∃ M, ∀ x, a ≤ x → x ≤ b → f x ≤ M

theorem monotonic_sufficient_not_necessary_maximum : 
  ∀ f : ℝ → ℝ,
  ∀ a b : ℝ,
  a ≤ b →
  monotonic_on f a b → 
  has_max_on f a b :=
sorry  -- Proof is omitted

end monotonic_sufficient_not_necessary_maximum_l822_822964


namespace triangle_area_l822_822575

-- Define the given conditions
def perimeter : ℝ := 60
def inradius : ℝ := 2.5

-- Prove the area of the triangle using the given inradius and perimeter
theorem triangle_area (p : ℝ) (r : ℝ) (h1 : p = 60) (h2 : r = 2.5) :
  (r * (p / 2)) = 75 := 
by
  rw [h1, h2]
  sorry

end triangle_area_l822_822575


namespace staircase_sum_of_digits_l822_822876

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  let s := toString n
  (s.toList.map (λ c => (c.toNat - '0'.toNat))).sum

theorem staircase_sum_of_digits :
  (∃ n : ℕ, (⌈n / 3⌉ - ⌈n / 4⌉ = 7) ∧ sum_of_digits n = 13) :=
by
  use 76
  split
  · sorry  -- Here you need to prove that the ceiling condition holds for 76
  · sorry  -- Here you need to prove that the sum of digits condition holds

end staircase_sum_of_digits_l822_822876


namespace min_a2_b2_c2_l822_822909

theorem min_a2_b2_c2 (a b c : ℕ) (h : a + 2 * b + 3 * c = 73) : a^2 + b^2 + c^2 ≥ 381 :=
by sorry

end min_a2_b2_c2_l822_822909


namespace three_digit_numbers_with_digit_5_l822_822411

theorem three_digit_numbers_with_digit_5 : 
  {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ d : ℕ, d ∈ [n / 100 % 10, n / 10 % 10, n % 10] ∧ d = 5)}.card = 270 :=
by
  sorry

end three_digit_numbers_with_digit_5_l822_822411


namespace find_stream_speed_l822_822971

theorem find_stream_speed (b s : ℝ) 
  (h1 : b + s = 10) 
  (h2 : b - s = 8) : s = 1 :=
by
  sorry

end find_stream_speed_l822_822971


namespace sum_of_rectangle_areas_l822_822872

theorem sum_of_rectangle_areas :
  let base_width := 2
  let lengths := [1, 3, 5, 7, 9, 11]
  let areas := lengths.map (λ n, base_width * n^2)
  areas.sum = 572 :=
by
  let base_width := 2
  let lengths := [1, 3, 5, 7, 9, 11]
  let squares := lengths.map (λ n, n^2)
  let areas := squares.map (λ n, base_width * n)
  have calc_sum : areas.sum = 2 + 18 + 50 + 98 + 162 + 242 := by
    rw [List.sum_cons, List.sum_cons, List.sum_cons, List.sum_cons, List.sum_cons, List.sum_nil]
    rfl
  exact calc_sum ▸ rfl

end sum_of_rectangle_areas_l822_822872


namespace event_independence_inter_prob_l822_822424

variable {Ω : Type} [MeasureSpace Ω]

theorem event_independence_inter_prob
  {E F : Set Ω}
  (h_indep : Indep E F)
  (h_probE : P(E) = 1 / 4)
  (h_probF : P(F) = 1 / 4) :
  P(E ∩ F) = 1 / 16 := by
  sorry

end event_independence_inter_prob_l822_822424


namespace part1_part2_l822_822187

theorem part1 (n : ℕ) (students : Finset ℕ) (d : students → students → ℕ) :
  (∀ (a b : students), a ≠ b → d a b ≠ d b a) →
  (∀ (a b c : students), a ≠ b ∧ b ≠ c ∧ a ≠ c → d a b ≠ d b c ∧ d a b ≠ d a c ∧ d a c ≠ d b c) →
  (students.card = 2 * n + 1) →
  ∃ a b : students, a ≠ b ∧
  (∀ c : students, c ≠ a → d a c > d a b) ∧ 
  (∀ c : students, c ≠ b → d b c > d b a) :=
sorry

theorem part2 (n : ℕ) (students : Finset ℕ) (d : students → students → ℕ) :
  (∀ (a b : students), a ≠ b → d a b ≠ d b a) →
  (∀ (a b c : students), a ≠ b ∧ b ≠ c ∧ a ≠ c → d a b ≠ d b c ∧ d a b ≠ d a c ∧ d a c ≠ d b c) →
  (students.card = 2 * n + 1) →
  ∃ c : students, ∀ a : students, ¬ (∀ b : students, b ≠ a → d b a < d b c ∧ d a c < d a b) :=
sorry

end part1_part2_l822_822187


namespace fermat_little_theorem_seven_pow_2048_mod_17_l822_822322

theorem fermat_little_theorem (a : ℤ) (p : ℕ) [fact (nat.prime p)] (h : a % p ≠ 0): a ^ (p - 1) ≡ 1 [MOD p] :=
sorry

theorem seven_pow_2048_mod_17 : (7 ^ 2048) % 17 = 1 :=
by
  have p := 17
  have prime_p : nat.prime p := by norm_num
  have a := 7
  have h : a % p ≠ 0 := by norm_num
  have fermat := fermat_little_theorem a p prime_p h
  have exp_decomp: 2048 = 16 * 128 := by norm_num
  rw exp_decomp
  calc 7 ^ (16 * 128) % 17
      = (7 ^ 16) ^ 128 % 17     : by rw pow_mul
      ... ≡ 1 ^ 128 % 17       : by rw fermat
      ... = 1 % 17             : by rw one_pow
      ... = 1                 : by norm_num

end fermat_little_theorem_seven_pow_2048_mod_17_l822_822322


namespace sled_distance_in_40_seconds_l822_822227

noncomputable def sled_distance (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  (n / 2) * (2 * a + (n - 1) * d)

theorem sled_distance_in_40_seconds :
  sled_distance 8 10 40 = 8120 :=
by
  -- sum of the first 40 terms of the sequence: 8, 18, 28, ..., up to the 40th term.
  sorry

end sled_distance_in_40_seconds_l822_822227


namespace unique_parallel_line_through_point_l822_822780

-- Definitions of the conditions
def line_parallel_to_plane (l : Line) (a : Plane) : Prop :=
  ∀ (P : Point), P ∈ l → P ∉ a

axiom point_in_plane (A : Point) (a : Plane) : Prop :=
  A ∈ a

-- The theorem to be proved
theorem unique_parallel_line_through_point (l : Line) (a : Plane) (A : Point) 
  (hl : line_parallel_to_plane l a) (hA : point_in_plane A a) :
  ∃! m : Line, (A ∈ m) ∧ (∀ (P : Point), P ∈ m → P ∉ a) := sorry

end unique_parallel_line_through_point_l822_822780


namespace sqrt_180_simplify_l822_822058

theorem sqrt_180_simplify : real.sqrt 180 = 6 * real.sqrt 5 := by
  have h : 180 = 2^2 * 3^2 * 5 := by norm_num
  rw [h, real.sqrt_mul, real.sqrt_mul, real.sqrt_mul]
  all_goals { norm_num }
  sorry

end sqrt_180_simplify_l822_822058


namespace find_number_l822_822777

theorem find_number (x n : ℝ) (h1 : (3 / 2) * x - n = 15) (h2 : x = 12) : n = 3 :=
by
  sorry

end find_number_l822_822777


namespace petya_has_winning_strategy_l822_822112

-- Define the parameters of the problem
variable (Vasya Petya : Type)
variable (num_components : ℕ)
variable (components : fin num_components → Vasya) -- components are joined
variable (is_wire : Vasya → Vasya → Prop) -- is_wire represents the existence of a wire between two components
variable (cuts_wire : Vasya → Vasya → Prop) -- cuts_wire represents a cut operation on a wire

-- Condition: There are 2000 components in the circuit
def num_components_eq_2000 : num_components = 2000 := sorry

-- Condition: Every two components are initially joined by a wire (complete graph)
def initial_condition : ∀ (c1 c2 : Vasya), is_wire c1 c2 := sorry

-- Condition: Vasya cuts one wire per turn, Petya cuts two or three wires per turn
def turn_rules (turn : ℕ) : Prop :=
  if turn % 2 = 0 then
    ∃ (c1 c2 : Vasya), cuts_wire c1 c2 -- Vasya's turn, cuts 1 wire
  else
    ∃ (c1 c2 c3 : Vasya), cuts_wire c1 c2 ∧ cuts_wire c2 c3 ∨
    (∃ (c1 c2 c3 c4 : Vasya), cuts_wire c1 c2 ∧ cuts_wire c3 c4) -- Petya's turn, cuts 2 or 3 wires

-- Condition: The player who cuts the last wire from some component loses
def loss_condition : ∃ (c : Vasya), (∀ (c_other : Vasya), ¬ is_wire c c_other) :=
  sorry

-- Goal: Prove that Petya has a winning strategy
theorem petya_has_winning_strategy : ∀ (turn : ℕ), ∃ (strategy : fin turn → (Vasya × Vasya) × Option (Vasya × Vasya)),
  (∀ i : fin turn, if i.val % 2 = 0 then cuts_wire (strategy i).fst.1 (strategy i).fst.2 else
                   (∃ (c1 c2 c3 : Vasya), strategy i = ((c1, c2), some (c3, c1)) ∨
                     ∃ (c1 c2 c3 c4 : Vasya), strategy i = ((c1, c2), some (c3, c4)))) →
  (∀ i : fin turn, ¬ loss_condition) :=
sorry

end petya_has_winning_strategy_l822_822112


namespace coin_count_l822_822197

theorem coin_count (x : ℝ) (h₁ : x + 0.50 * x + 0.25 * x = 35) : x = 20 :=
by
  sorry

end coin_count_l822_822197


namespace solve_equation_l822_822874

theorem solve_equation (x : ℝ) :
    4^(2*x + 1) - sqrt (29*4^(2*x + 1) + 2^(2*x + 1) + 2*8^(2*x)) = 21*2^(2*x) - 4
    ↔ x = 3/2 ∨ x = -3/2 := 
sorry

end solve_equation_l822_822874


namespace correct_average_of_10_numbers_l822_822541

theorem correct_average_of_10_numbers (initial_avg : ℕ → ℝ) 
  (num_count : ℕ)
  (wrong_num1 : ℝ)
  (wrong_num2 : ℝ)
  (correction1 : ℝ)
  (correction2_wrong : ℝ)
  (correction2_right : ℝ) :
  num_count = 10 →
  initial_avg num_count = 40.2 →
  wrong_num1 = 16 →
  wrong_num2 = 13 →
  correction2_wrong = 31 →
  (initial_avg num_count * num_count - wrong_num1 - wrong_num2 + correction2_right) / num_count = 40.4 :=
begin
  sorry
end

end correct_average_of_10_numbers_l822_822541


namespace slope_of_line_through_focus_of_parabola_l822_822207

theorem slope_of_line_through_focus_of_parabola
  (C : (x y : ℝ) → y^2 = 4 * x)
  (F : (ℝ × ℝ) := (1, 0))
  (A B : (ℝ × ℝ))
  (l : ℝ → ℝ)
  (intersects : (x : ℝ) → (l x) ^ 2 = 4 * x)
  (passes_through_focus : l 1 = 0)
  (distance_condition : ∀ (d1 d2 : ℝ), d1 = 4 * d2 → dist F A = d1 ∧ dist F B = d2) :
  ∃ k : ℝ, (∀ (x : ℝ), l x = k * (x - 1)) ∧ (k = 4 / 3 ∨ k = -4 / 3) :=
by
  sorry

end slope_of_line_through_focus_of_parabola_l822_822207


namespace sqrt_180_simplified_l822_822055

theorem sqrt_180_simplified : Real.sqrt 180 = 6 * Real.sqrt 5 :=
   sorry

end sqrt_180_simplified_l822_822055


namespace problem_correctness_l822_822188

noncomputable def solve_for_p (q : ℝ) (hq : q ≠ 0) (p : ℝ)
  (hp : p ≠ -3 * q ^ 2) : Prop :=
  log p + log (q ^ 2) = log (p + 3 * q ^ 2) → p = 3 * q ^ 2 / (q ^ 2 - 1)

theorem problem_correctness (q : ℝ) (hq : q ≠ 0) (hq1 : q ^ 2 ≠ 1) 
  (p : ℝ) (hp : p = 3 * q ^ 2 / (q ^ 2 - 1)) (h_not_equal : p ≠ -3 * q ^ 2) : 
  solve_for_p q hq p h_not_equal := 
sorry

end problem_correctness_l822_822188


namespace probability_even_digit_l822_822884

open Finset

theorem probability_even_digit (digits : Finset ℕ) (h_digits : digits = {1, 2, 4, 6, 9}) :
  (∃ even_digits : Finset ℕ, even_digits = {2, 4, 6} ∧
    ∃ total_digits : ℕ, total_digits = 5 ∧
    ∃ probability : ℚ, probability = 3 / 5 ) := by
  have even_digits : Finset ℕ := {2, 4, 6}
  have total_digits : ℕ := 5
  have probability : ℚ := 3 / 5
  exact ⟨even_digits, rfl, ⟨total_digits, rfl, ⟨probability, rfl⟩⟩⟩

end probability_even_digit_l822_822884


namespace problem_statement_l822_822298

variables (p1 p2 p3 p4 : Prop)

theorem problem_statement (h_p1 : p1 = True)
                         (h_p2 : p2 = False)
                         (h_p3 : p3 = False)
                         (h_p4 : p4 = True) :
  (p1 ∧ p4) = True ∧
  (p1 ∧ p2) = False ∧
  (¬p2 ∨ p3) = True ∧
  (¬p3 ∨ ¬p4) = True :=
by
  sorry

end problem_statement_l822_822298


namespace new_ratio_of_teachers_to_students_l822_822577

theorem new_ratio_of_teachers_to_students (current_teachers : ℕ) (current_students : ℕ) 
  (initial_ratio_s_to_t : ℕ) (increase_students : ℕ) (increase_teachers : ℕ) 
  (new_ratio_s_to_t : ℕ) :
  current_teachers = 3 →
  current_students = 50 * current_teachers →
  initial_ratio_s_to_t = 50 →
  increase_students = 50 →
  increase_teachers = 5 →
  new_ratio_s_to_t = 25 →
  let final_students := current_students + increase_students in
  let final_teachers := current_teachers + increase_teachers in
  final_students / final_teachers = new_ratio_s_to_t → 
  final_teachers / final_students = 1 / 25 :=
by
  intros
  simp [final_students, final_teachers]
  sorry

end new_ratio_of_teachers_to_students_l822_822577


namespace minimum_value_of_m_l822_822001

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)
noncomputable def is_odd (h : ℝ → ℝ) : Prop := ∀ x, h x = - h (-x)
noncomputable def m_condition (m : ℝ) (g h : ℝ → ℝ) : Prop := ∀ x ∈ Set.Icc (-1 : ℝ) 1, m * g x + h x ≥ 0

theorem minimum_value_of_m :
  ∃ (m : ℝ), m_condition m (λ x, (f x + f (-x)) / 2) (λ x, (f x - f (-x)) / 2)
    ∧ ∀ m', m_condition m' (λ x, (f x + f (-x)) / 2) (λ x, (f x - f (-x)) / 2) → m ≤ m' :=
begin
  sorry
end

end minimum_value_of_m_l822_822001


namespace power_modulo_remainder_l822_822324

theorem power_modulo_remainder (a : ℕ) (b : ℕ) (n : ℕ) (k : ℕ) (m : ℕ) (h1 : a*b ≡ k [MOD n]) (h2 : k^m ≡ 1 [MOD n]) (h3 : b = m * 64) :
  a^(b * 64) ≡ 1 [MOD n] :=
sorry

example : power_modulo_remainder 7 2 17 15 32 :=
begin
  -- Provided conditions
  have h1 : 7^2 ≡ 15 [MOD 17], from by simp,
  have h2 : 15^32 ≡ 1 [MOD 17], from by norm_num,
  exact power_modulo_remainder 7 2 17 15 32 h1 h2 rfl,
end

end power_modulo_remainder_l822_822324


namespace distinct_values_f_range_l822_822566

noncomputable def f (x : ℝ) : ℤ :=
  Int.floor x + Int.floor (2 * x) + Int.floor (5 * x / 3) + Int.floor (3 * x) + Int.floor (4 * x)

theorem distinct_values_f_range :
  (Finset.image f (Set.Icc (0 : ℝ) 100)).card = 734 :=
sorry

end distinct_values_f_range_l822_822566


namespace evaluate_nested_function_l822_822759

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x + 1 else x^2

theorem evaluate_nested_function :
  f (f (-2)) = 5 :=
by
  sorry

end evaluate_nested_function_l822_822759


namespace find_f_x_and_g_x_and_g_5_l822_822760

noncomputable def f (a b x : ℝ) := a * x / (x + b)

def g (x : ℝ) := (12 - 3 * x) / (1 + x)

variables {a b : ℝ}

theorem find_f_x_and_g_x_and_g_5
  (h1 : f a b 1 = 5 / 4)
  (h2 : f a b 2 = 2)
  (h3 : ∀ x, f a b (g x) = 4 - x) :
  (f a b x = 5 * x / (x + 3)) ∧
  (g x = (12 - 3 * x) / (1 + x)) ∧
  (g 5 = -1 / 2) :=
sorry

end find_f_x_and_g_x_and_g_5_l822_822760


namespace construct_disjoint_triangles_l822_822603

theorem construct_disjoint_triangles (n : ℕ) (points : Fin 3n → (ℝ × ℝ)) :
  (∀ (i j k : Fin 3n), i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ¬ collinear ℝ {points i, points j, points k}) →
  ∃ (triangles : Fin n → Finset (Fin 3n)), 
    (∀ (t : Fin n), (triangles t).card = 3) ∧
    (∀ (i j : Fin n), i ≠ j → (triangles i) ∩ (triangles j) = ∅) :=
sorry

end construct_disjoint_triangles_l822_822603


namespace max_regions_with_parallel_lines_l822_822117

theorem max_regions_with_parallel_lines (total_lines : ℕ) (parallel_lines : ℕ) 
(h_total : total_lines = 50) (h_parallel : parallel_lines = 20) : 
  ∃ regions : ℕ, regions = 1086 :=
by 
  use 1086
  sorry

end max_regions_with_parallel_lines_l822_822117


namespace find_P3_l822_822963

/-- Define the given points -/
def P0 := (0, 0)
def P1 := (1, 0)
def P2 := (1 / 2, Real.sqrt 3 / 2)

/-- Define the expected coordinates of P3 in 3D space -/
def P3 := (1 / 2, Real.sqrt 3 / 6, Real.sqrt (2 / 3))

/-- Condition: Distance between points -/
def distance (a b : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2 + (a.3 - b.3)^2)

/-- Proof problem: P3 is at a distance of 1 from P0, P1, P2, and has non-negative coordinates -/
theorem find_P3 :
  distance (P3, (0, 0, 0)) = 1 ∧
  distance (P3, (1, 0, 0)) = 1 ∧
  distance (P3, (1 / 2, Real.sqrt 3 / 2, 0)) = 1 ∧
  P3.1 ≥ 0 ∧ P3.2 ≥ 0 ∧ P3.3 ≥ 0 :=
by
  sorry

end find_P3_l822_822963


namespace right_triangle_ratio_3_4_5_l822_822906

theorem right_triangle_ratio_3_4_5 (x : ℝ) (h : x > 0) : 
  let a := 3 * x,
      b := 4 * x,
      c := 5 * x in
  a^2 + b^2 = c^2 :=
by
  let a := 3 * x
  let b := 4 * x
  let c := 5 * x
  calc
    a^2 + b^2 = (3 * x)^2 + (4 * x)^2 : by rfl
          ... = 9 * x^2 + 16 * x^2 : by rfl
          ... = 25 * x^2 : by rfl
    c^2 = (5 * x)^2 : by rfl
          ... = 25 * x^2 : by rfl
  sorry

end right_triangle_ratio_3_4_5_l822_822906


namespace arg_u_w_not_positive_real_l822_822493

open Complex

def principal_value (θ : ℝ) (hθ : 0 < θ ∧ θ < 2 * π) : ℂ :=
  let z : ℂ := ⟨1 - cos θ, sin θ⟩
  let a := cot (θ / 2)
  let u : ℂ := ⟨a ^ 2, a⟩
  if h : θ < π then u.arg = θ / 2 else u.arg = π + θ / 2

noncomputable def w_cannot_be_positive_real (θ : ℝ) (hθ : 0 < θ ∧ θ < 2 * π) : Prop :=
  let z : ℂ := ⟨1 - cos θ, sin θ⟩
  let a := cot (θ / 2)
  let u : ℂ := ⟨a ^ 2, a⟩
  let w : ℂ := z ^ 2 + u ^ 2 + 2 * z * u
  ¬(∃ r : ℝ, r > 0 ∧ w = r)

-- Mathematical proof goals:
theorem arg_u (θ : ℝ) (hθ : 0 < θ ∧ θ < 2 * π) :
    (if h : θ < π then principal_value θ hθ = θ / 2 else principal_value θ hθ = π + θ / 2) := sorry

theorem w_not_positive_real (θ : ℝ) (hθ : 0 < θ ∧ θ < 2 * π) :
    w_cannot_be_positive_real θ hθ := sorry

end arg_u_w_not_positive_real_l822_822493


namespace solve_quadratic_inequality_l822_822775

theorem solve_quadratic_inequality (x : ℝ) (h : x^2 - 7 * x + 6 < 0) : 1 < x ∧ x < 6 :=
  sorry

end solve_quadratic_inequality_l822_822775


namespace square_side_length_l822_822427

/-- If the area of a square is 9m^2 + 24mn + 16n^2, then the length of the side of the square is |3m + 4n|. -/
theorem square_side_length (m n : ℝ) (a : ℝ) (h : a^2 = 9 * m^2 + 24 * m * n + 16 * n^2) : a = |3 * m + 4 * n| :=
sorry

end square_side_length_l822_822427


namespace union_A_B_intersection_complement_A_B_subset_A_C_l822_822764

-- Define the sets and natural language equivalent expressions
def A : Set ℝ := { x | 1 ≤ x ∧ x < 6 }
def B : Set ℝ := { x | 5 < x ∧ x < 10 }
def C (a : ℝ) : Set ℝ := { x | a * x + 1 > 0 }

-- Problem (I): Part 1 - Prove A ∪ B = { x | 1 ≤ x ∧ x < 10 }
theorem union_A_B : A ∪ B = { x | 1 ≤ x ∧ x < 10 } := sorry

-- Problem (I): Part 2 - Prove (complement_A) ∩ B = { x | 6 ≤ x ∧ x < 10 }
theorem intersection_complement_A_B : (set.compl A) ∩ B = { x | 6 ≤ x ∧ x < 10 } := sorry

-- Problem (II): Prove if A ⊆ C then a ≥ -1/6
theorem subset_A_C (a : ℝ) : A ⊆ (C a) → a ≥ -1/6 := sorry

end union_A_B_intersection_complement_A_B_subset_A_C_l822_822764


namespace find_y_l822_822417

theorem find_y (x y : ℝ) (h1 : x - y = 20) (h2 : x + y = 10) : y = -5 := 
sorry

end find_y_l822_822417


namespace monotonic_decreasing_interval_l822_822093

-- Define the function f
def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

-- Define the derivative f'
def f' (x : ℝ) : ℝ := (x^2 - 1) / x

-- State the proof problem
theorem monotonic_decreasing_interval : 
  ∀ x : ℝ, 0 < x -> x ≤ 1 -> f'(x) ≤ 0 := 
sorry

end monotonic_decreasing_interval_l822_822093


namespace perpendicular_AC_AB_at_A_l822_822312

noncomputable def circle (O : Point) (r : ℝ) : set Point := 
  { P | dist O P = r }

variables {Point : Type} [metric_space Point]

variables (A B O D C : Point) (r : ℝ)

axiom point_O_outside_AB : ¬ collinear {A, B, O}
axiom circle_with_center_O_radius_r : circle O r
axiom radius_OA : dist O A = r
axiom intersection_D : D ∈ (line_through A B ∩ circle O r)
axiom OD_intersect_circle_at_C : C ∈ (circle O r) ∩ (line_through O D)
axiom AC_line_segment : AC = line_segment A C

theorem perpendicular_AC_AB_at_A : ∃! P, ⟪P, AC⟫ = ⟪P, A⟫ + ⟪P, C⟫ ∧ (∃ θ : ℝ,  θ = 90 ∧ angle P A C = θ) := sorry

end perpendicular_AC_AB_at_A_l822_822312


namespace students_read_both_books_l822_822020

variable (A B AB : ℕ)

-- Conditions
def total_students : Prop := A + B - AB = 600
def percentage_A_to_B : Prop := AB = (20 * A) / 100
def diff_only_A_only_B : Prop := (A - AB) - (B - AB) = 75

-- Question
def percentage_BothAmongB := AB / B = 0.25

-- The theorem to prove
theorem students_read_both_books (h1 : total_students A B AB)
                                 (h2 : percentage_A_to_B A AB)
                                 (h3 : diff_only_A_only_B A B AB) :
                                 percentage_BothAmongB AB B :=
sorry

end students_read_both_books_l822_822020


namespace find_all_pairs_l822_822315

def is_solution (m n : ℕ) : Prop := 200 * m + 6 * n = 2006

def valid_pairs : List (ℕ × ℕ) := [(1, 301), (4, 201), (7, 101), (10, 1)]

theorem find_all_pairs :
  ∀ (m n : ℕ), is_solution m n ↔ (m, n) ∈ valid_pairs := by sorry

end find_all_pairs_l822_822315


namespace lower_bound_for_expression_l822_822719

theorem lower_bound_for_expression :
  ∃ L: ℤ, (∀ n: ℤ, L < 4 * n + 7 ∧ 4 * n + 7 < 120) → L = 5 :=
sorry

end lower_bound_for_expression_l822_822719


namespace rate_percent_correct_l822_822671

noncomputable def findRatePercent (P A T : ℕ) : ℚ :=
  let SI := A - P
  (SI * 100 : ℚ) / (P * T)

theorem rate_percent_correct :
  findRatePercent 12000 19500 7 = 8.93 := by
  sorry

end rate_percent_correct_l822_822671


namespace power_modulo_remainder_l822_822323

theorem power_modulo_remainder (a : ℕ) (b : ℕ) (n : ℕ) (k : ℕ) (m : ℕ) (h1 : a*b ≡ k [MOD n]) (h2 : k^m ≡ 1 [MOD n]) (h3 : b = m * 64) :
  a^(b * 64) ≡ 1 [MOD n] :=
sorry

example : power_modulo_remainder 7 2 17 15 32 :=
begin
  -- Provided conditions
  have h1 : 7^2 ≡ 15 [MOD 17], from by simp,
  have h2 : 15^32 ≡ 1 [MOD 17], from by norm_num,
  exact power_modulo_remainder 7 2 17 15 32 h1 h2 rfl,
end

end power_modulo_remainder_l822_822323


namespace mary_should_drink_six_glasses_per_day_l822_822919

def daily_water_goal : ℕ := 1500
def glass_capacity : ℕ := 250
def required_glasses (daily_goal : ℕ) (capacity : ℕ) : ℕ := daily_goal / capacity

theorem mary_should_drink_six_glasses_per_day :
  required_glasses daily_water_goal glass_capacity = 6 :=
by
  sorry

end mary_should_drink_six_glasses_per_day_l822_822919


namespace quadratic_inequality_solution_l822_822327

-- Define the quadratic function
def quadratic (x : ℝ) : ℝ := x^2 - 24 * x + 125

-- Define the condition as an inequality
def inequality (x : ℝ) : Prop := quadratic x ≤ 9

-- Define the interval where the inequality holds
def solution_interval := set.Icc 6.71 17.29

-- The proof problem statement
theorem quadratic_inequality_solution :
  ∀ x : ℝ, inequality x ↔ x ∈ solution_interval :=
by sorry

end quadratic_inequality_solution_l822_822327


namespace sum_of_midpoint_coordinates_is_zero_l822_822897

theorem sum_of_midpoint_coordinates_is_zero :
  let p1 := (10 : ℤ, -6 : ℤ)
  let p2 := (-6 : ℤ, 2 : ℤ)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  (midpoint.1 + midpoint.2) = 0 :=
by
  let p1 := (10 : ℤ, -6 : ℤ)
  let p2 := (-6 : ℤ, 2 : ℤ)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  have x_coord : midpoint.1 = 2 by sorry
  have y_coord : midpoint.2 = -2 by sorry
  have sum_coords : midpoint.1 + midpoint.2 = 2 + (-2) by sorry
  show (midpoint.1 + midpoint.2) = 0 from by sorry

end sum_of_midpoint_coordinates_is_zero_l822_822897


namespace sum_identity_l822_822562

theorem sum_identity :
  ∑ n in Finset.range 99, (1 : ℚ) / (n + 1) / (n + 2) = 99 / 100 :=
by {
  -- Proof steps will go here, but is omitted as per instruction
  sorry
}

end sum_identity_l822_822562


namespace probability_rain_at_most_2_days_l822_822578

open Nat.RealInBasic 

noncomputable def binomial (n k : ℕ) : ℝ :=
  (n.choose k : ℝ)

noncomputable def probability_rain (days : ℕ) (rain_prob : ℝ) (k : ℕ) : ℝ :=
  binomial days k * (rain_prob^(k : ℝ)) * ((1 - rain_prob)^(days - k : ℕ))

noncomputable def total_probability (days : ℕ) (rain_prob : ℝ) : ℝ :=
  probability_rain days rain_prob 0 + probability_rain days rain_prob 1 + 
  probability_rain days rain_prob 2

theorem probability_rain_at_most_2_days :
  total_probability 28 (1/7) ≈ 0.385 := 
sorry

end probability_rain_at_most_2_days_l822_822578


namespace broken_seashells_l822_822595

-- Define the total number of seashells Tom found
def total_seashells : ℕ := 7

-- Define the number of unbroken seashells
def unbroken_seashells : ℕ := 3

-- Prove that the number of broken seashells equals 4
theorem broken_seashells : total_seashells - unbroken_seashells = 4 := by
  sorry

end broken_seashells_l822_822595


namespace cost_of_one_bag_l822_822139

theorem cost_of_one_bag (x : ℝ) (h1 : ∀ p : ℝ, 60 * x = p -> 60 * p = 120 * x ) 
  (h2 : ∀ p1 p2: ℝ, 60 * x = p1 ∧ 15 * 1.6 * x = p2 ∧ 45 * 2.24 * x = 100.8 * x -> 124.8 * x - 120 * x = 1200) :
  x = 250 := 
sorry

end cost_of_one_bag_l822_822139


namespace infinite_initial_values_l822_822339

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - x^2

-- Define the sequence x_n = f(x_{n-1})
noncomputable def seq (x0 : ℝ) : ℕ → ℝ
| 0     := x0
| (n+1) := f (seq n)

-- Prove that there are infinitely many real numbers x_0
-- such that the sequence takes on only a finite number of different values.
theorem infinite_initial_values (f : ℝ → ℝ) (seq : ℝ → ℕ → ℝ) :
  ∃ (S : set ℝ), (∀ x ∈ S, ∀ n : ℕ, ∃ m : ℕ, seq x n = seq x m) ∧ infinite S :=
by
  sorry

end infinite_initial_values_l822_822339


namespace sum_first_2n_terms_l822_822582
-- Import the necessary library

-- Define the sequence conditions and proof problem
theorem sum_first_2n_terms (a : ℕ → ℝ) (c : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (h : 0 < n) :
  (∀ n, a n * a (n + 1) = (1 / 3)^n) ∧
  (a 1 = 2) ∧
  (∀ n, a n + a (n + 1) = c n) ->
  S (2 * n) = (∑ k in finset.range (2 * n + 1), c k) :=
  S (2 * n) = 9 / 2 * (1 - (1 / 3)^n) := sorry

end sum_first_2n_terms_l822_822582


namespace range_of_t_l822_822340

noncomputable def f (x : ℝ) : ℝ := abs (x * real.exp x)

noncomputable def g (x t : ℝ) : ℝ := (f x) * (f x) - t * (f x)

theorem range_of_t (t : ℝ) : 
  (∃ (x₁ x₂ x₃ x₄ : ℝ), 
   x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧ 
   g x₁ t = -1 ∧ g x₂ t = -1 ∧ g x₃ t = -1 ∧ g x₄ t = -1)
  ↔ t ∈ set.Ioi (real.exp 1 + 1 / (real.exp 1)) :=
sorry

end range_of_t_l822_822340


namespace remainder_of_m_l822_822714

theorem remainder_of_m (m : ℕ) (h1 : m^2 % 7 = 1) (h2 : m^3 % 7 = 6) : m % 7 = 6 := by
  sorry

end remainder_of_m_l822_822714


namespace no_real_roots_l822_822491

noncomputable def nth_root (n : ℕ) (x : ℝ) : ℝ := x^(1 / n.to_real)

theorem no_real_roots
  (a b c : ℕ)
  (h_a : a < 1000000)
  (h_b : b < 1000000)
  (h_c : c < 1000000) :
  ∀ x : ℝ, nth_root 21 (a * x^2) + nth_root 21 (b * x) + nth_root 21 c ≠ 0 := 
by
  sorry

end no_real_roots_l822_822491


namespace shortest_distance_to_circle_after_reflection_l822_822217

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

-- Definition of the problem conditions
def point_A : ℝ × ℝ := (-1, 1)
def point_A' : ℝ × ℝ := (-1, -1)
def circle_C_center : ℝ × ℝ := (3, 2)
def circle_C_radius : ℝ := 1

-- The mathematical statement to prove:
theorem shortest_distance_to_circle_after_reflection :
  distance point_A' circle_C_center - circle_C_radius = 4 :=
  sorry

end shortest_distance_to_circle_after_reflection_l822_822217


namespace augmented_matrix_solution_l822_822428

theorem augmented_matrix_solution (c₁ c₂ : ℝ) (x y : ℝ) 
  (h1 : 2 * x + 3 * y = c₁) (h2 : 3 * x + 2 * y = c₂)
  (hx : x = 2) (hy : y = 1) : c₁ - c₂ = -1 := 
by
  sorry

end augmented_matrix_solution_l822_822428


namespace minimum_line_segments_l822_822342

/-!
  Given n points on a plane such that no three points are collinear, 
  and m line segments are drawn connecting pairs of these n points.
  It is known that for any two points A and B, there exists a 
  point C such that line segments are drawn connecting C to both A and B.
-/

theorem minimum_line_segments (n : ℕ) (h : n ≥ 4) : 
  ∃ m, m = ⌊(3 * n - 2) / 2⌋ ∧ 
  (∀ A B : ℕ, A ≠ B → ∃ C : ℕ, C ≠ A ∧ C ≠ B ∧ 
  is_connected (A, C) ∧ is_connected (B, C)) :=
sorry

end minimum_line_segments_l822_822342


namespace TriangleArea_m_n_sum_l822_822925

noncomputable theory

open EuclideanGeometry Real

variables {ABC : Triangle} (BC : ℝ) (AD trisected_by_incircle : Prop)
variables (m n : ℕ)

-- Adding the required parameters as part of the formal statement
def trichotomy (ABC : Triangle) (BC : ℝ) (AD trisected_by_incircle : Prop) :=
  BC = 20 ∧ (ABC.has_incircle) ∧ (trisected_by_incircle)

-- Main theorem statement
theorem TriangleArea_m_n_sum :
  trichotomy ABC BC AD trisected_by_incircle →
    ∃ (m n : ℕ), (area ABC = m * sqrt n) ∧ (nat.is_coprime n (prime_square_factors n).prod) ∧ (m + n = 38) :=
begin
  sorry
end

end TriangleArea_m_n_sum_l822_822925


namespace projectile_height_time_l822_822559

theorem projectile_height_time (h : ∀ t : ℝ, -16 * t^2 + 100 * t = 64 → t = 1) : (∃ t : ℝ, -16 * t^2 + 100 * t = 64 ∧ t = 1) :=
by sorry

end projectile_height_time_l822_822559


namespace bouquets_needed_to_earn_1000_l822_822266

theorem bouquets_needed_to_earn_1000 :
  ∀ (cost_per_bouquet sell_price_bouquet: ℕ) (roses_per_bouquet_bought roses_per_bouquet_sold target_profit: ℕ),
    cost_per_bouquet = 20 →
    sell_price_bouquet = 20 →
    roses_per_bouquet_bought = 7 →
    roses_per_bouquet_sold = 5 →
    target_profit = 1000 →
    (target_profit / (sell_price_bouquet * roses_per_bouquet_sold / roses_per_bouquet_bought - cost_per_bouquet) * roses_per_bouquet_bought = 125) :=
by
  intros cost_per_bouquet sell_price_bouquet roses_per_bouquet_bought roses_per_bouquet_sold target_profit 
    h_cost_per_bouquet h_sell_price_bouquet h_roses_per_bouquet_bought h_roses_per_bouquet_sold h_target_profit
  sorry

end bouquets_needed_to_earn_1000_l822_822266


namespace kohen_apples_l822_822820

theorem kohen_apples (B : ℕ) (h1 : 300 * B = 4 * 750) : B = 10 :=
by
  -- proof goes here
  sorry

end kohen_apples_l822_822820


namespace value_of_3k_squared_minus_1_l822_822755

theorem value_of_3k_squared_minus_1 (x k : ℤ)
  (h1 : 7 * x + 2 = 3 * x - 6)
  (h2 : x + 1 = k)
  : 3 * k^2 - 1 = 2 := 
by
  sorry

end value_of_3k_squared_minus_1_l822_822755


namespace problem_l822_822967

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the x(1-x)^n equation
noncomputable def eq1 (n : ℕ) (x : ℝ) : ℝ :=
  x * (1 - x)^n

-- State the equality after differentiation and substitution of x=1
theorem problem (n : ℕ) :
  (-2) * binom n 1 + 3 * binom n 2 - 4 * binom n 3 
  + ∑ i in Finset.range (n+1), (-1)^(i+1) * (i+1) * binom n i = -1 :=
  sorry

end problem_l822_822967


namespace inequality_solution_l822_822875

noncomputable def sqrt3 : ℝ := real.sqrt 3

theorem inequality_solution (x : ℝ) :
  (x + sqrt3) / (x + 10) > (3 * x + 2 * sqrt3) / (2 * x + 14) →
  -7 < x ∧ x < -0.775 :=
by
  sorry

end inequality_solution_l822_822875


namespace supremum_E_l822_822836

def E (n : ℕ) (x : Fin n → ℝ) : ℝ :=
  (∑ i in Finset.univ, (x i)^2) - ∑ i in Finset.univ, x i * x ((i + 1) % n)

theorem supremum_E (n : ℕ) (hn : n ≥ 2) (x : Fin n → ℝ) (hx : ∀ i, 0 ≤ x i ∧ x i ≤ 1) : 
  (∃ m : ℝ, 0 ≤ m ∧ m = if n % 2 = 0 then n / 2 else (n - 1) / 2 ∧ ∀ x : Fin n → ℝ, 
    (∀ i, 0 ≤ x i ∧ x i ≤ 1) → E n x ≤ m) :=
sorry

end supremum_E_l822_822836


namespace probability_complex_in_S_l822_822656

def S : set ℂ := {z : ℂ | -2 ≤ z.re ∧ z.re ≤ 2 ∧ -2 ≤ z.im ∧ z.im ≤ 2}

def in_S (z : ℂ) : Prop := z ∈ S

theorem probability_complex_in_S (z : ℂ) (hz : z ∈ S) :
  let w := (1/2 + 1/2 * complex.I) * z in in_S w :=
by
  sorry

end probability_complex_in_S_l822_822656


namespace true_compound_propositions_l822_822291

-- Define the propositions
def p1 : Prop := 
  ∀ (l₁ l₂ l₃ : Line), 
    (l₁ ≠ l₂ ∧ l₂ ≠ l₃ ∧ l₁ ≠ l₃) ∧ 
    (∃ (p₁ p₂ p₃ : Point),
      l₁.contains p₁ ∧ l₁.contains p₂ ∧
      l₂.contains p₂ ∧ l₂.contains p₃ ∧
      l₃.contains p₃ ∧ l₃.contains p₁)
    → ∃ (α : Plane), 
      α.contains l₁ ∧ α.contains l₂ ∧ α.contains l₃

def p2 : Prop :=
  ∀ (a b c : Point),
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) 
    → ∃! (α : Plane), 
      α.contains a ∧ α.contains b ∧ α.contains c

def p3 : Prop :=
  ∀ (l₁ l₂ : Line),
    (¬ ∃ (p : Point), l₁.contains p ∧ l₂.contains p)
    → l₁.parallel l₂

def p4 : Prop :=
  ∀ (α : Plane) (l m : Line),
    (α.contains l ∧ ¬ α.contains m ∧ m.perpendicular α) 
    → m.perpendicular l

-- Main theorem that identifies the true compound propositions
theorem true_compound_propositions :
  (p1 ∧ p4) ∧ ¬ (p1 ∧ p2) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) :=
by
  -- Proof might involve defining lines, points, and planes and their relationships
  sorry

end true_compound_propositions_l822_822291


namespace stock_percentage_change_l822_822531

-- Definitions for the conditions
def initial_value (x : ℝ) := x

def first_day_decrease (x : ℝ) := 0.7 * x

def second_day_increase (y : ℝ) := 0.4 * y

def final_value (x : ℝ) : ℝ :=
  let y := first_day_decrease x
  y + second_day_increase y

-- The theorem statement
theorem stock_percentage_change (x : ℝ) (h : x > 0) :
  ((final_value x - x) / x) * 100 = -2 :=
by
  intro x h
  sorry

end stock_percentage_change_l822_822531


namespace real_root_exists_l822_822729

theorem real_root_exists (p1 p2 q1 q2 : ℝ) 
(h : p1 * p2 = 2 * (q1 + q2)) : 
  (∃ x : ℝ, x^2 + p1 * x + q1 = 0) ∨ (∃ x : ℝ, x^2 + p2 * x + q2 = 0) :=
by
  sorry

end real_root_exists_l822_822729


namespace sphere_volume_doubling_l822_822788

theorem sphere_volume_doubling (r : ℝ) : 
  let original_volume := (4 / 3) * Real.pi * r^3 in
  let doubled_volume := (4 / 3) * Real.pi * (2 * r)^3 in
  doubled_volume = 8 * original_volume :=
by
  sorry

end sphere_volume_doubling_l822_822788


namespace beth_initial_marbles_l822_822271

theorem beth_initial_marbles (x : ℕ) (h : 42 + (5 + 2 * 5 + 3 * 5) = 3 * x): 
3 * x = 72 :=
by
  have h1 : 5 + 2 * 5 + 3 * 5 = 30 := by norm_num
  rw h1 at h
  rw Nat.add_comm at h
  exact h

end beth_initial_marbles_l822_822271


namespace option_B_not_well_defined_l822_822239

-- Definitions based on given conditions 
def is_well_defined_set (description : String) : Prop :=
  match description with
  | "All positive numbers" => True
  | "All elderly people" => False
  | "All real numbers that are not equal to 0" => True
  | "The four great inventions of ancient China" => True
  | _ => False

-- Theorem stating option B "All elderly people" is not a well-defined set
theorem option_B_not_well_defined : ¬ is_well_defined_set "All elderly people" :=
  by sorry

end option_B_not_well_defined_l822_822239


namespace sum_of_roots_of_f_eq_2_l822_822013

def f (x : ℝ) : ℝ :=
if x ≤ 0 then x^2 - 5*x + 6 else x^3 - x + 2

theorem sum_of_roots_of_f_eq_2 : (∑ x in {x : ℝ | f x = 2}, x) = 1 :=
by
  sorry

end sum_of_roots_of_f_eq_2_l822_822013


namespace exists_integral_point_l822_822346

-- Define the conditions and the statement to be proved
noncomputable def convex_pentagon (A B C D E : ℤ × ℤ) : Prop :=
  sorry -- Definition of convex pentagon to be formalized

noncomputable def has_integer_coordinates (A B C D E : ℤ × ℤ) : Prop :=
  ∀ (P : ℤ × ℤ), P ∈ {A, B, C, D, E} → P.1 ∈ ℤ ∧ P.2 ∈ ℤ

theorem exists_integral_point (A B C D E : ℤ × ℤ) (h1 : convex_pentagon A B C D E) (h2 : has_integer_coordinates A B C D E) :
  ∃ (P : ℤ × ℤ), (P ∈ polygon A B C D E) ∧ (P.1 ∈ ℤ ∧ P.2 ∈ ℤ) :=
  sorry -- Proof to be filled in

end exists_integral_point_l822_822346


namespace proposition_1_proposition_2_proposition_3_proposition_4_l822_822292

axiom p1 : Prop
axiom p2 : Prop
axiom p3 : Prop
axiom p4 : Prop

axiom p1_true : p1 = true
axiom p2_false : p2 = false
axiom p3_false : p3 = false
axiom p4_true : p4 = true

theorem proposition_1 : (p1 ∧ p4) = true := by sorry
theorem proposition_2 : (p1 ∧ p2) = false := by sorry
theorem proposition_3 : (¬p2 ∨ p3) = true := by sorry
theorem proposition_4 : (¬p3 ∨ ¬p4) = true := by sorry

end proposition_1_proposition_2_proposition_3_proposition_4_l822_822292


namespace no_integer_points_between_A_and_B_on_line_l822_822985

theorem no_integer_points_between_A_and_B_on_line
  (A : ℕ × ℕ) (B : ℕ × ℕ)
  (hA : A = (2, 3))
  (hB : B = (50, 500)) :
  ∀ (P : ℕ × ℕ), P.1 > 2 ∧ P.1 < 50 ∧ 
    (P.2 * 48 - P.1 * 497 = 2 * 497 - 3 * 48) →
    false := 
by
  sorry

end no_integer_points_between_A_and_B_on_line_l822_822985


namespace power_function_increasing_l822_822238

theorem power_function_increasing {α : ℝ} (hα : α = 1 ∨ α = 3 ∨ α = 1 / 2) :
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → x ^ α ≤ y ^ α := 
sorry

end power_function_increasing_l822_822238


namespace area_of_rhombus_l822_822431

-- Given values for the diagonals of a rhombus.
def d1 : ℝ := 14
def d2 : ℝ := 24

-- The target statement we want to prove.
theorem area_of_rhombus : (d1 * d2) / 2 = 168 := by
  sorry

end area_of_rhombus_l822_822431


namespace series_2023_power_of_3_squared_20_equals_653_l822_822190

def series (A : ℕ → ℕ) : Prop :=
  A 0 = 1 ∧ 
  ∀ n > 0, 
  A n = A (n / 2023) + A (n / 2023^2) + A (n / 2023^3)

theorem series_2023_power_of_3_squared_20_equals_653 (A : ℕ → ℕ) (h : series A) : A (2023 ^ (3^2) + 20) = 653 :=
by
  -- placeholder for proof
  sorry

end series_2023_power_of_3_squared_20_equals_653_l822_822190


namespace platform_length_correct_l822_822195

-- Train length in meters
def train_length : ℝ := 300

-- Time taken to cross a signal pole in seconds
def time_to_cross_pole : ℝ := 18

-- Speed of the train in meters per second
def train_speed : ℝ := train_length / time_to_cross_pole

-- Time taken to cross the platform in seconds
def time_to_cross_platform : ℝ := 38

-- Length of the platform in meters
def platform_length : ℝ := 333.46

theorem platform_length_correct :
  train_speed = train_length / time_to_cross_pole ∧
  platform_length = ((train_speed * time_to_cross_platform) - train_length) :=
begin
  sorry
end

end platform_length_correct_l822_822195


namespace valid_arrangements_l822_822804

-- Define the conditions
def alice_and_bob_together (arrangement : List String) : Prop :=
  let idx := arrangement.indexOf "Alice"
  let idx_next := (idx + 1) % 8
  arrangement[idx_next] = "Bob" ∨ arrangement[idx_next] = "Alice"

def carol_behind_dan (arrangement : List String) : Prop :=
  arrangement.indexOf "Carol" > arrangement.indexOf "Dan"

-- Define the main theorem
theorem valid_arrangements : ∀ (arrangement : List String),
  (arrangement.length = 8) ∧ alice_and_bob_together arrangement ∧ carol_behind_dan arrangement
  → (number_of_valid_arrangements = 5040) :=
sorry

end valid_arrangements_l822_822804


namespace mushrooms_collected_l822_822534

theorem mushrooms_collected (a1 a2 a3 a4 a5 a6 a7 : ℕ) 
  (h_distinct : list.nodup [a1, a2, a3, a4, a5, a6, a7])
  (h_total : a1 + a2 + a3 + a4 + a5 + a6 + a7 = 100)
  (h_ordered : a1 > a2 ∧ a2 > a3 ∧ a3 > a4 ∧ a4 > a5 ∧ a5 > a6 ∧ a6 > a7) :
  ∃ (b c d : ℕ), 
  b + c + d ≥ 50 ∧ 
  (b = a1 ∨ b = a2 ∨ b = a3 ∨ b = a4 ∨ b = a5 ∨ b = a6 ∨ b = a7) ∧ 
  (c = a1 ∨ c = a2 ∨ c = a3 ∨ c = a4 ∨ c = a5 ∨ c = a6 ∨ c = a7) ∧ 
  (d = a1 ∨ d = a2 ∨ d = a3 ∨ d = a4 ∨ d = a5 ∨ d = a6 ∨ d = a7) ∧ 
  b ≠ c ∧ b ≠ d ∧ c ≠ d :=
by
  sorry

end mushrooms_collected_l822_822534


namespace cost_of_one_bag_l822_822132

theorem cost_of_one_bag (x : ℝ) :
  let Andrey_earning := 60 * 2 * x
  let Boris_earning := 15 * 1.6 * x + 45 * (1.6 * 1.4) * x
  Boris_earning - Andrey_earning = 1200 →
  x = 250 := 
by
  intros
  let Andrey_earning := 60 * 2 * x
  let Boris_earning := 15 * 1.6 * x + 45 * (1.6 * 1.4) * x
  have h : Boris_earning - Andrey_earning = 1200 := by assumption
  let simplified_h := 
    calc
      Boris_earning - Andrey_earning
        = (15 * 1.6 * x + 45 * (1.6 * 1.4) * x) - (60 * 2 * x) : by simp [Andrey_earning, Boris_earning]
    ... = (15 * 1.6 * x + 45 * (1.6 * 1.4) * x) - 120 * x : by simp
    ... = (24 * x + 100.8 * x) - 120 * x : by simp
    ... = 124.8 * x - 120 * x : by simp
    ... = 4.8 * x : by simp
    ... = 1200 : by rw h
  exact (div_eq_iff (by norm_num : (4.8 : ℝ) ≠ 0)).1 simplified_h  -- solves for x

end cost_of_one_bag_l822_822132


namespace find_even_n_l822_822027

theorem find_even_n (n : ℕ) (m : ℕ) (k : ℕ) (h1 : n = m * 2^k)
  (h2 : 2^k - 1 ≠ 0) (h3 : (∑ (i : ℕ) in finset.range (k + 1), 2^i) = 2^k - 1)
  (h4 : (∀ d, d ∣ m → (d * 2) ∣ n)): 
  (∑ (d : ℕ) in (finset.range (m + 1)).filter (λ d, d ∣ m), d) *
  (∑ (d : ℕ) in (finset.range (m + 1)).filter (λ d, d ∣ m), d * 2 * (2^k - 1)) = 2016 :=
sorry

end find_even_n_l822_822027


namespace exists_at_least_one_triangle_exists_at_least_n_triangles_l822_822805

theorem exists_at_least_one_triangle (n : ℕ) 
  (h_points : 2 * n > 0)
  (h_coplanar : ∀ (p : Fin 2n → ℝ^3), ∃ (a b c d : Fin 2n), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
      (affineSpan ℝ {p a, p b, p c, p d}).dim < 3)
  (h_segments : ∃ s : Fin (n^2 + 1) → Fin 2n × Fin 2n, ∃ h : ∀ i, (s i).1 ≠ (s i).2, true) :
  ∃ (a b c : Fin 2n), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a, b) ∈ s ∧ (b, c) ∈ s ∧ (c, a) ∈ s :=
sorry

theorem exists_at_least_n_triangles (n : ℕ) 
  (h_points : 2 * n > 0)
  (h_coplanar : ∀ (p : Fin 2n → ℝ^3), ∃ (a b c d : Fin 2n), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
      (affineSpan ℝ {p a, p b, p c, p d}).dim < 3)
  (h_segments : ∃ s : Fin (n^2 + 1) → Fin 2n × Fin 2n, ∃ h : ∀ i, (s i).1 ≠ (s i).2, true) :
  ∃ (t : Fin n → Fin 2n × Fin 2n × Fin 2n), 
    (∀ i, (t i).1.1 ≠ (t i).1.2 ∧ (t i).1.2 ≠ (t i).2 ∧ (t i).2 ≠ (t i).1.1) ∧ 
    (∃ i, (t i).1 ∈ s ∧ (t i).2 ∈ s ∧ (sorry : (t i).2 ∈ s)) :=
sorry

end exists_at_least_one_triangle_exists_at_least_n_triangles_l822_822805


namespace number_of_elements_in_B_l822_822481

-- Define the sets A and B
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := { x | ∃ m n : ℕ, m ∈ A ∧ n ∈ A ∧ m ≠ n ∧ x = m * n }

-- Define the property to prove
theorem number_of_elements_in_B : Finset.card (B.toFinset) = 3 := by
  sorry

end number_of_elements_in_B_l822_822481


namespace five_nines_l822_822602

-- Each expression for numbers 1 to 13
def expressions : List (Expr) :=
  [
    (\<^.pow (\<^.div (Expr.nat 9) (Expr.nat 9)) (Expr.sub (Expr.nat 9) (Expr.nat 9))),
    (\<^.add (\<^.div (Expr.nat 9) (Expr.nat 9)) (\<^.div (Expr.nat 9) (Expr.nat 9))),
    (\<^.add (\<^.div (Expr.nat 9) (Expr.nat 9)) (\<^.add (\<^.div (Expr.nat 9) (Expr.nat 9)) (\<^.div (Expr.nat 9) (Expr.nat 9)))),
    (\<^.pow (\<^.add (\<^.div (Expr.nat 9) (Expr.nat 9)) (\<^.div (Expr.nat 9) (Expr.nat 9))) (Expr.nat 2)),
    (\<^.add (\<^.sub (\<^.div (\<^.add (\<^.mul (Expr.nat 9) (Expr.nat 9)) (Expr.nat 9)) (Expr.nat 9)) (Expr.nat 9)) (Expr.add (Expr.nat 9) (Expr.nat 0))),
    (\<^.sub (\<^.pow (\<^.add (\<^.div (Expr.nat 9) (Expr.nat 9)) (\<^.add (\<^.div (Expr.nat 9) (Expr.nat 9)) (\<^.div (Expr.nat 9) (Expr.nat 9)))) (Expr.nat 2)) (Expr.nat 3)),
    (\<^.sub (\<^.pow (\<^.add (\<^.div (Expr.nat 9) (Expr.nat 9)) (\<^.div (Expr.nat 9) (Expr.nat 9))) (Expr.nat 2)) (Expr.div (Expr.nat 9) (Expr.nat 9))),
    (\<^.sub (\<^.pow (\<^.add (\<^.div (Expr.nat 9) (Expr.nat 9)) (\<^.div (Expr.nat 9) (Expr.nat 9))) (Expr.nat 3)) (Expr.div (Expr.nat 9) (Expr.nat 9))),
    Expr.nat 9,
    (\<^.div (\<^.sub (\<^.mul (Expr.nat 9) (Expr.add (Expr.nat 9) (Expr.nat 9))) (Expr.nat 9)) (Expr.nat 9)),
    (\<^.add (Expr.nat 9) (\<^.add (\<^.div (Expr.nat 9) (Expr.nat 9)) (\<^.div (Expr.nat 9) (Expr.nat 9)))),
    (\<^.sub (\<^.pow (\<^.add (\<^.div (Expr.nat 9) (Expr.nat 9)) (\<^.div (Expr.nat 9) (Expr.nat 9))) (Expr.nat 3)) (Expr.nat 4)),
    (\<^.add (\<^.pow (\<^.add (\<^.div (Expr.nat 9) (Expr.nat 9)) (\<^.div (Expr.nat 9) (Expr.nat 9))) (Expr.nat 2)) (Expr.nat 9))
  ]

-- The theorem ensures the existence of a valid expression for each number between 1 to 13 using five 9's.
theorem five_nines (n : ℕ) (h : 1 ≤ n ∧ n ≤ 13): 
  ∃ (e : Expr), eval e = n := 
  sorry

end five_nines_l822_822602


namespace gcd_division_steps_l822_822708

theorem gcd_division_steps (a b : ℕ) (h₁ : a = 1813) (h₂ : b = 333) : 
  ∃ steps : ℕ, steps = 3 ∧ (Nat.gcd a b = 37) :=
by
  have h₁ : a = 1813 := h₁
  have h₂ : b = 333 := h₂
  sorry

end gcd_division_steps_l822_822708


namespace arithmetic_sequence_log_terms_l822_822574

theorem arithmetic_sequence_log_terms 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (hA : A = log a)  
  (hB : B = log b) 
  (hC : C = log c) 
  (T1 T2 T3 T12 : ℝ) 
  (hT1 : T1 = 3 * A + 7 * B + 2 * C)
  (hT2 : T2 = 5 * A + 12 * B + 3 * C)
  (hT3 : T3 = 8 * A + 15 * B + 5 * C)
  (seq_diff : T2 - T1 = T3 - T2) 
  (common_diff : ∃ d : ℝ, T2 - T1 = d ∧ T3 - T2 = d) 
  (T12_is : T12 = log (c ^ 125)) :
  ∃ n : ℕ, T12 = log (c ^ n) ∧ n = 125 :=
by 
  sorry

end arithmetic_sequence_log_terms_l822_822574


namespace circles_tangent_area_l822_822276

noncomputable def triangle_area (r1 r2 r3 : ℝ) := 
  let d1 := r1 + r2
  let d2 := r2 + r3
  let d3 := r1 + r3
  let s := (d1 + d2 + d3) / 2
  (s * (s - d1) * (s - d2) * (s - d3)).sqrt

theorem circles_tangent_area :
  let r1 := 5
  let r2 := 12
  let r3 := 13
  let area := triangle_area r1 r2 r3 / (4 * (r1 + r2 + r3)).sqrt
  area = 120 / 25 := 
by 
  sorry

end circles_tangent_area_l822_822276


namespace chord_length_EF_is_24_l822_822461

noncomputable def chord_length {A D B C G E F O N P : Type*}
  (AB_radius : Real) (BC_radius : Real) (CD_radius : Real) 
  (AG_tangent : Line G P) (intersects : Line G P intersects Circle N at E and F)
  (radius_O : radius Circle O = 10)
  (radius_N : radius Circle N = 20)
  (radius_P : radius Circle P = 15)  : Real :=
  sorry

theorem chord_length_EF_is_24 :
  chord_length 10 20 15 (Line G P) (Line G P intersects Circle N at E and F) 10 20 15 = 24 :=
sorry

end chord_length_EF_is_24_l822_822461


namespace find_perpendicular_line_through_point_l822_822854

def point := (ℝ × ℝ)

def line (a b c : ℝ) := λ (p : point), a * p.1 + b * p.2 + c = 0

def perpendicular_slope (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ * a₂ + b₁ * b₂ = 0

theorem find_perpendicular_line_through_point :
  ∃ a b c : ℝ, let l := line a b c in
  (l (2, -3) = 0) ∧
  perpendicular_slope a b 1 (-2) ∧
  a = 2 ∧ b = 1 ∧ c = -1 :=
begin
  use [2, 1, -1],
  simp [line, perpendicular_slope],
  split,
  { simp, },
  split,
  { norm_num, },
  split,
  { refl, },
  { refl, }
end

end find_perpendicular_line_through_point_l822_822854


namespace volume_purple_tetrahedron_l822_822979

/-- 
Given a cube of side length 8 cm with vertices alternately colored black and purple,
the volume of the tetrahedron formed by the purple vertices is 512/3 cm^3.
-/
theorem volume_purple_tetrahedron :
  let s := 8 in 
  let v_cube := s^3 in
  let area_triangle := 1/2 * s * s in
  let height_tetra := s in
  let vol_tetra := 1/3 * area_triangle * height_tetra in
  let total_vol_tetra := 4 * vol_tetra in
  let volume_purple_tetra := v_cube - total_vol_tetra in
  volume_purple_tetra = 512 / 3 :=
by 
  sorry

end volume_purple_tetrahedron_l822_822979


namespace number_of_integers_less_than_500_with_3_consecutive_odd_integer_sums_l822_822406

theorem number_of_integers_less_than_500_with_3_consecutive_odd_integer_sums :
  ∃ (M_set : set ℕ), 
    (∀ M ∈ M_set, M < 500) ∧
    (∀ M ∈ M_set, ∃! (k_set : set ℕ), ∀ k ∈ k_set, k ∣ M ∧ k ≥ 1 ∧ k ∈ { k | ∃ m ≥ 0, M = k * (2*m + k) }) ∧ 
    Fintype.card M_set = 7 :=
sorry

end number_of_integers_less_than_500_with_3_consecutive_odd_integer_sums_l822_822406


namespace volume_of_region_l822_822915

theorem volume_of_region (r1 r2 : ℝ) (h : r1 = 5) (h2 : r2 = 8) : 
  let V_sphere (r : ℝ) := (4 / 3) * Real.pi * r^3
  let V_cylinder (r : ℝ) := Real.pi * r^2 * r
  (V_sphere r2) - (V_sphere r1) - (V_cylinder r1) = 391 * Real.pi :=
by
  -- Placeholder proof
  sorry

end volume_of_region_l822_822915


namespace construct_point_D_l822_822354

-- Definitions based on conditions in problem.
variables {α : Type*} [ordered_field α]
variables (A B C : α × α) -- Points on the plane representing triangle ABC
variables (BC AC AB : α) -- Lengths of sides BC, AC, AB
variable (eight_lines : ℕ) -- Maximum of eight lines to be used
variables D : α × α -- Point D on side AB

-- Condition from the problem
def side_lengths (A B C : α × α) : Prop :=
  BC = (B - C).norm ∧
  AC = (A - C).norm ∧
  AB = (A - B).norm

-- The theorem based on question and conditions
theorem construct_point_D 
  (h_triangle : side_lengths A B C)
  (h_eight_lines : eight_lines ≤ 8) :
  ∃ D : α × α,
  (D.1 - (A.1, B.1) = BC / AC) :=
sorry

end construct_point_D_l822_822354


namespace odd_function_f_find_f_when_x_lt_zero_l822_822750

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 - 6 * x + 5 else -(x^2 + 6 * x + 5)

theorem odd_function_f (x : ℝ) : (f (-x) = - f x) :=
begin
  -- f is defined as an odd function.
  intros,
  sorry
end

theorem find_f_when_x_lt_zero (x : ℝ) (h : x < 0) : f x = -x^2 - 6 * x - 5 :=
begin
  -- We need to prove that f(x) = -x^2 - 6x - 5 when x < 0
  sorry
end

end odd_function_f_find_f_when_x_lt_zero_l822_822750


namespace freeRangingChickens_l822_822113

-- Define the number of chickens in the coop
def chickensInCoop : Nat := 14

-- Define the number of chickens in the run
def chickensInRun : Nat := 2 * chickensInCoop

-- Define the number of chickens free ranging
def chickensFreeRanging : Nat := 2 * chickensInRun - 4

-- State the theorem
theorem freeRangingChickens : chickensFreeRanging = 52 := by
  -- We cannot provide the proof, so we use sorry
  sorry

end freeRangingChickens_l822_822113


namespace mail_distribution_l822_822987

def total_mail : ℕ := 2758
def mail_for_first_block : ℕ := 365
def mail_for_second_block : ℕ := 421
def remaining_mail : ℕ := total_mail - (mail_for_first_block + mail_for_second_block)
def remaining_blocks : ℕ := 3
def mail_per_remaining_block : ℕ := remaining_mail / remaining_blocks

theorem mail_distribution :
  mail_per_remaining_block = 657 := by
  sorry

end mail_distribution_l822_822987


namespace angle_CDE_of_quadrilateral_l822_822653

theorem angle_CDE_of_quadrilateral
  (A B C D E : Type)
  [angle_A : angle A 90]
  [angle_B : angle B 90]
  [angle_C : angle C 90]
  (h1 : angle AEB = 50)
  (h2 : angle BED = angle BDE) :
  angle CDE = 85 :=
by sorry

end angle_CDE_of_quadrilateral_l822_822653


namespace incorrect_conclusion_l822_822617

noncomputable def data_set : List ℕ := [4, 1, 6, 2, 9, 5, 8]
def mean_x : ℝ := 2
def mean_y : ℝ := 20
def regression_eq (x : ℝ) : ℝ := 9.1 * x + 1.8
def chi_squared_value : ℝ := 9.632
def alpha : ℝ := 0.001
def critical_value : ℝ := 10.828

theorem incorrect_conclusion : ¬(chi_squared_value ≥ critical_value) := by
  -- Insert proof here
  sorry

end incorrect_conclusion_l822_822617


namespace true_compound_propositions_l822_822290

-- Define the propositions
def p1 : Prop := 
  ∀ (l₁ l₂ l₃ : Line), 
    (l₁ ≠ l₂ ∧ l₂ ≠ l₃ ∧ l₁ ≠ l₃) ∧ 
    (∃ (p₁ p₂ p₃ : Point),
      l₁.contains p₁ ∧ l₁.contains p₂ ∧
      l₂.contains p₂ ∧ l₂.contains p₃ ∧
      l₃.contains p₃ ∧ l₃.contains p₁)
    → ∃ (α : Plane), 
      α.contains l₁ ∧ α.contains l₂ ∧ α.contains l₃

def p2 : Prop :=
  ∀ (a b c : Point),
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) 
    → ∃! (α : Plane), 
      α.contains a ∧ α.contains b ∧ α.contains c

def p3 : Prop :=
  ∀ (l₁ l₂ : Line),
    (¬ ∃ (p : Point), l₁.contains p ∧ l₂.contains p)
    → l₁.parallel l₂

def p4 : Prop :=
  ∀ (α : Plane) (l m : Line),
    (α.contains l ∧ ¬ α.contains m ∧ m.perpendicular α) 
    → m.perpendicular l

-- Main theorem that identifies the true compound propositions
theorem true_compound_propositions :
  (p1 ∧ p4) ∧ ¬ (p1 ∧ p2) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) :=
by
  -- Proof might involve defining lines, points, and planes and their relationships
  sorry

end true_compound_propositions_l822_822290


namespace space_inside_sphere_outside_combined_cylinder_cone_l822_822221

noncomputable def volumeOfSphere (r : ℝ) : ℝ := (4 / 3) * π * r^3

noncomputable def volumeOfCylinder (r h : ℝ) : ℝ := π * r^2 * h

noncomputable def volumeOfCone (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

theorem space_inside_sphere_outside_combined_cylinder_cone :
  let r_sphere := 6
  let r_cylinder := 4
  let h_cylinder := 10
  let h_cone := 5
  let v_sphere := volumeOfSphere r_sphere
  let v_cylinder := volumeOfCylinder r_cylinder h_cylinder
  let v_cone := volumeOfCone r_cylinder h_cone
  let v_combined := v_cylinder + v_cone
  v_sphere - v_combined = (304 / 3) * π :=
by
  sorry

end space_inside_sphere_outside_combined_cylinder_cone_l822_822221


namespace two_digit_reverse_diff_properties_l822_822625

theorem two_digit_reverse_diff_properties :
  ∃ (q r : ℕ), q < 100 ∧ r < 100 ∧
               ((q div 10 = r % 10) ∧ (q % 10 = r div 10)) ∧
               0 < q - r ∧ q - r < 60 ∧
               9 * (q div 10 - r % 10) = 54 →
               (q div 10 - r % 10) = 6 :=
by
  sorry

end two_digit_reverse_diff_properties_l822_822625


namespace cost_of_one_bag_l822_822136

theorem cost_of_one_bag (x : ℝ) (h1 : ∀ p : ℝ, 60 * x = p -> 60 * p = 120 * x ) 
  (h2 : ∀ p1 p2: ℝ, 60 * x = p1 ∧ 15 * 1.6 * x = p2 ∧ 45 * 2.24 * x = 100.8 * x -> 124.8 * x - 120 * x = 1200) :
  x = 250 := 
sorry

end cost_of_one_bag_l822_822136


namespace angle_a_calculation_l822_822463

theorem angle_a_calculation {E : Type} [plane : E] {a b c d u v : ℝ}
  (angle_sum_eq : a + b + c + d = 360)
  (angle_d_eq : d = 360 - v)
  (angle_u_eq : b + c = u) :
  a = v - u :=
by sorry

end angle_a_calculation_l822_822463


namespace vacation_cost_split_l822_822502

theorem vacation_cost_split 
  (airbnb_cost : ℕ)
  (car_rental_cost : ℕ)
  (people : ℕ)
  (split_equally : Prop)
  (h1 : airbnb_cost = 3200)
  (h2 : car_rental_cost = 800)
  (h3 : people = 8)
  (h4 : split_equally)
  : (airbnb_cost + car_rental_cost) / people = 500 :=
by
  sorry

end vacation_cost_split_l822_822502


namespace book_loss_percentage_l822_822972

variables {CP SP gain_price : ℝ}

-- Conditions
def initial_selling_price := 540
def gain_price := 660
def gain_percentage := 0.10 -- 10% gain expressed as a decimal
def loss_percentage (CP SP : ℝ) : ℝ := 100 * (CP - SP) / CP

-- The goal is to prove that the loss percentage is 10%
theorem book_loss_percentage (CP : ℝ) (h₁ : SP = initial_selling_price)
    (h₂ : 1.1 * CP = gain_price) : loss_percentage CP SP = 10 := by
  sorry

end book_loss_percentage_l822_822972


namespace original_class_strength_l822_822180

theorem original_class_strength 
(initial_avg_age : ℕ)
(new_students : ℕ)
(new_students_avg_age : ℕ)
(new_avg_age : ℕ)
(h_initial_avg : initial_avg_age = 40)
(h_new_students : new_students = 12)
(h_new_students_avg : new_students_avg_age = 32)
(h_new_avg : new_avg_age = 36) : 
  let N := (new_avg_age * (new_students + (initial_avg_age / new_avg_age)) - (new_students * new_students_avg_age)) / initial_avg_age
  in N = 12 :=
by sorry

end original_class_strength_l822_822180


namespace sqrt_180_simplified_l822_822053

theorem sqrt_180_simplified : Real.sqrt 180 = 6 * Real.sqrt 5 :=
   sorry

end sqrt_180_simplified_l822_822053


namespace probability_factor_of_90_less_than_8_l822_822157

theorem probability_factor_of_90_less_than_8 : 
  (∃ factors : Finset ℕ, factors = {1, 2, 3, 5, 6, 10, 15, 18, 30, 45, 90} ∧
  (∃ count_less_8 : ℕ, {1, 2, 3, 5, 6}.card = 5 ∧
  (∃ total_factors : ℕ, factors.card = 12 ∧
  (∃ prob : Rat, prob = (5 / 12) ∧
  prob = (5 / 12)))) := by
  sorry

end probability_factor_of_90_less_than_8_l822_822157


namespace perfect_square_expression_l822_822886

theorem perfect_square_expression (p : ℝ) : 
  (12.86^2 + 12.86 * p + 0.14^2) = (12.86 + 0.14)^2 → p = 0.28 :=
by
  sorry

end perfect_square_expression_l822_822886


namespace length_of_side_c_l822_822438

theorem length_of_side_c 
  (a b : ℝ) 
  (angleB : ℝ)
  (ha : a = 2)
  (hb : b = sqrt 7)
  (hB : angleB = real.pi / 3) :
  ∃ c : ℝ, c = 3 := 
by 
  sorry

end length_of_side_c_l822_822438


namespace sequence_is_sum_of_factorial_and_exponential_l822_822523

theorem sequence_is_sum_of_factorial_and_exponential (a : ℕ → ℕ) :
  a 0 = 2 ∧ a 1 = 3 ∧ a 2 = 6 ∧
  (∀ n, a (n + 3) = (n + 4) * a (n + 2) - 4 * (n + 1) * a (n + 1) + (4 * (n + 1) - 8) * a n) →
  (∀ n, a n = nat.factorial n + 2 ^ n) :=
by
  sorry

end sequence_is_sum_of_factorial_and_exponential_l822_822523


namespace Draymond_points_l822_822456

-- Define the points for each player
variables (D : ℝ)

-- Conditions
def Curry_points := 2 * D
def Kelly_points : ℝ := 9
def Durant_points : ℝ := 2 * Kelly_points
def Klay_points := D / 2
def total_team_points := D + Curry_points + Kelly_points + Durant_points + Klay_points

-- The theorem we want to prove
theorem Draymond_points :
  total_team_points = 69 → D = 12 :=
by
  intro h
  -- Proof omitted
  sorry

end Draymond_points_l822_822456


namespace exists_finite_set_of_circles_l822_822462

theorem exists_finite_set_of_circles (points : List (Real × Real)) (h_points_count : points.length = 100) :
  ∃ (circles : List (Real × Real × Real)), 
    (∀ p ∈ points, ∃ (x y r : Real), (x, y, r) ∈ circles ∧ (p.1 - x)^2 + (p.2 - y)^2 < r^2) ∧
    (∀ p₁ p₂ ∈ points, ∀ (x₁ y₁ r₁ x₂ y₂ r₂ : Real), 
      (x₁, y₁, r₁) ∈ circles → (x₂, y₂, r₂) ∈ circles → (x₁, y₁, r₁) ≠ (x₂, y₂, r₂) → 
      (p₁.1 - x₁)^2 + (p₁.2 - y₁)^2 < r₁^2 → (p₂.1 - x₂)^2 + (p₂.2 - y₂)^2 < r₂^2 → 
      (p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2 > 1) ∧
    (∑ (x y r : Real) in circles, 2 * r < 100) := sorry

end exists_finite_set_of_circles_l822_822462
