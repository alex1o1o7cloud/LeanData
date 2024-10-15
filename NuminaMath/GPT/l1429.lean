import Mathlib

namespace NUMINAMATH_GPT_total_cost_38_pencils_56_pens_l1429_142903

def numberOfPencils : ℕ := 38
def costPerPencil : ℝ := 2.50
def numberOfPens : ℕ := 56
def costPerPen : ℝ := 3.50
def totalCost := numberOfPencils * costPerPencil + numberOfPens * costPerPen

theorem total_cost_38_pencils_56_pens : totalCost = 291 := 
by
  -- leaving the proof as a placeholder
  sorry

end NUMINAMATH_GPT_total_cost_38_pencils_56_pens_l1429_142903


namespace NUMINAMATH_GPT_half_sum_squares_ge_product_l1429_142957

theorem half_sum_squares_ge_product (x y : ℝ) : 
  1 / 2 * (x^2 + y^2) ≥ x * y := 
by 
  sorry

end NUMINAMATH_GPT_half_sum_squares_ge_product_l1429_142957


namespace NUMINAMATH_GPT_count_correct_propositions_l1429_142978

def line_parallel_plane (a : Line) (M : Plane) : Prop := sorry
def line_perpendicular_plane (a : Line) (M : Plane) : Prop := sorry
def line_parallel_line (a b : Line) : Prop := sorry
def line_perpendicular_line (a b : Line) : Prop := sorry
def plane_perpendicular_plane (M N : Plane) : Prop := sorry

theorem count_correct_propositions 
  (a b c : Line) 
  (M N : Plane) 
  (h1 : ¬ (line_parallel_plane a M ∧ line_parallel_plane b M → line_parallel_line a b)) 
  (h2 : line_parallel_plane a M ∧ line_perpendicular_plane b M → line_perpendicular_line b a) 
  (h3 : ¬ ((line_parallel_plane a M ∧ line_perpendicular_plane b M ∧ line_perpendicular_line c a ∧ line_perpendicular_line c b) → line_perpendicular_plane c M))
  (h4 : line_perpendicular_plane a M ∧ line_parallel_plane a N → plane_perpendicular_plane M N) :
  (0 + 1 + 0 + 1) = 2 :=
sorry

end NUMINAMATH_GPT_count_correct_propositions_l1429_142978


namespace NUMINAMATH_GPT_calc_result_l1429_142958

theorem calc_result (a : ℤ) : 3 * a - 5 * a + a = -a := by
  sorry

end NUMINAMATH_GPT_calc_result_l1429_142958


namespace NUMINAMATH_GPT_no_half_probability_socks_l1429_142924

theorem no_half_probability_socks (n m : ℕ) (h_sum : n + m = 2009) : 
  (n - m)^2 ≠ 2009 :=
sorry

end NUMINAMATH_GPT_no_half_probability_socks_l1429_142924


namespace NUMINAMATH_GPT_no_perfect_square_after_swap_l1429_142972

def is_consecutive_digits (a b c d : ℕ) : Prop := 
  (b = a + 1) ∧ (c = b + 1) ∧ (d = c + 1)

def swap_hundreds_tens (n : ℕ) : ℕ := 
  let d4 := n / 1000
  let d3 := (n % 1000) / 100
  let d2 := (n % 100) / 10
  let d1 := n % 10
  d4 * 1000 + d2 * 100 + d3 * 10 + d1

theorem no_perfect_square_after_swap : ¬ ∃ (n : ℕ), 
  1000 ≤ n ∧ n < 10000 ∧ 
  (let d4 := n / 1000
   let d3 := (n % 1000) / 100
   let d2 := (n % 100) / 10
   let d1 := n % 10
   is_consecutive_digits d4 d3 d2 d1) ∧ 
  let new_number := swap_hundreds_tens n
  (∃ m : ℕ, m * m = new_number) := 
sorry

end NUMINAMATH_GPT_no_perfect_square_after_swap_l1429_142972


namespace NUMINAMATH_GPT_regular_octagon_exterior_angle_l1429_142952

theorem regular_octagon_exterior_angle : 
  ∀ (n : ℕ), n = 8 → (180 * (n - 2) / n) + (180 - (180 * (n - 2) / n)) = 180 := by
  sorry

end NUMINAMATH_GPT_regular_octagon_exterior_angle_l1429_142952


namespace NUMINAMATH_GPT_union_example_l1429_142953

theorem union_example (P Q : Set ℕ) (hP : P = {1, 2, 3, 4}) (hQ : Q = {2, 4}) :
  P ∪ Q = {1, 2, 3, 4} :=
by
  sorry

end NUMINAMATH_GPT_union_example_l1429_142953


namespace NUMINAMATH_GPT_value_of_other_bills_is_40_l1429_142918

-- Define the conditions using Lean definitions
def class_fund_contains_only_10_and_other_bills (total_amount : ℕ) (num_other_bills num_10_bills : ℕ) : Prop :=
  total_amount = 120 ∧ num_other_bills = 3 ∧ num_10_bills = 2 * num_other_bills

def value_of_each_other_bill (total_amount num_other_bills : ℕ) : ℕ :=
  total_amount / num_other_bills

-- The theorem we want to prove
theorem value_of_other_bills_is_40 (total_amount num_other_bills : ℕ) 
  (h : class_fund_contains_only_10_and_other_bills total_amount num_other_bills (2 * num_other_bills)) :
  value_of_each_other_bill total_amount num_other_bills = 40 := 
by 
  -- We use the conditions here to ensure they are part of the proof even if we skip the actual proof with sorry
  have h1 : total_amount = 120 := by sorry
  have h2 : num_other_bills = 3 := by sorry
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_value_of_other_bills_is_40_l1429_142918


namespace NUMINAMATH_GPT_floor_sqrt_80_l1429_142975

theorem floor_sqrt_80 : ∀ (x : ℝ), 8 ^ 2 < 80 ∧ 80 < 9 ^ 2 → x = 8 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_floor_sqrt_80_l1429_142975


namespace NUMINAMATH_GPT_gcd_7429_13356_l1429_142935

theorem gcd_7429_13356 : Nat.gcd 7429 13356 = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_7429_13356_l1429_142935


namespace NUMINAMATH_GPT_finite_parabolas_do_not_cover_plane_l1429_142987

theorem finite_parabolas_do_not_cover_plane (parabolas : Finset (ℝ → ℝ)) :
  ¬ (∀ x y : ℝ, ∃ p ∈ parabolas, y < p x) :=
by sorry

end NUMINAMATH_GPT_finite_parabolas_do_not_cover_plane_l1429_142987


namespace NUMINAMATH_GPT_reunion_handshakes_l1429_142951

/-- 
Given 15 boys at a reunion:
- 5 are left-handed and will only shake hands with other left-handed boys.
- Each boy shakes hands exactly once with each of the others unless they forget.
- Three boys each forget to shake hands with two others.

Prove that the total number of handshakes is 49. 
-/
theorem reunion_handshakes : 
  let total_boys := 15
  let left_handed := 5
  let forgetful_boys := 3
  let forgotten_handshakes_per_boy := 2

  let total_handshakes := total_boys * (total_boys - 1) / 2
  let left_left_handshakes := left_handed * (left_handed - 1) / 2
  let left_right_handshakes := left_handed * (total_boys - left_handed)
  let distinct_forgotten_handshakes := forgetful_boys * forgotten_handshakes_per_boy / 2

  total_handshakes 
    - left_right_handshakes 
    - distinct_forgotten_handshakes
    - left_left_handshakes
  = 49 := 
sorry

end NUMINAMATH_GPT_reunion_handshakes_l1429_142951


namespace NUMINAMATH_GPT_rental_cost_equal_mileage_l1429_142933

theorem rental_cost_equal_mileage:
  ∃ x : ℝ, (17.99 + 0.18 * x = 18.95 + 0.16 * x) ∧ x = 48 := 
by
  sorry

end NUMINAMATH_GPT_rental_cost_equal_mileage_l1429_142933


namespace NUMINAMATH_GPT_firstGradeMuffins_l1429_142948

-- Define the conditions as the number of muffins baked by each class
def mrsBrierMuffins : ℕ := 18
def mrsMacAdamsMuffins : ℕ := 20
def mrsFlanneryMuffins : ℕ := 17

-- Define the total number of muffins baked
def totalMuffins : ℕ := mrsBrierMuffins + mrsMacAdamsMuffins + mrsFlanneryMuffins

-- Prove that the total number of muffins baked is 55
theorem firstGradeMuffins : totalMuffins = 55 := by
  sorry

end NUMINAMATH_GPT_firstGradeMuffins_l1429_142948


namespace NUMINAMATH_GPT_int_squares_l1429_142980

theorem int_squares (n : ℕ) (h : ∃ k : ℕ, n^4 - n^3 + 3 * n^2 + 5 = k^2) : n = 2 := by
  sorry

end NUMINAMATH_GPT_int_squares_l1429_142980


namespace NUMINAMATH_GPT_expression_eq_one_if_and_only_if_k_eq_one_l1429_142966

noncomputable def expression (a b c k : ℝ) :=
  (k * a^2 * b^2 + a^2 * c^2 + b^2 * c^2) /
  ((a^2 - b * c) * (b^2 - a * c) + (a^2 - b * c) * (c^2 - a * b) + (b^2 - a * c) * (c^2 - a * b))

theorem expression_eq_one_if_and_only_if_k_eq_one
  (a b c k : ℝ) (h : a + b + c = 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hk : k ≠ 0) :
  expression a b c k = 1 ↔ k = 1 :=
by
  sorry

end NUMINAMATH_GPT_expression_eq_one_if_and_only_if_k_eq_one_l1429_142966


namespace NUMINAMATH_GPT_pow_two_div_factorial_iff_exists_l1429_142993

theorem pow_two_div_factorial_iff_exists (n : ℕ) (hn : n > 0) : 
  (∃ k : ℕ, k > 0 ∧ n = 2^(k-1)) ↔ 2^(n-1) ∣ n! := 
by {
  sorry
}

end NUMINAMATH_GPT_pow_two_div_factorial_iff_exists_l1429_142993


namespace NUMINAMATH_GPT_trigonometric_identity_l1429_142971

noncomputable def trigonometric_identity_proof : Prop :=
  let cos_30 := Real.sqrt 3 / 2;
  let sin_60 := Real.sqrt 3 / 2;
  let sin_30 := 1 / 2;
  let cos_60 := 1 / 2;
  (1 - 1 / cos_30) * (1 + 1 / sin_60) * (1 - 1 / sin_30) * (1 + 1 / cos_60) = 1

theorem trigonometric_identity : trigonometric_identity_proof :=
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1429_142971


namespace NUMINAMATH_GPT_least_possible_value_l1429_142936

theorem least_possible_value (x y : ℝ) : 
  ∃ (x y : ℝ), (xy + 1)^2 + (x + y + 1)^2 = 0 := 
sorry

end NUMINAMATH_GPT_least_possible_value_l1429_142936


namespace NUMINAMATH_GPT_journey_duration_l1429_142944

theorem journey_duration
  (distance : ℕ) (speed : ℕ) (h1 : distance = 48) (h2 : speed = 8) :
  distance / speed = 6 := 
by
  sorry

end NUMINAMATH_GPT_journey_duration_l1429_142944


namespace NUMINAMATH_GPT_final_price_on_monday_l1429_142923

-- Definitions based on the conditions
def saturday_price : ℝ := 50
def sunday_increase : ℝ := 1.2
def monday_discount : ℝ := 0.2

-- The statement to prove
theorem final_price_on_monday : 
  let sunday_price := saturday_price * sunday_increase
  let monday_price := sunday_price * (1 - monday_discount)
  monday_price = 48 :=
by
  sorry

end NUMINAMATH_GPT_final_price_on_monday_l1429_142923


namespace NUMINAMATH_GPT_polynomial_roots_cubed_l1429_142986

noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 5*x - 3
noncomputable def g (x : ℝ) : ℝ := x^3 - 2*x^2 - 5*x + 3

theorem polynomial_roots_cubed {r : ℝ} (h : f r = 0) :
  g (r^3) = 0 := by
  sorry

end NUMINAMATH_GPT_polynomial_roots_cubed_l1429_142986


namespace NUMINAMATH_GPT_ratio_of_55_to_11_l1429_142991

theorem ratio_of_55_to_11 : (55 / 11) = 5 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_55_to_11_l1429_142991


namespace NUMINAMATH_GPT_charlie_paints_140_square_feet_l1429_142981

-- Define the conditions
def total_area : ℕ := 320
def ratio_allen : ℕ := 4
def ratio_ben : ℕ := 5
def ratio_charlie : ℕ := 7
def total_parts : ℕ := ratio_allen + ratio_ben + ratio_charlie
def area_per_part := total_area / total_parts
def charlie_parts := 7

-- Prove the main statement
theorem charlie_paints_140_square_feet : charlie_parts * area_per_part = 140 := by
  sorry

end NUMINAMATH_GPT_charlie_paints_140_square_feet_l1429_142981


namespace NUMINAMATH_GPT_container_capacity_l1429_142904

-- Definitions based on the conditions
def tablespoons_per_cup := 3
def ounces_per_cup := 8
def tablespoons_added := 15

-- Problem statement
theorem container_capacity : 
  (tablespoons_added / tablespoons_per_cup) * ounces_per_cup = 40 :=
  sorry

end NUMINAMATH_GPT_container_capacity_l1429_142904


namespace NUMINAMATH_GPT_carrie_fourth_day_miles_l1429_142970

theorem carrie_fourth_day_miles (d1 d2 d3 d4: ℕ) (charge_interval charges: ℕ) 
  (h1: d1 = 135) 
  (h2: d2 = d1 + 124) 
  (h3: d3 = 159) 
  (h4: charge_interval = 106) 
  (h5: charges = 7):
  d4 = 742 - (d1 + d2 + d3) :=
by
  sorry

end NUMINAMATH_GPT_carrie_fourth_day_miles_l1429_142970


namespace NUMINAMATH_GPT_opposite_of_2023_is_neg2023_l1429_142943

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end NUMINAMATH_GPT_opposite_of_2023_is_neg2023_l1429_142943


namespace NUMINAMATH_GPT_width_of_box_is_correct_l1429_142929

noncomputable def length_of_box : ℝ := 62
noncomputable def height_lowered : ℝ := 0.5
noncomputable def volume_removed_in_gallons : ℝ := 5812.5
noncomputable def gallons_to_cubic_feet : ℝ := 1 / 7.48052

theorem width_of_box_is_correct :
  let volume_removed_in_cubic_feet := volume_removed_in_gallons * gallons_to_cubic_feet
  let area_of_base := length_of_box * W
  let needed_volume := area_of_base * height_lowered
  volume_removed_in_cubic_feet = needed_volume →
  W = 25.057 :=
by
  sorry

end NUMINAMATH_GPT_width_of_box_is_correct_l1429_142929


namespace NUMINAMATH_GPT_divisibility_of_difference_by_9_l1429_142946

theorem divisibility_of_difference_by_9 (a b : ℕ) (ha : 0 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) :
  9 ∣ ((10 * a + b) - (10 * b + a)) :=
by {
  -- The problem statement
  sorry
}

end NUMINAMATH_GPT_divisibility_of_difference_by_9_l1429_142946


namespace NUMINAMATH_GPT_even_function_value_l1429_142960

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 3 * a + b

theorem even_function_value (h_even : ∀ x, f a b x = f a b (-x))
    (h_domain : a - 1 = -2 * a) :
    f a (0 : ℝ) (1 / 2) = 13 / 12 :=
by
  sorry

end NUMINAMATH_GPT_even_function_value_l1429_142960


namespace NUMINAMATH_GPT_exists_equal_subinterval_l1429_142964

open Set Metric Function

variable {a b : ℝ}
variable {f : ℕ → ℝ → ℝ}
variable {n m : ℕ}

-- Define the conditions
def continuous_on_interval (f : ℕ → ℝ → ℝ) (a b : ℝ) :=
  ∀ n, ContinuousOn (f n) (Icc a b)

def root_cond (f : ℕ → ℝ → ℝ) (a b : ℝ) :=
  ∀ x ∈ Icc a b, ∃ m n, m ≠ n ∧ f m x = f n x

-- The main theorem statement
theorem exists_equal_subinterval (f : ℕ → ℝ → ℝ) (a b : ℝ) 
  (h_cont : continuous_on_interval f a b) 
  (h_root : root_cond f a b) : 
  ∃ (c d : ℝ), c < d ∧ Icc c d ⊆ Icc a b ∧ ∃ m n, m ≠ n ∧ ∀ x ∈ Icc c d, f m x = f n x := 
sorry

end NUMINAMATH_GPT_exists_equal_subinterval_l1429_142964


namespace NUMINAMATH_GPT_values_of_a_l1429_142983

noncomputable def M : Set ℝ := {x | x^2 = 1}

noncomputable def N (a : ℝ) : Set ℝ := 
  if a = 0 then ∅ else {x | a * x = 1}

theorem values_of_a (a : ℝ) : (N a ⊆ M) ↔ (a = -1 ∨ a = 0 ∨ a = 1) := by
  sorry

end NUMINAMATH_GPT_values_of_a_l1429_142983


namespace NUMINAMATH_GPT_jail_time_ratio_l1429_142911

def arrests (days : ℕ) (cities : ℕ) (arrests_per_day : ℕ) : ℕ := days * cities * arrests_per_day
def jail_days_before_trial (total_arrests : ℕ) (days_before_trial : ℕ) : ℕ := total_arrests * days_before_trial
def weeks_from_days (days : ℕ) : ℕ := days / 7
def time_after_trial (total_jail_time_weeks : ℕ) (weeks_before_trial : ℕ) : ℕ := total_jail_time_weeks - weeks_before_trial
def total_possible_jail_time (total_arrests : ℕ) (sentence_weeks : ℕ) : ℕ := total_arrests * sentence_weeks
def ratio (after_trial_weeks : ℕ) (total_possible_weeks : ℕ) : ℚ := after_trial_weeks / total_possible_weeks

theorem jail_time_ratio 
    (days : ℕ := 30) 
    (cities : ℕ := 21)
    (arrests_per_day : ℕ := 10)
    (days_before_trial : ℕ := 4)
    (total_jail_time_weeks : ℕ := 9900)
    (sentence_weeks : ℕ := 2) :
    ratio 
      (time_after_trial 
        total_jail_time_weeks 
        (weeks_from_days 
          (jail_days_before_trial 
            (arrests days cities arrests_per_day) 
            days_before_trial))) 
      (total_possible_jail_time 
        (arrests days cities arrests_per_day) 
        sentence_weeks) = 1/2 := 
by
  -- We leave the proof as an exercise
  sorry

end NUMINAMATH_GPT_jail_time_ratio_l1429_142911


namespace NUMINAMATH_GPT_area_of_border_l1429_142942

theorem area_of_border
  (h_photo : Nat := 9)
  (w_photo : Nat := 12)
  (border_width : Nat := 3) :
  (let area_photo := h_photo * w_photo
    let h_frame := h_photo + 2 * border_width
    let w_frame := w_photo + 2 * border_width
    let area_frame := h_frame * w_frame
    let area_border := area_frame - area_photo
    area_border = 162) := 
  sorry

end NUMINAMATH_GPT_area_of_border_l1429_142942


namespace NUMINAMATH_GPT_complex_in_second_quadrant_l1429_142990

-- Define the complex number z based on the problem conditions.
def z : ℂ := Complex.I + (Complex.I^6)

-- State the condition to check whether z is in the second quadrant.
def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- Formulate the theorem stating that the complex number z is in the second quadrant.
theorem complex_in_second_quadrant : is_in_second_quadrant z :=
by
  sorry

end NUMINAMATH_GPT_complex_in_second_quadrant_l1429_142990


namespace NUMINAMATH_GPT_min_value_xy_l1429_142912

theorem min_value_xy (x y : ℝ) (h : x * y = 1) : x^2 + 4 * y^2 ≥ 4 := by
  sorry

end NUMINAMATH_GPT_min_value_xy_l1429_142912


namespace NUMINAMATH_GPT_couscous_problem_l1429_142937

def total_couscous (S1 S2 S3 : ℕ) : ℕ :=
  S1 + S2 + S3

def couscous_per_dish (total : ℕ) (dishes : ℕ) : ℕ :=
  total / dishes

theorem couscous_problem 
  (S1 S2 S3 : ℕ) (dishes : ℕ) 
  (h1 : S1 = 7) (h2 : S2 = 13) (h3 : S3 = 45) (h4 : dishes = 13) :
  couscous_per_dish (total_couscous S1 S2 S3) dishes = 5 := by  
  sorry

end NUMINAMATH_GPT_couscous_problem_l1429_142937


namespace NUMINAMATH_GPT_sandy_red_marbles_l1429_142921

theorem sandy_red_marbles (jessica_marbles : ℕ) (sandy_marbles : ℕ) 
  (h₀ : jessica_marbles = 3 * 12)
  (h₁ : sandy_marbles = 4 * jessica_marbles) : 
  sandy_marbles = 144 :=
by
  sorry

end NUMINAMATH_GPT_sandy_red_marbles_l1429_142921


namespace NUMINAMATH_GPT_opposite_of_three_l1429_142982

theorem opposite_of_three :
  ∃ x : ℤ, 3 + x = 0 ∧ x = -3 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_three_l1429_142982


namespace NUMINAMATH_GPT_overlapping_segments_length_l1429_142962

theorem overlapping_segments_length 
    (total_length : ℝ) 
    (actual_distance : ℝ) 
    (num_overlaps : ℕ) 
    (h1 : total_length = 98) 
    (h2 : actual_distance = 83)
    (h3 : num_overlaps = 6) :
    (total_length - actual_distance) / num_overlaps = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_overlapping_segments_length_l1429_142962


namespace NUMINAMATH_GPT_number_of_solutions_l1429_142999

-- Define the equation
def equation (x : ℝ) : Prop := (3 * x^2 - 15 * x) / (x^2 - 7 * x + 10) = x - 4

-- State the problem with conditions and conclusion
theorem number_of_solutions : (∀ x : ℝ, x ≠ 2 ∧ x ≠ 5 → equation x) ↔ (∃ x1 x2 : ℝ, x1 ≠ 2 ∧ x1 ≠ 5 ∧ x2 ≠ 2 ∧ x2 ≠ 5 ∧ equation x1 ∧ equation x2) :=
by
  sorry

end NUMINAMATH_GPT_number_of_solutions_l1429_142999


namespace NUMINAMATH_GPT_number_of_members_l1429_142902

def cost_knee_pads : ℤ := 6
def cost_jersey : ℤ := cost_knee_pads + 7
def total_cost_per_member : ℤ := 2 * (cost_knee_pads + cost_jersey)
def total_expenditure : ℤ := 3120

theorem number_of_members (n : ℤ) (h : n * total_cost_per_member = total_expenditure) : n = 82 :=
sorry

end NUMINAMATH_GPT_number_of_members_l1429_142902


namespace NUMINAMATH_GPT_perpendicular_transfer_l1429_142949

variables {Line Plane : Type} 
variables (a b : Line) (α β : Plane)

def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (l : Line) (p : Plane) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry

theorem perpendicular_transfer
  (h1 : perpendicular a α)
  (h2 : parallel_planes α β) :
  perpendicular a β := 
sorry

end NUMINAMATH_GPT_perpendicular_transfer_l1429_142949


namespace NUMINAMATH_GPT_domain_of_f_x_plus_2_l1429_142909

theorem domain_of_f_x_plus_2 (f : ℝ → ℝ) (dom_f_x_minus_1 : ∀ x, 1 ≤ x ∧ x ≤ 2 → 0 ≤ x-1 ∧ x-1 ≤ 1) :
  ∀ y, 0 ≤ y ∧ y ≤ 1 ↔ -2 ≤ y-2 ∧ y-2 ≤ -1 :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_x_plus_2_l1429_142909


namespace NUMINAMATH_GPT_union_of_M_N_l1429_142996

-- Define the sets M and N
def M : Set ℕ := {0, 2, 3}
def N : Set ℕ := {1, 3}

-- State the theorem to prove that M ∪ N = {0, 1, 2, 3}
theorem union_of_M_N : M ∪ N = {0, 1, 2, 3} :=
by
  sorry -- Proof goes here

end NUMINAMATH_GPT_union_of_M_N_l1429_142996


namespace NUMINAMATH_GPT_sum_of_terms_l1429_142915

-- Defining the arithmetic progression
def arithmetic_progression (a d : ℕ) (n : ℕ) : ℕ :=
  a + (n - 1) * d

-- Given conditions
theorem sum_of_terms (a d : ℕ) (h : (a + 3 * d) + (a + 11 * d) = 20) :
  12 * (a + 11 * d) / 2 = 60 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_terms_l1429_142915


namespace NUMINAMATH_GPT_arithmetic_operations_correct_l1429_142976

theorem arithmetic_operations_correct :
  (3 + (3 / 3) = (77 / 7) - 7) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_operations_correct_l1429_142976


namespace NUMINAMATH_GPT_gcd_pow_diff_l1429_142906

theorem gcd_pow_diff :
  gcd (2 ^ 2100 - 1) (2 ^ 2091 - 1) = 511 := 
sorry

end NUMINAMATH_GPT_gcd_pow_diff_l1429_142906


namespace NUMINAMATH_GPT_michael_has_16_blocks_l1429_142917

-- Define the conditions
def number_of_boxes : ℕ := 8
def blocks_per_box : ℕ := 2

-- Define the expected total number of blocks
def total_blocks : ℕ := 16

-- State the theorem
theorem michael_has_16_blocks (n_boxes blocks_per_b : ℕ) :
  n_boxes = number_of_boxes → 
  blocks_per_b = blocks_per_box → 
  n_boxes * blocks_per_b = total_blocks :=
by intros h1 h2; rw [h1, h2]; sorry

end NUMINAMATH_GPT_michael_has_16_blocks_l1429_142917


namespace NUMINAMATH_GPT_simplify_expression_l1429_142914

theorem simplify_expression (x : ℝ) : (2 * x)^5 - (3 * x^2 * x^3) = 29 * x^5 := 
  sorry

end NUMINAMATH_GPT_simplify_expression_l1429_142914


namespace NUMINAMATH_GPT_derivative_at_pi_l1429_142932

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (x^2)

theorem derivative_at_pi :
  deriv f π = -1 / (π^2) :=
sorry

end NUMINAMATH_GPT_derivative_at_pi_l1429_142932


namespace NUMINAMATH_GPT_money_left_after_shopping_l1429_142947

def initial_budget : ℝ := 999.00
def shoes_price : ℝ := 165.00
def yoga_mat_price : ℝ := 85.00
def sports_watch_price : ℝ := 215.00
def hand_weights_price : ℝ := 60.00
def sales_tax_rate : ℝ := 0.07
def discount_rate : ℝ := 0.10

def total_cost_before_discount : ℝ :=
  shoes_price + yoga_mat_price + sports_watch_price + hand_weights_price

def discount_on_watch : ℝ := sports_watch_price * discount_rate

def discounted_watch_price : ℝ := sports_watch_price - discount_on_watch

def total_cost_after_discount : ℝ :=
  shoes_price + yoga_mat_price + discounted_watch_price + hand_weights_price

def sales_tax : ℝ := total_cost_after_discount * sales_tax_rate

def total_cost_including_tax : ℝ := total_cost_after_discount + sales_tax

def money_left : ℝ := initial_budget - total_cost_including_tax

theorem money_left_after_shopping : 
  money_left = 460.25 :=
by
  sorry

end NUMINAMATH_GPT_money_left_after_shopping_l1429_142947


namespace NUMINAMATH_GPT_sin_half_angle_l1429_142928

theorem sin_half_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end NUMINAMATH_GPT_sin_half_angle_l1429_142928


namespace NUMINAMATH_GPT_scientific_notation_to_standard_form_l1429_142926

theorem scientific_notation_to_standard_form :
  - 3.96 * 10^5 = -396000 :=
sorry

end NUMINAMATH_GPT_scientific_notation_to_standard_form_l1429_142926


namespace NUMINAMATH_GPT_parabola_focus_eq_l1429_142969

theorem parabola_focus_eq (focus : ℝ × ℝ) (hfocus : focus = (0, 1)) :
  ∃ (p : ℝ), p = 1 ∧ ∀ (x y : ℝ), x^2 = 4 * p * y → x^2 = 4 * y :=
by { sorry }

end NUMINAMATH_GPT_parabola_focus_eq_l1429_142969


namespace NUMINAMATH_GPT_shopkeeper_percentage_profit_l1429_142955

variable {x : ℝ} -- cost price per kg of apples

theorem shopkeeper_percentage_profit 
  (total_weight : ℝ)
  (first_half_sold_at : ℝ)
  (second_half_sold_at : ℝ)
  (first_half_profit : ℝ)
  (second_half_profit : ℝ)
  (total_cost_price : ℝ)
  (total_selling_price : ℝ)
  (total_profit : ℝ)
  (percentage_profit : ℝ) :
  total_weight = 100 →
  first_half_sold_at = 0.5 * total_weight →
  second_half_sold_at = 0.5 * total_weight →
  first_half_profit = 25 →
  second_half_profit = 30 →
  total_cost_price = x * total_weight →
  total_selling_price = (first_half_sold_at * (1 + first_half_profit / 100) * x) + (second_half_sold_at * (1 + second_half_profit / 100) * x) →
  total_profit = total_selling_price - total_cost_price →
  percentage_profit = (total_profit / total_cost_price) * 100 →
  percentage_profit = 27.5 := by
  sorry

end NUMINAMATH_GPT_shopkeeper_percentage_profit_l1429_142955


namespace NUMINAMATH_GPT_area_of_inscribed_triangle_l1429_142995

noncomputable def calculate_triangle_area_inscribed_in_circle 
  (arc1 : ℝ) (arc2 : ℝ) (arc3 : ℝ) (total_circumference := arc1 + arc2 + arc3)
  (radius := total_circumference / (2 * Real.pi))
  (theta := (2 * Real.pi) / total_circumference)
  (angle1 := 5 * theta) (angle2 := 7 * theta) (angle3 := 8 * theta) : ℝ :=
  0.5 * (radius ^ 2) * (Real.sin angle1 + Real.sin angle2 + Real.sin angle3)

theorem area_of_inscribed_triangle : 
  calculate_triangle_area_inscribed_in_circle 5 7 8 = 119.85 / (Real.pi ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_area_of_inscribed_triangle_l1429_142995


namespace NUMINAMATH_GPT_type_b_quantity_l1429_142930

theorem type_b_quantity 
  (x : ℕ)
  (hx : x + 2 * x + 4 * x = 140) : 
  2 * x = 40 := 
sorry

end NUMINAMATH_GPT_type_b_quantity_l1429_142930


namespace NUMINAMATH_GPT_min_re_z4_re_z4_l1429_142945

theorem min_re_z4_re_z4 (z : ℂ) (h : z.re ≠ 0) : 
  ∃ t : ℝ, (t = (z.im / z.re)) ∧ ((1 - 6 * (t^2) + (t^4)) = -8) := sorry

end NUMINAMATH_GPT_min_re_z4_re_z4_l1429_142945


namespace NUMINAMATH_GPT_man_and_son_work_together_l1429_142988

theorem man_and_son_work_together (man_days son_days : ℕ) (h_man : man_days = 15) (h_son : son_days = 10) :
  (1 / (1 / man_days + 1 / son_days) = 6) :=
by
  rw [h_man, h_son]
  sorry

end NUMINAMATH_GPT_man_and_son_work_together_l1429_142988


namespace NUMINAMATH_GPT_calculate_length_of_train_l1429_142963

noncomputable def length_of_train (speed_train_kmh : ℕ) (speed_man_kmh : ℕ) (time_seconds : ℝ) : ℝ :=
  let relative_speed_kmh := speed_train_kmh + speed_man_kmh
  let relative_speed_ms := (relative_speed_kmh : ℝ) * 1000 / 3600
  relative_speed_ms * time_seconds

theorem calculate_length_of_train :
  length_of_train 50 5 7.2 = 110 := by
  -- This is where the actual proof would go, but it's omitted for now as per instructions.
  sorry

end NUMINAMATH_GPT_calculate_length_of_train_l1429_142963


namespace NUMINAMATH_GPT_simplify_expression_l1429_142989

theorem simplify_expression :
  (-2 : ℝ) ^ 2005 + (-2) ^ 2006 + (3 : ℝ) ^ 2007 - (2 : ℝ) ^ 2008 =
  -7 * (2 : ℝ) ^ 2005 + (3 : ℝ) ^ 2007 := 
by
    sorry

end NUMINAMATH_GPT_simplify_expression_l1429_142989


namespace NUMINAMATH_GPT_division_of_decimals_l1429_142900

theorem division_of_decimals : 0.36 / 0.004 = 90 := by
  sorry

end NUMINAMATH_GPT_division_of_decimals_l1429_142900


namespace NUMINAMATH_GPT_ellipse_condition_l1429_142934

theorem ellipse_condition (k : ℝ) :
  (4 < k ∧ k < 9) ↔ (9 - k > 0 ∧ k - 4 > 0 ∧ 9 - k ≠ k - 4) :=
by sorry

end NUMINAMATH_GPT_ellipse_condition_l1429_142934


namespace NUMINAMATH_GPT_abs_inequality_solution_l1429_142974

theorem abs_inequality_solution (x : ℝ) :
  abs (2 * x - 5) ≤ 7 ↔ -1 ≤ x ∧ x ≤ 6 :=
sorry

end NUMINAMATH_GPT_abs_inequality_solution_l1429_142974


namespace NUMINAMATH_GPT_ellipse_focus_distance_l1429_142916

theorem ellipse_focus_distance (m : ℝ) (a b c : ℝ)
  (ellipse_eq : ∀ x y : ℝ, x^2 / m + y^2 / 16 = 1)
  (focus_distance : ∀ P : ℝ × ℝ, ∃ F1 F2 : ℝ × ℝ, dist P F1 = 3 ∧ dist P F2 = 7) :
  m = 25 := 
  sorry

end NUMINAMATH_GPT_ellipse_focus_distance_l1429_142916


namespace NUMINAMATH_GPT_tens_digit_of_9_to_2023_l1429_142910

theorem tens_digit_of_9_to_2023 :
  (9^2023 % 100) / 10 % 10 = 8 :=
sorry

end NUMINAMATH_GPT_tens_digit_of_9_to_2023_l1429_142910


namespace NUMINAMATH_GPT_difference_is_cube_sum_1996_impossible_l1429_142959

theorem difference_is_cube (n : ℕ) (M m : ℕ) 
  (M_eq : M = (3 * n^3 - 4 * n^2 + 5 * n - 2) / 2)
  (m_eq : m = (n^3 + 2 * n^2 - n) / 2) :
  M - m = (n - 1)^3 := 
by {
  sorry
}

theorem sum_1996_impossible (n : ℕ) (M m : ℕ) 
  (M_eq : M = (3 * n^3 - 4 * n^2 + 5 * n - 2) / 2)
  (m_eq : m = (n^3 + 2 * n^2 - n) / 2) :
  ¬(1996 ∈ {x | m ≤ x ∧ x ≤ M}) := 
by {
  sorry
}

end NUMINAMATH_GPT_difference_is_cube_sum_1996_impossible_l1429_142959


namespace NUMINAMATH_GPT_min_value_l1429_142965

open Real

-- Definitions
variables (a b : ℝ)
axiom a_gt_zero : a > 0
axiom b_gt_one : b > 1
axiom sum_eq : a + b = 3 / 2

-- The theorem to be proved.
theorem min_value (a : ℝ) (b : ℝ) (a_gt_zero : a > 0) (b_gt_one : b > 1) (sum_eq : a + b = 3 / 2) :
  ∃ (m : ℝ), m = 6 + 4 * sqrt 2 ∧ ∀ (x y : ℝ), (x > 0) → (y > 1) → (x + y = 3 / 2) → (∃ (z : ℝ), z = 2 / x + 1 / (y - 1) ∧ z ≥ m) :=
sorry

end NUMINAMATH_GPT_min_value_l1429_142965


namespace NUMINAMATH_GPT_positive_root_of_quadratic_eqn_l1429_142938

theorem positive_root_of_quadratic_eqn 
  (b : ℝ)
  (h1 : ∃ x0 : ℝ, x0^2 - 4 * x0 + b = 0 ∧ (-x0)^2 + 4 * (-x0) - b = 0) 
  : ∃ x : ℝ, (x^2 + b * x - 4 = 0) ∧ x = 2 := 
by
  sorry

end NUMINAMATH_GPT_positive_root_of_quadratic_eqn_l1429_142938


namespace NUMINAMATH_GPT_regression_total_sum_of_squares_l1429_142979

variables (y : Fin 10 → ℝ) (y_hat : Fin 10 → ℝ)
variables (residual_sum_of_squares : ℝ) 

-- Given conditions
def R_squared := 0.95
def RSS := 120.53

-- The total sum of squares is what we need to prove
noncomputable def total_sum_of_squares := 2410.6

-- Statement to prove
theorem regression_total_sum_of_squares :
  1 - RSS / total_sum_of_squares = R_squared := by
sorry

end NUMINAMATH_GPT_regression_total_sum_of_squares_l1429_142979


namespace NUMINAMATH_GPT_negation_of_proposition_l1429_142998

noncomputable def original_proposition :=
  ∀ a b : ℝ, (a * b = 0) → (a = 0)

theorem negation_of_proposition :
  ¬ original_proposition ↔ ∃ a b : ℝ, (a * b = 0) ∧ (a ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1429_142998


namespace NUMINAMATH_GPT_sin_half_angle_l1429_142950

theorem sin_half_angle 
  (θ : ℝ) 
  (h_cos : |Real.cos θ| = 1 / 5) 
  (h_theta : 5 * Real.pi / 2 < θ ∧ θ < 3 * Real.pi)
  : Real.sin (θ / 2) = - (Real.sqrt 15) / 5 := 
by
  sorry

end NUMINAMATH_GPT_sin_half_angle_l1429_142950


namespace NUMINAMATH_GPT_sqrt_9_minus_1_eq_2_l1429_142956

theorem sqrt_9_minus_1_eq_2 : Real.sqrt 9 - 1 = 2 := by
  sorry

end NUMINAMATH_GPT_sqrt_9_minus_1_eq_2_l1429_142956


namespace NUMINAMATH_GPT_ordered_pair_solution_l1429_142968

theorem ordered_pair_solution :
  ∃ x y : ℚ, 7 * x - 50 * y = 3 ∧ 3 * y - x = 5 ∧ x = -259 / 29 ∧ y = -38 / 29 :=
by sorry

end NUMINAMATH_GPT_ordered_pair_solution_l1429_142968


namespace NUMINAMATH_GPT_option_C_true_l1429_142992

theorem option_C_true (a b : ℝ):
    (a^2 + b^2 ≥ 2 * a * b) ↔ ((a^2 + b^2 > 2 * a * b) ∨ (a^2 + b^2 = 2 * a * b)) :=
by
  sorry

end NUMINAMATH_GPT_option_C_true_l1429_142992


namespace NUMINAMATH_GPT_rectangle_area_problem_l1429_142940

/--
Given a rectangle with dimensions \(3x - 4\) and \(4x + 6\),
show that the area of the rectangle equals \(12x^2 + 2x - 24\) if and only if \(x \in \left(\frac{4}{3}, \infty\right)\).
-/
theorem rectangle_area_problem 
  (x : ℝ) 
  (h1 : 3 * x - 4 > 0)
  (h2 : 4 * x + 6 > 0) :
  (3 * x - 4) * (4 * x + 6) = 12 * x^2 + 2 * x - 24 ↔ x > 4 / 3 :=
sorry

end NUMINAMATH_GPT_rectangle_area_problem_l1429_142940


namespace NUMINAMATH_GPT_john_unanswered_questions_l1429_142994

theorem john_unanswered_questions (c w u : ℕ) 
  (h1 : 25 + 5 * c - 2 * w = 95) 
  (h2 : 6 * c - w + 3 * u = 105) 
  (h3 : c + w + u = 30) : 
  u = 2 := 
sorry

end NUMINAMATH_GPT_john_unanswered_questions_l1429_142994


namespace NUMINAMATH_GPT_john_total_distance_l1429_142984

-- Define the given conditions
def initial_speed : ℝ := 45 -- mph
def first_leg_time : ℝ := 2 -- hours
def second_leg_time : ℝ := 3 -- hours
def distance_before_lunch : ℝ := initial_speed * first_leg_time
def distance_after_lunch : ℝ := initial_speed * second_leg_time

-- Define the total distance
def total_distance : ℝ := distance_before_lunch + distance_after_lunch

-- Prove the total distance is 225 miles
theorem john_total_distance : total_distance = 225 := by
  sorry

end NUMINAMATH_GPT_john_total_distance_l1429_142984


namespace NUMINAMATH_GPT_min_value_expression_l1429_142901

theorem min_value_expression (y : ℝ) (hy : y > 0) : 9 * y + 1 / y^6 ≥ 10 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1429_142901


namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_l1429_142939

noncomputable def x : ℚ := 0.6 + 41 / 990  

theorem repeating_decimal_to_fraction (h : x = 0.6 + 41 / 990) : x = 127 / 198 :=
by sorry

end NUMINAMATH_GPT_repeating_decimal_to_fraction_l1429_142939


namespace NUMINAMATH_GPT_machine_does_not_print_13824_l1429_142973

-- Definitions corresponding to the conditions:
def machine_property (S : Set ℕ) : Prop :=
  ∀ n ∈ S, (2 * n) ∉ S ∧ (3 * n) ∉ S

def machine_prints_2 (S : Set ℕ) : Prop :=
  2 ∈ S

-- Statement to be proved
theorem machine_does_not_print_13824 (S : Set ℕ) 
  (H1 : machine_property S) 
  (H2 : machine_prints_2 S) : 
  13824 ∉ S :=
sorry

end NUMINAMATH_GPT_machine_does_not_print_13824_l1429_142973


namespace NUMINAMATH_GPT_square_floor_tile_count_l1429_142931

theorem square_floor_tile_count (n : ℕ) (h : 2 * n - 1 = 49) : n^2 = 625 := by
  sorry

end NUMINAMATH_GPT_square_floor_tile_count_l1429_142931


namespace NUMINAMATH_GPT_rectangle_area_l1429_142961

theorem rectangle_area
  (width : ℕ) (length : ℕ)
  (h1 : width = 7)
  (h2 : length = 4 * width) :
  length * width = 196 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1429_142961


namespace NUMINAMATH_GPT_find_abc_l1429_142985

noncomputable def a_b_c_exist : Prop :=
  ∃ (a b c : ℝ), 
    (a + b + c = 21/4) ∧ 
    (1/a + 1/b + 1/c = 21/4) ∧ 
    (a * b * c = 1) ∧ 
    (a < b) ∧ (b < c) ∧ 
    (a = 1/4) ∧ (b = 1) ∧ (c = 4)

theorem find_abc : a_b_c_exist :=
sorry

end NUMINAMATH_GPT_find_abc_l1429_142985


namespace NUMINAMATH_GPT_inequality_proof_l1429_142927

theorem inequality_proof 
  (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (h1 : a ≤ 2 * b) 
  (h2 : 2 * b ≤ 4 * a) :
  4 * a * b ≤ 2 * (a ^ 2 + b ^ 2) ∧ 2 * (a ^ 2 + b ^ 2) ≤ 5 * a * b := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1429_142927


namespace NUMINAMATH_GPT_rectangle_area_l1429_142919

theorem rectangle_area (c h x : ℝ) (h_pos : 0 < h) (c_pos : 0 < c) : 
  (A : ℝ) = (x * (c * x / h)) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1429_142919


namespace NUMINAMATH_GPT_total_flower_petals_l1429_142954

def num_lilies := 8
def petals_per_lily := 6
def num_tulips := 5
def petals_per_tulip := 3

theorem total_flower_petals :
  (num_lilies * petals_per_lily) + (num_tulips * petals_per_tulip) = 63 :=
by
  sorry

end NUMINAMATH_GPT_total_flower_petals_l1429_142954


namespace NUMINAMATH_GPT_range_of_a_l1429_142967

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x^3 - 3 * a^2 * x + 1 ≠ 3)) 
  → (-1 < a ∧ a < 1) := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1429_142967


namespace NUMINAMATH_GPT_pq_iff_cond_l1429_142913

def p (a : ℝ) := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0
def q (a : ℝ) := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem pq_iff_cond (a : ℝ) : (p a ∧ q a) ↔ (a ≤ -2 ∨ a = 1) := 
by
  sorry

end NUMINAMATH_GPT_pq_iff_cond_l1429_142913


namespace NUMINAMATH_GPT_calculate_two_squared_l1429_142997

theorem calculate_two_squared : 2^2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_calculate_two_squared_l1429_142997


namespace NUMINAMATH_GPT_fraction_meaningful_l1429_142977

theorem fraction_meaningful (x : ℝ) : (x ≠ 2) ↔ (x - 2 ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_l1429_142977


namespace NUMINAMATH_GPT_problem_solution_l1429_142941

theorem problem_solution : ∃ n : ℕ, (n > 0) ∧ (21 - 3 * n > 15) ∧ (∀ m : ℕ, (m > 0) ∧ (21 - 3 * m > 15) → m = n) :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1429_142941


namespace NUMINAMATH_GPT_count_even_numbers_is_320_l1429_142908

noncomputable def count_even_numbers_with_distinct_digits : Nat := 
  let unit_choices := 5  -- Choices for the unit digit (0, 2, 4, 6, 8)
  let hundreds_choices := 8  -- Choices for the hundreds digit (1 to 9, excluding the unit digit)
  let tens_choices := 8  -- Choices for the tens digit (0 to 9, excluding the hundreds and unit digit)
  unit_choices * hundreds_choices * tens_choices

theorem count_even_numbers_is_320 : count_even_numbers_with_distinct_digits = 320 := by
  sorry

end NUMINAMATH_GPT_count_even_numbers_is_320_l1429_142908


namespace NUMINAMATH_GPT_common_terms_count_l1429_142922

theorem common_terms_count (β : ℕ) (h1 : β = 55) (h2 : β + 1 = 56) : 
  ∃ γ : ℕ, γ = 6 :=
by
  sorry

end NUMINAMATH_GPT_common_terms_count_l1429_142922


namespace NUMINAMATH_GPT_whole_numbers_between_sqrt_18_and_sqrt_98_l1429_142907

theorem whole_numbers_between_sqrt_18_and_sqrt_98 :
  let lower_bound := Real.sqrt 18
  let upper_bound := Real.sqrt 98
  let smallest_whole_num := 5
  let largest_whole_num := 9
  (largest_whole_num - smallest_whole_num + 1) = 5 :=
by
  -- Introduce variables
  let lower_bound := Real.sqrt 18
  let upper_bound := Real.sqrt 98
  let smallest_whole_num := 5
  let largest_whole_num := 9
  -- Sorry indicates the proof steps are skipped
  sorry

end NUMINAMATH_GPT_whole_numbers_between_sqrt_18_and_sqrt_98_l1429_142907


namespace NUMINAMATH_GPT_orthocenter_circumradii_equal_l1429_142905

-- Define a triangle with its orthocenter and circumradius
variables {A B C H : Point} (R r : ℝ)

-- Assume H is the orthocenter of triangle ABC
def is_orthocenter (H : Point) (A B C : Point) : Prop := 
  sorry -- This should state the definition or properties of an orthocenter

-- Assume the circumradius of triangle ABC is R 
def is_circumradius_ABC (A B C : Point) (R : ℝ) : Prop :=
  sorry -- This should capture the circumradius property

-- Assume circumradius of triangle BHC is r
def is_circumradius_BHC (B H C : Point) (r : ℝ) : Prop :=
  sorry -- This should capture the circumradius property
  
-- Prove that if H is the orthocenter of triangle ABC, the circumradius of ABC is R 
-- and the circumradius of BHC is r, then R = r
theorem orthocenter_circumradii_equal (h_orthocenter : is_orthocenter H A B C) 
  (h_circumradius_ABC : is_circumradius_ABC A B C R)
  (h_circumradius_BHC : is_circumradius_BHC B H C r) : R = r :=
  sorry

end NUMINAMATH_GPT_orthocenter_circumradii_equal_l1429_142905


namespace NUMINAMATH_GPT_Jason_more_blue_marbles_l1429_142925

theorem Jason_more_blue_marbles (Jason_blue_marbles Tom_blue_marbles : ℕ) 
  (hJ : Jason_blue_marbles = 44) (hT : Tom_blue_marbles = 24) :
  Jason_blue_marbles - Tom_blue_marbles = 20 :=
by
  sorry

end NUMINAMATH_GPT_Jason_more_blue_marbles_l1429_142925


namespace NUMINAMATH_GPT_irreducible_fraction_l1429_142920

-- Statement of the theorem
theorem irreducible_fraction (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 :=
by
  sorry -- Proof would be placed here

end NUMINAMATH_GPT_irreducible_fraction_l1429_142920
