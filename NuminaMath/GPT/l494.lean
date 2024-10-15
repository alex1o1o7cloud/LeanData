import Mathlib

namespace NUMINAMATH_GPT_redistribution_not_always_possible_l494_49479

theorem redistribution_not_always_possible (a b : ℕ) (h : a ≠ b) :
  ¬(∃ k : ℕ, a - k = b + k ∧ 0 ≤ k ∧ k ≤ a ∧ k ≤ b) ↔ (a + b) % 2 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_redistribution_not_always_possible_l494_49479


namespace NUMINAMATH_GPT_tangents_arithmetic_sequence_implies_sines_double_angles_arithmetic_sequence_l494_49449

open Real

theorem tangents_arithmetic_sequence_implies_sines_double_angles_arithmetic_sequence (α β γ : ℝ) 
  (h1 : α + β + γ = π)  -- Assuming α, β, γ are angles in a triangle
  (h2 : tan α + tan γ = 2 * tan β) :
  sin (2 * α) + sin (2 * γ) = 2 * sin (2 * β) :=
by
  sorry

end NUMINAMATH_GPT_tangents_arithmetic_sequence_implies_sines_double_angles_arithmetic_sequence_l494_49449


namespace NUMINAMATH_GPT_math_problem_proof_l494_49445

variable {a : ℝ} (ha : a > 0)

theorem math_problem_proof : ((36 * a^9)^4 * (63 * a^9)^4 = a^(72)) :=
by sorry

end NUMINAMATH_GPT_math_problem_proof_l494_49445


namespace NUMINAMATH_GPT_dividend_ratio_l494_49403

theorem dividend_ratio
  (expected_earnings_per_share : ℝ)
  (actual_earnings_per_share : ℝ)
  (dividend_per_share_increase : ℝ)
  (threshold_earnings_increase : ℝ)
  (shares_owned : ℕ)
  (h_expected_earnings : expected_earnings_per_share = 0.8)
  (h_actual_earnings : actual_earnings_per_share = 1.1)
  (h_dividend_increase : dividend_per_share_increase = 0.04)
  (h_threshold_increase : threshold_earnings_increase = 0.1)
  (h_shares_owned : shares_owned = 100)
  : (shares_owned * (expected_earnings_per_share + 
      (actual_earnings_per_share - expected_earnings_per_share) / threshold_earnings_increase * dividend_per_share_increase)) /
    (shares_owned * actual_earnings_per_share) = 46 / 55 :=
by
  sorry

end NUMINAMATH_GPT_dividend_ratio_l494_49403


namespace NUMINAMATH_GPT_scientific_notation_of_4370000_l494_49416

theorem scientific_notation_of_4370000 :
  4370000 = 4.37 * 10^6 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_4370000_l494_49416


namespace NUMINAMATH_GPT_actual_books_bought_l494_49488

def initial_spending : ℕ := 180
def planned_books (x : ℕ) : Prop := initial_spending / x - initial_spending / (5 * x / 4) = 9

theorem actual_books_bought (x : ℕ) (hx : planned_books x) : (5 * x / 4) = 5 :=
by
  sorry

end NUMINAMATH_GPT_actual_books_bought_l494_49488


namespace NUMINAMATH_GPT_pieces_eaten_first_night_l494_49476

def initial_candy_debby : ℕ := 32
def initial_candy_sister : ℕ := 42
def candy_after_first_night : ℕ := 39

theorem pieces_eaten_first_night :
  (initial_candy_debby + initial_candy_sister) - candy_after_first_night = 35 := by
  sorry

end NUMINAMATH_GPT_pieces_eaten_first_night_l494_49476


namespace NUMINAMATH_GPT_determine_ratio_l494_49434

def p (x : ℝ) : ℝ := (x - 4) * (x + 3)
def q (x : ℝ) : ℝ := (x - 4) * (x + 3)

theorem determine_ratio : q 1 ≠ 0 ∧ p 1 / q 1 = 1 := by
  have hq : q 1 ≠ 0 := by
    simp [q]
    norm_num
  have hpq : p 1 / q 1 = 1 := by
    simp [p, q]
    norm_num
  exact ⟨hq, hpq⟩

end NUMINAMATH_GPT_determine_ratio_l494_49434


namespace NUMINAMATH_GPT_question_1_question_2_question_3_l494_49490

variable (a b : ℝ)

-- (a * b)^n = a^n * b^n for natural numbers n
theorem question_1 (n : ℕ) : (a * b)^n = a^n * b^n := sorry

-- Calculate 2^5 * (-1/2)^5
theorem question_2 : 2^5 * (-1/2)^5 = -1 := sorry

-- Calculate (-0.125)^2022 * 2^2021 * 4^2020
theorem question_3 : (-0.125)^2022 * 2^2021 * 4^2020 = 1 / 32 := sorry

end NUMINAMATH_GPT_question_1_question_2_question_3_l494_49490


namespace NUMINAMATH_GPT_shaded_area_l494_49400

theorem shaded_area (PQ : ℝ) (n_squares : ℕ) (d_intersect : ℝ)
  (h1 : PQ = 8) (h2 : n_squares = 20) (h3 : d_intersect = 8) : ∃ (A : ℝ), A = 160 := 
by {
  sorry
}

end NUMINAMATH_GPT_shaded_area_l494_49400


namespace NUMINAMATH_GPT_addition_even_odd_is_odd_subtraction_even_odd_is_odd_squared_sum_even_odd_is_odd_l494_49469

section OperationsAlwaysYieldOdd

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem addition_even_odd_is_odd (a b : ℤ) (h₁ : is_even a) (h₂ : is_odd b) : is_odd (a + b) :=
sorry

theorem subtraction_even_odd_is_odd (a b : ℤ) (h₁ : is_even a) (h₂ : is_odd b) : is_odd (a - b) :=
sorry

theorem squared_sum_even_odd_is_odd (a b : ℤ) (h₁ : is_even a) (h₂ : is_odd b) : is_odd ((a + b) * (a + b)) :=
sorry

end OperationsAlwaysYieldOdd

end NUMINAMATH_GPT_addition_even_odd_is_odd_subtraction_even_odd_is_odd_squared_sum_even_odd_is_odd_l494_49469


namespace NUMINAMATH_GPT_tan_sum_identity_l494_49436

noncomputable def tan_25 := Real.tan (Real.pi / 180 * 25)
noncomputable def tan_35 := Real.tan (Real.pi / 180 * 35)
noncomputable def sqrt_3 := Real.sqrt 3

theorem tan_sum_identity :
  tan_25 + tan_35 + sqrt_3 * tan_25 * tan_35 = 1 :=
by
  sorry

end NUMINAMATH_GPT_tan_sum_identity_l494_49436


namespace NUMINAMATH_GPT_not_rented_two_bedroom_units_l494_49495

theorem not_rented_two_bedroom_units (total_units : ℕ)
  (units_rented_ratio : ℚ)
  (total_rented_units : ℕ)
  (one_bed_room_rented_ratio two_bed_room_rented_ratio three_bed_room_rented_ratio : ℚ)
  (one_bed_room_rented_count two_bed_room_rented_count three_bed_room_rented_count : ℕ)
  (x : ℕ) 
  (total_two_bed_room_units rented_two_bed_room_units : ℕ)
  (units_ratio_condition : 2*x + 3*x + 4*x = total_rented_units)
  (total_units_condition : total_units = 1200)
  (ratio_condition : units_rented_ratio = 7/12)
  (rented_units_condition : total_rented_units = (7/12) * total_units)
  (one_bed_condition : one_bed_room_rented_ratio = 2/5)
  (two_bed_condition : two_bed_room_rented_ratio = 1/2)
  (three_bed_condition : three_bed_room_rented_ratio = 3/8)
  (one_bed_count : one_bed_room_rented_count = 2 * x)
  (two_bed_count : two_bed_room_rented_count = 3 * x)
  (three_bed_count : three_bed_room_rented_count = 4 * x)
  (x_value : x = total_rented_units / 9)
  (total_two_bed_units_calc : total_two_bed_room_units = 2 * two_bed_room_rented_count)
  : total_two_bed_room_units - two_bed_room_rented_count = 231 :=
  by
  sorry

end NUMINAMATH_GPT_not_rented_two_bedroom_units_l494_49495


namespace NUMINAMATH_GPT_football_defeat_points_l494_49458

theorem football_defeat_points (V D F : ℕ) (x : ℕ) :
    3 * V + D + x * F = 8 →
    27 + 6 * x = 32 →
    x = 0 :=
by
    intros h1 h2
    sorry

end NUMINAMATH_GPT_football_defeat_points_l494_49458


namespace NUMINAMATH_GPT_knights_in_company_l494_49468

theorem knights_in_company :
  ∃ k : ℕ, (k = 0 ∨ k = 6) ∧ k ≤ 39 ∧
  ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 39) →
    (∃ i : ℕ, (1 ≤ i ∧ i ≤ 39) ∧ n * k = 1 + (i - 1) * k) →
    ∃ i : ℕ, ∃ nk : ℕ, (nk = i * k ∧ nk ≤ 39 ∧ (nk ∣ k → i = 1 + (i - 1))) :=
by
  sorry

end NUMINAMATH_GPT_knights_in_company_l494_49468


namespace NUMINAMATH_GPT_correct_answers_count_l494_49484

theorem correct_answers_count
  (c w : ℕ)
  (h1 : c + w = 150)
  (h2 : 4 * c - 2 * w = 420) :
  c = 120 := by
  sorry

end NUMINAMATH_GPT_correct_answers_count_l494_49484


namespace NUMINAMATH_GPT_greatest_xy_value_l494_49456

theorem greatest_xy_value :
  ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 7 * x + 5 * y = 200 ∧ x * y = 285 :=
by 
  sorry

end NUMINAMATH_GPT_greatest_xy_value_l494_49456


namespace NUMINAMATH_GPT_part_one_part_two_l494_49420

section part_one
variables {x : ℝ}

def f (x : ℝ) : ℝ := |x - 3| + |x - 2|

theorem part_one : ∀ x : ℝ, f x ≥ 3 ↔ (x ≤ 1 ∨ x ≥ 4) := by
  sorry
end part_one

section part_two
variables {a x : ℝ}

def g (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 2|

theorem part_two : (∀ x ∈ (Set.Icc 1 2), g a x ≤ |x - 4|) → (a ∈ Set.Icc (-3) 0) := by
  sorry
end part_two

end NUMINAMATH_GPT_part_one_part_two_l494_49420


namespace NUMINAMATH_GPT_trajectory_description_l494_49497

def trajectory_of_A (x y : ℝ) (m : ℝ) : Prop :=
  m * x^2 - y^2 = m ∧ y ≠ 0
  
theorem trajectory_description (x y m : ℝ) (h : m ≠ 0) :
  trajectory_of_A x y m →
    (m < -1 → (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1 ∧ ¬(x = -1 ∧ y = 0) ∧ ¬(x = 1 ∧ y = 0)))) ∧
    (m = -1 → (x^2 + y^2 = 1 ∧ ¬(x = -1 ∧ y = 0) ∧ ¬(x = 1 ∧ y = 0))) ∧
    (-1 < m ∧ m < 0 → (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1 ∧ ¬(x = -1 ∧ y = 0) ∧ ¬(x = 1 ∧ y = 0)))) ∧
    (m > 0 → (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1 ∧ ¬(x = -1 ∧ y = 0) ∧ ¬(x = 1 ∧ y = 0)))) :=
by
  intro h_trajectory
  sorry

end NUMINAMATH_GPT_trajectory_description_l494_49497


namespace NUMINAMATH_GPT_framing_feet_required_l494_49432

noncomputable def original_width := 5
noncomputable def original_height := 7
noncomputable def enlargement_factor := 4
noncomputable def border_width := 3
noncomputable def inches_per_foot := 12

theorem framing_feet_required :
  let enlarged_width := original_width * enlargement_factor
  let enlarged_height := original_height * enlargement_factor
  let final_width := enlarged_width + 2 * border_width
  let final_height := enlarged_height + 2 * border_width
  let perimeter := 2 * (final_width + final_height)
  let framing_feet := perimeter / inches_per_foot
  framing_feet = 10 :=
by
  sorry

end NUMINAMATH_GPT_framing_feet_required_l494_49432


namespace NUMINAMATH_GPT_luca_loss_years_l494_49451

variable (months_in_year : ℕ := 12)
variable (barbi_kg_per_month : ℚ := 1.5)
variable (luca_kg_per_year : ℚ := 9)
variable (luca_additional_kg : ℚ := 81)

theorem luca_loss_years (barbi_yearly_loss : ℚ :=
                          barbi_kg_per_month * months_in_year) :
  (81 + barbi_yearly_loss) / luca_kg_per_year = 11 := by
  let total_loss_by_luca := 81 + barbi_yearly_loss
  sorry

end NUMINAMATH_GPT_luca_loss_years_l494_49451


namespace NUMINAMATH_GPT_cafeteria_ordered_red_apples_l494_49425

theorem cafeteria_ordered_red_apples
  (R : ℕ) 
  (h : (R + 17) - 10 = 32) : 
  R = 25 :=
sorry

end NUMINAMATH_GPT_cafeteria_ordered_red_apples_l494_49425


namespace NUMINAMATH_GPT_sin_three_pi_over_two_l494_49459

theorem sin_three_pi_over_two : Real.sin (3 * Real.pi / 2) = -1 := 
by
  sorry

end NUMINAMATH_GPT_sin_three_pi_over_two_l494_49459


namespace NUMINAMATH_GPT_complement_union_A_B_in_U_l494_49401

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

def A : Set ℤ := {-1, 2}

def B : Set ℤ := {x | x^2 - 4 * x + 3 = 0}

theorem complement_union_A_B_in_U :
  (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end NUMINAMATH_GPT_complement_union_A_B_in_U_l494_49401


namespace NUMINAMATH_GPT_runner_injury_point_l494_49461

theorem runner_injury_point
  (v d : ℝ)
  (h1 : 2 * (40 - d) / v = d / v + 11)
  (h2 : 2 * (40 - d) / v = 22) :
  d = 20 := 
by
  sorry

end NUMINAMATH_GPT_runner_injury_point_l494_49461


namespace NUMINAMATH_GPT_find_m_values_l494_49478

-- Given function
def f (m x : ℝ) : ℝ := m * x^2 + 3 * m * x + m - 1

-- Theorem statement
theorem find_m_values (m : ℝ) :
  (∃ x y, f m x = 0 ∧ f m y = 0 ∧ (x = 0 ∨ y = 0)) →
  (m = 1 ∨ m = -(5/4)) :=
by sorry

end NUMINAMATH_GPT_find_m_values_l494_49478


namespace NUMINAMATH_GPT_perimeter_of_square_l494_49430

theorem perimeter_of_square
  (s : ℝ) -- s is the side length of the square
  (h_divided_rectangles : ∀ r, r ∈ {r : ℝ × ℝ | r = (s, s / 6)} → true) -- the square is divided into six congruent rectangles
  (h_perimeter_rect : 2 * (s + s / 6) = 42) -- the perimeter of each of these rectangles is 42 inches
  : 4 * s = 72 := 
sorry

end NUMINAMATH_GPT_perimeter_of_square_l494_49430


namespace NUMINAMATH_GPT_probability_point_not_above_x_axis_l494_49464

theorem probability_point_not_above_x_axis (A B C D : ℝ × ℝ) :
  A = (9, 4) →
  B = (3, -2) →
  C = (-3, -2) →
  D = (3, 4) →
  (1 / 2 : ℚ) = 1 / 2 := 
by 
  intros hA hB hC hD 
  sorry

end NUMINAMATH_GPT_probability_point_not_above_x_axis_l494_49464


namespace NUMINAMATH_GPT_smallest_value_square_l494_49440

theorem smallest_value_square (z : ℂ) (hz : z.re > 0) (A : ℝ) :
  (A = 24 / 25) →
  abs ((Complex.abs z + 1 / Complex.abs z)^2 - (2 - 14 / 25)) = 0 :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_square_l494_49440


namespace NUMINAMATH_GPT_chocolates_cost_l494_49471

-- Define the conditions given in the problem.
def boxes_needed (candies_total : ℕ) (candies_per_box : ℕ) : ℕ := 
    candies_total / candies_per_box

def total_cost_without_discount (num_boxes : ℕ) (cost_per_box : ℕ) : ℕ := 
    num_boxes * cost_per_box

def discount (total_cost : ℕ) : ℕ := 
    total_cost * 10 / 100

def final_cost (total_cost : ℕ) (discount : ℕ) : ℕ :=
    total_cost - discount

-- Theorem stating the total cost of buying 660 chocolate after discount is $138.60
theorem chocolates_cost (candies_total : ℕ) (candies_per_box : ℕ) (cost_per_box : ℕ) : 
     candies_total = 660 ∧ candies_per_box = 30 ∧ cost_per_box = 7 → 
     final_cost (total_cost_without_discount (boxes_needed candies_total candies_per_box) cost_per_box) 
          (discount (total_cost_without_discount (boxes_needed candies_total candies_per_box) cost_per_box)) = 13860 := 
by 
    intros h
    let ⟨h1, h2, h3⟩ := h 
    sorry 

end NUMINAMATH_GPT_chocolates_cost_l494_49471


namespace NUMINAMATH_GPT_cake_fractions_l494_49408

theorem cake_fractions (x y z : ℚ) 
  (h1 : x + y + z = 1)
  (h2 : 2 * z = x)
  (h3 : z = 1 / 2 * (y + 2 / 3 * x)) :
  x = 6 / 11 ∧ y = 2 / 11 ∧ z = 3 / 11 :=
sorry

end NUMINAMATH_GPT_cake_fractions_l494_49408


namespace NUMINAMATH_GPT_subtract_rational_from_zero_yields_additive_inverse_l494_49424

theorem subtract_rational_from_zero_yields_additive_inverse (a : ℚ) : 0 - a = -a := by
  sorry

end NUMINAMATH_GPT_subtract_rational_from_zero_yields_additive_inverse_l494_49424


namespace NUMINAMATH_GPT_average_speed_including_stoppages_l494_49492

/--
If the average speed of a bus excluding stoppages is 50 km/hr, and
the bus stops for 12 minutes per hour, then the average speed of the
bus including stoppages is 40 km/hr.
-/
theorem average_speed_including_stoppages
  (u : ℝ) (Δt : ℝ) (h₁ : u = 50) (h₂ : Δt = 12) : 
  (u * (60 - Δt) / 60) = 40 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_including_stoppages_l494_49492


namespace NUMINAMATH_GPT_simplify_2A_minus_B_evaluate_2A_minus_B_at_1_2_l494_49409

variable (a b : ℤ)
def A : ℤ := b^2 - a^2 + 5 * a * b
def B : ℤ := 3 * a * b + 2 * b^2 - a^2

theorem simplify_2A_minus_B : 2 * A a b - B a b = -a^2 + 7 * a * b := by
  sorry

theorem evaluate_2A_minus_B_at_1_2 : 2 * A 1 2 - B 1 2 = 13 := by
  sorry

end NUMINAMATH_GPT_simplify_2A_minus_B_evaluate_2A_minus_B_at_1_2_l494_49409


namespace NUMINAMATH_GPT_fewer_ducks_than_chickens_and_geese_l494_49452

/-- There are 42 chickens and 48 ducks on the farm, and there are as many geese as there are chickens. 
Prove that there are 36 fewer ducks than the number of chickens and geese combined. -/
theorem fewer_ducks_than_chickens_and_geese (chickens ducks geese : ℕ)
  (h_chickens : chickens = 42)
  (h_ducks : ducks = 48)
  (h_geese : geese = chickens):
  ducks + 36 = chickens + geese :=
by
  sorry

end NUMINAMATH_GPT_fewer_ducks_than_chickens_and_geese_l494_49452


namespace NUMINAMATH_GPT_exists_m_for_division_l494_49429

theorem exists_m_for_division (n : ℕ) (h : 0 < n) : ∃ m : ℕ, n ∣ (2016 ^ m + m) := by
  sorry

end NUMINAMATH_GPT_exists_m_for_division_l494_49429


namespace NUMINAMATH_GPT_carla_total_time_l494_49498

def time_sharpening : ℝ := 15
def time_peeling : ℝ := 3 * time_sharpening
def time_chopping : ℝ := 0.5 * time_peeling
def time_breaks : ℝ := 2 * 5

def total_time : ℝ :=
  time_sharpening + time_peeling + time_chopping + time_breaks

theorem carla_total_time : total_time = 92.5 :=
by sorry

end NUMINAMATH_GPT_carla_total_time_l494_49498


namespace NUMINAMATH_GPT_final_price_wednesday_l494_49421

theorem final_price_wednesday :
  let coffee_price := 6
  let cheesecake_price := 10
  let sandwich_price := 8
  let coffee_discount := 0.25
  let cheesecake_discount_wednesday := 0.10
  let additional_discount := 3
  let sales_tax := 0.05
  let discounted_coffee_price := coffee_price - coffee_price * coffee_discount
  let discounted_cheesecake_price := cheesecake_price - cheesecake_price * cheesecake_discount_wednesday
  let total_price_before_additional_discount := discounted_coffee_price + discounted_cheesecake_price + sandwich_price
  let total_price_after_additional_discount := total_price_before_additional_discount - additional_discount
  let total_price_with_tax := total_price_after_additional_discount + total_price_after_additional_discount * sales_tax
  let final_price := total_price_with_tax.round
  final_price = 19.43 :=
by
  sorry

end NUMINAMATH_GPT_final_price_wednesday_l494_49421


namespace NUMINAMATH_GPT_mitch_total_scoops_l494_49438

theorem mitch_total_scoops :
  (3 : ℝ) / (1/3 : ℝ) + (2 : ℝ) / (1/3 : ℝ) = 15 :=
by
  sorry

end NUMINAMATH_GPT_mitch_total_scoops_l494_49438


namespace NUMINAMATH_GPT_total_points_needed_l494_49406

def num_students : ℕ := 25
def num_weeks : ℕ := 2
def vegetables_per_student_per_week : ℕ := 2
def points_per_vegetable : ℕ := 2

theorem total_points_needed : 
  (num_students * (vegetables_per_student_per_week * num_weeks) * points_per_vegetable) = 200 := by
  sorry

end NUMINAMATH_GPT_total_points_needed_l494_49406


namespace NUMINAMATH_GPT_gcd_18_30_is_6_l494_49433

def gcd_18_30 : ℕ :=
  gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 :=
by {
  -- The step here will involve using properties of gcd and prime factorization,
  -- but we are given the result directly for the purpose of this task.
  sorry
}

end NUMINAMATH_GPT_gcd_18_30_is_6_l494_49433


namespace NUMINAMATH_GPT_triangle_side_count_l494_49489

theorem triangle_side_count
    (x : ℝ)
    (h1 : x + 15 > 40)
    (h2 : x + 40 > 15)
    (h3 : 15 + 40 > x)
    (hx : ∃ (x : ℕ), 25 < x ∧ x < 55) :
    ∃ n : ℕ, n = 29 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_count_l494_49489


namespace NUMINAMATH_GPT_width_of_first_tv_is_24_l494_49496

-- Define the conditions
def height_first_tv := 16
def cost_first_tv := 672
def width_new_tv := 48
def height_new_tv := 32
def cost_new_tv := 1152
def cost_per_sq_inch_diff := 1

-- Define the width of the first TV
def width_first_tv := 24

-- Define the areas
def area_first_tv (W : ℕ) := W * height_first_tv
def area_new_tv := width_new_tv * height_new_tv

-- Define the cost per square inch
def cost_per_sq_inch_first_tv (W : ℕ) := cost_first_tv / area_first_tv W
def cost_per_sq_inch_new_tv := cost_new_tv / area_new_tv

-- The proof statement
theorem width_of_first_tv_is_24 :
  cost_per_sq_inch_first_tv width_first_tv = cost_per_sq_inch_new_tv + cost_per_sq_inch_diff
  := by
    unfold cost_per_sq_inch_first_tv
    unfold area_first_tv
    unfold cost_per_sq_inch_new_tv
    unfold area_new_tv
    sorry -- proof to be filled in

end NUMINAMATH_GPT_width_of_first_tv_is_24_l494_49496


namespace NUMINAMATH_GPT_score_after_7_hours_l494_49419

theorem score_after_7_hours (score : ℕ) (time : ℕ) : 
  (score / time = 90 / 5) → time = 7 → score = 126 :=
by
  sorry

end NUMINAMATH_GPT_score_after_7_hours_l494_49419


namespace NUMINAMATH_GPT_find_A_l494_49410

theorem find_A (A : ℕ) (B : ℕ) (h₀ : 0 ≤ B) (h₁ : B ≤ 999) :
  1000 * A + B = (A * (A + 1)) / 2 → A = 1999 := sorry

end NUMINAMATH_GPT_find_A_l494_49410


namespace NUMINAMATH_GPT_range_of_k_no_third_quadrant_l494_49486

theorem range_of_k_no_third_quadrant (k : ℝ) : ¬(∃ x : ℝ, ∃ y : ℝ, x < 0 ∧ y < 0 ∧ y = k * x + 3) → k ≤ 0 := 
sorry

end NUMINAMATH_GPT_range_of_k_no_third_quadrant_l494_49486


namespace NUMINAMATH_GPT_proof_problems_l494_49404

def otimes (a b : ℝ) : ℝ :=
  a * (1 - b)

theorem proof_problems :
  (otimes 2 (-2) = 6) ∧
  ¬ (∀ (a b : ℝ), otimes a b = otimes b a) ∧
  (∀ (a b : ℝ), a + b = 0 → otimes a a + otimes b b = 2 * a * b) ∧
  ¬ (∀ (a b : ℝ), otimes a b = 0 → a = 0) :=
by
  sorry
 
end NUMINAMATH_GPT_proof_problems_l494_49404


namespace NUMINAMATH_GPT_find_a_l494_49450

theorem find_a (a : ℚ) (h : ∃ r s : ℚ, (r*x + s)^2 = ax^2 + 18*x + 16) : a = 81 / 16 := 
by sorry 

end NUMINAMATH_GPT_find_a_l494_49450


namespace NUMINAMATH_GPT_shara_age_l494_49472

-- Definitions derived from conditions
variables (S : ℕ) (J : ℕ)

-- Jaymee's age is twice Shara's age plus 2
def jaymee_age_relation : Prop := J = 2 * S + 2

-- Jaymee's age is given as 22
def jaymee_age : Prop := J = 22

-- The proof problem to prove Shara's age equals 10
theorem shara_age (h1 : jaymee_age_relation S J) (h2 : jaymee_age J) : S = 10 :=
by 
  sorry

end NUMINAMATH_GPT_shara_age_l494_49472


namespace NUMINAMATH_GPT_range_of_a_l494_49417

-- Define the quadratic function f
def f (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

-- State the theorem that describes the condition and proves the answer
theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁ < 4 → x₂ < 4 → f a x₁ ≥ f a x₂) → a ≤ -3 :=
by
  -- The proof would go here; for now, we skip it
  sorry

end NUMINAMATH_GPT_range_of_a_l494_49417


namespace NUMINAMATH_GPT_melted_ice_cream_depth_l494_49427

theorem melted_ice_cream_depth :
  ∀ (r_sphere r_cylinder : ℝ) (h : ℝ),
    r_sphere = 3 →
    r_cylinder = 10 →
    (4 / 3) * π * r_sphere^3 = 100 * π * h →
    h = 9 / 25 :=
  by
    intros r_sphere r_cylinder h
    intros hr_sphere hr_cylinder
    intros h_volume_eq
    sorry

end NUMINAMATH_GPT_melted_ice_cream_depth_l494_49427


namespace NUMINAMATH_GPT_child_height_at_last_visit_l494_49487

-- Definitions for the problem
def h_current : ℝ := 41.5 -- current height in inches
def Δh : ℝ := 3 -- height growth in inches

-- The proof statement
theorem child_height_at_last_visit : h_current - Δh = 38.5 := by
  sorry

end NUMINAMATH_GPT_child_height_at_last_visit_l494_49487


namespace NUMINAMATH_GPT_min_value_of_f_min_value_achieved_min_value_f_l494_49407

noncomputable def f (x : ℝ) := x + 2 / (2 * x + 1) - 1

theorem min_value_of_f : ∀ x : ℝ, x > 0 → f x ≥ 1/2 := 
by sorry

theorem min_value_achieved : f (1/2) = 1/2 := 
by sorry

theorem min_value_f : ∃ x : ℝ, x > 0 ∧ f x = 1/2 := 
⟨1/2, by norm_num, by sorry⟩

end NUMINAMATH_GPT_min_value_of_f_min_value_achieved_min_value_f_l494_49407


namespace NUMINAMATH_GPT_salary_reduction_l494_49437

theorem salary_reduction (S : ℝ) (x : ℝ) 
  (H1 : S > 0) 
  (H2 : 1.25 * S * (1 - 0.01 * x) = 1.0625 * S) : 
  x = 15 := 
  sorry

end NUMINAMATH_GPT_salary_reduction_l494_49437


namespace NUMINAMATH_GPT_weight_ratio_l494_49453

noncomputable def weight_ratio_proof : Prop :=
  ∃ (R S : ℝ), 
  (R + S = 72) ∧ 
  (1.10 * R + 1.17 * S = 82.8) ∧ 
  (R / S = 1 / 2.5)

theorem weight_ratio : weight_ratio_proof := 
  by
    sorry

end NUMINAMATH_GPT_weight_ratio_l494_49453


namespace NUMINAMATH_GPT_trains_meet_1050_km_from_delhi_l494_49465

def distance_train_meet (t1_departure t2_departure : ℕ) (s1 s2 : ℕ) : ℕ :=
  let t_gap := t2_departure - t1_departure      -- Time difference between the departures in hours
  let d1 := s1 * t_gap                          -- Distance covered by the first train until the second train starts
  let relative_speed := s2 - s1                 -- Relative speed of the second train with respect to the first train
  d1 + s2 * (d1 / relative_speed)               -- Distance from Delhi where they meet

theorem trains_meet_1050_km_from_delhi :
  distance_train_meet 9 14 30 35 = 1050 := by
  -- Definitions based on the problem's conditions
  let t1 := 9          -- First train departs at 9 a.m.
  let t2 := 14         -- Second train departs at 2 p.m. (14:00 in 24-hour format)
  let s1 := 30         -- Speed of the first train in km/h
  let s2 := 35         -- Speed of the second train in km/h
  sorry -- proof to be filled in

end NUMINAMATH_GPT_trains_meet_1050_km_from_delhi_l494_49465


namespace NUMINAMATH_GPT_sedrich_more_jelly_beans_l494_49482

-- Define the given conditions
def napoleon_jelly_beans : ℕ := 17
def mikey_jelly_beans : ℕ := 19
def sedrich_jelly_beans (x : ℕ) : ℕ := napoleon_jelly_beans + x

-- Define the main theorem to be proved
theorem sedrich_more_jelly_beans (x : ℕ) :
  2 * (napoleon_jelly_beans + sedrich_jelly_beans x) = 4 * mikey_jelly_beans → x = 4 :=
by
  -- Proving the theorem
  sorry

end NUMINAMATH_GPT_sedrich_more_jelly_beans_l494_49482


namespace NUMINAMATH_GPT_toby_deleted_nine_bad_shots_l494_49412

theorem toby_deleted_nine_bad_shots 
  (x : ℕ)
  (h1 : 63 > x)
  (h2 : (63 - x) + 15 - 3 = 84)
  : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_toby_deleted_nine_bad_shots_l494_49412


namespace NUMINAMATH_GPT_intersection_points_zero_l494_49402

theorem intersection_points_zero (a b c: ℝ) (h1: b^2 = a * c) (h2: a * c > 0) : 
  ∀ x: ℝ, ¬ (a * x^2 + b * x + c = 0) := 
by 
  sorry

end NUMINAMATH_GPT_intersection_points_zero_l494_49402


namespace NUMINAMATH_GPT_determine_fraction_l494_49457

noncomputable def q (x : ℝ) : ℝ := (x + 3) * (x - 2)

noncomputable def p (a b x : ℝ) : ℝ := a * x^2 + b * x

theorem determine_fraction (a b : ℝ) (h : a + b = 1 / 4) :
  (p a b (-1)) / (q (-1)) = (a - b) / 4 :=
by
  sorry

end NUMINAMATH_GPT_determine_fraction_l494_49457


namespace NUMINAMATH_GPT_johns_profit_l494_49493

variable (numDucks : ℕ) (duckCost : ℕ) (duckWeight : ℕ) (sellPrice : ℕ)

def totalCost (numDucks duckCost : ℕ) : ℕ :=
  numDucks * duckCost

def totalWeight (numDucks duckWeight : ℕ) : ℕ :=
  numDucks * duckWeight

def totalRevenue (totalWeight sellPrice : ℕ) : ℕ :=
  totalWeight * sellPrice

def profit (totalRevenue totalCost : ℕ) : ℕ :=
  totalRevenue - totalCost

theorem johns_profit :
  totalCost 30 10 = 300 →
  totalWeight 30 4 = 120 →
  totalRevenue 120 5 = 600 →
  profit 600 300 = 300 :=
  by
    intros
    sorry

end NUMINAMATH_GPT_johns_profit_l494_49493


namespace NUMINAMATH_GPT_center_of_circle_point_not_on_circle_l494_49439

-- Definitions and conditions
def circle_eq (x y : ℝ) := x^2 - 6 * x + y^2 + 2 * y - 11 = 0

-- The problem statement split into two separate theorems

-- Proving the center of the circle is (3, -1)
theorem center_of_circle : 
  ∃ h k : ℝ, (∀ x y, circle_eq x y ↔ (x - h)^2 + (y - k)^2 = 21) ∧ (h, k) = (3, -1) := sorry

-- Proving the point (5, -1) does not lie on the circle
theorem point_not_on_circle : ¬ circle_eq 5 (-1) := sorry

end NUMINAMATH_GPT_center_of_circle_point_not_on_circle_l494_49439


namespace NUMINAMATH_GPT_math_pattern_l494_49474

theorem math_pattern (n : ℕ) : (2 * n - 1) * (2 * n + 1) = (2 * n) ^ 2 - 1 :=
by
  sorry

end NUMINAMATH_GPT_math_pattern_l494_49474


namespace NUMINAMATH_GPT_butterfly_1023_distance_l494_49463

noncomputable def omega : Complex := Complex.exp (Complex.I * Real.pi / 4)

noncomputable def Q (n : ℕ) : Complex :=
  match n with
  | 0     => 0
  | k + 1 => Q k + (k + 1) * omega ^ k

noncomputable def butterfly_distance (n : ℕ) : ℝ := Complex.abs (Q n)

theorem butterfly_1023_distance : butterfly_distance 1023 = 511 * Real.sqrt (2 + Real.sqrt 2) :=
  sorry

end NUMINAMATH_GPT_butterfly_1023_distance_l494_49463


namespace NUMINAMATH_GPT_replace_asterisks_l494_49443

theorem replace_asterisks (x : ℝ) (h : (x / 21) * (x / 84) = 1) : x = 42 :=
sorry

end NUMINAMATH_GPT_replace_asterisks_l494_49443


namespace NUMINAMATH_GPT_fraction_calculation_l494_49444

theorem fraction_calculation :
  ( (3 / 7 + 5 / 8 + 1 / 3) / (5 / 12 + 2 / 9) = 2097 / 966 ) :=
by
  sorry

end NUMINAMATH_GPT_fraction_calculation_l494_49444


namespace NUMINAMATH_GPT_ratio_x_y_l494_49483

theorem ratio_x_y (x y : ℝ) (h : (1/x - 1/y) / (1/x + 1/y) = 2023) : (x + y) / (x - y) = -1 := 
by
  sorry

end NUMINAMATH_GPT_ratio_x_y_l494_49483


namespace NUMINAMATH_GPT_solve_for_x_l494_49442

theorem solve_for_x (x : ℝ) (h : (8 - x)^2 = x^2) : x = 4 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l494_49442


namespace NUMINAMATH_GPT_area_of_square_eq_36_l494_49491

theorem area_of_square_eq_36 :
  ∃ (s q : ℝ), q = 6 ∧ s = 10 ∧ (∃ p : ℝ, p = 24 ∧ (p / 4) * (p / 4) = 36) := 
by
  sorry

end NUMINAMATH_GPT_area_of_square_eq_36_l494_49491


namespace NUMINAMATH_GPT_max_volume_solid_l494_49460

-- Define volumes of individual cubes
def cube_volume (side: ℕ) : ℕ := side * side * side

-- Calculate the total number of cubes in the solid
def total_cubes (base_layer : ℕ) (second_layer : ℕ) : ℕ := base_layer + second_layer

-- Define the base layer and second layer cubes
def base_layer_cubes : ℕ := 4 * 4
def second_layer_cubes : ℕ := 2 * 2

-- Define the total volume of the solid
def total_volume (side_length : ℕ) (base_layer : ℕ) (second_layer : ℕ) : ℕ := 
  total_cubes base_layer second_layer * cube_volume side_length

theorem max_volume_solid :
  total_volume 3 base_layer_cubes second_layer_cubes = 540 := by
  sorry

end NUMINAMATH_GPT_max_volume_solid_l494_49460


namespace NUMINAMATH_GPT_company_fund_initial_amount_l494_49415

theorem company_fund_initial_amount (n : ℕ) :
  (70 * n + 75 = 80 * n - 20) →
  (n = 9) →
  (80 * n - 20 = 700) :=
by
  intros h1 h2
  rw [h2] at h1
  linarith

end NUMINAMATH_GPT_company_fund_initial_amount_l494_49415


namespace NUMINAMATH_GPT_find_xy_l494_49405

theorem find_xy :
  ∃ (x y : ℝ), (x - 14)^2 + (y - 15)^2 + (x - y)^2 = 1/3 ∧ x = 14 + 1/3 ∧ y = 14 + 2/3 :=
by
  sorry

end NUMINAMATH_GPT_find_xy_l494_49405


namespace NUMINAMATH_GPT_circle_areas_equal_l494_49494

theorem circle_areas_equal :
  let r1 := 15
  let d2 := 30
  let r2 := d2 / 2
  let A1 := Real.pi * r1^2
  let A2 := Real.pi * r2^2
  A1 = A2 :=
by
  sorry

end NUMINAMATH_GPT_circle_areas_equal_l494_49494


namespace NUMINAMATH_GPT_num_females_math_not_english_is_15_l494_49447

-- Define the conditions
def male_math := 120
def female_math := 80
def female_english := 120
def male_english := 80
def total_students := 260
def both_male := 75

def female_math_not_english : Nat :=
  female_math - (female_english + female_math - (total_students - (male_math + male_english - both_male)))

theorem num_females_math_not_english_is_15 :
  female_math_not_english = 15 :=
by
  -- This is where the proof will be, but for now, we use 'sorry' to skip it.
  sorry

end NUMINAMATH_GPT_num_females_math_not_english_is_15_l494_49447


namespace NUMINAMATH_GPT_sin_neg_1290_l494_49411

theorem sin_neg_1290 : Real.sin (-(1290 : ℝ) * Real.pi / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_sin_neg_1290_l494_49411


namespace NUMINAMATH_GPT_sum_of_coefficients_zero_l494_49485

theorem sum_of_coefficients_zero (A B C D E F : ℝ) :
  (∀ x : ℝ,
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
      A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 :=
by
  intro h
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_zero_l494_49485


namespace NUMINAMATH_GPT_rectangular_diagonal_length_l494_49423

theorem rectangular_diagonal_length (x y z : ℝ) 
  (h_surface_area : 2 * (x * y + y * z + z * x) = 11)
  (h_edge_sum : x + y + z = 6) :
  Real.sqrt (x^2 + y^2 + z^2) = 5 := 
by
  sorry

end NUMINAMATH_GPT_rectangular_diagonal_length_l494_49423


namespace NUMINAMATH_GPT_probability_all_white_balls_drawn_l494_49413

theorem probability_all_white_balls_drawn (total_balls white_balls black_balls drawn_balls : ℕ) 
  (h_total : total_balls = 15) (h_white : white_balls = 7) (h_black : black_balls = 8) (h_drawn : drawn_balls = 7) :
  (Nat.choose 7 7 : ℚ) / (Nat.choose 15 7 : ℚ) = 1 / 6435 := by
sorry

end NUMINAMATH_GPT_probability_all_white_balls_drawn_l494_49413


namespace NUMINAMATH_GPT_minimum_value_problem_l494_49422

theorem minimum_value_problem (a b c : ℝ) (hb : a > 0 ∧ b > 0 ∧ c > 0)
  (h : (a / b + b / c + c / a) + (b / a + c / b + a / c) = 10) : 
  ∃ x, (x = 47) ∧ (a / b + b / c + c / a) * (b / a + c / b + a / c) ≥ x :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_problem_l494_49422


namespace NUMINAMATH_GPT_inscribed_sphere_radius_l494_49446

theorem inscribed_sphere_radius 
  (V : ℝ) (S1 S2 S3 S4 : ℝ) (R : ℝ) :
  (1 / 3) * R * (S1 + S2 + S3 + S4) = V ↔ R = (3 * V) / (S1 + S2 + S3 + S4) := 
by
  sorry

end NUMINAMATH_GPT_inscribed_sphere_radius_l494_49446


namespace NUMINAMATH_GPT_log_expression_l494_49462

theorem log_expression :
  (Real.log 2)^2 + Real.log 2 * Real.log 5 + Real.log 5 = 1 := by
  sorry

end NUMINAMATH_GPT_log_expression_l494_49462


namespace NUMINAMATH_GPT_problem_l494_49454

namespace arithmetic_sequence

def is_arithmetic_sequence (a : ℕ → ℚ) := ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem problem 
  (a : ℕ → ℚ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_cond : a 1 + a 7 + a 13 = 4) : a 2 + a 12 = 8 / 3 :=
sorry

end arithmetic_sequence

end NUMINAMATH_GPT_problem_l494_49454


namespace NUMINAMATH_GPT_good_number_sum_l494_49441

def is_good (n : ℕ) : Prop := ∃ k : ℕ, n = k * (k + 1)

theorem good_number_sum (a : ℕ) (h1 : a > 6) (h2 : is_good a) :
  ∃ x y : ℕ, is_good x ∧ is_good y ∧ a * (a + 1) = x * (x + 1) + 3 * y * (y + 1) :=
sorry

end NUMINAMATH_GPT_good_number_sum_l494_49441


namespace NUMINAMATH_GPT_estimate_fish_population_l494_49426

theorem estimate_fish_population :
  ∀ (initial_tagged: ℕ) (august_sample: ℕ) (tagged_in_august: ℕ) (leaving_rate: ℝ) (new_rate: ℝ),
  initial_tagged = 50 →
  august_sample = 80 →
  tagged_in_august = 4 →
  leaving_rate = 0.30 →
  new_rate = 0.45 →
  ∃ (april_population : ℕ),
  april_population = 550 :=
by
  intros initial_tagged august_sample tagged_in_august leaving_rate new_rate
  intros h_initial_tagged h_august_sample h_tagged_in_august h_leaving_rate h_new_rate
  existsi 550
  sorry

end NUMINAMATH_GPT_estimate_fish_population_l494_49426


namespace NUMINAMATH_GPT_painting_price_after_5_years_l494_49466

variable (P : ℝ)
-- Conditions on price changes over the years
def year1_price (P : ℝ) := P * 1.30
def year2_price (P : ℝ) := year1_price P * 0.80
def year3_price (P : ℝ) := year2_price P * 1.25
def year4_price (P : ℝ) := year3_price P * 0.90
def year5_price (P : ℝ) := year4_price P * 1.15

theorem painting_price_after_5_years (P : ℝ) :
  year5_price P = 1.3455 * P := by
  sorry

end NUMINAMATH_GPT_painting_price_after_5_years_l494_49466


namespace NUMINAMATH_GPT_european_math_school_gathering_l494_49475

theorem european_math_school_gathering :
  ∃ n : ℕ, n < 400 ∧ n % 17 = 16 ∧ n % 19 = 12 ∧ n = 288 :=
by
  sorry

end NUMINAMATH_GPT_european_math_school_gathering_l494_49475


namespace NUMINAMATH_GPT_sqrt_9025_squared_l494_49431

-- Define the square root function and its properties
noncomputable def sqrt (x : ℕ) : ℕ := sorry

axiom sqrt_def (n : ℕ) (hn : 0 ≤ n) : (sqrt n) ^ 2 = n

-- Prove the specific case
theorem sqrt_9025_squared : (sqrt 9025) ^ 2 = 9025 :=
sorry

end NUMINAMATH_GPT_sqrt_9025_squared_l494_49431


namespace NUMINAMATH_GPT_solvable_eq_l494_49435

theorem solvable_eq (x : ℝ) :
    Real.sqrt ((3 + Real.sqrt 5) ^ x) + Real.sqrt ((3 - Real.sqrt 5) ^ x) = 6 →
    (x = 2 ∨ x = -2) :=
by
  sorry

end NUMINAMATH_GPT_solvable_eq_l494_49435


namespace NUMINAMATH_GPT_transistors_2004_l494_49428

-- Definition of Moore's law specifying the initial amount and the doubling period
def moores_law (initial : ℕ) (years : ℕ) (doubling_period : ℕ) : ℕ :=
  initial * 2 ^ (years / doubling_period)

-- Condition: The number of transistors in 1992
def initial_1992 : ℕ := 2000000

-- Condition: The number of years between 1992 and 2004
def years_between : ℕ := 2004 - 1992

-- Condition: Doubling period every 2 years
def doubling_period : ℕ := 2

-- Goal: Prove the number of transistors in 2004 using the conditions above
theorem transistors_2004 : moores_law initial_1992 years_between doubling_period = 128000000 :=
by
  sorry

end NUMINAMATH_GPT_transistors_2004_l494_49428


namespace NUMINAMATH_GPT_min_time_shoe_horses_l494_49477

variable (blacksmiths horses hooves_per_horse minutes_per_hoof : ℕ)
variable (total_time : ℕ)

theorem min_time_shoe_horses (h_blacksmiths : blacksmiths = 48) 
                            (h_horses : horses = 60)
                            (h_hooves_per_horse : hooves_per_horse = 4)
                            (h_minutes_per_hoof : minutes_per_hoof = 5)
                            (h_total_time : total_time = (horses * hooves_per_horse * minutes_per_hoof) / blacksmiths) :
                            total_time = 25 := 
by
  sorry

end NUMINAMATH_GPT_min_time_shoe_horses_l494_49477


namespace NUMINAMATH_GPT_total_fruits_on_display_l494_49414

-- Declare the variables and conditions as hypotheses
variables (apples oranges bananas : ℕ)
variables (h1 : apples = 2 * oranges)
variables (h2 : oranges = 2 * bananas)
variables (h3 : bananas = 5)

-- Define what we want to prove
theorem total_fruits_on_display : apples + oranges + bananas = 35 :=
by sorry

end NUMINAMATH_GPT_total_fruits_on_display_l494_49414


namespace NUMINAMATH_GPT_leo_assignment_third_part_time_l494_49473

-- Define all the conditions as variables
def first_part_time : ℕ := 25
def first_break : ℕ := 10
def second_part_time : ℕ := 2 * first_part_time
def second_break : ℕ := 15
def total_time : ℕ := 150

-- The calculated total time of the first two parts and breaks
def time_spent_on_first_two_parts_and_breaks : ℕ :=
  first_part_time + first_break + second_part_time + second_break

-- The remaining time for the third part of the assignment
def third_part_time : ℕ :=
  total_time - time_spent_on_first_two_parts_and_breaks

-- The theorem to prove that the time Leo took to finish the third part is 50 minutes
theorem leo_assignment_third_part_time : third_part_time = 50 := by
  sorry

end NUMINAMATH_GPT_leo_assignment_third_part_time_l494_49473


namespace NUMINAMATH_GPT_nonnegative_integers_existence_l494_49470

open Classical

theorem nonnegative_integers_existence (x y : ℕ) : 
  (∃ (a b c d : ℕ), x = a + 2 * b + 3 * c + 7 * d ∧ y = b + 2 * c + 5 * d) ↔ (5 * x ≥ 7 * y) :=
by
  sorry

end NUMINAMATH_GPT_nonnegative_integers_existence_l494_49470


namespace NUMINAMATH_GPT_isosceles_triangle_sin_vertex_angle_l494_49480

theorem isosceles_triangle_sin_vertex_angle (A : ℝ) (hA : 0 < A ∧ A < π / 2) 
  (hSinA : Real.sin A = 5 / 13) : 
  Real.sin (2 * A) = 120 / 169 :=
by 
  -- This placeholder indicates where the proof would go
  sorry

end NUMINAMATH_GPT_isosceles_triangle_sin_vertex_angle_l494_49480


namespace NUMINAMATH_GPT_wasting_water_notation_l494_49448

theorem wasting_water_notation (saving_wasting : ℕ → ℤ)
  (h_pos : saving_wasting 30 = 30) :
  saving_wasting 10 = -10 :=
by
  sorry

end NUMINAMATH_GPT_wasting_water_notation_l494_49448


namespace NUMINAMATH_GPT_max_real_roots_among_polynomials_l494_49499

noncomputable def largest_total_real_roots (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) : ℕ :=
  4  -- representing the largest total number of real roots

theorem max_real_roots_among_polynomials
  (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
  largest_total_real_roots a b c h_a h_b h_c = 4 :=
sorry

end NUMINAMATH_GPT_max_real_roots_among_polynomials_l494_49499


namespace NUMINAMATH_GPT_range_of_x_l494_49455

theorem range_of_x {f : ℝ → ℝ} (h_even : ∀ x, f x = f (-x)) 
  (h_mono_dec : ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x)
  (h_f2 : f 2 = 0)
  (h_pos : ∀ x, f (x - 1) > 0) : 
  ∀ x, -1 < x ∧ x < 3 ↔ f (x - 1) > 0 :=
sorry

end NUMINAMATH_GPT_range_of_x_l494_49455


namespace NUMINAMATH_GPT_value_of_neg2_neg4_l494_49467

def operation (a b x y : ℤ) : ℤ := a * x - b * y

theorem value_of_neg2_neg4 (a b : ℤ) (h : operation a b 1 2 = 8) : operation a b (-2) (-4) = -16 := by
  sorry

end NUMINAMATH_GPT_value_of_neg2_neg4_l494_49467


namespace NUMINAMATH_GPT_sequence_general_term_l494_49418

def recurrence_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n / (1 + a n)

theorem sequence_general_term :
  ∀ a : ℕ → ℚ, recurrence_sequence a → ∀ n : ℕ, n ≥ 1 → a n = 2 / (2 * n - 1) :=
by
  intro a h n hn
  sorry

end NUMINAMATH_GPT_sequence_general_term_l494_49418


namespace NUMINAMATH_GPT_least_actual_square_area_l494_49481

theorem least_actual_square_area :
  let side_measured := 7
  let lower_bound := 6.5
  let actual_area := lower_bound * lower_bound
  actual_area = 42.25 :=
by
  sorry

end NUMINAMATH_GPT_least_actual_square_area_l494_49481
