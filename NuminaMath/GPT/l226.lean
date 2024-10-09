import Mathlib

namespace min_k_l_sum_l226_22668

theorem min_k_l_sum (k l : ℕ) (hk : 120 * k = l^3) (hpos_k : k > 0) (hpos_l : l > 0) :
  k + l = 255 :=
sorry

end min_k_l_sum_l226_22668


namespace compute_b_l226_22684

open Real

theorem compute_b
  (a : ℚ) 
  (b : ℚ) 
  (h₀ : (3 + sqrt 5) ^ 3 + a * (3 + sqrt 5) ^ 2 + b * (3 + sqrt 5) + 12 = 0) 
  : b = -14 :=
sorry

end compute_b_l226_22684


namespace value_of_a2019_l226_22692

noncomputable def a : ℕ → ℝ
| 0 => 3
| (n + 1) => 1 / (1 - a n)

theorem value_of_a2019 : a 2019 = 2 / 3 :=
sorry

end value_of_a2019_l226_22692


namespace problem_l226_22605

theorem problem (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (x * y) + x = x * f y + f x)
  (h2 : f (1 / 2) = 0) : 
  f (-201) = 403 :=
sorry

end problem_l226_22605


namespace sum_last_two_digits_l226_22620

theorem sum_last_two_digits (a b : ℕ) (h₁ : a = 7) (h₂ : b = 13) : 
  (a^25 + b^25) % 100 = 0 :=
by
  sorry

end sum_last_two_digits_l226_22620


namespace fish_farm_estimated_mass_l226_22696

noncomputable def total_fish_mass_in_pond 
  (initial_fry: ℕ) 
  (survival_rate: ℝ) 
  (haul1_count: ℕ) (haul1_avg_weight: ℝ) 
  (haul2_count: ℕ) (haul2_avg_weight: ℝ) 
  (haul3_count: ℕ) (haul3_avg_weight: ℝ) : ℝ :=
  let surviving_fish := initial_fry * survival_rate
  let total_mass_haul1 := haul1_count * haul1_avg_weight
  let total_mass_haul2 := haul2_count * haul2_avg_weight
  let total_mass_haul3 := haul3_count * haul3_avg_weight
  let average_weight_per_fish := (total_mass_haul1 + total_mass_haul2 + total_mass_haul3) / (haul1_count + haul2_count + haul3_count)
  average_weight_per_fish * surviving_fish

theorem fish_farm_estimated_mass :
  total_fish_mass_in_pond 
    80000           -- initial fry
    0.95            -- survival rate
    40 2.5          -- first haul: 40 fish, 2.5 kg each
    25 2.2          -- second haul: 25 fish, 2.2 kg each
    35 2.8          -- third haul: 35 fish, 2.8 kg each
    = 192280 := by
  sorry

end fish_farm_estimated_mass_l226_22696


namespace pears_sold_l226_22600

theorem pears_sold (m a total : ℕ) (h1 : a = 2 * m) (h2 : m = 120) (h3 : a = 240) : total = 360 :=
by
  sorry

end pears_sold_l226_22600


namespace number_of_integer_values_x_floor_2_sqrt_x_eq_12_l226_22660

theorem number_of_integer_values_x_floor_2_sqrt_x_eq_12 :
  ∃! n : ℕ, n = 7 ∧ (∀ x : ℕ, (⌊2 * Real.sqrt x⌋ = 12 ↔ 36 ≤ x ∧ x < 43)) :=
by 
  sorry

end number_of_integer_values_x_floor_2_sqrt_x_eq_12_l226_22660


namespace temperature_comparison_l226_22618

theorem temperature_comparison: ¬ (-3 > -0.3) :=
by
  sorry -- Proof goes here, skipped for now.

end temperature_comparison_l226_22618


namespace range_positive_of_odd_increasing_l226_22667

-- Define f as an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- Define f as an increasing function on (-∞,0)
def is_increasing_on_neg (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → y < 0 → f (x) < f (y)

-- Given an odd function that is increasing on (-∞,0) and f(-1) = 0, prove the range of x for which f(x) > 0 is (-1, 0) ∪ (1, +∞)
theorem range_positive_of_odd_increasing (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_increasing : is_increasing_on_neg f)
  (h_f_neg_one : f (-1) = 0) :
  {x : ℝ | f x > 0} = {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | 1 < x} :=
by
  sorry

end range_positive_of_odd_increasing_l226_22667


namespace simplify_expression_l226_22673

variable (x : ℝ)

theorem simplify_expression : 2 * x - 3 * (2 - x) + 4 * (2 + x) - 5 * (1 - 3 * x) = 24 * x - 3 := 
  sorry

end simplify_expression_l226_22673


namespace area_of_rectangular_field_l226_22625

theorem area_of_rectangular_field (W D : ℝ) (hW : W = 15) (hD : D = 17) :
  ∃ L : ℝ, (W * L = 120) ∧ D^2 = L^2 + W^2 :=
by 
  use 8
  sorry

end area_of_rectangular_field_l226_22625


namespace curve_is_line_l226_22693

theorem curve_is_line (r θ : ℝ) (h : r = 2 / (Real.sin θ + Real.cos θ)) : 
  ∃ m b, ∀ θ, r * Real.cos θ = m * (r * Real.sin θ) + b :=
sorry

end curve_is_line_l226_22693


namespace compare_fractions_l226_22604

theorem compare_fractions (a : ℝ) : 
  (a = 0 → (1 / (1 - a)) = (1 + a)) ∧ 
  (0 < a ∧ a < 1 → (1 / (1 - a)) > (1 + a)) ∧ 
  (a > 1 → (1 / (1 - a)) < (1 + a)) := by
  sorry

end compare_fractions_l226_22604


namespace popsicle_sticks_difference_l226_22665

def popsicle_sticks_boys (boys : ℕ) (sticks_per_boy : ℕ) : ℕ :=
  boys * sticks_per_boy

def popsicle_sticks_girls (girls : ℕ) (sticks_per_girl : ℕ) : ℕ :=
  girls * sticks_per_girl

theorem popsicle_sticks_difference : 
    popsicle_sticks_boys 10 15 - popsicle_sticks_girls 12 12 = 6 := by
  sorry

end popsicle_sticks_difference_l226_22665


namespace inequality_solution_set_l226_22685

theorem inequality_solution_set (x : ℝ) : 4 * x^2 - 4 * x + 1 ≥ 0 := 
by
  sorry

end inequality_solution_set_l226_22685


namespace percentage_of_import_tax_l226_22691

noncomputable def total_value : ℝ := 2560
noncomputable def taxable_threshold : ℝ := 1000
noncomputable def import_tax : ℝ := 109.20

theorem percentage_of_import_tax :
  let excess_value := total_value - taxable_threshold
  let percentage_tax := (import_tax / excess_value) * 100
  percentage_tax = 7 := 
by
  sorry

end percentage_of_import_tax_l226_22691


namespace profit_calculation_l226_22633

theorem profit_calculation (investment_john investment_mike profit_john profit_mike: ℕ) 
  (total_profit profit_shared_ratio profit_remaining_profit: ℚ)
  (h_investment_john : investment_john = 700)
  (h_investment_mike : investment_mike = 300)
  (h_total_profit : total_profit = 3000)
  (h_shared_ratio : profit_shared_ratio = total_profit / 3 / 2)
  (h_remaining_profit : profit_remaining_profit = 2 * total_profit / 3)
  (h_profit_john : profit_john = profit_shared_ratio + (7 / 10) * profit_remaining_profit)
  (h_profit_mike : profit_mike = profit_shared_ratio + (3 / 10) * profit_remaining_profit)
  (h_profit_difference : profit_john = profit_mike + 800) :
  total_profit = 3000 := 
by
  sorry

end profit_calculation_l226_22633


namespace lottery_probability_l226_22682

theorem lottery_probability :
  let megaBallProbability := 1 / 30
  let winnerBallCombination := Nat.choose 50 5
  let winnerBallProbability := 1 / winnerBallCombination
  megaBallProbability * winnerBallProbability = 1 / 63562800 :=
by
  let megaBallProbability := 1 / 30
  let winnerBallCombination := Nat.choose 50 5
  have winnerBallCombinationEval: winnerBallCombination = 2118760 := by sorry
  let winnerBallProbability := 1 / winnerBallCombination
  have totalProbability: megaBallProbability * winnerBallProbability = 1 / 63562800 := by sorry
  exact totalProbability

end lottery_probability_l226_22682


namespace count_consecutive_sets_sum_15_l226_22661

theorem count_consecutive_sets_sum_15 : 
  ∃ n : ℕ, 
    (n > 0 ∧
    ∃ a : ℕ, 
      (n ≥ 2 ∧ 
      ∃ s : (Finset ℕ), 
        (∀ x ∈ s, x ≥ 1) ∧ 
        (s.sum id = 15))
  ) → 
  n = 2 :=
  sorry

end count_consecutive_sets_sum_15_l226_22661


namespace part_a_part_b_l226_22613

def g (n : ℕ) : ℕ := (n.digits 10).prod

theorem part_a : ∀ n : ℕ, g n ≤ n :=
by
  -- Proof omitted
  sorry

theorem part_b : {n : ℕ | n^2 - 12*n + 36 = g n} = {4, 9} :=
by
  -- Proof omitted
  sorry

end part_a_part_b_l226_22613


namespace arithmetic_sequence_twelfth_term_l226_22644

theorem arithmetic_sequence_twelfth_term :
  let a1 := (1 : ℚ) / 2;
  let a2 := (5 : ℚ) / 6;
  let d := a2 - a1;
  (a1 + 11 * d) = (25 : ℚ) / 6 :=
by
  let a1 := (1 : ℚ) / 2;
  let a2 := (5 : ℚ) / 6;
  let d := a2 - a1;
  exact sorry

end arithmetic_sequence_twelfth_term_l226_22644


namespace total_surface_area_space_l226_22611

theorem total_surface_area_space (h r1 : ℝ) (h_cond : h = 8) (r1_cond : r1 = 3) : 
  (2 * π * (r1 + 1) * h - 2 * π * r1 * h) = 16 * π := 
by
  sorry

end total_surface_area_space_l226_22611


namespace rainfall_difference_l226_22638

-- Defining the conditions
def march_rainfall : ℝ := 0.81
def april_rainfall : ℝ := 0.46

-- Stating the theorem
theorem rainfall_difference : march_rainfall - april_rainfall = 0.35 := by
  -- insert proof steps here
  sorry

end rainfall_difference_l226_22638


namespace second_experimental_point_is_correct_l226_22699

-- Define the temperature range
def lower_bound : ℝ := 1400
def upper_bound : ℝ := 1600

-- Define the golden ratio constant
def golden_ratio : ℝ := 0.618

-- Calculate the first experimental point using 0.618 method
def first_point : ℝ := lower_bound + golden_ratio * (upper_bound - lower_bound)

-- Calculate the second experimental point
def second_point : ℝ := upper_bound - (first_point - lower_bound)

-- Theorem stating the calculated second experimental point equals 1476.4
theorem second_experimental_point_is_correct :
  second_point = 1476.4 := by
  sorry

end second_experimental_point_is_correct_l226_22699


namespace angles_equal_sixty_degrees_l226_22687

/-- Given a triangle ABC with sides a, b, c and respective angles α, β, γ, and with circumradius R,
if the following equation holds:
    (a * cos α + b * cos β + c * cos γ) / (a * sin β + b * sin γ + c * sin α) = (a + b + c) / (9 * R),
prove that α = β = γ = 60 degrees. -/
theorem angles_equal_sixty_degrees 
  (a b c R : ℝ) 
  (α β γ : ℝ) 
  (h : (a * Real.cos α + b * Real.cos β + c * Real.cos γ) / (a * Real.sin β + b * Real.sin γ + c * Real.sin α) = (a + b + c) / (9 * R)) :
  α = 60 ∧ β = 60 ∧ γ = 60 := 
sorry

end angles_equal_sixty_degrees_l226_22687


namespace ones_digit_542_mul_3_is_6_l226_22637

/--
Given that the ones (units) digit of 542 is 2, prove that the ones digit of 542 multiplied by 3 is 6.
-/
theorem ones_digit_542_mul_3_is_6 (h: ∃ n : ℕ, 542 = 10 * n + 2) : (542 * 3) % 10 = 6 := 
by
  sorry

end ones_digit_542_mul_3_is_6_l226_22637


namespace Jill_earnings_l226_22674

theorem Jill_earnings :
  let earnings_first_month := 10 * 30
  let earnings_second_month := 20 * 30
  let earnings_third_month := 20 * 15
  earnings_first_month + earnings_second_month + earnings_third_month = 1200 :=
by
  sorry

end Jill_earnings_l226_22674


namespace solution_set_of_quadratic_inequality_l226_22695

theorem solution_set_of_quadratic_inequality (x : ℝ) :
  (x^2 ≤ 4) ↔ (-2 ≤ x ∧ x ≤ 2) :=
by 
  sorry

end solution_set_of_quadratic_inequality_l226_22695


namespace triangle_count_l226_22616

def count_triangles (smallest intermediate larger even_larger whole_structure : Nat) : Nat :=
  smallest + intermediate + larger + even_larger + whole_structure

theorem triangle_count :
  count_triangles 2 6 6 6 12 = 32 :=
by
  sorry

end triangle_count_l226_22616


namespace student_ticket_price_l226_22617

-- Define the conditions
variables (S T : ℝ)
def condition1 := 4 * S + 3 * T = 79
def condition2 := 12 * S + 10 * T = 246

-- Prove that the price of a student ticket is 9 dollars, given the equations above
theorem student_ticket_price (h1 : condition1 S T) (h2 : condition2 S T) : T = 9 :=
sorry

end student_ticket_price_l226_22617


namespace solution_set_inequality_l226_22675

theorem solution_set_inequality (a x : ℝ) (h : a > 0) :
  (∀ x, (a + 1 ≤ x ∧ x ≤ a + 3) ↔ (|((2 * x - 3 - 2 * a) / (x - a))| ≤ 1)) := 
sorry

end solution_set_inequality_l226_22675


namespace find_number_l226_22679

theorem find_number (x : ℕ) (h : (537 - x) / (463 + x) = 1 / 9) : x = 437 :=
sorry

end find_number_l226_22679


namespace sanda_exercise_each_day_l226_22690

def exercise_problem (javier_exercise_daily sanda_exercise_total total_minutes : ℕ) (days_in_week : ℕ) :=
  javier_exercise_daily * days_in_week + sanda_exercise_total = total_minutes

theorem sanda_exercise_each_day 
  (javier_exercise_daily : ℕ := 50)
  (days_in_week : ℕ := 7)
  (total_minutes : ℕ := 620)
  (days_sanda_exercised : ℕ := 3): 
  ∃ (sanda_exercise_each_day : ℕ), exercise_problem javier_exercise_daily (sanda_exercise_each_day * days_sanda_exercised) total_minutes days_in_week → sanda_exercise_each_day = 90 :=
by 
  sorry

end sanda_exercise_each_day_l226_22690


namespace chocolates_sold_at_selling_price_l226_22631
noncomputable def chocolates_sold (C S : ℝ) (n : ℕ) : Prop :=
  (35 * C = n * S) ∧ ((S - C) / C * 100) = 66.67

theorem chocolates_sold_at_selling_price : ∃ n : ℕ, ∀ C S : ℝ,
  chocolates_sold C S n → n = 21 :=
by
  sorry

end chocolates_sold_at_selling_price_l226_22631


namespace find_sale_in_second_month_l226_22645

def sale_in_second_month (sale1 sale3 sale4 sale5 sale6 target_average : ℕ) (S : ℕ) : Prop :=
  sale1 + S + sale3 + sale4 + sale5 + sale6 = target_average * 6

theorem find_sale_in_second_month :
  sale_in_second_month 5420 6200 6350 6500 7070 6200 5660 :=
by
  sorry

end find_sale_in_second_month_l226_22645


namespace max_squares_covered_by_card_l226_22635

noncomputable def card_coverage_max_squares (card_side : ℝ) (square_side : ℝ) : ℕ :=
  if card_side = 2 ∧ square_side = 1 then 9 else 0

theorem max_squares_covered_by_card : card_coverage_max_squares 2 1 = 9 := by
  sorry

end max_squares_covered_by_card_l226_22635


namespace grid_shaded_area_l226_22676

theorem grid_shaded_area :
  let grid_side := 12
  let grid_area := grid_side^2
  let radius_small := 1.5
  let radius_large := 3
  let area_small := π * radius_small^2
  let area_large := π * radius_large^2
  let total_area_circles := 3 * area_small + area_large
  let visible_area := grid_area - total_area_circles
  let A := 144
  let B := 15.75
  A = 144 ∧ B = 15.75 ∧ (A + B = 159.75) →
  visible_area = 144 - 15.75 * π :=
by
  intros
  sorry

end grid_shaded_area_l226_22676


namespace smallest_divisor_l226_22678

theorem smallest_divisor (N D : ℕ) (hN : N = D * 7) (hD : D > 0) (hsq : (N / D) = 7) :
  D = 7 :=
by 
  sorry

end smallest_divisor_l226_22678


namespace line_of_symmetry_l226_22636

-- Definitions of the circles and the line
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 4 * y - 1 = 0
def line (x y : ℝ) : Prop := x - y - 2 = 0

-- The theorem stating the symmetry condition
theorem line_of_symmetry :
  ∀ (x y : ℝ), circle1 x y ↔ ∃ (x' y' : ℝ), line ((x + x') / 2) ((y + y') / 2) ∧ circle2 x' y' :=
sorry

end line_of_symmetry_l226_22636


namespace minimal_value_expression_l226_22654

theorem minimal_value_expression (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + c = 1) :
  (a + (ab)^(1/3) + (abc)^(1/4)) ≥ (1/3 + 1/(3 * (3^(1/3))) + 1/(3 * (3^(1/4)))) :=
sorry

end minimal_value_expression_l226_22654


namespace red_chairs_count_l226_22609

-- Given conditions
variables {R Y B : ℕ} -- Assuming the number of chairs are natural numbers

-- Main theorem statement
theorem red_chairs_count : 
  Y = 4 * R ∧ B = Y - 2 ∧ R + Y + B = 43 -> R = 5 :=
by
  sorry

end red_chairs_count_l226_22609


namespace sum_ab_eq_five_l226_22681

theorem sum_ab_eq_five (a b : ℕ) (h : (∃ (ab : ℕ), ab = a * 10 + b ∧ 3 / 13 = ab / 100)) : a + b = 5 :=
sorry

end sum_ab_eq_five_l226_22681


namespace cost_per_gift_l226_22641

theorem cost_per_gift (a b c : ℕ) (hc : c = 70) (ha : a = 3) (hb : b = 4) :
  c / (a + b) = 10 :=
by sorry

end cost_per_gift_l226_22641


namespace distance_between_foci_l226_22626

-- Let the hyperbola be defined by the equation xy = 4.
def hyperbola (x y : ℝ) : Prop := x * y = 4

-- Prove that the distance between the foci of this hyperbola is 8.
theorem distance_between_foci : ∀ (x y : ℝ), hyperbola x y → ∃ d, d = 8 :=
by {
    sorry
}

end distance_between_foci_l226_22626


namespace total_fish_l226_22657

theorem total_fish (x y : ℕ) : (19 - 2 * x) + (27 - 4 * y) = 46 - 2 * x - 4 * y :=
  by
    sorry

end total_fish_l226_22657


namespace find_width_l226_22671

namespace RectangleProblem

variables {w l : ℝ}

-- Conditions
def length_is_three_times_width (w l : ℝ) : Prop := l = 3 * w
def sum_of_length_and_width_equals_three_times_area (w l : ℝ) : Prop := l + w = 3 * (l * w)

-- Theorem statement
theorem find_width (w l : ℝ) (h1 : length_is_three_times_width w l) (h2 : sum_of_length_and_width_equals_three_times_area w l) :
  w = 4 / 9 :=
sorry

end RectangleProblem

end find_width_l226_22671


namespace pyramid_volume_l226_22653

theorem pyramid_volume (S : ℝ) :
  ∃ (V : ℝ),
  (∀ (a b h : ℝ), S = a * b ∧
  h = a * (Real.tan (60 * (Real.pi / 180))) ∧
  h = b * (Real.tan (30 * (Real.pi / 180))) ∧
  V = (1/3) * S * h) →
  V = (S * Real.sqrt S) / 3 :=
by
  sorry

end pyramid_volume_l226_22653


namespace sum_x_y_eq_8_l226_22672

theorem sum_x_y_eq_8 (x y S : ℝ) (h1 : x + y = S) (h2 : y - 3 * x = 7) (h3 : y - x = 7.5) : S = 8 :=
by
  sorry

end sum_x_y_eq_8_l226_22672


namespace distance_traveled_downstream_l226_22634

noncomputable def boat_speed_in_still_water : ℝ := 12
noncomputable def current_speed : ℝ := 4
noncomputable def travel_time_in_minutes : ℝ := 18
noncomputable def travel_time_in_hours : ℝ := travel_time_in_minutes / 60

theorem distance_traveled_downstream :
  let effective_speed := boat_speed_in_still_water + current_speed
  let distance := effective_speed * travel_time_in_hours
  distance = 4.8 := 
by
  sorry

end distance_traveled_downstream_l226_22634


namespace simplify_expr_l226_22614

theorem simplify_expr (x : ℝ) : (3 * x)^5 + (4 * x) * (x^4) = 247 * x^5 :=
by
  sorry

end simplify_expr_l226_22614


namespace determine_location_with_coords_l226_22612

-- Define the conditions as a Lean structure
structure Location where
  longitude : ℝ
  latitude : ℝ

-- Define the specific location given in option ①
def location_118_40 : Location :=
  {longitude := 118, latitude := 40}

-- Define the theorem and its statement
theorem determine_location_with_coords :
  ∃ loc : Location, loc = location_118_40 := 
  by
  sorry -- Placeholder for the proof

end determine_location_with_coords_l226_22612


namespace pencils_in_each_box_l226_22698

theorem pencils_in_each_box (total_pencils : ℕ) (total_boxes : ℕ) (pencils_per_box : ℕ) 
  (h1 : total_pencils = 648) (h2 : total_boxes = 162) : 
  total_pencils / total_boxes = pencils_per_box := 
by
  sorry

end pencils_in_each_box_l226_22698


namespace three_digit_even_two_odd_no_repetition_l226_22643

-- Define sets of digits
def digits : List ℕ := [0, 1, 3, 4, 5, 6]
def evens : List ℕ := [0, 4, 6]
def odds : List ℕ := [1, 3, 5]

noncomputable def total_valid_numbers : ℕ :=
  let choose_0 := 12 -- Given by A_{2}^{1} A_{3}^{2} = 12
  let without_0 := 36 -- Given by C_{2}^{1} * C_{3}^{2} * A_{3}^{3} = 36
  choose_0 + without_0

theorem three_digit_even_two_odd_no_repetition : total_valid_numbers = 48 :=
by
  -- Proof would be provided here
  sorry

end three_digit_even_two_odd_no_repetition_l226_22643


namespace complement_intersection_l226_22689

open Set

-- Definitions of sets
def U : Set ℕ := {2, 3, 4, 5, 6}
def A : Set ℕ := {2, 5, 6}
def B : Set ℕ := {3, 5}

-- The theorem statement
theorem complement_intersection :
  (U \ B) ∩ A = {2, 6} := by
  sorry

end complement_intersection_l226_22689


namespace quadratic_polynomial_l226_22608

noncomputable def p (x : ℝ) : ℝ := (14 * x^2 + 4 * x + 12) / 15

theorem quadratic_polynomial :
  p (-2) = 4 ∧ p 1 = 2 ∧ p 3 = 10 :=
by
  have : p (-2) = (14 * (-2 : ℝ) ^ 2 + 4 * (-2 : ℝ) + 12) / 15 := rfl
  have : p 1 = (14 * (1 : ℝ) ^ 2 + 4 * (1 : ℝ) + 12) / 15 := rfl
  have : p 3 = (14 * (3 : ℝ) ^ 2 + 4 * (3 : ℝ) + 12) / 15 := rfl
  -- You can directly state the equalities or keep track of the computation steps.
  sorry

end quadratic_polynomial_l226_22608


namespace smallest_y_for_square_l226_22648

theorem smallest_y_for_square (y M : ℕ) (h1 : 2310 * y = M^2) (h2 : 2310 = 2 * 3 * 5 * 7 * 11) : y = 2310 :=
by sorry

end smallest_y_for_square_l226_22648


namespace tourism_revenue_scientific_notation_l226_22651

-- Define the conditions given in the problem.
def total_tourism_revenue := 12.41 * 10^9

-- Prove the scientific notation of the total tourism revenue.
theorem tourism_revenue_scientific_notation :
  total_tourism_revenue = 1.241 * 10^9 :=
sorry

end tourism_revenue_scientific_notation_l226_22651


namespace simplify_trig_expression_l226_22650

theorem simplify_trig_expression (α : ℝ) :
  (2 * Real.sin (Real.pi - α) + Real.sin (2 * α)) / (2 * Real.cos (α / 2) ^ 2) = 2 * Real.sin α :=
by
  sorry

end simplify_trig_expression_l226_22650


namespace circumscribed_circle_radius_l226_22601

variables (A B C : ℝ) (a b c : ℝ) (R : ℝ) (area : ℝ)

-- Given conditions
def sides_ratio := a / b = 7 / 5 ∧ b / c = 5 / 3
def triangle_area := area = 45 * Real.sqrt 3
def sides := (a, b, c)
def angles := (A, B, C)

-- Prove radius
theorem circumscribed_circle_radius 
  (h_ratio : sides_ratio a b c)
  (h_area : triangle_area area) :
  R = 14 :=
sorry

end circumscribed_circle_radius_l226_22601


namespace weight_of_each_hardcover_book_l226_22656

theorem weight_of_each_hardcover_book
  (weight_limit : ℕ := 80)
  (hardcover_books : ℕ := 70)
  (textbooks : ℕ := 30)
  (knick_knacks : ℕ := 3)
  (textbook_weight : ℕ := 2)
  (knick_knack_weight : ℕ := 6)
  (over_weight : ℕ := 33)
  (total_weight : ℕ := hardcover_books * x + textbooks * textbook_weight + knick_knacks * knick_knack_weight)
  (weight_eq : total_weight = weight_limit + over_weight) :
  x = 1 / 2 :=
by {
  sorry
}

end weight_of_each_hardcover_book_l226_22656


namespace divisor_is_679_l226_22649

noncomputable def x : ℕ := 8
noncomputable def y : ℕ := 9
noncomputable def z : ℝ := 549.7025036818851
noncomputable def p : ℕ := x^3
noncomputable def q : ℕ := y^3
noncomputable def r : ℕ := p * q

theorem divisor_is_679 (k : ℝ) (h : r / k = z) : k = 679 := by
  sorry

end divisor_is_679_l226_22649


namespace comparison_a_b_c_l226_22607

theorem comparison_a_b_c :
  let a := (1 / 2) ^ (1 / 3)
  let b := (1 / 3) ^ (1 / 2)
  let c := Real.log (3 / Real.pi)
  c < b ∧ b < a :=
by
  sorry

end comparison_a_b_c_l226_22607


namespace solve_absolute_value_eq_l226_22659

theorem solve_absolute_value_eq (x : ℝ) : (|x - 3| = 5 - x) → x = 4 :=
by
  sorry

end solve_absolute_value_eq_l226_22659


namespace range_of_m_l226_22697

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + y - x * y = 0) : 
  ∀ m : ℝ, (xy ≥ m^2 - 6 * m ↔ -2 ≤ m ∧ m ≤ 8) :=
sorry

end range_of_m_l226_22697


namespace min_b1_b2_l226_22606

-- Define the sequence recurrence relation
def sequence_recurrence (b : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, b (n + 2) = (b n + 2011) / (1 + b (n + 1))

-- Problem statement: Prove the minimum value of b₁ + b₂ is 2012
theorem min_b1_b2 (b : ℕ → ℕ) (h : ∀ n ≥ 1, 0 < b n) (rec : sequence_recurrence b) :
  b 1 + b 2 ≥ 2012 :=
sorry

end min_b1_b2_l226_22606


namespace no_positive_integer_solutions_l226_22610

theorem no_positive_integer_solutions (x y z : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  x^3 + 2 * y^3 ≠ 4 * z^3 :=
by
  sorry

end no_positive_integer_solutions_l226_22610


namespace crayons_count_l226_22688

theorem crayons_count
  (crayons_given : Nat := 563)
  (crayons_lost : Nat := 558)
  (crayons_left : Nat := 332) :
  crayons_given + crayons_lost + crayons_left = 1453 := 
sorry

end crayons_count_l226_22688


namespace poly_coeff_sum_l226_22664

theorem poly_coeff_sum (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) (x : ℝ) :
  (2 * x + 3)^8 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + 
                 a_3 * (x + 1)^3 + a_4 * (x + 1)^4 + 
                 a_5 * (x + 1)^5 + a_6 * (x + 1)^6 + 
                 a_7 * (x + 1)^7 + a_8 * (x + 1)^8 →
  a_0 + a_2 + a_4 + a_6 + a_8 = 3281 :=
by
  sorry

end poly_coeff_sum_l226_22664


namespace fraction_to_decimal_l226_22652

theorem fraction_to_decimal :
  (7 : ℚ) / 16 = 0.4375 :=
by sorry

end fraction_to_decimal_l226_22652


namespace tens_digit_of_2023_pow_2024_minus_2025_l226_22670

theorem tens_digit_of_2023_pow_2024_minus_2025 : 
  ∀ (n : ℕ), n = 2023^2024 - 2025 → ((n % 100) / 10) = 0 :=
by
  intros n h
  sorry

end tens_digit_of_2023_pow_2024_minus_2025_l226_22670


namespace correct_operations_result_l226_22666

-- Define conditions and the problem statement
theorem correct_operations_result (x : ℝ) (h1: x / 8 - 12 = 18) : (x * 8) * 12 = 23040 :=
by
  sorry

end correct_operations_result_l226_22666


namespace remaining_course_distance_l226_22669

def total_distance_km : ℝ := 10.5
def distance_to_break_km : ℝ := 1.5
def additional_distance_m : ℝ := 3730.0

theorem remaining_course_distance :
  let total_distance_m := total_distance_km * 1000
  let distance_to_break_m := distance_to_break_km * 1000
  let total_traveled_m := distance_to_break_m + additional_distance_m
  total_distance_m - total_traveled_m = 5270 := by
  sorry

end remaining_course_distance_l226_22669


namespace unique_zero_function_l226_22663

theorem unique_zero_function (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (f x + x + y) = f (x + y) + y * f y) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end unique_zero_function_l226_22663


namespace tiffany_initial_lives_l226_22602

theorem tiffany_initial_lives (x : ℕ) 
    (H1 : x - 14 + 27 = 56) : x = 43 :=
sorry

end tiffany_initial_lives_l226_22602


namespace students_taking_neither_l226_22621

-- Definitions based on conditions
def total_students : ℕ := 60
def students_CS : ℕ := 40
def students_Elec : ℕ := 35
def students_both_CS_and_Elec : ℕ := 25

-- Lean statement to prove the number of students taking neither computer science nor electronics
theorem students_taking_neither : total_students - (students_CS + students_Elec - students_both_CS_and_Elec) = 10 :=
by
  sorry

end students_taking_neither_l226_22621


namespace function_D_min_value_is_2_l226_22630

noncomputable def function_A (x : ℝ) : ℝ := x + 2
noncomputable def function_B (x : ℝ) : ℝ := Real.sin x + 2
noncomputable def function_C (x : ℝ) : ℝ := abs x + 2
noncomputable def function_D (x : ℝ) : ℝ := x^2 + 1

theorem function_D_min_value_is_2
  (x : ℝ) :
  ∃ x, function_D x = 2 := by
  sorry
 
end function_D_min_value_is_2_l226_22630


namespace find_M_plus_N_l226_22642

theorem find_M_plus_N (M N : ℕ) 
  (h1 : 5 / 7 = M / 63) 
  (h2 : 5 / 7 = 70 / N) : 
  M + N = 143 :=
by
  sorry

end find_M_plus_N_l226_22642


namespace routes_from_Bristol_to_Carlisle_l226_22694

-- Given conditions as definitions
def routes_Bristol_to_Birmingham : ℕ := 8
def routes_Birmingham_to_Manchester : ℕ := 5
def routes_Manchester_to_Sheffield : ℕ := 4
def routes_Sheffield_to_Newcastle : ℕ := 3
def routes_Newcastle_to_Carlisle : ℕ := 2

-- Define the total number of routes from Bristol to Carlisle
def total_routes_Bristol_to_Carlisle : ℕ := routes_Bristol_to_Birmingham *
                                            routes_Birmingham_to_Manchester *
                                            routes_Manchester_to_Sheffield *
                                            routes_Sheffield_to_Newcastle *
                                            routes_Newcastle_to_Carlisle

-- The theorem to be proved
theorem routes_from_Bristol_to_Carlisle :
  total_routes_Bristol_to_Carlisle = 960 :=
by
  -- Proof will be provided here
  sorry

end routes_from_Bristol_to_Carlisle_l226_22694


namespace visitors_on_saturday_l226_22646

theorem visitors_on_saturday (S : ℕ) (h1 : S + (S + 40) = 440) : S = 200 := by
  sorry

end visitors_on_saturday_l226_22646


namespace system_solution_l226_22677
-- importing the Mathlib library

-- define the problem with necessary conditions
theorem system_solution (x y : ℝ → ℝ) (x0 y0 : ℝ) 
    (h1 : ∀ t, deriv x t = y t) 
    (h2 : ∀ t, deriv y t = -x t) 
    (h3 : x 0 = x0)
    (h4 : y 0 = y0):
    (∀ t, x t = x0 * Real.cos t + y0 * Real.sin t) ∧ (∀ t, y t = -x0 * Real.sin t + y0 * Real.cos t) ∧ (∀ t, (x t)^2 + (y t)^2 = x0^2 + y0^2) := 
by 
    sorry

end system_solution_l226_22677


namespace find_angle_x_l226_22662

-- Definitions as conditions from the problem statement
def angle_PQR := 120
def angle_PQS (x : ℝ) := 2 * x
def angle_QRS (x : ℝ) := x

-- The theorem to prove
theorem find_angle_x (x : ℝ) (h1 : angle_PQR = 120) (h2 : angle_PQS x + angle_QRS x = angle_PQR) : x = 40 :=
by
  sorry

end find_angle_x_l226_22662


namespace caroline_socks_gift_l226_22640

theorem caroline_socks_gift :
  ∀ (initial lost donated_fraction purchased total received),
    initial = 40 →
    lost = 4 →
    donated_fraction = 2 / 3 →
    purchased = 10 →
    total = 25 →
    received = total - (initial - lost - donated_fraction * (initial - lost) + purchased) →
    received = 3 :=
by
  intros initial lost donated_fraction purchased total received
  intro h_initial h_lost h_donated_fraction h_purchased h_total h_received
  sorry

end caroline_socks_gift_l226_22640


namespace line_through_chord_with_midpoint_l226_22647

theorem line_through_chord_with_midpoint (x y : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ,
    (x = x1 ∧ y = y1 ∨ x = x2 ∧ y = y2) ∧
    x = -1 ∧ y = 1 ∧
    x1^2 / 4 + y1^2 / 3 = 1 ∧
    x2^2 / 4 + y2^2 / 3 = 1) →
  3 * x - 4 * y + 7 = 0 :=
by
  sorry

end line_through_chord_with_midpoint_l226_22647


namespace ratio_area_rectangle_to_square_l226_22619

variable (s : ℝ)
variable (area_square : ℝ := s^2)
variable (longer_side_rectangle : ℝ := 1.2 * s)
variable (shorter_side_rectangle : ℝ := 0.85 * s)
variable (area_rectangle : ℝ := longer_side_rectangle * shorter_side_rectangle)

theorem ratio_area_rectangle_to_square :
  area_rectangle / area_square = 51 / 50 := by
  sorry

end ratio_area_rectangle_to_square_l226_22619


namespace solve_for_x_l226_22655

theorem solve_for_x (x : ℝ) (h : x / 5 + 3 = 4) : x = 5 :=
sorry

end solve_for_x_l226_22655


namespace appropriate_length_of_presentation_l226_22639

theorem appropriate_length_of_presentation (wpm : ℕ) (min_time min_words max_time max_words total_words : ℕ) 
  (h1 : total_words = 160) 
  (h2 : min_time = 45) 
  (h3 : min_words = min_time * wpm) 
  (h4 : max_time = 60) 
  (h5 : max_words = max_time * wpm) : 
  7200 ≤ 9400 ∧ 9400 ≤ 9600 :=
by 
  sorry

end appropriate_length_of_presentation_l226_22639


namespace babylon_game_proof_l226_22624

section BabylonGame

-- Defining the number of holes on the sphere
def number_of_holes : Nat := 26

-- The number of 45° angles formed by the pairs of rays
def num_45_degree_angles : Nat := 40

-- The number of 60° angles formed by the pairs of rays
def num_60_degree_angles : Nat := 48

-- The other angles that can occur between pairs of rays
def other_angles : List Real := [31.4, 81.6, 90]

-- Constructs possible given the conditions
def constructible (shape : String) : Bool :=
  shape = "regular tetrahedron" ∨ shape = "regular octahedron"

-- Constructs not possible given the conditions
def non_constructible (shape : String) : Bool :=
  shape = "joined regular tetrahedrons"

-- Proof problem statement
theorem babylon_game_proof :
  (number_of_holes = 26) →
  (num_45_degree_angles = 40) →
  (num_60_degree_angles = 48) →
  (other_angles = [31.4, 81.6, 90]) →
  (constructible "regular tetrahedron" = True) →
  (constructible "regular octahedron" = True) →
  (non_constructible "joined regular tetrahedrons" = True) :=
  by
    sorry

end BabylonGame

end babylon_game_proof_l226_22624


namespace simplify_expression_l226_22622

open Complex

theorem simplify_expression :
  ((4 + 6 * I) / (4 - 6 * I) * (4 - 6 * I) / (4 + 6 * I) + (4 - 6 * I) / (4 + 6 * I) * (4 + 6 * I) / (4 - 6 * I)) = 2 :=
by
  sorry

end simplify_expression_l226_22622


namespace webinar_end_time_correct_l226_22623

-- Define start time and duration as given conditions
def startTime : Nat := 3*60 + 15  -- 3:15 p.m. in minutes after noon
def duration : Nat := 350         -- duration of the webinar in minutes

-- Define the expected end time in minutes after noon (9:05 p.m. is 9*60 + 5 => 545 minutes after noon)
def endTimeExpected : Nat := 9*60 + 5

-- Statement to prove that the calculated end time matches the expected end time
theorem webinar_end_time_correct : startTime + duration = endTimeExpected :=
by
  sorry

end webinar_end_time_correct_l226_22623


namespace percent_increase_salary_l226_22683

theorem percent_increase_salary (new_salary increase : ℝ) (h_new_salary : new_salary = 90000) (h_increase : increase = 25000) :
  (increase / (new_salary - increase)) * 100 = 38.46 := by
  -- Given values
  have h1 : new_salary = 90000 := h_new_salary
  have h2 : increase = 25000 := h_increase
  -- Compute original salary
  let original_salary : ℝ := new_salary - increase
  -- Compute percent increase
  let percent_increase : ℝ := (increase / original_salary) * 100
  -- Show that the percent increase is 38.46
  have h3 : percent_increase = 38.46 := sorry
  exact h3

end percent_increase_salary_l226_22683


namespace james_initial_marbles_l226_22603

theorem james_initial_marbles (m n : ℕ) (h1 : n = 4) (h2 : m / (n - 1) = 21) :
  m = 28 :=
by sorry

end james_initial_marbles_l226_22603


namespace cos_F_in_triangle_l226_22686

theorem cos_F_in_triangle (D E F : ℝ) (sin_D : ℝ) (cos_E : ℝ) (cos_F : ℝ) 
  (h1 : sin_D = 4 / 5) 
  (h2 : cos_E = 12 / 13) 
  (D_plus_E_plus_F : D + E + F = π) :
  cos_F = -16 / 65 :=
by
  sorry

end cos_F_in_triangle_l226_22686


namespace intersection_is_correct_l226_22629

-- Define the sets A and B based on given conditions
def setA : Set ℝ := {x | ∃ y, y = Real.log (x - 2)}
def setB : Set ℝ := {y | ∃ x, y = Real.sqrt x + 4}

-- Define the intersection of sets A and B
def intersection : Set ℝ := {z | z ≥ 4}

-- The theorem stating that the intersection of A and B is exactly the set [4, +∞)
theorem intersection_is_correct : {x | ∃ y, y = Real.log (x - 2)} ∩ {y | ∃ x, y = Real.sqrt x + 4} = {z | z ≥ 4} :=
by
  sorry

end intersection_is_correct_l226_22629


namespace solve_inequality_l226_22615

theorem solve_inequality (x : ℝ) (h : x ≠ -1) : (2 - x) / (x + 1) ≥ 0 ↔ -1 < x ∧ x ≤ 2 :=
by
  sorry

end solve_inequality_l226_22615


namespace number_of_two_legged_birds_l226_22680

theorem number_of_two_legged_birds
  (b m i : ℕ)  -- Number of birds (b), mammals (m), and insects (i)
  (h_heads : b + m + i = 300)  -- Condition on total number of heads
  (h_legs : 2 * b + 4 * m + 6 * i = 980)  -- Condition on total number of legs
  : b = 110 :=
by
  sorry

end number_of_two_legged_birds_l226_22680


namespace distance_comparison_l226_22632

def distance_mart_to_home : ℕ := 800
def distance_home_to_academy : ℕ := 1300
def distance_academy_to_restaurant : ℕ := 1700

theorem distance_comparison :
  (distance_mart_to_home + distance_home_to_academy) - distance_academy_to_restaurant = 400 :=
by
  sorry

end distance_comparison_l226_22632


namespace find_f_2_l226_22658

noncomputable def f : ℝ → ℝ := sorry

axiom f_additive (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x + y) = f x + f y
axiom f_8 : f 8 = 3

theorem find_f_2 : f 2 = 3 / 4 := 
by sorry

end find_f_2_l226_22658


namespace at_least_two_equal_l226_22628

theorem at_least_two_equal (x y z : ℝ) (h : (x - y) / (2 + x * y) + (y - z) / (2 + y * z) + (z - x) / (2 + z * x) = 0) : 
x = y ∨ y = z ∨ z = x := 
by
  sorry

end at_least_two_equal_l226_22628


namespace minimum_ab_l226_22627

theorem minimum_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : ab = a + 4 * b + 5) : ab ≥ 25 :=
sorry

end minimum_ab_l226_22627
