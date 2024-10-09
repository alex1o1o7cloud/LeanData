import Mathlib

namespace bleach_to_detergent_ratio_changed_factor_l1800_180087

theorem bleach_to_detergent_ratio_changed_factor :
  let original_bleach : ℝ := 4
  let original_detergent : ℝ := 40
  let original_water : ℝ := 100
  let altered_detergent : ℝ := 60
  let altered_water : ℝ := 300

  -- Calculate the factor by which the volume increased
  let original_total_volume := original_detergent + original_water
  let altered_total_volume := altered_detergent + altered_water
  let volume_increase_factor := altered_total_volume / original_total_volume

  -- The calculated factor of the ratio change
  let original_ratio_bleach_to_detergent := original_bleach / original_detergent

  altered_detergent > 0 → altered_water > 0 →
  volume_increase_factor * original_ratio_bleach_to_detergent = 2.5714 :=
by
  -- Insert proof here
  sorry

end bleach_to_detergent_ratio_changed_factor_l1800_180087


namespace living_room_size_is_96_l1800_180006

-- Define the total area of the apartment
def total_area : ℕ := 16 * 10

-- Define the number of units
def units : ℕ := 5

-- Define the size of one unit
def size_of_one_unit : ℕ := total_area / units

-- Define the size of the living room
def living_room_size : ℕ := size_of_one_unit * 3

-- Proving that the living room size is indeed 96 square feet
theorem living_room_size_is_96 : living_room_size = 96 := 
by
  -- not providing proof, thus using sorry
  sorry

end living_room_size_is_96_l1800_180006


namespace probability_smallest_divides_larger_two_l1800_180004

noncomputable def number_of_ways := 20

noncomputable def successful_combinations := 11

theorem probability_smallest_divides_larger_two : (successful_combinations : ℚ) / number_of_ways = 11 / 20 :=
by
  sorry

end probability_smallest_divides_larger_two_l1800_180004


namespace square_of_1024_l1800_180057

theorem square_of_1024 : 1024^2 = 1048576 :=
by
  sorry

end square_of_1024_l1800_180057


namespace regular_polygon_sides_l1800_180092

theorem regular_polygon_sides (n : ℕ) (h1 : 180 * (n - 2) = 144 * n) : n = 10 := 
by
  sorry

end regular_polygon_sides_l1800_180092


namespace units_digit_p_l1800_180066

theorem units_digit_p (p : ℕ) (h1 : p % 2 = 0) (h2 : ((p ^ 3 % 10) - (p ^ 2 % 10)) % 10 = 0) 
(h3 : (p + 4) % 10 = 0) : p % 10 = 6 :=
sorry

end units_digit_p_l1800_180066


namespace factorize_x_squared_minus_one_l1800_180090

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l1800_180090


namespace greatest_missed_problems_l1800_180019

theorem greatest_missed_problems (total_problems : ℕ) (passing_percentage : ℝ) (missed_problems : ℕ) : 
  total_problems = 50 ∧ passing_percentage = 0.85 → missed_problems = 7 :=
by
  sorry

end greatest_missed_problems_l1800_180019


namespace gcd_polynomial_l1800_180097

theorem gcd_polynomial (b : ℤ) (k : ℤ) (hk : k % 2 = 1) (h_b : b = 1193 * k) :
  Int.gcd (2 * b^2 + 31 * b + 73) (b + 17) = 1 := 
  sorry

end gcd_polynomial_l1800_180097


namespace solve_for_x_l1800_180075

theorem solve_for_x : ∀ (x : ℝ), (x = 3 / 4) →
  3 - (1 / (4 * (1 - x))) = 2 * (1 / (4 * (1 - x))) :=
by
  intros x h
  rw [h]
  sorry

end solve_for_x_l1800_180075


namespace camp_boys_count_l1800_180088

/-- The ratio of boys to girls and total number of individuals in the camp including teachers
is given, we prove the number of boys is 26. -/
theorem camp_boys_count 
  (b g t : ℕ) -- b = number of boys, g = number of girls, t = number of teachers
  (h1 : b = 3 * (t - 5))  -- boys count related to some integer "t" minus teachers
  (h2 : g = 4 * (t - 5))  -- girls count related to some integer "t" minus teachers
  (total_individuals : t = 65) : 
  b = 26 :=
by
  have h : 3 * (t - 5) + 4 * (t - 5) + 5 = 65 := sorry
  sorry

end camp_boys_count_l1800_180088


namespace find_x_l1800_180003

variable (x : ℝ)
def vector_a : ℝ × ℝ := (x, 2)
def vector_b : ℝ × ℝ := (x - 1, 1)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem find_x (h1 : dot_product (vector_a x + vector_b x) (vector_a x - vector_b x) = 0) : x = -1 := by 
  sorry

end find_x_l1800_180003


namespace enclosed_area_eq_32_over_3_l1800_180021

def line (x : ℝ) : ℝ := 2 * x + 3
def parabola (x : ℝ) : ℝ := x^2

theorem enclosed_area_eq_32_over_3 :
  ∫ x in (-(1:ℝ))..(3:ℝ), (line x - parabola x) = 32 / 3 :=
by
  sorry

end enclosed_area_eq_32_over_3_l1800_180021


namespace rationalize_denominator_correct_l1800_180034

noncomputable def rationalize_denominator : Prop :=
  (1 / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2)

theorem rationalize_denominator_correct : rationalize_denominator :=
by
  sorry

end rationalize_denominator_correct_l1800_180034


namespace inverse_proportion_l1800_180000

theorem inverse_proportion {x y : ℝ} :
  (y = (3 / x)) -> ¬(y = x / 3) ∧ ¬(y = 3 / (x + 1)) ∧ ¬(y = 3 * x) :=
by
  sorry

end inverse_proportion_l1800_180000


namespace evaluate_expression_l1800_180048

theorem evaluate_expression : 8^3 + 3 * 8^2 + 3 * 8 + 1 = 729 := by
  sorry

end evaluate_expression_l1800_180048


namespace unique_solution_for_a_eq_1_l1800_180068

def equation (a x : ℝ) : Prop :=
  5^(x^2 - 6 * a * x + 9 * a^2) = a * x^2 - 6 * a^2 * x + 9 * a^3 + a^2 - 6 * a + 6

theorem unique_solution_for_a_eq_1 :
  (∃! x : ℝ, equation 1 x) ∧ 
  (∀ a : ℝ, (∃! x : ℝ, equation a x) → a = 1) :=
sorry

end unique_solution_for_a_eq_1_l1800_180068


namespace candies_taken_away_per_incorrect_answer_eq_2_l1800_180089

/-- Define constants and assumptions --/
def candy_per_correct := 3
def correct_answers := 7
def extra_correct_answers := 2
def total_candies_if_extra_correct := 31

/-- The number of candies taken away per incorrect answer --/
def x : ℤ := sorry

/-- Prove that the number of candies taken away for each incorrect answer is 2. --/
theorem candies_taken_away_per_incorrect_answer_eq_2 : 
  ∃ x : ℤ, ((correct_answers + extra_correct_answers) * candy_per_correct - total_candies_if_extra_correct = x + (extra_correct_answers * candy_per_correct - (total_candies_if_extra_correct - correct_answers * candy_per_correct))) ∧ x = 2 := 
by
  exists 2
  sorry

end candies_taken_away_per_incorrect_answer_eq_2_l1800_180089


namespace paper_folding_holes_l1800_180072

def folded_paper_holes (folds: Nat) (holes: Nat) : Nat :=
  match folds with
  | 0 => holes
  | n+1 => 2 * folded_paper_holes n holes

theorem paper_folding_holes : folded_paper_holes 3 1 = 8 :=
by
  -- sorry to skip the proof
  sorry

end paper_folding_holes_l1800_180072


namespace true_propositions_count_l1800_180076

theorem true_propositions_count :
  (∃ x₀ : ℤ, x₀^3 < 0) ∧
  ((∀ a : ℝ, (∃ x : ℝ, a*x^2 + 2*x + 1 = 0 ∧ x < 0) ↔ a ≤ 1) → false) ∧ 
  (¬ (∀ x : ℝ, x^2 = 1/4 * x^2 → y = 1 → false)) →
  true_prop_count = 1 := 
sorry

end true_propositions_count_l1800_180076


namespace geometric_series_sum_l1800_180061

theorem geometric_series_sum (a r : ℝ) (n : ℕ) (last_term : ℝ) 
  (h_a : a = 1) (h_r : r = -3) 
  (h_last_term : last_term = 6561) 
  (h_last_term_eq : a * r^n = last_term) : 
  a * (r^n - 1) / (r - 1) = 4921.25 :=
by
  -- Proof goes here
  sorry

end geometric_series_sum_l1800_180061


namespace tangent_ellipse_hyperbola_l1800_180038

theorem tangent_ellipse_hyperbola (m : ℝ) :
  (∀ x y : ℝ, x^2 + 9 * y^2 = 9 → x^2 - m * (y + 3)^2 = 4) → m = 5 / 9 :=
by
  sorry

end tangent_ellipse_hyperbola_l1800_180038


namespace unknown_number_value_l1800_180053

theorem unknown_number_value (a x : ℕ) (h₁ : a = 105) (h₂ : a^3 = 21 * x * 35 * 63) : x = 25 := by
  sorry

end unknown_number_value_l1800_180053


namespace hexagonal_pyramid_volume_l1800_180052

theorem hexagonal_pyramid_volume (a : ℝ) (h : a > 0) (lateral_surface_area : ℝ) (base_area : ℝ)
  (H_base_area : base_area = (3 * Real.sqrt 3 / 2) * a^2)
  (H_lateral_surface_area : lateral_surface_area = 10 * base_area) :
  (1 / 3) * base_area * (a * Real.sqrt 3 / 2) * 3 * Real.sqrt 11 = (9 * a^3 * Real.sqrt 11) / 4 :=
by sorry

end hexagonal_pyramid_volume_l1800_180052


namespace find_third_number_l1800_180086

theorem find_third_number : ∃ (x : ℝ), 0.3 * 0.8 + x * 0.5 = 0.29 ∧ x = 0.1 :=
by
  use 0.1
  sorry

end find_third_number_l1800_180086


namespace f_prime_neg_one_l1800_180063

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

axiom h1 : ∀ x : ℝ, f x = f (-x)
axiom h2 : ∀ x : ℝ, f (x + 1) - f (1 - x) = 2 * x

theorem f_prime_neg_one : f' (-1) = -1 := by
  -- The proof is omitted
  sorry

end f_prime_neg_one_l1800_180063


namespace advertising_department_employees_l1800_180037

theorem advertising_department_employees (N S A_s x : ℕ) (hN : N = 1000) (hS : S = 80) (hA_s : A_s = 4) 
(h_stratified : x / N = A_s / S) : x = 50 :=
sorry

end advertising_department_employees_l1800_180037


namespace trapezoid_area_is_correct_l1800_180064

def square_side_lengths : List ℕ := [1, 3, 5, 7]
def total_base_length : ℕ := square_side_lengths.sum
def tallest_square_height : ℕ := 7

noncomputable def trapezoid_area_between_segment_and_base : ℚ :=
  let height_at_x (x : ℚ) : ℚ := x * (7/16)
  let base_1 := 4
  let base_2 := 9
  let height_1 := height_at_x base_1
  let height_2 := height_at_x base_2
  ((height_1 + height_2) * (base_2 - base_1) / 2)

theorem trapezoid_area_is_correct :
  trapezoid_area_between_segment_and_base = 14.21875 :=
sorry

end trapezoid_area_is_correct_l1800_180064


namespace incorrect_proposition_l1800_180047

-- Variables and conditions
variable (p q : Prop)
variable (m x a b c : ℝ)
variable (hreal : 1 + 4 * m ≥ 0)

-- Theorem statement
theorem incorrect_proposition :
  ¬ (∀ m > 0, (∃ x : ℝ, x^2 + x - m = 0) → m > 0) :=
sorry

end incorrect_proposition_l1800_180047


namespace combined_loss_percentage_l1800_180042

theorem combined_loss_percentage
  (cost_price_radio : ℕ := 8000)
  (quantity_radio : ℕ := 5)
  (discount_radio : ℚ := 0.1)
  (tax_radio : ℚ := 0.06)
  (sale_price_radio : ℕ := 7200)
  (cost_price_tv : ℕ := 20000)
  (quantity_tv : ℕ := 3)
  (discount_tv : ℚ := 0.15)
  (tax_tv : ℚ := 0.07)
  (sale_price_tv : ℕ := 18000)
  (cost_price_phone : ℕ := 15000)
  (quantity_phone : ℕ := 4)
  (discount_phone : ℚ := 0.08)
  (tax_phone : ℚ := 0.05)
  (sale_price_phone : ℕ := 14500) :
  let total_cost_price := (quantity_radio * cost_price_radio) + (quantity_tv * cost_price_tv) + (quantity_phone * cost_price_phone)
  let total_sale_price := (quantity_radio * sale_price_radio) + (quantity_tv * sale_price_tv) + (quantity_phone * sale_price_phone)
  let total_loss := total_cost_price - total_sale_price
  let loss_percentage := (total_loss * 100 : ℚ) / total_cost_price
  loss_percentage = 7.5 :=
by
  sorry

end combined_loss_percentage_l1800_180042


namespace sum_of_num_and_denom_l1800_180080

-- Define the repeating decimal G
def G : ℚ := 739 / 999

-- State the theorem
theorem sum_of_num_and_denom (a b : ℕ) (hb : b ≠ 0) (h : G = a / b) : a + b = 1738 := sorry

end sum_of_num_and_denom_l1800_180080


namespace cos_7pi_over_6_eq_neg_sqrt3_over_2_l1800_180074

theorem cos_7pi_over_6_eq_neg_sqrt3_over_2 : Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 := by
  sorry

end cos_7pi_over_6_eq_neg_sqrt3_over_2_l1800_180074


namespace solution_set_of_inequality_l1800_180098

theorem solution_set_of_inequality :
  {x : ℝ | (x + 3) * (x - 2) < 0} = {x | -3 < x ∧ x < 2} :=
by sorry

end solution_set_of_inequality_l1800_180098


namespace no_such_function_exists_l1800_180095

-- Let's define the assumptions as conditions
def condition1 (f : ℝ → ℝ) := ∀ x : ℝ, f (x^2) - (f x)^2 ≥ 1 / 4
def distinct_values (f : ℝ → ℝ) := ∀ x y : ℝ, x ≠ y → f x ≠ f y

-- Now we state the main theorem
theorem no_such_function_exists : ¬ ∃ f : ℝ → ℝ, condition1 f ∧ distinct_values f :=
sorry

end no_such_function_exists_l1800_180095


namespace correctly_calculated_value_l1800_180084

theorem correctly_calculated_value (x : ℝ) (hx : x + 0.42 = 0.9) : (x - 0.42) + 0.5 = 0.56 := by
  -- proof to be provided
  sorry

end correctly_calculated_value_l1800_180084


namespace locus_midpoint_l1800_180081

/-- Given a fixed point A (4, -2) and a moving point B on the curve x^2 + y^2 = 4,
    prove that the locus of the midpoint P of the line segment AB satisfies the equation 
    (x - 2)^2 + (y + 1)^2 = 1. -/
theorem locus_midpoint (A B P : ℝ × ℝ)
  (hA : A = (4, -2))
  (hB : ∃ (x y : ℝ), B = (x, y) ∧ x^2 + y^2 = 4)
  (hP : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  (P.1 - 2)^2 + (P.2 + 1)^2 = 1 :=
sorry

end locus_midpoint_l1800_180081


namespace parabola_focus_directrix_distance_l1800_180010

theorem parabola_focus_directrix_distance :
  ∀ {x y : ℝ}, y^2 = (1/4) * x → dist (1/16, 0) (-1/16, 0) = 1/8 := by
sorry

end parabola_focus_directrix_distance_l1800_180010


namespace number_of_white_tshirts_in_one_pack_l1800_180027

namespace TShirts

variable (W : ℕ)

noncomputable def total_white_tshirts := 2 * W
noncomputable def total_blue_tshirts := 4 * 3
noncomputable def cost_per_tshirt := 3
noncomputable def total_cost := 66

theorem number_of_white_tshirts_in_one_pack :
  2 * W * cost_per_tshirt + total_blue_tshirts * cost_per_tshirt = total_cost → W = 5 :=
by
  sorry

end TShirts

end number_of_white_tshirts_in_one_pack_l1800_180027


namespace calculator_unit_prices_and_min_cost_l1800_180077

-- Definitions for conditions
def unit_price_type_A (x : ℕ) : Prop :=
  ∀ y : ℕ, (y = x + 10) → (550 / x = 600 / y)

def purchase_constraint (a : ℕ) : Prop :=
  25 ≤ a ∧ a ≤ 100

def total_cost (a : ℕ) (x y : ℕ) : ℕ :=
  110 * a + 120 * (100 - a)

-- Statement to prove
theorem calculator_unit_prices_and_min_cost :
  ∃ x y, unit_price_type_A x ∧ unit_price_type_A x ∧ total_cost 100 x y = 11000 :=
by
  sorry

end calculator_unit_prices_and_min_cost_l1800_180077


namespace pipe_pumping_rate_l1800_180054

theorem pipe_pumping_rate (R : ℕ) (h : 5 * R + 5 * 192 = 1200) : R = 48 := by
  sorry

end pipe_pumping_rate_l1800_180054


namespace find_fraction_l1800_180024

theorem find_fraction (x y : ℕ) (h₁ : x / (y + 1) = 1 / 2) (h₂ : (x + 1) / y = 1) : x = 2 ∧ y = 3 := by
  sorry

end find_fraction_l1800_180024


namespace oplus_calculation_l1800_180033

def my_oplus (a b : ℝ) : ℝ := 4 * a + 5 * b

theorem oplus_calculation : my_oplus 2 3 = 23 := 
by
    sorry

end oplus_calculation_l1800_180033


namespace maximum_area_right_triangle_hypotenuse_8_l1800_180044

theorem maximum_area_right_triangle_hypotenuse_8 :
  ∃ a b : ℝ, (a^2 + b^2 = 64) ∧ (a * b) / 2 = 16 :=
by
  sorry

end maximum_area_right_triangle_hypotenuse_8_l1800_180044


namespace value_of_m_plus_n_l1800_180096

-- Conditions
variables (m n : ℤ)
def P_symmetric_Q_x_axis := (m - 1 = 2 * m - 4) ∧ (n + 2 = -2)

-- Proof Problem Statement
theorem value_of_m_plus_n (h : P_symmetric_Q_x_axis m n) : (m + n) ^ 2023 = -1 := sorry

end value_of_m_plus_n_l1800_180096


namespace radius_of_circumcircle_l1800_180039

-- Definitions of sides of a triangle and its area
variables {a b c t : ℝ}

-- Condition that t is the area of a triangle with sides a, b, and c
def is_triangle_area (a b c t : ℝ) : Prop := -- Placeholder condition stating these values form a triangle
sorry

-- Statement to prove the given radius formula for the circumscribed circle
theorem radius_of_circumcircle (h : is_triangle_area a b c t) : 
  ∃ r : ℝ, r = abc / (4 * t) :=
sorry

end radius_of_circumcircle_l1800_180039


namespace range_of_m_l1800_180078

noncomputable def distance (m : ℝ) : ℝ := (|m| * Real.sqrt 2 / 2)
theorem range_of_m (m : ℝ) :
  (∃ A B : ℝ × ℝ,
    (A.1 + A.2 + m = 0 ∧ B.1 + B.2 + m = 0) ∧
    (A.1 ^ 2 + A.2 ^ 2 = 2 ∧ B.1 ^ 2 + B.2 ^ 2 = 2) ∧
    (Real.sqrt (A.1 ^ 2 + A.2 ^ 2) + Real.sqrt (B.1 ^ 2 + B.2 ^ 2) ≥ 
     Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)) ∧ (distance m < Real.sqrt 2)) ↔ 
  m ∈ Set.Ioo (-2 : ℝ) (-Real.sqrt 2) ∪ Set.Ioo (Real.sqrt 2) 2 := 
sorry

end range_of_m_l1800_180078


namespace central_angle_of_section_l1800_180028

theorem central_angle_of_section (A : ℝ) (hA : 0 < A) (prob : ℝ) (hprob : prob = 1 / 4) :
  ∃ θ : ℝ, (θ / 360) = prob :=
by
  use 90
  sorry

end central_angle_of_section_l1800_180028


namespace percent_employed_l1800_180026

theorem percent_employed (E : ℝ) : 
  let employed_males := 0.21
  let percent_females := 0.70
  let percent_males := 0.30 -- 1 - percent_females
  (percent_males * E = employed_males) → E = 70 := 
by 
  let employed_males := 0.21
  let percent_females := 0.70
  let percent_males := 0.30
  intro h
  sorry

end percent_employed_l1800_180026


namespace min_value_x_add_y_l1800_180062

variable {x y : ℝ}
variable (hx : 0 < x) (hy : 0 < y)
variable (h : 2 * x + 8 * y - x * y = 0)

theorem min_value_x_add_y : x + y ≥ 18 :=
by
  /- Proof goes here -/
  sorry

end min_value_x_add_y_l1800_180062


namespace tony_drive_time_l1800_180055

noncomputable def time_to_first_friend (d₁ d₂ t₂ : ℝ) : ℝ :=
  let v := d₂ / t₂
  d₁ / v

theorem tony_drive_time (d₁ d₂ t₂ : ℝ) (h_d₁ : d₁ = 120) (h_d₂ : d₂ = 200) (h_t₂ : t₂ = 5) : 
    time_to_first_friend d₁ d₂ t₂ = 3 := by
  rw [h_d₁, h_d₂, h_t₂]
  -- Further simplification would follow here based on the proof steps, which we are omitting
  sorry

end tony_drive_time_l1800_180055


namespace max_workers_l1800_180011

variable {n : ℕ} -- number of workers on the smaller field
variable {S : ℕ} -- area of the smaller field
variable (a : ℕ) -- productivity of each worker

theorem max_workers 
  (h_area : ∀ large small : ℕ, large = 2 * small) 
  (h_workers : ∀ large small : ℕ, large = small + 4) 
  (h_inequality : ∀ (S : ℕ) (n a : ℕ), S / (a * n) > (2 * S) / (a * (n + 4))) :
  2 * n + 4 ≤ 10 :=
by
  -- h_area implies the area requirement
  -- h_workers implies the worker requirement
  -- h_inequality implies the time requirement
  sorry

end max_workers_l1800_180011


namespace joan_picked_apples_l1800_180035

theorem joan_picked_apples (a b c : ℕ) (h1 : b = 27) (h2 : c = 70) (h3 : c = a + b) : a = 43 :=
by
  sorry

end joan_picked_apples_l1800_180035


namespace eventually_periodic_sequence_l1800_180065

theorem eventually_periodic_sequence
  (a : ℕ → ℕ) (h_pos : ∀ n, 0 < a n)
  (h_div : ∀ n m, (a (n + 2 * m)) ∣ (a n + a (n + m))) :
  ∃ N d, 0 < N ∧ 0 < d ∧ ∀ n, N < n → a n = a (n + d) :=
by
  sorry

end eventually_periodic_sequence_l1800_180065


namespace toms_age_ratio_l1800_180099

variables (T N : ℕ)

-- Conditions
def toms_age (T : ℕ) := T
def sum_of_children_ages (T : ℕ) := T
def years_ago (T N : ℕ) := T - N
def children_ages_years_ago (T N : ℕ) := T - 4 * N

-- Given statement
theorem toms_age_ratio (h1 : toms_age T = sum_of_children_ages T)
  (h2 : years_ago T N = 3 * children_ages_years_ago T N) :
  T / N = 11 / 2 :=
sorry

end toms_age_ratio_l1800_180099


namespace isolating_and_counting_bacteria_process_l1800_180069

theorem isolating_and_counting_bacteria_process
  (soil_sampling : Prop)
  (spreading_dilution_on_culture_medium : Prop)
  (decompose_urea : Prop) :
  (soil_sampling ∧ spreading_dilution_on_culture_medium ∧ decompose_urea) →
  (Sample_dilution ∧ Selecting_colonies_that_can_grow ∧ Identification) :=
sorry

end isolating_and_counting_bacteria_process_l1800_180069


namespace hyperbola_y_relation_l1800_180025

theorem hyperbola_y_relation {k y₁ y₂ : ℝ} 
  (A_on_hyperbola : y₁ = k / 2) 
  (B_on_hyperbola : y₂ = k / 3) 
  (k_positive : 0 < k) : 
  y₁ > y₂ := 
sorry

end hyperbola_y_relation_l1800_180025


namespace ReuleauxTriangleFitsAll_l1800_180094

-- Assume definitions for fits into various slots

def FitsTriangular (s : Type) : Prop := sorry
def FitsSquare (s : Type) : Prop := sorry
def FitsCircular (s : Type) : Prop := sorry
def ReuleauxTriangle (s : Type) : Prop := sorry

theorem ReuleauxTriangleFitsAll (s : Type) (h : ReuleauxTriangle s) : 
  FitsTriangular s ∧ FitsSquare s ∧ FitsCircular s := 
  sorry

end ReuleauxTriangleFitsAll_l1800_180094


namespace lines_parallel_if_perpendicular_to_same_plane_l1800_180031

-- Define a plane as a placeholder for other properties
axiom Plane : Type
-- Define Line as a placeholder for other properties
axiom Line : Type

-- Definition of what it means for a line to be perpendicular to a plane
axiom perpendicular_to_plane (l : Line) (π : Plane) : Prop

-- Definition of parallel lines
axiom parallel_lines (l1 l2 : Line) : Prop

-- Define the proof problem in Lean 4
theorem lines_parallel_if_perpendicular_to_same_plane
    (π : Plane) (l1 l2 : Line)
    (h1 : perpendicular_to_plane l1 π)
    (h2 : perpendicular_to_plane l2 π) :
    parallel_lines l1 l2 :=
sorry

end lines_parallel_if_perpendicular_to_same_plane_l1800_180031


namespace grace_putting_down_mulch_hours_l1800_180029

/-- Grace's earnings conditions and hours calculation in September. -/
theorem grace_putting_down_mulch_hours :
  ∃ h : ℕ, 
    6 * 63 + 11 * 9 + 9 * h = 567 ∧
    h = 10 :=
by
  sorry

end grace_putting_down_mulch_hours_l1800_180029


namespace round_robin_tournament_participant_can_mention_all_l1800_180079

theorem round_robin_tournament_participant_can_mention_all :
  ∀ (n : ℕ) (participants : Fin n → Fin n → Prop),
  (∀ i j : Fin n, i ≠ j → (participants i j ∨ participants j i)) →
  (∃ A : Fin n, ∀ (B : Fin n), B ≠ A → (participants A B ∨ ∃ C : Fin n, participants A C ∧ participants C B)) := by
  sorry

end round_robin_tournament_participant_can_mention_all_l1800_180079


namespace find_ad_l1800_180043

-- Defining the two-digit and three-digit numbers
def two_digit (a b : ℕ) : ℕ := 10 * a + b
def three_digit (a b : ℕ) : ℕ := 100 + two_digit a b

def two_digit' (c d : ℕ) : ℕ := 10 * c + d
def three_digit' (c d : ℕ) : ℕ := 100 * c + 10 * d + 1

-- The main problem
theorem find_ad (a b c d : ℕ) (h1 : three_digit a b = three_digit' c d + 15) (h2 : two_digit a b = two_digit' c d + 24) :
    two_digit a d = 32 := by
  sorry

end find_ad_l1800_180043


namespace Tim_placed_rulers_l1800_180091

variable (initial_rulers final_rulers : ℕ)
variable (placed_rulers : ℕ)

-- Given conditions
def initial_rulers_def : initial_rulers = 11 := sorry
def final_rulers_def : final_rulers = 25 := sorry

-- Goal
theorem Tim_placed_rulers : placed_rulers = final_rulers - initial_rulers :=
  by
  sorry

end Tim_placed_rulers_l1800_180091


namespace find_ratio_l1800_180005

open Real

variables (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (q : ℝ)

-- The geometric sequence conditions
def geometric_sequence := ∀ n : ℕ, a (n + 1) = a n * q

-- Sum of the first n terms for the geometric sequence
def sum_of_first_n_terms := ∀ n : ℕ, S n = (a 0) * (1 - q ^ n) / (1 - q)

-- Given conditions
def given_conditions :=
  a 0 + a 2 = 5 / 2 ∧
  a 1 + a 3 = 5 / 4

-- The goal to prove
theorem find_ratio (geo_seq : geometric_sequence a q) (sum_terms : sum_of_first_n_terms a S q) (cond : given_conditions a) :
  S 4 / a 4 = 31 :=
  sorry

end find_ratio_l1800_180005


namespace solve_system_l1800_180060

-- Define the system of equations
def eq1 (x y : ℚ) : Prop := 4 * x - 3 * y = -10
def eq2 (x y : ℚ) : Prop := 6 * x + 5 * y = -13

-- Define the solution
def solution (x y : ℚ) : Prop := x = -89 / 38 ∧ y = 0.21053

-- Prove that the given solution satisfies both equations
theorem solve_system : ∃ x y : ℚ, eq1 x y ∧ eq2 x y ∧ solution x y :=
by
  sorry

end solve_system_l1800_180060


namespace provisions_last_for_more_days_l1800_180015

def initial_men : ℕ := 2000
def initial_days : ℕ := 65
def additional_men : ℕ := 3000
def days_used : ℕ := 15
def remaining_provisions :=
  initial_men * initial_days - initial_men * days_used
def total_men_after_reinforcement := initial_men + additional_men
def remaining_days := remaining_provisions / total_men_after_reinforcement

theorem provisions_last_for_more_days :
  remaining_days = 20 := by
  sorry

end provisions_last_for_more_days_l1800_180015


namespace sqrt_25_eq_pm_five_l1800_180050

theorem sqrt_25_eq_pm_five (x : ℝ) : x^2 = 25 ↔ x = 5 ∨ x = -5 := 
sorry

end sqrt_25_eq_pm_five_l1800_180050


namespace max_g_eq_25_l1800_180071

-- Define the function g on positive integers.
def g : ℕ → ℤ
| n => if n < 12 then n + 14 else g (n - 7)

-- Prove that the maximum value of g is 25.
theorem max_g_eq_25 : ∀ n : ℕ, 1 ≤ n → g n ≤ 25 ∧ (∃ n : ℕ, 1 ≤ n ∧ g n = 25) := by
  sorry

end max_g_eq_25_l1800_180071


namespace evaluate_expression_l1800_180022

theorem evaluate_expression (a x : ℤ) (h : x = a + 7) : x - a + 3 = 10 := by
  sorry

end evaluate_expression_l1800_180022


namespace divisible_by_bn_l1800_180041

variables {u v a b : ℤ} {n : ℕ}

theorem divisible_by_bn 
  (h1 : ∀ x : ℤ, x^2 + a*x + b = 0 → x = u ∨ x = v)
  (h2 : a^2 % b = 0) 
  (h3 : ∀ m : ℕ, m = 2 * n) : 
  ∀ n : ℕ, (u^m + v^m) % (b^n) = 0 := 
  sorry

end divisible_by_bn_l1800_180041


namespace distinct_powers_exist_l1800_180001

theorem distinct_powers_exist :
  ∃ (a1 a2 b1 b2 c1 c2 d1 d2 : ℕ),
    (∃ n, a1 = n^2) ∧ (∃ m, a2 = m^2) ∧
    (∃ p, b1 = p^3) ∧ (∃ q, b2 = q^3) ∧
    (∃ r, c1 = r^5) ∧ (∃ s, c2 = s^5) ∧
    (∃ t, d1 = t^7) ∧ (∃ u, d2 = u^7) ∧
    a1 - a2 = b1 - b2 ∧ b1 - b2 = c1 - c2 ∧ c1 - c2 = d1 - d2 ∧
    a1 ≠ b1 ∧ a1 ≠ c1 ∧ a1 ≠ d1 ∧ b1 ≠ c1 ∧ b1 ≠ d1 ∧ c1 ≠ d1 := 
sorry

end distinct_powers_exist_l1800_180001


namespace xiamen_fabric_production_l1800_180083

theorem xiamen_fabric_production:
  (∃ x y : ℕ, (3 * ((2 * x) / 3) + 3 * (y / 3) = 600) ∧ (2 * ((2 * x) / 3) = 3 * (y / 3))) ∧
  (∀ x y : ℕ, (3 * ((2 * x) / 3) + 3 * (y / 3) = 600) ∧ (2 * ((2 * x) / 3) = 3 * (y / 3)) →
    x = 360 ∧ y = 240 ∧ y / 3 = 240) := 
by
  sorry

end xiamen_fabric_production_l1800_180083


namespace change_in_surface_area_zero_l1800_180013

-- Original rectangular solid dimensions
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

-- Smaller prism dimensions
structure SmallerPrism where
  length : ℝ
  width : ℝ
  height : ℝ

-- Conditions
def originalSolid : RectangularSolid := { length := 4, width := 3, height := 2 }
def removedPrism : SmallerPrism := { length := 1, width := 1, height := 2 }

-- Surface area calculation function
def surface_area (solid : RectangularSolid) : ℝ := 
  2 * (solid.length * solid.width + solid.length * solid.height + solid.width * solid.height)

-- Calculate the change in surface area
theorem change_in_surface_area_zero :
  let original_surface_area := surface_area originalSolid
  let removed_surface_area := (removedPrism.length * removedPrism.height)
  let new_exposed_area := (removedPrism.length * removedPrism.height)
  (original_surface_area - removed_surface_area + new_exposed_area) = original_surface_area :=
by
  sorry

end change_in_surface_area_zero_l1800_180013


namespace reflections_composition_rotation_l1800_180070

variable {α : ℝ} -- defining the angle α
variable {O : ℝ × ℝ} -- defining the point O, assuming the plane is represented as ℝ × ℝ

-- Define the lines that form the sides of the angle
variable (L1 L2 : ℝ × ℝ → Prop)

-- Assume α is the angle between L1 and L2 with O as the vertex
variable (hL1 : (L1 O))
variable (hL2 : (L2 O))

-- Assume reflections across L1 and L2
def reflect (P : ℝ × ℝ) (L : ℝ × ℝ → Prop) : ℝ × ℝ := sorry

theorem reflections_composition_rotation :
  ∀ A : ℝ × ℝ, (reflect (reflect A L1) L2) = sorry := 
sorry

end reflections_composition_rotation_l1800_180070


namespace coordinate_system_and_parametric_equations_l1800_180008

/-- Given the parametric equation of curve C1 is 
  x = 2 * cos φ and y = 3 * sin φ (where φ is the parameter)
  and a coordinate system with the origin as the pole and the positive half-axis of x as the polar axis.
  The polar equation of curve C2 is ρ = 2.
  The vertices of square ABCD are all on C2, arranged counterclockwise,
  with the polar coordinates of point A being (2, π/3).
  Find the Cartesian coordinates of A, B, C, and D, and prove that
  for any point P on C1, |PA|^2 + |PB|^2 + |PC|^2 + |PD|^2 is within the range [32, 52]. -/
theorem coordinate_system_and_parametric_equations
  (φ : ℝ)
  (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ)
  (P : ℝ → ℝ × ℝ)
  (A B C D : ℝ × ℝ)
  (t : ℝ)
  (H1 : ∀ φ, P φ = (2 * Real.cos φ, 3 * Real.sin φ))
  (H2 : A = (1, Real.sqrt 3) ∧ B = (-Real.sqrt 3, 1) ∧ C = (-1, -Real.sqrt 3) ∧ D = (Real.sqrt 3, -1))
  (H3 : ∀ p : ℝ × ℝ, ∃ φ, p = P φ)
  : ∀ x y, ∃ (φ : ℝ), P φ = (x, y) →
    ∃ t, t = |P φ - A|^2 + |P φ - B|^2 + |P φ - C|^2 + |P φ - D|^2 ∧ 32 ≤ t ∧ t ≤ 52 := 
sorry

end coordinate_system_and_parametric_equations_l1800_180008


namespace find_principal_sum_l1800_180017

noncomputable def principal_sum (P R : ℝ) : ℝ := P * (R + 6) / 100 - P * R / 100

theorem find_principal_sum (P R : ℝ) (h : P * (R + 6) / 100 - P * R / 100 = 30) : P = 500 :=
by sorry

end find_principal_sum_l1800_180017


namespace probability_of_adjacent_vertices_in_decagon_l1800_180093

/-- Define the number of vertices in the decagon -/
def num_vertices : ℕ := 10

/-- Define the total number of ways to choose two distinct vertices from the decagon -/
def total_possible_outcomes : ℕ := num_vertices * (num_vertices - 1) / 2

/-- Define the number of favorable outcomes where the two chosen vertices are adjacent -/
def favorable_outcomes : ℕ := num_vertices

/-- Define the probability of selecting two adjacent vertices -/
def probability_adjacent_vertices : ℚ := favorable_outcomes / total_possible_outcomes

/-- The main theorem statement -/
theorem probability_of_adjacent_vertices_in_decagon : probability_adjacent_vertices = 2 / 9 := 
  sorry

end probability_of_adjacent_vertices_in_decagon_l1800_180093


namespace initial_number_of_persons_l1800_180020

noncomputable def avg_weight_change : ℝ := 5.5
noncomputable def old_person_weight : ℝ := 68
noncomputable def new_person_weight : ℝ := 95.5
noncomputable def weight_diff : ℝ := new_person_weight - old_person_weight

theorem initial_number_of_persons (N : ℝ) 
  (h1 : avg_weight_change * N = weight_diff) : N = 5 :=
  by
  sorry

end initial_number_of_persons_l1800_180020


namespace exists_n_divisible_by_5_l1800_180049

theorem exists_n_divisible_by_5 
  (a b c d m : ℤ) 
  (h_div : a * m ^ 3 + b * m ^ 2 + c * m + d ≡ 0 [ZMOD 5]) 
  (h_d_nonzero : d ≠ 0) : 
  ∃ n : ℤ, d * n ^ 3 + c * n ^ 2 + b * n + a ≡ 0 [ZMOD 5] :=
sorry

end exists_n_divisible_by_5_l1800_180049


namespace jellybean_problem_l1800_180007

theorem jellybean_problem 
    (T L A : ℕ) 
    (h1 : T = L + 24) 
    (h2 : A = L / 2) 
    (h3 : T = 34) : 
    A = 5 := 
by 
  sorry

end jellybean_problem_l1800_180007


namespace units_digit_7_pow_5_l1800_180046

theorem units_digit_7_pow_5 : (7^5) % 10 = 7 := 
by
  sorry

end units_digit_7_pow_5_l1800_180046


namespace repeated_root_condition_l1800_180040

theorem repeated_root_condition (m : ℝ) : m = 10 → ∃ x, (5 * x) / (x - 2) + 1 = m / (x - 2) ∧ x = 2 :=
by
  sorry

end repeated_root_condition_l1800_180040


namespace trader_sold_meters_l1800_180002

variable (x : ℕ) (SP P CP : ℕ)

theorem trader_sold_meters (h_SP : SP = 660) (h_P : P = 5) (h_CP : CP = 5) : x = 66 :=
  by
  sorry

end trader_sold_meters_l1800_180002


namespace math_olympiad_scores_l1800_180051

theorem math_olympiad_scores (a : Fin 20 → ℕ) 
  (h_unique : ∀ i j, i ≠ j → a i ≠ a j)
  (h_sum : ∀ i j k : Fin 20, i ≠ j → j ≠ k → i ≠ k → a i < a j + a k) :
  ∀ i : Fin 20, a i > 18 := 
sorry

end math_olympiad_scores_l1800_180051


namespace determine_exponent_l1800_180056

-- Declare variables
variables {x y : ℝ}
variable {n : ℕ}

-- Use condition that the terms are like terms
theorem determine_exponent (h : - x ^ 2 * y ^ n = 3 * y * x ^ 2) : n = 1 :=
sorry

end determine_exponent_l1800_180056


namespace tallest_stack_is_b_l1800_180032

def number_of_pieces_a : ℕ := 8
def number_of_pieces_b : ℕ := 11
def number_of_pieces_c : ℕ := 6

def height_per_piece_a : ℝ := 2
def height_per_piece_b : ℝ := 1.5
def height_per_piece_c : ℝ := 2.5

def total_height_a : ℝ := number_of_pieces_a * height_per_piece_a
def total_height_b : ℝ := number_of_pieces_b * height_per_piece_b
def total_height_c : ℝ := number_of_pieces_c * height_per_piece_c

theorem tallest_stack_is_b : (total_height_b = 16.5) ∧ (total_height_b > total_height_a) ∧ (total_height_b > total_height_c) := 
by
  sorry

end tallest_stack_is_b_l1800_180032


namespace sales_tax_percentage_l1800_180023

theorem sales_tax_percentage (total_amount : ℝ) (tip_percentage : ℝ) (food_price : ℝ) (tax_percentage : ℝ) : 
  total_amount = 158.40 ∧ tip_percentage = 0.20 ∧ food_price = 120 → tax_percentage = 0.10 :=
by
  intros h
  sorry

end sales_tax_percentage_l1800_180023


namespace number_of_tiles_is_47_l1800_180009

theorem number_of_tiles_is_47 : 
  ∃ (n : ℕ), (n % 2 = 1) ∧ (n % 3 = 2) ∧ (n % 5 = 2) ∧ n = 47 :=
by
  sorry

end number_of_tiles_is_47_l1800_180009


namespace triangle_one_interior_angle_61_degrees_l1800_180045

theorem triangle_one_interior_angle_61_degrees
  (x : ℝ) : 
  (x + 75 + 2 * x + 25 + 3 * x - 22 = 360) → 
  (1 / 2 * (2 * x + 25) = 61 ∨ 
   1 / 2 * (3 * x - 22) = 61 ∨ 
   1 / 2 * (x + 75) = 61) :=
by
  intros h_sum
  sorry

end triangle_one_interior_angle_61_degrees_l1800_180045


namespace evaluate_polynomial_l1800_180082

theorem evaluate_polynomial : (99^4 - 4 * 99^3 + 6 * 99^2 - 4 * 99 + 1) = 92199816 := 
by 
  sorry

end evaluate_polynomial_l1800_180082


namespace deployment_plans_l1800_180016

/-- Given 6 volunteers and needing to select 4 to fill different positions of 
  translator, tour guide, shopping guide, and cleaner, and knowing that neither 
  supporters A nor B can work as the translator, the total number of deployment plans is 240. -/
theorem deployment_plans (volunteers : Fin 6) (A B : Fin 6) : 
  ∀ {translator tour_guide shopping_guide cleaner : Fin 6},
  A ≠ translator ∧ B ≠ translator → 
  ∃ plans : Finset (Fin 6 × Fin 6 × Fin 6 × Fin 6), plans.card = 240 :=
by 
sorry

end deployment_plans_l1800_180016


namespace correct_calculated_value_l1800_180085

theorem correct_calculated_value (n : ℕ) (h1 : n = 32 * 3) : n / 4 = 24 := 
by
  -- proof steps will be filled here
  sorry

end correct_calculated_value_l1800_180085


namespace factor_polynomial_l1800_180014

theorem factor_polynomial :
  ∃ (a b c d e f : ℤ), a < d ∧
    (a * x^2 + b * x + c) * (d * x^2 + e * x + f) = x^2 - 6 * x + 9 - 64 * x^4 ∧
    (a = -8 ∧ b = 1 ∧ c = -3 ∧ d = 8 ∧ e = 1 ∧ f = -3) := by
  sorry

end factor_polynomial_l1800_180014


namespace product_of_roots_of_cubic_polynomial_l1800_180036

theorem product_of_roots_of_cubic_polynomial :
  let a := 1
  let b := -15
  let c := 75
  let d := -50
  ∀ x : ℝ, (x^3 - 15*x^2 + 75*x - 50 = 0) →
  (x + 1) * (x + 1) * (x + 50) - 1 * (x + 50) = 0 → -d / a = 50 :=
by
  sorry

end product_of_roots_of_cubic_polynomial_l1800_180036


namespace compare_sqrt_l1800_180018

theorem compare_sqrt : 3 * Real.sqrt 2 > Real.sqrt 17 := by
  sorry

end compare_sqrt_l1800_180018


namespace number_of_operations_to_equal_l1800_180030

theorem number_of_operations_to_equal (a b : ℤ) (da db : ℤ) (initial_diff change_per_operation : ℤ) (n : ℤ) 
(h1 : a = 365) 
(h2 : b = 24) 
(h3 : da = 19) 
(h4 : db = 12) 
(h5 : initial_diff = a - b) 
(h6 : change_per_operation = da + db) 
(h7 : initial_diff = 341) 
(h8 : change_per_operation = 31) 
(h9 : initial_diff = change_per_operation * n) :
n = 11 := 
by
  sorry

end number_of_operations_to_equal_l1800_180030


namespace find_x_plus_y_l1800_180058

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.cos y = 2023) 
                           (h2 : x + 2023 * Real.sin y = 2022) 
                           (h3 : (Real.pi / 2) ≤ y ∧ y ≤ Real.pi) : 
  x + y = 2023 + Real.pi / 2 :=
sorry

end find_x_plus_y_l1800_180058


namespace sum_of_consecutive_ints_product_eq_336_l1800_180012

def consecutive_ints_sum (a b c : ℤ) : Prop :=
  b = a + 1 ∧ c = b + 1

theorem sum_of_consecutive_ints_product_eq_336 (a b c : ℤ) (h1 : consecutive_ints_sum a b c) (h2 : a * b * c = 336) :
  a + b + c = 21 :=
sorry

end sum_of_consecutive_ints_product_eq_336_l1800_180012


namespace work_completion_by_C_l1800_180067

theorem work_completion_by_C
  (A_work_rate : ℝ)
  (B_work_rate : ℝ)
  (C_work_rate : ℝ)
  (A_days_worked : ℝ)
  (B_days_worked : ℝ)
  (C_days_worked : ℝ)
  (A_total_days : ℝ)
  (B_total_days : ℝ)
  (C_completion_partial_work : ℝ)
  (H1 : A_work_rate = 1 / 40)
  (H2 : B_work_rate = 1 / 40)
  (H3 : A_days_worked = 10)
  (H4 : B_days_worked = 10)
  (H5 : C_days_worked = 10)
  (H6 : C_completion_partial_work = 1/2) :
  C_work_rate = 1 / 20 :=
by
  sorry

end work_completion_by_C_l1800_180067


namespace incorrect_average_calculated_initially_l1800_180059

theorem incorrect_average_calculated_initially 
    (S : ℕ) 
    (h1 : (S + 75) / 10 = 51) 
    (h2 : (S + 25) = a) 
    : a / 10 = 46 :=
by
  sorry

end incorrect_average_calculated_initially_l1800_180059


namespace inequality_for_pos_reals_l1800_180073

open Real Nat

theorem inequality_for_pos_reals
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (h : 1/a + 1/b = 1)
  (n : ℕ) :
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n + 1) :=
by 
  sorry

end inequality_for_pos_reals_l1800_180073
