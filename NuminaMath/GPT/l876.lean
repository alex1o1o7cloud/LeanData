import Mathlib

namespace NUMINAMATH_GPT_number_of_zeros_at_end_of_factorial_30_l876_87667

-- Lean statement for the equivalence proof problem
def count_factors_of (p n : Nat) : Nat :=
  n / p + n / (p * p) + n / (p * p * p) + n / (p * p * p * p) + n / (p * p * p * p * p)

def zeros_at_end_of_factorial (n : Nat) : Nat :=
  count_factors_of 5 n

theorem number_of_zeros_at_end_of_factorial_30 : zeros_at_end_of_factorial 30 = 7 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_zeros_at_end_of_factorial_30_l876_87667


namespace NUMINAMATH_GPT_find_min_value_l876_87629

noncomputable def minValue (a b c : ℝ) : ℝ :=
  (3 / a) - (4 / b) + (5 / c)

theorem find_min_value (a b c : ℝ) (h1 : c > 0) (h2 : 4 * a^2 - 2 * a * b + 4 * b^2 = c) (h3 : ∀ x y : ℝ, |2 * a + b| ≥ |2 * x + y|) :
  minValue a b c = -2 :=
sorry

end NUMINAMATH_GPT_find_min_value_l876_87629


namespace NUMINAMATH_GPT_find_b_value_l876_87650

variable (a p q b : ℝ)
variable (h1 : p * 0 + q * (3 * a) + b * 1 = 1)
variable (h2 : p * (9 * a) + q * (-1) + b * 2 = 1)
variable (h3 : p * 0 + q * (3 * a) + b * 0 = 1)

theorem find_b_value : b = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_b_value_l876_87650


namespace NUMINAMATH_GPT_number_of_children_proof_l876_87611

-- Let A be the number of mushrooms Anya has
-- Let V be the number of mushrooms Vitya has
-- Let S be the number of mushrooms Sasha has
-- Let xs be the list of mushrooms of other children

def mushrooms_distribution (A V S : ℕ) (xs : List ℕ) : Prop :=
  let n := 3 + xs.length
  -- First condition
  let total_mushrooms := A + V + S + xs.sum
  let equal_share := total_mushrooms / n
  (A / 2 = equal_share) ∧ (V + A / 2 = equal_share) ∧ (S = equal_share) ∧
  (∀ x ∈ xs, x = equal_share) ∧
  -- Second condition
  (S + A = V + xs.sum)

theorem number_of_children_proof (A V S : ℕ) (xs : List ℕ) :
  mushrooms_distribution A V S xs → 3 + xs.length = 6 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_number_of_children_proof_l876_87611


namespace NUMINAMATH_GPT_find_cost_of_pencil_and_pen_l876_87676

variable (p q r : ℝ)

-- Definitions based on conditions
def condition1 := 3 * p + 2 * q + r = 4.20
def condition2 := p + 3 * q + 2 * r = 4.75
def condition3 := 2 * r = 3.00

-- The theorem to prove
theorem find_cost_of_pencil_and_pen (p q r : ℝ) (h1 : condition1 p q r) (h2 : condition2 p q r) (h3 : condition3 r) :
  p + q = 1.12 :=
by
  sorry

end NUMINAMATH_GPT_find_cost_of_pencil_and_pen_l876_87676


namespace NUMINAMATH_GPT_find_range_of_a_l876_87682

noncomputable def f (a x : ℝ) : ℝ := a / x - Real.exp (-x)

theorem find_range_of_a (p q a : ℝ) (h : 0 < a) (hpq : p < q) :
  (∀ x : ℝ, 0 < x → x ∈ Set.Icc p q → f a x ≤ 0) → 
  (0 < a ∧ a < 1 / Real.exp 1) :=
by
  sorry

end NUMINAMATH_GPT_find_range_of_a_l876_87682


namespace NUMINAMATH_GPT_find_values_of_x_and_y_l876_87617

theorem find_values_of_x_and_y (x y : ℝ) :
  (2.5 * x = y^2 + 43) ∧ (2.1 * x = y^2 - 12) → (x = 137.5 ∧ y = Real.sqrt 300.75) :=
by
  sorry

end NUMINAMATH_GPT_find_values_of_x_and_y_l876_87617


namespace NUMINAMATH_GPT_unique_sum_of_squares_power_of_two_l876_87677

theorem unique_sum_of_squares_power_of_two (n : ℕ) :
  ∃! (a b : ℕ), 2^n = a^2 + b^2 := 
sorry

end NUMINAMATH_GPT_unique_sum_of_squares_power_of_two_l876_87677


namespace NUMINAMATH_GPT_steps_climbed_l876_87691

-- Definitions
def flights : ℕ := 9
def feet_per_flight : ℕ := 10
def inches_per_step : ℕ := 18

-- Proving the number of steps John climbs up
theorem steps_climbed : 
  let feet_total := flights * feet_per_flight
  let inches_total := feet_total * 12
  let steps := inches_total / inches_per_step
  steps = 60 := 
by
  let feet_total := flights * feet_per_flight
  let inches_total := feet_total * 12
  let steps := inches_total / inches_per_step
  sorry

end NUMINAMATH_GPT_steps_climbed_l876_87691


namespace NUMINAMATH_GPT_eqn_y_value_l876_87618

theorem eqn_y_value (y : ℝ) (h : (2 / y) + ((3 / y) / (6 / y)) = 1.5) : y = 2 :=
sorry

end NUMINAMATH_GPT_eqn_y_value_l876_87618


namespace NUMINAMATH_GPT_m_leq_neg3_l876_87638

theorem m_leq_neg3 (m : ℝ) (h : ∀ x ∈ Set.Icc (0 : ℝ) 1, x^2 - 4 * x ≥ m) : m ≤ -3 := 
  sorry

end NUMINAMATH_GPT_m_leq_neg3_l876_87638


namespace NUMINAMATH_GPT_find_teaspoons_of_salt_l876_87641

def sodium_in_salt (S : ℕ) : ℕ := 50 * S
def sodium_in_parmesan (P : ℕ) : ℕ := 25 * P

-- Initial total sodium amount with 8 ounces of parmesan
def initial_total_sodium (S : ℕ) : ℕ := sodium_in_salt S + sodium_in_parmesan 8

-- Reduced sodium after removing 4 ounces of parmesan
def reduced_sodium (S : ℕ) : ℕ := initial_total_sodium S * 2 / 3

-- Reduced sodium with 4 fewer ounces of parmesan cheese
def new_total_sodium (S : ℕ) : ℕ := sodium_in_salt S + sodium_in_parmesan 4

theorem find_teaspoons_of_salt : ∃ (S : ℕ), reduced_sodium S = new_total_sodium S ∧ S = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_teaspoons_of_salt_l876_87641


namespace NUMINAMATH_GPT_tangent_slope_at_one_l876_87683

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem tangent_slope_at_one : deriv f 1 = 2 * Real.exp 1 := sorry

end NUMINAMATH_GPT_tangent_slope_at_one_l876_87683


namespace NUMINAMATH_GPT_largest_three_digit_integer_l876_87674

theorem largest_three_digit_integer (n : ℕ) :
  75 * n ≡ 300 [MOD 450] →
  n < 1000 →
  ∃ m : ℕ, n = m ∧ (∀ k : ℕ, 75 * k ≡ 300 [MOD 450] ∧ k < 1000 → k ≤ n) := by
  sorry

end NUMINAMATH_GPT_largest_three_digit_integer_l876_87674


namespace NUMINAMATH_GPT_quotient_when_divided_by_44_l876_87690

theorem quotient_when_divided_by_44 :
  ∃ N Q : ℕ, (N % 44 = 0) ∧ (N % 39 = 15) ∧ (N / 44 = Q) ∧ (Q = 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_quotient_when_divided_by_44_l876_87690


namespace NUMINAMATH_GPT_largest_divisor_of_m_l876_87652

theorem largest_divisor_of_m (m : ℕ) (h1 : m > 0) (h2 : 216 ∣ m^2) : 36 ∣ m :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_m_l876_87652


namespace NUMINAMATH_GPT_meaningful_fraction_l876_87628

theorem meaningful_fraction {a : ℝ} : 2 * a - 1 ≠ 0 ↔ a ≠ 1 / 2 :=
by sorry

end NUMINAMATH_GPT_meaningful_fraction_l876_87628


namespace NUMINAMATH_GPT_slope_of_line_in_terms_of_angle_l876_87606

variable {x y : ℝ}

theorem slope_of_line_in_terms_of_angle (h : 2 * Real.sqrt 3 * x - 2 * y - 1 = 0) :
    ∃ α : ℝ, 0 ≤ α ∧ α < Real.pi ∧ Real.tan α = Real.sqrt 3 ∧ α = Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_line_in_terms_of_angle_l876_87606


namespace NUMINAMATH_GPT_max_value_quadratic_l876_87627

noncomputable def quadratic (x : ℝ) : ℝ := -3 * (x - 2)^2 - 3

theorem max_value_quadratic : ∀ x : ℝ, quadratic x ≤ -3 ∧ (∀ y : ℝ, quadratic y = -3 → (∀ z : ℝ, quadratic z ≤ quadratic y)) :=
by
  sorry

end NUMINAMATH_GPT_max_value_quadratic_l876_87627


namespace NUMINAMATH_GPT_product_eval_l876_87635

theorem product_eval (a : ℝ) (h : a = 1) : (a - 3) * (a - 2) * (a - 1) * a * (a + 1) = 0 :=
by
  sorry

end NUMINAMATH_GPT_product_eval_l876_87635


namespace NUMINAMATH_GPT_milkshakes_more_than_ice_cream_cones_l876_87662

def ice_cream_cones_sold : ℕ := 67
def milkshakes_sold : ℕ := 82

theorem milkshakes_more_than_ice_cream_cones : milkshakes_sold - ice_cream_cones_sold = 15 := by
  sorry

end NUMINAMATH_GPT_milkshakes_more_than_ice_cream_cones_l876_87662


namespace NUMINAMATH_GPT_earrings_cost_l876_87680

theorem earrings_cost (initial_savings necklace_cost remaining_savings : ℕ) 
  (h_initial : initial_savings = 80) 
  (h_necklace : necklace_cost = 48) 
  (h_remaining : remaining_savings = 9) : 
  initial_savings - remaining_savings - necklace_cost = 23 := 
by {
  -- insert proof steps here -- 
  sorry
}

end NUMINAMATH_GPT_earrings_cost_l876_87680


namespace NUMINAMATH_GPT_cubical_storage_unit_blocks_l876_87698

theorem cubical_storage_unit_blocks :
  let side_length := 8
  let thickness := 1
  let total_volume := side_length ^ 3
  let interior_side_length := side_length - 2 * thickness
  let interior_volume := interior_side_length ^ 3
  let blocks_required := total_volume - interior_volume
  blocks_required = 296 := by
    sorry

end NUMINAMATH_GPT_cubical_storage_unit_blocks_l876_87698


namespace NUMINAMATH_GPT_find_number_l876_87665

theorem find_number (x : ℝ) (h : 0.3 * x + 0.1 * 0.5 = 0.29) : x = 0.8 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l876_87665


namespace NUMINAMATH_GPT_genevieve_coffee_drink_l876_87633

theorem genevieve_coffee_drink :
  let gallons := 4.5
  let small_thermos_count := 12
  let small_thermos_capacity_ml := 250
  let large_thermos_count := 6
  let large_thermos_capacity_ml := 500
  let genevieve_small_thermos_drink_count := 2
  let genevieve_large_thermos_drink_count := 1
  let ounces_per_gallon := 128
  let mls_per_ounce := 29.5735
  let total_mls := (gallons * ounces_per_gallon) * mls_per_ounce
  let genevieve_ml_drink := (genevieve_small_thermos_drink_count * small_thermos_capacity_ml) 
                            + (genevieve_large_thermos_drink_count * large_thermos_capacity_ml)
  let genevieve_ounces_drink := genevieve_ml_drink / mls_per_ounce
  genevieve_ounces_drink = 33.814 :=
by sorry

end NUMINAMATH_GPT_genevieve_coffee_drink_l876_87633


namespace NUMINAMATH_GPT_total_people_at_beach_l876_87622

-- Specifications of the conditions
def joined_people : ℕ := 100
def left_people : ℕ := 40
def family_count : ℕ := 3

-- Theorem stating the total number of people at the beach in the evening
theorem total_people_at_beach :
  joined_people - left_people + family_count = 63 := by
  sorry

end NUMINAMATH_GPT_total_people_at_beach_l876_87622


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_negativity_l876_87681

variable (b c : ℝ)

def f (x : ℝ) : ℝ := x^2 + b*x + c

theorem sufficient_but_not_necessary_condition_for_negativity (b c : ℝ) :
  (c < 0 → ∃ x : ℝ, f b c x < 0) ∧ (∃ b c : ℝ, ∃ x : ℝ, c ≥ 0 ∧ f b c x < 0) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_negativity_l876_87681


namespace NUMINAMATH_GPT_ned_weekly_revenue_l876_87624

-- Conditions
def normal_mouse_cost : ℕ := 120
def percentage_increase : ℕ := 30
def mice_sold_per_day : ℕ := 25
def days_store_is_open_per_week : ℕ := 4

-- Calculate cost of a left-handed mouse
def left_handed_mouse_cost : ℕ := normal_mouse_cost + (normal_mouse_cost * percentage_increase / 100)

-- Calculate daily revenue
def daily_revenue : ℕ := mice_sold_per_day * left_handed_mouse_cost

-- Calculate weekly revenue
def weekly_revenue : ℕ := daily_revenue * days_store_is_open_per_week

-- Theorem to prove
theorem ned_weekly_revenue : weekly_revenue = 15600 := 
by 
  sorry

end NUMINAMATH_GPT_ned_weekly_revenue_l876_87624


namespace NUMINAMATH_GPT_calculate_fg3_l876_87644

def g (x : ℕ) := x^3
def f (x : ℕ) := 3 * x - 2

theorem calculate_fg3 : f (g 3) = 79 :=
by
  sorry

end NUMINAMATH_GPT_calculate_fg3_l876_87644


namespace NUMINAMATH_GPT_white_space_area_is_31_l876_87631

-- Definitions and conditions from the problem
def board_width : ℕ := 4
def board_length : ℕ := 18
def board_area : ℕ := board_width * board_length

def area_C : ℕ := 4 + 2 + 2
def area_O : ℕ := (4 * 3) - (2 * 1)
def area_D : ℕ := (4 * 3) - (2 * 1)
def area_E : ℕ := 4 + 3 + 3 + 3

def total_black_area : ℕ := area_C + area_O + area_D + area_E

def white_space_area : ℕ := board_area - total_black_area

-- Proof problem statement
theorem white_space_area_is_31 : white_space_area = 31 := by
  sorry

end NUMINAMATH_GPT_white_space_area_is_31_l876_87631


namespace NUMINAMATH_GPT_certain_number_l876_87651

theorem certain_number (a n b : ℕ) (h1 : a = 30) (h2 : a * n = b^2) (h3 : ∀ m : ℕ, (m * n = b^2 → a ≤ m)) :
  n = 30 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_l876_87651


namespace NUMINAMATH_GPT_jack_jill_meeting_distance_l876_87673

-- Definitions for Jack's and Jill's initial conditions
def jack_speed_uphill := 12 -- km/hr
def jack_speed_downhill := 15 -- km/hr
def jill_speed_uphill := 14 -- km/hr
def jill_speed_downhill := 18 -- km/hr

def head_start := 0.2 -- hours
def total_distance := 12 -- km
def turn_point_distance := 7 -- km
def return_distance := 5 -- km

-- Statement of the problem to prove the distance from the turning point where they meet
theorem jack_jill_meeting_distance :
  let jack_time_to_turn := (turn_point_distance : ℚ) / jack_speed_uphill
  let jill_time_to_turn := (turn_point_distance : ℚ) / jill_speed_uphill
  let x_meet := (8.95 : ℚ) / 29
  7 - (14 * ((x_meet - 0.2) / 1)) = (772 / 145 : ℚ) := 
sorry

end NUMINAMATH_GPT_jack_jill_meeting_distance_l876_87673


namespace NUMINAMATH_GPT_people_left_is_10_l876_87678

def initial_people : ℕ := 12
def people_joined : ℕ := 15
def final_people : ℕ := 17
def people_left := initial_people - final_people + people_joined

theorem people_left_is_10 : people_left = 10 :=
by sorry

end NUMINAMATH_GPT_people_left_is_10_l876_87678


namespace NUMINAMATH_GPT_max_imaginary_part_of_root_l876_87699

theorem max_imaginary_part_of_root (z : ℂ) (h : z^6 - z^4 + z^2 - 1 = 0) (hne : z^2 ≠ 1) : 
  ∃ θ : ℝ, -90 ≤ θ ∧ θ ≤ 90 ∧ Complex.im z = Real.sin θ ∧ θ = 90 := 
sorry

end NUMINAMATH_GPT_max_imaginary_part_of_root_l876_87699


namespace NUMINAMATH_GPT_joint_purchases_popular_joint_purchases_unpopular_among_neighbors_l876_87640

section JointPurchases

/-- Given that joint purchases allow significant cost savings, reduced overhead costs,
improved quality assessment, and community trust, prove that joint purchases 
are popular in many countries despite the risks. -/
theorem joint_purchases_popular
    (cost_savings : Prop)
    (reduced_overhead_costs : Prop)
    (improved_quality_assessment : Prop)
    (community_trust : Prop)
    : Prop :=
    cost_savings ∧ reduced_overhead_costs ∧ improved_quality_assessment ∧ community_trust

/-- Given that high transaction costs, organizational difficulties,
convenience of proximity to stores, and potential disputes are challenges for neighbors,
prove that joint purchases of groceries and household goods are unpopular among neighbors. -/
theorem joint_purchases_unpopular_among_neighbors
    (high_transaction_costs : Prop)
    (organizational_difficulties : Prop)
    (convenience_proximity : Prop)
    (potential_disputes : Prop)
    : Prop :=
    high_transaction_costs ∧ organizational_difficulties ∧ convenience_proximity ∧ potential_disputes

end JointPurchases

end NUMINAMATH_GPT_joint_purchases_popular_joint_purchases_unpopular_among_neighbors_l876_87640


namespace NUMINAMATH_GPT_amount_spent_on_first_shop_l876_87636

-- Define the conditions
def booksFromFirstShop : ℕ := 65
def costFromSecondShop : ℕ := 2000
def booksFromSecondShop : ℕ := 35
def avgPricePerBook : ℕ := 85

-- Calculate the total books and the total amount spent
def totalBooks : ℕ := booksFromFirstShop + booksFromSecondShop
def totalAmountSpent : ℕ := totalBooks * avgPricePerBook

-- Prove the amount spent on the books from the first shop is Rs. 6500
theorem amount_spent_on_first_shop : 
  (totalAmountSpent - costFromSecondShop) = 6500 :=
by
  sorry

end NUMINAMATH_GPT_amount_spent_on_first_shop_l876_87636


namespace NUMINAMATH_GPT_fraction_difference_l876_87620

theorem fraction_difference (x y : ℝ) (h : x - y = 3 * x * y) : (1 / x) - (1 / y) = -3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_difference_l876_87620


namespace NUMINAMATH_GPT_min_focal_length_of_hyperbola_l876_87642

theorem min_focal_length_of_hyperbola (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_area : a * b = 8) :
  ∃ c ≥ 4, 2 * c = 8 :=
by sorry

end NUMINAMATH_GPT_min_focal_length_of_hyperbola_l876_87642


namespace NUMINAMATH_GPT_union_M_N_l876_87648

open Set Classical

noncomputable def M : Set ℝ := {x | x^2 = x}
noncomputable def N : Set ℝ := {x | Real.log x ≤ 0}

theorem union_M_N : M ∪ N = Icc 0 1 := by
  sorry

end NUMINAMATH_GPT_union_M_N_l876_87648


namespace NUMINAMATH_GPT_not_p_is_necessary_but_not_sufficient_l876_87603

-- Definitions based on the conditions
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) - a (n + 1) = d

def not_p (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∃ n : ℕ, a (n + 2) - a (n + 1) ≠ d

def not_q (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ¬ is_arithmetic_sequence a d

-- Proof problem statement
theorem not_p_is_necessary_but_not_sufficient (d : ℝ) (a : ℕ → ℝ) :
  (not_p a d → not_q a d) ∧ (not_q a d → not_p a d) = False := 
sorry

end NUMINAMATH_GPT_not_p_is_necessary_but_not_sufficient_l876_87603


namespace NUMINAMATH_GPT_determinant_value_l876_87672

-- Given definitions and conditions
def determinant (a b c d : ℤ) : ℤ := a * d - b * c
def special_determinant (m : ℤ) : ℤ := determinant (m^2) (m-3) (1-2*m) (m-2)

-- The proof problem
theorem determinant_value (m : ℤ) (h : m^2 - 2 * m - 3 = 0) : special_determinant m = 9 := sorry

end NUMINAMATH_GPT_determinant_value_l876_87672


namespace NUMINAMATH_GPT_tan_alpha_second_quadrant_l876_87688

theorem tan_alpha_second_quadrant (α : ℝ) 
(h_cos : Real.cos α = -4/5) 
(h_quadrant : π/2 < α ∧ α < π) : 
  Real.tan α = -3/4 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_second_quadrant_l876_87688


namespace NUMINAMATH_GPT_inverse_proportion_quadrants_l876_87610

theorem inverse_proportion_quadrants (a k : ℝ) (ha : a ≠ 0) (h : (3 * a, a) ∈ {p : ℝ × ℝ | p.snd = k / p.fst}) :
  k = 3 * a^2 ∧ k > 0 ∧
  (
    (∀ x y : ℝ, x > 0 → y = k / x → y > 0) ∨
    (∀ x y : ℝ, x < 0 → y = k / x → y < 0)
  ) :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_quadrants_l876_87610


namespace NUMINAMATH_GPT_height_difference_percentage_l876_87626

theorem height_difference_percentage (H_A H_B : ℝ) (h : H_B = H_A * 1.8181818181818183) :
  (H_A < H_B) → ((H_B - H_A) / H_B) * 100 = 45 := 
by 
  sorry

end NUMINAMATH_GPT_height_difference_percentage_l876_87626


namespace NUMINAMATH_GPT_ball_returns_to_Ben_after_three_throws_l876_87671

def circle_throw (n : ℕ) (skip : ℕ) (start : ℕ) : ℕ :=
  (start + skip) % n

theorem ball_returns_to_Ben_after_three_throws :
  ∀ (n : ℕ) (skip : ℕ) (start : ℕ),
  n = 15 → skip = 5 → start = 1 →
  (circle_throw n skip (circle_throw n skip (circle_throw n skip start))) = start :=
by
  intros n skip start hn hskip hstart
  sorry

end NUMINAMATH_GPT_ball_returns_to_Ben_after_three_throws_l876_87671


namespace NUMINAMATH_GPT_find_f_107_5_l876_87670

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : ∀ x, f x = f (-x)
axiom func_eq : ∀ x, f (x + 3) = - (1 / f x)
axiom cond_interval : ∀ x, -3 ≤ x ∧ x ≤ -2 → f x = 4 * x

theorem find_f_107_5 : f 107.5 = 1 / 10 := by {
  sorry
}

end NUMINAMATH_GPT_find_f_107_5_l876_87670


namespace NUMINAMATH_GPT_abs_inequality_interval_notation_l876_87686

variable (x : ℝ)

theorem abs_inequality_interval_notation :
  {x : ℝ | |x - 1| < 1} = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_abs_inequality_interval_notation_l876_87686


namespace NUMINAMATH_GPT_paul_number_proof_l876_87684

theorem paul_number_proof (a b : ℕ) (h₀ : 0 ≤ a ∧ a ≤ 9) (h₁ : 0 ≤ b ∧ b ≤ 9) (h₂ : a - b = 7) :
  (10 * a + b = 81) ∨ (10 * a + b = 92) :=
  sorry

end NUMINAMATH_GPT_paul_number_proof_l876_87684


namespace NUMINAMATH_GPT_dave_files_left_l876_87621

theorem dave_files_left 
  (initial_apps : ℕ) 
  (initial_files : ℕ) 
  (apps_left : ℕ)
  (files_more_than_apps : ℕ) 
  (h1 : initial_apps = 11) 
  (h2 : initial_files = 3) 
  (h3 : apps_left = 2)
  (h4 : files_more_than_apps = 22) 
  : ∃ (files_left : ℕ), files_left = apps_left + files_more_than_apps :=
by
  use 24
  sorry

end NUMINAMATH_GPT_dave_files_left_l876_87621


namespace NUMINAMATH_GPT_probability_after_50_bell_rings_l876_87668

noncomputable def game_probability : ℝ :=
  let p_keep_money := (1 : ℝ) / 4
  let p_give_money := (3 : ℝ) / 4
  let p_same_distribution := p_keep_money^3 + 2 * p_give_money^3
  p_same_distribution^50

theorem probability_after_50_bell_rings : abs (game_probability - 0.002) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_probability_after_50_bell_rings_l876_87668


namespace NUMINAMATH_GPT_x_over_y_l876_87613

theorem x_over_y (x y : ℝ) (h : 16 * x = 0.24 * 90 * y) : x / y = 1.35 :=
sorry

end NUMINAMATH_GPT_x_over_y_l876_87613


namespace NUMINAMATH_GPT_intercept_form_impossible_values_l876_87605

-- Define the problem statement
theorem intercept_form_impossible_values (m : ℝ) :
  (¬ (∃ a b c : ℝ, m ≠ 0 ∧ a * m = 0 ∧ b * m = 0 ∧ c * m = 1) ↔ (m = 4 ∨ m = -3 ∨ m = 5)) :=
sorry

end NUMINAMATH_GPT_intercept_form_impossible_values_l876_87605


namespace NUMINAMATH_GPT_haley_initial_shirts_l876_87687

-- Defining the conditions
def returned_shirts := 6
def endup_shirts := 5

-- The theorem statement
theorem haley_initial_shirts : returned_shirts + endup_shirts = 11 := by 
  sorry

end NUMINAMATH_GPT_haley_initial_shirts_l876_87687


namespace NUMINAMATH_GPT_inverse_sum_l876_87655

def g (x : ℝ) : ℝ := x^3

theorem inverse_sum : g⁻¹ 8 + g⁻¹ (-64) = -2 :=
by
  -- proof steps will go here
  sorry

end NUMINAMATH_GPT_inverse_sum_l876_87655


namespace NUMINAMATH_GPT_correct_statements_l876_87600

variables {d : ℝ} {S : ℕ → ℝ} {a : ℕ → ℝ}

axiom arithmetic_sequence (n : ℕ) : S n = n * a 1 + (n * (n - 1) / 2) * d

theorem correct_statements (h1 : S 6 = S 12) :
  (S 18 = 0) ∧ (d > 0 → a 6 + a 12 < 0) ∧ (d < 0 → |a 6| > |a 12|) :=
sorry

end NUMINAMATH_GPT_correct_statements_l876_87600


namespace NUMINAMATH_GPT_smallest_positive_q_with_property_l876_87649

theorem smallest_positive_q_with_property :
  ∃ q : ℕ, (
    q > 0 ∧
    ∀ m : ℕ, (1 ≤ m ∧ m ≤ 1006) →
    ∃ n : ℤ, 
      (m * q : ℤ) / 1007 < n ∧
      (m + 1) * q / 1008 > n) ∧
   q = 2015 := 
sorry

end NUMINAMATH_GPT_smallest_positive_q_with_property_l876_87649


namespace NUMINAMATH_GPT_none_of_these_true_l876_87689

def op_star (a b : ℕ) := b ^ a -- Define the binary operation

theorem none_of_these_true :
  ¬ (∀ a b : ℕ, 0 < a ∧ 0 < b → op_star a b = op_star b a) ∧
  ¬ (∀ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c → op_star a (op_star b c) = op_star (op_star a b) c) ∧
  ¬ (∀ a b n : ℕ, 0 < a ∧ 0 < b ∧ 0 < n → (op_star a b) ^ n = op_star n (op_star a b)) ∧
  ¬ (∀ a b n : ℕ, 0 < a ∧ 0 < b ∧ 0 < n → op_star a (b ^ n) = op_star n (op_star b a)) :=
sorry

end NUMINAMATH_GPT_none_of_these_true_l876_87689


namespace NUMINAMATH_GPT_bob_makes_weekly_profit_l876_87614

def weekly_profit (p_cost p_sell : ℝ) (m_daily d_week : ℕ) : ℝ :=
  (p_sell - p_cost) * m_daily * (d_week : ℝ)

theorem bob_makes_weekly_profit :
  weekly_profit 0.75 1.5 12 7 = 63 := 
by
  sorry

end NUMINAMATH_GPT_bob_makes_weekly_profit_l876_87614


namespace NUMINAMATH_GPT_gcd_pow_sub_l876_87645

theorem gcd_pow_sub (m n : ℕ) (h₁ : m = 2 ^ 2000 - 1) (h₂ : n = 2 ^ 1990 - 1) :
  Nat.gcd m n = 1023 :=
sorry

end NUMINAMATH_GPT_gcd_pow_sub_l876_87645


namespace NUMINAMATH_GPT_find_number_satisfying_equation_l876_87643

theorem find_number_satisfying_equation :
  ∃ x : ℝ, (196 * x^3) / 568 = 43.13380281690141 ∧ x = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_number_satisfying_equation_l876_87643


namespace NUMINAMATH_GPT_tan_theta_eq_sqrt_3_of_f_maximum_l876_87625

theorem tan_theta_eq_sqrt_3_of_f_maximum (θ : ℝ) 
  (h : ∀ x : ℝ, 3 * Real.sin (x + (Real.pi / 6)) ≤ 3 * Real.sin (θ + (Real.pi / 6))) : 
  Real.tan θ = Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_tan_theta_eq_sqrt_3_of_f_maximum_l876_87625


namespace NUMINAMATH_GPT_area_of_square_inscribed_in_circle_l876_87608

theorem area_of_square_inscribed_in_circle (a : ℝ) :
  ∃ S : ℝ, S = (2 * a^2) / 3 :=
sorry

end NUMINAMATH_GPT_area_of_square_inscribed_in_circle_l876_87608


namespace NUMINAMATH_GPT_average_fish_per_person_l876_87692

theorem average_fish_per_person (Aang Sokka Toph : ℕ) 
  (haang : Aang = 7) (hsokka : Sokka = 5) (htoph : Toph = 12) : 
  (Aang + Sokka + Toph) / 3 = 8 := by
  sorry

end NUMINAMATH_GPT_average_fish_per_person_l876_87692


namespace NUMINAMATH_GPT_verna_sherry_total_weight_l876_87653

theorem verna_sherry_total_weight (haley verna sherry : ℕ)
  (h1 : verna = haley + 17)
  (h2 : verna = sherry / 2)
  (h3 : haley = 103) :
  verna + sherry = 360 :=
by
  sorry

end NUMINAMATH_GPT_verna_sherry_total_weight_l876_87653


namespace NUMINAMATH_GPT_T_5_3_l876_87666

def T (x y : ℕ) : ℕ := 4 * x + 5 * y + x * y

theorem T_5_3 : T 5 3 = 50 :=
by
  sorry

end NUMINAMATH_GPT_T_5_3_l876_87666


namespace NUMINAMATH_GPT_sum_of_all_possible_values_of_M_l876_87669

-- Given conditions
-- M * (M - 8) = -7
-- We need to prove that the sum of all possible values of M is 8

theorem sum_of_all_possible_values_of_M : 
  ∃ M1 M2 : ℝ, (M1 * (M1 - 8) = -7) ∧ (M2 * (M2 - 8) = -7) ∧ (M1 + M2 = 8) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_all_possible_values_of_M_l876_87669


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l876_87659

theorem eccentricity_of_ellipse :
  let a : ℝ := 4
  let b : ℝ := 3
  let c : ℝ := Real.sqrt (a^2 - b^2)
  let e : ℝ := c / a
  e = Real.sqrt 7 / 4 :=
by
  sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l876_87659


namespace NUMINAMATH_GPT_simplify_evaluate_expression_l876_87612

theorem simplify_evaluate_expression (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (1 + 4 / (x - 3)) / ((x^2 + 2*x + 1) / (2*x - 6)) = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_evaluate_expression_l876_87612


namespace NUMINAMATH_GPT_range_of_a_l876_87695

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else Real.log x / Real.log a

theorem range_of_a {a : ℝ} (h : ∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) : 
  a > 1/7 ∧ a < 1/3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l876_87695


namespace NUMINAMATH_GPT_jimmy_income_l876_87658

theorem jimmy_income (r_income : ℕ) (r_increase : ℕ) (combined_percent : ℚ) (j_income : ℕ) : 
  r_income = 15000 → 
  r_increase = 7000 → 
  combined_percent = 0.55 → 
  (combined_percent * (r_income + r_increase + j_income) = r_income + r_increase) → 
  j_income = 18000 := 
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_jimmy_income_l876_87658


namespace NUMINAMATH_GPT_fraction_addition_l876_87697

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end NUMINAMATH_GPT_fraction_addition_l876_87697


namespace NUMINAMATH_GPT_total_revenue_correct_l876_87634

noncomputable def revenue_calculation : ℕ :=
  let fair_tickets := 60
  let fair_price := 15
  let baseball_tickets := fair_tickets / 3
  let baseball_price := 10
  let play_tickets := 2 * fair_tickets
  let play_price := 12
  fair_tickets * fair_price
  + baseball_tickets * baseball_price
  + play_tickets * play_price

theorem total_revenue_correct : revenue_calculation = 2540 :=
  by
  sorry

end NUMINAMATH_GPT_total_revenue_correct_l876_87634


namespace NUMINAMATH_GPT_doctor_lindsay_adult_patients_per_hour_l876_87607

def number_of_adult_patients_per_hour (A : ℕ) : Prop :=
  let children_per_hour := 3
  let cost_per_adult := 50
  let cost_per_child := 25
  let daily_income := 2200
  let hours_worked := 8
  let income_per_hour := daily_income / hours_worked
  let income_from_children_per_hour := children_per_hour * cost_per_child
  let income_from_adults_per_hour := A * cost_per_adult
  income_from_adults_per_hour + income_from_children_per_hour = income_per_hour

theorem doctor_lindsay_adult_patients_per_hour : 
  ∃ A : ℕ, number_of_adult_patients_per_hour A ∧ A = 4 :=
sorry

end NUMINAMATH_GPT_doctor_lindsay_adult_patients_per_hour_l876_87607


namespace NUMINAMATH_GPT_polar_to_rectangular_l876_87601

theorem polar_to_rectangular : 
  ∀ (r θ : ℝ), 
  r = 8 → 
  θ = 7 * Real.pi / 4 → 
  (r * Real.cos θ, r * Real.sin θ) = (4 * Real.sqrt 2, -4 * Real.sqrt 2) :=
by 
  intros r θ hr hθ
  rw [hr, hθ]
  sorry

end NUMINAMATH_GPT_polar_to_rectangular_l876_87601


namespace NUMINAMATH_GPT_greatest_multiple_of_5_and_6_less_than_800_l876_87616

theorem greatest_multiple_of_5_and_6_less_than_800 : 
  ∃ n : ℕ, n < 800 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 800 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n :=
sorry

end NUMINAMATH_GPT_greatest_multiple_of_5_and_6_less_than_800_l876_87616


namespace NUMINAMATH_GPT_same_terminal_side_l876_87693

theorem same_terminal_side (k : ℤ) : 
  {α | ∃ k : ℤ, α = k * 360 + (-263 : ℤ)} = 
  {α | ∃ k : ℤ, α = k * 360 - 263} := 
by sorry

end NUMINAMATH_GPT_same_terminal_side_l876_87693


namespace NUMINAMATH_GPT_find_base_length_of_isosceles_triangle_l876_87615

noncomputable def is_isosceles_triangle_with_base_len (a b : ℝ) : Prop :=
  a = 2 ∧ ((a + a + b = 5) ∨ (a + b + b = 5))

theorem find_base_length_of_isosceles_triangle :
  ∃ (b : ℝ), is_isosceles_triangle_with_base_len 2 b ∧ (b = 1.5 ∨ b = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_base_length_of_isosceles_triangle_l876_87615


namespace NUMINAMATH_GPT_ratio_of_rectangle_to_triangle_l876_87675

variable (L W : ℝ)

theorem ratio_of_rectangle_to_triangle (hL : L > 0) (hW : W > 0) : 
    L * W / (1/2 * L * W) = 2 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_rectangle_to_triangle_l876_87675


namespace NUMINAMATH_GPT_simplify_expression_l876_87664

variable (x y z : ℝ)

theorem simplify_expression (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hne : y - z / x ≠ 0) : 
  (x - z / y) / (y - z / x) = x / y := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l876_87664


namespace NUMINAMATH_GPT_sqrt_condition_l876_87604

theorem sqrt_condition (x : ℝ) : (3 * x - 5 ≥ 0) → (x ≥ 5 / 3) :=
by
  intros h
  have h1 : 3 * x ≥ 5 := by linarith
  have h2 : x ≥ 5 / 3 := by linarith
  exact h2

end NUMINAMATH_GPT_sqrt_condition_l876_87604


namespace NUMINAMATH_GPT_derivative_of_volume_is_surface_area_l876_87694

noncomputable def V_sphere (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3

theorem derivative_of_volume_is_surface_area (R : ℝ) (h : 0 < R) : 
  (deriv V_sphere R) = 4 * Real.pi * R^2 :=
by sorry

end NUMINAMATH_GPT_derivative_of_volume_is_surface_area_l876_87694


namespace NUMINAMATH_GPT_room_breadth_l876_87639

theorem room_breadth :
  ∀ (length breadth carpet_width cost_per_meter total_cost : ℝ),
  length = 15 →
  carpet_width = 75 / 100 →
  cost_per_meter = 30 / 100 →
  total_cost = 36 →
  total_cost = cost_per_meter * (total_cost / cost_per_meter) →
  length * breadth = (total_cost / cost_per_meter) * carpet_width →
  breadth = 6 :=
by
  intros length breadth carpet_width cost_per_meter total_cost
  intros h_length h_carpet_width h_cost_per_meter h_total_cost h_total_cost_eq h_area_eq
  sorry

end NUMINAMATH_GPT_room_breadth_l876_87639


namespace NUMINAMATH_GPT_convert_to_dms_l876_87685

-- Define the conversion factors
def degrees_to_minutes (d : ℝ) : ℝ := d * 60
def minutes_to_seconds (m : ℝ) : ℝ := m * 60

-- The main proof statement
theorem convert_to_dms (d : ℝ) :
  d = 24.29 →
  (24, 17, 24) = (24, degrees_to_minutes (0.29), minutes_to_seconds 0.4) :=
by
  sorry

end NUMINAMATH_GPT_convert_to_dms_l876_87685


namespace NUMINAMATH_GPT_find_z_when_x_is_1_l876_87679

-- We start by defining the conditions
variable (x y z : ℝ)
variable (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
variable (h_inv : ∃ k₁ : ℝ, ∀ x, x^2 * y = k₁)
variable (h_dir : ∃ k₂ : ℝ, ∀ y, y / z = k₂)
variable (h_y : y = 8) (h_z : z = 32) (h_x4 : x = 4)

-- Now we need to define the problem statement: 
-- proving that z = 512 when x = 1
theorem find_z_when_x_is_1 (h_x1 : x = 1) : z = 512 :=
  sorry

end NUMINAMATH_GPT_find_z_when_x_is_1_l876_87679


namespace NUMINAMATH_GPT_gcd_143_117_l876_87654

theorem gcd_143_117 : Nat.gcd 143 117 = 13 :=
by
  have h1 : 143 = 11 * 13 := by rfl
  have h2 : 117 = 9 * 13 := by rfl
  sorry

end NUMINAMATH_GPT_gcd_143_117_l876_87654


namespace NUMINAMATH_GPT_lex_apples_l876_87609

theorem lex_apples (A : ℕ) (h1 : A / 5 < 100) (h2 : A = (A / 5) + ((A / 5) + 9) + 42) : A = 85 :=
by
  sorry

end NUMINAMATH_GPT_lex_apples_l876_87609


namespace NUMINAMATH_GPT_original_rope_length_l876_87632

variable (S : ℕ) (L : ℕ)

-- Conditions
axiom shorter_piece_length : S = 20
axiom longer_piece_length : L = 2 * S

-- Prove that the original length of the rope is 60 meters
theorem original_rope_length : S + L = 60 :=
by
  -- proof steps will go here
  sorry

end NUMINAMATH_GPT_original_rope_length_l876_87632


namespace NUMINAMATH_GPT_all_suits_different_in_groups_of_four_l876_87637

-- Define the alternation pattern of the suits in the deck of 36 cards
def suits : List String := ["spades", "clubs", "hearts", "diamonds"]

-- Formalize the condition that each 4-card group in the deck contains all different suits
def suits_includes_all (cards : List String) : Prop :=
  ∀ i j, i < 4 → j < 4 → i ≠ j → cards.get? i ≠ cards.get? j

-- The main theorem statement
theorem all_suits_different_in_groups_of_four (L : List String)
  (hL : L.length = 36)
  (hA : ∀ n, n < 9 → L.get? (4*n) = some "spades" ∧ L.get? (4*n + 1) = some "clubs" ∧ L.get? (4*n + 2) = some "hearts" ∧ L.get? (4*n + 3) = some "diamonds"):
  ∀ cut reversed_deck, (@List.append String (List.reverse (List.take cut L)) (List.drop cut L) = reversed_deck)
  → ∀ n, n < 9 → suits_includes_all (List.drop (4*n) (List.take 4 reversed_deck)) := sorry

end NUMINAMATH_GPT_all_suits_different_in_groups_of_four_l876_87637


namespace NUMINAMATH_GPT_unique_plants_in_all_beds_l876_87696

theorem unique_plants_in_all_beds:
  let A := 600
  let B := 500
  let C := 400
  let D := 300
  let AB := 80
  let AC := 70
  let ABD := 40
  let BC := 0
  let AD := 0
  let BD := 0
  let CD := 0
  let ABC := 0
  let ACD := 0
  let BCD := 0
  let ABCD := 0
  A + B + C + D - AB - AC - BC - AD - BD - CD + ABC + ABD + ACD + BCD - ABCD = 1690 :=
by
  sorry

end NUMINAMATH_GPT_unique_plants_in_all_beds_l876_87696


namespace NUMINAMATH_GPT_cistern_filling_time_l876_87663

theorem cistern_filling_time :
  let rate_P := (1 : ℚ) / 12
  let rate_Q := (1 : ℚ) / 15
  let combined_rate := rate_P + rate_Q
  let time_combined := 6
  let filled_after_combined := combined_rate * time_combined
  let remaining_after_combined := 1 - filled_after_combined
  let time_Q := remaining_after_combined / rate_Q
  time_Q = 1.5 := sorry

end NUMINAMATH_GPT_cistern_filling_time_l876_87663


namespace NUMINAMATH_GPT_apple_cost_l876_87619

theorem apple_cost (cost_per_pound : ℚ) (weight : ℚ) (total_cost : ℚ) : cost_per_pound = 1 ∧ weight = 18 → total_cost = 18 :=
by
  sorry

end NUMINAMATH_GPT_apple_cost_l876_87619


namespace NUMINAMATH_GPT_is_positive_integer_iff_l876_87661

theorem is_positive_integer_iff (p : ℕ) : 
  (p > 0 → ∃ k : ℕ, (4 * p + 17 = k * (3 * p - 7))) ↔ (3 ≤ p ∧ p ≤ 40) := 
sorry

end NUMINAMATH_GPT_is_positive_integer_iff_l876_87661


namespace NUMINAMATH_GPT_solve_fraction_eq_l876_87660

theorem solve_fraction_eq (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ -2) :
  (2 / (x - 2) = 3 / (x + 2)) → x = 10 :=
by
  sorry

end NUMINAMATH_GPT_solve_fraction_eq_l876_87660


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l876_87602

theorem isosceles_triangle_base_length (a b : ℝ) (h : a = 4 ∧ b = 4) : a + b = 8 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l876_87602


namespace NUMINAMATH_GPT_ice_cream_scoops_l876_87646

theorem ice_cream_scoops (total_money : ℝ) (spent_on_restaurant : ℝ) (remaining_money : ℝ) 
  (cost_per_scoop_after_discount : ℝ) (remaining_each : ℝ) 
  (initial_savings : ℝ) (service_charge_percent : ℝ) (restaurant_percent : ℝ) 
  (ice_cream_discount_percent : ℝ) (money_each : ℝ) :
  total_money = 400 ∧
  spent_on_restaurant = 320 ∧
  remaining_money = 80 ∧
  cost_per_scoop_after_discount = 5 ∧
  remaining_each = 8 ∧
  initial_savings = 200 ∧
  service_charge_percent = 0.20 ∧
  restaurant_percent = 0.80 ∧
  ice_cream_discount_percent = 0.10 ∧
  money_each = 5 → 
  ∃ (scoops_per_person : ℕ), scoops_per_person = 5 :=
by
  sorry

end NUMINAMATH_GPT_ice_cream_scoops_l876_87646


namespace NUMINAMATH_GPT_least_number_to_produce_multiple_of_112_l876_87630

theorem least_number_to_produce_multiple_of_112 : ∃ k : ℕ, 72 * k = 112 * m → k = 14 :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_produce_multiple_of_112_l876_87630


namespace NUMINAMATH_GPT_long_furred_brown_dogs_l876_87623

theorem long_furred_brown_dogs :
  ∀ (T L B N LB : ℕ), T = 60 → L = 45 → B = 35 → N = 12 →
  (LB = L + B - (T - N)) → LB = 32 :=
by
  intros T L B N LB hT hL hB hN hLB
  sorry

end NUMINAMATH_GPT_long_furred_brown_dogs_l876_87623


namespace NUMINAMATH_GPT_minimum_y_value_l876_87647

noncomputable def minimum_y (x a : ℝ) : ℝ :=
  abs (x - a) + abs (x - 15) + abs (x - a - 15)

theorem minimum_y_value (a x : ℝ) (h1 : 0 < a) (h2 : a < 15) (h3 : a ≤ x) (h4 : x ≤ 15) :
  minimum_y x a = 15 :=
by
  sorry

end NUMINAMATH_GPT_minimum_y_value_l876_87647


namespace NUMINAMATH_GPT_not_both_zero_l876_87656

theorem not_both_zero (x y : ℝ) (h : x^2 + y^2 ≠ 0) : ¬ (x = 0 ∧ y = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_not_both_zero_l876_87656


namespace NUMINAMATH_GPT_scientific_notation_of_16907_l876_87657

theorem scientific_notation_of_16907 :
  16907 = 1.6907 * 10^4 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_16907_l876_87657
