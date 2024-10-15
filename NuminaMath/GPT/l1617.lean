import Mathlib

namespace NUMINAMATH_GPT_exists_real_ge_3_l1617_161740

-- Definition of the existential proposition
theorem exists_real_ge_3 : ∃ x : ℝ, x ≥ 3 :=
sorry

end NUMINAMATH_GPT_exists_real_ge_3_l1617_161740


namespace NUMINAMATH_GPT_james_birthday_stickers_l1617_161763

def initial_stickers : ℕ := 39
def final_stickers : ℕ := 61

def birthday_stickers (s_initial s_final : ℕ) : ℕ := s_final - s_initial

theorem james_birthday_stickers :
  birthday_stickers initial_stickers final_stickers = 22 := by
  sorry

end NUMINAMATH_GPT_james_birthday_stickers_l1617_161763


namespace NUMINAMATH_GPT_luke_bus_time_l1617_161735

theorem luke_bus_time
  (L : ℕ)   -- Luke's bus time to work in minutes
  (P : ℕ)   -- Paula's bus time to work in minutes
  (B : ℕ)   -- Luke's bike time home in minutes
  (h1 : P = 3 * L / 5) -- Paula's bus time is \( \frac{3}{5} \) of Luke's bus time
  (h2 : B = 5 * L)     -- Luke's bike time is 5 times his bus time
  (h3 : L + P + B + P = 504) -- Total travel time is 504 minutes
  : L = 70 := 
sorry

end NUMINAMATH_GPT_luke_bus_time_l1617_161735


namespace NUMINAMATH_GPT_am_gm_example_l1617_161790

open Real

theorem am_gm_example (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + 1)^3 / b + (b + 1)^3 / c + (c + 1)^3 / a ≥ 81 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_am_gm_example_l1617_161790


namespace NUMINAMATH_GPT_emily_toys_l1617_161786

theorem emily_toys (initial_toys sold_toys: Nat) (h₀ : initial_toys = 7) (h₁ : sold_toys = 3) : initial_toys - sold_toys = 4 := by
  sorry

end NUMINAMATH_GPT_emily_toys_l1617_161786


namespace NUMINAMATH_GPT_circle_center_sum_is_one_l1617_161754

def circle_center_sum (h k : ℝ) : Prop :=
  ∀ (x y : ℝ), (x^2 + y^2 + 4 * x - 6 * y = 3) → ((h = -2) ∧ (k = 3))

theorem circle_center_sum_is_one :
  ∀ h k : ℝ, circle_center_sum h k → h + k = 1 :=
by
  intros h k hc
  sorry

end NUMINAMATH_GPT_circle_center_sum_is_one_l1617_161754


namespace NUMINAMATH_GPT_ratio_B_A_l1617_161798

theorem ratio_B_A (A B : ℤ) (h : ∀ (x : ℝ), x ≠ -6 → x ≠ 0 → x ≠ 5 → 
  (A / (x + 6) + B / (x^2 - 5*x) = (x^3 - 3*x^2 + 12) / (x^3 + x^2 - 30*x))) :
  (B : ℚ) / A = 2.2 := by
  sorry

end NUMINAMATH_GPT_ratio_B_A_l1617_161798


namespace NUMINAMATH_GPT_victoria_more_scoops_l1617_161745

theorem victoria_more_scoops (Oli_scoops : ℕ) (Victoria_scoops : ℕ) 
  (hOli : Oli_scoops = 4) (hVictoria : Victoria_scoops = 2 * Oli_scoops) : 
  (Victoria_scoops - Oli_scoops) = 4 :=
by
  sorry

end NUMINAMATH_GPT_victoria_more_scoops_l1617_161745


namespace NUMINAMATH_GPT_wheel_speed_l1617_161778

theorem wheel_speed (s : ℝ) (t : ℝ) :
  (12 / 5280) * 3600 = s * t →
  (12 / 5280) * 3600 = (s + 4) * (t - (1 / 18000)) →
  s = 8 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_wheel_speed_l1617_161778


namespace NUMINAMATH_GPT_parabola_transformation_correct_l1617_161738

-- Definitions and conditions
def original_parabola (x : ℝ) : ℝ := 2 * x^2

def transformed_parabola (x : ℝ) : ℝ := 2 * (x + 3)^2 - 4

-- Theorem to prove that the above definition is correct
theorem parabola_transformation_correct : 
  ∀ x : ℝ, transformed_parabola x = 2 * (x + 3)^2 - 4 :=
by
  intros x
  rfl -- This uses the definition of 'transformed_parabola' directly

end NUMINAMATH_GPT_parabola_transformation_correct_l1617_161738


namespace NUMINAMATH_GPT_gcd_2835_9150_l1617_161732

theorem gcd_2835_9150 : Nat.gcd 2835 9150 = 15 := by
  sorry

end NUMINAMATH_GPT_gcd_2835_9150_l1617_161732


namespace NUMINAMATH_GPT_number_of_students_l1617_161753

-- Defining the parameters and conditions
def passing_score : ℕ := 65
def average_score_whole_class : ℕ := 66
def average_score_passed : ℕ := 71
def average_score_failed : ℕ := 56
def increased_score : ℕ := 5
def post_increase_average_passed : ℕ := 75
def post_increase_average_failed : ℕ := 59
def num_students_lb : ℕ := 15 
def num_students_ub : ℕ := 30

-- Lean statement to prove the number of students in the class
theorem number_of_students (x y n : ℕ) 
  (h1 : average_score_passed * x + average_score_failed * y = average_score_whole_class * (x + y))
  (h2 : (average_score_whole_class + increased_score) * (x + y) = post_increase_average_passed * (x + n) + post_increase_average_failed * (y - n))
  (h3 : num_students_lb < x + y ∧ x + y < num_students_ub)
  (h4 : x = 2 * y)
  (h5 : y = 4 * n) : x + y = 24 :=
sorry

end NUMINAMATH_GPT_number_of_students_l1617_161753


namespace NUMINAMATH_GPT_field_division_l1617_161797

theorem field_division (A B : ℝ) (h1 : A + B = 700) (h2 : B - A = (1 / 5) * ((A + B) / 2)) : A = 315 :=
by
  sorry

end NUMINAMATH_GPT_field_division_l1617_161797


namespace NUMINAMATH_GPT_planes_parallel_if_any_line_parallel_l1617_161736

-- Definitions for Lean statements:
variable (P1 P2 : Set Point)
variable (line : Set Point)

-- Conditions
def is_parallel_to_plane (line : Set Point) (plane : Set Point) : Prop := sorry

def is_parallel_plane (plane1 plane2 : Set Point) : Prop := sorry

-- Lean statement to be proved:
theorem planes_parallel_if_any_line_parallel (h : ∀ line, 
  line ⊆ P1 → is_parallel_to_plane line P2) : is_parallel_plane P1 P2 := sorry

end NUMINAMATH_GPT_planes_parallel_if_any_line_parallel_l1617_161736


namespace NUMINAMATH_GPT_find_pairs_l1617_161729

def is_solution_pair (m n : ℕ) : Prop :=
  Nat.lcm m n = 3 * m + 2 * n + 1

theorem find_pairs :
  { pairs : List (ℕ × ℕ) // ∀ (m n : ℕ), (m, n) ∈ pairs ↔ is_solution_pair m n } :=
by
  let pairs := [(3,10), (4,9)]
  have key : ∀ (m n : ℕ), (m, n) ∈ pairs ↔ is_solution_pair m n := sorry
  exact ⟨pairs, key⟩

end NUMINAMATH_GPT_find_pairs_l1617_161729


namespace NUMINAMATH_GPT_GCF_LCM_computation_l1617_161766

-- Definitions and axioms we need
def GCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

-- The theorem to prove
theorem GCF_LCM_computation : GCF (LCM 8 14) (LCM 7 12) = 28 :=
by sorry

end NUMINAMATH_GPT_GCF_LCM_computation_l1617_161766


namespace NUMINAMATH_GPT_spider_total_distance_l1617_161794

theorem spider_total_distance : 
  ∀ (pos1 pos2 pos3 : ℝ), pos1 = 3 → pos2 = -1 → pos3 = 8.5 → 
  |pos2 - pos1| + |pos3 - pos2| = 13.5 := 
by 
  intros pos1 pos2 pos3 hpos1 hpos2 hpos3 
  sorry

end NUMINAMATH_GPT_spider_total_distance_l1617_161794


namespace NUMINAMATH_GPT_car_average_speed_l1617_161714

-- Definitions based on conditions
def distance_first_hour : ℤ := 100
def distance_second_hour : ℤ := 60
def time_first_hour : ℤ := 1
def time_second_hour : ℤ := 1

-- Total distance and time calculations
def total_distance : ℤ := distance_first_hour + distance_second_hour
def total_time : ℤ := time_first_hour + time_second_hour

-- The average speed of the car
def average_speed : ℤ := total_distance / total_time

-- Proof statement
theorem car_average_speed : average_speed = 80 := by
  sorry

end NUMINAMATH_GPT_car_average_speed_l1617_161714


namespace NUMINAMATH_GPT_expression_equals_41_l1617_161758

theorem expression_equals_41 (x : ℝ) (h : 3*x^2 + 9*x + 5 ≠ 0) : 
  (3*x^2 + 9*x + 15) / (3*x^2 + 9*x + 5) = 41 :=
by
  sorry

end NUMINAMATH_GPT_expression_equals_41_l1617_161758


namespace NUMINAMATH_GPT_textbook_weight_difference_l1617_161725

variable (chemWeight : ℝ) (geomWeight : ℝ)

def chem_weight := chemWeight = 7.12
def geom_weight := geomWeight = 0.62

theorem textbook_weight_difference : chemWeight - geomWeight = 6.50 :=
by
  sorry

end NUMINAMATH_GPT_textbook_weight_difference_l1617_161725


namespace NUMINAMATH_GPT_paul_can_buy_toys_l1617_161701

-- Definitions of the given conditions
def initial_dollars : ℕ := 3
def allowance : ℕ := 7
def toy_cost : ℕ := 5

-- Required proof statement
theorem paul_can_buy_toys : (initial_dollars + allowance) / toy_cost = 2 := by
  sorry

end NUMINAMATH_GPT_paul_can_buy_toys_l1617_161701


namespace NUMINAMATH_GPT_divisibility_of_n_l1617_161793

def n : ℕ := (2^4 - 1) * (3^6 - 1) * (5^10 - 1) * (7^12 - 1)

theorem divisibility_of_n : 
    (5 ∣ n) ∧ (7 ∣ n) ∧ (11 ∣ n) ∧ (13 ∣ n) := 
by 
  sorry

end NUMINAMATH_GPT_divisibility_of_n_l1617_161793


namespace NUMINAMATH_GPT_evaluate_fraction_sum_l1617_161721

theorem evaluate_fraction_sum (a b c : ℝ) (h : a ≠ 40) (h_a : b ≠ 75) (h_b : c ≠ 85)
  (h_cond : (a / (40 - a)) + (b / (75 - b)) + (c / (85 - c)) = 8) :
  (8 / (40 - a)) + (15 / (75 - b)) + (17 / (85 - c)) = 40 := 
sorry

end NUMINAMATH_GPT_evaluate_fraction_sum_l1617_161721


namespace NUMINAMATH_GPT_hyperbola_range_k_l1617_161709

noncomputable def hyperbola_equation (x y k : ℝ) : Prop :=
    (x^2) / (|k|-2) + (y^2) / (5-k) = 1

theorem hyperbola_range_k (k : ℝ) :
    (∃ x y, hyperbola_equation x y k) → (k > 5 ∨ (-2 < k ∧ k < 2)) :=
by 
    sorry

end NUMINAMATH_GPT_hyperbola_range_k_l1617_161709


namespace NUMINAMATH_GPT_find_p_plus_q_l1617_161752

noncomputable def probability_only_one (factor : ℕ → Prop) : ℚ := 0.08 -- Condition 1
noncomputable def probability_exaclty_two (factor1 factor2 : ℕ → Prop) : ℚ := 0.12 -- Condition 2
noncomputable def probability_all_three_given_two (factor1 factor2 factor3 : ℕ → Prop) : ℚ := 1 / 4 -- Condition 3
def women_without_D_has_no_risk_factors (total_women women_with_D women_with_all_factors women_without_D_no_risk_factors : ℕ) : ℚ :=
  women_without_D_no_risk_factors / (total_women - women_with_D)

theorem find_p_plus_q : ∃ (p q : ℕ), (women_without_D_has_no_risk_factors 100 (8 + 2 * 12 + 4) 4 28 = p / q) ∧ (Nat.gcd p q = 1) ∧ p + q = 23 :=
by
  sorry

end NUMINAMATH_GPT_find_p_plus_q_l1617_161752


namespace NUMINAMATH_GPT_prob_of_2_digit_in_frac_1_over_7_l1617_161733

noncomputable def prob (n : ℕ) : ℚ := (3/2)^(n-1) / (3/2 - 1)

theorem prob_of_2_digit_in_frac_1_over_7 :
  let infinite_series_sum := ∑' n : ℕ, (2/3)^(6 * n + 3)
  ∑' (n : ℕ), prob (6 * n + 3) = 108 / 665 :=
by
  sorry

end NUMINAMATH_GPT_prob_of_2_digit_in_frac_1_over_7_l1617_161733


namespace NUMINAMATH_GPT_tan_alpha_eq_neg_four_thirds_l1617_161782

theorem tan_alpha_eq_neg_four_thirds
  (α : ℝ) (hα1 : 0 < α ∧ α < π) 
  (hα2 : Real.sin α + Real.cos α = 1 / 5) : 
  Real.tan α = - 4 / 3 := 
  sorry

end NUMINAMATH_GPT_tan_alpha_eq_neg_four_thirds_l1617_161782


namespace NUMINAMATH_GPT_solution_set_ineq_l1617_161702

noncomputable
def f (x : ℝ) : ℝ := sorry
noncomputable
def g (x : ℝ) : ℝ := sorry

axiom h_f_odd : ∀ x : ℝ, f (-x) = -f x
axiom h_g_even : ∀ x : ℝ, g (-x) = g x
axiom h_deriv_pos : ∀ x : ℝ, x < 0 → deriv f x * g x + f x * deriv g x > 0
axiom h_g_neg_three_zero : g (-3) = 0

theorem solution_set_ineq : { x : ℝ | f x * g x < 0 } = { x : ℝ | x < -3 } ∪ { x : ℝ | 0 < x ∧ x < 3 } := 
by sorry

end NUMINAMATH_GPT_solution_set_ineq_l1617_161702


namespace NUMINAMATH_GPT_algebraic_expression_value_l1617_161717

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 2 * x + 7 = 6) : 4 * x^2 + 8 * x - 5 = -9 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1617_161717


namespace NUMINAMATH_GPT_equal_spacing_between_paintings_l1617_161761

/--
Given:
- The width of each painting is 30 centimeters.
- The total width of the wall in the exhibition hall is 320 centimeters.
- There are six pieces of artwork.
Prove that: The distance between the end of the wall and the artwork, and between the artworks, is 20 centimeters.
-/
theorem equal_spacing_between_paintings :
  let width_painting := 30 -- in centimeters
  let total_wall_width := 320 -- in centimeters
  let num_paintings := 6
  let total_paintings_width := num_paintings * width_painting
  let remaining_space := total_wall_width - total_paintings_width
  let num_spaces := num_paintings + 1
  let space_between := remaining_space / num_spaces
  space_between = 20 := sorry

end NUMINAMATH_GPT_equal_spacing_between_paintings_l1617_161761


namespace NUMINAMATH_GPT_dogs_daily_food_total_l1617_161728

theorem dogs_daily_food_total :
  let first_dog_food := 0.125
  let second_dog_food := 0.25
  let third_dog_food := 0.375
  let fourth_dog_food := 0.5
  first_dog_food + second_dog_food + third_dog_food + fourth_dog_food = 1.25 :=
by
  sorry

end NUMINAMATH_GPT_dogs_daily_food_total_l1617_161728


namespace NUMINAMATH_GPT_find_multiple_l1617_161784

-- Given conditions
variables (P W m : ℕ)
variables (h1 : P * 24 = W) (h2 : m * P * 6 = W / 2)

-- The statement to prove
theorem find_multiple (P W m : ℕ) (h1 : P * 24 = W) (h2 : m * P * 6 = W / 2) : m = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_multiple_l1617_161784


namespace NUMINAMATH_GPT_hamburgers_total_l1617_161712

theorem hamburgers_total (initial_hamburgers : ℝ) (additional_hamburgers : ℝ) (h₁ : initial_hamburgers = 9.0) (h₂ : additional_hamburgers = 3.0) : initial_hamburgers + additional_hamburgers = 12.0 :=
by
  rw [h₁, h₂]
  norm_num

end NUMINAMATH_GPT_hamburgers_total_l1617_161712


namespace NUMINAMATH_GPT_inequality_holds_for_all_x_l1617_161749

theorem inequality_holds_for_all_x (m : ℝ) : 
  (∀ x : ℝ, (m * x^2 + 2 * m * x - 4 < 2 * x^2 + 4 * x)) ↔ -2 < m ∧ m ≤ 2 := 
by
  sorry

end NUMINAMATH_GPT_inequality_holds_for_all_x_l1617_161749


namespace NUMINAMATH_GPT_central_angle_of_cone_development_diagram_l1617_161785

-- Given conditions: radius of the base of the cone and slant height
def radius_base := 1
def slant_height := 3

-- Target theorem: prove the central angle of the lateral surface development diagram is 120 degrees
theorem central_angle_of_cone_development_diagram : 
  ∃ n : ℝ, (2 * π) = (n * π * slant_height) / 180 ∧ n = 120 :=
by
  use 120
  sorry

end NUMINAMATH_GPT_central_angle_of_cone_development_diagram_l1617_161785


namespace NUMINAMATH_GPT_johnny_closed_days_l1617_161703

theorem johnny_closed_days :
  let dishes_per_day := 40
  let pounds_per_dish := 1.5
  let price_per_pound := 8
  let weekly_expenditure := 1920
  let daily_pounds := dishes_per_day * pounds_per_dish
  let daily_cost := daily_pounds * price_per_pound
  let days_open := weekly_expenditure / daily_cost
  let days_in_week := 7
  let days_closed := days_in_week - days_open
  days_closed = 3 :=
by
  sorry

end NUMINAMATH_GPT_johnny_closed_days_l1617_161703


namespace NUMINAMATH_GPT_parabola_vertex_sum_l1617_161741

theorem parabola_vertex_sum (p q r : ℝ) 
  (h1 : ∃ (a b c : ℝ), ∀ (x : ℝ), a * x ^ 2 + b * x + c = y)
  (h2 : ∃ (vertex_x vertex_y : ℝ), vertex_x = 3 ∧ vertex_y = -1)
  (h3 : ∀ (x : ℝ), y = p * x ^ 2 + q * x + r)
  (h4 : y = p * (0 - 3) ^ 2 + r - 1)
  (h5 : y = 8)
  : p + q + r = 3 := 
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_sum_l1617_161741


namespace NUMINAMATH_GPT_image_of_element_2_l1617_161744

-- Define the mapping f and conditions
def f (x : ℕ) : ℕ := 2 * x + 1

-- Define the element and its image using f
def element_in_set_A : ℕ := 2
def image_in_set_B : ℕ := f element_in_set_A

-- The theorem to prove
theorem image_of_element_2 : image_in_set_B = 5 :=
by
  -- This is where the proof would go, but we omit it with sorry
  sorry

end NUMINAMATH_GPT_image_of_element_2_l1617_161744


namespace NUMINAMATH_GPT_distance_yolkino_palkino_l1617_161756

theorem distance_yolkino_palkino (d_1 d_2 : ℕ) (h : ∀ k : ℕ, d_1 + d_2 = 13) : 
  ∀ k : ℕ, d_1 + d_2 = 13 → (d_1 + d_2 = 13) :=
by
  sorry

end NUMINAMATH_GPT_distance_yolkino_palkino_l1617_161756


namespace NUMINAMATH_GPT_spider_distance_l1617_161780

/--
A spider crawls along a number line, starting at -3.
It crawls to -7, then turns around and crawls to 8.
--/
def spiderCrawl (start : ℤ) (point1 : ℤ) (point2 : ℤ): ℤ :=
  let dist1 := abs (point1 - start)
  let dist2 := abs (point2 - point1)
  dist1 + dist2

theorem spider_distance :
  spiderCrawl (-3) (-7) 8 = 19 :=
by
  sorry

end NUMINAMATH_GPT_spider_distance_l1617_161780


namespace NUMINAMATH_GPT_evaluate_expression_l1617_161711

theorem evaluate_expression : 5^2 + 15 / 3 - (3 * 2)^2 = -6 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1617_161711


namespace NUMINAMATH_GPT_vertex_of_parabola_l1617_161772

theorem vertex_of_parabola :
  ∃ h k : ℝ, (∀ x : ℝ, 3 * (x + 4)^2 - 9 = 3 * (x - h)^2 + k) ∧ (h, k) = (-4, -9) :=
by
  sorry

end NUMINAMATH_GPT_vertex_of_parabola_l1617_161772


namespace NUMINAMATH_GPT_tom_jerry_coffee_total_same_amount_total_coffee_l1617_161715

noncomputable def total_coffee_drunk (x : ℚ) : ℚ := 
  let jerry_coffee := 1.25 * x
  let tom_drinks := (2/3) * x
  let jerry_drinks := (2/3) * jerry_coffee
  let jerry_remainder := (5/12) * x
  let jerry_gives_tom := (5/48) * x + 3
  tom_drinks + jerry_gives_tom

theorem tom_jerry_coffee_total (x : ℚ) : total_coffee_drunk x = jerry_drinks + (1.25 * x - jerry_gives_tom) := sorry

theorem same_amount_total_coffee (x : ℚ) 
  (h : total_coffee_drunk x = (5/4) * x - ((5/48) * x + 3)) : 
  (1.25 * x + x = 36) :=
by sorry

end NUMINAMATH_GPT_tom_jerry_coffee_total_same_amount_total_coffee_l1617_161715


namespace NUMINAMATH_GPT_six_digit_divisibility_by_37_l1617_161724

theorem six_digit_divisibility_by_37 (a b c d e f : ℕ) (H : (100 * a + 10 * b + c + 100 * d + 10 * e + f) % 37 = 0) : 
  (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f) % 37 = 0 := 
sorry

end NUMINAMATH_GPT_six_digit_divisibility_by_37_l1617_161724


namespace NUMINAMATH_GPT_intersection_A_complement_B_l1617_161773

-- Definition of real numbers
def R := ℝ

-- Definitions of sets A and B
def A := {x : ℝ | x > 0}
def B := {x : ℝ | x^2 - x - 2 > 0}

-- Definition of the complement of B in R
def B_complement := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

-- The final statement we need to prove
theorem intersection_A_complement_B :
  A ∩ B_complement = {x : ℝ | 0 < x ∧ x ≤ 2} :=
sorry

end NUMINAMATH_GPT_intersection_A_complement_B_l1617_161773


namespace NUMINAMATH_GPT_youngest_child_age_l1617_161723

theorem youngest_child_age (x : ℕ) (h1 : Prime x)
  (h2 : Prime (x + 2))
  (h3 : Prime (x + 6))
  (h4 : Prime (x + 8))
  (h5 : Prime (x + 12))
  (h6 : Prime (x + 14)) :
  x = 5 := 
sorry

end NUMINAMATH_GPT_youngest_child_age_l1617_161723


namespace NUMINAMATH_GPT_number_of_men_in_second_group_l1617_161710

theorem number_of_men_in_second_group 
  (work : ℕ)
  (days_first_group days_second_group : ℕ)
  (men_first_group men_second_group : ℕ)
  (h1 : work = men_first_group * days_first_group)
  (h2 : work = men_second_group * days_second_group)
  (h3 : men_first_group = 20)
  (h4 : days_first_group = 30)
  (h5 : days_second_group = 24) :
  men_second_group = 25 :=
by
  sorry

end NUMINAMATH_GPT_number_of_men_in_second_group_l1617_161710


namespace NUMINAMATH_GPT_proof_abc_div_def_l1617_161706

def abc_div_def (a b c d e f : ℚ) : Prop := 
  a / b = 1 / 3 ∧ b / c = 2 ∧ c / d = 1 / 2 ∧ d / e = 3 ∧ e / f = 1 / 8 → (a * b * c) / (d * e * f) = 1 / 16

theorem proof_abc_div_def (a b c d e f : ℚ) :
  abc_div_def a b c d e f :=
by 
  sorry

end NUMINAMATH_GPT_proof_abc_div_def_l1617_161706


namespace NUMINAMATH_GPT_complement_of_A_in_I_l1617_161751

def I : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 6, 7}
def C_I_A : Set ℕ := {1, 3, 5}

theorem complement_of_A_in_I :
  (I \ A) = C_I_A := by
  sorry

end NUMINAMATH_GPT_complement_of_A_in_I_l1617_161751


namespace NUMINAMATH_GPT_h_at_3_l1617_161713

noncomputable def f (x : ℝ) := 3 * x + 4
noncomputable def g (x : ℝ) := Real.sqrt (f x) - 3
noncomputable def h (x : ℝ) := g (f x)

theorem h_at_3 : h 3 = Real.sqrt 43 - 3 := by
  sorry

end NUMINAMATH_GPT_h_at_3_l1617_161713


namespace NUMINAMATH_GPT_tom_age_ratio_l1617_161774

theorem tom_age_ratio (T N : ℕ) 
  (h1 : T = T)
  (h2 : T - N = 3 * (T - 5 * N)) : T / N = 7 :=
by sorry

end NUMINAMATH_GPT_tom_age_ratio_l1617_161774


namespace NUMINAMATH_GPT_fixed_rate_calculation_l1617_161727

theorem fixed_rate_calculation (f n : ℕ) (h1 : f + 4 * n = 220) (h2 : f + 7 * n = 370) : f = 20 :=
by
  sorry

end NUMINAMATH_GPT_fixed_rate_calculation_l1617_161727


namespace NUMINAMATH_GPT_evaluate_expression_l1617_161720

theorem evaluate_expression : 
  908 * 501 - (731 * 1389 - (547 * 236 + 842 * 731 - 495 * 361)) = 5448 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1617_161720


namespace NUMINAMATH_GPT_sum_three_times_integers_15_to_25_l1617_161704

noncomputable def sumArithmeticSequence (a d n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d)) / 2

theorem sum_three_times_integers_15_to_25 :
  let a := 15
  let d := 1
  let n := 25 - 15 + 1
  3 * sumArithmeticSequence a d n = 660 := by
  -- This part can be filled in with the actual proof
  sorry

end NUMINAMATH_GPT_sum_three_times_integers_15_to_25_l1617_161704


namespace NUMINAMATH_GPT_revenue_effect_l1617_161776

noncomputable def price_increase_factor : ℝ := 1.425
noncomputable def sales_decrease_factor : ℝ := 0.627

theorem revenue_effect (P Q R_new : ℝ) (h_price_increase : P ≠ 0) (h_sales_decrease : Q ≠ 0) :
  R_new = (P * price_increase_factor) * (Q * sales_decrease_factor) →
  ((R_new - P * Q) / (P * Q)) * 100 = -10.6825 :=
by
  sorry

end NUMINAMATH_GPT_revenue_effect_l1617_161776


namespace NUMINAMATH_GPT_probability_more_wins_than_losses_l1617_161781

theorem probability_more_wins_than_losses
  (n_matches : ℕ)
  (win_prob lose_prob tie_prob : ℚ)
  (h_sum_probs : win_prob + lose_prob + tie_prob = 1)
  (h_win_prob : win_prob = 1/3)
  (h_lose_prob : lose_prob = 1/3)
  (h_tie_prob : tie_prob = 1/3)
  (h_n_matches : n_matches = 8) :
  ∃ (m n : ℕ), Nat.gcd m n = 1 ∧ m / n = 5483 / 13122 ∧ (m + n) = 18605 :=
by
  sorry

end NUMINAMATH_GPT_probability_more_wins_than_losses_l1617_161781


namespace NUMINAMATH_GPT_candy_count_l1617_161757

theorem candy_count (initial_candy : ℕ) (eaten_candy : ℕ) (received_candy : ℕ) (final_candy : ℕ) :
  initial_candy = 33 → eaten_candy = 17 → received_candy = 19 → final_candy = 35 :=
by
  intros h_initial h_eaten h_received
  sorry

end NUMINAMATH_GPT_candy_count_l1617_161757


namespace NUMINAMATH_GPT_triangle_is_isosceles_right_l1617_161755

theorem triangle_is_isosceles_right (a b S : ℝ) (h : S = (1/4) * (a^2 + b^2)) :
  ∃ C : ℝ, C = 90 ∧ a = b :=
by
  sorry

end NUMINAMATH_GPT_triangle_is_isosceles_right_l1617_161755


namespace NUMINAMATH_GPT_total_area_correct_l1617_161768

noncomputable def radius : ℝ := 7
noncomputable def width : ℝ := 2 * radius
noncomputable def length : ℝ := 3 * width
noncomputable def rect_area : ℝ := length * width
noncomputable def square_side : ℝ := radius * Real.sqrt 2
noncomputable def square_area : ℝ := square_side ^ 2
noncomputable def total_area : ℝ := rect_area + square_area

theorem total_area_correct : total_area = 686 := 
by
  -- Definitions provided above represent the problem's conditions
  -- The value calculated manually is 686
  -- Proof steps skipped for initial statement creation
  sorry

end NUMINAMATH_GPT_total_area_correct_l1617_161768


namespace NUMINAMATH_GPT_parametric_line_eq_l1617_161707

-- Define the parameterized functions for x and y 
def parametric_x (t : ℝ) : ℝ := 3 * t + 7
def parametric_y (t : ℝ) : ℝ := 5 * t - 8

-- Define the equation of the line (here it's a relation that relates x and y)
def line_equation (x y : ℝ) : Prop := 
  y = (5 / 3) * x - (59 / 3)

theorem parametric_line_eq : 
  ∃ t : ℝ, line_equation (parametric_x t) (parametric_y t) := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_parametric_line_eq_l1617_161707


namespace NUMINAMATH_GPT_rectangle_area_error_l1617_161719

theorem rectangle_area_error
  (L W : ℝ)
  (measured_length : ℝ := 1.15 * L)
  (measured_width : ℝ := 1.20 * W)
  (true_area : ℝ := L * W)
  (measured_area : ℝ := measured_length * measured_width)
  (percentage_error : ℝ := ((measured_area - true_area) / true_area) * 100) :
  percentage_error = 38 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_error_l1617_161719


namespace NUMINAMATH_GPT_value_of_a_l1617_161722

theorem value_of_a (a : ℕ) (h1 : a * 9^3 = 3 * 15^5) (h2 : a = 5^5) : a = 3125 := by
  sorry

end NUMINAMATH_GPT_value_of_a_l1617_161722


namespace NUMINAMATH_GPT_find_a3_plus_a9_l1617_161734

variable (a : ℕ → ℝ)
variable (d : ℝ)
variable (n : ℕ)

-- Conditions stating sequence is arithmetic and a₁ + a₆ + a₁₁ = 3
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def a_1_6_11_sum (a : ℕ → ℝ) : Prop :=
  a 1 + a 6 + a 11 = 3

theorem find_a3_plus_a9 
  (h_arith : is_arithmetic_sequence a d)
  (h_sum : a_1_6_11_sum a) : 
  a 3 + a 9 = 2 := 
sorry

end NUMINAMATH_GPT_find_a3_plus_a9_l1617_161734


namespace NUMINAMATH_GPT_joan_total_socks_l1617_161743

theorem joan_total_socks (n : ℕ) (h1 : n / 3 = 60) : n = 180 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_joan_total_socks_l1617_161743


namespace NUMINAMATH_GPT_find_other_vertices_l1617_161783

theorem find_other_vertices
  (A : ℝ × ℝ) (B C : ℝ × ℝ)
  (S : ℝ × ℝ) (M : ℝ × ℝ)
  (hA : A = (7, 3))
  (hS : S = (5, -5 / 3))
  (hM : M = (3, -1))
  (h_centroid : S = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)) 
  (h_orthocenter : ∀ u v : ℝ × ℝ, u ≠ v → u - v = (4, 4) → (u - v) • (C - B) = 0) :
  B = (1, -1) ∧ C = (7, -7) :=
sorry

end NUMINAMATH_GPT_find_other_vertices_l1617_161783


namespace NUMINAMATH_GPT_cookies_total_is_60_l1617_161718

def Mona_cookies : ℕ := 20
def Jasmine_cookies : ℕ := Mona_cookies - 5
def Rachel_cookies : ℕ := Jasmine_cookies + 10
def Total_cookies : ℕ := Mona_cookies + Jasmine_cookies + Rachel_cookies

theorem cookies_total_is_60 : Total_cookies = 60 := by
  sorry

end NUMINAMATH_GPT_cookies_total_is_60_l1617_161718


namespace NUMINAMATH_GPT_infer_correct_l1617_161739

theorem infer_correct (a b c : ℝ) (h1: c < b) (h2: b < a) (h3: a + b + c = 0) :
  (c * b^2 ≤ ab^2) ∧ (ab > ac) :=
by
  sorry

end NUMINAMATH_GPT_infer_correct_l1617_161739


namespace NUMINAMATH_GPT_range_of_a_l1617_161777

theorem range_of_a (a x : ℝ) (h1 : 1 ≤ x ∧ x ≤ 3) (h2 : ∀ x, 1 ≤ x ∧ x ≤ 3 → |x - a| < 2) : 1 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1617_161777


namespace NUMINAMATH_GPT_rate_in_still_water_l1617_161775

-- Definitions of given conditions
def downstream_speed : ℝ := 26
def upstream_speed : ℝ := 12

-- The statement we need to prove
theorem rate_in_still_water : (downstream_speed + upstream_speed) / 2 = 19 := by
  sorry

end NUMINAMATH_GPT_rate_in_still_water_l1617_161775


namespace NUMINAMATH_GPT_wall_clock_time_at_car_5PM_l1617_161742

-- Define the initial known conditions
def initial_time : ℕ := 7 -- 7:00 AM
def wall_time_at_10AM : ℕ := 10 -- 10:00 AM
def car_time_at_10AM : ℕ := 11 -- 11:00 AM
def car_time_at_5PM : ℕ := 17 -- 5:00 PM = 17:00 in 24-hour format

-- Define the calculations for the rate of the car clock
def rate_of_car_clock : ℚ := (car_time_at_10AM - initial_time : ℚ) / (wall_time_at_10AM - initial_time : ℚ) -- rate = 4/3

-- Prove the actual time according to the wall clock when the car clock shows 5:00 PM
theorem wall_clock_time_at_car_5PM :
  let elapsed_real_time := (car_time_at_5PM - car_time_at_10AM) * (3 : ℚ) / (4 : ℚ)
  let actual_time := wall_time_at_10AM + elapsed_real_time
  (actual_time : ℚ) = 15 + (15 / 60 : ℚ) := -- 3:15 PM as 15.25 in 24-hour time
by
  sorry

end NUMINAMATH_GPT_wall_clock_time_at_car_5PM_l1617_161742


namespace NUMINAMATH_GPT_product_of_three_numbers_summing_to_eleven_l1617_161765

def numbers : List ℕ := [2, 3, 4, 6]

theorem product_of_three_numbers_summing_to_eleven : 
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ a + b + c = 11 ∧ a * b * c = 36 := 
by
  sorry

end NUMINAMATH_GPT_product_of_three_numbers_summing_to_eleven_l1617_161765


namespace NUMINAMATH_GPT_common_ratio_geometric_sequence_l1617_161788

theorem common_ratio_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) ∧ a 1 = 32 ∧ a 6 = -1 → q = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_geometric_sequence_l1617_161788


namespace NUMINAMATH_GPT_perimeter_C_l1617_161799

theorem perimeter_C : 
  ∀ {x y : ℕ}, 
    (6 * x + 2 * y = 56) → (4 * x + 6 * y = 56) → 
    (2 * x + 6 * y = 40) :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_perimeter_C_l1617_161799


namespace NUMINAMATH_GPT_unit_digit_7_power_2023_l1617_161762

theorem unit_digit_7_power_2023 : (7 ^ 2023) % 10 = 3 := by
  sorry

end NUMINAMATH_GPT_unit_digit_7_power_2023_l1617_161762


namespace NUMINAMATH_GPT_a_plus_b_eq_neg7_l1617_161708

theorem a_plus_b_eq_neg7 (a b : ℝ) :
  (∀ x : ℝ, (x^2 - 2 * x - 3 > 0) ∨ (x^2 + a * x + b ≤ 0)) ∧
  (∀ x : ℝ, (3 < x ∧ x ≤ 4) → ((x^2 - 2 * x - 3 > 0) ∧ (x^2 + a * x + b ≤ 0))) →
  a + b = -7 :=
by
  sorry

end NUMINAMATH_GPT_a_plus_b_eq_neg7_l1617_161708


namespace NUMINAMATH_GPT_inequality_holds_l1617_161746

variable {a b c : ℝ}

theorem inequality_holds (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) :=
sorry

end NUMINAMATH_GPT_inequality_holds_l1617_161746


namespace NUMINAMATH_GPT_harold_wrapping_paper_cost_l1617_161795

theorem harold_wrapping_paper_cost :
  let rolls_for_shirt_boxes := 20 / 5
  let rolls_for_xl_boxes := 12 / 3
  let total_rolls := rolls_for_shirt_boxes + rolls_for_xl_boxes
  let cost_per_roll := 4  -- dollars
  (total_rolls * cost_per_roll) = 32 := by
  sorry

end NUMINAMATH_GPT_harold_wrapping_paper_cost_l1617_161795


namespace NUMINAMATH_GPT_yura_finishes_on_correct_date_l1617_161787

-- Let there be 91 problems in the textbook.
-- Let Yura start solving problems on September 6th.
def total_problems : Nat := 91
def start_date : Nat := 6

-- Each morning, starting from September 7, he solves one problem less than the previous morning.
def problems_solved (n : Nat) : Nat := if n = 0 then 0 else problems_solved (n - 1) - 1

-- On the evening of September 8, there are 46 problems left to solve.
def remaining_problems_sept_8 : Nat := 46

-- The question is to find on which date Yura will finish solving the textbook
def finish_date : Nat := 12

theorem yura_finishes_on_correct_date : ∃ z : Nat, (problems_solved z * 3 = total_problems - remaining_problems_sept_8) ∧ (z = 15) ∧ finish_date = 12 := by sorry

end NUMINAMATH_GPT_yura_finishes_on_correct_date_l1617_161787


namespace NUMINAMATH_GPT_equal_roots_iff_k_eq_one_l1617_161737

theorem equal_roots_iff_k_eq_one (k : ℝ) : (∀ x : ℝ, 2 * k * x^2 + 4 * k * x + 2 = 0 → ∀ y : ℝ, 2 * k * y^2 + 4 * k * y + 2 = 0 → x = y) ↔ k = 1 := sorry

end NUMINAMATH_GPT_equal_roots_iff_k_eq_one_l1617_161737


namespace NUMINAMATH_GPT_distance_run_l1617_161748

theorem distance_run (D : ℝ) (A_time : ℝ) (B_time : ℝ) (A_beats_B : ℝ) : 
  A_time = 90 ∧ B_time = 180 ∧ A_beats_B = 2250 → D = 2250 :=
by
  sorry

end NUMINAMATH_GPT_distance_run_l1617_161748


namespace NUMINAMATH_GPT_function_decreasing_interval_l1617_161770

noncomputable def function_y (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

noncomputable def derivative_y' (x : ℝ) : ℝ := (x + 1) * (x - 1) / x

theorem function_decreasing_interval : ∀ x: ℝ, 0 < x ∧ x < 1 → (derivative_y' x < 0) := by
  sorry

end NUMINAMATH_GPT_function_decreasing_interval_l1617_161770


namespace NUMINAMATH_GPT_multiplication_expression_l1617_161705

theorem multiplication_expression : 45 * 27 + 18 * 45 = 2025 := by
  sorry

end NUMINAMATH_GPT_multiplication_expression_l1617_161705


namespace NUMINAMATH_GPT_derek_age_calculation_l1617_161760

theorem derek_age_calculation 
  (bob_age : ℕ)
  (evan_age : ℕ)
  (derek_age : ℕ) 
  (h1 : bob_age = 60)
  (h2 : evan_age = (2 * bob_age) / 3)
  (h3 : derek_age = evan_age - 10) : 
  derek_age = 30 :=
by
  -- The proof is to be filled in
  sorry

end NUMINAMATH_GPT_derek_age_calculation_l1617_161760


namespace NUMINAMATH_GPT_average_of_first_two_is_1_point_1_l1617_161730

theorem average_of_first_two_is_1_point_1
  (a1 a2 a3 a4 a5 a6 : ℝ) 
  (h1 : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = 2.5)
  (h2 : (a1 + a2) / 2 = x)
  (h3 : (a3 + a4) / 2 = 1.4)
  (h4 : (a5 + a6) / 2 = 5) :
  x = 1.1 := 
sorry

end NUMINAMATH_GPT_average_of_first_two_is_1_point_1_l1617_161730


namespace NUMINAMATH_GPT_largest_integer_divides_expression_l1617_161769

theorem largest_integer_divides_expression (x : ℤ) (h : Even x) :
  3 ∣ (10 * x + 1) * (10 * x + 5) * (5 * x + 3) :=
sorry

end NUMINAMATH_GPT_largest_integer_divides_expression_l1617_161769


namespace NUMINAMATH_GPT_symmetry_axis_is_2_range_of_a_l1617_161779

-- Definitions given in the conditions
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Condition 1: Constants a, b, c and a ≠ 0
variables (a b c : ℝ) (a_ne_zero : a ≠ 0)

-- Condition 2: Inequality constraint
axiom inequality_constraint : a^2 + 2 * a * c + c^2 < b^2

-- Condition 3: y-values are the same when x=t+2 and x=-t+2
axiom y_symmetry (t : ℝ) : quadratic_function a b c (t + 2) = quadratic_function a b c (-t + 2)

-- Question 1: Proving the symmetry axis is x=2
theorem symmetry_axis_is_2 : ∀ t : ℝ, (t + 2 + (-t + 2)) / 2 = 2 :=
by sorry

-- Question 2: Proving the range of a if y=2 when x=-2
theorem range_of_a (h : quadratic_function a b c (-2) = 2) (b_eq_neg4a : b = -4 * a) : 2 / 15 < a ∧ a < 2 / 7 :=
by sorry

end NUMINAMATH_GPT_symmetry_axis_is_2_range_of_a_l1617_161779


namespace NUMINAMATH_GPT_amount_needed_is_72_l1617_161726

-- Define the given conditions
def original_price : ℝ := 90
def discount_rate : ℝ := 20

-- The goal is to prove that the amount of money needed after the discount is $72
theorem amount_needed_is_72 (P : ℝ) (D : ℝ) (hP : P = original_price) (hD : D = discount_rate) : P - (D / 100 * P) = 72 := 
by sorry

end NUMINAMATH_GPT_amount_needed_is_72_l1617_161726


namespace NUMINAMATH_GPT_dewei_less_than_daliah_l1617_161791

theorem dewei_less_than_daliah
  (daliah_amount : ℝ := 17.5)
  (zane_amount : ℝ := 62)
  (zane_multiple_dewei : zane_amount = 4 * (zane_amount / 4)) :
  (daliah_amount - (zane_amount / 4)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_dewei_less_than_daliah_l1617_161791


namespace NUMINAMATH_GPT_find_k_if_equal_roots_l1617_161747

theorem find_k_if_equal_roots (a b k : ℚ) 
  (h1 : 2 * a + b = -4) 
  (h2 : 2 * a * b + a^2 = -60) 
  (h3 : -2 * a^2 * b = k)
  (h4 : a ≠ b)
  (h5 : k > 0) :
  k = 6400 / 27 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_k_if_equal_roots_l1617_161747


namespace NUMINAMATH_GPT_matrix_cubed_l1617_161796

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![2, -2], ![2, -1]]

theorem matrix_cubed :
  (A * A * A) = ![![ -4, 2], ![-2, 1]] :=
by
  sorry

end NUMINAMATH_GPT_matrix_cubed_l1617_161796


namespace NUMINAMATH_GPT_patient_treatment_volume_l1617_161731

noncomputable def total_treatment_volume : ℝ :=
  let drop_rate1 := 15     -- drops per minute for the first drip
  let ml_rate1 := 6 / 120  -- milliliters per drop for the first drip
  let drop_rate2 := 25     -- drops per minute for the second drip
  let ml_rate2 := 7.5 / 90 -- milliliters per drop for the second drip
  let total_time := 4 * 60 -- total minutes including breaks
  let break_time := 4 * 10 -- total break time in minutes
  let actual_time := total_time - break_time -- actual running time in minutes
  let total_drops1 := actual_time * drop_rate1
  let total_drops2 := actual_time * drop_rate2
  let volume1 := total_drops1 * ml_rate1
  let volume2 := total_drops2 * ml_rate2
  volume1 + volume2 -- total volume from both drips

theorem patient_treatment_volume : total_treatment_volume = 566.67 :=
  by
    -- Place the necessary calculation steps as assumptions or directly as one-liner
    sorry

end NUMINAMATH_GPT_patient_treatment_volume_l1617_161731


namespace NUMINAMATH_GPT_f_of_g_of_3_l1617_161750

def f (x : ℝ) : ℝ := 4 * x - 5
def g (x : ℝ) : ℝ := (x + 2)^2
theorem f_of_g_of_3 : f (g 3) = 95 := by
  sorry

end NUMINAMATH_GPT_f_of_g_of_3_l1617_161750


namespace NUMINAMATH_GPT_rectangle_semi_perimeter_l1617_161716

variables (BC AC AM x y : ℝ)

theorem rectangle_semi_perimeter (hBC : BC = 5) (hAC : AC = 12) (hAM : AM = x)
  (hMN_AC : ∀ (MN : ℝ), MN = 5 / 12 * AM)
  (hNP_BC : ∀ (NP : ℝ), NP = AC - AM)
  (hy_def : y = (5 / 12 * x) + (12 - x)) :
  y = (144 - 7 * x) / 12 :=
sorry

end NUMINAMATH_GPT_rectangle_semi_perimeter_l1617_161716


namespace NUMINAMATH_GPT_maria_total_flowers_l1617_161767

-- Define the initial conditions
def dozens := 3
def flowers_per_dozen := 12
def free_flowers_per_dozen := 2

-- Define the total number of flowers
def total_flowers := dozens * flowers_per_dozen + dozens * free_flowers_per_dozen

-- Assert the proof statement
theorem maria_total_flowers : total_flowers = 42 := sorry

end NUMINAMATH_GPT_maria_total_flowers_l1617_161767


namespace NUMINAMATH_GPT_Rachel_brought_25_cookies_l1617_161789

def Mona_cookies : ℕ := 20
def Jasmine_cookies : ℕ := Mona_cookies - 5
def Total_cookies : ℕ := 60

theorem Rachel_brought_25_cookies : (Total_cookies - (Mona_cookies + Jasmine_cookies) = 25) :=
by
  sorry

end NUMINAMATH_GPT_Rachel_brought_25_cookies_l1617_161789


namespace NUMINAMATH_GPT_principal_amount_is_26_l1617_161764

-- Define the conditions
def rate : Real := 0.07
def time : Real := 6
def simple_interest : Real := 10.92

-- Define the simple interest formula
def simple_interest_formula (P R T : Real) : Real := P * R * T

-- State the theorem to prove
theorem principal_amount_is_26 : 
  ∃ (P : Real), simple_interest_formula P rate time = simple_interest ∧ P = 26 :=
by
  sorry

end NUMINAMATH_GPT_principal_amount_is_26_l1617_161764


namespace NUMINAMATH_GPT_magician_weeks_worked_l1617_161759

theorem magician_weeks_worked
  (hourly_rate : ℕ)
  (hours_per_day : ℕ)
  (total_payment : ℕ)
  (days_per_week : ℕ)
  (h1 : hourly_rate = 60)
  (h2 : hours_per_day = 3)
  (h3 : total_payment = 2520)
  (h4 : days_per_week = 7) :
  total_payment / (hourly_rate * hours_per_day * days_per_week) = 2 := 
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_magician_weeks_worked_l1617_161759


namespace NUMINAMATH_GPT_cindy_correct_answer_l1617_161700

theorem cindy_correct_answer (x : ℝ) (h₀ : (x - 12) / 4 = 32) : (x - 7) / 5 = 27 :=
by
  sorry

end NUMINAMATH_GPT_cindy_correct_answer_l1617_161700


namespace NUMINAMATH_GPT_geometric_sequence_problem_l1617_161792

noncomputable def geometric_sequence_sum_condition 
  (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 1 + a 2 + a 3 + a 4 + a 5 = 6) ∧ 
  (a 1 ^ 2 + a 2 ^ 2 + a 3 ^ 2 + a 4 ^ 2 + a 5 ^ 2 = 18) ∧ 
  (∀ n, a n = a 1 * q ^ (n - 1)) ∧ 
  (q ≠ 1)

theorem geometric_sequence_problem 
  (a : ℕ → ℝ) (q : ℝ) 
  (h : geometric_sequence_sum_condition a q) : 
  a 1 - a 2 + a 3 - a 4 + a 5 = 3 := 
by 
  sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l1617_161792


namespace NUMINAMATH_GPT_triangle_equilateral_l1617_161771

noncomputable def is_equilateral {R p : ℝ} (A B C : ℝ) : Prop :=
  R * (Real.tan A + Real.tan B + Real.tan C) = 2 * p  →
  ∀ {a b c : ℝ}, a = b ∧ b = c ∧ c = a

theorem triangle_equilateral
  {A B C : ℝ}
  {R p : ℝ}
  (h : R * (Real.tan A + Real.tan B + Real.tan C) = 2 * p) :
  ∀ {a b c : ℝ}, a = b ∧ b = c ∧ c = a :=
sorry

end NUMINAMATH_GPT_triangle_equilateral_l1617_161771
