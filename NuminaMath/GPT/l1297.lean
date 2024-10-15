import Mathlib

namespace NUMINAMATH_GPT_tangent_ellipse_hyperbola_l1297_129777

theorem tangent_ellipse_hyperbola (n : ℝ) :
  (∀ x y : ℝ, x^2 + 9 * y^2 = 9 ↔ x^2 - n * (y - 1)^2 = 4) →
  n = 9 / 5 :=
by sorry

end NUMINAMATH_GPT_tangent_ellipse_hyperbola_l1297_129777


namespace NUMINAMATH_GPT_evaluate_4_over_04_eq_400_l1297_129713

noncomputable def evaluate_fraction : Float :=
  (0.4)^4 / (0.04)^3

theorem evaluate_4_over_04_eq_400 : evaluate_fraction = 400 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_4_over_04_eq_400_l1297_129713


namespace NUMINAMATH_GPT_radius_range_of_circle_l1297_129723

theorem radius_range_of_circle (r : ℝ) :
  (∀ (x y : ℝ), (x - 3)^2 + (y + 5)^2 = r^2 → 
  (abs (4*x - 3*y - 2) = 1)) →
  4 < r ∧ r < 6 :=
by
  sorry

end NUMINAMATH_GPT_radius_range_of_circle_l1297_129723


namespace NUMINAMATH_GPT_ellipse_x_intercept_other_l1297_129711

noncomputable def foci : (ℝ × ℝ) × (ℝ × ℝ) := ((0, 3), (4, 0))
noncomputable def sum_of_distances : ℝ := 7
noncomputable def first_intercept : (ℝ × ℝ) := (0, 0)

theorem ellipse_x_intercept_other 
  (foci : (ℝ × ℝ) × (ℝ × ℝ))
  (sum_of_distances : ℝ)
  (first_intercept : (ℝ × ℝ))
  (hx : foci = ((0, 3), (4, 0)))
  (d_sum : sum_of_distances = 7)
  (intercept : first_intercept = (0, 0)) :
  ∃ (x : ℝ), x > 0 ∧ ((x, 0) = (56 / 11, 0)) := 
sorry

end NUMINAMATH_GPT_ellipse_x_intercept_other_l1297_129711


namespace NUMINAMATH_GPT_gage_skating_time_l1297_129717

theorem gage_skating_time :
  let gage_times_in_minutes1 := 1 * 60 + 15 -- 1 hour 15 minutes converted to minutes
  let gage_times_in_minutes2 := 2 * 60      -- 2 hours converted to minutes
  let total_skating_time_8_days := 5 * gage_times_in_minutes1 + 3 * gage_times_in_minutes2
  let required_total_time := 10 * 95       -- 10 days * 95 minutes per day
  required_total_time - total_skating_time_8_days = 215 :=
by
  sorry

end NUMINAMATH_GPT_gage_skating_time_l1297_129717


namespace NUMINAMATH_GPT_most_probable_standard_parts_in_batch_l1297_129760

theorem most_probable_standard_parts_in_batch :
  let q := 0.075
  let p := 1 - q
  let n := 39
  ∃ k₀ : ℤ, 36 ≤ k₀ ∧ k₀ ≤ 37 := 
by
  sorry

end NUMINAMATH_GPT_most_probable_standard_parts_in_batch_l1297_129760


namespace NUMINAMATH_GPT_sum_of_fourth_powers_l1297_129784

theorem sum_of_fourth_powers (a b c : ℝ) 
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 18.5 :=
sorry

end NUMINAMATH_GPT_sum_of_fourth_powers_l1297_129784


namespace NUMINAMATH_GPT_find_integer_l1297_129735

theorem find_integer (x : ℕ) (h : (4 * x) ^ 2 - 3 * x = 1764) : x = 18 := 
by 
  sorry

end NUMINAMATH_GPT_find_integer_l1297_129735


namespace NUMINAMATH_GPT_winner_percentage_l1297_129718

theorem winner_percentage (total_votes winner_votes : ℕ) (h1 : winner_votes = 744) (h2 : total_votes - winner_votes = 288) :
  (winner_votes : ℤ) * 100 / total_votes = 62 := 
by
  sorry

end NUMINAMATH_GPT_winner_percentage_l1297_129718


namespace NUMINAMATH_GPT_right_triangle_area_eq_8_over_3_l1297_129749

-- Definitions arising from the conditions in the problem
variable (a b c : ℝ)

-- The conditions as Lean definitions
def condition1 : Prop := b = (2/3) * a
def condition2 : Prop := b = (2/3) * c

-- The question translated into a proof problem: proving that the area of the triangle equals 8/3
theorem right_triangle_area_eq_8_over_3 (h1 : condition1 a b) (h2 : condition2 b c) (h3 : a^2 + b^2 = c^2) : 
  (1/2) * a * b = 8/3 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_area_eq_8_over_3_l1297_129749


namespace NUMINAMATH_GPT_park_attraction_children_count_l1297_129795

theorem park_attraction_children_count
  (C : ℕ) -- Number of children
  (entrance_fee : ℕ := 5) -- Entrance fee per person
  (kids_attr_fee : ℕ := 2) -- Attraction fee for kids
  (adults_attr_fee : ℕ := 4) -- Attraction fee for adults
  (parents : ℕ := 2) -- Number of parents
  (grandmother : ℕ := 1) -- Number of grandmothers
  (total_cost : ℕ := 55) -- Total cost paid
  (entry_eq : entrance_fee * (C + parents + grandmother) + kids_attr_fee * C + adults_attr_fee * (parents + grandmother) = total_cost) :
  C = 4 :=
by
  sorry

end NUMINAMATH_GPT_park_attraction_children_count_l1297_129795


namespace NUMINAMATH_GPT_total_travel_time_l1297_129753

-- We define the conditions as assumptions in the Lean 4 statement.

def riding_rate := 10  -- miles per hour
def time_first_part_minutes := 30  -- initial 30 minutes in minutes
def additional_distance_1 := 15  -- another 15 miles
def rest_time_minutes := 30  -- resting for 30 minutes
def remaining_distance := 20  -- remaining distance of 20 miles

theorem total_travel_time : 
    let time_first_part := time_first_part_minutes
    let time_second_part := (additional_distance_1 : ℚ) / riding_rate * 60  -- convert hours to minutes
    let time_third_part := rest_time_minutes
    let time_fourth_part := (remaining_distance : ℚ) / riding_rate * 60  -- convert hours to minutes
    time_first_part + time_second_part + time_third_part + time_fourth_part = 270 :=
by
  sorry

end NUMINAMATH_GPT_total_travel_time_l1297_129753


namespace NUMINAMATH_GPT_Carl_avg_gift_bags_l1297_129765

theorem Carl_avg_gift_bags :
  ∀ (known expected extravagant remaining : ℕ), 
  known = 50 →
  expected = 40 →
  extravagant = 10 →
  remaining = 60 →
  (known + expected) - extravagant - remaining = 30 := by
  intros
  sorry

end NUMINAMATH_GPT_Carl_avg_gift_bags_l1297_129765


namespace NUMINAMATH_GPT_find_m_if_even_l1297_129720

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def my_function (m : ℝ) (x : ℝ) : ℝ :=
  (m - 1) * x^2 + (m - 2) * x + (m^2 - 7 * m + 12)

theorem find_m_if_even (m : ℝ) :
  is_even_function (my_function m) → m = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_m_if_even_l1297_129720


namespace NUMINAMATH_GPT_evaluate_expr_l1297_129796

theorem evaluate_expr : 3 * (3 * (3 * (3 * (3 * (3 * 2 * 2) * 2) * 2) * 2) * 2) * 2 = 1458 := by
  sorry

end NUMINAMATH_GPT_evaluate_expr_l1297_129796


namespace NUMINAMATH_GPT_symmetric_about_pi_over_4_l1297_129712

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x + Real.cos x

theorem symmetric_about_pi_over_4 (a : ℝ) :
  (∀ x : ℝ, f a (x + π / 4) = f a (-(x + π / 4))) → a = 1 := by
  unfold f
  sorry

end NUMINAMATH_GPT_symmetric_about_pi_over_4_l1297_129712


namespace NUMINAMATH_GPT_circle_equation_passing_through_P_l1297_129752

-- Define the problem conditions
def P : ℝ × ℝ := (3, 1)
def l₁ (x y : ℝ) := x + 2 * y + 3 = 0
def l₂ (x y : ℝ) := x + 2 * y - 7 = 0

-- The main theorem statement
theorem circle_equation_passing_through_P :
  ∃ (α β : ℝ), 
    ((α = 4 ∧ β = -1) ∨ (α = 4 / 5 ∧ β = 3 / 5)) ∧ 
    ((x - α)^2 + (y - β)^2 = 5) :=
  sorry

end NUMINAMATH_GPT_circle_equation_passing_through_P_l1297_129752


namespace NUMINAMATH_GPT_value_of_f_neg2_l1297_129743

def f (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + 4 * x + c

theorem value_of_f_neg2 (a b c : ℝ) (h1 : f a b c 5 + f a b c (-5) = 6) (h2 : f a b c 2 = 8) :
  f a b c (-2) = -2 := by
  sorry

end NUMINAMATH_GPT_value_of_f_neg2_l1297_129743


namespace NUMINAMATH_GPT_rectangle_area_from_diagonal_l1297_129772

theorem rectangle_area_from_diagonal (x : ℝ) (w : ℝ) (h_lw : 3 * w = 3 * w) (h_diag : x^2 = 10 * w^2) : 
    (3 * w^2 = (3 / 10) * x^2) :=
by 
sorry

end NUMINAMATH_GPT_rectangle_area_from_diagonal_l1297_129772


namespace NUMINAMATH_GPT_total_tickets_l1297_129719

theorem total_tickets (tickets_first_day tickets_second_day tickets_third_day : ℕ) 
  (h1 : tickets_first_day = 5 * 4) 
  (h2 : tickets_second_day = 32)
  (h3 : tickets_third_day = 28) :
  tickets_first_day + tickets_second_day + tickets_third_day = 80 := by
  sorry

end NUMINAMATH_GPT_total_tickets_l1297_129719


namespace NUMINAMATH_GPT_max_blue_points_l1297_129778

-- We define the number of spheres and the categorization of red and green spheres
def number_of_spheres : ℕ := 2016

-- Definition of the number of red spheres
def red_spheres (r : ℕ) : Prop := r <= number_of_spheres

-- Definition of the number of green spheres as the complement of red spheres
def green_spheres (r : ℕ) : ℕ := number_of_spheres - r

-- Definition of the number of blue points as the intersection of red and green spheres
def blue_points (r : ℕ) : ℕ := r * green_spheres r

-- Theorem: Given the conditions, the maximum number of blue points is 1016064
theorem max_blue_points : ∃ r : ℕ, red_spheres r ∧ blue_points r = 1016064 := by
  sorry

end NUMINAMATH_GPT_max_blue_points_l1297_129778


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l1297_129728

theorem geometric_sequence_common_ratio (a_1 q : ℝ) (hne1 : q ≠ 1)
  (h : (a_1 * (1 - q^4) / (1 - q)) = 5 * (a_1 * (1 - q^2) / (1 - q))) :
  q = -1 ∨ q = 2 ∨ q = -2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l1297_129728


namespace NUMINAMATH_GPT_books_sold_in_february_l1297_129774

theorem books_sold_in_february (F : ℕ) 
  (h_avg : (15 + F + 17) / 3 = 16): 
  F = 16 := 
by 
  sorry

end NUMINAMATH_GPT_books_sold_in_february_l1297_129774


namespace NUMINAMATH_GPT_green_blue_tile_difference_is_15_l1297_129780

def initial_blue_tiles : Nat := 13
def initial_green_tiles : Nat := 6
def second_blue_tiles : Nat := 2 * initial_blue_tiles
def second_green_tiles : Nat := 2 * initial_green_tiles
def border_green_tiles : Nat := 36
def total_blue_tiles : Nat := initial_blue_tiles + second_blue_tiles
def total_green_tiles : Nat := initial_green_tiles + second_green_tiles + border_green_tiles
def tile_difference : Nat := total_green_tiles - total_blue_tiles

theorem green_blue_tile_difference_is_15 : tile_difference = 15 := by
  sorry

end NUMINAMATH_GPT_green_blue_tile_difference_is_15_l1297_129780


namespace NUMINAMATH_GPT_nested_inverse_value_l1297_129738

def f (x : ℝ) : ℝ := 5 * x + 6

noncomputable def f_inv (y : ℝ) : ℝ := (y - 6) / 5

theorem nested_inverse_value :
  f_inv (f_inv 16) = -4/5 :=
by
  sorry

end NUMINAMATH_GPT_nested_inverse_value_l1297_129738


namespace NUMINAMATH_GPT_compute_P_part_l1297_129755

noncomputable def P (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem compute_P_part (a b c d : ℝ) 
  (H1 : P 1 a b c d = 1993) 
  (H2 : P 2 a b c d = 3986) 
  (H3 : P 3 a b c d = 5979) : 
  (1 / 4) * (P 11 a b c d + P (-7) a b c d) = 4693 :=
by
  sorry

end NUMINAMATH_GPT_compute_P_part_l1297_129755


namespace NUMINAMATH_GPT_sum_of_squares_due_to_regression_eq_72_l1297_129768

theorem sum_of_squares_due_to_regression_eq_72
    (total_squared_deviations : ℝ)
    (correlation_coefficient : ℝ)
    (h1 : total_squared_deviations = 120)
    (h2 : correlation_coefficient = 0.6)
    : total_squared_deviations * correlation_coefficient^2 = 72 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sum_of_squares_due_to_regression_eq_72_l1297_129768


namespace NUMINAMATH_GPT_quadratic_common_root_l1297_129716

theorem quadratic_common_root (b : ℤ) :
  (∃ x, 2 * x^2 + (3 * b - 1) * x - 3 = 0 ∧ 6 * x^2 - (2 * b - 3) * x - 1 = 0) ↔ b = 2 := 
sorry

end NUMINAMATH_GPT_quadratic_common_root_l1297_129716


namespace NUMINAMATH_GPT_original_recipe_serves_7_l1297_129799

theorem original_recipe_serves_7 (x : ℕ)
  (h1 : 2 / x = 10 / 35) :
  x = 7 := by
  sorry

end NUMINAMATH_GPT_original_recipe_serves_7_l1297_129799


namespace NUMINAMATH_GPT_repeated_two_digit_number_divisible_by_101_l1297_129709

theorem repeated_two_digit_number_divisible_by_101 (a b : ℕ) :
  (10 ≤ a ∧ a ≤ 99 ∧ 0 ≤ b ∧ b ≤ 9) →
  ∃ k, (100000 * a + 10000 * b + 1000 * a + 100 * b + 10 * a + b) = 101 * k :=
by
  intro h
  sorry

end NUMINAMATH_GPT_repeated_two_digit_number_divisible_by_101_l1297_129709


namespace NUMINAMATH_GPT_books_received_l1297_129797

theorem books_received (initial_books : ℕ) (total_books : ℕ) (h1 : initial_books = 54) (h2 : total_books = 77) : (total_books - initial_books) = 23 :=
by
  sorry

end NUMINAMATH_GPT_books_received_l1297_129797


namespace NUMINAMATH_GPT_root_of_quadratic_property_l1297_129704

theorem root_of_quadratic_property (m : ℝ) (h : m^2 - 2 * m - 1 = 0) :
  m^2 + (1 / m^2) = 6 :=
sorry

end NUMINAMATH_GPT_root_of_quadratic_property_l1297_129704


namespace NUMINAMATH_GPT_coeffs_sum_eq_40_l1297_129766

theorem coeffs_sum_eq_40 (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) (x : ℝ)
  (h : (2 * x - 1) ^ 5 = a_0 * x ^ 5 + a_1 * x ^ 4 + a_2 * x ^ 3 + a_3 * x ^ 2 + a_4 * x + a_5) :
  a_2 + a_3 = 40 :=
sorry

end NUMINAMATH_GPT_coeffs_sum_eq_40_l1297_129766


namespace NUMINAMATH_GPT_multiple_of_960_l1297_129786

theorem multiple_of_960 (a : ℤ) (h1 : a % 10 = 4) (h2 : ¬ (a % 4 = 0)) :
  ∃ k : ℤ, a * (a^2 - 1) * (a^2 - 4) = 960 * k :=
  sorry

end NUMINAMATH_GPT_multiple_of_960_l1297_129786


namespace NUMINAMATH_GPT_largest_odd_digit_multiple_of_11_l1297_129792

theorem largest_odd_digit_multiple_of_11 (n : ℕ) (h1 : n < 10000) (h2 : ∀ d ∈ (n.digits 10), d % 2 = 1) (h3 : 11 ∣ n) : n ≤ 9559 :=
sorry

end NUMINAMATH_GPT_largest_odd_digit_multiple_of_11_l1297_129792


namespace NUMINAMATH_GPT_hide_and_seek_l1297_129790

theorem hide_and_seek
  (A B V G D : Prop)
  (h1 : A → (B ∧ ¬V))
  (h2 : B → (G ∨ D))
  (h3 : ¬V → (¬B ∧ ¬D))
  (h4 : ¬A → (B ∧ ¬G)) :
  (B ∧ V ∧ D) :=
by
  sorry

end NUMINAMATH_GPT_hide_and_seek_l1297_129790


namespace NUMINAMATH_GPT_intersection_M_N_l1297_129788

def M : Set ℕ := {3, 5, 6, 8}
def N : Set ℕ := {4, 5, 7, 8}

theorem intersection_M_N : M ∩ N = {5, 8} :=
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1297_129788


namespace NUMINAMATH_GPT_sum_mod_six_l1297_129705

theorem sum_mod_six (n : ℤ) : ((10 - 2 * n) + (4 * n + 2)) % 6 = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_mod_six_l1297_129705


namespace NUMINAMATH_GPT_gift_arrangement_l1297_129791

theorem gift_arrangement (n k : ℕ) (h_n : n = 5) (h_k : k = 4) : 
  (n * Nat.factorial k) = 120 :=
by
  sorry

end NUMINAMATH_GPT_gift_arrangement_l1297_129791


namespace NUMINAMATH_GPT_probability_of_gui_field_in_za_field_l1297_129737

noncomputable def area_gui_field (base height : ℕ) : ℚ :=
  (1 / 2 : ℚ) * base * height

noncomputable def area_za_field (small_base large_base height : ℕ) : ℚ :=
  (1 / 2 : ℚ) * (small_base + large_base) * height

theorem probability_of_gui_field_in_za_field :
  let b1 := 10
  let b2 := 20
  let h1 := 10
  let base_gui := 8
  let height_gui := 5
  let za_area := area_za_field b1 b2 h1
  let gui_area := area_gui_field base_gui height_gui
  (gui_area / za_area) = (2 / 15 : ℚ) := by
    sorry

end NUMINAMATH_GPT_probability_of_gui_field_in_za_field_l1297_129737


namespace NUMINAMATH_GPT_arithmetic_sequence_geometric_sum_l1297_129707

theorem arithmetic_sequence_geometric_sum (a1 : ℝ) (S : ℕ → ℝ)
  (h1 : ∀ (n : ℕ), S 1 = a1)
  (h2 : ∀ (n : ℕ), S 2 = 2 * a1 - 1)
  (h3 : ∀ (n : ℕ), S 4 = 4 * a1 - 6)
  (h4 : (2 * a1 - 1)^2 = a1 * (4 * a1 - 6)) 
  : a1 = -1/2 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_geometric_sum_l1297_129707


namespace NUMINAMATH_GPT_total_sum_vowels_l1297_129732

theorem total_sum_vowels :
  let A := 3
  let E := 5
  let I := 4
  let O := 2
  let U := 6
  A + E + I + O + U = 20 := by
  let A := 3
  let E := 5
  let I := 4
  let O := 2
  let U := 6
  sorry

end NUMINAMATH_GPT_total_sum_vowels_l1297_129732


namespace NUMINAMATH_GPT_parallelepiped_length_l1297_129741

theorem parallelepiped_length (n : ℕ) :
  (n - 2) * (n - 4) * (n - 6) = 2 * n * (n - 2) * (n - 4) / 3 →
  n = 18 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_parallelepiped_length_l1297_129741


namespace NUMINAMATH_GPT_sum_not_prime_30_l1297_129736

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_not_prime_30 (p1 p2 : ℕ) (hp1 : is_prime p1) (hp2 : is_prime p2) (h : p1 + p2 = 30) : false :=
sorry

end NUMINAMATH_GPT_sum_not_prime_30_l1297_129736


namespace NUMINAMATH_GPT_part_one_part_two_l1297_129715

-- Part 1:
-- Define the function f
def f (x : ℝ) : ℝ := abs (2 * x - 3) + abs (2 * x + 2)

-- Define the inequality problem
theorem part_one (x : ℝ) : f x < x + 5 ↔ 0 < x ∧ x < 2 :=
by sorry

-- Part 2:
-- Define the condition for part 2
theorem part_two (a : ℝ) : (∀ x : ℝ, f x > a + 4 / a) ↔ (a ∈ Set.Ioo 1 4 ∨ a < 0) :=
by sorry

end NUMINAMATH_GPT_part_one_part_two_l1297_129715


namespace NUMINAMATH_GPT_speed_of_sisters_sailboat_l1297_129710

variable (v_j : ℝ) (d : ℝ) (t_wait : ℝ)

-- Conditions
def janet_speed : Prop := v_j = 30
def lake_distance : Prop := d = 60
def janet_wait_time : Prop := t_wait = 3

-- Question to Prove
def sister_speed (v_s : ℝ) : Prop :=
  janet_speed v_j ∧ lake_distance d ∧ janet_wait_time t_wait →
  v_s = 12

-- The main theorem
theorem speed_of_sisters_sailboat (v_j d t_wait : ℝ) (h1 : janet_speed v_j) (h2 : lake_distance d) (h3 : janet_wait_time t_wait) :
  ∃ v_s : ℝ, sister_speed v_j d t_wait v_s :=
by
  sorry

end NUMINAMATH_GPT_speed_of_sisters_sailboat_l1297_129710


namespace NUMINAMATH_GPT_equivalent_fraction_l1297_129730

theorem equivalent_fraction :
  (6 + 6 + 6 + 6) / ((-2) * (-2) * (-2) * (-2)) = (4 * 6) / ((-2)^4) :=
by 
  sorry

end NUMINAMATH_GPT_equivalent_fraction_l1297_129730


namespace NUMINAMATH_GPT_product_complex_numbers_l1297_129721

noncomputable def Q : ℂ := 3 + 4 * Complex.I
noncomputable def E : ℂ := 2 * Complex.I
noncomputable def D : ℂ := 3 - 4 * Complex.I
noncomputable def R : ℝ := 2

theorem product_complex_numbers : Q * E * D * (R : ℂ) = 100 * Complex.I := by
  sorry

end NUMINAMATH_GPT_product_complex_numbers_l1297_129721


namespace NUMINAMATH_GPT_candy_bar_split_l1297_129767
noncomputable def split (total: ℝ) (people: ℝ): ℝ := total / people

theorem candy_bar_split: split 5.0 3.0 = 1.67 :=
by
  sorry

end NUMINAMATH_GPT_candy_bar_split_l1297_129767


namespace NUMINAMATH_GPT_product_of_abcd_l1297_129740

theorem product_of_abcd :
  ∃ (a b c d : ℚ), 
    3 * a + 4 * b + 6 * c + 8 * d = 42 ∧ 
    4 * (d + c) = b ∧ 
    4 * b + 2 * c = a ∧ 
    c - 2 = d ∧ 
    a * b * c * d = (367 * 76 * 93 * -55) / (37^2 * 74^2) :=
sorry

end NUMINAMATH_GPT_product_of_abcd_l1297_129740


namespace NUMINAMATH_GPT_sum_of_roots_l1297_129781

def quadratic_polynomial (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

def condition (a b c : ℝ) (x : ℝ) :=
  quadratic_polynomial a b c (x^3 + x) ≥ quadratic_polynomial a b c (x^2 + 1)

theorem sum_of_roots (a b c : ℝ) (h : ∀ x : ℝ, condition a b c x) :
  b = -4 * a → -(b / a) = 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l1297_129781


namespace NUMINAMATH_GPT_rectangle_perimeter_l1297_129739

theorem rectangle_perimeter 
  (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℕ)
  (relatively_prime : Nat.gcd (a_4 + a_7 + a_9) (a_2 + a_8 + a_6) = 1)
  (h1 : a_1 + a_2 = a_4)
  (h2 : a_1 + a_4 = a_5)
  (h3 : a_4 + a_5 = a_7)
  (h4 : a_5 + a_7 = a_9)
  (h5 : a_2 + a_4 + a_7 = a_8)
  (h6 : a_2 + a_8 = a_6)
  (h7 : a_1 + a_5 + a_9 = a_3)
  (h8 : a_3 + a_6 = a_8 + a_7) :
  2 * ((a_4 + a_7 + a_9) + (a_2 + a_8 + a_6)) = 164 := 
sorry -- proof omitted

end NUMINAMATH_GPT_rectangle_perimeter_l1297_129739


namespace NUMINAMATH_GPT_find_values_l1297_129701

theorem find_values (x y : ℤ) 
  (h1 : x / 5 + 7 = y / 4 - 7)
  (h2 : x / 3 - 4 = y / 2 + 4) : 
  x = -660 ∧ y = -472 :=
by 
  sorry

end NUMINAMATH_GPT_find_values_l1297_129701


namespace NUMINAMATH_GPT_train_length_proof_l1297_129744

-- Definitions for conditions
def jogger_speed_kmh : ℕ := 9
def train_speed_kmh : ℕ := 45
def initial_distance_ahead_m : ℕ := 280
def time_to_pass_s : ℕ := 40

-- Conversion factors
def km_per_hr_to_m_per_s (speed_kmh : ℕ) : ℕ := speed_kmh * 1000 / 3600

-- Converted speeds
def jogger_speed_m_per_s : ℕ := km_per_hr_to_m_per_s jogger_speed_kmh
def train_speed_m_per_s : ℕ := km_per_hr_to_m_per_s train_speed_kmh

-- Relative speed
def relative_speed_m_per_s : ℕ := train_speed_m_per_s - jogger_speed_m_per_s

-- Distance covered relative to the jogger
def distance_covered_relative_m : ℕ := relative_speed_m_per_s * time_to_pass_s

-- Length of the train
def length_of_train_m : ℕ := distance_covered_relative_m + initial_distance_ahead_m

-- Theorem to prove 
theorem train_length_proof : length_of_train_m = 680 := 
by
   sorry

end NUMINAMATH_GPT_train_length_proof_l1297_129744


namespace NUMINAMATH_GPT_divided_number_l1297_129758

theorem divided_number (x y : ℕ) (h1 : 7 * x + 5 * y = 146) (h2 : y = 11) : x + y = 24 :=
sorry

end NUMINAMATH_GPT_divided_number_l1297_129758


namespace NUMINAMATH_GPT_cookies_per_tray_l1297_129773

def num_trays : ℕ := 4
def num_packs : ℕ := 8
def cookies_per_pack : ℕ := 12
def total_cookies : ℕ := num_packs * cookies_per_pack

theorem cookies_per_tray : total_cookies / num_trays = 24 := by
  sorry

end NUMINAMATH_GPT_cookies_per_tray_l1297_129773


namespace NUMINAMATH_GPT_mn_eq_one_l1297_129731

noncomputable def f (x : ℝ) : ℝ := |Real.log x / Real.log 2|

variables (m n : ℝ) (hmn : m < n) (hm_pos : 0 < m) (hn_pos : 0 < n) (hmn_equal : f m = f n)

theorem mn_eq_one : m * n = 1 := by
  sorry

end NUMINAMATH_GPT_mn_eq_one_l1297_129731


namespace NUMINAMATH_GPT_perfect_square_pattern_l1297_129756

theorem perfect_square_pattern {a b n : ℕ} (h₁ : 1000 ≤ n ∧ n ≤ 9999)
  (h₂ : n = (10 * a + b) ^ 2)
  (h₃ : n = 1100 * a + 11 * b) : n = 7744 :=
  sorry

end NUMINAMATH_GPT_perfect_square_pattern_l1297_129756


namespace NUMINAMATH_GPT_hall_length_width_difference_l1297_129798

theorem hall_length_width_difference :
  ∃ (L W : ℝ), 
  (W = (1 / 2) * L) ∧
  (L * W = 288) ∧
  (L - W = 12) :=
by
  -- The mathematical proof follows from the conditions given
  sorry

end NUMINAMATH_GPT_hall_length_width_difference_l1297_129798


namespace NUMINAMATH_GPT_exponential_increasing_l1297_129750

theorem exponential_increasing (a : ℝ) (h_pos : a > 0) (h_neq_one : a ≠ 1) :
  (∀ x y : ℝ, x < y → a^x < a^y) ↔ a > 1 :=
by
  sorry

end NUMINAMATH_GPT_exponential_increasing_l1297_129750


namespace NUMINAMATH_GPT_theta_third_quadrant_l1297_129751

theorem theta_third_quadrant (θ : ℝ) (h1 : Real.sin θ < 0) (h2 : Real.tan θ > 0) : 
  π < θ ∧ θ < 3 * π / 2 :=
by 
  sorry

end NUMINAMATH_GPT_theta_third_quadrant_l1297_129751


namespace NUMINAMATH_GPT_sufficient_not_necessary_a_eq_one_l1297_129764

noncomputable def f (a x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + a^2) - x)

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x + f (-x) = 0

theorem sufficient_not_necessary_a_eq_one 
  (a : ℝ) 
  (h₁ : a = 1) 
  : is_odd_function (f a) := sorry

end NUMINAMATH_GPT_sufficient_not_necessary_a_eq_one_l1297_129764


namespace NUMINAMATH_GPT_smallest_k_for_divisibility_l1297_129775

theorem smallest_k_for_divisibility : (∃ k : ℕ, ∀ z : ℂ, z^8 + z^7 + z^4 + z^3 + z^2 + z + 1 ∣ z^k - 1 ∧ (∀ m : ℕ, m < k → ∃ z : ℂ, ¬(z^8 + z^7 + z^4 + z^3 + z^2 + z + 1 ∣ z^m - 1))) ↔ k = 14 := sorry

end NUMINAMATH_GPT_smallest_k_for_divisibility_l1297_129775


namespace NUMINAMATH_GPT_average_speed_palindrome_l1297_129734

open Nat

theorem average_speed_palindrome :
  ∀ (initial final : ℕ) (time : ℕ), (initial = 12321) →
    (final = 12421) →
    (time = 3) →
    (∃ speed : ℚ, speed = (final - initial) / time ∧ speed = 33.33) :=
by
  intros initial final time h_initial h_final h_time
  sorry

end NUMINAMATH_GPT_average_speed_palindrome_l1297_129734


namespace NUMINAMATH_GPT_count_difference_l1297_129747

-- Given definitions
def count_six_digit_numbers_in_ascending_order_by_digits : ℕ := by
  -- Calculation using binomial coefficient
  exact Nat.choose 9 6

def count_six_digit_numbers_with_one : ℕ := by
  -- Calculation using binomial coefficient with fixed '1' in one position
  exact Nat.choose 8 5

def count_six_digit_numbers_without_one : ℕ := by
  -- Calculation subtracting with and without 1
  exact count_six_digit_numbers_in_ascending_order_by_digits - count_six_digit_numbers_with_one

-- Theorem to prove
theorem count_difference : 
  count_six_digit_numbers_with_one - count_six_digit_numbers_without_one = 28 :=
by
  sorry

end NUMINAMATH_GPT_count_difference_l1297_129747


namespace NUMINAMATH_GPT_left_square_side_length_l1297_129733

theorem left_square_side_length (x : ℕ) (h1 : ∀ y : ℕ, y = x + 17)
                                (h2 : ∀ z : ℕ, z = x + 11)
                                (h3 : 3 * x + 28 = 52) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_left_square_side_length_l1297_129733


namespace NUMINAMATH_GPT_reduction_of_cycle_l1297_129771

noncomputable def firstReductionPercentage (P : ℝ) (x : ℝ) : Prop :=
  P * (1 - (x / 100)) * 0.8 = 0.6 * P

theorem reduction_of_cycle (P x : ℝ) (hP : 0 < P) : firstReductionPercentage P x → x = 25 :=
by
  intros h
  unfold firstReductionPercentage at h
  sorry

end NUMINAMATH_GPT_reduction_of_cycle_l1297_129771


namespace NUMINAMATH_GPT_students_enjoy_both_music_and_sports_l1297_129762

theorem students_enjoy_both_music_and_sports :
  ∀ (T M S N B : ℕ), T = 55 → M = 35 → S = 45 → N = 4 → B = M + S - (T - N) → B = 29 :=
by
  intros T M S N B hT hM hS hN hB
  rw [hT, hM, hS, hN] at hB
  exact hB

end NUMINAMATH_GPT_students_enjoy_both_music_and_sports_l1297_129762


namespace NUMINAMATH_GPT_minimum_value_ineq_l1297_129785

variable {a b c : ℝ}

theorem minimum_value_ineq (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2 * a + 1) * (b^2 + 2 * b + 1) * (c^2 + 2 * c + 1) / (a * b * c) ≥ 64 :=
sorry

end NUMINAMATH_GPT_minimum_value_ineq_l1297_129785


namespace NUMINAMATH_GPT_minimum_value_of_fractions_l1297_129727

theorem minimum_value_of_fractions (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : 1 / a + 1 / b = 1) : 
  ∃ a b, (0 < a) ∧ (0 < b) ∧ (1 / a + 1 / b = 1) ∧ (∃ t, ∀ x y, (0 < x) ∧ (0 < y) ∧ (1 / x + 1 / y = 1) -> t = (1 / (x - 1) + 4 / (y - 1))) := 
sorry

end NUMINAMATH_GPT_minimum_value_of_fractions_l1297_129727


namespace NUMINAMATH_GPT_inequality_proof_l1297_129776

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a / (b + c) + b / (c + d) + c / (d + a) + d / (a + b) ≥ 2 := 
sorry

end NUMINAMATH_GPT_inequality_proof_l1297_129776


namespace NUMINAMATH_GPT_tan_sum_formula_l1297_129700

theorem tan_sum_formula (α β : ℝ) (h1 : Real.tan α = 2) (h2 : Real.tan β = 3) : 
  Real.tan (α + β) = -1 := by
sorry

end NUMINAMATH_GPT_tan_sum_formula_l1297_129700


namespace NUMINAMATH_GPT_sin_c_eq_tan_b_find_side_length_c_l1297_129742

-- (1) Prove that sinC = tanB
theorem sin_c_eq_tan_b {a b c : ℝ} {C : ℝ} (h1 : a / b = 1 + Real.cos C) : 
  Real.sin C = Real.tan B := by
  sorry

-- (2) If given conditions, find the value of c
theorem find_side_length_c {a b c : ℝ} {B C : ℝ} 
  (h1 : Real.cos B = 2 * Real.sqrt 7 / 7)
  (h2 : 0 < C ∧ C < Real.pi / 2)
  (h3 : 1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) 
  : c = Real.sqrt 7 := by
  sorry

end NUMINAMATH_GPT_sin_c_eq_tan_b_find_side_length_c_l1297_129742


namespace NUMINAMATH_GPT_sum_of_coefficients_zero_l1297_129724

open Real

theorem sum_of_coefficients_zero (a b c p1 p2 q1 q2 : ℝ)
  (h1 : ∃ p1 p2 : ℝ, p1 ≠ p2 ∧ a * p1^2 + b * p1 + c = 0 ∧ a * p2^2 + b * p2 + c = 0)
  (h2 : ∃ q1 q2 : ℝ, q1 ≠ q2 ∧ c * q1^2 + b * q1 + a = 0 ∧ c * q2^2 + b * q2 + a = 0)
  (h3 : q1 = p1 + (p2 - p1) / 2 ∧ p2 = p1 + (p2 - p1) ∧ q2 = p1 + 3 * (p2 - p1) / 2) :
  a + c = 0 := sorry

end NUMINAMATH_GPT_sum_of_coefficients_zero_l1297_129724


namespace NUMINAMATH_GPT_perfect_squares_ending_in_5_or_6_lt_2000_l1297_129708

theorem perfect_squares_ending_in_5_or_6_lt_2000 :
  ∃ (n : ℕ), n = 9 ∧ ∀ k, 1 ≤ k ∧ k ≤ 44 → 
  (∃ m, m * m < 2000 ∧ (m % 10 = 5 ∨ m % 10 = 6)) :=
by
  sorry

end NUMINAMATH_GPT_perfect_squares_ending_in_5_or_6_lt_2000_l1297_129708


namespace NUMINAMATH_GPT_product_is_correct_l1297_129745

-- Define the variables and conditions
variables {a b c d : ℚ}

-- State the conditions
def conditions (a b c d : ℚ) :=
  3 * a + 2 * b + 4 * c + 6 * d = 36 ∧
  4 * (d + c) = b ∧
  4 * b + 2 * c = a ∧
  c - 2 = d

-- The theorem statement
theorem product_is_correct (a b c d : ℚ) (h : conditions a b c d) :
  a * b * c * d = -315 / 32 :=
sorry

end NUMINAMATH_GPT_product_is_correct_l1297_129745


namespace NUMINAMATH_GPT_fraction_to_decimal_l1297_129757

theorem fraction_to_decimal :
  (45 : ℚ) / (5 ^ 3) = 0.360 :=
by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l1297_129757


namespace NUMINAMATH_GPT_expected_attempts_for_10_suitcases_l1297_129763

noncomputable def expected_attempts (n : ℕ) : ℝ :=
  (1 / 2) * (n * (n + 1) / 2) + (n / 2) - (Real.log n + 0.577)

theorem expected_attempts_for_10_suitcases :
  abs (expected_attempts 10 - 29.62) < 1 :=
by
  sorry

end NUMINAMATH_GPT_expected_attempts_for_10_suitcases_l1297_129763


namespace NUMINAMATH_GPT_jana_walk_distance_l1297_129706

-- Define the time taken to walk one mile and the rest period
def walk_time_per_mile : ℕ := 24
def rest_time_per_mile : ℕ := 6

-- Define the total time spent per mile (walking + resting)
def total_time_per_mile : ℕ := walk_time_per_mile + rest_time_per_mile

-- Define the total available time
def total_available_time : ℕ := 78

-- Define the number of complete cycles of walking and resting within the total available time
def complete_cycles : ℕ := total_available_time / total_time_per_mile

-- Define the distance walked per cycle (in miles)
def distance_per_cycle : ℝ := 1.0

-- Define the total distance walked
def total_distance_walked : ℝ := complete_cycles * distance_per_cycle

-- The proof statement
theorem jana_walk_distance : total_distance_walked = 2.0 := by
  sorry

end NUMINAMATH_GPT_jana_walk_distance_l1297_129706


namespace NUMINAMATH_GPT_find_b6b8_l1297_129770

-- Define sequences {a_n} (arithmetic sequence) and {b_n} (geometric sequence)
variable {a : ℕ → ℝ} {b : ℕ → ℝ}

-- Given conditions
axiom h1 : ∀ n m : ℕ, a m = a n + (m - n) * (a (n + 1) - a n) -- Arithmetic sequence property
axiom h2 : 2 * a 3 - (a 7) ^ 2 + 2 * a 11 = 0
axiom h3 : ∀ n : ℕ, b (n + 1) / b n = b 2 / b 1 -- Geometric sequence property
axiom h4 : b 7 = a 7
axiom h5 : ∀ n : ℕ, b n > 0                 -- Assuming b_n has positive terms
axiom h6 : ∀ n : ℕ, a n > 0                 -- Positive terms in sequence a_n

-- Proof objective
theorem find_b6b8 : b 6 * b 8 = 16 :=
by sorry

end NUMINAMATH_GPT_find_b6b8_l1297_129770


namespace NUMINAMATH_GPT_women_in_the_minority_l1297_129722

theorem women_in_the_minority (total_employees : ℕ) (female_employees : ℕ) (h : female_employees < total_employees * 20 / 100) : 
  (female_employees < total_employees / 2) :=
by
  sorry

end NUMINAMATH_GPT_women_in_the_minority_l1297_129722


namespace NUMINAMATH_GPT_max_value_of_function_cos_sin_l1297_129779

noncomputable def max_value_function (x : ℝ) : ℝ := 
  (Real.cos x)^3 + (Real.sin x)^2 - Real.cos x

theorem max_value_of_function_cos_sin : 
  ∃ x ∈ (Set.univ : Set ℝ), max_value_function x = (32 / 27) := 
sorry

end NUMINAMATH_GPT_max_value_of_function_cos_sin_l1297_129779


namespace NUMINAMATH_GPT_find_f_2_l1297_129703

noncomputable def f (a b x : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f_2 (a b : ℝ) (h : f a b (-2) = 10) : f a b 2 = -26 :=
by
  sorry

end NUMINAMATH_GPT_find_f_2_l1297_129703


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1297_129783

-- Problem (1)
theorem problem1 : 6 - -2 + -4 - 3 = 1 :=
by sorry

-- Problem (2)
theorem problem2 : 8 / -2 * (1 / 3 : ℝ) * (-(1 + 1/2: ℝ)) = 2 :=
by sorry

-- Problem (3)
theorem problem3 : (13 + (2 / 7 - 1 / 14) * 56) / (-1 / 4) = -100 :=
by sorry

-- Problem (4)
theorem problem4 : 
  |-(5 / 6 : ℝ)| / ((-(3 + 1 / 5: ℝ)) / (-4)^2 + (-7 / 4) * (4 / 7)) = -(25 / 36) :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1297_129783


namespace NUMINAMATH_GPT_longest_side_of_triangle_l1297_129759

variable (x y : ℝ)

def side1 := 10
def side2 := 2*y + 3
def side3 := 3*x + 2

theorem longest_side_of_triangle
  (h_perimeter : side1 + side2 + side3 = 45)
  (h_side2_pos : side2 > 0)
  (h_side3_pos : side3 > 0) :
  side3 = 32 :=
sorry

end NUMINAMATH_GPT_longest_side_of_triangle_l1297_129759


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1297_129769

-- Definitions based on conditions
def A : Set ℝ := { x | x + 2 = 0 }
def B : Set ℝ := { x | x^2 - 4 = 0 }

-- Theorem statement proving the question == answer given conditions
theorem intersection_of_A_and_B : A ∩ B = {-2} :=
by 
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1297_129769


namespace NUMINAMATH_GPT_average_star_rating_l1297_129754

/-- Define specific constants for the problem. --/
def reviews_5_star := 6
def reviews_4_star := 7
def reviews_3_star := 4
def reviews_2_star := 1
def total_reviews := 18

/-- Calculate the total stars given the number of each type of review. --/
def total_stars : ℕ := 
  (reviews_5_star * 5) + 
  (reviews_4_star * 4) + 
  (reviews_3_star * 3) + 
  (reviews_2_star * 2)

/-- Prove that the average star rating is 4. --/
theorem average_star_rating : total_stars / total_reviews = 4 := by 
  sorry

end NUMINAMATH_GPT_average_star_rating_l1297_129754


namespace NUMINAMATH_GPT_SarahCansYesterday_l1297_129725

variable (S : ℕ)
variable (LaraYesterday : ℕ := S + 30)
variable (SarahToday : ℕ := 40)
variable (LaraToday : ℕ := 70)
variable (YesterdayTotal : ℕ := LaraYesterday + S)
variable (TodayTotal : ℕ := SarahToday + LaraToday)

theorem SarahCansYesterday : 
  TodayTotal + 20 = YesterdayTotal -> 
  S = 50 :=
by
  sorry

end NUMINAMATH_GPT_SarahCansYesterday_l1297_129725


namespace NUMINAMATH_GPT_city_raised_money_for_charity_l1297_129746

-- Definitions based on conditions from part a)
def price_regular_duck : ℝ := 3.0
def price_large_duck : ℝ := 5.0
def number_regular_ducks_sold : ℕ := 221
def number_large_ducks_sold : ℕ := 185

-- Definition to represent the main theorem: Total money raised
noncomputable def total_money_raised : ℝ :=
  price_regular_duck * number_regular_ducks_sold + price_large_duck * number_large_ducks_sold

-- Theorem to prove that the total money raised is $1588.00
theorem city_raised_money_for_charity : total_money_raised = 1588.0 := by
  sorry

end NUMINAMATH_GPT_city_raised_money_for_charity_l1297_129746


namespace NUMINAMATH_GPT_registered_voters_democrats_l1297_129761

variables (D R : ℝ)

theorem registered_voters_democrats :
  (D + R = 100) →
  (0.80 * D + 0.30 * R = 65) →
  D = 70 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_registered_voters_democrats_l1297_129761


namespace NUMINAMATH_GPT_positive_real_solution_eq_l1297_129787

theorem positive_real_solution_eq :
  ∃ x : ℝ, 0 < x ∧ ( (1/4) * (5 * x^2 - 4) = (x^2 - 40 * x - 5) * (x^2 + 20 * x + 2) ) ∧ x = 20 + 10 * Real.sqrt 41 :=
by
  sorry

end NUMINAMATH_GPT_positive_real_solution_eq_l1297_129787


namespace NUMINAMATH_GPT_matrix_addition_correct_l1297_129726

def matrixA : Matrix (Fin 2) (Fin 2) ℤ := ![![4, -3], ![0, 5]]
def matrixB : Matrix (Fin 2) (Fin 2) ℤ := ![![-6, 2], ![7, -10]]
def matrixC : Matrix (Fin 2) (Fin 2) ℤ := ![![-2, -1], ![7, -5]]

theorem matrix_addition_correct : matrixA + matrixB = matrixC := by
  sorry

end NUMINAMATH_GPT_matrix_addition_correct_l1297_129726


namespace NUMINAMATH_GPT_reliability_is_correct_l1297_129789

-- Define the probabilities of each switch functioning properly.
def P_A : ℝ := 0.9
def P_B : ℝ := 0.8
def P_C : ℝ := 0.7

-- Define the system reliability.
def reliability : ℝ := P_A * P_B * P_C

-- The theorem stating the reliability of the system.
theorem reliability_is_correct : reliability = 0.504 := by
  sorry

end NUMINAMATH_GPT_reliability_is_correct_l1297_129789


namespace NUMINAMATH_GPT_no_such_m_exists_l1297_129748

theorem no_such_m_exists : ¬ ∃ m : ℝ, ∀ x : ℝ, m * x^2 - 2 * x - m + 1 < 0 :=
sorry

end NUMINAMATH_GPT_no_such_m_exists_l1297_129748


namespace NUMINAMATH_GPT_sequence_property_l1297_129794

theorem sequence_property : 
  (∀ (a : ℕ → ℝ), a 1 = 2 ∧ (∀ n ≥ 1, a (n + 1) = a n + (2 * a n) / n) → a 200 = 40200) :=
by
  sorry

end NUMINAMATH_GPT_sequence_property_l1297_129794


namespace NUMINAMATH_GPT_maximize_profit_l1297_129702

-- Define the relationships and constants
def P (x : ℝ) : ℝ := -750 * x + 15000
def material_cost_per_unit : ℝ := 4
def fixed_cost : ℝ := 7000

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - material_cost_per_unit) * P x - fixed_cost

-- The statement of the problem, proving the maximization condition
theorem maximize_profit :
  ∃ x : ℝ, x = 12 ∧ profit 12 = 41000 := by
  sorry

end NUMINAMATH_GPT_maximize_profit_l1297_129702


namespace NUMINAMATH_GPT_two_numbers_product_l1297_129729

theorem two_numbers_product (x y : ℕ) 
  (h1 : x + y = 90) 
  (h2 : x - y = 10) : x * y = 2000 :=
by
  sorry

end NUMINAMATH_GPT_two_numbers_product_l1297_129729


namespace NUMINAMATH_GPT_sum_quotient_dividend_divisor_l1297_129793

theorem sum_quotient_dividend_divisor (N : ℕ) (divisor : ℕ) (quotient : ℕ) (sum : ℕ) 
    (h₁ : N = 40) (h₂ : divisor = 2) (h₃ : quotient = N / divisor)
    (h₄ : sum = quotient + N + divisor) : sum = 62 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_sum_quotient_dividend_divisor_l1297_129793


namespace NUMINAMATH_GPT_Lakeview_High_School_Basketball_Team_l1297_129782

theorem Lakeview_High_School_Basketball_Team :
  ∀ (total_players taking_physics taking_both statistics: ℕ),
  total_players = 25 →
  taking_physics = 10 →
  taking_both = 5 →
  statistics = 20 :=
sorry

end NUMINAMATH_GPT_Lakeview_High_School_Basketball_Team_l1297_129782


namespace NUMINAMATH_GPT_length_of_OP_is_sqrt_200_div_3_l1297_129714

open Real

def square (a : ℝ) := a * a

theorem length_of_OP_is_sqrt_200_div_3 (KL MO MP OP : ℝ) (h₁ : KL = 10)
  (h₂: MO = MP) (h₃: square (10) = 100)
  (h₄ : 1 / 6 * 100 = 1 / 2 * (MO * MP)) : OP = sqrt (200/3) :=
by
  sorry

end NUMINAMATH_GPT_length_of_OP_is_sqrt_200_div_3_l1297_129714
