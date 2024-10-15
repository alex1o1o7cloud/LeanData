import Mathlib

namespace NUMINAMATH_GPT_intersection_M_N_l1655_165593

def M : Set ℤ := { -2, -1, 0, 1, 2 }
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = { -2 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1655_165593


namespace NUMINAMATH_GPT_Ed_lost_marble_count_l1655_165588

variable (D : ℕ) -- Number of marbles Doug has

noncomputable def Ed_initial := D + 19 -- Ed initially had D + 19 marbles
noncomputable def Ed_now := D + 8 -- Ed now has D + 8 marbles
noncomputable def Ed_lost := Ed_initial D - Ed_now D -- Ed lost Ed_initial - Ed_now marbles

theorem Ed_lost_marble_count : Ed_lost D = 11 := by 
  sorry

end NUMINAMATH_GPT_Ed_lost_marble_count_l1655_165588


namespace NUMINAMATH_GPT_equipment_unit_prices_purchasing_scenarios_l1655_165540

theorem equipment_unit_prices
  (x : ℝ)
  (price_A_eq_price_B_minus_10 : ∀ y, ∃ z, z = y + 10)
  (eq_purchases_equal_cost_A : ∀ n : ℕ, 300 / x = n)
  (eq_purchases_equal_cost_B : ∀ n : ℕ, 360 / (x + 10) = n) :
  x = 50 ∧ (x + 10) = 60 :=
by
  sorry

theorem purchasing_scenarios
  (m n : ℕ)
  (price_A : ℝ := 50)
  (price_B : ℝ := 60)
  (budget : ℝ := 1000)
  (purchase_eq_budget : 50 * m + 60 * n = 1000)
  (pos_integers : m > 0 ∧ n > 0) :
  (m = 14 ∧ n = 5) ∨ (m = 8 ∧ n = 10) ∨ (m = 2 ∧ n = 15) :=
by
  sorry

end NUMINAMATH_GPT_equipment_unit_prices_purchasing_scenarios_l1655_165540


namespace NUMINAMATH_GPT_trig_identity_l1655_165508

theorem trig_identity (α : ℝ) :
  (Real.cos (α - 35 * Real.pi / 180) * Real.cos (25 * Real.pi / 180 + α) +
   Real.sin (α - 35 * Real.pi / 180) * Real.sin (25 * Real.pi / 180 + α)) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l1655_165508


namespace NUMINAMATH_GPT_coolers_total_capacity_l1655_165585

theorem coolers_total_capacity :
  ∃ (C1 C2 C3 : ℕ), 
    C1 = 100 ∧ 
    C2 = C1 + (C1 / 2) ∧ 
    C3 = C2 / 2 ∧ 
    (C1 + C2 + C3 = 325) :=
sorry

end NUMINAMATH_GPT_coolers_total_capacity_l1655_165585


namespace NUMINAMATH_GPT_calculate_f_50_l1655_165537

noncomputable def f (x : ℝ) : ℝ := sorry

theorem calculate_f_50 (f : ℝ → ℝ) (h_fun : ∀ x y : ℝ, f (x * y) = y * f x) (h_f2 : f 2 = 10) :
  f 50 = 250 :=
by
  sorry

end NUMINAMATH_GPT_calculate_f_50_l1655_165537


namespace NUMINAMATH_GPT_books_left_over_l1655_165504

theorem books_left_over (n_boxes : ℕ) (books_per_box : ℕ) (new_box_capacity : ℕ) :
  n_boxes = 1575 → books_per_box = 45 → new_box_capacity = 46 →
  (n_boxes * books_per_box) % new_box_capacity = 15 :=
by
  intro h1 h2 h3
  rw [h1, h2, h3]
  -- Actual proof steps would go here
  sorry

end NUMINAMATH_GPT_books_left_over_l1655_165504


namespace NUMINAMATH_GPT_total_growing_space_is_correct_l1655_165542

def garden_bed_area (length : ℕ) (width : ℕ) (count : ℕ) : ℕ :=
  length * width * count

def total_growing_space : ℕ :=
  garden_bed_area 5 4 3 +
  garden_bed_area 6 3 4 +
  garden_bed_area 7 5 2 +
  garden_bed_area 8 4 1

theorem total_growing_space_is_correct :
  total_growing_space = 234 := by
  sorry

end NUMINAMATH_GPT_total_growing_space_is_correct_l1655_165542


namespace NUMINAMATH_GPT_tan_double_angle_identity_l1655_165559

theorem tan_double_angle_identity (theta : ℝ) (h1 : 0 < theta ∧ theta < Real.pi / 2)
  (h2 : Real.sin theta - Real.cos theta = Real.sqrt 5 / 5) :
  Real.tan (2 * theta) = -(4 / 3) := 
by
  sorry

end NUMINAMATH_GPT_tan_double_angle_identity_l1655_165559


namespace NUMINAMATH_GPT_min_value_of_function_l1655_165500

open Real

theorem min_value_of_function (x : ℝ) (h : x > 2) : (∃ a : ℝ, (∀ y : ℝ, y = (4 / (x - 2) + x) → y ≥ a) ∧ a = 6) :=
sorry

end NUMINAMATH_GPT_min_value_of_function_l1655_165500


namespace NUMINAMATH_GPT_find_m_from_intersection_l1655_165579

-- Define the sets A and B
def A : Set ℕ := {1, 2, 3}
def B (m : ℕ) : Set ℕ := {2, m, 4}

-- Prove the relationship given the conditions
theorem find_m_from_intersection (m : ℕ) (h : A ∩ B m = {2, 3}) : m = 3 := 
by 
  sorry

end NUMINAMATH_GPT_find_m_from_intersection_l1655_165579


namespace NUMINAMATH_GPT_cost_of_child_ticket_l1655_165531

theorem cost_of_child_ticket
  (total_seats : ℕ)
  (adult_ticket_price : ℕ)
  (num_children : ℕ)
  (total_revenue : ℕ)
  (H1 : total_seats = 250)
  (H2 : adult_ticket_price = 6)
  (H3 : num_children = 188)
  (H4 : total_revenue = 1124) :
  let num_adults := total_seats - num_children
  let revenue_from_adults := num_adults * adult_ticket_price
  let cost_of_child_ticket := (total_revenue - revenue_from_adults) / num_children
  cost_of_child_ticket = 4 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_child_ticket_l1655_165531


namespace NUMINAMATH_GPT_non_negative_integer_solutions_of_inequality_system_l1655_165557

theorem non_negative_integer_solutions_of_inequality_system :
  (∀ x : ℚ, 3 * (x - 1) < 5 * x + 1 → (x - 1) / 2 ≥ 2 * x - 4 → (x = 0 ∨ x = 1 ∨ x = 2)) :=
by
  sorry

end NUMINAMATH_GPT_non_negative_integer_solutions_of_inequality_system_l1655_165557


namespace NUMINAMATH_GPT_sequence_general_formula_l1655_165599

theorem sequence_general_formula (a : ℕ → ℝ) (h1 : a 1 = 3) (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 3 * a n - 4) :
  ∀ n : ℕ, n ≥ 1 → a n = 3^(n - 1) + 2 :=
sorry

end NUMINAMATH_GPT_sequence_general_formula_l1655_165599


namespace NUMINAMATH_GPT_tile_size_l1655_165535

theorem tile_size (length width : ℕ) (total_tiles : ℕ) 
  (h_length : length = 48) 
  (h_width : width = 72) 
  (h_total_tiles : total_tiles = 96) : 
  ((length * width) / total_tiles) = 36 := 
by
  sorry

end NUMINAMATH_GPT_tile_size_l1655_165535


namespace NUMINAMATH_GPT_greatest_possible_bxa_l1655_165562

-- Define the property of the number being divisible by 35
def div_by_35 (n : ℕ) : Prop :=
  n % 35 = 0

-- Define the main proof problem
theorem greatest_possible_bxa :
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ div_by_35 (10 * a + b) ∧ (∀ (a' b' : ℕ), a' < 10 → b' < 10 → div_by_35 (10 * a' + b') → b * a ≥ b' * a') :=
sorry

end NUMINAMATH_GPT_greatest_possible_bxa_l1655_165562


namespace NUMINAMATH_GPT_rectangle_perimeter_l1655_165536

theorem rectangle_perimeter (a b : ℚ) (ha : ¬ a.den = 1) (hb : ¬ b.den = 1) (hab : a ≠ b) (h : (a - 2) * (b - 2) = -7) : 2 * (a + b) = 20 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1655_165536


namespace NUMINAMATH_GPT_area_comparison_l1655_165501

noncomputable def area_difference_decagon (s : ℝ) : ℝ := 
  let R := s / (2 * Real.sin (Real.pi / 10))
  let r := s / (2 * Real.tan (Real.pi / 10))
  Real.pi * (R^2 - r^2)

noncomputable def area_difference_nonagon (s : ℝ) : ℝ := 
  let R := s / (2 * Real.sin (Real.pi / 9))
  let r := s / (2 * Real.tan (Real.pi / 9))
  Real.pi * (R^2 - r^2)

theorem area_comparison :
  (area_difference_decagon 3 > area_difference_nonagon 3) :=
sorry

end NUMINAMATH_GPT_area_comparison_l1655_165501


namespace NUMINAMATH_GPT_ticket_cost_before_rally_l1655_165502

-- We define the variables and constants given in the problem
def total_attendance : ℕ := 750
def tickets_before_rally : ℕ := 475
def tickets_at_door : ℕ := total_attendance - tickets_before_rally
def cost_at_door : ℝ := 2.75
def total_receipts : ℝ := 1706.25

-- Problem statement: Prove that the cost of each ticket bought before the rally (x) is 2 dollars.
theorem ticket_cost_before_rally (x : ℝ) 
  (h₁ : tickets_before_rally * x + tickets_at_door * cost_at_door = total_receipts) :
  x = 2 :=
by
  sorry

end NUMINAMATH_GPT_ticket_cost_before_rally_l1655_165502


namespace NUMINAMATH_GPT_find_x_l1655_165594

theorem find_x (x y : ℤ) (h1 : y = 3) (h2 : x + 3 * y = 10) : x = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1655_165594


namespace NUMINAMATH_GPT_change_combinations_50_cents_l1655_165573

-- Define the conditions for creating 50 cents using standard coins
def ways_to_make_change (pennies nickels dimes : ℕ) : ℕ :=
  pennies + 5 * nickels + 10 * dimes

theorem change_combinations_50_cents : 
  ∃ num_ways, 
    num_ways = 28 ∧
    ∀ (pennies nickels dimes : ℕ), 
      pennies + 5 * nickels + 10 * dimes = 50 → 
      -- Exclude using only a single half-dollar
      ¬(num_ways = if (pennies = 0 ∧ nickels = 0 ∧ dimes = 0) then 1 else 28) := 
sorry

end NUMINAMATH_GPT_change_combinations_50_cents_l1655_165573


namespace NUMINAMATH_GPT_slope_of_line_l1655_165570

noncomputable def slope_range : Set ℝ :=
  {α | (5 * Real.pi / 6) ≤ α ∧ α < Real.pi}

theorem slope_of_line (x a : ℝ) :
  let k := -1 / (a^2 + Real.sqrt 3)
  ∃ α ∈ slope_range, k = Real.tan α :=
sorry

end NUMINAMATH_GPT_slope_of_line_l1655_165570


namespace NUMINAMATH_GPT_percentage_B_D_l1655_165521

variables (A B C D : ℝ)

-- Conditions as hypotheses
theorem percentage_B_D
  (h1 : B = 1.71 * A)
  (h2 : C = 1.80 * A)
  (h3 : D = 1.90 * B)
  (h4 : B = 1.62 * C)
  (h5 : A = 0.65 * D)
  (h6 : C = 0.55 * D) : 
  B = 1.1115 * D :=
sorry

end NUMINAMATH_GPT_percentage_B_D_l1655_165521


namespace NUMINAMATH_GPT_point_on_hyperbola_l1655_165510

theorem point_on_hyperbola (p r : ℝ) (h1 : p > 0) (h2 : r > 0)
  (h_el : ∀ (x y : ℝ), x^2 / 4 + y^2 / 2 = 1)
  (h_par : ∀ (x y : ℝ), y^2 = 2 * p * x)
  (h_circum : ∀ (a b c : ℝ), a = 2 * r - 2 * p) :
  r^2 - p^2 = 1 := sorry

end NUMINAMATH_GPT_point_on_hyperbola_l1655_165510


namespace NUMINAMATH_GPT_ellipse_with_foci_on_x_axis_l1655_165539

theorem ellipse_with_foci_on_x_axis {a : ℝ} (h1 : a - 5 > 0) (h2 : 2 > 0) (h3 : a - 5 > 2) :
  a > 7 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_with_foci_on_x_axis_l1655_165539


namespace NUMINAMATH_GPT_book_pages_count_l1655_165558

theorem book_pages_count :
  (∀ n : ℕ, n = 4 → 42 * n = 168) ∧
  (∀ n : ℕ, n = 2 → 50 * n = 100) ∧
  (∀ p1 p2 : ℕ, p1 = 168 ∧ p2 = 100 → p1 + p2 = 268) ∧
  (∀ p : ℕ, p = 268 → p + 30 = 298) →
  298 = 298 := by
  sorry

end NUMINAMATH_GPT_book_pages_count_l1655_165558


namespace NUMINAMATH_GPT_Yella_last_week_usage_l1655_165578

/-- 
Yella's computer usage last week was some hours. If she plans to use the computer 8 hours a day for this week, 
her computer usage for this week is 35 hours less. Given these conditions, prove that Yella's computer usage 
last week was 91 hours.
-/
theorem Yella_last_week_usage (daily_usage : ℕ) (days_in_week : ℕ) (difference : ℕ)
  (h1: daily_usage = 8)
  (h2: days_in_week = 7)
  (h3: difference = 35) :
  daily_usage * days_in_week + difference = 91 := 
by
  sorry

end NUMINAMATH_GPT_Yella_last_week_usage_l1655_165578


namespace NUMINAMATH_GPT_power_exponent_multiplication_l1655_165506

variable (a : ℝ)

theorem power_exponent_multiplication : (a^3)^2 = a^6 := sorry

end NUMINAMATH_GPT_power_exponent_multiplication_l1655_165506


namespace NUMINAMATH_GPT_sum_of_powers_of_two_l1655_165524

theorem sum_of_powers_of_two (n : ℕ) (h : 1 ≤ n ∧ n ≤ 511) : 
  ∃ (S : Finset ℕ), S ⊆ ({2^8, 2^7, 2^6, 2^5, 2^4, 2^3, 2^2, 2^1, 2^0} : Finset ℕ) ∧ 
  S.sum id = n :=
by
  sorry

end NUMINAMATH_GPT_sum_of_powers_of_two_l1655_165524


namespace NUMINAMATH_GPT_jane_nail_polish_drying_time_l1655_165581

theorem jane_nail_polish_drying_time :
  let base_coat := 4
  let color_coat_1 := 5
  let color_coat_2 := 6
  let color_coat_3 := 7
  let index_finger_1 := 8
  let index_finger_2 := 10
  let middle_finger := 12
  let ring_finger := 11
  let pinky_finger := 14
  let top_coat := 9
  base_coat + color_coat_1 + color_coat_2 + color_coat_3 + index_finger_1 + index_finger_2 + middle_finger + ring_finger + pinky_finger + top_coat = 86 :=
by sorry

end NUMINAMATH_GPT_jane_nail_polish_drying_time_l1655_165581


namespace NUMINAMATH_GPT_intersect_sphere_circle_l1655_165566

-- Define the given sphere equation
def sphere (h k l R : ℝ) (x y z : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 + (z - l)^2 = R^2

-- Define the equation of a circle in the plane x = x0 parallel to the yz-plane
def circle_in_plane (x0 y0 z0 r : ℝ) (y z : ℝ) : Prop :=
  (y - y0)^2 + (z - z0)^2 = r^2

-- Define the intersecting circle from the sphere equation in the x = c plane
def intersecting_circle (h k l c R : ℝ) (y z : ℝ) : Prop :=
  (y - k)^2 + (z - l)^2 = R^2 - (h - c)^2

-- The main proof statement
theorem intersect_sphere_circle (h k l R c x0 y0 z0 r: ℝ) :
  ∀ y z, intersecting_circle h k l c R y z ↔ circle_in_plane x0 y0 z0 r y z :=
sorry

end NUMINAMATH_GPT_intersect_sphere_circle_l1655_165566


namespace NUMINAMATH_GPT_Natalia_Tuesday_distance_l1655_165577

theorem Natalia_Tuesday_distance :
  ∃ T : ℕ, (40 + T + T / 2 + (40 + T / 2) = 180) ∧ T = 33 :=
by
  existsi 33
  -- proof can be filled here
  sorry

end NUMINAMATH_GPT_Natalia_Tuesday_distance_l1655_165577


namespace NUMINAMATH_GPT_Jake_weight_correct_l1655_165589

def Mildred_weight : ℕ := 59
def Carol_weight : ℕ := Mildred_weight + 9
def Jake_weight : ℕ := 2 * Carol_weight

theorem Jake_weight_correct : Jake_weight = 136 := by
  sorry

end NUMINAMATH_GPT_Jake_weight_correct_l1655_165589


namespace NUMINAMATH_GPT_ravi_jump_height_l1655_165561

theorem ravi_jump_height (j1 j2 j3 : ℕ) (average : ℕ) (ravi_jump_height : ℕ) (h : j1 = 23 ∧ j2 = 27 ∧ j3 = 28) 
  (ha : average = (j1 + j2 + j3) / 3) (hr : ravi_jump_height = 3 * average / 2) : ravi_jump_height = 39 :=
by
  sorry

end NUMINAMATH_GPT_ravi_jump_height_l1655_165561


namespace NUMINAMATH_GPT_richmond_tigers_revenue_l1655_165546

theorem richmond_tigers_revenue
  (total_tickets : ℕ)
  (first_half_tickets : ℕ)
  (catA_first_half : ℕ)
  (catB_first_half : ℕ)
  (catC_first_half : ℕ)
  (priceA : ℕ)
  (priceB : ℕ)
  (priceC : ℕ)
  (catA_second_half : ℕ)
  (catB_second_half : ℕ)
  (catC_second_half : ℕ)
  (total_revenue_second_half : ℕ)
  (h_total_tickets : total_tickets = 9570)
  (h_first_half_tickets : first_half_tickets = 3867)
  (h_catA_first_half : catA_first_half = 1350)
  (h_catB_first_half : catB_first_half = 1150)
  (h_catC_first_half : catC_first_half = 1367)
  (h_priceA : priceA = 50)
  (h_priceB : priceB = 40)
  (h_priceC : priceC = 30)
  (h_catA_second_half : catA_second_half = 1350)
  (h_catB_second_half : catB_second_half = 1150)
  (h_catC_second_half : catC_second_half = 1367)
  (h_total_revenue_second_half : total_revenue_second_half = 154510)
  :
  catA_second_half * priceA + catB_second_half * priceB + catC_second_half * priceC = total_revenue_second_half :=
by
  sorry

end NUMINAMATH_GPT_richmond_tigers_revenue_l1655_165546


namespace NUMINAMATH_GPT_middle_number_l1655_165575

theorem middle_number {a b c : ℚ} 
  (h1 : a + b = 15) 
  (h2 : a + c = 20) 
  (h3 : b + c = 23) 
  (h4 : c = 2 * a) : 
  b = 25 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_middle_number_l1655_165575


namespace NUMINAMATH_GPT_geometric_seq_ratio_l1655_165574

theorem geometric_seq_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : a 5 - a 3 = 12) 
  (h2 : a 6 - a 4 = 24) 
  (ha : ∃ a₁ : ℝ, (∀ n : ℕ, a n = a₁ * q ^ (n - 1)))
  (hS : ∀ n : ℕ, S n = a₁ * (1 - q ^ n) / (1 - q)) :
  ∀ n : ℕ, S n / a n = 2 - 2 ^ (1 - n) :=
sorry

end NUMINAMATH_GPT_geometric_seq_ratio_l1655_165574


namespace NUMINAMATH_GPT_latest_time_temp_decreasing_l1655_165591

theorem latest_time_temp_decreasing (t : ℝ) 
  (h1 : -t^2 + 12 * t + 55 = 82) 
  (h2 : ∀ t0 : ℝ, -2 * t0 + 12 < 0 → t > t0) : 
  t = 6 + (3 * Real.sqrt 28 / 2) :=
sorry

end NUMINAMATH_GPT_latest_time_temp_decreasing_l1655_165591


namespace NUMINAMATH_GPT_each_persons_share_l1655_165516

def total_bill : ℝ := 211.00
def number_of_people : ℕ := 5
def tip_rate : ℝ := 0.15

theorem each_persons_share :
  (total_bill * (1 + tip_rate)) / number_of_people = 48.53 := 
by sorry

end NUMINAMATH_GPT_each_persons_share_l1655_165516


namespace NUMINAMATH_GPT_base7_addition_l1655_165534

theorem base7_addition : (21 : ℕ) + 254 = 505 :=
by sorry

end NUMINAMATH_GPT_base7_addition_l1655_165534


namespace NUMINAMATH_GPT_find_a_l1655_165572

theorem find_a (a : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (hf : ∀ x, f x = a * x^3 + 3 * x^2 + 2)
  (hf' : ∀ x, f' x = 3 * a * x^2 + 6 * x) 
  (h : f' (-1) = 4) : 
  a = (10 : ℝ) / 3 := 
sorry

end NUMINAMATH_GPT_find_a_l1655_165572


namespace NUMINAMATH_GPT_count_implications_l1655_165518

theorem count_implications (p q r : Prop) :
  ((p ∧ q ∧ ¬r → ((q → p) → ¬r)) ∧ 
   (¬p ∧ ¬q ∧ ¬r → ((q → p) → ¬r)) ∧ 
   (p ∧ ¬q ∧ r → ¬ ((q → p) → ¬r)) ∧ 
   (¬p ∧ q ∧ ¬r → ((q → p) → ¬r))) →
   (3 = 3) := sorry

end NUMINAMATH_GPT_count_implications_l1655_165518


namespace NUMINAMATH_GPT_passengers_in_7_buses_l1655_165551

theorem passengers_in_7_buses (passengers_total buses_total_given buses_required : ℕ) 
    (h1 : passengers_total = 456) 
    (h2 : buses_total_given = 12) 
    (h3 : buses_required = 7) :
    (passengers_total / buses_total_given) * buses_required = 266 := 
sorry

end NUMINAMATH_GPT_passengers_in_7_buses_l1655_165551


namespace NUMINAMATH_GPT_phi_varphi_difference_squared_l1655_165582

theorem phi_varphi_difference_squared :
  ∀ (Φ φ : ℝ), (Φ ≠ φ) → (Φ^2 - 2*Φ - 1 = 0) → (φ^2 - 2*φ - 1 = 0) →
  (Φ - φ)^2 = 8 :=
by
  intros Φ φ distinct hΦ hφ
  sorry

end NUMINAMATH_GPT_phi_varphi_difference_squared_l1655_165582


namespace NUMINAMATH_GPT_point_coordinates_l1655_165513

def point : Type := ℝ × ℝ

def x_coordinate (P : point) : ℝ := P.1

def y_coordinate (P : point) : ℝ := P.2

theorem point_coordinates (P : point) (h1 : x_coordinate P = -3) (h2 : abs (y_coordinate P) = 5) :
  P = (-3, 5) ∨ P = (-3, -5) :=
by
  sorry

end NUMINAMATH_GPT_point_coordinates_l1655_165513


namespace NUMINAMATH_GPT_cos_diff_of_symmetric_sines_l1655_165576

theorem cos_diff_of_symmetric_sines (a β : Real) (h1 : Real.sin a = 1 / 3) 
  (h2 : Real.sin β = 1 / 3) (h3 : Real.cos a = -Real.cos β) : 
  Real.cos (a - β) = -7 / 9 := by
  sorry

end NUMINAMATH_GPT_cos_diff_of_symmetric_sines_l1655_165576


namespace NUMINAMATH_GPT_probability_max_min_difference_is_five_l1655_165505

theorem probability_max_min_difference_is_five : 
  let total_outcomes := 6 ^ 4
  let outcomes_without_1 := 5 ^ 4
  let outcomes_without_6 := 5 ^ 4
  let outcomes_without_1_and_6 := 4 ^ 4
  total_outcomes - 2 * outcomes_without_1 + outcomes_without_1_and_6 = 302 →
  (302 : ℚ) / total_outcomes = 151 / 648 :=
by
  intros
  sorry

end NUMINAMATH_GPT_probability_max_min_difference_is_five_l1655_165505


namespace NUMINAMATH_GPT_collinear_k_perpendicular_k_l1655_165512

def vector := ℝ × ℝ

def a : vector := (1, 3)
def b : vector := (3, -4)

def collinear (u v : vector) : Prop :=
  u.1 * v.2 = u.2 * v.1

def perpendicular (u v : vector) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

def k_vector_a_minus_b (k : ℝ) (a b : vector) : vector :=
  (k * a.1 - b.1, k * a.2 - b.2)

def a_plus_b (a b : vector) : vector :=
  (a.1 + b.1, a.2 + b.2)

theorem collinear_k (k : ℝ) : collinear (k_vector_a_minus_b k a b) (a_plus_b a b) ↔ k = -1 :=
sorry

theorem perpendicular_k (k : ℝ) : perpendicular (k_vector_a_minus_b k a b) (a_plus_b a b) ↔ k = 16 :=
sorry

end NUMINAMATH_GPT_collinear_k_perpendicular_k_l1655_165512


namespace NUMINAMATH_GPT_polynomial_degree_one_condition_l1655_165595

theorem polynomial_degree_one_condition (P : ℝ → ℝ) (c : ℝ) :
  (∀ a b : ℝ, a < b → (P = fun x => x + c) ∨ (P = fun x => -x + c)) ∧
  (∀ a b : ℝ, a < b →
    (max (P a) (P b) - min (P a) (P b) = b - a)) :=
sorry

end NUMINAMATH_GPT_polynomial_degree_one_condition_l1655_165595


namespace NUMINAMATH_GPT_probability_student_less_than_25_l1655_165554

-- Defining the problem conditions
def total_students : ℕ := 100
def percent_male : ℕ := 40
def percent_female : ℕ := 100 - percent_male
def percent_male_25_or_older : ℕ := 40
def percent_female_25_or_older : ℕ := 30

-- Calculation based on the conditions
def num_male_students := (percent_male * total_students) / 100
def num_female_students := (percent_female * total_students) / 100
def num_male_25_or_older := (percent_male_25_or_older * num_male_students) / 100
def num_female_25_or_older := (percent_female_25_or_older * num_female_students) / 100

def num_25_or_older := num_male_25_or_older + num_female_25_or_older
def num_less_than_25 := total_students - num_25_or_older
def probability_less_than_25 := (num_less_than_25: ℚ) / total_students

-- Define the theorem
theorem probability_student_less_than_25 :
  probability_less_than_25 = 0.66 := by
  sorry

end NUMINAMATH_GPT_probability_student_less_than_25_l1655_165554


namespace NUMINAMATH_GPT_divisors_of_2700_l1655_165528

def prime_factors_2700 : ℕ := 2^2 * 3^3 * 5^2

def number_of_positive_divisors (n : ℕ) (a b c : ℕ) : ℕ :=
  (a + 1) * (b + 1) * (c + 1)

theorem divisors_of_2700 : number_of_positive_divisors 2700 2 3 2 = 36 := by
  sorry

end NUMINAMATH_GPT_divisors_of_2700_l1655_165528


namespace NUMINAMATH_GPT_interval_solution_l1655_165550

-- Let the polynomial be defined
def polynomial (x : ℝ) : ℝ := x^3 - 12 * x^2 + 30 * x

-- Prove the inequality for the specified intervals
theorem interval_solution :
  { x : ℝ | polynomial x > 0 } = { x : ℝ | (0 < x ∧ x < 5) ∨ x > 6 } :=
by
  sorry

end NUMINAMATH_GPT_interval_solution_l1655_165550


namespace NUMINAMATH_GPT_minimum_value_of_f_l1655_165507

noncomputable def f (x : ℝ) : ℝ := (Real.cos (2 * x) + 2 * Real.sin x)

theorem minimum_value_of_f : ∃ x : ℝ, (∀ y : ℝ, f x ≤ f y) ∧ f x = -3 := 
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l1655_165507


namespace NUMINAMATH_GPT_geometric_seq_sum_identity_l1655_165590

noncomputable def is_geometric_seq (a : ℕ → ℝ) : Prop :=
  ∃ q ≠ 0, ∀ n, a (n + 1) = q * a n

theorem geometric_seq_sum_identity (a : ℕ → ℝ) (q : ℝ) (hq : q ≠ 0)
  (hgeom : is_geometric_seq a) 
  (h1 : a 2 + a 6 = 3) 
  (h2 : a 6 + a 10 = 12) : 
  a 8 + a 12 = 24 :=
sorry

end NUMINAMATH_GPT_geometric_seq_sum_identity_l1655_165590


namespace NUMINAMATH_GPT_smallest_square_l1655_165525

theorem smallest_square 
  (a b : ℕ) 
  (h1 : 15 * a + 16 * b = m ^ 2) 
  (h2 : 16 * a - 15 * b = n ^ 2)
  (hm : m > 0) 
  (hn : n > 0) : 
  min (15 * a + 16 * b) (16 * a - 15 * b) = 481 ^ 2 := 
sorry

end NUMINAMATH_GPT_smallest_square_l1655_165525


namespace NUMINAMATH_GPT_multiple_of_cans_of_corn_l1655_165553

theorem multiple_of_cans_of_corn (peas corn : ℕ) (h1 : peas = 35) (h2 : corn = 10) (h3 : peas = 10 * x + 15) : x = 2 := 
by
  sorry

end NUMINAMATH_GPT_multiple_of_cans_of_corn_l1655_165553


namespace NUMINAMATH_GPT_range_of_x_coordinate_l1655_165526

def is_on_line (A : ℝ × ℝ) : Prop := A.1 + A.2 = 6

def is_on_circle (C : ℝ × ℝ) : Prop := (C.1 - 1)^2 + (C.2 - 1)^2 = 4

def angle_BAC_is_60_degrees (A B C : ℝ × ℝ) : Prop :=
  -- This definition is simplified as an explanation. Angle computation in Lean might be more intricate.
  sorry 

theorem range_of_x_coordinate (A : ℝ × ℝ) (B C : ℝ × ℝ)
  (hA_on_line : is_on_line A)
  (hB_on_circle : is_on_circle B)
  (hC_on_circle : is_on_circle C)
  (h_angle_BAC : angle_BAC_is_60_degrees A B C) :
  1 ≤ A.1 ∧ A.1 ≤ 5 :=
sorry

end NUMINAMATH_GPT_range_of_x_coordinate_l1655_165526


namespace NUMINAMATH_GPT_suyeong_ran_distance_l1655_165522

theorem suyeong_ran_distance 
  (circumference : ℝ) 
  (laps : ℕ) 
  (h_circumference : circumference = 242.7)
  (h_laps : laps = 5) : 
  (circumference * laps = 1213.5) := 
  by sorry

end NUMINAMATH_GPT_suyeong_ran_distance_l1655_165522


namespace NUMINAMATH_GPT_car_avg_mpg_B_to_C_is_11_11_l1655_165563

noncomputable def avg_mpg_B_to_C (D : ℝ) : ℝ :=
  let avg_mpg_total := 42.857142857142854
  let x := (100 : ℝ) / 9
  let total_distance := (3 / 2) * D
  let total_gallons := (D / 40) + (D / (2 * x))
  (total_distance / total_gallons)

/-- Prove the car's average miles per gallon from town B to town C is 100/9 mpg. -/
theorem car_avg_mpg_B_to_C_is_11_11 (D : ℝ) (h1 : D > 0):
  avg_mpg_B_to_C D = 100 / 9 :=
by
  sorry

end NUMINAMATH_GPT_car_avg_mpg_B_to_C_is_11_11_l1655_165563


namespace NUMINAMATH_GPT_intersection_sum_x_coordinates_mod_17_l1655_165527

theorem intersection_sum_x_coordinates_mod_17 :
  ∃ x : ℤ, (∃ y₁ y₂ : ℤ, (y₁ ≡ 7 * x + 3 [ZMOD 17]) ∧ (y₂ ≡ 13 * x + 4 [ZMOD 17]))
       ∧ x ≡ 14 [ZMOD 17]  :=
by
  sorry

end NUMINAMATH_GPT_intersection_sum_x_coordinates_mod_17_l1655_165527


namespace NUMINAMATH_GPT_distance_from_pole_to_line_l1655_165547

-- Definitions based on the problem condition
def polar_equation_line (ρ θ : ℝ) := ρ * (Real.cos θ + Real.sin θ) = Real.sqrt 3

-- The statement of the proof problem
theorem distance_from_pole_to_line (ρ θ : ℝ) (h : polar_equation_line ρ θ) :
  ρ = Real.sqrt 6 / 2 := sorry

end NUMINAMATH_GPT_distance_from_pole_to_line_l1655_165547


namespace NUMINAMATH_GPT_bowling_ball_weight_l1655_165584

theorem bowling_ball_weight :
  ∃ (b : ℝ) (c : ℝ),
    8 * b = 5 * c ∧
    4 * c = 100 ∧
    b = 15.625 :=
by 
  sorry

end NUMINAMATH_GPT_bowling_ball_weight_l1655_165584


namespace NUMINAMATH_GPT_integers_multiples_of_d_l1655_165533

theorem integers_multiples_of_d (d m n : ℕ) 
  (h1 : 2 ≤ m) 
  (h2 : 1 ≤ n) 
  (gcd_m_n : Nat.gcd m n = d) 
  (gcd_m_4n1 : Nat.gcd m (4 * n + 1) = 1) : 
  m % d = 0 :=
sorry

end NUMINAMATH_GPT_integers_multiples_of_d_l1655_165533


namespace NUMINAMATH_GPT_ice_cream_ratio_l1655_165580

-- Definitions based on the conditions
def oli_scoops : ℕ := 4
def victoria_scoops : ℕ := oli_scoops + 4

-- Statement to prove the ratio
theorem ice_cream_ratio :
  victoria_scoops / oli_scoops = 2 :=
by
  -- The exact proof strategy here is omitted with 'sorry'
  sorry

end NUMINAMATH_GPT_ice_cream_ratio_l1655_165580


namespace NUMINAMATH_GPT_train_crossing_time_l1655_165552

/-- A train 400 m long traveling at a speed of 36 km/h crosses an electric pole in 40 seconds. -/
theorem train_crossing_time (length : ℝ) (speed_kmph : ℝ) (speed_mps : ℝ) (time : ℝ) 
  (h1 : length = 400)
  (h2 : speed_kmph = 36)
  (h3 : speed_mps = speed_kmph * 1000 / 3600)
  (h4 : time = length / speed_mps) :
  time = 40 :=
by {
  sorry
}

end NUMINAMATH_GPT_train_crossing_time_l1655_165552


namespace NUMINAMATH_GPT_triangle_area_l1655_165517

theorem triangle_area :
  let a := 4
  let c := 5
  let b := Real.sqrt (c^2 - a^2)
  (1 / 2) * a * b = 6 :=
by sorry

end NUMINAMATH_GPT_triangle_area_l1655_165517


namespace NUMINAMATH_GPT_modulus_of_product_l1655_165565

namespace ComplexModule

open Complex

-- Definition of the complex numbers z1 and z2
def z1 : ℂ := 1 + I
def z2 : ℂ := 2 - I

-- Definition of their product z1z2
def z1z2 : ℂ := z1 * z2

-- Statement we need to prove (the modulus of z1z2 is √10)
theorem modulus_of_product : abs z1z2 = Real.sqrt 10 := by
  sorry

end ComplexModule

end NUMINAMATH_GPT_modulus_of_product_l1655_165565


namespace NUMINAMATH_GPT_factorial_mod_prime_l1655_165586
-- Import all necessary libraries

-- State the conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- The main problem statement
theorem factorial_mod_prime (n : ℕ) (h : n = 10) : factorial n % 13 = 7 := by
  sorry

end NUMINAMATH_GPT_factorial_mod_prime_l1655_165586


namespace NUMINAMATH_GPT_pool_depth_multiple_l1655_165571

theorem pool_depth_multiple
  (johns_pool : ℕ)
  (sarahs_pool : ℕ)
  (h1 : johns_pool = 15)
  (h2 : sarahs_pool = 5)
  (h3 : johns_pool = x * sarahs_pool + 5) :
  x = 2 := by
  sorry

end NUMINAMATH_GPT_pool_depth_multiple_l1655_165571


namespace NUMINAMATH_GPT_count_possible_integer_values_l1655_165556

theorem count_possible_integer_values :
  ∃ n : ℕ, (∀ x : ℤ, (25 < x ∧ x < 55) ↔ (26 ≤ x ∧ x ≤ 54)) ∧ n = 29 := by
  sorry

end NUMINAMATH_GPT_count_possible_integer_values_l1655_165556


namespace NUMINAMATH_GPT_solve_problem_l1655_165532

def question : ℝ := -7.8
def answer : ℕ := 22

theorem solve_problem : 2 * (⌊|question|⌋) + (|⌊question⌋|) = answer := by
  sorry

end NUMINAMATH_GPT_solve_problem_l1655_165532


namespace NUMINAMATH_GPT_volume_of_box_l1655_165592

-- Defining the initial parameters of the problem
def length_sheet := 48
def width_sheet := 36
def side_length_cut_square := 3

-- Define the transformed dimensions after squares are cut off
def length_box := length_sheet - 2 * side_length_cut_square
def width_box := width_sheet - 2 * side_length_cut_square
def height_box := side_length_cut_square

-- The target volume of the box
def target_volume := 3780

-- Prove that the volume of the box is equal to the target volume
theorem volume_of_box : length_box * width_box * height_box = target_volume := by
  -- Calculate the expected volume
  -- Expected volume = 42 m * 30 m * 3 m
  -- Which equals 3780 m³
  sorry

end NUMINAMATH_GPT_volume_of_box_l1655_165592


namespace NUMINAMATH_GPT_find_smallest_a_l1655_165511
open Real

noncomputable def a_min := 2 / 9

theorem find_smallest_a (a b c : ℝ)
  (h1 : (1/4, -9/8) = (1/4, a * (1/4) * (1/4) - 9/8))
  (h2 : ∃ n : ℤ, a + b + c = n)
  (h3 : a > 0)
  (h4 : b = - a / 2)
  (h5 : c = a / 16 - 9 / 8): 
  a = a_min :=
by {
  -- Lean code equivalent to the provided mathematical proof will be placed here.
  sorry
}

end NUMINAMATH_GPT_find_smallest_a_l1655_165511


namespace NUMINAMATH_GPT_least_common_multiple_1260_980_l1655_165541

def LCM (a b : ℕ) : ℕ :=
  a * b / Nat.gcd a b

theorem least_common_multiple_1260_980 : LCM 1260 980 = 8820 := by
  sorry

end NUMINAMATH_GPT_least_common_multiple_1260_980_l1655_165541


namespace NUMINAMATH_GPT_base6_addition_problem_l1655_165523

theorem base6_addition_problem (X Y : ℕ) (h1 : Y + 3 = X) (h2 : X + 2 = 7) : X + Y = 7 := 
by
  sorry

end NUMINAMATH_GPT_base6_addition_problem_l1655_165523


namespace NUMINAMATH_GPT_tobias_downloads_l1655_165597

theorem tobias_downloads : 
  ∀ (m : ℕ), (∀ (price_per_app total_spent : ℝ), 
  price_per_app = 2.00 + 2.00 * 0.10 ∧ 
  total_spent = 52.80 → 
  m = total_spent / price_per_app) → 
  m = 24 := 
  sorry

end NUMINAMATH_GPT_tobias_downloads_l1655_165597


namespace NUMINAMATH_GPT_students_more_than_pets_l1655_165587

-- Definitions for the conditions
def number_of_classrooms := 5
def students_per_classroom := 22
def rabbits_per_classroom := 3
def hamsters_per_classroom := 2

-- Total number of students in all classrooms
def total_students := number_of_classrooms * students_per_classroom

-- Total number of pets in all classrooms
def total_pets := number_of_classrooms * (rabbits_per_classroom + hamsters_per_classroom)

-- The theorem to prove
theorem students_more_than_pets : 
  total_students - total_pets = 85 :=
by
  sorry

end NUMINAMATH_GPT_students_more_than_pets_l1655_165587


namespace NUMINAMATH_GPT_ratio_second_to_third_l1655_165530

-- Define the three numbers A, B, C, and their conditions.
variables (A B C : ℕ)

-- Conditions derived from the problem statement.
def sum_condition : Prop := A + B + C = 98
def ratio_condition : Prop := 3 * A = 2 * B
def second_number_value : Prop := B = 30

-- The main theorem stating the problem to prove.
theorem ratio_second_to_third (h1 : sum_condition A B C) (h2 : ratio_condition A B) (h3 : second_number_value B) :
  B = 30 ∧ A = 20 ∧ C = 48 → B / C = 5 / 8 :=
by
  sorry

end NUMINAMATH_GPT_ratio_second_to_third_l1655_165530


namespace NUMINAMATH_GPT_units_digit_of_product_l1655_165583

def is_units_digit (n : ℕ) (d : ℕ) : Prop := n % 10 = d

theorem units_digit_of_product : 
  is_units_digit (6 * 8 * 9 * 10 * 12) 0 := 
by
  sorry

end NUMINAMATH_GPT_units_digit_of_product_l1655_165583


namespace NUMINAMATH_GPT_roger_current_money_l1655_165549

noncomputable def roger_initial_money : ℕ := 16
noncomputable def roger_birthday_money : ℕ := 28
noncomputable def roger_game_spending : ℕ := 25

theorem roger_current_money : roger_initial_money + roger_birthday_money - roger_game_spending = 19 := by
  sorry

end NUMINAMATH_GPT_roger_current_money_l1655_165549


namespace NUMINAMATH_GPT_smallest_flash_drives_l1655_165564

theorem smallest_flash_drives (total_files : ℕ) (flash_drive_space: ℝ)
  (files_size : ℕ → ℝ)
  (h1 : total_files = 40)
  (h2 : flash_drive_space = 2.0)
  (h3 : ∀ n, (n < 4 → files_size n = 1.2) ∧ 
              (4 ≤ n ∧ n < 20 → files_size n = 0.9) ∧ 
              (20 ≤ n → files_size n = 0.6)) :
  ∃ min_flash_drives, min_flash_drives = 20 :=
sorry

end NUMINAMATH_GPT_smallest_flash_drives_l1655_165564


namespace NUMINAMATH_GPT_union_of_complements_l1655_165567

def U : Set ℕ := {x | 0 ≤ x ∧ x < 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {x | x^2 + 4 = 5 * x}
def complement_U (S : Set ℕ) : Set ℕ := {x ∈ U | x ∉ S}

theorem union_of_complements :
  complement_U A ∪ complement_U B = {0, 2, 3, 4, 5} := by
sorry

end NUMINAMATH_GPT_union_of_complements_l1655_165567


namespace NUMINAMATH_GPT_x0_in_M_implies_x0_in_N_l1655_165568

def M : Set ℝ := {x | ∃ (k : ℤ), x = k + 1 / 2}
def N : Set ℝ := {x | ∃ (k : ℤ), x = k / 2 + 1}

theorem x0_in_M_implies_x0_in_N (x0 : ℝ) (h : x0 ∈ M) : x0 ∈ N := 
sorry

end NUMINAMATH_GPT_x0_in_M_implies_x0_in_N_l1655_165568


namespace NUMINAMATH_GPT_min_radius_of_circumcircle_l1655_165509

theorem min_radius_of_circumcircle {a b : ℝ} (ha : a = 3) (hb : b = 4) : 
∃ R : ℝ, R = 2.5 ∧ (∃ c : ℝ, c = Real.sqrt (a^2 + b^2) ∧ a^2 + b^2 = c^2 ∧ 2 * R = c) :=
by 
  sorry

end NUMINAMATH_GPT_min_radius_of_circumcircle_l1655_165509


namespace NUMINAMATH_GPT_min_value_of_m_squared_plus_n_squared_l1655_165529

theorem min_value_of_m_squared_plus_n_squared (m n : ℝ) 
  (h : 4 * m - 3 * n - 5 * Real.sqrt 2 = 0) : m^2 + n^2 = 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_m_squared_plus_n_squared_l1655_165529


namespace NUMINAMATH_GPT_triangle_angle_and_side_ratio_l1655_165596

theorem triangle_angle_and_side_ratio
  (A B C : Real)
  (a b c : Real)
  (h1 : a / Real.sin A = b / Real.sin B)
  (h2 : b / Real.sin B = c / Real.sin C)
  (h3 : (a + c) / b = (Real.sin A - Real.sin B) / (Real.sin A - Real.sin C)) :
  C = Real.pi / 3 ∧ (1 < (a + b) / c ∧ (a + b) / c < 2) :=
by
  sorry


end NUMINAMATH_GPT_triangle_angle_and_side_ratio_l1655_165596


namespace NUMINAMATH_GPT_simplify_expression_l1655_165555

variable (z : ℝ)

theorem simplify_expression: (4 - 5 * z^2) - (2 + 7 * z^2 - z) = 2 - 12 * z^2 + z :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1655_165555


namespace NUMINAMATH_GPT_optimal_selling_price_minimize_loss_l1655_165519

theorem optimal_selling_price_minimize_loss 
  (C : ℝ) (h1 : 17 * C = 720 + 5 * C) 
  (h2 : ∀ x : ℝ, x * (1 - 0.1) = 720 * 0.9)
  (h3 : ∀ y : ℝ, y * (1 + 0.05) = 648 * 1.05)
  (selling_price : ℝ)
  (optimal_selling_price : selling_price = 60) :
  selling_price = C :=
by 
  sorry

end NUMINAMATH_GPT_optimal_selling_price_minimize_loss_l1655_165519


namespace NUMINAMATH_GPT_value_of_expression_l1655_165538

theorem value_of_expression (a b : ℝ) (h : a - b = 1) : a^2 - b^2 - 2 * b = 1 := 
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1655_165538


namespace NUMINAMATH_GPT_tangent_segments_area_l1655_165548

theorem tangent_segments_area (r : ℝ) (l : ℝ) (area : ℝ) :
  r = 4 ∧ l = 6 → area = 9 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_tangent_segments_area_l1655_165548


namespace NUMINAMATH_GPT_atLeastOneNotLessThanTwo_l1655_165569

open Real

theorem atLeastOneNotLessThanTwo (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + 1 / b < 2 ∧ b + 1 / c < 2 ∧ c + 1 / a < 2) → False := 
by
  sorry

end NUMINAMATH_GPT_atLeastOneNotLessThanTwo_l1655_165569


namespace NUMINAMATH_GPT_peter_total_distance_is_six_l1655_165520

def total_distance_covered (d : ℝ) :=
  let first_part_time := (2/3) * d / 4
  let second_part_time := (1/3) * d / 5
  (first_part_time + second_part_time) = 1.4

theorem peter_total_distance_is_six :
  ∃ d : ℝ, total_distance_covered d ∧ d = 6 := 
by
  -- Proof can be filled here
  sorry

end NUMINAMATH_GPT_peter_total_distance_is_six_l1655_165520


namespace NUMINAMATH_GPT_sonia_and_joss_time_spent_moving_l1655_165545

def total_time_spent_moving (fill_time_per_trip drive_time_per_trip trips : ℕ) :=
  (fill_time_per_trip + drive_time_per_trip) * trips

def total_time_in_hours (total_time_in_minutes : ℕ) : ℚ :=
  total_time_in_minutes / 60

theorem sonia_and_joss_time_spent_moving :
  total_time_in_hours (total_time_spent_moving 15 30 6) = 4.5 :=
by
  sorry

end NUMINAMATH_GPT_sonia_and_joss_time_spent_moving_l1655_165545


namespace NUMINAMATH_GPT_chimes_1000_on_march_7_l1655_165598

theorem chimes_1000_on_march_7 : 
  ∀ (initial_time : Nat) (start_date : Nat) (chimes_before_noon : Nat) 
  (chimes_per_day : Nat) (target_chime : Nat) (final_date : Nat),
  initial_time = 10 * 60 + 15 ∧
  start_date = 26 ∧
  chimes_before_noon = 25 ∧
  chimes_per_day = 103 ∧
  target_chime = 1000 ∧
  final_date = start_date + (target_chime - chimes_before_noon) / chimes_per_day ∧
  (target_chime - chimes_before_noon) % chimes_per_day ≤ chimes_per_day
  → final_date = 7 := 
by
  intros
  sorry

end NUMINAMATH_GPT_chimes_1000_on_march_7_l1655_165598


namespace NUMINAMATH_GPT_incenter_ineq_l1655_165560

open Real

-- Definitions of the incenter and angle bisector intersection points
def incenter (A B C : Point) : Point := sorry
def angle_bisector_intersect (A B C I : Point) (angle_vertex : Point) : Point := sorry
def AI (A I : Point) : ℝ := sorry
def AA' (A A' : Point) : ℝ := sorry
def BI (B I : Point) : ℝ := sorry
def BB' (B B' : Point) : ℝ := sorry
def CI (C I : Point) : ℝ := sorry
def CC' (C C' : Point) : ℝ := sorry

-- Statement of the problem
theorem incenter_ineq 
    (A B C I A' B' C' : Point)
    (h1 : I = incenter A B C)
    (h2 : A' = angle_bisector_intersect A B C I A)
    (h3 : B' = angle_bisector_intersect A B C I B)
    (h4 : C' = angle_bisector_intersect A B C I C) :
    (1/4 : ℝ) < (AI A I * BI B I * CI C I) / (AA' A A' * BB' B B' * CC' C C') ∧ 
    (AI A I * BI B I * CI C I) / (AA' A A' * BB' B B' * CC' C C') ≤ (8/27 : ℝ) :=
sorry

end NUMINAMATH_GPT_incenter_ineq_l1655_165560


namespace NUMINAMATH_GPT_circle_center_l1655_165544

theorem circle_center (a b : ℝ)
  (passes_through_point : (a - 0)^2 + (b - 9)^2 = r^2)
  (is_tangent : (a - 3)^2 + (b - 9)^2 = r^2 ∧ b = 6 * (a - 3) + 9 ∧ (b - 9) / (a - 3) = -1/6) :
  a = 3/2 ∧ b = 37/4 := 
by 
  sorry

end NUMINAMATH_GPT_circle_center_l1655_165544


namespace NUMINAMATH_GPT_total_perimeter_l1655_165543

/-- 
A rectangular plot where the long sides are three times the length of the short sides. 
One short side is 80 feet. Prove the total perimeter is 640 feet.
-/
theorem total_perimeter (s : ℕ) (h : s = 80) : 8 * s = 640 :=
  by sorry

end NUMINAMATH_GPT_total_perimeter_l1655_165543


namespace NUMINAMATH_GPT_sandwiches_provided_now_l1655_165514

-- Define the initial number of sandwich kinds
def initialSandwichKinds : ℕ := 23

-- Define the number of sold out sandwich kinds
def soldOutSandwichKinds : ℕ := 14

-- Define the proof that the actual number of sandwich kinds provided now
theorem sandwiches_provided_now : initialSandwichKinds - soldOutSandwichKinds = 9 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_sandwiches_provided_now_l1655_165514


namespace NUMINAMATH_GPT_reading_rate_l1655_165515

-- Definitions based on conditions
def one_way_trip_time : ℕ := 4
def round_trip_time : ℕ := 2 * one_way_trip_time
def read_book_time : ℕ := 2 * round_trip_time
def book_pages : ℕ := 4000

-- The theorem to prove Juan's reading rate is 250 pages per hour.
theorem reading_rate : book_pages / read_book_time = 250 := by
  sorry

end NUMINAMATH_GPT_reading_rate_l1655_165515


namespace NUMINAMATH_GPT_number_of_students_l1655_165503

theorem number_of_students (n : ℕ) (h1 : n < 60) (h2 : n % 6 = 4) (h3 : n % 8 = 5) : n = 46 := by
  sorry

end NUMINAMATH_GPT_number_of_students_l1655_165503
