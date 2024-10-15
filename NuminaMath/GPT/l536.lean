import Mathlib

namespace NUMINAMATH_GPT_problem_a_problem_b_l536_53624

-- Problem a
theorem problem_a (p q : ℕ) (h1 : ∃ n : ℤ, 2 * p - q = n^2) (h2 : ∃ m : ℤ, 2 * p + q = m^2) : ∃ k : ℤ, q = 2 * k :=
sorry

-- Problem b
theorem problem_b (m : ℕ) (h1 : ∃ n : ℕ, 2 * m - 4030 = n^2) (h2 : ∃ k : ℕ, 2 * m + 4030 = k^2) : (m = 2593 ∨ m = 12097 ∨ m = 81217 ∨ m = 2030113) :=
sorry

end NUMINAMATH_GPT_problem_a_problem_b_l536_53624


namespace NUMINAMATH_GPT_percentage_error_in_side_l536_53638

theorem percentage_error_in_side
  (s s' : ℝ) -- the actual and measured side lengths
  (h : (s' * s' - s * s) / (s * s) * 100 = 41.61) : 
  ((s' - s) / s) * 100 = 19 :=
sorry

end NUMINAMATH_GPT_percentage_error_in_side_l536_53638


namespace NUMINAMATH_GPT_quadratic_inequality_range_l536_53665

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Ioc (-2 : ℝ) 2 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_range_l536_53665


namespace NUMINAMATH_GPT_largest_multiple_of_7_smaller_than_neg_50_l536_53604

theorem largest_multiple_of_7_smaller_than_neg_50 : ∃ n, (∃ k : ℤ, n = 7 * k) ∧ n < -50 ∧ ∀ m, (∃ j : ℤ, m = 7 * j) ∧ m < -50 → m ≤ n :=
by
  sorry

end NUMINAMATH_GPT_largest_multiple_of_7_smaller_than_neg_50_l536_53604


namespace NUMINAMATH_GPT_tom_hockey_games_l536_53655

theorem tom_hockey_games (g_this_year g_last_year : ℕ) 
  (h1 : g_this_year = 4)
  (h2 : g_last_year = 9) 
  : g_this_year + g_last_year = 13 := 
by
  sorry

end NUMINAMATH_GPT_tom_hockey_games_l536_53655


namespace NUMINAMATH_GPT_cylinder_volume_ratio_l536_53605

noncomputable def ratio_of_volumes (r h V_small V_large : ℝ) : ℝ := V_large / V_small

theorem cylinder_volume_ratio (r : ℝ) (h : ℝ) 
  (original_height : ℝ := 3 * r)
  (height_small : ℝ := r / 4)
  (height_large : ℝ := 3 * r - height_small)
  (A_small : ℝ := 2 * π * r * (r + height_small))
  (A_large : ℝ := 2 * π * r * (r + height_large))
  (V_small : ℝ := π * r^2 * height_small) 
  (V_large : ℝ := π * r^2 * height_large) :
  A_large = 3 * A_small → 
  ratio_of_volumes r height_small V_small V_large = 11 := by 
  sorry

end NUMINAMATH_GPT_cylinder_volume_ratio_l536_53605


namespace NUMINAMATH_GPT_simplify_sqrt_20_minus_sqrt_5_plus_sqrt_one_fifth_simplify_fraction_of_sqrt_12_plus_sqrt_18_minus_sqrt_half_times_sqrt_3_l536_53630

-- Problem (1)
theorem simplify_sqrt_20_minus_sqrt_5_plus_sqrt_one_fifth :
  (Real.sqrt 20 - Real.sqrt 5 + Real.sqrt (1 / 5) = 6 * Real.sqrt 5 / 5) :=
by
  sorry

-- Problem (2)
theorem simplify_fraction_of_sqrt_12_plus_sqrt_18_minus_sqrt_half_times_sqrt_3 :
  (Real.sqrt 12 + Real.sqrt 18) / Real.sqrt 3 - 2 * Real.sqrt (1 / 2) * Real.sqrt 3 = 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_sqrt_20_minus_sqrt_5_plus_sqrt_one_fifth_simplify_fraction_of_sqrt_12_plus_sqrt_18_minus_sqrt_half_times_sqrt_3_l536_53630


namespace NUMINAMATH_GPT_resistance_per_band_is_10_l536_53686

noncomputable def resistance_per_band := 10
def total_squat_weight := 30
def dumbbell_weight := 10
def number_of_bands := 2

theorem resistance_per_band_is_10 :
  (total_squat_weight - dumbbell_weight) / number_of_bands = resistance_per_band := 
by
  sorry

end NUMINAMATH_GPT_resistance_per_band_is_10_l536_53686


namespace NUMINAMATH_GPT_simplified_polynomial_l536_53659

theorem simplified_polynomial : ∀ (x : ℝ), (3 * x + 2) * (3 * x - 2) - (3 * x - 1) ^ 2 = 6 * x - 5 := by
  sorry

end NUMINAMATH_GPT_simplified_polynomial_l536_53659


namespace NUMINAMATH_GPT_base_six_conversion_addition_l536_53622

def base_six_to_base_ten (n : ℕ) : ℕ :=
  4 * 6^0 + 1 * 6^1 + 2 * 6^2

theorem base_six_conversion_addition : base_six_to_base_ten 214 + 15 = 97 :=
by
  sorry

end NUMINAMATH_GPT_base_six_conversion_addition_l536_53622


namespace NUMINAMATH_GPT_total_children_l536_53645

theorem total_children (sons daughters : ℕ) (h1 : sons = 3) (h2 : daughters = 6 * sons) : (sons + daughters) = 21 :=
by
  sorry

end NUMINAMATH_GPT_total_children_l536_53645


namespace NUMINAMATH_GPT_system_of_equations_solution_l536_53679

theorem system_of_equations_solution :
  ∀ (x : Fin 100 → ℝ), 
  (x 0 + x 1 + x 2 = 0) ∧ 
  (x 1 + x 2 + x 3 = 0) ∧ 
  -- Continue for all other equations up to
  (x 98 + x 99 + x 0 = 0) ∧ 
  (x 99 + x 0 + x 1 = 0)
  → ∀ (i : Fin 100), x i = 0 :=
by
  intros x h
  -- We can insert the detailed solving steps here
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l536_53679


namespace NUMINAMATH_GPT_discount_received_l536_53691

theorem discount_received (original_cost : ℝ) (amt_spent : ℝ) (discount : ℝ) 
  (h1 : original_cost = 467) (h2 : amt_spent = 68) : 
  discount = 399 :=
by
  sorry

end NUMINAMATH_GPT_discount_received_l536_53691


namespace NUMINAMATH_GPT_cost_of_tax_free_items_l536_53687

theorem cost_of_tax_free_items : 
  ∀ (total_spent : ℝ) (sales_tax : ℝ) (tax_rate : ℝ) (taxable_cost : ℝ),
  total_spent = 25 ∧ sales_tax = 0.30 ∧ tax_rate = 0.05 ∧ sales_tax = tax_rate * taxable_cost → 
  total_spent - taxable_cost = 19 :=
by
  intros total_spent sales_tax tax_rate taxable_cost
  intro h
  sorry

end NUMINAMATH_GPT_cost_of_tax_free_items_l536_53687


namespace NUMINAMATH_GPT_algebraic_expression_value_l536_53690

theorem algebraic_expression_value (x : ℝ) (h : x ^ 2 - 3 * x = 4) : 2 * x ^ 2 - 6 * x - 3 = 5 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l536_53690


namespace NUMINAMATH_GPT_domain_of_sqrt_expr_l536_53652

theorem domain_of_sqrt_expr (x : ℝ) : x ≥ 3 ∧ x < 8 ↔ x ∈ Set.Ico 3 8 :=
by
  sorry

end NUMINAMATH_GPT_domain_of_sqrt_expr_l536_53652


namespace NUMINAMATH_GPT_megan_roles_other_than_lead_l536_53698

def total_projects : ℕ := 800

def theater_percentage : ℚ := 50 / 100
def films_percentage : ℚ := 30 / 100
def television_percentage : ℚ := 20 / 100

def theater_lead_percentage : ℚ := 55 / 100
def theater_support_percentage : ℚ := 30 / 100
def theater_ensemble_percentage : ℚ := 10 / 100
def theater_cameo_percentage : ℚ := 5 / 100

def films_lead_percentage : ℚ := 70 / 100
def films_support_percentage : ℚ := 20 / 100
def films_minor_percentage : ℚ := 7 / 100
def films_cameo_percentage : ℚ := 3 / 100

def television_lead_percentage : ℚ := 60 / 100
def television_support_percentage : ℚ := 25 / 100
def television_recurring_percentage : ℚ := 10 / 100
def television_guest_percentage : ℚ := 5 / 100

theorem megan_roles_other_than_lead :
  let theater_projects := total_projects * theater_percentage
  let films_projects := total_projects * films_percentage
  let television_projects := total_projects * television_percentage

  let theater_other_roles := (theater_projects * theater_support_percentage) + 
                             (theater_projects * theater_ensemble_percentage) + 
                             (theater_projects * theater_cameo_percentage)

  let films_other_roles := (films_projects * films_support_percentage) + 
                           (films_projects * films_minor_percentage) + 
                           (films_projects * films_cameo_percentage)

  let television_other_roles := (television_projects * television_support_percentage) + 
                                (television_projects * television_recurring_percentage) + 
                                (television_projects * television_guest_percentage)
  
  theater_other_roles + films_other_roles + television_other_roles = 316 :=
by
  sorry

end NUMINAMATH_GPT_megan_roles_other_than_lead_l536_53698


namespace NUMINAMATH_GPT_arithmetic_sequence_min_value_Sn_l536_53631

-- Define the sequence a_n and the sum S_n
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- The given condition
axiom condition : ∀ n : ℕ, n > 0 → (2 * S n / n) + n = 2 * a n + 1

-- Arithmetic sequence proof
theorem arithmetic_sequence : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 1 :=
by sorry

-- Minimum value of S_n when a_4, a_7, a_9 are geometric
theorem min_value_Sn (G : ℝ) (h : a 4 * a 9 = a 7 ^ 2) : ∃ n : ℕ, S n = -78 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_min_value_Sn_l536_53631


namespace NUMINAMATH_GPT_jeff_total_jars_l536_53692

theorem jeff_total_jars (x : ℕ) : 
  16 * x + 28 * x + 40 * x + 52 * x = 2032 → 4 * x = 56 :=
by
  intro h
  -- additional steps to solve the problem would go here.
  sorry

end NUMINAMATH_GPT_jeff_total_jars_l536_53692


namespace NUMINAMATH_GPT_mutually_exclusive_but_not_complementary_l536_53661

open Classical

namespace CardDistribution

inductive Card
| red | yellow | blue | white

inductive Person
| A | B | C | D

def Event_A_gets_red (distrib: Person → Card) : Prop :=
  distrib Person.A = Card.red

def Event_D_gets_red (distrib: Person → Card) : Prop :=
  distrib Person.D = Card.red

theorem mutually_exclusive_but_not_complementary :
  ∀ (distrib: Person → Card),
  (Event_A_gets_red distrib → ¬Event_D_gets_red distrib) ∧
  ¬(∀ (distrib: Person → Card), Event_A_gets_red distrib ∨ Event_D_gets_red distrib) := 
by
  sorry

end CardDistribution

end NUMINAMATH_GPT_mutually_exclusive_but_not_complementary_l536_53661


namespace NUMINAMATH_GPT_range_of_3a_minus_2b_l536_53639

theorem range_of_3a_minus_2b (a b : ℝ) (h1 : 1 ≤ a - b ∧ a - b ≤ 2) (h2 : 2 ≤ a + b ∧ a + b ≤ 4) :
  7 / 2 ≤ 3 * a - 2 * b ∧ 3 * a - 2 * b ≤ 7 :=
sorry

end NUMINAMATH_GPT_range_of_3a_minus_2b_l536_53639


namespace NUMINAMATH_GPT_min_value_of_quadratic_l536_53651

theorem min_value_of_quadratic (y1 y2 y3 : ℝ) (h1 : 0 < y1) (h2 : 0 < y2) (h3 : 0 < y3) (h_eq : 2 * y1 + 3 * y2 + 4 * y3 = 75) :
  y1^2 + 2 * y2^2 + 3 * y3^2 ≥ 5625 / 29 :=
sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l536_53651


namespace NUMINAMATH_GPT_at_least_one_solves_l536_53644

-- Given probabilities
def pA : ℝ := 0.8
def pB : ℝ := 0.6

-- Probability that at least one solves the problem
def prob_at_least_one_solves : ℝ := 1 - ((1 - pA) * (1 - pB))

-- Statement: Prove that the probability that at least one solves the problem is 0.92
theorem at_least_one_solves : prob_at_least_one_solves = 0.92 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_at_least_one_solves_l536_53644


namespace NUMINAMATH_GPT_map_distance_ratio_l536_53634

theorem map_distance_ratio (actual_distance_km : ℝ) (map_distance_cm : ℝ) (h_actual_distance : actual_distance_km = 5) (h_map_distance : map_distance_cm = 2) :
  map_distance_cm / (actual_distance_km * 100000) = 1 / 250000 :=
by
  -- Given the actual distance in kilometers and map distance in centimeters, prove the scale ratio
  -- skip the proof
  sorry

end NUMINAMATH_GPT_map_distance_ratio_l536_53634


namespace NUMINAMATH_GPT_plane_speeds_l536_53618

-- Define the speeds of the planes
def speed_slower (x : ℕ) := x
def speed_faster (x : ℕ) := 2 * x

-- Define the distances each plane travels in 3 hours
def distance_slower (x : ℕ) := 3 * speed_slower x
def distance_faster (x : ℕ) := 3 * speed_faster x

-- Define the total distance
def total_distance (x : ℕ) := distance_slower x + distance_faster x

-- Prove the speeds given the total distance
theorem plane_speeds (x : ℕ) (h : total_distance x = 2700) : speed_slower x = 300 ∧ speed_faster x = 600 :=
by {
  sorry
}

end NUMINAMATH_GPT_plane_speeds_l536_53618


namespace NUMINAMATH_GPT_marble_count_l536_53612

theorem marble_count (x : ℕ) 
  (h1 : ∀ (Liam Mia Noah Olivia: ℕ), Mia = 3 * Liam ∧ Noah = 4 * Mia ∧ Olivia = 2 * Noah)
  (h2 : Liam + Mia + Noah + Olivia = 156)
  : x = 4 :=
by sorry

end NUMINAMATH_GPT_marble_count_l536_53612


namespace NUMINAMATH_GPT_g_at_4_l536_53678

def g (x : ℝ) : ℝ := 5 * x + 6

theorem g_at_4 : g 4 = 26 :=
by
  sorry

end NUMINAMATH_GPT_g_at_4_l536_53678


namespace NUMINAMATH_GPT_max_brownie_pieces_l536_53671

theorem max_brownie_pieces (base height piece_width piece_height : ℕ) 
    (h_base : base = 30) (h_height : height = 24)
    (h_piece_width : piece_width = 3) (h_piece_height : piece_height = 4) :
  (base / piece_width) * (height / piece_height) = 60 :=
by sorry

end NUMINAMATH_GPT_max_brownie_pieces_l536_53671


namespace NUMINAMATH_GPT_count_non_squares_or_cubes_l536_53667

theorem count_non_squares_or_cubes (n : ℕ) (h₀ : 1 ≤ n ∧ n ≤ 200) : 
  ∃ c, c = 182 ∧ 
  (∃ k, k^2 = n ∨ ∃ m, m^3 = n) → false :=
by
  sorry

end NUMINAMATH_GPT_count_non_squares_or_cubes_l536_53667


namespace NUMINAMATH_GPT_graph_is_two_lines_l536_53608

theorem graph_is_two_lines (x y : ℝ) : (x^2 - 25 * y^2 - 10 * x + 50 = 0) ↔
  (x = 5 + 5 * y) ∨ (x = 5 - 5 * y) :=
by
  sorry

end NUMINAMATH_GPT_graph_is_two_lines_l536_53608


namespace NUMINAMATH_GPT_symmetric_points_sum_l536_53602

theorem symmetric_points_sum (a b : ℝ) (h1 : B = (-A)) (h2 : A = (1, a)) (h3 : B = (b, 2)) : a + b = -3 := by
  sorry

end NUMINAMATH_GPT_symmetric_points_sum_l536_53602


namespace NUMINAMATH_GPT_distribute_balls_in_boxes_l536_53633

theorem distribute_balls_in_boxes :
  let balls := 5
  let boxes := 4
  (4 ^ balls) = 1024 :=
by
  sorry

end NUMINAMATH_GPT_distribute_balls_in_boxes_l536_53633


namespace NUMINAMATH_GPT_y_coord_vertex_C_l536_53674

/-- The coordinates of vertices A, B, and D are given as A(0,0), B(0,1), and D(3,1).
 Vertex C is directly above vertex B. The quadrilateral ABCD has a vertical line of symmetry 
 and the area of quadrilateral ABCD is 18 square units.
 Prove that the y-coordinate of vertex C is 11. -/
theorem y_coord_vertex_C (h : ℝ) 
  (A : ℝ × ℝ := (0, 0)) 
  (B : ℝ × ℝ := (0, 1)) 
  (D : ℝ × ℝ := (3, 1)) 
  (C : ℝ × ℝ := (0, h)) 
  (symmetry : C.fst = B.fst) 
  (area : 18 = 3 * 1 + (1 / 2) * 3 * (h - 1)) :
  h = 11 := 
by
  sorry

end NUMINAMATH_GPT_y_coord_vertex_C_l536_53674


namespace NUMINAMATH_GPT_total_trip_cost_l536_53694

def distance_AC : ℝ := 4000
def distance_AB : ℝ := 4250
def bus_rate : ℝ := 0.10
def plane_rate : ℝ := 0.15
def boarding_fee : ℝ := 150

theorem total_trip_cost :
  let distance_BC := Real.sqrt (distance_AB ^ 2 - distance_AC ^ 2)
  let flight_cost := distance_AB * plane_rate + boarding_fee
  let bus_cost := distance_BC * bus_rate
  flight_cost + bus_cost = 931.15 :=
by
  sorry

end NUMINAMATH_GPT_total_trip_cost_l536_53694


namespace NUMINAMATH_GPT_product_positivity_l536_53669

theorem product_positivity (a b c d e f : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0)
  (h : ∀ {x y z u v w : ℝ}, x * y * z > 0 → ¬(u * v * w > 0) ∧ ¬(x * v * u > 0) ∧ ¬(y * z * u > 0) ∧ ¬(y * v * x > 0)) :
  b * d * e > 0 ∧ a * c * d < 0 ∧ a * c * e < 0 ∧ b * d * f < 0 ∧ b * e * f < 0 := 
by
  sorry

end NUMINAMATH_GPT_product_positivity_l536_53669


namespace NUMINAMATH_GPT_mark_has_3_tanks_l536_53682

-- Define conditions
def pregnant_fish_per_tank : ℕ := 4
def young_per_fish : ℕ := 20
def total_young : ℕ := 240

-- Theorem statement that Mark has 3 tanks
theorem mark_has_3_tanks : (total_young / (pregnant_fish_per_tank * young_per_fish)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_mark_has_3_tanks_l536_53682


namespace NUMINAMATH_GPT_athlete_difference_l536_53619

-- Define the conditions
def initial_athletes : ℕ := 300
def rate_of_leaving : ℕ := 28
def time_of_leaving : ℕ := 4
def rate_of_arriving : ℕ := 15
def time_of_arriving : ℕ := 7

-- Define intermediary calculations
def number_leaving : ℕ := rate_of_leaving * time_of_leaving
def remaining_athletes : ℕ := initial_athletes - number_leaving
def number_arriving : ℕ := rate_of_arriving * time_of_arriving
def total_sunday_night : ℕ := remaining_athletes + number_arriving

-- Theorem statement
theorem athlete_difference : initial_athletes - total_sunday_night = 7 :=
by
  sorry

end NUMINAMATH_GPT_athlete_difference_l536_53619


namespace NUMINAMATH_GPT_lines_intersection_points_l536_53664

theorem lines_intersection_points :
  let line1 (x y : ℝ) := 2 * y - 3 * x = 4
  let line2 (x y : ℝ) := 3 * x + y = 5
  let line3 (x y : ℝ) := 6 * x - 4 * y = 8
  ∃ p1 p2 : (ℝ × ℝ),
    (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧
    (line2 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧
    (p1 = (2, 5)) ∧ (p2 = (14/9, 1/3)) :=
by
  sorry

end NUMINAMATH_GPT_lines_intersection_points_l536_53664


namespace NUMINAMATH_GPT_cost_for_23_days_l536_53636

structure HostelStay where
  charge_first_week : ℝ
  charge_additional_week : ℝ

def cost_of_stay (days : ℕ) (hostel : HostelStay) : ℝ :=
  let first_week_days := min days 7
  let remaining_days := days - first_week_days
  let additional_full_weeks := remaining_days / 7 
  let additional_days := remaining_days % 7
  (first_week_days * hostel.charge_first_week) + 
  (additional_full_weeks * 7 * hostel.charge_additional_week) + 
  (additional_days * hostel.charge_additional_week)

theorem cost_for_23_days :
  cost_of_stay 23 { charge_first_week := 18.00, charge_additional_week := 11.00 } = 302.00 :=
by
  sorry

end NUMINAMATH_GPT_cost_for_23_days_l536_53636


namespace NUMINAMATH_GPT_greatest_x_l536_53680

theorem greatest_x (x : ℕ) (h : x^2 < 32) : x ≤ 5 := 
sorry

end NUMINAMATH_GPT_greatest_x_l536_53680


namespace NUMINAMATH_GPT_evaluate_m_l536_53632

theorem evaluate_m (m : ℕ) : 2 ^ m = (64 : ℝ) ^ (1 / 3) → m = 2 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_m_l536_53632


namespace NUMINAMATH_GPT_combine_exponent_remains_unchanged_l536_53656

-- Define combining like terms condition
def combining_like_terms (terms : List (ℕ × String)) : List (ℕ × String) := sorry

-- Define the problem statement
theorem combine_exponent_remains_unchanged (terms : List (ℕ × String)) : 
  (combining_like_terms terms).map Prod.snd = terms.map Prod.snd :=
sorry

end NUMINAMATH_GPT_combine_exponent_remains_unchanged_l536_53656


namespace NUMINAMATH_GPT_andy_demerits_for_joke_l536_53611

def max_demerits := 50
def demerits_late_per_instance := 2
def instances_late := 6
def remaining_demerits := 23
def total_demerits := max_demerits - remaining_demerits
def demerits_late := demerits_late_per_instance * instances_late
def demerits_joke := total_demerits - demerits_late

theorem andy_demerits_for_joke : demerits_joke = 15 := by
  sorry

end NUMINAMATH_GPT_andy_demerits_for_joke_l536_53611


namespace NUMINAMATH_GPT_line_AB_l536_53629

-- Statements for circles and intersection
def circle_C1 (x y: ℝ) : Prop := x^2 + y^2 = 1
def circle_C2 (x y: ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 1

-- Points A and B are defined as the intersection points of circles C1 and C2
axiom A (x y: ℝ) : circle_C1 x y ∧ circle_C2 x y
axiom B (x y: ℝ) : circle_C1 x y ∧ circle_C2 x y

-- The goal is to prove that the line passing through points A and B has the equation x - y = 0
theorem line_AB (x y: ℝ) : circle_C1 x y → circle_C2 x y → (x - y = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_AB_l536_53629


namespace NUMINAMATH_GPT_degree_of_expression_l536_53675

open Polynomial

noncomputable def expr1 : Polynomial ℤ := (monomial 5 3 - monomial 3 2 + 4) * (monomial 12 2 - monomial 8 1 + monomial 6 5 - 15)
noncomputable def expr2 : Polynomial ℤ := (monomial 3 2 - 4) ^ 6
noncomputable def final_expr : Polynomial ℤ := expr1 - expr2

theorem degree_of_expression : degree final_expr = 18 := by
  sorry

end NUMINAMATH_GPT_degree_of_expression_l536_53675


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l536_53668

open Real

-- Define α as an internal angle of a triangle
def is_internal_angle (α : ℝ) : Prop := (0 < α ∧ α < π)

-- Given conditions
axiom α : ℝ
axiom h1 : is_internal_angle α

-- Prove: if (α ≠ π / 6) then (sin α ≠ 1 / 2) is a necessary but not sufficient condition 
theorem necessary_but_not_sufficient_condition : 
  (α ≠ π / 6) ∧ ¬((α ≠ π / 6) → (sin α ≠ 1 / 2)) ∧ ((sin α ≠ 1 / 2) → (α ≠ π / 6)) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l536_53668


namespace NUMINAMATH_GPT_current_tree_height_in_inches_l536_53614

-- Constants
def initial_height_ft : ℝ := 10
def growth_percentage : ℝ := 0.50
def feet_to_inches : ℝ := 12

-- Conditions
def growth_ft : ℝ := growth_percentage * initial_height_ft
def current_height_ft : ℝ := initial_height_ft + growth_ft

-- Question/Answer equivalence
theorem current_tree_height_in_inches :
  (current_height_ft * feet_to_inches) = 180 :=
by 
  sorry

end NUMINAMATH_GPT_current_tree_height_in_inches_l536_53614


namespace NUMINAMATH_GPT_equal_roots_quadratic_eq_l536_53657

theorem equal_roots_quadratic_eq (m n : ℝ) (h : m^2 - 4 * n = 0) : m = 2 ∧ n = 1 :=
by
  sorry

end NUMINAMATH_GPT_equal_roots_quadratic_eq_l536_53657


namespace NUMINAMATH_GPT_multiple_of_shorter_piece_l536_53683

theorem multiple_of_shorter_piece :
  ∃ (m : ℕ), 
  (35 + (m * 35 + 15) = 120) ∧
  (m = 2) :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_shorter_piece_l536_53683


namespace NUMINAMATH_GPT_arithmetic_sequence_general_term_geometric_sequence_inequality_l536_53640

-- Sequence {a_n} and its sum S_n
def S (a : ℕ → ℤ) (n : ℕ) : ℤ := (Finset.range n).sum a

-- Sequence {b_n}
def b (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  2 * (S a (n + 1) - S a n) * (S a n) - n * (S a (n + 1) + S a n)

-- Arithmetic sequence and related conditions
theorem arithmetic_sequence_general_term (a : ℕ → ℤ) (d : ℤ)
  (h1 : ∀ n, a n = a 0 + n * d) (h2 : ∀ n, b a n = 0) :
  (∀ n, a n = 0) ∨ (∀ n, a n = n) :=
sorry

-- Conditions for sequences and finding the set of positive integers n
theorem geometric_sequence_inequality (a : ℕ → ℤ)
  (h1 : a 1 = 1) (h2 : a 2 = 3)
  (h3 : ∀ n, a (2 * n - 1) = 2^(n-1))
  (h4 : ∀ n, a (2 * n) = 3 * 2^(n-1)) :
  {n : ℕ | b a (2 * n) < b a (2 * n - 1)} = {1, 2, 3, 4, 5, 6} :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_term_geometric_sequence_inequality_l536_53640


namespace NUMINAMATH_GPT_find_interest_rate_l536_53642

theorem find_interest_rate
  (P : ℝ) (CI : ℝ) (T : ℝ) (n : ℕ)
  (comp_int_formula : CI = P * ((1 + (r / (n : ℝ))) ^ (n * T)) - P) :
  r = 0.099 :=
by
  have h : CI = 788.13 := sorry
  have hP : P = 5000 := sorry
  have hT : T = 1.5 := sorry
  have hn : (n : ℝ) = 2 := sorry
  sorry

end NUMINAMATH_GPT_find_interest_rate_l536_53642


namespace NUMINAMATH_GPT_bamboo_break_height_l536_53658

theorem bamboo_break_height (x : ℝ) (h₁ : 0 < x) (h₂ : x < 9) (h₃ : x^2 + 3^2 = (9 - x)^2) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_bamboo_break_height_l536_53658


namespace NUMINAMATH_GPT_tangent_circle_radius_l536_53663

theorem tangent_circle_radius (r1 r2 d : ℝ) (h1 : r2 = 2) (h2 : d = 5) (tangent : abs (r1 - r2) = d ∨ r1 + r2 = d) :
  r1 = 3 ∨ r1 = 7 :=
by
  sorry

end NUMINAMATH_GPT_tangent_circle_radius_l536_53663


namespace NUMINAMATH_GPT_seeds_planted_on_wednesday_l536_53615

theorem seeds_planted_on_wednesday
  (total_seeds : ℕ) (seeds_thursday : ℕ) (seeds_wednesday : ℕ)
  (h_total : total_seeds = 22) (h_thursday : seeds_thursday = 2) :
  seeds_wednesday = 20 ↔ total_seeds - seeds_thursday = seeds_wednesday :=
by
  -- the proof would go here
  sorry

end NUMINAMATH_GPT_seeds_planted_on_wednesday_l536_53615


namespace NUMINAMATH_GPT_amount_paid_for_peaches_l536_53627

noncomputable def cost_of_berries : ℝ := 7.19
noncomputable def change_received : ℝ := 5.98
noncomputable def total_bill : ℝ := 20

theorem amount_paid_for_peaches :
  total_bill - change_received - cost_of_berries = 6.83 :=
by
  sorry

end NUMINAMATH_GPT_amount_paid_for_peaches_l536_53627


namespace NUMINAMATH_GPT_pigeons_on_branches_and_under_tree_l536_53643

theorem pigeons_on_branches_and_under_tree (x y : ℕ) 
  (h1 : y - 1 = (x + 1) / 2)
  (h2 : x - 1 = y + 1) : x = 7 ∧ y = 5 :=
by
  sorry

end NUMINAMATH_GPT_pigeons_on_branches_and_under_tree_l536_53643


namespace NUMINAMATH_GPT_alcohol_percentage_correct_in_mixed_solution_l536_53617

-- Define the ratios of alcohol to water
def ratio_A : ℚ := 21 / 25
def ratio_B : ℚ := 2 / 5

-- Define the mixing ratio of solutions A and B
def mix_ratio_A : ℚ := 5 / 11
def mix_ratio_B : ℚ := 6 / 11

-- Define the function to compute the percentage of alcohol in the mixed solution
def alcohol_percentage_mixed : ℚ := 
  (mix_ratio_A * ratio_A + mix_ratio_B * ratio_B) * 100

-- The theorem to be proven
theorem alcohol_percentage_correct_in_mixed_solution : 
  alcohol_percentage_mixed = 60 :=
by
  sorry

end NUMINAMATH_GPT_alcohol_percentage_correct_in_mixed_solution_l536_53617


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l536_53689

noncomputable def condition_sufficiency (m : ℝ) : Prop :=
  ∀ (x : ℝ), x^2 + m*x + 1 > 0

theorem necessary_but_not_sufficient_condition (m : ℝ) : m < 2 → (¬ condition_sufficiency m ∨ condition_sufficiency m) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l536_53689


namespace NUMINAMATH_GPT_carnations_in_last_three_bouquets_l536_53666

/--
Trevor buys six bouquets of carnations.
In the first bouquet, there are 9.5 carnations.
In the second bouquet, there are 14.25 carnations.
In the third bouquet, there are 18.75 carnations.
The average number of carnations in all six bouquets is 16.
Prove that the total number of carnations in the fourth, fifth, and sixth bouquets combined is 53.5.
-/
theorem carnations_in_last_three_bouquets:
  let bouquet1 := 9.5
  let bouquet2 := 14.25
  let bouquet3 := 18.75
  let total_bouquets := 6
  let average_per_bouquet := 16
  let total_carnations := average_per_bouquet * total_bouquets
  let remaining_carnations := total_carnations - (bouquet1 + bouquet2 + bouquet3)
  remaining_carnations = 53.5 :=
by
  sorry

end NUMINAMATH_GPT_carnations_in_last_three_bouquets_l536_53666


namespace NUMINAMATH_GPT_find_non_divisible_and_product_l536_53650

-- Define the set of numbers
def numbers : List Nat := [3543, 3552, 3567, 3579, 3581]

-- Function to get the digits of a number
def digits (n : Nat) : List Nat := n.digits 10

-- Function to sum the digits
def sum_of_digits (n : Nat) : Nat := (digits n).sum

-- Function to check divisibility by 3
def divisible_by_3 (n : Nat) : Bool := sum_of_digits n % 3 = 0

-- Find the units digit of a number
def units_digit (n : Nat) : Nat := n % 10

-- Find the tens digit of a number
def tens_digit (n : Nat) : Nat := (n / 10) % 10

-- The problem statement
theorem find_non_divisible_and_product :
  ∃ n ∈ numbers, ¬ divisible_by_3 n ∧ units_digit n * tens_digit n = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_non_divisible_and_product_l536_53650


namespace NUMINAMATH_GPT_ceil_floor_difference_l536_53695

theorem ceil_floor_difference : 
  (Int.ceil ((15 : ℚ) / 8 * ((-34 : ℚ) / 4)) - Int.floor (((15 : ℚ) / 8) * Int.ceil ((-34 : ℚ) / 4)) = 0) :=
by 
  sorry

end NUMINAMATH_GPT_ceil_floor_difference_l536_53695


namespace NUMINAMATH_GPT_find_d_l536_53623

theorem find_d :
  ∃ d : ℝ, ∀ x : ℝ, x * (4 * x - 3) < d ↔ - (9/4 : ℝ) < x ∧ x < (3/2 : ℝ) ∧ d = 27 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l536_53623


namespace NUMINAMATH_GPT_doritos_ratio_l536_53610

noncomputable def bags_of_chips : ℕ := 80
noncomputable def bags_per_pile : ℕ := 5
noncomputable def piles : ℕ := 4

theorem doritos_ratio (D T : ℕ) (h1 : T = bags_of_chips)
  (h2 : D = piles * bags_per_pile) :
  (D : ℚ) / T = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_doritos_ratio_l536_53610


namespace NUMINAMATH_GPT_complex_number_real_imaginary_opposite_l536_53620

theorem complex_number_real_imaginary_opposite (a : ℝ) (i : ℂ) (comp : z = (1 - a * i) * i):
  (z.re = -z.im) → a = 1 :=
by 
  sorry

end NUMINAMATH_GPT_complex_number_real_imaginary_opposite_l536_53620


namespace NUMINAMATH_GPT_find_a_l536_53616

-- Definitions for conditions
def line_equation (a : ℝ) (x y : ℝ) := a * x - y - 1 = 0
def angle_of_inclination (θ : ℝ) := θ = Real.pi / 3

-- The main theorem statement
theorem find_a (a : ℝ) (θ : ℝ) (h1 : angle_of_inclination θ) (h2 : a = Real.tan θ) : a = Real.sqrt 3 :=
 by
   -- skipping the proof
   sorry

end NUMINAMATH_GPT_find_a_l536_53616


namespace NUMINAMATH_GPT_tyler_common_ratio_l536_53672

theorem tyler_common_ratio (a r : ℝ) 
  (h1 : a / (1 - r) = 10)
  (h2 : (a + 4) / (1 - r) = 15) : 
  r = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_tyler_common_ratio_l536_53672


namespace NUMINAMATH_GPT_part_a_part_b_l536_53621

-- Part (a): Proving at most one integer solution for general k
theorem part_a (k : ℤ) : 
  ∀ (x1 x2 : ℤ), (x1^3 - 24*x1 + k = 0 ∧ x2^3 - 24*x2 + k = 0) → x1 = x2 :=
sorry

-- Part (b): Proving exactly one integer solution for k = -2016
theorem part_b :
  ∃! (x : ℤ), x^3 + 24*x - 2016 = 0 :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l536_53621


namespace NUMINAMATH_GPT_johns_age_l536_53646

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end NUMINAMATH_GPT_johns_age_l536_53646


namespace NUMINAMATH_GPT_sqrt_E_nature_l536_53641

def E (x : ℤ) : ℤ :=
  let a := x
  let b := x + 1
  let c := a * b
  let d := b * c
  a^2 + b^2 + c^2 + d^2

theorem sqrt_E_nature : ∀ x : ℤ, (∃ n : ℤ, n^2 = E x) ∧ (∃ m : ℤ, m^2 ≠ E x) :=
  by
  sorry

end NUMINAMATH_GPT_sqrt_E_nature_l536_53641


namespace NUMINAMATH_GPT_marks_difference_l536_53625

theorem marks_difference (A B C D E : ℕ) 
  (h1 : (A + B + C) / 3 = 48) 
  (h2 : (A + B + C + D) / 4 = 47) 
  (h3 : E > D) 
  (h4 : (B + C + D + E) / 4 = 48) 
  (h5 : A = 43) : 
  E - D = 3 := 
sorry

end NUMINAMATH_GPT_marks_difference_l536_53625


namespace NUMINAMATH_GPT_price_per_bottle_is_half_l536_53626

theorem price_per_bottle_is_half (P : ℚ) 
  (Remy_bottles_morning : ℕ) (Nick_bottles_morning : ℕ) 
  (Total_sales_evening : ℚ) (Evening_more : ℚ) : 
  Remy_bottles_morning = 55 → 
  Nick_bottles_morning = Remy_bottles_morning - 6 → 
  Total_sales_evening = 55 → 
  Evening_more = 3 → 
  104 * P + 3 = 55 → 
  P = 1 / 2 := 
by
  intros h_remy_55 h_nick_remy h_total_55 h_evening_3 h_sales_eq
  sorry

end NUMINAMATH_GPT_price_per_bottle_is_half_l536_53626


namespace NUMINAMATH_GPT_ratatouille_cost_per_quart_l536_53637

def eggplants := 88 * 0.22
def zucchini := 60.8 * 0.15
def tomatoes := 73.6 * 0.25
def onions := 43.2 * 0.07
def basil := (16 / 4) * 2.70
def bell_peppers := 12 * 0.20

def total_cost := eggplants + zucchini + tomatoes + onions + basil + bell_peppers
def yield := 4.5

def cost_per_quart := total_cost / yield

theorem ratatouille_cost_per_quart : cost_per_quart = 14.02 := 
by
  unfold cost_per_quart total_cost eggplants zucchini tomatoes onions basil bell_peppers 
  sorry

end NUMINAMATH_GPT_ratatouille_cost_per_quart_l536_53637


namespace NUMINAMATH_GPT_tank_capacity_l536_53603

variable (C : ℝ)  -- total capacity of the tank

-- The tank is 5/8 full initially
axiom h1 : (5/8) * C + 15 = (19/24) * C

theorem tank_capacity : C = 90 :=
by
  sorry

end NUMINAMATH_GPT_tank_capacity_l536_53603


namespace NUMINAMATH_GPT_boat_downstream_distance_l536_53647

theorem boat_downstream_distance (V_b V_s : ℝ) (t_downstream t_upstream : ℝ) (d_upstream : ℝ) 
  (h1 : t_downstream = 8) (h2 : t_upstream = 15) (h3 : d_upstream = 75) (h4 : V_s = 3.75) 
  (h5 : V_b - V_s = (d_upstream / t_upstream)) : (V_b + V_s) * t_downstream = 100 :=
by
  sorry

end NUMINAMATH_GPT_boat_downstream_distance_l536_53647


namespace NUMINAMATH_GPT_george_speed_l536_53606

theorem george_speed : 
  ∀ (d_tot d_1st : ℝ) (v_tot v_1st : ℝ) (v_2nd : ℝ),
    d_tot = 1 ∧ d_1st = 1 / 2 ∧ v_tot = 3 ∧ v_1st = 2 ∧ ((d_tot / v_tot) = (d_1st / v_1st + d_1st / v_2nd)) →
    v_2nd = 6 :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_george_speed_l536_53606


namespace NUMINAMATH_GPT_non_divisible_by_twenty_l536_53677

theorem non_divisible_by_twenty (k : ℤ) (h : ∃ m : ℤ, k * (k + 1) * (k + 2) = 5 * m) :
  ¬ (∃ l : ℤ, k * (k + 1) * (k + 2) = 20 * l) := sorry

end NUMINAMATH_GPT_non_divisible_by_twenty_l536_53677


namespace NUMINAMATH_GPT_curves_intersect_at_four_points_l536_53684

theorem curves_intersect_at_four_points (a : ℝ) :
  (∃ x y : ℝ, (x^2 + y^2 = a^2 ∧ y = -x^2 + a ) ∧ 
   (0 = x ∧ y = a) ∧ 
   (∃ t : ℝ, x = t ∧ (y = 1 ∧ x^2 = a - 1))) ↔ a = 2 := 
by
  sorry

end NUMINAMATH_GPT_curves_intersect_at_four_points_l536_53684


namespace NUMINAMATH_GPT_minimum_value_fraction_l536_53670

theorem minimum_value_fraction (a b : ℝ) (h1 : a > 1) (h2 : b > 2) (h3 : 2 * a + b - 6 = 0) :
  (1 / (a - 1) + 2 / (b - 2)) = 4 := 
  sorry

end NUMINAMATH_GPT_minimum_value_fraction_l536_53670


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l536_53653

variable (a : ℝ)

def quadratic_solution_set_1 := {x : ℝ | x^2 + 2*x + a = 0}
def quadratic_solution_set_2 := {x : ℝ | a*x^2 + 2*x + 2 = 0}

theorem problem_part1 :
  (quadratic_solution_set_1 a = ∅ ∨ quadratic_solution_set_2 a = ∅) ∧ ¬ (quadratic_solution_set_1 a = ∅ ∧ quadratic_solution_set_2 a = ∅) →
  (1/2 < a ∧ a ≤ 1) :=
sorry

theorem problem_part2 :
  quadratic_solution_set_1 a ∪ quadratic_solution_set_2 a ≠ ∅ →
  a ≤ 1 :=
sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l536_53653


namespace NUMINAMATH_GPT_smallest_value_of_x_l536_53628

theorem smallest_value_of_x :
  ∃ x, (12 * x^2 - 58 * x + 70 = 0) ∧ x = 7 / 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_of_x_l536_53628


namespace NUMINAMATH_GPT_ab_plus_cd_value_l536_53635

theorem ab_plus_cd_value (a b c d : ℝ)
  (h1 : a + b + c = 5)
  (h2 : a + b + d = 1)
  (h3 : a + c + d = 12)
  (h4 : b + c + d = 7) :
  a * b + c * d = 176 / 9 := 
sorry

end NUMINAMATH_GPT_ab_plus_cd_value_l536_53635


namespace NUMINAMATH_GPT_diameter_of_lid_is_2_inches_l536_53696

noncomputable def π : ℝ := 3.14
def C : ℝ := 6.28

theorem diameter_of_lid_is_2_inches (d : ℝ) : d = C / π → d = 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_diameter_of_lid_is_2_inches_l536_53696


namespace NUMINAMATH_GPT_new_average_after_exclusion_l536_53685

theorem new_average_after_exclusion (S : ℕ) (h1 : S = 27 * 5) (excluded : ℕ) (h2 : excluded = 35) : (S - excluded) / 4 = 25 :=
by
  sorry

end NUMINAMATH_GPT_new_average_after_exclusion_l536_53685


namespace NUMINAMATH_GPT_contestant_advancing_probability_l536_53613

noncomputable def probability_correct : ℝ := 0.8
noncomputable def probability_incorrect : ℝ := 1 - probability_correct

def sequence_pattern (q1 q2 q3 q4 : Bool) : Bool :=
  -- Pattern INCORRECT, CORRECT, CORRECT, CORRECT
  q1 == false ∧ q2 == true ∧ q3 == true ∧ q4 == true

def probability_pattern (p_corr p_incorr : ℝ) : ℝ :=
  p_incorr * p_corr * p_corr * p_corr

theorem contestant_advancing_probability :
  (probability_pattern probability_correct probability_incorrect = 0.1024) :=
by
  -- Proof required here
  sorry

end NUMINAMATH_GPT_contestant_advancing_probability_l536_53613


namespace NUMINAMATH_GPT_no_rain_either_day_l536_53676

noncomputable def P_A := 0.62
noncomputable def P_B := 0.54
noncomputable def P_A_and_B := 0.44
noncomputable def P_A_or_B := P_A + P_B - P_A_and_B -- Applying Inclusion-Exclusion principle.
noncomputable def P_A_and_B_complement := 1 - P_A_or_B -- Complement of P(A ∪ B).

theorem no_rain_either_day :
  P_A_and_B_complement = 0.28 :=
by
  unfold P_A_and_B_complement P_A_or_B
  unfold P_A P_B P_A_and_B
  simp
  sorry

end NUMINAMATH_GPT_no_rain_either_day_l536_53676


namespace NUMINAMATH_GPT_sum_of_corners_of_9x9_grid_l536_53660

theorem sum_of_corners_of_9x9_grid : 
    let topLeft := 1
    let topRight := 9
    let bottomLeft := 73
    let bottomRight := 81
    topLeft + topRight + bottomLeft + bottomRight = 164 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_corners_of_9x9_grid_l536_53660


namespace NUMINAMATH_GPT_train_speed_l536_53648

theorem train_speed (length : ℝ) (time : ℝ) (conversion_factor: ℝ)
  (h_length : length = 100) 
  (h_time : time = 5) 
  (h_conversion : conversion_factor = 3.6) :
  (length / time * conversion_factor) = 72 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l536_53648


namespace NUMINAMATH_GPT_B_finishes_in_4_days_l536_53607

theorem B_finishes_in_4_days
  (A_days : ℕ) (B_days : ℕ) (working_days_together : ℕ) 
  (A_rate : ℝ) (B_rate : ℝ) (combined_rate : ℝ) (work_done : ℝ) (remaining_work : ℝ)
  (B_rate_alone : ℝ) (days_B: ℝ) :
  A_days = 5 →
  B_days = 10 →
  working_days_together = 2 →
  A_rate = 1 / A_days →
  B_rate = 1 / B_days →
  combined_rate = A_rate + B_rate →
  work_done = combined_rate * working_days_together →
  remaining_work = 1 - work_done →
  B_rate_alone = 1 / B_days →
  days_B = remaining_work / B_rate_alone →
  days_B = 4 := 
by
  intros
  sorry

end NUMINAMATH_GPT_B_finishes_in_4_days_l536_53607


namespace NUMINAMATH_GPT_factorization_sum_l536_53601

theorem factorization_sum (a b c : ℤ) 
  (h1 : ∀ x : ℝ, (x + a) * (x + b) = x^2 + 13 * x + 40)
  (h2 : ∀ x : ℝ, (x - b) * (x - c) = x^2 - 19 * x + 88) :
  a + b + c = 24 := 
sorry

end NUMINAMATH_GPT_factorization_sum_l536_53601


namespace NUMINAMATH_GPT_area_fraction_of_rhombus_in_square_l536_53699

theorem area_fraction_of_rhombus_in_square :
  let n := 7                 -- grid size
  let side_length := n - 1   -- side length of the square
  let square_area := side_length^2 -- area of the square
  let rhombus_side := Real.sqrt 2 -- side length of the rhombus
  let rhombus_area := 2      -- area of the rhombus
  (rhombus_area / square_area) = 1 / 18 := sorry

end NUMINAMATH_GPT_area_fraction_of_rhombus_in_square_l536_53699


namespace NUMINAMATH_GPT_max_area_rectangle_l536_53662

theorem max_area_rectangle (l w : ℕ) (h : 2 * l + 2 * w = 40) : l * w ≤ 100 :=
sorry

end NUMINAMATH_GPT_max_area_rectangle_l536_53662


namespace NUMINAMATH_GPT_find_f_half_l536_53681

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_half (x : ℝ) (h₀ : 0 ≤ x ∧ x ≤ Real.pi / 2) (h₁ : f (Real.sin x) = x) : 
  f (1 / 2) = Real.pi / 6 :=
sorry

end NUMINAMATH_GPT_find_f_half_l536_53681


namespace NUMINAMATH_GPT_new_temperature_l536_53693

-- Define the initial temperature
variable (t : ℝ)

-- Define the temperature drop
def temperature_drop : ℝ := 2

-- State the theorem
theorem new_temperature (t : ℝ) (temperature_drop : ℝ) : t - temperature_drop = t - 2 :=
by
  sorry

end NUMINAMATH_GPT_new_temperature_l536_53693


namespace NUMINAMATH_GPT_total_number_of_cars_l536_53649

theorem total_number_of_cars (T A R : ℕ)
  (h1 : T - A = 37)
  (h2 : R ≥ 41)
  (h3 : ∀ x, x ≤ 59 → A = x + 37) :
  T = 133 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_cars_l536_53649


namespace NUMINAMATH_GPT_range_of_m_l536_53600

theorem range_of_m (x1 x2 m : Real) (h_eq : ∀ x : Real, x^2 - 2*x + m + 2 = 0)
  (h_abs : |x1| + |x2| ≤ 3)
  (h_real : ∀ x : Real, ∃ y : Real, x^2 - 2*x + m + 2 = 0) : -13 / 4 ≤ m ∧ m ≤ -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l536_53600


namespace NUMINAMATH_GPT_problem_statement_l536_53697

theorem problem_statement (m n : ℤ) (h : 3 * m - n = 1) : 9 * m ^ 2 - n ^ 2 - 2 * n = 1 := 
by sorry

end NUMINAMATH_GPT_problem_statement_l536_53697


namespace NUMINAMATH_GPT_change_correct_l536_53673

def cost_gum : ℕ := 350
def cost_protractor : ℕ := 500
def amount_paid : ℕ := 1000

theorem change_correct : amount_paid - (cost_gum + cost_protractor) = 150 := by
  sorry

end NUMINAMATH_GPT_change_correct_l536_53673


namespace NUMINAMATH_GPT_ratio_of_part_diminished_by_10_to_whole_number_l536_53688

theorem ratio_of_part_diminished_by_10_to_whole_number (N : ℝ) (x : ℝ) (h1 : 1/5 * N + 4 = x * N - 10) (h2 : N = 280) :
  x = 1 / 4 :=
by
  rw [h2] at h1
  sorry

end NUMINAMATH_GPT_ratio_of_part_diminished_by_10_to_whole_number_l536_53688


namespace NUMINAMATH_GPT_max_valid_subset_cardinality_l536_53609

def set_S : Finset ℕ := Finset.range 1998 \ {0}

def is_valid_subset (A : Finset ℕ) : Prop :=
  ∀ (x y : ℕ), x ≠ y → x ∈ A → y ∈ A → (x + y) % 117 ≠ 0

theorem max_valid_subset_cardinality :
  ∃ (A : Finset ℕ), is_valid_subset A ∧ 995 = A.card :=
sorry

end NUMINAMATH_GPT_max_valid_subset_cardinality_l536_53609


namespace NUMINAMATH_GPT_gcd_1443_999_l536_53654

theorem gcd_1443_999 : Nat.gcd 1443 999 = 111 := by
  sorry

end NUMINAMATH_GPT_gcd_1443_999_l536_53654
