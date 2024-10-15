import Mathlib

namespace NUMINAMATH_GPT_gcd_euclidean_algorithm_l2344_234420

theorem gcd_euclidean_algorithm (a b : ℕ) : 
  ∃ d : ℕ, d = gcd a b ∧ ∀ m : ℕ, (m ∣ a ∧ m ∣ b) → m ∣ d :=
by
  sorry

end NUMINAMATH_GPT_gcd_euclidean_algorithm_l2344_234420


namespace NUMINAMATH_GPT_inf_many_non_prime_additions_l2344_234493

theorem inf_many_non_prime_additions :
  ∃ᶠ (a : ℕ) in at_top, ∀ n : ℕ, n > 0 → ¬ Prime (n^4 + a) :=
by {
  sorry -- proof to be provided
}

end NUMINAMATH_GPT_inf_many_non_prime_additions_l2344_234493


namespace NUMINAMATH_GPT_minimum_xy_l2344_234424

theorem minimum_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1/x + 1/y = 1/2) : xy ≥ 16 :=
sorry

end NUMINAMATH_GPT_minimum_xy_l2344_234424


namespace NUMINAMATH_GPT_gcd_372_684_is_12_l2344_234463

theorem gcd_372_684_is_12 :
  Nat.gcd 372 684 = 12 :=
sorry

end NUMINAMATH_GPT_gcd_372_684_is_12_l2344_234463


namespace NUMINAMATH_GPT_swim_club_members_l2344_234444

theorem swim_club_members (X : ℝ) 
  (h1 : 0.30 * X = 0.30 * X)
  (h2 : 0.70 * X = 42) : X = 60 :=
sorry

end NUMINAMATH_GPT_swim_club_members_l2344_234444


namespace NUMINAMATH_GPT_projectile_first_reach_height_56_l2344_234410

theorem projectile_first_reach_height_56 (t : ℝ) (h1 : ∀ t, y = -16 * t^2 + 60 * t) :
    (∃ t : ℝ, y = 56 ∧ t = 1.75 ∧ (∀ t', t' < 1.75 → y ≠ 56)) :=
by
  sorry

end NUMINAMATH_GPT_projectile_first_reach_height_56_l2344_234410


namespace NUMINAMATH_GPT_product_divisible_by_15_l2344_234435

theorem product_divisible_by_15 (n : ℕ) (hn1 : n % 2 = 1) (hn2 : n > 0) :
  15 ∣ (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) :=
sorry

end NUMINAMATH_GPT_product_divisible_by_15_l2344_234435


namespace NUMINAMATH_GPT_most_stable_performance_l2344_234469

structure Shooter :=
(average_score : ℝ)
(variance : ℝ)

def A := Shooter.mk 8.9 0.45
def B := Shooter.mk 8.9 0.42
def C := Shooter.mk 8.9 0.51

theorem most_stable_performance : 
  B.variance < A.variance ∧ B.variance < C.variance :=
by
  sorry

end NUMINAMATH_GPT_most_stable_performance_l2344_234469


namespace NUMINAMATH_GPT_tyler_cd_purchase_l2344_234461

theorem tyler_cd_purchase :
  ∀ (initial_cds : ℕ) (given_away_fraction : ℝ) (final_cds : ℕ) (bought_cds : ℕ),
    initial_cds = 21 →
    given_away_fraction = 1 / 3 →
    final_cds = 22 →
    bought_cds = 8 →
    final_cds = initial_cds - initial_cds * given_away_fraction + bought_cds :=
by
  intros
  sorry

end NUMINAMATH_GPT_tyler_cd_purchase_l2344_234461


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l2344_234431

theorem sum_of_squares_of_roots : 
  ∀ r1 r2 : ℝ, (r1 + r2 = 10) → (r1 * r2 = 9) → (r1 > 5 ∨ r2 > 5) → (r1^2 + r2^2 = 82) :=
by
  intros r1 r2 h1 h2 h3
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l2344_234431


namespace NUMINAMATH_GPT_page_mistakenly_added_twice_l2344_234429

theorem page_mistakenly_added_twice (n k: ℕ) (h₁: n = 77) (h₂: (n * (n + 1)) / 2 + k = 3050) : k = 47 :=
by
  -- sorry here to indicate the proof is not needed
  sorry

end NUMINAMATH_GPT_page_mistakenly_added_twice_l2344_234429


namespace NUMINAMATH_GPT_amy_race_time_l2344_234432

theorem amy_race_time (patrick_time : ℕ) (manu_time : ℕ) (amy_time : ℕ)
  (h1 : patrick_time = 60)
  (h2 : manu_time = patrick_time + 12)
  (h3 : amy_time = manu_time / 2) : 
  amy_time = 36 := 
sorry

end NUMINAMATH_GPT_amy_race_time_l2344_234432


namespace NUMINAMATH_GPT_box_surface_area_is_276_l2344_234415

-- Define the dimensions of the box
variables {l w h : ℝ}

-- Define the pricing function
def pricing (x y z : ℝ) : ℝ := 0.30 * x + 0.40 * y + 0.50 * z

-- Define the condition for the box fee
def box_fee (x y z : ℝ) (fee : ℝ) := pricing x y z = fee

-- Define the constraint that no faces are squares
def no_square_faces (l w h : ℝ) : Prop := 
  l ≠ w ∧ w ≠ h ∧ h ≠ l

-- Define the surface area calculation
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

-- The main theorem stating the problem
theorem box_surface_area_is_276 (l w h : ℝ) 
  (H1 : box_fee l w h 8.10 ∧ box_fee w h l 8.10)
  (H2 : box_fee l w h 8.70 ∧ box_fee w h l 8.70)
  (H3 : no_square_faces l w h) : 
  surface_area l w h = 276 := 
sorry

end NUMINAMATH_GPT_box_surface_area_is_276_l2344_234415


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2344_234422

open Set

def A : Set Int := {x | x + 2 = 0}

def B : Set Int := {x | x^2 - 4 = 0}

theorem intersection_of_A_and_B : A ∩ B = {-2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l2344_234422


namespace NUMINAMATH_GPT_relay_race_athlete_orders_l2344_234491

def athlete_count : ℕ := 4
def cannot_run_first_leg (athlete : ℕ) : Prop := athlete = 1
def cannot_run_fourth_leg (athlete : ℕ) : Prop := athlete = 2

theorem relay_race_athlete_orders : 
  ∃ (number_of_orders : ℕ), number_of_orders = 14 := 
by 
  -- Proof is omitted because it’s not required as per instructions.
  sorry

end NUMINAMATH_GPT_relay_race_athlete_orders_l2344_234491


namespace NUMINAMATH_GPT_dave_trips_l2344_234471

theorem dave_trips :
  let trays_at_a_time := 12
  let trays_table_1 := 26
  let trays_table_2 := 49
  let trays_table_3 := 65
  let trays_table_4 := 38
  let total_trays := trays_table_1 + trays_table_2 + trays_table_3 + trays_table_4
  let trips := (total_trays + trays_at_a_time - 1) / trays_at_a_time
  trips = 15 := by
    repeat { sorry }

end NUMINAMATH_GPT_dave_trips_l2344_234471


namespace NUMINAMATH_GPT_rectangle_length_eq_15_l2344_234477

theorem rectangle_length_eq_15 (w l s p_rect p_square : ℝ)
    (h_w : w = 9)
    (h_s : s = 12)
    (h_p_square : p_square = 4 * s)
    (h_p_rect : p_rect = 2 * w + 2 * l)
    (h_eq_perimeters : p_square = p_rect) : l = 15 := by
  sorry

end NUMINAMATH_GPT_rectangle_length_eq_15_l2344_234477


namespace NUMINAMATH_GPT_polygon_side_intersections_l2344_234400

theorem polygon_side_intersections :
  let m6 := 6
  let m7 := 7
  let m8 := 8
  let m9 := 9
  let pairs := [(m6, m7), (m6, m8), (m6, m9), (m7, m8), (m7, m9), (m8, m9)]
  let count_intersections (m n : ℕ) : ℕ := 2 * min m n
  let total_intersections := pairs.foldl (fun total pair => total + count_intersections pair.1 pair.2) 0
  total_intersections = 80 :=
by
  sorry

end NUMINAMATH_GPT_polygon_side_intersections_l2344_234400


namespace NUMINAMATH_GPT_range_of_m_l2344_234446

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - x

theorem range_of_m (m : ℝ) (x : ℝ) (h1 : x ∈ Set.Icc (-1 : ℝ) 2) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, f x < m) ↔ 2 < m := 
by 
  sorry

end NUMINAMATH_GPT_range_of_m_l2344_234446


namespace NUMINAMATH_GPT_sock_pairing_l2344_234498

def sockPicker : Prop :=
  let white_socks := 5
  let brown_socks := 5
  let blue_socks := 2
  let total_socks := 12
  let choose (n k : ℕ) := Nat.choose n k
  (choose white_socks 2 + choose brown_socks 2 + choose blue_socks 2 = 21) ∧
  (choose (white_socks + brown_socks) 2 = 45) ∧
  (45 = 45)

theorem sock_pairing :
  sockPicker :=
by sorry

end NUMINAMATH_GPT_sock_pairing_l2344_234498


namespace NUMINAMATH_GPT_evaluate_expression_l2344_234407

theorem evaluate_expression (b : ℕ) (h : b = 4) : (b ^ b - b * (b - 1) ^ b) ^ b = 21381376 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2344_234407


namespace NUMINAMATH_GPT_total_metal_rods_needed_l2344_234405

def metal_rods_per_sheet : ℕ := 10
def sheets_per_panel : ℕ := 3
def metal_rods_per_beam : ℕ := 4
def beams_per_panel : ℕ := 2
def panels : ℕ := 10

theorem total_metal_rods_needed : 
  (sheets_per_panel * metal_rods_per_sheet + beams_per_panel * metal_rods_per_beam) * panels = 380 :=
by
  exact rfl

end NUMINAMATH_GPT_total_metal_rods_needed_l2344_234405


namespace NUMINAMATH_GPT_two_A_plus_B_l2344_234484

theorem two_A_plus_B (A B : ℕ) (h1 : A = Nat.gcd (Nat.gcd 12 18) 30) (h2 : B = Nat.lcm (Nat.lcm 12 18) 30) : 2 * A + B = 192 :=
by
  sorry

end NUMINAMATH_GPT_two_A_plus_B_l2344_234484


namespace NUMINAMATH_GPT_find_f_28_l2344_234421

theorem find_f_28 (f : ℕ → ℚ) (h1 : ∀ n : ℕ, f (n + 1) = (3 * f n + n) / 3) (h2 : f 1 = 1) :
  f 28 = 127 := by
sorry

end NUMINAMATH_GPT_find_f_28_l2344_234421


namespace NUMINAMATH_GPT_negation_exists_positive_real_square_plus_one_l2344_234441

def exists_positive_real_square_plus_one : Prop :=
  ∃ (x : ℝ), x^2 + 1 > 0

def forall_non_positive_real_square_plus_one : Prop :=
  ∀ (x : ℝ), x^2 + 1 ≤ 0

theorem negation_exists_positive_real_square_plus_one :
  ¬ exists_positive_real_square_plus_one ↔ forall_non_positive_real_square_plus_one :=
by
  sorry

end NUMINAMATH_GPT_negation_exists_positive_real_square_plus_one_l2344_234441


namespace NUMINAMATH_GPT_factor_expression_l2344_234497

theorem factor_expression (x y : ℤ) : 231 * x^2 * y + 33 * x * y = 33 * x * y * (7 * x + 1) := by
  sorry

end NUMINAMATH_GPT_factor_expression_l2344_234497


namespace NUMINAMATH_GPT_average_children_in_families_with_children_l2344_234408

theorem average_children_in_families_with_children
  (n : ℕ)
  (c_avg : ℕ)
  (c_no_children : ℕ)
  (total_children : ℕ)
  (families_with_children : ℕ)
  (avg_children_families_with_children : ℚ) :
  n = 15 →
  c_avg = 3 →
  c_no_children = 3 →
  total_children = n * c_avg →
  families_with_children = n - c_no_children →
  avg_children_families_with_children = total_children / families_with_children →
  avg_children_families_with_children = 3.8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_average_children_in_families_with_children_l2344_234408


namespace NUMINAMATH_GPT_triangle_angle_C_30_degrees_l2344_234417

theorem triangle_angle_C_30_degrees 
  (A B C : ℝ) 
  (h1 : 3 * Real.sin A + 4 * Real.cos B = 6) 
  (h2 : 4 * Real.sin B + 3 * Real.cos A = 1) 
  (h3 : A + B + C = 180) 
  : C = 30 :=
  sorry

end NUMINAMATH_GPT_triangle_angle_C_30_degrees_l2344_234417


namespace NUMINAMATH_GPT_points_per_enemy_l2344_234475

-- Definitions: total enemies, enemies not destroyed, points earned
def total_enemies : ℕ := 11
def enemies_not_destroyed : ℕ := 3
def points_earned : ℕ := 72

-- To prove: points per enemy
theorem points_per_enemy : points_earned / (total_enemies - enemies_not_destroyed) = 9 := 
by
  sorry

end NUMINAMATH_GPT_points_per_enemy_l2344_234475


namespace NUMINAMATH_GPT_water_fraction_after_replacements_l2344_234467

-- Initially given conditions
def radiator_capacity : ℚ := 20
def initial_water_fraction : ℚ := 1
def antifreeze_quarts : ℚ := 5
def replacements : ℕ := 5

-- Derived condition
def water_remain_fraction : ℚ := 3 / 4

-- Statement of the problem
theorem water_fraction_after_replacements :
  (water_remain_fraction ^ replacements) = 243 / 1024 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_water_fraction_after_replacements_l2344_234467


namespace NUMINAMATH_GPT_russian_writer_surname_l2344_234476

def is_valid_surname (x y z w v u : ℕ) : Prop :=
  x = z ∧
  y = w ∧
  v = x + 9 ∧
  u = y + w - 2 ∧
  3 * x = y - 4 ∧
  x + y + z + w + v + u = 83

def position_to_letter (n : ℕ) : String :=
  if n = 4 then "Г"
  else if n = 16 then "О"
  else if n = 13 then "Л"
  else if n = 30 then "Ь"
  else "?"

theorem russian_writer_surname : ∃ x y z w v u : ℕ, 
  is_valid_surname x y z w v u ∧
  position_to_letter x ++ position_to_letter y ++ position_to_letter z ++ position_to_letter w ++ position_to_letter v ++ position_to_letter u = "Гоголь" :=
by
  sorry

end NUMINAMATH_GPT_russian_writer_surname_l2344_234476


namespace NUMINAMATH_GPT_max_sum_at_11_l2344_234434

noncomputable def is_arithmetic_seq (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_seq (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n + 1) * a 0 + (n * (n + 1) / 2) * (a 1 - a 0)

theorem max_sum_at_11 (a : ℕ → ℚ) (d : ℚ) (h_arith : is_arithmetic_seq a) (h_a1_gt_0 : a 0 > 0)
 (h_sum_eq : sum_seq a 13 = sum_seq a 7) : 
  ∃ n : ℕ, sum_seq a n = sum_seq a 10 + (a 10 + a 11) := sorry


end NUMINAMATH_GPT_max_sum_at_11_l2344_234434


namespace NUMINAMATH_GPT_find_angle_A_min_perimeter_l2344_234473

theorem find_angle_A (a b c : ℝ) (A B C : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) 
  (h₄ : a > 0 ∧ b > 0 ∧ c > 0) (h5 : b + c * Real.cos A = c + a * Real.cos C) 
  (hTriangle : A + B + C = Real.pi)
  (hSineLaw : Real.sin B = Real.sin C * Real.cos A + Real.sin A * Real.cos C) :
  A = Real.pi / 3 := 
by 
  sorry

theorem min_perimeter (a b c : ℝ) (A : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) 
  (h4 : a > 0 ∧ b > 0 ∧ c > 0 ∧ A = Real.pi / 3)
  (h_area : 1 / 2 * b * c * Real.sin A = Real.sqrt 3)
  (h_cosine : a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) :
  a + b + c = 6 :=
by 
  sorry

end NUMINAMATH_GPT_find_angle_A_min_perimeter_l2344_234473


namespace NUMINAMATH_GPT_find_number_of_appliances_l2344_234402

-- Declare the constants related to the problem.
def commission_per_appliance : ℝ := 50
def commission_percent : ℝ := 0.1
def total_selling_price : ℝ := 3620
def total_commission : ℝ := 662

-- Define the theorem to solve for the number of appliances sold.
theorem find_number_of_appliances (n : ℝ) 
  (H : n * commission_per_appliance + commission_percent * total_selling_price = total_commission) : 
  n = 6 := 
sorry

end NUMINAMATH_GPT_find_number_of_appliances_l2344_234402


namespace NUMINAMATH_GPT_purely_imaginary_sol_l2344_234413

theorem purely_imaginary_sol (x : ℝ) 
  (h1 : (x^2 - 1) = 0)
  (h_imag : (x^2 + 3 * x + 2) ≠ 0) :
  x = 1 :=
sorry

end NUMINAMATH_GPT_purely_imaginary_sol_l2344_234413


namespace NUMINAMATH_GPT_students_remaining_after_four_stops_l2344_234483

theorem students_remaining_after_four_stops :
  let initial_students := 60 
  let fraction_remaining := (2 / 3 : ℚ)
  let stop1_students := initial_students * fraction_remaining
  let stop2_students := stop1_students * fraction_remaining
  let stop3_students := stop2_students * fraction_remaining
  let stop4_students := stop3_students * fraction_remaining
  stop4_students = (320 / 27 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_students_remaining_after_four_stops_l2344_234483


namespace NUMINAMATH_GPT_remaining_pencils_l2344_234448

/-
Given the initial number of pencils in the drawer and the number of pencils Sally took out,
prove that the number of pencils remaining in the drawer is 5.
-/
def pencils_in_drawer (initial_pencils : ℕ) (pencils_taken : ℕ) : ℕ :=
  initial_pencils - pencils_taken

theorem remaining_pencils : pencils_in_drawer 9 4 = 5 := by
  sorry

end NUMINAMATH_GPT_remaining_pencils_l2344_234448


namespace NUMINAMATH_GPT_problem1_problem2_l2344_234495

variable {a b : ℝ}

theorem problem1 (h : a > b) : a - 3 > b - 3 :=
by sorry

theorem problem2 (h : a > b) : -4 * a < -4 * b :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l2344_234495


namespace NUMINAMATH_GPT_zora_is_shorter_by_eight_l2344_234487

noncomputable def zora_height (z : ℕ) (b : ℕ) (i : ℕ) (zara : ℕ) (average_height : ℕ) : Prop :=
  i = z + 4 ∧
  zara = b ∧
  average_height = 61 ∧
  (z + i + zara + b) / 4 = average_height

theorem zora_is_shorter_by_eight (Z B : ℕ)
  (h1 : zora_height Z B (Z + 4) 64 61) : (B - Z) = 8 :=
by
  sorry

end NUMINAMATH_GPT_zora_is_shorter_by_eight_l2344_234487


namespace NUMINAMATH_GPT_geom_seq_mult_l2344_234426

variable {α : Type*} [LinearOrderedField α]

def is_geom_seq (a : ℕ → α) :=
  ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)

theorem geom_seq_mult (a : ℕ → α) (h : is_geom_seq a) (hpos : ∀ n, 0 < a n) (h4_8 : a 4 * a 8 = 4) :
  a 5 * a 6 * a 7 = 8 := 
sorry

end NUMINAMATH_GPT_geom_seq_mult_l2344_234426


namespace NUMINAMATH_GPT_minimum_routes_A_C_l2344_234486

namespace SettlementRoutes

-- Define three settlements A, B, and C
variable (A B C : Type)

-- Assume there are more than one roads connecting each settlement pair directly
variable (k m n : ℕ) -- k: roads between A and B, m: roads between B and C, n: roads between A and C

-- Conditions: Total paths including intermediate nodes
axiom h1 : k + m * n = 34
axiom h2 : m + k * n = 29

-- Theorem: Minimum number of routes connecting A and C is 26
theorem minimum_routes_A_C : ∃ n k m : ℕ, k + m * n = 34 ∧ m + k * n = 29 ∧ n + k * m = 26 := sorry

end SettlementRoutes

end NUMINAMATH_GPT_minimum_routes_A_C_l2344_234486


namespace NUMINAMATH_GPT_sum_of_a_and_b_l2344_234433

noncomputable def a : ℕ :=
sorry

noncomputable def b : ℕ :=
sorry

theorem sum_of_a_and_b :
  (100 ≤ a ∧ a ≤ 999) ∧ (1000 ≤ b ∧ b ≤ 9999) ∧ (10000 * a + b = 7 * a * b) ->
  a + b = 1458 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_a_and_b_l2344_234433


namespace NUMINAMATH_GPT_distance_walked_on_third_day_l2344_234454

theorem distance_walked_on_third_day:
  ∃ x : ℝ, 
    4 * x + 2 * x + x + (1 / 2) * x + (1 / 4) * x + (1 / 8) * x = 378 ∧
    x = 48 := 
by
  sorry

end NUMINAMATH_GPT_distance_walked_on_third_day_l2344_234454


namespace NUMINAMATH_GPT_chess_tournament_participants_l2344_234404

/-- If each participant of a chess tournament plays exactly one game with each of the remaining participants, and 231 games are played during the tournament, then the number of participants is 22. -/
theorem chess_tournament_participants (n : ℕ) (h : (n - 1) * n / 2 = 231) : n = 22 :=
sorry

end NUMINAMATH_GPT_chess_tournament_participants_l2344_234404


namespace NUMINAMATH_GPT_not_possible_perimeter_72_l2344_234459

variable (a b : ℕ)
variable (P : ℕ)

def valid_perimeter_range (a b : ℕ) : Set ℕ := 
  { P | ∃ x, 15 < x ∧ x < 35 ∧ P = a + b + x }

theorem not_possible_perimeter_72 :
  (a = 10) → (b = 25) → ¬ (72 ∈ valid_perimeter_range 10 25) := 
by
  sorry

end NUMINAMATH_GPT_not_possible_perimeter_72_l2344_234459


namespace NUMINAMATH_GPT_owner_overtakes_thief_l2344_234436

theorem owner_overtakes_thief :
  let thief_speed_initial := 45 -- kmph
  let discovery_time := 0.5 -- hours
  let owner_speed := 50 -- kmph
  let mud_road_speed := 35 -- kmph
  let mud_road_distance := 30 -- km
  let speed_bumps_speed := 40 -- kmph
  let speed_bumps_distance := 5 -- km
  let traffic_speed := 30 -- kmph
  let head_start_distance := thief_speed_initial * discovery_time
  let mud_road_time := mud_road_distance / mud_road_speed
  let speed_bumps_time := speed_bumps_distance / speed_bumps_speed
  let total_distance_before_traffic := mud_road_distance + speed_bumps_distance
  let total_time_before_traffic := mud_road_time + speed_bumps_time
  let distance_owner_travelled := owner_speed * total_time_before_traffic
  head_start_distance + total_distance_before_traffic < distance_owner_travelled →
  discovery_time + total_time_before_traffic = 1.482 :=
by sorry


end NUMINAMATH_GPT_owner_overtakes_thief_l2344_234436


namespace NUMINAMATH_GPT_evaluate_expression_at_one_l2344_234447

theorem evaluate_expression_at_one : 
  (4 + (4 + x^2) / x) / ((x + 2) / x) = 3 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_one_l2344_234447


namespace NUMINAMATH_GPT_complex_pure_imaginary_l2344_234453

theorem complex_pure_imaginary (a : ℝ) : (↑a + Complex.I) / (1 - Complex.I) = 0 + b * Complex.I → a = 1 :=
by
  intro h
  -- Proof content here
  sorry

end NUMINAMATH_GPT_complex_pure_imaginary_l2344_234453


namespace NUMINAMATH_GPT_consecutive_vertices_product_l2344_234452

theorem consecutive_vertices_product (n : ℕ) (hn : n = 90) :
  ∃ (i : ℕ), 1 ≤ i ∧ i ≤ n ∧ ((i * (i % n + 1)) ≥ 2014) := 
sorry

end NUMINAMATH_GPT_consecutive_vertices_product_l2344_234452


namespace NUMINAMATH_GPT_distribution_ways_l2344_234462

theorem distribution_ways (n_problems n_friends : ℕ) (h_problems : n_problems = 6) (h_friends : n_friends = 8) : (n_friends ^ n_problems) = 262144 :=
by
  rw [h_problems, h_friends]
  norm_num

end NUMINAMATH_GPT_distribution_ways_l2344_234462


namespace NUMINAMATH_GPT_lucas_change_l2344_234439

-- Define the initial amount of money Lucas has
def initial_amount : ℕ := 20

-- Define the cost of one avocado
def cost_per_avocado : ℕ := 2

-- Define the number of avocados Lucas buys
def number_of_avocados : ℕ := 3

-- Calculate the total cost of avocados
def total_cost : ℕ := number_of_avocados * cost_per_avocado

-- Calculate the remaining amount of money (change)
def remaining_amount : ℕ := initial_amount - total_cost

-- The proposition to prove: Lucas brings home $14
theorem lucas_change : remaining_amount = 14 := by
  sorry

end NUMINAMATH_GPT_lucas_change_l2344_234439


namespace NUMINAMATH_GPT_cannot_determine_position_l2344_234403

-- Define the conditions
def east_longitude_122_north_latitude_43_6 : Prop := true
def row_6_seat_3_in_cinema : Prop := true
def group_1_in_classroom : Prop := false
def island_50_nautical_miles_north_northeast_another : Prop := true

-- Define the theorem
theorem cannot_determine_position :
  ¬ ((east_longitude_122_north_latitude_43_6 = false) ∧
     (row_6_seat_3_in_cinema = false) ∧
     (island_50_nautical_miles_north_northeast_another = false) ∧
     (group_1_in_classroom = true)) :=
by
  sorry

end NUMINAMATH_GPT_cannot_determine_position_l2344_234403


namespace NUMINAMATH_GPT_cesaro_lupu_real_analysis_l2344_234427

noncomputable def proof_problem (a b c x y z : ℝ) : Prop :=
  (0 < a ∧ a < 1) ∧ (0 < b ∧ b < 1) ∧ (0 < c ∧ c < 1) ∧
  (0 < x) ∧ (0 < y) ∧ (0 < z) ∧
  (a^x = b * c) ∧ (b^y = c * a) ∧ (c^z = a * b) →
  (1 / (2 + x) + 1 / (2 + y) + 1 / (2 + z) ≤ 3 / 4)

theorem cesaro_lupu_real_analysis (a b c x y z : ℝ) :
  proof_problem a b c x y z :=
by sorry

end NUMINAMATH_GPT_cesaro_lupu_real_analysis_l2344_234427


namespace NUMINAMATH_GPT_parallelepiped_properties_l2344_234414

/--
In an oblique parallelepiped with the following properties:
- The height is 12 dm,
- The projection of the lateral edge on the base plane is 5 dm,
- A cross-section perpendicular to the lateral edge is a rhombus with:
  - An area of 24 dm²,
  - A diagonal of 8 dm,
Prove that:
1. The lateral surface area is 260 dm².
2. The volume is 312 dm³.
-/
theorem parallelepiped_properties
    (height : ℝ)
    (projection_lateral_edge : ℝ)
    (area_rhombus : ℝ)
    (diagonal_rhombus : ℝ)
    (lateral_surface_area : ℝ)
    (volume : ℝ) :
  height = 12 ∧
  projection_lateral_edge = 5 ∧
  area_rhombus = 24 ∧
  diagonal_rhombus = 8 ∧
  lateral_surface_area = 260 ∧
  volume = 312 :=
by
  sorry

end NUMINAMATH_GPT_parallelepiped_properties_l2344_234414


namespace NUMINAMATH_GPT_quadratic_no_roots_c_positive_l2344_234489

theorem quadratic_no_roots_c_positive
  (a b c : ℝ)
  (h_no_roots : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0)
  (h_positive : a + b + c > 0) :
  c > 0 :=
sorry

end NUMINAMATH_GPT_quadratic_no_roots_c_positive_l2344_234489


namespace NUMINAMATH_GPT_log2_x_value_l2344_234492

noncomputable def log_base (a b : ℝ) : ℝ := Real.log b / Real.log a

theorem log2_x_value
  (x : ℝ)
  (h : log_base (5 * x) (2 * x) = log_base (625 * x) (8 * x)) :
  log_base 2 x = Real.log 5 / (2 * Real.log 2 - 3 * Real.log 5) :=
by
  sorry

end NUMINAMATH_GPT_log2_x_value_l2344_234492


namespace NUMINAMATH_GPT_number_of_students_l2344_234465

theorem number_of_students (T : ℕ) (n : ℕ) (h1 : (T + 20) / n = T / n + 1 / 2) : n = 40 :=
  sorry

end NUMINAMATH_GPT_number_of_students_l2344_234465


namespace NUMINAMATH_GPT_find_sin_angle_BAD_l2344_234472

def isosceles_right_triangle (A B C : ℝ → ℝ → Prop) (AB BC AC : ℝ) : Prop :=
  AB = 2 ∧ BC = 2 ∧ AC = 2 * Real.sqrt 2

def right_triangle_on_hypotenuse (A C D : ℝ → ℝ → Prop) (AC CD DA : ℝ) (DAC : ℝ) : Prop :=
  AC = 2 * Real.sqrt 2 ∧ CD = DA / 2 ∧ DAC = Real.pi / 6

def equal_perimeters (AC CD DA : ℝ) : Prop := 
  AC + CD + DA = 4 + 2 * Real.sqrt 2

theorem find_sin_angle_BAD :
  ∀ (A B C D : ℝ → ℝ → Prop) (AB BC AC CD DA : ℝ),
  isosceles_right_triangle A B C AB BC AC →
  right_triangle_on_hypotenuse A C D AC CD DA (Real.pi / 6) →
  equal_perimeters AC CD DA →
  Real.sin (2 * (Real.pi / 4 + Real.pi / 6)) = 1 / 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_sin_angle_BAD_l2344_234472


namespace NUMINAMATH_GPT_reduced_price_equals_50_l2344_234482

noncomputable def reduced_price (P : ℝ) : ℝ := 0.75 * P

theorem reduced_price_equals_50 (P : ℝ) (X : ℝ) 
  (h1 : 1000 = X * P)
  (h2 : 1000 = (X + 5) * 0.75 * P) : reduced_price P = 50 :=
sorry

end NUMINAMATH_GPT_reduced_price_equals_50_l2344_234482


namespace NUMINAMATH_GPT_problem_1_problem_2_l2344_234442

theorem problem_1 
  (h1 : 1 < Real.sqrt 2 ∧ Real.sqrt 2 < 2) : 
  Int.floor (5 - Real.sqrt 2) = 3 :=
sorry

theorem problem_2 
  (h2 : Real.sqrt 3 > 1) : 
  abs (1 - 2 * Real.sqrt 3) = 2 * Real.sqrt 3 - 1 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l2344_234442


namespace NUMINAMATH_GPT_right_triangle_condition_l2344_234457

theorem right_triangle_condition (A B C : ℝ) (a b c : ℝ) :
  (A + B = 90) → (A + B + C = 180) → (C = 90) := 
by
  sorry

end NUMINAMATH_GPT_right_triangle_condition_l2344_234457


namespace NUMINAMATH_GPT_sum_bases_exponents_max_product_l2344_234460

theorem sum_bases_exponents_max_product (A : ℕ) (hA : A = 3 ^ 670 * 2 ^ 2) : 
    (3 + 2 + 670 + 2 = 677) := by
  sorry

end NUMINAMATH_GPT_sum_bases_exponents_max_product_l2344_234460


namespace NUMINAMATH_GPT_contrapositive_statement_l2344_234411

theorem contrapositive_statement 
  (a : ℝ) (b : ℝ) 
  (h1 : a > 0) 
  (h3 : a + b < 0) : 
  b < 0 :=
sorry

end NUMINAMATH_GPT_contrapositive_statement_l2344_234411


namespace NUMINAMATH_GPT_find_greatest_natural_number_l2344_234428

-- Definitions for terms used in the conditions

def sum_of_squares (m : ℕ) : ℕ :=
  (m * (m + 1) * (2 * m + 1)) / 6

def is_perfect_square (a : ℕ) : Prop :=
  ∃ b : ℕ, b * b = a

-- Conditions defined in Lean terms
def condition1 (n : ℕ) : Prop := n ≤ 2010

def condition2 (n : ℕ) : Prop := 
  let sum1 := sum_of_squares n
  let sum2 := sum_of_squares (2 * n) - sum_of_squares n
  is_perfect_square (sum1 * sum2)

-- Main theorem statement
theorem find_greatest_natural_number : ∃ n, n ≤ 2010 ∧ condition2 n ∧ ∀ m, m ≤ 2010 ∧ condition2 m → m ≤ n := 
by 
  sorry

end NUMINAMATH_GPT_find_greatest_natural_number_l2344_234428


namespace NUMINAMATH_GPT_exists_face_with_fewer_than_six_sides_l2344_234423

theorem exists_face_with_fewer_than_six_sides
  (N K M : ℕ) 
  (h_euler : N - K + M = 2)
  (h_vertices : M ≤ 2 * K / 3) : 
  ∃ n_i : ℕ, n_i < 6 :=
by
  sorry

end NUMINAMATH_GPT_exists_face_with_fewer_than_six_sides_l2344_234423


namespace NUMINAMATH_GPT_probability_three_blue_jellybeans_l2344_234458

theorem probability_three_blue_jellybeans:
  let total_jellybeans := 20
  let blue_jellybeans := 10
  let red_jellybeans := 10
  let draws := 3
  let q := (1 / 2) * (9 / 19) * (4 / 9)
  q = 2 / 19 :=
sorry

end NUMINAMATH_GPT_probability_three_blue_jellybeans_l2344_234458


namespace NUMINAMATH_GPT_picture_frame_length_l2344_234485

theorem picture_frame_length (h : ℕ) (l : ℕ) (P : ℕ) (h_eq : h = 12) (P_eq : P = 44) (perimeter_eq : P = 2 * (l + h)) : l = 10 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_picture_frame_length_l2344_234485


namespace NUMINAMATH_GPT_initial_scooter_value_l2344_234449

theorem initial_scooter_value (V : ℝ) (h : V * (3/4)^2 = 22500) : V = 40000 :=
by
  sorry

end NUMINAMATH_GPT_initial_scooter_value_l2344_234449


namespace NUMINAMATH_GPT_proof_equation_of_line_l2344_234430
   
   -- Define the point P
   structure Point where
     x : ℝ
     y : ℝ
     
   -- Define conditions
   def passesThroughP (line : ℝ → ℝ → Prop) : Prop :=
     line 2 (-1)
     
   def interceptRelation (line : ℝ → ℝ → Prop) : Prop :=
     ∃ a : ℝ, a ≠ 0 ∧ (∀ x y, line x y ↔ (x / a + y / (2 * a) = 1))
   
   -- Define the line equation
   def line_equation (line : ℝ → ℝ → Prop) : Prop :=
     passesThroughP line ∧ interceptRelation line
     
   -- The final statement
   theorem proof_equation_of_line (line : ℝ → ℝ → Prop) :
     line_equation line →
     (∀ x y, line x y ↔ (2 * x + y = 3)) ∨ (∀ x y, line x y ↔ (x + 2 * y = 0)) :=
   by
     sorry
   
end NUMINAMATH_GPT_proof_equation_of_line_l2344_234430


namespace NUMINAMATH_GPT_m_range_l2344_234470

noncomputable def range_of_m (m : ℝ) : Prop :=
  ∃ x : ℝ, 
    x ≠ 2 ∧ 
    (x + m) / (x - 2) - 3 = (x - 1) / (2 - x) ∧ 
    x ≥ 0

theorem m_range (m : ℝ) : 
  range_of_m m ↔ m ≥ -5 ∧ m ≠ -3 := 
sorry

end NUMINAMATH_GPT_m_range_l2344_234470


namespace NUMINAMATH_GPT_geometric_sequence_sum_l2344_234419

noncomputable def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a 1 * q^n

theorem geometric_sequence_sum :
  ∃ (a : ℕ → ℝ) (q : ℝ), 
  (∀ n : ℕ, a (n + 1) = a 1 * q^n) ∧ 
  (a 2 * a 4 = 1) ∧ 
  (a 1 * (q^0 + q^1 + q^2) = 7) ∧ 
  (a 1 / (1 - q) * (1 - q^5) = 31 / 4) := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l2344_234419


namespace NUMINAMATH_GPT_remainder_of_sum_division_l2344_234499

def a1 : ℕ := 2101
def a2 : ℕ := 2103
def a3 : ℕ := 2105
def a4 : ℕ := 2107
def a5 : ℕ := 2109
def n : ℕ := 12

theorem remainder_of_sum_division : ((a1 + a2 + a3 + a4 + a5) % n) = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_remainder_of_sum_division_l2344_234499


namespace NUMINAMATH_GPT_selling_price_correct_l2344_234478

/-- Define the total number of units to be sold -/
def total_units : ℕ := 5000

/-- Define the variable cost per unit -/
def variable_cost_per_unit : ℕ := 800

/-- Define the total fixed costs -/
def fixed_costs : ℕ := 1000000

/-- Define the desired profit -/
def desired_profit : ℕ := 1500000

/-- The selling price p must be calculated such that revenues exceed expenses by the desired profit -/
theorem selling_price_correct : 
  ∃ p : ℤ, p = 1300 ∧ (total_units * p) - (fixed_costs + (total_units * variable_cost_per_unit)) = desired_profit :=
by
  sorry

end NUMINAMATH_GPT_selling_price_correct_l2344_234478


namespace NUMINAMATH_GPT_exact_consecutive_hits_l2344_234445

/-
Prove the number of ways to arrange 8 shots with exactly 3 hits such that exactly 2 out of the 3 hits are consecutive is 30.
-/

def count_distinct_sequences (total_shots : ℕ) (hits : ℕ) (consecutive_hits : ℕ) : ℕ :=
  if total_shots = 8 ∧ hits = 3 ∧ consecutive_hits = 2 then 30 else 0

theorem exact_consecutive_hits :
  count_distinct_sequences 8 3 2 = 30 :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_exact_consecutive_hits_l2344_234445


namespace NUMINAMATH_GPT_medicine_dose_per_part_l2344_234496

-- Define the given conditions
def kg_weight : ℕ := 30
def ml_per_kg : ℕ := 5
def parts : ℕ := 3

-- The theorem statement
theorem medicine_dose_per_part : 
  (kg_weight * ml_per_kg) / parts = 50 :=
by
  sorry

end NUMINAMATH_GPT_medicine_dose_per_part_l2344_234496


namespace NUMINAMATH_GPT_lattice_point_distance_l2344_234468

theorem lattice_point_distance (d : ℝ) : 
  (∃ (r : ℝ), r = 2020 ∧ (∀ (A B C D : ℝ), 
  A = 0 ∧ B = 4040 ∧ C = 2020 ∧ D = 4040) 
  ∧ (∃ (P Q : ℝ), P = 0.25 ∧ Q = 1)) → 
  d = 0.3 := 
by
  sorry

end NUMINAMATH_GPT_lattice_point_distance_l2344_234468


namespace NUMINAMATH_GPT_vanessa_earnings_l2344_234438

def cost : ℕ := 4
def total_bars : ℕ := 11
def bars_unsold : ℕ := 7
def bars_sold : ℕ := total_bars - bars_unsold
def money_made : ℕ := bars_sold * cost

theorem vanessa_earnings : money_made = 16 := by
  sorry

end NUMINAMATH_GPT_vanessa_earnings_l2344_234438


namespace NUMINAMATH_GPT_shortest_wire_length_l2344_234418

theorem shortest_wire_length (d1 d2 : ℝ) (r1 r2 : ℝ) (t : ℝ) :
  d1 = 8 ∧ d2 = 20 ∧ r1 = 4 ∧ r2 = 10 ∧ t = 8 * Real.sqrt 10 + 17.4 * Real.pi → 
  ∃ l : ℝ, l = t :=
by 
  sorry

end NUMINAMATH_GPT_shortest_wire_length_l2344_234418


namespace NUMINAMATH_GPT_ratio_of_men_to_women_l2344_234455
open Nat

theorem ratio_of_men_to_women 
  (total_players : ℕ) 
  (players_per_group : ℕ) 
  (extra_women_per_group : ℕ) 
  (H_total_players : total_players = 20) 
  (H_players_per_group : players_per_group = 3) 
  (H_extra_women_per_group : extra_women_per_group = 1) 
  : (7 / 13 : ℝ) = 7 / 13 :=
by
  -- Conditions
  have H1 : total_players = 20 := H_total_players
  have H2 : players_per_group = 3 := H_players_per_group
  have H3 : extra_women_per_group = 1 := H_extra_women_per_group
  -- The correct answer
  sorry

end NUMINAMATH_GPT_ratio_of_men_to_women_l2344_234455


namespace NUMINAMATH_GPT_eval_oplus_otimes_l2344_234406

-- Define the operations ⊕ and ⊗
def my_oplus (a b : ℕ) := a + b + 1
def my_otimes (a b : ℕ) := a * b - 1

-- Statement of the proof problem
theorem eval_oplus_otimes : my_oplus (my_oplus 5 7) (my_otimes 2 4) = 21 :=
by
  sorry

end NUMINAMATH_GPT_eval_oplus_otimes_l2344_234406


namespace NUMINAMATH_GPT_midpoint_sum_l2344_234494

theorem midpoint_sum (x1 y1 x2 y2 : ℕ) (h₁ : x1 = 4) (h₂ : y1 = 7) (h₃ : x2 = 12) (h₄ : y2 = 19) :
  (x1 + x2) / 2 + (y1 + y2) / 2 = 21 :=
by
  sorry

end NUMINAMATH_GPT_midpoint_sum_l2344_234494


namespace NUMINAMATH_GPT_solve_for_c_l2344_234440

variables (m c b a : ℚ) -- Declaring variables as rationals for added precision

theorem solve_for_c (h : m = (c * b * a) / (a - c)) : 
  c = (m * a) / (m + b * a) := 
by 
  sorry -- Proof not required as per the instructions

end NUMINAMATH_GPT_solve_for_c_l2344_234440


namespace NUMINAMATH_GPT_symmetric_point_x_axis_l2344_234416

theorem symmetric_point_x_axis (x y z : ℝ) : 
    (x, -y, -z) = (-2, -1, -9) :=
by 
  sorry

end NUMINAMATH_GPT_symmetric_point_x_axis_l2344_234416


namespace NUMINAMATH_GPT_remainder_of_8673_div_7_l2344_234425

theorem remainder_of_8673_div_7 : 8673 % 7 = 3 :=
by
  -- outline structure, proof to be inserted
  sorry

end NUMINAMATH_GPT_remainder_of_8673_div_7_l2344_234425


namespace NUMINAMATH_GPT_min_value_of_expression_l2344_234451

theorem min_value_of_expression :
  ∃ (a b : ℝ), (∃ (x : ℝ), 1 ≤ x ∧ x ≤ 2 ∧ x^2 + a * x + b - 3 = 0) ∧ a^2 + (b - 4)^2 = 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l2344_234451


namespace NUMINAMATH_GPT_count_multiples_4_6_not_5_9_l2344_234480

/-- The number of integers between 1 and 500 that are multiples of both 4 and 6 but not of either 5 or 9 is 22. -/
theorem count_multiples_4_6_not_5_9 :
  let lcm_4_6 := (Nat.lcm 4 6)
  let lcm_4_6_5 := (Nat.lcm lcm_4_6 5)
  let lcm_4_6_9 := (Nat.lcm lcm_4_6 9)
  let lcm_4_6_5_9 := (Nat.lcm lcm_4_6_5 9)
  let count_multiples (x : Nat) := (500 / x)
  count_multiples lcm_4_6 - count_multiples lcm_4_6_5 - count_multiples lcm_4_6_9 + count_multiples lcm_4_6_5_9 = 22 :=
by
  let lcm_4_6 := (Nat.lcm 4 6)
  let lcm_4_6_5 := (Nat.lcm lcm_4_6 5)
  let lcm_4_6_9 := (Nat.lcm lcm_4_6 9)
  let lcm_4_6_5_9 := (Nat.lcm lcm_4_6_5 9)
  let count_multiples (x : Nat) := (500 / x)
  show count_multiples lcm_4_6 - count_multiples lcm_4_6_5 - count_multiples lcm_4_6_9 + count_multiples lcm_4_6_5_9 = 22
  sorry

end NUMINAMATH_GPT_count_multiples_4_6_not_5_9_l2344_234480


namespace NUMINAMATH_GPT_joe_speed_l2344_234412

theorem joe_speed (pete_speed : ℝ) (joe_speed : ℝ) (time_run : ℝ) (distance : ℝ) 
  (h1 : joe_speed = 2 * pete_speed)
  (h2 : time_run = 2 / 3)
  (h3 : distance = 16)
  (h4 : distance = 3 * pete_speed * time_run) :
  joe_speed = 16 :=
by sorry

end NUMINAMATH_GPT_joe_speed_l2344_234412


namespace NUMINAMATH_GPT_similar_triangles_perimeters_and_area_ratios_l2344_234474

theorem similar_triangles_perimeters_and_area_ratios
  (m1 m2 : ℝ) (p_sum : ℝ) (ratio_p : ℝ) (ratio_a : ℝ) :
  m1 = 10 →
  m2 = 4 →
  p_sum = 140 →
  ratio_p = 5 / 2 →
  ratio_a = 25 / 4 →
  (∃ (p1 p2 : ℝ), p1 + p2 = p_sum ∧ p1 = (5 / 7) * p_sum ∧ p2 = (2 / 7) * p_sum ∧ ratio_a = (ratio_p)^2) :=
by
  sorry

end NUMINAMATH_GPT_similar_triangles_perimeters_and_area_ratios_l2344_234474


namespace NUMINAMATH_GPT_coin_flips_probability_equal_heads_l2344_234481

def fair_coin (p : ℚ) := p = 1 / 2
def second_coin (p : ℚ) := p = 3 / 5
def third_coin (p : ℚ) := p = 2 / 3

theorem coin_flips_probability_equal_heads :
  ∀ p1 p2 p3, fair_coin p1 → second_coin p2 → third_coin p3 →
  ∃ m n, m + n = 119 ∧ m / n = 29 / 90 :=
by
  sorry

end NUMINAMATH_GPT_coin_flips_probability_equal_heads_l2344_234481


namespace NUMINAMATH_GPT_wine_barrels_l2344_234464

theorem wine_barrels :
  ∃ x y : ℝ, (6 * x + 4 * y = 48) ∧ (5 * x + 3 * y = 38) :=
by
  -- Proof is left out
  sorry

end NUMINAMATH_GPT_wine_barrels_l2344_234464


namespace NUMINAMATH_GPT_count_blanks_l2344_234479

theorem count_blanks (B : ℝ) (h1 : 10 + B = T) (h2 : 0.7142857142857143 = B / T) : B = 25 :=
by
  -- The conditions are taken into account as definitions or parameters
  -- We skip the proof itself by using 'sorry'
  sorry

end NUMINAMATH_GPT_count_blanks_l2344_234479


namespace NUMINAMATH_GPT_last_digit_of_x95_l2344_234456

theorem last_digit_of_x95 (x : ℕ) : 
  (x^95 % 10) - (3^58 % 10) = 4 % 10 → (x^95 % 10 = 3) := by
  sorry

end NUMINAMATH_GPT_last_digit_of_x95_l2344_234456


namespace NUMINAMATH_GPT_gasoline_tank_capacity_l2344_234450

theorem gasoline_tank_capacity (x : ℕ) (h1 : 5 * x / 6 - 2 * x / 3 = 15) : x = 90 :=
sorry

end NUMINAMATH_GPT_gasoline_tank_capacity_l2344_234450


namespace NUMINAMATH_GPT_average_score_l2344_234488

theorem average_score (avg1 avg2 : ℕ) (matches1 matches2 : ℕ) (h_avg1 : avg1 = 60) (h_matches1 : matches1 = 10) (h_avg2 : avg2 = 70) (h_matches2 : matches2 = 15) : 
  (matches1 * avg1 + matches2 * avg2) / (matches1 + matches2) = 66 :=
by
  sorry

end NUMINAMATH_GPT_average_score_l2344_234488


namespace NUMINAMATH_GPT_point_not_in_image_of_plane_l2344_234490

def satisfies_plane (P : ℝ × ℝ × ℝ) (A B C D : ℝ) : Prop :=
  let (x, y, z) := P
  A * x + B * y + C * z + D = 0

theorem point_not_in_image_of_plane :
  let A := (2, -3, 1)
  let aA := 1
  let aB := 1
  let aC := -2
  let aD := 2
  let k := 5 / 2
  let a'A := aA
  let a'B := aB
  let a'C := aC
  let a'D := k * aD
  ¬ satisfies_plane A a'A a'B a'C a'D :=
by
  -- TODO: Proof needed
  sorry

end NUMINAMATH_GPT_point_not_in_image_of_plane_l2344_234490


namespace NUMINAMATH_GPT_integers_abs_no_greater_than_2_pos_div_by_3_less_than_10_non_neg_int_less_than_5_sum_eq_6_in_nat_expressing_sequence_l2344_234443

-- Proof problem 1
theorem integers_abs_no_greater_than_2 :
    {n : ℤ | |n| ≤ 2} = {-2, -1, 0, 1, 2} :=
by {
  sorry
}

-- Proof problem 2
theorem pos_div_by_3_less_than_10 :
    {n : ℕ | n > 0 ∧ n % 3 = 0 ∧ n < 10} = {3, 6, 9} :=
by {
  sorry
}

-- Proof problem 3
theorem non_neg_int_less_than_5 :
    {n : ℤ | n = |n| ∧ n < 5} = {0, 1, 2, 3, 4} :=
by {
  sorry
}

-- Proof problem 4
theorem sum_eq_6_in_nat :
    {p : ℕ × ℕ | p.1 + p.2 = 6 ∧ p.1 > 0 ∧ p.2 > 0} = {(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)} :=
by {
  sorry
}

-- Proof problem 5
theorem expressing_sequence:
    {-3, -1, 1, 3, 5} = {x : ℤ | ∃ k : ℤ, x = 2 * k - 1 ∧ -1 ≤ k ∧ k ≤ 3} :=
by {
  sorry
}

end NUMINAMATH_GPT_integers_abs_no_greater_than_2_pos_div_by_3_less_than_10_non_neg_int_less_than_5_sum_eq_6_in_nat_expressing_sequence_l2344_234443


namespace NUMINAMATH_GPT_train_constant_speed_is_48_l2344_234401

theorem train_constant_speed_is_48 
  (d_12_00 d_12_15 d_12_45 : ℝ)
  (h1 : 72.5 ≤ d_12_00 ∧ d_12_00 < 73.5)
  (h2 : 61.5 ≤ d_12_15 ∧ d_12_15 < 62.5)
  (h3 : 36.5 ≤ d_12_45 ∧ d_12_45 < 37.5)
  (constant_speed : ℝ → ℝ): 
  (constant_speed d_12_15 - constant_speed d_12_00 = 48) ∧
  (constant_speed d_12_45 - constant_speed d_12_15 = 48) :=
by
  sorry

end NUMINAMATH_GPT_train_constant_speed_is_48_l2344_234401


namespace NUMINAMATH_GPT_total_truck_loads_l2344_234409

-- Using definitions from conditions in (a)
def sand : ℝ := 0.16666666666666666
def dirt : ℝ := 0.3333333333333333
def cement : ℝ := 0.16666666666666666

-- The proof statement based on the correct answer in (b)
theorem total_truck_loads : sand + dirt + cement = 0.6666666666666666 := 
by
  sorry

end NUMINAMATH_GPT_total_truck_loads_l2344_234409


namespace NUMINAMATH_GPT_series_converges_to_l2344_234466

noncomputable def series_sum := ∑' n : Nat, (4 * n + 3) / ((4 * n + 1) ^ 2 * (4 * n + 5) ^ 2)

theorem series_converges_to : series_sum = 1 / 200 := 
by 
  sorry

end NUMINAMATH_GPT_series_converges_to_l2344_234466


namespace NUMINAMATH_GPT_distance_A_focus_l2344_234437

-- Definitions from the problem conditions
def parabola_eq (x y : ℝ) : Prop := x^2 = 4 * y
def point_A (x : ℝ) : Prop := parabola_eq x 4
def focus_y_coord : ℝ := 1 -- Derived from the standard form of the parabola x^2 = 4py where p=1

-- State the theorem in Lean 4
theorem distance_A_focus (x : ℝ) (hA : point_A x) : |4 - focus_y_coord| = 3 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_distance_A_focus_l2344_234437
