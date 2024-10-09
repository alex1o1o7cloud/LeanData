import Mathlib

namespace percent_equivalence_l894_89451

theorem percent_equivalence (y : ℝ) (h : y ≠ 0) : 0.21 * y = 0.21 * y :=
by sorry

end percent_equivalence_l894_89451


namespace range_of_a_l894_89462

def A (a : ℝ) : Set (ℝ × ℝ) := {p | ∃ (x y : ℝ), p = (x, y) ∧ x^2 + a * x - y + 2 = 0}
def B : Set (ℝ × ℝ) := {p | ∃ (x y : ℝ), p = (x, y) ∧ 2 * x - y + 1 = 0 ∧ x > 0}

theorem range_of_a (a : ℝ) : (∃ p, p ∈ A a ∧ p ∈ B) ↔ a ∈ Set.Iic 0 := by
  sorry

end range_of_a_l894_89462


namespace units_digit_42_pow_5_add_27_pow_5_l894_89424

theorem units_digit_42_pow_5_add_27_pow_5 :
  (42 ^ 5 + 27 ^ 5) % 10 = 9 :=
by
  sorry

end units_digit_42_pow_5_add_27_pow_5_l894_89424


namespace maximize_Miraflores_win_l894_89485

-- Definitions based on given conditions
def voters_count (n : ℕ) : ℕ := 2 * n
def support_Miraflores (n : ℕ) : ℕ := n + 1
def support_opponent (n : ℕ) : ℕ := n - 1

-- Theorem statement
theorem maximize_Miraflores_win (n : ℕ) (hn : n > 0) : 
  ∃ (d1 d2 : ℕ), d1 = 1 ∧ d2 = 2 * n - 1 ∧ support_Miraflores n > support_opponent n := 
sorry

end maximize_Miraflores_win_l894_89485


namespace gasoline_price_increase_l894_89436

theorem gasoline_price_increase
  (P Q : ℝ)
  (h1 : (P * Q) * 1.10 = P * (1 + X / 100) * Q * 0.88) :
  X = 25 :=
by
  -- proof here
  sorry

end gasoline_price_increase_l894_89436


namespace sqrt_x_minus_1_meaningful_l894_89410

theorem sqrt_x_minus_1_meaningful (x : ℝ) : (∃ y : ℝ, y = Real.sqrt (x - 1)) ↔ x ≥ 1 :=
by
  sorry

end sqrt_x_minus_1_meaningful_l894_89410


namespace bird_height_l894_89482

theorem bird_height (cat_height dog_height avg_height : ℕ) 
  (cat_height_eq : cat_height = 92)
  (dog_height_eq : dog_height = 94)
  (avg_height_eq : avg_height = 95) :
  let total_height := avg_height * 3 
  let bird_height := total_height - (cat_height + dog_height)
  bird_height = 99 := 
by
  sorry

end bird_height_l894_89482


namespace businesses_can_apply_l894_89450

-- Define conditions
def total_businesses : ℕ := 72
def businesses_fired : ℕ := 36 -- Half of total businesses (72 / 2)
def businesses_quit : ℕ := 24 -- One third of total businesses (72 / 3)

-- Theorem: Number of businesses Brandon can still apply to
theorem businesses_can_apply : (total_businesses - (businesses_fired + businesses_quit)) = 12 := 
by
  sorry

end businesses_can_apply_l894_89450


namespace necessary_but_not_sufficient_condition_l894_89483

variable (p q : Prop)

theorem necessary_but_not_sufficient_condition (hpq : p ∨ q) (h : p ∧ q) : p ∧ q ↔ (p ∨ q) := by
  sorry

end necessary_but_not_sufficient_condition_l894_89483


namespace watermelon_ratio_l894_89460

theorem watermelon_ratio (michael_weight : ℕ) (john_weight : ℕ) (clay_weight : ℕ)
  (h₁ : michael_weight = 8) 
  (h₂ : john_weight = 12) 
  (h₃ : john_weight * 2 = clay_weight) :
  clay_weight / michael_weight = 3 :=
by {
  sorry
}

end watermelon_ratio_l894_89460


namespace lesser_fraction_l894_89456

theorem lesser_fraction 
  (x y : ℚ)
  (h_sum : x + y = 13 / 14)
  (h_prod : x * y = 1 / 5) :
  min x y = 87 / 700 := sorry

end lesser_fraction_l894_89456


namespace exists_consecutive_nat_with_integer_quotient_l894_89449

theorem exists_consecutive_nat_with_integer_quotient :
  ∃ n : ℕ, (n + 1) / n = 2 :=
by
  sorry

end exists_consecutive_nat_with_integer_quotient_l894_89449


namespace time_spent_on_type_a_problems_l894_89479

theorem time_spent_on_type_a_problems 
  (total_problems : ℕ)
  (exam_time_minutes : ℕ)
  (type_a_problems : ℕ)
  (type_b_problem_time : ℕ)
  (total_time_type_a : ℕ)
  (h1 : total_problems = 200)
  (h2 : exam_time_minutes = 180)
  (h3 : type_a_problems = 50)
  (h4 : ∀ x : ℕ, type_b_problem_time = 2 * x)
  (h5 : ∀ x : ℕ, total_time_type_a = type_a_problems * type_b_problem_time)
  : total_time_type_a = 72 := 
by
  sorry

end time_spent_on_type_a_problems_l894_89479


namespace total_spent_on_index_cards_l894_89478

-- Definitions for conditions
def index_cards_per_student : ℕ := 10
def periods_per_day : ℕ := 6
def students_per_class : ℕ := 30
def cost_per_pack : ℕ := 3
def cards_per_pack : ℕ := 50

-- Theorem to be proven
theorem total_spent_on_index_cards :
  let total_students := students_per_class * periods_per_day
  let total_cards := total_students * index_cards_per_student
  let packs_needed := total_cards / cards_per_pack
  let total_cost := packs_needed * cost_per_pack
  total_cost = 108 :=
by
  sorry

end total_spent_on_index_cards_l894_89478


namespace find_a_l894_89430

def mul_op (a b : ℝ) : ℝ := 2 * a - b^2

theorem find_a (a : ℝ) (h : mul_op a 3 = 7) : a = 8 :=
sorry

end find_a_l894_89430


namespace circles_point_distance_l894_89486

noncomputable section

-- Define the data for the circles and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def CircleA (R : ℝ) (K : ℝ × ℝ) : Circle := 
  { center := K, radius := R }

def CircleB (R : ℝ) (K : ℝ × ℝ) : Circle := 
  { center := (K.1 + 2 * R, K.2), radius := R }

-- Define the condition that two circles touch each other at point K
def circles_touch (C1 C2 : Circle) (K : ℝ × ℝ) : Prop :=
  dist C1.center K = C1.radius ∧ dist C2.center K = C2.radius ∧ dist C1.center C2.center = C1.radius + C2.radius

-- Define the angle condition ∠AKB = 90°
def angle_AKB_is_right (A K B : ℝ × ℝ) : Prop :=
  -- Using the fact that a dot product being zero implies orthogonality
  let vec1 := (A.1 - K.1, A.2 - K.2)
  let vec2 := (B.1 - K.1, B.2 - K.2)
  vec1.1 * vec2.1 + vec1.2 * vec2.2 = 0

-- Define the points A and B being on their respective circles
def on_circle (A : ℝ × ℝ) (C : Circle) : Prop :=
  dist A C.center = C.radius

-- Define the theorem
theorem circles_point_distance 
  (R : ℝ) (K A B : ℝ × ℝ) 
  (C1 := CircleA R K) 
  (C2 := CircleB R K) 
  (h1 : circles_touch C1 C2 K) 
  (h2 : on_circle A C1) 
  (h3 : on_circle B C2) 
  (h4 : angle_AKB_is_right A K B) : 
  dist A B = 2 * R := 
sorry

end circles_point_distance_l894_89486


namespace thm1_thm2_thm3_thm4_l894_89419

variables {Point Line Plane : Type}
variables (m n : Line) (α β : Plane)

-- Definitions relating lines and planes
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_planes (p q : Plane) : Prop := sorry
def perpendicular_planes (p q : Plane) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry

-- Theorem 1: This statement is false, so we negate its for proof.
theorem thm1 (h1 : parallel_line_plane m α) (h2 : parallel_line_plane n β) (h3 : parallel_planes α β) :
  ¬ parallel_lines m n :=
sorry

-- Theorem 2: This statement is true, we need to prove it.
theorem thm2 (h1 : perpendicular_line_plane m α) (h2 : perpendicular_line_plane n β) (h3 : perpendicular_planes α β) :
  perpendicular_lines m n :=
sorry

-- Theorem 3: This statement is true, we need to prove it.
theorem thm3 (h1 : perpendicular_line_plane m α) (h2 : parallel_line_plane n β) (h3 : parallel_planes α β) :
  perpendicular_lines m n :=
sorry

-- Theorem 4: This statement is false, so we negate its for proof.
theorem thm4 (h1 : parallel_line_plane m α) (h2 : perpendicular_line_plane n β) (h3 : perpendicular_planes α β) :
  ¬ parallel_lines m n :=
sorry

end thm1_thm2_thm3_thm4_l894_89419


namespace treasures_found_second_level_l894_89458

theorem treasures_found_second_level:
  ∀ (P T1 S T2 : ℕ), 
    P = 4 → 
    T1 = 6 → 
    S = 32 → 
    S = P * T1 + P * T2 → 
    T2 = 2 := 
by
  intros P T1 S T2 hP hT1 hS hTotal
  sorry

end treasures_found_second_level_l894_89458


namespace expression_equality_l894_89420

theorem expression_equality :
  - (2^3) = (-2)^3 :=
by sorry

end expression_equality_l894_89420


namespace find_symmetric_point_l894_89484

structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def line_equation (t : ℝ) : Point :=
  { x := -t, y := 1.5, z := 2 + t }

def M : Point := { x := -1, y := 0, z := -1 }

def is_midpoint (M M' M0 : Point) : Prop :=
  M0.x = (M.x + M'.x) / 2 ∧
  M0.y = (M.y + M'.y) / 2 ∧
  M0.z = (M.z + M'.z) / 2

theorem find_symmetric_point (M0 : Point) (h_line : ∃ t, M0 = line_equation t) :
  ∃ M' : Point, is_midpoint M M' M0 ∧ M' = { x := 3, y := 3, z := 3 } :=
sorry

end find_symmetric_point_l894_89484


namespace min_platforms_needed_l894_89427

theorem min_platforms_needed :
  let slabs_7_tons := 120
  let slabs_9_tons := 80
  let weight_7_tons := 7
  let weight_9_tons := 9
  let max_weight_per_platform := 40
  let total_weight := slabs_7_tons * weight_7_tons + slabs_9_tons * weight_9_tons
  let platforms_needed_per_7_tons := slabs_7_tons / 3
  let platforms_needed_per_9_tons := slabs_9_tons / 2
  platforms_needed_per_7_tons = 40 ∧ platforms_needed_per_9_tons = 40 ∧ 3 * platforms_needed_per_7_tons = slabs_7_tons ∧ 2 * platforms_needed_per_9_tons = slabs_9_tons →
  platforms_needed_per_7_tons = 40 ∧ platforms_needed_per_9_tons = 40 :=
by
  sorry

end min_platforms_needed_l894_89427


namespace piravena_total_round_trip_cost_l894_89487

noncomputable def piravena_round_trip_cost : ℝ :=
  let distance_AB := 4000
  let bus_cost_per_km := 0.20
  let flight_cost_per_km := 0.12
  let flight_booking_fee := 120
  let flight_cost := distance_AB * flight_cost_per_km + flight_booking_fee
  let bus_cost := distance_AB * bus_cost_per_km
  flight_cost + bus_cost

theorem piravena_total_round_trip_cost : piravena_round_trip_cost = 1400 := by
  -- Problem conditions for reference:
  -- distance_AC = 3000
  -- distance_AB = 4000
  -- bus_cost_per_km = 0.20
  -- flight_cost_per_km = 0.12
  -- flight_booking_fee = 120
  -- Piravena decides to fly from A to B but returns by bus
  sorry

end piravena_total_round_trip_cost_l894_89487


namespace triangle_area_integral_bound_l894_89494

def S := 200
def AC := 20
def dist_A_to_tangent := 25
def dist_C_to_tangent := 16
def largest_integer_not_exceeding (S : ℕ) (n : ℕ) : ℕ := n

theorem triangle_area_integral_bound (AC : ℕ) (dist_A_to_tangent : ℕ) (dist_C_to_tangent : ℕ) (S : ℕ) : 
  AC = 20 ∧ dist_A_to_tangent = 25 ∧ dist_C_to_tangent = 16 → largest_integer_not_exceeding S 20 = 10 :=
by
  sorry

end triangle_area_integral_bound_l894_89494


namespace range_of_m_l894_89418

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, e ≤ x ∧ x ≤ e^2 ∧ f x - m * x - 1/2 + m ≤ 0) →
  1/2 ≤ m := by
  sorry

end range_of_m_l894_89418


namespace rectangle_area_l894_89408

theorem rectangle_area
  (x : ℝ)
  (perimeter_eq_160 : 10 * x = 160) :
  4 * (4 * x * x) = 1024 :=
by
  -- We would solve the problem and show the steps here
  sorry

end rectangle_area_l894_89408


namespace cara_total_bread_l894_89470

theorem cara_total_bread 
  (d : ℕ) (L : ℕ) (B : ℕ) (S : ℕ) 
  (h_dinner : d = 240) 
  (h_lunch : d = 8 * L) 
  (h_breakfast : d = 6 * B) 
  (h_snack : d = 4 * S) : 
  d + L + B + S = 370 := 
sorry

end cara_total_bread_l894_89470


namespace average_speed_trip_l894_89492

theorem average_speed_trip 
  (total_distance : ℕ)
  (first_distance : ℕ)
  (first_speed : ℕ)
  (second_distance : ℕ)
  (second_speed : ℕ)
  (h1 : total_distance = 60)
  (h2 : first_distance = 30)
  (h3 : first_speed = 60)
  (h4 : second_distance = 30)
  (h5 : second_speed = 30) :
  40 = total_distance / ((first_distance / first_speed) + (second_distance / second_speed)) :=
by sorry

end average_speed_trip_l894_89492


namespace factory_needs_to_produce_l894_89488

-- Define the given conditions
def weekly_production_target : ℕ := 6500
def production_mon_tue_wed : ℕ := 3 * 1200
def production_thu : ℕ := 800
def total_production_mon_thu := production_mon_tue_wed + production_thu
def required_production_fri := weekly_production_target - total_production_mon_thu

-- The theorem we need to prove
theorem factory_needs_to_produce : required_production_fri = 2100 :=
by
  -- The proof would go here
  sorry

end factory_needs_to_produce_l894_89488


namespace eval_expression_l894_89441

theorem eval_expression : (256 : ℝ) ^ ((-2 : ℝ) ^ (-3 : ℝ)) = 1 / 2 := by
  sorry

end eval_expression_l894_89441


namespace average_weight_l894_89435

def weights (A B C : ℝ) : Prop :=
  (A + B + C = 135) ∧
  (B + C = 86) ∧
  (B = 31)

theorem average_weight (A B C : ℝ) (h : weights A B C) :
  (A + B) / 2 = 40 :=
by
  sorry

end average_weight_l894_89435


namespace man_speed_3_kmph_l894_89498

noncomputable def bullet_train_length : ℝ := 200 -- The length of the bullet train in meters
noncomputable def bullet_train_speed_kmph : ℝ := 69 -- The speed of the bullet train in km/h
noncomputable def time_to_pass_man : ℝ := 10 -- The time taken to pass the man in seconds
noncomputable def conversion_factor_kmph_to_mps : ℝ := 1000 / 3600 -- Conversion factor from km/h to m/s
noncomputable def bullet_train_speed_mps : ℝ := bullet_train_speed_kmph * conversion_factor_kmph_to_mps -- Speed of the bullet train in m/s
noncomputable def relative_speed : ℝ := bullet_train_length / time_to_pass_man -- Relative speed at which train passes the man
noncomputable def speed_of_man_mps : ℝ := relative_speed - bullet_train_speed_mps -- Speed of the man in m/s
noncomputable def conversion_factor_mps_to_kmph : ℝ := 3.6 -- Conversion factor from m/s to km/h
noncomputable def speed_of_man_kmph : ℝ := speed_of_man_mps * conversion_factor_mps_to_kmph -- Speed of the man in km/h

theorem man_speed_3_kmph :
  speed_of_man_kmph = 3 :=
by
  sorry

end man_speed_3_kmph_l894_89498


namespace triangle_area_l894_89466

def right_triangle_area (hypotenuse leg1 : ℕ) : ℕ :=
  if (hypotenuse ^ 2 - leg1 ^ 2) > 0 then (1 / 2) * leg1 * (hypotenuse ^ 2 - leg1 ^ 2).sqrt else 0

theorem triangle_area (hypotenuse leg1 : ℕ) (h_hypotenuse : hypotenuse = 13) (h_leg1 : leg1 = 5) :
  right_triangle_area hypotenuse leg1 = 30 :=
by
  rw [h_hypotenuse, h_leg1]
  sorry

end triangle_area_l894_89466


namespace burn_all_bridges_mod_1000_l894_89477

theorem burn_all_bridges_mod_1000 :
  let m := 2013 * 2 ^ 2012
  let n := 3 ^ 2012
  (m + n) % 1000 = 937 :=
by
  sorry

end burn_all_bridges_mod_1000_l894_89477


namespace area_sum_eq_l894_89447

-- Define the conditions given in the problem
variables {A B C P Q R M N : Type*}

-- Define the properties of the points
variables (triangle_ABC : Triangle A B C)
          (point_P : OnSegment P A B)
          (point_Q : OnSegment Q B C)
          (point_R : OnSegment R A C)
          (parallelogram_PQCR : Parallelogram P Q C R)
          (intersection_M : Intersection M (LineSegment AQ) (LineSegment PR))
          (intersection_N : Intersection N (LineSegment BR) (LineSegment PQ))

-- Define the areas of the triangles involved
variables (area_AMP area_BNP area_CQR : ℝ)

-- Define the conditions for the areas of the triangles
variables (h_area_AMP : area_AMP = Area (Triangle A M P))
          (h_area_BNP : area_BNP = Area (Triangle B N P))
          (h_area_CQR : area_CQR = Area (Triangle C Q R))

-- The theorem to be proved
theorem area_sum_eq :
  area_AMP + area_BNP = area_CQR :=
sorry

end area_sum_eq_l894_89447


namespace zoe_total_expenditure_is_correct_l894_89481

noncomputable def zoe_expenditure : ℝ :=
  let initial_app_cost : ℝ := 5
  let monthly_fee : ℝ := 8
  let first_two_months_fee : ℝ := 2 * monthly_fee
  let yearly_cost_without_discount : ℝ := 12 * monthly_fee
  let discount : ℝ := 0.15 * yearly_cost_without_discount
  let discounted_annual_plan : ℝ := yearly_cost_without_discount - discount
  let actual_annual_plan : ℝ := discounted_annual_plan - first_two_months_fee
  let in_game_items_cost : ℝ := 10
  let discounted_in_game_items_cost : ℝ := in_game_items_cost - (0.10 * in_game_items_cost)
  let upgraded_feature_cost : ℝ := 12
  let discounted_upgraded_feature_cost : ℝ := upgraded_feature_cost - (0.10 * upgraded_feature_cost)
  initial_app_cost + first_two_months_fee + actual_annual_plan + discounted_in_game_items_cost + discounted_upgraded_feature_cost

theorem zoe_total_expenditure_is_correct : zoe_expenditure = 122.4 :=
by
  sorry

end zoe_total_expenditure_is_correct_l894_89481


namespace original_number_is_16_l894_89467

theorem original_number_is_16 (x : ℤ) (h1 : 3 * (2 * x + 5) = 111) : x = 16 :=
by
  sorry

end original_number_is_16_l894_89467


namespace lives_after_bonus_l894_89455

variable (X Y Z : ℕ)

theorem lives_after_bonus (X Y Z : ℕ) : (X - Y + 3 * Z) = (X - Y + 3 * Z) :=
sorry

end lives_after_bonus_l894_89455


namespace triangle_third_side_length_l894_89491

theorem triangle_third_side_length (x: ℕ) (h1: x % 2 = 0) (h2: 2 + 14 > x) (h3: 14 - 2 < x) : x = 14 :=
by 
  sorry

end triangle_third_side_length_l894_89491


namespace dollar_triple_60_l894_89461

-- Define the function $N
def dollar (N : Real) : Real :=
  0.4 * N + 2

-- Proposition proving that $$(($60)) = 6.96
theorem dollar_triple_60 : dollar (dollar (dollar 60)) = 6.96 := by
  sorry

end dollar_triple_60_l894_89461


namespace product_213_16_l894_89413

theorem product_213_16 :
  (213 * 16 = 3408) :=
by
  have h1 : (0.16 * 2.13 = 0.3408) := by sorry
  sorry

end product_213_16_l894_89413


namespace erasers_per_friend_l894_89406

variable (erasers friends : ℕ)

theorem erasers_per_friend (h1 : erasers = 3840) (h2 : friends = 48) :
  erasers / friends = 80 :=
by sorry

end erasers_per_friend_l894_89406


namespace candy_box_price_increase_l894_89425

theorem candy_box_price_increase
  (C : ℝ) -- Original price of the candy box
  (S : ℝ := 12) -- Original price of a can of soda
  (combined_price : C + S = 16) -- Combined price before increase
  (candy_box_increase : C + 0.25 * C = 1.25 * C) -- Price increase definition
  (soda_increase : S + 0.50 * S = 18) -- New price of soda after increase
  : 1.25 * C = 5 := sorry

end candy_box_price_increase_l894_89425


namespace train_length_l894_89464

-- Definitions based on conditions
def train_speed_kmh := 54 -- speed of the train in km/h
def time_to_cross_sec := 16 -- time to cross the telegraph post in seconds
def kmh_to_ms (speed_kmh : ℕ) : ℕ :=
  speed_kmh * 5 / 18 -- conversion factor from km/h to m/s

-- Prove that the length of the train is 240 meters
theorem train_length (h1 : train_speed_kmh = 54) (h2 : time_to_cross_sec = 16) : 
  (kmh_to_ms train_speed_kmh * time_to_cross_sec) = 240 := by
  sorry

end train_length_l894_89464


namespace find_value_of_a_l894_89428

theorem find_value_of_a (a : ℝ) :
  (∀ x y : ℝ, (ax + y - 1 = 0) → (x - y + 3 = 0) → (-a) * 1 = -1) → a = 1 :=
by
  sorry

end find_value_of_a_l894_89428


namespace line_symmetric_fixed_point_l894_89489

theorem line_symmetric_fixed_point (k : ℝ) :
  (∀ x, (∃ y, y = k * (x - 4))) →
  (∃ p : ℝ × ℝ, p = (2, 1) ∧ ∀ x, (∃ y, y = k * (x - 4))) →
  (∃ p : ℝ × ℝ, p = (2, 1)) →
  (∃ q : ℝ × ℝ, q = (0, 2)) →
  True := 
by sorry

end line_symmetric_fixed_point_l894_89489


namespace simplify_and_evaluate_l894_89474

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 2 + 1) :
  (1 + 1 / a) / ((a^2 - 1) / a) = (Real.sqrt 2 / 2) :=
by
  sorry

end simplify_and_evaluate_l894_89474


namespace original_number_is_19_l894_89454

theorem original_number_is_19 (x : ℤ) (h : (x + 4) % 23 = 0) : x = 19 := 
by 
  sorry

end original_number_is_19_l894_89454


namespace percentage_of_september_authors_l894_89437

def total_authors : ℕ := 120
def september_authors : ℕ := 15

theorem percentage_of_september_authors : 
  (september_authors / total_authors : ℚ) * 100 = 12.5 :=
by
  sorry

end percentage_of_september_authors_l894_89437


namespace remainder_1425_1427_1429_mod_12_l894_89495

theorem remainder_1425_1427_1429_mod_12 : 
  (1425 * 1427 * 1429) % 12 = 3 :=
by
  sorry

end remainder_1425_1427_1429_mod_12_l894_89495


namespace yellow_jelly_bean_probability_l894_89415

theorem yellow_jelly_bean_probability :
  let p_red := 0.15
  let p_orange := 0.35
  let p_green := 0.25
  let p_yellow := 1 - (p_red + p_orange + p_green)
  p_yellow = 0.25 := by
    let p_red := 0.15
    let p_orange := 0.35
    let p_green := 0.25
    let p_yellow := 1 - (p_red + p_orange + p_green)
    show p_yellow = 0.25
    sorry

end yellow_jelly_bean_probability_l894_89415


namespace general_formula_sequence_less_than_zero_maximum_sum_value_l894_89421

variable (n : ℕ)

-- Helper definition
def arithmetic_seq (d : ℤ) (a₁ : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

-- Conditions given in the problem
def a₁ : ℤ := 31
def a₄ : ℤ := 7
def d : ℤ := (a₄ - a₁) / 3

-- Definitions extracted from problem conditions
def an (n : ℕ) : ℤ := arithmetic_seq d a₁ n
def Sn (n : ℕ) : ℤ := n * a₁ + (n * (n - 1) / 2) * d

-- Proving the general formula aₙ = -8n + 39
theorem general_formula :
  ∀ (n : ℕ), an n = -8 * n + 39 :=
by
  sorry

-- Proving when the sequence starts to be less than 0
theorem sequence_less_than_zero :
  ∀ (n : ℕ), n ≥ 5 → an n < 0 :=
by
  sorry

-- Proving that the sum Sn has a maximum value
theorem maximum_sum_value :
  Sn 4 = 76 ∧ ∀ (n : ℕ), Sn n ≤ 76 :=
by
  sorry

end general_formula_sequence_less_than_zero_maximum_sum_value_l894_89421


namespace lisa_caffeine_l894_89444

theorem lisa_caffeine (caffeine_per_cup : ℕ) (daily_goal : ℕ) (cups_drank : ℕ) : caffeine_per_cup = 80 → daily_goal = 200 → cups_drank = 3 → (caffeine_per_cup * cups_drank - daily_goal) = 40 :=
by
  -- This is a theorem statement, thus no proof is provided here.
  sorry

end lisa_caffeine_l894_89444


namespace number_of_ways_to_fulfill_order_l894_89459

open Finset Nat

/-- Bill must buy exactly eight donuts from a shop offering five types, 
with at least two of the first type and one of each of the other four types. 
Prove that there are exactly 15 different ways to fulfill this order. -/
theorem number_of_ways_to_fulfill_order : 
  let total_donuts := 8
  let types_of_donuts := 5
  let mandatory_first_type := 2
  let mandatory_each_other_type := 1
  let remaining_donuts := total_donuts - (mandatory_first_type + 4 * mandatory_each_other_type)
  let combinations := (remaining_donuts + types_of_donuts - 1).choose (types_of_donuts - 1)
  combinations = 15 := 
by
  sorry

end number_of_ways_to_fulfill_order_l894_89459


namespace smallest_prime_divisor_of_3_pow_19_add_11_pow_23_l894_89412

theorem smallest_prime_divisor_of_3_pow_19_add_11_pow_23 :
  ∀ (n : ℕ), Prime n → n ∣ 3^19 + 11^23 → n = 2 :=
by
  sorry

end smallest_prime_divisor_of_3_pow_19_add_11_pow_23_l894_89412


namespace marys_number_l894_89499

theorem marys_number (j m : ℕ) (h₁ : j * m = 2002)
  (h₂ : ∃ k, k * m = 2002 ∧ k ≠ j)
  (h₃ : ∃ l, j * l = 2002 ∧ l ≠ m) :
  m = 1001 :=
sorry

end marys_number_l894_89499


namespace sale_in_fifth_month_l894_89452

theorem sale_in_fifth_month 
  (sale_month_1 : ℕ) (sale_month_2 : ℕ) (sale_month_3 : ℕ) (sale_month_4 : ℕ) 
  (sale_month_6 : ℕ) (average_sale : ℕ) 
  (h1 : sale_month_1 = 5266) (h2 : sale_month_2 = 5744) (h3 : sale_month_3 = 5864) 
  (h4 : sale_month_4 = 6122) (h6 : sale_month_6 = 4916) (h_avg : average_sale = 5750) :
  ∃ sale_month_5, sale_month_5 = 6588 :=
by
  sorry

end sale_in_fifth_month_l894_89452


namespace regular_polygon_sides_l894_89429

-- Define the measure of each exterior angle
def exterior_angle (n : ℕ) (angle : ℝ) : Prop :=
  angle = 40.0

-- Define the sum of exterior angles of any polygon
def sum_exterior_angles (n : ℕ) (total_angle : ℝ) : Prop :=
  total_angle = 360.0

-- Theorem to prove
theorem regular_polygon_sides (n : ℕ) :
  (exterior_angle n 40.0) ∧ (sum_exterior_angles n 360.0) → n = 9 :=
by
  sorry

end regular_polygon_sides_l894_89429


namespace apples_sold_l894_89416

theorem apples_sold (a1 a2 a3 : ℕ) (h1 : a3 = a2 / 4 + 8) (h2 : a2 = a1 / 4 + 8) (h3 : a3 = 18) : a1 = 128 :=
by
  sorry

end apples_sold_l894_89416


namespace pump_rates_l894_89476

theorem pump_rates (x y z : ℝ)
(h1 : x + y + z = 14)
(h2 : z = x + 3)
(h3 : y = 11 - 2 * x)
(h4 : 9 / x = (28 - 2 * y) / z)
: x = 3 ∧ y = 5 ∧ z = 6 :=
by
  sorry

end pump_rates_l894_89476


namespace seventh_graders_trip_count_l894_89475

theorem seventh_graders_trip_count (fifth_graders sixth_graders teachers_per_grade parents_per_grade grades buses seats_per_bus : ℕ) 
  (hf : fifth_graders = 109) 
  (hs : sixth_graders = 115)
  (ht : teachers_per_grade = 4) 
  (hp : parents_per_grade = 2) 
  (hg : grades = 3) 
  (hb : buses = 5)
  (hsb : seats_per_bus = 72) : 
  ∃ seventh_graders : ℕ, seventh_graders = 118 := 
by
  sorry

end seventh_graders_trip_count_l894_89475


namespace bead_count_l894_89453

variable (total_beads : ℕ) (blue_beads : ℕ) (red_beads : ℕ) (white_beads : ℕ) (silver_beads : ℕ)

theorem bead_count : total_beads = 40 ∧ blue_beads = 5 ∧ red_beads = 2 * blue_beads ∧ white_beads = blue_beads + red_beads ∧ silver_beads = total_beads - (blue_beads + red_beads + white_beads) → silver_beads = 10 :=
by
  intro h
  sorry

end bead_count_l894_89453


namespace terminating_decimal_expansion_l894_89468

theorem terminating_decimal_expansion (a b : ℝ) :
  (13 / 200 = a / 10^b) → a = 52 ∧ b = 3 ∧ a / 10^b = 0.052 :=
by sorry

end terminating_decimal_expansion_l894_89468


namespace ratio_of_rectangles_l894_89465

noncomputable def rect_ratio (a b c d e f : ℝ) 
  (h1: a / c = 3 / 5) 
  (h2: b / d = 3 / 5) 
  (h3: a / e = 7 / 4) 
  (h4: b / f = 7 / 4) : ℝ :=
  let A_A := a * b
  let A_B := (a * 5 / 3) * (b * 5 / 3)
  let A_C := (a * 4 / 7) * (b * 4 / 7)
  let A_BC := A_B + A_C
  A_A / A_BC

theorem ratio_of_rectangles (a b c d e f : ℝ) 
  (h1: a / c = 3 / 5) 
  (h2: b / d = 3 / 5) 
  (h3: a / e = 7 / 4) 
  (h4: b / f = 7 / 4) : 
  rect_ratio a b c d e f h1 h2 h3 h4 = 441 / 1369 :=
by
  sorry

end ratio_of_rectangles_l894_89465


namespace find_x_value_l894_89440

theorem find_x_value (x : ℝ) (h : 3 * x + 6 * x + x + 2 * x = 360) : x = 30 :=
by sorry

end find_x_value_l894_89440


namespace anna_score_below_90_no_A_l894_89497

def score_implies_grade (score : ℝ) : Prop :=
  score > 90 → true

theorem anna_score_below_90_no_A (score : ℝ) (A_grade : Prop) (h : score_implies_grade score) :
  score < 90 → ¬ A_grade :=
by sorry

end anna_score_below_90_no_A_l894_89497


namespace pam_total_apples_l894_89448

theorem pam_total_apples (pam_bags : ℕ) (gerald_bags_apples : ℕ) (gerald_bags_factor : ℕ) 
  (pam_bags_count : pam_bags = 10)
  (gerald_apples_count : gerald_bags_apples = 40)
  (gerald_bags_ratio : gerald_bags_factor = 3) : 
  pam_bags * gerald_bags_factor * gerald_bags_apples = 1200 := by
  sorry

end pam_total_apples_l894_89448


namespace parallel_lines_sufficient_not_necessary_condition_l894_89405

theorem parallel_lines_sufficient_not_necessary_condition {a : ℝ} :
  (a = 4) → (∀ x y : ℝ, (a * x + 8 * y - 3 = 0) ↔ (2 * x + a * y - a = 0)) :=
by sorry

end parallel_lines_sufficient_not_necessary_condition_l894_89405


namespace expression_evaluation_l894_89439

theorem expression_evaluation :
  5 * 423 + 4 * 423 + 3 * 423 + 421 = 5497 := by
  sorry

end expression_evaluation_l894_89439


namespace robot_steps_difference_zero_l894_89446

/-- Define the robot's position at second n --/
def robot_position (n : ℕ) : ℤ :=
  let cycle_length := 7
  let cycle_steps := 4 - 3
  let full_cycles := n / cycle_length
  let remainder := n % cycle_length
  full_cycles + if remainder = 0 then 0 else
    if remainder ≤ 4 then remainder else 4 - (remainder - 4)

/-- The main theorem to prove x_2007 - x_2011 = 0 --/
theorem robot_steps_difference_zero : 
  robot_position 2007 - robot_position 2011 = 0 :=
by sorry

end robot_steps_difference_zero_l894_89446


namespace find_number_of_children_l894_89472

theorem find_number_of_children (C B : ℕ) (H1 : B = 2 * C) (H2 : B = 4 * (C - 360)) : C = 720 := 
by
  sorry

end find_number_of_children_l894_89472


namespace find_a_l894_89411

theorem find_a (a : ℝ) : (dist (⟨-2, -1⟩ : ℝ × ℝ) (⟨a, 3⟩ : ℝ × ℝ) = 5) ↔ (a = 1 ∨ a = -5) :=
by
  sorry

end find_a_l894_89411


namespace second_discount_percentage_l894_89471

theorem second_discount_percentage 
    (original_price : ℝ) (final_price : ℝ) (first_discount : ℝ) (third_discount : ℝ) (second_discount : ℝ) :
      original_price = 9795.3216374269 →
      final_price = 6700 →
      first_discount = 0.20 →
      third_discount = 0.05 →
      (original_price * (1 - first_discount) * (1 - second_discount / 100) * (1 - third_discount) = final_price) →
      second_discount = 10 :=
by
  intros h_orig h_final h_first h_third h_eq
  sorry

end second_discount_percentage_l894_89471


namespace first_four_cards_all_red_l894_89445

noncomputable def probability_first_four_red_cards : ℚ :=
  (26 / 52) * (25 / 51) * (24 / 50) * (23 / 49)

theorem first_four_cards_all_red :
  probability_first_four_red_cards = 276 / 9801 :=
by
  -- The proof itself is not required; we are only stating it.
  sorry

end first_four_cards_all_red_l894_89445


namespace greatest_integer_value_l894_89423

theorem greatest_integer_value (x : ℤ) : 7 - 3 * x > 20 → x ≤ -5 :=
by
  intros h
  sorry

end greatest_integer_value_l894_89423


namespace avg_height_and_weight_of_class_l894_89434

-- Defining the given conditions
def num_students : ℕ := 70
def num_girls : ℕ := 40
def num_boys : ℕ := 30

def avg_height_30_girls : ℕ := 160
def avg_height_10_girls : ℕ := 156
def avg_height_15_boys_high : ℕ := 170
def avg_height_15_boys_low : ℕ := 160
def avg_weight_girls : ℕ := 55
def avg_weight_boys : ℕ := 60

-- Theorem stating the given question
theorem avg_height_and_weight_of_class :
  ∃ (avg_height avg_weight : ℚ),
    avg_height = (30 * 160 + 10 * 156 + 15 * 170 + 15 * 160) / num_students ∧
    avg_weight = (40 * 55 + 30 * 60) / num_students ∧
    avg_height = 161.57 ∧
    avg_weight = 57.14 :=
by
  -- include the solution steps here if required
  -- examples using appropriate constructs like ring, norm_num, etc.
  sorry

end avg_height_and_weight_of_class_l894_89434


namespace perfect_game_points_l894_89407

theorem perfect_game_points (points_per_game games_played total_points : ℕ) 
  (h1 : points_per_game = 21) 
  (h2 : games_played = 11) 
  (h3 : total_points = points_per_game * games_played) : 
  total_points = 231 := 
by 
  sorry

end perfect_game_points_l894_89407


namespace probability_of_qualification_l894_89422

-- Define the probability of hitting a target and the number of shots
def probability_hit : ℝ := 0.4
def number_of_shots : ℕ := 3

-- Define the probability of hitting a specific number of targets
noncomputable def binomial (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

-- Define the event of qualifying by hitting at least 2 targets
noncomputable def probability_qualify (n : ℕ) (p : ℝ) : ℝ :=
  binomial n 2 p + binomial n 3 p

-- The theorem we want to prove
theorem probability_of_qualification : probability_qualify number_of_shots probability_hit = 0.352 :=
  by sorry

end probability_of_qualification_l894_89422


namespace provisions_initial_days_l894_89402

theorem provisions_initial_days (D : ℕ) (P : ℕ) (Q : ℕ) (X : ℕ) (Y : ℕ)
  (h1 : P = 300) 
  (h2 : X = 30) 
  (h3 : Y = 90) 
  (h4 : Q = 200) 
  (h5 : P * D = P * X + Q * Y) : D + X = 120 :=
by
  -- We need to prove that the initial number of days the provisions were meant to last is 120.
  sorry

end provisions_initial_days_l894_89402


namespace mirror_full_body_view_l894_89414

theorem mirror_full_body_view (AB MN : ℝ) (h : AB > 0): 
  (MN = 1/2 * AB) ↔
  ∀ (P : ℝ), (0 < P) → (P < AB) → 
    (P < MN + (AB - P)) ∧ (P > AB - MN + P) := 
by
  sorry

end mirror_full_body_view_l894_89414


namespace quadrilateral_area_l894_89433

theorem quadrilateral_area {ABCQ : ℝ} 
  (side_length : ℝ) 
  (D P E N : ℝ → Prop) 
  (midpoints : ℝ) 
  (W X Y Z : ℝ → Prop) :
  side_length = 4 → 
  (∀ a b : ℝ, D a ∧ P b → a = 1 ∧ b = 1) → 
  (∀ c d : ℝ, E c ∧ N d → c = 1 ∧ d = 1) →
  (∀ w x y z : ℝ, W w ∧ X x ∧ Y y ∧ Z z → w = 0.5 ∧ x = 0.5 ∧ y = 0.5 ∧ z = 0.5) →
  ∃ (area : ℝ), area = 0.25 :=
by
  sorry

end quadrilateral_area_l894_89433


namespace fair_hair_percentage_l894_89463

-- Define the main entities
variables (E F W : ℝ)

-- Define the conditions given in the problem
def women_with_fair_hair : Prop := W = 0.32 * E
def fair_hair_women_ratio : Prop := W = 0.40 * F

-- Define the theorem to prove
theorem fair_hair_percentage
  (hwf: women_with_fair_hair E W)
  (fhr: fair_hair_women_ratio W F) :
  (F / E) * 100 = 80 :=
by
  sorry

end fair_hair_percentage_l894_89463


namespace clark_discount_l894_89490

noncomputable def price_per_part : ℕ := 80
noncomputable def num_parts : ℕ := 7
noncomputable def total_paid : ℕ := 439

theorem clark_discount : (price_per_part * num_parts - total_paid) = 121 :=
by
  -- proof goes here
  sorry

end clark_discount_l894_89490


namespace letters_symmetry_l894_89438

theorem letters_symmetry (people : Fin 20) (sends : Fin 20 → Finset (Fin 20)) (h : ∀ p, (sends p).card = 10) :
  ∃ i j : Fin 20, i ≠ j ∧ j ∈ sends i ∧ i ∈ sends j :=
by
  sorry

end letters_symmetry_l894_89438


namespace tangent_line_equation_l894_89404

theorem tangent_line_equation (e x y : ℝ) (h_curve : y = x^3 / e) (h_point : x = e ∧ y = e^2) :
  3 * e * x - y - 2 * e^2 = 0 :=
sorry

end tangent_line_equation_l894_89404


namespace range_of_a_l894_89432

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), 0 < x ∧ x < 4 → |x - 1| < a) ↔ 3 ≤ a :=
sorry

end range_of_a_l894_89432


namespace pirate_schooner_problem_l894_89431

theorem pirate_schooner_problem (p : ℕ) (h1 : 10 < p) 
  (h2 : 0.54 * (p - 10) = (54 : ℝ) / 100 * (p - 10)) 
  (h3 : 0.34 * (p - 10) = (34 : ℝ) / 100 * (p - 10)) 
  (h4 : 2 / 3 * p = (2 : ℝ) / 3 * p) : 
  p = 60 := 
sorry

end pirate_schooner_problem_l894_89431


namespace ducks_and_chickens_l894_89401

theorem ducks_and_chickens : 
  (∃ ducks chickens : ℕ, ducks = 7 ∧ chickens = 6 ∧ ducks + chickens = 13) :=
by
  sorry

end ducks_and_chickens_l894_89401


namespace Carla_more_miles_than_Daniel_after_5_hours_l894_89457

theorem Carla_more_miles_than_Daniel_after_5_hours (Carla_distance : ℝ) (Daniel_distance : ℝ) (h_Carla : Carla_distance = 100) (h_Daniel : Daniel_distance = 75) : 
  Carla_distance - Daniel_distance = 25 := 
by
  sorry

end Carla_more_miles_than_Daniel_after_5_hours_l894_89457


namespace infinitely_many_composite_numbers_l894_89443

-- We define n in a specialized form.
def n (m : ℕ) : ℕ := (3 * m) ^ 3

-- We state that m is an odd positive integer.
def odd_positive_integer (m : ℕ) : Prop := m > 0 ∧ (m % 2 = 1)

-- The main statement: for infinitely many odd values of n, 2^n + n - 1 is composite.
theorem infinitely_many_composite_numbers : 
  ∃ (m : ℕ), odd_positive_integer m ∧ Nat.Prime (n m) ∧ ∃ d : ℕ, d > 1 ∧ d < n m ∧ (2^(n m) + n m - 1) % d = 0 :=
by
  sorry

end infinitely_many_composite_numbers_l894_89443


namespace more_boys_than_girls_l894_89493

theorem more_boys_than_girls (total_people : ℕ) (num_girls : ℕ) (num_boys : ℕ) (more_boys : ℕ) : 
  total_people = 133 ∧ num_girls = 50 ∧ num_boys = total_people - num_girls ∧ more_boys = num_boys - num_girls → more_boys = 33 :=
by 
  sorry

end more_boys_than_girls_l894_89493


namespace range_of_x_l894_89417

theorem range_of_x (x a1 a2 y : ℝ) (d r : ℝ) (hx : x ≠ 0) 
  (h_arith : a1 = x + d ∧ a2 = x + 2 * d ∧ y = x + 3 * d)
  (h_geom : b1 = x * r ∧ b2 = x * r^2 ∧ y = x * r^3) : 4 ≤ x :=
by
  -- Assume x ≠ 0 as given and the sequences are arithmetic and geometric
  have hx3d := h_arith.2.2
  have hx3r := h_geom.2.2
  -- Substituting y in both sequences
  simp only [hx3d, hx3r] at *
  -- Solving for d and determining constraints
  sorry

end range_of_x_l894_89417


namespace linear_dependent_vectors_l894_89400

variable (m : ℝ) (a b : ℝ) 

theorem linear_dependent_vectors :
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ 
    a • (⟨2, 3⟩ : ℝ × ℝ) + b • (⟨5, m⟩ : ℝ × ℝ) = (⟨0, 0⟩ : ℝ × ℝ)) ↔ m = 15 / 2 :=
sorry

end linear_dependent_vectors_l894_89400


namespace binomial_7_2_l894_89426

open Nat

theorem binomial_7_2 : (Nat.choose 7 2) = 21 :=
by
  sorry

end binomial_7_2_l894_89426


namespace find_tan_half_sum_of_angles_l894_89473

theorem find_tan_half_sum_of_angles (x y : ℝ) 
  (h₁ : Real.cos x + Real.cos y = 1)
  (h₂ : Real.sin x + Real.sin y = 1 / 2) : 
  Real.tan ((x + y) / 2) = 1 / 2 := 
by 
  sorry

end find_tan_half_sum_of_angles_l894_89473


namespace probability_of_winning_l894_89496

-- Define the conditions
def total_tickets : ℕ := 10
def winning_tickets : ℕ := 3
def people : ℕ := 5
def losing_tickets : ℕ := total_tickets - winning_tickets

-- The probability calculation as per the conditions
def probability_at_least_one_wins : ℚ :=
  1 - ((Nat.choose losing_tickets people : ℚ) / (Nat.choose total_tickets people))

-- The statement to be proven
theorem probability_of_winning :
  probability_at_least_one_wins = 11 / 12 := 
sorry

end probability_of_winning_l894_89496


namespace mixtilinear_incircle_radius_l894_89480
open Real

variable (AB BC AC : ℝ)
variable (r_A : ℝ)

def triangle_conditions : Prop :=
  AB = 65 ∧ BC = 33 ∧ AC = 56

theorem mixtilinear_incircle_radius 
  (h : triangle_conditions AB BC AC)
  : r_A = 12.89 := 
sorry

end mixtilinear_incircle_radius_l894_89480


namespace actual_distance_traveled_l894_89442

-- Definitions based on conditions
def original_speed : ℕ := 12
def increased_speed : ℕ := 20
def distance_difference : ℕ := 24

-- We need to prove the actual distance traveled by the person.
theorem actual_distance_traveled : 
  ∃ t : ℕ, increased_speed * t = original_speed * t + distance_difference → original_speed * t = 36 :=
by
  sorry

end actual_distance_traveled_l894_89442


namespace smallest_C_l894_89409

-- Defining the problem and the conditions
theorem smallest_C (k : ℕ) (C : ℕ) :
  (∀ n : ℕ, n ≥ k → (C * Nat.choose (2 * n) (n + k)) % (n + k + 1) = 0) ↔
  C = 2 * k + 1 :=
by sorry

end smallest_C_l894_89409


namespace GregsAgeIs16_l894_89469

def CindyAge := 5
def JanAge := CindyAge + 2
def MarciaAge := 2 * JanAge
def GregAge := MarciaAge + 2

theorem GregsAgeIs16 : GregAge = 16 := by
  sorry

end GregsAgeIs16_l894_89469


namespace digit_D_value_l894_89403

/- The main conditions are:
1. A, B, C, D are digits (0 through 9)
2. Addition equation: AB + CA = D0
3. Subtraction equation: AB - CA = 00
-/

theorem digit_D_value (A B C D : ℕ) (hA : A < 10) (hB : B < 10) (hC : C < 10) (hD : D < 10)
  (add_eq : 10 * A + B + 10 * C + A = 10 * D + 0)
  (sub_eq : 10 * A + B - (10 * C + A) = 0) :
  D = 1 :=
sorry

end digit_D_value_l894_89403
