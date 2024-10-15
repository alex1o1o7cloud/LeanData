import Mathlib

namespace NUMINAMATH_GPT_emily_journey_length_l703_70320

theorem emily_journey_length
  (y : ℝ)
  (h1 : y / 5 + 30 + y / 3 + y / 6 = y) :
  y = 100 :=
by
  sorry

end NUMINAMATH_GPT_emily_journey_length_l703_70320


namespace NUMINAMATH_GPT_minimum_value_l703_70334

theorem minimum_value (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 128) : 
  ∃ (m : ℝ), (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → a * b * c = 128 → (a^2 + 8 * a * b + 4 * b^2 + 8 * c^2) ≥ m) 
  ∧ m = 384 :=
sorry


end NUMINAMATH_GPT_minimum_value_l703_70334


namespace NUMINAMATH_GPT_find_x_value_l703_70342

theorem find_x_value (X : ℕ) 
  (top_left : ℕ := 2)
  (top_second : ℕ := 3)
  (top_last : ℕ := 4)
  (bottom_left : ℕ := 3)
  (bottom_middle : ℕ := 5) 
  (top_sum_eq: 2 + 3 + X + 4 = 9 + X)
  (bottom_sum_eq: 3 + 5 + (X + 1) = 9 + X) : 
  X = 1 := by 
  sorry

end NUMINAMATH_GPT_find_x_value_l703_70342


namespace NUMINAMATH_GPT_minimum_value_sum_l703_70322

theorem minimum_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a / (b + 3 * c) + b / (8 * c + 4 * a) + 9 * c / (3 * a + 2 * b)) ≥ 47 / 48 :=
by sorry

end NUMINAMATH_GPT_minimum_value_sum_l703_70322


namespace NUMINAMATH_GPT_cylinder_height_relationship_l703_70339

theorem cylinder_height_relationship
  (r1 h1 r2 h2 : ℝ)
  (vol_eq : π * r1^2 * h1 = π * r2^2 * h2)
  (radius_rel : r2 = (6 / 5) * r1) : h1 = (36 / 25) * h2 := 
sorry

end NUMINAMATH_GPT_cylinder_height_relationship_l703_70339


namespace NUMINAMATH_GPT_range_of_a_l703_70368

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 2*x + a ≥ 0) ↔ (1 ≤ a) :=
by sorry

end NUMINAMATH_GPT_range_of_a_l703_70368


namespace NUMINAMATH_GPT_factorization_x3_minus_9xy2_l703_70376

theorem factorization_x3_minus_9xy2 (x y : ℝ) : x^3 - 9 * x * y^2 = x * (x + 3 * y) * (x - 3 * y) :=
by sorry

end NUMINAMATH_GPT_factorization_x3_minus_9xy2_l703_70376


namespace NUMINAMATH_GPT_find_length_of_EF_l703_70350

-- Definitions based on conditions
noncomputable def AB : ℝ := 300
noncomputable def DC : ℝ := 180
noncomputable def BC : ℝ := 200
noncomputable def E_as_fraction_of_BC : ℝ := (3 / 5)

-- Derived definition based on given conditions
noncomputable def EB : ℝ := E_as_fraction_of_BC * BC
noncomputable def EC : ℝ := BC - EB
noncomputable def EF : ℝ := (EC / BC) * DC

-- The theorem we need to prove
theorem find_length_of_EF : EF = 72 := by
  sorry

end NUMINAMATH_GPT_find_length_of_EF_l703_70350


namespace NUMINAMATH_GPT_original_water_amount_l703_70301

theorem original_water_amount (W : ℝ) 
    (evap_rate : ℝ := 0.03) 
    (days : ℕ := 22) 
    (evap_percent : ℝ := 0.055) 
    (total_evap : ℝ := evap_rate * days) 
    (evap_condition : evap_percent * W = total_evap) : W = 12 :=
by sorry

end NUMINAMATH_GPT_original_water_amount_l703_70301


namespace NUMINAMATH_GPT_james_initial_bars_l703_70385

def initial_chocolate_bars (sold_last_week sold_this_week needs_to_sell : ℕ) : ℕ :=
  sold_last_week + sold_this_week + needs_to_sell

theorem james_initial_bars : 
  initial_chocolate_bars 5 7 6 = 18 :=
by 
  sorry

end NUMINAMATH_GPT_james_initial_bars_l703_70385


namespace NUMINAMATH_GPT_area_one_magnet_is_150_l703_70333

noncomputable def area_one_magnet : ℕ :=
  let length := 15
  let total_circumference := 70
  let combined_width := (total_circumference / 2 - length) / 2
  let width := combined_width
  length * width

theorem area_one_magnet_is_150 :
  area_one_magnet = 150 :=
by
  -- This will skip the actual proof for now
  sorry

end NUMINAMATH_GPT_area_one_magnet_is_150_l703_70333


namespace NUMINAMATH_GPT_min_value_proof_l703_70348

noncomputable def min_value : ℝ := 3 + 2 * Real.sqrt 2

theorem min_value_proof (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : m + n = 1) :
  (1 / m + 2 / n) = min_value :=
sorry

end NUMINAMATH_GPT_min_value_proof_l703_70348


namespace NUMINAMATH_GPT_simplify_polynomial_l703_70388

def poly1 (x : ℝ) : ℝ := 5 * x^12 - 3 * x^9 + 6 * x^8 - 2 * x^7
def poly2 (x : ℝ) : ℝ := 7 * x^12 + 2 * x^11 - x^9 + 4 * x^7 + 2 * x^5 - x + 3
def expected (x : ℝ) : ℝ := 12 * x^12 + 2 * x^11 - 4 * x^9 + 6 * x^8 + 2 * x^7 + 2 * x^5 - x + 3

theorem simplify_polynomial (x : ℝ) : poly1 x + poly2 x = expected x :=
  by sorry

end NUMINAMATH_GPT_simplify_polynomial_l703_70388


namespace NUMINAMATH_GPT_faye_total_crayons_l703_70371

  def num_rows : ℕ := 16
  def crayons_per_row : ℕ := 6
  def total_crayons : ℕ := num_rows * crayons_per_row

  theorem faye_total_crayons : total_crayons = 96 :=
  by
  sorry
  
end NUMINAMATH_GPT_faye_total_crayons_l703_70371


namespace NUMINAMATH_GPT_value_is_20_l703_70316

-- Define the conditions
def number : ℕ := 5
def value := number + 3 * number

-- State the theorem
theorem value_is_20 : value = 20 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_value_is_20_l703_70316


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l703_70374

-- Define the first circle
def circle1 (m : ℝ) : Set (ℝ × ℝ) :=
  { p | (p.1 + m)^2 + p.2^2 = 1 }

-- Define the second circle
def circle2 : Set (ℝ × ℝ) :=
  { p | (p.1 - 2)^2 + p.2^2 = 4 }

-- Define the condition -1 ≤ m ≤ 1
def condition (m : ℝ) : Prop :=
  -1 ≤ m ∧ m ≤ 1

-- Define the property for circles having common points
def circlesHaveCommonPoints (m : ℝ) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ circle1 m ∧ p ∈ circle2

-- The final statement
theorem necessary_and_sufficient_condition (m : ℝ) :
  condition m → circlesHaveCommonPoints m ↔ (-5 ≤ m ∧ m ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l703_70374


namespace NUMINAMATH_GPT_total_pencils_l703_70382

-- Defining the number of pencils each person has.
def jessica_pencils : ℕ := 8
def sandy_pencils : ℕ := 8
def jason_pencils : ℕ := 8

-- Theorem stating the total number of pencils
theorem total_pencils : jessica_pencils + sandy_pencils + jason_pencils = 24 := by
  sorry

end NUMINAMATH_GPT_total_pencils_l703_70382


namespace NUMINAMATH_GPT_billy_scores_two_points_each_round_l703_70389

def billy_old_score := 725
def billy_rounds := 363
def billy_target_score := billy_old_score + 1
def billy_points_per_round := billy_target_score / billy_rounds

theorem billy_scores_two_points_each_round :
  billy_points_per_round = 2 := by
  sorry

end NUMINAMATH_GPT_billy_scores_two_points_each_round_l703_70389


namespace NUMINAMATH_GPT_find_x3_plus_y3_l703_70390

theorem find_x3_plus_y3 (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 + y^2 = 167) : x^3 + y^3 = 2005 :=
sorry

end NUMINAMATH_GPT_find_x3_plus_y3_l703_70390


namespace NUMINAMATH_GPT_Benjamin_skating_time_l703_70302

-- Definitions based on the conditions in the problem
def distance : ℕ := 80 -- Distance skated in kilometers
def speed : ℕ := 10 -- Speed in kilometers per hour

-- Theorem to prove that the skating time is 8 hours
theorem Benjamin_skating_time : distance / speed = 8 := by
  -- Proof goes here, we skip it with sorry
  sorry

end NUMINAMATH_GPT_Benjamin_skating_time_l703_70302


namespace NUMINAMATH_GPT_minimum_value_of_n_l703_70305

open Int

theorem minimum_value_of_n (n d : ℕ) (h1 : n > 0) (h2 : d > 0) (h3 : d % n = 0)
    (h4 : 10 * n - 20 = 90) : n = 11 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_n_l703_70305


namespace NUMINAMATH_GPT_wait_at_least_15_seconds_probability_l703_70367

-- Define the duration of the red light
def red_light_duration : ℕ := 40

-- Define the minimum waiting time for the green light
def min_wait_time : ℕ := 15

-- Define the duration after which pedestrian does not need to wait 15 seconds
def max_arrival_time : ℕ := red_light_duration - min_wait_time

-- Lean statement to prove the required probability
theorem wait_at_least_15_seconds_probability :
  (max_arrival_time : ℝ) / red_light_duration = 5 / 8 :=
by
  -- Proof omitted with sorry
  sorry

end NUMINAMATH_GPT_wait_at_least_15_seconds_probability_l703_70367


namespace NUMINAMATH_GPT_rotten_pineapples_l703_70358

theorem rotten_pineapples (initial sold fresh remaining rotten: ℕ) 
  (h1: initial = 86) 
  (h2: sold = 48) 
  (h3: fresh = 29) 
  (h4: remaining = initial - sold) 
  (h5: rotten = remaining - fresh) : 
  rotten = 9 := by 
  sorry

end NUMINAMATH_GPT_rotten_pineapples_l703_70358


namespace NUMINAMATH_GPT_sum_of_possible_values_l703_70331

theorem sum_of_possible_values (M : ℝ) (h : M * (M + 4) = 12) : M + (if M = -6 then 2 else -6) = -4 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_possible_values_l703_70331


namespace NUMINAMATH_GPT_tea_mixture_price_l703_70378

theorem tea_mixture_price :
  ∀ (price_A price_B : ℝ) (ratio_A ratio_B : ℝ),
  price_A = 65 →
  price_B = 70 →
  ratio_A = 1 →
  ratio_B = 1 →
  (price_A * ratio_A + price_B * ratio_B) / (ratio_A + ratio_B) = 67.5 :=
by
  intros price_A price_B ratio_A ratio_B h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_tea_mixture_price_l703_70378


namespace NUMINAMATH_GPT_find_certain_number_l703_70386

theorem find_certain_number (x : ℕ) (certain_number : ℕ)
  (h1 : certain_number * x = 675)
  (h2 : x = 27) : certain_number = 25 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_certain_number_l703_70386


namespace NUMINAMATH_GPT_greatest_roses_for_680_l703_70328

/--
Greatest number of roses that can be purchased for $680
given the following costs:
- $4.50 per individual rose
- $36 per dozen roses
- $50 per two dozen roses
--/
theorem greatest_roses_for_680 (cost_individual : ℝ) 
  (cost_dozen : ℝ) 
  (cost_two_dozen : ℝ) 
  (budget : ℝ) 
  (dozen : ℕ) 
  (two_dozen : ℕ) 
  (total_budget : ℝ) 
  (individual_cost : ℝ) 
  (dozen_cost : ℝ) 
  (two_dozen_cost : ℝ) 
  (roses_dozen : ℕ) 
  (roses_two_dozen : ℕ):
  individual_cost = 4.50 → dozen_cost = 36 → two_dozen_cost = 50 →
  budget = 680 → dozen = 12 → two_dozen = 24 →
  (∀ n : ℕ, n * two_dozen_cost ≤ budget → n * two_dozen + (budget - n * two_dozen_cost) / individual_cost ≤ total_budget) →
  total_budget = 318 := 
by
  sorry

end NUMINAMATH_GPT_greatest_roses_for_680_l703_70328


namespace NUMINAMATH_GPT_soccer_balls_donated_l703_70392

def num_classes_per_school (elem_classes mid_classes : ℕ) : ℕ :=
  elem_classes + mid_classes

def total_classes (num_schools : ℕ) (classes_per_school : ℕ) : ℕ :=
  num_schools * classes_per_school

def total_soccer_balls (num_classes : ℕ) (balls_per_class : ℕ) : ℕ :=
  num_classes * balls_per_class

theorem soccer_balls_donated 
  (elem_classes mid_classes num_schools balls_per_class : ℕ) 
  (h_elem_classes : elem_classes = 4) 
  (h_mid_classes : mid_classes = 5) 
  (h_num_schools : num_schools = 2) 
  (h_balls_per_class : balls_per_class = 5) :
  total_soccer_balls (total_classes num_schools (num_classes_per_school elem_classes mid_classes)) balls_per_class = 90 :=
by
  sorry

end NUMINAMATH_GPT_soccer_balls_donated_l703_70392


namespace NUMINAMATH_GPT_total_age_of_wines_l703_70377

theorem total_age_of_wines (age_carlo_rosi : ℕ) (age_franzia : ℕ) (age_twin_valley : ℕ) 
    (h1 : age_carlo_rosi = 40) (h2 : age_franzia = 3 * age_carlo_rosi) (h3 : age_carlo_rosi = 4 * age_twin_valley) : 
    age_franzia + age_carlo_rosi + age_twin_valley = 170 := 
by
    sorry

end NUMINAMATH_GPT_total_age_of_wines_l703_70377


namespace NUMINAMATH_GPT_max_cos_a_l703_70365

theorem max_cos_a (a b : ℝ) (h : Real.cos (a + b) = Real.cos a - Real.cos b) : 
  Real.cos a ≤ 1 := 
sorry

end NUMINAMATH_GPT_max_cos_a_l703_70365


namespace NUMINAMATH_GPT_five_level_pyramid_has_80_pieces_l703_70369

-- Definitions based on problem conditions
def rods_per_level (level : ℕ) : ℕ :=
  if level = 1 then 4
  else if level = 2 then 8
  else if level = 3 then 12
  else if level = 4 then 16
  else if level = 5 then 20
  else 0

def connectors_per_level_transition : ℕ := 4

-- The total rods used for a five-level pyramid
def total_rods_five_levels : ℕ :=
  rods_per_level 1 + rods_per_level 2 + rods_per_level 3 + rods_per_level 4 + rods_per_level 5

-- The total connectors used for a five-level pyramid
def total_connectors_five_levels : ℕ :=
  connectors_per_level_transition * 5

-- The total pieces required for a five-level pyramid
def total_pieces_five_levels : ℕ :=
  total_rods_five_levels + total_connectors_five_levels

-- Main theorem statement for the proof problem
theorem five_level_pyramid_has_80_pieces : 
  total_pieces_five_levels = 80 :=
by
  -- We expect the total_pieces_five_levels to be equal to 80
  sorry

end NUMINAMATH_GPT_five_level_pyramid_has_80_pieces_l703_70369


namespace NUMINAMATH_GPT_new_student_info_l703_70329

-- Definitions of the information pieces provided by each classmate.
structure StudentInfo where
  last_name : String
  gender : String
  total_score : Nat
  specialty : String

def student_A : StudentInfo := {
  last_name := "Ji",
  gender := "Male",
  total_score := 260,
  specialty := "Singing"
}

def student_B : StudentInfo := {
  last_name := "Zhang",
  gender := "Female",
  total_score := 220,
  specialty := "Dancing"
}

def student_C : StudentInfo := {
  last_name := "Chen",
  gender := "Male",
  total_score := 260,
  specialty := "Singing"
}

def student_D : StudentInfo := {
  last_name := "Huang",
  gender := "Female",
  total_score := 220,
  specialty := "Drawing"
}

def student_E : StudentInfo := {
  last_name := "Zhang",
  gender := "Female",
  total_score := 240,
  specialty := "Singing"
}

-- The theorem we need to prove based on the given conditions.
theorem new_student_info :
  ∃ info : StudentInfo,
    info.last_name = "Huang" ∧
    info.gender = "Male" ∧
    info.total_score = 240 ∧
    info.specialty = "Dancing" :=
  sorry

end NUMINAMATH_GPT_new_student_info_l703_70329


namespace NUMINAMATH_GPT_farmer_land_l703_70337

theorem farmer_land (initial_land remaining_land : ℚ) (h1 : initial_land - initial_land / 10 = remaining_land) (h2 : remaining_land = 10) : initial_land = 100 / 9 := by
  sorry

end NUMINAMATH_GPT_farmer_land_l703_70337


namespace NUMINAMATH_GPT_number_of_boxwoods_l703_70340

variables (x : ℕ)
def charge_per_trim := 5
def charge_per_shape := 15
def number_of_shaped_boxwoods := 4
def total_charge := 210
def total_shaping_charge := number_of_shaped_boxwoods * charge_per_shape

theorem number_of_boxwoods (h : charge_per_trim * x + total_shaping_charge = total_charge) : x = 30 :=
by
  sorry

end NUMINAMATH_GPT_number_of_boxwoods_l703_70340


namespace NUMINAMATH_GPT_cos_180_eq_neg_one_l703_70373

/-- The cosine of 180 degrees is -1. -/
theorem cos_180_eq_neg_one : Real.cos (180 * Real.pi / 180) = -1 :=
by sorry

end NUMINAMATH_GPT_cos_180_eq_neg_one_l703_70373


namespace NUMINAMATH_GPT_length_of_base_of_isosceles_triangle_l703_70355

noncomputable def length_congruent_sides : ℝ := 8
noncomputable def perimeter_triangle : ℝ := 26

theorem length_of_base_of_isosceles_triangle : 
  ∀ (b : ℝ), 
  2 * length_congruent_sides + b = perimeter_triangle → 
  b = 10 :=
by
  intros b h
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_length_of_base_of_isosceles_triangle_l703_70355


namespace NUMINAMATH_GPT_min_oranges_in_new_box_l703_70357

theorem min_oranges_in_new_box (m n : ℕ) (x : ℕ) (h1 : m + n ≤ 60) 
    (h2 : 59 * m = 60 * n + x) : x = 30 :=
sorry

end NUMINAMATH_GPT_min_oranges_in_new_box_l703_70357


namespace NUMINAMATH_GPT_parabola_point_distance_l703_70335

open Real

noncomputable def parabola_coords (y z: ℝ) : Prop :=
  y^2 = 12 * z

noncomputable def distance (x1 y1 x2 y2: ℝ) : ℝ :=
  sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem parabola_point_distance (x y: ℝ) :
  parabola_coords y x ∧ distance x y 3 0 = 9 ↔ ( x = 6 ∧ (y = 6 * sqrt 2 ∨ y = -6 * sqrt 2)) :=
by
  sorry

end NUMINAMATH_GPT_parabola_point_distance_l703_70335


namespace NUMINAMATH_GPT_product_of_last_two_digits_l703_70311

theorem product_of_last_two_digits (n : ℤ) (A B : ℤ) :
  (n % 8 = 0) ∧ (A + B = 15) ∧ (n % 10 = B) ∧ (n / 10 % 10 = A) →
  A * B = 54 :=
by
-- Add proof here
sorry

end NUMINAMATH_GPT_product_of_last_two_digits_l703_70311


namespace NUMINAMATH_GPT_points_divisibility_l703_70315

theorem points_divisibility {k n : ℕ} (hkn : k ≤ n) (hpositive : 0 < n) 
  (hcondition : ∀ x : Fin n, (∃ m : ℕ, (∀ y : Fin n, x.val ≤ y.val → y.val ≤ x.val + 1 → True) ∧ m % k = 0)) :
  k ∣ n :=
sorry

end NUMINAMATH_GPT_points_divisibility_l703_70315


namespace NUMINAMATH_GPT_man_speed_in_still_water_l703_70384

noncomputable def speed_of_man_in_still_water (vm vs : ℝ) : Prop :=
  -- Condition 1: v_m + v_s = 8
  vm + vs = 8 ∧
  -- Condition 2: v_m - v_s = 5
  vm - vs = 5

-- Proving the speed of the man in still water is 6.5 km/h
theorem man_speed_in_still_water : ∃ (v_m : ℝ), (∃ v_s : ℝ, speed_of_man_in_still_water v_m v_s) ∧ v_m = 6.5 :=
by
  sorry

end NUMINAMATH_GPT_man_speed_in_still_water_l703_70384


namespace NUMINAMATH_GPT_greatest_possible_value_l703_70346

theorem greatest_possible_value (x : ℝ) (hx : x^3 + (1 / x^3) = 9) : x + (1 / x) = 3 := by
  sorry

end NUMINAMATH_GPT_greatest_possible_value_l703_70346


namespace NUMINAMATH_GPT_discount_is_25_percent_l703_70338

noncomputable def discount_percentage (M : ℝ) (C : ℝ) (SP : ℝ) : ℝ :=
  ((M - SP) / M) * 100

theorem discount_is_25_percent (M : ℝ) (C : ℝ) (SP : ℝ) 
  (h1 : C = 0.64 * M) 
  (h2 : SP = C * 1.171875) : 
  discount_percentage M C SP = 25 := 
by 
  sorry

end NUMINAMATH_GPT_discount_is_25_percent_l703_70338


namespace NUMINAMATH_GPT_equal_sum_partition_l703_70360

theorem equal_sum_partition (n : ℕ) (a : Fin n.succ → ℕ)
  (h1 : a 0 = 1)
  (h2 : ∀ i : Fin n, a i ≤ a i.succ ∧ a i.succ ≤ 2 * a i)
  (h3 : (Finset.univ : Finset (Fin n.succ)).sum a % 2 = 0) :
  ∃ (partition : Finset (Fin n.succ)), 
    (partition.sum a = (partitionᶜ : Finset (Fin n.succ)).sum a) :=
by sorry

end NUMINAMATH_GPT_equal_sum_partition_l703_70360


namespace NUMINAMATH_GPT_inequality_true_for_all_real_l703_70304

theorem inequality_true_for_all_real (a : ℝ) : 
  3 * (1 + a^2 + a^4) ≥ (1 + a + a^2)^2 :=
sorry

end NUMINAMATH_GPT_inequality_true_for_all_real_l703_70304


namespace NUMINAMATH_GPT_oreo_shop_ways_l703_70341

theorem oreo_shop_ways (α β : ℕ) (products total_ways : ℕ) :
  let oreo_flavors := 6
  let milk_flavors := 4
  let total_flavors := oreo_flavors + milk_flavors
  (α + β = products) ∧ (products = 4) ∧ (total_ways = 2143) ∧ 
  (α ≤ 2 * total_flavors) ∧ (β ≤ 4 * oreo_flavors) →
  total_ways = 2143 :=
by sorry


end NUMINAMATH_GPT_oreo_shop_ways_l703_70341


namespace NUMINAMATH_GPT_contrapositive_equivalence_l703_70300
-- Importing the necessary libraries

-- Declaring the variables P and Q as propositions
variables (P Q : Prop)

-- The statement that we need to prove
theorem contrapositive_equivalence :
  (P → ¬ Q) ↔ (Q → ¬ P) :=
sorry

end NUMINAMATH_GPT_contrapositive_equivalence_l703_70300


namespace NUMINAMATH_GPT_expand_product_l703_70310

theorem expand_product (x : ℝ) :
  (3 * x + 4) * (2 * x - 5) = 6 * x^2 - 7 * x - 20 :=
sorry

end NUMINAMATH_GPT_expand_product_l703_70310


namespace NUMINAMATH_GPT_ethanol_percentage_in_fuel_B_l703_70372

theorem ethanol_percentage_in_fuel_B 
  (tank_capacity : ℕ)
  (fuel_A_vol : ℕ)
  (ethanol_in_A_percentage : ℝ)
  (ethanol_total : ℝ)
  (ethanol_A_vol : ℝ)
  (fuel_B_vol : ℕ)
  (ethanol_B_vol : ℝ)
  (ethanol_B_percentage : ℝ) 
  (h1 : tank_capacity = 204)
  (h2 : fuel_A_vol = 66)
  (h3 : ethanol_in_A_percentage = 0.12)
  (h4 : ethanol_total = 30)
  (h5 : ethanol_A_vol = fuel_A_vol * ethanol_in_A_percentage)
  (h6 : ethanol_B_vol = ethanol_total - ethanol_A_vol)
  (h7 : fuel_B_vol = tank_capacity - fuel_A_vol)
  (h8 : ethanol_B_percentage = (ethanol_B_vol / fuel_B_vol) * 100) :
  ethanol_B_percentage = 16 :=
by sorry

end NUMINAMATH_GPT_ethanol_percentage_in_fuel_B_l703_70372


namespace NUMINAMATH_GPT_ice_cream_not_sold_total_l703_70366

theorem ice_cream_not_sold_total :
  let chocolate_initial := 50
  let mango_initial := 54
  let vanilla_initial := 80
  let strawberry_initial := 40
  let chocolate_sold := (3 / 5 : ℚ) * chocolate_initial
  let mango_sold := (2 / 3 : ℚ) * mango_initial
  let vanilla_sold := (75 / 100 : ℚ) * vanilla_initial
  let strawberry_sold := (5 / 8 : ℚ) * strawberry_initial
  let chocolate_not_sold := chocolate_initial - chocolate_sold
  let mango_not_sold := mango_initial - mango_sold
  let vanilla_not_sold := vanilla_initial - vanilla_sold
  let strawberry_not_sold := strawberry_initial - strawberry_sold
  chocolate_not_sold + mango_not_sold + vanilla_not_sold + strawberry_not_sold = 73 :=
by sorry

end NUMINAMATH_GPT_ice_cream_not_sold_total_l703_70366


namespace NUMINAMATH_GPT_discount_difference_l703_70351

noncomputable def single_discount (amount : ℝ) (rate : ℝ) : ℝ :=
  amount * (1 - rate)

noncomputable def successive_discounts (amount : ℝ) (rates : List ℝ) : ℝ :=
  rates.foldl (λ acc rate => acc * (1 - rate)) amount

theorem discount_difference:
  let amount := 12000
  let single_rate := 0.35
  let successive_rates := [0.25, 0.08, 0.02]
  single_discount amount single_rate - successive_discounts amount successive_rates = 314.4 := 
  sorry

end NUMINAMATH_GPT_discount_difference_l703_70351


namespace NUMINAMATH_GPT_max_value_expr_l703_70399

theorem max_value_expr (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 4) : 
  10 * x + 3 * y + 15 * z ≤ 9.455 :=
sorry

end NUMINAMATH_GPT_max_value_expr_l703_70399


namespace NUMINAMATH_GPT_half_of_number_l703_70356

theorem half_of_number (N : ℕ) (h : (4 / 15 * 5 / 7 * N) - (4 / 9 * 2 / 5 * N) = 24) : N / 2 = 945 :=
by
  sorry

end NUMINAMATH_GPT_half_of_number_l703_70356


namespace NUMINAMATH_GPT_train_platform_length_l703_70343

theorem train_platform_length (train_length : ℕ) (platform_crossing_time : ℕ) (pole_crossing_time : ℕ) (length_of_platform : ℕ) :
  train_length = 300 →
  platform_crossing_time = 27 →
  pole_crossing_time = 18 →
  ((train_length * platform_crossing_time / pole_crossing_time) = train_length + length_of_platform) →
  length_of_platform = 150 :=
by
  intros h_train_length h_platform_time h_pole_time h_eq
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_train_platform_length_l703_70343


namespace NUMINAMATH_GPT_second_trial_temperatures_l703_70375

-- Definitions based on the conditions
def range_start : ℝ := 60
def range_end : ℝ := 70
def golden_ratio : ℝ := 0.618

-- Calculations for trial temperatures
def lower_trial_temp : ℝ := range_start + (range_end - range_start) * golden_ratio
def upper_trial_temp : ℝ := range_end - (range_end - range_start) * golden_ratio

-- Lean 4 statement to prove the trial temperatures
theorem second_trial_temperatures :
  lower_trial_temp = 66.18 ∧ upper_trial_temp = 63.82 :=
by
  sorry

end NUMINAMATH_GPT_second_trial_temperatures_l703_70375


namespace NUMINAMATH_GPT_log_comparison_l703_70381

theorem log_comparison (a b c : ℝ) 
  (h₁ : a = Real.log 6 / Real.log 3)
  (h₂ : b = Real.log 10 / Real.log 5)
  (h₃ : c = Real.log 14 / Real.log 7) :
  a > b ∧ b > c :=
  sorry

end NUMINAMATH_GPT_log_comparison_l703_70381


namespace NUMINAMATH_GPT_amount_solution_y_correct_l703_70345

-- Define conditions
def solution_x_alcohol_percentage : ℝ := 0.10
def solution_y_alcohol_percentage : ℝ := 0.30
def volume_solution_x : ℝ := 300.0
def target_alcohol_percentage : ℝ := 0.18

-- Define the main question as a theorem
theorem amount_solution_y_correct (y : ℝ) :
  (30 + 0.3 * y = 0.18 * (300 + y)) → y = 200 :=
by
  sorry

end NUMINAMATH_GPT_amount_solution_y_correct_l703_70345


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_l703_70307

-- Define the right triangle conditions and hypotenuse calculation
theorem right_triangle_hypotenuse (a b c : ℝ) (h1 : b = a + 3) (h2 : 1 / 2 * a * b = 120) :
  c^2 = 425 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_l703_70307


namespace NUMINAMATH_GPT_train_meetings_between_stations_l703_70361

theorem train_meetings_between_stations
  (travel_time : ℕ := 3 * 60 + 30) -- Travel time in minutes
  (first_departure : ℕ := 6 * 60) -- First departure time in minutes from 0 (midnight)
  (departure_interval : ℕ := 60) -- Departure interval in minutes
  (A_departure_time : ℕ := 9 * 60) -- Departure time from Station A at 9:00 AM in minutes
  :
  ∃ n : ℕ, n = 7 :=
by
  sorry

end NUMINAMATH_GPT_train_meetings_between_stations_l703_70361


namespace NUMINAMATH_GPT_sum_of_abc_is_33_l703_70396

theorem sum_of_abc_is_33 (a b c N : ℕ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c)
    (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hN1 : N = 5 * a + 3 * b + 5 * c)
    (hN2 : N = 4 * a + 5 * b + 4 * c) (hN_range : 131 < N ∧ N < 150) :
  a + b + c = 33 := 
sorry

end NUMINAMATH_GPT_sum_of_abc_is_33_l703_70396


namespace NUMINAMATH_GPT_shortest_path_length_l703_70379

theorem shortest_path_length (x y z : ℕ) (h1 : x + y = z + 1) (h2 : x + z = y + 5) (h3 : y + z = x + 7) : 
  min (min x y) z = 3 :=
by sorry

end NUMINAMATH_GPT_shortest_path_length_l703_70379


namespace NUMINAMATH_GPT_blackBurgerCost_l703_70309

def ArevaloFamilyBill (smokySalmonCost blackBurgerCost chickenKatsuCost totalBill : ℝ) : Prop :=
  smokySalmonCost = 40 ∧ chickenKatsuCost = 25 ∧ 
  totalBill = smokySalmonCost + blackBurgerCost + chickenKatsuCost + 
    0.15 * (smokySalmonCost + blackBurgerCost + chickenKatsuCost)

theorem blackBurgerCost (smokySalmonCost chickenKatsuCost change : ℝ) (B : ℝ) 
  (h1 : smokySalmonCost = 40) 
  (h2 : chickenKatsuCost = 25)
  (h3 : 100 - change = 92) 
  (h4 : ArevaloFamilyBill smokySalmonCost B chickenKatsuCost 92) : 
  B = 15 :=
sorry

end NUMINAMATH_GPT_blackBurgerCost_l703_70309


namespace NUMINAMATH_GPT_arithmetic_mean_location_l703_70317

theorem arithmetic_mean_location (a b : ℝ) : 
    abs ((a + b) / 2 - a) = abs (b - (a + b) / 2) := 
by 
    sorry

end NUMINAMATH_GPT_arithmetic_mean_location_l703_70317


namespace NUMINAMATH_GPT_total_quantities_l703_70330

theorem total_quantities (N : ℕ) (S S₃ S₂ : ℕ)
  (h1 : S = 12 * N)
  (h2 : S₃ = 12)
  (h3 : S₂ = 48)
  (h4 : S = S₃ + S₂) :
  N = 5 :=
by
  sorry

end NUMINAMATH_GPT_total_quantities_l703_70330


namespace NUMINAMATH_GPT_A_share_of_profit_l703_70314

section InvestmentProfit

variables (capitalA capitalB : ℕ) -- initial capitals
variables (withdrawA advanceB : ℕ) -- changes after 8 months
variables (profit : ℕ) -- total profit

def investment_months (initial : ℕ) (final : ℕ) (first_period : ℕ) (second_period : ℕ) : ℕ :=
  initial * first_period + final * second_period

def ratio (a b : ℕ) : ℚ := (a : ℚ) / (b : ℚ)

def A_share (total_profit : ℕ) (ratioA ratioB : ℚ) : ℚ :=
  (ratioA / (ratioA + ratioB)) * total_profit

theorem A_share_of_profit :
  let capitalA := 3000
  let capitalB := 4000
  let withdrawA := 1000
  let advanceB := 1000
  let profit := 756
  let A_investment_months := investment_months capitalA (capitalA - withdrawA) 8 4
  let B_investment_months := investment_months capitalB (capitalB + advanceB) 8 4
  let ratioA := ratio A_investment_months B_investment_months
  let ratioB := ratio B_investment_months A_investment_months
  A_share profit ratioA ratioB = 288 := sorry

end InvestmentProfit

end NUMINAMATH_GPT_A_share_of_profit_l703_70314


namespace NUMINAMATH_GPT_area_of_room_l703_70318

def length : ℝ := 12
def width : ℝ := 8

theorem area_of_room : length * width = 96 :=
by sorry

end NUMINAMATH_GPT_area_of_room_l703_70318


namespace NUMINAMATH_GPT_T_n_sum_general_term_b_b_n_comparison_l703_70380

noncomputable def sequence_a (n : ℕ) : ℕ := sorry  -- Placeholder for sequence {a_n}
noncomputable def S (n : ℕ) : ℕ := sorry  -- Placeholder for sum of first n terms S_n
noncomputable def sequence_b (n : ℕ) (q : ℝ) : ℝ := sorry  -- Placeholder for sequence {b_n}

axiom sequence_a_def : ∀ n : ℕ, 2 * sequence_a (n + 1) = sequence_a n + sequence_a (n + 2)
axiom sequence_a_5 : sequence_a 5 = 5
axiom S_7 : S 7 = 28

noncomputable def T (n : ℕ) : ℝ := (2 * n : ℝ) / (n + 1 : ℝ)

theorem T_n_sum : ∀ n : ℕ, T n = 2 * (1 - 1 / (n + 1)) := sorry

axiom b1 : ℝ
axiom b_def : ∀ (n : ℕ) (q : ℝ), q > 0 → sequence_b (n + 1) q = sequence_b n q + q ^ (sequence_a n)

theorem general_term_b (q : ℝ) (n : ℕ) (hq : q > 0) : 
  (if q = 1 then sequence_b n q = n else sequence_b n q = (1 - q ^ n) / (1 - q)) := sorry

theorem b_n_comparison (q : ℝ) (n : ℕ) (hq : q > 0) : 
  sequence_b n q * sequence_b (n + 2) q < (sequence_b (n + 1) q) ^ 2 := sorry

end NUMINAMATH_GPT_T_n_sum_general_term_b_b_n_comparison_l703_70380


namespace NUMINAMATH_GPT_jordan_purchase_total_rounded_l703_70387

theorem jordan_purchase_total_rounded :
  let p1 := 2.49
  let p2 := 6.51
  let p3 := 11.49
  let r1 := 2 -- rounded value of p1
  let r2 := 7 -- rounded value of p2
  let r3 := 11 -- rounded value of p3
  r1 + r2 + r3 = 20 :=
by
  let p1 := 2.49
  let p2 := 6.51
  let p3 := 11.49
  let r1 := 2
  let r2 := 7
  let r3 := 11
  show r1 + r2 + r3 = 20
  sorry

end NUMINAMATH_GPT_jordan_purchase_total_rounded_l703_70387


namespace NUMINAMATH_GPT_batsman_average_after_17th_inning_l703_70349

theorem batsman_average_after_17th_inning :
  ∀ (A : ℕ), (16 * A + 50) / 17 = A + 2 → A = 16 → A + 2 = 18 := by
  intros A h1 h2
  rw [h2] at h1
  linarith

end NUMINAMATH_GPT_batsman_average_after_17th_inning_l703_70349


namespace NUMINAMATH_GPT_reading_comprehension_application_method_1_application_method_2_l703_70391

-- Reading Comprehension Problem in Lean 4
theorem reading_comprehension (x : ℝ) (h : x^2 + x + 5 = 8) : 2 * x^2 + 2 * x - 4 = 2 :=
by sorry

-- Application of Methods Problem (1) in Lean 4
theorem application_method_1 (x : ℝ) (h : x^2 + x + 2 = 9) : -2 * x^2 - 2 * x + 3 = -11 :=
by sorry

-- Application of Methods Problem (2) in Lean 4
theorem application_method_2 (a b : ℝ) (h : 8 * a + 2 * b = 5) : a * (-2)^3 + b * (-2) + 3 = -2 :=
by sorry

end NUMINAMATH_GPT_reading_comprehension_application_method_1_application_method_2_l703_70391


namespace NUMINAMATH_GPT_sqrt_two_irrational_l703_70321

theorem sqrt_two_irrational : ¬ ∃ (p q : ℕ), (q ≠ 0) ∧ (Nat.gcd p q = 1) ∧ (p ^ 2 = 2 * q ^ 2) := by
  sorry

end NUMINAMATH_GPT_sqrt_two_irrational_l703_70321


namespace NUMINAMATH_GPT_ratio_after_addition_l703_70370

theorem ratio_after_addition (a b : ℕ) (h1 : a * 3 = b * 2) (h2 : b - a = 8) : (a + 4) * 7 = (b + 4) * 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_after_addition_l703_70370


namespace NUMINAMATH_GPT_charlie_share_l703_70332

variable (A B C : ℝ)

theorem charlie_share :
  A = (1/3) * B →
  B = (1/2) * C →
  A + B + C = 10000 →
  C = 6000 :=
by
  intros hA hB hSum
  sorry

end NUMINAMATH_GPT_charlie_share_l703_70332


namespace NUMINAMATH_GPT_valid_values_of_X_Y_l703_70347

-- Stating the conditions
def odd_combinations := 125
def even_combinations := 64
def revenue_diff (X Y : ℕ) := odd_combinations * X - even_combinations * Y = 5
def valid_limit (n : ℕ) := 0 < n ∧ n < 250

-- The theorem we want to prove
theorem valid_values_of_X_Y (X Y : ℕ) :
  revenue_diff X Y ∧ valid_limit X ∧ valid_limit Y ↔ (X = 41 ∧ Y = 80) ∨ (X = 105 ∧ Y = 205) :=
  sorry

end NUMINAMATH_GPT_valid_values_of_X_Y_l703_70347


namespace NUMINAMATH_GPT_negation_proposition_l703_70397

theorem negation_proposition (x y : ℝ) :
  (¬ (x^2 + y^2 = 0 → x = 0 ∧ y = 0)) ↔ (x^2 + y^2 ≠ 0 → (x ≠ 0 ∨ y ≠ 0)) := 
sorry

end NUMINAMATH_GPT_negation_proposition_l703_70397


namespace NUMINAMATH_GPT_scaled_multiplication_l703_70398

theorem scaled_multiplication (h : 268 * 74 = 19832) : 2.68 * 0.74 = 1.9832 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_scaled_multiplication_l703_70398


namespace NUMINAMATH_GPT_find_percentage_l703_70352

theorem find_percentage (P N : ℕ) (h₁ : N = 125) (h₂ : N = (P * N / 100) + 105) : P = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_percentage_l703_70352


namespace NUMINAMATH_GPT_cookout_kids_2004_l703_70327

variable (kids2005 kids2004 kids2006 : ℕ)

theorem cookout_kids_2004 :
  (kids2006 = 20) →
  (2 * kids2005 = 3 * kids2006) →
  (2 * kids2004 = kids2005) →
  kids2004 = 60 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_cookout_kids_2004_l703_70327


namespace NUMINAMATH_GPT_combined_cost_l703_70394

variable (bench_cost : ℝ) (table_cost : ℝ)

-- Conditions
axiom bench_cost_def : bench_cost = 250.0
axiom table_cost_def : table_cost = 2 * bench_cost

-- Goal
theorem combined_cost (bench_cost : ℝ) (table_cost : ℝ) 
  (h1 : bench_cost = 250.0) (h2 : table_cost = 2 * bench_cost) : 
  table_cost + bench_cost = 750.0 :=
by
  sorry

end NUMINAMATH_GPT_combined_cost_l703_70394


namespace NUMINAMATH_GPT_cats_eat_fish_l703_70362

theorem cats_eat_fish (c d: ℕ) (h1 : 1 < c) (h2 : c < 10) (h3 : c * d = 91) : c + d = 20 := by
  sorry

end NUMINAMATH_GPT_cats_eat_fish_l703_70362


namespace NUMINAMATH_GPT_minimize_PA2_PB2_PC2_l703_70323

def point : Type := ℝ × ℝ

noncomputable def distance_sq (P Q : point) : ℝ := 
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

noncomputable def PA_sq (P : point) : ℝ := distance_sq P (5, 0)
noncomputable def PB_sq (P : point) : ℝ := distance_sq P (0, 5)
noncomputable def PC_sq (P : point) : ℝ := distance_sq P (-4, -3)

noncomputable def circumcircle (P : point) : Prop := 
  P.1^2 + P.2^2 = 25

noncomputable def objective_function (P : point) : ℝ := 
  PA_sq P + PB_sq P + PC_sq P

theorem minimize_PA2_PB2_PC2 : ∃ P : point, circumcircle P ∧ 
  (∀ Q : point, circumcircle Q → objective_function P ≤ objective_function Q) :=
sorry

end NUMINAMATH_GPT_minimize_PA2_PB2_PC2_l703_70323


namespace NUMINAMATH_GPT_initial_worth_is_30_l703_70312

-- Definitions based on conditions
def numberOfCoinsLeft := 2
def amountLeft := 12

-- Definition of the value of each gold coin based on amount left and number of coins left
def valuePerCoin : ℕ := amountLeft / numberOfCoinsLeft

-- Define the total worth of sold coins
def soldCoinsWorth (coinsSold : ℕ) : ℕ := coinsSold * valuePerCoin

-- The total initial worth of Roman's gold coins
def totalInitialWorth : ℕ := amountLeft + soldCoinsWorth 3

-- The proof goal
theorem initial_worth_is_30 : totalInitialWorth = 30 :=
by
  sorry

end NUMINAMATH_GPT_initial_worth_is_30_l703_70312


namespace NUMINAMATH_GPT_sum_of_bases_l703_70324

theorem sum_of_bases (R1 R2 : ℕ)
  (h1 : ∀ F1 : ℚ, F1 = (4 * R1 + 8) / (R1 ^ 2 - 1) → F1 = (5 * R2 + 9) / (R2 ^ 2 - 1))
  (h2 : ∀ F2 : ℚ, F2 = (8 * R1 + 4) / (R1 ^ 2 - 1) → F2 = (9 * R2 + 5) / (R2 ^ 2 - 1)) :
  R1 + R2 = 24 :=
sorry

end NUMINAMATH_GPT_sum_of_bases_l703_70324


namespace NUMINAMATH_GPT_three_point_seven_five_minus_one_point_four_six_l703_70326

theorem three_point_seven_five_minus_one_point_four_six : 3.75 - 1.46 = 2.29 :=
by sorry

end NUMINAMATH_GPT_three_point_seven_five_minus_one_point_four_six_l703_70326


namespace NUMINAMATH_GPT_relationship_points_l703_70313

noncomputable def is_on_inverse_proportion (m x y : ℝ) : Prop :=
  y = (-m^2 - 2) / x

theorem relationship_points (a b c m : ℝ) :
  is_on_inverse_proportion m a (-1) ∧
  is_on_inverse_proportion m b 2 ∧
  is_on_inverse_proportion m c 3 →
  a > c ∧ c > b :=
by
  sorry

end NUMINAMATH_GPT_relationship_points_l703_70313


namespace NUMINAMATH_GPT_coordinate_of_M_l703_70353

-- Definition and given conditions
def L : ℚ := 1 / 6
def P : ℚ := 1 / 12

def divides_into_three_equal_parts (L P M N : ℚ) : Prop :=
  M = L + (P - L) / 3 ∧ N = L + 2 * (P - L) / 3

theorem coordinate_of_M (M N : ℚ) 
  (h1 : divides_into_three_equal_parts L P M N) : 
  M = 1 / 9 := 
by 
  sorry
  
end NUMINAMATH_GPT_coordinate_of_M_l703_70353


namespace NUMINAMATH_GPT_price_of_light_bulb_and_motor_l703_70363

theorem price_of_light_bulb_and_motor
  (x : ℝ) (motor_price : ℝ)
  (h1 : x + motor_price = 12)
  (h2 : 10 / x = 2 * 45 / (12 - x)) :
  x = 3 ∧ motor_price = 9 :=
sorry

end NUMINAMATH_GPT_price_of_light_bulb_and_motor_l703_70363


namespace NUMINAMATH_GPT_candy_box_original_price_l703_70395

theorem candy_box_original_price (P : ℝ) (h₁ : 1.25 * P = 10) : P = 8 := 
sorry

end NUMINAMATH_GPT_candy_box_original_price_l703_70395


namespace NUMINAMATH_GPT_range_of_a_l703_70325

theorem range_of_a (a : ℝ) : (∀ (x : ℝ), x > 0 → x / (x ^ 2 + 3 * x + 1) ≤ a) → a ≥ 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l703_70325


namespace NUMINAMATH_GPT_richard_remaining_distance_l703_70306

noncomputable def remaining_distance : ℝ :=
  let d1 := 45
  let d2 := d1 / 2 - 8
  let d3 := 2 * d2 - 4
  let d4 := (d1 + d2 + d3) / 3 + 3
  let d5 := 0.7 * d4
  let total_walked := d1 + d2 + d3 + d4 + d5
  635 - total_walked

theorem richard_remaining_distance : abs (remaining_distance - 497.5166) < 0.0001 :=
by
  sorry

end NUMINAMATH_GPT_richard_remaining_distance_l703_70306


namespace NUMINAMATH_GPT_second_assistant_smoked_pipes_l703_70393

theorem second_assistant_smoked_pipes
    (x y z : ℚ)
    (H1 : (2 / 3) * x = (4 / 9) * y)
    (H2 : x + y = 1)
    (H3 : (x + z) / (y - z) = y / x) :
    z = 1 / 5 → x = 2 / 5 ∧ y = 3 / 5 →
    ∀ n : ℕ, n = 5 :=
by
  sorry

end NUMINAMATH_GPT_second_assistant_smoked_pipes_l703_70393


namespace NUMINAMATH_GPT_det_of_commuting_matrices_l703_70364

theorem det_of_commuting_matrices (n : ℕ) (hn : n ≥ 2) (A B : Matrix (Fin n) (Fin n) ℝ)
  (hA : A * A = -1) (hAB : A * B = B * A) : 
  0 ≤ B.det := 
sorry

end NUMINAMATH_GPT_det_of_commuting_matrices_l703_70364


namespace NUMINAMATH_GPT_expression_is_five_l703_70336

-- Define the expression
def given_expression : ℤ := abs (abs (-abs (-2 + 1) - 2) + 2)

-- Prove that the expression equals 5
theorem expression_is_five : given_expression = 5 :=
by
  -- We skip the proof for now
  sorry

end NUMINAMATH_GPT_expression_is_five_l703_70336


namespace NUMINAMATH_GPT_profit_450_l703_70383

-- Define the conditions
def cost_per_garment : ℕ := 40
def wholesale_price : ℕ := 60

-- Define the piecewise function for wholesale price P
noncomputable def P (x : ℕ) : ℕ :=
  if h : 0 < x ∧ x ≤ 100 then wholesale_price
  else if h : 100 < x ∧ x ≤ 500 then 62 - x / 50
  else 0

-- Define the profit function L
noncomputable def L (x : ℕ) : ℕ :=
  if h : 0 < x ∧ x ≤ 100 then (P x - cost_per_garment) * x
  else if h : 100 < x ∧ x ≤ 500 then (22 * x - x^2 / 50)
  else 0

-- State the theorem
theorem profit_450 : L 450 = 5850 :=
by
  sorry

end NUMINAMATH_GPT_profit_450_l703_70383


namespace NUMINAMATH_GPT_rectangle_dimensions_l703_70359

theorem rectangle_dimensions (w l : ℕ) 
  (h1 : l = 2 * w) 
  (h2 : 2 * (w + l) = 6 * w ^ 2) : 
  w = 1 ∧ l = 2 :=
by sorry

end NUMINAMATH_GPT_rectangle_dimensions_l703_70359


namespace NUMINAMATH_GPT_minimum_value_of_y_l703_70303

theorem minimum_value_of_y (x : ℝ) (h : x > 0) : (∃ y, y = (x^2 + 1) / x ∧ y ≥ 2) ∧ (∃ y, y = (x^2 + 1) / x ∧ y = 2) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_y_l703_70303


namespace NUMINAMATH_GPT_quadratic_solution_sum_l703_70308

theorem quadratic_solution_sum (m n p : ℕ) (h : m.gcd (n.gcd p) = 1)
  (h₀ : ∀ x, x * (5 * x - 11) = -6 ↔ x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p) :
  m + n + p = 70 :=
sorry

end NUMINAMATH_GPT_quadratic_solution_sum_l703_70308


namespace NUMINAMATH_GPT_minimum_value_of_expression_l703_70354

noncomputable def min_value_expr (x y z : ℝ) : ℝ := (x + 3 * y) * (y + 3 * z) * (2 * x * z + 1)

theorem minimum_value_of_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) :
  min_value_expr x y z = 24 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l703_70354


namespace NUMINAMATH_GPT_complement_intersection_l703_70319

noncomputable def U : Set ℤ := {-1, 0, 2}
noncomputable def A : Set ℤ := {-1, 2}
noncomputable def B : Set ℤ := {0, 2}
noncomputable def C_U_A : Set ℤ := U \ A

theorem complement_intersection :
  (C_U_A ∩ B) = {0} :=
by {
  -- sorry to skip the proof part as per instruction
  sorry
}

end NUMINAMATH_GPT_complement_intersection_l703_70319


namespace NUMINAMATH_GPT_top_angle_isosceles_triangle_l703_70344

open Real

theorem top_angle_isosceles_triangle (A B C : ℝ) (abc_is_isosceles : (A = B ∨ B = C ∨ A = C))
  (angle_A : A = 40) : (B = 40 ∨ B = 100) :=
sorry

end NUMINAMATH_GPT_top_angle_isosceles_triangle_l703_70344
