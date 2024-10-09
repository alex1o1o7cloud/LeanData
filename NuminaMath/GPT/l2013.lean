import Mathlib

namespace algebraic_expression_value_l2013_201347

open Real

theorem algebraic_expression_value (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49 / (x - 3)^2 = 23 :=
sorry

end algebraic_expression_value_l2013_201347


namespace boxes_calculation_l2013_201368

theorem boxes_calculation (total_bottles : ℕ) (bottles_per_bag : ℕ) (bags_per_box : ℕ) (boxes : ℕ) :
  total_bottles = 8640 → bottles_per_bag = 12 → bags_per_box = 6 → boxes = total_bottles / (bottles_per_bag * bags_per_box) → boxes = 120 :=
by
  intros h_total h_bottles_per_bag h_bags_per_box h_boxes
  rw [h_total, h_bottles_per_bag, h_bags_per_box] at h_boxes
  norm_num at h_boxes
  exact h_boxes

end boxes_calculation_l2013_201368


namespace train_cross_signal_pole_time_l2013_201334

theorem train_cross_signal_pole_time :
  ∀ (train_length platform_length platform_cross_time signal_cross_time : ℝ),
  train_length = 300 →
  platform_length = 300 →
  platform_cross_time = 36 →
  signal_cross_time = train_length / ((train_length + platform_length) / platform_cross_time) →
  signal_cross_time = 18 :=
by
  intros train_length platform_length platform_cross_time signal_cross_time h_train_length h_platform_length h_platform_cross_time h_signal_cross_time
  rw [h_train_length, h_platform_length, h_platform_cross_time] at h_signal_cross_time
  sorry

end train_cross_signal_pole_time_l2013_201334


namespace range_of_a_l2013_201301

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 0 → 4 * a > x^2 - x^3) → a > 1 / 27 :=
by
  -- Proof to be filled
  sorry

end range_of_a_l2013_201301


namespace m_squared_plus_reciprocal_squared_l2013_201363

theorem m_squared_plus_reciprocal_squared (m : ℝ) (h : m^2 - 2 * m - 1 = 0) : m^2 + 1 / m^2 = 6 :=
by
  sorry

end m_squared_plus_reciprocal_squared_l2013_201363


namespace symmetric_complex_division_l2013_201382

theorem symmetric_complex_division :
  (∀ (z1 z2 : ℂ), z1 = 3 - (1 : ℂ) * Complex.I ∧ z2 = -(Complex.re z1) + (Complex.im z1) * Complex.I 
   → (z1 / z2) = -4/5 + (3/5) * Complex.I) := sorry

end symmetric_complex_division_l2013_201382


namespace library_visits_l2013_201320

theorem library_visits
  (william_visits_per_week : ℕ := 2)
  (jason_visits_per_week : ℕ := 4 * william_visits_per_week)
  (emma_visits_per_week : ℕ := 3 * jason_visits_per_week)
  (zoe_visits_per_week : ℕ := william_visits_per_week / 2)
  (chloe_visits_per_week : ℕ := emma_visits_per_week / 3)
  (jason_total_visits : ℕ := jason_visits_per_week * 8)
  (emma_total_visits : ℕ := emma_visits_per_week * 8)
  (zoe_total_visits : ℕ := zoe_visits_per_week * 8)
  (chloe_total_visits : ℕ := chloe_visits_per_week * 8)
  (total_visits : ℕ := jason_total_visits + emma_total_visits + zoe_total_visits + chloe_total_visits) :
  total_visits = 328 := by
  sorry

end library_visits_l2013_201320


namespace find_cost_of_article_l2013_201309

-- Define the given conditions and the corresponding proof statement.
theorem find_cost_of_article
  (tax_rate : ℝ) (selling_price1 : ℝ)
  (selling_price2 : ℝ) (profit_increase_rate : ℝ)
  (cost : ℝ) : tax_rate = 0.05 →
              selling_price1 = 360 →
              selling_price2 = 340 →
              profit_increase_rate = 0.05 →
              (selling_price1 / (1 + tax_rate) - cost = 1.05 * (selling_price2 / (1 + tax_rate) - cost)) →
              cost = 57.13 :=
by sorry

end find_cost_of_article_l2013_201309


namespace equilateral_triangle_coloring_l2013_201345

theorem equilateral_triangle_coloring (color : Fin 3 → Prop) :
  (∀ i, color i = true ∨ color i = false) →
  ∃ i j : Fin 3, i ≠ j ∧ color i = color j :=
by
  sorry

end equilateral_triangle_coloring_l2013_201345


namespace speed_of_second_train_l2013_201302

def speed_of_first_train := 40 -- speed of the first train in kmph
def distance_from_mumbai := 120 -- distance from Mumbai where the trains meet in km
def head_start_time := 1 -- head start time in hours for the first train
def total_remaining_distance := distance_from_mumbai - speed_of_first_train * head_start_time -- remaining distance for the first train to travel in km after head start
def time_to_meet_first_train := total_remaining_distance / speed_of_first_train -- time in hours for the first train to reach the meeting point after head start
def second_train_meeting_time := time_to_meet_first_train -- the second train takes the same time to meet the first train
def distance_covered_by_second_train := distance_from_mumbai -- same meeting point distance for second train from Mumbai

theorem speed_of_second_train : 
  ∃ v : ℝ, v = distance_covered_by_second_train / second_train_meeting_time ∧ v = 60 :=
by
  sorry

end speed_of_second_train_l2013_201302


namespace repaved_before_today_correct_l2013_201390

variable (total_repaved_so_far repaved_today repaved_before_today : ℕ)

axiom given_conditions : total_repaved_so_far = 4938 ∧ repaved_today = 805 

theorem repaved_before_today_correct :
  total_repaved_so_far = 4938 →
  repaved_today = 805 →
  repaved_before_today = total_repaved_so_far - repaved_today →
  repaved_before_today = 4133 :=
by
  intros
  sorry

end repaved_before_today_correct_l2013_201390


namespace simplify_expression_l2013_201311

-- Define the given condition as a hypothesis
theorem simplify_expression (a b c : ℝ) (h : a + b + c = 0) :
  a * (1 / b + 1 / c) + b * (1 / c + 1 / a) + c * (1 / a + 1 / b) + 3 = 0 :=
by
  sorry -- Proof will be provided here.

end simplify_expression_l2013_201311


namespace limit_to_infinity_zero_l2013_201337

variable (f : ℝ → ℝ)

theorem limit_to_infinity_zero (h_continuous : Continuous f)
  (h_alpha : ∀ (α : ℝ), α > 0 → Filter.Tendsto (fun n : ℕ => f (n * α)) Filter.atTop (nhds 0)) :
  Filter.Tendsto f Filter.atTop (nhds 0) :=
sorry

end limit_to_infinity_zero_l2013_201337


namespace angle_between_vectors_is_45_degrees_l2013_201373

-- Define the vectors
def u : ℝ × ℝ := (4, -1)
def v : ℝ × ℝ := (5, 3)

-- Define the theorem to prove the angle between these vectors is 45 degrees
theorem angle_between_vectors_is_45_degrees : 
  let dot_product := (4 * 5) + (-1 * 3)
  let norm_u := Real.sqrt ((4^2) + (-1)^2)
  let norm_v := Real.sqrt ((5^2) + (3^2))
  let cos_theta := dot_product / (norm_u * norm_v)
  let theta := Real.arccos cos_theta
  45 = (theta * 180 / Real.pi) :=
by
  sorry

end angle_between_vectors_is_45_degrees_l2013_201373


namespace fruit_mix_apples_count_l2013_201300

variable (a o b p : ℕ)

theorem fruit_mix_apples_count :
  a + o + b + p = 240 →
  o = 3 * a →
  b = 2 * o →
  p = 5 * b →
  a = 6 :=
by
  intros h1 h2 h3 h4
  sorry

end fruit_mix_apples_count_l2013_201300


namespace domain_of_f_monotonicity_of_f_l2013_201362

noncomputable def f (a x : ℝ) := Real.log (a ^ x - 1) / Real.log a

theorem domain_of_f (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  (a > 1 → ∀ x : ℝ, f a x ∈ Set.Ioi 0) ∧ (0 < a ∧ a < 1 → ∀ x : ℝ, f a x ∈ Set.Iio 0) :=
sorry

theorem monotonicity_of_f (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  (a > 1 → StrictMono (f a)) ∧ (0 < a ∧ a < 1 → StrictMono (f a)) :=
sorry

end domain_of_f_monotonicity_of_f_l2013_201362


namespace toys_produced_on_sunday_l2013_201321

-- Given conditions
def factory_production (day: ℕ) : ℕ :=
  2500 + 25 * day

theorem toys_produced_on_sunday : factory_production 6 = 2650 :=
by {
  -- The proof steps are omitted as they are not required.
  sorry
}

end toys_produced_on_sunday_l2013_201321


namespace total_volume_of_all_cubes_l2013_201387

/-- Carl has 4 cubes each with a side length of 3 -/
def carl_cubes_side_length := 3
def carl_cubes_count := 4

/-- Kate has 6 cubes each with a side length of 4 -/
def kate_cubes_side_length := 4
def kate_cubes_count := 6

/-- Total volume of 10 cubes with given conditions -/
theorem total_volume_of_all_cubes : 
  carl_cubes_count * (carl_cubes_side_length ^ 3) + 
  kate_cubes_count * (kate_cubes_side_length ^ 3) = 492 := by
  sorry

end total_volume_of_all_cubes_l2013_201387


namespace pascal_triangle_41_l2013_201384

theorem pascal_triangle_41:
  ∃ (n : Nat), ∀ (k : Nat), n = 41 ∧ (Nat.choose n k = 41) :=
sorry

end pascal_triangle_41_l2013_201384


namespace difference_between_numbers_l2013_201356

theorem difference_between_numbers (x y : ℕ) 
  (h1 : x + y = 20000) 
  (h2 : y = 7 * x) : y - x = 15000 :=
by
  sorry

end difference_between_numbers_l2013_201356


namespace sum_divisible_by_7_l2013_201392

theorem sum_divisible_by_7 (n : ℕ) : (8^n + 6) % 7 = 0 := 
by
  sorry

end sum_divisible_by_7_l2013_201392


namespace perfect_square_trinomial_k_l2013_201329

theorem perfect_square_trinomial_k (k : ℤ) :
  (∃ (a b : ℤ), (a * x + b) ^ 2 = x ^ 2 + k * x + 9) → (k = 6 ∨ k = -6) :=
by
  sorry

end perfect_square_trinomial_k_l2013_201329


namespace sum_of_volumes_of_two_cubes_l2013_201316

-- Definitions for edge length and volume formula
def edge_length : ℕ := 5

def volume (s : ℕ) : ℕ := s ^ 3

-- Statement to prove the sum of volumes of two cubes with edge length 5 cm
theorem sum_of_volumes_of_two_cubes : volume edge_length + volume edge_length = 250 :=
by
  sorry

end sum_of_volumes_of_two_cubes_l2013_201316


namespace families_with_neither_l2013_201339

theorem families_with_neither (total_families : ℕ) (families_with_cats : ℕ) (families_with_dogs : ℕ) (families_with_both : ℕ) :
  total_families = 40 → families_with_cats = 18 → families_with_dogs = 24 → families_with_both = 10 → 
  total_families - (families_with_cats + families_with_dogs - families_with_both) = 8 :=
by
  intros h1 h2 h3 h4
  sorry

end families_with_neither_l2013_201339


namespace parallelogram_area_l2013_201393

theorem parallelogram_area (base height : ℝ) (h_base : base = 10) (h_height : height = 20) :
  base * height = 200 := 
by 
  sorry

end parallelogram_area_l2013_201393


namespace P_inter_M_l2013_201352

def set_P : Set ℝ := {x | 0 ≤ x ∧ x < 3}
def set_M : Set ℝ := {x | x^2 ≤ 9}

theorem P_inter_M :
  set_P ∩ set_M = {x | 0 ≤ x ∧ x < 3} := sorry

end P_inter_M_l2013_201352


namespace kenneth_money_left_l2013_201370

theorem kenneth_money_left (I : ℕ) (C_b : ℕ) (N_b : ℕ) (C_w : ℕ) (N_w : ℕ) (L : ℕ) :
  I = 50 → C_b = 2 → N_b = 2 → C_w = 1 → N_w = 2 → L = I - (N_b * C_b + N_w * C_w) → L = 44 :=
by
  intros h₀ h₁ h₂ h₃ h₄ h₅
  sorry

end kenneth_money_left_l2013_201370


namespace find_function_l2013_201396

theorem find_function (f : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → f x + 2 * f (1 / x) = 3 * x) : 
  ∀ x : ℝ, x ≠ 0 → f x = -x + 2 / x := 
by
  sorry

end find_function_l2013_201396


namespace graph_does_not_pass_through_second_quadrant_l2013_201380

theorem graph_does_not_pass_through_second_quadrant :
  ¬ ∃ x : ℝ, x < 0 ∧ 2 * x - 3 > 0 :=
by
  -- Include the necessary steps to complete the proof, but for now we provide a placeholder:
  sorry

end graph_does_not_pass_through_second_quadrant_l2013_201380


namespace correct_option_is_B_l2013_201327

def natural_growth_rate (birth_rate death_rate : ℕ) : ℕ :=
  birth_rate - death_rate

def option_correct (birth_rate death_rate : ℕ) :=
  (∃ br dr, natural_growth_rate br dr = br - dr)

theorem correct_option_is_B (birth_rate death_rate : ℕ) :
  option_correct birth_rate death_rate :=
by 
  sorry

end correct_option_is_B_l2013_201327


namespace alpha_quadrant_l2013_201331

variable {α : ℝ}

theorem alpha_quadrant
  (sin_alpha_neg : Real.sin α < 0)
  (tan_alpha_pos : Real.tan α > 0) :
  ∃ k : ℤ, k = 1 ∧ π < α - 2 * π * k ∧ α - 2 * π * k < 3 * π :=
by
  sorry

end alpha_quadrant_l2013_201331


namespace fraction_reduction_by_11_l2013_201326

theorem fraction_reduction_by_11 (k : ℕ) :
  (k^2 - 5 * k + 8) % 11 = 0 → 
  (k^2 + 6 * k + 19) % 11 = 0 :=
by
  sorry

end fraction_reduction_by_11_l2013_201326


namespace min_total_number_of_stamps_l2013_201354

theorem min_total_number_of_stamps
  (r s t : ℕ)
  (h1 : 1 ≤ r)
  (h2 : 1 ≤ s)
  (h3 : 85 * r + 66 * s = 100 * t) :
  r + s = 7 := 
sorry

end min_total_number_of_stamps_l2013_201354


namespace reduced_price_l2013_201313

noncomputable def reduced_price_per_dozen (P : ℝ) : ℝ := 12 * (P / 2)

theorem reduced_price (X P : ℝ) (h1 : X * P = 50) (h2 : (X + 50) * (P / 2) = 50) : reduced_price_per_dozen P = 6 :=
sorry

end reduced_price_l2013_201313


namespace value_of_a_l2013_201315

theorem value_of_a (a : ℝ) (A : Set ℝ) (B : Set ℝ) 
  (hA : A = {-1, 1, 2}) 
  (hB : B = {a + 1, a ^ 2 + 3}) 
  (h_inter : A ∩ B = {2}) : 
  a = 1 := 
by sorry

end value_of_a_l2013_201315


namespace ratio_of_sides_l2013_201365

theorem ratio_of_sides (a b c d : ℝ) 
  (h1 : a / c = 4 / 5) 
  (h2 : b / d = 4 / 5) : b / d = 4 / 5 :=
sorry

end ratio_of_sides_l2013_201365


namespace min_a_for_inequality_l2013_201357

theorem min_a_for_inequality :
  (∀ (x : ℝ), |x + a| - |x + 1| ≤ 2 * a) → a ≥ 1/3 :=
sorry

end min_a_for_inequality_l2013_201357


namespace books_read_in_eight_hours_l2013_201310

-- Definitions to set up the problem
def reading_speed : ℕ := 120
def book_length : ℕ := 360
def available_time : ℕ := 8

-- Theorem statement
theorem books_read_in_eight_hours : (available_time * reading_speed) / book_length = 2 := 
by
  sorry

end books_read_in_eight_hours_l2013_201310


namespace quotient_is_zero_l2013_201303

def square_mod_16 (n : ℕ) : ℕ :=
  (n * n) % 16

def distinct_remainders_in_range : List ℕ :=
  List.eraseDup $
    List.map square_mod_16 (List.range' 1 15)

def sum_of_distinct_remainders : ℕ :=
  distinct_remainders_in_range.sum

theorem quotient_is_zero :
  (sum_of_distinct_remainders / 16) = 0 :=
by
  sorry

end quotient_is_zero_l2013_201303


namespace student_A_more_stable_than_B_l2013_201379

theorem student_A_more_stable_than_B 
    (avg_A : ℝ := 98) (avg_B : ℝ := 98) 
    (var_A : ℝ := 0.2) (var_B : ℝ := 0.8) : 
    var_A < var_B :=
by sorry

end student_A_more_stable_than_B_l2013_201379


namespace perfect_square_expression_l2013_201324

theorem perfect_square_expression (x y : ℝ) (k : ℝ) :
  (∃ f : ℝ → ℝ, ∀ x y, f x = f y → 4 * x^2 - (k - 1) * x * y + 9 * y^2 = (f x) ^ 2) ↔ (k = 13 ∨ k = -11) :=
by
  sorry

end perfect_square_expression_l2013_201324


namespace total_profit_l2013_201397

-- Define the relevant variables and conditions
variables (x y : ℝ) -- Cost prices of the two music players

-- Given conditions
axiom cost_price_first : x * 1.2 = 132
axiom cost_price_second : y * 1.1 = 132

theorem total_profit : 132 + 132 - y - x = 34 :=
by
  -- The proof body is not required
  sorry

end total_profit_l2013_201397


namespace molecular_weight_of_compound_is_correct_l2013_201306

noncomputable def molecular_weight (nC nH nN nO : ℕ) (wC wH wN wO : ℝ) :=
  nC * wC + nH * wH + nN * wN + nO * wO

theorem molecular_weight_of_compound_is_correct :
  molecular_weight 8 18 2 4 12.01 1.008 14.01 16.00 = 206.244 :=
by
  sorry

end molecular_weight_of_compound_is_correct_l2013_201306


namespace possible_age_of_youngest_child_l2013_201374

noncomputable def valid_youngest_age (father_fee : ℝ) (child_fee_per_year : ℝ) (total_bill : ℝ) (triplet_age : ℝ) : ℝ :=
  total_bill - father_fee -  (3 * triplet_age * child_fee_per_year)

theorem possible_age_of_youngest_child (father_fee : ℝ) (child_fee_per_year : ℝ) (total_bill : ℝ) (t y : ℝ)
  (h1 : father_fee = 16)
  (h2 : child_fee_per_year = 0.8)
  (h3 : total_bill = 43.2)
  (age_condition : y = (total_bill - father_fee) / child_fee_per_year - 3 * t) :
  y = 1 ∨ y = 4 :=
by
  sorry

end possible_age_of_youngest_child_l2013_201374


namespace movie_time_difference_l2013_201375

theorem movie_time_difference
  (Nikki_movie : ℝ)
  (Michael_movie : ℝ)
  (Ryn_movie : ℝ)
  (Joyce_movie : ℝ)
  (total_hours : ℝ)
  (h1 : Nikki_movie = 30)
  (h2 : Michael_movie = Nikki_movie / 3)
  (h3 : Ryn_movie = (4 / 5) * Nikki_movie)
  (h4 : total_hours = 76)
  (h5 : total_hours = Michael_movie + Nikki_movie + Ryn_movie + Joyce_movie) :
  Joyce_movie - Michael_movie = 2 := 
by {
  sorry
}

end movie_time_difference_l2013_201375


namespace probability_sum_3_correct_l2013_201395

noncomputable def probability_of_sum_3 : ℚ := 2 / 36

theorem probability_sum_3_correct :
  probability_of_sum_3 = 1 / 18 :=
by
  sorry

end probability_sum_3_correct_l2013_201395


namespace negation_exists_geq_l2013_201383

theorem negation_exists_geq :
  ¬ (∀ x : ℝ, x^3 - x^2 + 1 < 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 ≥ 0 :=
by
  sorry

end negation_exists_geq_l2013_201383


namespace bread_needed_for_sandwiches_l2013_201323

def students_per_group := 5
def groups := 5
def sandwiches_per_student := 2
def pieces_of_bread_per_sandwich := 2

theorem bread_needed_for_sandwiches : 
  students_per_group * groups * sandwiches_per_student * pieces_of_bread_per_sandwich = 100 := 
by
  sorry

end bread_needed_for_sandwiches_l2013_201323


namespace find_point_Q_l2013_201318

theorem find_point_Q {a b c : ℝ} 
  (h1 : ∀ x y z : ℝ, (x + 1)^2 + (y - 3)^2 + (z + 2)^2 = (x - a)^2 + (y - b)^2 + (z - c)^2) 
  (h2 : ∀ x y z: ℝ, 8 * x - 6 * y + 12 * z = 34) : 
  (a = 3) ∧ (b = -6) ∧ (c = 8) :=
by
  sorry

end find_point_Q_l2013_201318


namespace impossible_to_obtain_one_l2013_201346

theorem impossible_to_obtain_one (N : ℕ) (h : N % 3 = 0) : ¬(∃ k : ℕ, (∀ m : ℕ, (∃ q : ℕ, (N + 3 * m = 5 * q) ∧ (q = 1 → m + 1 ≤ k)))) :=
sorry

end impossible_to_obtain_one_l2013_201346


namespace simple_interest_calculation_l2013_201399

variable (P : ℝ) (R : ℝ) (T : ℝ)

def simple_interest (P R T : ℝ) : ℝ := P * R * T

theorem simple_interest_calculation (hP : P = 10000) (hR : R = 0.09) (hT : T = 1) :
    simple_interest P R T = 900 := by
  rw [hP, hR, hT]
  sorry

end simple_interest_calculation_l2013_201399


namespace mark_garden_total_flowers_l2013_201386

theorem mark_garden_total_flowers :
  let yellow := 10
  let purple := yellow + (80 / 100) * yellow
  let total_yellow_purple := yellow + purple
  let green := (25 / 100) * total_yellow_purple
  total_yellow_purple + green = 35 :=
by
  let yellow := 10
  let purple := yellow + (80 / 100) * yellow
  let total_yellow_purple := yellow + purple
  let green := (25 / 100) * total_yellow_purple
  simp [yellow, purple, total_yellow_purple, green]
  sorry

end mark_garden_total_flowers_l2013_201386


namespace smaller_number_l2013_201372

theorem smaller_number (x y : ℤ) (h1 : x + y = 22) (h2 : x - y = 16) : y = 3 :=
by
  sorry

end smaller_number_l2013_201372


namespace number_of_boxes_l2013_201359

-- Define the conditions
def apples_per_crate : ℕ := 180
def number_of_crates : ℕ := 12
def rotten_apples : ℕ := 160
def apples_per_box : ℕ := 20

-- Define the statement to prove
theorem number_of_boxes : (apples_per_crate * number_of_crates - rotten_apples) / apples_per_box = 100 := 
by 
  sorry -- Proof skipped

end number_of_boxes_l2013_201359


namespace distance_between_parallel_sides_l2013_201369

theorem distance_between_parallel_sides (a b : ℝ) (h : ℝ) (A : ℝ) :
  a = 20 → b = 10 → A = 150 → (A = 1 / 2 * (a + b) * h) → h = 10 :=
by
  intros h₀ h₁ h₂ h₃
  sorry

end distance_between_parallel_sides_l2013_201369


namespace apples_in_second_group_l2013_201376

theorem apples_in_second_group : 
  ∀ (A O : ℝ) (x : ℕ), 
  6 * A + 3 * O = 1.77 ∧ x * A + 5 * O = 1.27 ∧ A = 0.21 → 
  x = 2 :=
by
  intros A O x h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end apples_in_second_group_l2013_201376


namespace product_of_p_and_q_l2013_201330

theorem product_of_p_and_q (p q : ℝ) (hpq_sum : p + q = 10) (hpq_cube_sum : p^3 + q^3 = 370) : p * q = 21 :=
by
  sorry

end product_of_p_and_q_l2013_201330


namespace total_oranges_l2013_201322

theorem total_oranges (a b c : ℕ) 
  (h₁ : a = 22) 
  (h₂ : b = a + 17) 
  (h₃ : c = b - 11) : 
  a + b + c = 89 := 
by
  sorry

end total_oranges_l2013_201322


namespace part_a_l2013_201388

theorem part_a (x y : ℝ) : (x + y) * (x^2 - x * y + y^2) = x^3 + y^3 := sorry

end part_a_l2013_201388


namespace value_of_c_l2013_201314

theorem value_of_c (c : ℝ) : (∀ x : ℝ, (-x^2 + c * x + 10 < 0) ↔ (x < 2 ∨ x > 8)) → c = 10 :=
by
  sorry

end value_of_c_l2013_201314


namespace problem1_problem2_l2013_201394

def f (x y : ℝ) : ℝ := x^2 * y

def P0 : ℝ × ℝ := (5, 4)

def Δx : ℝ := 0.1
def Δy : ℝ := -0.2

def Δf (f : ℝ → ℝ → ℝ) (P : ℝ × ℝ) (Δx Δy : ℝ) : ℝ :=
  f (P.1 + Δx) (P.2 + Δy) - f P.1 P.2

def df (f : ℝ → ℝ → ℝ) (P : ℝ × ℝ) (Δx Δy : ℝ) : ℝ :=
  (2 * P.1 * P.2) * Δx + (P.1^2) * Δy

theorem problem1 : Δf f P0 Δx Δy = -1.162 := 
  sorry

theorem problem2 : df f P0 Δx Δy = -1 :=
  sorry

end problem1_problem2_l2013_201394


namespace no_integer_solutions_19x2_minus_76y2_eq_1976_l2013_201328

theorem no_integer_solutions_19x2_minus_76y2_eq_1976 :
  ∀ x y : ℤ, 19 * x^2 - 76 * y^2 ≠ 1976 :=
by sorry

end no_integer_solutions_19x2_minus_76y2_eq_1976_l2013_201328


namespace black_grid_after_rotation_l2013_201361
open ProbabilityTheory

noncomputable def probability_black_grid_after_rotation : ℚ := 6561 / 65536

theorem black_grid_after_rotation (p : ℚ) (h : p = 1 / 2) :
  probability_black_grid_after_rotation = (3 / 4) ^ 8 := 
sorry

end black_grid_after_rotation_l2013_201361


namespace Hannah_cut_strands_l2013_201377

variable (H : ℕ)

theorem Hannah_cut_strands (h : 2 * (H + 3) = 22) : H = 8 :=
by
  sorry

end Hannah_cut_strands_l2013_201377


namespace units_digit_17_pow_17_l2013_201304

theorem units_digit_17_pow_17 : (17^17 % 10) = 7 := by
  sorry

end units_digit_17_pow_17_l2013_201304


namespace boat_stream_ratio_l2013_201385

theorem boat_stream_ratio (B S : ℝ) (h : 2 * (B - S) = B + S) : B / S = 3 :=
by
  sorry

end boat_stream_ratio_l2013_201385


namespace distance_between_A_and_B_l2013_201378

theorem distance_between_A_and_B 
  (v_pas0 v_freight0 : ℝ) -- original speeds of passenger and freight train
  (t_freight : ℝ) -- time taken by freight train
  (d : ℝ) -- distance sought
  (h1 : t_freight = d / v_freight0) 
  (h2 : d + 288 = v_pas0 * t_freight) 
  (h3 : (d / (v_freight0 + 10)) + 2.4 = d / (v_pas0 + 10))
  : d = 360 := 
sorry

end distance_between_A_and_B_l2013_201378


namespace initial_candies_count_l2013_201332

-- Definitions based on conditions
def NelliesCandies : Nat := 12
def JacobsCandies : Nat := NelliesCandies / 2
def LanasCandies : Nat := JacobsCandies - 3
def TotalCandiesEaten : Nat := NelliesCandies + JacobsCandies + LanasCandies
def RemainingCandies : Nat := 3 * 3
def InitialCandies := TotalCandiesEaten + RemainingCandies

-- Theorem stating the initial candies count
theorem initial_candies_count : InitialCandies = 30 := by 
  sorry

end initial_candies_count_l2013_201332


namespace count_mod_6_mod_11_lt_1000_l2013_201344

theorem count_mod_6_mod_11_lt_1000 : ∃ n : ℕ, (∀ x : ℕ, (x < n + 1) ∧ ((6 + 11 * x) < 1000) ∧ (6 + 11 * x) % 11 = 6) ∧ (n + 1 = 91) :=
by
  sorry

end count_mod_6_mod_11_lt_1000_l2013_201344


namespace rational_sum_of_cubic_roots_inverse_l2013_201305

theorem rational_sum_of_cubic_roots_inverse 
  (p q r : ℚ) 
  (h1 : p ≠ 0) 
  (h2 : q ≠ 0) 
  (h3 : r ≠ 0) 
  (h4 : ∃ a b c : ℚ, a = (pq^2)^(1/3) ∧ b = (qr^2)^(1/3) ∧ c = (rp^2)^(1/3) ∧ a + b + c ≠ 0) 
  : ∃ s : ℚ, s = 1/((pq^2)^(1/3)) + 1/((qr^2)^(1/3)) + 1/((rp^2)^(1/3)) :=
sorry

end rational_sum_of_cubic_roots_inverse_l2013_201305


namespace jackson_pbj_sandwiches_l2013_201358

-- The number of Wednesdays and Fridays in the 36-week school year
def total_weeks : ℕ := 36
def total_wednesdays : ℕ := total_weeks
def total_fridays : ℕ := total_weeks

-- Public holidays on Wednesdays and Fridays
def holidays_wednesdays : ℕ := 2
def holidays_fridays : ℕ := 3

-- Days Jackson missed
def missed_wednesdays : ℕ := 1
def missed_fridays : ℕ := 2

-- Number of times Jackson asks for a ham and cheese sandwich every 4 weeks
def weeks_for_ham_and_cheese : ℕ := total_weeks / 4

-- Number of ham and cheese sandwich days
def ham_and_cheese_wednesdays : ℕ := weeks_for_ham_and_cheese
def ham_and_cheese_fridays : ℕ := weeks_for_ham_and_cheese * 2

-- Remaining days for peanut butter and jelly sandwiches
def remaining_wednesdays : ℕ := total_wednesdays - holidays_wednesdays - missed_wednesdays
def remaining_fridays : ℕ := total_fridays - holidays_fridays - missed_fridays

def pbj_wednesdays : ℕ := remaining_wednesdays - ham_and_cheese_wednesdays
def pbj_fridays : ℕ := remaining_fridays - ham_and_cheese_fridays

-- Total peanut butter and jelly sandwiches
def total_pbj : ℕ := pbj_wednesdays + pbj_fridays

theorem jackson_pbj_sandwiches : total_pbj = 37 := by
  -- We don't require the proof steps, just the statement
  sorry

end jackson_pbj_sandwiches_l2013_201358


namespace part_a_part_b_l2013_201336

def bright (n : ℕ) := ∃ a b : ℕ, n = a^2 + b^3

theorem part_a (r s : ℕ) (h₀ : r > 0) (h₁ : s > 0) : 
  ∃ᶠ n in at_top, bright (r + n) ∧ bright (s + n) := 
by sorry

theorem part_b (r s : ℕ) (h₀ : r > 0) (h₁ : s > 0) : 
  ∃ᶠ m in at_top, bright (r * m) ∧ bright (s * m) := 
by sorry

end part_a_part_b_l2013_201336


namespace jared_march_texts_l2013_201381

def T (n : ℕ) : ℕ := ((n ^ 2) + 1) * (n.factorial)

theorem jared_march_texts : T 5 = 3120 := by
  -- The details of the proof would go here, but we use sorry to skip it
  sorry

end jared_march_texts_l2013_201381


namespace complementary_angle_difference_l2013_201367

def is_complementary (a b : ℝ) : Prop := a + b = 90

def in_ratio (a b : ℝ) (m n : ℝ) : Prop := a / b = m / n

theorem complementary_angle_difference (a b : ℝ) (h1 : is_complementary a b) (h2 : in_ratio a b 5 1) : abs (a - b) = 60 := 
by
  sorry

end complementary_angle_difference_l2013_201367


namespace triangle_prime_sides_l2013_201366

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem triangle_prime_sides :
  ∃ (a b c : ℕ), a ≤ b ∧ b ≤ c ∧ is_prime a ∧ is_prime b ∧ is_prime c ∧ 
  a + b + c = 25 ∧
  (a = b ∨ b = c ∨ a = c) ∧
  (∀ (x y z : ℕ), x ≤ y ∧ y ≤ z ∧ is_prime x ∧ is_prime y ∧ is_prime z ∧ x + y + z = 25 → (x, y, z) = (3, 11, 11) ∨ (x, y, z) = (7, 7, 11)) :=
by
  sorry

end triangle_prime_sides_l2013_201366


namespace num_integer_solutions_abs_eq_3_l2013_201353

theorem num_integer_solutions_abs_eq_3 :
  (∀ (x y : ℤ), (|x| + |y| = 3) → 
  ∃ (s : Finset (ℤ × ℤ)), s.card = 12 ∧ (∀ (a b : ℤ), (a, b) ∈ s ↔ (|a| + |b| = 3))) :=
by
  sorry

end num_integer_solutions_abs_eq_3_l2013_201353


namespace cos_value_l2013_201312

theorem cos_value {α : ℝ} (h : Real.sin (π / 6 + α) = 1 / 3) : Real.cos (π / 3 - α) = 1 / 3 := 
by sorry

end cos_value_l2013_201312


namespace ten_faucets_fill_50_gallon_in_60_seconds_l2013_201389

-- Define the conditions
def five_faucets_fill_tub (faucet_rate : ℝ) : Prop :=
  5 * faucet_rate * 8 = 200

def all_faucets_same_rate (tub_capacity time : ℝ) (num_faucets : ℕ) (faucet_rate : ℝ) : Prop :=
  num_faucets * faucet_rate * time = tub_capacity

-- Define the main theorem to be proven
theorem ten_faucets_fill_50_gallon_in_60_seconds (faucet_rate : ℝ) :
  (∃ faucet_rate, five_faucets_fill_tub faucet_rate) →
  all_faucets_same_rate 50 1 10 faucet_rate →
  10 * faucet_rate * (1 / 60) = 50 :=
by
  sorry

end ten_faucets_fill_50_gallon_in_60_seconds_l2013_201389


namespace balboa_earnings_correct_l2013_201349

def students_from_allen_days : Nat := 7 * 3
def students_from_balboa_days : Nat := 4 * 5
def students_from_carver_days : Nat := 5 * 9
def total_student_days : Nat := students_from_allen_days + students_from_balboa_days + students_from_carver_days
def total_payment : Nat := 744
def daily_wage : Nat := total_payment / total_student_days
def balboa_earnings : Nat := daily_wage * students_from_balboa_days

theorem balboa_earnings_correct : balboa_earnings = 180 := by
  sorry

end balboa_earnings_correct_l2013_201349


namespace chef_meals_prepared_l2013_201360

theorem chef_meals_prepared (S D_added D_total L R : ℕ)
  (hS : S = 12)
  (hD_added : D_added = 5)
  (hD_total : D_total = 10)
  (hR : R + D_added = D_total)
  (hL : L = S + R) : L = 17 :=
by
  sorry

end chef_meals_prepared_l2013_201360


namespace ratio_of_a_plus_b_to_b_plus_c_l2013_201307

variable (a b c : ℝ)

theorem ratio_of_a_plus_b_to_b_plus_c (h1 : b / a = 3) (h2 : c / b = 4) : (a + b) / (b + c) = 4 / 15 :=
by
  sorry

end ratio_of_a_plus_b_to_b_plus_c_l2013_201307


namespace find_k_l2013_201335

theorem find_k (k : ℝ) : 
  (∀ x : ℝ, -4 < x ∧ x < 3 → k * (x^2 + 6 * x - k) * (x^2 + x - 12) > 0) ↔ (k ≤ -9) :=
by sorry

end find_k_l2013_201335


namespace find_y_intercept_l2013_201319

theorem find_y_intercept (m b x y : ℝ) (h1 : m = 2) (h2 : (x, y) = (239, 480)) (line_eq : y = m * x + b) : b = 2 :=
by
  sorry

end find_y_intercept_l2013_201319


namespace i_pow_2016_eq_one_l2013_201308
open Complex

theorem i_pow_2016_eq_one : (Complex.I ^ 2016) = 1 := by
  have h : Complex.I ^ 4 = 1 :=
    by rw [Complex.I_pow_four]
  exact sorry

end i_pow_2016_eq_one_l2013_201308


namespace fraction_value_l2013_201355

variable {x y : ℝ}

theorem fraction_value (hx : x ≠ 0) (hy : y ≠ 0) (h : (2 * x - 3 * y) / (x + 2 * y) = 3) :
  (x - 2 * y) / (2 * x + 3 * y) = 11 / 15 :=
  sorry

end fraction_value_l2013_201355


namespace sin_double_angle_15_eq_half_l2013_201338

theorem sin_double_angle_15_eq_half : 2 * Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) = 1 / 2 := 
sorry

end sin_double_angle_15_eq_half_l2013_201338


namespace students_in_class_l2013_201398

theorem students_in_class (S : ℕ) 
  (h1 : (1 / 4) * (9 / 10 : ℚ) * S = 9) : S = 40 :=
sorry

end students_in_class_l2013_201398


namespace circle_intersection_probability_l2013_201317

noncomputable def probability_circles_intersect : ℝ :=
  1

theorem circle_intersection_probability :
  ∀ (A_X B_X : ℝ), (0 ≤ A_X) → (A_X ≤ 2) → (0 ≤ B_X) → (B_X ≤ 2) →
  (∃ y, y ≥ 1 ∧ y ≤ 2) →
  ∃ p : ℝ, p = probability_circles_intersect ∧
  p = 1 :=
by
  sorry

end circle_intersection_probability_l2013_201317


namespace second_consecutive_odd_integer_l2013_201350

theorem second_consecutive_odd_integer (n : ℤ) : 
  (n - 2) + (n + 2) = 152 → n = 76 := 
by 
  sorry

end second_consecutive_odd_integer_l2013_201350


namespace min_value_of_t_l2013_201364

theorem min_value_of_t (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) : 
  ∃ t : ℝ, t = 3 + 2 * Real.sqrt 2 ∧ t = 1 / a + 1 / b :=
sorry

end min_value_of_t_l2013_201364


namespace inclination_angle_of_vertical_line_l2013_201342

theorem inclination_angle_of_vertical_line :
  ∀ x : ℝ, x = Real.tan (60 * Real.pi / 180) → ∃ θ : ℝ, θ = 90 := by
  sorry

end inclination_angle_of_vertical_line_l2013_201342


namespace emily_total_beads_l2013_201333

theorem emily_total_beads (necklaces : ℕ) (beads_per_necklace : ℕ) (total_beads : ℕ) : 
  necklaces = 11 → 
  beads_per_necklace = 28 → 
  total_beads = necklaces * beads_per_necklace → 
  total_beads = 308 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  assumption

end emily_total_beads_l2013_201333


namespace shaded_area_is_correct_l2013_201348

noncomputable def octagon_side_length := 3
noncomputable def octagon_area := 2 * (1 + Real.sqrt 2) * octagon_side_length^2
noncomputable def semicircle_radius := octagon_side_length / 2
noncomputable def semicircle_area := (1 / 2) * Real.pi * semicircle_radius^2
noncomputable def total_semicircle_area := 8 * semicircle_area
noncomputable def shaded_region_area := octagon_area - total_semicircle_area

theorem shaded_area_is_correct : shaded_region_area = 54 + 36 * Real.sqrt 2 - 9 * Real.pi :=
by
  -- Proof goes here, but we're inserting sorry to skip it
  sorry

end shaded_area_is_correct_l2013_201348


namespace problem1_problem2_l2013_201351

-- Proof problem for the first condition
theorem problem1 {p : ℕ} (hp : Nat.Prime p) 
  (h : ∃ n : ℕ, (7^(p-1) - 1) = p * n^2) : p = 3 :=
sorry

-- Proof problem for the second condition
theorem problem2 {p : ℕ} (hp : Nat.Prime p)
  (h : ∃ n : ℕ, (11^(p-1) - 1) = p * n^2) : false :=
sorry

end problem1_problem2_l2013_201351


namespace no_triples_exist_l2013_201343

theorem no_triples_exist (m p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hm : m > 0) :
  2^m * p^2 + 1 ≠ q^7 :=
sorry

end no_triples_exist_l2013_201343


namespace solve_keychain_problem_l2013_201341

def keychain_problem : Prop :=
  let f_class := 6
  let f_club := f_class / 2
  let thread_total := 108
  let total_friends := f_class + f_club
  let threads_per_keychain := thread_total / total_friends
  threads_per_keychain = 12

theorem solve_keychain_problem : keychain_problem :=
  by sorry

end solve_keychain_problem_l2013_201341


namespace roots_not_integers_l2013_201340

theorem roots_not_integers (a b c : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : c % 2 = 1) :
    ¬ ∃ x₁ x₂ : ℤ, a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 :=
by
  sorry

end roots_not_integers_l2013_201340


namespace team_total_mistakes_l2013_201371

theorem team_total_mistakes (total_questions : ℕ) (riley_mistakes : ℕ) (ofelia_correction: (ℕ → ℕ) ) : total_questions = 35 → riley_mistakes = 3 → (∀ riley_correct_answers, riley_correct_answers = total_questions - riley_mistakes → ofelia_correction riley_correct_answers = (riley_correct_answers / 2) + 5) → (riley_mistakes + (total_questions - (ofelia_correction (total_questions - riley_mistakes)))) = 17 :=
by
  intros h1 h2 h3
  sorry

end team_total_mistakes_l2013_201371


namespace cousin_reading_time_l2013_201325

theorem cousin_reading_time (my_time_hours : ℕ) (speed_ratio : ℕ) (my_time_minutes := my_time_hours * 60) :
  (my_time_hours = 3) ∧ (speed_ratio = 5) → 
  (my_time_minutes / speed_ratio = 36) :=
by
  sorry

end cousin_reading_time_l2013_201325


namespace lizzie_scored_six_l2013_201391

-- Definitions based on the problem conditions
def lizzie_score : Nat := sorry
def nathalie_score := lizzie_score + 3
def aimee_score := 2 * (lizzie_score + nathalie_score)

-- Total score condition
def total_score := 50
def teammates_score := 17
def combined_score := total_score - teammates_score

-- Proven statement
theorem lizzie_scored_six:
  (lizzie_score + nathalie_score + aimee_score = combined_score) → lizzie_score = 6 :=
by sorry

end lizzie_scored_six_l2013_201391
