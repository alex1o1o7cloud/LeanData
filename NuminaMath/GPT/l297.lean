import Mathlib

namespace sample_capacity_n_l297_29715

theorem sample_capacity_n
  (n : ℕ) 
  (engineers technicians craftsmen : ℕ) 
  (total_population : ℕ)
  (stratified_interval systematic_interval : ℕ) :
  engineers = 6 →
  technicians = 12 →
  craftsmen = 18 →
  total_population = engineers + technicians + craftsmen →
  total_population = 36 →
  (∃ n : ℕ, n ∣ total_population ∧ 6 ∣ n ∧ 35 % (n + 1) = 0) →
  n = 6 :=
by
  sorry

end sample_capacity_n_l297_29715


namespace coin_flip_prob_nickel_halfdollar_heads_l297_29766

def coin_prob : ℚ :=
  let total_outcomes := 2^5
  let successful_outcomes := 2^3
  successful_outcomes / total_outcomes

theorem coin_flip_prob_nickel_halfdollar_heads :
  coin_prob = 1 / 4 :=
by
  sorry

end coin_flip_prob_nickel_halfdollar_heads_l297_29766


namespace ram_money_l297_29760

theorem ram_money (R G K : ℝ) (h1 : R / G = 7 / 17) (h2 : G / K = 7 / 17) (h3 : K = 3468) :
  R = 588 := by
  sorry

end ram_money_l297_29760


namespace increased_speed_l297_29799

theorem increased_speed (S : ℝ) : 
  (∀ (usual_speed : ℝ) (usual_time : ℝ) (distance : ℝ), 
    usual_speed = 20 ∧ distance = 100 ∧ usual_speed * usual_time = distance ∧ S * (usual_time - 1) = distance) → 
  S = 25 :=
by
  intros h1
  sorry

end increased_speed_l297_29799


namespace first_term_value_l297_29763

noncomputable def find_first_term (a r : ℝ) := a / (1 - r) = 27 ∧ a^2 / (1 - r^2) = 108

theorem first_term_value :
  ∃ (a r : ℝ), find_first_term a r ∧ a = 216 / 31 :=
by
  sorry

end first_term_value_l297_29763


namespace even_odd_difference_l297_29721

def even_sum_n (n : ℕ) : ℕ := (n * (n + 1))
def odd_sum_n (n : ℕ) : ℕ := n * n

theorem even_odd_difference : even_sum_n 100 - odd_sum_n 100 = 100 := by
  -- The proof goes here
  sorry

end even_odd_difference_l297_29721


namespace n_pow_8_minus_1_divisible_by_480_l297_29795

theorem n_pow_8_minus_1_divisible_by_480 (n : ℤ) (h1 : ¬ (2 ∣ n)) (h2 : ¬ (3 ∣ n)) (h3 : ¬ (5 ∣ n)) : 
  480 ∣ (n^8 - 1) := 
sorry

end n_pow_8_minus_1_divisible_by_480_l297_29795


namespace doctor_visit_cost_l297_29788

theorem doctor_visit_cost (cast_cost : ℝ) (insurance_coverage : ℝ) (out_of_pocket : ℝ) (visit_cost : ℝ) :
  cast_cost = 200 → insurance_coverage = 0.60 → out_of_pocket = 200 → 0.40 * (visit_cost + cast_cost) = out_of_pocket → visit_cost = 300 :=
by
  intros h_cast h_insurance h_out_of_pocket h_equation
  sorry

end doctor_visit_cost_l297_29788


namespace totalPeaches_l297_29753

-- Define the number of red, yellow, and green peaches
def redPeaches := 7
def yellowPeaches := 15
def greenPeaches := 8

-- Define the total number of peaches and the proof statement
theorem totalPeaches : redPeaches + yellowPeaches + greenPeaches = 30 := by
  sorry

end totalPeaches_l297_29753


namespace cos_sub_eq_five_over_eight_l297_29751

theorem cos_sub_eq_five_over_eight (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3 / 2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5 / 8 := 
by sorry

end cos_sub_eq_five_over_eight_l297_29751


namespace units_digit_fraction_l297_29706

theorem units_digit_fraction (h1 : 30 = 2 * 3 * 5) (h2 : 31 = 31) (h3 : 32 = 2^5) 
    (h4 : 33 = 3 * 11) (h5 : 34 = 2 * 17) (h6 : 35 = 5 * 7) (h7 : 7200 = 2^4 * 3^2 * 5^2) :
    ((30 * 31 * 32 * 33 * 34 * 35) / 7200) % 10 = 2 :=
by
  sorry

end units_digit_fraction_l297_29706


namespace number_of_students_l297_29762

theorem number_of_students (total_stars : ℕ) (stars_per_student : ℕ) (h1 : total_stars = 372) (h2 : stars_per_student = 3) : total_stars / stars_per_student = 124 :=
by
  sorry

end number_of_students_l297_29762


namespace part1_part2_l297_29730

noncomputable def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + (a^2 - 5) = 0}

theorem part1 (a : ℝ) (h : A ∩ B a = {2}) : a = -1 ∨ a = -3 := by
  sorry

theorem part2 (a : ℝ) (h : A ∪ B a = A) : a ≤ -3 := by
  sorry

end part1_part2_l297_29730


namespace exponentiation_identity_l297_29785

theorem exponentiation_identity (x : ℝ) : (-x^7)^4 = x^28 := 
sorry

end exponentiation_identity_l297_29785


namespace set_A_correct_l297_29792

-- Definition of the sets and conditions
def A : Set ℤ := {-3, 0, 2, 6}
def B : Set ℤ := {-1, 3, 5, 8}

theorem set_A_correct : 
  (∃ a1 a2 a3 a4 : ℤ, A = {a1, a2, a3, a4} ∧ 
  {a1 + a2 + a3, a1 + a2 + a4, a1 + a3 + a4, a2 + a3 + a4} = B) → 
  A = {-3, 0, 2, 6} :=
by 
  sorry

end set_A_correct_l297_29792


namespace bn_six_eight_product_l297_29717

noncomputable def sequence_an (n : ℕ) : ℝ := sorry  -- given that an is an arithmetic sequence and an ≠ 0
noncomputable def sequence_bn (n : ℕ) : ℝ := sorry  -- given that bn is a geometric sequence

theorem bn_six_eight_product :
  (∀ n : ℕ, sequence_an n ≠ 0) →
  2 * sequence_an 3 - sequence_an 7 ^ 2 + 2 * sequence_an 11 = 0 →
  sequence_bn 7 = sequence_an 7 →
  sequence_bn 6 * sequence_bn 8 = 16 :=
sorry

end bn_six_eight_product_l297_29717


namespace real_and_equal_roots_condition_l297_29710

theorem real_and_equal_roots_condition (k : ℝ) : 
  ∀ k : ℝ, (∃ (x : ℝ), 3 * x^2 + 6 * k * x + 9 = 0) ↔ (k = Real.sqrt 3 ∨ k = -Real.sqrt 3) :=
by
  sorry

end real_and_equal_roots_condition_l297_29710


namespace equations_not_equivalent_l297_29722

theorem equations_not_equivalent :
  ∀ x : ℝ, (x + 7 + 10 / (2 * x - 1) = 8 - x + 10 / (2 * x - 1)) ↔ false :=
by
  intro x
  sorry

end equations_not_equivalent_l297_29722


namespace strips_overlap_area_l297_29716

theorem strips_overlap_area :
  ∀ (length_left length_right area_only_left area_only_right : ℕ) (S : ℚ),
    length_left = 9 →
    length_right = 7 →
    area_only_left = 27 →
    area_only_right = 18 →
    (area_only_left + S) / (area_only_right + S) = 9 / 7 →
    S = 13.5 :=
by
  intros length_left length_right area_only_left area_only_right S
  intro h1 h2 h3 h4 h5
  sorry

end strips_overlap_area_l297_29716


namespace could_be_simple_random_sampling_l297_29724

-- Conditions
def boys : Nat := 20
def girls : Nat := 30
def total_students : Nat := boys + girls
def sample_size : Nat := 10
def boys_in_sample : Nat := 4
def girls_in_sample : Nat := 6

-- Theorem Statement
theorem could_be_simple_random_sampling :
  boys = 20 ∧ girls = 30 ∧ sample_size = 10 ∧ boys_in_sample = 4 ∧ girls_in_sample = 6 →
  (∃ (sample_method : String), sample_method = "simple random sampling"):=
by 
  sorry

end could_be_simple_random_sampling_l297_29724


namespace calculate_error_percentage_l297_29707

theorem calculate_error_percentage (x : ℝ) (hx : x > 0) (x_eq_9 : x = 9) :
  (abs ((x * (x - 8)) / (8 * x)) * 100) = 12.5 := by
  sorry

end calculate_error_percentage_l297_29707


namespace calculate_expr_eq_two_l297_29702

def calculate_expr : ℕ :=
  3^(0^(2^8)) + (3^0^2)^8

theorem calculate_expr_eq_two : calculate_expr = 2 := 
by
  sorry

end calculate_expr_eq_two_l297_29702


namespace plane_speed_east_l297_29734

def plane_travel_problem (v : ℕ) : Prop :=
  let time : ℕ := 35 / 10 
  let distance_east := v * time
  let distance_west := 275 * time
  let total_distance := distance_east + distance_west
  total_distance = 2100

theorem plane_speed_east : ∃ v : ℕ, plane_travel_problem v ∧ v = 325 :=
sorry

end plane_speed_east_l297_29734


namespace tank_ratio_l297_29796

variable (C D : ℝ)
axiom h1 : 3 / 4 * C = 2 / 5 * D

theorem tank_ratio : C / D = 8 / 15 := by
  sorry

end tank_ratio_l297_29796


namespace root_power_sum_eq_l297_29772

open Real

theorem root_power_sum_eq :
  ∀ {a b c : ℝ},
  (a ≠ b) → (b ≠ c) → (a ≠ c) →
  (a^3 - 3 * a + 1 = 0) → (b^3 - 3 * b + 1 = 0) → (c^3 - 3 * c + 1 = 0) →
  a^8 + b^8 + c^8 = 186 :=
by
  intros a b c h1 h2 h3 ha hb hc
  sorry

end root_power_sum_eq_l297_29772


namespace shorter_side_length_l297_29752

theorem shorter_side_length (L W : ℝ) (h₁ : L * W = 120) (h₂ : 2 * L + 2 * W = 46) : L = 8 ∨ W = 8 := 
by 
  sorry

end shorter_side_length_l297_29752


namespace Christine_distance_went_l297_29729

-- Definitions from conditions
def Speed : ℝ := 20 -- miles per hour
def Time : ℝ := 4  -- hours

-- Statement of the problem
def Distance_went : ℝ := Speed * Time

-- The theorem we need to prove
theorem Christine_distance_went : Distance_went = 80 :=
by
  sorry

end Christine_distance_went_l297_29729


namespace egyptian_fraction_decomposition_l297_29781

theorem egyptian_fraction_decomposition (n : ℕ) (hn : 0 < n) : 
  (2 : ℚ) / (2 * n + 1) = (1 : ℚ) / (n + 1) + (1 : ℚ) / ((n + 1) * (2 * n + 1)) := 
by {
  sorry
}

end egyptian_fraction_decomposition_l297_29781


namespace megan_initial_markers_l297_29711

theorem megan_initial_markers (gave : ℕ) (total : ℕ) (initial : ℕ) 
  (h1 : gave = 109) 
  (h2 : total = 326) 
  (h3 : initial + gave = total) : 
  initial = 217 := 
by 
  sorry

end megan_initial_markers_l297_29711


namespace greatest_integer_solution_l297_29743

theorem greatest_integer_solution (n : ℤ) (h : n^2 - 13 * n + 40 ≤ 0) : n ≤ 8 :=
sorry

end greatest_integer_solution_l297_29743


namespace part1_part2_l297_29741

-- Part 1
theorem part1 (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := sorry

-- Part 2
theorem part2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) : 
  a * b + b * c + c * a ≤ 1 / 3 := sorry

end part1_part2_l297_29741


namespace oranges_for_price_of_apples_l297_29700

-- Given definitions based on the conditions provided
def cost_of_apples_same_as_bananas (a b : ℕ) : Prop := 12 * a = 6 * b
def cost_of_bananas_same_as_cucumbers (b c : ℕ) : Prop := 3 * b = 5 * c
def cost_of_cucumbers_same_as_oranges (c o : ℕ) : Prop := 2 * c = 1 * o

-- The theorem to prove
theorem oranges_for_price_of_apples (a b c o : ℕ) 
  (hab : cost_of_apples_same_as_bananas a b)
  (hbc : cost_of_bananas_same_as_cucumbers b c)
  (hco : cost_of_cucumbers_same_as_oranges c o) : 
  24 * a = 10 * o :=
sorry

end oranges_for_price_of_apples_l297_29700


namespace luther_latest_line_count_l297_29771

theorem luther_latest_line_count :
  let silk := 10
  let cashmere := silk / 2
  let blended := 2
  silk + cashmere + blended = 17 :=
by
  sorry

end luther_latest_line_count_l297_29771


namespace students_in_both_math_and_chem_l297_29727

theorem students_in_both_math_and_chem (students total math physics chem math_physics physics_chem : ℕ) :
  total = 36 →
  students ≤ 2 →
  math = 26 →
  physics = 15 →
  chem = 13 →
  math_physics = 6 →
  physics_chem = 4 →
  math + physics + chem - math_physics - physics_chem - students = total →
  students = 8 := by
  intros h_total h_students h_math h_physics h_chem h_math_physics h_physics_chem h_equation
  sorry

end students_in_both_math_and_chem_l297_29727


namespace general_term_of_sequence_l297_29746

theorem general_term_of_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) 
    (h1 : ∀ n, S n = 2 * a n - 1) 
    (a₁ : a 1 = 1) :
  ∀ n, a n = 2^(n - 1) := 
sorry

end general_term_of_sequence_l297_29746


namespace internet_bill_proof_l297_29798

variable (current_bill : ℕ)
variable (internet_bill_30Mbps : ℕ)
variable (annual_savings : ℕ)
variable (additional_amount_20Mbps : ℕ)

theorem internet_bill_proof
  (h1 : current_bill = 20)
  (h2 : internet_bill_30Mbps = 40)
  (h3 : annual_savings = 120)
  (monthly_savings : ℕ := annual_savings / 12)
  (h4 : monthly_savings = 10)
  (h5 : internet_bill_30Mbps - (current_bill + additional_amount_20Mbps) = 10) :
  additional_amount_20Mbps = 10 :=
by
  sorry

end internet_bill_proof_l297_29798


namespace david_produces_8_more_widgets_l297_29732

variable (w t : ℝ)

def widgets_monday (w t : ℝ) : ℝ :=
  w * t

def widgets_tuesday (w t : ℝ) : ℝ :=
  (w + 4) * (t - 2)

theorem david_produces_8_more_widgets (h : w = 2 * t) : 
  widgets_monday w t - widgets_tuesday w t = 8 :=
by
  sorry

end david_produces_8_more_widgets_l297_29732


namespace seat_number_X_l297_29787

theorem seat_number_X (X : ℕ) (h1 : 42 - 30 = X - 6) : X = 18 :=
by
  sorry

end seat_number_X_l297_29787


namespace sqrt_sin_cos_expression_l297_29704

theorem sqrt_sin_cos_expression (α β : ℝ) : 
  Real.sqrt ((1 - Real.sin α * Real.sin β)^2 - (Real.cos α * Real.cos β)^2) = |Real.sin α - Real.sin β| :=
sorry

end sqrt_sin_cos_expression_l297_29704


namespace janice_time_left_l297_29774

def time_before_movie : ℕ := 2 * 60
def homework_time : ℕ := 30
def cleaning_time : ℕ := homework_time / 2
def walking_dog_time : ℕ := homework_time + 5
def taking_trash_time : ℕ := homework_time * 1 / 6

theorem janice_time_left : time_before_movie - (homework_time + cleaning_time + walking_dog_time + taking_trash_time) = 35 :=
by
  sorry

end janice_time_left_l297_29774


namespace time_to_cross_bridge_l297_29786

def train_length : ℕ := 600  -- train length in meters
def bridge_length : ℕ := 100  -- overbridge length in meters
def speed_km_per_hr : ℕ := 36  -- speed of the train in kilometers per hour

-- Convert speed from km/h to m/s
def speed_m_per_s : ℕ := speed_km_per_hr * 1000 / 3600

-- Compute the total distance
def total_distance : ℕ := train_length + bridge_length

-- Prove the time to cross the overbridge
theorem time_to_cross_bridge : total_distance / speed_m_per_s = 70 := by
  sorry

end time_to_cross_bridge_l297_29786


namespace largest_a_value_l297_29712

theorem largest_a_value (a b c : ℝ) (h1 : a + b + c = 7) (h2 : ab + ac + bc = 12) : 
  a ≤ (7 + Real.sqrt 46) / 3 :=
sorry

end largest_a_value_l297_29712


namespace quadratic_real_roots_condition_l297_29740

theorem quadratic_real_roots_condition (a b c : ℝ) (q : b^2 - 4 * a * c ≥ 0) (h : a ≠ 0) : 
  (b^2 - 4 * a * c ≥ 0 ∧ a ≠ 0) ↔ ((∃ x1 x2 : ℝ, a * x1 ^ 2 + b * x1 + c = 0 ∧ a * x2 ^ 2 + b * x2 + c = 0) ∨ (∃ x : ℝ, a * x ^ 2 + b * x + c = 0)) :=
by
  sorry

end quadratic_real_roots_condition_l297_29740


namespace consecutive_even_product_l297_29728

theorem consecutive_even_product (x : ℤ) (h : x * (x + 2) = 224) : x * (x + 2) = 224 := by
  sorry

end consecutive_even_product_l297_29728


namespace train_passes_tree_in_20_seconds_l297_29773

def train_passing_time 
  (length_of_train : ℕ)
  (speed_kmh : ℕ)
  (conversion_factor : ℚ) : ℚ :=
  length_of_train / (speed_kmh * conversion_factor)

theorem train_passes_tree_in_20_seconds 
  (length_of_train : ℕ := 350)
  (speed_kmh : ℕ := 63)
  (conversion_factor : ℚ := 1000 / 3600) : 
  train_passing_time length_of_train speed_kmh conversion_factor = 20 :=
  sorry

end train_passes_tree_in_20_seconds_l297_29773


namespace bicycles_difference_on_october_1_l297_29738

def initial_inventory : Nat := 200
def february_decrease : Nat := 4
def march_decrease : Nat := 6
def april_decrease : Nat := 8
def may_decrease : Nat := 10
def june_decrease : Nat := 12
def july_decrease : Nat := 14
def august_decrease : Nat := 16 + 20
def september_decrease : Nat := 18
def shipment : Nat := 50

def total_decrease : Nat := february_decrease + march_decrease + april_decrease + may_decrease + june_decrease + july_decrease + august_decrease + september_decrease
def stock_increase : Nat := shipment
def net_decrease : Nat := total_decrease - stock_increase

theorem bicycles_difference_on_october_1 : initial_inventory - net_decrease = 58 := by
  sorry

end bicycles_difference_on_october_1_l297_29738


namespace total_number_of_birds_l297_29770

def geese : ℕ := 58
def ducks : ℕ := 37
def swans : ℕ := 42

theorem total_number_of_birds : geese + ducks + swans = 137 := by
  sorry

end total_number_of_birds_l297_29770


namespace dogs_not_liking_any_l297_29736

variables (totalDogs : ℕ) (dogsLikeWatermelon : ℕ) (dogsLikeSalmon : ℕ) (dogsLikeBothSalmonWatermelon : ℕ)
          (dogsLikeChicken : ℕ) (dogsLikeWatermelonNotSalmon : ℕ) (dogsLikeSalmonChickenNotWatermelon : ℕ)

theorem dogs_not_liking_any : totalDogs = 80 → dogsLikeWatermelon = 21 → dogsLikeSalmon = 58 →
  dogsLikeBothSalmonWatermelon = 12 → dogsLikeChicken = 15 →
  dogsLikeWatermelonNotSalmon = 7 → dogsLikeSalmonChickenNotWatermelon = 10 →
  (totalDogs - ((dogsLikeSalmon - (dogsLikeBothSalmonWatermelon + dogsLikeSalmonChickenNotWatermelon)) +
                (dogsLikeWatermelon - (dogsLikeBothSalmonWatermelon + dogsLikeWatermelonNotSalmon)) +
                (dogsLikeChicken - (dogsLikeWatermelonNotSalmon + dogsLikeSalmonChickenNotWatermelon)) +
                dogsLikeBothSalmonWatermelon + dogsLikeWatermelonNotSalmon + dogsLikeSalmonChickenNotWatermelon)) = 13 :=
by
  intros h_totalDogs h_dogsLikeWatermelon h_dogsLikeSalmon h_dogsLikeBothSalmonWatermelon 
         h_dogsLikeChicken h_dogsLikeWatermelonNotSalmon h_dogsLikeSalmonChickenNotWatermelon
  sorry

end dogs_not_liking_any_l297_29736


namespace train_speed_is_correct_l297_29718

noncomputable def speed_of_train (distance : ℝ) (time : ℝ) : ℝ :=
  (distance / time) * 3.6

theorem train_speed_is_correct :
  speed_of_train 200 19.99840012798976 = 36.00287976960864 :=
by
  sorry

end train_speed_is_correct_l297_29718


namespace expression_nonnegative_l297_29749

theorem expression_nonnegative (x : ℝ) : 
  0 ≤ x → x < 3 → 0 ≤ (x - 12 * x^2 + 36 * x^3) / (9 - x^3) :=
  sorry

end expression_nonnegative_l297_29749


namespace value_of_m_l297_29778

theorem value_of_m (a b c : ℤ) (m : ℤ) (h1 : 0 ≤ m) (h2 : m ≤ 26) 
  (h3 : (a + b + c) % 27 = m) (h4 : ((a - b) * (b - c) * (c - a)) % 27 = m) : 
  m = 0 :=
  by
  -- Proof is to be filled in
  sorry

end value_of_m_l297_29778


namespace initial_paintings_l297_29768

theorem initial_paintings (paintings_per_day : ℕ) (days : ℕ) (total_paintings : ℕ) (initial_paintings : ℕ) 
  (h1 : paintings_per_day = 2) 
  (h2 : days = 30) 
  (h3 : total_paintings = 80) 
  (h4 : total_paintings = initial_paintings + paintings_per_day * days) : 
  initial_paintings = 20 := by
  sorry

end initial_paintings_l297_29768


namespace function_positivity_range_l297_29748

theorem function_positivity_range (m x : ℝ): 
  (∀ x, (2 * x^2 + (4 - m) * x + 4 - m > 0) ∨ (m * x > 0)) ↔ m < 4 :=
sorry

end function_positivity_range_l297_29748


namespace how_many_unanswered_l297_29720

theorem how_many_unanswered (c w u : ℕ) (h1 : 25 + 5 * c - 2 * w = 95)
                            (h2 : 6 * c + u = 110) (h3 : c + w + u = 30) : u = 10 :=
by
  sorry

end how_many_unanswered_l297_29720


namespace weaving_increase_is_sixteen_over_twentynine_l297_29723

-- Conditions for the problem as definitions
def first_day_weaving := 5
def total_days := 30
def total_weaving := 390

-- The arithmetic series sum formula for 30 days
def sum_arithmetic_series (a d : ℚ) (n : ℕ) := n * a + (n * (n-1) / 2) * d

-- The question is to prove the increase in chi per day is 16/29
theorem weaving_increase_is_sixteen_over_twentynine
  (d : ℚ)
  (h : sum_arithmetic_series first_day_weaving d total_days = total_weaving) :
  d = 16 / 29 :=
sorry

end weaving_increase_is_sixteen_over_twentynine_l297_29723


namespace total_cost_correct_l297_29790

def cost_barette : ℕ := 3
def cost_comb : ℕ := 1

def kristine_barrettes : ℕ := 1
def kristine_combs : ℕ := 1

def crystal_barrettes : ℕ := 3
def crystal_combs : ℕ := 1

def total_spent (cost_barette : ℕ) (cost_comb : ℕ) 
  (kristine_barrettes : ℕ) (kristine_combs : ℕ) 
  (crystal_barrettes : ℕ) (crystal_combs : ℕ) : ℕ :=
  (kristine_barrettes * cost_barette + kristine_combs * cost_comb) + 
  (crystal_barrettes * cost_barette + crystal_combs * cost_comb)

theorem total_cost_correct :
  total_spent cost_barette cost_comb kristine_barrettes kristine_combs crystal_barrettes crystal_combs = 14 :=
by
  sorry

end total_cost_correct_l297_29790


namespace isosceles_triangle_length_l297_29713

theorem isosceles_triangle_length (a : ℝ) (h_graph_A : ∃ y, (a, y) ∈ {p : ℝ × ℝ | p.snd = -p.fst^2})
  (h_graph_B : ∃ y, (-a, y) ∈ {p : ℝ × ℝ | p.snd = -p.fst^2}) 
  (h_isosceles : ∃ O : ℝ × ℝ, O = (0, 0) ∧ 
    dist (a, -a^2) O = dist (-a, -a^2) O ∧ dist (a, -a^2) (-a, -a^2) = dist (-a, -a^2) O) :
  dist (a, -a^2) (0, 0) = 2 * Real.sqrt 3 := sorry

end isosceles_triangle_length_l297_29713


namespace medians_sum_le_circumradius_l297_29709

-- Definition of the problem
variable (a b c R : ℝ) (m_a m_b m_c : ℝ)

-- Conditions: medians of triangle ABC, and R is the circumradius
def is_median (m : ℝ) (a b c : ℝ) : Prop :=
  m^2 = (2*b^2 + 2*c^2 - a^2) / 4

-- Main theorem to prove
theorem medians_sum_le_circumradius (h_ma : is_median m_a a b c)
  (h_mb : is_median m_b b a c) (h_mc : is_median m_c c a b) 
  (h_R : a^2 + b^2 + c^2 ≤ 9 * R^2) :
  m_a + m_b + m_c ≤ 9 / 2 * R :=
sorry

end medians_sum_le_circumradius_l297_29709


namespace Jackie_exercise_hours_l297_29737

variable (work_hours : ℕ) (sleep_hours : ℕ) (free_time_hours : ℕ) (total_hours_in_day : ℕ)
variable (time_for_exercise : ℕ)

noncomputable def prove_hours_exercising (work_hours sleep_hours free_time_hours total_hours_in_day : ℕ) : Prop :=
  work_hours = 8 ∧
  sleep_hours = 8 ∧
  free_time_hours = 5 ∧
  total_hours_in_day = 24 → 
  time_for_exercise = total_hours_in_day - (work_hours + sleep_hours + free_time_hours)

theorem Jackie_exercise_hours :
  prove_hours_exercising 8 8 5 24 3 :=
by
  -- Proof is omitted as per instruction
  sorry

end Jackie_exercise_hours_l297_29737


namespace no_real_number_pairs_satisfy_equation_l297_29731

theorem no_real_number_pairs_satisfy_equation :
  ∀ (a b : ℝ), a ≠ 0 → b ≠ 0 →
  ¬ (1 / a + 1 / b = 1 / (2 * a + 3 * b)) :=
by
  intros a b ha hb
  sorry

end no_real_number_pairs_satisfy_equation_l297_29731


namespace trajectory_equation_l297_29794

open Real

-- Define points M and N
def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (2, 0)

-- Define the moving point P
def P (x y : ℝ) : Prop := 
  (4 * Real.sqrt ((x + 2) ^ 2 + y ^ 2) + 4 * (x - 2) = 0) → 
  (y ^ 2 = -8 * x)

-- The theorem stating the desired proof problem
theorem trajectory_equation (x y : ℝ) : P x y :=
sorry

end trajectory_equation_l297_29794


namespace dance_lesson_cost_l297_29783

-- Define the conditions
variable (total_lessons : Nat) (free_lessons : Nat) (paid_lessons_cost : Nat)

-- State the problem with the given conditions
theorem dance_lesson_cost
  (h1 : total_lessons = 10)
  (h2 : free_lessons = 2)
  (h3 : paid_lessons_cost = 80) :
  let number_of_paid_lessons := total_lessons - free_lessons
  number_of_paid_lessons ≠ 0 -> 
  (paid_lessons_cost / number_of_paid_lessons) = 10 := by
  sorry

end dance_lesson_cost_l297_29783


namespace ellipse_slope_product_l297_29782

variables {a b x1 y1 x2 y2 : ℝ} (h₁ : a > b) (h₂ : b > 0) (h₃ : (b^2 * x1^2 + a^2 * y1^2 = a^2 * b^2) ∧ (b^2 * x2^2 + a^2 * y2^2 = a^2 * b^2))

theorem ellipse_slope_product : 
  (a > b) → (b > 0) → (b^2 * x1^2 + a^2 * y1^2 = a^2 * b^2) → 
  (b^2 * x2^2 + a^2 * y2^2 = a^2 * b^2) → 
  ( (y1 + y2)/(x1 + x2) ) * ( (y1 - y2)/(x1 - x2) ) = - (b^2 / a^2) :=
by
  intros ha hb hxy1 hxy2
  sorry

end ellipse_slope_product_l297_29782


namespace greatest_of_consecutive_integers_sum_18_l297_29703

theorem greatest_of_consecutive_integers_sum_18 
  (x : ℤ) 
  (h1 : x + (x + 1) + (x + 2) = 18) : 
  max x (max (x + 1) (x + 2)) = 7 := 
sorry

end greatest_of_consecutive_integers_sum_18_l297_29703


namespace find_other_side_length_l297_29733

variable (total_shingles : ℕ)
variable (shingles_per_sqft : ℕ)
variable (num_roofs : ℕ)
variable (side_length : ℕ)

theorem find_other_side_length
  (h1 : total_shingles = 38400)
  (h2 : shingles_per_sqft = 8)
  (h3 : num_roofs = 3)
  (h4 : side_length = 20)
  : (total_shingles / shingles_per_sqft / num_roofs / 2) / side_length = 40 :=
by
  sorry

end find_other_side_length_l297_29733


namespace find_X_l297_29789

variable (E X : ℕ)

-- Theorem statement
theorem find_X (hE : E = 9)
              (hSum : E * 100 + E * 10 + E + E * 100 + E * 10 + E = 1798) :
              X = 7 :=
sorry

end find_X_l297_29789


namespace sum_of_coefficients_l297_29714

theorem sum_of_coefficients (a₅ a₄ a₃ a₂ a₁ a₀ : ℤ)
  (h₀ : (x - 2)^5 = a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀)
  (h₁ : a₅ + a₄ + a₃ + a₂ + a₁ + a₀ = -1)
  (h₂ : a₀ = -32) :
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 :=
sorry

end sum_of_coefficients_l297_29714


namespace equation_solution_l297_29726

open Real

theorem equation_solution (x : ℝ) : 
  (x = 4 ∨ x = -1 → 3 * (2 * x - 5) ≠ (2 * x - 5) ^ 2) ∧
  (3 * (2 * x - 5) = (2 * x - 5) ^ 2 → x = 5 / 2 ∨ x = 4) :=
by
  sorry

end equation_solution_l297_29726


namespace largest_divisor_69_86_l297_29745

theorem largest_divisor_69_86 (n : ℕ) (h₁ : 69 % n = 5) (h₂ : 86 % n = 6) : n = 16 := by
  sorry

end largest_divisor_69_86_l297_29745


namespace proof_complement_union_l297_29776

open Set

variable (U A B: Set Nat)

def complement_equiv_union (U A B: Set Nat) : Prop :=
  (U \ A) ∪ B = {0, 2, 3, 6}

theorem proof_complement_union: 
  U = {0, 1, 3, 5, 6, 8} → 
  A = {1, 5, 8} → 
  B = {2} → 
  complement_equiv_union U A B :=
by
  intros h1 h2 h3
  subst h1
  subst h2
  subst h3
  -- Proof omitted
  sorry

end proof_complement_union_l297_29776


namespace inequality1_inequality2_l297_29756

theorem inequality1 (x : ℝ) : x ≠ 2 → (x + 1)/(x - 2) ≥ 3 → 2 < x ∧ x ≤ 7/2 :=
sorry

theorem inequality2 (x a : ℝ) : 
  (x^2 - a * x - 2 * a^2 ≤ 0) → 
  (a = 0 → x = 0) ∧ 
  (a > 0 → -a ≤ x ∧ x ≤ 2 * a) ∧ 
  (a < 0 → 2 * a ≤ x ∧ x ≤ -a) :=
sorry

end inequality1_inequality2_l297_29756


namespace _l297_29747

noncomputable def charlesPictures : Prop :=
  ∀ (bought : ℕ) (drew_today : ℕ) (drew_yesterday_after_work : ℕ) (left : ℕ),
    (bought = 20) →
    (drew_today = 6) →
    (drew_yesterday_after_work = 6) →
    (left = 2) →
    (bought - left - drew_today - drew_yesterday_after_work = 6)

-- We can use this statement "charlesPictures" to represent the theorem to be proved in Lean 4.

end _l297_29747


namespace evaluate_expression_l297_29754

theorem evaluate_expression : (-3)^7 / 3^5 + 2^5 - 7^2 = -26 := 
by
  sorry

end evaluate_expression_l297_29754


namespace quadratic_min_value_l297_29780

theorem quadratic_min_value (p q r : ℝ) (h : ∀ x : ℝ, x^2 + p * x + q + r ≥ -r) : q = p^2 / 4 :=
sorry

end quadratic_min_value_l297_29780


namespace profit_percentage_l297_29701

theorem profit_percentage (CP SP : ℝ) (hCP : CP = 550) (hSP : SP = 715) : 
  ((SP - CP) / CP) * 100 = 30 := sorry

end profit_percentage_l297_29701


namespace polynomial_sum_of_squares_l297_29758

theorem polynomial_sum_of_squares (P : Polynomial ℝ) 
  (hP : ∀ x : ℝ, 0 ≤ P.eval x) : 
  ∃ (f g : Polynomial ℝ), P = f * f + g * g := 
sorry

end polynomial_sum_of_squares_l297_29758


namespace solve_system_of_equations_solve_fractional_equation_l297_29765

noncomputable def solution1 (x y : ℚ) := (3 * x - 5 * y = 3) ∧ (x / 2 - y / 3 = 1) ∧ (x = 8 / 3) ∧ (y = 1)

noncomputable def solution2 (x : ℚ) := (x / (x - 1) + 1 = 3 / (2 * x - 2)) ∧ (x = 5 / 4)

theorem solve_system_of_equations (x y : ℚ) : solution1 x y := by
  sorry

theorem solve_fractional_equation (x : ℚ) : solution2 x := by
  sorry

end solve_system_of_equations_solve_fractional_equation_l297_29765


namespace intersection_of_A_and_B_l297_29777

-- Define the sets A and B based on the given conditions
def A := {x : ℝ | x > 1}
def B := {x : ℝ | x ≤ 3}

-- Lean statement to prove the intersection of A and B matches the correct answer
theorem intersection_of_A_and_B : (A ∩ B) = {x : ℝ | 1 < x ∧ x ≤ 3} :=
by {
  sorry
}

end intersection_of_A_and_B_l297_29777


namespace molecular_weight_of_one_mole_l297_29735

theorem molecular_weight_of_one_mole (total_weight : ℝ) (number_of_moles : ℕ) 
    (h : total_weight = 204) (n : number_of_moles = 3) : 
    (total_weight / number_of_moles) = 68 :=
by
  have h_weight : total_weight = 204 := h
  have h_moles : number_of_moles = 3 := n
  rw [h_weight, h_moles]
  norm_num

end molecular_weight_of_one_mole_l297_29735


namespace positive_intervals_of_product_l297_29755

theorem positive_intervals_of_product (x : ℝ) : 
  ((x + 2) * (x - 3) > 0) ↔ (x < -2 ∨ x > 3) := 
sorry

end positive_intervals_of_product_l297_29755


namespace continuous_at_5_l297_29705

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x > 5 then x - 2 else 3 * x + b

theorem continuous_at_5 {b : ℝ} : ContinuousAt (fun x => f x b) 5 ↔ b = -12 := by
  sorry

end continuous_at_5_l297_29705


namespace relationship_coefficients_l297_29708

-- Definitions based directly on the conditions
def has_extrema (a b c : ℝ) : Prop := b^2 - 3 * a * c > 0
def passes_through_origin (x1 x2 y1 y2 : ℝ) : Prop := x1 * y2 = x2 * y1

-- Main statement proving the relationship among the coefficients
theorem relationship_coefficients (a b c d : ℝ) (x1 x2 y1 y2 : ℝ)
  (h_extrema : has_extrema a b c)
  (h_line : passes_through_origin x1 x2 y1 y2)
  (hx1 : x1 ≠ 0) (hx2 : x2 ≠ 0)
  (h_y1 : y1 = a * x1^3 + b * x1^2 + c * x1 + d)
  (h_y2 : y2 = a * x2^3 + b * x2^2 + c * x2 + d) :
  9 * a * d = b * c :=
sorry

end relationship_coefficients_l297_29708


namespace descent_phase_duration_l297_29739

noncomputable def start_time_in_seconds : ℕ := 45 * 60 + 39
noncomputable def end_time_in_seconds : ℕ := 47 * 60 + 33

theorem descent_phase_duration :
  end_time_in_seconds - start_time_in_seconds = 114 := by
  sorry

end descent_phase_duration_l297_29739


namespace roundness_of_8000000_l297_29793

def is_prime (n : Nat) : Prop := sorry

def prime_factors_exponents (n : Nat) : List (Nat × Nat) := sorry

def roundness (n : Nat) : Nat := 
  (prime_factors_exponents n).foldr (λ p acc => p.2 + acc) 0

theorem roundness_of_8000000 : roundness 8000000 = 15 :=
sorry

end roundness_of_8000000_l297_29793


namespace john_bought_three_sodas_l297_29744

-- Define the conditions

def cost_per_soda := 2
def total_money_paid := 20
def change_received := 14

-- Definition indicating the number of sodas bought
def num_sodas_bought := (total_money_paid - change_received) / cost_per_soda

-- Question: Prove that John bought 3 sodas given these conditions
theorem john_bought_three_sodas : num_sodas_bought = 3 := by
  -- Proof: This is an example of how you may structure the proof
  sorry

end john_bought_three_sodas_l297_29744


namespace initial_ratio_of_milk_to_water_l297_29769

theorem initial_ratio_of_milk_to_water 
  (M W : ℕ) 
  (h1 : M + 10 + W = 30)
  (h2 : (M + 10) * 2 = W * 5)
  (h3 : M + W = 20) : 
  M = 11 ∧ W = 9 := 
by 
  sorry

end initial_ratio_of_milk_to_water_l297_29769


namespace total_sample_size_is_72_l297_29759

-- Definitions based on the given conditions:
def production_A : ℕ := 600
def production_B : ℕ := 1200
def production_C : ℕ := 1800
def total_production : ℕ := production_A + production_B + production_C
def sampled_B : ℕ := 2

-- Main theorem to prove the sample size:
theorem total_sample_size_is_72 : 
  ∃ (n : ℕ), 
    (∃ s_A s_B s_C, 
      s_A = (production_A * sampled_B * total_production) / production_B^2 ∧ 
      s_B = sampled_B ∧ 
      s_C = (production_C * sampled_B * total_production) / production_B^2 ∧
      n = s_A + s_B + s_C) ∧ 
  (n = 72) :=
sorry

end total_sample_size_is_72_l297_29759


namespace centroid_inverse_square_sum_l297_29757

theorem centroid_inverse_square_sum
  (α β γ p q r : ℝ)
  (h1 : 1/α^2 + 1/β^2 + 1/γ^2 = 1)
  (hp : p = α / 3)
  (hq : q = β / 3)
  (hr : r = γ / 3) :
  (1/p^2 + 1/q^2 + 1/r^2 = 9) :=
sorry

end centroid_inverse_square_sum_l297_29757


namespace leo_average_speed_last_segment_l297_29750

theorem leo_average_speed_last_segment :
  let total_distance := 135
  let total_time_hr := 135 / 60.0
  let segment_time_hr := 45 / 60.0
  let first_segment_distance := 55 * segment_time_hr
  let second_segment_distance := 70 * segment_time_hr
  let last_segment_distance := total_distance - (first_segment_distance + second_segment_distance)
  last_segment_distance / segment_time_hr = 55 :=
by
  sorry

end leo_average_speed_last_segment_l297_29750


namespace value_of_x_squared_plus_y_squared_l297_29767

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h1 : x^2 = 8 * x + y) (h2 : y^2 = x + 8 * y) (h3 : x ≠ y) : 
  x^2 + y^2 = 63 := sorry

end value_of_x_squared_plus_y_squared_l297_29767


namespace tangent_line_correct_l297_29761

-- Define the curve y = x^3 - 1
def curve (x : ℝ) : ℝ := x^3 - 1

-- Define the derivative of the curve
def derivative_curve (x : ℝ) : ℝ := 3 * x^2

-- Define the point of tangency
def tangent_point : ℝ × ℝ := (1, curve 1)

-- Define the tangent line equation at x = 1
def tangent_line (x : ℝ) : ℝ := 3 * x - 3

-- The formal statement to be proven
theorem tangent_line_correct :
  ∀ x : ℝ, curve x = x^3 - 1 ∧ derivative_curve x = 3 * x^2 ∧ tangent_point = (1, 0) → 
    tangent_line 1 = 3 * 1 - 3 :=
by
  sorry

end tangent_line_correct_l297_29761


namespace sum_quotient_product_diff_l297_29725

theorem sum_quotient_product_diff (x y : ℚ) (h₁ : x + y = 6) (h₂ : x / y = 6) : 
  (x * y) - (x - y) = 6 / 49 :=
  sorry

end sum_quotient_product_diff_l297_29725


namespace max_possible_scores_l297_29719

theorem max_possible_scores (num_questions : ℕ) (points_correct : ℤ) (points_incorrect : ℤ) (points_unanswered : ℤ) :
  num_questions = 10 →
  points_correct = 4 →
  points_incorrect = -1 →
  points_unanswered = 0 →
  ∃ n, n = 45 :=
by
  sorry

end max_possible_scores_l297_29719


namespace probability_red_and_at_least_one_even_l297_29791

-- Definitions based on conditions
def total_balls : ℕ := 12
def red_balls : Finset ℕ := {1, 2, 3, 4, 5, 6}
def black_balls : Finset ℕ := {7, 8, 9, 10, 11, 12}

-- Condition to check if a ball is red
def is_red (n : ℕ) : Prop := n ∈ red_balls

-- Condition to check if a ball has an even number
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Total number of ways to draw two balls with replacement
def total_ways : ℕ := total_balls * total_balls

-- Number of ways to draw both red balls
def red_red_ways : ℕ := Finset.card red_balls * Finset.card red_balls

-- Number of ways to draw both red balls with none even
def red_odd_numbers : Finset ℕ := {1, 3, 5}
def red_red_odd_ways : ℕ := Finset.card red_odd_numbers * Finset.card red_odd_numbers

-- Number of ways to draw both red balls with at least one even
def desired_outcomes : ℕ := red_red_ways - red_red_odd_ways

-- The probability
def probability : ℚ := desired_outcomes / total_ways

theorem probability_red_and_at_least_one_even :
  probability = 3 / 16 :=
by
  sorry

end probability_red_and_at_least_one_even_l297_29791


namespace min_value_a1_plus_a7_l297_29797

theorem min_value_a1_plus_a7 (a : ℕ → ℝ) (r : ℝ) 
  (h1 : ∀ n, a n > 0) 
  (h2 : ∀ n, a (n+1) = a n * r) 
  (h3 : a 3 * a 5 = 64) : 
  a 1 + a 7 ≥ 16 := 
sorry

end min_value_a1_plus_a7_l297_29797


namespace ticket_price_values_l297_29775

theorem ticket_price_values : 
  ∃ (x_values : Finset ℕ), 
    (∀ x ∈ x_values, x ∣ 60 ∧ x ∣ 80) ∧ 
    x_values.card = 6 :=
by
  sorry

end ticket_price_values_l297_29775


namespace probability_kyle_catherine_not_david_l297_29779

/--
Kyle, David, and Catherine each try independently to solve a problem. 
Their individual probabilities for success are 1/3, 2/7, and 5/9.
Prove that the probability that Kyle and Catherine, but not David, will solve the problem is 25/189.
-/
theorem probability_kyle_catherine_not_david :
  let P_K := 1 / 3
  let P_D := 2 / 7
  let P_C := 5 / 9
  let P_D_c := 1 - P_D
  P_K * P_C * P_D_c = 25 / 189 :=
by
  sorry

end probability_kyle_catherine_not_david_l297_29779


namespace investment_period_two_years_l297_29784

theorem investment_period_two_years
  (P : ℝ) (r : ℝ) (A : ℝ) (n : ℕ) (hP : P = 6000) (hr : r = 0.10) (hA : A = 7260) (hn : n = 1) : 
  ∃ t : ℝ, t = 2 ∧ A = P * (1 + r / n) ^ (n * t) :=
by
  sorry

end investment_period_two_years_l297_29784


namespace scientific_notation_of_218000000_l297_29764

theorem scientific_notation_of_218000000 :
  218000000 = 2.18 * 10^8 :=
sorry

end scientific_notation_of_218000000_l297_29764


namespace axis_of_symmetry_l297_29742

-- Define the given parabolic function
def parabola (x : ℝ) : ℝ := (2 - x) * x

-- Define the axis of symmetry property for the given parabola
theorem axis_of_symmetry : ∀ x : ℝ, ((2 - x) * x) = -((x - 1)^2) + 1 → (∃ x_sym : ℝ, x_sym = 1) :=
by
  sorry

end axis_of_symmetry_l297_29742
