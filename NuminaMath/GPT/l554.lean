import Mathlib

namespace find_a2_plus_b2_l554_55467

theorem find_a2_plus_b2 (a b : ℝ) :
  (∀ x, |a * Real.sin x + b * Real.cos x - 1| + |b * Real.sin x - a * Real.cos x| ≤ 11)
  → a^2 + b^2 = 50 :=
by
  sorry

end find_a2_plus_b2_l554_55467


namespace sum_of_roots_of_polynomial_l554_55425

theorem sum_of_roots_of_polynomial (a b c : ℝ) (h : 3*a^3 - 7*a^2 + 6*a = 0) : 
    (∀ x, 3*x^2 - 7*x + 6 = 0 → x = a ∨ x = b ∨ x = c) →
    (∀ (x : ℝ), (x = a ∨ x = b ∨ x = c → 3*x^3 - 7*x^2 + 6*x = 0)) → 
    a + b + c = 7 / 3 :=
sorry

end sum_of_roots_of_polynomial_l554_55425


namespace volume_of_larger_cube_is_343_l554_55441

-- We will define the conditions first
def smaller_cube_side_length : ℤ := 1
def number_of_smaller_cubes : ℤ := 343
def volume_small_cube (l : ℤ) : ℤ := l^3
def diff_surface_area (l L : ℤ) : ℤ := (number_of_smaller_cubes * 6 * l^2) - (6 * L^2)

-- Main statement to prove the volume of the larger cube
theorem volume_of_larger_cube_is_343 :
  ∃ L, volume_small_cube smaller_cube_side_length * number_of_smaller_cubes = L^3 ∧
        diff_surface_area smaller_cube_side_length L = 1764 ∧
        volume_small_cube L = 343 :=
by
  sorry

end volume_of_larger_cube_is_343_l554_55441


namespace sum_X_Y_Z_W_eq_156_l554_55483

theorem sum_X_Y_Z_W_eq_156 
  (X Y Z W : ℕ) 
  (h_arith_seq : Y - X = Z - Y)
  (h_geom_seq : Z / Y = 9 / 5)
  (h_W : W = Z^2 / Y) 
  (h_pos : 0 < X ∧ 0 < Y ∧ 0 < Z ∧ 0 < W) :
  X + Y + Z + W = 156 :=
sorry

end sum_X_Y_Z_W_eq_156_l554_55483


namespace num_ways_to_distribute_7_balls_in_4_boxes_l554_55437

def num_ways_to_distribute_balls (balls boxes : ℕ) : ℕ :=
  -- Implement the function to calculate the number of ways here, but we'll keep it as a placeholder for now.
  sorry

theorem num_ways_to_distribute_7_balls_in_4_boxes : 
  num_ways_to_distribute_balls 7 4 = 3 := 
sorry

end num_ways_to_distribute_7_balls_in_4_boxes_l554_55437


namespace men_sent_to_other_project_l554_55496

-- Let the initial number of men be 50
def initial_men : ℕ := 50
-- Let the time to complete the work initially be 10 days
def initial_days : ℕ := 10
-- Calculate the total work in man-days
def total_work : ℕ := initial_men * initial_days

-- Let the total time taken after sending some men to another project be 30 days
def new_days : ℕ := 30
-- Let the number of men sent to another project be x
variable (x : ℕ)
-- Let the new number of men be (initial_men - x)
def new_men : ℕ := initial_men - x

theorem men_sent_to_other_project (x : ℕ):
total_work = new_men x * new_days -> x = 33 :=
by
  sorry

end men_sent_to_other_project_l554_55496


namespace hyperbola_correct_eqn_l554_55414

open Real

def hyperbola_eqn (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 12 = 1

theorem hyperbola_correct_eqn (e c a b x y : ℝ)
  (h_eccentricity : e = 2)
  (h_foci_distance : c = 4)
  (h_major_axis_half_length : a = 2)
  (h_minor_axis_half_length_square : b^2 = c^2 - a^2) :
  hyperbola_eqn x y :=
by
  sorry

end hyperbola_correct_eqn_l554_55414


namespace greatest_x_is_53_l554_55498

-- Define the polynomial expression
def polynomial (x : ℤ) : ℤ := x^2 + 2 * x + 13

-- Define the condition for the expression to be an integer
def isIntegerWhenDivided (x : ℤ) : Prop := (polynomial x) % (x - 5) = 0

-- Define the theorem to prove the greatest integer value of x
theorem greatest_x_is_53 : ∃ x : ℤ, isIntegerWhenDivided x ∧ (∀ y : ℤ, isIntegerWhenDivided y → y ≤ x) ∧ x = 53 :=
by
  sorry

end greatest_x_is_53_l554_55498


namespace number_of_people_in_group_l554_55405

variable (T L : ℕ)

theorem number_of_people_in_group
  (h1 : 90 + L = T)
  (h2 : (L : ℚ) / T = 0.4) :
  T = 150 := by
  sorry

end number_of_people_in_group_l554_55405


namespace book_pages_l554_55431

theorem book_pages (n days_n : ℕ) (first_day_pages break_days : ℕ) (common_difference total_pages_read : ℕ) (portion_of_book : ℚ) :
    n = 14 → days_n = 12 → first_day_pages = 10 → break_days = 2 → common_difference = 2 →
    total_pages_read = 252 → portion_of_book = 3/4 →
    (total_pages_read : ℚ) * (4/3) = 336 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end book_pages_l554_55431


namespace darma_peanut_consumption_l554_55444

theorem darma_peanut_consumption :
  ∀ (t : ℕ) (rate : ℕ),
  (rate = 20 / 15) →  -- Given the rate of peanut consumption
  (t = 6 * 60) →     -- Given that the total time is 6 minutes
  (rate * t = 480) :=  -- Prove that the total number of peanuts eaten in 6 minutes is 480
by
  intros t rate h_rate h_time
  sorry

end darma_peanut_consumption_l554_55444


namespace student_papers_count_l554_55417

theorem student_papers_count {F n k: ℝ}
  (h1 : 35 * k = 0.6 * n * F)
  (h2 : 5 * k > 0.5 * F)
  (h3 : 6 * k > 0.5 * F)
  (h4 : 7 * k > 0.5 * F)
  (h5 : 8 * k > 0.5 * F)
  (h6 : 9 * k > 0.5 * F) :
  n = 5 :=
by
  sorry

end student_papers_count_l554_55417


namespace asymptote_slope_l554_55470

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 144 - y^2 / 81 = 1

-- Lean statement to prove slope of asymptotes
theorem asymptote_slope :
  (∀ x y : ℝ, hyperbola x y → (y/x) = 3/4 ∨ (y/x) = -(3/4)) :=
by
  sorry

end asymptote_slope_l554_55470


namespace max_m_n_value_l554_55443

theorem max_m_n_value : ∀ (m n : ℝ), (n = -m^2 + 3) → m + n ≤ 13 / 4 :=
by
  intros m n h
  -- The proof will go here, which is omitted for now.
  sorry

end max_m_n_value_l554_55443


namespace length_of_AD_l554_55451

theorem length_of_AD 
  (A B C D : Type) 
  (vertex_angle_equal: ∀ {a b c d : Type}, a = A →
    ∀ (AB AC AD : ℝ), (AB = 24) → (AC = 54) → (AD = 36)) 
  (right_triangles : ∀ {a b : Type}, a = A → ∀ {AB AC : ℝ}, (AB > 0) → (AC > 0) → (AB ^ 2 + AC ^ 2 = AD ^ 2)) :
  ∃ (AD : ℝ), AD = 36 :=
by
  sorry

end length_of_AD_l554_55451


namespace molecular_weight_correct_l554_55435

-- Define atomic weights of elements
def atomic_weight_Ba : ℝ := 137.33
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01
def atomic_weight_D : ℝ := 2.01

-- Define the number of each type of atom in the compound
def num_Ba : ℕ := 2
def num_O : ℕ := 3
def num_H : ℕ := 4
def num_D : ℕ := 1

-- Define the molecular weight calculation
def molecular_weight : ℝ :=
  (num_Ba * atomic_weight_Ba) +
  (num_O * atomic_weight_O) +
  (num_H * atomic_weight_H) +
  (num_D * atomic_weight_D)

-- Theorem stating the molecular weight is 328.71 g/mol
theorem molecular_weight_correct :
  molecular_weight = 328.71 :=
by
  -- The proof will go here
  sorry

end molecular_weight_correct_l554_55435


namespace find_parabola_eq_find_range_of_b_l554_55438

-- Problem 1: Finding the equation of the parabola
theorem find_parabola_eq (p : ℝ) (h1 : p > 0) (x1 x2 y1 y2 : ℝ) 
  (A : (x1 + 4) * 2 = 2 * p * y1) (C : (x2 + 4) * 2 = 2 * p * y2)
  (h3 : x1^2 = 2 * p * y1) (h4 : x2^2 = 2 * p * y2) 
  (h5 : y2 = 4 * y1) :
  x1^2 = 4 * y1 :=
sorry

-- Problem 2: Finding the range of b
theorem find_range_of_b (k : ℝ) (h : k > 0 ∨ k < -4) : 
  ∃ b : ℝ, b = 2 * (k + 1)^2 ∧ b > 2 :=
sorry

end find_parabola_eq_find_range_of_b_l554_55438


namespace commission_percentage_l554_55480

theorem commission_percentage 
  (total_amount : ℝ) 
  (h1 : total_amount = 800) 
  (commission_first_500 : ℝ) 
  (h2 : commission_first_500 = 0.20 * 500) 
  (excess_amount : ℝ) 
  (h3 : excess_amount = (total_amount - 500)) 
  (commission_excess : ℝ) 
  (h4 : commission_excess = 0.25 * excess_amount) 
  (total_commission : ℝ) 
  (h5 : total_commission = commission_first_500 + commission_excess) 
  : (total_commission / total_amount) * 100 = 21.875 := 
by
  sorry

end commission_percentage_l554_55480


namespace max_moves_440_l554_55469

-- Define the set of initial numbers
def initial_numbers : List ℕ := List.range' 1 22

-- Define what constitutes a valid move
def is_valid_move (a b : ℕ) : Prop := b ≥ a + 2

-- Perform the move operation
def perform_move (numbers : List ℕ) (a b : ℕ) : List ℕ :=
  (numbers.erase a).erase b ++ [a + 1, b - 1]

-- Define the maximum number of moves we need to prove
theorem max_moves_440 : ∃ m, m = 440 ∧
  ∀ (moves_done : ℕ) (numbers : List ℕ),
    moves_done <= m → ∃ a b, a ∈ numbers ∧ b ∈ numbers ∧
                             is_valid_move a b ∧
                             numbers = initial_numbers →
                             perform_move numbers a b ≠ numbers
 := sorry

end max_moves_440_l554_55469


namespace max_ratio_MO_MF_on_parabola_l554_55475

theorem max_ratio_MO_MF_on_parabola (F M : ℝ × ℝ) : 
  let O := (0, 0)
  let focus := (1 / 2, 0)
  ∀ (M : ℝ × ℝ), (M.snd ^ 2 = 2 * M.fst) →
  F = focus →
  (∃ m > 0, M.fst = m ∧ M.snd ^ 2 = 2 * m) →
  (∃ t, t = m - (1 / 4)) →
  ∃ value, value = (2 * Real.sqrt 3) / 3 ∧
  ∃ rat, rat = dist M O / dist M F ∧
  rat = value := 
by
  admit

end max_ratio_MO_MF_on_parabola_l554_55475


namespace zoo_animal_difference_l554_55477

variable (giraffes non_giraffes : ℕ)

theorem zoo_animal_difference (h1 : giraffes = 300) (h2 : giraffes = 3 * non_giraffes) : giraffes - non_giraffes = 200 :=
by 
  sorry

end zoo_animal_difference_l554_55477


namespace exists_distinct_positive_integers_l554_55406

theorem exists_distinct_positive_integers (n : ℕ) (h : 0 < n) :
  ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x^(n-1) + y^n = z^(n+1) :=
sorry

end exists_distinct_positive_integers_l554_55406


namespace wendy_total_gas_to_add_l554_55426

-- Conditions as definitions
def truck_tank_capacity : ℕ := 20
def car_tank_capacity : ℕ := 12
def truck_current_gas : ℕ := truck_tank_capacity / 2
def car_current_gas : ℕ := car_tank_capacity / 3

-- The proof problem statement
theorem wendy_total_gas_to_add :
  (truck_tank_capacity - truck_current_gas) + (car_tank_capacity - car_current_gas) = 18 := 
by
  sorry

end wendy_total_gas_to_add_l554_55426


namespace find_n_given_combination_l554_55488

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

theorem find_n_given_combination : ∃ n : ℕ, binomial_coefficient (n+1) 2 = 21 ↔ n = 6 := by
  sorry

end find_n_given_combination_l554_55488


namespace quadratic_form_h_l554_55407

theorem quadratic_form_h (x h : ℝ) (a k : ℝ) (h₀ : 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) : 
  h = -3 / 2 :=
by
  sorry

end quadratic_form_h_l554_55407


namespace find_second_dimension_of_smaller_box_l554_55442

def volume_large_box : ℕ := 12 * 14 * 16
def volume_small_box (x : ℕ) : ℕ := 3 * x * 2
def max_small_boxes : ℕ := 64

theorem find_second_dimension_of_smaller_box (x : ℕ) : volume_large_box = max_small_boxes * volume_small_box x → x = 7 :=
by
  intros h
  unfold volume_large_box at h
  unfold volume_small_box at h
  sorry

end find_second_dimension_of_smaller_box_l554_55442


namespace cricket_players_count_l554_55495

theorem cricket_players_count (hockey: ℕ) (football: ℕ) (softball: ℕ) (total: ℕ) : 
  hockey = 15 ∧ football = 21 ∧ softball = 19 ∧ total = 77 → ∃ cricket, cricket = 22 := by
  sorry

end cricket_players_count_l554_55495


namespace exponent_calculation_l554_55478

theorem exponent_calculation : 10^6 * (10^2)^3 / 10^4 = 10^8 := by
  sorry

end exponent_calculation_l554_55478


namespace vowel_soup_sequences_count_l554_55433

theorem vowel_soup_sequences_count :
  let vowels := 5
  let sequence_length := 6
  vowels ^ sequence_length = 15625 :=
by
  sorry

end vowel_soup_sequences_count_l554_55433


namespace value_of_v_star_star_l554_55485

noncomputable def v_star (v : ℝ) : ℝ :=
  v - v / 3
  
theorem value_of_v_star_star (v : ℝ) (h : v = 8.999999999999998) : v_star (v_star v) = 4.000000000000000 := by
  sorry

end value_of_v_star_star_l554_55485


namespace pythagorean_triangle_exists_l554_55459

theorem pythagorean_triangle_exists (a : ℤ) (h : a ≥ 5) : 
  ∃ (b c : ℤ), c ≥ b ∧ b ≥ a ∧ a^2 + b^2 = c^2 :=
by {
  sorry
}

end pythagorean_triangle_exists_l554_55459


namespace gas_volume_ranking_l554_55436

theorem gas_volume_ranking (Russia_V: ℝ) (Non_West_V: ℝ) (West_V: ℝ)
  (h_russia: Russia_V = 302790.13)
  (h_non_west: Non_West_V = 26848.55)
  (h_west: West_V = 21428): Russia_V > Non_West_V ∧ Non_West_V > West_V :=
by
  have h1: Russia_V = 302790.13 := h_russia
  have h2: Non_West_V = 26848.55 := h_non_west
  have h3: West_V = 21428 := h_west
  sorry


end gas_volume_ranking_l554_55436


namespace parallel_line_slope_l554_55440

theorem parallel_line_slope (x y : ℝ) : 
  (∃ b : ℝ, y = (1 / 2) * x + b) → 
  (∃ a : ℝ, 3 * x - 6 * y = a) → 
  ∃ k : ℝ, k = 1 / 2 :=
by
  intros h1 h2
  sorry

end parallel_line_slope_l554_55440


namespace plane_split_into_8_regions_l554_55409

-- Define the conditions as separate lines in the plane.
def line1 (x y : ℝ) : Prop := y = 2 * x
def line2 (x y : ℝ) : Prop := y = (1 / 2) * x
def line3 (x y : ℝ) : Prop := x = y

-- Define a theorem stating that these lines together split the plane into 8 regions.
theorem plane_split_into_8_regions :
  (∀ (x y : ℝ), line1 x y ∨ line2 x y ∨ line3 x y) →
  -- The plane is split into exactly 8 regions by these lines
  ∃ (regions : ℕ), regions = 8 :=
sorry

end plane_split_into_8_regions_l554_55409


namespace units_digit_of_m_squared_plus_3_to_the_m_l554_55427

theorem units_digit_of_m_squared_plus_3_to_the_m (m : ℕ) (h : m = 2010^2 + 2^2010) : 
  (m^2 + 3^m) % 10 = 7 :=
by {
  sorry -- proof goes here
}

end units_digit_of_m_squared_plus_3_to_the_m_l554_55427


namespace final_inventory_is_correct_l554_55401

def initial_inventory : ℕ := 4500
def bottles_sold_monday : ℕ := 2445
def bottles_sold_tuesday : ℕ := 900
def bottles_sold_per_day_remaining_week : ℕ := 50
def supplier_delivery : ℕ := 650

def bottles_sold_first_two_days : ℕ := bottles_sold_monday + bottles_sold_tuesday
def days_remaining : ℕ := 5
def bottles_sold_remaining_week : ℕ := days_remaining * bottles_sold_per_day_remaining_week
def total_bottles_sold_week : ℕ := bottles_sold_first_two_days + bottles_sold_remaining_week
def remaining_inventory : ℕ := initial_inventory - total_bottles_sold_week
def final_inventory : ℕ := remaining_inventory + supplier_delivery

theorem final_inventory_is_correct :
  final_inventory = 1555 :=
by
  sorry

end final_inventory_is_correct_l554_55401


namespace smaller_number_is_three_l554_55461

theorem smaller_number_is_three (x y : ℝ) (h₁ : x + y = 15) (h₂ : x * y = 36) : min x y = 3 :=
sorry

end smaller_number_is_three_l554_55461


namespace roots_opposite_sign_eq_magnitude_l554_55486

theorem roots_opposite_sign_eq_magnitude (c d e n : ℝ) (h : ((n+2) * (x^2 + c*x + d)) = (n-2) * (2*x - e)) :
  n = (-4 - 2 * c) / (c - 2) :=
by
  sorry

end roots_opposite_sign_eq_magnitude_l554_55486


namespace find_x_l554_55415

def vector_a : ℝ × ℝ × ℝ := (2, -3, 1)
def vector_b (x : ℝ) : ℝ × ℝ × ℝ := (4, -6, x)
def dot_product : (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ) → ℝ
  | (a1, a2, a3), (b1, b2, b3) => a1 * b1 + a2 * b2 + a3 * b3

theorem find_x (x : ℝ) (h : dot_product vector_a (vector_b x) = 0) : x = -26 :=
by 
  sorry

end find_x_l554_55415


namespace tan_315_eq_neg1_l554_55499

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := by
  -- The statement means we need to prove that the tangent of 315 degrees is -1
  sorry

end tan_315_eq_neg1_l554_55499


namespace find_pairs_l554_55468

theorem find_pairs (a b : ℕ) (q r : ℕ) (h1 : a > 0) (h2 : b > 0)
  (h3 : a^2 + b^2 = q * (a + b) + r) (h4 : 0 ≤ r) (h5 : r < a + b)
  (h6 : q^2 + r = 1977) :
  (a, b) = (50, 37) ∨ (a, b) = (50, 7) ∨ (a, b) = (37, 50) ∨ (a, b) = (7, 50) :=
  sorry

end find_pairs_l554_55468


namespace expand_product_l554_55484

theorem expand_product (x : ℝ) (hx : x ≠ 0) : 
  (3 / 7) * (7 / x^3 - 14 * x^4) = 3 / x^3 - 6 * x^4 :=
by
  sorry

end expand_product_l554_55484


namespace find_x_l554_55472

theorem find_x :
  ∃ x : ℝ, x = (1/x) * (-x) - 3*x + 4 ∧ x = 3/4 :=
by
  sorry

end find_x_l554_55472


namespace model_tower_height_l554_55450

theorem model_tower_height (real_height : ℝ) (real_volume : ℝ) (model_volume : ℝ) (h_cond : real_height = 80) (vol_cond : real_volume = 200000) (model_vol_cond : model_volume = 0.2) : 
  ∃ h : ℝ, h = 0.8 :=
by sorry

end model_tower_height_l554_55450


namespace equilateral_triangle_ratio_is_correct_l554_55429

noncomputable def equilateral_triangle_area_perimeter_ratio (a : ℝ) (h_eq : a = 10) : ℝ :=
  let altitude := (Real.sqrt 3 / 2) * a
  let area := (1 / 2) * a * altitude
  let perimeter := 3 * a
  area / perimeter

theorem equilateral_triangle_ratio_is_correct :
  equilateral_triangle_area_perimeter_ratio 10 (by rfl) = 5 * Real.sqrt 3 / 6 :=
by
  sorry

end equilateral_triangle_ratio_is_correct_l554_55429


namespace student_first_subject_percentage_l554_55434

variable (P : ℝ)

theorem student_first_subject_percentage 
  (H1 : 80 = 80)
  (H2 : 75 = 75)
  (H3 : (P + 80 + 75) / 3 = 75) :
  P = 70 :=
by
  sorry

end student_first_subject_percentage_l554_55434


namespace sum_of_possible_values_l554_55463

noncomputable def solution : ℕ :=
  sorry

theorem sum_of_possible_values (x : ℝ) (h : |x - 5| - 4 = 0) : solution = 10 :=
by
  sorry

end sum_of_possible_values_l554_55463


namespace select_3_males_2_females_select_at_least_1_captain_select_at_least_1_female_select_both_captain_and_female_l554_55471

variable (n m : ℕ) -- n for males, m for females
variable (mc fc : ℕ) -- mc for male captain, fc for female captain

def num_ways_3_males_2_females : ℕ :=
  (Nat.choose 6 3) * (Nat.choose 4 2)

def num_ways_at_least_1_captain : ℕ :=
  (2 * (Nat.choose 8 4)) + (Nat.choose 8 3)

def num_ways_at_least_1_female : ℕ :=
  (Nat.choose 10 5) - (Nat.choose 6 5)

def num_ways_both_captain_and_female : ℕ :=
  (Nat.choose 10 5) - (Nat.choose 8 5) - (Nat.choose 5 4)

theorem select_3_males_2_females : num_ways_3_males_2_females = 120 := by
  sorry
  
theorem select_at_least_1_captain : num_ways_at_least_1_captain = 196 := by
  sorry
  
theorem select_at_least_1_female : num_ways_at_least_1_female = 246 := by
  sorry
  
theorem select_both_captain_and_female : num_ways_both_captain_and_female = 191 := by
  sorry

end select_3_males_2_females_select_at_least_1_captain_select_at_least_1_female_select_both_captain_and_female_l554_55471


namespace function_behavior_on_intervals_l554_55424

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem function_behavior_on_intervals :
  (∀ x : ℝ, 0 < x ∧ x < Real.exp 1 → 0 < deriv f x) ∧
  (∀ x : ℝ, Real.exp 1 < x ∧ x < 10 → deriv f x < 0) := sorry

end function_behavior_on_intervals_l554_55424


namespace part1_part2_l554_55446

-- Part (1)
theorem part1 (x y : ℝ) (h1 : abs x = 3) (h2 : abs y = 7) (hx : x > 0) (hy : y < 0) : x + y = -4 :=
sorry

-- Part (2)
theorem part2 (x y : ℝ) (h1 : abs x = 3) (h2 : abs y = 7) (hxy : x < y) : x - y = -10 ∨ x - y = -4 :=
sorry

end part1_part2_l554_55446


namespace smallest_positive_period_of_f_cos_2x0_l554_55474

noncomputable def f (x : ℝ) : ℝ := 
  2 * Real.sin x * Real.cos x + 2 * (Real.sqrt 3) * (Real.cos x)^2 - Real.sqrt 3

theorem smallest_positive_period_of_f :
  (∃ p > 0, ∀ x, f x = f (x + p)) ∧
  (∀ q > 0, (∀ x, f x = f (x + q)) -> q ≥ Real.pi) :=
sorry

theorem cos_2x0 (x0 : ℝ) (h0 : x0 ∈ Set.Icc (Real.pi / 4) (Real.pi / 2)) 
  (h1 : f (x0 - Real.pi / 12) = 6 / 5) :
  Real.cos (2 * x0) = (3 - 4 * Real.sqrt 3) / 10 :=
sorry

end smallest_positive_period_of_f_cos_2x0_l554_55474


namespace chair_cost_l554_55411

namespace ChairCost

-- Conditions
def total_cost : ℕ := 135
def table_cost : ℕ := 55
def chairs_count : ℕ := 4

-- Problem Statement
theorem chair_cost : (total_cost - table_cost) / chairs_count = 20 :=
by
  sorry

end ChairCost

end chair_cost_l554_55411


namespace solution_correctness_l554_55420

noncomputable def solution_set : Set ℝ := { x : ℝ | (x + 1) * (x - 2) > 0 }

theorem solution_correctness (x : ℝ) :
  (x ∈ solution_set) ↔ (x < -1 ∨ x > 2) :=
by sorry

end solution_correctness_l554_55420


namespace translation_of_segment_l554_55454

structure Point where
  x : ℝ
  y : ℝ

variables (A B A' : Point)

def translation_vector (P Q : Point) : Point :=
  { x := Q.x - P.x,
    y := Q.y - P.y }

def translate (P Q : Point) : Point :=
  { x := P.x + Q.x,
    y := P.y + Q.y }

theorem translation_of_segment (hA : A = {x := -2, y := 0})
                                (hB : B = {x := 0, y := 3})
                                (hA' : A' = {x := 2, y := 1}) :
  translate B (translation_vector A A') = {x := 4, y := 4} := by
  sorry

end translation_of_segment_l554_55454


namespace find_c_l554_55439

theorem find_c (a b c : ℝ) (h_line : 4 * a - 3 * b + c = 0) 
  (h_min : (a - 1)^2 + (b - 1)^2 = 4) : c = 9 ∨ c = -11 := 
    sorry

end find_c_l554_55439


namespace total_games_played_l554_55400

theorem total_games_played (n : ℕ) (h : n = 8) : (n.choose 2) = 28 := by
  sorry

end total_games_played_l554_55400


namespace salami_pizza_fraction_l554_55402

theorem salami_pizza_fraction 
    (d_pizza : ℝ) 
    (n_salami_diameter : ℕ) 
    (n_salami_total : ℕ) 
    (h1 : d_pizza = 16)
    (h2 : n_salami_diameter = 8) 
    (h3 : n_salami_total = 32) 
    : 
    (32 * (Real.pi * (d_pizza / (2 * n_salami_diameter / 2)) ^ 2)) / (Real.pi * (d_pizza / 2) ^ 2) = 1 / 2 := 
by 
  sorry

end salami_pizza_fraction_l554_55402


namespace perpendicular_lines_condition_l554_55455

theorem perpendicular_lines_condition (a : ℝ) :
  (a = 2) ↔ (∃ m1 m2 : ℝ, (m1 = -1/(4 : ℝ)) ∧ (m2 = (4 : ℝ)) ∧ (m1 * m2 = -1)) :=
by sorry

end perpendicular_lines_condition_l554_55455


namespace factorization_correct_l554_55423

-- Define the expression
def expression (a b : ℝ) : ℝ := 3 * a^2 - 3 * b^2

-- Define the factorized form of the expression
def factorized (a b : ℝ) : ℝ := 3 * (a + b) * (a - b)

-- The main statement we need to prove
theorem factorization_correct (a b : ℝ) : expression a b = factorized a b :=
by 
  sorry -- Proof to be filled in

end factorization_correct_l554_55423


namespace find_value_l554_55464

variable (number : ℝ) (V : ℝ)

theorem find_value
  (h1 : number = 8)
  (h2 : 0.75 * number + V = 8) : V = 2 := by
  sorry

end find_value_l554_55464


namespace solve_quadratic_eq_l554_55428

theorem solve_quadratic_eq (x : ℝ) : x^2 - 4 * x = 2 ↔ (x = 2 + Real.sqrt 6) ∨ (x = 2 - Real.sqrt 6) :=
by
  sorry

end solve_quadratic_eq_l554_55428


namespace ratio_female_democrats_l554_55462

theorem ratio_female_democrats (total_participants male_participants female_participants total_democrats female_democrats : ℕ)
  (h1 : total_participants = 750)
  (h2 : male_participants + female_participants = total_participants)
  (h3 : total_democrats = total_participants / 3)
  (h4 : female_democrats = 125)
  (h5 : total_democrats = male_participants / 4 + female_democrats) :
  (female_democrats / female_participants : ℝ) = 1 / 2 :=
sorry

end ratio_female_democrats_l554_55462


namespace students_not_making_cut_l554_55457

theorem students_not_making_cut :
  let girls := 39
  let boys := 4
  let called_back := 26
  let total := girls + boys
  total - called_back = 17 :=
by
  -- add the proof here
  sorry

end students_not_making_cut_l554_55457


namespace temp_pot_C_to_F_l554_55458

-- Definitions
def boiling_point_C : ℕ := 100
def boiling_point_F : ℕ := 212
def melting_point_C : ℕ := 0
def melting_point_F : ℕ := 32
def temp_pot_C : ℕ := 55
def celsius_to_fahrenheit (c : ℕ) : ℕ := (c * 9 / 5) + 32

-- Theorem to be proved
theorem temp_pot_C_to_F : celsius_to_fahrenheit temp_pot_C = 131 := by
  sorry

end temp_pot_C_to_F_l554_55458


namespace sum_of_numbers_l554_55466

theorem sum_of_numbers :
  1357 + 7531 + 3175 + 5713 = 17776 :=
by
  sorry

end sum_of_numbers_l554_55466


namespace inequality_solution_set_l554_55447

noncomputable def solution_set := { x : ℝ | 0 < x ∧ x < 2 }

theorem inequality_solution_set : 
  { x : ℝ | (4 / x > |x|) } = solution_set :=
by sorry

end inequality_solution_set_l554_55447


namespace find_a8_l554_55412

noncomputable def a (n : ℕ) : ℤ := sorry

noncomputable def b (n : ℕ) : ℤ := a (n + 1) - a n

theorem find_a8 :
  (a 1 = 3) ∧
  (∀ n : ℕ, b n = b 1 + n * 2) ∧
  (b 3 = -2) ∧
  (b 10 = 12) →
  a 8 = 3 :=
by sorry

end find_a8_l554_55412


namespace inequality_solution_l554_55487

theorem inequality_solution (x : ℝ) : x^3 - 9 * x^2 + 27 * x > 0 → (x > 0 ∧ x < 3) ∨ (x > 6) := sorry

end inequality_solution_l554_55487


namespace range_of_a_l554_55476

open Real

-- Definitions of the propositions p and q
def p (a : ℝ) : Prop := (2 - a > 0) ∧ (a + 1 > 0)

def discriminant (a : ℝ) : ℝ := 16 + 4 * a

def q (a : ℝ) : Prop := discriminant a ≥ 0

/--
Given propositions p and q defined above,
prove that the range of real number values for a 
such that ¬p ∧ q is true is
- 4 ≤ a ∧ a ≤ -1 ∨ a ≥ 2
--/
theorem range_of_a (a : ℝ) : (¬ p a ∧ q a) → (-4 ≤ a ∧ a ≤ -1 ∨ a ≥ 2) :=
by
  sorry

end range_of_a_l554_55476


namespace third_term_of_arithmetic_sequence_l554_55456

theorem third_term_of_arithmetic_sequence (a d : ℝ) (h : a + (a + 4 * d) = 10) : a + 2 * d = 5 :=
by {
  sorry
}

end third_term_of_arithmetic_sequence_l554_55456


namespace person_b_lap_time_l554_55491

noncomputable def lap_time_b (a_lap_time : ℕ) (meet_time : ℕ) : ℕ :=
  let combined_speed := 1 / meet_time
  let a_speed := 1 / a_lap_time
  let b_speed := combined_speed - a_speed
  1 / b_speed

theorem person_b_lap_time 
  (a_lap_time : ℕ) 
  (meet_time : ℕ) 
  (h1 : a_lap_time = 80) 
  (h2 : meet_time = 30) : 
  lap_time_b a_lap_time meet_time = 48 := 
by 
  rw [lap_time_b, h1, h2]
  -- Provided steps to solve the proof, skipped here only for statement
  sorry

end person_b_lap_time_l554_55491


namespace system_soln_l554_55432

theorem system_soln (a1 b1 a2 b2 : ℚ)
  (h1 : a1 * 3 + b1 * 6 = 21)
  (h2 : a2 * 3 + b2 * 6 = 12) :
  (3 = 3 ∧ -3 = -3) ∧ (a1 * (2 * 3 + -3) + b1 * (3 - -3) = 21) ∧ (a2 * (2 * 3 + -3) + b2 * (3 - -3) = 12) :=
by
  sorry

end system_soln_l554_55432


namespace blueberry_basket_count_l554_55403

noncomputable def number_of_blueberry_baskets 
    (plums_in_basket : ℕ) 
    (plum_baskets : ℕ) 
    (blueberries_in_basket : ℕ) 
    (total_fruits : ℕ) : ℕ := 
  let total_plums := plum_baskets * plums_in_basket
  let total_blueberries := total_fruits - total_plums
  total_blueberries / blueberries_in_basket

theorem blueberry_basket_count
  (plums_in_basket : ℕ) 
  (plum_baskets : ℕ) 
  (blueberries_in_basket : ℕ) 
  (total_fruits : ℕ)
  (h1 : plums_in_basket = 46)
  (h2 : plum_baskets = 19)
  (h3 : blueberries_in_basket = 170)
  (h4 : total_fruits = 1894) : 
  number_of_blueberry_baskets plums_in_basket plum_baskets blueberries_in_basket total_fruits = 6 := by
  sorry

end blueberry_basket_count_l554_55403


namespace A_minus_one_not_prime_l554_55430

theorem A_minus_one_not_prime (n : ℕ) (h : 0 < n) (m : ℕ) (h1 : 10^(m-1) < 14^n) (h2 : 14^n < 10^m) :
  ¬ (Nat.Prime (2^n * 10^m + 14^n - 1)) :=
by
  sorry

end A_minus_one_not_prime_l554_55430


namespace false_proposition_A_false_proposition_B_true_proposition_C_true_proposition_D_l554_55452

-- Proposition A
theorem false_proposition_A (a b c : ℝ) (hac : a > b) (hca : b > 0) : ac * c^2 = b * c^2 :=
  sorry

-- Proposition B
theorem false_proposition_B (a b : ℝ) (hab : a < b) : (1/a) < (1/b) :=
  sorry

-- Proposition C
theorem true_proposition_C (a b : ℝ) (hab : a > b) (hba : b > 0) : a^2 > a * b ∧ a * b > b^2 :=
  sorry

-- Proposition D
theorem true_proposition_D (a b : ℝ) (hba : a > |b|) : a^2 > b^2 :=
  sorry

end false_proposition_A_false_proposition_B_true_proposition_C_true_proposition_D_l554_55452


namespace find_third_number_l554_55497

theorem find_third_number (x : ℕ) (h : 3 * 16 + 3 * 17 + 3 * x + 11 = 170) : x = 20 := by
  sorry

end find_third_number_l554_55497


namespace profit_is_5000_l554_55489

namespace HorseshoeProfit

-- Defining constants and conditions
def initialOutlay : ℝ := 10000
def costPerSet : ℝ := 20
def sellingPricePerSet : ℝ := 50
def numberOfSets : ℝ := 500

-- Calculating the profit
def profit : ℝ :=
  let revenue := numberOfSets * sellingPricePerSet
  let manufacturingCosts := initialOutlay + (costPerSet * numberOfSets)
  revenue - manufacturingCosts

-- The main theorem: the profit is $5,000
theorem profit_is_5000 : profit = 5000 := by
  sorry

end HorseshoeProfit

end profit_is_5000_l554_55489


namespace simplify_fractions_l554_55492

-- Define the fractions and their product.
def fraction1 : ℚ := 14 / 3
def fraction2 : ℚ := 9 / -42

-- Define the product of the fractions with scalar multiplication by 5.
def product : ℚ := 5 * fraction1 * fraction2

-- The target theorem to prove the equivalence.
theorem simplify_fractions : product = -5 := 
sorry  -- Proof is omitted

end simplify_fractions_l554_55492


namespace ellen_painted_17_lilies_l554_55421

theorem ellen_painted_17_lilies :
  (∃ n : ℕ, n * 5 + 10 * 7 + 6 * 3 + 20 * 2 = 213) → 
    ∃ n : ℕ, n = 17 := 
by sorry

end ellen_painted_17_lilies_l554_55421


namespace part_a_l554_55473

-- Define the sequences and their properties
variables {n : ℕ} (h1 : n ≥ 3)
variables (a b : ℕ → ℝ)
variables (h_arith : ∀ k, a (k+1) = a k + d)
variables (h_geom : ∀ k, b (k+1) = b k * q)
variables (h_a1_b1 : a 1 = b 1)
variables (h_an_bn : a n = b n)

-- State the theorem to be proven
theorem part_a (k : ℕ) (h_k : 2 ≤ k ∧ k ≤ n - 1) : a k > b k :=
  sorry

end part_a_l554_55473


namespace simplify_expression_l554_55418

theorem simplify_expression (x : ℝ) : x^2 * x^4 + x * x^2 * x^3 = 2 * x^6 := by
  sorry

end simplify_expression_l554_55418


namespace adam_bought_dog_food_packages_l554_55453

-- Define the constants and conditions
def num_cat_food_packages : ℕ := 9
def cans_per_cat_food_package : ℕ := 10
def cans_per_dog_food_package : ℕ := 5
def additional_cat_food_cans : ℕ := 55

-- Define the variable for dog food packages and our equation
def num_dog_food_packages (d : ℕ) : Prop :=
  (num_cat_food_packages * cans_per_cat_food_package) = (d * cans_per_dog_food_package + additional_cat_food_cans)

-- The theorem statement representing the proof problem
theorem adam_bought_dog_food_packages : ∃ d : ℕ, num_dog_food_packages d ∧ d = 7 :=
sorry

end adam_bought_dog_food_packages_l554_55453


namespace jack_sees_color_change_l554_55482

noncomputable def traffic_light_cycle := 95    -- Total duration of the traffic light cycle
noncomputable def change_window := 15          -- Duration window where color change occurs
def observation_interval := 5                  -- Length of Jack's observation interval

/-- Probability that Jack sees the color change during his observation. -/
def probability_of_observing_change (cycle: ℕ) (window: ℕ) : ℚ :=
  window / cycle

theorem jack_sees_color_change :
  probability_of_observing_change traffic_light_cycle change_window = 3 / 19 :=
by
  -- We only need the statement for verification
  sorry

end jack_sees_color_change_l554_55482


namespace find_common_ratio_l554_55494

variable {a : ℕ → ℝ}
variable {q : ℝ}

noncomputable def geometric_sequence_q (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 2 + a 4 = 20 ∧ a 3 + a 5 = 40

theorem find_common_ratio (h : geometric_sequence_q a q) : q = 2 :=
by
  sorry

end find_common_ratio_l554_55494


namespace pencil_lead_loss_l554_55493

theorem pencil_lead_loss (L r : ℝ) (h : r = L * 1/10):
  ((9/10 * r^3) * (2/3)) / (r^3) = 3/5 := 
by
  sorry

end pencil_lead_loss_l554_55493


namespace sum_of_solutions_l554_55445

theorem sum_of_solutions (x : ℝ) (hx : x + 36 / x = 12) : x = 6 ∨ x = -6 := sorry

end sum_of_solutions_l554_55445


namespace percentage_discount_l554_55448

theorem percentage_discount (original_price sale_price : ℝ) (h1 : original_price = 25) (h2 : sale_price = 18.75) : 
  100 * (original_price - sale_price) / original_price = 25 := 
by
  -- Begin Proof
  sorry

end percentage_discount_l554_55448


namespace sequence_a4_l554_55413

theorem sequence_a4 :
  (∀ n : ℕ, n > 0 → ∀ (a : ℕ → ℝ),
    (a 1 = 1) →
    (∀ n > 0, a (n + 1) = (1 / 2) * a n + 1 / (2 ^ n)) →
    a 4 = 1 / 2) :=
by
  sorry

end sequence_a4_l554_55413


namespace cubic_difference_l554_55422

theorem cubic_difference (x : ℝ) (h : (x + 16) ^ (1/3) - (x - 16) ^ (1/3) = 4) : 
  235 < x^2 ∧ x^2 < 240 := 
sorry

end cubic_difference_l554_55422


namespace probability_correct_l554_55460

structure SockDrawSetup where
  total_socks : ℕ
  color_pairs : ℕ
  socks_per_color : ℕ
  draw_size : ℕ

noncomputable def probability_one_pair (S : SockDrawSetup) : ℚ :=
  let total_combinations := Nat.choose S.total_socks S.draw_size
  let favorable_combinations := (Nat.choose S.color_pairs 3) * (Nat.choose 3 1) * 2 * 2
  favorable_combinations / total_combinations

theorem probability_correct (S : SockDrawSetup) (h1 : S.total_socks = 12) (h2 : S.color_pairs = 6) (h3 : S.socks_per_color = 2) (h4 : S.draw_size = 6) :
  probability_one_pair S = 20 / 77 :=
by
  apply sorry

end probability_correct_l554_55460


namespace general_formula_arithmetic_sequence_l554_55479

theorem general_formula_arithmetic_sequence :
  (∃ (a_n : ℕ → ℕ) (d : ℕ), d ≠ 0 ∧ 
    (a_2 = a_1 + d) ∧ 
    (a_4 = a_1 + 3 * d) ∧ 
    (a_2^2 = a_1 * a_4) ∧
    (a_5 = a_1 + 4 * d) ∧ 
    (a_6 = a_1 + 5 * d) ∧ 
    (a_5 + a_6 = 11) ∧ 
    ∀ n, a_n = a_1 + (n - 1) * d) → 
  ∀ n, a_n = n := 
sorry

end general_formula_arithmetic_sequence_l554_55479


namespace yevgeniy_age_2014_l554_55404

theorem yevgeniy_age_2014 (birth_year : ℕ) (h1 : birth_year = 1900 + (birth_year % 100))
  (h2 : 2011 - birth_year = (birth_year / 1000) + ((birth_year % 1000) / 100) + ((birth_year % 100) / 10) + (birth_year % 10)) :
  2014 - birth_year = 23 :=
by
  sorry

end yevgeniy_age_2014_l554_55404


namespace at_least_one_equation_has_real_roots_l554_55481

noncomputable def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  (4 * b^2 - 4 * a * c > 0) ∨ (4 * c^2 - 4 * a * b > 0) ∨ (4 * a^2 - 4 * b * c > 0)

theorem at_least_one_equation_has_real_roots (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) :
  has_two_distinct_real_roots a b c :=
by
  sorry

end at_least_one_equation_has_real_roots_l554_55481


namespace true_propositions_identification_l554_55419

-- Definitions related to the propositions
def converse_prop1 (x y : ℝ) := (x + y = 0) → (x + y = 0)
-- Converse of additive inverses: If x and y are additive inverses, then x + y = 0
def converse_prop1_true (x y : ℝ) : Prop := (x + y = 0) → (x + y = 0)

def negation_prop2 : Prop := ¬(∀ (a b c d : ℝ), (a = b → c = d) → (a + b = c + d))
-- Negation of congruent triangles have equal areas: If two triangles are not congruent, areas not equal
def negation_prop2_false : Prop := ¬(∀ (a b c : ℝ), (a = b ∧ b ≠ c → a ≠ c))

def contrapositive_prop3 (q : ℝ) := (q ≤ 1) → (4 - 4 * q ≥ 0)
-- Contrapositive of real roots: If the equation x^2 + 2x + q = 0 does not have real roots then q > 1
def contrapositive_prop3_true (q : ℝ) : Prop := (4 - 4 * q < 0) → (q > 1)

def converse_prop4 (a b c : ℝ) := (a = b ∧ b = c ∧ c = a) → False
-- Converse of scalene triangle: If a triangle has three equal interior angles, it is a scalene triangle
def converse_prop4_false (a b c : ℝ) : Prop := (a = b ∧ b = c ∧ c = a) → False

theorem true_propositions_identification :
  (∀ x y : ℝ, converse_prop1_true x y) ∧
  ¬negation_prop2_false ∧
  (∀ q : ℝ, contrapositive_prop3_true q) ∧
  ¬(∀ a b c : ℝ, converse_prop4_false a b c) := by
  sorry

end true_propositions_identification_l554_55419


namespace find_square_number_divisible_by_six_l554_55408

theorem find_square_number_divisible_by_six :
  ∃ x : ℕ, (∃ k : ℕ, x = k^2) ∧ x % 6 = 0 ∧ 24 < x ∧ x < 150 ∧ (x = 36 ∨ x = 144) :=
by {
  sorry
}

end find_square_number_divisible_by_six_l554_55408


namespace sufficient_but_not_necessary_perpendicular_l554_55410

theorem sufficient_but_not_necessary_perpendicular (a : ℝ) :
  (∃ a' : ℝ, a' = -1 ∧ (a' = -1 → (0 : ℝ) ≠ 3 * a' - 1)) ∨
  (∃ a' : ℝ, a' ≠ -1 ∧ (a' ≠ -1 → (0 : ℝ) ≠ 3 * a' - 1)) →
  (3 * a' - 1) * (a' - 3) = -1 := sorry

end sufficient_but_not_necessary_perpendicular_l554_55410


namespace john_billed_for_28_minutes_l554_55465

variable (monthlyFee : ℝ) (costPerMinute : ℝ) (totalBill : ℝ)
variable (minutesBilled : ℝ)

def is_billed_correctly (monthlyFee totalBill costPerMinute minutesBilled : ℝ) : Prop :=
  totalBill - monthlyFee = minutesBilled * costPerMinute ∧ minutesBilled = 28

theorem john_billed_for_28_minutes : 
  is_billed_correctly 5 12.02 0.25 28 := 
by
  sorry

end john_billed_for_28_minutes_l554_55465


namespace seller_loss_l554_55416

/--
Given:
1. The buyer took goods worth 10 rubles (v_goods : Real := 10).
2. The buyer gave 25 rubles (payment : Real := 25).
3. The seller exchanged 25 rubles of genuine currency with the neighbor (exchange : Real := 25).
4. The seller received 25 rubles in counterfeit currency from the neighbor (counterfeit : Real := 25).
5. The seller gave 15 rubles in genuine currency as change (change : Real := 15).
6. The neighbor discovered the counterfeit and the seller returned 25 rubles to the neighbor (returned : Real := 25).

Prove that the net loss incurred by the seller is 30 rubles.
-/
theorem seller_loss :
  let v_goods := 10
  let payment := 25
  let exchange := 25
  let counterfeit := 25
  let change := 15
  let returned := 25
  (exchange + change) - v_goods = 30 :=
by
  sorry

end seller_loss_l554_55416


namespace prime_sum_product_l554_55490

theorem prime_sum_product (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_sum : p + q = 91) : p * q = 178 := 
by
  sorry

end prime_sum_product_l554_55490


namespace base_number_is_three_l554_55449

theorem base_number_is_three (some_number : ℝ) (y : ℕ) (h1 : 9^y = some_number^14) (h2 : y = 7) : some_number = 3 :=
by { sorry }

end base_number_is_three_l554_55449
