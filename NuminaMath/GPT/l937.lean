import Mathlib

namespace equal_real_roots_a_value_l937_93746

theorem equal_real_roots_a_value (a : ℝ) :
  a ≠ 0 →
  let b := -4
  let c := 3
  b * b - 4 * a * c = 0 →
  a = 4 / 3 :=
by
  intros h_nonzero h_discriminant
  sorry

end equal_real_roots_a_value_l937_93746


namespace amount_given_by_mom_l937_93713

def amount_spent_by_Mildred : ℕ := 25
def amount_spent_by_Candice : ℕ := 35
def amount_left : ℕ := 40

theorem amount_given_by_mom : 
  (amount_spent_by_Mildred + amount_spent_by_Candice + amount_left) = 100 := by
  sorry

end amount_given_by_mom_l937_93713


namespace shaded_area_of_joined_squares_l937_93705

theorem shaded_area_of_joined_squares:
  ∀ (a b : ℕ) (area_of_shaded : ℝ),
  (a = 6) → (b = 8) → 
  (area_of_shaded = (6 * 6 : ℝ) + (8 * 8 : ℝ) / 2) →
  area_of_shaded = 50.24 := 
by
  intros a b area_of_shaded h1 h2 h3
  -- skipping the proof for now
  sorry

end shaded_area_of_joined_squares_l937_93705


namespace lumber_cut_length_l937_93780

-- Define lengths of the pieces
def length_W : ℝ := 5
def length_X : ℝ := 3
def length_Y : ℝ := 5
def length_Z : ℝ := 4

-- Define distances from line M to the left end of the pieces
def distance_X : ℝ := 3
def distance_Y : ℝ := 2
def distance_Z : ℝ := 1.5

-- Define the total length of the pieces
def total_length : ℝ := 17

-- Define the length per side when cut by L
def length_per_side : ℝ := 8.5

theorem lumber_cut_length :
    (∃ (d : ℝ), 4 * d - 6.5 = 8.5 ∧ d = 3.75) :=
by
  sorry

end lumber_cut_length_l937_93780


namespace arithmetic_sequence_S10_l937_93761

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + n * d

def Sn (a d : ℤ) (n : ℕ) : ℤ :=
  n * a + (n * (n - 1)) / 2 * d

theorem arithmetic_sequence_S10 :
  ∃ (a d : ℤ), d ≠ 0 ∧ Sn a d 8 = 16 ∧
  (arithmetic_sequence a d 3)^2 = (arithmetic_sequence a d 2) * (arithmetic_sequence a d 6) ∧
  Sn a d 10 = 30 :=
by
  sorry

end arithmetic_sequence_S10_l937_93761


namespace abs_a_eq_5_and_a_add_b_eq_0_l937_93737

theorem abs_a_eq_5_and_a_add_b_eq_0 (a b : ℤ) (h1 : |a| = 5) (h2 : a + b = 0) :
  a - b = 10 ∨ a - b = -10 :=
by
  sorry

end abs_a_eq_5_and_a_add_b_eq_0_l937_93737


namespace small_monkey_dolls_cheaper_than_large_l937_93707

theorem small_monkey_dolls_cheaper_than_large (S : ℕ) 
  (h1 : 300 / 6 = 50) 
  (h2 : 300 / S = 75) 
  (h3 : 75 - 50 = 25) : 
  6 - S = 2 := 
sorry

end small_monkey_dolls_cheaper_than_large_l937_93707


namespace present_worth_proof_l937_93756

-- Define the conditions
def banker's_gain (BG : ℝ) : Prop := BG = 16
def true_discount (TD : ℝ) : Prop := TD = 96

-- Define the relationship from the problem
def relationship (BG TD PW : ℝ) : Prop := BG = TD - PW

-- Define the present worth of the sum
def present_worth : ℝ := 80

-- Theorem stating that the present worth of the sum is Rs. 80 given the conditions
theorem present_worth_proof (BG TD PW : ℝ)
  (hBG : banker's_gain BG)
  (hTD : true_discount TD)
  (hRelation : relationship BG TD PW) :
  PW = present_worth := by
  sorry

end present_worth_proof_l937_93756


namespace div_remainder_l937_93732

theorem div_remainder (x : ℕ) (h : x = 2^40) : 
  (2^160 + 160) % (2^80 + 2^40 + 1) = 159 :=
by
  sorry

end div_remainder_l937_93732


namespace marcy_drinks_in_250_minutes_l937_93760

-- Define a function to represent that Marcy takes n minutes to drink x liters of water.
def time_to_drink (minutes_per_sip : ℕ) (sip_volume_ml : ℕ) (total_volume_liters : ℕ) : ℕ :=
  let total_volume_ml := total_volume_liters * 1000
  let sips := total_volume_ml / sip_volume_ml
  sips * minutes_per_sip

theorem marcy_drinks_in_250_minutes :
  time_to_drink 5 40 2 = 250 :=
  by
    -- The function definition and its application will show this value holds.
    sorry

end marcy_drinks_in_250_minutes_l937_93760


namespace arithmetic_sequence_sum_l937_93762

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℚ) (T : ℕ → ℚ) 
  (h1 : a 3 = 7) (h2 : a 5 + a 7 = 26) :
  (∀ n, a n = 2 * n + 1) ∧
  (∀ n, S n = n^2 + 2 * n) ∧
  (∀ n, b n = 1 / ((2 * n + 1)^2 - 1)) ∧
  (∀ n, T n = n / (4 * (n + 1))) :=
by
  sorry

end arithmetic_sequence_sum_l937_93762


namespace correct_option_D_l937_93781

variables {a b m : Type}
variables {α β : Type}

axiom parallel (x y : Type) : Prop
axiom perpendicular (x y : Type) : Prop

variables (a_parallel_b : parallel a b)
variables (a_parallel_alpha : parallel a α)

variables (alpha_perpendicular_beta : perpendicular α β)
variables (a_parallel_alpha : parallel a α)

variables (alpha_parallel_beta : parallel α β)
variables (m_perpendicular_alpha : perpendicular m α)

theorem correct_option_D : parallel α β ∧ perpendicular m α → perpendicular m β := sorry

end correct_option_D_l937_93781


namespace replace_movie_cost_l937_93718

def num_popular_action_movies := 20
def num_moderate_comedy_movies := 30
def num_unpopular_drama_movies := 10
def num_popular_comedy_movies := 15
def num_moderate_action_movies := 25

def trade_in_rate_action := 3
def trade_in_rate_comedy := 2
def trade_in_rate_drama := 1

def dvd_cost_popular := 12
def dvd_cost_moderate := 8
def dvd_cost_unpopular := 5

def johns_movie_cost : Nat :=
  let total_trade_in := 
    (num_popular_action_movies + num_moderate_action_movies) * trade_in_rate_action +
    (num_moderate_comedy_movies + num_popular_comedy_movies) * trade_in_rate_comedy +
    num_unpopular_drama_movies * trade_in_rate_drama
  let total_dvd_cost :=
    (num_popular_action_movies + num_popular_comedy_movies) * dvd_cost_popular +
    (num_moderate_comedy_movies + num_moderate_action_movies) * dvd_cost_moderate +
    num_unpopular_drama_movies * dvd_cost_unpopular
  total_dvd_cost - total_trade_in

theorem replace_movie_cost : johns_movie_cost = 675 := 
by
  sorry

end replace_movie_cost_l937_93718


namespace smallest_prime_that_is_6_more_than_perfect_square_and_9_less_than_next_perfect_square_l937_93771

theorem smallest_prime_that_is_6_more_than_perfect_square_and_9_less_than_next_perfect_square :
  ∃ p : ℕ, Prime p ∧ (∃ k m : ℤ, k^2 = p - 6 ∧ m^2 = p + 9 ∧ m^2 - k^2 = 15) ∧ p = 127 :=
sorry

end smallest_prime_that_is_6_more_than_perfect_square_and_9_less_than_next_perfect_square_l937_93771


namespace value_of_sum_l937_93738

theorem value_of_sum (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (hc_solution : c^2 + a * c + b = 0) (hd_solution : d^2 + a * d + b = 0)
  (ha_solution : a^2 + c * a + d = 0) (hb_solution : b^2 + c * b + d = 0)
: a + b + c + d = -2 := sorry -- The proof is omitted as requested

end value_of_sum_l937_93738


namespace solve_quad_linear_system_l937_93791

theorem solve_quad_linear_system :
  (∃ x y : ℝ, x^2 - 6 * x + 8 = 0 ∧ y + 2 * x = 12 ∧ ((x, y) = (4, 4) ∨ (x, y) = (2, 8))) :=
sorry

end solve_quad_linear_system_l937_93791


namespace number_of_ways_to_choose_one_class_number_of_ways_to_choose_one_class_each_grade_number_of_ways_to_choose_two_classes_different_grades_l937_93723

-- Define the number of classes in each grade.
def num_classes_first_year : ℕ := 14
def num_classes_second_year : ℕ := 14
def num_classes_third_year : ℕ := 15

-- Prove the number of different ways to choose students from 1 class.
theorem number_of_ways_to_choose_one_class :
  (num_classes_first_year + num_classes_second_year + num_classes_third_year) = 43 := 
by {
  -- Numerical calculation
  sorry
}

-- Prove the number of different ways to choose students from one class in each grade.
theorem number_of_ways_to_choose_one_class_each_grade :
  (num_classes_first_year * num_classes_second_year * num_classes_third_year) = 2940 := 
by {
  -- Numerical calculation
  sorry
}

-- Prove the number of different ways to choose students from 2 classes from different grades.
theorem number_of_ways_to_choose_two_classes_different_grades :
  (num_classes_first_year * num_classes_second_year + num_classes_first_year * num_classes_third_year + num_classes_second_year * num_classes_third_year) = 616 := 
by {
  -- Numerical calculation
  sorry
}

end number_of_ways_to_choose_one_class_number_of_ways_to_choose_one_class_each_grade_number_of_ways_to_choose_two_classes_different_grades_l937_93723


namespace max_number_of_squares_with_twelve_points_l937_93764

-- Define the condition: twelve marked points in a grid
def twelve_points_marked_on_grid : Prop := 
  -- Assuming twelve specific points represented in a grid-like structure
  -- (This will be defined concretely in the proof implementation context)
  sorry

-- Define the problem statement to be proved
theorem max_number_of_squares_with_twelve_points : 
  twelve_points_marked_on_grid → (∃ n, n = 11) :=
by 
  sorry

end max_number_of_squares_with_twelve_points_l937_93764


namespace solve_for_k_l937_93739

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n / 2

theorem solve_for_k (k : ℤ) (h1 : k % 2 = 1) (h2 : f (f (f k)) = 57) : k = 223 :=
by
  -- Proof will be provided here
  sorry

end solve_for_k_l937_93739


namespace largest_triangle_perimeter_l937_93741

theorem largest_triangle_perimeter :
  ∀ (x : ℕ), 1 < x ∧ x < 15 → (7 + 8 + x = 29) :=
by
  intro x
  intro h
  sorry

end largest_triangle_perimeter_l937_93741


namespace simplified_expression_l937_93709

variable (x y : ℝ)

theorem simplified_expression (hx : x ≠ 0) (hy : y ≠ 0) :
  (3 / 5) * Real.sqrt (x * y^2) / ((-4 / 15) * Real.sqrt (y / x)) * ((-5 / 6) * Real.sqrt (x^3 * y)) =
  (15 * x^2 * y * Real.sqrt x) / 8 :=
by
  sorry

end simplified_expression_l937_93709


namespace find_two_digit_numbers_l937_93750

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem find_two_digit_numbers :
  ∀ (A : ℕ), (10 ≤ A ∧ A ≤ 99) →
    (sum_of_digits A)^2 = sum_of_digits (A^2) →
    (A = 11 ∨ A = 12 ∨ A = 13 ∨ A = 20 ∨ A = 21 ∨ A = 22 ∨ A = 30 ∨ A = 31 ∨ A = 50) :=
by sorry

end find_two_digit_numbers_l937_93750


namespace quantiville_jacket_junction_l937_93766

theorem quantiville_jacket_junction :
  let sales_tax_rate := 0.07
  let original_price := 120.0
  let discount := 0.25
  let amy_total := (original_price * (1 + sales_tax_rate)) * (1 - discount)
  let bob_total := (original_price * (1 - discount)) * (1 + sales_tax_rate)
  let carla_total := ((original_price * (1 + sales_tax_rate)) * (1 - discount)) * (1 + sales_tax_rate)
  (carla_total - amy_total) = 6.744 :=
by
  sorry

end quantiville_jacket_junction_l937_93766


namespace largest_number_of_stamps_per_page_l937_93731

theorem largest_number_of_stamps_per_page :
  Nat.gcd (Nat.gcd 1200 1800) 2400 = 600 :=
sorry

end largest_number_of_stamps_per_page_l937_93731


namespace smallest_n_13n_congruent_456_mod_5_l937_93792

theorem smallest_n_13n_congruent_456_mod_5 : ∃ n : ℕ, (n > 0) ∧ (13 * n ≡ 456 [MOD 5]) ∧ (∀ m : ℕ, (m > 0 ∧ 13 * m ≡ 456 [MOD 5]) → n ≤ m) :=
by
  sorry

end smallest_n_13n_congruent_456_mod_5_l937_93792


namespace cells_at_end_of_8th_day_l937_93708

theorem cells_at_end_of_8th_day :
  let initial_cells := 5
  let factor := 3
  let toxin_factor := 1 / 2
  let cells_after_toxin := (initial_cells * factor * factor * factor * toxin_factor : ℤ)
  let final_cells := cells_after_toxin * factor 
  final_cells = 201 :=
by
  sorry

end cells_at_end_of_8th_day_l937_93708


namespace cost_of_notebook_l937_93735

theorem cost_of_notebook (s n c : ℕ) 
    (h1 : s > 18) 
    (h2 : n ≥ 2) 
    (h3 : c > n) 
    (h4 : s * c * n = 2376) : 
    c = 11 := 
  sorry

end cost_of_notebook_l937_93735


namespace circle_line_intersection_points_l937_93700

noncomputable def radius : ℝ := 6
noncomputable def distance : ℝ := 5

theorem circle_line_intersection_points :
  radius > distance -> number_of_intersection_points = 2 := 
by
  sorry

end circle_line_intersection_points_l937_93700


namespace sam_drove_distance_l937_93788

theorem sam_drove_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) :
  marguerite_distance = 150 ∧ marguerite_time = 3 ∧ sam_time = 4 →
  (sam_time * (marguerite_distance / marguerite_time) = 200) :=
by
  sorry

end sam_drove_distance_l937_93788


namespace not_prime_4k4_plus_1_not_prime_k4_plus_4_l937_93789

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem not_prime_4k4_plus_1 (k : ℕ) (hk : k > 0) : ¬ is_prime (4 * k^4 + 1) :=
by sorry

theorem not_prime_k4_plus_4 (k : ℕ) (hk : k > 0) : ¬ is_prime (k^4 + 4) :=
by sorry

end not_prime_4k4_plus_1_not_prime_k4_plus_4_l937_93789


namespace students_per_group_l937_93795

def total_students : ℕ := 30
def number_of_groups : ℕ := 6

theorem students_per_group :
  total_students / number_of_groups = 5 :=
by
  sorry

end students_per_group_l937_93795


namespace percentage_saved_l937_93749

theorem percentage_saved (rent milk groceries education petrol misc savings : ℝ) 
  (salary : ℝ) 
  (h_rent : rent = 5000) 
  (h_milk : milk = 1500) 
  (h_groceries : groceries = 4500) 
  (h_education : education = 2500) 
  (h_petrol : petrol = 2000) 
  (h_misc : misc = 700) 
  (h_savings : savings = 1800) 
  (h_salary : salary = rent + milk + groceries + education + petrol + misc + savings) : 
  (savings / salary) * 100 = 10 :=
by
  sorry

end percentage_saved_l937_93749


namespace children_in_school_l937_93701

theorem children_in_school (C B : ℕ) (h1 : B = 2 * C) (h2 : B = 4 * (C - 370)) : C = 740 :=
by
  sorry

end children_in_school_l937_93701


namespace alien_home_planet_people_count_l937_93796

noncomputable def alien_earth_abduction (total_abducted returned_percentage taken_to_other_planet : ℕ) : ℕ :=
  let returned := total_abducted * returned_percentage / 100
  let remaining := total_abducted - returned
  remaining - taken_to_other_planet

theorem alien_home_planet_people_count :
  alien_earth_abduction 200 80 10 = 30 :=
by
  sorry

end alien_home_planet_people_count_l937_93796


namespace factor_difference_of_squares_l937_93747

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) := by
  sorry

end factor_difference_of_squares_l937_93747


namespace tangent_half_angle_sum_eq_product_l937_93736

variable {α β γ : ℝ}

theorem tangent_half_angle_sum_eq_product (h : α + β + γ = 2 * Real.pi) :
  Real.tan (α / 2) + Real.tan (β / 2) + Real.tan (γ / 2) =
  Real.tan (α / 2) * Real.tan (β / 2) * Real.tan (γ / 2) :=
sorry

end tangent_half_angle_sum_eq_product_l937_93736


namespace ratio_of_first_to_second_l937_93727

theorem ratio_of_first_to_second (x y : ℕ) 
  (h1 : x + y + (1 / 3 : ℚ) * x = 110)
  (h2 : y = 30) :
  x / y = 2 :=
by
  sorry

end ratio_of_first_to_second_l937_93727


namespace find_positive_integer_l937_93716

theorem find_positive_integer (n : ℕ) (hn_pos : n > 0) :
  (∃ a b : ℕ, n = a^2 ∧ n + 100 = b^2) → n = 576 :=
by sorry

end find_positive_integer_l937_93716


namespace right_triangles_sides_l937_93786

theorem right_triangles_sides (a b c p S r DH FC FH: ℝ)
  (h₁ : a = 10)
  (h₂ : b = 10)
  (h₃ : c = 12)
  (h₄ : p = (a + b + c) / 2)
  (h₅ : S = Real.sqrt (p * (p - a) * (p - b) * (p - c)))
  (h₆ : r = S / p)
  (h₇ : DH = (c / 2) - r)
  (h₈ : FC = (a * r) / DH)
  (h₉ : FH = Real.sqrt (FC^2 - DH^2))
: FC = 3 ∧ DH = 4 ∧ FH = 5 := by
  sorry

end right_triangles_sides_l937_93786


namespace lunks_for_apples_l937_93755

theorem lunks_for_apples : 
  (∀ (a : ℕ) (b : ℕ) (k : ℕ), 3 * b * k = 5 * a → 15 * k = 9 * a ∧ 2 * a * 9 = 4 * b * 9 → 15 * 2 * a / 4 = 18) :=
by
  intro a b k h1 h2
  sorry

end lunks_for_apples_l937_93755


namespace geometric_sequence_sixth_term_l937_93703

theorem geometric_sequence_sixth_term:
  ∃ q : ℝ, 
  ∀ (a₁ a₈ a₆ : ℝ), 
    a₁ = 6 ∧ a₈ = 768 ∧ a₈ = a₁ * q^7 ∧ a₆ = a₁ * q^5 
    → a₆ = 192 :=
by
  sorry

end geometric_sequence_sixth_term_l937_93703


namespace find_number_l937_93774

theorem find_number (x : ℝ) (h : 61 + 5 * 12 / (180 / x) = 62): x = 3 :=
by
  sorry

end find_number_l937_93774


namespace number_of_ferns_is_six_l937_93740

def num_fronds_per_fern : Nat := 7
def num_leaves_per_frond : Nat := 30
def total_leaves : Nat := 1260

theorem number_of_ferns_is_six :
  total_leaves = num_fronds_per_fern * num_leaves_per_frond * 6 :=
by
  sorry

end number_of_ferns_is_six_l937_93740


namespace salt_quantity_l937_93714

-- Conditions translated to Lean definitions
def cost_of_sugar_per_kg : ℝ := 1.50
def total_cost_sugar_2kg_and_salt (x : ℝ) : ℝ := 5.50
def total_cost_sugar_3kg_and_1kg_salt : ℝ := 5.00

-- Theorem statement
theorem salt_quantity (x : ℝ) : 
  2 * cost_of_sugar_per_kg + x * cost_of_sugar_per_kg / 3 = total_cost_sugar_2kg_and_salt x 
  → 3 * cost_of_sugar_per_kg + x = total_cost_sugar_3kg_and_1kg_salt 
  → x = 5 := 
sorry

end salt_quantity_l937_93714


namespace sum_of_coefficients_l937_93734

theorem sum_of_coefficients (A B C : ℤ)
  (h : ∀ x, x^3 + A * x^2 + B * x + C = (x + 3) * x * (x - 3))
  : A + B + C = -9 :=
sorry

end sum_of_coefficients_l937_93734


namespace determine_m_range_l937_93777

theorem determine_m_range (m : ℝ) (h : (∃ (x y : ℝ), x^2 + y^2 + 2 * m * x + 2 = 0) ∧ 
                                    (∃ (r : ℝ) (h_r : r^2 = m^2 - 2), π * r^2 ≥ 4 * π)) :
  (m ≤ -Real.sqrt 6 ∨ m ≥ Real.sqrt 6) :=
by
  sorry

end determine_m_range_l937_93777


namespace angle_between_AB_CD_l937_93717

def point := (ℝ × ℝ × ℝ)

def A : point := (-3, 0, 1)
def B : point := (2, 1, -1)
def C : point := (-2, 2, 0)
def D : point := (1, 3, 2)

noncomputable def angle_between_lines (p1 p2 p3 p4 : point) : ℝ := sorry

theorem angle_between_AB_CD :
  angle_between_lines A B C D = Real.arccos (2 * Real.sqrt 105 / 35) :=
sorry

end angle_between_AB_CD_l937_93717


namespace men_seated_l937_93706

theorem men_seated (total_passengers : ℕ) (women_ratio : ℚ) (children_count : ℕ) (men_standing_ratio : ℚ) 
  (women_with_prams : ℕ) (disabled_passengers : ℕ) 
  (h_total_passengers : total_passengers = 48) 
  (h_women_ratio : women_ratio = 2 / 3) 
  (h_children_count : children_count = 5) 
  (h_men_standing_ratio : men_standing_ratio = 1 / 8) 
  (h_women_with_prams : women_with_prams = 3) 
  (h_disabled_passengers : disabled_passengers = 2) : 
  (total_passengers * (1 - women_ratio) - total_passengers * (1 - women_ratio) * men_standing_ratio = 14) :=
by sorry

end men_seated_l937_93706


namespace no_solutions_l937_93775

theorem no_solutions
  (x y z : ℤ)
  (h : x^2 + y^2 = 4 * z - 1) : False :=
sorry

end no_solutions_l937_93775


namespace ann_trip_longer_than_mary_l937_93784

-- Define constants for conditions
def mary_hill_length : ℕ := 630
def mary_speed : ℕ := 90
def ann_hill_length : ℕ := 800
def ann_speed : ℕ := 40

-- Define a theorem to express the question and correct answer
theorem ann_trip_longer_than_mary : 
  (ann_hill_length / ann_speed - mary_hill_length / mary_speed) = 13 :=
by
  -- Now insert sorry to leave the proof unfinished
  sorry

end ann_trip_longer_than_mary_l937_93784


namespace speed_of_current_l937_93743

theorem speed_of_current (v : ℝ) : 
  (∀ s, s = 3 → s / (3 - v) = 2.3076923076923075) → v = 1.7 := 
by
  intro h
  sorry

end speed_of_current_l937_93743


namespace first_year_payment_l937_93712

theorem first_year_payment (X : ℝ) (second_year : ℝ) (third_year : ℝ) (fourth_year : ℝ) 
    (total_payments : ℝ) 
    (h1 : second_year = X + 2)
    (h2 : third_year = X + 5)
    (h3 : fourth_year = X + 9)
    (h4 : total_payments = X + second_year + third_year + fourth_year) :
    total_payments = 96 → X = 20 :=
by
    sorry

end first_year_payment_l937_93712


namespace triangle_inequality_squares_l937_93704

theorem triangle_inequality_squares (a b c : ℝ) (h₁ : a < b + c) (h₂ : b < a + c) (h₃ : c < a + b) :
  a^2 + b^2 + c^2 < 2 * (a * b + b * c + a * c) :=
sorry

end triangle_inequality_squares_l937_93704


namespace num_terms_arithmetic_seq_l937_93757

theorem num_terms_arithmetic_seq (a d l : ℝ) (n : ℕ)
  (h1 : a = 3.25) 
  (h2 : d = 4)
  (h3 : l = 55.25)
  (h4 : l = a + (↑n - 1) * d) :
  n = 14 :=
by
  sorry

end num_terms_arithmetic_seq_l937_93757


namespace water_purification_problem_l937_93702

variable (x : ℝ) (h : x > 0)

theorem water_purification_problem
  (h1 : ∀ (p : ℝ), p = 2400)
  (h2 : ∀ (eff : ℝ), eff = 1.2)
  (h3 : ∀ (time_saved : ℝ), time_saved = 40) :
  (2400 * 1.2 / x) - (2400 / x) = 40 := by
  sorry

end water_purification_problem_l937_93702


namespace smaller_circle_radius_l937_93751

theorem smaller_circle_radius (r R : ℝ) (A1 A2 : ℝ) (hR : R = 5.0) (hA : A1 + A2 = 25 * Real.pi)
  (hap : A2 = A1 + 25 * Real.pi / 2) : r = 5 * Real.sqrt 2 / 2 :=
by
  -- Placeholder for the actual proof
  sorry

end smaller_circle_radius_l937_93751


namespace hyperbola_focus_distance_l937_93772

theorem hyperbola_focus_distance :
  ∀ (x y : ℝ), (x^2 / 4 - y^2 / 3 = 1) → ∀ (F₁ F₂ : ℝ × ℝ), ∃ P : ℝ × ℝ, dist P F₁ = 3 → dist P F₂ = 7 :=
by
  sorry

end hyperbola_focus_distance_l937_93772


namespace shares_of_valuable_stock_l937_93785

theorem shares_of_valuable_stock 
  (price_val : ℕ := 78)
  (price_oth : ℕ := 39)
  (shares_oth : ℕ := 26)
  (total_asset : ℕ := 2106)
  (x : ℕ) 
  (h_val_stock : total_asset = 78 * x + 39 * 26) : 
  x = 14 :=
by
  sorry

end shares_of_valuable_stock_l937_93785


namespace triangle_inequality_proof_l937_93765

theorem triangle_inequality_proof (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 :=
sorry

end triangle_inequality_proof_l937_93765


namespace candle_ratio_proof_l937_93745

noncomputable def candle_height_ratio := 
  ∃ (x y : ℝ), 
    (x / 6) * 3 = x / 2 ∧
    (y / 8) * 3 = 3 * y / 8 ∧
    (x / 2) = (5 * y / 8) →
    x / y = 5 / 4

theorem candle_ratio_proof : candle_height_ratio :=
by sorry

end candle_ratio_proof_l937_93745


namespace det_matrixB_eq_neg_one_l937_93720

variable (x y : ℝ)

def matrixB : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![x, 3],
  ![-4, y]
]

theorem det_matrixB_eq_neg_one 
  (h : matrixB x y - (matrixB x y)⁻¹ = 2 • (1 : Matrix (Fin 2) (Fin 2) ℝ)) :
  Matrix.det (matrixB x y) = -1 := sorry

end det_matrixB_eq_neg_one_l937_93720


namespace A_det_nonzero_A_inv_is_correct_l937_93742

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 4], ![2, 9]]

def A_inv : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![9, -4], ![-2, 1]]

theorem A_det_nonzero : det A ≠ 0 := 
  sorry

theorem A_inv_is_correct : A * A_inv = 1 := 
  sorry

end A_det_nonzero_A_inv_is_correct_l937_93742


namespace polygon_perimeter_l937_93744

theorem polygon_perimeter (side_length : ℝ) (ext_angle_deg : ℝ) (n : ℕ) (h1 : side_length = 8) 
  (h2 : ext_angle_deg = 90) (h3 : ext_angle_deg = 360 / n) : 
  4 * side_length = 32 := 
  by 
    sorry

end polygon_perimeter_l937_93744


namespace cost_price_represents_articles_l937_93797

theorem cost_price_represents_articles (C S : ℝ) (N : ℕ)
  (h1 : N * C = 16 * S)
  (h2 : S = C * 1.125) :
  N = 18 :=
by
  sorry

end cost_price_represents_articles_l937_93797


namespace exists_sequence_for_k_l937_93763

variable (n : ℕ) (k : ℕ)

noncomputable def exists_sequence (n k : ℕ) : Prop :=
  ∃ (x : ℕ → ℕ), ∀ i : ℕ, i < n → x i < x (i + 1)

theorem exists_sequence_for_k (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) :
  exists_sequence n k :=
  sorry

end exists_sequence_for_k_l937_93763


namespace a_alone_time_to_complete_work_l937_93748

theorem a_alone_time_to_complete_work :
  (W : ℝ) →
  (A : ℝ) →
  (B : ℝ) →
  (h1 : A + B = W / 6) →
  (h2 : B = W / 12) →
  A = W / 12 :=
by
  -- Given conditions
  intros W A B h1 h2
  -- Proof is not needed as per instructions
  sorry

end a_alone_time_to_complete_work_l937_93748


namespace at_most_one_true_l937_93770

theorem at_most_one_true (p q : Prop) (h : ¬(p ∧ q)) : ¬(p ∧ q ∧ ¬(¬p ∧ ¬q)) :=
by
  sorry

end at_most_one_true_l937_93770


namespace g_value_at_2_over_9_l937_93783

theorem g_value_at_2_over_9 (g : ℝ → ℝ) 
  (hg0 : g 0 = 0)
  (hgmono : ∀ ⦃x y⦄, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y)
  (hg_symm : ∀ x, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x)
  (hg_frac : ∀ x, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3) :
  g (2 / 9) = 8 / 27 :=
sorry

end g_value_at_2_over_9_l937_93783


namespace quadratic_inequality_solution_set_l937_93730

theorem quadratic_inequality_solution_set :
  {x : ℝ | x * (x - 2) > 0} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 2} :=
by
  sorry

end quadratic_inequality_solution_set_l937_93730


namespace opposite_of_neg_half_l937_93752

-- Define the opposite of a number
def opposite (x : ℝ) : ℝ := -x

-- The theorem we want to prove
theorem opposite_of_neg_half : opposite (-1/2) = 1/2 :=
by
  -- Proof goes here
  sorry

end opposite_of_neg_half_l937_93752


namespace larger_segment_of_triangle_l937_93754

theorem larger_segment_of_triangle (a b c : ℝ) (h : ℝ) (hc : c = 100) (ha : a = 40) (hb : b = 90) 
  (h_triangle : a^2 + h^2 = x^2)
  (h_triangle2 : b^2 + h^2 = (100 - x)^2) :
  100 - x = 82.5 :=
sorry

end larger_segment_of_triangle_l937_93754


namespace range_of_a_l937_93776

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + (1 - a) * x + 1 < 0) → a < -1 ∨ a > 3 :=
by
  sorry

end range_of_a_l937_93776


namespace fg_of_1_eq_15_l937_93721

def f (x : ℝ) := 2 * x - 3
def g (x : ℝ) := (x + 2) ^ 2

theorem fg_of_1_eq_15 : f (g 1) = 15 :=
by
  sorry

end fg_of_1_eq_15_l937_93721


namespace jimmy_change_l937_93753

noncomputable def change_back (pen_cost notebook_cost folder_cost highlighter_cost sticky_notes_cost total_paid discount tax : ℝ) : ℝ :=
  let total_before_discount := (5 * pen_cost) + (6 * notebook_cost) + (4 * folder_cost) + (3 * highlighter_cost) + (2 * sticky_notes_cost)
  let total_after_discount := total_before_discount * (1 - discount)
  let final_total := total_after_discount * (1 + tax)
  (total_paid - final_total)

theorem jimmy_change :
  change_back 1.65 3.95 4.35 2.80 1.75 150 0.25 0.085 = 100.16 :=
by
  sorry

end jimmy_change_l937_93753


namespace largest_multiple_of_7_less_than_neg50_l937_93799

theorem largest_multiple_of_7_less_than_neg50 : ∃ x, (∃ k : ℤ, x = 7 * k) ∧ x < -50 ∧ ∀ y, (∃ m : ℤ, y = 7 * m) → y < -50 → y ≤ x :=
sorry

end largest_multiple_of_7_less_than_neg50_l937_93799


namespace expression_value_l937_93787

theorem expression_value : (5^2 - 5) * (6^2 - 6) - (7^2 - 7) = 558 := by
  sorry

end expression_value_l937_93787


namespace existence_of_intersection_l937_93773

def setA (m : ℝ) : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ (x^2 + m * x - y + 2 = 0) }
def setB : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ (x - y + 1 = 0) ∧ (0 ≤ x ∧ x ≤ 2) }

theorem existence_of_intersection (m : ℝ) : (∃ (p : ℝ × ℝ), p ∈ (setA m ∩ setB)) ↔ m ≤ -1 := 
sorry

end existence_of_intersection_l937_93773


namespace least_number_of_square_tiles_l937_93798

theorem least_number_of_square_tiles
  (length_cm : ℕ) (width_cm : ℕ)
  (h1 : length_cm = 816) (h2 : width_cm = 432) :
  ∃ tile_count : ℕ, tile_count = 153 :=
by
  sorry

end least_number_of_square_tiles_l937_93798


namespace swimming_speed_in_still_water_l937_93719

theorem swimming_speed_in_still_water :
  ∀ (speed_of_water person's_speed time distance: ℝ),
  speed_of_water = 8 →
  time = 1.5 →
  distance = 12 →
  person's_speed - speed_of_water = distance / time →
  person's_speed = 16 :=
by
  intro speed_of_water person's_speed time distance hw ht hd heff
  rw [hw, ht, hd] at heff
  -- steps to isolate person's_speed should be done here, but we leave it as sorry
  sorry

end swimming_speed_in_still_water_l937_93719


namespace sara_change_l937_93782

def cost_of_first_book : ℝ := 5.5
def cost_of_second_book : ℝ := 6.5
def amount_given : ℝ := 20.0
def total_cost : ℝ := cost_of_first_book + cost_of_second_book
def change : ℝ := amount_given - total_cost

theorem sara_change : change = 8 :=
by
  have total_cost_correct : total_cost = 12.0 := by sorry
  have change_correct : change = amount_given - total_cost := by sorry
  show change = 8
  sorry

end sara_change_l937_93782


namespace triangle_sides_inequality_l937_93769

theorem triangle_sides_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^2 - 2 * a * b + b^2 - c^2 < 0 :=
by
  sorry

end triangle_sides_inequality_l937_93769


namespace correct_equation_l937_93724

-- Define the necessary conditions and parameters
variables (x : ℝ)

-- Length of the rectangle
def length := x 

-- Width is 6 meters less than the length
def width := x - 6

-- The area of the rectangle
def area := 720

-- Proof statement
theorem correct_equation : 
  x * (x - 6) = 720 :=
sorry

end correct_equation_l937_93724


namespace geom_seq_sum_l937_93715

theorem geom_seq_sum (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 3)
  (h2 : a 1 + a 2 + a 3 = 21)
  (h3 : ∀ n, a (n + 1) = a n * q) : a 4 + a 5 + a 6 = 168 :=
sorry

end geom_seq_sum_l937_93715


namespace perimeter_of_ABFCDE_l937_93790

-- Define the problem parameters
def square_perimeter : ℤ := 60
def side_length (p : ℤ) : ℤ := p / 4
def equilateral_triangle_side (l : ℤ) : ℤ := l
def new_shape_sides : ℕ := 6
def new_perimeter (s : ℤ) : ℤ := new_shape_sides * s

-- Define the theorem to be proved
theorem perimeter_of_ABFCDE (p : ℤ) (s : ℕ) (len : ℤ) : len = side_length p → len = equilateral_triangle_side len →
  new_perimeter len = 90 :=
by
  intros h1 h2
  sorry

end perimeter_of_ABFCDE_l937_93790


namespace value_of_expression_l937_93779

theorem value_of_expression (x y : ℝ) (h1 : x = 12) (h2 : y = 18) : 3 * (x - y) * (x + y) = -540 :=
by
  rw [h1, h2]
  sorry

end value_of_expression_l937_93779


namespace find_f_21_l937_93710

def f : ℝ → ℝ := sorry

lemma f_condition (x : ℝ) : f (2 / x + 1) = Real.log x := sorry

theorem find_f_21 : f 21 = -1 := sorry

end find_f_21_l937_93710


namespace composite_10201_in_all_bases_greater_than_two_composite_10101_in_all_bases_l937_93768

-- Definition for part (a)
def composite_base_greater_than_two (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ (n^4 + 2*n^2 + 1) = a * b

-- Proof statement for part (a)
theorem composite_10201_in_all_bases_greater_than_two (n : ℕ) (h : n > 2) : composite_base_greater_than_two n :=
by sorry

-- Definition for part (b)
def composite_in_all_bases (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ (n^4 + n^2 + 1) = a * b

-- Proof statement for part (b)
theorem composite_10101_in_all_bases (n : ℕ) : composite_in_all_bases n :=
by sorry

end composite_10201_in_all_bases_greater_than_two_composite_10101_in_all_bases_l937_93768


namespace eds_weight_l937_93793

variable (Al Ben Carl Ed : ℕ)

def weight_conditions : Prop :=
  Carl = 175 ∧ Ben = Carl - 16 ∧ Al = Ben + 25 ∧ Ed = Al - 38

theorem eds_weight (h : weight_conditions Al Ben Carl Ed) : Ed = 146 :=
by
  -- Conditions
  have h1 : Carl = 175    := h.1
  have h2 : Ben = Carl - 16 := h.2.1
  have h3 : Al = Ben + 25   := h.2.2.1
  have h4 : Ed = Al - 38    := h.2.2.2
  -- Proof itself is omitted, sorry placeholder
  sorry

end eds_weight_l937_93793


namespace transformation_correct_l937_93726

theorem transformation_correct (a b c : ℝ) (h : a / c = b / c) (hc : c ≠ 0) : a = b :=
sorry

end transformation_correct_l937_93726


namespace stools_count_l937_93767

theorem stools_count : ∃ x y : ℕ, 3 * x + 4 * y = 39 ∧ x = 3 := 
by
  sorry

end stools_count_l937_93767


namespace find_number_l937_93759

theorem find_number (x : ℝ) (h : 100 - x = x + 40) : x = 30 :=
sorry

end find_number_l937_93759


namespace existence_of_rational_solutions_a_nonexistence_of_rational_solutions_b_l937_93725

def is_rational_non_integer (a : ℚ) : Prop :=
  ¬ (∃ n : ℤ, a = n)

theorem existence_of_rational_solutions_a :
  ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧
    ∃ k1 k2 : ℤ, 19 * x + 8 * y = k1 ∧ 8 * x + 3 * y = k2 :=
sorry

theorem nonexistence_of_rational_solutions_b :
  ¬ (∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧
    ∃ k1 k2 : ℤ, 19 * x^2 + 8 * y^2 = k1 ∧ 8 * x^2 + 3 * y^2 = k2) :=
sorry

end existence_of_rational_solutions_a_nonexistence_of_rational_solutions_b_l937_93725


namespace opposite_of_neg3_is_3_l937_93758

theorem opposite_of_neg3_is_3 : -(-3) = 3 := by
  sorry

end opposite_of_neg3_is_3_l937_93758


namespace line_passes_through_fixed_point_l937_93728

theorem line_passes_through_fixed_point (a b c : ℝ) (h : a - b + c = 0) : a * 1 + b * (-1) + c = 0 := 
by sorry

end line_passes_through_fixed_point_l937_93728


namespace shorter_base_of_isosceles_trapezoid_l937_93733

theorem shorter_base_of_isosceles_trapezoid
  (a b : ℝ)
  (h : a > b)
  (h_division : (a + b) / 2 = (a - b) / 2 + 10) :
  b = 10 :=
by
  sorry

end shorter_base_of_isosceles_trapezoid_l937_93733


namespace car_b_speed_l937_93794

/--
A car A going at 30 miles per hour set out on an 80-mile trip at 9:00 a.m.
Exactly 10 minutes later, a car B left from the same place and followed the same route.
Car B caught up with car A at 10:30 a.m.
Prove that the speed of car B is 33.75 miles per hour.
-/
theorem car_b_speed
    (v_a : ℝ) (t_start_a t_start_b t_end : ℝ) (v_b : ℝ)
    (h1 : v_a = 30) 
    (h2 : t_start_a = 9) 
    (h3 : t_start_b = 9 + (10 / 60)) 
    (h4 : t_end = 10.5) 
    (h5 : t_end - t_start_b = (4 / 3))
    (h6 : v_b * (t_end - t_start_b) = v_a * (t_end - t_start_a) + (v_a * (10 / 60))) :
  v_b = 33.75 := 
sorry

end car_b_speed_l937_93794


namespace binom_18_4_eq_3060_l937_93711

theorem binom_18_4_eq_3060 : Nat.choose 18 4 = 3060 := by
  sorry

end binom_18_4_eq_3060_l937_93711


namespace days_elapsed_l937_93729

theorem days_elapsed
  (initial_amount : ℕ)
  (daily_spending : ℕ)
  (total_savings : ℕ)
  (doubling_factor : ℕ)
  (additional_amount : ℕ)
  :
  initial_amount = 50 →
  daily_spending = 15 →
  doubling_factor = 2 →
  additional_amount = 10 →
  2 * (initial_amount - daily_spending) * total_savings + additional_amount = 500 →
  total_savings = 7 :=
by
  intros h_initial h_spending h_doubling h_additional h_total
  sorry

end days_elapsed_l937_93729


namespace ramu_spent_on_repairs_l937_93722

theorem ramu_spent_on_repairs 
    (initial_cost : ℝ) (selling_price : ℝ) (profit_percent : ℝ) (R : ℝ) 
    (h1 : initial_cost = 42000) 
    (h2 : selling_price = 64900) 
    (h3 : profit_percent = 18) 
    (h4 : profit_percent / 100 = (selling_price - (initial_cost + R)) / (initial_cost + R)) : 
    R = 13000 :=
by
  rw [h1, h2, h3] at h4
  sorry

end ramu_spent_on_repairs_l937_93722


namespace sufficient_but_not_necessary_l937_93778

theorem sufficient_but_not_necessary (a : ℝ) : a = 1 → |a| = 1 ∧ (|a| = 1 → a = 1 → false) :=
by
  sorry

end sufficient_but_not_necessary_l937_93778
