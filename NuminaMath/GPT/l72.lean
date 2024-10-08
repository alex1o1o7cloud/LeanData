import Mathlib

namespace vertex_of_parabola_l72_72066

theorem vertex_of_parabola (a : ℝ) :
  (∃ (k : ℝ), ∀ x : ℝ, y = -4*x - 1 → x = 2 ∧ (a - 4) = -4 * 2 - 1) → 
  (2, -9) = (2, a - 4) → a = -5 :=
by
  sorry

end vertex_of_parabola_l72_72066


namespace negation_proof_l72_72261

theorem negation_proof : ¬ (∃ x : ℝ, (x ≤ -1) ∨ (x ≥ 2)) ↔ ∀ x : ℝ, -1 < x ∧ x < 2 := 
by 
  -- proof skipped
  sorry

end negation_proof_l72_72261


namespace chocolate_bars_produced_per_minute_l72_72950

theorem chocolate_bars_produced_per_minute
  (sugar_per_bar : ℝ)
  (total_sugar : ℝ)
  (time_in_minutes : ℝ) 
  (bars_per_min : ℝ) :
  sugar_per_bar = 1.5 →
  total_sugar = 108 →
  time_in_minutes = 2 →
  bars_per_min = 36 :=
sorry

end chocolate_bars_produced_per_minute_l72_72950


namespace john_yasmin_child_ratio_l72_72074

theorem john_yasmin_child_ratio
  (gabriel_grandkids : ℕ)
  (yasmin_children : ℕ)
  (john_children : ℕ)
  (h1 : gabriel_grandkids = 6)
  (h2 : yasmin_children = 2)
  (h3 : john_children + yasmin_children = gabriel_grandkids) :
  john_children / yasmin_children = 2 :=
by 
  sorry

end john_yasmin_child_ratio_l72_72074


namespace gh_two_value_l72_72089

def g (x : ℤ) : ℤ := 3 * x ^ 2 + 2
def h (x : ℤ) : ℤ := -5 * x ^ 3 + 2

theorem gh_two_value : g (h 2) = 4334 := by
  sorry

end gh_two_value_l72_72089


namespace count_two_digit_numbers_with_at_least_one_5_l72_72715

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def has_digit_5 (n : ℕ) : Prop := ∃ (a b : ℕ), is_two_digit (10 * a + b) ∧ (a = 5 ∨ b = 5)

theorem count_two_digit_numbers_with_at_least_one_5 : 
  ∃ count : ℕ, (∀ n, is_two_digit n → has_digit_5 n → n ∈ Finset.range (100)) ∧ count = 18 := 
sorry

end count_two_digit_numbers_with_at_least_one_5_l72_72715


namespace rows_identical_l72_72285

theorem rows_identical {n : ℕ} {a : Fin n → ℝ} {k : Fin n → Fin n}
  (h_inc : ∀ i j : Fin n, i < j → a i < a j)
  (h_perm : ∀ i j : Fin n, k i ≠ k j → a (k i) ≠ a (k j))
  (h_sum_inc : ∀ i j : Fin n, i < j → a i + a (k i) < a j + a (k j)) :
  ∀ i : Fin n, a i = a (k i) :=
by
  sorry

end rows_identical_l72_72285


namespace base_angle_of_isosceles_triangle_l72_72077

-- Definitions based on the problem conditions
def is_isosceles_triangle (A B C: ℝ) := (A = B) ∨ (B = C) ∨ (C = A)
def angle_sum_triangle (A B C: ℝ) := A + B + C = 180

-- The main theorem we want to prove
theorem base_angle_of_isosceles_triangle (A B C: ℝ)
(h1: is_isosceles_triangle A B C)
(h2: A = 50 ∨ B = 50 ∨ C = 50):
C = 50 ∨ C = 65 :=
by
  sorry

end base_angle_of_isosceles_triangle_l72_72077


namespace sequence_values_l72_72412

variable {a1 a2 b2 : ℝ}

theorem sequence_values
  (arithmetic : 2 * a1 = 1 + a2 ∧ 2 * a2 = a1 + 4)
  (geometric : b2 ^ 2 = 1 * 4) :
  (a1 + a2) / b2 = 5 / 2 :=
by
  sorry

end sequence_values_l72_72412


namespace xyz_value_l72_72951

theorem xyz_value
  (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 45)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19)
  (h3 : x^2 * y^2 + y^2 * z^2 + z^2 * x^2 = 11) :
  x * y * z = 26 / 3 :=
sorry

end xyz_value_l72_72951


namespace find_length_of_AB_l72_72326

-- Definitions of the conditions
def areas_ratio (A B C D : Point) (areaABC areaADC : ℝ) :=
  (areaABC / areaADC) = (7 / 3)

def total_length (A B C D : Point) (AB CD : ℝ) :=
  AB + CD = 280

-- Statement of the proof problem
theorem find_length_of_AB
  (A B C D : Point)
  (AB CD : ℝ)
  (areaABC areaADC : ℝ)
  (h_height_not_zero : h ≠ 0) -- Assumption to ensure height is non-zero
  (h_areas_ratio : areas_ratio A B C D areaABC areaADC)
  (h_total_length : total_length A B C D AB CD) :
  AB = 196 := sorry

end find_length_of_AB_l72_72326


namespace determinant_of_given_matrix_l72_72480

-- Define the given matrix
def given_matrix (z : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![z + 2, z, z], ![z, z + 3, z], ![z, z, z + 4]]

-- Define the proof statement
theorem determinant_of_given_matrix (z : ℂ) : Matrix.det (given_matrix z) = 22 * z + 24 :=
by
  sorry

end determinant_of_given_matrix_l72_72480


namespace cost_per_bracelet_l72_72976

/-- Each friend and the number of their name's letters -/
def friends_letters_counts : List (String × Nat) :=
  [("Jessica", 7), ("Tori", 4), ("Lily", 4), ("Patrice", 7)]

/-- Total cost spent by Robin -/
def total_cost : Nat := 44

/-- Calculate the total number of bracelets -/
def total_bracelets : Nat :=
  friends_letters_counts.foldr (λ p acc => p.snd + acc) 0

theorem cost_per_bracelet : (total_cost / total_bracelets) = 2 :=
  by
    sorry

end cost_per_bracelet_l72_72976


namespace rhombus_area_l72_72423

theorem rhombus_area (d1 d2 : ℝ) (θ : ℝ) (h1 : d1 = 8) (h2 : d2 = 10) (h3 : Real.sin θ = 3 / 5) : 
  (1 / 2) * d1 * d2 * Real.sin θ = 24 :=
by
  sorry

end rhombus_area_l72_72423


namespace last_digit_square_of_second_l72_72218

def digit1 := 1
def digit2 := 3
def digit3 := 4
def digit4 := 9

theorem last_digit_square_of_second :
  digit4 = digit2 ^ 2 :=
by
  -- Conditions
  have h1 : digit1 = digit2 / 3 := by sorry
  have h2 : digit3 = digit1 + digit2 := by sorry
  sorry

end last_digit_square_of_second_l72_72218


namespace mrs_sheridan_fish_distribution_l72_72180

theorem mrs_sheridan_fish_distribution :
  let initial_fish := 125
  let additional_fish := 250
  let total_fish := initial_fish + additional_fish
  let small_aquarium_capacity := 150
  let fish_in_small_aquarium := small_aquarium_capacity
  let fish_in_large_aquarium := total_fish - fish_in_small_aquarium
  fish_in_large_aquarium = 225 :=
by {
  let initial_fish := 125
  let additional_fish := 250
  let total_fish := initial_fish + additional_fish
  let small_aquarium_capacity := 150
  let fish_in_small_aquarium := small_aquarium_capacity
  let fish_in_large_aquarium := total_fish - fish_in_small_aquarium

  have : fish_in_large_aquarium = 225 := by sorry
  exact this
}

end mrs_sheridan_fish_distribution_l72_72180


namespace min_xyz_product_l72_72844

open Real

theorem min_xyz_product
  (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h_sum : x + y + z = 1)
  (h_no_more_than_twice : x ≤ 2 * y ∧ y ≤ 2 * x ∧ y ≤ 2 * z ∧ z ≤ 2 * y) :
  ∃ p : ℝ, (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z → x + y + z = 1 → x ≤ 2 * y ∧ y ≤ 2 * x ∧ y ≤ 2 * z ∧ z ≤ 2 * y → x * y * z ≥ p) ∧ p = 1 / 32 :=
by
  sorry

end min_xyz_product_l72_72844


namespace mango_coconut_ratio_l72_72607

open Function

theorem mango_coconut_ratio
  (mango_trees : ℕ)
  (coconut_trees : ℕ)
  (total_trees : ℕ)
  (R : ℚ)
  (H1 : mango_trees = 60)
  (H2 : coconut_trees = R * 60 - 5)
  (H3 : total_trees = 85)
  (H4 : total_trees = mango_trees + coconut_trees) :
  R = 1/2 :=
by
  sorry

end mango_coconut_ratio_l72_72607


namespace cubic_has_three_natural_roots_l72_72962

theorem cubic_has_three_natural_roots (p : ℝ) :
  (∃ (x1 x2 x3 : ℕ), 5 * (x1:ℝ)^3 - 5 * (p + 1) * (x1:ℝ)^2 + (71 * p - 1) * (x1:ℝ) + 1 = 66 * p ∧
                     5 * (x2:ℝ)^3 - 5 * (p + 1) * (x2:ℝ)^2 + (71 * p - 1) * (x2:ℝ) + 1 = 66 * p ∧
                     5 * (x3:ℝ)^3 - 5 * (p + 1) * (x3:ℝ)^2 + (71 * p - 1) * (x3:ℝ) + 1 = 66 * p) ↔ p = 76 :=
by sorry

end cubic_has_three_natural_roots_l72_72962


namespace part1_part2_l72_72847

def p (x : ℝ) : Prop := x^2 - 10*x + 16 ≤ 0
def q (x m : ℝ) : Prop := m > 0 ∧ x^2 - 4*m*x + 3*m^2 ≤ 0

theorem part1 (x : ℝ) : 
  (∃ (m : ℝ), m = 1 ∧ (p x ∨ q x m)) → 1 ≤ x ∧ x ≤ 8 :=
by
  intros
  sorry

theorem part2 (m : ℝ) :
  (∀ x, q x m → p x) ∧ ∃ x, ¬ q x m ∧ p x → 2 ≤ m ∧ m ≤ 8/3 :=
by
  intros
  sorry

end part1_part2_l72_72847


namespace vector_product_magnitude_l72_72449

noncomputable def vector_magnitude (a b : ℝ) (theta : ℝ) : ℝ :=
  abs a * abs b * Real.sin theta

theorem vector_product_magnitude 
  (a b : ℝ) 
  (theta : ℝ) 
  (ha : abs a = 4) 
  (hb : abs b = 3) 
  (h_dot : a * b = -2) 
  (theta_range : 0 ≤ theta ∧ theta ≤ Real.pi)
  (cos_theta : Real.cos theta = -1/6) 
  (sin_theta : Real.sin theta = Real.sqrt 35 / 6) :
  vector_magnitude a b theta = 2 * Real.sqrt 35 :=
sorry

end vector_product_magnitude_l72_72449


namespace lambda_range_l72_72062

noncomputable def sequence_a (n : ℕ) : ℝ :=
  if n = 0 then 1 else
  sequence_a (n - 1) / (sequence_a (n - 1) + 2)

noncomputable def sequence_b (lambda : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then -3/2 * lambda else
  (n - 2 * lambda) * (1 / sequence_a (n - 1) + 1)

def is_monotonically_increasing (seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → seq (n+1) > seq n

theorem lambda_range (lambda : ℝ) (hn : is_monotonically_increasing (sequence_b lambda)) : lambda < 4/5 := sorry

end lambda_range_l72_72062


namespace cubical_storage_unit_blocks_l72_72776

theorem cubical_storage_unit_blocks :
  let side_length := 8
  let thickness := 1
  let total_volume := side_length ^ 3
  let interior_side_length := side_length - 2 * thickness
  let interior_volume := interior_side_length ^ 3
  let blocks_required := total_volume - interior_volume
  blocks_required = 296 := by
    sorry

end cubical_storage_unit_blocks_l72_72776


namespace difference_of_squares_divisible_by_18_l72_72871

-- Definitions of odd integers.
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- The main theorem stating the equivalence.
theorem difference_of_squares_divisible_by_18 (a b : ℤ) 
  (ha : is_odd a) (hb : is_odd b) : 
  ((3 * a + 2) ^ 2 - (3 * b + 2) ^ 2) % 18 = 0 := 
by
  sorry

end difference_of_squares_divisible_by_18_l72_72871


namespace ellipse_chord_through_focus_l72_72443

theorem ellipse_chord_through_focus (x y : ℝ) (a b : ℝ := 6) (c : ℝ := 3 * Real.sqrt 3)
  (F : ℝ × ℝ := (3 * Real.sqrt 3, 0)) (AF BF : ℝ) :
  (x^2 / 36) + (y^2 / 9) = 1 ∧ ((x - 3 * Real.sqrt 3)^2 + y^2 = (3/2)^2) ∧
  (AF = 3 / 2) ∧ F.1 = 3 * Real.sqrt 3 ∧ F.2 = 0 →
  BF = 3 / 2 :=
sorry

end ellipse_chord_through_focus_l72_72443


namespace probability_of_selecting_female_l72_72549

theorem probability_of_selecting_female (total_students female_students male_students : ℕ)
  (h_total : total_students = female_students + male_students)
  (h_female : female_students = 3)
  (h_male : male_students = 1) :
  (female_students : ℚ) / total_students = 3 / 4 :=
by
  sorry

end probability_of_selecting_female_l72_72549


namespace T_5_3_l72_72800

def T (x y : ℕ) : ℕ := 4 * x + 5 * y + x * y

theorem T_5_3 : T 5 3 = 50 :=
by
  sorry

end T_5_3_l72_72800


namespace Robert_books_read_in_six_hours_l72_72178

theorem Robert_books_read_in_six_hours (P H T: ℕ)
    (h1: P = 270)
    (h2: H = 90)
    (h3: T = 6):
    T * H / P = 2 :=
by 
    -- sorry placeholder to indicate that this is where the proof goes.
    sorry

end Robert_books_read_in_six_hours_l72_72178


namespace intersection_A_B_l72_72675

-- Define set A and its condition
def A : Set ℝ := { y | ∃ (x : ℝ), y = x^2 }

-- Define set B and its condition
def B : Set ℝ := { x | ∃ (y : ℝ), y = Real.sqrt (1 - x^2) }

-- Define the set intersection A ∩ B
def A_intersect_B : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

-- The theorem statement
theorem intersection_A_B :
  A ∩ B = { x : ℝ | 0 ≤ x ∧ x ≤ 1 } :=
sorry

end intersection_A_B_l72_72675


namespace gas_cycle_work_done_l72_72764

noncomputable def p0 : ℝ := 10^5
noncomputable def V0 : ℝ := 1

theorem gas_cycle_work_done :
  (3 * Real.pi * p0 * V0 = 942) :=
by
  have h1 : p0 = 10^5 := by rfl
  have h2 : V0 = 1 := by rfl
  sorry

end gas_cycle_work_done_l72_72764


namespace original_price_calculation_l72_72076

variable (P : ℝ)
variable (selling_price : ℝ := 1040)
variable (loss_percentage : ℝ := 20)

theorem original_price_calculation :
  P = 1300 :=
by
  have sell_percent := 100 - loss_percentage
  have SP_eq := selling_price = (sell_percent / 100) * P
  sorry

end original_price_calculation_l72_72076


namespace xy_div_eq_one_third_l72_72696

theorem xy_div_eq_one_third (x y z : ℝ) 
  (h1 : x + y = 2 * x + z)
  (h2 : x - 2 * y = 4 * z)
  (h3 : x + y + z = 21)
  (h4 : y / z = 6) : 
  x / y = 1 / 3 :=
by
  sorry

end xy_div_eq_one_third_l72_72696


namespace first_candidate_percentage_l72_72974

theorem first_candidate_percentage (total_votes : ℕ) (invalid_percentage : ℕ) (second_candidate_votes : ℕ) 
  (h_total_votes : total_votes = 7500) 
  (h_invalid_percentage : invalid_percentage = 20) 
  (h_second_candidate_votes : second_candidate_votes = 2700) : 
  (100 * (total_votes * (1 - (invalid_percentage / 100)) - second_candidate_votes) / (total_votes * (1 - (invalid_percentage / 100)))) = 55 :=
by
  sorry

end first_candidate_percentage_l72_72974


namespace coprime_exponents_iff_l72_72167

theorem coprime_exponents_iff (p q : ℕ) : 
  Nat.gcd (2^p - 1) (2^q - 1) = 1 ↔ Nat.gcd p q = 1 :=
by 
  sorry

end coprime_exponents_iff_l72_72167


namespace chairs_made_after_tables_l72_72933

def pieces_of_wood : Nat := 672
def wood_per_table : Nat := 12
def wood_per_chair : Nat := 8
def number_of_tables : Nat := 24

theorem chairs_made_after_tables (pieces_of_wood wood_per_table wood_per_chair number_of_tables : Nat) :
  wood_per_table * number_of_tables <= pieces_of_wood ->
  (pieces_of_wood - wood_per_table * number_of_tables) / wood_per_chair = 48 :=
by
  sorry

end chairs_made_after_tables_l72_72933


namespace probability_point_between_C_and_E_l72_72230

noncomputable def length_between_points (total_length : ℝ) (ratio : ℝ) : ℝ :=
ratio * total_length

theorem probability_point_between_C_and_E
  (A B C D E : ℝ)
  (h1 : A < B)
  (h2 : C < E)
  (h3 : B - A = 4 * (D - A))
  (h4 : B - A = 8 * (B - C))
  (h5 : B - E = 2 * (E - C)) :
  (E - C) / (B - A) = 1 / 24 :=
by 
  sorry

end probability_point_between_C_and_E_l72_72230


namespace balloons_remaining_l72_72616

-- Define the initial conditions
def initial_balloons : ℕ := 9
def lost_balloons : ℕ := 2

-- State the theorem
theorem balloons_remaining : initial_balloons - lost_balloons = 7 := by
  -- Add the solution proof steps here
  sorry

end balloons_remaining_l72_72616


namespace diagonal_of_rectangle_l72_72517

theorem diagonal_of_rectangle (l w d : ℝ) (h_length : l = 15) (h_area : l * w = 120) (h_diagonal : d^2 = l^2 + w^2) : d = 17 :=
by
  sorry

end diagonal_of_rectangle_l72_72517


namespace optimal_saving_is_45_cents_l72_72415

def initial_price : ℝ := 18
def fixed_discount : ℝ := 3
def percentage_discount : ℝ := 0.15

def price_after_fixed_discount (price fixed_discount : ℝ) : ℝ :=
  price - fixed_discount

def price_after_percentage_discount (price percentage_discount : ℝ) : ℝ :=
  price * (1 - percentage_discount)

def optimal_saving (initial_price fixed_discount percentage_discount : ℝ) : ℝ :=
  let price1 := price_after_fixed_discount initial_price fixed_discount
  let final_price1 := price_after_percentage_discount price1 percentage_discount
  let price2 := price_after_percentage_discount initial_price percentage_discount
  let final_price2 := price_after_fixed_discount price2 fixed_discount
  final_price1 - final_price2

theorem optimal_saving_is_45_cents : optimal_saving initial_price fixed_discount percentage_discount = 0.45 :=
by 
  sorry

end optimal_saving_is_45_cents_l72_72415


namespace Kaleb_second_half_points_l72_72917

theorem Kaleb_second_half_points (first_half_points total_points : ℕ) (h1 : first_half_points = 43) (h2 : total_points = 66) : total_points - first_half_points = 23 := by
  sorry

end Kaleb_second_half_points_l72_72917


namespace length_of_platform_l72_72923

theorem length_of_platform (v t_m t_p L_t L_p : ℝ)
    (h1 : v = 33.3333333)
    (h2 : t_m = 22)
    (h3 : t_p = 45)
    (h4 : L_t = v * t_m)
    (h5 : L_t + L_p = v * t_p) :
    L_p = 766.666666 :=
by
  sorry

end length_of_platform_l72_72923


namespace avg_age_initial_group_l72_72945

theorem avg_age_initial_group (N : ℕ) (A avg_new_persons avg_entire_group : ℝ) (hN : N = 15)
  (h_avg_new_persons : avg_new_persons = 15) (h_avg_entire_group : avg_entire_group = 15.5) :
  (A * (N : ℝ) + 15 * avg_new_persons) = ((N + 15) : ℝ) * avg_entire_group → A = 16 :=
by
  intro h
  have h_initial : N = 15 := hN
  have h_new : avg_new_persons = 15 := h_avg_new_persons
  have h_group : avg_entire_group = 15.5 := h_avg_entire_group
  sorry

end avg_age_initial_group_l72_72945


namespace age_ratio_in_two_years_l72_72670

theorem age_ratio_in_two_years :
  ∀ (B M : ℕ), B = 10 → M = B + 12 → (M + 2) / (B + 2) = 2 := by
  intros B M hB hM
  sorry

end age_ratio_in_two_years_l72_72670


namespace range_of_function_l72_72094

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x

theorem range_of_function :
  ∀ (x : ℝ), 1 ≤ x ∧ x < 5 → -4 ≤ f x ∧ f x < 5 :=
by
  intro x hx
  sorry

end range_of_function_l72_72094


namespace sequence_initial_term_l72_72876

theorem sequence_initial_term (a : ℕ) :
  let a_1 := a
  let a_2 := 2
  let a_3 := a_1 + a_2
  let a_4 := a_1 + a_2 + a_3
  let a_5 := a_1 + a_2 + a_3 + a_4
  let a_6 := a_1 + a_2 + a_3 + a_4 + a_5
  a_6 = 56 → a = 5 :=
by
  intros h
  sorry

end sequence_initial_term_l72_72876


namespace ball_returns_to_Ben_after_three_throws_l72_72783

def circle_throw (n : ℕ) (skip : ℕ) (start : ℕ) : ℕ :=
  (start + skip) % n

theorem ball_returns_to_Ben_after_three_throws :
  ∀ (n : ℕ) (skip : ℕ) (start : ℕ),
  n = 15 → skip = 5 → start = 1 →
  (circle_throw n skip (circle_throw n skip (circle_throw n skip start))) = start :=
by
  intros n skip start hn hskip hstart
  sorry

end ball_returns_to_Ben_after_three_throws_l72_72783


namespace eccentricity_of_hyperbola_l72_72379

variable (a b c e : ℝ)

-- The hyperbola definition and conditions.
def hyperbola (a b : ℝ) := (a > 0) ∧ (b > 0) ∧ (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)

-- Eccentricity is greater than 1 and less than the specified upper bound
def eccentricity_range (e : ℝ) := 1 < e ∧ e < (2 * Real.sqrt 3) / 3

-- Main theorem statement: Given the hyperbola with conditions, prove eccentricity lies in the specified range.
theorem eccentricity_of_hyperbola (h : hyperbola a b) (h_line : ∀ (x y : ℝ), y = x * (Real.sqrt 3) / 3 - 0 -> y^2 ≤ (c^2 - x^2 * a^2)) :
  eccentricity_range e :=
sorry

end eccentricity_of_hyperbola_l72_72379


namespace quadratic_has_equal_roots_l72_72827

-- Proposition: If the quadratic equation 3x^2 + 6x + m = 0 has two equal real roots, then m = 3.

theorem quadratic_has_equal_roots (m : ℝ) : 3 * 6 - 12 * m = 0 → m = 3 :=
by
  intro h
  sorry

end quadratic_has_equal_roots_l72_72827


namespace percentage_change_difference_l72_72855

-- Define initial and final percentages
def initial_yes : ℝ := 0.4
def initial_no : ℝ := 0.6
def final_yes : ℝ := 0.6
def final_no : ℝ := 0.4

-- Definition for the percentage of students who changed their opinion
def y_min : ℝ := 0.2 -- 20%
def y_max : ℝ := 0.6 -- 60%

-- Calculate the difference
def difference_y : ℝ := y_max - y_min

theorem percentage_change_difference :
  difference_y = 0.4 := by
  sorry

end percentage_change_difference_l72_72855


namespace sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l72_72358

def seven_pow_seven_minus_seven_pow_four : ℤ := 7^7 - 7^4
def prime_factors_of_three_hundred_forty_two : List ℤ := [2, 3, 19]

theorem sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four : 
  let distinct_prime_factors := prime_factors_of_three_hundred_forty_two.head!
  + prime_factors_of_three_hundred_forty_two.tail!.head!
  + prime_factors_of_three_hundred_forty_two.tail!.tail!.head!
  seven_pow_seven_minus_seven_pow_four = 7^4 * (7^3 - 1) ∧
  7^3 - 1 = 342 ∧
  prime_factors_of_three_hundred_forty_two = [2, 3, 19] ∧
  distinct_prime_factors = 24 := 
sorry

end sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l72_72358


namespace fraction_left_after_3_days_l72_72392

-- Defining work rates of A and B
def A_rate := 1 / 15
def B_rate := 1 / 20

-- Total work rate of A and B when working together
def combined_rate := A_rate + B_rate

-- Work completed by A and B in 3 days
def work_done := 3 * combined_rate

-- Fraction of work left
def fraction_work_left := 1 - work_done

-- Statement to prove:
theorem fraction_left_after_3_days : fraction_work_left = 13 / 20 :=
by
  have A_rate_def: A_rate = 1 / 15 := rfl
  have B_rate_def: B_rate = 1 / 20 := rfl
  have combined_rate_def: combined_rate = A_rate + B_rate := rfl
  have work_done_def: work_done = 3 * combined_rate := rfl
  have fraction_work_left_def: fraction_work_left = 1 - work_done := rfl
  sorry

end fraction_left_after_3_days_l72_72392


namespace sum_of_solutions_eq_zero_l72_72030

noncomputable def f (x : ℝ) : ℝ := 2 ^ |x| + 5 * |x|

theorem sum_of_solutions_eq_zero (x : ℝ) (hx : f x = 28) :
  x + -x = 0 :=
by
  sorry

end sum_of_solutions_eq_zero_l72_72030


namespace probability_of_green_ball_is_2_over_5_l72_72182

noncomputable def container_probabilities : ℚ :=
  let prob_A_selected : ℚ := 1/2
  let prob_B_selected : ℚ := 1/2
  let prob_green_in_A : ℚ := 5/10
  let prob_green_in_B : ℚ := 3/10

  prob_A_selected * prob_green_in_A + prob_B_selected * prob_green_in_B

theorem probability_of_green_ball_is_2_over_5 :
  container_probabilities = 2 / 5 := by
  sorry

end probability_of_green_ball_is_2_over_5_l72_72182


namespace math_proof_problem_l72_72963

-- Definitions for conditions:
def condition1 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 3 / 2) = -f x
def condition2 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x - 3 / 4) = -f (- (x - 3 / 4))

-- Statements to prove:
def statement1 (f : ℝ → ℝ) : Prop := ∃ p, p ≠ 0 ∧ ∀ x, f (x + p) = f x
def statement2 (f : ℝ → ℝ) : Prop := ∀ x, f (-(3 / 4) - x) = f (-(3 / 4) + x)
def statement3 (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def statement4 (f : ℝ → ℝ) : Prop := ¬(∀ x y : ℝ, x < y → f x ≤ f y)

theorem math_proof_problem (f : ℝ → ℝ) (h1 : condition1 f) (h2 : condition2 f) :
  statement1 f ∧ statement2 f ∧ statement3 f ∧ statement4 f :=
by
  sorry

end math_proof_problem_l72_72963


namespace cos_neg_300_eq_positive_half_l72_72924

theorem cos_neg_300_eq_positive_half : Real.cos (-300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_neg_300_eq_positive_half_l72_72924


namespace number_of_sides_l72_72457

theorem number_of_sides (n : ℕ) (h : (n - 2) * 180 = 900) : n = 7 := 
by {
  sorry
}

end number_of_sides_l72_72457


namespace profit_percent_300_l72_72220

theorem profit_percent_300 (SP : ℝ) (h : SP ≠ 0) (CP : ℝ) (h1 : CP = 0.25 * SP) : 
  (SP - CP) / CP * 100 = 300 := 
  sorry

end profit_percent_300_l72_72220


namespace set_union_proof_l72_72860

theorem set_union_proof (a b : ℝ) (A B : Set ℝ) 
  (hA : A = {1, 2^a})
  (hB : B = {a, b}) 
  (h_inter : A ∩ B = {1/4}) :
  A ∪ B = {-2, 1, 1/4} := 
by 
  sorry

end set_union_proof_l72_72860


namespace obtuse_triangles_in_17_gon_l72_72970

noncomputable def number_of_obtuse_triangles (n : ℕ): ℕ := 
  if h : n ≥ 3 then (n * (n - 1) * (n - 2)) / 6 else 0

theorem obtuse_triangles_in_17_gon : number_of_obtuse_triangles 17 = 476 := sorry

end obtuse_triangles_in_17_gon_l72_72970


namespace divisor_of_100_by_quotient_9_and_remainder_1_l72_72395

theorem divisor_of_100_by_quotient_9_and_remainder_1 :
  ∃ d : ℕ, 100 = d * 9 + 1 ∧ d = 11 :=
by
  sorry

end divisor_of_100_by_quotient_9_and_remainder_1_l72_72395


namespace quadratic_sum_l72_72158

theorem quadratic_sum (x : ℝ) :
  ∃ a h k : ℝ, (5*x^2 - 10*x - 3 = a*(x - h)^2 + k) ∧ (a + h + k = -2) :=
sorry

end quadratic_sum_l72_72158


namespace gcd_problem_l72_72403

open Int -- Open the integer namespace to use gcd.

theorem gcd_problem : Int.gcd (Int.gcd 188094 244122) 395646 = 6 :=
by
  -- provide the proof here
  sorry

end gcd_problem_l72_72403


namespace min_possible_value_box_l72_72691

theorem min_possible_value_box (a b : ℤ) (h_ab : a * b = 35) : a^2 + b^2 ≥ 74 := sorry

end min_possible_value_box_l72_72691


namespace sum_of_all_possible_values_of_M_l72_72781

-- Given conditions
-- M * (M - 8) = -7
-- We need to prove that the sum of all possible values of M is 8

theorem sum_of_all_possible_values_of_M : 
  ∃ M1 M2 : ℝ, (M1 * (M1 - 8) = -7) ∧ (M2 * (M2 - 8) = -7) ∧ (M1 + M2 = 8) :=
by
  sorry

end sum_of_all_possible_values_of_M_l72_72781


namespace eccentricity_of_hyperbola_l72_72109

theorem eccentricity_of_hyperbola :
  let a := Real.sqrt 5
  let b := 2
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  (∃ (x y : ℝ), (x^2 / 5) - (y^2 / 4) = 1 ∧ e = (3 * Real.sqrt 5) / 5) := sorry

end eccentricity_of_hyperbola_l72_72109


namespace incorrect_statement_A_l72_72119

theorem incorrect_statement_A (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b) :
  ¬ (a - a^2 > b - b^2) := sorry

end incorrect_statement_A_l72_72119


namespace cos_sin_combination_l72_72352

theorem cos_sin_combination (x : ℝ) (h : 2 * Real.cos x + 3 * Real.sin x = 4) : 
  3 * Real.cos x - 2 * Real.sin x = 0 := 
by 
  sorry

end cos_sin_combination_l72_72352


namespace total_heads_of_cabbage_l72_72471

-- Problem definition for the first patch
def first_patch : ℕ := 12 * 15

-- Problem definition for the second patch
def second_patch : ℕ := 10 + 12 + 14 + 16 + 18 + 20 + 22 + 24

-- Problem statement
theorem total_heads_of_cabbage : first_patch + second_patch = 316 := by
  sorry

end total_heads_of_cabbage_l72_72471


namespace magic_square_base_l72_72380

theorem magic_square_base :
  ∃ b : ℕ, (b + 1 + (b + 5) + 2 = 9 + (b + 3)) ∧ b = 3 :=
by
  use 3
  -- Proof in Lean goes here
  sorry

end magic_square_base_l72_72380


namespace age_comparison_l72_72908

variable (P A F X : ℕ)

theorem age_comparison :
  P = 50 →
  P = 5 / 4 * A →
  P = 5 / 6 * F →
  X = 50 - A →
  X = 10 :=
by { sorry }

end age_comparison_l72_72908


namespace min_value_fraction_l72_72968

open Real

theorem min_value_fraction (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 3) :
  (∃ x, x = (a + b) / (a * b * c) ∧ x = 16 / 9) :=
by
  sorry

end min_value_fraction_l72_72968


namespace distinct_meals_l72_72106

-- Define the conditions
def number_of_entrees : ℕ := 4
def number_of_drinks : ℕ := 3
def number_of_desserts : ℕ := 2

-- Define the main theorem
theorem distinct_meals : number_of_entrees * number_of_drinks * number_of_desserts = 24 := 
by
  -- sorry is used to skip the proof
  sorry

end distinct_meals_l72_72106


namespace sum_of_reciprocals_of_shifted_roots_l72_72985

noncomputable def cubic_poly (x : ℝ) := 45 * x^3 - 75 * x^2 + 33 * x - 2

theorem sum_of_reciprocals_of_shifted_roots (a b c : ℝ) 
  (ha : cubic_poly a = 0) 
  (hb : cubic_poly b = 0) 
  (hc : cubic_poly c = 0)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c)
  (h_bounds_a : 0 < a ∧ a < 1)
  (h_bounds_b : 0 < b ∧ b < 1)
  (h_bounds_c : 0 < c ∧ c < 1) :
  (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) = 4 / 3 := 
sorry

end sum_of_reciprocals_of_shifted_roots_l72_72985


namespace probability_of_two_accurate_forecasts_l72_72289

noncomputable def event_A : Type := {forecast : ℕ | forecast = 1}

def prob_A : ℝ := 0.9
def prob_A' : ℝ := 1 - prob_A

-- Define that there are 3 independent trials
def num_forecasts : ℕ := 3

-- Given
def probability_two_accurate (x : ℕ) : ℝ :=
if x = 2 then 3 * (prob_A^2 * prob_A') else 0

-- Statement to be proved
theorem probability_of_two_accurate_forecasts : probability_two_accurate 2 = 0.243 := by
  -- Proof will go here
  sorry

end probability_of_two_accurate_forecasts_l72_72289


namespace dorothy_score_l72_72335

theorem dorothy_score (T I D : ℝ) 
  (hT : T = 2 * I)
  (hI : I = (3 / 5) * D)
  (hSum : T + I + D = 252) : 
  D = 90 := 
by {
  sorry
}

end dorothy_score_l72_72335


namespace factorize_expression_l72_72401

variable (b : ℝ)

theorem factorize_expression : 2 * b^3 - 4 * b^2 + 2 * b = 2 * b * (b - 1)^2 := by
  sorry

end factorize_expression_l72_72401


namespace sum_of_number_and_conjugate_l72_72617

noncomputable def x : ℝ := 16 - Real.sqrt 2023
noncomputable def y : ℝ := 16 + Real.sqrt 2023

theorem sum_of_number_and_conjugate : x + y = 32 :=
by
  sorry

end sum_of_number_and_conjugate_l72_72617


namespace sam_balloons_l72_72859

theorem sam_balloons (f d t S : ℝ) (h₁ : f = 10.0) (h₂ : d = 16.0) (h₃ : t = 40.0) (h₄ : f + S - d = t) : S = 46.0 := 
by 
  -- Replace "sorry" with a valid proof to solve this problem
  sorry

end sam_balloons_l72_72859


namespace gcd_78_36_l72_72381

theorem gcd_78_36 : Nat.gcd 78 36 = 6 :=
by
  sorry

end gcd_78_36_l72_72381


namespace largest_three_digit_integer_l72_72802

theorem largest_three_digit_integer (n : ℕ) :
  75 * n ≡ 300 [MOD 450] →
  n < 1000 →
  ∃ m : ℕ, n = m ∧ (∀ k : ℕ, 75 * k ≡ 300 [MOD 450] ∧ k < 1000 → k ≤ n) := by
  sorry

end largest_three_digit_integer_l72_72802


namespace joe_paint_usage_l72_72481

noncomputable def paint_used_after_four_weeks : ℝ := 
  let total_paint := 480
  let first_week_paint := (1/5) * total_paint
  let second_week_paint := (1/6) * (total_paint - first_week_paint)
  let third_week_paint := (1/7) * (total_paint - first_week_paint - second_week_paint)
  let fourth_week_paint := (2/9) * (total_paint - first_week_paint - second_week_paint - third_week_paint)
  first_week_paint + second_week_paint + third_week_paint + fourth_week_paint

theorem joe_paint_usage :
  abs (paint_used_after_four_weeks - 266.66) < 0.01 :=
sorry

end joe_paint_usage_l72_72481


namespace trajectory_midpoints_parabola_l72_72953

theorem trajectory_midpoints_parabola {k : ℝ} (hk : k ≠ 0) :
  ∀ (x1 x2 y1 y2 : ℝ), 
    y1 = 2 * x1^2 → 
    y2 = 2 * x2^2 → 
    y2 - y1 = 2 * (x2 + x1) * (x2 - x1) → 
    x = (x1 + x2) / 2 → 
    k = (y2 - y1) / (x2 - x1) → 
    x = 1 / (4 * k) := 
sorry

end trajectory_midpoints_parabola_l72_72953


namespace product_is_even_l72_72574

theorem product_is_even (a b c : ℤ) : Even ((a - b) * (b - c) * (c - a)) := by
  sorry

end product_is_even_l72_72574


namespace algebra_expression_value_l72_72026

theorem algebra_expression_value (a b : ℝ) (h1 : a + b = 10) (h2 : a * b = 11) : a^2 - a * b + b^2 = 67 :=
by
  sorry

end algebra_expression_value_l72_72026


namespace simple_interest_problem_l72_72833

theorem simple_interest_problem (P : ℝ) (R : ℝ) (T : ℝ) : T = 10 → 
  ((P * R * T) / 100 = (4 / 5) * P) → R = 8 :=
by
  intros hT hsi
  sorry

end simple_interest_problem_l72_72833


namespace kmph_to_mps_l72_72562

theorem kmph_to_mps (s : ℝ) (h : s = 0.975) : s * (1000 / 3600) = 0.2708 := by
  -- We include the assumption s = 0.975 as part of the problem condition.
  -- Import Mathlib to gain access to real number arithmetic.
  -- sorry is added to indicate a place where the proof should go.
  sorry

end kmph_to_mps_l72_72562


namespace coeffs_sum_of_binomial_expansion_l72_72881

theorem coeffs_sum_of_binomial_expansion :
  (3 * x - 2) ^ 6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 →
  a_0 = 64 →
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = -63 :=
by
  sorry

end coeffs_sum_of_binomial_expansion_l72_72881


namespace diagonal_inequality_l72_72644

theorem diagonal_inequality (A B C D : ℝ × ℝ) (h1 : A.1 = 0) (h2 : B.1 = 0) (h3 : C.2 = 0) (h4 : D.2 = 0) 
  (ha : A.2 < B.2) (hd : D.1 < C.1) : 
  (Real.sqrt (A.2^2 + C.1^2)) * (Real.sqrt (B.2^2 + D.1^2)) > (Real.sqrt (A.2^2 + D.1^2)) * (Real.sqrt (B.2^2 + C.1^2)) :=
sorry

end diagonal_inequality_l72_72644


namespace rays_total_grocery_bill_l72_72227

-- Conditions
def hamburger_meat_cost : ℝ := 5.0
def crackers_cost : ℝ := 3.50
def frozen_veg_cost_per_bag : ℝ := 2.0
def frozen_veg_bags : ℕ := 4
def cheese_cost : ℝ := 3.50
def discount_rate : ℝ := 0.10

-- Total cost before discount
def total_cost_before_discount : ℝ :=
  hamburger_meat_cost + crackers_cost + (frozen_veg_cost_per_bag * frozen_veg_bags) + cheese_cost

-- Discount amount
def discount_amount : ℝ := discount_rate * total_cost_before_discount

-- Total cost after discount
def total_cost_after_discount : ℝ := total_cost_before_discount - discount_amount

-- Theorem: Ray's total grocery bill
theorem rays_total_grocery_bill : total_cost_after_discount = 18.0 :=
  by
    sorry

end rays_total_grocery_bill_l72_72227


namespace sum_of_squares_of_six_odds_not_2020_l72_72640

theorem sum_of_squares_of_six_odds_not_2020 :
  ¬ ∃ a1 a2 a3 a4 a5 a6 : ℤ, (∀ i ∈ [a1, a2, a3, a4, a5, a6], i % 2 = 1) ∧ (a1^2 + a2^2 + a3^2 + a4^2 + a5^2 + a6^2 = 2020) :=
by
  sorry

end sum_of_squares_of_six_odds_not_2020_l72_72640


namespace range_of_a_l72_72740

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else Real.log x / Real.log a

theorem range_of_a {a : ℝ} (h : ∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) : 
  a > 1/7 ∧ a < 1/3 :=
sorry

end range_of_a_l72_72740


namespace find_g_of_3_l72_72228

theorem find_g_of_3 (f g : ℝ → ℝ) (h₁ : ∀ x, f x = 2 * x + 3) (h₂ : ∀ x, g (x + 2) = f x) :
  g 3 = 5 :=
sorry

end find_g_of_3_l72_72228


namespace minimize_total_cost_l72_72405

open Real

noncomputable def total_cost (x : ℝ) (h : 50 ≤ x ∧ x ≤ 100) : ℝ :=
  (130 / x) * 2 * (2 + (x^2 / 360)) + (14 * 130 / x)

theorem minimize_total_cost :
  ∀ (x : ℝ) (h : 50 ≤ x ∧ x ≤ 100),
  total_cost x h = (2340 / x) + (13 * x / 18)
  ∧ (x = 18 * sqrt 10 → total_cost x h = 26 * sqrt 10) :=
by
  sorry

end minimize_total_cost_l72_72405


namespace image_digit_sum_l72_72936

theorem image_digit_sum 
  (cat chicken crab bear goat: ℕ)
  (h1 : 5 * crab = 10)
  (h2 : 4 * crab + goat = 11)
  (h3 : 2 * goat + crab + 2 * bear = 16)
  (h4 : cat + bear + 2 * goat + crab = 13)
  (h5 : 2 * crab + 2 * chicken + goat = 17) :
  cat = 1 ∧ chicken = 5 ∧ crab = 2 ∧ bear = 4 ∧ goat = 3 := by
  sorry

end image_digit_sum_l72_72936


namespace remainder_b_div_6_l72_72839

theorem remainder_b_div_6 (a b : ℕ) (r_a r_b : ℕ) 
  (h1 : a ≡ r_a [MOD 6]) 
  (h2 : b ≡ r_b [MOD 6]) 
  (h3 : a > b) 
  (h4 : (a - b) % 6 = 5) 
  : b % 6 = 0 := 
sorry

end remainder_b_div_6_l72_72839


namespace ages_l72_72451

-- Definitions of ages
variables (S M : ℕ) -- S: son's current age, M: mother's current age

-- Given conditions
def father_age : ℕ := 44
def son_father_relationship (S : ℕ) : Prop := father_age = S + S
def son_mother_relationship (S M : ℕ) : Prop := (S - 5) = (M - 10)

-- Theorem to prove the ages
theorem ages (S M : ℕ) (h1 : son_father_relationship S) (h2 : son_mother_relationship S M) :
  S = 22 ∧ M = 27 :=
by 
  sorry

end ages_l72_72451


namespace Gerald_toy_cars_l72_72725

theorem Gerald_toy_cars :
  let initial_toy_cars := 20
  let fraction_donated := 1 / 4
  let donated_toy_cars := initial_toy_cars * fraction_donated
  let remaining_toy_cars := initial_toy_cars - donated_toy_cars
  remaining_toy_cars = 15 := 
by
  sorry

end Gerald_toy_cars_l72_72725


namespace cistern_filling_time_l72_72806

theorem cistern_filling_time :
  let rate_P := (1 : ℚ) / 12
  let rate_Q := (1 : ℚ) / 15
  let combined_rate := rate_P + rate_Q
  let time_combined := 6
  let filled_after_combined := combined_rate * time_combined
  let remaining_after_combined := 1 - filled_after_combined
  let time_Q := remaining_after_combined / rate_Q
  time_Q = 1.5 := sorry

end cistern_filling_time_l72_72806


namespace store_money_left_l72_72138

variable (total_items : Nat) (original_price : ℝ) (discount_percent : ℝ)
variable (percent_sold : ℝ) (amount_owed : ℝ)

theorem store_money_left
  (h_total_items : total_items = 2000)
  (h_original_price : original_price = 50)
  (h_discount_percent : discount_percent = 0.80)
  (h_percent_sold : percent_sold = 0.90)
  (h_amount_owed : amount_owed = 15000)
  : (total_items * original_price * (1 - discount_percent) * percent_sold - amount_owed) = 3000 := 
by 
  sorry

end store_money_left_l72_72138


namespace minimum_value_f_on_interval_l72_72842

noncomputable def f (x : ℝ) : ℝ := (Real.cos x)^3 / (Real.sin x) + (Real.sin x)^3 / (Real.cos x)

theorem minimum_value_f_on_interval : ∃ x ∈ Set.Ioo 0 (Real.pi / 2), f x = 1 ∧ ∀ y ∈ Set.Ioo 0 (Real.pi / 2), f y ≥ 1 :=
by sorry

end minimum_value_f_on_interval_l72_72842


namespace sum_of_104th_parenthesis_is_correct_l72_72058

def b (n : ℕ) : ℕ := 2 * n + 1

def sumOf104thParenthesis : ℕ :=
  let cycleCount := 104 / 4
  let numbersBefore104 := 260
  let firstNumIndex := numbersBefore104 + 1
  let firstNum := b firstNumIndex
  let secondNum := b (firstNumIndex + 1)
  let thirdNum := b (firstNumIndex + 2)
  let fourthNum := b (firstNumIndex + 3)
  firstNum + secondNum + thirdNum + fourthNum

theorem sum_of_104th_parenthesis_is_correct : sumOf104thParenthesis = 2104 :=
  by
    sorry

end sum_of_104th_parenthesis_is_correct_l72_72058


namespace total_sample_any_candy_42_percent_l72_72493

-- Define percentages as rational numbers to avoid dealing with decimals directly
def percent_of_caught_A : ℚ := 12 / 100
def percent_of_not_caught_A : ℚ := 7 / 100
def percent_of_caught_B : ℚ := 5 / 100
def percent_of_not_caught_B : ℚ := 6 / 100
def percent_of_caught_C : ℚ := 9 / 100
def percent_of_not_caught_C : ℚ := 3 / 100

-- Sum up the percentages for those caught and not caught for each type of candy
def total_percent_A : ℚ := percent_of_caught_A + percent_of_not_caught_A
def total_percent_B : ℚ := percent_of_caught_B + percent_of_not_caught_B
def total_percent_C : ℚ := percent_of_caught_C + percent_of_not_caught_C

-- Sum of the total percentages for all types
def total_percent_sample_any_candy : ℚ := total_percent_A + total_percent_B + total_percent_C

theorem total_sample_any_candy_42_percent :
  total_percent_sample_any_candy = 42 / 100 :=
by
  sorry

end total_sample_any_candy_42_percent_l72_72493


namespace x_intercept_of_quadratic_l72_72361

theorem x_intercept_of_quadratic (a b c : ℝ) (h_vertex : ∃ x y : ℝ, y = a * x^2 + b * x + c ∧ x = 4 ∧ y = -2) 
(h_intercept : ∃ x y : ℝ, y = a * x^2 + b * x + c ∧ x = 1 ∧ y = 0) : 
∃ x : ℝ, x = 7 ∧ ∃ y : ℝ, y = a * x^2 + b * x + c ∧ y = 0 :=
sorry

end x_intercept_of_quadratic_l72_72361


namespace train_crossing_time_l72_72008

/-- 
Prove that the time it takes for a train traveling at 90 kmph with a length of 100.008 meters to cross a pole is 4.00032 seconds.
-/
theorem train_crossing_time (speed_kmph : ℝ) (length_meters : ℝ) : 
  speed_kmph = 90 → length_meters = 100.008 → (length_meters / (speed_kmph * (1000 / 3600))) = 4.00032 :=
by
  intros h1 h2
  sorry

end train_crossing_time_l72_72008


namespace weight_of_replaced_person_l72_72105

theorem weight_of_replaced_person :
  (∃ (W : ℝ), 
    let avg_increase := 1.5 
    let num_persons := 5 
    let new_person_weight := 72.5 
    (avg_increase * num_persons = new_person_weight - W)
  ) → 
  ∃ (W : ℝ), W = 65 :=
by
  sorry

end weight_of_replaced_person_l72_72105


namespace correct_alarm_clock_time_l72_72054

-- Definitions for the conditions
def alarm_set_time : ℕ := 7 * 60 -- in minutes
def museum_arrival_time : ℕ := 8 * 60 + 50 -- in minutes
def museum_touring_time : ℕ := 1 * 60 + 30 -- in minutes
def alarm_home_time : ℕ := 11 * 60 + 50 -- in minutes

-- The problem: proving the correct time the clock should be set to
theorem correct_alarm_clock_time : 
  (alarm_home_time - (2 * ((museum_arrival_time - alarm_set_time) + museum_touring_time / 2)) = 12 * 60) :=
  by
    sorry

end correct_alarm_clock_time_l72_72054


namespace primes_with_no_sum_of_two_cubes_l72_72853

theorem primes_with_no_sum_of_two_cubes (p : ℕ) [Fact (Nat.Prime p)] :
  (∃ n : ℤ, ∀ x y : ℤ, x^3 + y^3 ≠ n % p) ↔ p = 7 :=
sorry

end primes_with_no_sum_of_two_cubes_l72_72853


namespace exists_two_factorizations_in_C_another_number_with_property_l72_72596

def in_set_C (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 4 * k + 1

def is_prime_wrt_C (k : ℕ) : Prop :=
  ¬ ∃ a b : ℕ, in_set_C a ∧ in_set_C b ∧ k = a * b

theorem exists_two_factorizations_in_C : 
  ∃ (a b a' b' : ℕ), 
  in_set_C 4389 ∧ 
  in_set_C a ∧ in_set_C b ∧ in_set_C a' ∧ in_set_C b' ∧ 
  (4389 = a * b ∧ 4389 = a' * b') ∧ (a ≠ a' ∨ b ≠ b') :=
sorry

theorem another_number_with_property : 
 ∃ (n a b a' b' : ℕ), 
 n ≠ 4389 ∧ in_set_C n ∧ 
 in_set_C a ∧ in_set_C b ∧ in_set_C a' ∧ in_set_C b' ∧ 
 (n = a * b ∧ n = a' * b') ∧ (a ≠ a' ∨ b ≠ b') :=
sorry

end exists_two_factorizations_in_C_another_number_with_property_l72_72596


namespace periodic_function_of_f_l72_72086

theorem periodic_function_of_f (f : ℝ → ℝ) (c : ℝ) (h : ∀ x, f (x + c) = (2 / (1 + f x)) - 1) : ∀ x, f (x + 2 * c) = f x :=
sorry

end periodic_function_of_f_l72_72086


namespace jessica_carrots_l72_72684

theorem jessica_carrots
  (joan_carrots : ℕ)
  (total_carrots : ℕ)
  (jessica_carrots : ℕ) :
  joan_carrots = 29 →
  total_carrots = 40 →
  jessica_carrots = total_carrots - joan_carrots →
  jessica_carrots = 11 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end jessica_carrots_l72_72684


namespace stateA_issues_more_than_stateB_l72_72884

-- Definitions based on conditions
def stateA_format : ℕ := 26^5 * 10^1
def stateB_format : ℕ := 26^3 * 10^3

-- Proof problem statement
theorem stateA_issues_more_than_stateB : stateA_format - stateB_format = 10123776 := by
  sorry

end stateA_issues_more_than_stateB_l72_72884


namespace money_distribution_l72_72371

theorem money_distribution (a b : ℝ) 
  (h1 : 4 * a - b = 40)
  (h2 : 6 * a + b = 110) :
  a = 15 ∧ b = 20 :=
by
  sorry

end money_distribution_l72_72371


namespace calculate_fg3_l72_72751

def g (x : ℕ) := x^3
def f (x : ℕ) := 3 * x - 2

theorem calculate_fg3 : f (g 3) = 79 :=
by
  sorry

end calculate_fg3_l72_72751


namespace necessary_but_not_sufficient_l72_72160

theorem necessary_but_not_sufficient (x : ℝ) :
  (x < 1 ∨ x > 4) → (x^2 - 3 * x + 2 > 0) ∧ ¬((x^2 - 3 * x + 2 > 0) → (x < 1 ∨ x > 4)) :=
by
  sorry

end necessary_but_not_sufficient_l72_72160


namespace solution_set_of_inequality_l72_72568

variables {R : Type*} [LinearOrderedField R]

def odd_function (f : R → R) : Prop := ∀ x : R, f (-x) = -f x

def increasing_on (f : R → R) (S : Set R) : Prop :=
∀ ⦃x y⦄, x ∈ S → y ∈ S → x < y → f x < f y

theorem solution_set_of_inequality
  {f : R → R}
  (h_odd : odd_function f)
  (h_neg_one : f (-1) = 0)
  (h_increasing : increasing_on f {x : R | x > 0}) :
  {x : R | x * f x > 0} = {x : R | x < -1} ∪ {x : R | x > 1} :=
sorry

end solution_set_of_inequality_l72_72568


namespace speed_of_stream_l72_72181

/-- Given Athul's rowing conditions, prove the speed of the stream is 1 km/h. -/
theorem speed_of_stream 
  (A S : ℝ)
  (h1 : 16 = (A - S) * 4)
  (h2 : 24 = (A + S) * 4) : 
  S = 1 := 
sorry

end speed_of_stream_l72_72181


namespace correct_statements_eq_l72_72888

-- Definitions used in the Lean 4 statement should only directly appear in the conditions
variable {a b c : ℝ} 

-- Use the condition directly
theorem correct_statements_eq (h : a / c = b / c) (hc : c ≠ 0) : a = b := 
by
  -- This is where the proof would go
  sorry

end correct_statements_eq_l72_72888


namespace total_legs_of_all_animals_l72_72994

def num_kangaroos : ℕ := 23
def num_goats : ℕ := 3 * num_kangaroos
def legs_of_kangaroo : ℕ := 2
def legs_of_goat : ℕ := 4

theorem total_legs_of_all_animals : num_kangaroos * legs_of_kangaroo + num_goats * legs_of_goat = 322 :=
by 
  sorry

end total_legs_of_all_animals_l72_72994


namespace gcd_pow_sub_l72_72752

theorem gcd_pow_sub (m n : ℕ) (h₁ : m = 2 ^ 2000 - 1) (h₂ : n = 2 ^ 1990 - 1) :
  Nat.gcd m n = 1023 :=
sorry

end gcd_pow_sub_l72_72752


namespace gcd_of_polynomial_and_multiple_l72_72297

theorem gcd_of_polynomial_and_multiple (b : ℕ) (hb : 714 ∣ b) : 
  Nat.gcd (5 * b^3 + 2 * b^2 + 6 * b + 102) b = 102 := by
  sorry

end gcd_of_polynomial_and_multiple_l72_72297


namespace green_more_than_red_l72_72418

def red_peaches : ℕ := 7
def green_peaches : ℕ := 8

theorem green_more_than_red : green_peaches - red_peaches = 1 := by
  sorry

end green_more_than_red_l72_72418


namespace magic_square_y_value_l72_72039

theorem magic_square_y_value 
  (a b c d e y : ℝ)
  (h1 : y + 4 + c = 81 + a + c)
  (h2 : y + (y - 77) + e = 81 + b + e)
  (h3 : y + 25 + 81 = 4 + (y - 77) + (2 * y - 158)) : 
  y = 168.5 :=
by
  -- required steps to complete the proof
  sorry

end magic_square_y_value_l72_72039


namespace largest_n_exists_l72_72711

theorem largest_n_exists :
  ∃ (n : ℕ), (∃ (x y z : ℕ), n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 3 * x + 3 * y + 3 * z - 8) ∧
    ∀ (m : ℕ), (∃ (x y z : ℕ), m^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 3 * x + 3 * y + 3 * z - 8) →
    n ≥ m :=
  sorry

end largest_n_exists_l72_72711


namespace base2_to_base4_conversion_l72_72735

theorem base2_to_base4_conversion :
  (2 ^ 8 + 2 ^ 6 + 2 ^ 4 + 2 ^ 3 + 2 ^ 2 + 1) = (1 * 4^3 + 1 * 4^2 + 3 * 4^1 + 1 * 4^0) :=
by 
  sorry

end base2_to_base4_conversion_l72_72735


namespace midpoint_sum_coordinates_l72_72723

theorem midpoint_sum_coordinates (x y : ℝ) 
  (midpoint_cond_x : (x + 10) / 2 = 4) 
  (midpoint_cond_y : (y + 4) / 2 = -8) : 
  x + y = -22 :=
by
  sorry

end midpoint_sum_coordinates_l72_72723


namespace aaron_walking_speed_l72_72014

-- Definitions of the conditions
def distance_jog : ℝ := 3 -- in miles
def speed_jog : ℝ := 2 -- in miles/hour
def total_time : ℝ := 3 -- in hours

-- The problem statement
theorem aaron_walking_speed :
  ∃ (v : ℝ), v = (distance_jog / (total_time - (distance_jog / speed_jog))) ∧ v = 2 :=
by
  sorry

end aaron_walking_speed_l72_72014


namespace volume_of_cube_for_tetrahedron_l72_72512

theorem volume_of_cube_for_tetrahedron (h : ℝ) (b1 b2 : ℝ) (V : ℝ) 
  (h_condition : h = 15) (b1_condition : b1 = 8) (b2_condition : b2 = 12)
  (V_condition : V = 3375) : 
  V = (max h (max b1 b2)) ^ 3 := by
  -- To illustrate the mathematical context and avoid concrete steps,
  -- sorry provides the completion of the logical binding to the correct answer
  sorry

end volume_of_cube_for_tetrahedron_l72_72512


namespace problem_p_s_difference_l72_72868

def P : ℤ := 12 - (3 * 4)
def S : ℤ := (12 - 3) * 4

theorem problem_p_s_difference : P - S = -36 := by
  sorry

end problem_p_s_difference_l72_72868


namespace minimum_phi_l72_72687

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 3))
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x)

-- Define the condition for g overlapping with f after shifting by φ
noncomputable def shifted_g (x φ : ℝ) : ℝ := Real.sin (2 * x + 2 * φ)

theorem minimum_phi (φ : ℝ) (h : φ > 0) :
  (∃ (x : ℝ), shifted_g x φ = f x) ↔ (∃ k : ℕ, φ = Real.pi / 6 + k * Real.pi) :=
sorry

end minimum_phi_l72_72687


namespace find_A_l72_72646

noncomputable def telephone_number_satisfies_conditions (A B C D E F G H I J : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ I ∧ A ≠ J ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ I ∧ B ≠ J ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ I ∧ C ≠ J ∧
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ I ∧ D ≠ J ∧
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ I ∧ E ≠ J ∧
  F ≠ G ∧ F ≠ H ∧ F ≠ I ∧ F ≠ J ∧
  G ≠ H ∧ G ≠ I ∧ G ≠ J ∧
  H ≠ I ∧ H ≠ J ∧
  I ≠ J ∧
  A > B ∧ B > C ∧
  D > E ∧ E > F ∧
  G > H ∧ H > I ∧ I > J ∧
  E = D - 2 ∧ F = D - 4 ∧ -- Given D, E, F are consecutive even digits
  H = G - 2 ∧ I = G - 4 ∧ J = G - 6 ∧ -- Given G, H, I, J are consecutive odd digits
  A + B + C = 9

theorem find_A :
  ∃ (A B C D E F G H I J : ℕ), telephone_number_satisfies_conditions A B C D E F G H I J ∧ A = 8 :=
by {
  sorry
}

end find_A_l72_72646


namespace jack_jill_meeting_distance_l72_72768

-- Definitions for Jack's and Jill's initial conditions
def jack_speed_uphill := 12 -- km/hr
def jack_speed_downhill := 15 -- km/hr
def jill_speed_uphill := 14 -- km/hr
def jill_speed_downhill := 18 -- km/hr

def head_start := 0.2 -- hours
def total_distance := 12 -- km
def turn_point_distance := 7 -- km
def return_distance := 5 -- km

-- Statement of the problem to prove the distance from the turning point where they meet
theorem jack_jill_meeting_distance :
  let jack_time_to_turn := (turn_point_distance : ℚ) / jack_speed_uphill
  let jill_time_to_turn := (turn_point_distance : ℚ) / jill_speed_uphill
  let x_meet := (8.95 : ℚ) / 29
  7 - (14 * ((x_meet - 0.2) / 1)) = (772 / 145 : ℚ) := 
sorry

end jack_jill_meeting_distance_l72_72768


namespace min_transfers_to_uniform_cards_l72_72265

theorem min_transfers_to_uniform_cards (n : ℕ) (h : n = 101) (s : Fin n) :
  ∃ k : ℕ, (∀ s1 s2 : Fin n → ℕ, 
    (∀ i, s1 i = i + 1) ∧ (∀ j, s2 j = 51) → -- Initial and final conditions
    k ≤ 42925) := 
sorry

end min_transfers_to_uniform_cards_l72_72265


namespace range_of_k_l72_72587

theorem range_of_k (k : ℝ) : (∀ x : ℝ, x < 3 → x - k < 2 * k) → 1 ≤ k :=
by
  sorry

end range_of_k_l72_72587


namespace part1_part2_l72_72889

-- Part (1)
theorem part1 : -6 * -2 + -5 * 16 = -68 := by
  sorry

-- Part (2)
theorem part2 : -1^4 + (1 / 4) * (2 * -6 - (-4)^2) = -8 := by
  sorry

end part1_part2_l72_72889


namespace weeks_in_semester_l72_72593

-- Define the conditions and the question as a hypothesis
def annie_club_hours : Nat := 13

theorem weeks_in_semester (w : Nat) (h : 13 * (w - 2) = 52) : w = 6 := by
  sorry

end weeks_in_semester_l72_72593


namespace tax_percentage_excess_l72_72295

/--
In Country X, each citizen is taxed an amount equal to 15 percent of the first $40,000 of income,
plus a certain percentage of all income in excess of $40,000. A citizen of Country X is taxed a total of $8,000
and her income is $50,000.

Prove that the percentage of the tax on the income in excess of $40,000 is 20%.
-/
theorem tax_percentage_excess (total_tax : ℝ) (first_income : ℝ) (additional_income : ℝ) (income : ℝ) (tax_first_part : ℝ) (tax_rate_first_part : ℝ) (tax_rate_excess : ℝ) (tax_excess : ℝ) :
  total_tax = 8000 →
  first_income = 40000 →
  additional_income = 10000 →
  income = first_income + additional_income →
  tax_rate_first_part = 0.15 →
  tax_first_part = tax_rate_first_part * first_income →
  tax_excess = total_tax - tax_first_part →
  tax_rate_excess * additional_income = tax_excess →
  tax_rate_excess = 0.20 :=
by
  intro h_total_tax h_first_income h_additional_income h_income h_tax_rate_first_part h_tax_first_part h_tax_excess h_tax_equation
  sorry

end tax_percentage_excess_l72_72295


namespace square_side_length_l72_72552

theorem square_side_length (A : ℝ) (h : A = 625) : ∃ l : ℝ, l^2 = A ∧ l = 25 :=
by {
  sorry
}

end square_side_length_l72_72552


namespace sum_of_digits_18_l72_72915

def distinct_digits (A B C D : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

theorem sum_of_digits_18 (A B C D : ℕ) 
(h1 : A + D = 10)
(h2 : B + C + 1 = 10 + D)
(h3 : C + B + 1 = 10 + B)
(h4 : D + A + 1 = 11)
(h_distinct : distinct_digits A B C D) :
  A + B + C + D = 18 :=
sorry

end sum_of_digits_18_l72_72915


namespace number_of_zeros_at_end_of_factorial_30_l72_72769

-- Lean statement for the equivalence proof problem
def count_factors_of (p n : Nat) : Nat :=
  n / p + n / (p * p) + n / (p * p * p) + n / (p * p * p * p) + n / (p * p * p * p * p)

def zeros_at_end_of_factorial (n : Nat) : Nat :=
  count_factors_of 5 n

theorem number_of_zeros_at_end_of_factorial_30 : zeros_at_end_of_factorial 30 = 7 :=
by 
  sorry

end number_of_zeros_at_end_of_factorial_30_l72_72769


namespace surface_area_LShape_l72_72667

-- Define the structures and conditions
structure UnitCube where
  x : ℕ
  y : ℕ
  z : ℕ

def LShape (cubes : List UnitCube) : Prop :=
  -- Condition 1: Exactly 7 unit cubes
  cubes.length = 7 ∧
  -- Condition 2: 4 cubes in a line along x-axis (bottom row)
  ∃ a b c d : UnitCube, 
    (a.x + 1 = b.x ∧ b.x + 1 = c.x ∧ c.x + 1 = d.x ∧
     a.y = b.y ∧ b.y = c.y ∧ c.y = d.y ∧
     a.z = b.z ∧ b.z = c.z ∧ c.z = d.z) ∧
  -- Condition 3: 3 cubes stacked along z-axis at one end of the row
  ∃ e f g : UnitCube,
    (d.x = e.x ∧ e.x = f.x ∧ f.x = g.x ∧
     d.y = e.y ∧ e.y = f.y ∧ f.y = g.y ∧
     e.z + 1 = f.z ∧ f.z + 1 = g.z)

-- Define the surface area function
def surfaceArea (cubes : List UnitCube) : ℕ :=
  4*7 - 2*3 + 4 -- correct answer calculation according to manual analysis of exposed faces

-- The theorem to be proven
theorem surface_area_LShape : 
  ∀ (cubes : List UnitCube), LShape cubes → surfaceArea cubes = 26 :=
by sorry

end surface_area_LShape_l72_72667


namespace roots_equation_1352_l72_72505

theorem roots_equation_1352 {c d : ℝ} (hc : c^2 - 6 * c + 8 = 0) (hd : d^2 - 6 * d + 8 = 0) :
  c^3 + c^4 * d^2 + c^2 * d^4 + d^3 = 1352 :=
by
  sorry

end roots_equation_1352_l72_72505


namespace trajectory_equation_l72_72387

theorem trajectory_equation (x y : ℝ) (M O A : ℝ × ℝ)
    (hO : O = (0, 0)) (hA : A = (3, 0))
    (h_ratio : dist M O / dist M A = 1 / 2) : 
    x^2 + y^2 + 2 * x - 3 = 0 :=
by
  -- Definition of points
  let M := (x, y)
  exact sorry

end trajectory_equation_l72_72387


namespace rectangle_area_from_square_area_and_proportions_l72_72440

theorem rectangle_area_from_square_area_and_proportions :
  ∃ (a b w : ℕ), a = 16 ∧ b = 3 * w ∧ w = Int.natAbs (Int.sqrt a) ∧ w * b = 48 :=
by
  sorry

end rectangle_area_from_square_area_and_proportions_l72_72440


namespace four_people_pairing_l72_72234

theorem four_people_pairing
    (persons : Fin 4 → Type)
    (common_language : ∀ (i j : Fin 4), Prop)
    (communicable : ∀ (i j k : Fin 4), common_language i j ∨ common_language j k ∨ common_language k i)
    : ∃ (i j : Fin 4) (k l : Fin 4), i ≠ j ∧ k ≠ l ∧ common_language i j ∧ common_language k l := 
sorry

end four_people_pairing_l72_72234


namespace age_problem_l72_72501

theorem age_problem (A B C D E : ℕ)
  (h1 : A = B + 2)
  (h2 : B = 2 * C)
  (h3 : D = C / 2)
  (h4 : E = D - 3)
  (h5 : A + B + C + D + E = 52) : B = 16 :=
by
  sorry

end age_problem_l72_72501


namespace propositionA_necessary_but_not_sufficient_for_propositionB_l72_72545

-- Definitions for propositions and conditions
def propositionA (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0
def propositionB (a : ℝ) : Prop := 0 < a ∧ a < 1

-- Theorem statement for the necessary but not sufficient condition
theorem propositionA_necessary_but_not_sufficient_for_propositionB (a : ℝ) :
  (propositionA a) → (¬ propositionB a) ∧ (propositionB a → propositionA a) :=
by
  sorry

end propositionA_necessary_but_not_sufficient_for_propositionB_l72_72545


namespace polynomial_coefficients_sum_l72_72046

theorem polynomial_coefficients_sum :
  let p := (5 * x^3 - 3 * x^2 + x - 8) * (8 - 3 * x)
  let a := -15
  let b := 49
  let c := -27
  let d := 32
  let e := -64
  16 * a + 8 * b + 4 * c + 2 * d + e = 44 := 
by
  sorry

end polynomial_coefficients_sum_l72_72046


namespace seats_usually_taken_l72_72205

def total_tables : Nat := 15
def seats_per_table : Nat := 10
def proportion_left_unseated : Rat := 1 / 10
def proportion_taken : Rat := 1 - proportion_left_unseated

theorem seats_usually_taken :
  proportion_taken * (total_tables * seats_per_table) = 135 := by
  sorry

end seats_usually_taken_l72_72205


namespace jeff_current_cats_l72_72251

def initial_cats : ℕ := 20
def monday_found_kittens : ℕ := 2 + 3
def monday_stray_cats : ℕ := 4
def tuesday_injured_cats : ℕ := 1
def tuesday_health_issues_cats : ℕ := 2
def tuesday_family_cats : ℕ := 3
def wednesday_adopted_cats : ℕ := 4 * 2
def wednesday_pregnant_cats : ℕ := 2
def thursday_adopted_cats : ℕ := 3
def thursday_donated_cats : ℕ := 3
def friday_adopted_cats : ℕ := 2
def friday_found_cats : ℕ := 3

theorem jeff_current_cats : 
  initial_cats 
  + monday_found_kittens + monday_stray_cats 
  + (tuesday_injured_cats + tuesday_health_issues_cats + tuesday_family_cats)
  + (wednesday_pregnant_cats - wednesday_adopted_cats)
  + (thursday_donated_cats - thursday_adopted_cats)
  + (friday_found_cats - friday_adopted_cats) 
  = 30 := by
  sorry

end jeff_current_cats_l72_72251


namespace find_xy_l72_72047

noncomputable def star (a b c d : ℝ) : ℝ × ℝ :=
  (a * c + b * d, a * d + b * c)

theorem find_xy (a b x y : ℝ) (h : star a b x y = (a, b)) (h' : a^2 ≠ b^2) : (x, y) = (1, 0) :=
  sorry

end find_xy_l72_72047


namespace find_x_y_l72_72747

theorem find_x_y (x y : ℕ) (h1 : 1 ≤ x) (h2 : 1 ≤ y) (h3 : y ≥ x) (h4 : x + y ≤ 20) 
  (h5 : ¬(∃ s, (x * y = s) → x + y = s ∧ ∃ a b : ℕ, a * b = s ∧ a ≠ x ∧ b ≠ y))
  (h6 : ∃ s_t, (x + y = s_t) → x * y = s_t):
  x = 2 ∧ y = 11 :=
by {
  sorry
}

end find_x_y_l72_72747


namespace max_value_a_plus_2b_l72_72208

theorem max_value_a_plus_2b {a b : ℝ} (h_positive : 0 < a ∧ 0 < b) (h_eqn : a^2 + 2 * a * b + 4 * b^2 = 6) :
  a + 2 * b ≤ 2 * Real.sqrt 2 :=
sorry

end max_value_a_plus_2b_l72_72208


namespace polynomial_product_l72_72273

theorem polynomial_product (a b c : ℝ) :
  a * (b - c) ^ 3 + b * (c - a) ^ 3 + c * (a - b) ^ 3 = (a - b) * (b - c) * (c - a) * (a + b + c) :=
by sorry

end polynomial_product_l72_72273


namespace perfect_square_trinomial_l72_72270

theorem perfect_square_trinomial (a b m : ℝ) :
  (∃ x : ℝ, a^2 + mab + b^2 = (x + b)^2 ∨ a^2 + mab + b^2 = (x - b)^2) ↔ (m = 2 ∨ m = -2) :=
by
  sorry

end perfect_square_trinomial_l72_72270


namespace least_possible_integral_QR_l72_72215

theorem least_possible_integral_QR (PQ PR SR SQ QR : ℝ) (hPQ : PQ = 7) (hPR : PR = 10) (hSR : SR = 15) (hSQ : SQ = 24) :
  9 ≤ QR ∧ QR < 17 :=
by
  sorry

end least_possible_integral_QR_l72_72215


namespace average_of_remaining_four_l72_72893

theorem average_of_remaining_four (avg10 : ℕ → ℕ) (avg6 : ℕ → ℕ) 
  (h_avg10 : avg10 10 = 80) 
  (h_avg6 : avg6 6 = 58) : 
  (avg10 10 - avg6 6 * 6) / 4 = 113 :=
sorry

end average_of_remaining_four_l72_72893


namespace unique_positive_integer_solution_l72_72430

theorem unique_positive_integer_solution :
  ∃! (x : ℕ), (4 * x)^2 - 2 * x = 2652 := sorry

end unique_positive_integer_solution_l72_72430


namespace inequality_always_true_l72_72869

theorem inequality_always_true (x : ℝ) : x^2 + 1 ≥ 2 * |x| := 
sorry

end inequality_always_true_l72_72869


namespace jellybean_ratio_l72_72260

theorem jellybean_ratio (gigi_je : ℕ) (rory_je : ℕ) (lorelai_je : ℕ) (h_gigi : gigi_je = 15) (h_rory : rory_je = gigi_je + 30) (h_lorelai : lorelai_je = 180) : lorelai_je / (rory_je + gigi_je) = 3 :=
by
  -- Introduce the given hypotheses
  rw [h_gigi, h_rory, h_lorelai]
  -- Simplify the expression
  sorry

end jellybean_ratio_l72_72260


namespace chloe_total_score_l72_72439

def points_per_treasure : ℕ := 9
def treasures_first_level : ℕ := 6
def treasures_second_level : ℕ := 3

def score_first_level : ℕ := treasures_first_level * points_per_treasure
def score_second_level : ℕ := treasures_second_level * points_per_treasure
def total_score : ℕ := score_first_level + score_second_level

theorem chloe_total_score : total_score = 81 := by
  sorry

end chloe_total_score_l72_72439


namespace find_n_values_l72_72235

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def A_n_k (n k : ℕ) : ℕ := (10^n + 54 * 10^k - 1) / 9

def every_A_n_k_prime (n : ℕ) : Prop :=
  ∀ k, k < n → is_prime (A_n_k n k)

theorem find_n_values :
  ∀ n : ℕ, every_A_n_k_prime n → n = 1 ∨ n = 2 := sorry

end find_n_values_l72_72235


namespace only_a_zero_is_perfect_square_l72_72172

theorem only_a_zero_is_perfect_square (a : ℕ) : (∃ (k : ℕ), a^2 + 2 * a = k^2) → a = 0 := by
  sorry

end only_a_zero_is_perfect_square_l72_72172


namespace range_of_a_for_function_is_real_l72_72634

noncomputable def quadratic_expr (a x : ℝ) : ℝ := a * x^2 - 4 * x + a - 3

theorem range_of_a_for_function_is_real :
  (∀ x : ℝ, quadratic_expr a x > 0) → 0 ≤ a ∧ a ≤ 4 :=
by
  sorry

end range_of_a_for_function_is_real_l72_72634


namespace prasanna_speed_l72_72789

variable (v_L : ℝ) (d t : ℝ)

theorem prasanna_speed (hLaxmiSpeed : v_L = 18) (htime : t = 1) (hdistance : d = 45) : 
  ∃ v_P : ℝ, v_P = 27 :=
  sorry

end prasanna_speed_l72_72789


namespace find_a_if_even_function_l72_72652

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) ^ 2 + a * x + Real.sin (x + Real.pi / 2)

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem find_a_if_even_function :
  (∃ a : ℝ, is_even_function (f a)) → ∃ a : ℝ, a = 2 :=
by
  intros h
  sorry

end find_a_if_even_function_l72_72652


namespace daves_earnings_l72_72310

theorem daves_earnings
  (hourly_wage : ℕ)
  (monday_hours : ℕ)
  (tuesday_hours : ℕ)
  (monday_earning : monday_hours * hourly_wage = 36)
  (tuesday_earning : tuesday_hours * hourly_wage = 12) :
  monday_hours * hourly_wage + tuesday_hours * hourly_wage = 48 :=
by
  sorry

end daves_earnings_l72_72310


namespace area_of_enclosed_figure_l72_72342

noncomputable def area_enclosed_by_curves : ℝ :=
  ∫ (x : ℝ) in (0 : ℝ)..(1 : ℝ), ((x)^(1/2) - x^2)

theorem area_of_enclosed_figure :
  area_enclosed_by_curves = (1 / 3) :=
by
  sorry

end area_of_enclosed_figure_l72_72342


namespace total_goals_by_other_players_l72_72553

theorem total_goals_by_other_players (total_players goals_season games_played : ℕ)
  (third_players_goals avg_goals_per_third : ℕ)
  (h1 : total_players = 24)
  (h2 : goals_season = 150)
  (h3 : games_played = 15)
  (h4 : third_players_goals = total_players / 3)
  (h5 : avg_goals_per_third = 1)
  : (goals_season - (third_players_goals * avg_goals_per_third * games_played)) = 30 :=
by
  sorry

end total_goals_by_other_players_l72_72553


namespace volume_of_regular_tetrahedron_l72_72578

noncomputable def volume_of_tetrahedron (a : ℝ) : ℝ :=
  (a ^ 3 * Real.sqrt 2) / 12

theorem volume_of_regular_tetrahedron (a : ℝ) : 
  volume_of_tetrahedron a = (a ^ 3 * Real.sqrt 2) / 12 := 
by
  sorry

end volume_of_regular_tetrahedron_l72_72578


namespace part_I_solution_set_part_II_range_a_l72_72894

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem part_I_solution_set :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} :=
by
  sorry

theorem part_II_range_a (a : ℝ) :
  (∀ x : ℝ, f x ≥ a^2 - a) ↔ (-1 ≤ a ∧ a ≤ 2) :=
by
  sorry

end part_I_solution_set_part_II_range_a_l72_72894


namespace trucks_on_lot_l72_72499

-- We'll state the conditions as hypotheses and then conclude the total number of trucks.
theorem trucks_on_lot (T : ℕ)
  (h₁ : ∀ N : ℕ, 50 ≤ N ∧ N ≤ 20 → N / 2 = 10)
  (h₂ : T ≥ 20 + 10): T = 30 :=
sorry

end trucks_on_lot_l72_72499


namespace mushroom_picking_l72_72354

theorem mushroom_picking (n T : ℕ) (hn_min : n ≥ 5) (hn_max : n ≤ 7)
  (hmax : ∀ (M_max M_min : ℕ), M_max = T / 5 → M_min = T / 7 → 
    T ≠ 0 → M_max ≤ T / n ∧ M_min ≥ T / n) : n = 6 :=
by
  sorry

end mushroom_picking_l72_72354


namespace find_x_l72_72955

theorem find_x
  (x : ℝ)
  (h1 : (x - 2)^2 + (15 - 5)^2 = 13^2)
  (h2 : x > 0) : 
  x = 2 + Real.sqrt 69 :=
sorry

end find_x_l72_72955


namespace nigella_base_salary_is_3000_l72_72023

noncomputable def nigella_base_salary : ℝ :=
  let house_A_cost := 60000
  let house_B_cost := 3 * house_A_cost
  let house_C_cost := (2 * house_A_cost) - 110000
  let commission_A := 0.02 * house_A_cost
  let commission_B := 0.02 * house_B_cost
  let commission_C := 0.02 * house_C_cost
  let total_earnings := 8000
  let total_commission := commission_A + commission_B + commission_C
  total_earnings - total_commission

theorem nigella_base_salary_is_3000 : 
  nigella_base_salary = 3000 :=
by sorry

end nigella_base_salary_is_3000_l72_72023


namespace arithmetic_sequence_sum_l72_72123

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ)     -- arithmetic sequence
  (d : ℝ)         -- common difference
  (h: ∀ n, a (n + 1) = a n + d)     -- definition of arithmetic sequence
  (h_sum : a 2 + a 4 + a 5 + a 6 + a 8 = 25) : 
  a 2 + a 8 = 10 := 
  sorry

end arithmetic_sequence_sum_l72_72123


namespace boat_speed_in_still_water_l72_72674

def speed_of_boat (V_b : ℝ) : Prop :=
  let stream_speed := 4  -- speed of the stream in km/hr
  let downstream_distance := 168  -- distance traveled downstream in km
  let time := 6  -- time taken to travel downstream in hours
  (downstream_distance = (V_b + stream_speed) * time)

theorem boat_speed_in_still_water : ∃ V_b, speed_of_boat V_b ∧ V_b = 24 := 
by
  exists 24
  unfold speed_of_boat
  simp
  sorry

end boat_speed_in_still_water_l72_72674


namespace distance_between_B_and_C_l72_72672

theorem distance_between_B_and_C
  (A B C : Type)
  (AB : ℝ)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (h_AB : AB = 10)
  (h_angle_A : angle_A = 60)
  (h_angle_B : angle_B = 75) :
  ∃ BC : ℝ, BC = 5 * Real.sqrt 6 :=
by
  sorry

end distance_between_B_and_C_l72_72672


namespace find_cost_of_pencil_and_pen_l72_72738

variable (p q r : ℝ)

-- Definitions based on conditions
def condition1 := 3 * p + 2 * q + r = 4.20
def condition2 := p + 3 * q + 2 * r = 4.75
def condition3 := 2 * r = 3.00

-- The theorem to prove
theorem find_cost_of_pencil_and_pen (p q r : ℝ) (h1 : condition1 p q r) (h2 : condition2 p q r) (h3 : condition3 r) :
  p + q = 1.12 :=
by
  sorry

end find_cost_of_pencil_and_pen_l72_72738


namespace evaluate_expression_l72_72703

theorem evaluate_expression : 1 + 1 / (2 + 1 / (3 + 2)) = 16 / 11 := 
by 
  sorry

end evaluate_expression_l72_72703


namespace gcd_max_two_digits_l72_72983

theorem gcd_max_two_digits (a b : ℕ) (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) (h3 : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 100 :=
by
  -- Definitions
  sorry

end gcd_max_two_digits_l72_72983


namespace minimum_value_f_condition_f_geq_zero_l72_72843

noncomputable def f (x a : ℝ) := Real.exp x - a * x - 1

theorem minimum_value_f (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, f x a ≥ f (Real.log a) a) ∧ f (Real.log a) a = a - a * Real.log a - 1 :=
by 
  sorry

theorem condition_f_geq_zero (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, f x a ≥ 0) ↔ a = 1 :=
by 
  sorry

end minimum_value_f_condition_f_geq_zero_l72_72843


namespace pure_imaginary_condition_fourth_quadrant_condition_l72_72591

theorem pure_imaginary_condition (m : ℝ) (h1: m * (m - 1) = 0) (h2: m ≠ 1) : m = 0 :=
by
  sorry

theorem fourth_quadrant_condition (m : ℝ) (h3: m + 1 > 0) (h4: m^2 - 1 < 0) : -1 < m ∧ m < 1 :=
by
  sorry

end pure_imaginary_condition_fourth_quadrant_condition_l72_72591


namespace Q_ratio_eq_one_l72_72275

noncomputable def g (x : ℂ) : ℂ := x^2007 - 2 * x^2006 + 2

theorem Q_ratio_eq_one (Q : ℂ → ℂ) (s : ℕ → ℂ) (h_root : ∀ j : ℕ, j < 2007 → g (s j) = 0) 
  (h_Q : ∀ j : ℕ, j < 2007 → Q (s j + (1 / s j)) = 0) :
  Q 1 / Q (-1) = 1 := by
  sorry

end Q_ratio_eq_one_l72_72275


namespace second_quadrant_coordinates_l72_72081

theorem second_quadrant_coordinates (x y : ℝ) (h1 : x < 0) (h2 : y > 0) (h3 : |x| = 2) (h4 : y^2 = 1) :
    (x, y) = (-2, 1) :=
  sorry

end second_quadrant_coordinates_l72_72081


namespace arithmetic_sequence_sum_l72_72080

theorem arithmetic_sequence_sum (S : ℕ → ℤ) (a_1 : ℤ) (h1 : a_1 = -2017) 
  (h2 : (S 2009) / 2009 - (S 2007) / 2007 = 2) : 
  S 2017 = -2017 :=
by
  -- definitions and steps would go here
  sorry

end arithmetic_sequence_sum_l72_72080


namespace rectangle_perimeter_l72_72555

theorem rectangle_perimeter (a b : ℕ) (h1 : a ≠ b) (h2 : (a * b) = 4 * (2 * a + 2 * b) - 12) :
    (2 * (a + b) = 72) ∨ (2 * (a + b) = 100) := by
  sorry

end rectangle_perimeter_l72_72555


namespace find_a_if_lines_perpendicular_l72_72137

-- Define the lines and the statement about their perpendicularity
theorem find_a_if_lines_perpendicular 
    (a : ℝ)
    (h_perpendicular : (2 * a) / (3 * (a - 1)) = 1) :
    a = 3 :=
by
  sorry

end find_a_if_lines_perpendicular_l72_72137


namespace train_crosses_signal_post_time_l72_72865

theorem train_crosses_signal_post_time 
  (length_train : ℕ) 
  (length_bridge : ℕ) 
  (time_bridge_minutes : ℕ) 
  (time_signal_post_seconds : ℕ) 
  (h_length_train : length_train = 600) 
  (h_length_bridge : length_bridge = 1800) 
  (h_time_bridge_minutes : time_bridge_minutes = 2) 
  (h_time_signal_post : time_signal_post_seconds = 30) : 
  (length_train / ((length_train + length_bridge) / (time_bridge_minutes * 60))) = time_signal_post_seconds :=
by
  sorry

end train_crosses_signal_post_time_l72_72865


namespace man_l72_72527

theorem man's_age_twice_son_in_2_years 
  (S : ℕ) (M : ℕ) (h1 : S = 18) (h2 : M = 38) (h3 : M = S + 20) : 
  ∃ X : ℕ, (M + X = 2 * (S + X)) ∧ X = 2 :=
by
  sorry

end man_l72_72527


namespace find_20_paise_coins_l72_72282

theorem find_20_paise_coins (x y : ℕ) (h1 : x + y = 324) (h2 : 20 * x + 25 * y = 7100) : x = 200 :=
by
  -- Given the conditions, we need to prove x = 200.
  -- Steps and proofs are omitted here.
  sorry

end find_20_paise_coins_l72_72282


namespace little_john_money_left_l72_72393

-- Define the variables with the given conditions
def initAmount : ℚ := 5.10
def spentOnSweets : ℚ := 1.05
def givenToEachFriend : ℚ := 1.00

-- The problem statement
theorem little_john_money_left :
  (initAmount - spentOnSweets - 2 * givenToEachFriend) = 2.05 :=
by
  sorry

end little_john_money_left_l72_72393


namespace monomial_properties_l72_72935

noncomputable def monomial_coeff : ℚ := -(3/5 : ℚ)

def monomial_degree (x y : ℤ) : ℕ :=
  1 + 2

theorem monomial_properties (x y : ℤ) :
  monomial_coeff = -(3/5) ∧ monomial_degree x y = 3 :=
by
  -- Proof is to be filled here
  sorry

end monomial_properties_l72_72935


namespace probability_after_50_bell_rings_l72_72757

noncomputable def game_probability : ℝ :=
  let p_keep_money := (1 : ℝ) / 4
  let p_give_money := (3 : ℝ) / 4
  let p_same_distribution := p_keep_money^3 + 2 * p_give_money^3
  p_same_distribution^50

theorem probability_after_50_bell_rings : abs (game_probability - 0.002) < 0.01 :=
by
  sorry

end probability_after_50_bell_rings_l72_72757


namespace combined_weight_of_Alexa_and_Katerina_l72_72724

variable (total_weight: ℝ) (alexas_weight: ℝ) (michaels_weight: ℝ)

theorem combined_weight_of_Alexa_and_Katerina
  (h1: total_weight = 154)
  (h2: alexas_weight = 46)
  (h3: michaels_weight = 62) :
  total_weight - michaels_weight = 92 :=
by 
  sorry

end combined_weight_of_Alexa_and_Katerina_l72_72724


namespace child_tickets_sold_l72_72171

theorem child_tickets_sold
  (A C : ℕ) 
  (h1 : A + C = 900)
  (h2 : 7 * A + 4 * C = 5100) :
  C = 400 :=
by
  sorry

end child_tickets_sold_l72_72171


namespace inscribed_rectangle_area_l72_72612

theorem inscribed_rectangle_area (A S x : ℝ) (hA : A = 18) (hS : S = (x * x) * 2) (hx : x = 2):
  S = 8 :=
by
  -- The proofs steps will go here
  sorry

end inscribed_rectangle_area_l72_72612


namespace solve_for_y_l72_72748

theorem solve_for_y (y : ℝ) (h : (7 * y - 2) / (y + 4) - 5 / (y + 4) = 2 / (y + 4)) : 
  y = (9 / 7) :=
by
  sorry

end solve_for_y_l72_72748


namespace smallest_solution_l72_72541

theorem smallest_solution (x : ℝ) (h₁ : x ≥ 0 → x^2 - 3*x - 2 = 0 → x = (3 + Real.sqrt 17) / 2)
                         (h₂ : x < 0 → x^2 + 3*x + 2 = 0 → (x = -1 ∨ x = -2)) :
  x = -2 :=
by
  sorry

end smallest_solution_l72_72541


namespace range_of_b_l72_72564

theorem range_of_b (y : ℝ) (b : ℝ) (h1 : |y - 2| + |y - 5| < b) (h2 : b > 1) : b > 3 := 
sorry

end range_of_b_l72_72564


namespace one_python_can_eat_per_week_l72_72191

-- Definitions based on the given conditions
def burmese_pythons := 5
def alligators_eaten := 15
def weeks := 3

-- Theorem statement to prove the number of alligators one python can eat per week
theorem one_python_can_eat_per_week : (alligators_eaten / burmese_pythons) / weeks = 1 := 
by 
-- sorry is used to skip the actual proof
sorry

end one_python_can_eat_per_week_l72_72191


namespace fraction_addition_l72_72775

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l72_72775


namespace lcm_six_ten_fifteen_is_30_l72_72170

-- Define the numbers and their prime factorizations
def six := 6
def ten := 10
def fifteen := 15

noncomputable def lcm_six_ten_fifteen : ℕ :=
  Nat.lcm (Nat.lcm six ten) fifteen

-- The theorem to prove the LCM
theorem lcm_six_ten_fifteen_is_30 : lcm_six_ten_fifteen = 30 :=
  sorry

end lcm_six_ten_fifteen_is_30_l72_72170


namespace hyperbola_condition_l72_72283

theorem hyperbola_condition (k : ℝ) : 
  (0 ≤ k ∧ k < 3) → (∃ a b : ℝ, a * b < 0 ∧ 
    (a = k + 1) ∧ (b = k - 5)) ∧ (∀ m : ℝ, -1 < m ∧ m < 5 → ∃ a b : ℝ, a * b < 0 ∧ 
    (a = m + 1) ∧ (b = m - 5)) :=
by
  sorry

end hyperbola_condition_l72_72283


namespace is_positive_integer_iff_l72_72795

theorem is_positive_integer_iff (p : ℕ) : 
  (p > 0 → ∃ k : ℕ, (4 * p + 17 = k * (3 * p - 7))) ↔ (3 ≤ p ∧ p ≤ 40) := 
sorry

end is_positive_integer_iff_l72_72795


namespace stickers_earned_correct_l72_72837

-- Define the initial and final number of stickers.
def initial_stickers : ℕ := 39
def final_stickers : ℕ := 61

-- Define how many stickers Pat earned during the week
def stickers_earned : ℕ := final_stickers - initial_stickers

-- State the main theorem
theorem stickers_earned_correct : stickers_earned = 22 :=
by
  show final_stickers - initial_stickers = 22
  sorry

end stickers_earned_correct_l72_72837


namespace solve_for_y_l72_72905

-- The given condition as a hypothesis
variables {x y : ℝ}

-- The theorem statement
theorem solve_for_y (h : 3 * x - y + 5 = 0) : y = 3 * x + 5 :=
sorry

end solve_for_y_l72_72905


namespace add_55_result_l72_72386

theorem add_55_result (x : ℤ) (h : x - 69 = 37) : x + 55 = 161 :=
sorry

end add_55_result_l72_72386


namespace evaluate_dollar_op_l72_72910

def dollar_op (x y : ℤ) := x * (y + 2) + 2 * x * y

theorem evaluate_dollar_op : dollar_op 4 (-1) = -4 :=
by
  -- Proof steps here
  sorry

end evaluate_dollar_op_l72_72910


namespace percentage_of_boys_currently_l72_72678

variables (B G : ℕ)

theorem percentage_of_boys_currently
  (h1 : B + G = 50)
  (h2 : B + 50 = 95) :
  (B * 100) / 50 = 90 :=
by
  sorry

end percentage_of_boys_currently_l72_72678


namespace length_of_diagonal_AC_l72_72478

-- Definitions based on the conditions
variable (AB BC CD DA AC : ℝ)
variable (angle_ADC : ℝ)

-- Conditions
def conditions : Prop :=
  AB = 12 ∧ BC = 12 ∧ CD = 15 ∧ DA = 15 ∧ angle_ADC = 120

theorem length_of_diagonal_AC (h : conditions AB BC CD DA angle_ADC) : AC = 15 :=
sorry

end length_of_diagonal_AC_l72_72478


namespace number_of_female_fish_l72_72470

-- Defining the constants given in the problem
def total_fish : ℕ := 45
def fraction_male : ℚ := 2 / 3

-- The statement we aim to prove in Lean
theorem number_of_female_fish : 
  (total_fish : ℚ) * (1 - fraction_male) = 15 :=
by
  sorry

end number_of_female_fish_l72_72470


namespace find_number_satisfying_equation_l72_72750

theorem find_number_satisfying_equation :
  ∃ x : ℝ, (196 * x^3) / 568 = 43.13380281690141 ∧ x = 5 :=
by
  sorry

end find_number_satisfying_equation_l72_72750


namespace value_of_a_l72_72928

noncomputable def A : Set ℝ := { x | abs x = 1 }
def B (a : ℝ) : Set ℝ := { x | a * x = 1 }
def is_superset (A B : Set ℝ) : Prop := ∀ x, x ∈ B → x ∈ A

theorem value_of_a (a : ℝ) (h : is_superset A (B a)) : a = 1 ∨ a = 0 ∨ a = -1 :=
  sorry

end value_of_a_l72_72928


namespace x_intercept_of_line_is_7_over_2_l72_72444

-- Definitions for the conditions
def point1 : ℝ × ℝ := (2, -3)
def point2 : ℝ × ℝ := (6, 5)

-- Define what it means to be the x-intercept of the line
def x_intercept_of_line (x : ℝ) : Prop :=
  ∃ m b : ℝ, (point1.snd) = m * (point1.fst) + b ∧ (point2.snd) = m * (point2.fst) + b ∧ 0 = m * x + b

-- The theorem stating the x-intercept
theorem x_intercept_of_line_is_7_over_2 : x_intercept_of_line (7 / 2) :=
sorry

end x_intercept_of_line_is_7_over_2_l72_72444


namespace soccer_team_physics_players_l72_72327

-- Define the number of players on the soccer team
def total_players := 15

-- Define the number of players taking mathematics
def math_players := 10

-- Define the number of players taking both mathematics and physics
def both_subjects_players := 4

-- Define the number of players taking physics
def physics_players := total_players - math_players + both_subjects_players

-- The theorem to prove
theorem soccer_team_physics_players : physics_players = 9 :=
by
  -- using the conditions defined above
  sorry

end soccer_team_physics_players_l72_72327


namespace sum_eq_prod_S1_sum_eq_prod_S2_l72_72979

def S1 : List ℕ := [1, 1, 1, 1, 1, 1, 2, 8]
def S2 : List ℕ := [1, 1, 1, 1, 1, 2, 2, 3]

def sum_list (l : List ℕ) : ℕ := l.foldr Nat.add 0
def prod_list (l : List ℕ) : ℕ := l.foldr Nat.mul 1

theorem sum_eq_prod_S1 : sum_list S1 = prod_list S1 := 
by
  sorry

theorem sum_eq_prod_S2 : sum_list S2 = prod_list S2 := 
by
  sorry

end sum_eq_prod_S1_sum_eq_prod_S2_l72_72979


namespace rectangular_prism_parallel_edges_l72_72466

theorem rectangular_prism_parallel_edges (length width height : ℕ) (h1 : length ≠ width) (h2 : width ≠ height) (h3 : length ≠ height) : 
  ∃ pairs : ℕ, pairs = 6 := by
  sorry

end rectangular_prism_parallel_edges_l72_72466


namespace select_and_swap_ways_l72_72821

theorem select_and_swap_ways :
  let n := 8
  let k := 3
  Nat.choose n k * 2 = 112 := 
by
  let n := 8
  let k := 3
  sorry

end select_and_swap_ways_l72_72821


namespace smallest_x_plus_y_l72_72969

theorem smallest_x_plus_y 
  (x y : ℕ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y) 
  (hxy_neq : x ≠ y)
  (h_eq : (1/x + 1/y = 1/10)) : x + y = 45 :=
sorry

end smallest_x_plus_y_l72_72969


namespace find_x_if_vectors_parallel_l72_72226

theorem find_x_if_vectors_parallel (x : ℝ)
  (a : ℝ × ℝ := (x - 1, 2))
  (b : ℝ × ℝ := (2, 1)) :
  (∃ k : ℝ, a = (k * b.1, k * b.2)) → x = 5 :=
by sorry

end find_x_if_vectors_parallel_l72_72226


namespace people_left_is_10_l72_72744

def initial_people : ℕ := 12
def people_joined : ℕ := 15
def final_people : ℕ := 17
def people_left := initial_people - final_people + people_joined

theorem people_left_is_10 : people_left = 10 :=
by sorry

end people_left_is_10_l72_72744


namespace tangent_inclination_point_l72_72680

theorem tangent_inclination_point :
  ∃ a : ℝ, (2 * a = 1) ∧ ((a, a^2) = (1 / 2, 1 / 4)) :=
by
  sorry

end tangent_inclination_point_l72_72680


namespace reflected_light_eq_l72_72895

theorem reflected_light_eq
  (incident_light : ∀ x y : ℝ, 2 * x - y + 6 = 0)
  (reflection_line : ∀ x y : ℝ, y = x) :
  ∃ x y : ℝ, x + 2 * y + 18 = 0 :=
sorry

end reflected_light_eq_l72_72895


namespace union_sets_intersection_complement_l72_72874

open Set

noncomputable def U := (univ : Set ℝ)
def A := { x : ℝ | x ≥ 2 }
def B := { x : ℝ | x < 5 }

theorem union_sets : A ∪ B = univ := by
  sorry

theorem intersection_complement : (U \ A) ∩ B = { x : ℝ | x < 2 } := by
  sorry

end union_sets_intersection_complement_l72_72874


namespace correct_time_fraction_l72_72061

theorem correct_time_fraction : (3 / 4 : ℝ) * (3 / 4 : ℝ) = (9 / 16 : ℝ) :=
by
  sorry

end correct_time_fraction_l72_72061


namespace count_males_not_in_orchestra_l72_72966

variable (females_band females_orchestra females_choir females_all
          males_band males_orchestra males_choir males_all total_students : ℕ)
variable (males_band_not_in_orchestra : ℕ)

theorem count_males_not_in_orchestra :
  females_band = 120 ∧ females_orchestra = 90 ∧ females_choir = 50 ∧ females_all = 30 ∧
  males_band = 90 ∧ males_orchestra = 120 ∧ males_choir = 40 ∧ males_all = 20 ∧
  total_students = 250 ∧ males_band_not_in_orchestra = (males_band - (males_band + males_orchestra + males_choir - males_all - total_students)) 
  → males_band_not_in_orchestra = 20 :=
by
  intros
  sorry

end count_males_not_in_orchestra_l72_72966


namespace almond_croissant_price_l72_72337

theorem almond_croissant_price (R : ℝ) (T : ℝ) (W : ℕ) (total_spent : ℝ) (regular_price : ℝ) (weeks_in_year : ℕ) :
  R = 3.50 →
  T = 468 →
  W = 52 →
  (total_spent = 468) →
  (weekly_regular : ℝ) = 52 * 3.50 →
  (almond_total_cost : ℝ) = (total_spent - weekly_regular) →
  (A : ℝ) = (almond_total_cost / 52) →
  A = 5.50 := by
  intros hR hT hW htotal_spent hweekly_regular halmond_total_cost hA
  sorry

end almond_croissant_price_l72_72337


namespace problem_part_I_problem_part_II_l72_72065

theorem problem_part_I (A B C : ℝ)
  (h1 : 0 < A) 
  (h2 : A < π / 2)
  (h3 : 1 + (Real.sqrt 3 / 3) * Real.sin (2 * A) = 2 * (Real.sin ((B + C) / 2))^2) : 
  A = π / 3 := 
sorry

theorem problem_part_II (A B C R S : ℝ)
  (h1 : A = π / 3)
  (h2 : R = 2 * Real.sqrt 3) 
  (h3 : S = (1 / 2) * (6 * (Real.sin A)) * (Real.sqrt 3 / 2)) :
  S = 9 * Real.sqrt 3 :=
sorry

end problem_part_I_problem_part_II_l72_72065


namespace find_value_at_l72_72528

-- Defining the function f
variable (f : ℝ → ℝ)

-- Conditions
-- Condition 1: f is an odd function
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Condition 2: f has a period of 4
def periodic_function (f : ℝ → ℝ) := ∀ x, f (x + 4) = f x

-- Condition 3: In the interval [0,1], f(x) = 3x
def definition_on_interval (f : ℝ → ℝ) := ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 3 * x

-- Statement to prove
theorem find_value_at (f : ℝ → ℝ) 
  (odd_f : odd_function f) 
  (periodic_f : periodic_function f) 
  (def_on_interval : definition_on_interval f) :
  f 11.5 = -1.5 := by 
  sorry

end find_value_at_l72_72528


namespace work_days_of_a_l72_72569

variable (da wa wb wc : ℕ)
variable (hcp : 3 * wc = 5 * wa)
variable (hbw : 4 * wc = 5 * wb)
variable (hwc : wc = 100)
variable (hear : 60 * da + 9 * 80 + 4 * 100 = 1480)

theorem work_days_of_a : da = 6 :=
by
  sorry

end work_days_of_a_l72_72569


namespace largest_value_of_x_l72_72032

theorem largest_value_of_x (x : ℝ) (h : |x - 8| = 15) : x ≤ 23 :=
by
  sorry -- Proof to be provided

end largest_value_of_x_l72_72032


namespace solve_problem_l72_72332

theorem solve_problem (Δ q : ℝ) (h1 : 2 * Δ + q = 134) (h2 : 2 * (Δ + q) + q = 230) : Δ = 43 := by
  sorry

end solve_problem_l72_72332


namespace pyramid_values_l72_72305

theorem pyramid_values :
  ∃ (A B C D : ℕ),
    (A = 3000) ∧
    (D = 623) ∧
    (B = 700) ∧
    (C = 253) ∧
    (A = 1100 + 1800) ∧
    (D + 451 ≥ 1065) ∧ (D + 451 ≤ 1075) ∧ -- rounding to nearest ten
    (B + 440 ≥ 1050) ∧ (B + 440 ≤ 1150) ∧
    (B + 1070 ≥ 1700) ∧ (B + 1070 ≤ 1900) ∧
    (C + 188 ≥ 430) ∧ (C + 188 ≤ 450) ∧    -- rounding to nearest ten
    (C + 451 ≥ 695) ∧ (C + 451 ≤ 705) :=  -- using B = 700 for rounding range
sorry

end pyramid_values_l72_72305


namespace solve_m_l72_72676

theorem solve_m (x y m : ℝ) (h1 : 4 * x + 2 * y = 3 * m) (h2 : 3 * x + y = m + 2) (h3 : y = -x) : m = 1 := 
by {
  sorry
}

end solve_m_l72_72676


namespace triangle_rectangle_ratio_l72_72302

/--
An equilateral triangle and a rectangle both have perimeters of 60 inches.
The rectangle has a length to width ratio of 2:1.
We need to prove that the ratio of the length of the side of the triangle to
the length of the rectangle is 1.
-/
theorem triangle_rectangle_ratio
  (triangle_perimeter rectangle_perimeter : ℕ)
  (triangle_side rectangle_length rectangle_width : ℕ)
  (h1 : triangle_perimeter = 60)
  (h2 : rectangle_perimeter = 60)
  (h3 : rectangle_length = 2 * rectangle_width)
  (h4 : triangle_side = triangle_perimeter / 3)
  (h5 : rectangle_perimeter = 2 * rectangle_length + 2 * rectangle_width)
  (h6 : rectangle_width = 10)
  (h7 : rectangle_length = 20)
  : triangle_side / rectangle_length = 1 := 
sorry

end triangle_rectangle_ratio_l72_72302


namespace value_of_2a_minus_b_minus_4_l72_72425

theorem value_of_2a_minus_b_minus_4 (a b : ℝ) (h : 2 * a - b = 2) : 2 * a - b - 4 = -2 :=
by
  sorry

end value_of_2a_minus_b_minus_4_l72_72425


namespace not_both_zero_l72_72766

theorem not_both_zero (x y : ℝ) (h : x^2 + y^2 ≠ 0) : ¬ (x = 0 ∧ y = 0) :=
by {
  sorry
}

end not_both_zero_l72_72766


namespace arithmetic_seq_common_difference_l72_72199

theorem arithmetic_seq_common_difference (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 7 * a 11 = 6) (h2 : a 4 + a (14) = 5) : 
  d = 1 / 4 ∨ d = -1 / 4 :=
sorry

end arithmetic_seq_common_difference_l72_72199


namespace rectangle_other_side_length_l72_72408

/-- Theorem: Consider a rectangle with one side of length 10 cm. Another rectangle of dimensions 
10 cm x 1 cm fits diagonally inside this rectangle. We need to prove that the length 
of the other side of the larger rectangle is 2.96 cm. -/
theorem rectangle_other_side_length :
  ∃ (x : ℝ), (x ≠ 0) ∧ (0 < x) ∧ (10 * 10 - x * x = 1 * 1) ∧ x = 2.96 :=
sorry

end rectangle_other_side_length_l72_72408


namespace two_times_koi_minus_X_is_64_l72_72835

-- Definitions based on the conditions
def n : ℕ := 39
def X : ℕ := 14

-- Main proof statement
theorem two_times_koi_minus_X_is_64 : 2 * n - X = 64 :=
by
  sorry

end two_times_koi_minus_X_is_64_l72_72835


namespace foldable_shape_is_axisymmetric_l72_72002

def is_axisymmetric_shape (shape : Type) : Prop :=
  (∃ l : (shape → shape), (∀ x, l x = x))

theorem foldable_shape_is_axisymmetric (shape : Type) (l : shape → shape) 
  (h1 : ∀ x, l x = x) : is_axisymmetric_shape shape := by
  sorry

end foldable_shape_is_axisymmetric_l72_72002


namespace nala_seashells_l72_72686

theorem nala_seashells (a b c : ℕ) (h1 : a = 5) (h2 : b = 7) (h3 : c = 2 * (a + b)) : a + b + c = 36 :=
by {
  sorry
}

end nala_seashells_l72_72686


namespace necessary_but_not_sufficient_condition_l72_72679

theorem necessary_but_not_sufficient_condition
  (a : ℝ)
  (h : ∃ x : ℝ, a * x^2 - 2 * x + 1 < 0) :
  (a < 2 ∧ a < 3) :=
by
  sorry

end necessary_but_not_sufficient_condition_l72_72679


namespace at_least_one_not_greater_than_neg_two_l72_72898

open Real

theorem at_least_one_not_greater_than_neg_two
  {a b c : ℝ} (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  a + (1 / b) ≤ -2 ∨ b + (1 / c) ≤ -2 ∨ c + (1 / a) ≤ -2 :=
sorry

end at_least_one_not_greater_than_neg_two_l72_72898


namespace tile_5x7_rectangle_with_L_trominos_l72_72880

theorem tile_5x7_rectangle_with_L_trominos :
  ∀ k : ℕ, ¬ (∃ (tile : ℕ → ℕ → ℕ), (∀ i j, tile (i+1) (j+1) = tile (i+3) (j+3)) ∧
    ∀ i j, (i < 5 ∧ j < 7) → (tile i j = k)) :=
by sorry

end tile_5x7_rectangle_with_L_trominos_l72_72880


namespace slips_drawn_l72_72973

theorem slips_drawn (P : ℚ) (P_value : P = 24⁻¹) :
  ∃ n : ℕ, (n ≤ 5 ∧ P = (Nat.choose 5 n) / (Nat.choose 10 n) ∧ n = 4) := by
{
  sorry
}

end slips_drawn_l72_72973


namespace portions_of_milk_l72_72316

theorem portions_of_milk (liters_to_ml : ℕ) (total_liters : ℕ) (portion : ℕ) (total_volume_ml : ℕ) (num_portions : ℕ) :
  liters_to_ml = 1000 →
  total_liters = 2 →
  portion = 200 →
  total_volume_ml = total_liters * liters_to_ml →
  num_portions = total_volume_ml / portion →
  num_portions = 10 := by
  sorry

end portions_of_milk_l72_72316


namespace gcd_8251_6105_l72_72153

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 39 := by
  sorry

end gcd_8251_6105_l72_72153


namespace total_percent_decrease_baseball_card_l72_72957

theorem total_percent_decrease_baseball_card
  (original_value : ℝ)
  (first_year_decrease : ℝ := 0.20)
  (second_year_decrease : ℝ := 0.30)
  (value_after_first_year : ℝ := original_value * (1 - first_year_decrease))
  (final_value : ℝ := value_after_first_year * (1 - second_year_decrease))
  (total_percent_decrease : ℝ := ((original_value - final_value) / original_value) * 100) :
  total_percent_decrease = 44 :=
by 
  sorry

end total_percent_decrease_baseball_card_l72_72957


namespace unique_plants_in_all_beds_l72_72749

theorem unique_plants_in_all_beds:
  let A := 600
  let B := 500
  let C := 400
  let D := 300
  let AB := 80
  let AC := 70
  let ABD := 40
  let BC := 0
  let AD := 0
  let BD := 0
  let CD := 0
  let ABC := 0
  let ACD := 0
  let BCD := 0
  let ABCD := 0
  A + B + C + D - AB - AC - BC - AD - BD - CD + ABC + ABD + ACD + BCD - ABCD = 1690 :=
by
  sorry

end unique_plants_in_all_beds_l72_72749


namespace roots_of_equation_l72_72897

theorem roots_of_equation :
  ∀ x : ℝ, x * (x - 1) + 3 * (x - 1) = 0 ↔ x = -3 ∨ x = 1 :=
by {
  sorry
}

end roots_of_equation_l72_72897


namespace hurleys_age_l72_72666

-- Definitions and conditions
variable (H R : ℕ)
variable (cond1 : R - H = 20)
variable (cond2 : (R + 40) + (H + 40) = 128)

-- Theorem statement
theorem hurleys_age (H R : ℕ) (cond1 : R - H = 20) (cond2 : (R + 40) + (H + 40) = 128) : H = 14 := 
by
  sorry

end hurleys_age_l72_72666


namespace solutions_to_deqs_l72_72413

noncomputable def x1 (t : ℝ) : ℝ := -1 / t^2
noncomputable def x2 (t : ℝ) : ℝ := -t * Real.log t

theorem solutions_to_deqs (t : ℝ) (ht : 0 < t) :
  (deriv x1 t = 2 * t * (x1 t)^2) ∧ (deriv x2 t = x2 t / t - 1) :=
by
  sorry

end solutions_to_deqs_l72_72413


namespace platform_length_l72_72103

theorem platform_length (train_length : ℝ) (time_pole : ℝ) (time_platform : ℝ) (speed : ℝ) (platform_length : ℝ) :
  train_length = 300 → time_pole = 18 → time_platform = 38 → speed = train_length / time_pole →
  platform_length = (speed * time_platform) - train_length → platform_length = 333.46 :=
by
  introv h1 h2 h3 h4 h5
  sorry

end platform_length_l72_72103


namespace FC_value_l72_72526

theorem FC_value (DC CB AD FC : ℝ) (h1 : DC = 10) (h2 : CB = 9)
  (h3 : AB = (1 / 3) * AD) (h4 : ED = (3 / 4) * AD) : FC = 13.875 := by
  sorry

end FC_value_l72_72526


namespace imaginary_unit_problem_l72_72857

variable {a b : ℝ}

theorem imaginary_unit_problem (h : i * (a + i) = b + 2 * i) : a + b = 1 :=
sorry

end imaginary_unit_problem_l72_72857


namespace area_of_triangle_ABC_l72_72319

structure Point := (x y : ℝ)

def A := Point.mk 2 3
def B := Point.mk 9 3
def C := Point.mk 4 12

def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * ((B.x - A.x) * (C.y - A.y))

theorem area_of_triangle_ABC :
  area_of_triangle A B C = 31.5 :=
by
  -- Proof is omitted
  sorry

end area_of_triangle_ABC_l72_72319


namespace monotonic_decreasing_interval_l72_72258

open Real

noncomputable def decreasing_interval (k: ℤ): Set ℝ :=
  {x | k * π - π / 3 < x ∧ x < k * π + π / 6 }

theorem monotonic_decreasing_interval (k : ℤ) :
  ∀ x, x ∈ decreasing_interval k ↔ (k * π - π / 3 < x ∧ x < k * π + π / 6) :=
by 
  intros x
  sorry

end monotonic_decreasing_interval_l72_72258


namespace cakes_sold_l72_72525

/-- If a baker made 54 cakes and has 13 cakes left, then the number of cakes he sold is 41. -/
theorem cakes_sold (original_cakes : ℕ) (cakes_left : ℕ) 
  (h1 : original_cakes = 54) (h2 : cakes_left = 13) : 
  original_cakes - cakes_left = 41 := 
by 
  sorry

end cakes_sold_l72_72525


namespace largest_prime_divisor_of_360_is_5_l72_72398

theorem largest_prime_divisor_of_360_is_5 (p : ℕ) (hp₁ : Nat.Prime p) (hp₂ : p ∣ 360) : p ≤ 5 :=
by 
sorry

end largest_prime_divisor_of_360_is_5_l72_72398


namespace largest_of_five_consecutive_sum_l72_72206

theorem largest_of_five_consecutive_sum (n : ℕ) 
  (h : n + (n+1) + (n+2) + (n+3) + (n+4) = 90) : 
  n + 4 = 20 :=
sorry

end largest_of_five_consecutive_sum_l72_72206


namespace friend_gives_30_l72_72463

noncomputable def total_earnings := 10 + 30 + 50 + 40 + 70

noncomputable def equal_share := total_earnings / 5

noncomputable def contribution_of_highest_earner := 70

noncomputable def amount_to_give := contribution_of_highest_earner - equal_share

theorem friend_gives_30 : amount_to_give = 30 := by
  sorry

end friend_gives_30_l72_72463


namespace H_H_H_one_eq_three_l72_72390

noncomputable def H : ℝ → ℝ := sorry

theorem H_H_H_one_eq_three :
  H 1 = -3 ∧ H (-3) = 3 ∧ H 3 = 3 → H (H (H 1)) = 3 :=
by
  sorry

end H_H_H_one_eq_three_l72_72390


namespace maximum_M_l72_72093

-- Define the sides of a triangle condition
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Theorem statement
theorem maximum_M (a b c : ℝ) (h : is_triangle a b c) : 
  (a^2 + b^2) / (c^2) > (1/2) :=
sorry

end maximum_M_l72_72093


namespace find_sum_s_u_l72_72665

theorem find_sum_s_u (p r s u : ℝ) (q t : ℝ) 
  (h_q : q = 5) 
  (h_t : t = -p - r) 
  (h_sum_imaginary : q + s + u = 4) :
  s + u = -1 := 
sorry

end find_sum_s_u_l72_72665


namespace proof_math_problem_l72_72823

-- Define the conditions
structure Conditions where
  person1_start_noon : ℕ -- Person 1 starts from Appleminster at 12:00 PM
  person2_start_2pm : ℕ -- Person 2 starts from Boniham at 2:00 PM
  meet_time : ℕ -- They meet at 4:55 PM
  finish_time_simultaneously : Bool -- They finish their journey simultaneously

-- Define the problem
def math_problem (c : Conditions) : Prop :=
  let arrival_time := 7 * 60 -- 7:00 PM in minutes
  c.person1_start_noon = 0 ∧ -- Noon as 0 minutes (12:00 PM)
  c.person2_start_2pm = 120 ∧ -- 2:00 PM as 120 minutes
  c.meet_time = 295 ∧ -- 4:55 PM as 295 minutes
  c.finish_time_simultaneously = true → arrival_time = 420 -- 7:00 PM in minutes

-- Prove the problem statement, skipping actual proof
theorem proof_math_problem (c : Conditions) : math_problem c :=
  by sorry

end proof_math_problem_l72_72823


namespace downstream_distance_l72_72434

theorem downstream_distance
    (speed_still_water : ℝ)
    (current_rate : ℝ)
    (travel_time_minutes : ℝ)
    (h_still_water : speed_still_water = 20)
    (h_current_rate : current_rate = 4)
    (h_travel_time : travel_time_minutes = 24) :
    (speed_still_water + current_rate) * (travel_time_minutes / 60) = 9.6 :=
by
  -- Proof goes here
  sorry

end downstream_distance_l72_72434


namespace sarahs_score_l72_72929

theorem sarahs_score (g s : ℕ) (h₁ : s = g + 30) (h₂ : (s + g) / 2 = 95) : s = 110 := by
  sorry

end sarahs_score_l72_72929


namespace value_of_x_l72_72438

theorem value_of_x (x : ℝ) : (3 * x + 5) / 7 = 13 → x = 86 / 3 :=
by
  intro h
  sorry

end value_of_x_l72_72438


namespace minimum_a_l72_72078

theorem minimum_a (a b x : ℕ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : b - a = 2013) (h₃ : x > 0) (h₄ : x^2 - a * x + b = 0) : a = 93 :=
by
  sorry

end minimum_a_l72_72078


namespace operation_value_l72_72484

def operation (a b : ℤ) : ℤ := 3 * a - 3 * b + 4

theorem operation_value : operation 6 8 = -2 := by
  sorry

end operation_value_l72_72484


namespace segment_length_l72_72808

theorem segment_length (A B C : ℝ) (hAB : abs (A - B) = 3) (hBC : abs (B - C) = 5) :
  abs (A - C) = 2 ∨ abs (A - C) = 8 := by
  sorry

end segment_length_l72_72808


namespace distance_between_5th_and_29th_red_light_in_feet_l72_72121

-- Define the repeating pattern length and individual light distance
def pattern_length := 7
def red_light_positions := {k | k % pattern_length < 3}
def distance_between_lights := 8 / 12  -- converting inches to feet

-- Positions of the 5th and 29th red lights in terms of pattern repetition
def position_of_nth_red_light (n : ℕ) : ℕ :=
  ((n-1) / 3) * pattern_length + (n-1) % 3 + 1

def position_5th_red_light := position_of_nth_red_light 5
def position_29th_red_light := position_of_nth_red_light 29

theorem distance_between_5th_and_29th_red_light_in_feet :
  (position_29th_red_light - position_5th_red_light - 1) * distance_between_lights = 37 := by
  sorry

end distance_between_5th_and_29th_red_light_in_feet_l72_72121


namespace Robert_salary_loss_l72_72533

-- Define the conditions as hypotheses
variable (S : ℝ) (decrease_percent increase_percent : ℝ)
variable (decrease_percent_eq : decrease_percent = 0.6)
variable (increase_percent_eq : increase_percent = 0.6)

-- Define the problem statement to prove that Robert loses 36% of his salary.
theorem Robert_salary_loss (S : ℝ) (decrease_percent increase_percent : ℝ) 
  (decrease_percent_eq : decrease_percent = 0.6) 
  (increase_percent_eq : increase_percent = 0.6) :
  let new_salary := S * (1 - decrease_percent)
  let increased_salary := new_salary * (1 + increase_percent)
  let loss_percentage := (S - increased_salary) / S * 100 
  loss_percentage = 36 := 
by
  sorry

end Robert_salary_loss_l72_72533


namespace f_2007_l72_72873

def A : Set ℚ := {x : ℚ | x ≠ 0 ∧ x ≠ 1}

noncomputable def f : A → ℝ := sorry

theorem f_2007 :
  (∀ x : ℚ, x ∈ A → f ⟨x, sorry⟩ + f ⟨1 - (1/x), sorry⟩ = Real.log (|x|)) →
  f ⟨2007, sorry⟩ = Real.log (|2007|) :=
sorry

end f_2007_l72_72873


namespace Inez_initial_money_l72_72809

theorem Inez_initial_money (X : ℝ) (h : X - (X / 2 + 50) = 25) : X = 150 :=
by
  sorry

end Inez_initial_money_l72_72809


namespace time_to_pass_platform_l72_72435

-- Definitions based on the conditions
def train_length : ℕ := 1500 -- (meters)
def tree_crossing_time : ℕ := 120 -- (seconds)
def platform_length : ℕ := 500 -- (meters)

-- Define the train's speed
def train_speed := train_length / tree_crossing_time

-- Define the total distance the train needs to cover to pass the platform
def total_distance := train_length + platform_length

-- The proof statement
theorem time_to_pass_platform : 
  total_distance / train_speed = 160 :=
by sorry

end time_to_pass_platform_l72_72435


namespace speed_of_train_l72_72509

-- Define the conditions
def length_of_train : ℕ := 240
def length_of_bridge : ℕ := 150
def time_to_cross : ℕ := 20

-- Compute the expected speed of the train
def expected_speed : ℝ := 19.5

-- The statement that needs to be proven
theorem speed_of_train : (length_of_train + length_of_bridge) / time_to_cross = expected_speed := by
  -- sorry is used to skip the actual proof
  sorry

end speed_of_train_l72_72509


namespace ratio_of_savings_to_earnings_l72_72811

-- Definitions based on the given conditions
def earnings_washing_cars : ℤ := 20
def earnings_walking_dogs : ℤ := 40
def total_savings : ℤ := 150
def months : ℤ := 5

-- Statement to prove the ratio of savings per month to total earnings per month
theorem ratio_of_savings_to_earnings :
  (total_savings / months) = (earnings_washing_cars + earnings_walking_dogs) / 2 := by
  sorry

end ratio_of_savings_to_earnings_l72_72811


namespace daily_reading_goal_l72_72325

-- Define the problem conditions
def total_days : ℕ := 30
def goal_pages : ℕ := 600
def busy_days_13_16 : ℕ := 4
def busy_days_20_25 : ℕ := 6
def flight_day : ℕ := 1
def flight_pages : ℕ := 100

-- Define the mathematical equivalent proof problem in Lean 4
theorem daily_reading_goal :
  (total_days - busy_days_13_16 - busy_days_20_25 - flight_day) * 27 + flight_pages >= goal_pages :=
by
  sorry

end daily_reading_goal_l72_72325


namespace bottom_price_l72_72287

open Nat

theorem bottom_price (B T : ℕ) (h1 : T = B + 300) (h2 : 3 * B + 3 * T = 21000) : B = 3350 := by
  sorry

end bottom_price_l72_72287


namespace max_cookies_andy_can_eat_l72_72198

theorem max_cookies_andy_can_eat (A B C : ℕ) (hB_pos : B > 0) (hC_pos : C > 0) (hB : B ∣ A) (hC : C ∣ A) (h_sum : A + B + C = 36) :
  A ≤ 30 := by
  sorry

end max_cookies_andy_can_eat_l72_72198


namespace uncovered_area_frame_l72_72758

def length_frame : ℕ := 40
def width_frame : ℕ := 32
def length_photo : ℕ := 32
def width_photo : ℕ := 28

def area_frame (l_f w_f : ℕ) : ℕ := l_f * w_f
def area_photo (l_p w_p : ℕ) : ℕ := l_p * w_p

theorem uncovered_area_frame :
  area_frame length_frame width_frame - area_photo length_photo width_photo = 384 :=
by
  sorry

end uncovered_area_frame_l72_72758


namespace customers_left_l72_72159

-- Definitions based on problem conditions
def initial_customers : ℕ := 14
def remaining_customers : ℕ := 3

-- Theorem statement based on the question and the correct answer
theorem customers_left : initial_customers - remaining_customers = 11 := by
  sorry

end customers_left_l72_72159


namespace quadratic_function_positive_l72_72071

theorem quadratic_function_positive (a m : ℝ) (h : a > 0) (h_fm : (m^2 + m + a) < 0) : (m + 1)^2 + (m + 1) + a > 0 :=
by sorry

end quadratic_function_positive_l72_72071


namespace bedroom_light_energy_usage_l72_72355

-- Define the conditions and constants
def noahs_bedroom_light_usage (W : ℕ) : ℕ := W
def noahs_office_light_usage (W : ℕ) : ℕ := 3 * W
def noahs_living_room_light_usage (W : ℕ) : ℕ := 4 * W
def total_energy_used (W : ℕ) : ℕ := 2 * (noahs_bedroom_light_usage W + noahs_office_light_usage W + noahs_living_room_light_usage W)
def energy_consumption := 96

-- The main theorem to be proven
theorem bedroom_light_energy_usage : ∃ W : ℕ, total_energy_used W = energy_consumption ∧ W = 6 :=
by
  sorry

end bedroom_light_energy_usage_l72_72355


namespace find_x_l72_72660

theorem find_x (x : ℝ) : 9999 * x = 724787425 ↔ x = 72487.5 := 
sorry

end find_x_l72_72660


namespace integer_parts_are_divisible_by_17_l72_72877

-- Define that a is the greatest positive root of the given polynomial
def is_greatest_positive_root (a : ℝ) : Prop :=
  (∀ x : ℝ, x^3 - 3 * x^2 + 1 = 0 → x ≤ a) ∧ a > 0 ∧ (a^3 - 3 * a^2 + 1 = 0)

-- Define the main theorem to prove
theorem integer_parts_are_divisible_by_17 (a : ℝ)
  (h_root : is_greatest_positive_root a) :
  (⌊a ^ 1788⌋ % 17 = 0) ∧ (⌊a ^ 1988⌋ % 17 = 0) := 
sorry

end integer_parts_are_divisible_by_17_l72_72877


namespace general_term_arithmetic_sequence_sum_first_n_terms_l72_72992

noncomputable def a_n (n : ℕ) : ℤ :=
  3 * n - 1

def b_n (n : ℕ) (b : ℕ → ℚ) : Prop :=
  (b 1 = 1) ∧ (b 2 = 1 / 3) ∧ ∀ n : ℕ, a_n n * b (n + 1) = n * b n

def sum_b_n (n : ℕ) (b : ℕ → ℚ) : ℚ :=
  (3 / 2) - (1 / (2 * (3 ^ (n - 1))))

theorem general_term_arithmetic_sequence (n : ℕ) :
  a_n n = 3 * n - 1 := by sorry

theorem sum_first_n_terms (n : ℕ) (b : ℕ → ℚ) (h : b_n n b) :
  sum_b_n n b = (3 / 2) - (1 / (2 * (3 ^ (n - 1)))) := by sorry

end general_term_arithmetic_sequence_sum_first_n_terms_l72_72992


namespace supplement_greater_than_complement_l72_72932

variable (angle1 : ℝ)

def is_acute (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

theorem supplement_greater_than_complement (h : is_acute angle1) :
  180 - angle1 = 90 + (90 - angle1) :=
by {
  sorry
}

end supplement_greater_than_complement_l72_72932


namespace max_x_minus_y_l72_72389

theorem max_x_minus_y (x y : ℝ) (h : 2 * (x^3 + y^3) = x + y) : x - y ≤ (Real.sqrt 2 / 2) :=
by {
  sorry
}

end max_x_minus_y_l72_72389


namespace range_of_independent_variable_l72_72219

theorem range_of_independent_variable
  (x : ℝ) 
  (h1 : 2 - 3*x ≥ 0) 
  (h2 : x ≠ 0) 
  : x ≤ 2/3 ∧ x ≠ 0 :=
by 
  sorry

end range_of_independent_variable_l72_72219


namespace point_on_coordinate_axes_l72_72422

-- Definitions and assumptions from the problem conditions
variables {a b : ℝ}

-- The theorem statement asserts that point M(a, b) must be located on the coordinate axes given ab = 0
theorem point_on_coordinate_axes (h : a * b = 0) : 
  (a = 0) ∨ (b = 0) :=
by
  sorry

end point_on_coordinate_axes_l72_72422


namespace employee_n_weekly_wage_l72_72343

theorem employee_n_weekly_wage (Rm Rn : ℝ) (Hm Hn : ℝ) 
    (h1 : (Rm * Hm) + (Rn * Hn) = 770) 
    (h2 : (Rm * Hm) = 1.3 * (Rn * Hn)) :
    Rn * Hn = 335 :=
by
  sorry

end employee_n_weekly_wage_l72_72343


namespace cell_division_relationship_l72_72038

noncomputable def number_of_cells_after_divisions (x : ℕ) : ℕ :=
  2^x

theorem cell_division_relationship (x : ℕ) : 
  number_of_cells_after_divisions x = 2^x := 
by 
  sorry

end cell_division_relationship_l72_72038


namespace simplify_expression_l72_72942

variable (y : ℝ)

theorem simplify_expression : 
  (3 * y - 2) * (5 * y ^ 12 + 3 * y ^ 11 + 5 * y ^ 9 + 3 * y ^ 8) = 
  15 * y ^ 13 - y ^ 12 + 3 * y ^ 11 + 15 * y ^ 10 - y ^ 9 - 6 * y ^ 8 :=
by
  sorry

end simplify_expression_l72_72942


namespace find_multiple_of_A_l72_72813

def shares_division_problem (A B C : ℝ) (x : ℝ) : Prop :=
  C = 160 ∧
  x * A = 5 * B ∧
  x * A = 10 * C ∧
  A + B + C = 880

theorem find_multiple_of_A (A B C x : ℝ) (h : shares_division_problem A B C x) : x = 4 :=
by sorry

end find_multiple_of_A_l72_72813


namespace set_P_equality_l72_72683

open Set

variable {U : Set ℝ} (P : Set ℝ)
variable (h_univ : U = univ) (h_def : P = {x | abs (x - 2) ≥ 1})

theorem set_P_equality : P = {x | x ≥ 3 ∨ x ≤ 1} :=
by
  sorry

end set_P_equality_l72_72683


namespace minimum_value_y_l72_72826

theorem minimum_value_y (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  (∀ x : ℝ, x = (1 / a + 4 / b) → x ≥ 9 / 2) :=
sorry

end minimum_value_y_l72_72826


namespace hours_per_day_l72_72308

variable (M : ℕ)

noncomputable def H : ℕ := 9
noncomputable def D1 : ℕ := 24
noncomputable def Men2 : ℕ := 12
noncomputable def D2 : ℕ := 16

theorem hours_per_day (H_new : ℝ) : 
  (M * H * D1 : ℝ) = (Men2 * H_new * D2) → 
  H_new = (M * 9 : ℝ) / 8 := 
  sorry

end hours_per_day_l72_72308


namespace alan_total_spending_l72_72602

-- Define the conditions
def eggs_bought : ℕ := 20
def price_per_egg : ℕ := 2
def chickens_bought : ℕ := 6
def price_per_chicken : ℕ := 8

-- Total cost calculation
def cost_eggs : ℕ := eggs_bought * price_per_egg
def cost_chickens : ℕ := chickens_bought * price_per_chicken
def total_amount_spent : ℕ := cost_eggs + cost_chickens

-- Prove the total amount spent
theorem alan_total_spending : total_amount_spent = 88 := by
  show cost_eggs + cost_chickens = 88
  sorry

end alan_total_spending_l72_72602


namespace chef_earns_less_than_manager_l72_72233

noncomputable def manager_wage : ℝ := 6.50
noncomputable def dishwasher_wage : ℝ := manager_wage / 2
noncomputable def chef_wage : ℝ := dishwasher_wage + 0.2 * dishwasher_wage

theorem chef_earns_less_than_manager :
  manager_wage - chef_wage = 2.60 :=
by
  sorry

end chef_earns_less_than_manager_l72_72233


namespace circle_radius_l72_72276

/-- Let a circle have a maximum distance of 11 cm and a minimum distance of 5 cm from a point P.
Prove that the radius of the circle can be either 3 cm or 8 cm. -/
theorem circle_radius (max_dist min_dist : ℕ) (h_max : max_dist = 11) (h_min : min_dist = 5) :
  (∃ r : ℕ, r = 3 ∨ r = 8) :=
by
  sorry

end circle_radius_l72_72276


namespace exists_integers_a_b_c_d_and_n_l72_72731

theorem exists_integers_a_b_c_d_and_n (n a b c d : ℕ)
  (h1 : a = 10) 
  (h2 : b = 15) 
  (h3 : c = 8) 
  (h4 : d = 3) 
  (h5 : n = 16) :
  a^4 + b^4 + c^4 + 2 * d^4 = n^4 := by
  -- Proof goes here
  sorry

end exists_integers_a_b_c_d_and_n_l72_72731


namespace boat_travel_distance_downstream_l72_72838

def boat_speed : ℝ := 22 -- Speed of boat in still water in km/hr
def stream_speed : ℝ := 5 -- Speed of the stream in km/hr
def time_downstream : ℝ := 7 -- Time taken to travel downstream in hours
def effective_speed_downstream : ℝ := boat_speed + stream_speed -- Effective speed downstream

theorem boat_travel_distance_downstream : effective_speed_downstream * time_downstream = 189 := by
  -- Since effective_speed_downstream = 27 (22 + 5)
  -- Distance = Speed * Time
  -- Hence, Distance = 27 km/hr * 7 hours = 189 km
  sorry

end boat_travel_distance_downstream_l72_72838


namespace water_level_function_l72_72852

def water_level (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5) : ℝ :=
  0.3 * x + 6

theorem water_level_function :
  ∀ (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5), water_level x h = 6 + 0.3 * x :=
by
  intros
  unfold water_level
  sorry -- Proof skipped

end water_level_function_l72_72852


namespace arithmetic_expression_evaluation_l72_72729

theorem arithmetic_expression_evaluation : 
  (5 * 7 - (3 * 2 + 5 * 4) / 2) = 22 := 
by
  sorry

end arithmetic_expression_evaluation_l72_72729


namespace find_a_l72_72571

open Real

-- Define the circle equation
def circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 5

-- Define the line equation passing through P(2,2)
def line_through_P (m b x y : ℝ) : Prop := y = m * x + b ∧ (2, 2) = (x, y)

-- Define the line equation ax - y + 1 = 0
def perpendicular_line (a x y : ℝ) : Prop := a * x - y + 1 = 0

theorem find_a : ∃ a : ℝ, ∀ x y m b : ℝ,
    circle x y ∧ line_through_P m b x y ∧
    (line_through_P m b x y → perpendicular_line a x y) → a = 2 :=
by
  intros
  sorry

end find_a_l72_72571


namespace arithmetic_series_sum_proof_middle_term_proof_l72_72108

def arithmetic_series_sum (a d n : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

def middle_term (a l : ℤ) : ℤ :=
  (a + l) / 2

theorem arithmetic_series_sum_proof :
  let a := -51
  let d := 2
  let n := 27
  let l := 1
  arithmetic_series_sum a d n = -675 :=
by
  sorry

theorem middle_term_proof :
  let a := -51
  let l := 1
  middle_term a l = -25 :=
by
  sorry

end arithmetic_series_sum_proof_middle_term_proof_l72_72108


namespace solution_set_f_div_x_lt_zero_l72_72996

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_f_div_x_lt_zero :
  (∀ x, f (2 + (2 - x)) = f x) ∧
  (∀ x1 x2 : ℝ, x1 < 2 ∧ x2 < 2 ∧ x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) < 0) ∧
  f 4 = 0 →
  { x : ℝ | f x / x < 0 } = { x | x < 0 } ∪ { x | 0 < x ∧ x < 4 } :=
sorry

end solution_set_f_div_x_lt_zero_l72_72996


namespace base4_base9_digit_difference_l72_72485

theorem base4_base9_digit_difference (n : ℕ) (h1 : n = 523) (h2 : ∀ (k : ℕ), 4^(k - 1) ≤ n -> n < 4^k -> k = 5)
  (h3 : ∀ (k : ℕ), 9^(k - 1) ≤ n -> n < 9^k -> k = 3) : (5 - 3 = 2) :=
by
  -- Let's provide our specific instantiations for h2 and h3
  have base4_digits := h2 5;
  have base9_digits := h3 3;
  -- Clear sorry
  rfl

end base4_base9_digit_difference_l72_72485


namespace average_stickers_per_pack_l72_72688

-- Define the conditions given in the problem
def pack1 := 5
def pack2 := 7
def pack3 := 7
def pack4 := 10
def pack5 := 11
def num_packs := 5
def total_stickers := pack1 + pack2 + pack3 + pack4 + pack5

-- Statement to prove the average number of stickers per pack
theorem average_stickers_per_pack :
  (total_stickers / num_packs) = 8 := by
  sorry

end average_stickers_per_pack_l72_72688


namespace tom_change_l72_72610

theorem tom_change :
  let SNES_value := 150
  let credit_percent := 0.80
  let amount_given := 80
  let game_value := 30
  let NES_sale_price := 160
  let credit_for_SNES := credit_percent * SNES_value
  let amount_to_pay_for_NES := NES_sale_price - credit_for_SNES
  let effective_amount_paid := amount_to_pay_for_NES - game_value
  let change_received := amount_given - effective_amount_paid
  change_received = 70 :=
by
  sorry

end tom_change_l72_72610


namespace bryden_receives_amount_l72_72203

variable (q : ℝ) (p : ℝ) (num_quarters : ℝ)

-- Define the conditions
def face_value_of_quarter : Prop := q = 0.25
def percentage_offer : Prop := p = 25 * q
def number_of_quarters : Prop := num_quarters = 5

-- Define the theorem to be proved
theorem bryden_receives_amount (h1 : face_value_of_quarter q) (h2 : percentage_offer q p) (h3 : number_of_quarters num_quarters) :
  (p * num_quarters * q) = 31.25 :=
by
  sorry

end bryden_receives_amount_l72_72203


namespace mowing_work_rate_l72_72906

variables (A B C : ℚ)

theorem mowing_work_rate :
  A + B = 1/28 → A + B + C = 1/21 → C = 1/84 :=
by
  intros h1 h2
  sorry

end mowing_work_rate_l72_72906


namespace bob_raise_per_hour_l72_72637

theorem bob_raise_per_hour
  (hours_per_week : ℕ := 40)
  (monthly_housing_reduction : ℤ := 60)
  (weekly_earnings_increase : ℤ := 5)
  (weeks_per_month : ℕ := 4) :
  ∃ (R : ℚ), 40 * R - (monthly_housing_reduction / weeks_per_month) + weekly_earnings_increase = 0 ∧
              R = 0.25 := 
by
  sorry

end bob_raise_per_hour_l72_72637


namespace sequence_term_l72_72252

theorem sequence_term (x : ℕ → ℝ)
  (h₀ : ∀ n ≥ 2, 2 / x n = 1 / x (n - 1) + 1 / x (n + 1))
  (h₁ : x 2 = 2 / 3)
  (h₂ : x 4 = 2 / 5) :
  x 10 = 2 / 11 := 
sorry

end sequence_term_l72_72252


namespace perpendicular_condition_centroid_coordinates_l72_72583

structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := -1, y := 0}
def B : Point := {x := 4, y := 0}
def C (c : ℝ) : Point := {x := 0, y := c}

def vec (P Q : Point) : Point :=
  {x := Q.x - P.x, y := Q.y - P.y}

def dot_product (P Q : Point) : ℝ :=
  P.x * Q.x + P.y * Q.y

theorem perpendicular_condition (c : ℝ) (h : dot_product (vec A (C c)) (vec B (C c)) = 0) :
  c = 2 ∨ c = -2 :=
by
  -- proof to be filled in
  sorry

theorem centroid_coordinates (c : ℝ) (h : c = 2 ∨ c = -2) :
  (c = 2 → Point.mk 1 (2 / 3) = Point.mk 1 (2 / 3)) ∧
  (c = -2 → Point.mk 1 (-2 / 3) = Point.mk 1 (-2 / 3)) :=
by
  -- proof to be filled in
  sorry

end perpendicular_condition_centroid_coordinates_l72_72583


namespace triangle_perimeter_l72_72513

theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ) (h1 : a = 3) (h2 : b = 3) 
    (h3 : c^2 = a * Real.cos B + b * Real.cos A) : 
    a + b + c = 7 :=
by 
  sorry

end triangle_perimeter_l72_72513


namespace problem_equivalent_l72_72262

variable (f : ℝ → ℝ)

theorem problem_equivalent (h₁ : ∀ x, deriv f x = deriv (deriv f) x)
                            (h₂ : ∀ x, deriv (deriv f) x < f x) : 
                            f 2 < Real.exp 2 * f 0 ∧ f 2017 < Real.exp 2017 * f 0 := sorry

end problem_equivalent_l72_72262


namespace number_added_is_10_l72_72822

theorem number_added_is_10 (x y a : ℕ) (h1 : y = 40) 
  (h2 : x * 4 = 3 * y) 
  (h3 : (x + a) * 5 = 4 * (y + a)) : a = 10 := 
by
  sorry

end number_added_is_10_l72_72822


namespace train_length_proof_l72_72641

noncomputable def train_length (speed_kmh : ℕ) (time_s : ℕ) : ℕ :=
  let speed_ms := speed_kmh * 5 / 18
  speed_ms * time_s

theorem train_length_proof : train_length 144 16 = 640 := by
  sorry

end train_length_proof_l72_72641


namespace mapleton_math_team_combinations_l72_72147

open Nat

theorem mapleton_math_team_combinations (girls boys : ℕ) (team_size girl_on_team boy_on_team : ℕ)
    (h_girls : girls = 4) (h_boys : boys = 5) (h_team_size : team_size = 4)
    (h_girl_on_team : girl_on_team = 3) (h_boy_on_team : boy_on_team = 1) :
    (Nat.choose girls girl_on_team) * (Nat.choose boys boy_on_team) = 20 := by
  sorry

end mapleton_math_team_combinations_l72_72147


namespace math_proof_problem_l72_72779

noncomputable def expr : ℚ :=
  ((5 / 8 * (3 / 7) + 1 / 4 * (2 / 6)) - (2 / 3 * (1 / 4) - 1 / 5 * (4 / 9))) * 
  ((7 / 9 * (2 / 5) * (1 / 2) * 5040 + 1 / 3 * (3 / 8) * (9 / 11) * 4230))

theorem math_proof_problem : expr = 336 := 
  by
  sorry

end math_proof_problem_l72_72779


namespace simplify_vectors_l72_72657

variables {Point : Type} [AddGroup Point] (A B C D : Point)

def vector (P Q : Point) : Point := Q - P

theorem simplify_vectors :
  vector A B + vector B C - vector A D = vector D C :=
by
  sorry

end simplify_vectors_l72_72657


namespace factorial_quotient_52_50_l72_72314

theorem factorial_quotient_52_50 : (Nat.factorial 52) / (Nat.factorial 50) = 2652 := 
by 
  sorry

end factorial_quotient_52_50_l72_72314


namespace triangle_angle_l72_72084

variable (a b c : ℝ)
variable (C : ℝ)

theorem triangle_angle (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : (a^2 + b^2) * (a^2 + b^2 - c^2) = 3 * a^2 * b^2) :
  C = Real.arccos ((a^4 + b^4 - a^2 * b^2) / (2 * a * b * (a^2 + b^2))) :=
sorry

end triangle_angle_l72_72084


namespace child_support_calculation_l72_72469

noncomputable def owed_child_support (yearly_salary : ℕ) (raise_pct: ℝ) 
(raise_years_additional_salary: ℕ) (payment_percentage: ℝ) 
(payment_years_salary_before_raise: ℕ) (already_paid : ℝ) : ℝ :=
  let initial_salary := yearly_salary * payment_years_salary_before_raise
  let increase_amount := yearly_salary * raise_pct
  let new_salary := yearly_salary + increase_amount
  let salary_after_raise := new_salary * raise_years_additional_salary
  let total_income := initial_salary + salary_after_raise
  let total_support_due := total_income * payment_percentage
  total_support_due - already_paid

theorem child_support_calculation:
  owed_child_support 30000 0.2 4 0.3 3 1200 = 69000 :=
by
  sorry

end child_support_calculation_l72_72469


namespace gwen_total_books_l72_72229

theorem gwen_total_books
  (mystery_shelves : ℕ) (picture_shelves : ℕ) (books_per_shelf : ℕ)
  (mystery_shelves_count : mystery_shelves = 3)
  (picture_shelves_count : picture_shelves = 5)
  (each_shelf_books : books_per_shelf = 9) :
  (mystery_shelves * books_per_shelf + picture_shelves * books_per_shelf) = 72 := by
  sorry

end gwen_total_books_l72_72229


namespace line_through_point_parallel_to_given_line_line_through_point_sum_intercepts_is_minus_four_l72_72570

theorem line_through_point_parallel_to_given_line :
  ∃ c : ℤ, (∀ x y : ℤ, 2 * x + 3 * y + c = 0 ↔ (x, y) = (2, 1)) ∧ c = -7 :=
sorry

theorem line_through_point_sum_intercepts_is_minus_four :
  ∃ (a b : ℤ), (∀ x y : ℤ, (x / a) + (y / b) = 1 ↔ (x, y) = (-3, 1)) ∧ (a + b = -4) ∧ 
  ((a = -6 ∧ b = 2) ∨ (a = -2 ∧ b = -2)) ∧ 
  ((∀ x y : ℤ, x - 3 * y + 6 = 0 ↔ (x, y) = (-3, 1)) ∨ 
  (∀ x y : ℤ, x + y + 2 = 0 ↔ (x, y) = (-3, 1))) :=
sorry

end line_through_point_parallel_to_given_line_line_through_point_sum_intercepts_is_minus_four_l72_72570


namespace problem_solution_l72_72019

theorem problem_solution :
  20 * ((180 / 3) + (40 / 5) + (16 / 32) + 2) = 1410 := by
  sorry

end problem_solution_l72_72019


namespace principal_sum_investment_l72_72072

theorem principal_sum_investment 
    (P R : ℝ) 
    (h1 : (P * 5 * (R + 2)) / 100 - (P * 5 * R) / 100 = 180)
    (h2 : (P * 5 * (R + 3)) / 100 - (P * 5 * R) / 100 = 270) :
    P = 1800 :=
by
  -- These are the hypotheses generated for Lean, the proof steps are omitted
  sorry

end principal_sum_investment_l72_72072


namespace percentage_increase_efficiency_l72_72732

-- Defining the times taken by Sakshi and Tanya
def sakshi_time : ℕ := 12
def tanya_time : ℕ := 10

-- Defining the efficiency in terms of work per day for Sakshi and Tanya
def sakshi_efficiency : ℚ := 1 / sakshi_time
def tanya_efficiency : ℚ := 1 / tanya_time

-- The statement of the proof: percentage increase
theorem percentage_increase_efficiency : 
  100 * ((tanya_efficiency - sakshi_efficiency) / sakshi_efficiency) = 20 := 
by
  -- The actual proof will go here
  sorry

end percentage_increase_efficiency_l72_72732


namespace find_a_b_l72_72374

theorem find_a_b (a b : ℝ) : 
  (∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - a * x - b < 0) → a = 5 ∧ b = -6 :=
sorry

end find_a_b_l72_72374


namespace min_value_frac_eq_nine_halves_l72_72238

theorem min_value_frac_eq_nine_halves {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 2*x + y = 2) :
  ∃ (x y : ℝ), 2 / x + 1 / y = 9 / 2 := by
  sorry

end min_value_frac_eq_nine_halves_l72_72238


namespace find_m_l72_72341

noncomputable def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2*k + 1

theorem find_m (m : ℕ) (h₀ : 0 < m) (h₁ : (m ^ 2 - 2 * m - 3:ℤ) < 0) (h₂ : is_odd (m ^ 2 - 2 * m - 3)) : m = 2 := 
sorry

end find_m_l72_72341


namespace sum_of_roots_quadratic_eq_l72_72522

theorem sum_of_roots_quadratic_eq : ∀ P Q : ℝ, (3 * P^2 - 9 * P + 6 = 0) ∧ (3 * Q^2 - 9 * Q + 6 = 0) → P + Q = 3 :=
by
  sorry

end sum_of_roots_quadratic_eq_l72_72522


namespace extra_marks_15_l72_72003

theorem extra_marks_15 {T P : ℝ} (h1 : 0.30 * T = P - 30) (h2 : 0.45 * T = P + 15) (h3 : P = 120) : 
  0.45 * T - P = 15 := 
by
  sorry

end extra_marks_15_l72_72003


namespace polyhedron_with_12_edges_l72_72912

def prism_edges (n : Nat) : Nat :=
  3 * n

def pyramid_edges (n : Nat) : Nat :=
  2 * n

def Quadrangular_prism : Nat := prism_edges 4
def Quadrangular_pyramid : Nat := pyramid_edges 4
def Pentagonal_pyramid : Nat := pyramid_edges 5
def Pentagonal_prism : Nat := prism_edges 5

theorem polyhedron_with_12_edges :
  (Quadrangular_prism = 12) ∧
  (Quadrangular_pyramid ≠ 12) ∧
  (Pentagonal_pyramid ≠ 12) ∧
  (Pentagonal_prism ≠ 12) := by
  sorry

end polyhedron_with_12_edges_l72_72912


namespace polygon_interior_plus_exterior_l72_72625

theorem polygon_interior_plus_exterior (n : ℕ) 
  (h : (n - 2) * 180 + 60 = 1500) : n = 10 :=
sorry

end polygon_interior_plus_exterior_l72_72625


namespace relation_between_u_and_v_l72_72588

def diameter_circle_condition (AB : ℝ) (r : ℝ) : Prop := AB = 2*r
def chord_tangent_condition (AD BC CD : ℝ) (r : ℝ) : Prop := 
  AD + BC = 2*r ∧ CD*CD = (2*r)*(AD + BC)
def point_selection_condition (AD AF CD : ℝ) : Prop := AD = AF + CD

theorem relation_between_u_and_v (AB AD AF BC CD u v r: ℝ)
  (h1: diameter_circle_condition AB r)
  (h2: chord_tangent_condition AD BC CD r)
  (h3: point_selection_condition AD AF CD)
  (h4: u = AF)
  (h5: v^2 = r^2):
  v^2 = u^3 / (2*r - u) := by
  sorry

end relation_between_u_and_v_l72_72588


namespace card_tag_sum_l72_72097

noncomputable def W : ℕ := 200
noncomputable def X : ℝ := 2 / 3 * W
noncomputable def Y : ℝ := W + X
noncomputable def Z : ℝ := Real.sqrt Y
noncomputable def P : ℝ := X^3
noncomputable def Q : ℝ := Nat.factorial W / 100000
noncomputable def R : ℝ := 3 / 5 * (P + Q)
noncomputable def S : ℝ := W^1 + X^2 + Z^3

theorem card_tag_sum :
  W + X + Y + Z + P + S = 2373589.26 + Q + R :=
by
  sorry

end card_tag_sum_l72_72097


namespace least_students_with_brown_eyes_and_lunch_box_l72_72213

variable (U : Finset ℕ) (B L : Finset ℕ)
variables (hU : U.card = 25) (hB : B.card = 15) (hL : L.card = 18)

theorem least_students_with_brown_eyes_and_lunch_box : 
  (B ∩ L).card ≥ 8 := by
  sorry

end least_students_with_brown_eyes_and_lunch_box_l72_72213


namespace probability_odd_sum_of_6_balls_drawn_l72_72701

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_odd_sum_of_6_balls_drawn :
  let n := 11
  let k := 6
  let total_ways := binom n k
  let odd_count := 6
  let even_count := 5
  let cases := 
    (binom odd_count 1 * binom even_count (k - 1)) +
    (binom odd_count 3 * binom even_count (k - 3)) +
    (binom odd_count 5 * binom even_count (k - 5))
  let favorable_outcomes := cases
  let probability := favorable_outcomes / total_ways
  probability = 118 / 231 := 
by {
  sorry
}

end probability_odd_sum_of_6_balls_drawn_l72_72701


namespace lily_disproves_tom_claim_l72_72918

-- Define the cards and the claim
inductive Card
| A : Card
| R : Card
| Circle : Card
| Square : Card
| Triangle : Card

def has_consonant (c : Card) : Prop :=
  match c with
  | Card.R => true
  | _ => false

def has_triangle (c : Card) : Card → Prop :=
  fun c' =>
    match c with
    | Card.R => c' = Card.Triangle
    | _ => true

def tom_claim (c : Card) (c' : Card) : Prop :=
  has_consonant c → has_triangle c c'

-- Proof problem statement:
theorem lily_disproves_tom_claim (c : Card) (c' : Card) : c = Card.R → ¬ has_triangle c c' → ¬ tom_claim c c' :=
by
  intros
  sorry

end lily_disproves_tom_claim_l72_72918


namespace sum_of_solutions_eq_35_over_3_l72_72960

theorem sum_of_solutions_eq_35_over_3 (a b : ℝ) 
  (h1 : 2 * a + b = 14) (h2 : a + 2 * b = 21) : 
  a + b = 35 / 3 := 
by
  sorry

end sum_of_solutions_eq_35_over_3_l72_72960


namespace company_max_revenue_l72_72246

structure Conditions where
  max_total_time : ℕ -- maximum total time in minutes
  max_total_cost : ℕ -- maximum total cost in yuan
  rate_A : ℕ -- rate per minute for TV A in yuan
  rate_B : ℕ -- rate per minute for TV B in yuan
  revenue_A : ℕ -- revenue per minute for TV A in million yuan
  revenue_B : ℕ -- revenue per minute for TV B in million yuan

def company_conditions : Conditions :=
  { max_total_time := 300,
    max_total_cost := 90000,
    rate_A := 500,
    rate_B := 200,
    revenue_A := 3, -- as 0.3 million yuan converted to 3 tenths (integer representation)
    revenue_B := 2  -- as 0.2 million yuan converted to 2 tenths (integer representation)
  }

def advertising_strategy
  (conditions : Conditions)
  (time_A : ℕ) (time_B : ℕ) : Prop :=
  time_A + time_B ≤ conditions.max_total_time ∧
  time_A * conditions.rate_A + time_B * conditions.rate_B ≤ conditions.max_total_cost

def revenue
  (conditions : Conditions)
  (time_A : ℕ) (time_B : ℕ) : ℕ :=
  time_A * conditions.revenue_A + time_B * conditions.revenue_B

theorem company_max_revenue (time_A time_B : ℕ)
  (h : advertising_strategy company_conditions time_A time_B) :
  revenue company_conditions time_A time_B = 70 := 
  by
  have h1 : time_A = 100 := sorry
  have h2 : time_B = 200 := sorry
  sorry

end company_max_revenue_l72_72246


namespace additional_men_joined_l72_72987

noncomputable def solve_problem := 
  let M := 1000
  let days_initial := 17
  let days_new := 11.333333333333334
  let total_provisions := M * days_initial
  let additional_men := (total_provisions / days_new) - M
  additional_men

theorem additional_men_joined : solve_problem = 500 := by
  sorry

end additional_men_joined_l72_72987


namespace equal_numbers_l72_72146

namespace MathProblem

theorem equal_numbers 
  (x y z : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (h : x^2 / y + y^2 / z + z^2 / x = x^2 / z + z^2 / y + y^2 / x) : 
  x = y ∨ x = z ∨ y = z :=
by
  sorry

end MathProblem

end equal_numbers_l72_72146


namespace range_of_a_l72_72681

theorem range_of_a (x a : ℝ) (hp : x^2 + 2 * x - 3 > 0) (hq : x > a)
  (h_suff : x^2 + 2 * x - 3 > 0 → ¬ (x > a)):
  a ≥ 1 := 
by
  sorry

end range_of_a_l72_72681


namespace solution_set_l72_72256

def inequality_solution (x : ℝ) : Prop :=
  4 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 9

theorem solution_set :
  { x : ℝ | inequality_solution x } = { x : ℝ | (63 / 26 : ℝ) < x ∧ x ≤ (28 / 11 : ℝ) } :=
by
  sorry

end solution_set_l72_72256


namespace pencil_distribution_l72_72382

theorem pencil_distribution (C C' : ℕ) (pencils : ℕ) (remaining : ℕ) (less_per_class : ℕ) 
  (original_classes : C = 4) 
  (total_pencils : pencils = 172) 
  (remaining_pencils : remaining = 7) 
  (less_pencils : less_per_class = 28)
  (actual_classes : C' > C) 
  (distribution_mistake : (pencils - remaining) / C' + less_per_class = pencils / C) :
  C' = 11 := 
sorry

end pencil_distribution_l72_72382


namespace pyramid_surface_area_l72_72537

-- Definitions based on conditions
def upper_base_edge_length : ℝ := 2
def lower_base_edge_length : ℝ := 4
def side_edge_length : ℝ := 2

-- Problem statement in Lean
theorem pyramid_surface_area :
  let slant_height := Real.sqrt ((side_edge_length ^ 2) - (1 ^ 2))
  let perimeter_base := (4 * upper_base_edge_length) + (4 * lower_base_edge_length)
  let lsa := (perimeter_base * slant_height) / 2
  let total_surface_area := lsa + (upper_base_edge_length ^ 2) + (lower_base_edge_length ^ 2)
  total_surface_area = 10 * Real.sqrt 3 + 20 := sorry

end pyramid_surface_area_l72_72537


namespace smallest_sum_xy_l72_72293

theorem smallest_sum_xy (x y : ℕ) (hx : x ≠ y) (h : 0 < x ∧ 0 < y) (hxy : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 15) :
  x + y = 64 :=
sorry

end smallest_sum_xy_l72_72293


namespace count_three_digit_perfect_squares_l72_72197

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem count_three_digit_perfect_squares : 
  ∃ (count : ℕ), count = 22 ∧
  ∀ (n : ℕ), is_three_digit_number n → is_perfect_square n → true :=
sorry

end count_three_digit_perfect_squares_l72_72197


namespace converse_even_power_divisible_l72_72249

theorem converse_even_power_divisible (n : ℕ) (h_even : ∀ (k : ℕ), n = 2 * k → (3^n + 63) % 72 = 0) :
  (3^n + 63) % 72 = 0 → ∃ (k : ℕ), n = 2 * k :=
by sorry

end converse_even_power_divisible_l72_72249


namespace fraction_of_area_in_triangle_l72_72157

theorem fraction_of_area_in_triangle :
  let vertex1 := (3, 3)
  let vertex2 := (5, 5)
  let vertex3 := (3, 5)
  let base := (5 - 3)
  let height := (5 - 3)
  let area_triangle := (1 / 2) * base * height
  let area_square := 6 * 6
  let fraction := area_triangle / area_square
  fraction = (1 / 18) :=
by 
  sorry

end fraction_of_area_in_triangle_l72_72157


namespace max_value_fraction_l72_72462

theorem max_value_fraction (x : ℝ) : x ≠ 0 → 1 / (x^4 + 4*x^2 + 2 + 8/x^2 + 16/x^4) ≤ 1 / 31 :=
by sorry

end max_value_fraction_l72_72462


namespace clock_correct_time_fraction_l72_72029

/-- 
  A 24-hour digital clock displays the hour and minute of a day, 
  counting from 00:00 to 23:59. However, due to a glitch, whenever 
  the clock is supposed to display a '2', it mistakenly displays a '5'.

  Prove that the fraction of a day during which the clock shows the correct 
  time is 23/40.
-/
theorem clock_correct_time_fraction :
  let total_hours := 24
  let affected_hours := 6
  let correct_hours := total_hours - affected_hours
  let total_minutes := 60
  let affected_minutes := 14
  let correct_minutes := total_minutes - affected_minutes
  (correct_hours / total_hours) * (correct_minutes / total_minutes) = 23 / 40 :=
by
  let total_hours := 24
  let affected_hours := 6
  let correct_hours := total_hours - affected_hours
  let total_minutes := 60
  let affected_minutes := 14
  let correct_minutes := total_minutes - affected_minutes
  have h1 : correct_hours = 18 := rfl
  have h2 : correct_minutes = 46 := rfl
  have h3 : 18 / 24 = 3 / 4 := by norm_num
  have h4 : 46 / 60 = 23 / 30 := by norm_num
  have h5 : (3 / 4) * (23 / 30) = 23 / 40 := by norm_num
  exact h5

end clock_correct_time_fraction_l72_72029


namespace trig_identity_l72_72498

theorem trig_identity : 
  Real.sin (600 * Real.pi / 180) + Real.tan (240 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end trig_identity_l72_72498


namespace right_triangle_point_selection_l72_72279

theorem right_triangle_point_selection : 
  let n := 200 
  let rows := 2
  (rows * (n - 22 + 1)) + 2 * (rows * (n - 122 + 1)) + (n * (2 * (n - 1))) = 80268 := 
by 
  let rows := 2
  let n := 200
  let case1a := rows * (n - 22 + 1)
  let case1b := 2 * (rows * (n - 122 + 1))
  let case2 := n * (2 * (n - 1))
  have h : case1a + case1b + case2 = 80268 := by sorry
  exact h

end right_triangle_point_selection_l72_72279


namespace remainder_of_3_to_40_plus_5_mod_5_l72_72730

theorem remainder_of_3_to_40_plus_5_mod_5 : (3^40 + 5) % 5 = 1 :=
by
  sorry

end remainder_of_3_to_40_plus_5_mod_5_l72_72730


namespace point_on_x_axis_l72_72590

theorem point_on_x_axis (m : ℝ) (h : (2 * m + 3) = 0) : m = -3 / 2 :=
sorry

end point_on_x_axis_l72_72590


namespace sufficient_but_not_necessary_condition_for_negativity_l72_72784

variable (b c : ℝ)

def f (x : ℝ) : ℝ := x^2 + b*x + c

theorem sufficient_but_not_necessary_condition_for_negativity (b c : ℝ) :
  (c < 0 → ∃ x : ℝ, f b c x < 0) ∧ (∃ b c : ℝ, ∃ x : ℝ, c ≥ 0 ∧ f b c x < 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_for_negativity_l72_72784


namespace interior_angle_ratio_l72_72940

theorem interior_angle_ratio (exterior_angle1 exterior_angle2 exterior_angle3 : ℝ)
  (h_ratio : 3 * exterior_angle1 = 4 * exterior_angle2 ∧ 
             4 * exterior_angle1 = 5 * exterior_angle3 ∧ 
             3 * exterior_angle1 + 4 * exterior_angle2 + 5 * exterior_angle3 = 360 ) : 
  3 * (180 - exterior_angle1) = 2 * (180 - exterior_angle2) ∧ 
  2 * (180 - exterior_angle2) = 1 * (180 - exterior_angle3) :=
sorry

end interior_angle_ratio_l72_72940


namespace non_chocolate_candy_count_l72_72712

theorem non_chocolate_candy_count (total_candy : ℕ) (total_bags : ℕ) 
  (chocolate_hearts_bags : ℕ) (chocolate_kisses_bags : ℕ) (each_bag_pieces : ℕ) 
  (non_chocolate_bags : ℕ) : 
  total_candy = 63 ∧ 
  total_bags = 9 ∧ 
  chocolate_hearts_bags = 2 ∧ 
  chocolate_kisses_bags = 3 ∧ 
  total_candy / total_bags = each_bag_pieces ∧ 
  total_bags - (chocolate_hearts_bags + chocolate_kisses_bags) = non_chocolate_bags ∧ 
  non_chocolate_bags * each_bag_pieces = 28 :=
by
  -- use "sorry" to skip the proof
  sorry

end non_chocolate_candy_count_l72_72712


namespace adult_ticket_cost_l72_72635

theorem adult_ticket_cost (A Tc : ℝ) (T C : ℕ) (M : ℝ) 
  (hTc : Tc = 3.50) 
  (hT : T = 21) 
  (hC : C = 16) 
  (hM : M = 83.50) 
  (h_eq : 16 * Tc + (↑(T - C)) * A = M) : 
  A = 5.50 :=
by sorry

end adult_ticket_cost_l72_72635


namespace triangle_side_b_l72_72166

open Real

variable {a b c : ℝ} (A B C : ℝ)

theorem triangle_side_b (h1 : a^2 - c^2 = 2 * b) (h2 : sin B = 6 * cos A * sin C) : b = 3 :=
sorry

end triangle_side_b_l72_72166


namespace only_correct_option_is_C_l72_72658

-- Definitions of the conditions as per the given problem
def option_A (a : ℝ) : Prop := a^2 * a^3 = a^6
def option_B (a : ℝ) : Prop := (a^2)^3 = a^5
def option_C (a b : ℝ) : Prop := (a * b)^3 = a^3 * b^3
def option_D (a : ℝ) : Prop := a^8 / a^2 = a^4

-- The theorem stating that only option C is correct
theorem only_correct_option_is_C (a b : ℝ) : 
  ¬(option_A a) ∧ ¬(option_B a) ∧ option_C a b ∧ ¬(option_D a) :=
by sorry

end only_correct_option_is_C_l72_72658


namespace vernal_equinox_shadow_length_l72_72700

-- Lean 4 statement
theorem vernal_equinox_shadow_length :
  ∀ (a : ℕ → ℝ), (a 4 = 10.5) → (a 10 = 4.5) → 
  (∀ (n m : ℕ), a (n + 1) = a n + (a 2 - a 1)) → 
  a 7 = 7.5 :=
by
  intros a h_4 h_10 h_progression
  sorry

end vernal_equinox_shadow_length_l72_72700


namespace correct_option_l72_72454

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_decreasing_on_nonneg_real (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → 0 ≤ x₂ → x₁ < x₂ → f x₁ > f x₂

theorem correct_option (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_decr : is_decreasing_on_nonneg_real f) :
  f 2 < f (-1) ∧ f (-1) < f 0 :=
by
  sorry

end correct_option_l72_72454


namespace smallest_possible_value_of_other_integer_l72_72706

theorem smallest_possible_value_of_other_integer 
  (n : ℕ) (hn_pos : 0 < n) (h_eq : (Nat.lcm 75 n) / (Nat.gcd 75 n) = 45) : n = 135 :=
by sorry

end smallest_possible_value_of_other_integer_l72_72706


namespace derivative_of_volume_is_surface_area_l72_72737

noncomputable def V_sphere (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3

theorem derivative_of_volume_is_surface_area (R : ℝ) (h : 0 < R) : 
  (deriv V_sphere R) = 4 * Real.pi * R^2 :=
by sorry

end derivative_of_volume_is_surface_area_l72_72737


namespace earnings_difference_l72_72490

theorem earnings_difference :
  let oula_deliveries := 96
  let tona_deliveries := oula_deliveries * 3 / 4
  let area_A_fee := 100
  let area_B_fee := 125
  let area_C_fee := 150
  let oula_area_A_deliveries := 48
  let oula_area_B_deliveries := 32
  let oula_area_C_deliveries := 16
  let tona_area_A_deliveries := 27
  let tona_area_B_deliveries := 18
  let tona_area_C_deliveries := 9
  let oula_total_earnings := oula_area_A_deliveries * area_A_fee + oula_area_B_deliveries * area_B_fee + oula_area_C_deliveries * area_C_fee
  let tona_total_earnings := tona_area_A_deliveries * area_A_fee + tona_area_B_deliveries * area_B_fee + tona_area_C_deliveries * area_C_fee
  oula_total_earnings - tona_total_earnings = 4900 := by
sorry

end earnings_difference_l72_72490


namespace simplify_expression_l72_72746

variable (x y z : ℝ)

theorem simplify_expression (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hne : y - z / x ≠ 0) : 
  (x - z / y) / (y - z / x) = x / y := 
by 
  sorry

end simplify_expression_l72_72746


namespace least_number_to_subtract_l72_72575

theorem least_number_to_subtract (n : ℕ) (k : ℕ) (h : 1387 = n + k * 15) : n = 7 :=
by
  sorry

end least_number_to_subtract_l72_72575


namespace number_of_good_games_l72_72551

def total_games : ℕ := 11
def bad_games : ℕ := 5
def good_games : ℕ := total_games - bad_games

theorem number_of_good_games : good_games = 6 := by
  sorry

end number_of_good_games_l72_72551


namespace amino_inequality_l72_72100

theorem amino_inequality
  (x y z : ℝ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hz : z ≠ 0)
  (h : x + y + z = x * y * z) :
  ( (x^2 - 1) / x )^2 + ( (y^2 - 1) / y )^2 + ( (z^2 - 1) / z )^2 ≥ 4 := by
  sorry

end amino_inequality_l72_72100


namespace find_smallest_divisor_l72_72142

theorem find_smallest_divisor {n : ℕ} 
  (h : n = 44402) 
  (hdiv1 : (n + 2) % 30 = 0) 
  (hdiv2 : (n + 2) % 48 = 0) 
  (hdiv3 : (n + 2) % 74 = 0) 
  (hdiv4 : (n + 2) % 100 = 0) : 
  ∃ d, d = 37 ∧ d ∣ (n + 2) :=
sorry

end find_smallest_divisor_l72_72142


namespace find_original_number_l72_72214

-- Define the given conditions
def increased_by_twenty_percent (x : ℝ) : ℝ := x * 1.20

-- State the theorem
theorem find_original_number (x : ℝ) (h : increased_by_twenty_percent x = 480) : x = 400 :=
by
  sorry

end find_original_number_l72_72214


namespace number_of_elements_in_sequence_l72_72149

theorem number_of_elements_in_sequence :
  ∀ (a₀ d : ℕ) (n : ℕ), 
  a₀ = 4 →
  d = 2 →
  n = 64 →
  (a₀ + (n - 1) * d = 130) →
  n = 64 := 
by
  -- We will skip the proof steps as indicated
  sorry

end number_of_elements_in_sequence_l72_72149


namespace quotient_when_divided_by_44_l72_72754

theorem quotient_when_divided_by_44 :
  ∃ N Q : ℕ, (N % 44 = 0) ∧ (N % 39 = 15) ∧ (N / 44 = Q) ∧ (Q = 3) :=
by {
  sorry
}

end quotient_when_divided_by_44_l72_72754


namespace solve_system_of_equations_l72_72556

theorem solve_system_of_equations :
  ∃ (x y : ℤ), (x - y = 2) ∧ (2 * x + y = 7) ∧ (x = 3) ∧ (y = 1) :=
by
  sorry

end solve_system_of_equations_l72_72556


namespace parallel_lines_slope_equality_l72_72756

theorem parallel_lines_slope_equality (m : ℝ) : (∀ x y : ℝ, 3 * x + y - 3 = 0) ∧ (∀ x y : ℝ, 6 * x + m * y + 1 = 0) → m = 2 :=
by 
  sorry

end parallel_lines_slope_equality_l72_72756


namespace sufficient_but_not_necessary_l72_72000

theorem sufficient_but_not_necessary (a b : ℝ) (h : a > b ∧ b > 0) : a^2 > b^2 ∧ ¬ (a^2 > b^2 → a > b ∧ b > 0) :=
by
  sorry

end sufficient_but_not_necessary_l72_72000


namespace area_enclosed_by_cosine_l72_72360

theorem area_enclosed_by_cosine :
  ∫ x in -Real.pi..Real.pi, (1 + Real.cos x) = 2 * Real.pi := by
  sorry

end area_enclosed_by_cosine_l72_72360


namespace digit_150_in_fraction_l72_72514

-- Define the decimal expansion repeating sequence for the fraction 31/198
def repeat_seq : List Nat := [1, 5, 6, 5, 6, 5]

-- Define a function to get the nth digit of the repeating sequence
def nth_digit (n : Nat) : Nat :=
  repeat_seq.get! ((n - 1) % repeat_seq.length)

-- State the theorem to be proved
theorem digit_150_in_fraction : nth_digit 150 = 5 := 
sorry

end digit_150_in_fraction_l72_72514


namespace distinct_symbols_count_l72_72695

/-- A modified Morse code symbol is represented by a sequence of dots, dashes, and spaces, where spaces can only appear between dots and dashes but not at the beginning or end of the sequence. -/
def valid_sequence_length_1 := 2
def valid_sequence_length_2 := 2^2
def valid_sequence_length_3 := 2^3 + 3
def valid_sequence_length_4 := 2^4 + 3 * 2^4 + 3 * 2^4 
def valid_sequence_length_5 := 2^5 + 4 * 2^5 + 6 * 2^5 + 4 * 2^5

theorem distinct_symbols_count : 
  valid_sequence_length_1 + valid_sequence_length_2 + valid_sequence_length_3 + valid_sequence_length_4 + valid_sequence_length_5 = 609 := by
  sorry

end distinct_symbols_count_l72_72695


namespace ratio_both_to_onlyB_is_2_l72_72128

variables (num_A num_B both: ℕ)

-- Given conditions
axiom A_eq_2B : num_A = 2 * num_B
axiom both_eq_500 : both = 500
axiom both_multiple_of_only_B : ∃ k : ℕ, both = k * (num_B - both)
axiom only_A_eq_1000 : (num_A - both) = 1000

-- Define the Lean theorem statement
theorem ratio_both_to_onlyB_is_2 : (both : ℝ) / (num_B - both : ℝ) = 2 := 
sorry

end ratio_both_to_onlyB_is_2_l72_72128


namespace one_point_one_billion_in_scientific_notation_l72_72721

noncomputable def one_point_one_billion : ℝ := 1.1 * 10^9

theorem one_point_one_billion_in_scientific_notation :
  1.1 * 10^9 = 1100000000 :=
by
  sorry

end one_point_one_billion_in_scientific_notation_l72_72721


namespace solve_for_x_l72_72848

variable (x : ℝ)

theorem solve_for_x (h : 2 * x - 3 * x + 4 * x = 150) : x = 50 := by
  sorry

end solve_for_x_l72_72848


namespace percentage_of_sikh_boys_l72_72347

-- Define the conditions
def total_boys : ℕ := 650
def muslim_boys : ℕ := (44 * total_boys) / 100
def hindu_boys : ℕ := (28 * total_boys) / 100
def other_boys : ℕ := 117
def sikh_boys : ℕ := total_boys - (muslim_boys + hindu_boys + other_boys)

-- Define and prove the theorem
theorem percentage_of_sikh_boys : (sikh_boys * 100) / total_boys = 10 :=
by
  have h_muslims: muslim_boys = 286 := by sorry
  have h_hindus: hindu_boys = 182 := by sorry
  have h_total: muslim_boys + hindu_boys + other_boys = 585 := by sorry
  have h_sikhs: sikh_boys = 65 := by sorry
  have h_percentage: (65 * 100) / 650 = 10 := by sorry
  exact h_percentage

end percentage_of_sikh_boys_l72_72347


namespace no_three_parabolas_l72_72351

theorem no_three_parabolas (a b c : ℝ) : ¬ (b^2 > 4*a*c ∧ a^2 > 4*b*c ∧ c^2 > 4*a*b) := by
  sorry

end no_three_parabolas_l72_72351


namespace jim_gave_away_cards_l72_72118

theorem jim_gave_away_cards
  (sets_brother : ℕ := 15)
  (sets_sister : ℕ := 8)
  (sets_friend : ℕ := 4)
  (sets_cousin : ℕ := 6)
  (sets_classmate : ℕ := 3)
  (cards_per_set : ℕ := 25) :
  (sets_brother + sets_sister + sets_friend + sets_cousin + sets_classmate) * cards_per_set = 900 :=
by
  sorry

end jim_gave_away_cards_l72_72118


namespace min_k_l72_72107

def a_n (n : ℕ) : ℕ :=
  n

def b_n (n : ℕ) : ℚ :=
  a_n n / 3^n

def T_n (n : ℕ) : ℚ :=
  (List.range n).foldl (λ acc i => acc + b_n (i + 1)) 0

theorem min_k (k : ℕ) (h : ∀ n : ℕ, n ≥ k → |T_n n - 3/4| < 1/(4*n)) : k = 4 :=
  sorry

end min_k_l72_72107


namespace persons_attended_total_l72_72010

theorem persons_attended_total (p q : ℕ) (a : ℕ) (c : ℕ) (total_amount : ℕ) (adult_ticket : ℕ) (child_ticket : ℕ) 
  (h1 : adult_ticket = 60) (h2 : child_ticket = 25) (h3 : total_amount = 14000) 
  (h4 : a = 200) (h5 : p = a + c)
  (h6 : a * adult_ticket + c * child_ticket = total_amount):
  p = 280 :=
by
  sorry

end persons_attended_total_l72_72010


namespace true_statements_count_l72_72056

def reciprocal (n : ℕ) : ℚ := 1 / n

theorem true_statements_count :
  let s1 := reciprocal 4 + reciprocal 8 = reciprocal 12
  let s2 := reciprocal 8 - reciprocal 5 = reciprocal 3
  let s3 := reciprocal 3 * reciprocal 9 = reciprocal 27
  let s4 := reciprocal 15 / reciprocal 3 = reciprocal 5
  (if s1 then 1 else 0) + 
  (if s2 then 1 else 0) + 
  (if s3 then 1 else 0) + 
  (if s4 then 1 else 0) = 2 :=
by
  sorry

end true_statements_count_l72_72056


namespace sin_minus_cos_third_quadrant_l72_72790

theorem sin_minus_cos_third_quadrant (α : ℝ) (h_tan : Real.tan α = 2) (h_quadrant : π < α ∧ α < 3 * π / 2) : 
  Real.sin α - Real.cos α = -Real.sqrt 5 / 5 := 
by 
  sorry

end sin_minus_cos_third_quadrant_l72_72790


namespace geometric_sequence_a_l72_72647

theorem geometric_sequence_a (a : ℝ) (h1 : a > 0) (h2 : ∃ r : ℝ, 280 * r = a ∧ a * r = 180 / 49) :
  a = 32.07 :=
by sorry

end geometric_sequence_a_l72_72647


namespace games_played_l72_72496

def total_points : ℝ := 120.0
def points_per_game : ℝ := 12.0
def num_games : ℝ := 10.0

theorem games_played : (total_points / points_per_game) = num_games := 
by 
  sorry

end games_played_l72_72496


namespace correct_angle_calculation_l72_72274

theorem correct_angle_calculation (α β : ℝ) (hα : 0 < α ∧ α < 90) (hβ : 90 < β ∧ β < 180) :
    22.5 < 0.25 * (α + β) ∧ 0.25 * (α + β) < 67.5 → 0.25 * (α + β) = 45.3 :=
by
  sorry

end correct_angle_calculation_l72_72274


namespace meaningful_square_root_l72_72573

theorem meaningful_square_root (x : ℝ) (h : x - 1 ≥ 0) : x ≥ 1 := 
by {
  -- This part would contain the proof
  sorry
}

end meaningful_square_root_l72_72573


namespace gcd_pow_diff_l72_72982

theorem gcd_pow_diff (m n: ℤ) (H1: m = 2^2025 - 1) (H2: n = 2^2016 - 1) : Int.gcd m n = 511 := by
  sorry

end gcd_pow_diff_l72_72982


namespace p_correct_l72_72015

noncomputable def p : ℝ → ℝ := sorry

axiom p_at_3 : p 3 = 10

axiom p_condition (x y : ℝ) : p x * p y = p x + p y + p (x * y) - 2

theorem p_correct : ∀ x, p x = x^2 + 1 :=
sorry

end p_correct_l72_72015


namespace average_production_per_day_for_entire_month_l72_72538

-- Definitions based on the conditions
def average_first_25_days := 65
def average_last_5_days := 35
def number_of_days_in_first_period := 25
def number_of_days_in_last_period := 5
def total_days_in_month := 30

-- The goal is to prove that the average production per day for the entire month is 60 TVs/day.
theorem average_production_per_day_for_entire_month :
  (average_first_25_days * number_of_days_in_first_period + 
   average_last_5_days * number_of_days_in_last_period) / total_days_in_month = 60 := 
by
  sorry

end average_production_per_day_for_entire_month_l72_72538


namespace quadratic_inequality_range_of_k_l72_72540

theorem quadratic_inequality (a b x : ℝ) (h1 : a = 1) (h2 : b > 1) :
  (a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) :=
sorry

theorem range_of_k (x y k : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : (1/x) + (2/y) = 1) (h4 : 2 * x + y ≥ k^2 + k + 2) :
  -3 ≤ k ∧ k ≤ 2 :=
sorry

end quadratic_inequality_range_of_k_l72_72540


namespace divisibility_of_sum_of_fifths_l72_72017

theorem divisibility_of_sum_of_fifths (x y z : ℤ) (h : x ≠ y ∧ y ≠ z ∧ z ≠ x) :
  ∃ k : ℤ, (x - y) ^ 5 + (y - z) ^ 5 + (z - x) ^ 5 = 5 * k * (x - y) * (y - z) * (z - x) :=
sorry

end divisibility_of_sum_of_fifths_l72_72017


namespace LCM_GCD_even_nonnegative_l72_72597

theorem LCM_GCD_even_nonnegative (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  : ∃ (n : ℕ), (n = Nat.lcm a b + Nat.gcd a b - a - b) ∧ (n % 2 = 0) ∧ (0 ≤ n) := 
sorry

end LCM_GCD_even_nonnegative_l72_72597


namespace proof_problem_l72_72375

axiom sqrt (x : ℝ) : ℝ
axiom cbrt (x : ℝ) : ℝ
noncomputable def sqrtValue : ℝ :=
  sqrt 81

theorem proof_problem (m n : ℝ) (hm : sqrt m = 3) (hn : cbrt n = -4) : sqrt (2 * m - n - 1) = 9 ∨ sqrt (2 * m - n - 1) = -9 :=
by
  sorry

end proof_problem_l72_72375


namespace arithmetic_sequence_identification_l72_72419

variable (a : ℕ → ℤ)
variable (d : ℤ)

def is_arithmetic (a : ℕ → ℤ) (d : ℤ) : Prop := ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_identification (h : is_arithmetic a d) :
  (is_arithmetic (fun n => a n + 3) d) ∧
  ¬ (is_arithmetic (fun n => a n ^ 2) d) ∧
  (is_arithmetic (fun n => a (n + 1) - a n) d) ∧
  (is_arithmetic (fun n => 2 * a n) (2 * d)) ∧
  (is_arithmetic (fun n => 2 * a n + n) (2 * d + 1)) :=
by
  sorry

end arithmetic_sequence_identification_l72_72419


namespace unique_sum_of_squares_power_of_two_l72_72739

theorem unique_sum_of_squares_power_of_two (n : ℕ) :
  ∃! (a b : ℕ), 2^n = a^2 + b^2 := 
sorry

end unique_sum_of_squares_power_of_two_l72_72739


namespace inequality_solution_set_l72_72934

theorem inequality_solution_set {m n : ℝ} (h : ∀ x : ℝ, -3 < x ∧ x < 6 ↔ x^2 - m * x - 6 * n < 0) : m + n = 6 :=
by
  sorry

end inequality_solution_set_l72_72934


namespace bill_difference_zero_l72_72173

theorem bill_difference_zero (l m : ℝ) 
  (hL : (25 / 100) * l = 5) 
  (hM : (15 / 100) * m = 3) : 
  l - m = 0 := 
sorry

end bill_difference_zero_l72_72173


namespace student_estimated_score_l72_72716

theorem student_estimated_score :
  (6 * 5 + 3 * 5 * (3 / 4) + 2 * 5 * (1 / 3) + 1 * 5 * (1 / 4)) = 41.25 :=
by
 sorry

end student_estimated_score_l72_72716


namespace number_of_acceptable_outfits_l72_72288

-- Definitions based on conditions
def total_shirts := 5
def total_pants := 4
def restricted_shirts := 2
def restricted_pants := 1

-- Defining the problem statement
theorem number_of_acceptable_outfits : 
  (total_shirts * total_pants - restricted_shirts * restricted_pants + restricted_shirts * (total_pants - restricted_pants)) = 18 :=
by sorry

end number_of_acceptable_outfits_l72_72288


namespace expected_value_dodecahedral_die_l72_72595

-- Define the faces of the die
def faces : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Define the scoring rule
def score (n : ℕ) : ℕ :=
  if n ≤ 6 then 2 * n else n

-- The probability of each face
def prob : ℚ := 1 / 12

-- Calculate the expected value
noncomputable def expected_value : ℚ :=
  prob * (score 1 + score 2 + score 3 + score 4 + score 5 + score 6 + 
          score 7 + score 8 + score 9 + score 10 + score 11 + score 12)

-- State the theorem to be proved
theorem expected_value_dodecahedral_die : expected_value = 8.25 := 
  sorry

end expected_value_dodecahedral_die_l72_72595


namespace kai_ice_plate_division_l72_72269

-- Define the "L"-shaped ice plate with given dimensions
structure LShapedIcePlate (a : ℕ) :=
(horiz_length : ℕ)
(vert_length : ℕ)
(horiz_eq_vert : horiz_length = a ∧ vert_length = a)

-- Define the correctness of dividing the L-shaped plate into four equal parts
def can_be_divided_into_four_equal_parts (a : ℕ) (piece : LShapedIcePlate a) : Prop :=
∃ cut_points_v1 cut_points_v2 cut_points_h1 cut_points_h2,
  -- The cut points for vertical and horizontal cuts to turn the large "L" shape into four smaller "L" shapes
  piece.horiz_length = cut_points_v1 + cut_points_v2 ∧
  piece.vert_length = cut_points_h1 + cut_points_h2 ∧
  cut_points_v1 = a / 2 ∧ cut_points_v2 = a - a / 2 ∧
  cut_points_h1 = a / 2 ∧ cut_points_h2 = a - a / 2

-- Prove the main theorem
theorem kai_ice_plate_division (a : ℕ) (h : a > 0) (plate : LShapedIcePlate a) : 
  can_be_divided_into_four_equal_parts a plate :=
sorry

end kai_ice_plate_division_l72_72269


namespace cryptarithm_solution_l72_72245

theorem cryptarithm_solution (A B : ℕ) (h_digit_A : A < 10) (h_digit_B : B < 10)
  (h_equation : 9 * (10 * A + B) = 110 * A + B) :
  A = 2 ∧ B = 5 :=
sorry

end cryptarithm_solution_l72_72245


namespace linear_function_passing_quadrants_l72_72383

theorem linear_function_passing_quadrants (b : ℝ) :
  (∀ x : ℝ, (y = x + b) ∧ (y > 0 ↔ (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0))) →
  b > 0 :=
sorry

end linear_function_passing_quadrants_l72_72383


namespace largest_divisor_of_m_l72_72771

theorem largest_divisor_of_m (m : ℕ) (h1 : m > 0) (h2 : 216 ∣ m^2) : 36 ∣ m :=
sorry

end largest_divisor_of_m_l72_72771


namespace am_gm_inequality_example_am_gm_inequality_equality_condition_l72_72359

theorem am_gm_inequality_example (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  (x + y^3) * (x^3 + y) ≥ 4 * x^2 * y^2 :=
sorry

theorem am_gm_inequality_equality_condition (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  ((x + y^3) * (x^3 + y) = 4 * x^2 * y^2) ↔ (x = 0 ∧ y = 0 ∨ x = 1 ∧ y = 1) :=
sorry

end am_gm_inequality_example_am_gm_inequality_equality_condition_l72_72359


namespace real_roots_quadratic_l72_72042

theorem real_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, (m - 1) * x^2 + 3 * x - 1 = 0) ↔ (m ≥ -5/4 ∧ m ≠ 1) := by
  sorry

end real_roots_quadratic_l72_72042


namespace range_of_a_monotonically_decreasing_l72_72561

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - 3 * x

theorem range_of_a_monotonically_decreasing (a : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x < 1 → deriv (f a) x ≤ 0) ↔ 0 < a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_monotonically_decreasing_l72_72561


namespace problem_l72_72998

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin (2 * x + Real.pi / 4)

theorem problem
  (h1 : f (Real.pi / 8) = 2)
  (h2 : f (5 * Real.pi / 8) = -2) :
  (∀ x : ℝ, f x = 1 ↔ 
    (∃ k : ℤ, x = -Real.pi / 24 + k * Real.pi) ∨
    (∃ k : ℤ, x = 7 * Real.pi / 24 + k * Real.pi)) :=
by
  sorry

end problem_l72_72998


namespace isosceles_triangle_count_l72_72079

noncomputable def valid_points : List (ℕ × ℕ) :=
  [(2, 5), (5, 5)]

theorem isosceles_triangle_count 
  (A B : ℕ × ℕ) 
  (H_A : A = (2, 2)) 
  (H_B : B = (5, 2)) : 
  valid_points.length = 2 :=
  sorry

end isosceles_triangle_count_l72_72079


namespace cos_value_l72_72131

variable (α : ℝ)

theorem cos_value (h : Real.sin (π / 4 + α) = 2 / 3) : Real.cos (π / 4 - α) = 2 / 3 := 
by 
  sorry 

end cos_value_l72_72131


namespace flower_count_l72_72414

variables (o y p : ℕ)

theorem flower_count (h1 : y + p = 7) (h2 : o + p = 10) (h3 : o + y = 5) : o + y + p = 11 := sorry

end flower_count_l72_72414


namespace find_a_n_l72_72671

noncomputable def is_arithmetic_seq (a b : ℕ) (seq : ℕ → ℕ) : Prop :=
  ∀ n, seq n = a + n * b

noncomputable def is_geometric_seq (b a : ℕ) (seq : ℕ → ℕ) : Prop :=
  ∀ n, seq n = b * a ^ n

theorem find_a_n (a b : ℕ) 
  (a_positive : a > 1)
  (b_positive : b > 1)
  (a_seq : ℕ → ℕ)
  (b_seq : ℕ → ℕ)
  (arith_seq : is_arithmetic_seq a b a_seq)
  (geom_seq : is_geometric_seq b a b_seq)
  (init_condition : a_seq 0 < b_seq 0)
  (next_condition : b_seq 1 < a_seq 2)
  (relation_condition : ∀ n, ∃ m, a_seq m + 3 = b_seq n) :
  ∀ n, a_seq n = 5 * n - 3 :=
sorry

end find_a_n_l72_72671


namespace average_speed_of_trip_l72_72986

theorem average_speed_of_trip :
  let total_distance := 50 -- in kilometers
  let distance1 := 25 -- in kilometers
  let speed1 := 66 -- in kilometers per hour
  let distance2 := 25 -- in kilometers
  let speed2 := 33 -- in kilometers per hour
  let time1 := distance1 / speed1 -- time taken for the first part
  let time2 := distance2 / speed2 -- time taken for the second part
  let total_time := time1 + time2 -- total time for the trip
  let average_speed := total_distance / total_time -- average speed of the trip
  average_speed = 44 := by
{
  sorry
}

end average_speed_of_trip_l72_72986


namespace sum_of_number_and_reverse_divisible_by_11_l72_72458

theorem sum_of_number_and_reverse_divisible_by_11 (A B : ℕ) (hA : 0 ≤ A) (hA9 : A ≤ 9) (hB : 0 ≤ B) (hB9 : B ≤ 9) :
  11 ∣ ((10 * A + B) + (10 * B + A)) :=
by
  sorry

end sum_of_number_and_reverse_divisible_by_11_l72_72458


namespace solve_system_eqns_l72_72494

theorem solve_system_eqns (x y : ℚ) 
    (h1 : (x - 30) / 3 = (2 * y + 7) / 4)
    (h2 : x - y = 10) :
  x = -81 / 2 ∧ y = -101 / 2 := 
sorry

end solve_system_eqns_l72_72494


namespace minutes_practiced_other_days_l72_72364

theorem minutes_practiced_other_days (total_hours : ℕ) (minutes_per_day : ℕ) (num_days : ℕ) :
  total_hours = 450 ∧ minutes_per_day = 86 ∧ num_days = 2 → (total_hours - num_days * minutes_per_day) = 278 := by
  sorry

end minutes_practiced_other_days_l72_72364


namespace evaluate_expression_l72_72139

theorem evaluate_expression : 2 - (-3) - 4 - (-5) - 6 - (-7) - 8 = -1 :=
by
  sorry

end evaluate_expression_l72_72139


namespace real_roots_system_l72_72344

theorem real_roots_system :
  ∃ (x y : ℝ), 
    (x * y * (x^2 + y^2) = 78 ∧ x^4 + y^4 = 97) ↔ 
    (x, y) = (3, 2) ∨ (x, y) = (2, 3) ∨ (x, y) = (-3, -2) ∨ (x, y) = (-2, -3) := 
by 
  sorry

end real_roots_system_l72_72344


namespace same_terminal_side_l72_72736

theorem same_terminal_side (k : ℤ) : 
  {α | ∃ k : ℤ, α = k * 360 + (-263 : ℤ)} = 
  {α | ∃ k : ℤ, α = k * 360 - 263} := 
by sorry

end same_terminal_side_l72_72736


namespace conic_sections_of_equation_l72_72547

theorem conic_sections_of_equation :
  ∀ x y : ℝ, y^4 - 9*x^6 = 3*y^2 - 1 →
  (∃ y, y^2 - 3*x^3 = 4 ∨ y^2 + 3*x^3 = 0) :=
by 
  sorry

end conic_sections_of_equation_l72_72547


namespace smallest_multiple_of_6_8_12_l72_72237

theorem smallest_multiple_of_6_8_12 : ∃ n : ℕ, n > 0 ∧ n % 6 = 0 ∧ n % 8 = 0 ∧ n % 12 = 0 ∧ ∀ m : ℕ, (m > 0 ∧ m % 6 = 0 ∧ m % 8 = 0 ∧ m % 12 = 0) → n ≤ m := 
sorry

end smallest_multiple_of_6_8_12_l72_72237


namespace cost_of_45_lilies_l72_72033

-- Defining the conditions
def price_per_lily (n : ℕ) : ℝ :=
  if n <= 30 then 2
  else 1.8

-- Stating the problem in Lean 4
theorem cost_of_45_lilies :
  price_per_lily 15 * 15 = 30 → (price_per_lily 45 * 45 = 81) :=
by
  intro h
  sorry

end cost_of_45_lilies_l72_72033


namespace arabella_dance_steps_l72_72452

theorem arabella_dance_steps :
  exists T1 T2 T3 : ℕ,
    T1 = 30 ∧
    T3 = T1 + T2 ∧
    T1 + T2 + T3 = 90 ∧
    (T2 / T1 : ℚ) = 1 / 2 :=
by
  sorry

end arabella_dance_steps_l72_72452


namespace cos_2000_eq_neg_inv_sqrt_l72_72980

theorem cos_2000_eq_neg_inv_sqrt (a : ℝ) (h : Real.tan (20 * Real.pi / 180) = a) :
  Real.cos (2000 * Real.pi / 180) = -1 / Real.sqrt (1 + a^2) :=
sorry

end cos_2000_eq_neg_inv_sqrt_l72_72980


namespace average_of_values_l72_72112

theorem average_of_values (z : ℝ) : 
  (0 + 3 * z + 6 * z + 12 * z + 24 * z) / 5 = 9 * z :=
by
  sorry

end average_of_values_l72_72112


namespace Maddie_spent_on_tshirts_l72_72129

theorem Maddie_spent_on_tshirts
  (packs_white : ℕ)
  (count_white_per_pack : ℕ)
  (packs_blue : ℕ)
  (count_blue_per_pack : ℕ)
  (cost_per_tshirt : ℕ)
  (h_white : packs_white = 2)
  (h_count_w : count_white_per_pack = 5)
  (h_blue : packs_blue = 4)
  (h_count_b : count_blue_per_pack = 3)
  (h_cost : cost_per_tshirt = 3) :
  (packs_white * count_white_per_pack + packs_blue * count_blue_per_pack) * cost_per_tshirt = 66 := by
  sorry

end Maddie_spent_on_tshirts_l72_72129


namespace donation_problem_l72_72313

theorem donation_problem
  (A B C D : Prop)
  (h1 : ¬A ↔ (B ∨ C ∨ D))
  (h2 : B ↔ D)
  (h3 : C ↔ ¬B) 
  (h4 : D ↔ ¬B): A := 
by
  sorry

end donation_problem_l72_72313


namespace min_focal_length_of_hyperbola_l72_72804

theorem min_focal_length_of_hyperbola (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_area : a * b = 8) :
  ∃ c ≥ 4, 2 * c = 8 :=
by sorry

end min_focal_length_of_hyperbola_l72_72804


namespace steve_speed_ratio_l72_72468

variable (distance : ℝ)
variable (total_time : ℝ)
variable (speed_back : ℝ)
variable (speed_to : ℝ)

noncomputable def speed_ratio (distance : ℝ) (total_time : ℝ) (speed_back : ℝ) : ℝ := 
  let time_to := total_time - distance / speed_back
  let speed_to := distance / time_to
  speed_back / speed_to

theorem steve_speed_ratio (h1 : distance = 10) (h2 : total_time = 6) (h3 : speed_back = 5) :
  speed_ratio distance total_time speed_back = 2 := by
  sorry

end steve_speed_ratio_l72_72468


namespace sum_first_9000_terms_l72_72102

noncomputable def geometric_sequence_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_first_9000_terms (a r : ℝ) :
  geometric_sequence_sum a r 3000 = 500 →
  geometric_sequence_sum a r 6000 = 950 →
  geometric_sequence_sum a r 9000 = 1355 :=
by
  intros h1 h2
  sorry

end sum_first_9000_terms_l72_72102


namespace find_teaspoons_of_salt_l72_72803

def sodium_in_salt (S : ℕ) : ℕ := 50 * S
def sodium_in_parmesan (P : ℕ) : ℕ := 25 * P

-- Initial total sodium amount with 8 ounces of parmesan
def initial_total_sodium (S : ℕ) : ℕ := sodium_in_salt S + sodium_in_parmesan 8

-- Reduced sodium after removing 4 ounces of parmesan
def reduced_sodium (S : ℕ) : ℕ := initial_total_sodium S * 2 / 3

-- Reduced sodium with 4 fewer ounces of parmesan cheese
def new_total_sodium (S : ℕ) : ℕ := sodium_in_salt S + sodium_in_parmesan 4

theorem find_teaspoons_of_salt : ∃ (S : ℕ), reduced_sodium S = new_total_sodium S ∧ S = 2 :=
by
  sorry

end find_teaspoons_of_salt_l72_72803


namespace find_k_and_b_l72_72154

theorem find_k_and_b (k b : ℝ) :
  (∃ P Q : ℝ × ℝ, P ≠ Q ∧
  ((P.1 - 1)^2 + P.2^2 = 1) ∧ 
  ((Q.1 - 1)^2 + Q.2^2 = 1) ∧ 
  (P.2 = k * P.1) ∧ 
  (Q.2 = k * Q.1) ∧ 
  (P.1 - P.2 + b = 0) ∧ 
  (Q.1 - Q.2 + b = 0) ∧ 
  ((P.1 + Q.1) / 2 = (P.2 + Q.2) / 2)) →
  k = -1 ∧ b = -1 :=
sorry

end find_k_and_b_l72_72154


namespace apple_pies_l72_72719

theorem apple_pies (total_apples not_ripe_apples apples_per_pie : ℕ) 
    (h1 : total_apples = 34) 
    (h2 : not_ripe_apples = 6) 
    (h3 : apples_per_pie = 4) : 
    (total_apples - not_ripe_apples) / apples_per_pie = 7 :=
by 
    sorry

end apple_pies_l72_72719


namespace geometric_progression_a5_value_l72_72115

theorem geometric_progression_a5_value
  (a : ℕ → ℝ)
  (h_geom : ∃ r : ℝ, ∀ n, a (n + 1) = a n * r)
  (h_roots : ∃ x y, x^2 - 5*x + 4 = 0 ∧ y^2 - 5*y + 4 = 0 ∧ x = a 3 ∧ y = a 7) :
  a 5 = 2 :=
by
  sorry

end geometric_progression_a5_value_l72_72115


namespace remainder_23_pow_2003_mod_7_l72_72090

theorem remainder_23_pow_2003_mod_7 : 23 ^ 2003 % 7 = 4 :=
by sorry

end remainder_23_pow_2003_mod_7_l72_72090


namespace circle_tangent_to_x_axis_at_origin_l72_72045

theorem circle_tangent_to_x_axis_at_origin
  (D E F : ℝ)
  (h_circle : ∀ x y : ℝ, x^2 + y^2 + Dx + Ey + F = 0)
  (h_tangent : ∃ x, x^2 + (0 : ℝ)^2 + Dx + E * 0 + F = 0 ∧ ∃ r : ℝ, ∀ x y, x^2 + (y - r)^2 = r^2) :
  D = 0 ∧ E ≠ 0 ∧ F ≠ 0 :=
by
  sorry

end circle_tangent_to_x_axis_at_origin_l72_72045


namespace max_pies_without_ingredients_l72_72825

theorem max_pies_without_ingredients :
  let total_pies := 36
  let chocolate_pies := total_pies / 3
  let marshmallow_pies := total_pies / 4
  let cayenne_pies := total_pies / 2
  let soy_nuts_pies := total_pies / 8
  let max_ingredient_pies := max (max chocolate_pies marshmallow_pies) (max cayenne_pies soy_nuts_pies)
  total_pies - max_ingredient_pies = 18 :=
by
  sorry

end max_pies_without_ingredients_l72_72825


namespace susan_added_oranges_l72_72559

-- Conditions as definitions
def initial_oranges_in_box : ℝ := 55.0
def final_oranges_in_box : ℝ := 90.0

-- Define the quantity of oranges Susan put into the box
def susan_oranges := final_oranges_in_box - initial_oranges_in_box

-- Theorem statement to prove that the number of oranges Susan put into the box is 35.0
theorem susan_added_oranges : susan_oranges = 35.0 := by
  unfold susan_oranges
  sorry

end susan_added_oranges_l72_72559


namespace range_of_b_not_strictly_decreasing_l72_72628

def f (b x : ℝ) : ℝ := -x^3 + b*x^2 - (2*b + 3)*x + 2 - b

theorem range_of_b_not_strictly_decreasing :
  {b : ℝ | ¬(∀ (x1 x2 : ℝ), x1 < x2 → f b x1 > f b x2)} = {b | b < -1 ∨ b > 3} :=
by
  sorry

end range_of_b_not_strictly_decreasing_l72_72628


namespace triangle_QR_length_l72_72978

/-- Conditions for the triangles PQR and SQR sharing a side QR with given side lengths. -/
structure TriangleSetup where
  (PQ PR SR SQ QR : ℝ)
  (PQ_pos : PQ > 0)
  (PR_pos : PR > 0)
  (SR_pos : SR > 0)
  (SQ_pos : SQ > 0)
  (shared_side_QR : QR = QR)

/-- The problem statement asserting the least possible length of QR. -/
theorem triangle_QR_length (t : TriangleSetup) 
  (h1 : t.PQ = 8)
  (h2 : t.PR = 15)
  (h3 : t.SR = 10)
  (h4 : t.SQ = 25) :
  t.QR = 15 :=
by
  sorry

end triangle_QR_length_l72_72978


namespace chessboard_fraction_sum_l72_72035

theorem chessboard_fraction_sum (r s m n : ℕ) (h_r : r = 1296) (h_s : s = 204) (h_frac : (17 : ℚ) / 108 = (s : ℕ) / (r : ℕ)) : m + n = 125 :=
sorry

end chessboard_fraction_sum_l72_72035


namespace integer_solutions_to_equation_l72_72367

-- Define the problem statement in Lean 4
theorem integer_solutions_to_equation :
  ∀ (x y : ℤ), (x ≠ 0) → (y ≠ 0) → (1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 19) →
      (x, y) = (38, 38) ∨ (x, y) = (380, 20) ∨ (x, y) = (-342, 18) ∨ 
      (x, y) = (20, 380) ∨ (x, y) = (18, -342) :=
by
  sorry

end integer_solutions_to_equation_l72_72367


namespace find_m_for_one_real_solution_l72_72012

variables {m x : ℝ}

-- Given condition
def equation := (x + 4) * (x + 1) = m + 2 * x

-- The statement to prove
theorem find_m_for_one_real_solution : (∃ m : ℝ, m = 7 / 4 ∧ ∀ (x : ℝ), (x + 4) * (x + 1) = m + 2 * x) :=
by
  -- The proof starts here, which we will skip with sorry
  sorry

end find_m_for_one_real_solution_l72_72012


namespace intersection_of_sets_l72_72854

def A := { x : ℝ | x^2 - 2 * x - 8 < 0 }
def B := { x : ℝ | x >= 0 }
def intersection := { x : ℝ | 0 <= x ∧ x < 4 }

theorem intersection_of_sets : (A ∩ B) = intersection := 
sorry

end intersection_of_sets_l72_72854


namespace liam_total_time_l72_72518

noncomputable def total_time_7_laps : Nat :=
let time_first_200 := 200 / 5  -- Time in seconds for the first 200 meters
let time_next_300 := 300 / 6   -- Time in seconds for the next 300 meters
let time_per_lap := time_first_200 + time_next_300
let laps := 7
let total_time := laps * time_per_lap
total_time

theorem liam_total_time : total_time_7_laps = 630 := by
sorry

end liam_total_time_l72_72518


namespace initial_amount_celine_had_l72_72694

-- Define the costs and quantities
def laptop_cost : ℕ := 600
def smartphone_cost : ℕ := 400
def num_laptops : ℕ := 2
def num_smartphones : ℕ := 4
def change_received : ℕ := 200

-- Calculate costs and total amount
def cost_laptops : ℕ := num_laptops * laptop_cost
def cost_smartphones : ℕ := num_smartphones * smartphone_cost
def total_cost : ℕ := cost_laptops + cost_smartphones
def initial_amount : ℕ := total_cost + change_received

-- The statement to prove
theorem initial_amount_celine_had : initial_amount = 3000 := by
  sorry

end initial_amount_celine_had_l72_72694


namespace smallest_whole_number_divisible_by_8_leaves_remainder_1_l72_72436

theorem smallest_whole_number_divisible_by_8_leaves_remainder_1 :
  ∃ (n : ℕ), n ≡ 1 [MOD 2] ∧ n ≡ 1 [MOD 3] ∧ n ≡ 1 [MOD 4] ∧ n ≡ 1 [MOD 5] ∧ n ≡ 1 [MOD 7] ∧ n % 8 = 0 ∧ n = 7141 :=
by
  sorry

end smallest_whole_number_divisible_by_8_leaves_remainder_1_l72_72436


namespace solve_equation_l72_72277

theorem solve_equation (x : ℝ) (h : (3 * x) / (x + 1) = 9 / (x + 1)) : x = 3 :=
by sorry

end solve_equation_l72_72277


namespace train_length_l72_72255

noncomputable def speed_kmph := 80
noncomputable def time_seconds := 5

 noncomputable def speed_mps := (speed_kmph * 1000) / 3600

 noncomputable def length_train : ℝ := speed_mps * time_seconds

theorem train_length : length_train = 111.1 := by
  sorry

end train_length_l72_72255


namespace at_least_one_less_than_two_l72_72726

theorem at_least_one_less_than_two (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b > 2) :
  (1 + b) / a < 2 ∨ (1 + a) / b < 2 := by
sorry

end at_least_one_less_than_two_l72_72726


namespace max_min_values_f_decreasing_interval_f_l72_72271

noncomputable def a : ℝ × ℝ := (1 / 2, Real.sqrt 3 / 2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := ((a.1 * (b x).1) + (a.2 * (b x).2)) + 2

theorem max_min_values_f (k : ℤ) :
  (∃ (x1 : ℝ), (x1 = 2 * k * Real.pi + Real.pi / 6) ∧ f x1 = 3) ∧
  (∃ (x2 : ℝ), (x2 = 2 * k * Real.pi - 5 * Real.pi / 6) ∧ f x2 = 1) := 
sorry

theorem decreasing_interval_f :
  ∀ x, (Real.pi / 6 ≤ x ∧ x ≤ 7 * Real.pi / 6) → (∀ y, f x ≥ f y → x ≤ y) := 
sorry

end max_min_values_f_decreasing_interval_f_l72_72271


namespace rectangle_length_l72_72615

/--
The perimeter of a rectangle is 150 cm. The length is 15 cm greater than the width.
This theorem proves that the length of the rectangle is 45 cm under these conditions.
-/
theorem rectangle_length (P w l : ℝ) (h1 : P = 150) (h2 : l = w + 15) (h3 : P = 2 * l + 2 * w) : l = 45 :=
by
  sorry

end rectangle_length_l72_72615


namespace john_will_lose_weight_in_80_days_l72_72477

-- Assumptions based on the problem conditions
def calories_eaten : ℕ := 1800
def calories_burned : ℕ := 2300
def calories_to_lose_one_pound : ℕ := 4000
def pounds_to_lose : ℕ := 10

-- Definition of the net calories burned per day
def net_calories_burned_per_day : ℕ := calories_burned - calories_eaten

-- Definition of total calories to lose the target weight
def total_calories_to_lose_target_weight (pounds_to_lose : ℕ) : ℕ :=
  calories_to_lose_one_pound * pounds_to_lose

-- Definition of days to lose the target weight
def days_to_lose_weight (target_calories : ℕ) (daily_net_calories : ℕ) : ℕ :=
  target_calories / daily_net_calories

-- Prove that John will lose 10 pounds in 80 days
theorem john_will_lose_weight_in_80_days :
  days_to_lose_weight (total_calories_to_lose_target_weight pounds_to_lose) net_calories_burned_per_day = 80 := by
  sorry

end john_will_lose_weight_in_80_days_l72_72477


namespace sum_of_three_consecutive_odd_integers_l72_72239

theorem sum_of_three_consecutive_odd_integers (x : ℤ) 
  (h1 : x % 2 = 1)                          -- x is odd
  (h2 : (x + 4) % 2 = 1)                    -- x+4 is odd
  (h3 : x + (x + 4) = 150)                  -- sum of first and third integers is 150
  : x + (x + 2) + (x + 4) = 225 :=          -- sum of the three integers is 225
by
  sorry

end sum_of_three_consecutive_odd_integers_l72_72239


namespace determinant_value_l72_72807

-- Given definitions and conditions
def determinant (a b c d : ℤ) : ℤ := a * d - b * c
def special_determinant (m : ℤ) : ℤ := determinant (m^2) (m-3) (1-2*m) (m-2)

-- The proof problem
theorem determinant_value (m : ℤ) (h : m^2 - 2 * m - 3 = 0) : special_determinant m = 9 := sorry

end determinant_value_l72_72807


namespace projection_matrix_solution_l72_72981

theorem projection_matrix_solution 
  (a c : ℚ) 
  (P : Matrix (Fin 2) (Fin 2) ℚ := ![![a, 18/45], ![c, 27/45]])
  (hP : P * P = P) :
  (a, c) = (9/25, 12/25) :=
by
  sorry

end projection_matrix_solution_l72_72981


namespace total_rainfall_cm_l72_72772

theorem total_rainfall_cm :
  let monday := 0.12962962962962962
  let tuesday := 3.5185185185185186 * 0.1
  let wednesday := 0.09259259259259259
  let thursday := 0.10222222222222223 * 2.54
  let friday := 12.222222222222221 * 0.1
  let saturday := 0.2222222222222222
  let sunday := 0.17444444444444446 * 2.54
  monday + tuesday + wednesday + thursday + friday + saturday + sunday = 2.721212629851652 :=
by
  sorry

end total_rainfall_cm_l72_72772


namespace find_b_value_l72_72741

variable (a p q b : ℝ)
variable (h1 : p * 0 + q * (3 * a) + b * 1 = 1)
variable (h2 : p * (9 * a) + q * (-1) + b * 2 = 1)
variable (h3 : p * 0 + q * (3 * a) + b * 0 = 1)

theorem find_b_value : b = 0 :=
by
  sorry

end find_b_value_l72_72741


namespace smallest_gcd_bc_l72_72296

theorem smallest_gcd_bc (a b c : ℕ) (h1 : Nat.gcd a b = 240) (h2 : Nat.gcd a c = 1001) : Nat.gcd b c = 1 :=
sorry

end smallest_gcd_bc_l72_72296


namespace quadratic_eq_equal_roots_l72_72629

theorem quadratic_eq_equal_roots (m x : ℝ) (h : (x^2 - m * x + m - 1 = 0) ∧ ((x - 1)^2 = 0)) : 
    m = 2 ∧ ((x = 1 ∧ x = 1)) :=
by
  sorry

end quadratic_eq_equal_roots_l72_72629


namespace nuts_per_cookie_l72_72654

theorem nuts_per_cookie (h1 : (1/4:ℝ) * 60 = 15)
(h2 : (0.40:ℝ) * 60 = 24)
(h3 : 60 - 15 - 24 = 21)
(h4 : 72 / (15 + 21) = 2) :
72 / 36 = 2 := by
suffices h : 72 / 36 = 2 from h
exact h4

end nuts_per_cookie_l72_72654


namespace abs_inequality_interval_notation_l72_72796

variable (x : ℝ)

theorem abs_inequality_interval_notation :
  {x : ℝ | |x - 1| < 1} = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

end abs_inequality_interval_notation_l72_72796


namespace mary_number_l72_72306

-- Definitions of the properties and conditions
def is_two_digit_number (x : ℕ) : Prop :=
  10 ≤ x ∧ x < 100

def switch_digits (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  10 * b + a

def conditions_met (x : ℕ) : Prop :=
  is_two_digit_number x ∧ 91 ≤ switch_digits (4 * x - 7) ∧ switch_digits (4 * x - 7) ≤ 95

-- The statement to prove
theorem mary_number : ∃ x : ℕ, conditions_met x ∧ x = 14 :=
by {
  sorry
}

end mary_number_l72_72306


namespace canvas_bag_lower_carbon_solution_l72_72232

theorem canvas_bag_lower_carbon_solution :
  let canvas_release_oz := 9600
  let plastic_per_trip_oz := 32
  canvas_release_oz / plastic_per_trip_oz = 300 :=
by
  sorry

end canvas_bag_lower_carbon_solution_l72_72232


namespace find_k_l72_72546

theorem find_k (x y k : ℝ) (h_line : 2 - k * x = -4 * y) (h_point : x = 3 ∧ y = -2) : k = -2 :=
by
  -- Given the conditions that the point (3, -2) lies on the line 2 - kx = -4y, 
  -- we want to prove that k = -2
  sorry

end find_k_l72_72546


namespace smallest_positive_period_l72_72952

noncomputable def tan_period (a b x : ℝ) : ℝ := 
  Real.tan ((a + b) * x / 2)

theorem smallest_positive_period 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  ∃ p > 0, ∀ x, tan_period a b (x + p) = tan_period a b x ∧ p = 2 * Real.pi :=
by
  sorry

end smallest_positive_period_l72_72952


namespace sin_pi_minus_alpha_l72_72705

theorem sin_pi_minus_alpha (α : ℝ) (h : Real.sin α = 5/13) : Real.sin (π - α) = 5/13 :=
by
  sorry

end sin_pi_minus_alpha_l72_72705


namespace fraction_spent_at_toy_store_l72_72941

theorem fraction_spent_at_toy_store 
  (total_allowance : ℝ)
  (arcade_fraction : ℝ)
  (candy_store_amount : ℝ) 
  (remaining_allowance : ℝ)
  (toy_store_amount : ℝ)
  (H1 : total_allowance = 2.40)
  (H2 : arcade_fraction = 3 / 5)
  (H3 : candy_store_amount = 0.64)
  (H4 : remaining_allowance = total_allowance - (arcade_fraction * total_allowance))
  (H5 : toy_store_amount = remaining_allowance - candy_store_amount) :
  toy_store_amount / remaining_allowance = 1 / 3 := 
sorry

end fraction_spent_at_toy_store_l72_72941


namespace value_of_a_2015_l72_72502

def a : ℕ → Int
| 0 => 1
| 1 => 5
| n+2 => a (n+1) - a n

theorem value_of_a_2015 : a 2014 = -5 := by
  sorry

end value_of_a_2015_l72_72502


namespace calculate_savings_l72_72714

/-- Given the income is 19000 and the income to expenditure ratio is 5:4, prove the savings of 3800. -/
theorem calculate_savings (i : ℕ) (exp : ℕ) (rat : ℕ → ℕ → Prop)
  (h_income : i = 19000)
  (h_ratio : rat 5 4)
  (h_exp_eq : ∃ x, i = 5 * x ∧ exp = 4 * x) :
  i - exp = 3800 :=
by 
  sorry

end calculate_savings_l72_72714


namespace comb_7_2_equals_21_l72_72021

theorem comb_7_2_equals_21 : (Nat.choose 7 2) = 21 := by
  sorry

end comb_7_2_equals_21_l72_72021


namespace man_double_son_age_in_two_years_l72_72020

theorem man_double_son_age_in_two_years (S M Y : ℕ) (h1 : S = 14) (h2 : M = S + 16) (h3 : Y = 2) : 
  M + Y = 2 * (S + Y) :=
by
  sorry

-- Explanation:
-- h1 establishes the son's current age.
-- h2 establishes the man's current age in relation to the son's age.
-- h3 gives the solution Y = 2 years.
-- We need to prove that M + Y = 2 * (S + Y).

end man_double_son_age_in_two_years_l72_72020


namespace abc_one_eq_sum_l72_72698

theorem abc_one_eq_sum (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a * b * c = 1) :
  (a^2 * b^2) / ((a^2 + b * c) * (b^2 + a * c))
  + (a^2 * c^2) / ((a^2 + b * c) * (c^2 + a * b))
  + (b^2 * c^2) / ((b^2 + a * c) * (c^2 + a * b))
  = 1 / (a^2 + 1 / a) + 1 / (b^2 + 1 / b) + 1 / (c^2 + 1 / c) := by
  sorry

end abc_one_eq_sum_l72_72698


namespace ratio_of_art_to_math_books_l72_72577

-- The conditions provided
def total_budget : ℝ := 500
def price_math_book : ℝ := 20
def num_math_books : ℕ := 4
def num_art_books : ℕ := num_math_books
def price_art_book : ℝ := 20
def num_science_books : ℕ := num_math_books + 6
def price_science_book : ℝ := 10
def cost_music_books : ℝ := 160

-- Desired proof statement
theorem ratio_of_art_to_math_books : num_art_books / num_math_books = 1 :=
by
  sorry

end ratio_of_art_to_math_books_l72_72577


namespace shifted_sine_monotonically_increasing_l72_72728

noncomputable def shifted_sine_function (x : ℝ) : ℝ :=
  3 * Real.sin (2 * x - (2 * Real.pi / 3))

theorem shifted_sine_monotonically_increasing :
  ∀ x y : ℝ, (x ∈ Set.Icc (Real.pi / 12) (7 * Real.pi / 12)) → (y ∈ Set.Icc (Real.pi / 12) (7 * Real.pi / 12)) → x < y → shifted_sine_function x < shifted_sine_function y :=
by
  sorry

end shifted_sine_monotonically_increasing_l72_72728


namespace pencils_ratio_l72_72263

theorem pencils_ratio (T S Ti : ℕ) 
  (h1 : T = 6 * S)
  (h2 : T = 12)
  (h3 : Ti = 16) : Ti / S = 8 := by
  sorry

end pencils_ratio_l72_72263


namespace intersection_point_in_polar_coordinates_l72_72702

theorem intersection_point_in_polar_coordinates (theta : ℝ) (rho : ℝ) (h₁ : theta = π / 3) (h₂ : rho = 2 * Real.cos theta) (h₃ : rho > 0) : rho = 1 :=
by
  -- Proof skipped
  sorry

end intersection_point_in_polar_coordinates_l72_72702


namespace evaluate_expression_l72_72243

noncomputable def a : ℕ := 3^2 + 5^2 + 7^2
noncomputable def b : ℕ := 2^2 + 4^2 + 6^2

theorem evaluate_expression : (a / b : ℚ) - (b / a : ℚ) = 3753 / 4656 :=
by
  sorry

end evaluate_expression_l72_72243


namespace dressing_p_percentage_l72_72143

-- Define the percentages of vinegar and oil in dressings p and q
def vinegar_in_p : ℝ := 0.30
def vinegar_in_q : ℝ := 0.10

-- Define the desired percentage of vinegar in the new dressing
def vinegar_in_new_dressing : ℝ := 0.12

-- Define the total mass of the new dressing
def total_mass_new_dressing : ℝ := 100.0

-- Define the mass of dressing p in the new dressing
def mass_of_p (x : ℝ) : ℝ := x

-- Define the mass of dressing q in the new dressing
def mass_of_q (x : ℝ) : ℝ := total_mass_new_dressing - x

-- Define the amount of vinegar contributed by dressings p and q
def vinegar_from_p (x : ℝ) : ℝ := vinegar_in_p * mass_of_p x
def vinegar_from_q (x : ℝ) : ℝ := vinegar_in_q * mass_of_q x

-- Define the total vinegar in the new dressing
def total_vinegar (x : ℝ) : ℝ := vinegar_from_p x + vinegar_from_q x

-- Problem statement: prove the percentage of dressing p in the new dressing
theorem dressing_p_percentage (x : ℝ) (hx : total_vinegar x = vinegar_in_new_dressing * total_mass_new_dressing) :
  (mass_of_p x / total_mass_new_dressing) * 100 = 10 :=
by
  sorry

end dressing_p_percentage_l72_72143


namespace lcm_14_18_20_l72_72448

theorem lcm_14_18_20 : Nat.lcm (Nat.lcm 14 18) 20 = 1260 :=
by
  -- Define the prime factorizations
  have fact_14 : 14 = 2 * 7 := by norm_num
  have fact_18 : 18 = 2 * 3^2 := by norm_num
  have fact_20 : 20 = 2^2 * 5 := by norm_num
  
  -- Calculate the LCM based on the highest powers of each prime
  have lcm : Nat.lcm (Nat.lcm 14 18) 20 = 2^2 * 3^2 * 5 * 7 :=
    by
      sorry -- Proof details are not required

  -- Final verification that this calculation matches 1260
  exact lcm

end lcm_14_18_20_l72_72448


namespace xsquared_plus_5x_minus_6_condition_l72_72655

theorem xsquared_plus_5x_minus_6_condition (x : ℝ) : 
  (x^2 + 5 * x - 6 > 0) → (x > 2) ∨ (((x > 1) ∨ (x < -6)) ∧ ¬(x > 2)) := 
sorry

end xsquared_plus_5x_minus_6_condition_l72_72655


namespace non_right_triangle_option_l72_72186

-- Definitions based on conditions
def optionA (A B C : ℝ) : Prop := A + B = C
def optionB (A B C : ℝ) : Prop := A - B = C
def optionC (A B C : ℝ) : Prop := A / B = 1 / 2 ∧ B / C = 2 / 3
def optionD (A B C : ℝ) : Prop := A = B ∧ A = 3 * C

-- Given conditions for a right triangle
def is_right_triangle (A B C : ℝ) : Prop := ∃(θ : ℝ), θ = 90 ∧ (A = θ ∨ B = θ ∨ C = θ)

-- The proof problem
theorem non_right_triangle_option (A B C : ℝ) :
  optionD A B C ∧ ¬(is_right_triangle A B C) := sorry

end non_right_triangle_option_l72_72186


namespace problem1_problem2_problem3_l72_72132

-- 1. Prove that (3ab³)² = 9a²b⁶
theorem problem1 (a b : ℝ) : (3 * a * b^3)^2 = 9 * a^2 * b^6 :=
by sorry

-- 2. Prove that x ⋅ x³ + x² ⋅ x² = 2x⁴
theorem problem2 (x : ℝ) : x * x^3 + x^2 * x^2 = 2 * x^4 :=
by sorry

-- 3. Prove that (12x⁴ - 6x³) ÷ 3x² = 4x² - 2x
theorem problem3 (x : ℝ) : (12 * x^4 - 6 * x^3) / (3 * x^2) = 4 * x^2 - 2 * x :=
by sorry

end problem1_problem2_problem3_l72_72132


namespace sqrt_factorial_mul_factorial_l72_72396

theorem sqrt_factorial_mul_factorial :
  (Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24) := by
  sorry

end sqrt_factorial_mul_factorial_l72_72396


namespace range_of_f_l72_72064

open Set

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_of_f : range f = {y : ℝ | y ≠ 1} :=
by
  sorry

end range_of_f_l72_72064


namespace find_m_n_l72_72176

def f (x : ℝ) (m : ℝ) (n : ℝ) : ℝ := x^3 + m * x^2 + n * x + 1

theorem find_m_n (m n : ℝ) (x : ℝ) (hx : x ≠ 0 ∧ f x m n = 1 ∧ (3 * x^2 + 2 * m * x + n = 0) ∧ (∀ y, f y m n ≥ -31 ∧ f (-2) m n = -31)) :
  m = 12 ∧ n = 36 :=
sorry

end find_m_n_l72_72176


namespace Larry_spends_108_minutes_l72_72155

-- Define conditions
def half_hour_twice_daily := 30 * 2
def fifth_of_an_hour_daily := 60 / 5
def quarter_hour_twice_daily := 15 * 2
def tenth_of_an_hour_daily := 60 / 10

-- Define total times spent on each pet
def total_time_dog := half_hour_twice_daily + fifth_of_an_hour_daily
def total_time_cat := quarter_hour_twice_daily + tenth_of_an_hour_daily

-- Define the total time spent on pets
def total_time_pets := total_time_dog + total_time_cat

-- Lean theorem statement
theorem Larry_spends_108_minutes : total_time_pets = 108 := 
  by 
    sorry

end Larry_spends_108_minutes_l72_72155


namespace surface_area_of_circumscribing_sphere_l72_72264

theorem surface_area_of_circumscribing_sphere :
  let l := 2
  let h := 3
  let d := Real.sqrt (l^2 + l^2 + h^2)
  let r := d / 2
  let A := 4 * Real.pi * r^2
  A = 17 * Real.pi :=
by
  let l := 2
  let h := 3
  let d := Real.sqrt (l^2 + l^2 + h^2)
  let r := d / 2
  let A := 4 * Real.pi * r^2
  show A = 17 * Real.pi
  sorry

end surface_area_of_circumscribing_sphere_l72_72264


namespace find_principal_l72_72907

theorem find_principal (P : ℝ) (r : ℝ) (t : ℝ) (CI SI : ℝ) 
  (h_r : r = 0.20) 
  (h_t : t = 2) 
  (h_diff : CI - SI = 144) 
  (h_CI : CI = P * (1 + r)^t - P) 
  (h_SI : SI = P * r * t) : 
  P = 3600 :=
by
  sorry

end find_principal_l72_72907


namespace mass_of_man_l72_72548

def boat_length : ℝ := 3 -- boat length in meters
def boat_breadth : ℝ := 2 -- boat breadth in meters
def boat_sink_depth : ℝ := 0.01 -- boat sink depth in meters
def water_density : ℝ := 1000 -- density of water in kg/m^3

/- Theorem: The mass of the man is equal to 60 kg given the parameters defined above. -/
theorem mass_of_man : (water_density * (boat_length * boat_breadth * boat_sink_depth)) = 60 :=
by
  simp [boat_length, boat_breadth, boat_sink_depth, water_density]
  sorry

end mass_of_man_l72_72548


namespace determine_C_l72_72321
noncomputable def A : ℕ := sorry
noncomputable def B : ℕ := sorry
noncomputable def C : ℕ := sorry

-- Conditions
axiom cond1 : A + B + 1 = C + 10
axiom cond2 : B = A + 2

-- Proof statement
theorem determine_C : C = 1 :=
by {
  -- using the given conditions, deduce that C must equal 1
  sorry
}

end determine_C_l72_72321


namespace acquaintances_at_ends_equal_l72_72446

theorem acquaintances_at_ends_equal 
  (n : ℕ) -- number of participants
  (a b : ℕ → ℕ) -- functions which return the number of acquaintances before/after for each participant
  (h_ai_bi : ∀ (i : ℕ), 1 < i ∧ i < n → a i = b i) -- condition for participants except first and last
  (h_a1 : a 1 = 0) -- the first person has no one before them
  (h_bn : b n = 0) -- the last person has no one after them
  :
  a n = b 1 :=
by
  sorry

end acquaintances_at_ends_equal_l72_72446


namespace find_other_number_l72_72649

theorem find_other_number (A B : ℕ) (hcf : ℕ) (lcm : ℕ) 
  (H1 : hcf = 12) 
  (H2 : lcm = 312) 
  (H3 : A = 24) 
  (H4 : hcf * lcm = A * B) : 
  B = 156 :=
by sorry

end find_other_number_l72_72649


namespace yards_in_a_mile_l72_72339

def mile_eq_furlongs : Prop := 1 = 5 * 1
def furlong_eq_rods : Prop := 1 = 50 * 1
def rod_eq_yards : Prop := 1 = 5 * 1

theorem yards_in_a_mile (h1 : mile_eq_furlongs) (h2 : furlong_eq_rods) (h3 : rod_eq_yards) :
  1 * (5 * (50 * 5)) = 1250 :=
by
-- Given conditions, translate them:
-- h1 : 1 mile = 5 furlongs -> 1 * 1 = 5 * 1
-- h2 : 1 furlong = 50 rods -> 1 * 1 = 50 * 1
-- h3 : 1 rod = 5 yards -> 1 * 1 = 5 * 1
-- Prove that the number of yards in one mile is 1250
sorry

end yards_in_a_mile_l72_72339


namespace find_k_parallel_vectors_l72_72814

def vector_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem find_k_parallel_vectors (k : ℝ) :
  let a := (1, k)
  let b := (-2, 6)
  vector_parallel a b → k = -3 :=
by
  sorry

end find_k_parallel_vectors_l72_72814


namespace kiley_slices_eaten_l72_72007

def slices_of_cheesecake (total_calories_per_cheesecake calories_per_slice : ℕ) : ℕ :=
  total_calories_per_cheesecake / calories_per_slice

def slices_eaten (total_slices percentage_ate : ℚ) : ℚ :=
  total_slices * percentage_ate

theorem kiley_slices_eaten :
  ∀ (total_calories_per_cheesecake calories_per_slice : ℕ) (percentage_ate : ℚ),
  total_calories_per_cheesecake = 2800 →
  calories_per_slice = 350 →
  percentage_ate = (25 / 100 : ℚ) →
  slices_eaten (slices_of_cheesecake total_calories_per_cheesecake calories_per_slice) percentage_ate = 2 :=
by
  intros total_calories_per_cheesecake calories_per_slice percentage_ate h1 h2 h3
  rw [h1, h2, h3]
  sorry

end kiley_slices_eaten_l72_72007


namespace range_of_a_l72_72465

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x

theorem range_of_a {a : ℝ} :
  (∀ x > 1, f a x > 1) → a ∈ Set.Ici 1 := by
  sorry

end range_of_a_l72_72465


namespace necessary_but_not_sufficient_condition_geometric_sequence_l72_72939

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n ≥ 1 → a (n + 1) = a n * a (n - 1) / a n 

theorem necessary_but_not_sufficient_condition_geometric_sequence (a : ℕ → ℝ) (h : ∀ n : ℕ, n ≥ 2 → a (n + 1) * a (n - 1) = a n ^ 2) :
  (is_geometric_sequence a → (∀ n : ℕ, n ≥ 2 → a (n + 1) * a (n - 1) = a n ^ 2)) ∧ (∃ b : ℕ → ℝ, (b n = 0 ∨ b n = b (n - 1) ∨ b n = b (n + 1)) ∧ ¬ is_geometric_sequence b) := 
sorry

end necessary_but_not_sufficient_condition_geometric_sequence_l72_72939


namespace sequence_solution_l72_72836

-- Define the sequence x_n
def x (n : ℕ) : ℚ := n / (n + 2016)

-- Given condition: x_2016 = x_m * x_n
theorem sequence_solution (m n : ℕ) (h : x 2016 = x m * x n) : 
  m = 4032 ∧ n = 6048 := 
  by sorry

end sequence_solution_l72_72836


namespace right_triangle_to_acute_triangle_l72_72101

theorem right_triangle_to_acute_triangle 
  (a b c d : ℝ) (h_triangle : a^2 + b^2 = c^2) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_increase : d > 0):
  (a + d)^2 + (b + d)^2 > (c + d)^2 := 
by {
  sorry
}

end right_triangle_to_acute_triangle_l72_72101


namespace reciprocal_problem_l72_72473

theorem reciprocal_problem (x : ℚ) (h : 8 * x = 5) : 150 * (x⁻¹) = 240 := 
by 
  sorry

end reciprocal_problem_l72_72473


namespace prob_both_hit_prob_exactly_one_hit_prob_at_least_one_hit_l72_72200

-- Define events and their probabilities.
def prob_A : ℝ := 0.8
def prob_B : ℝ := 0.8

-- Given P(A and B) = P(A) * P(B)
def prob_AB : ℝ := prob_A * prob_B

-- Statements to prove
theorem prob_both_hit : prob_AB = 0.64 :=
by
  -- P(A and B) = 0.8 * 0.8 = 0.64
  exact sorry

theorem prob_exactly_one_hit : (prob_A * (1 - prob_B) + (1 - prob_A) * prob_B) = 0.32 :=
by
  -- P(A and not B) + P(not A and B) = 0.8 * 0.2 + 0.2 * 0.8 = 0.32
  exact sorry

theorem prob_at_least_one_hit : (1 - (1 - prob_A) * (1 - prob_B)) = 0.96 :=
by
  -- 1 - P(not A and not B) = 1 - 0.04 = 0.96
  exact sorry

end prob_both_hit_prob_exactly_one_hit_prob_at_least_one_hit_l72_72200


namespace find_b_value_l72_72919

-- Definitions based on the problem conditions
def line_bisects_circle (b : ℝ) : Prop :=
  ∃ c : ℝ × ℝ, (c.fst = 4 ∧ c.snd = -1) ∧
                (c.snd = c.fst + b)

-- Theorem statement for the problem
theorem find_b_value : line_bisects_circle (-5) :=
by
  sorry

end find_b_value_l72_72919


namespace cubic_root_of_determinant_l72_72330

open Complex 
open Matrix

noncomputable def matrix_d (a b c n : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![
    ![b + n^3 * c, n * (c - b), n^2 * (b - c)],
    ![n^2 * (c - a), c + n^3 * a, n * (a - c)],
    ![n * (b - a), n^2 * (a - b), a + n^3 * b]
  ]

theorem cubic_root_of_determinant (a b c n : ℂ) (h : a * b * c = 1) :
  (det (matrix_d a b c n))^(1/3 : ℂ) = n^3 + 1 :=
  sorry

end cubic_root_of_determinant_l72_72330


namespace negative_870_in_third_quadrant_l72_72849

noncomputable def angle_in_third_quadrant (theta : ℝ) : Prop :=
  180 < theta ∧ theta < 270

theorem negative_870_in_third_quadrant:
  angle_in_third_quadrant 210 :=
by
  sorry

end negative_870_in_third_quadrant_l72_72849


namespace stops_time_proof_l72_72013

variable (departure_time arrival_time driving_time stop_time_in_minutes : ℕ)
variable (h_departure : departure_time = 7 * 60)
variable (h_arrival : arrival_time = 20 * 60)
variable (h_driving : driving_time = 12 * 60)
variable (total_minutes := arrival_time - departure_time)

theorem stops_time_proof :
  stop_time_in_minutes = (total_minutes - driving_time) := by
  sorry

end stops_time_proof_l72_72013


namespace worker_total_amount_l72_72190

-- Definitions of the conditions
def pay_per_day := 20
def deduction_per_idle_day := 3
def total_days := 60
def idle_days := 40
def worked_days := total_days - idle_days
def earnings := worked_days * pay_per_day
def deductions := idle_days * deduction_per_idle_day

-- Statement of the problem
theorem worker_total_amount : earnings - deductions = 280 := by
  sorry

end worker_total_amount_l72_72190


namespace middle_integer_of_sum_is_120_l72_72566

-- Define the condition that three consecutive integers sum to 360
def consecutive_integers_sum_to (n : ℤ) (sum : ℤ) : Prop :=
  (n - 1) + n + (n + 1) = sum

-- The statement to prove
theorem middle_integer_of_sum_is_120 (n : ℤ) :
  consecutive_integers_sum_to n 360 → n = 120 :=
by
  sorry

end middle_integer_of_sum_is_120_l72_72566


namespace smallest_positive_int_linear_combination_l72_72892

theorem smallest_positive_int_linear_combination (m n : ℤ) :
  ∃ k : ℤ, 4509 * m + 27981 * n = k ∧ k > 0 ∧ k ≤ 4509 * m + 27981 * n → k = 3 :=
by
  sorry

end smallest_positive_int_linear_combination_l72_72892


namespace true_proposition_l72_72947

-- Define the propositions p and q
def p : Prop := ∃ x0 : ℝ, x0 ^ 2 - x0 + 1 ≥ 0

def q : Prop := ∀ (a b : ℝ), a < b → 1 / a > 1 / b

-- Prove that p ∧ ¬q is true
theorem true_proposition : p ∧ ¬q :=
by
  sorry

end true_proposition_l72_72947


namespace number_of_sheets_l72_72087

theorem number_of_sheets (S E : ℕ) (h1 : S - E = 60) (h2 : 5 * E = S) : S = 150 := by
  sorry

end number_of_sheets_l72_72087


namespace find_m_l72_72722

theorem find_m {m : ℝ} :
  (4 - m) / (m + 2) = 1 → m = 1 :=
by
  sorry

end find_m_l72_72722


namespace tenth_term_arithmetic_sequence_l72_72212

theorem tenth_term_arithmetic_sequence (a d : ℤ)
  (h1 : a + 3 * d = 23)
  (h2 : a + 7 * d = 55) :
  a + 9 * d = 71 :=
sorry

end tenth_term_arithmetic_sequence_l72_72212


namespace compare_negative_fractions_l72_72236

theorem compare_negative_fractions :
  (-3 : ℚ) / 4 > (-4 : ℚ) / 5 :=
sorry

end compare_negative_fractions_l72_72236


namespace find_XY_in_306090_triangle_l72_72004

-- Definitions of the problem
def angleZ := 90
def angleX := 60
def hypotenuseXZ := 12
def isRightTriangle (XYZ : Type) (angleZ : ℕ) : Prop := angleZ = 90
def is306090Triangle (XYZ : Type) (angleX : ℕ) (angleZ : ℕ) : Prop := (angleX = 60) ∧ (angleZ = 90)

-- Lean theorem statement
theorem find_XY_in_306090_triangle 
  (XYZ : Type)
  (hypotenuseXZ : ℕ)
  (h1 : isRightTriangle XYZ angleZ)
  (h2 : is306090Triangle XYZ angleX angleZ) :
  XY = 8 := 
sorry

end find_XY_in_306090_triangle_l72_72004


namespace exists_integer_K_l72_72022

theorem exists_integer_K (Z : ℕ) (K : ℕ) : 
  1000 < Z ∧ Z < 2000 ∧ Z = K^4 → 
  ∃ K, K = 6 := 
by
  sorry

end exists_integer_K_l72_72022


namespace distinct_square_roots_l72_72407

theorem distinct_square_roots (m : ℝ) (h : 2 * m - 4 ≠ 3 * m - 1) : ∃ n : ℝ, (2 * m - 4) * (2 * m - 4) = n ∧ (3 * m - 1) * (3 * m - 1) = n ∧ n = 4 :=
by
  sorry

end distinct_square_roots_l72_72407


namespace find_parameters_infinite_solutions_l72_72424

def system_has_infinite_solutions (a b : ℝ) :=
  ∀ x y : ℝ, 2 * (a - b) * x + 6 * y = a ∧ 3 * b * x + (a - b) * b * y = 1

theorem find_parameters_infinite_solutions :
  ∀ (a b : ℝ), 
  system_has_infinite_solutions a b ↔ 
    (a = (3 + Real.sqrt 17) / 2 ∧ b = (Real.sqrt 17 - 3) / 2) ∨
    (a = (3 - Real.sqrt 17) / 2 ∧ b = (-3 - Real.sqrt 17) / 2) ∨
    (a = -2 ∧ b = 1) ∨
    (a = -1 ∧ b = 2) :=
sorry

end find_parameters_infinite_solutions_l72_72424


namespace frosting_time_difference_l72_72281

def normally_frost_time_per_cake := 5
def sprained_frost_time_per_cake := 8
def number_of_cakes := 10

theorem frosting_time_difference :
  (sprained_frost_time_per_cake * number_of_cakes) -
  (normally_frost_time_per_cake * number_of_cakes) = 30 :=
by
  sorry

end frosting_time_difference_l72_72281


namespace bug_crawl_distance_l72_72044

theorem bug_crawl_distance : 
  let start : ℤ := 3
  let first_stop : ℤ := -4
  let second_stop : ℤ := 7
  let final_stop : ℤ := -1
  |first_stop - start| + |second_stop - first_stop| + |final_stop - second_stop| = 26 := 
by
  sorry

end bug_crawl_distance_l72_72044


namespace henry_friend_fireworks_l72_72535

-- Definitions of variables and conditions
variable 
  (F : ℕ) -- Number of fireworks Henry's friend bought

-- Main theorem statement
theorem henry_friend_fireworks (h1 : 6 + 2 + F = 11) : F = 3 :=
by
  sorry

end henry_friend_fireworks_l72_72535


namespace find_k_l72_72177

theorem find_k : ∀ (x y k : ℤ), (x = -y) → (2 * x + 5 * y = k) → (x - 3 * y = 16) → (k = -12) :=
by
  intros x y k h1 h2 h3
  sorry

end find_k_l72_72177


namespace ratio_x_y_l72_72858

noncomputable def side_length_x (x : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
    a = 5 ∧ b = 12 ∧ c = 13 ∧ 
    (12 - x) / x = 5 / 12 ∧
    12 * x = 5 * x + 60 ∧
    7 * x = 60

noncomputable def side_length_y (y : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
    a = 5 ∧ b = 12 ∧ c = 13 ∧
    y = 60 / 17

theorem ratio_x_y (x y : ℝ) (hx : side_length_x x) (hy : side_length_y y) : x / y = 17 / 7 :=
by
  sorry

end ratio_x_y_l72_72858


namespace book_pages_l72_72536

theorem book_pages (x : ℕ) : 
  (x - (1/5 * x + 12)) - (1/4 * (x - (1/5 * x + 12)) + 15) - (1/3 * ((x - (1/5 * x + 12)) - (1/4 * (x - (1/5 * x + 12)) + 15)) + 18) = 62 →
  x = 240 :=
by
  -- This is where the proof would go, but it's omitted for this task.
  sorry

end book_pages_l72_72536


namespace total_games_played_l72_72810

theorem total_games_played (n : ℕ) (h : n = 7) : (n.choose 2) = 21 := by
  sorry

end total_games_played_l72_72810


namespace simplify_expression_l72_72059

variable (a : ℝ) (ha : a ≠ -3)

theorem simplify_expression : (a^2) / (a + 3) - 9 / (a + 3) = a - 3 :=
by
  sorry

end simplify_expression_l72_72059


namespace johns_sister_age_l72_72913

variable (j d s : ℝ)

theorem johns_sister_age 
  (h1 : j = d - 15)
  (h2 : j + d = 100)
  (h3 : s = j - 5) :
  s = 37.5 := 
sorry

end johns_sister_age_l72_72913


namespace earnings_bc_l72_72224

variable (A B C : ℕ)

theorem earnings_bc :
  A + B + C = 600 →
  A + C = 400 →
  C = 100 →
  B + C = 300 :=
by
  intros h1 h2 h3
  sorry

end earnings_bc_l72_72224


namespace determinant_expression_l72_72972

noncomputable def matrixDet (α β : ℝ) : ℝ :=
  Matrix.det ![
    ![Real.sin α * Real.cos β, -Real.sin α * Real.sin β, Real.cos α],
    ![-Real.sin β, -Real.cos β, 0],
    ![Real.cos α * Real.cos β, Real.cos α * Real.sin β, Real.sin α]]

theorem determinant_expression (α β: ℝ) : matrixDet α β = Real.sin α ^ 3 := 
by 
  sorry

end determinant_expression_l72_72972


namespace jerry_age_is_13_l72_72524

variable (M J : ℕ)

theorem jerry_age_is_13 (h1 : M = 2 * J - 6) (h2 : M = 20) : J = 13 := by
  sorry

end jerry_age_is_13_l72_72524


namespace sum_of_cubes_divisible_by_nine_l72_72242

theorem sum_of_cubes_divisible_by_nine (n : ℕ) (h : 0 < n) : 9 ∣ (n^3 + (n + 1)^3 + (n + 2)^3) :=
by sorry

end sum_of_cubes_divisible_by_nine_l72_72242


namespace calculate_expression_l72_72504

theorem calculate_expression : 1453 - 250 * 2 + 130 / 5 = 979 := by
  sorry

end calculate_expression_l72_72504


namespace difference_between_length_and_breadth_l72_72291

theorem difference_between_length_and_breadth (L W : ℝ) (h1 : W = 1/2 * L) (h2 : L * W = 800) : L - W = 20 :=
by
  sorry

end difference_between_length_and_breadth_l72_72291


namespace joint_purchases_popular_joint_purchases_unpopular_among_neighbors_l72_72767

section JointPurchases

/-- Given that joint purchases allow significant cost savings, reduced overhead costs,
improved quality assessment, and community trust, prove that joint purchases 
are popular in many countries despite the risks. -/
theorem joint_purchases_popular
    (cost_savings : Prop)
    (reduced_overhead_costs : Prop)
    (improved_quality_assessment : Prop)
    (community_trust : Prop)
    : Prop :=
    cost_savings ∧ reduced_overhead_costs ∧ improved_quality_assessment ∧ community_trust

/-- Given that high transaction costs, organizational difficulties,
convenience of proximity to stores, and potential disputes are challenges for neighbors,
prove that joint purchases of groceries and household goods are unpopular among neighbors. -/
theorem joint_purchases_unpopular_among_neighbors
    (high_transaction_costs : Prop)
    (organizational_difficulties : Prop)
    (convenience_proximity : Prop)
    (potential_disputes : Prop)
    : Prop :=
    high_transaction_costs ∧ organizational_difficulties ∧ convenience_proximity ∧ potential_disputes

end JointPurchases

end joint_purchases_popular_joint_purchases_unpopular_among_neighbors_l72_72767


namespace candy_not_chocolate_l72_72125

theorem candy_not_chocolate (candy_total : ℕ) (bags : ℕ) (choc_heart_bags : ℕ) (choc_kiss_bags : ℕ) : 
  candy_total = 63 ∧ bags = 9 ∧ choc_heart_bags = 2 ∧ choc_kiss_bags = 3 → 
  (candy_total - (choc_heart_bags * (candy_total / bags) + choc_kiss_bags * (candy_total / bags))) = 28 :=
by
  intros h
  sorry

end candy_not_chocolate_l72_72125


namespace gcd_lcm_sum_correct_l72_72294

def gcd_lcm_sum : ℕ :=
  let gcd_40_60 := Nat.gcd 40 60
  let lcm_20_15 := Nat.lcm 20 15
  gcd_40_60 + 2 * lcm_20_15

theorem gcd_lcm_sum_correct : gcd_lcm_sum = 140 := by
  -- Definitions based on conditions
  let gcd_40_60 := Nat.gcd 40 60
  let lcm_20_15 := Nat.lcm 20 15
  
  -- sorry to skip the proof
  sorry

end gcd_lcm_sum_correct_l72_72294


namespace probability_x_gt_3y_in_rectangle_l72_72141

noncomputable def probability_of_x_gt_3y :ℝ :=
  let base := 2010
  let height := 2011
  let triangle_height := 670
  (1/2 * base * triangle_height) / (base * height)

theorem probability_x_gt_3y_in_rectangle:
  probability_of_x_gt_3y = 335 / 2011 := 
by
  sorry

end probability_x_gt_3y_in_rectangle_l72_72141


namespace stacy_height_now_l72_72875

-- Definitions based on the given conditions
def S_initial : ℕ := 50
def J_initial : ℕ := 45
def J_growth : ℕ := 1
def S_growth : ℕ := J_growth + 6

-- Prove statement about Stacy's current height
theorem stacy_height_now : S_initial + S_growth = 57 := by
  sorry

end stacy_height_now_l72_72875


namespace mark_increase_reading_time_l72_72353

theorem mark_increase_reading_time : 
  (let hours_per_day := 2
   let days_per_week := 7
   let desired_weekly_hours := 18
   let current_weekly_hours := hours_per_day * days_per_week
   let increase_per_week := desired_weekly_hours - current_weekly_hours
   increase_per_week = 4) :=
by
  let hours_per_day := 2
  let days_per_week := 7
  let desired_weekly_hours := 18
  let current_weekly_hours := hours_per_day * days_per_week
  let increase_per_week := desired_weekly_hours - current_weekly_hours
  have h1 : current_weekly_hours = 14 := by norm_num
  have h2 : increase_per_week = desired_weekly_hours - current_weekly_hours := rfl
  have h3 : increase_per_week = 18 - 14 := by rw [h2, h1]
  have h4 : increase_per_week = 4 := by norm_num
  exact h4

end mark_increase_reading_time_l72_72353


namespace area_of_region_inside_circle_outside_rectangle_l72_72016

theorem area_of_region_inside_circle_outside_rectangle
  (EF FH : ℝ)
  (hEF : EF = 6)
  (hFH : FH = 5)
  (r : ℝ)
  (h_radius : r = (EF^2 + FH^2).sqrt) :
  π * r^2 - EF * FH = 61 * π - 30 :=
by
  sorry

end area_of_region_inside_circle_outside_rectangle_l72_72016


namespace power_sum_l72_72508

theorem power_sum (h : (9 : ℕ) = 3^2) : (2^567 + (9^5 / 3^2) : ℕ) = 2^567 + 6561 := by
  sorry

end power_sum_l72_72508


namespace middle_number_is_10_l72_72519

theorem middle_number_is_10 (A B C : ℝ) (h1 : B - C = A - B) (h2 : A * B = 85) (h3 : B * C = 115) : B = 10 :=
by
  sorry

end middle_number_is_10_l72_72519


namespace equilateral_triangle_l72_72495

variable (A B C A₀ B₀ C₀ : Type) [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup A₀] [AddGroup B₀] [AddGroup C₀]

variable (midpoint : ∀ (X₁ X₂ : Type), Type) 
variable (circumcircle : ∀ (X Y Z : Type), Type)

def medians_meet_circumcircle := ∀ (A A₁ B B₁ C C₁ : Type) 
  [AddGroup A] [AddGroup A₁] [AddGroup B] [AddGroup B₁] [AddGroup C] [AddGroup C₁], 
  Prop

def areas_equal := ∀ (ABC₀ AB₀C A₀BC : Type) 
  [AddGroup ABC₀] [AddGroup AB₀C] [AddGroup A₀BC], 
  Prop

theorem equilateral_triangle (A B C A₀ B₀ C₀ A₁ B₁ C₁ : Type)
  [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup A₀] [AddGroup B₀] [AddGroup C₀]
  [AddGroup A₁] [AddGroup B₁] [AddGroup C₁] 
  (midpoint_cond : ∀ (X Y Z : Type), Z = midpoint X Y)
  (circumcircle_cond : ∀ (X Y Z : Type), Z = circumcircle X Y Z)
  (medians_meet_circumcircle : Prop)
  (areas_equal: Prop) :
    A = B ∧ B = C ∧ C = A :=
  sorry

end equilateral_triangle_l72_72495


namespace sum_of_cubes_l72_72922

theorem sum_of_cubes (x y z : ℝ) (h1 : x + y + z = 7) (h2 : xy + xz + yz = 9) (h3 : xyz = -18) :
  x^3 + y^3 + z^3 = 100 :=
by
  sorry

end sum_of_cubes_l72_72922


namespace guess_probability_greater_than_two_thirds_l72_72278

theorem guess_probability_greater_than_two_thirds :
  (1335 : ℝ) / 2002 > 2 / 3 :=
by {
  -- Placeholder for proof
  sorry
}

end guess_probability_greater_than_two_thirds_l72_72278


namespace pau_total_ordered_correct_l72_72604

-- Define the initial pieces of fried chicken ordered by Kobe
def kobe_order : ℝ := 5

-- Define Pau's initial order as twice Kobe's order plus 2.5 pieces
def pau_initial_order : ℝ := (2 * kobe_order) + 2.5

-- Define Shaquille's initial order as 50% more than Pau's initial order
def shaq_initial_order : ℝ := pau_initial_order * 1.5

-- Define the total pieces of chicken Pau will have eaten by the end
def pau_total_ordered : ℝ := 2 * pau_initial_order

-- Prove that Pau will have eaten 25 pieces of fried chicken by the end
theorem pau_total_ordered_correct : pau_total_ordered = 25 := by
  sorry

end pau_total_ordered_correct_l72_72604


namespace unique_position_all_sequences_one_l72_72650

-- Define the main theorem
theorem unique_position_all_sequences_one (n : ℕ) (sequences : Fin (2^(n-1)) → Fin n → Bool) :
  (∀ a b c : Fin (2^(n-1)), ∃ p : Fin n, sequences a p = true ∧ sequences b p = true ∧ sequences c p = true) →
  ∃! p : Fin n, ∀ i : Fin (2^(n-1)), sequences i p = true :=
by
  sorry

end unique_position_all_sequences_one_l72_72650


namespace number_of_geese_l72_72733

theorem number_of_geese (A x n k : ℝ) 
  (h1 : A = k * x * n)
  (h2 : A = (k + 20) * x * (n - 75))
  (h3 : A = (k - 15) * x * (n + 100)) 
  : n = 300 :=
sorry

end number_of_geese_l72_72733


namespace cooking_time_l72_72114

theorem cooking_time (total_potatoes cooked_potatoes potato_time : ℕ) 
    (h1 : total_potatoes = 15) 
    (h2 : cooked_potatoes = 6) 
    (h3 : potato_time = 8) : 
    total_potatoes - cooked_potatoes * potato_time = 72 :=
by
    sorry

end cooking_time_l72_72114


namespace possible_values_x_l72_72195

theorem possible_values_x : 
  let x := Nat.gcd 112 168 
  ∃ d : Finset ℕ, d.card = 8 ∧ ∀ y ∈ d, y ∣ 112 ∧ y ∣ 168 := 
by
  let x := Nat.gcd 112 168
  have : x = 56 := by norm_num
  use Finset.filter (fun n => 56 % n = 0) (Finset.range 57)
  sorry

end possible_values_x_l72_72195


namespace time_in_1876_minutes_from_6AM_is_116PM_l72_72394

def minutesToTime (startTime : Nat) (minutesToAdd : Nat) : Nat × Nat :=
  let totalMinutes := startTime + minutesToAdd
  let totalHours := totalMinutes / 60
  let remainderMinutes := totalMinutes % 60
  let resultHours := (totalHours % 24)
  (resultHours, remainderMinutes)

theorem time_in_1876_minutes_from_6AM_is_116PM :
  minutesToTime (6 * 60) 1876 = (13, 16) :=
  sorry

end time_in_1876_minutes_from_6AM_is_116PM_l72_72394


namespace right_angled_triangle_solution_l72_72307

-- Define the necessary constants
def t : ℝ := 504 -- area in cm^2
def c : ℝ := 65 -- hypotenuse in cm

-- The definitions of the right-angled triangle's properties
def is_right_angled_triangle (a b : ℝ) : Prop :=
  a ^ 2 + b ^ 2 = c ^ 2 ∧ a * b = 2 * t

-- The proof problem statement
theorem right_angled_triangle_solution :
  ∃ (a b : ℝ), is_right_angled_triangle a b ∧ ((a = 63 ∧ b = 16) ∨ (a = 16 ∧ b = 63)) :=
sorry

end right_angled_triangle_solution_l72_72307


namespace computation_result_l72_72040

theorem computation_result : 143 - 13 + 31 + 17 = 178 := 
by
  sorry

end computation_result_l72_72040


namespace percentage_students_left_in_classroom_l72_72582

def total_students : ℕ := 250
def fraction_painting : ℚ := 3 / 10
def fraction_field : ℚ := 2 / 10
def fraction_science : ℚ := 1 / 5

theorem percentage_students_left_in_classroom :
  let gone_painting := total_students * fraction_painting
  let gone_field := total_students * fraction_field
  let gone_science := total_students * fraction_science
  let students_gone := gone_painting + gone_field + gone_science
  let students_left := total_students - students_gone
  (students_left / total_students) * 100 = 30 :=
by sorry

end percentage_students_left_in_classroom_l72_72582


namespace empty_atm_l72_72432

theorem empty_atm (a : ℕ → ℕ) (b : ℕ → ℕ) (h1 : a 9 < b 9)
    (h2 : ∀ k : ℕ, 1 ≤ k → k ≤ 8 → a k ≠ b k) 
    (n : ℕ) (h₀ : n = 1) : 
    ∃ (sequence : ℕ → ℕ), (∀ i, sequence i ≤ n) → (∀ k, ∃ i, k > i → sequence k = 0) :=
sorry

end empty_atm_l72_72432


namespace inverse_proportion_quad_l72_72427

theorem inverse_proportion_quad (k : ℝ) : (∀ x : ℝ, x > 0 → (k + 1) / x < 0) ∧ (∀ x : ℝ, x < 0 → (k + 1) / x > 0) ↔ k < -1 :=
by
  sorry

end inverse_proportion_quad_l72_72427


namespace number_of_integer_solutions_l72_72357

theorem number_of_integer_solutions : 
  ∃ S : Finset ℤ, (∀ x ∈ S, (x + 3)^2 ≤ 4) ∧ S.card = 5 := by
  sorry

end number_of_integer_solutions_l72_72357


namespace net_profit_100_patches_l72_72329

theorem net_profit_100_patches :
  let cost_per_patch := 1.25
  let num_patches_ordered := 100
  let selling_price_per_patch := 12.00
  let total_cost := cost_per_patch * num_patches_ordered
  let total_revenue := selling_price_per_patch * num_patches_ordered
  let net_profit := total_revenue - total_cost
  net_profit = 1075 :=
by
  sorry

end net_profit_100_patches_l72_72329


namespace cole_average_speed_l72_72363

noncomputable def cole_average_speed_to_work : ℝ :=
  let time_to_work := 1.2
  let return_trip_speed := 105
  let total_round_trip_time := 2
  let time_to_return := total_round_trip_time - time_to_work
  let distance_to_work := return_trip_speed * time_to_return
  distance_to_work / time_to_work

theorem cole_average_speed : cole_average_speed_to_work = 70 := by
  sorry

end cole_average_speed_l72_72363


namespace train_route_l72_72420

-- Definition of letter positions
def letter_position : Char → Nat
| 'A' => 1
| 'B' => 2
| 'K' => 11
| 'L' => 12
| 'U' => 21
| 'V' => 22
| _ => 0

-- Definition of decode function
def decode (s : List Nat) : String :=
match s with
| [21, 2, 12, 21] => "Baku"
| [21, 22, 12, 21] => "Ufa"
| _ => ""

-- Assert encoded strings
def departure_encoded : List Nat := [21, 2, 12, 21]
def arrival_encoded : List Nat := [21, 22, 12, 21]

-- Theorem statement
theorem train_route :
  decode departure_encoded = "Ufa" ∧ decode arrival_encoded = "Baku" :=
by
  sorry

end train_route_l72_72420


namespace area_of_rectangular_garden_l72_72503

-- Definitions based on conditions
def width : ℕ := 15
def length : ℕ := 3 * width
def area : ℕ := length * width

-- The theorem we want to prove
theorem area_of_rectangular_garden : area = 675 :=
by sorry

end area_of_rectangular_garden_l72_72503


namespace cos_identity_l72_72692

theorem cos_identity (α : ℝ) : 
  3.4028 * (Real.cos α)^4 + 4 * (Real.cos α)^3 - 8 * (Real.cos α)^2 - 3 * (Real.cos α) + 1 = 
  2 * (Real.cos (7 * α / 2)) * (Real.cos (α / 2)) := 
by sorry

end cos_identity_l72_72692


namespace milkshakes_per_hour_l72_72164

variable (L : ℕ) -- number of milkshakes Luna can make per hour

theorem milkshakes_per_hour
  (h1 : ∀ (A : ℕ), A = 3) -- Augustus makes 3 milkshakes per hour
  (h2 : ∀ (H : ℕ), H = 8) -- they have been making milkshakes for 8 hours
  (h3 : ∀ (Total : ℕ), Total = 80) -- together they made 80 milkshakes
  (h4 : ∀ (Augustus_milkshakes : ℕ), Augustus_milkshakes = 3 * 8) -- Augustus made 24 milkshakes in 8 hours
 : L = 7 := sorry

end milkshakes_per_hour_l72_72164


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l72_72632

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l72_72632


namespace no_such_m_for_equivalence_existence_of_m_for_implication_l72_72668

def P (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
def S (x : ℝ) (m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem no_such_m_for_equivalence :
  ¬ ∃ m : ℝ, ∀ x : ℝ, P x ↔ S x m :=
sorry

theorem existence_of_m_for_implication :
  ∃ m : ℝ, (∀ x : ℝ, S x m → P x) ∧ m ≤ 3 :=
sorry

end no_such_m_for_equivalence_existence_of_m_for_implication_l72_72668


namespace sum_of_coefficients_l72_72416

-- Define a namespace to encapsulate the problem
namespace PolynomialCoefficients

-- Problem statement as a Lean theorem
theorem sum_of_coefficients (α β : ℂ) (h1 : α + β = 1) (h2 : α * β = 1) :
  α^2005 + β^2005 = 1 :=
sorry -- Placeholder for the proof

end PolynomialCoefficients

end sum_of_coefficients_l72_72416


namespace area_of_region_l72_72312

theorem area_of_region : 
  (∃ (x y : ℝ), x^2 + y^2 + 6 * x - 4 * y - 11 = 0) -> 
  ∃ (A : ℝ), A = 24 * Real.pi :=
by 
  sorry

end area_of_region_l72_72312


namespace no_nat_solutions_l72_72883

theorem no_nat_solutions (x y : ℕ) : (2 * x + y) * (2 * y + x) ≠ 2017 ^ 2017 := by sorry

end no_nat_solutions_l72_72883


namespace correct_addition_result_l72_72431

-- Definitions corresponding to the conditions
def mistaken_addend := 240
def correct_addend := 420
def incorrect_sum := 390

-- The proof statement
theorem correct_addition_result : 
  (incorrect_sum - mistaken_addend + correct_addend) = 570 :=
by
  sorry

end correct_addition_result_l72_72431


namespace abs_value_identity_l72_72977

theorem abs_value_identity (a : ℝ) (h : a + |a| = 0) : a - |2 * a| = 3 * a :=
by
  sorry

end abs_value_identity_l72_72977


namespace part1_part2_l72_72991

-- Definition of the branches of the hyperbola
def C1 (P : ℝ × ℝ) : Prop := P.1 * P.2 = 1
def C2 (P : ℝ × ℝ) : Prop := P.1 * P.2 = 1

-- Problem Part 1: Proving that P, Q, and R cannot lie on the same branch
theorem part1 (P Q R : ℝ × ℝ) (hP : C1 P) (hQ : C1 Q) (hR : C1 R) : False := by
  sorry

-- Problem Part 2: Finding the coordinates of Q and R
theorem part2 : 
  ∃ Q R : ℝ × ℝ, C1 Q ∧ C1 R ∧ 
                (Q = (2 - Real.sqrt 3, 1 / (2 - Real.sqrt 3))) ∧ 
                (R = (2 + Real.sqrt 3, 1 / (2 + Real.sqrt 3))) := 
by
  sorry

end part1_part2_l72_72991


namespace num_points_within_and_on_boundary_is_six_l72_72410

noncomputable def num_points_within_boundary : ℕ :=
  let points := [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (4, 1)]
  points.length

theorem num_points_within_and_on_boundary_is_six :
  num_points_within_boundary = 6 :=
  by
    -- proof steps would go here
    sorry

end num_points_within_and_on_boundary_is_six_l72_72410


namespace smallest_integer_satisfying_conditions_l72_72018

-- Define the conditions explicitly as hypotheses
def satisfies_congruence_3_2 (n : ℕ) : Prop :=
  n % 3 = 2

def satisfies_congruence_7_2 (n : ℕ) : Prop :=
  n % 7 = 2

def satisfies_congruence_8_2 (n : ℕ) : Prop :=
  n % 8 = 2

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

-- Define the smallest positive integer satisfying the above conditions
theorem smallest_integer_satisfying_conditions : ∃ (n : ℕ), n > 1 ∧ satisfies_congruence_3_2 n ∧ satisfies_congruence_7_2 n ∧ satisfies_congruence_8_2 n ∧ is_perfect_square n :=
  by
    sorry

end smallest_integer_satisfying_conditions_l72_72018


namespace simplify_expression_l72_72605

theorem simplify_expression (a : ℝ) : (2 * a - 3)^2 - (a + 5) * (a - 5) = 3 * a^2 - 12 * a + 34 :=
by
  sorry

end simplify_expression_l72_72605


namespace question_eq_answer_l72_72467

theorem question_eq_answer (w x y z k : ℕ) 
  (h : 2^w * 3^x * 5^y * 7^z * 11^k = 2520) : 
  2 * w + 3 * x + 5 * y + 7 * z + 11 * k = 24 :=
sorry

end question_eq_answer_l72_72467


namespace functional_equation_solution_l72_72560

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), (∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y) →
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
by
  sorry

end functional_equation_solution_l72_72560


namespace theta_plus_2phi_eq_pi_div_4_l72_72964

noncomputable def theta (θ : ℝ) (φ : ℝ) : Prop := 
  ((Real.tan θ = 5 / 12) ∧ 
   (Real.sin φ = 1 / 2) ∧ 
   (0 < θ ∧ θ < Real.pi / 2) ∧ 
   (0 < φ ∧ φ < Real.pi / 2)  )

theorem theta_plus_2phi_eq_pi_div_4 (θ φ : ℝ) (h : theta θ φ) : 
    θ + 2 * φ = Real.pi / 4 :=
by 
  sorry

end theta_plus_2phi_eq_pi_div_4_l72_72964


namespace variance_daily_reading_time_l72_72755

theorem variance_daily_reading_time :
  let mean10 := 2.7
  let var10 := 1
  let num10 := 800

  let mean11 := 3.1
  let var11 := 2
  let num11 := 600

  let mean12 := 3.3
  let var12 := 3
  let num12 := 600

  let num_total := num10 + num11 + num12

  let total_mean := (2.7 * 800 + 3.1 * 600 + 3.3 * 600) / 2000

  let var_total := (800 / 2000) * (1 + (2.7 - total_mean)^2) +
                   (600 / 2000) * (2 + (3.1 - total_mean)^2) +
                   (600 / 2000) * (3 + (3.3 - total_mean)^2)

  var_total = 1.966 :=
by
  sorry

end variance_daily_reading_time_l72_72755


namespace find_y_value_l72_72043

theorem find_y_value (x y : ℝ) 
    (h1 : x^2 + 3 * x + 6 = y - 2) 
    (h2 : x = -5) : 
    y = 18 := 
  by 
  sorry

end find_y_value_l72_72043


namespace john_total_spent_l72_72373

-- Define the initial conditions
def other_toys_cost : ℝ := 1000
def lightsaber_cost : ℝ := 2 * other_toys_cost

-- Define the total cost spent by John
def total_cost : ℝ := other_toys_cost + lightsaber_cost

-- Prove that the total cost is $3000
theorem john_total_spent :
  total_cost = 3000 :=
by
  -- Sorry will be used to skip the proof
  sorry

end john_total_spent_l72_72373


namespace sqrt_450_eq_15_sqrt_2_l72_72887

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l72_72887


namespace problem_proof_l72_72824

theorem problem_proof (n : ℕ) 
  (h : ∃ k, 2 * k = n) :
  4 ∣ n :=
sorry

end problem_proof_l72_72824


namespace max_true_statements_l72_72304

theorem max_true_statements (y : ℝ) :
  (0 < y^3 ∧ y^3 < 2 → ∀ (y : ℝ),  y^3 > 2 → False) ∧
  ((-2 < y ∧ y < 0) → ∀ (y : ℝ), (0 < y ∧ y < 2) → False) →
  ∃ (s1 s2 : Prop), 
    ((0 < y^3 ∧ y^3 < 2) = s1 ∨ (y^3 > 2) = s1 ∨ (-2 < y ∧ y < 0) = s1 ∨ (0 < y ∧ y < 2) = s1 ∨ (0 < y - y^3 ∧ y - y^3 < 2) = s1) ∧
    ((0 < y^3 ∧ y^3 < 2) = s2 ∨ (y^3 > 2) = s2 ∨ (-2 < y ∧ y < 0) = s2 ∨ (0 < y ∧ y < 2) = s2 ∨ (0 < y - y^3 ∧ y - y^3 < 2) = s2) ∧ 
    (s1 ∧ s2) → 
    ∃ m : ℕ, m = 2 := 
sorry

end max_true_statements_l72_72304


namespace gear_q_revolutions_per_minute_l72_72068

-- Define the constants and conditions
def revolutions_per_minute_p : ℕ := 10
def revolutions_per_minute_q : ℕ := sorry
def time_in_minutes : ℝ := 1.5
def extra_revolutions_q : ℕ := 45

-- Calculate the number of revolutions for gear p in 90 seconds
def revolutions_p_in_90_seconds := revolutions_per_minute_p * time_in_minutes

-- Condition that gear q makes exactly 45 more revolutions than gear p in 90 seconds
def revolutions_q_in_90_seconds := revolutions_p_in_90_seconds + extra_revolutions_q

-- Correct answer
def correct_answer : ℕ := 40

-- Prove that gear q makes 40 revolutions per minute
theorem gear_q_revolutions_per_minute : 
    revolutions_per_minute_q = correct_answer :=
sorry

end gear_q_revolutions_per_minute_l72_72068


namespace custom_op_example_l72_72662

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x

theorem custom_op_example : (custom_op 7 4) - (custom_op 4 7) = -9 :=
by
  sorry

end custom_op_example_l72_72662


namespace compute_ratio_l72_72254

theorem compute_ratio (x y z a : ℝ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : z ≠ x) (h4 : x + y + z = a) (h5 : a ≠ 0) :
  (xy + yz + zx) / (x^2 + y^2 + z^2) = 1 / 3 :=
by
  -- Proof will be filled in here
  sorry

end compute_ratio_l72_72254


namespace distance_focus_directrix_l72_72406

theorem distance_focus_directrix (θ : ℝ) : 
  (∃ d : ℝ, (∀ (ρ : ℝ), ρ = 5 / (3 - 2 * Real.cos θ)) ∧ d = 5 / 2) :=
sorry

end distance_focus_directrix_l72_72406


namespace rectangular_prism_cut_corners_edges_l72_72682

def original_edges : Nat := 12
def corners : Nat := 8
def new_edges_per_corner : Nat := 3
def total_new_edges : Nat := corners * new_edges_per_corner

theorem rectangular_prism_cut_corners_edges :
  original_edges + total_new_edges = 36 := sorry

end rectangular_prism_cut_corners_edges_l72_72682


namespace infinite_series_sum_l72_72487

theorem infinite_series_sum : 
  ∑' k : ℕ, (8 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1))) = 2 :=
by 
  sorry

end infinite_series_sum_l72_72487


namespace calculate_cells_after_12_days_l72_72863

theorem calculate_cells_after_12_days :
  let initial_cells := 5
  let division_factor := 3
  let days := 12
  let period := 3
  let n := days / period
  initial_cells * division_factor ^ (n - 1) = 135 := by
  sorry

end calculate_cells_after_12_days_l72_72863


namespace incorrect_equation_is_wrong_l72_72704

-- Specifications and conditions
def speed_person_a : ℝ := 7
def speed_person_b : ℝ := 6.5
def head_start : ℝ := 5

-- Define the time variable
variable (x : ℝ)

-- The correct equation based on the problem statement
def correct_equation : Prop := speed_person_a * x - head_start = speed_person_b * x

-- The incorrect equation to prove incorrect
def incorrect_equation : Prop := speed_person_b * x = speed_person_a * x - head_start

-- The Lean statement to prove that the incorrect equation is indeed incorrect
theorem incorrect_equation_is_wrong (h : correct_equation x) : ¬ incorrect_equation x := by
  sorry

end incorrect_equation_is_wrong_l72_72704


namespace second_to_last_digit_of_special_number_l72_72369

theorem second_to_last_digit_of_special_number :
  ∀ (N : ℕ), (N % 10 = 0) ∧ (∃ k : ℕ, k > 0 ∧ N = 2 * 5^k) →
  (N / 10) % 10 = 5 :=
by
  sorry

end second_to_last_digit_of_special_number_l72_72369


namespace denote_depth_below_sea_level_l72_72001

theorem denote_depth_below_sea_level (above_sea_level : Int) (depth_haidou_1 : Int) :
  (above_sea_level > 0) ∧ (depth_haidou_1 < 0) → depth_haidou_1 = -10907 :=
by
  intros h
  sorry

end denote_depth_below_sea_level_l72_72001


namespace functional_relationship_l72_72608

variable (x y k1 k2 : ℝ)

axiom h1 : y = k1 * x + k2 / (x - 2)
axiom h2 : (y = -1) ↔ (x = 1)
axiom h3 : (y = 5) ↔ (x = 3)

theorem functional_relationship :
  (∀ x y, y = k1 * x + k2 / (x - 2) ∧
    ((x = 1) → y = -1) ∧
    ((x = 3) → y = 5) → y = x + 2 / (x - 2)) :=
by
  sorry

end functional_relationship_l72_72608


namespace shaded_area_l72_72201

def radius (R : ℝ) : Prop := R > 0
def angle (α : ℝ) : Prop := α = 20 * (Real.pi / 180)

theorem shaded_area (R : ℝ) (hR : radius R) (hα : angle (20 * (Real.pi / 180))) :
  let S0 := Real.pi * R^2 / 2
  let sector_radius := 2 * R
  let sector_angle := 20 * (Real.pi / 180)
  (2 * sector_radius * sector_radius * sector_angle / 2) / sector_angle = 2 * Real.pi * R^2 / 9 :=
by
  sorry

end shaded_area_l72_72201


namespace mary_average_speed_l72_72459

noncomputable def trip_distance : ℝ := 1.5 + 1.5
noncomputable def trip_time_minutes : ℝ := 45 + 15
noncomputable def trip_time_hours : ℝ := trip_time_minutes / 60

theorem mary_average_speed :
  (trip_distance / trip_time_hours) = 3 := by
  sorry

end mary_average_speed_l72_72459


namespace james_final_weight_l72_72194

noncomputable def initial_weight : ℝ := 120
noncomputable def muscle_gain : ℝ := 0.20 * initial_weight
noncomputable def fat_gain : ℝ := muscle_gain / 4
noncomputable def final_weight (initial_weight muscle_gain fat_gain : ℝ) : ℝ :=
  initial_weight + muscle_gain + fat_gain

theorem james_final_weight :
  final_weight initial_weight muscle_gain fat_gain = 150 :=
by
  sorry

end james_final_weight_l72_72194


namespace length_of_common_chord_l72_72259

theorem length_of_common_chord (x y : ℝ) :
  (x + 1)^2 + (y - 3)^2 = 9 ∧ x^2 + y^2 - 4 * x + 2 * y - 11 = 0 → 
  ∃ l : ℝ, l = 24 / 5 :=
by
  sorry

end length_of_common_chord_l72_72259


namespace tunnel_depth_l72_72819

theorem tunnel_depth (topWidth : ℝ) (bottomWidth : ℝ) (area : ℝ) (h : ℝ)
  (h1 : topWidth = 15)
  (h2 : bottomWidth = 5)
  (h3 : area = 400)
  (h4 : area = (1 / 2) * (topWidth + bottomWidth) * h) :
  h = 40 := 
sorry

end tunnel_depth_l72_72819


namespace kayla_apples_l72_72862

variable (x y : ℕ)
variable (h1 : x + (10 + 4 * x) = 340)
variable (h2 : y = 10 + 4 * x)

theorem kayla_apples : y = 274 :=
by
  sorry

end kayla_apples_l72_72862


namespace tan_alpha_second_quadrant_l72_72761

theorem tan_alpha_second_quadrant (α : ℝ) 
(h_cos : Real.cos α = -4/5) 
(h_quadrant : π/2 < α ∧ α < π) : 
  Real.tan α = -3/4 :=
by
  sorry

end tan_alpha_second_quadrant_l72_72761


namespace candies_of_different_flavors_l72_72345

theorem candies_of_different_flavors (total_treats chewing_gums chocolate_bars : ℕ) (h1 : total_treats = 155) (h2 : chewing_gums = 60) (h3 : chocolate_bars = 55) :
  total_treats - (chewing_gums + chocolate_bars) = 40 := 
by 
  sorry

end candies_of_different_flavors_l72_72345


namespace one_div_m_plus_one_div_n_l72_72606

theorem one_div_m_plus_one_div_n
  {m n : ℕ} 
  (h1 : Nat.gcd m n = 5) 
  (h2 : Nat.lcm m n = 210)
  (h3 : m + n = 75) :
  (1 : ℚ) / m + (1 : ℚ) / n = 1 / 14 :=
by
  sorry

end one_div_m_plus_one_div_n_l72_72606


namespace taylor_probability_l72_72930

open Nat Real

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  n.choose k

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial_coefficient n k : ℚ) * p^k * (1 - p)^(n - k)

theorem taylor_probability :
  (binomial_probability 5 2 (3/5) = 144 / 625) :=
by
  sorry

end taylor_probability_l72_72930


namespace minimum_selling_price_l72_72659

def monthly_sales : ℕ := 50
def base_cost : ℕ := 1200
def shipping_cost : ℕ := 20
def store_fee : ℕ := 10000
def repair_fee : ℕ := 5000
def profit_margin : ℕ := 20

def total_monthly_expenses : ℕ := store_fee + repair_fee
def total_cost_per_machine : ℕ := base_cost + shipping_cost + total_monthly_expenses / monthly_sales
def min_selling_price : ℕ := total_cost_per_machine * (1 + profit_margin / 100)

theorem minimum_selling_price : min_selling_price = 1824 := 
by
  sorry 

end minimum_selling_price_l72_72659


namespace venus_speed_mph_l72_72053

theorem venus_speed_mph (speed_mps : ℝ) (seconds_per_hour : ℝ) (mph : ℝ) 
  (h1 : speed_mps = 21.9) 
  (h2 : seconds_per_hour = 3600)
  (h3 : mph = speed_mps * seconds_per_hour) : 
  mph = 78840 := 
  by 
  sorry

end venus_speed_mph_l72_72053


namespace min_value_problem_inequality_solution_l72_72069

-- Definition of the function
noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x + 2|

-- Part (i): Minimum value problem
theorem min_value_problem (a : ℝ) (minF : ∀ x : ℝ, f x a ≥ 2) : a = 0 ∨ a = -4 :=
by
  sorry

-- Part (ii): Inequality solving problem
theorem inequality_solution (x : ℝ) (a : ℝ := 2) : f x a ≤ 6 ↔ -3 ≤ x ∧ x ≤ 3 :=
by
  sorry

end min_value_problem_inequality_solution_l72_72069


namespace min_segments_for_octagon_perimeter_l72_72126

/-- Given an octagon formed by cutting a smaller rectangle from a larger rectangle,
the minimum number of distinct line segment lengths needed to calculate the perimeter 
of this octagon is 3. --/
theorem min_segments_for_octagon_perimeter (a b c d e f g h : ℝ)
  (cond : a = c ∧ b = d ∧ e = g ∧ f = h) :
  ∃ (u v w : ℝ), u ≠ v ∧ v ≠ w ∧ u ≠ w :=
by
  sorry

end min_segments_for_octagon_perimeter_l72_72126


namespace eccentricity_of_hyperbola_l72_72350

-- Definitions and conditions
def hyperbola (x y a b : ℝ) : Prop :=
  (a > 0 ∧ b > 0 ∧ a > b) ∧ (x^2 / a^2 - y^2 / b^2 = 1)

def regular_hexagon_side_length (a b c : ℝ) : Prop :=
  2 * a = (Real.sqrt 3 + 1) * c

-- Goal: Prove the eccentricity of the hyperbola
theorem eccentricity_of_hyperbola (a b c : ℝ) (x y : ℝ) :
  hyperbola x y a b →
  regular_hexagon_side_length a b c →
  2 * a = (Real.sqrt 3 + 1) * c →
  c ≠ 0 →
  a ≠ 0 →
  b ≠ 0 →
  (c / a = Real.sqrt 3 + 1) :=
by
  intros h_hyp h_hex h_eq h_c_ne_zero h_a_ne_zero h_b_ne_zero
  sorry -- Proof goes here

end eccentricity_of_hyperbola_l72_72350


namespace quadratic_roster_method_l72_72845

theorem quadratic_roster_method :
  {x : ℝ | x^2 - 3 * x + 2 = 0} = {1, 2} :=
by
  sorry

end quadratic_roster_method_l72_72845


namespace tangent_line_hyperbola_eq_l72_72565

noncomputable def tangent_line_ellipse (a b x0 y0 x y : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a > 0) 
  (h_ell : x0 ^ 2 / a ^ 2 + y0 ^ 2 / b ^ 2 = 1) : Prop :=
  (x0 * x) / (a ^ 2) + (y0 * y) / (b ^ 2) = 1

noncomputable def tangent_line_hyperbola (a b x0 y0 x y : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h_hyp : x0 ^ 2 / a ^ 2 - y0 ^ 2 / b ^ 2 = 1) : Prop :=
  (x0 * x) / (a ^ 2) - (y0 * y) / (b ^ 2) = 1

theorem tangent_line_hyperbola_eq (a b x0 y0 x y : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a > 0)
  (h_ellipse_tangent : tangent_line_ellipse a b x0 y0 x y h1 h2 h3 (by sorry))
  (h_hyperbola : x0 ^ 2 / a ^ 2 - y0 ^ 2 / b ^ 2 = 1) : 
  tangent_line_hyperbola a b x0 y0 x y h3 h2 h_hyperbola :=
by sorry

end tangent_line_hyperbola_eq_l72_72565


namespace last_four_digits_of_5_pow_2017_l72_72885

theorem last_four_digits_of_5_pow_2017 : (5 ^ 2017) % 10000 = 3125 :=
by sorry

end last_four_digits_of_5_pow_2017_l72_72885


namespace exists_sequence_a_l72_72031

def c (n : ℕ) : ℕ := 2017 ^ n

axiom f : ℕ → ℝ

axiom condition_1 : ∀ m n : ℕ, f (m + n) ≤ 2017 * f m * f (n + 325)

axiom condition_2 : ∀ n : ℕ, 0 < f (c (n + 1)) ∧ f (c (n + 1)) < (f (c n)) ^ 2017

theorem exists_sequence_a :
  ∃ (a : ℕ → ℕ), ∀ n k : ℕ, a k < n → f n ^ c k < f (c k) ^ n := sorry

end exists_sequence_a_l72_72031


namespace determine_multiplier_l72_72292

theorem determine_multiplier (x : ℝ) : 125 * x - 138 = 112 → x = 2 :=
by
  sorry

end determine_multiplier_l72_72292


namespace non_degenerate_ellipse_condition_l72_72631

theorem non_degenerate_ellipse_condition (k : ℝ) :
  (∃ x y : ℝ, 3 * x^2 + 6 * y^2 - 12 * x + 18 * y = k) ↔ k > -51 / 2 :=
sorry

end non_degenerate_ellipse_condition_l72_72631


namespace kevin_watermelons_l72_72543

theorem kevin_watermelons (w1 w2 w_total : ℝ) (h1 : w1 = 9.91) (h2 : w2 = 4.11) (h_total : w_total = 14.02) : 
  w1 + w2 = w_total → 2 = 2 :=
by
  sorry

end kevin_watermelons_l72_72543


namespace product_of_six_numbers_l72_72529

theorem product_of_six_numbers (x y : ℕ) (h1 : x ≠ 0) (h2 : y ≠ 0) 
  (h3 : x^3 * y^2 = 108) : 
  x * y * (x * y) * (x^2 * y) * (x^3 * y^2) * (x^5 * y^3) = 136048896 := 
by
  sorry

end product_of_six_numbers_l72_72529


namespace correct_statements_proof_l72_72187

theorem correct_statements_proof :
  (∀ (a b : ℤ), a - 3 = b - 3 → a = b) ∧
  ¬ (∀ (a b c : ℤ), a = b → a + c = b - c) ∧
  (∀ (a b m : ℤ), m ≠ 0 → (a / m) = (b / m) → a = b) ∧
  ¬ (∀ (a : ℤ), a^2 = 2 * a → a = 2) :=
by
  -- Here we would prove the statements individually:
  -- sorry is a placeholder suggesting that the proofs need to be filled in.
  sorry

end correct_statements_proof_l72_72187


namespace inverse_variation_l72_72098

theorem inverse_variation (x y k : ℝ) (h1 : y = k / x^2) (h2 : k = 8) (h3 : y = 0.5) : x = 4 := by
  sorry

end inverse_variation_l72_72098


namespace point_P_in_third_quadrant_l72_72693

def point_in_third_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0

theorem point_P_in_third_quadrant :
  point_in_third_quadrant (-3) (-2) :=
by
  sorry -- Proof of the statement, as per the steps given.

end point_P_in_third_quadrant_l72_72693


namespace cosine_difference_formula_l72_72421

theorem cosine_difference_formula
  (α : ℝ)
  (h1 : 0 < α)
  (h2 : α < (Real.pi / 2))
  (h3 : Real.tan α = 2) :
  Real.cos (α - (Real.pi / 4)) = (3 * Real.sqrt 10) / 10 := 
by
  sorry

end cosine_difference_formula_l72_72421


namespace number_of_bricks_required_l72_72663

def brick_length : ℝ := 0.20
def brick_width : ℝ := 0.10
def brick_height : ℝ := 0.075

def wall_length : ℝ := 25.0
def wall_width : ℝ := 2.0
def wall_height : ℝ := 0.75

def brick_volume := brick_length * brick_width * brick_height
def wall_volume := wall_length * wall_width * wall_height

theorem number_of_bricks_required :
  wall_volume / brick_volume = 25000 := by
  sorry

end number_of_bricks_required_l72_72663


namespace find_value_l72_72550

theorem find_value
  (x a y b z c : ℝ)
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 3) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 16 :=
by 
  sorry

end find_value_l72_72550


namespace proof_of_problem_l72_72523

noncomputable def problem_statement (a b c x y z : ℝ) : Prop :=
  23 * x + b * y + c * z = 0 ∧
  a * x + 33 * y + c * z = 0 ∧
  a * x + b * y + 52 * z = 0 ∧
  a ≠ 23 ∧
  x ≠ 0 →
  (a + 10) / (a + 10 - 23) + b / (b - 33) + c / (c - 52) = 1

theorem proof_of_problem (a b c x y z : ℝ) (h : problem_statement a b c x y z) : 
  (a + 10) / (a + 10 - 23) + b / (b - 33) + c / (c - 52) = 1 :=
sorry

end proof_of_problem_l72_72523


namespace pages_in_book_l72_72257

-- Define the initial conditions
variable (P : ℝ) -- total number of pages in the book
variable (h_read_20_percent : 0.20 * P = 320 * 0.20 / 0.80) -- Nate has read 20% of the book and the rest 80%

-- The goal is to show that P = 400
theorem pages_in_book (P : ℝ) :
  (0.80 * P = 320) → P = 400 :=
by
  sorry

end pages_in_book_l72_72257


namespace average_monthly_balance_l72_72051

def january_balance : ℕ := 150
def february_balance : ℕ := 300
def march_balance : ℕ := 450
def april_balance : ℕ := 300
def number_of_months : ℕ := 4

theorem average_monthly_balance :
  (january_balance + february_balance + march_balance + april_balance) / number_of_months = 300 := by
  sorry

end average_monthly_balance_l72_72051


namespace morales_sisters_revenue_l72_72400

variable (Gabriela Alba Maricela : Nat)
variable (trees_per_grove : Nat := 110)
variable (oranges_per_tree : (Nat × Nat × Nat) := (600, 400, 500))
variable (oranges_per_cup : Nat := 3)
variable (price_per_cup : Nat := 4)

theorem morales_sisters_revenue :
  let G := trees_per_grove * oranges_per_tree.fst
  let A := trees_per_grove * oranges_per_tree.snd
  let M := trees_per_grove * oranges_per_tree.snd.snd
  let total_oranges := G + A + M
  let total_cups := total_oranges / oranges_per_cup
  let total_revenue := total_cups * price_per_cup
  total_revenue = 220000 :=
by 
  sorry

end morales_sisters_revenue_l72_72400


namespace quadratic_points_order_l72_72175

theorem quadratic_points_order (c y1 y2 : ℝ) 
  (hA : y1 = 0^2 - 6 * 0 + c)
  (hB : y2 = 4^2 - 6 * 4 + c) : 
  y1 > y2 := 
by 
  sorry

end quadratic_points_order_l72_72175


namespace decreasing_function_on_real_l72_72903

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) : f (x + y) = f x + f y
axiom f_negative (x : ℝ) : x > 0 → f x < 0
axiom f_not_identically_zero : ∃ x, f x ≠ 0

theorem decreasing_function_on_real :
  ∀ x1 x2 : ℝ, x1 > x2 → f x1 < f x2 :=
sorry

end decreasing_function_on_real_l72_72903


namespace find_constants_l72_72689

variables {A B C x : ℝ}

theorem find_constants (h : (A = 6) ∧ (B = -5) ∧ (C = 5)) :
  (x^2 + 5*x - 6) / (x^3 - x) = A / x + (B*x + C) / (x^2 - 1) :=
by sorry

end find_constants_l72_72689


namespace sum_digits_n_plus_one_l72_72189

/-- 
Let S(n) be the sum of the digits of a positive integer n.
Given S(n) = 29, prove that the possible values of S(n + 1) are 3, 12, or 30.
-/
theorem sum_digits_n_plus_one (S : ℕ → ℕ) (n : ℕ) (h : S n = 29) :
  S (n + 1) = 3 ∨ S (n + 1) = 12 ∨ S (n + 1) = 30 := 
sorry

end sum_digits_n_plus_one_l72_72189


namespace problem_statement_l72_72083

theorem problem_statement : ¬ (487.5 * 10^(-10) = 0.0000004875) :=
by
  sorry

end problem_statement_l72_72083


namespace perpendicular_line_equation_l72_72461

theorem perpendicular_line_equation (x y : ℝ) :
  (2, -1) ∈ ({ p : ℝ × ℝ | p.1 * 2 + p.2 * 1 - 3 = 0 }) ∧ 
  (∀ p : ℝ × ℝ, (p.1 * 2 + p.2 * (-4) + 5 = 0) → (p.2 * 1 + p.1 * 2 = 0)) :=
sorry

end perpendicular_line_equation_l72_72461


namespace prove_a5_l72_72134

-- Definition of the conditions
def expansion (x : ℤ) (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℤ) :=
  (x - 1) ^ 8 = a_0 + a_1 * (1 + x) + a_2 * (1 + x)^2 + a_3 * (1 + x)^3 + a_4 * (1 + x)^4 + 
               a_5 * (1 + x)^5 + a_6 * (1 + x)^6 + a_7 * (1 + x)^7 + a_8 * (1 + x)^8

-- Given condition
axiom condition (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℤ) : ∀ x : ℤ, expansion x a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8

-- The target problem: proving a_5 = -448
theorem prove_a5 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℤ) : a_5 = -448 :=
by
  sorry

end prove_a5_l72_72134


namespace stickers_at_end_of_week_l72_72486

theorem stickers_at_end_of_week (initial_stickers earned_stickers total_stickers : Nat) :
  initial_stickers = 39 →
  earned_stickers = 22 →
  total_stickers = initial_stickers + earned_stickers →
  total_stickers = 61 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end stickers_at_end_of_week_l72_72486


namespace none_of_these_true_l72_72753

def op_star (a b : ℕ) := b ^ a -- Define the binary operation

theorem none_of_these_true :
  ¬ (∀ a b : ℕ, 0 < a ∧ 0 < b → op_star a b = op_star b a) ∧
  ¬ (∀ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c → op_star a (op_star b c) = op_star (op_star a b) c) ∧
  ¬ (∀ a b n : ℕ, 0 < a ∧ 0 < b ∧ 0 < n → (op_star a b) ^ n = op_star n (op_star a b)) ∧
  ¬ (∀ a b n : ℕ, 0 < a ∧ 0 < b ∧ 0 < n → op_star a (b ^ n) = op_star n (op_star b a)) :=
sorry

end none_of_these_true_l72_72753


namespace ellipse_standard_equation_l72_72601

-- Define the conditions
def equation1 (x y : ℝ) : Prop := x^2 + (y^2 / 2) = 1
def equation2 (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1
def equation3 (x y : ℝ) : Prop := x^2 + (y^2 / 4) = 1
def equation4 (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

-- Define the points
def point1 (x y : ℝ) : Prop := (x = 1 ∧ y = 0)
def point2 (x y : ℝ) : Prop := (x = 0 ∧ y = 2)

-- Define the main theorem
theorem ellipse_standard_equation :
  (equation4 1 0 ∧ equation4 0 2) ↔
  ((equation1 1 0 ∧ equation1 0 2) ∨
   (equation2 1 0 ∧ equation2 0 2) ∨
   (equation3 1 0 ∧ equation3 0 2) ∨
   (equation4 1 0 ∧ equation4 0 2)) :=
by
  sorry

end ellipse_standard_equation_l72_72601


namespace inequality_reciprocal_l72_72317

theorem inequality_reciprocal (a b : ℝ)
  (h : a * b > 0) : a > b ↔ 1 / a < 1 / b := 
sorry

end inequality_reciprocal_l72_72317


namespace average_fish_per_person_l72_72786

theorem average_fish_per_person (Aang Sokka Toph : ℕ) 
  (haang : Aang = 7) (hsokka : Sokka = 5) (htoph : Toph = 12) : 
  (Aang + Sokka + Toph) / 3 = 8 := by
  sorry

end average_fish_per_person_l72_72786


namespace magnitude_b_magnitude_c_area_l72_72472

-- Define the triangle ABC and parameters
variables {A B C : ℝ} {a b c : ℝ}
variables (A_pos : 0 < A) (A_lt_pi_div2 : A < Real.pi / 2)
variables (triangle_condition : a = Real.sqrt 15) (sin_A : Real.sin A = 1 / 4)

-- Problem 1
theorem magnitude_b (cos_B : Real.cos B = Real.sqrt 5 / 3) :
  b = (8 * Real.sqrt 15) / 3 := by
  sorry

-- Problem 2
theorem magnitude_c_area (b_eq_4a : b = 4 * a) :
  c = 15 ∧ (1 / 2 * b * c * Real.sin A = (15 / 2) * Real.sqrt 15) := by
  sorry

end magnitude_b_magnitude_c_area_l72_72472


namespace jimmy_income_l72_72743

theorem jimmy_income (r_income : ℕ) (r_increase : ℕ) (combined_percent : ℚ) (j_income : ℕ) : 
  r_income = 15000 → 
  r_increase = 7000 → 
  combined_percent = 0.55 → 
  (combined_percent * (r_income + r_increase + j_income) = r_income + r_increase) → 
  j_income = 18000 := 
by
  intros h1 h2 h3 h4
  sorry

end jimmy_income_l72_72743


namespace smallest_n_for_divisibility_condition_l72_72864

theorem smallest_n_for_divisibility_condition :
  ∃ n : ℕ, (n > 0) ∧ (∀ (x y z : ℕ), (x > 0) ∧ (y > 0) ∧ (z > 0) →
    (x ∣ y^3) → (y ∣ z^3) → (z ∣ x^3) → (xyz ∣ (x + y + z)^n)) ∧
    n = 13 :=
by
  use 13
  sorry

end smallest_n_for_divisibility_condition_l72_72864


namespace tray_height_l72_72370

noncomputable def height_of_tray : ℝ :=
  let side_length := 120
  let cut_distance := 4 * Real.sqrt 2
  let angle := 45 * (Real.pi / 180)
  -- Define the function that calculates height based on given conditions
  
  sorry

theorem tray_height (side_length : ℝ) (cut_distance : ℝ) (angle : ℝ) : 
  side_length = 120 ∧ cut_distance = 4 * Real.sqrt 2 ∧ angle = 45 * (Real.pi / 180) →
  height_of_tray = 4 * Real.sqrt 2 :=
by
  intros
  unfold height_of_tray
  sorry

end tray_height_l72_72370


namespace is_equilateral_l72_72599

open Complex

noncomputable def z1 : ℂ := sorry
noncomputable def z2 : ℂ := sorry
noncomputable def z3 : ℂ := sorry

-- Assume the conditions of the problem
axiom z1_distinct_z2 : z1 ≠ z2
axiom z2_distinct_z3 : z2 ≠ z3
axiom z3_distinct_z1 : z3 ≠ z1
axiom z1_unit_circle : abs z1 = 1
axiom z2_unit_circle : abs z2 = 1
axiom z3_unit_circle : abs z3 = 1
axiom condition : (1 / (2 + abs (z1 + z2)) + 1 / (2 + abs (z2 + z3)) + 1 / (2 + abs (z3 + z1))) = 1
axiom acute_angled_triangle : sorry

theorem is_equilateral (A B C : ℂ) (hA : A = z1) (hB : B = z2) (hC : C = z3) : 
  (sorry : Prop) := sorry

end is_equilateral_l72_72599


namespace percent_correct_both_l72_72247

-- Definitions based on given conditions in the problem
def P_A : ℝ := 0.63
def P_B : ℝ := 0.50
def P_not_A_and_not_B : ℝ := 0.20

-- Definition of the desired result using the inclusion-exclusion principle based on the given conditions
def P_A_and_B : ℝ := P_A + P_B - (1 - P_not_A_and_not_B)

-- Theorem stating our goal: proving the probability of both answering correctly is 0.33
theorem percent_correct_both : P_A_and_B = 0.33 := by
  sorry

end percent_correct_both_l72_72247


namespace ratio_of_perimeters_l72_72830

theorem ratio_of_perimeters (s₁ s₂ : ℝ) (h : (s₁^2 / s₂^2) = (16 / 49)) : (4 * s₁) / (4 * s₂) = 4 / 7 :=
by
  -- Proof goes here
  sorry

end ratio_of_perimeters_l72_72830


namespace least_clock_equivalent_l72_72456

def clock_equivalent (a b : ℕ) : Prop :=
  ∃ k : ℕ, a + 12 * k = b

theorem least_clock_equivalent (h : ℕ) (hh : h > 3) (hq : clock_equivalent h (h * h)) :
  h = 4 :=
by
  sorry

end least_clock_equivalent_l72_72456


namespace triangle_acute_angle_l72_72253

theorem triangle_acute_angle 
  (a b c : ℝ) 
  (h1 : a^3 = b^3 + c^3)
  (h2 : a > b)
  (h3 : a > c)
  (h4 : b > 0) 
  (h5 : c > 0) 
  (h6 : a > 0) 
  : 
  (a^2 < b^2 + c^2) :=
sorry

end triangle_acute_angle_l72_72253


namespace total_phones_in_Delaware_l72_72586

def population : ℕ := 974000
def phones_per_1000 : ℕ := 673

theorem total_phones_in_Delaware : (population / 1000) * phones_per_1000 = 655502 := by
  sorry

end total_phones_in_Delaware_l72_72586


namespace max_integer_value_l72_72664

theorem max_integer_value (x : ℝ) : 
  ∃ m : ℤ, ∀ (x : ℝ), (3 * x^2 + 9 * x + 17) / (3 * x^2 + 9 * x + 7) ≤ m ∧ m = 41 :=
by sorry

end max_integer_value_l72_72664


namespace melanie_dimes_l72_72388

theorem melanie_dimes (original_dimes dad_dimes mom_dimes total_dimes : ℕ) :
  original_dimes = 7 →
  mom_dimes = 4 →
  total_dimes = 19 →
  (total_dimes = original_dimes + dad_dimes + mom_dimes) →
  dad_dimes = 8 :=
by
  intros h1 h2 h3 h4
  sorry -- The proof is omitted as instructed.

end melanie_dimes_l72_72388


namespace cubes_sum_is_214_5_l72_72959

noncomputable def r_plus_s_plus_t : ℝ := 12
noncomputable def rs_plus_rt_plus_st : ℝ := 47
noncomputable def rst : ℝ := 59.5

theorem cubes_sum_is_214_5 :
    (r_plus_s_plus_t * ((r_plus_s_plus_t)^2 - 3 * rs_plus_rt_plus_st) + 3 * rst) = 214.5 := by
    sorry

end cubes_sum_is_214_5_l72_72959


namespace total_pounds_of_peppers_l72_72511

-- Definitions based on the conditions
def greenPeppers : ℝ := 0.3333333333333333
def redPeppers : ℝ := 0.3333333333333333

-- Goal statement expressing the problem
theorem total_pounds_of_peppers :
  greenPeppers + redPeppers = 0.6666666666666666 := 
by
  sorry

end total_pounds_of_peppers_l72_72511


namespace fraction_solved_l72_72404

theorem fraction_solved (N f : ℝ) (h1 : N * f^2 = 6^3) (h2 : N * f^2 = 7776) : f = 1 / 6 :=
by sorry

end fraction_solved_l72_72404


namespace max_imaginary_part_of_root_l72_72798

theorem max_imaginary_part_of_root (z : ℂ) (h : z^6 - z^4 + z^2 - 1 = 0) (hne : z^2 ≠ 1) : 
  ∃ θ : ℝ, -90 ≤ θ ∧ θ ≤ 90 ∧ Complex.im z = Real.sin θ ∧ θ = 90 := 
sorry

end max_imaginary_part_of_root_l72_72798


namespace axis_of_parabola_l72_72113

-- Define the given equation of the parabola
def parabola (x y : ℝ) : Prop := x^2 = -8 * y

-- Define the standard form of a vertical parabola and the value we need to prove (axis of the parabola)
def standard_form (p y : ℝ) : Prop := y = 2

-- The proof problem: Given the equation of the parabola, prove the equation of its axis.
theorem axis_of_parabola : 
  ∀ x y : ℝ, (parabola x y) → (standard_form y 2) :=
by
  intros x y h
  sorry

end axis_of_parabola_l72_72113


namespace find_y_l72_72997

theorem find_y (a b y : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : y > 0)
  (h4 : (2 * a)^(4 * b) = a^b * y^(3 * b)) : y = 2^(4 / 3) * a :=
by
  sorry

end find_y_l72_72997


namespace coprime_integers_exist_l72_72834

theorem coprime_integers_exist (a b c : ℚ) (t : ℤ) (h1 : a + b + c = t) (h2 : a^2 + b^2 + c^2 = t) (h3 : t ≥ 0) : 
  ∃ (u v : ℤ), Int.gcd u v = 1 ∧ abc = (u^2 : ℚ) / (v^3 : ℚ) :=
by sorry

end coprime_integers_exist_l72_72834


namespace number_of_aquariums_l72_72409

theorem number_of_aquariums (total_animals animals_per_aquarium : ℕ) (h1 : total_animals = 40) (h2 : animals_per_aquarium = 2) :
  total_animals / animals_per_aquarium = 20 := by
  sorry

end number_of_aquariums_l72_72409


namespace check_correct_digit_increase_l72_72391

-- Definition of the numbers involved
def number1 : ℕ := 732
def number2 : ℕ := 648
def number3 : ℕ := 985
def given_sum : ℕ := 2455
def calc_sum : ℕ := number1 + number2 + number3
def difference : ℕ := given_sum - calc_sum

-- Specify the smallest digit that needs to be increased by 1
def smallest_digit_to_increase : ℕ := 8

-- Theorem to check the validity of the problem's claim
theorem check_correct_digit_increase :
  (smallest_digit_to_increase = 8) →
  (calc_sum + 10 = given_sum - 80) :=
by
  intro h
  sorry

end check_correct_digit_increase_l72_72391


namespace combined_total_pets_l72_72162

structure People := 
  (dogs : ℕ)
  (cats : ℕ)

def Teddy : People := {dogs := 7, cats := 8}
def Ben : People := {dogs := 7 + 9, cats := 0}
def Dave : People := {dogs := 7 - 5, cats := 8 + 13}

def total_pets (p : People) : ℕ := p.dogs + p.cats

theorem combined_total_pets : 
  total_pets Teddy + total_pets Ben + total_pets Dave = 54 := by
  sorry

end combined_total_pets_l72_72162


namespace union_M_N_l72_72778

open Set Classical

noncomputable def M : Set ℝ := {x | x^2 = x}
noncomputable def N : Set ℝ := {x | Real.log x ≤ 0}

theorem union_M_N : M ∪ N = Icc 0 1 := by
  sorry

end union_M_N_l72_72778


namespace solve_frac_eqn_l72_72135

theorem solve_frac_eqn (x : ℝ) :
  (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) +
   1 / ((x - 5) * (x - 7)) + 1 / ((x - 7) * (x - 9)) = 1 / 8) ↔ 
  (x = 13 ∨ x = -3) :=
by
  sorry

end solve_frac_eqn_l72_72135


namespace parabola_distance_focus_l72_72111

theorem parabola_distance_focus (x y : ℝ) (h1 : y^2 = 4 * x) (h2 : (x - 1)^2 + y^2 = 16) : x = 3 := by
  sorry

end parabola_distance_focus_l72_72111


namespace water_amount_in_sport_formulation_l72_72411

/-
The standard formulation has the ratios:
F : CS : W = 1 : 12 : 30
Where F is flavoring, CS is corn syrup, and W is water.
-/

def standard_flavoring_ratio : ℚ := 1
def standard_corn_syrup_ratio : ℚ := 12
def standard_water_ratio : ℚ := 30

/-
In the sport formulation:
1) The ratio of flavoring to corn syrup is three times as great as in the standard formulation.
2) The ratio of flavoring to water is half that of the standard formulation.
-/
def sport_flavor_to_corn_ratio : ℚ := 3 * (standard_flavoring_ratio / standard_corn_syrup_ratio)
def sport_flavor_to_water_ratio : ℚ := 1 / 2 * (standard_flavoring_ratio / standard_water_ratio)

/-
The sport formulation contains 6 ounces of corn syrup.
The target is to find the amount of water in the sport formulation.
-/
def corn_syrup_in_sport_formulation : ℚ := 6
def flavoring_in_sport_formulation : ℚ := sport_flavor_to_corn_ratio * corn_syrup_in_sport_formulation

def water_in_sport_formulation : ℚ := 
  (flavoring_in_sport_formulation / sport_flavor_to_water_ratio)

theorem water_amount_in_sport_formulation : water_in_sport_formulation = 90 := by
  sorry

end water_amount_in_sport_formulation_l72_72411


namespace cosine_120_eq_negative_half_l72_72476

theorem cosine_120_eq_negative_half :
  let θ := 120 * Real.pi / 180
  let point := (Real.cos θ, Real.sin θ)
  point.1 = -1 / 2 := by
  sorry

end cosine_120_eq_negative_half_l72_72476


namespace sum_angles_bisected_l72_72946

theorem sum_angles_bisected (θ₁ θ₂ θ₃ θ₄ : ℝ) 
  (h₁ : 0 < θ₁) (h₂ : 0 < θ₂) (h₃ : 0 < θ₃) (h₄ : 0 < θ₄)
  (h_sum : θ₁ + θ₂ + θ₃ + θ₄ = 360) :
  (θ₁ / 2 + θ₃ / 2 = 180 ∨ θ₂ / 2 + θ₄ / 2 = 180) ∧ (θ₂ / 2 + θ₄ / 2 = 180 ∨ θ₁ / 2 + θ₃ / 2 = 180) := 
by 
  sorry

end sum_angles_bisected_l72_72946


namespace oleg_bought_bar_for_60_rubles_l72_72713

theorem oleg_bought_bar_for_60_rubles (n : ℕ) (h₁ : 96 = n * (1 + n / 100)) : n = 60 :=
by {
  sorry
}

end oleg_bought_bar_for_60_rubles_l72_72713


namespace number_of_true_statements_is_two_l72_72324

def line_plane_geometry : Type :=
  -- Types representing lines and planes
  sorry

def l : line_plane_geometry := sorry
def alpha : line_plane_geometry := sorry
def m : line_plane_geometry := sorry
def beta : line_plane_geometry := sorry

def is_perpendicular (x y : line_plane_geometry) : Prop := sorry
def is_parallel (x y : line_plane_geometry) : Prop := sorry
def is_contained_in (x y : line_plane_geometry) : Prop := sorry

axiom l_perpendicular_alpha : is_perpendicular l alpha
axiom m_contained_in_beta : is_contained_in m beta

def statement_1 : Prop := is_parallel alpha beta → is_perpendicular l m
def statement_2 : Prop := is_perpendicular alpha beta → is_parallel l m
def statement_3 : Prop := is_parallel l m → is_perpendicular alpha beta

theorem number_of_true_statements_is_two : 
  (statement_1 ↔ true) ∧ (statement_2 ↔ false) ∧ (statement_3 ↔ true) := 
sorry

end number_of_true_statements_is_two_l72_72324


namespace pascal_fifth_element_15th_row_l72_72708

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l72_72708


namespace lifespan_histogram_l72_72589

theorem lifespan_histogram :
  (class_interval = 20) →
  (height_vertical_axis_60_80 = 0.03) →
  (total_people = 1000) →
  (number_of_people_60_80 = 600) :=
by
  intro class_interval height_vertical_axis_60_80 total_people
  -- Perform necessary calculations (omitting actual proof as per instructions)
  sorry

end lifespan_histogram_l72_72589


namespace haley_initial_shirts_l72_72797

-- Defining the conditions
def returned_shirts := 6
def endup_shirts := 5

-- The theorem statement
theorem haley_initial_shirts : returned_shirts + endup_shirts = 11 := by 
  sorry

end haley_initial_shirts_l72_72797


namespace hyperbola_standard_equations_l72_72699

-- Definitions derived from conditions
def focal_distance (c : ℝ) : Prop := c = 8
def eccentricity (e : ℝ) : Prop := e = 4 / 3
def equilateral_focus (c : ℝ) : Prop := c^2 = 36

-- Theorem stating the standard equations given the conditions
noncomputable def hyperbola_equation1 (y2 : ℝ) (x2 : ℝ) : Prop :=
y2 / 36 - x2 / 28 = 1

noncomputable def hyperbola_equation2 (x2 : ℝ) (y2 : ℝ) : Prop :=
x2 / 18 - y2 / 18 = 1

theorem hyperbola_standard_equations
  (c y2 x2 : ℝ)
  (c_focus : focal_distance c)
  (e_value : eccentricity (4 / 3))
  (equi_focus : equilateral_focus c) :
  hyperbola_equation1 y2 x2 ∧ hyperbola_equation2 x2 y2 :=
by
  sorry

end hyperbola_standard_equations_l72_72699


namespace sum_of_a_and_c_l72_72949

variable {R : Type} [LinearOrderedField R]

theorem sum_of_a_and_c
    (ha hb hc hd : R) 
    (h_intersect : (1, 7) ∈ {p | p.2 = -2 * abs (p.1 - ha) + hb} ∧ (1, 7) ∈ {p | p.2 = 2 * abs (p.1 - hc) + hd}
                 ∧ (9, 1) ∈ {p | p.2 = -2 * abs (p.1 - ha) + hb} ∧ (9, 1) ∈ {p | p.2 = 2 * abs (p.1 - hc) + hd}) :
  ha + hc = 10 :=
by
  sorry

end sum_of_a_and_c_l72_72949


namespace total_eyes_in_family_l72_72572

def mom_eyes := 1
def dad_eyes := 3
def num_kids := 3
def kid_eyes := 4

theorem total_eyes_in_family : mom_eyes + dad_eyes + (num_kids * kid_eyes) = 16 :=
by
  sorry

end total_eyes_in_family_l72_72572


namespace calc_value_l72_72492

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem calc_value :
  ((diamond (diamond 3 4) 2) - (diamond 3 (diamond 4 2))) = -13 / 28 :=
by sorry

end calc_value_l72_72492


namespace eccentricity_of_ellipse_l72_72801

theorem eccentricity_of_ellipse :
  let a : ℝ := 4
  let b : ℝ := 3
  let c : ℝ := Real.sqrt (a^2 - b^2)
  let e : ℝ := c / a
  e = Real.sqrt 7 / 4 :=
by
  sorry

end eccentricity_of_ellipse_l72_72801


namespace percentage_students_50_59_is_10_71_l72_72301

theorem percentage_students_50_59_is_10_71 :
  let n_90_100 := 3
  let n_80_89 := 6
  let n_70_79 := 8
  let n_60_69 := 4
  let n_50_59 := 3
  let n_below_50 := 4
  let total_students := n_90_100 + n_80_89 + n_70_79 + n_60_69 + n_50_59 + n_below_50
  let fraction := (n_50_59 : ℚ) / total_students
  let percentage := (fraction * 100)
  percentage = 10.71 := by sorry

end percentage_students_50_59_is_10_71_l72_72301


namespace at_least_one_negative_l72_72911

theorem at_least_one_negative (a b : ℝ) (h1 : a ≠ b) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : a^2 + 1 / b = b^2 + 1 / a) : a < 0 ∨ b < 0 :=
by
  sorry

end at_least_one_negative_l72_72911


namespace minimum_y_value_l72_72780

noncomputable def minimum_y (x a : ℝ) : ℝ :=
  abs (x - a) + abs (x - 15) + abs (x - a - 15)

theorem minimum_y_value (a x : ℝ) (h1 : 0 < a) (h2 : a < 15) (h3 : a ≤ x) (h4 : x ≤ 15) :
  minimum_y x a = 15 :=
by
  sorry

end minimum_y_value_l72_72780


namespace negatively_added_marks_l72_72636

theorem negatively_added_marks 
  (correct_marks_per_question : ℝ) 
  (total_marks : ℝ) 
  (total_questions : ℕ) 
  (correct_answers : ℕ) 
  (x : ℝ) 
  (h1 : correct_marks_per_question = 4)
  (h2 : total_marks = 420)
  (h3 : total_questions = 150)
  (h4 : correct_answers = 120) 
  (h5 : total_marks = (correct_answers * correct_marks_per_question) - ((total_questions - correct_answers) * x)) :
  x = 2 :=
by 
  sorry

end negatively_added_marks_l72_72636


namespace my_car_mpg_l72_72624

-- Definitions from the conditions.
def total_miles := 100
def total_gallons := 5

-- The statement we need to prove.
theorem my_car_mpg : (total_miles / total_gallons : ℕ) = 20 :=
by
  sorry

end my_car_mpg_l72_72624


namespace angle_in_third_quadrant_l72_72340

theorem angle_in_third_quadrant (θ : ℝ) (hθ : θ = 2023) :
    ∃ k : ℤ, (2023 - k * 360) = 223 ∧ 180 ≤ 223 ∧ 223 < 270 := by
sorry

end angle_in_third_quadrant_l72_72340


namespace roots_of_polynomial_l72_72122

theorem roots_of_polynomial :
  ∀ x : ℝ, x * (x + 2)^2 * (3 - x) * (5 + x) = 0 ↔ (x = 0 ∨ x = -2 ∨ x = 3 ∨ x = -5) :=
by
  sorry

end roots_of_polynomial_l72_72122


namespace ratio_books_to_pens_l72_72144

-- Define the given ratios and known constants.
def ratio_pencils : ℕ := 14
def ratio_pens : ℕ := 4
def ratio_books : ℕ := 3
def actual_pencils : ℕ := 140

-- Assume the actual number of pens can be calculated from ratio.
def actual_pens : ℕ := (actual_pencils / ratio_pencils) * ratio_pens

-- Prove that the ratio of exercise books to pens is as expected.
theorem ratio_books_to_pens (h1 : actual_pencils = 140) 
                            (h2 : actual_pens = 40) : 
  ((actual_pencils / ratio_pencils) * ratio_books) / actual_pens = 3 / 4 :=
by
  -- The following proof steps are omitted as per instruction
  sorry

end ratio_books_to_pens_l72_72144


namespace hours_spent_gaming_l72_72286

def total_hours_in_day : ℕ := 24

def sleeping_fraction : ℚ := 1/3

def studying_fraction : ℚ := 3/4

def gaming_fraction : ℚ := 1/4

theorem hours_spent_gaming :
  let sleeping_hours := total_hours_in_day * sleeping_fraction
  let remaining_hours_after_sleeping := total_hours_in_day - sleeping_hours
  let studying_hours := remaining_hours_after_sleeping * studying_fraction
  let remaining_hours_after_studying := remaining_hours_after_sleeping - studying_hours
  remaining_hours_after_studying * gaming_fraction = 1 :=
by
  sorry

end hours_spent_gaming_l72_72286


namespace common_difference_is_one_l72_72584

variable {a : ℕ → ℝ}
variable {d : ℝ}

-- Conditions given in the problem
axiom h1 : a 1 ^ 2 + a 10 ^ 2 = 101
axiom h2 : a 5 + a 6 = 11
axiom h3 : ∀ n m, n < m → a n < a m
noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n+1) = a n + d

-- Theorem stating the common difference d is 1
theorem common_difference_is_one : is_arithmetic_sequence a d → d = 1 := 
by
  sorry

end common_difference_is_one_l72_72584


namespace verna_sherry_total_weight_l72_72787

theorem verna_sherry_total_weight (haley verna sherry : ℕ)
  (h1 : verna = haley + 17)
  (h2 : verna = sherry / 2)
  (h3 : haley = 103) :
  verna + sherry = 360 :=
by
  sorry

end verna_sherry_total_weight_l72_72787


namespace hyperbola_asymptote_l72_72057

theorem hyperbola_asymptote (a : ℝ) (h : a > 0)
  (has_asymptote : ∀ x : ℝ, abs (9 / a * x) = abs (3 * x))
  : a = 3 :=
sorry

end hyperbola_asymptote_l72_72057


namespace sandra_savings_l72_72037

theorem sandra_savings :
  let num_notepads := 8
  let original_price_per_notepad := 3.75
  let discount_rate := 0.25
  let discount_per_notepad := original_price_per_notepad * discount_rate
  let discounted_price_per_notepad := original_price_per_notepad - discount_per_notepad
  let total_cost_without_discount := num_notepads * original_price_per_notepad
  let total_cost_with_discount := num_notepads * discounted_price_per_notepad
  let total_savings := total_cost_without_discount - total_cost_with_discount
  total_savings = 7.50 :=
sorry

end sandra_savings_l72_72037


namespace sum_difference_of_odd_and_even_integers_l72_72320

noncomputable def sum_of_first_n_odds (n : ℕ) : ℕ :=
  n * n

noncomputable def sum_of_first_n_evens (n : ℕ) : ℕ :=
  n * (n + 1)

theorem sum_difference_of_odd_and_even_integers :
  sum_of_first_n_evens 50 - sum_of_first_n_odds 50 = 50 := 
by
  sorry

end sum_difference_of_odd_and_even_integers_l72_72320


namespace escalator_time_l72_72718

theorem escalator_time (escalator_speed person_speed length : ℕ) 
    (h1 : escalator_speed = 12) 
    (h2 : person_speed = 2) 
    (h3 : length = 196) : 
    (length / (escalator_speed + person_speed) = 14) :=
by
  sorry

end escalator_time_l72_72718


namespace cost_of_each_soda_l72_72677

theorem cost_of_each_soda (total_paid : ℕ) (number_of_sodas : ℕ) (change_received : ℕ) 
  (h1 : total_paid = 20) 
  (h2 : number_of_sodas = 3) 
  (h3 : change_received = 14) : 
  (total_paid - change_received) / number_of_sodas = 2 :=
by
  sorry

end cost_of_each_soda_l72_72677


namespace find_range_of_a_l72_72774

noncomputable def f (a x : ℝ) : ℝ := a / x - Real.exp (-x)

theorem find_range_of_a (p q a : ℝ) (h : 0 < a) (hpq : p < q) :
  (∀ x : ℝ, 0 < x → x ∈ Set.Icc p q → f a x ≤ 0) → 
  (0 < a ∧ a < 1 / Real.exp 1) :=
by
  sorry

end find_range_of_a_l72_72774


namespace root_expression_value_l72_72580

theorem root_expression_value (p q r : ℝ) (hpq : p + q + r = 15) (hpqr : p * q + q * r + r * p = 25) (hpqrs : p * q * r = 10) :
  (p / (2 / p + q * r) + q / (2 / q + r * p) + r / (2 / r + p * q) = 175 / 12) :=
by sorry

end root_expression_value_l72_72580


namespace distinct_real_numbers_condition_l72_72521

noncomputable def f (a b x : ℝ) : ℝ := 1 / (a * x + b)

theorem distinct_real_numbers_condition (a b x1 x2 x3 : ℝ) :
  f a b x1 = x2 → f a b x2 = x3 → f a b x3 = x1 → x1 ≠ x2 → x2 ≠ x3 → x1 ≠ x3 → a = -b^2 :=
by
  sorry

end distinct_real_numbers_condition_l72_72521


namespace index_difference_l72_72506

noncomputable def index_females (n k1 k2 k3 : ℕ) : ℚ :=
  ((n - k1 + k2 : ℚ) / n) * (1 + k3 / 10)

noncomputable def index_males (n k1 l1 l2 : ℕ) : ℚ :=
  ((n - (n - k1) + l1 : ℚ) / n) * (1 + l2 / 10)

theorem index_difference (n k1 k2 k3 l1 l2 : ℕ)
  (h_n : n = 35) (h_k1 : k1 = 15) (h_k2 : k2 = 5) (h_k3 : k3 = 8)
  (h_l1 : l1 = 6) (h_l2 : l2 = 10) : 
  index_females n k1 k2 k3 - index_males n k1 l1 l2 = 3 / 35 :=
by
  sorry

end index_difference_l72_72506


namespace bees_flew_in_l72_72217

theorem bees_flew_in (initial_bees : ℕ) (total_bees : ℕ) (new_bees : ℕ) (h1 : initial_bees = 16) (h2 : total_bees = 23) (h3 : total_bees = initial_bees + new_bees) : new_bees = 7 :=
by
  sorry

end bees_flew_in_l72_72217


namespace smallest_possible_perimeter_of_scalene_triangle_with_prime_sides_l72_72266

/-- Define what it means for a number to be a prime greater than 3 -/
def is_prime_gt_3 (n : ℕ) : Prop :=
  Prime n ∧ 3 < n

/-- Define a scalene triangle with side lengths that are distinct primes greater than 3 -/
def is_scalene_triangle_with_distinct_primes (a b c : ℕ) : Prop :=
  is_prime_gt_3 a ∧ is_prime_gt_3 b ∧ is_prime_gt_3 c ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b > c ∧ b + c > a ∧ a + c > b

/-- The proof problem statement -/
theorem smallest_possible_perimeter_of_scalene_triangle_with_prime_sides :
  ∃ (a b c : ℕ), is_scalene_triangle_with_distinct_primes a b c ∧ Prime (a + b + c) ∧ (a + b + c = 23) :=
sorry

end smallest_possible_perimeter_of_scalene_triangle_with_prime_sides_l72_72266


namespace train_speed_168_l72_72563

noncomputable def speed_of_train (L : ℕ) (V_man : ℕ) (T : ℕ) : ℚ :=
  let V_man_mps := (V_man * 5) / 18
  let relative_speed := L / T
  let V_train_mps := relative_speed - V_man_mps
  V_train_mps * (18 / 5)

theorem train_speed_168 :
  speed_of_train 500 12 10 = 168 :=
by
  sorry

end train_speed_168_l72_72563


namespace quadractic_integer_roots_l72_72872

theorem quadractic_integer_roots (n : ℕ) (h : n > 0) :
  (∃ x y : ℤ, x^2 - 4 * x + n = 0 ∧ y^2 - 4 * y + n = 0) ↔ (n = 3 ∨ n = 4) :=
by
  sorry

end quadractic_integer_roots_l72_72872


namespace garbage_accumulation_correct_l72_72642

-- Given conditions
def garbage_days_per_week : ℕ := 3
def garbage_per_collection : ℕ := 200
def duration_weeks : ℕ := 2

-- Week 1: Full garbage accumulation
def week1_garbage_accumulation : ℕ := garbage_days_per_week * garbage_per_collection

-- Week 2: Half garbage accumulation due to the policy
def week2_garbage_accumulation : ℕ := week1_garbage_accumulation / 2

-- Total garbage accumulation over the 2 weeks
def total_garbage_accumulation (week1 week2 : ℕ) : ℕ := week1 + week2

-- Proof statement
theorem garbage_accumulation_correct :
  total_garbage_accumulation week1_garbage_accumulation week2_garbage_accumulation = 900 := by
  sorry

end garbage_accumulation_correct_l72_72642


namespace hash_op_calculation_l72_72620

-- Define the new operation
def hash_op (a b : ℚ) : ℚ :=
  a^2 + a * b - 5

-- Prove that (-3) # 6 = -14
theorem hash_op_calculation : hash_op (-3) 6 = -14 := by
  sorry

end hash_op_calculation_l72_72620


namespace number_of_distinguishable_large_triangles_l72_72110

theorem number_of_distinguishable_large_triangles (colors : Fin 8) :
  ∃(large_triangles : Fin 960), true :=
by
  sorry

end number_of_distinguishable_large_triangles_l72_72110


namespace min_value_4x_plus_inv_l72_72717

noncomputable def min_value_function (x : ℝ) := 4 * x + 1 / (4 * x - 5)

theorem min_value_4x_plus_inv (x : ℝ) (h : x > 5 / 4) : min_value_function x = 7 :=
sorry

end min_value_4x_plus_inv_l72_72717


namespace dividend_rate_l72_72639

theorem dividend_rate (face_value market_value expected_interest interest_rate : ℝ)
  (h1 : face_value = 52)
  (h2 : expected_interest = 0.12)
  (h3 : market_value = 39)
  : ((expected_interest * market_value) / face_value) * 100 = 9 := by
  sorry

end dividend_rate_l72_72639


namespace arithmetic_seq_a12_l72_72442

-- Define an arithmetic sequence
def arithmetic_seq (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

-- Prove that a_12 = 12 given the conditions
theorem arithmetic_seq_a12 :
  ∃ a₁, (arithmetic_seq a₁ 2 2 = -8) → (arithmetic_seq a₁ 2 12 = 12) :=
by
  sorry

end arithmetic_seq_a12_l72_72442


namespace expression_never_prime_l72_72598

noncomputable def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem expression_never_prime (n : ℕ) (h : is_prime n) : ¬is_prime (n^2 + 75) :=
sorry

end expression_never_prime_l72_72598


namespace free_time_left_after_cleaning_l72_72124

-- Define the time it takes for each task
def vacuuming_time : ℤ := 45
def dusting_time : ℤ := 60
def mopping_time : ℤ := 30
def brushing_time_per_cat : ℤ := 5
def number_of_cats : ℤ := 3
def total_free_time_in_minutes : ℤ := 3 * 60 -- 3 hours converted to minutes

-- Define the total cleaning time
def total_cleaning_time : ℤ := vacuuming_time + dusting_time + mopping_time + (brushing_time_per_cat * number_of_cats)

-- Prove that the free time left after cleaning is 30 minutes
theorem free_time_left_after_cleaning : (total_free_time_in_minutes - total_cleaning_time) = 30 :=
by
  sorry

end free_time_left_after_cleaning_l72_72124


namespace solution_set_of_inequality_l72_72856

theorem solution_set_of_inequality (x : ℝ) : {x | x * (x - 1) > 0} = { x | x < 0 } ∪ { x | x > 1 } :=
sorry

end solution_set_of_inequality_l72_72856


namespace range_of_m_l72_72450

open Set Real

noncomputable def f (x m : ℝ) : ℝ := abs (x^2 - 4 * x + 9 - 2 * m) + 2 * m

theorem range_of_m
  (h1 : ∀ x ∈ Icc (0 : ℝ) 4, f x m ≤ 9) : m ≤ 7 / 2 :=
by
  sorry

end range_of_m_l72_72450


namespace third_side_triangle_l72_72299

theorem third_side_triangle (a : ℝ) :
  (5 < a ∧ a < 13) → (a = 8) :=
sorry

end third_side_triangle_l72_72299


namespace cars_in_garage_l72_72196

/-
Conditions:
1. Total wheels in the garage: 22
2. Riding lawnmower wheels: 4
3. Timmy's bicycle wheels: 2
4. Each of Timmy's parents' bicycles: 2 wheels, and there are 2 bicycles.
5. Joey's tricycle wheels: 3
6. Timmy's dad's unicycle wheels: 1

Question: How many cars are inside the garage?

Correct Answer: The number of cars is 2.
-/
theorem cars_in_garage (total_wheels : ℕ) (lawnmower_wheels : ℕ)
  (timmy_bicycle_wheels : ℕ) (parents_bicycles_wheels : ℕ)
  (joey_tricycle_wheels : ℕ) (dad_unicycle_wheels : ℕ) 
  (cars_wheels : ℕ) (cars : ℕ) :
  total_wheels = 22 →
  lawnmower_wheels = 4 →
  timmy_bicycle_wheels = 2 →
  parents_bicycles_wheels = 2 * 2 →
  joey_tricycle_wheels = 3 →
  dad_unicycle_wheels = 1 →
  cars_wheels = total_wheels - (lawnmower_wheels + timmy_bicycle_wheels + parents_bicycles_wheels + joey_tricycle_wheels + dad_unicycle_wheels) →
  cars = cars_wheels / 4 →
  cars = 2 := by
  sorry

end cars_in_garage_l72_72196


namespace second_game_score_count_l72_72510

-- Define the conditions and problem
def total_points (A1 A2 A3 B1 B2 B3 : ℕ) : Prop :=
  A1 + A2 + A3 + B1 + B2 + B3 = 31

def valid_game_1 (A1 B1 : ℕ) : Prop :=
  A1 ≥ 11 ∧ A1 - B1 ≥ 2

def valid_game_2 (A2 B2 : ℕ) : Prop :=
  B2 ≥ 11 ∧ B2 - A2 ≥ 2

def valid_game_3 (A3 B3 : ℕ) : Prop :=
  A3 ≥ 11 ∧ A3 - B3 ≥ 2

def game_sequence (A1 A2 A3 B1 B2 B3 : ℕ) : Prop :=
  valid_game_1 A1 B1 ∧ valid_game_2 A2 B2 ∧ valid_game_3 A3 B3

noncomputable def second_game_score_possibilities : ℕ := 
  8 -- This is derived from calculating the valid scores where B wins the second game.

theorem second_game_score_count (A1 A2 A3 B1 B2 B3 : ℕ) (h_total : total_points A1 A2 A3 B1 B2 B3) (h_sequence : game_sequence A1 A2 A3 B1 B2 B3) :
  second_game_score_possibilities = 8 := sorry

end second_game_score_count_l72_72510


namespace extreme_values_sin_2x0_l72_72368

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sqrt 3 * Real.cos (Real.pi / 2 + x)^2 - 
  2 * Real.sin (Real.pi + x) * Real.cos x - Real.sqrt 3

-- Part (1)
theorem extreme_values : 
  (∀ x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), 1 ≤ f x ∧ f x ≤ 2) :=
sorry

-- Part (2)
theorem sin_2x0 (x0 : ℝ) (h : x0 ∈ Set.Icc (3 * Real.pi / 4) Real.pi) (hx : f (x0 - Real.pi / 6) = 10 / 13) : 
  Real.sin (2 * x0) = - (5 + 12 * Real.sqrt 3) / 26 :=
sorry

end extreme_values_sin_2x0_l72_72368


namespace difference_of_sums_1500_l72_72225

def sum_of_first_n_odd_numbers (n : ℕ) : ℕ :=
  n * n

def sum_of_first_n_even_numbers (n : ℕ) : ℕ :=
  n * (n + 1)

theorem difference_of_sums_1500 :
  sum_of_first_n_even_numbers 1500 - sum_of_first_n_odd_numbers 1500 = 1500 :=
by
  sorry

end difference_of_sums_1500_l72_72225


namespace find_weight_of_b_l72_72567

variable (a b c d : ℝ)

def average_weight_of_four : Prop := (a + b + c + d) / 4 = 45

def average_weight_of_a_and_b : Prop := (a + b) / 2 = 42

def average_weight_of_b_and_c : Prop := (b + c) / 2 = 43

def ratio_of_d_to_a : Prop := d / a = 3 / 4

theorem find_weight_of_b (h1 : average_weight_of_four a b c d)
                        (h2 : average_weight_of_a_and_b a b)
                        (h3 : average_weight_of_b_and_c b c)
                        (h4 : ratio_of_d_to_a a d) :
    b = 29.43 :=
  by sorry

end find_weight_of_b_l72_72567


namespace product_of_roots_l72_72426

theorem product_of_roots (a b c : ℂ) (h_roots : 3 * (Polynomial.C a) * (Polynomial.C b) * (Polynomial.C c) = -7) :
  a * b * c = -7 / 3 :=
by sorry

end product_of_roots_l72_72426


namespace problem_a9_b9_l72_72366

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

-- Define the conditions
axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

-- Prove the goal
theorem problem_a9_b9 : a^9 + b^9 = 76 :=
by
  -- the proof will come here
  sorry

end problem_a9_b9_l72_72366


namespace new_trailer_homes_count_l72_72690

theorem new_trailer_homes_count :
  let old_trailers : ℕ := 30
  let old_avg_age : ℕ := 15
  let years_since : ℕ := 3
  let new_avg_age : ℕ := 10
  let total_age := (old_trailers * (old_avg_age + years_since)) + (3 * new_trailers)
  let total_trailers := old_trailers + new_trailers
  let total_avg_age := total_age / total_trailers
  total_avg_age = new_avg_age → new_trailers = 34 :=
by
  sorry

end new_trailer_homes_count_l72_72690


namespace intersection_S_T_eq_interval_l72_72099

-- Define the sets S and T
def S : Set ℝ := {x | x ≥ 2}
def T : Set ℝ := {x | x ≤ 5}

-- Prove the intersection of S and T is [2, 5]
theorem intersection_S_T_eq_interval : S ∩ T = {x | 2 ≤ x ∧ x ≤ 5} :=
by
  sorry

end intersection_S_T_eq_interval_l72_72099


namespace Alyssa_initial_puppies_l72_72661

theorem Alyssa_initial_puppies : 
  ∀ (a b c : ℕ), b = 7 → c = 5 → a = b + c → a = 12 := 
by
  intros a b c hb hc hab
  rw [hb, hc] at hab
  exact hab

end Alyssa_initial_puppies_l72_72661


namespace area_ratio_l72_72156

noncomputable def pentagon_area (R s : ℝ) := (5 / 2) * R * s * Real.sin (Real.pi * 2 / 5)
noncomputable def triangle_area (s : ℝ) := (s^2) / 4

theorem area_ratio (R s : ℝ) (h : R = s / (2 * Real.sin (Real.pi / 5))) :
  (pentagon_area R s) / (triangle_area s) = 5 * (Real.sin ((2 * Real.pi) / 5) / Real.sin (Real.pi / 5)) :=
by
  sorry

end area_ratio_l72_72156


namespace solve_fraction_eq_l72_72221

theorem solve_fraction_eq (x : ℝ) (h : x ≠ -2) : (x = -1) ↔ ((x^2 + 2 * x + 3) / (x + 2) = x + 3) := 
by 
  sorry

end solve_fraction_eq_l72_72221


namespace elizabeth_net_profit_l72_72151

-- Define the conditions
def cost_per_bag : ℝ := 3.00
def bags_produced : ℕ := 20
def selling_price_per_bag : ℝ := 6.00
def bags_sold_full_price : ℕ := 15
def discount_percentage : ℝ := 0.25

-- Define the net profit computation
def net_profit : ℝ :=
  let revenue_full_price := bags_sold_full_price * selling_price_per_bag
  let remaining_bags := bags_produced - bags_sold_full_price
  let discounted_price_per_bag := selling_price_per_bag * (1 - discount_percentage)
  let revenue_discounted := remaining_bags * discounted_price_per_bag
  let total_revenue := revenue_full_price + revenue_discounted
  let total_cost := bags_produced * cost_per_bag
  total_revenue - total_cost

theorem elizabeth_net_profit : net_profit = 52.50 := by
  sorry

end elizabeth_net_profit_l72_72151


namespace diana_can_paint_statues_l72_72656

theorem diana_can_paint_statues (total_paint : ℚ) (paint_per_statue : ℚ) 
  (h1 : total_paint = 3 / 6) (h2 : paint_per_statue = 1 / 6) : 
  total_paint / paint_per_statue = 3 :=
by
  sorry

end diana_can_paint_statues_l72_72656


namespace michael_left_money_l72_72376

def michael_initial_money : Nat := 100
def michael_spent_on_snacks : Nat := 25
def michael_spent_on_rides : Nat := 3 * michael_spent_on_snacks
def michael_spent_on_games : Nat := 15
def total_expenditure : Nat := michael_spent_on_snacks + michael_spent_on_rides + michael_spent_on_games
def michael_money_left : Nat := michael_initial_money - total_expenditure

theorem michael_left_money : michael_money_left = 15 := by
  sorry

end michael_left_money_l72_72376


namespace neg_disj_imp_neg_conj_l72_72623

theorem neg_disj_imp_neg_conj (p q : Prop) (h : ¬(p ∨ q)) : ¬p ∧ ¬q :=
sorry

end neg_disj_imp_neg_conj_l72_72623


namespace ages_of_boys_l72_72900

theorem ages_of_boys (a b c : ℕ) (h1 : a + b + c = 29) (h2 : a = b) (h3 : c = 11) : a = 9 :=
by
  sorry

end ages_of_boys_l72_72900


namespace jumps_per_second_l72_72377

-- Define the conditions and known values
def record_jumps : ℕ := 54000
def hours : ℕ := 5
def seconds_per_hour : ℕ := 3600

-- Define the target question as a theorem to prove
theorem jumps_per_second :
  (record_jumps / (hours * seconds_per_hour)) = 3 := by
  sorry

end jumps_per_second_l72_72377


namespace polygon_perimeter_eq_21_l72_72579

-- Definitions and conditions from the given problem
def rectangle_side_a := 6
def rectangle_side_b := 4
def triangle_hypotenuse := 5

-- The combined polygon perimeter proof statement
theorem polygon_perimeter_eq_21 :
  let rectangle_perimeter := 2 * (rectangle_side_a + rectangle_side_b)
  let adjusted_perimeter := rectangle_perimeter - rectangle_side_b + triangle_hypotenuse
  adjusted_perimeter = 21 :=
by 
  -- Skip the proof part by adding sorry
  sorry

end polygon_perimeter_eq_21_l72_72579


namespace algebra_expression_value_l72_72886

theorem algebra_expression_value (x y : ℤ) (h : x - 2 * y + 2 = 5) : 2 * x - 4 * y - 1 = 5 :=
by
  sorry

end algebra_expression_value_l72_72886


namespace domain_of_c_is_all_reals_l72_72626

theorem domain_of_c_is_all_reals (k : ℝ) : 
  (∀ x : ℝ, -3 * x^2 + 5 * x + k ≠ 0) ↔ k < -(25 / 12) :=
by
  sorry

end domain_of_c_is_all_reals_l72_72626


namespace weight_of_one_bowling_ball_l72_72365

def weight_of_one_canoe : ℕ := 35

def ten_bowling_balls_equal_four_canoes (W: ℕ) : Prop :=
  ∀ w, (10 * w = 4 * W)

theorem weight_of_one_bowling_ball (W: ℕ) (h : W = weight_of_one_canoe) : 
  (10 * 14 = 4 * W) → 14 = 140 / 10 :=
by
  intros H
  sorry

end weight_of_one_bowling_ball_l72_72365


namespace incorrect_number_read_l72_72145

theorem incorrect_number_read (incorrect_avg correct_avg : ℕ) (n correct_number incorrect_sum correct_sum : ℕ)
  (h1 : incorrect_avg = 17)
  (h2 : correct_avg = 20)
  (h3 : n = 10)
  (h4 : correct_number = 56)
  (h5 : incorrect_sum = n * incorrect_avg)
  (h6 : correct_sum = n * correct_avg)
  (h7 : correct_sum - incorrect_sum = correct_number - X) :
  X = 26 :=
by
  sorry

end incorrect_number_read_l72_72145


namespace clock_equiv_4_cubic_l72_72611

theorem clock_equiv_4_cubic :
  ∃ x : ℕ, x > 3 ∧ x % 12 = (x^3) % 12 ∧ (∀ y : ℕ, y > 3 ∧ y % 12 = (y^3) % 12 → x ≤ y) :=
by
  use 4
  sorry

end clock_equiv_4_cubic_l72_72611


namespace restroom_students_l72_72025

theorem restroom_students (R : ℕ) (h1 : 4 * 6 = 24) (h2 : (2/3 : ℚ) * 24 = 16)
  (h3 : 23 = 16 + (3 * R - 1) + R) : R = 2 :=
by
  sorry

end restroom_students_l72_72025


namespace unoccupied_seats_in_business_class_l72_72861

def airplane_seating (fc bc ec : ℕ) (pfc pbc pec : ℕ) : Nat :=
  let num_ec := ec / 2
  let num_bc := num_ec - pfc
  bc - num_bc

theorem unoccupied_seats_in_business_class :
  airplane_seating 10 30 50 3 (50/2) (50/2) = 8 := by
    sorry

end unoccupied_seats_in_business_class_l72_72861


namespace doubled_cylinder_volume_l72_72653

theorem doubled_cylinder_volume (r h : ℝ) (V : ℝ) (original_volume : V = π * r^2 * h) (V' : ℝ) : (2 * 2 * π * r^2 * h = 40) := 
by 
  have original_volume := 5
  sorry

end doubled_cylinder_volume_l72_72653


namespace minimum_a_l72_72937

noncomputable def f (x a : ℝ) := Real.exp x * (x^3 - 3 * x + 3) - a * Real.exp x - x

theorem minimum_a (a : ℝ) : (∃ x, x ≥ -2 ∧ f x a ≤ 0) ↔ a ≥ 1 - 1 / Real.exp 1 :=
by
  sorry

end minimum_a_l72_72937


namespace polynomial_nonnegative_iff_eq_l72_72648

variable {R : Type} [LinearOrderedField R]

def polynomial_p (x a b c : R) : R :=
  (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)

theorem polynomial_nonnegative_iff_eq (a b c : R) :
  (∀ x : R, polynomial_p x a b c ≥ 0) ↔ a = b ∧ b = c :=
by
  sorry

end polynomial_nonnegative_iff_eq_l72_72648


namespace count_integer_values_l72_72211

theorem count_integer_values (π : Real) (hπ : Real.pi = π):
  ∃ n : ℕ, n = 27 ∧ ∀ x : ℤ, |(x:Real)| < 4 * π + 1 ↔ -13 ≤ x ∧ x ≤ 13 :=
by sorry

end count_integer_values_l72_72211


namespace find_number_l72_72474

theorem find_number (x : ℝ) : (0.75 * x = 0.45 * 1500 + 495) -> x = 1560 :=
by
  sorry

end find_number_l72_72474


namespace square_field_diagonal_l72_72222

theorem square_field_diagonal (a : ℝ) (d : ℝ) (h : a^2 = 800) : d = 40 :=
by
  sorry

end square_field_diagonal_l72_72222


namespace exists_four_functions_l72_72707

theorem exists_four_functions 
  (f : ℝ → ℝ)
  (h_periodic : ∀ x, f (x + 2 * Real.pi) = f x) :
  ∃ (f1 f2 f3 f4 : ℝ → ℝ), 
    (∀ x, f1 (-x) = f1 x ∧ f1 (x + Real.pi) = f1 x) ∧
    (∀ x, f2 (-x) = f2 x ∧ f2 (x + Real.pi) = f2 x) ∧
    (∀ x, f3 (-x) = f3 x ∧ f3 (x + Real.pi) = f3 x) ∧
    (∀ x, f4 (-x) = f4 x ∧ f4 (x + Real.pi) = f4 x) ∧
    (∀ x, f x = f1 x + f2 x * Real.cos x + f3 x * Real.sin x + f4 x * Real.sin (2 * x)) :=
sorry

end exists_four_functions_l72_72707


namespace part1_part2_l72_72989

-- Part 1
theorem part1 (x y : ℝ) : (2 * x - 3 * y) ^ 2 - (y + 3 * x) * (3 * x - y) = -5 * x ^ 2 - 12 * x * y + 10 * y ^ 2 := 
sorry

-- Part 2
theorem part2 : (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) - 2 ^ 16 = -1 := 
sorry

end part1_part2_l72_72989


namespace ratio_of_rectangle_to_triangle_l72_72765

variable (L W : ℝ)

theorem ratio_of_rectangle_to_triangle (hL : L > 0) (hW : W > 0) : 
    L * W / (1/2 * L * W) = 2 := 
by
  sorry

end ratio_of_rectangle_to_triangle_l72_72765


namespace no_real_roots_l72_72475

theorem no_real_roots (k : ℝ) (h : k ≠ 0) : ¬∃ x : ℝ, x^2 + k * x + 3 * k^2 = 0 :=
by
  sorry

end no_real_roots_l72_72475


namespace angle_in_third_quadrant_l72_72558

-- Define the concept of an angle being in a specific quadrant
def is_in_third_quadrant (θ : ℝ) : Prop :=
  180 < θ ∧ θ < 270

-- Prove that -1200° is in the third quadrant
theorem angle_in_third_quadrant :
  is_in_third_quadrant (240) → is_in_third_quadrant (-1200 % 360 + 360 * (if -1200 % 360 ≤ 0 then 1 else 0)) :=
by
  sorry

end angle_in_third_quadrant_l72_72558


namespace work_days_together_l72_72193

-- Conditions
variable {W : ℝ} (h_a_alone : ∀ (W : ℝ), W / a_work_time = W / 16)
variable {a_work_time : ℝ} (h_work_time_a : a_work_time = 16)

-- Question translated to proof problem
theorem work_days_together (D : ℝ) :
  (10 * (W / D) + 12 * (W / 16) = W) → D = 40 :=
by
  intros h
  have eq1 : 10 * (W / D) + 12 * (W / 16) = W := h
  sorry

end work_days_together_l72_72193


namespace propositions_correct_l72_72927

def vertical_angles (α β : ℝ) : Prop := ∃ γ, α = γ ∧ β = γ

def problem_statement : Prop :=
  (∀ α β, vertical_angles α β → α = β) ∧
  ¬(∀ α β, α = β → vertical_angles α β) ∧
  ¬(∀ α β, ¬vertical_angles α β → ¬(α = β)) ∧
  (∀ α β, ¬(α = β) → ¬vertical_angles α β)

theorem propositions_correct :
  problem_statement :=
by
  sorry

end propositions_correct_l72_72927


namespace fraction_inequality_l72_72673

theorem fraction_inequality (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) :
  b / (a - c) < a / (b - d) :=
sorry

end fraction_inequality_l72_72673


namespace smallest_n_for_4n_square_and_5n_cube_l72_72334

theorem smallest_n_for_4n_square_and_5n_cube :
  ∃ (n : ℕ), (n > 0 ∧ (∃ k : ℕ, 4 * n = k^2) ∧ (∃ m : ℕ, 5 * n = m^3)) ∧ n = 400 :=
by
  sorry

end smallest_n_for_4n_square_and_5n_cube_l72_72334


namespace find_z_when_x_is_1_l72_72745

-- We start by defining the conditions
variable (x y z : ℝ)
variable (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
variable (h_inv : ∃ k₁ : ℝ, ∀ x, x^2 * y = k₁)
variable (h_dir : ∃ k₂ : ℝ, ∀ y, y / z = k₂)
variable (h_y : y = 8) (h_z : z = 32) (h_x4 : x = 4)

-- Now we need to define the problem statement: 
-- proving that z = 512 when x = 1
theorem find_z_when_x_is_1 (h_x1 : x = 1) : z = 512 :=
  sorry

end find_z_when_x_is_1_l72_72745


namespace total_worth_of_stock_l72_72585

theorem total_worth_of_stock (X : ℝ) :
  (0.30 * 0.10 * X + 0.40 * -0.05 * X + 0.30 * -0.10 * X = -500) → X = 25000 :=
by
  intro h
  -- Proof to be completed
  sorry

end total_worth_of_stock_l72_72585


namespace central_angle_proof_l72_72417

noncomputable def central_angle (l r : ℝ) : ℝ :=
  l / r

theorem central_angle_proof :
  central_angle 300 100 = 3 :=
by
  -- The statement of the theorem aligns with the given problem conditions and the expected answer.
  sorry

end central_angle_proof_l72_72417


namespace isosceles_triangle_base_angle_l72_72812

/-- In an isosceles triangle, if one angle is 110 degrees, then each base angle measures 35 degrees. -/
theorem isosceles_triangle_base_angle (α β γ : ℝ) (h1 : α + β + γ = 180)
  (h2 : α = β ∨ α = γ ∨ β = γ) (h3 : α = 110 ∨ β = 110 ∨ γ = 110) :
  β = 35 ∨ γ = 35 :=
sorry

end isosceles_triangle_base_angle_l72_72812


namespace trig_relation_l72_72832

theorem trig_relation : (Real.pi/4 < 1) ∧ (1 < Real.pi/2) → Real.tan 1 > Real.sin 1 ∧ Real.sin 1 > Real.cos 1 := 
by 
  intro h
  sorry

end trig_relation_l72_72832


namespace experts_expected_points_probability_fifth_envelope_l72_72720

theorem experts_expected_points (n : ℕ) (h1 : n = 100) (h2 : n = 13) :
  ∃ e : ℚ, e = 465 :=
sorry

theorem probability_fifth_envelope (m : ℕ) (h1 : m = 13) :
  ∃ p : ℚ, p = 0.715 :=
sorry

end experts_expected_points_probability_fifth_envelope_l72_72720


namespace archie_touchdown_passes_l72_72921

-- Definitions based on the conditions
def richard_avg_first_14_games : ℕ := 6
def richard_avg_last_2_games : ℕ := 3
def richard_games_first : ℕ := 14
def richard_games_last : ℕ := 2

-- Total touchdowns Richard made in the first 14 games
def touchdowns_first_14 := richard_games_first * richard_avg_first_14_games

-- Total touchdowns Richard needs in the final 2 games
def touchdowns_last_2 := richard_games_last * richard_avg_last_2_games

-- Total touchdowns Richard made in the season
def richard_touchdowns_season := touchdowns_first_14 + touchdowns_last_2

-- Archie's record is one less than Richard's total touchdowns for the season
def archie_record := richard_touchdowns_season - 1

-- Proposition to prove Archie's touchdown passes in a season
theorem archie_touchdown_passes : archie_record = 89 := by
  sorry

end archie_touchdown_passes_l72_72921


namespace combined_frosting_rate_l72_72034

theorem combined_frosting_rate (time_Cagney time_Lacey total_time : ℕ) (Cagney_rate Lacey_rate : ℚ) :
  (time_Cagney = 20) →
  (time_Lacey = 30) →
  (total_time = 5 * 60) →
  (Cagney_rate = 1 / time_Cagney) →
  (Lacey_rate = 1 / time_Lacey) →
  ((Cagney_rate + Lacey_rate) * total_time) = 25 :=
by
  intros
  -- conditions are given and used in the statement.
  -- proof follows from these conditions. 
  sorry

end combined_frosting_rate_l72_72034


namespace pages_already_read_l72_72011

theorem pages_already_read (total_pages : ℕ) (pages_left : ℕ) (h_total : total_pages = 563) (h_left : pages_left = 416) :
  total_pages - pages_left = 147 :=
by
  sorry

end pages_already_read_l72_72011


namespace true_false_questions_count_l72_72613

/-- 
 In an answer key for a quiz, there are some true-false questions followed by 3 multiple-choice questions with 4 answer choices each. 
 The correct answers to all true-false questions cannot be the same. 
 There are 384 ways to write the answer key. How many true-false questions are there?
-/
theorem true_false_questions_count : 
  ∃ n : ℕ, 2^n - 2 = 6 ∧ (2^n - 2) * 4^3 = 384 := 
sorry

end true_false_questions_count_l72_72613


namespace cube_largest_ne_sum_others_l72_72346

theorem cube_largest_ne_sum_others (n : ℕ) : (n + 1)^3 ≠ n^3 + (n - 1)^3 :=
by
  sorry

end cube_largest_ne_sum_others_l72_72346


namespace solve_for_m_l72_72999

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 4

theorem solve_for_m (m : ℤ) (h : ∃ x : ℝ, 2^x + x = 4 ∧ m ≤ x ∧ x ≤ m + 1) : m = 1 :=
by
  sorry

end solve_for_m_l72_72999


namespace solution_set_of_inequality_l72_72709

theorem solution_set_of_inequality (x : ℝ) : (|x + 1| - |x - 3| ≥ 0) ↔ (1 ≤ x) := 
sorry

end solution_set_of_inequality_l72_72709


namespace orthographic_projection_area_l72_72851

theorem orthographic_projection_area (s : ℝ) (h : s = 1) : 
  let S := (Real.sqrt 3) / 4 
  let factor := (Real.sqrt 2) / 2
  let S' := (factor ^ 2) * S
  S' = (Real.sqrt 6) / 16 :=
by
  let S := (Real.sqrt 3) / 4
  let factor := (Real.sqrt 2) / 2
  let S' := (factor ^ 2) * S
  sorry

end orthographic_projection_area_l72_72851


namespace sequence_mono_iff_b_gt_neg3_l72_72489

theorem sequence_mono_iff_b_gt_neg3 (b : ℝ) : 
  (∀ n : ℕ, 1 ≤ n → (n + 1) ^ 2 + b * (n + 1) > n ^ 2 + b * n) → b > -3 := 
by
  sorry

end sequence_mono_iff_b_gt_neg3_l72_72489


namespace range_of_a_l72_72531

def p (x : ℝ) : Prop := (1/2 ≤ x ∧ x ≤ 1)

def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, p x → q x a) ∧ (∃ x : ℝ, q x a ∧ ¬ p x) → 
  (0 ≤ a ∧ a ≤ 1/2) :=
by
  sorry

end range_of_a_l72_72531


namespace cos_double_angle_l72_72006

theorem cos_double_angle (a : ℝ) (h : Real.sin a = 3/5) : Real.cos (2 * a) = 7/25 :=
by
  sorry

end cos_double_angle_l72_72006


namespace no_valid_prime_pairs_l72_72479

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem no_valid_prime_pairs :
  ∀ x y : ℕ, is_prime x → is_prime y → y < x → x ≤ 200 → (x % y = 0) → ((x +1) % (y +1) = 0) → false :=
by
  sorry

end no_valid_prime_pairs_l72_72479


namespace problem_statement_l72_72520

open Real

theorem problem_statement (α : ℝ) 
  (h1 : cos (α + π / 4) = (7 * sqrt 2) / 10)
  (h2 : cos (2 * α) = 7 / 25) :
  sin α + cos α = 1 / 5 :=
sorry

end problem_statement_l72_72520


namespace percent_decrease_l72_72967

theorem percent_decrease(call_cost_1980 call_cost_2010 : ℝ) (h₁ : call_cost_1980 = 50) (h₂ : call_cost_2010 = 5) :
  ((call_cost_1980 - call_cost_2010) / call_cost_1980 * 100) = 90 :=
by
  sorry

end percent_decrease_l72_72967


namespace problem_statement_l72_72323

variable {x y : ℝ}

def star (a b : ℝ) : ℝ := (a + b)^2

theorem problem_statement (x y : ℝ) : star ((x + y)^2) ((y + x)^2) = 4 * (x + y)^4 := by
  sorry

end problem_statement_l72_72323


namespace base_256_6_digits_l72_72152

theorem base_256_6_digits (b : ℕ) (h1 : b ^ 5 ≤ 256) (h2 : 256 < b ^ 6) : b = 3 := 
sorry

end base_256_6_digits_l72_72152


namespace expr_B_not_simplified_using_difference_of_squares_l72_72460

def expr_A (x y : ℝ) := (-x - y) * (-x + y)
def expr_B (x y : ℝ) := (-x + y) * (x - y)
def expr_C (x y : ℝ) := (y + x) * (x - y)
def expr_D (x y : ℝ) := (y - x) * (x + y)

theorem expr_B_not_simplified_using_difference_of_squares (x y : ℝ) :
  ∃ x y, ¬ ∃ a b, expr_B x y = a^2 - b^2 :=
sorry

end expr_B_not_simplified_using_difference_of_squares_l72_72460


namespace carla_gas_cost_l72_72385

theorem carla_gas_cost:
  let distance_grocery := 8
  let distance_school := 6
  let distance_bank := 12
  let distance_practice := 9
  let distance_dinner := 15
  let distance_home := 2 * distance_practice
  let total_distance := distance_grocery + distance_school + distance_bank + distance_practice + distance_dinner + distance_home
  let miles_per_gallon := 25
  let price_per_gallon_first := 2.35
  let price_per_gallon_second := 2.65
  let total_gallons := total_distance / miles_per_gallon
  let gallons_per_fill_up := total_gallons / 2
  let cost_first := gallons_per_fill_up * price_per_gallon_first
  let cost_second := gallons_per_fill_up * price_per_gallon_second
  let total_cost := cost_first + cost_second
  total_cost = 6.80 :=
by sorry

end carla_gas_cost_l72_72385


namespace general_term_formula_l72_72328

theorem general_term_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (hS : ∀ n, S n = 2 * a n - 1) : 
  ∀ n, a n = 2^(n-1) := 
by
  sorry

end general_term_formula_l72_72328


namespace triangle_area_13_14_15_l72_72899

noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
  let cos_C := (a^2 + b^2 - c^2) / (2 * a * b)
  let sin_C := Real.sqrt (1 - cos_C^2)
  (1/2) * a * b * sin_C

theorem triangle_area_13_14_15 : area_of_triangle 13 14 15 = 84 :=
by sorry

end triangle_area_13_14_15_l72_72899


namespace y_positive_if_and_only_if_x_greater_than_negative_five_over_two_l72_72428

variable (x y : ℝ)

-- Condition: y is defined as a function of x
def y_def := y = 2 * x + 5

-- Theorem: y > 0 if and only if x > -5/2
theorem y_positive_if_and_only_if_x_greater_than_negative_five_over_two 
  (h : y_def x y) : y > 0 ↔ x > -5 / 2 := by sorry

end y_positive_if_and_only_if_x_greater_than_negative_five_over_two_l72_72428


namespace eventually_repeating_last_two_digits_l72_72250

theorem eventually_repeating_last_two_digits (K : ℕ) : ∃ N : ℕ, ∃ t : ℕ, 
    (∃ s : ℕ, t = s * 77 + N) ∨ (∃ u : ℕ, t = u * 54 + N) ∧ (t % 100) / 10 = (t % 100) % 10 :=
sorry

end eventually_repeating_last_two_digits_l72_72250


namespace minimize_f_l72_72619

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x + (Real.sin x)^2

theorem minimize_f :
  ∃ x : ℝ, (-π / 4 < x ∧ x ≤ π / 2) ∧
  ∀ y : ℝ, (-π / 4 < y ∧ y ≤ π / 2) → f y ≥ f x ∧ f x = 1 ∧ x = π / 2 :=
by
  sorry

end minimize_f_l72_72619


namespace Cannot_Halve_Triangles_With_Diagonals_l72_72554

structure Polygon where
  vertices : Nat
  edges : Nat

def is_convex (n : Nat) (P : Polygon) : Prop :=
  P.vertices = n ∧ P.edges = n

def non_intersecting_diagonals (P : Polygon) : Prop :=
  -- Assuming a placeholder for the actual non-intersecting diagonals condition
  true

def count_triangles (P : Polygon) (d : non_intersecting_diagonals P) : Nat :=
  P.vertices - 2 -- This is the simplification used for counting triangles

def count_all_diagonals_triangles (P : Polygon) (d : non_intersecting_diagonals P) : Nat :=
  -- Placeholder for function to count triangles formed exclusively by diagonals
  1000

theorem Cannot_Halve_Triangles_With_Diagonals (P : Polygon) (h : is_convex 2002 P) (d : non_intersecting_diagonals P) :
  count_triangles P d = 2000 → ¬ (count_all_diagonals_triangles P d = 1000) :=
by
  intro h1
  sorry

end Cannot_Halve_Triangles_With_Diagonals_l72_72554


namespace problem_l72_72618

theorem problem (K : ℕ) : 16 ^ 3 * 8 ^ 3 = 2 ^ K → K = 21 := by
  sorry

end problem_l72_72618


namespace john_roommates_multiple_of_bob_l72_72209

theorem john_roommates_multiple_of_bob (bob_roommates john_roommates : ℕ) (multiple : ℕ) 
  (h1 : bob_roommates = 10) 
  (h2 : john_roommates = 25) 
  (h3 : john_roommates = multiple * bob_roommates + 5) : 
  multiple = 2 :=
by
  sorry

end john_roommates_multiple_of_bob_l72_72209


namespace abs_neg_two_thirds_l72_72557

theorem abs_neg_two_thirds : abs (-2/3 : ℝ) = 2/3 :=
by
  sorry

end abs_neg_two_thirds_l72_72557


namespace min_value_of_exponential_l72_72402

theorem min_value_of_exponential (x y : ℝ) (h : x + 2 * y = 3) : 2^x + 4^y = 4 * Real.sqrt 2 := by
  sorry

end min_value_of_exponential_l72_72402


namespace area_of_inscribed_square_l72_72362

noncomputable def circle_eq (x y : ℝ) : Prop := 
  3*x^2 + 3*y^2 - 15*x + 9*y + 27 = 0

theorem area_of_inscribed_square :
  (∃ x y : ℝ, circle_eq x y) →
  ∃ s : ℝ, s^2 = 25 :=
by
  sorry

end area_of_inscribed_square_l72_72362


namespace second_smallest_five_digit_in_pascals_triangle_l72_72050

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem second_smallest_five_digit_in_pascals_triangle :
  (∃ n k : ℕ, n > 0 ∧ k > 0 ∧ (10000 ≤ binomial n k) ∧ (binomial n k < 100000) ∧
    (∀ m l : ℕ, m > 0 ∧ l > 0 ∧ (10000 ≤ binomial m l) ∧ (binomial m l < 100000) →
    (binomial n k < binomial m l → binomial n k ≥ 31465)) ∧  binomial n k = 31465) :=
sorry

end second_smallest_five_digit_in_pascals_triangle_l72_72050


namespace certain_number_l72_72770

theorem certain_number (a n b : ℕ) (h1 : a = 30) (h2 : a * n = b^2) (h3 : ∀ m : ℕ, (m * n = b^2 → a ≤ m)) :
  n = 30 :=
by
  sorry

end certain_number_l72_72770


namespace find_angle_A_l72_72165

theorem find_angle_A (a b : ℝ) (B A : ℝ) 
  (ha : a = Real.sqrt 2) 
  (hb : b = Real.sqrt 3) 
  (hB : B = Real.pi / 3) : 
  A = Real.pi / 4 := 
sorry

end find_angle_A_l72_72165


namespace nat_nums_division_by_7_l72_72336

theorem nat_nums_division_by_7 (n : ℕ) : 
  (∃ q r, n = 7 * q + r ∧ q = r ∧ 1 ≤ r ∧ r < 7) ↔ 
  n = 8 ∨ n = 16 ∨ n = 24 ∨ n = 32 ∨ n = 40 ∨ n = 48 := by
  sorry

end nat_nums_division_by_7_l72_72336


namespace negation_of_proposition_l72_72384

theorem negation_of_proposition :
  (¬ ∀ (a b : ℝ), (a = b → a^2 = a * b)) = ∀ (a b : ℝ), (a ≠ b → a^2 ≠ a * b) :=
sorry

end negation_of_proposition_l72_72384


namespace infinite_series_correct_l72_72879

noncomputable def infinite_series_sum : ℚ := 
  ∑' n : ℕ, (n+1)^2 * (1/999)^n

theorem infinite_series_correct : infinite_series_sum = 997005 / 996004 :=
  sorry

end infinite_series_correct_l72_72879


namespace solve_fraction_eq_l72_72794

theorem solve_fraction_eq (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ -2) :
  (2 / (x - 2) = 3 / (x + 2)) → x = 10 :=
by
  sorry

end solve_fraction_eq_l72_72794


namespace jane_savings_l72_72850

noncomputable def cost_promotion_A (price: ℝ) : ℝ :=
  price + (price / 2)

noncomputable def cost_promotion_B (price: ℝ) : ℝ :=
  price + (price - (price * 0.25))

theorem jane_savings (price : ℝ) (h_price_pos : 0 < price) : 
  cost_promotion_B price - cost_promotion_A price = 12.5 :=
by
  let price := 50
  unfold cost_promotion_A
  unfold cost_promotion_B
  norm_num
  sorry

end jane_savings_l72_72850


namespace first_laptop_cost_l72_72539

variable (x : ℝ)

def cost_first_laptop (x : ℝ) : ℝ := x
def cost_second_laptop (x : ℝ) : ℝ := 3 * x
def total_cost (x : ℝ) : ℝ := cost_first_laptop x + cost_second_laptop x
def budget : ℝ := 2000

theorem first_laptop_cost : total_cost x = budget → x = 500 :=
by
  intros h
  sorry

end first_laptop_cost_l72_72539


namespace quadratic_has_one_solution_l72_72055

theorem quadratic_has_one_solution (k : ℝ) : (4 : ℝ) * (4 : ℝ) - k ^ 2 = 0 → k = 8 ∨ k = -8 := by
  sorry

end quadratic_has_one_solution_l72_72055


namespace tank_capacity_l72_72183

noncomputable def inflow_A (rate : ℕ) (time_hr : ℕ) : ℕ :=
rate * 60 * time_hr

noncomputable def inflow_B (rate : ℕ) (time_hr : ℕ) : ℕ :=
rate * 60 * time_hr

noncomputable def inflow_C (rate : ℕ) (time_hr : ℕ) : ℕ :=
rate * 60 * time_hr

noncomputable def outflow_X (rate : ℕ) (time_hr : ℕ) : ℕ :=
rate * 60 * time_hr

noncomputable def outflow_Y (rate : ℕ) (time_hr : ℕ) : ℕ :=
rate * 60 * time_hr

theorem tank_capacity
  (fA : ℕ := inflow_A 8 7)
  (fB : ℕ := inflow_B 12 3)
  (fC : ℕ := inflow_C 6 4)
  (oX : ℕ := outflow_X 20 7)
  (oY : ℕ := outflow_Y 15 5) :
  fA + fB + fC = 6960 ∧ oX + oY = 12900 ∧ 12900 - 6960 = 5940 :=
by
  sorry

end tank_capacity_l72_72183


namespace y_intercept_l72_72372

theorem y_intercept : ∀ (x y : ℝ), 4 * x + 7 * y = 28 → (0, 4) = (0, y) :=
by
  intros x y h
  sorry

end y_intercept_l72_72372


namespace total_books_on_shelves_l72_72990

-- Definitions based on conditions
def num_shelves : Nat := 150
def books_per_shelf : Nat := 15

-- The statement to be proved
theorem total_books_on_shelves : num_shelves * books_per_shelf = 2250 := by
  sorry

end total_books_on_shelves_l72_72990


namespace complete_the_square_l72_72248

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 6 * x - 10 = 0

-- State the proof problem
theorem complete_the_square (x : ℝ) (h : quadratic_eq x) : (x - 3)^2 = 19 :=
by 
  -- Skip the proof using sorry
  sorry

end complete_the_square_l72_72248


namespace find_k_value_l72_72267

theorem find_k_value (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 4 * x + 4 = 0) ∧
  (∀ x1 x2 : ℝ, (k * x1^2 + 4 * x1 + 4 = 0 ∧ k * x2^2 + 4 * x2 + 4 = 0) → x1 = x2) →
  (k = 0 ∨ k = 1) :=
by
  intros h
  sorry

end find_k_value_l72_72267


namespace two_pow_p_plus_three_pow_p_not_nth_power_l72_72241

theorem two_pow_p_plus_three_pow_p_not_nth_power (p n : ℕ) (prime_p : Nat.Prime p) (one_lt_n : 1 < n) :
  ¬ ∃ k : ℕ, 2 ^ p + 3 ^ p = k ^ n :=
sorry

end two_pow_p_plus_three_pow_p_not_nth_power_l72_72241


namespace sum_of_distinct_abc_eq_roots_l72_72441

noncomputable def f (x y : ℝ) : ℝ :=
  x^2 * ((x + 2*y)^2 - y^2 + x - 1)

-- Main theorem statement
theorem sum_of_distinct_abc_eq_roots (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h1 : f a (b+c) = f b (c+a)) (h2 : f b (c+a) = f c (a+b)) :
  a + b + c = (1 + Real.sqrt 5) / 2 ∨ a + b + c = (1 - Real.sqrt 5) / 2 :=
sorry

end sum_of_distinct_abc_eq_roots_l72_72441


namespace pizza_slices_left_l72_72161

theorem pizza_slices_left (initial_slices : ℕ) (people : ℕ) (slices_per_person : ℕ) 
  (h1 : initial_slices = 16) (h2 : people = 6) (h3 : slices_per_person = 2) : 
  initial_slices - people * slices_per_person = 4 := 
by
  sorry

end pizza_slices_left_l72_72161


namespace problem_one_problem_two_l72_72685

noncomputable def f (x m : ℝ) : ℝ := x^2 + m * x + 4

-- Problem (I)
theorem problem_one (m : ℝ) :
  (∀ x, 1 < x ∧ x < 2 → f x m < 0) ↔ m ≤ -5 :=
sorry

-- Problem (II)
theorem problem_two (m : ℝ) :
  (∀ x, (x = 1 ∨ x = 2) → abs ((f x m - x^2) / m) < 1) ↔ (-4 < m ∧ m ≤ -2) :=
sorry

end problem_one_problem_two_l72_72685


namespace sum_of_three_numbers_l72_72944

theorem sum_of_three_numbers (a b c : ℤ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : a + 15 = (a + b + c) / 3) (h4 : (a + b + c) / 3 = c - 20) (h5 : b = 7) :
  a + b + c = 36 :=
sorry

end sum_of_three_numbers_l72_72944


namespace subset_condition_intersection_condition_l72_72669

-- Definitions of the sets A and B
def A : Set ℝ := {2, 4}
def B (a : ℝ) : Set ℝ := {a, 3 * a}

-- Theorem statements
theorem subset_condition (a : ℝ) : A ⊆ B a → (4 / 3) ≤ a ∧ a ≤ 2 := 
by 
  sorry

theorem intersection_condition (a : ℝ) : (A ∩ B a).Nonempty → (2 / 3) < a ∧ a < 4 := 
by 
  sorry

end subset_condition_intersection_condition_l72_72669


namespace determine_borrow_lend_years_l72_72104

theorem determine_borrow_lend_years (P : ℝ) (Rb Rl G : ℝ) (n : ℝ) 
  (hP : P = 9000) 
  (hRb : Rb = 4 / 100) 
  (hRl : Rl = 6 / 100) 
  (hG : G = 180) 
  (h_gain : G = P * Rl * n - P * Rb * n) : 
  n = 1 := 
sorry

end determine_borrow_lend_years_l72_72104


namespace find_number_l72_72799

theorem find_number (x : ℝ) (h : 0.3 * x + 0.1 * 0.5 = 0.29) : x = 0.8 :=
by
  sorry

end find_number_l72_72799


namespace dvd_cost_l72_72576

-- Given conditions
def vhs_trade_in_value : Int := 2
def number_of_movies : Int := 100
def total_replacement_cost : Int := 800

-- Statement to prove
theorem dvd_cost :
  ((number_of_movies * vhs_trade_in_value) + (number_of_movies * 6) = total_replacement_cost) :=
by
  sorry

end dvd_cost_l72_72576


namespace parabola_focus_l72_72993

-- Definitions used in the conditions
def parabola_eq (p : ℝ) (x : ℝ) : ℝ := 2 * p * x^2
def passes_through (p : ℝ) : Prop := parabola_eq p 1 = 4

-- The proof that the coordinates of the focus are (0, 1/16) given the conditions
theorem parabola_focus (p : ℝ) (h : passes_through p) : p = 2 → (0, 1 / 16) = (0, 1 / (4 * p)) :=
by
  sorry

end parabola_focus_l72_72993


namespace pond_field_area_ratio_l72_72127

theorem pond_field_area_ratio (w l s A_field A_pond : ℕ) (h1 : l = 2 * w) (h2 : l = 96) (h3 : s = 8) (h4 : A_field = l * w) (h5 : A_pond = s * s) :
  A_pond.toFloat / A_field.toFloat = 1 / 72 := 
by
  sorry

end pond_field_area_ratio_l72_72127


namespace initial_number_of_friends_l72_72070

theorem initial_number_of_friends (F : ℕ) (h : 6 * (F + 2) = 60) : F = 8 :=
by {
  sorry
}

end initial_number_of_friends_l72_72070


namespace math_problem_l72_72027

theorem math_problem :
  (Real.pi - 3.14)^0 + Real.sqrt ((Real.sqrt 2 - 1)^2) = Real.sqrt 2 :=
by
  sorry

end math_problem_l72_72027


namespace P_lt_Q_l72_72995

theorem P_lt_Q (x : ℝ) (hx : x > 0) (P : ℝ) (Q : ℝ) 
  (hP : P = Real.sqrt (1 + x)) 
  (hQ : Q = 1 + x / 2) : P < Q := 
by
  sorry

end P_lt_Q_l72_72995


namespace trigonometric_proof_l72_72841

noncomputable def cos30 : ℝ := Real.sqrt 3 / 2
noncomputable def tan60 : ℝ := Real.sqrt 3
noncomputable def sin45 : ℝ := Real.sqrt 2 / 2
noncomputable def cos45 : ℝ := Real.sqrt 2 / 2

theorem trigonometric_proof :
  2 * cos30 - tan60 + sin45 * cos45 = 1 / 2 :=
by
  sorry

end trigonometric_proof_l72_72841


namespace sum_unchanged_difference_changes_l72_72349

-- Definitions from conditions
def original_sum (a b c : ℤ) := a + b + c
def new_first (a : ℤ) := a - 329
def new_second (b : ℤ) := b + 401

-- Problem statement for sum unchanged
theorem sum_unchanged (a b c : ℤ) (h : original_sum a b c = 1281) :
  original_sum (new_first a) (new_second b) (c - 72) = 1281 := by
  sorry

-- Definitions for difference condition
def abs_diff (x y : ℤ) := abs (x - y)
def alter_difference (a b c : ℤ) :=
  abs_diff (new_first a) (new_second b) + abs_diff (new_first a) c + abs_diff b c

-- Problem statement addressing the difference
theorem difference_changes (a b c : ℤ) (h : original_sum a b c = 1281) :
  alter_difference a b c = abs_diff (new_first a) (new_second b) + abs_diff (c - 730) (new_first a) + abs_diff (c - 730) (new_first a) := by
  sorry

end sum_unchanged_difference_changes_l72_72349


namespace slower_train_speed_l72_72603

-- Define the given conditions
def speed_faster_train : ℝ := 50  -- km/h
def length_faster_train : ℝ := 75.006  -- meters
def passing_time : ℝ := 15  -- seconds

-- Conversion factor
def mps_to_kmph : ℝ := 3.6

-- Define the problem to be proved
theorem slower_train_speed : 
  ∃ speed_slower_train : ℝ, 
    speed_slower_train = speed_faster_train - (75.006 / 15) * mps_to_kmph := 
  by
    exists 31.99856
    sorry

end slower_train_speed_l72_72603


namespace combined_dog_years_difference_l72_72828

theorem combined_dog_years_difference 
  (Max_age : ℕ) 
  (small_breed_rate medium_breed_rate large_breed_rate : ℕ) 
  (Max_turns_age : ℕ) 
  (small_breed_diff medium_breed_diff large_breed_diff combined_diff : ℕ) :
  Max_age = 3 →
  small_breed_rate = 5 →
  medium_breed_rate = 7 →
  large_breed_rate = 9 →
  Max_turns_age = 6 →
  small_breed_diff = small_breed_rate * Max_turns_age - Max_turns_age →
  medium_breed_diff = medium_breed_rate * Max_turns_age - Max_turns_age →
  large_breed_diff = large_breed_rate * Max_turns_age - Max_turns_age →
  combined_diff = small_breed_diff + medium_breed_diff + large_breed_diff →
  combined_diff = 108 :=
by
  intros
  sorry

end combined_dog_years_difference_l72_72828


namespace smallest_positive_q_with_property_l72_72777

theorem smallest_positive_q_with_property :
  ∃ q : ℕ, (
    q > 0 ∧
    ∀ m : ℕ, (1 ≤ m ∧ m ≤ 1006) →
    ∃ n : ℤ, 
      (m * q : ℤ) / 1007 < n ∧
      (m + 1) * q / 1008 > n) ∧
   q = 2015 := 
sorry

end smallest_positive_q_with_property_l72_72777


namespace scientific_notation_460_billion_l72_72901

theorem scientific_notation_460_billion : 460000000000 = 4.6 * 10^11 := 
sorry

end scientific_notation_460_billion_l72_72901


namespace relationship_l72_72140

-- Define sequences
variable (a b : ℕ → ℝ)

-- Define conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, 1 < m → m < n → a m = a 1 + (m - 1) * (a n - a 1) / (n - 1)

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, 1 < m → m < n → b m = b 1 * (b n / b 1)^(m - 1) / (n - 1)

noncomputable def sequences_conditions : Prop :=
  a 1 = b 1 ∧ a 1 > 0 ∧ ∀ n, a n = b n ∧ b n > 0

-- The main theorem
theorem relationship (h: sequences_conditions a b) : ∀ m n : ℕ, 1 < m → m < n → a m ≥ b m := 
by
  sorry

end relationship_l72_72140


namespace unique_outfits_count_l72_72867

theorem unique_outfits_count (s : Fin 5) (p : Fin 6) (restricted_pairings : (Fin 1 × Fin 2) → Prop) 
  (r : restricted_pairings (0, 0) ∧ restricted_pairings (0, 1)) : ∃ n, n = 28 ∧ 
  ∃ (outfits : Fin 5 → Fin 6 → Prop), 
    (∀ s p, outfits s p) ∧ 
    (∀ p, ¬outfits 0 p ↔ p = 0 ∨ p = 1) := by
  sorry

end unique_outfits_count_l72_72867


namespace aunt_may_milk_leftover_l72_72820

noncomputable def milk_leftover : Real :=
let morning_milk := 5 * 13 + 4 * 0.5 + 10 * 0.25
let evening_milk := 5 * 14 + 4 * 0.6 + 10 * 0.2

let morning_spoiled := morning_milk * 0.1
let cheese_produced := morning_milk * 0.15
let remaining_morning_milk := morning_milk - morning_spoiled - cheese_produced
let ice_cream_sale := remaining_morning_milk * 0.7

let evening_spoiled := evening_milk * 0.05
let remaining_evening_milk := evening_milk - evening_spoiled
let cheese_shop_sale := remaining_evening_milk * 0.8

let leftover_previous_day := 15
let remaining_morning_after_sale := remaining_morning_milk - ice_cream_sale
let remaining_evening_after_sale := remaining_evening_milk - cheese_shop_sale

leftover_previous_day + remaining_morning_after_sale + remaining_evening_after_sale

theorem aunt_may_milk_leftover : 
  milk_leftover = 44.7735 := 
sorry

end aunt_may_milk_leftover_l72_72820


namespace mass_of_fat_max_mass_of_carbohydrates_l72_72075

-- Definitions based on conditions
def total_mass : ℤ := 500
def fat_percentage : ℚ := 5 / 100
def protein_to_mineral_ratio : ℤ := 4

-- Lean 4 statement for Part 1: mass of fat
theorem mass_of_fat : (total_mass : ℚ) * fat_percentage = 25 := sorry

-- Definitions to utilize in Part 2
def max_percentage_protein_carbs : ℚ := 85 / 100
def mass_protein (x : ℚ) : ℚ := protein_to_mineral_ratio * x

-- Lean 4 statement for Part 2: maximum mass of carbohydrates
theorem max_mass_of_carbohydrates (x : ℚ) :
  x ≥ 50 → (total_mass - 25 - x - mass_protein x) ≤ 225 := sorry

end mass_of_fat_max_mass_of_carbohydrates_l72_72075


namespace quotient_of_division_l72_72268

theorem quotient_of_division (L : ℕ) (S : ℕ) (Q : ℕ) (h1 : L = 1631) (h2 : L - S = 1365) (h3 : L = S * Q + 35) :
  Q = 6 :=
by
  sorry

end quotient_of_division_l72_72268


namespace increasing_order_magnitudes_l72_72938

variable (x : ℝ)

noncomputable def y := x^x
noncomputable def z := x^(x^x)

theorem increasing_order_magnitudes (h1 : 1 < x) (h2 : x < 1.1) : x < y x ∧ y x < z x :=
by
  have h3 : y x = x^x := rfl
  have h4 : z x = x^(x^x) := rfl
  sorry

end increasing_order_magnitudes_l72_72938


namespace area_R2_l72_72284

-- Definitions from conditions
def side_R1 : ℕ := 3
def area_R1 : ℕ := 24
def diagonal_ratio : ℤ := 2

-- Introduction of the theorem
theorem area_R2 (similar: ℤ) (a b: ℕ) :
  a * b = area_R1 ∧
  a = 3 ∧
  b * 3 = 8 * a ∧
  (a^2 + b^2 = 292) ∧
  similar * (a^2 + b^2) = 2 * 2 * 73 →
  (6 * 16 = 96) := by
sorry

end area_R2_l72_72284


namespace necessary_but_not_sufficient_l72_72516

theorem necessary_but_not_sufficient (a : ℝ) : (a < 2 → a^2 < 2 * a) ∧ (a^2 < 2 * a → 0 < a ∧ a < 2) := sorry

end necessary_but_not_sufficient_l72_72516


namespace find_principal_amount_l72_72117

theorem find_principal_amount (P r : ℝ) (A2 A3 : ℝ) (n2 n3 : ℕ) 
  (h1 : n2 = 2) (h2 : n3 = 3) 
  (h3 : A2 = 8820) 
  (h4 : A3 = 9261) 
  (h5 : r = 0.05) 
  (h6 : A2 = P * (1 + r)^n2) 
  (h7 : A3 = P * (1 + r)^n3) : 
  P = 8000 := 
by 
  sorry

end find_principal_amount_l72_72117


namespace general_formula_l72_72088

noncomputable def a : ℕ → ℕ
| 0       => 5
| (n + 1) => 2 * a n + 3

theorem general_formula : ∀ n, a n = 2 ^ (n + 2) - 3 :=
by
  sorry

end general_formula_l72_72088


namespace min_value_of_expr_min_value_at_specific_points_l72_72896

noncomputable def min_value_expr (p q r : ℝ) : ℝ := 8 * p^4 + 18 * q^4 + 50 * r^4 + 1 / (8 * p * q * r)

theorem min_value_of_expr : ∀ (p q r : ℝ), p > 0 → q > 0 → r > 0 → min_value_expr p q r ≥ 6 :=
by
  intro p q r hp hq hr
  sorry

theorem min_value_at_specific_points : min_value_expr (1 / (8 : ℝ)^(1 / 4)) (1 / (18 : ℝ)^(1 / 4)) (1 / (50 : ℝ)^(1 / 4)) = 6 :=
by
  sorry

end min_value_of_expr_min_value_at_specific_points_l72_72896


namespace krios_population_limit_l72_72049

theorem krios_population_limit (initial_population : ℕ) (acre_per_person : ℕ) (total_acres : ℕ) (doubling_years : ℕ) :
  initial_population = 150 →
  acre_per_person = 2 →
  total_acres = 35000 →
  doubling_years = 30 →
  ∃ (years_from_2005 : ℕ), years_from_2005 = 210 ∧ (initial_population * 2^(years_from_2005 / doubling_years)) ≥ total_acres / acre_per_person :=
by
  intros
  sorry

end krios_population_limit_l72_72049


namespace inequality_mn_l72_72954

theorem inequality_mn (m n : ℤ)
  (h : ∃ x : ℤ, (x + m) * (x + n) = x + m + n) : 
  2 * (m^2 + n^2) < 5 * m * n := 
sorry

end inequality_mn_l72_72954


namespace cos_210_eq_neg_sqrt3_over_2_l72_72300

theorem cos_210_eq_neg_sqrt3_over_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_210_eq_neg_sqrt3_over_2_l72_72300


namespace sum_of_coefficients_l72_72925

-- Define the polynomial
def polynomial (x : ℝ) : ℝ :=
  2 * (4 * x ^ 8 + 7 * x ^ 6 - 9 * x ^ 3 + 3) + 6 * (x ^ 7 - 2 * x ^ 4 + 8 * x ^ 2 - 2)

-- State the theorem to prove the sum of the coefficients
theorem sum_of_coefficients : polynomial 1 = 40 :=
by
  sorry

end sum_of_coefficients_l72_72925


namespace translation_2_units_left_l72_72433

-- Define the initial parabola
def parabola1 (x : ℝ) : ℝ := x^2 + 1

-- Define the translated parabola
def parabola2 (x : ℝ) : ℝ := x^2 + 4 * x + 5

-- State that parabola2 is obtained by translating parabola1
-- And prove that this translation is 2 units to the left
theorem translation_2_units_left :
  ∀ x : ℝ, parabola2 x = parabola1 (x + 2) := 
by
  sorry

end translation_2_units_left_l72_72433


namespace triple_layers_area_l72_72734

-- Defining the conditions
def hall : Type := {x // x = 10 * 10}
def carpet1 : hall := ⟨60, sorry⟩ -- First carpet size: 6 * 8
def carpet2 : hall := ⟨36, sorry⟩ -- Second carpet size: 6 * 6
def carpet3 : hall := ⟨35, sorry⟩ -- Third carpet size: 5 * 7

-- The final theorem statement
theorem triple_layers_area : ∃ area : ℕ, area = 6 :=
by
  have intersection_area : ℕ := 2 * 3
  use intersection_area
  sorry

end triple_layers_area_l72_72734


namespace earrings_cost_l72_72760

theorem earrings_cost (initial_savings necklace_cost remaining_savings : ℕ) 
  (h_initial : initial_savings = 80) 
  (h_necklace : necklace_cost = 48) 
  (h_remaining : remaining_savings = 9) : 
  initial_savings - remaining_savings - necklace_cost = 23 := 
by {
  -- insert proof steps here -- 
  sorry
}

end earrings_cost_l72_72760


namespace class_avg_GPA_l72_72643

theorem class_avg_GPA (n : ℕ) (h1 : n > 0) : 
  ((1 / 4 : ℝ) * 92 + (3 / 4 : ℝ) * 76 = 80) :=
sorry

end class_avg_GPA_l72_72643


namespace prove_y_identity_l72_72891

theorem prove_y_identity (y : ℤ) (h1 : y^2 = 2209) : (y + 2) * (y - 2) = 2205 :=
by
  sorry

end prove_y_identity_l72_72891


namespace investment_time_period_l72_72036

theorem investment_time_period :
  ∀ (A P : ℝ) (R : ℝ) (T : ℝ),
  A = 896 → P = 799.9999999999999 → R = 5 →
  (A - P) = (P * R * T / 100) → T = 2.4 :=
by
  intros A P R T hA hP hR hSI
  sorry

end investment_time_period_l72_72036


namespace find_y_value_l72_72048

theorem find_y_value 
  (k : ℝ) 
  (y : ℝ) 
  (hx81 : y = 3 * Real.sqrt 2)
  (h_eq : ∀ (x : ℝ), y = k * x ^ (1 / 4)) 
  : (∃ y, y = 2 ∧ y = k * 4 ^ (1 / 4))
:= sorry

end find_y_value_l72_72048


namespace jasmine_percentage_l72_72399

namespace ProofExample

variables (original_volume : ℝ) (initial_percent_jasmine : ℝ) (added_jasmine : ℝ) (added_water : ℝ)
variables (initial_jasmine : ℝ := initial_percent_jasmine * original_volume / 100)
variables (total_jasmine : ℝ := initial_jasmine + added_jasmine)
variables (total_volume : ℝ := original_volume + added_jasmine + added_water)
variables (final_percent_jasmine : ℝ := (total_jasmine / total_volume) * 100)

theorem jasmine_percentage 
  (h1 : original_volume = 80)
  (h2 : initial_percent_jasmine = 10)
  (h3 : added_jasmine = 8)
  (h4 : added_water = 12)
  : final_percent_jasmine = 16 := 
sorry

end ProofExample

end jasmine_percentage_l72_72399


namespace trisha_bought_amount_initially_l72_72532

-- Define the amounts spent on each item
def meat : ℕ := 17
def chicken : ℕ := 22
def veggies : ℕ := 43
def eggs : ℕ := 5
def dogs_food : ℕ := 45
def amount_left : ℕ := 35

-- Define the total amount spent
def total_spent : ℕ := meat + chicken + veggies + eggs + dogs_food

-- Define the amount brought at the beginning
def amount_brought_at_beginning : ℕ := total_spent + amount_left

-- Theorem stating the amount Trisha brought at the beginning is 167
theorem trisha_bought_amount_initially : amount_brought_at_beginning = 167 := by
  -- Formal proof would go here, we use sorry to skip the proof
  sorry

end trisha_bought_amount_initially_l72_72532


namespace max_distance_from_center_of_square_l72_72482

theorem max_distance_from_center_of_square :
  let A := (0, 0)
  let B := (1, 0)
  let C := (1, 1)
  let D := (0, 1)
  let O := (0.5, 0.5)
  ∃ P : ℝ × ℝ, 
  (let u := dist P A
   let v := dist P B
   let w := dist P C
   u^2 + v^2 + w^2 = 2)
  → dist O P = (1 + 2 * Real.sqrt 2) / (3 * Real.sqrt 2) :=
by sorry

end max_distance_from_center_of_square_l72_72482


namespace investment_share_l72_72483

variable (P_investment Q_investment : ℝ)

theorem investment_share (h1 : Q_investment = 60000) (h2 : P_investment / Q_investment = 2 / 3) : P_investment = 40000 := by
  sorry

end investment_share_l72_72483


namespace cube_sum_of_edges_corners_faces_eq_26_l72_72710

theorem cube_sum_of_edges_corners_faces_eq_26 :
  let edges := 12
  let corners := 8
  let faces := 6
  edges + corners + faces = 26 :=
by
  let edges := 12
  let corners := 8
  let faces := 6
  sorry

end cube_sum_of_edges_corners_faces_eq_26_l72_72710


namespace value_of_livestock_l72_72073

variable (x y : ℝ)

theorem value_of_livestock :
  (5 * x + 2 * y = 10) ∧ (2 * x + 5 * y = 8) :=
sorry

end value_of_livestock_l72_72073


namespace sequence_general_formula_l72_72216

theorem sequence_general_formula (a : ℕ → ℕ) (S : ℕ → ℕ) :
  a 2 = 4 →
  S 4 = 30 →
  (∀ n, n ≥ 2 → a (n + 1) + a (n - 1) = 2 * (a n + 1)) →
  ∀ n, a n = n^2 :=
by
  intros h1 h2 h3
  sorry

end sequence_general_formula_l72_72216


namespace arithmetic_geometric_sequence_sum_l72_72948

theorem arithmetic_geometric_sequence_sum 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∃ x y z : ℝ, (x = a ∧ y = -4 ∧ z = b ∨ x = b ∧ y = -4 ∧ z = a) 
                   ∧ (x + z = 2 * y) ∧ (x * z = y^2)) : 
  a + b = 10 :=
by sorry

end arithmetic_geometric_sequence_sum_l72_72948


namespace scientific_notation_of_16907_l72_72742

theorem scientific_notation_of_16907 :
  16907 = 1.6907 * 10^4 :=
sorry

end scientific_notation_of_16907_l72_72742


namespace primary_school_capacity_l72_72609

variable (x : ℝ)

/-- In a town, there are four primary schools. Two of them can teach 400 students at a time, 
and the other two can teach a certain number of students at a time. These four primary schools 
can teach a total of 1480 students at a time. -/
theorem primary_school_capacity 
  (h1 : 2 * 400 + 2 * x = 1480) : 
  x = 340 :=
sorry

end primary_school_capacity_l72_72609


namespace least_common_denominator_l72_72272

-- Define the list of numbers
def numbers : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9]

-- Define the least common multiple function
noncomputable def lcm_list (l : List ℕ) : ℕ :=
  l.foldr Nat.lcm 1

-- Define the main theorem
theorem least_common_denominator : lcm_list numbers = 2520 := 
  by sorry

end least_common_denominator_l72_72272


namespace line_intersects_circle_l72_72759

theorem line_intersects_circle
    (r : ℝ) (d : ℝ)
    (hr : r = 6) (hd : d = 5) : d < r :=
by
    rw [hr, hd]
    exact by norm_num

end line_intersects_circle_l72_72759


namespace molecular_weight_of_one_mole_l72_72931

noncomputable def molecular_weight (total_weight : ℝ) (moles : ℕ) : ℝ :=
total_weight / moles

theorem molecular_weight_of_one_mole (h : molecular_weight 252 6 = 42) : molecular_weight 252 6 = 42 := by
  exact h

end molecular_weight_of_one_mole_l72_72931


namespace circumscribedCircleDiameter_is_10sqrt2_l72_72975

noncomputable def circumscribedCircleDiameter (a : ℝ) (A : ℝ) : ℝ :=
  a / Real.sin A

theorem circumscribedCircleDiameter_is_10sqrt2 :
  circumscribedCircleDiameter 10 (Real.pi / 4) = 10 * Real.sqrt 2 :=
by
  sorry

end circumscribedCircleDiameter_is_10sqrt2_l72_72975


namespace perfect_cubes_in_range_l72_72244

theorem perfect_cubes_in_range (K : ℤ) (hK_pos : K > 1) (Z : ℤ) 
  (hZ_eq : Z = K ^ 3) (hZ_range: 600 < Z ∧ Z < 2000) :
  K = 9 ∨ K = 10 ∨ K = 11 ∨ K = 12 :=
by
  sorry

end perfect_cubes_in_range_l72_72244


namespace factorize_expression_l72_72600

variable (a b : ℝ)

theorem factorize_expression : ab^2 - a = a * (b + 1) * (b - 1) :=
sorry

end factorize_expression_l72_72600


namespace tetrahedron_ratio_l72_72207

theorem tetrahedron_ratio (a b c d : ℝ) (h₁ : a^2 = b^2 + c^2) (h₂ : b^2 = a^2 + d^2) (h₃ : c^2 = a^2 + b^2) : 
  a / d = Real.sqrt ((1 + Real.sqrt 5) / 2) :=
sorry

end tetrahedron_ratio_l72_72207


namespace expression_value_l72_72530

theorem expression_value : (36 + 9) ^ 2 - (9 ^ 2 + 36 ^ 2) = -1894224 :=
by
  sorry

end expression_value_l72_72530


namespace pizza_slices_with_both_toppings_l72_72909

theorem pizza_slices_with_both_toppings :
  let T := 16
  let c := 10
  let p := 12
  let n := c + p - T
  n = 6 :=
by
  let T := 16
  let c := 10
  let p := 12
  let n := c + p - T
  show n = 6
  sorry

end pizza_slices_with_both_toppings_l72_72909


namespace M_inter_N_eq_l72_72697

open Set

def M : Set ℕ := {1, 2, 3, 4}
def N : Set ℕ := {3, 4, 5, 6}

theorem M_inter_N_eq : M ∩ N = {3, 4} := 
by 
  sorry

end M_inter_N_eq_l72_72697


namespace probability_of_same_color_when_rolling_two_24_sided_dice_l72_72491

-- Defining the conditions
def numSides : ℕ := 24
def purpleSides : ℕ := 5
def blueSides : ℕ := 8
def redSides : ℕ := 10
def goldSides : ℕ := 1

-- Required to use rational numbers for probabilities
def probability (eventSides : ℕ) (totalSides : ℕ) : ℚ := eventSides / totalSides

-- Main theorem statement
theorem probability_of_same_color_when_rolling_two_24_sided_dice :
  probability purpleSides numSides * probability purpleSides numSides +
  probability blueSides numSides * probability blueSides numSides +
  probability redSides numSides * probability redSides numSides +
  probability goldSides numSides * probability goldSides numSides =
  95 / 288 :=
by
  sorry

end probability_of_same_color_when_rolling_two_24_sided_dice_l72_72491


namespace g_is_even_l72_72831

noncomputable def g (x : ℝ) : ℝ := Real.log (Real.cos x + Real.sqrt (1 + Real.sin x ^ 2))

theorem g_is_even : ∀ x : ℝ, g (-x) = g (x) :=
by
  intro x
  sorry

end g_is_even_l72_72831


namespace find_missing_number_l72_72878

theorem find_missing_number (n : ℤ) (h : 1234562 - n * 3 * 2 = 1234490) : 
  n = 12 :=
by
  sorry

end find_missing_number_l72_72878


namespace Xiaohuo_books_l72_72627

def books_proof_problem : Prop :=
  ∃ (X_H X_Y X_Z : ℕ), 
    (X_H + X_Y + X_Z = 1248) ∧ 
    (X_H = X_Y + 64) ∧ 
    (X_Y = X_Z - 32) ∧ 
    (X_H = 448)

theorem Xiaohuo_books : books_proof_problem :=
by
  sorry

end Xiaohuo_books_l72_72627


namespace book_cost_price_l72_72866

theorem book_cost_price (SP : ℝ) (P : ℝ) (C : ℝ) (hSP: SP = 260) (hP: P = 0.20) : C = 216.67 :=
by 
  sorry

end book_cost_price_l72_72866


namespace max_m_n_sq_l72_72816

theorem max_m_n_sq (m n : ℕ) (hm : 1 ≤ m ∧ m ≤ 1981) (hn : 1 ≤ n ∧ n ≤ 1981)
  (h : (n^2 - m * n - m^2)^2 = 1) : m^2 + n^2 ≤ 3524578 :=
sorry

end max_m_n_sq_l72_72816


namespace first_term_arithmetic_sum_l72_72988

theorem first_term_arithmetic_sum 
  (T : ℕ → ℚ) (b : ℚ) (d : ℚ) (h₁ : ∀ n, T n = n * (2 * b + (n - 1) * d) / 2)
  (h₂ : d = 5)
  (h₃ : ∀ n, (T (4 * n)) / (T n) = (16 : ℚ)) : 
  b = 5 / 2 :=
sorry

end first_term_arithmetic_sum_l72_72988


namespace new_sales_volume_monthly_profit_maximize_profit_l72_72956

-- Define assumptions and variables
variables (x : ℝ) (p : ℝ) (v : ℝ) (profit : ℝ)

-- Part 1: New sales volume after price increase
theorem new_sales_volume (h : 0 < x ∧ x < 20) : v = 600 - 10 * x :=
sorry

-- Part 2: Price and quantity for a monthly profit of 10,000 yuan
theorem monthly_profit (h : profit = (40 + x - 30) * (600 - 10 * x)) (h2: profit = 10000) : p = 50 ∧ v = 500 :=
sorry

-- Part 3: Price for maximizing monthly sales profit
theorem maximize_profit (h : profit = (40 + x - 30) * (600 - 10 * x)) : (∃ x_max: ℝ, x_max < 20 ∧ ∀ x, x < 20 → profit ≤ -10 * (x - 25)^2 + 12250 ∧ p = 59 ∧ profit = 11890) :=
sorry

end new_sales_volume_monthly_profit_maximize_profit_l72_72956


namespace triangle_angle_bisector_YE_l72_72829

noncomputable def triangle_segs_YE : ℝ := (36 : ℝ) / 7

theorem triangle_angle_bisector_YE
  (XYZ: Type)
  (XY XZ YZ YE EZ: ℝ)
  (YZ_length : YZ = 12)
  (side_ratios : XY / XZ = 3 / 4 ∧ XY / YZ  = 3 / 5 ∧ XZ / YZ = 4 / 5)
  (angle_bisector : YE / EZ = XY / XZ)
  (seg_sum : YE + EZ = YZ) :
  YE = (36 : ℝ) / 7 :=
by sorry

end triangle_angle_bisector_YE_l72_72829


namespace find_BC_length_l72_72322

noncomputable def area_triangle (A B C : ℝ) : ℝ :=
  1/2 * A * B * C

theorem find_BC_length (A B C : ℝ) (angleA : ℝ)
  (h1 : area_triangle 5 A (Real.sin (π / 6)) = 5 * Real.sqrt 3)
  (h2 : B = 5)
  (h3 : angleA = π / 6) :
  C = Real.sqrt 13 :=
by
  sorry

end find_BC_length_l72_72322


namespace q1_q2_q3_l72_72890

-- (1) Given |a| = 3, |b| = 1, and a < b, prove a + b = -2 or -4.
theorem q1 (a b : ℚ) (h1 : |a| = 3) (h2 : |b| = 1) (h3 : a < b) : a + b = -2 ∨ a + b = -4 := sorry

-- (2) Given rational numbers a and b such that ab ≠ 0, prove the value of (a/|a|) + (b/|b|) is 2, -2, or 0.
theorem q2 (a b : ℚ) (h1 : a ≠ 0) (h2 : b ≠ 0) : (a / |a|) + (b / |b|) = 2 ∨ (a / |a|) + (b / |b|) = -2 ∨ (a / |a|) + (b / |b|) = 0 := sorry

-- (3) Given rational numbers a, b, c such that a + b + c = 0 and abc < 0, prove the value of (b+c)/|a| + (a+c)/|b| + (a+b)/|c| is -1.
theorem q3 (a b c : ℚ) (h1 : a + b + c = 0) (h2 : a * b * c < 0) : (b + c) / |a| + (a + c) / |b| + (a + b) / |c| = -1 := sorry

end q1_q2_q3_l72_72890


namespace find_x_l72_72397

theorem find_x : 
  (5 * 12 / (180 / 3) = 1) → (∃ x : ℕ, 1 + x = 81 ∧ x = 80) :=
by
  sorry

end find_x_l72_72397


namespace pelican_speed_l72_72592

theorem pelican_speed
  (eagle_speed falcon_speed hummingbird_speed total_distance time : ℕ)
  (eagle_distance falcon_distance hummingbird_distance : ℕ)
  (H1 : eagle_speed = 15)
  (H2 : falcon_speed = 46)
  (H3 : hummingbird_speed = 30)
  (H4 : time = 2)
  (H5 : total_distance = 248)
  (H6 : eagle_distance = eagle_speed * time)
  (H7 : falcon_distance = falcon_speed * time)
  (H8 : hummingbird_distance = hummingbird_speed * time)
  (total_other_birds_distance : ℕ)
  (H9 : total_other_birds_distance = eagle_distance + falcon_distance + hummingbird_distance)
  (pelican_distance : ℕ)
  (H10 : pelican_distance = total_distance - total_other_birds_distance)
  (pelican_speed : ℕ)
  (H11 : pelican_speed = pelican_distance / time) :
  pelican_speed = 33 := 
  sorry

end pelican_speed_l72_72592


namespace fractionSpentOnMachinery_l72_72024

-- Given conditions
def companyCapital (C : ℝ) : Prop := 
  ∃ remainingCapital, remainingCapital = 0.675 * C ∧ 
  ∃ rawMaterial, rawMaterial = (1/4) * C ∧ 
  ∃ remainingAfterRaw, remainingAfterRaw = (3/4) * C ∧ 
  ∃ spentOnMachinery, spentOnMachinery = remainingAfterRaw - remainingCapital

-- Question translated to Lean statement
theorem fractionSpentOnMachinery (C : ℝ) (h : companyCapital C) : 
  ∃ remainingAfterRaw spentOnMachinery,
    spentOnMachinery / remainingAfterRaw = 1/10 :=
by 
  sorry

end fractionSpentOnMachinery_l72_72024


namespace steps_climbed_l72_72785

-- Definitions
def flights : ℕ := 9
def feet_per_flight : ℕ := 10
def inches_per_step : ℕ := 18

-- Proving the number of steps John climbs up
theorem steps_climbed : 
  let feet_total := flights * feet_per_flight
  let inches_total := feet_total * 12
  let steps := inches_total / inches_per_step
  steps = 60 := 
by
  let feet_total := flights * feet_per_flight
  let inches_total := feet_total * 12
  let steps := inches_total / inches_per_step
  sorry

end steps_climbed_l72_72785


namespace gcd_50420_35313_l72_72984

theorem gcd_50420_35313 : Int.gcd 50420 35313 = 19 := 
sorry

end gcd_50420_35313_l72_72984


namespace inverse_function_point_l72_72622

theorem inverse_function_point (f : ℝ → ℝ) (h_inv : Function.LeftInverse f f⁻¹) (h_point : f 2 = -1) : f⁻¹ (-1) = 2 :=
by
  sorry

end inverse_function_point_l72_72622


namespace convert_to_dms_l72_72792

-- Define the conversion factors
def degrees_to_minutes (d : ℝ) : ℝ := d * 60
def minutes_to_seconds (m : ℝ) : ℝ := m * 60

-- The main proof statement
theorem convert_to_dms (d : ℝ) :
  d = 24.29 →
  (24, 17, 24) = (24, degrees_to_minutes (0.29), minutes_to_seconds 0.4) :=
by
  sorry

end convert_to_dms_l72_72792


namespace contrapositive_example_l72_72542

theorem contrapositive_example (a b m : ℝ) :
  (a > b → a * (m^2 + 1) > b * (m^2 + 1)) ↔ (a * (m^2 + 1) ≤ b * (m^2 + 1) → a ≤ b) :=
by sorry

end contrapositive_example_l72_72542


namespace Maggie_takes_75_percent_l72_72067

def Debby's_portion : ℚ := 0.25
def Maggie's_share : ℚ := 4500
def Total_amount : ℚ := 6000
def Maggie's_portion : ℚ := Maggie's_share / Total_amount

theorem Maggie_takes_75_percent : Maggie's_portion = 0.75 :=
by
  sorry

end Maggie_takes_75_percent_l72_72067


namespace trigonometric_inequality_l72_72817

-- Define the necessary mathematical objects and structures:
noncomputable def sin (x : ℝ) : ℝ := sorry -- Assume sine function as given

-- The theorem statement
theorem trigonometric_inequality {x y z A B C : ℝ} 
  (hA : A + B + C = π) -- A, B, C are angles of a triangle
  :
  ((x + y + z) / 2) ^ 2 ≥ x * y * (sin A) ^ 2 + y * z * (sin B) ^ 2 + z * x * (sin C) ^ 2 :=
sorry

end trigonometric_inequality_l72_72817


namespace product_of_two_consecutive_integers_sum_lt_150_l72_72280

theorem product_of_two_consecutive_integers_sum_lt_150 :
  ∃ (n : Nat), n * (n + 1) = 5500 ∧ 2 * n + 1 < 150 :=
by
  sorry

end product_of_two_consecutive_integers_sum_lt_150_l72_72280


namespace zoo_initial_animals_l72_72185

theorem zoo_initial_animals (X : ℕ) :
  X - 6 + 1 + 3 + 8 + 16 = 90 → X = 68 :=
by
  intro h
  sorry

end zoo_initial_animals_l72_72185


namespace pyramid_height_l72_72534

def height_of_pyramid (n : ℕ) : ℕ :=
  2 * (n - 1)

theorem pyramid_height (n : ℕ) : height_of_pyramid n = 2 * (n - 1) :=
by
  -- The proof would typically go here
  sorry

end pyramid_height_l72_72534


namespace find_p_root_relation_l72_72150

theorem find_p_root_relation (p : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ 0 ∧ x2 = 3 * x1 ∧ x1^2 + p * x1 + 2 * p = 0 ∧ x2^2 + p * x2 + 2 * p = 0) ↔ (p = 0 ∨ p = 32 / 3) :=
by sorry

end find_p_root_relation_l72_72150


namespace op_add_mul_example_l72_72130

def op_add (a b : ℤ) : ℤ := a + b - 1
def op_mul (a b : ℤ) : ℤ := a * b - 1

theorem op_add_mul_example : op_mul (op_add 6 8) (op_add 3 5) = 90 :=
by
  -- Rewriting it briefly without proof steps
  sorry

end op_add_mul_example_l72_72130


namespace find_f_107_5_l72_72782

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : ∀ x, f x = f (-x)
axiom func_eq : ∀ x, f (x + 3) = - (1 / f x)
axiom cond_interval : ∀ x, -3 ≤ x ∧ x ≤ -2 → f x = 4 * x

theorem find_f_107_5 : f 107.5 = 1 / 10 := by {
  sorry
}

end find_f_107_5_l72_72782


namespace total_steps_l72_72497

def steps_on_feet (jason_steps : Nat) (nancy_ratio : Nat) : Nat :=
  jason_steps + (nancy_ratio * jason_steps)

theorem total_steps (jason_steps : Nat) (nancy_ratio : Nat) (h1 : jason_steps = 8) (h2 : nancy_ratio = 3) :
  steps_on_feet jason_steps nancy_ratio = 32 :=
by
  sorry

end total_steps_l72_72497


namespace problem_part1_problem_part2_l72_72188

noncomputable def arithmetic_sequence (a : ℕ → ℕ) :=
  ∀ n : ℕ, a (n + 1) = a n + 2

theorem problem_part1 (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : a 1 = 2) (h2 : S 2 = a 3) (h3 : arithmetic_sequence a) :
  a 2 = 4 := 
sorry

theorem problem_part2 (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : a 1 = 2) (h2 : S 2 = a 3) (h3 : arithmetic_sequence a) 
  (h4 : ∀ n : ℕ, S n = n * (a 1 + a n) / 2) :
  S 10 = 110 :=
sorry

end problem_part1_problem_part2_l72_72188


namespace sam_pens_count_l72_72290

-- Lean 4 statement
theorem sam_pens_count :
  ∃ (black_pens blue_pens pencils red_pens : ℕ),
    (black_pens = blue_pens + 10) ∧
    (blue_pens = 2 * pencils) ∧
    (pencils = 8) ∧
    (red_pens = pencils - 2) ∧
    (black_pens + blue_pens + red_pens = 48) :=
by {
  sorry
}

end sam_pens_count_l72_72290


namespace solve_x_l72_72192

theorem solve_x (x : ℝ) (h : (4 * x + 3) / (3 * x ^ 2 + 4 * x - 4) = 3 * x / (3 * x - 2)) :
  x = (-1 + Real.sqrt 10) / 3 ∨ x = (-1 - Real.sqrt 10) / 3 :=
by sorry

end solve_x_l72_72192


namespace find_angle_phi_l72_72916

-- Definitions for the conditions given in the problem
def folded_paper_angle (φ : ℝ) : Prop := 0 < φ ∧ φ < 90

def angle_XOY := 144

-- The main statement to be proven
theorem find_angle_phi (φ : ℝ) (h1 : folded_paper_angle φ) : φ = 81 :=
sorry

end find_angle_phi_l72_72916


namespace smaller_rectangle_perimeter_l72_72240

def problem_conditions (a b : ℝ) : Prop :=
  2 * (a + b) = 96 ∧ 
  8 * b + 11 * a = 342 ∧
  a + b = 48 ∧ 
  (a * (b - 1) <= 0 ∧ b * (a - 1) <= 0 ∧ a > 0 ∧ b > 0)

theorem smaller_rectangle_perimeter (a b : ℝ) (hab : problem_conditions a b) :
  2 * (a / 12 + b / 9) = 9 :=
  sorry

end smaller_rectangle_perimeter_l72_72240


namespace divisibility_by_29_and_29pow4_l72_72063

theorem divisibility_by_29_and_29pow4 (x y z : ℤ) (h : 29 ∣ (x^4 + y^4 + z^4)) : 29^4 ∣ (x^4 + y^4 + z^4) :=
by
  sorry

end divisibility_by_29_and_29pow4_l72_72063


namespace max_product_l72_72437

-- Problem statement: Define the conditions and the conclusion
theorem max_product (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 4) : mn ≤ 4 :=
by
  sorry -- Proof placeholder

end max_product_l72_72437


namespace quadratic_inequality_has_real_solution_l72_72052

-- Define the quadratic function and the inequality
def quadratic (a x : ℝ) : ℝ := x^2 - 8 * x + a
def quadratic_inequality (a : ℝ) : Prop := ∃ x : ℝ, quadratic a x < 0

-- Define the condition for 'a' within the interval (0, 16)
def condition_on_a (a : ℝ) : Prop := 0 < a ∧ a < 16

-- The main statement to prove
theorem quadratic_inequality_has_real_solution (a : ℝ) (h : condition_on_a a) : quadratic_inequality a :=
sorry

end quadratic_inequality_has_real_solution_l72_72052


namespace min_air_routes_l72_72544

theorem min_air_routes (a b c : ℕ) (h1 : a + b ≥ 14) (h2 : b + c ≥ 14) (h3 : c + a ≥ 14) : 
  a + b + c ≥ 21 :=
sorry

end min_air_routes_l72_72544


namespace consecutive_arithmetic_sequence_l72_72500

theorem consecutive_arithmetic_sequence (a b c : ℝ) 
  (h : (2 * b - a)^2 + (2 * b - c)^2 = 2 * (2 * b^2 - a * c)) : 
  2 * b = a + c :=
by
  sorry

end consecutive_arithmetic_sequence_l72_72500


namespace hausdorff_dimension_union_sup_l72_72204

open Set

noncomputable def Hausdorff_dimension (A : Set ℝ) : ℝ :=
sorry -- Definition for Hausdorff dimension is nontrivial and can be added here

theorem hausdorff_dimension_union_sup {A : ℕ → Set ℝ} :
  Hausdorff_dimension (⋃ i, A i) = ⨆ i, Hausdorff_dimension (A i) :=
sorry

end hausdorff_dimension_union_sup_l72_72204


namespace sqrt_function_of_x_l72_72085

theorem sqrt_function_of_x (x : ℝ) (h : x > 0) : ∃! y : ℝ, y = Real.sqrt x :=
by
  sorry

end sqrt_function_of_x_l72_72085


namespace M_intersection_N_equals_M_l72_72762

variable (x a : ℝ)

def M : Set ℝ := { y | ∃ x, y = x^2 + 1 }
def N : Set ℝ := { y | ∃ a, y = 2 * a^2 - 4 * a + 1 }

theorem M_intersection_N_equals_M : M ∩ N = M := by
  sorry

end M_intersection_N_equals_M_l72_72762


namespace number_of_rice_packets_l72_72041

theorem number_of_rice_packets
  (initial_balance : ℤ) 
  (price_per_rice_packet : ℤ)
  (num_wheat_flour_packets : ℤ) 
  (price_per_wheat_flour_packet : ℤ)
  (price_soda : ℤ) 
  (remaining_balance : ℤ)
  (spent : ℤ)
  (eqn : initial_balance - (price_per_rice_packet * 2 + num_wheat_flour_packets * price_per_wheat_flour_packet + price_soda) = remaining_balance) :
  price_per_rice_packet * 2 + num_wheat_flour_packets * price_per_wheat_flour_packet + price_soda = spent 
    → initial_balance - spent = remaining_balance
    → 2 = 2 :=
by 
  sorry

end number_of_rice_packets_l72_72041


namespace polyhedron_with_n_edges_l72_72315

noncomputable def construct_polyhedron_with_n_edges (n : ℤ) : Prop :=
  ∃ (k : ℤ) (m : ℤ), (k = 8 ∨ k = 9 ∨ k = 10) ∧ (n = k + 3 * m)

theorem polyhedron_with_n_edges (n : ℤ) (h : n ≥ 8) : 
  construct_polyhedron_with_n_edges n :=
sorry

end polyhedron_with_n_edges_l72_72315


namespace average_hamburgers_per_day_l72_72133

theorem average_hamburgers_per_day (total_hamburgers : ℕ) (days_in_week : ℕ) (h₁ : total_hamburgers = 63) (h₂ : days_in_week = 7) :
  total_hamburgers / days_in_week = 9 := by
  sorry

end average_hamburgers_per_day_l72_72133


namespace paul_initial_savings_l72_72965

theorem paul_initial_savings (additional_allowance: ℕ) (cost_per_toy: ℕ) (number_of_toys: ℕ) (total_savings: ℕ) :
  additional_allowance = 7 →
  cost_per_toy = 5 →
  number_of_toys = 2 →
  total_savings + additional_allowance = cost_per_toy * number_of_toys →
  total_savings = 3 :=
by
  intros h_additional h_cost h_number h_total
  sorry

end paul_initial_savings_l72_72965


namespace maximum_area_of_garden_l72_72028

noncomputable def max_area (perimeter : ℕ) : ℕ :=
  let half_perimeter := perimeter / 2
  let x := half_perimeter / 2
  x * x

theorem maximum_area_of_garden :
  max_area 148 = 1369 :=
by
  sorry

end maximum_area_of_garden_l72_72028


namespace roll_probability_l72_72445

noncomputable def probability_allison_rolls_greater : ℚ :=
  let p_brian := 5 / 6  -- Probability of Brian rolling 5 or lower
  let p_noah := 1       -- Probability of Noah rolling 5 or lower (since all faces roll 5 or lower)
  p_brian * p_noah

theorem roll_probability :
  probability_allison_rolls_greater = 5 / 6 := by
  sorry

end roll_probability_l72_72445


namespace certain_event_drawing_triangle_interior_angles_equal_180_deg_l72_72331

-- Define a triangle in the Euclidean space
structure Triangle (α : Type) [plane : TopologicalSpace α] :=
(a b c : α)

-- Define the sum of the interior angles of a triangle
noncomputable def sum_of_interior_angles {α : Type} [TopologicalSpace α] (T : Triangle α) : ℝ :=
180

-- The proof statement
theorem certain_event_drawing_triangle_interior_angles_equal_180_deg {α : Type} [TopologicalSpace α]
(T : Triangle α) : 
(sum_of_interior_angles T = 180) :=
sorry

end certain_event_drawing_triangle_interior_angles_equal_180_deg_l72_72331


namespace new_average_weight_l72_72840

theorem new_average_weight (original_players : ℕ) (new_players : ℕ) 
  (average_weight_original : ℝ) (weight_new_player1 : ℝ) (weight_new_player2 : ℝ) : 
  original_players = 7 → 
  new_players = 2 →
  average_weight_original = 76 → 
  weight_new_player1 = 110 → 
  weight_new_player2 = 60 → 
  (original_players * average_weight_original + weight_new_player1 + weight_new_player2) / (original_players + new_players) = 78 :=
by 
  intros h1 h2 h3 h4 h5;
  sorry

end new_average_weight_l72_72840


namespace max_value_expr_l72_72179

open Real

noncomputable def expr (x : ℝ) : ℝ :=
  (x^4 + 3 * x^2 - sqrt (x^8 + 9)) / x^2

theorem max_value_expr : ∀ (x y : ℝ), (0 < x) → (y = x + 1 / x) → expr x = 15 / 7 :=
by
  intros x y hx hy
  sorry

end max_value_expr_l72_72179


namespace fifth_number_in_eighth_row_l72_72333

theorem fifth_number_in_eighth_row : 
  (∀ n : ℕ, ∃ k : ℕ, k = n * n ∧ 
    ∀ m : ℕ, 1 ≤ m ∧ m ≤ n → 
      k - (n - m) = 54 → m = 5 ∧ n = 8) := by sorry

end fifth_number_in_eighth_row_l72_72333


namespace new_number_shifting_digits_l72_72464

-- Definitions for the three digits
variables (h t u : ℕ)

-- The original three-digit number
def original_number : ℕ := 100 * h + 10 * t + u

-- The new number formed by placing the digits "12" after the three-digit number
def new_number : ℕ := original_number h t u * 100 + 12

-- The goal is to prove that this new number equals 10000h + 1000t + 100u + 12
theorem new_number_shifting_digits (h t u : ℕ) :
  new_number h t u = 10000 * h + 1000 * t + 100 * u + 12 := 
by
  sorry -- Proof to be filled in

end new_number_shifting_digits_l72_72464


namespace expectation_of_xi_l72_72594

noncomputable def compute_expectation : ℝ := 
  let m : ℝ := 0.3
  let E : ℝ := (1 * 0.5) + (3 * m) + (5 * 0.2)
  E

theorem expectation_of_xi :
  let m: ℝ := 1 - 0.5 - 0.2 
  (0.5 + m + 0.2 = 1) → compute_expectation = 2.4 := 
by
  sorry

end expectation_of_xi_l72_72594


namespace cos2_alpha_plus_2sin2_alpha_l72_72581

theorem cos2_alpha_plus_2sin2_alpha {α : ℝ} (h : Real.tan α = 3 / 4) : 
    Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 := 
by 
  sorry

end cos2_alpha_plus_2sin2_alpha_l72_72581


namespace inner_circle_radius_l72_72082

theorem inner_circle_radius :
  ∃ (r : ℝ) (a b c d : ℕ), 
    (r = (-78 + 70 * Real.sqrt 3) / 26) ∧ 
    (a = 78) ∧ 
    (b = 70) ∧ 
    (c = 3) ∧ 
    (d = 26) ∧ 
    (Nat.gcd a d = 1) ∧ 
    (a + b + c + d = 177) := 
sorry

end inner_circle_radius_l72_72082


namespace arithmetic_sequence_sum_l72_72846

variable (S : ℕ → ℕ)   -- S is a function that gives the sum of the first k*n terms

theorem arithmetic_sequence_sum
  (n : ℕ)
  (h1 : S n = 45)
  (h2 : S (2 * n) = 60) :
  S (3 * n) = 65 := sorry

end arithmetic_sequence_sum_l72_72846


namespace machine_A_production_l72_72231

-- Definitions based on the conditions
def machine_production (A B: ℝ) (TA TB: ℝ) : Prop :=
  B = 1.10 * A ∧
  TA = TB + 10 ∧
  A * TA = 660 ∧
  B * TB = 660

-- The main statement to be proved: Machine A produces 6 sprockets per hour.
theorem machine_A_production (A B: ℝ) (TA TB: ℝ) 
  (h : machine_production A B TA TB) : 
  A = 6 := 
by sorry

end machine_A_production_l72_72231


namespace daily_construction_areas_minimum_area_A_must_build_l72_72453

-- Definitions based on conditions and questions
variable {area : ℕ}
variable {daily_A : ℕ}
variable {daily_B : ℕ}
variable (h_area : area = 5100)
variable (h_A_B_diff : daily_A = daily_B + 2)
variable (h_A_days : 900 / daily_A = 720 / daily_B)

-- Proof statements for the questions in the problem
theorem daily_construction_areas (daily_B : ℕ) (daily_A : ℕ) :
  daily_B = 8 ∧ daily_A = 10 :=
by sorry

theorem minimum_area_A_must_build (daily_A : ℕ) (daily_B : ℕ) (area_A : ℕ) :
  (area_A ≥ 2 * (5100 - area_A)) → (area_A ≥ 3400) :=
by sorry

end daily_construction_areas_minimum_area_A_must_build_l72_72453


namespace inf_arith_seq_contains_inf_geo_seq_l72_72971

-- Condition: Infinite arithmetic sequence of natural numbers
variable (a d : ℕ) (h : ∀ n : ℕ, n ≥ 1 → ∃ k : ℕ, k = a + (n - 1) * d)

-- Theorem: There exists an infinite geometric sequence within the arithmetic sequence
theorem inf_arith_seq_contains_inf_geo_seq :
  ∃ r : ℕ, ∀ n : ℕ, ∃ k : ℕ, k = a * r ^ (n - 1) := sorry

end inf_arith_seq_contains_inf_geo_seq_l72_72971


namespace jovana_added_23_pounds_l72_72092

def initial_weight : ℕ := 5
def final_weight : ℕ := 28

def added_weight : ℕ := final_weight - initial_weight

theorem jovana_added_23_pounds : added_weight = 23 := 
by sorry

end jovana_added_23_pounds_l72_72092


namespace circle_standard_equation_l72_72902

noncomputable def circle_through_ellipse_vertices : Prop :=
  ∃ (a : ℝ) (r : ℝ), a < 0 ∧
    (∀ (x y : ℝ),   -- vertices of the ellipse
      ((x = 4 ∧ y = 0) ∨ (x = 0 ∧ (y = 2 ∨ y = -2)))
      → (x + a)^2 + y^2 = r^2) ∧
    ( a = -3/2 ∧ r = 5/2 ∧ 
      ∀ (x y : ℝ), (x + 3/2)^2 + y^2 = (5/2)^2
    )

theorem circle_standard_equation :
  circle_through_ellipse_vertices :=
sorry

end circle_standard_equation_l72_72902


namespace proof_simplify_expression_l72_72645

noncomputable def simplify_expression (a b : ℝ) : ℝ :=
  (a / b + b / a)^2 - 1 / (a^2 * b^2)

theorem proof_simplify_expression 
  (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^3 + b^3 = a + b) :
  simplify_expression a b = 2 / (a * b) := by
  sorry

end proof_simplify_expression_l72_72645


namespace goldfish_growth_solution_l72_72958

def goldfish_growth_problem : Prop :=
  ∃ n : ℕ, 
    (∀ k, (k < n → 3 * (5:ℕ)^k ≠ 243 * (3:ℕ)^k)) ∧
    3 * (5:ℕ)^n = 243 * (3:ℕ)^n

theorem goldfish_growth_solution : goldfish_growth_problem :=
sorry

end goldfish_growth_solution_l72_72958


namespace complex_distance_l72_72447

theorem complex_distance (i : Complex) (h : i = Complex.I) :
  Complex.abs (3 / (2 - i)^2) = 3 / 5 := 
by
  sorry

end complex_distance_l72_72447


namespace max_unique_sums_l72_72095

-- Define the coin values in cents
def penny := 1
def nickel := 5
def quarter := 25
def half_dollar := 50

-- Define the set of all coins and their counts
structure Coins :=
  (pennies : ℕ := 3)
  (nickels : ℕ := 3)
  (quarters : ℕ := 1)
  (half_dollars : ℕ := 2)

-- Define the list of all possible pairs and their sums
def possible_sums : Finset ℕ :=
  { 2, 6, 10, 26, 30, 51, 55, 75, 100 }

-- Prove that the count of unique sums is 9
theorem max_unique_sums (c : Coins) : c.pennies = 3 → c.nickels = 3 → c.quarters = 1 → c.half_dollars = 2 →
  possible_sums.card = 9 := 
by
  intros
  sorry

end max_unique_sums_l72_72095


namespace tangent_slope_at_one_l72_72763

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem tangent_slope_at_one : deriv f 1 = 2 * Real.exp 1 := sorry

end tangent_slope_at_one_l72_72763


namespace min_students_wearing_both_l72_72818

theorem min_students_wearing_both (n : ℕ) (H1 : n % 3 = 0) (H2 : n % 6 = 0) (H3 : n = 6) :
  ∃ x : ℕ, x = 1 ∧ 
           (∃ b : ℕ, b = n / 3) ∧
           (∃ r : ℕ, r = 5 * n / 6) ∧
           6 = b + r - x :=
by sorry

end min_students_wearing_both_l72_72818


namespace least_5_digit_number_divisible_by_15_25_40_75_125_140_l72_72638

theorem least_5_digit_number_divisible_by_15_25_40_75_125_140 : 
  ∃ n : ℕ, (10000 ≤ n) ∧ (n < 100000) ∧ 
  (15 ∣ n) ∧ (25 ∣ n) ∧ (40 ∣ n) ∧ (75 ∣ n) ∧ (125 ∣ n) ∧ (140 ∣ n) ∧ (n = 21000) :=
by
  sorry

end least_5_digit_number_divisible_by_15_25_40_75_125_140_l72_72638


namespace infinite_solutions_iff_l72_72096

theorem infinite_solutions_iff (a b c d : ℤ) :
  (∃ᶠ x in at_top, ∃ᶠ y in at_top, x^2 + a * x + b = y^2 + c * y + d) ↔ (a^2 - 4 * b = c^2 - 4 * d) :=
by sorry

end infinite_solutions_iff_l72_72096


namespace min_a_b_l72_72633

theorem min_a_b (a b : ℕ) (h1 : 43 * a + 17 * b = 731) (h2 : a ≤ 17) (h3 : b ≤ 43) : a + b = 17 :=
by
  sorry

end min_a_b_l72_72633


namespace total_cost_of_dresses_l72_72815

-- Define the costs of each dress
variables (patty_cost ida_cost jean_cost pauline_cost total_cost : ℕ)

-- Given conditions
axiom pauline_cost_is_30 : pauline_cost = 30
axiom jean_cost_is_10_less_than_pauline : jean_cost = pauline_cost - 10
axiom ida_cost_is_30_more_than_jean : ida_cost = jean_cost + 30
axiom patty_cost_is_10_more_than_ida : patty_cost = ida_cost + 10

-- Statement to prove total cost
theorem total_cost_of_dresses : total_cost = pauline_cost + jean_cost + ida_cost + patty_cost 
                                 → total_cost = 160 :=
by {
  -- Proof is left as an exercise
  sorry
}

end total_cost_of_dresses_l72_72815


namespace smallest_four_digit_number_l72_72309

theorem smallest_four_digit_number (N : ℕ) (a b : ℕ) (h1 : N = 100 * a + b) (h2 : N = (a + b)^2) (h3 : 1000 ≤ N) (h4 : N < 10000) : N = 2025 :=
sorry

end smallest_four_digit_number_l72_72309


namespace stock_price_is_108_l72_72148

noncomputable def dividend_income (FV : ℕ) (D : ℕ) : ℕ :=
  FV * D / 100

noncomputable def face_value_of_stock (I : ℕ) (D : ℕ) : ℕ :=
  I * 100 / D

noncomputable def price_of_stock (Inv : ℕ) (FV : ℕ) : ℕ :=
  Inv * 100 / FV

theorem stock_price_is_108 (I D Inv : ℕ) (hI : I = 450) (hD : D = 10) (hInv : Inv = 4860) :
  price_of_stock Inv (face_value_of_stock I D) = 108 :=
by
  -- Placeholder for proof
  sorry

end stock_price_is_108_l72_72148


namespace inequality_system_solution_l72_72136

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_system_solution_l72_72136


namespace tetrahedron_ratio_l72_72356

open Real

theorem tetrahedron_ratio (a b : ℝ) (h1 : a = PA ∧ PB = a) (h2 : PC = b ∧ AB = b ∧ BC = b ∧ CA = b) (h3 : a < b) :
  (sqrt 6 - sqrt 2) / 2 < a / b ∧ a / b < 1 :=
by
  sorry

end tetrahedron_ratio_l72_72356


namespace star_polygon_points_eq_24_l72_72338

theorem star_polygon_points_eq_24 (n : ℕ) 
  (A_i B_i : ℕ → ℝ) 
  (h_congruent_A : ∀ i j, A_i i = A_i j) 
  (h_congruent_B : ∀ i j, B_i i = B_i j) 
  (h_angle_difference : ∀ i, A_i i = B_i i - 15) : 
  n = 24 := 
sorry

end star_polygon_points_eq_24_l72_72338


namespace average_weight_l72_72920

theorem average_weight {w : ℝ} 
  (h1 : 62 < w) 
  (h2 : w < 72) 
  (h3 : 60 < w) 
  (h4 : w < 70) 
  (h5 : w ≤ 65) : w = 63.5 :=
by
  sorry

end average_weight_l72_72920


namespace point_in_second_quadrant_l72_72169

-- Define the point in question
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Given conditions based on the problem statement
def P (x : ℝ) : Point :=
  Point.mk (-2) (x^2 + 1)

-- The theorem we aim to prove
theorem point_in_second_quadrant (x : ℝ) : (P x).x < 0 ∧ (P x).y > 0 → 
  -- This condition means that the point is in the second quadrant
  (P x).x < 0 ∧ (P x).y > 0 :=
by
  sorry

end point_in_second_quadrant_l72_72169


namespace ice_cream_scoops_l72_72773

theorem ice_cream_scoops (total_money : ℝ) (spent_on_restaurant : ℝ) (remaining_money : ℝ) 
  (cost_per_scoop_after_discount : ℝ) (remaining_each : ℝ) 
  (initial_savings : ℝ) (service_charge_percent : ℝ) (restaurant_percent : ℝ) 
  (ice_cream_discount_percent : ℝ) (money_each : ℝ) :
  total_money = 400 ∧
  spent_on_restaurant = 320 ∧
  remaining_money = 80 ∧
  cost_per_scoop_after_discount = 5 ∧
  remaining_each = 8 ∧
  initial_savings = 200 ∧
  service_charge_percent = 0.20 ∧
  restaurant_percent = 0.80 ∧
  ice_cream_discount_percent = 0.10 ∧
  money_each = 5 → 
  ∃ (scoops_per_person : ℕ), scoops_per_person = 5 :=
by
  sorry

end ice_cream_scoops_l72_72773


namespace minimum_value_sum_l72_72202

theorem minimum_value_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a / (3 * b) + b / (5 * c) + c / (6 * a)) >= (3 / (90^(1/3))) :=
by 
  sorry

end minimum_value_sum_l72_72202


namespace total_weight_on_scale_l72_72926

-- Define the weights of Alexa and Katerina
def alexa_weight : ℕ := 46
def katerina_weight : ℕ := 49

-- State the theorem to prove the total weight on the scale
theorem total_weight_on_scale : alexa_weight + katerina_weight = 95 := by
  sorry

end total_weight_on_scale_l72_72926


namespace milkshakes_more_than_ice_cream_cones_l72_72805

def ice_cream_cones_sold : ℕ := 67
def milkshakes_sold : ℕ := 82

theorem milkshakes_more_than_ice_cream_cones : milkshakes_sold - ice_cream_cones_sold = 15 := by
  sorry

end milkshakes_more_than_ice_cream_cones_l72_72805


namespace probability_of_sum_20_is_correct_l72_72630

noncomputable def probability_sum_20 : ℚ :=
  let total_outcomes := 12 * 12
  let favorable_outcomes := 5
  favorable_outcomes / total_outcomes

theorem probability_of_sum_20_is_correct :
  probability_sum_20 = 5 / 144 :=
by
  sorry

end probability_of_sum_20_is_correct_l72_72630


namespace repaired_shoes_last_time_l72_72507

theorem repaired_shoes_last_time :
  let cost_of_repair := 13.50
  let cost_of_new := 32.00
  let duration_of_new := 2.0
  let surcharge := 0.1852
  let avg_cost_new := cost_of_new / duration_of_new
  let avg_cost_repair (T : ℝ) := cost_of_repair / T
  (avg_cost_new = (1 + surcharge) * avg_cost_repair 1) ↔ T = 1 := 
by
  sorry

end repaired_shoes_last_time_l72_72507


namespace find_second_number_l72_72210

theorem find_second_number (a b c : ℝ) (h1 : a + b + c = 3.622) (h2 : a = 3.15) (h3 : c = 0.458) : b = 0.014 :=
sorry

end find_second_number_l72_72210


namespace alicia_masks_left_l72_72621

theorem alicia_masks_left (T G L : ℕ) (hT : T = 90) (hG : G = 51) (hL : L = T - G) : L = 39 :=
by
  rw [hT, hG] at hL
  exact hL

end alicia_masks_left_l72_72621


namespace ratio_a_b_equals_sqrt2_l72_72961

variable (A B C a b c : ℝ) -- Define the variables representing the angles and sides.

-- Assuming the sides a, b, c are positive and a triangle is formed (non-degenerate)
axiom triangle_ABC : 0 < a ∧ 0 < b ∧ 0 < c

-- Assuming the sum of the angles in a triangle equals 180 degrees (π radians)
axiom sum_angles_triangle : A + B + C = Real.pi

-- Given condition
axiom given_condition : b * Real.cos C + c * Real.cos B = Real.sqrt 2 * b

-- Problem statement to be proven
theorem ratio_a_b_equals_sqrt2 : (a / b) = Real.sqrt 2 :=
by
  -- Assume the problem statement is correct
  sorry

end ratio_a_b_equals_sqrt2_l72_72961


namespace times_reaching_35m_l72_72303

noncomputable def projectile_height (t : ℝ) : ℝ :=
  -4.9 * t^2 + 30 * t

theorem times_reaching_35m :
  ∃ t1 t2 : ℝ, (abs (t1 - 1.57) < 0.01 ∧ abs (t2 - 4.55) < 0.01) ∧
               projectile_height t1 = 35 ∧ projectile_height t2 = 35 :=
by
  sorry

end times_reaching_35m_l72_72303


namespace average_price_of_remaining_cans_l72_72298

theorem average_price_of_remaining_cans (price_all price_returned : ℕ) (average_all average_returned : ℚ) 
    (h1 : price_all = 6) (h2 : average_all = 36.5) (h3 : price_returned = 2) (h4 : average_returned = 49.5) : 
    (price_all - price_returned) ≠ 0 → 
    4 * 30 = 6 * 36.5 - 2 * 49.5 :=
by
  intros hne
  sorry

end average_price_of_remaining_cans_l72_72298


namespace fenced_area_with_cutout_l72_72009

def rectangle_area (length width : ℝ) : ℝ := length * width

def square_area (side : ℝ) : ℝ := side * side

theorem fenced_area_with_cutout :
  rectangle_area 20 18 - square_area 4 = 344 :=
by
  -- This is where the proof would go, but it is omitted as per instructions.
  sorry

end fenced_area_with_cutout_l72_72009


namespace sum_of_roots_eq_neg_five_l72_72223

theorem sum_of_roots_eq_neg_five (x₁ x₂ : ℝ) (h₁ : x₁^2 + 5 * x₁ - 2 = 0) (h₂ : x₂^2 + 5 * x₂ - 2 = 0) (h_distinct : x₁ ≠ x₂) :
  x₁ + x₂ = -5 := sorry

end sum_of_roots_eq_neg_five_l72_72223


namespace total_time_for_5_smoothies_l72_72174

-- Definitions for the conditions
def freeze_time : ℕ := 40
def blend_time_per_smoothie : ℕ := 3
def chop_time_apples_per_smoothie : ℕ := 2
def chop_time_bananas_per_smoothie : ℕ := 3
def chop_time_strawberries_per_smoothie : ℕ := 4
def chop_time_mangoes_per_smoothie : ℕ := 5
def chop_time_pineapples_per_smoothie : ℕ := 6
def number_of_smoothies : ℕ := 5

-- Total chopping time per smoothie
def chop_time_per_smoothie : ℕ := chop_time_apples_per_smoothie + 
                                  chop_time_bananas_per_smoothie + 
                                  chop_time_strawberries_per_smoothie + 
                                  chop_time_mangoes_per_smoothie + 
                                  chop_time_pineapples_per_smoothie

-- Total chopping time for 5 smoothies
def total_chop_time : ℕ := chop_time_per_smoothie * number_of_smoothies

-- Total blending time for 5 smoothies
def total_blend_time : ℕ := blend_time_per_smoothie * number_of_smoothies

-- Total time to make 5 smoothies
def total_time : ℕ := total_chop_time + total_blend_time

-- Theorem statement
theorem total_time_for_5_smoothies : total_time = 115 := by
  sorry

end total_time_for_5_smoothies_l72_72174


namespace domain_of_sqrt_and_fraction_l72_72870

def domain_of_function (x : ℝ) : Prop :=
  2 * x - 3 ≥ 0 ∧ x ≠ 3

theorem domain_of_sqrt_and_fraction :
  {x : ℝ | domain_of_function x} = {x : ℝ | x ≥ 3 / 2} \ {3} :=
by sorry

end domain_of_sqrt_and_fraction_l72_72870


namespace gcd_143_117_l72_72788

theorem gcd_143_117 : Nat.gcd 143 117 = 13 :=
by
  have h1 : 143 = 11 * 13 := by rfl
  have h2 : 117 = 9 * 13 := by rfl
  sorry

end gcd_143_117_l72_72788


namespace squares_on_sides_of_triangle_l72_72311

theorem squares_on_sides_of_triangle (A B C : ℕ) (hA : A = 3^2) (hB : B = 4^2) (hC : C = 5^2) : 
  A + B = C :=
by 
  rw [hA, hB, hC] 
  exact Nat.add_comm 9 16 ▸ rfl

end squares_on_sides_of_triangle_l72_72311


namespace find_a_if_circle_l72_72727

noncomputable def curve_eq (a x y : ℝ) : ℝ :=
  a^2 * x^2 + (a + 2) * y^2 + 2 * a * x + a

def is_circle_condition (a : ℝ) : Prop :=
  ∀ x y : ℝ, curve_eq a x y = 0 → (∃ k : ℝ, curve_eq a x y = k * (x^2 + y^2))

theorem find_a_if_circle :
  (∀ a : ℝ, is_circle_condition a → a = -1) :=
by
  sorry

end find_a_if_circle_l72_72727


namespace length_of_ON_l72_72651

noncomputable def proof_problem : Prop :=
  let hyperbola := { x : ℝ × ℝ | x.1 ^ 2 - x.2 ^ 2 = 1 }
  ∃ (F1 F2 P : ℝ × ℝ) (O : ℝ × ℝ) (N : ℝ × ℝ),
    O = (0, 0) ∧
    P ∈ hyperbola ∧
    N = ((P.1 + F1.1) / 2, (P.2 + F1.2) / 2) ∧
    dist P F1 = 5 ∧
    ∃ r : ℝ, r = 1.5 ∧ (dist O N = r)

theorem length_of_ON : proof_problem :=
sorry

end length_of_ON_l72_72651


namespace max_value_of_f_in_interval_l72_72168

noncomputable def f (x m : ℝ) : ℝ := -x^3 + 3 * x^2 + m

theorem max_value_of_f_in_interval (m : ℝ) (h₁ : ∀ x ∈ [-2, 2], - x^3 + 3 * x^2 + m ≥ 1) : 
  ∃ x ∈ [-2, 2], f x m = 21 :=
by
  sorry

end max_value_of_f_in_interval_l72_72168


namespace units_digit_sum_factorials_500_l72_72429

-- Define the unit digit computation function
def unit_digit (n : ℕ) : ℕ := n % 10

-- Define the factorial function
def fact : ℕ → ℕ
| 0 => 1
| (n+1) => (n+1) * fact n

-- Define the sum of factorials from 1 to n
def sum_factorials (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).sum (λ i => fact i)

-- Define the problem statement
theorem units_digit_sum_factorials_500 : unit_digit (sum_factorials 500) = 3 :=
sorry

end units_digit_sum_factorials_500_l72_72429


namespace no_largest_integer_exists_l72_72318

/--
  Define a predicate to check whether an integer is a non-square.
-/
def is_non_square (n : ℕ) : Prop :=
  ¬ ∃ m : ℕ, m * m = n

/--
  Define the main theorem which states that there is no largest positive integer
  that cannot be expressed as the sum of a positive integral multiple of 36
  and a positive non-square integer less than 36.
-/
theorem no_largest_integer_exists : ¬ ∃ (n : ℕ), 
  ∀ (a : ℕ) (b : ℕ), a > 0 ∧ b > 0 ∧ b < 36 ∧ is_non_square b →
  n ≠ 36 * a + b :=
sorry

end no_largest_integer_exists_l72_72318


namespace probability_point_in_cube_l72_72348

noncomputable def volume_cube (s : ℝ) : ℝ := s ^ 3

noncomputable def radius_sphere (d : ℝ) : ℝ := d / 2

noncomputable def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem probability_point_in_cube :
  let s := 1 -- side length of the cube
  let v_cube := volume_cube s
  let d := Real.sqrt 3 -- diagonal of the cube
  let r := radius_sphere d
  let v_sphere := volume_sphere r
  v_cube / v_sphere = (2 * Real.sqrt 3) / (3 * Real.pi) :=
by
  sorry

end probability_point_in_cube_l72_72348


namespace find_x_l72_72163

theorem find_x
  (PQR_straight : ∀ x y : ℝ, x + y = 76 → 3 * x + 2 * y = 180)
  (h : x + y = 76) :
  x = 28 :=
by
  sorry

end find_x_l72_72163


namespace geom_prog_common_ratio_unique_l72_72904

theorem geom_prog_common_ratio_unique (b q : ℝ) (hb : b > 0) (hq : q > 1) :
  (∃ b : ℝ, (q = (1 + Real.sqrt 5) / 2) ∧ 
    (0 < b ∧ b * q ≠ b ∧ b * q^2 ≠ b ∧ b * q^3 ≠ b) ∧ 
    ((2 * b * q = b + b * q^2) ∨ (2 * b * q = b + b * q^3) ∨ (2 * b * q^2 = b + b * q^3))) := 
sorry

end geom_prog_common_ratio_unique_l72_72904


namespace circle_equation_tangent_line1_tangent_line2_l72_72488

-- Definitions of points
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (2, 0)
def P : ℝ × ℝ := (2, 3)

-- Equation for the circle given the point constraints
def circle_eq : Prop := 
  ∀ x y : ℝ, ((x - 1)^2 + y^2 = 1) ↔ ((x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 0))

-- Equations for the tangent lines passing through point P and tangent to the circle
def tangent_eq1 : Prop := 
  P.1 = 2

def tangent_eq2 : Prop :=
  4 * P.1 - 3 * P.2 + 1 = 0

-- Statements to be proven
theorem circle_equation : circle_eq := 
  sorry 

theorem tangent_line1 : tangent_eq1 := 
  sorry 

theorem tangent_line2 : tangent_eq2 := 
  sorry 

end circle_equation_tangent_line1_tangent_line2_l72_72488


namespace coolant_left_l72_72378

theorem coolant_left (initial_volume : ℝ) (initial_concentration : ℝ) (x : ℝ) (replacement_concentration : ℝ) (final_concentration : ℝ) :
  initial_volume = 19 ∧ 
  initial_concentration = 0.30 ∧ 
  replacement_concentration = 0.80 ∧ 
  final_concentration = 0.50 ∧ 
  (0.30 * initial_volume - 0.30 * x + 0.80 * x = 0.50 * initial_volume) →
  initial_volume - x = 11.4 :=
by sorry

end coolant_left_l72_72378


namespace shoes_produced_min_pairs_for_profit_l72_72882

-- given conditions
def production_cost (n : ℕ) : ℕ := 4000 + 50 * n

-- Question (1)
theorem shoes_produced (C : ℕ) (h : C = 36000) : ∃ n : ℕ, production_cost n = C :=
by sorry

-- given conditions for part (2)
def selling_price (price_per_pair : ℕ) (n : ℕ) : ℕ := price_per_pair * n
def profit (price_per_pair : ℕ) (n : ℕ) : ℕ := selling_price price_per_pair n - production_cost n

-- Question (2)
theorem min_pairs_for_profit (price_per_pair profit_goal : ℕ) (h : price_per_pair = 90) (h1 : profit_goal = 8500) :
  ∃ n : ℕ, profit price_per_pair n ≥ profit_goal :=
by sorry

end shoes_produced_min_pairs_for_profit_l72_72882


namespace positive_integer_pairs_l72_72914

theorem positive_integer_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  a^b = b^(a^2) ↔ (a = 1 ∧ b = 1) ∨ (a = 2 ∧ b = 16) ∨ (a = 3 ∧ b = 27) :=
by sorry

end positive_integer_pairs_l72_72914


namespace correct_reflection_l72_72455

section
variable {Point : Type}
variables (PQ : Point → Point → Prop) (shaded_figure : Point → Prop)
variables (A B C D E : Point → Prop)

-- Condition: The line segment PQ is the axis of reflection.
-- Condition: The shaded figure is positioned above the line PQ and touches it at two points.
-- Define the reflection operation (assuming definitions for points and reflections are given).

def reflected (fig : Point → Prop) (axis : Point → Point → Prop) : Point → Prop := sorry  -- Define properly

-- The correct answer: The reflected figure should match figure (A).
theorem correct_reflection :
  reflected shaded_figure PQ = A :=
sorry
end

end correct_reflection_l72_72455


namespace value_of_x_l72_72184

theorem value_of_x (x : ℚ) (h : (x + 10 + 17 + 3 * x + 15 + 3 * x + 6) / 5 = 26) : x = 82 / 7 :=
by
  sorry

end value_of_x_l72_72184


namespace inverse_sum_l72_72791

def g (x : ℝ) : ℝ := x^3

theorem inverse_sum : g⁻¹ 8 + g⁻¹ (-64) = -2 :=
by
  -- proof steps will go here
  sorry

end inverse_sum_l72_72791


namespace rachel_age_is_19_l72_72614

def rachel_and_leah_ages (R L : ℕ) : Prop :=
  (R = L + 4) ∧ (R + L = 34)

theorem rachel_age_is_19 : ∃ L : ℕ, rachel_and_leah_ages 19 L :=
by {
  sorry
}

end rachel_age_is_19_l72_72614


namespace paul_number_proof_l72_72793

theorem paul_number_proof (a b : ℕ) (h₀ : 0 ≤ a ∧ a ≤ 9) (h₁ : 0 ≤ b ∧ b ≤ 9) (h₂ : a - b = 7) :
  (10 * a + b = 81) ∨ (10 * a + b = 92) :=
  sorry

end paul_number_proof_l72_72793


namespace number_of_solutions_proof_l72_72120

noncomputable def number_of_real_solutions (x y z w : ℝ) : ℝ :=
  if (x = z + w + 2 * z * w * x) ∧ (y = w + x + 2 * w * x * y) ∧ (z = x + y + 2 * x * y * z) ∧ (w = y + z + 2 * y * z * w) then
    5
  else
    0

theorem number_of_solutions_proof :
  ∃ x y z w : ℝ, x = z + w + 2 * z * w * x ∧ y = w + x + 2 * w * x * y ∧ z = x + y + 2 * x * y * z ∧ w = y + z + 2 * y * z * w → number_of_real_solutions x y z w = 5 :=
by
  sorry

end number_of_solutions_proof_l72_72120


namespace smallest_number_value_l72_72091

variable (a b c : ℕ)

def conditions (a b c : ℕ) : Prop :=
  a + b + c = 100 ∧
  c = 2 * a ∧
  c - b = 10

theorem smallest_number_value (h : conditions a b c) : a = 22 :=
by
  sorry

end smallest_number_value_l72_72091


namespace cubic_roots_cosines_l72_72005

theorem cubic_roots_cosines
  {p q r : ℝ}
  (h_eq : ∀ x : ℝ, x^3 + p * x^2 + q * x + r = 0)
  (h_roots : ∃ (α β γ : ℝ), (α > 0) ∧ (β > 0) ∧ (γ > 0) ∧ (α + β + γ = -p) ∧ 
             (α * β + β * γ + γ * α = q) ∧ (α * β * γ = -r)) :
  2 * r + 1 = p^2 - 2 * q :=
by
  sorry

end cubic_roots_cosines_l72_72005


namespace factor_expression_l72_72943

theorem factor_expression (b : ℝ) : 45 * b^2 + 135 * b^3 = 45 * b^2 * (1 + 3 * b) :=
by
  sorry

end factor_expression_l72_72943


namespace average_of_11_numbers_l72_72060

theorem average_of_11_numbers (a b c d e f g h i j k : ℕ) 
  (h₀ : (a + b + c + d + e + f) / 6 = 19)
  (h₁ : (f + g + h + i + j + k) / 6 = 27)
  (h₂ : f = 34) :
  (a + b + c + d + e + f + g + h + i + j + k) / 11 = 22 := 
by
  sorry

end average_of_11_numbers_l72_72060


namespace range_of_M_l72_72116

theorem range_of_M (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
  (h1 : x + y + z = 30) (h2 : 3 * x + y - z = 50) :
  120 ≤ 5 * x + 4 * y + 2 * z ∧ 5 * x + 4 * y + 2 * z ≤ 130 :=
by
  -- We would start the proof here by using the given constraints
  sorry

end range_of_M_l72_72116


namespace andy_older_than_rahim_l72_72515

-- Define Rahim's current age
def Rahim_current_age : ℕ := 6

-- Define Andy's age in 5 years
def Andy_age_in_5_years : ℕ := 2 * Rahim_current_age

-- Define Andy's current age
def Andy_current_age : ℕ := Andy_age_in_5_years - 5

-- Define the difference in age between Andy and Rahim right now
def age_difference : ℕ := Andy_current_age - Rahim_current_age

-- Theorem stating the age difference between Andy and Rahim right now is 1 year
theorem andy_older_than_rahim : age_difference = 1 :=
by
  -- Proof is skipped
  sorry

end andy_older_than_rahim_l72_72515
