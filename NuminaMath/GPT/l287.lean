import Mathlib

namespace train_passes_pole_in_10_seconds_l287_28733

theorem train_passes_pole_in_10_seconds :
  let L := 150 -- length of the train in meters
  let S_kmhr := 54 -- speed in kilometers per hour
  let S_ms := S_kmhr * 1000 / 3600 -- speed in meters per second
  (L / S_ms = 10) := 
by
  sorry

end train_passes_pole_in_10_seconds_l287_28733


namespace Ben_hits_7_l287_28732

def regions : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
def Alice_score : ℕ := 18
def Ben_score : ℕ := 13
def Cindy_score : ℕ := 19
def Dave_score : ℕ := 16
def Ellen_score : ℕ := 20
def Frank_score : ℕ := 5

def hit_score (name : String) (region1 region2 : ℕ) (score : ℕ) : Prop :=
  region1 ∈ regions ∧ region2 ∈ regions ∧ region1 ≠ region2 ∧ region1 + region2 = score

theorem Ben_hits_7 :
  ∃ r1 r2, hit_score "Ben" r1 r2 Ben_score ∧ (r1 = 7 ∨ r2 = 7) :=
sorry

end Ben_hits_7_l287_28732


namespace cos_double_angle_l287_28760

theorem cos_double_angle (α : ℝ) (h : Real.sin (α + Real.pi / 2) = 1 / 2) : Real.cos (2 * α) = -1 / 2 := by
  sorry

end cos_double_angle_l287_28760


namespace mean_weight_is_70_357_l287_28756

def weights_50 : List ℕ := [57]
def weights_60 : List ℕ := [60, 64, 64, 66, 69]
def weights_70 : List ℕ := [71, 73, 73, 75, 77, 78, 79, 79]

def weights := weights_50 ++ weights_60 ++ weights_70

def total_weight : ℕ := List.sum weights
def total_players : ℕ := List.length weights
def mean_weight : ℚ := (total_weight : ℚ) / total_players

theorem mean_weight_is_70_357 :
  mean_weight = 70.357 := 
sorry

end mean_weight_is_70_357_l287_28756


namespace sqrt_of_25_l287_28777

theorem sqrt_of_25 : ∃ x : ℝ, x^2 = 25 ∧ (x = 5 ∨ x = -5) :=
by {
  sorry
}

end sqrt_of_25_l287_28777


namespace find_n_l287_28746

theorem find_n (n : ℕ) (h1 : Nat.lcm n 14 = 56) (h2 : Nat.gcd n 14 = 10) : n = 40 :=
by
  sorry

end find_n_l287_28746


namespace plane_equation_l287_28703

-- Define the point and the normal vector
def point : ℝ × ℝ × ℝ := (8, -2, 2)
def normal_vector : ℝ × ℝ × ℝ := (8, -2, 2)

-- Define integers A, B, C, D such that the plane equation satisfies the conditions
def A : ℤ := 4
def B : ℤ := -1
def C : ℤ := 1
def D : ℤ := -18

-- Prove the equation of the plane
theorem plane_equation (x y z : ℝ) :
  A * x + B * y + C * z + D = 0 ↔ 4 * x - y + z - 18 = 0 :=
by
  sorry

end plane_equation_l287_28703


namespace tolu_pencils_l287_28744

theorem tolu_pencils (price_per_pencil : ℝ) (robert_pencils : ℕ) (melissa_pencils : ℕ) (total_money_spent : ℝ) (tolu_pencils : ℕ) :
  price_per_pencil = 0.20 →
  robert_pencils = 5 →
  melissa_pencils = 2 →
  total_money_spent = 2.00 →
  tolu_pencils * price_per_pencil = 2.00 - (5 * 0.20 + 2 * 0.20) →
  tolu_pencils = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end tolu_pencils_l287_28744


namespace solve_quadratic_l287_28731

theorem solve_quadratic (x : ℝ) : x^2 = x ↔ (x = 0 ∨ x = 1) :=
by
  sorry

end solve_quadratic_l287_28731


namespace minimum_rectangles_needed_l287_28757

def type1_corners := 12
def type2_corners := 12
def group_size := 3

theorem minimum_rectangles_needed (cover_type1: ℕ) (cover_type2: ℕ)
  (type1_corners coverable_by_one: ℕ) (type2_groups_num: ℕ) :
  type1_corners = 12 → type2_corners = 12 → type2_groups_num = 4 →
  group_size = 3 → cover_type1 + cover_type2 = 12 :=
by
  intros h1 h2 h3 h4 
  sorry

end minimum_rectangles_needed_l287_28757


namespace find_k_and_a_l287_28750

noncomputable def polynomial_P : Polynomial ℝ := Polynomial.C 5 + Polynomial.X * (Polynomial.C (-18) + Polynomial.X * (Polynomial.C 13 + Polynomial.X * (Polynomial.C (-4) + Polynomial.X)))
noncomputable def polynomial_D (k : ℝ) : Polynomial ℝ := Polynomial.C k + Polynomial.X * (Polynomial.C (-1) + Polynomial.X)
noncomputable def polynomial_R (a : ℝ) : Polynomial ℝ := Polynomial.C a + (Polynomial.C 2 * Polynomial.X)

theorem find_k_and_a : 
  ∃ k a : ℝ, polynomial_P = polynomial_D k * Polynomial.C 1 + polynomial_R a ∧ k = 10 ∧ a = 5 :=
sorry

end find_k_and_a_l287_28750


namespace factor_expression_l287_28739

theorem factor_expression (x : ℕ) : 75 * x + 45 = 15 * (5 * x + 3) :=
by
  sorry

end factor_expression_l287_28739


namespace calculate_value_l287_28794

def a : ℕ := 2500
def b : ℕ := 2109
def d : ℕ := 64

theorem calculate_value : (a - b) ^ 2 / d = 2389 := by
  sorry

end calculate_value_l287_28794


namespace proportion_problem_l287_28775

theorem proportion_problem 
  (x : ℝ) 
  (third_number : ℝ) 
  (h1 : 0.75 / x = third_number / 8) 
  (h2 : x = 0.6) 
  : third_number = 10 := 
by 
  sorry

end proportion_problem_l287_28775


namespace reduce_repeating_decimal_l287_28782

noncomputable def repeating_decimal_to_fraction (a : ℚ) (n : ℕ) : ℚ :=
  a + (n / 99)

theorem reduce_repeating_decimal : repeating_decimal_to_fraction 2 7 = 205 / 99 := by
  -- proof omitted
  sorry

end reduce_repeating_decimal_l287_28782


namespace avg_of_x_y_is_41_l287_28759

theorem avg_of_x_y_is_41 
  (x y : ℝ) 
  (h : (4 + 6 + 8 + x + y) / 5 = 20) 
  : (x + y) / 2 = 41 := 
by 
  sorry

end avg_of_x_y_is_41_l287_28759


namespace evaluate_expression_l287_28704

lemma pow_mod_four_cycle (n : ℕ) : (n % 4) = 1 → (i : ℂ)^n = i :=
by sorry

lemma pow_mod_four_cycle2 (n : ℕ) : (n % 4) = 2 → (i : ℂ)^n = -1 :=
by sorry

lemma pow_mod_four_cycle3 (n : ℕ) : (n % 4) = 3 → (i : ℂ)^n = -i :=
by sorry

lemma pow_mod_four_cycle4 (n : ℕ) : (n % 4) = 0 → (i : ℂ)^n = 1 :=
by sorry

theorem evaluate_expression : 
  (i : ℂ)^(2021) + (i : ℂ)^(2022) + (i : ℂ)^(2023) + (i : ℂ)^(2024) = 0 :=
by sorry

end evaluate_expression_l287_28704


namespace prod_mod7_eq_zero_l287_28788

theorem prod_mod7_eq_zero :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 := 
by {
  sorry
}

end prod_mod7_eq_zero_l287_28788


namespace factorization_of_z6_minus_64_l287_28728

theorem factorization_of_z6_minus_64 :
  ∀ (z : ℝ), (z^6 - 64) = (z - 2) * (z^2 + 2*z + 4) * (z + 2) * (z^2 - 2*z + 4) := 
by
  intros z
  sorry

end factorization_of_z6_minus_64_l287_28728


namespace definite_integral_abs_poly_l287_28766

theorem definite_integral_abs_poly :
  ∫ x in (-2 : ℝ)..(2 : ℝ), |x^2 - 2*x| = 8 :=
by
  sorry

end definite_integral_abs_poly_l287_28766


namespace number_of_girls_sampled_in_third_grade_l287_28753

-- Number of total students in the high school
def total_students : ℕ := 3000

-- Number of students in each grade
def first_grade_students : ℕ := 800
def second_grade_students : ℕ := 1000
def third_grade_students : ℕ := 1200

-- Number of boys and girls in each grade
def first_grade_boys : ℕ := 500
def first_grade_girls : ℕ := 300

def second_grade_boys : ℕ := 600
def second_grade_girls : ℕ := 400

def third_grade_boys : ℕ := 800
def third_grade_girls : ℕ := 400

-- Total number of students sampled
def total_sampled_students : ℕ := 150

-- Hypothesis: stratified sampling method according to grade proportions
theorem number_of_girls_sampled_in_third_grade :
  third_grade_girls * (total_sampled_students / total_students) = 20 :=
by
  -- We will add the proof here
  sorry

end number_of_girls_sampled_in_third_grade_l287_28753


namespace set_intersection_l287_28717

noncomputable def A : Set ℝ := { x | x / (x - 1) < 0 }
noncomputable def B : Set ℝ := { x | 0 < x ∧ x < 3 }
noncomputable def expected_intersection : Set ℝ := { x | 0 < x ∧ x < 1 }

theorem set_intersection (x : ℝ) : (x ∈ A ∧ x ∈ B) ↔ x ∈ expected_intersection :=
by
  sorry

end set_intersection_l287_28717


namespace inequality_proof_l287_28741

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 + 4 * a / (b + c)) * (1 + 4 * b / (a + c)) * (1 + 4 * c / (a + b)) > 25 :=
sorry

end inequality_proof_l287_28741


namespace john_minimum_pizzas_l287_28783

theorem john_minimum_pizzas (car_cost bag_cost earnings_per_pizza gas_cost p : ℕ) 
  (h_car : car_cost = 6000)
  (h_bag : bag_cost = 200)
  (h_earnings : earnings_per_pizza = 12)
  (h_gas : gas_cost = 4)
  (h_p : 8 * p >= car_cost + bag_cost) : p >= 775 := 
sorry

end john_minimum_pizzas_l287_28783


namespace min_value_xy_expression_l287_28789

theorem min_value_xy_expression (x y : ℝ) : ∃ c : ℝ, (∀ x y : ℝ, (xy - 1)^2 + (x + y)^2 ≥ c) ∧ c = 1 :=
by {
  -- Placeholder for proof
  sorry
}

end min_value_xy_expression_l287_28789


namespace chimes_in_a_day_l287_28730

-- Definitions for the conditions
def strikes_in_12_hours : ℕ :=
  (1 + 12) * 12 / 2

def strikes_in_24_hours : ℕ :=
  2 * strikes_in_12_hours

def half_hour_strikes : ℕ :=
  24 * 2

def total_chimes_in_a_day : ℕ :=
  strikes_in_24_hours + half_hour_strikes

-- Statement to prove
theorem chimes_in_a_day : total_chimes_in_a_day = 204 :=
by 
  -- The proof would be placed here
  sorry

end chimes_in_a_day_l287_28730


namespace no_such_function_exists_l287_28711

def satisfies_condition (f : ℤ → ℤ) : Prop :=
  ∀ x y z : ℤ, f (x * y) + f (x * z) - f x * f (y * z) ≤ -1

theorem no_such_function_exists : (∃ f : ℤ → ℤ, satisfies_condition f) = false :=
by
  sorry

end no_such_function_exists_l287_28711


namespace delete_middle_divides_l287_28772

def digits (n : ℕ) : ℕ × ℕ × ℕ × ℕ × ℕ :=
  let a := n / 10000
  let b := (n % 10000) / 1000
  let c := (n % 1000) / 100
  let d := (n % 100) / 10
  let e := n % 10
  (a, b, c, d, e)

def delete_middle_digit (n : ℕ) : ℕ :=
  let (a, b, c, d, e) := digits n
  1000 * a + 100 * b + 10 * d + e

theorem delete_middle_divides (n : ℕ) (hn : 10000 ≤ n ∧ n < 100000) :
  (delete_middle_digit n) ∣ n :=
sorry

end delete_middle_divides_l287_28772


namespace work_days_for_A_l287_28786

theorem work_days_for_A (x : ℕ) : 
  (∀ a b, 
    (a = 1 / (x : ℚ)) ∧ 
    (b = 1 / 20) ∧ 
    (8 * (a + b) = 14 / 15) → 
    x = 15) :=
by
  intros a b h
  have ha : a = 1 / (x : ℚ) := h.1
  have hb : b = 1 / 20 := h.2.1
  have hab : 8 * (a + b) = 14 / 15 := h.2.2
  sorry

end work_days_for_A_l287_28786


namespace min_value_circles_tangents_l287_28720

theorem min_value_circles_tangents (a b : ℝ) (h1 : (∃ x y : ℝ, x^2 + y^2 + 2 * a * x + a^2 - 4 = 0) ∧ 
  (∃ x y : ℝ, x^2 + y^2 - 4 * b * y - 1 + 4 * b^2 = 0))
  (h2 : ∃ k : ℕ, k = 3) (h3 : a ≠ 0) (h4 : b ≠ 0) : 
  (∃ m : ℝ, m = 1 ∧  ∀ x : ℝ, (x = (1 / a^2) + (1 / b^2)) → x ≥ m) :=
  sorry

end min_value_circles_tangents_l287_28720


namespace average_expression_l287_28743

-- Define a theorem to verify the given problem
theorem average_expression (E a : ℤ) (h1 : a = 34) (h2 : (E + (3 * a - 8)) / 2 = 89) : E = 84 :=
by
  -- Proof goes here
  sorry

end average_expression_l287_28743


namespace minimum_value_exists_l287_28779

theorem minimum_value_exists (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : 2 * m + n = 1) : 
  ∃ (min_val : ℝ), min_val = (3 + 2 * Real.sqrt 2) ∧ (1 / m + 1 / n ≥ min_val) :=
by {
  -- Proof will be provided here.
  sorry
}

end minimum_value_exists_l287_28779


namespace right_triangle_area_and_perimeter_l287_28784

theorem right_triangle_area_and_perimeter (a c : ℕ) (h₁ : c = 13) (h₂ : a = 5) :
  ∃ (b : ℕ), b^2 = c^2 - a^2 ∧
             (1/2 : ℝ) * (a : ℝ) * (b : ℝ) = 30 ∧
             (a + b + c : ℕ) = 30 :=
by
  sorry

end right_triangle_area_and_perimeter_l287_28784


namespace solve_y_l287_28729

theorem solve_y (y : ℝ) : (12 - y)^2 = 4 * y^2 ↔ y = 4 ∨ y = -12 := by
  sorry

end solve_y_l287_28729


namespace express_in_scientific_notation_l287_28787

theorem express_in_scientific_notation : (250000 : ℝ) = 2.5 * 10^5 := 
by {
  -- proof
  sorry
}

end express_in_scientific_notation_l287_28787


namespace encyclopedia_total_pages_l287_28738

noncomputable def totalPages : ℕ :=
450 + 3 * 90 +
650 + 5 * 68 +
712 + 4 * 75 +
820 + 6 * 120 +
530 + 2 * 110 +
900 + 7 * 95 +
680 + 4 * 80 +
555 + 3 * 180 +
990 + 5 * 53 +
825 + 6 * 150 +
410 + 2 * 200 +
1014 + 7 * 69

theorem encyclopedia_total_pages : totalPages = 13659 := by
  sorry

end encyclopedia_total_pages_l287_28738


namespace area_of_rectangular_garden_l287_28770

-- Definition of conditions
def width : ℕ := 14
def length : ℕ := 3 * width

-- Statement for proof of the area of the rectangular garden
theorem area_of_rectangular_garden :
  length * width = 588 := 
by
  sorry

end area_of_rectangular_garden_l287_28770


namespace function_domain_l287_28771

theorem function_domain (x : ℝ) :
  (x + 5 ≥ 0) ∧ (x + 2 ≠ 0) ↔ (x ≥ -5) ∧ (x ≠ -2) :=
by
  sorry

end function_domain_l287_28771


namespace students_with_average_age_of_16_l287_28710

theorem students_with_average_age_of_16
  (N : ℕ) (A : ℕ) (N14 : ℕ) (A15 : ℕ) (N16 : ℕ)
  (h1 : N = 15) (h2 : A = 15) (h3 : N14 = 5) (h4 : A15 = 11) :
  N16 = 9 :=
sorry

end students_with_average_age_of_16_l287_28710


namespace unique_integer_solution_l287_28707

theorem unique_integer_solution (a b : ℤ) : 
  ∀ x₁ x₂ : ℤ, (x₁ - a) * (x₁ - b) * (x₁ - 3) + 1 = 0 ∧ (x₂ - a) * (x₂ - b) * (x₂ - 3) + 1 = 0 → x₁ = x₂ :=
by
  sorry

end unique_integer_solution_l287_28707


namespace part_1_part_2_part_3_l287_28767

/-- Defining a structure to hold the values of x and y as given in the problem --/
structure PhoneFeeData (α : Type) :=
  (x : α) (y : α)

def problem_data : List (PhoneFeeData ℝ) :=
  [
    ⟨1, 18.4⟩, ⟨2, 18.8⟩, ⟨3, 19.2⟩, ⟨4, 19.6⟩, ⟨5, 20⟩, ⟨6, 20.4⟩
  ]

noncomputable def phone_fee_equation (x : ℝ) : ℝ := 0.4 * x + 18

theorem part_1 :
  ∀ data ∈ problem_data, phone_fee_equation data.x = data.y :=
by
  sorry

theorem part_2 : phone_fee_equation 10 = 22 :=
by
  sorry

theorem part_3 : ∀ x : ℝ, phone_fee_equation x = 26 → x = 20 :=
by
  sorry

end part_1_part_2_part_3_l287_28767


namespace range_of_a_l287_28723

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x < 0 ∧ 5^x = (a + 3) / (a - 3)) → a < -3 :=
by
  sorry

end range_of_a_l287_28723


namespace standard_deviation_upper_bound_l287_28701

theorem standard_deviation_upper_bound (Mean StdDev : ℝ) (h : Mean = 54) (h2 : 54 - 3 * StdDev > 47) : StdDev < 2.33 :=
by
  sorry

end standard_deviation_upper_bound_l287_28701


namespace length_of_row_of_small_cubes_l287_28797

/-!
# Problem: Calculate the length of a row of smaller cubes

A cube with an edge length of 0.5 m is cut into smaller cubes, each with an edge length of 2 mm.
Prove that the length of the row formed by arranging the smaller cubes in a continuous line 
is 31 km and 250 m.
-/

noncomputable def large_cube_edge_length_m : ℝ := 0.5
noncomputable def small_cube_edge_length_mm : ℝ := 2

theorem length_of_row_of_small_cubes :
  let length_mm := 31250000
  (31 : ℝ) * 1000 + (250 : ℝ) = length_mm / 1000 + 250 := 
sorry

end length_of_row_of_small_cubes_l287_28797


namespace no_sqrt_negative_number_l287_28762

theorem no_sqrt_negative_number (a b c d : ℝ) (hA : a = (-3)^2) (hB : b = 0) (hC : c = 1/8) (hD : d = -6^3) : 
  ¬ (∃ x : ℝ, x^2 = d) :=
by
  sorry

end no_sqrt_negative_number_l287_28762


namespace isosceles_base_angle_l287_28764

theorem isosceles_base_angle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = B ∨ A = C) (h3 : A = 80 ∨ B = 80 ∨ C = 80) : (A = 80 ∧ B = 80) ∨ (A = 80 ∧ C = 80) ∨ (B = 80 ∧ C = 50) ∨ (C = 80 ∧ B = 50) :=
sorry

end isosceles_base_angle_l287_28764


namespace amoeba_population_after_ten_days_l287_28719

-- Definitions based on the conditions
def initial_population : ℕ := 3
def amoeba_growth (n : ℕ) : ℕ := initial_population * 2^n

-- Lean statement for the proof problem
theorem amoeba_population_after_ten_days : amoeba_growth 10 = 3072 :=
by 
  sorry

end amoeba_population_after_ten_days_l287_28719


namespace isabella_canadian_dollars_sum_l287_28768

def sum_of_digits (n : Nat) : Nat :=
  (n % 10) + ((n / 10) % 10)

theorem isabella_canadian_dollars_sum (d : Nat) (H: 10 * d = 7 * d + 280) : sum_of_digits d = 12 :=
by
  sorry

end isabella_canadian_dollars_sum_l287_28768


namespace mean_of_other_four_l287_28716

theorem mean_of_other_four (a b c d e : ℕ) (h_mean : (a + b + c + d + e + 90) / 6 = 75)
  (h_max : max a (max b (max c (max d (max e 90)))) = 90)
  (h_twice : b = 2 * a) :
  (a + c + d + e) / 4 = 60 :=
by
  sorry

end mean_of_other_four_l287_28716


namespace cost_of_outfit_l287_28705

theorem cost_of_outfit (P T J : ℝ) 
  (h1 : 4 * P + 8 * T + 2 * J = 2400)
  (h2 : 2 * P + 14 * T + 3 * J = 2400)
  (h3 : 3 * P + 6 * T = 1500) :
  P + 4 * T + J = 860 := 
sorry

end cost_of_outfit_l287_28705


namespace proportional_distribution_ratio_l287_28712

theorem proportional_distribution_ratio (B : ℝ) (r : ℝ) (S : ℝ) 
  (h1 : B = 80) 
  (h2 : S = 164)
  (h3 : S = (B / (1 - r)) + (B * (1 - r))) : 
  r = 0.2 := 
sorry

end proportional_distribution_ratio_l287_28712


namespace box_filling_possibilities_l287_28718

def possible_numbers : List ℕ := [2015, 2016, 2017, 2018, 2019]

def fill_the_boxes (D O G C W : ℕ) : Prop :=
  D + O + G = C + O + W

theorem box_filling_possibilities :
  (∃ D O G C W : ℕ, 
    D ∈ possible_numbers ∧
    O ∈ possible_numbers ∧
    G ∈ possible_numbers ∧
    C ∈ possible_numbers ∧
    W ∈ possible_numbers ∧
    D ≠ O ∧ D ≠ G ∧ D ≠ C ∧ D ≠ W ∧
    O ≠ G ∧ O ≠ C ∧ O ≠ W ∧
    G ≠ C ∧ G ≠ W ∧
    C ≠ W ∧
    fill_the_boxes D O G C W) → 
    ∃ ways : ℕ, ways = 24 :=
  sorry

end box_filling_possibilities_l287_28718


namespace min_value_of_M_l287_28754

noncomputable def f (p q x : ℝ) : ℝ := x^2 + p * x + q

theorem min_value_of_M (p q M : ℝ) :
  (M = max (|f p q 1|) (max (|f p q (-1)|) (|f p q 0|))) →
  (0 > f p q 1 → 0 > f p q (-1) → 0 > f p q 0 → M = 1 / 2) :=
sorry

end min_value_of_M_l287_28754


namespace multiplication_72515_9999_l287_28715

theorem multiplication_72515_9999 : 72515 * 9999 = 725077485 :=
by
  sorry

end multiplication_72515_9999_l287_28715


namespace exists_small_triangle_l287_28780

-- Definitions and conditions based on the identified problem points
def square_side_length : ℝ := 1
def total_points : ℕ := 53
def vertex_points : ℕ := 4
def interior_points : ℕ := 49
def total_area : ℝ := square_side_length ^ 2
def max_triangle_area : ℝ := 0.01

-- The main theorem statement
theorem exists_small_triangle
  (sq_side : ℝ := square_side_length)
  (total_pts : ℕ := total_points)
  (vertex_pts : ℕ := vertex_points)
  (interior_pts : ℕ := interior_points)
  (total_ar : ℝ := total_area)
  (max_area : ℝ := max_triangle_area)
  (h_side : sq_side = 1)
  (h_pts : total_pts = 53)
  (h_vertex : vertex_pts = 4)
  (h_interior : interior_pts = 49)
  (h_total_area : total_ar = 1) :
  ∃ (t : ℝ), t ≤ max_area :=
sorry

end exists_small_triangle_l287_28780


namespace angle_equivalence_l287_28751

theorem angle_equivalence : (2023 % 360 = -137 % 360) := 
by 
  sorry

end angle_equivalence_l287_28751


namespace max_area_quadrilateral_sum_opposite_angles_l287_28745

theorem max_area_quadrilateral (a b c d : ℝ) (h₁ : a = 3) (h₂ : b = 3) (h₃ : c = 4) (h₄ : d = 4) :
  ∃ (area : ℝ), area = 12 :=
by {
  sorry
}

theorem sum_opposite_angles (a b c d : ℝ) (θ₁ θ₂ θ₃ θ₄ : ℝ) 
  (h₁ : a = 3) (h₂ : b = 3) (h₃ : c = 4) (h₄ : d = 4) 
  (h_area : ∃ (area : ℝ), area = 12) 
  (h_opposite1 : θ₁ + θ₃ = 180) (h_opposite2 : θ₂ + θ₄ = 180) :
  ∃ θ, θ = 180 :=
by {
  sorry
}

end max_area_quadrilateral_sum_opposite_angles_l287_28745


namespace Coe_speed_theorem_l287_28709

-- Define the conditions
def Teena_speed : ℝ := 55
def initial_distance_behind : ℝ := 7.5
def time_hours : ℝ := 1.5
def distance_ahead : ℝ := 15

-- Define Coe's speed
def Coe_speed := 50

-- State the theorem
theorem Coe_speed_theorem : 
  let distance_Teena_covers := Teena_speed * time_hours
  let total_relative_distance := distance_Teena_covers + initial_distance_behind
  let distance_Coe_covers := total_relative_distance - distance_ahead
  let computed_Coe_speed := distance_Coe_covers / time_hours
  computed_Coe_speed = Coe_speed :=
by sorry

end Coe_speed_theorem_l287_28709


namespace quadrilateral_with_three_right_angles_is_rectangle_l287_28722

-- Define a quadrilateral with angles
structure Quadrilateral :=
  (a1 a2 a3 a4 : ℝ)
  (sum_angles : a1 + a2 + a3 + a4 = 360)

-- Define a right angle
def is_right_angle (angle : ℝ) : Prop :=
  angle = 90

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
  is_right_angle q.a1 ∧ is_right_angle q.a2 ∧ is_right_angle q.a3 ∧ is_right_angle q.a4

-- The main theorem: if a quadrilateral has three right angles, it is a rectangle
theorem quadrilateral_with_three_right_angles_is_rectangle 
  (q : Quadrilateral) 
  (h1 : is_right_angle q.a1) 
  (h2 : is_right_angle q.a2) 
  (h3 : is_right_angle q.a3) 
  : is_rectangle q :=
sorry

end quadrilateral_with_three_right_angles_is_rectangle_l287_28722


namespace find_a_l287_28799

theorem find_a (a : ℝ) (x : ℝ) : (a - 1) * x^|a| + 4 = 0 → |a| = 1 → a ≠ 1 → a = -1 :=
by
  intros
  sorry

end find_a_l287_28799


namespace rain_at_house_l287_28725

/-- Define the amounts of rain on the three days Greg was camping. -/
def rain_day1 : ℕ := 3
def rain_day2 : ℕ := 6
def rain_day3 : ℕ := 5

/-- Define the total rain experienced by Greg while camping. -/
def total_rain_camping := rain_day1 + rain_day2 + rain_day3

/-- Define the difference in the rain experienced by Greg while camping and at his house. -/
def rain_difference : ℕ := 12

/-- Define the total amount of rain at Greg's house. -/
def total_rain_house := total_rain_camping + rain_difference

/-- Prove that the total rain at Greg's house is 26 mm. -/
theorem rain_at_house : total_rain_house = 26 := by
  /- We know that total_rain_camping = 14 mm and rain_difference = 12 mm -/
  /- Therefore, total_rain_house = 14 mm + 12 mm = 26 mm -/
  sorry

end rain_at_house_l287_28725


namespace John_ASMC_score_l287_28748

def ASMC_score (c w : ℕ) : ℕ := 25 + 5 * c - 2 * w

theorem John_ASMC_score (c w : ℕ) (h1 : ASMC_score c w = 100) (h2 : c + w ≤ 25) :
  c = 19 ∧ w = 10 :=
by {
  sorry
}

end John_ASMC_score_l287_28748


namespace digging_project_length_l287_28740

theorem digging_project_length (L : ℝ) (V1 V2 : ℝ) (depth1 length1 depth2 breadth1 breadth2 : ℝ) 
  (h1 : depth1 = 100) (h2 : length1 = 25) (h3 : breadth1 = 30) (h4 : V1 = depth1 * length1 * breadth1)
  (h5 : depth2 = 75) (h6 : breadth2 = 50) (h7 : V2 = depth2 * L * breadth2) (h8 : V1 / V2 = 1) :
  L = 20 :=
by
  sorry

end digging_project_length_l287_28740


namespace line_through_A_with_zero_sum_of_intercepts_l287_28721

-- Definitions
def passesThroughPoint (A : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  l A.1 A.2

def sumInterceptsZero (l : ℝ → ℝ → Prop) : Prop :=
  ∃ a b : ℝ, l a 0 ∧ l 0 b ∧ a + b = 0

-- Theorem statement
theorem line_through_A_with_zero_sum_of_intercepts (l : ℝ → ℝ → Prop) :
  passesThroughPoint (1, 4) l ∧ sumInterceptsZero l →
  (∀ x y, l x y ↔ 4 * x - y = 0) ∨ (∀ x y, l x y ↔ x - y + 3 = 0) :=
sorry

end line_through_A_with_zero_sum_of_intercepts_l287_28721


namespace boys_and_girls_in_class_l287_28778

theorem boys_and_girls_in_class (b g : ℕ) (h1 : b + g = 21) (h2 : 5 * b + 2 * g = 69) 
: b = 9 ∧ g = 12 := by
  sorry

end boys_and_girls_in_class_l287_28778


namespace shifted_line_does_not_pass_through_third_quadrant_l287_28706

-- The condition: The original line is y = -2x - 1
def original_line (x : ℝ) : ℝ := -2 * x - 1

-- The condition: The line is shifted 3 units to the right
def shifted_line (x : ℝ) : ℝ := -2 * (x - 3) - 1

theorem shifted_line_does_not_pass_through_third_quadrant :
  ¬(∃ (x y : ℝ), y = shifted_line x ∧ x < 0 ∧ y < 0) :=
sorry

end shifted_line_does_not_pass_through_third_quadrant_l287_28706


namespace monochromatic_triangle_l287_28769

def R₃ (n : ℕ) : ℕ := sorry

theorem monochromatic_triangle {n : ℕ} (h1 : R₃ 2 = 6)
  (h2 : ∀ n, R₃ (n + 1) ≤ (n + 1) * R₃ n - n + 1) :
  R₃ n ≤ 3 * Nat.factorial n :=
by
  induction n with
  | zero => sorry -- base case proof
  | succ n ih => sorry -- inductive step proof

end monochromatic_triangle_l287_28769


namespace jenn_has_five_jars_l287_28795

/-- Each jar can hold 160 quarters, the bike costs 180 dollars, 
    Jenn will have 20 dollars left over, 
    and a quarter is worth 0.25 dollars.
    Prove that Jenn has 5 jars full of quarters. -/
theorem jenn_has_five_jars :
  let quarters_per_jar := 160
  let bike_cost := 180
  let money_left := 20
  let total_money_needed := bike_cost + money_left
  let quarter_value := 0.25
  let total_quarters_needed := total_money_needed / quarter_value
  let jars := total_quarters_needed / quarters_per_jar
  
  jars = 5 :=
by
  sorry

end jenn_has_five_jars_l287_28795


namespace fuel_tank_ethanol_l287_28724

theorem fuel_tank_ethanol (x : ℝ) (H : 0.12 * x + 0.16 * (208 - x) = 30) : x = 82 := 
by
  sorry

end fuel_tank_ethanol_l287_28724


namespace soccer_camp_afternoon_kids_l287_28774

def num_kids_in_camp : ℕ := 2000
def fraction_going_to_soccer_camp : ℚ := 1 / 2
def fraction_going_to_soccer_camp_in_morning : ℚ := 1 / 4

noncomputable def num_kids_going_to_soccer_camp := num_kids_in_camp * fraction_going_to_soccer_camp
noncomputable def num_kids_going_to_soccer_camp_in_morning := num_kids_going_to_soccer_camp * fraction_going_to_soccer_camp_in_morning
noncomputable def num_kids_going_to_soccer_camp_in_afternoon := num_kids_going_to_soccer_camp - num_kids_going_to_soccer_camp_in_morning

theorem soccer_camp_afternoon_kids : num_kids_going_to_soccer_camp_in_afternoon = 750 :=
by
  sorry

end soccer_camp_afternoon_kids_l287_28774


namespace simplify_expression_l287_28734

theorem simplify_expression (x y : ℤ) (h1 : x = -2) (h2 : y = -1) :
  (2 * (x - 2 * y) * (2 * x + y) - (x + 2 * y)^2 + x * (8 * y - 3 * x)) / (6 * y) = 2 :=
by sorry

end simplify_expression_l287_28734


namespace phase_shift_of_sine_l287_28735

theorem phase_shift_of_sine :
  let a := 3
  let b := 4
  let c := - (Real.pi / 4)
  let phase_shift := -(c / b)
  phase_shift = Real.pi / 16 :=
by
  sorry

end phase_shift_of_sine_l287_28735


namespace ellipse_equation_range_of_M_x_coordinate_l287_28700

-- Proof 1: Proving the equation of the ellipse
theorem ellipse_equation {a b : ℝ} (h_ab : a > b) (h_b0 : b > 0) (e : ℝ)
  (h_e : e = (Real.sqrt 3) / 3) (vertex : ℝ × ℝ) (h_vertex : vertex = (Real.sqrt 3, 0)) :
  (∃ (a b : ℝ), a > b ∧ b > 0 ∧ e = (Real.sqrt 3) / 3 ∧ vertex = (Real.sqrt 3, 0) ∧ (∀ (x y : ℝ), (x^2) / 3 + (y^2) / 2 = 1)) :=
sorry

-- Proof 2: Proving the range of x-coordinate of point M
theorem range_of_M_x_coordinate (k : ℝ) (h_k : k ≠ 0) :
  (∃ M_x : ℝ, by sorry) :=
sorry


end ellipse_equation_range_of_M_x_coordinate_l287_28700


namespace smallest_number_h_divisible_8_11_24_l287_28747

theorem smallest_number_h_divisible_8_11_24 : 
  ∃ h : ℕ, (h + 5) % 8 = 0 ∧ (h + 5) % 11 = 0 ∧ (h + 5) % 24 = 0 ∧ h = 259 :=
by
  sorry

end smallest_number_h_divisible_8_11_24_l287_28747


namespace g_2023_eq_0_l287_28727

noncomputable def g (x : ℕ) : ℝ := sorry

axiom g_defined (x : ℕ) : ∃ y : ℝ, g x = y

axiom g_initial : g 1 = 1

axiom g_functional (a b : ℕ) : g (a + b) = g a + g b - 2 * g (a * b + 1)

theorem g_2023_eq_0 : g 2023 = 0 :=
sorry

end g_2023_eq_0_l287_28727


namespace binom_np_n_mod_p2_l287_28726

   theorem binom_np_n_mod_p2 (p n : ℕ) (hp : Nat.Prime p) : (Nat.choose (n * p) n) % (p ^ 2) = n % (p ^ 2) :=
   by
     sorry
   
end binom_np_n_mod_p2_l287_28726


namespace intersection_correct_l287_28752

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := { y | ∃ x ∈ A, y = 2 * x - 1 }

def intersection : Set ℕ := { x | x ∈ A ∧ x ∈ B }

theorem intersection_correct : intersection = {1, 3} := by
  sorry

end intersection_correct_l287_28752


namespace total_eggs_collected_l287_28791

-- Define the variables given in the conditions
def Benjamin_eggs := 6
def Carla_eggs := 3 * Benjamin_eggs
def Trisha_eggs := Benjamin_eggs - 4

-- State the theorem using the conditions and correct answer in the equivalent proof problem
theorem total_eggs_collected :
  Benjamin_eggs + Carla_eggs + Trisha_eggs = 26 := by
  -- Proof goes here.
  sorry

end total_eggs_collected_l287_28791


namespace correct_operation_l287_28785

theorem correct_operation (a b : ℝ) : 
  (3 * Real.sqrt 7 + 7 * Real.sqrt 3 ≠ 10 * Real.sqrt 10) ∧ 
  (Real.sqrt (2 * a) * Real.sqrt (3) * a = Real.sqrt (6) * a) ∧ 
  (Real.sqrt a - Real.sqrt b ≠ Real.sqrt (a - b)) ∧ 
  (Real.sqrt (20 / 45) ≠ 4 / 9) :=
by
  sorry

end correct_operation_l287_28785


namespace seventh_grader_count_l287_28776

variables {x n : ℝ}

noncomputable def number_of_seventh_graders (x n : ℝ) :=
  10 * x = 10 * x ∧  -- Condition 1
  4.5 * n = 4.5 * n ∧  -- Condition 2
  11 * x = 11 * x ∧  -- Condition 3
  5.5 * n = 5.5 * n ∧  -- Condition 4
  5.5 * n = (11 * x * (11 * x - 1)) / 2 ∧  -- Condition 5
  n = x * (11 * x - 1)  -- Condition 6

theorem seventh_grader_count (x n : ℝ) (h : number_of_seventh_graders x n) : x = 1 :=
  sorry

end seventh_grader_count_l287_28776


namespace x_coordinate_of_P_l287_28792

noncomputable section

open Real

-- Define the standard properties of the parabola and point P
def parabola (p : ℝ) (x y : ℝ) := (y ^ 2 = 4 * x)

def distance (P F : ℝ × ℝ) : ℝ := 
  sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2)

-- Position of the focus for the given parabola y^2 = 4x; Focus F(1, 0)
def focus : ℝ × ℝ := (1, 0)

-- The given conditions translated into Lean form
def on_parabola (x y : ℝ) := parabola 2 x y ∧ distance (x, y) focus = 5

-- The theorem we need to prove: If point P satisfies these conditions, then its x-coordinate is 4
theorem x_coordinate_of_P (P : ℝ × ℝ) (h : on_parabola P.1 P.2) : P.1 = 4 :=
by
  sorry

end x_coordinate_of_P_l287_28792


namespace tire_miles_used_l287_28761

theorem tire_miles_used (total_miles : ℕ) (number_of_tires : ℕ) (tires_in_use : ℕ)
  (h_total_miles : total_miles = 40000) (h_number_of_tires : number_of_tires = 6)
  (h_tires_in_use : tires_in_use = 4) : 
  (total_miles * tires_in_use) / number_of_tires = 26667 := 
by 
  sorry

end tire_miles_used_l287_28761


namespace solve_for_x_l287_28763

theorem solve_for_x (x : ℚ) (h : x > 0) (hx : 3 * x^2 + 7 * x - 20 = 0) : x = 5 / 3 :=
by 
  sorry

end solve_for_x_l287_28763


namespace angle_complement_30_l287_28713

def complement_angle (x : ℝ) : ℝ := 90 - x

theorem angle_complement_30 (x : ℝ) (h : x = complement_angle x - 30) : x = 30 :=
by
  sorry

end angle_complement_30_l287_28713


namespace fraction_non_throwers_left_handed_l287_28765

theorem fraction_non_throwers_left_handed (total_players : ℕ) (num_throwers : ℕ) (total_right_handed : ℕ) (all_throwers_right_handed : ∀ x, x < num_throwers → true) (num_right_handed := total_right_handed - num_throwers) (non_throwers := total_players - num_throwers) (num_left_handed := non_throwers - num_right_handed) : 
    total_players = 70 → 
    num_throwers = 40 → 
    total_right_handed = 60 → 
    (∃ f: ℚ, f = num_left_handed / non_throwers ∧ f = 1/3) := 
by {
  sorry
}

end fraction_non_throwers_left_handed_l287_28765


namespace gum_needed_l287_28742

-- Definitions based on problem conditions
def num_cousins : ℕ := 4
def gum_per_cousin : ℕ := 5

-- Proposition that we need to prove
theorem gum_needed : num_cousins * gum_per_cousin = 20 := by
  sorry

end gum_needed_l287_28742


namespace ratio_of_areas_of_triangles_l287_28755

theorem ratio_of_areas_of_triangles 
  (a b c d e f : ℕ)
  (h1 : a = 7) (h2 : b = 24) (h3 : c = 25)
  (h4 : d = 9) (h5 : e = 40) (h6 : f = 41) : 
  (84 : ℚ) / (180 : ℚ) = 7 / 15 := by
  have hPQR : a^2 + b^2 = c^2 := by
    rw [h1, h2, h3]
    norm_num
  have hSTU : d^2 + e^2 = f^2 := by
    rw [h4, h5, h6]
    norm_num
  have areaPQR : (1/2 : ℚ) * a * b = 84 := by
    rw [h1, h2]
    norm_num
  have areaSTU : (1/2 : ℚ) * d * e = 180 := by
    rw [h4, h5]
    norm_num
  sorry

end ratio_of_areas_of_triangles_l287_28755


namespace total_eggs_examined_l287_28796

def trays := 7
def eggs_per_tray := 10

theorem total_eggs_examined : trays * eggs_per_tray = 70 :=
by 
  sorry

end total_eggs_examined_l287_28796


namespace probability_same_gate_l287_28736

open Finset

-- Definitions based on the conditions
def num_gates : ℕ := 3
def total_combinations : ℕ := num_gates * num_gates -- total number of combinations for both persons
def favorable_combinations : ℕ := num_gates         -- favorable combinations (both choose same gate)

-- Problem statement
theorem probability_same_gate : 
  ∃ (p : ℚ), p = (favorable_combinations : ℚ) / (total_combinations : ℚ) ∧ p = (1 / 3 : ℚ) := 
by
  sorry

end probability_same_gate_l287_28736


namespace count_arithmetic_sequence_l287_28793

theorem count_arithmetic_sequence :
  ∃ n, 195 - (n - 1) * 3 = 12 ∧ n = 62 :=
by {
  sorry
}

end count_arithmetic_sequence_l287_28793


namespace expression_evaluation_l287_28702

theorem expression_evaluation : (2 - (-3) - 4 + (-5) + 6 - (-7) - 8 = 1) := 
by 
  sorry

end expression_evaluation_l287_28702


namespace gym_membership_cost_l287_28737

theorem gym_membership_cost 
    (cheap_monthly_fee : ℕ := 10)
    (cheap_signup_fee : ℕ := 50)
    (expensive_monthly_multiplier : ℕ := 3)
    (months_in_year : ℕ := 12)
    (expensive_signup_multiplier : ℕ := 4) :
    let cheap_gym_cost := cheap_monthly_fee * months_in_year + cheap_signup_fee
    let expensive_monthly_fee := cheap_monthly_fee * expensive_monthly_multiplier
    let expensive_gym_cost := expensive_monthly_fee * months_in_year + expensive_monthly_fee * expensive_signup_multiplier
    let total_cost := cheap_gym_cost + expensive_gym_cost
    total_cost = 650 :=
by
  sorry -- Proof is omitted because the focus is on the statement equivalency.

end gym_membership_cost_l287_28737


namespace percentage_failing_both_l287_28714

-- Define the conditions as constants
def percentage_failing_hindi : ℝ := 0.25
def percentage_failing_english : ℝ := 0.48
def percentage_passing_both : ℝ := 0.54

-- Define the percentage of students who failed in at least one subject
def percentage_failing_at_least_one : ℝ := 1 - percentage_passing_both

-- The main theorem statement we want to prove
theorem percentage_failing_both :
  percentage_failing_at_least_one = percentage_failing_hindi + percentage_failing_english - 0.27 := by
sorry

end percentage_failing_both_l287_28714


namespace solve_inequality_l287_28798

theorem solve_inequality (a x : ℝ) : 
  (ax^2 + (a - 1) * x - 1 < 0) ↔ (
  (a = 0 ∧ x > -1) ∨ 
  (a > 0 ∧ -1 < x ∧ x < 1/a) ∨
  (-1 < a ∧ a < 0 ∧ (x < 1/a ∨ x > -1)) ∨ 
  (a = -1 ∧ x ≠ -1) ∨ 
  (a < -1 ∧ (x < -1 ∨ x > 1/a))
) := sorry

end solve_inequality_l287_28798


namespace determine_contents_l287_28790

inductive Color
| White
| Black

open Color

-- Definitions of the mislabeled boxes
def mislabeled (box : Nat → List Color) : Prop :=
  ¬ (box 1 = [Black, Black] ∧ box 2 = [Black, White]
     ∧ box 3 = [White, White])

-- Draw a ball from a box revealing its content
def draw_ball (box : Nat → List Color) (i : Nat) (c : Color) : Prop :=
  c ∈ box i

-- theorem statement
theorem determine_contents (box : Nat → List Color) (c : Color) (h : draw_ball box 3 c) (hl : mislabeled box) :
  (c = White → box 3 = [White, White] ∧ box 2 = [Black, White] ∧ box 1 = [Black, Black]) ∧
  (c = Black → box 3 = [Black, Black] ∧ box 2 = [Black, White] ∧ box 1 = [White, White]) :=
by
  sorry

end determine_contents_l287_28790


namespace room_width_l287_28708

theorem room_width (w : ℝ) (h1 : 21 > 0) (h2 : 2 > 0) 
  (h3 : (25 * (w + 4) - 21 * w = 148)) : w = 12 :=
by {
  sorry
}

end room_width_l287_28708


namespace compare_negatives_l287_28758

theorem compare_negatives : -4 < -2.1 := 
sorry

end compare_negatives_l287_28758


namespace garden_perimeter_l287_28781

-- We are given:
variables (a b : ℝ)
variables (h1 : b = 3 * a)
variables (h2 : a^2 + b^2 = 34^2)
variables (h3 : a * b = 240)

-- We must prove:
theorem garden_perimeter (h4 : a^2 + 9 * a^2 = 1156) (h5 : 10 * a^2 = 1156) (h6 : a^2 = 115.6) 
  (h7 : 3 * a^2 = 240) (h8 : a^2 = 80) :
  2 * (a + b) = 72 := 
by
  sorry

end garden_perimeter_l287_28781


namespace arrangements_A_and_B_together_arrangements_A_not_head_B_not_tail_arrangements_A_and_B_not_next_arrangements_one_person_between_A_and_B_l287_28749

open Nat

axiom students : Fin 7 → Type -- Define students indexed by their position in the line.

noncomputable def fact : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * fact n

theorem arrangements_A_and_B_together :
  (2 * fact 6) = 1440 := 
by 
  sorry

theorem arrangements_A_not_head_B_not_tail :
  (fact 7 - 2 * fact 6 + fact 5) = 3720 := 
by 
  sorry

theorem arrangements_A_and_B_not_next :
  (3600) = 3600 := 
by 
  sorry

theorem arrangements_one_person_between_A_and_B :
  (fact 5 * 2) = 1200 := 
by 
  sorry

end arrangements_A_and_B_together_arrangements_A_not_head_B_not_tail_arrangements_A_and_B_not_next_arrangements_one_person_between_A_and_B_l287_28749


namespace possible_values_f2001_l287_28773

noncomputable def f : ℕ → ℝ := sorry

lemma functional_equation (a b d : ℕ) (h₁ : 1 < a) (h₂ : 1 < b) (h₃ : d = Nat.gcd a b) :
  f (a * b) = f d * (f (a / d) + f (b / d)) :=
sorry

theorem possible_values_f2001 :
  f 2001 = 0 ∨ f 2001 = 1 / 2 :=
sorry

end possible_values_f2001_l287_28773
