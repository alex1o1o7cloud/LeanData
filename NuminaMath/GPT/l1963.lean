import Mathlib

namespace geometric_sum_equals_fraction_l1963_196341

theorem geometric_sum_equals_fraction (n : ℕ) (a r : ℝ) 
  (h_a : a = 1) (h_r : r = 1 / 2) 
  (h_sum : a * (1 - r^n) / (1 - r) = 511 / 512) : 
  n = 9 := 
by 
  sorry

end geometric_sum_equals_fraction_l1963_196341


namespace sequence_remainder_zero_l1963_196357

theorem sequence_remainder_zero :
  let a := 3
  let d := 8
  let n := 32
  let aₙ := a + (n - 1) * d
  let Sₙ := n * (a + aₙ) / 2
  aₙ = 251 → Sₙ % 8 = 0 :=
by
  intros
  sorry

end sequence_remainder_zero_l1963_196357


namespace trapezoid_combined_area_correct_l1963_196303

noncomputable def combined_trapezoid_area_proof : Prop :=
  let EF : ℝ := 60
  let GH : ℝ := 40
  let altitude_EF_GH : ℝ := 18
  let trapezoid_EFGH_area : ℝ := (1 / 2) * (EF + GH) * altitude_EF_GH

  let IJ : ℝ := 30
  let KL : ℝ := 25
  let altitude_IJ_KL : ℝ := 10
  let trapezoid_IJKL_area : ℝ := (1 / 2) * (IJ + KL) * altitude_IJ_KL

  let combined_area : ℝ := trapezoid_EFGH_area + trapezoid_IJKL_area

  combined_area = 1175

theorem trapezoid_combined_area_correct : combined_trapezoid_area_proof := by
  sorry

end trapezoid_combined_area_correct_l1963_196303


namespace cost_of_jeans_l1963_196393

theorem cost_of_jeans 
  (price_socks : ℕ)
  (price_tshirt : ℕ)
  (price_jeans : ℕ)
  (h1 : price_socks = 5)
  (h2 : price_tshirt = price_socks + 10)
  (h3 : price_jeans = 2 * price_tshirt) :
  price_jeans = 30 :=
  by
    -- Sorry skips the proof, complies with the instructions
    sorry

end cost_of_jeans_l1963_196393


namespace three_digit_number_is_504_l1963_196327

theorem three_digit_number_is_504 (x : ℕ) [Decidable (x = 504)] :
  100 ≤ x ∧ x ≤ 999 →
  (x - 7) % 7 = 0 ∧
  (x - 8) % 8 = 0 ∧
  (x - 9) % 9 = 0 →
  x = 504 :=
by
  sorry

end three_digit_number_is_504_l1963_196327


namespace white_roses_per_table_decoration_l1963_196317

theorem white_roses_per_table_decoration (x : ℕ) :
  let bouquets := 5
  let table_decorations := 7
  let roses_per_bouquet := 5
  let total_roses := 109
  5 * roses_per_bouquet + 7 * x = total_roses → x = 12 :=
by
  intros
  sorry

end white_roses_per_table_decoration_l1963_196317


namespace quadratic_has_two_distinct_real_roots_l1963_196339

-- Define the quadratic equation and its coefficients
def a := 1
def b := -4
def c := -3

-- Define the discriminant function for a quadratic equation ax^2 + bx + c = 0
def discriminant (a b c : ℤ) : ℤ := b^2 - 4 * a * c

-- State the problem in Lean: Prove that the quadratic equation x^2 - 4x - 3 = 0 has a positive discriminant.
theorem quadratic_has_two_distinct_real_roots : discriminant a b c > 0 :=
by
  sorry -- This is where the proof would go

end quadratic_has_two_distinct_real_roots_l1963_196339


namespace f_zero_unique_l1963_196311

theorem f_zero_unique (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = f x + f (xy)) : f 0 = 0 :=
by {
  -- proof goes here
  sorry
}

end f_zero_unique_l1963_196311


namespace evaluate_expression_l1963_196392

variable (a : ℤ) (x : ℤ)

theorem evaluate_expression (h : x = a + 9) : x - a + 5 = 14 :=
by
  sorry

end evaluate_expression_l1963_196392


namespace number_of_girls_l1963_196352

variable (G B : ℕ)

theorem number_of_girls (h1 : G + B = 2000)
    (h2 : 0.28 * (B : ℝ) + 0.32 * (G : ℝ) = 596) : 
    G = 900 := 
sorry

end number_of_girls_l1963_196352


namespace correct_multiplication_l1963_196374

theorem correct_multiplication (n : ℕ) (h₁ : 15 * n = 45) : 5 * n = 15 :=
by
  -- skipping the proof
  sorry

end correct_multiplication_l1963_196374


namespace radar_placement_and_coverage_area_l1963_196312

noncomputable def max_distance (n : ℕ) (r : ℝ) (w : ℝ) : ℝ :=
  (15 : ℝ) / Real.sin (Real.pi / n)

noncomputable def coverage_area (n : ℕ) (r : ℝ) (w : ℝ) : ℝ :=
  (480 : ℝ) * Real.pi / Real.tan (Real.pi / n)

theorem radar_placement_and_coverage_area 
  (n : ℕ) (r w : ℝ) (hn : n = 8) (hr : r = 17) (hw : w = 16) :
  max_distance n r w = (15 : ℝ) / Real.sin (Real.pi / 8) ∧
  coverage_area n r w = (480 : ℝ) * Real.pi / Real.tan (Real.pi / 8) :=
by
  sorry

end radar_placement_and_coverage_area_l1963_196312


namespace quadratic_trinomial_constant_l1963_196370

theorem quadratic_trinomial_constant (m : ℝ) (h : |m| = 2) (h2 : m - 2 ≠ 0) : m = -2 :=
sorry

end quadratic_trinomial_constant_l1963_196370


namespace posters_total_l1963_196375

-- Definitions based on conditions
def Mario_posters : Nat := 18
def Samantha_posters : Nat := Mario_posters + 15

-- Statement to prove: They made 51 posters altogether
theorem posters_total : Mario_posters + Samantha_posters = 51 := 
by sorry

end posters_total_l1963_196375


namespace customers_sampling_candy_l1963_196315

theorem customers_sampling_candy (total_customers caught fined not_caught : ℝ) 
    (h1 : total_customers = 100) 
    (h2 : caught = 0.22 * total_customers) 
    (h3 : not_caught / (caught / 0.9) = 0.1) :
    (not_caught + caught) / total_customers = 0.2444 := 
by sorry

end customers_sampling_candy_l1963_196315


namespace proof_sin_315_eq_neg_sqrt_2_div_2_l1963_196335

noncomputable def sin_315_eq_neg_sqrt_2_div_2 : Prop :=
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2)

theorem proof_sin_315_eq_neg_sqrt_2_div_2 : sin_315_eq_neg_sqrt_2_div_2 := 
  by
    sorry

end proof_sin_315_eq_neg_sqrt_2_div_2_l1963_196335


namespace pyramid_base_side_length_l1963_196350

theorem pyramid_base_side_length (A : ℝ) (h : ℝ) (s : ℝ) :
  A = 120 ∧ h = 40 ∧ (A = 1 / 2 * s * h) → s = 6 :=
by
  intros
  sorry

end pyramid_base_side_length_l1963_196350


namespace power_mod_8_l1963_196349

theorem power_mod_8 (n : ℕ) (h : n % 2 = 0) : 3^n % 8 = 1 :=
by sorry

end power_mod_8_l1963_196349


namespace sum_of_squares_not_divisible_by_17_l1963_196362

theorem sum_of_squares_not_divisible_by_17
  (x y z : ℤ)
  (h_sum_div : 17 ∣ (x + y + z))
  (h_prod_div : 17 ∣ (x * y * z))
  (h_coprime_xy : Int.gcd x y = 1)
  (h_coprime_yz : Int.gcd y z = 1)
  (h_coprime_zx : Int.gcd z x = 1) :
  ¬ (17 ∣ (x^2 + y^2 + z^2)) := 
sorry

end sum_of_squares_not_divisible_by_17_l1963_196362


namespace picture_area_l1963_196347

theorem picture_area (x y : ℤ) (hx : 1 < x) (hy : 1 < y) (h : (x + 2) * (y + 4) = 45) : x * y = 15 := by
  sorry

end picture_area_l1963_196347


namespace square_side_length_l1963_196319

theorem square_side_length 
  (A B C D E : Type) 
  (AB AC hypotenuse square_side_length : ℝ) 
  (h1: AB = 9) 
  (h2: AC = 12) 
  (h3: hypotenuse = Real.sqrt (9^2 + 12^2)) 
  (h4: square_side_length = 300 / 41) 
  : square_side_length = 300 / 41 := 
by 
  sorry

end square_side_length_l1963_196319


namespace circle_center_sum_l1963_196336

theorem circle_center_sum (h k : ℝ) :
  (∃ h k : ℝ, ∀ x y : ℝ, (x^2 + y^2 = 6 * x + 8 * y - 15) → (h, k) = (3, 4)) →
  h + k = 7 :=
by
  sorry

end circle_center_sum_l1963_196336


namespace total_intersections_l1963_196302

def north_south_streets : ℕ := 10
def east_west_streets : ℕ := 10

theorem total_intersections :
  (north_south_streets * east_west_streets = 100) :=
by
  sorry

end total_intersections_l1963_196302


namespace exists_a_satisfying_inequality_l1963_196342

theorem exists_a_satisfying_inequality (x : ℝ) : 
  x < -2 ∨ (0 < x ∧ x < 1) ∨ 1 < x → 
  ∃ a ∈ Set.Icc (-1 : ℝ) 2, (2 - a) * x^3 + (1 - 2 * a) * x^2 - 6 * x + 5 + 4 * a - a^2 < 0 := 
by 
  intros h
  sorry

end exists_a_satisfying_inequality_l1963_196342


namespace exponent_equality_l1963_196369

theorem exponent_equality (n : ℕ) : (4^8 = 4^n) → (n = 8) := by
  intro h
  sorry

end exponent_equality_l1963_196369


namespace find_value_of_expression_l1963_196344

theorem find_value_of_expression (m n : ℝ) (h : |m - n - 5| + (2 * m + n - 4)^2 = 0) : 3 * m + n = 7 := 
sorry

end find_value_of_expression_l1963_196344


namespace max_fraction_l1963_196323

theorem max_fraction (a b : ℕ) (h1 : a + b = 101) (h2 : (a : ℚ) / b ≤ 1 / 3) : (a, b) = (25, 76) :=
sorry

end max_fraction_l1963_196323


namespace average_age_of_5_students_l1963_196379

theorem average_age_of_5_students
  (avg_age_20_students : ℕ → ℕ → ℕ → ℕ)
  (total_age_20 : avg_age_20_students 20 20 0 = 400)
  (total_age_9 : 9 * 16 = 144)
  (age_20th_student : ℕ := 186) :
  avg_age_20_students 5 ((400 - 144 - 186) / 5) 5 = 14 :=
by
  sorry

end average_age_of_5_students_l1963_196379


namespace percentage_female_officers_on_duty_l1963_196387

theorem percentage_female_officers_on_duty:
  ∀ (total_on_duty female_on_duty total_female_officers : ℕ),
    total_on_duty = 160 →
    female_on_duty = total_on_duty / 2 →
    total_female_officers = 500 →
    female_on_duty / total_female_officers * 100 = 16 :=
by
  intros total_on_duty female_on_duty total_female_officers h1 h2 h3
  -- Ensure types are correct
  change total_on_duty = 160 at h1
  change female_on_duty = total_on_duty / 2 at h2
  change total_female_officers = 500 at h3
  sorry

end percentage_female_officers_on_duty_l1963_196387


namespace simplify_expression_l1963_196391

variable (x : ℝ)

theorem simplify_expression :
  (2 * x * (4 * x^2 - 3) - 4 * (x^2 - 3 * x + 6)) = (8 * x^3 - 4 * x^2 + 6 * x - 24) := 
by 
  sorry

end simplify_expression_l1963_196391


namespace billy_distance_l1963_196343

-- Definitions
def distance_billy_spit (b : ℝ) : ℝ := b
def distance_madison_spit (m : ℝ) (b : ℝ) : Prop := m = 1.20 * b
def distance_ryan_spit (r : ℝ) (m : ℝ) : Prop := r = 0.50 * m

-- Conditions
variables (m : ℝ) (b : ℝ) (r : ℝ)
axiom madison_farther: distance_madison_spit m b
axiom ryan_shorter: distance_ryan_spit r m
axiom ryan_distance: r = 18

-- Proof problem
theorem billy_distance : b = 30 := by
  sorry

end billy_distance_l1963_196343


namespace circle_area_isosceles_triangle_l1963_196320

noncomputable def circle_area (a b c : ℝ) (is_isosceles : a = b ∧ (4 = a ∨ 4 = b) ∧ c = 3) : ℝ := sorry

theorem circle_area_isosceles_triangle :
  circle_area 4 4 3 ⟨rfl,Or.inl rfl, rfl⟩ = (64 / 13.75) * Real.pi := by
sorry

end circle_area_isosceles_triangle_l1963_196320


namespace dog_food_consumption_per_meal_l1963_196337

theorem dog_food_consumption_per_meal
  (dogs : ℕ) (meals_per_day : ℕ) (total_food_kg : ℕ) (days : ℕ)
  (h_dogs : dogs = 4) (h_meals_per_day : meals_per_day = 2)
  (h_total_food_kg : total_food_kg = 100) (h_days : days = 50) :
  (total_food_kg * 1000 / days / meals_per_day / dogs) = 250 :=
by
  sorry

end dog_food_consumption_per_meal_l1963_196337


namespace correct_statement_l1963_196372

-- Definition of quadrants
def is_second_quadrant (θ : ℝ) : Prop := 90 < θ ∧ θ < 180
def is_first_quadrant (θ : ℝ) : Prop := 0 < θ ∧ θ < 90
def is_third_quadrant (θ : ℝ) : Prop := -180 < θ ∧ θ < -90
def is_obtuse_angle (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- Statement of the problem
theorem correct_statement : is_obtuse_angle θ → is_second_quadrant θ :=
by sorry

end correct_statement_l1963_196372


namespace question1_question2_question3_l1963_196340

-- Question 1
theorem question1 (a b m n : ℤ) (h : a + b * Real.sqrt 5 = (m + n * Real.sqrt 5)^2) :
  a = m^2 + 5 * n^2 ∧ b = 2 * m * n :=
sorry

-- Question 2
theorem question2 (x m n: ℕ) (h : x + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) :
  (m = 1 ∧ n = 2 ∧ x = 13) ∨ (m = 2 ∧ n = 1 ∧ x = 7) :=
sorry

-- Question 3
theorem question3 : Real.sqrt (5 + 2 * Real.sqrt 6) = Real.sqrt 2 + Real.sqrt 3 :=
sorry

end question1_question2_question3_l1963_196340


namespace leoCurrentWeight_l1963_196332

def currentWeightProblem (L K : Real) : Prop :=
  (L + 15 = 1.75 * K) ∧ (L + K = 250)

theorem leoCurrentWeight (L K : Real) (h : currentWeightProblem L K) : L = 154 :=
by
  sorry

end leoCurrentWeight_l1963_196332


namespace min_cube_edge_division_l1963_196329

theorem min_cube_edge_division (n : ℕ) (h : n^3 ≥ 1996) : n = 13 :=
by {
  sorry
}

end min_cube_edge_division_l1963_196329


namespace function_range_l1963_196398

theorem function_range (f : ℝ → ℝ) (s : Set ℝ) (h : s = Set.Ico (-5 : ℝ) 2) (h_f : ∀ x ∈ s, f x = 3 * x - 1) :
  Set.image f s = Set.Ico (-16 : ℝ) 5 :=
sorry

end function_range_l1963_196398


namespace simplify_neg_expression_l1963_196377

variable (a b c : ℝ)

theorem simplify_neg_expression : 
  - (a - (b - c)) = -a + b - c :=
sorry

end simplify_neg_expression_l1963_196377


namespace simplify_and_evaluate_expression_l1963_196326

theorem simplify_and_evaluate_expression (x : ℤ) (hx : x = 3) : 
  (1 - (x / (x + 1))) / ((x^2 - 2 * x + 1) / (x^2 - 1)) = 1 / 2 := by
  rw [hx]
  -- Here we perform the necessary rewrites and simplifications as shown in the steps
  sorry

end simplify_and_evaluate_expression_l1963_196326


namespace orthocenter_of_triangle_ABC_l1963_196378

def point : Type := ℝ × ℝ × ℝ

def A : point := (2, 3, 4)
def B : point := (6, 4, 2)
def C : point := (4, 5, 6)

def orthocenter (A B C : point) : point := sorry -- We'll skip the function implementation here

theorem orthocenter_of_triangle_ABC :
  orthocenter A B C = (13/7, 41/14, 55/7) :=
sorry

end orthocenter_of_triangle_ABC_l1963_196378


namespace average_pages_per_book_l1963_196313

theorem average_pages_per_book :
  let pages := [120, 150, 180, 210, 240]
  let num_books := 5
  let total_pages := pages.sum
  total_pages / num_books = 180 := by
  sorry

end average_pages_per_book_l1963_196313


namespace complement_U_A_l1963_196304

def U : Set ℕ := {1, 3, 5, 7, 9}
def A : Set ℕ := {1, 5, 7}

theorem complement_U_A : (U \ A) = {3, 9} :=
by
  sorry

end complement_U_A_l1963_196304


namespace trig_identity_proof_l1963_196355

theorem trig_identity_proof :
  let sin240 := - (Real.sin (120 * Real.pi / 180))
  let tan240 := Real.tan (240 * Real.pi / 180)
  Real.sin (600 * Real.pi / 180) + tan240 = Real.sqrt 3 / 2 :=
by
  sorry

end trig_identity_proof_l1963_196355


namespace dice_sum_surface_l1963_196330

theorem dice_sum_surface (X : ℕ) (hX : 1 ≤ X ∧ X ≤ 6) : 
  ∃ Y : ℕ, Y = 28175 + 2 * X ∧ (Y = 28177 ∨ Y = 28179 ∨ Y = 28181 ∨ Y = 28183 ∨ 
  Y = 28185 ∨ Y = 28187) :=
by
  sorry

end dice_sum_surface_l1963_196330


namespace cistern_empty_time_l1963_196345

theorem cistern_empty_time
  (fill_time_without_leak : ℝ := 4)
  (additional_time_due_to_leak : ℝ := 2) :
  (1 / (fill_time_without_leak + additional_time_due_to_leak - fill_time_without_leak / fill_time_without_leak)) = 12 :=
by
  sorry

end cistern_empty_time_l1963_196345


namespace Michael_made_97_dollars_l1963_196305

def price_large : ℕ := 22
def price_medium : ℕ := 16
def price_small : ℕ := 7

def quantity_large : ℕ := 2
def quantity_medium : ℕ := 2
def quantity_small : ℕ := 3

def calculate_total_money (price_large price_medium price_small : ℕ) 
                           (quantity_large quantity_medium quantity_small : ℕ) : ℕ :=
  (price_large * quantity_large) + (price_medium * quantity_medium) + (price_small * quantity_small)

theorem Michael_made_97_dollars :
  calculate_total_money price_large price_medium price_small quantity_large quantity_medium quantity_small = 97 := 
by
  sorry

end Michael_made_97_dollars_l1963_196305


namespace sandy_age_l1963_196399

variable (S M N : ℕ)

theorem sandy_age (h1 : M = S + 20)
                  (h2 : (S : ℚ) / M = 7 / 9)
                  (h3 : S + M + N = 120)
                  (h4 : N - M = (S - M) / 2) :
                  S = 70 := 
sorry

end sandy_age_l1963_196399


namespace difference_of_cubes_not_div_by_twice_diff_l1963_196308

theorem difference_of_cubes_not_div_by_twice_diff (a b : ℤ) (h_a : a % 2 = 1) (h_b : b % 2 = 1) (h_neq : a ≠ b) :
  ¬ (2 * (a - b)) ∣ ((a^3) - (b^3)) := 
sorry

end difference_of_cubes_not_div_by_twice_diff_l1963_196308


namespace min_value_l1963_196396

-- Definition of the conditions
def positive (a : ℝ) : Prop := a > 0

theorem min_value (a : ℝ) (h : positive a) : 
  ∃ m : ℝ, (m = 2 * Real.sqrt 6) ∧ (∀ x : ℝ, positive x → (3 / (2 * x) + 4 * x) ≥ m) :=
sorry

end min_value_l1963_196396


namespace solution_l1963_196394

namespace Proof

open Set

def proof_problem : Prop :=
  let U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {3, 4, 5, 6}
  A ∩ (U \ B) = {1, 2}

theorem solution : proof_problem := by
  -- The pre-defined proof_problem must be shown here
  -- Proof: sorry
  sorry

end Proof

end solution_l1963_196394


namespace rectangle_width_l1963_196309

theorem rectangle_width (L W : ℝ) 
  (h1 : L * W = 750) 
  (h2 : 2 * L + 2 * W = 110) : 
  W = 25 :=
sorry

end rectangle_width_l1963_196309


namespace no_perfect_square_solution_l1963_196390

theorem no_perfect_square_solution (n : ℕ) (x : ℕ) (hx : x < 10^n) :
  ¬ (∀ y, 0 ≤ y ∧ y ≤ 9 → ∃ z : ℤ, ∃ k : ℤ, 10^(n+1) * z + 10 * x + y = k^2) :=
sorry

end no_perfect_square_solution_l1963_196390


namespace probability_heads_exactly_8_in_10_l1963_196384

def fair_coin_probability (n k : ℕ) : ℚ := (Nat.choose n k : ℚ) / (2 ^ n)

theorem probability_heads_exactly_8_in_10 :
  fair_coin_probability 10 8 = 45 / 1024 :=
by 
  sorry

end probability_heads_exactly_8_in_10_l1963_196384


namespace betty_age_l1963_196301

-- Define the constants and conditions
variables (A M B : ℕ)
variables (h1 : A = 2 * M) (h2 : A = 4 * B) (h3 : M = A - 8)

-- Define the theorem to prove Betty's age
theorem betty_age : B = 4 :=
by sorry

end betty_age_l1963_196301


namespace sum_of_ages_l1963_196318

variable {P M Mo : ℕ}

-- Conditions
axiom ratio1 : 3 * M = 5 * P
axiom ratio2 : 3 * Mo = 5 * M
axiom age_difference : Mo - P = 80

-- Statement that needs to be proved
theorem sum_of_ages : P + M + Mo = 245 := by
  sorry

end sum_of_ages_l1963_196318


namespace function_identity_l1963_196365

theorem function_identity
    (f : ℝ → ℝ)
    (h1 : ∀ x : ℝ, f x ≤ x)
    (h2 : ∀ x y : ℝ, f (x + y) ≤ f x + f y) :
    ∀ x : ℝ, f x = x :=
by
    sorry

end function_identity_l1963_196365


namespace cats_not_eating_either_l1963_196300

/-- In a shelter with 80 cats, 15 cats like tuna, 60 cats like chicken, 
and 10 like both tuna and chicken, prove that 15 cats do not eat either. -/
theorem cats_not_eating_either (total_cats : ℕ) (like_tuna : ℕ) (like_chicken : ℕ) (like_both : ℕ)
    (h1 : total_cats = 80) (h2 : like_tuna = 15) (h3 : like_chicken = 60) (h4 : like_both = 10) :
    (total_cats - (like_tuna - like_both + like_chicken - like_both + like_both) = 15) := 
by
    sorry

end cats_not_eating_either_l1963_196300


namespace original_mixture_volume_l1963_196359

theorem original_mixture_volume (x : ℝ) (h1 : 0.20 * x / (x + 3) = 1 / 6) : x = 15 :=
  sorry

end original_mixture_volume_l1963_196359


namespace jasmine_percentage_after_adding_l1963_196331

def initial_solution_volume : ℕ := 80
def initial_jasmine_percentage : ℝ := 0.10
def additional_jasmine_volume : ℕ := 5
def additional_water_volume : ℕ := 15

theorem jasmine_percentage_after_adding :
  let initial_jasmine_volume := initial_jasmine_percentage * initial_solution_volume
  let total_jasmine_volume := initial_jasmine_volume + additional_jasmine_volume
  let total_solution_volume := initial_solution_volume + additional_jasmine_volume + additional_water_volume
  let final_jasmine_percentage := (total_jasmine_volume / total_solution_volume) * 100
  final_jasmine_percentage = 13 := by
  sorry

end jasmine_percentage_after_adding_l1963_196331


namespace seventh_observation_l1963_196325

theorem seventh_observation (avg6 : ℕ) (new_avg7 : ℕ) (old_avg : ℕ) (new_avg_diff : ℕ) (n : ℕ) (m : ℕ) (h1 : avg6 = 12) (h2 : new_avg_diff = 1) (h3 : n = 6) (h4 : m = 7) :
  ((n * old_avg = avg6 * old_avg) ∧ (m * new_avg7 = avg6 * old_avg + m - n)) →
  m * new_avg7 = 77 →
  avg6 * old_avg = 72 →
  77 - 72 = 5 :=
by
  sorry

end seventh_observation_l1963_196325


namespace parallelogram_sides_are_parallel_l1963_196353

theorem parallelogram_sides_are_parallel 
  {a b c : ℤ} (h_area : c * (a^2 + b^2) = 2011 * b) : 
  (∃ k : ℤ, a = 2011 * k ∧ (b = 2011 ∨ b = -2011)) :=
by
  sorry

end parallelogram_sides_are_parallel_l1963_196353


namespace x_plus_y_equals_six_l1963_196334

theorem x_plus_y_equals_six (x y : ℝ) (h₁ : y - x = 1) (h₂ : y^2 = x^2 + 6) : x + y = 6 :=
by
  sorry

end x_plus_y_equals_six_l1963_196334


namespace arithmetic_sequence_terms_l1963_196361

theorem arithmetic_sequence_terms (a : ℕ → ℝ) (n : ℕ) (S : ℕ → ℝ) 
  (h2 : a 1 + a 2 + a 3 = 34)
  (h3 : a n + a (n-1) + a (n-2) = 146)
  (h4 : S n = 390)
  (h5 : ∀ i j, a i + a j = a (i+1) + a (j-1)) :
  n = 13 :=
sorry

end arithmetic_sequence_terms_l1963_196361


namespace minimum_value_problem_l1963_196356

open Real

theorem minimum_value_problem (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h_sum : x + y + z = 6) :
  9 / x + 16 / y + 25 / z ≥ 24 :=
by
  sorry

end minimum_value_problem_l1963_196356


namespace abs_val_of_5_minus_e_l1963_196364

theorem abs_val_of_5_minus_e : ∀ (e : ℝ), e = 2.718 → |5 - e| = 2.282 :=
by
  intros e he
  sorry

end abs_val_of_5_minus_e_l1963_196364


namespace find_original_price_l1963_196306

-- Define the conditions provided in the problem
def original_price (P : ℝ) : Prop :=
  let first_discount := 0.90 * P
  let second_discount := 0.85 * first_discount
  let taxed_price := 1.08 * second_discount
  taxed_price = 450

-- State and prove the main theorem
theorem find_original_price (P : ℝ) (h : original_price P) : P = 544.59 :=
  sorry

end find_original_price_l1963_196306


namespace fraction_lt_sqrt2_bound_l1963_196376

theorem fraction_lt_sqrt2_bound (m n : ℕ) (h : (m : ℝ) / n < Real.sqrt 2) :
  (m : ℝ) / n < Real.sqrt 2 * (1 - 1 / (4 * (n * n))) :=
sorry

end fraction_lt_sqrt2_bound_l1963_196376


namespace muffin_half_as_expensive_as_banana_l1963_196383

-- Define Susie's expenditure in terms of muffin cost (m) and banana cost (b)
def susie_expenditure (m b : ℝ) : ℝ := 5 * m + 2 * b

-- Define Calvin's expenditure as three times Susie's expenditure
def calvin_expenditure_via_susie (m b : ℝ) : ℝ := 3 * (susie_expenditure m b)

-- Define Calvin's direct expenditure on muffins and bananas
def calvin_direct_expenditure (m b : ℝ) : ℝ := 3 * m + 12 * b

-- Formulate the theorem stating the relationship between muffin and banana costs
theorem muffin_half_as_expensive_as_banana (m b : ℝ) 
  (h₁ : susie_expenditure m b = 5 * m + 2 * b)
  (h₂ : calvin_expenditure_via_susie m b = calvin_direct_expenditure m b) : 
  m = (1/2) * b := 
by {
  -- These conditions automatically fulfill the given problem requirements.
  sorry
}

end muffin_half_as_expensive_as_banana_l1963_196383


namespace factor_sum_l1963_196333

theorem factor_sum : 
  (∃ d e, x^2 + 9 * x + 20 = (x + d) * (x + e)) ∧ 
  (∃ e f, x^2 - x - 56 = (x + e) * (x - f)) → 
  ∃ d e f, d + e + f = 19 :=
by
  sorry

end factor_sum_l1963_196333


namespace spinner_prob_C_l1963_196382

theorem spinner_prob_C (P_A P_B P_C : ℚ) (h_A : P_A = 1/3) (h_B : P_B = 5/12) (h_total : P_A + P_B + P_C = 1) : 
  P_C = 1/4 := 
sorry

end spinner_prob_C_l1963_196382


namespace david_account_amount_l1963_196351

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem david_account_amount : compound_interest 5000 0.06 2 1 = 5304.50 := by
  sorry

end david_account_amount_l1963_196351


namespace solve_for_k_l1963_196385

theorem solve_for_k (x k : ℝ) (h : k ≠ 0) 
(h_eq : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 7)) : k = 7 :=
by
  -- Proof would go here
  sorry

end solve_for_k_l1963_196385


namespace Kenny_running_to_basketball_ratio_l1963_196386

theorem Kenny_running_to_basketball_ratio (basketball_hours trumpet_hours running_hours : ℕ) 
    (h1 : basketball_hours = 10)
    (h2 : trumpet_hours = 2 * running_hours)
    (h3 : trumpet_hours = 40) :
    running_hours = 20 ∧ basketball_hours = 10 ∧ (running_hours / basketball_hours = 2) :=
by
  sorry

end Kenny_running_to_basketball_ratio_l1963_196386


namespace tom_total_payment_l1963_196346

variable (apples_kg : ℕ := 8)
variable (apples_rate : ℕ := 70)
variable (mangoes_kg : ℕ := 9)
variable (mangoes_rate : ℕ := 65)
variable (oranges_kg : ℕ := 5)
variable (oranges_rate : ℕ := 50)
variable (bananas_kg : ℕ := 3)
variable (bananas_rate : ℕ := 30)
variable (discount_apples : ℝ := 0.10)
variable (discount_oranges : ℝ := 0.15)

def total_cost_apple : ℝ := apples_kg * apples_rate
def total_cost_mango : ℝ := mangoes_kg * mangoes_rate
def total_cost_orange : ℝ := oranges_kg * oranges_rate
def total_cost_banana : ℝ := bananas_kg * bananas_rate
def discount_apples_amount : ℝ := discount_apples * total_cost_apple
def discount_oranges_amount : ℝ := discount_oranges * total_cost_orange
def apples_after_discount : ℝ := total_cost_apple - discount_apples_amount
def oranges_after_discount : ℝ := total_cost_orange - discount_oranges_amount

theorem tom_total_payment :
  apples_after_discount + total_cost_mango + oranges_after_discount + total_cost_banana = 1391.5 := by
  sorry

end tom_total_payment_l1963_196346


namespace temperature_decrease_l1963_196348

theorem temperature_decrease (rise_1_degC : ℝ) (decrease_2_degC : ℝ) 
  (h : rise_1_degC = 1) : decrease_2_degC = -2 :=
by 
  -- This is the statement with the condition and problem to be proven:
  sorry

end temperature_decrease_l1963_196348


namespace area_of_sector_l1963_196395

theorem area_of_sector (r : ℝ) (theta : ℝ) (h_r : r = 6) (h_theta : theta = 60) : (θ / 360 * π * r^2 = 6 * π) :=
by sorry

end area_of_sector_l1963_196395


namespace find_yellow_shells_l1963_196324

-- Define the conditions
def total_shells : ℕ := 65
def purple_shells : ℕ := 13
def pink_shells : ℕ := 8
def blue_shells : ℕ := 12
def orange_shells : ℕ := 14

-- Define the result as the proof goal
theorem find_yellow_shells (total_shells purple_shells pink_shells blue_shells orange_shells : ℕ) : 
  total_shells = 65 →
  purple_shells = 13 →
  pink_shells = 8 →
  blue_shells = 12 →
  orange_shells = 14 →
  65 - (13 + 8 + 12 + 14) = 18 :=
by
  intros
  sorry

end find_yellow_shells_l1963_196324


namespace minimum_value_of_f_l1963_196368

noncomputable def f (x : ℝ) : ℝ := x + (1 / x) - 2

theorem minimum_value_of_f : ∃ x : ℝ, x > 0 ∧ f x = 0 :=
by
  sorry

end minimum_value_of_f_l1963_196368


namespace find_difference_l1963_196389

theorem find_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 := 
by
  sorry

end find_difference_l1963_196389


namespace population_time_interval_l1963_196338

theorem population_time_interval (T : ℕ) 
  (birth_rate : ℕ) (death_rate : ℕ) (net_increase_day : ℕ) (seconds_in_day : ℕ)
  (h_birth_rate : birth_rate = 8) 
  (h_death_rate : death_rate = 6) 
  (h_net_increase_day : net_increase_day = 86400)
  (h_seconds_in_day : seconds_in_day = 86400) : 
  T = 2 := sorry

end population_time_interval_l1963_196338


namespace no_solution_intervals_l1963_196307

theorem no_solution_intervals (a : ℝ) :
  (a < -13 ∨ a > 0) → ¬ ∃ x : ℝ, 6 * abs (x - 4 * a) + abs (x - a^2) + 5 * x - 3 * a = 0 :=
by sorry

end no_solution_intervals_l1963_196307


namespace wendy_distance_difference_l1963_196367

-- Defining the distances ran and walked by Wendy
def distance_ran : ℝ := 19.83
def distance_walked : ℝ := 9.17

-- The theorem to prove the difference in distance
theorem wendy_distance_difference : distance_ran - distance_walked = 10.66 := by
  -- Proof goes here
  sorry

end wendy_distance_difference_l1963_196367


namespace games_bought_l1963_196397

/-- 
Given:
1. Geoffrey received €20 from his grandmother.
2. Geoffrey received €25 from his aunt.
3. Geoffrey received €30 from his uncle.
4. Geoffrey now has €125 in his wallet.
5. Geoffrey has €20 left after buying games.
6. Each game costs €35.

Prove that Geoffrey bought 3 games.
-/
theorem games_bought 
  (grandmother_money aunt_money uncle_money total_money left_money game_cost spent_money games_bought : ℤ)
  (h1 : grandmother_money = 20)
  (h2 : aunt_money = 25)
  (h3 : uncle_money = 30)
  (h4 : total_money = 125)
  (h5 : left_money = 20)
  (h6 : game_cost = 35)
  (h7 : spent_money = total_money - left_money)
  (h8 : games_bought = spent_money / game_cost) :
  games_bought = 3 := 
sorry

end games_bought_l1963_196397


namespace intersection_of_M_and_N_l1963_196373

def M := {x : ℝ | 3 * x - x^2 > 0}
def N := {x : ℝ | x^2 - 4 * x + 3 > 0}
def I := {x : ℝ | 0 < x ∧ x < 1}

theorem intersection_of_M_and_N : M ∩ N = I :=
by
  sorry

end intersection_of_M_and_N_l1963_196373


namespace quadratic_inequality_solution_set_l1963_196380

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2 * x^2 - x - 3 ≥ 0} = {x : ℝ | x ≤ -1 ∨ x ≥ 3 / 2} :=
sorry

end quadratic_inequality_solution_set_l1963_196380


namespace person_A_takes_12_more_minutes_l1963_196321

-- Define distances, speeds, times
variables (S : ℝ) (v_A v_B : ℝ) (t : ℝ)

-- Define conditions as hypotheses
def conditions (h1 : t = 2/5) (h2 : v_A = (2/3) * S / (t + 4/5)) (h3 : v_B = (2/3) * S / t) : Prop :=
  (v_A * (t + 4/5) = 2/3 * S) ∧ (v_B * t = 2/3 * S) ∧ (v_A * (t + 4/5 + 1/2 * t + 1/10) + 1/10 * v_B = S)

-- The proof problem statement
theorem person_A_takes_12_more_minutes
  (S : ℝ) (v_A v_B : ℝ) (t : ℝ)
  (h1 : t = 2/5) (h2 : v_A = (2/3) * S / (t + 4/5)) (h3 : v_B = (2/3) * S / t)
  (h4 : conditions S v_A v_B t h1 h2 h3) : (t + 4/5) + 6/5 = 96 / 60 + 12 / 60 :=
sorry

end person_A_takes_12_more_minutes_l1963_196321


namespace quadrant_of_tan_and_cos_l1963_196314

theorem quadrant_of_tan_and_cos (α : ℝ) (h1 : Real.tan α < 0) (h2 : Real.cos α < 0) : 
  ∃ Q, (Q = 2) :=
by
  sorry


end quadrant_of_tan_and_cos_l1963_196314


namespace expand_product_l1963_196316

theorem expand_product (x : ℝ) : (x + 5) * (x + 9) = x^2 + 14 * x + 45 :=
by
  sorry

end expand_product_l1963_196316


namespace difference_in_money_in_cents_l1963_196360

theorem difference_in_money_in_cents (p : ℤ) (h₁ : ℤ) (h₂ : ℤ) 
  (h₁ : Linda_nickels = 7 * p - 2) (h₂ : Carol_nickels = 3 * p + 4) :
  5 * (Linda_nickels - Carol_nickels) = 20 * p - 30 := 
by sorry

end difference_in_money_in_cents_l1963_196360


namespace parabola_equation_l1963_196358

theorem parabola_equation (h k a : ℝ) (same_shape : ∀ x, -2 * x^2 + 2 = a * x^2 + k) (vertex : h = 4 ∧ k = -2) :
  ∀ x, -2 * (x - 4)^2 - 2 = a * (x - h)^2 + k :=
by
  -- This is where the actual proof would go
  simp
  sorry

end parabola_equation_l1963_196358


namespace find_initial_avg_height_l1963_196381

noncomputable def initially_calculated_avg_height (A : ℚ) (boys : ℕ) (wrong_height right_height : ℚ) (actual_avg_height : ℚ) :=
  boys = 35 ∧
  wrong_height = 166 ∧
  right_height = 106 ∧
  actual_avg_height = 182 ∧
  35 * A - (wrong_height - right_height) = 35 * actual_avg_height

theorem find_initial_avg_height : ∃ A : ℚ, initially_calculated_avg_height A 35 166 106 182 ∧ A = 183.71 :=
by
  sorry

end find_initial_avg_height_l1963_196381


namespace equal_sum_sequence_a18_l1963_196366

theorem equal_sum_sequence_a18
    (a : ℕ → ℕ)
    (h1 : a 1 = 2)
    (h2 : ∀ n, a n + a (n + 1) = 5) :
    a 18 = 3 :=
sorry

end equal_sum_sequence_a18_l1963_196366


namespace age_of_youngest_child_l1963_196328

theorem age_of_youngest_child (x : ℕ) 
  (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 50) : x = 4 := 
by {
  sorry
}

end age_of_youngest_child_l1963_196328


namespace math_proof_l1963_196310

open Real

noncomputable def function (a b x : ℝ): ℝ := a * x^3 + b * x^2

theorem math_proof (a b : ℝ) :
  (function a b 1 = 3) ∧
  (deriv (function a b) 1 = 0) ∧
  (∃ (a b : ℝ), a = -6 ∧ b = 9 ∧ 
    function a b = -6 * (x^3) + 9 * (x^2)) ∧
  (∀ x, (0 < x ∧ x < 1) → deriv (function a b) x > 0) ∧
  (∀ x, (x < 0 ∨ x > 1) → deriv (function a b) x < 0) ∧
  (min (function a b (-2)) (function a b 2) = (-12)) ∧
  (max (function a b (-2)) (function a b 2) = 84) :=
by
  sorry

end math_proof_l1963_196310


namespace part1_part2_l1963_196388

open Classical

theorem part1 (x : ℝ) (a : ℝ) (b : ℝ) :
  (a = 1) ∧ (b = 2) ∧ (∀ x, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) :=
by
  sorry

theorem part2 (x y k : ℝ) (a b : ℝ) :
  a = 1 ∧ b = 2 ∧ (x > 0) ∧ (y > 0) ∧ (1 / x + 2 / y = 1) ∧ (2 * x + y ≥ k^2 + k + 2) → -3 ≤ k ∧ k ≤ 2 :=
by
  sorry

end part1_part2_l1963_196388


namespace present_age_of_son_l1963_196363

theorem present_age_of_son (S F : ℕ) (h1 : F = S + 25) (h2 : F + 2 = 2 * (S + 2)) : S = 23 :=
by
  sorry

end present_age_of_son_l1963_196363


namespace frog_jump_paths_l1963_196322

noncomputable def φ : ℕ × ℕ → ℕ
| (0, 0) => 1
| (x, y) =>
  let φ_x1 := if x > 1 then φ (x - 1, y) else 0
  let φ_x2 := if x > 1 then φ (x - 2, y) else 0
  let φ_y1 := if y > 1 then φ (x, y - 1) else 0
  let φ_y2 := if y > 1 then φ (x, y - 2) else 0
  φ_x1 + φ_x2 + φ_y1 + φ_y2

theorem frog_jump_paths : φ (4, 4) = 556 := sorry

end frog_jump_paths_l1963_196322


namespace joan_books_l1963_196354

theorem joan_books : 
  (33 - 26 = 7) :=
by
  sorry

end joan_books_l1963_196354


namespace select_eight_genuine_dinars_l1963_196371

theorem select_eight_genuine_dinars (coins : Fin 11 → ℝ) :
  (∃ (fake_coin : Option (Fin 11)), 
    ((∀ i j : Fin 11, i ≠ j → coins i = coins j) ∨
    (∀ (genuine_coins impostor_coins : Finset (Fin 11)), 
      genuine_coins ∪ impostor_coins = Finset.univ →
      impostor_coins.card = 1 →
      (∃ difference : ℝ, ∀ i ∈ genuine_coins, coins i = difference) ∧
      (∃ i ∈ impostor_coins, coins i ≠ difference)))) →
  (∃ (selected_coins : Finset (Fin 11)), selected_coins.card = 8 ∧
   (∀ i j : Fin 11, i ∈ selected_coins → j ∈ selected_coins → coins i = coins j)) :=
sorry

end select_eight_genuine_dinars_l1963_196371
