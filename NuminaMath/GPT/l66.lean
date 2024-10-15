import Mathlib

namespace NUMINAMATH_GPT_trip_distance_first_part_l66_6602

theorem trip_distance_first_part (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 70) (h3 : 32 = 70 / ((x / 48) + ((70 - x) / 24))) : x = 35 :=
by
  sorry

end NUMINAMATH_GPT_trip_distance_first_part_l66_6602


namespace NUMINAMATH_GPT_point_in_second_quadrant_l66_6678

variable (m : ℝ)

-- Defining the conditions
def x_negative (m : ℝ) := 3 - m < 0
def y_positive (m : ℝ) := m - 1 > 0

theorem point_in_second_quadrant (h1 : x_negative m) (h2 : y_positive m) : m > 3 :=
by
  sorry

end NUMINAMATH_GPT_point_in_second_quadrant_l66_6678


namespace NUMINAMATH_GPT_ratio_Rose_to_Mother_l66_6636

variable (Rose_age : ℕ) (Mother_age : ℕ)

-- Define the conditions
axiom sum_of_ages : Rose_age + Mother_age = 100
axiom Rose_is_25 : Rose_age = 25
axiom Mother_is_75 : Mother_age = 75

-- Define the main theorem to prove the ratio
theorem ratio_Rose_to_Mother : (Rose_age : ℚ) / (Mother_age : ℚ) = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_ratio_Rose_to_Mother_l66_6636


namespace NUMINAMATH_GPT_polynomial_expansion_l66_6608

theorem polynomial_expansion (x : ℝ) :
  (x - 2) * (x + 2) * (x^2 + 4 * x + 4) = x^4 + 4 * x^3 - 16 * x - 16 :=
by sorry

end NUMINAMATH_GPT_polynomial_expansion_l66_6608


namespace NUMINAMATH_GPT_amount_returned_l66_6693

theorem amount_returned (deposit_usd : ℝ) (exchange_rate : ℝ) (h1 : deposit_usd = 10000) (h2 : exchange_rate = 58.15) : 
  deposit_usd * exchange_rate = 581500 := 
by 
  sorry

end NUMINAMATH_GPT_amount_returned_l66_6693


namespace NUMINAMATH_GPT_minimize_distance_on_ellipse_l66_6600

theorem minimize_distance_on_ellipse (a m n : ℝ) (hQ : 0 < a ∧ a ≠ Real.sqrt 3)
  (hP : m^2 / 3 + n^2 / 2 = 1) :
  |minimize_distance| = Real.sqrt 3 ∨ |minimize_distance| = 3 * a := sorry

end NUMINAMATH_GPT_minimize_distance_on_ellipse_l66_6600


namespace NUMINAMATH_GPT_range_of_f_l66_6655

def f (x : Int) : Int :=
  x + 1

def domain : Set Int :=
  {-1, 1, 2}

theorem range_of_f :
  Set.image f domain = {0, 2, 3} :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l66_6655


namespace NUMINAMATH_GPT_added_number_after_doubling_l66_6624

theorem added_number_after_doubling (x y : ℤ) (h1 : x = 4) (h2 : 3 * (2 * x + y) = 51) : y = 9 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_added_number_after_doubling_l66_6624


namespace NUMINAMATH_GPT_quiz_sum_correct_l66_6604

theorem quiz_sum_correct (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x > y) (h_sub : x - y = 4) (h_mul : x * y = 104) :
  x + y = 20 := by
  sorry

end NUMINAMATH_GPT_quiz_sum_correct_l66_6604


namespace NUMINAMATH_GPT_GouguPrinciple_l66_6695

-- Definitions according to conditions
def volumes_not_equal (A B : Type) : Prop := sorry -- p: volumes of A and B are not equal
def cross_sections_not_equal (A B : Type) : Prop := sorry -- q: cross-sectional areas of A and B are not always equal

-- The theorem to be proven
theorem GouguPrinciple (A B : Type) (h1 : volumes_not_equal A B) : cross_sections_not_equal A B :=
sorry

end NUMINAMATH_GPT_GouguPrinciple_l66_6695


namespace NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l66_6671

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∃ n, is_three_digit n ∧ 17 ∣ n ∧ ∀ k, is_three_digit k ∧ 17 ∣ k → k ≤ n :=
by
  sorry

end NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l66_6671


namespace NUMINAMATH_GPT_monotonic_increase_l66_6684

noncomputable def f (x : ℝ) : ℝ := Real.log (x - 2)

theorem monotonic_increase : ∀ x1 x2 : ℝ, 2 < x1 → x1 < x2 → f x1 < f x2 :=
by
  sorry

end NUMINAMATH_GPT_monotonic_increase_l66_6684


namespace NUMINAMATH_GPT_gcd_16_12_eq_4_l66_6631

theorem gcd_16_12_eq_4 : Int.gcd 16 12 = 4 := by
  sorry

end NUMINAMATH_GPT_gcd_16_12_eq_4_l66_6631


namespace NUMINAMATH_GPT_product_of_real_roots_l66_6625

theorem product_of_real_roots (x : ℝ) (hx : x ^ (Real.log x / Real.log 5) = 5) :
  (∃ a b : ℝ, a ^ (Real.log a / Real.log 5) = 5 ∧ b ^ (Real.log b / Real.log 5) = 5 ∧ a * b = 1) :=
sorry

end NUMINAMATH_GPT_product_of_real_roots_l66_6625


namespace NUMINAMATH_GPT_fraction_simplification_l66_6687

theorem fraction_simplification :
  (3 / (2 - (3 / 4))) = 12 / 5 := 
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l66_6687


namespace NUMINAMATH_GPT_system_solution_unique_l66_6640

theorem system_solution_unique
  (a b m n : ℝ)
  (h1 : a * 1 + b * 2 = 10)
  (h2 : m * 1 - n * 2 = 8) :
  (a / 2 * (4 + -2) + b / 3 * (4 - -2) = 10) ∧
  (m / 2 * (4 + -2) - n / 3 * (4 - -2) = 8) := 
  by
    sorry

end NUMINAMATH_GPT_system_solution_unique_l66_6640


namespace NUMINAMATH_GPT_g_2002_value_l66_6656

noncomputable def g : ℕ → ℤ := sorry

theorem g_2002_value :
  (∀ a b n : ℕ, a + b = 2^n → g a + g b = n^3) →
  (g 2 + g 46 = 180) →
  g 2002 = 1126 := 
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_g_2002_value_l66_6656


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l66_6621

def M : Set ℝ := {x | x ≥ 0 ∧ x < 16}
def N : Set ℝ := {x | x ≥ 1/3}

theorem intersection_of_M_and_N :
  M ∩ N = {x | 1/3 ≤ x ∧ x < 16} := by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l66_6621


namespace NUMINAMATH_GPT_max_candies_l66_6607

/-- There are 28 ones written on the board. Every minute, Karlsson erases two arbitrary numbers
and writes their sum on the board, and then eats an amount of candy equal to the product of 
the two erased numbers. Prove that the maximum number of candies he could eat in 28 minutes is 378. -/
theorem max_candies (karlsson_eats_max_candies : ℕ → ℕ → ℕ) (n : ℕ) (initial_count : n = 28) :
  (∀ a b, karlsson_eats_max_candies a b = a * b) →
  (∃ max_candies, max_candies = 378) :=
sorry

end NUMINAMATH_GPT_max_candies_l66_6607


namespace NUMINAMATH_GPT_new_recipe_water_l66_6637

theorem new_recipe_water (flour water sugar : ℕ)
  (h_orig : flour = 10 ∧ water = 6 ∧ sugar = 3)
  (h_new : ∀ (new_flour new_water new_sugar : ℕ), 
            new_flour = 10 ∧ new_water = 3 ∧ new_sugar = 3)
  (h_sugar : sugar = 4) :
  new_water = 4 := 
  sorry

end NUMINAMATH_GPT_new_recipe_water_l66_6637


namespace NUMINAMATH_GPT_intersection_of_S_and_T_l66_6688

-- Define S and T based on given conditions
def S : Set ℝ := { x | x^2 + 2 * x = 0 }
def T : Set ℝ := { x | x^2 - 2 * x = 0 }

-- Prove the intersection of S and T
theorem intersection_of_S_and_T : S ∩ T = {0} :=
sorry

end NUMINAMATH_GPT_intersection_of_S_and_T_l66_6688


namespace NUMINAMATH_GPT_time_saved_l66_6616

theorem time_saved (speed_with_tide distance1 time1 distance2 time2: ℝ) 
  (h1: speed_with_tide = 5) 
  (h2: distance1 = 5) 
  (h3: time1 = 1) 
  (h4: distance2 = 40) 
  (h5: time2 = 10) : 
  time2 - (distance2 / speed_with_tide) = 2 := 
sorry

end NUMINAMATH_GPT_time_saved_l66_6616


namespace NUMINAMATH_GPT_peanut_butter_candy_count_l66_6669

theorem peanut_butter_candy_count (B G P : ℕ) 
  (hB : B = 43)
  (hG : G = B + 5)
  (hP : P = 4 * G) :
  P = 192 := by
  sorry

end NUMINAMATH_GPT_peanut_butter_candy_count_l66_6669


namespace NUMINAMATH_GPT_average_sqft_per_person_texas_l66_6654

theorem average_sqft_per_person_texas :
  let population := 17000000
  let area_sqmiles := 268596
  let usable_land_percentage := 0.8
  let sqfeet_per_sqmile := 5280 * 5280
  let total_sqfeet := area_sqmiles * sqfeet_per_sqmile
  let usable_sqfeet := usable_land_percentage * total_sqfeet
  let avg_sqfeet_per_person := usable_sqfeet / population
  352331 <= avg_sqfeet_per_person ∧ avg_sqfeet_per_person < 500000 :=
by
  sorry

end NUMINAMATH_GPT_average_sqft_per_person_texas_l66_6654


namespace NUMINAMATH_GPT_journey_distance_l66_6612

theorem journey_distance 
  (T : ℝ) 
  (s1 s2 s3 : ℝ) 
  (hT : T = 36) 
  (hs1 : s1 = 21)
  (hs2 : s2 = 45)
  (hs3 : s3 = 24) : ∃ (D : ℝ), D = 972 :=
  sorry

end NUMINAMATH_GPT_journey_distance_l66_6612


namespace NUMINAMATH_GPT_r_minus_p_value_l66_6680

theorem r_minus_p_value (p q r : ℝ)
  (h₁ : (p + q) / 2 = 10)
  (h₂ : (q + r) / 2 = 22) :
  r - p = 24 :=
by
  sorry

end NUMINAMATH_GPT_r_minus_p_value_l66_6680


namespace NUMINAMATH_GPT_calc_1_calc_2_calc_3_calc_4_l66_6627

-- Problem 1
theorem calc_1 : 26 - 7 + (-6) + 17 = 30 := 
by
  sorry

-- Problem 2
theorem calc_2 : -81 / (9 / 4) * (-4 / 9) / (-16) = -1 := 
by
  sorry

-- Problem 3
theorem calc_3 : ((2 / 3) - (3 / 4) + (1 / 6)) * (-36) = -3 := 
by
  sorry

-- Problem 4
theorem calc_4 : -1^4 + 12 / (-2)^2 + (1 / 4) * (-8) = 0 := 
by
  sorry


end NUMINAMATH_GPT_calc_1_calc_2_calc_3_calc_4_l66_6627


namespace NUMINAMATH_GPT_real_solutions_to_system_l66_6679

theorem real_solutions_to_system :
  ∃ (s : Finset (ℝ × ℝ × ℝ × ℝ)), 
    (∀ (x y z w : ℝ), 
    (x = z + w + 2*z*w*x) ∧ 
    (y = w + x + 2*w*x*y) ∧ 
    (z = x + y + 2*x*y*z) ∧ 
    (w = y + z + 2*y*z*w) ↔ 
    (x, y, z, w) ∈ s) ∧
    (s.card = 15) :=
sorry

end NUMINAMATH_GPT_real_solutions_to_system_l66_6679


namespace NUMINAMATH_GPT_bailey_credit_cards_l66_6609

theorem bailey_credit_cards (dog_treats : ℕ) (chew_toys : ℕ) (rawhide_bones : ℕ) (items_per_charge : ℕ) (total_items : ℕ) (credit_cards : ℕ)
  (h1 : dog_treats = 8)
  (h2 : chew_toys = 2)
  (h3 : rawhide_bones = 10)
  (h4 : items_per_charge = 5)
  (h5 : total_items = dog_treats + chew_toys + rawhide_bones)
  (h6 : credit_cards = total_items / items_per_charge) :
  credit_cards = 4 :=
by
  sorry

end NUMINAMATH_GPT_bailey_credit_cards_l66_6609


namespace NUMINAMATH_GPT_ground_beef_sold_ratio_l66_6648

variable (beef_sold_Thursday : ℕ) (beef_sold_Saturday : ℕ) (avg_sold_per_day : ℕ) (days : ℕ)

theorem ground_beef_sold_ratio (h₁ : beef_sold_Thursday = 210)
                             (h₂ : beef_sold_Saturday = 150)
                             (h₃ : avg_sold_per_day = 260)
                             (h₄ : days = 3) :
  let total_sold := avg_sold_per_day * days
  let beef_sold_Friday := total_sold - beef_sold_Thursday - beef_sold_Saturday
  (beef_sold_Friday : ℕ) / (beef_sold_Thursday : ℕ) = 2 := by
  sorry

end NUMINAMATH_GPT_ground_beef_sold_ratio_l66_6648


namespace NUMINAMATH_GPT_compare_sums_of_sines_l66_6660

theorem compare_sums_of_sines {A B C : ℝ} 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (h_sum : A + B + C = π) :
  (if A < π / 2 ∧ B < π / 2 ∧ C < π / 2 then
    (1 / (Real.sin (2 * A)) + 1 / (Real.sin (2 * B)) + 1 / (Real.sin (2 * C))
      ≥ 1 / (Real.sin A) + 1 / (Real.sin B) + 1 / (Real.sin C))
  else
    (1 / (Real.sin (2 * A)) + 1 / (Real.sin (2 * B)) + 1 / (Real.sin (2 * C))
      < 1 / (Real.sin A) + 1 / (Real.sin B) + 1 / (Real.sin C))) :=
sorry

end NUMINAMATH_GPT_compare_sums_of_sines_l66_6660


namespace NUMINAMATH_GPT_annual_population_increase_l66_6619

theorem annual_population_increase (x : ℝ) (initial_pop : ℝ) :
    (initial_pop * (1 + (x - 1) / 100)^3 = initial_pop * 1.124864) → x = 5.04 :=
by
  -- Provided conditions
  intros h
  -- The hypothesis conditionally establishes that this will derive to show x = 5.04
  sorry

end NUMINAMATH_GPT_annual_population_increase_l66_6619


namespace NUMINAMATH_GPT_choose_agency_l66_6614

variables (a : ℝ) (x : ℕ)

def cost_agency_A (a : ℝ) (x : ℕ) : ℝ :=
  a + 0.55 * a * x

def cost_agency_B (a : ℝ) (x : ℕ) : ℝ :=
  0.75 * (x + 1) * a

theorem choose_agency (a : ℝ) (x : ℕ) : if (x = 1) then 
                                            (cost_agency_B a x ≤ cost_agency_A a x)
                                         else if (x ≥ 2) then 
                                            (cost_agency_A a x ≤ cost_agency_B a x)
                                         else
                                            true :=
by
  sorry

end NUMINAMATH_GPT_choose_agency_l66_6614


namespace NUMINAMATH_GPT_no_prime_p_for_base_eqn_l66_6632

theorem no_prime_p_for_base_eqn (p : ℕ) (hp: p.Prime) :
  let f (p : ℕ) := 1009 * p^3 + 307 * p^2 + 115 * p + 126 + 7
  let g (p : ℕ) := 143 * p^2 + 274 * p + 361
  f p = g p → false :=
sorry

end NUMINAMATH_GPT_no_prime_p_for_base_eqn_l66_6632


namespace NUMINAMATH_GPT_binary_division_correct_l66_6683

def b1100101 := 0b1100101
def b1101 := 0b1101
def b101 := 0b101
def expected_result := 0b11111010

theorem binary_division_correct : ((b1100101 * b1101) / b101) = expected_result :=
by {
  sorry
}

end NUMINAMATH_GPT_binary_division_correct_l66_6683


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l66_6611

theorem eccentricity_of_ellipse (a b c e : ℝ)
  (h1 : a^2 = 25)
  (h2 : b^2 = 9)
  (h3 : c = Real.sqrt (a^2 - b^2))
  (h4 : e = c / a) :
  e = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l66_6611


namespace NUMINAMATH_GPT_line_does_not_pass_second_quadrant_l66_6659

theorem line_does_not_pass_second_quadrant (a : ℝ) :
  (∀ x y : ℝ, (a + 1) * x + y + 2 - a = 0 → ¬(x < 0 ∧ y > 0)) ↔ a ≤ -1 :=
by
  sorry

end NUMINAMATH_GPT_line_does_not_pass_second_quadrant_l66_6659


namespace NUMINAMATH_GPT_quadratic_solution_eq_l66_6651

noncomputable def p : ℝ :=
  (8 + Real.sqrt 364) / 10

noncomputable def q : ℝ :=
  (8 - Real.sqrt 364) / 10

theorem quadratic_solution_eq (p q : ℝ) (h₁ : 5 * p^2 - 8 * p - 15 = 0) (h₂ : 5 * q^2 - 8 * q - 15 = 0) : 
  (p - q) ^ 2 = 14.5924 :=
sorry

end NUMINAMATH_GPT_quadratic_solution_eq_l66_6651


namespace NUMINAMATH_GPT_spherical_to_rectangular_l66_6618

theorem spherical_to_rectangular :
  let ρ := 6
  let θ := 7 * Real.pi / 4
  let φ := Real.pi / 3
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z) = (-3 * Real.sqrt 6, -3 * Real.sqrt 6, 3) :=
by
  sorry

end NUMINAMATH_GPT_spherical_to_rectangular_l66_6618


namespace NUMINAMATH_GPT_tangent_line_through_origin_max_value_on_interval_min_value_on_interval_l66_6666
noncomputable def f (x : ℝ) : ℝ := x^2 / Real.exp x

theorem tangent_line_through_origin (x y : ℝ) :
  (∃ a : ℝ, (x, y) = (a, f a) ∧ (0, 0) = (0, 0) ∧ y - f a = ((2 * a - a^2) / Real.exp a) * (x - a)) →
  y = x / Real.exp 1 :=
sorry

theorem max_value_on_interval : ∃ (x : ℝ), x = 9 / Real.exp 3 :=
  sorry

theorem min_value_on_interval : ∃ (x : ℝ), x = 0 :=
  sorry

end NUMINAMATH_GPT_tangent_line_through_origin_max_value_on_interval_min_value_on_interval_l66_6666


namespace NUMINAMATH_GPT_intersection_of_sets_l66_6699

-- Define the sets M and N
def M : Set ℝ := { x | 2 < x ∧ x < 3 }
def N : Set ℝ := { x | 2 < x ∧ x ≤ 5 / 2 }

-- State the theorem to prove
theorem intersection_of_sets : M ∩ N = { x | 2 < x ∧ x ≤ 5 / 2 } :=
by 
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l66_6699


namespace NUMINAMATH_GPT_circle_repr_eq_l66_6667

theorem circle_repr_eq (a : ℝ) : (∃ x y : ℝ, x^2 + y^2 - 2*x + 4*y + 1 + a = 0) ↔ a < 4 :=
by
  sorry

end NUMINAMATH_GPT_circle_repr_eq_l66_6667


namespace NUMINAMATH_GPT_least_possible_value_l66_6615

noncomputable def least_value_expression (x : ℝ) : ℝ :=
  (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024

theorem least_possible_value : ∃ x : ℝ, least_value_expression x = 2023 :=
  sorry

end NUMINAMATH_GPT_least_possible_value_l66_6615


namespace NUMINAMATH_GPT_EM_parallel_AC_l66_6622

-- Define the points A, B, C, D, E, and M
variables (A B C D E M : Type) 

-- Define the conditions described in the problem
variables {x y : Real}

-- Given that ABCD is an isosceles trapezoid with AB parallel to CD and AB > CD
variable (isosceles_trapezoid : Prop)

-- E is the foot of the perpendicular from D to AB
variable (foot_perpendicular : Prop)

-- M is the midpoint of BD
variable (midpoint : Prop)

-- We need to prove that EM is parallel to AC
theorem EM_parallel_AC (h1 : isosceles_trapezoid) (h2 : foot_perpendicular) (h3 : midpoint) : Prop := sorry

end NUMINAMATH_GPT_EM_parallel_AC_l66_6622


namespace NUMINAMATH_GPT_find_c_l66_6661

theorem find_c (a b c n : ℝ) (h : n = (2 * a * b * c) / (c - a)) : c = (n * a) / (n - 2 * a * b) :=
by
  sorry

end NUMINAMATH_GPT_find_c_l66_6661


namespace NUMINAMATH_GPT_average_interest_rate_l66_6691

theorem average_interest_rate 
  (total : ℝ)
  (rate1 rate2 yield1 yield2 : ℝ)
  (amount1 amount2 : ℝ)
  (h_total : total = amount1 + amount2)
  (h_rate1 : rate1 = 0.03)
  (h_rate2 : rate2 = 0.07)
  (h_yield_equal : yield1 = yield2)
  (h_yield1 : yield1 = rate1 * amount1)
  (h_yield2 : yield2 = rate2 * amount2) :
  (yield1 + yield2) / total = 0.042 :=
by
  sorry

end NUMINAMATH_GPT_average_interest_rate_l66_6691


namespace NUMINAMATH_GPT_find_xyz_l66_6686

theorem find_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x + 1/y = 5) (h2 : y + 1/z = 2) (h3 : z + 1/x = 3) :
  x * y * z = 10 + 3 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_GPT_find_xyz_l66_6686


namespace NUMINAMATH_GPT_reciprocal_sum_l66_6603

theorem reciprocal_sum :
  let a := (1 / 4 : ℚ)
  let b := (1 / 5 : ℚ)
  1 / (a + b) = 20 / 9 :=
by
  let a := (1 / 4 : ℚ)
  let b := (1 / 5 : ℚ)
  have h : a + b = 9 / 20 := by sorry
  have h_rec : 1 / (a + b) = 20 / 9 := by sorry
  exact h_rec

end NUMINAMATH_GPT_reciprocal_sum_l66_6603


namespace NUMINAMATH_GPT_probability_point_in_sphere_eq_2pi_div_3_l66_6650

open Real Topology

noncomputable def volume_of_region := 4 * 2 * 2

noncomputable def volume_of_sphere_radius_2 : ℝ :=
  (4 / 3) * π * (2 ^ 3)

noncomputable def probability_in_sphere : ℝ :=
  volume_of_sphere_radius_2 / volume_of_region

theorem probability_point_in_sphere_eq_2pi_div_3 :
  probability_in_sphere = (2 * π) / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_point_in_sphere_eq_2pi_div_3_l66_6650


namespace NUMINAMATH_GPT_probability_of_selecting_at_least_one_female_l66_6673

open BigOperators

noncomputable def prob_at_least_one_female_selected : ℚ :=
  let total_choices := Nat.choose 10 3
  let all_males_choices := Nat.choose 6 3
  1 - (all_males_choices / total_choices : ℚ)

theorem probability_of_selecting_at_least_one_female :
  prob_at_least_one_female_selected = 5 / 6 := by
  sorry

end NUMINAMATH_GPT_probability_of_selecting_at_least_one_female_l66_6673


namespace NUMINAMATH_GPT_baker_final_stock_l66_6623

-- Given conditions as Lean definitions
def initial_cakes : Nat := 173
def additional_cakes : Nat := 103
def damaged_percentage : Nat := 25
def sold_first_day : Nat := 86
def sold_next_day_percentage : Nat := 10

-- Calculate new cakes Baker adds to the stock after accounting for damaged cakes
def new_undamaged_cakes : Nat := (additional_cakes * (100 - damaged_percentage)) / 100

-- Calculate stock after adding new cakes
def stock_after_new_cakes : Nat := initial_cakes + new_undamaged_cakes

-- Calculate stock after first day's sales
def stock_after_first_sale : Nat := stock_after_new_cakes - sold_first_day

-- Calculate cakes sold on the second day
def sold_next_day : Nat := (stock_after_first_sale * sold_next_day_percentage) / 100

-- Final stock calculations
def final_stock : Nat := stock_after_first_sale - sold_next_day

-- Prove that Baker has 148 cakes left
theorem baker_final_stock : final_stock = 148 := by
  sorry

end NUMINAMATH_GPT_baker_final_stock_l66_6623


namespace NUMINAMATH_GPT_second_most_eater_l66_6630

variable (C M K B T : ℕ)  -- Assuming the quantities of food each child ate are positive integers

theorem second_most_eater
  (h1 : C > M)
  (h2 : B < K)
  (h3 : T < K)
  (h4 : K < M) :
  ∃ x, x = M ∧ (∀ y, y ≠ C → x ≥ y) ∧ (∃ z, z ≠ C ∧ z > M) :=
by {
  sorry
}

end NUMINAMATH_GPT_second_most_eater_l66_6630


namespace NUMINAMATH_GPT_find_smaller_number_l66_6668

theorem find_smaller_number (x y : ℕ) (h1 : x = 2 * y - 3) (h2 : x + y = 51) : y = 18 :=
sorry

end NUMINAMATH_GPT_find_smaller_number_l66_6668


namespace NUMINAMATH_GPT_hyperbola_focal_distance_distance_focus_to_asymptote_l66_6657

theorem hyperbola_focal_distance :
  let a := 1
  let b := Real.sqrt 3
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  (2 * c = 4) :=
by sorry

theorem distance_focus_to_asymptote :
  let a := 1
  let b := Real.sqrt 3
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  let focus := (c, 0)
  let A := -Real.sqrt 3
  let B := 1
  let C := 0
  let distance := (|A * focus.fst + B * focus.snd + C|) / Real.sqrt (A ^ 2 + B ^ 2)
  (distance = Real.sqrt 3) :=
by sorry

end NUMINAMATH_GPT_hyperbola_focal_distance_distance_focus_to_asymptote_l66_6657


namespace NUMINAMATH_GPT_find_perfect_matching_l66_6605

-- Define the boys and girls
inductive Boy | B1 | B2 | B3
inductive Girl | G1 | G2 | G3

-- Define the knowledge relationship
def knows : Boy → Girl → Prop
| Boy.B1, Girl.G1 => true
| Boy.B1, Girl.G2 => true
| Boy.B2, Girl.G1 => true
| Boy.B2, Girl.G3 => true
| Boy.B3, Girl.G2 => true
| Boy.B3, Girl.G3 => true
| _, _ => false

-- Proposition to prove
theorem find_perfect_matching :
  ∃ (pairing : Boy → Girl), 
    (∀ b : Boy, knows b (pairing b)) ∧ 
    (∀ g : Girl, ∃ b : Boy, pairing b = g) :=
by
  sorry

end NUMINAMATH_GPT_find_perfect_matching_l66_6605


namespace NUMINAMATH_GPT_difference_in_gems_l66_6638

theorem difference_in_gems (r d : ℕ) (h : d = 3 * r) : d - r = 2 * r := 
by 
  sorry

end NUMINAMATH_GPT_difference_in_gems_l66_6638


namespace NUMINAMATH_GPT_box_volume_l66_6665

theorem box_volume
  (l w h : ℝ)
  (A1 : l * w = 36)
  (A2 : w * h = 18)
  (A3 : l * h = 8) :
  l * w * h = 102 := 
sorry

end NUMINAMATH_GPT_box_volume_l66_6665


namespace NUMINAMATH_GPT_news_spread_time_l66_6689

theorem news_spread_time (n : ℕ) (m : ℕ) :
  (2^m < n ∧ n < 2^(m+k+1) ∧ (n % 2 = 1) ∧ n % 2 = 1) →
  ∃ t : ℕ, t = (if n % 2 = 1 then m+2 else m+1) := 
sorry

end NUMINAMATH_GPT_news_spread_time_l66_6689


namespace NUMINAMATH_GPT_cat_mouse_position_258_l66_6697

-- Define the cycle positions for the cat
def cat_position (n : ℕ) : String :=
  match n % 4 with
  | 0 => "top left"
  | 1 => "top right"
  | 2 => "bottom right"
  | _ => "bottom left"

-- Define the cycle positions for the mouse
def mouse_position (n : ℕ) : String :=
  match n % 8 with
  | 0 => "top middle"
  | 1 => "top right"
  | 2 => "right middle"
  | 3 => "bottom right"
  | 4 => "bottom middle"
  | 5 => "bottom left"
  | 6 => "left middle"
  | _ => "top left"

theorem cat_mouse_position_258 : 
  cat_position 258 = "top right" ∧ mouse_position 258 = "top right" := by
  sorry

end NUMINAMATH_GPT_cat_mouse_position_258_l66_6697


namespace NUMINAMATH_GPT_minimum_value_l66_6633

theorem minimum_value (x y z : ℝ) (h : 2 * x - 3 * y + z = 3) :
  ∃ y_min, y_min = -2 / 7 ∧ x = 6 / 7 ∧ (x^2 + (y - 1)^2 + z^2) = 18 / 7 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_l66_6633


namespace NUMINAMATH_GPT_jack_buttons_total_l66_6646

theorem jack_buttons_total :
  (3 * 3) * 7 = 63 :=
by
  sorry

end NUMINAMATH_GPT_jack_buttons_total_l66_6646


namespace NUMINAMATH_GPT_smallest_integer_with_eight_factors_l66_6626

theorem smallest_integer_with_eight_factors : ∃ n : ℕ, 
  ∀ m : ℕ, (∀ p : ℕ, ∃ k : ℕ, m = p^k → (k + 1) * (p + 1) = 8) → (n ≤ m) ∧ 
  (∀ d : ℕ, d ∣ n → d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∨ d = 12 ∨ d = 24) :=
sorry

end NUMINAMATH_GPT_smallest_integer_with_eight_factors_l66_6626


namespace NUMINAMATH_GPT_intersection_ab_correct_l66_6663

noncomputable def set_A : Set ℝ := { x : ℝ | x > 1/3 }
def set_B : Set ℝ := { x : ℝ | ∃ y : ℝ, x^2 + y^2 = 4 ∧ y ≥ -2 ∧ y ≤ 2 }
def intersection_AB : Set ℝ := { x : ℝ | 1/3 < x ∧ x ≤ 2 }

theorem intersection_ab_correct : set_A ∩ set_B = intersection_AB := 
by 
  -- proof omitted
  sorry

end NUMINAMATH_GPT_intersection_ab_correct_l66_6663


namespace NUMINAMATH_GPT_cross_fraction_eq1_cross_fraction_eq2_cross_fraction_eq3_l66_6670

-- Problem 1
theorem cross_fraction_eq1 (x : ℝ) : (x + 12 / x = -7) → 
  ∃ (x₁ x₂ : ℝ), (x₁ = -3 ∧ x₂ = -4 ∧ x = x₁ ∨ x = x₂) :=
sorry

-- Problem 2
theorem cross_fraction_eq2 (a b : ℝ) 
    (h1 : a * b = -6) 
    (h2 : a + b = -5) : (a ≠ 0 ∧ b ≠ 0) →
    (b / a + a / b + 1 = -31 / 6) :=
sorry

-- Problem 3
theorem cross_fraction_eq3 (k x₁ x₂ : ℝ)
    (hk : k > 2)
    (hx1 : x₁ = 2022 * k - 2022)
    (hx2 : x₂ = k + 1) :
    (x₁ > x₂) →
    (x₁ + 4044) / x₂ = 2022 :=
sorry

end NUMINAMATH_GPT_cross_fraction_eq1_cross_fraction_eq2_cross_fraction_eq3_l66_6670


namespace NUMINAMATH_GPT_range_of_m_l66_6642

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 4^x + m * 2^x + m^2 - 1 = 0) ↔ - (2 * Real.sqrt 3) / 3 ≤ m ∧ m < 1 :=
sorry

end NUMINAMATH_GPT_range_of_m_l66_6642


namespace NUMINAMATH_GPT_hillside_camp_boys_percentage_l66_6653

theorem hillside_camp_boys_percentage (B G : ℕ) 
  (h1 : B + G = 60) 
  (h2 : G = 6) : (B: ℕ) / 60 * 100 = 90 :=
by
  sorry

end NUMINAMATH_GPT_hillside_camp_boys_percentage_l66_6653


namespace NUMINAMATH_GPT_julie_savings_multiple_l66_6641

theorem julie_savings_multiple (S : ℝ) (hS : 0 < S) :
  (12 * 0.25 * S) / (0.75 * S) = 4 :=
by
  sorry

end NUMINAMATH_GPT_julie_savings_multiple_l66_6641


namespace NUMINAMATH_GPT_add_sub_decimals_l66_6677

theorem add_sub_decimals :
  (0.513 + 0.0067 - 0.048 = 0.4717) :=
by
  sorry

end NUMINAMATH_GPT_add_sub_decimals_l66_6677


namespace NUMINAMATH_GPT_geometric_sequence_sum_is_9_l66_6690

theorem geometric_sequence_sum_is_9 {a : ℕ → ℝ} (q : ℝ) 
  (h3a7 : a 3 * a 7 = 8) 
  (h4a6 : a 4 + a 6 = 6)
  (h_geom : ∀ n, a (n + 1) = a n * q) : a 2 + a 8 = 9 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_is_9_l66_6690


namespace NUMINAMATH_GPT_ganesh_average_speed_l66_6675

noncomputable def averageSpeed (D : ℝ) : ℝ :=
  let time_uphill := D / 60
  let time_downhill := D / 36
  let total_time := time_uphill + time_downhill
  let total_distance := 2 * D
  total_distance / total_time

theorem ganesh_average_speed (D : ℝ) (hD : D > 0) : averageSpeed D = 45 := by
  sorry

end NUMINAMATH_GPT_ganesh_average_speed_l66_6675


namespace NUMINAMATH_GPT_length_of_third_wall_l66_6643

-- Define the dimensions of the first two walls
def wall1_length : ℕ := 30
def wall1_height : ℕ := 12
def wall1_area : ℕ := wall1_length * wall1_height

def wall2_length : ℕ := 30
def wall2_height : ℕ := 12
def wall2_area : ℕ := wall2_length * wall2_height

-- Total area needed
def total_area_needed : ℕ := 960

-- Calculate the area for the third wall
def two_walls_area : ℕ := wall1_area + wall2_area
def third_wall_area : ℕ := total_area_needed - two_walls_area

-- Height of the third wall
def third_wall_height : ℕ := 12

-- Calculate the length of the third wall
def third_wall_length : ℕ := third_wall_area / third_wall_height

-- Final claim: Length of the third wall is 20 feet
theorem length_of_third_wall : third_wall_length = 20 := by
  sorry

end NUMINAMATH_GPT_length_of_third_wall_l66_6643


namespace NUMINAMATH_GPT_sum_is_220_l66_6674

def second_number := 60
def first_number := 2 * second_number
def third_number := first_number / 3
def sum_of_numbers := first_number + second_number + third_number

theorem sum_is_220 : sum_of_numbers = 220 :=
by
  sorry

end NUMINAMATH_GPT_sum_is_220_l66_6674


namespace NUMINAMATH_GPT_price_of_rice_packet_l66_6692

-- Definitions based on conditions
def initial_amount : ℕ := 500
def wheat_flour_price : ℕ := 25
def wheat_flour_quantity : ℕ := 3
def soda_price : ℕ := 150
def remaining_balance : ℕ := 235
def total_spending (P : ℕ) : ℕ := initial_amount - remaining_balance

-- Theorem to prove
theorem price_of_rice_packet (P : ℕ) (h: 2 * P + wheat_flour_quantity * wheat_flour_price + soda_price = total_spending P) : P = 20 :=
sorry

end NUMINAMATH_GPT_price_of_rice_packet_l66_6692


namespace NUMINAMATH_GPT_emily_points_l66_6682

theorem emily_points (r1 r2 r3 r4 r5 m4 m5 l : ℤ)
  (h1 : r1 = 16)
  (h2 : r2 = 33)
  (h3 : r3 = 21)
  (h4 : r4 = 10)
  (h5 : r5 = 4)
  (hm4 : m4 = 2)
  (hm5 : m5 = 3)
  (hl : l = 48) :
  r1 + r2 + r3 + r4 * m4 + r5 * m5 - l = 54 := by
  sorry

end NUMINAMATH_GPT_emily_points_l66_6682


namespace NUMINAMATH_GPT_tetrahedron_vertex_angle_sum_l66_6639

theorem tetrahedron_vertex_angle_sum (A B C D : Type) (angles_at : Type → Type → Type → ℝ) :
  (∃ A, (∀ X Y Z W, X = A ∨ Y = A ∨ Z = A ∨ W = A → angles_at X Y A + angles_at Z W A > 180)) →
  ¬ (∃ A B, A ≠ B ∧ 
    (∀ X Y, X = A ∨ Y = A → angles_at X Y A + angles_at Y X A > 180) ∧ 
    (∀ X Y, X = B ∨ Y = B → angles_at X Y B + angles_at Y X B > 180)) := 
sorry

end NUMINAMATH_GPT_tetrahedron_vertex_angle_sum_l66_6639


namespace NUMINAMATH_GPT_find_q_sum_l66_6635

variable (q : ℕ → ℕ)

def conditions :=
  q 3 = 2 ∧ 
  q 8 = 20 ∧ 
  q 16 = 12 ∧ 
  q 21 = 30

theorem find_q_sum (h : conditions q) : 
  (q 1 + q 2 + q 3 + q 4 + q 5 + q 6 + q 7 + q 8 + q 9 + q 10 + q 11 + 
   q 12 + q 13 + q 14 + q 15 + q 16 + q 17 + q 18 + q 19 + q 20 + q 21 + q 22) = 352 := 
  sorry

end NUMINAMATH_GPT_find_q_sum_l66_6635


namespace NUMINAMATH_GPT_solution_to_system_l66_6620

theorem solution_to_system :
  ∀ (x y z : ℝ), 
  x * (3 * y^2 + 1) = y * (y^2 + 3) →
  y * (3 * z^2 + 1) = z * (z^2 + 3) →
  z * (3 * x^2 + 1) = x * (x^2 + 3) →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ 
  (x = -1 ∧ y = -1 ∧ z = -1) :=
by
  sorry

end NUMINAMATH_GPT_solution_to_system_l66_6620


namespace NUMINAMATH_GPT_solve_for_x2_plus_9y2_l66_6613

variable (x y : ℝ)

def condition1 : Prop := x + 3 * y = 3
def condition2 : Prop := x * y = -6

theorem solve_for_x2_plus_9y2 (h1 : condition1 x y) (h2 : condition2 x y) :
  x^2 + 9 * y^2 = 45 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x2_plus_9y2_l66_6613


namespace NUMINAMATH_GPT_prove_incorrect_conclusion_l66_6601

-- Define the parabola as y = ax^2 + bx + c
def parabola_eq (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the points
def point1 (a b c : ℝ) : Prop := parabola_eq a b c (-2) = 0
def point2 (a b c : ℝ) : Prop := parabola_eq a b c (-1) = 4
def point3 (a b c : ℝ) : Prop := parabola_eq a b c 0 = 6
def point4 (a b c : ℝ) : Prop := parabola_eq a b c 1 = 6

-- Define the conditions
def conditions (a b c : ℝ) : Prop :=
  point1 a b c ∧ point2 a b c ∧ point3 a b c ∧ point4 a b c

-- Define the incorrect conclusion
def incorrect_conclusion (a b c : ℝ) : Prop :=
  ¬ (parabola_eq a b c 2 = 0)

-- The statement to be proven
theorem prove_incorrect_conclusion (a b c : ℝ) (h : conditions a b c) : incorrect_conclusion a b c :=
sorry

end NUMINAMATH_GPT_prove_incorrect_conclusion_l66_6601


namespace NUMINAMATH_GPT_arccos_neg_half_eq_two_pi_over_three_l66_6644

theorem arccos_neg_half_eq_two_pi_over_three :
  Real.arccos (-1/2) = 2 * Real.pi / 3 := sorry

end NUMINAMATH_GPT_arccos_neg_half_eq_two_pi_over_three_l66_6644


namespace NUMINAMATH_GPT_mass_percentage_ca_in_compound_l66_6694

noncomputable def mass_percentage_ca_in_cac03 : ℝ :=
  let mm_ca := 40.08
  let mm_c := 12.01
  let mm_o := 16.00
  let mm_caco3 := mm_ca + mm_c + 3 * mm_o
  (mm_ca / mm_caco3) * 100

theorem mass_percentage_ca_in_compound (mp : ℝ) (h : mp = mass_percentage_ca_in_cac03) : mp = 40.04 := by
  sorry

end NUMINAMATH_GPT_mass_percentage_ca_in_compound_l66_6694


namespace NUMINAMATH_GPT_mittens_per_box_l66_6606

theorem mittens_per_box (total_boxes : ℕ) (scarves_per_box : ℕ) (total_clothing : ℕ) 
  (h_total_boxes : total_boxes = 4) 
  (h_scarves_per_box : scarves_per_box = 2) 
  (h_total_clothing : total_clothing = 32) : 
  (total_clothing - total_boxes * scarves_per_box) / total_boxes = 6 := 
by
  -- Sorry, proof is omitted
  sorry

end NUMINAMATH_GPT_mittens_per_box_l66_6606


namespace NUMINAMATH_GPT_find_x_l66_6672

theorem find_x (x : ℝ) 
  (a : ℝ × ℝ := (2*x - 1, x + 3)) 
  (b : ℝ × ℝ := (x, 2*x + 1))
  (c : ℝ × ℝ := (1, 2))
  (h : (a.1 - b.1) * c.1 + (a.2 - b.2) * c.2 = 0) :
  x = 3 :=
  sorry

end NUMINAMATH_GPT_find_x_l66_6672


namespace NUMINAMATH_GPT_system_of_equations_solution_l66_6617

theorem system_of_equations_solution :
  ∃ x y : ℚ, (3 * x + 4 * y = 10) ∧ (12 * x - 8 * y = 8) ∧ (x = 14 / 9) ∧ (y = 4 / 3) :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l66_6617


namespace NUMINAMATH_GPT_permutation_formula_l66_6628

noncomputable def permutation (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem permutation_formula (n k : ℕ) (h : 1 ≤ k ∧ k ≤ n) : permutation n k = Nat.factorial n / Nat.factorial (n - k) :=
by
  unfold permutation
  sorry

end NUMINAMATH_GPT_permutation_formula_l66_6628


namespace NUMINAMATH_GPT_three_Z_five_l66_6696

def Z (a b : ℤ) : ℤ := b + 7 * a - 3 * a^2

theorem three_Z_five : Z 3 5 = -1 := by
  sorry

end NUMINAMATH_GPT_three_Z_five_l66_6696


namespace NUMINAMATH_GPT_roots_quadratic_sum_product_l66_6652

theorem roots_quadratic_sum_product :
  (∀ x1 x2 : ℝ, (∀ x, x^2 - 4 * x + 3 = 0 → x = x1 ∨ x = x2) → (x1 + x2 - x1 * x2 = 1)) :=
by
  sorry

end NUMINAMATH_GPT_roots_quadratic_sum_product_l66_6652


namespace NUMINAMATH_GPT_solve_congruence_l66_6634

theorem solve_congruence (n : ℕ) (hn : n < 47) 
  (congr_13n : 13 * n ≡ 9 [MOD 47]) : n ≡ 20 [MOD 47] :=
sorry

end NUMINAMATH_GPT_solve_congruence_l66_6634


namespace NUMINAMATH_GPT_intersection_A_B_l66_6649

-- Define the sets A and B based on the given conditions
def A := { x : ℝ | (1 / 9) ≤ (3:ℝ)^x ∧ (3:ℝ)^x ≤ 1 }
def B := { x : ℝ | x^2 < 1 }

-- State the theorem for the intersection of sets A and B
theorem intersection_A_B :
  A ∩ B = { x : ℝ | -1 < x ∧ x ≤ 0 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l66_6649


namespace NUMINAMATH_GPT_problem1_asymptotes_problem2_equation_l66_6647

-- Problem 1: Asymptotes of a hyperbola
theorem problem1_asymptotes (a : ℝ) (x y : ℝ) (hx : (y + a) ^ 2 - (x - a) ^ 2 = 2 * a)
  (hpt : 3 = x ∧ 1 = y) : 
  (y = x - 2 * a) ∨ (y = - x) := 
by 
  sorry

-- Problem 2: Equation of a hyperbola
theorem problem2_equation (a b c : ℝ) (x y : ℝ) 
  (hasymptote : y = x + 1 ∨ y = - (x + 1))  (hfocal : 2 * c = 4)
  (hc_squared : c ^ 2 = a ^ 2 + b ^ 2) (ha_eq_b : a = b): 
  y^2 - (x + 1)^2 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_problem1_asymptotes_problem2_equation_l66_6647


namespace NUMINAMATH_GPT_ratio_of_female_to_male_officers_on_duty_l66_6676

theorem ratio_of_female_to_male_officers_on_duty 
    (p : ℝ) (T : ℕ) (F : ℕ) 
    (hp : p = 0.19) (hT : T = 152) (hF : F = 400) : 
    (76 / 76) = 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_female_to_male_officers_on_duty_l66_6676


namespace NUMINAMATH_GPT_expected_balls_in_original_position_proof_l66_6698

-- Define the problem conditions as Lean definitions
def n_balls : ℕ := 10

def probability_not_moved_by_one_rotation : ℚ := 7 / 10

def probability_not_moved_by_two_rotations : ℚ := (7 / 10) * (7 / 10)

def expected_balls_in_original_position : ℚ := n_balls * probability_not_moved_by_two_rotations

-- The statement representing the proof problem
theorem expected_balls_in_original_position_proof :
  expected_balls_in_original_position = 4.9 :=
  sorry

end NUMINAMATH_GPT_expected_balls_in_original_position_proof_l66_6698


namespace NUMINAMATH_GPT_cupboard_slots_l66_6658

theorem cupboard_slots (shelves_from_top shelves_from_bottom slots_from_left slots_from_right : ℕ)
  (h_top : shelves_from_top = 1)
  (h_bottom : shelves_from_bottom = 3)
  (h_left : slots_from_left = 0)
  (h_right : slots_from_right = 6) :
  (shelves_from_top + 1 + shelves_from_bottom) * (slots_from_left + 1 + slots_from_right) = 35 := by
  sorry

end NUMINAMATH_GPT_cupboard_slots_l66_6658


namespace NUMINAMATH_GPT_evaluate_g_at_4_l66_6662

def g (x : ℕ) := 5 * x + 2

theorem evaluate_g_at_4 : g 4 = 22 := by
  sorry

end NUMINAMATH_GPT_evaluate_g_at_4_l66_6662


namespace NUMINAMATH_GPT_range_of_slope_angle_l66_6664

theorem range_of_slope_angle (l : ℝ → ℝ) (theta : ℝ) 
    (h_line_eqn : ∀ x y, l x = y ↔ x - y * Real.sin theta + 2 = 0) : 
    ∃ α : ℝ, α ∈ Set.Icc (Real.pi / 4) (3 * Real.pi / 4) :=
sorry

end NUMINAMATH_GPT_range_of_slope_angle_l66_6664


namespace NUMINAMATH_GPT_number_of_pupils_not_in_programX_is_639_l66_6610

-- Definitions for the conditions
def total_girls_elementary : ℕ := 192
def total_boys_elementary : ℕ := 135
def total_girls_middle : ℕ := 233
def total_boys_middle : ℕ := 163
def total_girls_high : ℕ := 117
def total_boys_high : ℕ := 89

def programX_girls_elementary : ℕ := 48
def programX_boys_elementary : ℕ := 28
def programX_girls_middle : ℕ := 98
def programX_boys_middle : ℕ := 51
def programX_girls_high : ℕ := 40
def programX_boys_high : ℕ := 25

-- Question formulation
theorem number_of_pupils_not_in_programX_is_639 :
  (total_girls_elementary - programX_girls_elementary) +
  (total_boys_elementary - programX_boys_elementary) +
  (total_girls_middle - programX_girls_middle) +
  (total_boys_middle - programX_boys_middle) +
  (total_girls_high - programX_girls_high) +
  (total_boys_high - programX_boys_high) = 639 := 
  by
  sorry

end NUMINAMATH_GPT_number_of_pupils_not_in_programX_is_639_l66_6610


namespace NUMINAMATH_GPT_number_of_ways_to_choose_books_l66_6629

def num_books := 15
def books_to_choose := 3

theorem number_of_ways_to_choose_books : Nat.choose num_books books_to_choose = 455 := by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_choose_books_l66_6629


namespace NUMINAMATH_GPT_find_two_digit_number_l66_6685

theorem find_two_digit_number (n s p : ℕ) (h1 : n = 4 * s) (h2 : n = 3 * p) : n = 24 := 
  sorry

end NUMINAMATH_GPT_find_two_digit_number_l66_6685


namespace NUMINAMATH_GPT_total_amount_received_l66_6645

-- Definitions based on conditions
def days_A : Nat := 6
def days_B : Nat := 8
def days_ABC : Nat := 3

def share_A : Nat := 300
def share_B : Nat := 225
def share_C : Nat := 75

-- The theorem stating the total amount received for the work
theorem total_amount_received (dA dB dABC : Nat) (sA sB sC : Nat)
  (h1 : dA = days_A) (h2 : dB = days_B) (h3 : dABC = days_ABC)
  (h4 : sA = share_A) (h5 : sB = share_B) (h6 : sC = share_C) : 
  sA + sB + sC = 600 := by
  sorry

end NUMINAMATH_GPT_total_amount_received_l66_6645


namespace NUMINAMATH_GPT_kids_waiting_for_swings_l66_6681

theorem kids_waiting_for_swings (x : ℕ) (h1 : 2 * 60 = 120) 
  (h2 : ∀ y, y = 2 → (y * x = 2 * x)) 
  (h3 : 15 * (2 * x) = 30 * x)
  (h4 : 120 * x - 30 * x = 270) : x = 3 :=
sorry

end NUMINAMATH_GPT_kids_waiting_for_swings_l66_6681
