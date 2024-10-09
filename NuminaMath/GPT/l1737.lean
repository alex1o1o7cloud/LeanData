import Mathlib

namespace range_of_a_l1737_173722

theorem range_of_a (a x y : ℝ) (h1 : x - y = a + 3) (h2 : 2 * x + y = 5 * a) (h3 : x < y) : a < -3 :=
by
  sorry

end range_of_a_l1737_173722


namespace georgia_makes_muffins_l1737_173720

-- Definitions based on conditions
def muffinRecipeMakes : ℕ := 6
def numberOfStudents : ℕ := 24
def durationInMonths : ℕ := 9

-- Theorem to prove the given problem
theorem georgia_makes_muffins :
  (numberOfStudents / muffinRecipeMakes) * durationInMonths = 36 :=
by
  -- We'll skip the proof with sorry
  sorry

end georgia_makes_muffins_l1737_173720


namespace find_A_plus_B_l1737_173734

theorem find_A_plus_B {A B : ℚ} (h : ∀ x : ℚ, 
                     (Bx - 17) / (x^2 - 9 * x + 20) = A / (x - 4) + 5 / (x - 5)) : 
                     A + B = 9 / 5 := sorry

end find_A_plus_B_l1737_173734


namespace prob_statement_l1737_173736

open Set

-- Definitions from the conditions
def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | x^2 + 2 * x < 0}

-- Proposition to be proved
theorem prob_statement : A ∩ (Bᶜ) = {-2, 0, 1, 2} :=
by
  sorry

end prob_statement_l1737_173736


namespace valid_range_and_difference_l1737_173742

/- Assume side lengths as given expressions -/
def BC (x : ℝ) : ℝ := x + 11
def AC (x : ℝ) : ℝ := x + 6
def AB (x : ℝ) : ℝ := 3 * x + 2

/- Define the inequalities representing the triangle inequalities and largest angle condition -/
def triangle_inequality1 (x : ℝ) : Prop := AB x + AC x > BC x
def triangle_inequality2 (x : ℝ) : Prop := AB x + BC x > AC x
def triangle_inequality3 (x : ℝ) : Prop := AC x + BC x > AB x
def largest_angle_condition (x : ℝ) : Prop := BC x > AB x

/- Define the combined condition for x, ensuring all relevant conditions are met -/
def valid_x_range (x : ℝ) : Prop :=
  1 < x ∧ x < 4.5 ∧ triangle_inequality1 x ∧ triangle_inequality2 x ∧ triangle_inequality3 x ∧ largest_angle_condition x

/- Compute n - m for the interval (m, n) where x lies -/
def n_minus_m : ℝ :=
  4.5 - 1

/- Main theorem stating the final result -/
theorem valid_range_and_difference :
  (∃ x : ℝ, valid_x_range x) ∧ (n_minus_m = 7 / 2) :=
by
  sorry

end valid_range_and_difference_l1737_173742


namespace r_can_complete_work_in_R_days_l1737_173745

theorem r_can_complete_work_in_R_days (W : ℝ) : 
  (∀ p q r P Q R : ℝ, 
    (P = W / 24) ∧
    (Q = W / 9) ∧
    (10.000000000000002 * (W / 24) + 3 * (W / 9 + W / R) = W) 
  -> R = 12) :=
by
  intros
  sorry

end r_can_complete_work_in_R_days_l1737_173745


namespace correct_linear_regression_statement_l1737_173760

-- Definitions based on the conditions:
def linear_regression (b a e : ℝ) (x : ℝ) : ℝ := b * x + a + e

def statement_A (b a e : ℝ) (x : ℝ) : Prop := linear_regression b a e x = b * x + a + e

def statement_B (b a e : ℝ) (x : ℝ) : Prop := ∀ x1 x2, (linear_regression b a e x1 ≠ linear_regression b a e x2) → (x1 ≠ x2)

def statement_C (b a e : ℝ) (x : ℝ) : Prop := ∃ (other_factors : ℝ), linear_regression b a e x = b * x + a + other_factors + e

def statement_D (b a e : ℝ) (x : ℝ) : Prop := (e ≠ 0) → false

-- The proof statement
theorem correct_linear_regression_statement (b a e : ℝ) (x : ℝ) :
  (statement_C b a e x) :=
sorry

end correct_linear_regression_statement_l1737_173760


namespace clock_strikes_l1737_173749

theorem clock_strikes (t n : ℕ) (h_t : 13 * t = 26) (h_n : 2 * n - 1 * t = 22) : n = 6 :=
by
  sorry

end clock_strikes_l1737_173749


namespace p_is_sufficient_but_not_necessary_l1737_173729

-- Definitions based on conditions
def p (x y : Int) : Prop := x + y ≠ -2
def q (x y : Int) : Prop := ¬(x = -1 ∧ y = -1)

theorem p_is_sufficient_but_not_necessary (x y : Int) : 
  (p x y → q x y) ∧ ¬(q x y → p x y) :=
by
  sorry

end p_is_sufficient_but_not_necessary_l1737_173729


namespace sum_of_special_right_triangle_areas_l1737_173703

noncomputable def is_special_right_triangle (a b : ℕ) : Prop :=
  let area := (a * b) / 2
  area = 3 * (a + b)

noncomputable def special_right_triangle_areas : List ℕ :=
  [(18, 9), (9, 18), (15, 10), (10, 15), (12, 12)].map (λ p => (p.1 * p.2) / 2)

theorem sum_of_special_right_triangle_areas : 
  special_right_triangle_areas.eraseDups.sum = 228 := by
  sorry

end sum_of_special_right_triangle_areas_l1737_173703


namespace star_7_3_eq_neg_5_l1737_173757

def star_operation (a b : ℤ) : ℤ := 4 * a + 3 * b - 2 * a * b

theorem star_7_3_eq_neg_5 : star_operation 7 3 = -5 :=
by
  -- proof goes here
  sorry

end star_7_3_eq_neg_5_l1737_173757


namespace inequality_sinx_plus_y_cosx_plus_y_l1737_173778

open Real

theorem inequality_sinx_plus_y_cosx_plus_y (
  y x : ℝ
) (hx : x ∈ Set.Icc (π / 4) (3 * π / 4)) (hy : y ∈ Set.Icc (π / 4) (3 * π / 4)) :
  sin (x + y) + cos (x + y) ≤ sin x + cos x + sin y + cos y :=
sorry

end inequality_sinx_plus_y_cosx_plus_y_l1737_173778


namespace roberto_outfits_l1737_173783

-- Roberto's wardrobe constraints
def num_trousers : ℕ := 5
def num_shirts : ℕ := 6
def num_jackets : ℕ := 4
def num_shoes : ℕ := 3
def restricted_jacket_shoes : ℕ := 2

-- The total number of valid outfits
def total_outfits_with_constraint : ℕ := 330

-- Proving the equivalent of the problem statement
theorem roberto_outfits :
  (num_trousers * num_shirts * (num_jackets - 1) * num_shoes) + (num_trousers * num_shirts * 1 * restricted_jacket_shoes) = total_outfits_with_constraint :=
by
  sorry

end roberto_outfits_l1737_173783


namespace polynomial_identity_and_sum_of_squares_l1737_173751

theorem polynomial_identity_and_sum_of_squares :
  ∃ (p q r s t u : ℤ), (∀ (x : ℤ), 512 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) ∧
    p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 5472 :=
sorry

end polynomial_identity_and_sum_of_squares_l1737_173751


namespace quadratic_real_roots_l1737_173795

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, (k + 1) * x^2 + 4 * x - 1 = 0) ↔ k ≥ -5 ∧ k ≠ -1 :=
by
  sorry

end quadratic_real_roots_l1737_173795


namespace parabola_chord_length_l1737_173708

theorem parabola_chord_length (x₁ x₂ : ℝ) (y₁ y₂ : ℝ) 
(h1 : y₁^2 = 4 * x₁) 
(h2 : y₂^2 = 4 * x₂) 
(h3 : x₁ + x₂ = 6) : 
|y₁ - y₂| = 8 :=
sorry

end parabola_chord_length_l1737_173708


namespace expression_value_as_fraction_l1737_173770

theorem expression_value_as_fraction (x y : ℕ) (hx : x = 3) (hy : y = 5) : 
  ( ( (1 / (y : ℚ)) / (1 / (x : ℚ)) ) ^ 2 ) = 9 / 25 := 
by
  sorry

end expression_value_as_fraction_l1737_173770


namespace greater_than_neg4_1_l1737_173750

theorem greater_than_neg4_1 (k : ℤ) (h1 : k = -4) : k > (-4.1 : ℝ) :=
by sorry

end greater_than_neg4_1_l1737_173750


namespace depth_of_pond_l1737_173789

theorem depth_of_pond (L W V D : ℝ) (hL : L = 20) (hW : W = 10) (hV : V = 1000) (hV_formula : V = L * W * D) : D = 5 := by
  -- at this point, you could start the proof which involves deriving D from hV and hV_formula using arithmetic rules.
  sorry

end depth_of_pond_l1737_173789


namespace paintable_area_correct_l1737_173756

-- Defining lengths
def bedroom_length : ℕ := 15
def bedroom_width : ℕ := 11
def bedroom_height : ℕ := 9

-- Defining the number of bedrooms
def num_bedrooms : ℕ := 4

-- Defining the total area not to be painted per bedroom
def area_not_painted_per_bedroom : ℕ := 80

-- The total wall area calculation
def total_wall_area_per_bedroom : ℕ :=
  2 * (bedroom_length * bedroom_height) + 2 * (bedroom_width * bedroom_height)

-- The paintable wall area per bedroom calculation
def paintable_area_per_bedroom : ℕ :=
  total_wall_area_per_bedroom - area_not_painted_per_bedroom

-- The total paintable area across all bedrooms calculation
def total_paintable_area : ℕ :=
  paintable_area_per_bedroom * num_bedrooms

-- The theorem statement
theorem paintable_area_correct : total_paintable_area = 1552 := by
  sorry -- Proof is omitted

end paintable_area_correct_l1737_173756


namespace classroom_position_l1737_173704

theorem classroom_position (a b c d : ℕ) (h : (1, 2) = (a, b)) : (3, 2) = (c, d) :=
by
  sorry

end classroom_position_l1737_173704


namespace min_dist_circle_to_line_l1737_173702

noncomputable def circle_eq (x y : ℝ) := x^2 + y^2 - 2*x - 2*y

noncomputable def line_eq (x y : ℝ) := x + y - 8

theorem min_dist_circle_to_line : 
  (∀ x y : ℝ, circle_eq x y = 0 → ∃ d : ℝ, d ≥ 0 ∧ 
    (∀ x₁ y₁ : ℝ, circle_eq x₁ y₁ = 0 → ∀ x₂ y₂ : ℝ, line_eq x₂ y₂ = 0 → d ≤ dist (x₁, y₁) (x₂, y₂)) ∧ 
    d = 2 * Real.sqrt 2) :=
by
  sorry

end min_dist_circle_to_line_l1737_173702


namespace students_passed_both_tests_l1737_173787

theorem students_passed_both_tests
  (n : ℕ) (A : ℕ) (B : ℕ) (C : ℕ)
  (h1 : n = 100) 
  (h2 : A = 60) 
  (h3 : B = 40) 
  (h4 : C = 20) :
  A + B - ((n - C) - (A + B - n)) = 20 :=
by
  sorry

end students_passed_both_tests_l1737_173787


namespace compute_sixth_power_sum_l1737_173781

theorem compute_sixth_power_sum (ζ1 ζ2 ζ3 : ℂ) 
  (h1 : ζ1 + ζ2 + ζ3 = 2)
  (h2 : ζ1^2 + ζ2^2 + ζ3^2 = 5)
  (h3 : ζ1^4 + ζ2^4 + ζ3^4 = 29) :
  ζ1^6 + ζ2^6 + ζ3^6 = 101.40625 := 
by
  sorry

end compute_sixth_power_sum_l1737_173781


namespace sqrt_of_16_is_4_l1737_173741

def arithmetic_square_root (x : ℕ) : ℕ :=
  if x = 0 then 0 else Nat.sqrt x

theorem sqrt_of_16_is_4 : arithmetic_square_root 16 = 4 :=
by
  sorry

end sqrt_of_16_is_4_l1737_173741


namespace intersection_correct_l1737_173743

noncomputable def set_M : Set ℝ := { x | x^2 + x - 6 ≤ 0 }
noncomputable def set_N : Set ℝ := { x | abs (2 * x + 1) > 3 }
noncomputable def set_intersection : Set ℝ := { x | (x ∈ set_M) ∧ (x ∈ set_N) }

theorem intersection_correct : 
  set_intersection = { x : ℝ | (-3 ≤ x ∧ x < -2) ∨ (1 < x ∧ x ≤ 2) } := 
by 
  sorry

end intersection_correct_l1737_173743


namespace Randy_drew_pictures_l1737_173715

variable (P Q R: ℕ)

def Peter_drew_pictures (P : ℕ) : Prop := P = 8
def Quincy_drew_pictures (Q P : ℕ) : Prop := Q = P + 20
def Total_drawing (R P Q : ℕ) : Prop := R + P + Q = 41

theorem Randy_drew_pictures
  (P_eq : Peter_drew_pictures P)
  (Q_eq : Quincy_drew_pictures Q P)
  (Total_eq : Total_drawing R P Q) :
  R = 5 :=
by 
  sorry

end Randy_drew_pictures_l1737_173715


namespace extreme_value_f_at_a_eq_1_monotonic_intervals_f_exists_no_common_points_l1737_173799

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x - a * Real.log (x - 1)

-- Problem 1: Prove that the extreme value of f(x) when a = 1 is \frac{3}{4} + \ln 2
theorem extreme_value_f_at_a_eq_1 : 
  f (3/2) 1 = 3/4 + Real.log 2 :=
sorry

-- Problem 2: Prove the monotonic intervals of f(x) based on the value of a
theorem monotonic_intervals_f :
  ∀ a : ℝ, 
    (if a ≤ 0 then 
      ∀ x, 1 < x → f x' a > 0
     else
      ∀ x, 1 < x ∧ x ≤ (a + 2) / 2 → f x a ≤ 0 ∧ ∀ x, x ≥ (a + 2) / 2 → f x a > 0) :=
sorry

-- Problem 3: Prove that for a ≥ 1, there exists an a such that f(x) has no common points with y = \frac{5}{8} + \ln 2
theorem exists_no_common_points (h : 1 ≤ a) :
  ∃ x : ℝ, f x a ≠ 5/8 + Real.log 2 :=
sorry

end extreme_value_f_at_a_eq_1_monotonic_intervals_f_exists_no_common_points_l1737_173799


namespace number_with_20_multiples_l1737_173705

theorem number_with_20_multiples : ∃ n : ℕ, (∀ k : ℕ, (1 ≤ k) → (k ≤ 100) → (n ∣ k) → (k / n ≤ 20) ) ∧ n = 5 := 
  sorry

end number_with_20_multiples_l1737_173705


namespace boat_ratio_l1737_173746

theorem boat_ratio (b c d1 d2 : ℝ) 
  (h1 : b = 20) 
  (h2 : c = 4) 
  (h3 : d1 = 4) 
  (h4 : d2 = 2) : 
  (d1 + d2) / ((d1 / (b + c)) + (d2 / (b - c))) / b = 36 / 35 :=
by 
  sorry

end boat_ratio_l1737_173746


namespace circle_equation_standard_form_l1737_173719

theorem circle_equation_standard_form (x y : ℝ) :
  (∃ (center : ℝ × ℝ), center.1 = -1 ∧ center.2 = 2 * center.1 ∧ (center.2 = -2) ∧ (center.1 + 1)^2 + center.2^2 = 4 ∧ (center.1 = -1) ∧ (center.2 = -2)) ->
  (x + 1)^2 + (y + 2)^2 = 4 :=
sorry

end circle_equation_standard_form_l1737_173719


namespace square_pieces_placement_l1737_173725

theorem square_pieces_placement (n : ℕ) (H : n = 8) :
  {m : ℕ // m = 17} :=
sorry

end square_pieces_placement_l1737_173725


namespace A_intersection_B_eq_intersection_set_l1737_173726

def A : Set ℝ := {x : ℝ | x * (x - 2) < 0}
def B : Set ℝ := {x : ℝ | x > 1}
def intersection_set := {x : ℝ | 1 < x ∧ x < 2}

theorem A_intersection_B_eq_intersection_set : A ∩ B = intersection_set := by
  sorry

end A_intersection_B_eq_intersection_set_l1737_173726


namespace minimize_material_l1737_173716

theorem minimize_material (π V R h : ℝ) (hV : V > 0) (h_cond : π * R^2 * h = V) :
  R = h / 2 :=
sorry

end minimize_material_l1737_173716


namespace find_x_l1737_173730

theorem find_x (x y : ℤ) (hx : x > y) (hy : y > 0)
  (coins_megan : ℤ := 42)
  (coins_shana : ℤ := 35)
  (shana_win : ℕ := 2)
  (total_megan : shana_win * x + (total_races - shana_win) * y = coins_shana)
  (total_shana : (total_races - shana_win) * x + shana_win * y = coins_megan) :
  x = 4 := by
  sorry

end find_x_l1737_173730


namespace resistor_parallel_l1737_173728

theorem resistor_parallel (x y r : ℝ)
  (h1 : x = 5)
  (h2 : r = 2.9166666666666665)
  (h3 : 1 / r = 1 / x + 1 / y) : y = 7 :=
by
  -- proof omitted
  sorry

end resistor_parallel_l1737_173728


namespace range_of_x_range_of_a_l1737_173792

-- Definitions of propositions p and q
def p (a x : ℝ) := (x - a) * (x - 3 * a) < 0
def q (x : ℝ) := (x - 3) / (x - 2) ≤ 0

-- Question 1
theorem range_of_x (a x : ℝ) : a = 1 → p a x ∧ q x → 2 < x ∧ x < 3 := by
  sorry

-- Question 2
theorem range_of_a (a : ℝ) : (∀ x, ¬p a x → ¬q x) → (∀ x, q x → p a x) → 1 < a ∧ a ≤ 2 := by
  sorry

end range_of_x_range_of_a_l1737_173792


namespace part_I_part_II_l1737_173744

noncomputable def f (a x : ℝ) : ℝ := a - 1 / (2^x + 1)

theorem part_I (a : ℝ) : ∀ x : ℝ, (0 < (2^x * Real.log 2) / (2^x + 1)^2) :=
by
  sorry

theorem part_II (h : ∀ x : ℝ, f a x = -f a (-x)) : 
  a = (1:ℝ)/2 ∧ ∀ x : ℝ, -((1:ℝ)/2) < f (1/2) x ∧ f (1/2) x < (1:ℝ)/2 :=
by
  sorry

end part_I_part_II_l1737_173744


namespace candy_given_away_l1737_173700

-- Define the conditions
def pieces_per_student := 2
def number_of_students := 9

-- Define the problem statement as a theorem
theorem candy_given_away : pieces_per_student * number_of_students = 18 := by
  -- This is where the proof would go, but we omit it with sorry.
  sorry

end candy_given_away_l1737_173700


namespace mechanical_pencils_fraction_l1737_173793

theorem mechanical_pencils_fraction (total_pencils : ℕ) (frac_mechanical : ℚ)
    (mechanical_pencils : ℕ) (standard_pencils : ℕ) (new_total_pencils : ℕ) 
    (new_standard_pencils : ℕ) (new_frac_mechanical : ℚ):
  total_pencils = 120 →
  frac_mechanical = 1 / 4 →
  mechanical_pencils = frac_mechanical * total_pencils →
  standard_pencils = total_pencils - mechanical_pencils →
  new_standard_pencils = 3 * standard_pencils →
  new_total_pencils = mechanical_pencils + new_standard_pencils →
  new_frac_mechanical = mechanical_pencils / new_total_pencils →
  new_frac_mechanical = 1 / 10 :=
by
  sorry

end mechanical_pencils_fraction_l1737_173793


namespace remainder_is_20_l1737_173772

theorem remainder_is_20 :
  ∀ (larger smaller quotient remainder : ℕ),
    (larger = 1634) →
    (larger - smaller = 1365) →
    (larger = quotient * smaller + remainder) →
    (quotient = 6) →
    remainder = 20 :=
by
  intros larger smaller quotient remainder h_larger h_difference h_division h_quotient
  sorry

end remainder_is_20_l1737_173772


namespace cleaner_flow_rate_after_second_unclogging_l1737_173723

theorem cleaner_flow_rate_after_second_unclogging
  (rate1 rate2 : ℕ) (time1 time2 total_time total_cleaner : ℕ)
  (used_cleaner1 used_cleaner2 : ℕ)
  (final_rate : ℕ)
  (H1 : rate1 = 2)
  (H2 : rate2 = 3)
  (H3 : time1 = 15)
  (H4 : time2 = 10)
  (H5 : total_time = 30)
  (H6 : total_cleaner = 80)
  (H7 : used_cleaner1 = rate1 * time1)
  (H8 : used_cleaner2 = rate2 * time2)
  (H9 : used_cleaner1 + used_cleaner2 ≤ total_cleaner)
  (H10 : final_rate = (total_cleaner - (used_cleaner1 + used_cleaner2)) / (total_time - (time1 + time2))) :
  final_rate = 4 := by
  sorry

end cleaner_flow_rate_after_second_unclogging_l1737_173723


namespace set_C_cannot_form_triangle_l1737_173764

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Given conditions
def set_A := (3, 6, 8)
def set_B := (3, 8, 9)
def set_C := (3, 6, 9)
def set_D := (6, 8, 9)

theorem set_C_cannot_form_triangle : ¬ is_triangle 3 6 9 :=
by
  -- Proof is omitted
  sorry

end set_C_cannot_form_triangle_l1737_173764


namespace luisa_mpg_l1737_173796

theorem luisa_mpg
  (d_grocery d_mall d_pet d_home : ℕ)
  (cost_per_gal total_cost : ℚ)
  (total_miles : ℕ )
  (total_gallons : ℚ)
  (mpg : ℚ):
  d_grocery = 10 →
  d_mall = 6 →
  d_pet = 5 →
  d_home = 9 →
  cost_per_gal = 3.5 →
  total_cost = 7 →
  total_miles = d_grocery + d_mall + d_pet + d_home →
  total_gallons = total_cost / cost_per_gal →
  mpg = total_miles / total_gallons →
  mpg = 15 :=
by
  intros
  sorry

end luisa_mpg_l1737_173796


namespace problem_proof_l1737_173701

open Set

noncomputable def A : Set ℝ := {x | abs (4 * x - 1) < 9}
noncomputable def B : Set ℝ := {x | x / (x + 3) ≥ 0}
noncomputable def complement_A : Set ℝ := {x | x ≤ -2 ∨ x ≥ 5 / 2}
noncomputable def correct_answer : Set ℝ := Iio (-3) ∪ Ici (5 / 2)

theorem problem_proof : (compl A) ∩ B = correct_answer := 
  by
    sorry

end problem_proof_l1737_173701


namespace inequality_a_b_c_d_l1737_173767

theorem inequality_a_b_c_d 
  (a b c d : ℝ) 
  (h0 : 0 ≤ a) 
  (h1 : a ≤ b) 
  (h2 : b ≤ c) 
  (h3 : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := 
by
  sorry

end inequality_a_b_c_d_l1737_173767


namespace sequence_a_n_sequence_b_n_range_k_l1737_173747

-- Define the geometric sequence {a_n} with initial conditions
def a (n : ℕ) : ℕ :=
  3 * 2^(n-1)

-- Define the sequence {b_n} with the given recurrence relation
def b : ℕ → ℕ
| 0 => 1
| (n+1) => 2 * (b n) + 1

theorem sequence_a_n (n : ℕ) : 
  (a n = 3 * 2^(n-1)) := sorry

theorem sequence_b_n (n : ℕ) :
  (b n = 2^n - 1) := sorry

-- Define the condition for k and the inequality
def condition_k (k : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → (k * (↑(b n) + 5) / 2 - 3 * 2^(n-1) ≥ 8*n + 2*k - 24)

-- Prove the range for k
theorem range_k (k : ℝ) :
  (condition_k k ↔ k ≥ 4) := sorry

end sequence_a_n_sequence_b_n_range_k_l1737_173747


namespace min_x2_y2_z2_given_condition_l1737_173798

theorem min_x2_y2_z2_given_condition (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : 
  ∃ (c : ℝ), c = 3 ∧ (∀ x y z : ℝ, x^3 + y^3 + z^3 - 3 * x * y * z = 8 → x^2 + y^2 + z^2 ≥ c) := 
sorry

end min_x2_y2_z2_given_condition_l1737_173798


namespace find_x_l1737_173707

variable (x : ℝ)

theorem find_x (h : 2 * x - 12 = -(x + 3)) : x = 3 := 
sorry

end find_x_l1737_173707


namespace find_x_plus_y_l1737_173761

-- Define the points A, B, and C with given conditions
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 1, y := 1}
def C : Point := {x := 2, y := 4}

-- Define what it means for C to divide AB in the ratio 2:1
open Point

def divides_in_ratio (A B C : Point) (r₁ r₂ : ℝ) :=
  (C.x = (r₁ * A.x + r₂ * B.x) / (r₁ + r₂))
  ∧ (C.y = (r₁ * A.y + r₂ * B.y) / (r₁ + r₂))

-- Prove that x + y = 8 given the conditions
theorem find_x_plus_y {x y : ℝ} (B : Point) (H_B : B = {x := x, y := y}) :
  divides_in_ratio A B C 2 1 →
  x + y = 8 :=
by
  intro h
  sorry

end find_x_plus_y_l1737_173761


namespace anna_total_value_l1737_173773

theorem anna_total_value (total_bills : ℕ) (five_dollar_bills : ℕ) (ten_dollar_bills : ℕ)
  (h1 : total_bills = 12) (h2 : five_dollar_bills = 4) (h3 : ten_dollar_bills = total_bills - five_dollar_bills) :
  5 * five_dollar_bills + 10 * ten_dollar_bills = 100 := by
  sorry

end anna_total_value_l1737_173773


namespace variance_of_X_is_correct_l1737_173775

/-!
  There is a batch of products, among which there are 12 genuine items and 4 defective items.
  If 3 items are drawn with replacement, and X represents the number of defective items drawn,
  prove that the variance of X is 9 / 16 given that X follows a binomial distribution B(3, 1 / 4).
-/

noncomputable def variance_of_binomial : Prop :=
  let n := 3
  let p := 1 / 4
  let variance := n * p * (1 - p)
  variance = 9 / 16

theorem variance_of_X_is_correct : variance_of_binomial := by
  sorry

end variance_of_X_is_correct_l1737_173775


namespace washing_machine_regular_wash_l1737_173727

variable {R : ℕ}

/-- A washing machine uses 20 gallons of water for a heavy wash,
2 gallons of water for a light wash, and an additional light wash
is added when bleach is used. Given conditions:
- Two heavy washes are done.
- Three regular washes are done.
- One light wash is done.
- Two loads are bleached.
- Total water used is 76 gallons.
Prove the washing machine uses 10 gallons of water for a regular wash. -/
theorem washing_machine_regular_wash (h : 2 * 20 + 3 * R + 1 * 2 + 2 * 2 = 76) : R = 10 :=
by
  sorry

end washing_machine_regular_wash_l1737_173727


namespace margie_drive_distance_l1737_173765

-- Conditions
def car_mpg : ℝ := 45  -- miles per gallon
def gas_price : ℝ := 5 -- dollars per gallon
def money_spent : ℝ := 25 -- dollars

-- Question: Prove that Margie can drive 225 miles with $25 worth of gas.
theorem margie_drive_distance (h1 : car_mpg = 45) (h2 : gas_price = 5) (h3 : money_spent = 25) :
  money_spent / gas_price * car_mpg = 225 := by
  sorry

end margie_drive_distance_l1737_173765


namespace max_ball_height_l1737_173776

/-- 
The height (in feet) of a ball traveling on a parabolic path is given by -20t^2 + 80t + 36,
where t is the time after launch. This theorem shows that the maximum height of the ball is 116 feet.
-/
theorem max_ball_height : ∃ t : ℝ, ∀ t', -20 * t^2 + 80 * t + 36 ≤ -20 * t'^2 + 80 * t' + 36 → -20 * t^2 + 80 * t + 36 = 116 :=
sorry

end max_ball_height_l1737_173776


namespace train_cross_time_l1737_173762

def length_of_train : Float := 135.0 -- in meters
def speed_of_train_kmh : Float := 45.0 -- in kilometers per hour
def length_of_bridge : Float := 240.03 -- in meters

def speed_of_train_ms : Float := speed_of_train_kmh * 1000.0 / 3600.0

def total_distance : Float := length_of_train + length_of_bridge

def time_to_cross : Float := total_distance / speed_of_train_ms

theorem train_cross_time : time_to_cross = 30.0024 :=
by
  sorry

end train_cross_time_l1737_173762


namespace problem_l1737_173771

def f (x : ℝ) : ℝ := sorry  -- f is a function from ℝ to ℝ

theorem problem (h : ∀ x : ℝ, 3 * f x + f (2 - x) = 4 * x^2 + 1) : f 5 = 133 / 4 := 
by 
  sorry -- the proof is omitted

end problem_l1737_173771


namespace basketball_players_l1737_173759

theorem basketball_players {total : ℕ} (total_boys : total = 22) 
                           (football_boys : ℕ) (football_boys_count : football_boys = 15) 
                           (neither_boys : ℕ) (neither_boys_count : neither_boys = 3) 
                           (both_boys : ℕ) (both_boys_count : both_boys = 18) : 
                           (total - neither_boys = 19) := 
by
  sorry

end basketball_players_l1737_173759


namespace positive_solution_range_l1737_173780

theorem positive_solution_range (a : ℝ) (h : a > 0) (x : ℝ) : (∃ x, (a / (x + 3) = 1 / 2) ∧ x > 0) ↔ a > 3 / 2 := by
  sorry

end positive_solution_range_l1737_173780


namespace mean_temperature_l1737_173706

def temperatures : List Int := [-8, -3, -3, -6, 2, 4, 1]

theorem mean_temperature :
  (temperatures.sum / temperatures.length : Int) = -2 := by
  sorry

end mean_temperature_l1737_173706


namespace find_t_of_decreasing_function_l1737_173779

theorem find_t_of_decreasing_function 
  (f : ℝ → ℝ)
  (h_decreasing : ∀ x y, x < y → f x > f y)
  (h_A : f 0 = 4)
  (h_B : f 3 = -2)
  (h_solution_set : ∀ x, |f (x + 1) - 1| < 3 ↔ -1 < x ∧ x < 2) :
  (1 : ℝ) = 1 :=
by
  sorry

end find_t_of_decreasing_function_l1737_173779


namespace factorize_expr_l1737_173784

theorem factorize_expr (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) :=
  sorry

end factorize_expr_l1737_173784


namespace turtles_remaining_l1737_173755

/-- 
In one nest, there are x baby sea turtles, while in the other nest, there are 2x baby sea turtles.
One-fourth of the turtles in the first nest and three-sevenths of the turtles in the second nest
got swept to the sea. Prove the total number of turtles still on the sand is (53/28)x.
-/
theorem turtles_remaining (x : ℕ) (h1 : ℕ := x) (h2 : ℕ := 2 * x) : ((3/4) * x + (8/7) * (2 * x)) = (53/28) * x :=
by
  sorry

end turtles_remaining_l1737_173755


namespace todd_has_40_left_after_paying_back_l1737_173712

def todd_snowcone_problem : Prop :=
  let borrowed := 100
  let repay := 110
  let cost_ingredients := 75
  let snowcones_sold := 200
  let price_per_snowcone := 0.75
  let total_earnings := snowcones_sold * price_per_snowcone
  let remaining_money := total_earnings - repay
  remaining_money = 40

theorem todd_has_40_left_after_paying_back : todd_snowcone_problem :=
by
  -- Add proof here if needed
  sorry

end todd_has_40_left_after_paying_back_l1737_173712


namespace geometric_sequence_value_of_b_l1737_173739

theorem geometric_sequence_value_of_b : 
  ∃ b : ℝ, 180 * (b / 180) = b ∧ (b / 180) * b = 64 / 25 ∧ b > 0 ∧ b = 21.6 :=
by sorry

end geometric_sequence_value_of_b_l1737_173739


namespace cistern_fill_time_l1737_173790

theorem cistern_fill_time (hA : ℝ) (hB : ℝ) (hC : ℝ) : hA = 12 → hB = 18 → hC = 15 → 
  1 / ((1 / hA) + (1 / hB) - (1 / hC)) = 180 / 13 :=
by
  intros hA_eq hB_eq hC_eq
  rw [hA_eq, hB_eq, hC_eq]
  sorry

end cistern_fill_time_l1737_173790


namespace range_of_a_l1737_173766

noncomputable def e := Real.exp 1

theorem range_of_a (a : Real) 
  (h : ∀ x : Real, 1 ≤ x ∧ x ≤ 2 → Real.exp x - a ≥ 0) : 
  a ≤ e :=
by
  sorry

end range_of_a_l1737_173766


namespace percentage_of_percentage_l1737_173737

theorem percentage_of_percentage (a b : ℝ) (h_a : a = 0.03) (h_b : b = 0.05) : (a / b) * 100 = 60 :=
by
  sorry

end percentage_of_percentage_l1737_173737


namespace area_of_flowerbed_l1737_173794

theorem area_of_flowerbed :
  ∀ (a b : ℕ), 2 * (a + b) = 24 → b + 1 = 3 * (a + 1) → 
  let shorter_side := 3 * a
  let longer_side := 3 * b
  shorter_side * longer_side = 144 :=
by
  sorry

end area_of_flowerbed_l1737_173794


namespace largest_integer_of_four_l1737_173788

theorem largest_integer_of_four (A B C D : ℤ)
  (h_diff: A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_order: A < B ∧ B < C ∧ C < D)
  (h_avg: (A + B + C + D) / 4 = 74)
  (h_A_min: A ≥ 29) : D = 206 :=
by
  sorry

end largest_integer_of_four_l1737_173788


namespace unique_integer_solution_l1737_173717

theorem unique_integer_solution :
  ∃! (z : ℤ), 5 * z ≤ 2 * z - 8 ∧ -3 * z ≥ 18 ∧ 7 * z ≤ -3 * z - 21 :=
by
  sorry

end unique_integer_solution_l1737_173717


namespace four_sin_t_plus_cos_2t_bounds_l1737_173791

theorem four_sin_t_plus_cos_2t_bounds (t : ℝ) : -5 ≤ 4 * Real.sin t + Real.cos (2 * t) ∧ 4 * Real.sin t + Real.cos (2 * t) ≤ 3 := by
  sorry

end four_sin_t_plus_cos_2t_bounds_l1737_173791


namespace new_ratio_of_boarders_to_day_scholars_l1737_173714

theorem new_ratio_of_boarders_to_day_scholars
  (B_initial D_initial : ℕ)
  (B_initial_eq : B_initial = 560)
  (ratio_initial : B_initial / D_initial = 7 / 16)
  (new_boarders : ℕ)
  (new_boarders_eq : new_boarders = 80)
  (B_new : ℕ)
  (B_new_eq : B_new = B_initial + new_boarders)
  (D_new : ℕ)
  (D_new_eq : D_new = D_initial) :
  B_new / D_new = 1 / 2 :=
by
  sorry

end new_ratio_of_boarders_to_day_scholars_l1737_173714


namespace probability_of_consecutive_triplets_l1737_173763

def total_ways_to_select_3_days (n : ℕ) : ℕ :=
  Nat.choose n 3

def number_of_consecutive_triplets (n : ℕ) : ℕ :=
  n - 2

theorem probability_of_consecutive_triplets :
  let total_ways := total_ways_to_select_3_days 10
  let consecutive_triplets := number_of_consecutive_triplets 10
  (consecutive_triplets : ℚ) / total_ways = 1 / 15 :=
by
  sorry

end probability_of_consecutive_triplets_l1737_173763


namespace range_of_x_for_odd_function_l1737_173768

theorem range_of_x_for_odd_function (f : ℝ → ℝ) (domain : Set ℝ)
  (h_odd : ∀ x ∈ domain, f (-x) = -f x)
  (h_mono : ∀ x y, 0 < x -> x < y -> f x < f y)
  (h_f3 : f 3 = 0)
  (h_ineq : ∀ x, x ∈ domain -> x * (f x - f (-x)) < 0) : 
  ∀ x, x * f x < 0 ↔ -3 < x ∧ x < 0 ∨ 0 < x ∧ x < 3 :=
by sorry

end range_of_x_for_odd_function_l1737_173768


namespace complement_of_M_in_U_l1737_173797

def U := Set.univ (α := ℝ)
def M := {x : ℝ | x < -2 ∨ x > 8}
def compl_M := {x : ℝ | -2 ≤ x ∧ x ≤ 8}

theorem complement_of_M_in_U : compl_M = U \ M :=
by
  sorry

end complement_of_M_in_U_l1737_173797


namespace anns_age_l1737_173735

theorem anns_age (a b : ℕ) (h1 : a + b = 54) 
(h2 : b = a - (a - b) + (a - b)): a = 29 :=
sorry

end anns_age_l1737_173735


namespace mikes_age_is_18_l1737_173732

-- Define variables for Mike's age (m) and his uncle's age (u)
variables (m u : ℕ)

-- Condition 1: Mike is 18 years younger than his uncle
def condition1 : Prop := m = u - 18

-- Condition 2: The sum of their ages is 54 years
def condition2 : Prop := m + u = 54

-- Statement: Prove that Mike's age is 18 given the conditions
theorem mikes_age_is_18 (h1 : condition1 m u) (h2 : condition2 m u) : m = 18 :=
by
  -- Proof skipped with sorry
  sorry

end mikes_age_is_18_l1737_173732


namespace complement_intersection_l1737_173785

open Set

variable {R : Type} [LinearOrderedField R]

def P : Set R := {x | x^2 - 2*x ≥ 0}
def Q : Set R := {x | 1 < x ∧ x ≤ 3}

theorem complement_intersection : (compl P ∩ Q) = {x : R | 1 < x ∧ x < 2} := by
  sorry

end complement_intersection_l1737_173785


namespace fraction_left_handed_non_throwers_is_one_third_l1737_173774

theorem fraction_left_handed_non_throwers_is_one_third :
  let total_players := 70
  let throwers := 31
  let right_handed := 57
  let non_throwers := total_players - throwers
  let right_handed_non_throwers := right_handed - throwers
  let left_handed_non_throwers := non_throwers - right_handed_non_throwers
  (left_handed_non_throwers : ℝ) / non_throwers = 1 / 3 := by
  sorry

end fraction_left_handed_non_throwers_is_one_third_l1737_173774


namespace correct_remainder_l1737_173718

-- Define the problem
def count_valid_tilings (n k : Nat) : Nat :=
  Nat.factorial (n + k) / (Nat.factorial n * Nat.factorial k) * (3 ^ (n + k) - 3 * 2 ^ (n + k) + 3)

noncomputable def tiles_mod_1000 : Nat :=
  let pairs := [(8, 0), (6, 1), (4, 2), (2, 3), (0, 4)]
  let M := pairs.foldl (λ acc (nk : Nat × Nat) => acc + count_valid_tilings nk.1 nk.2) 0
  M % 1000

theorem correct_remainder : tiles_mod_1000 = 328 :=
  by sorry

end correct_remainder_l1737_173718


namespace cover_points_with_two_disks_l1737_173769

theorem cover_points_with_two_disks :
  ∀ (points : Fin 2014 → ℝ × ℝ),
    (∀ (i j k : Fin 2014), i ≠ j → j ≠ k → i ≠ k → 
      dist (points i) (points j) ≤ 1 ∨ dist (points j) (points k) ≤ 1 ∨ dist (points i) (points k) ≤ 1) →
    ∃ (A B : ℝ × ℝ), ∀ (p : Fin 2014),
      dist (points p) A ≤ 1 ∨ dist (points p) B ≤ 1 :=
by
  sorry

end cover_points_with_two_disks_l1737_173769


namespace triangle_side_length_x_l1737_173731

theorem triangle_side_length_x
  (y : ℝ) (z : ℝ) (cos_Y_minus_Z : ℝ)
  (hy : y = 7)
  (hz : z = 3)
  (hcos : cos_Y_minus_Z = 7 / 8) :
  ∃ x : ℝ, x = Real.sqrt 18.625 :=
by
  sorry

end triangle_side_length_x_l1737_173731


namespace lambs_goats_solution_l1737_173711

theorem lambs_goats_solution : ∃ l g : ℕ, l > 0 ∧ g > 0 ∧ 30 * l + 32 * g = 1200 ∧ l = 24 ∧ g = 15 :=
by
  existsi 24
  existsi 15
  repeat { split }
  sorry

end lambs_goats_solution_l1737_173711


namespace quadratic_completion_l1737_173786

theorem quadratic_completion :
  (∀ x : ℝ, (∃ a h k : ℝ, (x ^ 2 - 2 * x - 1 = a * (x - h) ^ 2 + k) ∧ (a = 1) ∧ (h = 1) ∧ (k = -2))) :=
sorry

end quadratic_completion_l1737_173786


namespace sally_total_expense_l1737_173754

-- Definitions based on the problem conditions
def peaches_price_after_coupon : ℝ := 12.32
def peaches_coupon : ℝ := 3.00
def cherries_weight : ℝ := 2.00
def cherries_price_per_kg : ℝ := 11.54
def apples_weight : ℝ := 4.00
def apples_price_per_kg : ℝ := 5.00
def apples_discount_percentage : ℝ := 0.15
def oranges_count : ℝ := 6.00
def oranges_price_per_unit : ℝ := 1.25
def oranges_promotion : ℝ := 3.00 -- Buy 2, get 1 free means she pays for 4 out of 6

-- Calculation of the total expense
def total_expense : ℝ :=
  (peaches_price_after_coupon + peaches_coupon) + 
  (cherries_weight * cherries_price_per_kg) + 
  ((apples_weight * apples_price_per_kg) * (1 - apples_discount_percentage)) +
  (4 * oranges_price_per_unit)

-- Statement to verify total expense
theorem sally_total_expense : total_expense = 60.40 := by
  sorry

end sally_total_expense_l1737_173754


namespace ratio_female_to_male_l1737_173738

theorem ratio_female_to_male
  (a b c : ℕ)
  (ha : a = 60)
  (hb : b = 80)
  (hc : c = 65) :
  f / m = 1 / 3 := 
by
  sorry

end ratio_female_to_male_l1737_173738


namespace square_side_length_l1737_173753

-- Define the given dimensions and total length
def rectangle_width : ℕ := 2
def total_length : ℕ := 7

-- Define the unknown side length of the square
variable (Y : ℕ)

-- State the problem and provide the conclusion
theorem square_side_length : Y + rectangle_width = total_length -> Y = 5 :=
by 
  sorry

end square_side_length_l1737_173753


namespace restore_original_price_l1737_173733

theorem restore_original_price (original_price promotional_price : ℝ) (h₀ : original_price = 1) (h₁ : promotional_price = original_price * 0.8) : (original_price - promotional_price) / promotional_price = 0.25 :=
by sorry

end restore_original_price_l1737_173733


namespace jack_bill_age_difference_l1737_173752

theorem jack_bill_age_difference :
  ∃ (a b : ℕ), (0 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (7 * a - 29 * b = 14) ∧ ((10 * a + b) - (10 * b + a) = 36) :=
by
  sorry

end jack_bill_age_difference_l1737_173752


namespace division_remainder_l1737_173713

theorem division_remainder :
  ∃ (R D Q : ℕ), D = 3 * Q ∧ D = 3 * R + 3 ∧ 251 = D * Q + R ∧ R = 8 := by
  sorry

end division_remainder_l1737_173713


namespace terrier_hush_interval_l1737_173777

-- Definitions based on conditions
def poodle_barks_per_terrier_bark : ℕ := 2
def total_poodle_barks : ℕ := 24
def terrier_hushes : ℕ := 6

-- Derived values based on definitions
def total_terrier_barks := total_poodle_barks / poodle_barks_per_terrier_bark
def interval_hush := total_terrier_barks / terrier_hushes

-- The theorem stating the terrier's hush interval
theorem terrier_hush_interval : interval_hush = 2 := by
  have h1 : total_terrier_barks = 12 := by sorry
  have h2 : interval_hush = 2 := by sorry
  exact h2

end terrier_hush_interval_l1737_173777


namespace find_speed_of_current_l1737_173758

variable {m c : ℝ}

theorem find_speed_of_current
  (h1 : m + c = 15)
  (h2 : m - c = 10) :
  c = 2.5 :=
sorry

end find_speed_of_current_l1737_173758


namespace find_b_l1737_173721

theorem find_b (a b c y1 y2 : ℝ) (h1 : y1 = a * 2^2 + b * 2 + c) 
              (h2 : y2 = a * (-2)^2 + b * (-2) + c) 
              (h3 : y1 - y2 = -12) : b = -3 :=
by 
  sorry

end find_b_l1737_173721


namespace expand_expression_l1737_173740

theorem expand_expression (x y : ℝ) : 24 * (3 * x - 4 * y + 6) = 72 * x - 96 * y + 144 := 
by
  sorry

end expand_expression_l1737_173740


namespace number_of_students_l1737_173709

theorem number_of_students 
  (P S : ℝ)
  (total_cost : ℝ) 
  (percent_free : ℝ) 
  (lunch_cost : ℝ)
  (h1 : percent_free = 0.40)
  (h2 : total_cost = 210)
  (h3 : lunch_cost = 7)
  (h4 : P = 0.60 * S)
  (h5 : P * lunch_cost = total_cost) :
  S = 50 :=
by
  sorry

end number_of_students_l1737_173709


namespace b_negative_l1737_173724

variable {R : Type*} [LinearOrderedField R]

theorem b_negative (a b : R) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∀ x : R, 0 ≤ x → (x - a) * (x - b) * (x - (2*a + b)) ≥ 0) : b < 0 := 
sorry

end b_negative_l1737_173724


namespace wooden_block_even_blue_faces_l1737_173710

theorem wooden_block_even_blue_faces :
  let length := 6
  let width := 6
  let height := 2
  let total_cubes := length * width * height
  let corners := 8
  let edges_not_corners := 24
  let faces_not_edges := 24
  let interior := 16
  let even_blue_faces := edges_not_corners + interior
  total_cubes = 72 →
  even_blue_faces = 40 :=
by
  sorry

end wooden_block_even_blue_faces_l1737_173710


namespace problem1_problem2_problem3_problem4_l1737_173748

theorem problem1 : (-3 + 8 - 7 - 15) = -17 := 
sorry

theorem problem2 : (23 - 6 * (-3) + 2 * (-4)) = 33 := 
sorry

theorem problem3 : (-8 / (4 / 5) * (-2 / 3)) = 20 / 3 := 
sorry

theorem problem4 : (-2^2 - 9 * (-1 / 3)^2 + abs (-4)) = -1 := 
sorry

end problem1_problem2_problem3_problem4_l1737_173748


namespace prob_A_wins_match_is_correct_l1737_173782

/-- Definitions -/

def prob_A_wins_game : ℝ := 0.6

def prob_B_wins_game : ℝ := 1 - prob_A_wins_game

def prob_A_wins_match (p: ℝ) : ℝ :=
  p * p * (1 - p) + p * (1 - p) * p + p * p

/-- Theorem -/

theorem prob_A_wins_match_is_correct : 
  prob_A_wins_match prob_A_wins_game = 0.648 :=
by
  sorry

end prob_A_wins_match_is_correct_l1737_173782
