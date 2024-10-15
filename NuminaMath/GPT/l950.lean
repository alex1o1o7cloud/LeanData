import Mathlib

namespace NUMINAMATH_GPT_eq_fractions_l950_95068

theorem eq_fractions : 
  (1 + 1 / (1 + 1 / (1 + 1 / 2))) = 8 / 5 := 
  sorry

end NUMINAMATH_GPT_eq_fractions_l950_95068


namespace NUMINAMATH_GPT_caitlin_age_l950_95054

theorem caitlin_age (aunt_anna_age : ℕ) (brianna_age : ℕ) (caitlin_age : ℕ) 
  (h1 : aunt_anna_age = 60)
  (h2 : brianna_age = aunt_anna_age / 3)
  (h3 : caitlin_age = brianna_age - 7)
  : caitlin_age = 13 :=
by
  sorry

end NUMINAMATH_GPT_caitlin_age_l950_95054


namespace NUMINAMATH_GPT_area_of_quadrilateral_l950_95044

noncomputable def quadrilateral_area
  (AB CD r : ℝ) (k : ℝ) 
  (h_perpendicular : AB * CD = 0)
  (h_equal_diameters : AB = 2 * r ∧ CD = 2 * r)
  (h_ratio : BC / AD = k) : ℝ := 
  (3 * r^2 * abs (1 - k^2)) / (1 + k^2)

theorem area_of_quadrilateral
  (AB CD r : ℝ) (k : ℝ)
  (h_perpendicular : AB * CD = 0)
  (h_equal_diameters : AB = 2 * r ∧ CD = 2 * r)
  (h_ratio : BC / AD = k) :
  quadrilateral_area AB CD r k h_perpendicular h_equal_diameters h_ratio = (3 * r^2 * abs (1 - k^2)) / (1 + k^2) :=
sorry

end NUMINAMATH_GPT_area_of_quadrilateral_l950_95044


namespace NUMINAMATH_GPT_find_k_value_l950_95026

variable (S : ℕ → ℤ) (n : ℕ)

-- Conditions
def is_arithmetic_sum (S : ℕ → ℤ) : Prop :=
  ∃ (a d : ℤ), ∀ n : ℕ, S n = n * (2 * a + (n - 1) * d) / 2

axiom S3_eq_S8 (S : ℕ → ℤ) (hS : is_arithmetic_sum S) : S 3 = S 8
axiom Sk_eq_S7 (S : ℕ → ℤ) (k : ℕ) (hS: is_arithmetic_sum S)  : S 7 = S k

theorem find_k_value (S : ℕ → ℤ) (hS: is_arithmetic_sum S) :  S 3 = S 8 → S 7 = S 4 :=
by
  sorry

end NUMINAMATH_GPT_find_k_value_l950_95026


namespace NUMINAMATH_GPT_Jessie_l950_95046

theorem Jessie's_friends (total_muffins : ℕ) (muffins_per_person : ℕ) (num_people : ℕ) :
  total_muffins = 20 → muffins_per_person = 4 → num_people = total_muffins / muffins_per_person → num_people - 1 = 4 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_Jessie_l950_95046


namespace NUMINAMATH_GPT_fraction_of_paper_per_book_l950_95034

theorem fraction_of_paper_per_book (total_fraction_used : ℚ) (num_books : ℕ) (h1 : total_fraction_used = 5 / 8) (h2 : num_books = 5) : 
  (total_fraction_used / num_books) = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_paper_per_book_l950_95034


namespace NUMINAMATH_GPT_four_digit_even_and_multiple_of_7_sum_l950_95082

def num_four_digit_even_numbers : ℕ := 4500
def num_four_digit_multiples_of_7 : ℕ := 1286
def C : ℕ := num_four_digit_even_numbers
def D : ℕ := num_four_digit_multiples_of_7

theorem four_digit_even_and_multiple_of_7_sum :
  C + D = 5786 := by
  sorry

end NUMINAMATH_GPT_four_digit_even_and_multiple_of_7_sum_l950_95082


namespace NUMINAMATH_GPT_final_price_after_increase_and_decrease_l950_95001

variable (P : ℝ)

theorem final_price_after_increase_and_decrease (h : P > 0) : 
  let increased_price := P * 1.15
  let final_price := increased_price * 0.85
  final_price = P * 0.9775 :=
by
  sorry

end NUMINAMATH_GPT_final_price_after_increase_and_decrease_l950_95001


namespace NUMINAMATH_GPT_sell_price_equal_percentage_l950_95078

theorem sell_price_equal_percentage (SP : ℝ) (CP : ℝ) :
  (SP - CP) / CP * 100 = (CP - 1280) / CP * 100 → 
  (1937.5 = CP + 0.25 * CP) → 
  SP = 1820 :=
by 
  -- Note: skip proof with sorry
  apply sorry

end NUMINAMATH_GPT_sell_price_equal_percentage_l950_95078


namespace NUMINAMATH_GPT_expression_odd_if_p_q_odd_l950_95093

variable (p q : ℕ)

def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem expression_odd_if_p_q_odd (hp : is_odd p) (hq : is_odd q) : is_odd (5 * p * q) :=
sorry

end NUMINAMATH_GPT_expression_odd_if_p_q_odd_l950_95093


namespace NUMINAMATH_GPT_min_value_of_squared_sums_l950_95085

theorem min_value_of_squared_sums (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : 
  ∃ B, (B = x^2 + y^2 + z^2) ∧ (B ≥ 4) := 
by {
  sorry -- Proof will be provided here.
}

end NUMINAMATH_GPT_min_value_of_squared_sums_l950_95085


namespace NUMINAMATH_GPT_average_age_union_l950_95002

open Real

variables {a b c d A B C D : ℝ}

theorem average_age_union (h1 : A / a = 40)
                         (h2 : B / b = 30)
                         (h3 : C / c = 45)
                         (h4 : D / d = 35)
                         (h5 : (A + B) / (a + b) = 37)
                         (h6 : (A + C) / (a + c) = 42)
                         (h7 : (A + D) / (a + d) = 39)
                         (h8 : (B + C) / (b + c) = 40)
                         (h9 : (B + D) / (b + d) = 37)
                         (h10 : (C + D) / (c + d) = 43) : 
  (A + B + C + D) / (a + b + c + d) = 44.5 := 
sorry

end NUMINAMATH_GPT_average_age_union_l950_95002


namespace NUMINAMATH_GPT_proper_subset_of_A_l950_95009

def A : Set ℝ := {x | x^2 < 5 * x}

theorem proper_subset_of_A :
  (∀ x, x ∈ Set.Ioc 1 5 → x ∈ A ∧ ∀ y, y ∈ A → y ∉ Set.Ioc 1 5 → ¬(Set.Ioc 1 5 = A)) :=
sorry

end NUMINAMATH_GPT_proper_subset_of_A_l950_95009


namespace NUMINAMATH_GPT_perfect_square_trinomial_l950_95048

theorem perfect_square_trinomial (b : ℝ) : 
  (∃ (x : ℝ), 4 * x^2 + b * x + 1 = (2 * x + 1) ^ 2) ↔ (b = 4 ∨ b = -4) := 
by 
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l950_95048


namespace NUMINAMATH_GPT_length_of_CD_l950_95057

theorem length_of_CD
    (AB BC AC AD CD : ℝ)
    (h1 : AB = 6)
    (h2 : BC = 1 / 2 * AB)
    (h3 : AC = AB + BC)
    (h4 : AD = AC)
    (h5 : CD = AD + AC) :
    CD = 18 := by
  sorry

end NUMINAMATH_GPT_length_of_CD_l950_95057


namespace NUMINAMATH_GPT_problem_l950_95067

noncomputable def a : ℝ := Real.exp 1 - 2
noncomputable def b : ℝ := 1 - Real.log 2
noncomputable def c : ℝ := Real.exp (Real.exp 1) - Real.exp 2

theorem problem (a_def : a = Real.exp 1 - 2) 
                (b_def : b = 1 - Real.log 2) 
                (c_def : c = Real.exp (Real.exp 1) - Real.exp 2) : 
                c > a ∧ a > b := 
by 
  rw [a_def, b_def, c_def]
  sorry

end NUMINAMATH_GPT_problem_l950_95067


namespace NUMINAMATH_GPT_geom_seq_common_ratio_l950_95061

noncomputable def log_custom_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem geom_seq_common_ratio (a : ℝ) :
  let u₁ := a + log_custom_base 2 3
  let u₂ := a + log_custom_base 4 3
  let u₃ := a + log_custom_base 8 3
  u₂ / u₁ = u₃ / u₂ →
  u₂ / u₁ = 1 / 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_geom_seq_common_ratio_l950_95061


namespace NUMINAMATH_GPT_minimum_expression_value_l950_95098

theorem minimum_expression_value (a b c : ℝ) (hbpos : b > 0) (hab : b > a) (hcb : b > c) (hca : c > a) :
  (a + 2 * b) ^ 2 / b ^ 2 + (b - 2 * c) ^ 2 / b ^ 2 + (c - 2 * a) ^ 2 / b ^ 2 ≥ 65 / 16 := 
sorry

end NUMINAMATH_GPT_minimum_expression_value_l950_95098


namespace NUMINAMATH_GPT_four_divides_sum_of_squares_iff_even_l950_95015

theorem four_divides_sum_of_squares_iff_even (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (4 ∣ (a^2 + b^2 + c^2)) ↔ (Even a ∧ Even b ∧ Even c) :=
by
  sorry

end NUMINAMATH_GPT_four_divides_sum_of_squares_iff_even_l950_95015


namespace NUMINAMATH_GPT_eq_three_div_x_one_of_eq_l950_95063

theorem eq_three_div_x_one_of_eq (x : ℝ) (hx : 1 - 6 / x + 9 / (x ^ 2) = 0) : (3 / x) = 1 :=
sorry

end NUMINAMATH_GPT_eq_three_div_x_one_of_eq_l950_95063


namespace NUMINAMATH_GPT_rotate180_of_point_A_l950_95073

-- Define the point A and the transformation
def point_A : ℝ × ℝ := (-3, 2)
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- Theorem statement for the problem
theorem rotate180_of_point_A :
  rotate180 point_A = (3, -2) :=
sorry

end NUMINAMATH_GPT_rotate180_of_point_A_l950_95073


namespace NUMINAMATH_GPT_time_released_rope_first_time_l950_95022

theorem time_released_rope_first_time :
  ∀ (rate_ascent : ℕ) (rate_descent : ℕ) (time_first_ascent : ℕ) (time_second_ascent : ℕ) (highest_elevation : ℕ)
    (total_elevation_gained : ℕ) (elevation_difference : ℕ) (time_descent : ℕ),
  rate_ascent = 50 →
  rate_descent = 10 →
  time_first_ascent = 15 →
  time_second_ascent = 15 →
  highest_elevation = 1400 →
  total_elevation_gained = (rate_ascent * time_first_ascent) + (rate_ascent * time_second_ascent) →
  elevation_difference = total_elevation_gained - highest_elevation →
  time_descent = elevation_difference / rate_descent →
  time_descent = 10 :=
by
  intros rate_ascent rate_descent time_first_ascent time_second_ascent highest_elevation total_elevation_gained elevation_difference time_descent
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_time_released_rope_first_time_l950_95022


namespace NUMINAMATH_GPT_rain_difference_l950_95042

theorem rain_difference
    (rain_monday : ℕ → ℝ)
    (rain_tuesday : ℕ → ℝ)
    (rain_wednesday : ℕ → ℝ)
    (rain_thursday : ℕ → ℝ)
    (h_monday : ∀ n : ℕ, n = 10 → rain_monday n = 1.25)
    (h_tuesday : ∀ n : ℕ, n = 12 → rain_tuesday n = 2.15)
    (h_wednesday : ∀ n : ℕ, n = 8 → rain_wednesday n = 1.60)
    (h_thursday : ∀ n : ℕ, n = 6 → rain_thursday n = 2.80) :
    let total_rain_monday := 10 * 1.25
    let total_rain_tuesday := 12 * 2.15
    let total_rain_wednesday := 8 * 1.60
    let total_rain_thursday := 6 * 2.80
    (total_rain_tuesday + total_rain_thursday) - (total_rain_monday + total_rain_wednesday) = 17.3 :=
by
  sorry

end NUMINAMATH_GPT_rain_difference_l950_95042


namespace NUMINAMATH_GPT_troy_needs_additional_money_l950_95037

-- Defining the initial conditions
def price_of_new_computer : ℕ := 80
def initial_savings : ℕ := 50
def money_from_selling_old_computer : ℕ := 20

-- Defining the question and expected answer
def required_additional_money : ℕ :=
  price_of_new_computer - (initial_savings + money_from_selling_old_computer)

-- The proof statement
theorem troy_needs_additional_money : required_additional_money = 10 := by
  sorry

end NUMINAMATH_GPT_troy_needs_additional_money_l950_95037


namespace NUMINAMATH_GPT_power_mod_equality_l950_95029

theorem power_mod_equality (n : ℕ) : 
  (47 % 8 = 7) → (23 % 8 = 7) → (47 ^ 2500 - 23 ^ 2500) % 8 = 0 := 
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_power_mod_equality_l950_95029


namespace NUMINAMATH_GPT_solution_set_of_f_gt_0_range_of_m_l950_95027

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 1) - abs (x + 2)

theorem solution_set_of_f_gt_0 :
  {x : ℝ | f x > 0} = {x : ℝ | x < -1 / 3} ∪ {x | x > 3} :=
by sorry

theorem range_of_m (m : ℝ) :
  (∃ x_0 : ℝ, f x_0 + 2 * m^2 < 4 * m) ↔ -1 / 2 < m ∧ m < 5 / 2 :=
by sorry

end NUMINAMATH_GPT_solution_set_of_f_gt_0_range_of_m_l950_95027


namespace NUMINAMATH_GPT_find_geometric_sequence_values_l950_95062

structure GeometricSequence (a b c d : ℝ) : Prop where
  ratio1 : b / a = c / b
  ratio2 : c / b = d / c

theorem find_geometric_sequence_values (x u v y : ℝ)
    (h1 : x + y = 20)
    (h2 : u + v = 34)
    (h3 : x^2 + u^2 + v^2 + y^2 = 1300) :
    (GeometricSequence x u v y ∧ ((x = 16 ∧ u = 4 ∧ v = 32 ∧ y = 2) ∨ (x = 4 ∧ u = 16 ∧ v = 2 ∧ y = 32))) :=
by
  sorry

end NUMINAMATH_GPT_find_geometric_sequence_values_l950_95062


namespace NUMINAMATH_GPT_infinite_triangles_with_sides_x_y_10_l950_95025

theorem infinite_triangles_with_sides_x_y_10 (x y : Nat) (hx : 0 < x) (hy : 0 < y) : 
  (∃ n : Nat, n > 5 ∧ ∀ m ≥ n, ∃ x y : Nat, 0 < x ∧ 0 < y ∧ x + y > 10 ∧ x + 10 > y ∧ y + 10 > x) :=
sorry

end NUMINAMATH_GPT_infinite_triangles_with_sides_x_y_10_l950_95025


namespace NUMINAMATH_GPT_there_are_six_bases_ending_in_one_for_625_in_decimal_l950_95040

theorem there_are_six_bases_ending_in_one_for_625_in_decimal :
  (∃ ls : List ℕ, ls = [2, 3, 4, 6, 8, 12] ∧ ∀ b ∈ ls, 2 ≤ b ∧ b ≤ 12 ∧ 624 % b = 0 ∧ List.length ls = 6) :=
by
  sorry

end NUMINAMATH_GPT_there_are_six_bases_ending_in_one_for_625_in_decimal_l950_95040


namespace NUMINAMATH_GPT_gcd_polynomials_l950_95055

def P (n : ℤ) : ℤ := n^3 - 6 * n^2 + 11 * n - 6
def Q (n : ℤ) : ℤ := n^2 - 4 * n + 4

theorem gcd_polynomials (n : ℤ) (h : n ≥ 3) : Int.gcd (P n) (Q n) = n - 2 :=
by
  sorry

end NUMINAMATH_GPT_gcd_polynomials_l950_95055


namespace NUMINAMATH_GPT_greg_age_is_18_l950_95089

def diana_age : ℕ := 15
def eduardo_age (c : ℕ) : ℕ := 2 * c
def chad_age (c : ℕ) : ℕ := c
def faye_age (c : ℕ) : ℕ := c - 1
def greg_age (c : ℕ) : ℕ := 2 * (c - 1)
def diana_relation (c : ℕ) : Prop := 15 = (2 * c) - 5

theorem greg_age_is_18 (c : ℕ) (h : diana_relation c) :
  greg_age c = 18 :=
by
  sorry

end NUMINAMATH_GPT_greg_age_is_18_l950_95089


namespace NUMINAMATH_GPT_celina_total_cost_l950_95053

def hoodieCost : ℝ := 80
def hoodieTaxRate : ℝ := 0.05

def flashlightCost := 0.20 * hoodieCost
def flashlightTaxRate : ℝ := 0.10

def bootsInitialCost : ℝ := 110
def bootsDiscountRate : ℝ := 0.10
def bootsTaxRate : ℝ := 0.05

def waterFilterCost : ℝ := 65
def waterFilterDiscountRate : ℝ := 0.25
def waterFilterTaxRate : ℝ := 0.08

def campingMatCost : ℝ := 45
def campingMatDiscountRate : ℝ := 0.15
def campingMatTaxRate : ℝ := 0.08

def backpackCost : ℝ := 105
def backpackTaxRate : ℝ := 0.08

def totalCost : ℝ := 
  let hoodieTotal := (hoodieCost * (1 + hoodieTaxRate))
  let flashlightTotal := (flashlightCost * (1 + flashlightTaxRate))
  let bootsTotal := ((bootsInitialCost * (1 - bootsDiscountRate)) * (1 + bootsTaxRate))
  let waterFilterTotal := ((waterFilterCost * (1 - waterFilterDiscountRate)) * (1 + waterFilterTaxRate))
  let campingMatTotal := ((campingMatCost * (1 - campingMatDiscountRate)) * (1 + campingMatTaxRate))
  let backpackTotal := (backpackCost * (1 + backpackTaxRate))
  hoodieTotal + flashlightTotal + bootsTotal + waterFilterTotal + campingMatTotal + backpackTotal

theorem celina_total_cost: totalCost = 413.91 := by
  sorry

end NUMINAMATH_GPT_celina_total_cost_l950_95053


namespace NUMINAMATH_GPT_incorrect_desc_is_C_l950_95038
noncomputable def incorrect_geometric_solid_desc : Prop :=
  ¬ (∀ (plane_parallel: Prop), 
      plane_parallel ∧ 
      (∀ (frustum: Prop), frustum ↔ 
        (∃ (base section_cut cone : Prop), 
          cone ∧ 
          (section_cut = plane_parallel) ∧ 
          (frustum = (base ∧ section_cut)))))

theorem incorrect_desc_is_C (plane_parallel frustum base section_cut cone : Prop) :
  incorrect_geometric_solid_desc := 
by
  sorry

end NUMINAMATH_GPT_incorrect_desc_is_C_l950_95038


namespace NUMINAMATH_GPT_total_work_completion_days_l950_95013

theorem total_work_completion_days :
  let Amit_work_rate := 1 / 15
  let Ananthu_work_rate := 1 / 90
  let Chandra_work_rate := 1 / 45

  let Amit_days_worked_alone := 3
  let Ananthu_days_worked_alone := 6
  
  let work_by_Amit := Amit_days_worked_alone * Amit_work_rate
  let work_by_Ananthu := Ananthu_days_worked_alone * Ananthu_work_rate
  
  let initial_work_done := work_by_Amit + work_by_Ananthu
  let remaining_work := 1 - initial_work_done

  let combined_work_rate := Amit_work_rate + Ananthu_work_rate + Chandra_work_rate
  let days_all_worked_together := remaining_work / combined_work_rate

  Amit_days_worked_alone + Ananthu_days_worked_alone + days_all_worked_together = 17 :=
by
  sorry

end NUMINAMATH_GPT_total_work_completion_days_l950_95013


namespace NUMINAMATH_GPT_smallest_positive_integer_neither_prime_nor_square_no_prime_factor_less_than_50_l950_95077

def is_not_prime (n : ℕ) : Prop := ¬ Prime n

def is_not_square (n : ℕ) : Prop := ∀ m : ℕ, m * m ≠ n

def no_prime_factor_less_than_50 (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p ∣ n → p ≥ 50

theorem smallest_positive_integer_neither_prime_nor_square_no_prime_factor_less_than_50 :
  (∃ n : ℕ, 0 < n ∧ is_not_prime n ∧ is_not_square n ∧ no_prime_factor_less_than_50 n ∧
  (∀ m : ℕ, 0 < m ∧ is_not_prime m ∧ is_not_square m ∧ no_prime_factor_less_than_50 m → n ≤ m)) →
  ∃ n : ℕ, n = 3127 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_positive_integer_neither_prime_nor_square_no_prime_factor_less_than_50_l950_95077


namespace NUMINAMATH_GPT_sum_series_eq_11_div_18_l950_95010

theorem sum_series_eq_11_div_18 :
  (∑' n : ℕ, if n = 0 then 0 else 1 / (n * (n + 3))) = 11 / 18 :=
by
  sorry

end NUMINAMATH_GPT_sum_series_eq_11_div_18_l950_95010


namespace NUMINAMATH_GPT_expression_evaluation_l950_95084

theorem expression_evaluation (a b : ℕ) (h1 : a = 25) (h2 : b = 15) : (a + b)^2 - (a^2 + b^2) = 750 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l950_95084


namespace NUMINAMATH_GPT_speed_of_man_in_still_water_l950_95031

variable (v_m v_s : ℝ)

theorem speed_of_man_in_still_water
  (h1 : (v_m + v_s) * 4 = 24)
  (h2 : (v_m - v_s) * 5 = 20) :
  v_m = 5 := 
sorry

end NUMINAMATH_GPT_speed_of_man_in_still_water_l950_95031


namespace NUMINAMATH_GPT_vertex_on_x_axis_iff_t_eq_neg_4_l950_95004

theorem vertex_on_x_axis_iff_t_eq_neg_4 (t : ℝ) :
  (∃ x : ℝ, (4 + t) = 0) ↔ t = -4 :=
by
  sorry

end NUMINAMATH_GPT_vertex_on_x_axis_iff_t_eq_neg_4_l950_95004


namespace NUMINAMATH_GPT_greta_received_more_letters_l950_95036

noncomputable def number_of_letters_difference : ℕ :=
  let B := 40
  let M (G : ℕ) := 2 * (G + B)
  let total (G : ℕ) := G + B + M G
  let G := 50 -- Solved from the total equation
  G - B

theorem greta_received_more_letters : number_of_letters_difference = 10 :=
by
  sorry

end NUMINAMATH_GPT_greta_received_more_letters_l950_95036


namespace NUMINAMATH_GPT_least_k_divisible_480_l950_95058

theorem least_k_divisible_480 (k : ℕ) (h : k^4 % 480 = 0) : k = 101250 :=
sorry

end NUMINAMATH_GPT_least_k_divisible_480_l950_95058


namespace NUMINAMATH_GPT_factorization_of_expression_l950_95024

theorem factorization_of_expression (a b c : ℝ) : 
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / 
  ((a - b)^3 + (b - c)^3 + (c - a)^3) = 
  (a^2 + ab + b^2) * (b^2 + bc + c^2) * (c^2 + ca + a^2) :=
by
  sorry

end NUMINAMATH_GPT_factorization_of_expression_l950_95024


namespace NUMINAMATH_GPT_solve_equation_l950_95028

theorem solve_equation :
  ∃ x : ℝ, (x + 2) / 4 - (2 * x - 3) / 6 = 2 ∧ x = -12 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l950_95028


namespace NUMINAMATH_GPT_sum_of_intercepts_l950_95095

theorem sum_of_intercepts (a b c : ℕ) :
  (∃ y, x = 2 * y^2 - 6 * y + 3 ∧ x = a ∧ y = 0) ∧
  (∃ y1 y2, x = 0 ∧ 2 * y1^2 - 6 * y1 + 3 = 0 ∧ 2 * y2^2 - 6 * y2 + 3 = 0 ∧ y1 + y2 = b + c) →
  a + b + c = 6 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_intercepts_l950_95095


namespace NUMINAMATH_GPT_value_of_expression_l950_95097

theorem value_of_expression : (165^2 - 153^2) / 12 = 318 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l950_95097


namespace NUMINAMATH_GPT_cost_of_tree_planting_l950_95003

theorem cost_of_tree_planting 
  (initial_temp final_temp : ℝ) (temp_drop_per_tree cost_per_tree : ℝ) 
  (h_initial: initial_temp = 80) (h_final: final_temp = 78.2) 
  (h_temp_drop_per_tree: temp_drop_per_tree = 0.1) 
  (h_cost_per_tree: cost_per_tree = 6) : 
  (final_temp - initial_temp) / temp_drop_per_tree * cost_per_tree = 108 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_tree_planting_l950_95003


namespace NUMINAMATH_GPT_speed_of_second_train_l950_95020

-- Definitions of conditions
def distance_train1 : ℝ := 200
def speed_train1 : ℝ := 50
def distance_train2 : ℝ := 240
def time_train1_and_train2 : ℝ := 4

-- Statement of the problem
theorem speed_of_second_train : (distance_train2 / time_train1_and_train2) = 60 := by
  sorry

end NUMINAMATH_GPT_speed_of_second_train_l950_95020


namespace NUMINAMATH_GPT_quadratic_completion_l950_95030

theorem quadratic_completion :
  ∀ x : ℝ, (x^2 - 4*x + 1 = 0) ↔ ((x - 2)^2 = 3) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_completion_l950_95030


namespace NUMINAMATH_GPT_people_between_katya_and_polina_l950_95051

-- Definitions based on given conditions
def is_next_to (a b : ℕ) : Prop := (b = a + 1) ∨ (b = a - 1)
def position_alena : ℕ := 1
def position_lena : ℕ := 5
def position_sveta (pos_sveta : ℕ) : Prop := pos_sveta + 1 = position_lena
def position_katya (pos_katya : ℕ) : Prop := pos_katya = 3
def position_polina (pos_polina : ℕ) : Prop := (is_next_to position_alena pos_polina)

-- The question: prove the number of people between Katya and Polina is 0
theorem people_between_katya_and_polina : 
  ∃ (pos_katya pos_polina : ℕ),
    position_katya pos_katya ∧ 
    position_polina pos_polina ∧ 
    pos_polina + 1 = pos_katya ∧
    pos_katya = 3 ∧ pos_polina = 2 := 
sorry

end NUMINAMATH_GPT_people_between_katya_and_polina_l950_95051


namespace NUMINAMATH_GPT_boat_speed_still_water_l950_95049

theorem boat_speed_still_water : 
  ∀ (b s : ℝ), (b + s = 11) → (b - s = 5) → b = 8 := 
by 
  intros b s h1 h2
  sorry

end NUMINAMATH_GPT_boat_speed_still_water_l950_95049


namespace NUMINAMATH_GPT_find_a14_l950_95069

-- Define the arithmetic sequence properties
def sum_of_first_n_terms (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * a1 + (n * (n - 1) / 2) * d

def nth_term (a1 d : ℤ) (n : ℕ) : ℤ :=
  a1 + (n - 1) * d

theorem find_a14 (a1 d : ℤ) (S11 : sum_of_first_n_terms a1 d 11 = 55)
  (a10 : nth_term a1 d 10 = 9) : nth_term a1 d 14 = 13 :=
sorry

end NUMINAMATH_GPT_find_a14_l950_95069


namespace NUMINAMATH_GPT_find_h_parallel_line_l950_95016

theorem find_h_parallel_line:
  ∃ h : ℚ, (3 * (h : ℚ) - 2 * (24 : ℚ) = 7) → (h = 47 / 3) :=
by
  sorry

end NUMINAMATH_GPT_find_h_parallel_line_l950_95016


namespace NUMINAMATH_GPT_white_balls_count_l950_95065

theorem white_balls_count {T W : ℕ} (h1 : 3 * 4 = T) (h2 : T - 3 = W) : W = 9 :=
by 
    sorry

end NUMINAMATH_GPT_white_balls_count_l950_95065


namespace NUMINAMATH_GPT_jennifer_spent_124_dollars_l950_95081

theorem jennifer_spent_124_dollars 
  (initial_cans : ℕ := 40)
  (cans_per_set : ℕ := 5)
  (additional_cans_per_set : ℕ := 6)
  (total_cans_mark : ℕ := 30)
  (price_per_can_whole : ℕ := 2)
  (discount_threshold_whole : ℕ := 10)
  (discount_amount_whole : ℕ := 4) : 
  (initial_cans + additional_cans_per_set * (total_cans_mark / cans_per_set)) * price_per_can_whole - 
  (discount_amount_whole * ((initial_cans + additional_cans_per_set * (total_cans_mark / cans_per_set)) / discount_threshold_whole)) = 124 := by
  sorry

end NUMINAMATH_GPT_jennifer_spent_124_dollars_l950_95081


namespace NUMINAMATH_GPT_Vasya_not_11_more_than_Kolya_l950_95076

def is_L_shaped (n : ℕ) : Prop :=
  n % 2 = 1

def total_cells : ℕ :=
  14400

theorem Vasya_not_11_more_than_Kolya (k v : ℕ) :
  (is_L_shaped k) → (is_L_shaped v) → (k + v = total_cells) → (k % 2 = 0) → (v % 2 = 0) → (v - k ≠ 11) := 
by
  sorry

end NUMINAMATH_GPT_Vasya_not_11_more_than_Kolya_l950_95076


namespace NUMINAMATH_GPT_ratio_of_candies_l950_95006

theorem ratio_of_candies (emily_candies jennifer_candies bob_candies : ℕ)
  (h1 : emily_candies = 6)
  (h2 : bob_candies = 4)
  (h3 : jennifer_candies = 2 * emily_candies) : 
  jennifer_candies / bob_candies = 3 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_candies_l950_95006


namespace NUMINAMATH_GPT_contrapositive_example_l950_95033

theorem contrapositive_example 
  (x y : ℝ) (h : x^2 + y^2 = 0 → x = 0 ∧ y = 0) : 
  (x ≠ 0 ∨ y ≠ 0) → x^2 + y^2 ≠ 0 :=
sorry

end NUMINAMATH_GPT_contrapositive_example_l950_95033


namespace NUMINAMATH_GPT_find_m_if_f_even_l950_95007

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + (m - 2) * x + (m^2 - 7 * m + 12)

theorem find_m_if_f_even :
  (∀ x : ℝ, f m (-x) = f m x) → m = 2 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_find_m_if_f_even_l950_95007


namespace NUMINAMATH_GPT_digit_in_92nd_place_l950_95090

/-- The fraction 5/33 is expressed in decimal form as a repeating decimal 0.151515... -/
def fraction_to_decimal : ℚ := 5 / 33

/-- The repeated pattern in the decimal expansion of 5/33 is 15, which is a cycle of length 2 -/
def repeated_pattern (n : ℕ) : ℕ :=
  if n % 2 = 0 then 5 else 1

/-- The digit at the 92nd place in the decimal expansion of 5/33 is 5 -/
theorem digit_in_92nd_place : repeated_pattern 92 = 5 :=
by sorry

end NUMINAMATH_GPT_digit_in_92nd_place_l950_95090


namespace NUMINAMATH_GPT_two_colonies_limit_l950_95064

def doubles_each_day (size: ℕ) (day: ℕ) : ℕ := size * 2 ^ day

theorem two_colonies_limit (habitat_limit: ℕ) (initial_size: ℕ) : 
  (∀ t, doubles_each_day initial_size t = habitat_limit → t = 20) → 
  initial_size > 0 →
  ∀ t, doubles_each_day (2 * initial_size) t = habitat_limit → t = 20 :=
by
  sorry

end NUMINAMATH_GPT_two_colonies_limit_l950_95064


namespace NUMINAMATH_GPT_qin_jiushao_algorithm_correct_operations_l950_95023

def qin_jiushao_algorithm_operations (f : ℝ → ℝ) (x : ℝ) : ℕ × ℕ := sorry

def f (x : ℝ) : ℝ := 4 * x^5 - x^2 + 2
def x : ℝ := 3

theorem qin_jiushao_algorithm_correct_operations :
  qin_jiushao_algorithm_operations f x = (5, 2) :=
sorry

end NUMINAMATH_GPT_qin_jiushao_algorithm_correct_operations_l950_95023


namespace NUMINAMATH_GPT_psychologist_charge_difference_l950_95041

variables (F A : ℝ)

theorem psychologist_charge_difference
  (h1 : F + 4 * A = 375)
  (h2 : F + A = 174) :
  (F - A) = 40 :=
by sorry

end NUMINAMATH_GPT_psychologist_charge_difference_l950_95041


namespace NUMINAMATH_GPT_coordinates_after_5_seconds_l950_95045

-- Define the initial coordinates of point P
def initial_coordinates : ℚ × ℚ := (-10, 10)

-- Define the velocity vector of point P
def velocity_vector : ℚ × ℚ := (4, -3)

-- Asserting the coordinates of point P after 5 seconds
theorem coordinates_after_5_seconds : 
   initial_coordinates + 5 • velocity_vector = (10, -5) :=
by 
  sorry

end NUMINAMATH_GPT_coordinates_after_5_seconds_l950_95045


namespace NUMINAMATH_GPT_simplify_evaluate_expression_l950_95005

theorem simplify_evaluate_expression (a b : ℚ) (h1 : a = -2) (h2 : b = 1/5) :
    2 * a * b^2 - (6 * a^3 * b + 2 * (a * b^2 - (1/2) * a^3 * b)) = 8 := 
by
  sorry

end NUMINAMATH_GPT_simplify_evaluate_expression_l950_95005


namespace NUMINAMATH_GPT_length_of_train_l950_95071

-- Definitions of given conditions
def train_speed (kmh : ℤ) := 25
def man_speed (kmh : ℤ) := 2
def crossing_time (sec : ℤ) := 28

-- Relative speed calculation (in meters per second)
def relative_speed := (train_speed 1 + man_speed 1) * (5 / 18 : ℚ)

-- Distance calculation (in meters)
def distance_covered := relative_speed * (crossing_time 1 : ℚ)

-- The theorem statement: Length of the train equals distance covered in crossing time
theorem length_of_train : distance_covered = 210 := by
  sorry

end NUMINAMATH_GPT_length_of_train_l950_95071


namespace NUMINAMATH_GPT_v_is_82_875_percent_of_z_l950_95072

theorem v_is_82_875_percent_of_z (x y z w v : ℝ) 
  (h1 : x = 1.30 * y)
  (h2 : y = 0.60 * z)
  (h3 : w = 1.25 * x)
  (h4 : v = 0.85 * w) : 
  v = 0.82875 * z :=
by
  sorry

end NUMINAMATH_GPT_v_is_82_875_percent_of_z_l950_95072


namespace NUMINAMATH_GPT_teal_more_green_count_l950_95014

open Set

-- Define the survey data structure
def Survey : Type := {p : ℕ // p ≤ 150}

def people_surveyed : ℕ := 150
def more_blue (s : Survey) : Prop := sorry
def more_green (s : Survey) : Prop := sorry

-- Define the given conditions
def count_more_blue : ℕ := 90
def count_more_both : ℕ := 40
def count_neither : ℕ := 20

-- Define the proof statement
theorem teal_more_green_count :
  (count_more_both + (people_surveyed - (count_neither + (count_more_blue - count_more_both)))) = 80 :=
by {
  -- Sorry is used as a placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_teal_more_green_count_l950_95014


namespace NUMINAMATH_GPT_find_x_l950_95080

-- Given condition
def condition (x : ℝ) : Prop := 3 * x - 5 * x + 8 * x = 240

-- Statement (problem to prove)
theorem find_x (x : ℝ) (h : condition x) : x = 40 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_l950_95080


namespace NUMINAMATH_GPT_sum_of_digits_product_is_13_l950_95056

def base_eight_to_base_ten (n : ℕ) : ℕ := sorry
def product_base_eight (n1 n2 : ℕ) : ℕ := sorry
def digits_sum_base_ten (n : ℕ) : ℕ := sorry

theorem sum_of_digits_product_is_13 :
  let N1 := base_eight_to_base_ten 35
  let N2 := base_eight_to_base_ten 42
  let product := product_base_eight N1 N2
  digits_sum_base_ten product = 13 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_product_is_13_l950_95056


namespace NUMINAMATH_GPT_peter_age_problem_l950_95039

theorem peter_age_problem
  (P J : ℕ) 
  (h1 : J = P + 12)
  (h2 : P - 10 = 1/3 * (J - 10)) : P = 16 :=
sorry

end NUMINAMATH_GPT_peter_age_problem_l950_95039


namespace NUMINAMATH_GPT_denomination_of_second_note_l950_95043

theorem denomination_of_second_note
  (x : ℕ)
  (y : ℕ)
  (z : ℕ)
  (h1 : x = y)
  (h2 : y = z)
  (h3 : x + y + z = 75)
  (h4 : 1 * x + y * x + 10 * x = 400):
  y = 5 := by
  sorry

end NUMINAMATH_GPT_denomination_of_second_note_l950_95043


namespace NUMINAMATH_GPT_divisor_is_ten_l950_95091

variable (x y : ℝ)

theorem divisor_is_ten
  (h : ((5 * x - x / y) / (5 * x)) * 100 = 98) : y = 10 := by
  sorry

end NUMINAMATH_GPT_divisor_is_ten_l950_95091


namespace NUMINAMATH_GPT_ball_box_distribution_l950_95059

theorem ball_box_distribution:
  ∃ (C : ℕ → ℕ → ℕ) (A : ℕ → ℕ → ℕ),
  C 4 2 * A 3 3 = sorry := 
by sorry

end NUMINAMATH_GPT_ball_box_distribution_l950_95059


namespace NUMINAMATH_GPT_length_AB_l950_95088

theorem length_AB (x : ℝ) (h1 : 0 < x)
  (hG : G = (0 + 1) / 2)
  (hH : H = (0 + G) / 2)
  (hI : I = (0 + H) / 2)
  (hJ : J = (0 + I) / 2)
  (hAJ : J - 0 = 2) :
  x = 32 := by
  sorry

end NUMINAMATH_GPT_length_AB_l950_95088


namespace NUMINAMATH_GPT_distance_to_cut_pyramid_l950_95066

theorem distance_to_cut_pyramid (V A V1 : ℝ) (h1 : V > 0) (h2 : A > 0) :
  ∃ d : ℝ, d = (3 / A) * (V - (V^2 * (V - V1))^(1 / 3)) :=
by
  sorry

end NUMINAMATH_GPT_distance_to_cut_pyramid_l950_95066


namespace NUMINAMATH_GPT_red_balls_l950_95087

theorem red_balls (w r : ℕ) (h1 : w = 12) (h2 : w * 3 = r * 4) : r = 9 :=
sorry

end NUMINAMATH_GPT_red_balls_l950_95087


namespace NUMINAMATH_GPT_min_side_value_l950_95096

-- Definitions based on the conditions provided
variables (a b c : ℕ) (h1 : a - b = 5) (h2 : (a + b + c) % 2 = 0)

theorem min_side_value (h1 : a - b = 5) (h2 : (a + b + c) % 2 = 0) : c ≥ 7 :=
sorry

end NUMINAMATH_GPT_min_side_value_l950_95096


namespace NUMINAMATH_GPT_chords_in_circle_l950_95099

theorem chords_in_circle (n : ℕ) (h_n : n = 10) : (n.choose 2) = 45 := sorry

end NUMINAMATH_GPT_chords_in_circle_l950_95099


namespace NUMINAMATH_GPT_probability_of_color_change_l950_95074

def traffic_light_cycle := 90
def green_duration := 45
def yellow_duration := 5
def red_duration := 40
def green_to_yellow := green_duration
def yellow_to_red := green_duration + yellow_duration
def red_to_green := traffic_light_cycle
def observation_interval := 4
def valid_intervals := [green_to_yellow - observation_interval + 1, green_to_yellow, 
                        yellow_to_red - observation_interval + 1, yellow_to_red, 
                        red_to_green - observation_interval + 1, red_to_green]
def total_valid_intervals := valid_intervals.length * observation_interval

theorem probability_of_color_change : 
  (total_valid_intervals : ℚ) / traffic_light_cycle = 2 / 15 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_color_change_l950_95074


namespace NUMINAMATH_GPT_game_necessarily_ends_winning_strategy_l950_95086

-- Definitions and conditions based on problem:
def Card := Fin 2009

def isWhite (c : Fin 2009) : Prop := sorry -- Placeholder for actual white card predicate

def validMove (k : Fin 2009) : Prop := k.val < 1969 ∧ isWhite k

def applyMove (k : Fin 2009) (cards : Fin 2009 → Prop) : Fin 2009 → Prop :=
  fun c => if c.val ≥ k.val ∧ c.val < k.val + 41 then ¬isWhite c else isWhite c

-- Theorem statements to match proof problem:
theorem game_necessarily_ends : ∃ n, n = 2009 → (∀ (cards : Fin 2009 → Prop), (∃ k < 1969, validMove k) → (∀ k < 1969, ¬(validMove k))) :=
sorry

theorem winning_strategy (cards : Fin 2009 → Prop) : ∃ strategy : (Fin 2009 → Prop) → Fin 2009, ∀ s, (s = applyMove (strategy s) s) → strategy s = sorry :=
sorry

end NUMINAMATH_GPT_game_necessarily_ends_winning_strategy_l950_95086


namespace NUMINAMATH_GPT_optimal_order_l950_95075

variables (p1 p2 p3 : ℝ)
variables (hp3_lt_p1 : p3 < p1) (hp1_lt_p2 : p1 < p2)

theorem optimal_order (hcond1 : p2 * (p1 + p3 - p1 * p3) > p1 * (p2 + p3 - p2 * p3))
    : true :=
by {
  -- the details of the proof would go here, but we skip it with sorry
  sorry
}

end NUMINAMATH_GPT_optimal_order_l950_95075


namespace NUMINAMATH_GPT_solve_gcd_problem_l950_95000

def gcd_problem : Prop :=
  gcd 153 119 = 17

theorem solve_gcd_problem : gcd_problem :=
  by
    sorry

end NUMINAMATH_GPT_solve_gcd_problem_l950_95000


namespace NUMINAMATH_GPT_marika_father_age_twice_l950_95050

theorem marika_father_age_twice (t : ℕ) (h : t = 2036) :
  let marika_age := 10 + (t - 2006)
  let father_age := 50 + (t - 2006)
  father_age = 2 * marika_age :=
by {
  -- let marika_age := 10 + (t - 2006),
  -- let father_age := 50 + (t - 2006),
  sorry
}

end NUMINAMATH_GPT_marika_father_age_twice_l950_95050


namespace NUMINAMATH_GPT_number_of_eggs_l950_95008

-- Define the conditions as assumptions
variables (marbles : ℕ) (eggs : ℕ)
variables (eggs_A eggs_B eggs_C : ℕ)
variables (marbles_A marbles_B marbles_C : ℕ)

-- Conditions from the problem
axiom eggs_total : marbles = 4
axiom marbles_total : eggs = 15
axiom eggs_groups : eggs_A ≠ eggs_B ∧ eggs_B ≠ eggs_C ∧ eggs_A ≠ eggs_C
axiom marbles_diff1 : marbles_B - marbles_A = eggs_B
axiom marbles_diff2 : marbles_C - marbles_B = eggs_C

-- Prove that the number of eggs in each group is as specified in the answer
theorem number_of_eggs :
  eggs_A = 12 ∧ eggs_B = 1 ∧ eggs_C = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_eggs_l950_95008


namespace NUMINAMATH_GPT_number_of_integers_l950_95092

theorem number_of_integers (n : ℤ) : 
    (100 < n ∧ n < 300) ∧ (n % 7 = n % 9) → 
    (∃ count: ℕ, count = 21) := by
  sorry

end NUMINAMATH_GPT_number_of_integers_l950_95092


namespace NUMINAMATH_GPT_determine_constants_l950_95012

theorem determine_constants (a b c d : ℝ) 
  (periodic : (2 * (2 * Real.pi / b) = 4 * Real.pi))
  (vert_shift : d = 3)
  (max_val : (d + a = 8))
  (min_val : (d - a = -2)) :
  a = 5 ∧ b = 1 :=
by
  sorry

end NUMINAMATH_GPT_determine_constants_l950_95012


namespace NUMINAMATH_GPT_bus_stop_minutes_per_hour_l950_95079

/-- Given the average speed of a bus excluding stoppages is 60 km/hr
and including stoppages is 15 km/hr, prove that the bus stops for 45 minutes per hour. -/
theorem bus_stop_minutes_per_hour
  (speed_no_stops : ℝ := 60)
  (speed_with_stops : ℝ := 15) :
  ∃ t : ℝ, t = 45 :=
by
  sorry

end NUMINAMATH_GPT_bus_stop_minutes_per_hour_l950_95079


namespace NUMINAMATH_GPT_smallest_portion_bread_l950_95094

theorem smallest_portion_bread (a d : ℚ) (h1 : 5 * a = 100) (h2 : 24 * d = 11 * a) :
  a - 2 * d = 5 / 3 :=
by
  -- Solution proof goes here...
  sorry -- placeholder for the proof

end NUMINAMATH_GPT_smallest_portion_bread_l950_95094


namespace NUMINAMATH_GPT_mean_of_combined_sets_l950_95083

theorem mean_of_combined_sets (mean_set1 mean_set2 : ℝ) (n1 n2 : ℕ) 
  (h1 : mean_set1 = 15) (h2 : mean_set2 = 20) (h3 : n1 = 5) (h4 : n2 = 8) :
  (n1 * mean_set1 + n2 * mean_set2) / (n1 + n2) = 235 / 13 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_combined_sets_l950_95083


namespace NUMINAMATH_GPT_total_area_of_field_l950_95035

noncomputable def total_field_area (A1 A2 : ℝ) : ℝ := A1 + A2

theorem total_area_of_field :
  ∀ (A1 A2 : ℝ),
    A1 = 405 ∧ (A2 - A1 = (1/5) * ((A1 + A2) / 2)) →
    total_field_area A1 A2 = 900 :=
by
  intros A1 A2 h
  sorry

end NUMINAMATH_GPT_total_area_of_field_l950_95035


namespace NUMINAMATH_GPT_muffin_machine_completion_time_l950_95017

theorem muffin_machine_completion_time :
  let start_time := 9 * 60 -- minutes
  let partial_completion_time := (12 * 60) + 15 -- minutes
  let partial_duration := partial_completion_time - start_time
  let fraction_of_day := 1 / 4
  let total_duration := partial_duration / fraction_of_day
  start_time + total_duration = (22 * 60) := -- 10:00 PM in minutes
by
  sorry

end NUMINAMATH_GPT_muffin_machine_completion_time_l950_95017


namespace NUMINAMATH_GPT_exists_int_x_l950_95021

theorem exists_int_x (K M N : ℤ) (h1 : K ≠ 0) (h2 : M ≠ 0) (h3 : N ≠ 0) (h_coprime : Int.gcd K M = 1) :
  ∃ x : ℤ, K ∣ (M * x + N) :=
by
  sorry

end NUMINAMATH_GPT_exists_int_x_l950_95021


namespace NUMINAMATH_GPT_page_cost_in_cents_l950_95019

theorem page_cost_in_cents (notebooks pages_per_notebook total_cost : ℕ)
  (h_notebooks : notebooks = 2)
  (h_pages_per_notebook : pages_per_notebook = 50)
  (h_total_cost : total_cost = 5 * 100) :
  (total_cost / (notebooks * pages_per_notebook)) = 5 :=
by
  sorry

end NUMINAMATH_GPT_page_cost_in_cents_l950_95019


namespace NUMINAMATH_GPT_find_a5_over_T9_l950_95052

-- Define arithmetic sequences and their sums
variables {a_n : ℕ → ℚ} {b_n : ℕ → ℚ}
variables {S_n : ℕ → ℚ} {T_n : ℕ → ℚ}

-- Conditions
def arithmetic_seq_a (a_n : ℕ → ℚ) : Prop :=
  ∀ n, a_n n = a_n 1 + (n - 1) * (a_n 2 - a_n 1)

def arithmetic_seq_b (b_n : ℕ → ℚ) : Prop :=
  ∀ n, b_n n = b_n 1 + (n - 1) * (b_n 2 - b_n 1)

def sum_a (S_n : ℕ → ℚ) (a_n : ℕ → ℚ) : Prop :=
  ∀ n, S_n n = n * (a_n 1 + a_n n) / 2

def sum_b (T_n : ℕ → ℚ) (b_n : ℕ → ℚ) : Prop :=
  ∀ n, T_n n = n * (b_n 1 + b_n n) / 2

def given_condition (S_n : ℕ → ℚ) (T_n : ℕ → ℚ) : Prop :=
  ∀ n, S_n n / T_n n = (n + 3) / (2 * n - 1)

-- Goal statement
theorem find_a5_over_T9 (h_a : arithmetic_seq_a a_n) (h_b : arithmetic_seq_b b_n)
  (sum_a_S : sum_a S_n a_n) (sum_b_T : sum_b T_n b_n) (cond : given_condition S_n T_n) :
  a_n 5 / T_n 9 = 4 / 51 :=
  sorry

end NUMINAMATH_GPT_find_a5_over_T9_l950_95052


namespace NUMINAMATH_GPT_min_value_inequality_l950_95032

theorem min_value_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 3) :
  (1 / (a + b)) + (1 / (b + c)) + (1 / (c + a)) ≥ 3 / 2 :=
sorry

end NUMINAMATH_GPT_min_value_inequality_l950_95032


namespace NUMINAMATH_GPT_inequality_proof_l950_95047

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 + 4 * a / (b + c)) * (1 + 4 * b / (c + a)) * (1 + 4 * c / (a + b)) > 25 := by
  sorry

end NUMINAMATH_GPT_inequality_proof_l950_95047


namespace NUMINAMATH_GPT_tangent_triangle_area_l950_95060

noncomputable def area_of_tangent_triangle : ℝ :=
  let f : ℝ → ℝ := fun x => Real.log x
  let f' : ℝ → ℝ := fun x => 1 / x
  let tangent_line : ℝ → ℝ := fun x => x - 1
  let x_intercept : ℝ := 1
  let y_intercept : ℝ := -1
  let base := 1
  let height := 1
  (1 / 2) * base * height

theorem tangent_triangle_area :
  area_of_tangent_triangle = 1 / 2 :=
sorry

end NUMINAMATH_GPT_tangent_triangle_area_l950_95060


namespace NUMINAMATH_GPT_arithmetic_sequence_a3_l950_95018

theorem arithmetic_sequence_a3 (a : ℕ → ℕ) (h1 : a 6 = 6) (h2 : a 9 = 9) : a 3 = 3 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a3_l950_95018


namespace NUMINAMATH_GPT_num_enemies_left_l950_95011

-- Definitions of conditions
def points_per_enemy : Nat := 5
def total_enemies : Nat := 8
def earned_points : Nat := 10

-- Theorem statement to prove the number of undefeated enemies
theorem num_enemies_left (points_per_enemy total_enemies earned_points : Nat) : 
    (earned_points / points_per_enemy) <= total_enemies →
    total_enemies - (earned_points / points_per_enemy) = 6 := by
  sorry

end NUMINAMATH_GPT_num_enemies_left_l950_95011


namespace NUMINAMATH_GPT_willie_final_stickers_l950_95070

-- Conditions
def willie_start_stickers : ℝ := 36.0
def emily_gives_willie : ℝ := 7.0

-- Theorem
theorem willie_final_stickers : willie_start_stickers + emily_gives_willie = 43.0 :=
by
  sorry

end NUMINAMATH_GPT_willie_final_stickers_l950_95070
