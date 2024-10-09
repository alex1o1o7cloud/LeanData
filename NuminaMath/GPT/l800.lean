import Mathlib

namespace stocking_stuffers_total_l800_80031

-- Defining the number of items per category
def candy_canes := 4
def beanie_babies := 2
def books := 1
def small_toys := 3
def gift_cards := 1

-- Total number of stocking stuffers per child
def items_per_child := candy_canes + beanie_babies + books + small_toys + gift_cards

-- Number of children
def number_of_children := 3

-- Total number of stocking stuffers for all children
def total_stocking_stuffers := items_per_child * number_of_children

-- Statement to be proved
theorem stocking_stuffers_total : total_stocking_stuffers = 33 := by
  sorry

end stocking_stuffers_total_l800_80031


namespace min_value_ineq_l800_80045

noncomputable def problem_statement : Prop :=
  ∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ x + y = 4 ∧ (∀ (a b : ℝ), 0 < a → 0 < b → a + b = 4 → (1/a + 4/b) ≥ 9/4)

theorem min_value_ineq : problem_statement :=
by
  unfold problem_statement
  sorry

end min_value_ineq_l800_80045


namespace orange_probability_l800_80025

theorem orange_probability (total_apples : ℕ) (total_oranges : ℕ) (other_fruits : ℕ)
  (h1 : total_apples = 20) (h2 : total_oranges = 10) (h3 : other_fruits = 0) :
  (total_oranges : ℚ) / (total_apples + total_oranges + other_fruits) = 1 / 3 :=
by
  sorry

end orange_probability_l800_80025


namespace square_area_increase_l800_80098

variable (a : ℕ)

theorem square_area_increase (a : ℕ) :
  (a + 6) ^ 2 - a ^ 2 = 12 * a + 36 :=
by
  sorry

end square_area_increase_l800_80098


namespace cow_count_16_l800_80079

theorem cow_count_16 (D C : ℕ) 
  (h1 : ∃ (L H : ℕ), L = 2 * D + 4 * C ∧ H = D + C ∧ L = 2 * H + 32) : C = 16 :=
by
  obtain ⟨L, H, ⟨hL, hH, hCond⟩⟩ := h1
  sorry

end cow_count_16_l800_80079


namespace triangle_inequality_l800_80032

open Real

variables {a b c S : ℝ}

-- Assuming a, b, c are the sides of a triangle
axiom triangle_sides : a > 0 ∧ b > 0 ∧ c > 0
-- Assuming S is the area of the triangle
axiom Herons_area : S = sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))

theorem triangle_inequality : 
  a^2 + b^2 + c^2 ≥ 4 * S * sqrt 3 ∧ (a^2 + b^2 + c^2 = 4 * S * sqrt 3 ↔ a = b ∧ b = c) := sorry

end triangle_inequality_l800_80032


namespace sequence_non_zero_l800_80088

theorem sequence_non_zero :
  ∀ n : ℕ, ∃ a : ℕ → ℤ,
  (a 1 = 1) ∧
  (a 2 = 2) ∧
  (∀ n : ℕ, (a (n+1) % 2 = 1 ∧ a n % 2 = 1) → (a (n+2) = 5 * a (n+1) - 3 * a n)) ∧
  (∀ n : ℕ, (a (n+1) % 2 = 0 ∧ a n % 2 = 0) → (a (n+2) = a (n+1) - a n)) ∧
  (a n ≠ 0) :=
by
  sorry

end sequence_non_zero_l800_80088


namespace value_of_M_after_subtracting_10_percent_l800_80072

-- Define the given conditions and desired result formally in Lean 4
theorem value_of_M_after_subtracting_10_percent (M : ℝ) (h : 0.25 * M = 0.55 * 2500) :
  M - 0.10 * M = 4950 :=
by
  sorry

end value_of_M_after_subtracting_10_percent_l800_80072


namespace simplify_fraction_l800_80090

theorem simplify_fraction : 
  (1:ℚ) / 462 + 17 / 42 = 94 / 231 := 
by
  sorry

end simplify_fraction_l800_80090


namespace salt_percentage_l800_80035

theorem salt_percentage (salt water : ℝ) (h_salt : salt = 10) (h_water : water = 40) : 
  salt / water = 0.2 :=
by
  sorry

end salt_percentage_l800_80035


namespace distance_greater_than_school_l800_80084

-- Let d1, d2, and d3 be the distances given as the conditions
def distance_orchard_to_house : ℕ := 800
def distance_house_to_pharmacy : ℕ := 1300
def distance_pharmacy_to_school : ℕ := 1700

-- The total distance from orchard to pharmacy via the house
def total_distance_orchard_to_pharmacy : ℕ :=
  distance_orchard_to_house + distance_house_to_pharmacy

-- The difference between the total distance from orchard to pharmacy and the distance from pharmacy to school
def distance_difference : ℕ :=
  total_distance_orchard_to_pharmacy - distance_pharmacy_to_school

-- The theorem to prove
theorem distance_greater_than_school :
  distance_difference = 400 := sorry

end distance_greater_than_school_l800_80084


namespace non_zero_real_x_solution_l800_80093

theorem non_zero_real_x_solution (x : ℝ) (hx : x ≠ 0) : (9 * x) ^ 18 = (18 * x) ^ 9 → x = 2 / 9 := by
  sorry

end non_zero_real_x_solution_l800_80093


namespace exist_rel_prime_k_l_divisible_l800_80055

theorem exist_rel_prime_k_l_divisible (a b p : ℤ) : 
  ∃ (k l : ℤ), Int.gcd k l = 1 ∧ p ∣ (a * k + b * l) := 
sorry

end exist_rel_prime_k_l_divisible_l800_80055


namespace simple_interest_rate_l800_80021

theorem simple_interest_rate (P : ℝ) (R : ℝ) (T : ℝ) 
  (hT : T = 10) (hSI : (P * R * T) / 100 = (1 / 5) * P) : R = 2 :=
by
  sorry

end simple_interest_rate_l800_80021


namespace discount_percentage_in_february_l800_80018

theorem discount_percentage_in_february (C : ℝ) (h1 : C > 0) 
(markup1 : ℝ) (markup2 : ℝ) (profit : ℝ) (D : ℝ) :
  markup1 = 0.20 → markup2 = 0.25 → profit = 0.125 →
  1.50 * C * (1 - D) = 1.125 * C → D = 0.25 :=
by
  intros
  sorry

end discount_percentage_in_february_l800_80018


namespace percentage_greater_than_88_l800_80056

theorem percentage_greater_than_88 (x : ℝ) (percentage : ℝ) (h : x = 88 + percentage * 88) (hx : x = 132) : 
  percentage = 0.5 :=
by
  sorry

end percentage_greater_than_88_l800_80056


namespace not_right_triangle_A_l800_80059

def is_right_triangle (a b c : Real) : Prop :=
  a^2 + b^2 = c^2

theorem not_right_triangle_A : ¬ (is_right_triangle 1.5 2 3) :=
by sorry

end not_right_triangle_A_l800_80059


namespace value_of_expression_l800_80043

theorem value_of_expression (x y : ℕ) (h₁ : x = 3) (h₂ : y = 4) : x * y - x = 9 := 
by
  sorry

end value_of_expression_l800_80043


namespace difference_in_overlap_l800_80058

variable (total_students : ℕ) (geometry_students : ℕ) (biology_students : ℕ)

theorem difference_in_overlap
  (h1 : total_students = 232)
  (h2 : geometry_students = 144)
  (h3 : biology_students = 119) :
  let max_overlap := min geometry_students biology_students;
  let min_overlap := geometry_students + biology_students - total_students;
  max_overlap - min_overlap = 88 :=
by 
  sorry

end difference_in_overlap_l800_80058


namespace equivalence_l800_80069

-- Non-computable declaration to avoid the computational complexity.
noncomputable def is_isosceles_right_triangle (x₁ x₂ : Complex) : Prop :=
  x₂ = x₁ * Complex.I ∨ x₁ = x₂ * Complex.I

-- Definition of the polynomial roots condition.
def roots_form_isosceles_right_triangle (a b : Complex) : Prop :=
  ∃ x₁ x₂ : Complex,
    x₁ + x₂ = -a ∧
    x₁ * x₂ = b ∧
    is_isosceles_right_triangle x₁ x₂

-- Main theorem statement that matches the mathematical equivalency.
theorem equivalence (a b : Complex) : a^2 = 2*b ∧ b ≠ 0 ↔ roots_form_isosceles_right_triangle a b :=
sorry

end equivalence_l800_80069


namespace circumcircle_circumference_thm_triangle_perimeter_thm_l800_80008

-- Definition and theorem for the circumference of the circumcircle
def circumcircle_circumference (a b c R : ℝ) (cosC : ℝ) :=
  cosC = 2 / 3 ∧ c = Real.sqrt 5 ∧ 2 * R = c / (Real.sqrt (1 - cosC^2)) 
  ∧ 2 * R * Real.pi = 3 * Real.pi

theorem circumcircle_circumference_thm (a b c R : ℝ) (cosC : ℝ) :
  circumcircle_circumference a b c R cosC → 2 * R * Real.pi = 3 * Real.pi :=
by
  intro h;
  sorry

-- Definition and theorem for the perimeter of the triangle
def triangle_perimeter (a b c : ℝ) (cosC : ℝ) :=
  cosC = 2 / 3 ∧ c = Real.sqrt 5 ∧ 2 * a = 3 * b ∧ (a + b + c) = 5 + Real.sqrt 5

theorem triangle_perimeter_thm (a b c : ℝ) (cosC : ℝ) :
  triangle_perimeter a b c cosC → (a + b + c) = 5 + Real.sqrt 5 :=
by
  intro h;
  sorry

end circumcircle_circumference_thm_triangle_perimeter_thm_l800_80008


namespace chromium_percentage_in_second_alloy_l800_80042

theorem chromium_percentage_in_second_alloy
  (x : ℝ)
  (h1 : chromium_percentage_in_first_alloy = 15)
  (h2 : weight_first_alloy = 15)
  (h3 : weight_second_alloy = 35)
  (h4 : chromium_percentage_in_new_alloy = 10.1)
  (h5 : total_weight = weight_first_alloy + weight_second_alloy)
  (h6 : chromium_in_new_alloy = chromium_percentage_in_new_alloy / 100 * total_weight)
  (h7 : chromium_in_first_alloy = chromium_percentage_in_first_alloy / 100 * weight_first_alloy)
  (h8 : chromium_in_second_alloy = x / 100 * weight_second_alloy)
  (h9 : chromium_in_new_alloy = chromium_in_first_alloy + chromium_in_second_alloy) :
  x = 8 := by
  sorry

end chromium_percentage_in_second_alloy_l800_80042


namespace distance_between_intersections_l800_80061

def ellipse_eq (x y : ℝ) : Prop := (x^2) / 9 + (y^2) / 25 = 1

def is_focus_of_ellipse (fx fy : ℝ) : Prop := (fx = 0 ∧ (fy = 4 ∨ fy = -4))

def parabola_eq (x y : ℝ) : Prop := y = x^2 / 8 + 2

theorem distance_between_intersections :
  let d := 12 * Real.sqrt 2 / 5
  ∃ x1 x2 y1 y2 : ℝ, 
    ellipse_eq x1 y1 ∧ 
    parabola_eq x1 y1 ∧
    ellipse_eq x2 y2 ∧
    parabola_eq x2 y2 ∧ 
    (x2 - x1)^2 + (y2 - y1)^2 = d^2 :=
by
  sorry

end distance_between_intersections_l800_80061


namespace weight_of_mixture_l800_80020

noncomputable def total_weight_of_mixture (zinc_weight: ℝ) (zinc_ratio: ℝ) (total_ratio: ℝ) : ℝ :=
  (zinc_weight / zinc_ratio) * total_ratio

theorem weight_of_mixture (zinc_ratio: ℝ) (copper_ratio: ℝ) (tin_ratio: ℝ) (zinc_weight: ℝ) :
  total_weight_of_mixture zinc_weight zinc_ratio (zinc_ratio + copper_ratio + tin_ratio) = 98.95 :=
by 
  let ratio_sum := zinc_ratio + copper_ratio + tin_ratio
  let part_weight := zinc_weight / zinc_ratio
  let mixture_weight := part_weight * ratio_sum
  have h : mixture_weight = 98.95 := sorry
  exact h

end weight_of_mixture_l800_80020


namespace min_value_reciprocal_sum_l800_80087

theorem min_value_reciprocal_sum (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) (h_sum : x + y = 1) : 
  ∃ z, z = 4 ∧ (∀ x y, x > 0 ∧ y > 0 ∧ x + y = 1 -> z ≤ (1/x + 1/y)) :=
sorry

end min_value_reciprocal_sum_l800_80087


namespace solve_expression_l800_80046

theorem solve_expression : 6 / 3 - 2 - 8 + 2 * 8 = 8 := 
by 
  sorry

end solve_expression_l800_80046


namespace solution_pair_exists_l800_80012

theorem solution_pair_exists :
  ∃ (p q : ℚ), 
    ∀ (x : ℚ), 
      (p * x^4 + q * x^3 + 45 * x^2 - 25 * x + 10 = 
      (5 * x^2 - 3 * x + 2) * 
      ( (5 / 2) * x^2 - 5 * x + 5)) ∧ 
      (p = (25 / 2)) ∧ 
      (q = (-65 / 2)) :=
by
  sorry

end solution_pair_exists_l800_80012


namespace distance_at_40_kmph_l800_80029

theorem distance_at_40_kmph (x y : ℕ) 
  (h1 : x + y = 250) 
  (h2 : x / 40 + y / 60 = 6) : 
  x = 220 :=
by
  sorry

end distance_at_40_kmph_l800_80029


namespace correct_reference_l800_80075

variable (house : String) 
variable (beautiful_garden_in_front : Bool)
variable (I_like_this_house : Bool)
variable (enough_money_to_buy : Bool)

-- Statement: Given the conditions, prove that the correct word to fill in the blank is "it".
theorem correct_reference : I_like_this_house ∧ beautiful_garden_in_front ∧ ¬ enough_money_to_buy → "it" = "correct choice" :=
by
  sorry

end correct_reference_l800_80075


namespace quadratic_roots_ratio_l800_80028

theorem quadratic_roots_ratio (k : ℝ) (k1 k2 : ℝ) (a b : ℝ) 
  (h_roots : ∀ x : ℝ, k * x * x + (1 - 6 * k) * x + 8 = 0 ↔ (x = a ∨ x = b))
  (h_ab : a ≠ b)
  (h_cond : a / b + b / a = 3 / 7)
  (h_ks : k^1 - 6 * (k1 + k2) + 8 = 0)
  (h_vieta : k1 + k2 = 200 / 36 ∧ k1 * k2 = 49 / 36) : 
  (k1 / k2 + k2 / k1 = 6.25) :=
by sorry

end quadratic_roots_ratio_l800_80028


namespace largest_base_5_three_digit_in_base_10_l800_80063

theorem largest_base_5_three_digit_in_base_10 :
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  n = 124 :=
by
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  show n = 124
  sorry

end largest_base_5_three_digit_in_base_10_l800_80063


namespace TeamC_fee_l800_80086

structure Team :=
(work_rate : ℚ)

def teamA : Team := ⟨1 / 36⟩
def teamB : Team := ⟨1 / 24⟩
def teamC : Team := ⟨1 / 18⟩

def total_fee : ℚ := 36000

def combined_work_rate_first_half (A B C : Team) : ℚ :=
(A.work_rate + B.work_rate + C.work_rate) * 1 / 2

def combined_work_rate_second_half (A C : Team) : ℚ :=
(A.work_rate + C.work_rate) * 1 / 2

def total_work_completed_by_TeamC (A B C : Team) : ℚ :=
C.work_rate * combined_work_rate_first_half A B C + C.work_rate * combined_work_rate_second_half A C

theorem TeamC_fee (A B C : Team) (total_fee : ℚ) :
  total_work_completed_by_TeamC A B C * total_fee = 20000 :=
by
  sorry

end TeamC_fee_l800_80086


namespace pq_solution_l800_80011

theorem pq_solution :
  ∃ (p q : ℤ), (20 * x ^ 2 - 110 * x - 120 = (5 * x + p) * (4 * x + q))
    ∧ (5 * q + 4 * p = -110) ∧ (p * q = -120)
    ∧ (p + 2 * q = -8) :=
by
  sorry

end pq_solution_l800_80011


namespace weight_of_b_l800_80074

theorem weight_of_b (A B C : ℝ)
  (h1 : A + B + C = 129)
  (h2 : A + B = 96)
  (h3 : B + C = 84) : B = 51 := 
by
  sorry

end weight_of_b_l800_80074


namespace find_c_l800_80036

theorem find_c (c : ℝ) (h : (-(c / 3) + -(c / 5) = 30)) : c = -56.25 :=
sorry

end find_c_l800_80036


namespace locus_area_l800_80070

theorem locus_area (R : ℝ) (r : ℝ) (hR : R = 6 * Real.sqrt 7) (hr : r = Real.sqrt 7) :
    ∃ (L : ℝ), (L = 2 * Real.sqrt 42 ∧ L^2 * Real.pi = 168 * Real.pi) :=
by
  sorry

end locus_area_l800_80070


namespace no_uniformly_colored_rectangle_l800_80083

open Int

def point := (ℤ × ℤ)

def is_green (P : point) : Prop :=
  3 ∣ (P.1 + P.2)

def is_red (P : point) : Prop :=
  ¬ is_green P

def is_rectangle (A B C D : point) : Prop :=
  A.1 = D.1 ∧ B.1 = C.1 ∧ A.2 = B.2 ∧ D.2 = C.2

def rectangle_area (A B : point) : ℤ :=
  abs (B.1 - A.1) * abs (B.2 - A.2)

theorem no_uniformly_colored_rectangle :
  ∀ (A B C D : point) (k : ℕ), 
  is_rectangle A B C D →
  rectangle_area A C = 2^k →
  ¬ (is_green A ∧ is_green B ∧ is_green C ∧ is_green D) ∧
  ¬ (is_red A ∧ is_red B ∧ is_red C ∧ is_red D) :=
by sorry

end no_uniformly_colored_rectangle_l800_80083


namespace infinite_series_sum_eq_two_l800_80073

theorem infinite_series_sum_eq_two : 
  ∑' k : ℕ, (if k = 0 then 0 else (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1))))) = 2 :=
by
  sorry

end infinite_series_sum_eq_two_l800_80073


namespace solve_system_of_equations_l800_80050

theorem solve_system_of_equations
  (a b c : ℝ) (x y z : ℝ)
  (h1 : x + y = a)
  (h2 : y + z = b)
  (h3 : z + x = c) :
  x = (a + c - b) / 2 ∧ y = (a + b - c) / 2 ∧ z = (b + c - a) / 2 :=
by
  sorry

end solve_system_of_equations_l800_80050


namespace polynomial_divisibility_l800_80041

theorem polynomial_divisibility (n : ℕ) : (∀ x : ℤ, (x^2 + x + 1 ∣ x^(2*n) + x^n + 1)) ↔ (3 ∣ n) := by
  sorry

end polynomial_divisibility_l800_80041


namespace shirt_price_l800_80071

theorem shirt_price (T S : ℝ) (h1 : T + S = 80.34) (h2 : T = S - 7.43) : T = 36.455 :=
by 
sorry

end shirt_price_l800_80071


namespace fraction_of_water_l800_80004

theorem fraction_of_water (total_weight sand_ratio water_weight gravel_weight : ℝ)
  (htotal : total_weight = 49.99999999999999)
  (hsand_ratio : sand_ratio = 1/2)
  (hwater : water_weight = total_weight - total_weight * sand_ratio - gravel_weight)
  (hgravel : gravel_weight = 15)
  : (water_weight / total_weight) = 1/5 :=
by
  sorry

end fraction_of_water_l800_80004


namespace zachary_pushups_l800_80017

theorem zachary_pushups (d z : ℕ) (h1 : d = z + 30) (h2 : d = 37) : z = 7 := by
  sorry

end zachary_pushups_l800_80017


namespace not_mysterious_diff_consecutive_odd_l800_80095

/-- A mysterious number is defined as the difference of squares of two consecutive even numbers. --/
def is_mysterious (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (2 * k + 2)^2 - (2 * k)^2

/-- The difference of the squares of two consecutive odd numbers. --/
def diff_squares_consecutive_odd (k : ℤ) : ℤ :=
  (2 * k + 1)^2 - (2 * k - 1)^2

/-- Prove that the difference of squares of two consecutive odd numbers is not a mysterious number. --/
theorem not_mysterious_diff_consecutive_odd (k : ℤ) : ¬ is_mysterious (Int.natAbs (diff_squares_consecutive_odd k)) :=
by
  sorry

end not_mysterious_diff_consecutive_odd_l800_80095


namespace value_of_x0_l800_80092

noncomputable def f (x : ℝ) : ℝ := x^3

theorem value_of_x0 (x0 : ℝ) (h1 : f x0 = x0^3) (h2 : deriv f x0 = 3) :
  x0 = 1 ∨ x0 = -1 :=
by
  sorry

end value_of_x0_l800_80092


namespace four_digit_number_with_divisors_l800_80067

def is_four_digit (n : Nat) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_minimal_divisor (n p : Nat) : Prop :=
  p > 1 ∧ n % p = 0
  
def is_maximal_divisor (n q : Nat) : Prop :=
  q < n ∧ n % q = 0
  
theorem four_digit_number_with_divisors :
  ∃ (n p : Nat), is_four_digit n ∧ is_minimal_divisor n p ∧ n = 49 * p * p :=
by
  sorry

end four_digit_number_with_divisors_l800_80067


namespace land_percentage_relationship_l800_80027

variable {V : ℝ} -- Total taxable value of all land in the village
variable {x y z : ℝ} -- Percentages of Mr. William's land in types A, B, C

-- Conditions
axiom total_tax_collected : 0.80 * (x / 100 * V) + 0.90 * (y / 100 * V) + 0.95 * (z / 100 * V) = 3840
axiom mr_william_tax : 0.80 * (x / 100 * V) + 0.90 * (y / 100 * V) + 0.95 * (z / 100 * V) = 480

-- Prove the relationship
theorem land_percentage_relationship : (0.80 * x + 0.90 * y + 0.95 * z = 48000 / V) → (x + y + z = 100) := by
  sorry

end land_percentage_relationship_l800_80027


namespace remainder_of_expression_l800_80007

theorem remainder_of_expression (n : ℤ) (h : n % 100 = 99) : (n^2 + 2*n + 3 + n^3) % 100 = 1 :=
by
  sorry

end remainder_of_expression_l800_80007


namespace intersection_a_zero_range_of_a_l800_80016

variable (x a : ℝ)

def setA : Set ℝ := { x | - 1 < x ∧ x < 6 }
def setB (a : ℝ) : Set ℝ := { x | 2 * a - 1 ≤ x ∧ x < a + 5 }

theorem intersection_a_zero :
  setA x ∧ setB 0 x ↔ - 1 < x ∧ x < 5 := by
  sorry

theorem range_of_a (h : ∀ x, setA x ∨ setB a x → setA x) :
  (0 < a ∧ a ≤ 1) ∨ 6 ≤ a :=
  sorry

end intersection_a_zero_range_of_a_l800_80016


namespace max_apartment_size_is_600_l800_80051

-- Define the cost per square foot and Max's budget
def cost_per_square_foot : ℝ := 1.2
def max_budget : ℝ := 720

-- Define the largest apartment size that Max should consider
def largest_apartment_size (s : ℝ) : Prop :=
  cost_per_square_foot * s = max_budget

-- State the theorem that we need to prove
theorem max_apartment_size_is_600 : largest_apartment_size 600 :=
  sorry

end max_apartment_size_is_600_l800_80051


namespace agreed_period_of_service_l800_80001

theorem agreed_period_of_service (x : ℕ) (rs800 : ℕ) (rs400 : ℕ) (servant_period : ℕ) (received_amount : ℕ) (uniform : ℕ) (half_period : ℕ) :
  rs800 = 800 ∧ rs400 = 400 ∧ servant_period = 9 ∧ received_amount = 400 ∧ half_period = x / 2 ∧ servant_period = half_period → x = 18 :=
by sorry

end agreed_period_of_service_l800_80001


namespace evaluate_at_minus_two_l800_80009

def f (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

theorem evaluate_at_minus_two : f (-2) = -1 := 
by 
  unfold f 
  sorry

end evaluate_at_minus_two_l800_80009


namespace arithmetic_sequence_general_formula_l800_80024

variable (a : ℤ) 

def is_arithmetic_sequence (a1 a2 a3 : ℤ) : Prop :=
  2 * a2 = a1 + a3

theorem arithmetic_sequence_general_formula :
  ∀ {a1 a2 a3 : ℤ}, is_arithmetic_sequence a1 a2 a3 → a1 = a - 1 ∧ a2 = a + 1 ∧ a3 = 2 * a + 3 → 
  ∀ n : ℕ, a_n = 2 * n - 3
:= by
  sorry

end arithmetic_sequence_general_formula_l800_80024


namespace probability_at_least_75_cents_l800_80053

def total_coins : ℕ := 3 + 5 + 4 + 3 -- total number of coins

def pennies : ℕ := 3
def nickels : ℕ := 5
def dimes : ℕ := 4
def quarters : ℕ := 3

def successful_outcomes_case1 : ℕ := (Nat.choose 3 3) * (Nat.choose 12 3)
def successful_outcomes_case2 : ℕ := (Nat.choose 3 2) * (Nat.choose 4 2) * (Nat.choose 5 2)

def total_outcomes : ℕ := Nat.choose 15 6
def successful_outcomes : ℕ := successful_outcomes_case1 + successful_outcomes_case2

def probability : ℚ := successful_outcomes / total_outcomes

theorem probability_at_least_75_cents :
  probability = 400 / 5005 := by
  sorry

end probability_at_least_75_cents_l800_80053


namespace phone_charges_equal_l800_80010

theorem phone_charges_equal (x : ℝ) : 
  (0.60 + 14 * x = 0.08 * 18) → (x = 0.06) :=
by
  intro h
  have : 14 * x = 1.44 - 0.60 := sorry
  have : 14 * x = 0.84 := sorry
  have : x = 0.06 := sorry
  exact this

end phone_charges_equal_l800_80010


namespace necessary_but_not_sufficient_l800_80080

theorem necessary_but_not_sufficient (a b : ℝ) : 
 (a > b) ↔ (a-1 > b+1) :=
by {
  sorry
}

end necessary_but_not_sufficient_l800_80080


namespace perfect_square_divisible_by_12_l800_80019

theorem perfect_square_divisible_by_12 (k : ℤ) : 12 ∣ (k^2 * (k^2 - 1)) :=
by sorry

end perfect_square_divisible_by_12_l800_80019


namespace simplify_expression_l800_80044

theorem simplify_expression (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) :
  let x := q/r + r/q
  let y := p/r + r/p
  let z := p/q + q/p
  (x^2 + y^2 + z^2 - 2 * x * y * z) = 4 :=
by
  let x := q/r + r/q
  let y := p/r + r/p
  let z := p/q + q/p
  sorry

end simplify_expression_l800_80044


namespace race_distance_between_Sasha_and_Kolya_l800_80030

theorem race_distance_between_Sasha_and_Kolya
  (vS vL vK : ℝ)
  (h1 : vK = 0.9 * vL)
  (h2 : ∀ t_S, 100 = vS * t_S → vL * t_S = 90)
  (h3 : ∀ t_L, 100 = vL * t_L → vK * t_L = 90)
  : ∀ t_S, 100 = vS * t_S → (100 - vK * t_S) = 19 :=
by
  sorry


end race_distance_between_Sasha_and_Kolya_l800_80030


namespace divisible_by_1995_l800_80026

theorem divisible_by_1995 (n : ℕ) : 
  1995 ∣ (256^(2*n) * 7^(2*n) - 168^(2*n) - 32^(2*n) + 3^(2*n)) := 
sorry

end divisible_by_1995_l800_80026


namespace min_value_f_when_a1_l800_80076

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + |x - a|

theorem min_value_f_when_a1 : ∀ x : ℝ, f x 1 ≥ 3/4 :=
by sorry

end min_value_f_when_a1_l800_80076


namespace no_integer_solutions_l800_80049

theorem no_integer_solutions (x y : ℤ) : 2 * x^2 - 5 * y^2 ≠ 7 :=
  sorry

end no_integer_solutions_l800_80049


namespace original_square_area_l800_80057

-- Definitions based on the given problem conditions
variable (s : ℝ) (A : ℝ)
def is_square (s : ℝ) : Prop := s > 0
def oblique_projection (s : ℝ) (A : ℝ) : Prop :=
  (A = s^2 ∨ A = 4^2) ∧ s = 4

-- The theorem statement based on the problem question and correct answer
theorem original_square_area :
  is_square s →
  oblique_projection s A →
  ∃ A, A = 16 ∨ A = 64 := 
sorry

end original_square_area_l800_80057


namespace increased_amount_is_30_l800_80047

noncomputable def F : ℝ := (3 / 2) * 179.99999999999991
noncomputable def F' : ℝ := (5 / 3) * 179.99999999999991
noncomputable def J : ℝ := 179.99999999999991
noncomputable def increased_amount : ℝ := F' - F

theorem increased_amount_is_30 : increased_amount = 30 :=
by
  -- Placeholder for proof. Actual proof goes here.
  sorry

end increased_amount_is_30_l800_80047


namespace maximum_value_of_k_l800_80013

-- Define the variables and conditions
variables {a b c k : ℝ}
axiom h₀ : a > b
axiom h₁ : b > c
axiom h₂ : 4 / (a - b) + 1 / (b - c) + k / (c - a) ≥ 0

-- State the theorem
theorem maximum_value_of_k : k ≤ 9 := sorry

end maximum_value_of_k_l800_80013


namespace rebecca_eggs_l800_80034

theorem rebecca_eggs (groups eggs_per_group : ℕ) (h1 : groups = 3) (h2 : eggs_per_group = 6) : 
  (groups * eggs_per_group = 18) :=
by
  sorry

end rebecca_eggs_l800_80034


namespace find_x_l800_80096

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 80) : x = 26 :=
by 
  sorry

end find_x_l800_80096


namespace probability_all_boxes_non_empty_equals_4_over_9_l800_80000

structure PaintingPlacement :=
  (paintings : Finset ℕ)
  (boxes : Finset ℕ)
  (num_paintings : paintings.card = 4)
  (num_boxes : boxes.card = 3)

noncomputable def probability_non_empty_boxes (pp : PaintingPlacement) : ℚ :=
  let total_outcomes := 3^4
  let favorable_outcomes := Nat.choose 4 2 * Nat.factorial 3
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

theorem probability_all_boxes_non_empty_equals_4_over_9
  (pp : PaintingPlacement) : pp.paintings.card = 4 → pp.boxes.card = 3 →
  probability_non_empty_boxes pp = 4 / 9 :=
by
  intros h1 h2
  sorry

end probability_all_boxes_non_empty_equals_4_over_9_l800_80000


namespace sin_B_sin_C_l800_80038

open Real

noncomputable def triangle_condition (A B C : ℝ) (a b c : ℝ) : Prop :=
  cos (2 * A) - 3 * cos (B + C) = 1 ∧
  (1 / 2) * b * c * sin A = 5 * sqrt 3 ∧
  b = 5

theorem sin_B_sin_C {A B C a b c : ℝ} (h : triangle_condition A B C a b c) :
  (sin B) * (sin C) = 5 / 7 := 
sorry

end sin_B_sin_C_l800_80038


namespace hypotenuse_of_right_triangle_l800_80015

theorem hypotenuse_of_right_triangle (a b : ℕ) (ha : a = 140) (hb : b = 336) :
  Nat.sqrt (a * a + b * b) = 364 := by
  sorry

end hypotenuse_of_right_triangle_l800_80015


namespace solve_for_x_l800_80065

theorem solve_for_x (x : ℝ) (h : 3 * x - 4 * x + 7 * x = 120) : x = 20 :=
by
  sorry

end solve_for_x_l800_80065


namespace impossible_300_numbers_l800_80048

theorem impossible_300_numbers (n : ℕ) (hn : n = 300) (a : ℕ → ℕ) (hp : ∀ i, 0 < a i)
(hdiff : ∃ k, ∀ i ≠ k, a i = a ((i + 1) % n) - a ((i - 1 + n) % n)) 
: false :=
by {
  sorry
}

end impossible_300_numbers_l800_80048


namespace mean_inequalities_l800_80005

noncomputable def arith_mean (a : List ℝ) : ℝ := 
  (a.foldr (· + ·) 0) / a.length

noncomputable def geom_mean (a : List ℝ) : ℝ := 
  Real.exp ((a.foldr (λ x y => Real.log x + y) 0) / a.length)

noncomputable def harm_mean (a : List ℝ) : ℝ := 
  a.length / (a.foldr (λ x y => 1 / x + y) 0)

def is_positive (a : List ℝ) : Prop := 
  ∀ x ∈ a, x > 0

def bounds (a : List ℝ) (m g h : ℝ) : Prop := 
  let α := List.minimum a
  let β := List.maximum a
  α ≤ h ∧ h ≤ g ∧ g ≤ m ∧ m ≤ β

theorem mean_inequalities (a : List ℝ) (h g m : ℝ) (h_assoc: h = harm_mean a) (g_assoc: g = geom_mean a) (m_assoc: m = arith_mean a) :
  is_positive a → bounds a m g h :=
  
sorry

end mean_inequalities_l800_80005


namespace no_five_coins_sum_to_43_l800_80033

def coin_values : Set ℕ := {1, 5, 10, 25}

theorem no_five_coins_sum_to_43 :
  ¬ ∃ (a b c d e : ℕ), a ∈ coin_values ∧ b ∈ coin_values ∧ c ∈ coin_values ∧ d ∈ coin_values ∧ e ∈ coin_values ∧ (a + b + c + d + e = 43) :=
sorry

end no_five_coins_sum_to_43_l800_80033


namespace zero_pow_2014_l800_80064

-- Define the condition that zero raised to any positive power is zero
def zero_pow_pos {n : ℕ} (h : 0 < n) : (0 : ℝ)^n = 0 := by
  sorry

-- Use this definition to prove the specific case of 0 ^ 2014 = 0
theorem zero_pow_2014 : (0 : ℝ)^(2014) = 0 := by
  have h : 0 < 2014 := by decide
  exact zero_pow_pos h

end zero_pow_2014_l800_80064


namespace circle_through_point_and_same_center_l800_80039

theorem circle_through_point_and_same_center :
  ∃ (x_0 y_0 r : ℝ),
    (∀ (x y : ℝ), (x - x_0)^2 + (y - y_0)^2 = r^2 ↔
      x^2 + y^2 - 4 * x + 6 * y - 3 = 0)
    ∧
    ∀ (x y : ℝ), (x - x_0)^2 + (y - y_0)^2 = r^2 ↔
      (x - 2)^2 + (y + 3)^2 = 25 := sorry

end circle_through_point_and_same_center_l800_80039


namespace find_other_number_l800_80094

/--
Given two numbers A and B, where:
    * The reciprocal of the HCF of A and B is \( \frac{1}{13} \).
    * The reciprocal of the LCM of A and B is \( \frac{1}{312} \).
    * A = 24
Prove that B = 169.
-/
theorem find_other_number 
  (A B : ℕ) 
  (h1 : A = 24)
  (h2 : (Nat.gcd A B) = 13)
  (h3 : (Nat.lcm A B) = 312) : 
  B = 169 := 
by 
  sorry

end find_other_number_l800_80094


namespace find_number_l800_80078

theorem find_number (x : ℝ) (h : 0.26 * x = 93.6) : x = 360 := sorry

end find_number_l800_80078


namespace operations_on_S_l800_80052

def is_element_of_S (x : ℤ) : Prop :=
  x = 0 ∨ ∃ n : ℤ, x = 2 * n

theorem operations_on_S (a b : ℤ) (ha : is_element_of_S a) (hb : is_element_of_S b) :
  (is_element_of_S (a + b)) ∧
  (is_element_of_S (a - b)) ∧
  (is_element_of_S (a * b)) ∧
  (¬ is_element_of_S (a / b)) ∧
  (¬ is_element_of_S ((a + b) / 2)) :=
by
  sorry

end operations_on_S_l800_80052


namespace real_number_a_pure_imaginary_l800_80054

-- Definition of an imaginary number
def pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

-- Given conditions and the proof problem statement
theorem real_number_a_pure_imaginary (a : ℝ) :
  pure_imaginary (⟨(a + 1) / 2, (1 - a) / 2⟩) → a = -1 :=
by
  sorry

end real_number_a_pure_imaginary_l800_80054


namespace machine_transport_equation_l800_80068

theorem machine_transport_equation (x : ℝ) :
  (∀ (rateA rateB : ℝ), rateB = rateA + 60 → (500 / rateA = 800 / rateB) → rateA = x → rateB = x + 60) :=
by
  sorry

end machine_transport_equation_l800_80068


namespace combined_fractions_value_l800_80022

theorem combined_fractions_value (N : ℝ) (h1 : 0.40 * N = 168) : 
  (1/4) * (1/3) * (2/5) * N = 14 :=
by
  sorry

end combined_fractions_value_l800_80022


namespace find_y_l800_80097

theorem find_y (y : ℝ) (h : (y^2 - 11 * y + 24) / (y - 1) + (4 * y^2 + 20 * y - 25) / (4*y - 5) = 5) :
  y = 3 ∨ y = 4 :=
sorry

end find_y_l800_80097


namespace rad_times_trivia_eq_10000_l800_80002

theorem rad_times_trivia_eq_10000 
  (h a r v d m i t : ℝ)
  (H1 : h * a * r * v * a * r * d = 100)
  (H2 : m * i * t = 100)
  (H3 : h * m * m * t = 100) :
  (r * a * d) * (t * r * i * v * i * a) = 10000 := 
  sorry

end rad_times_trivia_eq_10000_l800_80002


namespace determine_range_of_m_l800_80091

noncomputable def range_m (m : ℝ) (x : ℝ) : Prop :=
  ∀ x, (-x^2 + 7 * x + 8 < 0 → (x < 1 - 2 * |m| ∨ x > 1 + 2 * |m|)) ∧
       (∃ x, -x^2 + 7 * x + 8 ≥ 0)

theorem determine_range_of_m (m : ℝ) :
  (-1 ≤ m ∧ m ≤ 1) ↔
  (∀ x, (-x^2 + 7 * x + 8 < 0 → (x < 1 - 2 * |m| ∨ x > 1 + 2 * |m|)) ∧
         (∃ x, -x^2 + 7 * x + 8 ≥ 0)) :=
by
  sorry

end determine_range_of_m_l800_80091


namespace length_of_rope_l800_80023

-- Define the given conditions
variable (L : ℝ)
variable (h1 : 0.6 * L = 0.69)

-- The theorem to prove
theorem length_of_rope (L : ℝ) (h1 : 0.6 * L = 0.69) : L = 1.15 :=
by
  sorry

end length_of_rope_l800_80023


namespace find_alpha_l800_80085

noncomputable def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * (a 2 / a 1)

-- Given that {a_n} is a geometric sequence,
-- a_1 and a_8 are roots of the equation
-- x^2 - 2x * sin(alpha) - √3 * sin(alpha) = 0,
-- and (a_1 + a_8)^2 = 2 * a_3 * a_6 + 6,
-- prove that alpha = π / 3.
theorem find_alpha :
  ∃ α : ℝ,
  (∀ (a : ℕ → ℝ), isGeometricSequence a ∧ 
  (∃ (a1 a8 : ℝ), 
    (a1 + a8)^2 = 2 * a 3 * a 6 + 6 ∧
    a1 + a8 = 2 * Real.sin α ∧
    a1 * a8 = - Real.sqrt 3 * Real.sin α)) →
  α = Real.pi / 3 :=
by 
  sorry

end find_alpha_l800_80085


namespace value_of_a_l800_80081

theorem value_of_a
  (a b : ℚ)
  (h1 : b / a = 4)
  (h2 : b = 18 - 6 * a) :
  a = 9 / 5 := by
  sorry

end value_of_a_l800_80081


namespace regular_square_pyramid_side_edge_length_l800_80037

theorem regular_square_pyramid_side_edge_length 
  (base_edge_length : ℝ)
  (volume : ℝ)
  (h_base_edge_length : base_edge_length = 4 * Real.sqrt 2)
  (h_volume : volume = 32) :
  ∃ side_edge_length : ℝ, side_edge_length = 5 :=
by sorry

end regular_square_pyramid_side_edge_length_l800_80037


namespace sum_of_center_coordinates_l800_80089

def center_of_circle_sum (x y : ℝ) : Prop :=
  (x - 6)^2 + (y + 5)^2 = 101

theorem sum_of_center_coordinates : center_of_circle_sum x y → x + y = 1 :=
sorry

end sum_of_center_coordinates_l800_80089


namespace child_ticket_cost_l800_80077

theorem child_ticket_cost 
    (total_people : ℕ) 
    (total_money_collected : ℤ) 
    (adult_ticket_price : ℤ) 
    (children_attended : ℕ) 
    (adults_count : ℕ) 
    (total_adult_cost : ℤ) 
    (total_child_cost : ℤ) 
    (c : ℤ)
    (total_people_eq : total_people = 22)
    (total_money_collected_eq : total_money_collected = 50)
    (adult_ticket_price_eq : adult_ticket_price = 8)
    (children_attended_eq : children_attended = 18)
    (adults_count_eq : adults_count = total_people - children_attended)
    (total_adult_cost_eq : total_adult_cost = adults_count * adult_ticket_price)
    (total_child_cost_eq : total_child_cost = children_attended * c)
    (money_collected_eq : total_money_collected = total_adult_cost + total_child_cost) 
  : c = 1 := 
  by
    sorry

end child_ticket_cost_l800_80077


namespace undefined_values_l800_80082

theorem undefined_values (b : ℝ) : (b^2 - 9 = 0) ↔ (b = -3 ∨ b = 3) := by
  sorry

end undefined_values_l800_80082


namespace rachel_age_when_emily_half_her_age_l800_80099

theorem rachel_age_when_emily_half_her_age (emily_current_age rachel_current_age : ℕ) 
  (h1 : emily_current_age = 20) 
  (h2 : rachel_current_age = 24) 
  (age_difference : ℕ) 
  (h3 : rachel_current_age - emily_current_age = age_difference) 
  (emily_age_when_half : ℕ) 
  (rachel_age_when_half : ℕ) 
  (h4 : emily_age_when_half = rachel_age_when_half / 2)
  (h5 : rachel_age_when_half = emily_age_when_half + age_difference) :
  rachel_age_when_half = 8 :=
by
  sorry

end rachel_age_when_emily_half_her_age_l800_80099


namespace share_equally_l800_80062

variable (Emani Howard : ℕ)
axiom h1 : Emani = 150
axiom h2 : Emani = Howard + 30

theorem share_equally : (Emani + Howard) / 2 = 135 :=
by sorry

end share_equally_l800_80062


namespace milk_left_l800_80060

theorem milk_left (initial_milk : ℝ) (given_away : ℝ) (h_initial : initial_milk = 5) (h_given : given_away = 18 / 4) :
  ∃ remaining_milk : ℝ, remaining_milk = initial_milk - given_away ∧ remaining_milk = 1 / 2 :=
by
  use 1 / 2
  sorry

end milk_left_l800_80060


namespace sheila_hourly_rate_is_6_l800_80003

variable (weekly_earnings : ℕ) (hours_mwf : ℕ) (days_mwf : ℕ) (hours_tt: ℕ) (days_tt : ℕ)
variable [NeZero hours_mwf] [NeZero days_mwf] [NeZero hours_tt] [NeZero days_tt]

-- Define Sheila's working hours and weekly earnings as given conditions
def weekly_hours := (hours_mwf * days_mwf) + (hours_tt * days_tt)
def hourly_rate := weekly_earnings / weekly_hours

-- Specific values from the given problem
def sheila_weekly_earnings : ℕ := 216
def sheila_hours_mwf : ℕ := 8
def sheila_days_mwf : ℕ := 3
def sheila_hours_tt : ℕ := 6
def sheila_days_tt : ℕ := 2

-- The theorem to prove
theorem sheila_hourly_rate_is_6 :
  (sheila_weekly_earnings / ((sheila_hours_mwf * sheila_days_mwf) + (sheila_hours_tt * sheila_days_tt))) = 6 := by
  sorry

end sheila_hourly_rate_is_6_l800_80003


namespace solution_of_xyz_l800_80014

theorem solution_of_xyz (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * y + z = 47)
  (h2 : y * z + x = 47)
  (h3 : z * x + y = 47) : x + y + z = 48 := 
sorry

end solution_of_xyz_l800_80014


namespace special_day_jacket_price_l800_80006

noncomputable def original_price : ℝ := 240
noncomputable def first_discount_rate : ℝ := 0.4
noncomputable def special_day_discount_rate : ℝ := 0.25

noncomputable def first_discounted_price : ℝ :=
  original_price * (1 - first_discount_rate)
  
noncomputable def special_day_price : ℝ :=
  first_discounted_price * (1 - special_day_discount_rate)

theorem special_day_jacket_price : special_day_price = 108 := by
  -- definitions and calculations go here
  sorry

end special_day_jacket_price_l800_80006


namespace seq_is_arithmetic_l800_80066

-- Define the sequence sum S_n and the sequence a_n
noncomputable def S (a : ℕ) (n : ℕ) : ℕ := a * n^2 + n
noncomputable def a_n (a : ℕ) (n : ℕ) : ℕ := S a n - S a (n - 1)

-- Define the property of being an arithmetic sequence
def is_arithmetic_seq (a_n : ℕ → ℕ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, n ≥ 1 → (a_n (n + 1) : ℤ) - (a_n n : ℤ) = d

-- The theorem to be proven
theorem seq_is_arithmetic (a : ℕ) (h : 0 < a) : is_arithmetic_seq (a_n a) :=
by
  sorry

end seq_is_arithmetic_l800_80066


namespace solve_for_t_l800_80040

theorem solve_for_t (t : ℚ) :
  (t+2) * (4*t-4) = (4*t-6) * (t+3) + 3 → t = 7/2 :=
by {
  sorry
}

end solve_for_t_l800_80040
