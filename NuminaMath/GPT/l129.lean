import Mathlib

namespace NUMINAMATH_GPT_third_number_in_list_l129_12921

theorem third_number_in_list :
  let nums : List ℕ := [201, 202, 205, 206, 209, 209, 210, 212, 212]
  nums.nthLe 2 (by simp [List.length]) = 205 :=
sorry

end NUMINAMATH_GPT_third_number_in_list_l129_12921


namespace NUMINAMATH_GPT_find_integer_pairs_l129_12932

theorem find_integer_pairs : 
  ∀ (x y : Int), x^3 = y^3 + 2 * y^2 + 1 ↔ (x, y) = (1, 0) ∨ (x, y) = (1, -2) ∨ (x, y) = (-2, -3) :=
by
  intros x y
  sorry

end NUMINAMATH_GPT_find_integer_pairs_l129_12932


namespace NUMINAMATH_GPT_total_paint_is_correct_l129_12993

/-- Given conditions -/
def paint_per_large_canvas := 3
def paint_per_small_canvas := 2
def large_paintings := 3
def small_paintings := 4

/-- Define total paint used using the given conditions -/
noncomputable def total_paint_used : ℕ := 
  (paint_per_large_canvas * large_paintings) + (paint_per_small_canvas * small_paintings)

/-- Theorem statement to show the total paint used equals 17 ounces -/
theorem total_paint_is_correct : total_paint_used = 17 := by
  sorry

end NUMINAMATH_GPT_total_paint_is_correct_l129_12993


namespace NUMINAMATH_GPT_three_digit_number_mul_seven_results_638_l129_12918

theorem three_digit_number_mul_seven_results_638 (N : ℕ) 
  (hN1 : 100 ≤ N) 
  (hN2 : N < 1000)
  (hN3 : ∃ (x : ℕ), 7 * N = 1000 * x + 638) : N = 234 := 
sorry

end NUMINAMATH_GPT_three_digit_number_mul_seven_results_638_l129_12918


namespace NUMINAMATH_GPT_least_number_subtracted_l129_12973

theorem least_number_subtracted (n : ℕ) (h : n = 427398) : ∃ x, x = 8 ∧ (n - x) % 10 = 0 :=
by
  sorry

end NUMINAMATH_GPT_least_number_subtracted_l129_12973


namespace NUMINAMATH_GPT_monthly_production_increase_l129_12909

/-- A salt manufacturing company produced 3000 tonnes in January and increased its
    production by some tonnes every month over the previous month until the end
    of the year. Given that the average daily production was 116.71232876712328 tonnes,
    determine the monthly production increase. -/
theorem monthly_production_increase :
  let initial_production := 3000
  let daily_average_production := 116.71232876712328
  let days_per_year := 365
  let total_yearly_production := daily_average_production * days_per_year
  let months_per_year := 12
  ∃ (x : ℝ), total_yearly_production = (months_per_year / 2) * (2 * initial_production + (months_per_year - 1) * x) → x = 100 :=
sorry

end NUMINAMATH_GPT_monthly_production_increase_l129_12909


namespace NUMINAMATH_GPT_polygon_max_sides_l129_12966

theorem polygon_max_sides (n : ℕ) (h : (n - 2) * 180 < 2005) : n ≤ 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_polygon_max_sides_l129_12966


namespace NUMINAMATH_GPT_sum_of_number_and_radical_conjugate_l129_12997

theorem sum_of_number_and_radical_conjugate : 
  (10 - Real.sqrt 2018) + (10 + Real.sqrt 2018) = 20 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_number_and_radical_conjugate_l129_12997


namespace NUMINAMATH_GPT_k_range_proof_l129_12915

/- Define points in the Cartesian plane as ordered pairs. -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/- Define two points P and Q. -/
def P : Point := { x := -1, y := 1 }
def Q : Point := { x := 2, y := 2 }

/- Define the line equation. -/
def line_equation (k : ℝ) (x : ℝ) : ℝ :=
  k * x - 1

/- Define the range of k. -/
def k_range (k : ℝ) : Prop :=
  1 / 3 < k ∧ k < 3 / 2

/- Theorem statement. -/
theorem k_range_proof (k : ℝ) (intersects_PQ_extension : ∀ k : ℝ, ∀ x : ℝ, ((P.y ≤ line_equation k x ∧ line_equation k x ≤ Q.y) ∧ line_equation k x ≠ Q.y) → k_range k) :
  ∀ k, k_range k :=
by
  sorry

end NUMINAMATH_GPT_k_range_proof_l129_12915


namespace NUMINAMATH_GPT_find_greatest_consecutive_integer_l129_12985

theorem find_greatest_consecutive_integer (n : ℤ) 
  (h : n^2 + (n + 1)^2 = 452) : n + 1 = 15 :=
sorry

end NUMINAMATH_GPT_find_greatest_consecutive_integer_l129_12985


namespace NUMINAMATH_GPT_evaluate_expression_at_3_l129_12968

theorem evaluate_expression_at_3 :
  (∀ x ≠ 2, (x = 3) → (x^2 - 5 * x + 6) / (x - 2) = 0) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_3_l129_12968


namespace NUMINAMATH_GPT_symmetric_point_origin_l129_12920

-- Define the notion of symmetry with respect to the origin
def symmetric_with_origin (p : ℤ × ℤ) : ℤ × ℤ :=
  (-p.1, -p.2)

-- Define the given point
def given_point : ℤ × ℤ :=
  (-2, 5)

-- State the theorem to be proven
theorem symmetric_point_origin : 
  symmetric_with_origin given_point = (2, -5) :=
by 
  -- The proof will go here, use sorry for now
  sorry

end NUMINAMATH_GPT_symmetric_point_origin_l129_12920


namespace NUMINAMATH_GPT_hayley_initial_meatballs_l129_12963

theorem hayley_initial_meatballs (x : ℕ) (stolen : ℕ) (left : ℕ) (h1 : stolen = 14) (h2 : left = 11) (h3 : x - stolen = left) : x = 25 := 
by 
  sorry

end NUMINAMATH_GPT_hayley_initial_meatballs_l129_12963


namespace NUMINAMATH_GPT_ab_plus_cd_is_composite_l129_12911

theorem ab_plus_cd_is_composite 
  (a b c d : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h_order : a > b ∧ b > c ∧ c > d)
  (h_eq : a^2 + a * c - c^2 = b^2 + b * d - d^2) : 
  ∃ p q : ℕ, p > 1 ∧ q > 1 ∧ ab + cd = p * q :=
by
  sorry

end NUMINAMATH_GPT_ab_plus_cd_is_composite_l129_12911


namespace NUMINAMATH_GPT_complement_of_P_in_U_l129_12974

/-- Definitions of sets U and P -/
def U := { y : ℝ | ∃ x : ℝ, x > 1 ∧ y = Real.log x / Real.log 2 }
def P := { y : ℝ | ∃ x : ℝ, x > 2 ∧ y = 1 / x }

/-- The complement of P in U -/
def complement_U_P := { y : ℝ | y = 0 ∨ y ≥ 1 / 2 }

/-- Proving the complement of P in U is as expected -/
theorem complement_of_P_in_U : { y : ℝ | y ∈ U ∧ y ∉ P } = complement_U_P := by
  sorry

end NUMINAMATH_GPT_complement_of_P_in_U_l129_12974


namespace NUMINAMATH_GPT_rice_price_per_kg_l129_12908

theorem rice_price_per_kg (price1 price2 : ℝ) (amount1 amount2 : ℝ) (total_cost total_weight : ℝ) (P : ℝ)
  (h1 : price1 = 6.60)
  (h2 : amount1 = 49)
  (h3 : price2 = 9.60)
  (h4 : amount2 = 56)
  (h5 : total_cost = price1 * amount1 + price2 * amount2)
  (h6 : total_weight = amount1 + amount2)
  (h7 : P = total_cost / total_weight) :
  P = 8.20 := 
by sorry

end NUMINAMATH_GPT_rice_price_per_kg_l129_12908


namespace NUMINAMATH_GPT_polar_center_coordinates_l129_12982

-- Define polar coordinate system equation
def polar_circle (ρ θ : ℝ) := ρ = 2 * Real.sin θ

-- Define the theorem: Given the equation of a circle in polar coordinates, its center in polar coordinates.
theorem polar_center_coordinates :
  (∀ (θ : ℝ), 0 ≤ θ ∧ θ < 2 * Real.pi → ∃ ρ, polar_circle ρ θ) →
  (∀ ρ θ, polar_circle ρ θ → 0 ≤ θ ∧ θ < 2 * Real.pi → (ρ = 1 ∧ θ = Real.pi / 2) ∨ (ρ = -1 ∧ θ = 3 * Real.pi / 2)) :=
by {
  sorry 
}

end NUMINAMATH_GPT_polar_center_coordinates_l129_12982


namespace NUMINAMATH_GPT_race_distance_l129_12919

theorem race_distance (a b c d : ℝ) 
  (h₁ : d / a = (d - 25) / b)
  (h₂ : d / b = (d - 15) / c)
  (h₃ : d / a = (d - 37) / c) : 
  d = 125 :=
by
  sorry

end NUMINAMATH_GPT_race_distance_l129_12919


namespace NUMINAMATH_GPT_toms_dog_is_12_l129_12998

def toms_cat_age : ℕ := 8
def toms_rabbit_age : ℕ := toms_cat_age / 2
def toms_dog_age : ℕ := toms_rabbit_age * 3

theorem toms_dog_is_12 : toms_dog_age = 12 :=
by
  sorry

end NUMINAMATH_GPT_toms_dog_is_12_l129_12998


namespace NUMINAMATH_GPT_quadratic_real_roots_exists_l129_12984

theorem quadratic_real_roots_exists :
  ∃ (x1 x2 : ℝ), (x1 ≠ x2) ∧ (x1 * x1 - 6 * x1 + 8 = 0) ∧ (x2 * x2 - 6 * x2 + 8 = 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_exists_l129_12984


namespace NUMINAMATH_GPT_axis_of_symmetry_range_of_m_l129_12965

/-- The conditions given in the original mathematical problem -/
noncomputable def f (x : ℝ) : ℝ :=
  let OA := (2 * Real.cos x, Real.sqrt 3)
  let OB := (Real.sin x + Real.sqrt 3 * Real.cos x, -1)
  (OA.1 * OB.1 + OA.2 * OB.2) + 2

/-- Question 1: The axis of symmetry for the function f(x) -/
theorem axis_of_symmetry :
  ∃ k : ℤ, ∀ x : ℝ, (2 * x + Real.pi / 3 = Real.pi / 2 + k * Real.pi) ↔ (x = k * Real.pi / 2 + Real.pi / 12) :=
sorry

/-- Question 2: The range of m such that g(x) = f(x) + m has zero points for x in (0, π/2) -/
theorem range_of_m (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  (∃ c : ℝ, (f x + c = 0)) ↔ ( -4 ≤ c ∧ c < Real.sqrt 3 - 2) :=
sorry

end NUMINAMATH_GPT_axis_of_symmetry_range_of_m_l129_12965


namespace NUMINAMATH_GPT_Q_investment_l129_12962

-- Given conditions
variables (P Q : Nat) (P_investment : P = 30000) (profit_ratio : 2 / 3 = P / Q)

-- Target statement
theorem Q_investment : Q = 45000 :=
by 
  sorry

end NUMINAMATH_GPT_Q_investment_l129_12962


namespace NUMINAMATH_GPT_regular_pentagonal_prism_diagonal_count_l129_12900

noncomputable def diagonal_count (n : ℕ) : ℕ := 
  if n = 5 then 10 else 0

theorem regular_pentagonal_prism_diagonal_count :
  diagonal_count 5 = 10 := 
  by
    sorry

end NUMINAMATH_GPT_regular_pentagonal_prism_diagonal_count_l129_12900


namespace NUMINAMATH_GPT_profit_percent_l129_12936

variable {P C : ℝ}

theorem profit_percent (h1: 2 / 3 * P = 0.82 * C) : ((P - C) / C) * 100 = 23 := by
  have h2 : C = (2 / 3 * P) / 0.82 := by sorry
  have h3 : (P - C) / C = (P - (2 / 3 * P) / 0.82) / ((2 / 3 * P) / 0.82) := by sorry
  have h4 : (P - (2 / 3 * P) / 0.82) / ((2 / 3 * P) / 0.82) = (0.82 * P - 2 / 3 * P) / (2 / 3 * P) := by sorry
  have h5 : (0.82 * P - 2 / 3 * P) / (2 / 3 * P) = 0.1533 := by sorry
  have h6 : 0.1533 * 100 = 23 := by sorry
  sorry

end NUMINAMATH_GPT_profit_percent_l129_12936


namespace NUMINAMATH_GPT_boats_distance_one_minute_before_collision_l129_12922

theorem boats_distance_one_minute_before_collision :
  let speedA := 5  -- miles/hr
  let speedB := 21 -- miles/hr
  let initial_distance := 20 -- miles
  let combined_speed := speedA + speedB -- combined speed in miles/hr
  let speed_per_minute := combined_speed / 60 -- convert to miles/minute
  let time_to_collision := initial_distance / speed_per_minute -- time in minutes until collision
  initial_distance - (time_to_collision - 1) * speed_per_minute = 0.4333 :=
by
  sorry

end NUMINAMATH_GPT_boats_distance_one_minute_before_collision_l129_12922


namespace NUMINAMATH_GPT_negation_of_proposition_l129_12934

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, 0 < x → (x^2 + x > 0)) ↔ ∃ x : ℝ, 0 < x ∧ (x^2 + x ≤ 0) :=
sorry

end NUMINAMATH_GPT_negation_of_proposition_l129_12934


namespace NUMINAMATH_GPT_union_A_B_eq_intersection_A_B_complement_eq_l129_12970

open Set

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x^2 - 4 * x ≤ 0}
def B_complement : Set ℝ := {x | x < 0 ∨ x > 4}

theorem union_A_B_eq : A ∪ B = {x | -1 ≤ x ∧ x ≤ 4} := by
  sorry

theorem intersection_A_B_complement_eq : A ∩ B_complement = {x | -1 ≤ x ∧ x < 0} := by
  sorry

end NUMINAMATH_GPT_union_A_B_eq_intersection_A_B_complement_eq_l129_12970


namespace NUMINAMATH_GPT_Nina_now_l129_12979

def Lisa_age (l m n : ℝ) := l + m + n = 36
def Nina_age (l n : ℝ) := n - 5 = 2 * l
def Mike_age (l m : ℝ) := m + 2 = (l + 2) / 2

theorem Nina_now (l m n : ℝ) (h1 : Lisa_age l m n) (h2 : Nina_age l n) (h3 : Mike_age l m) : n = 34.6 := by
  sorry

end NUMINAMATH_GPT_Nina_now_l129_12979


namespace NUMINAMATH_GPT_hh_two_eq_902_l129_12928

def h (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1

theorem hh_two_eq_902 : h (h 2) = 902 := 
by
  sorry

end NUMINAMATH_GPT_hh_two_eq_902_l129_12928


namespace NUMINAMATH_GPT_geometric_sequence_sum_l129_12954

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 0 + a 1 = 1) (h2 : a 1 + a 2 = 2) : a 5 + a 6 = 32 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l129_12954


namespace NUMINAMATH_GPT_geometric_sequences_identical_l129_12925

theorem geometric_sequences_identical
  (a_0 q r : ℝ)
  (a_n b_n c_n : ℕ → ℝ)
  (H₁ : ∀ n, a_n n = a_0 * q ^ n)
  (H₂ : ∀ n, b_n n = a_0 * r ^ n)
  (H₃ : ∀ n, c_n n = a_n n + b_n n)
  (H₄ : ∃ s : ℝ, ∀ n, c_n n = c_n 0 * s ^ n):
  ∀ n, a_n n = b_n n := sorry

end NUMINAMATH_GPT_geometric_sequences_identical_l129_12925


namespace NUMINAMATH_GPT_find_c_l129_12957

theorem find_c (c : ℕ) (h : 111111222222 = c * (c + 1)) : c = 333333 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_find_c_l129_12957


namespace NUMINAMATH_GPT_carrie_hours_per_day_l129_12953

theorem carrie_hours_per_day (h : ℕ) 
  (worked_4_days : ∀ n, n = 4 * h) 
  (paid_per_hour : ℕ := 22)
  (cost_of_supplies : ℕ := 54)
  (profit : ℕ := 122) :
  88 * h - cost_of_supplies = profit → h = 2 := 
by 
  -- Assume problem conditions and solve
  sorry

end NUMINAMATH_GPT_carrie_hours_per_day_l129_12953


namespace NUMINAMATH_GPT_trisha_dogs_food_expense_l129_12969

theorem trisha_dogs_food_expense :
  ∀ (meat chicken veggies eggs initial remaining final: ℤ),
    meat = 17 → 
    chicken = 22 → 
    veggies = 43 → 
    eggs = 5 → 
    remaining = 35 → 
    initial = 167 →
    final = initial - (meat + chicken + veggies + eggs) - remaining →
    final = 45 := 
by
  intros meat chicken veggies eggs initial remaining final h_meat h_chicken h_veggies h_eggs h_remaining h_initial h_final
  sorry

end NUMINAMATH_GPT_trisha_dogs_food_expense_l129_12969


namespace NUMINAMATH_GPT_smallest_number_of_coins_l129_12988

theorem smallest_number_of_coins (d q : ℕ) (h₁ : 10 * d + 25 * q = 265) (h₂ : d > q) :
  d + q = 16 :=
sorry

end NUMINAMATH_GPT_smallest_number_of_coins_l129_12988


namespace NUMINAMATH_GPT_nonWhiteHomesWithoutFireplace_l129_12947

-- Definitions based on the conditions
def totalHomes : ℕ := 400
def whiteHomes (h : ℕ) : ℕ := h / 4
def nonWhiteHomes (h w : ℕ) : ℕ := h - w
def nonWhiteHomesWithFireplace (nh : ℕ) : ℕ := nh / 5

-- Theorem statement to prove the required result
theorem nonWhiteHomesWithoutFireplace : 
  let h := totalHomes
  let w := whiteHomes h
  let nh := nonWhiteHomes h w
  let nf := nonWhiteHomesWithFireplace nh
  nh - nf = 240 :=
by
  let h := totalHomes
  let w := whiteHomes h
  let nh := nonWhiteHomes h w
  let nf := nonWhiteHomesWithFireplace nh
  show nh - nf = 240
  sorry

end NUMINAMATH_GPT_nonWhiteHomesWithoutFireplace_l129_12947


namespace NUMINAMATH_GPT_rainfall_march_correct_l129_12944

def rainfall_march : ℝ :=
  let april := 4.5
  let may := 3.95
  let june := 3.09
  let july := 4.67
  let average := 4
  let total_expected := 5 * average
  let total_april_to_july := april + may + june + july
  total_expected - total_april_to_july

theorem rainfall_march_correct (march_rainfall : ℝ) :
  let april := 4.5
  let may := 3.95
  let june := 3.09
  let july := 4.67
  let average := 4
  let total_expected := 5 * average
  let total_april_to_july := april + may + june + july
  march_rainfall = total_expected - total_april_to_july :=
by
  sorry

end NUMINAMATH_GPT_rainfall_march_correct_l129_12944


namespace NUMINAMATH_GPT_johns_drive_distance_l129_12940

/-- John's driving problem -/
theorem johns_drive_distance
  (d t : ℝ)
  (h1 : d = 25 * (t + 1.5))
  (h2 : d = 25 + 45 * (t - 1.25)) :
  d = 123.4375 := 
sorry

end NUMINAMATH_GPT_johns_drive_distance_l129_12940


namespace NUMINAMATH_GPT_solve_equation_l129_12929

noncomputable def solution_set (x : ℝ) : Prop :=
  ∃ k : ℤ, x = Real.arcsin (3/4) + 2 * k * Real.pi ∨ x = Real.pi - Real.arcsin (3/4) + 2 * k * Real.pi

theorem solve_equation (x : ℝ) :
  (5 * Real.sin x = 4 + 2 * Real.cos (2 * x)) ↔ solution_set x := 
sorry

end NUMINAMATH_GPT_solve_equation_l129_12929


namespace NUMINAMATH_GPT_first_number_is_105_percent_of_second_kilograms_reduced_by_10_percent_l129_12937

-- Proof problem 1: Given a number is 5% more than another number
theorem first_number_is_105_percent_of_second (x y : ℚ) (h : x = y * 1.05) : x = y * (1 + 0.05) :=
by {
  -- proof here
  sorry
}

-- Proof problem 2: 10 kilograms reduced by 10%
theorem kilograms_reduced_by_10_percent (kg : ℚ) (h : kg = 10) : kg * (1 - 0.1) = 9 :=
by {
  -- proof here
  sorry
}

end NUMINAMATH_GPT_first_number_is_105_percent_of_second_kilograms_reduced_by_10_percent_l129_12937


namespace NUMINAMATH_GPT_sally_and_mary_picked_16_lemons_l129_12943

theorem sally_and_mary_picked_16_lemons (sally_lemons mary_lemons : ℕ) (sally_picked : sally_lemons = 7) (mary_picked : mary_lemons = 9) :
  sally_lemons + mary_lemons = 16 :=
by {
  sorry
}

end NUMINAMATH_GPT_sally_and_mary_picked_16_lemons_l129_12943


namespace NUMINAMATH_GPT_count_scalene_triangles_under_16_l129_12904

theorem count_scalene_triangles_under_16 : 
  ∃ (n : ℕ), n = 6 ∧ ∀ (a b c : ℕ), 
  a < b ∧ b < c ∧ a + b + c < 16 ∧ a + b > c ∧ a + c > b ∧ b + c > a ↔ 
  (a, b, c) ∈ [(2, 3, 4), (2, 4, 5), (2, 5, 6), (3, 4, 5), (3, 5, 6), (4, 5, 6)] :=
by sorry

end NUMINAMATH_GPT_count_scalene_triangles_under_16_l129_12904


namespace NUMINAMATH_GPT_count_logical_propositions_l129_12949

def proposition_1 : Prop := ∃ d : ℕ, d = 1
def proposition_2 : Prop := ∀ n : ℕ, n % 10 = 0 → n % 5 = 0
def proposition_3 : Prop := ∀ t : Prop, t → ¬t

theorem count_logical_propositions :
  (proposition_1 ∧ proposition_3) →
  (proposition_1 ∧ proposition_2 ∧ proposition_3) →
  (∃ (n : ℕ), n = 10 ∧ n % 5 = 0) ∧ n = 2 :=
sorry

end NUMINAMATH_GPT_count_logical_propositions_l129_12949


namespace NUMINAMATH_GPT_man_swim_downstream_distance_l129_12950

-- Define the given conditions
def t_d : ℝ := 6
def t_u : ℝ := 6
def d_u : ℝ := 18
def V_m : ℝ := 4.5

-- The distance the man swam downstream
def distance_downstream : ℝ := 36

-- Prove that given the conditions, the man swam 36 km downstream
theorem man_swim_downstream_distance (V_c : ℝ) :
  (d_u / (V_m - V_c) = t_u) →
  (distance_downstream / (V_m + V_c) = t_d) →
  distance_downstream = 36 :=
by
  sorry

end NUMINAMATH_GPT_man_swim_downstream_distance_l129_12950


namespace NUMINAMATH_GPT_positive_integer_solutions_inequality_l129_12992

theorem positive_integer_solutions_inequality :
  {x : ℕ | 2 * x + 9 ≥ 3 * (x + 2)} = {1, 2, 3} :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_solutions_inequality_l129_12992


namespace NUMINAMATH_GPT_intersection_A_B_eq_C_l129_12977

def A : Set ℝ := { x | 4 - x^2 ≥ 0 }
def B : Set ℝ := { x | x > -1 }
def C : Set ℝ := { x | -1 < x ∧ x ≤ 2 }

theorem intersection_A_B_eq_C : A ∩ B = C := 
by {
  sorry
}

end NUMINAMATH_GPT_intersection_A_B_eq_C_l129_12977


namespace NUMINAMATH_GPT_factorization_problem_l129_12952

theorem factorization_problem (a b c : ℝ) :
  let E := a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3)
  let P := -(a^2 + ab + b^2 + bc + c^2 + ac)
  E = (a - b) * (b - c) * (c - a) * P :=
by
  sorry

end NUMINAMATH_GPT_factorization_problem_l129_12952


namespace NUMINAMATH_GPT_problem_statement_l129_12980

noncomputable def f (x : ℝ) (b c : ℝ) := x^2 + b * x + c

theorem problem_statement (b c : ℝ) (h : ∀ x : ℝ, f (x - 1) b c = f (3 - x) b c) : f 0 b c < f (-2) b c ∧ f (-2) b c < f 5 b c := 
by sorry

end NUMINAMATH_GPT_problem_statement_l129_12980


namespace NUMINAMATH_GPT_exist_line_l1_exist_line_l2_l129_12903

noncomputable def P : ℝ × ℝ := ⟨3, 2⟩

def line1_eq (x y : ℝ) : Prop := 2 * x - y - 4 = 0
def line2_eq (x y : ℝ) : Prop := x - 2 * y + 1 = 0
def perpend_line_eq (x y : ℝ) : Prop := 3 * x + 4 * y - 15 = 0

def line_l1 (x y : ℝ) : Prop := 4 * x - 3 * y - 6 = 0
def line_l2_case1 (x y : ℝ) : Prop := 2 * x - 3 * y = 0
def line_l2_case2 (x y : ℝ) : Prop := x + y - 5 = 0

theorem exist_line_l1 : ∃ (x y : ℝ), line1_eq x y ∧ line2_eq x y ∧ perpend_line_eq x y → line_l1 x y :=
by
  sorry

theorem exist_line_l2 : ∃ (x y : ℝ), line1_eq x y ∧ line2_eq x y ∧ ((line_l2_case1 x y) ∨ (line_l2_case2 x y)) :=
by
  sorry

end NUMINAMATH_GPT_exist_line_l1_exist_line_l2_l129_12903


namespace NUMINAMATH_GPT_two_m_plus_three_b_l129_12991

noncomputable def m : ℚ := (-(3/2) - (1/2)) / (2 - (-1))

noncomputable def b : ℚ := (1/2) - m * (-1)

theorem two_m_plus_three_b :
  2 * m + 3 * b = -11 / 6 :=
by
  sorry

end NUMINAMATH_GPT_two_m_plus_three_b_l129_12991


namespace NUMINAMATH_GPT_find_z_to_8_l129_12994

noncomputable def complex_number_z (z : ℂ) : Prop :=
  z + z⁻¹ = 2 * Complex.cos (Real.pi / 4)

theorem find_z_to_8 (z : ℂ) (h : complex_number_z z) : (z ^ 8 + (z ^ 8)⁻¹ = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_z_to_8_l129_12994


namespace NUMINAMATH_GPT_bob_buys_nose_sprays_l129_12917

theorem bob_buys_nose_sprays (cost_per_spray : ℕ) (promotion : ℕ → ℕ) (total_paid : ℕ)
  (h1 : cost_per_spray = 3)
  (h2 : ∀ n, promotion n = 2 * n)
  (h3 : total_paid = 15) : (total_paid / cost_per_spray) * 2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_bob_buys_nose_sprays_l129_12917


namespace NUMINAMATH_GPT_true_statement_D_l129_12927

-- Definitions related to the problem conditions
def supplementary_angles (a b : ℝ) : Prop := a + b = 180

def exterior_angle_sum_of_polygon (n : ℕ) : ℝ := 360

def acute_angle (a : ℝ) : Prop := a < 90

def triangle_inequality (a b c : ℝ) : Prop := 
  a + b > c ∧ a + c > b ∧ b + c > a

-- The theorem to be proven based on the correct evaluation
theorem true_statement_D (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0):
  triangle_inequality a b c :=
by 
  sorry

end NUMINAMATH_GPT_true_statement_D_l129_12927


namespace NUMINAMATH_GPT_area_of_figure_M_l129_12905

noncomputable def figure_M_area : Real :=
  sorry

theorem area_of_figure_M :
  figure_M_area = 3 :=
  sorry

end NUMINAMATH_GPT_area_of_figure_M_l129_12905


namespace NUMINAMATH_GPT_baseball_team_wins_more_than_three_times_losses_l129_12924

theorem baseball_team_wins_more_than_three_times_losses
    (total_games : ℕ)
    (wins : ℕ)
    (losses : ℕ)
    (h1 : total_games = 130)
    (h2 : wins = 101)
    (h3 : wins + losses = total_games) :
    wins - 3 * losses = 14 :=
by
    -- Proof goes here
    sorry

end NUMINAMATH_GPT_baseball_team_wins_more_than_three_times_losses_l129_12924


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_l129_12906

-- Define the first equation (x-3)^2 + 2x(x-3) = 0
def equation1 (x : ℝ) : Prop := (x - 3)^2 + 2 * x * (x - 3) = 0

-- Define the second equation x^2 - 4x + 1 = 0
def equation2 (x : ℝ) : Prop := x^2 - 4 * x + 1 = 0

-- Theorem stating the solutions for the first equation
theorem solve_equation1 : ∀ (x : ℝ), equation1 x ↔ x = 3 ∨ x = 1 :=
by
  intro x
  sorry  -- Proof is omitted

-- Theorem stating the solutions for the second equation
theorem solve_equation2 : ∀ (x : ℝ), equation2 x ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 :=
by
  intro x
  sorry  -- Proof is omitted

end NUMINAMATH_GPT_solve_equation1_solve_equation2_l129_12906


namespace NUMINAMATH_GPT_highest_score_not_necessarily_at_least_12_l129_12939

section

-- Define the number of teams
def teams : ℕ := 12

-- Define the number of games each team plays
def games_per_team : ℕ := teams - 1

-- Define the total number of games
def total_games : ℕ := (teams * games_per_team) / 2

-- Define the points system
def points_for_win : ℕ := 2
def points_for_draw : ℕ := 1

-- Define the total points in the tournament
def total_points : ℕ := total_games * points_for_win

-- The highest score possible statement
def highest_score_must_be_at_least_12_statement : Prop :=
  ∀ (scores : Fin teams → ℕ), (∃ i, scores i ≥ 12)

-- Theorem stating that the statement "The highest score must be at least 12" is false
theorem highest_score_not_necessarily_at_least_12 (h : ∀ (scores : Fin teams → ℕ), (∃ i, scores i ≥ 12)) : False :=
  sorry

end

end NUMINAMATH_GPT_highest_score_not_necessarily_at_least_12_l129_12939


namespace NUMINAMATH_GPT_initial_students_l129_12956

def students_got_off : ℕ := 3
def students_left : ℕ := 7

theorem initial_students (h1 : students_got_off = 3) (h2 : students_left = 7) :
    students_got_off + students_left = 10 :=
by
  sorry

end NUMINAMATH_GPT_initial_students_l129_12956


namespace NUMINAMATH_GPT_cos_half_angle_inequality_1_cos_half_angle_inequality_2_l129_12902

open Real

variable {A B C : ℝ} (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hA_sum : A + B + C = π)

theorem cos_half_angle_inequality_1 :
  cos (A / 2) < cos (B / 2) + cos (C / 2) :=
by sorry

theorem cos_half_angle_inequality_2 :
  cos (A / 2) < sin (B / 2) + sin (C / 2) :=
by sorry

end NUMINAMATH_GPT_cos_half_angle_inequality_1_cos_half_angle_inequality_2_l129_12902


namespace NUMINAMATH_GPT_roots_quad_sum_abs_gt_four_sqrt_three_l129_12999

theorem roots_quad_sum_abs_gt_four_sqrt_three
  (p r1 r2 : ℝ)
  (h1 : r1 + r2 = -p)
  (h2 : r1 * r2 = 12)
  (h3 : p^2 > 48) : 
  |r1 + r2| > 4 * Real.sqrt 3 := 
by 
  sorry

end NUMINAMATH_GPT_roots_quad_sum_abs_gt_four_sqrt_three_l129_12999


namespace NUMINAMATH_GPT_ascending_order_perimeters_l129_12912

noncomputable def hypotenuse (r : ℝ) : ℝ := r * Real.sqrt 2

noncomputable def perimeter_P (r : ℝ) : ℝ := (2 + 3 * Real.sqrt 2) * r
noncomputable def perimeter_Q (r : ℝ) : ℝ := (6 + Real.sqrt 2) * r
noncomputable def perimeter_R (r : ℝ) : ℝ := (4 + 3 * Real.sqrt 2) * r

theorem ascending_order_perimeters (r : ℝ) (h_r_pos : 0 < r) : 
  perimeter_P r < perimeter_Q r ∧ perimeter_Q r < perimeter_R r := by
  sorry

end NUMINAMATH_GPT_ascending_order_perimeters_l129_12912


namespace NUMINAMATH_GPT_matrix_det_eq_l129_12978

open Matrix

def matrix3x3 (x : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![x + 1, x, x],
    ![x, x + 2, x],
    ![x, x, x + 3]
  ]

theorem matrix_det_eq (x : ℝ) : det (matrix3x3 x) = 2 * x^2 + 11 * x + 6 :=
  sorry

end NUMINAMATH_GPT_matrix_det_eq_l129_12978


namespace NUMINAMATH_GPT_hash_hash_hash_45_l129_12907

def hash (N : ℝ) : ℝ := 0.4 * N + 3

theorem hash_hash_hash_45 : hash (hash (hash 45)) = 7.56 :=
by
  sorry

end NUMINAMATH_GPT_hash_hash_hash_45_l129_12907


namespace NUMINAMATH_GPT_fraction_subtraction_equivalence_l129_12990

theorem fraction_subtraction_equivalence :
  (8 / 19) - (5 / 57) = 1 / 3 :=
by sorry

end NUMINAMATH_GPT_fraction_subtraction_equivalence_l129_12990


namespace NUMINAMATH_GPT_percentage_brand_A_l129_12901

theorem percentage_brand_A
  (A B : ℝ)
  (h1 : 0.6 * A + 0.65 * B = 0.5 * (A + B))
  : (A / (A + B)) * 100 = 60 :=
by
  sorry

end NUMINAMATH_GPT_percentage_brand_A_l129_12901


namespace NUMINAMATH_GPT_fencing_cost_is_correct_l129_12967

def length : ℕ := 60
def cost_per_meter : ℕ := 27 -- using the closest integer value to 26.50
def breadth (l : ℕ) : ℕ := l - 20
def perimeter (l b : ℕ) : ℕ := 2 * l + 2 * b
def total_cost (P : ℕ) (c : ℕ) : ℕ := P * c

theorem fencing_cost_is_correct :
  total_cost (perimeter length (breadth length)) cost_per_meter = 5300 :=
  sorry

end NUMINAMATH_GPT_fencing_cost_is_correct_l129_12967


namespace NUMINAMATH_GPT_highest_geometric_frequency_count_l129_12945

-- Define the problem conditions and the statement to be proved
theorem highest_geometric_frequency_count :
  ∀ (vol : ℕ) (num_groups : ℕ) (cum_freq_first_seven : ℝ)
  (remaining_freqs : List ℕ) (total_freq_remaining : ℕ)
  (r : ℕ) (a : ℕ),
  vol = 100 → 
  num_groups = 10 → 
  cum_freq_first_seven = 0.79 → 
  total_freq_remaining = 21 → 
  r > 1 →
  remaining_freqs = [a, a * r, a * r ^ 2] → 
  a * (1 + r + r ^ 2) = total_freq_remaining → 
  ∃ max_freq, max_freq ∈ remaining_freqs ∧ max_freq = 12 :=
by
  intro vol num_groups cum_freq_first_seven remaining_freqs total_freq_remaining r a
  intros h_vol h_num_groups h_cum_freq_first h_total_freq_remaining h_r_pos h_geom_seq h_freq_sum
  use 12
  sorry

end NUMINAMATH_GPT_highest_geometric_frequency_count_l129_12945


namespace NUMINAMATH_GPT_slope_product_constant_l129_12981

noncomputable def parabola_equation (p : ℝ) (hp : p > 0) : Prop :=
  ∀ x y : ℝ, (x ^ 2 = 2 * y ↔ x ^ 2 = 2 * p * y)

theorem slope_product_constant :
  ∀ (x1 y1 x2 y2 k1 k2 : ℝ) (P A B : ℝ × ℝ),
  P = (2, 2) →
  A = (x1, y1) →
  B = (x2, y2) →
  (∀ k: ℝ, y1 = k * (x1 + 2) + 4 ∧ y2 = k * (x2 + 2) + 4) →
  k1 = (y1 - 2) / (x1 - 2) →
  k2 = (y2 - 2) / (x2 - 2) →
  (x1 + x2 = 2 * k) →
  (x1 * x2 = -4 * k - 8) →
  k1 * k2 = -1 := 
  sorry

end NUMINAMATH_GPT_slope_product_constant_l129_12981


namespace NUMINAMATH_GPT_temperature_conversion_l129_12948

theorem temperature_conversion :
  ∀ (k t : ℝ),
    (t = (5 / 9) * (k - 32) ∧ k = 95) →
    t = 35 := by
  sorry

end NUMINAMATH_GPT_temperature_conversion_l129_12948


namespace NUMINAMATH_GPT_area_of_polygon_DEFG_l129_12958

-- Given conditions
def isosceles_triangle (A B C : Type) (AB AC BC : ℝ) : Prop :=
  AB = AC ∧ AB = 2 ∧ AC = 2 ∧ BC = 1

def square (side : ℝ) : ℝ :=
  side * side

def constructed_square_areas_equal (AB AC : ℝ) (D E F G : Type) : Prop :=
  square AB = square AC ∧ square AB = 4 ∧ square AC = 4

-- Question to prove
theorem area_of_polygon_DEFG (A B C D E F G : Type) (AB AC BC : ℝ) 
  (h1 : isosceles_triangle A B C AB AC BC) 
  (h2 : constructed_square_areas_equal AB AC D E F G) : 
  square AB + square AC = 8 :=
by
  sorry

end NUMINAMATH_GPT_area_of_polygon_DEFG_l129_12958


namespace NUMINAMATH_GPT_fraction_to_zero_power_l129_12946

theorem fraction_to_zero_power :
  756321948 ≠ 0 ∧ -3958672103 ≠ 0 →
  (756321948 / -3958672103 : ℝ) ^ 0 = 1 :=
by
  intro h
  have numerator_nonzero : 756321948 ≠ 0 := h.left
  have denominator_nonzero : -3958672103 ≠ 0 := h.right
  -- Skipping the rest of the proof.
  sorry

end NUMINAMATH_GPT_fraction_to_zero_power_l129_12946


namespace NUMINAMATH_GPT_polygon_sides_l129_12995

theorem polygon_sides (n : Nat) (h : (360 : ℝ) / (180 * (n - 2)) = 2 / 9) : n = 11 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l129_12995


namespace NUMINAMATH_GPT_exists_coprime_linear_combination_l129_12951

theorem exists_coprime_linear_combination (a b p : ℤ) :
  ∃ k l : ℤ, Int.gcd k l = 1 ∧ p ∣ (a * k + b * l) :=
  sorry

end NUMINAMATH_GPT_exists_coprime_linear_combination_l129_12951


namespace NUMINAMATH_GPT_cost_of_goods_l129_12971

theorem cost_of_goods
  (x y z : ℝ)
  (h1 : 3 * x + 7 * y + z = 315)
  (h2 : 4 * x + 10 * y + z = 420) :
  x + y + z = 105 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_goods_l129_12971


namespace NUMINAMATH_GPT_find_f_65_l129_12986

theorem find_f_65 (f : ℝ → ℝ) (h_eq : ∀ x y : ℝ, f (x * y) = x * f y) (h_f1 : f 1 = 40) : f 65 = 2600 :=
by
  sorry

end NUMINAMATH_GPT_find_f_65_l129_12986


namespace NUMINAMATH_GPT_motorist_travel_time_l129_12983

noncomputable def total_time (dist1 dist2 speed1 speed2 : ℝ) : ℝ :=
  (dist1 / speed1) + (dist2 / speed2)

theorem motorist_travel_time (speed1 speed2 : ℝ) (total_dist : ℝ) (half_dist : ℝ) :
  speed1 = 60 → speed2 = 48 → total_dist = 324 → half_dist = total_dist / 2 →
  total_time half_dist half_dist speed1 speed2 = 6.075 :=
by
  intros h1 h2 h3 h4
  simp [total_time, h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_motorist_travel_time_l129_12983


namespace NUMINAMATH_GPT_find_f_of_2_l129_12989

-- Given definitions:
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

def defined_on_neg_inf_to_0 (f : ℝ → ℝ) : Prop := ∀ x, x < 0 → f x = 2 * x^3 + x^2

-- The main theorem to prove:
theorem find_f_of_2 (f : ℝ → ℝ) 
  (h_odd : odd_function f)
  (h_def : defined_on_neg_inf_to_0 f) :
  f 2 = 12 :=
sorry

end NUMINAMATH_GPT_find_f_of_2_l129_12989


namespace NUMINAMATH_GPT_marble_color_197_l129_12964

-- Define the types and properties of the marbles
inductive Color where
  | red | blue | green

-- Define a function to find the color of the nth marble in the cycle pattern
def colorOfMarble (n : Nat) : Color :=
  let cycleLength := 15
  let positionInCycle := n % cycleLength
  if positionInCycle < 6 then Color.red  -- first 6 marbles are red
  else if positionInCycle < 11 then Color.blue  -- next 5 marbles are blue
  else Color.green  -- last 4 marbles are green

-- The theorem asserting the color of the 197th marble
theorem marble_color_197 : colorOfMarble 197 = Color.red :=
sorry

end NUMINAMATH_GPT_marble_color_197_l129_12964


namespace NUMINAMATH_GPT_largest_divisor_of_n_pow4_minus_n_for_composites_l129_12955

def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬(∀ k : ℕ, 1 < k ∧ k < n → ¬(k ∣ n))

theorem largest_divisor_of_n_pow4_minus_n_for_composites : ∀ n : ℕ, is_composite n → 6 ∣ (n^4 - n) :=
by
  intro n
  intro hn
  -- we would add proof steps here if necessary
  sorry

end NUMINAMATH_GPT_largest_divisor_of_n_pow4_minus_n_for_composites_l129_12955


namespace NUMINAMATH_GPT_remainder_103_107_div_11_l129_12961

theorem remainder_103_107_div_11 :
  (103 * 107) % 11 = 10 :=
by
  sorry

end NUMINAMATH_GPT_remainder_103_107_div_11_l129_12961


namespace NUMINAMATH_GPT_value_of_n_l129_12960

theorem value_of_n {k n : ℕ} (h1 : k = 71 * n + 11) (h2 : (k : ℝ) / (n : ℝ) = 71.2) : n = 55 :=
sorry

end NUMINAMATH_GPT_value_of_n_l129_12960


namespace NUMINAMATH_GPT_smallest_number_to_add_for_divisibility_l129_12941

theorem smallest_number_to_add_for_divisibility :
  ∃ x : ℕ, 1275890 + x ≡ 0 [MOD 2375] ∧ x = 1360 :=
by sorry

end NUMINAMATH_GPT_smallest_number_to_add_for_divisibility_l129_12941


namespace NUMINAMATH_GPT_remainder_division_l129_12910

theorem remainder_division
  (P E M S F N T : ℕ)
  (h1 : P = E * M + S)
  (h2 : M = N * F + T) :
  (∃ r, P = (EF + 1) * (P / (EF + 1)) + r ∧ r = ET + S - N) :=
sorry

end NUMINAMATH_GPT_remainder_division_l129_12910


namespace NUMINAMATH_GPT_isosceles_triangle_angles_l129_12938

theorem isosceles_triangle_angles (A B C : ℝ) (h_iso: (A = B) ∨ (B = C) ∨ (A = C)) (angle_A : A = 50) :
  (B = 50) ∨ (B = 65) ∨ (B = 80) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_angles_l129_12938


namespace NUMINAMATH_GPT_no_triangle_possible_l129_12916

-- Define the lengths of the sticks
def stick_lengths : List ℕ := [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

-- The theorem stating the impossibility of forming a triangle with any combination of these lengths
theorem no_triangle_possible : ¬ ∃ (a b c : ℕ), a ∈ stick_lengths ∧ b ∈ stick_lengths ∧ c ∈ stick_lengths ∧
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  (a + b > c ∧ a + c > b ∧ b + c > a) := 
by
  sorry

end NUMINAMATH_GPT_no_triangle_possible_l129_12916


namespace NUMINAMATH_GPT_tabitha_honey_days_l129_12996

noncomputable def days_of_honey (cups_per_day servings_per_cup total_servings : ℕ) : ℕ :=
  total_servings / (cups_per_day * servings_per_cup)

theorem tabitha_honey_days :
  let cups_per_day := 3
  let servings_per_cup := 1
  let ounces_container := 16
  let servings_per_ounce := 6
  let total_servings := ounces_container * servings_per_ounce
  days_of_honey cups_per_day servings_per_cup total_servings = 32 :=
by
  sorry

end NUMINAMATH_GPT_tabitha_honey_days_l129_12996


namespace NUMINAMATH_GPT_sum_of_digits_least_N_l129_12930

-- Define the function P(N)
def P (N : ℕ) : ℚ := (Nat.ceil (3 * N / 5 + 1) : ℕ) / (N + 1)

-- Define the predicate that checks if P(N) is less than 321/400
def P_lt_321_over_400 (N : ℕ) : Prop := P N < (321 / 400 : ℚ)

-- Define a function that sums the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- The main statement: we claim the least multiple of 5 satisfying the condition
-- That the sum of its digits is 12
theorem sum_of_digits_least_N :
  ∃ N : ℕ, 
    (N % 5 = 0) ∧ 
    P_lt_321_over_400 N ∧ 
    (∀ N' : ℕ, (N' % 5 = 0) → P_lt_321_over_400 N' → N' ≥ N) ∧ 
    sum_of_digits N = 12 := 
sorry

end NUMINAMATH_GPT_sum_of_digits_least_N_l129_12930


namespace NUMINAMATH_GPT_suzhou_metro_scientific_notation_l129_12931

theorem suzhou_metro_scientific_notation : 
  (∃(a : ℝ) (n : ℤ), 
    1 ≤ abs a ∧ abs a < 10 ∧ 15.6 * 10^9 = a * 10^n) → 
    (a = 1.56 ∧ n = 9) := 
by
  sorry

end NUMINAMATH_GPT_suzhou_metro_scientific_notation_l129_12931


namespace NUMINAMATH_GPT_probability_of_interval_l129_12933

-- Define the random variable ξ and its probability distribution P(ξ = k)
variables (ξ : ℕ → ℝ) (P : ℕ → ℝ)

-- Define a constant a
noncomputable def a : ℝ := 5/4

-- Given conditions
axiom condition1 : ∀ k, k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 4 → P k = a / (k * (k + 1))
axiom condition2 : P 1 + P 2 + P 3 + P 4 = 1

-- Statement to prove
theorem probability_of_interval : P 1 + P 2 = 5/6 :=
by sorry

end NUMINAMATH_GPT_probability_of_interval_l129_12933


namespace NUMINAMATH_GPT_solve_proof_problem_l129_12923

noncomputable def proof_problem : Prop :=
  let short_videos_per_day := 2
  let short_video_time := 2
  let longer_videos_per_day := 1
  let week_days := 7
  let total_weekly_video_time := 112
  let total_short_video_time_per_week := short_videos_per_day * short_video_time * week_days
  let total_longer_video_time_per_week := total_weekly_video_time - total_short_video_time_per_week
  let longer_video_multiple := total_longer_video_time_per_week / short_video_time
  longer_video_multiple = 42

theorem solve_proof_problem : proof_problem :=
by
  /- Proof goes here -/
  sorry

end NUMINAMATH_GPT_solve_proof_problem_l129_12923


namespace NUMINAMATH_GPT_max_value_of_f_symmetric_about_point_concave_inequality_l129_12935

noncomputable def f (x : ℝ) : ℝ := x^2 / (1 - x)

theorem max_value_of_f : ∃ x, f x = -4 :=
by
  sorry

theorem symmetric_about_point : ∀ x, f (1 - x) + f (1 + x) = -4 :=
by
  sorry

theorem concave_inequality (x1 x2 : ℝ) (h1 : x1 > 1) (h2 : x2 > 1) : 
  f ((x1 + x2) / 2) ≥ (f x1 + f x2) / 2 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_f_symmetric_about_point_concave_inequality_l129_12935


namespace NUMINAMATH_GPT_perimeter_of_garden_l129_12914

-- Definitions based on conditions
def length : ℕ := 150
def breadth : ℕ := 150
def is_square (l b : ℕ) := l = b

-- Theorem statement proving the perimeter given conditions
theorem perimeter_of_garden : is_square length breadth → 4 * length = 600 :=
by
  intro h
  rw [h]
  norm_num
  sorry

end NUMINAMATH_GPT_perimeter_of_garden_l129_12914


namespace NUMINAMATH_GPT_simplify_expression_l129_12976

-- Defining the original expression
def original_expr (y : ℝ) : ℝ := 3 * y^3 - 7 * y^2 + 12 * y + 5 - (2 * y^3 - 4 + 3 * y^2 - 9 * y)

-- Defining the simplified expression
def simplified_expr (y : ℝ) : ℝ := y^3 - 10 * y^2 + 21 * y + 9

-- The statement to prove
theorem simplify_expression (y : ℝ) : original_expr y = simplified_expr y :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l129_12976


namespace NUMINAMATH_GPT_stratified_sampling_numbers_l129_12959

-- Definitions of the conditions
def total_teachers : ℕ := 300
def senior_teachers : ℕ := 90
def intermediate_teachers : ℕ := 150
def junior_teachers : ℕ := 60
def sample_size : ℕ := 40

-- Hypothesis of proportions
def proportion_senior := senior_teachers / total_teachers
def proportion_intermediate := intermediate_teachers / total_teachers
def proportion_junior := junior_teachers / total_teachers

-- Expected sample counts using stratified sampling method
def expected_senior_drawn := proportion_senior * sample_size
def expected_intermediate_drawn := proportion_intermediate * sample_size
def expected_junior_drawn := proportion_junior * sample_size

-- Proof goal
theorem stratified_sampling_numbers :
  (expected_senior_drawn = 12) ∧ 
  (expected_intermediate_drawn = 20) ∧ 
  (expected_junior_drawn = 8) :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_numbers_l129_12959


namespace NUMINAMATH_GPT_garden_area_l129_12987

theorem garden_area (w l : ℕ) (h1 : l = 3 * w + 30) (h2 : 2 * (w + l) = 780) : 
  w * l = 27000 := 
by 
  sorry

end NUMINAMATH_GPT_garden_area_l129_12987


namespace NUMINAMATH_GPT_projectile_hits_ground_at_5_over_2_l129_12913

theorem projectile_hits_ground_at_5_over_2 :
  ∃ t : ℚ, (-20) * t ^ 2 + 26 * t + 60 = 0 ∧ t = 5 / 2 :=
sorry

end NUMINAMATH_GPT_projectile_hits_ground_at_5_over_2_l129_12913


namespace NUMINAMATH_GPT_perpendicular_line_through_point_l129_12975

theorem perpendicular_line_through_point (x y : ℝ) (c : ℝ) (P : ℝ × ℝ) :
  P = (-1, 2) →
  (∀ x y c : ℝ, (2*x - y + c = 0) ↔ (x+2*y-1=0) → (x+2*y-1=0)) →
  ∃ c : ℝ, 2*(-1) - 2 + c = 0 ∧ (2*x - y + c = 0) :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_line_through_point_l129_12975


namespace NUMINAMATH_GPT_land_to_water_time_ratio_l129_12942

-- Define the conditions
def distance_water : ℕ := 50
def distance_land : ℕ := 300
def speed_ratio : ℕ := 3

-- Define the Lean theorem statement
theorem land_to_water_time_ratio (x : ℝ) (hx : x > 0) : 
  (distance_land / (speed_ratio * x)) / (distance_water / x) = 2 := by
  sorry

end NUMINAMATH_GPT_land_to_water_time_ratio_l129_12942


namespace NUMINAMATH_GPT_number_of_children_l129_12926

theorem number_of_children (A V S : ℕ) (x : ℕ → ℕ) (n : ℕ) 
  (h1 : (A / 2) + V = (A + V + S + (Finset.range (n - 3)).sum x) / n)
  (h2 : S + A = V + (Finset.range (n - 3)).sum x) : 
  n = 6 :=
sorry

end NUMINAMATH_GPT_number_of_children_l129_12926


namespace NUMINAMATH_GPT_yoongi_more_points_l129_12972

def yoongiPoints : ℕ := 4
def jungkookPoints : ℕ := 6 - 3

theorem yoongi_more_points : yoongiPoints > jungkookPoints := by
  sorry

end NUMINAMATH_GPT_yoongi_more_points_l129_12972
