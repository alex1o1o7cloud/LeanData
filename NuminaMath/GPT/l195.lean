import Mathlib

namespace koala_food_consumed_l195_19587

theorem koala_food_consumed (x y : ℝ) (h1 : 0.40 * x = 12) (h2 : 0.20 * y = 2) : 
  x = 30 ∧ y = 10 := 
by
  sorry

end koala_food_consumed_l195_19587


namespace f_satisfies_conditions_l195_19574

def g (n : Int) : Int :=
  if n >= 1 then 1 else 0

def f (n m : Int) : Int :=
  if m = 0 then n
  else n % m

theorem f_satisfies_conditions (n m : Int) : 
  (f 0 m = 0) ∧ 
  (f (n + 1) m = (1 - g m + g m * g (m - 1 - f n m)) * (1 + f n m)) := by
  sorry

end f_satisfies_conditions_l195_19574


namespace range_of_a_outside_circle_l195_19500

  variable (a : ℝ)

  def point_outside_circle (a : ℝ) : Prop :=
    let x := a
    let y := 2
    let distance_sqr := (x - a) ^ 2 + (y - 3 / 2) ^ 2
    let r_sqr := 1 / 4
    distance_sqr > r_sqr

  theorem range_of_a_outside_circle {a : ℝ} (h : point_outside_circle a) :
      2 < a ∧ a < 9 / 4 := sorry
  
end range_of_a_outside_circle_l195_19500


namespace hyperbola_equation_l195_19550

theorem hyperbola_equation (a b : ℝ) (h₁ : a^2 + b^2 = 25) (h₂ : 2 * b / a = 1) : 
  a = 2 * Real.sqrt 5 ∧ b = Real.sqrt 5 ∧ (∀ x y : ℝ, x^2 / (a^2) - y^2 / (b^2) = 1 ↔ x^2 / 20 - y^2 / 5 = 1) :=
by
  sorry

end hyperbola_equation_l195_19550


namespace largest_divisor_of_product_of_five_consecutive_integers_l195_19592

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∃ n, (∀ k : ℤ, n ∣ (k * (k + 1) * (k + 2) * (k + 3) * (k + 4))) ∧ n = 60 :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l195_19592


namespace cos_double_angle_l195_19504

theorem cos_double_angle (α : ℝ) (h : Real.tan α = -3) : Real.cos (2 * α) = -4 / 5 := sorry

end cos_double_angle_l195_19504


namespace total_animal_legs_l195_19549

theorem total_animal_legs (total_animals : ℕ) (sheep : ℕ) (chickens : ℕ) : 
  total_animals = 20 ∧ sheep = 10 ∧ chickens = 10 ∧ 
  2 * chickens + 4 * sheep = 60 :=
by 
  sorry

end total_animal_legs_l195_19549


namespace scientific_notation_correct_l195_19559

def number_in_scientific_notation : ℝ := 1600000
def expected_scientific_notation : ℝ := 1.6 * 10^6

theorem scientific_notation_correct :
  number_in_scientific_notation = expected_scientific_notation := by
  sorry

end scientific_notation_correct_l195_19559


namespace find_missing_dimension_l195_19586

def carton_volume (l w h : ℕ) : ℕ := l * w * h

def soapbox_base_area (l w : ℕ) : ℕ := l * w

def total_base_area (n l w : ℕ) : ℕ := n * soapbox_base_area l w

def missing_dimension (carton_volume total_base_area : ℕ) : ℕ := carton_volume / total_base_area

theorem find_missing_dimension 
  (carton_l carton_w carton_h : ℕ) 
  (soapbox_l soapbox_w : ℕ) 
  (n : ℕ) 
  (h_carton_l : carton_l = 25)
  (h_carton_w : carton_w = 48)
  (h_carton_h : carton_h = 60)
  (h_soapbox_l : soapbox_l = 8)
  (h_soapbox_w : soapbox_w = 6)
  (h_n : n = 300) :
  missing_dimension (carton_volume carton_l carton_w carton_h) (total_base_area n soapbox_l soapbox_w) = 5 := 
by 
  sorry

end find_missing_dimension_l195_19586


namespace units_digit_diff_is_seven_l195_19557

noncomputable def units_digit_resulting_difference (a b c : ℕ) (h1 : a = c - 3) :=
  let original := 100 * a + 10 * b + c
  let reversed := 100 * c + 10 * b + a
  let difference := original - reversed
  difference % 10

theorem units_digit_diff_is_seven (a b c : ℕ) (h1 : a = c - 3) :
  units_digit_resulting_difference a b c h1 = 7 :=
by sorry

end units_digit_diff_is_seven_l195_19557


namespace stuffed_animal_sales_l195_19522

theorem stuffed_animal_sales (Q T J : ℕ) 
  (h1 : Q = 100 * T) 
  (h2 : J = T + 15) 
  (h3 : Q = 2000) : 
  Q - J = 1965 := 
by
  sorry

end stuffed_animal_sales_l195_19522


namespace area_of_smallest_square_that_encloses_circle_l195_19553

def radius : ℕ := 5

def diameter (r : ℕ) : ℕ := 2 * r

def side_length (d : ℕ) : ℕ := d

def area_of_square (s : ℕ) : ℕ := s * s

theorem area_of_smallest_square_that_encloses_circle :
  area_of_square (side_length (diameter radius)) = 100 := by
  sorry

end area_of_smallest_square_that_encloses_circle_l195_19553


namespace common_tangents_count_l195_19534

def circleC1 : Prop := ∃ (x y : ℝ), x^2 + y^2 + 2 * x - 6 * y - 15 = 0
def circleC2 : Prop := ∃ (x y : ℝ), x^2 + y^2 - 4 * x + 2 * y + 4 = 0

theorem common_tangents_count (C1 : circleC1) (C2 : circleC2) : 
  ∃ (n : ℕ), n = 2 := 
sorry

end common_tangents_count_l195_19534


namespace total_expenses_l195_19590

def tulips : ℕ := 250
def carnations : ℕ := 375
def roses : ℕ := 320
def cost_per_flower : ℕ := 2

theorem total_expenses :
  tulips + carnations + roses * cost_per_flower = 1890 := 
sorry

end total_expenses_l195_19590


namespace min_abs_val_sum_l195_19518

noncomputable def abs_val_sum_min : ℝ := (4:ℝ)^(1/3)

theorem min_abs_val_sum (a b c : ℝ) (h : |(a - b) * (b - c) * (c - a)| = 1) :
  |a| + |b| + |c| >= abs_val_sum_min :=
sorry

end min_abs_val_sum_l195_19518


namespace roots_sum_of_squares_l195_19591

theorem roots_sum_of_squares {r s : ℝ} (h : Polynomial.roots (X^2 - 3*X + 1) = {r, s}) : r^2 + s^2 = 7 :=
by
  sorry

end roots_sum_of_squares_l195_19591


namespace hyperbola_eccentricity_asymptotic_lines_l195_19597

-- Define the conditions and the proof goal:

theorem hyperbola_eccentricity_asymptotic_lines {a b c e : ℝ} 
  (h_asym : ∀ x y : ℝ, (y = x ∨ y = -x) ↔ (a = b)) 
  (h_c : c = Real.sqrt (a ^ 2 + b ^ 2))
  (h_e : e = c / a) : e = Real.sqrt 2 := sorry

end hyperbola_eccentricity_asymptotic_lines_l195_19597


namespace unit_prices_and_purchasing_schemes_l195_19516

theorem unit_prices_and_purchasing_schemes :
  ∃ (x y : ℕ),
    (14 * x + 8 * y = 1600) ∧
    (3 * x = 4 * y) ∧
    (x = 80) ∧ 
    (y = 60) ∧
    ∃ (m : ℕ), 
      (m ≥ 29) ∧ 
      (m ≤ 30) ∧ 
      (80 * m + 60 * (50 - m) ≤ 3600) ∧
      (m = 29 ∨ m = 30) := 
sorry

end unit_prices_and_purchasing_schemes_l195_19516


namespace distance_3D_l195_19545

theorem distance_3D : 
  let A := (2, -2, 0)
  let B := (8, 8, 3)
  let d := Real.sqrt ((8 - 2) ^ 2 + (8 - (-2)) ^ 2 + (3 - 0) ^ 2)
  d = Real.sqrt 145 :=
by
  let A := (2, -2, 0)
  let B := (8, 8, 3)
  let d := Real.sqrt ((8 - 2) ^ 2 + (8 - (-2)) ^ 2 + (3 - 0) ^ 2)
  dsimp [A, B, d]
  sorry

end distance_3D_l195_19545


namespace lottery_probability_correct_l195_19513

def number_of_winnerballs_ways : ℕ := Nat.choose 50 6

def probability_megaBall : ℚ := 1 / 30

def probability_winnerBalls : ℚ := 1 / number_of_winnerballs_ways

def combined_probability : ℚ := probability_megaBall * probability_winnerBalls

theorem lottery_probability_correct : combined_probability = 1 / 476721000 := by
  sorry

end lottery_probability_correct_l195_19513


namespace prime_square_mod_180_l195_19544

theorem prime_square_mod_180 (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : 5 < p) :
  ∃ (s : Finset ℕ), s.card = 2 ∧ ∀ r ∈ s, ∃ n : ℕ, p^2 % 180 = r :=
sorry

end prime_square_mod_180_l195_19544


namespace ravenswood_forest_percentage_l195_19594

def ravenswood_gnomes (westerville_gnomes : ℕ) : ℕ := 4 * westerville_gnomes
def remaining_gnomes (total_gnomes taken_percentage: ℕ) : ℕ := (total_gnomes * (100 - taken_percentage)) / 100

theorem ravenswood_forest_percentage:
  ∀ (westerville_gnomes : ℕ) (remaining : ℕ) (total_gnomes : ℕ),
  westerville_gnomes = 20 →
  total_gnomes = ravenswood_gnomes westerville_gnomes →
  remaining = 48 →
  remaining_gnomes total_gnomes 40 = remaining :=
by
  sorry

end ravenswood_forest_percentage_l195_19594


namespace sum_of_ai_powers_l195_19554

theorem sum_of_ai_powers :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ),
  (∀ x : ℝ, (1 + x) * (1 - 2 * x)^8 = 
            a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + 
            a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + 
            a_7 * x^7 + a_8 * x^8 + a_9 * x^9) →
  a_1 * 2 + a_2 * 2^2 + a_3 * 2^3 + 
  a_4 * 2^4 + a_5 * 2^5 + a_6 * 2^6 + 
  a_7 * 2^7 + a_8 * 2^8 + a_9 * 2^9 = 3^9 - 1 :=
by
  sorry

end sum_of_ai_powers_l195_19554


namespace reflection_matrix_correct_l195_19539

-- Definitions based on the conditions given in the problem
def reflect_over_line_y_eq_x_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, 1], ![1, 0]]

-- The main theorem to state the equivalence
theorem reflection_matrix_correct :
  reflect_over_line_y_eq_x_matrix = ![![0, 1], ![1, 0]] :=
by
  sorry

end reflection_matrix_correct_l195_19539


namespace sum_abs_coeffs_expansion_l195_19584

theorem sum_abs_coeffs_expansion (x : ℝ) :
  (|1 - 0 * x| + |1 - 3 * x| + |1 - 3^2 * x^2| + |1 - 3^3 * x^3| + |1 - 3^4 * x^4| + |1 - 3^5 * x^5| = 1024) :=
sorry

end sum_abs_coeffs_expansion_l195_19584


namespace sqrt_value_l195_19593

theorem sqrt_value {A B C : ℝ} (x y : ℝ) 
  (h1 : A = 5 * Real.sqrt (2 * x + 1)) 
  (h2 : B = 3 * Real.sqrt (x + 3)) 
  (h3 : C = Real.sqrt (10 * x + 3 * y)) 
  (h4 : A + B = C) 
  (h5 : 2 * x + 1 = x + 3) : 
  Real.sqrt (2 * y - x^2) = 14 :=
by
  sorry

end sqrt_value_l195_19593


namespace hieu_catches_up_beatrice_in_5_minutes_probability_beatrice_hieu_same_place_range_of_x_for_meeting_probability_l195_19572

namespace CatchUpProblem

-- Part (a)
theorem hieu_catches_up_beatrice_in_5_minutes :
  ∀ (d_b_walked : ℕ) (relative_speed : ℕ) (catch_up_time : ℕ),
  d_b_walked = 5 / 6 ∧ relative_speed = 10 ∧ catch_up_time = 5 :=
sorry

-- Part (b)(i)
theorem probability_beatrice_hieu_same_place :
  ∀ (total_pairs : ℕ) (valid_pairs : ℕ) (probability : Rat),
  total_pairs = 3600 ∧ valid_pairs = 884 ∧ probability = 221 / 900 :=
sorry

-- Part (b)(ii)
theorem range_of_x_for_meeting_probability :
  ∀ (probability : Rat) (valid_pairs : ℕ) (total_pairs : ℕ) (lower_bound : ℕ) (upper_bound : ℕ),
  probability = 13 / 200 ∧ valid_pairs = 234 ∧ total_pairs = 3600 ∧ 
  lower_bound = 10 ∧ upper_bound = 120 / 11 :=
sorry

end CatchUpProblem

end hieu_catches_up_beatrice_in_5_minutes_probability_beatrice_hieu_same_place_range_of_x_for_meeting_probability_l195_19572


namespace rectangle_area_is_180_l195_19547

def area_of_square (side : ℕ) : ℕ := side * side
def length_of_rectangle (radius : ℕ) : ℕ := (2 * radius) / 5
def area_of_rectangle (length breadth : ℕ) : ℕ := length * breadth

theorem rectangle_area_is_180 :
  ∀ (side breadth : ℕ), 
    area_of_square side = 2025 → 
    breadth = 10 → 
    area_of_rectangle (length_of_rectangle side) breadth = 180 :=
by
  intros side breadth h_area h_breadth
  sorry

end rectangle_area_is_180_l195_19547


namespace evaluate_expression_l195_19583

theorem evaluate_expression : (502 * 502) - (501 * 503) = 1 := sorry

end evaluate_expression_l195_19583


namespace problem_l195_19578

def f (x : ℤ) := 3 * x + 2

theorem problem : f (f (f 3)) = 107 := by
  sorry

end problem_l195_19578


namespace meadow_to_campsite_distance_l195_19512

variable (d1 d2 d_total d_meadow_to_campsite : ℝ)

theorem meadow_to_campsite_distance
  (h1 : d1 = 0.2)
  (h2 : d2 = 0.4)
  (h_total : d_total = 0.7)
  (h_before_meadow : d_before_meadow = d1 + d2)
  (h_distance : d_meadow_to_campsite = d_total - d_before_meadow) :
  d_meadow_to_campsite = 0.1 :=
by 
  sorry

end meadow_to_campsite_distance_l195_19512


namespace binom_odd_n_eq_2_pow_m_minus_1_l195_19541

open Nat

/-- For which n will binom n k be odd for every 0 ≤ k ≤ n?
    Prove that n = 2^m - 1 for some m ≥ 1. -/
theorem binom_odd_n_eq_2_pow_m_minus_1 (n : ℕ) :
  (∀ k : ℕ, k ≤ n → Nat.choose n k % 2 = 1) ↔ (∃ m : ℕ, m ≥ 1 ∧ n = 2^m - 1) :=
by
  sorry

end binom_odd_n_eq_2_pow_m_minus_1_l195_19541


namespace total_spent_after_discount_and_tax_l195_19598

-- Define prices for each item
def price_bracelet := 4
def price_keychain := 5
def price_coloring_book := 3
def price_sticker := 1
def price_toy_car := 6

-- Define discounts and tax rates
def discount_bracelet := 0.10
def sales_tax := 0.05

-- Define the quantity of each item purchased by Paula, Olive, and Nathan
def quantity_paula_bracelets := 3
def quantity_paula_keychains := 2
def quantity_paula_coloring_books := 1
def quantity_paula_stickers := 4

def quantity_olive_coloring_books := 1
def quantity_olive_bracelets := 2
def quantity_olive_toy_cars := 1
def quantity_olive_stickers := 3

def quantity_nathan_toy_cars := 4
def quantity_nathan_stickers := 5
def quantity_nathan_keychains := 1

-- Function to calculate total cost before discount and tax
def total_cost_before_discount_and_tax (bracelets keychains coloring_books stickers toy_cars : Nat) : Float :=
  Float.ofNat (bracelets * price_bracelet) +
  Float.ofNat (keychains * price_keychain) +
  Float.ofNat (coloring_books * price_coloring_book) +
  Float.ofNat (stickers * price_sticker) +
  Float.ofNat (toy_cars * price_toy_car)

-- Function to calculate discount on bracelets
def bracelet_discount (bracelets : Nat) : Float :=
  Float.ofNat (bracelets * price_bracelet) * discount_bracelet

-- Function to calculate total cost after discount and before tax
def total_cost_after_discount (total_cost discount : Float) : Float :=
  total_cost - discount

-- Function to calculate total cost after tax
def total_cost_after_tax (total_cost : Float) (tax_rate : Float) : Float :=
  total_cost * (1 + tax_rate)

-- Proof statement (no proof provided, only the statement)
theorem total_spent_after_discount_and_tax : 
  total_cost_after_tax (
    total_cost_after_discount
      (total_cost_before_discount_and_tax quantity_paula_bracelets quantity_paula_keychains quantity_paula_coloring_books quantity_paula_stickers 0)
      (bracelet_discount quantity_paula_bracelets)
    +
    total_cost_after_discount
      (total_cost_before_discount_and_tax quantity_olive_bracelets 0 quantity_olive_coloring_books quantity_olive_stickers quantity_olive_toy_cars)
      (bracelet_discount quantity_olive_bracelets)
    +
    total_cost_before_discount_and_tax 0 quantity_nathan_keychains 0 quantity_nathan_stickers quantity_nathan_toy_cars
  ) sales_tax = 85.05 := 
sorry

end total_spent_after_discount_and_tax_l195_19598


namespace find_principal_l195_19535

theorem find_principal (SI : ℝ) (R : ℝ) (T : ℝ) (hSI : SI = 4025.25) (hR : R = 9) (hT : T = 5) :
    let P := SI / (R * T / 100)
    P = 8950 :=
by
  -- we will put proof steps here
  sorry

end find_principal_l195_19535


namespace AlissaMorePresents_l195_19579

/-- Ethan has 31 presents -/
def EthanPresents : ℕ := 31

/-- Alissa has 53 presents -/
def AlissaPresents : ℕ := 53

/-- How many more presents does Alissa have than Ethan? -/
theorem AlissaMorePresents : AlissaPresents - EthanPresents = 22 := by
  -- Place the proof here
  sorry

end AlissaMorePresents_l195_19579


namespace tina_sells_more_than_katya_l195_19552

noncomputable def katya_rev : ℝ := 8 * 1.5
noncomputable def ricky_rev : ℝ := 9 * 2.0
noncomputable def combined_rev : ℝ := katya_rev + ricky_rev
noncomputable def tina_target : ℝ := 2 * combined_rev
noncomputable def tina_glasses : ℝ := tina_target / 3.0
noncomputable def difference_glasses : ℝ := tina_glasses - 8

theorem tina_sells_more_than_katya :
  difference_glasses = 12 := by
  sorry

end tina_sells_more_than_katya_l195_19552


namespace no_positive_integer_solutions_l195_19558

theorem no_positive_integer_solutions (m : ℕ) (h_pos : m > 0) :
  ¬ ∃ x : ℚ, m * x^2 + 40 * x + m = 0 :=
by {
  -- the proof goes here
  sorry
}

end no_positive_integer_solutions_l195_19558


namespace difference_between_twice_smaller_and_larger_is_three_l195_19542

theorem difference_between_twice_smaller_and_larger_is_three
(S L x : ℕ) 
(h1 : L = 2 * S - x) 
(h2 : S + L = 39)
(h3 : S = 14) : 
2 * S - L = 3 := 
sorry

end difference_between_twice_smaller_and_larger_is_three_l195_19542


namespace flower_combinations_l195_19530

theorem flower_combinations (t l : ℕ) (h : 4 * t + 3 * l = 60) : 
  ∃ (t_values : Finset ℕ), (∀ x ∈ t_values, 0 ≤ x ∧ x ≤ 15 ∧ x % 3 = 0) ∧
  t_values.card = 6 :=
sorry

end flower_combinations_l195_19530


namespace john_read_books_in_15_hours_l195_19506

theorem john_read_books_in_15_hours (hreads_faster_ratio : ℝ) (brother_time : ℝ) (john_read_time : ℝ) : john_read_time = brother_time / hreads_faster_ratio → 3 * john_read_time = 15 :=
by
  intros H
  sorry

end john_read_books_in_15_hours_l195_19506


namespace infinite_non_congruent_right_triangles_l195_19526

noncomputable def right_triangle_equal_perimeter_area : Prop :=
  ∃ (a b c : ℝ), (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a^2 + b^2 = c^2) ∧ 
  (a + b + c = (1/2) * a * b)

theorem infinite_non_congruent_right_triangles :
  ∃ (k : ℕ), right_triangle_equal_perimeter_area :=
sorry

end infinite_non_congruent_right_triangles_l195_19526


namespace max_value_f_l195_19563

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 2 + Real.sqrt 3 * Real.cos x - 3 / 4

theorem max_value_f :
  ∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x ≤ 1 :=
sorry

end max_value_f_l195_19563


namespace chess_team_girls_l195_19514

theorem chess_team_girls (B G : ℕ) (h1 : B + G = 26) (h2 : (G / 2) + B = 16) : G = 20 := by
  sorry

end chess_team_girls_l195_19514


namespace abs_f_at_1_eq_20_l195_19543

noncomputable def fourth_degree_polynomial (f : ℝ → ℝ) : Prop :=
  ∃ p : Polynomial ℝ, p.degree = 4 ∧ ∀ x, f x = p.eval x

theorem abs_f_at_1_eq_20 
  (f : ℝ → ℝ)
  (h_f_poly : fourth_degree_polynomial f)
  (h_f_neg2 : |f (-2)| = 10)
  (h_f_0 : |f 0| = 10)
  (h_f_3 : |f 3| = 10)
  (h_f_7 : |f 7| = 10) :
  |f 1| = 20 := 
sorry

end abs_f_at_1_eq_20_l195_19543


namespace math_problem_l195_19548

theorem math_problem 
  (a : Int) (b : Int) (c : Int)
  (h_a : a = -1)
  (h_b : b = 1)
  (h_c : c = 0) :
  a + c - b = -2 := 
by
  sorry

end math_problem_l195_19548


namespace neon_signs_blink_together_l195_19567

theorem neon_signs_blink_together (a b : ℕ) (ha : a = 9) (hb : b = 15) : Nat.lcm a b = 45 := by
  rw [ha, hb]
  have : Nat.lcm 9 15 = 45 := by sorry
  exact this

end neon_signs_blink_together_l195_19567


namespace radius_of_circle_tangent_to_xaxis_l195_19520

theorem radius_of_circle_tangent_to_xaxis
  (Ω : Set (ℝ × ℝ)) (Γ : Set (ℝ × ℝ))
  (hΓ : ∀ x y : ℝ, (x, y) ∈ Γ ↔ y^2 = 4 * x)
  (F : ℝ × ℝ) (hF : F = (1, 0))
  (hΩ_tangent : ∃ r : ℝ, ∀ x y : ℝ, (x - 1)^2 + (y - r)^2 = r^2 ∧ (1, 0) ∈ Ω)
  (hΩ_intersect : ∀ x y : ℝ, (x, y) ∈ Ω → (x, y) ∈ Γ → (x, y) = (1, 0)) :
  ∃ r : ℝ, r = 4 * Real.sqrt 3 / 9 :=
sorry

end radius_of_circle_tangent_to_xaxis_l195_19520


namespace functional_equation_solution_l195_19573

theorem functional_equation_solution (f : ℚ → ℚ) (H : ∀ x y : ℚ, f (x + y) = f x + f y) :
  ∃ k : ℚ, ∀ x : ℚ, f x = k * x := 
sorry

end functional_equation_solution_l195_19573


namespace julia_money_given_l195_19599

-- Define the conditions
def num_snickers : ℕ := 2
def num_mms : ℕ := 3
def cost_snickers : ℚ := 1.5
def cost_mms : ℚ := 2 * cost_snickers
def change_received : ℚ := 8

-- The total cost Julia had to pay
def total_cost : ℚ := (num_snickers * cost_snickers) + (num_mms * cost_mms)

-- Julia gave this amount of money to the cashier
def money_given : ℚ := total_cost + change_received

-- The problem to prove
theorem julia_money_given : money_given = 20 := by
  sorry

end julia_money_given_l195_19599


namespace relationship_between_c_squared_and_ab_l195_19510

theorem relationship_between_c_squared_and_ab (a b c : ℝ) 
  (h_pos_a : a > 0) 
  (h_pos_b : b > 0) 
  (h_pos_c : c > 0) 
  (h_c : c = (a + b) / 2) : 
  c^2 ≥ a * b := 
sorry

end relationship_between_c_squared_and_ab_l195_19510


namespace general_term_formula_no_pos_int_for_S_n_gt_40n_plus_600_exists_pos_int_for_S_n_gt_40n_plus_600_l195_19582

noncomputable def arith_seq (n : ℕ) (d : ℝ) :=
  2 + (n - 1) * d

theorem general_term_formula :
  ∃ d, ∀ n, arith_seq n d = 2 ∨ arith_seq n d = 4 * n - 2 :=
by sorry

theorem no_pos_int_for_S_n_gt_40n_plus_600 :
  ∀ n, (arith_seq n 0) * n ≤ 40 * n + 600 :=
by sorry

theorem exists_pos_int_for_S_n_gt_40n_plus_600 :
  ∃ n, (arith_seq n 4) * n > 40 * n + 600 ∧ n = 31 :=
by sorry

end general_term_formula_no_pos_int_for_S_n_gt_40n_plus_600_exists_pos_int_for_S_n_gt_40n_plus_600_l195_19582


namespace tan_195_l195_19580

theorem tan_195 (a : ℝ) (h : Real.cos 165 = a) : Real.tan 195 = - (Real.sqrt (1 - a^2)) / a := 
sorry

end tan_195_l195_19580


namespace students_errors_proof_l195_19588

noncomputable def students (x y0 y1 y2 y3 y4 y5 : ℕ): ℕ :=
  x + y5 + y4 + y3 + y2 + y1 + y0

noncomputable def errors (x y1 y2 y3 y4 y5 : ℕ): ℕ :=
  6 * x + 5 * y5 + 4 * y4 + 3 * y3 + 2 * y2 + y1

theorem students_errors_proof
  (x y0 y1 y2 y3 y4 y5 : ℕ)
  (h1 : students x y0 y1 y2 y3 y4 y5 = 333)
  (h2 : errors x y1 y2 y3 y4 y5 ≤ 1000) :
  x ≤ y3 + y2 + y1 + y0 :=
by
  sorry

end students_errors_proof_l195_19588


namespace plywood_cut_difference_l195_19568

theorem plywood_cut_difference :
  let original_width := 6
  let original_height := 9
  let total_area := original_width * original_height
  let num_pieces := 6
  let area_per_piece := total_area / num_pieces
  -- Let possible perimeters based on given conditions
  let max_perimeter := 20
  let min_perimeter := 15
  max_perimeter - min_perimeter = 5 :=
by
  sorry

end plywood_cut_difference_l195_19568


namespace show_revenue_l195_19519

theorem show_revenue (tickets_first_showing : ℕ) 
                     (tickets_second_showing : ℕ) 
                     (ticket_price : ℕ) :
                      tickets_first_showing = 200 →
                      tickets_second_showing = 3 * tickets_first_showing →
                      ticket_price = 25 →
                      (tickets_first_showing + tickets_second_showing) * ticket_price = 20000 :=
by
  intros h1 h2 h3
  have h4 : tickets_first_showing + tickets_second_showing = 800 := sorry -- Calculation step
  have h5 : (tickets_first_showing + tickets_second_showing) * ticket_price = 20000 := sorry -- Calculation step
  exact h5

end show_revenue_l195_19519


namespace principal_sum_l195_19560

/-!
# Problem Statement
Given:
1. The difference between compound interest (CI) and simple interest (SI) on a sum at 10% per annum for 2 years is 65.
2. The rate of interest \( R \) is 10%.
3. The time \( T \) is 2 years.

We need to prove that the principal sum \( P \) is 6500.
-/

theorem principal_sum (P : ℝ) (R : ℝ) (T : ℕ) (H : (P * (1 + R / 100)^T - P) - (P * R * T / 100) = 65) 
                      (HR : R = 10) (HT : T = 2) : P = 6500 := 
by 
  sorry

end principal_sum_l195_19560


namespace range_of_a_l195_19570

def p (x : ℝ) : Prop := x ≤ 1/2 ∨ x ≥ 1

def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) ≤ 0

def not_q (x a : ℝ) : Prop := x < a ∨ x > a + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, not_q x a → p x) ∧ (∃ x : ℝ, ¬ (p x → not_q x a)) →
  0 ≤ a ∧ a ≤ 1/2 :=
by
  sorry

end range_of_a_l195_19570


namespace x_divisible_by_5_l195_19564

theorem x_divisible_by_5 (x y : ℕ) (hx : x > 1) (h : 2 * x^2 - 1 = y^15) : 5 ∣ x := 
sorry

end x_divisible_by_5_l195_19564


namespace lcm_of_1_to_12_l195_19505

noncomputable def lcm_1_to_12 : ℕ := 2^3 * 3^2 * 5 * 7 * 11

theorem lcm_of_1_to_12 : lcm_1_to_12 = 27720 := by
  sorry

end lcm_of_1_to_12_l195_19505


namespace range_contains_pi_div_4_l195_19595

noncomputable def f (x : ℝ) : ℝ :=
  Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_contains_pi_div_4 : ∃ x : ℝ, f x = (Real.pi / 4) := by
  sorry

end range_contains_pi_div_4_l195_19595


namespace second_point_x_coord_l195_19551

open Function

variable (n : ℝ)

def line_eq (y : ℝ) : ℝ := 2 * y + 5

theorem second_point_x_coord (h₁ : ∀ (x y : ℝ), x = line_eq y → True) :
  ∃ m : ℝ, ∀ n : ℝ, m = 2 * n + 5 → (m + 1 = line_eq (n + 0.5)) :=
by
  sorry

end second_point_x_coord_l195_19551


namespace man_son_work_together_l195_19501

theorem man_son_work_together (man_days : ℝ) (son_days : ℝ) (combined_days : ℝ) :
  man_days = 4 → son_days = 12 → (1 / man_days + 1 / son_days) = 1 / combined_days → combined_days = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  sorry

end man_son_work_together_l195_19501


namespace cameron_speed_ratio_l195_19515

variables (C Ch : ℝ)
-- Danielle's speed is three times Cameron's speed
def Danielle_speed := 3 * C
-- Danielle's travel time from Granville to Salisbury is 30 minutes
def Danielle_time := 30
-- Chase's travel time from Granville to Salisbury is 180 minutes
def Chase_time := 180

-- Prove the ratio of Cameron's speed to Chase's speed is 2
theorem cameron_speed_ratio :
  (Danielle_speed C / Ch) = (Chase_time / Danielle_time) → (C / Ch) = 2 :=
by {
  sorry
}

end cameron_speed_ratio_l195_19515


namespace parabola_with_given_focus_l195_19531

-- Defining the given condition of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := (x^2) / 4 - (y^2) / 5 = 1

-- Defining the focus coordinates
def focus_coords : ℝ × ℝ := (-3, 0)

-- Proving that the standard equation of the parabola with the left focus of the hyperbola as its focus is y^2 = -12x
theorem parabola_with_given_focus :
  ∃ p : ℝ, (∃ focus : ℝ × ℝ, focus = focus_coords) → 
  ∀ y x : ℝ, y^2 = 4 * p * x → y^2 = -12 * x :=
by
  -- placeholder for proof
  sorry

end parabola_with_given_focus_l195_19531


namespace solve_for_a_l195_19569

noncomputable def a := 3.6

theorem solve_for_a (h : 4 * ((a * 0.48 * 2.5) / (0.12 * 0.09 * 0.5)) = 3200.0000000000005) : 
    a = 3.6 :=
by
  sorry

end solve_for_a_l195_19569


namespace g_minus3_is_correct_l195_19555

theorem g_minus3_is_correct (g : ℚ → ℚ) (h : ∀ x : ℚ, x ≠ 0 → 4 * g (1 / x) + 3 * g x / x = 3 * x^2) : 
  g (-3) = 247 / 39 :=
by
  sorry

end g_minus3_is_correct_l195_19555


namespace zach_saved_money_l195_19596

-- Definitions of known quantities
def cost_of_bike : ℝ := 100
def weekly_allowance : ℝ := 5
def mowing_earnings : ℝ := 10
def babysitting_rate : ℝ := 7
def babysitting_hours : ℝ := 2
def additional_earnings_needed : ℝ := 6

-- Calculate total earnings for this week
def total_earnings_this_week : ℝ := weekly_allowance + mowing_earnings + (babysitting_rate * babysitting_hours)

-- Prove that Zach has already saved $65
theorem zach_saved_money : (cost_of_bike - total_earnings_this_week - additional_earnings_needed) = 65 :=
by
  -- Sorry used as placeholder to skip the proof
  sorry

end zach_saved_money_l195_19596


namespace Mark_bill_total_l195_19581

theorem Mark_bill_total
  (original_bill : ℝ)
  (first_late_charge_rate : ℝ)
  (second_late_charge_rate : ℝ)
  (after_first_late_charge : ℝ)
  (final_total : ℝ) :
  original_bill = 500 ∧
  first_late_charge_rate = 0.02 ∧
  second_late_charge_rate = 0.02 ∧
  after_first_late_charge = original_bill * (1 + first_late_charge_rate) ∧
  final_total = after_first_late_charge * (1 + second_late_charge_rate) →
  final_total = 520.20 := by
  sorry

end Mark_bill_total_l195_19581


namespace extreme_points_l195_19556

noncomputable def f (x : ℝ) : ℝ :=
  x + 4 / x

theorem extreme_points (P : ℝ × ℝ) :
  (P = (2, f 2) ∨ P = (-2, f (-2))) ↔ 
  ∃ x : ℝ, x ≠ 0 ∧ (P = (x, f x)) ∧ 
    (∀ ε > 0, f (x - ε) < f x ∧ f x > f (x + ε) ∨ f (x - ε) > f x ∧ f x < f (x + ε)) := 
sorry

end extreme_points_l195_19556


namespace product_of_values_of_x_l195_19508

theorem product_of_values_of_x : 
  (∃ x : ℝ, |x^2 - 7| - 3 = -1) → 
  (∀ x1 x2 x3 x4 : ℝ, 
    (|x1^2 - 7| - 3 = -1) ∧
    (|x2^2 - 7| - 3 = -1) ∧
    (|x3^2 - 7| - 3 = -1) ∧
    (|x4^2 - 7| - 3 = -1) 
    → x1 * x2 * x3 * x4 = 45) :=
sorry

end product_of_values_of_x_l195_19508


namespace midpoint_on_hyperbola_l195_19502

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l195_19502


namespace molecular_weight_Dinitrogen_pentoxide_l195_19527

theorem molecular_weight_Dinitrogen_pentoxide :
  let atomic_weight_N := 14.01
  let atomic_weight_O := 16.00
  let molecular_formula := (2 * atomic_weight_N) + (5 * atomic_weight_O)
  molecular_formula = 108.02 :=
by
  sorry

end molecular_weight_Dinitrogen_pentoxide_l195_19527


namespace basic_astrophysics_degrees_l195_19577

def percentages : List ℚ := [12, 22, 14, 27, 7, 5, 3, 4]

def total_budget_percentage : ℚ := 100

def degrees_in_circle : ℚ := 360

def remaining_percentage (lst : List ℚ) (total : ℚ) : ℚ :=
  total - lst.sum / 100  -- convert sum to percentage

def degrees_of_percentage (percent : ℚ) (circle_degrees : ℚ) : ℚ :=
  percent * (circle_degrees / total_budget_percentage) -- conversion rate per percentage point

theorem basic_astrophysics_degrees :
  degrees_of_percentage (remaining_percentage percentages total_budget_percentage) degrees_in_circle = 21.6 :=
by
  sorry

end basic_astrophysics_degrees_l195_19577


namespace geometric_sequence_a6_l195_19538

theorem geometric_sequence_a6 (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 + a 3 = 5 / 2) (h2 : a 2 + a 4 = 5 / 4) 
  (h3 : ∀ n, a (n + 1) = a n * q) : a 6 = 1 / 16 :=
by
  sorry

end geometric_sequence_a6_l195_19538


namespace find_slope_of_l_l195_19503

noncomputable def parabola (x y : ℝ) := y ^ 2 = 4 * x

-- Definition of the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Definition of the point M
def M : ℝ × ℝ := (-1, 2)

-- Check if two vectors are perpendicular
def is_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Proof problem statement
theorem find_slope_of_l (x1 x2 y1 y2 k : ℝ)
  (h1 : parabola x1 y1)
  (h2 : parabola x2 y2)
  (h3 : is_perpendicular (x1 + 1, y1 - 2) (x2 + 1, y2 - 2))
  (eq1 : y1 = k * (x1 - 1))
  (eq2 : y2 = k * (x2 - 1)) :
  k = 1 := by
  sorry

end find_slope_of_l_l195_19503


namespace smallest_digit_to_correct_sum_l195_19507

theorem smallest_digit_to_correct_sum :
  ∃ (d : ℕ), d = 3 ∧
  (3 ∈ [3, 5, 7]) ∧
  (371 + 569 + 784 + (d*100) = 1824) := sorry

end smallest_digit_to_correct_sum_l195_19507


namespace quadratic_rewrite_ab_l195_19536

theorem quadratic_rewrite_ab : 
  ∃ (a b c : ℤ), (16*(x:ℝ)^2 - 40*x + 24 = (a*x + b)^2 + c) ∧ (a * b = -20) :=
by {
  sorry
}

end quadratic_rewrite_ab_l195_19536


namespace problem_statement_l195_19576

noncomputable def α : ℝ := 3 + Real.sqrt 8
noncomputable def β : ℝ := 3 - Real.sqrt 8
noncomputable def x : ℝ := α ^ 500
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem problem_statement : x * (1 - f) = 1 := 
by
  sorry

end problem_statement_l195_19576


namespace Ivan_increases_share_more_than_six_times_l195_19589

theorem Ivan_increases_share_more_than_six_times
  (p v s i : ℝ)
  (hp : p / (v + s + i) = 3 / 7)
  (hv : v / (p + s + i) = 1 / 3)
  (hs : s / (p + v + i) = 1 / 3) :
  ∃ k : ℝ, k > 6 ∧ i * k > 0.6 * (p + v + s + i * k) :=
by
  sorry

end Ivan_increases_share_more_than_six_times_l195_19589


namespace part1_part2_l195_19525

namespace Proof

def A (a b : ℝ) : ℝ := 3 * a ^ 2 - 4 * a * b
def B (a b : ℝ) : ℝ := a ^ 2 + 2 * a * b

theorem part1 (a b : ℝ) : 2 * A a b - 3 * B a b = 3 * a ^ 2 - 14 * a * b := by
  sorry
  
theorem part2 (a b : ℝ) (h : |3 * a + 1| + (2 - 3 * b) ^ 2 = 0) : A a b - 2 * B a b = 5 / 3 := by
  have ha : a = -1 / 3 := by
    sorry
  have hb : b = 2 / 3 := by
    sorry
  rw [ha, hb]
  sorry

end Proof

end part1_part2_l195_19525


namespace factorization_check_l195_19529

theorem factorization_check 
  (A : 4 - x^2 + 3 * x ≠ (2 - x) * (2 + x) + 3)
  (B : -x^2 + 3 * x + 4 ≠ -(x + 4) * (x - 1))
  (D : x^2 * y - x * y + x^3 * y ≠ x * (x * y - y + x^2 * y)) :
  1 - 2 * x + x^2 = (1 - x) ^ 2 :=
by
  sorry

end factorization_check_l195_19529


namespace nancy_weight_l195_19523

theorem nancy_weight (w : ℕ) (h : (60 * w) / 100 = 54) : w = 90 :=
by
  sorry

end nancy_weight_l195_19523


namespace initial_meals_is_70_l195_19566

-- Define variables and conditions
variables (A : ℕ)
def initial_meals_for_adults := A

-- Given conditions
def condition_1 := true  -- Group of 55 adults and some children (not directly used in proving A)
def condition_2 := true  -- Either a certain number of adults or 90 children (implicitly used in equation)
def condition_3 := (A - 21) * (90 / A) = 63  -- 21 adults have their meal, remaining food serves 63 children

-- The proof statement
theorem initial_meals_is_70 (h : (A - 21) * (90 / A) = 63) : A = 70 :=
sorry

end initial_meals_is_70_l195_19566


namespace solve_inequality_l195_19546

theorem solve_inequality (x : ℝ) : |x - 2| > 2 - x ↔ x > 2 :=
sorry

end solve_inequality_l195_19546


namespace find_number_is_9_l195_19575

noncomputable def number (y : ℕ) : ℕ := 3^(12 / y)

theorem find_number_is_9 (y : ℕ) (h_y : y = 6) (h_eq : (number y)^y = 3^12) : number y = 9 :=
by
  sorry

end find_number_is_9_l195_19575


namespace simplify_fraction_l195_19561

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := 
by sorry

end simplify_fraction_l195_19561


namespace point_M_quadrant_l195_19528

theorem point_M_quadrant (θ : ℝ) (h1 : π / 2 < θ) (h2 : θ < π) :
  (0 < Real.sin θ) ∧ (Real.cos θ < 0) :=
by
  sorry

end point_M_quadrant_l195_19528


namespace min_coins_for_any_amount_below_dollar_l195_19517

-- Definitions of coin values
def penny := 1
def nickel := 5
def dime := 10
def half_dollar := 50

-- Statement: The minimum number of coins required to pay any amount less than a dollar
theorem min_coins_for_any_amount_below_dollar :
  ∃ (n : ℕ), n = 11 ∧
  (∀ (amount : ℕ), 1 ≤ amount ∧ amount < 100 →
   ∃ (a b c d : ℕ), amount = a * penny + b * nickel + c * dime + d * half_dollar ∧ 
   a + b + c + d ≤ n) :=
sorry

end min_coins_for_any_amount_below_dollar_l195_19517


namespace veranda_width_l195_19533

def room_length : ℕ := 17
def room_width : ℕ := 12
def veranda_area : ℤ := 132

theorem veranda_width :
  ∃ (w : ℝ), (17 + 2 * w) * (12 + 2 * w) - 17 * 12 = 132 ∧ w = 2 :=
by
  use 2
  sorry

end veranda_width_l195_19533


namespace find_difference_l195_19511

variables (x y : ℝ)

theorem find_difference (h1 : x * (y + 2) = 100) (h2 : y * (x + 2) = 60) : x - y = 20 :=
sorry

end find_difference_l195_19511


namespace find_a4_plus_b4_l195_19524

theorem find_a4_plus_b4 (a b : ℝ)
  (h1 : (a^2 - b^2)^2 = 100)
  (h2 : a^3 * b^3 = 512) :
  a^4 + b^4 = 228 :=
by
  sorry

end find_a4_plus_b4_l195_19524


namespace f_zero_f_odd_f_range_l195_19571

noncomputable def f : ℝ → ℝ := sorry

-- Add the hypothesis for the conditions
axiom f_domain : ∀ x : ℝ, ∃ y : ℝ, f y = f x
axiom f_additive : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_value_one_third : f (1 / 3) = 1
axiom f_positive : ∀ x : ℝ, x > 0 → f x > 0

-- (1) Prove that f(0) = 0
theorem f_zero : f 0 = 0 := sorry

-- (2) Prove that f(x) is odd
theorem f_odd (x : ℝ) : f (-x) = -f x := sorry

-- (3) Given f(x) + f(2 + x) < 2, find the range of x
theorem f_range (x : ℝ) : f x + f (2 + x) < 2 → x < -2 / 3 := sorry

end f_zero_f_odd_f_range_l195_19571


namespace grid_divisible_by_rectangles_l195_19509

theorem grid_divisible_by_rectangles (n : ℕ) :
  (∃ m : ℕ, n * n = 7 * m) ↔ (∃ k : ℕ, n = 7 * k ∧ k > 1) :=
by
  sorry

end grid_divisible_by_rectangles_l195_19509


namespace find_original_price_of_petrol_l195_19521

open Real

noncomputable def original_price_of_petrol (P : ℝ) : Prop :=
  ∀ G : ℝ, 
  (G * P = 300) ∧ 
  ((G + 7) * 0.85 * P = 300) → 
  P = 7.56

-- Theorems should ideally be defined within certain scopes or namespaces
theorem find_original_price_of_petrol (P : ℝ) : original_price_of_petrol P :=
  sorry

end find_original_price_of_petrol_l195_19521


namespace max_value_of_seq_l195_19562

theorem max_value_of_seq (S : ℕ → ℤ) (a : ℕ → ℤ) (hS : ∀ n, S n = -n^2 + 6 * n + 7)
  (h_a_def : ∀ n, a n = S n - S (n - 1)) : ∃ max_val, max_val = 12 ∧ ∀ n, a n ≤ max_val :=
by
  sorry

end max_value_of_seq_l195_19562


namespace infinite_product_equals_nine_l195_19540

noncomputable def infinite_product : ℝ :=
  ∏' n : ℕ, ite (n = 0) 1 (3^(n * (1 / 2^n)))

theorem infinite_product_equals_nine : infinite_product = 9 := sorry

end infinite_product_equals_nine_l195_19540


namespace base5_minus_base8_to_base10_l195_19585

def base5_to_base10 (n : Nat) : Nat :=
  5 * 5^5 + 4 * 5^4 + 3 * 5^3 + 2 * 5^2 + 1 * 5^1 + 0 * 5^0

def base8_to_base10 (n : Nat) : Nat :=
  4 * 8^4 + 3 * 8^3 + 2 * 8^2 + 1 * 8^1 + 0 * 8^0

theorem base5_minus_base8_to_base10 :
  (base5_to_base10 543210 - base8_to_base10 43210) = 499 :=
by
  sorry

end base5_minus_base8_to_base10_l195_19585


namespace problem_l195_19537

theorem problem (x y : ℝ) : 
  2 * x + y = 11 → x + 2 * y = 13 → 10 * x^2 - 6 * x * y + y^2 = 530 :=
by
  sorry

end problem_l195_19537


namespace find_quantities_l195_19532

variables {a b x y : ℝ}

-- Original total expenditure condition
axiom h1 : a * x + b * y = 1500

-- New prices and quantities for the first scenario
axiom h2 : (a + 1.5) * (x - 10) + (b + 1) * y = 1529

-- New prices and quantities for the second scenario
axiom h3 : (a + 1) * (x - 5) + (b + 1) * y = 1563.5

-- Inequality constraint
axiom h4 : 205 < 2 * x + y ∧ 2 * x + y < 210

-- Range for 'a'
axiom h5 : 17.5 < a ∧ a < 18.5

-- Proving x and y are specific values.
theorem find_quantities :
  x = 76 ∧ y = 55 :=
sorry

end find_quantities_l195_19532


namespace percentage_import_tax_l195_19565

theorem percentage_import_tax (total_value import_paid excess_amount taxable_amount : ℝ) 
  (h1 : total_value = 2570) 
  (h2 : import_paid = 109.90) 
  (h3 : excess_amount = 1000) 
  (h4 : taxable_amount = total_value - excess_amount) : 
  taxable_amount = 1570 →
  (import_paid / taxable_amount) * 100 = 7 := 
by
  intros h_taxable_amount
  simp [h1, h2, h3, h4, h_taxable_amount]
  sorry -- Proof goes here

end percentage_import_tax_l195_19565
