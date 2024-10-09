import Mathlib

namespace factor_x4_plus_16_l683_68320

theorem factor_x4_plus_16 (x : ℝ) : x^4 + 16 = (x^2 + 2*x + 2) * (x^2 - 2*x + 2) := by
  sorry

end factor_x4_plus_16_l683_68320


namespace find_ratio_eq_eighty_six_l683_68350

-- Define the set S
def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 45}

-- Define the sum of the first n natural numbers function
def sum_n_nat (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define our specific scenario setup
def selected_numbers (x y : ℕ) : Prop :=
  x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ x * y = sum_n_nat 45 - (x + y)

-- Prove the resulting ratio condition
theorem find_ratio_eq_eighty_six (x y : ℕ) (h : selected_numbers x y) : 
  x < y → y / x = 86 :=
by
  sorry

end find_ratio_eq_eighty_six_l683_68350


namespace minimum_value_of_f_l683_68374

noncomputable def f (x : ℝ) : ℝ := 4 * x + 9 / x

theorem minimum_value_of_f : 
  (∀ (x : ℝ), x > 0 → f x ≥ 12) ∧ (∃ (x : ℝ), x > 0 ∧ f x = 12) :=
by {
  sorry
}

end minimum_value_of_f_l683_68374


namespace systems_solution_l683_68382

    theorem systems_solution : 
      (∃ x y : ℝ, 2 * x + 5 * y = -26 ∧ 3 * x - 5 * y = 36 ∧ 
                 (∃ a b : ℝ, a * x - b * y = -4 ∧ b * x + a * y = -8 ∧ 
                 (2 * a + b) ^ 2020 = 1)) := 
    by
      sorry
    
end systems_solution_l683_68382


namespace track_length_is_450_l683_68381

theorem track_length_is_450 (x : ℝ) (d₁ : ℝ) (d₂ : ℝ)
  (h₁ : d₁ = 150)
  (h₂ : x - d₁ = 120)
  (h₃ : d₂ = 200)
  (h₄ : ∀ (d₁ d₂ : ℝ) (t₁ t₂ : ℝ), t₁ / t₂ = d₁ / d₂)
  : x = 450 := by
  sorry

end track_length_is_450_l683_68381


namespace find_total_price_l683_68338

-- Define the cost parameters
variables (sugar_price salt_price : ℝ)

-- Define the given conditions
def condition_1 : Prop := 2 * sugar_price + 5 * salt_price = 5.50
def condition_2 : Prop := sugar_price = 1.50

-- Theorem to be proven
theorem find_total_price (h1 : condition_1 sugar_price salt_price) (h2 : condition_2 sugar_price) : 
  3 * sugar_price + 1 * salt_price = 5.00 :=
by
  sorry

end find_total_price_l683_68338


namespace Cary_walked_miles_round_trip_l683_68354

theorem Cary_walked_miles_round_trip : ∀ (m : ℕ), 
  150 * m - 200 = 250 → m = 3 := 
by
  intros m h
  sorry

end Cary_walked_miles_round_trip_l683_68354


namespace pyramid_volume_l683_68346

theorem pyramid_volume (S A : ℝ)
  (h_surface : 3 * S = 432)
  (h_half_triangular : A = 0.5 * S) :
  (1 / 3) * S * (12 * Real.sqrt 3) = 288 * Real.sqrt 3 :=
by
  sorry

end pyramid_volume_l683_68346


namespace propositions_using_logical_connectives_l683_68355

-- Define each of the propositions.
def prop1 := "October 1, 2004, is the National Day and also the Mid-Autumn Festival."
def prop2 := "Multiples of 10 are definitely multiples of 5."
def prop3 := "A trapezoid is not a rectangle."
def prop4 := "The solutions to the equation x^2 = 1 are x = ± 1."

-- Define logical connectives usage.
def uses_and (s : String) : Prop := 
  s = "October 1, 2004, is the National Day and also the Mid-Autumn Festival."
def uses_not (s : String) : Prop := 
  s = "A trapezoid is not a rectangle."
def uses_or (s : String) : Prop := 
  s = "The solutions to the equation x^2 = 1 are x = ± 1."

-- The lean theorem stating the propositions that use logical connectives
theorem propositions_using_logical_connectives :
  (uses_and prop1) ∧ (¬ uses_and prop2) ∧ (uses_not prop3) ∧ (uses_or prop4) := 
by
  sorry

end propositions_using_logical_connectives_l683_68355


namespace wire_divided_into_quarters_l683_68356

theorem wire_divided_into_quarters
  (l : ℕ) -- length of the wire
  (parts : ℕ) -- number of parts the wire is divided into
  (h_l : l = 28) -- wire is 28 cm long
  (h_parts : parts = 4) -- wire is divided into 4 parts
  : l / parts = 7 := -- each part is 7 cm long
by
  -- use sorry to skip the proof
  sorry

end wire_divided_into_quarters_l683_68356


namespace contrapositive_l683_68313

variables (p q : Prop)

theorem contrapositive (hpq : p → q) : ¬ q → ¬ p :=
by sorry

end contrapositive_l683_68313


namespace largest_2_digit_number_l683_68362

theorem largest_2_digit_number:
  ∃ (N: ℕ), N >= 10 ∧ N < 100 ∧ N % 4 = 0 ∧ (∀ k: ℕ, k ≥ 1 → (N^k) % 100 = N % 100) ∧ 
  (∀ M: ℕ, M >= 10 → M < 100 → M % 4 = 0 → (∀ k: ℕ, k ≥ 1 → (M^k) % 100 = M % 100) → N ≥ M) :=
sorry

end largest_2_digit_number_l683_68362


namespace rectangle_sides_l683_68311

theorem rectangle_sides :
  ∀ (x : ℝ), 
    (3 * x = 8) ∧ (8 / 3 * 3 = 8) →
    ((2 * (3 * x + x) = 3 * x^2) ∧ (2 * (3 * (8 / 3) + (8 / 3)) = 3 * (8 / 3)^2) →
    x = 8 / 3
      ∧ 3 * x = 8) := 
by
  sorry

end rectangle_sides_l683_68311


namespace negation_of_universal_proposition_l683_68351

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, |x| + x^2 ≥ 0)) ↔ (∃ x : ℝ, |x| + x^2 < 0) :=
by
  sorry

end negation_of_universal_proposition_l683_68351


namespace intersection_points_count_l683_68353

theorem intersection_points_count
  : (∀ n : ℤ, ∃ (x y : ℝ), (x - ⌊x⌋) ^ 2 + y ^ 2 = 2 * (x - ⌊x⌋) ∨ y = 1 / 3 * x) →
    (∃ count : ℕ, count = 12) :=
by
  sorry

end intersection_points_count_l683_68353


namespace interval_of_decrease_l683_68361

def quadratic (x : ℝ) := 3 * x^2 - 7 * x + 2

def decreasing_interval (y : ℝ) := y < 2 / 3

theorem interval_of_decrease :
  {x : ℝ | x < (1 / 3)} = {x : ℝ | x < (1 / 3)} :=
by sorry

end interval_of_decrease_l683_68361


namespace sarahs_team_mean_score_l683_68312

def mean_score_of_games (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

theorem sarahs_team_mean_score :
  mean_score_of_games [69, 68, 70, 61, 74, 62, 65, 74] = 67.875 :=
by
  sorry

end sarahs_team_mean_score_l683_68312


namespace sum_of_ages_l683_68387

def Maria_age (E : ℕ) : ℕ := E + 7

theorem sum_of_ages (M E : ℕ) (h1 : M = E + 7) (h2 : M + 10 = 3 * (E - 5)) :
  M + E = 39 :=
by
  sorry

end sum_of_ages_l683_68387


namespace range_of_a_if_distinct_zeros_l683_68393

theorem range_of_a_if_distinct_zeros (a : ℝ) :
(∀ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁ ∧ (x₁^3 - 3*x₁ + a = 0) ∧ (x₂^3 - 3*x₂ + a = 0) ∧ (x₃^3 - 3*x₃ + a = 0)) → -2 < a ∧ a < 2 :=
by
  sorry

end range_of_a_if_distinct_zeros_l683_68393


namespace numerical_value_expression_l683_68306

theorem numerical_value_expression (x y z : ℚ) (h1 : x - 4 * y - 2 * z = 0) (h2 : 3 * x + 2 * y - z = 0) (h3 : z ≠ 0) : 
  (x^2 - 5 * x * y) / (2 * y^2 + z^2) = 164 / 147 :=
by sorry

end numerical_value_expression_l683_68306


namespace relationship_of_y_values_l683_68324

theorem relationship_of_y_values (k : ℝ) (y₁ y₂ y₃ : ℝ) :
  (y₁ = (k^2 + 3) / (-3)) ∧ (y₂ = (k^2 + 3) / (-1)) ∧ (y₃ = (k^2 + 3) / 2) →
  y₂ < y₁ ∧ y₁ < y₃ :=
by
  intro h
  have h₁ : y₁ = (k^2 + 3) / (-3) := h.1
  have h₂ : y₂ = (k^2 + 3) / (-1) := h.2.1
  have h₃ : y₃ = (k^2 + 3) / 2 := h.2.2
  sorry

end relationship_of_y_values_l683_68324


namespace power_function_solution_l683_68301

def power_function_does_not_pass_through_origin (m : ℝ) : Prop :=
  (m^2 - m - 2) ≤ 0

def condition (m : ℝ) : Prop :=
  m^2 - 3 * m + 3 = 1

theorem power_function_solution (m : ℝ) :
  power_function_does_not_pass_through_origin m ∧ condition m → (m = 1 ∨ m = 2) :=
by sorry

end power_function_solution_l683_68301


namespace range_of_a_l683_68337

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ (-2 < a ∧ a ≤ 2) :=
sorry

end range_of_a_l683_68337


namespace find_number_l683_68328

theorem find_number (x n : ℤ) 
  (h1 : 0 < x) (h2 : x < 7) 
  (h3 : x < 15) 
  (h4 : -1 < x) (h5 : x < 5) 
  (h6 : x < 3) (h7 : 0 < x) 
  (h8 : x + n < 4) 
  (hx : x = 1): 
  n < 3 := 
sorry

end find_number_l683_68328


namespace abs_diff_of_slopes_l683_68389

theorem abs_diff_of_slopes (k1 k2 b : ℝ) (h : k1 * k2 < 0) (area_cond : (1 / 2) * 3 * |k1 - k2| * 3 = 9) :
  |k1 - k2| = 2 :=
by
  sorry

end abs_diff_of_slopes_l683_68389


namespace smallest_n_l683_68364

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, b * k = a

def meets_condition (n : ℕ) : Prop :=
  n > 0 ∧
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ n + 1 ∧ is_divisible (n^2 - n + 1) k ∧
  ∃ l : ℕ, 1 ≤ l ∧ l ≤ n + 1 ∧ ¬ is_divisible (n^2 - n + 1) l

theorem smallest_n : ∃ n : ℕ, meets_condition n ∧ n = 5 :=
by
  sorry

end smallest_n_l683_68364


namespace xy_equals_nine_l683_68373

theorem xy_equals_nine (x y : ℝ) (h : x * (x + 2 * y) = x ^ 2 + 18) : x * y = 9 :=
by
  sorry

end xy_equals_nine_l683_68373


namespace width_of_room_l683_68331

noncomputable def roomWidth (length : ℝ) (totalCost : ℝ) (costPerSquareMeter : ℝ) : ℝ :=
  let area := totalCost / costPerSquareMeter
  area / length

theorem width_of_room :
  roomWidth 5.5 24750 1200 = 3.75 :=
by
  sorry

end width_of_room_l683_68331


namespace hyperbola_eccentricity_l683_68309

-- Definitions translated from conditions
noncomputable def parabola_focus : ℝ × ℝ := (0, -Real.sqrt 5)
noncomputable def a : ℝ := 2
noncomputable def c : ℝ := Real.sqrt 5

-- Eccentricity formula for the hyperbola
noncomputable def eccentricity (c a : ℝ) : ℝ := c / a

-- Statement to be proved
theorem hyperbola_eccentricity :
  eccentricity c a = Real.sqrt 5 / 2 :=
by
  sorry

end hyperbola_eccentricity_l683_68309


namespace quadratic_roots_range_no_real_k_for_reciprocal_l683_68359

theorem quadratic_roots_range (k : ℝ) (h : 12 * k + 4 > 0) : k > -1 / 3 ∧ k ≠ 0 :=
by
  sorry

theorem no_real_k_for_reciprocal (k : ℝ) : ¬∃ (x1 x2 : ℝ), (kx^2 - 2*(k+1)*x + k-1 = 0) ∧ (1/x1 + 1/x2 = 0) :=
by
  sorry

end quadratic_roots_range_no_real_k_for_reciprocal_l683_68359


namespace two_pipes_fill_time_l683_68340

theorem two_pipes_fill_time (R : ℝ) (h1 : (3 : ℝ) * R * (8 : ℝ) = 1) : (2 : ℝ) * R * (12 : ℝ) = 1 :=
by 
  have hR : R = 1 / 24 := by linarith
  rw [hR]
  sorry

end two_pipes_fill_time_l683_68340


namespace smallest_z_value_l683_68376

theorem smallest_z_value :
  ∃ (x z : ℕ), (w = x - 2) ∧ (y = x + 2) ∧ (z = x + 4) ∧ ((x - 2)^3 + x^3 + (x + 2)^3 = (x + 4)^3) ∧ z = 2 := by
  sorry

end smallest_z_value_l683_68376


namespace contradiction_method_l683_68314

variable (a b : ℝ)

theorem contradiction_method (h1 : a > b) (h2 : 3 * a ≤ 3 * b) : false :=
by sorry

end contradiction_method_l683_68314


namespace quartic_poly_roots_l683_68321

noncomputable def roots_polynomial : List ℝ := [
  (1 + Real.sqrt 5) / 2,
  (1 - Real.sqrt 5) / 2,
  (3 + Real.sqrt 13) / 6,
  (3 - Real.sqrt 13) / 6
]

theorem quartic_poly_roots :
  ∀ x : ℝ, x ∈ roots_polynomial ↔ 3*x^4 - 4*x^3 - 5*x^2 - 4*x + 3 = 0 :=
by sorry

end quartic_poly_roots_l683_68321


namespace balls_sum_l683_68395

theorem balls_sum (m n : ℕ) (h₁ : ∀ a, a ∈ ({m, 8, n} : Finset ℕ)) -- condition: balls are identical except for color
  (h₂ : (8 : ℝ) / (m + 8 + n) = (m + n : ℝ) / (m + 8 + n)) : m + n = 8 :=
sorry

end balls_sum_l683_68395


namespace find_principal_l683_68360

def r : ℝ := 0.03
def t : ℝ := 3
def I (P : ℝ) : ℝ := P - 1820
def simple_interest (P : ℝ) : ℝ := P * r * t

theorem find_principal (P : ℝ) : simple_interest P = I P -> P = 2000 :=
by
  sorry

end find_principal_l683_68360


namespace trevor_eggs_left_l683_68357

def gertrude_eggs : Nat := 4
def blanche_eggs : Nat := 3
def nancy_eggs : Nat := 2
def martha_eggs : Nat := 2
def dropped_eggs : Nat := 2

theorem trevor_eggs_left : 
  (gertrude_eggs + blanche_eggs + nancy_eggs + martha_eggs - dropped_eggs) = 9 := 
  by sorry

end trevor_eggs_left_l683_68357


namespace find_x_values_l683_68348

theorem find_x_values (x : ℝ) :
  (2 / (x + 2) + 8 / (x + 4) ≥ 2) ↔ (x ∈ Set.Ici 2 ∨ x ∈ Set.Iic (-4)) := by
sorry

end find_x_values_l683_68348


namespace total_charts_16_l683_68377

def total_charts_brought (number_of_associate_professors : Int) (number_of_assistant_professors : Int) : Int :=
  number_of_associate_professors * 1 + number_of_assistant_professors * 2

theorem total_charts_16 (A B : Int)
  (h1 : 2 * A + B = 11)
  (h2 : A + B = 9) :
  total_charts_brought A B = 16 :=
by {
  -- the proof will go here
  sorry
}

end total_charts_16_l683_68377


namespace discount_equation_l683_68383

variable (P₀ P_f x : ℝ)
variable (h₀ : P₀ = 200)
variable (h₁ : P_f = 164)

theorem discount_equation :
  P₀ * (1 - x)^2 = P_f := by
  sorry

end discount_equation_l683_68383


namespace inequality_always_true_l683_68305

theorem inequality_always_true (x : ℝ) : (4 * x) / (x ^ 2 + 4) ≤ 1 := by
  sorry

end inequality_always_true_l683_68305


namespace minimum_value_of_f_l683_68300

noncomputable def f (x : ℝ) : ℝ := |x - 2| + |3 - x|

theorem minimum_value_of_f : ∀ x : ℝ, f x ≥ 1 ∧ (∃ x₀ : ℝ, f x₀ = 1) := by
  sorry

end minimum_value_of_f_l683_68300


namespace pan_dimensions_l683_68319

theorem pan_dimensions (m n : ℕ) : 
  (∃ m n, m * n = 48 ∧ (m-2) * (n-2) = 2 * (2*m + 2*n - 4) ∧ m > 2 ∧ n > 2) → 
  (m = 4 ∧ n = 12) ∨ (m = 12 ∧ n = 4) ∨ (m = 6 ∧ n = 8) ∨ (m = 8 ∧ n = 6) :=
by
  sorry

end pan_dimensions_l683_68319


namespace max_ratio_BO_BM_l683_68399

theorem max_ratio_BO_BM
  (C : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hC : C = (0, -4))
  (hCir : ∃ (P : ℝ × ℝ), (P.1 - 2)^2 + (P.2 - 4)^2 = 1 ∧ A = ((P.1 + C.1) / 2, (P.2 + C.2) / 2))
  (hPar : ∃ (x y : ℝ), B = (x, y) ∧ y^2 = 4 * x) :
  ∃ t, t = (4 * Real.sqrt 7)/7 ∧ t = Real.sqrt ((B.1^2 + 4 * B.1)/((B.1 + 1/2)^2)) := by
  -- Given conditions and definitions
  obtain ⟨P, hP, hA⟩ := hCir
  obtain ⟨x, y, hB⟩ := hPar
  use (4 * Real.sqrt 7) / 7
  sorry

end max_ratio_BO_BM_l683_68399


namespace stock_price_end_of_third_year_l683_68369

def first_year_price (initial_price : ℝ) (first_year_increase : ℝ) : ℝ :=
  initial_price + (initial_price * first_year_increase)

def second_year_price (price_end_first : ℝ) (second_year_decrease : ℝ) : ℝ :=
  price_end_first - (price_end_first * second_year_decrease)

def third_year_price (price_end_second : ℝ) (third_year_increase : ℝ) : ℝ :=
  price_end_second + (price_end_second * third_year_increase)

theorem stock_price_end_of_third_year :
  ∀ (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) (third_year_increase : ℝ),
    initial_price = 150 →
    first_year_increase = 0.5 →
    second_year_decrease = 0.3 →
    third_year_increase = 0.2 →
    third_year_price (second_year_price (first_year_price initial_price first_year_increase) second_year_decrease) third_year_increase = 189 :=
by
  intros initial_price first_year_increase second_year_decrease third_year_increase
  sorry

end stock_price_end_of_third_year_l683_68369


namespace new_ratio_books_clothes_l683_68326

theorem new_ratio_books_clothes :
  ∀ (B C E : ℝ), (B = 22.5) → (C = 18) → (E = 9) → (C_new = C - 9) → C_new = 9 → B / C_new = 2.5 :=
by
  intros B C E HB HC HE HCnew Hnew
  sorry

end new_ratio_books_clothes_l683_68326


namespace vector_dot_product_l683_68358

theorem vector_dot_product
  (AB : ℝ × ℝ) (BC : ℝ × ℝ)
  (t : ℝ)
  (hAB : AB = (2, 3))
  (hBC : BC = (3, t))
  (ht : t > 0)
  (hmagnitude : (3^2 + t^2).sqrt = (10:ℝ).sqrt) :
  (AB.1 * (AB.1 + BC.1) + AB.2 * (AB.2 + BC.2) = 22) :=
by
  sorry

end vector_dot_product_l683_68358


namespace max_points_right_triangle_l683_68308

theorem max_points_right_triangle (n : ℕ) :
  (∀ (pts : Fin n → ℝ × ℝ), ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k →
    let p1 := pts i
    let p2 := pts j
    let p3 := pts k
    let a := (p2.1 - p1.1)^2 + (p2.2 - p1.2)^2
    let b := (p3.1 - p2.1)^2 + (p3.2 - p2.2)^2
    let c := (p3.1 - p1.1)^2 + (p3.2 - p1.2)^2
    a + b = c ∨ b + c = a ∨ c + a = b) →
  n ≤ 4 :=
sorry

end max_points_right_triangle_l683_68308


namespace combined_average_age_l683_68349

-- Definitions based on given conditions
def num_fifth_graders : ℕ := 28
def avg_age_fifth_graders : ℝ := 10
def num_parents : ℕ := 45
def avg_age_parents : ℝ := 40

-- The statement to prove
theorem combined_average_age : (num_fifth_graders * avg_age_fifth_graders + num_parents * avg_age_parents) / (num_fifth_graders + num_parents) = 28.49 :=
  by
  sorry

end combined_average_age_l683_68349


namespace find_X_l683_68347

theorem find_X (X : ℕ) (h1 : 2 + 1 + 3 + X = 3 + 4 + 5) : X = 6 :=
by
  sorry

end find_X_l683_68347


namespace digit_sum_is_twelve_l683_68375

theorem digit_sum_is_twelve (n x y : ℕ) (h1 : n = 10 * x + y) (h2 : 0 ≤ x ∧ x ≤ 9) (h3 : 0 ≤ y ∧ y ≤ 9)
  (h4 : (1 / 2 : ℚ) * n = (1 / 4 : ℚ) * n + 3) : x + y = 12 :=
by
  sorry

end digit_sum_is_twelve_l683_68375


namespace product_of_two_numbers_l683_68372

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 21) (h2 : x^2 + y^2 = 527) : x * y = -43 :=
sorry

end product_of_two_numbers_l683_68372


namespace largest_of_four_numbers_l683_68333

theorem largest_of_four_numbers (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  max (max (max (a^2 + b^2) (2 * a * b)) a) (1 / 2) = a^2 + b^2 :=
by
  sorry

end largest_of_four_numbers_l683_68333


namespace complete_square_proof_l683_68394

def quadratic_eq := ∀ (x : ℝ), x^2 - 6 * x + 5 = 0
def form_completing_square (b c : ℝ) := ∀ (x : ℝ), (x + b)^2 = c

theorem complete_square_proof :
  quadratic_eq → (∃ b c : ℤ, form_completing_square (b : ℝ) (c : ℝ) ∧ b + c = 11) :=
by
  sorry

end complete_square_proof_l683_68394


namespace carol_to_cathy_ratio_l683_68329

-- Define the number of cars owned by Cathy, Lindsey, Carol, and Susan
def cathy_cars : ℕ := 5
def lindsey_cars : ℕ := cathy_cars + 4
def carol_cars : ℕ := cathy_cars
def susan_cars : ℕ := carol_cars - 2

-- Define the total number of cars in the problem statement
def total_cars : ℕ := 32

-- Theorem to prove the ratio of Carol's cars to Cathy's cars is 1:1
theorem carol_to_cathy_ratio : carol_cars = cathy_cars := by
  sorry

end carol_to_cathy_ratio_l683_68329


namespace value_to_subtract_l683_68307

variable (x y : ℝ)

theorem value_to_subtract (h1 : (x - 5) / 7 = 7) (h2 : (x - y) / 8 = 6) : y = 6 := by
  sorry

end value_to_subtract_l683_68307


namespace decreased_and_divided_l683_68318

theorem decreased_and_divided (x : ℝ) (h : (x - 5) / 7 = 7) : (x - 14) / 10 = 4 := by
  sorry

end decreased_and_divided_l683_68318


namespace second_train_length_l683_68365

theorem second_train_length
  (L1 : ℝ) (V1 : ℝ) (V2 : ℝ) (T : ℝ)
  (h1 : L1 = 300)
  (h2 : V1 = 72 * 1000 / 3600)
  (h3 : V2 = 36 * 1000 / 3600)
  (h4 : T = 79.99360051195904) :
  L1 + (V1 - V2) * T = 799.9360051195904 :=
by
  sorry

end second_train_length_l683_68365


namespace subtraction_and_multiplication_problem_l683_68345

theorem subtraction_and_multiplication_problem :
  (5 / 6 - 1 / 3) * 3 / 4 = 3 / 8 :=
by sorry

end subtraction_and_multiplication_problem_l683_68345


namespace simplify_and_evaluate_l683_68366

theorem simplify_and_evaluate :
  let x := 2 * Real.sqrt 3
  (x - Real.sqrt 2) * (x + Real.sqrt 2) + x * (x - 1) = 22 - 2 * Real.sqrt 3 := 
by
  let x := 2 * Real.sqrt 3
  sorry

end simplify_and_evaluate_l683_68366


namespace lemonade_sales_l683_68336

theorem lemonade_sales (total_amount small_amount medium_amount large_price sales_price_small sales_price_medium earnings_small earnings_medium : ℕ) (h1 : total_amount = 50) (h2 : sales_price_small = 1) (h3 : sales_price_medium = 2) (h4 : large_price = 3) (h5 : earnings_small = 11) (h6 : earnings_medium = 24) : large_amount = 5 :=
by
  sorry

end lemonade_sales_l683_68336


namespace initial_chocolate_bars_l683_68330

theorem initial_chocolate_bars (B : ℕ) 
  (H1 : Thomas_and_friends_take = B / 4)
  (H2 : One_friend_returns_5 = Thomas_and_friends_take - 5)
  (H3 : Piper_takes = Thomas_and_friends_take - 5 - 5)
  (H4 : Remaining_bars = B - Thomas_and_friends_take - Piper_takes)
  (H5 : Remaining_bars = 110) :
  B = 190 := 
sorry

end initial_chocolate_bars_l683_68330


namespace integer_ratio_condition_l683_68304

variable {x y : ℝ}

theorem integer_ratio_condition (h1 : 1 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 3) (h3 : ∃ t : ℤ, x = t * y) : 
  ∃ t : ℤ, t = -2 := sorry

end integer_ratio_condition_l683_68304


namespace solve_abs_eq_l683_68322

theorem solve_abs_eq (x : ℝ) (h : |x - 3| = |x + 1|) : x = 1 :=
  sorry

end solve_abs_eq_l683_68322


namespace students_chose_water_l683_68371

theorem students_chose_water (total_students : ℕ)
  (h1 : 75 * total_students / 100 = 90)
  (h2 : 25 * total_students / 100 = x) :
  x = 30 := 
sorry

end students_chose_water_l683_68371


namespace caps_difference_l683_68339

theorem caps_difference (Billie_caps Sammy_caps : ℕ) (Janine_caps := 3 * Billie_caps)
  (Billie_has : Billie_caps = 2) (Sammy_has : Sammy_caps = 8) :
  Sammy_caps - Janine_caps = 2 := by
  -- proof goes here
  sorry

end caps_difference_l683_68339


namespace condition_sufficient_but_not_necessary_condition_not_necessary_combined_condition_l683_68388

theorem condition_sufficient_but_not_necessary (x y : ℝ) :
  (x^2 + y^2 + 4*x + 3 ≤ 0) → ((x + 4) * (x + 3) ≥ 0) :=
sorry

theorem condition_not_necessary (x y : ℝ) :
  ((x + 4) * (x + 3) ≥ 0) → ¬ (x^2 + y^2 + 4*x + 3 ≤ 0) :=
sorry

-- Combine both into a single statement using conjunction
theorem combined_condition (x y : ℝ) :
  ((x^2 + y^2 + 4*x + 3 ≤ 0) → ((x + 4) * (x + 3) ≥ 0))
  ∧ ((x + 4) * (x + 3) ≥ 0 → ¬(x^2 + y^2 + 4*x + 3 ≤ 0)) :=
sorry

end condition_sufficient_but_not_necessary_condition_not_necessary_combined_condition_l683_68388


namespace mutually_exclusive_events_l683_68302

-- Definitions based on the given conditions
def sample_inspection (n : ℕ) := n = 10
def event_A (defective_products : ℕ) := defective_products ≥ 2
def event_B (defective_products : ℕ) := defective_products ≤ 1

-- The proof statement
theorem mutually_exclusive_events (n : ℕ) (defective_products : ℕ) 
  (h1 : sample_inspection n) (h2 : event_A defective_products) : 
  event_B defective_products = false :=
by
  sorry

end mutually_exclusive_events_l683_68302


namespace probability_of_shaded_shape_l683_68392

   def total_shapes : ℕ := 4
   def shaded_shapes : ℕ := 1

   theorem probability_of_shaded_shape : shaded_shapes / total_shapes = 1 / 4 := 
   by
     sorry
   
end probability_of_shaded_shape_l683_68392


namespace math_problem_statements_l683_68317

theorem math_problem_statements :
  (∀ a : ℝ, (a = -a) → (a = 0)) ∧
  (∀ b : ℝ, (1 / b = b) ↔ (b = 1 ∨ b = -1)) ∧
  (∀ c : ℝ, (c < -1) → (1 / c > c)) ∧
  (∀ d : ℝ, (d > 1) → (1 / d < d)) ∧
  (∃ n : ℕ, n > 0 ∧ ∀ m : ℕ, m > 0 → n ≤ m) :=
by {
  sorry
}

end math_problem_statements_l683_68317


namespace line_equation_l683_68384

-- Define the point A(2, 1)
def A : ℝ × ℝ := (2, 1)

-- Define the notion of a line with equal intercepts on the coordinates
def line_has_equal_intercepts (c : ℝ) : Prop :=
  ∃ (m b : ℝ), ∀ (x y : ℝ), y = m * x + b ↔ x = y ∧ y = c

-- Define the condition that the line passes through point A
def line_passes_through_A (m b : ℝ) : Prop :=
  A.2 = m * A.1 + b

-- Define the two possible equations for the line
def line_eq1 (x y : ℝ) : Prop :=
  x + y - 3 = 0

def line_eq2 (x y : ℝ) : Prop :=
  2 * x - y = 0

-- Combined conditions in a single theorem
theorem line_equation (m b c x y : ℝ) (h_pass : line_passes_through_A m b) (h_int : line_has_equal_intercepts c) :
  (line_eq1 x y ∨ line_eq2 x y) :=
sorry

end line_equation_l683_68384


namespace rectangle_area_l683_68368

theorem rectangle_area (x y : ℝ) (hx : 3 * y = 7 * x) (hp : 2 * (x + y) = 40) :
  x * y = 84 := by
  sorry

end rectangle_area_l683_68368


namespace maximum_cells_covered_at_least_five_times_l683_68352

theorem maximum_cells_covered_at_least_five_times :
  let areas := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let total_covered := List.sum areas
  let exact_coverage := 1 * 1 + 2 * 2 + 3 * 3 + 4 * 4
  let remaining_coverage := total_covered - exact_coverage
  let max_cells_covered_at_least_five := remaining_coverage / 5
  max_cells_covered_at_least_five = 5 :=
by
  sorry

end maximum_cells_covered_at_least_five_times_l683_68352


namespace length_of_one_side_of_hexagon_l683_68341

variable (P : ℝ) (n : ℕ)
-- Condition: perimeter P is 60 inches
def hexagon_perimeter_condition : Prop := P = 60
-- Hexagon has six sides
def hexagon_sides_condition : Prop := n = 6
-- The question asks for the side length
noncomputable def side_length_of_hexagon : ℝ := P / n

-- Prove that if a hexagon has a perimeter of 60 inches, then its side length is 10 inches
theorem length_of_one_side_of_hexagon (hP : hexagon_perimeter_condition P) (hn : hexagon_sides_condition n) :
  side_length_of_hexagon P n = 10 := by
  sorry

end length_of_one_side_of_hexagon_l683_68341


namespace solve_inequality_l683_68367

theorem solve_inequality (k x : ℝ) :
  (x^2 > (k + 1) * x - k) ↔ 
  (if k > 1 then (x < 1 ∨ x > k)
   else if k = 1 then (x ≠ 1)
   else (x < k ∨ x > 1)) :=
by
  sorry

end solve_inequality_l683_68367


namespace number_equation_l683_68327

variable (x : ℝ)

theorem number_equation :
  5 * x - 2 * x = 10 :=
sorry

end number_equation_l683_68327


namespace problem_l683_68315

theorem problem (a b : ℝ) (h : a > b) : a / 3 > b / 3 :=
sorry

end problem_l683_68315


namespace craig_apples_after_sharing_l683_68398

-- Defining the initial conditions
def initial_apples_craig : ℕ := 20
def shared_apples : ℕ := 7

-- The proof statement
theorem craig_apples_after_sharing : 
  initial_apples_craig - shared_apples = 13 := 
by
  sorry

end craig_apples_after_sharing_l683_68398


namespace calculate_blue_candles_l683_68363

-- Definitions based on identified conditions
def total_candles : Nat := 79
def yellow_candles : Nat := 27
def red_candles : Nat := 14
def blue_candles : Nat := total_candles - (yellow_candles + red_candles)

-- The proof statement
theorem calculate_blue_candles : blue_candles = 38 :=
by
  sorry

end calculate_blue_candles_l683_68363


namespace dan_initial_money_l683_68385

def money_left : ℕ := 3
def cost_candy : ℕ := 2
def initial_money : ℕ := money_left + cost_candy

theorem dan_initial_money :
  initial_money = 5 :=
by
  -- Definitions according to problem
  let money_left := 3
  let cost_candy := 2

  have h : initial_money = money_left + cost_candy := by rfl
  rw [h]

  -- Show the final equivalence
  show 3 + 2 = 5
  rfl

end dan_initial_money_l683_68385


namespace largest_multiple_of_7_neg_greater_than_neg_150_l683_68386

theorem largest_multiple_of_7_neg_greater_than_neg_150 : 
  ∃ (k : ℤ), k % 7 = 0 ∧ -k > -150 ∧ (∀ (m : ℤ), m % 7 = 0 ∧ -m > -150 → k ≥ m) ∧ k = 147 :=
by
  sorry

end largest_multiple_of_7_neg_greater_than_neg_150_l683_68386


namespace binomial_probability_X_eq_3_l683_68323

theorem binomial_probability_X_eq_3 :
  let n := 6
  let p := 1 / 2
  let k := 3
  let binom := Nat.choose n k
  (binom * p ^ k * (1 - p) ^ (n - k)) = 5 / 16 := by 
  sorry

end binomial_probability_X_eq_3_l683_68323


namespace basketball_team_selection_l683_68334

noncomputable def count_ways_excluding_twins (n k : ℕ) : ℕ :=
  let total_ways := Nat.choose n k
  let exhaustive_cases := Nat.choose (n - 2) (k - 2)
  total_ways - exhaustive_cases

theorem basketball_team_selection :
  count_ways_excluding_twins 12 5 = 672 :=
by
  sorry

end basketball_team_selection_l683_68334


namespace total_computers_sold_l683_68396

theorem total_computers_sold (T : ℕ) (h_half_sales_laptops : 2 * T / 2 = T)
        (h_third_sales_netbooks : 3 * T / 3 = T)
        (h_desktop_sales : T - T / 2 - T / 3 = 12) : T = 72 :=
by
  sorry

end total_computers_sold_l683_68396


namespace find_a1_of_geom_series_l683_68342

noncomputable def geom_series_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

theorem find_a1_of_geom_series (a₁ q : ℝ) (S : ℕ → ℝ)
  (h1 : S 6 = 9 * S 3)
  (h2 : S 5 = 62)
  (neq1 : q ≠ 1)
  (neqm1 : q ≠ -1) :
  a₁ = 2 :=
by
  have eq1 : S 6 = geom_series_sum a₁ q 6 := sorry
  have eq2 : S 3 = geom_series_sum a₁ q 3 := sorry
  have eq3 : S 5 = geom_series_sum a₁ q 5 := sorry
  sorry

end find_a1_of_geom_series_l683_68342


namespace proposition_b_proposition_d_l683_68343

-- Proposition B: For a > 0 and b > 0, if ab = 2, then the minimum value of a + 2b is 4
theorem proposition_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 2) : a + 2 * b ≥ 4 :=
  sorry

-- Proposition D: For a > 0 and b > 0, if a² + b² = 1, then the maximum value of a + b is sqrt(2).
theorem proposition_d (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + b^2 = 1) : a + b ≤ Real.sqrt 2 :=
  sorry

end proposition_b_proposition_d_l683_68343


namespace find_m_for_opposite_solutions_l683_68325

theorem find_m_for_opposite_solutions (x y m : ℝ) 
  (h1 : x = -y)
  (h2 : 3 * x + 5 * y = 2)
  (h3 : 2 * x + 7 * y = m - 18) : 
  m = 23 :=
sorry

end find_m_for_opposite_solutions_l683_68325


namespace arnolds_total_protein_l683_68370

theorem arnolds_total_protein (collagen_protein_per_two_scoops : ℕ) (protein_per_scoop : ℕ) 
    (steak_protein : ℕ) (scoops_of_collagen : ℕ) (scoops_of_protein : ℕ) :
    collagen_protein_per_two_scoops = 18 →
    protein_per_scoop = 21 →
    steak_protein = 56 →
    scoops_of_collagen = 1 →
    scoops_of_protein = 1 →
    (collagen_protein_per_two_scoops / 2 * scoops_of_collagen + protein_per_scoop * scoops_of_protein + steak_protein = 86) :=
by
  intros hc p s sc sp
  sorry

end arnolds_total_protein_l683_68370


namespace find_f_8_l683_68332

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f (-x) = f x
axiom periodicity : ∀ x : ℝ, f (x + 6) = f x
axiom function_on_interval : ∀ x : ℝ, -3 < x ∧ x < 0 → f x = 2 * x - 5

theorem find_f_8 : f 8 = -9 :=
by
  sorry

end find_f_8_l683_68332


namespace sum_of_solutions_eq_neg2_l683_68344

noncomputable def sum_of_real_solutions (a : ℝ) (h : a > 2) : ℝ :=
  -2

theorem sum_of_solutions_eq_neg2 (a : ℝ) (h : a > 2) :
  sum_of_real_solutions a h = -2 := sorry

end sum_of_solutions_eq_neg2_l683_68344


namespace kylie_earrings_l683_68390

def number_of_necklaces_monday := 10
def number_of_necklaces_tuesday := 2
def number_of_bracelets_wednesday := 5
def beads_per_necklace := 20
def beads_per_bracelet := 10
def beads_per_earring := 5
def total_beads := 325

theorem kylie_earrings : 
    (total_beads - ((number_of_necklaces_monday + number_of_necklaces_tuesday) * beads_per_necklace + number_of_bracelets_wednesday * beads_per_bracelet)) / beads_per_earring = 7 :=
by
    sorry

end kylie_earrings_l683_68390


namespace find_f_of_1_over_2016_l683_68310

noncomputable def f (x : ℝ) : ℝ := sorry

lemma f_property_0 : f 0 = 0 := sorry
lemma f_property_1 (x : ℝ) : f x + f (1 - x) = 1 := sorry
lemma f_property_2 (x : ℝ) : f (x / 3) = (1 / 2) * f x := sorry
lemma f_property_3 {x₁ x₂ : ℝ} (h₀ : 0 ≤ x₁) (h₁ : x₁ < x₂) (h₂ : x₂ ≤ 1): f x₁ ≤ f x₂ := sorry

theorem find_f_of_1_over_2016 : f (1 / 2016) = 1 / 128 := sorry

end find_f_of_1_over_2016_l683_68310


namespace volume_of_solid_l683_68391

def x_y_relation (x y : ℝ) : Prop := x = (y - 2)^(1/3)
def x1 (x : ℝ) : Prop := x = 1
def y1 (y : ℝ) : Prop := y = 1

theorem volume_of_solid :
  ∀ (x y : ℝ),
    (x_y_relation x y ∧ x1 x ∧ y1 y) →
    ∃ V : ℝ, V = (44 / 7) * Real.pi :=
by
  -- Proof will go here
  sorry

end volume_of_solid_l683_68391


namespace total_distance_l683_68379

theorem total_distance (D : ℝ) 
  (h1 : 1/4 * (3/8 * D) = 210) : D = 840 := 
by
  -- proof steps would go here
  sorry

end total_distance_l683_68379


namespace dozens_in_each_box_l683_68303

theorem dozens_in_each_box (boxes total_mangoes : ℕ) (h1 : boxes = 36) (h2 : total_mangoes = 4320) :
  (total_mangoes / 12) / boxes = 10 :=
by
  -- The proof will go here.
  sorry

end dozens_in_each_box_l683_68303


namespace equation_C_is_symmetric_l683_68380

def symm_y_axis (f : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), f x y ↔ f (-x) y

def equation_A (x y : ℝ) : Prop := x^2 - x + y^2 = 1
def equation_B (x y : ℝ) : Prop := x^2 * y + x * y^2 = 1
def equation_C (x y : ℝ) : Prop := x^2 - y^2 = 1
def equation_D (x y : ℝ) : Prop := x - y = 1

theorem equation_C_is_symmetric : symm_y_axis equation_C :=
by
  sorry

end equation_C_is_symmetric_l683_68380


namespace students_standing_together_l683_68335

theorem students_standing_together (s : Finset ℕ) (h_size : s.card = 6) (a b : ℕ) (h_ab : a ∈ s ∧ b ∈ s) (h_ab_together : ∃ (l : List ℕ), l.length = 6 ∧ a :: b :: l = l):
  ∃ (arrangements : ℕ), arrangements = 240 := by
  sorry

end students_standing_together_l683_68335


namespace determine_some_number_l683_68378

theorem determine_some_number (x : ℝ) (n : ℝ) (hx : x = 1.5) (h : (3 + 2 * x)^5 = (1 + n * x)^4) : n = 10 / 3 :=
by {
  sorry
}

end determine_some_number_l683_68378


namespace sufficient_condition_inequality_l683_68316

theorem sufficient_condition_inequality (k : ℝ) :
  (k = 0 ∨ (-3 < k ∧ k < 0)) → ∀ x : ℝ, 2 * k * x^2 + k * x - 3 / 8 < 0 :=
sorry

end sufficient_condition_inequality_l683_68316


namespace base4_arithmetic_l683_68397

theorem base4_arithmetic :
  (Nat.ofDigits 4 [2, 3, 1] * Nat.ofDigits 4 [2, 2] / Nat.ofDigits 4 [3]) = Nat.ofDigits 4 [2, 2, 1] := by
sorry

end base4_arithmetic_l683_68397
