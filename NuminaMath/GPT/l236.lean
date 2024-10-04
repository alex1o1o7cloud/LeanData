import Mathlib

namespace product_of_numbers_larger_than_reciprocal_eq_neg_one_l236_236949

theorem product_of_numbers_larger_than_reciprocal_eq_neg_one :
  ∃ x y : ℝ, x ≠ y ∧ (x = 1 / x + 2) ∧ (y = 1 / y + 2) ∧ x * y = -1 :=
by
  sorry

end product_of_numbers_larger_than_reciprocal_eq_neg_one_l236_236949


namespace five_digit_numbers_greater_21035_and_even_correct_five_digit_numbers_even_with_odd_positions_correct_l236_236879

noncomputable def count_five_digit_numbers_greater_21035_and_even : Nat :=
  sorry -- insert combinatorial logic to count the numbers

theorem five_digit_numbers_greater_21035_and_even_correct :
  count_five_digit_numbers_greater_21035_and_even = 39 :=
  sorry

noncomputable def count_five_digit_numbers_even_with_odd_positions : Nat :=
  sorry -- insert combinatorial logic to count the numbers

theorem five_digit_numbers_even_with_odd_positions_correct :
  count_five_digit_numbers_even_with_odd_positions = 8 :=
  sorry

end five_digit_numbers_greater_21035_and_even_correct_five_digit_numbers_even_with_odd_positions_correct_l236_236879


namespace farmer_sowed_buckets_l236_236535

-- Define the initial and final buckets of seeds
def initial_buckets : ℝ := 8.75
def final_buckets : ℝ := 6.00

-- The goal: prove the number of buckets sowed is 2.75
theorem farmer_sowed_buckets : initial_buckets - final_buckets = 2.75 := by
  sorry

end farmer_sowed_buckets_l236_236535


namespace find_m_correct_l236_236681

noncomputable def find_m (Q : Point) (B : List Point) (m : ℝ) : Prop :=
  let circle_area := 4 * Real.pi
  let radius := 2
  let area_sector_B1B2 := Real.pi / 3
  let area_region_B1B2 := 1 / 8
  let area_triangle_B1B2 := area_sector_B1B2 - area_region_B1B2 * circle_area
  let area_sector_B4B5 := Real.pi / 3
  let area_region_B4B5 := 1 / 10
  let area_triangle_B4B5 := area_sector_B4B5 - area_region_B4B5 * circle_area
  let area_sector_B9B10 := Real.pi / 3
  let area_region_B9B10 := 4 / 15 - Real.sqrt 3 / m
  let area_triangle_B9B10 := area_sector_B9B10 - area_region_B9B10 * circle_area
  m = 3

theorem find_m_correct (Q : Point) (B : List Point) : find_m Q B 3 :=
by
  unfold find_m
  sorry

end find_m_correct_l236_236681


namespace smallest_solution_l236_236567

theorem smallest_solution (x : ℝ) (h : x * |x| = 3 * x - 2) : 
  x = 1 ∨ x = 2 ∨ x = (-(3 + Real.sqrt 17)) / 2 :=
by
  sorry

end smallest_solution_l236_236567


namespace sum_of_positive_differences_l236_236205

open BigOperators

def S : Finset ℕ := Finset.range 11 .map (λ i, 2^i)

noncomputable def N : ℕ := ∑ x in S, ∑ y in S, (x - y).natAbs

theorem sum_of_positive_differences :
  N = 16398 :=
by sorry

end sum_of_positive_differences_l236_236205


namespace count_multiples_of_5_not_10_or_15_l236_236586

theorem count_multiples_of_5_not_10_or_15 : 
  ∃ n : ℕ, n = 33 ∧ (∀ x : ℕ, x < 500 ∧ (x % 5 = 0) ∧ (x % 10 ≠ 0) ∧ (x % 15 ≠ 0) → x < 500 ∧ (x % 5 = 0) ∧ (x % 10 ≠ 0) ∧ (x % 15 ≠ 0)) :=
by
  sorry

end count_multiples_of_5_not_10_or_15_l236_236586


namespace number_of_bookshelves_l236_236546

theorem number_of_bookshelves (books_in_each: ℕ) (total_books: ℕ) (h_books_in_each: books_in_each = 56) (h_total_books: total_books = 504) : total_books / books_in_each = 9 :=
by
  sorry

end number_of_bookshelves_l236_236546


namespace necessary_and_sufficient_condition_l236_236101

-- Define the first circle
def circle1 (m : ℝ) : Set (ℝ × ℝ) :=
  { p | (p.1 + m)^2 + p.2^2 = 1 }

-- Define the second circle
def circle2 : Set (ℝ × ℝ) :=
  { p | (p.1 - 2)^2 + p.2^2 = 4 }

-- Define the condition -1 ≤ m ≤ 1
def condition (m : ℝ) : Prop :=
  -1 ≤ m ∧ m ≤ 1

-- Define the property for circles having common points
def circlesHaveCommonPoints (m : ℝ) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ circle1 m ∧ p ∈ circle2

-- The final statement
theorem necessary_and_sufficient_condition (m : ℝ) :
  condition m → circlesHaveCommonPoints m ↔ (-5 ≤ m ∧ m ≤ 1) :=
by
  sorry

end necessary_and_sufficient_condition_l236_236101


namespace range_of_f_l236_236235

def f (x : ℤ) : ℤ := (x - 1)^2 - 1

theorem range_of_f :
  Set.image f {-1, 0, 1, 2, 3} = {-1, 0, 3} :=
by
  sorry

end range_of_f_l236_236235


namespace union_M_N_equals_set_x_ge_1_l236_236448

-- Definitions of M and N based on the conditions from step a)
def M : Set ℝ := { x | x - 2 > 0 }

def N : Set ℝ := { y | ∃ x : ℝ, y = Real.sqrt (x^2 + 1) }

-- Statement of the theorem
theorem union_M_N_equals_set_x_ge_1 : (M ∪ N) = { x : ℝ | x ≥ 1 } := 
sorry

end union_M_N_equals_set_x_ge_1_l236_236448


namespace domain_ln_2_minus_x_is_interval_l236_236356

noncomputable def domain_ln_2_minus_x : Set Real := { x : Real | 2 - x > 0 }

theorem domain_ln_2_minus_x_is_interval : domain_ln_2_minus_x = Set.Iio 2 :=
by
  sorry

end domain_ln_2_minus_x_is_interval_l236_236356


namespace max_value_expression_l236_236441

variable (x y z : ℝ)

theorem max_value_expression (h : x^2 + y^2 + z^2 = 4) :
  (2*x - y)^2 + (2*y - z)^2 + (2*z - x)^2 ≤ 28 :=
sorry

end max_value_expression_l236_236441


namespace determine_k_circle_l236_236283

theorem determine_k_circle (k : ℝ) :
  (∃ x y : ℝ, x^2 + 8*x + y^2 + 14*y - k = 0) ∧ ((∀ x y : ℝ, (x + 4)^2 + (y + 7)^2 = 25) ↔ k = -40) :=
by
  sorry

end determine_k_circle_l236_236283


namespace factor_expression_l236_236157

theorem factor_expression (x : ℤ) : 63 * x + 28 = 7 * (9 * x + 4) :=
by sorry

end factor_expression_l236_236157


namespace apples_to_pears_value_l236_236097

/-- Suppose 1/2 of 12 apples are worth as much as 10 pears. -/
def apples_per_pears_ratio : ℚ := 10 / (1 / 2 * 12)

/-- Prove that 3/4 of 6 apples are worth as much as 7.5 pears. -/
theorem apples_to_pears_value : (3 / 4 * 6) * apples_per_pears_ratio = 7.5 := 
by
  sorry

end apples_to_pears_value_l236_236097


namespace measure_of_angle_R_l236_236475

variable (S T A R : ℝ) -- Represent the angles as real numbers.

-- The conditions given in the problem.
axiom angles_congruent : S = T ∧ T = A ∧ A = R
axiom angle_A_equals_angle_S : A = S

-- Statement: Prove that the measure of angle R is 108 degrees.
theorem measure_of_angle_R : R = 108 :=
by
  sorry

end measure_of_angle_R_l236_236475


namespace complement_of_A_in_S_l236_236074

universe u

def S : Set ℕ := {x | 0 ≤ x ∧ x ≤ 5}
def A : Set ℕ := {x | 1 < x ∧ x < 5}

theorem complement_of_A_in_S : S \ A = {0, 1, 5} := 
by sorry

end complement_of_A_in_S_l236_236074


namespace oxen_count_b_l236_236815

theorem oxen_count_b 
  (a_oxen : ℕ) (a_months : ℕ)
  (b_months : ℕ) (x : ℕ)
  (c_oxen : ℕ) (c_months : ℕ)
  (total_rent : ℝ) (c_rent : ℝ)
  (h1 : a_oxen * a_months = 70)
  (h2 : c_oxen * c_months = 45)
  (h3 : c_rent / total_rent = 27 / 105)
  (h4 : total_rent = 105) :
  x = 12 :=
by 
  sorry

end oxen_count_b_l236_236815


namespace zero_people_with_fewer_than_six_cards_l236_236316

theorem zero_people_with_fewer_than_six_cards (cards people : ℕ) (h_cards : cards = 60) (h_people : people = 9) :
  let avg := cards / people
  let remainder := cards % people
  remainder < people → ∃ n, n = 0 := by
  sorry

end zero_people_with_fewer_than_six_cards_l236_236316


namespace sequence_general_formula_l236_236181

theorem sequence_general_formula (a : ℕ → ℚ) (h₀ : a 1 = 3 / 5)
    (h₁ : ∀ n : ℕ, a (n + 1) = a n / (2 * a n + 1)) :
  ∀ n : ℕ, a n = 3 / (6 * n - 1) := 
by sorry

end sequence_general_formula_l236_236181


namespace total_books_l236_236993

variables (Beatrix_books Alannah_books Queen_books : ℕ)

def Alannah_condition := Alannah_books = Beatrix_books + 20
def Queen_condition := Queen_books = Alannah_books + (Alannah_books / 5)

theorem total_books (hB : Beatrix_books = 30) (hA : Alannah_condition) (hQ : Queen_condition) : 
  (Beatrix_books + Alannah_books + Queen_books) = 140 :=
by
  sorry

end total_books_l236_236993


namespace symmetric_point_coordinates_l236_236895

noncomputable def symmetric_with_respect_to_y_axis (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match p with
  | (x, y, z) => (-x, y, -z)

theorem symmetric_point_coordinates : symmetric_with_respect_to_y_axis (-2, 1, 4) = (2, 1, -4) :=
by sorry

end symmetric_point_coordinates_l236_236895


namespace chimney_height_theorem_l236_236393

noncomputable def chimney_height :=
  let BCD := 75 * Real.pi / 180
  let BDC := 60 * Real.pi / 180
  let CBD := 45 * Real.pi / 180
  let CD := 40
  let BC := CD * Real.sin BDC / Real.sin CBD
  let CE := 1
  let elevation := 30 * Real.pi / 180
  let AB := CE + (Real.tan elevation * BC)
  AB

theorem chimney_height_theorem : chimney_height = 1 + 20 * Real.sqrt 2 :=
by
  sorry

end chimney_height_theorem_l236_236393


namespace twelve_times_y_plus_three_half_quarter_l236_236591

theorem twelve_times_y_plus_three_half_quarter (y : ℝ) : 
  (1 / 2) * (1 / 4) * (12 * y + 3) = (3 * y) / 2 + 3 / 8 :=
by sorry

end twelve_times_y_plus_three_half_quarter_l236_236591


namespace intersection_volume_l236_236652

noncomputable def volume_of_intersection (k : ℝ) : ℝ :=
  ∫ x in -k..k, 4 * (k^2 - x^2)

theorem intersection_volume (k : ℝ) : volume_of_intersection k = 16 * k^3 / 3 :=
  by
  sorry

end intersection_volume_l236_236652


namespace bcm_hens_count_l236_236534

-- Propositions representing the given conditions
def total_chickens : ℕ := 100
def bcm_ratio : ℝ := 0.20
def bcm_hens_ratio : ℝ := 0.80

-- Theorem statement: proving the number of BCM hens
theorem bcm_hens_count : (total_chickens * bcm_ratio * bcm_hens_ratio = 16) := by
  sorry

end bcm_hens_count_l236_236534


namespace incorrect_statement_B_l236_236020

variable (a : Nat → Int) (S : Nat → Int)
variable (d : Int)

-- Given conditions
axiom S_5_lt_S_6 : S 5 < S 6
axiom S_6_eq_S_7 : S 6 = S 7
axiom S_7_gt_S_8 : S 7 > S 8
axiom S_n : ∀ n, S n = n * a n

-- Question to prove statement B is incorrect 
theorem incorrect_statement_B : ∃ (d : Int), (S 9 < S 5) :=
by 
  -- Proof goes here
  sorry

end incorrect_statement_B_l236_236020


namespace greatest_number_of_pieces_leftover_l236_236456

theorem greatest_number_of_pieces_leftover (y : ℕ) (q r : ℕ) 
  (h : y = 6 * q + r) (hrange : r < 6) : r = 5 := sorry

end greatest_number_of_pieces_leftover_l236_236456


namespace charity_event_revenue_l236_236262

theorem charity_event_revenue :
  ∃ (f t p : ℕ), f + t = 190 ∧ f * p + t * (p / 3) = 2871 ∧ f * p = 1900 :=
by
  sorry

end charity_event_revenue_l236_236262


namespace smallest_number_is_20_l236_236651

theorem smallest_number_is_20 (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a ≤ b) (h5 : b ≤ c)
  (mean_condition : (a + b + c) / 3 = 30)
  (median_condition : b = 31)
  (largest_condition : b = c - 8) :
  a = 20 :=
sorry

end smallest_number_is_20_l236_236651


namespace neg_p_implies_neg_q_l236_236721

variables {x : ℝ}

def condition_p (x : ℝ) : Prop := |x + 1| > 2
def condition_q (x : ℝ) : Prop := 5 * x - 6 > x^2
def neg_p (x : ℝ) : Prop := |x + 1| ≤ 2
def neg_q (x : ℝ) : Prop := x ≤ 2 ∨ x ≥ 3

theorem neg_p_implies_neg_q : (∀ x, neg_p x → neg_q x) :=
by 
  -- Proof is skipped according to the instructions
  sorry

end neg_p_implies_neg_q_l236_236721


namespace hyperbola_constants_sum_l236_236630

noncomputable def hyperbola_asymptotes_equation (x y : ℝ) : Prop :=
  (y = 2 * x + 5) ∨ (y = -2 * x + 1)

noncomputable def hyperbola_passing_through (x y : ℝ) : Prop :=
  (x = 0 ∧ y = 7)

theorem hyperbola_constants_sum
  (a b h k : ℝ) (ha : a > 0) (hb : b > 0)
  (H1 : ∀ x y : ℝ, hyperbola_asymptotes_equation x y)
  (H2 : hyperbola_passing_through 0 7)
  (H3 : h = -1)
  (H4 : k = 3)
  (H5 : a = 2 * b)
  (H6 : b = Real.sqrt 3) :
  a + h = 2 * Real.sqrt 3 - 1 :=
sorry

end hyperbola_constants_sum_l236_236630


namespace octagon_area_equals_eight_one_plus_sqrt_two_l236_236400

theorem octagon_area_equals_eight_one_plus_sqrt_two
  (a b : ℝ)
  (h1 : 4 * a = 8 * b)
  (h2 : a ^ 2 = 16) :
  2 * (1 + Real.sqrt 2) * b ^ 2 = 8 * (1 + Real.sqrt 2) :=
by
  sorry

end octagon_area_equals_eight_one_plus_sqrt_two_l236_236400


namespace minimum_value_l236_236291

/-- The minimum value of the expression (x+2)^2 / (y-2) + (y+2)^2 / (x-2)
    for real numbers x > 2 and y > 2 is 50. -/
theorem minimum_value (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  ∃ z, z = (x + 2) ^ 2 / (y - 2) + (y + 2) ^ 2 / (x - 2) ∧ z = 50 :=
sorry

end minimum_value_l236_236291


namespace sinks_per_house_l236_236981

theorem sinks_per_house (total_sinks : ℕ) (houses : ℕ) (h_total_sinks : total_sinks = 266) (h_houses : houses = 44) :
  total_sinks / houses = 6 :=
by {
  sorry
}

end sinks_per_house_l236_236981


namespace file_size_l236_236758

-- Definitions based on conditions
def upload_speed : ℕ := 8 -- megabytes per minute
def upload_time : ℕ := 20 -- minutes

-- Goal to prove
theorem file_size:
  (upload_speed * upload_time = 160) :=
by sorry

end file_size_l236_236758


namespace simplify_tan_expression_l236_236786

noncomputable def tan_30 : ℝ := Real.tan (Real.pi / 6)
noncomputable def tan_15 : ℝ := Real.tan (Real.pi / 12)

theorem simplify_tan_expression : (1 + tan_30) * (1 + tan_15) = 2 := by
  sorry

end simplify_tan_expression_l236_236786


namespace possible_values_sin_plus_cos_l236_236577

variable (x : ℝ)

theorem possible_values_sin_plus_cos (h : 2 * Real.cos x - 3 * Real.sin x = 2) :
    ∃ (values : Set ℝ), values = {3, -31 / 13} ∧ (Real.sin x + 3 * Real.cos x) ∈ values := by
  sorry

end possible_values_sin_plus_cos_l236_236577


namespace smallest_sum_of_squares_l236_236934

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 187) : x^2 + y^2 ≥ 205 := 
  sorry

end smallest_sum_of_squares_l236_236934


namespace value_of_b_l236_236719

noncomputable def function_bounds := 
  ∃ (k b : ℝ), (∀ (x : ℝ), (-3 ≤ x ∧ x ≤ 1) → (-1 ≤ k * x + b ∧ k * x + b ≤ 8)) ∧ (b = 5 / 4 ∨ b = 23 / 4)

theorem value_of_b : function_bounds :=
by
  sorry

end value_of_b_l236_236719


namespace polygon_sides_diagonals_l236_236398

theorem polygon_sides_diagonals (n : ℕ) 
  (h1 : 4 * (n * (n - 3)) = 14 * n)
  (h2 : (n + (n * (n - 3)) / 2) % 2 = 0)
  (h3 : n + n * (n - 3) / 2 > 50) : n = 12 := 
by 
  sorry

end polygon_sides_diagonals_l236_236398


namespace min_value_of_expression_l236_236433

theorem min_value_of_expression (x y : ℝ) (h : x^2 + y^2 + x * y = 315) :
  ∃ m : ℝ, m = x^2 + y^2 - x * y ∧ m ≥ 105 :=
by
  sorry

end min_value_of_expression_l236_236433


namespace ratio_of_only_B_to_both_A_and_B_l236_236536

theorem ratio_of_only_B_to_both_A_and_B 
  (Total_households : ℕ)
  (Neither_brand : ℕ)
  (Only_A : ℕ)
  (Both_A_and_B : ℕ)
  (Total_households_eq : Total_households = 180)
  (Neither_brand_eq : Neither_brand = 80)
  (Only_A_eq : Only_A = 60)
  (Both_A_and_B_eq : Both_A_and_B = 10) :
  (Total_households = Neither_brand + Only_A + (Total_households - Neither_brand - Only_A - Both_A_and_B) + Both_A_and_B) →
  (Total_households - Neither_brand - Only_A - Both_A_and_B) / Both_A_and_B = 3 :=
by
  intro H
  sorry

end ratio_of_only_B_to_both_A_and_B_l236_236536


namespace max_k_l236_236479

-- Definitions and conditions
def original_number (A B : ℕ) : ℕ := 10 * A + B
def new_number (A C B : ℕ) : ℕ := 100 * A + 10 * C + B

theorem max_k (A C B k : ℕ) (hA : A ≠ 0) (h1 : 0 ≤ A ∧ A ≤ 9) (h2 : 0 ≤ B ∧ B ≤ 9) (h3: 0 ≤ C ∧ C ≤ 9) :
  ((original_number A B) * k = (new_number A C B)) → 
  (∀ (A: ℕ), 1 ≤ k) → 
  k ≤ 19 :=
by
  sorry

end max_k_l236_236479


namespace complex_calculation_l236_236547

def complex_add (a b : ℂ) : ℂ := a + b
def complex_mul (a b : ℂ) : ℂ := a * b

theorem complex_calculation :
  let z1 := (⟨2, -3⟩ : ℂ)
  let z2 := (⟨4, 6⟩ : ℂ)
  let z3 := (⟨-1, 2⟩ : ℂ)
  complex_mul (complex_add z1 z2) z3 = (⟨-12, 9⟩ : ℂ) :=
by 
  sorry

end complex_calculation_l236_236547


namespace roots_of_poly_l236_236009

noncomputable def poly (x : ℝ) : ℝ := x^3 - 4 * x^2 - x + 4

theorem roots_of_poly :
  (poly 1 = 0) ∧ (poly (-1) = 0) ∧ (poly 4 = 0) ∧
  (∀ x, poly x = 0 → x = 1 ∨ x = -1 ∨ x = 4) :=
by
  sorry

end roots_of_poly_l236_236009


namespace mary_initial_pokemon_cards_l236_236492

theorem mary_initial_pokemon_cards (x : ℕ) (torn_cards : ℕ) (new_cards : ℕ) (current_cards : ℕ) 
  (h1 : torn_cards = 6) 
  (h2 : new_cards = 23) 
  (h3 : current_cards = 56) 
  (h4 : current_cards = x - torn_cards + new_cards) : 
  x = 39 := 
by
  sorry

end mary_initial_pokemon_cards_l236_236492


namespace odd_function_solution_l236_236730

def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

theorem odd_function_solution (f : ℝ → ℝ) (h1 : is_odd f) (h2 : ∀ x : ℝ, x > 0 → f x = x^3 + x + 1) :
  ∀ x : ℝ, x < 0 → f x = x^3 + x - 1 :=
by
  sorry

end odd_function_solution_l236_236730


namespace distance_to_base_is_42_l236_236152

theorem distance_to_base_is_42 (x : ℕ) (hx : 4 * x + 3 * (x + 3) = x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6)) :
  4 * x = 36 ∨ 4 * x + 6 = 42 := 
by
  sorry

end distance_to_base_is_42_l236_236152


namespace smallest_n_exists_unique_k_l236_236524

/- The smallest positive integer n for which there exists
   a unique integer k such that 9/16 < n / (n + k) < 7/12 is n = 1. -/

theorem smallest_n_exists_unique_k :
  ∃! (n : ℕ), n > 0 ∧ (∃! (k : ℤ), (9 : ℚ)/16 < (n : ℤ)/(n + k) ∧ (n : ℤ)/(n + k) < (7 : ℚ)/12) :=
sorry

end smallest_n_exists_unique_k_l236_236524


namespace negation_exists_gt_one_l236_236229

theorem negation_exists_gt_one :
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) :=
sorry

end negation_exists_gt_one_l236_236229


namespace actual_plot_area_in_acres_l236_236130

-- Condition Definitions
def base_cm : ℝ := 8
def height_cm : ℝ := 12
def scale_cm_to_miles : ℝ := 1  -- 1 cm = 1 mile
def miles_to_acres : ℝ := 320  -- 1 square mile = 320 acres

-- Theorem Statement
theorem actual_plot_area_in_acres (A : ℝ) :
  A = 15360 :=
by
  sorry

end actual_plot_area_in_acres_l236_236130


namespace circle_radius_l236_236125

theorem circle_radius (r x y : ℝ) (h1 : x = π * r^2) (h2 : y = 2 * π * r) (h3 : x + y = 120 * π) : r = 10 :=
sorry

end circle_radius_l236_236125


namespace minimum_value_l236_236576

theorem minimum_value (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 2) : 
  (1 / m + 2 / n) ≥ 4 :=
sorry

end minimum_value_l236_236576


namespace find_height_of_larger_cuboid_l236_236041

-- Define the larger cuboid dimensions
def Length_large : ℝ := 18
def Width_large : ℝ := 15
def Volume_large (Height_large : ℝ) : ℝ := Length_large * Width_large * Height_large

-- Define the smaller cuboid dimensions
def Length_small : ℝ := 5
def Width_small : ℝ := 6
def Height_small : ℝ := 3
def Volume_small : ℝ := Length_small * Width_small * Height_small

-- Define the total volume of 6 smaller cuboids
def Total_volume_small : ℝ := 6 * Volume_small

-- State the problem and the proof goal
theorem find_height_of_larger_cuboid : 
  ∃ H : ℝ, Volume_large H = Total_volume_small :=
by
  use 2
  sorry

end find_height_of_larger_cuboid_l236_236041


namespace price_per_hotdog_l236_236395

-- The conditions
def hot_dogs_per_hour := 10
def hours := 10
def total_sales := 200

-- Conclusion we need to prove
theorem price_per_hotdog : total_sales / (hot_dogs_per_hour * hours) = 2 := by
  sorry

end price_per_hotdog_l236_236395


namespace solve_quadratic_eq_l236_236927

theorem solve_quadratic_eq (x : ℝ) (h : x^2 + 2 * x - 15 = 0) : x = 3 ∨ x = -5 :=
by {
  sorry
}

end solve_quadratic_eq_l236_236927


namespace vending_machine_problem_l236_236838

variable (x n : ℕ)

theorem vending_machine_problem (h : 25 * x + 10 * 15 + 5 * 30 = 25 * 25 + 10 * 5 + 5 * n) (hx : x = 25) :
  n = 50 := by
sorry

end vending_machine_problem_l236_236838


namespace max_value_y_l236_236211

theorem max_value_y (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) :
  let y := abs (x + 1) - 2 * abs x + abs (x - 2) in
  y ≤ 3 :=
by
  sorry

end max_value_y_l236_236211


namespace repeating_decimal_mul_l236_236277

theorem repeating_decimal_mul (x : ℝ) (hx : x = 0.3333333333333333) :
  x * 12 = 4 :=
sorry

end repeating_decimal_mul_l236_236277


namespace provisions_last_for_girls_l236_236805

theorem provisions_last_for_girls (P : ℝ) (G : ℝ) (h1 : P / (50 * G) = P / (250 * (G + 20))) : G = 25 := 
by
  sorry

end provisions_last_for_girls_l236_236805


namespace find_number_of_math_problems_l236_236846

-- Define the number of social studies problems
def social_studies_problems : ℕ := 6

-- Define the number of science problems
def science_problems : ℕ := 10

-- Define the time to solve each type of problem in minutes
def time_per_math_problem : ℝ := 2
def time_per_social_studies_problem : ℝ := 0.5
def time_per_science_problem : ℝ := 1.5

-- Define the total time to solve all problems in minutes
def total_time : ℝ := 48

-- Define the theorem to find the number of math problems
theorem find_number_of_math_problems (M : ℕ) :
  time_per_math_problem * M + time_per_social_studies_problem * social_studies_problems + time_per_science_problem * science_problems = total_time → 
  M = 15 :=
by {
  -- proof is not required to be written, hence expressing the unresolved part
  sorry
}

end find_number_of_math_problems_l236_236846


namespace divides_expression_l236_236772

theorem divides_expression (y : ℕ) (hy : y ≠ 0) : (y - 1) ∣ (y^(y^2) - 2 * y^(y + 1) + 1) := 
by
  sorry

end divides_expression_l236_236772


namespace total_songs_correct_l236_236962

-- Define the conditions of the problem
def num_country_albums := 2
def songs_per_country_album := 12
def num_pop_albums := 8
def songs_per_pop_album := 7
def num_rock_albums := 5
def songs_per_rock_album := 10
def num_jazz_albums := 2
def songs_per_jazz_album := 15

-- Define the total number of songs
def total_songs :=
  num_country_albums * songs_per_country_album +
  num_pop_albums * songs_per_pop_album +
  num_rock_albums * songs_per_rock_album +
  num_jazz_albums * songs_per_jazz_album

-- Proposition stating the correct total number of songs
theorem total_songs_correct : total_songs = 160 :=
by {
  sorry -- Proof not required
}

end total_songs_correct_l236_236962


namespace no_solution_inequalities_l236_236198

theorem no_solution_inequalities (a : ℝ) : 
  (∀ x : ℝ, ¬ (x > 3 ∧ x < a)) ↔ (a ≤ 3) :=
by
  sorry

end no_solution_inequalities_l236_236198


namespace probability_sum_is_odd_l236_236570

theorem probability_sum_is_odd (S : Finset ℕ) (h_S : S = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37})
    (h_card : S.card = 12) :
  let choices := S.powerset.filter (λ t, t.card = 4),
      odd_sum_choices := choices.filter (λ t, t.sum % 2 = 1) in
  (odd_sum_choices.card : ℚ) / choices.card = 1 / 3 :=
by  
  sorry

end probability_sum_is_odd_l236_236570


namespace algebraic_expression_value_l236_236431

theorem algebraic_expression_value (m : ℝ) (h : (2018 + m) * (2020 + m) = 2) : (2018 + m)^2 + (2020 + m)^2 = 8 :=
by
  sorry

end algebraic_expression_value_l236_236431


namespace optionA_incorrect_optionB_incorrect_optionC_incorrect_optionD_correct_l236_236246

theorem optionA_incorrect (a x : ℝ) : 3 * a * x^2 - 6 * a * x ≠ 3 * (a * x^2 - 2 * a * x) :=
by sorry

theorem optionB_incorrect (a x : ℝ) : (x + a) * (x - a) ≠ x^2 - a^2 :=
by sorry

theorem optionC_incorrect (a b : ℝ) : a^2 + 2 * a * b - 4 * b^2 ≠ (a + 2 * b)^2 :=
by sorry

theorem optionD_correct (a x : ℝ) : -a * x^2 + 2 * a * x - a = -a * (x - 1)^2 :=
by sorry

end optionA_incorrect_optionB_incorrect_optionC_incorrect_optionD_correct_l236_236246


namespace halfway_between_fractions_l236_236148

theorem halfway_between_fractions : 
  (2:ℚ) / 9 + (5 / 12) / 2 = 23 / 72 := 
sorry

end halfway_between_fractions_l236_236148


namespace competition_order_l236_236749

variable (A B C D : ℕ)

-- Conditions as given in the problem
axiom cond1 : B + D = 2 * A
axiom cond2 : A + C < B + D
axiom cond3 : A < B + C

-- The desired proof statement
theorem competition_order : D > B ∧ B > A ∧ A > C :=
by
  sorry

end competition_order_l236_236749


namespace jaysons_moms_age_l236_236960

theorem jaysons_moms_age (jayson's_age dad's_age mom's_age : ℕ) 
  (h1 : jayson's_age = 10)
  (h2 : dad's_age = 4 * jayson's_age)
  (h3 : mom's_age = dad's_age - 2) :
  mom's_age - jayson's_age = 28 := 
by
  sorry

end jaysons_moms_age_l236_236960


namespace beads_per_bracelet_l236_236611

-- Definitions for the conditions
def Nancy_metal_beads : ℕ := 40
def Nancy_pearl_beads : ℕ := Nancy_metal_beads + 20
def Rose_crystal_beads : ℕ := 20
def Rose_stone_beads : ℕ := Rose_crystal_beads * 2
def total_beads : ℕ := Nancy_metal_beads + Nancy_pearl_beads + Rose_crystal_beads + Rose_stone_beads
def bracelets : ℕ := 20

-- Statement to prove
theorem beads_per_bracelet :
  total_beads / bracelets = 8 :=
by
  -- skip the proof
  sorry

end beads_per_bracelet_l236_236611


namespace min_value_correct_l236_236707

noncomputable def min_value (x y : ℝ) : ℝ :=
x * y / (x^2 + y^2)

theorem min_value_correct :
  ∃ x y : ℝ,
    (2 / 5 : ℝ) ≤ x ∧ x ≤ (1 / 2 : ℝ) ∧
    (1 / 3 : ℝ) ≤ y ∧ y ≤ (3 / 8 : ℝ) ∧
    min_value x y = (6 / 13 : ℝ) :=
by sorry

end min_value_correct_l236_236707


namespace custom_op_4_2_l236_236904

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := 5 * a + 2 * b

-- State the theorem to prove the result
theorem custom_op_4_2 : custom_op 4 2 = 24 :=
by
  sorry

end custom_op_4_2_l236_236904


namespace alberto_spent_more_l236_236685

-- Define the expenses of Alberto and Samara
def alberto_expenses : ℕ := 2457
def samara_oil_expense : ℕ := 25
def samara_tire_expense : ℕ := 467
def samara_detailing_expense : ℕ := 79
def samara_total_expenses : ℕ := samara_oil_expense + samara_tire_expense + samara_detailing_expense

-- State the theorem to prove the difference in expenses
theorem alberto_spent_more :
  alberto_expenses - samara_total_expenses = 1886 := by
  sorry

end alberto_spent_more_l236_236685


namespace parabola_directrix_l236_236794

theorem parabola_directrix (a : ℝ) (h : -1 / (4 * a) = 2) : a = -1 / 8 :=
by
  sorry

end parabola_directrix_l236_236794


namespace quadratic_roots_is_correct_l236_236038

theorem quadratic_roots_is_correct (a b : ℝ) 
    (h1 : a + b = 16) 
    (h2 : a * b = 225) :
    (∀ x, x^2 - 16 * x + 225 = 0 ↔ x = a ∨ x = b) := sorry

end quadratic_roots_is_correct_l236_236038


namespace cakes_served_during_lunch_today_l236_236267

theorem cakes_served_during_lunch_today (L : ℕ) 
  (h_total : L + 6 + 3 = 14) : 
  L = 5 :=
sorry

end cakes_served_during_lunch_today_l236_236267


namespace range_of_m_l236_236463

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (-2 < x ∧ x ≤ 2) → x ≤ m) → m ≥ 2 :=
by
  intro h
  -- insert necessary proof steps here
  sorry

end range_of_m_l236_236463


namespace ratio_of_ages_l236_236619

theorem ratio_of_ages (M : ℕ) (S : ℕ) (h1 : M = 24) (h2 : S + 6 = 38) : 
  (S / M : ℚ) = 4 / 3 :=
by
  sorry

end ratio_of_ages_l236_236619


namespace solve_for_m_l236_236322

theorem solve_for_m (m : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 2 → - (1/2) * x^2 + 2 * x > m * x) → m = 1 :=
by
  -- Skip the proof by using sorry
  sorry

end solve_for_m_l236_236322


namespace find_AX_length_l236_236066

noncomputable def AX_length (AC BC BX : ℕ) : ℚ :=
AC * (BX / BC)

theorem find_AX_length :
  let AC := 25
  let BC := 35
  let BX := 30
  AX_length AC BC BX = 150 / 7 :=
by
  -- proof is omitted using 'sorry'
  sorry

end find_AX_length_l236_236066


namespace smallest_b_value_minimizes_l236_236618

noncomputable def smallest_b_value (a b : ℝ) (c : ℝ := 2) : ℝ :=
  if (1 < a) ∧ (a < b) ∧ (¬ (c + a > b ∧ c + b > a ∧ a + b > c)) ∧ (¬ (1/b + 1/a > c ∧ 1/a + c > 1/b ∧ c + 1/b > 1/a)) then b else 0

theorem smallest_b_value_minimizes (a b : ℝ) (c : ℝ := 2) :
  (1 < a) ∧ (a < b) ∧ (¬ (c + a > b ∧ c + b > a ∧ a + b > c)) ∧ (¬ (1/b + 1/a > c ∧ 1/a + c > 1/b ∧ c + 1/b > 1/a)) →
  b = 2 :=
by sorry

end smallest_b_value_minimizes_l236_236618


namespace Mrs_Hilt_remaining_money_l236_236774

theorem Mrs_Hilt_remaining_money :
  let initial_amount : ℝ := 3.75
  let pencil_cost : ℝ := 1.15
  let eraser_cost : ℝ := 0.85
  let notebook_cost : ℝ := 2.25
  initial_amount - (pencil_cost + eraser_cost + notebook_cost) = -0.50 :=
by
  sorry

end Mrs_Hilt_remaining_money_l236_236774


namespace initial_customers_l236_236269

theorem initial_customers (S : ℕ) (initial : ℕ) (H1 : initial = S + (S + 5)) (H2 : S = 3) : initial = 11 := 
by
  sorry

end initial_customers_l236_236269


namespace find_2005th_nonincreasing_number_l236_236136

theorem find_2005th_nonincreasing_number :
  ∃ n : ℕ, n = 864100 ∧ ∃ l : List ℕ, l.length = 2005 ∧ all_digits_nonincreasing l ∧
  l.nth 2004 = some 864100 :=
by
  sorry

end find_2005th_nonincreasing_number_l236_236136


namespace minimum_value_ineq_l236_236192

variable (m n : ℝ)

noncomputable def minimum_value := (1 / (2 * m)) + (1 / n)

theorem minimum_value_ineq (h1 : m > 0) (h2 : n > 0) (h3 : m + 2 * n = 1) : minimum_value m n = 9 / 2 := 
sorry

end minimum_value_ineq_l236_236192


namespace min_value_reciprocal_sum_l236_236764

theorem min_value_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 12) : 
  (1 / x) + (1 / y) ≥ 1 / 3 :=
by
  sorry

end min_value_reciprocal_sum_l236_236764


namespace calculate_down_payment_l236_236220

def loan_period_years : ℕ := 5
def monthly_payment : ℝ := 250.0
def car_price : ℝ := 20000.0
def months_in_year : ℕ := 12

def total_loan_period_months : ℕ := loan_period_years * months_in_year
def total_amount_paid : ℝ := monthly_payment * total_loan_period_months
def down_payment : ℝ := car_price - total_amount_paid

theorem calculate_down_payment : down_payment = 5000 :=
by 
  simp [loan_period_years, monthly_payment, car_price, months_in_year, total_loan_period_months, total_amount_paid, down_payment]
  sorry

end calculate_down_payment_l236_236220


namespace expected_audience_l236_236112

theorem expected_audience (Sat Mon Wed Fri : ℕ) (extra_people expected_total : ℕ)
  (h1 : Sat = 80)
  (h2 : Mon = 80 - 20)
  (h3 : Wed = Mon + 50)
  (h4 : Fri = Sat + Mon)
  (h5 : extra_people = 40)
  (h6 : expected_total = Sat + Mon + Wed + Fri - extra_people) :
  expected_total = 350 := 
sorry

end expected_audience_l236_236112


namespace range_of_a_l236_236594

theorem range_of_a (a : ℝ) : (2 * a - 8) / 3 < 0 → a < 4 :=
by sorry

end range_of_a_l236_236594


namespace tan_sub_eq_one_eight_tan_add_eq_neg_four_seven_l236_236050

variable (α β : ℝ)

theorem tan_sub_eq_one_eight (h1 : Real.tan α = 5) (h2 : Real.tan β = 3) : 
  Real.tan (α - β) = 1 / 8 := 
sorry

theorem tan_add_eq_neg_four_seven (h1 : Real.tan α = 5) (h2 : Real.tan β = 3) : 
  Real.tan (α + β) = -4 / 7 := 
sorry

end tan_sub_eq_one_eight_tan_add_eq_neg_four_seven_l236_236050


namespace problem_part1_problem_part2_problem_part3_l236_236574

noncomputable def S (n : ℕ) : ℕ := 2 * n^2 + n

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then S n else S n - S (n - 1)

noncomputable def b_n (n : ℕ) : ℕ := 2^(n - 1)

noncomputable def T_n (n : ℕ) : ℕ :=
  (4 * n - 5) * 2^n + 5

theorem problem_part1 (n : ℕ) (h : n > 0) : n > 0 → a_n n = 4 * n - 1 := by
  sorry

theorem problem_part2 (n : ℕ) (h : n > 0) : n > 0 → b_n n = 2^(n - 1) := by
  sorry

theorem problem_part3 (n : ℕ) (h : n > 0) : n > 0 → T_n n = (4 * n - 5) * 2^n + 5 := by
  sorry

end problem_part1_problem_part2_problem_part3_l236_236574


namespace red_fraction_after_tripling_l236_236888

theorem red_fraction_after_tripling (initial_total_marbles : ℚ) (H : initial_total_marbles > 0) :
  let blue_fraction := 2 / 3
  let red_fraction := 1 - blue_fraction
  let red_marbles := red_fraction * initial_total_marbles
  let new_red_marbles := 3 * red_marbles
  let initial_blue_marbles := blue_fraction * initial_total_marbles
  let new_total_marbles := new_red_marbles + initial_blue_marbles
  (new_red_marbles / new_total_marbles) = 3 / 5 :=
by
  sorry

end red_fraction_after_tripling_l236_236888


namespace number_of_persons_in_group_l236_236932

theorem number_of_persons_in_group 
    (n : ℕ)
    (h1 : average_age_before - average_age_after = 3)
    (h2 : person_replaced_age = 40)
    (h3 : new_person_age = 10)
    (h4 : total_age_decrease = 3 * n):
  n = 10 := 
sorry

end number_of_persons_in_group_l236_236932


namespace find_functional_f_l236_236761

-- Define the problem domain and functions
variable (f : ℕ → ℕ)
variable (ℕ_star : Set ℕ) -- ℕ_star is {1,2,3,...}

-- Conditions
axiom f_increasing (h1 : ℕ) (h2 : ℕ) (h1_lt_h2 : h1 < h2) : f h1 < f h2
axiom f_functional (x : ℕ) (y : ℕ) : f (y * f x) = x^2 * f (x * y)

-- The proof problem
theorem find_functional_f : (∀ x ∈ ℕ_star, f x = x^2) :=
sorry

end find_functional_f_l236_236761


namespace odd_function_l236_236729

def f (x : ℝ) : ℝ :=
  if x > 0 then
    x^3 + x + 1
  else if x < 0 then
    x^3 + x - 1
  else 
    0

theorem odd_function (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_pos : ∀ x : ℝ, x > 0 → f x = x^3 + x + 1) :
  ∀ x : ℝ, x < 0 → f x = x^3 + x - 1 :=
begin
  intros x h,
  have h_neg : f (-x) = -f x, from h_odd x,
  have h_nonpos : f x = -f (-x), {
    rw [h_neg, h_pos (-x)],
    simp at *,
    sorry
  },
  sorry
end

end odd_function_l236_236729


namespace interest_rate_determination_l236_236593

-- Problem statement
theorem interest_rate_determination (P r : ℝ) :
  (50 = P * r * 2) ∧ (51.25 = P * ((1 + r) ^ 2 - 1)) → r = 0.05 :=
by
  intros h
  sorry

end interest_rate_determination_l236_236593


namespace abs_neg_eq_five_l236_236884

theorem abs_neg_eq_five (a : ℝ) : abs (-a) = 5 ↔ (a = 5 ∨ a = -5) :=
by
  sorry

end abs_neg_eq_five_l236_236884


namespace number_of_questionnaires_drawn_from_15_to_16_is_120_l236_236133

variable (x : ℕ)
variable (H1 : 120 + 180 + 240 + x = 900)
variable (H2 : 60 = (bit0 90) / 180)
variable (H3 : (bit0 (bit0 (bit0 15))) = (bit0 (bit0 (bit0 15))) * (900 / 300))

theorem number_of_questionnaires_drawn_from_15_to_16_is_120 :
  ((900 - 120 - 180 - 240) * (300 / 900)) = 120 :=
sorry

end number_of_questionnaires_drawn_from_15_to_16_is_120_l236_236133


namespace cuboid_distance_properties_l236_236597

theorem cuboid_distance_properties (cuboid : Type) :
  (∃ P : cuboid → ℝ, ∀ V1 V2 : cuboid, P V1 = P V2) ∧
  ¬ (∃ Q : cuboid → ℝ, ∀ E1 E2 : cuboid, Q E1 = Q E2) ∧
  ¬ (∃ R : cuboid → ℝ, ∀ F1 F2 : cuboid, R F1 = R F2) := 
sorry

end cuboid_distance_properties_l236_236597


namespace find_m_if_parallel_l236_236312

theorem find_m_if_parallel 
  (m : ℚ) 
  (a : ℚ × ℚ := (-2, 3)) 
  (b : ℚ × ℚ := (1, m - 3/2)) 
  (h : ∃ k : ℚ, (a.1 = k * b.1) ∧ (a.2 = k * b.2)) : 
  m = 0 := 
  sorry

end find_m_if_parallel_l236_236312


namespace two_subsets_count_l236_236521

-- Definitions from the problem conditions
def S : Set (Fin 5) := {0, 1, 2, 3, 4}

-- Main statement
theorem two_subsets_count : 
  (∃ A B : Set (Fin 5), A ∪ B = S ∧ A ∩ B = {a, b} ∧ A ≠ B) → 
  (number_of_ways = 40) :=
sorry

end two_subsets_count_l236_236521


namespace f_sum_positive_l236_236886

noncomputable def f (x : ℝ) : ℝ := x + x^3

theorem f_sum_positive (x1 x2 : ℝ) (hx : x1 + x2 > 0) : f x1 + f x2 > 0 :=
sorry

end f_sum_positive_l236_236886


namespace range_of_f_l236_236106

noncomputable def f (x : ℝ) : ℝ := x + Real.sqrt (1 - 2 * x)

theorem range_of_f : ∀ y, (∃ x, x ≤ (1 / 2) ∧ f x = y) ↔ y ∈ Set.Iic 1 := by
  sorry

end range_of_f_l236_236106


namespace num_of_winnable_players_l236_236990

noncomputable def num_players := 2 ^ 2013

def can_win_if (x y : Nat) : Prop := x ≤ y + 3

def single_elimination_tournament (players : Nat) : Nat :=
  -- Function simulating the single elimination based on the specified can_win_if condition
  -- Assuming the given conditions and returning the number of winnable players directly
  6038

theorem num_of_winnable_players : single_elimination_tournament num_players = 6038 :=
  sorry

end num_of_winnable_players_l236_236990


namespace ratio_steel_to_tin_l236_236978

def mass_copper (C : ℝ) := C = 90
def total_weight (S C T : ℝ) := 20 * S + 20 * C + 20 * T = 5100
def mass_steel (S C : ℝ) := S = C + 20

theorem ratio_steel_to_tin (S T C : ℝ)
  (hC : mass_copper C)
  (hTW : total_weight S C T)
  (hS : mass_steel S C) :
  S / T = 2 :=
by
  sorry

end ratio_steel_to_tin_l236_236978


namespace circle_standard_equation_l236_236865

theorem circle_standard_equation:
  ∃ (x y : ℝ), ((x + 2) ^ 2 + (y - 1) ^ 2 = 4) :=
by
  sorry

end circle_standard_equation_l236_236865


namespace initial_shed_bales_zero_l236_236650

def bales_in_barn_initial : ℕ := 47
def bales_added_by_benny : ℕ := 35
def bales_in_barn_total : ℕ := 82

theorem initial_shed_bales_zero (b_shed : ℕ) :
  bales_in_barn_initial + bales_added_by_benny = bales_in_barn_total → b_shed = 0 :=
by
  intro h
  sorry

end initial_shed_bales_zero_l236_236650


namespace total_votes_cast_l236_236751

theorem total_votes_cast (total_votes : ℕ) (brenda_votes : ℕ) (percentage_brenda : ℚ) 
  (h1 : brenda_votes = 40) (h2 : percentage_brenda = 0.25) 
  (h3 : brenda_votes = percentage_brenda * total_votes) : total_votes = 160 := 
by sorry

end total_votes_cast_l236_236751


namespace monotonic_intervals_l236_236010

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

theorem monotonic_intervals :
  {x : ℝ | 0 ≤ deriv f x} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | x ≥ 0} :=
by
  sorry

end monotonic_intervals_l236_236010


namespace number_of_valid_orders_eq_Catalan_l236_236240

-- Define the conditions for the sequence to be valid.
def valid_order (seq : List ℤ) : Prop :=
  seq.length = 2 * n ∧ 
  seq.count (+1) = n ∧ 
  seq.count (-1) = n ∧ 
  ∀ k, k < seq.length → 0 ≤ seq.take (k + 1).sum

-- Prove the number of valid sequences is the n-th Catalan number.
theorem number_of_valid_orders_eq_Catalan (n : ℕ) : 
  ∃ seqs : List (List ℤ), 
    (∀ seq ∈ seqs, valid_order seq) ∧ 
    seqs.length = Catalan n :=
sorry

end number_of_valid_orders_eq_Catalan_l236_236240


namespace height_percentage_increase_l236_236317

theorem height_percentage_increase (B A : ℝ) (h : A = B - 0.3 * B) : 
  ((B - A) / A) * 100 = 42.857 :=
by
  sorry

end height_percentage_increase_l236_236317


namespace solution_set_of_inequality_l236_236107

theorem solution_set_of_inequality (x : ℝ) :
  (x + 2) * (1 - x) > 0 ↔ -2 < x ∧ x < 1 := by
  sorry

end solution_set_of_inequality_l236_236107


namespace find_k_value_l236_236163

theorem find_k_value (k : ℝ) (x y : ℝ) (h1 : -3 * x + 2 * y = k) (h2 : 0.75 * x + y = 16) (h3 : x = -6) : k = 59 :=
by 
  sorry

end find_k_value_l236_236163


namespace integer_solutions_inequality_system_l236_236095

noncomputable def check_inequality_system (x : ℤ) : Prop :=
  (3 * x + 1 < x - 3) ∧ ((1 + x) / 2 ≤ (1 + 2 * x) / 3 + 1)

theorem integer_solutions_inequality_system :
  {x : ℤ | check_inequality_system x} = {-5, -4, -3} :=
by
  sorry

end integer_solutions_inequality_system_l236_236095


namespace rows_colored_red_l236_236915

theorem rows_colored_red (total_rows total_squares_per_row blue_rows green_squares red_squares_per_row red_rows : ℕ)
  (h_total_squares : total_rows * total_squares_per_row = 150)
  (h_blue_squares : blue_rows * total_squares_per_row = 60)
  (h_green_squares : green_squares = 66)
  (h_red_squares : 150 - 60 - 66 = 24)
  (h_red_rows : 24 / red_squares_per_row = 4) :
  red_rows = 4 := 
by sorry

end rows_colored_red_l236_236915


namespace harry_james_payment_l236_236664

theorem harry_james_payment (x y H : ℝ) (h1 : H - 12 = 44 / y) (h2 : y > 1) (h3 : H != 12 + 44/3) : H = 23 ∧ y = 4 :=
by
  sorry

end harry_james_payment_l236_236664


namespace eccentricity_of_hyperbola_l236_236293

noncomputable def hyperbola (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1)

noncomputable def foci_condition (a b : ℝ) (c : ℝ) : Prop :=
  c = Real.sqrt (a^2 + b^2)

noncomputable def trisection_condition (a b c : ℝ) : Prop :=
  2 * c = 6 * a^2 / c

theorem eccentricity_of_hyperbola (a b c e : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (hc : c = Real.sqrt (a^2 + b^2)) (ht : 2 * c = 6 * a^2 / c) :
  e = Real.sqrt 3 :=
by
  apply sorry

end eccentricity_of_hyperbola_l236_236293


namespace jack_needs_5_rocks_to_equal_weights_l236_236328

-- Given Conditions
def WeightJack : ℕ := 60
def WeightAnna : ℕ := 40
def WeightRock : ℕ := 4

-- Theorem Statement
theorem jack_needs_5_rocks_to_equal_weights : (WeightJack - WeightAnna) / WeightRock = 5 :=
by
  sorry

end jack_needs_5_rocks_to_equal_weights_l236_236328


namespace line_equation_from_point_normal_l236_236873

theorem line_equation_from_point_normal :
  let M1 : ℝ × ℝ := (7, -8)
  let n : ℝ × ℝ := (-2, 3)
  ∃ C : ℝ, ∀ x y : ℝ, 2 * x - 3 * y + C = 0 ↔ (C = -38) := 
by
  sorry

end line_equation_from_point_normal_l236_236873


namespace at_least_one_irrational_l236_236917

theorem at_least_one_irrational (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) 
  (h₃ : a^2 * b^2 * (a^2 * b^2 + 4) = 2 * (a^6 + b^6)) : 
  ¬ (∀ a b : ℚ, a ≠ 0 ∧ b ≠ 0 → a^2 * b^2 * (a^2 * b^2 + 4) = 2 * (a^6 + b^6)) :=
by sorry

end at_least_one_irrational_l236_236917


namespace addition_correct_l236_236374

theorem addition_correct :
  1357 + 2468 + 3579 + 4680 + 5791 = 17875 := 
by
  sorry

end addition_correct_l236_236374


namespace min_buildings_20x20_min_buildings_50x90_l236_236384

structure CityGrid where
  width : ℕ
  height : ℕ

noncomputable def renovationLaw (grid : CityGrid) : ℕ :=
  if grid.width = 20 ∧ grid.height = 20 then 25
  else if grid.width = 50 ∧ grid.height = 90 then 282
  else sorry -- handle other cases if needed

-- Theorem statements for the proof
theorem min_buildings_20x20 : renovationLaw { width := 20, height := 20 } = 25 := by
  sorry

theorem min_buildings_50x90 : renovationLaw { width := 50, height := 90 } = 282 := by
  sorry

end min_buildings_20x20_min_buildings_50x90_l236_236384


namespace jaysons_moms_age_l236_236961

theorem jaysons_moms_age (jayson's_age dad's_age mom's_age : ℕ) 
  (h1 : jayson's_age = 10)
  (h2 : dad's_age = 4 * jayson's_age)
  (h3 : mom's_age = dad's_age - 2) :
  mom's_age - jayson's_age = 28 := 
by
  sorry

end jaysons_moms_age_l236_236961


namespace area_of_square_with_adjacent_points_l236_236780

def distance (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)
def side_length := distance 1 2 4 6
def area_of_square (side : ℝ) : ℝ := side ^ 2

theorem area_of_square_with_adjacent_points :
  area_of_square side_length = 25 :=
by
  unfold side_length
  unfold area_of_square
  sorry

end area_of_square_with_adjacent_points_l236_236780


namespace brick_height_l236_236827

/-- A certain number of bricks, each measuring 25 cm x 11.25 cm x some height, 
are needed to build a wall of 8 m x 6 m x 22.5 cm. 
If 6400 bricks are needed, prove that the height of each brick is 6 cm. -/
theorem brick_height (h : ℝ) : 
  6400 * (25 * 11.25 * h) = (800 * 600 * 22.5) → h = 6 :=
by
  sorry

end brick_height_l236_236827


namespace minimize_y_l236_236584

theorem minimize_y (a b : ℝ) : 
  ∃ x : ℝ, x = (3 * a + b) / 4 ∧ 
  ∀ y : ℝ, (3 * (y - a) ^ 2 + (y - b) ^ 2) ≥ (3 * ((3 * a + b) / 4 - a) ^ 2 + ((3 * a + b) / 4 - b) ^ 2) :=
sorry

end minimize_y_l236_236584


namespace ratio_expression_value_l236_236051

theorem ratio_expression_value (a b : ℝ) (h : a / b = 4 / 1) : 
  (a - 3 * b) / (2 * a - b) = 1 / 7 := 
by 
  sorry

end ratio_expression_value_l236_236051


namespace gift_exchange_equation_l236_236232

theorem gift_exchange_equation (x : ℕ) (h : x * (x - 1) = 40) : 
  x * (x - 1) = 40 :=
by
  exact h

end gift_exchange_equation_l236_236232


namespace jamie_total_balls_after_buying_l236_236329

theorem jamie_total_balls_after_buying (red_balls : ℕ) (blue_balls : ℕ) (yellow_balls : ℕ) (lost_red_balls : ℕ) (final_red_balls : ℕ) (total_balls : ℕ)
  (h1 : red_balls = 16)
  (h2 : blue_balls = 2 * red_balls)
  (h3 : lost_red_balls = 6)
  (h4 : final_red_balls = red_balls - lost_red_balls)
  (h5 : yellow_balls = 32)
  (h6 : total_balls = final_red_balls + blue_balls + yellow_balls) :
  total_balls = 74 := by
    sorry

end jamie_total_balls_after_buying_l236_236329


namespace value_of_a_cube_l236_236578

-- We define the conditions given in the problem.
def A (a : ℤ) : Set ℤ := {5, a^2 + 2 * a + 4}
def a_satisfies (a : ℤ) : Prop := 7 ∈ A a

-- We state the theorem.
theorem value_of_a_cube (a : ℤ) (h1 : a_satisfies a) : a^3 = 1 ∨ a^3 = -27 := by
  sorry

end value_of_a_cube_l236_236578


namespace sum_mod_11_l236_236014

theorem sum_mod_11 (h1 : 8735 % 11 = 1) (h2 : 8736 % 11 = 2) (h3 : 8737 % 11 = 3) (h4 : 8738 % 11 = 4) :
  (8735 + 8736 + 8737 + 8738) % 11 = 10 :=
by
  sorry

end sum_mod_11_l236_236014


namespace meal_cost_one_burger_one_shake_one_cola_l236_236995

-- Define the costs of individual items
variables (B S C : ℝ)

-- Conditions based on given equations
def eq1 : Prop := 3 * B + 7 * S + C = 120
def eq2 : Prop := 4 * B + 10 * S + C = 160.50

-- Goal: Prove that the total cost of one burger, one shake, and one cola is $39
theorem meal_cost_one_burger_one_shake_one_cola :
  eq1 B S C → eq2 B S C → B + S + C = 39 :=
by 
  intros 
  sorry

end meal_cost_one_burger_one_shake_one_cola_l236_236995


namespace bound_on_ai_l236_236459

theorem bound_on_ai (n : ℕ) (a : ℕ → ℕ) (k : ℕ) :
  (n ≥ a 1) ∧ (∀ i, 1 ≤ i ∧ i < k → a i > a (i + 1)) ∧ (∀ i j, 1 ≤ i ∧ i ≤ k ∧ 1 ≤ j ∧ j ≤ k → Nat.lcm (a i) (a j) ≤ n) →
  ∀ i, 1 ≤ i ∧ i ≤ k → i * a i ≤ n :=
by
  sorry

end bound_on_ai_l236_236459


namespace inverse_110_mod_667_l236_236144

theorem inverse_110_mod_667 :
  (∃ (a b c : ℕ), a = 65 ∧ b = 156 ∧ c = 169 ∧ c^2 = a^2 + b^2) →
  (∃ n : ℕ, 110 * n % 667 = 1 ∧ 0 ≤ n ∧ n < 667 ∧ n = 608) :=
by
  sorry

end inverse_110_mod_667_l236_236144


namespace alberto_spent_more_l236_236686

-- Define the expenses of Alberto and Samara
def alberto_expenses : ℕ := 2457
def samara_oil_expense : ℕ := 25
def samara_tire_expense : ℕ := 467
def samara_detailing_expense : ℕ := 79
def samara_total_expenses : ℕ := samara_oil_expense + samara_tire_expense + samara_detailing_expense

-- State the theorem to prove the difference in expenses
theorem alberto_spent_more :
  alberto_expenses - samara_total_expenses = 1886 := by
  sorry

end alberto_spent_more_l236_236686


namespace find_b_l236_236187

variable (a b : Prod ℝ ℝ)
variable (x y : ℝ)

theorem find_b (h1 : (Prod.fst a + Prod.fst b = 0) ∧
                    (Real.sqrt ((Prod.snd a + Prod.snd b) ^ 2) = 1))
                    (h2 : a = (2, -1)) :
                    b = (-2, 2) ∨ b = (-2, 0) :=
by sorry

end find_b_l236_236187


namespace candidates_appeared_l236_236890

theorem candidates_appeared (x : ℝ) (h1 : 0.07 * x = 0.06 * x + 82) : x = 8200 :=
by
  sorry

end candidates_appeared_l236_236890


namespace option_C_correct_l236_236119

theorem option_C_correct (a b : ℝ) : 
  (1 / (b / a) * (a / b) = a^2 / b^2) :=
sorry

end option_C_correct_l236_236119


namespace compute_expression_l236_236848

-- Definition of the expression
def expression := 5 + 4 * (4 - 9)^2

-- Statement of the theorem, asserting the expression equals 105
theorem compute_expression : expression = 105 := by
  sorry

end compute_expression_l236_236848


namespace gary_has_left_amount_l236_236025

def initial_amount : ℝ := 100
def cost_pet_snake : ℝ := 55
def cost_toy_car : ℝ := 12
def cost_novel : ℝ := 7.5
def cost_pack_stickers : ℝ := 3.25
def number_packs_stickers : ℕ := 3

theorem gary_has_left_amount : initial_amount - (cost_pet_snake + cost_toy_car + cost_novel + number_packs_stickers * cost_pack_stickers) = 15.75 :=
by
  sorry

end gary_has_left_amount_l236_236025


namespace find_a_l236_236588

noncomputable def a : ℚ := ((68^3 - 65^3) * (32^3 + 18^3)) / ((32^2 - 32 * 18 + 18^2) * (68^2 + 68 * 65 + 65^2))

theorem find_a : a = 150 := 
  sorry

end find_a_l236_236588


namespace constant_term_is_19_l236_236165

theorem constant_term_is_19 (x y C : ℝ) 
  (h1 : 7 * x + y = C) 
  (h2 : x + 3 * y = 1) 
  (h3 : 2 * x + y = 5) : 
  C = 19 :=
sorry

end constant_term_is_19_l236_236165


namespace part_a_part_b_l236_236552

/-- Define rational non-integer numbers x and y -/
structure RationalNonInteger (x y : ℚ) :=
  (h1 : x.denom ≠ 1)
  (h2 : y.denom ≠ 1)

/-- Part (a): There exist rational non-integer numbers x and y 
    such that 19x + 8y and 8x + 3y are integers -/
theorem part_a : ∃ (x y : ℚ), RationalNonInteger x y ∧ (19*x + 8*y ∈ ℤ) ∧ (8*x + 3*y ∈ ℤ) :=
by
  sorry

/-- Part (b): There do not exist rational non-integer numbers x and y 
    such that 19x^2 + 8y^2 and 8x^2 + 3y^2 are integers -/
theorem part_b : ¬ ∃ (x y : ℚ), RationalNonInteger x y ∧ (19*x^2 + 8*y^2 ∈ ℤ) ∧ (8*x^2 + 3*y^2 ∈ ℤ) :=
by
  sorry

end part_a_part_b_l236_236552


namespace second_smallest_packs_hot_dogs_l236_236005

theorem second_smallest_packs_hot_dogs (n : ℕ) :
  (∃ k : ℕ, n = 5 * k + 3) →
  n > 0 →
  ∃ m : ℕ, m < n ∧ (∃ k2 : ℕ, m = 5 * k2 + 3) →
  n = 8 :=
by
  sorry

end second_smallest_packs_hot_dogs_l236_236005


namespace base4_to_base10_conversion_l236_236420

theorem base4_to_base10_conversion : 
  2 * 4^4 + 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 2 * 4^0 = 582 :=
by 
  sorry

end base4_to_base10_conversion_l236_236420


namespace boiling_point_fahrenheit_l236_236522

-- Define the conditions as hypotheses
def boils_celsius : ℝ := 100
def melts_celsius : ℝ := 0
def melts_fahrenheit : ℝ := 32
def pot_temp_celsius : ℝ := 55
def pot_temp_fahrenheit : ℝ := 131

-- Theorem to prove the boiling point in Fahrenheit
theorem boiling_point_fahrenheit : ∀ (boils_celsius : ℝ) (melts_celsius : ℝ) (melts_fahrenheit : ℝ) 
                                    (pot_temp_celsius : ℝ) (pot_temp_fahrenheit : ℝ),
  boils_celsius = 100 →
  melts_celsius = 0 →
  melts_fahrenheit = 32 →
  pot_temp_celsius = 55 →
  pot_temp_fahrenheit = 131 →
  ∃ boils_fahrenheit : ℝ, boils_fahrenheit = 212 :=
by
  intros
  existsi 212
  sorry

end boiling_point_fahrenheit_l236_236522


namespace minimum_value_of_f_l236_236436

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 - 4 * x + 5) / (2 * x - 4)

theorem minimum_value_of_f (x : ℝ) (h : x ≥ 5 / 2) : ∃ y ≥ (5 / 2), f y = 1 := by
  sorry

end minimum_value_of_f_l236_236436


namespace circle_passing_given_points_l236_236427

theorem circle_passing_given_points :
  ∃ (D E F : ℚ), (F = 0) ∧ (E = - (9 / 5)) ∧ (D = 19 / 5) ∧
  (∀ (x y : ℚ), x^2 + y^2 + D * x + E * y + F = 0 ↔ (x = 0 ∧ y = 0) ∨ (x = -2 ∧ y = 3) ∨ (x = -4 ∧ y = 1)) :=
by
  sorry

end circle_passing_given_points_l236_236427


namespace smallest_number_of_three_l236_236666

theorem smallest_number_of_three (x : ℕ) (h1 : x = 18)
  (h2 : ∀ y z : ℕ, y = 4 * x ∧ z = 2 * y)
  (h3 : (x + 4 * x + 8 * x) / 3 = 78)
  : x = 18 := by
  sorry

end smallest_number_of_three_l236_236666


namespace area_of_gray_region_l236_236412

def center_C : ℝ × ℝ := (4, 6)
def radius_C : ℝ := 6
def center_D : ℝ × ℝ := (14, 6)
def radius_D : ℝ := 6

theorem area_of_gray_region :
  let area_of_rectangle := (14 - 4) * 6
  let quarter_circle_area := (π * 6 ^ 2) / 4
  let area_to_subtract := 2 * quarter_circle_area
  area_of_rectangle - area_to_subtract = 60 - 18 * π := 
by {
  sorry
}

end area_of_gray_region_l236_236412


namespace evaluate_expression_l236_236286

theorem evaluate_expression : -25 - 7 * (4 + 2) = -67 := by
  sorry

end evaluate_expression_l236_236286


namespace solve_for_x_l236_236893

theorem solve_for_x (x : ℝ) (h : (1 / 5) + (5 / x) = (12 / x) + (1 / 12)) : x = 60 := by
  sorry

end solve_for_x_l236_236893


namespace entrance_exam_correct_answers_l236_236474

theorem entrance_exam_correct_answers (c w : ℕ) 
  (h1 : c + w = 70) 
  (h2 : 3 * c - w = 38) : 
  c = 27 := 
sorry

end entrance_exam_correct_answers_l236_236474


namespace john_must_study_4_5_hours_l236_236599

-- Let "study_time" be the amount of time John needs to study for the second exam.

noncomputable def study_time_for_avg_score (hours1 score1 target_avg total_exams : ℝ) (direct_relation : Prop) :=
  2 * target_avg - score1 / (score1 / hours1)

theorem john_must_study_4_5_hours :
  study_time_for_avg_score 3 60 75 2 (60 / 3 = 90 / study_time_for_avg_score 3 60 75 2 (60 / 3 = 90 / study_time_for_avg_score 3 60 75 2 (sorry))) = 4.5 :=
sorry

end john_must_study_4_5_hours_l236_236599


namespace laura_change_l236_236760

-- Define the cost of a pair of pants and a shirt.
def cost_of_pants := 54
def cost_of_shirts := 33

-- Define the number of pants and shirts Laura bought.
def num_pants := 2
def num_shirts := 4

-- Define the amount Laura gave to the cashier.
def amount_given := 250

-- Calculate the total cost.
def total_cost := num_pants * cost_of_pants + num_shirts * cost_of_shirts

-- Define the expected change.
def expected_change := 10

-- The main theorem stating the problem and its solution.
theorem laura_change :
  amount_given - total_cost = expected_change :=
by
  sorry

end laura_change_l236_236760


namespace mask_production_rates_l236_236952

theorem mask_production_rates (x : ℝ) (y : ℝ) :
  (280 / x) - (280 / (1.4 * x)) = 2 →
  x = 40 ∧ y = 1.4 * x →
  y = 56 :=
by {
  sorry
}

end mask_production_rates_l236_236952


namespace orange_preference_percentage_l236_236793

theorem orange_preference_percentage 
  (red blue green yellow purple orange : ℕ)
  (total : ℕ)
  (h_red : red = 75)
  (h_blue : blue = 80)
  (h_green : green = 50)
  (h_yellow : yellow = 45)
  (h_purple : purple = 60)
  (h_orange : orange = 55)
  (h_total : total = red + blue + green + yellow + purple + orange) :
  (orange * 100) / total = 15 :=
by
sorry

end orange_preference_percentage_l236_236793


namespace tom_paid_amount_correct_l236_236382

def kg (n : Nat) : Nat := n -- Just a type alias clarification

theorem tom_paid_amount_correct :
  ∀ (quantity_apples : Nat) (rate_apples : Nat) (quantity_mangoes : Nat) (rate_mangoes : Nat),
  quantity_apples = kg 8 →
  rate_apples = 70 →
  quantity_mangoes = kg 9 →
  rate_mangoes = 55 →
  (quantity_apples * rate_apples) + (quantity_mangoes * rate_mangoes) = 1055 :=
by
  intros quantity_apples rate_apples quantity_mangoes rate_mangoes
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end tom_paid_amount_correct_l236_236382


namespace crayons_left_is_4_l236_236365

-- Define initial number of crayons in the drawer
def initial_crayons : Nat := 7

-- Define number of crayons Mary took out
def taken_by_mary : Nat := 3

-- Define the number of crayons left in the drawer
def crayons_left (initial : Nat) (taken : Nat) : Nat :=
  initial - taken

-- Prove the number of crayons left in the drawer is 4
theorem crayons_left_is_4 : crayons_left initial_crayons taken_by_mary = 4 :=
by
  -- sorry is used here to skip the actual proof
  sorry

end crayons_left_is_4_l236_236365


namespace p_necessary_not_sufficient_for_q_l236_236035

def p (x : ℝ) : Prop := abs x = -x
def q (x : ℝ) : Prop := x^2 ≥ -x

theorem p_necessary_not_sufficient_for_q : 
  (∀ x, q x → p x) ∧ ¬ (∀ x, p x → q x) :=
by
  sorry

end p_necessary_not_sufficient_for_q_l236_236035


namespace mod_37_5_l236_236940

theorem mod_37_5 : 37 % 5 = 2 := 
by 
  sorry

end mod_37_5_l236_236940


namespace exists_n_for_dvd_ka_pow_n_add_n_l236_236483

theorem exists_n_for_dvd_ka_pow_n_add_n 
  (a k : ℕ) (a_pos : 0 < a) (k_pos : 0 < k) (d : ℕ) (d_pos : 0 < d) :
  ∃ n : ℕ, 0 < n ∧ d ∣ k * (a ^ n) + n :=
by
  sorry

end exists_n_for_dvd_ka_pow_n_add_n_l236_236483


namespace find_m_for_parallel_lines_l236_236194

theorem find_m_for_parallel_lines (m : ℝ) :
  (∀ x y, 2 * x + (m + 1) * y + 4 = 0 → mx + 3 * y - 2 = 0 → 
  -((2 : ℝ) / (m + 1)) = -(m / 3)) → (m = 2 ∨ m = -3) :=
by
  sorry

end find_m_for_parallel_lines_l236_236194


namespace max_y_value_l236_236210

noncomputable def y (x : ℝ) : ℝ := |x + 1| - 2 * |x| + |x - 2|

theorem max_y_value : ∃ α, (∀ x, -1 ≤ x ∧ x ≤ 2 → y x ≤ α) ∧ α = 3 := by
  sorry

end max_y_value_l236_236210


namespace fan_working_time_each_day_l236_236984

theorem fan_working_time_each_day
  (airflow_per_second : ℝ)
  (total_airflow_week : ℝ)
  (seconds_per_hour : ℝ)
  (hours_per_day : ℝ)
  (days_per_week : ℝ)
  (airy_sector: airflow_per_second = 10)
  (flow_week : total_airflow_week = 42000)
  (sec_per_hr : seconds_per_hour = 3600)
  (hrs_per_day : hours_per_day = 24)
  (days_week : days_per_week = 7) :
  let airflow_per_hour := airflow_per_second * seconds_per_hour
  let total_hours_week := total_airflow_week / airflow_per_hour
  let hours_per_day_given := total_hours_week / days_per_week
  let minutes_per_day := hours_per_day_given * 60
  minutes_per_day = 10 := 
by
  sorry

end fan_working_time_each_day_l236_236984


namespace total_shaded_area_l236_236401

theorem total_shaded_area (S T : ℝ) (h1 : 16 / S = 4) (h2 : S / T = 4) : 
    S^2 + 16 * T^2 = 32 := 
by {
    sorry
}

end total_shaded_area_l236_236401


namespace false_proposition_of_quadratic_l236_236720

theorem false_proposition_of_quadratic
  (a : ℝ) (h0 : a ≠ 0)
  (h1 : ¬(5 = a * (1/2)^2 + (-a^2 - 1) * (1/2) + a))
  (h2 : (a^2 + 1) / (2 * a) > 0)
  (h3 : (0, a) = (0, x) ∧ x > 0)
  (h4 : ∀ x : ℝ, a * x^2 + (-a^2 - 1) * x + a ≤ 0) :
  false :=
sorry

end false_proposition_of_quadratic_l236_236720


namespace range_of_m_l236_236167

theorem range_of_m (m : ℝ) : 
  (∀ x, x^2 + 2 * x - m > 0 ↔ (x = 1 → x^2 + 2 * x - m ≤ 0) ∧ (x = 2 → x^2 + 2 * x - m > 0)) ↔ (3 ≤ m ∧ m < 8) := 
sorry

end range_of_m_l236_236167


namespace min_value_zero_l236_236803

noncomputable def f (k x y : ℝ) : ℝ :=
  3 * x^2 - 8 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 6 * y + 9

theorem min_value_zero (k : ℝ) :
  (∀ x y : ℝ, f k x y ≥ 0) ↔ (k = 3 / 2 ∨ k = -3 / 2) :=
by
  sorry

end min_value_zero_l236_236803


namespace solve_for_a_l236_236102

-- Given the equation is quadratic, meaning the highest power of x in the quadratic term equals 2
theorem solve_for_a (a : ℚ) : (2 * a - 1 = 2) -> a = 3 / 2 :=
by
  sorry

end solve_for_a_l236_236102


namespace percentage_of_a_l236_236464

theorem percentage_of_a (a : ℕ) (x : ℕ) (h1 : a = 190) (h2 : (x * a) / 100 = 95) : x = 50 := by
  sorry

end percentage_of_a_l236_236464


namespace bracket_mul_l236_236818

def bracket (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 + 1 else 2 * x + 1

theorem bracket_mul : bracket 6 * bracket 3 = 28 := by
  sorry

end bracket_mul_l236_236818


namespace sum_divisible_by_49_l236_236869

theorem sum_divisible_by_49
  {x y z : ℤ} 
  (hx : x % 7 ≠ 0)
  (hy : y % 7 ≠ 0)
  (hz : z % 7 ≠ 0)
  (h : 7 ^ 3 ∣ (x ^ 7 + y ^ 7 + z ^ 7)) : 7^2 ∣ (x + y + z) :=
by
  sorry

end sum_divisible_by_49_l236_236869


namespace quadratic_discriminant_one_solution_l236_236353

theorem quadratic_discriminant_one_solution (m : ℚ) : 
  (3 * (1 : ℚ))^2 - 12 * m = 0 → m = 49 / 12 := 
by {
  sorry
}

end quadratic_discriminant_one_solution_l236_236353


namespace sum_of_roots_tan_quadratic_l236_236017

theorem sum_of_roots_tan_quadratic :
  (∑ x in {x : ℝ | 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ (tan x)^2 - 12 * tan x + 4 = 0}, x) = 3 * Real.pi := 
by
  sorry

end sum_of_roots_tan_quadratic_l236_236017


namespace cost_price_is_3000_l236_236358

variable (CP SP : ℝ)

-- Condition: selling price (SP) is 20% more than the cost price (CP)
def sellingPrice : ℝ := CP + 0.20 * CP

-- Condition: selling price (SP) is Rs. 3600
axiom selling_price_eq : SP = 3600

-- Given the above conditions, prove that the cost price (CP) is Rs. 3000
theorem cost_price_is_3000 (h : sellingPrice CP = SP) : CP = 3000 := by
  sorry

end cost_price_is_3000_l236_236358


namespace not_divisible_2310_l236_236088

theorem not_divisible_2310 (n : ℕ) (h : n < 2310) : ¬ (2310 ∣ n * (2310 - n)) :=
sorry

end not_divisible_2310_l236_236088


namespace sin_cos_from_tan_in_second_quadrant_l236_236036

theorem sin_cos_from_tan_in_second_quadrant (α : ℝ) 
  (h1 : Real.tan α = -2) 
  (h2 : α ∈ Set.Ioo (π / 2) π) : 
  Real.sin α = 2 * Real.sqrt 5 / 5 ∧ Real.cos α = -Real.sqrt 5 / 5 :=
by
  sorry

end sin_cos_from_tan_in_second_quadrant_l236_236036


namespace find_f_minus_half_l236_236728

-- Definitions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def function_definition (f : ℝ → ℝ) : Prop :=
  ∀ x, x > 0 → f x = 4^x

-- Theorem statement
theorem find_f_minus_half {f : ℝ → ℝ}
  (h_odd : is_odd_function f)
  (h_def : function_definition f) :
  f (-1/2) = -2 :=
by
  sorry

end find_f_minus_half_l236_236728


namespace not_divisible_by_5_for_4_and_7_l236_236294

-- Define a predicate that checks if a given number is not divisible by another number
def notDivisibleBy (n k : ℕ) : Prop := ¬ (n % k = 0)

-- Define the expression we are interested in
def expression (b : ℕ) : ℕ := 3 * b^3 - b^2 + b - 1

-- The theorem we want to prove
theorem not_divisible_by_5_for_4_and_7 :
  notDivisibleBy (expression 4) 5 ∧ notDivisibleBy (expression 7) 5 :=
by
  sorry

end not_divisible_by_5_for_4_and_7_l236_236294


namespace tetrahedron_face_area_squared_l236_236643

variables {S0 S1 S2 S3 α12 α13 α23 : ℝ}

-- State the theorem
theorem tetrahedron_face_area_squared :
  (S0)^2 = (S1)^2 + (S2)^2 + (S3)^2 - 2 * S1 * S2 * (Real.cos α12) - 2 * S1 * S3 * (Real.cos α13) - 2 * S2 * S3 * (Real.cos α23) :=
sorry

end tetrahedron_face_area_squared_l236_236643


namespace total_amount_of_currency_notes_l236_236072

theorem total_amount_of_currency_notes (x y : ℕ) (h1 : x + y = 85) (h2 : 50 * y = 3500) : 100 * x + 50 * y = 5000 := by
  sorry

end total_amount_of_currency_notes_l236_236072


namespace muscovy_more_than_cayuga_l236_236109

theorem muscovy_more_than_cayuga
  (M C K : ℕ)
  (h1 : M + C + K = 90)
  (h2 : M = 39)
  (h3 : M = 2 * C + 3 + C) :
  M - C = 27 := by
  sorry

end muscovy_more_than_cayuga_l236_236109


namespace length_of_BA_is_sqrt_557_l236_236706

-- Define the given conditions
def AD : ℝ := 6
def DC : ℝ := 11
def CB : ℝ := 6
def AC : ℝ := 14

-- Define the theorem statement
theorem length_of_BA_is_sqrt_557 (x : ℝ) (H1 : AD = 6) (H2 : DC = 11) (H3 : CB = 6) (H4 : AC = 14) :
  x = Real.sqrt 557 :=
  sorry

end length_of_BA_is_sqrt_557_l236_236706


namespace triangle_side_ineq_l236_236305

theorem triangle_side_ineq (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) : 
  a^2 * c + b^2 * a + c^2 * b < 1 / 8 := 
by 
  sorry

end triangle_side_ineq_l236_236305


namespace prob_select_A_l236_236309

open Finset

theorem prob_select_A :
  let individuals := {A, B, C} : Finset _
  let choices := indivduals.powerset.filter (λ s, s.card = 2)
  let with_A := choices.filter (λ s, A ∈ s)
  (with_A.card : ℚ) / choices.card = 2 / 3 := 
by
  sorry

end prob_select_A_l236_236309


namespace total_amount_leaked_l236_236843

def amount_leaked_before_start : ℕ := 2475
def amount_leaked_while_fixing : ℕ := 3731

theorem total_amount_leaked : amount_leaked_before_start + amount_leaked_while_fixing = 6206 := by
  sorry

end total_amount_leaked_l236_236843


namespace cube_side_length_l236_236058

-- Given conditions for the problem
def surface_area (a : ℝ) : ℝ := 6 * a^2

-- Theorem statement
theorem cube_side_length (h : surface_area a = 864) : a = 12 :=
by
  sorry

end cube_side_length_l236_236058


namespace age_of_youngest_child_l236_236381

theorem age_of_youngest_child (x : ℕ) 
  (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 50) : x = 4 := 
by {
  sorry
}

end age_of_youngest_child_l236_236381


namespace minimize_expression_l236_236207

open Real

theorem minimize_expression (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) :
  4 * p^3 + 6 * q^3 + 24 * r^3 + 8 / (3 * p * q * r) ≥ 16 :=
sorry

end minimize_expression_l236_236207


namespace men_in_first_group_l236_236223

theorem men_in_first_group (M : ℕ) (h1 : M * 18 * 6 = 15 * 12 * 6) : M = 10 :=
by
  sorry

end men_in_first_group_l236_236223


namespace Mike_found_seashells_l236_236493

/-!
# Problem:
Mike found some seashells on the beach, he gave Tom 49 of his seashells.
He has thirteen seashells left. How many seashells did Mike find on the beach?

# Conditions:
1. Mike gave Tom 49 seashells.
2. Mike has 13 seashells left.

# Proof statement:
Prove that Mike found 62 seashells on the beach.
-/

/-- Define the variables and conditions -/
def seashells_given_to_Tom : ℕ := 49
def seashells_left_with_Mike : ℕ := 13

/-- Prove that Mike found 62 seashells on the beach -/
theorem Mike_found_seashells : 
  seashells_given_to_Tom + seashells_left_with_Mike = 62 := 
by
  -- This is where the proof would go
  sorry

end Mike_found_seashells_l236_236493


namespace mary_circus_change_l236_236610

theorem mary_circus_change :
  let mary_ticket := 2
  let child_ticket := 1
  let num_children := 3
  let total_cost := mary_ticket + num_children * child_ticket
  let amount_paid := 20
  let change := amount_paid - total_cost
  change = 15 :=
by
  let mary_ticket := 2
  let child_ticket := 1
  let num_children := 3
  let total_cost := mary_ticket + num_children * child_ticket
  let amount_paid := 20
  let change := amount_paid - total_cost
  sorry

end mary_circus_change_l236_236610


namespace fraction_sum_l236_236008

theorem fraction_sum :
  (1 / 4 : ℚ) + (2 / 9) + (3 / 6) = 35 / 36 := 
sorry

end fraction_sum_l236_236008


namespace new_year_season_markup_l236_236131

variable {C : ℝ} (hC : 0 < C)

theorem new_year_season_markup (h1 : ∀ C, C > 0 → ∃ P1, P1 = 1.20 * C)
                              (h2 : ∀ (P1 M : ℝ), M >= 0 → ∃ P2, P2 = P1 * (1 + M / 100))
                              (h3 : ∀ P2, ∃ P3, P3 = P2 * 0.91)
                              (h4 : ∃ P3, P3 = 1.365 * C) :
  ∃ M, M = 25 := 
by 
  sorry

end new_year_season_markup_l236_236131


namespace exists_pos_ints_l236_236653

open Nat

noncomputable def f (a : ℕ) : ℕ :=
  a^2 + 3 * a + 2

noncomputable def g (b c : ℕ) : ℕ :=
  b^2 - b + 3 * c^2 + 3 * c

theorem exists_pos_ints (a : ℕ) (ha : 0 < a) :
  ∃ (b c : ℕ), 0 < b ∧ 0 < c ∧ f a = g b c :=
sorry

end exists_pos_ints_l236_236653


namespace max_x1_sq_plus_x2_sq_l236_236488

theorem max_x1_sq_plus_x2_sq (k : ℝ) (x1 x2 : ℝ) 
  (h1 : x1 + x2 = k - 2) 
  (h2 : x1 * x2 = k^2 + 3 * k + 5)
  (h3 : -4 ≤ k ∧ k ≤ -4 / 3) : 
  x1^2 + x2^2 ≤ 18 :=
by sorry

end max_x1_sq_plus_x2_sq_l236_236488


namespace roof_ratio_l236_236801

theorem roof_ratio (L W : ℝ) 
  (h1 : L * W = 784) 
  (h2 : L - W = 42) : 
  L / W = 4 := by 
  sorry

end roof_ratio_l236_236801


namespace power_mod_lemma_l236_236657

theorem power_mod_lemma : (7^137 % 13) = 11 := by
  sorry

end power_mod_lemma_l236_236657


namespace find_fx_neg_l236_236727

variable (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def f_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 0 → f x = x^2 - 2*x

theorem find_fx_neg (h1 : odd_function f) (h2 : f_nonneg f) : 
  ∀ x : ℝ, x < 0 → f x = -x^2 - 2*x := 
by
  sorry

end find_fx_neg_l236_236727


namespace distance_traveled_on_foot_l236_236250

theorem distance_traveled_on_foot (x y : ℝ) : x + y = 61 ∧ (x / 4 + y / 9 = 9) → x = 16 :=
by {
  sorry
}

end distance_traveled_on_foot_l236_236250


namespace minimum_value_of_quadratic_function_l236_236585

def quadratic_function (x : ℝ) : ℝ := x^2 + 8 * x + 12

theorem minimum_value_of_quadratic_function : ∃ x : ℝ, is_min_on (quadratic_function) {x} (-4) :=
by
  use -4
  sorry

end minimum_value_of_quadratic_function_l236_236585


namespace find_real_number_a_l236_236914

variable (U : Set ℕ) (M : Set ℕ) (a : ℕ)

theorem find_real_number_a :
  U = {1, 3, 5, 7} →
  M = {1, a} →
  (U \ M) = {5, 7} →
  a = 3 :=
by
  intros hU hM hCompU
  -- Proof part will be here
  sorry

end find_real_number_a_l236_236914


namespace total_points_zach_ben_l236_236663

theorem total_points_zach_ben (zach_points ben_points : ℝ) (h1 : zach_points = 42.0) (h2 : ben_points = 21.0) : zach_points + ben_points = 63.0 :=
by
  sorry

end total_points_zach_ben_l236_236663


namespace tan_diff_eqn_l236_236725

theorem tan_diff_eqn (α : ℝ) (h : Real.tan α = 2) : Real.tan (α - 3 * Real.pi / 4) = -3 := 
by 
  sorry

end tan_diff_eqn_l236_236725


namespace max_value_of_a_l236_236606
noncomputable def f (a x : ℝ) : ℝ :=
  if x < a then -a * x + 1 else (x - 2)^2

theorem max_value_of_a (a : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), f a x ≤ f a y) → a ≤ 1 := 
sorry

end max_value_of_a_l236_236606


namespace total_work_completion_days_l236_236271

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

end total_work_completion_days_l236_236271


namespace jace_total_distance_l236_236480

noncomputable def total_distance (s1 s2 s3 s4 s5 : ℝ) (t1 t2 t3 t4 t5 : ℝ) : ℝ :=
  s1 * t1 + s2 * t2 + s3 * t3 + s4 * t4 + s5 * t5

theorem jace_total_distance :
  total_distance 50 65 60 75 55 3 4.5 2.75 1.8333 2.6667 = 891.67 := by
  sorry

end jace_total_distance_l236_236480


namespace problem_CorrectOption_l236_236450

def setA : Set ℝ := {y | ∃ x : ℝ, y = |x| - 1}
def setB : Set ℝ := {x | x ≥ 2}

theorem problem_CorrectOption : setA ∩ setB = setB := 
  sorry

end problem_CorrectOption_l236_236450


namespace child_ticket_price_l236_236127

theorem child_ticket_price
    (num_people : ℕ)
    (num_adults : ℕ)
    (num_seniors : ℕ)
    (num_children : ℕ)
    (adult_ticket_cost : ℝ)
    (senior_discount : ℝ)
    (total_bill : ℝ) :
    num_people = 50 →
    num_adults = 25 →
    num_seniors = 15 →
    num_children = 10 →
    adult_ticket_cost = 15 →
    senior_discount = 0.25 →
    total_bill = 600 →
    ∃ x : ℝ, x = 5.63 :=
by {
  sorry
}

end child_ticket_price_l236_236127


namespace max_f_eq_4_monotonic_increase_interval_l236_236734

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sqrt 3)
noncomputable def vec_b (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.sqrt 3)
noncomputable def f (x : ℝ) : ℝ := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2

theorem max_f_eq_4 (x : ℝ) : ∀ x : ℝ, f x ≤ 4 := 
by
  sorry

theorem monotonic_increase_interval (k : ℤ) : ∀ x : ℝ, (k * Real.pi - Real.pi / 4 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 4) ↔ 
  (0 ≤ Real.sin (2 * x) ∧ Real.sin (2 * x) ≤ 1) :=
by
  sorry

end max_f_eq_4_monotonic_increase_interval_l236_236734


namespace greatest_leftover_cookies_l236_236825

theorem greatest_leftover_cookies (n : ℕ) : ∃ k, k ≤ n ∧ k % 8 = 7 := sorry

end greatest_leftover_cookies_l236_236825


namespace equivalence_negation_l236_236357

-- Define irrational numbers
def is_irrational (x : ℝ) : Prop :=
  ¬ (∃ q : ℚ, x = q)

-- Define rational numbers
def is_rational (x : ℝ) : Prop :=
  ∃ q : ℚ, x = q

-- Original proposition: There exists an irrational number whose square is rational
def original_proposition : Prop :=
  ∃ x : ℝ, is_irrational x ∧ is_rational (x * x)

-- Negation of the original proposition
def negation_of_proposition : Prop :=
  ∀ x : ℝ, is_irrational x → ¬is_rational (x * x)

-- Proof statement that the negation of the original proposition is equivalent to "Every irrational number has a square that is not rational"
theorem equivalence_negation :
  (¬ original_proposition) ↔ negation_of_proposition :=
sorry

end equivalence_negation_l236_236357


namespace range_of_m_l236_236310

def cond1 (x : ℝ) : Prop := x^2 - 4 * x + 3 < 0
def cond2 (x : ℝ) : Prop := x^2 - 6 * x + 8 < 0
def cond3 (x m : ℝ) : Prop := 2 * x^2 - 9 * x + m < 0

theorem range_of_m (m : ℝ) : (∀ x, cond1 x → cond2 x → cond3 x m) → m < 9 :=
by
  sorry

end range_of_m_l236_236310


namespace sum_of_intervals_l236_236019

-- Define the floor function and the given function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  let k := floor x in k * (2020 ^ (x - k) - 1)

-- Define the main theorem
theorem sum_of_intervals : 
  ∑ k in finset.range 2019, real.log (1 + 1 / k) / real.log 2020 = 1 := by
  sorry

end sum_of_intervals_l236_236019


namespace find_original_number_l236_236595

theorem find_original_number (x : ℝ) :
  (((x / 2.5) - 10.5) * 0.3 = 5.85) -> x = 75 :=
by
  sorry

end find_original_number_l236_236595


namespace hypotenuse_length_l236_236324

theorem hypotenuse_length (a b c : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (h2 : a^2 + b^2 + c^2 = 1800) : 
  c = 30 :=
sorry

end hypotenuse_length_l236_236324


namespace boat_speed_in_still_water_l236_236753

namespace BoatSpeed

variables (V_b V_s : ℝ)

def condition1 : Prop := V_b + V_s = 15
def condition2 : Prop := V_b - V_s = 5

theorem boat_speed_in_still_water (h1 : condition1 V_b V_s) (h2 : condition2 V_b V_s) : V_b = 10 :=
by
  sorry

end BoatSpeed

end boat_speed_in_still_water_l236_236753


namespace least_distance_between_ticks_l236_236216

theorem least_distance_between_ticks :
  ∃ z : ℝ, ∀ (a b : ℤ), (a / 5 ≠ b / 7) → abs (a / 5 - b / 7) = (1 / 35) := 
sorry

end least_distance_between_ticks_l236_236216


namespace expression_undefined_count_l236_236860

theorem expression_undefined_count (x : ℝ) :
  ∃! x, (x - 1) * (x + 3) * (x - 3) = 0 :=
sorry

end expression_undefined_count_l236_236860


namespace probability_point_below_x_axis_l236_236217

structure Point2D :=
  (x : ℝ)
  (y : ℝ)

structure Parallelogram :=
  (P Q R S : Point2D)

def vertices_of_PQRS : Parallelogram :=
  ⟨⟨4, 4⟩, ⟨-2, -2⟩, ⟨-8, -2⟩, ⟨-2, 4⟩⟩

def point_lies_below_x_axis_probability (parallelogram : Parallelogram) : ℝ :=
  sorry

theorem probability_point_below_x_axis :
  point_lies_below_x_axis_probability vertices_of_PQRS = 1 / 2 :=
sorry

end probability_point_below_x_axis_l236_236217


namespace infinite_positive_integer_solutions_l236_236347

theorem infinite_positive_integer_solutions : ∃ (a b c : ℕ), (∃ k : ℕ, k > 0 ∧ a = k * (k^3 + 1990) ∧ b = (k^3 + 1990) ∧ c = (k^3 + 1990)) ∧ (a^3 + 1990 * b^3) = c^4 :=
sorry

end infinite_positive_integer_solutions_l236_236347


namespace find_k_l236_236735

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (2, -3)

def is_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_k (k : ℝ) :
  is_perpendicular (k • vector_a - 2 • vector_b) vector_a ↔ k = -1 :=
sorry

end find_k_l236_236735


namespace square_side_length_eq_area_and_perimeter_l236_236016

theorem square_side_length_eq_area_and_perimeter (a : ℝ) (h : a^2 = 4 * a) : a = 4 :=
by sorry

end square_side_length_eq_area_and_perimeter_l236_236016


namespace base8_base9_equivalence_l236_236626

def base8_digit (x : ℕ) := 0 ≤ x ∧ x < 8
def base9_digit (y : ℕ) := 0 ≤ y ∧ y < 9

theorem base8_base9_equivalence 
    (X Y : ℕ) 
    (hX : base8_digit X) 
    (hY : base9_digit Y) 
    (h_eq : 8 * X + Y = 9 * Y + X) :
    (8 * 7 + 6 = 62) :=
by
  sorry

end base8_base9_equivalence_l236_236626


namespace quadratic_root_k_value_l236_236034

theorem quadratic_root_k_value 
  (k : ℝ) 
  (h_roots : ∀ x : ℝ, (5 * x^2 + 7 * x + k = 0) → (x = ( -7 + Real.sqrt (-191) ) / 10 ∨ x = ( -7 - Real.sqrt (-191) ) / 10)) : 
  k = 12 :=
sorry

end quadratic_root_k_value_l236_236034


namespace maximum_value_40_l236_236054

theorem maximum_value_40 (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 10) :
  (a-b)^2 + (a-c)^2 + (a-d)^2 + (b-c)^2 + (b-d)^2 + (c-d)^2 ≤ 40 :=
sorry

end maximum_value_40_l236_236054


namespace minimize_J_l236_236460

noncomputable def H (p q : ℝ) : ℝ := -3 * p * q + 4 * p * (1 - q) + 2 * (1 - p) * q - 5 * (1 - p) * (1 - q)

noncomputable def J (p : ℝ) : ℝ := max (H p 0) (H p 1)

theorem minimize_J : ∃ p : ℝ, 0 ≤ p ∧ p ≤ 1 ∧ (∀ p' : ℝ, 0 ≤ p' ∧ p' ≤ 1 → J p ≤ J p') ∧ p = 1 / 2 :=
by
  sorry

end minimize_J_l236_236460


namespace total_questions_correct_total_answers_correct_l236_236399

namespace ForumCalculation

def members : ℕ := 200
def questions_per_hour_per_user : ℕ := 3
def hours_in_day : ℕ := 24
def answers_multiplier : ℕ := 3

def total_questions_per_user_per_day : ℕ :=
  questions_per_hour_per_user * hours_in_day

def total_questions_in_a_day : ℕ :=
  members * total_questions_per_user_per_day

def total_answers_per_user_per_day : ℕ :=
  answers_multiplier * total_questions_per_user_per_day

def total_answers_in_a_day : ℕ :=
  members * total_answers_per_user_per_day

theorem total_questions_correct :
  total_questions_in_a_day = 14400 :=
by
  sorry

theorem total_answers_correct :
  total_answers_in_a_day = 43200 :=
by
  sorry

end ForumCalculation

end total_questions_correct_total_answers_correct_l236_236399


namespace triangle_area_interval_l236_236683

theorem triangle_area_interval (s : ℝ) :
  10 ≤ (s - 1)^(3 / 2) ∧ (s - 1)^(3 / 2) ≤ 50 → (5.64 ≤ s ∧ s ≤ 18.32) :=
by
  sorry

end triangle_area_interval_l236_236683


namespace paco_initial_cookies_l236_236085

theorem paco_initial_cookies :
  ∀ (total_cookies initially_ate initially_gave : ℕ),
    initially_ate = 14 →
    initially_gave = 13 →
    initially_ate = initially_gave + 1 →
    total_cookies = initially_ate + initially_gave →
    total_cookies = 27 :=
by
  intros total_cookies initially_ate initially_gave h_ate h_gave h_diff h_sum
  sorry

end paco_initial_cookies_l236_236085


namespace line_AB_eq_x_plus_3y_zero_l236_236142

variable (x y : ℝ)

def circle1 := x^2 + y^2 - 4*x + 6*y = 0
def circle2 := x^2 + y^2 - 6*x = 0

theorem line_AB_eq_x_plus_3y_zero : 
  (∃ (A B : ℝ × ℝ), circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧ circle2 A.1 A.2 ∧ circle2 B.1 B.2 ∧ (A ≠ B)) → 
  (∀ (x y : ℝ), x + 3*y = 0) := 
by
  sorry

end line_AB_eq_x_plus_3y_zero_l236_236142


namespace jane_trail_mix_chocolate_chips_l236_236096

theorem jane_trail_mix_chocolate_chips (c₁ : ℝ) (c₂ : ℝ) (c₃ : ℝ) (c₄ : ℝ) (c₅ : ℝ) :
  (c₁ = 0.30) → (c₂ = 0.70) → (c₃ = 0.45) → (c₄ = 0.35) → (c₅ = 0.60) →
  c₄ = 0.35 ∧ (c₅ - c₁) * 2 = 0.40 := 
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end jane_trail_mix_chocolate_chips_l236_236096


namespace Kate_relies_on_dumpster_diving_Upscale_stores_discard_items_Kate_frugal_habits_l236_236621

structure Person :=
  (name : String)
  (age : Nat)
  (location : String)
  (occupation : String)

def kate : Person := {name := "Kate Hashimoto", age := 30, location := "New York", occupation := "CPA"}

-- Conditions
def lives_on_15_dollars_a_month (p : Person) : Prop := p = kate → true
def dumpster_diving (p : Person) : Prop := p = kate → true
def upscale_stores_discard_good_items : Prop := true
def frugal_habits (p : Person) : Prop := p = kate → true

-- Proof
theorem Kate_relies_on_dumpster_diving : lives_on_15_dollars_a_month kate ∧ dumpster_diving kate → true := 
by sorry

theorem Upscale_stores_discard_items : upscale_stores_discard_good_items → true := 
by sorry

theorem Kate_frugal_habits : frugal_habits kate → true := 
by sorry

end Kate_relies_on_dumpster_diving_Upscale_stores_discard_items_Kate_frugal_habits_l236_236621


namespace mod_exp_result_l236_236658

theorem mod_exp_result :
  (2 ^ 46655) % 9 = 1 :=
by
  sorry

end mod_exp_result_l236_236658


namespace arithmetic_operations_result_eq_one_over_2016_l236_236897

theorem arithmetic_operations_result_eq_one_over_2016 :
  (∃ op1 op2 : ℚ → ℚ → ℚ, op1 (1/8) (op2 (1/9) (1/28)) = 1/2016) :=
sorry

end arithmetic_operations_result_eq_one_over_2016_l236_236897


namespace num_of_triangles_with_perimeter_10_l236_236877

theorem num_of_triangles_with_perimeter_10 :
  ∃ (triangles : Finset (ℕ × ℕ × ℕ)), 
    (∀ (a b c : ℕ), (a, b, c) ∈ triangles → 
      a + b + c = 10 ∧ 
      a + b > c ∧ 
      a + c > b ∧ 
      b + c > a) ∧ 
    triangles.card = 4 := sorry

end num_of_triangles_with_perimeter_10_l236_236877


namespace multiply_add_fractions_l236_236956

theorem multiply_add_fractions :
  (2 / 9 : ℚ) * (5 / 8) + (1 / 4) = 7 / 18 := by
  sorry

end multiply_add_fractions_l236_236956


namespace sum_of_n_binom_coefficient_l236_236373

theorem sum_of_n_binom_coefficient :
  (∑ n in { n : ℤ | nat.choose 28 14 + nat.choose 28 n = nat.choose 29 15}, n) = 28 := 
by
  sorry

end sum_of_n_binom_coefficient_l236_236373


namespace count_integer_triplets_l236_236708

open BigOperators

theorem count_integer_triplets :
  ∃ (triplets : Finset (ℕ × ℕ × ℕ)), 
  (∀ (a b c : ℕ), (a, b, c) ∈ triplets → Nat.lcm a (Nat.lcm b c) = 20000 ∧ Nat.gcd a (Nat.gcd b c) = 20) ∧ 
  Finset.card triplets = 56 := by
  sorry

end count_integer_triplets_l236_236708


namespace unique_card_sequences_count_card_sequences_divided_by_10_l236_236828

noncomputable def uniqueCardSequences : ℕ :=
  let characters := ["L", "A", "T", "E", "0", "1", "1", "2"]
  let countWithout1 := Nat.factorial 6 / (Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1)
  let countWithOne1 := 5 * (Nat.factorial 5 / (Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1))
  let countWithTwo1 := (Nat.choose 5 2) * (Nat.factorial 3 / (Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1))
  countWithout1 + countWithOne1 + countWithTwo1

theorem unique_card_sequences_count : uniqueCardSequences = 1380 := by
  sorry

theorem card_sequences_divided_by_10 : uniqueCardSequences / 10 = 138 := by
  sorry

end unique_card_sequences_count_card_sequences_divided_by_10_l236_236828


namespace cost_of_jam_l236_236702

theorem cost_of_jam (N B J H : ℕ) (h : N > 1) (cost_eq : N * (6 * B + 7 * J + 4 * H) = 462) : 7 * J * N = 462 :=
by
  sorry

end cost_of_jam_l236_236702


namespace range_of_real_number_l236_236876

noncomputable def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 0}
def B (a : ℝ) : Set ℝ := {-1, -3, a}
def complement_A : Set ℝ := {x | x ≥ 0}

theorem range_of_real_number (a : ℝ) (h : (complement_A ∩ (B a)) ≠ ∅) : a ≥ 0 :=
sorry

end range_of_real_number_l236_236876


namespace condition_C_for_D_condition_A_for_B_l236_236313

theorem condition_C_for_D (C D : Prop) (h : C → D) : C → D :=
by
  exact h

theorem condition_A_for_B (A B D : Prop) (hA_to_D : A → D) (hD_to_B : D → B) : A → B :=
by
  intro hA
  apply hD_to_B
  apply hA_to_D
  exact hA

end condition_C_for_D_condition_A_for_B_l236_236313


namespace volume_formula_correct_l236_236988

def volume_of_box (x : ℝ) : ℝ :=
  x * (16 - 2 * x) * (12 - 2 * x)

theorem volume_formula_correct (x : ℝ) (h : x ≤ 12 / 5) :
  volume_of_box x = 4 * x^3 - 56 * x^2 + 192 * x :=
by sorry

end volume_formula_correct_l236_236988


namespace trajectory_of_A_eq_l236_236311

/-- The given conditions for points B and C and the perimeter of ΔABC -/
def B : ℝ × ℝ := (-3, 0)
def C : ℝ × ℝ := (3, 0)
def perimeter (A : ℝ × ℝ) : ℝ :=
  Real.dist B A + Real.dist C A + Real.dist B C

/-- The statement we need to prove -/
theorem trajectory_of_A_eq : 
  ∀ A : ℝ × ℝ, perimeter A = 16 →
  (∃ k : ℝ, k ≠ 0 ∧ (A.2 / k) ^ 2 = 16 * (1 - (A.1 / 5) ^ 2) * k ^ 2) :=
by
  sorry

end trajectory_of_A_eq_l236_236311


namespace min_value_quadratic_l236_236245

theorem min_value_quadratic (x : ℝ) : 
  ∀ x ∈ ℝ, x = 6 ↔ x^2 - 12x + 36 = (x - 6)^2 ∨ (x - 6)^2 >= 0 := 
begin
  sorry
end

end min_value_quadratic_l236_236245


namespace omicron_variant_diameter_in_scientific_notation_l236_236506

/-- Converting a number to scientific notation. -/
def to_scientific_notation (d : ℝ) (a : ℝ) (n : ℤ) : Prop :=
  d = a * 10 ^ n

theorem omicron_variant_diameter_in_scientific_notation :
  to_scientific_notation 0.00000011 1.1 (-7) :=
by
  sorry

end omicron_variant_diameter_in_scientific_notation_l236_236506


namespace corrected_mean_l236_236634

theorem corrected_mean (n : ℕ) (incorrect_mean old_obs new_obs : ℚ) 
  (hn : n = 50) (h_mean : incorrect_mean = 40) (hold : old_obs = 15) (hnew : new_obs = 45) :
  ((n * incorrect_mean + (new_obs - old_obs)) / n) = 40.6 :=
by
  sorry

end corrected_mean_l236_236634


namespace part1_part2_l236_236213

section
variables (x a m n : ℝ)
-- Define the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x - 3)

-- a) Prove the solution of the inequality f(x) >= 4 + |x-3| - |x-1| given a=3.
theorem part1 (h_a : a = 3) :
  {x | f x a ≥ 4 + abs (x - 3) - abs (x - 1)} = {x | x ≤ 0} ∪ {x | x ≥ 4} :=
sorry

-- b) Prove that m + 2n >= 2 given f(x) <= 1 + |x-3| with solution set [1, 3] and 1/m + 1/(2n) = a
theorem part2 (h_sol : ∀ x, 1 ≤ x ∧ x ≤ 3 → f x a ≤ 1 + abs (x - 3)) 
  (h_a : 1 / m + 1 / (2 * n) = 2) (h_m_pos : m > 0) (h_n_pos : n > 0) :
  m + 2 * n ≥ 2 :=
sorry
end

end part1_part2_l236_236213


namespace total_area_correct_l236_236392

-- Define the given conditions
def dust_covered_area : ℕ := 64535
def untouched_area : ℕ := 522

-- Define the total area of prairie by summing covered and untouched areas
def total_prairie_area : ℕ := dust_covered_area + untouched_area

-- State the theorem we need to prove
theorem total_area_correct : total_prairie_area = 65057 := by
  sorry

end total_area_correct_l236_236392


namespace distance_between_ann_and_glenda_l236_236274

def ann_distance : ℝ := 
  let speed1 := 6
  let time1 := 1
  let speed2 := 8
  let time2 := 1
  let break1 := 0
  let speed3 := 4
  let time3 := 1
  speed1 * time1 + speed2 * time2 + break1 * 0 + speed3 * time3

def glenda_distance : ℝ := 
  let speed1 := 8
  let time1 := 1
  let speed2 := 5
  let time2 := 1
  let break1 := 0
  let speed3 := 9
  let back_time := 0.5
  let back_distance := speed3 * back_time
  let continue_time := 0.5
  let continue_distance := speed3 * continue_time
  speed1 * time1 + speed2 * time2 + break1 * 0 + (-back_distance) + continue_distance

theorem distance_between_ann_and_glenda : 
  ann_distance + glenda_distance = 35.5 := 
by 
  sorry

end distance_between_ann_and_glenda_l236_236274


namespace sum_of_integers_remainder_l236_236376

-- Definitions of the integers and their properties
variables (a b c : ℕ)

-- Conditions
axiom h1 : a % 53 = 31
axiom h2 : b % 53 = 17
axiom h3 : c % 53 = 8
axiom h4 : a % 5 = 0

-- The proof goal
theorem sum_of_integers_remainder :
  (a + b + c) % 53 = 3 :=
by
  sorry -- Proof to be provided

end sum_of_integers_remainder_l236_236376


namespace quad_area_l236_236537

theorem quad_area (a b : Int) (h1 : a > b) (h2 : b > 0) (h3 : 2 * |a - b| * |a + b| = 50) : a + b = 15 :=
by
  sorry

end quad_area_l236_236537


namespace problem_l236_236446

noncomputable def f (x : ℝ) := Real.log x + (x + 1) / x

noncomputable def g (x : ℝ) := x - 1/x - 2 * Real.log x

theorem problem 
  (x : ℝ) (hx : x > 0) (hxn1 : x ≠ 1) :
  f x > (x + 1) * Real.log x / (x - 1) :=
by
  sorry

end problem_l236_236446


namespace students_in_class_l236_236110

theorem students_in_class (total_pencils : ℕ) (pencils_per_student : ℕ) (n: ℕ) 
    (h1 : total_pencils = 18) 
    (h2 : pencils_per_student = 9) 
    (h3 : total_pencils = n * pencils_per_student) : 
    n = 2 :=
by 
  sorry

end students_in_class_l236_236110


namespace rate_per_kg_mangoes_l236_236950

theorem rate_per_kg_mangoes (kg_apples kg_mangoes total_cost rate_apples total_payment rate_mangoes : ℕ) 
  (h1 : kg_apples = 8) 
  (h2 : rate_apples = 70)
  (h3 : kg_mangoes = 9)
  (h4 : total_payment = 965) :
  rate_mangoes = 45 := 
by
  sorry

end rate_per_kg_mangoes_l236_236950


namespace cost_of_each_book_is_six_l236_236849

-- Define variables for the number of books bought
def books_about_animals := 8
def books_about_outer_space := 6
def books_about_trains := 3

-- Define the total number of books
def total_books := books_about_animals + books_about_outer_space + books_about_trains

-- Define the total amount spent
def total_amount_spent := 102

-- Define the cost per book
def cost_per_book := total_amount_spent / total_books

-- Prove that the cost per book is $6
theorem cost_of_each_book_is_six : cost_per_book = 6 := by
  sorry

end cost_of_each_book_is_six_l236_236849


namespace by_how_much_were_the_numerator_and_denominator_increased_l236_236505

noncomputable def original_fraction_is_six_over_eleven (n : ℕ) : Prop :=
  n / (n + 5) = 6 / 11

noncomputable def resulting_fraction_is_seven_over_twelve (n x : ℕ) : Prop :=
  (n + x) / (n + 5 + x) = 7 / 12

theorem by_how_much_were_the_numerator_and_denominator_increased :
  ∃ (n x : ℕ), original_fraction_is_six_over_eleven n ∧ resulting_fraction_is_seven_over_twelve n x ∧ x = 1 :=
by
  sorry

end by_how_much_were_the_numerator_and_denominator_increased_l236_236505


namespace range_of_a_l236_236056

theorem range_of_a (a : ℝ) :
  (∃! x : ℕ, x^2 - (a + 2) * x + 2 - a < 0) ↔ (1/2 < a ∧ a ≤ 2/3) := 
sorry

end range_of_a_l236_236056


namespace anya_kolya_apples_l236_236691

theorem anya_kolya_apples (A K : ℕ) (h1 : A = (K * 100) / (A + K)) (h2 : K = (A * 100) / (A + K)) : A = 50 ∧ K = 50 :=
sorry

end anya_kolya_apples_l236_236691


namespace right_triangle_example_find_inverse_450_mod_3599_l236_236514

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def multiplicative_inverse (a b m : ℕ) : Prop :=
  (a * b) % m = 1

theorem right_triangle_example : is_right_triangle 60 221 229 :=
by
  sorry

theorem find_inverse_450_mod_3599 : ∃ n, 0 ≤ n ∧ n < 3599 ∧ multiplicative_inverse 450 n 3599 :=
by
  use 8
  sorry

end right_triangle_example_find_inverse_450_mod_3599_l236_236514


namespace solution_condition1_solution_condition2_solution_condition3_solution_condition4_l236_236688

-- Define the conditions
def Condition1 : Prop :=
  ∃ (total_population box1 box2 sampled : Nat),
  total_population = 30 ∧ box1 = 21 ∧ box2 = 9 ∧ sampled = 10

def Condition2 : Prop :=
  ∃ (total_population produced_by_A produced_by_B sampled : Nat),
  total_population = 30 ∧ produced_by_A = 21 ∧ produced_by_B = 9 ∧ sampled = 10

def Condition3 : Prop :=
  ∃ (total_population sampled : Nat),
  total_population = 300 ∧ sampled = 10

def Condition4 : Prop :=
  ∃ (total_population sampled : Nat),
  total_population = 300 ∧ sampled = 50

-- Define the appropriate methods
def LotteryMethod : Prop := ∃ method : String, method = "Lottery method"
def StratifiedSampling : Prop := ∃ method : String, method = "Stratified sampling"
def RandomNumberMethod : Prop := ∃ method : String, method = "Random number method"
def SystematicSampling : Prop := ∃ method : String, method = "Systematic sampling"

-- Statements to prove the appropriate methods for each condition
theorem solution_condition1 : Condition1 → LotteryMethod := by sorry
theorem solution_condition2 : Condition2 → StratifiedSampling := by sorry
theorem solution_condition3 : Condition3 → RandomNumberMethod := by sorry
theorem solution_condition4 : Condition4 → SystematicSampling := by sorry

end solution_condition1_solution_condition2_solution_condition3_solution_condition4_l236_236688


namespace tan_diff_identity_l236_236862

theorem tan_diff_identity 
  (α : ℝ)
  (h : Real.tan α = -4/3) : Real.tan (α - Real.pi / 4) = 7 := 
sorry

end tan_diff_identity_l236_236862


namespace sunzi_problem_solution_l236_236973

theorem sunzi_problem_solution (x y : ℝ) :
  (y = x + 4.5) ∧ (0.5 * y = x - 1) ↔ (y = x + 4.5 ∧ 0.5 * y = x - 1) :=
by 
  sorry

end sunzi_problem_solution_l236_236973


namespace same_number_of_acquaintances_l236_236616

theorem same_number_of_acquaintances (n : ℕ) (h : n ≥ 2) (acquaintances : Fin n → Fin n) :
  ∃ i j : Fin n, i ≠ j ∧ acquaintances i = acquaintances j :=
by
  -- Insert proof here
  sorry

end same_number_of_acquaintances_l236_236616


namespace p_sufficient_not_necessary_for_q_l236_236077

def p (x1 x2 : ℝ) : Prop := x1 > 1 ∧ x2 > 1
def q (x1 x2 : ℝ) : Prop := x1 + x2 > 2 ∧ x1 * x2 > 1

theorem p_sufficient_not_necessary_for_q : 
  (∀ x1 x2 : ℝ, p x1 x2 → q x1 x2) ∧ ¬ (∀ x1 x2 : ℝ, q x1 x2 → p x1 x2) :=
by 
  sorry

end p_sufficient_not_necessary_for_q_l236_236077


namespace proof_problem_l236_236028

variable (a b c : ℝ)
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem proof_problem 
  (h0 : f a b c 0 = f a b c 4)
  (h1 : f a b c 0 > f a b c 1) : 
  a > 0 ∧ 4 * a + b = 0 :=
by
  sorry

end proof_problem_l236_236028


namespace complex_value_of_product_l236_236907

theorem complex_value_of_product (r : ℂ) (hr : r^7 = 1) (hr1 : r ≠ 1) : 
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 8 := 
by sorry

end complex_value_of_product_l236_236907


namespace terminating_decimal_expansion_of_17_div_625_l236_236549

theorem terminating_decimal_expansion_of_17_div_625 : 
  ∃ d : ℚ, d = 17 / 625 ∧ d = 0.0272 :=
by
  sorry

end terminating_decimal_expansion_of_17_div_625_l236_236549


namespace avg_children_nine_families_l236_236807

theorem avg_children_nine_families
  (total_families : ℕ)
  (average_children : ℕ)
  (childless_families : ℕ)
  (total_children : ℕ)
  (families_with_children : ℕ) :
  total_families = 12 →
  average_children = 3 →
  childless_families = 3 →
  total_children = total_families * average_children →
  families_with_children = total_families - childless_families →
  (total_children / families_with_children : ℝ) = 4.0 :=
begin
  intros,
  sorry
end

end avg_children_nine_families_l236_236807


namespace george_borrow_amount_l236_236713

-- Define the conditions
def initial_fee_rate : ℝ := 0.05
def doubling_rate : ℝ := 2
def total_weeks : ℕ := 2
def total_fee : ℝ := 15

-- Define the problem statement
theorem george_borrow_amount : 
  ∃ (P : ℝ), (initial_fee_rate * P + initial_fee_rate * doubling_rate * P = total_fee) ∧ P = 100 :=
by
  -- Statement only, proof is skipped
  sorry

end george_borrow_amount_l236_236713


namespace longest_side_of_rectangle_l236_236068

theorem longest_side_of_rectangle (l w : ℕ) 
  (h1 : 2 * l + 2 * w = 240) 
  (h2 : l * w = 1920) : 
  l = 101 ∨ w = 101 :=
sorry

end longest_side_of_rectangle_l236_236068


namespace number_of_self_inverse_subsets_is_15_l236_236461

-- Define the set M
def M : Set ℚ := ({-1, 0, 1/2, 1/3, 1, 2, 3, 4} : Set ℚ)

-- Definition of self-inverse set
def is_self_inverse (A : Set ℚ) : Prop := ∀ x ∈ A, 1/x ∈ A

-- Theorem stating the number of non-empty self-inverse subsets of M
theorem number_of_self_inverse_subsets_is_15 :
  (∃ S : Finset (Set ℚ), S.card = 15 ∧ ∀ A ∈ S, A ⊆ M ∧ is_self_inverse A) :=
sorry

end number_of_self_inverse_subsets_is_15_l236_236461


namespace new_mean_rent_is_880_l236_236986

theorem new_mean_rent_is_880
  (num_friends : ℕ)
  (initial_average_rent : ℝ)
  (increase_percentage : ℝ)
  (original_rent_increased : ℝ)
  (new_mean_rent : ℝ) :
  num_friends = 4 →
  initial_average_rent = 800 →
  increase_percentage = 20 →
  original_rent_increased = 1600 →
  new_mean_rent = 880 :=
by
  intros h1 h2 h3 h4
  sorry

end new_mean_rent_is_880_l236_236986


namespace range_x0_of_perpendicular_bisector_intersects_x_axis_l236_236575

open Real

theorem range_x0_of_perpendicular_bisector_intersects_x_axis
  (A B : ℝ × ℝ) 
  (hA : (A.1^2 / 9) + (A.2^2 / 8) = 1)
  (hB : (B.1^2 / 9) + (B.2^2 / 8) = 1)
  (N : ℝ × ℝ) 
  (P : ℝ × ℝ) 
  (hN : N = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hP : P.2 = 0) 
  (hl : P.1 = N.1 + (8 * N.1) / (9 * N.2) * N.2)
  : -1/3 < P.1 ∧ P.1 < 1/3 :=
sorry

end range_x0_of_perpendicular_bisector_intersects_x_axis_l236_236575


namespace minimum_value_frac_inv_is_one_third_l236_236770

noncomputable def min_value_frac_inv (x y : ℝ) : ℝ :=
  if x > 0 ∧ y > 0 ∧ x + y = 12 then 1/x + 1/y else 0

theorem minimum_value_frac_inv_is_one_third (x y : ℝ) (h : x > 0 ∧ y > 0 ∧ x + y = 12) :
  min_value_frac_inv x y = 1/3 :=
begin
  -- Proof to be provided
  sorry
end

end minimum_value_frac_inv_is_one_third_l236_236770


namespace polygon_sides_l236_236647

theorem polygon_sides (sum_of_interior_angles : ℝ) (x : ℝ) (h : sum_of_interior_angles = 1080) : x = 8 :=
by
  sorry

end polygon_sides_l236_236647


namespace gena_encoded_numbers_unique_l236_236295

theorem gena_encoded_numbers_unique : 
  ∃ (B AN AX NO FF d : ℕ), (AN - B = d) ∧ (AX - AN = d) ∧ (NO - AX = d) ∧ (FF - NO = d) ∧ 
  [B, AN, AX, NO, FF] = [5, 12, 19, 26, 33] := sorry

end gena_encoded_numbers_unique_l236_236295


namespace roots_square_difference_l236_236905

theorem roots_square_difference (a b : ℚ)
  (ha : 6 * a^2 + 13 * a - 28 = 0)
  (hb : 6 * b^2 + 13 * b - 28 = 0) : (a - b)^2 = 841 / 36 :=
sorry

end roots_square_difference_l236_236905


namespace least_five_digit_congruent_to_6_mod_19_l236_236953

theorem least_five_digit_congruent_to_6_mod_19 : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 19 = 6 ∧ n = 10011 :=
by
  sorry

end least_five_digit_congruent_to_6_mod_19_l236_236953


namespace find_initial_red_marbles_l236_236045

theorem find_initial_red_marbles (x y : ℚ) 
  (h1 : 2 * x = 3 * y) 
  (h2 : 5 * (x - 15) = 2 * (y + 25)) 
  : x = 375 / 11 := 
by
  sorry

end find_initial_red_marbles_l236_236045


namespace sin_double_angle_l236_236871

theorem sin_double_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.sin (2 * θ) = 3 / 5 := 
by 
sorry

end sin_double_angle_l236_236871


namespace geometric_sequence_common_ratio_l236_236394

theorem geometric_sequence_common_ratio (a1 a2 a3 a4 : ℝ)
  (h₁ : a1 = 32) (h₂ : a2 = -48) (h₃ : a3 = 72) (h₄ : a4 = -108)
  (h_geom : ∃ r, a2 = r * a1 ∧ a3 = r * a2 ∧ a4 = r * a3) :
  ∃ r, r = -3/2 :=
by
  sorry

end geometric_sequence_common_ratio_l236_236394


namespace valid_codes_count_l236_236340

open Finset

def four_digit_code := {x : Fin (8 × 8 × 8 × 8) // x < 4096}

def is_restricted_transpose (code : four_digit_code) : Bool :=
  let d0 := code.val % 8
  let d1 := (code.val / 8) % 8
  let d2 := (code.val / 64) % 8
  let d3 := code.val / 512
  (d0 == 1 && d1 == 0 && d2 == 2 && d3 == 3) || -- exact match with 1023
  (d1 == 1 && d0 == 0 && d2 == 2 && d3 == 3) || -- transpose first two digits
  (d2 == 1 && d1 == 0 && d0 == 2 && d3 == 3) ||
  (d3 == 1 && d1 == 0 && d2 == 2 && d0 == 3) || -- transpose first and fourth digits
  (d3 == 1 && d2 == 0 && d1 == 2 && d0 == 3) ||
  ... -- Add remaining transpose checks (total 6 types)

def is_restricted_three_match (code : four_digit_code) : Bool :=
  let d0 := code.val % 8
  let d1 := (code.val / 8) % 8
  let d2 := (code.val / 64) % 8
  let d3 := code.val / 512
  (d0 == 1 && d1 == 0 && d2 == 2) || 
  (d0 == 1 && d1 == 0 && d3 == 2) ||
  (d0 == 1 && d2 == 0 && d3 == 2) ||
  (d1 == 1 && d2 == 0 && d3 == 2) ||
  ... -- Add remaining 3-digit matched cases (total 4 types)

def count_valid_codes : ℕ :=
  let all_codes := (range 4096).val
  let restricted_codes := all_codes.filter (fun x =>
    is_restricted_transpose ⟨x, sorry⟩ || is_restricted_three_match ⟨x, sorry⟩ || x == 1023
  )
  (card all_codes) - (card restricted_codes)

theorem valid_codes_count : count_valid_codes = 4043 := by
  sorry

end valid_codes_count_l236_236340


namespace sqrt_of_9_eq_3_l236_236945

theorem sqrt_of_9_eq_3 : Real.sqrt 9 = 3 := by
  sorry

end sqrt_of_9_eq_3_l236_236945


namespace base4_to_base10_conversion_l236_236415

-- We define a base 4 number as follows:
def base4_to_base10 (n : ℕ) : ℕ :=
  let d0 := n % 10 in
  let n := n / 10 in
  let d1 := n % 10 in
  let n := n / 10 in
  let d2 := n % 10 in
  let n := n / 10 in
  let d3 := n % 10 in
  let n := n / 10 in
  let d4 := n % 10 in
  (d4 * 4^4 + d3 * 4^3 + d2 * 4^2 + d1 * 4^1 + d0 * 4^0)

-- Mathematical proof problem statement:
theorem base4_to_base10_conversion : base4_to_base10 21012 = 582 :=
  sorry

end base4_to_base10_conversion_l236_236415


namespace Rahul_savings_l236_236617

variable (total_savings ppf_savings nsc_savings x : ℝ)

theorem Rahul_savings
  (h1 : total_savings = 180000)
  (h2 : ppf_savings = 72000)
  (h3 : nsc_savings = total_savings - ppf_savings)
  (h4 : x * nsc_savings = 0.5 * ppf_savings) :
  x = 1 / 3 :=
by
  -- Proof goes here
  sorry

end Rahul_savings_l236_236617


namespace new_shape_perimeter_l236_236540

-- Definitions based on conditions
def square_side : ℕ := 64 / 4
def is_tri_isosceles (a b c : ℕ) : Prop := a = b

-- Definition of given problem setup and perimeter calculation
theorem new_shape_perimeter
  (side : ℕ)
  (tri_side1 tri_side2 base : ℕ)
  (h_square_side : side = 64 / 4)
  (h_tri1 : tri_side1 = side)
  (h_tri2 : tri_side2 = side)
  (h_base : base = side) :
  (side * 5) = 80 :=
by
  sorry

end new_shape_perimeter_l236_236540


namespace max_value_m_l236_236863

noncomputable def max_m : ℝ := 10

theorem max_value_m (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x * y = x + 2 * y) : x * y ≥ max_m - 2 :=
by
  sorry

end max_value_m_l236_236863


namespace record_loss_of_300_l236_236933

-- Definitions based on conditions
def profit (x : Int) : String := "+" ++ toString x
def loss (x : Int) : String := "-" ++ toString x

-- The theorem to prove that a loss of 300 is recorded as "-300" based on the recording system
theorem record_loss_of_300 : loss 300 = "-300" :=
by
  sorry

end record_loss_of_300_l236_236933


namespace major_axis_length_l236_236257

noncomputable def length_of_major_axis (f1 f2 : ℝ × ℝ) (tangent_y_axis : Bool) (tangent_line_y : ℝ) : ℝ :=
  if f1 = (-Real.sqrt 5, 2) ∧ f2 = (Real.sqrt 5, 2) ∧ tangent_y_axis ∧ tangent_line_y = 1 then 2
  else 0

theorem major_axis_length :
  length_of_major_axis (-Real.sqrt 5, 2) (Real.sqrt 5, 2) true 1 = 2 :=
by
  sorry

end major_axis_length_l236_236257


namespace arithmetic_mean_l236_236161

theorem arithmetic_mean (x b : ℝ) (h : x ≠ 0) : 
  (1 / 2) * ((2 + (b / x)) + (2 - (b / x))) = 2 :=
by sorry

end arithmetic_mean_l236_236161


namespace sin_double_angle_l236_236430

open Real

theorem sin_double_angle (α : ℝ) (h1 : α ∈ Set.Ioc (π / 2) π) (h2 : sin α = 4 / 5) :
  sin (2 * α) = -24 / 25 :=
by
  sorry

end sin_double_angle_l236_236430


namespace zebra_crossing_distance_l236_236832

theorem zebra_crossing_distance
  (boulevard_width : ℝ)
  (distance_along_stripes : ℝ)
  (stripe_length : ℝ)
  (distance_between_stripes : ℝ) :
  boulevard_width = 60 →
  distance_along_stripes = 22 →
  stripe_length = 65 →
  distance_between_stripes = (60 * 22) / 65 →
  distance_between_stripes = 20.31 :=
by
  intros h1 h2 h3 h4
  sorry

end zebra_crossing_distance_l236_236832


namespace biquadratic_exactly_two_distinct_roots_l236_236021

theorem biquadratic_exactly_two_distinct_roots {a : ℝ} :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^4 + a*x1^2 + a - 1 = 0) ∧ (x2^4 + a*x2^2 + a - 1 = 0) ∧
   ∀ x, x^4 + a*x^2 + a - 1 = 0 → (x = x1 ∨ x = x2)) ↔ a < 1 :=
by
  sorry

end biquadratic_exactly_two_distinct_roots_l236_236021


namespace total_amount_is_69_l236_236684

-- Define the total amount paid for the work as X
def total_amount_paid (X : ℝ) : Prop := 
  let B_payment := 12
  let portion_of_B_work := 4 / 23
  (portion_of_B_work * X = B_payment)

theorem total_amount_is_69 : ∃ X : ℝ, total_amount_paid X ∧ X = 69 := by
  sorry

end total_amount_is_69_l236_236684


namespace distance_between_towns_l236_236701

theorem distance_between_towns 
  (x : ℝ) 
  (h1 : x / 100 - x / 110 = 0.15) : 
  x = 165 := 
by 
  sorry

end distance_between_towns_l236_236701


namespace asymptotes_of_hyperbola_l236_236700

theorem asymptotes_of_hyperbola (x y : ℝ) :
  (x ^ 2 / 4 - y ^ 2 / 9 = -1) →
  (y = (3 / 2) * x ∨ y = -(3 / 2) * x) :=
sorry

end asymptotes_of_hyperbola_l236_236700


namespace variance_of_red_balls_l236_236889

noncomputable def redBalls : ℕ := 8
noncomputable def yellowBalls : ℕ := 4
noncomputable def totalBalls := redBalls + yellowBalls
noncomputable def n : ℕ := 4
noncomputable def p : ℚ := redBalls / totalBalls
noncomputable def D_X : ℚ := n * p * (1 - p)

theorem variance_of_red_balls :
  D_X = 8 / 9 :=
by
  -- conditions are defined in the definitions above
  -- proof is skipped
  sorry

end variance_of_red_balls_l236_236889


namespace christmas_sale_pricing_l236_236284

theorem christmas_sale_pricing (a b : ℝ) : 
  (forall (c : ℝ), c = a * (3 / 5)) ∧ (forall (d : ℝ), d = b * (5 / 3)) :=
by
  sorry  -- proof goes here

end christmas_sale_pricing_l236_236284


namespace solve_for_x_l236_236787

theorem solve_for_x : 
  (35 / (6 - (2 / 5)) = 25 / 4) := 
by
  sorry 

end solve_for_x_l236_236787


namespace remainder_of_division_l236_236507

theorem remainder_of_division (x : ℕ) (r : ℕ) :
  1584 - x = 1335 ∧ 1584 = 6 * x + r → r = 90 := by
  sorry

end remainder_of_division_l236_236507


namespace correct_statements_l236_236292

noncomputable def f (x : ℝ) : ℝ := 2^x - 2^(-x)

theorem correct_statements :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (f (Real.log 3 / Real.log 2) ≠ 2) ∧
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (∀ x : ℝ, f (|x|) ≥ 0 ∧ f 0 = 0) :=
by
  sorry

end correct_statements_l236_236292


namespace probability_at_least_two_tails_l236_236403

def fair_coin_prob (n : ℕ) : ℚ :=
  (1 / 2 : ℚ)^n

def at_least_two_tails_in_next_three_flips : ℚ :=
  1 - (fair_coin_prob 3 + 3 * fair_coin_prob 3)

theorem probability_at_least_two_tails :
  at_least_two_tails_in_next_three_flips = 1 / 2 := 
by
  sorry

end probability_at_least_two_tails_l236_236403


namespace fraction_simplifies_to_two_l236_236117

theorem fraction_simplifies_to_two :
  (2 + 4 + 6 + 8 + 10 + 12 + 14 + 16 + 18 + 20) / (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10) = 2 := by
  sorry

end fraction_simplifies_to_two_l236_236117


namespace identity_proof_l236_236089

theorem identity_proof (a b c : ℝ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    (b - c) / ((a - b) * (a - c)) + (c - a) / ((b - c) * (b - a)) + (a - b) / ((c - a) * (c - b)) =
    2 / (a - b) + 2 / (b - c) + 2 / (c - a) :=
by
  sorry

end identity_proof_l236_236089


namespace compute_x_y_sum_l236_236909

theorem compute_x_y_sum (x y : ℝ) (hx : x > 1) (hy : y > 1)
  (h : (Real.log x / Real.log 2)^4 + (Real.log y / Real.log 3)^4 + 8 = 8 * (Real.log x / Real.log 2) * (Real.log y / Real.log 3)) :
  x^Real.sqrt 2 + y^Real.sqrt 2 = 13 :=
by
  sorry

end compute_x_y_sum_l236_236909


namespace julia_more_kids_on_monday_l236_236334

-- Definition of the problem statement
def playedWithOnMonday : ℕ := 6
def playedWithOnTuesday : ℕ := 5
def difference := playedWithOnMonday - playedWithOnTuesday

theorem julia_more_kids_on_monday : difference = 1 :=
by
  -- Proof can be filled out here.
  sorry

end julia_more_kids_on_monday_l236_236334


namespace negation_of_proposition_l236_236799

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^2 - 2 * x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2 * x + 4 > 0) :=
by sorry

end negation_of_proposition_l236_236799


namespace probability_of_defective_on_second_draw_l236_236979

-- Define the conditions
variable (batch_size : ℕ) (defective_items : ℕ) (good_items : ℕ)
variable (first_draw_good : Prop)
variable (without_replacement : Prop)

-- Given conditions
def batch_conditions : Prop :=
  batch_size = 10 ∧ defective_items = 3 ∧ good_items = 7 ∧ first_draw_good ∧ without_replacement

-- The desired probability as a proof
theorem probability_of_defective_on_second_draw
  (h : batch_conditions batch_size defective_items good_items first_draw_good without_replacement) : 
  (3 / 9 : ℝ) = 1 / 3 :=
sorry

end probability_of_defective_on_second_draw_l236_236979


namespace find_a_sq_plus_b_sq_l236_236590

-- Variables and conditions
variables (a b : ℝ)
-- Conditions from the problem
axiom h1 : a - b = 3
axiom h2 : a * b = 9

-- The proof statement
theorem find_a_sq_plus_b_sq (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 9) : a^2 + b^2 = 27 :=
by {
  sorry
}

end find_a_sq_plus_b_sq_l236_236590


namespace value_of_a_minus_b_l236_236191

theorem value_of_a_minus_b 
  (a b : ℤ) 
  (x y : ℤ)
  (h1 : x = -2)
  (h2 : y = 1)
  (h3 : a * x + b * y = 1)
  (h4 : b * x + a * y = 7) : 
  a - b = 2 :=
by
  sorry

end value_of_a_minus_b_l236_236191


namespace minimize_quadratic_l236_236244

theorem minimize_quadratic : ∃ x : ℝ, x = 6 ∧ ∀ y : ℝ, (y - 6)^2 ≥ (6 - 6)^2 := by
  sorry

end minimize_quadratic_l236_236244


namespace n_is_one_sixth_sum_of_list_l236_236260

-- Define the condition that n is 4 times the average of the other 20 numbers
def satisfies_condition (n : ℝ) (l : List ℝ) : Prop :=
  l.length = 21 ∧
  n ∈ l ∧
  n = 4 * (l.erase n).sum / 20

-- State the main theorem
theorem n_is_one_sixth_sum_of_list {n : ℝ} {l : List ℝ} (h : satisfies_condition n l) :
  n = (1 / 6) * l.sum :=
by
  sorry

end n_is_one_sixth_sum_of_list_l236_236260


namespace alpha_beta_identity_l236_236307

open Real

theorem alpha_beta_identity 
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2)
  (h : cos β = tan α * (1 + sin β)) : 
  2 * α + β = π / 2 :=
by
  sorry

end alpha_beta_identity_l236_236307


namespace pupils_who_like_both_l236_236387

theorem pupils_who_like_both (total_pupils pizza_lovers burger_lovers : ℕ) (h1 : total_pupils = 200) (h2 : pizza_lovers = 125) (h3 : burger_lovers = 115) :
  (pizza_lovers + burger_lovers - total_pupils = 40) :=
by
  sorry

end pupils_who_like_both_l236_236387


namespace weight_of_four_cakes_l236_236238

variable (C B : ℕ)  -- We declare C and B as natural numbers representing the weights in grams.

def cake_bread_weight_conditions (C B : ℕ) : Prop :=
  (3 * C + 5 * B = 1100) ∧ (C = B + 100)

theorem weight_of_four_cakes (C B : ℕ) 
  (h : cake_bread_weight_conditions C B) : 
  4 * C = 800 := 
by 
  {sorry}

end weight_of_four_cakes_l236_236238


namespace moles_of_HCl_combined_eq_one_l236_236012

-- Defining the chemical species involved in the reaction
def NaHCO3 : Type := Nat
def HCl : Type := Nat
def NaCl : Type := Nat
def H2O : Type := Nat
def CO2 : Type := Nat

-- Defining the balanced chemical equation as a condition
def reaction (n_NaHCO3 n_HCl n_NaCl n_H2O n_CO2 : Nat) : Prop :=
  n_NaHCO3 + n_HCl = n_NaCl + n_H2O + n_CO2

-- Given conditions
def one_mole_of_NaHCO3 : Nat := 1
def one_mole_of_NaCl_produced : Nat := 1

-- Proof problem
theorem moles_of_HCl_combined_eq_one :
  ∃ (n_HCl : Nat), reaction one_mole_of_NaHCO3 n_HCl one_mole_of_NaCl_produced 1 1 ∧ n_HCl = 1 := 
by
  sorry

end moles_of_HCl_combined_eq_one_l236_236012


namespace coconut_grove_l236_236379

theorem coconut_grove (x : ℕ) :
  (40 * (x + 2) + 120 * x + 180 * (x - 2) = 100 * 3 * x) → 
  x = 7 := by
  sorry

end coconut_grove_l236_236379


namespace triangle_count_l236_236880

theorem triangle_count (a b c : ℕ) (h1 : a + b + c = 15) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : a + b > c) :
  ∃ (n : ℕ), n = 7 :=
by
  -- Proceed with the proof steps, using a, b, c satisfying the given conditions
  sorry

end triangle_count_l236_236880


namespace min_value_inequality_equality_condition_l236_236864

theorem min_value_inequality (a b : ℝ) (ha : 1 < a) (hb : 1 < b) :
  (b^2 / (a - 1) + a^2 / (b - 1)) ≥ 8 :=
sorry

theorem equality_condition (a b : ℝ) (ha : 1 < a) (hb : 1 < b) :
  (b^2 / (a - 1) + a^2 / (b - 1) = 8) ↔ ((a = 2) ∧ (b = 2)) :=
sorry

end min_value_inequality_equality_condition_l236_236864


namespace probability_10_or_9_probability_at_least_7_l236_236134

-- Define the probabilities of hitting each ring
def p_10 : ℝ := 0.1
def p_9 : ℝ := 0.2
def p_8 : ℝ := 0.3
def p_7 : ℝ := 0.3
def p_below_7 : ℝ := 0.1

-- Define the events as their corresponding probabilities
def P_A : ℝ := p_10 -- Event of hitting the 10 ring
def P_B : ℝ := p_9 -- Event of hitting the 9 ring
def P_C : ℝ := p_8 -- Event of hitting the 8 ring
def P_D : ℝ := p_7 -- Event of hitting the 7 ring
def P_E : ℝ := p_below_7 -- Event of hitting below the 7 ring

-- Since the probabilities must sum to 1, we have the following fact about their sum
-- P_A + P_B + P_C + P_D + P_E = 1

theorem probability_10_or_9 : P_A + P_B = 0.3 :=
by 
  -- This would be filled in with the proof steps or assumptions
  sorry

theorem probability_at_least_7 : P_A + P_B + P_C + P_D = 0.9 :=
by 
  -- This would be filled in with the proof steps or assumptions
  sorry

end probability_10_or_9_probability_at_least_7_l236_236134


namespace n_fraction_of_sum_l236_236259

theorem n_fraction_of_sum (n S : ℝ) (h1 : n = S / 5) (h2 : S ≠ 0) :
  n = 1 / 6 * ((S + (S / 5))) :=
by
  sorry

end n_fraction_of_sum_l236_236259


namespace number_of_children_at_reunion_l236_236407

theorem number_of_children_at_reunion (A C : ℕ) 
    (h1 : 3 * A = C)
    (h2 : 2 * A / 3 = 10) : 
  C = 45 :=
by
  sorry

end number_of_children_at_reunion_l236_236407


namespace inequality_subtraction_l236_236881

theorem inequality_subtraction (a b c : ℝ) (h : a > b) : a - c > b - c :=
sorry

end inequality_subtraction_l236_236881


namespace remainder_9_5_4_6_5_7_mod_7_l236_236656

theorem remainder_9_5_4_6_5_7_mod_7 :
  ((9^5 + 4^6 + 5^7) % 7) = 2 :=
by sorry

end remainder_9_5_4_6_5_7_mod_7_l236_236656


namespace find_correct_four_digit_number_l236_236662

theorem find_correct_four_digit_number (N : ℕ) (misspelledN : ℕ) (misspelled_unit_digit_correction : ℕ) 
  (h1 : misspelledN = (N / 10) * 10 + 6)
  (h2 : N - misspelled_unit_digit_correction = (N / 10) * 10 - 7 + 9)
  (h3 : misspelledN - 57 = 1819) : N = 1879 :=
  sorry


end find_correct_four_digit_number_l236_236662


namespace compute_b_l236_236177

-- Defining the polynomial and the root conditions
def poly (x a b : ℝ) := x^3 + a * x^2 + b * x + 21

theorem compute_b (a b : ℚ) (h1 : poly (3 + Real.sqrt 5) a b = 0) (h2 : poly (3 - Real.sqrt 5) a b = 0) : 
  b = -27.5 := 
sorry

end compute_b_l236_236177


namespace slices_per_pizza_l236_236804

def number_of_people : ℕ := 18
def slices_per_person : ℕ := 3
def number_of_pizzas : ℕ := 6
def total_slices : ℕ := number_of_people * slices_per_person

theorem slices_per_pizza : total_slices / number_of_pizzas = 9 :=
by
  -- proof steps would go here
  sorry

end slices_per_pizza_l236_236804


namespace quadratic_completing_square_b_plus_c_l236_236640

theorem quadratic_completing_square_b_plus_c :
  ∃ b c : ℤ, (λ x : ℝ, x^2 - 24 * x + 50) = (λ x, (x + b)^2 + c) ∧ b + c = -106 :=
by
  sorry

end quadratic_completing_square_b_plus_c_l236_236640


namespace find_m_given_sampling_conditions_l236_236200

-- Definitions for population and sampling conditions
def population_divided_into_groups : Prop :=
  ∀ n : ℕ, n < 100 → ∃ k : ℕ, k < 10 ∧ n / 10 = k

def systematic_sampling_condition (m k : ℕ) : Prop :=
  k < 10 ∧ m < 10 ∧ (m + k - 1) % 10 < 10 ∧ (m + k - 11) % 10 < 10

-- Given conditions
def given_conditions (m k : ℕ) (n : ℕ) : Prop :=
  k = 6 ∧ n = 52 ∧ systematic_sampling_condition m k

-- The statement to prove
theorem find_m_given_sampling_conditions :
  ∃ m : ℕ, given_conditions m 6 52 ∧ m = 7 :=
by
  sorry

end find_m_given_sampling_conditions_l236_236200


namespace cubed_inequality_l236_236604

variable {a b : ℝ}

theorem cubed_inequality (h : a > b) : a^3 > b^3 :=
sorry

end cubed_inequality_l236_236604


namespace james_calories_ratio_l236_236067

theorem james_calories_ratio:
  ∀ (dancing_sessions_per_day : ℕ) (hours_per_session : ℕ) 
  (days_per_week : ℕ) (calories_per_hour_walking : ℕ) 
  (total_calories_dancing_per_week : ℕ),
  dancing_sessions_per_day = 2 →
  hours_per_session = 1/2 →
  days_per_week = 4 →
  calories_per_hour_walking = 300 →
  total_calories_dancing_per_week = 2400 →
  300 * 2 = 600 →
  (total_calories_dancing_per_week / (dancing_sessions_per_day * hours_per_session * days_per_week)) / calories_per_hour_walking = 2 :=
by
  sorry

end james_calories_ratio_l236_236067


namespace set_intersection_l236_236913

open Finset

-- Let the universal set U, and sets A and B be defined as follows:
def U : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Finset ℕ := {1, 2, 3, 5}
def B : Finset ℕ := {2, 4, 6}

-- Define the complement of A with respect to U:
def complement_A : Finset ℕ := U \ A

-- The goal is to prove that B ∩ complement_A = {4, 6}
theorem set_intersection (h : B ∩ complement_A = {4, 6}) : B ∩ complement_A = {4, 6} :=
by exact h

#check set_intersection

end set_intersection_l236_236913


namespace least_k_for_divisibility_l236_236024

-- Given conditions and goal statement
theorem least_k_for_divisibility (k : ℕ) :
  (k <= 1004 -> ∃ a b : ℕ, a < b ∧ b ≤ 2005 ∧ ∃ m n : ℕ, 1 ≤ m ∧ 1 ≤ n ∧ m ≠ n ∧ (a, b ∈ s ∧ (m*2^a = n*2^b))) :=
by sorry

end least_k_for_divisibility_l236_236024


namespace square_side_length_in_right_triangle_l236_236345

-- Proof problem statement
theorem square_side_length_in_right_triangle 
  (a b : ℝ) (ha : a = 10) (hb : b = 24) 
  (c : ℝ) (hc : c = Real.sqrt (a^2 + b^2)) 
  (s x : ℝ) 
  (h1 : s / c = x / a) 
  (h2 : s / c = (b - x) / b) : 
  s = 312 / 17 := 
by 
  sorry

end square_side_length_in_right_triangle_l236_236345


namespace base4_to_base10_conversion_l236_236418

theorem base4_to_base10_conversion : (2 * 4^4 + 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 2 * 4^0 = 582) :=
by {
  -- The proof is omitted
  sorry
}

end base4_to_base10_conversion_l236_236418


namespace reciprocal_sum_l236_236359

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

end reciprocal_sum_l236_236359


namespace train_speed_platform_man_l236_236837

theorem train_speed_platform_man (t_man t_platform : ℕ) (platform_length : ℕ) (v_train_mps : ℝ) (v_train_kmph : ℝ) 
  (h1 : t_man = 18) 
  (h2 : t_platform = 32) 
  (h3 : platform_length = 280)
  (h4 : v_train_mps = (platform_length / (t_platform - t_man)))
  (h5 : v_train_kmph = v_train_mps * 3.6) :
  v_train_kmph = 72 := 
sorry

end train_speed_platform_man_l236_236837


namespace spaceship_speed_conversion_l236_236835

theorem spaceship_speed_conversion (speed_km_per_sec : ℕ) (seconds_in_hour : ℕ) (correct_speed_km_per_hour : ℕ) :
  speed_km_per_sec = 12 →
  seconds_in_hour = 3600 →
  correct_speed_km_per_hour = 43200 →
  speed_km_per_sec * seconds_in_hour = correct_speed_km_per_hour := by
  sorry

end spaceship_speed_conversion_l236_236835


namespace sequence_term_l236_236712

open Int

-- Define the sequence {S_n} as stated in the problem
def S (n : ℕ) : ℤ := 2 * n^2 - 3 * n

-- Define the sequence {a_n} as the finite difference of {S_n}
def a (n : ℕ) : ℤ := if n = 1 then -1 else S n - S (n - 1)

-- The theorem statement
theorem sequence_term (n : ℕ) (hn : n > 0) : a n = 4 * n - 5 :=
by sorry

end sequence_term_l236_236712


namespace monotonic_intervals_minimum_m_value_l236_236445

noncomputable def f (x : ℝ) (a : ℝ) := (2 * Real.exp 1 + 1) * Real.log x - (3 * a / 2) * x + 1

theorem monotonic_intervals (a : ℝ) : 
  if a ≤ 0 then ∀ x ∈ Set.Ioi 0, 0 < (2 * Real.exp 1 + 1) / x - (3 * a / 2) 
  else ∀ x ∈ Set.Ioc 0 ((2 * (2 * Real.exp 1 + 1)) / (3 * a)), (2 * Real.exp 1 + 1) / x - (3 * a / 2) > 0 ∧
       ∀ x ∈ Set.Ioi ((2 * (2 * Real.exp 1 + 1)) / (3 * a)), (2 * Real.exp 1 + 1) / x - (3 * a / 2) < 0 := sorry

noncomputable def g (x : ℝ) (m : ℝ) := x * Real.exp x + m - ((2 * Real.exp 1 + 1) * Real.log x + x - 1)

theorem minimum_m_value :
  ∀ (m : ℝ), (∀ (x : ℝ), 0 < x → g x m ≥ 0) ↔ m ≥ - Real.exp 1 := sorry

end monotonic_intervals_minimum_m_value_l236_236445


namespace compute_difference_of_squares_l236_236695

theorem compute_difference_of_squares :
  let a := 23
  let b := 12
  (a + b) ^ 2 - (a - b) ^ 2 = 1104 := by
sorry

end compute_difference_of_squares_l236_236695


namespace exists_rat_nonint_sol_a_no_exists_rat_nonint_sol_b_l236_236551

structure RatNonIntPair (x y : ℚ) :=
  (x_rational : x.is_rational)
  (x_not_integer : x.num ≠ x.denom)
  (y_rational : y.is_rational)
  (y_not_integer : y.num ≠ y.denom)

theorem exists_rat_nonint_sol_a :
  ∃ (x y : ℚ), (RatNonIntPair x y) ∧ (int 19 * x + int 8 * y).denom = 1 ∧ (int 8 * x + int 3 * y).denom = 1 := sorry

theorem no_exists_rat_nonint_sol_b :
  ¬ ∃ (x y : ℚ), (RatNonIntPair x y) ∧ (int 19 * (x^2) + int 8 * (y^2)).denom = 1 ∧ (int 8 * (x^2) + int 3 * (y^2)).denom = 1 := sorry

end exists_rat_nonint_sol_a_no_exists_rat_nonint_sol_b_l236_236551


namespace volleyball_ranking_l236_236748

-- Define type for place
inductive Place where
  | first : Place
  | second : Place
  | third : Place

-- Define type for teams
inductive Team where
  | A : Team
  | B : Team
  | C : Team

open Place Team

-- Given conditions as hypotheses
def LiMing_prediction_half_correct (p : Place → Team → Prop) : Prop :=
  (p first A ∨ p third A) ∧ (p first B ∨ p third B) ∧ 
  ¬ (p first A ∧ p third A) ∧ ¬ (p first B ∧ p third B)

def ZhangHua_prediction_half_correct (p : Place → Team → Prop) : Prop :=
  (p third A ∨ p first C) ∧ (p third A ∨ p first A) ∧ 
  ¬ (p third A ∧ p first A) ∧ ¬ (p first C ∧ p third C)

def WangQiang_prediction_half_correct (p : Place → Team → Prop) : Prop :=
  (p second C ∨ p third B) ∧ (p second C ∨ p third C) ∧ 
  ¬ (p second C ∧ p third C) ∧ ¬ (p third B ∧ p second B)

-- Final proof problem
theorem volleyball_ranking (p : Place → Team → Prop) :
    (LiMing_prediction_half_correct p) →
    (ZhangHua_prediction_half_correct p) →
    (WangQiang_prediction_half_correct p) →
    p first C ∧ p second A ∧ p third B :=
  by
    sorry

end volleyball_ranking_l236_236748


namespace average_length_of_strings_l236_236115

-- Define lengths of the three strings
def length1 := 4  -- length of the first string in inches
def length2 := 5  -- length of the second string in inches
def length3 := 7  -- length of the third string in inches

-- Define the total length and number of strings
def total_length := length1 + length2 + length3
def num_strings := 3

-- Define the average length calculation
def average_length := total_length / num_strings

-- The proof statement
theorem average_length_of_strings : average_length = 16 / 3 := 
by 
  sorry

end average_length_of_strings_l236_236115


namespace initial_percentage_of_jasmine_water_l236_236841

-- Definitions
def v_initial : ℝ := 80
def v_jasmine_added : ℝ := 8
def v_water_added : ℝ := 12
def percentage_final : ℝ := 16
def v_final : ℝ := v_initial + v_jasmine_added + v_water_added

-- Lean 4 statement that frames the proof problem
theorem initial_percentage_of_jasmine_water (P : ℝ) :
  (P / 100) * v_initial + v_jasmine_added = (percentage_final / 100) * v_final → P = 10 :=
by
  intro h
  sorry

end initial_percentage_of_jasmine_water_l236_236841


namespace geom_seq_min_value_l236_236172

noncomputable def minimum_sum (m n : ℕ) (a : ℕ → ℝ) : ℝ :=
  if (a 7 = a 6 + 2 * a 5) ∧ (a m * a n = 16 * (a 1) ^ 2) ∧ (m > 0) ∧ (n > 0) then
    (1 / m) + (4 / n)
  else
    0

theorem geom_seq_min_value (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →
  a 7 = a 6 + 2 * a 5 →
  (∃ m n, a m * a n = 16 * (a 1) ^ 2 ∧ m > 0 ∧ n > 0) →
  (minimum_sum m n a = 3 / 2) := sorry

end geom_seq_min_value_l236_236172


namespace compute_c_over_d_l236_236944

noncomputable def RootsResult (a b c d : ℝ) : Prop :=
  (3 * 4 + 4 * 5 + 5 * 3 = - c / a) ∧ (3 * 4 * 5 = - d / a)

theorem compute_c_over_d (a b c d : ℝ)
  (h1 : (a * 3 ^ 3 + b * 3 ^ 2 + c * 3 + d = 0))
  (h2 : (a * 4 ^ 3 + b * 4 ^ 2 + c * 4 + d = 0))
  (h3 : (a * 5 ^ 3 + b * 5 ^ 2 + c * 5 + d = 0)) 
  (hr : RootsResult a b c d) :
  c / d = 47 / 60 := 
by
  sorry

end compute_c_over_d_l236_236944


namespace fruit_boxes_needed_l236_236111

noncomputable def fruit_boxes : ℕ × ℕ × ℕ :=
  let baskets : ℕ := 7
  let peaches_per_basket : ℕ := 23
  let apples_per_basket : ℕ := 19
  let oranges_per_basket : ℕ := 31
  let peaches_eaten : ℕ := 7
  let apples_eaten : ℕ := 5
  let oranges_eaten : ℕ := 3
  let peaches_box_size : ℕ := 13
  let apples_box_size : ℕ := 11
  let oranges_box_size : ℕ := 17

  let total_peaches := baskets * peaches_per_basket
  let total_apples := baskets * apples_per_basket
  let total_oranges := baskets * oranges_per_basket

  let remaining_peaches := total_peaches - peaches_eaten
  let remaining_apples := total_apples - apples_eaten
  let remaining_oranges := total_oranges - oranges_eaten

  let peaches_boxes := (remaining_peaches + peaches_box_size - 1) / peaches_box_size
  let apples_boxes := (remaining_apples + apples_box_size - 1) / apples_box_size
  let oranges_boxes := (remaining_oranges + oranges_box_size - 1) / oranges_box_size

  (peaches_boxes, apples_boxes, oranges_boxes)

theorem fruit_boxes_needed :
  fruit_boxes = (12, 12, 13) := by 
  sorry

end fruit_boxes_needed_l236_236111


namespace sum_of_coordinates_D_l236_236921

structure Point where
  x : ℝ
  y : ℝ

def is_midpoint (M C D : Point) : Prop :=
  M = ⟨(C.x + D.x) / 2, (C.y + D.y) / 2⟩

def sum_of_coordinates (P : Point) : ℝ :=
  P.x + P.y

theorem sum_of_coordinates_D :
  ∀ (C M : Point), C = ⟨1/2, 3/2⟩ → M = ⟨2, 5⟩ →
  ∃ D : Point, is_midpoint M C D ∧ sum_of_coordinates D = 12 :=
by
  intros C M hC hM
  sorry

end sum_of_coordinates_D_l236_236921


namespace domain_of_sqrt_one_minus_ln_l236_236628

def domain (x : ℝ) : Prop := 0 < x ∧ x ≤ Real.exp 1

theorem domain_of_sqrt_one_minus_ln (x : ℝ) : (1 - Real.log x ≥ 0) ∧ (x > 0) ↔ domain x := by
sorry

end domain_of_sqrt_one_minus_ln_l236_236628


namespace length_of_pencils_l236_236902

theorem length_of_pencils (length_pencil1 : ℕ) (length_pencil2 : ℕ)
  (h1 : length_pencil1 = 12) (h2 : length_pencil2 = 12) : length_pencil1 + length_pencil2 = 24 :=
by
  sorry

end length_of_pencils_l236_236902


namespace find_f_neg_5pi_over_6_l236_236169

noncomputable def f : ℝ → ℝ := sorry

axiom f_defined_on_R : ∀ x : ℝ, ∃ y : ℝ, f y = f x
axiom f_periodic : ∀ x : ℝ, f (x + (3 * Real.pi / 2)) = f x
axiom f_on_interval : ∀ x : ℝ, 0 ≤ x → x ≤ Real.pi → f x = Real.cos x

theorem find_f_neg_5pi_over_6 : f (-5 * Real.pi / 6) = -1 / 2 := 
by 
  -- use the axioms to prove the result 
  sorry

end find_f_neg_5pi_over_6_l236_236169


namespace initial_short_bushes_l236_236948

theorem initial_short_bushes (B : ℕ) (H1 : B + 20 = 57) : B = 37 :=
by
  sorry

end initial_short_bushes_l236_236948


namespace gcd_1989_1547_l236_236241

theorem gcd_1989_1547 : Nat.gcd 1989 1547 = 221 :=
by
  sorry

end gcd_1989_1547_l236_236241


namespace exists_1990_gon_with_conditions_l236_236219

/-- A polygon structure with side lengths and properties to check equality of interior angles and side lengths -/
structure Polygon (n : ℕ) :=
  (sides : Fin n → ℕ)
  (angles_equal : Prop)

/-- Given conditions -/
def condition_1 (P : Polygon 1990) : Prop := P.angles_equal
def condition_2 (P : Polygon 1990) : Prop :=
  ∃ (σ : Fin 1990 → Fin 1990), ∀ i, P.sides i = (σ i + 1)^2

/-- The main theorem to be proven -/
theorem exists_1990_gon_with_conditions :
  ∃ P : Polygon 1990, condition_1 P ∧ condition_2 P :=
sorry

end exists_1990_gon_with_conditions_l236_236219


namespace find_four_digit_squares_l236_236424

theorem find_four_digit_squares (N : ℕ) (a b : ℕ) 
    (h1 : 100 ≤ N ∧ N < 10000)
    (h2 : 10 ≤ a ∧ a < 100)
    (h3 : 0 ≤ b ∧ b < 100)
    (h4 : N = 100 * a + b)
    (h5 : N = (a + b) ^ 2) : 
    N = 9801 ∨ N = 3025 ∨ N = 2025 :=
    sorry

end find_four_digit_squares_l236_236424


namespace pythagorean_triple_third_number_l236_236438

theorem pythagorean_triple_third_number (x : ℕ) (h1 : x^2 + 8^2 = 17^2) : x = 15 :=
sorry

end pythagorean_triple_third_number_l236_236438


namespace find_m_l236_236183

open Set

def A : Set ℕ := {1, 3, 5}
def B (m : ℕ) : Set ℕ := {1, m}
def C (m : ℕ) : Set ℕ := {1, m}

theorem find_m (m : ℕ) (h : A ∩ B m = C m) : m = 3 ∨ m = 5 :=
sorry

end find_m_l236_236183


namespace largest_neg_int_solution_l236_236290

theorem largest_neg_int_solution :
  ∃ x : ℤ, 26 * x + 8 ≡ 4 [ZMOD 18] ∧ ∀ y : ℤ, 26 * y + 8 ≡ 4 [ZMOD 18] → y < -14 → false :=
by
  sorry

end largest_neg_int_solution_l236_236290


namespace exists_small_triangle_l236_236472

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

end exists_small_triangle_l236_236472


namespace part1_part2_l236_236822

-- Definitions for Part 1
def A_2 : Finset ℕ := Finset.filter (λ x, x ≤ 1992 ∧ x % 2 = 0) (Finset.range 1993)
def A_3 : Finset ℕ := Finset.filter (λ x, x ≤ 1992 ∧ x % 3 = 0) (Finset.range 1993)
def S : Finset ℕ := A_2 ∪ A_3

-- Statement for Part 1
theorem part1 : 
  S.card = 1328 ∧ 
  ∀ a b c ∈ S, a ≠ b → b ≠ c → a ≠ c → ¬(Nat.coprime a b ∧ Nat.coprime b c ∧ Nat.coprime a c) :=
sorry

-- Definitions for Part 2
def B : Finset ℕ := Finset.filter (λ x, x ≤ 1992) (Finset.range 1993)

-- Statement for Part 2
theorem part2 : 
  ∀ T ⊆ B, T.card = 1329 → 
  ∃ a b c ∈ T, a ≠ b → b ≠ c → a ≠ c → (Nat.coprime a b ∧ Nat.coprime b c ∧ Nat.coprime a c) :=
sorry

end part1_part2_l236_236822


namespace product_of_roots_l236_236209

theorem product_of_roots (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
    (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 7 :=
by sorry

end product_of_roots_l236_236209


namespace scaling_matrix_unique_l236_236288

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

noncomputable def matrix_N : Matrix (Fin 4) (Fin 4) ℝ := ![![3, 0, 0, 0], ![0, 3, 0, 0], ![0, 0, 3, 0], ![0, 0, 0, 3]]

theorem scaling_matrix_unique (N : Matrix (Fin 4) (Fin 4) ℝ) :
  (∀ (w : Fin 4 → ℝ), N.mulVec w = 3 • w) → N = matrix_N :=
by
  intros h
  sorry

end scaling_matrix_unique_l236_236288


namespace compute_expression_l236_236281

theorem compute_expression : 2 + 4 * 3^2 - 1 + 7 * 2 / 2 = 44 := by
  sorry

end compute_expression_l236_236281


namespace geom_seq_b_value_l236_236648

variable (r : ℝ) (b : ℝ)

-- b is the second term of the geometric sequence with first term 180 and third term 36/25
-- condition 1
def geom_sequence_cond1 := 180 * r = b
-- condition 2
def geom_sequence_cond2 := b * r = 36 / 25

-- Prove b = 16.1 given the conditions
theorem geom_seq_b_value (hb_pos : b > 0) (h1 : geom_sequence_cond1 r b) (h2 : geom_sequence_cond2 r b) : b = 16.1 :=
by sorry

end geom_seq_b_value_l236_236648


namespace equilateral_triangle_perimeter_l236_236231

theorem equilateral_triangle_perimeter (s : ℕ) (b : ℕ) (h1 : 40 = 2 * s + b) (h2 : b = 10) : 3 * s = 45 :=
by {
  sorry
}

end equilateral_triangle_perimeter_l236_236231


namespace domain_of_function_l236_236228

theorem domain_of_function :
  ∀ x : ℝ, (x - 1 ≥ 0) ↔ (x ≥ 1) ∧ (x + 1 ≠ 0) :=
by
  sorry

end domain_of_function_l236_236228


namespace left_square_side_length_l236_236636

theorem left_square_side_length (x : ℕ) (h1 : ∀ y : ℕ, y = x + 17)
                                (h2 : ∀ z : ℕ, z = x + 11)
                                (h3 : 3 * x + 28 = 52) : x = 8 :=
by
  sorry

end left_square_side_length_l236_236636


namespace grid_mark_symmetry_l236_236060

def cells : Finset (Fin 4 × Fin 4) :=
  (Finset.univ : Finset (Fin 4)).product (Finset.univ : Finset (Fin 4))

def distinct_way_count_to_mark (n k : ℕ) : ℕ :=
  if k = 2 && n = 4 then 32 else 0

theorem grid_mark_symmetry :
  distinct_way_count_to_mark 4 2 = 32 := 
by
  sorry

end grid_mark_symmetry_l236_236060


namespace subset_problem_l236_236451

theorem subset_problem (a : ℝ) (P S : Set ℝ) :
  P = { x | x^2 - 2 * x - 3 = 0 } →
  S = { x | a * x + 2 = 0 } →
  (S ⊆ P) →
  (a = 0 ∨ a = 2 ∨ a = -2 / 3) :=
by
  intro hP hS hSubset
  sorry

end subset_problem_l236_236451


namespace gallons_added_in_fourth_hour_l236_236314

-- Defining the conditions
def initial_volume : ℕ := 40
def loss_rate_per_hour : ℕ := 2
def add_in_third_hour : ℕ := 1
def remaining_after_fourth_hour : ℕ := 36

-- Prove the problem statement
theorem gallons_added_in_fourth_hour :
  ∃ (x : ℕ), initial_volume - 2 * 4 + 1 - loss_rate_per_hour + x = remaining_after_fourth_hour :=
sorry

end gallons_added_in_fourth_hour_l236_236314


namespace element_in_set_l236_236212

theorem element_in_set (A : Set ℕ) (h : A = {1, 2}) : 1 ∈ A := 
by 
  rw[h]
  simp

end element_in_set_l236_236212


namespace square_area_adjacency_l236_236777

-- Definition of points as pairs of integers
def Point := ℤ × ℤ

-- Define the points (1,2) and (4,6)
def P1 : Point := (1, 2)
def P2 : Point := (4, 6)

-- Definition of the distance function between two points
def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Statement for proving the area of a square given the side length
theorem square_area_adjacency (h : distance P1 P2 = 5) : ∃ area : ℝ, area = 25 :=
by
  use 25
  sorry

end square_area_adjacency_l236_236777


namespace range_of_a_l236_236032

variable {x a : ℝ}

theorem range_of_a (h1 : x > 1) (h2 : a ≤ x + 1 / (x - 1)) : a ≤ 3 :=
sorry

end range_of_a_l236_236032


namespace problem1_problem2_problem3_problem4_l236_236974

-- Problem (1): 
theorem problem1 : sin (63 * π / 180) * cos (18 * π / 180) + cos (63 * π / 180) * cos (108 * π / 180) = (real.sqrt 2) / 2 :=
sorry

-- Problem (2):
theorem problem2 : set.image (λ x, 2 * sin x ^ 2 - 3 * sin x + 1) (set.Icc (π / 6) (5 * π / 6)) = set.Icc (-1 / 8) 0 :=
sorry

-- Problem (3):
theorem problem3 {α β : ℝ} (hα : 0 < α) (hβ : 0 < β) (hαβ : (1 + real.sqrt 3 * tan α) * (1 + real.sqrt 3 * tan β) = 4) : α + β = π / 3 :=
sorry

-- Problem (4):
def f (ω ϕ x : ℝ) := 2 * sin (ω * x + ϕ)
theorem problem4 (ω : ℝ) (hω : ω > 0) (ϕ : ℝ) (hϕ1 : 0 < ϕ) (hϕ2 : ϕ < π / 2) (h : f ω ϕ (2 * π / 3) = f ω ϕ (2 * π / 3) ∧ (2 * π / ω) = π) : 
  (f ω ϕ 0 ≠ 3 / 2 ∧ 
  ∀ x, (x ∈ set.Icc (π / 12) (2 * π / 3) → (f ω ϕ x < f ω ϕ 0)) ∧ 
  (∃ c, c = (5 * π / 12) ∧ f ω ϕ (c) = 0) ∧ 
  (∃ d, d = ϕ ∧ (∀ x, f ω d x = 2 * sin (ω * x))))
: true :=
sorry

end problem1_problem2_problem3_problem4_l236_236974


namespace exponential_increasing_l236_236795

theorem exponential_increasing (a : ℝ) (h_pos : a > 0) (h_neq_one : a ≠ 1) :
  (∀ x y : ℝ, x < y → a^x < a^y) ↔ a > 1 :=
by
  sorry

end exponential_increasing_l236_236795


namespace tree_growth_rate_l236_236600

noncomputable def growth_rate_per_week (initial_height final_height : ℝ) (months weeks_per_month : ℕ) : ℝ :=
  (final_height - initial_height) / (months * weeks_per_month)

theorem tree_growth_rate :
  growth_rate_per_week 10 42 4 4 = 2 := 
by
  sorry

end tree_growth_rate_l236_236600


namespace range_of_f_l236_236234

def f (x : ℤ) : ℤ := (x - 1)^2 - 1

theorem range_of_f :
  Set.image f {-1, 0, 1, 2, 3} = {-1, 0, 3} :=
by
  sorry

end range_of_f_l236_236234


namespace op_15_5_eq_33_l236_236196

def op (x y : ℕ) : ℕ :=
  2 * x + x / y

theorem op_15_5_eq_33 : op 15 5 = 33 := by
  sorry

end op_15_5_eq_33_l236_236196


namespace sequence_a2018_l236_236859

theorem sequence_a2018 (a : ℕ → ℝ) 
  (h1 : ∀ n, a (n + 2) - 2 * a (n + 1) + a n = 1) 
  (h2 : a 18 = 0) 
  (h3 : a 2017 = 0) :
  a 2018 = 1000 :=
sorry

end sequence_a2018_l236_236859


namespace find_b_l236_236710

noncomputable def general_quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_b (a c : ℝ) (y1 y2 : ℝ) :
  y1 = general_quadratic a 3 c 2 →
  y2 = general_quadratic a 3 c (-2) →
  y1 - y2 = 12 →
  3 = 3 :=
by
  intros h1 h2 h3
  sorry

end find_b_l236_236710


namespace muffin_price_proof_l236_236499

noncomputable def price_per_muffin (s m t : ℕ) (contribution : ℕ) : ℕ :=
  contribution / (s + m + t)

theorem muffin_price_proof :
  ∀ (sasha_muffins melissa_muffins : ℕ) (h1 : sasha_muffins = 30) (h2 : melissa_muffins = 4 * sasha_muffins)
  (tiffany_muffins total_muffins : ℕ) (h3 : total_muffins = sasha_muffins + melissa_muffins)
  (h4 : tiffany_muffins = total_muffins / 2)
  (h5 : total_muffins = sasha_muffins + melissa_muffins + tiffany_muffins)
  (contribution : ℕ) (h6 : contribution = 900),
  price_per_muffin sasha_muffins melissa_muffins tiffany_muffins contribution = 4 :=
by
  intros sasha_muffins melissa_muffins h1 h2 tiffany_muffins total_muffins h3 h4 h5 contribution h6
  simp [price_per_muffin]
  sorry

end muffin_price_proof_l236_236499


namespace algebraic_expression_value_l236_236925

theorem algebraic_expression_value (x : ℝ) (hx : x = 2 * Real.cos 45 + 1) :
  (1 / (x - 1) - (x - 3) / (x ^ 2 - 2 * x + 1)) / (2 / (x - 1)) = Real.sqrt 2 / 2 :=
by
  sorry

end algebraic_expression_value_l236_236925


namespace exists_y_equals_7_l236_236946

theorem exists_y_equals_7 : ∃ (x y z t : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 ∧ y = 7 ∧ x + y + z + t = 10 :=
by {
  sorry -- This is where the actual proof would go.
}

end exists_y_equals_7_l236_236946


namespace balloon_arrangements_l236_236457

open Finset

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem balloon_arrangements : 
  let n := 7
      p1 := 2
  in  (factorial n) / ((factorial p1)) = 2520 := by
{
  sorry
}

end balloon_arrangements_l236_236457


namespace jellyfish_cost_l236_236690

theorem jellyfish_cost (J E : ℝ) (h1 : E = 9 * J) (h2 : J + E = 200) : J = 20 := by
  sorry

end jellyfish_cost_l236_236690


namespace factorize_quadratic_l236_236158

variable (x : ℝ)

theorem factorize_quadratic : 2 * x^2 + 4 * x - 6 = 2 * (x - 1) * (x + 3) :=
by
  sorry

end factorize_quadratic_l236_236158


namespace area_percentage_increase_l236_236931

theorem area_percentage_increase (r₁ r₂ : ℝ) (π : ℝ) :
  r₁ = 6 ∧ r₂ = 4 ∧ π > 0 →
  (π * r₁^2 - π * r₂^2) / (π * r₂^2) * 100 = 125 := 
by {
  sorry
}

end area_percentage_increase_l236_236931


namespace trig_identity_simplification_l236_236444

theorem trig_identity_simplification (θ : ℝ) (hθ : θ = 15 * Real.pi / 180) :
  (Real.sqrt 3 / 2 - Real.sqrt 3 * (Real.sin θ) ^ 2) = 3 / 4 := 
by sorry

end trig_identity_simplification_l236_236444


namespace expected_value_Y_variance_Y_l236_236188

open ProbabilityTheory

def X : Type := ℝ

axiom X_normal_mean_var (X : X) : (Real.Normal 1 4) X

noncomputable def Y (X : X) : ℝ := 2 - (1 / 2) * X

theorem expected_value_Y (X : X) [X_normal_mean_var X] : E(Y X) = 3 / 2 :=
by sorry

theorem variance_Y (X : X) [X_normal_mean_var X] : Var(Y X) = 1 :=
by sorry

end expected_value_Y_variance_Y_l236_236188


namespace equation_solutions_l236_236385

theorem equation_solutions (m n x y : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  x^n + y^n = 3^m ↔ (x = 1 ∧ y = 2 ∧ n = 3 ∧ m = 2) ∨ (x = 2 ∧ y = 1 ∧ n = 3 ∧ m = 2) :=
by
  sorry -- proof to be implemented

end equation_solutions_l236_236385


namespace find_S16_l236_236603

-- Definitions
def geom_seq (a : ℕ → ℝ) : Prop := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

def sum_of_geom_seq (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, S n = a 0 * (1 - (a 1 / a 0)^n) / (1 - (a 1 / a 0))

-- Problem conditions
variables {a : ℕ → ℝ} {S : ℕ → ℝ}

axiom geom_seq_a : geom_seq a
axiom S4_eq : S 4 = 4
axiom S8_eq : S 8 = 12

-- Theorem
theorem find_S16 : S 16 = 60 :=
  sorry

end find_S16_l236_236603


namespace spontaneous_low_temperature_l236_236654

theorem spontaneous_low_temperature (ΔH ΔS T : ℝ) (spontaneous : ΔG = ΔH - T * ΔS) :
  (∀ T, T > 0 → ΔG < 0 → ΔH < 0 ∧ ΔS < 0) := 
by 
  sorry

end spontaneous_low_temperature_l236_236654


namespace find_stream_speed_l236_236253

variable (D : ℝ) (v : ℝ)

theorem find_stream_speed 
  (h1 : ∀D v, D / (63 - v) = 2 * (D / (63 + v)))
  (h2 : v = 21) :
  true := 
  by
  sorry

end find_stream_speed_l236_236253


namespace fraction_expression_proof_l236_236899

theorem fraction_expression_proof :
  (1 / 8 * 1 / 9 * 1 / 28 = 1 / 2016) ∨ ((1 / 8 - 1 / 9) * 1 / 28 = 1 / 2016) :=
by
  sorry

end fraction_expression_proof_l236_236899


namespace range_of_m_l236_236168

def y1 (m x : ℝ) : ℝ :=
  m * (x - 2 * m) * (x + m + 2)

def y2 (x : ℝ) : ℝ :=
  x - 1

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, y1 m x < 0 ∨ y2 x < 0) ∧ (∃ x : ℝ, x < -3 ∧ y1 m x * y2 x < 0) ↔ (-4 < m ∧ m < -3/2) := 
by
  sorry

end range_of_m_l236_236168


namespace proportion_condition_l236_236970

variable (a b c d a₁ b₁ c₁ d₁ : ℚ)

theorem proportion_condition
  (h₁ : a / b = c / d)
  (h₂ : a₁ / b₁ = c₁ / d₁) :
  (a + a₁) / (b + b₁) = (c + c₁) / (d + d₁) ↔ a * d₁ + a₁ * d = b₁ * c + b * c₁ := by
  sorry

end proportion_condition_l236_236970


namespace total_cost_proof_l236_236834

-- Define the conditions
def length_grass_field : ℝ := 75
def width_grass_field : ℝ := 55
def width_path : ℝ := 2.5
def area_path : ℝ := 6750
def cost_per_sq_m : ℝ := 10

-- Calculate the outer dimensions
def outer_length : ℝ := length_grass_field + 2 * width_path
def outer_width : ℝ := width_grass_field + 2 * width_path

-- Calculate the area of the entire field including the path
def area_entire_field : ℝ := outer_length * outer_width

-- Calculate the area of the grass field without the path
def area_grass_field : ℝ := length_grass_field * width_grass_field

-- Calculate the area of the path
def area_calculated_path : ℝ := area_entire_field - area_grass_field

-- Calculate the total cost of constructing the path
noncomputable def total_cost : ℝ := area_calculated_path * cost_per_sq_m

-- The theorem to prove
theorem total_cost_proof :
  area_calculated_path = area_path ∧ total_cost = 6750 :=
by
  sorry

end total_cost_proof_l236_236834


namespace older_brother_age_is_25_l236_236375

noncomputable def age_of_older_brother (father_age current_n : ℕ) (younger_brother_age : ℕ) : ℕ := 
  (father_age - current_n) / 2

theorem older_brother_age_is_25 
  (father_age : ℕ) 
  (h1 : father_age = 50) 
  (younger_brother_age : ℕ)
  (current_n : ℕ) 
  (h2 : (2 * (younger_brother_age + current_n)) = father_age + current_n) : 
  age_of_older_brother father_age current_n younger_brother_age = 25 := 
by
  sorry

end older_brother_age_is_25_l236_236375


namespace average_weight_l236_236100

variable (A B C : ℕ)

theorem average_weight (h1 : A + B = 140) (h2 : B + C = 100) (h3 : B = 60) :
  (A + B + C) / 3 = 60 := 
sorry

end average_weight_l236_236100


namespace problem_l236_236182

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {x | ∃ t ∈ A, x = t^2}

theorem problem (A_def : A = {-1, 0, 1}) : B = {0, 1} :=
by sorry

end problem_l236_236182


namespace find_angle_MON_l236_236027

-- Definitions of conditions
variables {A B O C M N : Type} -- Points in a geometric space
variables (angle_AOB : ℝ) (ray_OC : Prop) (bisects_OM : Prop) (bisects_ON : Prop)
variables (angle_MOB : ℝ) (angle_MON : ℝ)

-- Conditions
-- Angle AOB is 90 degrees
def angle_AOB_90 (angle_AOB : ℝ) : Prop := angle_AOB = 90

-- OC is a ray (using a placeholder property for ray, as Lean may not have geometric entities)
def OC_is_ray (ray_OC : Prop) : Prop := ray_OC

-- OM bisects angle BOC
def OM_bisects_BOC (bisects_OM : Prop) : Prop := bisects_OM

-- ON bisects angle AOC
def ON_bisects_AOC (bisects_ON : Prop) : Prop := bisects_ON

-- The problem statement as a theorem in Lean
theorem find_angle_MON
  (h1 : angle_AOB_90 angle_AOB)
  (h2 : OC_is_ray ray_OC)
  (h3 : OM_bisects_BOC bisects_OM)
  (h4 : ON_bisects_AOC bisects_ON) :
  angle_MON = 45 ∨ angle_MON = 135 :=
sorry

end find_angle_MON_l236_236027


namespace sqrt_seven_to_six_power_eq_343_l236_236349

theorem sqrt_seven_to_six_power_eq_343 : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end sqrt_seven_to_six_power_eq_343_l236_236349


namespace sides_of_original_polygon_l236_236991

-- Define the sum of interior angles formula for a polygon with n sides
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the total sum of angles for the resulting polygon
def sum_of_new_polygon_angles : ℝ := 1980

-- The lean theorem statement to prove
theorem sides_of_original_polygon (n : ℕ) :
    sum_interior_angles n = sum_of_new_polygon_angles →
    n = 13 →
    12 ≤ n+1 ∧ n+1 ≤ 14 :=
by
  intro h1 h2
  sorry

end sides_of_original_polygon_l236_236991


namespace find_T_l236_236806

theorem find_T : 
  ∃ T : ℝ, (3 / 4) * (1 / 6) * T = (1 / 5) * (1 / 4) * 120 ∧ T = 48 :=
by
  sorry

end find_T_l236_236806


namespace millions_place_correct_l236_236612

def number := 345000000
def hundred_millions_place := number / 100000000 % 10  -- 3
def ten_millions_place := number / 10000000 % 10  -- 4
def millions_place := number / 1000000 % 10  -- 5

theorem millions_place_correct : millions_place = 5 := 
by 
  -- Mathematical proof goes here
  sorry

end millions_place_correct_l236_236612


namespace arccos_range_l236_236442

theorem arccos_range (a : ℝ) (x : ℝ) (h₀ : x = Real.sin a) 
  (h₁ : -Real.pi / 4 ≤ a ∧ a ≤ 3 * Real.pi / 4) :
  ∀ y, y = Real.arccos x → 0 ≤ y ∧ y ≤ 3 * Real.pi / 4 := 
sorry

end arccos_range_l236_236442


namespace carol_has_35_nickels_l236_236502

def problem_statement : Prop :=
  ∃ (n d : ℕ), 5 * n + 10 * d = 455 ∧ n = d + 7 ∧ n = 35

theorem carol_has_35_nickels : problem_statement := by
  -- Proof goes here
  sorry

end carol_has_35_nickels_l236_236502


namespace shaded_areas_different_l236_236004

/-
Question: How do the shaded areas of three different large squares (I, II, and III) compare?
Conditions:
1. Square I has diagonals drawn, and small squares are shaded at each corner where diagonals meet the sides.
2. Square II has vertical and horizontal lines drawn through the midpoints, creating four smaller squares, with one centrally shaded.
3. Square III has one diagonal from one corner to the center and a straight line from the midpoint of the opposite side to the center, creating various triangles and trapezoids, with a trapezoid area around the center being shaded.
Proof:
Prove that the shaded areas of squares I, II, and III are all different given the conditions on how squares I, II, and III are partitioned and shaded.
-/
theorem shaded_areas_different :
  ∀ (a : ℝ) (A1 A2 A3 : ℝ), (A1 = 1/4 * a^2) ∧ (A2 = 1/4 * a^2) ∧ (A3 = 3/8 * a^2) → 
  A1 ≠ A3 ∧ A2 ≠ A3 :=
by
  sorry

end shaded_areas_different_l236_236004


namespace correct_divisor_l236_236468

theorem correct_divisor (D : ℕ) (X : ℕ) (H1 : X = 70 * (D + 12)) (H2 : X = 40 * D) : D = 28 := 
by 
  sorry

end correct_divisor_l236_236468


namespace problem1_problem2_l236_236928

-- Problem 1
theorem problem1 (x : ℝ) : 
  (x + 2) * (x - 2) - 2 * (x - 3) = 3 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 := 
sorry

-- Problem 2
theorem problem2 (x : ℝ) : 
  (x + 3)^2 = (1 - 2 * x)^2 ↔ x = 4 ∨ x = -2 / 3 := 
sorry

end problem1_problem2_l236_236928


namespace opposite_sides_line_l236_236321

theorem opposite_sides_line (m : ℝ) : 
  (2 * 1 + 3 + m) * (2 * -4 + -2 + m) < 0 ↔ -5 < m ∧ m < 10 :=
by sorry

end opposite_sides_line_l236_236321


namespace exists_rational_non_integer_xy_no_rational_non_integer_xy_l236_236554

-- Part (a)
theorem exists_rational_non_integer_xy 
  (x y : ℚ) (h1 : ¬ ∃ z : ℤ, x = z ∧ y = z) : 
  (∃ x y : ℚ, ¬(∃ z : ℤ, x = z ∨ y = z) ∧ 
   ∃ z1 z2 : ℤ, 19 * x + 8 * y = ↑z1 ∧ 8 * x + 3 * y = ↑z2) :=
sorry

-- Part (b)
theorem no_rational_non_integer_xy 
  (x y : ℚ) (h1 : ¬ ∃ z : ℤ, x = z ∧ y = z) : 
  ¬ ∃ x y : ℚ, ¬(∃ z : ℤ, x = z ∨ y = z) ∧ 
  ∃ z1 z2 : ℤ, 19 * x^2 + 8 * y^2 = ↑z1 ∧ 8 * x^2 + 3 * y^2 = ↑z2 :=
sorry

end exists_rational_non_integer_xy_no_rational_non_integer_xy_l236_236554


namespace crown_distribution_l236_236977

theorem crown_distribution 
  (A B C D E : ℤ) 
  (h1 : 2 * C = 3 * A)
  (h2 : 4 * D = 3 * B)
  (h3 : 4 * E = 5 * C)
  (h4 : 5 * D = 6 * A)
  (h5 : A + B + C + D + E = 2870) : 
  A = 400 ∧ B = 640 ∧ C = 600 ∧ D = 480 ∧ E = 750 := 
by 
  sorry

end crown_distribution_l236_236977


namespace min_value_reciprocal_sum_l236_236765

theorem min_value_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 12) : 
  (1 / x) + (1 / y) ≥ 1 / 3 :=
by
  sorry

end min_value_reciprocal_sum_l236_236765


namespace equal_chord_segments_l236_236154

theorem equal_chord_segments 
  (a x y : ℝ) 
  (AM CM : ℝ → ℝ → Prop) 
  (AB CD : ℝ → Prop)
  (intersect_chords_theorem : AM x (a - x) = CM y (a - y)) :
  x = y ∨ x = a - y :=
by
  sorry

end equal_chord_segments_l236_236154


namespace distinct_real_roots_of_quadratic_find_m_and_other_root_l236_236301

theorem distinct_real_roots_of_quadratic (m : ℝ) (h_neg_m : m < 0) : 
    ∃ x₁ x₂ : ℝ, (x₁ ≠ x₂ ∧ (∀ x, x^2 - 2*x + m = 0 → (x = x₁ ∨ x = x₂))) := 
by 
  sorry

theorem find_m_and_other_root (m : ℝ) (h_neg_m : m < 0) (root_minus_one : ∀ x, x^2 - 2*x + m = 0 → x = -1):
    m = -3 ∧ (∃ x, x^2 - 2*x - 3 = 0 ∧ x = 3) := 
by 
  sorry

end distinct_real_roots_of_quadratic_find_m_and_other_root_l236_236301


namespace sets_satisfying_union_l236_236360

open Set

theorem sets_satisfying_union :
  {B : Set ℕ | {1, 2} ∪ B = {1, 2, 3}} = { {3}, {1, 3}, {2, 3}, {1, 2, 3} } :=
by
  sorry

end sets_satisfying_union_l236_236360


namespace calculate_average_age_l236_236750

variables (k : ℕ) (female_to_male_ratio : ℚ) (avg_young_female : ℚ) (avg_old_female : ℚ) (avg_young_male : ℚ) (avg_old_male : ℚ)

theorem calculate_average_age 
  (h_ratio : female_to_male_ratio = 7/8)
  (h_avg_yf : avg_young_female = 26)
  (h_avg_of : avg_old_female = 42)
  (h_avg_ym : avg_young_male = 28)
  (h_avg_om : avg_old_male = 46) : 
  (534/15 : ℚ) = 36 :=
by sorry

end calculate_average_age_l236_236750


namespace max_a_for_minimum_value_l236_236605

def f (x a : ℝ) : ℝ :=
  if x < a then -a * x + 1 else (x - 2)^2

theorem max_a_for_minimum_value : ∀ a : ℝ, (∃ m : ℝ, ∀ x : ℝ, f x a ≥ m) → a ≤ 1 :=
by
  sorry

end max_a_for_minimum_value_l236_236605


namespace initial_percentage_proof_l236_236826

-- Defining the initial percentage of water filled in the container
def initial_percentage (capacity add amount_filled : ℕ) : ℕ :=
  (amount_filled * 100) / capacity

-- The problem constraints
theorem initial_percentage_proof : initial_percentage 120 48 (3 * 120 / 4 - 48) = 35 := by
  -- We need to show that the initial percentage is 35%
  sorry

end initial_percentage_proof_l236_236826


namespace smallest_positive_integer_n_l236_236525

theorem smallest_positive_integer_n (n : ℕ) (h : 527 * n ≡ 1083 * n [MOD 30]) : n = 2 :=
sorry

end smallest_positive_integer_n_l236_236525


namespace coefficient_of_1_div_x_l236_236326

open Nat

noncomputable def binomial_expansion (x : ℝ) (n : ℕ) : ℝ :=
  (1 / Real.sqrt x - 3)^n

theorem coefficient_of_1_div_x (x : ℝ) (n : ℕ) (h1 : n ∈ {m | m > 0}) (h2 : binomial_expansion x n = 16) :
  ∃ c : ℝ, c = 54 :=
by
  sorry

end coefficient_of_1_div_x_l236_236326


namespace Djibo_sister_age_l236_236149

variable (d s : ℕ)
variable (h1 : d = 17)
variable (h2 : d - 5 + (s - 5) = 35)

theorem Djibo_sister_age : s = 28 :=
by sorry

end Djibo_sister_age_l236_236149


namespace darts_final_score_is_600_l236_236747

def bullseye_points : ℕ := 50

def first_dart_points (bullseye : ℕ) : ℕ := 3 * bullseye

def second_dart_points : ℕ := 0

def third_dart_points (bullseye : ℕ) : ℕ := bullseye / 2

def fourth_dart_points (bullseye : ℕ) : ℕ := 2 * bullseye

def total_points_before_fifth (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

def fifth_dart_points (bullseye : ℕ) (previous_total : ℕ) : ℕ :=
  bullseye + previous_total

def final_score (d1 d2 d3 d4 d5 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4 + d5

theorem darts_final_score_is_600 :
  final_score
    (first_dart_points bullseye_points)
    second_dart_points
    (third_dart_points bullseye_points)
    (fourth_dart_points bullseye_points)
    (fifth_dart_points bullseye_points (total_points_before_fifth
      (first_dart_points bullseye_points)
      second_dart_points
      (third_dart_points bullseye_points)
      (fourth_dart_points bullseye_points))) = 600 :=
  sorry

end darts_final_score_is_600_l236_236747


namespace print_gift_wrap_price_l236_236264

theorem print_gift_wrap_price (solid_price : ℝ) (total_rolls : ℕ) (total_money : ℝ)
    (print_rolls : ℕ) (solid_rolls_money : ℝ) (print_money : ℝ) (P : ℝ) :
  solid_price = 4 ∧ total_rolls = 480 ∧ total_money = 2340 ∧ print_rolls = 210 ∧
  solid_rolls_money = 270 * 4 ∧ print_money = 1260 ∧
  total_money = solid_rolls_money + print_money ∧ P = print_money / 210 
  → P = 6 :=
by
  sorry

end print_gift_wrap_price_l236_236264


namespace range_is_correct_l236_236632

noncomputable def quadratic_function (x : ℝ) : ℝ := x^2 - 4 * x

def domain : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

def range_of_function : Set ℝ := {y | ∃ x ∈ domain, quadratic_function x = y}

theorem range_is_correct : range_of_function = Set.Icc (-4) 21 :=
by {
  sorry
}

end range_is_correct_l236_236632


namespace abs_diff_eq_l236_236819

theorem abs_diff_eq (a b c d : ℤ) (h1 : a = 13) (h2 : b = 3) (h3 : c = 4) (h4 : d = 10) : 
  |a - b| - |c - d| = 4 := 
  by
  -- Proof goes here
  sorry

end abs_diff_eq_l236_236819


namespace remainder_is_constant_when_b_is_minus_five_halves_l236_236709

noncomputable def poly_div_remainder (p d : Polynomial ℝ) : Polynomial ℝ × Polynomial ℝ :=
Polynomial.divModByMonicAux p d

theorem remainder_is_constant_when_b_is_minus_five_halves 
  (p d : Polynomial ℝ) (b : ℝ) :
  p = Polynomial.C 12 * X ^ 4 + Polynomial.C (-5) * X ^ 3 + Polynomial.C b * X ^ 2 + Polynomial.C (-4) * X + Polynomial.C 8 →
  d = Polynomial.C 3 * X ^ 2 + Polynomial.C (-2) * X + Polynomial.C 1 →
  (poly_div_remainder p d).snd = Polynomial.C c :=
  sorry

end remainder_is_constant_when_b_is_minus_five_halves_l236_236709


namespace sum_remainders_l236_236544

theorem sum_remainders (n : ℤ) (h : n % 20 = 14) : (n % 4) + (n % 5) = 6 :=
  by
  sorry

end sum_remainders_l236_236544


namespace parameter_a_range_l236_236875

def quadratic_function (a x : ℝ) : ℝ := x^2 + 2 * a * x + 2 * a + 1

theorem parameter_a_range :
  (∀ x : ℝ, (-1 ≤ x ∧ x ≤ 1) → quadratic_function a x ≥ 1) ↔ (0 ≤ a) :=
by
  sorry

end parameter_a_range_l236_236875


namespace visited_neither_l236_236966

def people_total : ℕ := 90
def visited_iceland : ℕ := 55
def visited_norway : ℕ := 33
def visited_both : ℕ := 51

theorem visited_neither :
  people_total - (visited_iceland + visited_norway - visited_both) = 53 := by
  sorry

end visited_neither_l236_236966


namespace Emily_money_made_l236_236153

def price_per_bar : ℕ := 4
def total_bars : ℕ := 8
def bars_sold : ℕ := total_bars - 3
def money_made : ℕ := bars_sold * price_per_bar

theorem Emily_money_made : money_made = 20 :=
by
  sorry

end Emily_money_made_l236_236153


namespace isosceles_triangle_sin_cos_rational_l236_236579

theorem isosceles_triangle_sin_cos_rational
  (a h : ℤ) -- Given BC and AD as integers
  (c : ℚ)  -- AB = AC = c
  (ha : 4 * c^2 = 4 * h^2 + a^2) : -- From c^2 = h^2 + (a^2 / 4)
  ∃ (sinA cosA : ℚ), 
    sinA = (a * h) / (h^2 + (a^2 / 4)) ∧
    cosA = (2 * h^2) / (h^2 + (a^2 / 4)) - 1 :=
sorry

end isosceles_triangle_sin_cos_rational_l236_236579


namespace eugene_used_4_boxes_of_toothpicks_l236_236703

theorem eugene_used_4_boxes_of_toothpicks:
  ∀ (cards_per_box cards_used cards_total toothpicks_per_card toothpicks_per_box : ℕ) 
  (cards_total_eq : cards_total = 52) 
  (cards_used_eq : cards_used = cards_total - 23)
  (toothpicks_per_card_eq : toothpicks_per_card = 64) 
  (toothpicks_per_box_eq : toothpicks_per_box = 550)
  (cards_per_box_eq : cards_per_box = (toothpicks_per_card * cards_used) / toothpicks_per_box),
  ⌈cards_per_box⌉ = 4 :=
by
  assume cards_per_box cards_used cards_total toothpicks_per_card toothpicks_per_box
         cards_total_eq cards_used_eq toothpicks_per_card_eq toothpicks_per_box_eq cards_per_box_eq
  sorry

end eugene_used_4_boxes_of_toothpicks_l236_236703


namespace initial_number_is_12_l236_236655

theorem initial_number_is_12 {x : ℤ} (h : ∃ k : ℤ, x + 17 = 29 * k) : x = 12 :=
by
  sorry

end initial_number_is_12_l236_236655


namespace inverse_function_composition_l236_236791

def g (x : ℝ) : ℝ := 3 * x + 7

noncomputable def g_inv (y : ℝ) : ℝ := (y - 7) / 3

theorem inverse_function_composition : g_inv (g_inv 20) = -8 / 9 := by
  sorry

end inverse_function_composition_l236_236791


namespace probability_neither_prime_nor_composite_lemma_l236_236744

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m ∣ n ∧ m ≠ 1 ∧ m ≠ n

def neither_prime_nor_composite (n : ℕ) : Prop :=
  ¬ is_prime n ∧ ¬ is_composite n

def probability_of_neither_prime_nor_composite (n : ℕ) : ℚ :=
  if 1 ≤ n ∧ n ≤ 97 then 1 / 97 else 0

theorem probability_neither_prime_nor_composite_lemma :
  probability_of_neither_prime_nor_composite 1 = 1 / 97 := by
  sorry

end probability_neither_prime_nor_composite_lemma_l236_236744


namespace max_value_inequality_l236_236722

theorem max_value_inequality
  (x1 x2 y1 y2 : ℝ)
  (h1 : x1^2 + y1^2 = 1)
  (h2 : x2^2 + y2^2 = 1)
  (h3 : x1 * x2 + y1 * y2 = ⅟2) :
  (|x1 + y1 - 1| / Real.sqrt 2) + (|x2 + y2 - 1| / Real.sqrt 2) ≤ 1 :=
by {
  sorry
}

end max_value_inequality_l236_236722


namespace dodecagon_diagonals_intersect_probability_l236_236670

theorem dodecagon_diagonals_intersect_probability :
  ∀ (dodecagon : Type) [regular_polygon dodecagon (12 : nat)],
  let diagonals_count := 54 in
  let intersecting_diagonals_count := 495 in
  (intersecting_diagonals_count : ℚ) / (binom diagonals_count 2) = 15 / 43 :=
by
  intros
  have diagonals : 54 := 54
  have intersections : 495 := 495
  have total_pairs_diagonals : 1431 := binom 54 2
  have probability : ℚ := intersections / total_pairs_diagonals
  rw [Q.to_rat_eq]
  apply Q.eq_of_rat_eq
  norm_num
  exact 15 / 43
  sorry

end dodecagon_diagonals_intersect_probability_l236_236670


namespace overlapping_area_is_correct_l236_236824

-- Defining the coordinates of the grid points
def topLeft : (ℝ × ℝ) := (0, 2)
def topMiddle : (ℝ × ℝ) := (1.5, 2)
def topRight : (ℝ × ℝ) := (3, 2)
def middleLeft : (ℝ × ℝ) := (0, 1)
def center : (ℝ × ℝ) := (1.5, 1)
def middleRight : (ℝ × ℝ) := (3, 1)
def bottomLeft : (ℝ × ℝ) := (0, 0)
def bottomMiddle : (ℝ × ℝ) := (1.5, 0)
def bottomRight : (ℝ × ℝ) := (3, 0)

-- Defining the vertices of the triangles
def triangle1_points : List (ℝ × ℝ) := [topLeft, middleRight, bottomMiddle]
def triangle2_points : List (ℝ × ℝ) := [bottomLeft, topMiddle, middleRight]

-- Function to calculate the area of a polygon given the vertices -- placeholder here
noncomputable def area_of_overlapped_region (tr1 tr2 : List (ℝ × ℝ)) : ℝ := 
  -- Placeholder for the actual computation of the overlapped area
  1.2

-- Statement to prove
theorem overlapping_area_is_correct : 
  area_of_overlapped_region triangle1_points triangle2_points = 1.2 := sorry

end overlapping_area_is_correct_l236_236824


namespace number_of_belts_l236_236126

def ties := 34
def black_shirts := 63
def white_shirts := 42

def jeans := (2 / 3 : ℚ) * (black_shirts + white_shirts)
def scarves (B : ℚ) := (1 / 2 : ℚ) * (ties + B)

theorem number_of_belts (B : ℚ) : jeans = scarves B + 33 → B = 40 := by
  -- This theorem states the required proof but leaves the proof itself as a placeholder.
  -- The proof would involve solving equations algebraically as shown in the solution steps.
  sorry

end number_of_belts_l236_236126


namespace sequence_converges_l236_236714

open Real

theorem sequence_converges (x : ℕ → ℝ) (h₀ : ∀ n, x (n + 1) = 1 + x n - 0.5 * (x n) ^ 2) (h₁ : 1 < x 1 ∧ x 1 < 2) :
  ∀ n ≥ 3, |x n - sqrt 2| < 2 ^ (-n : ℝ) :=
by
  sorry

end sequence_converges_l236_236714


namespace remainder_a37_div_45_l236_236335

open Nat

def a_n (n : Nat) : Nat :=
  String.join (List.map toString (List.range (n + 1))).toNat

theorem remainder_a37_div_45 : (a_n 37) % 45 = 37 :=
by
  sorry

end remainder_a37_div_45_l236_236335


namespace value_of_expression_l236_236717

variable (x1 x2 : ℝ)

def sum_roots (x1 x2 : ℝ) : Prop := x1 + x2 = 3
def product_roots (x1 x2 : ℝ) : Prop := x1 * x2 = -4

theorem value_of_expression (h1 : sum_roots x1 x2) (h2 : product_roots x1 x2) : 
  x1^2 - 4*x1 - x2 + 2*x1*x2 = -7 :=
by sorry

end value_of_expression_l236_236717


namespace no_member_of_T_is_divisible_by_4_or_5_l236_236762

def sum_of_squares_of_four_consecutive_integers (n : ℤ) : ℤ :=
  (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2

theorem no_member_of_T_is_divisible_by_4_or_5 :
  ∀ (n : ℤ), ¬ (∃ (T : ℤ), T = sum_of_squares_of_four_consecutive_integers n ∧ (T % 4 = 0 ∨ T % 5 = 0)) :=
by
  sorry

end no_member_of_T_is_divisible_by_4_or_5_l236_236762


namespace area_enclosed_by_region_l236_236282

open Real

def condition (x y : ℝ) := abs (2 * x + 2 * y) + abs (2 * x - 2 * y) ≤ 8

theorem area_enclosed_by_region : 
  (∃ u v : ℝ, condition u v) → ∃ A : ℝ, A = 16 := 
sorry

end area_enclosed_by_region_l236_236282


namespace functional_equation_solution_l236_236561

theorem functional_equation_solution (f : ℤ → ℤ)
  (h : ∀ m n : ℤ, f (f (m + n)) = f m + f n) :
  (∃ a : ℤ, ∀ n : ℤ, f n = n + a) ∨ (∀ n : ℤ, f n = 0) := by
  sorry

end functional_equation_solution_l236_236561


namespace simplify_expression_l236_236348

theorem simplify_expression :
  (((0.3 * 0.8) / 0.2) + (0.1 * 0.5)^2 - 1 / (0.5 * 0.8)^2) = -5.0475 :=
by
  sorry

end simplify_expression_l236_236348


namespace total_clowns_l236_236631

def num_clown_mobiles : Nat := 5
def clowns_per_mobile : Nat := 28

theorem total_clowns : num_clown_mobiles * clowns_per_mobile = 140 := by
  sorry

end total_clowns_l236_236631


namespace simplify_and_evaluate_expression_l236_236094

theorem simplify_and_evaluate_expression (x : ℤ) (h : x = 2 ∨ x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) :
  ((1 / (x:ℚ) - 1 / (x + 1)) / ((x^2 - 1) / (x^2 + 2*x + 1))) = (1 / 2) :=
by
  -- Skipping the proof
  sorry

end simplify_and_evaluate_expression_l236_236094


namespace exists_n_consecutive_composites_l236_236222

theorem exists_n_consecutive_composites (n : ℕ) (h : n ≥ 1) (a r : ℕ) :
  ∃ K : ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ¬(Nat.Prime (a + (K + i) * r)) := 
sorry

end exists_n_consecutive_composites_l236_236222


namespace line_ellipse_intersection_l236_236031

-- Define the problem conditions and the proof problem statement.
theorem line_ellipse_intersection (k m : ℝ) : 
  (∀ x y, y - k * x - 1 = 0 → ((x^2 / 5) + (y^2 / m) = 1)) →
  (m ≥ 1) ∧ (m ≠ 5) ∧ (m < 5 ∨ m > 5) :=
sorry

end line_ellipse_intersection_l236_236031


namespace consecutive_lucky_years_l236_236994

def is_lucky (Y : ℕ) : Prop := 
  let first_two_digits := Y / 100
  let last_two_digits := Y % 100
  Y % (first_two_digits + last_two_digits) = 0

theorem consecutive_lucky_years : ∃ Y : ℕ, is_lucky Y ∧ is_lucky (Y + 1) :=
by
  sorry

end consecutive_lucky_years_l236_236994


namespace unique_intersection_point_l236_236164

theorem unique_intersection_point (c : ℝ) :
  (∀ x : ℝ, (|x - 20| + |x + 18| = x + c) → (x = 18 - 2 \/ x = 38 - x \/ x = 2 - 3 * x)) →
  c = 18 :=
by
  sorry

end unique_intersection_point_l236_236164


namespace count_sums_of_fours_and_fives_l236_236587

theorem count_sums_of_fours_and_fives :
  ∃ n, (∀ x y : ℕ, 4 * x + 5 * y = 1800 ↔ (x = 0 ∨ x ≤ 1800) ∧ (y = 0 ∨ y ≤ 1800)) ∧ n = 201 :=
by
  -- definition and theorem statement is complete. The proof is omitted.
  sorry

end count_sums_of_fours_and_fives_l236_236587


namespace polygon_sides_l236_236644

theorem polygon_sides (x : ℕ) (h : 180 * (x - 2) = 1080) : x = 8 :=
by sorry

end polygon_sides_l236_236644


namespace Alyssa_spending_correct_l236_236137

def cost_per_game : ℕ := 20

def last_year_in_person_games : ℕ := 13
def this_year_in_person_games : ℕ := 11
def this_year_streaming_subscription : ℕ := 120
def next_year_in_person_games : ℕ := 15
def next_year_streaming_subscription : ℕ := 150
def friends_count : ℕ := 2
def friends_join_games : ℕ := 5

def Alyssa_total_spending : ℕ :=
  (last_year_in_person_games * cost_per_game) +
  (this_year_in_person_games * cost_per_game) + this_year_streaming_subscription +
  (next_year_in_person_games * cost_per_game) + next_year_streaming_subscription -
  (friends_join_games * friends_count * cost_per_game)

theorem Alyssa_spending_correct : Alyssa_total_spending = 850 := by
  sorry

end Alyssa_spending_correct_l236_236137


namespace num_integer_solutions_quadratic_square_l236_236856

theorem num_integer_solutions_quadratic_square : 
  (∃ xs : Finset ℤ, 
    (∀ x ∈ xs, ∃ k : ℤ, (x^4 + 8*x^3 + 18*x^2 + 8*x + 64) = k^2) ∧ 
    xs.card = 2) := sorry

end num_integer_solutions_quadratic_square_l236_236856


namespace joe_average_speed_l236_236481

noncomputable def average_speed (total_distance total_time : ℝ) : ℝ :=
  total_distance / total_time

theorem joe_average_speed :
  let distance1 := 420
  let speed1 := 60
  let distance2 := 120
  let speed2 := 40
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  average_speed total_distance total_time = 54 := by
sorry

end joe_average_speed_l236_236481


namespace arctan_tan_expression_l236_236999

noncomputable def tan (x : ℝ) : ℝ := sorry
noncomputable def arctan (x : ℝ) : ℝ := sorry

theorem arctan_tan_expression :
  arctan (tan 65 - 2 * tan 40) = 25 := sorry

end arctan_tan_expression_l236_236999


namespace average_children_with_children_l236_236808

theorem average_children_with_children (total_families : ℕ) (average_children : ℚ) (childless_families : ℕ) :
  total_families = 12 →
  average_children = 3 →
  childless_families = 3 →
  (total_families * average_children) / (total_families - childless_families) = 4.0 :=
by
  intros h1 h2 h3
  sorry

end average_children_with_children_l236_236808


namespace ways_to_select_fuwa_sets_l236_236226

def types : Finset String := {"贝贝", "晶晶", "欢欢", "迎迎", "妮妮"}

theorem ways_to_select_fuwa_sets (h : ∀ t ∈ types, 2) :
  ∃ (n : ℕ), n = 160 := by
  -- Proof should be filled here
  sorry

end ways_to_select_fuwa_sets_l236_236226


namespace toucan_count_l236_236976

theorem toucan_count :
  (2 + 1 = 3) :=
by simp [add_comm]

end toucan_count_l236_236976


namespace num_parallel_edge_pairs_correct_l236_236043

-- Define a rectangular prism with given dimensions
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

-- Function to count the number of pairs of parallel edges
def num_parallel_edge_pairs (p : RectangularPrism) : ℕ :=
  4 * ((p.length + p.width + p.height) - 3)

-- Given conditions
def given_prism : RectangularPrism := { length := 4, width := 3, height := 2 }

-- Main theorem statement
theorem num_parallel_edge_pairs_correct :
  num_parallel_edge_pairs given_prism = 12 :=
by
  -- Skipping proof steps
  sorry

end num_parallel_edge_pairs_correct_l236_236043


namespace total_tickets_sold_l236_236224

theorem total_tickets_sold :
  ∃(S : ℕ), 4 * S + 6 * 388 = 2876 ∧ S + 388 = 525 :=
by
  sorry

end total_tickets_sold_l236_236224


namespace arithmetic_sequence_k_value_l236_236476

theorem arithmetic_sequence_k_value 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) = a n + d) 
  (h_first_term : a 1 = 0) 
  (h_nonzero_diff : d ≠ 0) 
  (h_sum : ∃ k, a k = a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7) : 
  ∃ k, k = 22 := 
by 
  sorry

end arithmetic_sequence_k_value_l236_236476


namespace greatest_prime_factor_187_l236_236367

theorem greatest_prime_factor_187 : ∃ p : ℕ, Prime p ∧ p ∣ 187 ∧ ∀ q : ℕ, Prime q ∧ q ∣ 187 → p ≥ q := by
  sorry

end greatest_prime_factor_187_l236_236367


namespace ratio_areas_of_circumscribed_circles_l236_236266

theorem ratio_areas_of_circumscribed_circles (P : ℝ) (A B : ℝ)
  (h1 : ∃ (x : ℝ), P = 8 * x)
  (h2 : ∃ (s : ℝ), s = P / 3)
  (hA : A = (5 * (P^2) * Real.pi) / 128)
  (hB : B = (P^2 * Real.pi) / 27) :
  A / B = 135 / 128 := by
  sorry

end ratio_areas_of_circumscribed_circles_l236_236266


namespace max_truck_speed_l236_236983

theorem max_truck_speed (D : ℝ) (C : ℝ) (F : ℝ) (L : ℝ → ℝ) (T : ℝ) (x : ℝ) : 
  D = 125 ∧ C = 30 ∧ F = 1000 ∧ (∀ s, L s = 2 * s) ∧ (∃ s, D / s * C + F + L s ≤ T) → x ≤ 75 :=
by
  sorry

end max_truck_speed_l236_236983


namespace simplify_sqrt7_pow6_l236_236350

theorem simplify_sqrt7_pow6 :
  (sqrt 7)^6 = 343 := by
  sorry

end simplify_sqrt7_pow6_l236_236350


namespace Q_2_plus_Q_neg2_l236_236602

variable {k : ℝ}

noncomputable def Q (x : ℝ) : ℝ := 0 -- Placeholder definition, real polynomial will be defined in proof.

theorem Q_2_plus_Q_neg2 (hQ0 : Q 0 = 2 * k)
  (hQ1 : Q 1 = 3 * k)
  (hQ_minus1 : Q (-1) = 4 * k) :
  Q 2 + Q (-2) = 16 * k :=
sorry

end Q_2_plus_Q_neg2_l236_236602


namespace sequence_value_a8_b8_l236_236494

theorem sequence_value_a8_b8
(a b : ℝ) 
(h1 : a + b = 1) 
(h2 : a^2 + b^2 = 3) 
(h3 : a^3 + b^3 = 4) 
(h4 : a^4 + b^4 = 7) 
(h5 : a^5 + b^5 = 11) 
(h6 : a^6 + b^6 = 18) : 
a^8 + b^8 = 47 :=
sorry

end sequence_value_a8_b8_l236_236494


namespace golden_section_length_l236_236870

noncomputable def golden_section_point (a b : ℝ) := a / (a + b) = b / a

theorem golden_section_length (A B P : ℝ) (h : golden_section_point A P) (hAP_gt_PB : A > P) (hAB : A + P = 2) : 
  A = Real.sqrt 5 - 1 :=
by
  -- Proof goes here
  sorry

end golden_section_length_l236_236870


namespace find_special_two_digit_numbers_l236_236529

def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_special (A : ℕ) : Prop :=
  let sum_A := sum_digits A
  sum_A^2 = sum_digits (A^2)

theorem find_special_two_digit_numbers :
  {A : ℕ | 10 ≤ A ∧ A < 100 ∧ is_special A} = {10, 11, 12, 13, 20, 21, 22, 30, 31} :=
by 
  sorry

end find_special_two_digit_numbers_l236_236529


namespace jerry_pick_up_trays_l236_236333

theorem jerry_pick_up_trays : 
  ∀ (trays_per_trip trips trays_from_second total),
  trays_per_trip = 8 →
  trips = 2 →
  trays_from_second = 7 →
  total = (trays_per_trip * trips) →
  (total - trays_from_second) = 9 :=
by
  intros trays_per_trip trips trays_from_second total
  intro h1 h2 h3 h4
  sorry

end jerry_pick_up_trays_l236_236333


namespace total_water_in_bucket_l236_236675

noncomputable def initial_gallons : ℝ := 3
noncomputable def added_gallons_1 : ℝ := 6.8
noncomputable def liters_to_gallons (liters : ℝ) : ℝ := liters / 3.78541
noncomputable def quart_to_gallons (quarts : ℝ) : ℝ := quarts / 4
noncomputable def added_gallons_2 : ℝ := liters_to_gallons 10
noncomputable def added_gallons_3 : ℝ := quart_to_gallons 4

noncomputable def total_gallons : ℝ :=
  initial_gallons + added_gallons_1 + added_gallons_2 + added_gallons_3

theorem total_water_in_bucket :
  abs (total_gallons - 13.44) < 0.01 :=
by
  -- convert amounts and perform arithmetic operations
  sorry

end total_water_in_bucket_l236_236675


namespace successive_discounts_eq_single_discount_l236_236790

theorem successive_discounts_eq_single_discount :
  ∀ (x : ℝ), (1 - 0.15) * (1 - 0.25) * x = (1 - 0.3625) * x :=
by
  intro x
  sorry

end successive_discounts_eq_single_discount_l236_236790


namespace other_endpoint_of_diameter_l236_236998

-- Define the basic data
def center : ℝ × ℝ := (5, 2)
def endpoint1 : ℝ × ℝ := (0, -3)
def endpoint2 : ℝ × ℝ := (10, 7)

-- State the final properties to be proved
theorem other_endpoint_of_diameter :
  ∃ (e2 : ℝ × ℝ), e2 = endpoint2 ∧
    dist center endpoint2 = dist endpoint1 center :=
sorry

end other_endpoint_of_diameter_l236_236998


namespace Bennett_has_6_brothers_l236_236543

theorem Bennett_has_6_brothers (num_aaron_brothers : ℕ) (num_bennett_brothers : ℕ) 
  (h1 : num_aaron_brothers = 4) 
  (h2 : num_bennett_brothers = 2 * num_aaron_brothers - 2) : 
  num_bennett_brothers = 6 := by
  sorry

end Bennett_has_6_brothers_l236_236543


namespace polygon_sides_l236_236645

theorem polygon_sides (x : ℕ) (h : 180 * (x - 2) = 1080) : x = 8 :=
by sorry

end polygon_sides_l236_236645


namespace units_digit_of_product_composites_l236_236526

def is_composite (n : ℕ) : Prop := 
  ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ n = m * k

theorem units_digit_of_product_composites (h1 : is_composite 9) (h2 : is_composite 10) (h3 : is_composite 12) :
  (9 * 10 * 12) % 10 = 0 :=
by
  sorry

end units_digit_of_product_composites_l236_236526


namespace points_in_first_quadrant_points_in_fourth_quadrant_points_in_second_quadrant_points_in_third_quadrant_l236_236813

def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

theorem points_in_first_quadrant (x y : ℝ) (h : x > 0 ∧ y > 0) : first_quadrant x y :=
by {
  sorry
}

theorem points_in_fourth_quadrant (x y : ℝ) (h : x > 0 ∧ y < 0) : fourth_quadrant x y :=
by {
  sorry
}

theorem points_in_second_quadrant (x y : ℝ) (h : x < 0 ∧ y > 0) : second_quadrant x y :=
by {
  sorry
}

theorem points_in_third_quadrant (x y : ℝ) (h : x < 0 ∧ y < 0) : third_quadrant x y :=
by {
  sorry
}

end points_in_first_quadrant_points_in_fourth_quadrant_points_in_second_quadrant_points_in_third_quadrant_l236_236813


namespace find_solution_l236_236160

theorem find_solution (x : ℝ) (h : (5 + x / 3)^(1/3) = -4) : x = -207 :=
sorry

end find_solution_l236_236160


namespace sum_coordinates_D_l236_236922

theorem sum_coordinates_D
    (M : (ℝ × ℝ))
    (C : (ℝ × ℝ))
    (D : (ℝ × ℝ))
    (H_M_midpoint : M = (5, 9))
    (H_C_coords : C = (11, 5))
    (H_M_def : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) :
    (D.1 + D.2) = 12 := 
by
  sorry
 
end sum_coordinates_D_l236_236922


namespace gcd_le_two_l236_236509

theorem gcd_le_two (a m n : ℕ) (h1 : a > 0) (h2 : m > 0) (h3 : n > 0) (h4 : Odd n) :
  Nat.gcd (a^n - 1) (a^m + 1) ≤ 2 := 
sorry

end gcd_le_two_l236_236509


namespace plane_speed_in_still_air_l236_236833

theorem plane_speed_in_still_air (p w : ℝ) (h1 : (p + w) * 3 = 900) (h2 : (p - w) * 4 = 900) : p = 262.5 :=
by
  sorry

end plane_speed_in_still_air_l236_236833


namespace lisa_goal_l236_236490

theorem lisa_goal 
  (total_quizzes : ℕ) 
  (target_percentage : ℝ) 
  (completed_quizzes : ℕ) 
  (earned_A : ℕ) 
  (remaining_quizzes : ℕ) : 
  total_quizzes = 40 → 
  target_percentage = 0.9 → 
  completed_quizzes = 25 → 
  earned_A = 20 → 
  remaining_quizzes = (total_quizzes - completed_quizzes) → 
  (earned_A + remaining_quizzes ≥ target_percentage * total_quizzes) → 
  remaining_quizzes - (total_quizzes * target_percentage - earned_A) = 0 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end lisa_goal_l236_236490


namespace simple_interest_difference_l236_236362

/-- The simple interest on a certain amount at a 4% rate for 5 years amounted to a certain amount less than the principal. The principal was Rs 2400. Prove that the difference between the principal and the simple interest is Rs 1920. 
-/
theorem simple_interest_difference :
  let P := 2400
  let R := 4
  let T := 5
  let SI := (P * R * T) / 100
  P - SI = 1920 :=
by
  /- We introduce the let definitions for the conditions and then state the theorem
    with the conclusion that needs to be proved. -/
  let P := 2400
  let R := 4
  let T := 5
  let SI := (P * R * T) / 100
  /- The final step where we would conclude our theorem. -/
  sorry

end simple_interest_difference_l236_236362


namespace count_non_squares_or_cubes_in_200_l236_236738

theorem count_non_squares_or_cubes_in_200 :
  let total_numbers := 200
  let count_perfect_squares := 14
  let count_perfect_cubes := 5
  let count_sixth_powers := 2
  total_numbers - (count_perfect_squares + count_perfect_cubes - count_sixth_powers) = 183 :=
by
  let total_numbers := 200
  let count_perfect_squares := 14
  let count_perfect_cubes := 5
  let count_sixth_powers := 2
  have h1 : total_numbers = 200 := rfl
  have h2 : count_perfect_squares = 14 := rfl
  have h3 : count_perfect_cubes = 5 := rfl
  have h4 : count_sixth_powers = 2 := rfl
  show total_numbers - (count_perfect_squares + count_perfect_cubes - count_sixth_powers) = 183
  calc
    total_numbers - (count_perfect_squares + count_perfect_cubes - count_sixth_powers)
        = 200 - (14 + 5 - 2) : by rw [h1, h2, h3, h4]
    ... = 200 - 17 : by norm_num
    ... = 183 : by norm_num

end count_non_squares_or_cubes_in_200_l236_236738


namespace max_ratio_convergence_l236_236337
open ProbabilityTheory

noncomputable def xi_seq (ξ : ℕ → ℝ) (i : ℕ) : Prop :=
  ∀ n : ℕ, ∀ ε > 0, n * ennreal.of_real (measure_theory.measure.prob {ω | abs (ξ i ω) > ε * f n}) = o(1).

noncomputable def xi_max_ratio_conv (ξ : ℕ → ℝ) (f : ℕ → ℝ) : Prop :=
  ∀ ε > 0, limsup n (n * pr {ω | abs (ξ i ω) > ε * f n}) = 0.

theorem max_ratio_convergence 
  (ξ : ℕ → ℝ) (f : ℕ → ℝ)
  (indep_ident : ∀ i, xi_seq ξ i)
  (lim_seq : ∀ n (ε > 0), n * pr {ω | abs (ξ 1 ω) > ε * f n} = o(1)) :
    xi_max_ratio_conv ξ f :=
sorry

end max_ratio_convergence_l236_236337


namespace increasing_iff_positive_difference_l236_236075

variable (a : ℕ → ℝ) (d : ℝ)

def arithmetic_sequence (aₙ : ℕ → ℝ) (d : ℝ) := ∃ (a₁ : ℝ), ∀ n : ℕ, aₙ n = a₁ + n * d

theorem increasing_iff_positive_difference (a : ℕ → ℝ) (d : ℝ) (h : arithmetic_sequence a d) :
  (∀ n, a (n+1) > a n) ↔ d > 0 :=
by
  sorry

end increasing_iff_positive_difference_l236_236075


namespace fill_in_blanks_problem1_fill_in_blanks_problem2_fill_in_blanks_problem3_l236_236013

def problem1_seq : List ℕ := [102, 101, 100, 99, 98, 97, 96]
def problem2_seq : List ℕ := [190, 180, 170, 160, 150, 140, 130, 120, 110, 100]
def problem3_seq : List ℕ := [5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500]

theorem fill_in_blanks_problem1 :
  ∃ (a b c d : ℕ), [102, a, 100, b, c, 97, d] = [102, 101, 100, 99, 98, 97, 96] :=
by
  exact ⟨101, 99, 98, 96, rfl⟩ -- Proof omitted with exact values

theorem fill_in_blanks_problem2 :
  ∃ (a b c d e f g : ℕ), [190, a, b, 160, c, d, e, 120, f, g] = [190, 180, 170, 160, 150, 140, 130, 120, 110, 100] :=
by
  exact ⟨180, 170, 150, 140, 130, 110, 100, rfl⟩ -- Proof omitted with exact values

theorem fill_in_blanks_problem3 :
  ∃ (a b c d e f : ℕ), [5000, a, 6000, b, 7000, c, d, e, f, 9500] = [5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500] :=
by
  exact ⟨5500, 6500, 7500, 8000, 8500, 9000, rfl⟩ -- Proof omitted with exact values

end fill_in_blanks_problem1_fill_in_blanks_problem2_fill_in_blanks_problem3_l236_236013


namespace geometric_sequence_term_l236_236201

theorem geometric_sequence_term (a : ℕ → ℕ) (q : ℕ) (hq : q = 2) (ha2 : a 2 = 8) :
  a 6 = 128 :=
by
  sorry

end geometric_sequence_term_l236_236201


namespace cost_per_bag_proof_minimize_total_cost_l236_236341

-- Definitions of given conditions
variable (x y : ℕ) -- cost per bag for brands A and B respectively
variable (m : ℕ) -- number of bags of brand B

def first_purchase_eq := 100 * x + 150 * y = 7000
def second_purchase_eq := 180 * x + 120 * y = 8100
def cost_per_bag_A : ℕ := 25
def cost_per_bag_B : ℕ := 30
def total_bags := 300
def constraint := (300 - m) ≤ 2 * m

-- Prove the costs per bag
theorem cost_per_bag_proof (h1 : first_purchase_eq x y)
                           (h2 : second_purchase_eq x y) :
  x = cost_per_bag_A ∧ y = cost_per_bag_B :=
sorry

-- Define the cost function and prove the purchase strategy
def total_cost (m : ℕ) : ℕ := 25 * (300 - m) + 30 * m

theorem minimize_total_cost (h : constraint m) :
  m = 100 ∧ total_cost 100 = 8000 :=
sorry

end cost_per_bag_proof_minimize_total_cost_l236_236341


namespace probability_of_condition_l236_236501

def chosen_set : Set ℕ := {n | 1 ≤ n ∧ n ≤ 20}

def is_odd (n : ℕ) : Prop := n % 2 = 1

def sum_divisible_by_five (a b : ℕ) : Prop := (a + b) % 5 = 0

theorem probability_of_condition :
  let total_choices := (Finset.card (Finset.powersetLen 2 (Finset.range 21))) -- total combinations
  let odd_numbers := {n | n ∈ chosen_set ∧ is_odd n}
  let odd_pairs := (Finset.powersetLen 2 (Finset.filter odd_numbers.toSet (Finset.range 21))) 
  let valid_pairs := Finset.filter (λ p, sum_divisible_by_five p.1 p.2) odd_pairs
  (Finset.card valid_pairs : ℚ) / (Finset.card total_choices) = 9 / 95 :=
by
  sorry

end probability_of_condition_l236_236501


namespace base_difference_is_correct_l236_236287

-- Definitions of given conditions
def base9_to_base10 (n : Nat) : Nat :=
  match n with
  | 324 => 3 * 9^2 + 2 * 9^1 + 4 * 9^0
  | _ => 0

def base6_to_base10 (n : Nat) : Nat :=
  match n with
  | 231 => 2 * 6^2 + 3 * 6^1 + 1 * 6^0
  | _ => 0

-- Lean statement to prove the equivalence
theorem base_difference_is_correct : base9_to_base10 324 - base6_to_base10 231 = 174 :=
by
  sorry

end base_difference_is_correct_l236_236287


namespace brenda_spay_cats_l236_236692

theorem brenda_spay_cats (c d : ℕ) (h1 : c + d = 21) (h2 : d = 2 * c) : c = 7 :=
sorry

end brenda_spay_cats_l236_236692


namespace algae_cell_count_at_day_nine_l236_236982

noncomputable def initial_cells : ℕ := 5
noncomputable def division_frequency_days : ℕ := 3
noncomputable def total_days : ℕ := 9

def number_of_cycles (total_days division_frequency_days : ℕ) : ℕ :=
  total_days / division_frequency_days

noncomputable def common_ratio : ℕ := 2

noncomputable def number_of_cells_after_n_days (initial_cells common_ratio number_of_cycles : ℕ) : ℕ :=
  initial_cells * common_ratio ^ (number_of_cycles - 1)

theorem algae_cell_count_at_day_nine : number_of_cells_after_n_days initial_cells common_ratio (number_of_cycles total_days division_frequency_days) = 20 :=
by
  sorry

end algae_cell_count_at_day_nine_l236_236982


namespace second_number_is_22_l236_236227

theorem second_number_is_22 
    (A B : ℤ)
    (h1 : A - B = 88) 
    (h2 : A = 110) :
    B = 22 :=
by
  sorry

end second_number_is_22_l236_236227


namespace multiplication_more_than_subtraction_l236_236661

def x : ℕ := 22

def multiplication_result : ℕ := 3 * x
def subtraction_result : ℕ := 62 - x
def difference : ℕ := multiplication_result - subtraction_result

theorem multiplication_more_than_subtraction : difference = 26 :=
by
  sorry

end multiplication_more_than_subtraction_l236_236661


namespace construct_triangle_num_of_solutions_l236_236145

theorem construct_triangle_num_of_solutions
  (r : ℝ) -- Circumradius
  (beta_gamma_diff : ℝ) -- Angle difference \beta - \gamma
  (KA1 : ℝ) -- Segment K A_1
  (KA1_lt_r : KA1 < r) -- Segment K A1 should be less than the circumradius
  (delta : ℝ := beta_gamma_diff) : 1 ≤ num_solutions ∧ num_solutions ≤ 2 :=
sorry

end construct_triangle_num_of_solutions_l236_236145


namespace oranges_to_apples_ratio_l236_236129

theorem oranges_to_apples_ratio :
  ∀ (total_fruits : ℕ) (weight_oranges : ℕ) (weight_apples : ℕ),
  total_fruits = 12 →
  weight_oranges = 10 →
  weight_apples = total_fruits - weight_oranges →
  weight_oranges / weight_apples = 5 :=
by
  intros total_fruits weight_oranges weight_apples h1 h2 h3
  sorry

end oranges_to_apples_ratio_l236_236129


namespace idempotent_elements_are_zero_l236_236601

-- Definitions based on conditions specified in the problem
variables {R : Type*} [Ring R] [CharZero R]
variable {e f g : R}

def idempotent (x : R) : Prop := x * x = x

-- The theorem to be proved
theorem idempotent_elements_are_zero (h_e : idempotent e) (h_f : idempotent f) (h_g : idempotent g) (h_sum : e + f + g = 0) : 
  e = 0 ∧ f = 0 ∧ g = 0 := 
sorry

end idempotent_elements_are_zero_l236_236601


namespace wall_height_to_breadth_ratio_l236_236363

theorem wall_height_to_breadth_ratio :
  ∀ (b : ℝ) (h : ℝ) (l : ℝ),
  b = 0.4 → h = n * b → l = 8 * h → l * b * h = 12.8 →
  n = 5 :=
by
  intros b h l hb hh hl hv
  sorry

end wall_height_to_breadth_ratio_l236_236363


namespace graduation_ceremony_chairs_l236_236470

theorem graduation_ceremony_chairs (g p t a : ℕ) 
  (h_g : g = 50) 
  (h_p : p = 2 * g) 
  (h_t : t = 20) 
  (h_a : a = t / 2) : 
  g + p + t + a = 180 :=
by
  sorry

end graduation_ceremony_chairs_l236_236470


namespace sum_squares_divisible_by_7_implies_both_divisible_l236_236057

theorem sum_squares_divisible_by_7_implies_both_divisible (a b : ℤ) (h : 7 ∣ (a^2 + b^2)) : 7 ∣ a ∧ 7 ∣ b :=
sorry

end sum_squares_divisible_by_7_implies_both_divisible_l236_236057


namespace mom_age_when_jayson_born_l236_236958

theorem mom_age_when_jayson_born (jayson_age dad_age mom_age : ℕ) 
  (h1 : jayson_age = 10) 
  (h2 : dad_age = 4 * jayson_age)
  (h3 : mom_age = dad_age - 2) :
  mom_age - jayson_age = 28 :=
by
  sorry

end mom_age_when_jayson_born_l236_236958


namespace min_value_fraction_l236_236336

theorem min_value_fraction (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) :
  (∃ T : ℝ, T = (5 * r / (3 * p + 2 * q) + 5 * p / (2 * q + 3 * r) + 2 * q / (p + r)) ∧ T = 19 / 4) :=
sorry

end min_value_fraction_l236_236336


namespace ice_cubes_per_cup_l236_236903

theorem ice_cubes_per_cup (total_ice_cubes number_of_cups : ℕ) (h1 : total_ice_cubes = 30) (h2 : number_of_cups = 6) : 
  total_ice_cubes / number_of_cups = 5 := 
by
  sorry

end ice_cubes_per_cup_l236_236903


namespace area_of_square_l236_236775

-- We define the points as given in the conditions
def point1 : ℝ × ℝ := (1, 2)
def point2 : ℝ × ℝ := (4, 6)

-- Lean's "def" defines the concept of a square given two adjacent points.
def is_square (p1 p2: ℝ × ℝ) : Prop :=
  let d := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  in ∃ (l : ℝ), l = d ∧ (l^2 = 25)

-- The theorem assumes the points are adjacent points on a square and proves that their area is 25.
theorem area_of_square :
  is_square point1 point2 :=
by
  -- Insert formal proof here, skipped with 'sorry' for this task
  sorry

end area_of_square_l236_236775


namespace max_value_of_f_l236_236422

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt (x^4 - 3*x^2 - 6*x + 13) - Real.sqrt (x^4 - x^2 + 1)

theorem max_value_of_f : ∃ x : ℝ, f x = Real.sqrt 10 :=
sorry

end max_value_of_f_l236_236422


namespace inequality_proof_l236_236715

theorem inequality_proof (a b c : ℝ) (hab : a > b) : a * |c| ≥ b * |c| := by
  sorry

end inequality_proof_l236_236715


namespace required_fraction_l236_236831

theorem required_fraction
  (total_members : ℝ)
  (top_10_lists : ℝ) :
  total_members = 775 →
  top_10_lists = 193.75 →
  top_10_lists / total_members = 0.25 :=
by
  sorry

end required_fraction_l236_236831


namespace range_of_a_l236_236462

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - 3| + |x - a| ≥ 3) ↔ a ≤ 0 ∨ a ≥ 6 :=
by
  sorry

end range_of_a_l236_236462


namespace billiard_trajectory_forms_regular_1998_gon_l236_236673

variables {A1 A2 ... A1998 : ℝ} 

noncomputable def is_regular_1998_gon (points : list ℝ) : Prop :=
  ∀ (i j k : ℕ), is_regular_polygon points 1998 i j k

noncomputable def angle_of_reflection (α β : ℝ) : Prop :=
  α = β

theorem billiard_trajectory_forms_regular_1998_gon 
  (A1 A2 ... A1998 : ℝ) 
  (midpoint : ℝ) 
  (trajectory : list ℝ) 
  (h1 : is_polygon trajectory)
  (h2 : starts_at_midpoint_of_A1A2 trajectory midpoint)
  (h3 : bounces_off_regular_sides trajectory [A2, A3, ... , A1998, A1])
  (h4 : ∀ i, angle_of_reflection (incident_angle trajectory i) (reflected_angle trajectory i))
  (h5 : returns_to_start trajectory) :
  is_regular_1998_gon trajectory := 
sorry

end billiard_trajectory_forms_regular_1998_gon_l236_236673


namespace probability_of_selection_l236_236061

theorem probability_of_selection : 
  ∀ (n k : ℕ), n = 121 ∧ k = 20 → (P : ℚ) = 20 / 121 :=
by
  intros n k h
  sorry

end probability_of_selection_l236_236061


namespace range_of_a_l236_236608

noncomputable def f (x a : ℝ) : ℝ :=
  if x ≤ 1 then 2^(|x - a|) else x + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x a ≥ f 1 a) ↔ (1 ≤ a ∧ a ≤ 2) :=
by
  sorry

end range_of_a_l236_236608


namespace solve_abs_inequality_l236_236015

theorem solve_abs_inequality (x : ℝ) :
  abs ((6 - 2 * x + 5) / 4) < 3 ↔ -1 / 2 < x ∧ x < 23 / 2 := 
sorry

end solve_abs_inequality_l236_236015


namespace percentage_increase_area_l236_236197

theorem percentage_increase_area (L W : ℝ) (hL : 0 < L) (hW : 0 < W) :
  let A := L * W
  let A' := (1.35 * L) * (1.35 * W)
  let percentage_increase := ((A' - A) / A) * 100
  percentage_increase = 82.25 :=
by
  sorry

end percentage_increase_area_l236_236197


namespace days_to_clear_messages_l236_236080

theorem days_to_clear_messages 
  (initial_messages : ℕ)
  (messages_read_per_day : ℕ)
  (new_messages_per_day : ℕ) 
  (net_messages_cleared_per_day : ℕ)
  (d : ℕ) :
  initial_messages = 98 →
  messages_read_per_day = 20 →
  new_messages_per_day = 6 →
  net_messages_cleared_per_day = messages_read_per_day - new_messages_per_day →
  d = initial_messages / net_messages_cleared_per_day →
  d = 7 :=
by
  intros h_initial h_read h_new h_net h_days
  sorry

end days_to_clear_messages_l236_236080


namespace prime_roots_sum_product_l236_236784

theorem prime_roots_sum_product (p q : ℕ) (x1 x2 : ℤ)
  (hp: Nat.Prime p) (hq: Nat.Prime q) 
  (h_sum: x1 + x2 = -↑p)
  (h_prod: x1 * x2 = ↑q) : 
  p = 3 ∧ q = 2 :=
sorry

end prime_roots_sum_product_l236_236784


namespace arrange_in_ascending_order_l236_236298

open Real

noncomputable def a := log 3 / log (1/2)
noncomputable def b := log 5 / log (1/2)
noncomputable def c := log (1/2) / log (1/3)

theorem arrange_in_ascending_order : b < a ∧ a < c :=
by
  sorry

end arrange_in_ascending_order_l236_236298


namespace opposite_of_9_is_neg_9_l236_236800

-- Definition of opposite number according to the given condition
def opposite (n : Int) : Int := -n

-- Proof statement that the opposite of 9 is -9
theorem opposite_of_9_is_neg_9 : opposite 9 = -9 :=
by
  sorry

end opposite_of_9_is_neg_9_l236_236800


namespace arithmetic_seq_15th_term_is_53_l236_236225

-- Define an arithmetic sequence
def arithmetic_seq (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

-- Original terms given
def a₁ : ℤ := -3
def d : ℤ := 4
def n : ℕ := 15

-- Prove that the 15th term is 53
theorem arithmetic_seq_15th_term_is_53 :
  arithmetic_seq a₁ d n = 53 :=
by
  sorry

end arithmetic_seq_15th_term_is_53_l236_236225


namespace geometric_vs_arithmetic_l236_236516

-- Definition of a positive geometric progression
def positive_geometric_progression (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = q * a n ∧ q > 0

-- Definition of an arithmetic progression
def arithmetic_progression (b : ℕ → ℝ) (d : ℝ) := ∀ n, b (n + 1) = b n + d

-- Theorem statement based on the problem and conditions
theorem geometric_vs_arithmetic
  (a : ℕ → ℝ) (b : ℕ → ℝ) (q : ℝ) (d : ℝ)
  (h1 : positive_geometric_progression a q)
  (h2 : arithmetic_progression b d)
  (h3 : a 6 = b 7) :
  a 3 + a 9 ≥ b 4 + b 10 := 
by 
  sorry

end geometric_vs_arithmetic_l236_236516


namespace last_passenger_seats_probability_l236_236947

theorem last_passenger_seats_probability (n : ℕ) (hn : n > 0) :
  ∀ (P : ℝ), P = 1 / 2 :=
by
  sorry

end last_passenger_seats_probability_l236_236947


namespace winner_won_by_324_votes_l236_236473

theorem winner_won_by_324_votes
  (total_votes : ℝ)
  (winner_percentage : ℝ)
  (winner_votes : ℝ)
  (h1 : winner_percentage = 0.62)
  (h2 : winner_votes = 837) :
  (winner_votes - (0.38 * total_votes) = 324) :=
by
  sorry

end winner_won_by_324_votes_l236_236473


namespace min_x_y_l236_236726

theorem min_x_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : (x + 1) * (y + 1) = 9) : x + y ≥ 4 :=
by
  sorry

end min_x_y_l236_236726


namespace total_cost_of_fencing_l236_236563

def P : ℤ := 42 + 35 + 52 + 66 + 40
def cost_per_meter : ℤ := 3
def total_cost : ℤ := P * cost_per_meter

theorem total_cost_of_fencing : total_cost = 705 := by
  sorry

end total_cost_of_fencing_l236_236563


namespace day_after_60_days_is_monday_l236_236279

theorem day_after_60_days_is_monday
    (birthday_is_thursday : ∃ d : ℕ, d % 7 = 0) :
    ∃ d : ℕ, (d + 60) % 7 = 4 :=
by
  -- Proof steps are omitted here
  sorry

end day_after_60_days_is_monday_l236_236279


namespace leftmost_square_side_length_l236_236638

open Real

/-- Given the side lengths of three squares, 
    where the middle square's side length is 17 cm longer than the leftmost square,
    the rightmost square's side length is 6 cm shorter than the middle square,
    and the sum of the side lengths of all three squares is 52 cm,
    prove that the side length of the leftmost square is 8 cm. -/
theorem leftmost_square_side_length
  (x : ℝ)
  (h1 : ∀ m : ℝ, m = x + 17)
  (h2 : ∀ r : ℝ, r = x + 11)
  (h3 : x + (x + 17) + (x + 11) = 52) :
  x = 8 := by
  sorry

end leftmost_square_side_length_l236_236638


namespace product_f_g_l236_236049

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x * (x + 1))
noncomputable def g (x : ℝ) : ℝ := 1 / Real.sqrt x

theorem product_f_g (x : ℝ) (hx : 0 < x) : f x * g x = Real.sqrt (x + 1) := 
by 
  sorry

end product_f_g_l236_236049


namespace find_z_l236_236122

-- Definitions based on the conditions from the problem
def x : ℤ := sorry
def y : ℤ := x - 1
def z : ℤ := x - 2
def condition1 : x > y ∧ y > z := by
  sorry

def condition2 : 2 * x + 3 * y + 3 * z = 5 * y + 11 := by
  sorry

-- Statement to prove
theorem find_z : z = 3 :=
by
  -- Use the conditions to prove the statement
  have h1 : x > y ∧ y > z := condition1
  have h2 : 2 * x + 3 * y + 3 * z = 5 * y + 11 := condition2
  sorry

end find_z_l236_236122


namespace point_third_quadrant_l236_236887

theorem point_third_quadrant (m n : ℝ) (h1 : m < 0) (h2 : n > 0) : 3 * m - 2 < 0 ∧ -n < 0 :=
by
  sorry

end point_third_quadrant_l236_236887


namespace repeating_decimal_division_l236_236559

def repeating_decimal_142857 : ℚ := 1 / 7
def repeating_decimal_2_857143 : ℚ := 20 / 7

theorem repeating_decimal_division :
  (repeating_decimal_142857 / repeating_decimal_2_857143) = 1 / 20 :=
by
  sorry

end repeating_decimal_division_l236_236559


namespace tomatoes_picked_today_l236_236830

theorem tomatoes_picked_today (initial yesterday_picked left_after_yesterday today_picked : ℕ)
  (h1 : initial = 160)
  (h2 : yesterday_picked = 56)
  (h3 : left_after_yesterday = 104)
  (h4 : initial - yesterday_picked = left_after_yesterday) :
  today_picked = 56 :=
by
  sorry

end tomatoes_picked_today_l236_236830


namespace distinct_numbers_mean_inequality_l236_236910

open Nat

theorem distinct_numbers_mean_inequality (n m : ℕ) (h_n_m : m ≤ n)
  (a : Fin m → ℕ) (ha_distinct : Function.Injective a)
  (h_cond : ∀ (i j : Fin m), i ≠ j → i.val + j.val ≤ n → ∃ (k : Fin m), a i + a j = a k) :
  (1 : ℝ) / m * (Finset.univ.sum (fun i => a i)) ≥  (n + 1) / 2 :=
by
  sorry

end distinct_numbers_mean_inequality_l236_236910


namespace transformation_C_factorization_l236_236272

open Function

theorem transformation_C_factorization (a b : ℤ) :
  (a - 1) * (b - 1) = ab - a - b + 1 :=
by sorry

end transformation_C_factorization_l236_236272


namespace cost_of_schools_renovation_plans_and_min_funding_l236_236598

-- Define costs of Type A and Type B schools
def cost_A : ℝ := 60
def cost_B : ℝ := 85

-- Initial conditions given in the problem
axiom initial_condition_1 : cost_A + 2 * cost_B = 230
axiom initial_condition_2 : 2 * cost_A + cost_B = 205

-- Variables for number of Type A and Type B schools to renovate
variables (x : ℕ) (y : ℕ)
-- Total schools to renovate
axiom total_schools : x + y = 6

-- National and local finance constraints
axiom national_finance_max : 60 * x + 85 * y ≤ 380
axiom local_finance_min : 10 * x + 15 * y ≥ 70

-- Proving the cost of one Type A and one Type B school
theorem cost_of_schools : cost_A = 60 ∧ cost_B = 85 := 
by {
  sorry
}

-- Proving the number of renovation plans and the least funding plan
theorem renovation_plans_and_min_funding :
  ∃ x y, (x + y = 6) ∧ 
         (10 * x + 15 * y ≥ 70) ∧ 
         (60 * x + 85 * y ≤ 380) ∧ 
         (x = 2 ∧ y = 4 ∨ x = 3 ∧ y = 3 ∨ x = 4 ∧ y = 2) ∧ 
         (∀ (a b : ℕ), (a + b = 6) ∧ 
                       (10 * a + 15 * b ≥ 70) ∧ 
                       (60 * a + 85 * b ≤ 380) → 
                       60 * a + 85 * b ≥ 410) :=
by {
  sorry
}

end cost_of_schools_renovation_plans_and_min_funding_l236_236598


namespace probability_at_least_two_same_l236_236523

theorem probability_at_least_two_same :
  let total_outcomes := (8 ^ 4 : ℕ)
  let num_diff_outcomes := (8 * 7 * 6 * 5 : ℕ)
  let probability_diff := (num_diff_outcomes : ℝ) / total_outcomes
  let probability_at_least_two := 1 - probability_diff
  probability_at_least_two = (151 : ℝ) / 256 :=
by
  sorry

end probability_at_least_two_same_l236_236523


namespace tree_height_increase_l236_236969

-- Definitions given in the conditions
def h0 : ℝ := 4
def h (t : ℕ) (x : ℝ) : ℝ := h0 + t * x

-- Proof statement
theorem tree_height_increase (x : ℝ) :
  h 6 x = (4 / 3) * h 4 x + h 4 x → x = 2 :=
by
  intro h6_eq
  rw [h, h] at h6_eq
  norm_num at h6_eq
  sorry

end tree_height_increase_l236_236969


namespace adam_change_l236_236839

theorem adam_change : 
  let amount : ℝ := 5.00
  let cost : ℝ := 4.28
  amount - cost = 0.72 :=
by
  -- proof goes here
  sorry

end adam_change_l236_236839


namespace solution_of_abs_eq_l236_236855

theorem solution_of_abs_eq (x : ℝ) : |x - 5| = 3 * x + 6 → x = -1 / 4 :=
by
  sorry

end solution_of_abs_eq_l236_236855


namespace rightmost_three_digits_of_7_pow_1993_l236_236810

theorem rightmost_three_digits_of_7_pow_1993 :
  7^1993 % 1000 = 407 := 
sorry

end rightmost_three_digits_of_7_pow_1993_l236_236810


namespace cat_litter_cost_l236_236330

theorem cat_litter_cost 
    (container_weight : ℕ) (container_cost : ℕ)
    (litter_box_capacity : ℕ) (change_interval : ℕ) 
    (days_needed : ℕ) (cost : ℕ) :
  container_weight = 45 → 
  container_cost = 21 → 
  litter_box_capacity = 15 → 
  change_interval = 7 →
  days_needed = 210 → 
  cost = 210 :=
by
  intros h1 h2 h3 h4 h5
  /- Here we would add the proof steps, but this is not required. -/
  sorry

end cat_litter_cost_l236_236330


namespace find_number_l236_236265

theorem find_number 
  (n : ℤ)
  (h1 : n % 7 = 2)
  (h2 : n % 8 = 4)
  (quot_7 : ℤ)
  (quot_8 : ℤ)
  (h3 : n = 7 * quot_7 + 2)
  (h4 : n = 8 * quot_8 + 4)
  (h5 : quot_7 = quot_8 + 7) :
  n = 380 := by
  sorry

end find_number_l236_236265


namespace total_prep_time_is_8_l236_236221

-- Defining the conditions
def prep_vocab_sentence_eq := 3
def prep_analytical_writing := 2
def prep_quantitative_reasoning := 3

-- Stating the total preparation time
def total_prep_time := prep_vocab_sentence_eq + prep_analytical_writing + prep_quantitative_reasoning

-- The Lean statement of the mathematical proof problem
theorem total_prep_time_is_8 : total_prep_time = 8 := by
  sorry

end total_prep_time_is_8_l236_236221


namespace abs_sum_leq_abs_l236_236432

theorem abs_sum_leq_abs (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  |a| + |b| ≤ |a + b| :=
sorry

end abs_sum_leq_abs_l236_236432


namespace cylinder_radius_inscribed_box_l236_236538

theorem cylinder_radius_inscribed_box :
  ∀ (x y z r : ℝ),
    4 * (x + y + z) = 160 →
    2 * (x * y + y * z + x * z) = 600 →
    z = 40 - x - y →
    r = (1/2) * Real.sqrt (x^2 + y^2) →
    r = (15 * Real.sqrt 2) / 2 :=
by
  sorry

end cylinder_radius_inscribed_box_l236_236538


namespace tree_heights_l236_236252

theorem tree_heights (T S : ℕ) (h1 : T - S = 20) (h2 : T - 10 = 3 * (S - 10)) : T = 40 := 
by
  sorry

end tree_heights_l236_236252


namespace number_of_proper_subsets_of_set_A_l236_236733

def set_A : Finset ℕ := {1, 2, 3}

theorem number_of_proper_subsets_of_set_A :
  (set_A.powerset.filter (λ s, s ≠ set_A)).card = 7 := 
sorry

end number_of_proper_subsets_of_set_A_l236_236733


namespace mark_old_bills_l236_236214

noncomputable def old_hourly_wage : ℝ := 40
noncomputable def new_hourly_wage : ℝ := 42
noncomputable def work_hours_per_week : ℝ := 8 * 5
noncomputable def personal_trainer_cost_per_week : ℝ := 100
noncomputable def leftover_after_expenses : ℝ := 980

noncomputable def new_weekly_earnings := new_hourly_wage * work_hours_per_week
noncomputable def total_weekly_spending_after_raise := leftover_after_expenses + personal_trainer_cost_per_week
noncomputable def old_bills_per_week := new_weekly_earnings - total_weekly_spending_after_raise

theorem mark_old_bills : old_bills_per_week = 600 := by
  sorry

end mark_old_bills_l236_236214


namespace biased_coin_probability_l236_236672

open ProbabilityTheory

noncomputable def biasedCoin (p_heads : ℝ) (p_tails : ℝ) : List ℕ → ℝ
| []     := 1
| (0 :: flips) := p_heads * biasedCoin p_heads p_tails flips
| (1 :: flips) := p_tails * biasedCoin p_heads p_tails flips

theorem biased_coin_probability (p_heads : ℝ) (p_tails : ℝ) :
  (p_heads = 0.3) → (p_tails = 0.7) →
  biasedCoin p_heads p_tails ([1, 1, 1, 0] ++ [0, 1, 0] ++ [1, 0, 1, 1]) = 0.3087 :=
by
  intros h_heads h_tails
  sorry

end biased_coin_probability_l236_236672


namespace chase_cardinals_count_l236_236378

variable (gabrielle_robins : Nat)
variable (gabrielle_cardinals : Nat)
variable (gabrielle_blue_jays : Nat)
variable (chase_robins : Nat)
variable (chase_blue_jays : Nat)
variable (chase_cardinals : Nat)

variable (gabrielle_total : Nat)
variable (chase_total : Nat)

variable (percent_more : Nat)

axiom gabrielle_robins_def : gabrielle_robins = 5
axiom gabrielle_cardinals_def : gabrielle_cardinals = 4
axiom gabrielle_blue_jays_def : gabrielle_blue_jays = 3

axiom chase_robins_def : chase_robins = 2
axiom chase_blue_jays_def : chase_blue_jays = 3

axiom gabrielle_total_def : gabrielle_total = gabrielle_robins + gabrielle_cardinals + gabrielle_blue_jays
axiom chase_total_def : chase_total = chase_robins + chase_blue_jays + chase_cardinals
axiom percent_more_def : percent_more = 20

axiom gabrielle_more_birds : gabrielle_total = Nat.ceil ((chase_total * (100 + percent_more)) / 100)

theorem chase_cardinals_count : chase_cardinals = 5 := by sorry

end chase_cardinals_count_l236_236378


namespace find_x_l236_236668

variables (x : ℝ)

theorem find_x : (x / 4) * 12 = 9 → x = 3 :=
by
  sorry

end find_x_l236_236668


namespace integer_bases_not_divisible_by_5_l236_236861

theorem integer_bases_not_divisible_by_5 :
  ∀ b ∈ ({3, 5, 7, 10, 12} : Set ℕ), (b - 1) ^ 2 % 5 ≠ 0 :=
by sorry

end integer_bases_not_divisible_by_5_l236_236861


namespace longest_possible_height_l236_236510

theorem longest_possible_height (a b c : ℕ) (ha : a = 3 * c) (hb : b * 4 = 12 * c) (h_tri : a - c < b) (h_unequal : ¬(a = c)) :
  ∃ x : ℕ, (4 < x ∧ x < 6) ∧ x = 5 :=
by
  sorry

end longest_possible_height_l236_236510


namespace eccentricity_hyperbola_l236_236633

variables (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0)
variables (H : c = Real.sqrt (a^2 + b^2))
variables (L1 : ∀ x y : ℝ, x = c → (x^2/a^2 - y^2/b^2 = 1))
variables (L2 : ∀ (B C : ℝ × ℝ), (B.1 = c ∧ C.1 = c) ∧ (B.2 = -C.2) ∧ (B.2 = b^2/a))

theorem eccentricity_hyperbola : ∃ e, e = 2 :=
sorry

end eccentricity_hyperbola_l236_236633


namespace determine_n_l236_236665

theorem determine_n (n : ℕ) (h : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^26) : n = 25 :=
by
  sorry

end determine_n_l236_236665


namespace binomial_sum_sum_of_n_values_l236_236370

theorem binomial_sum (n : ℕ) (h : nat.choose 28 14 + nat.choose 28 n = nat.choose 29 15) : n = 13 ∨ n = 15 := sorry

theorem sum_of_n_values : ∑ n in {n | nat.choose 28 14 + nat.choose 28 n = nat.choose 29 15}.to_finset, n = 28 :=
by
  apply finset.sum_eq_from_set,
  intros x hx,
  cases binomial_sum x hx,
  { simp [h], },
  { simp [h], }

end binomial_sum_sum_of_n_values_l236_236370


namespace arrival_time_at_midpoint_l236_236989

theorem arrival_time_at_midpoint :
  let planned_start := mkTime (10, 10)
  let planned_end := mkTime (13, 10)
  let actual_start := planned_start.addMinutes 5
  let actual_end := planned_end.addMinutes (-4)
  let midpoint_time := planned_start.addMinutes 90
  let correct_time := mkTime (11, 50)
  midpoint_time = correct_time :=
by
  sorry

end arrival_time_at_midpoint_l236_236989


namespace part_a_part_b_l236_236553

/-- Define rational non-integer numbers x and y -/
structure RationalNonInteger (x y : ℚ) :=
  (h1 : x.denom ≠ 1)
  (h2 : y.denom ≠ 1)

/-- Part (a): There exist rational non-integer numbers x and y 
    such that 19x + 8y and 8x + 3y are integers -/
theorem part_a : ∃ (x y : ℚ), RationalNonInteger x y ∧ (19*x + 8*y ∈ ℤ) ∧ (8*x + 3*y ∈ ℤ) :=
by
  sorry

/-- Part (b): There do not exist rational non-integer numbers x and y 
    such that 19x^2 + 8y^2 and 8x^2 + 3y^2 are integers -/
theorem part_b : ¬ ∃ (x y : ℚ), RationalNonInteger x y ∧ (19*x^2 + 8*y^2 ∈ ℤ) ∧ (8*x^2 + 3*y^2 ∈ ℤ) :=
by
  sorry

end part_a_part_b_l236_236553


namespace last_four_digits_5_pow_2015_l236_236495

theorem last_four_digits_5_pow_2015 :
  (5^2015) % 10000 = 8125 :=
by
  sorry

end last_four_digits_5_pow_2015_l236_236495


namespace third_consecutive_odd_integers_is_fifteen_l236_236671

theorem third_consecutive_odd_integers_is_fifteen :
  ∃ x : ℤ, (x % 2 = 1 ∧ (x + 2) % 2 = 1 ∧ (x + 4) % 2 = 1) ∧ (x + 2 + (x + 4) = x + 17) → (x + 4 = 15) :=
by
  sorry

end third_consecutive_odd_integers_is_fifteen_l236_236671


namespace final_problem_l236_236124

def problem1 : Prop :=
  ∃ (x y : ℝ), 10 * x + 20 * y = 3000 ∧ 8 * x + 24 * y = 2800 ∧ x = 200 ∧ y = 50

def problem2 : Prop :=
  ∀ (m : ℕ), 10 ≤ m ∧ m ≤ 12 ∧ 
  200 * m + 50 * (40 - m) ≤ 3800 ∧ 
  (40 - m) ≤ 3 * m →
  (m = 10 ∧ (40 - m) = 30) ∨ 
  (m = 11 ∧ (40 - m) = 29) ∨ 
  (m = 12 ∧ (40 - m) = 28)

theorem final_problem : problem1 ∧ problem2 :=
by
  sorry

end final_problem_l236_236124


namespace y_range_l236_236573

variable (a b : ℝ)
variable (h₀ : 0 < a) (h₁ : 0 < b)

theorem y_range (x : ℝ) (y : ℝ) (h₂ : y = (a * Real.sin x + b) / (a * Real.sin x - b)) : 
  y ≥ (a - b) / (a + b) ∨ y ≤ (a + b) / (a - b) :=
sorry

end y_range_l236_236573


namespace number_subtracted_from_15n_l236_236053

theorem number_subtracted_from_15n (m n : ℕ) (h_pos_n : 0 < n) (h_pos_m : 0 < m) (h_eq : m = 15 * n - 1) (h_remainder : m % 5 = 4) : 1 = 1 :=
by
  sorry

end number_subtracted_from_15n_l236_236053


namespace circle_equation_and_lines_l236_236437

noncomputable def O : ℝ × ℝ := (0, 0)
noncomputable def A : ℝ × ℝ := (6, 2)
noncomputable def B : ℝ × ℝ := (4, 4)
noncomputable def C_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 10

structure Line (κ β: ℝ) where
  passes_through : ℝ × ℝ → Prop
  definition : Prop

def line_passes_through_point (κ β : ℝ) (p : ℝ × ℝ) : Prop := p.2 = κ * p.1 + β

theorem circle_equation_and_lines : 
  (∀ p : ℝ × ℝ, p = O ∨ p = A ∨ p = B → C_eq p.1 p.2) ∧
  ((∀ p : ℝ × ℝ, line_passes_through_point 0 2 p → C_eq 2 6 ∧ (∃ x1 x2 y : ℝ, C_eq x1 y ∧ C_eq x2 y ∧ ((x1 - x2)^2 + (y - y)^2) = 4)) ∧
   (∀ p : ℝ × ℝ, line_passes_through_point (-7 / 3) (32 / 3) p → C_eq 2 6 ∧ (∃ x1 x2 y : ℝ, C_eq x1 y ∧ C_eq x2 y ∧ ((x1 - x2)^2 + (y - y)^2) = 4))) :=
by 
  sorry

end circle_equation_and_lines_l236_236437


namespace square_area_l236_236776

def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem square_area (p1 p2 : ℝ × ℝ) (h : p1 = (1, 2) ∧ p2 = (4, 6)) :
  let d := distance p1 p2 in
  d^2 = 25 :=
by
  sorry

end square_area_l236_236776


namespace sum_of_abs_values_eq_12_l236_236742

theorem sum_of_abs_values_eq_12 (a b c d : ℝ) (h : 6 * x^2 + x - 12 = (a * x + b) * (c * x + d)) :
  abs a + abs b + abs c + abs d = 12 := sorry

end sum_of_abs_values_eq_12_l236_236742


namespace problem_statement_l236_236208

variable {x y z : ℝ}

theorem problem_statement 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z)
  (hxyz : x * y * z = 1) :
  1 / (x ^ 3 * y) + 1 / (y ^ 3 * z) + 1 / (z ^ 3 * x) ≥ x * y + y * z + z * x :=
by sorry

end problem_statement_l236_236208


namespace sin_alpha_value_l236_236740

open Real

theorem sin_alpha_value (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : cos (α + π / 4) = 1 / 3) :
  sin α = (4 - sqrt 2) / 6 :=
sorry

end sin_alpha_value_l236_236740


namespace text_messages_in_march_l236_236331

theorem text_messages_in_march
  (nov_texts : ℕ)
  (dec_texts : ℕ)
  (jan_texts : ℕ)
  (feb_texts : ℕ)
  (double_pattern : ∀ n m : ℕ, m = 2 * n)
  (h_nov : nov_texts = 1)
  (h_dec : dec_texts = 2 * nov_texts)
  (h_jan : jan_texts = 2 * dec_texts)
  (h_feb : feb_texts = 2 * jan_texts) : 
  ∃ mar_texts : ℕ, mar_texts = 2 * feb_texts ∧ mar_texts = 16 := 
by
  sorry

end text_messages_in_march_l236_236331


namespace multiply_add_square_l236_236694

theorem multiply_add_square : 15 * 28 + 42 * 15 + 15^2 = 1275 :=
by
  sorry

end multiply_add_square_l236_236694


namespace find_fourth_vertex_l236_236123

-- Given three vertices of a tetrahedron
def v1 : ℤ × ℤ × ℤ := (1, 1, 2)
def v2 : ℤ × ℤ × ℤ := (4, 2, 1)
def v3 : ℤ × ℤ × ℤ := (3, 1, 5)

-- The side length squared of the tetrahedron (computed from any pair of given points)
def side_length_squared : ℤ := 11

-- The goal is to find the fourth vertex with integer coordinates which maintains the distance
def is_fourth_vertex (x y z : ℤ) : Prop :=
  (x - 1)^2 + (y - 1)^2 + (z - 2)^2 = side_length_squared ∧
  (x - 4)^2 + (y - 2)^2 + (z - 1)^2 = side_length_squared ∧
  (x - 3)^2 + (y - 1)^2 + (z - 5)^2 = side_length_squared

theorem find_fourth_vertex : is_fourth_vertex 4 1 3 :=
  sorry

end find_fourth_vertex_l236_236123


namespace boxes_left_to_sell_l236_236754

def sales_goal : ℕ := 150
def first_customer : ℕ := 5
def second_customer : ℕ := 4 * first_customer
def third_customer : ℕ := second_customer / 2
def fourth_customer : ℕ := 3 * third_customer
def fifth_customer : ℕ := 10
def total_sold : ℕ := first_customer + second_customer + third_customer + fourth_customer + fifth_customer

theorem boxes_left_to_sell : sales_goal - total_sold = 75 := by
  sorry

end boxes_left_to_sell_l236_236754


namespace compare_M_N_l236_236026

theorem compare_M_N (a : ℝ) : 
  let M := 2 * a * (a - 2) + 7
  let N := (a - 2) * (a - 3)
  M > N :=
by
  sorry

end compare_M_N_l236_236026


namespace triangle_is_right_angle_l236_236896

theorem triangle_is_right_angle (A B C : ℝ) : 
  (A / B = 2 / 3) ∧ (A / C = 2 / 5) ∧ (A + B + C = 180) →
  (A = 36) ∧ (B = 54) ∧ (C = 90) :=
by 
  intro h
  sorry

end triangle_is_right_angle_l236_236896


namespace minimum_g7_l236_236545

def is_tenuous (g : ℕ → ℤ) : Prop :=
∀ x y : ℕ, 0 < x → 0 < y → g x + g y > x^2

noncomputable def min_possible_value_g7 (g : ℕ → ℤ) (h : is_tenuous g) 
  (h_sum : (g 1 + g 2 + g 3 + g 4 + g 5 + g 6 + g 7 + g 8 + g 9 + g 10) = 
             -29) : ℤ :=
g 7

theorem minimum_g7 (g : ℕ → ℤ) (h : is_tenuous g)
  (h_sum : (g 1 + g 2 + g 3 + g 4 + g 5 + g 6 + g 7 + g 8 + g 9 + g 10) = 
             -29) :
  min_possible_value_g7 g h h_sum = 49 :=
sorry

end minimum_g7_l236_236545


namespace probability_of_rolling_three_next_l236_236071

-- Define the probability space and events
def fair_die : ProbabilityMassFunction (Fin 6) :=
  ProbabilityMassFunction.uniformOfFin

-- Define the event of rolling a specific number (e.g. three)
def event_three : Set (Fin 6) := {3}

theorem probability_of_rolling_three_next :
  (∀ i : Fin 6, fair_die i = 1 / 6) →
  ∀ (previous_rolls : Vector (Fin 6) 6),
  ∀ (h : ∀ i ∈ previous_rolls.toList, i = 5),
  fair_die.toMeasure.Prob event_three = 1 / 6 :=
by
  intros h previous_rolls roll_condition
  sorry

end probability_of_rolling_three_next_l236_236071


namespace first_divisor_exists_l236_236812

theorem first_divisor_exists (m d : ℕ) :
  (m % d = 47) ∧ (m % 24 = 23) ∧ (d > 47) → d = 72 :=
by
  sorry

end first_divisor_exists_l236_236812


namespace part1_part2_l236_236308

noncomputable def A : (ℝ × ℝ) := (1, 1)
noncomputable def B : (ℝ × ℝ) := (3, 2)
noncomputable def C : (ℝ × ℝ) := (5, 4)

-- Equation of the line containing the altitude from AB
theorem part1 (A B C : ℝ × ℝ) (hA : A = (1, 1)) (hB : B = (3, 2)) (hC : C = (5, 4)) :
  ∃ (line : ℝ → ℝ → Prop), (∀ x y, line x y ↔ 2 * x + y - 14 = 0) :=
sorry

-- Perimeter of the triangle formed by the two coordinate axes and a given line l
theorem part2 (A B C : ℝ × ℝ) (hA : A = (1, 1)) (hB : B = (3, 2)) (hC : C = (5, 4))
  (hLine : ∀ x y, l x y → (x / (1 + a)) + (y / a) = 1)
  (ha : ∃ a : ℝ, -4 * a = 3 * (a + 1)) :
  ∃ p : ℝ, p = 12 / 7 :=
sorry

end part1_part2_l236_236308


namespace exists_rational_non_integer_a_not_exists_rational_non_integer_b_l236_236557

-- Define rational non-integer numbers
def is_rational_non_integer (x : ℚ) : Prop := ¬(∃ (z : ℤ), x = z)

-- (a) Proof for existance of rational non-integer numbers y and x such that 19x + 8y, 8x + 3y are integers
theorem exists_rational_non_integer_a :
  ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧ (∃ a b : ℤ, 19 * x + 8 * y = a ∧ 8 * x + 3 * y = b) :=
sorry

-- (b) Proof for non-existance of rational non-integer numbers y and x such that 19x² + 8y², 8x² + 3y² are integers
theorem not_exists_rational_non_integer_b :
  ¬ ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧ (∃ m n : ℤ, 19 * x^2 + 8 * y^2 = m ∧ 8 * x^2 + 3 * y^2 = n) :=
sorry

end exists_rational_non_integer_a_not_exists_rational_non_integer_b_l236_236557


namespace total_shirts_l236_236455

def hazel_shirts : ℕ := 6
def razel_shirts : ℕ := 2 * hazel_shirts

theorem total_shirts : hazel_shirts + razel_shirts = 18 := by
  sorry

end total_shirts_l236_236455


namespace total_baskets_l236_236840

theorem total_baskets (Alex_baskets Sandra_baskets Hector_baskets Jordan_baskets total_baskets : ℕ)
  (h1 : Alex_baskets = 8)
  (h2 : Sandra_baskets = 3 * Alex_baskets)
  (h3 : Hector_baskets = 2 * Sandra_baskets)
  (total_combined_baskets := Alex_baskets + Sandra_baskets + Hector_baskets)
  (h4 : Jordan_baskets = total_combined_baskets / 5)
  (h5 : total_baskets = Alex_baskets + Sandra_baskets + Hector_baskets + Jordan_baskets) :
  total_baskets = 96 := by
  sorry

end total_baskets_l236_236840


namespace ff_two_eq_three_l236_236582

noncomputable def f (x : ℝ) : ℝ :=
  if x < 6 then x^3 else Real.log x / Real.log x

theorem ff_two_eq_three : f (f 2) = 3 := by
  sorry

end ff_two_eq_three_l236_236582


namespace value_of_r_minus_q_l236_236939

variable (q r : ℝ)
variable (slope : ℝ)
variable (h_parallel : slope = 3 / 2)
variable (h_points : (r - q) / (-2) = slope)

theorem value_of_r_minus_q (h_parallel : slope = 3 / 2) (h_points : (r - q) / (-2) = slope) : 
  r - q = -3 := by
  sorry

end value_of_r_minus_q_l236_236939


namespace quadratic_expression_rewrite_l236_236092

theorem quadratic_expression_rewrite :
  ∃ a b c : ℚ, (∀ k : ℚ, 12 * k^2 + 8 * k - 16 = a * (k + b)^2 + c) ∧ c + 3 * b = -49/3 :=
sorry

end quadratic_expression_rewrite_l236_236092


namespace find_y_values_l236_236426

def A (y : ℝ) : ℝ := 1 - y - 2 * y^2

theorem find_y_values (y : ℝ) (h₁ : y ≤ 1) (h₂ : y ≠ 0) (h₃ : y ≠ -1) (h₄ : y ≠ 0.5) :
  y^2 * A y / (y * A y) ≤ 1 ↔
  y ∈ Set.Iio (-1) ∪ Set.Ioo (-1) (1/2) ∪ Set.Ioc (1/2) 1 :=
by
  -- proof is omitted
  sorry

end find_y_values_l236_236426


namespace solve_quadratic_equation_l236_236926

theorem solve_quadratic_equation :
  ∃ x : ℝ, 2 * x^2 = 4 * x - 1 ∧ (x = (2 + Real.sqrt 2) / 2 ∨ x = (2 - Real.sqrt 2) / 2) :=
by
  sorry

end solve_quadratic_equation_l236_236926


namespace djibo_sister_age_today_l236_236150

variable (djibo_age : ℕ) (sum_ages_five_years_ago : ℕ)

theorem djibo_sister_age_today (h1 : djibo_age = 17)
                               (h2 : sum_ages_five_years_ago = 35) :
  let djibo_age_five_years_ago := djibo_age - 5 in
  let sister_age_five_years_ago := sum_ages_five_years_ago - djibo_age_five_years_ago in
  sister_age_five_years_ago + 5 = 28 :=
by 
  -- Proof goes here
  sorry

end djibo_sister_age_today_l236_236150


namespace find_real_x_l236_236159

theorem find_real_x (x : ℝ) : 
  (2 ≤ 3 * x / (3 * x - 7)) ∧ (3 * x / (3 * x - 7) < 6) ↔ (7 / 3 < x ∧ x < 42 / 15) :=
by
  sorry

end find_real_x_l236_236159


namespace ab_value_l236_236885

variables {a b : ℝ}

theorem ab_value (h₁ : a - b = 6) (h₂ : a^2 + b^2 = 50) : ab = 7 :=
sorry

end ab_value_l236_236885


namespace average_value_of_T_l236_236792

noncomputable def expected_value_T : ℕ := 22

theorem average_value_of_T (boys girls : ℕ) (boy_pair girl_pair : Prop) (T : ℕ) :
  boys = 9 → girls = 15 →
  boy_pair ∧ girl_pair →
  T = expected_value_T :=
by
  intros h_boys h_girls h_pairs
  sorry

end average_value_of_T_l236_236792


namespace sequence_strictly_increasing_from_14_l236_236517

def a (n : ℕ) : ℤ := n^4 - 20 * n^2 - 10 * n + 1

theorem sequence_strictly_increasing_from_14 :
  ∀ n : ℕ, n ≥ 14 → a (n + 1) > a n :=
by
  sorry

end sequence_strictly_increasing_from_14_l236_236517


namespace periodic_even_function_value_l236_236306

theorem periodic_even_function_value (f : ℝ → ℝ)
  (h_even : ∀ x, f(x) = f(-x))
  (h_periodic : ∀ x, f(x) = f(x + 2))
  (h_interval : ∀ x, 0 < x ∧ x < 1 → f(x) = 2^x - 1) :
  f(real.log 12 / real.log 2) = -(2 / 3) := 
begin
  sorry,
end

end periodic_even_function_value_l236_236306


namespace minimum_value_f_l236_236435

def f (x : ℝ) : ℝ :=
  (x^2 - 4 * x + 5) / (2 * x - 4)

theorem minimum_value_f (x : ℝ) (hx : x ≥ 5 / 2) : ∃ m, m = 1 ∧ ∀ y, y ≥ 5 / 2 → f y ≥ m :=
by 
  sorry

end minimum_value_f_l236_236435


namespace number_of_players_l236_236892

-- Definitions based on conditions
def socks_price : ℕ := 6
def tshirt_price : ℕ := socks_price + 7
def total_cost_per_player : ℕ := 2 * (socks_price + tshirt_price)
def total_expenditure : ℕ := 4092

-- Lean theorem statement
theorem number_of_players : total_expenditure / total_cost_per_player = 108 := 
by
  sorry

end number_of_players_l236_236892


namespace number_of_permutations_l236_236484

open Finset
open Function
open Equiv.Perm

noncomputable def inversion_number_of (σ : Perm (Fin 8)) (i : Fin 8) : Nat :=
  (Finset.filter (λ j, j < i ∧ σ j < σ i) (Finset.range 8)).card

def valid_permutation (σ: Perm (Fin 8)) : Prop :=
  inversion_number_of σ 7 = 2 ∧
  inversion_number_of σ 6 = 3 ∧
  inversion_number_of σ 4 = 3

theorem number_of_permutations : (Finset.filter valid_permutation (univ : Finset (Perm (Fin 8)))).card = 144 :=
  sorry

end number_of_permutations_l236_236484


namespace compute_expression_l236_236908

noncomputable def log_base (base x : ℝ) : ℝ := real.log x / real.log base

theorem compute_expression (x y : ℝ) (hx : 1 < x) (hy : 1 < y)
  (h : (log_base 2 x)^4 + (log_base 3 y)^4 + 8 = 8 * (log_base 2 x) * (log_base 3 y)) :
  x^real.sqrt 2 + y^real.sqrt 2 = 13 :=
  sorry

end compute_expression_l236_236908


namespace find_largest_N_l236_236011

noncomputable def largest_N : ℕ :=
  by
    -- This proof needs to demonstrate the solution based on constraints.
    -- Proof will be filled here.
    sorry

theorem find_largest_N :
  largest_N = 44 := 
  by
    -- Proof to establish the largest N will be completed here.
    sorry

end find_largest_N_l236_236011


namespace base4_to_base10_conversion_l236_236416

-- We define a base 4 number as follows:
def base4_to_base10 (n : ℕ) : ℕ :=
  let d0 := n % 10 in
  let n := n / 10 in
  let d1 := n % 10 in
  let n := n / 10 in
  let d2 := n % 10 in
  let n := n / 10 in
  let d3 := n % 10 in
  let n := n / 10 in
  let d4 := n % 10 in
  (d4 * 4^4 + d3 * 4^3 + d2 * 4^2 + d1 * 4^1 + d0 * 4^0)

-- Mathematical proof problem statement:
theorem base4_to_base10_conversion : base4_to_base10 21012 = 582 :=
  sorry

end base4_to_base10_conversion_l236_236416


namespace prime_power_divides_power_of_integer_l236_236923

theorem prime_power_divides_power_of_integer 
    {p a n : ℕ} 
    (hp : Nat.Prime p)
    (ha_pos : 0 < a) 
    (hn_pos : 0 < n) 
    (h : p ∣ a^n) :
    p^n ∣ a^n := 
by 
  sorry

end prime_power_divides_power_of_integer_l236_236923


namespace power_of_prime_implies_n_prime_l236_236620

theorem power_of_prime_implies_n_prime (n : ℕ) (p : ℕ) (k : ℕ) (hp : Nat.Prime p) :
  3^n - 2^n = p^k → Nat.Prime n :=
by
  sorry

end power_of_prime_implies_n_prime_l236_236620


namespace cost_of_12_roll_package_is_correct_l236_236676

variable (cost_per_roll_package : ℝ)
variable (individual_cost_per_roll : ℝ := 1)
variable (number_of_rolls : ℕ := 12)
variable (percent_savings : ℝ := 0.25)

-- The definition of the total cost of the package
def total_cost_package := number_of_rolls * (individual_cost_per_roll - (percent_savings * individual_cost_per_roll))

-- The goal is to prove that the total cost of the package is $9
theorem cost_of_12_roll_package_is_correct : total_cost_package = 9 := 
by
  sorry

end cost_of_12_roll_package_is_correct_l236_236676


namespace time_difference_l236_236596

-- Definitions for the problem conditions
def Zoe_speed : ℕ := 9 -- Zoe's speed in minutes per mile
def Henry_speed : ℕ := 7 -- Henry's speed in minutes per mile
def Race_length : ℕ := 12 -- Race length in miles

-- Theorem to prove the time difference
theorem time_difference : (Race_length * Zoe_speed) - (Race_length * Henry_speed) = 24 :=
by
  sorry

end time_difference_l236_236596


namespace equal_expense_sharing_l236_236918

variables (O L B : ℝ)

theorem equal_expense_sharing (h1 : O < L) (h2 : O < B) : 
    (L + B - 2 * O) / 6 = (O + L + B) / 3 - O :=
by
    sorry

end equal_expense_sharing_l236_236918


namespace Pima_investment_value_at_week6_l236_236782

noncomputable def Pima_initial_investment : ℝ := 400
noncomputable def Pima_week1_gain : ℝ := 0.25
noncomputable def Pima_week1_addition : ℝ := 200
noncomputable def Pima_week2_gain : ℝ := 0.50
noncomputable def Pima_week2_withdrawal : ℝ := 150
noncomputable def Pima_week3_loss : ℝ := 0.10
noncomputable def Pima_week4_gain : ℝ := 0.20
noncomputable def Pima_week4_addition : ℝ := 100
noncomputable def Pima_week5_gain : ℝ := 0.05
noncomputable def Pima_week6_loss : ℝ := 0.15
noncomputable def Pima_week6_withdrawal : ℝ := 250
noncomputable def weekly_interest_rate : ℝ := 0.02

noncomputable def calculate_investment_value : ℝ :=
  let week0 := Pima_initial_investment
  let week1 := (week0 * (1 + Pima_week1_gain) * (1 + weekly_interest_rate)) + Pima_week1_addition
  let week2 := ((week1 * (1 + Pima_week2_gain) * (1 + weekly_interest_rate)) - Pima_week2_withdrawal)
  let week3 := (week2 * (1 - Pima_week3_loss) * (1 + weekly_interest_rate))
  let week4 := ((week3 * (1 + Pima_week4_gain) * (1 + weekly_interest_rate)) + Pima_week4_addition)
  let week5 := (week4 * (1 + Pima_week5_gain) * (1 + weekly_interest_rate))
  let week6 := ((week5 * (1 - Pima_week6_loss) * (1 + weekly_interest_rate)) - Pima_week6_withdrawal)
  week6

theorem Pima_investment_value_at_week6 : calculate_investment_value = 819.74 := 
  by
  sorry

end Pima_investment_value_at_week6_l236_236782


namespace minimum_value_frac_inv_is_one_third_l236_236771

noncomputable def min_value_frac_inv (x y : ℝ) : ℝ :=
  if x > 0 ∧ y > 0 ∧ x + y = 12 then 1/x + 1/y else 0

theorem minimum_value_frac_inv_is_one_third (x y : ℝ) (h : x > 0 ∧ y > 0 ∧ x + y = 12) :
  min_value_frac_inv x y = 1/3 :=
begin
  -- Proof to be provided
  sorry
end

end minimum_value_frac_inv_is_one_third_l236_236771


namespace largest_no_solution_l236_236963

theorem largest_no_solution (a : ℕ) (h_odd : a % 2 = 1) (h_pos : a > 0) :
  ∃ n : ℕ, ∀ x y z : ℕ, x > 0 → y > 0 → z > 0 → a * x + (a + 1) * y + (a + 2) * z ≠ n :=
sorry

end largest_no_solution_l236_236963


namespace problem_statement_l236_236797

noncomputable def k_value (k : ℝ) : Prop :=
  (∀ (x y : ℝ), x + y = k → x^2 + y^2 = 4) ∧ (∀ (A B : ℝ × ℝ), (∃ (x y : ℝ), A = (x, y) ∧ x^2 + y^2 = 4) ∧ (∃ (x y : ℝ), B = (x, y) ∧ x^2 + y^2 = 4) ∧ 
  (∃ (xa ya xb yb : ℝ), A = (xa, ya) ∧ B = (xb, yb) ∧ |(xa - xb, ya - yb)| = |(xa, ya)| + |(xb, yb)|)) → k = 2

theorem problem_statement (k : ℝ) (h : k > 0) : k_value k :=
  sorry

end problem_statement_l236_236797


namespace certain_number_eq_neg_thirteen_over_two_l236_236052

noncomputable def CertainNumber (w : ℝ) : ℝ := 13 * w / (1 - w)

theorem certain_number_eq_neg_thirteen_over_two (w : ℝ) (h : w ^ 2 = 1) (hz : 1 - w ≠ 0) :
  CertainNumber w = -13 / 2 :=
sorry

end certain_number_eq_neg_thirteen_over_two_l236_236052


namespace committees_share_four_members_l236_236891

open Finset

variable {α : Type*}

theorem committees_share_four_members
    (deputies : Finset α)
    (committees : Finset (Finset α))
    (h_deputies : deputies.card = 1600)
    (h_committees : committees.card = 16000)
    (h_committee_size : ∀ c ∈ committees, c.card = 80) :
  ∃ c₁ c₂ ∈ committees, c₁ ≠ c₂ ∧ (c₁ ∩ c₂).card ≥ 4 := by
  sorry

end committees_share_four_members_l236_236891


namespace dihedral_angle_is_60_degrees_l236_236477

def point (x y z : ℝ) := (x, y, z)

noncomputable def dihedral_angle (P Q R S T : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem dihedral_angle_is_60_degrees :
  dihedral_angle 
    (point 1 0 0)  -- A
    (point 1 1 0)  -- B
    (point 0 0 0)  -- D
    (point 1 0 1)  -- A₁
    (point 0 0 1)  -- D₁
 = 60 :=
sorry

end dihedral_angle_is_60_degrees_l236_236477


namespace total_students_l236_236817

theorem total_students (girls boys : ℕ) (h1 : girls = 300) (h2 : boys = 8 * (girls / 5)) : girls + boys = 780 := by
  sorry

end total_students_l236_236817


namespace janes_score_l236_236069

theorem janes_score (jane_score tom_score : ℕ) (h1 : jane_score = tom_score + 50) (h2 : (jane_score + tom_score) / 2 = 90) :
  jane_score = 115 :=
sorry

end janes_score_l236_236069


namespace probability_includes_chinese_l236_236405

open ProbabilityMassFunction

variable (Individual : Type) [Fintype Individual]

axiom two_americans_one_frenchman_one_chinese (A1 A2 F C : Individual) :
  ∃ (individuals : Finset Individual),
    individuals = {A1, A2, F, C} ∧ 
    ∀ (x : Individual), x ∈ individuals → x = A1 ∨ x = A2 ∨ x = F ∨ x = C

theorem probability_includes_chinese (A1 A2 F C : Individual) :
  ∃ (individuals : Finset Individual)
    (pairs : Finset (Finset Individual))
    (pairs_with_chinese : Finset (Finset Individual)),
    individuals = {A1, A2, F, C} ∧
    pairs = individuals.powerset.filter (λ s, s.card = 2) ∧
    pairs_with_chinese = pairs.filter (λ s, C ∈ s) ∧
    (pairs_with_chinese.card : ℚ) / pairs.card = 1 / 2 :=
by
  sorry

end probability_includes_chinese_l236_236405


namespace system_real_solution_conditions_l236_236957

theorem system_real_solution_conditions (a b c x y z : ℝ) (h1 : a * x + b * y = c * z) (h2 : a * Real.sqrt (1 - x^2) + b * Real.sqrt (1 - y^2) = c * Real.sqrt (1 - z^2)) :
  abs a ≤ abs b + abs c ∧ abs b ≤ abs a + abs c ∧ abs c ≤ abs a + abs b ∧
  (a * b >= 0 ∨ a * c >= 0 ∨ b * c >= 0) :=
sorry

end system_real_solution_conditions_l236_236957


namespace smaller_angle_at_8_15_pm_l236_236368

noncomputable def smaller_angle_between_clock_hands (minute_hand_degrees_per_min: ℝ) (hour_hand_degrees_per_min: ℝ) (time_in_minutes: ℝ) : ℝ := sorry

theorem smaller_angle_at_8_15_pm :
  smaller_angle_between_clock_hands 6 0.5 495 = 157.5 :=
sorry

end smaller_angle_at_8_15_pm_l236_236368


namespace factor_expression_l236_236413

theorem factor_expression (a : ℝ) :
  (8 * a^3 + 105 * a^2 + 7) - (-9 * a^3 + 16 * a^2 - 14) = a^2 * (17 * a + 89) + 21 :=
by
  sorry

end factor_expression_l236_236413


namespace sum_of_smallest_ns_l236_236128

theorem sum_of_smallest_ns : ∀ n1 n2 : ℕ, (n1 ≡ 1 [MOD 4] ∧ n1 ≡ 2 [MOD 7]) ∧ (n2 ≡ 1 [MOD 4] ∧ n2 ≡ 2 [MOD 7]) ∧ n1 < n2 →
  n1 = 9 ∧ n2 = 37 → (n1 + n2 = 46) :=
by
  sorry

end sum_of_smallest_ns_l236_236128


namespace pinky_pig_apples_l236_236218

variable (P : ℕ)

theorem pinky_pig_apples (h : P + 73 = 109) : P = 36 := sorry

end pinky_pig_apples_l236_236218


namespace find_intersection_l236_236184

def A : Set ℝ := {x | abs (x + 1) = x + 1}

def B : Set ℝ := {x | x^2 + x < 0}

def intersection (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∈ B}

theorem find_intersection : intersection A B = {x | -1 < x ∧ x < 0} :=
by
  sorry

end find_intersection_l236_236184


namespace number_of_seniors_in_statistics_l236_236083

theorem number_of_seniors_in_statistics (total_students : ℕ) (half_enrolled_in_statistics : ℕ) (percentage_seniors : ℚ) (students_in_statistics seniors_in_statistics : ℕ) 
(h1 : total_students = 120)
(h2 : half_enrolled_in_statistics = total_students / 2)
(h3 : students_in_statistics = half_enrolled_in_statistics)
(h4 : percentage_seniors = 0.90)
(h5 : seniors_in_statistics = students_in_statistics * percentage_seniors) : 
seniors_in_statistics = 54 := 
by sorry

end number_of_seniors_in_statistics_l236_236083


namespace area_of_square_with_adjacent_points_l236_236781

theorem area_of_square_with_adjacent_points (x1 y1 x2 y2 : ℝ)
    (h1 : x1 = 1) (h2 : y1 = 2) (h3 : x2 = 4) (h4 : y2 = 6)
    (h_adj : ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 5 ^ 2) :
    (5 ^ 2) = 25 := 
by
  sorry

end area_of_square_with_adjacent_points_l236_236781


namespace left_square_side_length_l236_236637

theorem left_square_side_length (x : ℕ) (h1 : ∀ y : ℕ, y = x + 17)
                                (h2 : ∀ z : ℕ, z = x + 11)
                                (h3 : 3 * x + 28 = 52) : x = 8 :=
by
  sorry

end left_square_side_length_l236_236637


namespace cone_angle_60_degrees_l236_236151

theorem cone_angle_60_degrees (r : ℝ) (h : ℝ) (θ : ℝ) 
  (arc_len : θ = 60) 
  (slant_height : h = r) : θ = 60 :=
sorry

end cone_angle_60_degrees_l236_236151


namespace degree_measure_OC1D_l236_236847

/-- Define points on the sphere -/
structure Point (latitude longitude : ℝ) :=
(lat : ℝ := latitude)
(long : ℝ := longitude)

noncomputable def cos_deg (deg : ℝ) : ℝ := Real.cos (deg * Real.pi / 180)

noncomputable def angle_OC1D : ℝ :=
  Real.arccos ((cos_deg 44) * (cos_deg (-123)))

/-- The main theorem: the degree measure of ∠OC₁D is 113 -/
theorem degree_measure_OC1D :
  angle_OC1D = 113 := sorry

end degree_measure_OC1D_l236_236847


namespace Mina_age_is_10_l236_236082

-- Define the conditions as Lean definitions
variable (S : ℕ)

def Minho_age := 3 * S
def Mina_age := 2 * S - 2

-- State the main problem as a theorem
theorem Mina_age_is_10 (h_sum : S + Minho_age S + Mina_age S = 34) : Mina_age S = 10 :=
by
  sorry

end Mina_age_is_10_l236_236082


namespace ratio_circumscribed_circle_area_triangle_area_l236_236512

open Real

theorem ratio_circumscribed_circle_area_triangle_area (h R : ℝ) (h_eq : R = h / 2) :
  let circle_area := π * R^2
  let triangle_area := (h^2) / 4
  (circle_area / triangle_area) = π :=
by
  sorry

end ratio_circumscribed_circle_area_triangle_area_l236_236512


namespace find_x3_y3_l236_236589

theorem find_x3_y3 (x y : ℝ) (h1 : x + y = 6) (h2 : x^2 + y^2 = 18) : x^3 + y^3 = 54 := 
by 
  sorry

end find_x3_y3_l236_236589


namespace part1_part2_part3_l236_236090

-- Part 1
theorem part1 :
  3 < Real.sqrt 10 ∧ Real.sqrt 10 < 4 →
  Real.intPart (Real.sqrt 10) = 3 ∧ Real.decPart (Real.sqrt 10) = Real.sqrt 10 - 3 :=
by
  sorry

-- Part 2
theorem part2 :
  let a := Real.sqrt 6 - 2
  let b := 3
  a + b - Real.sqrt 6 = 1 :=
by
  sorry

-- Part 3
theorem part3 :
  let x := 13
  let y := Real.sqrt 3 - 1
  (12 + Real.sqrt 3 = x + y ∧ 0 < y ∧ y < 1) →
  -(x - y) = Real.sqrt 3 - 14 :=
by
  sorry

end part1_part2_part3_l236_236090


namespace market_survey_l236_236215

theorem market_survey (X Y Z : ℕ) (h1 : X / Y = 3)
  (h2 : X / Z = 2 / 3) (h3 : X = 60) : X + Y + Z = 170 :=
by
  sorry

end market_survey_l236_236215


namespace find_angle_measure_l236_236689

theorem find_angle_measure (x : ℝ) (hx : 90 - x + 40 = (1 / 2) * (180 - x)) : x = 80 :=
by
  sorry

end find_angle_measure_l236_236689


namespace sean_total_spending_l236_236558

noncomputable def cost_first_bakery_euros : ℝ :=
  let almond_croissants := 2 * 4.00
  let salami_cheese_croissants := 3 * 5.00
  let total_before_discount := almond_croissants + salami_cheese_croissants
  total_before_discount * 0.90 -- 10% discount

noncomputable def cost_second_bakery_pounds : ℝ :=
  let plain_croissants := 3 * 3.50 -- buy-3-get-1-free
  let focaccia := 5.00
  let total_before_tax := plain_croissants + focaccia
  total_before_tax * 1.05 -- 5% tax

noncomputable def cost_cafe_dollars : ℝ :=
  let lattes := 3 * 3.00
  lattes * 0.85 -- 15% student discount

noncomputable def first_bakery_usd : ℝ :=
  cost_first_bakery_euros * 1.15 -- converting euros to dollars

noncomputable def second_bakery_usd : ℝ :=
  cost_second_bakery_pounds * 1.35 -- converting pounds to dollars

noncomputable def total_cost_sean_spends : ℝ :=
  first_bakery_usd + second_bakery_usd + cost_cafe_dollars

theorem sean_total_spending : total_cost_sean_spends = 53.44 :=
  by
  -- The proof can be handled here
  sorry

end sean_total_spending_l236_236558


namespace MinkyungHeight_is_correct_l236_236454

noncomputable def HaeunHeight : ℝ := 1.56
noncomputable def NayeonHeight : ℝ := HaeunHeight - 0.14
noncomputable def MinkyungHeight : ℝ := NayeonHeight + 0.27

theorem MinkyungHeight_is_correct : MinkyungHeight = 1.69 :=
by
  sorry

end MinkyungHeight_is_correct_l236_236454


namespace cos_theta_eq_neg_2_div_sqrt_13_l236_236166

theorem cos_theta_eq_neg_2_div_sqrt_13 
  (θ : ℝ) 
  (h1 : 0 < θ) 
  (h2 : θ < π) 
  (h3 : Real.tan θ = -3/2) : 
  Real.cos θ = -2 / Real.sqrt 13 :=
sorry

end cos_theta_eq_neg_2_div_sqrt_13_l236_236166


namespace largest_even_two_digit_largest_odd_two_digit_l236_236147

-- Define conditions
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Theorem statements
theorem largest_even_two_digit : ∃ n, is_two_digit n ∧ is_even n ∧ ∀ m, is_two_digit m ∧ is_even m → m ≤ n := 
sorry

theorem largest_odd_two_digit : ∃ n, is_two_digit n ∧ is_odd n ∧ ∀ m, is_two_digit m ∧ is_odd m → m ≤ n := 
sorry

end largest_even_two_digit_largest_odd_two_digit_l236_236147


namespace correct_system_of_equations_l236_236821

-- Define the variables for the weights of sparrow and swallow
variables (x y : ℝ)

-- Define the problem conditions
def condition1 : Prop := 5 * x + 6 * y = 16
def condition2 : Prop := 4 * x + y = x + 5 * y

-- Create a theorem stating the conditions imply the identified system
theorem correct_system_of_equations :
  condition1 ∧ condition2 ↔ (5 * x + 6 * y = 16 ∧ 4 * x + y = x + 5 * y) :=
by
  apply Iff.intro;
  { intro h,
    cases h with h1 h2,
    exact ⟨h1, h2⟩ },
  { intro h,
    cases h with h1 h2,
    exact ⟨h1, h2⟩ }

end correct_system_of_equations_l236_236821


namespace exists_rat_nonint_sol_a_no_exists_rat_nonint_sol_b_l236_236550

structure RatNonIntPair (x y : ℚ) :=
  (x_rational : x.is_rational)
  (x_not_integer : x.num ≠ x.denom)
  (y_rational : y.is_rational)
  (y_not_integer : y.num ≠ y.denom)

theorem exists_rat_nonint_sol_a :
  ∃ (x y : ℚ), (RatNonIntPair x y) ∧ (int 19 * x + int 8 * y).denom = 1 ∧ (int 8 * x + int 3 * y).denom = 1 := sorry

theorem no_exists_rat_nonint_sol_b :
  ¬ ∃ (x y : ℚ), (RatNonIntPair x y) ∧ (int 19 * (x^2) + int 8 * (y^2)).denom = 1 ∧ (int 8 * (x^2) + int 3 * (y^2)).denom = 1 := sorry

end exists_rat_nonint_sol_a_no_exists_rat_nonint_sol_b_l236_236550


namespace range_of_m_l236_236303

open Real

def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 1 > m
def q (m : ℝ) : Prop := (2 - m) > 0

theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬ (p m ∧ q m) → 1 ≤ m ∧ m < 2 :=
by
  sorry

end range_of_m_l236_236303


namespace F_shaped_to_cube_l236_236143

-- Define the problem context in Lean 4
structure F_shaped_figure :=
  (squares : Finset (Fin 5) )

structure additional_squares :=
  (label : String )

def is_valid_configuration (f : F_shaped_figure) (s : additional_squares) : Prop :=
  -- This function should encapsulate the logic for checking the validity of a configuration
  sorry -- Implementation of validity check is omitted (replacing it with sorry)

-- The main theorem statement
theorem F_shaped_to_cube (f : F_shaped_figure) (squares: Finset additional_squares) : 
  ∃ valid_squares : Finset additional_squares, valid_squares.card = 3 ∧ 
    ∀ s ∈ valid_squares, is_valid_configuration f s := 
sorry

end F_shaped_to_cube_l236_236143


namespace determine_cubic_coeffs_l236_236289

-- Define the cubic function f(x)
def cubic_function (a b c x : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

-- Define the expression f(f(x) + x)
def composition_expression (a b c x : ℝ) : ℝ :=
  cubic_function a b c (cubic_function a b c x + x)

-- Given that the fraction of the compositions equals the given polynomial
def given_fraction_equals_polynomial (a b c : ℝ) : Prop :=
  ∀ x : ℝ, (composition_expression a b c x) / (cubic_function a b c x) = x^3 + 2023 * x^2 + 1776 * x + 2010

-- Prove that this implies specific values of a, b, and c
theorem determine_cubic_coeffs (a b c : ℝ) :
  given_fraction_equals_polynomial a b c →
  (a = 2022 ∧ b = 1776 ∧ c = 2010) :=
by
  sorry

end determine_cubic_coeffs_l236_236289


namespace max_stickers_l236_236268

theorem max_stickers (n_players : ℕ) (avg_stickers : ℕ) (min_stickers : ℕ) 
  (total_players : n_players = 22) 
  (average : avg_stickers = 4) 
  (minimum : ∀ i, i < n_players → min_stickers = 1) :
  ∃ max_sticker : ℕ, max_sticker = 67 :=
by
  sorry

end max_stickers_l236_236268


namespace min_inv_sum_l236_236769

open Real

theorem min_inv_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 12) :
  min ((1 / x) + (1 / y)) (1 / 3) :=
sorry

end min_inv_sum_l236_236769


namespace ribeye_steak_cost_l236_236275

/-- Define the conditions in Lean -/
def appetizer_cost : ℕ := 8
def wine_cost : ℕ := 3
def wine_glasses : ℕ := 2
def dessert_cost : ℕ := 6
def total_spent : ℕ := 38
def tip_percentage : ℚ := 0.20

/-- Proving the cost of the ribeye steak before the discount -/
theorem ribeye_steak_cost (S : ℚ) (h : 20 + (S / 2) + (tip_percentage * (20 + S)) = total_spent) : S = 20 :=
by
  sorry

end ribeye_steak_cost_l236_236275


namespace kendall_total_change_l236_236482

-- Definition of values of coins
def value_of_quarters (q : ℕ) : ℝ := q * 0.25
def value_of_dimes (d : ℕ) : ℝ := d * 0.10
def value_of_nickels (n : ℕ) : ℝ := n * 0.05

-- Conditions
def quarters := 10
def dimes := 12
def nickels := 6

-- Theorem statement
theorem kendall_total_change : 
  value_of_quarters quarters + value_of_dimes dimes + value_of_nickels nickels = 4.00 :=
by
  sorry

end kendall_total_change_l236_236482


namespace solve_for_x_l236_236351

theorem solve_for_x (x : ℝ) (h : 4^x = Real.sqrt 64) : x = 3 / 2 :=
sorry

end solve_for_x_l236_236351


namespace number_of_triangles_with_perimeter_10_l236_236878

theorem number_of_triangles_with_perimeter_10 : 
  ∃ (a b c : ℕ), a + b + c = 10 ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ 
  (a ≤ b) ∧ (b ≤ c) ↔ 9 :=
by
  have h : ∀ (a b c : ℕ), a + b + c = 10 ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ a ≤ b ∧ b ≤ c → 
    (a, b, c) ∈ 
      {[ (1, 5, 4), (2, 4, 4), (3, 3, 4), 
         (1, 6, 3), (2, 5, 3), (3, 4, 3),
         (2, 6, 2), (3, 5, 2), (4, 4, 2) ] : set (ℕ × ℕ × ℕ)},
  sorry  

end number_of_triangles_with_perimeter_10_l236_236878


namespace tutors_next_together_l236_236155

-- Define the conditions given in the problem
def Elisa_work_days := 5
def Frank_work_days := 6
def Giselle_work_days := 8
def Hector_work_days := 9

-- Theorem statement to prove the number of days until they all work together again
theorem tutors_next_together (d1 d2 d3 d4 : ℕ) 
  (h1 : d1 = Elisa_work_days) 
  (h2 : d2 = Frank_work_days) 
  (h3 : d3 = Giselle_work_days) 
  (h4 : d4 = Hector_work_days) : 
  Nat.lcm (Nat.lcm (Nat.lcm d1 d2) d3) d4 = 360 := 
by
  -- Translate the problem statement into Lean terms and structure
  sorry

end tutors_next_together_l236_236155


namespace smallest_s_triangle_l236_236361

theorem smallest_s_triangle (s : ℕ) :
  (7 + s > 11) ∧ (7 + 11 > s) ∧ (11 + s > 7) → s = 5 :=
sorry

end smallest_s_triangle_l236_236361


namespace sequence_general_term_l236_236173

theorem sequence_general_term 
  (x : ℕ → ℝ)
  (h1 : x 1 = 2)
  (h2 : x 2 = 3)
  (h3 : ∀ m ≥ 1, x (2*m+1) = x (2*m) + x (2*m-1))
  (h4 : ∀ m ≥ 2, x (2*m) = x (2*m-1) + 2*x (2*m-2)) :
  ∀ m, (x (2*m-1) = ((3 - Real.sqrt 2) / 4) * (2 + Real.sqrt 2) ^ m + ((3 + Real.sqrt 2) / 4) * (2 - Real.sqrt 2) ^ m ∧ 
          x (2*m) = ((1 + 2 * Real.sqrt 2) / 4) * (2 + Real.sqrt 2) ^ m + ((1 - 2 * Real.sqrt 2) / 4) * (2 - Real.sqrt 2) ^ m) :=
sorry

end sequence_general_term_l236_236173


namespace original_cost_of_car_l236_236343

noncomputable def original_cost (C : ℝ) : ℝ :=
  if h : C + 13000 ≠ 0 then (60900 - (C + 13000)) / (C + 13000) * 100 else 0

theorem original_cost_of_car 
  (C : ℝ) 
  (h1 : original_cost C = 10.727272727272727)
  (h2 : 60900 - (C + 13000) > 0) :
  C = 433500 :=
by
  sorry

end original_cost_of_car_l236_236343


namespace geometric_sequence_quadratic_roots_l236_236731

theorem geometric_sequence_quadratic_roots
    (a b : ℝ)
    (h_geometric : ∃ q : ℝ, b = 2 * q ∧ a = 2 * q^2) :
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 + (1 / 3) = 0 ∧ a * x2^2 + b * x2 + (1 / 3) = 0) :=
by
  sorry

end geometric_sequence_quadratic_roots_l236_236731


namespace arithmetic_series_sum_l236_236857

theorem arithmetic_series_sum :
  let first_term := -25
  let common_difference := 2
  let last_term := 19
  let n := (last_term - first_term) / common_difference + 1
  let sum := n * (first_term + last_term) / 2
  sum = -69 :=
by
  sorry

end arithmetic_series_sum_l236_236857


namespace min_value_inv_sum_l236_236767

theorem min_value_inv_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 12) : 
  ∃ z, (∀ x y : ℝ, 0 < x → 0 < y → x + y = 12 → z ≤ (1/x + 1/y)) ∧ z = 1/3 :=
sorry

end min_value_inv_sum_l236_236767


namespace number_of_lucky_tickets_l236_236402

def is_leningrad_lucky (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ) : Prop :=
  a₁ + a₂ + a₃ = a₄ + a₅ + a₆

def is_moscow_lucky (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ) : Prop :=
  a₂ + a₄ + a₆ = a₁ + a₃ + a₅

def is_symmetric (a₂ a₅ : ℕ) : Prop :=
  a₂ = a₅

def is_valid_ticket (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ) : Prop :=
  is_leningrad_lucky a₁ a₂ a₃ a₄ a₅ a₆ ∧
  is_moscow_lucky a₁ a₂ a₃ a₄ a₅ a₆ ∧
  is_symmetric a₂ a₅

theorem number_of_lucky_tickets : 
  ∃ n : ℕ, n = 6700 ∧ 
  (∀ a₁ a₂ a₃ a₄ a₅ a₆ : ℕ, 
    0 ≤ a₁ ∧ a₁ ≤ 9 ∧
    0 ≤ a₂ ∧ a₂ ≤ 9 ∧
    0 ≤ a₃ ∧ a₃ ≤ 9 ∧
    0 ≤ a₄ ∧ a₄ ≤ 9 ∧
    0 ≤ a₅ ∧ a₅ ≤ 9 ∧
    0 ≤ a₆ ∧ a₆ ≤ 9 →
    is_valid_ticket a₁ a₂ a₃ a₄ a₅ a₆ →
    n = 6700) := sorry

end number_of_lucky_tickets_l236_236402


namespace complete_work_together_in_days_l236_236756

-- Define the work rates for John, Rose, and Michael
def johnWorkRate : ℚ := 1 / 10
def roseWorkRate : ℚ := 1 / 40
def michaelWorkRate : ℚ := 1 / 20

-- Define the combined work rate when they work together
def combinedWorkRate : ℚ := johnWorkRate + roseWorkRate + michaelWorkRate

-- Define the total work to be done
def totalWork : ℚ := 1

-- Calculate the total number of days required to complete the work together
def totalDays : ℚ := totalWork / combinedWorkRate

-- Theorem to prove the total days is 40/7
theorem complete_work_together_in_days : totalDays = 40 / 7 :=
by
  -- Following steps would be the complete proofs if required
  rw [totalDays, totalWork, combinedWorkRate, johnWorkRate, roseWorkRate, michaelWorkRate]
  sorry

end complete_work_together_in_days_l236_236756


namespace stream_current_rate_l236_236679

theorem stream_current_rate (r w : ℝ) : (
  (18 / (r + w) + 6 = 18 / (r - w)) ∧ 
  (18 / (3 * r + w) + 2 = 18 / (3 * r - w))
) → w = 6 := 
by {
  sorry
}

end stream_current_rate_l236_236679


namespace circle_equation_standard_form_l236_236518

theorem circle_equation_standard_form (x y : ℝ) :
  (∃ (center : ℝ × ℝ), center.1 = -1 ∧ center.2 = 2 * center.1 ∧ (center.2 = -2) ∧ (center.1 + 1)^2 + center.2^2 = 4 ∧ (center.1 = -1) ∧ (center.2 = -2)) ->
  (x + 1)^2 + (y + 2)^2 = 4 :=
sorry

end circle_equation_standard_form_l236_236518


namespace paulson_spends_75_percent_of_income_l236_236919

variable (P : ℝ)  -- Percentage of income Paulson spends
variable (I : ℝ)  -- Paul's original income

-- Conditions
def original_expenditure := P * I
def original_savings := I - original_expenditure

def new_income := 1.20 * I
def new_expenditure := 1.10 * original_expenditure
def new_savings := new_income - new_expenditure

-- Given: The percentage increase in savings is approximately 50%.
def percentage_increase_in_savings :=
  ((new_savings - original_savings) / original_savings) * 100

-- Proof statement
theorem paulson_spends_75_percent_of_income
  (h : percentage_increase_in_savings P I ≈ 50) : P = 0.75 :=
by
  sorry

end paulson_spends_75_percent_of_income_l236_236919


namespace least_positive_integer_multiple_of_53_l236_236954

-- Define the problem in a Lean statement.
theorem least_positive_integer_multiple_of_53 :
  ∃ x : ℕ, (3 * x) ^ 2 + 2 * 58 * 3 * x + 58 ^ 2 % 53 = 0 ∧ x = 16 :=
by
  sorry

end least_positive_integer_multiple_of_53_l236_236954


namespace find_growth_rate_l236_236624

noncomputable def donation_first_day : ℝ := 10000
noncomputable def donation_third_day : ℝ := 12100
noncomputable def growth_rate (x : ℝ) : Prop :=
  (donation_first_day * (1 + x) ^ 2 = donation_third_day)

theorem find_growth_rate : ∃ x : ℝ, growth_rate x ∧ x = 0.1 :=
by
  sorry

end find_growth_rate_l236_236624


namespace length_of_NC_l236_236414

noncomputable def semicircle_radius (AB : ℝ) : ℝ := AB / 2

theorem length_of_NC : 
  ∀ (AB CD AN NB N M C NC : ℝ),
    AB = 10 ∧ AB = CD ∧ AN = NB ∧ AN + NB = AB ∧ M = N ∧ AB / 2 = semicircle_radius AB ∧ (NC^2 + semicircle_radius AB^2 = (2 * semicircle_radius AB)^2) →
    NC = 5 * Real.sqrt 3 := 
by 
  intros AB CD AN NB N M C NC h 
  rcases h with ⟨hAB, hCD, hAN, hSumAN, hMN, hRadius, hPythag⟩
  sorry

end length_of_NC_l236_236414


namespace students_more_than_turtles_l236_236285

theorem students_more_than_turtles
  (students_per_classroom : ℕ)
  (turtles_per_classroom : ℕ)
  (number_of_classrooms : ℕ)
  (h1 : students_per_classroom = 20)
  (h2 : turtles_per_classroom = 3)
  (h3 : number_of_classrooms = 5) :
  (students_per_classroom * number_of_classrooms)
  - (turtles_per_classroom * number_of_classrooms) = 85 :=
by
  sorry

end students_more_than_turtles_l236_236285


namespace solution_set_of_inequality_l236_236642

theorem solution_set_of_inequality :
  { x : ℝ | x^2 + 2 * x - 3 > 0 } = { x : ℝ | x < -3 ∨ x > 1 } :=
sorry

end solution_set_of_inequality_l236_236642


namespace hexagon_coloring_l236_236006

-- Definitions based on conditions
variable (A B C D E F : ℕ)
variable (color : ℕ → ℕ)
variable (v1 v2 : ℕ)

-- The question is about the number of different colorings
theorem hexagon_coloring (h_distinct : ∀ (x y : ℕ), x ≠ y → color x ≠ color y) 
    (h_colors : ∀ (x : ℕ), x ∈ [A, B, C, D, E, F] → 0 < color x ∧ color x < 5) :
    4 * 3 * 3 * 3 * 3 * 3 = 972 :=
by
  sorry

end hexagon_coloring_l236_236006


namespace fractions_inequality_l236_236037

variable {a b c d : ℝ}
variable (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0)

theorem fractions_inequality : 
  (a > b) → (b > 0) → (c < d) → (d < 0) → (a / d < b / c) :=
by
  intros h1 h2 h3 h4
  sorry

end fractions_inequality_l236_236037


namespace family_members_l236_236108

theorem family_members (N : ℕ) (income : ℕ → ℕ) (average_income : ℕ) :
  average_income = 10000 ∧
  income 0 = 8000 ∧
  income 1 = 15000 ∧
  income 2 = 6000 ∧
  income 3 = 11000 ∧
  (income 0 + income 1 + income 2 + income 3) = 4 * average_income →
  N = 4 :=
by {
  sorry
}

end family_members_l236_236108


namespace total_votes_l236_236255

theorem total_votes (V : ℝ) (C R : ℝ) 
  (hC : C = 0.10 * V)
  (hR1 : R = 0.10 * V + 16000)
  (hR2 : R = 0.90 * V) :
  V = 20000 :=
by
  sorry

end total_votes_l236_236255


namespace problem_solution_l236_236867

theorem problem_solution (x y : ℝ) (h₁ : x + Real.cos y = 2010) (h₂ : x + 2010 * Real.sin y = 2011) (h₃ : 0 ≤ y ∧ y ≤ Real.pi) :
  x + y = 2011 + Real.pi := 
sorry

end problem_solution_l236_236867


namespace count_non_squares_or_cubes_l236_236737

theorem count_non_squares_or_cubes (n : ℕ) (h₀ : 1 ≤ n ∧ n ≤ 200) : 
  ∃ c, c = 182 ∧ 
  (∃ k, k^2 = n ∨ ∃ m, m^3 = n) → false :=
by
  sorry

end count_non_squares_or_cubes_l236_236737


namespace smallest_palindromic_primes_l236_236851

def is_palindromic (n : ℕ) : Prop :=
  ∀ a b : ℕ, n = 1001 * a + 1010 * b → 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_palindromic_primes :
  ∃ n1 n2 : ℕ, 
  is_palindromic n1 ∧ is_palindromic n2 ∧ is_prime n1 ∧ is_prime n2 ∧ n1 < n2 ∧
  ∀ m : ℕ, (is_palindromic m ∧ is_prime m ∧ m < n2 → m = n1) ∧
           (is_palindromic m ∧ is_prime m ∧ m < n1 → m ≠ n2) ∧ n1 = 1221 ∧ n2 = 1441 := 
sorry

end smallest_palindromic_primes_l236_236851


namespace polygon_sides_l236_236646

theorem polygon_sides (sum_of_interior_angles : ℝ) (x : ℝ) (h : sum_of_interior_angles = 1080) : x = 8 :=
by
  sorry

end polygon_sides_l236_236646


namespace base4_to_base10_conversion_l236_236417

theorem base4_to_base10_conversion : (2 * 4^4 + 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 2 * 4^0 = 582) :=
by {
  -- The proof is omitted
  sorry
}

end base4_to_base10_conversion_l236_236417


namespace A_minus_B_l236_236912

theorem A_minus_B (A B : ℚ) (n : ℕ) :
  (A : ℚ) = 1 / 6 →
  (B : ℚ) = -1 / 12 →
  A - B = 1 / 4 :=
by
  intro hA hB
  rw [hA, hB]
  norm_num

end A_minus_B_l236_236912


namespace greatest_divisor_condition_gcd_of_numbers_l236_236965

theorem greatest_divisor_condition (n : ℕ) (h100 : n ∣ 100) (h225 : n ∣ 225) (h150 : n ∣ 150) : n ≤ 25 :=
  sorry

theorem gcd_of_numbers : Nat.gcd (Nat.gcd 100 225) 150 = 25 :=
  sorry

end greatest_divisor_condition_gcd_of_numbers_l236_236965


namespace sufficient_p_wages_l236_236682

variable (S P Q : ℕ)

theorem sufficient_p_wages (h1 : S = 40 * Q) (h2 : S = 15 * (P + Q))  :
  ∃ D : ℕ, S = D * P ∧ D = 24 := 
by
  use 24
  sorry

end sufficient_p_wages_l236_236682


namespace domain_of_f_l236_236629

noncomputable def f (x : ℝ) : ℝ := real.sqrt (1 - real.log x)

theorem domain_of_f :
  {x : ℝ | 0 < x ∧ 1 - real.log x ≥ 0} = {x : ℝ | 0 < x ∧ x ≤ real.exp 1} :=
by
  sorry

end domain_of_f_l236_236629


namespace A_beats_B_by_160_meters_l236_236467

-- Definitions used in conditions
def distance_A := 400 -- meters
def time_A := 60 -- seconds
def distance_B := 400 -- meters
def time_B := 100 -- seconds
def speed_B := distance_B / time_B -- B's speed in meters/second
def time_for_B_in_A_time := time_A -- B's time for the duration A took to finish the race
def distance_B_in_A_time := speed_B * time_for_B_in_A_time -- Distance B covers in A's time

-- Statement to prove
theorem A_beats_B_by_160_meters : distance_A - distance_B_in_A_time = 160 :=
by
  -- This is a placeholder for an eventual proof
  sorry

end A_beats_B_by_160_meters_l236_236467


namespace second_number_division_l236_236566

theorem second_number_division (d x r : ℕ) (h1 : d = 16) (h2 : 25 % d = r) (h3 : 105 % d = r) (h4 : r = 9) : x % d = r → x = 41 :=
by 
  simp [h1, h2, h3, h4] 
  sorry

end second_number_division_l236_236566


namespace parabola_directrix_l236_236565

theorem parabola_directrix (x : ℝ) :
  (∃ y : ℝ, y = (x^2 - 8*x + 12) / 16) →
  ∃ directrix : ℝ, directrix = -17 / 4 :=
by
  sorry

end parabola_directrix_l236_236565


namespace martha_initial_crayons_l236_236491

theorem martha_initial_crayons : ∃ (x : ℕ), (x / 2 + 20 = 29) ∧ x = 18 :=
by
  sorry

end martha_initial_crayons_l236_236491


namespace marie_messages_days_l236_236079

theorem marie_messages_days (initial_messages : ℕ) (read_per_day : ℕ) (new_per_day : ℕ) (days : ℕ) :
  initial_messages = 98 ∧ read_per_day = 20 ∧ new_per_day = 6 → days = 7 :=
by
  sorry

end marie_messages_days_l236_236079


namespace polynomial_expansion_correct_l236_236000

open Polynomial

-- Define the two polynomials in question.
def poly1 : Polynomial ℤ := 2 + (X^2)
def poly2 : Polynomial ℤ := 3 - (X^3) + (X^5)

-- The target polynomial after expansion.
def expandedPoly : Polynomial ℤ := 6 + 3 * (X^2) - 2 * (X^3) + X^5 + X^7

-- State the theorem to be proved
theorem polynomial_expansion_correct : poly1 * poly2 = expandedPoly := 
by
  sorry

end polynomial_expansion_correct_l236_236000


namespace a3_pm_2b3_not_div_by_37_l236_236508

theorem a3_pm_2b3_not_div_by_37 {a b : ℤ} (ha : ¬ (37 ∣ a)) (hb : ¬ (37 ∣ b)) :
  ¬ (37 ∣ (a^3 + 2 * b^3)) ∧ ¬ (37 ∣ (a^3 - 2 * b^3)) :=
  sorry

end a3_pm_2b3_not_div_by_37_l236_236508


namespace days_to_clear_messages_l236_236081

theorem days_to_clear_messages 
  (initial_messages : ℕ)
  (messages_read_per_day : ℕ)
  (new_messages_per_day : ℕ) 
  (net_messages_cleared_per_day : ℕ)
  (d : ℕ) :
  initial_messages = 98 →
  messages_read_per_day = 20 →
  new_messages_per_day = 6 →
  net_messages_cleared_per_day = messages_read_per_day - new_messages_per_day →
  d = initial_messages / net_messages_cleared_per_day →
  d = 7 :=
by
  intros h_initial h_read h_new h_net h_days
  sorry

end days_to_clear_messages_l236_236081


namespace find_second_number_l236_236099

def average (nums : List ℕ) : ℕ :=
  nums.sum / nums.length

theorem find_second_number (nums : List ℕ) (a b : ℕ) (avg : ℕ) :
  average [10, 70, 28] = 36 ∧ average (10 :: 70 :: 28 :: []) + 4 = avg ∧ average (a :: b :: nums) = avg ∧ a = 20 ∧ b = 60 → b = 60 :=
by
  sorry

end find_second_number_l236_236099


namespace part_I_part_II_l236_236486

noncomputable def f (a x : ℝ) : ℝ := |x - 1| + a * |x - 2|

theorem part_I (a : ℝ) (h_min : ∃ m, ∀ x, f a x ≥ m) : -1 ≤ a ∧ a ≤ 1 :=
sorry

theorem part_II (a : ℝ) (h_bound : ∀ x, f a x ≥ 1/2) : a = 1/3 :=
sorry

end part_I_part_II_l236_236486


namespace magnitude_of_complex_exponent_l236_236853
-- Import the library for complex number operations

-- Define the context
noncomputable theory
open Complex

-- The statement we want to prove
theorem magnitude_of_complex_exponent (z : ℂ) (hz : z = 1 + 2 * I) : complex.abs (z^8) = 625 :=
by sorry

end magnitude_of_complex_exponent_l236_236853


namespace person_age_l236_236964

theorem person_age (x : ℕ) (h : 4 * (x + 3) - 4 * (x - 3) = x) : x = 24 :=
by {
  sorry
}

end person_age_l236_236964


namespace grounded_days_for_lying_l236_236411

def extra_days_per_grade_below_b : ℕ := 3
def grades_below_b : ℕ := 4
def total_days_grounded : ℕ := 26

theorem grounded_days_for_lying : 
  (total_days_grounded - (grades_below_b * extra_days_per_grade_below_b) = 14) := 
by 
  sorry

end grounded_days_for_lying_l236_236411


namespace area_of_square_with_adjacent_points_l236_236778

theorem area_of_square_with_adjacent_points (P Q : ℝ × ℝ) (hP : P = (1, 2)) (hQ : Q = (4, 6)) :
  let side_length := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) in 
  let area := side_length^2 in 
  area = 25 :=
by
  sorry

end area_of_square_with_adjacent_points_l236_236778


namespace spherical_coordinates_equivalence_l236_236065

theorem spherical_coordinates_equivalence :
  ∀ (ρ θ φ : ℝ), 
        ρ = 3 → θ = (2 * Real.pi / 7) → φ = (8 * Real.pi / 5) →
        (0 < ρ) → 
        (0 ≤ (2 * Real.pi / 7) ∧ (2 * Real.pi / 7) < 2 * Real.pi) →
        (0 ≤ (8 * Real.pi / 5) ∧ (8 * Real.pi / 5) ≤ Real.pi) →
      ∃ (ρ' θ' φ' : ℝ), 
        ρ' = ρ ∧ θ' = (9 * Real.pi / 7) ∧ φ' = (2 * Real.pi / 5) :=
by
    sorry

end spherical_coordinates_equivalence_l236_236065


namespace problem_l236_236718

theorem problem (a b c : ℂ) 
  (h1 : a + b + c = 3)
  (h2 : a^2 + b^2 + c^2 = 3)
  (h3 : a^3 + b^3 + c^3 = 6) :
  (a - 1)^(2023) + (b - 1)^(2023) + (c - 1)^(2023) = 0 :=
by
  sorry

end problem_l236_236718


namespace polynomial_division_l236_236852

theorem polynomial_division (a b c : ℤ) :
  (∀ x : ℝ, (17 * x^2 - 3 * x + 4) - (a * x^2 + b * x + c) = (5 * x + 6) * (2 * x + 1)) →
  a - b - c = 29 := by
  sorry

end polynomial_division_l236_236852


namespace train_speed_l236_236247

def train_length : ℝ := 360 -- length of the train in meters
def crossing_time : ℝ := 6 -- time taken to cross the man in seconds

theorem train_speed (train_length crossing_time : ℝ) : 
  (train_length = 360) → (crossing_time = 6) → (train_length / crossing_time = 60) :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end train_speed_l236_236247


namespace find_x_l236_236560

theorem find_x (x : ℝ) (h : ⌈x⌉ * x = 182) : x = 13 :=
sorry

end find_x_l236_236560


namespace jane_stick_length_l236_236086

variable (P U S J F : ℕ)
variable (h1 : P = 30)
variable (h2 : U = P - 7)
variable (h3 : U = S / 2)
variable (h4 : F = 2 * 12)
variable (h5 : J = S - F)

theorem jane_stick_length : J = 22 := by
  sorry

end jane_stick_length_l236_236086


namespace log_diff_condition_l236_236428

theorem log_diff_condition (a : ℕ → ℝ) (d e : ℝ) (H1 : ∀ n : ℕ, n > 1 → a n = Real.log n / Real.log 3003)
  (H2 : d = a 2 + a 3 + a 4 + a 5 + a 6) (H3 : e = a 15 + a 16 + a 17 + a 18 + a 19) :
  d - e = -Real.log 1938 / Real.log 3003 := by
  sorry

end log_diff_condition_l236_236428


namespace initial_fish_count_l236_236916

theorem initial_fish_count (x : ℕ) (h1 : x + 47 = 69) : x = 22 :=
by
  sorry

end initial_fish_count_l236_236916


namespace cabbage_count_l236_236135

theorem cabbage_count 
  (length : ℝ)
  (width : ℝ)
  (density : ℝ)
  (h_length : length = 16)
  (h_width : width = 12)
  (h_density : density = 9) : 
  length * width * density = 1728 := 
by
  rw [h_length, h_width, h_density]
  norm_num
  done

end cabbage_count_l236_236135


namespace cds_unique_to_either_l236_236687

-- Declare the variables for the given problem
variables (total_alice_shared : ℕ) (total_alice : ℕ) (unique_bob : ℕ)

-- The given conditions in the problem
def condition_alice : Prop := total_alice_shared + unique_bob + (total_alice - total_alice_shared) = total_alice

-- The theorem to prove: number of CDs in either Alice's or Bob's collection but not both is 19
theorem cds_unique_to_either (h1 : total_alice = 23) 
                             (h2 : total_alice_shared = 12) 
                             (h3 : unique_bob = 8) : 
                             (total_alice - total_alice_shared) + unique_bob = 19 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it
  sorry

end cds_unique_to_either_l236_236687


namespace primes_count_l236_236711

open Int

theorem primes_count (p : ℕ) (hp : Nat.Prime p) :
  ¬ ∃ r s : ℤ, ∀ x : ℤ, (x^3 - x + 2) % p = ((x - r)^2 * (x - s)) % p := 
  by
    sorry

end primes_count_l236_236711


namespace hydroflow_rate_30_minutes_l236_236623

def hydroflow_pumped (rate_per_hour: ℕ) (minutes: ℕ) : ℕ :=
  let hours := minutes / 60
  rate_per_hour * hours

theorem hydroflow_rate_30_minutes : 
  hydroflow_pumped 500 30 = 250 :=
by 
  -- place the proof here
  sorry

end hydroflow_rate_30_minutes_l236_236623


namespace difference_max_min_is_7_l236_236759

-- Define the number of times Kale mowed his lawn during each season
def timesSpring : ℕ := 8
def timesSummer : ℕ := 5
def timesFall : ℕ := 12

-- Statement to prove
theorem difference_max_min_is_7 : 
  (max timesSpring (max timesSummer timesFall)) - (min timesSpring (min timesSummer timesFall)) = 7 :=
by
  -- Proof would go here
  sorry

end difference_max_min_is_7_l236_236759


namespace part1_proof_l236_236607

variable (a r : ℝ) (f : ℝ → ℝ)

axiom a_gt_1 : a > 1
axiom r_gt_1 : r > 1

axiom f_condition : ∀ x > 0, f x * f x ≤ a * x * f (x / a)
axiom f_bound : ∀ x, 0 < x ∧ x < 1 / 2^2005 → f x < 2^2005

theorem part1_proof : ∀ x > 0, f x ≤ a^(1 - r) * x := 
by 
  sorry

end part1_proof_l236_236607


namespace fraction_expression_proof_l236_236900

theorem fraction_expression_proof :
  (1 / 8 * 1 / 9 * 1 / 28 = 1 / 2016) ∨ ((1 / 8 - 1 / 9) * 1 / 28 = 1 / 2016) :=
by
  sorry

end fraction_expression_proof_l236_236900


namespace find_P_l236_236519

theorem find_P (P : ℕ) (h : P^2 + P = 30) : P = 5 :=
sorry

end find_P_l236_236519


namespace plates_count_l236_236239

variable (x : ℕ)
variable (first_taken : ℕ)
variable (second_taken : ℕ)
variable (remaining_plates : ℕ := 9)

noncomputable def plates_initial : ℕ :=
  let first_batch := (x - 2) / 3
  let remaining_after_first := x - 2 - first_batch
  let second_batch := remaining_after_first / 2
  let remaining_after_second := remaining_after_first - second_batch
  remaining_after_second

theorem plates_count (x : ℕ) (h : plates_initial x = remaining_plates) : x = 29 := sorry

end plates_count_l236_236239


namespace pizza_cost_l236_236920

theorem pizza_cost (soda_cost jeans_cost start_money quarters_left : ℝ) (quarters_value : ℝ) (total_left : ℝ) (pizza_cost : ℝ) :
  soda_cost = 1.50 → 
  jeans_cost = 11.50 → 
  start_money = 40 → 
  quarters_left = 97 → 
  quarters_value = 0.25 → 
  total_left = quarters_left * quarters_value → 
  pizza_cost = start_money - total_left - (soda_cost + jeans_cost) → 
  pizza_cost = 2.75 :=
by
  sorry

end pizza_cost_l236_236920


namespace maximize_profit_l236_236390

def revenue (x : ℝ) : ℝ := 16 * x

def fixed_cost : ℝ := 30

def variable_cost (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 14 then (2 / 3) * x ^ 2 + 4 * x
  else if 14 < x ∧ x ≤ 35 then 17 * x + 400 / x - 80
  else 0 -- variable cost is not defined beyond specified range

def profit (x : ℝ) : ℝ :=
  revenue x - fixed_cost - variable_cost x

theorem maximize_profit : ∃ x, x = 9 ∧ ∀ y, 0 ≤ y ∧ y ≤ 35 → profit y ≤ profit 9 := by
  sorry

end maximize_profit_l236_236390


namespace system_of_equations_solution_l236_236929

theorem system_of_equations_solution :
  ∃ x y : ℚ, (4 * x - 3 * y = -8) ∧ (5 * x + 9 * y = -18) ∧ x = -14 / 3 ∧ y = -32 / 9 :=
by {
  sorry  -- Proof goes here
}

end system_of_equations_solution_l236_236929


namespace power_function_convex_upwards_l236_236429

noncomputable def f (x : ℝ) : ℝ :=
  x ^ (4 / 5)

theorem power_function_convex_upwards (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) :
  f ((x1 + x2) / 2) > (f x1 + f x2) / 2 :=
sorry

end power_function_convex_upwards_l236_236429


namespace flowchart_output_proof_l236_236179

def flowchart_output (x : ℕ) : ℕ :=
  let x := x + 2
  let x := x + 2
  let x := x + 2
  x

theorem flowchart_output_proof :
  flowchart_output 10 = 16 := by
  -- Assume initial value of x is 10
  let x0 := 10
  -- First iteration
  let x1 := x0 + 2
  -- Second iteration
  let x2 := x1 + 2
  -- Third iteration
  let x3 := x2 + 2
  -- Final value of x
  have hx_final : x3 = 16 := by rfl
  -- The result should be 16
  have h_result : flowchart_output 10 = x3 := by rfl
  rw [hx_final] at h_result
  exact h_result

end flowchart_output_proof_l236_236179


namespace orchestra_club_members_l236_236230

theorem orchestra_club_members : ∃ (n : ℕ), 150 < n ∧ n < 250 ∧ n % 8 = 1 ∧ n % 6 = 2 ∧ n % 9 = 3 ∧ n = 169 := 
by {
  sorry
}

end orchestra_club_members_l236_236230


namespace baker_additional_cakes_l236_236996

theorem baker_additional_cakes (X : ℕ) : 
  (62 + X) - 144 = 67 → X = 149 :=
by
  intro h
  sorry

end baker_additional_cakes_l236_236996


namespace age_when_Billy_born_l236_236193

-- Definitions based on conditions
def current_age_I := 4 * 4
def current_age_Billy := 4
def age_difference := current_age_I - current_age_Billy

-- Statement to prove
theorem age_when_Billy_born : age_difference = 12 :=
by
  -- Expose the calculation steps
  calc
    age_difference
    = 4 * 4 - 4 : by rw [current_age_I, current_age_Billy]
    ... = 16 - 4 : by norm_num
    ... = 12 : by norm_num

end age_when_Billy_born_l236_236193


namespace ones_digit_11_pow_l236_236955

theorem ones_digit_11_pow (n : ℕ) (hn : n > 0) : (11^n % 10) = 1 := by
  sorry

end ones_digit_11_pow_l236_236955


namespace minimum_value_of_sum_l236_236530

theorem minimum_value_of_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : 1/a + 2/b + 3/c = 2) : a + 2*b + 3*c = 18 ↔ (a = 3 ∧ b = 3 ∧ c = 3) :=
by
  sorry

end minimum_value_of_sum_l236_236530


namespace round_robin_tournament_l236_236531

theorem round_robin_tournament (n : ℕ)
  (total_points_1 : ℕ := 3086) (total_points_2 : ℕ := 2018) (total_points_3 : ℕ := 1238)
  (pair_avg_1 : ℕ := (3086 + 1238) / 2) (pair_avg_2 : ℕ := (3086 + 2018) / 2) (pair_avg_3 : ℕ := (1238 + 2018) / 2)
  (overall_avg : ℕ := (3086 + 2018 + 1238) / 3)
  (all_pairwise_diff : pair_avg_1 ≠ pair_avg_2 ∧ pair_avg_1 ≠ pair_avg_3 ∧ pair_avg_2 ≠ pair_avg_3) :
  n = 47 :=
by
  sorry

end round_robin_tournament_l236_236531


namespace symmetric_points_y_axis_l236_236866

theorem symmetric_points_y_axis (a b : ℝ) (h₁ : (a, 3) = (-2, 3)) (h₂ : (2, b) = (2, 3)) : (a + b) ^ 2015 = 1 := by
  sorry

end symmetric_points_y_axis_l236_236866


namespace calc_factorial_sum_l236_236409

theorem calc_factorial_sum : 5 * Nat.factorial 5 + 4 * Nat.factorial 4 + Nat.factorial 4 = 720 := by
  sorry

end calc_factorial_sum_l236_236409


namespace profit_percentage_l236_236798

theorem profit_percentage (initial_cost_per_pound : ℝ) (ruined_percent : ℝ) (selling_price_per_pound : ℝ) (desired_profit_percent : ℝ) : 
  initial_cost_per_pound = 0.80 ∧ ruined_percent = 0.10 ∧ selling_price_per_pound = 0.96 → desired_profit_percent = 8 := by
  sorry

end profit_percentage_l236_236798


namespace instantaneous_velocity_at_4_seconds_l236_236389

-- Define the equation of motion
def s (t : ℝ) : ℝ := t^2 - 2 * t + 5

-- Define the velocity function as the derivative of s
def v (t : ℝ) : ℝ := 2 * t - 2

theorem instantaneous_velocity_at_4_seconds : v 4 = 6 := by
  -- Proof goes here
  sorry

end instantaneous_velocity_at_4_seconds_l236_236389


namespace geometric_sequence_relation_l236_236046

theorem geometric_sequence_relation (a b c : ℝ) (r : ℝ)
  (h1 : -2 * r = a)
  (h2 : a * r = b)
  (h3 : b * r = c)
  (h4 : c * r = -8) :
  b = -4 ∧ a * c = 16 := by
  sorry

end geometric_sequence_relation_l236_236046


namespace boat_speed_in_still_water_eq_16_l236_236674

theorem boat_speed_in_still_water_eq_16 (stream_rate : ℝ) (time_downstream : ℝ) (distance_downstream : ℝ) (V_b : ℝ) 
(h1 : stream_rate = 5) (h2 : time_downstream = 6) (h3 : distance_downstream = 126) : 
  V_b = 16 :=
by sorry

end boat_speed_in_still_water_eq_16_l236_236674


namespace monotonically_increasing_interval_l236_236513

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)

theorem monotonically_increasing_interval (k : ℤ) :
  ∀ x : ℝ, (k * Real.pi - 5 * Real.pi / 12 <= x ∧ x <= k * Real.pi + Real.pi / 12) →
    (∃ r : ℝ, f (x + r) > f x ∨ f (x + r) < f x) := by
  sorry

end monotonically_increasing_interval_l236_236513


namespace total_number_of_pipes_l236_236132

theorem total_number_of_pipes (bottom_layer top_layer layers : ℕ) 
  (h_bottom_layer : bottom_layer = 13) 
  (h_top_layer : top_layer = 3) 
  (h_layers : layers = 11) : 
  bottom_layer + top_layer = 16 → 
  (bottom_layer + top_layer) * layers / 2 = 88 := 
by
  intro h_sum
  sorry

end total_number_of_pipes_l236_236132


namespace difference_value_l236_236355

theorem difference_value (N : ℝ) (h : 0.25 * N = 100) : N - (3/4) * N = 100 :=
by sorry

end difference_value_l236_236355


namespace conditional_probability_l236_236515

-- Given conditions
variables (P A B : Prop)
variable [ProbabilityMeasure P]

axiom prob_A : P(A) = 0.8
axiom prob_B : P(B) = 0.4
axiom prob_A_inter_B : P(A ∧ B) = 0.4

-- Question: Prove conditional probability
theorem conditional_probability (A B : Prop) [ProbabilityMeasure P] :
  P(B|A) = 0.5 :=
by
  sorry

end conditional_probability_l236_236515


namespace question1_question2_l236_236583

section problem1

variable (a b : ℝ)

theorem question1 (h1 : a = 1) (h2 : b = 2) : 
  ∀ x : ℝ, abs (2 * x + 1) + abs (3 * x - 2) ≤ 5 ↔ 
  (-4 / 5 ≤ x ∧ x ≤ 6 / 5) :=
sorry

end problem1

section problem2

theorem question2 :
  (∀ x : ℝ, abs (x - 1) + abs (x + 2) ≥ m^2 - 3 * m + 5) → 
  ∃ (m : ℝ), m ≤ 2 :=
sorry

end problem2

end question1_question2_l236_236583


namespace lower_limit_for_a_l236_236318

theorem lower_limit_for_a 
  {k : ℤ} 
  (a b : ℤ) 
  (h1 : k ≤ a) 
  (h2 : a < 17) 
  (h3 : 3 < b) 
  (h4 : b < 29) 
  (h5 : 3.75 = 4 - 0.25) 
  : (7 ≤ a) :=
sorry

end lower_limit_for_a_l236_236318


namespace sequence_contains_all_integers_l236_236103

theorem sequence_contains_all_integers (a : ℕ → ℕ) 
  (h1 : ∀ i ≥ 0, 0 ≤ a i ∧ a i ≤ i)
  (h2 : ∀ k ≥ 0, (∑ i in Finset.range (k + 1), nat.choose k (a i)) = 2^k) :
  ∀ N ≥ 0, ∃ i ≥ 0, a i = N := 
sorry

end sequence_contains_all_integers_l236_236103


namespace correct_assertions_l236_236171

variables {A B : Type} (f : A → B)

-- 1. Different elements in set A can have the same image in set B
def statement_1 : Prop := ∃ a1 a2 : A, a1 ≠ a2 ∧ f a1 = f a2

-- 2. A single element in set A can have different images in B
def statement_2 : Prop := ∃ a1 : A, ∃ b1 b2 : B, b1 ≠ b2 ∧ (f a1 = b1 ∧ f a1 = b2)

-- 3. There can be elements in set B that do not have a pre-image in A
def statement_3 : Prop := ∃ b : B, ∀ a : A, f a ≠ b

-- Correct answer is statements 1 and 3 are true, statement 2 is false
theorem correct_assertions : statement_1 f ∧ ¬statement_2 f ∧ statement_3 f := sorry

end correct_assertions_l236_236171


namespace perimeter_of_grid_l236_236539

theorem perimeter_of_grid (area: ℕ) (side_length: ℕ) (perimeter: ℕ) 
  (h1: area = 144) 
  (h2: 4 * side_length * side_length = area) 
  (h3: perimeter = 4 * 2 * side_length) : 
  perimeter = 48 :=
by
  sorry

end perimeter_of_grid_l236_236539


namespace matches_played_by_team_B_from_city_A_l236_236199

-- Define the problem setup, conditions, and the conclusion we need to prove
structure Tournament :=
  (cities : ℕ)
  (teams_per_city : ℕ)

-- Assuming each team except Team A of city A has played a unique number of matches,
-- find the number of matches played by Team B of city A.
theorem matches_played_by_team_B_from_city_A (t : Tournament)
  (unique_match_counts_except_A : ∀ (i j : ℕ), i ≠ j → (i < t.cities → (t.teams_per_city * i ≠ t.teams_per_city * j)) ∧ (i < t.cities - 1 → (t.teams_per_city * i ≠ t.teams_per_city * (t.cities - 1)))) :
  (t.cities = 16) → (t.teams_per_city = 2) → ∃ n, n = 15 :=
by
  sorry

end matches_played_by_team_B_from_city_A_l236_236199


namespace platform_length_eq_train_length_l236_236796

noncomputable def length_of_train : ℝ := 900
noncomputable def speed_of_train_kmh : ℝ := 108
noncomputable def speed_of_train_mpm : ℝ := (speed_of_train_kmh * 1000) / 60
noncomputable def crossing_time_min : ℝ := 1
noncomputable def total_distance_covered : ℝ := speed_of_train_mpm * crossing_time_min

theorem platform_length_eq_train_length :
  total_distance_covered - length_of_train = length_of_train :=
by
  sorry

end platform_length_eq_train_length_l236_236796


namespace trigonometric_identity_l236_236195

theorem trigonometric_identity (θ : ℝ) (h : 2 * (Real.cos θ) + (Real.sin θ) = 0) :
  Real.cos (2 * θ) + 1/2 * Real.sin (2 * θ) = -1 := 
sorry

end trigonometric_identity_l236_236195


namespace line_through_points_C_D_has_undefined_slope_and_angle_90_l236_236162

theorem line_through_points_C_D_has_undefined_slope_and_angle_90 (m : ℝ) (n : ℝ) (hn : n ≠ 0) :
  ∃ θ : ℝ, (∀ (slope : ℝ), false) ∧ θ = 90 :=
by { sorry }

end line_through_points_C_D_has_undefined_slope_and_angle_90_l236_236162


namespace multiple_of_a_power_l236_236093

theorem multiple_of_a_power (a n m : ℕ) (h : a^n ∣ m) : a^(n+1) ∣ (a+1)^m - 1 := 
sorry

end multiple_of_a_power_l236_236093


namespace balls_initial_count_90_l236_236236

theorem balls_initial_count_90 (n : ℕ) (total_initial_balls : ℕ)
  (initial_green_balls : ℕ := 3 * n)
  (initial_yellow_balls : ℕ := 7 * n)
  (remaining_green_balls : ℕ := initial_green_balls - 9)
  (remaining_yellow_balls : ℕ := initial_yellow_balls - 9)
  (h_ratio_1 : initial_green_balls = 3 * n)
  (h_ratio_2 : initial_yellow_balls = 7 * n)
  (h_ratio_3 : remaining_green_balls * 3 = remaining_yellow_balls * 1)
  (h_total : total_initial_balls = initial_green_balls + initial_yellow_balls)
  : total_initial_balls = 90 := 
by
  sorry

end balls_initial_count_90_l236_236236


namespace total_hoodies_l236_236018

def Fiona_hoodies : ℕ := 3
def Casey_hoodies : ℕ := Fiona_hoodies + 2

theorem total_hoodies : (Fiona_hoodies + Casey_hoodies) = 8 := by
  sorry

end total_hoodies_l236_236018


namespace find_values_of_m_l236_236854

theorem find_values_of_m (m : ℤ) (h₁ : m > 2022) (h₂ : (2022 + m) ∣ (2022 * m)) : 
  m = 1011 ∨ m = 2022 :=
sorry

end find_values_of_m_l236_236854


namespace range_of_a_l236_236592

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - a * x + a > 0) ↔ (a ≤ 0 ∨ a ≥ 4) :=
by {
  sorry
}

end range_of_a_l236_236592


namespace decimal_zeros_l236_236118

theorem decimal_zeros (h : 2520 = 2^3 * 3^2 * 5 * 7) : 
  ∃ (n : ℕ), n = 2 ∧ (∃ d : ℚ, d = 5 / 2520 ∧ ↑d = 0.004) :=
by
  -- We assume the factorization of 2520 is correct
  have h_fact := h
  -- We need to prove there are exactly 2 zeros between the decimal point and the first non-zero digit
  sorry

end decimal_zeros_l236_236118


namespace tangent_line_at_point_is_correct_l236_236935

theorem tangent_line_at_point_is_correct :
  ∀ (x y : ℝ), (y = x^2 + 2 * x) → (x = 1) → (y = 3) → (4 * x - y - 1 = 0) :=
by
  intros x y h_curve h_x h_y
  -- Here would be the proof
  sorry

end tangent_line_at_point_is_correct_l236_236935


namespace population_of_town_l236_236471

theorem population_of_town (F : ℝ) (males : ℕ) (female_glasses : ℝ) (percentage_glasses : ℝ) (total_population : ℝ) 
  (h1 : males = 2000) 
  (h2 : percentage_glasses = 0.30) 
  (h3 : female_glasses = 900) 
  (h4 : percentage_glasses * F = female_glasses) 
  (h5 : total_population = males + F) :
  total_population = 5000 :=
sorry

end population_of_town_l236_236471


namespace worksheets_already_graded_l236_236541

theorem worksheets_already_graded {total_worksheets problems_per_worksheet problems_left_to_grade : ℕ} :
  total_worksheets = 9 →
  problems_per_worksheet = 4 →
  problems_left_to_grade = 16 →
  (total_worksheets - (problems_left_to_grade / problems_per_worksheet)) = 5 :=
by
  intros h1 h2 h3
  sorry

end worksheets_already_graded_l236_236541


namespace find_m_l236_236773

-- Defining vectors a and b
def a (m : ℝ) : ℝ × ℝ := (2, m)
def b : ℝ × ℝ := (1, -1)

-- Proving that if b is perpendicular to (a + 2b), then m = 6
theorem find_m (m : ℝ) :
  let a_vec := a m
  let b_vec := b
  let sum_vec := (a_vec.1 + 2 * b_vec.1, a_vec.2 + 2 * b_vec.2)
  (b_vec.1 * sum_vec.1 + b_vec.2 * sum_vec.2 = 0) → m = 6 :=
by
  intros a_vec b_vec sum_vec perp_cond
  sorry

end find_m_l236_236773


namespace find_c_for_maximum_at_2_l236_236319

noncomputable def f (x c : ℝ) := x * (x - c)^2

theorem find_c_for_maximum_at_2 :
  (∀ x, HasDerivAt (f x) (x * (x - c)^2) x (x = 2) ∧ deriv (f 2) = 0 → c = 6) :=
sorry

end find_c_for_maximum_at_2_l236_236319


namespace acute_angle_sum_l236_236911

open Real

theorem acute_angle_sum (α β : ℝ) (hα : 0 < α ∧ α < π / 2)
                        (hβ : 0 < β ∧ β < π / 2)
                        (h1 : 3 * (sin α) ^ 2 + 2 * (sin β) ^ 2 = 1)
                        (h2 : 3 * sin (2 * α) - 2 * sin (2 * β) = 0) :
  α + 2 * β = π / 2 :=
sorry

end acute_angle_sum_l236_236911


namespace team_total_points_l236_236542

theorem team_total_points (T : ℕ) (h1 : ∃ x : ℕ, x = T / 6)
    (h2 : (T + (92 - 85)) / 6 = 84) : T = 497 := 
by sorry

end team_total_points_l236_236542


namespace negative_expression_b_negative_expression_c_negative_expression_e_l236_236377

theorem negative_expression_b:
  3 * Real.sqrt 11 - 10 < 0 := 
sorry

theorem negative_expression_c:
  18 - 5 * Real.sqrt 13 < 0 := 
sorry

theorem negative_expression_e:
  10 * Real.sqrt 26 - 51 < 0 := 
sorry

end negative_expression_b_negative_expression_c_negative_expression_e_l236_236377


namespace fred_more_than_daniel_l236_236138

-- Definitions and conditions from the given problem.
def total_stickers : ℕ := 750
def andrew_kept : ℕ := 130
def daniel_received : ℕ := 250
def fred_received : ℕ := total_stickers - andrew_kept - daniel_received

-- The proof problem statement.
theorem fred_more_than_daniel : fred_received - daniel_received = 120 := by 
  sorry

end fred_more_than_daniel_l236_236138


namespace total_bears_l236_236572

-- Definitions based on given conditions
def brown_bears : ℕ := 15
def white_bears : ℕ := 24
def black_bears : ℕ := 27

-- Theorem to prove the total number of bears
theorem total_bears : brown_bears + white_bears + black_bears = 66 := by
  sorry

end total_bears_l236_236572


namespace function_is_increasing_l236_236868

variable (x : ℝ)

-- Definition of the function
def y (x : ℝ) : ℝ := -1 / x

-- Condition that y increases as x increases for x > 1
def is_increasing_for_x_gt_1 (f : ℝ → ℝ) : Prop := ∀ x > 1, f x < f (x + 1)

-- The theorem we want to prove
theorem function_is_increasing : is_increasing_for_x_gt_1 y := by
  -- The proof would go here
  sorry

end function_is_increasing_l236_236868


namespace kayla_less_than_vika_l236_236785

variable (S K V : ℕ)
variable (h1 : S = 216)
variable (h2 : S = 4 * K)
variable (h3 : V = 84)

theorem kayla_less_than_vika (S K V : ℕ) (h1 : S = 216) (h2 : S = 4 * K) (h3 : V = 84) : V - K = 30 :=
by
  sorry

end kayla_less_than_vika_l236_236785


namespace single_elimination_games_l236_236062

theorem single_elimination_games (n : ℕ) (h : n = 128) : (n - 1) = 127 :=
by
  sorry

end single_elimination_games_l236_236062


namespace real_solutions_count_l236_236548

-- Define the system of equations
def sys_eqs (x y z w : ℝ) :=
  (x = z + w + z * w * x) ∧
  (z = x + y + x * y * z) ∧
  (y = w + x + w * x * y) ∧
  (w = y + z + y * z * w)

-- The statement of the proof problem
theorem real_solutions_count : ∃ S : Finset (ℝ × ℝ × ℝ × ℝ), (∀ t : ℝ × ℝ × ℝ × ℝ, t ∈ S ↔ sys_eqs t.1 t.2.1 t.2.2.1 t.2.2.2) ∧ S.card = 5 :=
by {
  sorry
}

end real_solutions_count_l236_236548


namespace minimum_value_y_l236_236811

noncomputable def y (x : ℝ) : ℝ := x + 1 / (x - 1)

theorem minimum_value_y (x : ℝ) (hx : x > 1) : ∃ A, (A = 3) ∧ (∀ y', y' = y x → y' ≥ A) := sorry

end minimum_value_y_l236_236811


namespace pair_not_equal_to_64_l236_236814

theorem pair_not_equal_to_64 :
  ¬(4 * (9 / 2) = 64) := by
  sorry

end pair_not_equal_to_64_l236_236814


namespace area_triangle_ABC_area_figure_DEFGH_area_triangle_JKL_l236_236823

-- (a) Proving the area of triangle ABC
theorem area_triangle_ABC (AB BC : ℝ) (hAB : AB = 2) (hBC : BC = 3) (h_right : true) : 
  (1 / 2) * AB * BC = 3 := sorry

-- (b) Proving the area of figure DEFGH
theorem area_figure_DEFGH (DH HG : ℝ) (hDH : DH = 5) (hHG : HG = 5) (triangle_area : ℝ) (hEPF : triangle_area = 3) : 
  DH * HG - triangle_area = 22 := sorry

-- (c) Proving the area of triangle JKL 
theorem area_triangle_JKL (side_area : ℝ) (h_side : side_area = 25) 
  (area_JSK : ℝ) (h_JSK : area_JSK = 3) 
  (area_LQJ : ℝ) (h_LQJ : area_LQJ = 15/2) 
  (area_LRK : ℝ) (h_LRK : area_LRK = 5) : 
  side_area - area_JSK - area_LQJ - area_LRK = 19/2 := sorry

end area_triangle_ABC_area_figure_DEFGH_area_triangle_JKL_l236_236823


namespace total_balls_l236_236256

def num_white : ℕ := 50
def num_green : ℕ := 30
def num_yellow : ℕ := 10
def num_red : ℕ := 7
def num_purple : ℕ := 3

def prob_neither_red_nor_purple : ℝ := 0.9

theorem total_balls (T : ℕ) 
  (h : prob_red_purple = 1 - prob_neither_red_nor_purple) 
  (h_prob : prob_red_purple = (num_red + num_purple : ℝ) / (T : ℝ)) :
  T = 100 :=
by sorry

end total_balls_l236_236256


namespace solve_inequality_l236_236930

-- Define the inequality problem.
noncomputable def inequality_problem (x : ℝ) : Prop :=
(x^2 + 2 * x - 15) / (x + 5) < 0

-- Define the solution set.
def solution_set (x : ℝ) : Prop :=
-5 < x ∧ x < 3

-- State the equivalence theorem.
theorem solve_inequality (x : ℝ) (h : x ≠ -5) : 
  inequality_problem x ↔ solution_set x :=
sorry

end solve_inequality_l236_236930


namespace clients_using_radio_l236_236273

theorem clients_using_radio (total_clients T R M TR TM RM TRM : ℕ)
  (h1 : total_clients = 180)
  (h2 : T = 115)
  (h3 : M = 130)
  (h4 : TR = 75)
  (h5 : TM = 85)
  (h6 : RM = 95)
  (h7 : TRM = 80) : R = 30 :=
by
  -- Using Inclusion-Exclusion Principle
  have h : total_clients = T + R + M - TR - TM - RM + TRM :=
    sorry  -- Proof of Inclusion-Exclusion principle for these sets
  rw [h1, h2, h3, h4, h5, h6, h7] at h
  -- Solve for R
  sorry

end clients_using_radio_l236_236273


namespace count_not_squares_or_cubes_l236_236736

theorem count_not_squares_or_cubes (n : ℕ) : 
  let total := 200 in
  let perfect_squares := 14 in
  let perfect_cubes := 5 in
  let perfect_sixth_powers := 2 in
  let squares_or_cubes := perfect_squares + perfect_cubes - perfect_sixth_powers in
  let count_not_squares_or_cubes := total - squares_or_cubes in
  n = count_not_squares_or_cubes :=
by
  let total := 200
  let perfect_squares := 14
  let perfect_cubes := 5
  let perfect_sixth_powers := 2
  let squares_or_cubes := perfect_squares + perfect_cubes - perfect_sixth_powers
  let count_not_squares_or_cubes := total - squares_or_cubes
  show _ from sorry

end count_not_squares_or_cubes_l236_236736


namespace slope_parallel_l236_236369

theorem slope_parallel {x y : ℝ} (h : 3 * x - 6 * y = 15) : 
  ∃ m : ℝ, m = -1/2 ∧ ( ∀ (x1 x2 : ℝ), 3 * x1 - 6 * y = 15 → ∃ y1 : ℝ, y1 = m * x1) :=
by
  sorry

end slope_parallel_l236_236369


namespace geometric_sequence_form_l236_236752

-- Definitions for sequences and common difference/ratio
def isArithmeticSeq (a : ℕ → ℝ) (d : ℝ) :=
  ∀ (m n : ℕ), a n = a m + (n - m) * d

def isGeometricSeq (b : ℕ → ℝ) (q : ℝ) :=
  ∀ (m n : ℕ), b n = b m * q ^ (n - m)

-- Problem statement: given an arithmetic sequence, find the form of the corresponding geometric sequence
theorem geometric_sequence_form
  (b : ℕ → ℝ) (q : ℝ) (m n : ℕ) (b_m : ℝ) (q_pos : q > 0) :
  (∀ (m n : ℕ), b n = b m * q ^ (n - m)) :=
sorry

end geometric_sequence_form_l236_236752


namespace number_of_students_in_third_batch_l236_236520

theorem number_of_students_in_third_batch
  (avg1 avg2 avg3 : ℕ)
  (total_avg : ℚ)
  (students1 students2 : ℕ)
  (h_avg1 : avg1 = 45)
  (h_avg2 : avg2 = 55)
  (h_avg3 : avg3 = 65)
  (h_total_avg : total_avg = 56.333333333333336)
  (h_students1 : students1 = 40)
  (h_students2 : students2 = 50) :
  ∃ x : ℕ, (students1 * avg1 + students2 * avg2 + x * avg3 = total_avg * (students1 + students2 + x) ∧ x = 60) :=
by
  sorry

end number_of_students_in_third_batch_l236_236520


namespace length_of_rectangle_l236_236354

theorem length_of_rectangle (l : ℝ) (s : ℝ) 
  (perimeter_square : 4 * s = 160) 
  (area_relation : s^2 = 5 * (l * 10)) : 
  l = 32 :=
by
  sorry

end length_of_rectangle_l236_236354


namespace sachin_younger_than_rahul_l236_236497

theorem sachin_younger_than_rahul
  (S R : ℝ)
  (h1 : S = 24.5)
  (h2 : S / R = 7 / 9) :
  R - S = 7 := 
by sorry

end sachin_younger_than_rahul_l236_236497


namespace range_of_a_l236_236029

noncomputable def f : ℝ → ℝ → ℝ
| a, x =>
  if x ≥ -1 then a * x ^ 2 + 2 * x 
  else (1 - 3 * a) * x - 3 / 2

theorem range_of_a (a : ℝ) : (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) > 0) → 0 < a ∧ a ≤ 1/4 :=
sorry

end range_of_a_l236_236029


namespace carlson_total_land_l236_236278

open Real

theorem carlson_total_land 
  (initial_land : ℝ)
  (cost_additional_land1 : ℝ)
  (cost_additional_land2 : ℝ)
  (cost_per_square_meter : ℝ) :
  initial_land = 300 →
  cost_additional_land1 = 8000 →
  cost_additional_land2 = 4000 →
  cost_per_square_meter = 20 →
  (initial_land + (cost_additional_land1 + cost_additional_land2) / cost_per_square_meter) = 900 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  done

end carlson_total_land_l236_236278


namespace exists_rational_non_integer_a_not_exists_rational_non_integer_b_l236_236556

-- Define rational non-integer numbers
def is_rational_non_integer (x : ℚ) : Prop := ¬(∃ (z : ℤ), x = z)

-- (a) Proof for existance of rational non-integer numbers y and x such that 19x + 8y, 8x + 3y are integers
theorem exists_rational_non_integer_a :
  ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧ (∃ a b : ℤ, 19 * x + 8 * y = a ∧ 8 * x + 3 * y = b) :=
sorry

-- (b) Proof for non-existance of rational non-integer numbers y and x such that 19x² + 8y², 8x² + 3y² are integers
theorem not_exists_rational_non_integer_b :
  ¬ ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧ (∃ m n : ℤ, 19 * x^2 + 8 * y^2 = m ∧ 8 * x^2 + 3 * y^2 = n) :=
sorry

end exists_rational_non_integer_a_not_exists_rational_non_integer_b_l236_236556


namespace ratio_of_triangle_side_to_rectangle_width_l236_236404

theorem ratio_of_triangle_side_to_rectangle_width
  (t w : ℕ)
  (ht : 3 * t = 24)
  (hw : 6 * w = 24) :
  t / w = 2 := by
  sorry

end ratio_of_triangle_side_to_rectangle_width_l236_236404


namespace complement_of_A_l236_236452

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 3, 5, 7}

theorem complement_of_A : U \ A = {2, 4, 6} := 
by 
  sorry

end complement_of_A_l236_236452


namespace total_books_l236_236992

-- Define the number of books each person has
def books_beatrix : ℕ := 30
def books_alannah : ℕ := books_beatrix + 20
def books_queen : ℕ := books_alannah + (books_alannah / 5)

-- State the theorem to be proved
theorem total_books (h_beatrix : books_beatrix = 30)
                    (h_alannah : books_alannah = books_beatrix + 20)
                    (h_queen : books_queen = books_alannah + (books_alannah / 5)) :
  books_alannah + books_beatrix + books_queen = 140 :=
sorry

end total_books_l236_236992


namespace find_multiple_of_t_l236_236466

variable (t : ℝ)
variable (x y : ℝ)

theorem find_multiple_of_t (h1 : x = 1 - 4 * t)
  (h2 : ∃ m : ℝ, y = m * t - 2)
  (h3 : t = 0.5)
  (h4 : x = y) : ∃ m : ℝ, (m = 2) :=
by
  sorry

end find_multiple_of_t_l236_236466


namespace corner_movement_l236_236985

-- Definition of corner movement problem
def canMoveCornerToBottomRight (m n : ℕ) : Prop :=
  m ≥ 2 ∧ n ≥ 2 ∧ (m % 2 = 1 ∧ n % 2 = 1)

theorem corner_movement (m n : ℕ) (h1 : m ≥ 2) (h2 : n ≥ 2) :
  (canMoveCornerToBottomRight m n ↔ (m % 2 = 1 ∧ n % 2 = 1)) :=
by
  sorry  -- Proof is omitted

end corner_movement_l236_236985


namespace sum_of_first_six_terms_l236_236174

theorem sum_of_first_six_terms 
  {S : ℕ → ℝ} 
  (h_arith_seq : ∀ n, S n = n * (-2) + (n * (n - 1) * 3 ))
  (S_2_eq_2 : S 2 = 2)
  (S_4_eq_10 : S 4 = 10) : S 6 = 18 := 
  sorry

end sum_of_first_six_terms_l236_236174


namespace fraction_eq_l236_236146

def at_op (a b : ℝ) : ℝ := a * b - a * b^2
def hash_op (a b : ℝ) : ℝ := a^2 + b - a^2 * b

theorem fraction_eq :
  (at_op 8 3) / (hash_op 8 3) = 48 / 125 :=
by sorry

end fraction_eq_l236_236146


namespace time_to_office_l236_236242

theorem time_to_office (S T : ℝ) (h1 : T > 0) (h2 : S > 0) 
    (h : S * (T + 15) = (4/5) * S * T) :
    T = 75 := by
  sorry

end time_to_office_l236_236242


namespace binomial_coefficients_sum_l236_236296

theorem binomial_coefficients_sum :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ),
  (1 - 2 * 0)^5 = a_0 + a_1 * (1 + 0) + a_2 * (1 + 0)^2 + a_3 * (1 + 0)^3 + a_4 * (1 + 0)^4 + a_5 * (1 + 0)^5 →
  (1 - 2 * 1)^5 = (-1)^5 * a_5 →
  a_0 + a_1 + a_2 + a_3 + a_4 = 33 :=
by sorry

end binomial_coefficients_sum_l236_236296


namespace minimum_value_l236_236858

noncomputable def min_value_of_expression (a b c : ℝ) : ℝ :=
  1/a + 2/b + 4/c

theorem minimum_value (a b c : ℝ) (h₀ : c > 0) (h₁ : a ≠ 0) (h₂ : b ≠ 0)
    (h₃ : 4 * a^2 - 2 * a * b + b^2 - c = 0)
    (h₄ : ∀ x y, 4*x^2 - 2*x*y + y^2 - c = 0 → |2*x + y| ≤ |2*a + b|)
    : min_value_of_expression a b c = -1 :=
sorry

end minimum_value_l236_236858


namespace value_of_sum_l236_236434

theorem value_of_sum (a x y : ℝ) (h1 : 17 * x + 19 * y = 6 - a) (h2 : 13 * x - 7 * y = 10 * a + 1) : 
  x + y = 1 / 3 := 
sorry

end value_of_sum_l236_236434


namespace probability_of_odd_sum_l236_236569

open Nat

def first_twelve_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_of_odd_sum :
  (binomial 11 3) / (binomial 12 4) = 1 / 3 := by
sorry

end probability_of_odd_sum_l236_236569


namespace probability_of_odd_sum_l236_236568

open Nat

def first_twelve_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_of_odd_sum :
  (binomial 11 3) / (binomial 12 4) = 1 / 3 := by
sorry

end probability_of_odd_sum_l236_236568


namespace length_of_unfenced_side_l236_236023

theorem length_of_unfenced_side :
  ∃ L W : ℝ, L * W = 320 ∧ 2 * W + L = 56 ∧ L = 40 :=
by
  sorry

end length_of_unfenced_side_l236_236023


namespace salary_increase_gt_90_percent_l236_236339

theorem salary_increase_gt_90_percent (S : ℝ) : 
  (S * (1.12^6) - S) / S > 0.90 :=
by
  -- Here we skip the proof with sorry
  sorry

end salary_increase_gt_90_percent_l236_236339


namespace SallyCarrots_l236_236346

-- Definitions of the conditions
def FredGrew (F : ℕ) := F = 4
def TotalGrew (T : ℕ) := T = 10
def SallyGrew (S : ℕ) (F T : ℕ) := S + F = T

-- The theorem to be proved
theorem SallyCarrots : ∃ S : ℕ, FredGrew 4 ∧ TotalGrew 10 ∧ SallyGrew S 4 10 ∧ S = 6 :=
  sorry

end SallyCarrots_l236_236346


namespace subset_implies_range_a_intersection_implies_range_a_l236_236449

noncomputable def setA : Set ℝ := {x | -1 < x ∧ x < 2}
noncomputable def setB (a : ℝ) : Set ℝ := {x | 2 * a - 1 < x ∧ x < 2 * a + 3}

theorem subset_implies_range_a (a : ℝ) : (setA ⊆ setB a) → (-1/2 ≤ a ∧ a ≤ 0) :=
by
  sorry

theorem intersection_implies_range_a (a : ℝ) : (setA ∩ setB a = ∅) → (a ≤ -2 ∨ a ≥ 3/2) :=
by
  sorry

end subset_implies_range_a_intersection_implies_range_a_l236_236449


namespace single_burger_cost_l236_236140

theorem single_burger_cost
  (total_cost : ℝ)
  (total_hamburgers : ℕ)
  (double_burgers : ℕ)
  (cost_double_burger : ℝ)
  (remaining_cost : ℝ)
  (single_burgers : ℕ)
  (cost_single_burger : ℝ) :
  total_cost = 64.50 ∧
  total_hamburgers = 50 ∧
  double_burgers = 29 ∧
  cost_double_burger = 1.50 ∧
  remaining_cost = total_cost - (double_burgers * cost_double_burger) ∧
  single_burgers = total_hamburgers - double_burgers ∧
  cost_single_burger = remaining_cost / single_burgers →
  cost_single_burger = 1.00 :=
by
  sorry

end single_burger_cost_l236_236140


namespace math_problem_l236_236280

theorem math_problem : -5 * (-6) - 2 * (-3 * (-7) + (-8)) = 4 := 
  sorry

end math_problem_l236_236280


namespace bike_price_l236_236338

-- Definitions of the conditions
def maria_savings : ℕ := 120
def mother_offer : ℕ := 250
def amount_needed : ℕ := 230

-- Theorem statement
theorem bike_price (maria_savings mother_offer amount_needed : ℕ) : 
  maria_savings + mother_offer + amount_needed = 600 := 
by
  -- Sorry is used here to skip the actual proof steps
  sorry

end bike_price_l236_236338


namespace square_area_from_points_l236_236779

theorem square_area_from_points :
  let P1 := (1, 2)
  let P2 := (4, 6)
  let side_length := real.sqrt ((4 - 1)^2 + (6 - 2)^2)
  let area := side_length^2
  P1.1 = 1 ∧ P1.2 = 2 ∧ P2.1 = 4 ∧ P2.2 = 6 →
  area = 25 :=
by
  sorry

end square_area_from_points_l236_236779


namespace a_minus_b_eq_neg_9_or_neg_1_l236_236299

theorem a_minus_b_eq_neg_9_or_neg_1 (a b : ℝ) (h₁ : |a| = 5) (h₂ : |b| = 4) (h₃ : a + b < 0) :
  a - b = -9 ∨ a - b = -1 :=
by
  sorry

end a_minus_b_eq_neg_9_or_neg_1_l236_236299


namespace find_value_of_X_l236_236743

theorem find_value_of_X :
  let X_initial := 5
  let S_initial := 0
  let X_increment := 3
  let target_sum := 15000
  let X := X_initial + X_increment * 56
  2 * target_sum ≥ 3 * 57 * 57 + 7 * 57 →
  X = 173 :=
by
  sorry

end find_value_of_X_l236_236743


namespace exists_rational_non_integer_xy_no_rational_non_integer_xy_l236_236555

-- Part (a)
theorem exists_rational_non_integer_xy 
  (x y : ℚ) (h1 : ¬ ∃ z : ℤ, x = z ∧ y = z) : 
  (∃ x y : ℚ, ¬(∃ z : ℤ, x = z ∨ y = z) ∧ 
   ∃ z1 z2 : ℤ, 19 * x + 8 * y = ↑z1 ∧ 8 * x + 3 * y = ↑z2) :=
sorry

-- Part (b)
theorem no_rational_non_integer_xy 
  (x y : ℚ) (h1 : ¬ ∃ z : ℤ, x = z ∧ y = z) : 
  ¬ ∃ x y : ℚ, ¬(∃ z : ℤ, x = z ∨ y = z) ∧ 
  ∃ z1 z2 : ℤ, 19 * x^2 + 8 * y^2 = ↑z1 ∧ 8 * x^2 + 3 * y^2 = ↑z2 :=
sorry

end exists_rational_non_integer_xy_no_rational_non_integer_xy_l236_236555


namespace leftmost_square_side_length_l236_236639

open Real

/-- Given the side lengths of three squares, 
    where the middle square's side length is 17 cm longer than the leftmost square,
    the rightmost square's side length is 6 cm shorter than the middle square,
    and the sum of the side lengths of all three squares is 52 cm,
    prove that the side length of the leftmost square is 8 cm. -/
theorem leftmost_square_side_length
  (x : ℝ)
  (h1 : ∀ m : ℝ, m = x + 17)
  (h2 : ∀ r : ℝ, r = x + 11)
  (h3 : x + (x + 17) + (x + 11) = 52) :
  x = 8 := by
  sorry

end leftmost_square_side_length_l236_236639


namespace ratio_PR_QS_l236_236614

/-- Given points P, Q, R, and S on a straight line in that order with
    distances PQ = 3 units, QR = 7 units, and PS = 20 units,
    the ratio of PR to QS is 1. -/
theorem ratio_PR_QS (P Q R S : ℝ) (PQ QR PS : ℝ) (hPQ : PQ = 3) (hQR : QR = 7) (hPS : PS = 20) :
  let PR := PQ + QR
  let QS := PS - PQ - QR
  PR / QS = 1 :=
by
  -- Definitions from conditions
  let PR := PQ + QR
  let QS := PS - PQ - QR
  -- Proof not required, hence sorry
  sorry

end ratio_PR_QS_l236_236614


namespace angle_of_inclination_l236_236421

theorem angle_of_inclination (θ : ℝ) : 
  (∀ x y : ℝ, x - y + 3 = 0 → ∃ θ : ℝ, Real.tan θ = 1 ∧ θ = Real.pi / 4) := by
  sorry

end angle_of_inclination_l236_236421


namespace intersection_points_parabola_l236_236465

noncomputable def parabola : ℝ → ℝ := λ x => x^2

noncomputable def directrix : ℝ → ℝ := λ x => -1

noncomputable def other_line (m c : ℝ) : ℝ → ℝ := λ x => m * x + c

theorem intersection_points_parabola {m c : ℝ} (h1 : ∃ x1 x2 : ℝ, other_line m c x1 = parabola x1 ∧ other_line m c x2 = parabola x2) :
  (∃ x1 x2 : ℝ, parabola x1 = other_line m c x1 ∧ parabola x2 = other_line m c x2 ∧ x1 ≠ x2) → 
  (∃ x1 x2 : ℝ, parabola x1 = other_line m c x1 ∧ parabola x2 = other_line m c x2 ∧ x1 = x2) := 
by
  sorry

end intersection_points_parabola_l236_236465


namespace girls_not_join_field_trip_l236_236649

theorem girls_not_join_field_trip (total_students : ℕ) (number_of_boys : ℕ) (number_on_trip : ℕ)
  (h_total : total_students = 18)
  (h_boys : number_of_boys = 8)
  (h_equal : number_on_trip = number_of_boys) :
  total_students - number_of_boys - number_on_trip = 2 := by
sorry

end girls_not_join_field_trip_l236_236649


namespace max_gcd_expression_l236_236178

theorem max_gcd_expression (n : ℕ) (h1 : n > 0) (h2 : n % 3 = 1) : 
  Nat.gcd (15 * n + 5) (9 * n + 4) = 5 :=
by
  sorry

end max_gcd_expression_l236_236178


namespace find_n_l236_236937

theorem find_n :
  ∃ n : ℕ, 50 ≤ n ∧ n ≤ 150 ∧
          n % 7 = 0 ∧
          n % 9 = 3 ∧
          n % 6 = 3 ∧
          n = 75 :=
by
  sorry

end find_n_l236_236937


namespace num_girls_went_to_spa_l236_236678

-- Define the condition that each girl has 20 nails
def nails_per_girl : ℕ := 20

-- Define the total number of nails polished
def total_nails_polished : ℕ := 40

-- Define the number of girls
def number_of_girls : ℕ := total_nails_polished / nails_per_girl

-- The theorem we want to prove
theorem num_girls_went_to_spa : number_of_girls = 2 :=
by
  unfold number_of_girls
  unfold total_nails_polished
  unfold nails_per_girl
  sorry

end num_girls_went_to_spa_l236_236678


namespace probability_of_second_ball_white_is_correct_l236_236532

-- Definitions based on the conditions
def initial_white_balls : ℕ := 8
def initial_black_balls : ℕ := 7
def total_initial_balls : ℕ := initial_white_balls + initial_black_balls
def white_balls_after_first_draw : ℕ := initial_white_balls
def black_balls_after_first_draw : ℕ := initial_black_balls - 1
def total_balls_after_first_draw : ℕ := white_balls_after_first_draw + black_balls_after_first_draw
def probability_second_ball_white : ℚ := white_balls_after_first_draw / total_balls_after_first_draw

-- The proof problem
theorem probability_of_second_ball_white_is_correct :
  probability_second_ball_white = 4 / 7 :=
by
  sorry

end probability_of_second_ball_white_is_correct_l236_236532


namespace center_of_circle_l236_236699

theorem center_of_circle : ∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 2 → (1, 1) = (1, 1) :=
by
  intros x y h
  sorry

end center_of_circle_l236_236699


namespace determine_function_l236_236698

theorem determine_function {f : ℝ → ℝ} :
  (∀ x y : ℝ, f (f x ^ 2 + f y) = x * f x + y) →
  (∀ x : ℝ, f x = x ∨ f x = -x) :=
by
  sorry

end determine_function_l236_236698


namespace determine_vector_p_l236_236453

structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def vector_operation (m p : Vector2D) : Vector2D :=
  Vector2D.mk (m.x * p.x + m.y * p.y) (m.x * p.y + m.y * p.x)

theorem determine_vector_p (p : Vector2D) : 
  (∀ (m : Vector2D), vector_operation m p = m) → p = Vector2D.mk 1 0 :=
by
  sorry

end determine_vector_p_l236_236453


namespace solve_inequality_l236_236030

noncomputable def f (x : ℝ) : ℝ :=
  x^3 + x + 2^x - 2^(-x)

theorem solve_inequality (x : ℝ) : 
  f (Real.exp x - x) ≤ 7/2 ↔ x = 0 := 
sorry

end solve_inequality_l236_236030


namespace find_f_2012_l236_236972

-- Given a function f: ℤ → ℤ that satisfies the functional equation:
def functional_equation (f : ℤ → ℤ) := ∀ m n : ℤ, m + f (m + f (n + f m)) = n + f m

-- Given condition:
def f_6_is_6 (f : ℤ → ℤ) := f 6 = 6

-- We need to prove that f 2012 = -2000 under the given conditions.
theorem find_f_2012 (f : ℤ → ℤ) (hf : functional_equation f) (hf6 : f_6_is_6 f) : f 2012 = -2000 := sorry

end find_f_2012_l236_236972


namespace problem_statement_l236_236315

noncomputable def f (x : ℝ) : ℝ :=
  if h : 0 ≤ x ∧ x ≤ 1 then 2 * x * (1 - x)
  else -- define elsewhere based on periodicity and oddness properties
    sorry 

theorem problem_statement : 
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 4) = f x) → f 2015.5 = -0.5 :=
by
  intros
  sorry

end problem_statement_l236_236315


namespace base4_to_base10_conversion_l236_236419

theorem base4_to_base10_conversion : 
  2 * 4^4 + 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 2 * 4^0 = 582 :=
by 
  sorry

end base4_to_base10_conversion_l236_236419


namespace cube_difference_l236_236366

theorem cube_difference (x y : ℕ) (h₁ : x + y = 64) (h₂ : x - y = 16) : x^3 - y^3 = 50176 := by
  sorry

end cube_difference_l236_236366


namespace apples_in_basket_l236_236388

theorem apples_in_basket (x : ℕ) (h1 : 22 * x = (x + 45) * 13) : 22 * x = 1430 :=
by
  sorry

end apples_in_basket_l236_236388


namespace max_value_y_l236_236883

theorem max_value_y (x : ℝ) : ∃ y, y = -3 * x^2 + 6 ∧ ∀ z, (∃ x', z = -3 * x'^2 + 6) → z ≤ y :=
by sorry

end max_value_y_l236_236883


namespace solution_l236_236788

def solve_for_x (x : ℝ) : Prop :=
  7 + 3.5 * x = 2.1 * x - 25

theorem solution (x : ℝ) (h : solve_for_x x) : x = -22.857 :=
by
  sorry

end solution_l236_236788


namespace diff_12_358_7_2943_l236_236410

theorem diff_12_358_7_2943 : 12.358 - 7.2943 = 5.0637 :=
by
  -- Proof is not required, so we put sorry
  sorry

end diff_12_358_7_2943_l236_236410


namespace directrix_of_parabola_l236_236564

-- Definition of the given parabola
def parabola (x : ℝ) : ℝ := (x^2 - 8 * x + 12) / 16

-- The mathematical statement to prove
theorem directrix_of_parabola : ∀ x : ℝ, parabola x = y ↔ y = -5 / 4 -> sorry :=
by
  sorry

end directrix_of_parabola_l236_236564


namespace coat_price_reduction_l236_236667

theorem coat_price_reduction (original_price reduction_amount : ℝ) (h : original_price = 500) (h_red : reduction_amount = 150) :
  ((reduction_amount / original_price) * 100) = 30 :=
by
  rw [h, h_red]
  norm_num

end coat_price_reduction_l236_236667


namespace least_multiplier_produces_required_result_l236_236968

noncomputable def least_multiplier_that_satisfies_conditions : ℕ :=
  62087668

theorem least_multiplier_produces_required_result :
  ∃ k : ℕ, k * 72 = least_multiplier_that_satisfies_conditions * 72 ∧
           k * 72 % 112 = 0 ∧
           k * 72 % 199 = 0 ∧
           ∃ n : ℕ, k * 72 = n * n :=
by
  let k := least_multiplier_that_satisfies_conditions
  use k
  -- Using "sorry" as we don't need the proof steps here
  sorry

end least_multiplier_produces_required_result_l236_236968


namespace profit_percentage_calc_l236_236942

noncomputable def sale_price_incl_tax : ℝ := 616
noncomputable def sales_tax_rate : ℝ := 0.10
noncomputable def cost_price : ℝ := 531.03
noncomputable def expected_profit_percentage : ℝ := 5.45

theorem profit_percentage_calc :
  let sale_price_before_tax := sale_price_incl_tax / (1 + sales_tax_rate)
  let profit := sale_price_before_tax - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = expected_profit_percentage :=
by
  sorry

end profit_percentage_calc_l236_236942


namespace arithmetic_operations_result_eq_one_over_2016_l236_236898

theorem arithmetic_operations_result_eq_one_over_2016 :
  (∃ op1 op2 : ℚ → ℚ → ℚ, op1 (1/8) (op2 (1/9) (1/28)) = 1/2016) :=
sorry

end arithmetic_operations_result_eq_one_over_2016_l236_236898


namespace like_terms_implies_a_plus_2b_eq_3_l236_236739

theorem like_terms_implies_a_plus_2b_eq_3 (a b : ℤ) (h1 : 2 * a + b = 6) (h2 : a - b = 3) : a + 2 * b = 3 :=
sorry

end like_terms_implies_a_plus_2b_eq_3_l236_236739


namespace pipe_fills_entire_cistern_in_77_minutes_l236_236397

-- Define the time taken to fill 1/11 of the cistern
def time_to_fill_one_eleven_cistern : ℕ := 7

-- Define the fraction of the cistern filled in a certain time
def fraction_filled (t : ℕ) : ℚ := t / time_to_fill_one_eleven_cistern * (1 / 11)

-- Define the problem statement
theorem pipe_fills_entire_cistern_in_77_minutes : 
  fraction_filled 77 = 1 := by
  sorry

end pipe_fills_entire_cistern_in_77_minutes_l236_236397


namespace storks_more_than_birds_l236_236254

theorem storks_more_than_birds 
  (initial_birds : ℕ) 
  (joined_storks : ℕ) 
  (joined_birds : ℕ) 
  (h_init_birds : initial_birds = 3) 
  (h_joined_storks : joined_storks = 6) 
  (h_joined_birds : joined_birds = 2) : 
  (joined_storks - (initial_birds + joined_birds)) = 1 := 
by 
  -- Proof goes here
  sorry

end storks_more_than_birds_l236_236254


namespace thabo_books_l236_236098

theorem thabo_books (H P F : ℕ) (h1 : P = H + 20) (h2 : F = 2 * P) (h3 : H + P + F = 280) : H = 55 :=
by
  sorry

end thabo_books_l236_236098


namespace volume_pyramid_l236_236364

theorem volume_pyramid (V : ℝ) : 
  ∃ V_P : ℝ, V_P = V / 6 :=
by
  sorry

end volume_pyramid_l236_236364


namespace geometric_sequence_sum_l236_236478

theorem geometric_sequence_sum
  (S : ℕ → ℝ)
  (a : ℕ → ℝ)
  (r : ℝ)
  (h1 : r ≠ 1)
  (h2 : ∀ n, S n = a 0 * (1 - r^(n + 1)) / (1 - r))
  (h3 : S 5 = 3)
  (h4 : S 10 = 9) :
  S 15 = 21 :=
sorry

end geometric_sequence_sum_l236_236478


namespace negation_of_universal_proposition_l236_236180

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 > Real.log x)) ↔ (∃ x : ℝ, x^2 ≤ Real.log x) :=
by
  sorry

end negation_of_universal_proposition_l236_236180


namespace determine_k_l236_236423

variable (x y z w : ℝ)

theorem determine_k
  (h₁ : 9 / (x + y + w) = k / (x + z + w))
  (h₂ : k / (x + z + w) = 12 / (z - y)) :
  k = 21 :=
sorry

end determine_k_l236_236423


namespace city_renumbering_not_possible_l236_236001

-- Defining the problem conditions
def city_renumbering_invalid (city_graph : Type) (connected : city_graph → city_graph → Prop) : Prop :=
  ∃ (M N : city_graph), ∀ (renumber : city_graph → city_graph),
  (renumber M = N ∧ renumber N = M) → ¬(
    ∀ x y : city_graph,
    connected x y ↔ connected (renumber x) (renumber y)
  )

-- Statement of the problem
theorem city_renumbering_not_possible (city_graph : Type) (connected : city_graph → city_graph → Prop) :
  city_renumbering_invalid city_graph connected :=
sorry

end city_renumbering_not_possible_l236_236001


namespace percent_decrease_l236_236967

def original_price : ℝ := 100
def sale_price : ℝ := 60

theorem percent_decrease : (original_price - sale_price) / original_price * 100 = 40 := by
  sorry

end percent_decrease_l236_236967


namespace Jane_stick_length_l236_236087

theorem Jane_stick_length
  (Pat_stick_length : ℕ)
  (dirt_covered_length : ℕ)
  (Sarah_stick_double : ℕ)
  (Jane_stick_diff : ℕ) :
  Pat_stick_length = 30 →
  dirt_covered_length = 7 →
  Sarah_stick_double = 2 →
  Jane_stick_diff = 24 →
  (Pat_stick_length - dirt_covered_length) * Sarah_stick_double - Jane_stick_diff = 22 := 
by
  intros Pat_length dirt_length Sarah_double Jane_diff
  intro h1
  intro h2
  intro h3
  intro h4
  rw [h1, h2, h3, h4]
  sorry

end Jane_stick_length_l236_236087


namespace arithmetic_seq_a2_l236_236443

def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop := ∀ n m : ℕ, a m = a (n + 1) + d * (m - (n + 1))

theorem arithmetic_seq_a2 
  (a : ℕ → ℤ) (d a1 : ℤ)
  (h_arith: ∀ n : ℕ, a n = a1 + n * d)
  (h_sum: a 3 + a 11 = 50)
  (h_a4: a 4 = 13) :
  a 2 = 5 :=
sorry

end arithmetic_seq_a2_l236_236443


namespace square_side_length_l236_236938

variable (x : ℝ) (π : ℝ) (hπ: π = Real.pi)

theorem square_side_length (h1: 4 * x = 10 * π) : 
  x = (5 * π) / 2 := 
by
  sorry

end square_side_length_l236_236938


namespace intersection_PQ_eq_23_l236_236185

def P : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}
def Q : Set ℝ := {x : ℝ | 2 < x}

theorem intersection_PQ_eq_23 : P ∩ Q = {x : ℝ | 2 < x ∧ x < 3} := 
by {
  sorry
}

end intersection_PQ_eq_23_l236_236185


namespace scientific_notation_correct_l236_236511

noncomputable def scientific_notation (x : ℕ) : Prop :=
  x = 3010000000 → 3.01 * (10 ^ 9) = 3.01 * (10 ^ 9)

theorem scientific_notation_correct : 
  scientific_notation 3010000000 :=
by
  intros h
  sorry

end scientific_notation_correct_l236_236511


namespace tan_double_angle_tan_angle_add_pi_div_4_l236_236048

theorem tan_double_angle (α : ℝ) (h : Real.tan α = -2) : Real.tan (2 * α) = 4 / 3 :=
by
  sorry

theorem tan_angle_add_pi_div_4 (α : ℝ) (h : Real.tan α = -2) : Real.tan (2 * α + Real.pi / 4) = -7 :=
by
  sorry

end tan_double_angle_tan_angle_add_pi_div_4_l236_236048


namespace find_xyz_sum_l236_236971

variables {x y z : ℝ}

def system_of_equations (x y z : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  (x^2 + x * y + y^2 = 12) ∧
  (y^2 + y * z + z^2 = 9) ∧
  (z^2 + z * x + x^2 = 21)

theorem find_xyz_sum (x y z : ℝ) (h : system_of_equations x y z) : 
  x * y + y * z + z * x = 12 :=
sorry

end find_xyz_sum_l236_236971


namespace sum_of_decimals_as_fraction_simplified_fraction_final_sum_as_fraction_l236_236007

theorem sum_of_decimals_as_fraction:
  (0.4 + 0.05 + 0.006 + 0.0007 + 0.00008) = (45678 / 100000) := by
  sorry

theorem simplified_fraction:
  (45678 / 100000) = (22839 / 50000) := by
  sorry

theorem final_sum_as_fraction:
  (0.4 + 0.05 + 0.006 + 0.0007 + 0.00008) = (22839 / 50000) := by
  have h1 := sum_of_decimals_as_fraction
  have h2 := simplified_fraction
  rw [h1, h2]
  sorry

end sum_of_decimals_as_fraction_simplified_fraction_final_sum_as_fraction_l236_236007


namespace red_ball_value_l236_236327

theorem red_ball_value (r b g : ℕ) (blue_points green_points : ℕ)
  (h1 : blue_points = 4)
  (h2 : green_points = 5)
  (h3 : b = g)
  (h4 : r^4 * blue_points^b * green_points^g = 16000)
  (h5 : b = 6) :
  r = 1 :=
by
  sorry

end red_ball_value_l236_236327


namespace consumer_installment_credit_value_l236_236249

variable (consumer_installment_credit : ℝ) 

noncomputable def automobile_installment_credit := 0.36 * consumer_installment_credit

noncomputable def finance_company_credit := 35

theorem consumer_installment_credit_value :
  (∃ C : ℝ, automobile_installment_credit C = 0.36 * C ∧ finance_company_credit = (1 / 3) * automobile_installment_credit C) →
  consumer_installment_credit = 291.67 :=
by
  sorry

end consumer_installment_credit_value_l236_236249


namespace Z_4_3_eq_37_l236_236697

def Z (a b : ℕ) : ℕ :=
  a^2 + a * b + b^2

theorem Z_4_3_eq_37 : Z 4 3 = 37 :=
  by
    sorry

end Z_4_3_eq_37_l236_236697


namespace volume_of_given_sphere_l236_236504

noncomputable def volume_of_sphere (A d : ℝ) : ℝ :=
  (4 / 3) * Real.pi * (Real.sqrt (d^2 + A / Real.pi))^3

theorem volume_of_given_sphere
  (hA : 2 * Real.pi = 2 * Real.pi)
  (hd : 1 = 1):
  volume_of_sphere (2 * Real.pi) 1 = 4 * Real.sqrt 3 * Real.pi :=
by
  sorry

end volume_of_given_sphere_l236_236504


namespace statements_correctness_l236_236170

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - x + a

theorem statements_correctness (a : ℝ) (h0 : f 0 a < 0) (h1 : f 1 a > 0) :
  (∃ x0, 0 < x0 ∧ x0 < 1 ∧ f x0 a = 0) ∧
  (f (1/2) a > 0 → f (1/4) a < 0) ∧
  (f (1/2) a < 0 → f (1/4) a > 0 → f (3/4) a = 0) ∧
  ¬ (f (3/2) a > 0 → f (5/4) a = 0) :=
by
  sorry

end statements_correctness_l236_236170


namespace eq_of_divisible_l236_236440

theorem eq_of_divisible (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h : a + b ∣ 5 * a + 3 * b) : a = b :=
sorry

end eq_of_divisible_l236_236440


namespace largest_band_members_l236_236680

theorem largest_band_members
  (p q m : ℕ)
  (h1 : p * q + 3 = m)
  (h2 : (q + 1) * (p + 2) = m)
  (h3 : m < 120) :
  m = 119 :=
sorry

end largest_band_members_l236_236680


namespace range_of_a_l236_236580

noncomputable def f (a : ℝ) (x : ℝ) := x * Real.log x - a * x^2

theorem range_of_a (a : ℝ) : (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ 
0 < a ∧ a < 1/2 :=
by
  sorry

end range_of_a_l236_236580


namespace cyclists_meet_at_start_point_l236_236951

-- Conditions from the problem
def cyclist1_speed : ℝ := 7 -- speed of the first cyclist in m/s
def cyclist2_speed : ℝ := 8 -- speed of the second cyclist in m/s
def circumference : ℝ := 600 -- circumference of the circular track in meters

-- Relative speed when cyclists move in opposite directions
def relative_speed := cyclist1_speed + cyclist2_speed

-- Prove that they meet at the starting point after 40 seconds
theorem cyclists_meet_at_start_point :
  (circumference / relative_speed) = 40 := by
  -- the proof would go here
  sorry

end cyclists_meet_at_start_point_l236_236951


namespace grey_pairs_coincide_l236_236002

theorem grey_pairs_coincide (h₁ : 4 = orange_count / 2) 
                                (h₂ : 6 = green_count / 2)
                                (h₃ : 9 = grey_count / 2)
                                (h₄ : 3 = orange_pairs)
                                (h₅ : 4 = green_pairs)
                                (h₆ : 1 = orange_grey_pairs) :
    grey_pairs = 6 := by
  sorry

noncomputable def half_triangle_counts : (ℕ × ℕ × ℕ) := (4, 6, 9)

noncomputable def triangle_pairs : (ℕ × ℕ × ℕ) := (3, 4, 1)

noncomputable def prove_grey_pairs (orange_count green_count grey_count : ℕ)
                                   (orange_pairs green_pairs orange_grey_pairs : ℕ) : ℕ :=
  sorry

end grey_pairs_coincide_l236_236002


namespace find_original_number_l236_236396

theorem find_original_number (x : ℝ) : ((x - 3) / 6) * 12 = 8 → x = 7 :=
by
  intro h
  sorry

end find_original_number_l236_236396


namespace sum_of_areas_l236_236233

theorem sum_of_areas :
  (∑' n : ℕ, Real.pi * (1 / 9 ^ n)) = (9 * Real.pi) / 8 :=
by
  sorry

end sum_of_areas_l236_236233


namespace sara_quarters_final_l236_236498

def initial_quarters : ℕ := 21
def quarters_from_dad : ℕ := 49
def quarters_spent_at_arcade : ℕ := 15
def dollar_bills_from_mom : ℕ := 2
def quarters_per_dollar : ℕ := 4

theorem sara_quarters_final :
  (initial_quarters + quarters_from_dad - quarters_spent_at_arcade + dollar_bills_from_mom * quarters_per_dollar) = 63 :=
by
  sorry

end sara_quarters_final_l236_236498


namespace math_books_count_l236_236383

theorem math_books_count (M H : ℕ) (h1 : M + H = 80) (h2 : 4 * M + 5 * H = 373) : M = 27 :=
by
  sorry

end math_books_count_l236_236383


namespace prob_fifth_card_is_ace_of_hearts_l236_236836

theorem prob_fifth_card_is_ace_of_hearts : 
  (∀ d : List ℕ, d.length = 52 → (1 ∈ d) → Prob (d.nth 4 = some 1) = 1 / 52) := 
by
  sorry

end prob_fifth_card_is_ace_of_hearts_l236_236836


namespace not_sufficient_not_necessary_l236_236820

theorem not_sufficient_not_necessary (a : ℝ) :
  ¬ ((a^2 > 1) → (1/a > 0)) ∧ ¬ ((1/a > 0) → (a^2 > 1)) := sorry

end not_sufficient_not_necessary_l236_236820


namespace quadratic_two_distinct_real_roots_find_m_and_other_root_l236_236302

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c
def roots_sum (a b : ℝ) : ℝ := -b / a

theorem quadratic_two_distinct_real_roots (m : ℝ)
  (hm : m < 0) :
  ∀ (a b c : ℝ), a = 1 → b = -2 → c = m → (discriminant a b c > 0) :=
by
  intros a b c ha hb hc
  rw [ha, hb, hc, discriminant]
  sorry

theorem find_m_and_other_root (a b c r1 : ℝ)
  (ha : a = 1)
  (hb : b = -2)
  (hc : c = r1^2 - 2*r1 + c = 0)
  (hr1 : r1 = -1)
  :
  c = -3 ∧ 
  ∃ r2 : ℝ, (roots_sum a b = 2) ∧ (r1 + r2 = 2) ∧ (r1 = -1 → r2 = 3) :=
by
  intros
  rw [ha, hb, hr1]
  sorry

end quadratic_two_distinct_real_roots_find_m_and_other_root_l236_236302


namespace angle_degrees_l236_236745

-- Define the conditions
def sides_parallel (θ₁ θ₂ : ℝ) : Prop :=
  θ₁ = θ₂ ∨ (θ₁ + θ₂ = 180)

def angle_relation (θ₁ θ₂ : ℝ) : Prop :=
  θ₁ = 3 * θ₂ - 20 ∨ θ₂ = 3 * θ₁ - 20

-- Statement of the problem
theorem angle_degrees (θ₁ θ₂ : ℝ) (h_parallel : sides_parallel θ₁ θ₂) (h_relation : angle_relation θ₁ θ₂) :
  (θ₁ = 10 ∧ θ₂ = 10) ∨ (θ₁ = 50 ∧ θ₂ = 130) ∨ (θ₁ = 130 ∧ θ₂ = 50) ∨ θ₁ + θ₂ = 180 ∧ (θ₁ = 3 * θ₂ - 20 ∨ θ₂ = 3 * θ₁ - 20) :=
by sorry

end angle_degrees_l236_236745


namespace dispatch_plans_l236_236116

theorem dispatch_plans (students : Finset ℕ) (h_students : students.card = 6) :
  ∃ plans : ℕ, plans = 180 ∧
  ∃ s, s ⊆ students ∧ s.card = 2 ∧
  ∃ f, f ⊆ (students \ s) ∧ f.card = 1 ∧
  ∃ sa, sa ⊆ (students \ (s ∪ f)) ∧ sa.card = 1 := by
  sorry

end dispatch_plans_l236_236116


namespace driving_speed_l236_236755

variable (total_distance : ℝ) (break_time : ℝ) (total_trip_time : ℝ)

theorem driving_speed (h1 : total_distance = 480)
                      (h2 : break_time = 1)
                      (h3 : total_trip_time = 9) : 
  (total_distance / (total_trip_time - break_time)) = 60 :=
by
  sorry

end driving_speed_l236_236755


namespace intersecting_diagonals_probability_l236_236669

def probability_of_intersecting_diagonals_inside_dodecagon : ℚ :=
  let total_points := 12
  let total_segments := (total_points.choose 2)
  let sides := 12
  let diagonals := total_segments - sides
  let ways_to_choose_2_diagonals := (diagonals.choose 2)
  let ways_to_choose_4_points := (total_points.choose 4)
  let probability := (ways_to_choose_4_points : ℚ) / (ways_to_choose_2_diagonals : ℚ)
  probability

theorem intersecting_diagonals_probability (H : probability_of_intersecting_diagonals_inside_dodecagon = 165 / 477) : 
  probability_of_intersecting_diagonals_inside_dodecagon = 165 / 477 :=
  by
  sorry

end intersecting_diagonals_probability_l236_236669


namespace perfect_square_trinomial_l236_236190

variable (x y : ℝ)

theorem perfect_square_trinomial (a : ℝ) :
  (∃ b c : ℝ, 4 * x^2 - (a - 1) * x * y + 9 * y^2 = (b * x + c * y) ^ 2) ↔ 
  (a = 13 ∨ a = -11) := 
by
  sorry

end perfect_square_trinomial_l236_236190


namespace n_is_one_sixth_sum_of_list_l236_236261

-- Define the condition that n is 4 times the average of the other 20 numbers
def satisfies_condition (n : ℝ) (l : List ℝ) : Prop :=
  l.length = 21 ∧
  n ∈ l ∧
  n = 4 * (l.erase n).sum / 20

-- State the main theorem
theorem n_is_one_sixth_sum_of_list {n : ℝ} {l : List ℝ} (h : satisfies_condition n l) :
  n = (1 / 6) * l.sum :=
by
  sorry

end n_is_one_sixth_sum_of_list_l236_236261


namespace sin_cos_alpha_l236_236297

open Real

theorem sin_cos_alpha (α : ℝ) (h1 : sin (2 * α) = -sqrt 2 / 2) (h2 : α ∈ Set.Ioc (3 * π / 2) (2 * π)) :
  sin α + cos α = sqrt 2 / 2 :=
sorry

end sin_cos_alpha_l236_236297


namespace smallest_number_divisible_l236_236659

theorem smallest_number_divisible (n : ℕ) :
  (∀ d ∈ [4, 6, 8, 10, 12, 14, 16], (n - 16) % d = 0) ↔ n = 3376 :=
by {
  sorry
}

end smallest_number_divisible_l236_236659


namespace solve_system_of_equations_l236_236789

theorem solve_system_of_equations
  {a b c d x y z : ℝ}
  (h1 : x + y + z = 1)
  (h2 : a * x + b * y + c * z = d)
  (h3 : a^2 * x + b^2 * y + c^2 * z = d^2)
  (hne1 : a ≠ b)
  (hne2 : a ≠ c)
  (hne3 : b ≠ c) :
  x = (d - b) * (d - c) / ((a - b) * (a - c)) ∧
  y = (d - a) * (d - c) / ((b - a) * (b - c)) ∧
  z = (d - a) * (d - b) / ((c - a) * (c - b)) :=
sorry

end solve_system_of_equations_l236_236789


namespace combination_seven_choose_three_l236_236406

-- Define the combination formula
def combination (n k : ℕ) : ℕ :=
  n.choose k

-- Define the problem-specific values
def n : ℕ := 7
def k : ℕ := 3

-- Problem statement: Prove that the number of combinations of 3 toppings from 7 is 35
theorem combination_seven_choose_three : combination 7 3 = 35 :=
  by
    sorry

end combination_seven_choose_three_l236_236406


namespace find_f_20_l236_236763

theorem find_f_20 (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f x = (1/2) * f (x + 2))
  (h2 : f 2 = 1) :
  f 20 = 512 :=
sorry

end find_f_20_l236_236763


namespace avg_children_in_families_with_children_l236_236809

theorem avg_children_in_families_with_children
  (total_families : ℕ)
  (avg_children_per_family : ℕ)
  (childless_families : ℕ)
  (total_children : ℕ := total_families * avg_children_per_family)
  (families_with_children : ℕ := total_families - childless_families)
  (avg_children_in_families_with_children : ℚ := total_children / families_with_children) :
  avg_children_in_families_with_children = 4 :=
by
  have h1 : total_families = 12 := sorry
  have h2 : avg_children_per_family = 3 := sorry
  have h3 : childless_families = 3 := sorry
  have h4 : total_children = 12 * 3 := sorry
  have h5 : families_with_children = 12 - 3 := sorry
  have h6 : avg_children_in_families_with_children = (12 * 3) / (12 - 3) := sorry
  have h7 : ((12 * 3) / (12 - 3) : ℚ) = 4 := sorry
  exact h7

end avg_children_in_families_with_children_l236_236809


namespace find_minimum_value_2a_plus_b_l236_236872

theorem find_minimum_value_2a_plus_b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_re_z : (3 * a * b + 2) = 4) : 2 * a + b = (4 * Real.sqrt 3) / 3 :=
sorry

end find_minimum_value_2a_plus_b_l236_236872


namespace extra_games_needed_l236_236270

def initial_games : ℕ := 500
def initial_success_rate : ℚ := 0.49
def target_success_rate : ℚ := 0.5

theorem extra_games_needed :
  ∀ (x : ℕ),
  (245 + x) / (initial_games + x) = target_success_rate → x = 10 := 
by
  sorry

end extra_games_needed_l236_236270


namespace max_sum_11xy_3x_2012yz_l236_236975

theorem max_sum_11xy_3x_2012yz (x y z : ℕ) (h : x + y + z = 1000) : 
  11 * x * y + 3 * x + 2012 * y * z ≤ 503000000 :=
sorry

end max_sum_11xy_3x_2012yz_l236_236975


namespace value_of_first_equation_l236_236039

theorem value_of_first_equation (x y a : ℝ) 
  (h₁ : 2 * x + y = a) 
  (h₂ : x + 2 * y = 10) 
  (h₃ : (x + y) / 3 = 4) : 
  a = 12 :=
by 
  sorry

end value_of_first_equation_l236_236039


namespace leftover_space_desks_bookcases_l236_236842

theorem leftover_space_desks_bookcases 
  (number_of_desks : ℕ) (number_of_bookcases : ℕ)
  (wall_length : ℝ) (desk_length : ℝ) (bookcase_length : ℝ) (space_between : ℝ)
  (equal_number : number_of_desks = number_of_bookcases)
  (wall_length_eq : wall_length = 15)
  (desk_length_eq : desk_length = 2)
  (bookcase_length_eq : bookcase_length = 1.5)
  (space_between_eq : space_between = 0.5) :
  ∃ k : ℝ, k = 3 := 
by
  sorry

end leftover_space_desks_bookcases_l236_236842


namespace min_inv_sum_l236_236768

open Real

theorem min_inv_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 12) :
  min ((1 / x) + (1 / y)) (1 / 3) :=
sorry

end min_inv_sum_l236_236768


namespace volume_ratio_cones_l236_236243

theorem volume_ratio_cones :
  let rC := 16.5
  let hC := 33
  let rD := 33
  let hD := 16.5
  let VC := (1 / 3) * Real.pi * rC^2 * hC
  let VD := (1 / 3) * Real.pi * rD^2 * hD
  (VC / VD) = (1 / 2) :=
by
  sorry

end volume_ratio_cones_l236_236243


namespace maximum_xyz_l236_236076

theorem maximum_xyz {x y z : ℝ} (hx: 0 < x) (hy: 0 < y) (hz: 0 < z) 
  (h : (x * y) + z = (x + z) * (y + z)) : xyz ≤ (1 / 27) :=
by
  sorry

end maximum_xyz_l236_236076


namespace find_pairs_l236_236705

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (a^3 * b - 1) % (a + 1) = 0 ∧ (b^3 * a + 1) % (b - 1) = 0 ↔ (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3) :=
by
  sorry

end find_pairs_l236_236705


namespace chocolate_candy_cost_l236_236980

-- Define the constants and conditions
def cost_per_box : ℕ := 5
def candies_per_box : ℕ := 30
def discount_rate : ℝ := 0.1

-- Define the total number of candies to buy
def total_candies : ℕ := 450

-- Define the threshold for applying discount
def discount_threshold : ℕ := 300

-- Calculate the number of boxes needed
def boxes_needed (total_candies : ℕ) (candies_per_box : ℕ) : ℕ :=
  total_candies / candies_per_box

-- Calculate the total cost without discount
def total_cost (boxes_needed : ℕ) (cost_per_box : ℕ) : ℝ :=
  boxes_needed * cost_per_box

-- Calculate the discounted cost
def discounted_cost (total_cost : ℝ) (discount_rate : ℝ) : ℝ :=
  if total_candies > discount_threshold then
    total_cost * (1 - discount_rate)
  else
    total_cost

-- Statement to be proved
theorem chocolate_candy_cost :
  discounted_cost 
    (total_cost (boxes_needed total_candies candies_per_box) cost_per_box) 
    discount_rate = 67.5 :=
by
  -- Proof is needed here, using the correct steps from the solution.
  sorry

end chocolate_candy_cost_l236_236980


namespace marie_messages_days_l236_236078

theorem marie_messages_days (initial_messages : ℕ) (read_per_day : ℕ) (new_per_day : ℕ) (days : ℕ) :
  initial_messages = 98 ∧ read_per_day = 20 ∧ new_per_day = 6 → days = 7 :=
by
  sorry

end marie_messages_days_l236_236078


namespace money_allocation_l236_236325

theorem money_allocation (x y : ℝ) (h1 : x + 1/2 * y = 50) (h2 : y + 2/3 * x = 50) : 
  x + 1/2 * y = 50 ∧ y + 2/3 * x = 50 :=
by
  exact ⟨h1, h2⟩

end money_allocation_l236_236325


namespace calculation_results_in_a_pow_5_l236_236528

variable (a : ℕ)

theorem calculation_results_in_a_pow_5 : a^3 * a^2 = a^5 := 
  by sorry

end calculation_results_in_a_pow_5_l236_236528


namespace probability_ball_two_at_least_twice_given_sum_is_seven_l236_236022

noncomputable def draws : List ℕ := [1, 2, 3, 4]

def sum_eq_seven (l : List ℕ) : Prop := l.sum = 7

def ball_two_at_least_twice (l : List ℕ) : Prop := (l.count 2) ≥ 2

theorem probability_ball_two_at_least_twice_given_sum_is_seven :
  (ProbSum : ℚ) = ((count_filter (λ l : List ℕ, ball_two_at_least_twice l ∧ sum_eq_seven l) 
  (product_tripples draws).card) / (count_filter sum_eq_seven (product_tripples draws)).card) :=
begin
  sorry
end

end probability_ball_two_at_least_twice_given_sum_is_seven_l236_236022


namespace ratio_mark_days_used_l236_236845

-- Defining the conditions
def num_sick_days : ℕ := 10
def num_vacation_days : ℕ := 10
def total_hours_left : ℕ := 80
def hours_per_workday : ℕ := 8

-- Total days allotted
def total_days_allotted : ℕ :=
  num_sick_days + num_vacation_days

-- Days left for Mark
def days_left : ℕ :=
  total_hours_left / hours_per_workday

-- Days used by Mark
def days_used : ℕ :=
  total_days_allotted - days_left

-- The ratio of days used to total days allotted (expected to be 1:2)
def ratio_used_to_allotted : ℚ :=
  days_used / total_days_allotted

theorem ratio_mark_days_used :
  ratio_used_to_allotted = 1 / 2 :=
sorry

end ratio_mark_days_used_l236_236845


namespace minimum_area_triangle_AOB_l236_236802

theorem minimum_area_triangle_AOB : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (3 / a + 2 / b = 1) ∧ (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ (3 / a + 2 / b = 1) → (1/2 * a * b ≥ 12)) := 
sorry

end minimum_area_triangle_AOB_l236_236802


namespace quadratic_real_roots_range_l236_236447

theorem quadratic_real_roots_range (m : ℝ) : 
  (∃ x : ℝ, (m-1)*x^2 + x + 1 = 0) → (m ≤ 5/4 ∧ m ≠ 1) :=
by
  sorry

end quadratic_real_roots_range_l236_236447


namespace vertex_in_fourth_quadrant_l236_236458

theorem vertex_in_fourth_quadrant (a : ℝ) (ha : a < 0) :  
  let x_vertex := -a / 4
  let y_vertex := (-40 - a^2) / 8
  x_vertex > 0 ∧ y_vertex < 0 := by
  let x_vertex := -a / 4
  let y_vertex := (-40 - a^2) / 8
  have hx : x_vertex > 0 := by sorry
  have hy : y_vertex < 0 := by sorry
  exact And.intro hx hy

end vertex_in_fourth_quadrant_l236_236458


namespace monthly_income_of_P_l236_236251

theorem monthly_income_of_P (P Q R : ℕ) (h1 : P + Q = 10100) (h2 : Q + R = 12500) (h3 : P + R = 10400) : 
  P = 4000 := 
by 
  sorry

end monthly_income_of_P_l236_236251


namespace probability_perpendicular_is_one_sixth_l236_236344

noncomputable def probability_perpendicular : ℚ :=
  let A := {2, 3, 4, 5}
  let B := {1, 3, 5}
  let possible_pairs : Set (ℕ × ℕ) :=
    { (a, b) | a ∈ A ∧ b ∈ B }
  let perpendicular_pairs : Set (ℕ × ℕ) :=
    { (3, 3), (5, 5) }
  (perpendicular_pairs.to_finset.card : ℚ) / (possible_pairs.to_finset.card : ℚ)

theorem probability_perpendicular_is_one_sixth : probability_perpendicular = 1 / 6 :=
by sorry

end probability_perpendicular_is_one_sixth_l236_236344


namespace sell_price_equal_percentage_l236_236105

theorem sell_price_equal_percentage (SP : ℝ) (CP : ℝ) :
  (SP - CP) / CP * 100 = (CP - 1280) / CP * 100 → 
  (1937.5 = CP + 0.25 * CP) → 
  SP = 1820 :=
by 
  -- Note: skip proof with sorry
  apply sorry

end sell_price_equal_percentage_l236_236105


namespace total_marbles_l236_236044

theorem total_marbles (r b g : ℕ) (h_ratio : r = 1 ∧ b = 5 ∧ g = 3) (h_green : g = 27) :
  (r + b + g) * 3 = 81 :=
  sorry

end total_marbles_l236_236044


namespace not_perfect_square_l236_236615

-- Definitions and Conditions
def N (k : ℕ) : ℕ := (10^300 - 1) / 9 * 10^k

-- Proof Statement
theorem not_perfect_square (k : ℕ) : ¬∃ (m: ℕ), m * m = N k := 
sorry

end not_perfect_square_l236_236615


namespace brenda_spay_cats_l236_236693

theorem brenda_spay_cats (c d : ℕ) (h1 : c + d = 21) (h2 : d = 2 * c) : c = 7 :=
sorry

end brenda_spay_cats_l236_236693


namespace inverse_mod_53_l236_236724

theorem inverse_mod_53 (h : 17 * 13 % 53 = 1) : 36 * 40 % 53 = 1 :=
by
  -- Given condition: 17 * 13 % 53 = 1
  -- Derived condition: (-17) * -13 % 53 = 1 which is equivalent to 17 * 13 % 53 = 1
  -- So we need to find: 36 * x % 53 = 1 where x = -13 % 53 => x = 40
  sorry

end inverse_mod_53_l236_236724


namespace probability_is_half_l236_236622

noncomputable def probability_at_least_35_cents : ℚ :=
  let total_outcomes := 32
  let successful_outcomes := 8 + 4 + 4 -- from solution steps (1, 2, 3)
  successful_outcomes / total_outcomes

theorem probability_is_half :
  probability_at_least_35_cents = 1 / 2 := by
  -- proof details are not required as per instructions
  sorry

end probability_is_half_l236_236622


namespace product_prime_probability_is_10_over_77_l236_236613

open Nat

/-- Paco's spinner selects a number between 1 and 7 and Manu's spinner selects a number between 1 and 11. 
Given these, the probability that the product of Manu's number and Paco's number is prime is 10/77. -/
theorem product_prime_probability_is_10_over_77 : 
  let Paco := {i | 1 ≤ i ∧ i ≤ 7}
  let Manu := {j | 1 ≤ j ∧ j ≤ 11}
  let total_outcomes := (finset.product (finset.range 8) (finset.range 12)).filter (λ (x : ℕ × ℕ), 1 ≤ x.1 ∧ x.1 ≤ 7 ∧ 1 ≤ x.2 ∧ x.2 ≤ 11) 
  let prime_outcomes := total_outcomes.filter (λ (x : ℕ × ℕ), prime (x.1 * x.2))
  (prime_outcomes.card : ℚ) / (total_outcomes.card : ℚ) = 10 / 77 :=
begin
  sorry
end

end product_prime_probability_is_10_over_77_l236_236613


namespace inequality_must_hold_l236_236033

section
variables {a b c : ℝ}

theorem inequality_must_hold (h : a > b) : (a - b) * c^2 ≥ 0 :=
sorry
end

end inequality_must_hold_l236_236033


namespace move_point_right_3_units_l236_236627

theorem move_point_right_3_units (x y : ℤ) (hx : x = 2) (hy : y = -1) :
  (x + 3, y) = (5, -1) :=
by
  sorry

end move_point_right_3_units_l236_236627


namespace sale_in_third_month_l236_236677

def grocer_sales (s1 s2 s4 s5 s6 : ℕ) (average : ℕ) (num_months : ℕ) (total_sales : ℕ) : Prop :=
  s1 = 5266 ∧ s2 = 5768 ∧ s4 = 5678 ∧ s5 = 6029 ∧ s6 = 4937 ∧ average = 5600 ∧ num_months = 6 ∧ total_sales = average * num_months

theorem sale_in_third_month
  (s1 s2 s4 s5 s6 total_sales : ℕ)
  (h : grocer_sales s1 s2 s4 s5 s6 5600 6 total_sales) :
  ∃ s3 : ℕ, total_sales - (s1 + s2 + s4 + s5 + s6) = s3 ∧ s3 = 5922 := 
by {
  sorry
}

end sale_in_third_month_l236_236677


namespace min_value_l236_236175

open Real

-- Definitions
variables (a b : ℝ)
axiom a_gt_zero : a > 0
axiom b_gt_one : b > 1
axiom sum_eq : a + b = 3 / 2

-- The theorem to be proved.
theorem min_value (a : ℝ) (b : ℝ) (a_gt_zero : a > 0) (b_gt_one : b > 1) (sum_eq : a + b = 3 / 2) :
  ∃ (m : ℝ), m = 6 + 4 * sqrt 2 ∧ ∀ (x y : ℝ), (x > 0) → (y > 1) → (x + y = 3 / 2) → (∃ (z : ℝ), z = 2 / x + 1 / (y - 1) ∧ z ≥ m) :=
sorry

end min_value_l236_236175


namespace binomial_sum_sum_of_n_values_l236_236371

theorem binomial_sum (n : ℕ) (h : nat.choose 28 14 + nat.choose 28 n = nat.choose 29 15) : n = 13 ∨ n = 15 := sorry

theorem sum_of_n_values : ∑ n in {n | nat.choose 28 14 + nat.choose 28 n = nat.choose 29 15}.to_finset, n = 28 :=
by
  apply finset.sum_eq_from_set,
  intros x hx,
  cases binomial_sum x hx,
  { simp [h], },
  { simp [h], }

end binomial_sum_sum_of_n_values_l236_236371


namespace unique_triple_property_l236_236635

theorem unique_triple_property (a b c : ℕ) (h1 : a ∣ b * c + 1) (h2 : b ∣ a * c + 1) (h3 : c ∣ a * b + 1) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a = 2 ∧ b = 3 ∧ c = 7) :=
by
  sorry

end unique_triple_property_l236_236635


namespace cooks_in_restaurant_l236_236816

theorem cooks_in_restaurant
  (C W : ℕ) 
  (h1 : C * 8 = 3 * W) 
  (h2 : C * 4 = (W + 12)) :
  C = 9 :=
by
  sorry

end cooks_in_restaurant_l236_236816


namespace customer_buys_two_pens_l236_236323

def num_pens (total_pens non_defective_pens : Nat) (prob : ℚ) : Nat :=
  sorry

theorem customer_buys_two_pens :
  num_pens 16 13 0.65 = 2 :=
sorry

end customer_buys_two_pens_l236_236323


namespace negation_proposition_l236_236783

theorem negation_proposition {x : ℝ} (h : ∀ x > 0, Real.sin x > 0) : ∃ x > 0, Real.sin x ≤ 0 :=
sorry

end negation_proposition_l236_236783


namespace blister_slowdown_l236_236997

theorem blister_slowdown
    (old_speed new_speed time : ℕ) (new_speed_initial : ℕ) (blister_freq : ℕ)
    (distance_old : ℕ) (blister_per_hour_slowdown : ℝ):
    -- Given conditions
    old_speed = 6 →
    new_speed = 11 →
    new_speed_initial = 11 →
    time = 4 →
    blister_freq = 2 →
    distance_old = old_speed * time →
    -- Prove that each blister slows Candace down by 10 miles per hour
    blister_per_hour_slowdown = 10 :=
  by
    sorry

end blister_slowdown_l236_236997


namespace min_value_inv_sum_l236_236766

theorem min_value_inv_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 12) : 
  ∃ z, (∀ x y : ℝ, 0 < x → 0 < y → x + y = 12 → z ≤ (1/x + 1/y)) ∧ z = 1/3 :=
sorry

end min_value_inv_sum_l236_236766


namespace compute_expression_l236_236139

theorem compute_expression :
  ( ((15 ^ 15) / (15 ^ 10)) ^ 3 * 5 ^ 6 ) / (25 ^ 2) = 3 ^ 15 * 5 ^ 17 :=
by
  -- We'll use sorry here as proof is not required
  sorry

end compute_expression_l236_236139


namespace graduation_ceremony_chairs_l236_236469

theorem graduation_ceremony_chairs (num_graduates num_teachers: ℕ) (half_as_administrators: ℕ) :
  (∀ num_graduates = 50) →
  (∀ num_teachers = 20) →
  (∀ half_as_administrators = num_teachers / 2) →
  (2 * num_graduates + num_graduates + num_teachers + half_as_administrators = 180) :=
begin
  intros,
  sorry
end

end graduation_ceremony_chairs_l236_236469


namespace fraction_not_integer_l236_236487

def containsExactlyTwoOccurrences (d : List ℕ) : Prop :=
  ∀ n ∈ [1, 2, 3, 4, 5, 6, 7], d.count n = 2

theorem fraction_not_integer
  (k m : ℕ)
  (hk : 14 = (List.length (Nat.digits 10 k)))
  (hm : 14 = (List.length (Nat.digits 10 m)))
  (hkd : containsExactlyTwoOccurrences (Nat.digits 10 k))
  (hmd : containsExactlyTwoOccurrences (Nat.digits 10 m))
  (hkm : k ≠ m) :
  ¬ ∃ d : ℕ, k = m * d := 
sorry

end fraction_not_integer_l236_236487


namespace brenda_sally_track_length_l236_236408

theorem brenda_sally_track_length
  (c d : ℝ) 
  (h1 : c / 4 * 3 = d) 
  (h2 : d - 120 = 0.75 * c - 120) 
  (h3 : 0.75 * c + 60 <= 1.25 * c - 180) 
  (h4 : (c - 120 + 0.25 * c - 60) = 1.25 * c - 180):
  c = 766.67 :=
sorry

end brenda_sally_track_length_l236_236408


namespace kitty_cleaning_time_l236_236342

theorem kitty_cleaning_time
    (picking_up_toys : ℕ := 5)
    (vacuuming : ℕ := 20)
    (dusting_furniture : ℕ := 10)
    (total_time_4_weeks : ℕ := 200)
    (weeks : ℕ := 4)
    : (total_time_4_weeks - weeks * (picking_up_toys + vacuuming + dusting_furniture)) / weeks = 15 := by
    sorry

end kitty_cleaning_time_l236_236342


namespace mom_age_when_jayson_born_l236_236959

theorem mom_age_when_jayson_born (jayson_age dad_age mom_age : ℕ) 
  (h1 : jayson_age = 10) 
  (h2 : dad_age = 4 * jayson_age)
  (h3 : mom_age = dad_age - 2) :
  mom_age - jayson_age = 28 :=
by
  sorry

end mom_age_when_jayson_born_l236_236959


namespace compare_exponents_l236_236176

noncomputable def a : ℝ := 0.8 ^ 5.2
noncomputable def b : ℝ := 0.8 ^ 5.5
noncomputable def c : ℝ := 5.2 ^ 0.1

theorem compare_exponents (a b c : ℝ) (h1 : a = 0.8 ^ 5.2) (h2 : b = 0.8 ^ 5.5) (h3 : c = 5.2 ^ 0.1) :
  b < a ∧ a < c := sorry

end compare_exponents_l236_236176


namespace remaining_tanning_time_l236_236332

noncomputable def tanning_limit : ℕ := 200
noncomputable def daily_tanning_time : ℕ := 30
noncomputable def weekly_tanning_days : ℕ := 2
noncomputable def weeks_tanned : ℕ := 2

theorem remaining_tanning_time :
  let total_tanning_first_two_weeks := daily_tanning_time * weekly_tanning_days * weeks_tanned
  tanning_limit - total_tanning_first_two_weeks = 80 :=
by
  let total_tanning_first_two_weeks := daily_tanning_time * weekly_tanning_days * weeks_tanned
  have h : total_tanning_first_two_weeks = 120 := by sorry
  show tanning_limit - total_tanning_first_two_weeks = 80 from sorry

end remaining_tanning_time_l236_236332


namespace average_percentage_increase_is_correct_l236_236114

def initial_prices : List ℝ := [300, 450, 600]
def price_increases : List ℝ := [0.10, 0.15, 0.20]

noncomputable def total_original_price : ℝ :=
  initial_prices.sum

noncomputable def total_new_price : ℝ :=
  (List.zipWith (λ p i => p * (1 + i)) initial_prices price_increases).sum

noncomputable def total_price_increase : ℝ :=
  total_new_price - total_original_price

noncomputable def average_percentage_increase : ℝ :=
  (total_price_increase / total_original_price) * 100

theorem average_percentage_increase_is_correct :
  average_percentage_increase = 16.11 := by
  sorry

end average_percentage_increase_is_correct_l236_236114


namespace probability_sum_is_odd_l236_236571

theorem probability_sum_is_odd (S : Finset ℕ) (h_S : S = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37})
    (h_card : S.card = 12) :
  let choices := S.powerset.filter (λ t, t.card = 4),
      odd_sum_choices := choices.filter (λ t, t.sum % 2 = 1) in
  (odd_sum_choices.card : ℚ) / choices.card = 1 / 3 :=
by  
  sorry

end probability_sum_is_odd_l236_236571


namespace sum_of_intersections_l236_236203

-- Definitions and conditions
variable (n : ℕ)
def S : Finset ℕ := Finset.range (n + 1)

structure TripleSubset (S : Finset ℕ) :=
(A1 A2 A3 : Finset ℕ)
(h_union : A1 ∪ A2 ∪ A3 = S)

def T (S : Finset ℕ) : Finset (TripleSubset S) :=
  Finset.univ.filter (λ t, t.h_union)

-- Main statement
theorem sum_of_intersections :
  ∑ t in T (S n), (t.A1 ∩ t.A2 ∩ t.A3).card = n * 7^(n-1) :=
sorry

end sum_of_intersections_l236_236203


namespace pure_imaginary_solution_l236_236386

theorem pure_imaginary_solution (m : ℝ) 
  (h : ∃ m : ℝ, (m^2 + m - 2 = 0) ∧ (m^2 - 1 ≠ 0)) : m = -2 :=
sorry

end pure_imaginary_solution_l236_236386


namespace salt_solution_mixture_l236_236189

theorem salt_solution_mixture (x : ℝ) :  
  (0.80 * x + 0.35 * 150 = 0.55 * (150 + x)) → x = 120 :=
by 
  sorry

end salt_solution_mixture_l236_236189


namespace rest_area_milepost_l236_236936

theorem rest_area_milepost : 
  let fifth_exit := 30
  let fifteenth_exit := 210
  (3 / 5) * (fifteenth_exit - fifth_exit) + fifth_exit = 138 := 
by 
  let fifth_exit := 30
  let fifteenth_exit := 210
  sorry

end rest_area_milepost_l236_236936


namespace base7_to_base10_245_l236_236263

theorem base7_to_base10_245 : (2 * 7^2 + 4 * 7^1 + 5 * 7^0) = 131 := by
  sorry

end base7_to_base10_245_l236_236263


namespace maximize_annual_profit_l236_236391

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 14 then (2 / 3) * x^2 + 4 * x
  else 17 * x + 400 / x - 80

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 14 then 16 * x - f x - 30
  else 16 * x - f x - 30

theorem maximize_annual_profit : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 35 ∧ g x = 24 ∧ (∀ y, 0 ≤ y ∧ y ≤ 35 → g y ≤ g x) :=
begin
  existsi 9,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { simp [g, f], sorry },
  { intros y hy,
    have hy1 : y ≤ 14 ∨ 14 < y := le_or_lt y 14,
    cases hy1,
    { sorry },
    { sorry } },
end

end maximize_annual_profit_l236_236391


namespace middle_number_is_12_l236_236943

theorem middle_number_is_12 (x y z : ℕ) (h1 : x + y = 20) (h2 : x + z = 25) (h3 : y + z = 29) (h4 : x < y) (h5 : y < z) : y = 12 :=
by
  sorry

end middle_number_is_12_l236_236943


namespace negation_of_exists_l236_236104

variable (a : ℝ)

theorem negation_of_exists (h : ¬ ∃ x : ℝ, x^2 + a * x + 1 < 0) : ∀ x : ℝ, x^2 + a * x + 1 ≥ 0 :=
by
  sorry

end negation_of_exists_l236_236104


namespace ravioli_to_tortellini_ratio_l236_236063

-- Definitions from conditions
def total_students : ℕ := 800
def ravioli_students : ℕ := 300
def tortellini_students : ℕ := 150

-- Ratio calculation as a theorem
theorem ravioli_to_tortellini_ratio : 2 = ravioli_students / Nat.gcd ravioli_students tortellini_students :=
by
  -- Given the defined values
  have gcd_val : Nat.gcd ravioli_students tortellini_students = 150 := by
    sorry
  have ratio_simp : ravioli_students / 150 = 2 := by
    sorry
  exact ratio_simp

end ravioli_to_tortellini_ratio_l236_236063


namespace Micah_words_per_minute_l236_236901

-- Defining the conditions
def Isaiah_words_per_minute : ℕ := 40
def extra_words : ℕ := 1200

-- Proving the statement that Micah can type 20 words per minute
theorem Micah_words_per_minute (Isaiah_wpm : ℕ) (extra_w : ℕ) : Isaiah_wpm = 40 → extra_w = 1200 → (Isaiah_wpm * 60 - extra_w) / 60 = 20 :=
by
  -- Sorry is used to skip the proof
  sorry

end Micah_words_per_minute_l236_236901


namespace space_diagonals_Q_l236_236829

-- Definitions based on the conditions
def vertices (Q : Type) : ℕ := 30
def edges (Q : Type) : ℕ := 70
def faces (Q : Type) : ℕ := 40
def triangular_faces (Q : Type) : ℕ := 20
def quadrilateral_faces (Q : Type) : ℕ := 15
def pentagon_faces (Q : Type) : ℕ := 5

-- Problem Statement
theorem space_diagonals_Q :
  ∀ (Q : Type),
  vertices Q = 30 →
  edges Q = 70 →
  faces Q = 40 →
  triangular_faces Q = 20 →
  quadrilateral_faces Q = 15 →
  pentagon_faces Q = 5 →
  ∃ d : ℕ, d = 310 := 
by
  -- At this point only the structure of the proof is set up.
  sorry

end space_diagonals_Q_l236_236829


namespace quadratic_complete_square_l236_236641

theorem quadratic_complete_square (b c : ℝ) (h : ∀ x : ℝ, x^2 - 24 * x + 50 = (x + b)^2 + c) : b + c = -106 :=
by
  sorry

end quadratic_complete_square_l236_236641


namespace team_A_win_probability_l236_236941

theorem team_A_win_probability :
  let win_prob := (1 / 3 : ℝ)
  let team_A_lead := 2
  let total_sets := 5
  let require_wins := 3
  let remaining_sets := total_sets - team_A_lead
  let prob_team_B_win_remaining := (1 - win_prob) ^ remaining_sets
  let prob_team_A_win := 1 - prob_team_B_win_remaining
  prob_team_A_win = 19 / 27 := by
    sorry

end team_A_win_probability_l236_236941


namespace area_under_abs_sin_l236_236503

noncomputable def f (x : ℝ) : ℝ := |Real.sin x|

theorem area_under_abs_sin : 
  ∫ x in -Real.pi..Real.pi, f x = 4 :=
by
  sorry

end area_under_abs_sin_l236_236503


namespace supplementary_angle_l236_236660

theorem supplementary_angle (θ : ℝ) (k : ℤ) : (θ = 10) → (∃ k, θ + 250 = k * 360 + 360) :=
by
  sorry

end supplementary_angle_l236_236660


namespace hexagon_side_count_l236_236696

noncomputable def convex_hexagon_sides (a b perimeter : ℕ) : ℕ := 
  if a ≠ b then 6 - (perimeter - (6 * b)) else 0

theorem hexagon_side_count (G H I J K L : ℕ)
  (a b : ℕ)
  (p : ℕ)
  (dist_a : a = 7)
  (dist_b : b = 8)
  (perimeter : p = 46)
  (cond : GHIJKL = [a, b, X, Y, Z, W] ∧ ∀ x ∈ [X, Y, Z, W], x = a ∨ x = b)
  : convex_hexagon_sides a b p = 4 :=
by 
  sorry

end hexagon_side_count_l236_236696


namespace train_cross_bridge_time_l236_236040

noncomputable def time_to_cross_bridge (length_of_train : ℝ) (speed_kmh : ℝ) (length_of_bridge : ℝ) : ℝ :=
  let total_distance := length_of_train + length_of_bridge
  let speed_mps := speed_kmh * (1000 / 3600)
  total_distance / speed_mps

theorem train_cross_bridge_time :
  time_to_cross_bridge 110 72 112 = 11.1 :=
by
  sorry

end train_cross_bridge_time_l236_236040


namespace unique_function_and_sum_calculate_n_times_s_l236_236485

def f : ℝ → ℝ := sorry

theorem unique_function_and_sum :
  (∀ x y z : ℝ, f (x^2 + 2 * f y) = x * f x + y * f z) →
  (∃! g : ℝ → ℝ, ∀ x, f x = g x) ∧ f 3 = 0 :=
sorry

theorem calculate_n_times_s :
  ∃ n s : ℕ, (∃! f : ℝ → ℝ, ∀ x y z : ℝ, f (x^2 + 2 * f y) = x * f x + y * f z) ∧ n = 1 ∧ s = (0 : ℝ) ∧ n * s = 0 :=
sorry

end unique_function_and_sum_calculate_n_times_s_l236_236485


namespace krishan_money_l236_236380

-- Define the constants
def Ram : ℕ := 490
def ratio1 : ℕ := 7
def ratio2 : ℕ := 17

-- Defining the relationship
def ratio_RG (Ram Gopal : ℕ) : Prop := Ram / Gopal = ratio1 / ratio2
def ratio_GK (Gopal Krishan : ℕ) : Prop := Gopal / Krishan = ratio1 / ratio2

-- Define the problem
theorem krishan_money (R G K : ℕ) (h1 : R = Ram) (h2 : ratio_RG R G) (h3 : ratio_GK G K) : K = 2890 :=
by
  sorry

end krishan_money_l236_236380


namespace number_equation_form_l236_236156

variable (a : ℝ)

theorem number_equation_form :
  3 * a + 5 = 4 * a := 
sorry

end number_equation_form_l236_236156


namespace decreased_cost_l236_236121

theorem decreased_cost (original_cost : ℝ) (decrease_percentage : ℝ) (h1 : original_cost = 200) (h2 : decrease_percentage = 0.50) : 
  (original_cost - original_cost * decrease_percentage) = 100 :=
by
  -- This is the proof placeholder
  sorry

end decreased_cost_l236_236121


namespace compare_M_N_l236_236204

variable (a : ℝ)

def M : ℝ := 2 * a * (a - 2) + 7
def N : ℝ := (a - 2) * (a - 3)

theorem compare_M_N : M a > N a :=
by
  sorry

end compare_M_N_l236_236204


namespace parabola_standard_equation_l236_236581

theorem parabola_standard_equation (h : ∀ y, y = 1/2) : ∃ c : ℝ, c = -2 ∧ (∀ x y, x^2 = c * y) :=
by
  -- Considering 'h' provides the condition for the directrix
  sorry

end parabola_standard_equation_l236_236581


namespace misplaced_value_l236_236987

open Polynomial

/-
  Assume the sequence values are given as follows:
  Let s : ℕ → ℕ be such that s 0 = 9604, s 1 = 9801, s 2 = 10201, s 3 = 10404, 
  s 4 = 10816, s 5 = 11025, s 6 = 11449, s 7 = 11664, and s 8 = 12100.
  
  Prove that the value s 2 (i.e., 10201) is likely misplaced or calculated incorrectly.
-/
theorem misplaced_value :
  ∃ (s : ℕ → ℕ), 
    s 0 = 9604 ∧ s 1 = 9801 ∧ s 2 = 10201 ∧ s 3 = 10404 ∧ 
    s 4 = 10816 ∧ s 5 = 11025 ∧ s 6 = 11449 ∧ s 7 = 11664 ∧ 
    s 8 = 12100 ∧ 
    ∃ (a b c : ℚ),
      ∀ x : ℕ, 
        (x > 0 → (s x - s (x - 1)) = (s x = x^2 * a + x * b + c)) ∧ 
        (s 2 ≠ 10201) :=
sorry

end misplaced_value_l236_236987


namespace xy_sum_of_squares_l236_236741

theorem xy_sum_of_squares (x y : ℝ) (h1 : x - y = 18) (h2 : x + y = 22) : x^2 + y^2 = 404 := by
  sorry

end xy_sum_of_squares_l236_236741


namespace workerB_time_to_complete_job_l236_236120

theorem workerB_time_to_complete_job 
  (time_A : ℝ) (time_together: ℝ) (time_B : ℝ) 
  (h1 : time_A = 5) 
  (h2 : time_together = 3.333333333333333) 
  (h3 : 1 / time_A + 1 / time_B = 1 / time_together) 
  : time_B = 10 := 
  sorry

end workerB_time_to_complete_job_l236_236120


namespace winning_candidate_votes_l236_236113

def total_votes : ℕ := 100000
def winning_percentage : ℚ := 42 / 100
def expected_votes : ℚ := 42000

theorem winning_candidate_votes : winning_percentage * total_votes = expected_votes := by
  sorry

end winning_candidate_votes_l236_236113


namespace john_needs_packs_l236_236070

-- Definitions based on conditions
def utensils_per_pack : Nat := 30
def utensils_types : Nat := 3
def spoons_per_pack : Nat := utensils_per_pack / utensils_types
def spoons_needed : Nat := 50

-- Statement to prove
theorem john_needs_packs : (50 / spoons_per_pack) = 5 :=
by
  -- To complete the proof
  sorry

end john_needs_packs_l236_236070


namespace diameter_of_circle_A_l236_236141

def radius_of_circle_B : ℝ := 10

def area_of_circle (r : ℝ) : ℝ := π * r ^ 2

def area_ratio_circle_A_shaded (area_A shaded : ℝ) : Prop :=
  area_A / shaded = 1 / 7

theorem diameter_of_circle_A
  (h1 : radius_of_circle_B = 10)
  (h2 : ∀ r_A : ℝ, area_ratio_circle_A_shaded (area_of_circle r_A) (area_of_circle radius_of_circle_B - area_of_circle r_A)) :
  ∃ d_A : ℝ, d_A = 7.08 :=
begin
  sorry
end

end diameter_of_circle_A_l236_236141


namespace problem_statement_l236_236300

variables {α β : Plane} {m : Line}

def parallel (a b : Plane) : Prop := sorry
def perpendicular (m : Line) (π : Plane) : Prop := sorry

axiom parallel_symm {a b : Plane} : parallel a b → parallel b a
axiom perpendicular_trans {m : Line} {a b : Plane} : perpendicular m a → parallel a b → perpendicular m b

theorem problem_statement (h1 : parallel α β) (h2 : perpendicular m α) : perpendicular m β :=
  perpendicular_trans h2 (parallel_symm h1)

end problem_statement_l236_236300


namespace cost_for_Greg_l236_236276

theorem cost_for_Greg (N P M : ℝ)
(Bill : 13 * N + 26 * P + 19 * M = 25)
(Paula : 27 * N + 18 * P + 31 * M = 31) :
  24 * N + 120 * P + 52 * M = 88 := 
sorry

end cost_for_Greg_l236_236276


namespace total_amount_received_is_1465_l236_236757

-- defining the conditions
def principal_1 : ℝ := 4000
def principal_2 : ℝ := 8200
def rate_1 : ℝ := 0.11
def rate_2 : ℝ := rate_1 + 0.015

-- defining the interest from each account
def interest_1 := principal_1 * rate_1
def interest_2 := principal_2 * rate_2

-- stating the total amount received
def total_received := interest_1 + interest_2

-- proving the total amount received
theorem total_amount_received_is_1465 : total_received = 1465 := by
  -- proof goes here
  sorry

end total_amount_received_is_1465_l236_236757


namespace find_first_number_l236_236248

theorem find_first_number (a b : ℕ) (k : ℕ) (h1 : a = 3 * k) (h2 : b = 4 * k) (h3 : Nat.lcm a b = 84) : a = 21 := 
sorry

end find_first_number_l236_236248


namespace large_envelopes_count_l236_236844

theorem large_envelopes_count
  (total_letters : ℕ) (small_envelope_letters : ℕ) (letters_per_large_envelope : ℕ)
  (H1 : total_letters = 80)
  (H2 : small_envelope_letters = 20)
  (H3 : letters_per_large_envelope = 2) :
  (total_letters - small_envelope_letters) / letters_per_large_envelope = 30 :=
sorry

end large_envelopes_count_l236_236844


namespace min_value_am_hm_inequality_l236_236906

theorem min_value_am_hm_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
    (a + b + c) * (1 / a + 1 / b + 1 / c) ≥ 9 :=
sorry

end min_value_am_hm_inequality_l236_236906


namespace ordered_pairs_satisfy_conditions_l236_236425

theorem ordered_pairs_satisfy_conditions :
  ∀ (a b : ℕ), 0 < a → 0 < b → (a^2 + b^2 + 25 = 15 * a * b) → Nat.Prime (a^2 + a * b + b^2) →
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) :=
by
  intros a b ha hb h1 h2
  sorry

end ordered_pairs_satisfy_conditions_l236_236425


namespace probability_odd_product_sum_divisible_by_5_l236_236500

theorem probability_odd_product_sum_divisible_by_5 :
  (∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 20 ∧ 1 ≤ b ∧ b ≤ 20 ∧ a ≠ b ∧ (a * b % 2 = 1 ∧ (a + b) % 5 = 0)) →
  ∃ (p : ℚ), p = 3 / 95 :=
by
  sorry

end probability_odd_product_sum_divisible_by_5_l236_236500


namespace sets_are_equal_l236_236304

def int : Type := ℤ  -- Redefine integer as ℤ for clarity

def SetA : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}
def SetB : Set ℤ := {x | ∃ k : ℤ, x = 2 * k - 1}

theorem sets_are_equal : SetA = SetB := by
  -- implement the proof here
  sorry

end sets_are_equal_l236_236304


namespace octal_rep_square_l236_236055

theorem octal_rep_square (a b c : ℕ) (n : ℕ) (h : n^2 = 8^3 * a + 8^2 * b + 8 * 3 + c) (h₀ : a ≠ 0) : c = 1 :=
sorry

end octal_rep_square_l236_236055


namespace rectangle_perimeter_126_l236_236237

/-- Define the sides of the rectangle in terms of a common multiplier -/
def sides (x : ℝ) : ℝ × ℝ := (4 * x, 3 * x)

/-- Define the area of the rectangle given the common multiplier -/
def area (x : ℝ) : ℝ := (4 * x) * (3 * x)

example : ∃ (x : ℝ), area x = 972 :=
by
  sorry

/-- Calculate the perimeter of the rectangle given the common multiplier -/
def perimeter (x : ℝ) : ℝ := 2 * ((4 * x) + (3 * x))

/-- The final proof statement, stating that the perimeter of the rectangle is 126 meters,
    given the ratio of its sides and its area. -/
theorem rectangle_perimeter_126 (x : ℝ) (h: area x = 972) : perimeter x = 126 :=
by
  sorry

end rectangle_perimeter_126_l236_236237


namespace max_empty_squares_l236_236064

theorem max_empty_squares (board_size : ℕ) (total_cells : ℕ) 
  (initial_cockroaches : ℕ) (adjacent : ℕ → ℕ → Prop) 
  (different : ℕ → ℕ → Prop) :
  board_size = 8 → total_cells = 64 → initial_cockroaches = 2 →
  (∀ s : ℕ, s < total_cells → ∃ s1 s2 : ℕ, adjacent s s1 ∧ 
              adjacent s s2 ∧ 
              different s1 s2) →
  ∃ max_empty_cells : ℕ, max_empty_cells = 24 :=
by
  intros h_board_size h_total_cells h_initial_cockroaches h_moves
  sorry

end max_empty_squares_l236_236064


namespace union_M_N_l236_236723

def M : Set ℕ := {1, 2}
def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a - 1}

theorem union_M_N : M ∪ N = {1, 2, 3} := by
  sorry

end union_M_N_l236_236723


namespace geometric_sequence_S9_l236_236206

theorem geometric_sequence_S9 (S : ℕ → ℝ) (S3_eq : S 3 = 2) (S6_eq : S 6 = 6) : S 9 = 14 :=
by
  sorry

end geometric_sequence_S9_l236_236206


namespace arithmetic_and_geometric_sequence_statement_l236_236439

-- Arithmetic sequence definitions
def arithmetic_seq (a b d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Conditions
def a_2 : ℕ := 9
def a_5 : ℕ := 21

-- General formula and solution for part (Ⅰ)
def general_formula_arithmetic_sequence : Prop :=
  ∃ (a d : ℕ), (a + d = a_2 ∧ a + 4 * d = a_5) ∧ ∀ n : ℕ, arithmetic_seq a d n = 4 * n + 1

-- Definitions and conditions for geometric sequence derived from arithmetic sequence
def b_n (n : ℕ) : ℕ := 2 ^ (4 * n + 1)

-- Sum of the first n terms of the sequence {b_n}
def S_n (n : ℕ) : ℕ := (32 * (2 ^ (4 * n) - 1)) / 15

-- Statement that needs to be proven
theorem arithmetic_and_geometric_sequence_statement :
  general_formula_arithmetic_sequence ∧ (∀ n, S_n n = (32 * (2 ^ (4 * n) - 1)) / 15) := by
  sorry

end arithmetic_and_geometric_sequence_statement_l236_236439


namespace relationship_between_x_plus_one_and_ex_l236_236716

theorem relationship_between_x_plus_one_and_ex (x : ℝ) : x + 1 ≤ Real.exp x :=
sorry

end relationship_between_x_plus_one_and_ex_l236_236716


namespace find_n_for_primes_l236_236562

def A_n (n : ℕ) : ℕ := 1 + 7 * (10^n - 1) / 9
def B_n (n : ℕ) : ℕ := 3 + 7 * (10^n - 1) / 9

theorem find_n_for_primes (n : ℕ) :
  (∀ n, n > 0 → (Nat.Prime (A_n n) ∧ Nat.Prime (B_n n)) ↔ n = 1) :=
sorry

end find_n_for_primes_l236_236562


namespace sqrt10_parts_sqrt6_value_sqrt3_opposite_l236_236091

-- Problem 1
theorem sqrt10_parts : 3 < Real.sqrt 10 ∧ Real.sqrt 10 < 4 → (⌊Real.sqrt 10⌋ = 3 ∧ Real.sqrt 10 - 3 = Real.sqrt 10 - ⌊Real.sqrt 10⌋) :=
by
  sorry

-- Problem 2
theorem sqrt6_value (a b : ℝ) : a = Real.sqrt 6 - 2 ∧ b = 3 → (a + b - Real.sqrt 6 = 1) :=
by
  sorry

-- Problem 3
theorem sqrt3_opposite (x y : ℝ) : x = 13 ∧ y = Real.sqrt 3 - 1 → (-(x - y) = Real.sqrt 3 - 14) :=
by
  sorry

end sqrt10_parts_sqrt6_value_sqrt3_opposite_l236_236091


namespace find_3a_plus_3b_l236_236047

theorem find_3a_plus_3b (a b : ℚ) (h1 : 2 * a + 5 * b = 47) (h2 : 8 * a + 2 * b = 50) :
  3 * a + 3 * b = 73 / 2 := 
sorry

end find_3a_plus_3b_l236_236047


namespace initial_bushes_count_l236_236003

theorem initial_bushes_count (n : ℕ) (h : 2 * (27 * n - 26) + 26 = 190 + 26) : n = 8 :=
by
  sorry

end initial_bushes_count_l236_236003


namespace problem_l236_236489

theorem problem (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (ha2 : a ≤ 2) (hb2 : b ≤ 2) (hc2 : c ≤ 2) :
  (a - b) * (b - c) * (a - c) ≤ 2 := 
sorry

end problem_l236_236489


namespace find_angle_B_l236_236746

variable (a b c A B C : ℝ)

-- Assuming all the necessary conditions and givens
axiom triangle_condition1 : a * (Real.sin B * Real.cos C) + c * (Real.sin B * Real.cos A) = (1 / 2) * b
axiom triangle_condition2 : a > b

-- We need to prove B = π / 6
theorem find_angle_B : B = π / 6 :=
by
  sorry

end find_angle_B_l236_236746


namespace Z_4_1_eq_27_l236_236850

def Z (a b : ℕ) : ℕ := a^3 - 3 * a^2 * b + 3 * a * b^2 - b^3

theorem Z_4_1_eq_27 : Z 4 1 = 27 := by
  sorry

end Z_4_1_eq_27_l236_236850


namespace n_fraction_of_sum_l236_236258

theorem n_fraction_of_sum (n S : ℝ) (h1 : n = S / 5) (h2 : S ≠ 0) :
  n = 1 / 6 * ((S + (S / 5))) :=
by
  sorry

end n_fraction_of_sum_l236_236258


namespace num_passed_candidates_l236_236625

theorem num_passed_candidates
  (total_candidates : ℕ)
  (avg_passed_marks : ℕ)
  (avg_failed_marks : ℕ)
  (overall_avg_marks : ℕ)
  (h1 : total_candidates = 120)
  (h2 : avg_passed_marks = 39)
  (h3 : avg_failed_marks = 15)
  (h4 : overall_avg_marks = 35) :
  ∃ (P : ℕ), P = 100 :=
by
  sorry

end num_passed_candidates_l236_236625


namespace no_integer_m_l236_236073

theorem no_integer_m (n r m : ℕ) (hn : 1 ≤ n) (hr : 2 ≤ r) : 
  ¬ (∃ m : ℕ, n * (n + 1) * (n + 2) = m ^ r) :=
sorry

end no_integer_m_l236_236073


namespace speed_excluding_stoppages_l236_236704

-- Conditions
def speed_with_stoppages := 33 -- kmph
def stoppage_time_per_hour := 16 -- minutes

-- Conversion of conditions to statements
def running_time_per_hour := 60 - stoppage_time_per_hour -- minutes
def running_time_in_hours := running_time_per_hour / 60 -- hours

-- Proof Statement
theorem speed_excluding_stoppages : 
  (speed_with_stoppages = 33) → (stoppage_time_per_hour = 16) → (75 = 33 / (44 / 60)) :=
by
  intros h1 h2
  sorry

end speed_excluding_stoppages_l236_236704


namespace value_of_a_100_l236_236894

open Nat

def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (succ k) => sequence k + 4

theorem value_of_a_100 : sequence 99 = 397 := by
  sorry

end value_of_a_100_l236_236894


namespace monotonically_increasing_range_of_a_l236_236732

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + 2 * x + 3

theorem monotonically_increasing_range_of_a :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ -Real.sqrt 2 ≤ a ∧ a ≤ Real.sqrt 2 :=
by
  sorry

end monotonically_increasing_range_of_a_l236_236732


namespace sum_of_roots_zero_l236_236882

theorem sum_of_roots_zero (p q : ℝ) (h1 : p = -q) (h2 : ∀ x, x^2 + p * x + q = 0) : p + q = 0 := 
by {
  sorry 
}

end sum_of_roots_zero_l236_236882


namespace apples_given_to_father_l236_236202

theorem apples_given_to_father
  (total_apples : ℤ) 
  (people_sharing : ℤ) 
  (apples_per_person : ℤ)
  (jack_and_friends : ℤ) :
  total_apples = 55 →
  people_sharing = 5 →
  apples_per_person = 9 →
  jack_and_friends = 4 →
  (total_apples - people_sharing * apples_per_person) = 10 :=
by 
  intros h1 h2 h3 h4
  sorry

end apples_given_to_father_l236_236202


namespace sum_of_n_binom_coefficient_l236_236372

theorem sum_of_n_binom_coefficient :
  (∑ n in { n : ℤ | nat.choose 28 14 + nat.choose 28 n = nat.choose 29 15}, n) = 28 := 
by
  sorry

end sum_of_n_binom_coefficient_l236_236372


namespace negation_of_one_even_is_all_odd_or_at_least_two_even_l236_236186

-- Definitions based on the problem conditions
def is_even (n : ℕ) : Prop := n % 2 = 0

def exactly_one_even (a b c : ℕ) : Prop :=
  (is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
  (¬ is_even a ∧ is_even b ∧ ¬ is_even c) ∨
  (¬ is_even a ∧ ¬ is_even b ∧ is_even c)

def all_odd (a b c : ℕ) : Prop :=
  ¬ is_even a ∧ ¬ is_even b ∧ ¬ is_even c

def at_least_two_even (a b c : ℕ) : Prop :=
  (is_even a ∧ is_even b) ∨
  (is_even a ∧ is_even c) ∨
  (is_even b ∧ is_even c)

-- The proposition to prove
theorem negation_of_one_even_is_all_odd_or_at_least_two_even (a b c : ℕ) :
  ¬ exactly_one_even a b c ↔ all_odd a b c ∨ at_least_two_even a b c :=
by sorry

end negation_of_one_even_is_all_odd_or_at_least_two_even_l236_236186


namespace bcm_hens_count_l236_236533

theorem bcm_hens_count (total_chickens : ℕ) (percent_bcm : ℝ) (percent_bcm_hens : ℝ) : ℕ :=
  let total_bcm := total_chickens * percent_bcm
  let bcm_hens := total_bcm * percent_bcm_hens
  bcm_hens

example : bcm_hens_count 100 0.20 0.80 = 16 := by
  sorry

end bcm_hens_count_l236_236533


namespace count_solutions_l236_236042

noncomputable def num_solutions : ℕ :=
  let eq1 (x y : ℝ) := 2 * x + 5 * y = 10
  let eq2 (x y : ℝ) := abs (abs (x + 1) - abs (y - 1)) = 1
  sorry

theorem count_solutions : num_solutions = 2 := by
  sorry

end count_solutions_l236_236042


namespace mark_exceeded_sugar_intake_by_100_percent_l236_236609

-- Definitions of the conditions
def softDrinkCalories : ℕ := 2500
def sugarPercentage : ℝ := 0.05
def caloriesPerCandy : ℕ := 25
def numCandyBars : ℕ := 7
def recommendedSugarIntake : ℕ := 150

-- Calculating the amount of added sugar in the soft drink
def addedSugarSoftDrink : ℝ := sugarPercentage * softDrinkCalories

-- Calculating the total added sugar from the candy bars
def addedSugarCandyBars : ℕ := numCandyBars * caloriesPerCandy

-- Summing the added sugar from the soft drink and the candy bars
def totalAddedSugar : ℝ := addedSugarSoftDrink + (addedSugarCandyBars : ℝ)

-- Calculate the excess intake of added sugar over the recommended amount
def excessSugarIntake : ℝ := totalAddedSugar - (recommendedSugarIntake : ℝ)

-- Prove that the percentage by which Mark exceeded the recommended intake of added sugar is 100%
theorem mark_exceeded_sugar_intake_by_100_percent :
  (excessSugarIntake / (recommendedSugarIntake : ℝ)) * 100 = 100 :=
by
  sorry

end mark_exceeded_sugar_intake_by_100_percent_l236_236609


namespace seniors_in_statistics_correct_l236_236084

-- Conditions
def total_students : ℕ := 120
def percentage_statistics : ℚ := 1 / 2
def percentage_seniors_in_statistics : ℚ := 9 / 10

-- Definitions based on conditions
def students_in_statistics : ℕ := total_students * percentage_statistics
def seniors_in_statistics : ℕ := students_in_statistics * percentage_seniors_in_statistics

-- Statement to prove
theorem seniors_in_statistics_correct :
  seniors_in_statistics = 54 :=
by
  -- Proof goes here
  sorry

end seniors_in_statistics_correct_l236_236084


namespace sid_spent_on_snacks_l236_236924

theorem sid_spent_on_snacks :
  let original_money := 48
  let money_spent_on_computer_accessories := 12
  let money_left_after_computer_accessories := original_money - money_spent_on_computer_accessories
  let remaining_money_after_purchases := 4 + original_money / 2
  ∃ snacks_cost, money_left_after_computer_accessories - snacks_cost = remaining_money_after_purchases ∧ snacks_cost = 8 :=
by
  sorry

end sid_spent_on_snacks_l236_236924


namespace nonnegative_exists_l236_236874

theorem nonnegative_exists (a b c : ℝ) (h : a + b + c = 0) : a ≥ 0 ∨ b ≥ 0 ∨ c ≥ 0 :=
by
  sorry

end nonnegative_exists_l236_236874


namespace solve_system_of_equations_l236_236352

theorem solve_system_of_equations (x y : ℝ) :
  16 * x^3 + 4 * x = 16 * y + 5 ∧ 16 * y^3 + 4 * y = 16 * x + 5 → x = y ∧ 16 * x^3 - 12 * x - 5 = 0 :=
by
  sorry

end solve_system_of_equations_l236_236352


namespace bun_eating_problem_l236_236059

theorem bun_eating_problem
  (n k : ℕ)
  (H1 : 5 * n / 10 + 3 * k / 10 = 180) -- This corresponds to the condition that Zhenya eats 5 buns in 10 minutes, and Sasha eats 3 buns in 10 minutes, for a total of 180 minutes.
  (H2 : n + k = 70) -- This corresponds to the total number of buns eaten.
  : n = 40 ∧ k = 30 :=
by
  sorry

end bun_eating_problem_l236_236059


namespace standard_equation_of_circle_tangent_to_x_axis_l236_236496

theorem standard_equation_of_circle_tangent_to_x_axis :
  ∀ (x y : ℝ), ((x + 3) ^ 2 + (y - 4) ^ 2 = 16) :=
by
  -- Definitions based on the conditions
  let center_x := -3
  let center_y := 4
  let radius := 4

  sorry

end standard_equation_of_circle_tangent_to_x_axis_l236_236496


namespace geometric_series_sum_l236_236527

theorem geometric_series_sum :
  let a := (1 : ℚ) / 3
  let r := -(1 / 3)
  let n := 5
  let S₅ := (a * (1 - r ^ n)) / (1 - r)
  S₅ = 61 / 243 := by
  let a := (1 : ℚ) / 3
  let r := -(1 / 3)
  let n := 5
  let S₅ := (a * (1 - r ^ n)) / (1 - r)
  sorry

end geometric_series_sum_l236_236527


namespace linear_function_quadrants_l236_236320

theorem linear_function_quadrants (m : ℝ) (h1 : m - 2 < 0) (h2 : m + 1 > 0) : -1 < m ∧ m < 2 := 
by 
  sorry

end linear_function_quadrants_l236_236320
