import Mathlib

namespace complex_conjugate_multiplication_l362_362261

def z : ℂ := complex.I + 1
def z_conjugate : ℂ := 1 - complex.I

theorem complex_conjugate_multiplication (h : z = complex.I + 1) : z * z_conjugate = 2 := 
by 
-- proof goes here
sorry

end complex_conjugate_multiplication_l362_362261


namespace gasoline_tank_capacity_l362_362151

theorem gasoline_tank_capacity (x : ℕ) (h1 : 5 * x / 6 - 2 * x / 3 = 15) : x = 90 :=
sorry

end gasoline_tank_capacity_l362_362151


namespace sum_of_first_33_terms_arith_seq_l362_362602

noncomputable def sum_arith_prog (a_1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a_1 + (n - 1) * d)

theorem sum_of_first_33_terms_arith_seq :
  ∃ (a_1 d : ℝ), (4 * a_1 + 64 * d = 28) → (sum_arith_prog a_1 d 33 = 231) :=
by
  sorry

end sum_of_first_33_terms_arith_seq_l362_362602


namespace arrange_knights_l362_362198

def is_knight (p : ℕ × ℕ) := true -- simplified representation for position on the board

-- Checks if two knights attack each other
def attacks (p q : ℕ × ℕ) : Prop :=
  (|p.1 - q.1| = 2 ∧ |p.2 - q.2| = 1) ∨ (|p.1 - q.1| = 1 ∧ |p.2 - q.2| = 2)

-- Condition for each knight to attack exactly two other knights
def knights_condition (knights : list (ℕ × ℕ)) : Prop :=
  ∀ k ∈ knights, (list.filter (λ q, attacks k q) knights).length = 2

-- Proposition that 32 knights can be arranged properly
theorem arrange_knights :
  ∃ knights : list (ℕ × ℕ), knights.length = 32 ∧ knights_condition knights :=
sorry

end arrange_knights_l362_362198


namespace find_n_l362_362458

theorem find_n :
  ∃ (n : ℤ), 50 ≤ n ∧ n ≤ 120 ∧ (n % 8 = 0) ∧ (n % 7 = 5) ∧ (n % 6 = 3) ∧ n = 208 := 
by {
  sorry
}

end find_n_l362_362458


namespace max_composite_numbers_l362_362788

theorem max_composite_numbers (S : Finset ℕ) (h1 : ∀ n ∈ S, n < 1500) (h2 : ∀ m n ∈ S, m ≠ n → Nat.gcd m n = 1) : S.card ≤ 12 := sorry

end max_composite_numbers_l362_362788


namespace largest_divisor_of_square_difference_l362_362403

theorem largest_divisor_of_square_difference (m n : ℤ) (hm : m % 2 = 0) (hn : n % 2 = 0) (h : n < m) : 
  ∃ d, ∀ m n, (m % 2 = 0) → (n % 2 = 0) → (n < m) → d ∣ (m^2 - n^2) ∧ ∀ k, (∀ m n, (m % 2 = 0) → (n % 2 = 0) → (n < m) → k ∣ (m^2 - n^2)) → k ≤ d :=
sorry

end largest_divisor_of_square_difference_l362_362403


namespace total_cards_l362_362945

theorem total_cards (H F B : ℕ) (hH : H = 200) (hF : F = 4 * H) (hB : B = F - 50) : H + F + B = 1750 := 
by 
  sorry

end total_cards_l362_362945


namespace concurrency_of_lines_l362_362735

variables {C1 C2 : Type} -- Circles
variables {O1 O2 : Type} -- Centers of the circles
variables {A1 A2 : Type} -- External common tangent points
variables {B1 B2 : Type} -- Internal common tangent points

-- Assume circles C1 and C2 are external to each other with centers O1 and O2 respectively
axiom circles_external (C1 C2 : Type) (O1 O2 : Type) : Prop

-- Assume there is an external common tangent touching C1 at A1 and C2 at A2
axiom external_common_tangent (C1 C2 : Type) (A1 A2 : Type) : Prop

-- Assume there is an internal common tangent touching C1 at B1 and C2 at B2
axiom internal_common_tangent (C1 C2 : Type) (B1 B2 : Type) : Prop

theorem concurrency_of_lines
  (h1 : circles_external C1 C2 O1 O2)
  (h2 : external_common_tangent C1 C2 A1 A2)
  (h3 : internal_common_tangent C1 C2 B1 B2) :
  ∃ P : Type, Collinear P (Segment A1 B1) ∧ Collinear P (Segment A2 B2) ∧ Collinear P (Segment O1 O2) :=
sorry

end concurrency_of_lines_l362_362735


namespace tan_alpha_eq_2_l362_362257

theorem tan_alpha_eq_2 (α : ℝ) (h : Real.tan α = 2) : (Real.cos α + 3 * Real.sin α) / (3 * Real.cos α - Real.sin α) = 7 := by
  sorry

end tan_alpha_eq_2_l362_362257


namespace who_arrives_first_l362_362152

variables (D V_a V_b : ℝ)
variables (t_c t_m : ℝ)

-- Definitions based on conditions
def cyclist_one_third_distance := D / 3
def motorist_two_third_distance := 2 * D / 3

-- Use the equality of the times to find the relation between speeds
def speed_relation := (D / 3) / V_b = (2 * D / 3) / V_a

-- Substituting the given relation V_a = 2 * V_b
theorem who_arrives_first (h : V_a = 2 * V_b) : (2 * D / (3 * V_b)) < (D / (2 * V_b)) :=
by simp [h]; linarith

#check who_arrives_first

end who_arrives_first_l362_362152


namespace archie_antibiotics_l362_362197

theorem archie_antibiotics : 
  ∀ (cost_per_antibiotic total_cost days_in_week : ℕ),
  cost_per_antibiotic = 3 →
  total_cost = 63 →
  days_in_week = 7 →
  (total_cost / cost_per_antibiotic) / days_in_week = 3 :=
by
  intros cost_per_antibiotic total_cost days_in_week
  assume h1 : cost_per_antibiotic = 3
  assume h2 : total_cost = 63
  assume h3 : days_in_week = 7
  sorry

end archie_antibiotics_l362_362197


namespace max_oleg_composite_numbers_l362_362751

/-- Oleg's list of composite numbers less than 1500 and pairwise coprime. --/
def oleg_composite_numbers (numbers : List ℕ) : Prop :=
  ∀ n ∈ numbers, Nat.isComposite n ∧ n < 1500 ∧ (∀ m ∈ numbers, n ≠ m → Nat.gcd n m = 1)

/-- Maximum number of such numbers that Oleg could write. --/
theorem max_oleg_composite_numbers : ∃ numbers : List ℕ, oleg_composite_numbers numbers ∧ numbers.length = 12 := 
sorry

end max_oleg_composite_numbers_l362_362751


namespace triangle_transformation_complex_shape_l362_362705

def point := (ℝ × ℝ) 

def triangle (O A B : point) : Prop :=
  O = (0, 0) ∧ A = (2, 0) ∧ B = (0, 2)

noncomputable def transform (p : point) : point :=
  (p.1^2 - p.2^2, p.1 * p.2)

theorem triangle_transformation_complex_shape :
  ∀ (O A B : point), triangle O A B → 
  let O' := transform O in
  let A' := transform A in
  let B' := transform B in
  ∃ (C : Prop), C :=
sorry

end triangle_transformation_complex_shape_l362_362705


namespace total_savings_l362_362548

-- Definition to specify the denomination of each bill
def bill_value : ℕ := 100

-- Condition: Number of $100 bills Michelle has
def num_bills : ℕ := 8

-- The theorem to prove the total savings amount
theorem total_savings : num_bills * bill_value = 800 :=
by
  sorry

end total_savings_l362_362548


namespace probability_A_selected_l362_362828

def n : ℕ := 5
def k : ℕ := 2

def total_ways : ℕ := Nat.choose n k  -- C(n, k)

def favorable_ways : ℕ := Nat.choose (n - 1) (k - 1)  -- C(n-1, k-1)

theorem probability_A_selected : (favorable_ways : ℚ) / (total_ways : ℚ) = 2 / 5 :=
by
  sorry

end probability_A_selected_l362_362828


namespace num_words_with_A_l362_362313

theorem num_words_with_A :
  let total_words := 5^4,
      words_without_A := 4^4 in
  total_words - words_without_A = 369 :=
by
  sorry

end num_words_with_A_l362_362313


namespace add_base_6_l362_362566

theorem add_base_6 (a b c : ℕ) (h₀ : a = 3 * 6^3 + 4 * 6^2 + 2 * 6 + 1)
                    (h₁ : b = 4 * 6^3 + 5 * 6^2 + 2 * 6 + 5)
                    (h₂ : c = 1 * 6^4 + 2 * 6^3 + 3 * 6^2 + 5 * 6 + 0) : 
  a + b = c :=
by  
  sorry

end add_base_6_l362_362566


namespace matrix_ones_bound_l362_362186

theorem matrix_ones_bound
  (n : ℕ) 
  (A : Matrix (Fin (n^2 + n + 1)) (Fin (n^2 + n + 1)) (Fin 2))
  (h_no_rect : ∀ i1 i2 j1 j2, i1 ≠ i2 → j1 ≠ j2 → (A i1 j1 = 1) → (A i1 j2 = 1) → (A i2 j1 = 1) → (A i2 j2 = 1) → False):
  (∑ i j, A i j) ≤ (n + 1) * (n^2 + n + 1) :=
sorry

end matrix_ones_bound_l362_362186


namespace major_axis_length_is_three_l362_362157

-- Given the radius of the cylinder
def cylinder_radius : ℝ := 1

-- Given the percentage longer of the major axis than the minor axis
def percentage_longer (r : ℝ) : ℝ := 1.5

-- Given the function to calculate the minor axis using the radius
def minor_axis (r : ℝ) : ℝ := 2 * r

-- Given the function to calculate the major axis using the minor axis
def major_axis (minor_axis : ℝ) (factor : ℝ) : ℝ := minor_axis * factor

-- The conjecture states that the major axis length is 3
theorem major_axis_length_is_three : 
  major_axis (minor_axis cylinder_radius) (percentage_longer cylinder_radius) = 3 :=
by 
  -- Proof goes here
  sorry

end major_axis_length_is_three_l362_362157


namespace find_EF_l362_362935

theorem find_EF (h_sim : Similar (triangle A B C) (triangle D E F))
  (h_BC : BC = 8)
  (h_BA : BA = 5)
  (h_ED : ED = 3) :
  EF = 4.8 :=
sorry

end find_EF_l362_362935


namespace calculate_expression_l362_362998

theorem calculate_expression :
  150 * (150 - 4) - (150 * 150 - 8 + 2^3) = -600 :=
by
  sorry

end calculate_expression_l362_362998


namespace median_possible_values_l362_362706

variable {ι : Type} -- Representing the set S as a type
variable (S : Finset ℤ) -- S is a finite set of integers

def conditions (S: Finset ℤ) : Prop :=
  S.card = 9 ∧
  {5, 7, 10, 13, 17, 21} ⊆ S

theorem median_possible_values :
  ∀ S : Finset ℤ, conditions S → ∃ medians : Finset ℤ, medians.card = 7 :=
by
  sorry

end median_possible_values_l362_362706


namespace problem2_l362_362349

noncomputable def problem1 (a b c : ℝ) (A B C : ℝ) (h1 : 2 * (Real.sin A)^2 + (Real.sin B)^2 = (Real.sin C)^2)
    (h2 : b = 2 * a) (h3 : a = 2) : (1 / 2) * a * b * Real.sin C = Real.sqrt 15 :=
by
  sorry

theorem problem2 (a b c : ℝ) (h : 2 * a^2 + b^2 = c^2) :
  ∃ m : ℝ, (m = 2 * Real.sqrt 2) ∧ (∀ x y z : ℝ, 2 * x^2 + y^2 = z^2 → (z^2 / (x * y)) ≥ m) ∧ ((c / a) = 2) :=
by
  sorry

end problem2_l362_362349


namespace possible_values_g_l362_362726

theorem possible_values_g (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let g := (λ (a b c : ℝ), (a / (a + b)) + (b / (b + c)) + (c / (c + a))) in
  set_of (λ x, ∃ a b c, 0 < a ∧ 0 < b ∧ 0 < c ∧ g a b c = x) = {x | 1 < x ∧ x < 2} :=
by
  sorry

end possible_values_g_l362_362726


namespace find_two_digit_number_l362_362542

theorem find_two_digit_number (n : ℕ) (h1 : 10 ≤ n ∧ n < 100)
  (h2 : n % 2 = 0)
  (h3 : (n + 1) % 3 = 0)
  (h4 : (n + 2) % 4 = 0)
  (h5 : (n + 3) % 5 = 0) : n = 62 :=
by
  sorry

end find_two_digit_number_l362_362542


namespace minimize_expression_l362_362672

theorem minimize_expression : 
  let a := -1
  let b := -0.5
  (a + b) ≤ (a - b) ∧ (a + b) ≤ (a * b) ∧ (a + b) ≤ (a / b) := by
  let a := -1
  let b := -0.5
  sorry

end minimize_expression_l362_362672


namespace range_of_k_l362_362641

noncomputable def function_y (x k : ℝ) : ℝ := x^2 + (1 - k) * x - k

theorem range_of_k (k : ℝ) (h : ∃ x ∈ Ioo (2:ℝ) 3, function_y x k = 0) : 2 < k ∧ k < 3 := 
by
  sorry

end range_of_k_l362_362641


namespace likely_temperature_reading_l362_362872

noncomputable def temperature_reading (T : ℝ) : Prop :=
  34.0 < T ∧ T < 34.5

theorem likely_temperature_reading : ∃ T : ℝ, temperature_reading T ∧ T = 34.5 := 
by 
  apply exists.intro 34.5
  sorry

end likely_temperature_reading_l362_362872


namespace sqrt_product_l362_362098

theorem sqrt_product (a b : ℝ) (h1 : a = 49) (h2 : b = 25) : sqrt (a * sqrt b) = 7 * sqrt 5 :=
by sorry

end sqrt_product_l362_362098


namespace min_value_l362_362742

noncomputable def a1 : ℝ := 1

noncomputable def a2 (r : ℝ) : ℝ := r

noncomputable def a3 (r : ℝ) : ℝ := r^2

def expression (r : ℝ) : ℝ := 3 * a2 r + 7 * a3 r

theorem min_value : ∃ r : ℝ, expression r = -9 / 196 :=
  sorry

end min_value_l362_362742


namespace rectangle_cut_l362_362221

theorem rectangle_cut :
  ∃ (a b c d e : ℕ), a + b + c + d + e = 30 ∧
                     (a = 4 ∨ a = 5 ∨ a = 6 ∨ a = 7 ∨ a = 8) ∧
                     (b = 4 ∨ b = 5 ∨ b = 6 ∨ b = 7 ∨ b = 8) ∧
                     (c = 4 ∨ c = 5 ∨ c = 6 ∨ c = 7 ∨ c = 8) ∧
                     (d = 4 ∨ d = 5 ∨ d = 6 ∨ d = 7 ∨ d = 8) ∧
                     (e = 4 ∨ e = 5 ∨ e = 6 ∨ e = 7 ∨ e = 8) ∧
                     (a + 1 = b ∨ a + 1 = c ∨ a + 1 = d ∨ a + 1 = e) ∧
                     (b + 1 = c ∨ b + 1 = d ∨ b + 1 = e) ∧
                     (c + 1 = d ∨ c + 1 = e) ∧
                     (d + 1 = e) ∧
                     (8 = 2 * 4) ∧ (10 = 2 * 5) ∧ (12 = 2 * 6) ∧ (14 = 2 * 7) ∧ (16 = 2 * 8) ∧
                     (2*a = a*4 ∨ 2*a = a*5∨ 2*a = a*6∨ 2*a = a*7∨ 2*a = a*8) ∧
                     (a = 4 ∧ b = 5 ∧ c = 6 ∧ d = 7 ∧ e = 8)  :=
begin
  sorry
end

end rectangle_cut_l362_362221


namespace women_percentage_l362_362995

def percentWomen (E W M : ℕ) : ℚ :=
  (W: ℚ) / E * 100

theorem women_percentage (E W M : ℚ)
  (h_total : E = W + M)
  (h_married_employees : 0.60 * E)
  (h_married_men : (1/3) * M)
  (h_married_women : 0.7704918032786885 * W) :
  percentWomen E W M = 61.01694915254237 := sorry

end women_percentage_l362_362995


namespace real_number_condition_complex_number_condition_purely_imaginary_condition_l362_362259

variable (m : ℝ)

def is_real (m : ℝ) :=
  let z := (m * (m + 2)) / (m - 1) + (m^2 + 2 * m - 3) * I
  (m^2 + 2 * m - 3 = 0) ∧ (m - 1 ≠ 0)

def is_complex (m : ℝ) :=
  let z := (m * (m + 2)) / (m - 1) + (m^2 + 2 * m - 3) * I
  (m * (m + 2) = 0) ∧ (m - 1 ≠ 0)

def is_purely_imaginary (m : ℝ) :=
  let z := (m * (m + 2)) / (m - 1) + (m^2 + 2 * m - 3) * I
  (m * (m + 2) = 0) ∧ (m - 1 ≠ 0) ∧ (m^2 + 2 * m - 3 ≠ 0)

theorem real_number_condition : is_real m → m = -3 := by
  sorry

theorem complex_number_condition : is_complex m → (m = 0 ∨ m = -2) := by
  sorry

theorem purely_imaginary_condition : is_purely_imaginary m → (m = 0 ∨ m = -2) := by
  sorry

end real_number_condition_complex_number_condition_purely_imaginary_condition_l362_362259


namespace negation_of_universal_l362_362882

theorem negation_of_universal :
  ¬ (∀ x : ℝ, x ∈ set.Ici (-2) → x + 3 ≥ 1) ↔ (∃ x : ℝ, x ∈ set.Ici (-2) ∧ x + 3 < 1) :=
by
  sorry

end negation_of_universal_l362_362882


namespace max_composite_numbers_with_gcd_one_l362_362761

theorem max_composite_numbers_with_gcd_one : 
  ∃ (S : Finset ℕ), 
    (∀ x ∈ S, Nat.isComposite x) ∧ 
    (∀ x ∈ S, x < 1500) ∧ 
    (∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y → Nat.gcd x y = 1) ∧
    S.card = 12 :=
sorry

end max_composite_numbers_with_gcd_one_l362_362761


namespace pool_water_removal_l362_362472

theorem pool_water_removal
  (length width lower_by : ℝ)
  (conversion_factor : ℝ)
  (length_eq : length = 60)
  (width_eq : width = 10)
  (lower_by_eq : lower_by = 0.5)
  (conversion_factor_eq : conversion_factor = 7.5) :
  (length * width * lower_by * conversion_factor = 2250) :=
by {
  rw [length_eq, width_eq, lower_by_eq, conversion_factor_eq],
  norm_num,
}

end pool_water_removal_l362_362472


namespace tank_capacity_l362_362148

theorem tank_capacity
  (x : ℝ) -- define x as the full capacity of the tank in gallons
  (h1 : (5/6) * x - (2/3) * x = 15) -- first condition
  (h2 : (2/3) * x = y) -- second condition, though not actually needed
  : x = 90 := 
by sorry

end tank_capacity_l362_362148


namespace sqrt_mul_sqrt_l362_362046

theorem sqrt_mul_sqrt (a b c : ℝ) (hac : a = 49) (hbc : b = sqrt 25) (hc : c = 7 * sqrt 5) : sqrt (a * b) = c := 
by
  sorry

end sqrt_mul_sqrt_l362_362046


namespace find_original_three_digit_number_l362_362175

theorem find_original_three_digit_number 
  (n : ℕ) 
  (h1 : n % 10 = 2) 
  (h2 : (2 * 10 ^ (nat.log 10 n) + n / 10) = n + 18) : 
  n = 202 := 
sorry

end find_original_three_digit_number_l362_362175


namespace probability_of_two_germinates_is_48_over_125_l362_362138

noncomputable def probability_of_exactly_two_germinates : ℚ :=
  let p := 4/5
  let n := 3
  let k := 2
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_of_two_germinates_is_48_over_125 :
  probability_of_exactly_two_germinates = 48/125 := by
    sorry

end probability_of_two_germinates_is_48_over_125_l362_362138


namespace sqrt_mul_sqrt_l362_362051

theorem sqrt_mul_sqrt (a b c : ℝ) (hac : a = 49) (hbc : b = sqrt 25) (hc : c = 7 * sqrt 5) : sqrt (a * b) = c := 
by
  sorry

end sqrt_mul_sqrt_l362_362051


namespace age_ratio_correct_l362_362966

noncomputable def age_ratio_mother_to_daughter_a_year_ago (m : ℕ) (d : ℕ) : Prop :=
  m = 55 ∧ (m - d) = 27 ∧ ((m - 1) / (d - 1)) = 2

theorem age_ratio_correct : age_ratio_mother_to_daughter_a_year_ago 55 28 :=
by {
  -- Given
  have h1: 55 = 55, from rfl,
  have h2: 55 - 28 = 27, by norm_num,
  have h3: (54 / 27) = 2, by norm_num,

  -- Combine
  exact ⟨h1, h2, h3⟩
}

end age_ratio_correct_l362_362966


namespace sqrt_49_times_sqrt_25_eq_7sqrt5_l362_362071

theorem sqrt_49_times_sqrt_25_eq_7sqrt5 :
  (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  have h1 : Real.sqrt 25 = 5 := by sorry
  have h2 : 49 * 5 = 245 := by sorry
  have h3 : Real.sqrt 245 = 7 * Real.sqrt 5 := by sorry
  sorry

end sqrt_49_times_sqrt_25_eq_7sqrt5_l362_362071


namespace good_quadruple_inequality_l362_362821

theorem good_quadruple_inequality {p a b c : ℕ} (hp : Nat.Prime p) (hodd : p % 2 = 1) 
(habc_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
(hab : (a * b + 1) % p = 0) (hbc : (b * c + 1) % p = 0) (hca : (c * a + 1) % p = 0) :
  p + 2 ≤ (a + b + c) / 3 := 
by
  sorry

end good_quadruple_inequality_l362_362821


namespace prob_40_lt_xi_lt_60_l362_362162

variables (ξ : ℝ → ℝ) (σ : ℝ)

noncomputable theory
def normal_dist (μ σ : ℝ) := λ x : ℝ, (1 / (σ * sqrt (2 * π))) * exp (-(x - μ)^2 / (2 * σ^2))

axiom xi_is_normal : ξ ~ normal_dist 50 σ

axiom prob_xi_lt_40 : ∀ p : ℝ, P(λ x, ξ x < 40) = 0.3

theorem prob_40_lt_xi_lt_60 : P(λ x, 40 < ξ x ∧ ξ x < 60) = 0.4 :=
sorry

end prob_40_lt_xi_lt_60_l362_362162


namespace lena_savings_l362_362699

theorem lena_savings
  (original_markers : ℕ) (marker_price : ℝ) (discount_rate : ℝ) (deal_markers : ℕ)
  (original_total_cost : ℝ) (discounted_price : ℝ) (discounted_cost : ℝ) (savings : ℝ) :
  original_markers = 8 →
  marker_price = 3.00 →
  discount_rate = 0.30 →
  deal_markers = 4 →
  original_total_cost = original_markers * marker_price →
  discounted_price = marker_price * (1 - discount_rate) →
  let total_markers := original_markers + (original_markers / deal_markers) in
  discounted_cost = original_markers * discounted_price →
  savings = original_total_cost - discounted_cost →
  savings = 7.20 :=
by
  intros
  repeat { sorry }

end lena_savings_l362_362699


namespace greatest_integer_ln_l362_362683

theorem greatest_integer_ln (LM L N O P: Type) (hLM: ∥L - M∥ = 120) (hMid: ∥P - L∥ = ∥P - N∥) 
  (hPerp: ∠ (LO) = 90°) : 
  let LN := 120 * Real.sqrt 2
  greatest_integer_less_ln := 169 := 
by 
  sorry

end greatest_integer_ln_l362_362683


namespace original_number_doubled_added_trebled_l362_362154

theorem original_number_doubled_added_trebled (x : ℤ) : 3 * (2 * x + 9) = 75 → x = 8 :=
by
  intro h
  -- The proof is omitted as instructed.
  sorry

end original_number_doubled_added_trebled_l362_362154


namespace diff_extrema_eq_4_l362_362639

-- Define the function f(x)
def f (x a b c : ℝ) := x^3 + 3*a*x^2 + 3*b*x + c

noncomputable def diff_between_extrema (a b c : ℝ) : ℝ := 
  let f' (x : ℝ) := 3*x^2 + 6*a*x + 3*b
  if (f' 2 = 0) ∧ (f' 1 = -3) then 
    4 
  else
    sorry

-- State the theorem
theorem diff_extrema_eq_4 (a b c : ℝ) :
  (f (2 : ℝ) a b c - f (0 : ℝ) a b c) = -4 ∧ (f' a b c 1 = -3) → diff_between_extrema a b c = 4 :=
begin
  sorry
end

end diff_extrema_eq_4_l362_362639


namespace sin_cos_identity_l362_362612

theorem sin_cos_identity (θ a b : ℝ) (h1 : sin θ + cos θ = a) (h2 : sin θ - cos θ = b) : a^2 + b^2 = 2 :=
sorry

end sin_cos_identity_l362_362612


namespace sonia_probability_cups_l362_362351

theorem sonia_probability_cups :
  let cups := ["white", "white", "white", "red", "red", "red", "black", "black"] in
  let selected := ["white", "white", "red", "red", "black"] in
  let all_selections := list.permutations cups in
  let desired_selection := selected.permutations in
  let probability := (list.length desired_selection : ℝ) / (list.length all_selections : ℝ) in
  real.to_nnreal (probability) ≈ 0.32 :=
by
sry

end sonia_probability_cups_l362_362351


namespace cars_cannot_meet_l362_362951

-- Define the directions
inductive Direction
| straight
| turn_right_120_deg
| turn_left_120_deg

-- Define the car's movement at an intersection
structure Car :=
  (start_location : Point)
  (directions : List Direction)

def canMeet (car1 car2 : Car) : Prop :=
  let new_position (car : Car) (time : Nat) : Point := sorry -- Implementation of new position calculation based on time and directions
  let car1_position := new_position car1
  let car2_position := new_position car2
  ∀ t : Nat, car1_position t ≠ car2_position t

theorem cars_cannot_meet (carA carB : Car) (h_same_speed : ∀ t : Nat, (t > 0) ->  distance (carA.start_location) (carB.start_location) ≠ 0) :
  ¬ canMeet carA carB :=
  by
  -- high-level idea: use the distinct paths and timings to show impossibility of same position at the same time
  sorry

end cars_cannot_meet_l362_362951


namespace probability_kwoes_non_intersect_breads_l362_362202

-- Define the total number of ways to pick 3 points from 7
def total_combinations : ℕ := Nat.choose 7 3

-- Define the number of ways to pick 3 consecutive points from 7
def favorable_combinations : ℕ := 7

-- Define the probability of non-intersection
def non_intersection_probability : ℚ := favorable_combinations / total_combinations

-- Assert the final required probability
theorem probability_kwoes_non_intersect_breads :
  non_intersection_probability = 1 / 5 :=
by
  sorry

end probability_kwoes_non_intersect_breads_l362_362202


namespace blue_square_area_percentage_l362_362171

theorem blue_square_area_percentage (k : ℝ) (H1 : 0 < k) 
(Flag_area : ℝ := k^2) -- total area of the flag
(Cross_area : ℝ := 0.49 * Flag_area) -- total area of the cross and blue squares 
(one_blue_square_area : ℝ := Cross_area / 3) -- area of one blue square
(percentage : ℝ := one_blue_square_area / Flag_area * 100) :
percentage = 16.33 :=
by
  sorry

end blue_square_area_percentage_l362_362171


namespace difference_SP_l362_362553

-- Definitions for amounts
variables (P Q R S : ℕ)

-- Conditions given in the problem
def total_amount := P + Q + R + S = 1000
def P_condition := P = 2 * Q
def S_condition := S = 4 * R
def Q_R_equal := Q = R

-- Statement of the problem that needs to be proven
theorem difference_SP (P Q R S : ℕ) (h1 : total_amount P Q R S) 
  (h2 : P_condition P Q) (h3 : S_condition S R) (h4 : Q_R_equal Q R) : 
  S - P = 250 :=
by 
  sorry

end difference_SP_l362_362553


namespace solution_set_of_inequality_l362_362889

theorem solution_set_of_inequality (x : ℝ) :
  2 * x ≤ -1 → x > -1 → -1 < x ∧ x ≤ -1 / 2 :=
by
  intro h1 h2
  have h3 : x ≤ -1 / 2 := by linarith
  exact ⟨h2, h3⟩

end solution_set_of_inequality_l362_362889


namespace max_composite_numbers_l362_362814
open Nat

theorem max_composite_numbers : 
  ∃ X : Finset Nat, 
  (∀ x ∈ X, x < 1500 ∧ ¬Prime x) ∧ 
  (∀ x y ∈ X, x ≠ y → gcd x y = 1) ∧ 
  X.card = 12 := 
sorry

end max_composite_numbers_l362_362814


namespace minimize_volume_at_lambda_one_l362_362905

noncomputable def minimize_volume (a λ : ℝ) : Prop :=
let p := λ / (λ + 1) * a in
let q := 1 / (λ + 1) * a in
let volume_removed := (2 / 3) * p * q * a in
∀ λ, volume_removed λ ≥ volume_removed 1

theorem minimize_volume_at_lambda_one (a : ℝ) (h : 0 < a) : minimize_volume a 1 :=
by
  intros λ
  sorry

end minimize_volume_at_lambda_one_l362_362905


namespace files_missing_l362_362229

theorem files_missing (initial_files : ℕ) (morning_files : ℕ) (afternoon_files : ℕ) :
  initial_files = 60 →
  morning_files = initial_files / 2 →
  afternoon_files = 15 →
  initial_files - (morning_files + afternoon_files) = 15 :=
by
  intros h_initial h_morning h_afternoon
  rw [h_initial, h_morning, h_afternoon]
  have half_of_60 : 60 / 2 = 30 := by norm_num
  rw half_of_60
  norm_num
  sorry

end files_missing_l362_362229


namespace smallest_possible_odd_b_l362_362677

theorem smallest_possible_odd_b 
    (a b : ℕ) 
    (h1 : a + b = 90) 
    (h2 : Nat.Prime a) 
    (h3 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ b) 
    (h4 : a > b) 
    (h5 : b % 2 = 1) 
    : b = 85 := 
sorry

end smallest_possible_odd_b_l362_362677


namespace total_cards_l362_362944

theorem total_cards (H F B : ℕ) (hH : H = 200) (hF : F = 4 * H) (hB : B = F - 50) : H + F + B = 1750 := 
by 
  sorry

end total_cards_l362_362944


namespace find_m_l362_362407

-- Define the condition of arithmetic sequences sums
constant S : ℕ → ℚ
constant T : ℕ → ℚ

-- Hypotheses
axiom h1 : ∀ n : ℕ, n > 0 → S n / T n = (2 * n + 6) / (n + 1)
axiom h2 : ∀ m : ℕ, m > 0 → ∃ k : ℕ, T m = k * m * (m + 1)

-- Objective to prove
theorem find_m (m : ℕ) (hm : m = 2) (prime_bm : ∃ p : ℕ, p.prime ∧ p = m) : b m ∧ hm := by
  sorry

end find_m_l362_362407


namespace max_rectangle_area_squared_l362_362128

theorem max_rectangle_area_squared 
  (x y : ℝ) (h1 : abs (y - x) = (y + x + 1) * (5 - x - y))
  (h2 : parallel_to_lines : (∃ y = x, ∃ y = -x)) : 
  ∃ A : ℝ, (A^2 = 432) :=
sorry

end max_rectangle_area_squared_l362_362128


namespace sqrt_mul_sqrt_l362_362105

theorem sqrt_mul_sqrt (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : sqrt (a * sqrt b) = sqrt a * sqrt (sqrt b) := by
  sorry

example : sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  have h1 : sqrt 25 = 5 := by 
    norm_num
  have h2 : sqrt (49 * 5) = sqrt 245 := by  
    unfold sqrt
  have h3 : sqrt 245 = 7 * sqrt 5 := by 
    norm_num
  rw [h1, h2, h3]
  norm_num

end sqrt_mul_sqrt_l362_362105


namespace sqrt_49_times_sqrt_25_l362_362028

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 :=
by
  sorry

end sqrt_49_times_sqrt_25_l362_362028


namespace original_price_l362_362117

theorem original_price (total_payment : ℝ) (num_units : ℕ) (discount_rate : ℝ) 
(h1 : total_payment = 500) (h2 : num_units = 18) (h3 : discount_rate = 0.20) : 
  (total_payment / (1 - discount_rate) * num_units) = 625.05 :=
by
  sorry

end original_price_l362_362117


namespace compound_oxygen_atoms_l362_362143

theorem compound_oxygen_atoms (H C O : Nat) (mw : Nat) (H_weight C_weight O_weight : Nat) 
  (h_H : H = 2)
  (h_C : C = 1)
  (h_mw : mw = 62)
  (h_H_weight : H_weight = 1)
  (h_C_weight : C_weight = 12)
  (h_O_weight : O_weight = 16)
  : O = 3 :=
by
  sorry

end compound_oxygen_atoms_l362_362143


namespace max_oleg_composite_numbers_l362_362758

/-- Oleg's list of composite numbers less than 1500 and pairwise coprime. --/
def oleg_composite_numbers (numbers : List ℕ) : Prop :=
  ∀ n ∈ numbers, Nat.isComposite n ∧ n < 1500 ∧ (∀ m ∈ numbers, n ≠ m → Nat.gcd n m = 1)

/-- Maximum number of such numbers that Oleg could write. --/
theorem max_oleg_composite_numbers : ∃ numbers : List ℕ, oleg_composite_numbers numbers ∧ numbers.length = 12 := 
sorry

end max_oleg_composite_numbers_l362_362758


namespace probability_of_yellow_ball_l362_362477

theorem probability_of_yellow_ball 
  (red_balls : ℕ) 
  (yellow_balls : ℕ) 
  (blue_balls : ℕ) 
  (total_balls : ℕ)
  (h1 : red_balls = 2)
  (h2 : yellow_balls = 5)
  (h3 : blue_balls = 4)
  (h4 : total_balls = red_balls + yellow_balls + blue_balls) :
  (yellow_balls / total_balls : ℚ) = 5 / 11 :=
by 
  rw [h1, h2, h3] at h4  -- Substitute the ball counts into the total_balls definition.
  norm_num at h4  -- Simplify to verify the total is indeed 11.
  rw [h2, h4] -- Use the number of yellow balls and total number of balls to state the ratio.
  norm_num -- Normalize the fraction to show it equals 5/11.

#check probability_of_yellow_ball

end probability_of_yellow_ball_l362_362477


namespace range_of_m_l362_362339

theorem range_of_m (m : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - m * x - m < 0) ↔ (-4 ≤ m ∧ m ≤ 0) := 
by sorry

end range_of_m_l362_362339


namespace program_output_eq_l362_362567

theorem program_output_eq : ∀ (n : ℤ), n^2 + 3 * n - (2 * n^2 - n) = -n^2 + 4 * n := by
  intro n
  sorry

end program_output_eq_l362_362567


namespace correct_statement_for_certain_event_l362_362983

variable (Ω : Type)
variable (P : set Ω → ℝ)

-- condition 1: The probability of a certain event is 1.
axiom certain_event (A : set Ω) : P(A) = 1 ↔ is_certain_event A

-- condition 2: The probability of an impossible event is 0.
axiom impossible_event (B : set Ω) : P(B) = 0 ↔ is_impossible_event B

-- condition 3: The probability of a random (uncertain) event is between 0 and 1.
axiom random_event (C : set Ω) : 0 < P(C) ∧ P(C) < 1 ↔ is_random_event C

-- Prove that the correct statement is: "The probability of a certain event is definitely 1."
theorem correct_statement_for_certain_event (A : set Ω) (h : is_certain_event A) : P(A) = 1 :=
  by 
    apply certain_event.mp h

end correct_statement_for_certain_event_l362_362983


namespace part1_part2_l362_362716

-- Define the conditions
def triangle_conditions (a b c : ℝ) (A B C : ℝ) : Prop :=
  sin C * sin (A - B) = sin B * sin (C - A) 

-- Define the conclusion for part (1)
def proof_part1 (a b c : ℝ) (A B C : ℝ) (h : triangle_conditions a b c A B C) : Prop :=
  2 * a ^ 2 = b ^ 2 + c ^ 2

-- Define the conditions for part (2)
def triangle_conditions_part2 (a b c A : ℝ) : Prop :=
  a = 5 ∧ cos A = 25 / 31 

-- Define the conclusion for part (2)
def proof_part2 (a b c A : ℝ) (h : triangle_conditions_part2 a b c A) : Prop :=
  a + b + c = 14

-- The Lean statements for the complete problem
theorem part1 (a b c A B C : ℝ) 
  (h : triangle_conditions a b c A B C) : 
  proof_part1 a b c A B C h := 
sorry

theorem part2 (a b c A : ℝ) 
  (h : triangle_conditions_part2 a b c A) : 
  proof_part2 a b c A h := 
sorry

end part1_part2_l362_362716


namespace rowing_distance_l362_362962
-- Lean 4 Statement

theorem rowing_distance (v_m v_t D : ℝ) 
  (h1 : D = v_m + v_t)
  (h2 : 30 = 10 * (v_m - v_t))
  (h3 : 30 = 6 * (v_m + v_t)) :
  D = 5 :=
by sorry

end rowing_distance_l362_362962


namespace card_probability_l362_362901

theorem card_probability :
  let deck_size := 52
  let hearts_count := 13
  let first_card_prob := 1 / deck_size
  let second_card_prob := 1 / (deck_size - 1)
  let third_card_prob := hearts_count / (deck_size - 2)
  let total_prob := first_card_prob * second_card_prob * third_card_prob
  total_prob = 13 / 132600 :=
by
  sorry

end card_probability_l362_362901


namespace A_is_sufficient_but_not_necessary_for_D_l362_362862

variable {A B C D : Prop}

-- Defining the conditions
axiom h1 : A → B
axiom h2 : B ↔ C
axiom h3 : C → D

-- Statement to be proven
theorem A_is_sufficient_but_not_necessary_for_D : (A → D) ∧ ¬(D → A) :=
  by
  sorry

end A_is_sufficient_but_not_necessary_for_D_l362_362862


namespace find_BJ_length_l362_362680

-- Define the equilateral triangle and the points on it
variable (A B C G H F J : Point)
variable [equilateral_triangle ABC]
variable [on_segment G A B]
variable [on_segment F A C]
variable [on_segment H B C]
variable [midpoint J F H]

-- Define the lengths of the segments
variable (AG GF FH HC : ℝ)
variable [AG = 3]
variable [GF = 15]
variable [FH = 8]
variable [HC = 4]

-- Define the proof problem
theorem find_BJ_length : BJ = 19 := by
  sorry

end find_BJ_length_l362_362680


namespace angles_parallel_sides_l362_362670

theorem angles_parallel_sides:
  (∀ (a b c d : ℝ → ℝ → Prop)
    (p q r s : Prop),
    (a = b ∨ a = b - π ∨ a = b + π ∨ c = d ∨ c = d - π ∨ c = d + π)
    → (a p q = b r s)
    → (c p q = d r s)) : sorry :=
  sorry

end angles_parallel_sides_l362_362670


namespace minimum_value_of_f_l362_362597

noncomputable def f (x : ℝ) : ℝ := x^2 + (1 / x^2) + (1 / (x^2 + 1 / x^2))

theorem minimum_value_of_f (x : ℝ) (hx : x > 0) : ∃ y : ℝ, y = f x ∧ y >= 5 / 2 :=
by
  sorry

end minimum_value_of_f_l362_362597


namespace proof_g_l362_362333

def g (x : ℝ) : ℝ :=
  3 * x^3 - 4 * x + 5

theorem proof_g (x h : ℝ) : 
  g(x + h) - g(x) = h * (9 * x^2 + 9 * x * h + 3 * h^2 - 4) :=
by
  sorry

end proof_g_l362_362333


namespace inclination_angle_of_vertical_line_l362_362455

theorem inclination_angle_of_vertical_line : 
  ∀ (x : ℝ), x = real.sqrt 3 → ∃ (θ : ℝ), θ = 90 :=
by
  sorry

end inclination_angle_of_vertical_line_l362_362455


namespace larger_number_l362_362492

variables (x y : ℕ)

theorem larger_number (h1 : x + y = 47) (h2 : x - y = 7) : x = 27 :=
sorry

end larger_number_l362_362492


namespace cube_edge_length_volume_7_l362_362449

theorem cube_edge_length_volume_7 :
  ∃ s : ℝ, s ^ 3 = 7 ∧ s = real.cbrt 7 :=
begin
  sorry
end

end cube_edge_length_volume_7_l362_362449


namespace num_integers_with_factors_between_2000_and_3000_l362_362319

theorem num_integers_with_factors_between_2000_and_3000 :
  ∃ n : ℕ, n = 9 ∧ ∀ x, x ∈ set.Icc 2000 3000 → 10 ∣ x ∧ 24 ∣ x ∧ 30 ∣ x ↔ ∃ (k : ℕ), x = 120 * k :=
by
  sorry

end num_integers_with_factors_between_2000_and_3000_l362_362319


namespace limit_derivative_at_3_l362_362272

variable (f : ℝ → ℝ)

theorem limit_derivative_at_3 (h_deriv : ∀ x, HasDerivAt f (f' x) x) :
    tendsto (λ t, (f 3 - f (3 - t)) / t) (𝓝 0) (𝓝 (f' 3)) :=
by
  sorry

end limit_derivative_at_3_l362_362272


namespace sqrt_49_times_sqrt25_eq_7_sqrt5_l362_362031

-- Definitions based on conditions
def sqrt_25 := Real.sqrt 25
def sqrt_49 := Real.sqrt 49

theorem sqrt_49_times_sqrt25_eq_7_sqrt5 :
  Real.sqrt (49 * sqrt_25) = 7 * Real.sqrt 5 := by
sorry

end sqrt_49_times_sqrt25_eq_7_sqrt5_l362_362031


namespace annie_total_distance_traveled_l362_362193

-- Definitions of conditions
def walk_distance : ℕ := 5
def bus_distance : ℕ := 7
def total_distance_one_way : ℕ := walk_distance + bus_distance
def total_distance_round_trip : ℕ := total_distance_one_way * 2

-- Theorem statement to prove the total number of blocks traveled
theorem annie_total_distance_traveled : total_distance_round_trip = 24 :=
by
  sorry

end annie_total_distance_traveled_l362_362193


namespace sqrt_49_times_sqrt25_eq_7_sqrt5_l362_362034

-- Definitions based on conditions
def sqrt_25 := Real.sqrt 25
def sqrt_49 := Real.sqrt 49

theorem sqrt_49_times_sqrt25_eq_7_sqrt5 :
  Real.sqrt (49 * sqrt_25) = 7 * Real.sqrt 5 := by
sorry

end sqrt_49_times_sqrt25_eq_7_sqrt5_l362_362034


namespace log2_derivative_l362_362506

theorem log2_derivative (x : ℝ) (hx : x > 0) : 
  (deriv (λ x : ℝ, log 2 x)) x = 1 / (x * log 2) :=
sorry

end log2_derivative_l362_362506


namespace power_of_power_evaluate_3_power_3_power_2_l362_362236

theorem power_of_power (a m n : ℕ) : (a^m)^n = a^(m * n) :=
sorry

theorem evaluate_3_power_3_power_2 : (3^3)^2 = 729 :=
by
  have h1 : (3^3)^2 = 3^(3 * 2) := power_of_power 3 3 2
  have h2 : 3^(3 * 2) = 3^6 := rfl
  have h3 : 3^6 = 729 := sorry -- Placeholder for the actual multiplication calculation
  exact eq.trans (eq.trans h1 h2) h3

end power_of_power_evaluate_3_power_3_power_2_l362_362236


namespace product_xy_l362_362358

variables (x y : ℝ)
variables (EF GH FG HE : ℝ)
variable (parallelogram_EFGH : EFGH)

-- Conditions
def EF_value : EF = 42 := sorry
def GH_value : GH = 3 * x + 6 := sorry
def FG_value : FG = 4 * y ^ 2 + 1 := sorry
def HE_value : HE = 28 := sorry
def parallelogram_condition1 : EF = GH := sorry
def parallelogram_condition2 : FG = HE := sorry

-- Problem Statement
theorem product_xy : x * y = 18 * real.sqrt 3 := 
  by
  sorry

end product_xy_l362_362358


namespace min_m_value_l362_362361

noncomputable def a (n : ℕ) : ℕ := 4 * n - 3

noncomputable def S (n : ℕ) : ℝ := ∑ i in finset.range n, 1 / (a i)

theorem min_m_value (m : ℕ) : 
  (∀ n : ℕ, n > 0 → S (2 * n + 1) - S n ≤ m / 15) → m ≥ 5 :=
by
  sorry

end min_m_value_l362_362361


namespace coefficient_of_x_squared_in_binomial_expansion_l362_362222

def coefficient_in_binomial_expansion (n : ℕ) (a b : ℂ) (k : ℕ) : ℂ :=
  let C := Nat.choose n k
  in C * (a ^ (n - k)) * (b ^ k)

theorem coefficient_of_x_squared_in_binomial_expansion :
  coefficient_in_binomial_expansion 6 (x^2 / 2) (-1 / Real.sqrt x) 4 = (15 : ℚ) / 4 :=
sorry

end coefficient_of_x_squared_in_binomial_expansion_l362_362222


namespace segments_divided_16_times_l362_362543

theorem segments_divided_16_times :
  let n := 16 in
  let initial_length := 1 in
  let division_factor := 3 in
  let remaining_factor := 2 / 3 in
  let final_length := 1 / (division_factor ^ n) in
  let final_segment_count := 2 ^ n in
  number_of_segments (initial_length : ℝ) division_factor n = final_segment_count ∧ final_segment_length (initial_length : ℝ) division_factor n = final_length :=
by
  sorry

def number_of_segments (initial_length : ℝ) (division_factor : ℕ) (n : ℕ) : ℕ :=
  2 ^ n

def final_segment_length (initial_length : ℝ) (division_factor : ℕ) (n : ℕ) : ℝ :=
  initial_length / (division_factor ^ n)

end segments_divided_16_times_l362_362543


namespace positive_number_square_roots_l362_362343

theorem positive_number_square_roots (a : ℝ) 
  (h1 : (2 * a - 1) ^ 2 = (a - 2) ^ 2) 
  (h2 : ∃ b : ℝ, b > 0 ∧ ((2 * a - 1) = b ∨ (a - 2) = b)) : 
  ∃ n : ℝ, n = 1 :=
by
  sorry

end positive_number_square_roots_l362_362343


namespace inequality_of_sums_l362_362408

theorem inequality_of_sums
  (a1 a2 b1 b2 : ℝ)
  (h1 : 0 < a1)
  (h2 : 0 < a2)
  (h3 : a1 > a2)
  (h4 : b1 ≥ a1)
  (h5 : b1 * b2 ≥ a1 * a2) :
  b1 + b2 ≥ a1 + a2 :=
by
  -- Here we don't provide the proof
  sorry

end inequality_of_sums_l362_362408


namespace sum_of_roots_l362_362263

theorem sum_of_roots : 
  ∀ x1 x2 : ℝ, (x1 * x2) = 4 ∧ (x1 + x2) = 5 → (x1 + x2 = 5) :=
by
  intros x1 x2 h,
  cases h with prod sum,
  exact sum

end sum_of_roots_l362_362263


namespace translation_min_point_correct_l362_362454

-- Define the original equation
def original_eq (x : ℝ) := |x| - 5

-- Define the translation function
def translate_point (p : ℝ × ℝ) (tx ty : ℝ) : ℝ × ℝ := (p.1 + tx, p.2 + ty)

-- Define the minimum point of the original equation
def original_min_point : ℝ × ℝ := (0, original_eq 0)

-- Translate the original minimum point three units right and four units up
def new_min_point := translate_point original_min_point 3 4

-- Prove that the new minimum point is (3, -1)
theorem translation_min_point_correct : new_min_point = (3, -1) :=
by
  sorry

end translation_min_point_correct_l362_362454


namespace centroid_positions_correct_l362_362435

structure Point where
  x : ℕ
  y : ℕ

def is_point_valid (p : Point) : Prop :=
  (p.x = 0 ∨ p.x = 8 ∨ p.x % 1 = 0 ∧ p.x / 8 < 1) ∧ 
  (p.y = 0 ∨ p.y = 8 ∨ p.y % 1 = 0 ∧ p.y / 8 < 1)

def is_not_collinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) ≠ (r.x - p.x) * (q.y - p.y)

def centroid (p q r : Point) : Point :=
  { x := (p.x + q.x + r.x) / 3, y := (p.y + q.y + r.y) / 3 }

def within_bounds (p : Point) : Prop :=
  1 ≤ p.x ∧ p.x ≤ 7 ∧ 1 ≤ p.y ∧ p.y ≤ 7

noncomputable def count_valid_centroids : ℕ :=
  Set.card (SetOf (λ p₁ p₂ p₃ : Point, is_point_valid p₁ ∧ is_point_valid p₂ ∧ is_point_valid p₃ ∧ 
                                is_not_collinear p₁ p₂ p₃ ∧ within_bounds (centroid p₁ p₂ p₃)))

theorem centroid_positions_correct : count_valid_centroids = 49 :=
  sorry

end centroid_positions_correct_l362_362435


namespace simplify_and_evaluate_l362_362853

theorem simplify_and_evaluate (a b : ℝ) (h₁ : a = -1) (h₂ : b = 1/4) : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_and_evaluate_l362_362853


namespace bicycle_spokes_l362_362191

theorem bicycle_spokes (front_spokes : ℕ) (back_spokes : ℕ) 
  (h_front : front_spokes = 20) (h_back : back_spokes = 2 * front_spokes) :
  front_spokes + back_spokes = 60 :=
by
  rw [h_front, h_back]
  norm_num

end bicycle_spokes_l362_362191


namespace telescope_serial_number_count_l362_362524

def digits : List ℕ := [1, 2, 2, 3, 5, 5, 7, 9]
def is_prime (n : ℕ) : Bool :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def starts_with_prime (s : List ℕ) : Bool :=
  is_prime s.head

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else List.range (n - 1) |>.foldl (· * ·) n

theorem telescope_serial_number_count :
  ∑ start in [2, 3, 5, 7], (factorial 7) / ((factorial 2) * (factorial 2)) = 5040 := by
  sorry

end telescope_serial_number_count_l362_362524


namespace probability_correct_l362_362865

noncomputable def probability_sum_of_rounded_parts_eq_4 : Prop :=
  let intervals := [(0 : ℝ, 0.5), (0.5, 1.5)]
  let total_length := 3.5
  let length_of_intervals := intervals.foldl (fun acc i => acc + (i.snd - i.fst)) 0
  length_of_intervals / total_length = 3 / 7

theorem probability_correct : probability_sum_of_rounded_parts_eq_4 := by
  sorry

end probability_correct_l362_362865


namespace simplify_and_evaluate_l362_362855

theorem simplify_and_evaluate (a b : ℝ) (h₁ : a = -1) (h₂ : b = 1/4) : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_and_evaluate_l362_362855


namespace number_of_two_digit_integers_l362_362249

theorem number_of_two_digit_integers : 
  let digits := {0, 1, 2, 3, 4}
  let first_digit_choices := {1, 2, 3, 4}
  ∃ count : ℕ, 
  count = finset.card first_digit_choices * (finset.card digits - 1) ∧ 
  count = 16 :=
by
  let digits := {0, 1, 2, 3, 4}
  let first_digit_choices := {1, 2, 3, 4}
  existsi 16
  have h1 : finset.card first_digit_choices = 4 := by decide
  have h2 : finset.card digits - 1 = 4 := by decide
  have h3 : 4 * 4 = 16 := by norm_num
  split
  exact h3
  exact h3

end number_of_two_digit_integers_l362_362249


namespace functional_equation_solution_l362_362591

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution :
  (∀ x y : ℝ, f (f x + y) = 2 * x + f (f y - x)) →
  ∃ a : ℝ, ∀ x : ℝ, f x = x - a :=
by
  intro h
  sorry

end functional_equation_solution_l362_362591


namespace max_composite_numbers_l362_362802
open Nat

def is_composite (n : ℕ) : Prop := 1 < n ∧ ∃ d : ℕ, 1 < d ∧ d < n ∧ d ∣ n

def has_gcd_of_one (l : List ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ l → b ∈ l → a ≠ b → gcd a b = 1

def valid_composite_numbers (n : ℕ) : Prop :=
  ∀ m ∈ (List.range n).filter is_composite, m < 1500 →

-- Main theorem
theorem max_composite_numbers :
  ∃ l : List ℕ, l.length = 12 ∧ valid_composite_numbers l ∧ has_gcd_of_one l :=
sorry

end max_composite_numbers_l362_362802


namespace largest_k_divides_factorial_l362_362215

theorem largest_k_divides_factorial (k : ℕ) :
  let num := 2004
      num_factorization := (2^2) * 3 * 167
      fact_num := (fact num)
  in ∃ k:ℕ, (num ^ k ∣ fact_num) ∧ k = 12 := 
sorry

end largest_k_divides_factorial_l362_362215


namespace jean_average_speed_l362_362575

/-- 
  Chantal hikes 6 miles to a fire tower and back to the 3-mile point, walking:
  - 3 miles at 5 mph on a flat portion,
  - 3 miles uphill at 3 mph,
  - 3 miles downhill at 4 mph.
  Jean meets Chantal at the 3-mile point from the trailhead, 
  having hiked a constant speed. 
  
  Prove that Jean's average speed is 1.5 mph. 
-/
theorem jean_average_speed (t1 t2 t3 : ℝ) (h1 : t1 = 3/5) (h2 : t2 = 1) (h3 : t3 = 3/4) (T : ℝ) (hT : T = t1 + t2 + t3) : 
  ∀ (dist : ℝ) (s : ℝ) (h_dist : dist = 3) (h_s : s = dist / T), 
  (s = 1.5) :=
begin
  sorry
end

end jean_average_speed_l362_362575


namespace each_parent_pays_l362_362959

def initial_salary : ℕ := 60000
def raise_percentage : ℕ := 25
def num_kids : ℕ := 15

theorem each_parent_pays :
  let raise := initial_salary * raise_percentage / 100 in
  let new_salary := initial_salary + raise in
  new_salary / num_kids = 5000 := by
  let raise := initial_salary * raise_percentage / 100
  let new_salary := initial_salary + raise
  have divide_amount : new_salary / num_kids = 5000
  exact divide_amount
  sorry

end each_parent_pays_l362_362959


namespace rectangle_area_l362_362163

theorem rectangle_area (w l d : ℝ) 
  (h1 : l = 2 * w) 
  (h2 : d = 10)
  (h3 : d^2 = w^2 + l^2) : 
  l * w = 40 := 
by
  sorry

end rectangle_area_l362_362163


namespace remi_water_bottle_capacity_l362_362831

-- Let's define the problem conditions
def daily_refills : ℕ := 3
def days : ℕ := 7
def total_spilled : ℕ := 5 + 8 -- Total spilled water in ounces
def total_intake : ℕ := 407 -- Total amount of water drunk in 7 days

-- The capacity of Remi's water bottle is the quantity we need to prove
def bottle_capacity (x : ℕ) : Prop :=
  daily_refills * days * x - total_spilled = total_intake

-- Statement of the proof problem
theorem remi_water_bottle_capacity : bottle_capacity 20 :=
by
  sorry

end remi_water_bottle_capacity_l362_362831


namespace exists_consecutive_primes_sum_div_by_three_l362_362251

open Nat

def is_prime (n : ℕ) : Prop := 
  2 ≤ n ∧ ∀ m ∣ n, m = 1 ∨ m = n

theorem exists_consecutive_primes_sum_div_by_three : 
  ∃ (p1 p2 p3 p4 : ℕ), 
  (is_prime p1) ∧ (is_prime p2) ∧ (is_prime p3) ∧ (is_prime p4) ∧ 
  (p1 < p2) ∧ (p2 < p3) ∧ (p3 < p4) ∧ 
  (p1 = 5 → p2 = 7 → p3 = 11 → p4 = 13 → false) ∧ 
  (p1 + p2 + p3 + p4) % 3 = 0 := 
sorry

end exists_consecutive_primes_sum_div_by_three_l362_362251


namespace total_blocks_traveled_l362_362196

-- Given conditions as definitions
def annie_walked_blocks : ℕ := 5
def annie_rode_blocks : ℕ := 7

-- The total blocks Annie traveled
theorem total_blocks_traveled : annie_walked_blocks + annie_rode_blocks + (annie_walked_blocks + annie_rode_blocks) = 24 := by
  sorry

end total_blocks_traveled_l362_362196


namespace cevian_sum_equals_two_l362_362623

-- Definitions based on conditions
variables {A B C D E F O : Type*}
variables (AD BE CF : ℝ) (R : ℝ)
variables (circumcenter_O : O = circumcenter A B C)
variables (intersect_AD_O : AD = abs ((line A D).proj O))
variables (intersect_BE_O : BE = abs ((line B E).proj O))
variables (intersect_CF_O : CF = abs ((line C F).proj O))

-- Prove the main statement
theorem cevian_sum_equals_two (h : circumcenter_O ∧ intersect_AD_O ∧ intersect_BE_O ∧ intersect_CF_O) :
  1 / AD + 1 / BE + 1 / CF = 2 / R :=
sorry

end cevian_sum_equals_two_l362_362623


namespace sqrt_mul_sqrt_l362_362047

theorem sqrt_mul_sqrt (a b c : ℝ) (hac : a = 49) (hbc : b = sqrt 25) (hc : c = 7 * sqrt 5) : sqrt (a * b) = c := 
by
  sorry

end sqrt_mul_sqrt_l362_362047


namespace collinear_A1_E_N_l362_362366

open EuclideanGeometry

variables {V : Type*} [inner_product_space ℝ V]
variables {A B C D A1 B1 C1 D1 M N E : V}

/-- Collinearity definition in Euclidean space -/
def collinear (x y z : V) : Prop :=
∃ (k : ℝ), (y - x) = k • (z - x)

-- Given conditions
variable [h₁ : midpoint ℝ D D1 M]
variable [h₂ : midpoint ℝ B M E]
variable [h₃ : ∃ (k : ℝ), k = (2/3) ∧ N = k • (A + C)]
variable [h₄ : ∃ a b c : V, a = B - A ∧ b = D - A ∧ c = A1 - A]

-- Proof statement
theorem collinear_A1_E_N :
  collinear A1 E N :=
by
  -- Here we assume the steps leading to collinearity
  sorry

end collinear_A1_E_N_l362_362366


namespace odd_function_increasing_on_negative_interval_l362_362668

theorem odd_function_increasing_on_negative_interval {f : ℝ → ℝ}
  (h_odd : ∀ x, f (-x) = -f x)
  (h_increasing : ∀ x y, 3 ≤ x → x ≤ 7 → 3 ≤ y → y ≤ 7 → x < y → f x < f y)
  (h_min_value : f 3 = 1) :
  (∀ x y, -7 ≤ x → x ≤ -3 → -7 ≤ y → y ≤ -3 → x < y → f x < f y) ∧ f (-3) = -1 := 
sorry

end odd_function_increasing_on_negative_interval_l362_362668


namespace rectangle_same_color_exists_l362_362354

def M : Finset (ℕ × ℕ) := 
  {p | ∃ x y : ℕ, x < 13 ∧ y < 13 ∧ p = (x, y)}

axiom color : (ℕ × ℕ) → color
inductive color
| red : color
| white : color
| blue : color

theorem rectangle_same_color_exists (colored_M : (ℕ × ℕ) → color):
  ∃ (a b c d : ℕ × ℕ), a ∈ M ∧ b ∈ M ∧ c ∈ M ∧ d ∈ M ∧ 
  (color a = color b ∧ color b = color c ∧ color c = color d) ∧ 
  (a.1 = b.1 ∧ c.1 = d.1 ∧ a.2 = c.2 ∧ b.2 = d.2) :=
sorry

end rectangle_same_color_exists_l362_362354


namespace simplify_and_evaluate_l362_362848

theorem simplify_and_evaluate 
  (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) = 1 :=
by
  sorry

end simplify_and_evaluate_l362_362848


namespace james_weekly_earnings_l362_362380

def hourly_rate : ℕ := 20
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 4

theorem james_weekly_earnings : hourly_rate * (hours_per_day * days_per_week) = 640 := by
  sorry

end james_weekly_earnings_l362_362380


namespace sqrt_mul_sqrt_l362_362049

theorem sqrt_mul_sqrt (a b c : ℝ) (hac : a = 49) (hbc : b = sqrt 25) (hc : c = 7 * sqrt 5) : sqrt (a * b) = c := 
by
  sorry

end sqrt_mul_sqrt_l362_362049


namespace aj_ak_eq_ao_ar_j_is_incenter_l362_362734

open EuclideanGeometry

noncomputable theory

variables {A B C : Point} (is_isosceles : B ≠ C ∧ dist A B = dist A C)
(Gamma : Circle)
(hGamma : circumscribed_triangle Gamma A B C)
(gamma : Circle)
(hgamma : is_inscribed gamma A B C)
(P Q R : Point) (hP : tangent_at gamma A B P) (hQ : tangent_at gamma A C Q) (hR : tangent_at gamma Gamma R)
(O : Point) (hO : center O gamma)
(J : Point) (hJ : midpoint J P Q)
(K : Point) (hK : midpoint K B C)

theorem aj_ak_eq_ao_ar : 
  dist A J / dist A K = dist A O / dist A R :=
sorry

theorem j_is_incenter : 
  is_incenter J A B C :=
sorry

end aj_ak_eq_ao_ar_j_is_incenter_l362_362734


namespace centroid_of_triangle_l362_362420

variables {V : Type*} [InnerProductSpace ℝ V]
variables {A B C P G : V}

theorem centroid_of_triangle
  (h : G = (1 / 3) • (P + (A - P) + (B - P) + (C - P))) :
  is_centroid G A B C :=
sorry

end centroid_of_triangle_l362_362420


namespace sqrt_mul_sqrt_l362_362042

theorem sqrt_mul_sqrt (a b c : ℝ) (hac : a = 49) (hbc : b = sqrt 25) (hc : c = 7 * sqrt 5) : sqrt (a * b) = c := 
by
  sorry

end sqrt_mul_sqrt_l362_362042


namespace log_inequality_l362_362658

theorem log_inequality (x : ℝ) (hx : x > 1) :
  let a := Real.log x / Real.log 0.5 in a^2 > a ∧ a > 2 * a :=
by
  sorry

end log_inequality_l362_362658


namespace round_robin_cycles_l362_362971

-- Define the conditions
def teams : ℕ := 28
def wins_per_team : ℕ := 13
def losses_per_team : ℕ := 13
def total_teams_games := teams * (teams - 1) / 2
def sets_of_three_teams := (teams * (teams - 1) * (teams - 2)) / 6

-- Define the problem statement
theorem round_robin_cycles :
  -- We need to show that the number of sets of three teams {A, B, C} where A beats B, B beats C, and C beats A is 1092
  (sets_of_three_teams - (teams * (wins_per_team * (wins_per_team - 1)) / 2)) = 1092 :=
by
  sorry

end round_robin_cycles_l362_362971


namespace employee_payment_l362_362535

noncomputable section
open Classical

def wholesale_cost : ℝ := 200
def retail_markup : ℝ := 0.20
def employee_discount : ℝ := 0.05

def retail_price (wholesale_cost markup : ℝ) := wholesale_cost * (1 + markup)
def discounted_price (retail_price discount : ℝ) := retail_price * (1 - discount)

theorem employee_payment :
  let retail := retail_price wholesale_cost retail_markup in
  let employee_pay := discounted_price retail employee_discount in
  employee_pay = 228 :=
by
  sorry

end employee_payment_l362_362535


namespace sqrt_expression_simplified_l362_362011

theorem sqrt_expression_simplified :
  sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  sorry

end sqrt_expression_simplified_l362_362011


namespace sqrt_49_times_sqrt25_eq_7_sqrt5_l362_362030

-- Definitions based on conditions
def sqrt_25 := Real.sqrt 25
def sqrt_49 := Real.sqrt 49

theorem sqrt_49_times_sqrt25_eq_7_sqrt5 :
  Real.sqrt (49 * sqrt_25) = 7 * Real.sqrt 5 := by
sorry

end sqrt_49_times_sqrt25_eq_7_sqrt5_l362_362030


namespace composition_of_perpendicular_planes_is_axial_symmetry_axial_symmetry_as_composition_of_perpendicular_reflections_l362_362925

-- Part (a)
theorem composition_of_perpendicular_planes_is_axial_symmetry (A : ℝ^3) (α β : set ℝ^3) (s : ℝ^3) 
  (h1 : α ∩ β = s) (h2 : α ⊥ β) : 
  ∃ (A1 A' : ℝ^3), reflection A α = A1 ∧ reflection A1 β = A' ∧ axial_symmetry A A' s := sorry

-- Part (b)
theorem axial_symmetry_as_composition_of_perpendicular_reflections (s : ℝ^3) : 
  ∃ (α β : set ℝ^3), α ∩ β = s ∧ α ⊥ β ∧ 
  ∀ A : ℝ^3, ∃ (A1 A' : ℝ^3), reflection A α = A1 ∧ reflection A1 β = A' ∧ axial_symmetry A' A s := sorry

end composition_of_perpendicular_planes_is_axial_symmetry_axial_symmetry_as_composition_of_perpendicular_reflections_l362_362925


namespace remove_10_fac_to_make_square_l362_362564

theorem remove_10_fac_to_make_square : 
  ∃ (n : ℕ), n ! * ((∏ i in finset.range 21, if i = 10 then 1 else i !) / 10 !) = n * n :=
by
  sorry

end remove_10_fac_to_make_square_l362_362564


namespace numbers_not_crossed_out_l362_362589

/-- Total numbers between 1 and 90 after crossing out multiples of 3 and 5 is 48. -/
theorem numbers_not_crossed_out : 
  let n := 90 
  let multiples_of_3 := n / 3 
  let multiples_of_5 := n / 5 
  let multiples_of_15 := n / 15 
  let crossed_out := multiples_of_3 + multiples_of_5 - multiples_of_15
  n - crossed_out = 48 :=
by {
  sorry
}

end numbers_not_crossed_out_l362_362589


namespace max_oleg_composite_numbers_l362_362752

/-- Oleg's list of composite numbers less than 1500 and pairwise coprime. --/
def oleg_composite_numbers (numbers : List ℕ) : Prop :=
  ∀ n ∈ numbers, Nat.isComposite n ∧ n < 1500 ∧ (∀ m ∈ numbers, n ≠ m → Nat.gcd n m = 1)

/-- Maximum number of such numbers that Oleg could write. --/
theorem max_oleg_composite_numbers : ∃ numbers : List ℕ, oleg_composite_numbers numbers ∧ numbers.length = 12 := 
sorry

end max_oleg_composite_numbers_l362_362752


namespace hyperbola_eccentricity_l362_362547

variable (a b c e : ℝ)

-- Definitions from conditions
def PF2 : ℝ := b^2 / a
def F1F2 : ℝ := 2 * c
def angle_PF1Q : ℝ := π / 2

-- Problem statement: to prove that the eccentricity e of the hyperbola satisfies e = sqrt(2) + 1
theorem hyperbola_eccentricity (h1 : PF2 = b^2 / a)
                               (h2 : F1F2 = 2 * c)
                               (h3 : angle_PF1Q = π / 2)
                               (h4 : a ≠ 0) 
                               (h5 : b ≠ 0) 
                               (h6 : c ≠ 0) : 
    e = sqrt 2 + 1 := 
by 
    sorry

end hyperbola_eccentricity_l362_362547


namespace tan_theta_max_l362_362453

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (x + π / 6)

theorem tan_theta_max (θ : ℝ) (h : ∀ x, f x ≤ f θ) : Real.tan θ = Real.sqrt 3 :=
by
  sorry

end tan_theta_max_l362_362453


namespace count_valid_numbers_l362_362462

theorem count_valid_numbers :
  let valid_numbers (n : ℕ) := (n / 1000 = 1) ∧
                               (∀ d1 d2 d3 d4, n = 1000*d1 + 100*d2 + 10*d3 + d4 → 
                                 (d1 = 1 ∧ 
                                 ((d2 = d3 ∧ d2 ≠ d4 ∨ d2 = d4 ∧ d2 ≠ d3 ∨ d3 = d4 ∧ d2 ≠ d3) 
                                 ∨(d2 ≠ d3 ∨ d2 ≠ d4 ∨ d3 ≠ d4))) ∧
                                 (d1 + d2 + d3 + d4 < 17)) 
  in {n | valid_numbers n}.card = 270 :=
sorry

end count_valid_numbers_l362_362462


namespace oleg_max_composite_numbers_l362_362768

theorem oleg_max_composite_numbers : 
  ∃ (S : Finset ℕ), 
    (∀ (n ∈ S), n < 1500 ∧ ∃ p q, prime p ∧ prime q ∧ p ≠ q ∧ p * q = n) ∧ 
    (∀ (a b ∈ S), a ≠ b → gcd a b = 1) ∧ 
    S.card = 12 :=
sorry

end oleg_max_composite_numbers_l362_362768


namespace limit_solution_l362_362204

noncomputable def limit_problem : Prop :=
  ∀ (f : ℝ → ℝ) (L : ℝ), 
  (f = λ x, (1 - sqrt (cos x)) / (1 - cos (sqrt x))) → 
  is_limit f 0 L

theorem limit_solution : limit_problem :=
begin
  intros f L h,
  have : f = λ x, (1 - sqrt (cos x)) / (1 - cos (sqrt x)) := h,
  rw this,
  apply limit_const,
  sorry
end

end limit_solution_l362_362204


namespace oleg_max_composite_numbers_l362_362773

theorem oleg_max_composite_numbers : 
  ∃ (S : Finset ℕ), 
    (∀ (n ∈ S), n < 1500 ∧ ∃ p q, prime p ∧ prime q ∧ p ≠ q ∧ p * q = n) ∧ 
    (∀ (a b ∈ S), a ≠ b → gcd a b = 1) ∧ 
    S.card = 12 :=
sorry

end oleg_max_composite_numbers_l362_362773


namespace james_payment_correct_l362_362376

-- Definitions from conditions
def cost_steak_eggs : ℝ := 16
def cost_chicken_fried_steak : ℝ := 14
def total_cost := cost_steak_eggs + cost_chicken_fried_steak
def half_share := total_cost / 2
def tip := total_cost * 0.2
def james_total_payment := half_share + tip

-- Statement to be proven
theorem james_payment_correct : james_total_payment = 21 :=
by
  sorry

end james_payment_correct_l362_362376


namespace triangle_sides_relation_triangle_perimeter_l362_362713

theorem triangle_sides_relation
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : sin C * sin (A - B) = sin B * sin (C - A)) :
  2 * a^2 = b^2 + c^2 :=
sorry

theorem triangle_perimeter 
  (a b c : ℝ)
  (A B C : ℝ)
  (h_a : a = 5)
  (h_cosA : cos A = 25 / 31)
  (h_sin_relation : sin C * sin (A - B) = sin B * sin (C - A)) :
  a + b + c = 14 :=
sorry

end triangle_sides_relation_triangle_perimeter_l362_362713


namespace amount_collected_ii_class_l362_362930

theorem amount_collected_ii_class (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) : 
  (x * y * 53 = 1325) →
  50 * (x * y) = 1250 :=
by
  intro h
  have hxy : x * y = 1325 / 53 := sorry -- Divide both sides by 53
  ring_exp at hxy
  calc
  50 * (x * y) = 50 * (1325 / 53) : by rw [←hxy]
            ... = 1250 : by norm_num
  sorry

end amount_collected_ii_class_l362_362930


namespace sqrt_mul_sqrt_l362_362109

theorem sqrt_mul_sqrt (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : sqrt (a * sqrt b) = sqrt a * sqrt (sqrt b) := by
  sorry

example : sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  have h1 : sqrt 25 = 5 := by 
    norm_num
  have h2 : sqrt (49 * 5) = sqrt 245 := by  
    unfold sqrt
  have h3 : sqrt 245 = 7 * sqrt 5 := by 
    norm_num
  rw [h1, h2, h3]
  norm_num

end sqrt_mul_sqrt_l362_362109


namespace L_shape_perimeter_correct_l362_362880

-- Define the dimensions of the rectangles
def rect_height : ℕ := 3
def rect_width : ℕ := 4

-- Define the combined shape and perimeter calculation
def L_shape_perimeter (h w : ℕ) : ℕ := (2 * w) + (2 * h)

theorem L_shape_perimeter_correct : 
  L_shape_perimeter rect_height rect_width = 14 := 
  sorry

end L_shape_perimeter_correct_l362_362880


namespace wage_increase_l362_362466

theorem wage_increase (x : ℝ) : (y = 50 + 80 * x) → (y' = 50 + 80 * (x + 1)) → (y' - y = 80) :=
by
  intros h1 h2
  rw [h1, h2]
  linear_comb
  sorry

end wage_increase_l362_362466


namespace carla_drinks_water_l362_362211

-- Definitions from the conditions
def total_liquid (s w : ℕ) : Prop := s + w = 54
def soda_water_relation (s w : ℕ) : Prop := s = 3 * w - 6

-- Proof statement
theorem carla_drinks_water : ∀ (s w : ℕ), total_liquid s w ∧ soda_water_relation s w → w = 15 :=
by
  intros s w h,
  sorry

end carla_drinks_water_l362_362211


namespace sqrt_49_times_sqrt_25_l362_362029

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 :=
by
  sorry

end sqrt_49_times_sqrt_25_l362_362029


namespace car_trip_time_l362_362493

theorem car_trip_time (walking_mixed: 1.5 = 1.25 + x) 
                      (walking_both: 2.5 = 2 * 1.25) : 
  2 * x * 60 = 30 :=
by sorry

end car_trip_time_l362_362493


namespace polynomial_at_zero_l362_362729

noncomputable def cubic_polynomial : Type := Real → Real

def satisfies_conditions (f : cubic_polynomial) : Prop :=
  is_cubic_polynomial f ∧
  (|f(1)| = 6) ∧
  (|f(2)| = 6) ∧
  (|f(3)| = 18) ∧
  (|f(4)| = 18) ∧
  (|f(5)| = 30) ∧
  (|f(6)| = 30)

theorem polynomial_at_zero (f : cubic_polynomial) (h : satisfies_conditions f) : |f(0)| = 66 :=
  sorry

end polynomial_at_zero_l362_362729


namespace integer_quotient_is_perfect_square_l362_362739

theorem integer_quotient_is_perfect_square (a b : ℕ) (h : 0 < a ∧ 0 < b) (h_int : (a + b) ^ 2 % (4 * a * b + 1) = 0) :
  ∃ k : ℕ, (a + b) ^ 2 = k ^ 2 * (4 * a * b + 1) := sorry

end integer_quotient_is_perfect_square_l362_362739


namespace volume_ratio_l362_362646

noncomputable def cylinder_base_areas (S1 S2 : ℝ) (h1 h2 : ℝ) : Prop :=
  S1 / S2 = 9 / 4 ∧ h1 = h2

noncomputable def lateral_surface_areas_equal (R r H h : ℝ) : Prop :=
  2 * Mathlib.Real.pi * R * H = 2 * Mathlib.Real.pi * r * h

theorem volume_ratio (S1 S2 V1 V2 : ℝ) (R r H h : ℝ) 
  (h_base_areas : cylinder_base_areas S1 S2 (R^2 * Mathlib.Real.pi) (r^2 * Mathlib.Real.pi))
  (h_lateral_surface_areas : lateral_surface_areas_equal R r H h) : 
  V1 / V2 = 3 / 2 :=
sorry

end volume_ratio_l362_362646


namespace annie_money_left_l362_362992

/-- 
Annie has $120. The restaurant next door sells hamburgers for $4 each. 
The restaurant across the street sells milkshakes for $3 each. 
Annie buys 8 hamburgers and 6 milkshakes. 
Prove that Annie will have $70 left after her purchases.
-/
theorem annie_money_left
  (initial_money : ℕ)
  (hamburger_cost : ℕ)
  (milkshake_cost : ℕ)
  (hamburgers_bought : ℕ)
  (milkshakes_bought : ℕ)
  (initial_money_eq : initial_money = 120)
  (hamburger_cost_eq : hamburger_cost = 4)
  (milkshake_cost_eq : milkshake_cost = 3)
  (hamburgers_bought_eq : hamburgers_bought = 8)
  (milkshakes_bought_eq : milkshakes_bought = 6)
  : initial_money - (hamburgers_bought * hamburger_cost + milkshakes_bought * milkshake_cost) = 70 :=
by
  rw [initial_money_eq, hamburger_cost_eq, milkshake_cost_eq, hamburgers_bought_eq, milkshakes_bought_eq]
  norm_num
  sorry

end annie_money_left_l362_362992


namespace sqrt_49_times_sqrt_25_eq_7sqrt5_l362_362076

theorem sqrt_49_times_sqrt_25_eq_7sqrt5 :
  (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  have h1 : Real.sqrt 25 = 5 := by sorry
  have h2 : 49 * 5 = 245 := by sorry
  have h3 : Real.sqrt 245 = 7 * Real.sqrt 5 := by sorry
  sorry

end sqrt_49_times_sqrt_25_eq_7sqrt5_l362_362076


namespace b_95_mod_49_l362_362728

-- Define the sequence b_n
def b (n : ℕ) : ℕ := 7^n + 9^n

-- Goal: Prove that the remainder when b 95 is divided by 49 is 28
theorem b_95_mod_49 : b 95 % 49 = 28 := 
by
  sorry

end b_95_mod_49_l362_362728


namespace num_words_with_A_l362_362312

theorem num_words_with_A :
  let total_words := 5^4,
      words_without_A := 4^4 in
  total_words - words_without_A = 369 :=
by
  sorry

end num_words_with_A_l362_362312


namespace find_f1_l362_362436

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f1
  (h1 : ∀ x : ℝ, |f x - x^2| ≤ 1/4)
  (h2 : ∀ x : ℝ, |f x + 1 - x^2| ≤ 3/4) :
  f 1 = 3/4 := 
sorry

end find_f1_l362_362436


namespace period_sin_cos_l362_362911

def period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem period_sin_cos :
  ∃ T, period (λ x, sin (8 * x) + cos (4 * x)) T ∧ T = π / 2 :=
by
  use π / 2
  split
  · intro x
    rw [sin_add, cos_add]
    sorry
  · rfl

end period_sin_cos_l362_362911


namespace power_of_a_power_evaluate_3_pow_3_pow_2_l362_362233

theorem power_of_a_power (a m n : ℕ) : (a^m)^n = a^(m*n) := 
begin
  sorry,
end

theorem evaluate_3_pow_3_pow_2 : (3^3)^2 = 729 := 
begin
  have H1 : (3^3)^2 = 3^(3*2) := power_of_a_power 3 3 2,
  have H2 : 3^(3*2) = 3^6 := by refl,
  have H3 : 3^6 = 729 := by norm_num,
  exact eq.trans (eq.trans H1 H2) H3,
end

end power_of_a_power_evaluate_3_pow_3_pow_2_l362_362233


namespace number_of_primes_between_40_and_50_l362_362323

theorem number_of_primes_between_40_and_50 : 
  (finset.filter is_prime (finset.range' 41 10)).card = 3 :=
begin
  sorry
end

end number_of_primes_between_40_and_50_l362_362323


namespace baker_cakes_remaining_l362_362559

theorem baker_cakes_remaining (initial_cakes: ℕ) (fraction_sold: ℚ) (sold_cakes: ℕ) (cakes_remaining: ℕ) :
  initial_cakes = 149 ∧ fraction_sold = 2/5 ∧ sold_cakes = 59 ∧ cakes_remaining = initial_cakes - sold_cakes → cakes_remaining = 90 :=
by
  sorry

end baker_cakes_remaining_l362_362559


namespace sqrt_49_times_sqrt_25_eq_7sqrt5_l362_362066

theorem sqrt_49_times_sqrt_25_eq_7sqrt5 :
  (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  have h1 : Real.sqrt 25 = 5 := by sorry
  have h2 : 49 * 5 = 245 := by sorry
  have h3 : Real.sqrt 245 = 7 * Real.sqrt 5 := by sorry
  sorry

end sqrt_49_times_sqrt_25_eq_7sqrt5_l362_362066


namespace probability_both_red_l362_362137

noncomputable def balls := {red := 3, blue := 4, green := 4}
noncomputable def totalBalls := balls.red + balls.blue + balls.green

theorem probability_both_red (hred : balls.red = 3) (hblue : balls.blue = 4) (hgreen : balls.green = 4) :
  (3 / 11) * (2 / 10) = 3 / 55 :=
begin
  sorry
end

end probability_both_red_l362_362137


namespace abs_sin_diff_le_abs_sin_sub_l362_362827

theorem abs_sin_diff_le_abs_sin_sub (A B : ℝ) (hA : 0 ≤ A) (hA' : A ≤ π) (hB : 0 ≤ B) (hB' : B ≤ π) :
  |Real.sin A - Real.sin B| ≤ |Real.sin (A - B)| :=
by
  -- Proof would go here
  sorry

end abs_sin_diff_le_abs_sin_sub_l362_362827


namespace max_composite_numbers_l362_362796

-- Definitions and conditions
def is_composite (n : ℕ) : Prop := 2 < n ∧ ∃ d, d ∣ n ∧ 1 < d ∧ d < n

def less_than_1500 (n : ℕ) : Prop := n < 1500

def gcd_is_one (a b : ℕ) : Prop := Int.gcd a b = 1

-- The problem statement
theorem max_composite_numbers (numbers : List ℕ) (h_composite : ∀ n ∈ numbers, is_composite n) 
  (h_less_than_1500 : ∀ n ∈ numbers, less_than_1500 n)
  (h_pairwise_gcd_is_one : (numbers.Pairwise gcd_is_one)) : numbers.length ≤ 12 := 
  sorry

end max_composite_numbers_l362_362796


namespace sqrt_expression_simplified_l362_362012

theorem sqrt_expression_simplified :
  sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  sorry

end sqrt_expression_simplified_l362_362012


namespace find_phase_shift_l362_362598

theorem find_phase_shift (x : ℝ) : 
  let y := 2 * sin (2 * x + π / 3) in
  phase_shift (λ x => 2 * sin (2 * x + π / 3)) = -π / 6 :=
sorry

end find_phase_shift_l362_362598


namespace initial_population_l362_362134

/--
Suppose 5% of people in a village died by bombardment,
15% of the remaining population left the village due to fear,
and the population is now reduced to 3294.
Prove that the initial population was 4080.
-/
theorem initial_population (P : ℝ) 
  (H1 : 0.05 * P + 0.15 * (1 - 0.05) * P + 3294 = P) : P = 4080 :=
sorry

end initial_population_l362_362134


namespace sqrt_49_times_sqrt25_eq_7_sqrt5_l362_362032

-- Definitions based on conditions
def sqrt_25 := Real.sqrt 25
def sqrt_49 := Real.sqrt 49

theorem sqrt_49_times_sqrt25_eq_7_sqrt5 :
  Real.sqrt (49 * sqrt_25) = 7 * Real.sqrt 5 := by
sorry

end sqrt_49_times_sqrt25_eq_7_sqrt5_l362_362032


namespace find_radius_of_incircle_l362_362487

noncomputable def radius_of_incircle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
   (triangle : EuclideanGeometry.Triangle A B C) 
   (right_angle : triangle.C.angle = 90) 
   (angle_A : triangle.A.angle = 45) 
   (length_AC : triangle.AC.length = 12) : ℝ :=
6 - 3 * Math.sqrt 2

theorem find_radius_of_incircle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
   (triangle : EuclideanGeometry.Triangle A B C) 
   (right_angle : triangle.C.angle = 90) 
   (angle_A : triangle.A.angle = 45) 
   (length_AC : triangle.AC.length = 12) :
   radius_of_incircle A B C triangle right_angle angle_A length_AC = 6 - 3 * Math.sqrt 2 :=
by
  sorry

end find_radius_of_incircle_l362_362487


namespace discount_percentage_l362_362156

variable (C : ℝ) -- Cost price of the turtleneck sweater
variable (SP1 SP2 SP3 : ℝ) -- Selling prices at different stages
variable (D : ℝ) -- Discount percentage

-- Conditions
def initial_markup := SP1 = 1.20 * C
def new_year_markup := SP2 = 1.25 * SP1
def february_selling_price := SP3 = SP2 * (1 - D)
def february_profit := SP3 = 1.35 * C

-- Theorem to prove
theorem discount_percentage :
  initial_markup → new_year_markup → february_selling_price → february_profit → D = 0.10 :=
by
  intros h1 h2 h3 h4
  -- Proof can be done using the given conditions and solving the equation similar to the steps in the solution.
  sorry

end discount_percentage_l362_362156


namespace sqrt_expression_simplified_l362_362008

theorem sqrt_expression_simplified :
  sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  sorry

end sqrt_expression_simplified_l362_362008


namespace num_words_with_A_l362_362311

theorem num_words_with_A :
  let total_words := 5^4,
      words_without_A := 4^4 in
  total_words - words_without_A = 369 :=
by
  sorry

end num_words_with_A_l362_362311


namespace four_spheres_cover_rays_l362_362373

-- Define an abstract space with points and rays.
constant Point : Type
constant Sphere : Type
constant LightSource : Point → Prop
constant Ray : Point → Point → Prop
constant Intersect : Ray Point → Sphere → Prop

-- Define the main theorem.
theorem four_spheres_cover_rays (O : Point) (A B C D : Point) 
  (S1 S2 S3 S4 : Sphere) 
  (light_source : LightSource O)
  (ray_oa : Ray O A)
  (ray_ob : Ray O B)
  (ray_oc : Ray O C)
  (ray_od : Ray O D) :
  (∀ r : Ray O, r = ray_oa ∨ r = ray_ob ∨ r = ray_oc ∨ r = ray_od →
    (Intersect r S1 ∨ Intersect r S2 ∨ Intersect r S3 ∨ Intersect r S4)) :=
sorry

end four_spheres_cover_rays_l362_362373


namespace man_salary_l362_362513

variable (S : ℝ)

theorem man_salary (S : ℝ) (h1 : S - (1/3) * S - (1/4) * S - (1/5) * S = 1760) : S = 8123 := 
by 
  sorry

end man_salary_l362_362513


namespace sqrt_mul_sqrt_l362_362108

theorem sqrt_mul_sqrt (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : sqrt (a * sqrt b) = sqrt a * sqrt (sqrt b) := by
  sorry

example : sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  have h1 : sqrt 25 = 5 := by 
    norm_num
  have h2 : sqrt (49 * 5) = sqrt 245 := by  
    unfold sqrt
  have h3 : sqrt 245 = 7 * sqrt 5 := by 
    norm_num
  rw [h1, h2, h3]
  norm_num

end sqrt_mul_sqrt_l362_362108


namespace max_composite_numbers_l362_362780

theorem max_composite_numbers (s : set ℕ) (hs : ∀ n ∈ s, n < 1500 ∧ ∃ p : ℕ, prime p ∧ p ∣ n) (hs_gcd : ∀ x y ∈ s, x ≠ y → Nat.gcd x y = 1) :
  s.card ≤ 12 := 
by sorry

end max_composite_numbers_l362_362780


namespace triangle_inscribed_circle_ratio_l362_362457

theorem triangle_inscribed_circle_ratio
  (ABC : Triangle)
  (D : Point)
  (T : InscribedCircle ABC)
  (H1 : T.touches_side_at ABC.AC D)
  (second_circle : Circle)
  (H2 : second_circle.passes_through D)
  (H3 : second_circle.tangent_to_ray_at second_circle.BA ABC.A)
  (H4 : second_circle.touches_extension_of_side_beyond second_circle.BC C) :
  ratio AD DC = 3 :=
by sorry

end triangle_inscribed_circle_ratio_l362_362457


namespace simplify_expression_l362_362839

theorem simplify_expression (a b : ℚ) (ha : a = -1) (hb : b = 1/4) : 
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_expression_l362_362839


namespace log2_derivative_l362_362505

theorem log2_derivative (x : ℝ) (hx : x > 0) : 
  (deriv (λ x : ℝ, log 2 x)) x = 1 / (x * log 2) :=
sorry

end log2_derivative_l362_362505


namespace angle_FOG_eq_angle_A_l362_362693

variables {A B C D E F G O : Type} [Point A] [Triangle ABC]
variables (circumcenter : Circumcenter O (Triangle ABC))
variables (line_through_O : ∃ D E, Line O ∧ (Line AB D ∧ Line AC E)) -- Line intersect at D on AB and E on AC
variables (midpoint_B_E : Midpoint F B E)
variables (midpoint_C_D : Midpoint G C D)

theorem angle_FOG_eq_angle_A (triangle_ABC : Triangle ABC) (circumcenter_def : ∀ point : ∈ Line O, point ⟨= equidistant_from_vertices_O>)
  (line_O_D_E : ∃ D E, Line O ∧ (Line AB D ∧ Line AC E)) (F_def : Midpoint F B E) (G_def : Midpoint G C D) :
  ∠ F O G = ∠ A :=
sorry

end angle_FOG_eq_angle_A_l362_362693


namespace number_of_subsets_with_odd_sum_l362_362653

def S : Finset ℕ := {102, 107, 113, 139, 148, 159}

theorem number_of_subsets_with_odd_sum : (S.subsets.filter (λ s, s.card = 3 ∧ (s.sum % 2 = 1))).card = 8 :=
by
  sorry

end number_of_subsets_with_odd_sum_l362_362653


namespace exist_integers_not_div_by_7_l362_362393

theorem exist_integers_not_div_by_7 (k : ℕ) (hk : 0 < k) :
  ∃ (x y : ℤ), (¬ (7 ∣ x)) ∧ (¬ (7 ∣ y)) ∧ (x^2 + 6 * y^2 = 7^k) :=
sorry

end exist_integers_not_div_by_7_l362_362393


namespace minimize_distance_AP_BP_l362_362647

theorem minimize_distance_AP_BP :
  ∃ P : ℝ × ℝ, P.1 = 0 ∧ P.2 = -1 ∧
    ∀ P' : ℝ × ℝ, P'.1 = 0 → 
      (dist (3, 2) P + dist (1, -2) P) ≤ (dist (3, 2) P' + dist (1, -2) P') := by
sorry

end minimize_distance_AP_BP_l362_362647


namespace distance_to_nearest_river_l362_362981

theorem distance_to_nearest_river (d : ℝ) (h₁ : ¬ (d ≤ 12)) (h₂ : ¬ (d ≥ 15)) (h₃ : ¬ (d ≥ 10)) :
  12 < d ∧ d < 15 :=
by 
  sorry

end distance_to_nearest_river_l362_362981


namespace molecular_weight_of_compound_l362_362910

def atomic_weight_Al : Float := 26.98
def atomic_weight_O : Float := 16.00
def atomic_weight_Fe : Float := 55.85
def atomic_weight_H : Float := 1.01

def num_Al_atoms : Nat := 2
def num_O_atoms : Nat := 3
def num_Fe_atoms : Nat := 2
def num_H_atoms : Nat := 4

def molecular_weight (n_Al n_O n_Fe n_H : Nat) (w_Al w_O w_Fe w_H : Float) : Float :=
  (n_Al * w_Al) + (n_O * w_O) + (n_Fe * w_Fe) + (n_H * w_H)

theorem molecular_weight_of_compound :
  molecular_weight num_Al_atoms num_O_atoms num_Fe_atoms num_H_atoms atomic_weight_Al atomic_weight_O atomic_weight_Fe atomic_weight_H = 217.70 :=
by
  sorry

end molecular_weight_of_compound_l362_362910


namespace dasha_rectangle_l362_362580

theorem dasha_rectangle:
  ∃ (a b c : ℤ), a * (2 * b + 2 * c - a) = 43 ∧ a = 1 ∧ b + c = 22 :=
by
  sorry

end dasha_rectangle_l362_362580


namespace triangle_construction_l362_362616

-- Define the given points and circle
variables {M N P A B C O X : Point}
variables {circumcircle : Circle}

-- Define the properties of the circumcircle and points
axiom M_on_circumcircle : M ∈ circumcircle
axiom N_on_circumcircle : N ∈ circumcircle
axiom P_on_circumcircle : P ∈ circumcircle

-- Define the key geometric relationships and construction steps
axiom altitude_vertex : AltitudeIntersect (circumcircle.vertex) M
axiom angle_bisector : AngleBisectorIntersect (circumcircle.vertex) N
axiom median_vertex : MedianIntersect (circumcircle.vertex) P

-- The goal is to show the existence of a triangle ABC inscribed in the circumcircle
theorem triangle_construction : ∃ (A B C : Point), 
  A ∈ circumcircle ∧
  B ∈ circumcircle ∧
  C ∈ circumcircle ∧
  (
    (Line.through A P).intersects (Line.parallel (Line.through O N) M) = X ∧
    (Line.perpendicular X (Line.through O N)).intersects = {B, C}                     
  ) :=
sorry

end triangle_construction_l362_362616


namespace smallest_prime_factor_2023_l362_362913

def smallest_prime_factor (n : ℕ) : ℕ :=
  if h : ∃ p : ℕ, Nat.Prime p ∧ p ∣ n then
    Nat.find h
  else
    0

theorem smallest_prime_factor_2023 : smallest_prime_factor 2023 = 7 := 
by 
  sorry

end smallest_prime_factor_2023_l362_362913


namespace original_salary_l362_362931

theorem original_salary (S : ℝ) (h : 1.10 * S * 0.95 = 3135) : S = 3000 := 
by 
  sorry

end original_salary_l362_362931


namespace other_root_neg3_l362_362607

theorem other_root_neg3 (m : ℝ) (x : ℝ) : (x^2 + m*x + 6 = 0) → (-2) is root → x = -3 :=
by
  intro hm hroot
  have h_sum_roots : x + (-2) = -m := sorry -- Root sum property
  have h_product_roots : x * (-2) = 6 := sorry -- Root product property
  have hx : x = -3 := by
    calc
    x * (-2) = 6 : h_product_roots
    x = -3 : by sorry -- Solve for x
  exact hx

-- Provide a dummy proof, since we are not actually proving here

end other_root_neg3_l362_362607


namespace compute_N_mul_v1_l362_362709

open Matrix

noncomputable def N : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 2], ![3, 4]] -- A placeholder for matrix N

def v1 : Fin 2 → ℤ := ![7, 2]
def v2 : Fin 2 → ℤ := ![3, -2]
def v3 : Fin 2 → ℤ := ![-4, 6]
def v2_res : Fin 2 → ℤ := ![4, 1]
def v3_res : Fin 2 → ℤ := ![2, 3]
def expected : Fin 2 → ℤ := ![24, 11]

-- Assume the conditions of the problem
axiom condition1 : N.mul_vec v2 = v2_res
axiom condition2 : N.mul_vec v3 = v3_res

theorem compute_N_mul_v1 : N.mul_vec v1 = expected := by
  sorry

end compute_N_mul_v1_l362_362709


namespace unit_direction_vector_l362_362296

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def unit_vector_of_direction (v : ℝ × ℝ) : ℝ × ℝ :=
  let mag := magnitude v in
  (v.1 / mag, v.2 / mag)

theorem unit_direction_vector (a : ℝ × ℝ) (slope : ℝ) (line_eq : ℝ → ℝ) 
  (h : ∀ x, line_eq x = slope * x + 2) (ha₁ : a = (1, slope)) :
  unit_vector_of_direction a = (real.sqrt 5 / 5, 2 * real.sqrt 5 / 5) ∨
  unit_vector_of_direction a = (-real.sqrt 5 / 5, -2 * real.sqrt 5 / 5) :=
by
  sorry

end unit_direction_vector_l362_362296


namespace BE_greater_BF_l362_362421

-- Define the isosceles triangle and relevant points 
variables {A B C D E F : Type}
variables (P : Type)
variables [NormedAddTorsor ℝ P] [NormedSpace ℝ P]
variables [AffineSpace ℝ P]

-- Midpoint definition
def midpoint (a b : P) [AffineSpace ℝ P] : P := 
  lineMap a b (1 / 2 : ℝ)

-- Define the isosceles triangle with midpoint and perpendicular
variables 
(hisosceles : dist A B = dist B C)
(hmidpoint : D = midpoint A C)
(hperp : ∃ (E : P), E ≠ D ∧ dist D E ^ 2 + dist E C ^ 2 = dist D C ^ 2) -- E is the foot
(line_intersect :
  ∃ (F : P), ∃ (t u : ℝ),
    F = affineCombination ℝ P [A, E, B, D] [t, 1 - t, u, 1 - u])

-- Desired proof that BE > BF
theorem BE_greater_BF : dist B E > dist B F := sorry

end BE_greater_BF_l362_362421


namespace john_correct_answers_needed_l362_362353

theorem john_correct_answers_needed :
  -- conditions
  (total_questions = 30)
  (points_correct = 8)
  (points_incorrect = -2)
  (points_unanswered = 2)
  (answered_questions = 25)
  (unanswered_questions = 5)
  (required_points = 160)
  -- john's score calculation
  (unanswered_score = unanswered_questions * points_unanswered)
  (needed_score = required_points - unanswered_score)
  (incorrect_answers (correct_answers : ℕ) = answered_questions - correct_answers)
  (total_score (correct_answers : ℕ) = points_correct * correct_answers + points_incorrect * incorrect_answers correct_answers)
  -- condition
  (total_score x ≥ needed_score)
  -- what we need to prove
  ⊢ x ≥ 20 :=
sorry

end john_correct_answers_needed_l362_362353


namespace sin_alpha_beta_cos_2alpha_tan_half_beta_l362_362625

noncomputable def sin_alpha := -3 / 5
noncomputable def sin_beta := 12 / 13
noncomputable def alpha_gt_pi := π
noncomputable def alpha_lt_3pi2 := 3 * π / 2
noncomputable def beta_gt_pi2 := π / 2
noncomputable def beta_lt_pi := π

theorem sin_alpha_beta :
  sin_alpha = -3 / 5 →
  sin_beta = 12 / 13 →
  (π < α ∧ α < 3 * π / 2) →
  (π / 2 < β ∧ β < π) →
  sin (α - β) = 63 / 65 :=
sorry

theorem cos_2alpha :
  sin_alpha = -3 / 5 →
  (π < α ∧ α < 3 * π / 2) →
  cos (2 * α) = 7 / 25 :=
sorry

theorem tan_half_beta :
  sin_beta = 12 / 13 →
  (π / 2 < β ∧ β < π) →
  tan (β / 2) = 3 / 2 :=
sorry

end sin_alpha_beta_cos_2alpha_tan_half_beta_l362_362625


namespace sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l362_362089

theorem sqrt_49_mul_sqrt_25_eq_7_sqrt_5 : (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l362_362089


namespace max_non_attacking_bishops_l362_362562

theorem max_non_attacking_bishops (n : ℕ) (h : n ≥ 2) : 
  ∃ B : finset (ℕ × ℕ), B.card = 2 * n - 2 ∧ ∀ p q ∈ B, p ≠ q → (p.1 - p.2 ≠ q.1 - q.2 ∧ p.1 + p.2 ≠ q.1 + q.2) := 
sorry

end max_non_attacking_bishops_l362_362562


namespace table_tennis_total_rounds_l362_362902

-- Mathematical equivalent proof problem in Lean 4 statement
theorem table_tennis_total_rounds
  (A_played : ℕ) (B_played : ℕ) (C_referee : ℕ) (total_rounds : ℕ)
  (hA : A_played = 5) (hB : B_played = 4) (hC : C_referee = 2) :
  total_rounds = 7 :=
by
  -- Proof omitted
  sorry

end table_tennis_total_rounds_l362_362902


namespace sqrt_49_mul_sqrt_25_l362_362002

theorem sqrt_49_mul_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_l362_362002


namespace four_letter_words_with_A_l362_362310

theorem four_letter_words_with_A :
  let letters := ['A', 'B', 'C', 'D', 'E']
  in let total_4_letter_words := 5^4
  in let words_without_A := 4^4
  in total_4_letter_words - words_without_A = 369 := by
  sorry

end four_letter_words_with_A_l362_362310


namespace largest_C_inequality_l362_362248

theorem largest_C_inequality :
  ∃ C : ℝ, C = Real.sqrt (8 / 3) ∧ ∀ x y z : ℝ, x^2 + y^2 + z^2 + 2 ≥ C * (x + y + z) :=
by
  sorry

end largest_C_inequality_l362_362248


namespace number_of_students_taking_test_paper_C_l362_362538

variable (n : ℕ)

/-- The sequence of selected student numbers follows this arithmetic progression. -/
def a_n : ℕ := 20 * n - 2

/-- Condition for the students who take test paper C. -/
def is_test_paper_C (n : ℕ) : Prop :=
  561 ≤ a_n n ∧ a_n n ≤ 800

/-- Main theorem: Prove the number of students taking test paper C is 12. -/
theorem number_of_students_taking_test_paper_C :
  {n // is_test_paper_C n}.card = 12 :=
by
  sorry

end number_of_students_taking_test_paper_C_l362_362538


namespace wood_planks_for_legs_l362_362387

theorem wood_planks_for_legs (total_planks : ℕ) (tables : ℕ) (surface_planks_per_table : ℕ) (legs_planks_per_table : ℕ) 
  (h1 : total_planks = 45) (h2 : tables = 5) (h3 : surface_planks_per_table = 5) :
  legs_planks_per_table = 4 :=
by
  -- Conditions are already given
  let total_surface_planks := tables * surface_planks_per_table
  have h4 : total_surface_planks = 25, by sorry -- equivalent calculation to solution step 1
  let total_legs_planks := total_planks - total_surface_planks
  have h5 : total_legs_planks = 20, by sorry -- equivalent calculation to solution step 2
  let legs_planks_per_table := total_legs_planks / tables
  have h6 : legs_planks_per_table = 4, by sorry -- equivalent calculation to solution step 3
  exact h6

end wood_planks_for_legs_l362_362387


namespace z_solves_equation_l362_362601

noncomputable def z : ℂ := Complex.cis (π / 3)

theorem z_solves_equation :
  let solutions : Set ℂ := {Complex.cis (π / 3), Complex.cis (π / 3 + 2 * π / 3), Complex.cis (π / 3 + 4 * π / 3), 
    Complex.cis (π / 3 + 6 * π / 3), Complex.cis (π / 3 + 8 * π / 3),  Complex.cis (π / 3 + 10 * π / 3)} in
    ∀ (z : ℂ), z^6 = -64 → z ∈ solutions :=
begin
  sorry
end

end z_solves_equation_l362_362601


namespace find_original_denominator_l362_362173

noncomputable def original_denominator (d : ℕ) : Prop :=
  (10 / (d + 7) = 2 / 5)

theorem find_original_denominator : ∃ (d : ℕ), original_denominator d ∧ d = 18 := 
begin
  use 18,
  unfold original_denominator,
  field_simp,
  linarith,
  sorry
end

end find_original_denominator_l362_362173


namespace max_composite_numbers_l362_362794

-- Definitions and conditions
def is_composite (n : ℕ) : Prop := 2 < n ∧ ∃ d, d ∣ n ∧ 1 < d ∧ d < n

def less_than_1500 (n : ℕ) : Prop := n < 1500

def gcd_is_one (a b : ℕ) : Prop := Int.gcd a b = 1

-- The problem statement
theorem max_composite_numbers (numbers : List ℕ) (h_composite : ∀ n ∈ numbers, is_composite n) 
  (h_less_than_1500 : ∀ n ∈ numbers, less_than_1500 n)
  (h_pairwise_gcd_is_one : (numbers.Pairwise gcd_is_one)) : numbers.length ≤ 12 := 
  sorry

end max_composite_numbers_l362_362794


namespace triangle_sides_condition_triangle_perimeter_l362_362724

theorem triangle_sides_condition (a b c : ℝ) (A B C : ℝ) 
  (h1 : sin C * sin (A - B) = sin B * sin (C - A)) : 2 * a^2 = b^2 + c^2 :=
sorry

theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ) 
  (h2 : a = 5) (h3 : cos A = 25 / 31) : a + b + c = 14 :=
sorry

end triangle_sides_condition_triangle_perimeter_l362_362724


namespace correct_options_l362_362297

variables (Q : ℝ × ℝ) (C F P : ℝ × ℝ)

def parabola := ∃ a : ℝ, C = (λ (x : ℝ), (x * x - 4 * x) = 0)
def focus := F = (1, 0)
def pointP := P = (-2, 1)
def circle_tangent := ∀ (Q : ℝ × ℝ), Q = (λ (x : ℝ), (x, 2 * sqrt (x - 1))) → sphere (Q.1, Q.2) (abs ((Q.1 - 1))) (1)  
def perp_bisector := ∃ l : ℝ → ℝ, p = (-1 / 2, 1 / 2) ∧ l = (3x - y + 2 = 0)

theorem correct_options :
  (circle_tangent Q C F) ∧ (perp_bisector P F) := by
   sorry

end correct_options_l362_362297


namespace original_faculty_members_approx_l362_362170

noncomputable def original_faculty_members : ℝ :=
  let X := 195
  let first_reduction := 0.75 * X
  let after_hiring := first_reduction + 35
  let second_reduction := 0.85 * after_hiring
  second_reduction

theorem original_faculty_members_approx (X Y : ℝ) (H₁ : X = original_faculty_members) 
  (H₂ : Y ≈ 259) : X ≈ Y :=
by 
  sorry

end original_faculty_members_approx_l362_362170


namespace sqrt_49_times_sqrt_25_eq_7sqrt5_l362_362077

theorem sqrt_49_times_sqrt_25_eq_7sqrt5 :
  (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  have h1 : Real.sqrt 25 = 5 := by sorry
  have h2 : 49 * 5 = 245 := by sorry
  have h3 : Real.sqrt 245 = 7 * Real.sqrt 5 := by sorry
  sorry

end sqrt_49_times_sqrt_25_eq_7sqrt5_l362_362077


namespace number_of_real_solutions_l362_362708

theorem number_of_real_solutions (floor : ℝ → ℤ) 
  (h_floor : ∀ x, floor x = ⌊x⌋)
  (h_eq : ∀ x, 9 * x^2 - 45 * floor (x^2 - 1) + 94 = 0) :
  ∃ n : ℕ, n = 2 :=
by
  sorry

end number_of_real_solutions_l362_362708


namespace sqrt_expression_simplified_l362_362014

theorem sqrt_expression_simplified :
  sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  sorry

end sqrt_expression_simplified_l362_362014


namespace max_regions_by_five_spheres_l362_362372

noncomputable def a (n : ℕ) : ℕ :=
  if n = 2 then 4 else a (n - 1) + 2 * (n - 1)

noncomputable def b (n : ℕ) : ℕ :=
  if n = 2 then 4 else b (n - 1) + a (n - 1)

theorem max_regions_by_five_spheres : b 5 = 30 := by
  sorry

end max_regions_by_five_spheres_l362_362372


namespace john_sleep_hours_for_second_exam_l362_362696

def inverse_relationship (s1 s2 h1 h2 : ℝ) : Prop :=
  s1 * h1 = s2 * h2

-- Given conditions.
def sleep_score_first_exam : ℝ := 8
def score_first_exam : ℝ := 70
def average_score : ℝ := 80
def required_avg_score : Prop := (score_first_exam + 90) / 2 = average_score

-- Main problem statement.
theorem john_sleep_hours_for_second_exam :
  (inverse_relationship score_first_exam 90 sleep_score_first_exam h2) →
  ∃ h2, abs(h2 - 6.2) < 0.1 :=
begin
  sorry
end

end john_sleep_hours_for_second_exam_l362_362696


namespace hyperbola_parabola_parameters_l362_362620

noncomputable def hyperbola_eq (a b x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
noncomputable def parabola_eq (x y : ℝ) : Prop := y^2 = 4 * x
noncomputable def parabola_focus (x y : ℝ) := x = 1 ∧ y = 0
noncomputable def parabola_directrix (x y : ℝ) (d : ℝ) := d = x + 1
noncomputable def eccentricity (a b : ℝ) : ℝ := real.sqrt (1 + b^2 / a^2)

theorem hyperbola_parabola_parameters {a b e x0 y0 : ℝ} (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : hyperbola_eq a b x0 y0) (h4 : parabola_eq x0 y0) 
  (h5 : parabola_directrix x0 y0 2) 
  (h6 : e = eccentricity a b) : 
  2 * e - b^2 = 4 :=
sorry

end hyperbola_parabola_parameters_l362_362620


namespace sum_of_binomials_l362_362922

-- Definitions converted from the conditions
def largest_of_form (n : ℕ) (k : ℕ) : ℕ :=
  (n / k) * k

def C (n k : ℕ) : ℕ := Nat.binomial n k

-- Lean statement of the proof problem
theorem sum_of_binomials (n m : ℕ) (ω : ℂ) (fourth_root : ω^4 = 1)
  (omega_values : Set { z : ℂ // z^4 = 1 })
  (p1 : (1 + 1 : ℂ)^n = 2^n)
  (p2 : (1 + ω)^n = (ω + 1)^n)
  (p3 : (1 + ω^2 : ℂ)^n = (1 - 1)^n)
  (p4 : (1 + ω^3 : ℂ)^n = (1 - ω)^n) :
  ∑ k in Finset.range m, C n (4 * k + 3) =
    (2^n + ω * 2^n - (-ω^2)^n - ω * (-ω)^n) / (2 * (ω - ω^3)) := by
  sorry

end sum_of_binomials_l362_362922


namespace sqrt_expression_simplified_l362_362007

theorem sqrt_expression_simplified :
  sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  sorry

end sqrt_expression_simplified_l362_362007


namespace max_composite_numbers_with_gcd_one_l362_362764

theorem max_composite_numbers_with_gcd_one : 
  ∃ (S : Finset ℕ), 
    (∀ x ∈ S, Nat.isComposite x) ∧ 
    (∀ x ∈ S, x < 1500) ∧ 
    (∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y → Nat.gcd x y = 1) ∧
    S.card = 12 :=
sorry

end max_composite_numbers_with_gcd_one_l362_362764


namespace max_composite_numbers_l362_362776

theorem max_composite_numbers (s : set ℕ) (hs : ∀ n ∈ s, n < 1500 ∧ ∃ p : ℕ, prime p ∧ p ∣ n) (hs_gcd : ∀ x y ∈ s, x ≠ y → Nat.gcd x y = 1) :
  s.card ≤ 12 := 
by sorry

end max_composite_numbers_l362_362776


namespace not_all_rectangles_are_squares_l362_362989

-- Definitions based on conditions
def square (a : Type) [has_le a] [has_zero a] := 
  ∀ (s : a), (∀ (x : a), s ≤ x ∧ x ≤ s) → 
    (∀ (a b c d : a), angle(a, b, c, d) = 90) ∧ equalLength(a, b, c, d)

def rectangle (a : Type) [has_le a] [has_zero a] := 
  ∀ (r : a), (∀ (x : a), r ≤ x ∧ x ≤ r) → 
    (∀ (a b : a), angle(a, b) = 90) → (∀ (a b c d : a), sides(a, b, c, d) = equalOpposite)

-- The false statement based on the conditions
theorem not_all_rectangles_are_squares :
  ¬ ∀ (r : Type), rectangle r → square r :=
begin
  sorry
end

end not_all_rectangles_are_squares_l362_362989


namespace line_through_M_has_opposite_intercepts_l362_362247

-- Define the conditions for the problem
def point_M := (3 : ℝ, -4 : ℝ)

def line_passes_through_point (line_eq : ℝ → ℝ → Prop) (p : ℝ × ℝ) : Prop :=
  line_eq p.1 p.2

def opposite_intercepts (line_eq : ℝ → ℝ → Prop) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ (line_eq a 0) ∧ (line_eq 0 (-a))

-- The statement to prove in Lean
theorem line_through_M_has_opposite_intercepts :
  (∃ line_eq : ℝ → ℝ → Prop, 
    line_passes_through_point line_eq.point_M ∧ opposite_intercepts line_eq ∧
     ((line_eq = λ x y, x + y + 1 = 0) ∨ (line_eq = λ x y, 4 * x + 3 * y = 0)) :=
sorry

end line_through_M_has_opposite_intercepts_l362_362247


namespace simplify_and_evaluate_l362_362851

theorem simplify_and_evaluate 
  (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) = 1 :=
by
  sorry

end simplify_and_evaluate_l362_362851


namespace rate_of_return_proof_l362_362586

variable (r : ℝ)

-- Conditions
def total_investment : ℝ := 33000
def total_interest : ℝ := 970
def invested_at_r : ℝ := 13000
def invested_at_2_25_percent : ℝ := total_investment - invested_at_r
def rate_of_return_2_25_percent : ℝ := 0.0225

-- Definition of interest calculations
def interest_from_r : ℝ := invested_at_r * r
def interest_from_2_25_percent : ℝ := invested_at_2_25_percent * rate_of_return_2_25_percent

-- Equation resulting from the conditions
def interest_equation : Prop := interest_from_r + interest_from_2_25_percent = total_interest

-- Mathematically equivalent proof problem statement
theorem rate_of_return_proof : interest_equation r ∧ r = 0.04 :=
by
  sorry

end rate_of_return_proof_l362_362586


namespace speed_of_boat_in_still_water_l362_362468

theorem speed_of_boat_in_still_water
    (speed_stream : ℝ)
    (distance_downstream : ℝ)
    (distance_upstream : ℝ)
    (t : ℝ)
    (x : ℝ)
    (h1 : speed_stream = 10)
    (h2 : distance_downstream = 80)
    (h3 : distance_upstream = 40)
    (h4 : t = distance_downstream / (x + speed_stream))
    (h5 : t = distance_upstream / (x - speed_stream)) :
  x = 30 :=
by sorry

end speed_of_boat_in_still_water_l362_362468


namespace pepperoni_slices_l362_362698

theorem pepperoni_slices (total_slices : ℕ) (cut_in_half twice : ℕ) (slice_falls_off : ℕ) : 
  total_slices = 40 → 
  cut_in_half = total_slices / 2 → 
  twice = cut_in_half / 2 →
  slice_falls_off = 1 →
  twice - slice_falls_off = 9 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end pepperoni_slices_l362_362698


namespace smallest_difference_l362_362225

theorem smallest_difference (a b : ℕ) (h₁ : a * b = 1728) : 
  ∃ d, d = 12 ∧ (∀ a' b' : ℕ, a' * b' = 1728 → abs (a' - b') ≥ d) :=
sorry

end smallest_difference_l362_362225


namespace max_composite_numbers_l362_362803
open Nat

def is_composite (n : ℕ) : Prop := 1 < n ∧ ∃ d : ℕ, 1 < d ∧ d < n ∧ d ∣ n

def has_gcd_of_one (l : List ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ l → b ∈ l → a ≠ b → gcd a b = 1

def valid_composite_numbers (n : ℕ) : Prop :=
  ∀ m ∈ (List.range n).filter is_composite, m < 1500 →

-- Main theorem
theorem max_composite_numbers :
  ∃ l : List ℕ, l.length = 12 ∧ valid_composite_numbers l ∧ has_gcd_of_one l :=
sorry

end max_composite_numbers_l362_362803


namespace simplify_and_evaluate_l362_362849

theorem simplify_and_evaluate 
  (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) = 1 :=
by
  sorry

end simplify_and_evaluate_l362_362849


namespace subcommittees_count_l362_362316

theorem subcommittees_count 
  (n : ℕ) (k : ℕ) (hn : n = 7) (hk : k = 3) : 
  (nat.choose n k) = 35 := by 
  have h1 : 7 = 7 := rfl
  have h2 : 3 = 3 := rfl
  sorry

end subcommittees_count_l362_362316


namespace cyclical_permutation_divisible_by_41_l362_362147

theorem cyclical_permutation_divisible_by_41 
  (A B C D E : ℕ) 
  (h₀ : (10000 * A + 1000 * B + 100 * C + 10 * D + E) % 41 = 0) :
  let N := 10000 * A + 1000 * B + 100 * C + 10 * D + E in
  ((10000 * B + 1000 * C + 100 * D + 10 * E + A) % 41 = 0) ∧
  ((10000 * C + 1000 * D + 100 * E + 10 * A + B) % 41 = 0) ∧
  ((10000 * D + 1000 * E + 100 * A + 10 * B + C) % 41 = 0) ∧
  ((10000 * E + 1000 * A + 100 * B + 10 * C + D) % 41 = 0) := 
by
  sorry

end cyclical_permutation_divisible_by_41_l362_362147


namespace complex_point_location_l362_362657

theorem complex_point_location (a b : ℝ) : 
(a^2 - 6*a + 10 > 0) → 
(-b^2 + 4*b - 5 < 0) → 
complex.quadrant ((a^2 - 6*a + 10) + (-b^2 + 4*b - 5) * complex.I) = complex.quadrant.fourth := 
sorry

end complex_point_location_l362_362657


namespace power_of_a_power_evaluate_3_pow_3_pow_2_l362_362232

theorem power_of_a_power (a m n : ℕ) : (a^m)^n = a^(m*n) := 
begin
  sorry,
end

theorem evaluate_3_pow_3_pow_2 : (3^3)^2 = 729 := 
begin
  have H1 : (3^3)^2 = 3^(3*2) := power_of_a_power 3 3 2,
  have H2 : 3^(3*2) = 3^6 := by refl,
  have H3 : 3^6 = 729 := by norm_num,
  exact eq.trans (eq.trans H1 H2) H3,
end

end power_of_a_power_evaluate_3_pow_3_pow_2_l362_362232


namespace calculate_decimal_sum_and_difference_l362_362570

theorem calculate_decimal_sum_and_difference : 
  (0.5 + 0.003 + 0.070) - 0.008 = 0.565 := 
by 
  sorry

end calculate_decimal_sum_and_difference_l362_362570


namespace coachClass_seats_count_l362_362986

-- Defining the conditions as given in a)
variables (F : ℕ) -- Number of first-class seats
variables (totalSeats : ℕ := 567) -- Total number of seats is given as 567
variables (businessClassSeats : ℕ := 3 * F) -- Business class seats defined in terms of F
variables (coachClassSeats : ℕ := 7 * F + 5) -- Coach class seats defined in terms of F
variables (firstClassSeats : ℕ := F) -- The variable itself

-- The statement to prove
theorem coachClass_seats_count : 
  F + businessClassSeats + coachClassSeats = totalSeats →
  coachClassSeats = 362 :=
by
  sorry -- The proof would go here

end coachClass_seats_count_l362_362986


namespace gym_membership_count_l362_362960

theorem gym_membership_count :
  let charge_per_time := 18
  let times_per_month := 2
  let total_monthly_income := 10800
  let amount_per_member := charge_per_time * times_per_month
  let number_of_members := total_monthly_income / amount_per_member
  number_of_members = 300 :=
by
  let charge_per_time := 18
  let times_per_month := 2
  let total_monthly_income := 10800
  let amount_per_member := charge_per_time * times_per_month
  let number_of_members := total_monthly_income / amount_per_member
  sorry

end gym_membership_count_l362_362960


namespace adjacent_squares_difference_at_least_n_l362_362679

theorem adjacent_squares_difference_at_least_n (n : ℕ) (h : n > 0) :
  ∃ (grid : Fin n.succ × Fin n.succ → ℕ), 
  (∀ i j, 1 ≤ grid(i, j) ∧ grid(i, j) ≤ n^2) ∧ 
  (∀ i j k, grid(i, j) ≠ grid(k, (j + k % 2).mod n)) ∧ 
  (∃ i j, abs (grid(i, j) - grid(i + 1, j)) ≥ n ∨ abs (grid(i, j) - grid(i, j + 1)) ≥ n) := 
sorry

end adjacent_squares_difference_at_least_n_l362_362679


namespace combinations_of_coins_sum_to_50_l362_362650

theorem combinations_of_coins_sum_to_50 (p n d : ℕ) :
  (∑ d in (range 6), (∑ k in (range ((50 - 10 * d) / 5 + 1)), (50 - 10 * d - 5 * k + 1))) = 2933 :=
  sorry

end combinations_of_coins_sum_to_50_l362_362650


namespace count_4_letter_words_with_A_l362_362302

-- Define the alphabet set and the properties
def Alphabet : Finset (Char) := {'A', 'B', 'C', 'D', 'E'}
def total_words := (Alphabet.card ^ 4 : ℕ)
def total_words_without_A := (Alphabet.erase 'A').card ^ 4
def total_words_with_A := total_words - total_words_without_A

-- The key theorem to prove
theorem count_4_letter_words_with_A : total_words_with_A = 369 := sorry

end count_4_letter_words_with_A_l362_362302


namespace sqrt_product_l362_362090

theorem sqrt_product (a b : ℝ) (h1 : a = 49) (h2 : b = 25) : sqrt (a * sqrt b) = 7 * sqrt 5 :=
by sorry

end sqrt_product_l362_362090


namespace milk_water_equal_l362_362480

theorem milk_water_equal (a : ℕ) :
  let glass_a_initial := a
  let glass_b_initial := a
  let mixture_in_a := glass_a_initial + 1
  let milk_portion_in_a := 1 / mixture_in_a
  let water_portion_in_a := glass_a_initial / mixture_in_a
  let water_in_milk_glass := water_portion_in_a
  let milk_in_water_glass := milk_portion_in_a
  water_in_milk_glass = milk_in_water_glass := by
  sorry

end milk_water_equal_l362_362480


namespace find_radius_of_incircle_l362_362488

noncomputable def radius_of_incircle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
   (triangle : EuclideanGeometry.Triangle A B C) 
   (right_angle : triangle.C.angle = 90) 
   (angle_A : triangle.A.angle = 45) 
   (length_AC : triangle.AC.length = 12) : ℝ :=
6 - 3 * Math.sqrt 2

theorem find_radius_of_incircle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
   (triangle : EuclideanGeometry.Triangle A B C) 
   (right_angle : triangle.C.angle = 90) 
   (angle_A : triangle.A.angle = 45) 
   (length_AC : triangle.AC.length = 12) :
   radius_of_incircle A B C triangle right_angle angle_A length_AC = 6 - 3 * Math.sqrt 2 :=
by
  sorry

end find_radius_of_incircle_l362_362488


namespace least_odd_prime_factor_of_2023_8_plus_1_l362_362595

-- Define the example integers and an assumption for modular arithmetic
def n : ℕ := 2023
def p : ℕ := 97

-- Conditions and the theorem statement
theorem least_odd_prime_factor_of_2023_8_plus_1 :
  n ^ 8 ≡ -1 [MOD p] →
  ∀ q, prime q → q ∣ (n ^ 8 + 1) → q ≥ p :=
by
  sorry

end least_odd_prime_factor_of_2023_8_plus_1_l362_362595


namespace sqrt_mul_sqrt_l362_362107

theorem sqrt_mul_sqrt (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : sqrt (a * sqrt b) = sqrt a * sqrt (sqrt b) := by
  sorry

example : sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  have h1 : sqrt 25 = 5 := by 
    norm_num
  have h2 : sqrt (49 * 5) = sqrt 245 := by  
    unfold sqrt
  have h3 : sqrt 245 = 7 * sqrt 5 := by 
    norm_num
  rw [h1, h2, h3]
  norm_num

end sqrt_mul_sqrt_l362_362107


namespace local_extrema_diff_bounds_l362_362129

noncomputable def F (x : ℝ) (a1 a2 a0 : ℝ) := x^4 + a1 * x^3 + a2 * x^2 + a1 * x + a0

theorem local_extrema_diff_bounds (a1 a2 a0 M m : ℝ) (x : ℝ) :
  let F := F x a1 a2 a0 in
  (is_local_max F x M) → (is_local_min F x m) →
  (3/10) * ((a1^2 / 4) - (2 * a2 / 3))^2 < M - m ∧ M - m < 3 * ((a1^2 / 4) - (2 * a2 / 3))^2 :=
sorry

end local_extrema_diff_bounds_l362_362129


namespace symmetry_center_l362_362637

def f (x φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ)

theorem symmetry_center (φ x : ℝ) (hφ : |φ| < Real.pi / 2)
  (hpoint : f 0 φ = Real.sqrt 3) : 
  (x = -Real.pi / 6 ∧ f x φ = 0) :=
sorry

end symmetry_center_l362_362637


namespace james_payment_l362_362378

theorem james_payment (james_meal : ℕ) (friend_meal : ℕ) (tip_percent : ℕ) (final_payment : ℕ) : 
  james_meal = 16 → 
  friend_meal = 14 → 
  tip_percent = 20 → 
  final_payment = 18 :=
by
  -- Definitions
  let total_bill_before_tip := james_meal + friend_meal
  let tip := total_bill_before_tip * tip_percent / 100
  let final_bill := total_bill_before_tip + tip
  let half_bill := final_bill / 2
  -- Proof (to be filled in)
  sorry

end james_payment_l362_362378


namespace hexagon_opposite_sides_equal_l362_362444

theorem hexagon_opposite_sides_equal
  (x1 x2 x3 x4 x5 x6 : ℝ)
  (h : ∃ (A B : Type) [equilateral_triangle A] [equilateral_triangle B], 
    is_hexagon A B (set_of {x1, x2, x3, x4, x5, x6})) :
  x1 + x3 + x5 = x2 + x4 + x6 :=
  sorry

end hexagon_opposite_sides_equal_l362_362444


namespace total_cards_1750_l362_362942

theorem total_cards_1750 (football_cards baseball_cards hockey_cards total_cards : ℕ)
  (h1 : baseball_cards = football_cards - 50)
  (h2 : football_cards = 4 * hockey_cards)
  (h3 : hockey_cards = 200)
  (h4 : total_cards = football_cards + baseball_cards + hockey_cards) :
  total_cards = 1750 :=
sorry

end total_cards_1750_l362_362942


namespace asian_games_volunteer_selection_l362_362360

-- Define the conditions.

def total_volunteers : ℕ := 5
def volunteer_A_cannot_serve_language_services : Prop := true

-- Define the main problem.
-- We are supposed to find the number of ways to assign three roles given the conditions.
def num_ways_to_assign_roles : ℕ :=
  let num_ways_language_services := 4 -- A cannot serve this role, so 4 choices
  let num_ways_other_roles := 4 * 3 -- We need to choose and arrange 2 volunteers out of remaining
  num_ways_language_services * num_ways_other_roles

-- The target theorem.
theorem asian_games_volunteer_selection : num_ways_to_assign_roles = 48 :=
by
  sorry

end asian_games_volunteer_selection_l362_362360


namespace babysitting_rate_per_hour_l362_362119

def bike_cost : ℕ := 100
def weekly_allowance : ℕ := 5
def mowing_payment : ℕ := 10
def hours_babysitting : ℕ := 2
def current_savings : ℕ := 65
def needed_money : ℕ := 6

theorem babysitting_rate_per_hour (neighbor_pay_rate : ℕ) : neighbor_pay_rate = 3 / 2 :=
by
    have step1 : current_savings + needed_money = bike_cost := by trivial
    have step2 : current_savings + weekly_allowance + mowing_payment = current_savings + 5 + 10 := by trivial
    have step3 : current_savings + weekly_allowance + mowing_payment = 65 + 5 + 10 := by trivial
    have step4 : current_savings + weekly_allowance + mowing_payment = 80 := by trivial
    have step5 : 80 - (current_savings + needed_money) = 9 := by trivial
    have step6 : neighbor_pay_rate = 9 / hours_babysitting := by trivial
    have step7 : neighbor_pay_rate = 9 / 2 := by trivial
    have final_step : neighbor_pay_rate = 3 / 2 := by trivial
    exact final_step

end babysitting_rate_per_hour_l362_362119


namespace simplify_expression_l362_362836

theorem simplify_expression :
  ((5 ^ 7 + 2 ^ 8) * (1 ^ 5 - (-1) ^ 5) ^ 10) = 80263680 := by
  sorry

end simplify_expression_l362_362836


namespace prove_y_minus_x_l362_362891

theorem prove_y_minus_x (x y : ℚ) (h1 : x + y = 500) (h2 : x / y = 7 / 8) : y - x = 100 / 3 := 
by
  sorry

end prove_y_minus_x_l362_362891


namespace relationship_among_abc_l362_362627

noncomputable def a : ℝ := ∫ x in 0..2, x^2
noncomputable def b : ℝ := ∫ x in 0..2, Real.exp x
noncomputable def c : ℝ := ∫ x in 0..2, Real.sin x

theorem relationship_among_abc : c < a ∧ a < b := by
  have h₁ : a = 8 / 3 := by
    show a = ∫ x in 0..2, x^2
    sorry

  have h₂ : b = Real.exp 2 - 1 := by
    show b = ∫ x in 0..2, Real.exp x
    sorry

  have h₃ : c = 1 - Real.cos 2 := by
    show c = ∫ x in 0..2, Real.sin x
    sorry

  have : 2 < 8 / 3 ∧ 8 / 3 < Real.exp 2 - 1 := by
    sorry

  have : 1 < 1 - Real.cos 2 ∧ 1 - Real.cos 2 < 2 := by
    sorry

  exact ⟨this.right, this.left⟩

end relationship_among_abc_l362_362627


namespace polynomial_has_real_root_l362_362582

open Real

theorem polynomial_has_real_root (a : ℝ) : 
  ∃ x : ℝ, x^5 + a * x^4 - x^3 + a * x^2 - x + a = 0 :=
sorry

end polynomial_has_real_root_l362_362582


namespace common_point_of_function_and_inverse_l362_362984

-- Define the points P, Q, M, and N
def P : ℝ × ℝ := (1, 1)
def Q : ℝ × ℝ := (1, 2)
def M : ℝ × ℝ := (2, 3)
def N : ℝ × ℝ := (0.5, 0.25)

-- Define a predicate to check if a point lies on the line y = x
def lies_on_y_eq_x (point : ℝ × ℝ) : Prop := point.1 = point.2

-- The main theorem statement
theorem common_point_of_function_and_inverse (a : ℝ) : 
  lies_on_y_eq_x P ∧ ¬ lies_on_y_eq_x Q ∧ ¬ lies_on_y_eq_x M ∧ ¬ lies_on_y_eq_x N :=
by
  -- We write 'sorry' here to skip the proof
  sorry

end common_point_of_function_and_inverse_l362_362984


namespace segments_divided_16_times_l362_362544

theorem segments_divided_16_times :
  let n := 16 in
  let initial_length := 1 in
  let division_factor := 3 in
  let remaining_factor := 2 / 3 in
  let final_length := 1 / (division_factor ^ n) in
  let final_segment_count := 2 ^ n in
  number_of_segments (initial_length : ℝ) division_factor n = final_segment_count ∧ final_segment_length (initial_length : ℝ) division_factor n = final_length :=
by
  sorry

def number_of_segments (initial_length : ℝ) (division_factor : ℕ) (n : ℕ) : ℕ :=
  2 ^ n

def final_segment_length (initial_length : ℝ) (division_factor : ℕ) (n : ℕ) : ℝ :=
  initial_length / (division_factor ^ n)

end segments_divided_16_times_l362_362544


namespace prove_y_minus_x_l362_362619

-- Definitions as conditions
def four_sided_pyramid (colors : List String) :=
  (colors.length = 5 ∨ colors.length = 4) ∧ 
  ∀ (i j k l : ℕ), i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l ∧ 
  colors.nth i ≠ colors.nth j → 
  colors.nth i ≠ colors.nth k → 
  colors.nth i ≠ colors.nth l → 
  colors.nth j ≠ colors.nth k → 
  colors.nth j ≠ colors.nth l → 
  colors.nth k ≠ colors.nth l

def coloring_methods (colors : List String) : ℕ := 
  if colors.length = 5 
  then 420 
  else if colors.length = 4 
  then 72 
  else 0                                                        

-- The mathematical proof statement
noncomputable def y_minus_x_correct : Prop :=
  ∀ (colors : List String),
    four_sided_pyramid colors →
    let y := coloring_methods (colors.filter (λ c, colors.length = 5)),
    let x := coloring_methods (colors.filter (λ c, colors.length = 4)),
    y - x = 348

theorem prove_y_minus_x : y_minus_x_correct :=
  by sorry

end prove_y_minus_x_l362_362619


namespace incircle_radius_of_right_triangle_l362_362485

/-- Triangle ABC has a right angle at C, angle A = 45 degrees, and AC = 12. The radius of the incircle of triangle ABC is 12 - 6 * sqrt(2). -/
theorem incircle_radius_of_right_triangle
  (A B C : Type)
  (is_triangle : Triangle A B C)
  (right_angle_at_C : ∠ABC = 90)
  (angle_A_45_degrees : ∠BAC = 45)
  (AC_length : AC = 12) :
  incircle_radius (Triangle A B C) = 12 - 6 * real.sqrt 2 := 
sorry

end incircle_radius_of_right_triangle_l362_362485


namespace ranking_possibilities_l362_362887

theorem ranking_possibilities :
  ∃ (rank : Fin 5 → ℕ),
  (∀ i j, i ≠ j → rank i ≠ rank j) ∧
  (A_rank B_rank : Fin 5, rank A_rank < rank B_rank → rank B_rank = rank A_rank + 1 ∨ rank A_rank = rank B_rank + 1) ∧
  (C_rank D_rank : Fin 5, rank C_rank ≠ 0 ∧ rank D_rank ≠ 0) ∧
  (D_rank : Fin 5, rank D_rank ≠ 4) →
  fintype.card {rank // ∀ i j, i ≠ j → rank i ≠ rank j} = 16 :=
sorry

end ranking_possibilities_l362_362887


namespace find_15th_and_2014th_l362_362690

-- Definition of the sequence following the conditions provided
def sequence : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := let i := (n + 2) / 2 in
           if even (n + 2) then
             2 * i
           else
             2 * i - 1

theorem find_15th_and_2014th :
( sequence 15 = 25 ) ∧ ( sequence 2014 = 3965 ) :=
begin
  sorry
end

end find_15th_and_2014th_l362_362690


namespace max_area_parabola_l362_362273

open Real

noncomputable def max_area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1))

theorem max_area_parabola (a b c : ℝ) 
  (ha : a^2 = (a * a))
  (hb : b^2 = (b * b))
  (hc : c^2 = (c * c))
  (centroid_cond1 : (a + b + c) = 4)
  (centroid_cond2 : (a^2 + b^2 + c^2) = 6)
  : max_area_of_triangle (a^2, a) (b^2, b) (c^2, c) = (sqrt 3) / 9 := 
sorry

end max_area_parabola_l362_362273


namespace positive_difference_l362_362410

noncomputable def g (n : ℝ) : ℝ :=
if n < 0 then n^2 - 5 else 3 * n - 25

theorem positive_difference :
  let b1 := -Real.sqrt 17 in
  let b2 := 37 / 3 in
  g (-3) + g 3 + g b1 = 0 ∧ g (-3) + g 3 + g b2 = 0 →
  abs (b1 - b2) = Real.sqrt 17 + 37 / 3 :=
by
  sorry

end positive_difference_l362_362410


namespace polynomial_root_on_unit_circle_l362_362823

theorem polynomial_root_on_unit_circle (n : ℕ) (a b : ℂ) (h : n > 0) (ha : a ≠ 0) :
  ∃ z : ℂ, |z| = 1 ∧ (a * z^(2*n + 1) + b * z^(2*n) + conj b * z + conj a = 0) :=
by 
  sorry

end polynomial_root_on_unit_circle_l362_362823


namespace modified_euclidean_gcd_complexity_l362_362835

theorem modified_euclidean_gcd_complexity (a b : ℕ) : 
  ∃ (T : ℕ → ℕ → ℕ) (c : ℕ), 
  (∀ a b : ℕ, T a b = 0 ∨ T a b = T (if a % 2 = 0 then a / 2 else a) (if b % 2 = 0 then b / 2 else b) - 1) ∧ 
  (T a b = 0 ∨ T a b = (if a ≥ b then T (a - b) b else T a (b - a))) ∧
  T a b ≤ c * (log a)^2 + c * (log b)^2 :=
sorry

end modified_euclidean_gcd_complexity_l362_362835


namespace cos2theta_l362_362298

noncomputable def vector := ℝ

variables (a b : vector)
variables (norm_a : ‖a‖ = 10)
variables (norm_b : ‖b‖ = 15)
variables (norm_a_plus_b : ‖a + b‖ = 20)

theorem cos2theta (θ : ℝ) (h : θ = real.angle a b) : 
  real.cos (2 * θ) = -7/8 :=
sorry

end cos2theta_l362_362298


namespace four_letter_words_with_A_at_least_once_l362_362305

theorem four_letter_words_with_A_at_least_once (A B C D E : Type) :
  let total := 5^4 in
  let without_A := 4^4 in
  total - without_A = 369 :=
by {
  let total := 5^4;
  let without_A := 4^4;
  have : total - without_A = 369 := by sorry;
  exact this;
}

end four_letter_words_with_A_at_least_once_l362_362305


namespace student_weight_l362_362515

variable (S W : ℕ)

theorem student_weight (h1 : S - 5 = 2 * W) (h2 : S + W = 110) : S = 75 :=
by
  sorry

end student_weight_l362_362515


namespace arithmetic_progression_sum_l362_362503

theorem arithmetic_progression_sum (a d : ℝ) (n : ℕ) : 
  a + 10 * d = 5.25 → 
  a + 6 * d = 3.25 → 
  (n : ℝ) / 2 * (2 * a + (n - 1) * d) = 56.25 → 
  n = 15 :=
by
  intros h1 h2 h3
  sorry

end arithmetic_progression_sum_l362_362503


namespace sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l362_362081

theorem sqrt_49_mul_sqrt_25_eq_7_sqrt_5 : (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l362_362081


namespace james_payment_l362_362377

theorem james_payment (james_meal : ℕ) (friend_meal : ℕ) (tip_percent : ℕ) (final_payment : ℕ) : 
  james_meal = 16 → 
  friend_meal = 14 → 
  tip_percent = 20 → 
  final_payment = 18 :=
by
  -- Definitions
  let total_bill_before_tip := james_meal + friend_meal
  let tip := total_bill_before_tip * tip_percent / 100
  let final_bill := total_bill_before_tip + tip
  let half_bill := final_bill / 2
  -- Proof (to be filled in)
  sorry

end james_payment_l362_362377


namespace find_angle_ACB_l362_362347

-- Definitions used in Lean 4 statement
variables (A B C D : Type) [IsTriangle A B C]
variables (angle_ABC : ℕ) (BD : ℕ) (CD : ℕ)
variables (angle_DAB : ℕ) (theta : ℕ)

-- Conditions
hypothesis h1 : angle_ABC = 60
hypothesis h2 : 2 * BD = CD
hypothesis h3 : angle_DAB = 30

-- Proof goal: Show that using the given conditions, we can conclude angle ACB is 60 degrees.
theorem find_angle_ACB : θ = 60 := 
sorry

end find_angle_ACB_l362_362347


namespace sqrt_49_times_sqrt_25_l362_362018

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 :=
by
  sorry

end sqrt_49_times_sqrt_25_l362_362018


namespace can_determine_counterfeit_l362_362230

-- Define the conditions of the problem
structure ProblemConditions where
  totalCoins : ℕ := 100
  exaggeration : ℕ

-- Define the problem statement
theorem can_determine_counterfeit (P : ProblemConditions) : 
  ∃ strategy : ℕ → Prop, 
    ∀ (k : ℕ), strategy P.exaggeration -> 
    (∀ i, i < 100 → (P.totalCoins = 100 ∧ ∃ n, n > 0 ∧ 
     ∀ j, j < P.totalCoins → (P.totalCoins = j + 1 ∨ P.totalCoins = 99 + j))) := 
sorry

end can_determine_counterfeit_l362_362230


namespace amy_spent_32_l362_362985

theorem amy_spent_32 (x: ℝ) (h1: 0.15 * x + 1.6 * x + x = 55) : 1.6 * x = 32 :=
by
  sorry

end amy_spent_32_l362_362985


namespace xy_z_eq_inv_sqrt2_l362_362687

noncomputable def f (t : ℝ) : ℝ := (Real.sqrt 2) * t + 1 / ((Real.sqrt 2) * t)

theorem xy_z_eq_inv_sqrt2 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : (Real.sqrt 2) * x + 1 / ((Real.sqrt 2) * x) 
      + (Real.sqrt 2) * y + 1 / ((Real.sqrt 2) * y) 
      + (Real.sqrt 2) * z + 1 / ((Real.sqrt 2) * z) 
      = 6 - 2 * (Real.sqrt (2 * x)) * abs (y - z) 
            - (Real.sqrt (2 * y)) * (x - z) ^ 2 
            - (Real.sqrt (2 * z)) * (Real.sqrt (abs (x - y)))) :
  x = y ∧ y = z ∧ z = 1 / (Real.sqrt 2) :=
sorry

end xy_z_eq_inv_sqrt2_l362_362687


namespace parallelogram_side_length_l362_362464

theorem parallelogram_side_length (a b : ℕ) (h1 : 2 * (a + b) = 16) (h2 : a = 5) : b = 3 :=
by
  sorry

end parallelogram_side_length_l362_362464


namespace mono_increasing_interval_l362_362638

theorem mono_increasing_interval :
  ∀ (x : ℝ), 0 < x ∧ x < (5 * π / 12) →
    has_deriv_at (λ x : ℝ, sin (2 * x - π / 3)) (cos (2 * x - π / 3) * 2) x ∧ 
    cos (2 * x - π / 3) * 2 > 0 := 
by
  sorry

end mono_increasing_interval_l362_362638


namespace arithmetic_sequence_general_term_sum_of_first_n_terms_of_bn_l362_362362

theorem arithmetic_sequence_general_term (a : ℕ → ℤ) (b : ℕ → ℤ) (h₁ : a 2 + a 7 = -23) (h₂ : a 3 + a 8 = -29) :
  ∀ n, a n = -3 * n + 2 :=
begin
  -- will be proven here
  sorry
end

theorem sum_of_first_n_terms_of_bn (a : ℕ → ℤ) (b : ℕ → ℤ) (h₁ : a 2 + a 7 = -23) (h₂ : a 3 + a 8 = -29)
  (h₃ : ∀ n, a n + b n = 2^(n - 1)) :
  ∀ n, (finset.range n).sum b = (3 * n ^ 2 - n + 2 * (2 ^ n - 1)) / 2 :=
begin
  -- will be proven here
  sorry
end

end arithmetic_sequence_general_term_sum_of_first_n_terms_of_bn_l362_362362


namespace evaporation_days_l362_362139

theorem evaporation_days
    (initial_water : ℝ)
    (evap_rate : ℝ)
    (percent_evaporated : ℝ)
    (evaporated_water : ℝ)
    (days : ℝ)
    (h1 : initial_water = 10)
    (h2 : evap_rate = 0.012)
    (h3 : percent_evaporated = 0.06)
    (h4 : evaporated_water = initial_water * percent_evaporated)
    (h5 : days = evaporated_water / evap_rate) :
  days = 50 :=
by
  sorry

end evaporation_days_l362_362139


namespace jackson_holidays_l362_362374

theorem jackson_holidays (holidays_per_month : ℕ) (months_per_year : ℕ) (total_holidays : ℕ) :
  holidays_per_month = 3 → months_per_year = 12 → total_holidays = holidays_per_month * months_per_year →
  total_holidays = 36 :=
by
  intros
  sorry

end jackson_holidays_l362_362374


namespace compound_interest_is_correct_l362_362816

noncomputable def compound_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * (1 + R / 100)^T - P

noncomputable def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * R * T / 100

theorem compound_interest_is_correct :
  let P := 660 / (0.2 : ℝ)
  (compound_interest P 10 2) = 693 := 
by
  -- Definitions of simple_interest and compound_interest are used
  -- The problem conditions help us conclude
  let P := 660 / (0.2 : ℝ)
  have h1 : simple_interest P 10 2 = 660 := by sorry
  have h2 : compound_interest P 10 2 = 693 := by sorry
  exact h2

end compound_interest_is_correct_l362_362816


namespace circle_line_intersect_m_value_circle_AB_diameter_pass_origin_l362_362634

-- Define the circle and lines
def circle_eq (x y m : ℝ) : Prop := x^2 + y^2 - 2 * x - 4 * y + m = 0
def line1_eq (x y : ℝ) : Prop := 3 * x + 4 * y - 6 = 0
def line2_eq (x y : ℝ) : Prop := x - y = 1

-- Distance formula for point to a line
def distance_point_line (px py a b c : ℝ) : ℝ := abs (a * px + b * py + c) / sqrt (a^2 + b^2)

-- Problem Statements
theorem circle_line_intersect_m_value (m : ℝ) :
  (∀ x y : ℝ, circle_eq x y m ∧ line1_eq x y → |2 * sqrt (3)| = 2 * sqrt (3)) →
  (distance_point_line 1 2 3 4 (-6) = 1) →
  m = 1 :=
sorry

theorem circle_AB_diameter_pass_origin (m : ℝ) :
  (circle_eq x y m → line2_eq A B → (∃ m : ℝ, (x_1)* (x_2) + (y_1) * (y_2) = 0)) →
  (circle_eq x y m ∧ disc > 0 ∧ m < 3) →
  m = -2 :=
sorry

end circle_line_intersect_m_value_circle_AB_diameter_pass_origin_l362_362634


namespace sqrt_mul_sqrt_l362_362110

theorem sqrt_mul_sqrt (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : sqrt (a * sqrt b) = sqrt a * sqrt (sqrt b) := by
  sorry

example : sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  have h1 : sqrt 25 = 5 := by 
    norm_num
  have h2 : sqrt (49 * 5) = sqrt 245 := by  
    unfold sqrt
  have h3 : sqrt 245 = 7 * sqrt 5 := by 
    norm_num
  rw [h1, h2, h3]
  norm_num

end sqrt_mul_sqrt_l362_362110


namespace reduced_less_than_scaled_l362_362180

-- Define the conditions
def original_flow_rate : ℝ := 5.0
def reduced_flow_rate : ℝ := 2.0
def scaled_flow_rate : ℝ := 0.6 * original_flow_rate

-- State the theorem we need to prove
theorem reduced_less_than_scaled : scaled_flow_rate - reduced_flow_rate = 1.0 := 
by
  -- insert the detailed proof steps here
  sorry

end reduced_less_than_scaled_l362_362180


namespace q_properties_l362_362241

noncomputable def q (x : ℝ) : ℝ := 4 * x^2 - 8 * x - 12

theorem q_properties
  (q_val_3 : q 3 = 0)
  (q_val_neg_1 : q (-1) = 0)
  (q_val_neg_2 : q (-2) = 20) :
  q = (λ x, 4 * x^2 - 8 * x - 12) :=
by
  sorry

end q_properties_l362_362241


namespace sqrt_nested_l362_362058

theorem sqrt_nested : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_nested_l362_362058


namespace exists_point_D_l362_362622

variable {A B C D : Type}
variables {a b c : ℝ}
variables [Triangle ABC]
variable (AB AC BC : ℝ)

-- Assume the triangle and the condition 'AB < BC'
def TriangleABC (A B C : Type) (a b c : ℝ) [Triangle ABC] (AB AC BC : ℝ) : Prop :=
  Triangle ABC ∧ AB < BC ∧ ∃ (D : Type), D ∈ AC ∧ AB + BD + DA = BC

-- The formal Lean statement
theorem exists_point_D (h : TriangleABC A B C a b c AB AC BC) : 
  ∃ (D : A), D ∈ AC ∧ AB + BD + DA = BC :=
  sorry

end exists_point_D_l362_362622


namespace paint_left_for_third_day_l362_362955

theorem paint_left_for_third_day :
  (original_paint first_day_fraction second_day_fraction : ℝ) 
  (h1 : original_paint = 1)
  (h2 : first_day_fraction = 1/4)
  (h3 : second_day_fraction = 1/2) :
  let remaining_after_first_day := original_paint - first_day_fraction * original_paint in
  let remaining_after_second_day := remaining_after_first_day - second_day_fraction * remaining_after_first_day in
  remaining_after_second_day = 3 / 8 := 
by {
  sorry
}

end paint_left_for_third_day_l362_362955


namespace loss_percentage_l362_362870

theorem loss_percentage (C S : ℝ) (h : 5 * C = 20 * S) : (C - S) / C * 100 = 75 := by
  have h1 : C = 4 * S := by
    linarith
  rw [h1]
  have h2 : (4 * S - S) / (4 * S) * 100 = (3 * S) / (4 * S) * 100 := by
    linarith
  rw [h2]
  have h3 : (3 / 4) * 100 = 75 := by
    norm_num
  rw [h3]
  exact rfl

end loss_percentage_l362_362870


namespace lim_is_zero_l362_362206

theorem lim_is_zero :
  tendsto (λ x : ℝ, (1 - sqrt (cos x)) / (1 - cos (sqrt x))) (𝓝 0) (𝓝 0) :=
begin
  sorry
end

end lim_is_zero_l362_362206


namespace rolls_sold_to_uncle_l362_362253

theorem rolls_sold_to_uncle (total_rolls needed_rolls rolls_to_grandmother rolls_to_neighbor rolls_to_uncle : ℕ)
  (h1 : total_rolls = 45)
  (h2 : needed_rolls = 28)
  (h3 : rolls_to_grandmother = 1)
  (h4 : rolls_to_neighbor = 6)
  (h5 : rolls_to_uncle + rolls_to_grandmother + rolls_to_neighbor + needed_rolls = total_rolls) :
  rolls_to_uncle = 10 :=
by {
  sorry
}

end rolls_sold_to_uncle_l362_362253


namespace max_composite_numbers_l362_362792

-- Definitions and conditions
def is_composite (n : ℕ) : Prop := 2 < n ∧ ∃ d, d ∣ n ∧ 1 < d ∧ d < n

def less_than_1500 (n : ℕ) : Prop := n < 1500

def gcd_is_one (a b : ℕ) : Prop := Int.gcd a b = 1

-- The problem statement
theorem max_composite_numbers (numbers : List ℕ) (h_composite : ∀ n ∈ numbers, is_composite n) 
  (h_less_than_1500 : ∀ n ∈ numbers, less_than_1500 n)
  (h_pairwise_gcd_is_one : (numbers.Pairwise gcd_is_one)) : numbers.length ≤ 12 := 
  sorry

end max_composite_numbers_l362_362792


namespace alcohol_percentage_proof_l362_362518

noncomputable def percentage_alcohol_new_mixture 
  (original_solution_volume : ℕ)
  (percent_A : ℚ)
  (concentration_A : ℚ)
  (percent_B : ℚ)
  (concentration_B : ℚ)
  (percent_C : ℚ)
  (concentration_C : ℚ)
  (water_added_volume : ℕ) : ℚ :=
((original_solution_volume * percent_A * concentration_A) +
 (original_solution_volume * percent_B * concentration_B) +
 (original_solution_volume * percent_C * concentration_C)) /
 (original_solution_volume + water_added_volume) * 100

theorem alcohol_percentage_proof : 
  percentage_alcohol_new_mixture 24 0.30 0.80 0.40 0.90 0.30 0.95 16 = 53.1 := 
by 
  sorry

end alcohol_percentage_proof_l362_362518


namespace simplified_expression_value_l362_362843

theorem simplified_expression_value (a b : ℝ) (ha : a = -1) (hb : b = 1 / 4) :
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by
  sorry

end simplified_expression_value_l362_362843


namespace max_oleg_composite_numbers_l362_362755

/-- Oleg's list of composite numbers less than 1500 and pairwise coprime. --/
def oleg_composite_numbers (numbers : List ℕ) : Prop :=
  ∀ n ∈ numbers, Nat.isComposite n ∧ n < 1500 ∧ (∀ m ∈ numbers, n ≠ m → Nat.gcd n m = 1)

/-- Maximum number of such numbers that Oleg could write. --/
theorem max_oleg_composite_numbers : ∃ numbers : List ℕ, oleg_composite_numbers numbers ∧ numbers.length = 12 := 
sorry

end max_oleg_composite_numbers_l362_362755


namespace distance_between_jay_and_paul_l362_362181

-- Definitions of Jay's and Paul's walking rates and the total time
def jays_rate : ℝ := 0.75 / 15    -- rate in miles per minute
def pauls_rate : ℝ := 2.5 / 30    -- rate in miles per minute
def total_time : ℝ := 2 * 60      -- total time in minutes

-- Main theorem to be proven
theorem distance_between_jay_and_paul : 
  (jays_rate * total_time) + (pauls_rate * total_time) = 16 := 
by 
  sorry

end distance_between_jay_and_paul_l362_362181


namespace max_composite_numbers_l362_362806
open Nat

def is_composite (n : ℕ) : Prop := 1 < n ∧ ∃ d : ℕ, 1 < d ∧ d < n ∧ d ∣ n

def has_gcd_of_one (l : List ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ l → b ∈ l → a ≠ b → gcd a b = 1

def valid_composite_numbers (n : ℕ) : Prop :=
  ∀ m ∈ (List.range n).filter is_composite, m < 1500 →

-- Main theorem
theorem max_composite_numbers :
  ∃ l : List ℕ, l.length = 12 ∧ valid_composite_numbers l ∧ has_gcd_of_one l :=
sorry

end max_composite_numbers_l362_362806


namespace vija_always_wins_l362_362389

-- Define the game conditions and the players
structure Point where
  x : Int
  y : Int

def is_convex (points : List Point) : Prop := Sorry -- Define convex polygon condition (abstract for simplicity)

inductive Player
| Konya : Player
| Vija : Player

-- Main theorem stating Vija always wins under given conditions
theorem vija_always_wins (moves : List Point) (conditions : ∀ n, is_convex (moves.take (n+2).tail)) : 
  ∃ (final_move : Player), final_move = Player.Vija := 
sorry


end vija_always_wins_l362_362389


namespace compose_f_g_f_l362_362398

def f (x : ℝ) : ℝ := 2 * x + 5
def g (x : ℝ) : ℝ := 3 * x + 4

theorem compose_f_g_f (x : ℝ) : f (g (f 3)) = 79 := by
  sorry

end compose_f_g_f_l362_362398


namespace max_composite_numbers_l362_362783

theorem max_composite_numbers (S : Finset ℕ) (h1 : ∀ n ∈ S, n < 1500) (h2 : ∀ m n ∈ S, m ≠ n → Nat.gcd m n = 1) : S.card ≤ 12 := sorry

end max_composite_numbers_l362_362783


namespace cos_2θ_equals_zero_l362_362648

-- Define the problem statement in Lean
noncomputable def vectors_perpendicular (θ : ℝ) : Prop :=
  let a := (1, Real.cos θ)
  let b := (-1, 2 * Real.cos θ)
  a.1 * b.1 + a.2 * b.2 = 0

theorem cos_2θ_equals_zero (θ : ℝ) (h : vectors_perpendicular θ) : Real.cos (2 * θ) = 0 := by
  sorry

end cos_2θ_equals_zero_l362_362648


namespace sum_odd_divisors_300_l362_362584

theorem sum_odd_divisors_300 : 
  ∑ d in (Nat.divisors 300).filter Nat.Odd, d = 124 := 
sorry

end sum_odd_divisors_300_l362_362584


namespace power_of_power_evaluate_3_power_3_power_2_l362_362235

theorem power_of_power (a m n : ℕ) : (a^m)^n = a^(m * n) :=
sorry

theorem evaluate_3_power_3_power_2 : (3^3)^2 = 729 :=
by
  have h1 : (3^3)^2 = 3^(3 * 2) := power_of_power 3 3 2
  have h2 : 3^(3 * 2) = 3^6 := rfl
  have h3 : 3^6 = 729 := sorry -- Placeholder for the actual multiplication calculation
  exact eq.trans (eq.trans h1 h2) h3

end power_of_power_evaluate_3_power_3_power_2_l362_362235


namespace gcd_f100_f101_l362_362409

def f (x : ℤ) : ℤ := x^2 - 3 * x + 2023

theorem gcd_f100_f101 : Int.gcd (f 100) (f 101) = 2 :=
by
  sorry

end gcd_f100_f101_l362_362409


namespace max_composite_numbers_l362_362798

-- Definitions and conditions
def is_composite (n : ℕ) : Prop := 2 < n ∧ ∃ d, d ∣ n ∧ 1 < d ∧ d < n

def less_than_1500 (n : ℕ) : Prop := n < 1500

def gcd_is_one (a b : ℕ) : Prop := Int.gcd a b = 1

-- The problem statement
theorem max_composite_numbers (numbers : List ℕ) (h_composite : ∀ n ∈ numbers, is_composite n) 
  (h_less_than_1500 : ∀ n ∈ numbers, less_than_1500 n)
  (h_pairwise_gcd_is_one : (numbers.Pairwise gcd_is_one)) : numbers.length ≤ 12 := 
  sorry

end max_composite_numbers_l362_362798


namespace total_loss_is_1600_l362_362556

noncomputable def total_loss (P : ℝ) : ℝ :=
  let A := (1 / 9) * P in
  let Loss_Pyarelal := 1440 in
  let Loss_Ashok := Loss_Pyarelal * (A / P) in
  Loss_Ashok + Loss_Pyarelal

theorem total_loss_is_1600 (P : ℝ) (h : P ≠ 0) : total_loss P = 1600 :=
by
  sorry

end total_loss_is_1600_l362_362556


namespace find_x_plus_y_l362_362659

variable (x y : ℝ)

theorem find_x_plus_y (h1 : |x| + x + y = 8) (h2 : x + |y| - y = 10) : x + y = 14 / 5 := 
by
  sorry

end find_x_plus_y_l362_362659


namespace count_4_letter_words_with_A_l362_362301

-- Define the alphabet set and the properties
def Alphabet : Finset (Char) := {'A', 'B', 'C', 'D', 'E'}
def total_words := (Alphabet.card ^ 4 : ℕ)
def total_words_without_A := (Alphabet.erase 'A').card ^ 4
def total_words_with_A := total_words - total_words_without_A

-- The key theorem to prove
theorem count_4_letter_words_with_A : total_words_with_A = 369 := sorry

end count_4_letter_words_with_A_l362_362301


namespace sqrt_49_mul_sqrt_25_l362_362001

theorem sqrt_49_mul_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_l362_362001


namespace sqrt_49_times_sqrt_25_eq_7sqrt5_l362_362070

theorem sqrt_49_times_sqrt_25_eq_7sqrt5 :
  (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  have h1 : Real.sqrt 25 = 5 := by sorry
  have h2 : 49 * 5 = 245 := by sorry
  have h3 : Real.sqrt 245 = 7 * Real.sqrt 5 := by sorry
  sorry

end sqrt_49_times_sqrt_25_eq_7sqrt5_l362_362070


namespace minimum_employees_needed_l362_362954

theorem minimum_employees_needed (S H : Set ℕ) (hS : S.card = 120) (hH : H.card = 90) (hSH : (S ∩ H).card = 40) : 
  (S ∪ H).card = 170 := by
  sorry

end minimum_employees_needed_l362_362954


namespace triangle_sides_condition_triangle_perimeter_l362_362723

theorem triangle_sides_condition (a b c : ℝ) (A B C : ℝ) 
  (h1 : sin C * sin (A - B) = sin B * sin (C - A)) : 2 * a^2 = b^2 + c^2 :=
sorry

theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ) 
  (h2 : a = 5) (h3 : cos A = 25 / 31) : a + b + c = 14 :=
sorry

end triangle_sides_condition_triangle_perimeter_l362_362723


namespace most_negative_integer_l362_362494

theorem most_negative_integer {l : List ℤ} (h : ∀ n ∈ l, 0 < n) :
  (∃ k, (∀ l', l' ~ l → perm_closure l' ⊆ (Set.range List.perm) ∧ perm_closure l' = List.updateNth l' k 0) → 
    (∃ m, l.mem m ∧ m >= -3)) :=
sorry

end most_negative_integer_l362_362494


namespace simplify_and_evaluate_l362_362850

theorem simplify_and_evaluate 
  (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) = 1 :=
by
  sorry

end simplify_and_evaluate_l362_362850


namespace sqrt_nested_l362_362064

theorem sqrt_nested : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_nested_l362_362064


namespace sqrt_49_times_sqrt_25_l362_362019

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 :=
by
  sorry

end sqrt_49_times_sqrt_25_l362_362019


namespace exactly_one_pair_probability_l362_362228

def four_dice_probability : ℚ :=
  sorry  -- Here we skip the actual computation and proof

theorem exactly_one_pair_probability : four_dice_probability = 5/9 := by {
  -- Placeholder for proof, explanation, and calculation
  sorry
}

end exactly_one_pair_probability_l362_362228


namespace hex_B2F_to_base10_l362_362219

theorem hex_B2F_to_base10 :
  let b := 11
  let two := 2
  let f := 15
  let base := 16
  (b * base^2 + two * base^1 + f * base^0) = 2863 :=
by
  sorry

end hex_B2F_to_base10_l362_362219


namespace carla_water_drank_l362_362213

theorem carla_water_drank (W S : ℝ) (h1 : W + S = 54) (h2 : S = 3 * W - 6) : W = 15 :=
by
  sorry

end carla_water_drank_l362_362213


namespace solve_for_a_l362_362630

theorem solve_for_a (a : ℝ) (hi: ∃ b : ℝ, b ≠ 0 ∧ (a - complex.I) / (2 + complex.I) = b * complex.I) : a = 1 / 2 :=
sorry

end solve_for_a_l362_362630


namespace smallest_solution_neg_two_l362_362250

-- We set up the expressions and then state the smallest solution
def smallest_solution (x : ℝ) : Prop :=
  x * abs x = 3 * x + 2

theorem smallest_solution_neg_two :
  ∃ x : ℝ, smallest_solution x ∧ (∀ y : ℝ, smallest_solution y → y ≥ x) ∧ x = -2 :=
by
  sorry

end smallest_solution_neg_two_l362_362250


namespace inequality_abc_l362_362274

variable (a b c : ℝ)

theorem inequality_abc (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) :
  a / (a^3 - a^2 + 3) + b / (b^3 - b^2 + 3) + c / (c^3 - c^2 + 3) ≤ 1 := 
sorry

end inequality_abc_l362_362274


namespace incircle_radius_of_right_triangle_l362_362486

/-- Triangle ABC has a right angle at C, angle A = 45 degrees, and AC = 12. The radius of the incircle of triangle ABC is 12 - 6 * sqrt(2). -/
theorem incircle_radius_of_right_triangle
  (A B C : Type)
  (is_triangle : Triangle A B C)
  (right_angle_at_C : ∠ABC = 90)
  (angle_A_45_degrees : ∠BAC = 45)
  (AC_length : AC = 12) :
  incircle_radius (Triangle A B C) = 12 - 6 * real.sqrt 2 := 
sorry

end incircle_radius_of_right_triangle_l362_362486


namespace max_composite_numbers_with_gcd_one_l362_362760

theorem max_composite_numbers_with_gcd_one : 
  ∃ (S : Finset ℕ), 
    (∀ x ∈ S, Nat.isComposite x) ∧ 
    (∀ x ∈ S, x < 1500) ∧ 
    (∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y → Nat.gcd x y = 1) ∧
    S.card = 12 :=
sorry

end max_composite_numbers_with_gcd_one_l362_362760


namespace function_inequality_l362_362266

noncomputable def f : ℝ → ℝ :=
sorry

theorem function_inequality
  (even_f : ∀ x, f x = f (-x))
  (periodic_f : ∀ x, f (x + 1) = -f x)
  (decreasing_f : ∀ ⦃x y⦄, 0 ≤ x → x ≤ 1 → y ∈ Icc 0 1 → x < y → f x > f y) :
  f (7/5) < f (7/2) ∧ f (7/2) < f (7/3) :=
sorry

end function_inequality_l362_362266


namespace primes_between_40_and_50_l362_362324

theorem primes_between_40_and_50 : (finset.filter (λ n, nat.prime n) (finset.Icc 40 50)).card = 3 := sorry

end primes_between_40_and_50_l362_362324


namespace proof_equiv_l362_362267

noncomputable def g : ℝ → ℝ :=
  λ x, if x = -1 then 0
       else if x = 0 then 1
       else if x = 2 then 3
       else if x = 3 then 4
       else if x = 4 then 6
       else sorry -- For the sake of the problem, we only care about the given inputs

noncomputable def g_inv : ℝ → ℝ :=
  λ y, if y = 0 then -1
       else if y = 1 then 0
       else if y = 3 then 2
       else if y = 4 then 3
       else if y = 6 then 4
       else sorry -- Define the inverse only for the given outputs

theorem proof_equiv :
  g(g 2) + g(g_inv 3) + g_inv(g_inv 4) = 9 :=
by
  -- Insert proof steps here
  sorry

end proof_equiv_l362_362267


namespace lateral_surface_area_of_cone_l362_362442

theorem lateral_surface_area_of_cone (r h : ℝ) (hr : r = 3) (hh : h = 4) : 
  ∃ (lateral_surface_area : ℝ), lateral_surface_area = 15 * Real.pi :=
by
  use 15 * Real.pi
  sorry

end lateral_surface_area_of_cone_l362_362442


namespace max_composite_numbers_l362_362784

theorem max_composite_numbers (S : Finset ℕ) (h1 : ∀ n ∈ S, n < 1500) (h2 : ∀ m n ∈ S, m ≠ n → Nat.gcd m n = 1) : S.card ≤ 12 := sorry

end max_composite_numbers_l362_362784


namespace pumpkins_at_other_orchard_l362_362558

-- Defining the initial conditions
def sunshine_pumpkins : ℕ := 54
def other_orchard_pumpkins : ℕ := 14

-- Equation provided in the problem
def condition_equation (P : ℕ) : Prop := 54 = 3 * P + 12

-- Proving the main statement using the conditions
theorem pumpkins_at_other_orchard : condition_equation other_orchard_pumpkins :=
by
  unfold condition_equation
  sorry -- To be completed with the proof

end pumpkins_at_other_orchard_l362_362558


namespace max_composite_numbers_l362_362801
open Nat

def is_composite (n : ℕ) : Prop := 1 < n ∧ ∃ d : ℕ, 1 < d ∧ d < n ∧ d ∣ n

def has_gcd_of_one (l : List ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ l → b ∈ l → a ≠ b → gcd a b = 1

def valid_composite_numbers (n : ℕ) : Prop :=
  ∀ m ∈ (List.range n).filter is_composite, m < 1500 →

-- Main theorem
theorem max_composite_numbers :
  ∃ l : List ℕ, l.length = 12 ∧ valid_composite_numbers l ∧ has_gcd_of_one l :=
sorry

end max_composite_numbers_l362_362801


namespace simplify_and_evaluate_l362_362854

theorem simplify_and_evaluate (a b : ℝ) (h₁ : a = -1) (h₂ : b = 1/4) : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_and_evaluate_l362_362854


namespace smallest_prime_factor_of_2023_l362_362915

theorem smallest_prime_factor_of_2023 : Nat.prime 7 ∧ 7 ∣ 2023 ∧ ∀ p, Nat.prime p ∧ p ∣ 2023 → p ≥ 7 :=
by 
  sorry

end smallest_prime_factor_of_2023_l362_362915


namespace determine_shape_of_triangle_l362_362282

theorem determine_shape_of_triangle (a b c : ℕ) (h1 : (a - 3)^2 = 0) (h2 : sqrt (b - 4) = 0) (h3 : abs (c - 5) = 0) : a^2 + b^2 = c^2 → (3^2 + 4^2 = 5^2) :=
by 
  sorry

end determine_shape_of_triangle_l362_362282


namespace readers_both_l362_362352

-- Definitions
def total_readers : ℕ := 250
def sci_fi_readers : ℕ := 180
def lit_readers : ℕ := 88

-- Theorem statement
theorem readers_both (S L : set ℕ) (h1 : fintype.card S = sci_fi_readers)
    (h2 : fintype.card L = lit_readers) (h3 : fintype.card (S ∪ L) = total_readers) :
    fintype.card (S ∩ L) = 18 :=
by sorry

end readers_both_l362_362352


namespace average_first_16_even_numbers_l362_362909

def even (n : ℕ) : ℕ := 2 * n

theorem average_first_16_even_numbers : 
  (List.range 16).map even |>.sum / 16 = 17 := by
    sorry

end average_first_16_even_numbers_l362_362909


namespace simplify_fraction_l362_362656

variables {x y : ℝ}

theorem simplify_fraction (h : x / y = 2 / 5) : (3 * y - 2 * x) / (3 * y + 2 * x) = 11 / 19 :=
by
  sorry

end simplify_fraction_l362_362656


namespace log_ratio_squared_l362_362822

variables {x y : ℝ}
#check Math.log -- Ensure logarithm function is correctly identified
#check real.log -- Check the logarithm function for real numbers

-- Definitions/conditions as per the problem
def cond1 : Prop := x ≠ 1
def cond2 : Prop := y ≠ 1
def cond3 : Prop := real.log x / real.log 2 = real.log 8 / real.log y
def cond4 : Prop := x * y = 128

-- Tying all conditions together
def conditions : Prop := cond1 ∧ cond2 ∧ cond3 ∧ cond4

-- Statement to prove the equivalence
theorem log_ratio_squared : conditions → (real.log (x / y) / real.log 2)^2 = 20 := by 
  intros _,
  sorry -- proof goes here

end log_ratio_squared_l362_362822


namespace sequence_property_ans_2017th_term_l362_362341

noncomputable def a_sequence : ℕ → ℝ
| 1       := 5
| (n + 1) := (2 * n + 5) * a_sequence n / (2 * n + 3) + (2 * n + 5) * (2 * n + 3) * real.log (1 + 1 / n)

def property (n : ℕ) :=
  (2 * n + 3) * a_sequence (n + 1) - (2 * n + 5) * a_sequence n =
  (2 * n + 3) * (2 * n + 5) * real.log (1 + 1 / n)

theorem sequence_property :
  ∀ (n : ℕ), property n :=
sorry

theorem ans_2017th_term:
  a_sequence 2017 / (2 * 2017 + 3) = 1 + real.log 2017 :=
sorry

end sequence_property_ans_2017th_term_l362_362341


namespace pyramid_volume_l362_362537

noncomputable def volume_pyramid (S A_triangle A_square s h PF V : ℝ) : Prop :=
  -- All given conditions
  (S = 500) ∧
  (A_triangle = A_square) ∧
  (S = A_square + 4 * A_triangle) ∧
  (A_square = s^2) ∧
  (1/2 * s * h = A_triangle) ∧
  (PF = Math.sqrt(h^2 - (s/2)^2)) ∧
  -- Goal condition
  (V = 1/3 * A_square * PF) ∧
  -- Desired result
  (V = 500 * Math.sqrt(15) / 3)

theorem pyramid_volume : 
  exists (S A_triangle A_square s h PF V : ℝ), volume_pyramid S A_triangle A_square s h PF V := sorry

end pyramid_volume_l362_362537


namespace prince_wish_fulfilled_l362_362897

theorem prince_wish_fulfilled
  (k : ℕ)
  (k_gt_1 : 1 < k)
  (k_lt_13 : k < 13)
  (city : Fin 13 → Fin k) 
  (initial_goblets : Fin k → Fin 13)
  (is_gold : Fin 13 → Bool) :
  ∃ i j : Fin 13, i ≠ j ∧ city i = city j ∧ is_gold i = true ∧ is_gold j = true := 
sorry

end prince_wish_fulfilled_l362_362897


namespace impossible_to_get_60_pieces_possible_to_get_more_than_60_pieces_l362_362511

theorem impossible_to_get_60_pieces :
  ¬ ∃ (n m : ℕ), 1 + 7 * n + 11 * m = 60 :=
sorry

theorem possible_to_get_more_than_60_pieces :
  ∀ k > 60, ∃ (n m : ℕ), 1 + 7 * n + 11 * m = k :=
sorry

end impossible_to_get_60_pieces_possible_to_get_more_than_60_pieces_l362_362511


namespace solve_system_of_inequalities_l362_362433

theorem solve_system_of_inequalities (x y : ℤ) :
  (2 * x - y > 3 ∧ 3 - 2 * x + y > 0) ↔ (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) := 
by { sorry }

end solve_system_of_inequalities_l362_362433


namespace distinct_3_letter_words_l362_362978

theorem distinct_3_letter_words : 
  ∃ (S : finset (list char)), S.card = 33 ∧ ∀ w ∈ S, 
    w.length = 3 ∧ 
    (∀ l ∈ w.to_finset, l = 'c' ∨ l = 'o' ∨ l = 'm' ∨ l = 'b') ∧ 
    (w.count 'c' ≤ 1) ∧ (w.count 'o' ≤ 2) ∧ (w.count 'm' ≤ 1) ∧ (w.count 'b' ≤ 1) := 
sorry

end distinct_3_letter_words_l362_362978


namespace sqrt_49_times_sqrt25_eq_7_sqrt5_l362_362035

-- Definitions based on conditions
def sqrt_25 := Real.sqrt 25
def sqrt_49 := Real.sqrt 49

theorem sqrt_49_times_sqrt25_eq_7_sqrt5 :
  Real.sqrt (49 * sqrt_25) = 7 * Real.sqrt 5 := by
sorry

end sqrt_49_times_sqrt25_eq_7_sqrt5_l362_362035


namespace find_complex_z_l362_362246

open Complex

noncomputable def is_real (z : ℂ) : Prop := ∃ (r : ℝ), z = r

theorem find_complex_z (z : ℂ) (h1 : ∥conj z - 3∥ = ∥conj z - 3 * I∥)
  (h2 : is_real (z - 1 + 5 / (z - 1))) : z = 2 - 2 * I ∨ z = -1 + I :=
by
  sorry

end find_complex_z_l362_362246


namespace ruby_height_l362_362662

variable (Ruby Pablo Charlene Janet : ℕ)

theorem ruby_height :
  (Ruby = Pablo - 2) →
  (Pablo = Charlene + 70) →
  (Janet = 62) →
  (Charlene = 2 * Janet) →
  Ruby = 192 := 
by
  sorry

end ruby_height_l362_362662


namespace sqrt_mul_sqrt_l362_362044

theorem sqrt_mul_sqrt (a b c : ℝ) (hac : a = 49) (hbc : b = sqrt 25) (hc : c = 7 * sqrt 5) : sqrt (a * b) = c := 
by
  sorry

end sqrt_mul_sqrt_l362_362044


namespace volume_of_cone_is_correct_l362_362523

-- Define base radius of the cone
def base_radius : ℝ := 5

-- Define the height of the cone as three times the base radius
def height : ℝ := 3 * base_radius

-- Define the volume formula for a cone
def volume (r h : ℝ) : ℝ := (1 / 3) * Real.pi * (r ^ 2) * h

-- State the problem: Prove that the volume of the cone with the given dimensions is 392.5 cubic centimeters
theorem volume_of_cone_is_correct : 
  volume base_radius height = 392.5 := 
sorry

end volume_of_cone_is_correct_l362_362523


namespace roberto_raise_percentage_l362_362429

theorem roberto_raise_percentage
    (starting_salary : ℝ)
    (previous_salary : ℝ)
    (current_salary : ℝ)
    (h1 : starting_salary = 80000)
    (h2 : previous_salary = starting_salary * 1.40)
    (h3 : current_salary = 134400) :
    ((current_salary - previous_salary) / previous_salary) * 100 = 20 :=
by sorry

end roberto_raise_percentage_l362_362429


namespace walter_age_2009_l362_362200

noncomputable theory

-- Definitions from the given conditions
def year : ℤ := 2004
def year_sum : ℤ := 4018
def walter_age2004 : ℤ := 4018 - 2 * year

-- The final proof statement:
theorem walter_age_2009 : walter_age2004 + 5 = 7.5 :=
by sorry

end walter_age_2009_l362_362200


namespace dog_food_vs_cat_food_l362_362532

-- Define the quantities of dog food and cat food
def dog_food : ℕ := 600
def cat_food : ℕ := 327

-- Define the problem as a statement asserting the required difference
theorem dog_food_vs_cat_food : dog_food - cat_food = 273 := by
  sorry

end dog_food_vs_cat_food_l362_362532


namespace infinite_set_A_exists_l362_362701

theorem infinite_set_A_exists (k : ℕ) (hk : k > 1) :
  ∃ (A : Set (Set ℕ)), 
    (∀ (s t : Set ℕ), s ≠ t → s ∈ A → t ∈ A → ∃! n : ℕ, n ∈ s ∧ n ∈ t) ∧
    (∀ (B : Set (Set ℕ)), B ⊆ A → B.card = k + 1 → ⋂₀ B = ∅) :=
sorry

end infinite_set_A_exists_l362_362701


namespace triple_apply_l362_362332

def f (x : ℝ) : ℝ := 5 * x - 4

theorem triple_apply : f (f (f 2)) = 126 :=
by
  rw [f, f, f]
  sorry

end triple_apply_l362_362332


namespace number_of_true_propositions_l362_362286

theorem number_of_true_propositions : 
  let p1 := ¬(∀ x y : ℝ, x * y = 0 → x = 0 ∧ y = 0)
  let p2 := ¬(∀ x: Type, (x = square → x = rhombus))
  let p3 := ∀ a b c : ℝ, a > b → a * c^2 > b * c^2
  let p4 := ∀ m : ℝ, m > 2 → ∀ x : ℝ, x^2 - 2 * x + m > 0
  (if p1 then 1 else 0) + (if p2 then 1 else 0) + (if p3 then 1 else 0) + (if p4 then 1 else 0) = 1 := 
  by
    intros
    sorry

end number_of_true_propositions_l362_362286


namespace distance_traveled_by_car_l362_362174

theorem distance_traveled_by_car (total_distance : ℕ) (fraction_foot : ℚ) (fraction_bus : ℚ)
  (h_total : total_distance = 40) (h_fraction_foot : fraction_foot = 1/4)
  (h_fraction_bus : fraction_bus = 1/2) :
  (total_distance * (1 - fraction_foot - fraction_bus)) = 10 :=
by
  sorry

end distance_traveled_by_car_l362_362174


namespace sqrt_mul_sqrt_l362_362053

theorem sqrt_mul_sqrt (a b c : ℝ) (hac : a = 49) (hbc : b = sqrt 25) (hc : c = 7 * sqrt 5) : sqrt (a * b) = c := 
by
  sorry

end sqrt_mul_sqrt_l362_362053


namespace petya_fraction_of_travel_l362_362127

noncomputable def fraction_traveled_before_recalling_pen
  (total_time_to_school : ℕ) -- The road from Petya's house to the school takes 20 minutes.
  (time_before_bell : ℕ) -- If Petya continues his journey at the same speed, he will arrive at school 3 minutes before the bell rings.
  (time_late_when_returning : ℕ) -- If he returns home to get the pen and then goes to school at the same speed, he will be 7 minutes late for the start of the class.
  (time_of_pen_recall : ℕ) -- The total time of travel becomes 27 minutes when going back to fetch the pen.
  : ℚ :=
if (total_time_to_school = 20 ∧ time_before_bell = 3 ∧ time_late_when_returning = 7 ∧ time_of_pen_recall = 7) then
  1 / 4
else
  0 -- This else-case is arbitrary as the conditions are assumed to hold true.

theorem petya_fraction_of_travel (fraction_traveled_before_recalling_pen : ℚ) (total_time_to_school : ℕ) (time_before_bell : ℕ) (time_late_when_returning : ℕ) (time_of_pen_recall : ℕ) :
  (total_time_to_school = 20 ∧ time_before_bell = 3 ∧ time_late_when_returning = 7 ∧ time_of_pen_recall = 7) → fraction_traveled_before_recalling_pen total_time_to_school time_before_bell time_late_when_returning time_of_pen_recall = 1 / 4 :=
by {
  sorry
}

end petya_fraction_of_travel_l362_362127


namespace pyramid_volume_l362_362427

theorem pyramid_volume (AB BC PB PA : ℝ)
  (hAB : AB = 10)
  (hBC : BC = 6)
  (hPA : PA = sqrt (PB^2 - AB^2))
  (hPB : PB = 26)
  : (1 / 3) * (AB * BC) * PA = 480 :=
by
  sorry

end pyramid_volume_l362_362427


namespace sqrt_expression_simplified_l362_362010

theorem sqrt_expression_simplified :
  sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  sorry

end sqrt_expression_simplified_l362_362010


namespace time_spent_per_egg_in_seconds_l362_362239

-- Definitions based on the conditions in the problem
def minutes_per_roll : ℕ := 30
def number_of_rolls : ℕ := 7
def total_cleaning_time : ℕ := 225
def number_of_eggs : ℕ := 60

-- Problem statement
theorem time_spent_per_egg_in_seconds :
  (total_cleaning_time - number_of_rolls * minutes_per_roll) * 60 / number_of_eggs = 15 := by
  sorry

end time_spent_per_egg_in_seconds_l362_362239


namespace triangle_sides_relation_triangle_perimeter_l362_362711

theorem triangle_sides_relation
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : sin C * sin (A - B) = sin B * sin (C - A)) :
  2 * a^2 = b^2 + c^2 :=
sorry

theorem triangle_perimeter 
  (a b c : ℝ)
  (A B C : ℝ)
  (h_a : a = 5)
  (h_cosA : cos A = 25 / 31)
  (h_sin_relation : sin C * sin (A - B) = sin B * sin (C - A)) :
  a + b + c = 14 :=
sorry

end triangle_sides_relation_triangle_perimeter_l362_362711


namespace sqrt_49_times_sqrt25_eq_7_sqrt5_l362_362041

-- Definitions based on conditions
def sqrt_25 := Real.sqrt 25
def sqrt_49 := Real.sqrt 49

theorem sqrt_49_times_sqrt25_eq_7_sqrt5 :
  Real.sqrt (49 * sqrt_25) = 7 * Real.sqrt 5 := by
sorry

end sqrt_49_times_sqrt25_eq_7_sqrt5_l362_362041


namespace measure_of_each_interior_angle_of_regular_octagon_l362_362498

theorem measure_of_each_interior_angle_of_regular_octagon 
  (n : ℕ) (h_n : n = 8) : 
  let sum_of_interior_angles := 180 * (n - 2) in
  let measure_of_interior_angle := sum_of_interior_angles / n in
  measure_of_interior_angle = 135 :=
by
  sorry

end measure_of_each_interior_angle_of_regular_octagon_l362_362498


namespace S_2017_l362_362284

def a_sequence (n : ℕ) : ℝ :=
  let a1 := Real.tan (225 * Real.pi / 180) in
  let d := (13 * a1 - a1) / 4 in
  a1 + (n - 1) * d

def S (n : ℕ) : ℝ :=
  (List.range n).sum (λ i => (-1)^i * a_sequence (i + 1))

theorem S_2017 : S 2017 = -3021 := by
  sorry

end S_2017_l362_362284


namespace hyperbola_focus_and_asymptotes_l362_362184

def is_focus_on_y_axis (a b : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
∃ c : ℝ, eq (c^2 * a) (c^2 * b)

def are_asymptotes_perpendicular (eq : ℝ → ℝ → Prop) : Prop :=
∃ k1 k2 : ℝ, (k1 != 0 ∧ k2 != 0 ∧ eq k1 k2 ∧ eq (-k1) k2)

theorem hyperbola_focus_and_asymptotes :
  is_focus_on_y_axis 1 (-1) (fun y x => y^2 - x^2 = 4) ∧ are_asymptotes_perpendicular (fun y x => y = x) :=
by
  sorry

end hyperbola_focus_and_asymptotes_l362_362184


namespace triangle_identity_triangle_perimeter_l362_362720

theorem triangle_identity 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : sin C * sin (A - B) = sin B * sin (C - A)) :
  2 * a^2 = b^2 + c^2 :=
sorry

theorem triangle_perimeter 
  (a b c : ℝ) 
  (A : ℝ) 
  (h1 : 2 * a^2 = b^2 + c^2) 
  (ha : a = 5) 
  (h_cosA : cos A = 25 / 31) :
  a + b + c = 14 :=
sorry

end triangle_identity_triangle_perimeter_l362_362720


namespace max_composite_numbers_l362_362810
open Nat

theorem max_composite_numbers : 
  ∃ X : Finset Nat, 
  (∀ x ∈ X, x < 1500 ∧ ¬Prime x) ∧ 
  (∀ x y ∈ X, x ≠ y → gcd x y = 1) ∧ 
  X.card = 12 := 
sorry

end max_composite_numbers_l362_362810


namespace sqrt_mul_sqrt_l362_362113

theorem sqrt_mul_sqrt (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : sqrt (a * sqrt b) = sqrt a * sqrt (sqrt b) := by
  sorry

example : sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  have h1 : sqrt 25 = 5 := by 
    norm_num
  have h2 : sqrt (49 * 5) = sqrt 245 := by  
    unfold sqrt
  have h3 : sqrt 245 = 7 * sqrt 5 := by 
    norm_num
  rw [h1, h2, h3]
  norm_num

end sqrt_mul_sqrt_l362_362113


namespace find_DF_in_right_triangle_l362_362359

theorem find_DF_in_right_triangle 
  (D E F : Type) 
  (right_triangle : D ≠ E ∧ D ≠ F ∧ E ≠ F)
  (angle_D_right : ∠ D = 90) 
  (cos_F : ∀ (DF DE : ℝ), cos F = (3 * real.sqrt 58) / 58 → cos F = DF / DE)
  (DE_val : DE = real.sqrt 58) : 
  DF = 3 := 
by 
  sorry

end find_DF_in_right_triangle_l362_362359


namespace pyramid_edge_length_correct_l362_362469

-- Definitions for the conditions
def total_length (sum_of_edges : ℝ) := sum_of_edges = 14.8
def edges_count (num_of_edges : ℕ) := num_of_edges = 8

-- Definition for the question and corresponding answer to prove
def length_of_one_edge (sum_of_edges : ℝ) (num_of_edges : ℕ) (one_edge_length : ℝ) :=
  sum_of_edges / num_of_edges = one_edge_length

-- The statement that needs to be proven
theorem pyramid_edge_length_correct : total_length 14.8 → edges_count 8 → length_of_one_edge 14.8 8 1.85 :=
by
  intros h1 h2
  sorry

end pyramid_edge_length_correct_l362_362469


namespace two_a_minus_five_d_eq_zero_l362_362399

variables {α : Type*} [Field α]

def f (a b c d x : α) : α :=
  (2*a*x + 3*b) / (4*c*x - 5*d)

theorem two_a_minus_five_d_eq_zero
  (a b c d : α) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (hf : ∀ x, f a b c d (f a b c d x) = x) :
  2*a - 5*d = 0 :=
sorry

end two_a_minus_five_d_eq_zero_l362_362399


namespace prime_divisor_form_l362_362738


open Int

theorem prime_divisor_form (a b : ℤ) (h : IsCoprime a b) : 
  ∀ p : ℕ, Prime p → p ∣ (a ^ 2 + 2 * b ^ 2) → ∃ x y : ℤ, (p : ℤ) = x ^ 2 + 2 * y ^ 2 :=
sorry

end prime_divisor_form_l362_362738


namespace on_imaginary_axis_in_third_quadrant_l362_362617

noncomputable def complex_number (m : ℝ) : ℂ := 
(m^2 - 2 * m) + (m^2 + m - 6) * complex.I

theorem on_imaginary_axis (m : ℝ) : 
  (complex_number m).re = 0 ↔ m = 0 ∧ m^2 + m - 6 ≠ 0 := by
  sorry

theorem in_third_quadrant (m : ℝ) : 
  (complex_number m).re < 0 ∧ (complex_number m).im < 0 ↔ 0 < m ∧ m < 2 := by
  sorry

end on_imaginary_axis_in_third_quadrant_l362_362617


namespace sqrt_expression_simplified_l362_362015

theorem sqrt_expression_simplified :
  sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  sorry

end sqrt_expression_simplified_l362_362015


namespace find_xyz_l362_362243

theorem find_xyz
  (x y z : ℝ)
  (h1 : x + y + z = 38)
  (h2 : x * y * z = 2002)
  (h3 : 0 < x ∧ x ≤ 11)
  (h4 : z ≥ 14) :
  x = 11 ∧ y = 13 ∧ z = 14 :=
sorry

end find_xyz_l362_362243


namespace power_of_a_power_evaluate_3_pow_3_pow_2_l362_362231

theorem power_of_a_power (a m n : ℕ) : (a^m)^n = a^(m*n) := 
begin
  sorry,
end

theorem evaluate_3_pow_3_pow_2 : (3^3)^2 = 729 := 
begin
  have H1 : (3^3)^2 = 3^(3*2) := power_of_a_power 3 3 2,
  have H2 : 3^(3*2) = 3^6 := by refl,
  have H3 : 3^6 = 729 := by norm_num,
  exact eq.trans (eq.trans H1 H2) H3,
end

end power_of_a_power_evaluate_3_pow_3_pow_2_l362_362231


namespace father_l362_362136

theorem father's_age_equals_combined_ages_multiplied (x : ℕ) 
(man_age son_age daughter_age : ℕ) (h_man_age : man_age = 38) (h_son_age : son_age = 14) (h_daughter_age : daughter_age = 10)
(h_constant_diff : son_age - daughter_age = 4) :
x = 8 → man_age - x = (son_age - x + daughter_age - x) * 4 :=
by
  intros hx
  rw [h_man_age, h_son_age, h_daughter_age, hx],
  simp,
  sorry

end father_l362_362136


namespace complex_number_property_l362_362342

noncomputable def imaginary_unit : Complex := Complex.I

theorem complex_number_property (n : ℕ) (hn : 4^n = 256) : (1 + imaginary_unit)^n = -4 :=
by
  sorry

end complex_number_property_l362_362342


namespace inequality_solution_range_l362_362667

variable (a : ℝ)

def f (x : ℝ) := 2 * x^2 - 8 * x - 4

theorem inequality_solution_range :
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ f x - a > 0) ↔ a < -4 := 
by
  sorry

end inequality_solution_range_l362_362667


namespace sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l362_362085

theorem sqrt_49_mul_sqrt_25_eq_7_sqrt_5 : (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l362_362085


namespace knights_gold_goblets_l362_362900

theorem knights_gold_goblets (k : ℕ) (k_gt_1 : 1 < k) (k_lt_13 : k < 13)
  (goblets : Fin 13 → Bool) (gold_goblets : (Fin 13 → Bool) → ℕ) 
  (cities : Fin 13 → Fin k) :
  (∃ (i j : Fin 13), i ≠ j ∧ cities i = cities j ∧ goblets i ∧ goblets j) :=
begin
  sorry
end

end knights_gold_goblets_l362_362900


namespace number_of_ways_l362_362893

theorem number_of_ways (h_walk : ℕ) (h_drive : ℕ) (h_eq1 : h_walk = 3) (h_eq2 : h_drive = 4) : h_walk + h_drive = 7 :=
by 
  sorry

end number_of_ways_l362_362893


namespace sqrt_mul_sqrt_l362_362052

theorem sqrt_mul_sqrt (a b c : ℝ) (hac : a = 49) (hbc : b = sqrt 25) (hc : c = 7 * sqrt 5) : sqrt (a * b) = c := 
by
  sorry

end sqrt_mul_sqrt_l362_362052


namespace andrey_travel_distance_l362_362990

theorem andrey_travel_distance:
  ∃ s t: ℝ, 
    (s = 60 * (t + 4/3) + 20  ∧ s = 90 * (t - 1/3) + 60) ∧ s = 180 :=
by
  sorry

end andrey_travel_distance_l362_362990


namespace simplified_expression_value_l362_362845

theorem simplified_expression_value (a b : ℝ) (ha : a = -1) (hb : b = 1 / 4) :
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by
  sorry

end simplified_expression_value_l362_362845


namespace smallest_prime_factor_2023_l362_362914

def smallest_prime_factor (n : ℕ) : ℕ :=
  if h : ∃ p : ℕ, Nat.Prime p ∧ p ∣ n then
    Nat.find h
  else
    0

theorem smallest_prime_factor_2023 : smallest_prime_factor 2023 = 7 := 
by 
  sorry

end smallest_prime_factor_2023_l362_362914


namespace max_sum_of_digits_l362_362525

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem max_sum_of_digits : ∃ h m : ℕ, h < 24 ∧ m < 60 ∧
  sum_of_digits h + sum_of_digits m = 24 :=
by
  sorry

end max_sum_of_digits_l362_362525


namespace probability_of_yellow_ball_is_correct_l362_362475

-- Defining the conditions
def red_balls : ℕ := 2
def yellow_balls : ℕ := 5
def blue_balls : ℕ := 4

-- Define the total number of balls
def total_balls : ℕ := red_balls + yellow_balls + blue_balls

-- Define the probability of choosing a yellow ball
def probability_yellow_ball : ℚ := yellow_balls / total_balls

-- The theorem statement we need to prove
theorem probability_of_yellow_ball_is_correct :
  probability_yellow_ball = 5 / 11 :=
sorry

end probability_of_yellow_ball_is_correct_l362_362475


namespace andy_tomatoes_left_l362_362192

theorem andy_tomatoes_left :
  let plants := 50
  let tomatoes_per_plant := 15
  let total_tomatoes := plants * tomatoes_per_plant
  let tomatoes_dried := (2 / 3) * total_tomatoes
  let tomatoes_left_after_drying := total_tomatoes - tomatoes_dried
  let tomatoes_for_marinara := (1 / 2) * tomatoes_left_after_drying
  let tomatoes_left := tomatoes_left_after_drying - tomatoes_for_marinara
  tomatoes_left = 125 := sorry

end andy_tomatoes_left_l362_362192


namespace sqrt_49_mul_sqrt_25_l362_362004

theorem sqrt_49_mul_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_l362_362004


namespace sqrt_expression_simplified_l362_362009

theorem sqrt_expression_simplified :
  sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  sorry

end sqrt_expression_simplified_l362_362009


namespace common_chord_passes_through_C_l362_362131

noncomputable theory

variables {A B C A' B' : Point}
variables {ω1 ω2 : Circle}
variables {P Q : Point}

-- Definitions and conditions
def is_median (A B C A' : Point) : Prop := midpoint (B, C) = A'
def is_constructed_arc (A A' C : Point) (ω : Circle) : Prop := 
    is_median A B C A' ∧ ω = circle_on_median A A' C

def equal_angular_measure_arcs (ω1 ω2 : Circle) : Prop := 
    arc_measure ω1 = arc_measure ω2

-- Main theorem statement
theorem common_chord_passes_through_C
  (h_median_A : is_median A B C A')
  (h_median_B : is_median B A C B')
  (h_arc_A : is_constructed_arc A A' C ω1)
  (h_arc_B : is_constructed_arc B B' C ω2)
  (h_equal_arcs : equal_angular_measure_arcs ω1 ω2) :
  common_chord ω1 ω2 P Q → passes_through P Q C := 
sorry

end common_chord_passes_through_C_l362_362131


namespace cos_pi_div_three_sin_eq_sin_pi_div_three_cos_l362_362326

theorem cos_pi_div_three_sin_eq_sin_pi_div_three_cos :
  ∃ (count : ℕ), count = 2 ∧ 
    ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * π → 
      (cos ((π / 3) * sin x) = sin ((π / 3) * cos x) ↔ 
      ∃! x' : ℝ, x' = x) :=
by
  sorry

end cos_pi_div_three_sin_eq_sin_pi_div_three_cos_l362_362326


namespace apples_distribution_l362_362561

variable (p b t : ℕ)

theorem apples_distribution (p_eq : p = 40) (b_eq : b = p + 8) (t_eq : t = (3 * b) / 8) :
  t = 18 := by
  sorry

end apples_distribution_l362_362561


namespace max_composite_numbers_l362_362795

-- Definitions and conditions
def is_composite (n : ℕ) : Prop := 2 < n ∧ ∃ d, d ∣ n ∧ 1 < d ∧ d < n

def less_than_1500 (n : ℕ) : Prop := n < 1500

def gcd_is_one (a b : ℕ) : Prop := Int.gcd a b = 1

-- The problem statement
theorem max_composite_numbers (numbers : List ℕ) (h_composite : ∀ n ∈ numbers, is_composite n) 
  (h_less_than_1500 : ∀ n ∈ numbers, less_than_1500 n)
  (h_pairwise_gcd_is_one : (numbers.Pairwise gcd_is_one)) : numbers.length ≤ 12 := 
  sorry

end max_composite_numbers_l362_362795


namespace series_convergence_and_sum_l362_362695

theorem series_convergence_and_sum :
  ∃ S : ℝ, has_sum (λ n, if n ≥ 2 then 18 / (n^2 + n - 2) else 0) S ∧ S = 11 := 
sorry

end series_convergence_and_sum_l362_362695


namespace triangle_area_l362_362691

-- Given conditions
variables {A B C : ℝ} {a b c : ℝ} 
-- angle A in triangle ABC
axiom angle_A_eq_pi_div_3 : a * Real.sin B = sqrt 3 * b * Real.cos A → A = Real.pi / 3
-- Side lengths a, b, c and their relations to corresponding angles in triangle ABC
axiom side_lengths : a = 3 → b = 2 * c → Real.cos (Real.pi / 3) = 1 / 2 → c = sqrt 3 → b = 2 * sqrt 3

-- Proof of area
noncomputable def area_triangle_ABC (a b : ℝ) : ℝ :=
  1 / 2 * b * a * Real.sin (Real.pi / 3)

-- The final theorem
theorem triangle_area
  (h1 : a = 3)
  (h2 : b = 2 * sqrt 3)
  (h3 : Real.sin (Real.pi / 3) = sqrt 3 / 2) :
  area_triangle_ABC a b = (3 * sqrt 3) / 2 :=
by sorry

end triangle_area_l362_362691


namespace problem_l362_362907

noncomputable def poly (x : ℝ) : ℝ := 1 - 5 * x - 8 * x^2 + 10 * x^3 + 6 * x^4 + 12 * x^5 + 3 * x^6

noncomputable def horner_eval (x : ℝ) : ℝ×ℝ×ℝ×ℝ×ℝ := 
  let v0 := 3
  let v1 := v0 * x + 12
  let v2 := v1 * x + 6
  let v3 := v2 * x + 10
  let v4 := v3 * x - 8
  let v5 := v4 * x - 5
  (v0, v1, v2, v3, v4)

def max_min_diff (a b c d e : ℝ) : ℝ :=
  max a (max b (max c (max d e))) - min a (min b (min c (min d e)))

theorem problem :
  max_min_diff 3 0 6 (-14) 48 = 62 := by
  sorry

end problem_l362_362907


namespace semicircle_circumference_l362_362463

theorem semicircle_circumference (π : ℝ) (s circumference : ℝ) 
  (side_eq : s = 7) 
  (diameter_side : ∀ (l b : ℝ), l = 8 → b = 6 → 
    4 * s = 2 * (l + b)) : 
  circumference = ((π * s) / 2 + s) :=
by 
  have s_eq : s = 7 := side_eq
  have l := 8
  have b := 6
  have h2 : 2 * (l + b) = 28 := by norm_num
  have h3 : 4 * s = 28 := diameter_side l b rfl rfl
  have h4 : s = 7 := by linarith
  have h5 : circumference = ((π * s) / 2 + s) := by sorry
  
  assumption

end semicircle_circumference_l362_362463


namespace three_person_subcommittees_from_seven_l362_362318

-- Definition of the combinations formula (binomial coefficient)
def choose : ℕ → ℕ → ℕ
| n, k => if k = 0 then 1 else (n * choose (n - 1) (k - 1)) / k 

-- Problem statement in Lean 4
theorem three_person_subcommittees_from_seven : choose 7 3 = 35 :=
by
  -- We would fill in the steps here or use a sorry to skip the proof
  sorry

end three_person_subcommittees_from_seven_l362_362318


namespace stock_price_calculation_l362_362226

def stock_price_end_of_first_year (initial_price : ℝ) (increase_percent : ℝ) : ℝ :=
  initial_price * (1 + increase_percent)

def stock_price_end_of_second_year (price_first_year : ℝ) (decrease_percent : ℝ) : ℝ :=
  price_first_year * (1 - decrease_percent)

theorem stock_price_calculation 
  (initial_price : ℝ)
  (increase_percent : ℝ)
  (decrease_percent : ℝ)
  (final_price : ℝ) :
  initial_price = 120 ∧ 
  increase_percent = 0.80 ∧
  decrease_percent = 0.30 ∧
  final_price = 151.20 → 
  stock_price_end_of_second_year (stock_price_end_of_first_year initial_price increase_percent) decrease_percent = final_price :=
by
  sorry

end stock_price_calculation_l362_362226


namespace sum_of_squares_is_149_l362_362885

-- Define the integers and their sum and product
def integers_sum (b : ℤ) : ℤ := (b - 1) + b + (b + 1)
def integers_product (b : ℤ) : ℤ := (b - 1) * b * (b + 1)

-- Define the condition given in the problem
def condition (b : ℤ) : Prop :=
  integers_product b = 12 * integers_sum b + b^2

-- Define the sum of squares of three consecutive integers
def sum_of_squares (b : ℤ) : ℤ :=
  (b - 1)^2 + b^2 + (b + 1)^2

-- The main statement to be proved
theorem sum_of_squares_is_149 (b : ℤ) (h : condition b) : sum_of_squares b = 149 :=
by
  sorry

end sum_of_squares_is_149_l362_362885


namespace prob_sum_7_9_11_correct_l362_362875

def die1 : List ℕ := [1, 2, 3, 3, 4, 4]
def die2 : List ℕ := [2, 2, 5, 6, 7, 8]

def prob_sum_7_9_11 : ℚ := 
  (1/6 * 1/6 + 1/6 * 1/6) + 2/6 * 3/6

theorem prob_sum_7_9_11_correct :
  prob_sum_7_9_11 = 4 / 9 := 
by
  sorry

end prob_sum_7_9_11_correct_l362_362875


namespace probability_correct_dial_l362_362747

-- Define the problem conditions
def first_three_digits : Finset ℕ := {296, 299, 297}
def last_four_digits : Multiset ℕ := {0, 1, 6, 6}

-- Define the property of interest
def valid_numbers : Finset (List ℕ) :=
  first_three_digits.product (Multiset.toFinset (Multiset.permutations last_four_digits)).map (λ (p : ℕ × Multiset ℕ), p.1 :: p.2.toList)

theorem probability_correct_dial : 
  (1 : ℚ) / (valid_numbers.card : ℚ) = 1 / 36 :=
by
  -- Sorry to indicate the proof is omitted
  sorry

end probability_correct_dial_l362_362747


namespace oleg_max_composite_numbers_l362_362767

theorem oleg_max_composite_numbers : 
  ∃ (S : Finset ℕ), 
    (∀ (n ∈ S), n < 1500 ∧ ∃ p q, prime p ∧ prime q ∧ p ≠ q ∧ p * q = n) ∧ 
    (∀ (a b ∈ S), a ≠ b → gcd a b = 1) ∧ 
    S.card = 12 :=
sorry

end oleg_max_composite_numbers_l362_362767


namespace carla_drinks_water_l362_362212

-- Definitions from the conditions
def total_liquid (s w : ℕ) : Prop := s + w = 54
def soda_water_relation (s w : ℕ) : Prop := s = 3 * w - 6

-- Proof statement
theorem carla_drinks_water : ∀ (s w : ℕ), total_liquid s w ∧ soda_water_relation s w → w = 15 :=
by
  intros s w h,
  sorry

end carla_drinks_water_l362_362212


namespace total_cards_1750_l362_362943

theorem total_cards_1750 (football_cards baseball_cards hockey_cards total_cards : ℕ)
  (h1 : baseball_cards = football_cards - 50)
  (h2 : football_cards = 4 * hockey_cards)
  (h3 : hockey_cards = 200)
  (h4 : total_cards = football_cards + baseball_cards + hockey_cards) :
  total_cards = 1750 :=
sorry

end total_cards_1750_l362_362943


namespace find_quadrilateral_area_l362_362975

-- Defining the initial setup and conditions
variable {α : Type}
variable (triangle : α)
variable (A B C D E F : α)
variable (area_triangle_EFA area_triangle_FAB area_triangle_FBD area_triangle_BDC : ℕ)
variable (total_area_triangle : α → ℕ)

-- Given areas of specific triangles
def initial_areas : Prop :=
  area_triangle_EFA = 5 ∧
  area_triangle_FAB = 10 ∧
  area_triangle_FBD = 10 ∧
  area_triangle_BDC = 8

-- Total area of the quadrilateral
def quadrilateral_area : ℕ :=
  15

-- Main statement to prove
theorem find_quadrilateral_area (h : initial_areas) :
  total_area_triangle quadrilateral_area = 15 := sorry

end find_quadrilateral_area_l362_362975


namespace weekly_earnings_l362_362382

-- Definition of the conditions
def hourly_rate : ℕ := 20
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 4

-- Theorem that conforms to the problem statement
theorem weekly_earnings : hourly_rate * hours_per_day * days_per_week = 640 := by
  sorry

end weekly_earnings_l362_362382


namespace correct_option_l362_362412

-- Definitions for universe set, and subsets A and B
def S : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 5}

-- The proof goal
theorem correct_option : A ⊆ S \ B :=
by
  sorry

end correct_option_l362_362412


namespace sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l362_362083

theorem sqrt_49_mul_sqrt_25_eq_7_sqrt_5 : (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l362_362083


namespace ratio_a_c_l362_362340

theorem ratio_a_c (a b c : ℕ) (h1 : a / b = 5 / 3) (h2 : b / c = 1 / 5) : a / c = 1 / 3 :=
sorry

end ratio_a_c_l362_362340


namespace sqrt_mul_sqrt_l362_362043

theorem sqrt_mul_sqrt (a b c : ℝ) (hac : a = 49) (hbc : b = sqrt 25) (hc : c = 7 * sqrt 5) : sqrt (a * b) = c := 
by
  sorry

end sqrt_mul_sqrt_l362_362043


namespace least_odd_prime_factor_of_2023_pow_8_add_1_l362_362594

theorem least_odd_prime_factor_of_2023_pow_8_add_1 :
  ∃ (p : ℕ), Prime p ∧ (2023^8 + 1) % p = 0 ∧ p % 2 = 1 ∧ p = 97 :=
by
  sorry

end least_odd_prime_factor_of_2023_pow_8_add_1_l362_362594


namespace four_letter_words_with_A_at_least_once_l362_362304

theorem four_letter_words_with_A_at_least_once (A B C D E : Type) :
  let total := 5^4 in
  let without_A := 4^4 in
  total - without_A = 369 :=
by {
  let total := 5^4;
  let without_A := 4^4;
  have : total - without_A = 369 := by sorry;
  exact this;
}

end four_letter_words_with_A_at_least_once_l362_362304


namespace sqrt_49_times_sqrt_25_eq_7sqrt5_l362_362073

theorem sqrt_49_times_sqrt_25_eq_7sqrt5 :
  (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  have h1 : Real.sqrt 25 = 5 := by sorry
  have h2 : 49 * 5 = 245 := by sorry
  have h3 : Real.sqrt 245 = 7 * Real.sqrt 5 := by sorry
  sorry

end sqrt_49_times_sqrt_25_eq_7sqrt5_l362_362073


namespace problem_1a_problem_1b_problem_2_l362_362939

def S (x : ℕ) : ℕ := -- Sum of digits of the natural number x
  sorry

theorem problem_1a :
  ¬ ∃ x : ℕ, x + S(x) + S(S(x)) = 1993 :=
sorry

theorem problem_1b :
  ∃ x : ℕ, x < 1993 ∧
       x + S(x) + S(S(x)) + S(S(S(x))) = 1993 ∧
       x = 1963 :=
sorry

theorem problem_2 (n : ℕ) :
  ∃ a b c : ℕ, n = a^2 + b^2 + c^2 →
  ∃ x y z : ℕ, n^2 = x^2 + y^2 + z^2 :=
sorry

end problem_1a_problem_1b_problem_2_l362_362939


namespace smallest_integer_among_three_l362_362904

theorem smallest_integer_among_three 
  (x y z : ℕ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y)
  (hz_pos : 0 < z)
  (hxy : y - x ≤ 6)
  (hxz : z - x ≤ 6) 
  (hprod : x * y * z = 2808) : 
  x = 12 := 
sorry

end smallest_integer_among_three_l362_362904


namespace parallelepiped_properties_l362_362681

/--
In an oblique parallelepiped with the following properties:
- The height is 12 dm,
- The projection of the lateral edge on the base plane is 5 dm,
- A cross-section perpendicular to the lateral edge is a rhombus with:
  - An area of 24 dm²,
  - A diagonal of 8 dm,
Prove that:
1. The lateral surface area is 260 dm².
2. The volume is 312 dm³.
-/
theorem parallelepiped_properties
    (height : ℝ)
    (projection_lateral_edge : ℝ)
    (area_rhombus : ℝ)
    (diagonal_rhombus : ℝ)
    (lateral_surface_area : ℝ)
    (volume : ℝ) :
  height = 12 ∧
  projection_lateral_edge = 5 ∧
  area_rhombus = 24 ∧
  diagonal_rhombus = 8 ∧
  lateral_surface_area = 260 ∧
  volume = 312 :=
by
  sorry

end parallelepiped_properties_l362_362681


namespace intersection_of_A_and_B_find_a_and_b_l362_362936

open Set

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x^2 < 4}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

-- Part 1: Intersection of A and B
theorem intersection_of_A_and_B : (A ∩ B) = {x : ℝ | 1 < x ∧ x < 2} :=
by sorry

-- Part 2: Find values of a and b such that the solution set of 2x^2 + ax + b < 0 is B
theorem find_a_and_b (a b : ℝ) : 
  (∀ x : ℝ, (2 * x^2 + a * x + b < 0 ↔ 1 < x ∧ x < 3)) → 
  (a = -8 ∧ b = 6) :=
by sorry

end intersection_of_A_and_B_find_a_and_b_l362_362936


namespace sqrt_product_l362_362093

theorem sqrt_product (a b : ℝ) (h1 : a = 49) (h2 : b = 25) : sqrt (a * sqrt b) = 7 * sqrt 5 :=
by sorry

end sqrt_product_l362_362093


namespace sqrt_product_l362_362099

theorem sqrt_product (a b : ℝ) (h1 : a = 49) (h2 : b = 25) : sqrt (a * sqrt b) = 7 * sqrt 5 :=
by sorry

end sqrt_product_l362_362099


namespace base_of_triangle_is_3_point_8_l362_362649

noncomputable def base_of_triangle (area : ℝ) (height : ℝ) : ℝ :=
  (area * 2) / height

theorem base_of_triangle_is_3_point_8 :
  base_of_triangle 9.31 4.9 ≈ 3.8 := 
by
  sorry

end base_of_triangle_is_3_point_8_l362_362649


namespace sum_x_coordinates_of_other_vertices_l362_362327

theorem sum_x_coordinates_of_other_vertices
  (x1 y1 x2 y2 : ℝ)
  (h1 : (x1, y1) = (2, 23))
  (h2 : (x2, y2) = (8, -2)) :
  let x3 := 2 * 5 - x1,
      x4 := 2 * 5 - x2
  in x3 + x4 = 10 :=
by
  sorry

end sum_x_coordinates_of_other_vertices_l362_362327


namespace max_composite_numbers_l362_362797

-- Definitions and conditions
def is_composite (n : ℕ) : Prop := 2 < n ∧ ∃ d, d ∣ n ∧ 1 < d ∧ d < n

def less_than_1500 (n : ℕ) : Prop := n < 1500

def gcd_is_one (a b : ℕ) : Prop := Int.gcd a b = 1

-- The problem statement
theorem max_composite_numbers (numbers : List ℕ) (h_composite : ∀ n ∈ numbers, is_composite n) 
  (h_less_than_1500 : ∀ n ∈ numbers, less_than_1500 n)
  (h_pairwise_gcd_is_one : (numbers.Pairwise gcd_is_one)) : numbers.length ≤ 12 := 
  sorry

end max_composite_numbers_l362_362797


namespace total_time_to_climb_seven_flights_l362_362576

-- Define the conditions
def first_flight_time : ℕ := 15
def difference_between_flights : ℕ := 10
def num_of_flights : ℕ := 7

-- Define the sum of an arithmetic series function
def arithmetic_series_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Define the theorem
theorem total_time_to_climb_seven_flights :
  arithmetic_series_sum first_flight_time difference_between_flights num_of_flights = 315 :=
by
  sorry

end total_time_to_climb_seven_flights_l362_362576


namespace pollutant_decrease_time_l362_362144

theorem pollutant_decrease_time (P P₀ : ℝ) (t : ℝ) (hP : P = P₀ * exp (-0.02 * t)) (h_target : P = P₀ * (1 / 5)) :
  t = 80 :=
by
  sorry

end pollutant_decrease_time_l362_362144


namespace ruby_height_is_192_l362_362660

def height_janet := 62
def height_charlene := 2 * height_janet
def height_pablo := height_charlene + 70
def height_ruby := height_pablo - 2

theorem ruby_height_is_192 : height_ruby = 192 := by
  sorry

end ruby_height_is_192_l362_362660


namespace pirates_treasure_l362_362418

theorem pirates_treasure (x : ℕ) (total : ℕ) :
  (∑ k in Finset.range x, k * (k + 1) / 2 = 6 * x) →
  total = x + 6 * x :=
begin
  intro h,
  sorry
end

end pirates_treasure_l362_362418


namespace probability_x_greater_6_l362_362283

-- Conditions
variable (x : ℝ)
variable (σ : ℝ)
variable (hx : x ∈ Normal 4 σ^2)
variable (hx2 : Π x, x > 2 → P(x) = 0.6)

-- Question and answer
theorem probability_x_greater_6 : P(x > 6) = 0.4 :=
sorry

end probability_x_greater_6_l362_362283


namespace geometric_sequence_formula_max_value_m_l362_362279

/-- Given a geometric sequence {a_n} with a₁ = 1 and common ratio q > 0, and sums S₁(a₁), S₃(a₃),
and S₂(a₂) forming an arithmetic sequence, prove the general term formula for the sequence. --/
theorem geometric_sequence_formula (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) 
  (hq : q > 0) (ha1 : a 1 = 1) 
  (hSn : ∀ n, S n = (1 - q^n) / (1 - q))
  (ar_seq : 2 * (S 3 + a 3) = S 2 + a 2 + S 1 + a 1) :
  ∀ n, a n = (1 / 2)^(n - 1) :=
sorry

/-- For sequences {b_n} and {c_n} with conditions, prove the maximum value of m such that
T_n, the sum of the first n terms of {c_n}, is always greater than or equal to m, is 1/3 --/
theorem max_value_m (b c : ℕ → ℝ) (a : ℕ → ℝ) (T : ℕ → ℝ) (m : ℝ) 
  (hbn : ∀ n, b n / (n + 2) = -real.log 2 (a (n+1)))
  (hbcn : ∀ n, b n * c n = 1)
  (hTn :  ∀ n, T n = (1/2) * (1 + 1/2 - 1/(n+1) - 1/(n+2))) :
  (∀ n, T n ≥ m) → m ≤ 1/3 :=
sorry

end geometric_sequence_formula_max_value_m_l362_362279


namespace height_of_box_l362_362164

theorem height_of_box (h : ℚ) 
  (w : ℚ := 15) 
  (l : ℚ := 20)
  (triangle_area : ℚ := 36) :
  let h_frac := rat.mk_nat 63 13 in
  h = h_frac -> p + q = 76 :=
by
  sorry

end height_of_box_l362_362164


namespace principal_amount_is_400_l362_362924

theorem principal_amount_is_400
  (R : ℝ)
  (P : ℝ)
  (h1 : SI = (P * R * 10) / 100)
  (h2 : SI_new = (P * (R + 5) * 10) / 100)
  (h3 : SI_new - SI = 200) : P = 400 :=
begin
  sorry
end

end principal_amount_is_400_l362_362924


namespace primes_between_40_and_50_l362_362325

theorem primes_between_40_and_50 : (finset.filter (λ n, nat.prime n) (finset.Icc 40 50)).card = 3 := sorry

end primes_between_40_and_50_l362_362325


namespace find_x_l362_362256

def a : ℝ × ℝ × ℝ := (2, -1, 3)
def c (x : ℝ) : ℝ × ℝ × ℝ := (1, -2 * x, 0)
def b (x : ℝ) : ℝ × ℝ × ℝ := (4, x, 0)

theorem find_x (x : ℝ) : (a.1 + (b x).1, a.2 + (b x).2, a.3 + (b x).3) = (6, x - 1, 3) ∧ (6 + (x - 1) * (-2 * x) + 3 * (-2 * x)) = 0 → x = 1 :=
by
  sorry

end find_x_l362_362256


namespace expression_I_expression_II_l362_362999

theorem expression_I : (sqrt (2 + 1/4) - (-2)^0 - (27/8)^(-2/3) + 1.5^(-2) = 1/2) :=
by
  sorry

theorem expression_II : (log 2 5 / log 2 10 + log 10 2 - log 4 8 + 3 ^ log 3 2 = log 10 5 + 1/2) :=
by
  sorry

end expression_I_expression_II_l362_362999


namespace wall_area_calculation_l362_362383

variables (cost_per_gallon : ℝ) (coverage_per_gallon : ℝ) (total_contribution : ℝ) (coats_needed : ℝ)

-- Define the given conditions
def jason_contribution : ℝ := 180
def jeremy_contribution : ℝ := 180
def total_contribution := jason_contribution + jeremy_contribution
def cost_per_gallon := 45
def coverage_per_gallon := 400
def coats_needed := 2

-- Calculate the number of gallons that can be bought
def gallons_bought : ℝ := total_contribution / cost_per_gallon

-- Calculate the number of gallons needed for a single coat
def gallons_per_coat : ℝ := gallons_bought / coats_needed

-- Calculate the total area of the walls
def total_area_of_walls : ℝ := gallons_per_coat * coverage_per_gallon

-- The theorem to be proved
theorem wall_area_calculation : total_area_of_walls = 1600 :=
by
  rw [gallons_per_coat, total_contribution, cost_per_gallon, coverage_per_gallon, coats_needed]
  sorry

end wall_area_calculation_l362_362383


namespace opposite_meaning_for_option_C_l362_362552

def opposite_meaning (a b : Int) : Bool :=
  (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0)

theorem opposite_meaning_for_option_C :
  (opposite_meaning 300 (-500)) ∧ 
  ¬ (opposite_meaning 5 (-5)) ∧ 
  ¬ (opposite_meaning 180 90) ∧ 
  ¬ (opposite_meaning 1 (-1)) :=
by
  unfold opposite_meaning
  sorry

end opposite_meaning_for_option_C_l362_362552


namespace city_population_distribution_l362_362957

theorem city_population_distribution :
  (20 + 35) = 55 :=
by
  sorry

end city_population_distribution_l362_362957


namespace degree_f_x2_mul_g_x3_l362_362401

variable {R : Type*} [CommRing R]

noncomputable def f (x : R) : R :=
  sorry

noncomputable def g (x : R) : R :=
  sorry

axiom deg_f : Polynomial.degree (Polynomial.of_fn f) = 3
axiom deg_g : Polynomial.degree (Polynomial.of_fn g) = 6

theorem degree_f_x2_mul_g_x3 : 
  Polynomial.degree (Polynomial.of_fn (λ x, f (x^2) * g (x^3))) = 24 :=
by
  sorry

end degree_f_x2_mul_g_x3_l362_362401


namespace sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l362_362079

theorem sqrt_49_mul_sqrt_25_eq_7_sqrt_5 : (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l362_362079


namespace computer_price_after_9_years_l362_362445

theorem computer_price_after_9_years 
  (initial_price : ℝ) (decrease_factor : ℝ) (years : ℕ) 
  (initial_price_eq : initial_price = 8100)
  (decrease_factor_eq : decrease_factor = 1 - 1/3)
  (years_eq : years = 9) :
  initial_price * (decrease_factor ^ (years / 3)) = 2400 := 
by
  sorry

end computer_price_after_9_years_l362_362445


namespace cosine_shift_right_eq_l362_362483

notation "π" => Real.pi

theorem cosine_shift_right_eq :
  ∀ (x : ℝ), 2 * cos (2 * (x - π / 8)) = 2 * cos (2 * x - π / 4) :=
by
  intro x
  sorry

end cosine_shift_right_eq_l362_362483


namespace no_good_coloring_method_exists_l362_362125

noncomputable def circle (O: Type) (r: ℝ) := sorry

def good_circle (c : circle ℝ 2) :=
  ∀ (r : ℝ) (O : ℝ) (P₁ P₂ P₃: ℝ), 
    (r >= 1 ∧ inscribed_equilateral_triangle O r P₁ P₂ P₃) →
    (colored_differently P₁ P₂ P₃)

theorem no_good_coloring_method_exists :
  ¬ ∃ (coloring_method : ∀ (p : ℝ × ℝ), Color), 
    ∀ (c : circle ℝ 2), good_circle c := 
sorry

end no_good_coloring_method_exists_l362_362125


namespace length_of_BG_l362_362357

variable (A B C D E F G : Type)
variable [InnerProductSpace ℝ A]
variable [InnerProductSpace ℝ B]
variable [InnerProductSpace ℝ C]
variable [InnerProductSpace ℝ D]
variable [InnerProductSpace ℝ E]
variable [InnerProductSpace ℝ F]
variable [InnerProductSpace ℝ G]

-- Define the conditions of the parallelogram and the points
variables (AD BC EF EG BG : ℝ)
premise : (isParallelogram A B C D) -- A parallelogram property
premise : AD = BC -- Sides of parallelogram
premise : EF = 45 -- Given length of EF
premise : EG = 15 -- Given length of EG

-- Condition for the line intersections
variables (lineBF : Line B F)
variables (lineAC : Line A C)
variables (lineDC : Line D C)
variables (pointE : (Intersects lineBF lineAC E))
variables (pointG : (Intersects lineBF lineDC G))

-- Objective to prove
theorem length_of_BG : BG = 20 :=
by
  -- Here will be the proofs based on the conditions above
  sorry

end length_of_BG_l362_362357


namespace sqrt_product_l362_362096

theorem sqrt_product (a b : ℝ) (h1 : a = 49) (h2 : b = 25) : sqrt (a * sqrt b) = 7 * sqrt 5 :=
by sorry

end sqrt_product_l362_362096


namespace extreme_value_point_range_l362_362613

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∃ (x : ℝ), x > 0 ∧ e^x + a * x = 0

theorem extreme_value_point_range (a : ℝ) :
  (range_of_a a) → a < -1 :=
sorry

end extreme_value_point_range_l362_362613


namespace tables_needed_l362_362953

-- Conditions
def n_invited : ℕ := 18
def n_no_show : ℕ := 12
def capacity_per_table : ℕ := 3

-- Calculation of attendees
def n_attendees : ℕ := n_invited - n_no_show

-- Proof for the number of tables needed
theorem tables_needed : (n_attendees / capacity_per_table) = 2 := by
  -- Sorry will be here to show it's incomplete
  sorry

end tables_needed_l362_362953


namespace necessary_and_sufficient_condition_l362_362517

theorem necessary_and_sufficient_condition (a : ℝ) : (a > 1) ↔ ∀ x : ℝ, (x^2 - 2*x + a > 0) :=
by 
  sorry

end necessary_and_sufficient_condition_l362_362517


namespace problem_l362_362980

-- Definition of triangular number
def is_triangular (n k : ℕ) := n = k * (k + 1) / 2

-- Definition of choosing 2 marbles
def choose_2 (n m : ℕ) := n = m * (m - 1) / 2

-- Definition of Cathy's condition
def cathy_condition (n s : ℕ) := s * s < 2 * n ∧ 2 * n - s * s = 20

theorem problem (n k m s : ℕ) :
  is_triangular n k →
  choose_2 n m →
  cathy_condition n s →
  n = 210 :=
by
  sorry

end problem_l362_362980


namespace annie_total_distance_traveled_l362_362194

-- Definitions of conditions
def walk_distance : ℕ := 5
def bus_distance : ℕ := 7
def total_distance_one_way : ℕ := walk_distance + bus_distance
def total_distance_round_trip : ℕ := total_distance_one_way * 2

-- Theorem statement to prove the total number of blocks traveled
theorem annie_total_distance_traveled : total_distance_round_trip = 24 :=
by
  sorry

end annie_total_distance_traveled_l362_362194


namespace oleg_max_composite_numbers_l362_362774

theorem oleg_max_composite_numbers : 
  ∃ (S : Finset ℕ), 
    (∀ (n ∈ S), n < 1500 ∧ ∃ p q, prime p ∧ prime q ∧ p ≠ q ∧ p * q = n) ∧ 
    (∀ (a b ∈ S), a ≠ b → gcd a b = 1) ∧ 
    S.card = 12 :=
sorry

end oleg_max_composite_numbers_l362_362774


namespace integral_eval_l362_362328

theorem integral_eval : ∫ x in (1:ℝ)..(2:ℝ), (2*x + 1/x) = 3 + Real.log 2 := by
  sorry

end integral_eval_l362_362328


namespace triangle_inequality_l362_362423

variables (a b c : ℝ) (S : ℝ)
noncomputable def p : ℝ := (a + b + c) / 2
noncomputable def herons_area : ℝ := real.sqrt (p a b c * (p a b c - a) * (p a b c - b) * (p a b c - c))

theorem triangle_inequality (habc : 0 < a) (hbbc : 0 < b) (hcbc : 0 < c) (hS : S = herons_area a b c) :
  a^2 + b^2 + c^2 - (a - b)^2 - (b - c)^2 - (c - a)^2 ≥ 4 * real.sqrt 3 * S := sorry

end triangle_inequality_l362_362423


namespace total_shaded_cubes_l362_362879

/-
The large cube consists of 27 smaller cubes, each face is a 3x3 grid.
Opposite faces are shaded in an identical manner, with each face having 5 shaded smaller cubes.
-/

theorem total_shaded_cubes (number_of_smaller_cubes : ℕ)
  (face_shade_pattern : ∀ (face : ℕ), ℕ)
  (opposite_face_same_shade : ∀ (face1 face2 : ℕ), face1 = face2 → face_shade_pattern face1 = face_shade_pattern face2)
  (faces_possible : ∀ (face : ℕ), face < 6)
  (each_face_shaded_squares : ∀ (face : ℕ), face_shade_pattern face = 5)
  : ∃ (n : ℕ), n = 20 :=
by
  sorry

end total_shaded_cubes_l362_362879


namespace common_difference_l362_362633

-- Define arithmetic sequence and its sum
def S (n : ℕ) (a1 : ℝ) (d : ℝ) : ℝ :=
  n * a1 + d * (n * (n - 1)) / 2

-- Given conditions
variables (a1 d : ℝ) 

-- Conditions extracted from problem
axiom cond1 : S 4 a1 d = 3 * S 2 a1 d
axiom cond2 : a1 + 6 * d = 15

-- The goal statement
theorem common_difference :
  d = 2 :=
sorry

end common_difference_l362_362633


namespace local_minimum_at_2_l362_362666

def f (x a : ℝ) := x^3 - ((a / 2) + 3) * x^2 + 2 * a * x + 3

def f_prime (x a : ℝ) := 3 * x^2 - (a + 6) * x + 2 * a

theorem local_minimum_at_2 (a : ℝ) :
  (∀ x, f_prime x a = (x - 2) * (3 * x - a)) ∧ ∀ x, f_prime x a = 0 → f x a = f 2 a → a < 6 := sorry

end local_minimum_at_2_l362_362666


namespace find_angle_l362_362744

variables {a b c : ℝ^3}
variables (θ : ℝ)

-- Given conditions
def norm_a : ∥a∥ = 2 := sorry
def norm_b : ∥b∥ = 1 := sorry
def norm_c : ∥c∥ = 3 := sorry
def vector_equation : a × (b × c) + 2 • b = 0 := sorry

-- Theorem we want to prove
theorem find_angle (h₁ : norm_a) (h₂ : norm_b) (h₃ : norm_c) (h₄ : vector_equation) : θ = real.arccos (- (2/3)) :=
sorry

end find_angle_l362_362744


namespace oleg_max_composite_numbers_l362_362770

theorem oleg_max_composite_numbers : 
  ∃ (S : Finset ℕ), 
    (∀ (n ∈ S), n < 1500 ∧ ∃ p q, prime p ∧ prime q ∧ p ≠ q ∧ p * q = n) ∧ 
    (∀ (a b ∈ S), a ≠ b → gcd a b = 1) ∧ 
    S.card = 12 :=
sorry

end oleg_max_composite_numbers_l362_362770


namespace find_a_l362_362876

theorem find_a (a : ℝ) (h1 : f a = 7) (h2 : a > 0) (h3 : a < 3) : a = 2 :=
  by
  sorry

def f(x : ℝ) : ℝ := 2 * x^2 - 1

end find_a_l362_362876


namespace find_expression_for_a_n_l362_362689

theorem find_expression_for_a_n (a : ℕ → ℤ) (h : ∀ n : ℕ, n > 0 → ∑ i in Finset.range n, (i + 1) * a (i + 1) = 2 * n * (n - 1) * (n + 1)) : 
  ∀ n, a n = 6 * (n - 1) :=
by
  sorry

end find_expression_for_a_n_l362_362689


namespace part1_part2_l362_362717

-- Define the conditions
def triangle_conditions (a b c : ℝ) (A B C : ℝ) : Prop :=
  sin C * sin (A - B) = sin B * sin (C - A) 

-- Define the conclusion for part (1)
def proof_part1 (a b c : ℝ) (A B C : ℝ) (h : triangle_conditions a b c A B C) : Prop :=
  2 * a ^ 2 = b ^ 2 + c ^ 2

-- Define the conditions for part (2)
def triangle_conditions_part2 (a b c A : ℝ) : Prop :=
  a = 5 ∧ cos A = 25 / 31 

-- Define the conclusion for part (2)
def proof_part2 (a b c A : ℝ) (h : triangle_conditions_part2 a b c A) : Prop :=
  a + b + c = 14

-- The Lean statements for the complete problem
theorem part1 (a b c A B C : ℝ) 
  (h : triangle_conditions a b c A B C) : 
  proof_part1 a b c A B C h := 
sorry

theorem part2 (a b c A : ℝ) 
  (h : triangle_conditions_part2 a b c A) : 
  proof_part2 a b c A h := 
sorry

end part1_part2_l362_362717


namespace equal_roots_of_quadratic_l362_362254

theorem equal_roots_of_quadratic (k : ℝ) : 
  ( ∀ x : ℝ, 2 * k * x^2 + 7 * k * x + 2 = 0 → x = x ) ↔ k = 16 / 49 :=
by
  sorry

end equal_roots_of_quadratic_l362_362254


namespace rate_of_dividend_is_12_l362_362963

-- Defining the conditions
def total_investment : ℝ := 4455
def price_per_share : ℝ := 8.25
def annual_income : ℝ := 648
def face_value_per_share : ℝ := 10

-- Expected rate of dividend
def expected_rate_of_dividend : ℝ := 12

-- The proof problem statement: Prove that the rate of dividend is 12% given the conditions.
theorem rate_of_dividend_is_12 :
  ∃ (r : ℝ), r = 12 ∧ annual_income = 
    (total_investment / price_per_share) * (r / 100) * face_value_per_share :=
by 
  use 12
  sorry

end rate_of_dividend_is_12_l362_362963


namespace find_f_l362_362244

-- Define the median of three real numbers.
def median (x y z : ℝ) : ℝ :=
  if x ≤ y then
    if y ≤ z then y else max x z
  else
    if x ≤ z then x else max y z

-- Define the property of function f.
def median_property (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ (a b c : ℝ), median (f a b) (f b c) (f c a) = median a b c

-- Define the main theorem statement.
theorem find_f (f : ℝ → ℝ → ℝ) (h : median_property f) : 
  (∀ (x y : ℝ), f x y = x) ∨ (∀ (x y : ℝ), f x y = y) :=
sorry

end find_f_l362_362244


namespace enclosed_region_area_l362_362730

noncomputable def g (x : ℝ) := 2 - Real.sqrt (4 - x^2)

theorem enclosed_region_area :
  (∀ x, -2 ≤ x ∧ x ≤ 2 → g x = 2 - Real.sqrt (4 - x^2)) →
  (∃ (area : ℝ), area = π - 1 / 2) :=
by
  intro h
  use π - 1 / 2
  sorry

end enclosed_region_area_l362_362730


namespace smallest_n_l362_362502

theorem smallest_n (n : ℕ) (h1 : ∃ k : ℕ, 3 * n = k ^ 2) (h2 : ∃ m : ℕ, 5 * n = m ^ 5) : n = 151875 := sorry

end smallest_n_l362_362502


namespace coeff_quadratic_const_l362_362609

-- Given quadratic equation
def quad_eq : ℝ → ℝ := λ x => 3 * x^2 + 1 - 6 * x

-- Property to prove
theorem coeff_quadratic_const : (∀ x : ℝ, quad_eq x = -3 * x^2 + 6 * x - 1) →
  (∃ (a b : ℝ), a = -3 ∧ b = -1) :=
by
  intro h
  use [-3, -1]
  exact ⟨rfl, rfl⟩
  sorry

end coeff_quadratic_const_l362_362609


namespace max_shaded_squares_l362_362673

theorem max_shaded_squares (m n : ℕ) (h_m : m = 19) (h_n : n = 89) :
  ∃ k : ℕ, k = 890 ∧
  (∀ (grid : Matrix ℕ m n), (∀ i j, i < m - 1 → j < n - 1 → 
    (grid i j + grid (i+1) j + grid i (j+1) + grid (i+1) (j+1) ≤ 2) →
    (∑ i in Finset.range m, ∑ j in Finset.range n, grid i j) = k)) :=
sorry

end max_shaded_squares_l362_362673


namespace sum_of_bases_l362_362356

theorem sum_of_bases (S₁ S₂ G₁ G₂ : ℚ)
  (h₁ : G₁ = 4 * S₁ / (S₁^2 - 1) + 8 / (S₁^2 - 1))
  (h₂ : G₂ = 8 * S₁ / (S₁^2 - 1) + 4 / (S₁^2 - 1))
  (h₃ : G₁ = 3 * S₂ / (S₂^2 - 1) + 6 / (S₂^2 - 1))
  (h₄ : G₂ = 6 * S₂ / (S₂^2 - 1) + 3 / (S₂^2 - 1)) :
  S₁ + S₂ = 23 :=
by
  sorry

end sum_of_bases_l362_362356


namespace food_beverages_percentage_l362_362165

-- Given conditions
def rent_fraction : ℝ := 1/4
def food_beverages_fraction : ℝ := 1/4
def remaining_budget_fraction (B : ℝ) : ℝ := B - (rent_fraction * B)

-- The main theorem to prove: The percentage of the budget for food and beverages is 18.75%
theorem food_beverages_percentage (B : ℝ) (hB : B ≠ 0) :
  let remaining_budget := remaining_budget_fraction B
  let food_and_beverages := food_beverages_fraction * remaining_budget
  let percentage := (food_and_beverages / B) * 100
  percentage = 18.75 := 
by
  sorry

end food_beverages_percentage_l362_362165


namespace correct_unit_l362_362866

variables (A : Type) (school_area : ℕ)

def area_of_school := 15000
def units : Type := "Square meters"

theorem correct_unit (school_area : ℕ) : school_area = 15000 → units = "Square meters" :=
by
  intros h
  sorry

end correct_unit_l362_362866


namespace volume_of_one_pizza_piece_l362_362169

theorem volume_of_one_pizza_piece
  (h : ℝ) (d : ℝ) (n : ℕ)
  (h_eq : h = 1 / 2)
  (d_eq : d = 16)
  (n_eq : n = 16) :
  ((π * (d / 2)^2 * h) / n) = 2 * π :=
by
  rw [h_eq, d_eq, n_eq]
  sorry

end volume_of_one_pizza_piece_l362_362169


namespace sqrt_49_times_sqrt_25_l362_362026

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 :=
by
  sorry

end sqrt_49_times_sqrt_25_l362_362026


namespace number_of_segments_after_iterations_length_of_segments_after_iterations_segments_and_length_l362_362545

theorem number_of_segments_after_iterations (n : ℕ) : 
  ∀ (a : ℕ), a = 16 → (2^a = 2^16) :=
by
  intros n h
  rw h
  rfl

theorem length_of_segments_after_iterations : 
  ∀ (a : ℕ), a = 16 → (1 / 3^a = 1 / 3^16) :=
by
  intros n h
  rw h
  rfl

theorem segments_and_length (a : ℕ) : 
  a = 16 → ∃ (num_segments : ℕ) (segment_length : ℚ), 
  num_segments = 2^16 ∧ segment_length = 1 / 3^16 :=
by
  intros h
  use 2^16, 1 / 3^16
  split
  { rw number_of_segments_after_iterations a
    exact nat.eq_refl 16
    exact a
    exact h
  }
  { rw length_of_segments_after_iterations a
    exact nat.eq_refl 16
    exact a
    exact h
  }

end number_of_segments_after_iterations_length_of_segments_after_iterations_segments_and_length_l362_362545


namespace trig_identity_l362_362624

theorem trig_identity (x : ℝ) (h : 2 * Real.cos x - 3 * Real.sin x = 4) : 
  2 * Real.sin x + 3 * Real.cos x = 1 ∨ 2 * Real.sin x + 3 * Real.cos x = 3 :=
sorry

end trig_identity_l362_362624


namespace sum_of_integer_solutions_l362_362367

theorem sum_of_integer_solutions :
  (∑ x in finset.filter (λ x : ℤ, 3 * (x + 2) ≥ x - 1 ∧ (5 - x) / 2 < 4 - 2 * x) (finset.Icc (-3) 0)) = -6 :=
by
  sorry

end sum_of_integer_solutions_l362_362367


namespace increasing_function_of_positive_derivative_l362_362604

theorem increasing_function_of_positive_derivative {a b : ℝ} {f : ℝ → ℝ} (h : ∀ x ∈ Icc a b, 0 < deriv f x) :
  ∀ x y ∈ Icc a b, x < y → f x < f y :=
by
  sorry

end increasing_function_of_positive_derivative_l362_362604


namespace sqrt_nested_l362_362063

theorem sqrt_nested : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_nested_l362_362063


namespace num_polynomials_of_form_l362_362224

theorem num_polynomials_of_form (n : ℕ) (a : Fin (n + 1) → ℤ) :
  (Finset.filter (λ (k : Fin (n + 1) → ℤ), (∑ i, |k i|) + 2 * n = 5)
    (Finset.pi (Finset.fin (n + 1)) (λ i, Finset.range 6))).card = 11 :=
sorry

end num_polynomials_of_form_l362_362224


namespace max_composite_numbers_l362_362790

theorem max_composite_numbers (S : Finset ℕ) (h1 : ∀ n ∈ S, n < 1500) (h2 : ∀ m n ∈ S, m ≠ n → Nat.gcd m n = 1) : S.card ≤ 12 := sorry

end max_composite_numbers_l362_362790


namespace remainder_of_55_power_55_plus_55_div_56_l362_362928

theorem remainder_of_55_power_55_plus_55_div_56 :
  (55 ^ 55 + 55) % 56 = 54 :=
by
  -- to be filled with the proof
  sorry

end remainder_of_55_power_55_plus_55_div_56_l362_362928


namespace max_composite_numbers_l362_362786

theorem max_composite_numbers (S : Finset ℕ) (h1 : ∀ n ∈ S, n < 1500) (h2 : ∀ m n ∈ S, m ≠ n → Nat.gcd m n = 1) : S.card ≤ 12 := sorry

end max_composite_numbers_l362_362786


namespace max_composite_numbers_l362_362808
open Nat

theorem max_composite_numbers : 
  ∃ X : Finset Nat, 
  (∀ x ∈ X, x < 1500 ∧ ¬Prime x) ∧ 
  (∀ x y ∈ X, x ≠ y → gcd x y = 1) ∧ 
  X.card = 12 := 
sorry

end max_composite_numbers_l362_362808


namespace sin_C_is_correct_area_of_ABC_l362_362692

-- Given data and conditions
def A : Real := 135 * Real.pi / 180  -- Converting degrees to radians
def b : Real := 2
def c : Real := Real.sqrt 2

-- 1. Proof for sin C
theorem sin_C_is_correct : ∀ (A b c : Real), A = 135 * Real.pi / 180 → b = 2 → c = Real.sqrt 2 → 
  ∃ (C: Real), Real.sin C = Real.sqrt 10 / 10 :=
begin
  intros A b c hA hb hc,
  use Real.arcsin (Real.sqrt 10 / 10),
  sorry
end

-- Definition using the algebraic properties and trigonometric setup for the area problem
def D_is_on_BC (D : Real → Real → Real → Real) (A B C : Real) : Prop :=
  A * B + B * C = 0

-- 2. Proof for area of triangle ABD
theorem area_of_ABC : ∀ (A b c : Real), A = 135 * Real.pi / 180 → b = 2 → c = Real.sqrt 2 → 
  (AC_perp_AD : ∀ A C D, A * C + C * D = A * D → False) →
  ∃ (area_ABD : Real), area_ABD = 1 / 3 :=
begin
  intros A b c hA hb hc hAD,
  sorry
end

end sin_C_is_correct_area_of_ABC_l362_362692


namespace max_composite_numbers_l362_362782

theorem max_composite_numbers (s : set ℕ) (hs : ∀ n ∈ s, n < 1500 ∧ ∃ p : ℕ, prime p ∧ p ∣ n) (hs_gcd : ∀ x y ∈ s, x ≠ y → Nat.gcd x y = 1) :
  s.card ≤ 12 := 
by sorry

end max_composite_numbers_l362_362782


namespace find_S_value_l362_362446

-- Define the quadrilateral properties and conditions
variables {a b c d R S : ℝ}
axiom h1 : a^2 + b^2 = 25  -- For AB
axiom h2 : b^2 + c^2 = 16  -- For BC
axiom h3 : c^2 + d^2 = R^2  -- For CD
axiom h4 : d^2 + a^2 = S^2  -- For DA
axiom hR : R = 3  -- Given R value

-- Theorem statement with conclusion
theorem find_S_value : S = 3 * Real.sqrt 2 :=
begin
  sorry -- Proof is not required
end

end find_S_value_l362_362446


namespace max_composite_numbers_l362_362778

theorem max_composite_numbers (s : set ℕ) (hs : ∀ n ∈ s, n < 1500 ∧ ∃ p : ℕ, prime p ∧ p ∣ n) (hs_gcd : ∀ x y ∈ s, x ≠ y → Nat.gcd x y = 1) :
  s.card ≤ 12 := 
by sorry

end max_composite_numbers_l362_362778


namespace find_b_of_expression_l362_362344

theorem find_b_of_expression (y : ℝ) (b : ℝ) (hy : y > 0)
  (h : (7 / 10) * y = (8 * y) / b + (3 * y) / 10) : b = 20 :=
sorry

end find_b_of_expression_l362_362344


namespace additional_cost_per_international_letter_l362_362581

-- Definitions from the conditions
def domestic_cost (num_letters : ℕ) (cost_per_letter : ℝ) : ℝ :=
  num_letters * cost_per_letter

def international_cost (weight : ℝ) (rate_per_gram : ℝ) : ℝ :=
  weight * rate_per_gram

def total_cost (domestic : ℝ) (international1 : ℝ) (international2 : ℝ) : ℝ :=
  domestic + international1 + international2

-- Problem and conditions
def deborah_total_cost : ℝ := 6.30
def domestic_num_letters : ℕ := 2
def domestic_price_per_letter : ℝ := 1.08
def countryA_weight : ℝ := 25.0
def countryA_rate : ℝ := 0.05
def countryB_weight : ℝ := 45.0
def countryB_rate : ℝ := 0.04

-- Lean Statement of the Proof Problem
theorem additional_cost_per_international_letter :
  let domestic := domestic_cost domestic_num_letters domestic_price_per_letter in
  let international1 := international_cost countryA_weight countryA_rate in
  let international2 := international_cost countryB_weight countryB_rate in
  let calculated_total := total_cost domestic international1 international2 in
  let additional_cost := deborah_total_cost - calculated_total in
  let additional_cost_per_letter := additional_cost / 2 in
  additional_cost_per_letter = 0.55 :=
by
  sorry

end additional_cost_per_international_letter_l362_362581


namespace problem1_problem2_problem3_problem4_l362_362208

-- Problem 1
theorem problem1 : 2 * Real.sqrt 18 - Real.sqrt 50 + Real.cbrt 125 = 5 + Real.sqrt 2 :=
by sorry

-- Problem 2
theorem problem2 : (Real.sqrt 6 + Real.sqrt 5) * (Real.sqrt 6 - Real.sqrt 5) + (Real.sqrt 5 - 1)^2 = 7 - 2 * Real.sqrt 5 :=
by sorry

-- Problem 3
theorem problem3 : (Real.sqrt 48 - Real.sqrt 75) * Real.sqrt (4 / 3) = -2 :=
by sorry

-- Problem 4
theorem problem4 : (-1 / 2)^(-2) + (Real.pi + Real.sqrt 3)^0 - Real.cbrt 64 + abs (Real.sqrt 3 - 2) = 3 - Real.sqrt 3 :=
by sorry

end problem1_problem2_problem3_problem4_l362_362208


namespace sqrt_mul_sqrt_l362_362102

theorem sqrt_mul_sqrt (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : sqrt (a * sqrt b) = sqrt a * sqrt (sqrt b) := by
  sorry

example : sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  have h1 : sqrt 25 = 5 := by 
    norm_num
  have h2 : sqrt (49 * 5) = sqrt 245 := by  
    unfold sqrt
  have h3 : sqrt 245 = 7 * sqrt 5 := by 
    norm_num
  rw [h1, h2, h3]
  norm_num

end sqrt_mul_sqrt_l362_362102


namespace triangle_ratio_l362_362371

theorem triangle_ratio (X Y Z P : Type) (XY XZ YZ : ℝ) 
  (hXY : XY = 20) (hXZ : XZ = 30) (hYZ : YZ = 28)
  (is_angle_bisector : ∃ Q, ∃ R, ∃ X Q Z R · line(XXX, XY, XXX, XZ) ∧ ∠XYZ = ∠ZYX) :
  (triangle.area X Y P) / (triangle.area X Z P) = 2/3 := by
  sorry

end triangle_ratio_l362_362371


namespace cauliflower_area_l362_362961

theorem cauliflower_area
  (s : ℕ) (a : ℕ) 
  (H1 : s * s / a = 40401)
  (H2 : s * s / a = 40000) :
  a = 1 :=
sorry

end cauliflower_area_l362_362961


namespace expenditure_difference_l362_362220

noncomputable def final_price_x (initial_price : ℝ) :=
  let increased_10 := initial_price * 1.10
  let decreased_12 := increased_10 * 0.88
  let increased_5 := decreased_12 * 1.05
  let increased_7 := increased_5 * 1.07
  increased_7

noncomputable def final_price_y (initial_price : ℝ) :=
  let decreased_7 := initial_price * 0.93
  let increased_8 := decreased_7 * 1.08
  let increased_5 := increased_8 * 1.05
  let decreased_6 := increased_5 * 0.94
  decreased_6

noncomputable def amount_spent (final_price : ℝ) (percentage : ℝ) :=
  final_price * percentage

def net_difference_expenditure : ℝ :=
  let initial_price := 100
  let final_price_prod_x := final_price_x initial_price
  let final_price_prod_y := final_price_y initial_price
  let spent_by_A :=
    amount_spent final_price_prod_x 0.65 + amount_spent final_price_prod_y 0.80
  let spent_by_B :=
    amount_spent final_price_prod_x 0.75 + amount_spent final_price_prod_y 0.90
  spent_by_B - spent_by_A

theorem expenditure_difference :
  net_difference_expenditure = 20.79 :=
by
  sorry

end expenditure_difference_l362_362220


namespace sqrt_49_times_sqrt25_eq_7_sqrt5_l362_362038

-- Definitions based on conditions
def sqrt_25 := Real.sqrt 25
def sqrt_49 := Real.sqrt 49

theorem sqrt_49_times_sqrt25_eq_7_sqrt5 :
  Real.sqrt (49 * sqrt_25) = 7 * Real.sqrt 5 := by
sorry

end sqrt_49_times_sqrt25_eq_7_sqrt5_l362_362038


namespace karen_packs_ham_sandwich_three_days_l362_362388

theorem karen_packs_ham_sandwich_three_days :
  ∃ (H : ℕ), H = 3 ∧ 
    (let P := 2 in 
     let total_days := 5 in
     H = total_days - P ∧ 
     (let prob_cake := 1 / 5 in
      let prob_ham := H / total_days in
      prob_ham * prob_cake = 12 / 100)) :=
sorry

end karen_packs_ham_sandwich_three_days_l362_362388


namespace max_percentage_of_school_year_missable_is_five_l362_362473

-- Define the total number of school days in a year
def total_days : ℕ := 180

-- Define the days Hazel has already missed
def days_missed : ℕ := 6

-- Define the additional days Hazel can miss
def additional_days_allowed : ℕ := 3

-- Define the total days Hazel can miss
def total_days_missable : ℕ := days_missed + additional_days_allowed

-- Define the maximum percentage of the school year Hazel can miss
def max_percentage_missable := (total_days_missable / total_days.toFloat) * 100

-- The statement to prove
theorem max_percentage_of_school_year_missable_is_five : 
  max_percentage_missable = 5 :=
by
  sorry

end max_percentage_of_school_year_missable_is_five_l362_362473


namespace ratio_c_to_a_l362_362611

theorem ratio_c_to_a (a c : ℝ) (P₁ P₂ P₃ P₄ : ℝ × ℝ) :
  (dist P₁ P₂ = a ∧ dist P₂ P₃ = a ∧ dist P₃ P₁ = 2 * a ∧
   dist P₁ P₄ = a ∧ dist P₂ P₄ = 2 * a ∧ dist P₃ P₄ = c) →
  c = a * sqrt 3 :=
by
  sorry

end ratio_c_to_a_l362_362611


namespace total_blocks_traveled_l362_362195

-- Given conditions as definitions
def annie_walked_blocks : ℕ := 5
def annie_rode_blocks : ℕ := 7

-- The total blocks Annie traveled
theorem total_blocks_traveled : annie_walked_blocks + annie_rode_blocks + (annie_walked_blocks + annie_rode_blocks) = 24 := by
  sorry

end total_blocks_traveled_l362_362195


namespace ferry_boat_tourists_l362_362146

theorem ferry_boat_tourists :
  let trips := 7 in
  let initial_tourists := 100 in
  let decrement := 2 in
  (trips * (2 * initial_tourists + (trips - 1) * -decrement)) / 2 = 658 := by
  sorry

end ferry_boat_tourists_l362_362146


namespace max_composite_numbers_with_gcd_one_l362_362763

theorem max_composite_numbers_with_gcd_one : 
  ∃ (S : Finset ℕ), 
    (∀ x ∈ S, Nat.isComposite x) ∧ 
    (∀ x ∈ S, x < 1500) ∧ 
    (∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y → Nat.gcd x y = 1) ∧
    S.card = 12 :=
sorry

end max_composite_numbers_with_gcd_one_l362_362763


namespace rectangle_breadth_decrease_l362_362120

/-- 
If a rectangle has a length of 140 cm and a width of 40 cm, and the length 
is increased by 30%, the percentage decrease in width needed to maintain 
the same area is approximately 23.08%. 
-/
theorem rectangle_breadth_decrease
  (L W : ℕ) 
  (Area : ℕ := L * W)
  (Increased_Length : ℕ := L + (L * 30 / 100))
  (New_Width : ℤ := Area / Increased_Length) :
  L = 140 ∧ W = 40 →
  (1:ℤ) * Area = ↑New_Width * ↑Increased_Length →
  (Approximate_Percentage_Decrease : ℤ := (((W : ℤ) - New_Width) * 100) / (W : ℤ)) :
  Approximate_Percentage_Decrease ≈ 23.08 := sorry

end rectangle_breadth_decrease_l362_362120


namespace theorem_3_squeeze_theorem_l362_362937

open Filter

-- Theorem 3
theorem theorem_3 (v : ℕ → ℝ) (hv : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |v n| ≤ ε)
                  (u : ℕ → ℝ) (n0 : ℕ) (hu : ∀ n ≥ n0, |u n| ≤ |v n|) :
  tendsto u atTop (nhds 0) := sorry

-- Squeeze Theorem
theorem squeeze_theorem (u v w : ℕ → ℝ) (ℓ : ℝ)
                        (hu : tendsto u atTop (nhds ℓ))
                        (hw : tendsto w atTop (nhds ℓ))
                        (n0 : ℕ) (hv : ∀ n ≥ n0, u n ≤ v n ∧ v n ≤ w n) :
  tendsto v atTop (nhds ℓ) := sorry

end theorem_3_squeeze_theorem_l362_362937


namespace divide_decimals_l362_362577

theorem divide_decimals : (0.24 / 0.006) = 40 := by
  sorry

end divide_decimals_l362_362577


namespace dishes_served_total_l362_362976

-- Definitions based on conditions
def women_per_table := 7
def men_per_table := 2
def courses_per_woman := 3
def courses_per_man := 4
def tables := 7
def shared_courses_women := 1
def shared_courses_men := 2

-- Proposition stating the total number of dishes served by the waiter
theorem dishes_served_total :
  let total_courses_per_table := (women_per_table * courses_per_woman - shared_courses_women) + 
                                (men_per_table * courses_per_man - shared_courses_men)
  in total_courses_per_table * tables = 182 :=
by
  sorry

end dishes_served_total_l362_362976


namespace trader_cloth_sale_l362_362176

theorem trader_cloth_sale (total_SP : ℕ) (profit_per_meter : ℕ) (cost_per_meter : ℕ) (SP_per_meter : ℕ)
  (h1 : total_SP = 8400) (h2 : profit_per_meter = 12) (h3 : cost_per_meter = 128) (h4 : SP_per_meter = cost_per_meter + profit_per_meter) :
  ∃ (x : ℕ), SP_per_meter * x = total_SP ∧ x = 60 :=
by
  -- We will skip the proof using sorry
  sorry

end trader_cloth_sale_l362_362176


namespace triangle_sides_condition_triangle_perimeter_l362_362722

theorem triangle_sides_condition (a b c : ℝ) (A B C : ℝ) 
  (h1 : sin C * sin (A - B) = sin B * sin (C - A)) : 2 * a^2 = b^2 + c^2 :=
sorry

theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ) 
  (h2 : a = 5) (h3 : cos A = 25 / 31) : a + b + c = 14 :=
sorry

end triangle_sides_condition_triangle_perimeter_l362_362722


namespace probability_of_yellow_ball_is_correct_l362_362474

-- Defining the conditions
def red_balls : ℕ := 2
def yellow_balls : ℕ := 5
def blue_balls : ℕ := 4

-- Define the total number of balls
def total_balls : ℕ := red_balls + yellow_balls + blue_balls

-- Define the probability of choosing a yellow ball
def probability_yellow_ball : ℚ := yellow_balls / total_balls

-- The theorem statement we need to prove
theorem probability_of_yellow_ball_is_correct :
  probability_yellow_ball = 5 / 11 :=
sorry

end probability_of_yellow_ball_is_correct_l362_362474


namespace common_ratio_of_geometric_sequence_l362_362527

theorem common_ratio_of_geometric_sequence : 
  ∃ r : ℝ, (32 * r = -48) ∧ (32 * r^2 = 72) ∧ (32 * r^3 = -108) ∧ (32 * r^4 = 162) ∧ (r = -3 / 2) :=
by
  use -3 / 2
  constructor;
  { sorry };
  constructor;
  { sorry };
  constructor;
  { sorry };
  constructor;
  { sorry };

end common_ratio_of_geometric_sequence_l362_362527


namespace james_payment_correct_l362_362375

-- Definitions from conditions
def cost_steak_eggs : ℝ := 16
def cost_chicken_fried_steak : ℝ := 14
def total_cost := cost_steak_eggs + cost_chicken_fried_steak
def half_share := total_cost / 2
def tip := total_cost * 0.2
def james_total_payment := half_share + tip

-- Statement to be proven
theorem james_payment_correct : james_total_payment = 21 :=
by
  sorry

end james_payment_correct_l362_362375


namespace quadratic_function_formula_quadratic_function_range_quadratic_function_above_line_l362_362886

-- Problem: Find the explicit formula of f(x)
theorem quadratic_function_formula {f : ℝ → ℝ}
  (h₁ : ∀ x, f (x + 1) - f x = 2 * x)
  (h₂ : f 0 = 1) :
  f = (λ x, x^2 - x + 1) :=
sorry

-- Problem: Find the range of f(x) in the interval [-1, 1]
theorem quadratic_function_range :
  set.range (λ x : Icc (-1 : ℝ) 1, x^2 - x + 1) = set.Icc (3 / 4) 3 :=
sorry

-- Problem: Determine the range of the real number m
theorem quadratic_function_above_line {m : ℝ} :
  (∀ x ∈ set.Icc (-1 : ℝ) 1, x^2 - x + 1 > 2 * x + m) ↔ (m < -1) :=
sorry

end quadratic_function_formula_quadratic_function_range_quadratic_function_above_line_l362_362886


namespace subcommittees_count_l362_362315

theorem subcommittees_count 
  (n : ℕ) (k : ℕ) (hn : n = 7) (hk : k = 3) : 
  (nat.choose n k) = 35 := by 
  have h1 : 7 = 7 := rfl
  have h2 : 3 = 3 := rfl
  sorry

end subcommittees_count_l362_362315


namespace four_letter_words_with_A_l362_362307

theorem four_letter_words_with_A :
  let letters := ['A', 'B', 'C', 'D', 'E']
  in let total_4_letter_words := 5^4
  in let words_without_A := 4^4
  in total_4_letter_words - words_without_A = 369 := by
  sorry

end four_letter_words_with_A_l362_362307


namespace log_relationship_l362_362331

theorem log_relationship (a b c : ℝ) 
  (ha : a = Real.log 3 / Real.log 2) 
  (hb : b = Real.log 4 / Real.log 3) 
  (hc : c = Real.log 5 / Real.log 4) : 
  c < b ∧ b < a :=
by 
  sorry

end log_relationship_l362_362331


namespace count_valid_sequences_22_l362_362859

def Transformation := 
  | L : Transformation
  | R : Transformation
  | H : Transformation
  | V : Transformation
  | F : Transformation

def vertices := [(1,1), (-1,1), (-1,-1), (1,-1)]

def apply_transformation: Transformation × (ℝ × ℝ) → (ℝ × ℝ)
  | (Transformation.L, (x,y)) := (-y, x)
  | (Transformation.R, (x,y)) := (y, -x)
  | (Transformation.H, (x,y)) := (x, -y)
  | (Transformation.V, (x,y)) := (-x, y)
  | (Transformation.F, (x,y)) := (y, x)

def identity_transformation (seq : List Transformation) : Bool :=
  List.foldl (λ p t, apply_transformation (t, p)) vertices seq == vertices

def count_sequences (seq_length : ℕ) : ℕ :=
  if seq_length == 22
  then 5^21
  else 0

theorem count_valid_sequences_22 :
  count_sequences 22 = 5^21 :=
by sorry

end count_valid_sequences_22_l362_362859


namespace max_composite_numbers_with_gcd_one_l362_362766

theorem max_composite_numbers_with_gcd_one : 
  ∃ (S : Finset ℕ), 
    (∀ x ∈ S, Nat.isComposite x) ∧ 
    (∀ x ∈ S, x < 1500) ∧ 
    (∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y → Nat.gcd x y = 1) ∧
    S.card = 12 :=
sorry

end max_composite_numbers_with_gcd_one_l362_362766


namespace SufficientCondition_l362_362504

theorem SufficientCondition :
  ∀ x y z : ℤ, x = z ∧ y = x - 1 → x * (x - y) + y * (y - z) + z * (z - x) = 2 :=
by
  intros x y z h
  cases h with
  | intro h1 h2 =>
  sorry

end SufficientCondition_l362_362504


namespace ratio_of_segments_l362_362678

theorem ratio_of_segments (a b c r s : ℝ) (k : ℝ)
  (h₁ : a = 2 * k) 
  (h₂ : b = 5 * k)
  (h₃ : c = k * real.sqrt 29)
  (h₄ : r = (2 * k) ^ 2 / (k * real.sqrt 29))
  (h₅ : s = (5 * k) ^ 2 / (k * real.sqrt 29)) :
  r / s = 4 / 25 :=
begin
  sorry
end

end ratio_of_segments_l362_362678


namespace polygon_has_five_sides_l362_362397

theorem polygon_has_five_sides (a : ℝ) (ha : 0 < a) :
  let T := {p : ℝ × ℝ | 
             let x := p.1, y := p.2 in
             (a / 3) ≤ x ∧ x ≤ (5 * a / 2) ∧
             (a / 3) ≤ y ∧ y ≤ (5 * a / 2) ∧
             (x + y) ≥ (3 * a / 2) ∧
             (x + 2 * a) ≥ (2 * y) ∧
             (2 * y + 2 * a) ≥ (3 * x)} in
  ∃ sides : ℕ, sides = 5 ∧ 
  polygon_has_sides T sides := by
  sorry

end polygon_has_five_sides_l362_362397


namespace major_axis_length_l362_362158

theorem major_axis_length 
  (r : ℝ) (h1 : r = 2) 
  (h2 : ∀ (minor major : ℝ), minor = 2 * r → major = minor * 1.3):
  ∃ (major : ℝ), major = 5.2 := 
by 
  use 5.2
  sorry

end major_axis_length_l362_362158


namespace sqrt_expression_simplified_l362_362006

theorem sqrt_expression_simplified :
  sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  sorry

end sqrt_expression_simplified_l362_362006


namespace max_oleg_composite_numbers_l362_362757

/-- Oleg's list of composite numbers less than 1500 and pairwise coprime. --/
def oleg_composite_numbers (numbers : List ℕ) : Prop :=
  ∀ n ∈ numbers, Nat.isComposite n ∧ n < 1500 ∧ (∀ m ∈ numbers, n ≠ m → Nat.gcd n m = 1)

/-- Maximum number of such numbers that Oleg could write. --/
theorem max_oleg_composite_numbers : ∃ numbers : List ℕ, oleg_composite_numbers numbers ∧ numbers.length = 12 := 
sorry

end max_oleg_composite_numbers_l362_362757


namespace sqrt_expression_simplified_l362_362017

theorem sqrt_expression_simplified :
  sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  sorry

end sqrt_expression_simplified_l362_362017


namespace no_convex_1000gon_with_whole_number_angles_l362_362933

-- Predicate defining convex n-gon
def is_convex_ngon (n : ℕ) (angles : fin n → ℕ) : Prop :=
  ∀ i, angles i > 0 ∧ angles i < 180

-- Predicate for the existence of a convex 1000-gon where all angles are whole numbers
def exists_convex_1000gon_with_whole_number_angles : Prop :=
  ∃ (angles : fin 1000 → ℕ), is_convex_ngon 1000 angles

theorem no_convex_1000gon_with_whole_number_angles :
  ¬ exists_convex_1000gon_with_whole_number_angles :=
by
  sorry

end no_convex_1000gon_with_whole_number_angles_l362_362933


namespace generalized_barbier_theorem_l362_362133

noncomputable def convex_curve (K : set Point) (l : ℝ) : Prop :=
  ∀ (L : set Point), is_rectangle_around K L -> perimeter L = 4*l

theorem generalized_barbier_theorem (K : set Point) (h : convex_curve K l) :
  length K = π * l ∧ area K = l^2 * (π / 2 - 1) :=
sorry

-- Definitions for is_rectangle_around, perimeter, length, and area need to be declared 
-- appropriately in the context or library.

end generalized_barbier_theorem_l362_362133


namespace marcus_savings_l362_362746

theorem marcus_savings
  (running_shoes_price : ℝ)
  (running_shoes_discount : ℝ)
  (cashback : ℝ)
  (running_shoes_tax_rate : ℝ)
  (athletic_socks_price : ℝ)
  (athletic_socks_tax_rate : ℝ)
  (bogo : ℝ)
  (performance_tshirt_price : ℝ)
  (performance_tshirt_discount : ℝ)
  (performance_tshirt_tax_rate : ℝ)
  (total_budget : ℝ)
  (running_shoes_final_price : ℝ)
  (athletic_socks_final_price : ℝ)
  (performance_tshirt_final_price : ℝ) :
  running_shoes_price = 120 →
  running_shoes_discount = 30 / 100 →
  cashback = 10 →
  running_shoes_tax_rate = 8 / 100 →
  athletic_socks_price = 25 →
  athletic_socks_tax_rate = 6 / 100 →
  bogo = 2 →
  performance_tshirt_price = 55 →
  performance_tshirt_discount = 10 / 100 →
  performance_tshirt_tax_rate = 7 / 100 →
  total_budget = 250 →
  running_shoes_final_price = (running_shoes_price * (1 - running_shoes_discount) - cashback) * (1 + running_shoes_tax_rate) →
  athletic_socks_final_price = (athletic_socks_price * bogo) * (1 + athletic_socks_tax_rate) / bogo →
  performance_tshirt_final_price = (performance_tshirt_price * (1 - performance_tshirt_discount)) * (1 + performance_tshirt_tax_rate) →
  total_budget - (running_shoes_final_price + athletic_socks_final_price + performance_tshirt_final_price) = 103.86 :=
sorry

end marcus_savings_l362_362746


namespace simplify_expression_l362_362840

theorem simplify_expression (a b : ℚ) (ha : a = -1) (hb : b = 1/4) : 
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_expression_l362_362840


namespace dot_product_bounds_l362_362631

theorem dot_product_bounds
  (A : ℝ × ℝ)
  (hA : A.1 ^ 2 + (A.2 - 1) ^ 2 = 1) :
  -2 ≤ A.1 * 2 ∧ A.1 * 2 ≤ 2 := 
sorry

end dot_product_bounds_l362_362631


namespace complex_conjugate_of_z_l362_362868

def complex_z : ℂ := (3 - complex.i ^ 2015) / (1 + complex.i)

theorem complex_conjugate_of_z : complex.conj complex_z = 2 + complex.i :=
by
  sorry

end complex_conjugate_of_z_l362_362868


namespace least_odd_prime_factor_of_2023_8_plus_1_l362_362596

-- Define the example integers and an assumption for modular arithmetic
def n : ℕ := 2023
def p : ℕ := 97

-- Conditions and the theorem statement
theorem least_odd_prime_factor_of_2023_8_plus_1 :
  n ^ 8 ≡ -1 [MOD p] →
  ∀ q, prime q → q ∣ (n ^ 8 + 1) → q ≥ p :=
by
  sorry

end least_odd_prime_factor_of_2023_8_plus_1_l362_362596


namespace tire_diameter_correct_l362_362141

variable (r : ℝ) (d_m : ℝ) (m : ℝ) (f : ℝ)

noncomputable def car_tire_diameter : ℝ :=
  let D := d_m * m * f in
  let π := Real.pi in
  D / (r * π)

theorem tire_diameter_correct :
  r = 672.1628045157456 →
  d_m = 1 / 2 →
  m = 5280 →
  f = 12 →
  car_tire_diameter r d_m m f ≈ 15 := by
  intros
  sorry

end tire_diameter_correct_l362_362141


namespace arithmetic_sequence_condition_l362_362411

theorem arithmetic_sequence_condition (x : ℝ) (h1 : 0 ≤ x) (h2 : ∉ ℤ) :
  (∃ a A b : ℝ, A * 2 = a + b ∧ x = A ∧ a = fractionalPart x ∧ b = x.toInt) → (2 * x = fractionalPart x + x.toInt) :=
by
  sorry

end arithmetic_sequence_condition_l362_362411


namespace range_of_a_l362_362289

def f (a b x : ℝ) : ℝ := Real.log x + a * x^2 + b * x

def fp (a b x : ℝ) : ℝ := (2 * a * x^2 - (2 * a + 1) * x + 1) / x

theorem range_of_a (a b : ℝ)
  (h_deriv : ∀ x > 0, fp a b x = f' a b x)
  (h_max : fp a b 1 = 0) :
  x = 1 local maximum f a b ↔ a ∈ Iio (1 / 2) :=
sorry

end range_of_a_l362_362289


namespace sqrt_49_times_sqrt_25_eq_7sqrt5_l362_362067

theorem sqrt_49_times_sqrt_25_eq_7sqrt5 :
  (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  have h1 : Real.sqrt 25 = 5 := by sorry
  have h2 : 49 * 5 = 245 := by sorry
  have h3 : Real.sqrt 245 = 7 * Real.sqrt 5 := by sorry
  sorry

end sqrt_49_times_sqrt_25_eq_7sqrt5_l362_362067


namespace find_real_number_a_l362_362277

theorem find_real_number_a (a : ℝ) : 
  complex.im (complex.div (1 - a * complex.I) (1 + complex.I)) = -1 → a = 1 :=
by
  sorry

end find_real_number_a_l362_362277


namespace instantaneous_velocity_at_2_l362_362873

def displacement (t : ℝ) : ℝ := 100 * t - 5 * t^2

noncomputable def instantaneous_velocity_at (s : ℝ → ℝ) (t : ℝ) : ℝ :=
  (deriv s) t

theorem instantaneous_velocity_at_2 : instantaneous_velocity_at displacement 2 = 80 :=
by
  sorry

end instantaneous_velocity_at_2_l362_362873


namespace problem1_part1_problem1_part2_l362_362685

theorem problem1_part1 (t p : ℝ) (ht : t ≠ 0) (hp : p > 0) :
  let M := (0, t)
  let P := (t^2 / (2 * p), t)
  let N := (t^2 / p, t)
  let H := (2 * t^2 / p, 2 * t)
in abs ((2 * t^2 / p) / (t^2 / p)) = 2 := by
  sorry

theorem problem1_part2 (t p : ℝ) (ht : t ≠ 0) (hp : p > 0) :
  let M := (0, t)
  let P := (t^2 / (2 * p), t)
  let N := (t^2 / p, t)
  let H := (2 * t^2 / p, 2 * t)
  let line_eq := λ y: ℝ, (p / (2 * t) * y + t)
in discriminant (line_eq 0) (line_eq t) -- implement the discriminant check here.
= 0 := by
  sorry

end problem1_part1_problem1_part2_l362_362685


namespace value_of_expression_l362_362830

variable (x y : ℝ)

theorem value_of_expression (h1 : x + y = 3) (h2 : x * y = 1) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = 849 := by sorry

end value_of_expression_l362_362830


namespace hyperbola_properties_parabola_equation_l362_362295

noncomputable theory

section
variables {x y : ℝ}

def hyperbola := 16 * x^2 - 9 * y^2 = 144

def real_axis_length := 6
def imaginary_axis_length := 8
def eccentricity := 5 / 3

theorem hyperbola_properties :
  (16 * x^2 - 9 * y^2 = 144) →
  (real_axis_length = 6) ∧ (imaginary_axis_length = 8) ∧ (eccentricity = 5 / 3) := sorry

def parabola (p : ℝ) := y^2 = -2 * p * x

theorem parabola_equation :
  (16 * x^2 - 9 * y^2 = 144) →
  (vertex_x : ℝ = 0) → 
  (vertex_y : ℝ = 0) → 
  (focus_x : ℝ = -3) →
  (focus_y : ℝ = 0) → 
  (parabola 6) := sorry

end

end hyperbola_properties_parabola_equation_l362_362295


namespace correct_differentiation_count_l362_362551

-- Define the functions as per conditions
noncomputable def f1 (x : ℝ) : ℝ := exp x + exp (-x)
noncomputable def f2 (x : ℝ) : ℝ := log x / log 2
noncomputable def f3 (x : ℝ) : ℝ := exp x
noncomputable def f4 (x : ℝ) : ℝ := 1 / log x
noncomputable def f5 (x : ℝ) : ℝ := x * exp x
def y : ℝ := log 2

-- Define the second derivatives
noncomputable def f1'' (x : ℝ) := exp x + exp (-x)
noncomputable def f2'' (x : ℝ) := 1 / (x * log 2)
noncomputable def f3'' (x : ℝ) := exp x
noncomputable def f4'' (x : ℝ) := x  -- Note: This is the condition, not the true second derivative
noncomputable def f5'' (x : ℝ) := exp x + 1
def y'' := 1 / 2  -- Condition given in the problem

-- Prove that there are exactly two correct differentiation operations
theorem correct_differentiation_count : 
  ((f1'' == f1'') + 
   (f2'' == (λ x, 1 / (x * log 2))) + 
   (f3'' == f3'') + 
   (f4'' == f4'') + 
   (f5'' == (λ x, exp x + 1)) + 
   (y'' == (0))) = 2 := 
by 
  sorry

end correct_differentiation_count_l362_362551


namespace last_digit_2_pow_2023_l362_362750

-- Definitions
def last_digit_cycle : List ℕ := [2, 4, 8, 6]

-- Theorem statement
theorem last_digit_2_pow_2023 : (2 ^ 2023) % 10 = 8 :=
by
  -- We will assume and use the properties mentioned in the solution steps.
  -- The proof process is skipped here with 'sorry'.
  sorry

end last_digit_2_pow_2023_l362_362750


namespace max_A_excircle_area_ratio_max_A_excircle_area_ratio_eq_l362_362392

noncomputable def A_excircle_area_ratio (α : Real) (s : Real) : Real :=
  0.5 * Real.sin α

theorem max_A_excircle_area_ratio (α : Real) (s : Real) : (A_excircle_area_ratio α s) ≤ 0.5 :=
by
  sorry

theorem max_A_excircle_area_ratio_eq (s : Real) : 
  (A_excircle_area_ratio (Real.pi / 2) s) = 0.5 :=
by
  sorry

end max_A_excircle_area_ratio_max_A_excircle_area_ratio_eq_l362_362392


namespace circles_seen_at_equal_angles_from_third_vertex_l362_362682

variables {α β γ : ℝ} (r rₐ r_b r_c : ℝ) (A B C : Type) 
variables [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]

theorem circles_seen_at_equal_angles_from_third_vertex 
  (hA : α = β) 
  (hB : β = γ) 
  (hC : γ = α): α = β :=
by
  sorry

end circles_seen_at_equal_angles_from_third_vertex_l362_362682


namespace lara_likes_divisible_by_3_endings_l362_362416

theorem lara_likes_divisible_by_3_endings :
  { (A B : ℕ) // A ∈ finset.range 10 ∧ B ∈ finset.range 10 ∧ (10 * A + B) % 3 = 0 }.card = 34 :=
sorry

end lara_likes_divisible_by_3_endings_l362_362416


namespace equilateral_triangle_of_equal_inradii_l362_362459

theorem equilateral_triangle_of_equal_inradii
  (ABC : Triangle)
  (h_medians_divide : ∀ Δ ∈ (ABC.medians_divide), area Δ = (1/6) * area ABC)
  (h_four_inradii_equal : ∃ (Δ₁ Δ₂ Δ₃ Δ₄ : Triangle), (Δ₁ ∈ (ABC.medians_divide) ∧ Δ₂ ∈ (ABC.medians_divide) ∧ Δ₃ ∈ (ABC.medians_divide) ∧ Δ₄ ∈ (ABC.medians_divide)) ∧ (inradius Δ₁ = inradius Δ₂ ∧ inradius Δ₂ = inradius Δ₃ ∧ inradius Δ₃ = inradius Δ₄)) :
  is_equilateral ABC :=
sorry

end equilateral_triangle_of_equal_inradii_l362_362459


namespace find_d_plus_f_l362_362479

noncomputable def a : ℂ := sorry
noncomputable def c : ℂ := sorry
noncomputable def e : ℂ := -2 * a - c
noncomputable def d : ℝ := sorry
noncomputable def f : ℝ := sorry

theorem find_d_plus_f (a c e : ℂ) (d f : ℝ) (h₁ : e = -2 * a - c) (h₂ : a.im + d + f = 4) (h₃ : a.re + c.re + e.re = 0) (h₄ : 2 + d + f = 4) : d + f = 2 :=
by
  -- proof goes here
  sorry

end find_d_plus_f_l362_362479


namespace probability_inequality_l362_362252

noncomputable def X_i (p : ℝ) : ℕ → ℤ
| i := if rand ≤ p then 1
       else if rand ≤ 2 * p then -1
       else 0

def P (b : ℤ) (a : List ℤ) (p : ℝ) : ℝ :=
-- Probability definition (omitted for brevity)

theorem probability_inequality (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1/4) (n : ℕ) (b : ℤ) (a : List ℤ) :
  P 0 a p ≥ P b a p :=
sorry

end probability_inequality_l362_362252


namespace correlation_coefficient_linear_regression_prediction_l362_362745

theorem correlation_coefficient
  (n : ℕ)
  (xs ys : Fin n → ℝ)
  (mean_x mean_y : ℝ)
  (sum_squares_x : ℝ)
  (sum_squares_y : ℝ)
  (sum_xy_cov : ℝ)
  (sqrt_sum_squares_y : ℝ)
  (approx_sqrt_sum_squares_y : sqrt 441000 ≈ 664)
  (mean_x : mean_x = 3)
  (mean_y : mean_y = 590)
  (sum_squares_x : sum_squares_x = 10)
  (sum_squares_y : sum_squares_y = 176400)
  (sum_xy_cov : sum_xy_cov = 1320) :
  | (sum_xy_cov / (sqrt sum_squares_x * sqrt sum_squares_y)) - 0.99 | < 0.01 :=
by
  sorry

theorem linear_regression_prediction
  (x : ℝ)
  (mean_x mean_y : ℝ)
  (sum_squares_x : ℝ)
  (sum_xy_cov : ℝ)
  (b : ℝ)
  (a : ℝ)
  (approx_b : b = sum_xy_cov / sum_squares_x)
  (approx_a : a = mean_y - (sum_xy_cov / sum_squares_x * mean_x))
  (mean_x : mean_x = 3)
  (mean_y : mean_y = 590)
  (sum_squares_x : sum_squares_x = 10)
  (sum_xy_cov : sum_xy_cov = 1320)
  (predicted_y : ℝ)
  (predicted_y_is : predicted_y = b * 6 + a) :
  predicted_y = 986 :=
by
  sorry

end correlation_coefficient_linear_regression_prediction_l362_362745


namespace maximize_probability_l362_362116

open Nat Real

noncomputable def P (n : ℕ) : ℝ :=
  (n / 6) * (5 / 6) ^ (n - 1)

theorem maximize_probability (n : ℕ) :
  (P n ≤ P 5) ∨ (P n ≤ P 6) :=
sorry

end maximize_probability_l362_362116


namespace maximum_OA_plus_OB_l362_362688

noncomputable def C (a : ℝ) (θ : ℝ) : ℝ := 2 * a * Real.cos θ
noncomputable def l (θ : ℝ) : ℝ := (3/2) / Real.cos (θ - π / 3)
noncomputable def OA (a : ℝ) (θ : ℝ) : ℝ := 2 * Real.cos θ
noncomputable def OB (a : ℝ) (θ : ℝ) : ℝ := 2 * Real.cos (θ + π / 3)

-- main theorem
theorem maximum_OA_plus_OB {a : ℝ} (h₀ : 0 < a) 
(h₁ : ∀ θ, C a θ = l θ) 
(h₂ : ∀ A B, C a (angle A) = 2 * Real.cos (angle A) /\ C a (angle B) = 2 * Real.cos (angle B))
(h₃ : ∀ θ, OA a θ + OB a θ = 2 * Real.sqrt 3 * Real.cos (θ + π / 6)) :
  ∃ θ, (OA a θ + OB a θ) = 2 * Real.sqrt 3 :=
by
  sorry

end maximum_OA_plus_OB_l362_362688


namespace simplified_expression_value_l362_362846

theorem simplified_expression_value (a b : ℝ) (ha : a = -1) (hb : b = 1 / 4) :
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by
  sorry

end simplified_expression_value_l362_362846


namespace derivative_log_base2_l362_362507

theorem derivative_log_base2 (x : ℝ) (hx : 0 < x) : 
  deriv (fun x => real.log x / real.log 2) x = 1 / (x * real.log 2) :=
by 
  sorry

end derivative_log_base2_l362_362507


namespace calculate_total_earnings_l362_362140

theorem calculate_total_earnings :
  let num_floors := 10
  let rooms_per_floor := 20
  let hours_per_room := 8
  let earnings_per_hour := 20
  let total_rooms := num_floors * rooms_per_floor
  let total_hours := total_rooms * hours_per_room
  let total_earnings := total_hours * earnings_per_hour
  total_earnings = 32000 := by sorry

end calculate_total_earnings_l362_362140


namespace num_correct_propositions_is_zero_l362_362185

-- Definitions corresponding to the conditions in the problem
def g : ℝ → ℝ := sorry  -- We define g as a real function, actual contents are 'sorry' because we don't have specifics

-- Hypotheses based on conditions
def condition1 (a : ℝ) := g(a) < g(0) = 0
def condition3 (a : ℝ) := a ∈ set.Ici (-1)

-- Main statement: The number of correct propositions is 0
theorem num_correct_propositions_is_zero : 
  (forall a : ℝ, (condition1 a → isosceles_triangle (g a)) ∧ 
  (a → right_triangle (g a)) ∧ 
  (condition3 a → acute_triangle (g a))) → 
  (0) := sorry

end num_correct_propositions_is_zero_l362_362185


namespace simplified_expression_value_l362_362842

theorem simplified_expression_value (a b : ℝ) (ha : a = -1) (hb : b = 1 / 4) :
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by
  sorry

end simplified_expression_value_l362_362842


namespace probability_not_purple_l362_362338

/-- Given the odds for pulling a purple marble are 5:6, the probability of not pulling 
a purple marble out of the bag is 6/11. -/
theorem probability_not_purple (purple_odds : ℕ) (non_purple_odds : ℕ) :
  purple_odds = 5 → non_purple_odds = 6 → 
  (let total_outcomes := purple_odds + non_purple_odds in
  let favorable_outcomes := non_purple_odds in
  favorable_outcomes / total_outcomes = 6 / 11) :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end probability_not_purple_l362_362338


namespace hyperbola_asymptotes_l362_362293

theorem hyperbola_asymptotes (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : (∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 → eccentricity = 2))
  (h4 : eccentricity = 2) : 
  (∀ x : ℝ, y = (√3) * x ∨ y = - (√3) * x) :=
begin
  sorry
end

end hyperbola_asymptotes_l362_362293


namespace sqrt_mul_sqrt_l362_362045

theorem sqrt_mul_sqrt (a b c : ℝ) (hac : a = 49) (hbc : b = sqrt 25) (hc : c = 7 * sqrt 5) : sqrt (a * b) = c := 
by
  sorry

end sqrt_mul_sqrt_l362_362045


namespace sqrt_D_irrational_l362_362394

variable (k : ℤ)

def a := 3 * k
def b := 3 * k + 3
def c := a k + b k
def D := a k * a k + b k * b k + c k * c k

theorem sqrt_D_irrational : ¬ ∃ (r : ℚ), r * r = D k := 
by sorry

end sqrt_D_irrational_l362_362394


namespace remainder_determined_l362_362824

theorem remainder_determined (p a b : ℤ) (h₀: Nat.Prime (Int.natAbs p)) (h₁ : ¬ (p ∣ a)) (h₂ : ¬ (p ∣ b)) :
  ∃ (r : ℤ), (r ≡ a [ZMOD p]) ∧ (r ≡ b [ZMOD p]) ∧ (r ≡ (a * b) [ZMOD p]) →
  (a ≡ r [ZMOD p]) := sorry

end remainder_determined_l362_362824


namespace cell_survival_after_6_hours_l362_362949

def cell_sequence (a : ℕ → ℕ) : Prop :=
  (a 0 = 2) ∧ (∀ n, a (n + 1) = 2 * a n - 1)

theorem cell_survival_after_6_hours :
  ∃ a : ℕ → ℕ, cell_sequence a ∧ a 6 = 65 :=
by
  sorry

end cell_survival_after_6_hours_l362_362949


namespace findC_l362_362987

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem findC (k C : ℝ) (eq_roots : ∀ k = 1, discriminant (2*k) (4*k) C = 0) : C = 2 :=
by 
  sorry

end findC_l362_362987


namespace fraction_spent_on_clothes_l362_362968

-- Define initial conditions
def M : ℝ := 1499.9999999999998
def remaining_money_after_spending (f : ℝ) : ℝ := M * (1 - f) * (4 / 5) * (3 / 4)
def final_amount : ℝ := 600

-- The statement to prove
theorem fraction_spent_on_clothes (f : ℝ) (h : remaining_money_after_spending f = final_amount) : f = 1 / 3 :=
by
  sorry

end fraction_spent_on_clothes_l362_362968


namespace part1_part2_l362_362715

-- Define the conditions
def triangle_conditions (a b c : ℝ) (A B C : ℝ) : Prop :=
  sin C * sin (A - B) = sin B * sin (C - A) 

-- Define the conclusion for part (1)
def proof_part1 (a b c : ℝ) (A B C : ℝ) (h : triangle_conditions a b c A B C) : Prop :=
  2 * a ^ 2 = b ^ 2 + c ^ 2

-- Define the conditions for part (2)
def triangle_conditions_part2 (a b c A : ℝ) : Prop :=
  a = 5 ∧ cos A = 25 / 31 

-- Define the conclusion for part (2)
def proof_part2 (a b c A : ℝ) (h : triangle_conditions_part2 a b c A) : Prop :=
  a + b + c = 14

-- The Lean statements for the complete problem
theorem part1 (a b c A B C : ℝ) 
  (h : triangle_conditions a b c A B C) : 
  proof_part1 a b c A B C h := 
sorry

theorem part2 (a b c A : ℝ) 
  (h : triangle_conditions_part2 a b c A) : 
  proof_part2 a b c A h := 
sorry

end part1_part2_l362_362715


namespace prince_wish_fulfilled_l362_362898

theorem prince_wish_fulfilled
  (k : ℕ)
  (k_gt_1 : 1 < k)
  (k_lt_13 : k < 13)
  (city : Fin 13 → Fin k) 
  (initial_goblets : Fin k → Fin 13)
  (is_gold : Fin 13 → Bool) :
  ∃ i j : Fin 13, i ≠ j ∧ city i = city j ∧ is_gold i = true ∧ is_gold j = true := 
sorry

end prince_wish_fulfilled_l362_362898


namespace sqrt_49_times_sqrt25_eq_7_sqrt5_l362_362033

-- Definitions based on conditions
def sqrt_25 := Real.sqrt 25
def sqrt_49 := Real.sqrt 49

theorem sqrt_49_times_sqrt25_eq_7_sqrt5 :
  Real.sqrt (49 * sqrt_25) = 7 * Real.sqrt 5 := by
sorry

end sqrt_49_times_sqrt25_eq_7_sqrt5_l362_362033


namespace ratio_of_areas_trapezoid_triangle_l362_362370

-- Define the problem
theorem ratio_of_areas_trapezoid_triangle 
  (AHI_equilateral : ∀ (A H I : ℝ), equilateral_triangle A H I)
  (parallel_BC_DE_FG_HI : ∀ (B C D E F G H I : ℝ), parallel B C H I ∧ parallel D E H I ∧ parallel F G H I)
  (AB_BD_DF_FH_equal : ∀ (A B D F H : ℝ), A ≠ B ∧ B ≠ D ∧ D ≠ F ∧ F ≠ H → AB = BD ∧ BD = DF ∧ DF = FH)
  (F_on_AH_half : ∀ (A F H : ℝ), F ∈ segment A H ∧ AF = (1 / 2) * AH) :
  (area (trapezoid F G I H) / area (triangle A H I)) = 3 / 4 :=
by
  sorry

end ratio_of_areas_trapezoid_triangle_l362_362370


namespace sqrt_49_times_sqrt_25_eq_7sqrt5_l362_362068

theorem sqrt_49_times_sqrt_25_eq_7sqrt5 :
  (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  have h1 : Real.sqrt 25 = 5 := by sorry
  have h2 : 49 * 5 = 245 := by sorry
  have h3 : Real.sqrt 245 = 7 * Real.sqrt 5 := by sorry
  sorry

end sqrt_49_times_sqrt_25_eq_7sqrt5_l362_362068


namespace max_composite_numbers_l362_362779

theorem max_composite_numbers (s : set ℕ) (hs : ∀ n ∈ s, n < 1500 ∧ ∃ p : ℕ, prime p ∧ p ∣ n) (hs_gcd : ∀ x y ∈ s, x ≠ y → Nat.gcd x y = 1) :
  s.card ≤ 12 := 
by sorry

end max_composite_numbers_l362_362779


namespace sqrt_product_l362_362100

theorem sqrt_product (a b : ℝ) (h1 : a = 49) (h2 : b = 25) : sqrt (a * sqrt b) = 7 * sqrt 5 :=
by sorry

end sqrt_product_l362_362100


namespace minimum_triangle_area_l362_362869

theorem minimum_triangle_area (x y : ℝ) (A B C : ℝ × ℝ) (a1 a2 : ℝ) (Q : ℝ × ℝ) :
  C = (x, y) ∧
  Q = (0, 1) ∧
  A = (a1, 0) ∧
  B = (a2, 0) ∧
  x^2 + y^2 ≤ 8 + 2y ∧
  y ≥ 3 ∧
  (sqrt ((0 - a1) ^ 2 + (1 - 0) ^ 2) = 1 ∧ sqrt ((0 - a2) ^ 2 + (1 - 0) ^ 2) = 1) 
  →
  let area := abs ((a1 - a2) * y) / 2 in 
  area = 6 * sqrt 2 :=
begin
  sorry
end

end minimum_triangle_area_l362_362869


namespace distance_between_red_lights_l362_362861

def light_position (n : ℕ) : ℕ := 10 * ((n - 1) / 3) + 3 + (n - 1) % 3

def light_distance (n m : ℕ) : ℕ := abs (light_position n - light_position m)

def feet_between_lights (dist_in_inches : ℕ) : ℕ := dist_in_inches / 12

theorem distance_between_red_lights (L : ∀ (n m : ℕ), light_distance n m * 8 / 12 = (m - n) * 8 / 12) :
  feet_between_lights (light_distance 4 18 * 8) = 12 :=
sorry

end distance_between_red_lights_l362_362861


namespace determinant_formula_l362_362237

open Matrix BigOperators

variables {R : Type*} [CommRing R]

def mat : Matrix (Fin 3) (Fin 3) R :=
  !![ 
    a, y, y;
    y, a, y;
    y, y, a
  ]

theorem determinant_formula (a y : R) :
  det (mat a y) = a^3 - 2 * a * y^2 + 2 * y^3 :=
by sorry

end determinant_formula_l362_362237


namespace larger_number_l362_362489

theorem larger_number (x y : ℤ) (h1 : x + y = 47) (h2 : x - y = 7) : x = 27 :=
  sorry

end larger_number_l362_362489


namespace max_oleg_composite_numbers_l362_362754

/-- Oleg's list of composite numbers less than 1500 and pairwise coprime. --/
def oleg_composite_numbers (numbers : List ℕ) : Prop :=
  ∀ n ∈ numbers, Nat.isComposite n ∧ n < 1500 ∧ (∀ m ∈ numbers, n ≠ m → Nat.gcd n m = 1)

/-- Maximum number of such numbers that Oleg could write. --/
theorem max_oleg_composite_numbers : ∃ numbers : List ℕ, oleg_composite_numbers numbers ∧ numbers.length = 12 := 
sorry

end max_oleg_composite_numbers_l362_362754


namespace cuboid_height_l362_362671

-- Define the necessary constants
def width : ℕ := 30
def length : ℕ := 22
def sum_edges : ℕ := 224

-- Theorem stating the height of the cuboid
theorem cuboid_height (h : ℕ) : 4 * length + 4 * width + 4 * h = sum_edges → h = 4 := by
  sorry

end cuboid_height_l362_362671


namespace max_composite_numbers_l362_362807
open Nat

theorem max_composite_numbers : 
  ∃ X : Finset Nat, 
  (∀ x ∈ X, x < 1500 ∧ ¬Prime x) ∧ 
  (∀ x y ∈ X, x ≠ y → gcd x y = 1) ∧ 
  X.card = 12 := 
sorry

end max_composite_numbers_l362_362807


namespace hyperbola_eccentricity_proof_l362_362294

noncomputable def hyperbola_eccentricity_problem
  (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b)
  (Pf : ∃ P : ℝ × ℝ, (P.1 ^ 2 / a ^ 2 - P.2 ^ 2 / b ^ 2 = 1) ∧ 
                       (dist ⟨P.1, P.2⟩ ⟨-c, 0⟩ = a) ∧ 
                       (dist ⟨P.1, P.2⟩ ⟨c, 0⟩ = 3 * a)) : ℝ :=
let c := sqrt (a^2 + b^2) / 2 in
let e := c / a in
have : 2 * c * c = 10 * a * a, sorry,
(eccentricity (sqrt (1 + (b/a)^2) : ℝ)): ℝ :=
√10 / 2

theorem hyperbola_eccentricity_proof
  {a b : ℝ} (h_a_pos : 0 < a) (h_b_pos : 0 < b)
  (Pf : ∃ P : ℝ × ℝ, (P.1 ^ 2 / a ^ 2 - P.2 ^ 2 / b ^ 2 = 1) ∧ 
                       (dist ⟨P.1, P.2⟩ ⟨-c, 0⟩ = a) ∧ 
                       (dist ⟨P.1, P.2⟩ ⟨c, 0⟩ = 3 * a)) :
  h_eccentricity (c, a) =
  e := sorry

end hyperbola_eccentricity_proof_l362_362294


namespace number_of_real_solutions_l362_362583

open Real

theorem number_of_real_solutions :
  {x : ℝ | sqrt (9 - x) = x^3 * sqrt (9 - x)}.finite.card = 2 :=
by
  sorry

end number_of_real_solutions_l362_362583


namespace sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l362_362084

theorem sqrt_49_mul_sqrt_25_eq_7_sqrt_5 : (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l362_362084


namespace find_f_cosine_value_l362_362290

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  log a (sqrt (x ^ 2 + 1) + x) + 1 / (a ^ x - 1) + 1

theorem find_f_cosine_value (a α : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : f a (sin (π / 6 - α)) = 1 / 3) :
  f a (cos (α - 2 * π / 3)) = 2 / 3 :=
sorry

end find_f_cosine_value_l362_362290


namespace sum_of_100_gon_divisible_by_5_l362_362368

theorem sum_of_100_gon_divisible_by_5 (a : ℕ → ℤ) :
  let b n := (fin 100).val in
  let transform a n := a n - a (b (n+1)) in
  ∃ k, k = 5 ∧ (∑ i in finset.range 100, (transform^[k] a) i) % 5 = 0 :=
sorry

end sum_of_100_gon_divisible_by_5_l362_362368


namespace citizen_income_l362_362579

theorem citizen_income (total_tax : ℝ) (income : ℝ) :
  total_tax = 15000 →
  (income ≤ 20000 → total_tax = income * 0.10) ∧
  (20000 < income ∧ income ≤ 50000 → total_tax = (20000 * 0.10) + ((income - 20000) * 0.15)) ∧
  (50000 < income ∧ income ≤ 90000 → total_tax = (20000 * 0.10) + (30000 * 0.15) + ((income - 50000) * 0.20)) ∧
  (income > 90000 → total_tax = (20000 * 0.10) + (30000 * 0.15) + (40000 * 0.20) + ((income - 90000) * 0.25)) →
  income = 92000 :=
by
  sorry

end citizen_income_l362_362579


namespace inverse_sum_l362_362400

def f (x : ℝ) : ℝ := x^2 * abs x

theorem inverse_sum :
  let f_inv := (λ y, if y = 9 then 3 else if y = -27 then -3 else 0)
  in f_inv 9 + f_inv (-27) = 0 :=
by
  -- Insert proof here.
  sorry

end inverse_sum_l362_362400


namespace pizza_fraction_eaten_l362_362921

theorem pizza_fraction_eaten (
  initial_slices : ℕ := 16
) 
(bounds : Yves_siblings_eaten : ℕ := 9)
(
  siblings_eaten : ℕ := 4 slices
) :
  ∃ (pizza_eaten_fraction : ℤ): 
  (initial_slices = 9 * siblings_eaten)  :=
by {
    have siblings_eaten := 2*2 := !slices := 4 slices;

    admit,
  sorry
}

end pizza_fraction_eaten_l362_362921


namespace largest_four_digit_number_l362_362888

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem largest_four_digit_number := ∃ (n : ℕ), 
  (1000 ≤ n ∧ n < 10000) ∧ 
  (sum_of_digits n = 29) ∧ 
  (∀ i j : ℕ, i ≠ j → i ∈ n.digits → j ∈ n.digits → i ≠ j) ∧ 
  ∀ m : ℕ, (1000 ≤ m ∧ m < 10000 ∧ sum_of_digits m = 29 ∧ 
  (∀ i j : ℕ, i ≠ j → i ∈ m.digits → j ∈ m.digits → i ≠ j)) → n ≥ m :=
sorry

end largest_four_digit_number_l362_362888


namespace polyhedron_dissection_parallelepipeds_polyhedron_has_center_of_symmetry_l362_362227

structure ConvexPolyhedron (V : Type) :=
  (faces : set (set V))
  (center_of_symmetry : ∀ f ∈ faces, ∃ c : V, ∀ x ∈ f, c - x ∈ f)
  -- Additional convex property can be added here if needed

theorem polyhedron_dissection_parallelepipeds {V : Type} [add_comm_group V] [module ℝ V]
  (P : ConvexPolyhedron V) : 
  ∃ (S : set (set V)), (∀ s ∈ S, ∃ a b c : V, s = parallelepiped a b c)
  ∧ (⋃₀ S) = (⋃₀ P.faces) :=
  sorry

theorem polyhedron_has_center_of_symmetry {V : Type} [add_comm_group V] [module ℝ V]
  (P : ConvexPolyhedron V) : 
  ∃ c : V, ∀ f ∈ P.faces, ∀ x ∈ f, c - x ∈ (⋃₀ P.faces) :=
  sorry

end polyhedron_dissection_parallelepipeds_polyhedron_has_center_of_symmetry_l362_362227


namespace sqrt_mul_sqrt_l362_362104

theorem sqrt_mul_sqrt (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : sqrt (a * sqrt b) = sqrt a * sqrt (sqrt b) := by
  sorry

example : sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  have h1 : sqrt 25 = 5 := by 
    norm_num
  have h2 : sqrt (49 * 5) = sqrt 245 := by  
    unfold sqrt
  have h3 : sqrt 245 = 7 * sqrt 5 := by 
    norm_num
  rw [h1, h2, h3]
  norm_num

end sqrt_mul_sqrt_l362_362104


namespace find_k_l362_362335

noncomputable def is_perfect_square (k : ℝ) : Prop :=
  ∀ x : ℝ, ∃ a : ℝ, x^2 + 2*(k-1)*x + 64 = (x + a)^2

theorem find_k (k : ℝ) : is_perfect_square k ↔ (k = 9 ∨ k = -7) :=
sorry

end find_k_l362_362335


namespace equation_represents_3x_minus_7_equals_2x_plus_5_l362_362450

theorem equation_represents_3x_minus_7_equals_2x_plus_5 (x : ℝ) :
  (3 * x - 7 = 2 * x + 5) :=
sorry

end equation_represents_3x_minus_7_equals_2x_plus_5_l362_362450


namespace bicycle_total_spokes_l362_362189

open Nat

def front_wheel_spokes : Nat := 20

def back_wheel_spokes : Nat := 2 * front_wheel_spokes

def total_spokes : Nat := front_wheel_spokes + back_wheel_spokes

theorem bicycle_total_spokes : total_spokes = 60 := by
  have h1 : front_wheel_spokes = 20 := rfl
  have h2 : back_wheel_spokes = 2 * front_wheel_spokes := rfl
  have h3 : back_wheel_spokes = 40 := by rw [h1, mul_comm, mul_one]
  show total_spokes = 60 from calc
    total_spokes = front_wheel_spokes + back_wheel_spokes := rfl
    ... = 20 + 40 := by rw [h1, h3]
    ... = 60 := by rfl

end bicycle_total_spokes_l362_362189


namespace pizza_slice_volume_l362_362166

-- Define the parameters given in the conditions
def pizza_thickness : ℝ := 0.5
def pizza_diameter : ℝ := 16.0
def num_slices : ℝ := 16.0

-- Define the volume of one slice
theorem pizza_slice_volume : (π * (pizza_diameter / 2) ^ 2 * pizza_thickness / num_slices) = 2 * π := by
  sorry

end pizza_slice_volume_l362_362166


namespace probability_x_plus_y_lt_4_l362_362533

open Set

def square : Set (ℝ × ℝ) :=
  { p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3 }

theorem probability_x_plus_y_lt_4 :
  measure_theory.measure_space.volume (square ∩ { p | p.1 + p.2 < 4 }) / 
  measure_theory.measure_space.volume square = 7/9 := 
sorry

end probability_x_plus_y_lt_4_l362_362533


namespace max_composite_numbers_l362_362775

theorem max_composite_numbers (s : set ℕ) (hs : ∀ n ∈ s, n < 1500 ∧ ∃ p : ℕ, prime p ∧ p ∣ n) (hs_gcd : ∀ x y ∈ s, x ≠ y → Nat.gcd x y = 1) :
  s.card ≤ 12 := 
by sorry

end max_composite_numbers_l362_362775


namespace quad_diagonals_relation_l362_362330

-- Define the convex quadrilateral side lengths and diagonal lengths
variables (a b c d m n : ℝ)

-- Define the cosine of angle sum (A + C)
noncomputable def cos_sum_angle := Real.cos_angle_sum A C

-- Hypotheses for the problem
axiom convex_quad (h1 : convQuad ABCD)

-- Prove the given equality
theorem quad_diagonals_relation :
  m^2 * n^2 = a^2 * c^2 + b^2 * d^2 - 2 * a * b * c * d * cos_sum_angle :=
sorry

end quad_diagonals_relation_l362_362330


namespace original_group_size_l362_362948

theorem original_group_size (M : ℕ) (R : ℕ) :
  (M * R * 40 = (M - 5) * R * 50) → M = 25 :=
by
  sorry

end original_group_size_l362_362948


namespace statement_A_statement_D_l362_362509

variable (a b c d : ℝ)

-- Statement A: If ac² > bc², then a > b
theorem statement_A (h1 : a * c^2 > b * c^2) (h2 : c ≠ 0) : a > b := by
  sorry

-- Statement D: If a > b > 0, then a + 1/b > b + 1/a
theorem statement_D (h1 : a > b) (h2 : b > 0) : a + 1 / b > b + 1 / a := by
  sorry

end statement_A_statement_D_l362_362509


namespace sqrt_49_mul_sqrt_25_l362_362003

theorem sqrt_49_mul_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_l362_362003


namespace similar_triangles_impossible_l362_362906

/-- Definitions of the problem -/
variables (a b c d e f : ℝ)
variables (α β γ δ ε ϕ : ℝ)
variables (ABC DEF : Type)

/-- Property definitions -/
def similar_triangles (t1 t2 : Type) : Prop := 
(∀ (A1 A2 : ℛ) B1 B2 : ℛ, α = δ ∧ β = ε ∧ γ = ϕ)
∧ (a / d = b / e ∧ a / d = c / f)

theorem similar_triangles_impossible (h : similar_triangles ABC DEF)
  (h_angle1 : α ≠ δ) (h_angle2 : β ≠ ε) (h_angle3 : γ ≠ ϕ)
  (h_side1 : a / d ≠ b / e) (h_side2 : b / e ≠ c / f)
  (h_side3 : a / d ≠ c / f) : false :=
by sorry

end similar_triangles_impossible_l362_362906


namespace marble_arrangements_remainder_l362_362588

theorem marble_arrangements_remainder : 
  let b := 6
  let y := 17
  let total_marbles := b + y
  let requirement (arr : Fin total_marbles → Fin 2) := 
     let same_neighbors := (Finset.filter 
       (λ i, (arr i = arr (i + 1))) (Finset.range (total_marbles - 1))).card
     let diff_neighbors := (Finset.filter 
       (λ i, (arr i ≠ arr (i + 1))) (Finset.range (total_marbles - 1))).card
     same_neighbors = diff_neighbors
  in 
  (Finset.filter requirement (Finset.pi (Finset.range total_marbles) (λ _, ({0, 1} : Finset (Fin 2))))).card % 1000 = 376 := 
    sorry

end marble_arrangements_remainder_l362_362588


namespace even_composite_fraction_l362_362585

theorem even_composite_fraction : 
  ((4 * 6 * 8 * 10 * 12) : ℚ) / (14 * 16 * 18 * 20 * 22) = 1 / 42 :=
by 
  sorry

end even_composite_fraction_l362_362585


namespace sqrt_nested_l362_362057

theorem sqrt_nested : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_nested_l362_362057


namespace find_f2_l362_362642

theorem find_f2 (m : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x, f(x) = m + m / x) (h₂ : f(1) = 2) : f(2) = 3 / 2 :=
by
  sorry

end find_f2_l362_362642


namespace box_volume_expr_l362_362555

-- Define the dimensions of the metal sheet and the size of the square cut from each corner
variables {L W y : ℝ} (hL : L = 18) (hW : W = 12) (hy_pos : y > 0)

-- Define the length, width, and height of the box after cuts
def length := L - 2*y
def width := W - 2*y
def height := y

-- Define the volume of the box
def volume := length * width * height

-- Theorem stating the volume calculation
theorem box_volume_expr (hL : L = 18) (hW : W = 12) (hy_pos : y > 0) :
  volume = 4*y^3 - 60*y^2 + 216*y :=
by
  sorry

end box_volume_expr_l362_362555


namespace propositions_correct_l362_362260

def is_perpendicular (n : Line) (α : Plane) : Prop := sorry
def is_parallel (α β : Plane) : Prop := sorry
def are_skew_lines (n m: Line) : Prop := sorry
def is_subset (n : Line) (α : Plane) : Prop := sorry
def equidistant_points (α β : Plane) : Prop := sorry

theorem propositions_correct {m n : Line} {α β : Plane}
  (h₁ : is_perpendicular n α)
  (h₂ : is_perpendicular n β)
  (h₃ : equidistant_points α β)
  (h₄ : are_skew_lines n m)
  (h₅ : is_subset n α)
  (h₆ : is_parallel n β)
  (h₇ : is_subset m β)
  (h₈ : is_parallel m α):
  (1 + 0 + 1 = 2) := sorry

end propositions_correct_l362_362260


namespace max_composite_numbers_with_gcd_one_l362_362765

theorem max_composite_numbers_with_gcd_one : 
  ∃ (S : Finset ℕ), 
    (∀ x ∈ S, Nat.isComposite x) ∧ 
    (∀ x ∈ S, x < 1500) ∧ 
    (∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y → Nat.gcd x y = 1) ∧
    S.card = 12 :=
sorry

end max_composite_numbers_with_gcd_one_l362_362765


namespace correct_operation_is_B_l362_362919

theorem correct_operation_is_B (a : ℝ) :
  (a^2 + a^3 ≠ a^5) ∧ (a^2 * a^3 = a^5) ∧ (a^2 / a^3 ≠ a^5) ∧ ((a^2)^3 ≠ a^5) :=
by
  split
  -- a^2 + a^3 ≠ a^5
  · 
    intro h
    sorry

  split
  -- a^2 * a^3 = a^5
  · 
    ring_nf
    norm_cast
    
  split
  -- a^2 / a^3 ≠ a^5
  · 
    intro h
    sorry

  -- (a^2)^3 ≠ a^5
  · 
    intro h
    sorry

end correct_operation_is_B_l362_362919


namespace how_many_roses_cut_l362_362481

theorem how_many_roses_cut :
  ∀ (r_i r_f r_c : ℕ), r_i = 6 → r_f = 16 → r_c = r_f - r_i → r_c = 10 :=
by
  intros r_i r_f r_c hri hrf heq
  rw [hri, hrf] at heq
  exact heq

end how_many_roses_cut_l362_362481


namespace sqrt_nested_l362_362059

theorem sqrt_nested : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_nested_l362_362059


namespace max_composite_numbers_l362_362811
open Nat

theorem max_composite_numbers : 
  ∃ X : Finset Nat, 
  (∀ x ∈ X, x < 1500 ∧ ¬Prime x) ∧ 
  (∀ x y ∈ X, x ≠ y → gcd x y = 1) ∧ 
  X.card = 12 := 
sorry

end max_composite_numbers_l362_362811


namespace cards_added_l362_362895

theorem cards_added (initial_cards added_cards total_cards : ℕ) (h1 : initial_cards = 9) (h2 : total_cards = 13) :
  (total_cards - initial_cards = added_cards) → (added_cards = 4) :=
by
  intro h
  rw [h1, h2] at h
  exact h

end cards_added_l362_362895


namespace simplify_expression_l362_362838

theorem simplify_expression (a b : ℚ) (ha : a = -1) (hb : b = 1/4) : 
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_expression_l362_362838


namespace multiple_of_3_l362_362434

theorem multiple_of_3 (a b : ℤ) (h1 : ∃ m : ℤ, a = 3 * m) (h2 : ∃ n : ℤ, b = 9 * n) : ∃ k : ℤ, a + b = 3 * k :=
by
  sorry

end multiple_of_3_l362_362434


namespace reflection_line_sum_l362_362878

-- Prove that the sum of m and b is 10 given the reflection conditions

theorem reflection_line_sum
    (m b : ℚ)
    (H : ∀ (x y : ℚ), (2, 2) = (x, y) → (8, 6) = (2 * (5 - (3 / 2) * (2 - x)), 2 + m * (y - 2)) ∧ y = m * x + b) :
  m + b = 10 :=
sorry

end reflection_line_sum_l362_362878


namespace retail_profit_percent_l362_362187

variable (CP : ℝ) (MP : ℝ) (SP : ℝ)
variable (h_marked : MP = CP + 0.60 * CP)
variable (h_discount : SP = MP - 0.25 * MP)

theorem retail_profit_percent : CP = 100 → MP = CP + 0.60 * CP → SP = MP - 0.25 * MP → 
       (SP - CP) / CP * 100 = 20 := 
by
  intros h1 h2 h3
  sorry

end retail_profit_percent_l362_362187


namespace percentage_of_men_l362_362675

variable (M W : ℝ)
variable (h1 : M + W = 100)
variable (h2 : 0.20 * W + 0.70 * M = 40)

theorem percentage_of_men : M = 40 :=
by
  sorry

end percentage_of_men_l362_362675


namespace four_letter_words_with_A_at_least_once_l362_362303

theorem four_letter_words_with_A_at_least_once (A B C D E : Type) :
  let total := 5^4 in
  let without_A := 4^4 in
  total - without_A = 369 :=
by {
  let total := 5^4;
  let without_A := 4^4;
  have : total - without_A = 369 := by sorry;
  exact this;
}

end four_letter_words_with_A_at_least_once_l362_362303


namespace negation_of_implication_l362_362881

variable (α β : Real)

/-- The negation of the proposition "If α = β, then sin(α) = sin(β)" is 
"If sin(α) ≠ sin(β), then α ≠ β". -/
theorem negation_of_implication (h : ¬(α = β → sin α = sin β)) : 
  sin α ≠ sin β → α ≠ β := 
sorry

end negation_of_implication_l362_362881


namespace max_composite_numbers_l362_362809
open Nat

theorem max_composite_numbers : 
  ∃ X : Finset Nat, 
  (∀ x ∈ X, x < 1500 ∧ ¬Prime x) ∧ 
  (∀ x y ∈ X, x ≠ y → gcd x y = 1) ∧ 
  X.card = 12 := 
sorry

end max_composite_numbers_l362_362809


namespace intersection_circumcircle_l362_362704

-- Define the given data, points, and conditions
variables {A B C P Q R S T L M N : Point}
variables (ABC : Triangle) (L_mid : Midpoint L B C) (M_mid : Midpoint M C A) (N_mid : Midpoint N A B) 
          (P_on_AB : OnSegment P A B) (R_reflection : Reflection R P N) (Q_on_BC : OnSegment Q B C) 
          (S_reflection : Reflection S Q L) (PS_perp_QR : Perpendicular (Line P S) (Line Q R))

-- Define the proof problem statement
theorem intersection_circumcircle :
  ∃ T, Intersection (Line P S) (Line Q R) T → OnCircumcircle T L M N := 
sorry

end intersection_circumcircle_l362_362704


namespace sqrt_product_l362_362095

theorem sqrt_product (a b : ℝ) (h1 : a = 49) (h2 : b = 25) : sqrt (a * sqrt b) = 7 * sqrt 5 :=
by sorry

end sqrt_product_l362_362095


namespace percent_increase_fifth_triangle_l362_362988

noncomputable def initial_side_length : ℝ := 3
noncomputable def growth_factor : ℝ := 1.2
noncomputable def num_triangles : ℕ := 5

noncomputable def side_length (n : ℕ) : ℝ :=
  initial_side_length * growth_factor ^ (n - 1)

noncomputable def perimeter_length (n : ℕ) : ℝ :=
  3 * side_length n

noncomputable def percent_increase (n : ℕ) : ℝ :=
  ((perimeter_length n / perimeter_length 1) - 1) * 100

theorem percent_increase_fifth_triangle :
  percent_increase 5 = 107.4 :=
by
  sorry

end percent_increase_fifth_triangle_l362_362988


namespace sqrt_49_mul_sqrt_25_l362_362005

theorem sqrt_49_mul_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_l362_362005


namespace ten_digit_number_divisible_by_99_l362_362896

theorem ten_digit_number_divisible_by_99 (n : ℕ) (h1 : ∀ i, 0 ≤ i ∧ i ≤ 9 → n.digits.count i = 1) (h2 : n.digits.length = 10) (h3 : (n.digits.nth 9).get_or_else 0 ≠ 0) :
  let R := n.digits.reverse.digitsAsNumber in ∃ k, k = n * 10^10 + R ∧ k % 99 = 0 := 
sorry

end ten_digit_number_divisible_by_99_l362_362896


namespace period_tan_plus_cot_l362_362912

theorem period_tan_plus_cot (x : ℝ) : (tan x + cot x) = tan (x + π) + cot (x + π) :=
by sorry

end period_tan_plus_cot_l362_362912


namespace sum_of_possible_values_of_x_l362_362743

def f (x : ℝ) : ℝ :=
if x < 3 then 5 * x + 20 else 3 * x - 15

theorem sum_of_possible_values_of_x (h : ∀ x : ℝ, f x = 0 → x = -4 ∨ x = 5) : 
  (-4 : ℝ) + 5 = 1 :=
by
  sorry

end sum_of_possible_values_of_x_l362_362743


namespace average_price_of_six_toys_l362_362927

/-- Define the average cost of toys given the number of toys and their total cost -/
def avg_cost (total_cost : ℕ) (num_toys : ℕ) : ℕ :=
  total_cost / num_toys

/-- Define the total cost of toys given a list of individual toy costs -/
def total_cost (costs : List ℕ) : ℕ :=
  costs.foldl (· + ·) 0

/-- The main theorem -/
theorem average_price_of_six_toys :
  let dhoni_toys := 5
  let avg_cost_dhoni := 10
  let total_cost_dhoni := dhoni_toys * avg_cost_dhoni
  let david_toy_cost := 16
  let total_toys := dhoni_toys + 1
  total_cost_dhoni + david_toy_cost = 66 →
  avg_cost (66) (total_toys) = 11 :=
by
  -- Introduce the conditions and hypothesis
  intros total_cost_of_6_toys H
  -- Simplify the expression
  sorry  -- Proof skipped

end average_price_of_six_toys_l362_362927


namespace john_total_spent_l362_362386

noncomputable def calculate_total_spent : ℝ :=
  let orig_price_A := 900.0
  let discount_A := 0.15 * orig_price_A
  let price_A := orig_price_A - discount_A
  let tax_A := 0.06 * price_A
  let total_A := price_A + tax_A
  let orig_price_B := 600.0
  let discount_B := 0.25 * orig_price_B
  let price_B := orig_price_B - discount_B
  let tax_B := 0.09 * price_B
  let total_B := price_B + tax_B
  let total_other_toys := total_A + total_B
  let price_lightsaber := 2 * total_other_toys
  let tax_lightsaber := 0.04 * price_lightsaber
  let total_lightsaber := price_lightsaber + tax_lightsaber
  total_other_toys + total_lightsaber

theorem john_total_spent : calculate_total_spent = 4008.312 := by
  sorry

end john_total_spent_l362_362386


namespace find_non_AD_expression_l362_362183

variables {V : Type*} [add_comm_group V]

def expression_A (AB CD BC : V) := AB + CD + BC
def expression_B (AD EB BC CE : V) := AD + EB + BC + CE
def expression_C (MB MA BD : V) := MB - MA + BD
def expression_D (CB AD BC : V) := CB + AD - BC

theorem find_non_AD_expression (AB CD BC AD EB CE MB MA BD CB : V):
  (expression_D CB AD BC) ≠ AD ∧
  (expression_A AB CD BC) = AD ∧
  (expression_B AD EB BC CE) = AD ∧
  (expression_C MB MA BD) = AD :=
sorry

end find_non_AD_expression_l362_362183


namespace simplified_expression_value_l362_362844

theorem simplified_expression_value (a b : ℝ) (ha : a = -1) (hb : b = 1 / 4) :
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by
  sorry

end simplified_expression_value_l362_362844


namespace sqrt_product_l362_362091

theorem sqrt_product (a b : ℝ) (h1 : a = 49) (h2 : b = 25) : sqrt (a * sqrt b) = 7 * sqrt 5 :=
by sorry

end sqrt_product_l362_362091


namespace divisor_board_n_power_of_two_l362_362702

theorem divisor_board_n_power_of_two (n : ℕ) (h1 : 1 < n)
  (h2 : ∀ d, d ∣ n → ∃ i, d = d₁ i)
  (h3 : ∀ N, N ∈ board n → ∀ d, d ∣ N → d ∈ board n) :
  ∃ u : ℕ, n = 2^u :=
by sorry

-- Definitions for d_1, board, and any new constructs used
-- d₁ : (i : ℕ) → ℕ
-- board : ℕ → set ℕ := λ n, {d | d ∣ n} ∪ {di + dj | di dj ∈ {d | d ∣ n}}

end divisor_board_n_power_of_two_l362_362702


namespace max_winner_number_l362_362440

-- Conditions of the problem
def num_players : ℕ := 1024
def is_stronger (p1 p2 : ℕ) : Prop :=
  p1 < p2

-- Main theorem to be proven
theorem max_winner_number :
  ∃ k : ℕ, k = 20 ∧ ∀ (p1 p2 : ℕ), p1 ≠ p2 ∧ p1 ≤ num_players ∧ p2 ≤ num_players → 
  ((|p1 - p2| > 2 → is_stronger p1 p2) → k <= p2 ∧ is_stronger p1 p2) :=
sorry

end max_winner_number_l362_362440


namespace three_digit_integers_with_repeated_digits_count_l362_362321

theorem three_digit_integers_with_repeated_digits_count : 
  (∃ n, n = 252 ∧ ∀ x ∈ Set.range (Nat.succ 999), 
    100 ≤ x ∧ x < 1000 ∧ 
    (∃ (d2 d3 : ℕ), x = d2 * 10 * 10 + d3 * 10 + d1 ∧ 
      d2 ≠ 0 ∧ 
      ((d2 = d3) ∨ (d3 = d1) ∨ (d2 = d1))) ↔ n = 252 )
:= 
sorry

end three_digit_integers_with_repeated_digits_count_l362_362321


namespace smallest_positive_period_of_function_l362_362599

noncomputable def smallestPositivePeriod (f : ℝ → ℝ) := 
  ∃ p > 0, ∀ x ∈ {x : ℝ | ¬ ∃ k : ℤ, x = (1/4 : ℝ) * k * Real.pi + (Real.pi / 8) }, f (x + p) = f x

theorem smallest_positive_period_of_function :
  smallestPositivePeriod (λ x, sin (2 * x) + 2 * cos (2 * x) + 3 * tan (4 * x)) = π :=
sorry

end smallest_positive_period_of_function_l362_362599


namespace ellipse_equation_and_k_range_l362_362276

noncomputable def ellipse_properties (a b c : ℝ) (focus : ℝ × ℝ) (e : ℝ) (line : ℝ → ℝ) : Prop :=
(c = sqrt 3) ∧ (e = (sqrt 3 / 2)) ∧ (focus = (sqrt 3, 0)) ∧ 
  (line = (λ k, k * .x + sqrt 2)) ∧ 
  (∀ k : ℝ, (1 / 4 < k^2 ∧ k^2 < 1 / 3))

theorem ellipse_equation_and_k_range :
  ellipse_properties 2 1 (sqrt 3) (sqrt 3, 0) (sqrt 3 / 2) (λ k, k * .x + sqrt 2) :=
sorry

end ellipse_equation_and_k_range_l362_362276


namespace larger_number_l362_362490

theorem larger_number (x y : ℤ) (h1 : x + y = 47) (h2 : x - y = 7) : x = 27 :=
  sorry

end larger_number_l362_362490


namespace max_composite_numbers_l362_362787

theorem max_composite_numbers (S : Finset ℕ) (h1 : ∀ n ∈ S, n < 1500) (h2 : ∀ m n ∈ S, m ≠ n → Nat.gcd m n = 1) : S.card ≤ 12 := sorry

end max_composite_numbers_l362_362787


namespace circle_geometry_l362_362871

theorem circle_geometry (A B C D E : Point) (BAC CED : Angle) (BC CE : Length) :
  ∠BAC = ∠CED ∧ BC = 4 * CE → DB = 2 * DE :=
by
  sorry

end circle_geometry_l362_362871


namespace sqrt_mul_sqrt_l362_362048

theorem sqrt_mul_sqrt (a b c : ℝ) (hac : a = 49) (hbc : b = sqrt 25) (hc : c = 7 * sqrt 5) : sqrt (a * b) = c := 
by
  sorry

end sqrt_mul_sqrt_l362_362048


namespace problem_conditions_l362_362255

theorem problem_conditions (a : ℕ → ℤ) :
  (1 + x)^6 = a 0 + a 1 * (1 - x) + a 2 * (1 - x)^2 + a 3 * (1 - x)^3 + a 4 * (1 - x)^4 + a 5 * (1 - x)^5 + a 6 * (1 - x)^6 →
  a 6 = 1 ∧ a 1 + a 3 + a 5 = -364 :=
by sorry

end problem_conditions_l362_362255


namespace cone_volume_increase_l362_362123

theorem cone_volume_increase (r h : ℝ) (h_pos : h > 0) :
  let V := (1/3) * Real.pi * r^2 * h in
  let h' := 2 * h in
  let V' := (1/3) * Real.pi * r^2 * h' in
  V' = 2 * V :=
by
  let V := (1/3) * Real.pi * r^2 * h
  let h' := 2 * h
  let V' := (1/3) * Real.pi * r^2 * h'
  calc
    V' = (1/3) * Real.pi * r^2 * (2 * h) : by rw h'
    ... = 2 * ((1/3) * Real.pi * r^2 * h) : by ring
    ... = 2 * V                 : by rw V

end cone_volume_increase_l362_362123


namespace perpendicular_planes_l362_362731

variable {Line Plane : Type}
variable [IncidenceGeometry Line Plane]

-- Define the lines and planes
variable (m n : Line)
variable (alpha beta : Plane)

-- Given conditions are:
-- m and n are different lines
axiom h_lines : m ≠ n

-- alpha and beta are different planes
axiom h_planes : alpha ≠ beta

-- m is perpendicular to alpha
axiom perp_m_alpha : Perpendicular m alpha

-- m is parallel to beta
axiom para_m_beta : Parallel m beta

-- Prove that alpha is perpendicular to beta
theorem perpendicular_planes : Perpendicular alpha beta :=
  sorry

end perpendicular_planes_l362_362731


namespace max_composite_numbers_with_gcd_one_l362_362759

theorem max_composite_numbers_with_gcd_one : 
  ∃ (S : Finset ℕ), 
    (∀ x ∈ S, Nat.isComposite x) ∧ 
    (∀ x ∈ S, x < 1500) ∧ 
    (∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y → Nat.gcd x y = 1) ∧
    S.card = 12 :=
sorry

end max_composite_numbers_with_gcd_one_l362_362759


namespace total_profit_l362_362979

theorem total_profit (A B C : ℕ) (A_invest B_invest C_invest A_share : ℕ) (total_invest total_profit : ℕ)
  (h1 : A_invest = 6300)
  (h2 : B_invest = 4200)
  (h3 : C_invest = 10500)
  (h4 : A_share = 3630)
  (h5 : total_invest = A_invest + B_invest + C_invest)
  (h6 : total_profit * A_share = A_invest * total_invest) :
  total_profit = 12100 :=
by
  sorry

end total_profit_l362_362979


namespace coin_probability_not_unique_l362_362521

theorem coin_probability_not_unique :
  ∃ p : ℝ, 0 < p ∧ p < 1 ∧ 10 * (p^3) * ((1 - p)^2) = 144 / 625 :=
begin
  sorry
end

end coin_probability_not_unique_l362_362521


namespace probability_half_dollar_is_correct_l362_362528

def value_of_dimes : ℝ := 20.00
def value_of_half_dollars : ℝ := 30.00
def value_of_quarters : ℝ := 15.00

def worth_of_dime : ℝ := 0.10
def worth_of_half_dollar : ℝ := 0.50
def worth_of_quarter : ℝ := 0.25

def number_of_dimes : ℝ := value_of_dimes / worth_of_dime
def number_of_half_dollars : ℝ := value_of_half_dollars / worth_of_half_dollar
def number_of_quarters : ℝ := value_of_quarters / worth_of_quarter

def total_number_of_coins : ℝ := number_of_dimes + number_of_half_dollars + number_of_quarters

def probability_of_half_dollar : ℝ := number_of_half_dollars / total_number_of_coins

theorem probability_half_dollar_is_correct :
  probability_of_half_dollar = 3 / 16 := by
sorry

end probability_half_dollar_is_correct_l362_362528


namespace least_odd_prime_factor_of_2023_pow_8_add_1_l362_362593

theorem least_odd_prime_factor_of_2023_pow_8_add_1 :
  ∃ (p : ℕ), Prime p ∧ (2023^8 + 1) % p = 0 ∧ p % 2 = 1 ∧ p = 97 :=
by
  sorry

end least_odd_prime_factor_of_2023_pow_8_add_1_l362_362593


namespace Carrie_tshirts_spent_l362_362574

theorem Carrie_tshirts_spent:
  let cost_per_tshirt : ℝ := 9.65 in
  let number_of_tshirts : ℝ := 12 in
  let discount_rate : ℝ := 0.15 in
  let tax_rate : ℝ := 0.08 in
  let total_cost := cost_per_tshirt * number_of_tshirts in
  let discount := discount_rate * total_cost in
  let discounted_price := total_cost - discount in
  let sales_tax := tax_rate * discounted_price in
  let final_price := discounted_price + sales_tax in
  final_price = 106.30 :=
by
  sorry

end Carrie_tshirts_spent_l362_362574


namespace proportional_set_is_D_l362_362115

-- Define the sets of line segments
def SetA : List ℕ := [5, 6, 7, 8]
def SetB : List ℕ := [3, 6, 2, 5]
def SetC : List ℕ := [2, 4, 6, 8]
def SetD : List ℕ := [2, 3, 4, 6]

-- Define the condition for proportional segments
def isProportional (lst : List ℕ) : Prop :=
  lst.headI * lst.getLast sorry = lst.get (1 : Fin 4) * lst.get (2 : Fin 4)

-- Theorem statement
theorem proportional_set_is_D : isProportional SetD :=
by
  rw [isProportional]
  -- Proof would go here
  sorry

end proportional_set_is_D_l362_362115


namespace tetrahedron_edges_sum_of_squares_l362_362425

-- Given conditions
variables {a b c d e f x y z : ℝ}

-- Mathematical statement
theorem tetrahedron_edges_sum_of_squares :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 4 * (x^2 + y^2 + z^2) :=
sorry

end tetrahedron_edges_sum_of_squares_l362_362425


namespace max_composite_numbers_l362_362799
open Nat

def is_composite (n : ℕ) : Prop := 1 < n ∧ ∃ d : ℕ, 1 < d ∧ d < n ∧ d ∣ n

def has_gcd_of_one (l : List ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ l → b ∈ l → a ≠ b → gcd a b = 1

def valid_composite_numbers (n : ℕ) : Prop :=
  ∀ m ∈ (List.range n).filter is_composite, m < 1500 →

-- Main theorem
theorem max_composite_numbers :
  ∃ l : List ℕ, l.length = 12 ∧ valid_composite_numbers l ∧ has_gcd_of_one l :=
sorry

end max_composite_numbers_l362_362799


namespace max_ratio_of_distances_l362_362406

open Real

noncomputable theory


def points_on_circle (x y : ℤ) : Prop :=
  x^2 + y^2 = 25

def irrational_distance (p q : ℤ × ℤ) : Prop :=
  ¬ is_rat (Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2))

theorem max_ratio_of_distances
  (P Q R S : ℤ × ℤ)
  (hP : points_on_circle P.1 P.2)
  (hQ : points_on_circle Q.1 Q.2)
  (hR : points_on_circle R.1 R.2)
  (hS : points_on_circle S.1 S.2)
  (hPQ : irrational_distance P Q)
  (hRS : irrational_distance R S) :
  (Real.dist (P, Q)) / (Real.dist (R, S)) ≤ 5 * Real.sqrt 2 :=
sorry

end max_ratio_of_distances_l362_362406


namespace removed_number_is_34_l362_362860
open Real

theorem removed_number_is_34 (n : ℕ) (x : ℕ) (h₁ : 946 = (43 * (43 + 1)) / 2) (h₂ : 912 = 43 * (152 / 7)) : x = 34 :=
by
  sorry

end removed_number_is_34_l362_362860


namespace correct_equation_D_l362_362114

theorem correct_equation_D : (|5 - 3| = - (3 - 5)) :=
by
  sorry

end correct_equation_D_l362_362114


namespace inequality_solution_set_nonempty_l362_362632

-- Define the statement
theorem inequality_solution_set_nonempty (m : ℝ) : 
  (∃ x : ℝ, |x + 1| + |x - 1| < m) ↔ m > 2 :=
by
  sorry

end inequality_solution_set_nonempty_l362_362632


namespace pizza_slice_volume_l362_362167

-- Define the parameters given in the conditions
def pizza_thickness : ℝ := 0.5
def pizza_diameter : ℝ := 16.0
def num_slices : ℝ := 16.0

-- Define the volume of one slice
theorem pizza_slice_volume : (π * (pizza_diameter / 2) ^ 2 * pizza_thickness / num_slices) = 2 * π := by
  sorry

end pizza_slice_volume_l362_362167


namespace triangular_pyramid_cross_section_area_l362_362178

theorem triangular_pyramid_cross_section_area (base_area : ℝ) (planes_divide_equally : Prop)
    (base_area_eq : base_area = 18) 
    (planes_divide_equally : ∃ (plane1 plane2 ℝ), plane1 ≠ plane2 ∧ 
    ∀ t, 0 ≤ t → t ≤ 1 → (volume_of_tetrahedron_with_parallel_planes := (1/3) * volume_of_pyramid)) :
  ∃ area_of_cross_section, area_of_cross_section = 18 / real.cbrt 9 :=
by
  sorry

end triangular_pyramid_cross_section_area_l362_362178


namespace cost_of_chocolate_l362_362391

/-- Leonardo has 4 dollars in his pocket -/
def leonardo_dollars : ℕ := 4

/-- Leonardo borrowed 59 cents from his friend -/
def borrowed_cents : ℕ := 59

/-- Leonardo needs 41 more cents to purchase a chocolate -/
def needed_cents : ℕ := 41

/-- The cost of the chocolate in dollars -/
def chocolate_cost_in_dollars : ℕ :=
  let total_cents := (leonardo_dollars * 100) + borrowed_cents + needed_cents
  total_cents / 100

/-- Prove that the cost of the chocolate is 5 dollars -/
theorem cost_of_chocolate : chocolate_cost_in_dollars = 5 :=
by
  unfold chocolate_cost_in_dollars
  have h : (leonardo_dollars * 100) + borrowed_cents + needed_cents = 500 := by sorry
  rw [h]
  norm_num

end cost_of_chocolate_l362_362391


namespace trees_chopped_in_first_half_l362_362967

theorem trees_chopped_in_first_half (x : ℕ) (h1 : ∀ t, t = x + 300) (h2 : 3 * t = 1500) : x = 200 :=
by
  sorry

end trees_chopped_in_first_half_l362_362967


namespace probability_not_below_x_axis_l362_362817

open Real

structure Point (x y : ℝ)

def P : Point := ⟨4, 4⟩
def Q : Point := ⟨-2, -4⟩
def R : Point := ⟨-8, -4⟩
def S : Point := ⟨-2, 4⟩

-- Definition of the parallelogram from points
def parallelogram (A B C D : Point) : set (Point) := sorry -- Detailed definition removed for brevity

-- Total area of parallelogram PQRS
def area_PQRS : ℝ := 48

-- Area of the part of the parallelogram not below the x-axis
def area_PGHS : ℝ := 24

-- The probability that a randomly chosen point is not below the x-axis
theorem probability_not_below_x_axis : area_PGHS / area_PQRS = 1 / 2 :=
by
  -- Proof goes here
  sorry

end probability_not_below_x_axis_l362_362817


namespace count_4_letter_words_with_A_l362_362299

-- Define the alphabet set and the properties
def Alphabet : Finset (Char) := {'A', 'B', 'C', 'D', 'E'}
def total_words := (Alphabet.card ^ 4 : ℕ)
def total_words_without_A := (Alphabet.erase 'A').card ^ 4
def total_words_with_A := total_words - total_words_without_A

-- The key theorem to prove
theorem count_4_letter_words_with_A : total_words_with_A = 369 := sorry

end count_4_letter_words_with_A_l362_362299


namespace probability_of_A_l362_362829

def set_A : Set ℕ := {2, 3}
def set_B : Set ℕ := {1, 2, 3}

def inside_circle (m n : ℕ) : Prop := m^2 + n^2 < 9
def all_points (A B : Set ℕ) : Set (ℕ × ℕ) :=
  { (m, n) | m ∈ A ∧ n ∈ B }

theorem probability_of_A :
  let A := set_A,
      B := set_B,
      points := [(2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)],
      inside_points := [(2, 1), (2, 2)] in
    (inside_points.length : ℚ) / points.length = 1 / 3 :=
by {
  let points := [(2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)],
  let inside_points := [(2, 1), (2, 2)],
  have : (inside_points.length : ℚ) = 2, by norm_num,
  have : points.length = 6, by norm_num,
  rw [this, this],
  norm_num,
  sorry
}

end probability_of_A_l362_362829


namespace multiple_of_5_among_selected_l362_362938

/-- Let S be the set of numbers from 1 to 30. Prove that if we select at least 25 numbers from S, then at least one of the selected numbers is a multiple of 5. -/
theorem multiple_of_5_among_selected (S : set ℕ) (hS : S = {n | n ∈ finset.range 30.succ}):
  ∀ (T : finset ℕ), T.card = 25 → (∃ x ∈ T, x % 5 = 0) :=
sorry

end multiple_of_5_among_selected_l362_362938


namespace minimum_length_MN_l362_362618

variables (a : ℝ) (M N : ℝ × ℝ) (AA1 : set (ℝ × ℝ)) (BC : set (ℝ × ℝ)) (C1D1 : ℝ × ℝ)

-- Definitions of the lines and points based on the given problem
def point_on_line_AA1 (M : ℝ × ℝ) (a : ℝ) : Prop :=
  M.1 = 0 ∧ M.2 ∈ set.Icc 0 a

def point_on_line_BC (N : ℝ × ℝ) (a : ℝ) : Prop :=
  N.1 ∈ set.Icc 0 a ∧ N.2 = a

def line_intersects_edge (M N : ℝ × ℝ) (C1Dl : ℝ × ℝ) : Prop :=
  ∃ t ∈ set.Icc 0 1, (M.1 + t * (N.1 - M.1), M.2 + t * (N.2 - M.2)) = C1D1

-- The theorem statement
theorem minimum_length_MN
  (hM : point_on_line_AA1 M a)
  (hN : point_on_line_BC N a)
  (hMN_intersects : line_intersects_edge M N C1D1) :
  ∃ (MN_length : ℝ), MN_length = 3 * a :=
sorry -- Proof is omitted

end minimum_length_MN_l362_362618


namespace sqrt_49_mul_sqrt_25_l362_362000

theorem sqrt_49_mul_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_l362_362000


namespace sqrt_product_l362_362092

theorem sqrt_product (a b : ℝ) (h1 : a = 49) (h2 : b = 25) : sqrt (a * sqrt b) = 7 * sqrt 5 :=
by sorry

end sqrt_product_l362_362092


namespace three_person_subcommittees_from_seven_l362_362317

-- Definition of the combinations formula (binomial coefficient)
def choose : ℕ → ℕ → ℕ
| n, k => if k = 0 then 1 else (n * choose (n - 1) (k - 1)) / k 

-- Problem statement in Lean 4
theorem three_person_subcommittees_from_seven : choose 7 3 = 35 :=
by
  -- We would fill in the steps here or use a sorry to skip the proof
  sorry

end three_person_subcommittees_from_seven_l362_362317


namespace part1_intersection_part2_range_of_m_l362_362395

-- Define the universal set and the sets A and B
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 0 ∨ x > 3}
def B (m : ℝ) : Set ℝ := {x | x < m - 1 ∨ x > 2 * m}

-- Part (1): When m = 3, find A ∩ B
theorem part1_intersection:
  A ∩ B 3 = {x | x < 0 ∨ x > 6} :=
sorry

-- Part (2): If B ∪ A = B, find the range of values for m
theorem part2_range_of_m (m : ℝ) :
  (B m ∪ A = B m) → (1 ≤ m ∧ m ≤ 3 / 2) :=
sorry

end part1_intersection_part2_range_of_m_l362_362395


namespace vase_net_gain_l362_362415

theorem vase_net_gain 
  (selling_price : ℝ)
  (V1_cost : ℝ)
  (V2_cost : ℝ)
  (hyp1 : selling_price = 2.50)
  (hyp2 : 1.25 * V1_cost = selling_price)
  (hyp3 : 0.85 * V2_cost = selling_price) :
  (selling_price + selling_price) - (V1_cost + V2_cost) = 0.06 := 
by 
  sorry

end vase_net_gain_l362_362415


namespace sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l362_362078

theorem sqrt_49_mul_sqrt_25_eq_7_sqrt_5 : (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l362_362078


namespace derivative_log_base2_l362_362508

theorem derivative_log_base2 (x : ℝ) (hx : 0 < x) : 
  deriv (fun x => real.log x / real.log 2) x = 1 / (x * real.log 2) :=
by 
  sorry

end derivative_log_base2_l362_362508


namespace smallest_N_l362_362972

-- Definitions for the problem conditions
def is_rectangular_block (a b c : ℕ) (N : ℕ) : Prop :=
  N = a * b * c ∧ 143 = (a - 1) * (b - 1) * (c - 1)

-- Theorem to prove the smallest possible value of N
theorem smallest_N : ∃ a b c : ℕ, is_rectangular_block a b c 336 :=
by
  sorry

end smallest_N_l362_362972


namespace sqrt_mul_sqrt_l362_362106

theorem sqrt_mul_sqrt (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : sqrt (a * sqrt b) = sqrt a * sqrt (sqrt b) := by
  sorry

example : sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  have h1 : sqrt 25 = 5 := by 
    norm_num
  have h2 : sqrt (49 * 5) = sqrt 245 := by  
    unfold sqrt
  have h3 : sqrt 245 = 7 * sqrt 5 := by 
    norm_num
  rw [h1, h2, h3]
  norm_num

end sqrt_mul_sqrt_l362_362106


namespace find_theta_l362_362908

theorem find_theta :
  let ACB := 80
  let FEG := 64
  let DCE := 86
  let DEC := 83
  ∃ θ, θ = 180 - DCE - DEC :=
by
  existsi (180 - DCE - DEC)
  simp
  sorry

end find_theta_l362_362908


namespace complex_problem_l362_362629

noncomputable def complex_solution : ℂ :=
  let θ := real.mod_angle (5 * real.pi / 180) in
  complex.exp (θ * complex.I)

theorem complex_problem 
  (z : ℂ)
  (h : z + z⁻¹ = 2 * complex.cos (real.pi * 5 / 180)) :
  z ^ 2021 + (z ^ 2021)⁻¹ = real.sqrt 3 := 
sorry

end complex_problem_l362_362629


namespace zanqi_chestnuts_contribution_is_3_div_4_l362_362437

noncomputable def chestnuts_problem : Prop :=
  ∃ (a_1 a_5 : ℝ) (d : ℝ),
    2 * a_1 + (a_1 - a_5) + (a_1 - 2 * a_5) +
    (a_1 - 3 * a_5) + (a_1 - 4 * a_5) = 5 ∧
    d = -a_5 ∧
    a_3 = a_1 + 2 * d

theorem zanqi_chestnuts_contribution_is_3_div_4 : chestnuts_problem :=
  sorry

end zanqi_chestnuts_contribution_is_3_div_4_l362_362437


namespace triangle_acute_angle_l362_362369

variable (a b c : ℝ)
-- Natural number greater than 3
variable (n : ℕ) [Fact (n > 3)]

theorem triangle_acute_angle (h1 : c^2 = a^2 + b^2 → ∠ABC = 90) 
                             (h2 : c^3 = a^3 + b^3 → ∠ABC < 90)
                             (hn : c^n = a^n + b^n) : ∠ABC < 90 :=
sorry

end triangle_acute_angle_l362_362369


namespace MadHatterWaitsTwoHours_l362_362439

-- Define the conditions
def MadHatterRate : ℝ := 5/4
def MarchHareRate : ℝ := 5/6
def targetTime : ℝ := 5  -- Both plan to meet at 5:00 PM their respective times

-- Main theorem to be proved
theorem MadHatterWaitsTwoHours :
  let t_m := targetTime * (4/5)  -- Mad Hatter's real time in hours
  let t_h := targetTime * (6/5)  -- March Hare's real time in hours
  t_h - t_m = 2 := 
by
  sorry

end MadHatterWaitsTwoHours_l362_362439


namespace polynomial_with_shifted_roots_l362_362727

theorem polynomial_with_shifted_roots :
  (∃ a b c : ℝ, (a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ (∀ x, x^3 - 5*x + 7 = 0 ↔ x = a ∨ x = b ∨ x = c))
  → (∀ x, x^3 - 9*x^2 + 22*x - 5 = 0 ↔ x = a + 3 ∨ x = b + 3 ∨ x = c + 3)) :=
begin
  sorry
end

end polynomial_with_shifted_roots_l362_362727


namespace sqrt_49_times_sqrt_25_l362_362020

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 :=
by
  sorry

end sqrt_49_times_sqrt_25_l362_362020


namespace evaluate_delta_expression_l362_362365

def delta (x y : ℝ) : ℝ :=
  if x ≤ y then Real.sqrt (abs x) else y

theorem evaluate_delta_expression :
  (delta (-9) (-3)) * (delta 4 (-3)) = -9 := by
  sorry

end evaluate_delta_expression_l362_362365


namespace find_f_pi_six_value_l362_362288

noncomputable def f (x : ℝ) (f'₀ : ℝ) : ℝ := f'₀ * Real.sin x + Real.cos x

theorem find_f_pi_six_value (f'₀ : ℝ) (h : f'₀ = 2 + Real.sqrt 3) : f (π / 6) f'₀ = 1 + Real.sqrt 3 := 
by
  -- condition from the problem
  let f₀ := f (π / 6) f'₀
  -- final goal to prove
  sorry

end find_f_pi_six_value_l362_362288


namespace inequality_system_no_solution_l362_362608

theorem inequality_system_no_solution (a : ℝ) : (¬ ∃ x : ℝ, x < a - 3 ∧ x > 2a - 2) ↔ a ≥ -1 :=
by sorry

end inequality_system_no_solution_l362_362608


namespace sqrt_expression_simplified_l362_362016

theorem sqrt_expression_simplified :
  sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  sorry

end sqrt_expression_simplified_l362_362016


namespace obtuse_triangle_side_range_l362_362281

theorem obtuse_triangle_side_range
  (a : ℝ)
  (h1 : a > 0)
  (h2 : (a + 4)^2 > a^2 + (a + 2)^2)
  (h3 : (a + 2)^2 + (a + 4)^2 < a^2) : 
  2 < a ∧ a < 6 := 
sorry

end obtuse_triangle_side_range_l362_362281


namespace knights_gold_goblets_l362_362899

theorem knights_gold_goblets (k : ℕ) (k_gt_1 : 1 < k) (k_lt_13 : k < 13)
  (goblets : Fin 13 → Bool) (gold_goblets : (Fin 13 → Bool) → ℕ) 
  (cities : Fin 13 → Fin k) :
  (∃ (i j : Fin 13), i ≠ j ∧ cities i = cities j ∧ goblets i ∧ goblets j) :=
begin
  sorry
end

end knights_gold_goblets_l362_362899


namespace length_of_train_l362_362124

-- We state the conditions as definitions.
def length_of_train_equals_length_of_platform (l_train l_platform : ℝ) : Prop :=
l_train = l_platform

def speed_of_train (s : ℕ) : Prop :=
s = 216

def crossing_time (t : ℕ) : Prop :=
t = 1

-- Defining the goal according to the problem statement.
theorem length_of_train (l_train l_platform : ℝ) (s t : ℕ) 
  (h1 : length_of_train_equals_length_of_platform l_train l_platform) 
  (h2 : speed_of_train s) 
  (h3 : crossing_time t) : 
  l_train = 1800 :=
by
  sorry

end length_of_train_l362_362124


namespace initial_deposit_l362_362531

/-- 
A person deposits some money in a bank at an interest rate of 7% per annum (of the original amount). 
After two years, the total amount in the bank is $6384. Prove that the initial amount deposited is $5600.
-/
theorem initial_deposit (P : ℝ) (h : (P + 0.07 * P) + 0.07 * P = 6384) : P = 5600 :=
by
  sorry

end initial_deposit_l362_362531


namespace cars_in_section_G_l362_362417

theorem cars_in_section_G (total_cars_per_min: ℤ) (time_spent: ℤ) (rows_G: ℕ) (rows_H: ℕ) (cars_per_row_H: ℤ) (total_cars_walked: ℤ) : ∀ (cars_per_row_G : ℤ),
  rows_G * cars_per_row_G = total_cars_walked - rows_H * cars_per_row_H →
  cars_per_row_G = 10 :=
by {
  intros n h,
  sorry
}

end cars_in_section_G_l362_362417


namespace cube_face_sums_not_distinct_l362_362514

theorem cube_face_sums_not_distinct (label_set : set ℤ) (hlabel : label_set = {0, 1} ∨ label_set = {1, -1}) :
  ¬ ∃ sums : fin 6 → ℤ, (∀ i j : fin 6, i ≠ j → sums i ≠ sums j) ∧
  (∀ i : fin 6, ∃ vertices : fin 4 → ℤ, (∀ v, vertices v ∈ label_set) ∧ sums i = vertices 0 + vertices 1 + vertices 2 + vertices 3) :=
sorry

end cube_face_sums_not_distinct_l362_362514


namespace original_money_l362_362964
noncomputable def original_amount (x : ℝ) :=
  let after_first_loss := (2/3) * x
  let after_first_win := after_first_loss + 10
  let after_second_loss := after_first_win - (1/3) * after_first_win
  let after_second_win := after_second_loss + 20
  after_second_win

theorem original_money (x : ℝ) (h : original_amount x = x) : x = 48 :=
by {
  sorry
}

end original_money_l362_362964


namespace max_words_with_hamming_distance_l362_362456
-- Import required libraries:

open Function

-- Define the problem:
theorem max_words_with_hamming_distance (n : ℕ) :
  ∀ S : Finset (List Bool), (∀ (w1 w2 : List Bool), w1 ∈ S → w2 ∈ S → w1 ≠ w2 → Hamming.distance w1 w2 ≥ 3) → S.card ≤ 2^n / (n + 1) :=
sorry

end max_words_with_hamming_distance_l362_362456


namespace determine_n_l362_362732

theorem determine_n (n : ℕ) (h1 : 0 < n) 
(h2 : ∃ (sols : Finset (ℕ × ℕ × ℕ)), 
  (∀ (x y z : ℕ), (x, y, z) ∈ sols ↔ 3 * x + 2 * y + z = n ∧ x > 0 ∧ y > 0 ∧ z > 0) 
  ∧ sols.card = 55) : 
  n = 36 := 
by 
  sorry 

end determine_n_l362_362732


namespace sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l362_362087

theorem sqrt_49_mul_sqrt_25_eq_7_sqrt_5 : (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l362_362087


namespace product_sum_is_11513546_l362_362130

theorem product_sum_is_11513546 :
  (∑ n in finset.range 40, (2*(n+10)-1)*(2*(n+10)+1)*(2*(n+10)+3)) = 11513546 := sorry

end product_sum_is_11513546_l362_362130


namespace angle_bisector_divides_angle_l362_362424

variable {A B C D O : Type} [Triangle A B C] 
variable (O : CircleCenter A B C) (R : CircleRadius A B C) (AD : Altitude A B C A D)

theorem angle_bisector_divides_angle :
  let alpha := angle BAC;
  let beta := angle BAD;
  let gamma := angle BAO;
  angle_bisector_eq (angle BAC) (angle BAD) (angle BAO) ->
  angle BAD = angle DAO := sorry

end angle_bisector_divides_angle_l362_362424


namespace solve_equation_l362_362857

noncomputable def log_a (a x : ℝ) : ℝ :=
  Real.log x / Real.log a

theorem solve_equation (a x : ℝ) (h : a > 1) :
  (sqrt(log_a a (sqrt[4] (a * x)) + log_a x (sqrt[4] (a * x))) + 
   sqrt(log_a a (sqrt[4] (x / a)) + log_a x (sqrt[4] (a / x))) = a) ↔
  (x = a^(a^(-2)) ∨ x = a^(a^2)) := 
sorry

end solve_equation_l362_362857


namespace thief_speed_is_43_75_l362_362541

-- Given Information
def speed_owner : ℝ := 50
def time_head_start : ℝ := 0.5
def total_time_to_overtake : ℝ := 4

-- Question: What is the speed of the thief's car v?
theorem thief_speed_is_43_75 (v : ℝ) (hv : 4 * v = speed_owner * (total_time_to_overtake - time_head_start)) : v = 43.75 := 
by {
  -- The proof of this theorem is omitted as it is not required.
  sorry
}

end thief_speed_is_43_75_l362_362541


namespace oleg_max_composite_numbers_l362_362771

theorem oleg_max_composite_numbers : 
  ∃ (S : Finset ℕ), 
    (∀ (n ∈ S), n < 1500 ∧ ∃ p q, prime p ∧ prime q ∧ p ≠ q ∧ p * q = n) ∧ 
    (∀ (a b ∈ S), a ≠ b → gcd a b = 1) ∧ 
    S.card = 12 :=
sorry

end oleg_max_composite_numbers_l362_362771


namespace sufficient_but_not_necessary_condition_x_gt_5_x_gt_3_l362_362270

theorem sufficient_but_not_necessary_condition_x_gt_5_x_gt_3 :
  ∀ x : ℝ, (x > 5 → x > 3) ∧ (∃ x : ℝ, x > 3 ∧ x ≤ 5) :=
by
  sorry

end sufficient_but_not_necessary_condition_x_gt_5_x_gt_3_l362_362270


namespace exists_31_solutions_l362_362825

theorem exists_31_solutions :
  ∃ (S : Finset (ℕ × ℕ)), S.card ≥ 31 ∧ (∀ (x, y) ∈ S, 4 * x^3 - 3 * x + 1 = 2 * y^2 ∧ x ≤ 2005 ∧ x > 0 ∧ y > 0) := 
sorry

end exists_31_solutions_l362_362825


namespace sum_numbers_l362_362207

theorem sum_numbers :
  2345 + 3452 + 4523 + 5234 + 3245 + 2453 + 4532 + 5324 = 8888 := by
  sorry

end sum_numbers_l362_362207


namespace find_product_of_abc_l362_362890

theorem find_product_of_abc :
  ∃ (a b c m : ℝ), 
    a + b + c = 195 ∧
    m = 8 * a ∧
    m = b - 10 ∧
    m = c + 10 ∧
    a * b * c = 95922 := by
  sorry

end find_product_of_abc_l362_362890


namespace triangle_sides_condition_triangle_perimeter_l362_362725

theorem triangle_sides_condition (a b c : ℝ) (A B C : ℝ) 
  (h1 : sin C * sin (A - B) = sin B * sin (C - A)) : 2 * a^2 = b^2 + c^2 :=
sorry

theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ) 
  (h2 : a = 5) (h3 : cos A = 25 / 31) : a + b + c = 14 :=
sorry

end triangle_sides_condition_triangle_perimeter_l362_362725


namespace max_composite_numbers_l362_362781

theorem max_composite_numbers (s : set ℕ) (hs : ∀ n ∈ s, n < 1500 ∧ ∃ p : ℕ, prime p ∧ p ∣ n) (hs_gcd : ∀ x y ∈ s, x ≠ y → Nat.gcd x y = 1) :
  s.card ≤ 12 := 
by sorry

end max_composite_numbers_l362_362781


namespace probability_of_yellow_ball_l362_362476

theorem probability_of_yellow_ball 
  (red_balls : ℕ) 
  (yellow_balls : ℕ) 
  (blue_balls : ℕ) 
  (total_balls : ℕ)
  (h1 : red_balls = 2)
  (h2 : yellow_balls = 5)
  (h3 : blue_balls = 4)
  (h4 : total_balls = red_balls + yellow_balls + blue_balls) :
  (yellow_balls / total_balls : ℚ) = 5 / 11 :=
by 
  rw [h1, h2, h3] at h4  -- Substitute the ball counts into the total_balls definition.
  norm_num at h4  -- Simplify to verify the total is indeed 11.
  rw [h2, h4] -- Use the number of yellow balls and total number of balls to state the ratio.
  norm_num -- Normalize the fraction to show it equals 5/11.

#check probability_of_yellow_ball

end probability_of_yellow_ball_l362_362476


namespace wheel_diameter_calculation_l362_362977

def total_distance : ℝ := 1056
def revolutions : ℝ := 8.007279344858963
def correct_diameter : ℝ := 41.975

theorem wheel_diameter_calculation 
  (h1 : revolutions ≠ 0) : 
  ((total_distance / revolutions) / Real.pi) ≈ correct_diameter :=
by 
  sorry

end wheel_diameter_calculation_l362_362977


namespace sqrt_nested_l362_362061

theorem sqrt_nested : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_nested_l362_362061


namespace exists_x_eq_1_l362_362478

theorem exists_x_eq_1 (x y z t : ℕ) (h : x + y + z + t = 10) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (ht : 0 < t) :
  ∃ x, x = 1 :=
sorry

end exists_x_eq_1_l362_362478


namespace graph_behavior_l362_362877

def g (x : ℝ) : ℝ := -3 * x^4 + 5 * x^3 - 2

theorem graph_behavior : 
  (lim (at_top : filter ℝ) (λx, g x) = -∞) ∧ 
  (lim (at_bot : filter ℝ) (λx, g x) = -∞) :=
by 
  sorry

end graph_behavior_l362_362877


namespace QR_passes_through_fixed_point_l362_362404

-- Definition of a circle and tangents
structure Tangent (C : Type) (P : C) :=
  (is_tangent : ∃ (B : C), is_point_on_circle B C ∧ is_point_on_circle P C)

-- Points A, B, C, P, Q, R
variables {C : Type} [circle C]
variables (A : C)
variables {B C P Q R : C}

-- Tangents to the circle at points B and C
variables (AB_tangent_AC_tangent : Tangent C B)
variables (AC_tangent_AC_tangent : Tangent C C)

-- Arbitrary tangent L intersecting AB at P and AC at Q
variables (L_arbitrary_tangent : Tangent C P)
variables (L_arbitrary_tangent_ac : Tangent C Q)

-- Line through P parallel to AC intersects BC at R
variables (line_P_parallel_AC : is_parallel (line_through P) (line_through A C))

-- Proof Of QR Passing Through A Fixed Point As L Varies
theorem QR_passes_through_fixed_point (h1 : is_tangent AB_tangent_AC_tangent)
                                       (h2 : is_tangent AC_tangent_AC_tangent)
                                       (h3 : is_tangent L_arbitrary_tangent)
                                       (h4 : is_tangent L_arbitrary_tangent_ac)
                                       (h5 : is_parallel line_P_parallel_AC) :
                                       ∃ (F : C), ∀ (L : tangent C), passes_through (line_through Q R) F :=
sorry

end QR_passes_through_fixed_point_l362_362404


namespace denominator_of_fraction_l362_362496

theorem denominator_of_fraction (n : ℕ) (h1 : n = 20) (h2 : num = 35) (dec_value : ℝ) (h3 : dec_value = 2 / 10^n) : denom = 175 * 10^20 :=
by
  sorry

end denominator_of_fraction_l362_362496


namespace regular_octagon_interior_angle_l362_362500

theorem regular_octagon_interior_angle : 
  (∀ (n : ℕ), n = 8 → ∀ (sum_of_interior_angles : ℕ), sum_of_interior_angles = (n - 2) * 180 → ∀ (each_angle : ℕ), each_angle = sum_of_interior_angles / n → each_angle = 135) :=
  sorry

end regular_octagon_interior_angle_l362_362500


namespace probability_of_containing_cube_l362_362903

theorem probability_of_containing_cube : 
  ∃ (P : Set (ℝ × ℝ × ℝ)) (hP : P = {p : ℝ × ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 ∧ 0 ≤ p.3 ∧ p.3 ≤ 1}),
  ∀ (A B C : (ℝ × ℝ × ℝ)) (hA : A ∈ P) (hB : B ∈ P) (hC : C ∈ P),
  (∃ (Q : Set (ℝ × ℝ × ℝ)) (hQ : Q = {q : ℝ × ℝ × ℝ | q.1 >= 0 ∧ q.2 >= 0 ∧ q.3 >= 0 ∧ q.1 + 1/2 <= 1 ∧ q.2 + 1/2 <= 1 ∧ q.3 + 1/2 <= 1}), 
    (∀ q ∈ Q, q ∈ P) → 
    (∃ cube_center : ℝ × ℝ × ℝ, 
      Q = {q : ℝ × ℝ × ℝ | |q.1 - cube_center.1| <= 1/4 ∧ |q.2 - cube_center.2| <= 1/4 ∧ |q.3 - cube_center.3| <= 1/4 } ∧ 
      A, B, C ∈ Q ) → (probability_of_containing_cube = 1/8) :=
sorry

end probability_of_containing_cube_l362_362903


namespace female_students_count_l362_362867

variable (F : Nat)

theorem female_students_count
  (avg_all : 90)
  (avg_male : 8 * 84)
  (avg_female : F * 92)
  (total_avg : (8 + F) * 90)
  (eq : 8 * 84 + F * 92 = (8 + F) * 90) : F = 24 :=
by
  sorry

end female_students_count_l362_362867


namespace m_divisible_by_1979_l362_362740

theorem m_divisible_by_1979 (m n : ℕ) (hm : 0 < m) (hn : 0 < n)
  (hfrac : (m : ℚ)/n = ∑ k in Finset.range (1319+1), (-1)^(k+1) * (1/(k+1)))
  : 1979 ∣ m :=
sorry

end m_divisible_by_1979_l362_362740


namespace student_allowance_l362_362973

def spend_on_clothes (A : ℚ) := (4 / 7) * A
def spend_on_games (A : ℚ) := (4 / 7) * (3 / 5) * A
def spend_on_books (A : ℚ) := (4 / 7) * (3 / 5) * (5 / 9) * A
def spend_on_charity (A : ℚ) := (4 / 7) * (3 / 5) * (5 / 9) * (1 / 2) * A
def remaining_after_candy (A : ℚ) := (2 / 21) * A - 3.75

theorem student_allowance :
  ∃ A : ℚ, remaining_after_candy A = 0 → A = 39.375 :=
begin
  sorry
end

end student_allowance_l362_362973


namespace triangle_identity_triangle_perimeter_l362_362719

theorem triangle_identity 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : sin C * sin (A - B) = sin B * sin (C - A)) :
  2 * a^2 = b^2 + c^2 :=
sorry

theorem triangle_perimeter 
  (a b c : ℝ) 
  (A : ℝ) 
  (h1 : 2 * a^2 = b^2 + c^2) 
  (ha : a = 5) 
  (h_cosA : cos A = 25 / 31) :
  a + b + c = 14 :=
sorry

end triangle_identity_triangle_perimeter_l362_362719


namespace part_a_l362_362132

theorem part_a (n : ℕ) (hn : 0 < n) : 
  ∃ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x^(n-1) + y^n = z^(n+1) :=
sorry

end part_a_l362_362132


namespace limit_solution_l362_362203

noncomputable def limit_problem : Prop :=
  ∀ (f : ℝ → ℝ) (L : ℝ), 
  (f = λ x, (1 - sqrt (cos x)) / (1 - cos (sqrt x))) → 
  is_limit f 0 L

theorem limit_solution : limit_problem :=
begin
  intros f L h,
  have : f = λ x, (1 - sqrt (cos x)) / (1 - cos (sqrt x)) := h,
  rw this,
  apply limit_const,
  sorry
end

end limit_solution_l362_362203


namespace oleg_max_composite_numbers_l362_362772

theorem oleg_max_composite_numbers : 
  ∃ (S : Finset ℕ), 
    (∀ (n ∈ S), n < 1500 ∧ ∃ p q, prime p ∧ prime q ∧ p ≠ q ∧ p * q = n) ∧ 
    (∀ (a b ∈ S), a ≠ b → gcd a b = 1) ∧ 
    S.card = 12 :=
sorry

end oleg_max_composite_numbers_l362_362772


namespace geometric_proportion_l362_362495

theorem geometric_proportion (a b c d : ℝ) (h1 : a / b = c / d) (h2 : a / b = d / c) :
  (a = b ∧ b = c ∧ c = d) ∨ (|a| = |b| ∧ |b| = |c| ∧ |c| = |d| ∧ (a * b * c * d < 0)) :=
by
  sorry

end geometric_proportion_l362_362495


namespace expected_value_no_return_variance_with_return_l362_362615

-- Define the context and given problem
variable {totalBalls : Nat} {yellowBalls : Nat} {whiteBalls : Nat} {maxDraws : Nat} {X : Nat}

-- Conditions of the problem
def totalBalls := 6
def yellowBalls := 4
def whiteBalls := 2
def maxDraws := 3

/- If the ball drawn each time is not returned, prove that E(X) = 31/15 -/
theorem expected_value_no_return :
  E (X when no return) = 31 / 15 := sorry

/- If the ball drawn each time is returned, prove that D(X) = 62/81 -/
theorem variance_with_return :
  D (X when return) = 62 / 81 := sorry

end expected_value_no_return_variance_with_return_l362_362615


namespace expected_value_bounds_l362_362826

variable {α : Type}
variable (X : α → ℝ)
variable (p : α → ℝ)
variable (s : Finset α)

noncomputable def expected_value (X : α → ℝ) (p : α → ℝ) (s : Finset α) : ℝ :=
  ∑ i in s, X i * p i

variable (m M : ℝ)
variable (hx : ∀ x ∈ s, m ≤ X x ∧ X x ≤ M)
variable (hsum : ∑ i in s, p i = 1)

theorem expected_value_bounds :
  m ≤ expected_value X p s ∧ expected_value X p s ≤ M := 
sorry

end expected_value_bounds_l362_362826


namespace anna_reading_hours_l362_362991

/-- Anna is reading a 31-chapter textbook, 
    skips all chapters divisible by 3, 
    and it takes her 20 minutes to read each chapter.
    Prove that she spends a total of 7 hours reading the textbook. -/
theorem anna_reading_hours (total_chapters : ℕ) (skip_predicate : ℕ → Prop)
    (reading_time_per_chapter : ℕ) (total_minutes_in_hour : ℕ) 
    (chapters_not_divisible_by_3 : ℕ) (total_reading_time_in_minutes : ℕ) :
  total_chapters = 31 ∧ 
  (skip_predicate = λ n, n % 3 = 0) ∧
  reading_time_per_chapter = 20 ∧
  total_minutes_in_hour = 60 ∧ 
  chapters_not_divisible_by_3 = total_chapters - (total_chapters / 3) ∧
  total_reading_time_in_minutes = chapters_not_divisible_by_3 * reading_time_per_chapter  →
  (total_reading_time_in_minutes / total_minutes_in_hour) = 7 :=
begin 
  sorry
end

end anna_reading_hours_l362_362991


namespace num_words_with_A_l362_362314

theorem num_words_with_A :
  let total_words := 5^4,
      words_without_A := 4^4 in
  total_words - words_without_A = 369 :=
by
  sorry

end num_words_with_A_l362_362314


namespace mutually_exclusive_not_contradictory_l362_362350

theorem mutually_exclusive_not_contradictory:
  let balls := [3, 2, 1] in
  ∃ events : set (set ℕ), 
    "at_least_one_white_ball" ∈ events ∧ 
    "one_red_and_one_black_ball" ∈ events ∧ 
    (∀ (e1 e2 : set ℕ), e1 ≠ e2 → disjoint e1 e2) ∧
    ¬ (∀ e ∈ events, e ⊆ "contradictory") :=
begin
  sorry
end

end mutually_exclusive_not_contradictory_l362_362350


namespace tank_capacity_l362_362149

theorem tank_capacity
  (x : ℝ) -- define x as the full capacity of the tank in gallons
  (h1 : (5/6) * x - (2/3) * x = 15) -- first condition
  (h2 : (2/3) * x = y) -- second condition, though not actually needed
  : x = 90 := 
by sorry

end tank_capacity_l362_362149


namespace triangle_identity_triangle_perimeter_l362_362721

theorem triangle_identity 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : sin C * sin (A - B) = sin B * sin (C - A)) :
  2 * a^2 = b^2 + c^2 :=
sorry

theorem triangle_perimeter 
  (a b c : ℝ) 
  (A : ℝ) 
  (h1 : 2 * a^2 = b^2 + c^2) 
  (ha : a = 5) 
  (h_cosA : cos A = 25 / 31) :
  a + b + c = 14 :=
sorry

end triangle_identity_triangle_perimeter_l362_362721


namespace power_of_power_evaluate_3_power_3_power_2_l362_362234

theorem power_of_power (a m n : ℕ) : (a^m)^n = a^(m * n) :=
sorry

theorem evaluate_3_power_3_power_2 : (3^3)^2 = 729 :=
by
  have h1 : (3^3)^2 = 3^(3 * 2) := power_of_power 3 3 2
  have h2 : 3^(3 * 2) = 3^6 := rfl
  have h3 : 3^6 = 729 := sorry -- Placeholder for the actual multiplication calculation
  exact eq.trans (eq.trans h1 h2) h3

end power_of_power_evaluate_3_power_3_power_2_l362_362234


namespace value_of_a_plus_b_l362_362628

theorem value_of_a_plus_b (a b : ℝ) (h : |a - 2| = -(b + 5)^2) : a + b = -3 :=
sorry

end value_of_a_plus_b_l362_362628


namespace complex_expression_evaluation_l362_362285

theorem complex_expression_evaluation (z : ℂ) (h : z = 1 - I) :
  (z^2 - 2 * z) / (z - 1) = -2 * I :=
by
  sorry

end complex_expression_evaluation_l362_362285


namespace max_composite_numbers_l362_362805
open Nat

def is_composite (n : ℕ) : Prop := 1 < n ∧ ∃ d : ℕ, 1 < d ∧ d < n ∧ d ∣ n

def has_gcd_of_one (l : List ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ l → b ∈ l → a ≠ b → gcd a b = 1

def valid_composite_numbers (n : ℕ) : Prop :=
  ∀ m ∈ (List.range n).filter is_composite, m < 1500 →

-- Main theorem
theorem max_composite_numbers :
  ∃ l : List ℕ, l.length = 12 ∧ valid_composite_numbers l ∧ has_gcd_of_one l :=
sorry

end max_composite_numbers_l362_362805


namespace min_length_intersection_l362_362645

def set_with_length (a b : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ b}
def length_of_set (a b : ℝ) := b - a
def M (m : ℝ) := set_with_length m (m + 3/4)
def N (n : ℝ) := set_with_length (n - 1/3) n

theorem min_length_intersection (m n : ℝ) (h₁ : 0 ≤ m) (h₂ : m + 3/4 ≤ 1) (h₃ : 0 ≤ n - 1/3) (h₄ : n ≤ 1) : 
  length_of_set (max m (n - 1/3)) (min (m + 3/4) n) = 1/12 :=
by
  sorry

end min_length_intersection_l362_362645


namespace exists_f_with_f3_eq_9_forall_f_f3_le_9_l362_362218

-- Define the real-valued function f satisfying the given conditions
variable (f : ℝ → ℝ)
variable (f_real : ∀ x : ℝ, true)  -- f is real-valued and defined for all real numbers
variable (f_mul : ∀ x y : ℝ, f (x * y) = f x * f y)  -- f(xy) = f(x)f(y)
variable (f_add : ∀ x y : ℝ, f (x + y) ≤ 2 * (f x + f y))  -- f(x+y) ≤ 2(f(x) + f(y))
variable (f_2 : f 2 = 4)  -- f(2) = 4

-- Part a
theorem exists_f_with_f3_eq_9 : ∃ f : ℝ → ℝ, (∀ x : ℝ, true) ∧ 
                              (∀ x y : ℝ, f (x * y) = f x * f y) ∧ 
                              (∀ x y : ℝ, f (x + y) ≤ 2 * (f x + f y)) ∧ 
                              (f 2 = 4) ∧ 
                              (f 3 = 9) := 
sorry

-- Part b
theorem forall_f_f3_le_9 : ∀ f : ℝ → ℝ, 
                        (∀ x : ℝ, true) → 
                        (∀ x y : ℝ, f (x * y) = f x * f y) → 
                        (∀ x y : ℝ, f (x + y) ≤ 2 * (f x + f y)) → 
                        (f 2 = 4) → 
                        (f 3 ≤ 9) := 
sorry

end exists_f_with_f3_eq_9_forall_f_f3_le_9_l362_362218


namespace exponential_monotonicity_l362_362614

theorem exponential_monotonicity {a b c : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : c > 1) : c^a > c^b :=
by 
  sorry 

end exponential_monotonicity_l362_362614


namespace complement_M_l362_362669

section ComplementSet

variable (x : ℝ)

def M : Set ℝ := {x | 1 / x < 1}

theorem complement_M : {x | 0 ≤ x ∧ x ≤ 1} = Mᶜ := sorry

end ComplementSet

end complement_M_l362_362669


namespace open_safe_in_fewer_than_seven_attempts_l362_362539

-- Definitions based on the conditions of the problem
def is_good_code (code : List Nat) : Prop :=
  code.length = 7 ∧ code.nodup

def safe_password : List Nat := [safe_password_digit_1, safe_password_digit_2, safe_password_digit_3, 
                                 safe_password_digit_4, safe_password_digit_5, safe_password_digit_6, safe_password_digit_7]

-- Assumptions based on the problem conditions
axiom good_code_password : is_good_code safe_password 

axiom safe_opens (entered_code : List Nat) : Prop :=
  ∃ i, i < 7 ∧ entered_code.nth i = safe_password.nth i

-- Proof statement
theorem open_safe_in_fewer_than_seven_attempts :
  ∃ attempt1 attempt2 attempt3 attempt4 attempt5 attempt6 : List Nat,
    (is_good_code attempt1 ∧ is_good_code attempt2 ∧ is_good_code attempt3 ∧ 
     is_good_code attempt4 ∧ is_good_code attempt5 ∧ is_good_code attempt6) ∧
    (safe_opens attempt1 ∨ safe_opens attempt2 ∨ safe_opens attempt3 ∨ 
     safe_opens attempt4 ∨ safe_opens attempt5 ∨ safe_opens attempt6) :=
sorry

end open_safe_in_fewer_than_seven_attempts_l362_362539


namespace larger_number_l362_362491

variables (x y : ℕ)

theorem larger_number (h1 : x + y = 47) (h2 : x - y = 7) : x = 27 :=
sorry

end larger_number_l362_362491


namespace sqrt_nested_l362_362055

theorem sqrt_nested : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_nested_l362_362055


namespace quadratic_inequality_solution_set_l362_362432

theorem quadratic_inequality_solution_set :
  (∃ x : ℝ, 2 * x + 3 - x^2 > 0) ↔ (-1 < x ∧ x < 3) :=
sorry

end quadratic_inequality_solution_set_l362_362432


namespace sqrt_49_times_sqrt_25_eq_7sqrt5_l362_362075

theorem sqrt_49_times_sqrt_25_eq_7sqrt5 :
  (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  have h1 : Real.sqrt 25 = 5 := by sorry
  have h2 : 49 * 5 = 245 := by sorry
  have h3 : Real.sqrt 245 = 7 * Real.sqrt 5 := by sorry
  sorry

end sqrt_49_times_sqrt_25_eq_7sqrt5_l362_362075


namespace gold_tetrahedron_volume_l362_362216

theorem gold_tetrahedron_volume (side_length : ℝ) (h : side_length = 8) : 
  volume_of_tetrahedron_with_gold_vertices = 170.67 := 
by 
  sorry

end gold_tetrahedron_volume_l362_362216


namespace problem_statements_l362_362337

-- The problem description and conditions

def P_property (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f(x + a) = f(-x)

-- Theorems addressing each statement
theorem problem_statements (sin : ℝ → ℝ) (P : ℝ → Prop)
  (f : ℝ → ℝ) (g : ℝ → ℝ) (odd : ℝ → ℝ → Prop)
  (h1 : ∀ x, sin(x + Real.pi) = -sin(x))
  (h2 : ∀ x y, odd x y → y = -x)                                    -- Definition of odd function
  (h3 : ∀ a, P_property f a → P a)                                   -- Function having P(a) property
  (h4 : P_property f 2 ∧ odd f 2 ∧ f 1 = 1)                          -- Statement 2
  (h5 : P_property f 4 ∧ (∀ p, (p ≠ (1, 0)) → central_symmetric p) ∧
        (monotonic_decreasing_at (-1, 0) f))                         -- Statement 3
  (h6 : (P_property f 0 ∧ P_property f 3) ∧ (∀ x₁ x₂, 
        abs(f x₁ - f x₂) ≥ abs(g x₁ - g x₂)))                        -- Statement 4
  : (true ∧ ∀ x, f x ∈ {1, 3, 4}) :=
begin
  sorry
end

end problem_statements_l362_362337


namespace distinguishable_dodecahedron_colorings_l362_362534

noncomputable def num_distinguishable_dodecahedron_colorings : Nat :=
  11.factorial / 5

theorem distinguishable_dodecahedron_colorings :
  num_distinguishable_dodecahedron_colorings = 7983360 := by
  sorry

end distinguishable_dodecahedron_colorings_l362_362534


namespace probability_of_3_positive_answers_l362_362384

theorem probability_of_3_positive_answers (n k : ℕ) (p : ℚ) (h_n : n = 6) (h_k : k = 3) (h_p : p = 1/2) :
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k)) = 5 / 16 := by
  sorry

end probability_of_3_positive_answers_l362_362384


namespace sqrt_mul_sqrt_l362_362050

theorem sqrt_mul_sqrt (a b c : ℝ) (hac : a = 49) (hbc : b = sqrt 25) (hc : c = 7 * sqrt 5) : sqrt (a * b) = c := 
by
  sorry

end sqrt_mul_sqrt_l362_362050


namespace factor_of_cubic_polynomial_l362_362334

theorem factor_of_cubic_polynomial (k : ℤ) : (λ x : ℤ, x^3 + 3 * x^2 - 3 * x + k) (-1) = 0 → k = -5 :=
by
  sorry

end factor_of_cubic_polynomial_l362_362334


namespace find_c_l362_362155

open_locale classical

noncomputable def parabola_solution (a b c : ℝ) : Prop :=
  a = 1 ∧ b = 4/3 ∧ c = 4/3

theorem find_c
  (a b c : ℝ)
  (h1 : 1 = 1^2 + b + c)  
  (h2 : -8 = (-2)^2 + -2 * b + c) :
  c = 4 / 3 :=
    sorry

end find_c_l362_362155


namespace min_n_subsets_l362_362737

open Set Finite FiniteBasic

theorem min_n_subsets (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}) :
  ∃ (n : ℕ) (A : Fin n → Finset ℕ),
    (∀ i, (A i).card = 7) ∧
    (∀ i j, i < j → (A i ∩ A j).card ≤ 3) ∧
    (∀ M : Finset ℕ, M.card = 3 → ∃ k, M ⊆ A k) ∧
    n = 15 :=
by
  sorry

end min_n_subsets_l362_362737


namespace angle_quadrant_l362_362917

-- Definitions for the conditions given in the problem
def is_defined (θ : ℝ) : Prop := cos θ * tan θ > 0

-- Statement of the proof problem in Lean
theorem angle_quadrant (θ : ℝ) (h : is_defined θ) : 
  (0 ≤ θ ∧ θ < π/2) ∨ (π/2 < θ ∧ θ < π) :=
sorry

end angle_quadrant_l362_362917


namespace sqrt_49_times_sqrt_25_l362_362023

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 :=
by
  sorry

end sqrt_49_times_sqrt_25_l362_362023


namespace james_weekly_earnings_l362_362379

def hourly_rate : ℕ := 20
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 4

theorem james_weekly_earnings : hourly_rate * (hours_per_day * days_per_week) = 640 := by
  sorry

end james_weekly_earnings_l362_362379


namespace min_points_to_remove_no_equil_triangles_l362_362686

-- Definition representing the points in the triangular grid
def points_set : set (ℕ × ℕ) := {
  (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
  (0, 1), (1, 1), (2, 1), (3, 1),
  (0, 2), (1, 2), (2, 2),
  (0, 3), (1, 3),
  (0, 4)
}

-- A predicate to determine if three points form an equilateral triangle
def forms_equilateral_triangle (a b c : ℕ × ℕ) : Prop :=
  sorry -- actual math omitted for this example

-- Theorem statement
theorem min_points_to_remove_no_equil_triangles : ∃ (S : set (ℕ × ℕ)), 
  S ⊆ points_set ∧ 
  S.card = 7 ∧ 
  ∀ (T U V : ℕ × ℕ), T ∈ points_set \ S → U ∈ points_set \ S → V ∈ points_set \ S → ¬ forms_equilateral_triangle T U V := by
  sorry

end min_points_to_remove_no_equil_triangles_l362_362686


namespace sqrt_nested_l362_362056

theorem sqrt_nested : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_nested_l362_362056


namespace sum_of_other_endpoint_l362_362160

theorem sum_of_other_endpoint (x y : ℝ) (h₁ : (9 + x) / 2 = 5) (h₂ : (-6 + y) / 2 = -8) :
  x + y = -9 :=
sorry

end sum_of_other_endpoint_l362_362160


namespace part1_part2_part3_l362_362287

noncomputable def f (x : ℝ) : ℝ := x / (1 + x^2)

theorem part1 (x1 x2 : ℝ) (hx1 : 0 < x1) (hx1_1 : x1 < 1) (hx2 : 0 < x2) (hx2_1 : x2 < 1) :
  (x1 - x2) * (f x1 - f x2) ≥ 0 := sorry

theorem part2 (a : ℝ) (hx : ∀ x : ℝ, 0 < x → x < 1 → (3 * x^2 - x) / (1 + x^2) ≥ a * (x - 1/3)) :
  a = 9 / 10 := sorry

theorem part3 (x1 x2 x3 : ℝ) (hx1 : 0 < x1) (hx1_1 : x1 < 1) (hx2 : 0 < x2) (hx2_1 : x2 < 1) 
  (hx3 : 0 < x3) (hx3_1 : x3 < 1) (h_sum : x1 + x2 + x3 = 1) :
  let y := (3 * x1^2 - x1) / (1 + x1^2) + (3 * x2^2 - x2) / (1 + x2^2) + (3 * x3^2 - x3) / (1 + x3^2)
  in y = 0 := sorry

end part1_part2_part3_l362_362287


namespace f_2023_eq_1375_l362_362451

-- Define the function f and the conditions
noncomputable def f : ℕ → ℕ := sorry

axiom f_ff_eq (n : ℕ) (h : n > 0) : f (f n) = 3 * n
axiom f_3n2_eq (n : ℕ) (h : n > 0) : f (3 * n + 2) = 3 * n + 1

-- Prove the specific value for f(2023)
theorem f_2023_eq_1375 : f 2023 = 1375 := sorry

end f_2023_eq_1375_l362_362451


namespace measure_of_angle_y_l362_362363

theorem measure_of_angle_y (m n : ℝ) (parallel : m = n) :
  ∀ (A B H : ℝ) (angle_A : Real.angle A = 40) (angle_B : Real.angle B = 90) (angle_H : Real.angle H = 50),
  let y := 130 in y = 180 - angle_H :=
by
  sorry

end measure_of_angle_y_l362_362363


namespace common_difference_of_arithmetic_sequence_l362_362265

variable {a : ℕ → ℝ} {S : ℕ → ℝ}
noncomputable def S_n (n : ℕ) : ℝ := -n^2 + 4*n

theorem common_difference_of_arithmetic_sequence :
  (∀ n : ℕ, S n = S_n n) →
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d ∧ d = -2 :=
by
  intro h
  use -2
  sorry

end common_difference_of_arithmetic_sequence_l362_362265


namespace voters_count_l362_362969

/-- A video has a score of 90 points, and 65% of voters liked it. 
    Prove that the total number of voters is 300. -/
theorem voters_count (x : ℕ) (h1 : 0.65 * x - 0.35 * x = 90) : x = 300 :=
sorry

end voters_count_l362_362969


namespace solve_for_k_l362_362526

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n / 2

theorem solve_for_k (k : ℤ) (h1 : k % 2 = 1) (h2 : f (f (f k)) = 57) : k = 223 :=
by
  -- Proof will be provided here
  sorry

end solve_for_k_l362_362526


namespace problem_l362_362636

theorem problem 
  (m: ℝ)
  (h_eq: m^2 - m = m + 3)
  (h_domain: -3 - m ≤ m ∧ m ≤ m^2 - m)
  (h_odd: ∀ x ∈ [-3 - m, m^2 - m], f (-x) = -f x) 
  : f m < f 1 := by
  sorry

def f (x : ℝ) (m : ℝ) : ℝ := x^(2 - m)

end problem_l362_362636


namespace max_composite_numbers_l362_362785

theorem max_composite_numbers (S : Finset ℕ) (h1 : ∀ n ∈ S, n < 1500) (h2 : ∀ m n ∈ S, m ≠ n → Nat.gcd m n = 1) : S.card ≤ 12 := sorry

end max_composite_numbers_l362_362785


namespace leaks_empty_time_l362_362818

theorem leaks_empty_time (A L1 L2: ℝ) (hA: A = 1/2) (hL1_rate: A - L1 = 1/3) 
  (hL2_rate: A - L1 - L2 = 1/4) : 1 / (L1 + L2) = 4 :=
by
  sorry

end leaks_empty_time_l362_362818


namespace total_price_for_pizza_l362_362997

-- Definitions based on conditions
def num_friends : ℕ := 5
def amount_per_person : ℕ := 8

-- The claim to be proven
theorem total_price_for_pizza : num_friends * amount_per_person = 40 := by
  -- Since the proof detail is not required, we use 'sorry' to skip the proof.
  sorry

end total_price_for_pizza_l362_362997


namespace simplify_and_evaluate_l362_362856

theorem simplify_and_evaluate (a b : ℝ) (h₁ : a = -1) (h₂ : b = 1/4) : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_and_evaluate_l362_362856


namespace total_distance_is_144_l362_362390

-- Define the conditions given in the problem
variables (D : ℝ) (T : ℝ)
variables (out_speed : ℝ) (back_speed : ℝ)

-- Conditions as given in the problem statement
def condition1 := out_speed = 24
def condition2 := back_speed = 18
def condition3 := (D / out_speed) + (D / back_speed) = T
def condition4 := T = 7

-- The theorem statement
theorem total_distance_is_144 (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) :
  (2 * D) = 144 :=
sorry

end total_distance_is_144_l362_362390


namespace find_m_values_l362_362640

-- Given function
def f (m x : ℝ) : ℝ := m * x^2 + 3 * m * x + m - 1

-- Theorem statement
theorem find_m_values (m : ℝ) :
  (∃ x y, f m x = 0 ∧ f m y = 0 ∧ (x = 0 ∨ y = 0)) →
  (m = 1 ∨ m = -(5/4)) :=
by sorry

end find_m_values_l362_362640


namespace sqrt_mul_sqrt_l362_362111

theorem sqrt_mul_sqrt (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : sqrt (a * sqrt b) = sqrt a * sqrt (sqrt b) := by
  sorry

example : sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  have h1 : sqrt 25 = 5 := by 
    norm_num
  have h2 : sqrt (49 * 5) = sqrt 245 := by  
    unfold sqrt
  have h3 : sqrt 245 = 7 * sqrt 5 := by 
    norm_num
  rw [h1, h2, h3]
  norm_num

end sqrt_mul_sqrt_l362_362111


namespace max_oleg_composite_numbers_l362_362753

/-- Oleg's list of composite numbers less than 1500 and pairwise coprime. --/
def oleg_composite_numbers (numbers : List ℕ) : Prop :=
  ∀ n ∈ numbers, Nat.isComposite n ∧ n < 1500 ∧ (∀ m ∈ numbers, n ≠ m → Nat.gcd n m = 1)

/-- Maximum number of such numbers that Oleg could write. --/
theorem max_oleg_composite_numbers : ∃ numbers : List ℕ, oleg_composite_numbers numbers ∧ numbers.length = 12 := 
sorry

end max_oleg_composite_numbers_l362_362753


namespace maximum_value_of_k_l362_362135

theorem maximum_value_of_k :
  ∀ (black_squares : Finset (Fin 8 × Fin 8)),
  (black_squares.card = 7) →
  ∃ k : ℕ, (k = 8) ∧ (∀ (rect : Finset (Fin 8 × Fin 8)),
  ({p | p ∈ rect ∧ p ∉ black_squares}.card = k) → 
  k ≤ 8) :=
begin
  sorry
end

end maximum_value_of_k_l362_362135


namespace sufficient_condition_parallel_planes_l362_362275

-- Definitions of lines and planes
variables {Line Plane : Type} 
variable (contains : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (intersection : Line → Line → Set Point)
variable (M : Point)

-- Hypotheses based on conditions
variables (m n l1 l2 : Line) (α β : Plane)
variables (h1 : contains m α)
variables (h2 : contains n α)
variables (h3 : contains l1 β)
variables (h4 : contains l2 β)
variables (h5 : intersection l1 l2 = {M})

theorem sufficient_condition_parallel_planes :
  (parallel m l1) ∧ (parallel n l2) → parallel_plane α β :=
begin
  sorry
end

end sufficient_condition_parallel_planes_l362_362275


namespace regular_octagon_interior_angle_l362_362501

theorem regular_octagon_interior_angle : 
  (∀ (n : ℕ), n = 8 → ∀ (sum_of_interior_angles : ℕ), sum_of_interior_angles = (n - 2) * 180 → ∀ (each_angle : ℕ), each_angle = sum_of_interior_angles / n → each_angle = 135) :=
  sorry

end regular_octagon_interior_angle_l362_362501


namespace particles_meet_l362_362209

def radius (i : ℕ) (r₁ : ℝ) : ℝ :=
  2^(i-1) * r₁

def circumference (i : ℕ) (r₁ : ℝ) : ℝ :=
  2 * π * (radius i r₁)

/-- The particles will meet, given their paths around circles k₁, k₂, ..., k₁₀₀ -/
theorem particles_meet (r₁ : ℝ) : ∃ t₀ : ℝ, ∀ t₁ t₂ : ℝ, 
  (0 ≤ t₁ ∧ t₁ < circumference 100 r₁ ∧
  0 ≤ t₂ ∧ t₂ < circumference 100 r₁ ∧ 
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ 100 → (t₁ + circumference i r₁) % circumference 100 r₁ 
  = (t₂ + circumference (101 - i) r₁) % circumference 100 r₁) → t₁ = t₂ :=
sorry

end particles_meet_l362_362209


namespace max_composite_numbers_with_gcd_one_l362_362762

theorem max_composite_numbers_with_gcd_one : 
  ∃ (S : Finset ℕ), 
    (∀ x ∈ S, Nat.isComposite x) ∧ 
    (∀ x ∈ S, x < 1500) ∧ 
    (∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y → Nat.gcd x y = 1) ∧
    S.card = 12 :=
sorry

end max_composite_numbers_with_gcd_one_l362_362762


namespace part1_part2_l362_362714

-- Define the conditions
def triangle_conditions (a b c : ℝ) (A B C : ℝ) : Prop :=
  sin C * sin (A - B) = sin B * sin (C - A) 

-- Define the conclusion for part (1)
def proof_part1 (a b c : ℝ) (A B C : ℝ) (h : triangle_conditions a b c A B C) : Prop :=
  2 * a ^ 2 = b ^ 2 + c ^ 2

-- Define the conditions for part (2)
def triangle_conditions_part2 (a b c A : ℝ) : Prop :=
  a = 5 ∧ cos A = 25 / 31 

-- Define the conclusion for part (2)
def proof_part2 (a b c A : ℝ) (h : triangle_conditions_part2 a b c A) : Prop :=
  a + b + c = 14

-- The Lean statements for the complete problem
theorem part1 (a b c A B C : ℝ) 
  (h : triangle_conditions a b c A B C) : 
  proof_part1 a b c A B C h := 
sorry

theorem part2 (a b c A : ℝ) 
  (h : triangle_conditions_part2 a b c A) : 
  proof_part2 a b c A h := 
sorry

end part1_part2_l362_362714


namespace four_letter_words_with_A_at_least_once_l362_362306

theorem four_letter_words_with_A_at_least_once (A B C D E : Type) :
  let total := 5^4 in
  let without_A := 4^4 in
  total - without_A = 369 :=
by {
  let total := 5^4;
  let without_A := 4^4;
  have : total - without_A = 369 := by sorry;
  exact this;
}

end four_letter_words_with_A_at_least_once_l362_362306


namespace sqrt_nested_l362_362065

theorem sqrt_nested : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_nested_l362_362065


namespace triangle_area_l362_362345

theorem triangle_area (A : Real) (B : Real) (C : Real) (a : Real) (b : Real) (c : Real)
  (hA : A = pi / 3)
  (ha : a = sqrt 3)
  (hb : b = 1)
  (hsum : A + B + C = pi)
  (hsine : a = sin A * (a / b) * b) :
  let area := (1 / 2) * a * b
  in area = sqrt 3 / 2 :=
by
  sorry

end triangle_area_l362_362345


namespace sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l362_362086

theorem sqrt_49_mul_sqrt_25_eq_7_sqrt_5 : (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l362_362086


namespace find_sum_of_digits_in_base_l362_362863

theorem find_sum_of_digits_in_base (d A B : ℕ) (hd : d > 8) (hA : A < d) (hB : B < d) (h : (A * d + B) + (A * d + A) - (B * d + A) = 1 * d^2 + 8 * d + 0) : A + B = 10 :=
sorry

end find_sum_of_digits_in_base_l362_362863


namespace simplify_and_evaluate_l362_362847

theorem simplify_and_evaluate 
  (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) = 1 :=
by
  sorry

end simplify_and_evaluate_l362_362847


namespace first_player_wins_l362_362956

theorem first_player_wins (n m : ℕ) (hn : 2 ≤ n) (hm : 2 ≤ m) : 
  ∃ strategy : (ℕ × ℕ) → (ℕ × ℕ) → Prop, 
    (∀ move : (ℕ × ℕ), is_legal_move move (n, m) → strategy move (n, m))
    ∧ winning_strategy strategy :=
sorry

end first_player_wins_l362_362956


namespace find_natural_number_l362_362592

-- Define the problem statement
def satisfies_condition (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ (2 * n^2 - 2) = k * (n^3 - n)

-- The main theorem
theorem find_natural_number (n : ℕ) : satisfies_condition n ↔ n = 2 :=
sorry

end find_natural_number_l362_362592


namespace value_of_star_l362_362819

theorem value_of_star
  (digits : Fin 9 → ℕ)
  (line_sum : ℕ)
  (h_unique : ∀ i j, i ≠ j → digits i ≠ digits j)
  (h_range : ∀ i, 1 ≤ digits i ∧ digits i ≤ 9)
  (h_intersection : digits 0 = 1 ∧ digits 1 = 4 ∧ digits 2 = 2)
  (h_equal_sums : ∀ l, line_sum = digits (l 0) + digits (l 1) + digits (l 2) + digits (l 3))
  (line1 : Fin 9 → Fin 4)
  (line2 : Fin 9 → Fin 4)
  (line3 : Fin 9 → Fin 4)
  (line4 : Fin 9 → Fin 4)
  : line_sum = 13 ∧ digits 8 = 8 :=
by
  sorry

end value_of_star_l362_362819


namespace remaining_money_l362_362697

def octal_to_decimal (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | _ => (n % 10) * (8 ^ (n.toString.length - 1)) + octal_to_decimal (n / 10)
  end

theorem remaining_money (h_john_savings : ∀ n, octal_to_decimal 5555 = 2925) 
                       (h_laptop_cost : 1500 = 1500) :
  2925 - 1500 = 1425 :=
by sorry

end remaining_money_l362_362697


namespace sqrt_49_times_sqrt25_eq_7_sqrt5_l362_362036

-- Definitions based on conditions
def sqrt_25 := Real.sqrt 25
def sqrt_49 := Real.sqrt 49

theorem sqrt_49_times_sqrt25_eq_7_sqrt5 :
  Real.sqrt (49 * sqrt_25) = 7 * Real.sqrt 5 := by
sorry

end sqrt_49_times_sqrt25_eq_7_sqrt5_l362_362036


namespace part1_part2_l362_362626

variable (a b c : ℝ)

-- Condition part 1
axiom pos_real_a : 0 < a
axiom pos_real_b : 0 < b
axiom pos_real_c : 0 < c

-- Question part 1
theorem part1 : (a + b) * (a * b + c^2) ≥ 4 * a * b * c := by sorry

-- Condition part 2 (including the total sum constraint)
axiom sum_abc : a + b + c = 3

-- Question part 2
theorem part2 : sqrt (a + 1) + sqrt (b + 1) + sqrt (c + 1) ≤ 3 * sqrt 2 := by sorry

end part1_part2_l362_362626


namespace ping_pong_ball_probability_l362_362161

noncomputable def multiple_of_6_9_or_both_probability : ℚ :=
  let total_numbers := 72
  let multiples_of_6 := 12
  let multiples_of_9 := 8
  let multiples_of_both := 4
  (multiples_of_6 + multiples_of_9 - multiples_of_both) / total_numbers

theorem ping_pong_ball_probability :
  multiple_of_6_9_or_both_probability = 2 / 9 :=
by
  sorry

end ping_pong_ball_probability_l362_362161


namespace work_efficiency_l362_362934

theorem work_efficiency (days_A : ℕ) (days_B : ℕ) (h1 : days_A = 12) (h2 : B_is_twice_as_efficient : days_B = days_A / 2) : days_B = 6 :=
by
  -- Placeholder for actual proof
  sorry

end work_efficiency_l362_362934


namespace find_k_l362_362654

variables {x k : ℝ}

theorem find_k (h1 : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 8)) (h2 : k ≠ 0) : k = 8 :=
sorry

end find_k_l362_362654


namespace number_of_men_in_first_group_l362_362665

/-
Given the initial conditions:
1. Some men can color a 48 m long cloth in 2 days.
2. 6 men can color a 36 m long cloth in 1 day.

We need to prove that the number of men in the first group is equal to 9.
-/

theorem number_of_men_in_first_group (M : ℕ)
    (h1 : ∃ (x : ℕ), x * 48 = M * 2)
    (h2 : 6 * 36 = 36 * 1) :
    M = 9 :=
by
sorry

end number_of_men_in_first_group_l362_362665


namespace triangle_sides_relation_triangle_perimeter_l362_362710

theorem triangle_sides_relation
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : sin C * sin (A - B) = sin B * sin (C - A)) :
  2 * a^2 = b^2 + c^2 :=
sorry

theorem triangle_perimeter 
  (a b c : ℝ)
  (A B C : ℝ)
  (h_a : a = 5)
  (h_cosA : cos A = 25 / 31)
  (h_sin_relation : sin C * sin (A - B) = sin B * sin (C - A)) :
  a + b + c = 14 :=
sorry

end triangle_sides_relation_triangle_perimeter_l362_362710


namespace transportation_degrees_correct_l362_362950

-- Define the percentages for the different categories.
def salaries_percent := 0.60
def research_development_percent := 0.09
def utilities_percent := 0.05
def equipment_percent := 0.04
def supplies_percent := 0.02

-- Define the total percentage of non-transportation categories.
def non_transportation_percent := 
  salaries_percent + research_development_percent + utilities_percent + equipment_percent + supplies_percent

-- Define the full circle in degrees.
def full_circle_degrees := 360.0

-- Total percentage which must sum to 1 (i.e., 100%).
def total_budget_percent := 1.0

-- Calculate the percentage for transportation.
def transportation_percent := total_budget_percent - non_transportation_percent

-- Define the result for degrees allocated to transportation.
def transportation_degrees := transportation_percent * full_circle_degrees

-- Prove that the transportation degrees are 72.
theorem transportation_degrees_correct : transportation_degrees = 72.0 :=
by
  unfold transportation_degrees transportation_percent non_transportation_percent
  sorry

end transportation_degrees_correct_l362_362950


namespace part1_part2_l362_362635

noncomputable def g (x : ℝ) (k : ℝ) : ℝ := sqrt (k * x^2 + 4 * x + k + 3)

theorem part1 (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + 4 * x + k + 3 ≥ 0) ↔ (k ∈ set.Ici 1) := 
sorry

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := log (x^2 + a * x + b)

theorem part2 (a b k : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → k * x^2 + 4 * x + k + 3 ≥ 0) ∧
  (∀ x : ℝ, (kx^2 + 4x + k + 3 ≥ 0 ↔ -2 ≤ x ∧ x ≤ 3)) ∧
  (∀ x : ℝ, log (x^2 + a * x + b) > 0 ↔ (x < -2 ∨ x > 3)) →
  a = -1 ∧ b = -6 ∧ k ∈ set.Icc (-4) (-3/2) :=
sorry

end part1_part2_l362_362635


namespace sequence_polynomial_linear_or_constant_l362_362126

theorem sequence_polynomial_linear_or_constant 
  (a : ℕ → ℝ)
  (h : ∀ i ≥ 1, a (i - 1) + a (i + 1) = 2 * a i) :
  ∀ n : ℕ, ∃ a0 d : ℝ, ∀ x : ℝ, ∑ k in finset.range (n + 1), a k * (nat.choose n k) * x^k * (1 - x)^(n - k) = a0 + n * d * x :=
begin
  sorry
end

end sequence_polynomial_linear_or_constant_l362_362126


namespace proof_problem_l362_362413

variables {A B C D F : Type}
variables [TopologicalSpace A] [TopologicalSpace B] [TopologicalSpace C] [TopologicalSpace D] [TopologicalSpace F]

-- Definitions used in the conditions
def right_triangle (A B C : Type) := ∃ (right_angle : Angle A C B), right_angle = 90
def diameter_of_circle (A B : Type) (circle : Circle A) := A ∣ circle ∧ A = B
def tangent_to_circle_at_point (D : Type) (circle : Circle D) : Line D := sorry
def meets_extension (line : Line A) (point : Point A) : Line meets point := sorry

-- The proof problem with conditions and questions
theorem proof_problem 
  {circle : Circle A} {line : Line D} {right_angle : Angle A C B}
  (h_rt : right_triangle A B C)
  (h_diam : diameter_of_circle A B circle)
  (h_tangent : tangent_to_circle_at_point D circle = line)
  (h_extension : meets_extension (extend AC) F):

  -- Prove these statements
  ∠ F D C = ∠ F D A ∧
  ∠ C F D = 2 * ∠ A
:= sorry

end proof_problem_l362_362413


namespace sqrt_nested_l362_362062

theorem sqrt_nested : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_nested_l362_362062


namespace sqrt_49_times_sqrt_25_l362_362022

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 :=
by
  sorry

end sqrt_49_times_sqrt_25_l362_362022


namespace largest_integer_base7_four_digits_l362_362405

theorem largest_integer_base7_four_digits :
  ∃ M : ℕ, (∀ m : ℕ, 7^3 ≤ m^2 ∧ m^2 < 7^4 → m ≤ M) ∧ M = 48 :=
sorry

end largest_integer_base7_four_digits_l362_362405


namespace ratio_AB_AD_l362_362428

-- Declare the main variables and conditions
variables (ABCD EFGH : Type) 
variables (A B C D E F G H : ABCD) 
variables (x y s : ℝ)

-- Condition 1: Rectangle shares 40% of its area with square.
axiom share_area_40 : 0.4 * (x * y) = 0.25 * (s * s)

-- Condition 2: Square shares 25% of its area with rectangle.
axiom share_area_25 : 0.25 * (s * s) = 0.4 * (x * y)

-- We need to prove that the ratio of AB to AD is 10.
theorem ratio_AB_AD (h : share_area_40 = share_area_25) : x / y = 10 :=
by sorry

end ratio_AB_AD_l362_362428


namespace power_of_binomials_l362_362565

theorem power_of_binomials :
  (1 + Real.sqrt 2) ^ 2023 * (1 - Real.sqrt 2) ^ 2023 = -1 :=
by
  -- This is a placeholder for the actual proof steps.
  -- We use 'sorry' to indicate that the proof is omitted here.
  sorry

end power_of_binomials_l362_362565


namespace octahedron_side_length_l362_362179

theorem octahedron_side_length (P₁ P₂ P₃ P₄ P₁' P₂' P₃' P₄' : EucSpace)
  (h1 : dist P₁ P₂ = dist P₁ P₃)
  (h2 : dist P₁ P₂ = dist P₁ P₄)
  (h3 : dist P₂ P₃ = √(2))
  (h4 : dist P₂ P₄ = √(2))
  (h5 : dist P₃ P₄ = √(2))
  (octahedron_vertex : (∃ Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ : EucSpace,
      Q₁ ∈ segment P₁ P₂ ∧ Q₂ ∈ segment P₁ P₃ ∧ Q₃ ∈ segment P₁ P₄ ∧ 
      Q₄ ∈ segment P₁' P₂' ∧ Q₅ ∈ segment P₁' P₃' ∧ Q₆ ∈ segment P₁' P₄' ∧ 
      dist Q₁ Q₂ = dist Q₂ Q₃ ∧ dist Q₃ Q₄ = dist Q₄ Q₅ ∧ dist Q₅ Q₆ = dist Q₆ Q₁)) :
  ∃ s : ℝ, s = (3*sqrt(2))/(4) :=
sorry

end octahedron_side_length_l362_362179


namespace area_of_trajectory_of_P_l362_362346

theorem area_of_trajectory_of_P :
  ∀ (A B C O P : Type) (AC BC A_cos r x y : ℝ), 
    AC = 6 →
    BC = 7 →
    A_cos = 1 / 5 →
    (∃ (P : Type), (∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1) ↔ (O = incenter_of_triangle A B C)) →
    area_covered_by_trajectory O x y = (10 * sqrt (6)) / 3 :=
by sorry

end area_of_trajectory_of_P_l362_362346


namespace simplify_and_evaluate_l362_362852

theorem simplify_and_evaluate (a b : ℝ) (h₁ : a = -1) (h₂ : b = 1/4) : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_and_evaluate_l362_362852


namespace line_sum_slope_intercept_l362_362529

theorem line_sum_slope_intercept (m b : ℝ) (x y : ℝ)
  (hm : m = 3)
  (hpoint : (x, y) = (-2, 4))
  (heq : y = m * x + b) :
  m + b = 13 :=
by
  sorry

end line_sum_slope_intercept_l362_362529


namespace girls_trying_out_for_team_l362_362482

theorem girls_trying_out_for_team
  (boys : ℕ := 32) -- There were 32 boys trying out.
  (called_back : ℕ := 10) -- 10 students got called back.
  (didn't_make_cut : ℕ := 39) -- 39 students didn't make the cut.
  (total_students := called_back + didn't_make_cut) -- The total number of students is the sum of those who got called back and those who didn't make the cut.
  (students_trying_out := boys + G) -- Total students who tried out.
  : G = 17 := -- The number of girls who tried out is 17.
by
  unfold students_trying_out total_students 
  sorry

end girls_trying_out_for_team_l362_362482


namespace sum_of_digits_of_seven_digit_palindromes_l362_362540

theorem sum_of_digits_of_seven_digit_palindromes:
  let sum_digits (n : Nat) : Nat := n.digits.sum
  let seven_digit_palindromes_sum := 4500 * 9999999
  sum_digits seven_digit_palindromes_sum = 63 := by
  sorry

end sum_of_digits_of_seven_digit_palindromes_l362_362540


namespace sqrt_49_times_sqrt_25_eq_7sqrt5_l362_362069

theorem sqrt_49_times_sqrt_25_eq_7sqrt5 :
  (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  have h1 : Real.sqrt 25 = 5 := by sorry
  have h2 : 49 * 5 = 245 := by sorry
  have h3 : Real.sqrt 245 = 7 * Real.sqrt 5 := by sorry
  sorry

end sqrt_49_times_sqrt_25_eq_7sqrt5_l362_362069


namespace pebbles_ratio_l362_362414

variable (S : ℕ)

theorem pebbles_ratio :
  let initial_pebbles := 18
  let skipped_pebbles := 9
  let additional_pebbles := 30
  let final_pebbles := 39
  initial_pebbles - skipped_pebbles + additional_pebbles = final_pebbles →
  (skipped_pebbles : ℚ) / initial_pebbles = 1 / 2 :=
by
  intros
  sorry

end pebbles_ratio_l362_362414


namespace find_solution_set_l362_362600

open Real -- open the real numbers scope to use real number functions.

noncomputable def log_ineq_solution (x : ℝ) : Prop :=
  log (2, 1 - (1 / x)) > 1

-- def all necessary conditions
def condition1 (x : ℝ) : Prop :=
  1 - (1 / x) > 0

def condition2 (x : ℝ) : Prop :=
  x ≠ 0

-- theorem statement, combining everything together.
theorem find_solution_set (x : ℝ) (h₁ : condition1 x) (h₂ : condition2 x) :
  log_ineq_solution x ↔ (-1 < x ∧ x < 0) :=
sorry -- proof is omitted.

end find_solution_set_l362_362600


namespace num_integers_S_l362_362707

open Nat

theorem num_integers_S :
  let S := { n : ℕ | n > 1 ∧ ∃ k : ℕ, k * n = 999999 }
  ∃ k, 999999 = k ∧ S.card = 47 :=
by
  sorry

end num_integers_S_l362_362707


namespace jessica_final_balance_l362_362516

theorem jessica_final_balance :
  let B := (400 : ℕ) * 5 / 2 in
  let remaining_balance := B - 400 in
  let deposit := remaining_balance / 4 in
  let final_balance := remaining_balance + deposit in
  final_balance = 750 :=
by
  let B := (400 : ℕ) * 5 / 2
  let remaining_balance := B - 400
  let deposit := remaining_balance / 4
  let final_balance := remaining_balance + deposit
  show final_balance = 750
  sorry

end jessica_final_balance_l362_362516


namespace negation_of_cosine_statement_l362_362422

theorem negation_of_cosine_statement :
  (¬ ∀ x : ℝ, cos x ≥ 1) ↔ ∃ x : ℝ, cos x < 1 :=
by sorry

end negation_of_cosine_statement_l362_362422


namespace max_composite_numbers_l362_362777

theorem max_composite_numbers (s : set ℕ) (hs : ∀ n ∈ s, n < 1500 ∧ ∃ p : ℕ, prime p ∧ p ∣ n) (hs_gcd : ∀ x y ∈ s, x ≠ y → Nat.gcd x y = 1) :
  s.card ≤ 12 := 
by sorry

end max_composite_numbers_l362_362777


namespace six_digit_number_condition_l362_362396

theorem six_digit_number_condition (a b c : ℕ) (h : 1 ≤ a ∧ a ≤ 9) (hb : b < 10) (hc : c < 10) : 
  ∃ k : ℕ, 100000 * a + 10000 * b + 1000 * c + 100 * (2 * a) + 10 * (2 * b) + 2 * c = 2 * k := 
by
  sorry

end six_digit_number_condition_l362_362396


namespace windows_preference_count_l362_362519

-- Define the total number of students surveyed
def totalStudents : ℕ := 210

-- Define the number of students who preferred Mac to Windows
def numMac : ℕ := 60

-- Define the number of students who had no preference
def numNoPreference : ℕ := 90

-- Calculate the number of students who equally preferred both brands
def numBoth : ℕ := numMac / 3

-- Calculate the number of students who preferred Windows to Mac
def numWindows : ℕ := totalStudents - (numMac + numBoth + numNoPreference)

-- Prove that the number of students who preferred Windows to Mac is 40
theorem windows_preference_count :
  numWindows = 40 :=
by
  unfold numWindows numBoth numMac totalStudents numNoPreference
  norm_num
  sorry

end windows_preference_count_l362_362519


namespace CD_value_l362_362484

-- Definitions representing the conditions
variables (A B C D E : Type*) [ordered_ring A]
variables (BD : A) (angleDBA : A) (angleBDC : A) (ratioBC_AD : A)
variables (AD : A) (BC : A) (CD : A) (DE : A)

-- Given conditions
def cond1 := BD = 2
def cond2 := angleDBA = 30
def cond3 := angleBDC = 60
def cond4 := ratioBC_AD = 5 / 3
def cond5 := BC = CD + DE
def cond6 := DE = BD

-- Prove that CD = 4 / 3
theorem CD_value : CD = 4 / 3 :=
by
  -- Use the given conditions in the proof
  have h1 : BD = 2 := cond1
  have h2 : angleDBA = 30 := cond2
  have h3 : angleBDC = 60 := cond3
  have h4 : ratioBC_AD = 5 / 3 := cond4
  have h5 : BC = CD + DE := cond5
  have h6 : DE = BD := cond6
  
  -- Express the ratio in terms of CD and simplify
  have h7 : 5 / 3 = (CD + DE) / AD := h4

  -- Perform the necessary calculations
  sorry

end CD_value_l362_362484


namespace interest_earned_l362_362510

-- Define the conditions
variables (a r : ℝ)

-- Define the main theorem to prove the interest earned
theorem interest_earned (a r : ℝ) : 
  let interest_maturity := (12 * (a * r * 12 + a * r * 1)) / 2 in
  interest_maturity = 78 * a * r :=
by 
  -- Placeholder for proof
  sorry

end interest_earned_l362_362510


namespace problem1_problem2_problem3_l362_362568

theorem problem1 : (3 * Real.sqrt 2 - Real.sqrt 12) * (Real.sqrt 18 + 2 * Real.sqrt 3) = 6 :=
by
  sorry

theorem problem2 : (Real.sqrt 5 - Real.sqrt 6) ^ 2 - (Real.sqrt 5 + Real.sqrt 6) ^ 2 = -4 * Real.sqrt 30 :=
by
  sorry

theorem problem3 : (2 * Real.sqrt (3 / 2) - Real.sqrt (1 / 2)) * (1 / 2 * Real.sqrt 8 + Real.sqrt (2 / 3)) = (5 / 3) * Real.sqrt 3 + 1 :=
by
  sorry

end problem1_problem2_problem3_l362_362568


namespace max_composite_numbers_l362_362812
open Nat

theorem max_composite_numbers : 
  ∃ X : Finset Nat, 
  (∀ x ∈ X, x < 1500 ∧ ¬Prime x) ∧ 
  (∀ x y ∈ X, x ≠ y → gcd x y = 1) ∧ 
  X.card = 12 := 
sorry

end max_composite_numbers_l362_362812


namespace sqrt_49_times_sqrt25_eq_7_sqrt5_l362_362040

-- Definitions based on conditions
def sqrt_25 := Real.sqrt 25
def sqrt_49 := Real.sqrt 49

theorem sqrt_49_times_sqrt25_eq_7_sqrt5 :
  Real.sqrt (49 * sqrt_25) = 7 * Real.sqrt 5 := by
sorry

end sqrt_49_times_sqrt25_eq_7_sqrt5_l362_362040


namespace find_initial_speed_l362_362443

noncomputable def P : ℝ := 60000  -- Power in J/s (Watts)
noncomputable def s : ℝ := 450   -- Distance in meters
noncomputable def m : ℝ := 1000  -- Mass in kilograms
noncomputable def v0 : ℝ := 30   -- Required initial speed in m/s

-- Hypothesis: The resistive force is proportional to the speed
def F_resistance (α v : ℝ) := α * v

theorem find_initial_speed : v0 = (∛((P * s) / m)) :=
by
  sorry

end find_initial_speed_l362_362443


namespace circle_rolling_triangle_distance_l362_362142

theorem circle_rolling_triangle_distance (r t1 t2 t3 : ℕ) (h1 : r = 2) (h2 : t1 = 9) (h3 : t2 = 12) (h4 : t3 = 15) :
  distance_center_circle (r) (t1) (t2) (t3) = 12 := sorry

end circle_rolling_triangle_distance_l362_362142


namespace reflection_correct_l362_362920

/-- Definition of reflection across the line y = -x -/
def reflection_across_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

/-- Given points C and D, and their images C' and D' respectively, under reflection,
    prove the transformation is correct. -/
theorem reflection_correct :
  (reflection_across_y_eq_neg_x (-3, 2) = (3, -2)) ∧ (reflection_across_y_eq_neg_x (-2, 5) = (2, -5)) :=
  by
    sorry

end reflection_correct_l362_362920


namespace max_composite_numbers_l362_362800
open Nat

def is_composite (n : ℕ) : Prop := 1 < n ∧ ∃ d : ℕ, 1 < d ∧ d < n ∧ d ∣ n

def has_gcd_of_one (l : List ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ l → b ∈ l → a ≠ b → gcd a b = 1

def valid_composite_numbers (n : ℕ) : Prop :=
  ∀ m ∈ (List.range n).filter is_composite, m < 1500 →

-- Main theorem
theorem max_composite_numbers :
  ∃ l : List ℕ, l.length = 12 ∧ valid_composite_numbers l ∧ has_gcd_of_one l :=
sorry

end max_composite_numbers_l362_362800


namespace range_of_a_l362_362448

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ax^2 + 3 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 4 / 9) :=
sorry

end range_of_a_l362_362448


namespace simplify_expression_l362_362841

theorem simplify_expression (a b : ℚ) (ha : a = -1) (hb : b = 1/4) : 
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_expression_l362_362841


namespace find_a_l362_362982

theorem find_a
  (r1 r2 r3 : ℕ)
  (hr1 : r1 > 2) (hr2 : r2 > 2) (hr3 : r3 > 2)
  (a b c : ℤ)
  (hr : (Polynomial.X - Polynomial.C (r1 : ℤ)) * 
         (Polynomial.X - Polynomial.C (r2 : ℤ)) * 
         (Polynomial.X - Polynomial.C (r3 : ℤ)) = 
         Polynomial.X ^ 3 + Polynomial.C a * Polynomial.X ^ 2 + Polynomial.C b * Polynomial.X + Polynomial.C c)
  (h : a + b + c + 1 = -2009) :
  a = -58 := sorry

end find_a_l362_362982


namespace minimize_sector_perimeter_l362_362441

theorem minimize_sector_perimeter (R : ℝ) (h : (∃ α, (1 / 2) * R^2 * α = 100 ∧ (2 * R + R * α) = minimize (λ (R : ℝ), 2 * R + 200 / R))) : R = 10 :=
sorry

end minimize_sector_perimeter_l362_362441


namespace solution_set_inequality_l362_362258

def f (x : ℝ) : ℝ :=
  if x >= 0 then 1 else -1

theorem solution_set_inequality :
  { x : ℝ | x + (x + 2) * f(x + 2) ≤ 5 } = { x : ℝ | x ≤ 3/2 } :=
by
  sorry

end solution_set_inequality_l362_362258


namespace ruby_height_l362_362663

variable (Ruby Pablo Charlene Janet : ℕ)

theorem ruby_height :
  (Ruby = Pablo - 2) →
  (Pablo = Charlene + 70) →
  (Janet = 62) →
  (Charlene = 2 * Janet) →
  Ruby = 192 := 
by
  sorry

end ruby_height_l362_362663


namespace card_probability_l362_362182

/-- Alexio has 120 cards numbered from 1 to 120. The probability that a randomly
selected card is a multiple of 2, 4, or 6 is 1/2. -/
theorem card_probability : (∃ c : ℕ, 1 ≤ c ∧ c ≤ 120) → 
  ((∃ c : ℕ, 1 ≤ c ∧ c ≤ 120 ∧ (c % 2 = 0 ∨ c % 4 = 0 ∨ c % 6 = 0)) → 
  ∑ c in (finset.range 121).filter (λ c, c % 2 = 0 ∨ c % 4 = 0 ∨ c % 6 = 0), 1 / 120 = 1 / 2) :=
by sorry

end card_probability_l362_362182


namespace single_jalapeno_strips_l362_362864

-- Definitions based on conditions
def strips_per_sandwich : ℕ := 4
def minutes_per_sandwich : ℕ := 5
def hours_per_day : ℕ := 8
def total_jalapeno_peppers_used : ℕ := 48
def minutes_per_hour : ℕ := 60

-- Calculate intermediate steps
def total_minutes : ℕ := hours_per_day * minutes_per_hour
def total_sandwiches_served : ℕ := total_minutes / minutes_per_sandwich
def total_strips_needed : ℕ := total_sandwiches_served * strips_per_sandwich

theorem single_jalapeno_strips :
  total_strips_needed / total_jalapeno_peppers_used = 8 := 
by
  sorry

end single_jalapeno_strips_l362_362864


namespace simson_line_of_point_l362_362694

variables {t t1 t2 t3 z : ℂ}

def s1 : ℂ := t1 + t2 + t3
def s2 : ℂ := t1 * t2 + t2 * t3 + t3 * t1
def s3 : ℂ := t1 * t2 * t3

theorem simson_line_of_point : 
  ∀ {t : ℂ}, (abs t = 1) → 
  (t * z - s3 * conj(z) = (1 / (2 * t)) * (t ^ 3 + s1 * t ^ 2 - s2 * t - s3)) :=
by 
  sorry

end simson_line_of_point_l362_362694


namespace sum_of_reciprocals_of_square_numbers_l362_362571

open BigOperators

theorem sum_of_reciprocals_of_square_numbers : 
  let s : ℝ := ∑ n in Finset.range 1001 \+ 1, (1 : ℝ) / (n ^ 2)
  abs (s - 1.644) < 0.01 := 
by 
  sorry

end sum_of_reciprocals_of_square_numbers_l362_362571


namespace jims_speed_l362_362563

variable (x : ℝ)

theorem jims_speed (bob_speed : ℝ) (bob_head_start : ℝ) (time : ℝ) (bob_distance : ℝ) :
  bob_speed = 6 →
  bob_head_start = 1 →
  time = 1 / 3 →
  bob_distance = bob_speed * time →
  (x * time = bob_distance + bob_head_start) →
  x = 9 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end jims_speed_l362_362563


namespace train_crosses_platforms_l362_362974

noncomputable def length_of_second_platform 
  (length_of_train : ℕ) (length_of_first_platform : ℕ) (time_first : ℕ) (time_second : ℕ) : ℕ :=
  let speed := (length_of_train + length_of_first_platform) / time_first
          total_distance_second := speed * time_second
          second_platform_length := total_distance_second - length_of_train in
  second_platform_length

theorem train_crosses_platforms
  (length_of_train : ℕ)
  (length_of_first_platform : ℕ)
  (time_first : ℕ)
  (time_second : ℕ)
  (h_train_length : length_of_train = 30)
  (h_first_platform_length : length_of_first_platform = 180)
  (h_time_first : time_first = 15)
  (h_time_second : time_second = 20) :
  length_of_second_platform length_of_train length_of_first_platform time_first time_second = 250 :=
by 
  rw [h_train_length, h_first_platform_length, h_time_first, h_time_second]
  -- The rest of the proof follows from calculations
  sorry

end train_crosses_platforms_l362_362974


namespace gasoline_tank_capacity_l362_362150

theorem gasoline_tank_capacity (x : ℕ) (h1 : 5 * x / 6 - 2 * x / 3 = 15) : x = 90 :=
sorry

end gasoline_tank_capacity_l362_362150


namespace bicycle_spokes_l362_362190

theorem bicycle_spokes (front_spokes : ℕ) (back_spokes : ℕ) 
  (h_front : front_spokes = 20) (h_back : back_spokes = 2 * front_spokes) :
  front_spokes + back_spokes = 60 :=
by
  rw [h_front, h_back]
  norm_num

end bicycle_spokes_l362_362190


namespace number_of_D_students_l362_362674

def total_students : ℕ := 800

def fraction_A : ℚ := 1 / 5
def fraction_B : ℚ := 1 / 4
def fraction_C : ℚ := 1 / 2

def students_A : ℕ := (fraction_A * total_students).to_nat
def students_B : ℕ := (fraction_B * total_students).to_nat
def students_C : ℕ := (fraction_C * total_students).to_nat

theorem number_of_D_students :
  total_students - (students_A + students_B + students_C) = 40 := by
  sorry

end number_of_D_students_l362_362674


namespace positive_real_solutions_l362_362245

noncomputable def x1 := (75 + Real.sqrt 5773) / 2
noncomputable def x2 := (-50 + Real.sqrt 2356) / 2

theorem positive_real_solutions :
  ∀ x : ℝ, 
  0 < x → 
  (1/2 * (4*x^2 - 1) = (x^2 - 75*x - 15) * (x^2 + 50*x + 10)) ↔ 
  (x = x1 ∨ x = x2) :=
by
  sorry

end positive_real_solutions_l362_362245


namespace number_of_primes_between_40_and_50_l362_362322

theorem number_of_primes_between_40_and_50 : 
  (finset.filter is_prime (finset.range' 41 10)).card = 3 :=
begin
  sorry
end

end number_of_primes_between_40_and_50_l362_362322


namespace arithmetic_seq_sum_2017_l362_362271

theorem arithmetic_seq_sum_2017 
  (S : ℕ → ℝ) 
  (a : ℕ → ℝ) 
  (a1 : a 1 = -2017) 
  (h1 : ∀ n : ℕ, S n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1))
  (h2 : (S 2014) / 2014 - (S 2008) / 2008 = 6) : 
  S 2017 = -2017 :=
by
  sorry

end arithmetic_seq_sum_2017_l362_362271


namespace major_axis_length_l362_362554

-- Definitions and assumptions based on the problem's conditions
structure Ellipse where
  a b : ℝ
  h : a > 0
  k : a > b
  l : b > 0
  eccentricity : ℝ
  bounds : (Real.sqrt 3 / 3) ≤ eccentricity ∧ eccentricity ≤ (Real.sqrt 2 / 2)

def intersects_line (e : Ellipse) : Prop :=
  let P Q : EuclideanSpace ℝ (Fin 2) := sorry -- placeholders for points of intersection
  let origin : EuclideanSpace ℝ (Fin 2) := EuclideanSpace.single 0 0
  (e.a^2 + e.b^2 > 1) ∧ 
  (P + Q = EuclideanSpace.single 0 1) ∧ -- P and Q on the line y = 1 - x 
  ((P - origin)⬝(Q - origin) = 0) -- OP ⊥ OQ

def major_axis_in_range (e : Ellipse) : ℝ :=
  2 * e.a

theorem major_axis_length (e : Ellipse) (h1 : intersects_line e) : 
  (\(2 * Real.sqrt 5) ≤ major_axis_in_range e ∧ major_axis_in_range e ≤ 2 * Real.sqrt 6) :=  
sorry


end major_axis_length_l362_362554


namespace maximize_profit_l362_362153

noncomputable def profit (x : ℕ) : ℝ :=
  if x ≤ 200 then
    (0.40 - 0.24) * 30 * x
  else if x ≤ 300 then
    (0.40 - 0.24) * 10 * 200 + (0.40 - 0.24) * 20 * x - (0.24 - 0.08) * 10 * (x - 200)
  else
    (0.40 - 0.24) * 10 * 200 + (0.40 - 0.24) * 20 * 300 - (0.24 - 0.08) * 10 * (x - 200) - (0.24 - 0.08) * 20 * (x - 300)

theorem maximize_profit : ∀ x : ℕ, 
  profit 300 = 1120 ∧ (∀ y : ℕ, profit y ≤ 1120) :=
by
  sorry

end maximize_profit_l362_362153


namespace geometric_sequence_general_formula_l362_362364

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ a1 q : ℝ, ∀ n : ℕ, a n = a1 * q ^ (n - 1)

variables (a : ℕ → ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := a 1 + a 3 = 10
def condition2 : Prop := a 4 + a 6 = 5 / 4

-- The final statement to prove
theorem geometric_sequence_general_formula (h : geometric_sequence a) (h1 : condition1 a) (h2 : condition2 a) :
  ∀ n : ℕ, a n = 2 ^ (4 - n) :=
sorry

end geometric_sequence_general_formula_l362_362364


namespace sqrt_product_l362_362097

theorem sqrt_product (a b : ℝ) (h1 : a = 49) (h2 : b = 25) : sqrt (a * sqrt b) = 7 * sqrt 5 :=
by sorry

end sqrt_product_l362_362097


namespace distance_from_O_to_plane_l362_362883

-- Definitions of points and distances based on the problem conditions
variable {A B C O : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O]

-- Given conditions as functions or constants
def radius_O : ℝ := 15
def AB : ℝ := 14
def BC : ℝ := 15
def CA : ℝ := 13

-- We need to prove the distance from O to the plane of triangle ABC is 15√15/8
theorem distance_from_O_to_plane (r_O : ℝ) (ab : ℝ) (bc : ℝ) (ca : ℝ) : 
    r_O = 15 → 
    ab = 14 → 
    bc = 15 → 
    ca = 13 → 
    ∃ (p q r : ℝ), p + q + r = 38 ∧ 
    (distance_from_O_to_plane_ABC radius_O AB BC CA = (15 * sqrt 15 / 8)) :=
by
    intros h1 h2 h3 h4
    use [15, 15, 8]
    sorry

end distance_from_O_to_plane_l362_362883


namespace five_digit_number_probability_l362_362664

-- Define a predicate for a five-digit number
def is_five_digit_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldr (λ x acc => x + acc) 0

-- Define the alternating sum of digits function
def alternating_sum_of_digits (n : ℕ) : ℤ :=
  let digits := n.digits 10
  digits.enum.foldr (λ ⟨i, x⟩ acc => if i % 2 = 0 then acc + x else acc - x) 0

-- The divisible by 11 rule
def divisible_by_11 (n : ℕ) : Prop :=
  alternating_sum_of_digits n % 11 = 0

-- Prove the main statement
theorem five_digit_number_probability :
  let S := { n : ℕ | is_five_digit_number n ∧ sum_of_digits n = 43 }
  let D := { n ∈ S | divisible_by_11 n }
  (S.finite.toFinset.card : ℚ) ≠ 0 →
  (D.finite.toFinset.card : ℚ) / (S.finite.toFinset.card : ℚ) = 1 / 5 :=
by
  sorry

end five_digit_number_probability_l362_362664


namespace min_max_fraction_l362_362733

theorem min_max_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  ∃ m M : ℝ, 
    (m = 0) ∧ 
    (M = 2) ∧ 
    (∀ z : ℝ, 
      ∃ (a b : ℝ), 
      a ≠ 0 ∧ b ≠ 0 ∧ z = (|a + b|^2) / (|a|^2 + |b|^2)) :=
begin
  sorry
end

end min_max_fraction_l362_362733


namespace oleg_max_composite_numbers_l362_362769

theorem oleg_max_composite_numbers : 
  ∃ (S : Finset ℕ), 
    (∀ (n ∈ S), n < 1500 ∧ ∃ p q, prime p ∧ prime q ∧ p ≠ q ∧ p * q = n) ∧ 
    (∀ (a b ∈ S), a ≠ b → gcd a b = 1) ∧ 
    S.card = 12 :=
sorry

end oleg_max_composite_numbers_l362_362769


namespace sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l362_362080

theorem sqrt_49_mul_sqrt_25_eq_7_sqrt_5 : (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l362_362080


namespace alice_lost_second_game_l362_362549

/-- Alice, Belle, and Cathy had an arm-wrestling contest. In each game, two girls wrestled, while the third rested.
After each game, the winner played the next game against the girl who had rested.
Given that Alice played 10 times, Belle played 15 times, and Cathy played 17 times; prove Alice lost the second game. --/

theorem alice_lost_second_game (alice_plays : ℕ) (belle_plays : ℕ) (cathy_plays : ℕ) :
  alice_plays = 10 → belle_plays = 15 → cathy_plays = 17 → 
  ∃ (lost_second_game : String), lost_second_game = "Alice" := by
  intros hA hB hC
  sorry

end alice_lost_second_game_l362_362549


namespace parallelogram_faces_not_unique_to_parallelepipeds_l362_362550

-- Definition of a parallelepiped
def is_parallelepiped (P : Type) [Polyhedron P] : Prop :=
  ∀ F ∈ faces P, is_parallelogram F

-- Definition that there exists a polyhedron other than a parallelepiped with all parallelogram faces
def exists_non_parallelepiped_with_parallelogram_faces : Prop :=
  ∃ Q : Type, [Polyhedron Q] ∧ 
  (∀ F ∈ faces Q, is_parallelogram F) ∧ 
  ¬ is_parallelepiped Q

-- The theorem
theorem parallelogram_faces_not_unique_to_parallelepipeds :
  exists_non_parallelepiped_with_parallelogram_faces :=
sorry

end parallelogram_faces_not_unique_to_parallelepipeds_l362_362550


namespace part1_geometric_sequence_part1_general_term_formula_part2_range_a1_l362_362217

noncomputable def a (n : ℕ) : ℝ :=
if n = 1 then 3 else (4 * a (n - 1) - 2) / (a (n - 1) + 1)

def b (n : ℕ) : ℝ :=
(2 - a n) / (a n - 1)

theorem part1_geometric_sequence : ∀ n : ℕ, n > 0 →
  b (n + 1) / b n = -2 / 3 :=
sorry

theorem part1_general_term_formula : ∀ n : ℕ, n > 0 →
  b n = -1 / 2 * ((-2 / 3) ^ (n - 1)) :=
sorry

theorem part2_range_a1 :
  (∀ n : ℕ, n > 0 → a n > a (n + 1)) →
    (1 / 5 < a 1 ∧ a 1 < 1) ∨ (a 1 > 2) :=
sorry

end part1_geometric_sequence_part1_general_term_formula_part2_range_a1_l362_362217


namespace measure_of_each_interior_angle_of_regular_octagon_l362_362499

theorem measure_of_each_interior_angle_of_regular_octagon 
  (n : ℕ) (h_n : n = 8) : 
  let sum_of_interior_angles := 180 * (n - 2) in
  let measure_of_interior_angle := sum_of_interior_angles / n in
  measure_of_interior_angle = 135 :=
by
  sorry

end measure_of_each_interior_angle_of_regular_octagon_l362_362499


namespace train_length_l362_362177

-- Definitions and conditions based on the problem
def time : ℝ := 28.997680185585153
def bridge_length : ℝ := 150
def train_speed : ℝ := 10

-- The theorem to prove
theorem train_length : (train_speed * time) - bridge_length = 139.97680185585153 :=
by
  sorry

end train_length_l362_362177


namespace sqrt_nested_l362_362060

theorem sqrt_nested : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_nested_l362_362060


namespace four_letter_words_with_A_l362_362309

theorem four_letter_words_with_A :
  let letters := ['A', 'B', 'C', 'D', 'E']
  in let total_4_letter_words := 5^4
  in let words_without_A := 4^4
  in total_4_letter_words - words_without_A = 369 := by
  sorry

end four_letter_words_with_A_l362_362309


namespace sqrt_49_times_sqrt_25_l362_362025

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 :=
by
  sorry

end sqrt_49_times_sqrt_25_l362_362025


namespace company_partition_l362_362676

open Set
open Classical

-- Define a graph structure
structure Graph (V : Type u) :=
  (adj : V → V → Prop)
  (symm : ∀ {v u}, adj v u → adj u v)

-- Define an unsociable group in the graph
def unsociable {V : Type u} (G : Graph V) (S : Set V) : Prop :=
  odd (card S) ∧ 
  card S ≥ 3 ∧ 
  (∀ v ∈ S, ∀ u ∈ S, v ≠ u → G.adj v u)

-- Define the assumptions
variable {V : Type u}
variable {G : Graph V}
variable (P : Set (Set V))
variable (enemy_pairs : ∀ S ∈ P, unsociable G S)
variable (mx_unsociable_groups : card P ≤ 2015)

-- The proof problem
theorem company_partition (G : Graph V) (no_of_unsociable : card P ≤ 2015) :
  ∃ partition : Fin 11 → Set V, 
    (∀ i j, (∀ v ∈ partition i, ∀ u ∈ partition j, G.adj v u → i ≠ j) ∧ (∀ i, (v, u) ∈ relation.partition v u partition i)) :=
sorry

end company_partition_l362_362676


namespace system_of_equations_solution_l362_362858

theorem system_of_equations_solution (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) :
  (x ^ log x * y ^ log y = 243) ∧ ((3 / log x) * x * y ^ log y = 1) →
  (x = 9 ∧ y = 3) ∨ (x = 3 ∧ y = 9) ∨
  (x = 1/9 ∧ y = 1/3) ∨ (x = 1/3 ∧ y = 1/9) :=
by
  sorry

end system_of_equations_solution_l362_362858


namespace ellipse_equation_oa2_plus_ob2_constant_l362_362684

-- Problem 1
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : (1 / a^2) + (9 / (4 * b^2)) = 1) (h4 : sqrt(a^2 - b^2) / a = 1 / 2) :
  (a = 2) ∧ (b = sqrt(3)) ∧ (∀ x y : ℝ, (x^2 / 4 + y^2 / 3 = 1 ↔ (x = 1) ∧ (y = 3 / 2))) := 
sorry

-- Problem 2
theorem oa2_plus_ob2_constant (n : ℝ) (h : abs(n) < sqrt(6)) :
  ∃ A B : ℝ×ℝ, (A.2 = sqrt(3) / 2 * A.1 + n) ∧ (B.2 = sqrt(3) / 2 * B.1 + n) ∧ 
  ((A.1 ^ 2) / 4 + (A.2 ^ 2) / 3 = 1) ∧ ((B.1 ^ 2) / 4 + (B.2 ^ 2) / 3 = 1) ∧ 
  ((A.1 ^ 2 + A.2 ^ 2 + B.1 ^ 2 + B.2 ^ 2) = 7) := 
sorry

end ellipse_equation_oa2_plus_ob2_constant_l362_362684


namespace quadratic_roots_and_T_range_l362_362402

theorem quadratic_roots_and_T_range
  (m : ℝ)
  (h1 : m ≥ -1)
  (x1 x2 : ℝ)
  (h2 : x1^2 + 2*(m-2)*x1 + (m^2 - 3*m + 3) = 0)
  (h3 : x2^2 + 2*(m-2)*x2 + (m^2 - 3*m + 3) = 0)
  (h4 : x1 ≠ x2)
  (h5 : x1^2 + x2^2 = 6) :
  m = (5 - Real.sqrt 17) / 2 ∧ (0 < ((m * x1) / (1 - x1) + (m * x2) / (1 - x2)) ∧ ((m * x1) / (1 - x1) + (m * x2) / (1 - x2)) ≤ 4 ∧ ((m * x1) / (1 - x1) + (m * x2) / (1 - x2)) ≠ 2) :=
by
  sorry

end quadratic_roots_and_T_range_l362_362402


namespace number_of_segments_after_iterations_length_of_segments_after_iterations_segments_and_length_l362_362546

theorem number_of_segments_after_iterations (n : ℕ) : 
  ∀ (a : ℕ), a = 16 → (2^a = 2^16) :=
by
  intros n h
  rw h
  rfl

theorem length_of_segments_after_iterations : 
  ∀ (a : ℕ), a = 16 → (1 / 3^a = 1 / 3^16) :=
by
  intros n h
  rw h
  rfl

theorem segments_and_length (a : ℕ) : 
  a = 16 → ∃ (num_segments : ℕ) (segment_length : ℚ), 
  num_segments = 2^16 ∧ segment_length = 1 / 3^16 :=
by
  intros h
  use 2^16, 1 / 3^16
  split
  { rw number_of_segments_after_iterations a
    exact nat.eq_refl 16
    exact a
    exact h
  }
  { rw length_of_segments_after_iterations a
    exact nat.eq_refl 16
    exact a
    exact h
  }

end number_of_segments_after_iterations_length_of_segments_after_iterations_segments_and_length_l362_362546


namespace four_letter_words_with_A_l362_362308

theorem four_letter_words_with_A :
  let letters := ['A', 'B', 'C', 'D', 'E']
  in let total_4_letter_words := 5^4
  in let words_without_A := 4^4
  in total_4_letter_words - words_without_A = 369 := by
  sorry

end four_letter_words_with_A_l362_362308


namespace chord_length_squared_l362_362210

noncomputable def square_of_chord_length 
  (r1 r2 R : ℝ) 
  (h_tangent: R > r1 + r2) -- radius of large circle is greater than the sum of smaller circles' radii
  (h_radius : r1 = 4 ∧ r2 = 5 ∧ R = 10) 
  : ℝ :=
  (4 * (R^2 - ((r1 * r2 + r1 * r1)^2) / ((r1 + r2)^2))) -- formula for the square length of chord

theorem chord_length_squared 
  (r1 r2 R : ℝ) 
  (h_tangent: R > r1 + r2)
  (h_radius : r1 = 4 ∧ r2 = 5 ∧ R = 10) 
  : square_of_chord_length r1 r2 R h_tangent h_radius = 26000 / 81 :=
by 
suffices square_of_chord_length r1 r2 R h_tangent h_radius 
:= sorry

end chord_length_squared_l362_362210


namespace sqrt_49_times_sqrt_25_eq_7sqrt5_l362_362074

theorem sqrt_49_times_sqrt_25_eq_7sqrt5 :
  (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  have h1 : Real.sqrt 25 = 5 := by sorry
  have h2 : 49 * 5 = 245 := by sorry
  have h3 : Real.sqrt 245 = 7 * Real.sqrt 5 := by sorry
  sorry

end sqrt_49_times_sqrt_25_eq_7sqrt5_l362_362074


namespace sqrt_mul_sqrt_l362_362103

theorem sqrt_mul_sqrt (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : sqrt (a * sqrt b) = sqrt a * sqrt (sqrt b) := by
  sorry

example : sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  have h1 : sqrt 25 = 5 := by 
    norm_num
  have h2 : sqrt (49 * 5) = sqrt 245 := by  
    unfold sqrt
  have h3 : sqrt 245 = 7 * sqrt 5 := by 
    norm_num
  rw [h1, h2, h3]
  norm_num

end sqrt_mul_sqrt_l362_362103


namespace extreme_value_B_l362_362223

noncomputable def is_extreme_value_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  deriv f x₀ = 0 ∧ ((∀ x < x₀, deriv f x < 0) ∨ (∀ x < x₀, deriv f x > 0)) ∧ ((∀ x > x₀, deriv f x > 0) ∨ (∀ x > x₀, deriv f x < 0))

def f_B (x : ℝ) : ℝ := -Real.cos x

theorem extreme_value_B : is_extreme_value_point f_B 0 :=
  sorry

end extreme_value_B_l362_362223


namespace sqrt_49_times_sqrt_25_l362_362024

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 :=
by
  sorry

end sqrt_49_times_sqrt_25_l362_362024


namespace range_of_x_l362_362280

theorem range_of_x (S : ℕ → ℕ) (a : ℕ → ℕ) (x : ℕ) :
  (∀ n, n ≥ 2 → S (n - 1) + S n = 2 * n^2 + 1) →
  S 0 = 0 →
  a 1 = x →
  (∀ n, a n ≤ a (n + 1)) →
  2 < x ∧ x < 3 := 
sorry

end range_of_x_l362_362280


namespace sally_has_more_cards_l362_362430

-- Definitions and conditions
def initial_sally : ℕ := 27
def new_dan : ℕ := 41
def bought_sally : ℕ := 20
def traded_sally (x : ℕ) : ℕ := x

-- Statement: Prove that Sally has 6 - x more cards than Dan
theorem sally_has_more_cards (x : ℕ) : (initial_sally + bought_sally - traded_sally(x)) - new_dan = 6 - x :=
by
  -- Proof goes here
  sorry

end sally_has_more_cards_l362_362430


namespace num_pos_int_x_l362_362606

theorem num_pos_int_x (x : ℕ) : 
  (30 < x^2 + 5 * x + 10) ∧ (x^2 + 5 * x + 10 < 60) ↔ x = 3 ∨ x = 4 ∨ x = 5 := 
sorry

end num_pos_int_x_l362_362606


namespace positive_integer_solutions_l362_362651

theorem positive_integer_solutions (y : ℕ) (hy : 0 < y) : ∃ n : ℕ, n = 10 ∧ 
  (∀ y : ℕ, (5 < 2 * y + 4) → (y ≤ 10)) → 
  (∃ k : fin n, 5 < 2 * (k + 1) + 4) :=
by
  sorry

end positive_integer_solutions_l362_362651


namespace distinct_flavors_l362_362603

theorem distinct_flavors {b o : ℕ} (hb : b = 5) (ho : o = 4) :
  (count_distinct_flavors b o) = 17 := sorry

def count_distinct_flavors (b o : ℕ) : ℕ :=
  let xs := (finset.range (b + 1)).product (finset.range (o + 1))
  let ratios := xs.map (λ (p : ℕ × ℕ), (p.fst, p.snd))
  let distinct_ratios := ratios.to_finset.erase ⟨0, 0⟩
  let gcd := distinct_ratios.map (λ (p : ℕ × ℕ), nat.gcd p.fst p.snd)
  let normalized_ratios := distinct_ratios.map (λ (p : ℕ × ℕ), (p.fst / gcd p.fst p.snd, p.snd / gcd p.fst p.snd))
  normalized_ratios.to_finset.card

end distinct_flavors_l362_362603


namespace min_distance_MN_l362_362736

theorem min_distance_MN :
  let E := { p : ℝ × ℝ // (p.1^2 / 3) + p.2^2 = 1 } in
  let F : ℝ × ℝ := (-(Real.sqrt 2), 0) in
  let A B : ℝ × ℝ in
  let l : (ℝ × ℝ) → Prop :=
    fun p => ∃ m : ℝ, 0 < m ∧ p.2 = m * (p.1 + Real.sqrt 2) in
  let M N : ℝ × ℝ in
  ∀ A B M N,
    A ∈ E → B ∈ E → l A → l B →
    let AM : (ℝ × ℝ) → Prop :=
      fun q => AM ∧ q.2 = -1 / l A * (q.1 - A.1) in
    let BN : (ℝ × ℝ) → Prop :=
      fun q => BN ∧ q.2 = -1 / l B * (q.1 - B.1) in
    AM M → BN N →
    ∀ α : ℝ,
      let AB := (Real.sqrt (3 - 2 * Real.cos α^2)) in
      let MN := AB / Real.cos α in
      MN ≥ Real.sqrt 6 := sorry

end min_distance_MN_l362_362736


namespace construct_one_degree_l362_362621

theorem construct_one_degree (theta : ℝ) (h : theta = 19) : 1 = 19 * theta - 360 :=
by
  -- Proof here will be filled
  sorry

end construct_one_degree_l362_362621


namespace tangent_line_at_e_l362_362874

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_at_e (x : ℝ) (e : ℝ) (y : ℝ) : 
  (f(e) = e) ∧ (x = 2 * e - e) →
  y = 2 * x - e :=
by
  sorry

end tangent_line_at_e_l362_362874


namespace boat_speed_is_consistent_l362_362946

theorem boat_speed_is_consistent :
  ∃ (speed : ℤ), (∀ length width : ℤ, 
    (length = width) ∧ 
    (length * width = 100) ∧
    (length / 2 = speed) ∧ 
    (width / 0.5 = speed)) ∧ 
    (speed = 5) :=
begin
  sorry
end

end boat_speed_is_consistent_l362_362946


namespace minimum_x_condition_l362_362940

theorem minimum_x_condition (x y : ℝ) (hxy_pos : 0 < x ∧ 0 < y) (h : x - 2 * y = (x + 16 * y) / (2 * x * y)) : 
  x ≥ 4 :=
sorry

end minimum_x_condition_l362_362940


namespace ferry_q_longer_journey_l362_362122

-- Define the conditions as constants
def ferry_p_speed : ℝ := 6  -- km/h without current
def ferry_p_current_reduction : ℝ := 1  -- km/h
def ferry_p_travel_time : ℝ := 3  -- hours

def ferry_q_speed_diff : ℝ := 3  -- km/h faster than ferry p without current
def ferry_q_current_reduction : ℝ := 0.5  -- km/h, half the current ferry p faces
def ferry_q_distance_multiplier : ℝ := 2  -- distance multiplier

-- Calculations
def ferry_p_effective_speed : ℝ := ferry_p_speed - ferry_p_current_reduction
def ferry_p_distance : ℝ := ferry_p_effective_speed * ferry_p_travel_time

def ferry_q_distance : ℝ := ferry_q_distance_multiplier * ferry_p_distance
def ferry_q_speed : ℝ := ferry_p_speed + ferry_q_speed_diff
def ferry_q_effective_speed : ℝ := ferry_q_speed - ferry_q_current_reduction
def ferry_q_travel_time : ℝ := ferry_q_distance / ferry_q_effective_speed

def time_difference : ℝ := ferry_q_travel_time - ferry_p_travel_time

-- Lean 4 proof statement
theorem ferry_q_longer_journey : time_difference = 0.5294 :=
by sorry

end ferry_q_longer_journey_l362_362122


namespace capital_contribution_A_l362_362941

theorem capital_contribution_A (P C : ℚ) (x : ℚ) : 
  (B_profit_share : ℚ) (B_months : ℕ) (A_months : ℕ) 
  (profit_ratio : ℚ) (capital_ratio : ℚ)
  (B_profit_share = 2 / 3) 
  (A_months = 15) 
  (B_months = 10) 
  (profit_ratio = 1 / 2) 
  (capital_ratio = (15 * x) / (10 * (1 - x))) 
  (profit_ratio = capital_ratio) : 
  x = 1 / 4 := 
sorry

end capital_contribution_A_l362_362941


namespace sqrt_49_times_sqrt25_eq_7_sqrt5_l362_362037

-- Definitions based on conditions
def sqrt_25 := Real.sqrt 25
def sqrt_49 := Real.sqrt 49

theorem sqrt_49_times_sqrt25_eq_7_sqrt5 :
  Real.sqrt (49 * sqrt_25) = 7 * Real.sqrt 5 := by
sorry

end sqrt_49_times_sqrt25_eq_7_sqrt5_l362_362037


namespace sqrt_49_times_sqrt_25_eq_7sqrt5_l362_362072

theorem sqrt_49_times_sqrt_25_eq_7sqrt5 :
  (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  have h1 : Real.sqrt 25 = 5 := by sorry
  have h2 : 49 * 5 = 245 := by sorry
  have h3 : Real.sqrt 245 = 7 * Real.sqrt 5 := by sorry
  sorry

end sqrt_49_times_sqrt_25_eq_7sqrt5_l362_362072


namespace sculpture_height_l362_362573

theorem sculpture_height (base_height : ℕ) (total_height_ft : ℝ) (inches_per_foot : ℕ) 
  (h1 : base_height = 8) (h2 : total_height_ft = 3.5) (h3 : inches_per_foot = 12) : 
  (total_height_ft * inches_per_foot - base_height) = 34 := 
by
  sorry

end sculpture_height_l362_362573


namespace fraction_of_dehydrated_men_did_not_finish_l362_362894

theorem fraction_of_dehydrated_men_did_not_finish (total_men : ℕ)
  (tripped_fraction : ℚ) (dehydrated_fraction : ℚ) (finished_men : ℕ) 
  (tripped_men : ℕ) (remaining_men : ℕ) (dehydrated_men : ℕ) (did_not_finish_men : ℕ) 
  (dehydrated_did_not_finish_men : ℕ) :
  total_men = 80 → 
  tripped_fraction = 1/4 → 
  dehydrated_fraction = 2/3 → 
  finished_men = 52 → 
  tripped_men = tripped_fraction * total_men → 
  remaining_men = total_men - tripped_men → 
  dehydrated_men = dehydrated_fraction * remaining_men → 
  did_not_finish_men = total_men - finished_men → 
  dehydrated_did_not_finish_men = did_not_finish_men - tripped_men → 
  dehydrated_did_not_finish_men / dehydrated_men = 1/5 := 
by {
  intros,
  sorry
}

end fraction_of_dehydrated_men_did_not_finish_l362_362894


namespace function_properties_and_extrema_l362_362452

def f (x : ℝ) (a b : ℝ) : ℝ := 4 * x^3 + a * x^2 + b * x + 5

theorem function_properties_and_extrema :
  (∀ x, (f x -3 -18) isIncreasingOn Iio (-1) ∧ (f x -3 -18) isIncreasingOn Ioi (3/2) ∧ (f x -3 -18) isDecreasingOn Ioo (-1) (3/2)) →
  (∀ x, (f x -3 -18 = 4 * x^3 - 3 * x^2 - 18 * x + 5)) ∧
  (∀ x, x ∈ (Icc (-1 : ℝ) 2) → max ((f x -3 -18) (-1)) ((f x -3 -18) 2) = 10) ∧
  (∀ x, x ∈ (Icc (-1 : ℝ) 2) → min ((f x -3 -18) (-1)) ((f x -3 -18) (1.5)) = -25/4 ) := 
by
  sorry

end function_properties_and_extrema_l362_362452


namespace triangle_identity_triangle_perimeter_l362_362718

theorem triangle_identity 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : sin C * sin (A - B) = sin B * sin (C - A)) :
  2 * a^2 = b^2 + c^2 :=
sorry

theorem triangle_perimeter 
  (a b c : ℝ) 
  (A : ℝ) 
  (h1 : 2 * a^2 = b^2 + c^2) 
  (ha : a = 5) 
  (h_cosA : cos A = 25 / 31) :
  a + b + c = 14 :=
sorry

end triangle_identity_triangle_perimeter_l362_362718


namespace sqrt_product_l362_362101

theorem sqrt_product (a b : ℝ) (h1 : a = 49) (h2 : b = 25) : sqrt (a * sqrt b) = 7 * sqrt 5 :=
by sorry

end sqrt_product_l362_362101


namespace count_distinct_real_solutions_lt_1000_l362_362700

def P (x : ℝ) : ℝ := x^3 - 3 * x^2 + 3

theorem count_distinct_real_solutions_lt_1000 :
  (∃ n : ℕ, n < 1000 ∧ ¬(∃ a b : ℕ, a > 0 ∧ b > 0 ∧
    (∃ x : ℝ, (P^[a] x) = (P^[b] x) ∧ (number_of_distinct_real_solutions (P^[a] x = P^[b] x)) = n))) ↔ (984) := 
sorry -- Proof will be provided separately

end count_distinct_real_solutions_lt_1000_l362_362700


namespace knight_tour_impossible_l362_362419

theorem knight_tour_impossible 
  (knight_moves : ∀ pos: (ℕ × ℕ), (ℕ × ℕ) → Prop)
  (chessboard : Fin 8 × Fin 8)
  (white_squares black_squares : Fin 8 × Fin 8 → Prop)
  (pawn_pos : Fin 8 × Fin 8)
  (empty_squares : List (Fin 8 × Fin 8))
  (closed_tour : Prop) :
  (∀ pos : Fin 8 × Fin 8, knight_moves pos pos → white_squares pos ∧ black_squares pos) →
  closed_tour →
  ((List.length empty_squares) = 63) →
  (63 % 2 = 1) →
  ¬closed_tour := 
by
  intros
  sorry

end knight_tour_impossible_l362_362419


namespace monkey_reaches_top_in_17_minutes_l362_362965

def monkey_climbing_time (pole_height ascent descent: ℕ) (step: ℕ → ℕ) : ℕ :=
  let rec time (height remaining min: ℕ) :=
    if remaining ≤ height then
      min
    else
      let new_height := remaining + step min
      time height (new_height) (min + 1)
  time pole_height 0 0

theorem monkey_reaches_top_in_17_minutes :
  monkey_climbing_time 10 2 1 (λ min => if min % 2 = 0 then 2 else -1) = 17 := sorry

end monkey_reaches_top_in_17_minutes_l362_362965


namespace problem1_problem2_l362_362268

-- Problem 1: If k = 3, prove that |BC| / |AC| = 3

theorem problem1 (k : ℝ) (h : k = 3) :
  let l := λ x, k * (x - 1),
      A := (1, 0),
      B := (0, -k),
      C := (k / (k - 1), k / (k - 1)) in
  (|B.1 - C.1| / |A.1 - C.1| = 3) :=
by
  sorry

-- Problem 2: If |BC| = 2|AC|, prove the equation of line l is 2x - y - 2 = 0 or 2x + y - 2 = 0

theorem problem2 (k : ℝ) (l := λ x, k * (x - 1)) (h : |(0 : ℝ) - k / (k - 1)| = 2 * |1 - k / (k - 1)|) :
  let A := (1, 0),
      B := (0, -k),
      C := (k / (k - 1), k / (k - 1)),
      line_eq1 := (2 * x - y = 2),
      line_eq2 := (2 * x + y = 2) in
  (l = line_eq1 ∨ l = line_eq2) :=
by
  sorry

end problem1_problem2_l362_362268


namespace distinct_point_count_l362_362461

theorem distinct_point_count : 
  (∃! (x y : ℝ), (x + y = 5 ∧ 2 * x - 3 * y = -5) ∨ 
                  (x + y = 5 ∧ 3 * x + 2 * y = 12) ∨ 
                  (x - y = -1 ∧ 2 * x - 3 * y = -5) ∨ 
                  (x - y = -1 ∧ x + y = 5) ∧ 
                  (x + y = 5 ∧ 2 * x - 3 * y = -5) ∧ 
                  (x + y = 5 ∧ 3 * x + 2 * y = 12) ∧ 
                  (2 * x - 3 * y = -5 ∧ 3 * x + 2 * y = 12) ) := 
begin
  -- proof omitted
  sorry
end

end distinct_point_count_l362_362461


namespace max_oleg_composite_numbers_l362_362756

/-- Oleg's list of composite numbers less than 1500 and pairwise coprime. --/
def oleg_composite_numbers (numbers : List ℕ) : Prop :=
  ∀ n ∈ numbers, Nat.isComposite n ∧ n < 1500 ∧ (∀ m ∈ numbers, n ≠ m → Nat.gcd n m = 1)

/-- Maximum number of such numbers that Oleg could write. --/
theorem max_oleg_composite_numbers : ∃ numbers : List ℕ, oleg_composite_numbers numbers ∧ numbers.length = 12 := 
sorry

end max_oleg_composite_numbers_l362_362756


namespace highest_locker_number_labeled_l362_362557

theorem highest_locker_number_labeled :
  ∃ n : ℕ, highest_locker_number 294.94 = 3963 :=
sorry

end highest_locker_number_labeled_l362_362557


namespace simplify_expression_l362_362837

theorem simplify_expression (a b : ℚ) (ha : a = -1) (hb : b = 1/4) : 
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_expression_l362_362837


namespace acute_angle_one_circle_determine_31_deg_angle_l362_362932

-- Part (a)
def is_acute (∠AOB: Angle) : Prop :=
  let circle_center_O := Circle (center := O)
  circle_center_O.intersects (∠AOB.side OA) ∧ circle_center_O.intersects (∠AOB.side OB)

theorem acute_angle_one_circle (∠AOB: Angle) : is_acute ∠AOB → angle_acute ∠AOB :=
sorry

-- Part (b)
def determines_31_deg (∠AOB: Angle) : Prop :=
  let circle_center_O := Circle (center := O)
  let length_AB := length (circle_center_O.chord AB)
  let num_chords := 360 / 31
  circle_center_O.lay_off_chords (from := A) (length := length_AB) (num := num_chords)
  
theorem determine_31_deg_angle (∠AOB: Angle) : determines_31_deg ∠AOB → angle_31_deg ∠AOB :=
sorry

end acute_angle_one_circle_determine_31_deg_angle_l362_362932


namespace triangle_isosceles_or_right_angled_l362_362834

theorem triangle_isosceles_or_right_angled
  (β γ : ℝ)
  (h : Real.tan β * Real.sin γ ^ 2 = Real.tan γ * Real.sin β ^ 2) :
  (β = γ) ∨ (β + γ = π / 2) :=
sorry

end triangle_isosceles_or_right_angled_l362_362834


namespace sum_of_digits_perfect_square_eq_1991_l362_362572

-- Let's state the problem formally in Lean
theorem sum_of_digits_perfect_square_eq_1991 (n : ℕ) :
  let sum_digits (m : ℕ) := (m.digits 10).sum in
  sum_digits (n * n) ≠ 1991 :=
by
  sorry

end sum_of_digits_perfect_square_eq_1991_l362_362572


namespace total_ways_to_choose_president_and_vice_president_of_opposite_genders_l362_362952

theorem total_ways_to_choose_president_and_vice_president_of_opposite_genders (n m : ℕ) (Hn : n = 12) (Hm : m = 12) : 
  12 * 12 + 12 * 12 = 288 :=
by
  rw [Hn, Hm]
  sorry

end total_ways_to_choose_president_and_vice_president_of_opposite_genders_l362_362952


namespace stratified_sampling_l362_362355

-- Definition of the given variables and conditions
def total_students_grade10 : ℕ := 30
def total_students_grade11 : ℕ := 40
def selected_students_grade11 : ℕ := 8

-- Implementation of the stratified sampling proportion requirement
theorem stratified_sampling (x : ℕ) (hx : (x : ℚ) / total_students_grade10 = (selected_students_grade11 : ℚ) / total_students_grade11) :
  x = 6 :=
by
  sorry

end stratified_sampling_l362_362355


namespace carla_water_drank_l362_362214

theorem carla_water_drank (W S : ℝ) (h1 : W + S = 54) (h2 : S = 3 * W - 6) : W = 15 :=
by
  sorry

end carla_water_drank_l362_362214


namespace find_C_coordinates_l362_362820

-- Define the points A, B, and D as given in the problem
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 7, y := 1 }
def B : Point := { x := 5, y := -3 }
def D : Point := { x := 5, y := 1 }

-- Define the theorem statement
theorem find_C_coordinates (C : Point) (h1: A.x = A.x) (h2: B.x = D.x) (h3: D.x = C.x) 
  (h4: ∠ABC = 90) (h5: D.y = A.y) : C = { x := 5, y := 5 } := 
sorry

end find_C_coordinates_l362_362820


namespace exists_point_E_bisecting_and_equal_angles_l362_362262

-- Definitions for the given conditions
variables {A B C D H M N S T E : Point}
variables (circle : Circle) (quadrilateral : Quadrilateral)

-- Conditions definitions
def inscribed (quadrilateral : Quadrilateral) (circle : Circle) : Prop :=
  ∀ (A B C D: Point), Quadrilateral.is_cyclic ⟨A, B, C, D⟩ circle

def diagonals_perpendicular (A B C D H : Point) : Prop :=
  is_midpoint H A C ∧ is_midpoint H B D ∧ is_perpendicular A C B D

def midpoints_defined (B C D H M N : Point) : Prop :=
  is_midpoint M B C ∧ is_midpoint N C D

def rays_intersect (H M N S T A B D : Point) : Prop :=
  ray_through H M intersects_segment AD S ∧ ray_through H N intersects_segment AB T

-- Theorem statement
theorem exists_point_E_bisecting_and_equal_angles
  (h_inscribed : inscribed quadrilateral circle)
  (h_perpendicular : diagonals_perpendicular A B C D H)
  (h_midpoints : midpoints_defined B C D H M N)
  (h_intersections : rays_intersect H M N S T A B D) :
  ∃ E : Point, 
    (ray_through E H bisects_angle ∠BES) ∧ 
    (ray_through E H bisects_angle ∠TED) ∧ 
    ∠BEN = ∠MED :=
  sorry

end exists_point_E_bisecting_and_equal_angles_l362_362262


namespace f_is_monotonically_decreasing_l362_362291

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + π / 6) + Real.cos (2 * x)

theorem f_is_monotonically_decreasing : 
  ∃ (a b : ℝ), a = π / 12 ∧ b = 7 * π / 12 ∧ ∀ x y : ℝ, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x :=
by
  sorry

end f_is_monotonically_decreasing_l362_362291


namespace weekly_earnings_l362_362381

-- Definition of the conditions
def hourly_rate : ℕ := 20
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 4

-- Theorem that conforms to the problem statement
theorem weekly_earnings : hourly_rate * hours_per_day * days_per_week = 640 := by
  sorry

end weekly_earnings_l362_362381


namespace pizzas_in_park_l362_362385

-- Define the conditions and the proof problem
def pizza_cost : ℕ := 12
def delivery_charge : ℕ := 2
def park_distance : ℕ := 100  -- in meters
def building_distance : ℕ := 2000  -- in meters
def pizzas_delivered_to_building : ℕ := 2
def total_payment_received : ℕ := 64

-- Prove the number of pizzas delivered in the park
theorem pizzas_in_park : (64 - (pizzas_delivered_to_building * pizza_cost + delivery_charge)) / pizza_cost = 3 :=
by
  sorry -- Proof not required

end pizzas_in_park_l362_362385


namespace product_result_l362_362240
-- Importing the broad necessary library for math

-- Define the function representing the product
def product_of_terms : ℚ := ∏ k in finset.range (51 - 3 + 1), (1 - 1 / (k + 3))

-- The main theorem to be proved
theorem product_result :
  product_of_terms = 2 / 51 :=
sorry

end product_result_l362_362240


namespace bicycle_total_spokes_l362_362188

open Nat

def front_wheel_spokes : Nat := 20

def back_wheel_spokes : Nat := 2 * front_wheel_spokes

def total_spokes : Nat := front_wheel_spokes + back_wheel_spokes

theorem bicycle_total_spokes : total_spokes = 60 := by
  have h1 : front_wheel_spokes = 20 := rfl
  have h2 : back_wheel_spokes = 2 * front_wheel_spokes := rfl
  have h3 : back_wheel_spokes = 40 := by rw [h1, mul_comm, mul_one]
  show total_spokes = 60 from calc
    total_spokes = front_wheel_spokes + back_wheel_spokes := rfl
    ... = 20 + 40 := by rw [h1, h3]
    ... = 60 := by rfl

end bicycle_total_spokes_l362_362188


namespace simplify_evaluate_expression_l362_362431

noncomputable def x := Real.sqrt 12 + Real.sqrt 5 ^ 0 - (1 / 2) ^ (-1)

theorem simplify_evaluate_expression :
  ( (1 / (x + 1) + 1 / (x^2 - 1)) / (x / (x - 1)) ) = (Real.sqrt 3 / 6) :=
by
  -- Proof goes here
  sorry

end simplify_evaluate_expression_l362_362431


namespace sqrt_nested_l362_362054

theorem sqrt_nested : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_nested_l362_362054


namespace find_prime_number_l362_362460

open Nat

-- Definition of the problem in Lean
theorem find_prime_number :
  ∃ n : ℕ, Prime n ∧ 30 < n ∧ n < 40 ∧ n % 9 = 7 ∧ n = 43 :=
by
  sorry

end find_prime_number_l362_362460


namespace find_total_worth_of_stock_l362_362923

theorem find_total_worth_of_stock (X : ℝ)
  (h1 : 0.20 * X * 0.10 = 0.02 * X)
  (h2 : 0.80 * X * 0.05 = 0.04 * X)
  (h3 : 0.04 * X - 0.02 * X = 200) :
  X = 10000 :=
sorry

end find_total_worth_of_stock_l362_362923


namespace slope_of_horizontal_line_l362_362467

theorem slope_of_horizontal_line : 
  ∀ x y : ℝ, y + 3 = 0 → ∀ Δx : ℝ, Δx ≠ 0 → (0 : ℝ) / Δx = 0 :=
by
  intros x y h Δx hΔx
  have hy : y = -3 := by linarith
  have hΔy : 0 = 0 := rfl
  rw [hΔy] at *
  rw mul_zero 0
  apply rfl

end slope_of_horizontal_line_l362_362467


namespace cube_surface_area_150_of_volume_125_l362_362892

def volume (s : ℝ) : ℝ := s^3

def surface_area (s : ℝ) : ℝ := 6 * s^2

theorem cube_surface_area_150_of_volume_125 :
  ∀ (s : ℝ), volume s = 125 → surface_area s = 150 :=
by 
  intros s hs
  sorry

end cube_surface_area_150_of_volume_125_l362_362892


namespace sqrt_mul_sqrt_l362_362112

theorem sqrt_mul_sqrt (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : sqrt (a * sqrt b) = sqrt a * sqrt (sqrt b) := by
  sorry

example : sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  have h1 : sqrt 25 = 5 := by 
    norm_num
  have h2 : sqrt (49 * 5) = sqrt 245 := by  
    unfold sqrt
  have h3 : sqrt 245 = 7 * sqrt 5 := by 
    norm_num
  rw [h1, h2, h3]
  norm_num

end sqrt_mul_sqrt_l362_362112


namespace symmetric_axis_transformed_graph_l362_362578

theorem symmetric_axis_transformed_graph :
  ∀ x, sin (4 * (x - π / 4) / 2 - π / 6) = sin (2 * x - π / 12)
  → x = π / 3 :=
by
  sorry

end symmetric_axis_transformed_graph_l362_362578


namespace max_composite_numbers_l362_362793

-- Definitions and conditions
def is_composite (n : ℕ) : Prop := 2 < n ∧ ∃ d, d ∣ n ∧ 1 < d ∧ d < n

def less_than_1500 (n : ℕ) : Prop := n < 1500

def gcd_is_one (a b : ℕ) : Prop := Int.gcd a b = 1

-- The problem statement
theorem max_composite_numbers (numbers : List ℕ) (h_composite : ∀ n ∈ numbers, is_composite n) 
  (h_less_than_1500 : ∀ n ∈ numbers, less_than_1500 n)
  (h_pairwise_gcd_is_one : (numbers.Pairwise gcd_is_one)) : numbers.length ≤ 12 := 
  sorry

end max_composite_numbers_l362_362793


namespace mr_blue_expected_rose_petals_l362_362748

def mr_blue_flower_bed_rose_petals (length_paces : ℕ) (width_paces : ℕ) (pace_length_ft : ℝ) (petals_per_sqft : ℝ) : ℝ :=
  let length_ft := length_paces * pace_length_ft
  let width_ft := width_paces * pace_length_ft
  let area_sqft := length_ft * width_ft
  area_sqft * petals_per_sqft

theorem mr_blue_expected_rose_petals :
  mr_blue_flower_bed_rose_petals 18 24 1.5 0.4 = 388.8 :=
by
  simp [mr_blue_flower_bed_rose_petals]
  norm_num

end mr_blue_expected_rose_petals_l362_362748


namespace green_more_than_red_l362_362520

def red_peaches : ℕ := 7
def green_peaches : ℕ := 8

theorem green_more_than_red : green_peaches - red_peaches = 1 := by
  sorry

end green_more_than_red_l362_362520


namespace value_range_of_cos_sin_square_add_2_l362_362471

theorem value_range_of_cos_sin_square_add_2 : 
  ∀ x : ℝ, (cos x - (sin x)^2 + 2) ∈ Set.Icc (3 / 4 : ℝ) 3 :=
by
  sorry

end value_range_of_cos_sin_square_add_2_l362_362471


namespace percent_greater_than_z_l362_362884

variable {R : Type} [LinearOrderedField R]

variables (w x y z : R)

theorem percent_greater_than_z (h1 : x = 1.2 * y) (h2 : y = 1.2 * z) (h3 : w = 0.8 * x) :
  w = 1.152 * z :=
by
  have hx : x = 1.44 * z := by linarith [h1, h2]
  have hw : w = 0.8 * (1.44 * z) := by rw [h3, hx]
  linarith [hw]

end percent_greater_than_z_l362_362884


namespace sheena_weeks_to_complete_dresses_l362_362833

/- Sheena is sewing the bridesmaid's dresses for her sister's wedding.
There are 7 bridesmaids in the wedding.
Each bridesmaid's dress takes a different number of hours to sew due to different styles and sizes.
The hours needed to sew the bridesmaid's dresses are as follows: 15 hours, 18 hours, 20 hours, 22 hours, 24 hours, 26 hours, and 28 hours.
If Sheena sews the dresses 5 hours each week, prove that it will take her 31 weeks to complete all the dresses. -/

def bridesmaid_hours : List ℕ := [15, 18, 20, 22, 24, 26, 28]

def total_hours_needed (hours : List ℕ) : ℕ :=
  hours.sum

def weeks_needed (total_hours : ℕ) (hours_per_week : ℕ) : ℕ :=
  (total_hours + hours_per_week - 1) / hours_per_week

theorem sheena_weeks_to_complete_dresses :
  weeks_needed (total_hours_needed bridesmaid_hours) 5 = 31 := by
  sorry

end sheena_weeks_to_complete_dresses_l362_362833


namespace miles_difference_l362_362587

-- Defining the gas consumption rate
def gas_consumption_rate : ℝ := 4

-- Defining the miles driven today
def miles_today : ℝ := 400

-- Defining the total gas consumption for both days
def total_gas_consumption : ℝ := 4000

-- Defining the gas consumed today
def gas_consumed_today : ℝ := miles_today * gas_consumption_rate

-- Defining the gas consumed tomorrow
def gas_consumed_tomorrow := total_gas_consumption - gas_consumed_today

-- Defining the miles driven tomorrow
def miles_tomorrow := gas_consumed_tomorrow / gas_consumption_rate

-- The final theorem to prove
theorem miles_difference : (miles_tomorrow - miles_today) = 200 := 
by
    simp [gas_consumption_rate, miles_today, total_gas_consumption, gas_consumed_today, gas_consumed_tomorrow, miles_tomorrow]
    sorry

end miles_difference_l362_362587


namespace max_composite_numbers_l362_362813
open Nat

theorem max_composite_numbers : 
  ∃ X : Finset Nat, 
  (∀ x ∈ X, x < 1500 ∧ ¬Prime x) ∧ 
  (∀ x y ∈ X, x ≠ y → gcd x y = 1) ∧ 
  X.card = 12 := 
sorry

end max_composite_numbers_l362_362813


namespace sqrt_49_times_sqrt25_eq_7_sqrt5_l362_362039

-- Definitions based on conditions
def sqrt_25 := Real.sqrt 25
def sqrt_49 := Real.sqrt 49

theorem sqrt_49_times_sqrt25_eq_7_sqrt5 :
  Real.sqrt (49 * sqrt_25) = 7 * Real.sqrt 5 := by
sorry

end sqrt_49_times_sqrt25_eq_7_sqrt5_l362_362039


namespace correct_operation_l362_362918

theorem correct_operation :
  (∀ (a : ℤ), 3 * a + 2 * a ≠ 5 * a ^ 2) ∧
  (∀ (a : ℤ), a ^ 6 / a ^ 2 ≠ a ^ 3) ∧
  (∀ (a : ℤ), (-3 * a ^ 3) ^ 2 = 9 * a ^ 6) ∧
  (∀ (a : ℤ), (a + 2) ^ 2 ≠ a ^ 2 + 4) := 
by
  sorry

end correct_operation_l362_362918


namespace problem_part1_problem_part2_l362_362145

variables (a b c d : ℕ)

def circledast (a b : ℕ) : ℕ :=
if a = b then a else if b = 0 then 2 * a else sorry  -- This definition does not cover all cases but suffices for translation.

axiom circledast_self (a : ℕ) : circledast a a = a
axiom circledast_zero (a : ℕ) : circledast a 0 = 2 * a
axiom circledast_distributive (a b c d : ℕ) : circledast (a + c) (b + d) = circledast a b + circledast c d

theorem problem_part1 : circledast (2 + 3) (0 + 3) = 7 :=
by {
  rw circledast_distributive,
  rw circledast_zero 2,
  rw circledast_self 3,
  norm_num,
  sorry
}

theorem problem_part2 : circledast 1024 48 = 2000 :=
by {
  have h : 1024 = 976 + 48 := by norm_num,
  rw [h, circledast_distributive, circledast_zero 976, circledast_self 48],
  norm_num,
  sorry
}

end problem_part1_problem_part2_l362_362145


namespace sours_ratio_l362_362832

variable (cherry lemon orange : ℕ)

def total_sours (cherry lemon orange : ℕ) : Prop :=
  cherry + lemon + orange = 96

def orange_sours_constraint (orange : ℕ) : Prop :=
  orange = 0.25 * 96

def correct_ratio (cherry lemon : ℕ) : Prop :=
  (cherry / 8) = 4 ∧ (lemon / 8) = 5

theorem sours_ratio (h1 : cherry = 32)
                    (h2 : total_sours cherry lemon orange)
                    (h3 : orange_sours_constraint orange) :
                    correct_ratio cherry lemon :=
by
  sorry

end sours_ratio_l362_362832


namespace unique_solution_l362_362242

def one_digit_divisors := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def four_digit_numbers := Finset.Icc 1000 9999

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def one_digit_divisor_count (n : ℕ) : Prop := 
  (Finset.filter (λ d, d ∈ one_digit_divisors ∧ n % d = 0) one_digit_divisors).card = 9

def four_digit_divisor_count (n : ℕ) : Prop := 
  (Finset.filter (λ d, is_four_digit d ∧ n % d = 0) four_digit_numbers).card = 5

theorem unique_solution :
  ∃! (n : ℕ), is_four_digit n ∧ one_digit_divisor_count n ∧ four_digit_divisor_count n :=
begin
  use 5040,
  split,
  { unfold is_four_digit one_digit_divisor_count four_digit_divisor_count,
    split,
    exact ⟨by norm_num, by norm_num⟩,
    split,
    { sorry },  -- Proof that 5040 has exactly 9 one-digit divisors
    { sorry }   -- Proof that 5040 has exactly 5 four-digit divisors
  },
  { intros y hy,
    have h₁ : is_four_digit y := hy.1,
    have h₂ : one_digit_divisor_count y := hy.2.1,
    have h₃ : four_digit_divisor_count y := hy.2.2,
    sorry,  -- Proof that any other number satisfying these conditions must be 5040
  }
end

end unique_solution_l362_362242


namespace taxi_fare_l362_362470

theorem taxi_fare (x : ℝ) (H_start : 6 ≤ 13.2)
                  (H_fare : 13.2 = 6 + 2.4 * (x - 3)) :
                  6 + 2.4 * (x - 3) = 13.2 :=
by
  exact H_fare

end taxi_fare_l362_362470


namespace parabola_comparison_l362_362278

theorem parabola_comparison
  (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : -b / (2 * a) = 1) :
  let y := -a * d^2 + b * d,
  d := intersection₁ := d where y = (-2)/d,
  m := (d^9 - 2 * d^8 + d^6 - 8 * d^5 + 4 * d^4 - 8 * d^2) / (d^7 - 4 * d^6 + 4 * d^5),
  n := 1/d
  in
  m > n := 
sorry

end parabola_comparison_l362_362278


namespace desired_average_l362_362172

theorem desired_average (P1 P2 P3 : ℝ) (A : ℝ) 
  (hP1 : P1 = 74) 
  (hP2 : P2 = 84) 
  (hP3 : P3 = 67) 
  (hA : A = (P1 + P2 + P3) / 3) : 
  A = 75 :=
  sorry

end desired_average_l362_362172


namespace problem_solution_l362_362320

noncomputable def positiveIntPairsCount : ℕ :=
  sorry

theorem problem_solution :
  positiveIntPairsCount = 2 :=
sorry

end problem_solution_l362_362320


namespace distance_home_to_school_l362_362522

noncomputable def travel_time_late := 5 / 60    -- 5 minutes late in hours
noncomputable def travel_time_early := 10 / 60  -- 10 minutes early in hours

def distance_formula (speed : ℕ) (time_in_hours : ℚ) : ℚ := speed * time_in_hours

theorem distance_home_to_school :
  ∃ d : ℚ, let t := 5 / 12 in d = distance_formula 5 (t + travel_time_late)
  ∧ d = distance_formula 10 (t - travel_time_early)
  := sorry

end distance_home_to_school_l362_362522


namespace geom_seq_is_geometric_general_formula_seq_range_of_m_l362_362644

-- Definitions and conditions as per the problem statement
def seq (n: ℕ) : ℕ → ℕ
| 0 => 0
| 1 => 2
| 2 => 4
| (n+3) => 3 * seq (n + 2) - 2 * seq n

def geom_seq (n: ℕ) : Nat :=
seq (n + 1) - seq n

def g (n: ℕ) := 2 ^ n

def b_n (n : ℕ) := seq n - 1

def S (n : ℕ) := (List.range n).sum (λ k, seq k / ((b_n k) * (b_n (k + 1))))

-- Questions rewritten as Lean 4 statements
theorem geom_seq_is_geometric (n : ℕ) (h : n ≥ 2) :
  ∃ r, (∀ m ≥ 2, geom_seq (m + 1) = r * geom_seq m) ∧ geom_seq 2 = 2 ∧ r = 2 := 
sorry

theorem general_formula_seq (n : ℕ) :
  seq n = 2 ^ n :=
sorry

theorem range_of_m (n : ℕ) (m : ℝ) :
  (∃ n ∈ ℕ, S n ≥ 4 * m^2 - 3 * m) → -1/4 < m ∧ m < 1 :=
sorry

end geom_seq_is_geometric_general_formula_seq_range_of_m_l362_362644


namespace km_markers_two_distinct_digits_l362_362447

theorem km_markers_two_distinct_digits (A B : ℕ) (dist : ℕ) : 
  dist = 899 → A = 0 → B = 899 → 
  (count (λ k : ℕ, (two_distinct_digits k ∧ two_distinct_digits (899 - k))) (list.range (dist + 1)) = 40) := 
by
  intros
  sorry

-- Auxiliary definition to check if a number has exactly two distinct digits
def two_distinct_digits (n : ℕ) : Prop :=
  let digits := (n.digits 10).to_finset in
  digits.card = 2

-- Auxiliary definition to count elements satisfying a predicate in a list
def count {α : Type*} (p : α → Prop) [decidable_pred p] (l : list α) : ℕ :=
  l.countp p

end km_markers_two_distinct_digits_l362_362447


namespace three_digit_even_count_l362_362610

theorem three_digit_even_count : 
  let digits := {1, 2, 3, 4, 5}
  (card { n | 
      let d1 := n / 100 % 10
      let d2 := n / 10 % 10
      let d3 := n % 10
      n < 1000 ∧ 
      n ≥ 100 ∧ 
      d1 ∈ digits ∧ 
      d2 ∈ digits ∧ 
      d3 ∈ digits ∧ 
      d1 ≠ d2 ∧ 
      d2 ≠ d3 ∧ 
      d1 ≠ d3 ∧ 
      d3 % 2 = 0
  }) = 24 :=
by sorry

end three_digit_even_count_l362_362610


namespace bobby_total_candy_l362_362201

theorem bobby_total_candy (candy1 candy2 : ℕ) (h1 : candy1 = 26) (h2 : candy2 = 17) : candy1 + candy2 = 43 := 
by 
  sorry

end bobby_total_candy_l362_362201


namespace equal_distances_l362_362465

variables (a b : ℝ)

def A : ℝ × ℝ := (-a, -b)
def B : ℝ × ℝ := (a, -b)
def C : ℝ × ℝ := (a, b)
def D : ℝ × ℝ := (-a, b)

def A1 : ℝ × ℝ := (-2 * a, -b)
def A2 : ℝ × ℝ := (-a, -2 * b)

def A0 : ℝ × ℝ := ((-3/2) * a, (-3/2) * b)
def B0 : ℝ × ℝ := ((3/2) * a, (-3/2) * b)
def C0 : ℝ × ℝ := ((3/2) * a, (3/2) * b)
def D0 : ℝ × ℝ := ((-3/2) * a, (3/2) * b)

def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

theorem equal_distances :
  distance A0 C0 = distance B0 D0 :=
by
  sorry

end equal_distances_l362_362465


namespace paperclips_in_box_l362_362947

-- Define the necessary variables and conditions
variables (V1 V2 P1 P2 : ℝ)
variables (hV1 : V1 = 24)
variables (hP1 : P1 = 100)
variables (hV2 : V2 = 96)

-- Define the proportional relationship
def proportional_relationship : Prop := 
  (P1 / real.sqrt V1) = (P2 / real.sqrt V2)

-- State the theorem
theorem paperclips_in_box : proportional_relationship V1 V2 P1 P2 → P2 = 200 :=
by
  sorry

end paperclips_in_box_l362_362947


namespace inequality_proof_l362_362269

theorem inequality_proof
  (a b c d : ℝ) 
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d)
  (h_cond : a * b + b * c + c * d + d * a = 1) :
  (a^3 / (b + c + d) + b^3 / (c + d + a) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1 / 3) :=
by {
  sorry
}

end inequality_proof_l362_362269


namespace sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l362_362082

theorem sqrt_49_mul_sqrt_25_eq_7_sqrt_5 : (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l362_362082


namespace count_4_letter_words_with_A_l362_362300

-- Define the alphabet set and the properties
def Alphabet : Finset (Char) := {'A', 'B', 'C', 'D', 'E'}
def total_words := (Alphabet.card ^ 4 : ℕ)
def total_words_without_A := (Alphabet.erase 'A').card ^ 4
def total_words_with_A := total_words - total_words_without_A

-- The key theorem to prove
theorem count_4_letter_words_with_A : total_words_with_A = 369 := sorry

end count_4_letter_words_with_A_l362_362300


namespace lowest_possible_sale_price_l362_362970

theorem lowest_possible_sale_price (msrp : ℝ) (discount_percent : ℝ) (additional_discount_percent : ℝ) (lowest_possible_price : ℝ) 
  (h_msrp : msrp = 40) 
  (h_discount_range : 0.1 ≤ discount_percent ∧ discount_percent ≤ 0.3)
  (h_additional_discount : additional_discount_percent = 0.2) 
  (h_lowest_possible_price : lowest_possible_price = 22.4) : 
  let initial_discounted_price := msrp * (1 - discount_percent) in
  let final_price := initial_discounted_price * (1 - additional_discount_percent) in
  final_price = lowest_possible_price :=
by {
  simp [h_msrp, h_discount_range.right, h_additional_discount],
  norm_num,
}

end lowest_possible_sale_price_l362_362970


namespace baron_munchausen_l362_362560

noncomputable def weights : List ℕ :=
  List.map (λ i, 2^1000 - 2^i) (List.range 1000)

theorem baron_munchausen :
  (∑ i in weights, id i < 2^1010) ∧
  (∀ (other_list : List ℕ), 
    (other_list ≠ weights → (∑ i in other_list, id i ≠ ∑ i in weights, id i))) :=
by
  sorry

end baron_munchausen_l362_362560


namespace max_composite_numbers_l362_362804
open Nat

def is_composite (n : ℕ) : Prop := 1 < n ∧ ∃ d : ℕ, 1 < d ∧ d < n ∧ d ∣ n

def has_gcd_of_one (l : List ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ l → b ∈ l → a ≠ b → gcd a b = 1

def valid_composite_numbers (n : ℕ) : Prop :=
  ∀ m ∈ (List.range n).filter is_composite, m < 1500 →

-- Main theorem
theorem max_composite_numbers :
  ∃ l : List ℕ, l.length = 12 ∧ valid_composite_numbers l ∧ has_gcd_of_one l :=
sorry

end max_composite_numbers_l362_362804


namespace max_composite_numbers_l362_362789

theorem max_composite_numbers (S : Finset ℕ) (h1 : ∀ n ∈ S, n < 1500) (h2 : ∀ m n ∈ S, m ≠ n → Nat.gcd m n = 1) : S.card ≤ 12 := sorry

end max_composite_numbers_l362_362789


namespace new_cube_weight_twice_side_length_l362_362512

-- Define the conditions
variable {density : ℝ}
variable (s : ℝ) 
variable (original_weight : ℝ) 
variable (original_side : ℝ) 
variable (new_side : ℝ)
variable (new_weight : ℝ)

-- Assume the conditions
def conditions : Prop := 
  original_side = s ∧
  original_weight = 6 ∧
  (original_weight = density * original_side^3) ∧
  new_side = 2 * original_side ∧
  new_weight = density * new_side^3

-- Prove the statement
theorem new_cube_weight_twice_side_length : conditions s original_weight original_side new_side new_weight → new_weight = 48 := 
by 
  sorry

end new_cube_weight_twice_side_length_l362_362512


namespace lim_is_zero_l362_362205

theorem lim_is_zero :
  tendsto (λ x : ℝ, (1 - sqrt (cos x)) / (1 - cos (sqrt x))) (𝓝 0) (𝓝 0) :=
begin
  sorry
end

end lim_is_zero_l362_362205


namespace find_f2016_l362_362929

noncomputable def f : ℕ → ℕ := sorry

axiom cond1 : ∀ n : ℕ, f(f(n)) + f(n) = 2 * n + 3
axiom cond2 : f(0) = 1

theorem find_f2016 : f(2016) = 2017 := 
begin
  sorry
end

end find_f2016_l362_362929


namespace sqrt_49_times_sqrt_25_l362_362021

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 :=
by
  sorry

end sqrt_49_times_sqrt_25_l362_362021


namespace evaluate_expression_l362_362238

theorem evaluate_expression (c : ℕ) (h : c = 4) : 
  ((2 * c ^ c - (c + 1) * (c - 1) ^ c) ^ c) = 131044201 := by
  -- Given condition
  rw h
  -- Sorry proof to be completed
  sorry

end evaluate_expression_l362_362238


namespace mass_percentage_Al_in_AlBr3_l362_362497

theorem mass_percentage_Al_in_AlBr3 
  (molar_mass_Al : Real := 26.98) 
  (molar_mass_Br : Real := 79.90) 
  (molar_mass_AlBr3 : Real := molar_mass_Al + 3 * molar_mass_Br)
  : (molar_mass_Al / molar_mass_AlBr3) * 100 = 10.11 := 
by 
  -- Here we would provide the proof; skipping with sorry
  sorry

end mass_percentage_Al_in_AlBr3_l362_362497


namespace cubic_expression_l362_362329

theorem cubic_expression (a b c : ℝ) (h1 : a + b + c = 15) (h2 : ab + ac + bc = 50) : 
  a^3 + b^3 + c^3 - 3 * a * b * c = 1125 :=
sorry

end cubic_expression_l362_362329


namespace parallel_line_l362_362159

noncomputable def point (α : Type) := α × α
def line_f (α β : Type) [field α] [has_zero β] (f : α × α → β) (M N : point α) : Prop :=
  f M.1 M.2 = 0 ∧ f N.1 N.2 ≠ 0

theorem parallel_line {α β : Type} [field α] [has_zero β]
  {f : α × α → β} {x y x1 y1 x2 y2 : α}
  (hf1 : f (x1, y1) = 0)
  (hf2 : f (x2, y2) ≠ 0) :
  f (x, y) - f (x1, y1) - f (x2, y2) = 0 → 
  ∃ k : α, (f (x, y) = k ↔ f (x2, y2) = k) :=
sorry

end parallel_line_l362_362159


namespace number_of_edges_of_resulting_figure_l362_362815

-- Definitions for the conditions in the problem
def rectangular_sheet_of_paper : Type := sorry -- Placeholder for the type representing a rectangular sheet
def is_dot (paper : rectangular_sheet_of_paper) : Type := sorry -- Placeholder for the type representing a dot on the sheet
def black_dots_on_sheet (paper : rectangular_sheet_of_paper) (n : Nat) := sorry -- Placeholder for the type representing n black dots on the sheet

-- The proof statement
theorem number_of_edges_of_resulting_figure (paper : rectangular_sheet_of_paper) (dots : black_dots_on_sheet paper 16) : ∃ fig, figure_is_rectangle fig ∧ number_of_edges fig = 4 := 
sorry

end number_of_edges_of_resulting_figure_l362_362815


namespace no_such_integers_exists_l362_362426

theorem no_such_integers_exists 
  (a b c d : ℤ) 
  (h1 : a * 19^3 + b * 19^2 + c * 19 + d = 1) 
  (h2 : a * 62^3 + b * 62^2 + c * 62 + d = 2) : 
  false :=
by
  sorry

end no_such_integers_exists_l362_362426


namespace roots_conjugate_pair_l362_362741

theorem roots_conjugate_pair (p q : ℝ) :
  (∀ z : ℂ, (z^2 + (12 : ℂ) + (p : ℂ) * complex.I) * z + (30 : ℂ) + (q : ℂ) * complex.I = 0 → z = complex.conj z)
  → (p = 0 ∧ q = 0) :=
by
  sorry

end roots_conjugate_pair_l362_362741


namespace triangle_sides_relation_triangle_perimeter_l362_362712

theorem triangle_sides_relation
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : sin C * sin (A - B) = sin B * sin (C - A)) :
  2 * a^2 = b^2 + c^2 :=
sorry

theorem triangle_perimeter 
  (a b c : ℝ)
  (A B C : ℝ)
  (h_a : a = 5)
  (h_cosA : cos A = 25 / 31)
  (h_sin_relation : sin C * sin (A - B) = sin B * sin (C - A)) :
  a + b + c = 14 :=
sorry

end triangle_sides_relation_triangle_perimeter_l362_362712


namespace part_one_solution_set_part_two_range_of_a_l362_362292

def f (x : ℝ) (a : ℝ) : ℝ := |x - a| - 2

theorem part_one_solution_set (a : ℝ) (h : a = 1) : { x : ℝ | f x a + |2 * x - 3| > 0 } = { x : ℝ | x > 2 ∨ x < 2 / 3 } := 
sorry

theorem part_two_range_of_a : (∃ x : ℝ, f x (a) > |x - 3|) ↔ (a < 1 ∨ a > 5) :=
sorry

end part_one_solution_set_part_two_range_of_a_l362_362292


namespace valid_outfits_count_l362_362118

-- Definitions based on problem conditions
def shirts : Nat := 5
def pants : Nat := 6
def invalid_combination : Nat := 1

-- Problem statement
theorem valid_outfits_count : shirts * pants - invalid_combination = 29 := by 
  sorry

end valid_outfits_count_l362_362118


namespace some_dance_same_gender_l362_362994

theorem some_dance_same_gender :
  ∃ (p : ℕ) (n : ℕ), n = 20 ∧
  (∃ p3 p5 p6 : ℕ,
     p3 = 11 ∧ p5 = 1 ∧ p6 = 8 ∧
     (p = (p3 * 3 + p5 * 5 + p6 * 6) / 2) ∧
     2 * p ≠ p) :=
begin
  sorry
end

end some_dance_same_gender_l362_362994


namespace sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l362_362088

theorem sqrt_49_mul_sqrt_25_eq_7_sqrt_5 : (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l362_362088


namespace square_divisibility_l362_362926

theorem square_divisibility (n : ℤ) : n^2 % 4 = 0 ∨ n^2 % 4 = 1 := sorry

end square_divisibility_l362_362926


namespace quadruples_count_at_least_l362_362703

theorem quadruples_count_at_least (n : ℕ) (h : n > 100)
  (groups : list (list ℕ)) (hlen : groups.length = n)
  (hgroup : ∀ g ∈ groups, g.length = 4)
  (elems : finset ℕ) (helems : elems = finset.range (4 * n) + 1):
  ∃ quads : finset (ℕ × ℕ × ℕ × ℕ), quads.card ≥ (n - 6) * (n - 6) / 2 ∧
  (∀ (a b c d : ℕ), (a, b, c, d) ∈ quads → a < b ∧ b < c ∧ c < d ∧
    ∀ g ∈ groups, (a ∈ g) + (b ∈ g) + (c ∈ g) + (d ∈ g) = 1 ∧
    c - b ≤ |a * d - b * c| ∧ |a * d - b * c| ≤ d - a) :=
begin
  sorry
end

end quadruples_count_at_least_l362_362703


namespace sqrt_49_times_sqrt_25_l362_362027

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 :=
by
  sorry

end sqrt_49_times_sqrt_25_l362_362027


namespace sqrt_product_l362_362094

theorem sqrt_product (a b : ℝ) (h1 : a = 49) (h2 : b = 25) : sqrt (a * sqrt b) = 7 * sqrt 5 :=
by sorry

end sqrt_product_l362_362094


namespace max_moves_square_grid_max_moves_rectangular_grid_l362_362996

-- Define a type representing the grid dimension
structure GridDimension (m : Nat) (n : Nat)

-- Define an instance of the specific grids to express conditions
def SquareGrid := GridDimension 21 21
def RectangularGrid := GridDimension 20 21

-- Define the maximum_moves function that calculates the maximum number of moves for a given grid
def maximum_moves : GridDimension → Nat
| SquareGrid     := 3
| RectangularGrid := 4
| _              := sorry  -- For other cases, we leave as 'sorry'

-- Prove the maximum number of moves for the specific grids
theorem max_moves_square_grid : maximum_moves SquareGrid = 3 :=
by
  -- this needs to be proven
  sorry

theorem max_moves_rectangular_grid : maximum_moves RectangularGrid = 4 :=
by
  -- this needs to be proven
  sorry


end max_moves_square_grid_max_moves_rectangular_grid_l362_362996


namespace min_value_f_l362_362605

theorem min_value_f (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let f (a b c : ℝ) := (a / Real.sqrt (a^2 + 8 * b * c)) + 
                       (b / Real.sqrt (b^2 + 8 * a * c)) + 
                       (c / Real.sqrt (c^2 + 8 * a * b))
  in f a b c ≥ 1 :=
by 
  intro a b c ha hb hc
  let f (a b c : ℝ) := (a / Real.sqrt (a^2 + 8 * b * c)) + 
                       (b / Real.sqrt (b^2 + 8 * a * c)) + 
                       (c / Real.sqrt (c^2 + 8 * a * b))
  sorry

end min_value_f_l362_362605


namespace limit_exists_and_value_l362_362958

noncomputable def sequence (f : ℝ → ℝ) (a : ℝ) (n : ℕ) : ℝ :=
  Nat.recOn n a (λ n x, f x)

theorem limit_exists_and_value (f : ℝ → ℝ) (a : ℝ) (m k : ℤ)
  (hf : ∀ x : ℝ, f (x + 1) = f (x) + 1)
  (hx0 : ∀ n : ℕ, sequence f a (n + 1) = f (sequence f a n))
  (hxm : sequence f a m.toNat - sequence f a 0 = k) :
  ∃ l : ℝ, tendsto (λ n : ℕ, (sequence f a n) / n) atTop (𝓝 l) ∧ l = k / ↑m :=
sorry

end limit_exists_and_value_l362_362958


namespace remaining_soup_can_feed_adults_l362_362530

theorem remaining_soup_can_feed_adults :
  ∀ (total_cans : ℕ) (adults_per_can children_per_can children_fed : ℕ),
    total_cans = 8 →
    adults_per_can = 4 →
    children_per_can = 6 →
    children_fed = 18 →
    let used_cans := children_fed / children_per_can in
    let remaining_cans := total_cans - used_cans in
    let adults_fed := adults_per_can * remaining_cans in
    adults_fed = 20 :=
by
  intros total_cans adults_per_can children_per_can children_fed h_total h_adults h_children h_fed
  let used_cans := children_fed / children_per_can
  let remaining_cans := total_cans - used_cans
  let adults_fed := adults_per_can * remaining_cans
  have : used_cans = 3 := sorry  -- Calculative step
  have : remaining_cans = 5 := sorry  -- Calculative step
  have : adults_fed = 20 := sorry  -- Final calculation
  exact this

end remaining_soup_can_feed_adults_l362_362530


namespace fraction_girls_at_dance_is_half_l362_362993

-- Define the total number of students at Hamilton Middle School
def students_hamilton : ℕ := 300

-- Define the boy to girl ratio at Hamilton Middle School
def ratio_boy_girl_hamilton : ℕ × ℕ := (3, 2)

-- Define the total number of students at Lincoln Middle School
def students_lincoln : ℕ := 240

-- Define the boy to girl ratio at Lincoln Middle School
def ratio_boy_girl_lincoln : ℕ × ℕ := (3, 5)

-- Define the total number of students at the dance, which is the sum of students from both schools
def total_students_dance := students_hamilton + students_lincoln

-- Define a function to compute the number of girls given total students and ratio
def number_of_girls (total_students : ℕ) (boy_girl_ratio : ℕ × ℕ) : ℕ :=
  let (b, g) := boy_girl_ratio in
  let total_ratio := b + g in
  (g * total_students) / total_ratio

-- Compute the number of girls at Hamilton Middle School
def girls_hamilton := number_of_girls students_hamilton ratio_boy_girl_hamilton

-- Compute the number of girls at Lincoln Middle School
def girls_lincoln := number_of_girls students_lincoln ratio_boy_girl_lincoln

-- Compute the total number of girls at the dance
def total_girls_dance := girls_hamilton + girls_lincoln

-- Prove that the fraction of girls at the dance is 1/2
theorem fraction_girls_at_dance_is_half :
  (total_girls_dance : ℚ) / (total_students_dance : ℚ) = 1 / 2 :=
by
  -- Placeholder for proof
  sorry

end fraction_girls_at_dance_is_half_l362_362993


namespace count_sets_consecutive_sum_30_l362_362652

theorem count_sets_consecutive_sum_30 : 
  (∃ n a : ℕ, n ≥ 3 ∧ a ≥ 1 ∧ n * (2 * a + n - 1) = 60) → 
  ∃ s : finset (ℕ × ℕ), s.card = 3 := 
sorry

end count_sets_consecutive_sum_30_l362_362652


namespace leap_years_count_l362_362536

theorem leap_years_count (k m : ℤ) (hk : -4 ≤ k ∧ k ≤ 8) (hm : -5 ≤ m ∧ m ≤ 8) :
  let years := Multiset.of_list [1100 * k + 300, 1100 * m + 800]
  (years.filter (λ y, -5000 < y ∧ y < 10000)).card = 27 := by
  sorry

end leap_years_count_l362_362536


namespace IAOC_seating_arrangements_l362_362438

-- Definitions based on the conditions
def numMercury : ℕ := 4
def numVenus : ℕ := 4
def numEarth : ℕ := 4
def numChairs : ℕ := 12
def chairMercury : ℕ := 1
def chairVenus : ℕ := 12
def numArrangements := 1

-- Given conditions are translated to functions to check immediate left seats
noncomputable def isValidArrangement (arrangement : List ℕ) : Bool := 
  let chairs := (List.range numChairs).rotate' chairMercury
  ¬ (arrangement.nth! ((chairs.indexOf chairVenus) - 1) = numMercury) &&
  ¬ (arrangement.nth! ((chairs.indexOf numEarth) - 1) = numVenus) &&
  ¬ (arrangement.nth! ((chairs.indexOf numMercury) - 1) = numEarth)

-- Main theorem statement translated to Lean 4
theorem IAOC_seating_arrangements (N : ℕ) 
  (validArrangements : ℕ := 
    List.permutations [numMercury, numVenus, numEarth]
    .filter isValidArrangement 
    .length)
  : validArrangements * (fac numMercury) * (fac numVenus) * (fac numEarth) = N * (4!) ^ 3 := 
  by {
    have h : validArrangements = 216, sorry,
    rw h,
    ring
  }

end IAOC_seating_arrangements_l362_362438


namespace sqrt_diff_of_squares_l362_362569

theorem sqrt_diff_of_squares : (Real.sqrt 3 - 2) * (Real.sqrt 3 + 2) = -1 := by
  sorry

end sqrt_diff_of_squares_l362_362569


namespace triangle_angle_B_triangle_altitude_max_l362_362264

theorem triangle_angle_B (a b c : ℝ) (A B C : ℝ) 
  (h1 : b ≠ 0) (h2 : a ≠ c)
  (h3 : (a - b) * (Real.sin A + Real.sin B) = (a - c) * Real.sin C) :
  B = Real.pi / 3 := 
sorry

theorem triangle_altitude_max (a c : ℝ) (A B C h : ℝ)
  (h1 : a ≠ 0) (h2 : c ≠ 0) (h3 : a ≠ c)
  (h4 : B = Real.pi / 3) (h5 : b = 3)
  (h6 : (b:h) / (Real.sqrt (a^2 + c^2 - 2 * a * c * Real.cos B))) * (real.sin B) = (1 / 2) * b * h
  (h7 : b = 3) : 
  h ≤ (3 * Real.sqrt 3) / 2 := 
sorry

end triangle_angle_B_triangle_altitude_max_l362_362264


namespace volume_of_one_pizza_piece_l362_362168

theorem volume_of_one_pizza_piece
  (h : ℝ) (d : ℝ) (n : ℕ)
  (h_eq : h = 1 / 2)
  (d_eq : d = 16)
  (n_eq : n = 16) :
  ((π * (d / 2)^2 * h) / n) = 2 * π :=
by
  rw [h_eq, d_eq, n_eq]
  sorry

end volume_of_one_pizza_piece_l362_362168


namespace train_pass_time_approximately_12_seconds_l362_362121

-- Define the conditions and problem
def train_length : ℝ := 220 -- in meters
def train_speed_kmh : ℝ := 60 -- in km/h
def man_speed_kmh : ℝ := 6 -- in km/h

-- Convert speeds to m/s
def kmh_to_ms (kmh : ℝ) : ℝ := kmh * 1000 / 3600
def train_speed_ms : ℝ := kmh_to_ms train_speed_kmh
def man_speed_ms : ℝ := kmh_to_ms man_speed_kmh

-- Define relative speed
def relative_speed_ms : ℝ := train_speed_ms + man_speed_ms

-- Define the time required to pass the man
def time_to_pass : ℝ := train_length / relative_speed_ms

-- The theorem to prove
theorem train_pass_time_approximately_12_seconds : abs (time_to_pass - 12) < 1 :=
sorry

end train_pass_time_approximately_12_seconds_l362_362121


namespace triangle_sides_and_area_l362_362348

noncomputable def cosine_rule_b (a c : ℝ) (B : ℝ) : ℝ :=
  real.sqrt (a^2 + c^2 - 2 * a * c * real.cos B)

noncomputable def area_of_triangle (a c : ℝ) (B : ℝ) : ℝ :=
  0.5 * a * c * real.sin B

theorem triangle_sides_and_area :
  let a := 3 * real.sqrt 3
  let c := 2
  let B := real.pi * 150 / 180
  let b := cosine_rule_b a c B
  b = 7 ∧ area_of_triangle a c B = 3 * real.sqrt 3 / 2 :=
by {
  sorry
}

end triangle_sides_and_area_l362_362348


namespace serving_guests_possible_iff_even_l362_362590

theorem serving_guests_possible_iff_even (n : ℕ) : 
  (∀ seats : Finset ℕ, ∀ p : ℕ → ℕ, (∀ i : ℕ, i < n → p i ∈ seats) → 
    (∀ i j : ℕ, i < j → p i ≠ p j) → (n % 2 = 0)) = (n % 2 = 0) :=
by sorry

end serving_guests_possible_iff_even_l362_362590


namespace comic_books_stack_count_l362_362749

theorem comic_books_stack_count :
  let spiderman_books := 7
  let archie_books := 5
  let garfield_books := 4
  let total_books := spiderman_books + archie_books + garfield_books
  calc_factorial (n : ℕ) : ℕ :=
    if n = 0 then
      1
    else
      n * calc_factorial (n - 1)
  calc_ways (n : ℕ) (other : ℕ) : ℕ :=
    let spiderman_ways := calc_factorial spiderman_books
    let archie_ways := calc_factorial archie_books
    let garfield_ways := calc_factorial garfield_books
    let other_ways := calc_factorial other
    spiderman_ways * archie_ways * garfield_ways * other_ways
  spiderman_books = 7 → archie_books = 5 → garfield_books = 4 → calc_ways total_books (other => 2) = 29030400 :=
begin
  have h: 7 + 5 + 4 = 16, by norm_num,
  have calc_ways : (7! * 5! * 4! * 2! = 29030400), by norm_num,
  sorry
end

end comic_books_stack_count_l362_362749


namespace minimum_value_frac_range_of_f4_range_of_m_l362_362643

section Part1

variable (a b : ℝ)

def f (x : ℝ) := a * x + b

theorem minimum_value_frac (h₁ : f 2 = 1) (h₂ : 0 < a) (h₃ : 0 < b) :
  ∃ a b : ℝ, a = 1 / 4 ∧ b = 1 / 2 ∧ (∀ a' b', f 2 = 1 → 0 < a' → 0 < b' → (1 / a') + (2 / b') ≥ 8) :=
by sorry

end Part1

section Part2

variable (a b : ℝ)

def f (x : ℝ) := a * x + b

theorem range_of_f4 (h₁ : ∀ x, 1 ≤ x ∧ x ≤ 2 → 0 ≤ f x ∧ f x ≤ 1) :
  -2 ≤ f 4 ∧ f 4 ≤ 3 :=
by sorry

end Part2

section Part3

variable (m x : ℝ)

def g (x : ℝ) := x^2 - 2*x - 8

theorem range_of_m (h₁ : ∀ x, x > 2 → g x ≥ (m + 2) * x - m - 15) :
  m ≤ 2 :=
by sorry

end Part3

end minimum_value_frac_range_of_f4_range_of_m_l362_362643


namespace smallest_prime_factor_of_2023_l362_362916

theorem smallest_prime_factor_of_2023 : Nat.prime 7 ∧ 7 ∣ 2023 ∧ ∀ p, Nat.prime p ∧ p ∣ 2023 → p ≥ 7 :=
by 
  sorry

end smallest_prime_factor_of_2023_l362_362916


namespace max_composite_numbers_l362_362791

-- Definitions and conditions
def is_composite (n : ℕ) : Prop := 2 < n ∧ ∃ d, d ∣ n ∧ 1 < d ∧ d < n

def less_than_1500 (n : ℕ) : Prop := n < 1500

def gcd_is_one (a b : ℕ) : Prop := Int.gcd a b = 1

-- The problem statement
theorem max_composite_numbers (numbers : List ℕ) (h_composite : ∀ n ∈ numbers, is_composite n) 
  (h_less_than_1500 : ∀ n ∈ numbers, less_than_1500 n)
  (h_pairwise_gcd_is_one : (numbers.Pairwise gcd_is_one)) : numbers.length ≤ 12 := 
  sorry

end max_composite_numbers_l362_362791


namespace ruby_height_is_192_l362_362661

def height_janet := 62
def height_charlene := 2 * height_janet
def height_pablo := height_charlene + 70
def height_ruby := height_pablo - 2

theorem ruby_height_is_192 : height_ruby = 192 := by
  sorry

end ruby_height_is_192_l362_362661


namespace problem_statement_l362_362336

theorem problem_statement (x : ℚ) (h : 8 * x = 3) : 200 * (1 / x) = 1600 / 3 :=
by
  sorry

end problem_statement_l362_362336


namespace sqrt_expression_simplified_l362_362013

theorem sqrt_expression_simplified :
  sqrt (49 * sqrt 25) = 7 * sqrt 5 := by
  sorry

end sqrt_expression_simplified_l362_362013


namespace S_11_is_22_l362_362655

-- Definitions and conditions
variable (a_1 d : ℤ) -- first term and common difference of the arithmetic sequence
noncomputable def S (n : ℤ) : ℤ := n * (2 * a_1 + (n - 1) * d) / 2

-- The given condition
variable (h : S a_1 d 8 - S a_1 d 3 = 10)

-- The proof goal
theorem S_11_is_22 : S a_1 d 11 = 22 :=
by
  sorry

end S_11_is_22_l362_362655


namespace arthur_walked_in_total_l362_362199

def blocks_east : ℕ := 8
def blocks_north : ℕ := 15
def blocks_west : ℕ := 3
def block_length : ℝ := 1 / 4

theorem arthur_walked_in_total :
    (blocks_east + blocks_north + blocks_west) * block_length = 6.5 := by
  sorry

end arthur_walked_in_total_l362_362199
