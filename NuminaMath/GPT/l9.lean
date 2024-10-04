import Mathlib

namespace intersection_at_most_one_l9_9334

noncomputable def f (a x : ℝ) : ℝ := log a x
noncomputable def f_inv (a x : ℝ) : ℝ := a ^ x

theorem intersection_at_most_one (a : ℝ) (ha : a > 1) : ∀ x : ℝ, f a x = f_inv a x → x ≤ 1 := 
sorry

end intersection_at_most_one_l9_9334


namespace janet_faster_playtime_l9_9894

theorem janet_faster_playtime 
  (initial_minutes : ℕ)
  (initial_seconds : ℕ)
  (faster_rate : ℝ)
  (initial_time_in_seconds := initial_minutes * 60 + initial_seconds)
  (target_time_in_seconds := initial_time_in_seconds / faster_rate) :
  initial_minutes = 3 →
  initial_seconds = 20 →
  faster_rate = 1.25 →
  target_time_in_seconds = 160 :=
by
  intros h1 h2 h3
  sorry

end janet_faster_playtime_l9_9894


namespace chores_cleaning_tasks_l9_9517

theorem chores_cleaning_tasks :
  ∀ (total_tasks shower_tasks dinner_tasks : ℕ), 
    (∀ (time_per_task total_time_minutes : ℕ), 
       total_time_minutes = 120 ∧ time_per_task = 10 →
       total_tasks = total_time_minutes / time_per_task) → 
    (shower_tasks = 1 ∧ dinner_tasks = 4 →
     total_tasks = shower_tasks + dinner_tasks + x →
     x = 7) :=
begin
  intros total_tasks shower_tasks dinner_tasks h1 h2,
  obtain ⟨time_per_task, total_time_minutes, h3⟩ := h1,
  obtain ⟨h_shower, h_dinner⟩ := h2,
  rw h_shower at *,
  rw h_dinner at *,
  sorry,
end

end chores_cleaning_tasks_l9_9517


namespace symmetry_center_l9_9093

def original_function (x : Real) : Real :=
  4 * Real.sin (4 * x + Real.pi / 6)

def transformed_function (x : Real) : Real :=
  4 * Real.sin (2 * (x - Real.pi / 6) + Real.pi / 6)

theorem symmetry_center :
  ∃ k ∈ Int, (transformed_function (k * Real.pi / 2 + Real.pi / 12) = 0) → 
    (k * Real.pi / 2 + Real.pi / 12) = 7 * Real.pi / 12 :=
sorry

end symmetry_center_l9_9093


namespace equal_focal_distances_condition_l9_9563

theorem equal_focal_distances_condition (k : ℝ) : 
  (∀ x y : ℝ, (9*x^2 + 25*y^2 = 225) → 
              (∀ x y : ℝ, (x^2/(16-k) - y^2/k = 1) → 
              (2*sqrt(25 - 9) = 2*sqrt(16)))) ↔ 0 < k ∧ k < 16 :=
by
  sorry

end equal_focal_distances_condition_l9_9563


namespace arithmetic_geometric_sequences_sum_of_c_n_l9_9398

theorem arithmetic_geometric_sequences (d q : ℝ) (q_pos : 0 < q)
  (a_1 : ℕ → ℝ) (b_1 : ℕ → ℝ) 
  (ha_seq : ∀ n, a_1 (n + 1) = a_1 n + d)
  (hb_seq : ∀ n, b_1 (n + 1) = b_1 n * q)
  (h_a1_b1 : a_1 0 = 2 ∧ b_1 0 = 2)
  (h_S5 : 5 * (b_1 1 * q) = a_1 10 + (b_1 2 * q)) :
  (a_1 = fun n => n + 1) ∧ (b_1 = fun n => 2 ^ n) :=
by
  sorry

theorem sum_of_c_n (n : ℕ) 
  (c_1 : ℕ → ℝ) (hc_seq : ∀ n, c_1 n = (n + 1) / (2 ^ n)) :
  (∑ i in finset.range n, c_1 i) = 3 - (n + 3) / (2 ^ n) :=
by
  sorry

end arithmetic_geometric_sequences_sum_of_c_n_l9_9398


namespace virus_spread_in_fourth_round_l9_9610

theorem virus_spread_in_fourth_round :
  let initial_infected := 1 in
  let spread_rate := 20 in
  let round := 4 in
  initial_infected * spread_rate ^ (round - 1) = 8000 :=
by
  let initial_infected := 1
  let spread_rate := 20
  let round := 4
  sorry

end virus_spread_in_fourth_round_l9_9610


namespace eval_expression_at_neg3_l9_9298

def evaluate_expression (x : ℤ) : ℚ :=
  (5 + x * (5 + x) - 4 ^ 2 : ℤ) / (x - 4 + x ^ 3 : ℤ)

theorem eval_expression_at_neg3 :
  evaluate_expression (-3) = -17 / 20 := by
  sorry

end eval_expression_at_neg3_l9_9298


namespace intersection_and_complement_l9_9802

open Set

def A := {x : ℝ | -4 ≤ x ∧ x ≤ -2}
def B := {x : ℝ | x + 3 ≥ 0}

theorem intersection_and_complement : 
  (A ∩ B = {x | -3 ≤ x ∧ x ≤ -2}) ∧ (compl (A ∩ B) = {x | x < -3 ∨ x > -2}) :=
by
  sorry

end intersection_and_complement_l9_9802


namespace complex_power_l9_9687

def cos_30 := Float.sqrt 3 / 2
def sin_30 := 1 / 2

theorem complex_power (i : ℂ) : 
  (3 * (⟨cos_30, sin_30⟩ : ℂ)) ^ 4 = -40.5 + 40.5 * i * Float.sqrt 3 := 
  sorry

end complex_power_l9_9687


namespace S4_equals_15_l9_9862

noncomputable def S_n (q : ℝ) (n : ℕ) := (1 - q^n) / (1 - q)

theorem S4_equals_15 (q : ℝ) (n : ℕ) (h1 : S_n q 1 = 1) (h2 : S_n q 5 = 5 * S_n q 3 - 4) : 
  S_n q 4 = 15 :=
by
  sorry

end S4_equals_15_l9_9862


namespace total_cost_l9_9230

theorem total_cost (a b : ℕ) : 30 * a + 20 * b = 30 * a + 20 * b :=
by
  sorry

end total_cost_l9_9230


namespace S₄_eq_15_l9_9840

-- Definitions based on the given conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum a

def sequence_condition (a : ℕ → ℝ) : Prop :=
  is_geometric_sequence a ∧ a 1 = 1 ∧ sum_of_first_n_terms a 5 = 5 * sum_of_first_n_terms a 3 - 4

theorem S₄_eq_15 (a : ℕ → ℝ) (q : ℝ) :
  sequence_condition a →
  (∀ n, a n = 1 * q ^ (n-1)) → 
  sum_of_first_n_terms a 4 = 15 :=
sorry

end S₄_eq_15_l9_9840


namespace space_station_perimeter_l9_9014

theorem space_station_perimeter (r : ℝ) (θ : ℝ) (H1 : r = 3) (H2 : θ = 90) : 
  let total_perimeter := (3/4) * (2 * Real.pi * r) + 2 * r 
  in total_perimeter = (9 / 2) * Real.pi + 6 :=
by
  -- Define the fraction of the circle that is not part of the antenna slot
  have fraction := 3 / 4
  -- Calculate the circumference of the full circle
  have circumference := 2 * Real.pi * r
  -- Calculate the length of the arc
  have arc_length := fraction * circumference
  -- Calculate the length of the two radii
  have radii_length := 2 * r
  -- Combine both to get the total perimeter
  let total_perimeter := arc_length + radii_length
  -- Prove the final statement
  simp [H1, H2] at *,
  intro,
  sorry

end space_station_perimeter_l9_9014


namespace complex_power_rectangular_form_l9_9673

noncomputable def cos (θ : ℝ) : ℝ := sorry
noncomputable def sin (θ : ℝ) : ℝ := sorry

theorem complex_power_rectangular_form :
  (3 * complex.exp (complex.I * 30))^4 = -40.5 + 40.5 * complex.I * √3 :=
by
  sorry

end complex_power_rectangular_form_l9_9673


namespace warehouse_problem_l9_9225

/-- 
Problem Statement:
A certain unit decides to invest 3200 yuan to build a warehouse (in the shape of a rectangular prism) with a constant height.
The back wall will be built reusing the old wall at no cost, the front will be made of iron grilles at a cost of 40 yuan per meter in length,
and the two side walls will be built with bricks at a cost of 45 yuan per meter in length.
The top will have a cost of 20 yuan per square meter.
Let the length of the iron grilles be x meters and the length of one brick wall be y meters.
Find:
1. Write down the relationship between x and y.
2. Determine the maximum allowable value of the warehouse area S. In order to maximize S without exceeding the budget, how long should the front iron grille be designed
-/

theorem warehouse_problem (x y : ℝ) :
    (40 * x + 90 * y + 20 * x * y = 3200 ∧ 0 < x ∧ x < 80) →
    (y = (320 - 4 * x) / (9 + 2 * x) ∧ x = 15 ∧ y = 20 / 3 ∧ x * y = 100) :=
by
  sorry

end warehouse_problem_l9_9225


namespace probability_of_root_l9_9940

noncomputable def probability_has_root : Prop :=
  let Ω := Set.Icc (-Real.pi) Real.pi × Set.Icc (-Real.pi) Real.pi
  let condition (a b : ℝ) := a^2 + b^2 ≥ Real.pi
  let favorable := {p ∈ Ω | condition p.1 p.2}
  (∃ (μ : Measure (Set ℝ × Set ℝ)), μ Ω = 1 ∧ μ favorable = 3 / 4)

theorem probability_of_root : probability_has_root :=
  sorry

end probability_of_root_l9_9940


namespace squirrels_acorns_l9_9218

theorem squirrels_acorns (total_acorns : ℕ) (num_squirrels : ℕ) (required_per_squirrel : ℕ) 
  (h1 : total_acorns = 575) (h2 : num_squirrels = 5) (h3 : required_per_squirrel = 130) :
  let acorns_per_squirrel := total_acorns / num_squirrels in
  required_per_squirrel - acorns_per_squirrel = 15 :=
by
  sorry

end squirrels_acorns_l9_9218


namespace combined_area_win_bonus_l9_9227

theorem combined_area_win_bonus (r : ℝ) (P_win P_bonus : ℝ) : 
  r = 8 → P_win = 1 / 4 → P_bonus = 1 / 8 → 
  (P_win * (Real.pi * r^2) + P_bonus * (Real.pi * r^2) = 24 * Real.pi) :=
by
  intro h_r h_Pwin h_Pbonus
  rw [h_r, h_Pwin, h_Pbonus]
  -- Calculation is skipped as per the instructions
  sorry

end combined_area_win_bonus_l9_9227


namespace interest_percentage_correct_l9_9898

namespace BondInterest

def face_value : ℝ := 5000
def interest_rate : ℝ := 0.06
def selling_price : ℝ := 4615.384615384615

def interest : ℝ := face_value * interest_rate
def interest_percentage_of_selling_price : ℝ := (interest / selling_price) * 100

theorem interest_percentage_correct :
  interest_percentage_of_selling_price ≈ 6.5 :=
sorry

end BondInterest

end interest_percentage_correct_l9_9898


namespace count_valid_matrices_l9_9811

open Matrix

def valid_row_sum (A : Matrix (Fin 4) (Fin 4) ℤ) : Prop :=
  ∀ i, (∑ j, A i j) = 0

def valid_col_sum (A : Matrix (Fin 4) (Fin 4) ℤ) : Prop :=
  ∀ j, (∑ i, A i j) = 0

def valid_entries (A : Matrix (Fin 4) (Fin 4) ℤ) : Prop :=
  ∀ i j, A i j = 1 ∨ A i j = -1

def valid_matrix (A : Matrix (Fin 4) (Fin 4) ℤ) : Prop :=
  valid_row_sum A ∧ valid_col_sum A ∧ valid_entries A

theorem count_valid_matrices : {A : Matrix (Fin 4) (Fin 4) ℤ // valid_matrix A}.card = 90 :=
sorry

end count_valid_matrices_l9_9811


namespace sum_of_solutions_eq_neg2_l9_9721

theorem sum_of_solutions_eq_neg2 : 
  let sum_solutions (a b c : ℝ) := -(b / a)
  in
  sum_solutions 1 2 (-4) = -2 := 
by
  sorry

end sum_of_solutions_eq_neg2_l9_9721


namespace sum_binom_mod_l9_9055

/-- Given a prime number 2027, prove that the sum of binomial coefficients 
    from k = 0 to 64 of (2024 choose k) is congruent to 1090 modulo 2027.
-/
theorem sum_binom_mod (p : ℕ) (h_prime : p = 2027) : 
  ∑ k in finset.range 65, nat.choose 2024 k ≡ 1090 [MOD p] :=
by {
  -- This is where the proof would go
  sorry
}

end sum_binom_mod_l9_9055


namespace sum_of_arithmetic_sequence_is_54_l9_9780

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_arithmetic_sequence_is_54 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h_condition : 2 * a 8 = 6 + a 11) : 
  S 9 = 54 :=
sorry

end sum_of_arithmetic_sequence_is_54_l9_9780


namespace tom_used_10_plates_l9_9139

theorem tom_used_10_plates
  (weight_per_plate : ℕ := 30)
  (felt_weight : ℕ := 360)
  (heavier_factor : ℚ := 1.20) :
  (felt_weight / heavier_factor / weight_per_plate : ℚ) = 10 := by
  sorry

end tom_used_10_plates_l9_9139


namespace year_2013_is_not_lucky_l9_9291

-- Definitions based on conditions
def last_two_digits (year : ℕ) : ℕ := year % 100

def is_valid_date (month : ℕ) (day : ℕ) (year : ℕ) : Prop :=
  month * day = last_two_digits year

def is_lucky_year (year : ℕ) : Prop :=
  ∃ (month : ℕ) (day : ℕ), month <= 12 ∧ day <= 12 ∧ is_valid_date month day year

-- The main statement to prove
theorem year_2013_is_not_lucky : ¬ is_lucky_year 2013 :=
by {
  sorry
}

end year_2013_is_not_lucky_l9_9291


namespace monotonic_decreasing_interval_l9_9980

variable (f : ℝ → ℝ)

noncomputable def func (x : ℝ) : ℝ := real.sqrt (-x^2 + 2*x + 3)

axiom domain_f : ∀ x : ℝ, x ∈ set.Icc (-1:ℝ) 3 → func x = real.sqrt (-x^2 + 2*x + 3)
axiom incr_t : ∀ x y : ℝ, x ∈ set.Icc (-1:ℝ) 1 → y ∈ set.Icc (-1:ℝ) 1 → x ≤ y → (-x^2 + 2*x + 3) ≤ (-y^2 + 2*y + 3)
axiom decr_t : ∀ x y : ℝ, x ∈ set.Icc (1:ℝ) 3 → y ∈ set.Icc (1:ℝ) 3 → x ≤ y → (-x^2 + 2*x + 3) ≥ (-y^2 + 2*y + 3)
axiom incr_sqrt : ∀ t1 t2 : ℝ, t1 ∈ set.Ici 0 → t2 ∈ set.Ici 0 → t1 ≤ t2 → real.sqrt t1 ≤ real.sqrt t2

theorem monotonic_decreasing_interval : 
  ∀ x y : ℝ, x ∈ set.Icc (1:ℝ) 3 → y ∈ set.Icc (1:ℝ) 3 → x ≤ y → func y ≤ func x :=
sorry

end monotonic_decreasing_interval_l9_9980


namespace sum_of_a_and_b_l9_9430

theorem sum_of_a_and_b (a b : ℝ) (h1 : abs a = 5) (h2 : b = -2) (h3 : a * b > 0) : a + b = -7 := by
  sorry

end sum_of_a_and_b_l9_9430


namespace quadratic_completing_square_l9_9629

theorem quadratic_completing_square (b p : ℝ) (hb : b < 0)
  (h_quad_eq : ∀ x : ℝ, x^2 + b * x + (1 / 6) = (x + p)^2 + (1 / 18)) :
  b = - (2 / 3) :=
by
  sorry

end quadratic_completing_square_l9_9629


namespace complex_power_eq_rectangular_l9_9700

noncomputable def cos_30 := Real.cos (Real.pi / 6)
noncomputable def sin_30 := Real.sin (Real.pi / 6)

theorem complex_power_eq_rectangular :
  (3 * (cos_30 + (complex.I * sin_30)))^4 = -40.5 + 40.5 * complex.I * Real.sqrt 3 := 
sorry

end complex_power_eq_rectangular_l9_9700


namespace investment_triple_period_l9_9651

theorem investment_triple_period
    (P : ℝ) 
    (r : ℝ) 
    (ht : r = 0.3334) 
    (H : 3 < (1 + r) ^ n) :
  ∃ n : ℕ, n = 4 := 
begin
  have h_ln : log 3 < n * log(1 + r),
    simp [ht, H, log],

  use 4,
  linarith,
end,

end investment_triple_period_l9_9651


namespace XF_XG_eq_55_div_2_l9_9079

variable {O : Type*} [metric_space O]
variable (A B C D X Y E F G : O)
variable (AB BC CD DA AX CX : ℝ)
variable (BD : ℝ)
variable [inscribed A B C D O]

-- Conditions
def AB_length : AB = 4 := sorry
def BC_length : BC = 3 := sorry
def CD_length : CD = 7 := sorry
def DA_length : DA = 9 := sorry
def DX_ratio : DX / BD = 1/3 := sorry
def BY_ratio : BY / BD = 1/4 := sorry
def parallel_Y_ad : ∃ Y_parallel_AD, Y_parallel_AD ∣ AD := sorry
def parallel_EF_ac : ∃ E_parallel_AC, E_parallel_AC ∣ AC := sorry

-- Question (proof problem)
theorem XF_XG_eq_55_div_2 : XF * XG = 55 / 2 := sorry

end XF_XG_eq_55_div_2_l9_9079


namespace circle_center_radius_l9_9548

theorem circle_center_radius (x y : ℝ) :
  (∃ (h k r : ℝ), (h = -1.5) ∧ (k = 1) ∧ (r = sqrt 17 / 2) ∧ 
  (x^2 + y^2 + 3*x - 2*y - 1 = (x+h)^2 + (y+k)^2 - r^2)) :=
sorry

end circle_center_radius_l9_9548


namespace find_sixth_term_l9_9995

noncomputable def arithmetic_sequence (a1 d : ℕ) (n : ℕ) : ℕ :=
  a1 + (n - 1) * d

noncomputable def sum_first_n_terms (a1 d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem find_sixth_term :
  ∀ (a1 S3 : ℕ),
  a1 = 2 →
  S3 = 12 →
  ∃ d : ℕ, sum_first_n_terms a1 d 3 = S3 ∧ arithmetic_sequence a1 d 6 = 12 :=
by
  sorry

end find_sixth_term_l9_9995


namespace sum_of_ages_l9_9150

-- Define the variables for Viggo and his younger brother's ages
variables (v y : ℕ)

-- Condition: When Viggo's younger brother was 2, Viggo's age was 10 years more than twice his brother's age
def condition1 (v y : ℕ) := (y = 2 → v = 2 * y + 10)

-- Condition: Viggo's younger brother is currently 10 years old
def condition2 (y_current : ℕ) := y_current = 10

-- Define the current age of Viggo given the conditions
def viggo_current_age (v y y_current : ℕ) := v + (y_current - y)

-- Prove that the sum of their ages is 32
theorem sum_of_ages
  (v y y_current : ℕ)
  (h1 : condition1 v y)
  (h2 : condition2 y_current) :
  viggo_current_age v y y_current + y_current = 32 :=
by
  -- Apply sorry to skip the proof
  sorry

end sum_of_ages_l9_9150


namespace area_of_perpendicular_diagonal_trapezoid_l9_9962

theorem area_of_perpendicular_diagonal_trapezoid (h BD : ℝ) 
  (height_trapezoid : h = 4) 
  (BD_perpendicular : ⟂ BD AC)
  (known_diagonal : BD = 5) :
  ∃ (S : ℝ), S = 50 / 3 :=
by sorry

end area_of_perpendicular_diagonal_trapezoid_l9_9962


namespace count_odd_three_digit_numbers_sum_tens_units_eq_11_l9_9813

theorem count_odd_three_digit_numbers_sum_tens_units_eq_11 : 
  (∃ n : ℕ, n >= 100 ∧ n < 1000 ∧ odd n ∧ (n / 10 % 10 + n % 10 = 11)) = 18 := 
sorry

end count_odd_three_digit_numbers_sum_tens_units_eq_11_l9_9813


namespace rectangle_diagonal_length_l9_9115

theorem rectangle_diagonal_length (P L W : ℝ) (hP : P = 72) (hRatio : 5 * W = 4 * L) :
  ∃ d, d = 4 * real.sqrt 41 ∧ (d = real.sqrt (L^2 + W^2)) :=
by
  -- Define the values of k
  let k := P / (2 * (5 + 4))  -- Using the perimeter relation
  let L := 5 * k  -- Length
  let W := 4 * k  -- Width
  -- Define the diagonal
  let d := real.sqrt (L^2 + W^2)
  -- Expected answer
  existsi 4 * real.sqrt 41
  -- Demonstrate they are equal
  have : d = 4 * real.sqrt 41 := sorry
  exact ⟨rfl, this⟩

end rectangle_diagonal_length_l9_9115


namespace max_value_round_l9_9453

variable (H M M' T G U T' S R O U N D : ℕ)

def distinct_digits (a b c d e f g h i j : ℕ) : Prop :=
  list.nodup [a, b, c, d, e, f, g, h, i, j]

def no_leading_zeroes (H G : ℕ) : Prop :=
  H ≠ 0 ∧ G ≠ 0

theorem max_value_round :
  distinct_digits H M M' T G U T' S R O U N D ∧ no_leading_zeroes H G ∧
  H * 1000 + M * 100 + M' * 10 + T + G * 1000 + U * 100 + T' * 10 + S = R * 10000 + O * 1000 + U * 100 + N * 10 + D ->
  R * 10000 + O * 1000 + U * 100 + N * 10 + D = 16352 :=
begin
  sorry
end

end max_value_round_l9_9453


namespace centroid_incenter_orthogonal_l9_9884

noncomputable def is_incenter {A B C O : Point} : Prop :=
  let I := incenter A B C
  I = O

theorem centroid_incenter_orthogonal
  (A B C M O A' B' C' : Point)
  (touch_A' : A' ∈ line_segment B C)
  (touch_B' : B' ∈ line_segment C A)
  (touch_C' : C' ∈ line_segment A B)
  (median_intersection : is_centroid A B C M)
  (incenter_circ : is_incenter A B C O)
  (touch_condition : dist C A' = dist A B) :
  ∠ O M (line_through A B) = 90 := 
begin
  sorry
end

end centroid_incenter_orthogonal_l9_9884


namespace isosceles_triangle_PAD_l9_9900

theorem isosceles_triangle_PAD
  (Γ : Type*) [circle Γ]
  (A B C D P : Γ)
  (hInscribed : ∀ (T : triangle ℝ), triangle.is_inscribed_in_circle T Γ)
  (hFoot : is_foot_of_angle_bisector A D B C)
  (hTangent : is_tangent_to_circle_at A Γ P) :
  is_isosceles_at P A D :=
begin
  sorry
end

end isosceles_triangle_PAD_l9_9900


namespace rectangle_area_l9_9240

noncomputable def area_of_rectangle (x : ℝ) : ℝ := 
  let l := width
  let diagonal_len : ℝ := x
  let length := 2 * l
  let width := l
  let area := length * width
  have pythagorean : diagonal_len^2 = length^2 + width^2 := by sorry
  have expr_for_l_squared : width^2 = diagonal_len^2 / 5 := by sorry
  have area_expr := 2 * (diagonal_len^2 / 5) 
  area_expr
  
theorem rectangle_area (x : ℝ) : 
  ∃ (area_of_rectangle : ℝ), area_of_rectangle = (2/5) * x^2 :=
begin
  use (2/5) * x^2,
  sorry
end

end rectangle_area_l9_9240


namespace parabola_tangent_line_l9_9010

theorem parabola_tangent_line (a : ℝ) : 
  (∀ x : ℝ, (y = ax^2 + 6 ↔ y = x)) → a = 1 / 24 :=
by
  sorry

end parabola_tangent_line_l9_9010


namespace probability_difference_l9_9836

noncomputable def Ps (red black : ℕ) : ℚ :=
  let total := red + black
  (red * (red - 1) + black * (black - 1)) / (total * (total - 1))

noncomputable def Pd (red black : ℕ) : ℚ :=
  let total := red + black
  (red * black * 2) / (total * (total - 1))

noncomputable def abs_diff (Ps Pd : ℚ) : ℚ :=
  |Ps - Pd|

theorem probability_difference :
  let red := 1200
  let black := 800
  let total := red + black
  abs_diff (Ps red black) (Pd red black) = 789 / 19990 := by
  sorry

end probability_difference_l9_9836


namespace equidistant_line_through_P_l9_9308

theorem equidistant_line_through_P (P A B : Point) (L : Line) : 
  P = ⟨0, 1⟩ → A = ⟨3, 3⟩ → B = ⟨5, -1⟩ → 
  (L = {l : Line | passes_through l P ∧ equidistant_from l A B}) →
  (L = {l : Line | equation l = "y = 1"} ∨ 
   L = {l : Line | equation l = "2x + y - 1 = 0"}) :=
by {
  intros p_eq a_eq b_eq l_def,
  sorry
}

end equidistant_line_through_P_l9_9308


namespace find_A_area_triangle_l9_9407

noncomputable def cos_t : ℝ → ℝ := λ t, real.cos t
noncomputable def sin_t : ℝ → ℝ := λ t, real.sin t

def m (x : ℝ) : ℝ × ℝ := (1, sin_t (2 * x))
def n (x : ℝ) : ℝ × ℝ := (cos_t (2 * x), real.sqrt 3)
def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

axiom A_val : ∃ A, f A = 1

theorem find_A : 
  A_val → ∃ A, A = real.pi / 3 :=
by
  intro h
  sorry

theorem area_triangle (A : ℝ) (a b c : ℝ) : 
  A = real.pi / 3 ∧ a = real.sqrt 3 ∧ b + c = 3 → 
  (1 / 2) * a * b * sin_t A = real.sqrt 3 / 2 :=
by
  intro h
  sorry

end find_A_area_triangle_l9_9407


namespace range_of_m_l9_9829

noncomputable def value_preserving_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x ∈ set.Icc a b, f x ∈ set.Icc a b) ∧
  (∀ x y ∈ set.Icc a b, x ≤ y → f x ≤ f y ∨ f x ≥ f y)

theorem range_of_m {m : ℝ} :
  (∀ a b, value_preserving_interval (λ x : ℝ, x^2 - x / 2 + m) a b) ↔
  m ∈ set.Ico (5 / 16 : ℝ) (9 / 16 : ℝ) ∨ m ∈ set.Ico (-11 / 16 : ℝ) (-7 / 16 : ℝ) := by
  sorry

end range_of_m_l9_9829


namespace smallest_number_of_coins_l9_9280

-- Given Conditions as definitions
def isProperFactor (n : ℕ) (y : ℕ) : Prop := 
  y > 1 ∧ y < n ∧ n % y = 0

def properFactors (n : ℕ) : Finset ℕ :=
  (Finset.range (n - 1)).filter (isProperFactor n)

-- Property about the total number of factors
def numFactors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, n % d = 0).card

-- Main proof statement
theorem smallest_number_of_coins : ∃ n : ℕ, n = 131072 ∧ properFactors n . card = 16 :=
by
  sorry

end smallest_number_of_coins_l9_9280


namespace complex_fourth_power_l9_9697

-- Definitions related to the problem conditions
def trig_identity (θ : ℝ) : ℂ := complex.of_real (cos θ) + complex.I * complex.of_real (sin θ)
def z : ℂ := 3 * trig_identity (30 * real.pi / 180)
def result := (81 : ℝ) * (-1/2 + complex.I * (real.sqrt 3 / 2))

-- Statement of the problem
theorem complex_fourth_power : z^4 = result := by
  sorry

end complex_fourth_power_l9_9697


namespace smallest_prime_after_six_consecutive_nonprimes_l9_9172

-- Define what it means to be prime
def is_prime (n : ℕ) : Prop :=
  nat.prime n

-- Define the sequence of six consecutive nonprime numbers
def consecutive_nonprime_sequence (a : ℕ) : Prop :=
  ¬ is_prime a ∧ ¬ is_prime (a + 1) ∧ ¬ is_prime (a + 2) ∧
  ¬ is_prime (a + 3) ∧ ¬ is_prime (a + 4) ∧ ¬ is_prime (a + 5)

-- Define the condition that there is a smallest prime number greater than a sequence of six consecutive nonprime numbers
theorem smallest_prime_after_six_consecutive_nonprimes :
  ∃ p, is_prime p ∧ p > 96 ∧ consecutive_nonprime_sequence 90 :=
sorry

end smallest_prime_after_six_consecutive_nonprimes_l9_9172


namespace first_topping_cost_l9_9895

def total_cost_of_pizza_with_toppings (num_slices : ℕ) (cost_per_slice : ℕ) : ℕ := 
  num_slices * cost_per_slice

def total_cost_of_pizza_without_toppings (pizza_cost : ℕ) : ℕ := 
  pizza_cost

def total_cost_of_toppings (total_cost_with_toppings pizza_cost : ℕ) : ℕ := 
  total_cost_with_toppings - pizza_cost

def cost_of_2_dollar_toppings (num_toppings : ℕ) (cost_per_topping : ℕ) : ℕ := 
  num_toppings * cost_per_topping

def cost_of_50_cent_toppings (num_toppings : ℕ) (cost_per_topping : ℕ) : ℕ := 
  num_toppings * cost_per_topping

theorem first_topping_cost
  (num_slices : ℕ) (cost_per_slice : ℕ) (pizza_cost : ℕ) 
  (num_1_dollar_toppings : ℕ) (cost_1_dollar_topping : ℕ)
  (num_50_cent_toppings : ℕ) (cost_50_cent_topping : ℕ)
  (total_cost_with_toppings : ℕ) (first_topping_cost : ℕ) :
  num_slices = 8 →
  cost_per_slice = 2 →
  pizza_cost = 10 →
  num_1_dollar_toppings = 2 →
  cost_1_dollar_topping = 1 →
  num_50_cent_toppings = 4 →
  cost_50_cent_topping = 0.5 →
  total_cost_with_toppings = (num_slices * cost_per_slice) →
  first_topping_cost + (num_1_dollar_toppings * cost_1_dollar_topping) + (num_50_cent_toppings * cost_50_cent_topping) = 
  (total_cost_with_toppings - pizza_cost) →
  first_topping_cost = 2 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 
  sorry

-- end of Lean statement

end first_topping_cost_l9_9895


namespace complex_exp_form_pow_four_l9_9669

theorem complex_exp_form_pow_four :
  let θ := 30 * Real.pi / 180
  let cos_θ := Real.cos θ
  let sin_θ := Real.sin θ
  let z := 3 * (cos_θ + Complex.I * sin_θ)
  z ^ 4 = -40.5 + 40.5 * Complex.I * Real.sqrt 3 :=
by
  sorry

end complex_exp_form_pow_four_l9_9669


namespace length_of_chord_AB_equation_of_line_l_l9_9056

theorem length_of_chord_AB (l : ℝ → ℝ) (P A B : ℝ × ℝ) :
  P = (1, 0) ∧ (l 0 = P.2) ∧ l 1 = 1 ∧
  (∃ y : ℝ, (y - 1)^2 = 2 * (y - 1) ∧ (y = A.2 ∧ y - 1 = A.1) ∧ (y = B.2 ∧ y - 1 = B.1)) →
  dist A B = 2 * real.sqrt 6 :=
sorry

theorem equation_of_line_l (l : ℝ → ℝ) (P A B : ℝ × ℝ) :
  P = (1, 0) ∧ (l 0 = P.2) ∧ (l 1 = 1) ∧
  (∃ y1 y2 : ℝ, y1^2 = 2*(y1 - 1) ∧ y2^2 = 2*(y2 - 1) ∧
   A = (y1, y1 - 1) ∧ B = (y2, y2 - 1) ∧ (1 - A.1, A.2) = -2 * (1 - B.1, B.2)) →
  (l = λ y, (y - 1) / 2 + 1 ∨ l = λ y, -(y - 1) / 2 + 1) :=
sorry

end length_of_chord_AB_equation_of_line_l_l9_9056


namespace unique_value_of_n_l9_9207

-- Definitions based on the conditions:
def unit_digit (n : ℕ) : ℕ := n % 10
def num_divisors (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (λ m, n % m = 0).card

theorem unique_value_of_n (n : ℕ) (h1 : n < 60) (h2 : unit_digit n = 0) (h3 : ∃ p q, prime p ∧ prime q ∧ p ≠ q ∧ p ∣ n ∧ q ∣ n) 
(h4 : ∀ m, m < 60 → unit_digit m = 0 → ∃ p q, prime p ∧ prime q ∧ p ≠ q ∧ p ∣ m ∧ q ∣ m → ¬(num_divisors m = num_divisors n)) 
: n = 10 := 
by
  sorry

end unique_value_of_n_l9_9207


namespace exists_line_through_D_dividing_area_l9_9761

-- Definition of the problem conditions

variables {A B C C1 D G H : Point}
variables (triangle_ABC : Triangle)
variables (isosceles_triangle : IsoscelesTriangle triangle_ABC)
variables (altitude_CC1 : Altitude triangle_ABC C C1)
variables (point_D_on_CC1 : OnLine D altitude_CC1)

-- Proper theorem statement
theorem exists_line_through_D_dividing_area (triangle_ABC : Triangle)
  (isosceles_triangle : IsoscelesTriangle triangle_ABC)
  (altitude_CC1 : Altitude triangle_ABC C C1)
  (point_D_on_CC1 : OnLine D altitude_CC1):
  ∃ (G H : Point), LineSegment GH passes_through D ∧
  divides_into_equal_area triangle_ABC ( RegionOf LineSegment GH) :=
sorry

end exists_line_through_D_dividing_area_l9_9761


namespace karnataka_student_probability_l9_9597

/-- Out of 10 students in a class, 4 are from Maharashtra, 3 are from Karnataka, and 3 are from Goa.
    Given that 4 students are selected at random, prove that the probability that at least one of them
    is from Karnataka is 5/6. -/
theorem karnataka_student_probability :
  let total_students := 10 in
  let maharashtra_students := 4 in
  let karnataka_students := 3 in
  let goa_students := 3 in
  let selected_students := 4 in
  (∀ (total_students maharashtra_students karnataka_students goa_students selected_students),
    total_students = 10 → 
    maharashtra_students = 4 → 
    karnataka_students = 3 → 
    goa_students = 3 → 
    selected_students = 4 → 
  (1 - (↑((nat.choose 7 4) : ℚ) / ↑((nat.choose 10 4) : ℚ)) = 5 / 6)) := 
by sorry

end karnataka_student_probability_l9_9597


namespace alice_bob_meet_after_12_turns_l9_9255

-- Define the variables and their movements
noncomputable def alice_moves_per_turn : ℕ := 7
noncomputable def bob_moves_per_turn : ℕ := 12 - 4 -- Bob moves counterclockwise

-- Define the relative movement per turn
noncomputable def relative_movement_per_turn : ℕ := alice_moves_per_turn - bob_moves_per_turn

-- Prove that Alice and Bob meet after 12 turns
theorem alice_bob_meet_after_12_turns (k : ℕ) : 
  (relative_movement_per_turn * k) % 12 = 0 :=
begin
  -- Slot for the proof, which confirms that 12 is the minimum number of turns.
  sorry
end

end alice_bob_meet_after_12_turns_l9_9255


namespace range_of_p_l9_9887

variables {A B C a b c p : ℝ}
variables (h1 : 0 < A) (h2 : A < π / 2)
variables (h3 : 0 < B) (h4 : B < π / 2)
variables (h5 : 0 < C) (h6 : C < π / 2)
variables (h7 : a > 0) (h8 : b > 0) (h9 : c > 0)

-- Main condition given in the problem
variable (h_cond : sqrt 3 * sin B - c * sin B * (sqrt 3 * sin C - c * sin C) = 4 * cos B * cos C)

-- Additional condition given in the problem
variable (h_sinB_eq_p_sinC : sin B = p * sin C)

-- Goal: To find the range of p
theorem range_of_p (h_acute : A + B + C = π) : 1/2 < p ∧ p < 2 :=
sorry

end range_of_p_l9_9887


namespace angle_relation_l9_9260

-- Defining points and angles
variables (A B C D E : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
variables (AC CD DA BC DE : ℝ)
variables (angle_BAC angle_BAE : ℝ)

-- Given conditions
def conditions : Prop :=
  AC = CD ∧ CD = DA ∧ DA = BC ∧ BC = DE

-- Proof that angle BAE is four times angle BAC given conditions
theorem angle_relation (h : conditions AC CD DA BC DE) : angle_BAE = 4 * angle_BAC :=
by
  sorry

end angle_relation_l9_9260


namespace maximum_height_of_projectile_l9_9628

def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 36

theorem maximum_height_of_projectile : ∀ t : ℝ, (h t ≤ 116) :=
by sorry

end maximum_height_of_projectile_l9_9628


namespace circle_inequality_l9_9066

-- Given a circle of 100 pairwise distinct numbers a : ℕ → ℝ for 1 ≤ i ≤ 100
variables {a : ℕ → ℝ}
-- Hypothesis 1: distinct numbers
def distinct_numbers (a : ℕ → ℝ) := ∀ i j : ℕ, (1 ≤ i ∧ i ≤ 100) ∧ (1 ≤ j ∧ j ≤ 100) ∧ (i ≠ j) → a i ≠ a j

-- Theorem: Prove that there exist four consecutive numbers such that the sum of the first and the last number is strictly greater than the sum of the two middle numbers
theorem circle_inequality (h_distinct : distinct_numbers a) : 
  ∃ i : ℕ, (1 ≤ i ∧ i ≤ 100) ∧ (a i + a ((i + 3) % 100) > a ((i + 1) % 100) + a ((i + 2) % 100)) :=
sorry

end circle_inequality_l9_9066


namespace find_H_l9_9933

-- Define the vertices of the parallelogram
def E : ℝ × ℝ := (3, 6)
def F : ℝ × ℝ := (5, 10)
def G : ℝ × ℝ := (7, 6)
def H : ℝ × ℝ := (5, 2)

-- Reflection of a point across the x-axis
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Reflection of a point across the line y = x + 2
def reflect_y_eq_x_plus_2 (p : ℝ × ℝ) : ℝ × ℝ :=
let p_translated := (p.1, p.2 - 2) in -- translate down by 2
let p_reflected := (p_translated.2, p_translated.1) in -- reflect across y = x
(p_reflected.1, p_reflected.2 + 2) -- translate up by 2

-- Prove that reflecting H across the x-axis and then the line y = x + 2 gives H'' = (-4, 7)
theorem find_H'' : reflect_y_eq_x_plus_2 (reflect_x_axis H) = (-4, 7) :=
by {
  sorry
}

end find_H_l9_9933


namespace probability_of_multiple_of_105_l9_9442

theorem probability_of_multiple_of_105 :
  let s := {5, 7, 14, 25, 35, 49, 63}
  let pairs := (finset.univ.product finset.univ).filter (λ p : ℕ × ℕ, p.1 ∈ s ∧ p.2 ∈ s ∧ p.1 < p.2)
  let valid_pairs := pairs.filter (λ p : ℕ × ℕ, (p.1 * p.2) % 105 = 0)
  (valid_pairs.card : ℚ) / pairs.card = 1 / 21 := 
by {
  -- definitions and conditions are directly used here without solution steps
  sorry
}

end probability_of_multiple_of_105_l9_9442


namespace part_a_part_b_l9_9500

noncomputable def has_claire_strategy (n : ℕ) (x : ℕ → ℝ) : Prop :=
∀ B : ℝ, ∃ x' : ℕ → ℝ, (∀ i < n, x' i ≠ x i) ∧ ∀ m :(fin n → ℝ) → fin n → ℝ, 
  (∃ k : ℕ, k < B * n * log n ∧ increasing m) → ¬ (∀ i : fin n, x' i = m i) 

noncomputable def william_strategy (n : ℕ) : Prop :=
∃ A : ℝ, ∀ (x : ℕ → ℝ), ∃ m :(fin n → ℝ) → fin n → ℝ, increasing m ∧ 
  ∀ k : ℕ, k ≤ A * n * log n 

theorem part_a (n : ℕ) (h : 1 < n) : 
  william_strategy n := sorry

theorem part_b (n : ℕ) (h : 1 < n) :
  ∃ B : ℝ, has_claire_strategy n := sorry

end part_a_part_b_l9_9500


namespace problem_l9_9387

def setA : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}
def setB : Set ℝ := {x : ℝ | x ≤ 3}

theorem problem : setA ∩ setB = setA := sorry

end problem_l9_9387


namespace smallest_prime_after_six_nonprimes_l9_9162

open Nat

/-- 
  Proposition:
    97 is the smallest prime number that occurs after a sequence 
    of six consecutive positive integers all of which are nonprime.
-/
theorem smallest_prime_after_six_nonprimes : ∃ (n : ℕ), (n > 0 ∧ Prime (n + 97)) ∧ 
  (∀ k : ℕ, k < n → (Nonprime k) ∧ Nonprime (k+1) ∧ Nonprime (k+2) ∧ Nonprime (k+3) ∧ Nonprime (k+4) ∧ Nonprime (k+5)) ∧ 
  (Prime (n + 1) → n + 1 = 97) := 
by 
  sorry

end smallest_prime_after_six_nonprimes_l9_9162


namespace area_difference_quarter_circles_l9_9259

theorem area_difference_quarter_circles :
  let r1 := 28
  let r2 := 14
  let pi := (22 / 7)
  let quarter_area_big := (1 / 4) * pi * r1^2
  let quarter_area_small := (1 / 4) * pi * r2^2
  let rectangle_area := r1 * r2
  (quarter_area_big - (quarter_area_small + rectangle_area)) = 70 := by
  -- Placeholder for the proof
  sorry

end area_difference_quarter_circles_l9_9259


namespace smallest_possible_value_l9_9357

theorem smallest_possible_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 / (x : ℝ)) + (1 / (y : ℝ)) = 1 / 15) : x + y = 64 :=
sorry

end smallest_possible_value_l9_9357


namespace complex_power_result_l9_9681

theorem complex_power_result :
  (3 * Complex.cos (Real.pi / 6) + 3 * Complex.sin (Real.pi / 6) * Complex.i)^4 = 
  -40.5 + 40.5 * Complex.i * Real.sqrt 3 :=
by sorry

end complex_power_result_l9_9681


namespace cell_count_3_or_more_conditions_meet_l9_9729

theorem cell_count_3_or_more_conditions_meet :
  let n := 50
  ∃ (grid : fin n.succ × fin n.succ → ℕ),
    (∀ (i j : fin n.succ), // For every cell in the grid
      grid (i, j) = 0 ∨ grid (i, j) ≥ 3) ∧ // Each cell contains a number ≥ 3 or is 0
    (∑ i in finset.range n.succ,   
        ∑ j in finset.range n.succ, if grid (i, j) ≥ 3 then 1 else 0) = 1600 := 
begin
  -- Proof details would go here, but they are not required.
  sorry
end

end cell_count_3_or_more_conditions_meet_l9_9729


namespace product_seq_2012_eq_one_l9_9800

-- Define the sequence recursively
def seq : ℕ → ℚ
| 0       := 2
| (n + 1) := (1 + seq n) / (1 - seq n)

-- Define the product of the first 2012 terms of the sequence
def product_first_2012_terms : ℚ :=
List.prod (List.map seq (List.range 2012))

-- Prove that the product is equal to 1
theorem product_seq_2012_eq_one : product_first_2012_terms = 1 :=
sorry

end product_seq_2012_eq_one_l9_9800


namespace total_carrots_l9_9476

def Joan_carrots : ℕ := 29
def Jessica_carrots : ℕ := 11

theorem total_carrots : Joan_carrots + Jessica_carrots = 40 := by
  sorry

end total_carrots_l9_9476


namespace smallest_possible_value_l9_9359

theorem smallest_possible_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 / (x : ℝ)) + (1 / (y : ℝ)) = 1 / 15) : x + y = 64 :=
sorry

end smallest_possible_value_l9_9359


namespace value_of_X_is_73_over_2_l9_9126

noncomputable def X_value : ℚ :=
  let a_row := 24
  let a_col1 := 16
  let b_col1 := 20
  let c_col2 := -14
  -- Arithmetic sequences in row
  let d_row := -16/3
  let next_after_a_row := a_row + d_row
  let second_in_row := a_row + 2 * d_row
  let third_in_row := a_row + 3 * d_row
  -- Arithmetic sequences in the first column
  let d_col1 := 4
  let prior_a_col1 := a_col1 - d_col1
  let prior_prior_a_col1 := prior_a_col1 - d_col1
  -- Arithmetic sequences in the second column
  -- x corresponds to X in the diagram
  let d_col2 := (c_col2 - third_in_row) / 4
  let x := third_in_row - d_col2 in
  x

theorem value_of_X_is_73_over_2 : X_value = 73 / 2 :=
by
  sorry

end value_of_X_is_73_over_2_l9_9126


namespace cost_to_fill_bathtub_with_jello_l9_9471

-- Define the conditions
def pounds_per_gallon : ℝ := 8
def gallons_per_cubic_foot : ℝ := 7.5
def cubic_feet_of_water : ℝ := 6
def tablespoons_per_pound : ℝ := 1.5
def cost_per_tablespoon : ℝ := 0.5

-- The theorem stating the cost to fill the bathtub with jello
theorem cost_to_fill_bathtub_with_jello : 
  let total_gallons := cubic_feet_of_water * gallons_per_cubic_foot in
  let total_pounds := total_gallons * pounds_per_gallon in
  let total_tablespoons := total_pounds * tablespoons_per_pound in
  let total_cost := total_tablespoons * cost_per_tablespoon in
  total_cost = 270 := 
by {
  -- Here's where we would provide the proof steps, but just add sorry to skip it
  sorry
}

end cost_to_fill_bathtub_with_jello_l9_9471


namespace parabola_focus_distance_l9_9827

theorem parabola_focus_distance (x y : ℝ) (h1 : y^2 = 4 * x) (h2 : (x - 1)^2 + y^2 = 100) : x = 9 :=
sorry

end parabola_focus_distance_l9_9827


namespace smallest_prime_after_six_nonprimes_l9_9160

open Nat

/-- 
  Proposition:
    97 is the smallest prime number that occurs after a sequence 
    of six consecutive positive integers all of which are nonprime.
-/
theorem smallest_prime_after_six_nonprimes : ∃ (n : ℕ), (n > 0 ∧ Prime (n + 97)) ∧ 
  (∀ k : ℕ, k < n → (Nonprime k) ∧ Nonprime (k+1) ∧ Nonprime (k+2) ∧ Nonprime (k+3) ∧ Nonprime (k+4) ∧ Nonprime (k+5)) ∧ 
  (Prime (n + 1) → n + 1 = 97) := 
by 
  sorry

end smallest_prime_after_six_nonprimes_l9_9160


namespace log_eq_log_eq_a_eq_b_l9_9414

open Real

theorem log_eq_log_eq_a_eq_b :
  (a b : ℝ) (ha : a = logBase 16 400) (hb : b = logBase 4 20) : 
  a = b :=
by
  sorry

end log_eq_log_eq_a_eq_b_l9_9414


namespace horses_at_starting_point_l9_9574

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17]
def min_time := 210
def sum_of_digits := 3

theorem horses_at_starting_point : 
  ∃ T > 0, (T = min_time) ∧ (T.digits.sum = sum_of_digits) ∧ 
    (primes.count (λ p, T % p = 0) ≥ 4) :=
begin
  sorry
end

end horses_at_starting_point_l9_9574


namespace buses_required_is_12_l9_9245

-- Define the conditions given in the problem
def students : ℕ := 535
def bus_capacity : ℕ := 45

-- Define the minimum number of buses required
def buses_needed (students : ℕ) (bus_capacity : ℕ) : ℕ :=
  (students + bus_capacity - 1) / bus_capacity

-- The theorem stating the number of buses required is 12
theorem buses_required_is_12 :
  buses_needed students bus_capacity = 12 :=
sorry

end buses_required_is_12_l9_9245


namespace problem_l9_9142

open Real

def average (scores : List ℝ) : ℝ :=
  (scores.sum) / (scores.length : ℝ)

def variance (scores : List ℝ) : ℝ :=
  let avg := average scores
  (scores.map (λ x => (x - avg)^2)).sum / (scores.length : ℝ)

theorem problem (s_A : List ℝ) (s_B : List ℝ) 
  (hA : s_A = [11, 16, 23, 37, 39, 42, 48])
  (hB : s_B = [15, 26, 28, 30, 33, 34, 44]) :
  average s_A > average s_B ∧ variance s_A > variance s_B :=
by
  sorry

end problem_l9_9142


namespace trigonometric_order_l9_9331

variables (l : ℝ)
# Use noncomputable for definitions involving transcendental functions
noncomputable def a := Real.sin l
noncomputable def b := Real.tan l
noncomputable def c := Real.tan (9 / 2)

theorem trigonometric_order (hl1 : Real.pi / 4 < l) (hl2 : l < Real.pi / 3) :
  a l < b l ∧ b l < c :=
by
  -- placeholder for proof
  sorry

end trigonometric_order_l9_9331


namespace calculate_total_cost_l9_9229

-- Define the cost for type A and type B fast foods as constants
def cost_of_type_A : ℕ := 30
def cost_of_type_B : ℕ := 20

-- Define the number of servings as variables
variables (a b : ℕ)

-- Define a function that calculates the total cost
def total_cost (a b : ℕ) : ℕ :=
  cost_of_type_A * a + cost_of_type_B * b

-- The theorem statement: Prove that the total cost is as calculated
theorem calculate_total_cost (a b : ℕ) :
  total_cost a b = 30 * a + 20 * b := by
  unfold total_cost
  simp

end calculate_total_cost_l9_9229


namespace maximize_matches_l9_9620

theorem maximize_matches (n : ℕ) (h : n ≥ 3) (r s t : ℕ) (h_sum : r + s + t = n)
    (h_max : ∀ r' s' t' : ℕ, r' + s' + t' = n → r * s + s * t + t * r ≥ r' * s' + s' * t' + t' * r') :
    |r - s| ≤ 1 ∧ |s - t| ≤ 1 ∧ |r - t| ≤ 1 :=
sorry

end maximize_matches_l9_9620


namespace a_n_divisible_by_11_l9_9987

-- Define the sequence
def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ 
  a 2 = 3 ∧ 
  ∀ n, a (n + 2) = (n + 3) * a (n + 1) - (n + 2) * a n

-- Main statement
theorem a_n_divisible_by_11 (a : ℕ → ℤ) (h : seq a) :
  ∀ n, ∃ k : ℕ, a n % 11 = 0 ↔ n = 4 + 11 * k :=
sorry

end a_n_divisible_by_11_l9_9987


namespace players_joined_l9_9136

noncomputable def initial_friends : ℕ := 7
noncomputable def lives_per_player : ℕ := 7
noncomputable def total_lives : ℕ := 63

theorem players_joined (initial_friends lives_per_player total_lives : ℕ) (h1 : initial_friends = 7) (h2 : lives_per_player = 7) (h3 : total_lives == 63) : 
  ((total_lives - (initial_friends * lives_per_player)) / lives_per_player) = 2 :=
by
  rw [h1, h2, h3]
  sorry

end players_joined_l9_9136


namespace S₄_eq_15_l9_9842

-- Definitions based on the given conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum a

def sequence_condition (a : ℕ → ℝ) : Prop :=
  is_geometric_sequence a ∧ a 1 = 1 ∧ sum_of_first_n_terms a 5 = 5 * sum_of_first_n_terms a 3 - 4

theorem S₄_eq_15 (a : ℕ → ℝ) (q : ℝ) :
  sequence_condition a →
  (∀ n, a n = 1 * q ^ (n-1)) → 
  sum_of_first_n_terms a 4 = 15 :=
sorry

end S₄_eq_15_l9_9842


namespace complex_power_not_real_l9_9504

theorem complex_power_not_real (a b : ℕ) (n : ℕ) (h_gcd : Nat.gcd a b = 1) :
  ¬(IsReal (((Real.sqrt a) + (Complex.I * (Real.sqrt b))) ^ n)) ↔ ¬(⟨a, b⟩ ≠ ⟨1, 1⟩ ∧ ⟨a, b⟩ ≠ ⟨1, 3⟩ ∧ ⟨a, b⟩ ≠ ⟨3, 1⟩) :=
sorry

end complex_power_not_real_l9_9504


namespace units_digit_of_n_squared_plus_2_n_is_7_l9_9495

def n : ℕ := 2023 ^ 2 + 2 ^ 2023

theorem units_digit_of_n_squared_plus_2_n_is_7 : (n ^ 2 + 2 ^ n) % 10 = 7 := 
by
  sorry

end units_digit_of_n_squared_plus_2_n_is_7_l9_9495


namespace complex_power_l9_9688

def cos_30 := Float.sqrt 3 / 2
def sin_30 := 1 / 2

theorem complex_power (i : ℂ) : 
  (3 * (⟨cos_30, sin_30⟩ : ℂ)) ^ 4 = -40.5 + 40.5 * i * Float.sqrt 3 := 
  sorry

end complex_power_l9_9688


namespace sum_of_interior_angles_l9_9974

-- Define the conditions:
def exterior_angle (n : ℕ) := 45

def sum_exterior_angles := 360

-- Define the Lean statement for the proof problem
theorem sum_of_interior_angles : ∃ n : ℕ, 
  sum_exterior_angles / exterior_angle n = n ∧
  (180 * (n - 2) = 1080) :=
by
  use 8
  split
  calc
    sum_exterior_angles / exterior_angle 8 = 360 / 45 := rfl
    ... = 8 := rfl
  calc
    180 * (8 - 2) = 180 * 6 := rfl
    ... = 1080 := rfl

end sum_of_interior_angles_l9_9974


namespace a_lt_1_sufficient_but_not_necessary_l9_9194

noncomputable def represents_circle (a : ℝ) : Prop :=
  a^2 - 10 * a + 9 > 0

theorem a_lt_1_sufficient_but_not_necessary (a : ℝ) :
  represents_circle a → ((a < 1) ∨ (a > 9)) :=
sorry

end a_lt_1_sufficient_but_not_necessary_l9_9194


namespace stock_value_end_of_third_year_l9_9513

-- Definitions based on given conditions
def initial_investment : ℝ := 1620
def dividend_rate : ℝ := 0.08
def total_earnings : ℝ := 135
def market_increase_1st_year : ℝ := 0.03
def market_decrease_2nd_year : ℝ := 0.02
def annual_inflation_rate : ℝ := 0.02

-- Calculating the quoted value at the end of the third year
def original_stock_value (earnings : ℝ) (rate : ℝ) : ℝ := (earnings * 100) / (rate * 100)
def value_after_1st_year (original_value : ℝ) (increase_rate : ℝ) : ℝ := original_value + (original_value * increase_rate)
def value_after_2nd_year (value_1 : ℝ) (decrease_rate : ℝ) : ℝ := value_1 - (value_1 * decrease_rate)
def inflation_adjusted_value (value_2 : ℝ) (inflation_rate : ℝ) (years : ℕ) : ℝ := value_2 - (value_2 * inflation_rate * years)

def quoted_value_after_3_years : ℝ :=
  let original_value := original_stock_value total_earnings dividend_rate
  let value_1 := value_after_1st_year original_value market_increase_1st_year
  let value_2 := value_after_2nd_year value_1 market_decrease_2nd_year
  inflation_adjusted_value value_2 annual_inflation_rate 3

-- Proof that the calculated quoted value is equal to Rs. 1601.16
theorem stock_value_end_of_third_year :
  quoted_value_after_3_years = 1601.16 := by
  sorry

end stock_value_end_of_third_year_l9_9513


namespace problem_part1_problem_part2_l9_9340

noncomputable def quadratic_roots_conditions (x1 x2 m : ℝ) : Prop :=
  (x1 = 1) ∧ (x1 + x2 = 6) ∧ (x1 * x2 = 2 * m - 1)

noncomputable def existence_of_m (x1 x2 : ℝ) (m : ℝ) : Prop :=
  (x1 = 1) ∧ (x1 + x2 = 6) ∧ (x1 * x2 = 2 * m - 1) ∧ ((x1 - 1) * (x2 - 1) = 6 / (m - 5))

theorem problem_part1 : 
  ∃ x2 m, quadratic_roots_conditions 1 x2 m :=
sorry

theorem problem_part2 :
  ∃ m, ∃ x2, existence_of_m 1 x2 m ∧ m ≤ 5 :=
sorry

end problem_part1_problem_part2_l9_9340


namespace AL_less_than_KL_l9_9490

-- Define the given conditions as Lean definitions and statements
def circumcenter (A B C O : Point) : Prop := is_circumcenter A B C O
def midpoint_arc (B C K : Point) : Prop := is_midpoint_arc B C K
def lies_on_line (A L K : Point) : Prop := lies_on A L K
def similar_triangles (A H L K M L' : Point) : Prop := is_similar (triangle A H L) (triangle K M L')

-- Define the proof obligation (the theorem to prove)
theorem AL_less_than_KL (A B C O K L H M : Point)
  (hcirc : circumcenter A B C O)
  (hmid : midpoint_arc B C K)
  (hlines : lies_on_line A L K)
  (hsimilar : similar_triangles A H L K M L):
  AL < KL := sorry

end AL_less_than_KL_l9_9490


namespace other_problem_points_l9_9238

theorem other_problem_points
  (total_points : ℕ)
  (total_problems : ℕ)
  (four_point_value : ℕ)
  (num_four_point_problems : ℕ)
  (other_problem_points : ℕ)
  (correct_answer : ℕ)
  (total_points = 100)
  (total_problems = 30)
  (four_point_value = 4)
  (num_four_point_problems = 10)
  (correct_answer = 3) :
  ∃ (x : ℕ), x = correct_answer ∧
    total_points = num_four_point_problems * four_point_value + (total_problems - num_four_point_problems) * x :=
sorry

end other_problem_points_l9_9238


namespace intersection_complement_l9_9413

-- Define the universal set
def U : Set ℝ := Set.univ

-- Define the sets M and N
def M : Set ℝ := {x | log 2 x < 1}
def N : Set ℝ := {x | x ≥ 1}

-- Prove the intersection of M with the complement of N in U equals to {x | 0 < x < 1}
theorem intersection_complement : M ∩ (U \ N) = {x | 0 < x < 1} := by
  sorry

end intersection_complement_l9_9413


namespace team_formation_count_l9_9322

-- Definitions of conditions
def female_teachers := 4
def male_teachers := 5
def team_size := 3
def both_gender_included := ∀ (team : list (bool × ℕ)), 
  team.length = team_size ∧ 
  (∃ t ∈ team, t.fst = true) ∧ 
  (∃ t ∈ team, t.fst = false)

-- Lean statement to prove the number of different team formation plans
theorem team_formation_count :
  ∃ (n : ℕ), n = 70 ∧ 
  ∃ (teams : list (list (bool × ℕ))), 
    (∀ team ∈ teams, team.length = team_size) ∧ 
    (∀ team ∈ teams, both_gender_included team) := 
begin
  sorry
end

end team_formation_count_l9_9322


namespace special_numbers_count_l9_9815

-- Define conditions
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def ends_with_zero (n : ℕ) : Prop := n % 10 = 0
def divisible_by_30 (n : ℕ) : Prop := n % 30 = 0

-- Define the count of numbers with the specified conditions
noncomputable def count_special_numbers : ℕ :=
  (9990 - 1020) / 30 + 1

-- The proof problem
theorem special_numbers_count : count_special_numbers = 300 := sorry

end special_numbers_count_l9_9815


namespace smallest_sum_of_xy_l9_9366

theorem smallest_sum_of_xy (x y : ℕ) (h1 : x ≠ y) (h2 : 0 < x) (h3 : 0 < y) 
  (h4 : (1:ℚ)/x + (1:ℚ)/y = (1:ℚ)/15) : x + y = 64 :=
sorry

end smallest_sum_of_xy_l9_9366


namespace B_elements_l9_9803

def B : Set ℤ := {x | -3 < 2 * x - 1 ∧ 2 * x - 1 < 3}

theorem B_elements : B = {-1, 0, 1} :=
by
  sorry

end B_elements_l9_9803


namespace equation_has_more_than_four_real_solutions_l9_9102

theorem equation_has_more_than_four_real_solutions :
  ∃ S : Set ℝ, (|x - 2| + x - 2 = 0 ∀ x ∈ S) ∧ Set.Infinite S :=
sorry

end equation_has_more_than_four_real_solutions_l9_9102


namespace smallest_x_plus_y_l9_9376

theorem smallest_x_plus_y (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_neq : x ≠ y) (h_eq : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 15) : x + y = 64 :=
sorry

end smallest_x_plus_y_l9_9376


namespace sum_of_coefficients_eq_zero_l9_9787

theorem sum_of_coefficients_eq_zero 
  (A B C D E F : ℝ) :
  (∀ x, (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) 
  = A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 :=
by sorry

end sum_of_coefficients_eq_zero_l9_9787


namespace min_value_of_x2_add_y2_l9_9441

theorem min_value_of_x2_add_y2 (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 14^2) : x^2 + y^2 ≥ 1 :=
sorry

end min_value_of_x2_add_y2_l9_9441


namespace inequality_holds_l9_9320

variable (a b c d : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hc : c > 0)
variable (hd : d > 0)
variable (h : 1 / (a^3 + 1) + 1 / (b^3 + 1) + 1 / (c^3 + 1) + 1 / (d^3 + 1) = 2)

theorem inequality_holds (ha : a > 0)
  (hb : b > 0) 
  (hc : c > 0) 
  (hd : d > 0) 
  (h : 1 / (a^3 + 1) + 1 / (b^3 + 1) + 1 / (c^3 + 1) + 1 / (d^3 + 1) = 2) :
  (1 - a) / (a^2 - a + 1) + (1 - b) / (b^2 - b + 1) + (1 - c) / (c^2 - c + 1) + (1 - d) / (d^2 - d + 1) ≥ 0 := by
  sorry

end inequality_holds_l9_9320


namespace quadratic_sequence_l9_9712

theorem quadratic_sequence (a x₁ b x₂ c : ℝ)
  (h₁ : a + b = 2 * x₁)
  (h₂ : x₁ + x₂ = 2 * b)
  (h₃ : a + c = 2 * b)
  (h₄ : x₁ + x₂ = -6 / a)
  (h₅ : x₁ * x₂ = c / a) :
  b = -2 * a ∧ c = -5 * a :=
by
  sorry

end quadratic_sequence_l9_9712


namespace part_a_part_b_l9_9605

noncomputable section

open Real

theorem part_a (x y z : ℝ) (hx : x ≠ 1) (hy : y ≠ 1) (hz : z ≠ 1) (hxyz : x * y * z = 1) :
  (x^2 / (x-1)^2) + (y^2 / (y-1)^2) + (z^2 / (z-1)^2) ≥ 1 :=
sorry

theorem part_b : ∃ (infinitely_many : ℕ → (ℚ × ℚ × ℚ)), 
  ∀ n, ((infinitely_many n).1.1 ≠ 1) ∧ ((infinitely_many n).1.2 ≠ 1) ∧ ((infinitely_many n).2 ≠ 1) ∧ 
  ((infinitely_many n).1.1 * (infinitely_many n).1.2 * (infinitely_many n).2 = 1) ∧ 
  ((infinitely_many n).1.1^2 / ((infinitely_many n).1.1 - 1)^2 + 
   (infinitely_many n).1.2^2 / ((infinitely_many n).1.2 - 1)^2 + 
   (infinitely_many n).2^2 / ((infinitely_many n).2 - 1)^2 = 1) :=
sorry

end part_a_part_b_l9_9605


namespace problem_l9_9386

def setA : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}
def setB : Set ℝ := {x : ℝ | x ≤ 3}

theorem problem : setA ∩ setB = setA := sorry

end problem_l9_9386


namespace problem_inverse_range_m_l9_9748

theorem problem_inverse_range_m (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 2 / x + 1 / y = 1) : 
  (2 * x + y > m^2 + 8 * m) ↔ (m > -9 ∧ m < 1) := 
by
  sorry

end problem_inverse_range_m_l9_9748


namespace three_digit_even_numbers_count_l9_9279

def count_even_three_digit_numbers : Nat :=
  let digits := {0, 1, 2, 3, 4}
  let even_digits := {0, 2, 4}
  let all_except_zero := {1, 2, 3, 4}
  let count_case1 := 4 * 3 -- Fix units digit to 0 and choose and arrange 2 from {1, 2, 3, 4}
  let count_case2 := 2 * 3 * 3 -- Choose a unit digit from 2 or 4, then arrange two from remaining
  count_case1 + count_case2 

theorem three_digit_even_numbers_count : count_even_three_digit_numbers = 30 := by
  sorry

end three_digit_even_numbers_count_l9_9279


namespace one_correct_l9_9770

variables (m n a : Line)
variables (α β : Plane)

-- Hypotheses
def prop1 := m ⊆ α ∧ n ⊆ α ∧ ¬(m ∥ β) ∧ ¬(n ∥ β) → ¬(α ∥ β)
def prop2 := ¬(n ∥ m) ∧ n ⊥ α → m ⊥ α
def prop3 := ¬(α ∥ β) ∧ m ⊆ α ∧ n ⊆ β → ¬(m ∥ n)
def prop4 := m ⊥ a ∧ m ⊥ n → ¬(n ∥ a)

theorem one_correct :
  (prop1 m n α β → false) ∧
  (prop2 m n α → false) ∧
  (prop3 m n α β → false) ∧
  (prop4 m n a) :=
sorry

end one_correct_l9_9770


namespace dot_product_of_vectors_in_triangle_l9_9443

def AB := 7
def BC := 5
def AC := 6

theorem dot_product_of_vectors_in_triangle : (7 * 6 * (5 / 7) = 30) :=
by 
  have h_cosA : cos (λ.(((AB ^ 2 + AC ^ 2 - BC ^ 2) / (2 * AB * AC)))) = 5/7 := 
    by sorry
  have h_dot_product : AB * AC * (5 / 7) = 30 :=
    by linarith [h_cosA, AB, AC]
  exact h_dot_product

end dot_product_of_vectors_in_triangle_l9_9443


namespace value_of_c_range_of_omega_l9_9394

theorem value_of_c (a b A B C : ℝ) (h1 : a * Real.sin A - b * Real.sin B = 2 * Real.sin (A - B)) (h2 : a ≠ b) : 
  let c := 2 in
  c = 2 :=
sorry

theorem range_of_omega (ω : ℝ) : 
  (7 / 4 < ω) ∧ (ω ≤ 11 / 4) ↔ ∀ x ∈ Ioo (0 : ℝ) Real.pi, 
    let f := λ x, Real.sin (x - Real.pi / 4) + 2 in
    ∃ z₁ z₂, z₁ ≠ z₂ ∧ (z₁, z₂ ∈ f x) :=
sorry

end value_of_c_range_of_omega_l9_9394


namespace smallest_x_plus_y_l9_9374

theorem smallest_x_plus_y (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_neq : x ≠ y) (h_eq : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 15) : x + y = 64 :=
sorry

end smallest_x_plus_y_l9_9374


namespace PH_passes_midpoint_MN_l9_9452
-- Lean 4 statement follows


variables {α : Type*} [EuclideanGeometry α]

-- Points A, B, and C are vertices of the acute triangle ABC
variables (A B C M N H P : α)

-- Conditions of the problem
variable (acute_triangle : A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ ∠A < π/2 ∧ ∠B < π/2 ∧ ∠C < π/2)
variable (mid_M : M = midpoint A B)
variable (mid_N : N = midpoint B C)
variable (foot_H : H = foot (altitude B))
variable (circ_intersect : P ≠ H ∧ (∃ ω1 ω2 : circle, circumscribed ω1 (triangle A H N) ∧ circumscribed ω2 (triangle C H M) ∧ P ∈ intersection_points ω1 ω2))

-- Theorem statement to prove
theorem PH_passes_midpoint_MN
    (acute_triangle : A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ ∠A < π/2 ∧ ∠B < π/2 ∧ ∠C < π/2)
    (mid_M : M = midpoint A B)
    (mid_N : N = midpoint B C)
    (foot_H : H = foot (altitude B))
    (circ_intersect : P ≠ H ∧ (∃ ω1 ω2 : circle, circumscribed ω1 (triangle A H N) ∧ circumscribed ω2 (triangle C H M) ∧ P ∈ intersection_points ω1 ω2)) :
    ∃ S : α, S = midpoint M N ∧ collinear {P, H, S} :=
begin
    sorry
end

end PH_passes_midpoint_MN_l9_9452


namespace smallest_prime_after_six_consecutive_nonprimes_l9_9177

-- Define what it means to be prime
def is_prime (n : ℕ) : Prop :=
  nat.prime n

-- Define the sequence of six consecutive nonprime numbers
def consecutive_nonprime_sequence (a : ℕ) : Prop :=
  ¬ is_prime a ∧ ¬ is_prime (a + 1) ∧ ¬ is_prime (a + 2) ∧
  ¬ is_prime (a + 3) ∧ ¬ is_prime (a + 4) ∧ ¬ is_prime (a + 5)

-- Define the condition that there is a smallest prime number greater than a sequence of six consecutive nonprime numbers
theorem smallest_prime_after_six_consecutive_nonprimes :
  ∃ p, is_prime p ∧ p > 96 ∧ consecutive_nonprime_sequence 90 :=
sorry

end smallest_prime_after_six_consecutive_nonprimes_l9_9177


namespace find_sides_of_isosceles_triangle_l9_9547

noncomputable def isosceles_triangle_sides (b a : ℝ) : Prop :=
  ∃ (AI IL₁ : ℝ), AI = 5 ∧ IL₁ = 3 ∧
  b = 10 ∧ a = 12 ∧
  a = (6 / 5) * b ∧
  (b^2 = 8^2 + (3/5 * b)^2)

-- Proof problem statement
theorem find_sides_of_isosceles_triangle :
  ∀ (b a : ℝ), isosceles_triangle_sides b a → b = 10 ∧ a = 12 :=
by
  intros b a h
  sorry

end find_sides_of_isosceles_triangle_l9_9547


namespace tan_double_angle_l9_9001

theorem tan_double_angle (α : ℝ) (h1 : sin α = 3 / 5) (h2 : 0 < α ∧ α < π / 2) : tan (2 * α) = 24 / 7 := sorry

end tan_double_angle_l9_9001


namespace area_of_triangle_ABC_l9_9463

-- Let A, B, C be points such that triangle ABC is isosceles with AB = AC and BD is an altitude.
variables {A B C D E : Type} [point A] [point B] [point C] [point D] [point E]

-- Given conditions, define relevant properties and relationships
def is_isosceles_triangle (A B C : point) (AB : line A B) (AC : line A C) (BD : line B D) : Prop :=
  -- AB = AC and BD is the altitude 
  (distance A B = distance A C) ∧ (is_perpendicular BD AC)

def point_on_extension (E : point) (extension : line A C) (B E_l : dist_point B E 15) : Prop := 
  lies_on_extension E A C ∧ B E_l

def tan_angles_form_geometric_sequence (α β : ℝ) : Prop :=
  let γ := α - β in
  let δ := α + β in 
  (Real.tan γ) * (Real.tan δ) = (Real.tan α) ^ 2

def cot_angles_form_arith_sequence (α β : ℝ) : Prop :=
  let cot_β := Real.cot β in
  let cot_γ := Real.cot (β - α) in
  let cot_δ := Real.cot α in
  (cot_β, cot_γ, cot_δ) is_arith_seq 

def right_isosceles_triangle (a : ℝ) (BD BC : line BD BC) :=
  (BD = BC) / √2 ∧ area(A B C) = (1/2) * a^2

theorem area_of_triangle_ABC :
  ∃ (a : ℝ), is_isosceles_triangle A B C AB AC BD ∧ point_on_extension E AC BE ∧ 
              tan_angles_form_geometric_sequence α β ∧ cot_angles_form_arith_sequence α β ∧
              right_isosceles_triangle a BD BC 
              → (area(A B C) = 112.5) := sorry

end area_of_triangle_ABC_l9_9463


namespace maximum_marks_l9_9941

theorem maximum_marks (M : ℝ) :
  (0.45 * M = 80) → (M = 180) :=
by
  sorry

end maximum_marks_l9_9941


namespace complex_power_l9_9692

def cos_30 := Float.sqrt 3 / 2
def sin_30 := 1 / 2

theorem complex_power (i : ℂ) : 
  (3 * (⟨cos_30, sin_30⟩ : ℂ)) ^ 4 = -40.5 + 40.5 * i * Float.sqrt 3 := 
  sorry

end complex_power_l9_9692


namespace forty_fifth_digit_l9_9438

theorem forty_fifth_digit (digits_45 : String) :
  let sequence := List.range' 40 40 .map Nat.toString .join ;
  String.toList sequence.nth digits_45 contains '1' :=
by
  let numbers := List.range' 1 40 ++ List.range' 0 10
  have _: List.length numbers = 45
  let legend := (List.range' 1 10).sum
  rw legend
  sorry ⟩

end forty_fifth_digit_l9_9438


namespace shortest_distance_Dasha_to_Vasya_l9_9879

theorem shortest_distance_Dasha_to_Vasya:
  ∀ (d_DG d_VG d_GA d_GB d_AB : ℝ),
  d_DG = 15 ∧ d_VG = 17 ∧ d_GA = 12 ∧ d_GB = 10 ∧ d_AB = 8 →
  ∃ (d_DV : ℝ), d_DV = 18 :=
by 
  intros d_DG d_VG d_GA d_GB d_AB h,
  cases h with h_DG h,
  cases h with h_VG h,
  cases h with h_GA h,
  cases h with h_GB h_AB,
  use 18,
  sorry

end shortest_distance_Dasha_to_Vasya_l9_9879


namespace closest_integer_to_2_plus_sqrt_6_l9_9732

theorem closest_integer_to_2_plus_sqrt_6 (sqrt6_lower : 2 < Real.sqrt 6) (sqrt6_upper : Real.sqrt 6 < 2.5) : 
  abs (2 + Real.sqrt 6 - 4) < abs (2 + Real.sqrt 6 - 3) ∧ abs (2 + Real.sqrt 6 - 4) < abs (2 + Real.sqrt 6 - 5) :=
by
  sorry

end closest_integer_to_2_plus_sqrt_6_l9_9732


namespace smallest_prime_after_six_nonprimes_l9_9158

open Nat

/-- 
  Proposition:
    97 is the smallest prime number that occurs after a sequence 
    of six consecutive positive integers all of which are nonprime.
-/
theorem smallest_prime_after_six_nonprimes : ∃ (n : ℕ), (n > 0 ∧ Prime (n + 97)) ∧ 
  (∀ k : ℕ, k < n → (Nonprime k) ∧ Nonprime (k+1) ∧ Nonprime (k+2) ∧ Nonprime (k+3) ∧ Nonprime (k+4) ∧ Nonprime (k+5)) ∧ 
  (Prime (n + 1) → n + 1 = 97) := 
by 
  sorry

end smallest_prime_after_six_nonprimes_l9_9158


namespace volume_range_pyramid_SABCD_l9_9461

noncomputable def volume_pyramid_range (ABCD_base_side : ℝ) (h_lower h_upper : ℝ) : set ℝ :=
  { v : ℝ | ∃ h : ℝ, h_lower ≤ h ∧ h ≤ h_upper ∧ v = (1 / 3) * h * ABCD_base_side^2 }

theorem volume_range_pyramid_SABCD :
  ∀ (SC_lower SC_upper : ℝ), (2 * Real.sqrt 2 ≤ SC_lower) → (SC_upper ≤ 4) → 
  (volume_pyramid_range 2 (Real.sqrt 3) 2 = set.Icc (4 * Real.sqrt 3 / 3) (8 / 3)) :=
by
  intros SC_lower SC_upper h_lower_cond h_upper_cond
  sorry

end volume_range_pyramid_SABCD_l9_9461


namespace select_one_person_for_both_days_l9_9316

noncomputable def combination (n r : ℕ) := n.choose r

def volunteers := 5
def serve_both_days := combination volunteers 1
def remaining_for_saturday := volunteers - 1
def serve_saturday := combination remaining_for_saturday 1
def remaining_for_sunday := remaining_for_saturday - 1
def serve_sunday := combination remaining_for_sunday 1
def total_ways := serve_both_days * serve_saturday * serve_sunday

theorem select_one_person_for_both_days :
  total_ways = 60 := 
by
  -- We skip the proof details for now
  sorry

end select_one_person_for_both_days_l9_9316


namespace correct_proposition_only_l9_9650

theorem correct_proposition_only (a b x : ℝ) : 
  (¬ ∀ a : ℝ, ((a + 1 : ℂ).im = 0 → (a = -1))) ∧
  (¬ (a > b → ¬((a + complex.I^3).re > (b + complex.I^2).re))) ∧
  (¬ ((x^2 - 1) + (x^2 + 3*x + 2) * complex.I = complex.I * (x^2 + 3*x + 2) → x = 1 ∨ x = -1)) ∧
  (¬ ∃ z1 z2 : ℂ, z1 < z2 ∨ z2 < z1) :=
by
  sorry

end correct_proposition_only_l9_9650


namespace smallest_prime_after_six_nonprimes_l9_9161

open Nat

/-- 
  Proposition:
    97 is the smallest prime number that occurs after a sequence 
    of six consecutive positive integers all of which are nonprime.
-/
theorem smallest_prime_after_six_nonprimes : ∃ (n : ℕ), (n > 0 ∧ Prime (n + 97)) ∧ 
  (∀ k : ℕ, k < n → (Nonprime k) ∧ Nonprime (k+1) ∧ Nonprime (k+2) ∧ Nonprime (k+3) ∧ Nonprime (k+4) ∧ Nonprime (k+5)) ∧ 
  (Prime (n + 1) → n + 1 = 97) := 
by 
  sorry

end smallest_prime_after_six_nonprimes_l9_9161


namespace total_sandwiches_prepared_l9_9215

def num_people := 219.0
def sandwiches_per_person := 3.0

theorem total_sandwiches_prepared : num_people * sandwiches_per_person = 657.0 :=
by
  sorry

end total_sandwiches_prepared_l9_9215


namespace smallest_prime_after_six_nonprime_l9_9164

open Nat

noncomputable def is_nonprime (n : ℕ) : Prop :=
  ¬Prime n ∧ n ≠ 1

noncomputable def six_consecutive_nonprime (n : ℕ) : Prop :=
  is_nonprime n ∧ is_nonprime (n + 1) ∧ is_nonprime (n + 2) ∧ 
  is_nonprime (n + 3) ∧ is_nonprime (n + 4) ∧ is_nonprime (n + 5)

theorem smallest_prime_after_six_nonprime :
  ∃ n, six_consecutive_nonprime n ∧ (∀ m, m ≠ 97 → n + 6 < m → ¬Prime m) :=
by
  sorry

end smallest_prime_after_six_nonprime_l9_9164


namespace line_tangent_through_point_circle_eqn_l9_9874

theorem line_tangent_through_point_circle_eqn :
  let P := (3 : ℝ, 0 : ℝ)
  let circle_eq (x y : ℝ) := (x - 1)^2 + (y - 2 * Real.sqrt 3)^2 = 4
  ∃ line_eq : ℝ → ℝ → ℝ,
    (∀ x y, circle_eq x y → x - sqrt 3 * y + 3 = 0 → false) → 
    (line_eq = λ x y, x - sqrt 3 * y + 3) := 
sorry

end line_tangent_through_point_circle_eqn_l9_9874


namespace max_bottles_of_soda_l9_9221

-- Define the costs and exchange rate
def soda_cost (cost : ℝ) := cost = 2.5
def exchange_rate (rate : ℕ) := rate = 3

-- Define a function for maximum bottles drunk
def max_bottles (borrow : Bool) (yuan : ℝ) (cost : ℝ) (rate : ℕ) : ℕ :=
  if borrow then 18 else 17

-- The main theorem statement
theorem max_bottles_of_soda (borrow : Bool) (yuan : ℝ) 
  (H_cost : soda_cost 2.5) (H_rate : exchange_rate 3) :
  max_bottles borrow yuan 2.5 3 = if borrow then 18 else 17 := by
  sorry

end max_bottles_of_soda_l9_9221


namespace part_I_part_II_l9_9409

-- Part (I)
def parametric_eqs_line (t : ℝ) : ℝ × ℝ :=
  ( (√2 / 2) * t,
    (√2 / 2) * t + 4 * √2)

def polar_eq_circle (θ : ℝ) : ℝ :=
  4 * Real.cos (θ + π / 4)

def center_of_circle : Prop :=
  (√2, -√2) = (√2, -√2)  -- The coordinates given in the problem

-- Part (II)
def min_tangent_length_from_line_to_circle : Prop :=
  ∀ t : ℝ, Real.sqrt ((t + 4) ^ 2 + 32) ≥ 4 * √2

-- main statements
theorem part_I : center_of_circle := sorry
theorem part_II : min_tangent_length_from_line_to_circle := sorry

end part_I_part_II_l9_9409


namespace ants_meet_again_at_P_l9_9143

-- Definitions for given radii
def radius_large : ℝ := 7
def radius_small : ℝ := 3

-- Definitions for given speeds
def speed_large : ℝ := 5 * Real.pi
def speed_small : ℝ := 4 * Real.pi

-- Circumferences (These are intermediate results derived from the initial conditions)
def circumference_large : ℝ := 2 * radius_large * Real.pi
def circumference_small : ℝ := 2 * radius_small * Real.pi

-- Time for each ant to complete one lap
def time_large : ℚ := (circumference_large / speed_large : ℝ)
def time_small : ℚ := (circumference_small / speed_small : ℝ)

-- LCM Calculation (to find when they both meet at point P again)
-- LCM of two rational numbers is their LCM divided by their gcd, promoted to rationals
def lcm_rational (a b: ℚ) : ℚ := (a.num * b.num) / (Nat.gcd a.den b.den * (Nat.gcd a.num b.num))

-- Expected result (42 Minutes in rational form for ease of comparison)
def meet_again_time : ℚ := 42

theorem ants_meet_again_at_P : lcm_rational time_large time_small = meet_again_time := 
by sorry

end ants_meet_again_at_P_l9_9143


namespace hyperbola_eccentricity_l9_9489

-- Definitions of the conditions translated into Lean
def hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1^2) / (a^2) - (p.2^2) / (b^2) = 1}

def foci (a b c e : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((-c, 0), (c, 0))
  
-- The main theorem representing the proof problem
theorem hyperbola_eccentricity (a b c e : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) :
  let p := foci a b c e in
  let F1 := p.1 in
  let F2 := p.2 in
  ∃ A : ℝ × ℝ, A ∈ hyperbola a b ∧ ∠ F1 A F2 = 90 ∧ dist A F1 = 3 * dist A F2 →
  e = Real.sqrt 10 / 2 :=
sorry

end hyperbola_eccentricity_l9_9489


namespace decreases_as_x_increases_l9_9648

def y1 (x : ℝ) : ℝ := 6 * x
def y2 (x : ℝ) : ℝ := -6 * x
def y3 (x : ℝ) : ℝ := 6 / x
def y4 (x : ℝ) : ℝ := -6 / x

theorem decreases_as_x_increases (x1 x2 : ℝ) (h : x1 < x2) : y2 x1 > y2 x2 :=
by sorry

end decreases_as_x_increases_l9_9648


namespace find_n_l9_9050

def f (n : ℕ) : ℕ :=
  min (min (min (min (1 + n / 1) (4 + n / 4)) (9 + n / 9)) (16 + n / 16)) (25 + n / 25)

theorem find_n (n : ℕ) : 990208 ≤ n ∧ n < 991232 → f(n) = 1991 :=
begin
  sorry
end

end find_n_l9_9050


namespace problem_nine_chapters_l9_9454

theorem problem_nine_chapters (x y : ℝ) :
  (x + (1 / 2) * y = 50) →
  (y + (2 / 3) * x = 50) →
  (x + (1 / 2) * y = 50) ∧ (y + (2 / 3) * x = 50) :=
by
  intros h1 h2
  exact ⟨h1, h2⟩

end problem_nine_chapters_l9_9454


namespace two_digit_numbers_count_l9_9816

theorem two_digit_numbers_count : 
  (card {n : ℕ | ∃ a b : ℕ, n = 10 * a + b ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ a = b + 2}) = 8 :=
by 
  sorry

end two_digit_numbers_count_l9_9816


namespace rectangle_diagonal_l9_9117

theorem rectangle_diagonal (k : ℕ) (h1 : 2 * (5 * k + 4 * k) = 72) : 
  (Real.sqrt ((5 * k) ^ 2 + (4 * k) ^ 2)) = Real.sqrt 656 :=
by
  sorry

end rectangle_diagonal_l9_9117


namespace diagonal_inequality_of_obtuse_angles_l9_9936

theorem diagonal_inequality_of_obtuse_angles
  {A B C D : Type*}
  [metric_space A]
  [metric_space B]
  [metric_space C]
  [metric_space D]
  (angleB angleD : ℝ)
  (hB : angleB > 90)
  (hD : angleD > 90)
  (dist_AC dist_BD : ℝ) :
  dist_BD < dist_AC :=
sorry

end diagonal_inequality_of_obtuse_angles_l9_9936


namespace hyperbola_dot_product_l9_9797

theorem hyperbola_dot_product (x y : ℝ) (P F1 F2 : ℝ × ℝ)
  (h1 : x^2 - (y^2) / 3 = 1)
  (F1_eq : F1 = (-2, 0))
  (F2_eq : F2 = (2, 0))
  (e : ℝ)
  (e_eq : e = 2)
  (sin_ratio_cond : sin (∠ P F2 F1) / sin (∠ P F1 F2) = e)
  (dist_cond1 : dist P F1 = 4)
  (dist_cond2 : dist P F2 = 2) :
  let FP := (P.1 - F2.1, P.2 - F2.2)
      F1F2 := (F2.1 - F1.1, F2.2 - F1.2) in
  FP.1 * F1F2.1 + FP.2 * F1F2.2 = 2 :=
sorry

end hyperbola_dot_product_l9_9797


namespace cost_to_fill_bathtub_with_jello_l9_9472

-- Define the conditions
def pounds_per_gallon : ℝ := 8
def gallons_per_cubic_foot : ℝ := 7.5
def cubic_feet_of_water : ℝ := 6
def tablespoons_per_pound : ℝ := 1.5
def cost_per_tablespoon : ℝ := 0.5

-- The theorem stating the cost to fill the bathtub with jello
theorem cost_to_fill_bathtub_with_jello : 
  let total_gallons := cubic_feet_of_water * gallons_per_cubic_foot in
  let total_pounds := total_gallons * pounds_per_gallon in
  let total_tablespoons := total_pounds * tablespoons_per_pound in
  let total_cost := total_tablespoons * cost_per_tablespoon in
  total_cost = 270 := 
by {
  -- Here's where we would provide the proof steps, but just add sorry to skip it
  sorry
}

end cost_to_fill_bathtub_with_jello_l9_9472


namespace find_k_value_l9_9788

theorem find_k_value (x₁ x₂ x₃ x₄ : ℝ)
  (h1 : (x₁ + x₂ + x₃ + x₄) = 18)
  (h2 : (x₁ * x₂ + x₁ * x₃ + x₁ * x₄ + x₂ * x₃ + x₂ * x₄ + x₃ * x₄) = k)
  (h3 : (x₁ * x₂ * x₃ + x₁ * x₂ * x₄ + x₁ * x₃ * x₄ + x₂ * x₃ * x₄) = -200)
  (h4 : (x₁ * x₂ * x₃ * x₄) = -1984)
  (h5 : x₁ * x₂ = -32) :
  k = 86 :=
by sorry

end find_k_value_l9_9788


namespace tan_ratio_of_angles_l9_9905

theorem tan_ratio_of_angles (a b : ℝ) (h1 : Real.sin (a + b) = 3/4) (h2 : Real.sin (a - b) = 1/2) :
    (Real.tan a / Real.tan b) = 5 := 
by 
  sorry

end tan_ratio_of_angles_l9_9905


namespace compute_expression_l9_9200

-- Define the necessary variables and their values as constants
def numerator : ℝ := 3.6 * 0.48 * 2.50
def denominator : ℝ := 0.12 * 0.09 * 0.5
def fraction : ℝ := numerator / denominator
def final_result : ℝ := 3 * fraction

-- Theorem stating that 3 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2400
theorem compute_expression : final_result = 2400 := by 
  sorry

end compute_expression_l9_9200


namespace proof_m_plus_n_l9_9795

-- Define the conditions and constants
variables (a m n : ℝ)

-- Conditions given in the problem.
axiom ha1 : a > 0
axiom ha2 : a ≠ 1

-- The function always passes through the point (m, n).
axiom fixed_point : Function.eval (λ x, 2 * a^(x - 1) + 1) m = n

-- Define the target statement we want to prove.
theorem proof_m_plus_n : m + n = 4 := 
by sorry

end proof_m_plus_n_l9_9795


namespace period_of_sin_x_over_3_is_6pi_l9_9714

def period_sin_x_over_3 : ℝ := 6 * Real.pi

theorem period_of_sin_x_over_3_is_6pi : ∃ T : ℝ, ∀ x : ℝ, sin (x + period_sin_x_over_3) / 3 = sin (x / 3) := sorry

end period_of_sin_x_over_3_is_6pi_l9_9714


namespace length_of_the_bridge_l9_9197

theorem length_of_the_bridge
  (train_length : ℕ)
  (train_speed_kmh : ℕ)
  (cross_time_s : ℕ)
  (h_train_length : train_length = 120)
  (h_train_speed_kmh : train_speed_kmh = 45)
  (h_cross_time_s : cross_time_s = 30) :
  ∃ bridge_length : ℕ, bridge_length = 255 := 
by 
  sorry

end length_of_the_bridge_l9_9197


namespace sum_of_a_c_l9_9961

theorem sum_of_a_c (a b c d : ℝ) (h1 : -2 * abs (1 - a) + b = 7) (h2 : 2 * abs (1 - c) + d = 7)
    (h3 : -2 * abs (11 - a) + b = -1) (h4 : 2 * abs (11 - c) + d = -1) : a + c = 12 := by
  -- Definitions for conditions
  -- h1: intersection at (1, 7) for first graph
  -- h2: intersection at (1, 7) for second graph
  -- h3: intersection at (11, -1) for first graph
  -- h4: intersection at (11, -1) for second graph
  sorry

end sum_of_a_c_l9_9961


namespace segments_do_not_intersect_l9_9496

-- Define points in the plane
variables {A : ℕ → ℝ × ℝ}

-- Define distances and angles between points
def dist (P Q : ℝ × ℝ) : ℝ := (P.fst - Q.fst)^2 + (P.snd - Q.snd)^2

-- Define angle between triplets of points
def angle (A B C : ℝ × ℝ) : ℝ :=
  let u := (B.fst - A.fst, B.snd - A.snd)
  let v := (C.fst - B.fst, C.snd - B.snd)
  acos ((u.fst * v.fst + u.snd * v.snd) / (dist (0,0) u) / (dist (0,0) v))

-- The conditions
def condition1 (n : ℕ) : Prop :=
  ∀ i < n, dist (A i.succ) (A (i + 1)) ≤ 1/2^i * dist (A i.succ) (A (i + 1).succ)

def condition2 (n : ℕ) : Prop :=
  ∀ i < n - 1, 0 < angle (A i) (A (i.succ)) (A (i + 1).succ) ∧
                angle (A i) (A (i.succ)) (A (i + 1).succ) < angle (A (i.succ)) (A (i + 1)) (A (i + 2)) ∧
                angle (A (n - 2)) (A (n - 1)) (A n) < π

-- The goal to prove
theorem segments_do_not_intersect (n : ℕ) 
  (h1 : condition1 n) (h2 : condition2 n) 
  (k m : ℕ) (hk : 0 ≤ k) (hm : k ≤ m - 2) (hn : m - 2 < n - 2) : 
  ¬(∃ x, x ∈ A k.segment ∧ x ∈ A m.segment) := sorry

end segments_do_not_intersect_l9_9496


namespace maximum_shapes_in_grid_l9_9580

-- Define the grid size and shape properties
def grid_width : Nat := 8
def grid_height : Nat := 14
def shape_area : Nat := 3
def shape_grid_points : Nat := 8

-- Define the total grid points in the rectangular grid
def total_grid_points : Nat := (grid_width + 1) * (grid_height + 1)

-- Define the question and the condition that needs to be proved
theorem maximum_shapes_in_grid : (total_grid_points / shape_grid_points) = 16 := by
  sorry

end maximum_shapes_in_grid_l9_9580


namespace inequality_for_positive_numbers_l9_9076

theorem inequality_for_positive_numbers (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : 
  (a + b) * (a^4 + b^4) ≥ (a^2 + b^2) * (a^3 + b^3) :=
sorry

end inequality_for_positive_numbers_l9_9076


namespace larger_number_l9_9145

theorem larger_number (x y : ℝ) (h₁ : x + y = 45) (h₂ : x - y = 7) : x = 26 :=
by
  sorry

end larger_number_l9_9145


namespace tangent_slope_ln_2_l9_9793

def f (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

def f_prime (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem tangent_slope_ln_2 : 
  ∃ (x : ℝ), f_prime x = 3 / 2 ∧ x = Real.log 2 :=
by
  sorry

end tangent_slope_ln_2_l9_9793


namespace pure_imaginary_complex_number_l9_9402

theorem pure_imaginary_complex_number (m : ℝ) (z : ℂ) :
  (z = m^2 * (1 + complex.i) - m * (3 + 6 * complex.i)) →
  (∃ b : ℝ, z = complex.i * b) →
  m = 3 :=
by
  intro h1 h2
  sorry

end pure_imaginary_complex_number_l9_9402


namespace max_is_twice_emily_probability_l9_9297

noncomputable def probability_event_max_gt_twice_emily : ℝ :=
  let total_area := 1000 * 3000
  let triangle_area := 1/2 * 1000 * 1000
  let rectangle_area := 1000 * (3000 - 2000)
  let favorable_area := triangle_area + rectangle_area
  favorable_area / total_area

theorem max_is_twice_emily_probability :
  probability_event_max_gt_twice_emily = 1 / 2 :=
by
  sorry

end max_is_twice_emily_probability_l9_9297


namespace smallest_k_binom_congruence_l9_9736

theorem smallest_k_binom_congruence :
  ∃ k : ℕ, (∀ x b : ℕ, 0 < b → 0 < x → 
    binom (x + k * b) 12 % b = binom x 12 % b) ∧ 
    k = 27720 :=
sorry

end smallest_k_binom_congruence_l9_9736


namespace inradius_of_equal_area_and_perimeter_l9_9017

theorem inradius_of_equal_area_and_perimeter
  (a b c : ℝ)
  (A : ℝ)
  (h1 : A = a + b + c)
  (s : ℝ := (a + b + c) / 2)
  (h2 : A = s * (2 * A / (a + b + c))) :
  ∃ r : ℝ, r = 2 := by
  sorry

end inradius_of_equal_area_and_perimeter_l9_9017


namespace tiffany_bags_on_monday_l9_9137

theorem tiffany_bags_on_monday : 
  ∃ M : ℕ, M = 8 ∧ ∃ T : ℕ, T = 7 ∧ M = T + 1 :=
by
  sorry

end tiffany_bags_on_monday_l9_9137


namespace jello_cost_l9_9467

def cost_to_fill_tub_with_jello (water_volume_cubic_feet : ℕ) (gallons_per_cubic_foot : ℕ) 
    (pounds_per_gallon : ℕ) (tablespoons_per_pound : ℕ) (cost_per_tablespoon : ℕ) : ℕ :=
  water_volume_cubic_feet * gallons_per_cubic_foot * pounds_per_gallon * tablespoons_per_pound * cost_per_tablespoon

theorem jello_cost (water_volume_cubic_feet : ℕ) (gallons_per_cubic_foot : ℕ) 
    (pounds_per_gallon : ℕ) (tablespoons_per_pound : ℕ) (cost_per_tablespoon : ℕ) : 
    water_volume_cubic_feet = 6 ∧ gallons_per_cubic_foot = 7 ∧ pounds_per_gallon = 8 ∧ 
    tablespoons_per_pound = 1 ∧ cost_per_tablespoon = 1 →
    cost_to_fill_tub_with_jello water_volume_cubic_feet gallons_per_cubic_foot pounds_per_gallon tablespoons_per_pound cost_per_tablespoon = 270 :=
  by 
    sorry

end jello_cost_l9_9467


namespace spellbook_cost_in_gold_l9_9415

-- Define the constants
def num_spellbooks : ℕ := 5
def cost_potion_kit_in_silver : ℕ := 20
def num_potion_kits : ℕ := 3
def cost_owl_in_gold : ℕ := 28
def conversion_rate : ℕ := 9
def total_payment_in_silver : ℕ := 537

-- Define the problem to prove the cost of each spellbook in gold given the conditions
theorem spellbook_cost_in_gold : (total_payment_in_silver 
  - (cost_potion_kit_in_silver * num_potion_kits + cost_owl_in_gold * conversion_rate)) / num_spellbooks / conversion_rate = 5 := 
  by
  sorry

end spellbook_cost_in_gold_l9_9415


namespace z_in_third_quadrant_l9_9956

open Complex

noncomputable def z : ℂ := Complex.I * (-1 + Complex.I)

def coordinates_of_z : ℂ → (ℝ × ℝ)
| z := (z.re, z.im)

def is_third_quadrant (coords : ℝ × ℝ) : Prop :=
coords.1 < 0 ∧ coords.2 < 0

theorem z_in_third_quadrant : is_third_quadrant (coordinates_of_z z) :=
sorry

end z_in_third_quadrant_l9_9956


namespace smallest_sum_of_inverses_l9_9369

theorem smallest_sum_of_inverses 
  (x y : ℕ) (hx : x ≠ y) (h1 : 0 < x) (h2 : 0 < y) (h_condition : (1 / x : ℚ) + 1 / y = 1 / 15) :
  x + y = 64 := 
sorry

end smallest_sum_of_inverses_l9_9369


namespace zero_point_interval_l9_9965

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x - 2

theorem zero_point_interval :
  ∃ c ∈ Ioo 0 1, f c = 0 := by
  have h0 : f 0 < 0 := by
    simp [f, Real.exp_zero]
    norm_num
  have h1 : f 1 > 0 := by
    simp [f, Real.exp_one]
    linarith [Real.exp_pos 1]
  have h_continuous : Continuous f := by
    exact continuous_exp.add continuous_id.sub continuous_const
  
  obtain ⟨c, hc0, hc1⟩ := IntermediateValueTheoremIoo 0 1 0 f h_continuous h0 h1
  use [c, hc0, hc1]
  exact hc1

end zero_point_interval_l9_9965


namespace order_well_defined_l9_9195

noncomputable theory

variables {n : ℕ} (x : Units (Zmod n))

theorem order_well_defined (hx : x ≠ 0) :
  ∃ ω : ℕ, ω > 0 ∧ x ^ ω = 1 :=
by
  sorry

end order_well_defined_l9_9195


namespace number_of_ways_to_select_60_l9_9313

-- Define the conditions: five volunteers, and choose 2 people each day such that one serves both days
def select_ways (n : ℕ) (r : ℕ) : ℕ :=
  Nat.choose n r

theorem number_of_ways_to_select_60 : select_ways 5 1 * (select_ways 4 1 * select_ways 3 1) = 60 := by
  sorry

end number_of_ways_to_select_60_l9_9313


namespace S4_equals_15_l9_9861

noncomputable def S_n (q : ℝ) (n : ℕ) := (1 - q^n) / (1 - q)

theorem S4_equals_15 (q : ℝ) (n : ℕ) (h1 : S_n q 1 = 1) (h2 : S_n q 5 = 5 * S_n q 3 - 4) : 
  S_n q 4 = 15 :=
by
  sorry

end S4_equals_15_l9_9861


namespace complex_power_result_l9_9682

theorem complex_power_result :
  (3 * Complex.cos (Real.pi / 6) + 3 * Complex.sin (Real.pi / 6) * Complex.i)^4 = 
  -40.5 + 40.5 * Complex.i * Real.sqrt 3 :=
by sorry

end complex_power_result_l9_9682


namespace number_of_players_in_association_l9_9875

-- Define the variables and conditions based on the given problem
def socks_cost : ℕ := 6
def tshirt_cost := socks_cost + 8
def hat_cost := tshirt_cost - 3
def total_expenditure : ℕ := 4950
def cost_per_player := 2 * (socks_cost + tshirt_cost + hat_cost)

-- The statement to prove
theorem number_of_players_in_association :
  total_expenditure / cost_per_player = 80 := by
  sorry

end number_of_players_in_association_l9_9875


namespace solve_inequality_l9_9091

theorem solve_inequality (x : ℝ) : 
  (2 - (1 / (2 * x + 3)) < 4) →
  x ∈ set.Ioo ( -∞) -7 / 4 ∪ set.Ioo -3 / 2 (∞) :=
sorry

end solve_inequality_l9_9091


namespace rank_le_two_l9_9487

variables {n : ℕ} [fact (n > 0)]
variable (A : Matrix (Fin n) (Fin n) ℂ)
variable (h : ∀ i j k : Fin n, A i j + A j k + A k i = 0)

theorem rank_le_two : Matrix.rank A ≤ 2 := sorry

end rank_le_two_l9_9487


namespace minimum_value_l9_9772

noncomputable def ellipse_hyperbola_min_value (a1 b1 a2 b2 c : ℝ) (e1 e2 : ℝ) : ℝ := 
have h1 : a1 > b1 > 0 := sorry,
have h2 : a2 > 0 := sorry,
have h3 : b2 > 0 := sorry,
have h_c : a1^2 + a2^2 = 2 * c^2 := sorry,
have e1_def : e1 = c / a1 := sorry,
have e2_def : e2 = c / a2 := sorry,
9 * e1^2 + e2^2

theorem minimum_value (a1 b1 a2 b2 c e1 e2 : ℝ) 
  (h1 : a1 > b1 > 0) 
  (h2 : a2 > 0) 
  (h3 : b2 > 0) 
  (h_c : a1^2 + a2^2 = 2 * c^2) 
  (e1_def : e1 = c / a1) 
  (e2_def : e2 = c / a2) : 
  9 * e1^2 + e2^2 = 8 := 
sorry

end minimum_value_l9_9772


namespace correct_interpretation_l9_9561

-- Definitions for the options
def option_A : Prop := "80% of the areas in Xiamen city will have rain tomorrow"
def option_B : Prop := "It will rain for 80% of the time in Xiamen city tomorrow"
def option_C : Prop := "If you go out without rain gear tomorrow, you will definitely get rained on"
def option_D : Prop := "If you go out without rain gear tomorrow, there is a high possibility of getting rained on"

-- The main statement about the probability of rain
def probability_statement : Prop := "the probability of rain in Xiamen city tomorrow is 80%"

-- The theorem we need to prove
theorem correct_interpretation (h: probability_statement): option_D :=
by
  sorry

end correct_interpretation_l9_9561


namespace min_distance_from_circle_to_line_l9_9752

noncomputable def circle_center : (ℝ × ℝ) := (3, -1)
noncomputable def circle_radius : ℝ := 2

def on_circle (P : ℝ × ℝ) : Prop := (P.1 - circle_center.1) ^ 2 + (P.2 + circle_center.2) ^ 2 = circle_radius ^ 2
def on_line (Q : ℝ × ℝ) : Prop := Q.1 = -3

theorem min_distance_from_circle_to_line (P Q : ℝ × ℝ)
  (h1 : on_circle P) (h2 : on_line Q) : dist P Q = 4 := 
sorry

end min_distance_from_circle_to_line_l9_9752


namespace smallest_x_plus_y_l9_9352

theorem smallest_x_plus_y (x y : ℕ) (h_xy_not_equal : x ≠ y) (h_condition : (1 : ℚ) / x + 1 / y = 1 / 15) :
  x + y = 64 :=
sorry

end smallest_x_plus_y_l9_9352


namespace super_square_number_count_correct_l9_9434

-- Define what it means to be a super square number
def is_super_square_number (n : ℕ) : Prop :=
  let digits_sum := (n / 100) + ((n / 10) % 10) + (n % 10)
  in Mathlib.isPerfectSquare n ∧ Mathlib.isPerfectSquare digits_sum

-- Count the number of three-digit super square numbers
def count_super_square_numbers : ℕ :=
  Nat.count (λ n, 100 ≤ n ∧ n ≤ 999 ∧ is_super_square_number n) 1000

theorem super_square_number_count_correct :
  count_super_square_numbers = 13 :=
sorry

end super_square_number_count_correct_l9_9434


namespace range_of_f_pos_l9_9054

-- Given conditions:
-- 1. f is an odd function (x ∈ ℝ)
-- 2. f(-1) = 0
-- 3. For x > 0, xf''(x) - f(x) > 0

noncomputable def f : ℝ → ℝ := sorry
axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom f_neg_one : f (-1) = 0
axiom condition : ∀ x : ℝ, 0 < x → x * (deriv ∘ deriv) f x - f x > 0

theorem range_of_f_pos :
  { x : ℝ | f x > 0 } = { x : ℝ | -1 < x ∧ x < 0 } ∪ { x : ℝ | 1 < x ∧ x < ∞ } :=
sorry

end range_of_f_pos_l9_9054


namespace find_original_price_l9_9939

-- Defining constants and variables
def original_price (P : ℝ) : Prop :=
  let cost_after_repairs := P + 13000
  let selling_price := 66900
  let profit := selling_price - cost_after_repairs
  let profit_percent := profit / P * 100
  profit_percent = 21.636363636363637

theorem find_original_price : ∃ P : ℝ, original_price P :=
  by
  sorry

end find_original_price_l9_9939


namespace sum_of_ages_is_32_l9_9152

-- Define the values and conditions given in the problem
def viggo_age_when_brother_was_2 (brother_age : ℕ) : ℕ := 10 + 2 * brother_age
def age_difference (viggo_age_brother_2 : ℕ) (brother_age : ℕ) : ℕ := viggo_age_brother_2 - brother_age
def current_viggo_age (current_brother_age : ℕ) (difference : ℕ) := current_brother_age + difference

-- State the main theorem
theorem sum_of_ages_is_32 : 
  let brother_age_when_2 := 2 in
  let current_brother_age := 10 in
  let viggo_age_when_2 := viggo_age_when_brother_was_2 brother_age_when_2 in
  let difference := age_difference viggo_age_when_2 brother_age_when_2 in
  current_viggo_age current_brother_age difference + current_brother_age = 32 := 
by
  sorry

end sum_of_ages_is_32_l9_9152


namespace cubic_roots_sum_of_cubes_l9_9389

theorem cubic_roots_sum_of_cubes (a b c : ℝ)
  (h1 : (Polynomial.X^3 - 3 * Polynomial.X^2 + 4 * Polynomial.X - 5).isRoot a)
  (h2 : (Polynomial.X^3 - 3 * Polynomial.X^2 + 4 * Polynomial.X - 5).isRoot b)
  (h3 : (Polynomial.X^3 - 3 * Polynomial.X^2 + 4 * Polynomial.X - 5).isRoot c) :
  a^3 + b^3 + c^3 = 6 :=
by 
  sorry

end cubic_roots_sum_of_cubes_l9_9389


namespace complex_number_quadrant_l9_9120

theorem complex_number_quadrant :
  let z := (3 - Complex.i) / (1 - Complex.i) in
  (z.re > 0 ∧ z.im > 0) := by
sorry

end complex_number_quadrant_l9_9120


namespace smallest_prime_after_six_nonprimes_l9_9186

open Nat

theorem smallest_prime_after_six_nonprimes : 
  ∃ p : ℕ, prime p ∧ p > 0 ∧
  (∃ n : ℕ, ¬prime n ∧ ¬prime (n+1) ∧ ¬prime (n+2) ∧ ¬prime (n+3) ∧ ¬prime (n+4) ∧ ¬prime (n+5) ∧ prime (n+6) ∧ n+6 = p) ∧ 
  p = 89 :=
sorry

end smallest_prime_after_six_nonprimes_l9_9186


namespace sum_of_interior_angles_l9_9004

theorem sum_of_interior_angles (h : ∀ θ, θ = 40 → θ ∈ exterior_angles) 
: ∑ θ in exterior_angles, interior_angle θ = 1260 :=
sorry

end sum_of_interior_angles_l9_9004


namespace perimeter_13_circles_l9_9578

open Real

-- Define the parameters
def r := 2 * sqrt (2 - sqrt 3)
def centers (n : Nat) := (fin n) → ℝ
def distance_centers := 2

-- Formalize the statement given the conditions and correct answer
theorem perimeter_13_circles :
  let r := 2 * sqrt (2 - sqrt 3),
      total_perimeter := 44 * π * sqrt (2 - sqrt 3)
  in (∃ n, n = 13 ∧ (∀ i : fin n, ∃ center : centers n, ∀ j : fin n, (i ≠ j) → dist (center i) (center j) = distance_centers) → 
  (figure_perimeter r (distance_centers) n = total_perimeter)) :=
sorry

end perimeter_13_circles_l9_9578


namespace locus_of_points_l9_9348

variables {α β γ k : ℝ}
variables {x y z : ℝ}
variables {M M1 M2 : Type*}

def distance_to_sides (M : Type*) : M → ℝ → ℝ → ℝ := sorry

theorem locus_of_points (hM : ∀ M, (α * distance_to_sides M x + β * distance_to_sides M y + γ * distance_to_sides M z = k)) : 
  (∀ M, ∃ M1 M2, (distance_to_sides M1 x = distance_to_sides M x) ∧ (distance_to_sides M2 y = distance_to_sides M y) ∧ (distance_to_sides M2 z = distance_to_sides M z)) ∨ 
  (∀ M1 M2, M ∈ segment ℝ M1 M2) ∨ 
  (∀ M, ∃ M1 M2, (distance_to_sides M x ≠ distance_to_sides M1 x) ∧ (distance_to_sides M y ≠ distance_to_sides M2 y) ∧ (distance_to_sides M z ≠ distance_to_sides M2 z)) :=
sorry


end locus_of_points_l9_9348


namespace exists_u_function_l9_9038

noncomputable def problem_statement (f : (0 : ℝ) .. ∞ → ℝ) 
  (Hf : ∀ x : ℝ, 0 < x → f x = f (1 / x)) : 
  Prop :=
  ∃ u : ℝ → ℝ, (∀ x : ℝ, 0 < x → u ((x + 1 / x) / 2) = f x) ∧ ∀ y : ℝ, 1 ≤ y → ∃ z : ℝ, 0 ≤ z ∧ y = (Real.exp z + Real.exp (-z)) / 2

-- Main theorem statement
theorem exists_u_function {f : (0 : ℝ) .. ∞ → ℝ}
  (Hf : ∀ x : ℝ, 0 < x → f x = f (1 / x)) :
  ∃ u : ℝ → ℝ, (∀ x : ℝ, 0 < x → u ((x + 1 / x) / 2) = f x) :=
begin
  exact problem_statement f Hf,
end

end exists_u_function_l9_9038


namespace sequence_formula_and_sum_l9_9406

noncomputable def f (x : ℝ) : ℝ :=
((Real.sin x + Real.cos x)^2 - 1) / (Real.cos x^2 - Real.sin x^2)

theorem sequence_formula_and_sum (n : ℕ) (hn : 0 < n) :
  (∀ (x : ℝ), (0 < x) → (f x = Real.sqrt 3) → ∃ k : ℤ, x = (k * Real.pi / 2) + (Real.pi / 6)) ∧
  (∃ a_n : ℕ → ℝ, a_n n = (3 * n - 2) * Real.pi / 6) ∧
  (∃ b_n : ℕ → ℝ, b_n n = 3 * ((3 * n - 2) * Real.pi / 6) / ((4 * n^2 - 1) * (3 * n - 2))) ∧
  (∃ S_n : ℕ → ℝ, S_n n = ∑ i in finset.range n, b_n i = (2 * n * Real.pi) / (2 * n + 1)) :=
by
  sorry

end sequence_formula_and_sum_l9_9406


namespace max_non_overlapping_crosses_in_circle_l9_9276

def cross := { P : Type }

-- The main theorem statement asserting (question == answer)
theorem max_non_overlapping_crosses_in_circle (radius : ℝ) (side_length : ℝ) (cross_area : ℝ) (larger_circle_area : ℝ) :
  radius = 100 → side_length = 1 →
  cross_area = (Real.pi * (1/(2*Real.sqrt 2))^2) →
  larger_circle_area = (Real.pi * 100^2) →
  ∃ max_crosses : ℕ, max_crosses = 40000 := by
  intros
  sorry

end max_non_overlapping_crosses_in_circle_l9_9276


namespace rich_total_distance_l9_9084

-- Define the given conditions 
def distance_house_to_sidewalk := 20
def distance_down_road := 200
def total_distance_so_far := distance_house_to_sidewalk + distance_down_road
def distance_left_turn := 2 * total_distance_so_far
def distance_to_intersection := total_distance_so_far + distance_left_turn
def distance_half := distance_to_intersection / 2
def total_distance_one_way := distance_to_intersection + distance_half

-- Define the theorem to be proven 
theorem rich_total_distance : total_distance_one_way * 2 = 1980 :=
by 
  -- This line is to complete the 'prove' demand of the theorem
  sorry

end rich_total_distance_l9_9084


namespace greatest_prime_factor_g_100_l9_9596

def g (m : ℕ) : ℕ := (List.range' 2 (m - 1 + 2)).prod

theorem greatest_prime_factor_g_100 :
  Nat.greatest_prime_factor (g 100) = 97 :=
sorry

end greatest_prime_factor_g_100_l9_9596


namespace simplify_expression_l9_9532

theorem simplify_expression (x : ℝ) : (3 * x + 20) + (50 * x + 25) = 53 * x + 45 :=
by
  sorry

end simplify_expression_l9_9532


namespace school_A_wins_championship_probability_school_B_expectation_X_l9_9089

/-- 
Definition for school A winning the championship.
We define events as independent and use given probabilities.
-/
theorem school_A_wins_championship_probability :
  let p1 := 0.5 in
  let p2 := 0.4 in
  let p3 := 0.8 in
  -- Scenario calculations:
  let P1 := p1 * p2 * p3 in
  let P2 := (p1 * p2 * (1 - p3) + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3) in
  -- Total probability of winning at least 2 events
  P1 + P2 = 0.6 :=
by sorry

/-- 
Definition for expectation of total score X for school B.
Given probability distribution over possible scores.
-/
theorem school_B_expectation_X : 
  let p0 := 0.16 in
  let p10 := 0.44 in
  let p20 := 0.34 in
  let p30 := 0.06 in
  -- Expectation calculation:
  0 * p0 + 10 * p10 + 20 * p20 + 30 * p30 = 13 :=
by sorry

end school_A_wins_championship_probability_school_B_expectation_X_l9_9089


namespace evie_l9_9869

variable (Evie_current_age : ℕ) 

theorem evie's_age_in_one_year
  (h : Evie_current_age + 4 = 3 * (Evie_current_age - 2)) : 
  Evie_current_age + 1 = 6 :=
by
  sorry

end evie_l9_9869


namespace Beth_bought_10_cans_of_corn_l9_9655

theorem Beth_bought_10_cans_of_corn (a b : ℕ) (h1 : b = 15 + 2 * a) (h2 : b = 35) : a = 10 := by
  sorry

end Beth_bought_10_cans_of_corn_l9_9655


namespace part1_part2_part3_l9_9913

variables {R : Type*} [linear_ordered_field R] [topological_space R] [ordered_topological_field R]

variable (f : R → R)
variables {x1 x2 : R}
variable (a : R)
variable {n : ℕ}

-- Condition: f is an even function
def is_even (f : R → R) := ∀ x, f x = f (-x)

-- Condition: f is symmetric about x = 1
def is_symmetric_about_one (f : R → R) := ∀ x, f x = f (2 - x)

-- Condition: Functional Equation for f on [0, 1/2]
def functional_eqn (f : R → R) := ∀ x1 x2, (0 ≤ x1 ∧ x1 ≤ 1/2) ∧ (0 ≤ x2 ∧ x2 ≤ 1/2) → f (x1 + x2) = f x1 * f x2

-- Condition: f(1) = a where a > 0
axiom f_one_eq_a : f 1 = a
axiom a_gt_zero : 0 < a

-- Question (1): Prove f(1/2) = a^1/2 and f(1/4) = a^1/4
theorem part1 : f (1/2) = a ^ (1/2) ∧ f (1/4) = a ^ (1/4) :=
sorry

-- Question (2): Prove f(x) is a periodic function with period 2
theorem part2 : ∀ x, f (x) = f (x + 2) :=
sorry

-- Question (3): Prove lim_{n → ∞} (ln a_{n}) = 0 with a_{n} = f(2n + 1/(2n))
noncomputable def a_n := f (2 * n + 1 / (2 * n))
theorem part3 : tendsto (λ n : ℕ, real.log (a_n f)) at_top (nhds 0) :=
sorry

end part1_part2_part3_l9_9913


namespace general_formula_proof_l9_9756

noncomputable def a_seq (a : ℕ → ℝ) (a1 : a 1 = 2) (rel : ∀ n, 1 / a n - 1 / a (n + 1) = 2 / (4 * (∑ i in finset.range n.succ, a i.succ) - 1)) : Prop :=
  a 2 = 14 / 3 ∧ 
  (∀ n, (a n / (a (n + 1) - a n)) = (4 * n - 1) / 4) ∧ 
  (∀ n, a n = (8 * n - 2) / 3) 

theorem general_formula_proof : ∃ a : ℕ → ℝ, a_seq a (by sorry) (by sorry) :=
  sorry

end general_formula_proof_l9_9756


namespace probability_of_y_lt_sin_x_l9_9947

noncomputable def probability_y_lt_sin_x (x y : ℝ) : ℝ :=
  if h : (0 ≤ x ∧ x ≤ π / 2) ∧ (0 ≤ y ∧ y ≤ π / 2) then
    (integral (λ x, sin x) 0 (π / 2)) / ((π / 2) * (π / 2))
  else
    0

theorem probability_of_y_lt_sin_x : probability_y_lt_sin_x x y = 4 / π^2 := 
sorry

end probability_of_y_lt_sin_x_l9_9947


namespace initial_pollykawgs_computation_l9_9296

noncomputable def initial_pollykawgs_in_pond (daily_rate_matured : ℕ) (daily_rate_caught : ℕ)
  (total_days : ℕ) (catch_days : ℕ) : ℕ :=
let first_phase := (daily_rate_matured + daily_rate_caught) * catch_days
let second_phase := daily_rate_matured * (total_days - catch_days)
first_phase + second_phase

theorem initial_pollykawgs_computation :
  initial_pollykawgs_in_pond 50 10 44 20 = 2400 :=
by sorry

end initial_pollykawgs_computation_l9_9296


namespace conditions_for_inequality_l9_9138

theorem conditions_for_inequality (a b : ℝ) :
  (∀ x : ℝ, abs ((x^2 + a * x + b) / (x^2 + 2 * x + 2)) < 1) → 
  (a = 2 ∧ 0 < b ∧ b < 2) :=
sorry

end conditions_for_inequality_l9_9138


namespace extreme_points_count_l9_9769

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≤ 0 then (x + 1) ^ 3 * Real.exp (x + 1) else (if x > 0 then (-(x + 1)) ^ 3 * Real.exp (-(x + 1)) else 0)

theorem extreme_points_count :
  (∀ x : ℝ, f x = f (-x)) → (∀ x : ℝ, x ≤ 0 → f x = (x + 1) ^ 3 * Real.exp (x + 1)) →
  ∃ n : ℕ, n = 3 ∧ extreme_points f n := 
sorry

end extreme_points_count_l9_9769


namespace jello_cost_calculation_l9_9475

-- Conditions as definitions
def jello_per_pound : ℝ := 1.5
def tub_volume_cubic_feet : ℝ := 6
def cubic_foot_to_gallons : ℝ := 7.5
def gallon_weight_pounds : ℝ := 8
def cost_per_tablespoon_jello : ℝ := 0.5

-- Tub total water calculation
def tub_water_gallons (volume_cubic_feet : ℝ) (cubic_foot_to_gallons : ℝ) : ℝ :=
  volume_cubic_feet * cubic_foot_to_gallons

-- Water weight calculation
def water_weight_pounds (water_gallons : ℝ) (gallon_weight_pounds : ℝ) : ℝ :=
  water_gallons * gallon_weight_pounds

-- Jello mix required calculation
def jello_mix_tablespoons (water_pounds : ℝ) (jello_per_pound : ℝ) : ℝ :=
  water_pounds * jello_per_pound

-- Total cost calculation
def total_cost (jello_mix_tablespoons : ℝ) (cost_per_tablespoon_jello : ℝ) : ℝ :=
  jello_mix_tablespoons * cost_per_tablespoon_jello

-- Theorem statement
theorem jello_cost_calculation :
  total_cost (jello_mix_tablespoons (water_weight_pounds (tub_water_gallons tub_volume_cubic_feet cubic_foot_to_gallons) gallon_weight_pounds) jello_per_pound) cost_per_tablespoon_jello = 270 := 
by sorry

end jello_cost_calculation_l9_9475


namespace purchased_only_A_l9_9982

-- Definitions for the conditions
def total_B (x : ℕ) := x + 500
def total_A (y : ℕ) := 2 * y

-- Question formulated in Lean 4
theorem purchased_only_A : 
  ∃ C : ℕ, (∀ x y : ℕ, 2 * x = 500 → y = total_B x → 2 * y = total_A y → C = total_A y - 500) ∧ C = 1000 :=
  sorry

end purchased_only_A_l9_9982


namespace jello_cost_calculation_l9_9473

-- Conditions as definitions
def jello_per_pound : ℝ := 1.5
def tub_volume_cubic_feet : ℝ := 6
def cubic_foot_to_gallons : ℝ := 7.5
def gallon_weight_pounds : ℝ := 8
def cost_per_tablespoon_jello : ℝ := 0.5

-- Tub total water calculation
def tub_water_gallons (volume_cubic_feet : ℝ) (cubic_foot_to_gallons : ℝ) : ℝ :=
  volume_cubic_feet * cubic_foot_to_gallons

-- Water weight calculation
def water_weight_pounds (water_gallons : ℝ) (gallon_weight_pounds : ℝ) : ℝ :=
  water_gallons * gallon_weight_pounds

-- Jello mix required calculation
def jello_mix_tablespoons (water_pounds : ℝ) (jello_per_pound : ℝ) : ℝ :=
  water_pounds * jello_per_pound

-- Total cost calculation
def total_cost (jello_mix_tablespoons : ℝ) (cost_per_tablespoon_jello : ℝ) : ℝ :=
  jello_mix_tablespoons * cost_per_tablespoon_jello

-- Theorem statement
theorem jello_cost_calculation :
  total_cost (jello_mix_tablespoons (water_weight_pounds (tub_water_gallons tub_volume_cubic_feet cubic_foot_to_gallons) gallon_weight_pounds) jello_per_pound) cost_per_tablespoon_jello = 270 := 
by sorry

end jello_cost_calculation_l9_9473


namespace students_in_class_l9_9576

theorem students_in_class (S : ℕ) 
  (h1 : (1 / 4) * (9 / 10 : ℚ) * S = 9) : S = 40 :=
sorry

end students_in_class_l9_9576


namespace total_profit_percentage_of_pet_store_is_227_67_l9_9659

def gecko_cost := 6 * 100
def parrot_cost := 3 * 200
def tarantula_cost := 10 * 50

def gecko_discount := if 6 >= 5 then 0.10 * gecko_cost else 0
def tarantula_discount := if 10 >= 5 then 0.10 * tarantula_cost else 0

def total_cost := gecko_cost - gecko_discount + parrot_cost + tarantula_cost - tarantula_discount

def gecko_sale_price := 6 * (3 * 100 + 5)
def parrot_sale_price := 3 * (2 * 200 + 10)
def tarantula_sale_price := 10 * (4 * 50 + 15)

def total_sale_price := gecko_sale_price + parrot_sale_price + tarantula_sale_price

def total_profit := total_sale_price - total_cost

def profit_percentage := (total_profit / total_cost) * 100

theorem total_profit_percentage_of_pet_store_is_227_67 :
  profit_percentage = 227.67 := by
  sorry

end total_profit_percentage_of_pet_store_is_227_67_l9_9659


namespace check_correct_propositions_l9_9554

def correct_propositions : ℕ := 3

def prop1 (angle : Type) [symmetrical angle] : Prop :=
  ∃ (sym_axis : Type), is_bisector_of sym_axis angle

def prop2 (triangle : Type) [symmetrical triangle] : Prop :=
  ∀ (tri1 tri2 : triangle) (l : line), 
    (is_symmetrical_about tri1 tri2 l) → (congruent tri1 tri2)

def prop3 (pentagon : Type) [regular pentagon] : Prop :=
  ∃ (axes : set line), (size axes = 5) ∧ (∀ axis ∈ axes, is_symmetry_axis axis pentagon)

def prop4 (triangle : Type) [isosceles triangle] : Prop :=
  ∀ (altitude median bisector : line), 
    (is_altitude altitude triangle) → 
    (is_median median triangle) → 
    (is_bisector bisector triangle) → 
    (altitude = median ∧ median = bisector)

def prop5 (triangle : Type) [right triangle] : Prop :=
  ∀ (A B : vertex) (C : vertex), 
    (is_right_triangle A B C) → 
    (angle_C = 30) → 
    (side_opposite angle_C = hypotenuse / 2)

theorem check_correct_propositions :
  (prop1 angle) = false ∧ 
  (prop2 triangle) = true ∧ 
  (prop3 pentagon) = true ∧ 
  (prop4 triangle) = false ∧ 
  (prop5 triangle) = true ∧ 
  (correct_propositions = 3) :=
by sorry

end check_correct_propositions_l9_9554


namespace inequality_sum_gt_n_div_4_l9_9935

theorem inequality_sum_gt_n_div_4 {n : ℕ} (hn : n ≥ 4) (a : ℕ → ℝ) (hpos : ∀ i, 1 ≤ i → i ≤ n → 0 < a i):
  (∑ i in Finset.range n, a i / (a ((i + 1) % n) + a ((i + 2) % n))) > n / 4 := by
  sorry

end inequality_sum_gt_n_div_4_l9_9935


namespace hazel_drank_one_cup_l9_9809

theorem hazel_drank_one_cup (total_cups made_to_crew bike_sold friends_given remaining_cups : ℕ) 
  (H1 : total_cups = 56)
  (H2 : made_to_crew = total_cups / 2)
  (H3 : bike_sold = 18)
  (H4 : friends_given = bike_sold / 2)
  (H5 : remaining_cups = total_cups - (made_to_crew + bike_sold + friends_given)) :
  remaining_cups = 1 := 
sorry

end hazel_drank_one_cup_l9_9809


namespace limit_calculation_l9_9206

noncomputable def limit_expression : ℝ :=
  limit (λ x : ℝ, (2 * exp (x - 1) - 1) ^ ((3 * x - 1)/(x - 1)))

theorem limit_calculation : limit_expression = exp (4) :=
by sorry

end limit_calculation_l9_9206


namespace complex_power_eq_rectangular_l9_9705

noncomputable def cos_30 := Real.cos (Real.pi / 6)
noncomputable def sin_30 := Real.sin (Real.pi / 6)

theorem complex_power_eq_rectangular :
  (3 * (cos_30 + (complex.I * sin_30)))^4 = -40.5 + 40.5 * complex.I * Real.sqrt 3 := 
sorry

end complex_power_eq_rectangular_l9_9705


namespace problem_1_problem_2_problem_3_l9_9342

-- Declaration of the sequence and conditions
variable {a : ℕ → ℚ} {S : ℕ → ℚ}
def a_1 := 2
def Sn (n : ℕ) := finset.sum (finset.range n) a

-- Given condition
def condition (n : ℕ) := (1 / a n) - (1 / a (n + 1)) = 2 / (4 * Sn n - 1)

-- Problem 1: Prove the value of a_2
theorem problem_1 : a 2 = 14 / 3 :=
  sorry

-- Problem 2: Prove the general formula for b_n
def bn (n : ℕ) := a n / (a (n + 1) - a n)
theorem problem_2 : ∀ n, bn n = n - 1 / 4 :=
  sorry

-- Problem 3: Prove the existence of a positive integer n such that a_{n+3} / a_n is an integer
def an (n : ℕ) := 2 / 3 * (4 * n - 1)
theorem problem_3 : ∃ n : ℕ, (n > 0) ∧ ((a (n + 3)) / (a n)).denom = 1 :=
  sorry

end problem_1_problem_2_problem_3_l9_9342


namespace find_value_l9_9432

variable (x y a c : ℝ)

-- Conditions
def condition1 : Prop := x * y = 2 * c
def condition2 : Prop := (1 / x ^ 2) + (1 / y ^ 2) = 3 * a

-- Proof statement
theorem find_value : condition1 x y c ∧ condition2 x y a ↔ (x + y) ^ 2 = 12 * a * c ^ 2 + 4 * c := 
by 
  -- Placeholder for the actual proof
  sorry

end find_value_l9_9432


namespace intersection_point_l9_9734

def parametric_line (t : ℝ) : ℝ × ℝ × ℝ :=
  (-1 - 2 * t, 0, -1 + 3 * t)

def plane (x y z : ℝ) : Prop := x + 4 * y + 13 * z - 23 = 0

theorem intersection_point :
  ∃ t : ℝ, plane (-1 - 2 * t) 0 (-1 + 3 * t) ∧ parametric_line t = (-3, 0, 2) :=
by
  sorry

end intersection_point_l9_9734


namespace smallest_prime_after_six_nonprime_l9_9166

open Nat

noncomputable def is_nonprime (n : ℕ) : Prop :=
  ¬Prime n ∧ n ≠ 1

noncomputable def six_consecutive_nonprime (n : ℕ) : Prop :=
  is_nonprime n ∧ is_nonprime (n + 1) ∧ is_nonprime (n + 2) ∧ 
  is_nonprime (n + 3) ∧ is_nonprime (n + 4) ∧ is_nonprime (n + 5)

theorem smallest_prime_after_six_nonprime :
  ∃ n, six_consecutive_nonprime n ∧ (∀ m, m ≠ 97 → n + 6 < m → ¬Prime m) :=
by
  sorry

end smallest_prime_after_six_nonprime_l9_9166


namespace A_inter_B_eq_A_l9_9380

def A := {x : ℝ | 0 < x ∧ x ≤ 2}
def B := {x : ℝ | x ≤ 3}

theorem A_inter_B_eq_A : A ∩ B = A := 
by 
  sorry 

end A_inter_B_eq_A_l9_9380


namespace sin_phi_equal_sqrt_3_div_2_l9_9493

theorem sin_phi_equal_sqrt_3_div_2 (φ : ℝ) (h1 : 0 < φ ∧ φ < π / 2) 
  (h2 : ∃ a b c : ℝ, {a, b, c} = {cos φ, cos (2 * φ), cos (3 * φ)} ∧ 2 * b = a + c) : 
  sin φ = sqrt 3 / 2 :=
sorry

end sin_phi_equal_sqrt_3_div_2_l9_9493


namespace geometric_mean_side_lengths_of_squares_l9_9095

noncomputable def geometric_mean_of_squares_side_lengths :
  Float :=
let side_length_1 := Real.sqrt 64
let side_length_2 := Real.sqrt 81
let side_length_3 := Real.sqrt 144
let product := side_length_1 * side_length_2 * side_length_3
let geometric_mean := Real.cbrt product
geometric_mean

theorem geometric_mean_side_lengths_of_squares {a₁ a₂ a₃ : ℝ} (h₁ : a₁ = 64) (h₂ : a₂ = 81) (h₃ : a₃ = 144) :
  geometric_mean_of_squares_side_lengths = 9.524361:
by
  sorry

end geometric_mean_side_lengths_of_squares_l9_9095


namespace smallest_sum_of_inverses_l9_9368

theorem smallest_sum_of_inverses 
  (x y : ℕ) (hx : x ≠ y) (h1 : 0 < x) (h2 : 0 < y) (h_condition : (1 / x : ℚ) + 1 / y = 1 / 15) :
  x + y = 64 := 
sorry

end smallest_sum_of_inverses_l9_9368


namespace segment_PL_length_l9_9029

section
variables {X Y Z L P : Type} [metric_space X] [metric_space Y] [metric_space Z]
variables {φ : ℝ} {a b c : ℝ}

-- Hypotheses
variable (h1 : midpoint X Z P)
variable (h2 : angle_bisector Y X Z L)
variable (h3 : perpendicular L X Y)
variable (h4 : distance X Y = 13 * φ)
variable (h5 : distance Y Z = 8 * φ)

-- Target statement
theorem segment_PL_length : distance P L = (1 / 4) * φ * sqrt 233 :=
sorry
end

end segment_PL_length_l9_9029


namespace math_problem_l9_9335

theorem math_problem (n : ℕ) (x : Fin n → ℝ) (hp : ∀ i, 0 < x i) (hs : ∑ i, x i = 1) :
  (∑ i, (x i)^2 / (1 - x i)) ≥ 1 / (n - 1) :=
sorry

end math_problem_l9_9335


namespace geom_seq_sum_4_l9_9856

noncomputable def geom_seq (n : ℕ) (q : ℝ) (a₁ : ℝ) : ℝ :=
  a₁ * q^(n - 1)

noncomputable def sum_geom_seq (n : ℕ) (q : ℝ) (a₁ : ℝ) : ℝ :=
  if q = 1 then a₁ * n else (a₁ * (1 - q^n) / (1 - q))

theorem geom_seq_sum_4 {q : ℝ} (hq : q > 0) (hq1 : q ≠ 1) :
  let a₁ := 1 in
  let S5 := sum_geom_seq 5 q a₁ in
  let S3 := sum_geom_seq 3 q a₁ in
  S5 = 5 * S3 - 4 →
  sum_geom_seq 4 q a₁ = 15 :=
by
  sorry

end geom_seq_sum_4_l9_9856


namespace opposite_face_of_C_is_E_l9_9626

theorem opposite_face_of_C_is_E : 
  ∀ (A B C D E F : Type), 
  (fold_strip_to_cube A B C D E F) -> 
  opposite_face_label C E :=
by
  intros A B C D E F h
  sorry

end opposite_face_of_C_is_E_l9_9626


namespace smallest_degree_of_polynomial_with_given_roots_l9_9953

theorem smallest_degree_of_polynomial_with_given_roots :
  ∀ (P : Polynomial ℚ), (P ≠ 0) →
    P.is_root (2 - Real.sqrt 3) →
    P.is_root (-2 - Real.sqrt 3) →
    P.is_root (3 + Real.sqrt 5) →
    P.is_root (3 - Real.sqrt 5) →
    ∃ (Q : Polynomial ℚ), (Q = P) ∧ (Q.degree = 6) :=
by
  intros P hP h1 h2 h3 h4
  sorry

end smallest_degree_of_polynomial_with_given_roots_l9_9953


namespace total_charge_rush_hour_trip_l9_9034

def initial_fee : ℝ := 2.35
def non_rush_hour_cost_per_two_fifths_mile : ℝ := 0.35
def rush_hour_cost_increase_percentage : ℝ := 0.20
def traffic_delay_cost_per_mile : ℝ := 1.50
def distance_travelled : ℝ := 3.6

theorem total_charge_rush_hour_trip (initial_fee : ℝ) 
  (non_rush_hour_cost_per_two_fifths_mile : ℝ) 
  (rush_hour_cost_increase_percentage : ℝ)
  (traffic_delay_cost_per_mile : ℝ)
  (distance_travelled : ℝ) : 
  initial_fee = 2.35 → 
  non_rush_hour_cost_per_two_fifths_mile = 0.35 →
  rush_hour_cost_increase_percentage = 0.20 →
  traffic_delay_cost_per_mile = 1.50 →
  distance_travelled = 3.6 →
  (initial_fee + ((5/2) * (non_rush_hour_cost_per_two_fifths_mile * (1 + rush_hour_cost_increase_percentage))) * distance_travelled + (traffic_delay_cost_per_mile * distance_travelled)) = 11.53 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end total_charge_rush_hour_trip_l9_9034


namespace four_digit_palindromes_l9_9812

theorem four_digit_palindromes (n : ℕ) : n = 7 :=
  let palindromes := { x : ℕ | 1000 ≤ x ∧ x < 3000 ∧ (x.toString = x.toString.reverse) ∧ (x.digits.sum < 10) } in
  n = finset.card palindromes.to_finset
sorry

end four_digit_palindromes_l9_9812


namespace quadratic_roots_range_l9_9439

theorem quadratic_roots_range (a : ℝ) : 
  (∃ p q : ℝ, p > 0 ∧ q < 0 ∧ p + q = -2 * (a - 1) ∧ p * q = 2a + 6) → a < -3 :=
by 
  sorry

end quadratic_roots_range_l9_9439


namespace complex_power_result_l9_9685

theorem complex_power_result :
  (3 * Complex.cos (Real.pi / 6) + 3 * Complex.sin (Real.pi / 6) * Complex.i)^4 = 
  -40.5 + 40.5 * Complex.i * Real.sqrt 3 :=
by sorry

end complex_power_result_l9_9685


namespace angle_length_theorem_l9_9901

-- Define the points and their properties
variables {A B C D : Type} [InnerProductSpace ℝ D]
variables {angle : D → D → ℝ}

-- Conditions: D is an interior point of an acute triangle △ABC
-- and ∠ADB = ∠ACB + 90° and AC * BD = AD * BC
def angle_condition (A B C D : D) : Prop :=
  angle A D B = angle A C B + 90

def length_condition (A B C D : D) : Prop :=
  dist A C * dist B D = dist A D * dist B C

-- Define the theorem to prove:
theorem angle_length_theorem (A B C D : D) (h_angle : angle_condition A B C D) (h_length : length_condition A B C D) :
  (dist A B * dist C D) / (dist A C * dist B D) = sqrt 2 :=
by
  sorry

end angle_length_theorem_l9_9901


namespace period_of_sin_x_over_3_is_6pi_l9_9715

def period_sin_x_over_3 : ℝ := 6 * Real.pi

theorem period_of_sin_x_over_3_is_6pi : ∃ T : ℝ, ∀ x : ℝ, sin (x + period_sin_x_over_3) / 3 = sin (x / 3) := sorry

end period_of_sin_x_over_3_is_6pi_l9_9715


namespace seashells_given_l9_9710

theorem seashells_given (original_seashells : ℕ) (seashells_left : ℕ) 
    (h1 : original_seashells = 56) (h2 : seashells_left = 22) : 
  original_seashells - seashells_left = 34 :=
by
  rw [h1, h2]
  rfl

end seashells_given_l9_9710


namespace paul_collected_total_cans_l9_9519

theorem paul_collected_total_cans :
  let saturday_bags := 10
  let sunday_bags := 5
  let saturday_cans_per_bag := 12
  let sunday_cans_per_bag := 15
  let saturday_total_cans := saturday_bags * saturday_cans_per_bag
  let sunday_total_cans := sunday_bags * sunday_cans_per_bag
  let total_cans := saturday_total_cans + sunday_total_cans
  total_cans = 195 := 
by
  sorry

end paul_collected_total_cans_l9_9519


namespace A_ge_B_l9_9328

def A (a b : ℝ) : ℝ := a^3 + 3 * a^2 * b^2 + 2 * b^2 + 3 * b
def B (a b : ℝ) : ℝ := a^3 - a^2 * b^2 + b^2 + 3 * b

theorem A_ge_B (a b : ℝ) : A a b ≥ B a b := by
  sorry

end A_ge_B_l9_9328


namespace expected_value_X_l9_9589

noncomputable def fair_coin := MassFunction.replicate 2 0.5

def is_heads_tails (outcome : Finset ℕ) : Prop :=
  outcome.card = 2 ∧ 1 ∈ outcome ∧ 2 ∈ outcome

def X_binomial_distribution : ℕ → ℕ → MassFunction ℕ
| n p := MassFunction.bind (MassFunction.replicate n 0.5 ν) X

theorem expected_value_X :
  X_binomial_distribution 4 (is_heads_tails) = 2 := sorry

end expected_value_X_l9_9589


namespace geometric_sequence_s4_l9_9850

theorem geometric_sequence_s4
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (q : ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a (n + 1) = a n * q)
  (h3 : ∀ n : ℕ, S n = (∑ i in finset.range n, a (i + 1)))
  (h4 : S 5 = 5 * S 3 - 4) :
  S 4 = 15 :=
sorry

end geometric_sequence_s4_l9_9850


namespace committee_vote_change_l9_9445

-- Let x be the number of votes for the resolution initially.
-- Let y be the number of votes against the resolution initially.
-- The total number of voters is 500: x + y = 500.
-- The initial margin by which the resolution was defeated: y - x = m.
-- In the re-vote, the resolution passed with a margin three times the initial margin: x' - y' = 3m.
-- The number of votes for the re-vote was 13/12 of the votes against initially: x' = 13/12 * y.
-- The total number of voters remains 500 in the re-vote: x' + y' = 500.

theorem committee_vote_change (x y x' y' m : ℕ)
  (h1 : x + y = 500)
  (h2 : y - x = m)
  (h3 : x' - y' = 3 * m)
  (h4 : x' = 13 * y / 12)
  (h5 : x' + y' = 500) : x' - x = 40 := 
  by
  sorry

end committee_vote_change_l9_9445


namespace complex_exp_form_pow_four_l9_9670

theorem complex_exp_form_pow_four :
  let θ := 30 * Real.pi / 180
  let cos_θ := Real.cos θ
  let sin_θ := Real.sin θ
  let z := 3 * (cos_θ + Complex.I * sin_θ)
  z ^ 4 = -40.5 + 40.5 * Complex.I * Real.sqrt 3 :=
by
  sorry

end complex_exp_form_pow_four_l9_9670


namespace length_BC_of_trapezoid_l9_9094

theorem length_BC_of_trapezoid 
  (area_ABCD : ℝ) 
  (altitude : ℝ) 
  (AB : ℝ) 
  (CD : ℝ) 
  (H1 : area_ABCD = 200) 
  (H2 : altitude = 10) 
  (H3 : AB = 12) 
  (H4 : CD = 22) : 
  (BC : ℝ), BC = 20 - real.sqrt 11 - 6 * real.sqrt 6 := 
sorry

end length_BC_of_trapezoid_l9_9094


namespace complex_fourth_power_l9_9698

-- Definitions related to the problem conditions
def trig_identity (θ : ℝ) : ℂ := complex.of_real (cos θ) + complex.I * complex.of_real (sin θ)
def z : ℂ := 3 * trig_identity (30 * real.pi / 180)
def result := (81 : ℝ) * (-1/2 + complex.I * (real.sqrt 3 / 2))

-- Statement of the problem
theorem complex_fourth_power : z^4 = result := by
  sorry

end complex_fourth_power_l9_9698


namespace burger_meal_cost_l9_9520

-- Define the conditions
variables (B S : ℝ)
axiom cost_of_soda : S = (1 / 3) * B
axiom total_cost : B + S + 2 * (B + S) = 24

-- Prove that the cost of the burger meal is $6
theorem burger_meal_cost : B = 6 :=
by {
  -- We'll use both the axioms provided to show B equals 6
  sorry
}

end burger_meal_cost_l9_9520


namespace probability_sin_between_half_l9_9623

noncomputable def probability_sin_between (a b : ℝ) : ℝ :=
if x ∈ Set.Icc (-π / 2) (π / 2) then 
    (∫ y in Set.Ioo (-π / 6) (π / 6), indicator 1 y) / 
    (∫ y in Set.Icc (-π / 2) (π / 2), indicator 1 y) 
else 0

theorem probability_sin_between_half : 
  probability_sin_between (-1 / 2) (1 / 2) = 1 / 3 := 
sorry

end probability_sin_between_half_l9_9623


namespace eleventh_flip_tails_probability_l9_9035

-- Define a fair coin where the probability of getting heads or tails is 1/2
structure FairCoin where
  p_heads : ℚ              -- probability of heads
  p_tails : ℚ              -- probability of tails
  fair : p_heads = 1/2 ∧ p_tails = 1/2

-- The event of flipping the coin
def flip_coin (c : FairCoin) := c

-- The problem statement:
theorem eleventh_flip_tails_probability (c : FairCoin) (independent_flips : ∀ (n: ℕ), flip_coin c) :
  (independent_flips 10).p_tails = 1 / 2 := by
  -- proof omitted
  sorry

end eleventh_flip_tails_probability_l9_9035


namespace smallest_x_plus_y_l9_9350

theorem smallest_x_plus_y (x y : ℕ) (h_xy_not_equal : x ≠ y) (h_condition : (1 : ℚ) / x + 1 / y = 1 / 15) :
  x + y = 64 :=
sorry

end smallest_x_plus_y_l9_9350


namespace sin_cos_square_l9_9949

theorem sin_cos_square (α : ℝ) : (Math.sin α + Math.cos α) ^ 2 = 1 + Math.sin (2 * α) := 
by 
  sorry

end sin_cos_square_l9_9949


namespace number_of_items_l9_9012

variable (s d : ℕ)
variable (total_money cost_sandwich cost_drink discount : ℝ)
variable (s_purchase_criterion : s > 5)
variable (total_money_value : total_money = 50.00)
variable (cost_sandwich_value : cost_sandwich = 6.00)
variable (cost_drink_value : cost_drink = 1.50)
variable (discount_value : discount = 5.00)

theorem number_of_items (h1 : total_money = 50.00)
(h2 : cost_sandwich = 6.00)
(h3 : cost_drink = 1.50)
(h4 : discount = 5.00)
(h5 : s > 5) :
  s + d = 9 :=
by
  sorry

end number_of_items_l9_9012


namespace sum_of_interior_angles_of_regular_polygon_l9_9972

theorem sum_of_interior_angles_of_regular_polygon (n : ℕ) (h1: ∀ {n : ℕ}, ∃ (e : ℕ), (e = 360 / 45) ∧ (n = e)) :
  (180 * (n - 2)) = 1080 :=
by
  let n := (360 / 45)
  have h : n = 8 := by sorry
  calc
    180 * (n - 2) = 180 * (8 - 2) : by rw [h]
    ... = 1080 : by norm_num

end sum_of_interior_angles_of_regular_polygon_l9_9972


namespace truncated_cone_volume_l9_9132

noncomputable def volume_of_truncated_cone (R r h : ℝ) : ℝ :=
  let V_large := (1 / 3) * Real.pi * R^2 * (h + h)  -- Height of larger cone is h + x = h + h
  let V_small := (1 / 3) * Real.pi * r^2 * h       -- Height of smaller cone is h
  V_large - V_small

theorem truncated_cone_volume (R r h : ℝ) (hR : R = 8) (hr : r = 4) (hh : h = 6) :
  volume_of_truncated_cone R r h = 224 * Real.pi :=
by
  sorry

end truncated_cone_volume_l9_9132


namespace projection_matrix_correct_l9_9044

def normal_vector : ℝ × ℝ × ℝ := (2, -1, 2)

def Q : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    [5 / 9, 2 / 9, -4 / 9],
    [2 / 9, 8 / 9, 2 / 9],
    [-4 / 9, 2 / 9, 5 / 9]
  ]

theorem projection_matrix_correct : ∀ (v : ℝ × ℝ × ℝ),
  let Q_v := ![
    ((5 / 9 : ℝ) * v.1 + (2 / 9) * v.2 + (-4 / 9) * v.3),
    (2 / 9 * v.1 + 8 / 9 * v.2 + 2 / 9 * v.3),
    (-4 / 9 * v.1 + 2 / 9 * v.2 + 5 / 9 * v.3)
  ] in
  Q_v = (v.1 - (2 * (2 * v.1 - v.2 + 2 * v.3) / 9),
         v.2 - (-1 * (2 * v.1 - v.2 + 2 * v.3) / 9),
         v.3 - (2 * (2 * v.1 - v.2 + 2 * v.3) / 9)) := 
sorry

end projection_matrix_correct_l9_9044


namespace complex_power_result_l9_9679

theorem complex_power_result :
  (3 * Complex.cos (Real.pi / 6) + 3 * Complex.sin (Real.pi / 6) * Complex.i)^4 = 
  -40.5 + 40.5 * Complex.i * Real.sqrt 3 :=
by sorry

end complex_power_result_l9_9679


namespace unique_solutions_l9_9305

-- Noncomputable theory is required since we are dealing explicitly with real numbers
noncomputable theory
open Classical

-- Definitions for the conditions
def eq1 (x y z : ℝ) := x + 4*y + 6*z = 16
def eq2 (x y z : ℝ) := x + 6*y + 12*z = 24
def eq3 (x y z : ℝ) := x^2 + 4*y^2 + 36*z^2 = 76

-- The two solutions we need to prove are valid
def solution1 := (6 : ℝ, 1 : ℝ, 1 : ℝ)
def solution2 := (-2/3 : ℝ, 13/3 : ℝ, -1/9 : ℝ)

-- The theorem statement, proving that solution1 and solution2 are the only solutions
theorem unique_solutions (x y z : ℝ) (h1 : eq1 x y z) (h2 : eq2 x y z) (h3 : eq3 x y z) :
  (x = solution1.1 ∧ y = solution1.2 ∧ z = solution1.3) ∨ (x = solution2.1 ∧ y = solution2.2 ∧ z = solution2.3) :=
sorry

end unique_solutions_l9_9305


namespace integral_evaluation_l9_9730

theorem integral_evaluation {a : ℝ} (ha : 0 < a) : 
  ∫ x in -1..a^2, 1 / (x^2 + a^2) = (Real.pi / 2) / a := 
by sorry

end integral_evaluation_l9_9730


namespace symmetric_point_in_2nd_quadrant_l9_9832

theorem symmetric_point_in_2nd_quadrant {x y : ℝ} (h : x > 0 ∧ y < 0) :
  -x < 0 ∧ -y > 0 :=
by
  finish

end symmetric_point_in_2nd_quadrant_l9_9832


namespace smallest_sum_of_xy_l9_9361

theorem smallest_sum_of_xy (x y : ℕ) (h1 : x ≠ y) (h2 : 0 < x) (h3 : 0 < y) 
  (h4 : (1:ℚ)/x + (1:ℚ)/y = (1:ℚ)/15) : x + y = 64 :=
sorry

end smallest_sum_of_xy_l9_9361


namespace prime_log_sum_lt_one_l9_9988

open Real

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
∀ n : ℕ, (∏ i in finset.range (n^2), (a n - i)) > 0 ∧ (∏ i in finset.range (n^2), (a n - i)) % n^(n^2 - 1) = 0

theorem prime_log_sum_lt_one (a : ℕ → ℕ) (h_seq : sequence a) (P : finset ℕ) (hP : ∀ p ∈ P, Nat.Prime p) :
  ∑ p in P, (1 / log (a p)) < 1 :=
sorry

end prime_log_sum_lt_one_l9_9988


namespace movement_on_unit_circle_l9_9070

theorem movement_on_unit_circle 
  (P : ℝ × ℝ)
  (Q : ℝ × ℝ)
  (R : ℝ)
  (unit_circle : ∀ (x y : ℝ), x^2 + y^2 = 1 → (x, y) ∈ unit_circle)
  (start_P : P = (1, 0))
  (arc_length : R = 4 * Real.pi / 3) :
  Q = (-1/2, Real.sqrt 3/2) :=
sorry

end movement_on_unit_circle_l9_9070


namespace area_of_region_Q_l9_9506

theorem area_of_region_Q (P O : Point) (α : Plane) (dist_PO : dist P α = √3) (Q : Point)
  (hQ : Q ∈ α) :
  ∃ r1 r2 : ℝ, 1 = r1 ∧ 3 = r2 ∧
  (∀ Q ∈ α, π * r2^2 - π * r1^2 = 8 * π) := sorry

end area_of_region_Q_l9_9506


namespace total_first_class_equipment_l9_9266

theorem total_first_class_equipment (x y : ℕ) (h1 : x < y)
    (h2 : 0.9 * (y + 0.3 * x) > 1.02 * y)
    (h3 : 0.73 * x + 0.1 * y = 0.27 * x + 0.9 * y + 6) :
  y = 17 :=
by
  sorry -- Proof is skipped as per the instructions

end total_first_class_equipment_l9_9266


namespace simple_polygon_has_n_minus_3_internal_diagonals_l9_9246

-- Define a simple polygon and a function to count its internal diagonals
structure SimplePolygon (n : ℕ) :=
(is_simple : Prop)  -- representing the non-intersecting property

def count_internal_diagonals (p : SimplePolygon) (n : ℕ) : ℕ := sorry

-- The theorem statement
theorem simple_polygon_has_n_minus_3_internal_diagonals (p : SimplePolygon n) (h : p.is_simple) :
  count_internal_diagonals p n ≥ n - 3 :=
sorry

end simple_polygon_has_n_minus_3_internal_diagonals_l9_9246


namespace sum_of_QR_l9_9888

noncomputable def given_conditions := 
  ∃ (P Q R : Type) (angleQ : ℝ) (PQ PR : ℝ), 
    angleQ = real.pi / 4 ∧
    PQ = 100 ∧ 
    PR = 100 * real.sqrt 2

noncomputable def verify_QR (QR : ℝ) := 
  ∃ (P Q R : Type) (angleQ : ℝ) (PQ PR : ℝ), 
    angleQ = real.pi / 4 ∧
    PQ = 100 ∧
    PR = 100 * real.sqrt 2 ∧
    QR = 100 * real.sqrt 3

theorem sum_of_QR : given_conditions → verify_QR  (100 * real.sqrt 3) :=
by
  intro h
  sorry

end sum_of_QR_l9_9888


namespace find_an_l9_9338

open Nat

-- Define the sequence {a_n} with m terms and the sum sequence S(n)
def seq : ℕ → ℕ := sorry
def S (n : ℕ) : ℕ := n * n

-- State the theorem to be proved
theorem find_an (m : ℕ) (h : 1 ≤ m) (n : ℕ) (hn : 1 ≤ n ∧ n < m) :
  seq n = -2 * n - 1 := sorry

end find_an_l9_9338


namespace solve_log_equation_l9_9991

open Real

theorem solve_log_equation :
  ∃ x : ℝ, log 2 (4 ^ x + 4) = x + log 2 (2 ^ (x + 1) - 3) ∧ x = 2 :=
by
  sorry

end solve_log_equation_l9_9991


namespace cylinder_volume_proof_l9_9109

noncomputable def cylinder_volume (height radius : ℝ) : ℝ :=
  π * radius^2 * height

theorem cylinder_volume_proof :
  (let side := 1 in
   let radius := 1 / (2 * π) in
   let height := 1 in
   cylinder_volume height radius = 1 / (4 * π)) :=
by 
  let side := 1
  let radius := 1 / (2 * π)
  let height := 1
  show cylinder_volume height radius = 1 / (4 * π)
  by sorry

end cylinder_volume_proof_l9_9109


namespace parallel_iff_segment_length_l9_9282

variables {A B C D M N : Type} [ConvexQuadrilateral A B C D] [PointOnSegment M A B] [PointOnSegment N C D]
variables {k : ℝ} (hM : AM / BM = k) (hN : DN / CN = k)

theorem parallel_iff_segment_length :
  BC ∥ AD ↔ MN = (1 / (k + 1)) * AD + (k / (k + 1)) * BC :=
sorry

end parallel_iff_segment_length_l9_9282


namespace complex_power_result_l9_9684

theorem complex_power_result :
  (3 * Complex.cos (Real.pi / 6) + 3 * Complex.sin (Real.pi / 6) * Complex.i)^4 = 
  -40.5 + 40.5 * Complex.i * Real.sqrt 3 :=
by sorry

end complex_power_result_l9_9684


namespace house_number_digits_cost_l9_9426

/-
The constants represent:
- cost_1: the cost of 1 unit (1000 rubles)
- cost_12: the cost of 12 units (2000 rubles)
- cost_512: the cost of 512 units (3000 rubles)
- P: the cost per digit of a house number (1000 rubles)
- n: the number of digits in a house number
- The goal is to prove that the cost for 1, 12, and 512 units follows the pattern described
-/

theorem house_number_digits_cost :
  ∃ (P : ℕ),
    (P = 1000) ∧
    (∃ (cost_1 cost_12 cost_512 : ℕ),
      cost_1 = 1000 ∧
      cost_12 = 2000 ∧
      cost_512 = 3000 ∧
      (∃ n1 n2 n3 : ℕ,
        n1 = 1 ∧
        n2 = 2 ∧
        n3 = 3 ∧
        cost_1 = P * n1 ∧
        cost_12 = P * n2 ∧
        cost_512 = P * n3)) :=
by
  sorry

end house_number_digits_cost_l9_9426


namespace geom_seq_sum_4_l9_9857

noncomputable def geom_seq (n : ℕ) (q : ℝ) (a₁ : ℝ) : ℝ :=
  a₁ * q^(n - 1)

noncomputable def sum_geom_seq (n : ℕ) (q : ℝ) (a₁ : ℝ) : ℝ :=
  if q = 1 then a₁ * n else (a₁ * (1 - q^n) / (1 - q))

theorem geom_seq_sum_4 {q : ℝ} (hq : q > 0) (hq1 : q ≠ 1) :
  let a₁ := 1 in
  let S5 := sum_geom_seq 5 q a₁ in
  let S3 := sum_geom_seq 3 q a₁ in
  S5 = 5 * S3 - 4 →
  sum_geom_seq 4 q a₁ = 15 :=
by
  sorry

end geom_seq_sum_4_l9_9857


namespace math_problem_l9_9451

open Real

-- Definition for polar curve transformation to rectangular coordinates
def polar_to_rectangular (ρ θ a : ℝ) : Prop :=
  (ρ * sin θ)^2 = a * (ρ * cos θ) → ρ^2 * sin (θ)^2 = a * (ρ * cos (θ))

-- The polar coordinate system equation is given to be transformed
def curve_rectangular (x y a : ℝ) : Prop :=
  y^2 = a * x

-- Line parametric to standard conversion
def parametric_to_standard (x y t : ℝ) : Prop :=
  x = -2 + (sqrt 2)/2 * t ∧ y = -4 + (sqrt 2)/2 * t → y = x - 2

-- Given conditions for points A and B and relate the distance condition
def distance_condition (t₁ t₂ a : ℝ) : Prop :=
  (sqrt 2 * (a + 8))^2 = 20 * (a + 8) → a = 2

-- Main theorem to prove the problem equivalence
theorem math_problem (a : ℝ) (θ : ℝ) (ρ t x y t₁ t₂ : ℝ) : Prop :=
  polar_to_rectangular ρ θ a ∧ 
  curve_rectangular x y a ∧ 
  parametric_to_standard x y t ∧ 
  distance_condition t₁ t₂ a :=
  by
    unfold polar_to_rectangular
    unfold curve_rectangular
    unfold parametric_to_standard
    unfold distance_condition
    sorry

end math_problem_l9_9451


namespace monotonic_intervals_max_min_values_l9_9405

noncomputable def f (x : ℝ) : ℝ := 
  cos x * sin (x - π / 6) + cos (2 * x) + 1 / 4

theorem monotonic_intervals :
  ∀ k : ℤ, ∃ I : set ℝ, I = {x | k * π - 5 * π / 3 ≤ x ∧ x ≤ k * π + π / 12} ∧ 
  (∀ x ∈ I, ∃ ε > 0, ∀ y (h : x ≤ y ∧ y ≤ x + ε), f y ≥ f x) := 
sorry

theorem max_min_values :
  ∃ mx mn : ℝ, mn = -sqrt 3 / 4 ∧ mx = sqrt 3 / 2 ∧ 
  (∀ x : ℝ, x ∈ Ioo (- π / 12) (5 * π / 12) → f x ≤ mx ∧ f x ≥ mn) := 
sorry

end monotonic_intervals_max_min_values_l9_9405


namespace determine_x_l9_9290

theorem determine_x (x : ℝ) (h : (1 / (Real.log x / Real.log 3) + 1 / (Real.log x / Real.log 5) + 1 / (Real.log x / Real.log 6) = 1)) : 
    x = 90 := 
by 
  sorry

end determine_x_l9_9290


namespace min_value_nS_n_l9_9759

theorem min_value_nS_n (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) 
  (h2 : m ≥ 2)
  (h3 : S (m - 1) = -2)
  (h4 : S m = 0)
  (h5 : S (m + 1) = 3) :
  ∃ n : ℕ, n * S n = -9 :=
sorry

end min_value_nS_n_l9_9759


namespace mean_of_elements_increased_by_2_l9_9979

noncomputable def calculate_mean_after_increase (m : ℝ) (median_value : ℝ) (increase_value : ℝ) : ℝ :=
  let set := [m, m + 2, m + 4, m + 7, m + 11, m + 13]
  let increased_set := set.map (λ x => x + increase_value)
  increased_set.sum / increased_set.length

theorem mean_of_elements_increased_by_2 (m : ℝ) (h : (m + 4 + m + 7) / 2 = 10) :
  calculate_mean_after_increase m 10 2 = 38 / 3 :=
by 
  sorry

end mean_of_elements_increased_by_2_l9_9979


namespace complex_fourth_power_l9_9699

-- Definitions related to the problem conditions
def trig_identity (θ : ℝ) : ℂ := complex.of_real (cos θ) + complex.I * complex.of_real (sin θ)
def z : ℂ := 3 * trig_identity (30 * real.pi / 180)
def result := (81 : ℝ) * (-1/2 + complex.I * (real.sqrt 3 / 2))

-- Statement of the problem
theorem complex_fourth_power : z^4 = result := by
  sorry

end complex_fourth_power_l9_9699


namespace f_919_l9_9768

noncomputable def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then 6^(-x)
  else sorry  -- We are not concerned with defining f(x) outside of this range for the proof

axiom even_f : ∀ x : ℝ, f(x) = f(-x)

axiom periodic_f : ∀ x : ℝ, f(x + 4) = f(x - 2)

theorem f_919 : f 919 = 6 :=
by
  -- Step 1: Show that f is periodic with period 6
  have per_6 : ∀ x : ℝ, f(x + 6) = f(x),
  { intros x,
    calc
      f(x + 6) = f(x + 6 - 4 + 4) : by rw [← add_sub_assoc, sub_add_cancel]
          ... = f(x + 2) : by rw periodic_f
          ... = f(x) : by rw periodic_f },

  -- Step 2: Convert 919 to the form of x + 6k
  have mod_step : f 919 = f (919 % 6),
  { exact per_6 919 },

  -- Step 3: Simplify 919 % 6
  have mod_val : 919 % 6 = 1, by norm_num,

  -- Step 4: f(1) = f(-1) due to even property
  have f1_eq_fneg1 : f 1 = f (-1), from even_f 1,

  -- Step 5: Derive from given conditions
  have fneg1_val : f (-1) = 6, by
  { have h1 : -1 ∈ set.Icc (-3 : ℝ) 0, {
      simp [set.Icc, le_refl, neg_lt, real.zero_lt_one] },
    simpa using h1 },

  -- Final step: Combination of these steps
  calc
    f 919 = f 1 : by rw [mod_step, mod_val]
        ... = f (-1) : f1_eq_fneg1
        ... = 6 : fneg1_val

end f_919_l9_9768


namespace candy_probability_l9_9237

theorem candy_probability (m n : ℕ) (h_rel_prime : Nat.coprime m n) (total_prob : (15.choose 3 * 12.choose 3 * 9.choose 3 * 2) * 15.choose 3⁻¹ * 27.choose 3⁻¹ * 24.choose 3⁻¹ = m * n⁻¹) (h_m_n : m = 4 ∧ n = 2513) : m + n = 2517 := by
  cases h_m_n with h_m h_n
  rw [h_m, h_n]
  norm_num

end candy_probability_l9_9237


namespace derivative_of_sin_cos_eq_cos2x_l9_9549

noncomputable def f (x : ℝ) : ℝ := sin x * cos x

-- Mathematically equivalent proof problem:
-- Prove that the derivative of y = sin x * cos x is cos 2x.
theorem derivative_of_sin_cos_eq_cos2x :
  ∀ x : ℝ, deriv f x = cos (2 * x) :=
by
  intro x
  sorry

end derivative_of_sin_cos_eq_cos2x_l9_9549


namespace triangle_fraction_squared_l9_9243

theorem triangle_fraction_squared (a b c : ℝ) (h1 : b > a) 
  (h2 : a / b = (1 / 2) * (b / c)) (h3 : a + b + c = 12) 
  (h4 : c = Real.sqrt (a^2 + b^2)) : 
  (a / b)^2 = 1 / 2 := 
by 
  sorry

end triangle_fraction_squared_l9_9243


namespace maximum_captain_coins_l9_9535

theorem maximum_captain_coins (captain : ℕ) (crew : Fin 5 → ℕ) :
  ∃ (captain_max : ℕ), captain_max = 59 ∧ 
  (∀ (captain + crew.sum = 180) ∧ 
   (∀ i, crew i ≠ crew ((i + 1) % 5) ∧ crew i > crew ((i + 2) % 5) ∧ crew i > crew ((i - 1) % 5)) ∧ 
   (crew.filter (λ c, c > captain)).length ≥ 3) := 
sorry

end maximum_captain_coins_l9_9535


namespace real_root_exists_for_all_K_l9_9289

theorem real_root_exists_for_all_K (K : ℝ) : ∃ x : ℝ, x = K^2 * (x - 1) * (x - 2) * (x - 3) :=
sorry

end real_root_exists_for_all_K_l9_9289


namespace num_points_80_ray_not_40_ray_partitional_l9_9498

def is_n_ray_partitional (R : set (ℝ × ℝ)) (n : ℕ) (X : ℝ × ℝ) : Prop :=
  ∃ (rays : fin n → ℝ × ℝ), 
    (∀ i, i < n → let (x, y) := rays i in (x, y) ∈ R ∧ ¬ (x, y) = X) ∧ 
    (∀ i j, i ≠ j → let (xi, yi) := rays i, (xj, yj) := rays j in (xi, yi) ≠ (xj, yj)) ∧
    (∀ i, let (x, y) := rays i in (area (triangle X (x, y) boundary_of_R)) = (1 / n))

noncomputable def count_partitional_points (R : set (ℝ × ℝ)) (n : ℕ) : ℕ :=
  fintype.card {X : ℝ × ℝ // X ∈ R ∧ is_n_ray_partitional R n X}

theorem num_points_80_ray_not_40_ray_partitional (R : set (ℝ × ℝ)) (unit_square : R = {p | ∃ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1}) : 
  count_partitional_points R 80 - count_partitional_points R 40 = 1197 := 
sorry

end num_points_80_ray_not_40_ray_partitional_l9_9498


namespace expenditure_of_20_is_negative_l9_9278

section

-- Definition of expenditure
def is_income (x : ℤ) := x > 0
def is_expenditure (x : ℤ) := x < 0

-- Conditions
variable (income_30 : ℤ)
variable (_ : is_income income_30)
variable (_ : income_30 = 30)

-- Question: Representation of expenditure of 20
theorem expenditure_of_20_is_negative :
  ∃ (expenditure_20 : ℤ), is_expenditure expenditure_20 ∧ expenditure_20 = -20 :=
by
  use -20
  split
  { -- proof that -20 is an expenditure
    sorry }
  { -- proof that expenditure_20 = -20
    sorry }

end

end expenditure_of_20_is_negative_l9_9278


namespace Froglet_sane_l9_9955

-- Definitions for conditions
def Sane (x : Type) : Prop := sorry  -- Definition of being sane

variable {Servant : Type}
variable (LackeyLecc Froglet : Servant)
variable (Alike : Sane LackeyLecc ↔ Sane Froglet)

-- Theorem statement
theorem Froglet_sane : Sane Froglet :=
by
  sorry

end Froglet_sane_l9_9955


namespace number_of_customers_who_tipped_is_3_l9_9480

-- Definitions of conditions
def charge_per_lawn : ℤ := 33
def lawns_mowed : ℤ := 16
def total_earnings : ℤ := 558
def tip_per_customer : ℤ := 10

-- Calculate intermediate values
def earnings_from_mowing : ℤ := lawns_mowed * charge_per_lawn
def earnings_from_tips : ℤ := total_earnings - earnings_from_mowing
def number_of_tips : ℤ := earnings_from_tips / tip_per_customer

-- Theorem stating our proof
theorem number_of_customers_who_tipped_is_3 : number_of_tips = 3 := by
  sorry

end number_of_customers_who_tipped_is_3_l9_9480


namespace select_one_person_for_both_days_l9_9315

noncomputable def combination (n r : ℕ) := n.choose r

def volunteers := 5
def serve_both_days := combination volunteers 1
def remaining_for_saturday := volunteers - 1
def serve_saturday := combination remaining_for_saturday 1
def remaining_for_sunday := remaining_for_saturday - 1
def serve_sunday := combination remaining_for_sunday 1
def total_ways := serve_both_days * serve_saturday * serve_sunday

theorem select_one_person_for_both_days :
  total_ways = 60 := 
by
  -- We skip the proof details for now
  sorry

end select_one_person_for_both_days_l9_9315


namespace exists_k_zero_l9_9538

-- Define the recurrence relations
def a_next (a b : ℤ) : ℤ := Int.natAbs (a - b)
def b_next (b c : ℤ) : ℤ := Int.natAbs (b - c)
def c_next (c d : ℤ) : ℤ := Int.natAbs (c - d)
def d_next (d a : ℤ) : ℤ := Int.natAbs (d - a)

-- Define the sequences for a_n, b_n, c_n, and d_n
noncomputable def a : ℕ → ℤ → ℤ → ℤ
| 0, a₀, _ := a₀
| (n+1), a₀, b₀ := a_next (a n a₀ b₀) (b n b₀ (c n (c n a₀ b₀) (d n (d n a₀ b₀) a₀)))

noncomputable def b : ℕ → ℤ → ℤ → ℤ
| 0, b₀, _ := b₀
| (n+1), b₀, c₀ := b_next (b n b₀ c₀) (c n c₀ (d n (d n b₀ c₀) a₀))

noncomputable def c : ℕ → ℤ → ℤ → ℤ
| 0, c₀, _ := c₀
| (n+1), c₀, d₀ := c_next (c n c₀ d₀) (d n d₀ (a n (a n c₀ d₀) (b n b₀ (c n d₀ a₀))))

noncomputable def d : ℕ → ℤ → ℤ → ℤ
| 0, d₀, _ := d₀
| (n+1), d₀, a₀ := d_next (d n d₀ a₀) (a n a₀ (b n (b n (c n d₀ a₀) d₀) d₀))

-- Prove there exists k such that a_k = b_k = c_k = d_k = 0
theorem exists_k_zero (a₀ b₀ c₀ d₀ : ℤ) :
  ∃ k : ℕ, a k a₀ b₀ = 0 ∧ b k b₀ c₀ = 0 ∧ c k c₀ d₀ = 0 ∧ d k d₀ a₀ = 0 :=
sorry

end exists_k_zero_l9_9538


namespace calculate_log54_168_l9_9660

noncomputable def log54_168 {a b : ℝ} (h₁ : real.log_base 7 12 = a) (h₂ : real.log_base 12 24 = b) : ℝ :=
  real.log_base 54 168

theorem calculate_log54_168 (a b : ℝ) (h₁ : real.log_base 7 12 = a) (h₂ : real.log_base 12 24 = b) :
  log54_168 h₁ h₂ = (1 + a * b) / (a * (8 - 5 * b)) :=
sorry

end calculate_log54_168_l9_9660


namespace product_of_solutions_l9_9719

theorem product_of_solutions : 
  ∀ x : ℝ, 5 = -2 * x^2 + 6 * x → (∃ α β : ℝ, (α ≠ β ∧ (α * β = 5 / 2))) :=
by
  sorry

end product_of_solutions_l9_9719


namespace smallest_prime_after_six_nonprime_l9_9169

open Nat

noncomputable def is_nonprime (n : ℕ) : Prop :=
  ¬Prime n ∧ n ≠ 1

noncomputable def six_consecutive_nonprime (n : ℕ) : Prop :=
  is_nonprime n ∧ is_nonprime (n + 1) ∧ is_nonprime (n + 2) ∧ 
  is_nonprime (n + 3) ∧ is_nonprime (n + 4) ∧ is_nonprime (n + 5)

theorem smallest_prime_after_six_nonprime :
  ∃ n, six_consecutive_nonprime n ∧ (∀ m, m ≠ 97 → n + 6 < m → ¬Prime m) :=
by
  sorry

end smallest_prime_after_six_nonprime_l9_9169


namespace pairing_32_with_27_l9_9983

theorem pairing_32_with_27 :
  let numbers := [36, 27, 42, 32, 28, 31, 23, 17]
  let expected_pair_sum := 59
  (∃ pairs : list (ℕ × ℕ), (∀ p ∈ pairs, p.1 + p.2 = expected_pair_sum) ∧ (32, 27) ∈ pairs) :=
sorry

end pairing_32_with_27_l9_9983


namespace complex_exp_form_pow_four_l9_9667

theorem complex_exp_form_pow_four :
  let θ := 30 * Real.pi / 180
  let cos_θ := Real.cos θ
  let sin_θ := Real.sin θ
  let z := 3 * (cos_θ + Complex.I * sin_θ)
  z ^ 4 = -40.5 + 40.5 * Complex.I * Real.sqrt 3 :=
by
  sorry

end complex_exp_form_pow_four_l9_9667


namespace limit_value_l9_9833

variable (f : ℝ → ℝ) (x₀ : ℝ)

def tangent_slope_condition : Prop :=
1 = limit (λ Δx : ℝ, (f (x₀ + Δx) - f x₀) / Δx) 0

theorem limit_value (h : tangent_slope_condition f x₀) :
  limit (λ Δx : ℝ, (f x₀ - f (x₀ - 2 * Δx)) / Δx) 0 = 2 :=
sorry

end limit_value_l9_9833


namespace problem_correct_choice_l9_9379

-- Definitions of the propositions
def p : Prop := ∃ n : ℕ, 3 = 2 * n + 1
def q : Prop := ∃ n : ℕ, 5 = 2 * n

-- The problem statement
theorem problem_correct_choice : p ∨ q :=
sorry

end problem_correct_choice_l9_9379


namespace vote_count_l9_9088

theorem vote_count (x : ℕ) (hlikes : 0.75 * x = likes) (hdislikes : 0.25 * x = dislikes) (hnet : likes - dislikes = 140) : x = 280 :=
sorry

end vote_count_l9_9088


namespace tan_20_add_4sin_20_eq_sqrt3_l9_9090

theorem tan_20_add_4sin_20_eq_sqrt3 : Real.tan (20 * Real.pi / 180) + 4 * Real.sin (20 * Real.pi / 180) = Real.sqrt 3 := 
by
  sorry

end tan_20_add_4sin_20_eq_sqrt3_l9_9090


namespace complex_power_l9_9686

def cos_30 := Float.sqrt 3 / 2
def sin_30 := 1 / 2

theorem complex_power (i : ℂ) : 
  (3 * (⟨cos_30, sin_30⟩ : ℂ)) ^ 4 = -40.5 + 40.5 * i * Float.sqrt 3 := 
  sorry

end complex_power_l9_9686


namespace simplify_trig_expression_evaluate_trig_expression_l9_9604

-- For the first problem
theorem simplify_trig_expression (α : ℝ) :
  (tan (π + α) * cos (2 * π + α) * sin (α - π / 2)) / (cos (-α - 3 * π) * sin (-3 * π - α)) = 1 :=
by sorry

-- For the second problem
theorem evaluate_trig_expression (α : ℝ) (h : tan α = 1 / 2) :
  2 * sin α ^ 2 - sin α * cos α + cos α ^ 2 = 4 / 5 :=
by sorry

end simplify_trig_expression_evaluate_trig_expression_l9_9604


namespace complex_power_l9_9689

def cos_30 := Float.sqrt 3 / 2
def sin_30 := 1 / 2

theorem complex_power (i : ℂ) : 
  (3 * (⟨cos_30, sin_30⟩ : ℂ)) ^ 4 = -40.5 + 40.5 * i * Float.sqrt 3 := 
  sorry

end complex_power_l9_9689


namespace angle_between_vectors_is_90_degrees_l9_9492

def vector_a : ℝ × ℝ × ℝ := (2, -3, -4)

def vector_b : ℝ × ℝ × ℝ := (Real.sqrt 3, 5, -2)

def vector_c : ℝ × ℝ × ℝ := (8, -7, 19)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def vector_scalar_mul (k : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (k * v.1, k * v.2, k * v.3)

def vector_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

def new_vector : ℝ × ℝ × ℝ :=
  let a_b := dot_product vector_a vector_b
  let a_c := dot_product vector_a vector_c
  vector_sub (vector_scalar_mul a_b vector_c) (vector_scalar_mul a_c vector_b)

def angle_is_90_degrees : Prop := 
  dot_product vector_a new_vector = 0

theorem angle_between_vectors_is_90_degrees : angle_is_90_degrees :=
by
  sorry

end angle_between_vectors_is_90_degrees_l9_9492


namespace smallest_prime_after_six_consecutive_nonprimes_l9_9179

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def nonprimes_between (a b : ℕ) : Prop :=
  ∀ n : ℕ, a < n ∧ n < b → ¬ is_prime n

def find_first_nonprime_gap : ℕ :=
  if H : ∃ a b : ℕ, b = a + 7 ∧ nonprimes_between a b ∧ is_prime (b + 1) then
    Classical.choose H
  else
    0

theorem smallest_prime_after_six_consecutive_nonprimes : find_first_nonprime_gap = 97 := by
  sorry

end smallest_prime_after_six_consecutive_nonprimes_l9_9179


namespace smallest_positive_real_d_l9_9720

theorem smallest_positive_real_d (d : ℝ) : 
  (∀ x y : ℝ, x >= y^2 → sqrt x + d * |y - x| >= 2 * |y|) ↔ d >= 1 :=
sorry

end smallest_positive_real_d_l9_9720


namespace problem1_problem2_l9_9899

-- Define the subset A of {1, 2, ..., 2006} with 1004 elements
variable (A : Set ℕ) (hA : ∀ x ∈ A, x ∈ Finset.range (2006 + 1)) (hA_card : Finset.card A = 1004)

-- Problem 1: Prove there exist 3 distinct numbers a, b, c in A such that gcd(a, b) = 1 and gcd(a, b) | c
theorem problem1 : ∃ (a b c : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ Nat.gcd a b = 1 ∧ (Nat.gcd a b ∣ c) := sorry

-- Problem 2: Prove there exist 3 distinct numbers a, b, c in A such that gcd(a, b) ≠ 1 and gcd(a, b) does not divide c
theorem problem2 : ∃ (a b c : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ Nat.gcd a b ≠ 1 ∧ ¬ (Nat.gcd a b ∣ c) := sorry

end problem1_problem2_l9_9899


namespace trapezoid_dot_product_l9_9097

-- Definitions for the given problem
variable (AB CD : ℝ)
variable (perpendicular_diagonals : Prop)

-- Define the bases of the trapezoid and the condition of perpendicular diagonals
def is_trapezoid_with_perpendicular_diagonals := 
  AB = 41 ∧ CD = 24 ∧ perpendicular_diagonals

-- Statement of the problem to prove the dot product
theorem trapezoid_dot_product 
  (h : is_trapezoid_with_perpendicular_diagonals AB CD True)
  (a b : ℝ) 
  (h_length : a^2 + b^2 = 41^2) : 
  (λ a b, 24 * (a^2 + b^2) / 41) a b = 984 := 
by
  sorry

end trapezoid_dot_product_l9_9097


namespace external_radius_increase_l9_9156

theorem external_radius_increase 
  (C_in1 C_in2 : ℝ) 
  (h_C_in1 : C_in1 = 30) 
  (h_C_in2 : C_in2 = 40) 
  (thickness : ℝ) 
  (h_thickness : thickness = 1) 
  : let r_in1 := C_in1 / (2 * π),
        r_ex1 := r_in1 + thickness,
        r_in2 := C_in2 / (2 * π),
        r_ex2 := r_in2 + thickness,
        Δr_ex := r_ex2 - r_ex1
    in 
    Δr_ex = 5 / π := 
by 
  sorry

#eval external_radius_increase 30 40 1 (by rfl) (by rfl) (by rfl)

end external_radius_increase_l9_9156


namespace avg_rate_of_change_reciprocal_l9_9096

def avg_rate_of_change (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  (f b - f a) / (b - a)

theorem avg_rate_of_change_reciprocal :
  avg_rate_of_change (λ x => 1/x) 1 2 = -1/2 :=
by {
  sorry
}

end avg_rate_of_change_reciprocal_l9_9096


namespace Rebecca_tips_calculation_l9_9081

def price_haircut : ℤ := 30
def price_perm : ℤ := 40
def price_dye_job : ℤ := 60
def cost_hair_dye_box : ℤ := 10
def num_haircuts : ℕ := 4
def num_perms : ℕ := 1
def num_dye_jobs : ℕ := 2
def total_end_day : ℤ := 310

noncomputable def total_service_earnings : ℤ := 
  num_haircuts * price_haircut + num_perms * price_perm + num_dye_jobs * price_dye_job

noncomputable def total_hair_dye_cost : ℤ := 
  num_dye_jobs * cost_hair_dye_box

noncomputable def earnings_after_cost : ℤ := 
  total_service_earnings - total_hair_dye_cost

noncomputable def tips : ℤ := 
  total_end_day - earnings_after_cost

theorem Rebecca_tips_calculation : tips = 50 := by
  sorry

end Rebecca_tips_calculation_l9_9081


namespace basketball_handshakes_l9_9653

theorem basketball_handshakes
  (team1_size team2_size refs players : ℕ)
  (handshakes_between_teams : team1_size * team2_size = 36)
  (handshakes_with_refs : players * refs = 36)
  (total_number_of_players : players = 12)
  (number_of_refs : refs = 3)
  (team1_size_def : team1_size = 6)
  (team2_size_def : team2_size = 6) :
  team1_size * team2_size + players * refs = 72 :=
by
  rw [team1_size_def, team2_size_def, total_number_of_players, number_of_refs, handshakes_between_teams, handshakes_with_refs]
  exact rfl

end basketball_handshakes_l9_9653


namespace rationalize_denominator_l9_9525

noncomputable def X : ℕ := 25
noncomputable def Y : ℕ := 15
noncomputable def Z : ℕ := 9
noncomputable def W : ℕ := 2

theorem rationalize_denominator :
  let a := (1 / ((5 : ℝ)^(1/3) - (3 : ℝ)^(1/3)))
  in (X + Y + Z + W = 51) ∧
    (a = (25 : ℝ)^(1/3) + (15 : ℝ)^(1/3) + (9 : ℝ)^(1/3)) / (2 : ℝ) :=
begin
  sorry -- Proof omitted
end

end rationalize_denominator_l9_9525


namespace smallest_prime_after_six_nonprimes_l9_9189

open Nat

theorem smallest_prime_after_six_nonprimes : 
  ∃ p : ℕ, prime p ∧ p > 0 ∧
  (∃ n : ℕ, ¬prime n ∧ ¬prime (n+1) ∧ ¬prime (n+2) ∧ ¬prime (n+3) ∧ ¬prime (n+4) ∧ ¬prime (n+5) ∧ prime (n+6) ∧ n+6 = p) ∧ 
  p = 89 :=
sorry

end smallest_prime_after_six_nonprimes_l9_9189


namespace sum_of_inscribed_angles_l9_9099

theorem sum_of_inscribed_angles (x y : ℝ) 
  (h1 : x = (3 * 360 / 16) / 2) 
  (h2 : y = (5 * 360 / 16) / 2) : 
  x + y = 90 :=
by
  -- Definitions based on the conditions given in the problem
  rw [h1, h2],
  -- Simplifications for each inscribed angle
  norm_num,
  sorry -- Skipping additional simplifications and steps because proof is not required

end sum_of_inscribed_angles_l9_9099


namespace at_least_12_boxes_l9_9652

theorem at_least_12_boxes (extra_boxes : Nat) : 
  let total_boxes := 12 + extra_boxes
  extra_boxes ≥ 0 → total_boxes ≥ 12 :=
by
  intros
  sorry

end at_least_12_boxes_l9_9652


namespace marathon_distance_l9_9036

theorem marathon_distance (marathons : ℕ) (miles_per_marathon : ℕ) (extra_yards_per_marathon : ℕ) (yards_per_mile : ℕ) (total_miles_run : ℕ) (total_yards_run : ℕ) (remaining_yards : ℕ) :
  marathons = 15 →
  miles_per_marathon = 26 →
  extra_yards_per_marathon = 385 →
  yards_per_mile = 1760 →
  total_miles_run = (marathons * miles_per_marathon + extra_yards_per_marathon * marathons / yards_per_mile) →
  total_yards_run = (marathons * (miles_per_marathon * yards_per_mile + extra_yards_per_marathon)) →
  remaining_yards = total_yards_run - (total_miles_run * yards_per_mile) →
  0 ≤ remaining_yards ∧ remaining_yards < yards_per_mile →
  remaining_yards = 1500 :=
by
  intros
  sorry

end marathon_distance_l9_9036


namespace sum_of_digits_of_special_number_l9_9239

-- Define what it means for a number to be palindromic
def is_palindromic (n : ℕ) : Prop :=
  let s := (Nat.digits 10 n).reverse in
  Nat.digits 10 n = s

-- Define the main problem
theorem sum_of_digits_of_special_number : 
  ∃ (N : ℕ), 100 ≤ N ∧ N < 1000 ∧ (∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ is_palindromic a ∧ is_palindromic b ∧ is_palindromic c ∧ a + b + c = N) ∧ ¬ is_palindromic N ∧ (N.digits.sum = 2) :=
by 
  sorry

end sum_of_digits_of_special_number_l9_9239


namespace rotated_vector_l9_9644

def vector_initial : ℝ×ℝ×ℝ := (2, 1, 1)

def vector_result : ℝ×ℝ×ℝ := (-real.sqrt (6 / 11), 3 * real.sqrt (6 / 11), -real.sqrt (6 / 11))

def is_orthogonal (u v : ℝ×ℝ×ℝ) : Prop := 
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3 = 0

def magnitude (v : ℝ×ℝ×ℝ) : ℝ := 
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

theorem rotated_vector :
  let u := vector_initial in
  let v := vector_result in
  magnitude u = real.sqrt 6 ∧ 
  magnitude v = real.sqrt 6 ∧ 
  is_orthogonal u v ∧ 
  u.2 > 0 ∧ v.2 > 0 →
  v = (-real.sqrt (6 / 11), 3 * real.sqrt (6 / 11), -real.sqrt (6 / 11)) :=
by
  intros
  sorry

end rotated_vector_l9_9644


namespace train_speed_is_correct_l9_9251

def train (train_length bridge_length : ℝ) (time : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  total_distance / time

theorem train_speed_is_correct
  (train_length bridge_length time : ℝ) 
  (h_train_length : train_length = 140) 
  (h_bridge_length : bridge_length = 235) 
  (h_time : time = 30) : 
  let speed_m_s := train train_length bridge_length time in
  let conversion_factor := 3.6 in
  speed_m_s * conversion_factor = 45 := 
by
  sorry

end train_speed_is_correct_l9_9251


namespace max_sum_free_subset_l9_9584

def is_sum_free (A : Set ℕ) : Prop :=
  ∀ a b c : ℕ, a ∈ A → b ∈ A → c ∈ A → a + b ≠ c

theorem max_sum_free_subset (n : ℕ) (A : Set ℕ)
  (hA : A ⊆ {k : ℕ | 1 ≤ k ∧ k < 2 * n}) :
  is_sum_free A → A.size ≤ n := sorry

end max_sum_free_subset_l9_9584


namespace A_inter_B_eq_A_l9_9381

def A := {x : ℝ | 0 < x ∧ x ≤ 2}
def B := {x : ℝ | x ≤ 3}

theorem A_inter_B_eq_A : A ∩ B = A := 
by 
  sorry 

end A_inter_B_eq_A_l9_9381


namespace count_cells_with_at_least_three_rectangles_l9_9726

theorem count_cells_with_at_least_three_rectangles :
  let is_endpoint (i j : ℕ) : Prop :=
    (16 ≤ i ∧ i ≤ 50) ∨ (1 ≤ i ∧ i ≤ 35) ∨ (16 ≤ j ∧ j ≤ 50) ∨ (1 ≤ j ∧ j ≤ 35)
  let count_cells (cond : ℕ → ℕ → Prop) : ℕ :=
    ((finset.range 50).product (finset.range 50)).count (λ ⟨i, j⟩, cond i.succ j.succ)
  count_cells (λ i j, (16 ≤ i ∧ i ≤ 50) ∨ (1 ≤ i ∧ i ≤ 35) ∨ (16 ≤ j ∧ j ≤ 50) ∨ (1 ≤ j ∧ j ≤ 35)) ≥ 1600 := 
sorry

end count_cells_with_at_least_three_rectangles_l9_9726


namespace return_speed_37_5_l9_9609

def avg_speed (d : ℝ) (v1 v2 : ℝ) : ℝ :=
  2 * d / (d / v1 + d / v2)

theorem return_speed_37_5 :
  let d := 150
  let v1 := 75
  let avg := 50
  ∃ r, avg_speed d v1 r = avg ∧ r = 37.5 :=
by
  let d := 150
  let v1 := 75
  let avg := 50
  use 37.5
  have h1 : 2 * d = 300 := by norm_num
  have h2 : d / v1 = 2 := by norm_num
  have h3 : d / 37.5 = 4 := by norm_num
  have h_avg : avg_speed d v1 37.5 = 50 :=
    by simp [avg_speed, h1, h2, h3, div_eq_inv_mul]
  tauto

end return_speed_37_5_l9_9609


namespace smallest_prime_after_six_consecutive_nonprimes_l9_9176

-- Define what it means to be prime
def is_prime (n : ℕ) : Prop :=
  nat.prime n

-- Define the sequence of six consecutive nonprime numbers
def consecutive_nonprime_sequence (a : ℕ) : Prop :=
  ¬ is_prime a ∧ ¬ is_prime (a + 1) ∧ ¬ is_prime (a + 2) ∧
  ¬ is_prime (a + 3) ∧ ¬ is_prime (a + 4) ∧ ¬ is_prime (a + 5)

-- Define the condition that there is a smallest prime number greater than a sequence of six consecutive nonprime numbers
theorem smallest_prime_after_six_consecutive_nonprimes :
  ∃ p, is_prime p ∧ p > 96 ∧ consecutive_nonprime_sequence 90 :=
sorry

end smallest_prime_after_six_consecutive_nonprimes_l9_9176


namespace ryan_lost_initially_l9_9526

-- Define the number of leaves initially collected
def initial_leaves : ℤ := 89

-- Define the number of leaves broken afterwards
def broken_leaves : ℤ := 43

-- Define the number of leaves left in the collection
def remaining_leaves : ℤ := 22

-- Define the lost leaves
def lost_leaves (L : ℤ) : Prop :=
  initial_leaves - L - broken_leaves = remaining_leaves

theorem ryan_lost_initially : ∃ L : ℤ, lost_leaves L ∧ L = 24 :=
by
  sorry

end ryan_lost_initially_l9_9526


namespace number_of_boxes_l9_9921

theorem number_of_boxes (total_eggs : ℕ) (eggs_per_box : ℕ) (boxes : ℕ) : 
  total_eggs = 21 → eggs_per_box = 7 → boxes = total_eggs / eggs_per_box → boxes = 3 :=
by
  intros h_total_eggs h_eggs_per_box h_boxes
  rw [h_total_eggs, h_eggs_per_box] at h_boxes
  exact h_boxes

end number_of_boxes_l9_9921


namespace number_of_ways_to_select_60_l9_9314

-- Define the conditions: five volunteers, and choose 2 people each day such that one serves both days
def select_ways (n : ℕ) (r : ℕ) : ℕ :=
  Nat.choose n r

theorem number_of_ways_to_select_60 : select_ways 5 1 * (select_ways 4 1 * select_ways 3 1) = 60 := by
  sorry

end number_of_ways_to_select_60_l9_9314


namespace cos_sin_double_angle_l9_9817

theorem cos_sin_double_angle (θ : ℝ) (h : real.cos θ = 3 / 5) :
  real.cos (2 * θ) = -7 / 25 ∧ real.sin (2 * θ) = 24 / 25 :=
by
  sorry

end cos_sin_double_angle_l9_9817


namespace new_average_age_l9_9545

theorem new_average_age (n_initial : ℕ) (avg_age_initial : ℕ) (n_new : ℕ) (avg_age_new : ℕ)
    (h1 : n_initial = 12) 
    (h2 : avg_age_initial = 16)
    (h3 : n_new = 12)
    (h4 : avg_age_new = 15) :
    (n_initial * avg_age_initial + n_new * avg_age_new) / (n_initial + n_new) = 15.5 := by
  sorry


end new_average_age_l9_9545


namespace A_inter_B_eq_A_l9_9382

def A := {x : ℝ | 0 < x ∧ x ≤ 2}
def B := {x : ℝ | x ≤ 3}

theorem A_inter_B_eq_A : A ∩ B = A := 
by 
  sorry 

end A_inter_B_eq_A_l9_9382


namespace seven_elves_milk_distribution_l9_9528

theorem seven_elves_milk_distribution:
  ∃ (y : ℕ → ℚ), 
    (y 1 = 6/7 ∧ y 2 = 5/7 ∧ y 3 = 4/7 ∧ y 4 = 3/7 ∧ y 5 = 2/7 ∧ y 6 = 1/7 ∧ y 7 = 0) ∧
    (∑ i in finset.range 7, y (i + 1)) = 3 ∧
    (∀ i, y (i + 1) = (y (if i = 0 then 7 else i) + y (if i = 1 then 7 else i - 1) + y (if i = 2 then 7 else i - 2) + y (if i = 3 then 7 else i - 3) + y (if i = 4 then 7 else i - 4) + y (if i = 5 then 7 else i - 5)) / 6) :=
sorry

end seven_elves_milk_distribution_l9_9528


namespace third_competitor_eats_l9_9292

-- Define the conditions based on the problem description
def first_competitor_hot_dogs : ℕ := 12
def second_competitor_hot_dogs := 2 * first_competitor_hot_dogs
def third_competitor_hot_dogs := second_competitor_hot_dogs - (second_competitor_hot_dogs / 4)

-- The theorem we need to prove
theorem third_competitor_eats :
  third_competitor_hot_dogs = 18 := by
  sorry

end third_competitor_eats_l9_9292


namespace length_of_PQ_equals_l9_9871

noncomputable def side_length : ℝ := 10

def area_of_square : ℝ := side_length ^ 2

def equal_area : ℝ := area_of_square / 3

def PQ_length : ℝ := 20 / 3

theorem length_of_PQ_equals :
  ∀ (square : Type) (side_length : ℝ) (PQ_length : ℝ),
  (square = ⟨side_length, side_length, \lambda x, y, z, w => x=y ∨ x=z ∨ x=w ∨ y=z ∨ y=w ∨ z=w⟩) →
  (side_length = 10) →
  (area_of_square = side_length ^ 2) →
  (S₁ + S₂ + S₃ = area_of_square) →
  (equal_area = area_of_square / 3) →
  (PQ_length = 20 / 3) →
  PQ_length = 20 / 3 :=
begin
  intros,
  sorry, -- Proof to be filled in later
end

end length_of_PQ_equals_l9_9871


namespace smallest_prime_after_six_consecutive_nonprimes_l9_9184

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def nonprimes_between (a b : ℕ) : Prop :=
  ∀ n : ℕ, a < n ∧ n < b → ¬ is_prime n

def find_first_nonprime_gap : ℕ :=
  if H : ∃ a b : ℕ, b = a + 7 ∧ nonprimes_between a b ∧ is_prime (b + 1) then
    Classical.choose H
  else
    0

theorem smallest_prime_after_six_consecutive_nonprimes : find_first_nonprime_gap = 97 := by
  sorry

end smallest_prime_after_six_consecutive_nonprimes_l9_9184


namespace Amanda_cousins_ages_sum_l9_9647

theorem Amanda_cousins_ages_sum :
  ∃ (a1 a2 a3 a4 a5 : ℕ) (ages : List ℕ),
  (ages = [a1, a2, a3, a4, a5]) ∧
  (List.length ages = 5) ∧
  (List.mean ages = 10) ∧
  (List.median ages = 9) ∧
  (a1 + a2 + a3 + a4 + a5 = 50) ∧
  (a3 = 9) ∧
  (a1 + a5 = 23) :=
by
  sorry

end Amanda_cousins_ages_sum_l9_9647


namespace sum_of_digits_div_by_11_in_consecutive_39_l9_9072

-- Define the sum of digits function for natural numbers.
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The main theorem statement.
theorem sum_of_digits_div_by_11_in_consecutive_39 :
  ∀ (N : ℕ), ∃ k : ℕ, k < 39 ∧ (sum_of_digits (N + k)) % 11 = 0 :=
by sorry

end sum_of_digits_div_by_11_in_consecutive_39_l9_9072


namespace smallest_sum_of_xy_l9_9364

theorem smallest_sum_of_xy (x y : ℕ) (h1 : x ≠ y) (h2 : 0 < x) (h3 : 0 < y) 
  (h4 : (1:ℚ)/x + (1:ℚ)/y = (1:ℚ)/15) : x + y = 64 :=
sorry

end smallest_sum_of_xy_l9_9364


namespace distance_from_P_to_AB_l9_9885

theorem distance_from_P_to_AB {A B C P : Type} (h : Real) (area_triangle : Real) 
(altitude : area_triangle = 10) 
(area_condition: ∃ (smaller_area : Real), smaller_area = (1 / 5) * area_triangle)  
: ∃ (d : Real), d = 10 - 2 * Real.sqrt(5) :=
by 
  sorry

end distance_from_P_to_AB_l9_9885


namespace geometric_sequence_s4_l9_9852

theorem geometric_sequence_s4
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (q : ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a (n + 1) = a n * q)
  (h3 : ∀ n : ℕ, S n = (∑ i in finset.range n, a (i + 1)))
  (h4 : S 5 = 5 * S 3 - 4) :
  S 4 = 15 :=
sorry

end geometric_sequence_s4_l9_9852


namespace maximum_value_with_conditions_l9_9906

noncomputable def max_value (a b c d e : ℝ) := c * (a + 3 * b + 4 * d + 8 * e)

theorem maximum_value_with_conditions :
  ∃ a b c d e : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
    a^2 + b^2 + c^2 + d^2 + e^2 = 100 ∧
    let N := max_value a b c d e in
    N + a + b + c + d + e = 16 + 150 * Real.sqrt 10 + 5 * Real.sqrt 2 :=
sorry

end maximum_value_with_conditions_l9_9906


namespace expected_value_of_monicas_winnings_l9_9061

def die_outcome (n : ℕ) : ℤ :=
  if n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 then n else if n = 1 ∨ n = 4 ∨ n = 6 ∨ n = 8 then 0 else -5

noncomputable def expected_winnings : ℚ :=
  (1/2 : ℚ) * 0 + (1/8 : ℚ) * 2 + (1/8 : ℚ) * 3 + (1/8 : ℚ) * 5 + (1/8 : ℚ) * 7 + (1/8 : ℚ) * (-5)

theorem expected_value_of_monicas_winnings : expected_winnings = 3/2 := by
  sorry

end expected_value_of_monicas_winnings_l9_9061


namespace exists_sum_of_digits_div_11_l9_9075

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_sum_of_digits_div_11 (H : Finset ℕ) (h₁ : H.card = 39) :
  ∃ (a : ℕ) (h : a ∈ H), sum_of_digits a % 11 = 0 :=
by
  sorry

end exists_sum_of_digits_div_11_l9_9075


namespace min_shift_symmetric_y_axis_l9_9106

theorem min_shift_symmetric_y_axis :
  ∃ (m : ℝ), m = 7 * Real.pi / 6 ∧ 
             (∀ x : ℝ, 2 * Real.cos (x + Real.pi / 3) = 2 * Real.cos (x + Real.pi / 3 + m)) ∧ 
             m > 0 :=
by
  sorry

end min_shift_symmetric_y_axis_l9_9106


namespace smallest_possible_value_l9_9355

theorem smallest_possible_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 / (x : ℝ)) + (1 / (y : ℝ)) = 1 / 15) : x + y = 64 :=
sorry

end smallest_possible_value_l9_9355


namespace find_percentage_l9_9607

theorem find_percentage (P : ℝ) : 
  (∀ x : ℝ, x = 0.40 * 800 → x = P / 100 * 650 + 190) → P = 20 := 
by
  intro h
  sorry

end find_percentage_l9_9607


namespace fence_calculation_l9_9632

def width : ℕ := 4
def length : ℕ := 2 * width - 1
def fence_needed : ℕ := 2 * length + 2 * width

theorem fence_calculation : fence_needed = 22 := by
  sorry

end fence_calculation_l9_9632


namespace complex_fourth_power_l9_9693

-- Definitions related to the problem conditions
def trig_identity (θ : ℝ) : ℂ := complex.of_real (cos θ) + complex.I * complex.of_real (sin θ)
def z : ℂ := 3 * trig_identity (30 * real.pi / 180)
def result := (81 : ℝ) * (-1/2 + complex.I * (real.sqrt 3 / 2))

-- Statement of the problem
theorem complex_fourth_power : z^4 = result := by
  sorry

end complex_fourth_power_l9_9693


namespace compute_B_pow_101_l9_9482

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

theorem compute_B_pow_101 : B^101 = ![![0, 0, 1], ![1, 0, 0], ![0, 1, 0]] :=
by sorry

end compute_B_pow_101_l9_9482


namespace S₄_eq_15_l9_9844

-- Definitions based on the given conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum a

def sequence_condition (a : ℕ → ℝ) : Prop :=
  is_geometric_sequence a ∧ a 1 = 1 ∧ sum_of_first_n_terms a 5 = 5 * sum_of_first_n_terms a 3 - 4

theorem S₄_eq_15 (a : ℕ → ℝ) (q : ℝ) :
  sequence_condition a →
  (∀ n, a n = 1 * q ^ (n-1)) → 
  sum_of_first_n_terms a 4 = 15 :=
sorry

end S₄_eq_15_l9_9844


namespace sfl_entrances_l9_9192

theorem sfl_entrances (people_per_entrance total_people entrances : ℕ) 
  (h1: people_per_entrance = 283) 
  (h2: total_people = 1415) 
  (h3: total_people = people_per_entrance * entrances) 
  : entrances = 5 := 
  by 
  rw [h1, h2] at h3
  sorry

end sfl_entrances_l9_9192


namespace first_ant_arrives_first_l9_9141

noncomputable def time_crawling (d v : ℝ) : ℝ := d / v

noncomputable def time_riding_caterpillar (d v : ℝ) : ℝ := (d / 2) / (v / 2)

noncomputable def time_riding_grasshopper (d v : ℝ) : ℝ := (d / 2) / (10 * v)

noncomputable def time_ant1 (d v : ℝ) : ℝ := time_crawling d v

noncomputable def time_ant2 (d v : ℝ) : ℝ := time_riding_caterpillar d v + time_riding_grasshopper d v

theorem first_ant_arrives_first (d v : ℝ) (h_v_pos : 0 < v): time_ant1 d v < time_ant2 d v := by
  -- provide the justification for the theorem here
  sorry

end first_ant_arrives_first_l9_9141


namespace sum_of_ages_l9_9151

-- Define the variables for Viggo and his younger brother's ages
variables (v y : ℕ)

-- Condition: When Viggo's younger brother was 2, Viggo's age was 10 years more than twice his brother's age
def condition1 (v y : ℕ) := (y = 2 → v = 2 * y + 10)

-- Condition: Viggo's younger brother is currently 10 years old
def condition2 (y_current : ℕ) := y_current = 10

-- Define the current age of Viggo given the conditions
def viggo_current_age (v y y_current : ℕ) := v + (y_current - y)

-- Prove that the sum of their ages is 32
theorem sum_of_ages
  (v y y_current : ℕ)
  (h1 : condition1 v y)
  (h2 : condition2 y_current) :
  viggo_current_age v y y_current + y_current = 32 :=
by
  -- Apply sorry to skip the proof
  sorry

end sum_of_ages_l9_9151


namespace total_groups_correct_l9_9944

/-- Rebecca's grouping problem -/
def groups_created 
  (eggs : ℕ) (bananas : ℕ) (marbles : ℕ) (apples : ℕ) (oranges : ℕ)
  (eggs_per_group : ℕ) (bananas_per_group : ℕ) (marbles_per_group : ℕ)
  (apples_per_group : ℕ) (oranges_per_group : ℕ) : ℕ :=
(eggs / eggs_per_group) + (bananas / bananas_per_group) + (marbles / marbles_per_group) + (apples / apples_per_group) + (oranges / oranges_per_group)

theorem total_groups_correct :
  groups_created 75 99 48 (6 * 12) (0.5 * 12).toNat 4 5 6 12 2 = 54 :=
by
  sorry

end total_groups_correct_l9_9944


namespace convex_quad_area_inequality_l9_9337

variables {A B C D O M N : Type*}
noncomputable def area (P Q R : Type*) : ℝ := sorry -- Define the area function (details omitted)

-- Conditions
variable (h1 : convex_quadrilateral A B C D)
variable (h2 : intersect_diagonals A C B D O)
variable (h3 : intersect_line O A B M)
variable (h4 : intersect_line O C D N)
variable (h5 : area O M B > area O N D)
variable (h6 : area O C N > area O A M)

-- To Prove
theorem convex_quad_area_inequality :
  area O A M + area O B C + area O N D > area O A D + area O B M + area O C N :=
sorry

end convex_quad_area_inequality_l9_9337


namespace geom_seq_sum_4_l9_9859

noncomputable def geom_seq (n : ℕ) (q : ℝ) (a₁ : ℝ) : ℝ :=
  a₁ * q^(n - 1)

noncomputable def sum_geom_seq (n : ℕ) (q : ℝ) (a₁ : ℝ) : ℝ :=
  if q = 1 then a₁ * n else (a₁ * (1 - q^n) / (1 - q))

theorem geom_seq_sum_4 {q : ℝ} (hq : q > 0) (hq1 : q ≠ 1) :
  let a₁ := 1 in
  let S5 := sum_geom_seq 5 q a₁ in
  let S3 := sum_geom_seq 3 q a₁ in
  S5 = 5 * S3 - 4 →
  sum_geom_seq 4 q a₁ = 15 :=
by
  sorry

end geom_seq_sum_4_l9_9859


namespace find_paintings_l9_9062

noncomputable def cost_painting (P : ℕ) : ℝ := 40 * P
noncomputable def cost_toy : ℝ := 20 * 8
noncomputable def total_cost (P : ℕ) : ℝ := cost_painting P + cost_toy

noncomputable def sell_painting (P : ℕ) : ℝ := 36 * P
noncomputable def sell_toy : ℝ := 17 * 8
noncomputable def total_sell (P : ℕ) : ℝ := sell_painting P + sell_toy

noncomputable def total_loss (P : ℕ) : ℝ := total_cost P - total_sell P

theorem find_paintings : ∀ (P : ℕ), total_loss P = 64 → P = 10 :=
by
  intros P h
  sorry

end find_paintings_l9_9062


namespace alpha_beta_sum_eq_138_l9_9287

theorem alpha_beta_sum_eq_138 (α β : ℝ) :
  (∀ x : ℝ, x + 25 ≠ 0 → (x - α) / (x + β) = (x^2 - 64 * x + 975) / (x^2 + 99 * x - 2200)) →
  α + β = 138 :=
by {
  intro h,
  sorry
}

end alpha_beta_sum_eq_138_l9_9287


namespace tracy_popped_fraction_l9_9271

theorem tracy_popped_fraction :
  let brooke_balloons := 12 + 8,
      tracy_initial_balloons := 6 + 24,
      total_balloons := 35
  in ∃ (f : ℚ), brooke_balloons + (tracy_initial_balloons - tracy_initial_balloons * f) = total_balloons ∧ f = 1/2 :=
by
  sorry

end tracy_popped_fraction_l9_9271


namespace james_and_david_probability_l9_9033

noncomputable def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem james_and_david_probability :
  let total_workers := 22
  let chosen_workers := 4
  let j_and_d_chosen := 2
  (choose 20 2) / (choose 22 4) = (2 / 231) :=
by
  sorry

end james_and_david_probability_l9_9033


namespace resistance_between_opposite_vertices_of_cube_l9_9254

-- Define the parameters of the problem
def resistance_cube_edge : ℝ := 1

-- Define the function to calculate the equivalent resistance
noncomputable def equivalent_resistance_opposite_vertices (R : ℝ) : ℝ :=
  let R1 := R / 3
  let R2 := R / 6
  let R3 := R / 3
  R1 + R2 + R3

-- State the theorem to prove the resistance between two opposite vertices
theorem resistance_between_opposite_vertices_of_cube :
  equivalent_resistance_opposite_vertices resistance_cube_edge = 5 / 6 :=
by
  sorry

end resistance_between_opposite_vertices_of_cube_l9_9254


namespace find_m_equals_2000_l9_9020

theorem find_m_equals_2000 :
  ∃ m : ℝ, 1000 + (m - 1000) * 0.618 = 1618 ∧ m = 2000 :=
by
  use 2000
  split
  · norm_num
  · rfl

end find_m_equals_2000_l9_9020


namespace number_of_true_propositions_l9_9781

def is_surface_area_of_inscribed_sphere (S : ℝ) : Prop :=
  S = π

def proposition_1 (P : ℝ × ℝ × ℝ → Prop) (line_perpendicular_P : Prop → Prop) : Prop :=
  ∀ P, line_perpendicular_P (P (AD₁))

def proposition_2 (AM_intersects_CC1 : Prop → Prop) : Prop :=
  AM_intersects_CC1 (mid_pt (C₁D₁))

def proposition_3 (P : ℝ × ℝ × ℝ → Prop) (tetrahedron_volume_constant : Prop → Prop) : Prop :=
  tetrahedron_volume_constant (P (AD₁))

def proposition_4 (area_cross_section : ℝ) : Prop :=
  area_cross_section = (√6) / 2

def proposition_5 (minimum_AP_PD₁_value : ℝ) : Prop :=
  minimum_AP_PD₁_value = √(2 + √2)

theorem number_of_true_propositions 
  (h1 : is_surface_area_of_inscribed_sphere π)
  (h2 : proposition_1 (λ _, True) (λ _, True))
  (h3 : ¬ proposition_2 (λ _, False))
  (h4 : proposition_3 (λ _, True) (λ _, True))
  (h5 : proposition_4 ((√6) / 2))
  (h6 : proposition_5 √(2 + √2)) : 
  ∃ n, n = 4 :=
sorry

end number_of_true_propositions_l9_9781


namespace problem_solution_l9_9385

variable {A B : Set ℝ}

def definition_A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def definition_B : Set ℝ := {x | x ≤ 3}

theorem problem_solution : definition_A ∩ definition_B = definition_A :=
by
  sorry

end problem_solution_l9_9385


namespace rich_walked_distance_l9_9086

def total_distance_walked (d1 d2 : ℕ) := 
  d1 + d2 + 2 * (d1 + d2) + (d1 + d2 + 2 * (d1 + d2)) / 2

def distance_to_intersection (d1 d2 : ℕ) := 
  2 * (d1 + d2)

def distance_to_end_route (d1 d2 : ℕ) := 
  (d1 + d2 + distance_to_intersection d1 d2) / 2

def total_distance_one_way (d1 d2 : ℕ) := 
  (d1 + d2) + (distance_to_intersection d1 d2) + (distance_to_end_route d1 d2)

theorem rich_walked_distance
  (d1 : ℕ := 20)
  (d2 : ℕ := 200) :
  2 * total_distance_one_way d1 d2 = 1980 :=
by
  simp [total_distance_one_way, distance_to_intersection, distance_to_end_route, total_distance_walked]
  sorry

end rich_walked_distance_l9_9086


namespace kelly_chris_boxes_ratio_l9_9203

-- Define the number of boxes packed by Kelly and Chris
variables (total_boxes kelly_boxes chris_boxes : ℕ)

-- Assuming Chris packed 60% of the total boxes
def chris_packed_fraction : Prop := chris_boxes = (6 * total_boxes) / 10

-- Define the ratio of boxes packed by Kelly to the boxes packed by Chris
def ratio_kelly_to_chris : Prop := kelly_boxes * 3 = chris_boxes * 2

-- The result we want to prove: the ratio of Kelly's boxes to Chris's boxes is 2:3
theorem kelly_chris_boxes_ratio (h : chris_packed_fraction) : ratio_kelly_to_chris :=
sorry

end kelly_chris_boxes_ratio_l9_9203


namespace number_of_correct_propositions_l9_9317

def double_factorial : ℕ → ℕ
| 0           := 1
| 1           := 1
| n           := n * double_factorial (n-2)

theorem number_of_correct_propositions:
  (double_factorial 2003 * double_factorial 2002 = fact 2003) ∧
  (double_factorial 2002 = (2^1001) * fact 1001) ∧
  (double_factorial 2002 % 10 = 0) ∧
  (double_factorial 2003 % 10 = 5)
:=
sorry

end number_of_correct_propositions_l9_9317


namespace reach_heights_correct_l9_9101

noncomputable def times_reach_height (y : ℝ) : ℝ × ℝ :=
  let a : ℝ := 16
  let b : ℝ := -64
  let c : ℝ := y
  ((-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a), (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a))

theorem reach_heights_correct :
  times_reach_height 1 25 = (3.6, 0.4) ∧
  times_reach_height 1 49 = (3.0, 1.0) := by
  sorry

end reach_heights_correct_l9_9101


namespace problem1_l9_9052

variable {Point Circle : Type} [something Point Circle]
variable A B C D T₁ T₂ : Circle
variable O T : Point

# conditions
variables 
  (ABCD : rectangle)
  (arcT : arc passing through A C)
  (circleT₁ : tangent T to AD DC)
  (circleT₂ : tangent T to AB BC)
  (radius₁ : radius T₁)
  (radius₂ : radius T₂)
  (incircle_radius : radius of triangle ABC)

# properties
theorem problem1 (ABCD T₁ T₂ : Circle)
  (H1 : T₁ and T₂ are within ABCD)
  (H2 : tangent T to AD DC AB BC)
  (H3 : r₁ + r₂ = 2r) 
  (H4 : parallel common tangent to AC with length (|AB - BC|)) :
  (
      r₁ + r₂ = 2r 
  ∧ 
      (parallel common tangentAC) ∧ its_length = abs(.AB - B.C)
  ) :=
sorry

end problem1_l9_9052


namespace proof_g_properties_l9_9951

noncomputable def g (x : ℝ) : ℝ := sorry

axiom g_defined : ∀ x : ℝ, ∃ y : ℝ, g x = y
axiom g_positive : ∀ x : ℝ, g x > 0
axiom g_functional : ∀ a b : ℝ, g a * g b = g (a * b)

theorem proof_g_properties :
  (g 1 = 1) ∧ (∀ a, g (1 / a) = 1 / g a) :=
begin
  split,
  { sorry }, -- proof for g(1) = 1
  { intro a,
    sorry } -- proof for g(1/a) = 1/g(a)
end

end proof_g_properties_l9_9951


namespace hockey_team_selection_l9_9621

theorem hockey_team_selection :
  let total_players := 18
  let quadruplets := 4
  let starters := 7
  quad_choice_0 := @nat.choose (total_players - quadruplets) starters
  quad_choice_1 := quadruplets * @nat.choose (total_players - quadruplets) (starters - 1)
  quad_choice_2 := @nat.choose quadruplets 2 * @nat.choose (total_players - quadruplets) (starters - 2)
  quad_choice_0 + quad_choice_1 + quad_choice_2 = 27456
:=
by
  let total_players := 18
  let quadruplets := 4
  let starters := 7
  let quad_choice_0 := @nat.choose (total_players - quadruplets) starters
  let quad_choice_1 := quadruplets * @nat.choose (total_players - quadruplets) (starters - 1)
  let quad_choice_2 := @nat.choose quadruplets 2 * @nat.choose (total_players - quadruplets) (starters - 2)
  have := quad_choice_0 + quad_choice_1 + quad_choice_2
  sorry

end hockey_team_selection_l9_9621


namespace slope_of_l3_l9_9509

open Set

/-- Line \( l_1 \) has the equation \( 4x - 3y = 2 \) and passes through \( A = (0, -2) \). 
    Line \( l_2 \) has the equation \( y = 2 \) and intersects line \( l_1 \) at point \( B \). 
    Line \( l_3 \) has a positive slope, goes through point \( A \), and intersects \( l_2 \) at point \( C \). 
    The area of \( \triangle ABC \) is \( 6 \). Find the slope of \( l_3 \). -/
theorem slope_of_l3 (x y : ℝ) (A B C : ℝ × ℝ)
  (hl1 : ∃ A, A = (0, -2) ∧ 4 * A.1 - 3 * A.2 = 2)
  (hl2 : ∃ B, B = l2 ∧ B.2 = 2)
  (hl3 : ∃ C, C.1 = B.1 + 3 ∧ C.2 = 2 ∨ C.1 = B.1 - 3 ∧ C.2 = 2)
  (hABC : 1 / 2 * (C.1 - B.1) * 4 = 6) :
  ∃ m : ℝ, m = 4 / 5 :=
by
  sorry

end slope_of_l3_l9_9509


namespace angle_in_isosceles_triangle_with_parallel_lines_l9_9457

theorem angle_in_isosceles_triangle_with_parallel_lines
    (L1 L2 : Type) [Hyperplane L1] [Hyperplane L2]
    (BA BC : Type) [Between BA BC]
    (h1 : parallel L1 L2)
    (h2 : BA = BC)
    (angle_DBC : ℝ)
    (h3 : angle_DBC = 70) :
    ∃ x, x = 35 :=
begin
  use 35,
  sorry
end

end angle_in_isosceles_triangle_with_parallel_lines_l9_9457


namespace smallest_x_plus_y_l9_9351

theorem smallest_x_plus_y (x y : ℕ) (h_xy_not_equal : x ≠ y) (h_condition : (1 : ℚ) / x + 1 / y = 1 / 15) :
  x + y = 64 :=
sorry

end smallest_x_plus_y_l9_9351


namespace geometric_sequence_s4_l9_9853

theorem geometric_sequence_s4
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (q : ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a (n + 1) = a n * q)
  (h3 : ∀ n : ℕ, S n = (∑ i in finset.range n, a (i + 1)))
  (h4 : S 5 = 5 * S 3 - 4) :
  S 4 = 15 :=
sorry

end geometric_sequence_s4_l9_9853


namespace simplify_exponential_expression_l9_9594

theorem simplify_exponential_expression :
  (3 * (-5)^2)^(3/4) = (75)^(3/4) := 
  sorry

end simplify_exponential_expression_l9_9594


namespace probability_physics_majors_consecutive_l9_9544

open Nat

theorem probability_physics_majors_consecutive :
  let total_people := 10
  let num_math := 5
  let num_physics := 3
  let num_bio := 2
  let total_permutations := Nat.factorial total_people
  let favorable_outcomes := 10 * (Nat.factorial num_physics)
  let probability := favorable_outcomes / total_permutations
  in probability = (1 / 12) := by
  sorry

end probability_physics_majors_consecutive_l9_9544


namespace smallest_sum_of_three_diff_numbers_l9_9709

theorem smallest_sum_of_three_diff_numbers : 
  ∀ (s : Set ℤ), s = {8, -7, 2, -4, 20} → ∃ a b c, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a + b + c = -9) :=
by
  sorry

end smallest_sum_of_three_diff_numbers_l9_9709


namespace exists_2016_consecutive_with_16_primes_l9_9722

noncomputable def S (n : ℕ) : ℕ :=
  Nat.countPrimesInRange n (n + 2015)

theorem exists_2016_consecutive_with_16_primes :
  ∃ n : ℕ, S(n) = 16 :=
begin
  sorry
end

end exists_2016_consecutive_with_16_primes_l9_9722


namespace students_on_bus_after_all_stops_l9_9031

-- Define the initial number of students getting on the bus at the first stop.
def students_first_stop : ℕ := 39

-- Define the number of students added at the second stop.
def students_second_stop_add : ℕ := 29

-- Define the number of students getting off at the second stop.
def students_second_stop_remove : ℕ := 12

-- Define the number of students added at the third stop.
def students_third_stop_add : ℕ := 35

-- Define the number of students getting off at the third stop.
def students_third_stop_remove : ℕ := 18

-- Calculating the expected number of students on the bus after all stops.
def total_students_expected : ℕ :=
  students_first_stop + students_second_stop_add - students_second_stop_remove +
  students_third_stop_add - students_third_stop_remove

-- The theorem stating the number of students on the bus after all stops.
theorem students_on_bus_after_all_stops : total_students_expected = 73 := by
  sorry

end students_on_bus_after_all_stops_l9_9031


namespace problem_l9_9388

def setA : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}
def setB : Set ℝ := {x : ℝ | x ≤ 3}

theorem problem : setA ∩ setB = setA := sorry

end problem_l9_9388


namespace sum_f_from_neg12_to_13_l9_9914

def f (x : ℝ) : ℝ := sqrt 3 / (3^x + sqrt 3)

theorem sum_f_from_neg12_to_13 :
  (∑ k in (finset.range 26).map (λ i, i - 12 : ℕ → ℤ), f k) = 13 :=
sorry

end sum_f_from_neg12_to_13_l9_9914


namespace probability_computation_l9_9592

-- Definitions of individual success probabilities
def probability_Xavier_solving_problem : ℚ := 1 / 4
def probability_Yvonne_solving_problem : ℚ := 2 / 3
def probability_William_solving_problem : ℚ := 7 / 10
def probability_Zelda_solving_problem : ℚ := 5 / 8
def probability_Zelda_notsolving_problem : ℚ := 1 - probability_Zelda_solving_problem

-- The target probability that only Xavier, Yvonne, and William, but not Zelda, will solve the problem
def target_probability : ℚ := (1 / 4) * (2 / 3) * (7 / 10) * (3 / 8)

-- The simplified form of the computed probability
def simplified_target_probability : ℚ := 7 / 160

-- Lean 4 statement to prove the equality of the computed and the target probabilities
theorem probability_computation :
  target_probability = simplified_target_probability := by
  sorry

end probability_computation_l9_9592


namespace smallest_prime_after_six_consecutive_nonprimes_l9_9174

-- Define what it means to be prime
def is_prime (n : ℕ) : Prop :=
  nat.prime n

-- Define the sequence of six consecutive nonprime numbers
def consecutive_nonprime_sequence (a : ℕ) : Prop :=
  ¬ is_prime a ∧ ¬ is_prime (a + 1) ∧ ¬ is_prime (a + 2) ∧
  ¬ is_prime (a + 3) ∧ ¬ is_prime (a + 4) ∧ ¬ is_prime (a + 5)

-- Define the condition that there is a smallest prime number greater than a sequence of six consecutive nonprime numbers
theorem smallest_prime_after_six_consecutive_nonprimes :
  ∃ p, is_prime p ∧ p > 96 ∧ consecutive_nonprime_sequence 90 :=
sorry

end smallest_prime_after_six_consecutive_nonprimes_l9_9174


namespace smallest_collected_l9_9196

noncomputable def Yoongi_collections : ℕ := 4
noncomputable def Jungkook_collections : ℕ := 6 / 3
noncomputable def Yuna_collections : ℕ := 5

theorem smallest_collected : min (min Yoongi_collections Jungkook_collections) Yuna_collections = 2 :=
by
  sorry

end smallest_collected_l9_9196


namespace find_k_l9_9560

theorem find_k (k : ℝ)
  (h1 : k ≠ 0)
  (h2 : ∃ A B : ℝ × ℝ, A ≠ B ∧ (A.1 + A.2) = (B.1 + B.2) / 2 ∧ (A.1^2 + A.2^2 - 6 * A.1 - 4 * A.2 + 9 = 0) ∧ (B.1^2 + B.2^2 - 6 * B.1 - 4 * B.2 + 9 = 0)
     ∧ dist A B = 2 * Real.sqrt 3)
  (h3 : ∀ x y : ℝ, y = k * x + 3 → (x^2 + y^2 - 6 * x - 4 * y + 9) = 0)
  : k = 1 := sorry

end find_k_l9_9560


namespace tangent_line_at_x1_min_value_h_l9_9917

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x^2 - Real.exp 1 * x - 2

noncomputable def h (a : ℝ) (x : ℝ) : ℝ := 
  let f_prime := λ x, Real.exp x - 2 * a * x - Real.exp 1
  f_prime x

theorem tangent_line_at_x1 (a : ℝ) (hx1 : a = 1) :
    let m := -2
    let b := -3
    ∀ x, (2 : ℝ) * x + (1 : ℝ) * ((m : ℝ) * (x - 1)) + (b : ℝ) = 0 :=
by
    intro x
    sorry

theorem min_value_h (a : ℝ) (x : ℝ) (hx0 : 0 ≤ x) (hx1 : x ≤ 1):
    (a ≤ (1/2) → h a 0 = 1 - Real.exp 1) ∧ 
    (a > (Real.exp 1)/2 → h a 1 = -2 * a) ∧ 
    ((1/2) < a ∧ a ≤ (Real.exp 1)/2 → h a (Real.log (2 * a)) = 2 * a - 2 * a * Real.log (2 * a) - Real.exp 1) :=
by
    intro a hx0 hx1
    split
    a -> ...
    a -> ...
    sorry

end tangent_line_at_x1_min_value_h_l9_9917


namespace intersection_eq_l9_9437

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem intersection_eq : M ∩ N = {0, 1} := by
  sorry

end intersection_eq_l9_9437


namespace sum_of_least_and_greatest_in_second_row_l9_9928

-- Define the grid size and the center position
def grid_size : ℕ := 15
def center : ℕ := (grid_size + 1) / 2

-- Define the position of the greatest and least numbers in the second row
def top_row : ℕ := 2
def least_in_second_row : ℕ := 157
def greatest_in_second_row : ℕ := 210

-- Define the tuples describing the structure of the Lean 4 statement
theorem sum_of_least_and_greatest_in_second_row :
  let n := grid_size * grid_size in
  let grid := sorry in
  let least := least_in_second_row in
  let greatest := greatest_in_second_row in
  sum := least + greatest in
  sum = 367 :=
by
  -- The proof is omitted as per instructions
  sorry

end sum_of_least_and_greatest_in_second_row_l9_9928


namespace sum_of_digits_div_by_11_in_consecutive_39_l9_9073

-- Define the sum of digits function for natural numbers.
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The main theorem statement.
theorem sum_of_digits_div_by_11_in_consecutive_39 :
  ∀ (N : ℕ), ∃ k : ℕ, k < 39 ∧ (sum_of_digits (N + k)) % 11 = 0 :=
by sorry

end sum_of_digits_div_by_11_in_consecutive_39_l9_9073


namespace sum_first_10_terms_l9_9661

noncomputable def sum_first_n_terms_arith_seq (a d : ℕ) (n : ℕ) : ℕ :=
n * (2 * a + (n - 1) * d) / 2

theorem sum_first_10_terms (a d n l : ℕ) (h₀ : a = 5) (h₁ : d = 4) (h₂ : n = 10) (h₃ : l = 41) :
  sum_first_n_terms_arith_seq a d n = 230 :=
by
  -- Introducing the given conditions
  rw [h₀, h₁, h₂]

  -- Simplify the sum using the given facts in Lean
  unfold sum_first_n_terms_arith_seq
  norm_num
  sorry

end sum_first_10_terms_l9_9661


namespace trig_identity_l9_9330

theorem trig_identity (θ : ℝ) (h : Real.tan θ = 2) : 
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4 / 5 :=
by 
  sorry

end trig_identity_l9_9330


namespace find_b2_a2_a1_l9_9767

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

def geometric_sequence (b : ℕ → ℝ) : Prop :=
∀ n : ℕ, b (n + 1) / b n = b 1 / b 0

theorem find_b2_a2_a1 (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_a1 : a 0 = a₁) (h_a2 : a 2 = a₂)
  (h_b2 : b 2 = b₂) :
  b₂ * (a₂ - a₁) = 6 ∨ b₂ * (a₂ - a₁) = -6 :=
by
  sorry

end find_b2_a2_a1_l9_9767


namespace total_area_of_pyramid_l9_9323

open Real

noncomputable def total_area_pyramid : ℝ :=
  let side_length : ℝ := 6    -- side length of the square in cm
  let height : ℝ := 4         -- height of the pyramid in cm
  let square_area : ℝ := side_length^2  -- area of the square
  let triangle_height : ℝ := sqrt (height^2 + (side_length / 2 * sqrt 2)^2) -- height of one isosceles triangle
  let triangle_area : ℝ := (1 / 2) * side_length * triangle_height -- area of one triangle
  let total_triangle_area : ℝ := 4 * triangle_area  -- total area of four triangles
  square_area + total_triangle_area  -- total area of the figure

theorem total_area_of_pyramid 
  (side_length : ℝ) (height : ℝ) : 
  side_length = 6 → height = 4 → total_area_pyramid = 96 :=
by
  intros h₁ h₂
  rw [←h₁, ←h₂]
  unfold total_area_pyramid
  simp
  sorry

end total_area_of_pyramid_l9_9323


namespace smallest_x_plus_y_l9_9373

theorem smallest_x_plus_y (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_neq : x ≠ y) (h_eq : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 15) : x + y = 64 :=
sorry

end smallest_x_plus_y_l9_9373


namespace geometric_sequence_s4_l9_9854

theorem geometric_sequence_s4
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (q : ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a (n + 1) = a n * q)
  (h3 : ∀ n : ℕ, S n = (∑ i in finset.range n, a (i + 1)))
  (h4 : S 5 = 5 * S 3 - 4) :
  S 4 = 15 :=
sorry

end geometric_sequence_s4_l9_9854


namespace five_fold_function_application_l9_9037

def f (x : ℤ) : ℤ :=
if x ≥ 0 then -x^2 + 1 else x + 9

theorem five_fold_function_application : f (f (f (f (f 2)))) = -17 :=
by
  sorry

end five_fold_function_application_l9_9037


namespace commutativity_l9_9918

universe u

variable {M : Type u} [Nonempty M]
variable (star : M → M → M)

axiom star_assoc_right {a b : M} : (star (star a b) b) = a
axiom star_assoc_left {a b : M} : star a (star a b) = b

theorem commutativity (a b : M) : star a b = star b a :=
by sorry

end commutativity_l9_9918


namespace pants_cost_correct_l9_9063

def shirt_cost : ℕ := 43
def tie_cost : ℕ := 15
def total_paid : ℕ := 200
def change_received : ℕ := 2

def total_spent : ℕ := total_paid - change_received
def combined_cost : ℕ := shirt_cost + tie_cost
def pants_cost : ℕ := total_spent - combined_cost

theorem pants_cost_correct : pants_cost = 140 :=
by
  -- We'll leave the proof as an exercise.
  sorry

end pants_cost_correct_l9_9063


namespace elly_is_far_right_l9_9515

def Person := {molly, dolly, sally, elly, kelly}

def Position := { pos1, pos2, pos3, pos4, pos5 }

noncomputable def sitting_arrangement (p: Person) : Option Position := sorry

axiom molly_not_far_right : sitting_arrangement molly ≠ some pos5
axiom dolly_not_far_left : sitting_arrangement dolly ≠ some pos1
axiom sally_not_at_ends : (sitting_arrangement sally ≠ some pos1) ∧ (sitting_arrangement sally ≠ some pos5)
axiom kelly_not_next_to_sally : ¬((sitting_arrangement kelly = some pos2 ∧ sitting_arrangement sally = some pos3) ∨ 
                                 (sitting_arrangement kelly = some pos3 ∧ sitting_arrangement sally = some pos2) ∨
                                 (sitting_arrangement kelly = some pos4 ∧ sitting_arrangement sally = some pos3) ∨
                                 (sitting_arrangement kelly = some pos3 ∧ sitting_arrangement sally = some pos4))
axiom sally_not_next_to_dolly : ¬((sitting_arrangement sally = some pos2 ∧ sitting_arrangement dolly = some pos3) ∨ 
                                  (sitting_arrangement sally = some pos3 ∧ sitting_arrangement dolly = some pos2) ∨
                                  (sitting_arrangement sally = some pos4 ∧ sitting_arrangement dolly = some pos3) ∨
                                  (sitting_arrangement sally = some pos3 ∧ sitting_arrangement dolly = some pos4))
axiom elly_right_of_dolly : ∀ pos_d pos_e, sitting_arrangement dolly = some pos_d → sitting_arrangement elly = some pos_e → pos_d < pos_e

theorem elly_is_far_right : sitting_arrangement elly = some pos5 := sorry

end elly_is_far_right_l9_9515


namespace polynomial_constant_term_l9_9395

theorem polynomial_constant_term (a : ℝ) :
  let poly := (a * x - 1 / real.sqrt x) ^ 9 in
  (∀ x : ℝ, (∃ c : ℝ, 9 - (3 / 2 : ℝ) * 6 = 0 → (c * a ^ 3 = 672)) → a = 2 := by
  sorry

end polynomial_constant_term_l9_9395


namespace no_injective_with_functional_property_exists_surjective_with_functional_property_l9_9236

-- Given a function f from positive rationals to rationals with a specific property
def functional_property (f : ℚ → ℚ) : Prop :=
∀ x y : ℚ, 0 < x → 0 < y → f (x * y) = f x + f y

-- Part (a): There are no injective functions with this property
theorem no_injective_with_functional_property :
  ∀ f : ℚ → ℚ, functional_property f → ¬injective f :=
sorry

-- Part (b): There exist surjective functions with this property
theorem exists_surjective_with_functional_property :
  ∃ f : ℚ → ℚ, functional_property f ∧ surjective f :=
sorry

end no_injective_with_functional_property_exists_surjective_with_functional_property_l9_9236


namespace omega_value_l9_9440

theorem omega_value (ω : ℝ) 
  (h1 : ∀ x y : ℝ, 0 ≤ x ∧ x ≤ y ∧ y ≤ (π / 3) → sin(ω * x) ≤ sin(ω * y))
  (h2 : ∀ x y : ℝ, (π / 3) ≤ x ∧ x ≤ y ∧ y ≤ (π / 2) → sin(ω * x) ≥ sin(ω * y)) :
  ω = 3 / 2 := 
sorry

end omega_value_l9_9440


namespace find_sin_minus_cos_l9_9782

variable {a : ℝ}
variable {α : ℝ}

def point_of_angle (a : ℝ) (h : a < 0) := (3 * a, -4 * a)

theorem find_sin_minus_cos (a : ℝ) (h : a < 0) (ha : point_of_angle a h = (3 * a, -4 * a)) (sinα : ℝ) (cosα : ℝ) :
  sinα = 4 / 5 → cosα = -3 / 5 → sinα - cosα = 7 / 5 :=
by sorry

end find_sin_minus_cos_l9_9782


namespace age_difference_l9_9100

theorem age_difference (alice_pens : ℕ) (clara_pens : ℕ) (alice_age : ℕ) (clara_age_5_years : ℕ) 
(h1 : alice_pens = 60) 
(h2 : clara_pens = (2 * alice_pens) / 5) 
(h3 : alice_age = 20) 
(h4 : clara_age_5_years = 61) : 
alice_age - ((clara_age_5_years - 5 - alice_age) = (alice_pens - clara_pens)) :=
by sorry

end age_difference_l9_9100


namespace total_area_of_paintings_l9_9593

-- Definitions based on the conditions
def painting1_area := 3 * (5 * 5) -- 3 paintings of 5 feet by 5 feet
def painting2_area := 10 * 8 -- 1 painting of 10 feet by 8 feet
def painting3_area := 5 * 9 -- 1 painting of 5 feet by 9 feet

-- The proof statement we aim to prove
theorem total_area_of_paintings : painting1_area + painting2_area + painting3_area = 200 :=
by
  sorry

end total_area_of_paintings_l9_9593


namespace distance_between_lines_l9_9551

/-- Define the lines by their equations -/
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0
def line2 (x y : ℝ) : Prop := 6 * x + 8 * y + 6 = 0

/-- Define the simplified form of the second line -/
def simplified_line2 (x y : ℝ) : Prop := 3 * x + 4 * y + 3 = 0

/-- Prove the distance between the two lines is 3 -/
theorem distance_between_lines : 
  let A : ℝ := 3
  let B : ℝ := 4
  let C1 : ℝ := -12
  let C2 : ℝ := 3
  (|C2 - C1| / Real.sqrt (A^2 + B^2) = 3) :=
by
  sorry

end distance_between_lines_l9_9551


namespace smallest_prime_after_six_nonprimes_l9_9159

open Nat

/-- 
  Proposition:
    97 is the smallest prime number that occurs after a sequence 
    of six consecutive positive integers all of which are nonprime.
-/
theorem smallest_prime_after_six_nonprimes : ∃ (n : ℕ), (n > 0 ∧ Prime (n + 97)) ∧ 
  (∀ k : ℕ, k < n → (Nonprime k) ∧ Nonprime (k+1) ∧ Nonprime (k+2) ∧ Nonprime (k+3) ∧ Nonprime (k+4) ∧ Nonprime (k+5)) ∧ 
  (Prime (n + 1) → n + 1 = 97) := 
by 
  sorry

end smallest_prime_after_six_nonprimes_l9_9159


namespace compute_expression_l9_9281

theorem compute_expression : 
  Real.sqrt 8 - (2017 - Real.pi)^0 - 4^(-1 : Int) + (-1/2)^2 = 2 * Real.sqrt 2 - 1 := 
by 
  sorry

end compute_expression_l9_9281


namespace magnitude_of_combined_vector_l9_9750

variable (a b : EuclideanSpace ℝ (Fin 3))
variable (h1 : ∥a∥ = 1)
variable (h2 : ∥b∥ = Real.sqrt 2)
variable (h3 : InnerProductSpace.inner a b = 1)

theorem magnitude_of_combined_vector : ∥a - (2 : ℝ) • b∥ = Real.sqrt 5 := by
  sorry

end magnitude_of_combined_vector_l9_9750


namespace even_function_f_l9_9507

noncomputable def f (x : ℝ) : ℝ := 
  if -2 ≤ x ∧ x < -1 then x + 4 else
  if -1 ≤ x ∧ x ≤ 0 then -x + 2 else
  if 2 ≤ x ∧ x ≤ 3 then x else
  0 -- Extend this definition for all x as necessary

theorem even_function_f (x : ℝ) :
  (∀ x, f(x - 3/2) = f(x + 1/2)) ∧
  (∀ x, f(x) = f(-x)) ∧
  (∀ x, 2 ≤ x ∧ x ≤ 3 → f(x) = x) →
  (x ∈ [-2, 0] → f(x) = 
   if -2 ≤ x ∧ x < -1 then x + 4 else -x + 2) :=
begin
  intros h x hx,
  cases hx,
  sorry,
end

end even_function_f_l9_9507


namespace right_triangle_k_value_l9_9244

theorem right_triangle_k_value (x : ℝ) (k : ℝ) (s : ℝ) 
(h_triangle : 3*x + 4*x + 5*x = k * (1/2 * 3*x * 4*x)) 
(h_square : s = 10) (h_eq_apothems : 4*x = s/2) : 
k = 8 / 5 :=
by {
  sorry
}

end right_triangle_k_value_l9_9244


namespace smallest_prime_after_six_nonprimes_l9_9190

open Nat

theorem smallest_prime_after_six_nonprimes : 
  ∃ p : ℕ, prime p ∧ p > 0 ∧
  (∃ n : ℕ, ¬prime n ∧ ¬prime (n+1) ∧ ¬prime (n+2) ∧ ¬prime (n+3) ∧ ¬prime (n+4) ∧ ¬prime (n+5) ∧ prime (n+6) ∧ n+6 = p) ∧ 
  p = 89 :=
sorry

end smallest_prime_after_six_nonprimes_l9_9190


namespace chromium_alloy_l9_9870

theorem chromium_alloy (x : ℝ) (h1 : 0.12 * x + 0.10 * 35 = 0.106 * (x + 35)) : x = 15 := 
by 
  -- statement only, no proof required.
  sorry

end chromium_alloy_l9_9870


namespace polynomial_always_has_rational_root_l9_9235

def is_rational_root (p : ℚ → ℚ) (r : ℚ) : Prop :=
  p r = 0

def is_permutation_of_set {α : Type*} [DecidableEq α] (l : List α) (s : Finset α) : Prop :=
  ∀ x, x ∈ l ↔ x ∈ s

def polynomial_with_given_roots (a3 a2 a1 a0 : ℚ) (candidates : Finset ℚ) : Prop :=
  is_permutation_of_set [a3, a2, a1, a0] candidates

theorem polynomial_always_has_rational_root :
  ∀ a3 a2 a1 a0 : ℚ,
  polynomial_with_given_roots a3 a2 a1 a0 (Finset.of_list [1, -2, 3, 4, -6]) →
  ∃ r : ℚ, is_rational_root (λ x, x^4 + a3 * x^3 + a2 * x^2 + a1 * x + a0) r :=
by
  intros a3 a2 a1 a0 h
  use 1
  sorry

end polynomial_always_has_rational_root_l9_9235


namespace lunch_break_duration_l9_9583

/-- Define the total recess time as a sum of two 15-minute breaks and one 20-minute break. -/
def total_recess_time : ℕ := 15 + 15 + 20

/-- Define the total time spent outside of class. -/
def total_outside_class_time : ℕ := 80

/-- Prove that the lunch break is 30 minutes long. -/
theorem lunch_break_duration : total_outside_class_time - total_recess_time = 30 :=
by
  sorry

end lunch_break_duration_l9_9583


namespace length_CQ_l9_9464

variable (ABC : Triangle)
variable (A B C P Q : Point)
variable (PQ : Line)
variable (h1 : on A ABC)
variable (h2 : on B ABC)
variable (h3 : on C ABC)
variable (h4 : on P A)
variable (h5 : on Q B)
variable (h6 : parallel PQ B A)
variable (h7 : bisects Q AQ PR)
variable (h8 : distance PQ = 4)
variable (h9 : distance A B = 10)

theorem length_CQ :
  ∃ CQ : ℝ, CQ = 20 / 3 :=
sorry

end length_CQ_l9_9464


namespace problem_conditions_expression_of_f_min_max_values_solution_set_of_inequality_l9_9799

noncomputable def f (x : ℝ) : ℝ := 2 * (x + 1) * (x - 3)

theorem problem_conditions :
  f (-1) = 0 ∧ f 3 = 0 ∧ f 1 = -8 :=
by 
  -- These are the given conditions of the problem
  sorry

theorem expression_of_f :
  ∀ x, f x = 2 * (x + 1) * (x - 3) :=
by 
  -- Prove the function f(x) given the conditions
  sorry

theorem min_max_values :
  let interval := set.Icc (0 : ℝ) 3 in
  ∃ y_min y_max, 
    (∀ x ∈ interval, f x ≥ y_min) ∧ 
    (∀ x ∈ interval, f x ≤ y_max) ∧ 
    y_min = -8 ∧ 
    y_max = 0 :=
by 
  -- Prove the minimum and maximum values on the interval [0, 3]
  sorry

theorem solution_set_of_inequality :
  { x : ℝ | f x ≥ 0 } = 
    { x : ℝ | x ≤ -1 } ∪ { x : ℝ | x ≥ 3 } :=
by 
  -- Prove the solution set of the inequality f(x) ≥ 0
  sorry

end problem_conditions_expression_of_f_min_max_values_solution_set_of_inequality_l9_9799


namespace max_third_side_of_triangle_l9_9543

theorem max_third_side_of_triangle 
  (D E F : ℝ) (a b : ℝ) (h1 : a = 7) (h2 : b = 24)
  (h3 : ∠D + ∠E + ∠F = 180) 
  (h4 : cos 4 * D + cos 4 * E + cos 4 * F = 1) :
  ∃ c : ℝ, c ≤ 25 := 
sorry

end max_third_side_of_triangle_l9_9543


namespace rays_nickels_left_l9_9942

theorem rays_nickels_left (initial_cents : ℕ) (exchange_cents : ℕ) (peter_ratio randi_ratio paula_ratio : ℝ) :
    initial_cents = 475 → exchange_cents = 75 → peter_ratio = 2 / 5 → randi_ratio = 3 / 5 → paula_ratio = 1 / 10 →
    let initial_nickels := initial_cents / 5 in
    let exchanged_nickels := exchange_cents / 5 in
    let remaining_nickels := initial_nickels - exchanged_nickels in
    let dimes_received := exchange_cents / 10 in
    let peter_dimes := peter_ratio * dimes_received in
    let randi_dimes := randi_ratio * dimes_received in
    let paula_dimes := paula_ratio * dimes_received in
    let leftover_dimes := dimes_received - (peter_dimes + randi_dimes + paula_dimes) in
    let leftover_nickels := leftover_dimes * 2 in
    remaining_nickels + leftover_nickels = 82 :=
by
  intros h_initial h_exchange h_peter_ratio h_randi_ratio h_paula_ratio
  -- The proof goes here
  sorry

end rays_nickels_left_l9_9942


namespace smallest_x_plus_y_l9_9375

theorem smallest_x_plus_y (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_neq : x ≠ y) (h_eq : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 15) : x + y = 64 :=
sorry

end smallest_x_plus_y_l9_9375


namespace intersection_M_N_l9_9508

def M : Set ℤ := {0}
def N : Set ℤ := {x | -1 < x ∧ x < 1}

theorem intersection_M_N : M ∩ N = {0} :=
by
  sorry

end intersection_M_N_l9_9508


namespace factors_of_60_multiple_of_6_l9_9420

theorem factors_of_60_multiple_of_6 :
  let factors_of_60 := {d : ℕ | d is a divisor of 60}
  let multiples_of_6 := {d : ℕ | d % 6 = 0}
  let multiples_of_6_factors := multiples_of_60 ∩ factors_of_60
  card multiples_of_6_factors = 4 := by
  sorry

end factors_of_60_multiple_of_6_l9_9420


namespace no_four_nat_satisfy_l9_9078

theorem no_four_nat_satisfy:
  ∀ (x y z t : ℕ), 3 * x^4 + 5 * y^4 + 7 * z^4 ≠ 11 * t^4 :=
by
  sorry

end no_four_nat_satisfy_l9_9078


namespace find_f2_l9_9747

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^5 - a * x^3 + b * x - 6

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -22 :=
by
  sorry

end find_f2_l9_9747


namespace frustum_volume_fractional_part_l9_9247

-- Define the volume of a pyramid
def volume_pyramid (base_area : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * base_area * height

-- Define the volume area of a square given side length
def square_area (side : ℝ) : ℝ := side * side

-- Define the initial parameters
def base_edge : ℝ := 24
def original_height : ℝ := 10
def smaller_height : ℝ := original_height / 3

-- Compute areas
def original_base_area := square_area base_edge
def smaller_base_area := square_area (base_edge / 3)

-- Compute volumes
def original_volume := volume_pyramid original_base_area original_height
def smaller_volume := volume_pyramid smaller_base_area smaller_height
def frustum_volume := original_volume - smaller_volume

-- Fractional volume of the frustum relative to the original pyramid
def fractional_volume : ℝ := frustum_volume / original_volume

theorem frustum_volume_fractional_part :
  fractional_volume = 924 / 960 :=
by
  sorry

end frustum_volume_fractional_part_l9_9247


namespace green_paint_amount_l9_9998

theorem green_paint_amount (T W B : ℕ) (hT : T = 69) (hW : W = 20) (hB : B = 34) : 
  T - (W + B) = 15 := 
by
  sorry

end green_paint_amount_l9_9998


namespace probability_oil_tank_explode_probability_shots_geq_4_l9_9838

theorem probability_oil_tank_explode :
  let P_hit := 2 / 3
  let P_miss := 1 - P_hit
  let hits_needed := 2
  let bullets := 5
  let P_X_0 := P_miss ^ bullets
  let P_X_1 := C(bullets, 1) * P_hit * P_miss ^ (bullets - 1)
  let P_explode := 1 - P_X_0 - P_X_1
  in P_explode = 232 / 243 :=
sorry

theorem probability_shots_geq_4 :
  let P_hit := 2 / 3
  let P_miss := 1 - P_hit
  let bullets := 5
  let P_X_2 := C(bullets, 2) * (P_hit ^ 2) * (P_miss ^ (bullets - 2))
  let P_X_3 := C(bullets, 3) * (P_hit ^ 3) * (P_miss ^ (bullets - 3))
  let P_not_less_than_4 := 1 - P_X_2 - P_X_3
  in P_not_less_than_4 = 7 / 27 :=
sorry

end probability_oil_tank_explode_probability_shots_geq_4_l9_9838


namespace equipment_transfer_l9_9264

theorem equipment_transfer 
    (x y : ℕ) 
    (h1 : x < y)
    (h2 : 0.3 * x + 0.7 * x + 0.1 * (y + 0.3 * x) = 1.1 * x)
    (h3 : 0.1 * (y + 0.3 * x) = 0.1 * y + 0.03 * x)
    (h4 : 0.5 * (0.1 * (y + 0.3 * x)) = 0.05 * y + 0.015 * x)
    (h5 : 0.73 * x + 0.1 * y = 0.27 * x + 0.9 * y - 6)
    (h6 : 0.9 * (y + 0.3 * x) > 1.02 * y)
    (eqn : 0.46 * x - 0.8 * y = 6)
    (ineq : x > 4 / 9 * y) 
    : y = 17 := 
begin
  sorry
end

end equipment_transfer_l9_9264


namespace line_through_intersection_parallel_line_through_point_distance_l9_9309

-- Statement for Part I
theorem line_through_intersection_parallel (x y : ℝ) :
  (∃ (l1 l2 : ℝ → ℝ → Prop), 
    (l1 1 3 - 3 = 0) ∧ 
    (l2 1 (-1) + 1 = 0) ∧ 
    (l x = 2) ∧ 
    (l 1 3 - 3 = 0) ∧ 
    (l x (-1) + 1 = 0) → 
    (2 * x + y - 1 = 0)) :=
sorry

-- Statement for Part II
theorem line_through_point_distance 
  (x y : ℝ) 
  (A : ℝ × ℝ) 
  (dist_AB : ℝ) :
  (∃ (l1 : ℝ → ℝ → Prop) 
    (B : ℝ × ℝ → ℝ × ℝ → ℝ),
    A = (1, -1) ∧ 
    l1 2 ((-6) + 3 - 0) = 0 ∧ 
    dist_AB = 5 ∧ 
    (B A l1 = (1, 4)) → 
    (x = 1 ∨ (3 * x + 4 * y + 1 = 0))) :=
sorry

end line_through_intersection_parallel_line_through_point_distance_l9_9309


namespace remainder_div_1_l9_9049

def p (x : ℝ) : ℝ := x^6 - 4 * x^5 + 6 * x^4 - 4 * x^3 + x^2

theorem remainder_div_1 (x : ℝ) (t_2 : ℝ) : (∃ s1 : ℝ → ℝ, ∀ x, p(x) = (x - 1) * s1(x) + 0) →
  (∃ s2 : ℝ → ℝ, ∀ x, s1(x) = (x - 1) * s2(x) + t_2) →
  t_2 = 0 :=
by 
  intros h1 h2
  sorry

end remainder_div_1_l9_9049


namespace radian_measure_of_45_deg_l9_9985

theorem radian_measure_of_45_deg (pi : ℝ) (h : 180 = pi) : 
∃ r : ℝ, r = π/4 :=
by {
  sorry,
}

end radian_measure_of_45_deg_l9_9985


namespace width_of_plot_l9_9241

-- Define the conditions as separate definitions

def length_of_plot : ℕ := 90
def number_of_poles : ℕ := 28
def distance_between_poles : ℕ := 10

-- Define the statement to be proved, including the conditions and the desired result

theorem width_of_plot :
  ∃ (width : ℕ),
    let number_of_gaps := number_of_poles - 1,
    let total_length_of_fence := number_of_gaps * distance_between_poles,
    let perimeter := 2 * length_of_plot + 2 * width
    total_length_of_fence = perimeter ∧ width = 45 :=
by
  sorry

end width_of_plot_l9_9241


namespace greatest_distance_of_inscribed_squares_l9_9636

open Real

/-- Definition of the distances -/
def is_inscribed (inner_side outer_side : ℝ) : Prop :=
  inner_side = 6 ∧ outer_side = 8

/-- The greatest distance between a vertex of the inner square and a vertex of the outer square -/
theorem greatest_distance_of_inscribed_squares :
  ∀ (inner_side outer_side : ℝ), is_inscribed inner_side outer_side → 
    ∃ d : ℝ, d = sqrt 2 :=
by
  intro inner_side outer_side h
  cases h with h_inner h_outer
  use sqrt 2
  sorry

end greatest_distance_of_inscribed_squares_l9_9636


namespace triangle_area_l9_9455

noncomputable def triangle_area_calc : ℝ :=
let A := (-Real.sqrt 2 / 2, 0) in
let line_eq (x : ℝ) := Real.sqrt 2 * x + 1 in
let asymptote_eq (x : ℝ) := -Real.sqrt 2 * x in
let x_intercept := -Real.sqrt 2 / 4 in
let y_intercept := 1 / 2 in
(1 / 2) * (Real.abs (x_intercept - 0)) * (Real.abs y_intercept)

theorem triangle_area :
  let C1 (x y : ℝ) := 2 * x^2 - y^2 = 1 
  let vertex := (-Real.sqrt 2 / 2, 0)
  let line_parallel_asymptote (x : ℝ) := Real.sqrt 2 * (x + Real.sqrt 2 / 2)
  triangle_area_calc = Real.sqrt 2 / 8 :=
sorry

end triangle_area_l9_9455


namespace fraction_of_product_l9_9540

theorem fraction_of_product (c d: ℕ) 
  (h1: 5 * 64 + 4 * 8 + 3 = 355)
  (h2: 2 * (10 * c + d) = 355)
  (h3: c < 10)
  (h4: d < 10):
  (c * d : ℚ) / 12 = 5 / 4 :=
by
  sorry

end fraction_of_product_l9_9540


namespace sin_angle_GAC_correct_l9_9209

noncomputable def sin_angle_GAC (AB AD AE : ℝ) := 
  let AC := Real.sqrt (AB^2 + AD^2)
  let AG := Real.sqrt (AB^2 + AD^2 + AE^2)
  (AC / AG)

theorem sin_angle_GAC_correct : sin_angle_GAC 2 3 4 = Real.sqrt 377 / 29 := by
  sorry

end sin_angle_GAC_correct_l9_9209


namespace complex_exp_form_pow_four_l9_9665

theorem complex_exp_form_pow_four :
  let θ := 30 * Real.pi / 180
  let cos_θ := Real.cos θ
  let sin_θ := Real.sin θ
  let z := 3 * (cos_θ + Complex.I * sin_θ)
  z ^ 4 = -40.5 + 40.5 * Complex.I * Real.sqrt 3 :=
by
  sorry

end complex_exp_form_pow_four_l9_9665


namespace solve_triples_l9_9711

theorem solve_triples (a b c : ℝ) :
  (a^5 = 5*b^3 - 4*c) →
  (b^5 = 5*c^3 - 4*a) →
  (c^5 = 5*a^3 - 4*b) →
  (a, b, c) ∈ {(0, 0, 0), (1, 1, 1), (-1, -1, -1), (2, 2, 2), (-2, -2, -2)} :=
by
  sorry

end solve_triples_l9_9711


namespace circumradii_eq_l9_9344

/-- Given a triangle ABC with circumcenter O, the circumcircle of triangle AOC intersects BC at points C and D, 
    and AB at points A and E. Prove that the circumradii of triangles BDE and AOC are equal. -/
theorem circumradii_eq
  (A B C O D E : Type*) 
  (h1: ∃ (k1 k2 : Circle), Circumcircle A O C k1 ∧ Circumcircle B D E k2 ∧ IsOnCircle C D k1 ∧ IsOnCircle A E k1)
  : Circumradius B D E = Circumradius A O C :=
sorry

end circumradii_eq_l9_9344


namespace min_sum_seven_integers_l9_9343

theorem min_sum_seven_integers {a : ℕ → ℕ} 
  (pos : ∀ n, 1 ≤ a n)  -- All integers are positive
  (sorted : ∀ i j, i < j → a i ≤ a j)  -- Integers are in increasing order
  (med : a 3 = 4)  -- The median is 4
  (mode_6 : ∃ (n m : ℕ), n ≠ m ∧ a n = 6 ∧ a m = 6 ∧ ∀ k ≠ n ∧ k ≠ m, a k ≠ 6)  -- Unique mode is 6
  : (∑ n in (finset.range 7), a n) ≥ 26 := 
sorry

end min_sum_seven_integers_l9_9343


namespace problem1_problem2_l9_9274

-- Problem 1: Prove that \(\sqrt[3]{27} + \sqrt[3]{-8} - \sqrt{(-2)^2} + |1 - \sqrt{3}| = -2 + \sqrt{3}\)
theorem problem1 : real.cbrt 27 + real.cbrt (-8) - real.sqrt ((-2:ℝ)^2) + |1 - real.sqrt 3| = -2 + real.sqrt 3 :=
by sorry

-- Problem 2: Prove that \((-4x^2y^3)\left(\frac{1}{8}xyz\right) \div \left(\frac{1}{2}xy^2\right)^2 = -2xz\)
theorem problem2 (x y z : ℝ): (-4 * x^2 * y^3) * (1/8 * x * y * z) / (1/2 * x * y^2)^2 = -2 * x * z :=
by sorry

end problem1_problem2_l9_9274


namespace intersection_points_max_distance_curve_to_line_l9_9023

-- Definitions based on problem conditions
def curve_C (θ : ℝ) : ℝ × ℝ :=
  (3 * Real.cos θ, Real.sin θ)

def line_l (a t : ℝ) : ℝ × ℝ :=
  (a + 4 * t, 1 - t)

-- Proof Problem 1
theorem intersection_points (a : ℝ) :
  a = -1 →
  ∃ (θ t : ℝ), curve_C θ = line_l a t :=
sorry

-- Proof Problem 2
theorem max_distance_curve_to_line (a : ℝ) :
  (∃ θ, let d := (|3 * Real.cos θ + 4 * Real.sin θ - a - 4| / Real.sqrt 17) in
         d = Real.sqrt 17) →
  a = -16 ∨ a = 8 :=
sorry

end intersection_points_max_distance_curve_to_line_l9_9023


namespace price_reduction_for_1200_profit_price_reduction_for_max_profit_l9_9612

noncomputable def average_daily_sales : ℕ := 20
noncomputable def profit_per_piece : ℕ := 40
noncomputable def sales_increase_per_dollar_reduction : ℕ := 2
variable (x : ℝ)

def average_daily_profit (x : ℝ) : ℝ :=
  (profit_per_piece - x) * (average_daily_sales + sales_increase_per_dollar_reduction * x)

theorem price_reduction_for_1200_profit :
  ((average_daily_profit x) = 1200) → (x = 20) := by
  sorry

theorem price_reduction_for_max_profit :
  (∀ x, average_daily_profit x ≤ average_daily_profit 15) ∧ (average_daily_profit 15 = 1250) := by
  sorry

end price_reduction_for_1200_profit_price_reduction_for_max_profit_l9_9612


namespace f_2016_eq_sinx_l9_9048

noncomputable def f : ℕ → (ℝ → ℝ)
| 0       := λ x, Real.sin x
| (n + 1) := λ x, (f n) x'  -- Derivative of fₙ(x)

theorem f_2016_eq_sinx : ∀ x, f 2016 x = Real.sin x :=
by 
  have h_cycle : ∀ (n : ℕ) (x : ℝ), f (n + 4) x = f n x,
  { intro n,
    induction n with n ih,
    { intro x,
      simp [f, deriv_sin, deriv_cos], },
    { intro x,
      simp [f, ih x, deriv_sin, deriv_cos], } },
  intro x,
  have h_div : 2016 = 504 * 4 := rfl,
  rw [h_div, Nat.mul_add],
  exact h_cycle 504 x

end f_2016_eq_sinx_l9_9048


namespace prob_not_face_card_red_or_spades_l9_9223

def cards_set := finset (fin 52)
def red_cards := ({x | x > 0 ∧ x ≤ 26 : finset (fin 52)})
def spades_cards := ({x | x > 38 ∧ x ≤ 51 : finset (fin 52)})
def face_cards := ({x | (x % 13 = 10 ∨ x % 13 = 11 ∨ x % 13 = 12) : finset (fin 52)})

def red_or_spades : finset (fin 52) := red_cards ∪ spades_cards

noncomputable def prob_not_face_card := (red_or_spades.card - (red_or_spades ∩ face_cards).card) / red_or_spades.card

theorem prob_not_face_card_red_or_spades : prob_not_face_card = 10 / 13 := by
  sorry

end prob_not_face_card_red_or_spades_l9_9223


namespace log_x_64_l9_9818

theorem log_x_64 : 
  ∀ (x : ℝ), (log 8 (5 * x) = 2) → (log x 64 = 5) :=
by
  sorry

end log_x_64_l9_9818


namespace geometric_sequence_s4_l9_9851

theorem geometric_sequence_s4
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (q : ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a (n + 1) = a n * q)
  (h3 : ∀ n : ℕ, S n = (∑ i in finset.range n, a (i + 1)))
  (h4 : S 5 = 5 * S 3 - 4) :
  S 4 = 15 :=
sorry

end geometric_sequence_s4_l9_9851


namespace remainder_div_84_l9_9751

def a := (5 : ℕ) * 10 ^ 2015 + (5 : ℕ)

theorem remainder_div_84 (a : ℕ) (h : a = (5 : ℕ) * 10 ^ 2015 + (5 : ℕ)) : a % 84 = 63 := 
by 
  -- Placeholder for the actual steps to prove
  sorry

end remainder_div_84_l9_9751


namespace range_of_independent_variable_l9_9122

theorem range_of_independent_variable (x : ℝ) (h : 2 - x ≥ 0) : x ≤ 2 :=
sorry

end range_of_independent_variable_l9_9122


namespace calculate_total_cost_l9_9228

-- Define the cost for type A and type B fast foods as constants
def cost_of_type_A : ℕ := 30
def cost_of_type_B : ℕ := 20

-- Define the number of servings as variables
variables (a b : ℕ)

-- Define a function that calculates the total cost
def total_cost (a b : ℕ) : ℕ :=
  cost_of_type_A * a + cost_of_type_B * b

-- The theorem statement: Prove that the total cost is as calculated
theorem calculate_total_cost (a b : ℕ) :
  total_cost a b = 30 * a + 20 * b := by
  unfold total_cost
  simp

end calculate_total_cost_l9_9228


namespace complex_power_eq_rectangular_l9_9702

noncomputable def cos_30 := Real.cos (Real.pi / 6)
noncomputable def sin_30 := Real.sin (Real.pi / 6)

theorem complex_power_eq_rectangular :
  (3 * (cos_30 + (complex.I * sin_30)))^4 = -40.5 + 40.5 * complex.I * Real.sqrt 3 := 
sorry

end complex_power_eq_rectangular_l9_9702


namespace acute_angle_of_line_l9_9408

theorem acute_angle_of_line (x y : ℝ) (h : √3 * x - y + 6 = 0) :
  ∃ θ : ℝ, θ = 60 ∧ θ < 90 := 
sorry

end acute_angle_of_line_l9_9408


namespace smallest_prime_after_six_nonprime_l9_9170

open Nat

noncomputable def is_nonprime (n : ℕ) : Prop :=
  ¬Prime n ∧ n ≠ 1

noncomputable def six_consecutive_nonprime (n : ℕ) : Prop :=
  is_nonprime n ∧ is_nonprime (n + 1) ∧ is_nonprime (n + 2) ∧ 
  is_nonprime (n + 3) ∧ is_nonprime (n + 4) ∧ is_nonprime (n + 5)

theorem smallest_prime_after_six_nonprime :
  ∃ n, six_consecutive_nonprime n ∧ (∀ m, m ≠ 97 → n + 6 < m → ¬Prime m) :=
by
  sorry

end smallest_prime_after_six_nonprime_l9_9170


namespace machine_x_production_rate_l9_9057

-- Define X's rate of production
def S_x : ℝ := 6

-- Define B's rate of production
def S_b : ℝ := 1.10 * S_x

-- Define times for production
def T_b : ℝ := 660 / S_b
def T_x : ℝ := T_b + 10

theorem machine_x_production_rate :
  (S_x * T_x = 660) ∧
  (S_b * T_b = 660) ∧
  (S_b = 1.10 * S_x) ∧
  (T_x = T_b + 10) 
  → S_x = 6 := by
  -- The proof is not provided, so we use sorry.
  sorry

end machine_x_production_rate_l9_9057


namespace prime_exists_distinct_elements_l9_9484

theorem prime_exists_distinct_elements (p : ℕ) (p_prime : Nat.Prime p) (p_gt_5 : 5 < p) :
  ∃ (x y : ℕ), x ≠ 1 ∧ x ∈ {p - n^2 | n : ℕ, n^2 < p} ∧ y ∈ {p - n^2 | n : ℕ, n^2 < p} ∧ x ∣ y ∧ x ≠ y :=
by
  sorry

end prime_exists_distinct_elements_l9_9484


namespace wholesale_price_of_pen_l9_9930

-- Definitions and conditions
def wholesale_price (P : ℝ) : Prop :=
  (5 - P = 10 - 3 * P)

-- Statement of the proof problem
theorem wholesale_price_of_pen : ∃ P : ℝ, wholesale_price P ∧ P = 2.5 :=
by {
  sorry
}

end wholesale_price_of_pen_l9_9930


namespace identify_incorrect_option_l9_9321
-- Import the necessary library for mathematics

-- Define the conditions of the problem
variable (b a x y : ℝ) (hat_y : ℝ → ℝ)

-- Define the linear regression equation
def linear_regression_equation (b a x : ℝ) := b * x + a

-- Define the sample data point (x, y) and the regression equation value for that x
variable (sample : x = 0 → y = a)

-- Define the average increase when x increases by one unit
def avg_increase (b : ℝ) := b

-- Define the line passing through the point (x̄, ȳ)
variable (x̄ ȳ : ℝ)
variable (passes_through_mean : linear_regression_equation b a x̄ = ȳ)

-- State the theorem that identifies the incorrect statement
theorem identify_incorrect_option :
  ¬ (x = 0 → y = a) ∧
  (∀ x y, x + 1 → y + b) ∧
  (x = 0 → y = a ∨ y ≠ a) ∧
  (linear_regression_equation b a x̄ = ȳ)
:= sorry

end identify_incorrect_option_l9_9321


namespace problem_II_l9_9794

variable (a m x1 x2 : ℝ)
variable (f g : ℝ → ℝ)

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x + 5 - a / (Real.exp x)
noncomputable def g (x : ℝ) : ℝ := Real.exp x * f x

theorem problem_II (h1 : m ≥ 1) (h2 : g x1 + g x2 = 2 * g m) (h3 : x1 ≠ x2) (h4 : a ≥ 2 * Real.exp 1) : x1 + x2 < 2 * m := 
sorry

end problem_II_l9_9794


namespace person_B_spheres_needed_l9_9068

-- Translate conditions to Lean definitions
def sum_squares (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6
def sum_triangulars (m : ℕ) : ℕ := (m * (m + 1) * (m + 2)) / 6

-- Define the main theorem
theorem person_B_spheres_needed (n m : ℕ) (hA : sum_squares n = 2109)
    (hB : m ≥ 25) : sum_triangulars m = 2925 :=
    sorry

end person_B_spheres_needed_l9_9068


namespace Beth_bought_10_cans_of_corn_l9_9656

theorem Beth_bought_10_cans_of_corn (a b : ℕ) (h1 : b = 15 + 2 * a) (h2 : b = 35) : a = 10 := by
  sorry

end Beth_bought_10_cans_of_corn_l9_9656


namespace compare_numbers_l9_9193

noncomputable def first_number : ℝ :=
  let rec nested_radical (n : ℕ) : ℝ :=
    if n = 0 then 17 else sqrt (if n % 2 = 0 then 17 else 13) * nested_radical (n - 1)
  in nested_radical 2018

def second_number : ℝ := 17 * real.cbrt (13 / 17)

theorem compare_numbers : first_number < second_number := sorry

end compare_numbers_l9_9193


namespace perfect_fruits_l9_9447

def total_fruits := 120
def apples := 60
def oranges := 40
def mangoes := 20

-- Apple distributions
def apple_small := 1/4
def apple_medium := 1/2
def apple_large := 1/4
def apple_unripe := 1/3
def apple_partly_ripe := 1/6
def apple_fully_ripe := 1/2

-- Orange distributions
def orange_small := 1/3
def orange_medium := 1/3
def orange_large := 1/3
def orange_unripe := 1/2
def orange_partly_ripe := 1/4
def orange_fully_ripe := 1/4

-- Mango distributions
def mango_small := 1/5
def mango_medium := 2/5
def mango_large := 2/5
def mango_unripe := 1/4
def mango_partly_ripe := 1/2
def mango_fully_ripe := 1/4

-- Perfect fruits definitions
def perfect_apples := 30
def perfect_oranges := 10
def perfect_mangoes := 15

theorem perfect_fruits : 
  (apple_medium * apples + apple_large * apples) * apple_fully_ripe + (orange_large * oranges) * orange_fully_ripe + ((mango_medium + mango_large) * mangoes) * (mango_partly_ripe + mango_fully_ripe) = 55 := 
by 
  have perfect_apples : (apple_fully_ripe * apples = 30) := sorry,
  have perfect_oranges : (orange_fully_ripe * oranges = 10) := sorry,
  have perfect_mangoes : ((mango_partly_ripe + mango_fully_ripe) * mangoes = 15) := sorry,
  exact perfect_apples + perfect_oranges + perfect_mangoes

end perfect_fruits_l9_9447


namespace range_of_a_l9_9828

noncomputable def f (a x : ℝ) : ℝ :=
  (1 / 3) * x^3 - (1 / 2) * a * x^2 + (a - 1) * x + 1

theorem range_of_a (a : ℝ) :
  (∀ x ∈ set.Ioo (1 : ℝ) 4, deriv (f a) x ≤ 0) ∧ 
  (∀ x ∈ set.Ioi (6 : ℝ), 0 ≤ deriv (f a) x) →
  5 ≤ a ∧ a ≤ 7 :=
by
  sorry

end range_of_a_l9_9828


namespace trevor_eggs_left_l9_9466

def gertrude_eggs : Nat := 4
def blanche_eggs : Nat := 3
def nancy_eggs : Nat := 2
def martha_eggs : Nat := 2
def dropped_eggs : Nat := 2

theorem trevor_eggs_left : 
  (gertrude_eggs + blanche_eggs + nancy_eggs + martha_eggs - dropped_eggs) = 9 := 
  by sorry

end trevor_eggs_left_l9_9466


namespace estimate_sum_of_digits_binom_coefficient_l9_9497

theorem estimate_sum_of_digits_binom_coefficient :
  let N := ((Nat.digits 10 (Nat.factorial 1000 / (Nat.factorial 100 * Nat.factorial 900))).sum)
  N ≈ 621 :=
by
  sorry

end estimate_sum_of_digits_binom_coefficient_l9_9497


namespace identical_rooms_per_floor_l9_9808

theorem identical_rooms_per_floor 
    (total_floors : ℕ) (rooms_per_floor : ℕ) (available_rooms : ℕ) 
    (total_floors = 10)
    (available_floors = total_floors - 1)
    (available_rooms = 90) : 
    rooms_per_floor = 10 := 
by 
  sorry

end identical_rooms_per_floor_l9_9808


namespace zero_point_interval_l9_9964

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x - 2

theorem zero_point_interval :
  ∃ c ∈ Ioo 0 1, f c = 0 := by
  have h0 : f 0 < 0 := by
    simp [f, Real.exp_zero]
    norm_num
  have h1 : f 1 > 0 := by
    simp [f, Real.exp_one]
    linarith [Real.exp_pos 1]
  have h_continuous : Continuous f := by
    exact continuous_exp.add continuous_id.sub continuous_const
  
  obtain ⟨c, hc0, hc1⟩ := IntermediateValueTheoremIoo 0 1 0 f h_continuous h0 h1
  use [c, hc0, hc1]
  exact hc1

end zero_point_interval_l9_9964


namespace complex_fourth_power_l9_9695

-- Definitions related to the problem conditions
def trig_identity (θ : ℝ) : ℂ := complex.of_real (cos θ) + complex.I * complex.of_real (sin θ)
def z : ℂ := 3 * trig_identity (30 * real.pi / 180)
def result := (81 : ℝ) * (-1/2 + complex.I * (real.sqrt 3 / 2))

-- Statement of the problem
theorem complex_fourth_power : z^4 = result := by
  sorry

end complex_fourth_power_l9_9695


namespace problem1_problem2_l9_9606

noncomputable def expr1 : ℚ :=
  (9/4) ^ (1/2 : ℚ) + (-3.8 : ℚ) ^ 0 - 3 ^ (1/2 : ℚ) * (3/2 : ℚ) ^ (1/3 : ℚ) * (12 : ℚ) ^ (1/6 : ℚ)

noncomputable def expr2 : ℚ :=
  2 * log 3 2 - log 3 (32/9 : ℚ) + log 3 8 - (log 9 / log 2) * log 3 2

theorem problem1 : expr1 = (-1/2 : ℚ) :=
  sorry

theorem problem2 : expr2 = 2 :=
  sorry

end problem1_problem2_l9_9606


namespace harmonic_series_not_integer_l9_9937

theorem harmonic_series_not_integer (n : ℕ) (h : 1 < n) : ¬ ∃ k : ℤ, (∑ i in (finset.range n).image (λ i, i + 1), (1 : ℚ) / i) = k := 
sorry

end harmonic_series_not_integer_l9_9937


namespace solve_equation_l9_9570

theorem solve_equation : ∀ x : ℝ, x * (x + 1) = 2 * (x + 1) → (x = -1 ∨ x = 2) :=
by 
  intro x,
  intro h,
  have h1 : x * (x + 1) = 2 * (x + 1) ↔ (x + 1) * (x - 2) = 0,
  {
    split,
    {
      intro h2,
      calc
        x * (x + 1) - 2 * (x + 1)
            = (x * (x + 1) - 2 * (x + 1)) : by simp [mul_sub_right_distrib, h2]
        ... = (x + 1) * (x - 2) : by ring,
    },
    {
      intro h2,
      rw ← sub_eq_zero at h2,
      exact h2
    },
  },
  rw h1 at h,
  have h2 : x + 1 = 0 ∨ x - 2 = 0,
  {
    cases (mul_eq_zero.mp h),
    {
      split,
      exact h.left,
      split,
      exact h.right,
    },
    {
      split,
      intro h3,
      exact h3.left,
      intro h3,
      exact h3.right,
    },
  }, 
  exact h2

end solve_equation_l9_9570


namespace problem_l9_9040

theorem problem (a b : ℕ) (ha : 2^a ∣ 180) (h2 : ∀ n, 2^n ∣ 180 → n ≤ a) (hb : 5^b ∣ 180) (h5 : ∀ n, 5^n ∣ 180 → n ≤ b) : (1 / 3) ^ (b - a) = 3 := by
  sorry

end problem_l9_9040


namespace alpha_irrational_l9_9882

theorem alpha_irrational (α : ℝ) (h : ∀ (n : ℕ), decimal_digits α n = to_digit (floor (n * real.sqrt 2))) : irrational α := sorry

end alpha_irrational_l9_9882


namespace problem_part_I_problem_part_II_problem_part_III_l9_9753

-- Define the circle M
def M := {p : ℝ × ℝ // ∃ x y, p = (x, y) ∧ x^2 + (y - 4)^2 = 1}

-- Define the line l
def l := {p : ℝ × ℝ // ∃ x y, p = (x, y) ∧ 2 * x - y = 0}

-- Definition of point P on line l
def P (a : ℝ) := (a, 2 * a)

theorem problem_part_I (P_coords : ℝ × ℝ) (h : ∃ a, P_coords = P a) (angle_APB : ∠ PA P B = 60) :
P_coords = (2, 4) ∨ P_coords = (1.2, 2.4) := by sorry

theorem problem_part_II (P_coords : ℝ × ℝ) (hP : P_coords = (1, 2)) (CD_length : ℝ) 
(hCD : CD_length = sqrt 2) :
(equational_line_CD = "x + y - 3 = 0") ∨ (equational_line_CD = "7x + y - 9 = 0") := by sorry

theorem problem_part_III (P_coords : ℝ × ℝ) (h : ∃ a, P_coords = P a) :
∃ fixed_point, fixed_point = (1 / 2, 15 / 4) := by sorry

end problem_part_I_problem_part_II_problem_part_III_l9_9753


namespace find_d_l9_9881

theorem find_d (d : ℚ) :
  let x_intercept := -d / 6
      y_intercept := -d / 5
  in x_intercept + y_intercept = 15 → d = -450 / 11 := by
  sorry

end find_d_l9_9881


namespace sum_of_exponents_l9_9996

theorem sum_of_exponents (i j k : ℕ) (h : (∑ x in range (i + 1), 2^x) * (∑ y in range (j + 1), 3^y) * (∑ z in range (k + 1), 5^z) = 3600) :
  i + j + k = 7 := 
sorry

end sum_of_exponents_l9_9996


namespace shifted_parabola_passes_through_neg1_1_l9_9107

def original_parabola (x : ℝ) : ℝ := -(x + 1)^2 + 4

def shifted_parabola (x : ℝ) : ℝ := -(x - 1)^2 + 2

theorem shifted_parabola_passes_through_neg1_1 :
  shifted_parabola (-1) = 1 :=
by 
  -- Proof goes here
  sorry

end shifted_parabola_passes_through_neg1_1_l9_9107


namespace find_p_when_q_is_1_l9_9820

-- Define the proportionality constant k and the relationship
variables {k p q : ℝ}
def inversely_proportional (k q p : ℝ) : Prop := p = k / (q + 2)

-- Given conditions
theorem find_p_when_q_is_1 (h1 : inversely_proportional k 4 1) : 
  inversely_proportional k 1 2 :=
by 
  sorry

end find_p_when_q_is_1_l9_9820


namespace smallest_sum_of_xy_l9_9362

theorem smallest_sum_of_xy (x y : ℕ) (h1 : x ≠ y) (h2 : 0 < x) (h3 : 0 < y) 
  (h4 : (1:ℚ)/x + (1:ℚ)/y = (1:ℚ)/15) : x + y = 64 :=
sorry

end smallest_sum_of_xy_l9_9362


namespace quadratic_inequality_solution_l9_9638

theorem quadratic_inequality_solution (x : ℝ) : 2 * x^2 - 5 * x - 3 ≥ 0 ↔ x ≤ -1/2 ∨ x ≥ 3 := 
by
  sorry

end quadratic_inequality_solution_l9_9638


namespace geom_seq_sum_4_l9_9858

noncomputable def geom_seq (n : ℕ) (q : ℝ) (a₁ : ℝ) : ℝ :=
  a₁ * q^(n - 1)

noncomputable def sum_geom_seq (n : ℕ) (q : ℝ) (a₁ : ℝ) : ℝ :=
  if q = 1 then a₁ * n else (a₁ * (1 - q^n) / (1 - q))

theorem geom_seq_sum_4 {q : ℝ} (hq : q > 0) (hq1 : q ≠ 1) :
  let a₁ := 1 in
  let S5 := sum_geom_seq 5 q a₁ in
  let S3 := sum_geom_seq 3 q a₁ in
  S5 = 5 * S3 - 4 →
  sum_geom_seq 4 q a₁ = 15 :=
by
  sorry

end geom_seq_sum_4_l9_9858


namespace six_vertex_graph_has_3_clique_or_independent_set_l9_9530

-- Define a 3-vertex clique
def is_clique (G : SimpleGraph V) (H : Set V) : Prop :=
  ∀ ⦃x y⦄, x ∈ H → y ∈ H → x ≠ y → G.Adj x y

-- Define a 3-vertex independent set (anticlique)
def is_independent_set (G : SimpleGraph V) (H : Set V) : Prop :=
  ∀ ⦃x y⦄, x ∈ H → y ∈ H → ¬G.Adj x y

-- Main theorem stating that any 6-vertex graph has a 3-vertex clique or a 3-vertex independent set
theorem six_vertex_graph_has_3_clique_or_independent_set 
  (V : Type) [Fintype V] [Nonempty V] (G : SimpleGraph V) (hV : Fintype.card V = 6) :
  ∃ (H : Set V), (H.card = 3 ∧ (is_clique G H ∨ is_independent_set G H)) :=
  sorry

end six_vertex_graph_has_3_clique_or_independent_set_l9_9530


namespace area_BCLK_l9_9462

-- Trapezoid ABCD with area 1 and BC:AD = 1:2. K is the midpoint of AC. DK intersects AB at L.

variables {A B C D K L : Type} [metric_space A] [metric_space B] 
  [metric_space C] [metric_space D] [metric_space K] [metric_space L]
  
-- Condition: area of trapezoid ABCD is 1
def area_trapezoid_ABCD (A B C D : Type) [metric_space A] [metric_space B] 
  [metric_space C] [metric_space D] : ℝ := 1

-- Condition: bases BC and AD are in ratio 1:2
def base_ratio (BC AD : ℝ) : Prop := BC = AD / 2

-- Point K is the midpoint of AC
def midpoint (A C K : Type) [metric_space A] [metric_space C] [metric_space K] : Prop := sorry

-- Line DK intersects AB at L
def intersect (D K A B L : Type) [metric_space D] [metric_space K] 
  [metric_space A] [metric_space B] [metric_space L] : Prop := sorry

-- To prove: Area of quadrilateral BCLK is 7/18
theorem area_BCLK (A B C D K L : Type) [metric_space A] [metric_space B] 
  [metric_space C] [metric_space D] [metric_space K] [metric_space L]
  (h1 : area_trapezoid_ABCD A B C D = 1)
  (h2 : base_ratio (dist B C) (dist A D))
  (h3 : midpoint A C K)
  (h4 : intersect D K A B L) :
  area (B C K L) = 7 / 18 :=
sorry

end area_BCLK_l9_9462


namespace combinations_draw_three_slips_l9_9222

theorem combinations_draw_three_slips :
  ∑ k in (Finset.Icc 1 15), 1 = 15 →
  Nat.choose 15 3 = 455 :=
by
  intro h
  rw [Nat.choose_eq_factorial_div_factorials (lt_add_of_pos_right _ (by norm_num : 3 > 0)), Nat.factorial, Nat.factorial, Nat.factorial]
  sorry

end combinations_draw_three_slips_l9_9222


namespace plane_division_99_lines_l9_9608

theorem plane_division_99_lines (m : ℕ) (n : ℕ) : 
  m = 99 ∧ n < 199 → (n = 100 ∨ n = 198) :=
by 
  sorry

end plane_division_99_lines_l9_9608


namespace circumcenter_on_bisector_l9_9053

variables (A B C I B' l : Type)
variables [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq I] [DecidableEq B'] [DecidableEq l]

-- Represents points
variables [HasMem A l] [HasMem B l] [HasMem C l] [HasMem I l] [HasMem B' l]

-- Conditions
axiom h_triangle : is_triangle A B C
axiom h_reflection: is_reflection B B' l
axiom h_angle_bisector: is_angle_bisector l (angle A B C)
axiom h_incenter: is_incenter I (triangle A B C)

-- Prove the circumcenter of the triangle CB'I lies on the line l
theorem circumcenter_on_bisector :
  ∀ (O : Type), is_circumcenter O (triangle C B' I) → O ∈ l := sorry

end circumcenter_on_bisector_l9_9053


namespace green_rotten_fruits_without_smell_l9_9868

-- Definitions of initial conditions in the orchard
def total_apples := 200
def total_oranges := 150
def total_pears := 100

def percent_green_apples := 0.50
def percent_orange_oranges := 0.40
def percent_yellow_oranges := 0.60
def percent_green_pears := 0.30

def percent_rotten_apples := 0.40
def percent_rotten_oranges := 0.25
def percent_rotten_pears := 0.35

def percent_smelly_rotten_apples := 0.70
def percent_smelly_rotten_oranges := 0.50
def percent_smelly_rotten_pears := 0.80

-- Prove that the number of green rotten fruits without a strong smell is 14
theorem green_rotten_fruits_without_smell : 
  let rotten_apples := percent_rotten_apples * total_apples in
  let rotten_oranges := percent_rotten_oranges * total_oranges in
  let rotten_pears := percent_rotten_pears * total_pears in

  let green_rotten_apples := percent_green_apples * rotten_apples in
  let green_rotten_pears := percent_green_pears * rotten_pears in

  let smelly_green_rotten_apples := percent_smelly_rotten_apples * green_rotten_apples in
  let smelly_green_rotten_pears := percent_smelly_rotten_pears * green_rotten_pears in

  let non_smelly_green_rotten_apples := green_rotten_apples - smelly_green_rotten_apples in
  let non_smelly_green_rotten_pears := green_rotten_pears - smelly_green_rotten_pears in

  non_smelly_green_rotten_apples + non_smelly_green_rotten_pears = 14 :=
by sorry

end green_rotten_fruits_without_smell_l9_9868


namespace find_angle_A_find_length_b_l9_9773

-- Define the given conditions as Lean definitions
def angles_of_triangle (A B C : ℝ) : Prop := 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

def lengths_of_sides (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0

def vectors_m_n (A : ℝ) : Prop := 
  let m := (2 * Real.sqrt 3 * Real.sin (A / 2), Real.cos (A / 2) ^ 2)
  let n := (Real.cos (A / 2), -2)
  m.1 * n.1 + m.2 * n.2 = 0

-- Define to find the angle A
theorem find_angle_A (A : ℝ) (h1 : angles_of_triangle A B C) (h2 : vectors_m_n A) : A = Real.pi / 3 := 
  sorry

-- Define to find length of b
theorem find_length_b (a b : ℝ) (A B : ℝ) (h1 : A = Real.pi / 3) (h2 : a = 2) 
  (h3 : Real.cos B = Real.sqrt 3 / 3) (h4 : 0 < B < Real.pi) : b = 4 * Real.sqrt 2 / 3 := 
  sorry


end find_angle_A_find_length_b_l9_9773


namespace find_purchase_price_mobile_l9_9896

-- Define the initial conditions
def purchase_price_grinder := 15000
def sell_price_grinder := purchase_price_grinder - (0.04 * purchase_price_grinder)
def overall_profit := 600

-- Define the function for purchase price of mobile
def purchase_price_mobile (M : ℝ) : ℝ := M

-- Calculate the selling price of mobile
def sell_price_mobile (M : ℝ) := M * 1.15

-- Define the theorem to find the purchase price of the mobile
theorem find_purchase_price_mobile : 
  ∃ (M : ℝ), sell_price_mobile M - M - (purchase_price_grinder - sell_price_grinder) = overall_profit ∧ 
             M = 8000 := 
by 
  sorry

end find_purchase_price_mobile_l9_9896


namespace conjugate_z_is_minus1_plus_i_l9_9785

noncomputable def z : ℂ := (1 + complex.I)^3 / (1 - complex.I)^2
def conjugate_z : ℂ := complex.conj z

theorem conjugate_z_is_minus1_plus_i : conjugate_z = -1 + complex.I := 
sorry

end conjugate_z_is_minus1_plus_i_l9_9785


namespace smallest_possible_value_l9_9358

theorem smallest_possible_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 / (x : ℝ)) + (1 / (y : ℝ)) = 1 / 15) : x + y = 64 :=
sorry

end smallest_possible_value_l9_9358


namespace num_isosceles_triangles_is_24_l9_9634

-- Define the structure of the hexagonal prism
structure HexagonalPrism :=
  (height : ℝ)
  (side_length : ℝ)
  (num_vertices : ℕ)

-- Define the specific hexagonal prism from the problem
def prism := HexagonalPrism.mk 2 1 12

-- Function to count the number of isosceles triangles in a given hexagonal prism
noncomputable def count_isosceles_triangles (hp : HexagonalPrism) : ℕ := sorry

-- The theorem that needs to be proved
theorem num_isosceles_triangles_is_24 :
  count_isosceles_triangles prism = 24 :=
sorry

end num_isosceles_triangles_is_24_l9_9634


namespace maximum_captain_coins_l9_9536

theorem maximum_captain_coins (captain : ℕ) (crew : Fin 5 → ℕ) :
  ∃ (captain_max : ℕ), captain_max = 59 ∧ 
  (∀ (captain + crew.sum = 180) ∧ 
   (∀ i, crew i ≠ crew ((i + 1) % 5) ∧ crew i > crew ((i + 2) % 5) ∧ crew i > crew ((i - 1) % 5)) ∧ 
   (crew.filter (λ c, c > captain)).length ≥ 3) := 
sorry

end maximum_captain_coins_l9_9536


namespace transport_ferrying_items_l9_9416

structure FerryingTrip (Shore : Type) := 
  (boatman : Bool)
  (goat : Bool)
  (cabbage : Bool)
  (wolves : ℕ)
  (dog : Bool)

def initial_shore : FerryingTrip _ :=
  ⟨true, true, true, 2, true⟩

def is_safe (trip : FerryingTrip _) : Prop :=
  (trip.wolves = 0 ∨ ¬trip.goat) ∧ -- no wolf or no goat
  (trip.wolves = 0 ∨ ¬trip.dog) ∧  -- no wolf or no dog
  (¬trip.dog ∨ ¬trip.goat) ∧       -- dog in conflict with goat
  (trip.goat ∨ ¬trip.cabbage)      -- goat not indifferent to cabbage

def transported_to_other_shore (trip : FerryingTrip _) : Prop :=
  trip.boatman ∧
  ¬trip.goat ∧
  ¬trip.cabbage ∧
  trip.wolves = 0 ∧
  ¬trip.dog

theorem transport_ferrying_items (trip : FerryingTrip _) : 
  initial_shore = trip ∧ 
  is_safe trip → 
  transported_to_other_shore trip :=
sorry

end transport_ferrying_items_l9_9416


namespace angle_B_is_150_l9_9757

open Real

variables (A B C a b c : ℝ) (sin_A sin_B sin_C : ℝ)
variables (m n : ℝ × ℝ)

-- Conditions
def sin_A_def : sin_A = sin A := rfl
def sin_B_def : sin_B = sin B := rfl
def sin_C_def : sin_C = sin C := rfl
def m_def : m = (a + b, sin_C) := rfl
def n_def : n = (sqrt 3 * a + c, sin_B - sin_A) := rfl
def parallel_m_n : m.1 * n.2 = m.2 * n.1 := sorry -- \overrightarrow{m} \parallel \overrightarrow{n}

-- Theorem Statement
theorem angle_B_is_150 : B = 5 * π / 6 :=
sorry

end angle_B_is_150_l9_9757


namespace true_propositions_l9_9022

-- Define the lines and plane
variables (a b c : Type) [Line a] [Line b] [Line c] (γ : Type) [Plane γ]

-- Define the propositions
def prop1 := (a ∥ b) ∧ (b ∥ c) → (a ∥ c)
def prop2 := (a ⊥ b) ∧ (b ⊥ c) → (a ⊥ c)
def prop3 := (a ∥ γ) ∧ (b ∥ γ) → (a ∥ b)
def prop4 := (a ⊥ γ) ∧ (b ⊥ γ) → (a ∥ b)

-- The theorem to be proved
theorem true_propositions : prop1 ∧ ¬ prop2 ∧ ¬ prop3 ∧ prop4 := by
  sorry

end true_propositions_l9_9022


namespace value_of_a_is_neg_three_sixteenths_l9_9572

theorem value_of_a_is_neg_three_sixteenths :
  ∃ a : ℚ, (2, -6) ∈ ℚ × ℚ ∧ (-2 * a + 1, 4) ∈ ℚ × ℚ ∧ (3 * a + 2, 3) ∈ ℚ × ℚ ∧ 
    ∀ s1 s2 : ℚ, s1 = (4 - (-6)) / (-2 * a + 1 - 2) ∧ 
    s2 = (3 - (-6)) / (3 * a + 2 - 2) → s1 = s2 → a = -3/16 :=
begin
  sorry
end

end value_of_a_is_neg_three_sixteenths_l9_9572


namespace capital_of_a_l9_9198

variable (P P' TotalCapital Ca : ℝ)

theorem capital_of_a 
  (h1 : a_income_5_percent = (2/3) * P)
  (h2 : a_income_7_percent = (2/3) * P')
  (h3 : a_income_7_percent - a_income_5_percent = 200)
  (h4 : P = 0.05 * TotalCapital)
  (h5 : P' = 0.07 * TotalCapital)
  : Ca = (2/3) * TotalCapital :=
by
  sorry

end capital_of_a_l9_9198


namespace adult_dog_cost_is_100_l9_9465

-- Define the costs for cats, puppies, and dogs.
def cat_cost : ℕ := 50
def puppy_cost : ℕ := 150

-- Define the number of each type of animal.
def number_of_cats : ℕ := 2
def number_of_adult_dogs : ℕ := 3
def number_of_puppies : ℕ := 2

-- The total cost
def total_cost : ℕ := 700

-- Define what needs to be proven: the cost of getting each adult dog ready for adoption.
theorem adult_dog_cost_is_100 (D : ℕ) (h : number_of_cats * cat_cost + number_of_adult_dogs * D + number_of_puppies * puppy_cost = total_cost) : D = 100 :=
by 
  sorry

end adult_dog_cost_is_100_l9_9465


namespace probability_abs_ξ_lt_1_96_l9_9397

-- Random variable ξ following a standard normal distribution N(0,1)
noncomputable def ξ : ℝ → ℝ := sorry

-- Given condition
axiom H1 : ∀ x : ℝ, P(ξ ≤ x) = ∫ y in Iic x, (1 / (sqrt (2 * π))) * exp ((-y ^ 2) / 2) dy

-- Specific given probability
axiom H2 : P(ξ ≤ -1.96) = 0.025

-- The theorem to prove
theorem probability_abs_ξ_lt_1_96 : P(abs ξ < 1.96) = 0.950 :=
by
  sorry -- Proof omitted

end probability_abs_ξ_lt_1_96_l9_9397


namespace larger_number_is_26_l9_9147

theorem larger_number_is_26 {x y : ℤ} 
  (h1 : x + y = 45) 
  (h2 : x - y = 7) : 
  max x y = 26 :=
by
  sorry

end larger_number_is_26_l9_9147


namespace Andy_weight_loss_l9_9257

theorem Andy_weight_loss :
  ∀ (initial_weight : ℕ) (weight_gain : ℕ) (monthly_loss_fraction : ℚ) (months : ℕ),
  initial_weight = 156 →
  weight_gain = 36 →
  monthly_loss_fraction = 1/8 →
  months = 3 →
  let final_weight := (initial_weight + weight_gain) * (1 - monthly_loss_fraction) * months in
  initial_weight - final_weight = 36 :=
by {
  intros,
  simp at *,
  sorry -- Proof can be elaborated here
}

end Andy_weight_loss_l9_9257


namespace focus_of_ellipse_l9_9269

noncomputable def coordinates_of_focus (a b : ℝ) : ℝ × ℝ :=
  let c := (a^2 - b^2).sqrt / a in
  (4 + c, -1)

theorem focus_of_ellipse :
  let a := 4  -- semi-major axis length (half of 8)
  let b := 3  -- semi-minor axis length (half of 6)
  coordinates_of_focus a b = (4 + Real.sqrt 7, -1) := by
  sorry

end focus_of_ellipse_l9_9269


namespace beth_cans_of_corn_l9_9657

theorem beth_cans_of_corn (C P : ℕ) (h1 : P = 2 * C + 15) (h2 : P = 35) : C = 10 :=
by
  sorry

end beth_cans_of_corn_l9_9657


namespace systematic_sampling_student_number_l9_9019

theorem systematic_sampling_student_number
  (total_students : ℕ)
  (sample_size : ℕ)
  (sample_interval : ℕ)
  (sampled_students : set ℕ)
  (h1 : total_students = 36)
  (h2 : sample_size = 4)
  (h3 : sample_interval = total_students / sample_size)
  (h4 : sampled_students = {6, 24, 33})
  (h5 : ∃ n ∈ {1, 2, ..., total_students}, n ∉ sampled_students) :
  ∃ n, n = 15 ∧ n ∈ {1,.., total_students} ∧ n ∉ sampled_students := 
sorry

end systematic_sampling_student_number_l9_9019


namespace perpendicular_EF_angle_bisector_AMD_l9_9878

open EuclideanGeometry

theorem perpendicular_EF_angle_bisector_AMD
  (A B C D M E F : Point)
  (circumcircle_of_ABCD : isCyclicQuadrilateral A B C D)
  (intersection_point_M : Line A C ∩ Line B D = { M })
  (angle_bisectors_CAD_ACB : isAngleBisector (∠CAD) E ∧ isAngleBisector (∠ACB) F)
  (E_on_circumcircle : E ∈ circumcircle quadrilateral A B C D)
  (F_on_circumcircle : F ∈ circumcircle quadrilateral A B C D) :
  isPerpendicular (Line E F) (AngleBisector ∠AMD) :=
by
  sorry

end perpendicular_EF_angle_bisector_AMD_l9_9878


namespace nialls_children_ages_l9_9065

theorem nialls_children_ages : ∃ (a b c d : ℕ), 
  a < 18 ∧ b < 18 ∧ c < 18 ∧ d < 18 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a * b * c * d = 882 ∧ a + b + c + d = 32 :=
by
  sorry

end nialls_children_ages_l9_9065


namespace female_elementary_students_l9_9806

theorem female_elementary_students (total_students : ℕ) (half_students_are_girls : total_students / 2 = 15) (girls_not_in_elementary : 7) : 
  let total_girls := total_students / 2 in
  (total_girls - girls_not_in_elementary = 8) := 
begin
  sorry
end

end female_elementary_students_l9_9806


namespace curve_equation_l9_9403

variables (θ : ℝ) (x y x' y' : ℝ)

-- Definitions based on problem conditions
def C1 : Prop := (x = 2 * Real.cos θ) ∧ (y = 2 * Real.sin θ)
def compression (θ : ℝ) (x y : ℝ) : Prop :=
  (x' = (1 / 4) * x) ∧ (y' = (√3 / 4) * y)

-- The theorem we need to prove
theorem curve_equation (θ : ℝ) (x y x' y' : ℝ) (h1 : C1 θ x y) (h2 : compression θ x y x' y') :
  4 * x'^2 + (4 * y'^2 / 3) = 1 :=
sorry

end curve_equation_l9_9403


namespace problem_solution_l9_9421

def valid_digits (x : ℕ) : Prop :=
  x ≠ 4 ∧ (∃ d ∈ [7], d)

def is_valid_number (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000 ∧ valid_digits (n % 10) ∧ valid_digits ((n / 10) % 10) ∧ valid_digits (n / 100)

def count_valid_numbers : ℕ :=
  (List.range' 100 900).countp is_valid_number

theorem problem_solution : count_valid_numbers = 200 := 
  sorry

end problem_solution_l9_9421


namespace sqrt_sum_inequality_ratio_inequality_l9_9603

theorem sqrt_sum_inequality : sqrt 5 + sqrt 7 > 1 + sqrt 13 := sorry

theorem ratio_inequality {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy: x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := sorry

end sqrt_sum_inequality_ratio_inequality_l9_9603


namespace part1_part2_l9_9886

variable (A B : ℝ)
variable (a b : ℝ)
variable (sin cos : ℝ -> ℝ)

theorem part1 (h : sin (A - B) = (a / (a + b)) * sin A * cos B - (b / (a + b)) * sin B * cos A) : A = B :=
sorry

noncomputable def area_triangle (A : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  let B := A in
  let c := 2 * a * cos A in
  (1 / 2) * b * c * sin A

theorem part2 (A : ℝ) (a : ℝ) (hA : A = (7 * Real.pi) / 24) (ha : a = Real.sqrt 6) : 
  area_triangle A a a = (3 * (Real.sqrt 2 + Real.sqrt 6)) / 4 :=
sorry

end part1_part2_l9_9886


namespace determine_values_of_x_l9_9000

theorem determine_values_of_x (x : ℝ) : log 10 (2 * x^2 - 5 * x + 10) = 2 ↔ x = (5 + sqrt 745) / 4 ∨ x = (5 - sqrt 745) / 4 :=
by sorry

end determine_values_of_x_l9_9000


namespace contest_end_time_l9_9616

-- Definitions for the conditions
def start_time_pm : Nat := 15 -- 3:00 p.m. in 24-hour format
def duration_min : Nat := 720

-- Proof that the contest ended at 3:00 a.m.
theorem contest_end_time :
  let end_time := (start_time_pm + (duration_min / 60)) % 24
  end_time = 3 :=
by
  -- This would be the place to provide the proof
  sorry

end contest_end_time_l9_9616


namespace larger_number_is_26_l9_9148

theorem larger_number_is_26 {x y : ℤ} 
  (h1 : x + y = 45) 
  (h2 : x - y = 7) : 
  max x y = 26 :=
by
  sorry

end larger_number_is_26_l9_9148


namespace rectangle_area_l9_9865

noncomputable def area_of_rectangle : ℝ := 
  let x1 := -3
  let y1 := 1
  let x2 := 1
  let y2 := -2
  -- Calculate length and width
  let length := x2 - x1
  let width := y1 - y2
  -- Calculate area
  length * width

theorem rectangle_area : area_of_rectangle = 12 := by
  -- Definitions
  let x1 := -3
  let y1 := 1
  let x2 := 1
  let y2 := -2
  -- Calculations
  let length := x2 - x1
  let width := y1 - y2
  -- Proof
  calc
    area_of_rectangle = length * width : by rfl
                   ... = (x2 - x1) * (y1 - y2) : by simp [length, width]
                   ... = (1 - (-3)) * (1 - (-2)) : by simp
                   ... = 4 * 3 : by simp
                   ... = 12 : by norm_num

end rectangle_area_l9_9865


namespace solve_quadratic_equation_l9_9537

theorem solve_quadratic_equation (x : ℝ) :
  (x^2 - 2 * x - 5 = 0) ↔ (x = 1 + Real.sqrt 6 ∨ x = 1 - Real.sqrt 6) := 
sorry

end solve_quadratic_equation_l9_9537


namespace michael_pets_total_l9_9923

theorem michael_pets_total (p : ℕ) (h1 : 0.25 * p = 9) : p = 36 :=
by
  sorry

end michael_pets_total_l9_9923


namespace at_least_one_not_less_than_2_l9_9400

theorem at_least_one_not_less_than_2 (x y z : ℝ) (hp : 0 < x ∧ 0 < y ∧ 0 < z) :
  let a := x + 1/y
  let b := y + 1/z
  let c := z + 1/x
  (a ≥ 2 ∨ b ≥ 2 ∨ c ≥ 2) := by
    sorry

end at_least_one_not_less_than_2_l9_9400


namespace find_f_lg_lg3_l9_9213

theorem find_f_lg_lg3 (a b : ℝ) (h g : ℝ → ℝ) (f : ℝ → ℝ)
  (h_odd : ∀ x, h (-x) = - (h x))
  (g_odd : ∀ x, g (-x) = - (g x))
  (h_lin_comb : f = λ x, a * h x + b * g x + 4)
  (not_both_zero : ¬ (a = 0 ∧ b = 0))
  (lg_log3_10 : ℝ)
  (lg_log3_3 : ℝ)
  (f_lg_log3_10_eq_5 : f lg_log3_10 = 5)
  (lg_log3_10_def : lg_log3_10 = math.log (10) / math.log (2) / math.log (3))
  (lg_log3_3_def : lg_log3_3 = math.log (math.log (3)) / math.log (2)) :
  f lg_log3_3 = 3 :=
by
  sorry

end find_f_lg_lg3_l9_9213


namespace problem_solution_l9_9422

def valid_digits (x : ℕ) : Prop :=
  x ≠ 4 ∧ (∃ d ∈ [7], d)

def is_valid_number (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000 ∧ valid_digits (n % 10) ∧ valid_digits ((n / 10) % 10) ∧ valid_digits (n / 100)

def count_valid_numbers : ℕ :=
  (List.range' 100 900).countp is_valid_number

theorem problem_solution : count_valid_numbers = 200 := 
  sorry

end problem_solution_l9_9422


namespace larger_number_l9_9146

theorem larger_number (x y : ℝ) (h₁ : x + y = 45) (h₂ : x - y = 7) : x = 26 :=
by
  sorry

end larger_number_l9_9146


namespace find_logarithm_l9_9410

-- Definitions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 1

-- Theorem statement
theorem find_logarithm
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 2 + a 4 + a 6 = 18) :
  log 3 (a 5 + a 7 + a 9) = 3 :=
sorry

end find_logarithm_l9_9410


namespace angle_OBC_eq_angle_ODC_l9_9890

variables {A B C D O : Type*} [parallelogram ABCD]
variable (H1 : ∀ {A B C D O}, angle A O D = angle O C D)

theorem angle_OBC_eq_angle_ODC : angle O B C = angle O D C := 
sorry

end angle_OBC_eq_angle_ODC_l9_9890


namespace problem_l9_9428

-- Define the operations as given in the conditions.
def star (A B : ℝ) : ℝ := (A + B) / 2
def hash (A : ℝ) : ℝ := A * A + 1

-- State the problem as a theorem to be proven.
theorem problem : star (star (hash 4) 6) 9 = 10.25 :=
by
  -- proof can be filled in here
  sorry

end problem_l9_9428


namespace side_lengths_not_unique_l9_9444

variables {A B C D E : Type} [metric_space A] [dim 2 A]
variables (a b c d e : ℝ)
variables (BD DE EC : ℝ)
variables (φ : tripartition (angle BAC))

-- Given Conditions
def conditions (triangle_ABC : triangle A B C)
  (D E : A)
  (trisectBAC : ∀ D E, tripartition (angle BAC) D E)
  (BD_eq : BD = 5)
  (DE_eq : DE = 2)
  (EC_eq : EC = 8)
  (sideLengths : (BD, DE, EC)) : Prop :=
  (BD = 5) ∧ (DE = 2) ∧ (EC = 8)

-- Equivalent Mathematical Problem
theorem side_lengths_not_unique :
  ∀ (triangle_ABC : triangle A B C)
    (D E : A)
    (trisectBAC : ∀ D E, tripartition (angle BAC) D E)
    (BD_eq : BD = 5)
    (DE_eq : DE = 2)
    (EC_eq : EC = 8),
  ¬(∃ b c : ℝ, AC = b ∧ AB = c → unique b c) :=
begin
  sorry
end

end side_lengths_not_unique_l9_9444


namespace dimes_distribution_l9_9268

open Real

theorem dimes_distribution :
  let barry_dimes := 100 in
  let dan_dimes_initial := barry_dimes / 2 in
  let dan_dimes := dan_dimes_initial + 2 in
  let emily_dimes := 2 * dan_dimes_initial in
  let frank_dimes := emily_dimes - 7 in
    barry_dimes = 100 ∧ dan_dimes = 52 ∧ emily_dimes = 100 ∧ frank_dimes = 93 :=
by
  sorry

end dimes_distribution_l9_9268


namespace perfect_squares_count_l9_9419

theorem perfect_squares_count : 
  (finset.filter (λ x : ℕ, (x < 10000) ∧ ((x % 10 = 4) ∨ (x % 10 = 5) ∨ (x % 10 = 6))) 
    (finset.range 10000)).card = 50 :=
sorry

end perfect_squares_count_l9_9419


namespace equation_of_line_through_point_with_angle_l9_9958

open Real

noncomputable def line_equation (x y : ℝ) : Prop := x + y - 3 = 0

def point : ℝ × ℝ := (2, 1)

def inclination_angle : ℝ := 135

def tan_135_deg := Real.tan (135 * π / 180)

lemma inclination_tan : tan_135_deg = -1 :=
by sorry

theorem equation_of_line_through_point_with_angle {x y : ℝ} 
  (hx : point.1 = 2) (hy : point.2 = 1) 
  (hangle : inclination_angle = 135) 
  (htan : tan_135_deg = -1):
  line_equation x y :=
by sorry

end equation_of_line_through_point_with_angle_l9_9958


namespace period_tan_3x_l9_9718

theorem period_tan_3x : 
  (∀ T, T > 0 ∧ ∀ x, tan(x + T) = tan(x) ↔ T = π) →
  ∃ T', T' > 0 ∧ ∀ x, tan(3 * x + T') = tan(3 * x) ∧ T' = π / 3 :=
by
  sorry

end period_tan_3x_l9_9718


namespace sum_first_six_terms_of_geometric_sequence_l9_9329

open_locale classical

-- Definition of the problem conditions
def is_geometric_sequence (a : ℕ → ℤ) (r : ℤ) : Prop :=
  ∀ (n : ℕ), a (n + 1) = a n * r

-- The sum of the first n terms of a geometric sequence
def sum_geometric_sequence (a : ℕ → ℤ) (r : ℤ) (n : ℕ) : ℚ :=
  if r = 1 then n * a 0 else
  a 0 * (1 - r^(n + 1)) / (1 - r)

-- The main proof statement
theorem sum_first_six_terms_of_geometric_sequence (a : ℕ → ℤ) (r : ℤ)
  (h_geom : is_geometric_sequence a r)
  (h_a5 : a 4 = -2) (h_a8 : a 7 = 16) :
  sum_geometric_sequence a r 5 = 21 / 8 :=
sorry

end sum_first_six_terms_of_geometric_sequence_l9_9329


namespace sum_of_three_squares_as_sum_of_four_fractions_l9_9529

theorem sum_of_three_squares_as_sum_of_four_fractions (A B C x y z : ℤ) : 
  let a := x^2 + y^2 - z^2,
      b := 2 * x * z,
      c := 2 * y * z,
      N := A^2 + B^2 + C^2 in
  N = (A*a + B*b + C*c) ^ 2 / (x^2 + y^2 + z^2)^2 +
      (A*b - B*a) ^ 2 / (x^2 + y^2 + z^2)^2 +
      (B*c - C*b) ^ 2 / (x^2 + y^2 + z^2)^2 +
      (C*a - A*c) ^ 2 / (x^2 + y^2 + z^2)^2 := 
sorry

end sum_of_three_squares_as_sum_of_four_fractions_l9_9529


namespace probability_at_least_one_6_l9_9087

-- Define the probability of getting at least one 6 when a die is rolled twice
theorem probability_at_least_one_6 (total_outcomes favorable_outcomes : ℕ) (h1 : total_outcomes = 36)
  (h2 : favorable_outcomes = 11) :
  favorable_outcomes.to_rat / total_outcomes.to_rat = 11 / 36 := 
by
  sorry

end probability_at_least_one_6_l9_9087


namespace general_term_formula_T_n_value_l9_9341

-- Define the given sequence and its sum formula
def sequence_sum (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  -a n - (1/2)^(n-1) + 2

-- Define the term for the sequence
def sequence_term (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  match n with
  | 0 => 0
  | 1 => a 1
  | n => a n

-- Prove the general term formula
theorem general_term_formula (a : ℕ → ℝ) (n : ℕ) (h : ∀ m, m > 0 → sequence_sum m a = -a m - (1/2)^(m-1) + 2) :
  a n = n / 2^n := 
sorry

-- Define c_n and T_n as given in the problem
def c (n : ℕ) (a : ℕ → ℝ) : ℝ :=
(n + 1) / n * a n

def T (n : ℕ) (a : ℕ → ℝ) : ℝ :=
∑ i in (Finset.range n).map nat.succ, c i a

-- Prove the value of T_n
theorem T_n_value (a : ℕ → ℝ) (n : ℕ) (h : ∀ m, m > 0 → sequence_sum m a = -a m - (1/2)^(m-1) + 2) :
  T n a = 3 - (n + 3) / 2^n :=
sorry

end general_term_formula_T_n_value_l9_9341


namespace magic_square_possible_l9_9114

def is_magic_square (square : List (List ℕ)) : Prop :=
  let sums := (List.map List.sum square)
  let rows_valid := sums.All (fun s => s = sums.head)
  let cols := List.map (fun j => List.sum (List.map (fun i => square.nth.getOrElse i [][].nth.getOrElse j 0) [0, 1, 2])) [0, 1, 2]
  let cols_valid := cols.All (fun s => s = sums.head)
  let diag1 := List.sum (List.map (fun i => square.nth.getOrElse i [][].nth.getOrElse i 0) [0, 1, 2])
  let diag2 := List.sum (List.map (fun i => square.nth.getOrElse i [][].nth.getOrElse (2 - i) 0) [0, 1, 2])
  rows_valid ∧ cols_valid ∧ diag1 = sums.head ∧ diag2 = sums.head

theorem magic_square_possible :
  ∃ (square : List (List ℕ)), 
    (∀ i j, i < 3 → j < 3 → square.nth.getOrElse i [][].nth.getOrElse j 0 ∈ [820, 2420, 4020, 5620, 7220, 8820, 10420, 12020, 13620]) 
    ∧ is_magic_square square := 
begin
  -- The proof should go here, but it's omitted as per the requirements.
  sorry
end

end magic_square_possible_l9_9114


namespace jello_cost_calculation_l9_9474

-- Conditions as definitions
def jello_per_pound : ℝ := 1.5
def tub_volume_cubic_feet : ℝ := 6
def cubic_foot_to_gallons : ℝ := 7.5
def gallon_weight_pounds : ℝ := 8
def cost_per_tablespoon_jello : ℝ := 0.5

-- Tub total water calculation
def tub_water_gallons (volume_cubic_feet : ℝ) (cubic_foot_to_gallons : ℝ) : ℝ :=
  volume_cubic_feet * cubic_foot_to_gallons

-- Water weight calculation
def water_weight_pounds (water_gallons : ℝ) (gallon_weight_pounds : ℝ) : ℝ :=
  water_gallons * gallon_weight_pounds

-- Jello mix required calculation
def jello_mix_tablespoons (water_pounds : ℝ) (jello_per_pound : ℝ) : ℝ :=
  water_pounds * jello_per_pound

-- Total cost calculation
def total_cost (jello_mix_tablespoons : ℝ) (cost_per_tablespoon_jello : ℝ) : ℝ :=
  jello_mix_tablespoons * cost_per_tablespoon_jello

-- Theorem statement
theorem jello_cost_calculation :
  total_cost (jello_mix_tablespoons (water_weight_pounds (tub_water_gallons tub_volume_cubic_feet cubic_foot_to_gallons) gallon_weight_pounds) jello_per_pound) cost_per_tablespoon_jello = 270 := 
by sorry

end jello_cost_calculation_l9_9474


namespace guests_did_not_come_l9_9270

theorem guests_did_not_come 
  (total_cookies : ℕ) 
  (prepared_guests : ℕ) 
  (cookies_per_guest : ℕ) 
  (total_cookies_eq : total_cookies = 18) 
  (prepared_guests_eq : prepared_guests = 10)
  (cookies_per_guest_eq : cookies_per_guest = 18) 
  (total_cookies_computation : total_cookies = cookies_per_guest) :
  prepared_guests - total_cookies / cookies_per_guest = 9 :=
by
  sorry

end guests_did_not_come_l9_9270


namespace Humphrey_birds_l9_9059

-- Definitions for the given conditions:
def Marcus_birds : ℕ := 7
def Darrel_birds : ℕ := 9
def average_birds : ℕ := 9
def number_of_people : ℕ := 3

-- Proof statement
theorem Humphrey_birds : ∀ x : ℕ, (average_birds * number_of_people = Marcus_birds + Darrel_birds + x) → x = 11 :=
by
  intro x h
  sorry

end Humphrey_birds_l9_9059


namespace f_x_minus_one_l9_9008

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 5

theorem f_x_minus_one (x : ℝ) : f (x - 1) = x^2 - 4 * x + 8 :=
by
  sorry

end f_x_minus_one_l9_9008


namespace prism_cross_section_properties_l9_9990

noncomputable def prism_base_triangle_area (side : ℝ) (height : ℝ) : ℝ :=
  (sqrt 3 / 4) * side^2

noncomputable def cross_section_area (side : ℝ) (height : ℝ) : ℝ :=
  side^2 * height / 2

theorem prism_cross_section_properties :
  let side := 6
  let height := 1/3 * sqrt 7
  let angle_between_planes := 30
  cross_section_area side height = 39/4 ∧ angle_between_planes = 30 :=
by
  sorry

end prism_cross_section_properties_l9_9990


namespace part1_part2_part3_l9_9943

-- Part 1: Simplifying the Expression
theorem part1 (a b : ℝ) : 
  3 * (a - b)^2 - 6 * (a - b)^2 + 2 * (a - b)^2 = -(a - b)^2 :=
by sorry

-- Part 2: Finding the Value of an Expression
theorem part2 (x y : ℝ) (h : x^2 - 2 * y = 4) : 
  3 * x^2 - 6 * y - 21 = -9 :=
by sorry

-- Part 3: Evaluating a Compound Expression
theorem part3 (a b c d : ℝ) (h1 : a - 2 * b = 6) (h2 : 2 * b - c = -8) (h3 : c - d = 9) : 
  (a - c) + (2 * b - d) - (2 * b - c) = 7 :=
by sorry

end part1_part2_part3_l9_9943


namespace sin_sum_identity_l9_9738

theorem sin_sum_identity :
  sin (π / 4) * sin (7 * π / 12) + sin (π / 4) * sin (π / 12) = sqrt 3 / 2 :=
by
  sorry

end sin_sum_identity_l9_9738


namespace basis_non_collinear_A_l9_9590

theorem basis_non_collinear_A :
  ¬(let e1 := (-1, 2)
     let e2 := (5, 7)
     (e1.1 * e2.2 - e2.1 * e1.2 = 0)) :=
by {
  let e1 := (-1, 2)
  let e2 := (5, 7)
  have h : e1.1 * e2.2 - e2.1 * e1.2 ≠ 0,
  { calc
    e1.1 * e2.2 - e2.1 * e1.2
        = -1 * 7 - 5 * 2 : by rw [int.mul_comm] 
    ... = -7 - 10 : by rw [int.mul_comm]
    ... = -17 : by norm_num },
  exact h
}

end basis_non_collinear_A_l9_9590


namespace geom_seq_num_ordered_pairs_eq_1_l9_9125

noncomputable def num_ordered_pairs (a r : ℕ) : ℕ :=
  if a > 0 ∧ r > 0 ∧ (log 4 (a * (r^(0 : ℕ)) * (a * (r^(1 : ℕ))) * (a * (r^(2 : ℕ))) * (a * (r^(3 : ℕ))) * (a * (r^(4 : ℕ))) * (a * (r^(5 : ℕ))) * (a * (r^(6 : ℕ))) * (a * (r^(7 : ℕ))) * (a * (r^(8 : ℕ))) * (a * (r^(9 : ℕ)))) = 1808)
  then 1 else 0

theorem geom_seq_num_ordered_pairs_eq_1 : 
  (count_ordered_pairs : ℕ ) (num_ordered_pairs (2^66) (2^52)) = 1 :=
by
  sorry

end geom_seq_num_ordered_pairs_eq_1_l9_9125


namespace proof_equivalent_problem_l9_9778

def g : ℝ → ℝ := sorry

theorem proof_equivalent_problem :
  g 8 = 6 → ((3 * (5 / 3)) = (g (3 * (8 / 3)) / 3 + 3)) :=
by
  intro h1
  calc
    3 * (5 / 3) = 5 : by rfl
    ... = (6 / 3 + 3) : by rw [h1]
    ... = (g (3 * (8 / 3)) / 3 + 3) : by sorry

end proof_equivalent_problem_l9_9778


namespace not_divisible_by_3_product_units_tens_l9_9932

/-- The product of the units digit and the tens digit of a specific 4-digit number which is not divisible by 3 from the list {4621, 4631, 4641, 4651, 4661} is 3. -/
theorem not_divisible_by_3_product_units_tens (n : ℕ) (h1 : n ∈ {4621, 4631, 4641, 4651, 4661}) (h2 : ¬ (n % 3 = 0)) : 
  let d_units := n % 10,
      d_tens := (n / 10) % 10
  in d_units * d_tens = 3 := 
by
    sorry

end not_divisible_by_3_product_units_tens_l9_9932


namespace perimeter_of_flowerbed_l9_9631

def width : ℕ := 4
def length : ℕ := 2 * width - 1
def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

theorem perimeter_of_flowerbed : perimeter length width = 22 := by
  sorry

end perimeter_of_flowerbed_l9_9631


namespace combined_moment_l9_9418

-- Definitions based on given conditions
variables (P Q Z : ℝ) -- Positions of the points and center of mass
variables (p q : ℝ) -- Masses of the points
variables (Mom_s : ℝ → ℝ) -- Moment function relative to axis s

-- Given:
-- 1. Positions P and Q with masses p and q respectively
-- 2. Combined point Z with total mass p + q
-- 3. Moments relative to the axis s: Mom_s P and Mom_s Q
-- To Prove: Moment of the combined point Z relative to axis s
-- is the sum of the moments of P and Q relative to the same axis

theorem combined_moment (hZ : Z = (P * p + Q * q) / (p + q)) :
  Mom_s Z = Mom_s P + Mom_s Q :=
sorry

end combined_moment_l9_9418


namespace sin_neg_045_unique_solution_l9_9425

theorem sin_neg_045_unique_solution (x : ℝ) (hx : 0 ≤ x ∧ x < 180) (h: ℝ) :
  (h = Real.sin x → h = -0.45) → 
  ∃! x, 0 ≤ x ∧ x < 180 ∧ Real.sin x = -0.45 :=
by sorry

end sin_neg_045_unique_solution_l9_9425


namespace total_alligators_eaten_is_correct_l9_9436

-- Define conditions
def P1_eats_per_week := 1     -- P1 eats 1 alligator per week
def P2_eats_per_5_days := 1   -- P2 eats 1 alligator every 5 days
def P3_eats_per_10_days := 1  -- P3 eats 1 alligator every 10 days
def days_in_3_weeks := 21     -- Total period is 21 days (3 weeks)

-- Define the number of alligators eaten by each python in 21 days
def P1_eats_in_3_weeks := P1_eats_per_week * 3
def P2_eats_in_3_weeks := 21 / 5
def P3_eats_in_3_weeks := 21 / 10

-- The total number of alligators is the sum of alligators eaten by all pythons
def total_alligators_eaten := (P1_eats_in_3_weeks) + Int.floor (P2_eats_in_3_weeks) + Int.floor (P3_eats_in_3_weeks)

-- Statement to prove
theorem total_alligators_eaten_is_correct : total_alligators_eaten = 9 :=
by
  sorry

end total_alligators_eaten_is_correct_l9_9436


namespace time_spent_on_Type_A_problems_l9_9835

theorem time_spent_on_Type_A_problems (t : ℝ) (h1 : 25 * (8 * t) + 100 * (2 * t) = 120) : 
  25 * (8 * t) = 60 := by
  sorry

-- Conditions
-- t is the time spent on a Type C problem in minutes
-- 25 * (8 * t) + 100 * (2 * t) = 120 (time spent on Type A and B problems combined equals 120 minutes)

end time_spent_on_Type_A_problems_l9_9835


namespace find_a_10_l9_9755

def seq (a : ℕ → ℚ) : Prop :=
∀ n, a (n + 1) = 2 * a n / (a n + 2)

def initial_value (a : ℕ → ℚ) : Prop :=
a 1 = 1

theorem find_a_10 (a : ℕ → ℚ) (h1 : initial_value a) (h2 : seq a) : 
  a 10 = 2 / 11 := 
sorry

end find_a_10_l9_9755


namespace probability_allison_wins_l9_9646

theorem probability_allison_wins :
  let allison_roll := 7
  let brian_roll : ℕ := 1 ∨ 2 ∨ 3 ∨ 4 ∨ 5 ∨ 6
  let noah_roll : ℕ := 3 ∨ 3 ∨ 5 ∨ 5 ∨ 5 ∨ 5
  (1/1 * 1/1 = 1) :=
by
  sorry

end probability_allison_wins_l9_9646


namespace mr_bird_speed_to_be_on_time_l9_9924

theorem mr_bird_speed_to_be_on_time 
  (d : ℝ) 
  (t : ℝ)
  (h1 : d = 40 * (t + 1/20))
  (h2 : d = 60 * (t - 1/20)) :
  (d / t) = 48 :=
by
  sorry

end mr_bird_speed_to_be_on_time_l9_9924


namespace max_captain_coins_is_59_l9_9533

noncomputable def max_coins_for_captain (a b c d e f : ℕ) : ℕ :=
if h : a + b + c + d + e + f = 180 ∧
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0 ∧
  a + b + c + d + e + f = 180 ∧
  ((b > a ∧ b > c) +
   (c > b ∧ c > d) +
   (d > c ∧ d > e) +
   (e > d ∧ e > f) +
   (f > e ∧ f > a) ≥ 3)  then a else 0

theorem max_captain_coins_is_59 :
  ∃ a b c d e f : ℕ, max_coins_for_captain a b c d e f = 59 :=
begin
  use [59, 60, 1, 60, 0, 60],
  unfold max_coins_for_captain,
  split,
  { norm_num },
  { split; norm_num },
  { split; norm_num, split; norm_num, norm_num },
end

end max_captain_coins_is_59_l9_9533


namespace part1_part2_l9_9903

def S_n (a : ℕ → ℝ) (n : ℕ) : ℝ := (Finset.range (n + 1)).sum (λ i, a i)

theorem part1 (a : ℕ → ℝ) (h1 : ∀ n, a n > 0) 
              (h2 : ∀ n, a n ^ 2 + 2 * a n = 4 * S_n a n + 3) :
  ∀ n, a n = 2 * n + 1 :=
sorry

theorem part2 (a : ℕ → ℝ) (b : ℕ → ℝ)
              (h1 : ∀ n, a n > 0) 
              (h2 : ∀ n, a n ^ 2 + 2 * a n = 4 * S_n a n + 3) 
              (h3 : ∀ n, b n = 1 / (a n * a (n + 1))) :
  ∀ n, (Finset.range n).sum b = 1 / 6 - 1 / (4 * n + 6) :=
sorry

end part1_part2_l9_9903


namespace shares_own_if_dividend_paid_l9_9224

noncomputable def numSharesDividendPaid (dividend paid earningsPerShare expectedEarningsPerShare additionalDividendPerAdditionalEarnings : ℝ) : ℝ :=
dividendPaid / (expectedEarningsPerShare / 2 + additionalDividendPerAdditionalEarnings * (earningsPerShare - expectedEarningsPerShare) / (10 : ℝ) * 4)

theorem shares_own_if_dividend_paid
  (dividendPaid : ℝ)
  (expectedEarningsPerShare : ℝ)
  (earningsPerShare : ℝ)
  (additionalDividendPerAdditionalEarnings : ℝ)
  (h_expectedEarnings : expectedEarningsPerShare = 0.80)
  (h_dividendHalf : dividendPaid / expectedEarningsPerShare = 2.0)
  (h_earnings : earningsPerShare = 1.10)
  (h_additionalDividend : additionalDividendPerAdditionalEarnings = 0.04)
  : numSharesDividendPaid dividendPaid earningsPerShare expectedEarningsPerShare additionalDividendPerAdditionalEarnings = 600 :=
by sorry

end shares_own_if_dividend_paid_l9_9224


namespace measure_angle_AEC_l9_9024

theorem measure_angle_AEC 
  (angle_ABE'_supplementary : ∠ABE' + ∠ABE = 180)
  (angle_ABE'_given : ∠ABE' = 150)
  (angle_BAE_given : ∠BAE = 108)
  (triangle_sum_angles : ∀ A B C : Finset.Point, ∠BAE + ∠ABE + ∠AEC = 180) :
  ∠AEC = 42 := 
by
  sorry

end measure_angle_AEC_l9_9024


namespace heads_before_consecutive_tails_l9_9910

theorem heads_before_consecutive_tails:
  let p := (3 / 34 : ℚ),
      m := 3,
      n := 34
  in p = (m / n) ∧ (Nat.gcd m n = 1) ∧ (m + n = 37) :=
by
  sorry

end heads_before_consecutive_tails_l9_9910


namespace original_selling_price_l9_9199

variable (P : ℝ)
variable (S : ℝ) 

-- Conditions
axiom profit_10_percent : S = 1.10 * P
axiom profit_diff : 1.17 * P - S = 42

-- Goal
theorem original_selling_price : S = 660 := by
  sorry

end original_selling_price_l9_9199


namespace midsegment_length_of_trapezoid_l9_9412

theorem midsegment_length_of_trapezoid (a b c d : ℝ) (h1 : a > b) (h2 : ∃ (x y z w : ℝ), true) : 
  (∃ x y, 1 = 1) → (∃ x y z w, true) → ((a + b) / 2 = (a + b) / 2) := 
by
  sorry

end midsegment_length_of_trapezoid_l9_9412


namespace smallest_sum_of_inverses_l9_9371

theorem smallest_sum_of_inverses 
  (x y : ℕ) (hx : x ≠ y) (h1 : 0 < x) (h2 : 0 < y) (h_condition : (1 / x : ℚ) + 1 / y = 1 / 15) :
  x + y = 64 := 
sorry

end smallest_sum_of_inverses_l9_9371


namespace smartphone_cost_l9_9510

theorem smartphone_cost :
  let current_savings : ℕ := 40
  let weekly_saving : ℕ := 15
  let num_months : ℕ := 2
  let weeks_in_month : ℕ := 4 
  let total_weeks := num_months * weeks_in_month
  let total_savings := weekly_saving * total_weeks
  let total_money := current_savings + total_savings
  total_money = 160 := by
  sorry

end smartphone_cost_l9_9510


namespace solve_problem_l9_9391

def f (x : ℝ) : ℝ := x^2 - 4*x + 7
def g (x : ℝ) : ℝ := 2*x + 1

theorem solve_problem : f (g 3) - g (f 3) = 19 := by
  sorry

end solve_problem_l9_9391


namespace negation_example_l9_9112

theorem negation_example :
  (¬ (∀ x: ℝ, x > 0 → x^2 + x + 1 > 0)) ↔ (∃ x: ℝ, x > 0 ∧ x^2 + x + 1 ≤ 0) :=
by
  sorry

end negation_example_l9_9112


namespace sin_cos_difference_sin_cos_quotient_l9_9211

variable (α : ℝ)

-- Problem 1
theorem sin_cos_difference (h1 : sin α + cos α = 4 / 5) (h2 : 0 < α) (h3 : α < π) : sin α - cos α = sqrt 34 / 5 := sorry

-- Problem 2
theorem sin_cos_quotient (h1 : tan α = 2) : (2 * sin α - cos α) / (sin α + 3 * cos α) = 3 / 5 := sorry

end sin_cos_difference_sin_cos_quotient_l9_9211


namespace S4_equals_15_l9_9860

noncomputable def S_n (q : ℝ) (n : ℕ) := (1 - q^n) / (1 - q)

theorem S4_equals_15 (q : ℝ) (n : ℕ) (h1 : S_n q 1 = 1) (h2 : S_n q 5 = 5 * S_n q 3 - 4) : 
  S_n q 4 = 15 :=
by
  sorry

end S4_equals_15_l9_9860


namespace least_a_condition_exists_l9_9713

-- Define the point being inside the square
def point_in_square (P : ℝ × ℝ) (ABCD : set (ℝ × ℝ)) : Prop :=
  P ∈ interior ABCD

-- Define the areas of triangles PAB, PBC, PCD, PDA
variables (A₁ A₂ A₃ A₄ : ℝ)

-- Assumption: The sum of the areas of the triangles is equal to 1
def sum_of_areas : Prop := A₁ + A₂ + A₃ + A₄ = 1

-- Define the condition for the area ratio lying within the desired interval
def area_ratio_condition (a : ℝ) : Prop :=
  ∀ {A₁ A₂ A₃ A₄ : ℝ}, (A₁ ≠ 0) → (A₄ ≠ 0) → sum_of_areas A₁ A₂ A₃ A₄ →
  A₁ / A₄ ∈ set.Icc (1 / a) a

-- The main proof goal: Find the least a > 1
theorem least_a_condition_exists :
  ∃ a > 1, area_ratio_condition a :=
begin
  use ((1 + (real.sqrt 5)) / 2),
  split,
  { -- Prove that (1 + sqrt 5)/2 > 1
    linarith [real.sqrt_pos.2 (by norm_num : 5 > 0)],
  },
  { -- Prove the area ratio condition
    sorry
  }
end

end least_a_condition_exists_l9_9713


namespace derivative_at_1_l9_9333

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem derivative_at_1 : (deriv f 1) = 2 * Real.exp 1 := by
  sorry

end derivative_at_1_l9_9333


namespace find_d_l9_9627

def point_in_square (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 3030 ∧ 0 ≤ y ∧ y ≤ 3030

def point_in_ellipse (x y : ℝ) : Prop :=
  (x^2 / 2020^2) + (y^2 / 4040^2) ≤ 1

def point_within_distance (d : ℝ) (x y : ℝ) : Prop :=
  (∃ (a b : ℤ), (x - a) ^ 2 + (y - b) ^ 2 ≤ d ^ 2)

theorem find_d :
  (∃ d : ℝ, (∀ x y : ℝ, point_in_square x y → point_in_ellipse x y → point_within_distance d x y) ∧ (d = 0.5)) :=
by
  sorry

end find_d_l9_9627


namespace intervals_of_monotonicity_range_of_m_for_zero_l9_9771

section
variable (m : ℝ) (f g : ℝ → ℝ)

-- Define the functions f and g
def f (x : ℝ) := (2 * m / 3) * x^3 + x^2 - 3 * x - m * x + 2
def g (x : ℝ) := (f' x)

-- The first part of the proof deals with monotonicity of f when m = 1
theorem intervals_of_monotonicity (x : ℝ) (h : m = 1) :
  (f' x > 0 ↔ x > 1 ∨ x < -2) ∧
  (f' x < 0 ↔ -2 < x ∧ x < 1) :=
sorry

-- The second part of the proof deals with the range of m for which g has a zero in [-1, 1]
theorem range_of_m_for_zero (h : ∃ x ∈ Icc (-1 : ℝ) 1, g x = 0) :
  m ∈ Icc (-∞) ((-3 - sqrt 7) / 2) ∪ Icc (1) (∞) :=
sorry

end

end intervals_of_monotonicity_range_of_m_for_zero_l9_9771


namespace area_of_triangle_l9_9541

theorem area_of_triangle (A B C P : ℝ) (h1 : right_triangle A B C)
  (h2 : P_on_hypotenuse P A C) (h3 : angle A B P = 60) (h4 : dist A P = 2) (h5 : dist C P = 3) :
  area A B C = 25 * real.sqrt 3 / 8 :=
by sorry

end area_of_triangle_l9_9541


namespace minimum_value_of_f_l9_9960

open Real

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := 3 * sqrt 2 * cos (x + φ) + sin x

theorem minimum_value_of_f (φ : ℝ) (hφ : -π / 2 < φ ∧ φ < π / 2)
  (H : f (π / 2) φ = 4) : ∃ x : ℝ, f x φ = -5 := 
by
  sorry

end minimum_value_of_f_l9_9960


namespace eccentricity_correct_l9_9765

noncomputable def eccentricity_of_ellipse (a b : ℝ) (a_gt_b : a > b) (b_gt_0 : b > 0) (x y : ℝ) (h : x = 1 ∧ y = (√3)/2) 
  (on_ellipse : x^2 / a^2 + y^2 / b^2 = 1) (distance_sum : ∀ F1 F2 : ℝ × ℝ, |(1 - F1.1)| + |(1 - F2.1)| + |(√3/2 - F1.2)| + |(√3/2 - F2.2)| = 4) : ℝ := 
  let c := sqrt (a^2 - b^2) in
  c / a

theorem eccentricity_correct : eccentricity_of_ellipse 2 1 (by linarith) (by linarith) 1 (√3 / 2)
  (by {split; norm_num [sqrt_eq_rpow],}) (by sorry) = (√3) / 2 := 
sorry

end eccentricity_correct_l9_9765


namespace rectangle_side_b_value_l9_9945

section
variable {a b c d : ℕ}

-- Given conditions
def conditions : Prop :=
  (a : ℚ) / c = 3 / 4 ∧ (b : ℚ) / d = 3 / 4 ∧ c = 4 ∧ d = 8

-- Proof statement
theorem rectangle_side_b_value (h : conditions) : b = 6 :=
  sorry
end

end rectangle_side_b_value_l9_9945


namespace parabola_latus_rectum_eq_hyperbola_eccentricity_l9_9876

theorem parabola_latus_rectum_eq (y : ℝ) : 
  (y^2 = 8 * (-2)) := by
  sorry

theorem hyperbola_eccentricity (a b c : ℝ) (h : c^2 = a^2 + b^2) 
  (ha : a > 0) (hb : b > 0) (area_MON : 8 = (2 / 1) * (4 * b / a)) : 
  (sqrt (5) = c / a) := by
  sorry

end parabola_latus_rectum_eq_hyperbola_eccentricity_l9_9876


namespace mr_bird_speed_to_be_on_time_l9_9925

theorem mr_bird_speed_to_be_on_time 
  (d : ℝ) 
  (t : ℝ)
  (h1 : d = 40 * (t + 1/20))
  (h2 : d = 60 * (t - 1/20)) :
  (d / t) = 48 :=
by
  sorry

end mr_bird_speed_to_be_on_time_l9_9925


namespace range_of_a_plus_b_div_c_l9_9030

theorem range_of_a_plus_b_div_c
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : a / sin A = b / sin B)
  (h2 : b / sin B = c / sin C)
  (h3 : -cos B / cos C = (2 * a + b) / c)
  (h_non_zero : ∀ x, sin x ≠ 0) :
  1 < (a + b) / c ∧ (a + b) / c ≤ 2 * sqrt 3 / 3 := sorry

end range_of_a_plus_b_div_c_l9_9030


namespace ratio_of_girls_to_boys_l9_9064

theorem ratio_of_girls_to_boys (x y : ℕ) (h1 : x + y = 28) (h2 : x - y = 4) : x = 16 ∧ y = 12 ∧ x / y = 4 / 3 :=
by
  sorry

end ratio_of_girls_to_boys_l9_9064


namespace total_enclosed_area_l9_9662

noncomputable def line1 : ℝ → ℝ := λ x, -3 * x / 10 + 5
noncomputable def line2 : ℝ → ℝ := λ x, -x + 8
def intersection : (ℝ × ℝ) := (10, 2)
def vertex1 : (ℝ × ℝ) := (8, 0)
def vertex2 : (ℝ × ℝ) := (10, 2)
def triangle_base := 8
def triangle_height := 2

theorem total_enclosed_area :
  (∫ x in 2..10, line2 x - line1 x) + (1 / 2) * triangle_base * triangle_height = 29.8 :=
by
  sorry

end total_enclosed_area_l9_9662


namespace proof_focus_distances_sum_eq_l9_9904

noncomputable def focus_distances_sum_eq
  (P : ∀ x : ℝ, ℝ) (C : ℝ × ℝ → ℝ → Prop)
  (intersec_points : List (ℝ × ℝ))
  (focus : ℝ × ℝ)
  (d : ℝ × ℝ) : Bool :=
  P x = x^2 ∧
  (∃ k h r : ℝ, C (x, y) (x - k)^2 + (y - h)^2 = r^2) ∧
  intersec_points = [(-3, 9), (1, 1), (4, 16), d] ∧
  focus = (0, 0.25) ∧
  d = ( -((-3) + 1 + 4) ) ∧
  let distances := [dist ((-3, 9), focus), dist ((1, 1), focus), dist ((4, 16), focus), dist (d, focus)] in
  distances.sum = 30.985

theorem proof_focus_distances_sum_eq :
  focus_distances_sum_eq (λ x, x^2) (λ (p : ℝ × ℝ) (r : Prop), ((p.1 - k)^2 + (p.2 - h)^2 = r)) [(-3, 9), (1, 1), (4, 16)] (0, 0.25) (-2) = true :=
begin
  sorry
end

end proof_focus_distances_sum_eq_l9_9904


namespace find_vector_sum_l9_9798

variables (k : ℝ)

def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (-1, k)

theorem find_vector_sum (h_parallel : 2 / -1 = 4 / k) : 2 • a + b = (3, 6) :=
  by 
  -- Use the proportionality condition to deduce k
  have h1 : 2 * k = -4, sorry
  -- Substitute k = -2 in the second vector
  have hb_eq : b = (-1, -2), sorry
  -- Calculate 2 • a
  have ha2 : 2 • a = (4, 8), sorry
  -- Sum the vectors 2 • a and b
  have hsum : 2 • a + b = (3, 6), sorry
  exact hsum

end find_vector_sum_l9_9798


namespace smallest_prime_after_six_consecutive_nonprimes_l9_9173

-- Define what it means to be prime
def is_prime (n : ℕ) : Prop :=
  nat.prime n

-- Define the sequence of six consecutive nonprime numbers
def consecutive_nonprime_sequence (a : ℕ) : Prop :=
  ¬ is_prime a ∧ ¬ is_prime (a + 1) ∧ ¬ is_prime (a + 2) ∧
  ¬ is_prime (a + 3) ∧ ¬ is_prime (a + 4) ∧ ¬ is_prime (a + 5)

-- Define the condition that there is a smallest prime number greater than a sequence of six consecutive nonprime numbers
theorem smallest_prime_after_six_consecutive_nonprimes :
  ∃ p, is_prime p ∧ p > 96 ∧ consecutive_nonprime_sequence 90 :=
sorry

end smallest_prime_after_six_consecutive_nonprimes_l9_9173


namespace extremum_value_of_a_g_monotonicity_l9_9792

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 3 + x ^ 2

theorem extremum_value_of_a (a : ℝ) (h : (3 * a * (-4 / 3) ^ 2 + 2 * (-4 / 3) = 0)) : a = 1 / 2 :=
by
  -- We need to prove that a = 1 / 2 given the extremum condition.
  sorry

noncomputable def g (x : ℝ) : ℝ := (1 / 2 * x ^ 3 + x ^ 2) * Real.exp x

theorem g_monotonicity :
  (∀ x < -4, deriv g x < 0) ∧
  (∀ x, -4 < x ∧ x < -1 → deriv g x > 0) ∧
  (∀ x, -1 < x ∧ x < 0 → deriv g x < 0) ∧
  (∀ x > 0, deriv g x > 0) :=
by
  -- We need to prove the monotonicity of the function g in the specified intervals.
  sorry

end extremum_value_of_a_g_monotonicity_l9_9792


namespace cube_volume_l9_9566

theorem cube_volume (perimeter : ℝ) (h : perimeter = 32) :
  let s := perimeter / 4 in
  let volume := s^3 in
  volume = 512 :=
by
  sorry

end cube_volume_l9_9566


namespace perfect_square_trinomial_l9_9821

theorem perfect_square_trinomial (m : ℝ) :
  (∃ (a b : ℝ), (x : ℝ) -> (a = 1 ∨ a = -1) ∧ b^2 = 9 ∧ (x^2 + (m-1)x + 9 = (a*x + b)^2)) → (m = -5 ∨ m = 7) :=
by
  intro h
  obtain ⟨a, b, ha, hb, eqn⟩ := h
  cases ha
  { have : b = 3 ∨ b = -3 := by linarith [hb]; cases this; sorry }
  { have : b = 3 ∨ b = -3 := by linarith [hb]; cases this; sorry }

end perfect_square_trinomial_l9_9821


namespace centroid_coincides_l9_9028

variable {A B C A1 B1 C1 : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited A1] [Inhabited B1] [Inhabited C1]

-- Define a centroid function
noncomputable def centroid (X Y Z : Type) := sorry

-- Given conditions
-- 1. In triangle ABC, there is a right angle at C
axiom right_angle_C : ∀ (A B C : Type), ∃ (right_angle : B), true

-- 2. Construct similar triangles ABC1, BCA1, and CAB1 outwardly
axiom similar_triangles : ∀ (A B C A1 B1 C1 : Type), true

-- Main theorem
theorem centroid_coincides (A B C A1 B1 C1 : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited A1] [Inhabited B1] [Inhabited C1]
  (right_angle_C : ∀ (A B C : Type), ∃ (right_angle : B), true)
  (similar_triangles : ∀ (A B C A1 B1 C1 : Type), true) :
  centroid A1 B1 C1 = centroid A B C := 
sorry

end centroid_coincides_l9_9028


namespace boxes_needed_l9_9897

theorem boxes_needed (total_bananas : ℕ) (bananas_per_box : ℕ) (h1 : total_bananas = 40) (h2 : bananas_per_box = 4) : total_bananas / bananas_per_box = 10 :=
by
  rw [h1, h2]
  norm_num
  sorry

end boxes_needed_l9_9897


namespace smallest_possible_value_l9_9360

theorem smallest_possible_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 / (x : ℝ)) + (1 / (y : ℝ)) = 1 / 15) : x + y = 64 :=
sorry

end smallest_possible_value_l9_9360


namespace remainder_when_divided_by_20_l9_9735

theorem remainder_when_divided_by_20 (n : ℕ) : (4 * 6^n + 5^(n-1)) % 20 = 9 := 
by
  sorry

end remainder_when_divided_by_20_l9_9735


namespace check_disagreement_count_l9_9124

-- Primary Level
def total_parents_primary : ℕ := 300
def agreed_scholarships_primary : ℕ := 0.30 * total_parents_primary
def agreed_quality_primary : ℕ := 0.10 * total_parents_primary
def total_agreed_primary : ℕ := agreed_scholarships_primary + agreed_quality_primary
def disagreed_primary : ℕ := total_parents_primary - total_agreed_primary

-- Intermediate Level
def total_parents_intermediate : ℕ := 250
def agreed_scholarships_intermediate : ℕ := 0.20 * total_parents_intermediate
def agreed_quality_intermediate : ℕ := 0.05 * total_parents_intermediate
def total_agreed_intermediate : ℕ := agreed_scholarships_intermediate + agreed_quality_intermediate
def disagreed_intermediate : ℕ := total_parents_intermediate - total_agreed_intermediate

-- Secondary Level
def total_parents_secondary : ℕ := 250
def agreed_scholarships_secondary : ℕ := 0.15 * total_parents_secondary
def agreed_quality_secondary : ℕ := 0.08 * total_parents_secondary
def total_agreed_secondary : ℕ := agreed_scholarships_secondary + agreed_quality_secondary
def disagreed_secondary : ℕ := total_parents_secondary - total_agreed_secondary

theorem check_disagreement_count :
  disagreed_primary = 180 ∧
  disagreed_intermediate = 187 ∧
  disagreed_secondary = 192 := 
by
  -- Proof is intentionally omitted
  sorry

end check_disagreement_count_l9_9124


namespace cosine_theorem_l9_9214

theorem cosine_theorem (a b c : ℝ) (A : ℝ) (hA : 0 < A) (hA_lt_pi : A < π) :
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A := sorry

end cosine_theorem_l9_9214


namespace basketball_team_allocation_l9_9839

theorem basketball_team_allocation (classes : Fin 8) (players : Fin 10) :
  ∀ (contribution : Fin 8 → ℕ),
    (∀ c, 1 ≤ contribution c) →
    (∑ c, contribution c = 10) →
    (finset.univ.sum ((λ c => finset.univ.filter (λ x => ∑ i in finset.univ, if i ≠ c then 1 else 0 + x = 2)).card) +
     finset.univ.card) = 36 :=
by
  sorry

end basketball_team_allocation_l9_9839


namespace original_price_l9_9625

theorem original_price (P : ℝ) (profit : ℝ) (profit_percentage : ℝ)
  (h1 : profit = 675) (h2 : profit_percentage = 0.35) :
  P = 1928.57 :=
by
  -- The proof is skipped using sorry
  sorry

end original_price_l9_9625


namespace problem_solution_l9_9384

variable {A B : Set ℝ}

def definition_A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def definition_B : Set ℝ := {x | x ≤ 3}

theorem problem_solution : definition_A ∩ definition_B = definition_A :=
by
  sorry

end problem_solution_l9_9384


namespace complex_power_rectangular_form_l9_9675

noncomputable def cos (θ : ℝ) : ℝ := sorry
noncomputable def sin (θ : ℝ) : ℝ := sorry

theorem complex_power_rectangular_form :
  (3 * complex.exp (complex.I * 30))^4 = -40.5 + 40.5 * complex.I * √3 :=
by
  sorry

end complex_power_rectangular_form_l9_9675


namespace range_of_f_in_interval_value_of_b_in_triangle_ABC_l9_9790

noncomputable def f (x : ℝ) : ℝ :=
  let mx := (sqrt 3 * Real.sin x, Real.cos x)
  let nx := (Real.cos x, -Real.cos x)
  mx.1 * nx.1 + mx.2 * nx.2 - 1 / 2

theorem range_of_f_in_interval :
  (∀ x ∈ Icc 0 (Real.pi / 2), f x ∈ Icc (-3 / 2) 0) := 
sorry

theorem value_of_b_in_triangle_ABC (B : ℝ) :
  f B = 0 → 
  let c := 2
  let a := 3
  ∃ b : ℝ, b = sqrt 7 :=
sorry

end range_of_f_in_interval_value_of_b_in_triangle_ABC_l9_9790


namespace inequality_proof_l9_9502

theorem inequality_proof (x : Real) (h1 : ¬(x ∈ Set.Univ \ Set.Icc 1 1)) (h2 : 1 < x) :
  let a := Real.floor x
  let r := x - Real.floor x
  (Real.sqrt (x + r) / a - a / (x + r)) + ((x + a) / r - r / (x + a)) > 9 / 2 :=
by
  let a := Real.floor x
  let r := x - Real.floor x
  sorry

end inequality_proof_l9_9502


namespace option_D_does_not_hold_l9_9491

variable {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
n * (a 1) + (n * (n - 1) / 2) * (a 2 - a 1)

-- Given condition
axiom hS : sum_of_first_n_terms a 5 > sum_of_first_n_terms a 6

theorem option_D_does_not_hold (a : ℕ → ℝ) (d : ℝ) (hS : sum_of_first_n_terms a 5 > sum_of_first_n_terms a 6)
    (h_arith : is_arithmetic_sequence a d) :
    ¬ ((a 3) + (a 6) + (a 12) < 2 * (a 7)) := 
sorry

end option_D_does_not_hold_l9_9491


namespace complex_power_result_l9_9680

theorem complex_power_result :
  (3 * Complex.cos (Real.pi / 6) + 3 * Complex.sin (Real.pi / 6) * Complex.i)^4 = 
  -40.5 + 40.5 * Complex.i * Real.sqrt 3 :=
by sorry

end complex_power_result_l9_9680


namespace total_first_class_equipment_l9_9265

theorem total_first_class_equipment (x y : ℕ) (h1 : x < y)
    (h2 : 0.9 * (y + 0.3 * x) > 1.02 * y)
    (h3 : 0.73 * x + 0.1 * y = 0.27 * x + 0.9 * y + 6) :
  y = 17 :=
by
  sorry -- Proof is skipped as per the instructions

end total_first_class_equipment_l9_9265


namespace building_height_l9_9645

-- Definitions of the conditions
def wooden_box_height : ℝ := 3
def wooden_box_shadow : ℝ := 12
def building_shadow : ℝ := 36

-- The statement that needs to be proved
theorem building_height : ∃ (height : ℝ), height = 9 ∧ wooden_box_height / wooden_box_shadow = height / building_shadow :=
by
  sorry

end building_height_l9_9645


namespace area_of_triangular_region_l9_9253

-- Define the line equation and the triangle
def line_eq (x y : ℝ) := 3 * x + y = 9
def is_boundary (x y : ℝ) := (y = 0 ∧ x ≥ 0) ∨ (x = 0 ∧ y ≥ 0) ∨ line_eq x y

-- Define the triangular region as a set of points
def triangular_region (p: ℝ × ℝ) := 
  let (x, y) := p in
  x ≥ 0 ∧ y ≥ 0 ∧ line_eq x y

-- Define the computed area of the triangular region
def triangular_area := (3 * 9) / 2

-- The theorem statement
theorem area_of_triangular_region :
  triangular_area = 27 / 2 :=
by
  -- Proof goes here
  sorry

end area_of_triangular_region_l9_9253


namespace sum_of_four_l9_9846

variable {α : Type*} [LinearOrderedField α]

def geometric_sequence (a₁ q : α) : ℕ → α
| 0 := a₁
| (n+1) := geometric_sequence a₁ q n * q

def sum_geometric_sequence (a₁ q : α) (n : ℕ) : α :=
if q = 1 then a₁ * (n + 1) else a₁ * (1 - q^(n + 1)) / (1 - q)

theorem sum_of_four {a₁ q S₅ S₃ S₄ : α} (h₁ : a₁ = 1) (h₂ : sum_geometric_sequence a₁ q 4 = S₄) (h₃ : sum_geometric_sequence a₁ q 5 = S₅) (h₄ : sum_geometric_sequence a₁ q 3 = S₃) : S₅ = 5 * S₃ - 4 → S₄ = 15 :=
by
  sorry

end sum_of_four_l9_9846


namespace constant_term_expansion_l9_9546

noncomputable def constantTerm {n : ℕ} (term : ℕ) (cond : term = 5) 
  (largest : ∀ k : ℕ, binomial n k = binomial n 4 → k = 4) : ℤ :=
  let r := 4
  let binom := Nat.choose n r
  let coeff := (-2) ^ r
  binom * coeff

theorem constant_term_expansion : constantTerm (8) 5 (by simp) (by simp) = 1120 := by
  sorry

end constant_term_expansion_l9_9546


namespace equal_sum_of_red_and_blue_sides_l9_9098

theorem equal_sum_of_red_and_blue_sides 
  (O1 O2 O3 : Point)
  (r : Real)
  (C1 C2 C3 : Circle)
  (h_non_intersecting : disjoint C1 C2 ∧ disjoint C2 C3 ∧ disjoint C1 C3)
  (h_equal_radius : C1.radius = r ∧ C2.radius = r ∧ C3.radius = r)
  (h_triangle : triangle_form O1 O2 O3)
  (hexagon : Hexagon)
  (h_tangents : ∀ (P : Point), ∀ (C : Circle), tangent_drawn P C)
  (h_convex : is_convex hexagon)
  (h_colors : colored_alternately hexagon) : 
  ∑ red_sides hexagon = ∑ blue_sides hexagon := 
by sorry

end equal_sum_of_red_and_blue_sides_l9_9098


namespace third_competitor_hot_dogs_l9_9295

theorem third_competitor_hot_dogs (first_ate : ℕ) (second_multiplier : ℕ) (third_percent_less : ℕ) (second_ate third_ate : ℕ) 
  (H1 : first_ate = 12)
  (H2 : second_multiplier = 2)
  (H3 : third_percent_less = 25)
  (H4 : second_ate = first_ate * second_multiplier)
  (H5 : third_ate = second_ate - (third_percent_less * second_ate / 100)) : 
  third_ate = 18 := 
by 
  sorry

end third_competitor_hot_dogs_l9_9295


namespace laura_house_distance_l9_9479

-- Definitions based on conditions
def x : Real := 10  -- Distance from Laura's house to her school in miles

def distance_to_school_per_day := 2 * x
def school_days_per_week := 5
def distance_to_school_per_week := school_days_per_week * distance_to_school_per_day

def distance_to_supermarket := x + 10
def supermarket_trips_per_week := 2
def distance_to_supermarket_per_trip := 2 * distance_to_supermarket
def distance_to_supermarket_per_week := supermarket_trips_per_week * distance_to_supermarket_per_trip

def total_distance_per_week := 220

-- The proof statement
theorem laura_house_distance :
  distance_to_school_per_week + distance_to_supermarket_per_week = total_distance_per_week ∧ x = 10 := by
  sorry

end laura_house_distance_l9_9479


namespace initial_discount_percentage_l9_9234

variables (d x : ℝ)
def discount_price := (1 - x / 100) * d
def staff_price := 0.40 * (1 - x / 100) * d

theorem initial_discount_percentage (h1 : staff_price d x = 0.14 * d) : x = 65 := by
  unfold staff_price at h1
  unfold discount_price at h1
  sorry

end initial_discount_percentage_l9_9234


namespace example_non_gaussian_sum_l9_9523

-- Define a Gaussian random variable ξ
structure gaussian (μ σ : ℝ) :=
(pr : {ξ // true})

-- Define a Bernoulli random variable ζ
structure bernoulli :=
(pr : {ζ // ζ = -1 ∨ ζ = 1})

-- Define the sum of random variables and prove that it has a non-Gaussian distribution
theorem example_non_gaussian_sum :
  ∃ (ξ : gaussian 0 1) (ζ : bernoulli),
    let η := ζ.pr.1 * ξ.pr.1 in
    (ξ.pr.1 + η) ≠ gaussian 0 1 :=
by sorry

end example_non_gaussian_sum_l9_9523


namespace find_other_numbers_l9_9518

theorem find_other_numbers (a b c: ℤ) (smallest : ℤ) (x : ℤ)
  (h1 : multiset.mem 3 {a, b, c})
  (h2 : multiset.mem 9 {a, b, c})
  (h3 : multiset.mem 15 {a, b, c})
  (operation : ∀ x y z : ℤ, (x + y - z ∈ {a, b, c})) 
  (h_smallest : smallest = 2013)
  (hh : smallest ∈ {a, b, c}) :
  {a, b, c} = {2013, 2019, 2025} :=
by {
    sorry
}

end find_other_numbers_l9_9518


namespace length_of_train_300_l9_9252

noncomputable def length_of_train (cross_pole_time cross_platform_time platform_length : ℝ) : ℝ :=
  let v := L / cross_pole_time  -- Speed when crossing the pole
  L ← sorry
  v = (L + platform_length) / cross_platform_time
  L

theorem length_of_train_300 :
  (∀ L, ∀ (cross_pole_time cross_platform_time platform_length : ℝ),
    cross_pole_time = 18 ∧
    cross_platform_time = 36 ∧
    platform_length = 300 →
    length_of_train cross_pole_time cross_platform_time platform_length = 300) :=
by
  intro L
  intro cross_pole_time cross_platform_time platform_length
  intro h
  cases h with h_cross_pole_time h_rest
  cases h_rest with h_cross_platform_time h_platform_length
  simp_rw [h_cross_pole_time, h_cross_platform_time, h_platform_length]
  sorry

end length_of_train_300_l9_9252


namespace complex_power_eq_rectangular_l9_9706

noncomputable def cos_30 := Real.cos (Real.pi / 6)
noncomputable def sin_30 := Real.sin (Real.pi / 6)

theorem complex_power_eq_rectangular :
  (3 * (cos_30 + (complex.I * sin_30)))^4 = -40.5 + 40.5 * complex.I * Real.sqrt 3 := 
sorry

end complex_power_eq_rectangular_l9_9706


namespace cost_to_fill_bathtub_with_jello_l9_9470

-- Define the conditions
def pounds_per_gallon : ℝ := 8
def gallons_per_cubic_foot : ℝ := 7.5
def cubic_feet_of_water : ℝ := 6
def tablespoons_per_pound : ℝ := 1.5
def cost_per_tablespoon : ℝ := 0.5

-- The theorem stating the cost to fill the bathtub with jello
theorem cost_to_fill_bathtub_with_jello : 
  let total_gallons := cubic_feet_of_water * gallons_per_cubic_foot in
  let total_pounds := total_gallons * pounds_per_gallon in
  let total_tablespoons := total_pounds * tablespoons_per_pound in
  let total_cost := total_tablespoons * cost_per_tablespoon in
  total_cost = 270 := 
by {
  -- Here's where we would provide the proof steps, but just add sorry to skip it
  sorry
}

end cost_to_fill_bathtub_with_jello_l9_9470


namespace platform_length_eq_l9_9619

-- Definition of the problem conditions
def speed_kmph : ℝ := 72
def time_seconds : ℝ := 26
def length_of_train_m : ℝ := 250.0416

-- Conversion factor from kmph to m/s
def kmph_to_mps (speed: ℝ) : ℝ := speed * 1000 / 3600

-- Speed of the goods train in meters per second
def speed_mps : ℝ := kmph_to_mps speed_kmph

-- Distance covered by the train while crossing the platform
def total_distance_m : ℝ := speed_mps * time_seconds

-- Length of the platform
def length_of_platform_m : ℝ := total_distance_m - length_of_train_m

-- Assertion that length of platform is 269.9584 meters
theorem platform_length_eq : length_of_platform_m = 269.9584 := by
  -- Proof omitted
  sorry

end platform_length_eq_l9_9619


namespace sum_of_four_l9_9847

variable {α : Type*} [LinearOrderedField α]

def geometric_sequence (a₁ q : α) : ℕ → α
| 0 := a₁
| (n+1) := geometric_sequence a₁ q n * q

def sum_geometric_sequence (a₁ q : α) (n : ℕ) : α :=
if q = 1 then a₁ * (n + 1) else a₁ * (1 - q^(n + 1)) / (1 - q)

theorem sum_of_four {a₁ q S₅ S₃ S₄ : α} (h₁ : a₁ = 1) (h₂ : sum_geometric_sequence a₁ q 4 = S₄) (h₃ : sum_geometric_sequence a₁ q 5 = S₅) (h₄ : sum_geometric_sequence a₁ q 3 = S₃) : S₅ = 5 * S₃ - 4 → S₄ = 15 :=
by
  sorry

end sum_of_four_l9_9847


namespace part1_equation_of_tangent_part2_equation_of_secant_l9_9762

noncomputable def C := {p : ℝ×ℝ | (p.1 - 3)^2 + (p.2 - 4)^2 = 4}

def passesThroughA (l : ℝ × ℝ → Prop) : Prop := l (1, 0)

def isTangent (l : ℝ × ℝ → Prop) : Prop := 
∀ (p : ℝ × ℝ), l p → ∃! q : ℝ × ℝ, q ∈ C ∧ dist (3, 4) q = 2

def isSecant (l : ℝ × ℝ → Prop) : Prop := 
∃ P Q, l P ∧ l Q ∧ P ≠ Q ∧ P ∈ C ∧ Q ∈ C ∧ dist P Q = 2 * sqrt 2

theorem part1_equation_of_tangent (l : ℝ × ℝ → Prop) (h1 : passesThroughA l) (h2 : isTangent l) :
l = (λ p, p.1 = 1) ∨ l = (λ p, 3 * p.1 - 4 * p.2 = 3) := sorry

theorem part2_equation_of_secant (l : ℝ × ℝ → Prop) (h1 : passesThroughA l) (h2 : isSecant l) :
l = (λ p, p.1 - p.2 - 1 = 0) ∨ l = (λ p, 7 * p.1 - p.2 - 7 = 0) := sorry

end part1_equation_of_tangent_part2_equation_of_secant_l9_9762


namespace find_a_l9_9494

noncomputable def f (a x : ℝ) : ℝ := a^x + Real.logb a (x + 1)

theorem find_a : 
  ( ∀ a : ℝ, 
    (∀ x : ℝ,  0 ≤ x ∧ x ≤ 1 → f a 0 + f a 1 = a) → a = 1/2 ) :=
sorry

end find_a_l9_9494


namespace surface_area_of_large_cube_is_486_cm_squared_l9_9514

noncomputable def surfaceAreaLargeCube : ℕ :=
  let small_box_count := 27
  let edge_small_box := 3
  let edge_large_cube := (small_box_count^(1/3)) * edge_small_box
  6 * edge_large_cube^2

theorem surface_area_of_large_cube_is_486_cm_squared :
  surfaceAreaLargeCube = 486 := 
sorry

end surface_area_of_large_cube_is_486_cm_squared_l9_9514


namespace valid_license_plates_count_l9_9640

theorem valid_license_plates_count : 
  ∃ n : ℕ, n = 26^3 * 10^2 ∧ n = 1_757_600 :=
begin
  use 26^3 * 10^2,
  split,
  { refl },
  { norm_num },
end

end valid_license_plates_count_l9_9640


namespace complex_exp_form_pow_four_l9_9666

theorem complex_exp_form_pow_four :
  let θ := 30 * Real.pi / 180
  let cos_θ := Real.cos θ
  let sin_θ := Real.sin θ
  let z := 3 * (cos_θ + Complex.I * sin_θ)
  z ^ 4 = -40.5 + 40.5 * Complex.I * Real.sqrt 3 :=
by
  sorry

end complex_exp_form_pow_four_l9_9666


namespace sum_coeff_of_terms_without_y_l9_9128

theorem sum_coeff_of_terms_without_y (n : ℕ) (hn : 0 < n) :
  ∑ k in range (n + 1), (binomial n k) * (4 ^ (n - k)) * (-3 * 1) ^ k = 1 :=
by
  sorry

end sum_coeff_of_terms_without_y_l9_9128


namespace functional_equation_unique_solution_l9_9302

noncomputable def continuous_function_of_form (f : ℝ → ℝ) : Prop :=
  ∀ x, ∃ c d, f x = c * x + d

theorem functional_equation_unique_solution (f : ℝ → ℝ)
  (h_f_cont : continuous f)
  (h_f_eq : ∀ x y : ℝ, 0 < x → 0 < y →
    f(x + 1/x) + f(y + 1/y) = f(x + 1/y) + f(y + 1/x)) :
  continuous_function_of_form f :=
by
  sorry

end functional_equation_unique_solution_l9_9302


namespace smallest_prime_after_six_nonprime_l9_9167

open Nat

noncomputable def is_nonprime (n : ℕ) : Prop :=
  ¬Prime n ∧ n ≠ 1

noncomputable def six_consecutive_nonprime (n : ℕ) : Prop :=
  is_nonprime n ∧ is_nonprime (n + 1) ∧ is_nonprime (n + 2) ∧ 
  is_nonprime (n + 3) ∧ is_nonprime (n + 4) ∧ is_nonprime (n + 5)

theorem smallest_prime_after_six_nonprime :
  ∃ n, six_consecutive_nonprime n ∧ (∀ m, m ≠ 97 → n + 6 < m → ¬Prime m) :=
by
  sorry

end smallest_prime_after_six_nonprime_l9_9167


namespace quadrilateral_angles_equal_l9_9324

theorem quadrilateral_angles_equal (A B C D A' B' C' D' M : Type)
  (N : quadrilateral A B C D)
  (perpendiculars : from_each_vertex_perpendicular_to_diagonals N A' B' C' D')
  (diagonal_intersection : diagonals_intersect_at_point N M)
  (non_perpendicular_case : ¬ diagonals_perpendicular N) :
  quadrilateral_angles A'B'C'D' = quadrilateral_angles ABCD :=
sorry

end quadrilateral_angles_equal_l9_9324


namespace sum_of_interior_angles_l9_9973

-- Define the conditions:
def exterior_angle (n : ℕ) := 45

def sum_exterior_angles := 360

-- Define the Lean statement for the proof problem
theorem sum_of_interior_angles : ∃ n : ℕ, 
  sum_exterior_angles / exterior_angle n = n ∧
  (180 * (n - 2) = 1080) :=
by
  use 8
  split
  calc
    sum_exterior_angles / exterior_angle 8 = 360 / 45 := rfl
    ... = 8 := rfl
  calc
    180 * (8 - 2) = 180 * 6 := rfl
    ... = 1080 := rfl

end sum_of_interior_angles_l9_9973


namespace smallest_prime_after_six_nonprime_l9_9165

open Nat

noncomputable def is_nonprime (n : ℕ) : Prop :=
  ¬Prime n ∧ n ≠ 1

noncomputable def six_consecutive_nonprime (n : ℕ) : Prop :=
  is_nonprime n ∧ is_nonprime (n + 1) ∧ is_nonprime (n + 2) ∧ 
  is_nonprime (n + 3) ∧ is_nonprime (n + 4) ∧ is_nonprime (n + 5)

theorem smallest_prime_after_six_nonprime :
  ∃ n, six_consecutive_nonprime n ∧ (∀ m, m ≠ 97 → n + 6 < m → ¬Prime m) :=
by
  sorry

end smallest_prime_after_six_nonprime_l9_9165


namespace find_OC_l9_9208

theorem find_OC {A B O C : Type*} [AddGroup A] [Dist A]
  (hAB : dist A B = 24)
  (hBC : dist B C = 28)
  (hOA : dist O A = 15)
  (h_circ : ∀ (M : A), dist O M = dist O M)
  (hC_on_ray : ∃ t : ℝ, t > 1 ∧ dist A B = t * dist B C) : 
  dist O C = 41 :=
by
  sorry

end find_OC_l9_9208


namespace metropolis_partition_l9_9877

open Finset

theorem metropolis_partition (stations : Finset ℕ) {G : SimpleGraph ℕ} 
  (h_stations : stations.card = 1972)
  (h_edges : ∀ (u v : ℕ) (h_u : u ∈ stations) (h_v : v ∈ stations) (h_uv : ¬u = v), G.adj u v)
  (h_connected_after_closure : ∀ (S : Finset ℕ) (hS : S.card = 9) (H : ∀ s ∈ S, s ∈ stations), 
    ∃ t ∈ stations \ S, G.isConnected (stations \ S ∪ {t}))
  (h_transfer_AB : ∃ (A B : ℕ) (hA : A ∈ stations) (hB : B ∈ stations), G.minTransfers A B ≥ 99) :
  ∃ (groups : Finset (Finset ℕ)), 
    groups.card = 1000 ∧ (∀ g ∈ groups, ∀ a b ∈ g, ¬G.adj a b) :=
begin
  sorry
end

end metropolis_partition_l9_9877


namespace prices_of_books_book_purchasing_plans_l9_9602

-- Define the conditions
def cost_eq1 (x y : ℕ): Prop := 20 * x + 40 * y = 1520
def cost_eq2 (x y : ℕ): Prop := 20 * x - 20 * y = 440
def plan_conditions (x y : ℕ): Prop := (20 + y - x = 20) ∧ (x + y + 20 ≥ 72) ∧ (40 * x + 18 * (y + 20) ≤ 2000)

-- Prove price of each book
theorem prices_of_books : 
  ∃ (x y : ℕ), cost_eq1 x y ∧ cost_eq2 x y ∧ x = 40 ∧ y = 18 :=
by {
  sorry
}

-- Prove possible book purchasing plans
theorem book_purchasing_plans : 
  ∃ (x : ℕ), plan_conditions x (x + 20) ∧ 
  (x = 26 ∧ x + 20 = 46 ∨ 
   x = 27 ∧ x + 20 = 47 ∨ 
   x = 28 ∧ x + 20 = 48) :=
by {
  sorry
}

end prices_of_books_book_purchasing_plans_l9_9602


namespace central_angle_l9_9777

theorem central_angle (r l θ : ℝ) (condition1: 2 * r + l = 8) (condition2: (1 / 2) * l * r = 4) (theta_def : θ = l / r) : |θ| = 2 :=
by
  sorry

end central_angle_l9_9777


namespace complex_fourth_power_l9_9694

-- Definitions related to the problem conditions
def trig_identity (θ : ℝ) : ℂ := complex.of_real (cos θ) + complex.I * complex.of_real (sin θ)
def z : ℂ := 3 * trig_identity (30 * real.pi / 180)
def result := (81 : ℝ) * (-1/2 + complex.I * (real.sqrt 3 / 2))

-- Statement of the problem
theorem complex_fourth_power : z^4 = result := by
  sorry

end complex_fourth_power_l9_9694


namespace exists_2016_consecutive_with_16_primes_l9_9724

theorem exists_2016_consecutive_with_16_primes :
  ∃ n : ℕ, (finset.Icc n (n + 2015)).filter prime.card = 16 :=
sorry

end exists_2016_consecutive_with_16_primes_l9_9724


namespace gardener_first_number_of_rows_l9_9618

theorem gardener_first_number_of_rows (n : ℕ) (h : n = 84) : 
  ∃ k, k > 1 ∧ k ∣ n ∧ (∀ m, m ∣ n → m ≠ 1 → m ≥ k) :=
begin
  have h := (nat.factors_unique h) { factor_lcm := by simp, },
  use 2, -- The smallest factor greater than 1 is 2.
  split,
  { -- k > 1
    exact nat.succ_pos 1, 
    -- 2 ∣ 84
    apply dvd, exact nat.factors_unique h, simp,
  },
  intros m m_div_n m_ne_1,
  by_cases m = 2,
  { rwa h, },
  { 
    intros m_div_n m_ne_1, apply nat.ge_of_le_of_eq,
    apply m_div_n, rw nat.succ_pred_eq_of_pos, {
    exact nat.pred_le_pred, rw s_num_factors.mk_of_dvd, exact_m_div_n, intros
  },
end

end gardener_first_number_of_rows_l9_9618


namespace fence_calculation_l9_9633

def width : ℕ := 4
def length : ℕ := 2 * width - 1
def fence_needed : ℕ := 2 * length + 2 * width

theorem fence_calculation : fence_needed = 22 := by
  sorry

end fence_calculation_l9_9633


namespace octagonal_pyramid_base_edge_length_l9_9571

noncomputable def cot (θ : ℝ) : ℝ := Real.cos θ / Real.sin θ
noncomputable def toRad (deg : ℝ) : ℝ := deg * Real.pi / 180

theorem octagonal_pyramid_base_edge_length :
  let S := 2538.34  -- Surface area in dm^2
  let angleDeg := 80
  let angleRad := toRad angleDeg
  let cot22_5 := cot (toRad 22.5)
  let cot10 := cot (toRad 10)
  let sin10 := Real.sin (toRad 10)
  let sin22_5 := Real.sin (toRad 22.5)
  let sin32_5 := Real.sin (toRad 32.5)
  let x := Real.sqrt ((2538.34 * sin22_5 * sin10) / (2 * sin32_5))
  x ≈ 12.53 :=
by
  sorry

end octagonal_pyramid_base_edge_length_l9_9571


namespace periodic_even_function_l9_9051

open Real

noncomputable def f : ℝ → ℝ := sorry

theorem periodic_even_function (f : ℝ → ℝ)
  (h1 : ∀ x, f (x + 2) = f x)
  (h2 : ∀ x, f (-x) = f x)
  (h3 : ∀ x, 2 ≤ x ∧ x ≤ 3 → f x = x) :
  ∀ x, -2 ≤ x ∧ x ≤ 0 → f x = 3 - abs (x + 1) :=
sorry

end periodic_even_function_l9_9051


namespace median_length_of_right_triangle_l9_9021

noncomputable def hypotenuse_length (DE DF : ℝ) : ℝ := 
  real.sqrt (DE^2 + DF^2)

theorem median_length_of_right_triangle (DE DF : ℝ) (hDE : DE = 6) (hDF : DF = 8) :
  let EF := hypotenuse_length DE DF
  in EF / 2 = 5 := 
by
  sorry

end median_length_of_right_triangle_l9_9021


namespace find_angle_x_l9_9456

theorem find_angle_x (A B C D E F X Y : Type) 
  (AB CD EF : line) 
  (intersect_AXE : ∠ AXE = 75) 
  (intersect_EXB : ∠ EXB = 35)
  (intersect_CYD : ∠ CYD = 140)
  (X_on_EF : X ∈ EF)
  (Y_on_CD : Y ∈ CD) 
  (X_in_AXE : X ∈ AB) 
  (Y_in_AXE : Y ∈ CD)
  :
  ∃ (x : ℝ), x = 70 := 
by
  sorry

end find_angle_x_l9_9456


namespace equal_focal_distances_l9_9564

theorem equal_focal_distances (k : ℝ) (h₁ : k ≠ 0) (h₂ : 16 - k ≠ 0) 
  (h_hyperbola : ∀ x y, (x^2) / (16 - k) - (y^2) / k = 1)
  (h_ellipse : ∀ x y, 9 * x^2 + 25 * y^2 = 225) :
  0 < k ∧ k < 16 :=
sorry

end equal_focal_distances_l9_9564


namespace remainder_R5_l9_9485

noncomputable def P (x : ℤ) : ℤ :=
  ∏ a in range 1 2017, ∏ b in range 0 2017, ((a : ℤ) * x + b)

def R (x : ℤ) : ℤ :=
  P x % (x^5 - 1)

theorem remainder_R5 (h : R (5) % 2017 = 5) : True :=
begin
  trivial
end

end remainder_R5_l9_9485


namespace min_questions_30_cards_min_questions_31_cards_min_questions_32_cards_min_questions_50_circle_l9_9601

-- Problem definition for (1): 30 cards
theorem min_questions_30_cards (cards : Fin 30 → ℤ) (h : ∀ i, cards i = 1 ∨ cards i = -1) :
  (∀ f : Finset (Fin 30), f.card = 3 → ℤ) → minimum_questions [] 10 :=
sorry

-- Problem definition for (2): 31 cards
theorem min_questions_31_cards (cards : Fin 31 → ℤ) (h : ∀ i, cards i = 1 ∨ cards i = -1) :
  (∀ f : Finset (Fin 31), f.card = 3 → ℤ) → minimum_questions [] 11 :=
sorry

-- Problem definition for (3): 32 cards
theorem min_questions_32_cards (cards : Fin 32 → ℤ) (h : ∀ i, cards i = 1 ∨ cards i = -1) :
  (∀ f : Finset (Fin 32), f.card = 3 → ℤ) → minimum_questions [] 12 :=
sorry

-- Problem definition for (4): 50 numbers written in a circle
theorem min_questions_50_circle (cards : Fin 50 → ℤ) (h : ∀ i, cards i = 1 ∨ cards i = -1) :
  (∀ i : Fin 50, ℤ) → minimum_questions [circle_constraint] 50 :=
sorry

-- auxiliary definitions
def minimum_questions (constraints : List (Fin n → ℤ → Prop)) (q : ℕ) : Prop :=
∀ g, (constraints g) → ...

def circle_constraint (cards: Fin 50 → ℤ) (prod: ℤ) : Prop :=
∀ i, prod = cards i * cards ((i + 1) % 50) * cards ((i + 2) % 50)

end min_questions_30_cards_min_questions_31_cards_min_questions_32_cards_min_questions_50_circle_l9_9601


namespace shaded_region_area_correct_l9_9286

noncomputable def shaded_region_area : ℝ :=
  let radius := 1
  let triangle_side := radius * 2
  let triangle_height := Real.sqrt (triangle_side ^ 2 - (triangle_side / 2) ^ 2)
  let triangle_area := (1 / 2) * triangle_side * triangle_height
  let rectangle_area := 2 * radius
  let sector_area := (1 / 3) * π * radius ^ 2
  let total_area := triangle_area + 3 * rectangle_area + π
  let white_area := 3 * π
  total_area - white_area

theorem shaded_region_area_correct : shaded_region_area = 6 + Real.sqrt 3 - 2 * π := 
  sorry

end shaded_region_area_correct_l9_9286


namespace count_integers_P_leq_0_l9_9285

def P(x : ℤ) : ℤ := 
  (x - 1^3) * (x - 2^3) * (x - 3^3) * (x - 4^3) * (x - 5^3) *
  (x - 6^3) * (x - 7^3) * (x - 8^3) * (x - 9^3) * (x - 10^3) *
  (x - 11^3) * (x - 12^3) * (x - 13^3) * (x - 14^3) * (x - 15^3) *
  (x - 16^3) * (x - 17^3) * (x - 18^3) * (x - 19^3) * (x - 20^3) *
  (x - 21^3) * (x - 22^3) * (x - 23^3) * (x - 24^3) * (x - 25^3) *
  (x - 26^3) * (x - 27^3) * (x - 28^3) * (x - 29^3) * (x - 30^3) *
  (x - 31^3) * (x - 32^3) * (x - 33^3) * (x - 34^3) * (x - 35^3) *
  (x - 36^3) * (x - 37^3) * (x - 38^3) * (x - 39^3) * (x - 40^3) *
  (x - 41^3) * (x - 42^3) * (x - 43^3) * (x - 44^3) * (x - 45^3) *
  (x - 46^3) * (x - 47^3) * (x - 48^3) * (x - 49^3) * (x - 50^3)

theorem count_integers_P_leq_0 : 
  ∃ n : ℕ, n = 15650 ∧ ∀ k : ℤ, (P k ≤ 0) → (n = 15650) :=
by sorry

end count_integers_P_leq_0_l9_9285


namespace minimize_distance_l9_9763

/-- Define line l₁: x + 2y + t² = 0 and line l₂: 2x + 4y + 2t - 3 = 0 -/
def line1 (t : ℝ) : ℝ × ℝ → Prop := λ p, p.1 + 2 * p.2 + t^2 = 0
def line2 (t : ℝ) : ℝ × ℝ → Prop := λ p, 2 * p.1 + 4 * p.2 + 2 * t - 3 = 0

/-- The value of t that minimizes the distance between lines l₁ and l₂ is t = 1/2. -/
theorem minimize_distance (t : ℝ) : 
  (∀ p, line1 t p → line2 t p → t = 1/2) :=
sorry

end minimize_distance_l9_9763


namespace rotated_vector_l9_9643

def vector_initial : ℝ×ℝ×ℝ := (2, 1, 1)

def vector_result : ℝ×ℝ×ℝ := (-real.sqrt (6 / 11), 3 * real.sqrt (6 / 11), -real.sqrt (6 / 11))

def is_orthogonal (u v : ℝ×ℝ×ℝ) : Prop := 
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3 = 0

def magnitude (v : ℝ×ℝ×ℝ) : ℝ := 
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

theorem rotated_vector :
  let u := vector_initial in
  let v := vector_result in
  magnitude u = real.sqrt 6 ∧ 
  magnitude v = real.sqrt 6 ∧ 
  is_orthogonal u v ∧ 
  u.2 > 0 ∧ v.2 > 0 →
  v = (-real.sqrt (6 / 11), 3 * real.sqrt (6 / 11), -real.sqrt (6 / 11)) :=
by
  intros
  sorry

end rotated_vector_l9_9643


namespace sum_of_interior_angles_l9_9005

theorem sum_of_interior_angles (h : ∀ θ, θ = 40 → θ ∈ exterior_angles) 
: ∑ θ in exterior_angles, interior_angle θ = 1260 :=
sorry

end sum_of_interior_angles_l9_9005


namespace rectangle_diagonal_l9_9118

theorem rectangle_diagonal (k : ℕ) (h1 : 2 * (5 * k + 4 * k) = 72) : 
  (Real.sqrt ((5 * k) ^ 2 + (4 * k) ^ 2)) = Real.sqrt 656 :=
by
  sorry

end rectangle_diagonal_l9_9118


namespace proof_problem_l9_9911

variables {Point : Type} [MetricSpace Point]
variables (A B C H M N K L F J : Point)
variables (Triangle : Point → Point → Point → Prop)
variables (Midpoint : Point → Point → Point → Prop)
variables (LineThroughParallel : Point → Point → Point → Point → Prop)
variables (Circumcircle : Point → Point → Point → Set Point)
variables (Incenter : Point → Point → Point → Point)
variables (IntersectionPoint : Point → Point → Point → Point → Prop)

-- Conditions
variables (hABC : Triangle A B C)
variables (hH : ∃ H, orthocenter H A B C)
variables (hM : Midpoint M A B)
variables (hN : Midpoint N A C)
variables (hInside : ∃ H, polygonInside H [B, M, N, C])
variables (hCircumcircleBMHTangentCNH : tangent (Circumcircle B M H) (Circumcircle C N H))
variables (hParallelHBC : LineThroughParallel H B C K)
variables (hParallelHBC' : LineThroughParallel H B C L)
variables (hIntersectionMKNL : IntersectionPoint F M K ∧ IntersectionPoint F N L)
variables (hIncenterMHN : Incenter J M H N)

-- Main Statement
theorem proof_problem : Distance F J = Distance F A :=
sorry

end proof_problem_l9_9911


namespace multiplication_example_l9_9929

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem multiplication_example :
  ∃ a b : ℕ, sum_of_digits a = sum_of_digits b ∧ a * b = 2231 * 26 :=
by
  use 2231, 26
  unfold sum_of_digits
  have ha : sum_of_digits 2231 = 8 := rfl
  have hb : sum_of_digits 26 = 8 := rfl
  exact ⟨ha, hb, rfl⟩
  sorry

end multiplication_example_l9_9929


namespace choice_of_b_l9_9654

noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x - 2)
noncomputable def g (x : ℝ) : ℝ := f (x + 3)

theorem choice_of_b (b : ℝ) :
  (g (g x) = x) ↔ (b = -4) :=
sorry

end choice_of_b_l9_9654


namespace smallest_prime_after_six_consecutive_nonprimes_l9_9180

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def nonprimes_between (a b : ℕ) : Prop :=
  ∀ n : ℕ, a < n ∧ n < b → ¬ is_prime n

def find_first_nonprime_gap : ℕ :=
  if H : ∃ a b : ℕ, b = a + 7 ∧ nonprimes_between a b ∧ is_prime (b + 1) then
    Classical.choose H
  else
    0

theorem smallest_prime_after_six_consecutive_nonprimes : find_first_nonprime_gap = 97 := by
  sorry

end smallest_prime_after_six_consecutive_nonprimes_l9_9180


namespace chord_bisected_by_point_on_ellipse_l9_9786

/-- Given the ellipse x^2 / 16 + y^2 / 4 = 1, and point P(2, 1) lying on it, 
    and a chord bisected by P, prove the equation of the line where the chord lies. -/
theorem chord_bisected_by_point_on_ellipse :
  ∀ x y, (x * x) / 16 + (y * y) / 4 = 1 →
  ∀ P : ℝ × ℝ, P = (2, 1) →
  ∃ l : ℝ × ℝ → Prop, (∀ A B : ℝ × ℝ, l A ∧ l B ∧ (A.1 + B.1 = 4) ∧ (A.2 + B.2 = 2)) ∧
  (∀ A B : ℝ × ℝ, l A → l B → l (P.1, P.2) ∧ l A ∧ l B → 
    (l = (λ p : ℝ × ℝ, p.1 + 2 * p.2 - 4 = 0))) :=
by 
  sorry

end chord_bisected_by_point_on_ellipse_l9_9786


namespace smallest_prime_after_six_consecutive_nonprimes_l9_9183

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def nonprimes_between (a b : ℕ) : Prop :=
  ∀ n : ℕ, a < n ∧ n < b → ¬ is_prime n

def find_first_nonprime_gap : ℕ :=
  if H : ∃ a b : ℕ, b = a + 7 ∧ nonprimes_between a b ∧ is_prime (b + 1) then
    Classical.choose H
  else
    0

theorem smallest_prime_after_six_consecutive_nonprimes : find_first_nonprime_gap = 97 := by
  sorry

end smallest_prime_after_six_consecutive_nonprimes_l9_9183


namespace bananas_per_box_l9_9477

def total_bananas : ℕ := 40
def number_of_boxes : ℕ := 10

theorem bananas_per_box : total_bananas / number_of_boxes = 4 := by
  sorry

end bananas_per_box_l9_9477


namespace monotonic_increasing_interval_of_f_l9_9981

def f (x : ℝ) : ℝ := Real.logBase 0.6 (6 * x - x * x)

theorem monotonic_increasing_interval_of_f :
 ∀ x, 3 < x ∧ x < 6 → (f (x + 1) > f x) :=
 by
  sorry

end monotonic_increasing_interval_of_f_l9_9981


namespace find_smaller_l9_9149

variable {n m d u : ℝ}

-- Given conditions
axiom pos_n : 0 < n
axiom n_less_m : n < m
axiom sum_condition : ∃x y : ℝ, x + y + u = d ∧ (∃ k : ℝ, x = k * n ∧ y = k * m)

-- The proof goal
theorem find_smaller (x y : ℝ) (k : ℝ) (h_ratio_x: x = k * n) (h_ratio_y: y = k * m) :
  x + y + u = d →
  x = n * (d - u) / (n + m) →
  y = m * (d - u) / (n + m) →
  x < y :=
by
  intro h_sum hx_eq hy_eq
  rw [hx_eq, hy_eq]
  linarith

end find_smaller_l9_9149


namespace sum_partition_possible_l9_9764

theorem sum_partition_possible
  {n m : ℕ} 
  {x : Fin n → ℕ} {y : Fin m → ℕ} 
  (h_sum_eq : (∑ i, x i) = (∑ j, y j)) 
  (h_bound : (∑ i, x i) < m * n) :
  ∃ (A : Finset (Fin n)) (B : Finset (Fin m)), 
    (∑ i in A, x i) = (∑ j in B, y j) :=
by
  sorry

end sum_partition_possible_l9_9764


namespace sum_of_interior_angles_of_regular_polygon_l9_9978

theorem sum_of_interior_angles_of_regular_polygon :
  (∀ (n : ℕ), (n ≠ 0) ∧ ((360 / 45 = n) → (180 * (n - 2) = 1080))) := by sorry

end sum_of_interior_angles_of_regular_polygon_l9_9978


namespace sum_of_repeating_decimals_l9_9275

noncomputable def x := (2 : ℚ) / (3 : ℚ)
noncomputable def y := (5 : ℚ) / (11 : ℚ)

theorem sum_of_repeating_decimals : x + y = (37 : ℚ) / (33 : ℚ) :=
by {
  sorry
}

end sum_of_repeating_decimals_l9_9275


namespace eccentricity_of_conic_section_l9_9392

variable {m : ℝ}

def geometric_mean_condition (a b : ℝ) (m : ℝ) : Prop :=
  m^2 = a * b

def conic_section (a : ℝ) (b : ℝ) : Prop := 
  a = 2 ∧ b = 1 ∨ a = 1 ∧ b = 2

noncomputable def eccentricity (c a : ℝ) : ℝ :=
  c / a

theorem eccentricity_of_conic_section {a b c : ℝ} (h_m : geometric_mean_condition 2 8 m) (h_conic : conic_section a b) :
  (eccentricity c a = sqrt 3 / 2 ∨ eccentricity c a = sqrt 5) :=
sorry

end eccentricity_of_conic_section_l9_9392


namespace cell_count_3_or_more_conditions_meet_l9_9728

theorem cell_count_3_or_more_conditions_meet :
  let n := 50
  ∃ (grid : fin n.succ × fin n.succ → ℕ),
    (∀ (i j : fin n.succ), // For every cell in the grid
      grid (i, j) = 0 ∨ grid (i, j) ≥ 3) ∧ // Each cell contains a number ≥ 3 or is 0
    (∑ i in finset.range n.succ,   
        ∑ j in finset.range n.succ, if grid (i, j) ≥ 3 then 1 else 0) = 1600 := 
begin
  -- Proof details would go here, but they are not required.
  sorry
end

end cell_count_3_or_more_conditions_meet_l9_9728


namespace concyclic_X_O_D_P_l9_9889

theorem concyclic_X_O_D_P 
  (A B C O D E F X P : Point) 
  (hO : is_circumcenter O A B C)
  (hD : midpoint D B C) 
  (hE : midpoint E C A) 
  (hF : midpoint F A B) 
  (hX_cond : ∠ A E X = ∠ A F X)
  (hP : ∃ P, line_through A X ∧ on_circumcircle P A B C):
  concyclic X O D P :=
sorry

end concyclic_X_O_D_P_l9_9889


namespace rich_total_distance_l9_9083

-- Define the given conditions 
def distance_house_to_sidewalk := 20
def distance_down_road := 200
def total_distance_so_far := distance_house_to_sidewalk + distance_down_road
def distance_left_turn := 2 * total_distance_so_far
def distance_to_intersection := total_distance_so_far + distance_left_turn
def distance_half := distance_to_intersection / 2
def total_distance_one_way := distance_to_intersection + distance_half

-- Define the theorem to be proven 
theorem rich_total_distance : total_distance_one_way * 2 = 1980 :=
by 
  -- This line is to complete the 'prove' demand of the theorem
  sorry

end rich_total_distance_l9_9083


namespace integral_x_squared_l9_9999

theorem integral_x_squared :
  ∫ x in -1..1, x^2 = (2 : ℝ)/3 :=
by
  sorry

end integral_x_squared_l9_9999


namespace infinite_n_exist_l9_9077

theorem infinite_n_exist : ∃ᶠ n : ℕ in at_top, ∃ a b c d : ℕ+, gcd (gcd a b) (gcd c d) = 1 ∧ (a, b) ≠ (c, d) ∧ n = a^3 + b^3 ∧ n = c^3 + d^3 :=
begin
  sorry
end

end infinite_n_exist_l9_9077


namespace sum_of_interior_angles_of_regular_polygon_l9_9970

theorem sum_of_interior_angles_of_regular_polygon (n : ℕ) (h1: ∀ {n : ℕ}, ∃ (e : ℕ), (e = 360 / 45) ∧ (n = e)) :
  (180 * (n - 2)) = 1080 :=
by
  let n := (360 / 45)
  have h : n = 8 := by sorry
  calc
    180 * (n - 2) = 180 * (8 - 2) : by rw [h]
    ... = 1080 : by norm_num

end sum_of_interior_angles_of_regular_polygon_l9_9970


namespace trains_cross_platform_l9_9220

theorem trains_cross_platform
  (T_A_cross_platform : ℝ)
  (T_A_cross_pole : ℝ)
  (T_B_cross_platform : ℝ)
  (T_B_cross_pole : ℝ)
  (length_A : ℝ)
  (length_B : ℝ) :
  T_A_cross_platform = 38 ∧ T_A_cross_pole = 18 ∧ T_B_cross_platform = 54 ∧ T_B_cross_pole = 30 ∧ length_A = 300 ∧ length_B = 450 →
  ∃ (length_P : ℝ) (time_cross_platform : ℝ), length_P = 333.46 ∧ time_cross_platform ≈ 34.21 :=
by
  sorry

end trains_cross_platform_l9_9220


namespace older_sister_faster_than_younger_l9_9579

-- Define variables related to the problem
variables (v_o v_y : ℝ)
variables (x : ℝ)

-- Conditions from the problem
def conditions :=
  (4 > 0) ∧ (3.5 > 0) ∧ (x > 1.75) ∧ (3 = x * (3.5 - x))

-- Given the conditions, we are to prove the speed ratio
theorem older_sister_faster_than_younger :
  conditions →
  v_o = 1.5 * v_y :=
sorry

end older_sister_faster_than_younger_l9_9579


namespace graph_symmetric_l9_9830

theorem graph_symmetric {f : ℝ → ℝ} (h : ∀ x, f x = -4 ^ (-x)) :
  (∀ x y x' y' : ℝ, 
    (f x = y) →
    (y = log 4 x) → 
    ((y' - y) / (x' - x) = -1) ∧ ((x' + x) / 2 + (y' + y) / 2 = 0) →
    x' = -y ∧ y' = -x) :=
by {
  sorry
}

end graph_symmetric_l9_9830


namespace triangle_perimeter_is_eight_l9_9775

theorem triangle_perimeter_is_eight:
  let trin_eq : Polynomial ℝ := Polynomial.C 3 - Polynomial.X * Polynomial.C 4 + Polynomial.X^2 in
  let roots := trin_eq.roots in
  ∀ (a b : ℝ), a = 2 → b = 3 →
  (roots = {c | (a + b > c) ∧ (c + a > b) ∧ (c + b > a)}) →
  (a + b + (roots.to_finset.filter (λ c, a + b > c )).any) = 8 :=
by
  sorry

end triangle_perimeter_is_eight_l9_9775


namespace cost_price_per_meter_l9_9201

def total_length : ℝ := 9.25
def total_cost : ℝ := 397.75

theorem cost_price_per_meter : total_cost / total_length = 43 := sorry

end cost_price_per_meter_l9_9201


namespace primes_sum_to_80_l9_9300

open Nat

def is_sum_of_two_primes (n : ℕ) : Prop :=
  ∃ (a b : ℕ), Prime a ∧ Prime b ∧ a + b = n

theorem primes_sum_to_80 :
  (∃ (a b : ℕ), Prime a ∧ Prime b ∧ a + b = 80 ∧ a ≠ b) →
  ∃! (a b c d: ℕ), Prime a ∧ Prime b ∧ Prime c ∧ Prime d ∧
  a + b = 80 ∧ c + d = 80 ∧
  ((a = 7 ∧ b = 73) ∨ (a = 13 ∧ b = 67) ∨ (a = 19 ∧ b = 61) ∨ (a = 37 ∧ b = 43))
:=
begin
  sorry
end

end primes_sum_to_80_l9_9300


namespace marian_cookies_l9_9511

theorem marian_cookies (total_cookies trays : ℕ) (h_total_cookies : total_cookies = 276) (h_trays : trays = 23) :
  total_cookies / trays = 12 := by
  rw [h_total_cookies, h_trays]
  norm_num
  sorry

end marian_cookies_l9_9511


namespace ratio_d_e_l9_9555

-- Condition definitions
def roots : List ℝ := [1, -1/2, 3, 4]
def a : ℝ := 1  -- a is not given a specific non-zero value, we assume it to be 1 for simplicity
def d := sum [1 * 3 * 4, 1 * (-1/2) * 3, 1 * (-1/2) * 4, 3 * 4 * (-1/2)] * a
def e := -(1 * (-1/2) * 3 * 4) * a

-- Theorem statement
theorem ratio_d_e : d / e = -5 / 12 := by sorry

end ratio_d_e_l9_9555


namespace find_x_l9_9047

def infinite_sqrt (d : ℝ) : ℝ := sorry -- A placeholder since infinite nesting is non-trivial

def bowtie (c d : ℝ) : ℝ := c - infinite_sqrt d

theorem find_x (x : ℝ) (h : bowtie 7 x = 3) : x = 20 :=
sorry

end find_x_l9_9047


namespace smallest_prime_after_six_nonprimes_l9_9163

open Nat

/-- 
  Proposition:
    97 is the smallest prime number that occurs after a sequence 
    of six consecutive positive integers all of which are nonprime.
-/
theorem smallest_prime_after_six_nonprimes : ∃ (n : ℕ), (n > 0 ∧ Prime (n + 97)) ∧ 
  (∀ k : ℕ, k < n → (Nonprime k) ∧ Nonprime (k+1) ∧ Nonprime (k+2) ∧ Nonprime (k+3) ∧ Nonprime (k+4) ∧ Nonprime (k+5)) ∧ 
  (Prime (n + 1) → n + 1 = 97) := 
by 
  sorry

end smallest_prime_after_six_nonprimes_l9_9163


namespace a_n_expression_S_n_expression_T_n_expression_l9_9401

variable {a_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}
variable {b_n : ℕ → ℕ}
variable {T_n : ℕ → ℕ}
variable (n : ℕ)

-- Assumptions
axiom a3_eq : a_n 3 = 7
axiom a5_a7_sum_eq : a_n 5 + a_n 7 = 26
axiom a_arith_seq : ∃ a_1 d, ∀ n, a_n n = a_1 + (n - 1) * d

-- The proof statements
theorem a_n_expression : ∀ n, a_n n = 2 * n + 1 :=
sorry

theorem S_n_expression : ∀ n, S_n n = n^2 + 2 * n :=
sorry

noncomputable def b_n (n : ℕ) : ℕ := 1 / ((a_n n)^2 - 1)

theorem T_n_expression : ∀ n, T_n n = n / (4 * (n + 1)) :=
sorry

end a_n_expression_S_n_expression_T_n_expression_l9_9401


namespace limit_of_fraction_l9_9310

variable (a : Real) 

theorem limit_of_fraction (a : Real) : 
  tendsto (fun z => (a * z + 1) / z) atTop (nhds a) := 
by
  sorry

end limit_of_fraction_l9_9310


namespace circle_symmetric_D_plus_E_l9_9825

-- Definitions from the conditions
def center_of_circle (D E : ℝ) : ℝ × ℝ := (-D / 2, -E / 2)

def line_l1 (x y : ℝ) : Prop := x - y + 4 = 0
def line_l2 (x y : ℝ) : Prop := x + 3 * y = 0

-- The theorem statement
theorem circle_symmetric_D_plus_E 
  (D E F : ℝ) 
  (h_l1 : line_l1 (-D / 2) (-E / 2)) 
  (h_l2 : line_l2 (-D / 2) (-E / 2))
  (h_circle : ∃ x y, x^2 + y^2 + D * x + E * y + F = 0 ) : 
  D + E = 4 :=
sorry

end circle_symmetric_D_plus_E_l9_9825


namespace find_a_l9_9957

noncomputable def question (a : ℝ) : Prop :=
  ∀ x : ℝ,  y = real.exp (2 * a * x)

noncomputable def condition (a : ℝ) : Prop :=
  (∃ y : ℝ, y - real.exp (2 * a * 0) = 0)

noncomputable def tangent_perpendicular (a : ℝ) : Prop :=
  let slope_of_line := -1/2 in
  let slope_of_tangent := 2 * a in
  slope_of_tangent = slope_of_line

theorem find_a (a : ℝ) (h_tangent_perpendicular: tangent_perpendicular a): 
a = -1/4 :=
begin
  sorry
end

end find_a_l9_9957


namespace people_in_rooms_l9_9577

theorem people_in_rooms (x y : ℕ) (h1 : x + y = 76) (h2 : x - 30 = y - 40) : x = 33 ∧ y = 43 := by
  sorry

end people_in_rooms_l9_9577


namespace sum_of_squares_of_ten_consecutive_numbers_not_perfect_square_l9_9531

theorem sum_of_squares_of_ten_consecutive_numbers_not_perfect_square :
  ∀ (x : ℤ), ¬ ∃ (k : ℤ), 
    k^2 = (x-2)^2 + (x-1)^2 + x^2 + (x+1)^2 + (x+2)^2 + (x+3)^2 + (x+4)^2 + (x+5)^2 + (x+6)^2 + (x+7)^2 :=
by
  assume x,
  assume ⟨k, hk⟩,
  let N := (x-2)^2 + (x-1)^2 + x^2 + (x+1)^2 + (x+2)^2 + (x+3)^2 + (x+4)^2 + (x+5)^2 + (x+6)^2 + (x+7)^2,
  have hN : N = 10*x^2 + 50*x + 145, sorry,
  have factored_N : N = 5 * (2*x^2 + 10*x + 29), sorry,
  have hdiv : ¬ (2*x^2 + 10*x + 29) % 5 = 0, sorry,
  exact mt (λh, eq.trans (congr_arg _ h) hk) hdiv

end sum_of_squares_of_ten_consecutive_numbers_not_perfect_square_l9_9531


namespace common_ratio_is_3_l9_9025

open nat

-- Assume a geometric sequence {a_n} and its sum sequence {S_n}
variables {a : ℕ → ℝ} {S : ℕ → ℝ}
-- Assume common ratio q
variable {q : ℝ}

-- Conditions given in the problem
axiom h1 : a 5 = 2 * S 4 + 3
axiom h2 : a 6 = 2 * S 5 + 3
axiom geom_seq : ∀ n, a (n+1) = q * a n
axiom sum_geom_seq : ∀ n, S n = a 0 * (1 - q^(n+1)) / (1 - q)  -- sum formula for geometric series

-- The target theorem we want to prove
theorem common_ratio_is_3 : q = 3 :=
by {
    sorry
}

end common_ratio_is_3_l9_9025


namespace eccentricity_of_hyperbola_l9_9902

noncomputable def hyperbola (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}

theorem eccentricity_of_hyperbola (a b c : ℝ) (P Q F1 F2 M O : ℝ × ℝ)
  (h_a : a > 0) (h_b : b > 0)
  (h_F1 : F1 = (-c, 0)) (h_F2 : F2 = (c, 0))
  (h_P : P = (c, b^2 / a))
  (h_perpendicular : P.1 * F2.1 + P.2 * F2.2 = 0)
  (h_intersect : ∃ y, P.1 * y - F1.1 = 0)
  (h_inscribed_circle : (M = (c / 2, c / 2)) ∧ (abs ((3 * b^2 * c) / 2 - a * c ^ 2) / sqrt (4 * a^2 * c ^ 2 + b ^ 4) = c / 2)) :
  c = 2 * a :=
sorry

end eccentricity_of_hyperbola_l9_9902


namespace angles_geometric_sequence_l9_9288

theorem angles_geometric_sequence : 
  ∀ θ ∈ Ico 0 (2 * Real.pi),
  (θ % (Real.pi / 2) ≠ 0) →
  ((Real.sin θ) ^ 2 = (Real.cos θ) ^ 4 ∨ (Real.cos θ) ^ 4 = Real.sin θ) →
  ∃4 θs, θs = θ := sorry

end angles_geometric_sequence_l9_9288


namespace largest_shaded_area_of_figure_C_l9_9283

noncomputable def pi := Real.pi

def shaded_area_A : ℝ := 9 - 2.25 * pi
def shaded_area_B : ℝ := 9 - 2.25 * pi
def shaded_area_C : ℝ := 4 + pi

theorem largest_shaded_area_of_figure_C : 
  shaded_area_C > shaded_area_A ∧ shaded_area_C > shaded_area_B :=
by
  sorry

end largest_shaded_area_of_figure_C_l9_9283


namespace number_of_sets_including_six_with_sum_18_l9_9582

theorem number_of_sets_including_six_with_sum_18 :
  let numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  in let subsets := {s ∈ powerset numbers | s.card = 3 ∧ 6 ∈ s ∧ s.sum = 18}
  in subset.card subsets = 3 :=
by
  sorry

end number_of_sets_including_six_with_sum_18_l9_9582


namespace sin_minus_cos_eq_neg_sqrt_five_halves_l9_9824

theorem sin_minus_cos_eq_neg_sqrt_five_halves
  (θ : ℝ)
  (h1 : θ > 0)
  (h2 : θ < π)
  (h3 : sin θ * cos θ = -1 / 8) :
  sin θ - cos θ = -sqrt 5 / 2 :=
sorry

end sin_minus_cos_eq_neg_sqrt_five_halves_l9_9824


namespace probability_of_selecting_A_and_B_l9_9837

theorem probability_of_selecting_A_and_B :
  let students := Finset.range 5
  let total_ways := students.card.choose 3
  let favorable_ways := 3
  let probability := favorable_ways / total_ways in
  probability = (3:ℚ) / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l9_9837


namespace incorrect_operation_B_l9_9591

theorem incorrect_operation_B : (4 + 5)^2 ≠ 4^2 + 5^2 := 
  sorry

end incorrect_operation_B_l9_9591


namespace complex_exp_form_pow_four_l9_9668

theorem complex_exp_form_pow_four :
  let θ := 30 * Real.pi / 180
  let cos_θ := Real.cos θ
  let sin_θ := Real.sin θ
  let z := 3 * (cos_θ + Complex.I * sin_θ)
  z ^ 4 = -40.5 + 40.5 * Complex.I * Real.sqrt 3 :=
by
  sorry

end complex_exp_form_pow_four_l9_9668


namespace complex_power_result_l9_9683

theorem complex_power_result :
  (3 * Complex.cos (Real.pi / 6) + 3 * Complex.sin (Real.pi / 6) * Complex.i)^4 = 
  -40.5 + 40.5 * Complex.i * Real.sqrt 3 :=
by sorry

end complex_power_result_l9_9683


namespace min_f_value_l9_9311

def f (x y : ℝ) : ℝ := (6 * x^2 + 9 * x + 2 * y^2 + 3 * y + 20) / (9 * (x + y + 2))

theorem min_f_value :
  (∀ x y : ℝ, 0 ≤ x ∧ 0 ≤ y) → (∃ x y : ℝ, f x y ≤ f 0 0) → 
    (∃ x y : ℝ, f x y = (4 * Real.sqrt 10) / 3) :=
by sorry

end min_f_value_l9_9311


namespace ellipse_equation_line_AB_equation_l9_9559

theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (h1 : a > b) (h2 : 3 * a = b^2) (hPF1 : Real.dist (2, 3) F1 = 5) (hPF2 : Real.dist (2, 3) F2 = 3) 
  (F1 F2 : ℝ × ℝ) :
  ∃ x y : ℝ, (x, y) ∈ { (x, y) | x^2 / 16 + y^2 / 12 = 1 } :=
by
  sorry

theorem line_AB_equation (A B : ℝ × ℝ) (hAonE : (A.1 ^ 2 / 16 + A.2 ^ 2 / 12 = 1)) (hBonE : (B.1 ^ 2 / 16 + B.2 ^ 2 / 12 = 1))
  (P : ℝ × ℝ) (hP : P = (2, 3)) (hAngle : ∃ k : ℝ, ∠(A, P, F2) = ∠(B, P, F2))
  (h_through : ∃ l : ℝ, (1, -1) ∈ { (x, y) | y + l = (1 / 2) * (x - 1) }) :
  ∃ x y : ℝ, (x, y) ∈ { (x, y) | x - 2 * y - 3 = 0 } :=
by
  sorry

end ellipse_equation_line_AB_equation_l9_9559


namespace B_pow_5_eq_rB_plus_sI_l9_9046

def B : Matrix (Fin 2) (Fin 2) ℤ := !![1, 1; 4, 5]

def I : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 0, 1]

theorem B_pow_5_eq_rB_plus_sI : 
  ∃ (r s : ℤ), r = 1169 ∧ s = -204 ∧ B^5 = r • B + s • I := 
by
  use 1169
  use -204
  sorry

end B_pow_5_eq_rB_plus_sI_l9_9046


namespace sandy_marks_l9_9946

def marks_each_correct_sum : ℕ := 3

theorem sandy_marks (x : ℕ) 
  (total_attempts : ℕ := 30)
  (correct_sums : ℕ := 23)
  (marks_per_incorrect_sum : ℕ := 2)
  (total_marks_obtained : ℕ := 55)
  (incorrect_sums : ℕ := total_attempts - correct_sums)
  (lost_marks : ℕ := incorrect_sums * marks_per_incorrect_sum) :
  (correct_sums * x - lost_marks = total_marks_obtained) -> x = marks_each_correct_sum :=
by
  sorry

end sandy_marks_l9_9946


namespace solve_inequality_l9_9992

theorem solve_inequality (x : ℝ) : 
  (x - 2) / (x + 1) ≤ 2 ↔ x ∈ Set.Iic (-4) ∪ Set.Ioi (-1) := 
sorry

end solve_inequality_l9_9992


namespace line_through_point_inequality_l9_9776

theorem line_through_point_inequality (a b θ : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
(hline : a * sin θ + b * cos θ = a * b) : 
  (1 / a^2) + (1 / b^2) ≥ 1 :=
sorry

end line_through_point_inequality_l9_9776


namespace length_BE_l9_9459

def square (A B C D : Point) (side_length : ℝ) : Prop := 
  -- definition of square with side length of 2
  sorry

def is_congruent (R1 R2 : set Point) : Prop := 
  -- congruence definition for two rectangles
  sorry

def parallel (l1 l2 : Line) : Prop := 
  -- definition of parallel lines
  sorry
  
def perpendicular (l1 l2 : Line) : Prop := 
  -- definition of perpendicular lines
  sorry

def length (P Q : Point) : ℝ := 
  -- a function that calculates the distance between points P and Q
  sorry

theorem length_BE (A B C D E F J K G H : Point) (h w : ℝ) 
  (sq : square A B C D 2)
  (cong : is_congruent (set_of_Points J K H G) (set_of_Points E B C F))
  (par : parallel (line_of_Points B E) (line_of_Points C F))
  (perp : perpendicular (line_of_Points E J) (line_of_Points B E))
  (EJ_eq_h : length E J = h)
  (CF_eq_w : length C F = w)
  (BF_eq_FC : length B F = length F C) : 
  length B E = 2 / 3 :=
by
  sorry

end length_BE_l9_9459


namespace pyramid_section_area_ratio_l9_9567

theorem pyramid_section_area_ratio
  (ABCDE : Set Point)
  (A B C D E : Point)
  (ABCD_rect : isRectangle ABCD)
  (no_obtuse_lateral_faces : ∀ P Q R ∈ {A, B, C, D, E} , ¬isObtuseTriangle P Q R)
  (M : Point)
  (M_on_DC : M ∈ segment D C)
  (EM_perp_BC : isPerpendicular (lineThrough E M) (lineThrough B C))
  (relations : ∣distance A C∣ ≥ (5 / 4) * ∣distance E B∣ ∧ ∣distance E B∣ ≥ (5 / 3) * ∣distance E D∣)
  (cross_section_iso_trap : ∃ P Q R S, isIsoscelesTrapezoid P Q R S ∧ P = B ∧ ∃ mid, {Q, R} ⊆ lateralEdges ABCDE ∧ isMidpoint P Q mid) :
∣sectionArea ABCDE (cross_section_iso_trap.left)∣ / ∣baseArea ABCD∣ = (3 / 5) * sqrt(65 / 14) :=
sorry

end pyramid_section_area_ratio_l9_9567


namespace cant_form_set_l9_9649

def forms_set {α : Type*} (s : α → Prop) : Prop :=
  ∃ t : set α, ∀ x, x ∈ t ↔ s x

def is_well_defined (s : Prop) : Prop :=
  ∃ x, s

-- Definition for students in Beihai No.7 Middle School
def beihai_no7_students : set ℕ := {1, 2, 3, 4, 5}  -- Example set of students identified by IDs

-- Definition for 'taller' which must be subjective and thus ill-defined
def taller_students (students : set ℕ) : Prop := 
  ∃ h : ℕ -> ℕ -> Prop, (∀ x ∈ students, ∀ y ∈ students, x ≠ y → ¬(h x y ∨ h y x))

theorem cant_form_set : ¬ forms_set (λ x, x ∈ beihai_no7_students ∧ taller_students beihai_no7_students) :=
by sorry

end cant_form_set_l9_9649


namespace parallel_planes_k_value_l9_9396

-- Define the normal vectors for planes alpha and beta
def normal_vector_alpha : ℝ × ℝ × ℝ := (1, 2, -2)
def normal_vector_beta (k : ℝ) : ℝ × ℝ × ℝ := (-2, -4, k)

-- Define the condition that alpha is parallel to beta
def parallel (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ λ : ℝ, λ ≠ 0 ∧ a = (λ * b.1, λ * b.2, λ * b.3)

-- The main statement: prove that k = 4
theorem parallel_planes_k_value : ∀ k : ℝ,
  parallel normal_vector_alpha (normal_vector_beta k) → k = 4 :=
by
  intros k h
  sorry

end parallel_planes_k_value_l9_9396


namespace value_of_x_l9_9599

def condition (x : ℝ) : Prop :=
  3 * x = (20 - x) + 20

theorem value_of_x : ∃ x : ℝ, condition x ∧ x = 10 := 
by
  sorry

end value_of_x_l9_9599


namespace multiple_is_2_l9_9984

noncomputable def find_multiple_of_sum (a b : ℕ) (m : ℕ) : Prop :=
  a * b = m * (a + b) + 10

theorem multiple_is_2 (a b m : ℕ) (h1 : b = 9) (h2 : b - a = 5) (h3 : find_multiple_of_sum a b m) : m = 2 :=
by
  subst h1
  have ha : a = 4 := by linarith
  subst ha
  sorry

end multiple_is_2_l9_9984


namespace first_term_is_5_over_2_l9_9045

-- Define the arithmetic sequence and the sum of the first n terms.
def arith_seq (a d : ℕ) (n : ℕ) := a + (n - 1) * d
def S (a d : ℕ) (n : ℕ) := (n * (2 * a + (n - 1) * d)) / 2

-- Define the constant ratio condition.
def const_ratio (a d : ℕ) (n : ℕ) (c : ℕ) :=
  (S a d (3 * n) * 2) = c * (S a d n * 2)

-- Prove the first term is 5/2 given the conditions.
theorem first_term_is_5_over_2 (c : ℕ) (n : ℕ) (h : const_ratio a 5 n 9) : 
  a = 5 / 2 :=
sorry

end first_term_is_5_over_2_l9_9045


namespace intersection_of_A_and_B_l9_9041

-- Define the set A
def A : set ℝ := { x | x^2 + x - 6 < 0 }

-- Define the set B
def B : set ℝ := { x | x + 1 > 0 }

-- Statement of the proof problem
theorem intersection_of_A_and_B : A ∩ B = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

end intersection_of_A_and_B_l9_9041


namespace exists_2016_consecutive_with_16_primes_l9_9725

theorem exists_2016_consecutive_with_16_primes :
  ∃ n : ℕ, (finset.Icc n (n + 2015)).filter prime.card = 16 :=
sorry

end exists_2016_consecutive_with_16_primes_l9_9725


namespace expected_participants_2008_l9_9458

theorem expected_participants_2008 (initial_participants : ℕ) (annual_increase_rate : ℝ) :
  initial_participants = 1000 ∧ annual_increase_rate = 1.25 →
  (initial_participants * annual_increase_rate ^ 3) = 1953.125 :=
by
  sorry

end expected_participants_2008_l9_9458


namespace total_spent_correct_l9_9527

variable (P' : ℝ) (C : ℝ) (Q : ℝ) (T : ℝ)

-- Given conditions
def condition1 : P' = 12.32 := by sorry
def condition2 : C = 3 := by sorry
def condition3 : Q = 11.54 := by sorry
def condition4 : T = 26.86 := by sorry

-- Define the original price of the peaches
def original_price_peaches : ℝ := P' + C

-- Define the total amount spent
def total_spent : ℝ := original_price_peaches + Q

-- Prove the total amount spent equals to the correct answer
theorem total_spent_correct : total_spent = 26.86 := by
  -- Use given conditions
  rw [←condition1, ←condition2, ←condition3]
  -- Calculate the total
  let P := (12.32 + 3 : ℝ)
  let T := (P + 11.54 : ℝ)
  exact condition4

end total_spent_correct_l9_9527


namespace smallest_sector_angle_l9_9922

/-
  Prove that the measure of the smallest possible sector angle is 32 degrees,
  given the conditions that:
  - Mary divides a circle into 9 sectors.
  - The central angles are even integers.
  - The angles form an arithmetic sequence.
  - The sum of these angles is 360 degrees.
-/

theorem smallest_sector_angle (a1 d : ℤ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ 9 → even (a1 + (i - 1) * d))
  (h2 : finset.sum (finset.range 9) (λ i, a1 + i * d) = 360) : a1 = 32 := 
by
  sorry

end smallest_sector_angle_l9_9922


namespace mr_roper_lawn_cuts_l9_9516

theorem mr_roper_lawn_cuts :
  (∀ (x : ℕ), (15 * 6 + 6 * x) / 12 = 9) → x = 3 :=
begin
  sorry,
end

end mr_roper_lawn_cuts_l9_9516


namespace circle_tangent_y_eq_2_center_on_y_axis_radius_1_l9_9552

theorem circle_tangent_y_eq_2_center_on_y_axis_radius_1 :
  ∃ (y0 : ℝ), (∀ x y : ℝ, (x - 0)^2 + (y - y0)^2 = 1 ↔ y = y0 + 1 ∨ y = y0 - 1) := by
  sorry

end circle_tangent_y_eq_2_center_on_y_axis_radius_1_l9_9552


namespace a_cubed_value_l9_9542

theorem a_cubed_value (a b : ℝ) (k : ℝ) (h1 : a^3 * b^2 = k) (h2 : a = 5) (h3 : b = 2) : 
  ∃ (a : ℝ), (64 * a^3 = 500) → (a^3 = 125 / 16) :=
by
  sorry

end a_cubed_value_l9_9542


namespace minimize_perimeter_l9_9707

-- Consider point foot_of_altitude at side BC as A′, point foot_of_altitude at side AC as B′, and point foot_of_altitude at side AB as C'
noncomputable def foot_of_altitude (A B C : Point) : Point := -- Implementation depends on geometry library
sorry

-- Theorem to prove 
theorem minimize_perimeter (A B C A' B' C' : Point) 
  (hA : A' = foot_of_altitude A B C) 
  (hB : B' = foot_of_altitude B A C)
  (hC : C' = foot_of_altitude C A B)
  (acuteABC : acute_triangle A B C)
  :
  ∀ (A'' B'' C'' : Point), 
  on_segment A'' B C ∧ on_segment B'' A C ∧ on_segment C'' A B → 
  perimeter (A' B' C') ≤ perimeter (A'' B'' C'')
:= sorry

end minimize_perimeter_l9_9707


namespace original_function_l9_9092

theorem original_function (x : ℝ) :
  (∃ f : ℝ → ℝ, 
    (∀ x, (cos (x + π / 4) = f (x / 2 - π / 6)) ∧
          f x = cos (2x + 5 * π / 12))) := sorry

end original_function_l9_9092


namespace equal_products_l9_9758

noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def incenter (A B C: Point) : Point := sorry

variables (A B C D E F A' B' C' I O : Point)

-- Conditions given in the problem
variable (circumcircle : Circle := circumcenter O A B C)

-- Definitions based on the conditions
def AI := line_through A I
def BI := line_through B I
def CI := line_through C I

def D := intersection (line_through A I) (line_through B C)
def E := intersection (line_through B I) (line_through C A)
def F := intersection (line_through C I) (line_through A B)

def AD := line_through A D
def BE := line_through B E
def CF := line_through C F

def A' := second_intersection AD circumcircle
def B' := second_intersection BE circumcircle
def C' := second_intersection CF circumcircle

-- The proof problem
theorem equal_products : AA' * ID = BB' * IE = CC' * IF := sorry

end equal_products_l9_9758


namespace smallest_prime_after_six_nonprimes_l9_9188

open Nat

theorem smallest_prime_after_six_nonprimes : 
  ∃ p : ℕ, prime p ∧ p > 0 ∧
  (∃ n : ℕ, ¬prime n ∧ ¬prime (n+1) ∧ ¬prime (n+2) ∧ ¬prime (n+3) ∧ ¬prime (n+4) ∧ ¬prime (n+5) ∧ prime (n+6) ∧ n+6 = p) ∧ 
  p = 89 :=
sorry

end smallest_prime_after_six_nonprimes_l9_9188


namespace find_direction_vector_of_line_l9_9307

noncomputable def reflection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![(_3 / 5), (_4 / 5)], ![(_4 / 5), (-(_3 / 5))]]

def direction_vector (a b : ℤ) : Prop :=
  ∀ x y : ℚ, a > 0 ∧ Int.gcd a (Int.natAbs b) = 1 ∧ 
  reflection_matrix.mulVec ![a, b] = ![a, b]

theorem find_direction_vector_of_line : direction_vector 2 1 :=
by
  sorry

end find_direction_vector_of_line_l9_9307


namespace prove_expression_l9_9819

noncomputable def omega := Complex.exp (2 * Real.pi * Complex.I / 5)

lemma root_of_unity : omega^5 = 1 := sorry
lemma sum_of_roots : omega^0 + omega + omega^2 + omega^3 + omega^4 = 0 := sorry

noncomputable def z := omega + omega^2 + omega^3 + omega^4

theorem prove_expression : z^2 + z + 1 = 1 :=
by 
  have h1 : omega^5 = 1 := root_of_unity
  have h2 : omega^0 + omega + omega^2 + omega^3 + omega^4 = 0 := sum_of_roots
  show z^2 + z + 1 = 1
  {
    -- Proof omitted
    sorry
  }

end prove_expression_l9_9819


namespace smallest_sum_of_inverses_l9_9367

theorem smallest_sum_of_inverses 
  (x y : ℕ) (hx : x ≠ y) (h1 : 0 < x) (h2 : 0 < y) (h_condition : (1 / x : ℚ) + 1 / y = 1 / 15) :
  x + y = 64 := 
sorry

end smallest_sum_of_inverses_l9_9367


namespace pair_ab_l9_9954

def students_activities_ways (n_students n_activities : Nat) : Nat :=
  n_activities ^ n_students

def championships_outcomes (n_championships n_students : Nat) : Nat :=
  n_students ^ n_championships

theorem pair_ab (a b : Nat) :
  a = students_activities_ways 4 3 ∧ b = championships_outcomes 3 4 →
  (a, b) = (3^4, 4^3) := by
  sorry

end pair_ab_l9_9954


namespace percent_by_weight_Liquid_X_is_correct_l9_9920

def solution_A_percentage := 0.8 / 100
def solution_B_percentage := 1.8 / 100
def solution_A_temp := 40
def solution_B_temp := 20
def temp_factor := 0.2 / 100
def amount_solution_A := 300
def amount_solution_B := 700

def adjusted_solution_A_percentage :=
  solution_A_percentage - (temp_factor * ((solution_A_temp - solution_B_temp)/5))
def adjusted_solution_B_percentage :=
  solution_B_percentage + (temp_factor * ((solution_A_temp - solution_B_temp)/5))

def amount_Liquid_X_in_solution_A := adjusted_solution_A_percentage * amount_solution_A
def amount_Liquid_X_in_solution_B := adjusted_solution_B_percentage * amount_solution_B

def total_Liquid_X := amount_Liquid_X_in_solution_A + amount_Liquid_X_in_solution_B
def total_weight_mixed_solution := amount_solution_A + amount_solution_B

def percent_by_weight_Liquid_X := (total_Liquid_X / total_weight_mixed_solution) * 100

theorem percent_by_weight_Liquid_X_is_correct :
  percent_by_weight_Liquid_X = 1.82 := by
  sorry

end percent_by_weight_Liquid_X_is_correct_l9_9920


namespace smallest_prime_after_six_consecutive_nonprimes_l9_9182

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def nonprimes_between (a b : ℕ) : Prop :=
  ∀ n : ℕ, a < n ∧ n < b → ¬ is_prime n

def find_first_nonprime_gap : ℕ :=
  if H : ∃ a b : ℕ, b = a + 7 ∧ nonprimes_between a b ∧ is_prime (b + 1) then
    Classical.choose H
  else
    0

theorem smallest_prime_after_six_consecutive_nonprimes : find_first_nonprime_gap = 97 := by
  sorry

end smallest_prime_after_six_consecutive_nonprimes_l9_9182


namespace perimeter_of_flowerbed_l9_9630

def width : ℕ := 4
def length : ℕ := 2 * width - 1
def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

theorem perimeter_of_flowerbed : perimeter length width = 22 := by
  sorry

end perimeter_of_flowerbed_l9_9630


namespace equal_focal_distances_condition_l9_9562

theorem equal_focal_distances_condition (k : ℝ) : 
  (∀ x y : ℝ, (9*x^2 + 25*y^2 = 225) → 
              (∀ x y : ℝ, (x^2/(16-k) - y^2/k = 1) → 
              (2*sqrt(25 - 9) = 2*sqrt(16)))) ↔ 0 < k ∧ k < 16 :=
by
  sorry

end equal_focal_distances_condition_l9_9562


namespace area_of_hall_l9_9110

-- Define the conditions
def length := 25
def breadth := length - 5

-- Define the area calculation
def area := length * breadth

-- The statement to prove
theorem area_of_hall : area = 500 :=
by
  sorry

end area_of_hall_l9_9110


namespace hyperbola_eccentricity_is_sqrt13_l9_9796

noncomputable def eccentricity_of_hyperbola {a b : ℝ} (ha : a > 0) (hb : b > 0)
    (h_area : 2 * Real.sqrt 3 = (1:ℝ) * b / a * (2:ℝ)) : ℝ :=
  let c := Real.sqrt (a^2 + b^2) in
  c / a

theorem hyperbola_eccentricity_is_sqrt13 
    (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
    (h_parabola : ∀ y x : ℝ, y^2 = 4 * x) 
    (h_area : 2 * Real.sqrt 3 = (1:ℝ) * b / a * (2:ℝ)) : 
    eccentricity_of_hyperbola ha hb h_area = Real.sqrt 13 := 
begin
  -- Proof goes here
  sorry
end

end hyperbola_eccentricity_is_sqrt13_l9_9796


namespace complex_exp_form_pow_four_l9_9671

theorem complex_exp_form_pow_four :
  let θ := 30 * Real.pi / 180
  let cos_θ := Real.cos θ
  let sin_θ := Real.sin θ
  let z := 3 * (cos_θ + Complex.I * sin_θ)
  z ^ 4 = -40.5 + 40.5 * Complex.I * Real.sqrt 3 :=
by
  sorry

end complex_exp_form_pow_four_l9_9671


namespace sum_a_n_1_to_100_l9_9404

def f (n : ℕ) : ℤ := (n ^ 2 : ℕ) * Int.ofReal (Real.cos (Real.pi * n))

def a_n (n : ℕ) : ℤ := f n + f (n + 1)

noncomputable def a_sum : ℤ :=
  (List.range 100).map (λ n, a_n (n + 1)).sum

theorem sum_a_n_1_to_100 : a_sum = -100 := sorry

end sum_a_n_1_to_100_l9_9404


namespace minimum_points_forming_isosceles_minimum_n_value_is_6_l9_9242

def point : Type := ℕ × ℕ

def divide_triangle (triangle : Type) (n : ℕ) : set point :=
  -- This would be the set of 15 lattice points generated by dividing a regular triangle
  sorry 

def is_isosceles (a b c : point) : Prop :=
  -- Definition for an isosceles triangle given three points
  sorry 

theorem minimum_points_forming_isosceles (n : ℕ) (points : set point)
  (h_triangle : divide_triangle triangle 4 = points)
  (h_points_count : points.card = 15) :
  (∃ points_chosen : finset point, points_chosen.card = n)
  → ∃ a b c ∈ points_chosen, is_isosceles a b c :=
begin
  sorry
end

theorem minimum_n_value_is_6 :
  ∃ points : set point, ∃ n, minimum_points_forming_isosceles 6 points :=
begin
  sorry
end

end minimum_points_forming_isosceles_minimum_n_value_is_6_l9_9242


namespace total_cost_l9_9231

theorem total_cost (a b : ℕ) : 30 * a + 20 * b = 30 * a + 20 * b :=
by
  sorry

end total_cost_l9_9231


namespace log2_sqrt_gt_2_necessary_not_sufficient_l9_9327

variable {x : Real}

lemma log2_sqrt_gt_2_implies_x_gt_2 (h : log2 (x^2) > 2) : x > 2 :=
sorry

lemma log2_sqrt_gt_2_not_sufficient_x_gt_2 : (log2 (x^2) > 2) → ¬(x > 2) :=
sorry

theorem log2_sqrt_gt_2_necessary_not_sufficient (h : log2 (x^2) > 2) : (x > 2 ∧ (¬((log2 (x^2) > 2) → (x > 2)))) :=
  ⟨log2_sqrt_gt_2_implies_x_gt_2 h, log2_sqrt_gt_2_not_sufficient_x_gt_2⟩

end log2_sqrt_gt_2_necessary_not_sufficient_l9_9327


namespace number_of_arrangements_l9_9232

theorem number_of_arrangements : 
  ∃ (a b : ℕ), a = 4 ∧ b = 2 ∧ 
  (choose 4 2) * (choose 2 1) * (3 * 2 * 1) = 72 := 
by {
  existsi 4,
  existsi 2,
  simp,
  sorry
}

end number_of_arrangements_l9_9232


namespace median_first_fifteen_positive_integers_l9_9586

theorem median_first_fifteen_positive_integers : 
  let l := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  in list.nth l 7 = some 8 :=
by {
  let l := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  show list.nth l 7 = some 8,
  sorry
}

end median_first_fifteen_positive_integers_l9_9586


namespace quadrilateral_is_parallelogram_l9_9550

noncomputable def Point := (ℝ, ℝ) -- Defining point in 2-dimensional plane
structure Quadrilateral :=
  (A B C D O : Point)
  (convex : True)
  (intersects : ∃ t u ∈ (0:ℝ, 1:ℝ), 
                  A.1 * (1 - t) + C.1 * t = B.1 * (1 - u) + D.1 * u ∧
                  A.2 * (1 - t) + C.2 * t = B.2 * (1 - u) + D.2 * u 
               ∧ O.1 * (1 - t) + B.1 * t = D.1 * (1 - u) + A.1 * u
               ∧ O.2 * (1 - t) + B.2 * t = D.2 * (1 - u) + A.2 * u )
  (area_eq1 : (1/2) * abs ((A.1 - O.1) * (B.2 - O.2) - (B.1 - O.1) * (A.2 - O.2)) = 
                (1/2) * abs ((C.1 - O.1) * (D.2 - O.2) - (D.1 - O.1) * (C.2 - O.2))) 
  (area_eq2 : (1/2) * abs ((A.1 - O.1) * (D.2 - O.2) - (D.1 - O.1) * (A.2 - O.2)) = 
                (1/2) * abs ((B.1 - O.1) * (C.2 - O.2) - (C.1 - O.1) * (B.2 - O.2)))

def is_parallelogram (q : Quadrilateral) : Prop :=
  (q.A.1 + q.C.1) / 2 = q.O.1 ∧ (q.A.2 + q.C.2) / 2 = q.O.2 ∧
  (q.B.1 + q.D.1) / 2 = q.O.1 ∧ (q.B.2 + q.D.2) / 2 = q.O.2

theorem quadrilateral_is_parallelogram (q : Quadrilateral) : is_parallelogram q :=
sorry

end quadrilateral_is_parallelogram_l9_9550


namespace exists_sum_of_digits_div_11_l9_9074

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_sum_of_digits_div_11 (H : Finset ℕ) (h₁ : H.card = 39) :
  ∃ (a : ℕ) (h : a ∈ H), sum_of_digits a % 11 = 0 :=
by
  sorry

end exists_sum_of_digits_div_11_l9_9074


namespace g_monotonic_increasing_l9_9140

theorem g_monotonic_increasing (x : ℝ) (h : 0 < x ∧ x < π / 4) : 
  let f := λ x : ℝ, sin (2 * x) + (sqrt 3) * cos (2 * x)
  let g := λ x : ℝ, f (x - π / 6)
  g x = 2 * sin (2 * x) := by 
{ intuition, sorry }

end g_monotonic_increasing_l9_9140


namespace jello_cost_l9_9468

def cost_to_fill_tub_with_jello (water_volume_cubic_feet : ℕ) (gallons_per_cubic_foot : ℕ) 
    (pounds_per_gallon : ℕ) (tablespoons_per_pound : ℕ) (cost_per_tablespoon : ℕ) : ℕ :=
  water_volume_cubic_feet * gallons_per_cubic_foot * pounds_per_gallon * tablespoons_per_pound * cost_per_tablespoon

theorem jello_cost (water_volume_cubic_feet : ℕ) (gallons_per_cubic_foot : ℕ) 
    (pounds_per_gallon : ℕ) (tablespoons_per_pound : ℕ) (cost_per_tablespoon : ℕ) : 
    water_volume_cubic_feet = 6 ∧ gallons_per_cubic_foot = 7 ∧ pounds_per_gallon = 8 ∧ 
    tablespoons_per_pound = 1 ∧ cost_per_tablespoon = 1 →
    cost_to_fill_tub_with_jello water_volume_cubic_feet gallons_per_cubic_foot pounds_per_gallon tablespoons_per_pound cost_per_tablespoon = 270 :=
  by 
    sorry

end jello_cost_l9_9468


namespace smallest_union_card_l9_9915

-- Defining the sets X and Y and their cardinalities
variables (X Y : Type) [fintype X] [fintype Y]
variables (hX : fintype.card X = 30) (hY : fintype.card Y = 25)
variables (h_inter : ∃ Z : Type, fintype.card Z ≥ 10 ∧ Z ⊆ X ∧ Z ⊆ Y)

-- The theorem stating the smallest possible number of elements in X ∪ Y
theorem smallest_union_card : 
  ∃ k : ℕ, k = 45 ∧ k = fintype.card (X ∪ Y) :=
sorry

end smallest_union_card_l9_9915


namespace sum_reverse_base7_eq_58_l9_9737

-- Definitions for the digit reversal and base representations
def reverse_digits_base (n : ℕ) (b : ℕ) : ℕ :=
  let digits := Nat.digits b n
  Nat.ofDigits b digits.reverse

-- The theorem statement
theorem sum_reverse_base7_eq_58 :
  (∑ n in Finset.filter (λ (n : ℕ), reverse_digits_base n 7 = n ∧ reverse_digits_base n 16 = n) (Finset.range 100), n) = 58 :=
sorry

end sum_reverse_base7_eq_58_l9_9737


namespace ac_bd_sum_l9_9429

theorem ac_bd_sum (a b c d : ℝ) (h1 : a + b + c = 6) (h2 : a + b + d = -3) (h3 : a + c + d = 0) (h4 : b + c + d = -9) : 
  a * c + b * d = 23 := 
sorry

end ac_bd_sum_l9_9429


namespace intersection_reciprocal_sum_l9_9831

open Real

theorem intersection_reciprocal_sum :
    ∀ (a b : ℝ),
    (∃ x : ℝ, x - 1 = a ∧ 3 / x = b) ∧
    (a * b = 3) →
    ∃ s : ℝ, (s = (a + b) / 3 ∨ s = -(a + b) / 3) ∧ (1 / a + 1 / b = s) := by
  sorry

end intersection_reciprocal_sum_l9_9831


namespace correct_speed_l9_9926

def distance_40_late (d : ℝ) (t : ℝ) : Prop :=
  d = 40 * (t + 1/20)

def distance_60_early (d : ℝ) (t : ℝ) : Prop :=
  d = 60 * (t - 1/20)

theorem correct_speed (d t : ℝ) :
  distance_40_late d t →
  distance_60_early d t →
  (d = 12 ∧ t = 1/4) →
  (48 = d / t) :=
by {
  intros h1 h2 h3,
  sorry
}

end correct_speed_l9_9926


namespace number_of_cases_in_top_level_l9_9060

-- Definitions for the total number of soda cases
def pyramid_cases (n : ℕ) : ℕ :=
  n^2 + (n + 1)^2 + (n + 2)^2 + (n + 3)^2

-- Theorem statement: proving the number of cases in the top level
theorem number_of_cases_in_top_level (n : ℕ) (h : pyramid_cases n = 30) : n = 1 :=
by {
  sorry
}

end number_of_cases_in_top_level_l9_9060


namespace complex_power_eq_rectangular_l9_9701

noncomputable def cos_30 := Real.cos (Real.pi / 6)
noncomputable def sin_30 := Real.sin (Real.pi / 6)

theorem complex_power_eq_rectangular :
  (3 * (cos_30 + (complex.I * sin_30)))^4 = -40.5 + 40.5 * complex.I * Real.sqrt 3 := 
sorry

end complex_power_eq_rectangular_l9_9701


namespace tan_sum_pi_over_4_x_l9_9749

theorem tan_sum_pi_over_4_x (x : ℝ) (h1 : x > -π/2 ∧ x < 0) (h2 : Real.cos x = 4/5) :
  Real.tan (π/4 + x) = 1/7 :=
by
  sorry

end tan_sum_pi_over_4_x_l9_9749


namespace sum_interior_angles_l9_9007

theorem sum_interior_angles (h : ∀ (n : ℕ), (40 : ℝ) * n = 360) :
  ∑ i in finset.range (9 - 2), (180 : ℝ) = 1260 :=
by
  sorry

end sum_interior_angles_l9_9007


namespace smallest_x_plus_y_l9_9378

theorem smallest_x_plus_y (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_neq : x ≠ y) (h_eq : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 15) : x + y = 64 :=
sorry

end smallest_x_plus_y_l9_9378


namespace complex_power_rectangular_form_l9_9678

noncomputable def cos (θ : ℝ) : ℝ := sorry
noncomputable def sin (θ : ℝ) : ℝ := sorry

theorem complex_power_rectangular_form :
  (3 * complex.exp (complex.I * 30))^4 = -40.5 + 40.5 * complex.I * √3 :=
by
  sorry

end complex_power_rectangular_form_l9_9678


namespace original_cost_price_l9_9635

-- Definitions of the conditions
variable (P : ℝ) -- Original cost price for A
variable (A_to_B_profit B_to_C_profit D_to_E_profit C_to_D_discount : ℝ)
variable (E_pays : ℝ)

-- Assign concrete values given in the problem
def a_to_b_profit := 0.20
def b_to_c_profit := 0.25
def c_to_d_discount := 0.15
def d_to_e_profit := 0.30
def e_paying_amount := 289.1

-- Construct the equation based on given conditions
def final_price := 
  P * (1 + A_to_B_profit) * (1 + B_to_C_profit) *
  (1 - C_to_D_discount) * (1 + D_to_E_profit)

-- Prove that the solution satisfies the equation
theorem original_cost_price : 
  final_price P a_to_b_profit b_to_c_profit c_to_d_discount d_to_e_profit = e_paying_amount → 
  P = 142.8 :=
by
  sorry

end original_cost_price_l9_9635


namespace sum_of_four_l9_9848

variable {α : Type*} [LinearOrderedField α]

def geometric_sequence (a₁ q : α) : ℕ → α
| 0 := a₁
| (n+1) := geometric_sequence a₁ q n * q

def sum_geometric_sequence (a₁ q : α) (n : ℕ) : α :=
if q = 1 then a₁ * (n + 1) else a₁ * (1 - q^(n + 1)) / (1 - q)

theorem sum_of_four {a₁ q S₅ S₃ S₄ : α} (h₁ : a₁ = 1) (h₂ : sum_geometric_sequence a₁ q 4 = S₄) (h₃ : sum_geometric_sequence a₁ q 5 = S₅) (h₄ : sum_geometric_sequence a₁ q 3 = S₃) : S₅ = 5 * S₃ - 4 → S₄ = 15 :=
by
  sorry

end sum_of_four_l9_9848


namespace distinct_painting_ways_l9_9326

theorem distinct_painting_ways (n : ℕ) (h : n = 9) : 
  ∃ (ways : ℕ), ways = 72 := 
by 
  use (∑ i in finset.range 9, ∑ j in finset.range 9, if i ≠ j then 1 else 0)
  apply_eq
  sorry

end distinct_painting_ways_l9_9326


namespace four_ast_three_eq_64_l9_9740

variable {a b c : ℕ}
variable (ast : ℕ → ℕ → ℕ)

-- Conditions
axiom op_assoc : ∀ a b c, (ast a b) ∗ c = ast a (b * c)
axiom op_add : ∀ a b c, (ast a b) * (ast a c) = ast a (b + c)

-- Theorem to prove
theorem four_ast_three_eq_64 : ast 4 3 = 64 :=
sorry

end four_ast_three_eq_64_l9_9740


namespace at_least_five_pairs_l9_9345

open Set

-- Define a unit circle
def unit_circle (p : ℝ × ℝ) : Prop :=
  p.1^2 + p.2^2 = 1

-- Define the ten points inside the circle
variables (A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 : ℝ × ℝ)

-- Define the condition that points are inside the unit circle
def points_inside_unit_circle : Prop :=
  ∀ p ∈ {A1, A2, A3, A4, A5, A6, A7, A8, A9, A10}, p.1^2 + p.2^2 ≤ 1

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2))

-- Define the main theorem
theorem at_least_five_pairs : points_inside_unit_circle A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 →
  ∃ (pairs : List ((ℝ × ℝ) × (ℝ × ℝ))), 
  pairs.length ≥ 5 ∧ ∀ p ∈ pairs, distance p.1 p.2 ≤ 1 := 
sorry

end at_least_five_pairs_l9_9345


namespace A_wins_when_n_is_9_l9_9133

-- Definition of the game conditions and the strategy
def game (n : ℕ) (A_first : Bool) :=
  ∃ strategy : ℕ → ℕ,
    ∀ taken balls_left : ℕ,
      balls_left - taken > 0 →
      taken ≥ 1 → taken ≤ 3 →
      if A_first then
        (balls_left - taken = 0 → strategy (balls_left - taken) = 1) ∧
        (∀ t : ℕ, t >= 1 ∧ t ≤ 3 → strategy t = balls_left - taken - t)
      else
        (balls_left - taken = 0 → strategy (balls_left - taken) = 0) ∨
        (∀ t : ℕ, t >= 1 ∧ t ≤ 3 → strategy t = balls_left - taken - t)

-- Prove that for n = 9 A has a winning strategy
theorem A_wins_when_n_is_9 : game 9 true :=
sorry

end A_wins_when_n_is_9_l9_9133


namespace third_even_number_in_sequence_l9_9129

def sequence : List ℕ := [2, 4, 6, 8, 10, 12, 14]

theorem third_even_number_in_sequence :
  (sequence.nth 2).get_or_else 0 = 6 :=
by sorry

end third_even_number_in_sequence_l9_9129


namespace exists_subset_sum_bound_l9_9938

theorem exists_subset_sum_bound (X : Finset ℝ) (n : ℕ) (h : X.card = n) (h_pos : 0 < n) :
  ∃ (S : Finset ℝ) (m : ℤ), S ⊆ X ∧ S.nonempty ∧ abs (m + ∑ s in S, id s) ≤ 1 / (n + 1) :=
sorry

end exists_subset_sum_bound_l9_9938


namespace triangle_inequality_theorem_l9_9969

theorem triangle_inequality_theorem (a b c : ℕ) : a + b > c ∧ a + c > b ∧ b + c > a → True := 
by sorry

example : (3, 4, 6) can form a triangle while (3, 4, 7), (5, 7, 12), and (2, 3, 6) cannot :=
by
  have hA := ¬(triangle_inequality_theorem 3 4 7)
  have hB := (triangle_inequality_theorem 3 4 6)
  have hC := ¬(triangle_inequality_theorem 5 7 12)
  have hD := ¬(triangle_inequality_theorem 2 3 6)
  exact ⟨hB, hA, hC, hD⟩

end triangle_inequality_theorem_l9_9969


namespace circle_sine_intersection_2_points_l9_9226

theorem circle_sine_intersection_2_points :
  let circle_eq : ℝ → ℝ → Prop := λ x y, x ^ 2 + y ^ 2 = 4
  let sine_eq : ℝ → ℝ := λ x, Real.sin x
  let intersections := {p : ℝ × ℝ | p.1 ∈ Icc (-2 * Real.pi) (2 * Real.pi) ∧ circle_eq p.1 p.2 ∧ p.2 = sine_eq p.1}
  |intersections| = 2 :=
sorry

end circle_sine_intersection_2_points_l9_9226


namespace complement_of_larger_angle_equals_70_l9_9986

theorem complement_of_larger_angle_equals_70 :
  ∀ (x : ℝ), (90 - x) / x = 2 / 7 → 70 = 90 - x := 
by
  intro x
  assume h
  sorry

end complement_of_larger_angle_equals_70_l9_9986


namespace max_CP_l9_9069

noncomputable def equilateral_triangle (A B C : Type) [metric_space A] [metric_space B] [metric_space C] :=
distance A B = distance B C ∧ distance B C = distance C A

theorem max_CP 
  {A B C P : Type} 
  [metric_space A] [metric_space B] [metric_space C] [metric_space P] 
  (h_eq : equilateral_triangle A B C) 
  (hAP : distance A P = 2) 
  (hBP : distance B P = 3) : 
  ∃ max_CP : ℝ, max_CP = 5 ∧ ∀ CP, distance C P ≤ max_CP := 
begin
  sorry
end

end max_CP_l9_9069


namespace isosceles_triangle_at_intersection_l9_9488

-- Define the geometric setup and the proof problem
theorem isosceles_triangle_at_intersection
  (A B C : Point)
  (L1 L2 : Line)
  (Γ : Circle)
  (h1 : ∃ (P : Point), P ∈ L1 ∧ P ∈ L2 ∧ P = A)
  (h2 : tangent Γ L1 B)
  (h3 : tangent Γ L2 C) :
  triangle.is_isosceles_at A :=
sorry

end isosceles_triangle_at_intersection_l9_9488


namespace possible_values_of_x3_y3_z3_l9_9481

variable {x y z : ℂ}
def N : Matrix (Fin 3) (Fin 3) ℂ := ![![x, y, z], ![y, z, x], ![z, x, y]]
def I : Matrix (Fin 3) (Fin 3) ℂ := ![![1, 0, 0], ![0, 1, 0], ![0, 0, 1]]

theorem possible_values_of_x3_y3_z3 (hN : N * N = I) (hxyz : x * y * z = 2) :
  x^3 + y^3 + z^3 = 5 ∨ x^3 + y^3 + z^3 = 7 := 
sorry

end possible_values_of_x3_y3_z3_l9_9481


namespace zero_point_interval_l9_9967

noncomputable def f (x : ℝ) := Real.exp x + x - 2

theorem zero_point_interval : ∃ c ∈ Ioo (0 : ℝ) (1 : ℝ), f c = 0 := by
  have h0 : f 0 < 0 := by
    calc
      f 0 = Real.exp 0 + 0 - 2 := rfl
      ... = 1 - 2 := by norm_num
      ... = -1 := by norm_num

  have h1 : f 1 > 0 := by
    calc
      f 1 = Real.exp 1 + 1 - 2 := rfl
      ... = Real.exp 1 - 1 := by ring
      ... > 0 := by
        have : Real.exp 1 > 1 := Real.exp_pos 1
        linarith

  have h_cont : Continuous (f) := by continuity

  exact IntermediateValueTheorem (Continuous.continuous_on h_cont) (Set.mem_Icc.2 ⟨le_of_lt h0, h1⟩) 0

end zero_point_interval_l9_9967


namespace sum_of_ages_is_32_l9_9153

-- Define the values and conditions given in the problem
def viggo_age_when_brother_was_2 (brother_age : ℕ) : ℕ := 10 + 2 * brother_age
def age_difference (viggo_age_brother_2 : ℕ) (brother_age : ℕ) : ℕ := viggo_age_brother_2 - brother_age
def current_viggo_age (current_brother_age : ℕ) (difference : ℕ) := current_brother_age + difference

-- State the main theorem
theorem sum_of_ages_is_32 : 
  let brother_age_when_2 := 2 in
  let current_brother_age := 10 in
  let viggo_age_when_2 := viggo_age_when_brother_was_2 brother_age_when_2 in
  let difference := age_difference viggo_age_when_2 brother_age_when_2 in
  current_viggo_age current_brother_age difference + current_brother_age = 32 := 
by
  sorry

end sum_of_ages_is_32_l9_9153


namespace sum_of_four_l9_9845

variable {α : Type*} [LinearOrderedField α]

def geometric_sequence (a₁ q : α) : ℕ → α
| 0 := a₁
| (n+1) := geometric_sequence a₁ q n * q

def sum_geometric_sequence (a₁ q : α) (n : ℕ) : α :=
if q = 1 then a₁ * (n + 1) else a₁ * (1 - q^(n + 1)) / (1 - q)

theorem sum_of_four {a₁ q S₅ S₃ S₄ : α} (h₁ : a₁ = 1) (h₂ : sum_geometric_sequence a₁ q 4 = S₄) (h₃ : sum_geometric_sequence a₁ q 5 = S₅) (h₄ : sum_geometric_sequence a₁ q 3 = S₃) : S₅ = 5 * S₃ - 4 → S₄ = 15 :=
by
  sorry

end sum_of_four_l9_9845


namespace correct_conclusions_count_l9_9256

theorem correct_conclusions_count :
  (∀ x : ℝ, (tan x = sqrt 3 → ∃ k : ℤ, x = (↑k * real.pi + real.pi / 3)) → ¬(tan x = sqrt 3 ↔ x = real.pi / 3)) →
  (∀ x : ℝ, (x - sin x = 0 → x = 0) → (x - sin x ≠ 0 → x ≠ 0)) →
  (∀ (a b : euclidean_space ℝ (fin 3)), 
    (abs (inner a b) = norm a * norm b → a ≠ 0 → b ≠ 0 → a = (norm a / norm b : ℝ ) • b ∨ b = (norm b / norm a : ℝ) • a)) →
  1 := sorry

end correct_conclusions_count_l9_9256


namespace squirrels_acorns_l9_9219

theorem squirrels_acorns (total_acorns : ℕ) (num_squirrels : ℕ) (required_per_squirrel : ℕ) 
  (h1 : total_acorns = 575) (h2 : num_squirrels = 5) (h3 : required_per_squirrel = 130) :
  let acorns_per_squirrel := total_acorns / num_squirrels in
  required_per_squirrel - acorns_per_squirrel = 15 :=
by
  sorry

end squirrels_acorns_l9_9219


namespace find_BC_length_l9_9067

noncomputable def BC_length (O A M B C : Point) (r : ℝ) (α : ℝ) (h_circle : circle O r) (h_AO : collinear O A M)
(h_AMB : angle AM B = α) (h_OMC : angle O M C = α) (h_sin_alpha : sin α = sqrt 39 / 8) : ℝ :=
let OB := dist O B,
    OC := dist O C,
    h_r : OB = r := sorry,
    h_r' : OC = r := sorry,
    cos_α := sqrt (1 - (sqrt 39 / 8)^2),
    cos_2α := 2 * (cos_α)^2 - 1,
    cos_180_2α := - cos_2α
in sqrt (r^2 + r^2 - 2 * r * r * cos_180_2α)

theorem find_BC_length (O A M B C : Point) (r : ℝ) (α : ℝ) (h_circle : circle O r) (h_AO : collinear O A M)
(h_AMB : angle AM B = α) (h_OMC : angle O M C = α) (h_sin_alpha : sin α = sqrt 39 / 8)
: BC_length O A M B C r α h_circle h_AO h_AMB h_OMC h_sin_alpha = 20 :=
sorry

end find_BC_length_l9_9067


namespace axis_of_symmetry_g_l9_9791

axiom f : ℝ → ℝ
axiom g : ℝ → ℝ
axiom λ : ℝ
axiom h_symm_f : ∀ x, f (x) = f (-x - π/2)
axiom h_expand_shift : g = λ * (x - π/3)/2

theorem axis_of_symmetry_g (k : ℤ): g (2 * k * π + 11 * π / 6) = g (11 * π / 6) :=
by
  sorry

end axis_of_symmetry_g_l9_9791


namespace math_problem_equivalent_proof_l9_9784

noncomputable def α : ℝ := sorry -- We assume α exists.

def point_P : ℝ × ℝ := (-3, real.sqrt 3)

-- Conditions
def vertex_at_origin (α : ℝ) : Prop := ∃ x = 0 ∧ ∃ y = 0, true
def initial_side_positive_x_axis (α : ℝ) : Prop := ∃ x > 0, true
def terminal_side_through_P (α : ℝ) : Prop := ∃ x y : ℝ, ((x, y) = point_P)

-- Questions
def tan_α (α : ℝ) : ℝ := - real.sqrt 3 / 3

def determinant (α : ℝ) : ℝ := sin α * cos α - tan_α α

def function_f (α x : ℝ) : ℝ :=
  det ![(cos (x + α), -sin α), (sin (x + α), cos α)]

def y (α x : ℝ) : ℝ :=
  real.sqrt 3 * (function_f α (π / 2 - 2 * x)) + 2 * (function_f α x)^2

-- Lean theorem statement
theorem math_problem_equivalent_proof :
  vertex_at_origin α ∧ initial_side_positive_x_axis α ∧ terminal_side_through_P α →
  (tan_α α = - real.sqrt 3 / 3) ∧
  (determinant α = real.sqrt 3 / 12) ∧
  (∃ x, y α x = 3) :=
by {
    -- Proof omitted
    sorry
}

end math_problem_equivalent_proof_l9_9784


namespace vertical_asymptote_conditions_l9_9743

theorem vertical_asymptote_conditions (c : ℝ) :
  (∃ f : ℝ → ℝ, f = λ x, (x^2 - 2*x + c) / (x^2 - x - 6) ∧ 
   ((c = -3 ∧ (∀ x, x ≠ 3 → f(x) ≠ ∞)) ∨ (c = -8 ∧ (∀ x, x ≠ -2 → f(x) ≠ ∞)))) :=
begin
  sorry
end

end vertical_asymptote_conditions_l9_9743


namespace count_cells_with_at_least_three_rectangles_l9_9727

theorem count_cells_with_at_least_three_rectangles :
  let is_endpoint (i j : ℕ) : Prop :=
    (16 ≤ i ∧ i ≤ 50) ∨ (1 ≤ i ∧ i ≤ 35) ∨ (16 ≤ j ∧ j ≤ 50) ∨ (1 ≤ j ∧ j ≤ 35)
  let count_cells (cond : ℕ → ℕ → Prop) : ℕ :=
    ((finset.range 50).product (finset.range 50)).count (λ ⟨i, j⟩, cond i.succ j.succ)
  count_cells (λ i j, (16 ≤ i ∧ i ≤ 50) ∨ (1 ≤ i ∧ i ≤ 35) ∨ (16 ≤ j ∧ j ≤ 50) ∨ (1 ≤ j ∧ j ≤ 35)) ≥ 1600 := 
sorry

end count_cells_with_at_least_three_rectangles_l9_9727


namespace total_kids_played_tag_with_l9_9478

theorem total_kids_played_tag_with : 
  let kids_mon : Nat := 12
  let kids_tues : Nat := 7
  let kids_wed : Nat := 15
  let kids_thurs : Nat := 10
  let kids_fri : Nat := 18
  (kids_mon + kids_tues + kids_wed + kids_thurs + kids_fri) = 62 := by
  sorry

end total_kids_played_tag_with_l9_9478


namespace number_of_regions_divided_l9_9891

def regions_divided_by_planes (n : ℕ) : ℕ :=
  (n^3 + 5 * n + 6) / 6

theorem number_of_regions_divided (n : ℕ) : regions_divided_by_planes n = (n^3 + 5 * n + 6) / 6 :=
by regression
  sorry

end number_of_regions_divided_l9_9891


namespace complex_is_1_sub_sqrt3i_l9_9615

open Complex

theorem complex_is_1_sub_sqrt3i (z : ℂ) (h : z * (1 + Real.sqrt 3 * I) = abs (1 + Real.sqrt 3 * I)) : z = 1 - Real.sqrt 3 * I :=
sorry

end complex_is_1_sub_sqrt3i_l9_9615


namespace sequence_nonzero_l9_9336

noncomputable def sequence (a1 a2 : ℕ) : ℕ → ℕ
| 0     := a1
| 1     := a2
| (n+2) := if (sequence n * sequence (n+1)) % 2 = 0 then
              5 * sequence (n+1) - 3 * sequence n
           else
              sequence (n+1) - sequence n

theorem sequence_nonzero (n : ℕ) : sequence 1 2 n ≠ 0 := 
by 
sory

end sequence_nonzero_l9_9336


namespace digit_7_count_correct_l9_9558

def base8ToBase10 (n : Nat) : Nat :=
  -- converting base 8 number 1000 to base 10
  1 * 8^3 + 0 * 8^2 + 0 * 8^1 + 0 * 8^0

def countDigit7 (n : Nat) : Nat :=
  -- counts the number of times the digit '7' appears in numbers from 1 to n
  let digits := (List.range (n + 1)).map fun x => x.digits 10
  digits.foldl (fun acc ds => acc + ds.count 7) 0

theorem digit_7_count_correct : countDigit7 512 = 123 := by
  sorry

end digit_7_count_correct_l9_9558


namespace trapezoid_ABCD_BCE_area_l9_9450

noncomputable def triangle_area (a b c : ℝ) (angle_abc : ℝ) : ℝ :=
  1 / 2 * a * b * Real.sin angle_abc

noncomputable def area_of_triangle_BCE (AB DC AD : ℝ) (angle_DAB : ℝ) (area_triangle_DCB : ℝ) : ℝ :=
  let ratio := AB / DC
  (ratio / (1 + ratio)) * area_triangle_DCB

theorem trapezoid_ABCD_BCE_area :
  ∀ (AB DC AD : ℝ) (angle_DAB : ℝ) (area_triangle_DCB : ℝ),
    AB = 30 →
    DC = 24 →
    AD = 3 →
    angle_DAB = Real.pi / 3 →
    area_triangle_DCB = 18 * Real.sqrt 3 →
    area_of_triangle_BCE AB DC AD angle_DAB area_triangle_DCB = 10 * Real.sqrt 3 := 
by
  intros
  sorry

end trapezoid_ABCD_BCE_area_l9_9450


namespace arithmetic_sequence_a11_value_l9_9779

-- We start by defining the conditions:
variables {a : ℕ → ℝ}  -- a sequence of real numbers

-- Define the conditions
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def a2_eq_3 := a 2 = 3
def a6_eq_7 := a 6 = 7

-- The question: Prove that a_11 = 12
theorem arithmetic_sequence_a11_value (h_arith : is_arithmetic_sequence a) (h_a2 : a2_eq_3) (h_a6 : a6_eq_7) : a 11 = 12 :=
sorry

end arithmetic_sequence_a11_value_l9_9779


namespace jello_cost_l9_9469

def cost_to_fill_tub_with_jello (water_volume_cubic_feet : ℕ) (gallons_per_cubic_foot : ℕ) 
    (pounds_per_gallon : ℕ) (tablespoons_per_pound : ℕ) (cost_per_tablespoon : ℕ) : ℕ :=
  water_volume_cubic_feet * gallons_per_cubic_foot * pounds_per_gallon * tablespoons_per_pound * cost_per_tablespoon

theorem jello_cost (water_volume_cubic_feet : ℕ) (gallons_per_cubic_foot : ℕ) 
    (pounds_per_gallon : ℕ) (tablespoons_per_pound : ℕ) (cost_per_tablespoon : ℕ) : 
    water_volume_cubic_feet = 6 ∧ gallons_per_cubic_foot = 7 ∧ pounds_per_gallon = 8 ∧ 
    tablespoons_per_pound = 1 ∧ cost_per_tablespoon = 1 →
    cost_to_fill_tub_with_jello water_volume_cubic_feet gallons_per_cubic_foot pounds_per_gallon tablespoons_per_pound cost_per_tablespoon = 270 :=
  by 
    sorry

end jello_cost_l9_9469


namespace coefficients_sum_l9_9390

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) ^ 4

theorem coefficients_sum : 
  ∃ (a₀ a₁ a₂ a₃ a₄ : ℝ), 
  ((2 * x - 1) ^ 4 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4) 
  ∧ (a₁ + 2 * a₂ + 3 * a₃ + 4 * a₄ = 8) :=
sorry

end coefficients_sum_l9_9390


namespace arun_crosses_train_b_in_12_seconds_l9_9598

noncomputable def time_to_cross_train_b
  (length_a length_b : ℕ)
  (speed_a_kmh speed_b_kmh : ℕ) : ℕ :=
let speed_a_ms := speed_a_kmh * 1000 / 3600 in
let speed_b_ms := speed_b_kmh * 1000 / 3600 in
let total_length := length_a + length_b in
let relative_speed := speed_a_ms + speed_b_ms in
total_length / relative_speed

theorem arun_crosses_train_b_in_12_seconds :
  time_to_cross_train_b 150 150 54 36 = 12 :=
sorry

end arun_crosses_train_b_in_12_seconds_l9_9598


namespace smallest_prime_after_six_nonprimes_l9_9187

open Nat

theorem smallest_prime_after_six_nonprimes : 
  ∃ p : ℕ, prime p ∧ p > 0 ∧
  (∃ n : ℕ, ¬prime n ∧ ¬prime (n+1) ∧ ¬prime (n+2) ∧ ¬prime (n+3) ∧ ¬prime (n+4) ∧ ¬prime (n+5) ∧ prime (n+6) ∧ n+6 = p) ∧ 
  p = 89 :=
sorry

end smallest_prime_after_six_nonprimes_l9_9187


namespace remainder_of_3056_mod_32_l9_9587

theorem remainder_of_3056_mod_32 : 3056 % 32 = 16 := by
  sorry

end remainder_of_3056_mod_32_l9_9587


namespace ellipse_equation_l9_9760

theorem ellipse_equation (a b c : ℝ) 
  (h1 : 0 < b) (h2 : b < a) 
  (h3 : c = 3 * Real.sqrt 3) 
  (h4 : a = 6) 
  (h5 : b^2 = a^2 - c^2) :
  (∀ x y : ℝ, x^2 / 36 + y^2 / 9 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) :=
by
  sorry

end ellipse_equation_l9_9760


namespace range_of_m_l9_9834

noncomputable def f (x : ℝ) := x * Real.exp x
noncomputable def g (x : ℝ) := (-x^2 + x + 1) * Real.exp x

theorem range_of_m :
  ∀ (m : ℝ),
  (∃ (x_0 x_1 x_2 : ℝ), x_0 ≠ x_1 ∧ x_1 ≠ x_2 ∧ x_0 ≠ x_2 ∧
    g x_0 = m ∧ g x_1 = m ∧ g x_2 = m) ↔
  m ∈ Ioo (-5 / Real.exp 2) 0 :=
begin
  sorry
end

end range_of_m_l9_9834


namespace smallest_prime_after_six_nonprimes_l9_9157

open Nat

/-- 
  Proposition:
    97 is the smallest prime number that occurs after a sequence 
    of six consecutive positive integers all of which are nonprime.
-/
theorem smallest_prime_after_six_nonprimes : ∃ (n : ℕ), (n > 0 ∧ Prime (n + 97)) ∧ 
  (∀ k : ℕ, k < n → (Nonprime k) ∧ Nonprime (k+1) ∧ Nonprime (k+2) ∧ Nonprime (k+3) ∧ Nonprime (k+4) ∧ Nonprime (k+5)) ∧ 
  (Prime (n + 1) → n + 1 = 97) := 
by 
  sorry

end smallest_prime_after_six_nonprimes_l9_9157


namespace mn_minus_n_values_l9_9431

theorem mn_minus_n_values (m n : ℝ) (h1 : |m| = 4) (h2 : |n| = 2.5) (h3 : m * n < 0) :
  m * n - n = -7.5 ∨ m * n - n = -12.5 :=
sorry

end mn_minus_n_values_l9_9431


namespace period_of_altered_sine_function_l9_9716

-- Define the standard period of the sine function
def standard_period := 2 * Real.pi

-- Define the given function
def altered_sine_function (x : ℝ) := Real.sin (x / 3)

-- Define the period to prove
def new_period := 6 * Real.pi

-- The theorem statement to prove
theorem period_of_altered_sine_function : 
  ∀ x, altered_sine_function (x + new_period) = altered_sine_function x :=
by 
  sorry

end period_of_altered_sine_function_l9_9716


namespace Point_in_Second_Quadrant_l9_9873

-- Define the quadrants in a custom type
inductive Quadrant
| First
| Second
| Third
| Fourth

-- Define the point coordinates
structure Point where
  x : ℝ
  y : ℝ

-- The point P with given coordinates (-2, 3)
def P : Point := { x := -2, y := 3 }

-- The theorem stating that the point P lies in the second quadrant
theorem Point_in_Second_Quadrant (p : Point) (h1 : p.x = -2) (h2 : p.y = 3) : 
  (p.x < 0) ∧ (p.y > 0) → 
  Quadrant.Second :=
by
  sorry

end Point_in_Second_Quadrant_l9_9873


namespace points_in_circle_l9_9948

theorem points_in_circle (points : Fin 15 → EuclideanSpace 𝕜 2) (circle_center : EuclideanSpace 𝕜 2) 
  (h_points_in_circle : ∀ i, dist (points i) circle_center ≤ 2) : 
  ∃ (small_circle_center : EuclideanSpace 𝕜 2), 
    (∀ i, dist (small_circle_center) circle_center ≤ 1) ∧ 
    (∃ i₁ i₂ i₃, i₁ ≠ i₂ ∧ i₂ ≠ i₃ ∧ i₃ ≠ i₁ ∧ 
      dist (points i₁) small_circle_center ≤ 1 ∧ 
      dist (points i₂) small_circle_center ≤ 1 ∧
      dist (points i₃) small_circle_center ≤ 1) := 
sorry

end points_in_circle_l9_9948


namespace pentagon_area_correct_l9_9448

structure Point where
  x : ℝ
  y : ℝ

def pentagon_area : ℝ :=
  let p1 : Point := ⟨-7, 1⟩
  let p2 : Point := ⟨1, 1⟩
  let p3 : Point := ⟨1, -6⟩
  let p4 : Point := ⟨-7, -6⟩
  let p5 : Point := ⟨-1, Real.sqrt 3⟩

  let rectangle_area := 8 * 7
  let triangle_area := (7 * 6) / 2
  rectangle_area + triangle_area

theorem pentagon_area_correct : pentagon_area = 77 := 
by 
  sorry

end pentagon_area_correct_l9_9448


namespace second_even_number_is_82_l9_9127

theorem second_even_number_is_82 :
  ∃ n : ℕ, even n ∧ (n + (n+2) + (n+4) = 246 ∧ n+2 = 82) :=
begin
  use 80,
  split,
  { apply even_add.mpr,
    split; apply even_bit0 },
  split,
  { calculate_sum,
    sorry },
  { refl }
end

end second_even_number_is_82_l9_9127


namespace period_of_altered_sine_function_l9_9717

-- Define the standard period of the sine function
def standard_period := 2 * Real.pi

-- Define the given function
def altered_sine_function (x : ℝ) := Real.sin (x / 3)

-- Define the period to prove
def new_period := 6 * Real.pi

-- The theorem statement to prove
theorem period_of_altered_sine_function : 
  ∀ x, altered_sine_function (x + new_period) = altered_sine_function x :=
by 
  sorry

end period_of_altered_sine_function_l9_9717


namespace ordered_pairs_count_l9_9814

theorem ordered_pairs_count :
  ∃ (p : Finset (ℕ × ℕ)), (∀ (a b : ℕ), (a, b) ∈ p → a * b + 45 = 10 * Nat.lcm a b + 18 * Nat.gcd a b) ∧
  p.card = 4 :=
by
  sorry

end ordered_pairs_count_l9_9814


namespace root_in_interval_l9_9963

noncomputable def f (x : ℝ) := Real.exp x - 1 / x

theorem root_in_interval : 
  (∃ x, x ∈ set.Ioo (1 / 2 : ℝ) 1 ∧ f x = 0) :=
by
  have h1 : f (1 / 2 : ℝ) < 0 := by
    calc 
      f (1 / 2) = Real.exp (1 / 2) - 2 : by norm_num [f]
      ... < 0 : by norm_num [Real.exp_one]

  have h2 : f 1 > 0 := calc 
    f 1 = Real.exp 1 - 1 : by norm_num [f]
    ... > 0 : by norm_num [Real.exp_one]

  -- Apply Intermediate Value Theorem
  exact IntermediateValueTheorem (1 / 2 : ℝ) 1 f h1 h2

end root_in_interval_l9_9963


namespace expand_expression_l9_9731

theorem expand_expression (x : ℝ) : (x + 3) * (x^2 - x + 4) = x^3 + 2x^2 + x + 12 := 
by 
  sorry

end expand_expression_l9_9731


namespace count_special_three_digit_integers_l9_9424

theorem count_special_three_digit_integers : 
  let B := 8 * 9 * 9,
  let A := 7 * 8 * 8 
  in B - A = 200 := 
begin
  let B := 8 * 9 * 9,
  let A := 7 * 8 * 8,
  have h1 : B = 648, by norm_num,
  have h2 : A = 448, by norm_num,
  calc
    B - A = 648 - 448 : by rw [h1, h2]
    ... = 200 : by norm_num
end

end count_special_three_digit_integers_l9_9424


namespace ratio_is_five_thirds_l9_9130

noncomputable def ratio_of_numbers (a b : ℝ) : Prop :=
  (a + b = 4 * (a - b)) → (a = 2 * b) → (a / b = 5 / 3)

theorem ratio_is_five_thirds {a b : ℝ} (h1 : a + b = 4 * (a - b)) (h2 : a = 2 * b) :
  a / b = 5 / 3 :=
  sorry

end ratio_is_five_thirds_l9_9130


namespace total_path_traversed_by_P_is_12pi_l9_9708

-- Definition of the problem conditions
def equilateral_triangle (ABP : Triangle) := (ABP.side_length = 3)
def square (AXYZ : Square) := (AXYZ.side_length = 6)
def B_on_AX (B AX : Point) := B ∈ AX

-- The conditions for the movement of point P
def translation_and_rotation_condition (P : Point) : Prop :=
  ∃ ABP : Triangle, ∃ AXYZ : Square, equilateral_triangle ABP ∧ square AXYZ ∧ 
  B_on_AX ABP B AX ∧ 
  -- Description of the path traveled
  true -- Placeholder for detailed movement description

-- The theorem to prove
theorem total_path_traversed_by_P_is_12pi 
  (P : Point) (h : translation_and_rotation_condition P) : 
  (path_traveled P = 12 * π) :=
sorry

end total_path_traversed_by_P_is_12pi_l9_9708


namespace mass_when_length_30_l9_9018

-- Given constants and relationships
def x_values : List ℕ := [0, 1, 2, 3, 4, 5]
def y_values : List ℕ := [18, 20, 22, 24, 26, 28]

def relationship (x : ℕ) : ℕ := 18 + 2 * x

-- The proof statement
theorem mass_when_length_30 :
  ∃ x, relationship x = 30 :=
by
  use 6
  show relationship 6 = 30
  simp [relationship]
  sorry

end mass_when_length_30_l9_9018


namespace explicit_x_n_formula_l9_9989

theorem explicit_x_n_formula (x y : ℕ → ℕ) (n : ℕ) :
  x 0 = 2 ∧ y 0 = 1 ∧
  (∀ n, x (n + 1) = x n ^ 2 + y n ^ 2) ∧
  (∀ n, y (n + 1) = 2 * x n * y n) →
  x n = (3 ^ (2 ^ n) + 1) / 2 :=
by
  sorry

end explicit_x_n_formula_l9_9989


namespace max_captain_coins_is_59_l9_9534

noncomputable def max_coins_for_captain (a b c d e f : ℕ) : ℕ :=
if h : a + b + c + d + e + f = 180 ∧
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0 ∧
  a + b + c + d + e + f = 180 ∧
  ((b > a ∧ b > c) +
   (c > b ∧ c > d) +
   (d > c ∧ d > e) +
   (e > d ∧ e > f) +
   (f > e ∧ f > a) ≥ 3)  then a else 0

theorem max_captain_coins_is_59 :
  ∃ a b c d e f : ℕ, max_coins_for_captain a b c d e f = 59 :=
begin
  use [59, 60, 1, 60, 0, 60],
  unfold max_coins_for_captain,
  split,
  { norm_num },
  { split; norm_num },
  { split; norm_num, split; norm_num, norm_num },
end

end max_captain_coins_is_59_l9_9534


namespace range_of_x_l9_9746

noncomputable def rangeOfX (a1 a2 a3 : ℝ) (h : a1 > a2 ∧ a2 > a3 ∧ a3 > 0) : Set ℝ :=
  {x : ℝ | 0 < x ∧ x < 2 / a1}

theorem range_of_x (a1 a2 a3 : ℝ) (h : a1 > a2 ∧ a2 > a3 ∧ a3 > 0) :
  (∀ i ∈ {1, 2, 3}, (1 - ([a1, a2, a3].nth i).get_or_else 0 * x)^2 < 1) ↔ (0 < x ∧ x < 2 / a1) :=
by
  sorry

end range_of_x_l9_9746


namespace smallest_prime_after_six_nonprime_l9_9168

open Nat

noncomputable def is_nonprime (n : ℕ) : Prop :=
  ¬Prime n ∧ n ≠ 1

noncomputable def six_consecutive_nonprime (n : ℕ) : Prop :=
  is_nonprime n ∧ is_nonprime (n + 1) ∧ is_nonprime (n + 2) ∧ 
  is_nonprime (n + 3) ∧ is_nonprime (n + 4) ∧ is_nonprime (n + 5)

theorem smallest_prime_after_six_nonprime :
  ∃ n, six_consecutive_nonprime n ∧ (∀ m, m ≠ 97 → n + 6 < m → ¬Prime m) :=
by
  sorry

end smallest_prime_after_six_nonprime_l9_9168


namespace total_wheels_combined_l9_9135

-- Define the counts of vehicles and wheels per vehicle in each storage area
def bicycles_A : ℕ := 16
def tricycles_A : ℕ := 7
def unicycles_A : ℕ := 10
def four_wheelers_A : ℕ := 5

def bicycles_B : ℕ := 12
def tricycles_B : ℕ := 5
def unicycles_B : ℕ := 8
def four_wheelers_B : ℕ := 3

def wheels_bicycle : ℕ := 2
def wheels_tricycle : ℕ := 3
def wheels_unicycle : ℕ := 1
def wheels_four_wheeler : ℕ := 4

-- Calculate total wheels in Storage Area A
def total_wheels_A : ℕ :=
  bicycles_A * wheels_bicycle + tricycles_A * wheels_tricycle + unicycles_A * wheels_unicycle + four_wheelers_A * wheels_four_wheeler
  
-- Calculate total wheels in Storage Area B
def total_wheels_B : ℕ :=
  bicycles_B * wheels_bicycle + tricycles_B * wheels_tricycle + unicycles_B * wheels_unicycle + four_wheelers_B * wheels_four_wheeler

-- Theorem stating that the combined total number of wheels in both storage areas is 142
theorem total_wheels_combined : total_wheels_A + total_wheels_B = 142 := by
  sorry

end total_wheels_combined_l9_9135


namespace inverse_value_ratio_l9_9557

noncomputable def g (x : ℚ) : ℚ := (3 * x + 1) / (x - 4)

theorem inverse_value_ratio :
  (∃ (a b c d : ℚ), ∀ x, g ((a * x + b) / (c * x + d)) = x) → ∃ a c : ℚ, a / c = -4 :=
by
  sorry

end inverse_value_ratio_l9_9557


namespace average_age_of_girls_l9_9016
open Real

theorem average_age_of_girls (total_students girls boys : ℕ) 
  (avg_age_boys avg_age_school avg_age_girls : ℝ) : 
  total_students = 600 → 
  girls = 150 → 
  boys = total_students - girls → 
  avg_age_boys = 12 → 
  avg_age_school = 11.75 → 
  avg_age_school = ((boys * avg_age_boys + girls * avg_age_girls) / total_students) → 
  avg_age_girls = 11 :=
by {
  assume h1 h2 h3 h4 h5 h6,
  sorry
}

end average_age_of_girls_l9_9016


namespace smallest_sum_of_inverses_l9_9370

theorem smallest_sum_of_inverses 
  (x y : ℕ) (hx : x ≠ y) (h1 : 0 < x) (h2 : 0 < y) (h_condition : (1 / x : ℚ) + 1 / y = 1 / 15) :
  x + y = 64 := 
sorry

end smallest_sum_of_inverses_l9_9370


namespace ratio_of_second_to_first_l9_9258

def food_consumed (dog1 dog2 dog3 : ℕ ) : ℕ := dog1 + dog2 + dog3
def ratio_of_food (food1 food2 : ℕ) : ℚ := food2 / food1

theorem ratio_of_second_to_first (d1 d3 : ℕ) (avg : ℕ) (T : ℕ) (h1 : food_consumed d1 (avg * 3 - d1 - d3) d3 = T) :
  ratio_of_food d1 (avg * 3 - d1 - d3) = (avg * 3 - d1 - d3) / d1 :=
begin
  sorry
end

end ratio_of_second_to_first_l9_9258


namespace maximize_sector_area_l9_9754

noncomputable def maximum_sector_area (R: ℝ) (α: ℝ) (r: ℝ) (l: ℝ) : ℝ :=
  if h : l + 2 * r = 40 ∧ 0 < r ∧ r < 40 ∧ α = l / r then
    -(r - 10) ^ 2 + 100 
  else 
    0

theorem maximize_sector_area :
  ∃ α, ∃! S, ∃ r l, l + 2 * r = 40 ∧ 0 < r ∧ r < 40 ∧ α = l / r ∧ α = 2 ∧ S = 100 :=
by
  use 2
  use 100
  use 10
  use (40 - 2 * 10)
  split
  . sorry
  split
  . linarith
  . linarith
  . field_simp
    linarith
  . rfl
  . refl
  sorry

end maximize_sector_area_l9_9754


namespace smaller_number_l9_9997

theorem smaller_number (x y : ℤ) (h1 : x + y = 79) (h2 : x - y = 15) : y = 32 := by
  sorry

end smaller_number_l9_9997


namespace max_disjoint_pairs_l9_9501

noncomputable def max_pairs (n : ℕ) : ℕ :=
  (2 * n - 1) / 5

theorem max_disjoint_pairs (n : ℕ) (h : 1 ≤ n) :
  ∃ k, k = max_pairs n ∧ ∀ {a b c d : ℕ}, 
  ({1, 2, ..., n} ⊆ finset.range (n + 1)) → 
  (finset.pairwise_disjoint (λ p : (finset {a, b | a < b}),
  finset.pairwise_disjoint_by (λ x, a + b) (finset.image (λ (x y : ℕ), x + y)))) →
  (k = ⌊(2 * n - 1) / 5⌋) :=
sorry

end max_disjoint_pairs_l9_9501


namespace squirrels_acorns_l9_9216

theorem squirrels_acorns (squirrels : ℕ) (total_collected : ℕ) (acorns_needed_per_squirrel : ℕ) (total_needed : ℕ) (acorns_still_needed : ℕ) : 
  squirrels = 5 → 
  total_collected = 575 → 
  acorns_needed_per_squirrel = 130 → 
  total_needed = squirrels * acorns_needed_per_squirrel →
  acorns_still_needed = total_needed - total_collected →
  acorns_still_needed / squirrels = 15 :=
by
  sorry

end squirrels_acorns_l9_9216


namespace equivalent_problems_l9_9934

noncomputable theory

open Real

theorem equivalent_problems (n : ℤ) (hn : n ≥ 1) :
  (∃ w : ℤ, w > 0 ∧ (2 + sqrt 3)^(2 * n) - (2 - sqrt 3)^(2 * n) = w * sqrt 3) ∧
  (∃ k : ℤ, (2 + sqrt 3)^(2 * n) + (2 - sqrt 3)^(2 * n) = 2 * k) :=
by
  sorry

end equivalent_problems_l9_9934


namespace books_remaining_in_collection_l9_9624

-- Definitions corresponding to the conditions
def initial_books : ℕ := 75
def loaned_books : ℕ := 50  -- After rounding 49.99999999999999
def return_rate : ℝ := 0.70

-- The Lean statement we want to prove
theorem books_remaining_in_collection :
  let returned_books := return_rate * loaned_books
  let books_not_returned := loaned_books - returned_books.to_nat
  initial_books - books_not_returned = 60 :=
by {
  sorry
}

end books_remaining_in_collection_l9_9624


namespace hour_hand_degrees_per_hour_l9_9108

-- Definitions based on the conditions
def number_of_rotations_in_6_days : ℕ := 12
def degrees_per_rotation : ℕ := 360
def hours_in_6_days : ℕ := 6 * 24

-- Statement to prove
theorem hour_hand_degrees_per_hour :
  (number_of_rotations_in_6_days * degrees_per_rotation) / hours_in_6_days = 30 :=
by sorry

end hour_hand_degrees_per_hour_l9_9108


namespace decreasing_range_of_a_l9_9212

theorem decreasing_range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1)
    (h3 : ∀ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x < y → 
                  log a (2 - a * x) > log a (2 - a * y)) :
    1 < a ∧ a < 2 :=
  sorry

end decreasing_range_of_a_l9_9212


namespace number_of_diagonals_in_hexagonal_prism_l9_9866

structure HexagonalPrism (n : ℕ) :=
  (regular : Bool)
  (vertices_connected : (n = 6) → Prop)
  (neither_on_same_side_nor_base : Prop)

theorem number_of_diagonals_in_hexagonal_prism 
  (P : HexagonalPrism 6) 
  (h1 : P.regular = true)
  (h2 : P.vertices_connected 6)
  (h3 : P.neither_on_same_side_nor_base) : 
  (n : ℕ) → n * (n - 3) = 18 :=
by 
  sorry

end number_of_diagonals_in_hexagonal_prism_l9_9866


namespace rich_walked_distance_l9_9085

def total_distance_walked (d1 d2 : ℕ) := 
  d1 + d2 + 2 * (d1 + d2) + (d1 + d2 + 2 * (d1 + d2)) / 2

def distance_to_intersection (d1 d2 : ℕ) := 
  2 * (d1 + d2)

def distance_to_end_route (d1 d2 : ℕ) := 
  (d1 + d2 + distance_to_intersection d1 d2) / 2

def total_distance_one_way (d1 d2 : ℕ) := 
  (d1 + d2) + (distance_to_intersection d1 d2) + (distance_to_end_route d1 d2)

theorem rich_walked_distance
  (d1 : ℕ := 20)
  (d2 : ℕ := 200) :
  2 * total_distance_one_way d1 d2 = 1980 :=
by
  simp [total_distance_one_way, distance_to_intersection, distance_to_end_route, total_distance_walked]
  sorry

end rich_walked_distance_l9_9085


namespace minimum_three_pizzas_needed_l9_9600

def pizza := ℕ → Prop

def satisfies (prefs : pizza) (p : pizza) : Prop :=
  ∀ i, prefs i -> p i

def masha_prefs : pizza := λ i, i = 1 ∨ i = 2 -> i ≠ 3
def vanya_prefs : pizza := λ i, i = 4
def dasha_prefs : pizza := λ i, i ≠ 1
def nikita_prefs : pizza := λ i, i = 1 -> i ≠ 4
def igor_prefs : pizza := λ i, i ≠ 4 ∧ i = 3

def satisfies_all (p1 p2 : pizza) : Prop :=
  satisfies masha_prefs p1 ∨ satisfies masha_prefs p2 ∧
  satisfies vanya_prefs p1 ∨ satisfies vanya_prefs p2 ∧
  satisfies dasha_prefs p1 ∨ satisfies dasha_prefs p2 ∧
  satisfies nikita_prefs p1 ∨ satisfies nikita_prefs p2 ∧
  satisfies igor_prefs p1 ∨ satisfies igor_prefs p2

theorem minimum_three_pizzas_needed :
  ¬ ∃ (p1 p2 : pizza), satisfies_all p1 p2 → 
    ∃ (p3 : pizza), satisfies masha_prefs p3 ∧ satisfies vanya_prefs p3 ∧ 
    satisfies dasha_prefs p3 ∧ satisfies nikita_prefs p3 ∧ satisfies igor_prefs p3 := 
sorry

end minimum_three_pizzas_needed_l9_9600


namespace complex_fourth_power_l9_9696

-- Definitions related to the problem conditions
def trig_identity (θ : ℝ) : ℂ := complex.of_real (cos θ) + complex.I * complex.of_real (sin θ)
def z : ℂ := 3 * trig_identity (30 * real.pi / 180)
def result := (81 : ℝ) * (-1/2 + complex.I * (real.sqrt 3 / 2))

-- Statement of the problem
theorem complex_fourth_power : z^4 = result := by
  sorry

end complex_fourth_power_l9_9696


namespace sums_of_adjacent_cells_l9_9744

theorem sums_of_adjacent_cells (N : ℕ) (h : N ≥ 2) :
  ∃ (f : ℕ → ℕ → ℝ), (∀ i j, 1 ≤ i ∧ i < N → 1 ≤ j ∧ j < N → 
    (∃ (s : ℕ), 1 ≤ s ∧ s ≤ 2*(N-1)*N ∧ s = f i j + f (i + 1) j) ∧
    (∃ (s : ℕ), 1 ≤ s ∧ s ≤ 2*(N-1)*N ∧ s = f i j + f i (j + 1))) := sorry

end sums_of_adjacent_cells_l9_9744


namespace mul_point_five_point_three_l9_9273

theorem mul_point_five_point_three : 0.5 * 0.3 = 0.15 := 
by  sorry

end mul_point_five_point_three_l9_9273


namespace find_m_l9_9804

-- Mathematical conditions definitions
def line1 (x y : ℝ) (m : ℝ) : Prop := 3 * x + m * y - 1 = 0
def line2 (x y : ℝ) (m : ℝ) : Prop := (m + 2) * x - (m - 2) * y + 2 = 0

-- Given the lines are parallel
def lines_parallel (l1 l2 : ℝ → ℝ → ℝ → Prop) (m : ℝ) : Prop :=
  ∀ x y : ℝ, l1 x y m → l2 x y m → (3 / (m + 2)) = (m / (-(m - 2)))

-- The proof problem statement
theorem find_m (m : ℝ) : 
  lines_parallel (line1) (line2) m → (m = -6 ∨ m = 1) :=
by
  sorry

end find_m_l9_9804


namespace area_EPQR_26_l9_9082

-- Define the rectangle EFGH with specific dimensions
structure Rectangle where
  E F G H : (ℝ × ℝ)
  E_F_dist : (E.1 - F.1) ^ 2 + (E.2 - F.2) ^ 2 = 10 ^ 2
  F_G_dist : (F.1 - G.1) ^ 2 + (F.2 - G.2) ^ 2 = 6 ^ 2
  G_H_dist : (G.1 - H.1) ^ 2 + (G.2 - H.2) ^ 2 = 10 ^ 2
  H_E_dist : (H.1 - E.1) ^ 2 + (H.2 - E.2) ^ 2 = 6 ^ 2

-- Proof that verifies the area of quadrilateral EPQR given the conditions
theorem area_EPQR_26 (r : Rectangle) (mid_Q : Q = ((r.F.1 + r.G.1)/2, (r.F.2 + r.G.2)/2)) (mid_R : R = ((r.G.1 + r.H.1)/2, (r.G.2 + r.H.2)/2)) (mid_P : P = ((r.E.1 + r.F.1)/2, (r.E.2 + r.F.2)/2)) :
  area_EPQR r.E r.P r.Q r.R = 26 := sorry

end area_EPQR_26_l9_9082


namespace symmetry_axis_l9_9103

noncomputable def f (x : ℝ) : ℝ :=
  sin (x / 2) + sqrt 3 * cos (x / 2)

theorem symmetry_axis : ∃ k : ℤ, kπ + π / 2 = (x / 2) + π / 3 ∧ (x = -5π / 3) :=
by
  sorry

end symmetry_axis_l9_9103


namespace min_pos_period_of_f_l9_9111

def f (x : ℝ) : ℝ := (Real.sin (2 * x))^2

theorem min_pos_period_of_f :
  ∃ p > 0, ∀ x, f (x + p) = f x ∧
  (∀ q > 0, (∀ x, f (x + q) = f x) → p ≤ q) ∧ p = Real.pi / 2 :=
sorry

end min_pos_period_of_f_l9_9111


namespace complex_power_eq_rectangular_l9_9704

noncomputable def cos_30 := Real.cos (Real.pi / 6)
noncomputable def sin_30 := Real.sin (Real.pi / 6)

theorem complex_power_eq_rectangular :
  (3 * (cos_30 + (complex.I * sin_30)))^4 = -40.5 + 40.5 * complex.I * Real.sqrt 3 := 
sorry

end complex_power_eq_rectangular_l9_9704


namespace smallest_x_plus_y_l9_9349

theorem smallest_x_plus_y (x y : ℕ) (h_xy_not_equal : x ≠ y) (h_condition : (1 : ℚ) / x + 1 / y = 1 / 15) :
  x + y = 64 :=
sorry

end smallest_x_plus_y_l9_9349


namespace smallest_sum_of_xy_l9_9363

theorem smallest_sum_of_xy (x y : ℕ) (h1 : x ≠ y) (h2 : 0 < x) (h3 : 0 < y) 
  (h4 : (1:ℚ)/x + (1:ℚ)/y = (1:ℚ)/15) : x + y = 64 :=
sorry

end smallest_sum_of_xy_l9_9363


namespace min_distance_Rational_Man_Mathematic_Man_l9_9080

-- Definitions of paths for Rational Man and Mathematic Man
def Rational_Man_path (t : ℝ) : ℝ × ℝ := (Real.cos t, Real.sin t)
def Mathematic_Man_path (tau : ℝ) : ℝ × ℝ :=
  (3 + 2 * Real.cos tau, 4 * Real.sin tau)

-- Helper functions to calculate the distance
def distance (A M : ℝ × ℝ) : ℝ :=
  Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2)

-- Minimum distance computation
def min_distance : ℝ :=
  by
    sorry -- Proof goes here

-- The Lean statement of the problem
theorem min_distance_Rational_Man_Mathematic_Man :
  ∃ (t τ : ℝ), t ∈ Icc 0 (2 * Real.pi) ∧ τ ∈ Icc 0 (2 * Real.pi) ∧
  min_distance = distance (Rational_Man_path t) (Mathematic_Man_path τ) :=
  sorry -- Skipping proof details for now

end min_distance_Rational_Man_Mathematic_Man_l9_9080


namespace resulting_vector_l9_9641

def v : ℝ × ℝ × ℝ := (2, 1, 1)
def rotation_angle : ℝ := real.pi / 2  -- 90 degrees in radians
def passes_through_y_axis := true

theorem resulting_vector 
  (initial_vector : ℝ × ℝ × ℝ)
  (rotation_theta : ℝ)
  (passes_through_y : bool)
  (mag_condition : initial_vector.1^2 + initial_vector.2^2 + initial_vector.3^2 = 6)
  (orthogonality_condition : ∀ b : ℝ × ℝ × ℝ, (initial_vector.1 * b.1 + initial_vector.2 * b.2 + initial_vector.3 * b.3 = 0) → 
                                           (∃ y : ℝ × ℝ × ℝ, y.2 ≠ 0)) 
  : initial_vector = (2, 1, 1) ∧ rotation_theta = real.pi / 2 ∧ passes_through_y = true → 
    ∃ (x y z : ℝ), x = -Real.sqrt(6 / 11) ∧ y = 3 * Real.sqrt(6 / 11) ∧ z = -Real.sqrt(6 / 11) :=
by
  intro h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end resulting_vector_l9_9641


namespace actual_production_growth_rate_l9_9617

-- Define the conditions
variables (a : ℝ) (ha : a > 0)  -- a is the planned production value this year, assumed positive

-- Define the fact that planned production value is 11% increase from last year
def last_year_production := a / 1.11

-- Define the 1% increase in actual production this year
def actual_production := 1.01 * a

-- Define the expected growth rate compared to last year
def growth_rate := ((actual_production a - last_year_production a) / last_year_production a) * 100

-- The goal: prove the growth rate compared to last year is 12.11%
theorem actual_production_growth_rate : growth_rate a = 12.11 :=
by
  sorry

end actual_production_growth_rate_l9_9617


namespace equal_focal_distances_l9_9565

theorem equal_focal_distances (k : ℝ) (h₁ : k ≠ 0) (h₂ : 16 - k ≠ 0) 
  (h_hyperbola : ∀ x y, (x^2) / (16 - k) - (y^2) / k = 1)
  (h_ellipse : ∀ x y, 9 * x^2 + 25 * y^2 = 225) :
  0 < k ∧ k < 16 :=
sorry

end equal_focal_distances_l9_9565


namespace sum_largest_odd_factors_l9_9113

-- Mathematically prove the sum of the largest odd factors of numbers from 1 to 100 is 3344.

def largestOddFactor (n : ℕ) : ℕ :=
  if n % 2 = 1 then n else largestOddFactor (n / 2)

theorem sum_largest_odd_factors : 
  (∑ n in Finset.range 101, largestOddFactor n) = 3344 :=
by
  sorry

end sum_largest_odd_factors_l9_9113


namespace resulting_vector_l9_9642

def v : ℝ × ℝ × ℝ := (2, 1, 1)
def rotation_angle : ℝ := real.pi / 2  -- 90 degrees in radians
def passes_through_y_axis := true

theorem resulting_vector 
  (initial_vector : ℝ × ℝ × ℝ)
  (rotation_theta : ℝ)
  (passes_through_y : bool)
  (mag_condition : initial_vector.1^2 + initial_vector.2^2 + initial_vector.3^2 = 6)
  (orthogonality_condition : ∀ b : ℝ × ℝ × ℝ, (initial_vector.1 * b.1 + initial_vector.2 * b.2 + initial_vector.3 * b.3 = 0) → 
                                           (∃ y : ℝ × ℝ × ℝ, y.2 ≠ 0)) 
  : initial_vector = (2, 1, 1) ∧ rotation_theta = real.pi / 2 ∧ passes_through_y = true → 
    ∃ (x y z : ℝ), x = -Real.sqrt(6 / 11) ∧ y = 3 * Real.sqrt(6 / 11) ∧ z = -Real.sqrt(6 / 11) :=
by
  intro h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end resulting_vector_l9_9642


namespace gcd_lcm_product_360_l9_9121

theorem gcd_lcm_product_360 (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 360) :
    {d : ℕ | d = Nat.gcd a b } =
    {1, 2, 4, 8, 3, 6, 12, 24} := 
by
  sorry

end gcd_lcm_product_360_l9_9121


namespace solution_to_problem_l9_9805

-- Define vectors m and n in terms of λ
def m (λ : ℝ) : ℝ × ℝ := (λ + 1, 1)
def n (λ : ℝ) : ℝ × ℝ := (λ + 2, 2)

-- Define the condition for perpendicularity
def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

-- Statement to prove that λ must be -3
theorem solution_to_problem (λ : ℝ) (h : orthogonal (m λ.1 + n λ.1) (m λ.1 - n λ.1)) : λ = -3 :=
by
  sorry

end solution_to_problem_l9_9805


namespace vector_difference_magnitude_unit_length_l9_9399

open Real

noncomputable theory

variables (a b : EuclideanSpace ℝ (Fin 2)) 
          (angle : ℝ)
          (assume_a : ‖a‖ = 1)
          (assume_b : b = ![1, -1])
          (assume_angle : angle = π / 4)
          
theorem vector_difference_magnitude_unit_length :
  ‖a - b‖ = 1 := 
sorry

end vector_difference_magnitude_unit_length_l9_9399


namespace marginal_cost_per_product_calculation_l9_9959

def fixed_cost : ℝ := 12000
def total_cost : ℝ := 16000
def num_products : ℕ := 20

theorem marginal_cost_per_product_calculation :
  (total_cost - fixed_cost) / num_products = 200 := by
  sorry

end marginal_cost_per_product_calculation_l9_9959


namespace complex_power_rectangular_form_l9_9677

noncomputable def cos (θ : ℝ) : ℝ := sorry
noncomputable def sin (θ : ℝ) : ℝ := sorry

theorem complex_power_rectangular_form :
  (3 * complex.exp (complex.I * 30))^4 = -40.5 + 40.5 * complex.I * √3 :=
by
  sorry

end complex_power_rectangular_form_l9_9677


namespace X_on_EF_l9_9503

variables {α β γ : Type}
variables {A B C I D E F X : α} 
variables (ω : β)
variables [Incircle ABC ω I]
variables [TangencyPoints ω BC D]
variables [TangencyPoints ω CA E]
variables [TangencyPoints ω AB F]
variables [OrthogonalProjection C BI X]

theorem X_on_EF :
  X ∈ LineSegment E F :=
sorry

end X_on_EF_l9_9503


namespace player_1_winning_strategy_l9_9319

-- Define the properties and rules of the game
def valid_pair (a b : ℕ) : Prop := a > 0 ∧ b > 0 ∧ a + b = 2005

def move (current t a b : ℕ) : Prop := 
  current = t - a ∨ current = t - b

def first_player_wins (t a b : ℕ) : Prop :=
  ∀ k : ℕ, t > k * 2005 → ∃ m : ℕ, move (t - m) t a b

-- Main theorem statement
theorem player_1_winning_strategy : ∃ (t : ℕ) (a b : ℕ), valid_pair a b ∧ first_player_wins t a b :=
sorry

end player_1_winning_strategy_l9_9319


namespace sum_of_interior_angles_of_regular_polygon_l9_9977

theorem sum_of_interior_angles_of_regular_polygon :
  (∀ (n : ℕ), (n ≠ 0) ∧ ((360 / 45 = n) → (180 * (n - 2) = 1080))) := by sorry

end sum_of_interior_angles_of_regular_polygon_l9_9977


namespace percent_of_juniors_involved_in_sports_l9_9134

theorem percent_of_juniors_involved_in_sports
  (total_students : ℕ)
  (percent_juniors : ℝ)
  (juniors_in_sports : ℕ)
  (h1 : total_students = 500)
  (h2 : percent_juniors = 0.40)
  (h3 : juniors_in_sports = 140) :
  (juniors_in_sports : ℝ) / (total_students * percent_juniors) * 100 = 70 := 
by
  -- By conditions h1, h2, h3:
  sorry

end percent_of_juniors_involved_in_sports_l9_9134


namespace segments_equality_and_perpendicularity_l9_9267

noncomputable theory
open_locale classical

variables {A B C M₁ M₂ M₃ A₁ B₁ C₁ : EuclideanGeometry.Point}
variables {a b c : ℝ}

def midpoint (X Y : EuclideanGeometry.Point) : EuclideanGeometry.Point := sorry

def perpendicular_segment (X Y : EuclideanGeometry.Point) (d : ℝ) : EuclideanGeometry.Point := sorry

/- Given conditions -/
def conditions (A B C M₁ M₂ M₃ A₁ B₁ C₁ : EuclideanGeometry.Point) :=
  midpoint A B = M₃ ∧ midpoint B C = M₁ ∧ midpoint C A = M₂ ∧
  perpendicular_segment M₃ C₁ (1/2 * (EuclideanGeometry.dist A B)) ∧
  perpendicular_segment M₁ A₁ (1/2 * (EuclideanGeometry.dist B C)) ∧
  perpendicular_segment M₂ B₁ (1/2 * (EuclideanGeometry.dist C A))

/- Proof statement -/
theorem segments_equality_and_perpendicularity (A B C M₁ M₂ M₃ A₁ B₁ C₁ : EuclideanGeometry.Point) 
  (h : conditions A B C M₁ M₂ M₃ A₁ B₁ C₁) :
  EuclideanGeometry.dist A A₁ = EuclideanGeometry.dist B₁ C₁ ∧
  EuclideanGeometry.dist B B₁ = EuclideanGeometry.dist A₁ C₁ ∧
  EuclideanGeometry.dist C C₁ = EuclideanGeometry.dist A₁ B₁ ∧
  EuclideanGeometry.angle A₁ A B = 90 ∧
  EuclideanGeometry.angle B₁ B C = 90 ∧
  EuclideanGeometry.angle C₁ C A = 90 ∧
  EuclideanGeometry.collinear ({A₁, B₁, C₁}) :=
sorry

end segments_equality_and_perpendicularity_l9_9267


namespace decimal_addition_in_base_5_l9_9588

noncomputable def decimal_to_base (n : ℕ) (b : ℕ) : ℕ :=
  if b < 2 then 0 else
  let rec convert (n acc : ℕ) : ℕ :=
    if n = 0 then acc else convert (n / b) (acc * 10 + n % b)
  convert n 0

theorem decimal_addition_in_base_5 : decimal_to_base (34 + 27) 5 = 221 :=
by {
  -- ensuring that 34 + 27 = 61
  have h₁ : 34 + 27 = 61, by norm_num,
  rw h₁,
  -- converting 61 to base 5
  have h₂ : decimal_to_base 61 5 = 221, by norm_num,
  exact h₂,
}

end decimal_addition_in_base_5_l9_9588


namespace trader_loss_percent_l9_9003

noncomputable def loss_percent (SP1 SP2 CP1 CP2 : ℝ) : ℝ :=
  ((SP1 + SP2) - (CP1 + CP2)) / (CP1 + CP2) * 100

def trader_condition (selling_price : ℝ) (gain loss : ℝ) := 
  ∃ CP1 CP2 : ℝ, 
  (CP1 * (1 + gain) = selling_price) ∧
  (CP2 * (1 - loss) = selling_price)

theorem trader_loss_percent (selling_price : ℝ) (gain loss : ℝ) 
  (SP1 := selling_price) (SP2 := selling_price)
  (h_cond : trader_condition selling_price gain loss) :
  ∃ (CP1 CP2 : ℝ), 
  loss_percent SP1 SP2 CP1 CP2 ≈ 1.957 :=
by
  sorry

end trader_loss_percent_l9_9003


namespace range_of_x_in_sqrt_x_minus_2_l9_9460

theorem range_of_x_in_sqrt_x_minus_2 (x : ℝ) : (∃ y : ℝ, y = real.sqrt (x - 2)) → x ≥ 2 :=
by
  sorry

end range_of_x_in_sqrt_x_minus_2_l9_9460


namespace equipment_transfer_l9_9263

theorem equipment_transfer 
    (x y : ℕ) 
    (h1 : x < y)
    (h2 : 0.3 * x + 0.7 * x + 0.1 * (y + 0.3 * x) = 1.1 * x)
    (h3 : 0.1 * (y + 0.3 * x) = 0.1 * y + 0.03 * x)
    (h4 : 0.5 * (0.1 * (y + 0.3 * x)) = 0.05 * y + 0.015 * x)
    (h5 : 0.73 * x + 0.1 * y = 0.27 * x + 0.9 * y - 6)
    (h6 : 0.9 * (y + 0.3 * x) > 1.02 * y)
    (eqn : 0.46 * x - 0.8 * y = 6)
    (ineq : x > 4 / 9 * y) 
    : y = 17 := 
begin
  sorry
end

end equipment_transfer_l9_9263


namespace p_implies_q_not_q_implies_p_l9_9071

def p (a : ℝ) := a = Real.sqrt 2

def q (a : ℝ) := ∀ x y : ℝ, y = -(x : ℝ) → (x^2 + (y - a)^2 = 1)

theorem p_implies_q_not_q_implies_p (a : ℝ) : (p a → q a) ∧ (¬(q a → p a)) := 
    sorry

end p_implies_q_not_q_implies_p_l9_9071


namespace speed_of_l_l9_9202

-- Definitions for speeds and times
def constant_speed_factor : ℝ := 1.5
def start_time_l : ℝ := 9
def start_time_k : ℝ := 10
def meeting_time : ℝ := 12
def distance_apart : ℝ := 300

theorem speed_of_l (v : ℝ) (hlk : 1.5 * v = constant_speed_factor * v)
  (time_l : meeting_time - start_time_l = 3)
  (time_k : meeting_time - start_time_k = 2)
  (dist_l : time_l * v = 3 * v)
  (dist_k : time_k * (1.5 * v) = 3 * v)
  (total_distance : dist_l + dist_k = distance_apart)
  (total_eq : 6 * v = distance_apart) :
  v = 50 :=
begin
  sorry,
end

end speed_of_l_l9_9202


namespace problem_solution_l9_9383

variable {A B : Set ℝ}

def definition_A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def definition_B : Set ℝ := {x | x ≤ 3}

theorem problem_solution : definition_A ∩ definition_B = definition_A :=
by
  sorry

end problem_solution_l9_9383


namespace probability_green_light_is_8_over_15_l9_9250

def total_cycle_duration (red yellow green : ℕ) : ℕ :=
  red + yellow + green

def probability_green_light (red yellow green : ℕ) : ℚ :=
  green / (total_cycle_duration red yellow green : ℚ)

theorem probability_green_light_is_8_over_15 :
  probability_green_light 30 5 40 = 8 / 15 := by
  sorry

end probability_green_light_is_8_over_15_l9_9250


namespace sequence_closed_form_l9_9801

theorem sequence_closed_form (a : ℕ → ℤ) (h1 : a 1 = 1) (h_rec : ∀ n : ℕ, a (n + 1) = 2 * a n + 3) :
  ∀ n : ℕ, a n = 2^(n + 1) - 3 :=
by 
sorry

end sequence_closed_form_l9_9801


namespace complex_power_l9_9690

def cos_30 := Float.sqrt 3 / 2
def sin_30 := 1 / 2

theorem complex_power (i : ℂ) : 
  (3 * (⟨cos_30, sin_30⟩ : ℂ)) ^ 4 = -40.5 + 40.5 * i * Float.sqrt 3 := 
  sorry

end complex_power_l9_9690


namespace find_cookies_on_second_plate_l9_9131

theorem find_cookies_on_second_plate (a : ℕ → ℕ) :
  (a 1 = 5) ∧ (a 3 = 10) ∧ (a 4 = 14) ∧ (a 5 = 19) ∧ (a 6 = 25) ∧
  (∀ n, a (n + 2) - a (n + 1) = if (n + 1) % 2 = 0 then 5 else 4) →
  a 2 = 5 :=
by
  sorry

end find_cookies_on_second_plate_l9_9131


namespace smallest_x_plus_y_l9_9377

theorem smallest_x_plus_y (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_neq : x ≠ y) (h_eq : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 15) : x + y = 64 :=
sorry

end smallest_x_plus_y_l9_9377


namespace sum_of_interior_angles_of_regular_polygon_l9_9971

theorem sum_of_interior_angles_of_regular_polygon (n : ℕ) (h1: ∀ {n : ℕ}, ∃ (e : ℕ), (e = 360 / 45) ∧ (n = e)) :
  (180 * (n - 2)) = 1080 :=
by
  let n := (360 / 45)
  have h : n = 8 := by sorry
  calc
    180 * (n - 2) = 180 * (8 - 2) : by rw [h]
    ... = 1080 : by norm_num

end sum_of_interior_angles_of_regular_polygon_l9_9971


namespace average_monthly_increase_is_20_percent_l9_9568

-- Define the given conditions in Lean
def V_Jan : ℝ := 2 
def V_Mar : ℝ := 2.88 

-- Percentage increase each month over the previous month is the same
def consistent_growth_rate (x : ℝ) : Prop := 
  V_Jan * (1 + x)^2 = V_Mar

-- We need to prove that the monthly growth rate x is 0.2 (or 20%)
theorem average_monthly_increase_is_20_percent : 
  ∃ x : ℝ, consistent_growth_rate x ∧ x = 0.2 :=
by
  sorry

end average_monthly_increase_is_20_percent_l9_9568


namespace find_a_l9_9154

def distance (x y : ℝ) : ℝ := |x - y|

def sum_of_distances (a : ℝ) (k : ℕ) : ℝ :=
  ∑ i in finset.range 8, distance a (k + i)

noncomputable def satisfies_conditions (a b : ℝ) (k : ℕ) : Prop :=
  sum_of_distances a k = 612 ∧
  sum_of_distances b k = 240 ∧
  a + b = 100.5

theorem find_a (a b : ℝ) (k : ℕ) (h : satisfies_conditions a b k) :
  a = 27 ∨ a = -3 :=
sorry

end find_a_l9_9154


namespace boys_in_other_communities_l9_9015

theorem boys_in_other_communities (total_boys : ℕ) (muslim_percent hindu_percent sikh_percent : ℕ)
  (H_total : total_boys = 400)
  (H_muslim : muslim_percent = 44)
  (H_hindu : hindu_percent = 28)
  (H_sikh : sikh_percent = 10) :
  total_boys * (1 - (muslim_percent + hindu_percent + sikh_percent) / 100) = 72 :=
by
  sorry

end boys_in_other_communities_l9_9015


namespace third_competitor_eats_l9_9293

-- Define the conditions based on the problem description
def first_competitor_hot_dogs : ℕ := 12
def second_competitor_hot_dogs := 2 * first_competitor_hot_dogs
def third_competitor_hot_dogs := second_competitor_hot_dogs - (second_competitor_hot_dogs / 4)

-- The theorem we need to prove
theorem third_competitor_eats :
  third_competitor_hot_dogs = 18 := by
  sorry

end third_competitor_eats_l9_9293


namespace area_multiplier_l9_9435

-- Define the original area of the triangle
def original_area (a b θ : ℝ) : ℝ :=
  (a * b * Real.sin θ) / 2

-- Define the new area when sides are tripled
def new_area (a b θ : ℝ) : ℝ :=
  (3 * a) * (3 * b) * Real.sin θ / 2

-- State that the new area is 9 times the original area
theorem area_multiplier (a b θ : ℝ) : new_area a b θ = 9 * original_area a b θ :=
by
  calc
    new_area a b θ
        = (3 * a) * (3 * b) * Real.sin θ / 2 : rfl
    ... = 9 * (a * b * Real.sin θ) / 2 : by ring
    ... = 9 * original_area a b θ : rfl

end area_multiplier_l9_9435


namespace repeating_decimal_sum_of_digits_l9_9123

noncomputable def a_b_sum : ℕ := 
  let (a, b) := (2, 7) in a + b

theorem repeating_decimal_sum_of_digits : a_b_sum = 9 := 
by 
  sorry

end repeating_decimal_sum_of_digits_l9_9123


namespace smallest_prime_after_six_consecutive_nonprimes_l9_9181

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def nonprimes_between (a b : ℕ) : Prop :=
  ∀ n : ℕ, a < n ∧ n < b → ¬ is_prime n

def find_first_nonprime_gap : ℕ :=
  if H : ∃ a b : ℕ, b = a + 7 ∧ nonprimes_between a b ∧ is_prime (b + 1) then
    Classical.choose H
  else
    0

theorem smallest_prime_after_six_consecutive_nonprimes : find_first_nonprime_gap = 97 := by
  sorry

end smallest_prime_after_six_consecutive_nonprimes_l9_9181


namespace bryan_bookshelves_l9_9272

theorem bryan_bookshelves :
  ∀ (total_books books_per_shelf : ℕ),
  total_books = 42 →
  books_per_shelf = 2 →
  total_books / books_per_shelf = 21 :=
by
  intros total_books books_per_shelf h1 h2
  rw [h1, h2]
  norm_num

end bryan_bookshelves_l9_9272


namespace sum_of_four_l9_9849

variable {α : Type*} [LinearOrderedField α]

def geometric_sequence (a₁ q : α) : ℕ → α
| 0 := a₁
| (n+1) := geometric_sequence a₁ q n * q

def sum_geometric_sequence (a₁ q : α) (n : ℕ) : α :=
if q = 1 then a₁ * (n + 1) else a₁ * (1 - q^(n + 1)) / (1 - q)

theorem sum_of_four {a₁ q S₅ S₃ S₄ : α} (h₁ : a₁ = 1) (h₂ : sum_geometric_sequence a₁ q 4 = S₄) (h₃ : sum_geometric_sequence a₁ q 5 = S₅) (h₄ : sum_geometric_sequence a₁ q 3 = S₃) : S₅ = 5 * S₃ - 4 → S₄ = 15 :=
by
  sorry

end sum_of_four_l9_9849


namespace sum_of_interior_angles_l9_9975

-- Define the conditions:
def exterior_angle (n : ℕ) := 45

def sum_exterior_angles := 360

-- Define the Lean statement for the proof problem
theorem sum_of_interior_angles : ∃ n : ℕ, 
  sum_exterior_angles / exterior_angle n = n ∧
  (180 * (n - 2) = 1080) :=
by
  use 8
  split
  calc
    sum_exterior_angles / exterior_angle 8 = 360 / 45 := rfl
    ... = 8 := rfl
  calc
    180 * (8 - 2) = 180 * 6 := rfl
    ... = 1080 := rfl

end sum_of_interior_angles_l9_9975


namespace bottom_left_square_side_length_l9_9569

theorem bottom_left_square_side_length (x y : ℕ) 
  (h1 : 1 + (x - 1) = 1) 
  (h2 : 2 * x - 1 = (x - 2) + (x - 3) + y) :
  y = 4 :=
sorry

end bottom_left_square_side_length_l9_9569


namespace integral_sin_from_0_to_pi_div_2_l9_9299

theorem integral_sin_from_0_to_pi_div_2 :
  ∫ x in (0 : ℝ)..(Real.pi / 2), Real.sin x = 1 := by
  sorry

end integral_sin_from_0_to_pi_div_2_l9_9299


namespace smallest_prime_after_six_consecutive_nonprimes_l9_9171

-- Define what it means to be prime
def is_prime (n : ℕ) : Prop :=
  nat.prime n

-- Define the sequence of six consecutive nonprime numbers
def consecutive_nonprime_sequence (a : ℕ) : Prop :=
  ¬ is_prime a ∧ ¬ is_prime (a + 1) ∧ ¬ is_prime (a + 2) ∧
  ¬ is_prime (a + 3) ∧ ¬ is_prime (a + 4) ∧ ¬ is_prime (a + 5)

-- Define the condition that there is a smallest prime number greater than a sequence of six consecutive nonprime numbers
theorem smallest_prime_after_six_consecutive_nonprimes :
  ∃ p, is_prime p ∧ p > 96 ∧ consecutive_nonprime_sequence 90 :=
sorry

end smallest_prime_after_six_consecutive_nonprimes_l9_9171


namespace squirrels_acorns_l9_9217

theorem squirrels_acorns (squirrels : ℕ) (total_collected : ℕ) (acorns_needed_per_squirrel : ℕ) (total_needed : ℕ) (acorns_still_needed : ℕ) : 
  squirrels = 5 → 
  total_collected = 575 → 
  acorns_needed_per_squirrel = 130 → 
  total_needed = squirrels * acorns_needed_per_squirrel →
  acorns_still_needed = total_needed - total_collected →
  acorns_still_needed / squirrels = 15 :=
by
  sorry

end squirrels_acorns_l9_9217


namespace complex_power_l9_9691

def cos_30 := Float.sqrt 3 / 2
def sin_30 := 1 / 2

theorem complex_power (i : ℂ) : 
  (3 * (⟨cos_30, sin_30⟩ : ℂ)) ^ 4 = -40.5 + 40.5 * i * Float.sqrt 3 := 
  sorry

end complex_power_l9_9691


namespace remainder_2abc_mod_7_l9_9823

theorem remainder_2abc_mod_7
  (a b c : ℕ)
  (h₀ : 2 * a + 3 * b + c ≡ 1 [MOD 7])
  (h₁ : 3 * a + b + 2 * c ≡ 2 [MOD 7])
  (h₂ : a + b + c ≡ 3 [MOD 7])
  (ha : a < 7)
  (hb : b < 7)
  (hc : c < 7) :
  2 * a * b * c ≡ 0 [MOD 7] :=
sorry

end remainder_2abc_mod_7_l9_9823


namespace pow_div_pow_example_l9_9210

theorem pow_div_pow_example : (1000 : ℝ)^7 / (10 : ℝ)^17 = 10000 := by
  have h₁ : (1000 : ℝ) = (10 : ℝ) ^ 3 := by norm_num
  rw [h₁]
  -- the expression turns into ((10 : ℝ) ^ 3) ^ 7 / (10 : ℝ) ^ 17
  rw [pow_mul]
  -- simplifies to (10 : ℝ) ^ (3 * 7) / (10 : ℝ) ^ 17
  have h₂ : 3 * 7 = 21 := by norm_num
  rw [h₂]
  -- simplifies to (10 : ℝ) ^ 21 / (10 : ℝ) ^ 17
  rw [pow_div]
  -- simplifies to (10 : ℝ) ^ (21 - 17)
  have h₃ : 21 - 17 = 4 := by norm_num
  rw [h₃]
  -- simplifies to (10 : ℝ) ^ 4
  norm_num
  -- final answer 10000
  sorry

end pow_div_pow_example_l9_9210


namespace maximum_product_proof_l9_9417

noncomputable def maximum_product_of_five_two_digit_numbers : ℕ :=
  1785641760

theorem maximum_product_proof
  (a b c d e f g h i j : ℕ)
  (cond_a : a ≠ b) (cond_a' : a ≠ c) (cond_a'' : a ≠ d) (cond_a''' : a ≠ e) 
  (cond_a'''' : a ≠ f) (cond_a''''' : a ≠ g) (cond_a'''''' : a ≠ h) 
  (cond_a''''''' : a ≠ i) (cond_a'''''''' : a ≠ j)
  (cond_b : b ≠ c) (cond_b' : b ≠ d) (cond_b'' : b ≠ e) (cond_b''' : b ≠ f)
  (cond_b'''' : b ≠ g) (cond_b''''' : b ≠ h) (cond_b'''''' : b ≠ i)
  (cond_b''''''' : b ≠ j)
  (cond_c : c ≠ d) (cond_c' : c ≠ e) (cond_c'' : c ≠ f) (cond_c''' : c ≠ g)
  (cond_c'''' : c ≠ h) (cond_c''''' : c ≠ i) (cond_c'''''' : c ≠ j)
  (cond_d : d ≠ e) (cond_d' : d ≠ f) (cond_d'' : d ≠ g) (cond_d''' : d ≠ h)
  (cond_d'''' : d ≠ i)  (cond_d''''' : d ≠ j)
  (cond_e : e ≠ f) (cond_e' : e ≠ g) (cond_e'' : e ≠ h) (cond_e''' : e ≠ i)
  (cond_e'''' : e ≠ j)
  (cond_f : f ≠ g) (cond_f' : f ≠ h) (cond_f'' : f ≠ i) (cond_f''' : f ≠ j)
  (cond_g : g ≠ h) (cond_g' : g ≠ i) (cond_g'' : g ≠ j)
  (cond_h : h ≠ i) (cond_h' : h ≠ j)
  (cond_i : i ≠ j)
  (digits_range: a ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (digits_range' : b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (digits_range'' : c ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (digits_range''' : d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (digits_range'''' : e ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (digits_range''''' : f ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (digits_range'''''' : g ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (digits_range''''''' : h ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (digits_range'''''''' : i ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (digits_range''''''''' : j ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
    p1 * p2 * p3 * p4 * p5 = maximum_product_of_five_two_digit_numbers
    ∧ (∃ (digits : finset ℕ), digits = {a, b, c, d, e, f, g, h, i, j}
    ∧ ∀ digit ∈ digits, digit < 100)
    ∧ (∃ (nums : list ℕ), nums = [p1, p2, p3, p4, p5]
    ∧ ∀ n ∈ nums, 10 ≤ n ∧ n < 100)?
:= sorry

end maximum_product_proof_l9_9417


namespace max_dots_on_erased_half_l9_9119

theorem max_dots_on_erased_half {x : ℕ} (h : (37 + x) % 4 = 0) : x ≤ 3 :=
begin
  have h_mod : 37 % 4 = 1,
  { norm_num, },
  have h1 : (37 + x) % 4 = (1 + x) % 4,
  { rw [add_comm, ←Nat.add_mod], apply congr_arg, exact h_mod },
  have : (1 + x) % 4 = 0 := by assumption,
  have h2 : (1 + x) % 4 = (4 * (1 + x) / 4 + (1 + x) % 4) % 4,
  { rw Nat.add_div_self, use 0, },
  rw this at h2,
  have h3 : (4 * (1 + x) / 4 + 0) % 4 = 4 * (1 + x) / 4 % 4,
  { rw add_zero, },
  rw h2 at h3,
  sorry
end

end max_dots_on_erased_half_l9_9119


namespace constant_term_expansion_l9_9880

theorem constant_term_expansion (n : ℕ) (x : ℝ) (h : 2 ^ n = 4096) :
  let term := (-1) ^ 3 * nat.choose 12 3 in
  n = 12 → term = -220 :=
by
  sorry

end constant_term_expansion_l9_9880


namespace trajectory_eq_and_fixed_point_l9_9347

noncomputable def circle_eq (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16
def fixed_point (p : ℝ × ℝ) : Prop := p = (1, 0)
def ellipse_eq (x y : ℝ) : Prop:= x^2 / 4 + y^2 / 3 = 1
def line_eq (k x : ℝ) : ℝ := k * (x - 1)

theorem trajectory_eq_and_fixed_point (x y k t : ℝ) (M N P Q R : ℝ × ℝ) :
  circle_eq M.1 M.2 → fixed_point (1, 0) →
  (∀ p : ℝ × ℝ, ∃ q : ℝ × ℝ, circle_eq p.1 p.2 ∧ (q = perpendicular_bisector (p, (1, 0)))) →
  ellipse_eq N.1 N.2 →
  N ∈ list.map line_eq [P, Q] →
  ∃ R : ℝ × ℝ, R = (4, 0) := 
sorry

end trajectory_eq_and_fixed_point_l9_9347


namespace ufo_convention_males_l9_9261

-- Define the total number of attendees
constant total_attendees : Nat := 120

-- Define the conditions in the problem
constant num_female : Nat
constant num_male : Nat

axiom total_condition : num_female + num_male = total_attendees
axiom more_males_condition : num_male = num_female + 4

-- State the problem to prove the number of male attendees
theorem ufo_convention_males : num_male = 62 :=
by
  sorry

end ufo_convention_males_l9_9261


namespace marcus_saves_34_22_l9_9058

def max_spend : ℝ := 200
def shoe_price : ℝ := 120
def shoe_discount : ℝ := 0.30
def sock_price : ℝ := 25
def sock_discount : ℝ := 0.20
def shirt_price : ℝ := 55
def shirt_discount : ℝ := 0.10
def sales_tax_rate : ℝ := 0.08

def calc_discounted_price (price discount : ℝ) : ℝ := price * (1 - discount)

def total_cost_before_tax : ℝ :=
  calc_discounted_price shoe_price shoe_discount +
  calc_discounted_price sock_price sock_discount +
  calc_discounted_price shirt_price shirt_discount

def sales_tax : ℝ := total_cost_before_tax * sales_tax_rate

def final_cost : ℝ := total_cost_before_tax + sales_tax

def money_saved : ℝ := max_spend - final_cost

theorem marcus_saves_34_22 :
  money_saved = 34.22 :=
by sorry

end marcus_saves_34_22_l9_9058


namespace continuous_additive_function_is_linear_l9_9303

theorem continuous_additive_function_is_linear {f : ℝ → ℝ}
  (h₁ : ∀ x y : ℝ, f(x + y) = f(x) + f(y))
  (h₂ : continuous f) :
  ∃ c : ℝ, ∀ x : ℝ, f(x) = c * x :=
sorry

end continuous_additive_function_is_linear_l9_9303


namespace intersection_points_of_segments_l9_9931

noncomputable def num_intersection_points (A B C : Point) (P : Fin 60 → Point) (Q : Fin 50 → Point) : ℕ :=
  3000

theorem intersection_points_of_segments (A B C : Point) (P : Fin 60 → Point) (Q : Fin 50 → Point) :
  num_intersection_points A B C P Q = 3000 :=
  by sorry

end intersection_points_of_segments_l9_9931


namespace arrange_trees_with_conditions_l9_9446

structure Tree :=
  (x : ℝ) (y : ℝ) (is_apple : Bool)

noncomputable def validArrangement : Prop :=
  ∃ (A C O B1 B2 D1 D2 : Tree),
    -- Positions of the trees
    A = ⟨0, 0, true⟩ ∧
    C = ⟨1, 1, true⟩ ∧
    O = ⟨(1 / 2) + ε, (1 / 2) + ε, true⟩ ∧
    B1 = ⟨1, 0, false⟩ ∧
    B2 = ⟨1.1, 0, false⟩ ∧
    D1 = ⟨0, 1, false⟩ ∧
    D2 = ⟨0, 1.1, false⟩ ∧
    -- Ensure each tree's closest and furthest neighbours are the same type
    (∀ t ∈ [A, C, O], 
       (∃ closest : Tree, closest ≠ t ∧ closest.is_apple ∧
          ∀ other : Tree, other ≠ t ∧ other ≠ closest → dist t other ≥ dist t closest) ∧
       (∃ furthest : Tree, furthest ≠ t ∧ furthest.is_apple ∧
          ∀ other : Tree, other ≠ t ∧ other ≠ furthest → dist t other ≤ dist t furthest)) ∧
    (∀ t ∈ [B1, B2, D1, D2], 
       (∃ closest : Tree, closest ≠ t ∧ ¬closest.is_apple ∧
          ∀ other : Tree, other ≠ t ∧ other ≠ closest → dist t other ≥ dist t closest) ∧
       (∃ furthest : Tree, furthest ≠ t ∧ ¬furthest.is_apple ∧
          ∀ other : Tree, other ≠ t ∧ other ≠ furthest → dist t other ≤ dist t furthest))

theorem arrange_trees_with_conditions : validArrangement :=
  sorry

end arrange_trees_with_conditions_l9_9446


namespace not_persistent_no_persistent_smaller_than_given_l9_9893

noncomputable def isPersistent (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 0 → (∃ seq : list ℕ, seq.length = 10 ∧ (∀ d, d ∈ (list.range 10) ↔ d ∈ seq))

theorem not_persistent : ¬ isPersistent 526315789473684210 :=
  sorry

theorem no_persistent_smaller_than_given : ∀ n, (n < 526315789473684210) → ¬ isPersistent n :=
  begin
    intros n hn,
    exact not_persistent,
  end

end not_persistent_no_persistent_smaller_than_given_l9_9893


namespace seventh_diagram_shaded_triangles_l9_9867

-- Define the factorial function
def fact : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * fact n

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- The main theorem stating the relationship between the number of shaded sub-triangles and the factorial/Fibonacci sequence
theorem seventh_diagram_shaded_triangles :
  ∃ k : ℕ, (k : ℚ) = (fib 7 : ℚ) / (fact 7 : ℚ) ∧ k = 13 := sorry

end seventh_diagram_shaded_triangles_l9_9867


namespace television_price_reduction_l9_9639

theorem television_price_reduction (P : ℝ) (h₁ : 0 ≤ P):
  ((P - (P * 0.7 * 0.8)) / P) * 100 = 44 :=
by
  sorry

end television_price_reduction_l9_9639


namespace problem_part1_problem_part2_l9_9332

noncomputable def f (α : ℝ) := (Real.sin (π / 2 - α) + Real.sin (-π - α)) 
                                / (3 * Real.cos (2 * π + α) + Real.cos (3 * π / 2 - α))

theorem problem_part1 (α : ℝ) (h : f α = 3) : 
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α) = -1 / 3 := sorry

theorem problem_part2 (α : ℝ) (a r : ℝ) (h : Real.sin α = 2 * Real.cos α) 
  (h_center_on_xaxis : True)
  (h_distance : 2 / sqrt (2 ^ 2 + (-1) ^ 2) * abs a = sqrt 5)
  (h_chord_length : 2 * sqrt 2 = 2 * sqrt (r ^ 2 - (sqrt 5) ^ 2)) : 
  ∃ a, a = 5 / 2 ∨ a = -5 / 2 ∧ r ^ 2 = 7 ∧ 
        ∀ x y, (x - a) ^ 2 + y ^ 2 = 7 := sorry

end problem_part1_problem_part2_l9_9332


namespace find_x_l9_9301

noncomputable def series_sum (x : ℝ) : ℝ :=
∑' n : ℕ, (1 + 6 * n) * x^n

theorem find_x (x : ℝ) (h : series_sum x = 100) (hx : |x| < 1) : x = 3 / 5 := 
sorry

end find_x_l9_9301


namespace pencils_sold_for_profit_l9_9637

theorem pencils_sold_for_profit :
  let initial_pencils := 2000
  let cost_per_pencil := 0.15
  let selling_price_per_pencil := 0.35
  let desired_profit := 200.00
  let total_cost := initial_pencils * cost_per_pencil
  let total_revenue_needed := total_cost + desired_profit
  let pencils_to_sell := total_revenue_needed / selling_price_per_pencil
  Math.ceil pencils_to_sell = 1429 :=
by
  sorry

end pencils_sold_for_profit_l9_9637


namespace extreme_values_f_on_0_to_3_l9_9105

def f (x : ℝ) : ℝ := (1 / 3) * x ^ 3 - 4 * x + 4

theorem extreme_values_f_on_0_to_3 :
  (∀ x ∈ (Set.Icc (0 : ℝ) 3), f x ≤ 4) ∧
  (∃ x ∈ (Set.Icc (0 : ℝ) 3), f x = 4) ∧
  (∀ x ∈ (Set.Icc (0 : ℝ) 3), f x ≥ - (4 / 3)) ∧
  (∃ x ∈ (Set.Icc (0 : ℝ) 3), f x = - (4 / 3)) :=
by
  -- Proof omitted
  sorry

end extreme_values_f_on_0_to_3_l9_9105


namespace complex_power_rectangular_form_l9_9672

noncomputable def cos (θ : ℝ) : ℝ := sorry
noncomputable def sin (θ : ℝ) : ℝ := sorry

theorem complex_power_rectangular_form :
  (3 * complex.exp (complex.I * 30))^4 = -40.5 + 40.5 * complex.I * √3 :=
by
  sorry

end complex_power_rectangular_form_l9_9672


namespace zero_point_interval_l9_9966

noncomputable def f (x : ℝ) := Real.exp x + x - 2

theorem zero_point_interval : ∃ c ∈ Ioo (0 : ℝ) (1 : ℝ), f c = 0 := by
  have h0 : f 0 < 0 := by
    calc
      f 0 = Real.exp 0 + 0 - 2 := rfl
      ... = 1 - 2 := by norm_num
      ... = -1 := by norm_num

  have h1 : f 1 > 0 := by
    calc
      f 1 = Real.exp 1 + 1 - 2 := rfl
      ... = Real.exp 1 - 1 := by ring
      ... > 0 := by
        have : Real.exp 1 > 1 := Real.exp_pos 1
        linarith

  have h_cont : Continuous (f) := by continuity

  exact IntermediateValueTheorem (Continuous.continuous_on h_cont) (Set.mem_Icc.2 ⟨le_of_lt h0, h1⟩) 0

end zero_point_interval_l9_9966


namespace height_of_building_l9_9155

-- Define the given conditions
def shadow_building := (80 : ℝ)
def height_tree := (25 : ℝ)
def shadow_tree := (30 : ℝ)

-- Define the theorem statement to be proven
theorem height_of_building : 
  (shadow_building / shadow_tree) * height_tree ≈ 67 :=
by calc
  (shadow_building / shadow_tree) * height_tree 
    = (80 / 30) * 25 : by sorry
-- round the resultant value to the nearest whole number using approximation
    ≈ 67 : by sorry

end height_of_building_l9_9155


namespace min_distinct_integers_l9_9741

theorem min_distinct_integers (a : Fin 2006 → ℕ)
  (h_pos : ∀ i, 1 ≤ i → a i > 0)
  (h_distinct_ratios : ∀ i j, 1 ≤ i ∧ i < 2006 → 1 ≤ j ∧ j < 2006 → i ≠ j → a i / a (i+1) ≠ a j / a (j+1)) :
  ∃ S : Fin 2006 → ℕ, S = a ∧ S.toFinset.card = 1004 := 
sorry

end min_distinct_integers_l9_9741


namespace S₄_eq_15_l9_9843

-- Definitions based on the given conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum a

def sequence_condition (a : ℕ → ℝ) : Prop :=
  is_geometric_sequence a ∧ a 1 = 1 ∧ sum_of_first_n_terms a 5 = 5 * sum_of_first_n_terms a 3 - 4

theorem S₄_eq_15 (a : ℕ → ℝ) (q : ℝ) :
  sequence_condition a →
  (∀ n, a n = 1 * q ^ (n-1)) → 
  sum_of_first_n_terms a 4 = 15 :=
sorry

end S₄_eq_15_l9_9843


namespace vector_relation_AD_l9_9043

variables {P V : Type} [AddCommGroup V] [Module ℝ V]
variables (A B C D : P) (AB AC AD BC BD CD : V)
variables (hBC_CD : BC = 3 • CD)

theorem vector_relation_AD (h1 : BC = 3 • CD)
                           (h2 : AD = AB + BD)
                           (h3 : BD = BC + CD)
                           (h4 : BC = -AB + AC) :
  AD = - (1 / 3 : ℝ) • AB + (4 / 3 : ℝ) • AC :=
by
  sorry

end vector_relation_AD_l9_9043


namespace polynomial_sum_evaluation_l9_9505

noncomputable def q1 : Polynomial ℤ := Polynomial.X^3
noncomputable def q2 : Polynomial ℤ := Polynomial.X^2 + Polynomial.X + 1
noncomputable def q3 : Polynomial ℤ := Polynomial.X - 1
noncomputable def q4 : Polynomial ℤ := Polynomial.X^2 + 1

theorem polynomial_sum_evaluation :
  q1.eval 3 + q2.eval 3 + q3.eval 3 + q4.eval 3 = 52 :=
by
  sorry

end polynomial_sum_evaluation_l9_9505


namespace shaded_area_correct_l9_9613

-- Definitions of the given conditions
def circle_radius_P := 5
def square_side_length := 3
def inner_circle_radius := 3 / 2
def angle_PRU := 135 * (Real.pi/180)
def area_sector_PTU := 135/360 * Real.pi * circle_radius_P^2 
def area_triangle_RTU := (1 / 2) * 4 * 4 * Real.sin(angle_PRU)

-- Main statement to prove
theorem shaded_area_correct :
    area_sector_PTU - area_triangle_RTU = (25 * Real.pi) / 24 - 4 * Real.sqrt 2 := 
by 
    -- Sorry indicates that the proof is omitted.
    sorry

end shaded_area_correct_l9_9613


namespace erasers_in_each_box_l9_9994

theorem erasers_in_each_box (boxes : ℕ) (price_per_eraser : ℚ) (total_money_made : ℚ) (total_erasers_sold : ℕ) (erasers_per_box : ℕ) :
  boxes = 48 → price_per_eraser = 0.75 → total_money_made = 864 → total_erasers_sold = 1152 → total_erasers_sold / boxes = erasers_per_box → erasers_per_box = 24 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end erasers_in_each_box_l9_9994


namespace weight_of_crate_and_carton_truck_capacity_total_load_in_ounces_l9_9248

-- Definitions
variable (x y : ℝ)
-- The two given conditions
theorem weight_of_crate_and_carton (h1: 24 * x + 72 * y = 408) (h2: x = 3 * y) : 
  x = 8.5 ∧ y = 2.8333 := 
by
  sorry

-- Prove truck can accommodate 24 crates and 72 cartons without exceeding maximum load capacity
theorem truck_capacity (x y : ℝ) (h : x = 8.5 ∧ y = 2.8333) : 
  24 * x + 72 * y ≤ 1200 := 
by
  sorry

-- Prove total load conversion to ounces
theorem total_load_in_ounces (h : 24 * 8.5 + 72 * 2.8333 = 408) :
  let lbs := 408 * 2.20462 in
  let oz := lbs * 16 in
  oz = 14391.75936 ∧ Int.round oz = 14392 :=
by
  sorry

end weight_of_crate_and_carton_truck_capacity_total_load_in_ounces_l9_9248


namespace primes_equal_if_sqrt_sum_integer_l9_9952

theorem primes_equal_if_sqrt_sum_integer (p q : ℕ) [Prime p] [Prime q]
  (h : ∃ (z : ℤ), (↑(p * p) + 7 * ↑(p * q) + ↑(q * q)).sqrt + 
                  (↑(p * p) + 14 * ↑(p * q) + ↑(q * q)).sqrt = z) : p = q := by
  sorry

end primes_equal_if_sqrt_sum_integer_l9_9952


namespace smallest_prime_after_six_nonprimes_l9_9191

open Nat

theorem smallest_prime_after_six_nonprimes : 
  ∃ p : ℕ, prime p ∧ p > 0 ∧
  (∃ n : ℕ, ¬prime n ∧ ¬prime (n+1) ∧ ¬prime (n+2) ∧ ¬prime (n+3) ∧ ¬prime (n+4) ∧ ¬prime (n+5) ∧ prime (n+6) ∧ n+6 = p) ∧ 
  p = 89 :=
sorry

end smallest_prime_after_six_nonprimes_l9_9191


namespace division_problem_l9_9325

-- Define the set
def my_set : set ℕ := {2, 3, 9, 27, 81, 243, 567}

-- Define the dividend, divisor, quotient, and remainder
def dividend := 567
def divisor := 243
def quotient := 2
def remainder := 81

-- Define the conditions
def condition1 := dividend ∈ my_set
def condition2 := divisor ∈ my_set
def condition3 := quotient * divisor + remainder = dividend
def condition4 := 0 ≤ remainder ∧ remainder < divisor

-- The main theorem
theorem division_problem :
  dividend ∈ my_set ∧
  divisor ∈ my_set ∧
  quotient * divisor + remainder = dividend ∧
  0 ≤ remainder ∧ remainder < divisor :=
by
  -- adding the skipped proof steps here
  sorry

end division_problem_l9_9325


namespace percent_less_than_m_plus_d_l9_9595

theorem percent_less_than_m_plus_d (m : ℝ) (d : ℝ) (symmetric_about_m : Prop) :
  (∀ x : ℝ, percent_within_d m d 64) → percent_less_than (m + d) = 82 :=
by sorry

end percent_less_than_m_plus_d_l9_9595


namespace total_distance_traveled_in_12_hours_l9_9993

variable (n a1 d : ℕ) (u : ℕ → ℕ)

def arithmetic_seq_sum (n : ℕ) (a1 : ℕ) (d : ℕ) : ℕ :=
  n * a1 + (n * (n - 1) * d) / 2

theorem total_distance_traveled_in_12_hours :
  arithmetic_seq_sum 12 55 2 = 792 := by
  sorry

end total_distance_traveled_in_12_hours_l9_9993


namespace dot_product_BE_AC_l9_9766

variables {A B C D E : Type}
variables (rA rB rC rD rE : A)

noncomputable def side_length : ℝ := 4
noncomputable def BD_ratio : ℝ := 2

-- Define what it means for a triangle to be equilateral with side length 4
def is_equilateral_triangle (A B C : A) [metric_space A] [has_dist A] : Prop :=
dist A B = side_length ∧ dist B C = side_length ∧ dist C A = side_length

-- Define the condition BD = 2 * DC
def divides_in_ratio (B D C : A) (ratio : ℝ) [metric_space A] [has_dist A] : Prop :=
dist B D = ratio * dist D C

-- Define the midpoint condition
def is_midpoint (E D A : A) [add_comm_group A] [vector_space ℝ A] : Prop :=
E = 1/2 • (D + A)

theorem dot_product_BE_AC 
  [inner_product_space ℝ A] [metric_space A] [has_dist A] [add_comm_group A] [vector_space ℝ A]
  (h_triangle: is_equilateral_triangle rA rB rC)
  (h_divides: divides_in_ratio rB rD rC BD_ratio)
  (h_midpoint: is_midpoint rE rD rA) :
  inner (rB - rE) (rC - rA) = -4/3 :=
sorry

end dot_product_BE_AC_l9_9766


namespace hexagon_to_square_l9_9663

theorem hexagon_to_square :
  ∃ (s : ℝ), ∀ (h : ℝ), ∃ (square_side : ℝ), 
    h = (s * sqrt 3) / 2 ∧ square_side = h ∧
    ∀ (pieces : list (ℝ × ℝ)), 
    (sum (pieces.map (λ p, p.1))) = (sum (pieces.map (λ p, p.2))) ->
    ∃ (rearrangement : list (ℝ × ℝ)) ,
    rearrangement /= pieces ∧ 
    square_side^2 = sum (rearrangement.map (λ p, p.1)) :=
sorry

end hexagon_to_square_l9_9663


namespace rhombus_area_l9_9104

noncomputable def polynomial : Polynomial ℂ := 
  (Polynomial.C 1) + 
  (Polynomial.C (4 * Complex.I)) * Polynomial.X +
  (Polynomial.C ((-5) + 5 * Complex.I)) * Polynomial.X ^ 2 + 
  (Polynomial.C ((-10) - Complex.I)) * Polynomial.X ^ 3 + 
  (Polynomial.C (1 - 6 * Complex.I)) * Polynomial.X ^ 4

theorem rhombus_area (h : polynomial.roots.to_finset.card = 4) :
  let a := polynomial.roots[0],
      b := polynomial.roots[1],
      c := polynomial.roots[2],
      d := polynomial.roots[3] in
  is_rhombus a b c d ∧ complex_plane_area a b c d = 10 :=
sorry

end rhombus_area_l9_9104


namespace correct_propositions_l9_9284

def proposition_1 (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 > 0 ∧ x2 < 0 ∧ x1 ≠ x2 ∧ (x1 + x2 = 3 - a) ∧ (x1 * x2 = a) → a < 0

def proposition_2 : Prop :=
  ∀ x : ℝ, y = sqrt(x^2 - 1) + sqrt(1 - x^2) → 
  (¬ (∀ x : ℝ, y = sqrt(x^2 - 1) + sqrt(1 - x^2))) 

def proposition_3 (f : ℝ → ℝ) : Prop :=
  (∀ x, f x ≥ -2 ∧ f x ≤ 2) →  (∀ x, f (x + 1) ≥ -2 ∧ f (x + 1) ≤ 2)

def proposition_4 (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) → (∀ x, f (1 - x) = f (x - 1))

def proposition_5 (a : ℝ) : Prop :=
  ∀ a : ℝ, (¬ ∃ m : ℕ, (curve : ℝ → ℝ :=
    λ x, if x ∈ Icc (- sqrt 3) (sqrt 3) then 3 - x^2 else x^2 - 3) ∧ 
    (count_common_points curve y = m) ∧ (m = 1))

theorem correct_propositions : 
  (proposition_1 ∧ proposition_5) ∧ 
  ¬(proposition_2) ∧ ¬(proposition_3) ∧ ¬(proposition_4) :=
by sorry

end correct_propositions_l9_9284


namespace count_valid_n_l9_9312

theorem count_valid_n :
  ∃! (S : Finset ℕ), (∀ n ∈ S, 2 ≤ n ∧ n ≤ 98 ∧ (n % 2 = 0)) ∧ (S.card = 24) ∧ 
    ∀ n ∈ S, (∏ k in Finset.range(49), (n - (2 * (k + 1)))) < 0 :=
by
  sorry

end count_valid_n_l9_9312


namespace ufo_convention_males_l9_9262

-- Define the total number of attendees
constant total_attendees : Nat := 120

-- Define the conditions in the problem
constant num_female : Nat
constant num_male : Nat

axiom total_condition : num_female + num_male = total_attendees
axiom more_males_condition : num_male = num_female + 4

-- State the problem to prove the number of male attendees
theorem ufo_convention_males : num_male = 62 :=
by
  sorry

end ufo_convention_males_l9_9262


namespace sum_of_angles_S_R_l9_9522

theorem sum_of_angles_S_R 
  (A B R E C : Point) -- Points lie on a circle
  (arc_BR arc_RE : ℝ) -- The measures of arcs BR and RE
  (hBR : arc_BR = 48) (hRE : arc_RE = 46) -- Given measures of arcs BR and RE
  : ∃ (S R : ℝ), S + R = 47 := 
begin
  -- Define the total arc BE
  let arc_BE := arc_BR + arc_RE,
  have h_arc_BE : arc_BE = 94 := by linarith [hBR, hRE],
  -- Define angle S
  let S := (arc_BE - 0) / 2,
  have h_S : S = 47 := by norm_num [h_arc_BE],
  -- Define angle R
  let R := 0 / 2,
  have h_R : R = 0 := by norm_num,
  -- The sum of the measures of angles S and R
  use [S, R],
  exact h_S.symm ▸ h_R.symm ▸ rfl
end

end sum_of_angles_S_R_l9_9522


namespace complex_power_rectangular_form_l9_9674

noncomputable def cos (θ : ℝ) : ℝ := sorry
noncomputable def sin (θ : ℝ) : ℝ := sorry

theorem complex_power_rectangular_form :
  (3 * complex.exp (complex.I * 30))^4 = -40.5 + 40.5 * complex.I * √3 :=
by
  sorry

end complex_power_rectangular_form_l9_9674


namespace unique_solution_x3_minus_y3_eq_xy_plus_41_l9_9304

theorem unique_solution_x3_minus_y3_eq_xy_plus_41 :
  ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x^3 - y^3 = x * y + 41 ∧ x = 5 ∧ y = 4 :=
begin
  use (5, 4),
  repeat { split },
  { exact nat.prime.one_lt 5 },
  { exact nat.prime.one_lt 4 },
  { norm_num },
  { norm_num },
  { intros x y Hxy,
    cases Hxy with hx (hy (hsol heqx heqy)),
    subst_vars,
    unfold function.injective at *,
    norm_num at *,
    exact heqx, },
end

end unique_solution_x3_minus_y3_eq_xy_plus_41_l9_9304


namespace sum_interior_angles_l9_9006

theorem sum_interior_angles (h : ∀ (n : ℕ), (40 : ℝ) * n = 360) :
  ∑ i in finset.range (9 - 2), (180 : ℝ) = 1260 :=
by
  sorry

end sum_interior_angles_l9_9006


namespace integer_solutions_of_quadratic_l9_9742

theorem integer_solutions_of_quadratic (k : ℤ) :
  ∀ x : ℤ, (6 - k) * (9 - k) * x^2 - (117 - 15 * k) * x + 54 = 0 ↔
  k = 3 ∨ k = 7 ∨ k = 15 ∨ k = 6 ∨ k = 9 :=
by
  sorry

end integer_solutions_of_quadratic_l9_9742


namespace determine_a_l9_9411
open Set

theorem determine_a (a : ℝ) :
  let P := {-1, 1}
  let Q := {x : ℝ | a * x = 1}
  P ∪ Q = P → a ∈ {-1, 0, 1} :=
by
  let P := {-1, 1}
  let Q := {x : ℝ | a * x = 1}
  intro h
  sorry

end determine_a_l9_9411


namespace smallest_prime_after_six_nonprimes_l9_9185

open Nat

theorem smallest_prime_after_six_nonprimes : 
  ∃ p : ℕ, prime p ∧ p > 0 ∧
  (∃ n : ℕ, ¬prime n ∧ ¬prime (n+1) ∧ ¬prime (n+2) ∧ ¬prime (n+3) ∧ ¬prime (n+4) ∧ ¬prime (n+5) ∧ prime (n+6) ∧ n+6 = p) ∧ 
  p = 89 :=
sorry

end smallest_prime_after_six_nonprimes_l9_9185


namespace price_reduction_for_1200_profit_price_reduction_for_max_profit_l9_9611

noncomputable def average_daily_sales : ℕ := 20
noncomputable def profit_per_piece : ℕ := 40
noncomputable def sales_increase_per_dollar_reduction : ℕ := 2
variable (x : ℝ)

def average_daily_profit (x : ℝ) : ℝ :=
  (profit_per_piece - x) * (average_daily_sales + sales_increase_per_dollar_reduction * x)

theorem price_reduction_for_1200_profit :
  ((average_daily_profit x) = 1200) → (x = 20) := by
  sorry

theorem price_reduction_for_max_profit :
  (∀ x, average_daily_profit x ≤ average_daily_profit 15) ∧ (average_daily_profit 15 = 1250) := by
  sorry

end price_reduction_for_1200_profit_price_reduction_for_max_profit_l9_9611


namespace geom_seq_sum_4_l9_9855

noncomputable def geom_seq (n : ℕ) (q : ℝ) (a₁ : ℝ) : ℝ :=
  a₁ * q^(n - 1)

noncomputable def sum_geom_seq (n : ℕ) (q : ℝ) (a₁ : ℝ) : ℝ :=
  if q = 1 then a₁ * n else (a₁ * (1 - q^n) / (1 - q))

theorem geom_seq_sum_4 {q : ℝ} (hq : q > 0) (hq1 : q ≠ 1) :
  let a₁ := 1 in
  let S5 := sum_geom_seq 5 q a₁ in
  let S3 := sum_geom_seq 3 q a₁ in
  S5 = 5 * S3 - 4 →
  sum_geom_seq 4 q a₁ = 15 :=
by
  sorry

end geom_seq_sum_4_l9_9855


namespace nat_digit_problem_l9_9968

theorem nat_digit_problem :
  ∀ n : Nat, (n % 10 = (2016 * (n / 2016)) % 10) → (n = 4032 ∨ n = 8064 ∨ n = 12096 ∨ n = 16128) :=
by
  sorry

end nat_digit_problem_l9_9968


namespace work_completion_times_l9_9512

variable {M P S : ℝ} -- Let M, P, and S be work rates for Matt, Peter, and Sarah.

theorem work_completion_times (h1 : M + P + S = 1 / 15)
                             (h2 : 10 * (P + S) = 7 / 15) :
                             (1 / M = 50) ∧ (1 / (P + S) = 150 / 7) :=
by
  -- Proof comes here
  -- Calculation skipped
  sorry

end work_completion_times_l9_9512


namespace ponchik_cakes_l9_9872

/-
Given:
- The numbers of honey cakes eaten by Ponchik: 
  Z (instead of exercise), P (instead of walk), R (instead of run), and C (instead of swim)
- Ratios: 
  Z / P = 3 / 2, P / R = 5 / 3, and R / C = 6 / 5
- A total of 216 honey cakes eaten in a day.

Prove:
- The difference between honey cakes eaten instead of exercise and swim is 60.
-/

theorem ponchik_cakes (Z P R C : ℕ) 
  (h_ratio1 : Z / P = 3 / 2) 
  (h_ratio2 : P / R = 5 / 3) 
  (h_ratio3 : R / C = 6 / 5) 
  (h_total : Z + P + R + C = 216) : 
  Z - C = 60 := sorry

end ponchik_cakes_l9_9872


namespace dishonest_shopkeeper_gain_percentage_l9_9233

theorem dishonest_shopkeeper_gain_percentage (x y z : ℝ) :
  let false_weight := 892 / 1000 in
  let gain_percent (price : ℝ) :=
    ((1 - false_weight) / false_weight) * 100 in
  gain_percent x ≈ 12.107 ∧ gain_percent y ≈ 12.107 ∧ gain_percent z ≈ 12.107 :=
by
  sorry

end dishonest_shopkeeper_gain_percentage_l9_9233


namespace part1_part2_l9_9789

-- Definitions and conditions for Part 1
def f (a b x : ℝ) : ℝ := exp (a * x) - x - b

def has_two_zeros (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0

-- Problem statement for Part 1
theorem part1 (a : ℝ) : has_two_zeros (f a 1) ↔ (0 < a ∧ a < 1) ∨ (1 < a) :=
sorry

-- Definitions and conditions for Part 2
def min_lambda (a x1 x2 λ : ℝ) : Prop :=
  f a 0 x1 = 0 ∧ f a 0 x2 = 0 ∧ x1 < x2 ∧ 
  (∀ λ', (λ' * log x1 + log x2) ≥ e)

-- Problem statement for Part 2
theorem part2 (a x1 x2 λ : ℝ) (h : min_lambda a x1 x2 λ) : λ = exp 2 - 2 * exp 1 :=
sorry

end part1_part2_l9_9789


namespace prove_combined_area_l9_9011

noncomputable def garden1_length := 500
noncomputable def garden2_length := 625
noncomputable def garden3_length := 750

def garden1_breadth := 400
def garden2_breadth := 500
def garden3_breadth := 600

def garden1_area := garden1_length * garden1_breadth
def garden2_area := garden2_length * garden2_breadth
def garden3_area := garden3_length * garden3_breadth

def combined_area := garden1_area + garden2_area + garden3_area

theorem prove_combined_area : combined_area = 962500 := by
  sorry

end prove_combined_area_l9_9011


namespace beth_cans_of_corn_l9_9658

theorem beth_cans_of_corn (C P : ℕ) (h1 : P = 2 * C + 15) (h2 : P = 35) : C = 10 :=
by
  sorry

end beth_cans_of_corn_l9_9658


namespace exists_2016_consecutive_with_16_primes_l9_9723

noncomputable def S (n : ℕ) : ℕ :=
  Nat.countPrimesInRange n (n + 2015)

theorem exists_2016_consecutive_with_16_primes :
  ∃ n : ℕ, S(n) = 16 :=
begin
  sorry
end

end exists_2016_consecutive_with_16_primes_l9_9723


namespace correct_speed_l9_9927

def distance_40_late (d : ℝ) (t : ℝ) : Prop :=
  d = 40 * (t + 1/20)

def distance_60_early (d : ℝ) (t : ℝ) : Prop :=
  d = 60 * (t - 1/20)

theorem correct_speed (d t : ℝ) :
  distance_40_late d t →
  distance_60_early d t →
  (d = 12 ∧ t = 1/4) →
  (48 = d / t) :=
by {
  intros h1 h2 h3,
  sorry
}

end correct_speed_l9_9927


namespace no_extreme_in_0_1_two_parallel_tangents_three_parallel_tangent_groups_l9_9009

noncomputable def f (x t : ℝ) : ℝ := x^3 - t * x^2 + 1

theorem no_extreme_in_0_1 (t : ℝ) : 
  (t ≤ 0 ∨ t ≥ 3/2) → 
  ∀ x ∈ (set.Ioo 0 1), 
  deriv (λ x, f x t) x ≠ 0 := 
sorry

theorem two_parallel_tangents (t : ℝ) : 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ deriv (λ x, f x t) x1 = deriv (λ x, f x t) x2 :=
sorry

theorem three_parallel_tangent_groups (x1 x2 t : ℝ) (h_t : t = 3) 
  (dist : (f x1 3 - f x2 3) / sqrt (1 + (deriv (λ x, f x 3) x1)^2) = 4) : 
  ∃ x1 x2, x1 ≠ x2 ∧ x1 + x2 = 2 ∧  ∃ λ : ℝ, λ = (x1 - 1)^2 ∧ 
  (λ = 1 ∨ (λ^2 - 8 * λ + 10 = 0 ∧ λ ≠ 1)) ∧ 
  (x1, x2) ∈ {(a, b) | a > b ∧ a + b = 2} := 
sorry

end no_extreme_in_0_1_two_parallel_tangents_three_parallel_tangent_groups_l9_9009


namespace smallest_x_plus_y_l9_9353

theorem smallest_x_plus_y (x y : ℕ) (h_xy_not_equal : x ≠ y) (h_condition : (1 : ℚ) / x + 1 / y = 1 / 15) :
  x + y = 64 :=
sorry

end smallest_x_plus_y_l9_9353


namespace altitudes_through_O_l9_9521

variables {A B C O A1 B1 C1 : Type*}
variables [affine_space ℝ A] [affine_space ℝ B] [affine_space ℝ C] [affine_space ℝ O]
variables [affine_space ℝ A1] [affine_space ℝ B1] [affine_space ℝ C1]

-- O is the circumcenter of triangle ABC
def is_circumcenter (O : Type*) (A B C : Type*) [hA : affine_space ℝ A] [hB : affine_space ℝ B] [hC : affine_space ℝ C] : Prop :=
  sorry -- define the circumcenter condition

-- A1, B1, and C1 are symmetric to O with respect to the sides of triangle ABC
def is_symmetric (A1 B1 C1 O : Type*) (A B C : Type*) [hA1 : affine_space ℝ A1] [hB1 : affine_space ℝ B1] [hC1 : affine_space ℝ C1] [hO : affine_space ℝ O] [hA : affine_space ℝ A] [hB : affine_space ℝ B] [hC : affine_space ℝ C] : Prop :=
  sorry -- define the symmetry condition

-- O is the orthocenter of triangle A1B1C1 and the circumcenter of both triangles
theorem altitudes_through_O (O : Type*) (A B C A1 B1 C1 : Type*) 
  [hO : affine_space ℝ O] [hA : affine_space ℝ A] [hB : affine_space ℝ B] [hC : affine_space ℝ C]
  [hA1 : affine_space ℝ A1] [hB1 : affine_space ℝ B1] [hC1 : affine_space ℝ C1] 
  (h_circumcenter : is_circumcenter O A B C)
  (h_symmetric : is_symmetric A1 B1 C1 O A B C) :
  (∃ H1 H2, H1 = O ∧ H2 = O) :=
begin
  sorry -- proof required
end

end altitudes_through_O_l9_9521


namespace cost_of_largest_pot_equals_229_l9_9204

-- Define the conditions
variables (total_cost : ℝ) (num_pots : ℕ) (cost_diff : ℝ)

-- Assume given conditions
axiom h1 : num_pots = 6
axiom h2 : total_cost = 8.25
axiom h3 : cost_diff = 0.3

-- Define the function for the cost of the smallest pot and largest pot
noncomputable def smallest_pot_cost : ℝ :=
  (total_cost - (num_pots - 1) * cost_diff) / num_pots

noncomputable def largest_pot_cost : ℝ :=
  smallest_pot_cost total_cost num_pots cost_diff + (num_pots - 1) * cost_diff

-- Prove the cost of the largest pot equals 2.29
theorem cost_of_largest_pot_equals_229 (h1 : num_pots = 6) (h2 : total_cost = 8.25) (h3 : cost_diff = 0.3) :
  largest_pot_cost total_cost num_pots cost_diff = 2.29 :=
  by sorry

end cost_of_largest_pot_equals_229_l9_9204


namespace smallest_possible_value_l9_9356

theorem smallest_possible_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 / (x : ℝ)) + (1 / (y : ℝ)) = 1 / 15) : x + y = 64 :=
sorry

end smallest_possible_value_l9_9356


namespace circle_area_ratio_l9_9826

/-- If the diameter of circle R is 60% of the diameter of circle S, 
the area of circle R is 36% of the area of circle S. -/
theorem circle_area_ratio (D_S D_R A_S A_R : ℝ) (h : D_R = 0.60 * D_S) 
  (hS : A_S = Real.pi * (D_S / 2) ^ 2) (hR : A_R = Real.pi * (D_R / 2) ^ 2): 
  A_R = 0.36 * A_S := 
sorry

end circle_area_ratio_l9_9826


namespace geometric_arithmetic_sum_l9_9393

theorem geometric_arithmetic_sum :
  ∃ (r : ℝ), (a : ℕ → ℝ) (first_five_sum : ℝ),
  (∀ n, a n = 1 * r^n) ∧
  4 * (a 0) = 4 * 1 ∧
  2 * (a 1) = 2 * r ∧
  (a 2) = r^2 ∧
  (r^2 - 2 * r = 2 * r - 4 * 1) →
  first_five_sum = a 0 + a 1 + a 2 + a 3 + a 4 ∧
  first_five_sum = 31 :=
begin
  sorry
end

end geometric_arithmetic_sum_l9_9393


namespace sixty_percent_of_40_greater_than_four_fifths_of_25_l9_9427

theorem sixty_percent_of_40_greater_than_four_fifths_of_25 :
  (60 / 100 : ℝ) * 40 - (4 / 5 : ℝ) * 25 = 4 := by
  sorry

end sixty_percent_of_40_greater_than_four_fifths_of_25_l9_9427


namespace complex_power_eq_rectangular_l9_9703

noncomputable def cos_30 := Real.cos (Real.pi / 6)
noncomputable def sin_30 := Real.sin (Real.pi / 6)

theorem complex_power_eq_rectangular :
  (3 * (cos_30 + (complex.I * sin_30)))^4 = -40.5 + 40.5 * complex.I * Real.sqrt 3 := 
sorry

end complex_power_eq_rectangular_l9_9703


namespace rectangle_diagonal_length_l9_9116

theorem rectangle_diagonal_length (P L W : ℝ) (hP : P = 72) (hRatio : 5 * W = 4 * L) :
  ∃ d, d = 4 * real.sqrt 41 ∧ (d = real.sqrt (L^2 + W^2)) :=
by
  -- Define the values of k
  let k := P / (2 * (5 + 4))  -- Using the perimeter relation
  let L := 5 * k  -- Length
  let W := 4 * k  -- Width
  -- Define the diagonal
  let d := real.sqrt (L^2 + W^2)
  -- Expected answer
  existsi 4 * real.sqrt 41
  -- Demonstrate they are equal
  have : d = 4 * real.sqrt 41 := sorry
  exact ⟨rfl, this⟩

end rectangle_diagonal_length_l9_9116


namespace fixed_point_l9_9556

variable (a : ℝ)
variable (h : a > 0)
variable (h1 : a ≠ 1)

def f (x : ℝ) : ℝ := a^(x - 2) + 2

theorem fixed_point : f a 2 = 3 :=
by
  sorry

end fixed_point_l9_9556


namespace three_digit_number_division_l9_9449

theorem three_digit_number_division :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (∃ m : ℕ, 10 ≤ m ∧ m < 100 ∧ n / m = 8 ∧ n % m = 6) → n = 342 :=
by
  sorry

end three_digit_number_division_l9_9449


namespace part_a_part_b_l9_9486

variables {S : Type} {k t : ℕ}

noncomputable def number_of_mappings_a (S : finset (fin (2 * t))) : ℕ :=
  (2 * t)! / (t! * 2^t)

theorem part_a :
  ∀ (f : S → S), 
  (∀ x, f x ≠ x) ∧ (∀ x, f (f x) = x) →
  finset.card S = 2 * t → 
  number_of_mappings_a S = (2 * t)! / (t! * 2^t) :=
sorry

noncomputable def number_of_mappings_b (S : finset (fin k)) : ℕ :=
  ∑ i in finset.range (finset.card S / 2 + 1), 
    nat.choose (finset.card S) (2 * i) * (2 * i)! / (i! * 2^i)

theorem part_b :
  ∀ (f : S → S), 
  (∀ x, f (f x) = x) →
  finset.card S = k → 
  number_of_mappings_b S = 
    ∑ i in finset.range (k / 2 + 1), 
      nat.choose k (2 * i) * (2 * i)! / (i! * 2^i) :=
sorry

end part_a_part_b_l9_9486


namespace number_of_paths_l9_9810

theorem number_of_paths (E F G : Type) (n m : ℕ):
  n = 4 ∧ m = 6 ∧ combinations n 3 = 4 ∧ combinations m 3 = 20 →
  (combinations n 3) * (combinations m 3) = 80 :=
by
  intro h
  cases h with h_n h1
  cases h1 with h_m h2
  cases h2 with h_comb1 h3
  cases h3 with h_comb2 h4
  rw [h_comb1, h_comb2]
  exact h4
where
  combinations (a b : ℕ) := a! / (b! * (a - b)!)

end number_of_paths_l9_9810


namespace quiz_partition_l9_9614

theorem quiz_partition :
  ∀ (N : ℕ) (scores : fin (2 * N) → ℕ),
  (∀ i, scores i ≤ 10) ∧ 
  (∀ n ∈ {0,1,2,3,4,5,6,7,8,9,10}, ∃ i, scores i = n) → 
  (∑ i, scores i = 2 * N * 74 / 10) →
  ∃ (A B : fin (2 * N) → Prop), 
    (∀ i, A i ∨ B i) ∧ 
    (∀ i, A i ↔ ¬ B i) ∧
    (∑ i in finset.filter A (finset.univ : finset (fin (2 * N))), scores i = N * 74 / 10) ∧
    (∑ i in finset.filter B (finset.univ : finset (fin (2 * N))), scores i = N * 74 / 10) :=
begin
  sorry
end

end quiz_partition_l9_9614


namespace count_values_of_n_l9_9318

-- Define S(n) as the sum of the digits of n
def S (n : ℕ) : ℕ := (n.digits 10).sum

-- The main theorem to prove that there are 4 values of n such that n + S(n) + S(S(n)) = 2010
theorem count_values_of_n : (finset.filter (λ n, n + S(n) + S(S(n)) = 2010) (finset.range 2011)).card = 4 :=
by
  sorry

end count_values_of_n_l9_9318


namespace surface_area_of_geometric_body_l9_9573

-- Assume we have a spatial geometric body with certain views (conditions) 
def spatial_geometric_body : Type := sorry -- definition of the geometric body based on figure

-- A theorem that states, given the defined conditions, the surface area of this body is 40.
theorem surface_area_of_geometric_body (b : spatial_geometric_body) : surface_area b = 40 := 
sorry

end surface_area_of_geometric_body_l9_9573


namespace seventh_term_arithmetic_sequence_l9_9553

theorem seventh_term_arithmetic_sequence 
    (a_1 : ℚ) 
    (a_13 : ℚ) 
    (h1 : a_1 = 7 / 9) 
    (h13 : a_13 = 4 / 5) : 
    let d := (a_13 - a_1) / 12 in
    a_1 + 6 * d = 71 / 90 :=
by
  let d := (a_13 - a_1) / 12
  have h7 : a_1 + 6 * d = (a_1 + a_13) / 2 := sorry
  rw [h1, h13] at h7
  exact h7

end seventh_term_arithmetic_sequence_l9_9553


namespace S4_equals_15_l9_9864

noncomputable def S_n (q : ℝ) (n : ℕ) := (1 - q^n) / (1 - q)

theorem S4_equals_15 (q : ℝ) (n : ℕ) (h1 : S_n q 1 = 1) (h2 : S_n q 5 = 5 * S_n q 3 - 4) : 
  S_n q 4 = 15 :=
by
  sorry

end S4_equals_15_l9_9864


namespace smallest_sum_of_inverses_l9_9372

theorem smallest_sum_of_inverses 
  (x y : ℕ) (hx : x ≠ y) (h1 : 0 < x) (h2 : 0 < y) (h_condition : (1 / x : ℚ) + 1 / y = 1 / 15) :
  x + y = 64 := 
sorry

end smallest_sum_of_inverses_l9_9372


namespace S4_equals_15_l9_9863

noncomputable def S_n (q : ℝ) (n : ℕ) := (1 - q^n) / (1 - q)

theorem S4_equals_15 (q : ℝ) (n : ℕ) (h1 : S_n q 1 = 1) (h2 : S_n q 5 = 5 * S_n q 3 - 4) : 
  S_n q 4 = 15 :=
by
  sorry

end S4_equals_15_l9_9863


namespace probability_nonempty_intersection_l9_9909

theorem probability_nonempty_intersection (n : ℕ) (hn : 0 < n) : 
  (let total_outcomes := (2^n - 1)^2 in
   let favorable_outcomes := 4^n - 3^n in
   favorable_outcomes / total_outcomes = (4^n - 3^n) / (2^n - 1)^2) :=
by sorry

end probability_nonempty_intersection_l9_9909


namespace find_radius_of_tangent_circle_l9_9026

theorem find_radius_of_tangent_circle :
  (∃ (Ω : Set (ℝ × ℝ)) (r : ℝ), 
    (∀ (Γ : Set (ℝ × ℝ)), Γ = {p | ∃ (y : ℝ), p = (y^2 / 4, y)} → 
      ∃! (p : ℝ × ℝ), p ∈ Ω ∧ p ∈ Γ) ∧
    (∀ (x : ℝ), (x, 0) ∈ Ω → x = 1 ∧ 0 = 0) ∧ 
    (∀ (y : ℝ), (1 - r)^2 + (y - 1)^2 = r^2)) →
  r = 4 * Real.sqrt(3) / 9 :=
sorry

end find_radius_of_tangent_circle_l9_9026


namespace hannah_purchase_l9_9807

theorem hannah_purchase : 
  ∀ (x y z w : ℝ), 
  x + y + z + w = 24 →
  w = 3 * x →
  z = x - y →
  y + z = 24 / 5 :=
by {
  intros x y z w h1 h2 h3,
  sorry
}

end hannah_purchase_l9_9807


namespace third_competitor_hot_dogs_l9_9294

theorem third_competitor_hot_dogs (first_ate : ℕ) (second_multiplier : ℕ) (third_percent_less : ℕ) (second_ate third_ate : ℕ) 
  (H1 : first_ate = 12)
  (H2 : second_multiplier = 2)
  (H3 : third_percent_less = 25)
  (H4 : second_ate = first_ate * second_multiplier)
  (H5 : third_ate = second_ate - (third_percent_less * second_ate / 100)) : 
  third_ate = 18 := 
by 
  sorry

end third_competitor_hot_dogs_l9_9294


namespace find_f_inv_l9_9822

def f_inverse : ℝ → ℝ := λ y, (29 / 32)^(1 / 7)

theorem find_f_inv : (λ (y : ℝ), f_inverse y) (-3 / 128) = (29 / 32)^(1 / 7) :=
by
  sorry

end find_f_inv_l9_9822


namespace complex_power_rectangular_form_l9_9676

noncomputable def cos (θ : ℝ) : ℝ := sorry
noncomputable def sin (θ : ℝ) : ℝ := sorry

theorem complex_power_rectangular_form :
  (3 * complex.exp (complex.I * 30))^4 = -40.5 + 40.5 * complex.I * √3 :=
by
  sorry

end complex_power_rectangular_form_l9_9676


namespace sequence_a4_value_l9_9027

theorem sequence_a4_value : 
  ∀ (a : ℕ → ℕ), a 1 = 2 → (∀ n, n ≥ 2 → a n = a (n - 1) + n) → a 4 = 11 :=
by
  sorry

end sequence_a4_value_l9_9027


namespace smallest_positive_z_l9_9950

theorem smallest_positive_z (x z : ℝ) (hx : Real.sin x = 1) (hz : Real.sin (x + z) = -1/2) : z = 2 * Real.pi / 3 :=
by
  sorry

end smallest_positive_z_l9_9950


namespace minimum_b_value_l9_9916

noncomputable def f (x a : ℝ) : ℝ := (x - a)^2 + (Real.log (x^2 - 2 * a))^2

theorem minimum_b_value (a : ℝ) : ∃ x_0 > 0, f x_0 a ≤ (4 / 5) :=
sorry

end minimum_b_value_l9_9916


namespace value_of_m_l9_9783

theorem value_of_m (z1 z2 m : ℝ) (h1 : (Polynomial.X ^ 2 + 5 * Polynomial.X + Polynomial.C m).eval z1 = 0)
  (h2 : (Polynomial.X ^ 2 + 5 * Polynomial.X + Polynomial.C m).eval z2 = 0)
  (h3 : |z1 - z2| = 3) : m = 4 ∨ m = 17 / 2 := sorry

end value_of_m_l9_9783


namespace polynomials_sum_at_2_l9_9483

noncomputable def a (y : ℤ) : ℤ := y^4 + 20*y^2 + 8
noncomputable def b (y : ℤ) : ℤ := y^4 - 20*y^2 + 8

theorem polynomials_sum_at_2 : 
  ∀ y : ℤ, (y^8 - 32*y^4 + 64 = a(y) * b(y)) → a(2) + b(2) = 48 :=
by 
  intros y h
  have ha : a(y) = y^4 + 20*y^2 + 8 := rfl
  have hb : b(y) = y^4 - 20*y^2 + 8 := rfl
  have h_ab : y^8 - 32*y^4 + 64 = (y^4 + 20*y^2 + 8) * (y^4 - 20*y^2 + 8) := sorry
  have h2 := h.subst ha.symm.subst hb.symm.subst h_ab.symm.subst h
  exact h2.symm.trans sorry

end polynomials_sum_at_2_l9_9483


namespace fgh_supermarkets_in_us_l9_9205

theorem fgh_supermarkets_in_us : ∀ (C U : ℕ), 
  (C + U = 60) ∧ (U = C + 22) → U = 41 :=
by {
  assume C U,
  assume h : (C + U = 60) ∧ (U = C + 22),
  sorry
}

end fgh_supermarkets_in_us_l9_9205


namespace sum_of_interior_angles_of_regular_polygon_l9_9976

theorem sum_of_interior_angles_of_regular_polygon :
  (∀ (n : ℕ), (n ≠ 0) ∧ ((360 / 45 = n) → (180 * (n - 2) = 1080))) := by sorry

end sum_of_interior_angles_of_regular_polygon_l9_9976


namespace checkerboard_black_squares_l9_9664

theorem checkerboard_black_squares (n : ℕ) (hn : n = 33) :
  let black_squares : ℕ := (n * n + 1) / 2
  black_squares = 545 :=
by
  sorry

end checkerboard_black_squares_l9_9664


namespace pounds_needed_for_35_tacos_l9_9277

def pounds_of_meat_for_tacos (num_tacos : ℕ) : ℝ :=
  (4.0 / 10) * num_tacos

theorem pounds_needed_for_35_tacos :
  pounds_of_meat_for_tacos 35 = 14 := 
sorry

end pounds_needed_for_35_tacos_l9_9277


namespace count_special_three_digit_integers_l9_9423

theorem count_special_three_digit_integers : 
  let B := 8 * 9 * 9,
  let A := 7 * 8 * 8 
  in B - A = 200 := 
begin
  let B := 8 * 9 * 9,
  let A := 7 * 8 * 8,
  have h1 : B = 648, by norm_num,
  have h2 : A = 448, by norm_num,
  calc
    B - A = 648 - 448 : by rw [h1, h2]
    ... = 200 : by norm_num
end

end count_special_three_digit_integers_l9_9423


namespace find_f_of_500_l9_9908

theorem find_f_of_500
  (f : ℕ → ℕ)
  (h_pos : ∀ x y : ℕ, f x > 0 ∧ f y > 0) 
  (h_mul : ∀ x y : ℕ, f (x * y) = f x + f y) 
  (h_f10 : f 10 = 15)
  (h_f40 : f 40 = 23) :
  f 500 = 41 :=
sorry

end find_f_of_500_l9_9908


namespace max_coins_solution_l9_9585

def is_even (n : ℕ) := n % 2 = 0

def max_coins (n : ℕ) : ℕ :=
  if is_even n then n * n / 2
  else 2 * (n / 2) * (n / 2) + 2 * (n / 2) + 1

theorem max_coins_solution (n : ℕ) : 
  let f : ℕ := max_coins n in
  f = (if is_even n then n * n / 2 else 2 * (n / 2) * (n / 2) + 2 * (n / 2) + 1) := by
  sorry

end max_coins_solution_l9_9585


namespace system_solutions_l9_9306

theorem system_solutions (a b : ℝ) :
  (∃ (x y : ℝ), x^2 - 2*x + y^2 = 0 ∧ a*x + y = a*b) ↔ 0 ≤ b ∧ b ≤ 2 := 
sorry

end system_solutions_l9_9306


namespace correct_statements_l9_9339

noncomputable def f : ℝ → ℝ := sorry

axiom f_property_1 : ∀ x y : ℝ, f(x + y) + f(x - y) = 2 * f(x) * cos y
axiom f_property_2 : f 0 = 0
axiom f_property_3 : f (π / 2) = 1

theorem correct_statements :
  (f (π / 4) ≠ 1 / 2) ∧
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∃ T : ℝ, T = 2 * π ∧ ∀ x : ℝ, f (x + T) = f x) ∧
  (∀ x : ℝ, x ∈ Ioo 0 π → ¬(strict_mono_on f (Ioo 0 π))) :=
by
  sorry

end correct_statements_l9_9339


namespace part_a_part_b_l9_9739

noncomputable def f (n : ℕ) : ℕ := List.lcm (List.range (n + 1)).tail

theorem part_a (k : ℕ) :
  ∃ (a : ℕ), ∀ b, 0 ≤ b ∧ b < k → f (a + b) = f a :=
begin
  sorry
end

theorem part_b :
  ∃ (a b : ℕ), b = a + 1 ∧ f (a) < f (b) :=
begin
  sorry
end

example : ∀ (a b c : ℕ), b = a + 1 → c = b + 1 → f(a) < f(b) → ¬(f(b) < f(c)) :=
begin
  sorry
end

end part_a_part_b_l9_9739


namespace jordan_sequence_final_value_l9_9539

theorem jordan_sequence_final_value : 
  let initial_value := 10^8 in
  let value_after_steps (n : ℤ) := if n % 2 = 0 then initial_value / (2^((n + 1) / 2)) * (5 ^ (n / 2)) else 
                                   initial_value / (2^((n + 1) / 2)) * (5 ^ ((n + 1) / 2)) in
  value_after_steps 14 = 2 * 5^15 :=
by
  have h : initial_value = (2^8) * (5^8) := by norm_num
  rw h
  sorry

end jordan_sequence_final_value_l9_9539


namespace find_side_a_l9_9346

noncomputable def triangle_area (b c sinA : ℝ) : ℝ := 0.5 * b * c * sinA
def cos_angle (sinA : ℝ) : ℝ := real.sqrt (1 - sinA^2)
noncomputable def side_by_law_of_cosines (b c cosA : ℝ) : ℝ := real.sqrt (b^2 + c^2 - 2 * b * c * cosA)

theorem find_side_a (A B C : ℝ) (a b c : ℝ) (sinA cosA : ℝ)
  (h_b : b = 3) (h_c : c = 1) (h_area : triangle_area b c sinA = real.sqrt 2)
  (h_sinA : sinA = (2 * real.sqrt 2) / 3) (h_cosA: cosA = 1 / 3) :
  side_by_law_of_cosines b c cosA = 2 * real.sqrt 2 :=
by
  sorry

end find_side_a_l9_9346


namespace problem_statement_l9_9907

/-- Sequence definitions as per problem description -/
def c : ℕ → ℝ
| 0 := 3
| (n + 1) := c n + d n + 3 * real.sqrt (c n ^ 2 + d n ^ 2)

def d : ℕ → ℝ
| 0 := 2
| (n + 1) := c n + d n - 3 * real.sqrt (c n ^ 2 + d n ^ 2)

/-- Theorem stating that the desired sum of reciprocals equals 5/6 -/
theorem problem_statement : (1 / c 2012) + (1 / d 2012) = 5 / 6 :=
by sorry

end problem_statement_l9_9907


namespace digits_property_l9_9575

theorem digits_property (n : ℕ) (h : 100 ≤ n ∧ n < 1000) :
  (∃ (f : ℕ → Prop), ∀ d ∈ [n / 100, (n / 10) % 10, n % 10], f d ∧ (¬ d = 0 ∧ ¬ Nat.Prime d)) ↔ 
  (∀ d ∈ [n / 100, (n / 10) % 10, n % 10], d ∈ [1, 4, 6, 8, 9]) :=
sorry

end digits_property_l9_9575


namespace hexagon_area_l9_9042

-- Definitions based on given conditions
variables {AB BC CD DA : ℝ}
variable {x : ℝ}

-- Condition: AB = 13, BC = 6, CD = 25, DA = 8
def is_trapezoid_with_AB_par_CD (AB BC CD DA : ℝ) : Prop :=
  AB = 13 ∧ BC = 6 ∧ CD = 25 ∧ DA = 8 

-- The main theorem to prove the area of hexagon ABQCDP
theorem hexagon_area (h : is_trapezoid_with_AB_par_CD AB BC CD DA) (x : ℝ) :
  let area := 54.25 * real.sqrt 3 in
  AB * x + BC * x + (CD - AB) * x = area :=
begin
  sorry
end

end hexagon_area_l9_9042


namespace initial_ratio_milk_water_l9_9622

theorem initial_ratio_milk_water (M W : ℕ) (h1 : M + W = 165) (h2 : ∀ W', W' = W + 66 → M * 4 = 3 * W') : M / gcd M W = 3 ∧ W / gcd M W = 2 :=
by
  -- Proof here
  sorry

end initial_ratio_milk_water_l9_9622


namespace part1_part2_l9_9499

-- Definitions of the problem conditions
def f (x : ℤ) : ℤ := x * (5 * x + 2)

def D_f (n : ℕ) : ℕ := 
  -- Here would be the discriminant definition, but it cannot be directly defined without more context.
  sorry

-- Part 1: Prove that if n = 5^α or n = 2 * 5^α, then D_f(n) = n
theorem part1 (α : ℕ) : D_f (5^α) = 5^α ∧ D_f (2 * 5^α) = 2 * 5^α :=
  sorry

-- Part 2: Prove that D_f(n) is a multiple of 5
theorem part2 (n : ℕ) : ∃ k : ℕ, D_f(n) = 5 * k :=
  sorry

end part1_part2_l9_9499


namespace functional_equation_f2023_l9_9774

theorem functional_equation_f2023 (f : ℝ → ℝ) (h_add : ∀ x y : ℝ, f (x + y) = f x + f y) (h_one : f 1 = 1) :
  f 2023 = 2023 := sorry

end functional_equation_f2023_l9_9774


namespace find_matrix_M_l9_9733

theorem find_matrix_M (M : Matrix (Fin 2) (Fin 2) ℝ) (h : M^3 - 3 • M^2 + 4 • M = ![![6, 12], ![3, 6]]) :
  M = ![![2, 4], ![1, 2]] :=
sorry

end find_matrix_M_l9_9733


namespace sequence_proof_l9_9883

noncomputable def a (n : ℕ) : ℝ := 
  if h : n = 1 then 1
  else a (n - 1) + (1 / 3)^(n - 1)

theorem sequence_proof
  (n : ℕ) (hn : n ≥ 1) :
  a n = (3 / 2) * (1 - (1 / 3)^n) := 
sorry

end sequence_proof_l9_9883


namespace subsums_positive_recover_unique_l9_9249

theorem subsums_positive_recover_unique (n : ℕ) (a : Finₙ → ℝ) 
  (h : ∀ (S : Finset (Finₙ)), 0 < ∑ i in S, a i): 
  ∃! (a' : Finₙ → ℝ), (∀ (S : Finset (Finₙ)), 0 < ∑ i in S, a' i) ∧ (a' = a) := 
sorry

end subsums_positive_recover_unique_l9_9249


namespace probability_of_other_note_being_counterfeit_l9_9745

def total_notes := 20
def counterfeit_notes := 5

-- Binomial coefficient (n choose k)
noncomputable def binom (n k : ℕ) : ℚ := n.choose k

-- Probability of event A: both notes are counterfeit
noncomputable def P_A : ℚ :=
  binom counterfeit_notes 2 / binom total_notes 2

-- Probability of event B: at least one note is counterfeit
noncomputable def P_B : ℚ :=
  (binom counterfeit_notes 2 + binom counterfeit_notes 1 * binom (total_notes - counterfeit_notes) 1) / binom total_notes 2

-- Conditional probability P(A|B)
noncomputable def P_A_given_B : ℚ :=
  P_A / P_B

theorem probability_of_other_note_being_counterfeit :
  P_A_given_B = 2/17 :=
by
  sorry

end probability_of_other_note_being_counterfeit_l9_9745


namespace smallest_prime_after_six_consecutive_nonprimes_l9_9175

-- Define what it means to be prime
def is_prime (n : ℕ) : Prop :=
  nat.prime n

-- Define the sequence of six consecutive nonprime numbers
def consecutive_nonprime_sequence (a : ℕ) : Prop :=
  ¬ is_prime a ∧ ¬ is_prime (a + 1) ∧ ¬ is_prime (a + 2) ∧
  ¬ is_prime (a + 3) ∧ ¬ is_prime (a + 4) ∧ ¬ is_prime (a + 5)

-- Define the condition that there is a smallest prime number greater than a sequence of six consecutive nonprime numbers
theorem smallest_prime_after_six_consecutive_nonprimes :
  ∃ p, is_prime p ∧ p > 96 ∧ consecutive_nonprime_sequence 90 :=
sorry

end smallest_prime_after_six_consecutive_nonprimes_l9_9175


namespace liam_olivia_equal_debt_l9_9919

theorem liam_olivia_equal_debt :
  ∃ t : ℕ, 
    (200 + 200 * 0.08 * t = 300 + 300 * 0.04 * t) ∧ t = 25 :=
by 
  -- Liam's balance over time 
  let liam_balance (t : ℕ) := 200 + 16 * t
  -- Olivia's balance over time 
  let olivia_balance (t : ℕ) := 300 + 12 * t
  -- The number of days t where they owe the same amount
  have debt_equality : ∀ t, liam_balance t = olivia_balance t → t = 25,
    have eq : 200 + 16 * t = 300 + 12 * t,
    sorry
  use 25
  split
  exact debt_equality 25
  exact rfl

end liam_olivia_equal_debt_l9_9919


namespace smallest_prime_after_six_consecutive_nonprimes_l9_9178

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def nonprimes_between (a b : ℕ) : Prop :=
  ∀ n : ℕ, a < n ∧ n < b → ¬ is_prime n

def find_first_nonprime_gap : ℕ :=
  if H : ∃ a b : ℕ, b = a + 7 ∧ nonprimes_between a b ∧ is_prime (b + 1) then
    Classical.choose H
  else
    0

theorem smallest_prime_after_six_consecutive_nonprimes : find_first_nonprime_gap = 97 := by
  sorry

end smallest_prime_after_six_consecutive_nonprimes_l9_9178


namespace smallest_number_of_pieces_form_square_l9_9581

theorem smallest_number_of_pieces_form_square : 
  ∃ A, ∃ (N : ℕ), N ∈ {5, 8, 16, 20, 75} ∧ (∃ k : ℕ, N * A = k^2) ∧ (∀ M ∈ {5, 8, 16, 75}, ¬(∃ k : ℕ, M * A = k^2)) :=
begin
  sorry
end

end smallest_number_of_pieces_form_square_l9_9581


namespace coefficients_equality_l9_9002

theorem coefficients_equality (a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h : a_1 * (x-1)^4 + a_2 * (x-1)^3 + a_3 * (x-1)^2 + a_4 * (x-1) + a_5 = x^4)
  (h1 : a_1 = 1)
  (h2 : a_5 = 1)
  (h3 : 1 - a_2 + a_3 - a_4 + 1 = 0) :
  a_2 - a_3 + a_4 = 2 :=
sorry

end coefficients_equality_l9_9002


namespace intersecting_lines_c_d_sum_l9_9144

theorem intersecting_lines_c_d_sum (c d : ℝ) :
  (∀ x y, y = 2 * x + c → y = 11 → x = 3) →
  (∀ x y, y = 4 * x + d → y = 11 → x = 3) →
  c + d = 4 :=
begin
  intros h₁ h₂,
  sorry
end

end intersecting_lines_c_d_sum_l9_9144


namespace nathan_write_in_one_hour_l9_9032

/-- Jacob can write twice as fast as Nathan. Nathan wrote some letters in one hour. Together, they can write 750 letters in 10 hours. How many letters can Nathan write in one hour? -/
theorem nathan_write_in_one_hour
  (N : ℕ)  -- Assume N is the number of letters Nathan can write in one hour
  (H₁ : ∀ (J : ℕ), J = 2 * N)  -- Jacob writes twice faster, so letters written by Jacob in one hour is 2N
  (H₂ : 10 * (N + 2 * N) = 750)  -- Together they write 750 letters in 10 hours
  : N = 25 := by
  -- Proof will go here
  sorry

end nathan_write_in_one_hour_l9_9032


namespace sum_zero_l9_9039

noncomputable def a_n (n : ℕ) : ℝ := (Complex.norm (3 + Complex.i))^n * Real.cos (n * Real.arctan (1 / 3))
noncomputable def b_n (n : ℕ) : ℝ := (Complex.norm (3 + Complex.i))^n * Real.sin (n * Real.arctan (1 / 3))

theorem sum_zero : ∑' n, a_n n * b_n n / 10^n = 0 := sorry

end sum_zero_l9_9039


namespace point_placement_in_square_l9_9892

theorem point_placement_in_square : 
  ∃ (P : ℕ → ℝ × ℝ), 
    (∀i < 1600, P i.1 ∈ (Icc (0:ℝ) 1) × (Icc (0:ℝ) 1)) ∧ 
    (∀ (R : set (ℝ × ℝ)), 
      (∃ (x y : ℝ) (hx : x ∈ Icc (0:ℝ) 1) (hy : y ∈ Icc (0:ℝ) 1), 
        R = Icc x (x + 0.005) × Icc y (y + 0.005) ∧ 
        (∀ (px py : ℝ × ℝ), (px, py) ∈ R → px.1 = py.1 ∧ px.2 = py.2)) → 
      ∃i < 1600, P i ∈ R) :=
begin
  sorry
end

end point_placement_in_square_l9_9892


namespace elements_with_leading_digit_one_l9_9912

open Real

theorem elements_with_leading_digit_one :
  let T := {i : ℤ | 0 ≤ i ∧ i ≤ 1500}
  in
  (452 = intLength (2^1500)) → (count_leading_digit_one T 2 0 1500 = 1049) :=
by
  intros T h
  sorry

noncomputable def intLength (n: ℤ) : ℕ :=
  let logBase2 := log 10 (n : ℝ)
  floor logBase2 + 1

noncomputable def count_leading_digit_one (S : set ℤ) (base : ℤ) (low high : ℤ) : ℕ :=
  let log10_2 := log 10 2
  ((high - low) * log10_2) % 1 < log10_2
    ▸
 (ers count_leading_digit_one base low high)
.function_val_self

end elements_with_leading_digit_one_l9_9912


namespace hybrid_model_exists_l9_9013

-- Definitions of the conditions
def intersects_any_other_in_two_points (lines : set (ℝ × ℝ)) : Prop :=
∀ l₁ l₂ ∈ lines, l₁ ≠ l₂ → ∃ p₁ p₂ : ℝ × ℝ, p₁ ∈ l₁ ∧ p₁ ∈ l₂ ∧ p₂ ∈ l₁ ∧ p₂ ∈ l₂ ∧ p₁ ≠ p₂

def intersects_itself_in_one_point (lines : set (ℝ × ℝ)) : Prop :=
∀ l ∈ lines, ∃ p : ℝ × ℝ, p ∈ l ∧ ∀ q : ℝ × ℝ, q ∈ l → p = q

def constant_direction_principle (lines : set (ℝ × ℝ)) : Prop :=
-- Assuming an appropriate formalization that captures the essence of the principle
sorry

-- Model containing these lines
def hybrid_model (lines : set (ℝ × ℝ)) : Prop :=
intersects_any_other_in_two_points lines ∧ intersects_itself_in_one_point lines ∧ constant_direction_principle lines

-- The proof statement
theorem hybrid_model_exists : ∃ lines : set (ℝ × ℝ), hybrid_model lines :=
sorry

end hybrid_model_exists_l9_9013


namespace age_discrepancy_l9_9524

theorem age_discrepancy (R G M F A : ℕ)
  (hR : R = 12)
  (hG : G = 7 * R)
  (hM : M = G / 2)
  (hF : F = M + 5)
  (hA : A = G - 8)
  (hDiff : A - F = 10) :
  false :=
by
  -- proofs and calculations leading to contradiction go here
  sorry

end age_discrepancy_l9_9524


namespace smallest_sum_of_xy_l9_9365

theorem smallest_sum_of_xy (x y : ℕ) (h1 : x ≠ y) (h2 : 0 < x) (h3 : 0 < y) 
  (h4 : (1:ℚ)/x + (1:ℚ)/y = (1:ℚ)/15) : x + y = 64 :=
sorry

end smallest_sum_of_xy_l9_9365


namespace S₄_eq_15_l9_9841

-- Definitions based on the given conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum a

def sequence_condition (a : ℕ → ℝ) : Prop :=
  is_geometric_sequence a ∧ a 1 = 1 ∧ sum_of_first_n_terms a 5 = 5 * sum_of_first_n_terms a 3 - 4

theorem S₄_eq_15 (a : ℕ → ℝ) (q : ℝ) :
  sequence_condition a →
  (∀ n, a n = 1 * q ^ (n-1)) → 
  sum_of_first_n_terms a 4 = 15 :=
sorry

end S₄_eq_15_l9_9841


namespace smallest_x_plus_y_l9_9354

theorem smallest_x_plus_y (x y : ℕ) (h_xy_not_equal : x ≠ y) (h_condition : (1 : ℚ) / x + 1 / y = 1 / 15) :
  x + y = 64 :=
sorry

end smallest_x_plus_y_l9_9354


namespace exists_N_with_properties_p3_and_p5_no_N_with_properties_p2_and_p4_l9_9433

-- Define property p(k) where a number has the property if it can be decomposed into k consecutive natural numbers.
def hasPropertyP (n k : ℕ) : Prop :=
  ∃ (a : ℕ), (a > 1) ∧ (n = list.prod (list.iota k).map (λ x => a + x))

theorem exists_N_with_properties_p3_and_p5 :
  ∃ N, hasPropertyP N 3 ∧ hasPropertyP N 5 :=
by {
  let N := 720,
  use N,
  split,
  { unfold hasPropertyP,
    use 2,
    split,
    { simp },
    { simp }},
  { unfold hasPropertyP,
    use 8,
    split,
    { simp },
    { simp }},
  sorry
}

theorem no_N_with_properties_p2_and_p4 :
  ¬ ∃ N, hasPropertyP N 2 ∧ hasPropertyP N 4 :=
by {
  intro h,
  cases h with N HP,
  cases HP with HP2 HP4,
  sorry
}

end exists_N_with_properties_p3_and_p5_no_N_with_properties_p2_and_p4_l9_9433
