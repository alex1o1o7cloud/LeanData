import Mathlib

namespace interval_monotonically_decreasing_l680_680769

noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 + 2 * x + 3)

theorem interval_monotonically_decreasing :
  ∀ x y : ℝ, 1 < x → x < 3 → 1 < y → y < 3 → x < y → f y < f x := 
by sorry

end interval_monotonically_decreasing_l680_680769


namespace polynomial_not_factorizable_l680_680551

theorem polynomial_not_factorizable
  (n m : ℕ)
  (hnm : n > m)
  (hm1 : m > 1)
  (hn_odd : n % 2 = 1)
  (hm_odd : m % 2 = 1) :
  ¬ ∃ (g h : Polynomial ℤ), g.degree > 0 ∧ h.degree > 0 ∧ (x^n + x^m + x + 1 = g * h) :=
by
  sorry

end polynomial_not_factorizable_l680_680551


namespace count_integers_in_range_of_f_l680_680041

noncomputable def f (x : ℝ) : ℕ :=
  Int.floor x + Int.floor (Real.sqrt x) + Int.floor (Real.sqrt (Real.sqrt x))

theorem count_integers_in_range_of_f :
  (set.range f ∩ set.Icc 1 2023).card = 2071 := sorry

end count_integers_in_range_of_f_l680_680041


namespace radius_of_circle_nearest_integer_l680_680361

theorem radius_of_circle_nearest_integer (θ L : ℝ) (hθ : θ = 300) (hL : L = 2000) : 
  abs ((1200 / (Real.pi)) - 382) < 1 := 
by {
  sorry
}

end radius_of_circle_nearest_integer_l680_680361


namespace expression_value_l680_680653

theorem expression_value (x y z : ℤ) (h1 : x = 25) (h2 : y = 30) (h3 : z = 7) :
  (x - (y - z)) - ((x - y) - (z - 1)) = 13 :=
by
  sorry

end expression_value_l680_680653


namespace domain_of_function_l680_680419

def denominator (x : ℝ) : ℝ := x^3 - 4 * x

def function (x : ℝ) : ℝ := (x^4 - 4*x^3 + 6*x^2 - 4*x + 1) / denominator x

noncomputable def domain : Set ℝ := {x : ℝ | x ≠ 0 ∧ x ≠ 2 ∧ x ≠ -2}

theorem domain_of_function : ∀ x : ℝ, x ∈ domain ↔ x ∉ {0, 2, -2} := by
  intro x
  simp [domain]
  split
  ... -- sorry to skip the proofs

end domain_of_function_l680_680419


namespace second_to_last_digit_power_of_3_even_l680_680960

theorem second_to_last_digit_power_of_3_even (n : ℕ) : 
  let ⟨k, r⟩ := nat.div_mod n 4 in
  r ∈ {0, 1, 2, 3} ∧ 81 ≡ 1 [MOD 20] →
  let d := 3^n in 
  d / 10 % 10 % 2 = 0 :=
begin
  sorry
end

end second_to_last_digit_power_of_3_even_l680_680960


namespace no_knight_tour_possible_l680_680227

-- Define the structure of the chessboard
structure Chessboard (n : ℕ) where
  width : ℕ
  height : ℕ
  h_w : width = 4

-- Define the knight's move constraints and traversal properties
def KnightTourPossible (n : ℕ) : Prop :=
  ∃ tour : List (ℕ × ℕ), 
    (∀ (i : ℕ), i < n * 4 → (∃ r c, (r, c) ∈ tour)) ∧ -- Visits each square exactly once
    (∃ r c, List.head? tour = some (r, c) ∧ List.last? tour = some (r, c)) -- Returns to the starting square
  
-- The problem statement: Prove that no such knight's tour can exist
theorem no_knight_tour_possible (n : ℕ) : ¬KnightTourPossible n :=
sorry

end no_knight_tour_possible_l680_680227


namespace average_visitors_in_month_of_30_days_starting_with_sunday_l680_680324

def average_visitors_per_day (sundays_visitors : ℕ) (other_days_visitors : ℕ) (num_sundays : ℕ) (num_other_days : ℕ) : ℕ :=
  (sundays_visitors * num_sundays + other_days_visitors * num_other_days) / (num_sundays + num_other_days)

theorem average_visitors_in_month_of_30_days_starting_with_sunday :
  average_visitors_per_day 1000 700 5 25 = 750 := sorry

end average_visitors_in_month_of_30_days_starting_with_sunday_l680_680324


namespace age_relation_l680_680327

-- Given conditions
def son_age : ℕ := 16
def man_age : ℕ := son_age + 18

-- Proof problem statement
theorem age_relation (Y : ℕ) (h1 : man_age + Y = 2 * (son_age + Y)) : Y = 2 :=
by
  have son_age_eq : son_age = 16 := rfl
  have man_age_eq : man_age = son_age + 18 := rfl
  have man_age_simplified : man_age = 34 := by 
    rw [man_age_eq, son_age_eq]
    norm_num
  rw [man_age_simplified] at h1
  norm_num at h1
  exact h1.symm

end age_relation_l680_680327


namespace minimum_value_of_f_on_interval_l680_680623

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + 3

theorem minimum_value_of_f_on_interval :
  is_minimum_on_interval f (-2 : ℝ) (2 : ℝ) -37 :=
sorry

end minimum_value_of_f_on_interval_l680_680623


namespace total_shaded_area_is_2pi_l680_680613

theorem total_shaded_area_is_2pi (sm_radius large_radius : ℝ) 
  (h_sm_radius : sm_radius = 1) 
  (h_large_radius : large_radius = 2) 
  (sm_circle_area large_circle_area total_shaded_area : ℝ) 
  (h_sm_circle_area : sm_circle_area = π * sm_radius^2) 
  (h_large_circle_area : large_circle_area = π * large_radius^2) 
  (h_total_shaded_area : total_shaded_area = large_circle_area - 2 * sm_circle_area) :
  total_shaded_area = 2 * π :=
by
  -- Proof goes here
  sorry

end total_shaded_area_is_2pi_l680_680613


namespace conditional_probability_correct_l680_680234

/-- Define the total number of possible outcomes when rolling three dice -/
def total_outcomes : ℕ := 6 * 6 * 6

/-- Define the number of outcomes where no 6 appears -/
def no_six_outcomes : ℕ := 5 * 5 * 5

/-- Define event B: the number of outcomes where at least one 6 appears -/
def event_b_outcomes : ℕ := total_outcomes - no_six_outcomes

/-- Define event A: the number of outcomes where all three numbers are different
    Note: This calculation takes into account one dice being fixed at 6 and the remaining two being different. -/
def event_a_given_b_outcomes : ℕ := nat.choose 3 1 * 5 * 4

/-- Calculate the conditional probability P(A|B) -/
def conditional_probability : ℚ :=
  (event_a_given_b_outcomes : ℚ) / (event_b_outcomes : ℚ)

theorem conditional_probability_correct : conditional_probability = 60 / 91 :=
by
  sorry

end conditional_probability_correct_l680_680234


namespace range_of_a_l680_680478

open Set

variable (a : ℝ)
def A : Set ℝ := {x | abs x > 1}
def B : Set ℝ := {x | x < a}

theorem range_of_a (h : A ∪ B = A) : a ≤ -1 := 
sorry

end range_of_a_l680_680478


namespace simplify_expression_l680_680976

-- Define the problem context
variables {x y : ℝ} {i : ℂ}

-- The mathematical simplification problem
theorem simplify_expression :
  (x ^ 2 + i * y) ^ 3 * (x ^ 2 - i * y) ^ 3 = x ^ 12 - 9 * x ^ 8 * y ^ 2 - 9 * x ^ 4 * y ^ 4 - y ^ 6 :=
by {
  -- Proof steps would go here
  sorry
}

end simplify_expression_l680_680976


namespace determinant_cond_real_numbers_l680_680937

theorem determinant_cond_real_numbers (x y : ℝ) (hxy : x ≠ y)
  (hdet : matrix.det ![![2, 6, 12], ![4, x, y], ![4, y, x]] = 0) : x + y = 36 := 
sorry

end determinant_cond_real_numbers_l680_680937


namespace hyperbola_properties_l680_680837

-- Define the conditions and the final statements we need to prove
theorem hyperbola_properties (a : ℝ) (ha : a > 2) (E : ℝ → ℝ → Prop)
  (hE : ∀ x y, E x y ↔ (x^2 / a^2 - y^2 / (a^2 - 4) = 1))
  (e : ℝ) (he : e = (Real.sqrt (a^2 + (a^2 - 4))) / a) :
  (∃ E' : ℝ → ℝ → Prop,
   ∀ x y, E' x y ↔ (x^2 / 9 - y^2 / 5 = 1)) ∧
  (∃ foci line: ℝ → ℝ → Prop,
   (∀ P : ℝ × ℝ, (E P.1 P.2) →
    (∃ Q : ℝ × ℝ, (P.1 - Q.1) * (P.1 + (Real.sqrt (2*a^2-4))) = 0 ∧ Q.2=0 ∧ 
     line (P.1) (P.2) ↔ P.1 - P.2 = 2))) :=
by
  sorry

end hyperbola_properties_l680_680837


namespace exists_k_for_any_real_seq_l680_680224

theorem exists_k_for_any_real_seq (n : ℕ) (a : Fin n → ℝ) (b : Fin n → ℝ)
  (h₁ : ∀ i : Fin n, 0 ≤ b i) (h₂ : ∀ i j : Fin n, i ≤ j → b i ≥ b j) (h₃ : ∀ i : Fin n, b i ≤ 1) :
  ∃ k : Fin n, |∑ i in Finset.univ, b i * a i| ≤ |∑ i in Finset.range (k + 1), a i| :=
by
  sorry

end exists_k_for_any_real_seq_l680_680224


namespace product_greater_than_10_l680_680584

theorem product_greater_than_10 :
  (∏ k in Finset.range 50, (2 * (k + 1) : ℝ) / (2 * (k + 1) - 1)) > 10 :=
sorry

end product_greater_than_10_l680_680584


namespace exists_good_number_in_interval_l680_680031

def is_good_number (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≤ 5

theorem exists_good_number_in_interval (x : ℕ) (hx : x ≠ 0) :
  ∃ g : ℕ, is_good_number g ∧ x ≤ g ∧ g < ((9 * x) / 5) + 1 := 
sorry

end exists_good_number_in_interval_l680_680031


namespace length_of_second_train_l680_680647

/-- 
  Given:
  * Speed of train 1 is 60 km/hr.
  * Speed of train 2 is 40 km/hr.
  * Length of train 1 is 500 meters.
  * Time to cross each other is 44.99640028797697 seconds.

  Then the length of train 2 is 750 meters.
-/
theorem length_of_second_train (v1 v2 t : ℝ) (d1 L : ℝ) : 
  v1 = 60 ∧
  v2 = 40 ∧
  t = 44.99640028797697 ∧
  d1 = 500 ∧
  L = ((v1 + v2) * (1000 / 3600) * t - d1) →
  L = 750 :=
by sorry

end length_of_second_train_l680_680647


namespace range_of_a_l680_680088

  variable {A : Set ℝ} {B : Set ℝ}
  variable {a : ℝ}

  def A_def (a : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ 2 * a - 4 }
  def B_def : Set ℝ := { x | -1 < x ∧ x < 6 }

  theorem range_of_a (h : A_def a ∩ B_def = A_def a) : a < 5 :=
  sorry
  
end range_of_a_l680_680088


namespace length_EF_of_triangle_def_l680_680344

theorem length_EF_of_triangle_def (a b : ℝ) :
  let parabola := λ x, x^2 - 4*x + 4,
      D := (2, 0),
      E := (2 - a, parabola(2 - a)),
      F := (2 + a, parabola(2 + a)),
      EF := (2*a),
      area_ΔDEF := (1/2) * EF * (parabola(2 - a) - 0),
      length_EF := 8 in
  a * (a^2 - 4*a + 4) = 32 → EF = 8 := 
by { intros, sorry }

end length_EF_of_triangle_def_l680_680344


namespace log_arith_expression_l680_680654

theorem log_arith_expression : (log 2 128 / log 64 2 - log 2 256 / log 16 2) = 10 :=
by
  -- Define the logarithms using the base-2 logarithm and change of base formula
  have eq1 : log 64 2 = 1 / log 2 64 := by sorry
  have eq2 : log 16 2 = 1 / log 2 16 := by sorry

  -- Substitute the logarithms to simplify the main expression
  have log_2_128 : log 2 128 = 7 := by sorry
  have log_2_64 : log 2 64 = 6 := by sorry
  have log_2_256 : log 2 256 = 8 := by sorry
  have log_2_16 : log 2 16 = 4 := by sorry

  -- Compute the final expression
  calc
    (log 2 128 / log 64 2 - log 2 256 / log 16 2)
      = (log 2 128 * log 2 64 - log 2 256 * log 2 16) : by
        rw [eq1, eq2]
      = (7 * 6 - 8 * 4) : by
        rw [log_2_128, log_2_64, log_2_256, log_2_16]
      = 42 - 32 : by
        norm_num
      = 10 : by
        norm_num

end log_arith_expression_l680_680654


namespace perpendicular_tangents_and_point_l680_680578

theorem perpendicular_tangents_and_point :
  ∃ (x₀ y₀ : ℝ),
    (3 * x₀ + 4 * y₀ = 2) ∧
    (y₀ = x₀^2) ∧
    (x₀ = 1) ∧
    (y₀ = -1/4) ∧
    ∃ (k1 k2 : ℝ),
      (k1 = 2 + Real.sqrt 5) ∧
      (k2 = 2 - Real.sqrt 5) ∧
      (y₀ + k1 * (x - x₀) = -1/4 + (2 + Real.sqrt 5) * (x - 1)) ∧
      (y₀ + k2 * (x - x₀) = -1/4 + (2 - Real.sqrt 5) * (x - 1)) ∧
      (k1 * k2 = -1) :=
begin
  existsi (1 : ℝ),
  existsi (-1 / 4 : ℝ),
  split,
  { linarith },
  split,
  { norm_num, ring },
  split,
  { refl },
  split,
  { norm_num, ring },
  existsi (2 + Real.sqrt 5 : ℝ),
  existsi (2 - Real.sqrt 5 : ℝ),
  split,
  { refl },
  split,
  { refl },
  split,
  { sorry },
  split,
  { sorry },
  { sorry }
end

end perpendicular_tangents_and_point_l680_680578


namespace find_number_of_breeding_rabbits_l680_680569

def breeding_rabbits_condition (B : ℕ) : Prop :=
  ∃ (kittens_first_spring remaining_kittens_first_spring kittens_second_spring remaining_kittens_second_spring : ℕ),
    kittens_first_spring = 10 * B ∧
    remaining_kittens_first_spring = 5 * B + 5 ∧
    kittens_second_spring = 60 ∧
    remaining_kittens_second_spring = kittens_second_spring - 4 ∧
    B + remaining_kittens_first_spring + remaining_kittens_second_spring = 121

theorem find_number_of_breeding_rabbits (B : ℕ) : breeding_rabbits_condition B → B = 10 :=
by
  sorry

end find_number_of_breeding_rabbits_l680_680569


namespace find_derivative_at_e_l680_680131

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem find_derivative_at_e :
  let f' (x : ℝ) : ℝ := deriv f x,
      e_val := Real.exp 1 
    in f' e_val = 2 := by
  sorry

end find_derivative_at_e_l680_680131


namespace calculate_expression_l680_680045

theorem calculate_expression : ∀ x y : ℝ, x = 7 → y = 3 → (x - y) ^ 2 * (x + y) = 160 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end calculate_expression_l680_680045


namespace increasing_sequences_count_with_modulo_l680_680741

theorem increasing_sequences_count_with_modulo : 
  let n := 12
  let m := 1007
  let k := 508
  let mod_value := 1000
  let sequences_count := Nat.choose (497 + n - 1) n
  sequences_count % mod_value = k :=
by
  let n := 12
  let m := 1007
  let k := 508
  let mod_value := 1000
  let sequences_count := Nat.choose (497 + n - 1) n
  sorry

end increasing_sequences_count_with_modulo_l680_680741


namespace work_rate_combined_l680_680299

theorem work_rate_combined (a b c : ℝ) (ha : a = 21) (hb : b = 6) (hc : c = 12) :
  (1 / ((1 / a) + (1 / b) + (1 / c))) = 84 / 25 := by
  sorry

end work_rate_combined_l680_680299


namespace find_x_l680_680517

theorem find_x (x y z : ℕ) (h_pos : 0 < x) (h_pos : 0 < y) (h_pos : 0 < z) (h_eq1 : x + y + z = 37) (h_eq2 : 5 * y = 6 * z) : x = 21 :=
sorry

end find_x_l680_680517


namespace height_of_stack_is_15_l680_680742

def total_height_stack : ℝ :=
  let thickness := 0.5
  let top_diameter := 15.5
  let bottom_diameter := 1
  let decrease_per_plate := 0.5
  let n := ((top_diameter - bottom_diameter) / decrease_per_plate) + 1
  n * thickness

theorem height_of_stack_is_15 :
  total_height_stack = 15 := by
  -- proof goes here
  sorry

end height_of_stack_is_15_l680_680742


namespace probability_black_pen_l680_680302

section ProbabilityPens

variables (green_pens black_pens red_pens : ℕ)
variable  (total_pens : ℕ := green_pens + black_pens + red_pens)

theorem probability_black_pen (h1 : green_pens = 5) (h2 : black_pens = 6) (h3 : red_pens = 7) : 
  ((black_pens : ℚ) / (total_pens : ℚ)) = 1 / 3 := 
by 
  unfold total_pens 
  rw [h1, h2, h3] 
  norm_num
  sorry

end ProbabilityPens 

end probability_black_pen_l680_680302


namespace red_and_white_flowers_l680_680004

theorem red_and_white_flowers 
  (total_flowers : ℕ)
  (yellow_and_white_flowers : ℕ)
  (red_and_yellow_flowers : ℕ)
  (R : ℕ)
  (extra_red_flowers : ℕ) 
  (H1 : total_flowers = 44)
  (H2 : yellow_and_white_flowers = 13)
  (H3 : red_and_yellow_flowers = 17)
  (H4 : extra_red_flowers = 4)
  : R = 14 := 
by 
  unfold total_flowers at H1
  unfold yellow_and_white_flowers at H2
  unfold red_and_yellow_flowers at H3
  unfold extra_red_flowers at H4
  
  sorry

end red_and_white_flowers_l680_680004


namespace advertisement_cost_l680_680049

def numberOfAds : ℕ := 5
def durationPerAd : ℕ := 3
def costPerMinute : ℕ := 4000
def totalDuration : ℕ := numberOfAds * durationPerAd := by
  exact 15
def totalCost : ℕ := totalDuration * costPerMinute := by
  exact 60000

theorem advertisement_cost : 
  totalCost = 60000 := by
  sorry

end advertisement_cost_l680_680049


namespace similar_triangles_are_similar_l680_680295

theorem similar_triangles_are_similar (A B C D : Prop)
  (A_def : ∀ Δ1 Δ2 : Triangle, Δ1.has_right_angle → Δ2.has_right_angle → Δ1 ~ Δ2 = false)
  (B_def : ∀ r1 r2 : Rectangle, r1 ~ r2 = false)
  (C_def : ∀ ρ1 ρ2 : Rhombus, ρ1 ~ ρ2 = false)
  (D_def : ∀ t1 t2 : Triangle, t1 ~ t2 → t1 ~ t2) :
  D := by
  sorry

end similar_triangles_are_similar_l680_680295


namespace steve_correct_operations_l680_680979

theorem steve_correct_operations (x : ℕ) (h1 : x / 8 - 20 = 12) : ((x * 8) + 20) = 2068 :=
by
  sorry

end steve_correct_operations_l680_680979


namespace inverse_function_of_logb_l680_680991

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 (x + 1)

theorem inverse_function_of_logb :
  ∀ y : ℝ, y ≥ 0 → (f (2 ^ y - 1) = y) :=
begin
  sorry
end

end inverse_function_of_logb_l680_680991


namespace number_divisible_by_8_l680_680357

theorem number_divisible_by_8 : (5 * 1000000 + 3416) % 8 = 0 :=
by
  have h : 3416 % 8 = 0 := by sorry
  have h2 : (5 * 1000000) % 8 = 0 := by sorry
  have h3 : ((5 * 1000000 + 3416) % 8) = ((5 * 1000000) % 8 + 3416 % 8) % 8 := by sorry
  rw [h, h2, h3]
  exact h

end number_divisible_by_8_l680_680357


namespace volume_of_cut_cube_l680_680386

open Real

-- Define the problem conditions in Lean
def is_midpoint (P Q R : Prop) (x y z : ℝ) : Prop := 
  R = (P + Q) / 2

def volume_of_larger_solid (cube_edge : ℝ) : ℝ :=
  let V := cube_edge^3 in
  V - ((1/6) - (1/48))

theorem volume_of_cut_cube :
  ∀ (D M N : Prop) (edge_length : ℝ),
  edge_length = 1 ∧
  D = (0, 0, 0) ∧
  M = (1/2, 1, 0) ∧
  N = (1, 0, 1/2) →
  volume_of_larger_solid edge_length = 41 / 48 :=
by 
  sorry

end volume_of_cut_cube_l680_680386


namespace num_positive_integers_with_digit_sum_two_l680_680669

-- Define the range of numbers
def inRange (n : ℕ) : Prop := 10^7 ≤ n ∧ n < 10^8

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- State the theorem to be proved
theorem num_positive_integers_with_digit_sum_two :
  {n : ℕ | inRange n ∧ sum_of_digits n = 2}.to_finset.card = 28 :=
sorry

end num_positive_integers_with_digit_sum_two_l680_680669


namespace intersection_P_Q_l680_680205

def P (x : ℝ) : Prop := x^2 - x - 2 ≥ 0

def Q (y : ℝ) : Prop := ∃ x, P x ∧ y = (1/2) * x^2 - 1

theorem intersection_P_Q :
  {m | ∃ (x : ℝ), P x ∧ m = (1/2) * x^2 - 1} = {m | m ≥ 2} := sorry

end intersection_P_Q_l680_680205


namespace find_fx_l680_680510

theorem find_fx (f : ℝ → ℝ) 
  (h1 : ∀ x > 0, f (-x) = -(2 * x - 3)) 
  (h2 : ∀ x < 0, -f x = f (-x)) :
  ∀ x < 0, f x = 2 * x + 3 :=
by
  sorry

end find_fx_l680_680510


namespace probability_inside_circle_l680_680801
noncomputable def probability_y_ge_x_circle : ℝ :=
  let total_area := Real.pi in
  let semicircle_area := 0.5 * Real.pi in
  let additional_area := 0.25 in
  let region_area := semicircle_area + additional_area in
  region_area / total_area

theorem probability_inside_circle :
  ∀ (P : ℝ × ℝ), (P.1 ^ 2 + (P.2 - 1) ^ 2 = 1) → 
  (probability_y_ge_x_circle = 3 / 4 + 1 / (2 * Real.pi)) :=
sorry

end probability_inside_circle_l680_680801


namespace tan_sum_formula_l680_680125

theorem tan_sum_formula (α : ℝ) (h : tan α = 2) : tan (α + π/4) = -3 :=
by
  sorry

end tan_sum_formula_l680_680125


namespace smallest_k_for_720_l680_680076

/-- Given a number 720, prove that the smallest positive integer k such that 720 * k is both a perfect square and a perfect cube is 1012500. -/
theorem smallest_k_for_720 (k : ℕ) : (∃ k > 0, 720 * k = (n : ℕ) ^ 6) -> k = 1012500 :=
by sorry

end smallest_k_for_720_l680_680076


namespace find_y_coordinate_of_C_l680_680812

def point (x : ℝ) (y : ℝ) : Prop := y^2 = x + 4

def perp_slope (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  (y2 - y1) / (x2 - x1) * (y3 - y2) / (x3 - x2) = -1

def valid_y_coordinate_C (x0 : ℝ) : Prop :=
  x0 ≤ 0 ∨ 4 ≤ x0

theorem find_y_coordinate_of_C (x0 : ℝ) :
  (∀ (x y : ℝ), point x y) →
  (∃ (x2 y2 x3 y3 : ℝ), point x2 y2 ∧ point x3 y3 ∧ perp_slope 0 2 x2 y2 x3 y3) →
  valid_y_coordinate_C x0 :=
sorry

end find_y_coordinate_of_C_l680_680812


namespace find_AC_l680_680518

-- Given data:
variables {A B C D F : Type}
variable (dist : A → A → ℝ)
variables [noncomputable_field ℝ] [metric_space ℝ]
variables (A B C D F : ℝ)

-- Distances given and relationships
variables (AB_perp_AC : ∀ A B C, ∃ θ, θ = π / 2)
variables (AF_perp_BC : ∀ A F B, ∃ θ, θ = π / 2)
variables (BD_eq_1 : ∀ B D, dist B D = 1) 
variables (DC_eq_1 : ∀ D C, dist D C = 1)
variables (FC_eq_1 : ∀ F C, dist F C = 1)

-- Definitions for halfway point
def midpoint (x y : ℝ) : ℝ := (x + y) / 2
axiom D_is_midpoint : ∀ A D C, D = midpoint A C

-- The main theorem
theorem find_AC : dist A C = real.cbrt 2 :=
by
  -- Placeholders for the proof
  sorry

end find_AC_l680_680518


namespace part_I_part_II_max_at_π_12_l680_680128

def f (x : ℝ) : ℝ := cos (x - π / 6) ^ 2 - sin x ^ 2

theorem part_I : f (π / 12) = sqrt 3 / 2 :=
by sorry

theorem part_II : ∀ x, 0 ≤ x ∧ x ≤ π / 2 → f x ≤ sqrt 3 / 2 :=
by sorry

theorem max_at_π_12 : f (π / 12) = sqrt 3 / 2 :=
by sorry

end part_I_part_II_max_at_π_12_l680_680128


namespace percentage_difference_between_chef_and_dishwasher_l680_680014

theorem percentage_difference_between_chef_and_dishwasher
    (manager_wage : ℝ)
    (dishwasher_wage : ℝ)
    (chef_wage : ℝ)
    (h1 : manager_wage = 6.50)
    (h2 : dishwasher_wage = manager_wage / 2)
    (h3 : chef_wage = manager_wage - 2.60) :
    (chef_wage - dishwasher_wage) / dishwasher_wage * 100 = 20 :=
by
  -- The proof would go here
  sorry

end percentage_difference_between_chef_and_dishwasher_l680_680014


namespace find_p_l680_680146

-- Definitions
variables {n : ℕ} {p : ℝ}
def X : Type := ℕ -- Assume X is ℕ-valued

-- Conditions
axiom binomial_expectation : n * p = 6
axiom binomial_variance : n * p * (1 - p) = 3

-- Question to prove
theorem find_p : p = 1 / 2 :=
by
  sorry

end find_p_l680_680146


namespace find_angles_l680_680170

noncomputable def isosceles_triangle_angles (α β γ : ℝ) : Prop :=
  ∃ (ABC : Triangle), 
    ABC.is_isosceles ∧
    (∃ D E : Point, 
      D ∈ LineSegment ABC.A ABC.B ∧
      E ∈ LineSegment ABC.C ABC.B ∧
      Altitude ABC.A D ∧ 
      Altitude ABC.C E ∧
      Angle ABC.ABC.AM C = 48)

theorem find_angles 
  (α β γ : ℝ) 
  (h : isosceles_triangle_angles α β γ) : 
  α = 24 ∧ β = 24 ∧ γ = 132 := 
sorry

end find_angles_l680_680170


namespace equal_expression_value_l680_680372

theorem equal_expression_value :
  abs(abs(abs(-2 + 3) - 2) + 3) = 6 := 
by
  sorry

end equal_expression_value_l680_680372


namespace find_possible_m_values_l680_680433

theorem find_possible_m_values :
  ∃ (m_values : Set ℤ), 
    (∀ m ∈ m_values, ∃ α : ℤ, 
      (α = (m + 1) / 3 ∧ -1988 ≤ α ∧ α ≤ 1988 ∧ (3 * α) % 5 = 0)) ∧ 
    m_values.card = 2385 := 
sorry

end find_possible_m_values_l680_680433


namespace certain_percentage_of_50_l680_680316

theorem certain_percentage_of_50 (x : ℝ) 
    (h : (50 * x / 100) + (860 * 50 / 100) = 860) : 
    x = 860 := 
begin
  sorry
end

end certain_percentage_of_50_l680_680316


namespace angle_sum_less_than_ninety_l680_680525

noncomputable def acute_triangle (A B C : Type) [euclidean_geometry A B C] := 
  ∃ (O : circumcenter A B C), is_acute_triangle A B C

theorem angle_sum_less_than_ninety (A B C : Type) 
  [euclidean_geometry A B C] 
  [acute_triangle A B C] 
  (P : altitude A B C) 
  (O : circumcenter A B C) 
  (h : ∠ BCA ≥ ∠ ABC + 30) : 
  ∠ CAB + ∠ COP < 90 := 
sorry

end angle_sum_less_than_ninety_l680_680525


namespace correct_polynomial_result_l680_680710

theorem correct_polynomial_result (x : ℝ) :
  let initial_result := x^2 - 2 * x + 1 in
  let corrected_polynomial := initial_result + 3 * x^2 in
  let final_result := corrected_polynomial * (-3 * x^2) in
  final_result = -12 * x^4 + 6 * x^3 - 3 * x^2 :=
by
  sorry

end correct_polynomial_result_l680_680710


namespace circle_equation_l680_680440

variables {R : Type*} [Real R]

-- Define the conditions of the problem
def tangent_condition_1 (x y : R) : Prop := x - y = 0
def tangent_condition_2 (x y : R) : Prop := x - y + 4 = 0
def center_condition (x y : R) : Prop := y = -x + 2

-- Define the standard form equation of the circle
def circle_eq (x y : R) : Prop := x^2 + (y - 2)^2 = 2

-- Main theorem statement
theorem circle_equation (a : R) :
  center_condition a (2-a) → (tangent_condition_1 a (2-a) ∨ tangent_condition_2 a (2-a)) → circle_eq a (2-a) :=
by
  sorry

end circle_equation_l680_680440


namespace correct_share_jpy_correct_share_jpy_method_2_correct_change_in_share_jpy_l680_680389

namespace NWF

def total_nwf_funds : ℝ := 1213.76

def subtracted_amounts : ℝ := 3.36 + 38.4 + 4.25 + 226.6 + 340.56 + 0.29

def previous_share_jpy : ℝ := 72.98

def other_currency_shares : ℝ := 0.28 + 3.16 + 0.35 + 18.67 + 28.06 + 0.02

def current_share_jpy_method_1 : ℝ := (total_nwf_funds - subtracted_amounts) / total_nwf_funds * 100

def current_share_jpy_method_2 : ℝ := 100 - other_currency_shares

def change_in_share_jpy : ℝ := current_share_jpy_method_1 - previous_share_jpy

theorem correct_share_jpy : current_share_jpy_method_1 = 49.46 := 
by sorry

theorem correct_share_jpy_method_2 : current_share_jpy_method_2 = 49.46 := 
by sorry

theorem correct_change_in_share_jpy : change_in_share_jpy = -23.5 := 
by sorry

end NWF

end correct_share_jpy_correct_share_jpy_method_2_correct_change_in_share_jpy_l680_680389


namespace min_a_3b_l680_680926

theorem min_a_3b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (1 / (a + 3) + 1 / (b + 3) = 1 / 4)) : 
  a + 3*b ≥ 12 + 16*Real.sqrt 3 :=
by sorry

end min_a_3b_l680_680926


namespace range_of_a_l680_680435

variables (a x : ℝ)

def A := {x : ℝ | abs (3 * x - 4) > 2}
def B := {x : ℝ | x^2 - x - 2 > 0}
def C := {x : ℝ | (x - a) * (x - a - 1) ≥ 0}

def p := x ∈ ((C a) ∪ A)
def q := x ∈ ((C a) ∪ B)
def r := x ∈ (C a)

theorem range_of_a (h : ∀ x, r x → p x ∧ ¬(p x → r x)) : a ≥ 2 ∨ a ≤ -1 / 3 :=
sorry

end range_of_a_l680_680435


namespace f_finite_and_max_value_l680_680749

open BigOperators

-- We define the function f as per the problem description

noncomputable def A (n : ℕ) : ℕ := sorry -- placeholder for A(n)
noncomputable def B (n : ℕ) : ℕ := sorry -- placeholder for B(n)

def f (n : ℕ) : ℕ :=
  if B(n) = 1 then 1 else sorry -- largest prime factor of B(n)

-- We now state the theorem as required
theorem f_finite_and_max_value :
  {k : ℕ | ∃ n : ℕ, f n = k}.finite ∧ ∃ n : ℕ, f n = 2003 :=
sorry

end f_finite_and_max_value_l680_680749


namespace people_to_seat_around_table_l680_680887

theorem people_to_seat_around_table 
    (n : ℕ) 
    (h : fact (nat.factorial (n - 1) = 144)) : 
    n = 6 :=
sorry

end people_to_seat_around_table_l680_680887


namespace polynomial_properties_l680_680606

noncomputable def q (x : ℝ) : ℝ := x^3 - 142/13 * x^2 + 529/13 * x - 90

theorem polynomial_properties :
  (monic (λ x, x^3 - 142/13 * x^2 + 529/13 * x - 90)) ∧
  (is_real_poly (λ x, x^3 - 142/13 * x^2 + 529/13 * x - 90)) ∧ 
  (eval (2 + 3 * complex.I) (λ x, x^3 - 142/13 * x^2 + 529/13 * x - 90) = 0) ∧
  (eval 0 (λ x, x^3 - 142/13 * x^2 + 529/13 * x - 90) = -90):=
by
  -- Proof will be inserted here
  sorry 

end polynomial_properties_l680_680606


namespace total_round_trip_distance_is_correct_l680_680222

def distance_A_to_C : ℝ := 4000
def distance_A_to_B : ℝ := 4500

def distance_B_to_C : ℝ := real.sqrt (distance_A_to_B ^ 2 - distance_A_to_C ^ 2)

def total_distance : ℝ := distance_A_to_B + distance_B_to_C + distance_A_to_C

theorem total_round_trip_distance_is_correct :
    total_distance = 10562 := by
  -- Definitions and calculations as per the proof
  let CA := distance_A_to_C
  let AB := distance_A_to_B
  let BC := distance_B_to_C
  have h1 : CA = 4000 := rfl
  have h2 : AB = 4500 := rfl
  have h3 : BC = real.sqrt (AB ^ 2 - CA ^ 2) := rfl
  have h4 : total_distance = AB + BC + CA := rfl
  calc
    total_distance
        = AB + BC + CA : by rw [h4]
    ... = 4500 + real.sqrt (4500 ^ 2 - 4000 ^ 2) + 4000 : by rw [h2, h3, h1]
    ... = 4500 + 2062 + 4000 : by sorry -- continue calculation steps
    ... = 10562 : by sorry -- continue calculation steps

end total_round_trip_distance_is_correct_l680_680222


namespace floor_sqrt_120_l680_680399

theorem floor_sqrt_120 : (Real.floor (Real.sqrt 120)) = 10 :=
by
  have a := 10
  have b := 11
  have h1 : a < b := by norm_num
  have h2 : a^2 < 120 := by norm_num
  have h3 : 120 < b^2 := by norm_num
  sorry

end floor_sqrt_120_l680_680399


namespace find_k_value_l680_680841

theorem find_k_value (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 4 * x + 4 = 0) ∧
  (∀ x1 x2 : ℝ, (k * x1^2 + 4 * x1 + 4 = 0 ∧ k * x2^2 + 4 * x2 + 4 = 0) → x1 = x2) →
  (k = 0 ∨ k = 1) :=
by
  intros h
  sorry

end find_k_value_l680_680841


namespace probability_of_3_black_2_white_l680_680689

def total_balls := 15
def black_balls := 10
def white_balls := 5
def drawn_balls := 5
def drawn_black_balls := 3
def drawn_white_balls := 2

noncomputable def probability_black_white_draw : ℝ :=
  (Nat.choose black_balls drawn_black_balls * Nat.choose white_balls drawn_white_balls : ℝ) /
  (Nat.choose total_balls drawn_balls : ℝ)

theorem probability_of_3_black_2_white :
  probability_black_white_draw = 400 / 1001 := by
  sorry

end probability_of_3_black_2_white_l680_680689


namespace sin_product_identity_l680_680380

theorem sin_product_identity : 
  let sin_60 := (sqrt 3) / 2
  let cos_18 := (sqrt 5 + 1) / 4
  let sin_72 := cos_18
  let cos_42_18 := sin 66
  let sin_12_cos_24 := sin 36 / 2
  (sin 12 * sin 48 * sin_60 * sin_72 = (sqrt 5 + 1) * sqrt 3 / 16) :=
by
  sorry

end sin_product_identity_l680_680380


namespace largest_n_polynomial_equivalence_l680_680920

theorem largest_n_polynomial_equivalence 
  (α : ℝ) 
  (h_α : is_root (λ x : ℝ, x^6 - x - 1) α) 
  (n : ℕ) 
  (h_equiv : ∀ (p q : polynomial ℤ), p.eval α % 3 = q.eval α % 3 ↔ p ≡ q [MOD 3]) 
  (unique_equiv : ∀ (p : polynomial ℤ), ∃! k : ℕ, k < 728 ∧ p % (x^6 - x - 1) ≡ x^k [MOD 3]) :
  ∃ p : polynomial ℤ, p^3 - p ≡ x^727 [MOD 3] := 
sorry

end largest_n_polynomial_equivalence_l680_680920


namespace distance_to_origin_is_2_l680_680219

theorem distance_to_origin_is_2 (a : ℝ) (h : |a| = 2) : a - 2 = 0 ∨ a - 2 = -4 :=
by {
  have h1 : a = 2 ∨ a = -2,
  from abs_eq 2 a,
  cases h1 with ha ha,
  { left, rw ha, norm_num },
  { right, rw ha, norm_num }
}

end distance_to_origin_is_2_l680_680219


namespace prime_one_digits_l680_680225

theorem prime_one_digits (p : ℕ) (hp : p.prime) (h2 : p ≠ 2) (h5 : p ≠ 5) : ∃ k : ℕ, ∀ d : ℕ, d > 0 → d < pk.digits.size → pk.digits.get d = 1 :=
by
  sorry

end prime_one_digits_l680_680225


namespace polar_to_rect_eq_circle_l680_680262

-- Definitions of the conditions
def polar_to_rect (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- The theorem to state the proof problem
theorem polar_to_rect_eq_circle (ρ θ : ℝ) (h : ρ = 4 * Real.sin θ) :
  (polar_to_rect ρ θ).fst^2 + ((polar_to_rect ρ θ).snd - 2)^2 = 4 :=
by
  sorry

end polar_to_rect_eq_circle_l680_680262


namespace base9_to_base3_l680_680037

theorem base9_to_base3 (n : List ℕ) (h9 : n = [7, 2, 5, 4]) : 
  (9 * (9 * (9 * 7 + 2) + 5) + 4).digits 3 = [2, 1, 0, 2, 1, 2, 1, 1, 3] :=
by {
  have h7 : 7.digits 3 = [2, 1] := rfl,
  have h2 : 2.digits 3 = [0, 2] := rfl,
  have h5 : 5.digits 3 = [1, 2] := rfl,
  have h4 : 4.digits 3 = [1, 1] := rfl,
  sorry
}

end base9_to_base3_l680_680037


namespace size_of_barrel_l680_680489

theorem size_of_barrel (barrel_servings : ℕ) (serving_cheese_balls : ℕ) (total_cheese_balls : ℕ) (size_24oz_servings : ℕ) :
  size_24oz_servings = 60 → barrel_servings = 87.5 → serving_cheese_balls = 12 → total_cheese_balls = 1050 → 24 * total_cheese_balls / (size_24oz_servings * serving_cheese_balls) = 35 :=
by
  sorry

end size_of_barrel_l680_680489


namespace cone_volume_l680_680390

theorem cone_volume (d h : ℝ) (V : ℝ) (hd : d = 12) (hh : h = 8) :
  V = (1 / 3) * Real.pi * (d / 2) ^ 2 * h → V = 96 * Real.pi :=
by
  rw [hd, hh]
  sorry

end cone_volume_l680_680390


namespace factorize_expr_l680_680761

-- Define the expression
def expr (a : ℝ) := -3 * a + 12 * a^2 - 12 * a^3

-- State the theorem
theorem factorize_expr (a : ℝ) : expr a = -3 * a * (1 - 2 * a)^2 :=
by
  sorry

end factorize_expr_l680_680761


namespace Alice_met_lion_on_monday_l680_680012

def day := ℕ  -- let's represent days as natural numbers for simplicity
def monday := 0
def tuesday := 1
def wednesday := 2
def thursday := 3
def friday := 4
def saturday := 5
def sunday := 6
def week_length := 7

def lies_on (d : day) : Prop := 
  d % week_length = thursday ∨ d % week_length = friday

def yesterday (d : day) : day := (d + week_length - 1) % week_length
def tomorrow (d : day) : day := (d + 1) % week_length
def day_after_tomorrow (d : day) : day := (d + 2) % week_length

-- Condition 1: "I lied yesterday."
def condition1 (today : day) : Prop :=
  lies_on (yesterday today)

-- Condition 2: "The day after tomorrow, I will lie for two consecutive days."
def condition2 (today : day) : Prop :=
  lies_on (day_after_tomorrow today) ∧ lies_on (tomorrow (day_after_tomorrow today))

-- Theorem to prove
theorem Alice_met_lion_on_monday (today : day) :
  condition1 today → condition2 today → today = monday :=
sorry

end Alice_met_lion_on_monday_l680_680012


namespace sum_of_squares_of_roots_l680_680777

theorem sum_of_squares_of_roots :
  ∀ (r₁ r₂ : ℝ), (r₁ + r₂ = 15) → (r₁ * r₂ = 6) → (r₁^2 + r₂^2 = 213) :=
by
  intros r₁ r₂ h_sum h_prod
  -- Proof goes here, but skipping it for now
  sorry

end sum_of_squares_of_roots_l680_680777


namespace cos_A_range_l680_680181

noncomputable def range_cos_A (BC : ℝ) (hBC : BC > real.sqrt 2) : set ℝ :=
{ x | -1 < x ∧ x < 11 / 12 }

theorem cos_A_range (A B C : Type) [Euclidean_triangle A B C] (AB AC BC : ℝ)
  (hAB : AB = 3) (hAC : AC = 2) (hBC : BC > real.sqrt 2) :
  cos_angle AB AC BC ∈ range_cos_A BC hBC :=
sorry

end cos_A_range_l680_680181


namespace real_solutions_l680_680417

theorem real_solutions (x : ℝ)
  (h : x^3 + (3 - x)^3 = 18) :
  x = 1.5 + (real.sqrt 5 / 2) ∨ x = 1.5 - (real.sqrt 5 / 2) :=
sorry

end real_solutions_l680_680417


namespace railway_platform_length_l680_680713

theorem railway_platform_length : 
  ∀ (train_speed_kmh : ℕ) (train_length_m : ℕ) (crossing_time_s : ℝ), 
  train_speed_kmh = 132 →
  train_length_m = 110 →
  crossing_time_s = 7.499400047996161 →
  let train_speed_mps := (132 * 1000) / 3600 in
  let distance_crossed := train_speed_mps * crossing_time_s in
    distance_crossed - train_length_m ≈ 165 :=
  sorry

end railway_platform_length_l680_680713


namespace monotonic_intervals_number_of_zeros_l680_680467

noncomputable def f (a x : ℝ) := (1 / 2 * x^2 - a * x) * (Real.log x) + 2 * a * x - 3 / 4 * x^2

theorem monotonic_intervals (a : ℝ) (ha : 0 < a) (hae : a < Real.exp 1) : 
  ((∀ x, 0 < x → x < a → (deriv (f a) x) > 0) ∧ 
   (∀ x, 0 < x → a < x → x < Real.exp 1 → (deriv (f a) x) < 0) ∧ 
   (∀ x, Real.exp 1 < x → (deriv (f a) x) > 0)) := sorry

theorem number_of_zeros (a : ℝ) (ha : 0 < a) (hae : a < Real.exp 1) :
  (if 0 < a ∧ a < (Real.exp 1) / 4 then
     ∃ x1 x2, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0 
   else if a = (Real.exp 1) / 4 then
     ∃! x, f a x = 0 
   else if (Real.exp 1) / 4 < a ∧ a < Real.exp 1 then
     ∀ x, f a x ≠ 0) := sorry

end monotonic_intervals_number_of_zeros_l680_680467


namespace binary_representation_of_fourteen_l680_680415

theorem binary_representation_of_fourteen :
  (14 : ℕ) = 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 0 * 2^0 :=
by
  sorry

end binary_representation_of_fourteen_l680_680415


namespace star_emilio_sum_difference_l680_680978

theorem star_emilio_sum_difference :
  let star_sum := (List.range 41).sum
  let emilio_sum := 
    (List.range 41).map (λ n, 
      let s := n.digits 10
      s.foldr (λ d acc, acc * 10 + if d = 3 then 2 else d) 0
    ).sum in
  star_sum - emilio_sum = 104 :=
by
  sorry

end star_emilio_sum_difference_l680_680978


namespace work_completion_time_l680_680310

theorem work_completion_time
    (persons1 : ℕ)
    (days1 : ℕ)
    (initial_days : ℕ)
    (additional_persons : ℕ)
    (remaining_days : ℕ) :
    persons1 = 12 →
    days1 = 18 →
    initial_days = 6 →
    additional_persons = 4 →
    remaining_days = 12 →
    let total_persons := persons1 + additional_persons in
    let work_done_by_initial := initial_days / days1 in
    let remaining_work := 1 - work_done_by_initial in
    let work_done_per_day := total_persons / (persons1 * days1) in
    remaining_work / work_done_per_day = remaining_days :=
begin
    intros h_persons1 h_days1 h_initial_days h_additional_persons h_remaining_days,
    let total_persons := persons1 + additional_persons,
    let work_done_by_initial := initial_days / days1,
    let remaining_work := 1 - work_done_by_initial,
    let work_done_per_day := total_persons / (persons1 * days1),
    have h_remaining_work : remaining_work = 2 / 3,
    { rw [h_persons1, h_days1, h_initial_days],
      simp [work_done_by_initial, remaining_work] },
    have h_work_done_per_day : work_done_per_day = 1 / 27,
    { rw [h_persons1, h_days1, h_additional_persons],
      simp [work_done_per_day, total_persons] },
    rw [h_remaining_days],
    sorry
end

end work_completion_time_l680_680310


namespace cos_double_angle_l680_680496

theorem cos_double_angle (θ : ℝ) (h : Real.tan θ = -1/3) : Real.cos (2 * θ) = 4/5 :=
sorry

end cos_double_angle_l680_680496


namespace probability_no_adjacent_stands_l680_680056

-- Definitions based on the conditions
def fair_coin_flip : ℕ := 2 -- Each person can flip one of two possible outcomes (head or tail).

-- The main theorem stating the probability
theorem probability_no_adjacent_stands : 
  let total_outcomes := fair_coin_flip ^ 8 in -- Total number of possible sequences
  let favorable_outcomes := 47 in -- Number of valid sequences where no two adjacent people stand
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 47 / 256 :=
by
  sorry

end probability_no_adjacent_stands_l680_680056


namespace sector_properties_l680_680309

-- Definitions for the conditions
def central_angle (α : ℝ) : Prop := α = 2 * Real.pi / 3

def radius (r : ℝ) : Prop := r = 6

def sector_perimeter (l r : ℝ) : Prop := l + 2 * r = 20

-- The statement encapsulating the proof problem
theorem sector_properties :
  (central_angle (2 * Real.pi / 3) ∧ radius 6 →
    ∃ l S, l = 4 * Real.pi ∧ S = 12 * Real.pi) ∧
  (∃ l r, sector_perimeter l r ∧ 
    ∃ α S, α = 2 ∧ S = 25) := by
  sorry

end sector_properties_l680_680309


namespace no_such_k_and_m_exists_l680_680079

-- Define the sequence a_n based on the number of 1's in the binary representation
def a (n : ℕ) : ℕ :=
  if nat.popcount n % 2 = 0 then 0 else 1

-- The theorem to be proved
theorem no_such_k_and_m_exists :
  ∀ (k m : ℕ), m > 0 → ¬ (∀ j, 0 ≤ j → j ≤ m - 1 → a (k + j) = a (k + m + j) ∧ a (k + j) = a (k + 2 * m + j)) :=
by
  intro k m hm
  intro h
  sorry

end no_such_k_and_m_exists_l680_680079


namespace shaded_region_area_l680_680632

noncomputable theory

def PQ : ℝ := 10
def num_of_squares : ℕ := 20

theorem shaded_region_area (h1 : PQ = 10) (h2 : num_of_squares = 20) :
  let s := PQ / Real.sqrt 2,
      area_one_square := (10 ^ 2) / (2 * 9),
      total_area := num_of_squares * area_one_square
  in total_area = 1000 / 9 :=
by
  sorry

end shaded_region_area_l680_680632


namespace number_of_juniors_in_club_l680_680169

theorem number_of_juniors_in_club
  (total_students : ℕ)
  (junior_percentage_chess : ℝ)
  (senior_percentage_chess : ℝ)
  (equal_chess_team_numbers : ∀ (x y : ℕ), x = y)
  (total_students_eq : ∀ (J S : ℕ), J + S = total_students) :
  total_students = 36 → junior_percentage_chess = 0.4 → senior_percentage_chess = 0.2 →
  (∀ (J S : ℕ), 0.4 * J = 0.2 * S → J + S = 36 → J = 12) :=
by
  sorry

end number_of_juniors_in_club_l680_680169


namespace eliminate_irrationality_in_denominator_l680_680228

noncomputable def rationalize_denominator (a b : ℚ) (p : ℚ) (h : ¬(a^2 - b^2 * p = 0)) : ℚ :=
(a - b * real.sqrt p) / (a^2 - b^2 * p)

theorem eliminate_irrationality_in_denominator
  (a b : ℚ)
  (p1 p2 ... pk : ℚ)
  (h : ∀ (i : ℕ) (h_i : i ≤ k), p_i ∈ ℚ -> ∃ q_i : ℚ, sqrt(p_i) = q_i):
  ∃ r : ℚ, 1 / (a + b * sqrt(pk)) = r :=
begin
  let r := rationalize_denominator a b pk,
  use r,
  sorry
end

end eliminate_irrationality_in_denominator_l680_680228


namespace line_equation_l680_680660

theorem line_equation (x y : ℝ) (l : ℝ → ℝ → Prop) :
  (∀ x y, l x y ↔ y = k * x + b) ∧ l x y ∧ l (4 * x + 2 * y) (x + 3 * y) →
  l x y ↔ (x = y ∨ x = 2 * y) := 
by 
  intro h
  sorry

end line_equation_l680_680660


namespace trapezoid_AD_sqrt7_l680_680898

theorem trapezoid_AD_sqrt7 (A B C D : Point) (h_trap : Trapezoid A B C D) 
  (h_parallel : Parallel AB CD) (h_AB : AB = 1) (h_AC : AC = 2) 
  (h_BD : BD = 2 * Real.sqrt 3) (h_angle : ∠ACD = 60) : 
  AD = Real.sqrt 7 :=
sorry

end trapezoid_AD_sqrt7_l680_680898


namespace problem_proof_l680_680493

-- Definitions for the problem
def angle_A := 40
def angle_AFG := angle_AGF

-- Problem statement
theorem problem_proof :
  angle_A = 40 ∧ angle_AFG = angle_AGF → angle_B + angle_D = 70 := by
  sorry

end problem_proof_l680_680493


namespace selection_methods_l680_680005

theorem selection_methods (a b c d e : Type) : 
  let people := [a, b, c, d, e]
  let total_ways := Nat.mul 5 (Nat.pred 5) -- This corresponds to A_{5}^{2} = 5 * 4
  let restricted_ways := Nat.pred 4 -- This corresponds to A_{4}^{1} = 4
  let valid_ways := total_ways - restricted_ways
  in valid_ways = 16 :=
by
  sorry

end selection_methods_l680_680005


namespace average_sales_l680_680321

/-- The sales for the first five months -/
def sales_first_five_months := [5435, 5927, 5855, 6230, 5562]

/-- The sale for the sixth month -/
def sale_sixth_month := 3991

/-- The correct average sale to be achieved -/
def correct_average_sale := 5500

theorem average_sales :
  (sales_first_five_months.sum + sale_sixth_month) / 6 = correct_average_sale :=
by
  sorry

end average_sales_l680_680321


namespace natural_number_divisors_sum_l680_680635

theorem natural_number_divisors_sum (N : ℕ)
  (h : N + (N / (nat.smallest_divisor N)) + (N / (nat.second_smallest_divisor N))
     = 10 * (1 + (nat.smallest_divisor N) + (nat.second_smallest_divisor N))) :
  N = 40 :=
sorry

end natural_number_divisors_sum_l680_680635


namespace total_number_of_wheels_and_ratio_l680_680720

theorem total_number_of_wheels_and_ratio 
  (bicycles : ℕ) (tricycles : ℕ) (unicycles : ℕ) (cars : ℕ)
  (wheelsPerBicycle wheelsPerTricycle wheelsPerUnicycle wheelsPerCar : ℕ) 
  (hb : bicycles = 24) (ht : tricycles = 14) (hu : unicycles = 10) (hc : cars = 18)
  (wb : wheelsPerBicycle = 2) (wt : wheelsPerTricycle = 3) (wu : wheelsPerUnicycle = 1) (wc : wheelsPerCar = 4) :

  let total_wheels := bicycles * wheelsPerBicycle + tricycles * wheelsPerTricycle + unicycles * wheelsPerUnicycle + cars * wheelsPerCar in
  total_wheels = 172 ∧ (unicycles * wheelsPerUnicycle) / total_wheels = 5 / 86 := 
sorry

end total_number_of_wheels_and_ratio_l680_680720


namespace minimum_value_of_function_l680_680422

noncomputable def min_value_func : ℝ := 25

theorem minimum_value_of_function (x : ℝ) (hx : cos x ≠ 0 ∧ sin x ≠ 0) : 
  (\frac{4}{cos x ^ 2} + \frac{9}{sin x ^ 2}) ≥ min_value_func := sorry

end minimum_value_of_function_l680_680422


namespace ordinate_cannot_be_1_l680_680786

def f (x : ℝ) : ℝ := (x + 2) / (x + 1)

theorem ordinate_cannot_be_1 : ¬ ∃ x : ℝ, f x = 1 :=
by {
  sorry
}

end ordinate_cannot_be_1_l680_680786


namespace pyramid_volume_l680_680524

theorem pyramid_volume (A B C M : ℝ → ℝ → ℝ)
  (hAB : dist A B = 6)
  (hBC : dist B C = 4)
  (hAC : dist A C = 5)
  (height_prism : ℝ)
  (h_height_prism : height_prism = 5)
  (M_midpoint : ∀ (X Y : ℝ → ℝ → ℝ), M = (1 / 2) • (X + Y)) :
  let s := (hAB + hBC + hAC) / 2,
      area_ABC := sqrt (s * (s - hAB) * (s - hBC) * (s - hAC)),
      volume := (1 / 3) * area_ABC * height_prism
  in volume = 16.5 :=
by
  -- Placeholder for proof steps
  sorry

end pyramid_volume_l680_680524


namespace find_x_l680_680174

-- Define the angles involved and conditions
def angle_AXB := 180
def angle_YXZ := 70
def angle_XYZ := 60
def sum_of_triangle_angles := 180

-- Prove that x = 50 given the conditions
theorem find_x (h1 : angle_YXZ = 70) (h2 : angle_XYZ = 60) (h3 : angle_AXB = 180) (h4 : sum_of_triangle_angles = 180) :
  ∃ x : ℝ, x + angle_YXZ + angle_XYZ = sum_of_triangle_angles ∧ x = 50 :=
by
  use 50
  split
  · simp [angle_YXZ, angle_XYZ, sum_of_triangle_angles]
  · sorry -- Proof step placeholder

end find_x_l680_680174


namespace tasty_sequences_division_l680_680921

variable (n : ℕ := 2 ^ 2018)
variable (S : Finset ℕ := Finset.range (n + 1))

/-- Holds the murine relation for pairs of elements -/
def murine (i j : ℕ) (S : Finset (ℕ → Prop)) (S_i S_j : ℕ → Prop) : Prop :=
  (S_i i ∧ S_i j) ∨ (S_j i ∧ S_j j)

/-- Condition 1: for all i, i ∈ S_i -/
def cond1 {S : Finset (ℕ → Prop)} (S_i : ℕ → Prop) (i : ℕ) : Prop :=
  S_i i

/-- Condition 2: for all i, ⨃_{j ∈ S_i} S_j = S_i -/
def cond2 {S : Finset (ℕ → Prop)} (S_i : ℕ → Prop) (i : ℕ) : Prop :=
  ∀ j, S_i j → S_i = S_i

/-- Condition 3: no k distinct integers forming a murine cycle with k ≥ 3 -/
def cond3 {S : Finset (ℕ → Prop)} : Prop :=
  ∀ a : List ℕ, a.length ≥ 3 → a.Nodup → ¬ List.cycle a (λ b c, murine b c S S b S c)

/-- Condition 4: n divides 1 + Σ |S_i| -/
def cond4 {S : Finset (ℕ → Prop)} (n : ℕ) : Prop :=
  n ∣ (1 + S.sum (λ S_i, S_i.toFinset.card))

/-- The largest integer x such that 2^x divides the number of tasty sequences (S_1, ..., S_n) is 2018 -/
theorem tasty_sequences_division {S : Finset (ℕ → Prop)}
  (h1 : ∀ S_i i, cond1 S_i i)
  (h2 : ∀ S_i i, cond2 S_i i)
  (h3 : cond3 S)
  (h4 : cond4 n) :
  ∃ x, 2 ^ x ∣ (tasty_sequences_count n S) ∧ x = 2018 :=
sorry

end tasty_sequences_division_l680_680921


namespace geom_seq_identity_l680_680533

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∀ n, ∃ r, a (n+1) = r * a n

theorem geom_seq_identity (a : ℕ → ℝ) (r : ℝ) (h1 : geometric_sequence a) (h2 : a 2 + a 4 = 2) :
  a 1 * a 3 + 2 * a 2 * a 4 + a 3 * a 5 = 4 := 
  sorry

end geom_seq_identity_l680_680533


namespace decreasing_range_of_a_l680_680750

noncomputable def f (a x : ℝ) : ℝ :=
  a * x^2 - 2 * x + 1

theorem decreasing_range_of_a (a : ℝ) :
  (∀ x y ∈ Icc (1 : ℝ) 10, x ≤ y → f(a, y) ≤ f(a, x)) ↔ a ∈ Iic (1 / 10) :=
by
  sorry

end decreasing_range_of_a_l680_680750


namespace range_of_a_l680_680136

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 4 * x + a ≥ 0) → a ≥ 4 :=
by
  sorry

end range_of_a_l680_680136


namespace miles_driven_l680_680910

-- Definitions based on the conditions
def years : ℕ := 9
def months_in_a_year : ℕ := 12
def months_in_a_period : ℕ := 4
def miles_per_period : ℕ := 37000

-- The proof statement
theorem miles_driven : years * months_in_a_year / months_in_a_period * miles_per_period = 999000 := 
sorry

end miles_driven_l680_680910


namespace max_subtriangles_l680_680903

theorem max_subtriangles (T : Type) (k : ℕ) (k_condition : k = 2 ∨ k = 3 ∨ k = 4 ∨ k = 5) :
  ∃ C,  C = 19 ∧ 
        ∀ (network : T → Prop), 
        (∀ vertex, vertex ∈ network → count_segments vertex = k) ∧ 
        (∀ vertex, ¬ vertex splits_side T) →
        C = max_triangles network :=
sorry

end max_subtriangles_l680_680903


namespace intersection_of_M_and_N_l680_680476

noncomputable def M : set ℝ := {x | ∃ y, y = real.sqrt (-x^2 + 2*x + 8)}
noncomputable def N : set ℝ := {x | ∃ y, y = real.abs x + 1}

theorem intersection_of_M_and_N : (M ∩ N) = {x | -2 ≤ x ∧ x ≤ 4} :=
by
  sorry

end intersection_of_M_and_N_l680_680476


namespace Grisha_owes_correct_l680_680210

noncomputable def Grisha_owes (dish_cost : ℝ) : ℝ × ℝ :=
  let misha_paid := 3 * dish_cost
  let sasha_paid := 2 * dish_cost
  let friends_contribution := 50
  let equal_payment := 50 / 2
  (misha_paid - equal_payment, sasha_paid - equal_payment)

theorem Grisha_owes_correct :
  ∀ (dish_cost : ℝ), (dish_cost = 30) → Grisha_owes dish_cost = (40, 10) :=
by
  intro dish_cost h
  rw [h]
  unfold Grisha_owes
  simp
  sorry

end Grisha_owes_correct_l680_680210


namespace solve_logarithmic_equation_l680_680269

theorem solve_logarithmic_equation :
  ∀ x : ℝ, (real.log 2 (2 * x + 1) * real.log 2 (2 * x + 3) = 2) ↔ x = 0 :=
by
  sorry

end solve_logarithmic_equation_l680_680269


namespace other_continents_passengers_l680_680869

def passengers_from_other_continents (T N_A E A As : ℕ) : ℕ := T - (N_A + E + A + As)

theorem other_continents_passengers :
  passengers_from_other_continents 108 (108 / 12) (108 / 4) (108 / 9) (108 / 6) = 42 :=
by
  -- Proof goes here
  sorry

end other_continents_passengers_l680_680869


namespace max_problems_missed_to_pass_l680_680365

theorem max_problems_missed_to_pass (total_problems : ℕ) (min_percentage : ℚ) 
  (h_total_problems : total_problems = 40) 
  (h_min_percentage : min_percentage = 0.85) : 
  ∃ max_missed : ℕ, max_missed = total_problems - ⌈total_problems * min_percentage⌉₊ ∧ max_missed = 6 :=
by
  sorry

end max_problems_missed_to_pass_l680_680365


namespace avg_speed_back_is_30_l680_680347

variable (distance_to_work : ℝ)
variable (speed_to_work : ℝ)
variable (total_commute_time : ℝ)

def speed_back_from_work (distance_to_work speed_to_work total_commute_time : ℝ) : ℝ :=
  distance_to_work / (total_commute_time - (distance_to_work / speed_to_work))

theorem avg_speed_back_is_30 :
  speed_back_from_work 18 45 1 = 30 := 
by
  sorry

end avg_speed_back_is_30_l680_680347


namespace larger_page_sum_137_l680_680756

theorem larger_page_sum_137 (x y : ℕ) (h1 : x + y = 137) (h2 : y = x + 1) : y = 69 :=
sorry

end larger_page_sum_137_l680_680756


namespace gcd_binom_integer_l680_680200

theorem gcd_binom_integer (n m : ℕ) (hnm : n ≥ m) (hm : m ≥ 1) :
  (Nat.gcd m n) * Nat.choose n m % n = 0 := sorry

end gcd_binom_integer_l680_680200


namespace adam_age_is_8_l680_680350

variables (A : ℕ) -- Adam's current age
variable (tom_age : ℕ) -- Tom's current age
variable (combined_age : ℕ) -- Their combined age in 12 years

theorem adam_age_is_8 (h1 : tom_age = 12) -- Tom is currently 12 years old
                    (h2 : combined_age = 44) -- In 12 years, their combined age will be 44 years old
                    (h3 : A + 12 + (tom_age + 12) = combined_age) -- Equation representing the combined age in 12 years
                    : A = 8 :=
by
  sorry

end adam_age_is_8_l680_680350


namespace ecommerce_max_discount_l680_680050

theorem ecommerce_max_discount
  (cost_price : ℕ)
  (marked_price : ℕ)
  (profit_margin : ℕ)
  (discount : ℕ) :
  cost_price = 600 →
  marked_price = 900 →
  profit_margin = 5 →
  (marked_price * (1 - discount / 100) - cost_price) ≥ (cost_price * profit_margin / 100) →
  discount = 30 :=
by
  intros h_cost h_marked h_margin h_disc
  have h1 : marked_price - marked_price * discount / 100 - cost_price ≥ cost_price * profit_margin / 100,
  {
    rw [h_cost, h_marked, h_margin] at h_disc,
    exact h_disc,
  }
  have h2 : 300 - 9 * discount = 30,
  {
    sorry, -- substitute and solve the inequality here
  }
  sorry -- derive the final conclusion from h2

end ecommerce_max_discount_l680_680050


namespace quadratic_solution_l680_680241

theorem quadratic_solution (x : ℝ) : x^2 - 2 * x - 3 = 0 → (x = 3 ∨ x = -1) :=
by
  sorry

end quadratic_solution_l680_680241


namespace sequence_an_formula_l680_680896

theorem sequence_an_formula (a : ℕ → ℕ) (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * a n + 1) :
  ∀ n : ℕ, n ≥ 1 → a n = 2^n - 1 :=
by
  sorry

end sequence_an_formula_l680_680896


namespace additional_amount_l680_680973

theorem additional_amount (n : ℕ) (a : ℕ) : (7 * n = 3 * n + a) → n = 3 → a = 12 :=
by
  intros h1 h2
  rw [h2] at h1
  rw [show 7 * 3 = 21, by norm_num] at h1
  rw [show 3 * 3 = 9, by norm_num] at h1
  have : 21 = 9 + a := h1
  linarith

end additional_amount_l680_680973


namespace lottery_profit_l680_680706

-- Definitions

def Prob_A := (1:ℚ) / 5
def Prob_B := (4:ℚ) / 15
def Prob_C := (1:ℚ) / 5
def Prob_D := (2:ℚ) / 15
def Prob_E := (1:ℚ) / 5

def customers := 300

def first_prize_value := 9
def second_prize_value := 3
def third_prize_value := 1

-- Proof Problem Statement

theorem lottery_profit : 
  (first_prize_category == "D") ∧ 
  (second_prize_category == "B") ∧ 
  (300 * 3 - ((300 * Prob_D) * 9 + (300 * Prob_B) * 3 + (300 * (Prob_A + Prob_C + Prob_E)) * 1)) == 120 :=
by 
  -- Insert mathematical proof here using given probabilities and conditions
  sorry

end lottery_profit_l680_680706


namespace cost_of_first_shipment_1100_l680_680259

variables (S J : ℝ)
-- conditions
def second_shipment (S J : ℝ) := 5 * S + 15 * J = 550
def first_shipment (S J : ℝ) := 10 * S + 20 * J

-- goal
theorem cost_of_first_shipment_1100 (S J : ℝ) (h : second_shipment S J) : first_shipment S J = 1100 :=
sorry

end cost_of_first_shipment_1100_l680_680259


namespace park_will_have_9_oak_trees_l680_680276

def current_oak_trees : Nat := 5
def additional_oak_trees : Nat := 4
def total_oak_trees : Nat := current_oak_trees + additional_oak_trees

theorem park_will_have_9_oak_trees : total_oak_trees = 9 :=
by
  sorry

end park_will_have_9_oak_trees_l680_680276


namespace union_set_l680_680137

def M : Set ℝ := {x | -2 < x ∧ x < 1}
def P : Set ℝ := {x | -2 ≤ x ∧ x < 2}

theorem union_set : M ∪ P = {x : ℝ | -2 ≤ x ∧ x < 2} := by
  sorry

end union_set_l680_680137


namespace points_per_vegetable_correct_l680_680633

-- Given conditions
def total_points_needed : ℕ := 200
def number_of_students : ℕ := 25
def number_of_weeks : ℕ := 2
def veggies_per_student_per_week : ℕ := 2

-- Derived values
def total_veggies_eaten_by_class : ℕ :=
  number_of_students * number_of_weeks * veggies_per_student_per_week

def points_per_vegetable : ℕ :=
  total_points_needed / total_veggies_eaten_by_class

-- Theorem to be proven
theorem points_per_vegetable_correct :
  points_per_vegetable = 2 := by
sorry

end points_per_vegetable_correct_l680_680633


namespace comprehensive_score_l680_680711

variable (regularAssessmentScore : ℕ)
variable (finalExamScore : ℕ)
variable (regularAssessmentWeighting : ℝ)
variable (finalExamWeighting : ℝ)

theorem comprehensive_score 
  (h1 : regularAssessmentScore = 95)
  (h2 : finalExamScore = 90)
  (h3 : regularAssessmentWeighting = 0.20)
  (h4 : finalExamWeighting = 0.80) :
  (regularAssessmentScore * regularAssessmentWeighting + finalExamScore * finalExamWeighting) = 91 :=
sorry

end comprehensive_score_l680_680711


namespace totient_function_product_l680_680594

-- Assuming phi is defined as per Euler’s totient function and supporting structures are available
noncomputable def phi (n : ℕ) : ℕ := sorry

theorem totient_function_product (p : ℕ → ℕ) (λ : ℕ → ℕ) (n : ℕ)
  (prime_p : ∀ i, nat.prime (p i)) (pos_λ : ∀ i, 0 < λ i) :
  phi (∏ i in finset.range n, (p i) ^ (λ i)) =
  (∏ i in finset.range n, (p i) ^ ((λ i) - 1)) * (∏ i in finset.range n, phi (p i)) :=
sorry

end totient_function_product_l680_680594


namespace find_p_plus_q_l680_680644

noncomputable def triangleDEF := (15 : ℝ, 36 : ℝ, 39 : ℝ)

def rectangleWXYZ_area (ε : ℝ) (γ δ : ℝ) : ℝ :=
  γ * ε - δ * ε^2

theorem find_p_plus_q :
  let γ := 36 * (5 / 12)
  let δ := (5 / 12)
  ∃ (p q : ℕ), p + q = 17 :=
by
  -- Definitions and assumptions
  let a := 15 : ℝ
  let b := 36 : ℝ
  let c := 39 : ℝ
  let s := (a + b + c) / 2
  let areaDEF := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let ε := 18 : ℝ
  -- Calculations
  have eq1 : γ * ε - δ * ε^2 = areaDEF / 2 := sorry
  -- By solving linear and quadratic equations, we find δ = 5/12
  -- Now using p := 5, q := 12
  use 5
  use 12
  -- Check the sum
  show (5 + 12) = 17 from rfl

end find_p_plus_q_l680_680644


namespace number_of_whole_numbers_with_cube_roots_less_than_8_l680_680855

theorem number_of_whole_numbers_with_cube_roots_less_than_8 :
  ∃ (n : ℕ), (∀ (x : ℕ), (1 ≤ x ∧ x < 512) → x ≤ n) ∧ n = 511 := 
sorry

end number_of_whole_numbers_with_cube_roots_less_than_8_l680_680855


namespace problem_statement_l680_680878

/-- 
  Theorem: If the solution set of the inequality (ax-1)(x+2) > 0 is -3 < x < -2, 
  then a equals -1/3 
--/
theorem problem_statement (a : ℝ) :
  (forall x, (ax-1)*(x+2) > 0 -> -3 < x ∧ x < -2) → a = -1/3 := 
by
  sorry

end problem_statement_l680_680878


namespace problem_21_divisor_l680_680507

theorem problem_21_divisor 
    (k : ℕ) 
    (h1 : ∃ k, 21^k ∣ 435961) 
    (h2 : 21^k ∣ 435961) 
    : 7^k - k^7 = 1 := 
sorry

end problem_21_divisor_l680_680507


namespace normal_distribution_probability_l680_680461

noncomputable theory
open ProbabilityTheory

variable {X : ℝ → ℝ} {μ : measure_theory.measure ℝ}

def normal_distribution (X : ℝ → ℝ) (μ : measure_theory.measure ℝ) (mean variance : ℝ) : Prop :=
  ∃ (density : ℝ → ℝ), 
    μ = measure_theory.measure.add_haar ℝ density ∧ density = λ x, (1 / (variance * (2 * real.pi) ^ (1 / 2))) * real.exp (-(x - mean) ^ 2 / (2 * variance))

def P_le_0 (μ : measure_theory.measure ℝ) : ℝ :=
  measure_theory.measure.to_outer_measure μ 0

theorem normal_distribution_probability (σ : ℝ) :
  normal_distribution X μ 1 σ^2 → 
  P_le_0 μ = 0.1 →
  ∀ x, (P (x > 2)) = 0.1 :=
by
  intros h_dist h_prob x,
  sorry

end normal_distribution_probability_l680_680461


namespace birds_flew_up_count_l680_680682

def initial_birds : ℕ := 29
def final_birds : ℕ := 42

theorem birds_flew_up_count : final_birds - initial_birds = 13 :=
by sorry

end birds_flew_up_count_l680_680682


namespace card_A_eq_binom_l680_680555

theorem card_A_eq_binom (n r : ℕ) (Z : Finset (Fin n)) (x : Fin n) (hx : x ∈ Z) 
  (A : Finset (Finset (Fin n))) (hA : A = {S ∈ Z.powerset.filter (λ s, s.card = r) | x ∈ S}) 
  (h2_le_r : 2 ≤ r) (h_r_le_half_n : r ≤ n / 2) : 
  A.card = Nat.choose (n - 1) (r - 1) := by
  sorry

end card_A_eq_binom_l680_680555


namespace angle_C_is_70_l680_680161

namespace TriangleAngleSum

def angle_sum_in_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180

def sum_of_two_angles (A B : ℝ) : Prop :=
  A + B = 110

theorem angle_C_is_70 {A B C : ℝ} (h1 : angle_sum_in_triangle A B C) (h2 : sum_of_two_angles A B) : C = 70 :=
by
  sorry

end TriangleAngleSum

end angle_C_is_70_l680_680161


namespace median_eq_altitude_eq_l680_680892

-- Define the coordinates of the vertices
def A := (7, 8 : ℝ)
def B := (10, 4 : ℝ)
def C := (2, -4 : ℝ)

-- Prove that the equation of the median from B to AC is 8x - y - 48 = 0
theorem median_eq {x y : ℝ} :
  (y - 0) = 8 * (x - 6) ↔ 8 * x - y - 48 = 0 :=
by sorry

-- Prove that the equation of the altitude from B to AC is x + y - 15 = 0
theorem altitude_eq {x y : ℝ} :
  (y - 8) = -1 * (x - 7) ↔ x + y - 15 = 0 :=
by sorry

end median_eq_altitude_eq_l680_680892


namespace vasily_max_points_l680_680957

theorem vasily_max_points :
  ∀ (deck : Finset (Fin 36)) (polina_pref : Finset (Fin 18)) (vasily_hand : Finset (Fin 18)),
  deck.card = 36 ∧
  polina_pref ⊆ deck ∧
  vasily_hand ⊆ deck ∧
  disjoint polina_pref vasily_hand →
  Exists (λ points : ℕ, points = 15) :=
by
  sorry

end vasily_max_points_l680_680957


namespace original_cost_price_correct_l680_680346

lemma original_cost_price (C : ℝ) (discounted_loss_SF : 0.855C) (taxed_gain_SF : 1.2096C) :
  0.855 * C + 540 = 1.2096 * C :=
by
  sorry

def approx_original_cost_price (C : ℝ) : ℝ :=
  540 / 0.3546

theorem original_cost_price_correct (C : ℝ) (E : 0.855 * C + 540 = 1.2096 * C) :
  abs(C - 1522.47) < 1 :=
by
  sorry

end original_cost_price_correct_l680_680346


namespace geometric_seq_extremum_l680_680868

theorem geometric_seq_extremum (a b c d : ℝ) (h_seq : b^2 = a*c)
  (hx : ∀ x: ℝ, ∃ r: ℝ, a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 →
     (b = r * a ∧ c = r^2 * a ∧ d = r^3 * a) →
     (∂(((1/3 : ℝ) * a * x^3 + b * x^2 + c * x + d) ∂ x) = a * x^2 + 2 * b * x + c) ∧ 4 * b^2 - 4 * a * c = 0):
  ¬ ∃ x : ℝ, has_max (λ x, (1/3:ℝ) * a * x^3 + b * x^2 + c * x + d) x ∧ 
  ¬ ∃ x : ℝ, has_min (λ x, (1/3:ℝ) * a * x^3 + b * x^2 + c * x + d) x := 
begin
  sorry
end

end geometric_seq_extremum_l680_680868


namespace decreasing_function_at_3_and_2_l680_680123

variables {α : Type} [PartialOrder α]

def is_decreasing (f : ℝ → α) (s : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f y < f x

theorem decreasing_function_at_3_and_2
  (f : ℝ → ℝ) (h : is_decreasing f (Set.Ioi 0)) :
  f 3 < f 2 :=
by
  have h_interval : 0 < 2 ∧ 0 < 3 := ⟨by linarith, by linarith⟩
  exact h h_interval.2 h_interval.1 (by linarith)

end decreasing_function_at_3_and_2_l680_680123


namespace hexagon_bird_gathering_impossible_l680_680684

-- Define a structure to represent the hexagon and bird movements
structure Hexagon :=
(bird_positions : Fin 6 → Int)

-- Initial condition: each tree starts with one bird
def initial_hexagon : Hexagon :=
{ bird_positions := λ i, 1 }

-- Define the rule for bird movement
def move_birds (h : Hexagon) : Hexagon :=
sorry -- precise implementation of the move can be defined accordingly

theorem hexagon_bird_gathering_impossible : 
  ∀ (h : Hexagon), (h = initial_hexagon) → ¬ (∃ t : Fin 6, ∀ i : Fin 6, i ≠ t → h.bird_positions i = 0) :=
by
    sorry

end hexagon_bird_gathering_impossible_l680_680684


namespace fido_area_reach_l680_680061

theorem fido_area_reach (r : ℝ) (s : ℝ) (a b : ℕ) (h1 : r = s * (1 / (Real.tan (Real.pi / 8))))
  (h2 : a = 2) (h3 : b = 8)
  (h_fraction : (Real.pi * r ^ 2) / (2 * (1 + Real.sqrt 2) * (r ^ 2 * (Real.tan (Real.pi / 8)) ^ 2)) = (Real.pi * Real.sqrt a) / b) :
  a * b = 16 := by
  sorry

end fido_area_reach_l680_680061


namespace variance_equivalence_l680_680881

def variance (s : List ℝ) : ℝ :=
  let μ := s.sum / s.length
  (s.map (λ x => (x - μ) ^ 2)).sum / s.length

theorem variance_equivalence (x : ℝ) :
  variance [1, 2, 3, 4, x] = variance [2020, 2021, 2022, 2023, 2024] ↔ x = 0 ∨ x = 5 := by
  sorry

end variance_equivalence_l680_680881


namespace max_f_x_sin_A_value_l680_680482

noncomputable def vec_m (x : ℝ) : ℝ × ℝ := 
  (Real.cos (2 * x), (Real.sqrt 3 / 2) * (Real.sin x) - (1 / 2) * (Real.cos x))

noncomputable def vec_n (x : ℝ) : ℝ × ℝ :=
  (1, (Real.sqrt 3 / 2) * (Real.sin x) - (1 / 2) * (Real.cos x))

noncomputable def f (x : ℝ) : ℝ :=
  vec_m x.1 * vec_n x.1 + vec_m x.2 * vec_n x.2

theorem max_f_x (x : ℝ) : 
  exists k : ℤ, x = k * π - π / 12 :=
sorry

theorem sin_A_value (A B C : ℝ) (hA : 0 < A ∧ A < π / 2)
  (hB : 0 < B ∧ B < π / 2) (hC : 0 < C ∧ C < π / 2)
  (h_cosB : Real.cos B = 3 / 5) (h_fC : f C = -1 / 4) :
  Real.sin A = (4 + 3 * Real.sqrt 3) / 10 :=
sorry

end max_f_x_sin_A_value_l680_680482


namespace geometric_sequence_sum_l680_680893

theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) :
  a 1 = 1 ∧ a 5 = 8 * a 2 ∧ S n = 1023 → n = 10 :=
begin
  sorry
end

end geometric_sequence_sum_l680_680893


namespace binary_representation_of_14_l680_680409

theorem binary_representation_of_14 : nat.binary_repr 14 = "1110" :=
sorry

end binary_representation_of_14_l680_680409


namespace smallest_possible_value_n_l680_680755

noncomputable def smallest_n (y o b : ℕ) : ℕ :=
  let LCM := Nat.lcm 10 (Nat.lcm 16 18)  -- LCM of 10, 16, 18
  LCM / 15

theorem smallest_possible_value_n :
  ∃ (n : ℕ), n = 48 ∧ ∀ (y o b : ℕ),
    (∃ c1, 10 * y = c1 * 15 * 48) ∧
    (∃ c2, 16 * o = c2 * 15 * 48) ∧
    (∃ c3, 18 * b = c3 * 15 * 48) :=
begin
  use 48,
  split,
  {
    refl,  -- n = 48
  },
  {
    intros,
    split,
    {
      use (Nat.lcm 10 (Nat.lcm 16 18) / (10 * 15)), -- LCM / (10*15)
      ring,
    },
    split,
    {
      use (Nat.lcm 10 (Nat.lcm 16 18) / (16 * 15)), -- LCM / (16*15)
      ring,
    },
    {
      use (Nat.lcm 10 (Nat.lcm 16 18) / (18 * 15)), -- LCM / (18*15)
      ring,
    }
  }
end

end smallest_possible_value_n_l680_680755


namespace largest_among_numbers_l680_680753

theorem largest_among_numbers :
  ∀ (a b c d e : ℝ), 
  a = 0.997 ∧ b = 0.9799 ∧ c = 0.999 ∧ d = 0.9979 ∧ e = 0.979 →
  c > a ∧ c > b ∧ c > d ∧ c > e :=
by intros a b c d e habcde
   rcases habcde with ⟨ha, hb, hc, hd, he⟩
   simp [ha, hb, hc, hd, he]
   sorry

end largest_among_numbers_l680_680753


namespace floor_sqrt_120_l680_680397

theorem floor_sqrt_120 : (Int.floor (Real.sqrt 120) = 10) :=
by
  have h1 : 10^2 = 100 := rfl
  have h2 : 11^2 = 121 := rfl
  have h3 : 100 < 120 < 121 := by simp [h1, h2]
  have h4 : 10 < Real.sqrt 120 < 11 := by
    rw [Real.sqrt_lt, Real.sqrt_lt']
    use 120; exact h3
  exact Int.floor_eq_zero_or_incr (Real.sqrt 120) 10 (by linarith)
  sorry

end floor_sqrt_120_l680_680397


namespace least_pos_diff_between_sequences_l680_680593

-- Defining sequences A and B based on the given conditions
def sequenceA : ℕ → ℕ
| 0       := 3
| (n + 1) := if sequenceA n * 2 ≤ 300 then sequenceA n * 2 else sequenceA n

def sequenceB : ℕ → ℕ
| 0       := 15
| (n + 1) := if sequenceB n + 30 ≤ 300 then sequenceB n + 30 else sequenceB n

-- Means to find the minimum positive difference between elements of sequence A and B
def minPosDiff (seqA seqB : ℕ → ℕ) : ℕ :=
  let candidates := { (abs (seqA m - seqB n)) | m n : ℕ, seqA m ≤ 300 ∧ seqB n ≤ 300 ∧ seqA m ≠ seqB n }
  candidates.toFinset.min' sorry

-- Main statement
theorem least_pos_diff_between_sequences :
  minPosDiff sequenceA sequenceB = 3 :=
sorry

end least_pos_diff_between_sequences_l680_680593


namespace find_radius_circumscribed_circle_of_ABC_l680_680901

noncomputable def radius_circumscribed_circle (AB : ℝ) (BC : ℝ) (CA : ℝ) (angle_A : ℝ) : ℝ :=
2 * (CA / Real.sin angle_A)

theorem find_radius_circumscribed_circle_of_ABC (A B C : Type)
  [InnerProductSpace ℝ A]
  [InnerProductSpace ℝ B]
  [InnerProductSpace ℝ C]
  (angle_A : ℝ) (AB : ℝ) (area : ℝ): (angle_A = 2 * Real.pi / 3) ∧ (AB = 4) ∧ (area = 2 * Real.sqrt 3) →
  ∃ BC : ℝ, ∃ CA : ℝ, radius_circumscribed_circle AB BC CA angle_A = 2 * Real.sqrt 3 := 
sorry

end find_radius_circumscribed_circle_of_ABC_l680_680901


namespace actual_area_of_lawn_l680_680333

-- Definitions and conditions
variable (blueprint_area : ℝ)
variable (side_on_blueprint : ℝ)
variable (actual_side_length : ℝ)

-- Given conditions
def blueprint_conditions := 
  blueprint_area = 300 ∧ 
  side_on_blueprint = 5 ∧ 
  actual_side_length = 15

-- Prove the actual area of the lawn
theorem actual_area_of_lawn (blueprint_area : ℝ) (side_on_blueprint : ℝ) (actual_side_length : ℝ) (x : ℝ) :
  blueprint_conditions blueprint_area side_on_blueprint actual_side_length →
  (x = 27000000 ∧ x / 10000 = 2700) :=
by
  sorry

end actual_area_of_lawn_l680_680333


namespace union_A_B_l680_680111

-- Definitions based on the conditions
def A := { x : ℝ | x < -1 ∨ (2 ≤ x ∧ x < 3) }
def B := { x : ℝ | -2 ≤ x ∧ x < 4 }

-- The proof goal
theorem union_A_B : A ∪ B = { x : ℝ | x < 4 } :=
by
  sorry -- Proof placeholder

end union_A_B_l680_680111


namespace max_dot_product_on_circle_l680_680446

theorem max_dot_product_on_circle :
  ∀ (P : ℝ × ℝ) (O : ℝ × ℝ) (A : ℝ × ℝ),
  O = (0, 0) →
  A = (-2, 0) →
  P.1 ^ 2 + P.2 ^ 2 = 1 →
  let AO := (2, 0)
  let AP := (P.1 + 2, P.2)
  ∃ α : ℝ, P = (Real.cos α, Real.sin α) ∧ 
  ∃ max_val : ℝ, max_val = 6 ∧ 
  (2 * (Real.cos α + 2) = max_val) :=
by
  intro P O A hO hA hP 
  let AO := (2, 0)
  let AP := (P.1 + 2, P.2)
  sorry

end max_dot_product_on_circle_l680_680446


namespace triangle_sides_condition_triangle_angle_conditions_l680_680035

variable {a b : ℝ}
variable h : a > b

/-- Prove that there exists k in R such that 1 < k < (3 + sqrt 5)/2 given a > b and k = a / b. -/
theorem triangle_sides_condition (k : ℝ) (hk : k = a / b) : 
  1 < k ∧ k < (3 + Real.sqrt 5) / 2 := sorry

/-- Prove properties of the triangle given k = a / b. -/
theorem triangle_angle_conditions (k : ℝ) (hk : k = a / b) :
  (k = (1 + Real.sqrt 5) / 2 ∨
  (1 < k ∧ k < (1 + Real.sqrt 5) / 2) ∨
  ((1 + Real.sqrt 5) / 2 < k ∧ k < (3 + Real.sqrt 5) / 2)) := sorry

end triangle_sides_condition_triangle_angle_conditions_l680_680035


namespace positive_whole_numbers_cube_root_less_than_eight_l680_680852

theorem positive_whole_numbers_cube_root_less_than_eight : 
  { n : ℕ | n > 0 ∧ n < 512 }.card = 511 :=
by sorry

end positive_whole_numbers_cube_root_less_than_eight_l680_680852


namespace Alina_messages_comparison_l680_680717

theorem Alina_messages_comparison 
  (lucia_day1 : ℕ) (alina_day1 : ℕ) (lucia_day2 : ℕ) (alina_day2 : ℕ) (lucia_day3 : ℕ) (alina_day3 : ℕ)
  (h1 : lucia_day1 = 120)
  (h2 : alina_day1 = lucia_day1 - 20)
  (h3 : lucia_day2 = lucia_day1 / 3)
  (h4 : lucia_day3 = lucia_day1)
  (h5 : alina_day3 = alina_day1)
  (h6 : lucia_day1 + lucia_day2 + lucia_day3 + alina_day1 + alina_day2 + alina_day3 = 680) :
  alina_day2 = alina_day1 + 100 :=
sorry

end Alina_messages_comparison_l680_680717


namespace compute_fraction_l680_680023

theorem compute_fraction :
  ( ∏ k in (range 12).map (λ k => k + 1), (1 + (14 / k)) ) / 
  ( ∏ j in (range 10).map (λ j => j + 1), (1 + (12 / j)) ) = (26 / 11) :=
by 
  sorry

end compute_fraction_l680_680023


namespace product_of_c_l680_680774

theorem product_of_c (c : ℕ) (h₀ : 9 * x^2 + 24 * x + c = 0) (h₁ : 576 - 36 * c > 0) : 
  ∏ i in (finset.range 16).filter (λn, n > 0), i = 1307674368000 :=
by {
  sorry
}

end product_of_c_l680_680774


namespace product_even_permutation_l680_680932

theorem product_even_permutation (a : Fin 2015 → ℕ) (h : ∀ i j, i ≠ j → a i ≠ a j)
    (range_a : {x // 2015 ≤ x ∧ x ≤ 4029}): 
    ∃ i, (a i - (i + 1)) % 2 = 0 :=
by
  sorry

end product_even_permutation_l680_680932


namespace proof_l680_680103

noncomputable theory

-- Define the sequence {a_n} based on the given partial sum condition
def a (n : ℕ) : ℕ :=
  if n = 0 then 0 else 2 * n - 1

-- Define the sequence {b_n} based on the definition of a_n
def b (n : ℕ) : ℚ :=
  1 / (a n * a (n + 1))

-- Define partial sum S_n of the sequence {a_n}
def partial_sum_a (n : ℕ) : ℕ :=
  n * n

-- Define partial sum T_n of the sequence {b_n}
def partial_sum_b (n : ℕ) : ℚ :=
  Finset.sum (Finset.range n) b

-- Define the condition for λ which should hold for all n ∈ ℕ*
def condition (λ : ℚ) : Prop :=
  ∀ n : ℕ, n > 0 → λ * partial_sum_b n < n + (-1) ^ n

-- Prove that the general term a_n, partial sum T_n and the condition on λ hold.
theorem proof : (∀ n : ℕ, n > 0 → a n = 2 * n - 1) ∧
                (∀ n : ℕ, n > 0 → partial_sum_b n = n / (2 * n + 1)) ∧
                (∀ λ : ℚ, condition λ → λ ∈ Set.Ioo (-∞ : ℚ) 0) :=
by
  -- The full proof is omitted.
  sorry

end proof_l680_680103


namespace original_ratio_l680_680631

namespace OilBill

-- Definitions based on conditions
def JanuaryBill : ℝ := 179.99999999999991

def FebruaryBillWith30More (F : ℝ) : Prop := 
  3 * (F + 30) = 900

-- Statement of the problem proving the original ratio
theorem original_ratio (F : ℝ) (hF : FebruaryBillWith30More F) : 
  F / JanuaryBill = 3 / 2 :=
by
  -- This will contain the proof steps
  sorry

end OilBill

end original_ratio_l680_680631


namespace number_of_whole_numbers_with_cube_roots_less_than_8_l680_680856

theorem number_of_whole_numbers_with_cube_roots_less_than_8 :
  ∃ (n : ℕ), (∀ (x : ℕ), (1 ≤ x ∧ x < 512) → x ≤ n) ∧ n = 511 := 
sorry

end number_of_whole_numbers_with_cube_roots_less_than_8_l680_680856


namespace modulus_z1_at_a_neg2_real_a_values_for_overline_z1_plus_z2_real_l680_680616

open Complex

-- Definitions based on the conditions and problem statements
def z1 (a : ℝ) : ℂ := complex.mk (a + 5) (10 - a^2)
def z2 (a : ℝ) : ℂ := complex.mk (1 - 2a) (2a - 5)

-- The first problem statement
theorem modulus_z1_at_a_neg2 : abs (z1 (-2)) = 3 * Real.sqrt 5 :=
by
  sorry

-- The second problem statement
theorem real_a_values_for_overline_z1_plus_z2_real : (∀ a : ℝ, ∃ b : ℝ, b = (z1 a).re ∧ a^2 + 2 * a - 15 = 0 → (a = -5 ∨ a = 3)) :=
by
  sorry

end modulus_z1_at_a_neg2_real_a_values_for_overline_z1_plus_z2_real_l680_680616


namespace value_of_fraction_l680_680945

theorem value_of_fraction (n : ℕ) (x : Fin n → ℤ)
  (h1 : ∀ i, -1 ≤ x i ∧ x i ≤ 2)
  (h2 : ∑ i, x i = 19)
  (h3 : ∑ i, (x i)^2 = 99) :
  let M := ∑ i, (x i)^3 in
  let m := ∑ i, (x i)^3 in
  (M / m = 7) :=
by
  let M := (∑ i, (x i)^3)
  let m := (∑ i, (x i)^3)
  sorry

end value_of_fraction_l680_680945


namespace hexagon_area_difference_l680_680522

theorem hexagon_area_difference 
  (s_large : ℝ) 
  (s_small : ℝ) 
  (h1 : s_large = 8) 
  (h2 : s_small = s_large / 2) 
  (area_formula : ∀ s : ℝ, (3 * real.sqrt 3 / 2) * s^2) : 
  area_formula s_large - area_formula s_small = 72 * real.sqrt 3 :=
by
  sorry

end hexagon_area_difference_l680_680522


namespace floor_sqrt_120_l680_680402

theorem floor_sqrt_120 :
  (∀ x : ℝ, 10^2 = 100 ∧ 11^2 = 121 ∧ 100 < 120 ∧ 120 < 121 → 
  (∃ y : ℕ, y = 10 ∧ floor (real.sqrt 120) = y)) :=
by
  assume h,
  sorry

end floor_sqrt_120_l680_680402


namespace common_difference_is_2_l680_680172

variable {a : ℕ → ℤ}  -- a is a function from natural numbers to integers

-- Condition 1
def condition_a5 : a 5 = 6 := sorry

-- Condition 2
def condition_a3 : a 3 = 2 := sorry

-- Definition of common difference
def common_difference (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∃ m n, (m ≠ n) ∧ (a m - a n) = d * (m - n)

-- The proof goal
theorem common_difference_is_2 : common_difference a 2 :=
by
  let h1 := condition_a5
  let h2 := condition_a3
  sorry

end common_difference_is_2_l680_680172


namespace shaded_areas_equality_iff_tan_phi_eq_2phi_l680_680530

theorem shaded_areas_equality_iff_tan_phi_eq_2phi (φ : ℝ) (h : 0 < φ ∧ φ < π / 4) :
  (let r := 1; let sector_ACD := φ * r^2 / 2; let triangle_ABC := r^2 * tan φ / 2 in
   let segment_ABD := triangle_ABC - sector_ACD; let lower_shaded_sector := sector_ACD in 
   segment_ABD = lower_shaded_sector) ↔ tan φ = 2 * φ := sorry

end shaded_areas_equality_iff_tan_phi_eq_2phi_l680_680530


namespace range_of_x_l680_680445

variable {f : ℝ → ℝ} (hf_odd : ∀ x, f (-x) = -f x)
variable {F : ℝ → ℝ} 

-- Define the derivative condition on f.
axiom hf_deriv_cond : ∀ x, x ≤ 0 → x * (deriv f x) < f (-x)

-- Define F(x) as x * f(x)
def F (x : ℝ) : ℝ := x * f x

theorem range_of_x (hf : ∀ x, F 3 > F (2 * x - 1)) : -1 < x ∧ x < 2 :=
sorry

end range_of_x_l680_680445


namespace exists_bi_ge_one_l680_680187

-- Definitions of the conditions
variables {n : ℕ} (a : Fin n → ℝ)
hypothesis (h1 : ∀ i, 0 ≤ a i)           -- nonnegative real numbers
hypothesis (h2 : ∑ i in Finset.univ, a i = n / 2)  -- the sum is n/2

-- Define bi
def b (i : Fin n) : ℝ :=
  let indices := Finset.range n in
  (indices.sum (λ k, ∏ j in Finset.range k, a ((i + ⟨j, sorry⟩) % n))) +
  2 * ∏ j in Finset.range n, a ((i + ⟨j, sorry⟩) % n)

-- The goal to prove
theorem exists_bi_ge_one : ∃ i : Fin n, b a i ≥ 1 :=
sorry

end exists_bi_ge_one_l680_680187


namespace fido_yard_area_reach_l680_680060

theorem fido_yard_area_reach (s r : ℝ) (h1 : r = s / (2 * Real.sqrt 2)) (h2 : ∃ (a b : ℕ), (Real.pi * Real.sqrt a) / b = Real.pi * (r ^ 2) / (2 * s^2 * Real.sqrt 2) ) :
  ∃ (a b : ℕ), a * b = 64 :=
by
  sorry

end fido_yard_area_reach_l680_680060


namespace part_one_part_two_l680_680798
-- Import the Mathlib library for necessary definitions and theorems.

-- Define the conditions as hypotheses.
variables {a b c : ℝ} (h : a + b + c = 3) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- Part (1): State the inequality involving sums of reciprocals.
theorem part_one : (1 / (a + b) + 1 / (b + c) + 1 / (c + a)) ≥ 3 / 2 := 
by
  sorry

-- Part (2): Define the range for m in terms of the inequality condition.
theorem part_two : ∃m: ℝ, (∀a b c : ℝ, a + b + c = 3 → 0 < a → 0 < b → 0 < c → (-x^2 + m*x + 2 ≤ a^2 + b^2 + c^2)) ↔ (-2 ≤ m) ∧ (m ≤ 2) :=
by 
  sorry

end part_one_part_two_l680_680798


namespace solve_quadratic_equation_l680_680238

-- Variables for the quadratic equation ax^2 + bx + c = 0
variables (a b c x : ℝ)

-- Conditions from the problem
def quadratic_equation := (a = 1) ∧ (b = -1) ∧ (c = -5)

-- Proposition that x is a solution to the quadratic equation
def is_solution := x^2 - x - 5 = 0

-- Roots calculated using the quadratic formula
def roots := x = (1 + Real.sqrt 21) / 2 ∨ x = (1 - Real.sqrt 21) / 2

theorem solve_quadratic_equation (h : quadratic_equation) : is_solution ((1 + Real.sqrt 21) / 2) ∧ is_solution ((1 - Real.sqrt 21) / 2) :=
by {
  sorry
}

end solve_quadratic_equation_l680_680238


namespace find_a_even_function_l680_680989

theorem find_a_even_function (f : ℝ → ℝ) (a : ℝ)
  (h_even : ∀ x, f x = f (-x))
  (h_domain : ∀ x, 2 * a + 1 ≤ x ∧ x ≤ a + 5) :
  a = -2 :=
sorry

end find_a_even_function_l680_680989


namespace find_legs_l680_680694

noncomputable def right_triangle_legs (BC AD BD : ℝ) (h1 : BC = 527) (h2 : AD = 98) (h3 : BD = 527) : Prop :=
BC = 175 ∧ AD = 600

theorem find_legs (AD BD : ℝ) : right_triangle_legs 175 600 527 :=
by
  sorry

end find_legs_l680_680694


namespace fido_area_reach_l680_680062

theorem fido_area_reach (r : ℝ) (s : ℝ) (a b : ℕ) (h1 : r = s * (1 / (Real.tan (Real.pi / 8))))
  (h2 : a = 2) (h3 : b = 8)
  (h_fraction : (Real.pi * r ^ 2) / (2 * (1 + Real.sqrt 2) * (r ^ 2 * (Real.tan (Real.pi / 8)) ^ 2)) = (Real.pi * Real.sqrt a) / b) :
  a * b = 16 := by
  sorry

end fido_area_reach_l680_680062


namespace tangent_line_intersects_circle_l680_680784

theorem tangent_line_intersects_circle (a : ℝ) :
  let y := λ x : ℝ, exp x * (x^2 + a * x + 1 - 2 * a)
  let y' := λ x : ℝ, exp x * (x^2 + a * x + 2 * x + 1 - a)
  let P := (0, 1 - 2 * a)
  let l := λ x : ℝ, (1 - a) * x + 1 - 2 * a
  let circle := λ x y : ℝ, x^2 + 2 * x + y^2 - 12
  (-2, -1) ∈ λ x y : ℝ, circle x y → ∃ x y : ℝ, circle x y
:=
sorry

end tangent_line_intersects_circle_l680_680784


namespace function_conditions_met_l680_680096

def f (x : ℝ) : ℝ := Real.cos (2 * x)

theorem function_conditions_met :
  (f 0 ≠ 0) ∧
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x : ℝ, f (x + Real.pi) = f x) :=
by
  sorry

end function_conditions_met_l680_680096


namespace rita_remaining_money_l680_680965

theorem rita_remaining_money :
  let dresses_cost := 5 * 20
  let pants_cost := 3 * 12
  let jackets_cost := 4 * 30
  let transport_cost := 5
  let total_expenses := dresses_cost + pants_cost + jackets_cost + transport_cost
  let initial_money := 400
  let remaining_money := initial_money - total_expenses
  remaining_money = 139 := 
by
  sorry

end rita_remaining_money_l680_680965


namespace mrs_hilt_total_cost_l680_680215

theorem mrs_hilt_total_cost :
  let cost_hotdogs := 3 * 0.60 + 3 * 0.75 + 2 * 0.90,
      cost_ice_cream_cones := 2 * 1.50 + 3 * 2.00,
      cost_lemonade := 2.50 + 3.00 + 3.50 in
  cost_hotdogs + cost_ice_cream_cones + cost_lemonade = 23.85 :=
by
  let cost_hotdogs := 3 * 0.60 + 3 * 0.75 + 2 * 0.90
  let cost_ice_cream_cones := 2 * 1.50 + 3 * 2.00
  let cost_lemonade := 2.50 + 3.00 + 3.50
  show cost_hotdogs + cost_ice_cream_cones + cost_lemonade = 23.85
  calc
    cost_hotdogs + cost_ice_cream_cones + cost_lemonade = 5.85 + 9.00 + 9.00 : by ring
    ... = 23.85 : by norm_num

end mrs_hilt_total_cost_l680_680215


namespace sum_of_squares_of_tom_rates_l680_680283

theorem sum_of_squares_of_tom_rates :
  ∃ r b k : ℕ, 3 * r + 4 * b + 2 * k = 104 ∧
               3 * r + 6 * b + 2 * k = 140 ∧
               r^2 + b^2 + k^2 = 440 :=
by
  sorry

end sum_of_squares_of_tom_rates_l680_680283


namespace smaller_angle_at_7_30_is_45_degrees_l680_680140

noncomputable def calculateAngle (hour minute : Nat) : Real :=
  let minuteAngle := (minute * 6 : Real)
  let hourAngle := (hour % 12 * 30 : Real) + (minute / 60 * 30 : Real)
  let diff := abs (hourAngle - minuteAngle)
  if diff > 180 then 360 - diff else diff

theorem smaller_angle_at_7_30_is_45_degrees :
  calculateAngle 7 30 = 45 := 
sorry

end smaller_angle_at_7_30_is_45_degrees_l680_680140


namespace find_a_if_lines_parallel_l680_680109

theorem find_a_if_lines_parallel (a : ℝ) (h1 : ∃ y : ℝ, y = - (a / 4) * (1 : ℝ) + (1 / 4)) (h2 : ∃ y : ℝ, y = - (1 / a) * (1 : ℝ) + (1 / (2 * a))) : a = -2 :=
sorry

end find_a_if_lines_parallel_l680_680109


namespace sum_of_squares_not_divisible_by_17_l680_680270

theorem sum_of_squares_not_divisible_by_17
  (x y z : ℤ)
  (h_sum_div : 17 ∣ (x + y + z))
  (h_prod_div : 17 ∣ (x * y * z))
  (h_coprime_xy : Int.gcd x y = 1)
  (h_coprime_yz : Int.gcd y z = 1)
  (h_coprime_zx : Int.gcd z x = 1) :
  ¬ (17 ∣ (x^2 + y^2 + z^2)) := 
sorry

end sum_of_squares_not_divisible_by_17_l680_680270


namespace gain_percent_correct_l680_680355

def purchase_price : ℝ := 4700
def repair_cost : ℝ := 600
def selling_price : ℝ := 5800

-- Define gain percent calculation
def gain_percent (P R S : ℝ) : ℝ := ((S - (P + R)) / (P + R)) * 100

-- The statement we need to prove
theorem gain_percent_correct :
  gain_percent purchase_price repair_cost selling_price ≈ 9.43 :=
by
  sorry

end gain_percent_correct_l680_680355


namespace combinatorial_identity_solution_l680_680089

theorem combinatorial_identity_solution (x : ℕ) (h : nat.choose 10 (2 * x) = nat.choose 10 (x + 1)) : 
  x = 1 ∨ x = 3 :=
by sorry

end combinatorial_identity_solution_l680_680089


namespace Alice_fills_needed_l680_680000

def cups_needed : ℚ := 15/4
def cup_capacity : ℚ := 1/3
def fills_needed : ℚ := 12

theorem Alice_fills_needed : (cups_needed / cup_capacity).ceil = fills_needed := by
  -- Proof is omitted with sorry
  sorry

end Alice_fills_needed_l680_680000


namespace length_of_side_b_correct_l680_680610

noncomputable def length_of_side_b (a b c : ℝ) (area : ℝ) : ℝ :=
  sqrt ((a ^ 2 + (2 * area / c) ^ 2))

theorem length_of_side_b_correct :
  ∀ (b : ℝ), length_of_side_b 1 b 7 5 = sqrt (149) / 7 :=
by
  intros b
  rw [length_of_side_b]
  sorry

end length_of_side_b_correct_l680_680610


namespace triangle_side_length_eq_five_l680_680246

noncomputable def side_length_equilateral_triangle 
  (A B C P : Type) [MetricSpace P] 
  (equilateral: ∀ x y z : A, dist x y = dist y z) 
  (AP: dist A P = Real.sqrt 2)
  (BP: dist B P = 2)
  (CP: dist C P = 1) : 
  A :=
  5

theorem triangle_side_length_eq_five 
  (A B C P: Type) [MetricSpace P] 
  (equilateral: ∀ x y z : A, dist x y = dist y z) 
  (AP: dist A P = Real.sqrt 2)
  (BP: dist B P = 2)
  (CP: dist C P = 1) : 
  dist A B = 5 
  := sorry

end triangle_side_length_eq_five_l680_680246


namespace books_exactly_three_common_l680_680573

noncomputable def books_probability : ℚ :=
  let total_ways := Nat.choose 12 5 * Nat.choose 12 5
  let favorable_ways := Nat.choose 12 3 * Nat.choose 9 2 * Nat.choose 7 2
  favorable_ways / total_ways

theorem books_exactly_three_common :
  books_probability = 55 / 209 :=
sorry

end books_exactly_three_common_l680_680573


namespace max_edges_on_shortest_path_l680_680168

-- Given conditions
def number_of_cities : ℕ := 2018
def min_connections (city : ℕ) : Prop := city >= 3 
def is_connected (cities : ℕ → Prop) : Prop := 
  ∀ c1 c2 : ℕ, cities c1 → cities c2 → ∃ p : List ℕ, valid_path p c1 c2

-- Valid path definition (can be extended or adjusted as necessary for formal proof)
def valid_path (p : List ℕ) (from to : ℕ) : Prop := sorry

-- Proof problem statement
theorem max_edges_on_shortest_path
  (h1 : ∀ city, min_connections city)
  (h2 : is_connected (λ c, c = number_of_cities)) :
  ∃ m : ℕ, m = 1511 :=
sorry

end max_edges_on_shortest_path_l680_680168


namespace range_of_a_if_inequality_holds_l680_680512

theorem range_of_a_if_inequality_holds :
  (∀ x ∈ set.Icc (-1 : ℝ) 2, x^2 - 2 * a * x - 3 * a ≤ 0) →
  a ∈ set.Ici (1 : ℝ) :=
by
  sorry

end range_of_a_if_inequality_holds_l680_680512


namespace rita_remaining_money_l680_680966

theorem rita_remaining_money :
  let dresses_cost := 5 * 20
  let pants_cost := 3 * 12
  let jackets_cost := 4 * 30
  let transport_cost := 5
  let total_expenses := dresses_cost + pants_cost + jackets_cost + transport_cost
  let initial_money := 400
  let remaining_money := initial_money - total_expenses
  remaining_money = 139 := 
by
  sorry

end rita_remaining_money_l680_680966


namespace snacks_combination_count_l680_680352

def snacks := ["apple", "orange", "banana", "granola bar"]

theorem snacks_combination_count : (nat.choose 4 2) = 6 := by
  sorry

end snacks_combination_count_l680_680352


namespace equilateral_triangle_area_l680_680363

theorem equilateral_triangle_area
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (ha_on_curve : a * (1 / a) = 1)
  (hb_on_curve : b * (1 / b) = 1)
  (equilateral : distance (a, 1 / a) (0, 0) = distance (b, 1 / b) (0, 0) ∧ distance (a, 1 / a) (b, 1 / b) = distance (a, 1 / a) (0, 0)) :
  let side_length := distance (a, 1 / a) (0, 0) in
  (sqrt 3 / 4) * side_length ^ 2 = sqrt 3 :=
by sorry

end equilateral_triangle_area_l680_680363


namespace sequence_value_x_l680_680637

theorem sequence_value_x :
  ∃ x : ℕ, x = 33 ∧
           6 - 3 = 1 * 3 ∧
           12 - 6 = 2 * 3 ∧
           21 - 12 = 3 * 3 ∧
           x - 21 = 4 * 3 ∧
           48 - x = 5 * 3 :=
by {
  let x := 21 + 4 * 3,
  use x,
  split,
  { refl },
  repeat { split; norm_num, }
}

end sequence_value_x_l680_680637


namespace round_robin_winners_l680_680676

theorem round_robin_winners (n : ℕ) : 
  ∃ (V : fin (n+2) → fin (2^(n+1))),
    ∀ i j : fin (n+2), i < j → defeats V i V j :=
sorry

end round_robin_winners_l680_680676


namespace area_remaining_unit_l680_680612

theorem area_remaining_unit :
  ∀ (total_units total_area : ℕ) (unit1_count unit1_length unit1_width unit2_count unit2_length unit2_width : ℕ)
  (remaining_units_area remaining_units_count : ℕ),
    (total_units = 120) →
    (total_area = 15300) →
    (unit1_count = 50) →
    (unit1_length = 10) →
    (unit1_width = 7) →
    (unit2_count = 35) →
    (unit2_length = 14) →
    (unit2_width = 6) →
    (remaining_units_count = total_units - unit1_count - unit2_count) →
    (remaining_units_area =
      total_area - (unit1_count * unit1_length * unit1_width) - (unit2_count * unit2_length * unit2_width)) →
    (remaining_units_area / remaining_units_count = 253.14) := 
by
  intros total_units total_area unit1_count unit1_length unit1_width unit2_count unit2_length unit2_width
        remaining_units_area remaining_units_count
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end area_remaining_unit_l680_680612


namespace min_abs_sum_l680_680998

theorem min_abs_sum (x y z : ℝ) (hx : 0 ≤ x) (hxy : x ≤ y) (hyz : y ≤ z) (hz : z ≤ 4) 
  (hy_eq : y^2 = x^2 + 2) (hz_eq : z^2 = y^2 + 2) : 
  |x - y| + |y - z| = 4 - 2 * Real.sqrt 3 :=
sorry

end min_abs_sum_l680_680998


namespace sum_equiv_l680_680735

theorem sum_equiv {f : ℕ → ℕ → ℝ} (h : ∀ (n k : ℕ), n ≥ 3 ∧ 1 ≤ k ∧ k ≤ n - 2 → f n k = (k^2) / (3^(n+k))) :
  (∑' n=3, ∑' k=1, if h : k ≤ n - 2 then f n k else 0) = 135 / 512 :=
by sorry

end sum_equiv_l680_680735


namespace monotonic_intervals_range_of_m_l680_680086

-- Define the vectors a and b
def vectorA (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin x, cos x + sin x)
def vectorB (x : ℝ) : ℝ × ℝ := (2 * cos x, sin x - cos x)

-- Define the dot product function f(x)
def f (x : ℝ) : ℝ := (vectorA x).1 * (vectorB x).1 + (vectorA x).2 * (vectorB x).2

-- Theorem for monotonic intervals of f(x)
theorem monotonic_intervals (k : ℤ) (x : ℝ) :
  (f x = 2 * sin (2 * x - π / 6)) →
  ((-π / 6 + k * π ≤ x ∧ x ≤ π / 3 + k * π) ∨ (π / 3 + k * π ≤ x ∧ x ≤ 5 * π / 6 + k * π)) :=
sorry

-- Theorem for the range of m
theorem range_of_m (m x : ℝ) :
  x ∈ set.Icc (5 * π / 24) (5 * π / 12) →
  (0 ≤ m ∧ m ≤ 4) ↔ (∀ t : ℝ, m * t^2 + m * t + 3 ≥ f x) :=
sorry

end monotonic_intervals_range_of_m_l680_680086


namespace ellipse_equation_ellipse_chord_length_l680_680105

-- Conditions
variable {F1 : (ℝ × ℝ) := (2, 0)}
variable {directrix : ℝ -> Prop := λ x, x = 8}
variable {eccentricity : ℝ := 1 / 2}

-- Question 1: Prove the equation of the ellipse
theorem ellipse_equation (x y : ℝ) (h : sqrt ((x - 2)^2 + y^2) / abs (8 - x) = 1 / 2) :
  x^2 / 16 + y^2 / 12 = 1 :=
by
  sorry

-- Question 2: Prove the length of the chord cut from the ellipse by a given line
theorem ellipse_chord_length (x1 y1 x2 y2 : ℝ) 
  (h1 : y1 = x1 + 2)
  (h2 : y2 = x2 + 2)
  (h3 : x1^2 / 16 + y1^2 / 12 = 1)
  (h4 : x2^2 / 16 + y2^2 / 12 = 1) :
  dist (x1, y1) (x2, y2) = 48 / 7 :=
by
  sorry

end ellipse_equation_ellipse_chord_length_l680_680105


namespace solve_for_n_l680_680819

variables (P s m k r : ℝ)

theorem solve_for_n 
  (h : P = (s + m) / ((1 + k) ^ n + r)) :
  n = log ((s + m - P * r) / P) / log (1 + k) :=
sorry

end solve_for_n_l680_680819


namespace find_x_l680_680494

theorem find_x
  (log2 : Real := 0.3010)
  (log3 : Real := 0.4771)
  (x : Real := 3.46) :
  2 ^ (x + 4) = 176 :=
by
  sorry

end find_x_l680_680494


namespace distance_from_origin_z111_l680_680040

noncomputable def complex_sequence : ℕ → ℂ 
| 1       := 0
| (n + 1) := complex_sequence n ^ 2 - complex.i

theorem distance_from_origin_z111 :
  complex.abs (complex_sequence 111) = real.sqrt 2 :=
sorry

end distance_from_origin_z111_l680_680040


namespace domain_f_monotonic_increase_f_l680_680830

noncomputable def f (x : ℝ) := Real.log (x^2 - 2*x - 3) / Real.log 2

theorem domain_f : { x : ℝ | x > 3 ∨ x < -1 } = { x | f x ≠ ∅ } :=
by
  -- proof to be provided
  sorry

theorem monotonic_increase_f : ∀ x1 x2 : ℝ, x1 < x2 → 3 < x1 → f x1 < f x2 :=
by
  -- proof to be provided
  sorry

end domain_f_monotonic_increase_f_l680_680830


namespace smallest_real_in_domain_of_gg_l680_680498

def g (x : ℝ) : ℝ := real.sqrt (x - 5)

theorem smallest_real_in_domain_of_gg : (∃ x : ℝ, x = 30 ∧ ∀ y : ℝ, g(y) ∈ set.Icc (5, ∞) → y ≥ 30) := 
by {
    sorry
}

end smallest_real_in_domain_of_gg_l680_680498


namespace exponent_calculation_l680_680379

theorem exponent_calculation :
  ((19 ^ 11) / (19 ^ 8) * (19 ^ 3) = 47015881) :=
by
  sorry

end exponent_calculation_l680_680379


namespace ellipse_chord_recdefs_l680_680810

variable (a b x y d1 d2 : ℝ)
variable (h1 : a > b)
variable (h2 : b > 0)
variable (h3 : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1)
variable (h4 : ∃ F A B, true)
variable (h5 : ∀ (F A B : ℝ), |A - F| = d1 ∧ |B - F| = d2)

theorem ellipse_chord_recdefs
  (d1 d2 : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1)
  (h4 : ∃ F A B, true) 
  (h5 : ∀ (F A B : ℝ), |A - F| = d1 ∧ |B - F| = d2)
  : 1 / d1 + 1 / d2 = 2 * a / b^2 := 
sorry

end ellipse_chord_recdefs_l680_680810


namespace example_proof_l680_680836

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom axiom1 (x y : ℝ) : f (x - y) = f x * g y - g x * f y
axiom axiom2 (x : ℝ) : f x ≠ 0
axiom axiom3 : f 1 = f 2

theorem example_proof : g (-1) + g 1 = 1 := by
  sorry

end example_proof_l680_680836


namespace double_sum_equality_l680_680725

theorem double_sum_equality : 
  (∑ n in (rangeFrom 3), ∑ k in finset.range (n-2) \ k -> k + 2, (k^2 : ℝ) / 3^(n+k)) = 729 / 17576 :=
sorry

end double_sum_equality_l680_680725


namespace summation_proof_l680_680736

open BigOperators

theorem summation_proof :
  ∑ n in finset.range (∞).filter (λ n, n ≥ 3), ∑ k in finset.range (n - 2).filter (λ k, k ≥ 1), k^2 * (3:ℝ) ^ (- (n + k)) = 5 / 72 := 
by 
  sorry

end summation_proof_l680_680736


namespace range_of_a_l680_680832

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - x + 2

-- Prove that if f(x) is decreasing on ℝ, then a must be less than or equal to -3
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (3 * a * x^2 + 6 * x - 1) < 0 ) → a ≤ -3 :=
sorry

end range_of_a_l680_680832


namespace find_sum_of_natural_numbers_l680_680762

theorem find_sum_of_natural_numbers :
  ∃ (square triangle : ℕ), square^2 + 12 = triangle^2 ∧ square + triangle = 6 :=
by
  sorry

end find_sum_of_natural_numbers_l680_680762


namespace find_u_2002_l680_680257

def f : ℕ → ℕ
| 1 := 4
| 2 := 1
| 3 := 3
| 4 := 5
| 5 := 2
| _ := 0  -- Since the function is only defined for 1 ≤ x ≤ 5

def sequence (u : ℕ → ℕ) : ℕ → ℕ
| 0     := 4
| (n+1) := f (u n)

theorem find_u_2002 : sequence u 2002 = 2 :=
sorry

end find_u_2002_l680_680257


namespace area_of_semicircle_l680_680686

/-- Define the problem: A \(1 \times 3\) rectangle is inscribed in a semicircle, with the longer side
on the diameter. Prove that the area of the semicircle is \(\frac{13\pi}{8}\). --/
theorem area_of_semicircle (r d : ℝ) (h : d = 3 ∧ r = (3 / 2)) : 
  let diag := (real.sqrt (1^2 + 3^2)) in
  let radius := (diag / 2) in
  let area_circle := (real.pi * radius^2) in
  let area_semicircle := (area_circle / 2) in
area_semicircle = (13 * real.pi / 8) :=
by
  -- sorry to indicate the proof is needed here
  sorry

end area_of_semicircle_l680_680686


namespace george_speed_to_arrive_on_time_l680_680791

/-
George needs to walk 1.5 miles to school and usually walks at a speed of 4 miles per hour, arriving just as school begins. 
However, today he spent the first 1 mile daydreaming and walked at a reduced speed of 2 miles per hour. 
Prove that to arrive on time today, George must cover the remaining 0.5 miles at a speed of 4 mph.
-/
theorem george_speed_to_arrive_on_time :
  ∀ (distance_to_school usual_speed first_mile_distance first_mile_speed remaining_distance required_speed : ℚ),
    distance_to_school = 1.5 → 
    usual_speed = 4 → 
    first_mile_distance = 1 → 
    first_mile_speed = 2 → 
    remaining_distance = 0.5 → 
    required_speed = 4 → 
    let usual_time := distance_to_school / usual_speed,
        first_mile_time := first_mile_distance / first_mile_speed,
        remaining_time := usual_time - first_mile_time in
    remaining_distance / remaining_time = required_speed := 
begin
  intros,
  sorry
end

end george_speed_to_arrive_on_time_l680_680791


namespace rearrange_3008_different_numbers_l680_680845

theorem rearrange_3008_different_numbers : 
  let digits := [3, 0, 0, 8]
  let count_valid_permutations (xs : List Nat) := 
    ∑ x in xs.eraseDup, if x ≠ 0 then (xs.erase x).perm.permNRoot else 0
  count_valid_permutations digits = 6 := by
  sorry

end rearrange_3008_different_numbers_l680_680845


namespace sum_of_shaded_cells_l680_680995

theorem sum_of_shaded_cells :
  ∃ (grid : fin 3 → fin 3 → ℕ), 
    (∀ i j, grid i j ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
    (finset.univ.sum (λ n, grid n n) = 7) ∧
    (finset.univ.sum (λ n, grid n (2 - n)) = 21) ∧
    (finset.univ.sum (λ n, finset.univIf (grid (n.div2) (n.mod2) ∈ {3, 4, 5, 6, 7})) = 25) :=
sorry

end sum_of_shaded_cells_l680_680995


namespace sum_of_star_tips_l680_680575

theorem sum_of_star_tips (n : ℕ) (h1 : set.evenly_spaced_on_circle n) (h2 : n = 9) : 
 ∑ k in finset.range n, angle_measure_at_tip k h1 h2 = 720 :=
sorry

end sum_of_star_tips_l680_680575


namespace perfect_squares_unique_l680_680042

theorem perfect_squares_unique (n : ℕ) (h1 : ∃ k : ℕ, 20 * n = k^2) (h2 : ∃ p : ℕ, 5 * n + 275 = p^2) :
  n = 125 :=
by
  sorry

end perfect_squares_unique_l680_680042


namespace a_4_eq_28_l680_680535

def sequence_sum (n : ℕ) : ℕ := 4 * n^2

def a_n (n : ℕ) : ℕ := sequence_sum n - sequence_sum (n - 1)

theorem a_4_eq_28 : a_n 4 = 28 :=
by {
  let S_4 := sequence_sum 4,
  let S_3 := sequence_sum 3,
  have h1 : S_4 = 64 := by norm_num [sequence_sum],
  have h2 : S_3 = 36 := by norm_num [sequence_sum],
  have h : a_n 4 = S_4 - S_3 := rfl,
  rw [h, h1, h2],
  norm_num,
}

end a_4_eq_28_l680_680535


namespace prime_pairs_l680_680764

-- Define the predicate to check whether a number is a prime.
def is_prime (n : Nat) : Prop := Nat.Prime n

-- Define the main theorem.
theorem prime_pairs (p q : Nat) (hp : is_prime p) (hq : is_prime q) : 
  (p^3 - q^5 = (p + q)^2) → (p = 7 ∧ q = 3) :=
by
  sorry

end prime_pairs_l680_680764


namespace cos_double_angle_l680_680116

variables {α β θ : ℝ}

-- Definition from Condition 1
def condition1 : Prop := sin θ + cos θ = 2 * sin α

-- Definition from Condition 2
def condition2 : Prop := sin (2 * θ) = 2 * sin β ^ 2

-- Theorem statement
theorem cos_double_angle (h1 : condition1) (h2 : condition2) : cos (2 * β) = 2 * cos (2 * α) :=
sorry

end cos_double_angle_l680_680116


namespace rachel_assembly_time_l680_680590

theorem rachel_assembly_time :
  let chairs := 20
  let tables := 8
  let bookshelves := 5
  let time_per_chair := 6
  let time_per_table := 8
  let time_per_bookshelf := 12
  let total_chairs_time := chairs * time_per_chair
  let total_tables_time := tables * time_per_table
  let total_bookshelves_time := bookshelves * time_per_bookshelf
  total_chairs_time + total_tables_time + total_bookshelves_time = 244 := by
  sorry

end rachel_assembly_time_l680_680590


namespace cone_central_angle_l680_680516

noncomputable def central_angle_of_lateral_surface_development (r : ℝ) (l : ℝ) : ℝ :=
  2 * π * r / l

theorem cone_central_angle (r l : ℝ) (h1 : 3 * π * r^2 = π * r^2 + π * r * l) :
  central_angle_of_lateral_surface_development r l = π :=
by
  rw [central_angle_of_lateral_surface_development]
  sorry

end cone_central_angle_l680_680516


namespace chess_team_girls_count_l680_680339

theorem chess_team_girls_count (B G : ℕ) 
  (h1 : B + G = 26) 
  (h2 : (3 / 4 : ℝ) * B + (1 / 4 : ℝ) * G = 13) : G = 13 := 
sorry

end chess_team_girls_count_l680_680339


namespace tangent_parallel_l680_680156

noncomputable def f (x : ℝ) : ℝ := x^4 - x

theorem tangent_parallel (P : ℝ × ℝ) (hP : P.1 = 1) (hP_cond : P.2 = f P.1) 
  (tangent_parallel : ∀ x, deriv f x = 3) : P = (1, 0) := 
by 
  have h_deriv : deriv f 1 = 4 * 1^3 - 1 := by sorry
  have slope_eq : deriv f (P.1) = 3 := by sorry
  have solve_a : P.1 = 1 := by sorry
  have solve_b : f 1 = 0 := by sorry
  exact sorry

end tangent_parallel_l680_680156


namespace count_numbers_with_cube_root_lt_8_l680_680848

theorem count_numbers_with_cube_root_lt_8 : 
  ∀ n : ℕ, (n > 0) → (n < 8^3) → n ≤ 8^3 - 1 :=
by
  -- We need to prove that the count of such numbers is 511
  sorry

end count_numbers_with_cube_root_lt_8_l680_680848


namespace star_graph_coloring_l680_680687

theorem star_graph_coloring {n: ℕ} (h: 1 ≤ n):
  (n + 1) * ((n - 1)^n + (-1)^n * (n - 1)) = (number_of_ways_to_color_star_graph n) :=
sorry

end star_graph_coloring_l680_680687


namespace largest_integer_in_list_l680_680702

theorem largest_integer_in_list :
  ∃ (l : List ℕ), l.length = 5 ∧
  (∀ x, x ∈ l → x > 0) ∧
  (∃ count, count 7 l = 2) ∧
  (l.nth 2 = some 10) ∧
  (l.sum / 5 = 11) ∧
  (l.maximum = some 21) :=
sorry

end largest_integer_in_list_l680_680702


namespace probability_both_red_is_one_fourth_l680_680709

noncomputable def probability_of_both_red (total_cards : ℕ) (red_cards : ℕ) (draws : ℕ) : ℚ :=
  (red_cards / total_cards) ^ draws

theorem probability_both_red_is_one_fourth :
  probability_of_both_red 52 26 2 = 1/4 :=
by
  sorry

end probability_both_red_is_one_fourth_l680_680709


namespace AQ_len_l680_680538

-- Define the conditions involved
variable (A B C P Q R : Type)
variable [MetricSpace PQR]
variable [MetricSpace ABC]
variable [HasAngle B P Q R]
variable [HasAngle C P Q R]
variable (PQR_Equilateral : EquilateralTriangle P Q R)
variable (BC_Obtuse : IsObtuse B)
variable (PC_len : PC = 4)
variable (BP_CQ_len : BP = 3 ∧ CQ = 3)

-- Define the theorem that needs to be proved
theorem AQ_len (H1 : PQR_Equilateral) (H2 : BC_Obtuse) (H3 : PC_len) (H4 : BP_CQ_len) : AQ = 10 / 3 :=
by sorry

end AQ_len_l680_680538


namespace ell_inequality_l680_680939

open Set Finset

noncomputable def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def ell (P : Finset ℕ) : ℕ := 
  Sup {n | ∀ k ∈ range (n+1), ∃ p ∈ P, (k + n) % p = 0}

noncomputable def min_prime (P : Finset ℕ) : ℕ := Inf {p : ℕ | p ∈ P}

theorem ell_inequality (P : Finset ℕ) (hP : ∀ p ∈ P, is_prime p) (h_nonempty : P ≠ ∅) : 
  ell(P) ≥ P.card ∧ 
  (ell(P) = P.card ↔ min_prime P > P.card) := 
by 
  sorry

end ell_inequality_l680_680939


namespace simplify_expression_l680_680977

variable (a : ℝ)

def expression1 : ℝ := (3 / (a + 1) - 1) / ((a - 2) / (a^2 + 2*a + 1))
def simplifiedExpression (a : ℝ) : ℝ := -a - 1

# Check that the variable is within a valid range:
axiom h1 : a ≠ -1
axiom h2 : a ≠ 2

theorem simplify_expression : expression1 a = simplifiedExpression a :=
by sorry

end simplify_expression_l680_680977


namespace rita_remaining_amount_l680_680972

theorem rita_remaining_amount :
  let num_dresses := 5
  let num_pants := 3
  let num_jackets := 4
  let cost_per_dress := 20
  let cost_per_pants := 12
  let cost_per_jacket := 30
  let additional_cost := 5
  let initial_amount := 400
  let total_cost := num_dresses * cost_per_dress + num_pants * cost_per_pants + num_jackets * cost_per_jacket + additional_cost
  remaining_amount = initial_amount - total_cost
  in remaining_amount = 139 :=
by
  let num_dresses := 5
  let num_pants := 3
  let num_jackets := 4
  let cost_per_dress := 20
  let cost_per_pants := 12
  let cost_per_jacket := 30
  let additional_cost := 5
  let initial_amount := 400
  let total_cost := num_dresses * cost_per_dress + num_pants * cost_per_pants + num_jackets * cost_per_jacket + additional_cost
  let remaining_amount := initial_amount - total_cost
  show remaining_amount = 139
  sorry

end rita_remaining_amount_l680_680972


namespace find_y_given_conditions_l680_680503

theorem find_y_given_conditions (x : ℤ) (y : ℤ) (h1 : x^2 - 3 * x + 7 = y + 2) (h2 : x = -5) : y = 45 :=
by
  sorry

end find_y_given_conditions_l680_680503


namespace largest_triangle_perimeter_with_7_9_x_l680_680714

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def triangle_side_x_valid (x : ℕ) : Prop :=
  is_divisible_by_3 x ∧ 2 < x ∧ x < 16

theorem largest_triangle_perimeter_with_7_9_x (x : ℕ) (h : triangle_side_x_valid x) : 
  ∃ P : ℕ, P = 7 + 9 + x ∧ P = 31 :=
by
  sorry

end largest_triangle_perimeter_with_7_9_x_l680_680714


namespace sin_2gamma_l680_680223

-- Definitions of the given points and conditions
variables {α β γ δ : ℝ}
variables {a b c d : ℝ}
variables {A B C D P : ℝ → Prop}

-- Conditions
def equally_spaced (A B C D : ℝ → Prop) (x : ℝ) : Prop :=
  ∃ (position : ℝ → ℝ), 
    (position A = 0) ∧ (position B = x) ∧ (position C = 2 * x) ∧ (position D = 3 * x)

axiom cos_APC : cos α = 4 / 5
axiom cos_BPD : cos β = 3 / 5

-- Question to prove
theorem sin_2gamma :
  equally_spaced A B C D ∧
  (α = ∠ A P C) ∧
  (β = ∠ B P D) ∧
  (γ = ∠ B P C) ∧
  (δ = ∠ A P D) ∧
  cos α = 4 / 5 ∧
  cos β = 3 / 5 
  → sin (2 * γ) = 18 / 25 :=
sorry

end sin_2gamma_l680_680223


namespace snack_combinations_equal_six_l680_680353

theorem snack_combinations_equal_six : ∃ (comb : ℕ), comb = nat.choose 4 2 ∧ comb = 6 :=
by
  use nat.choose 4 2
  split
  . rfl
  . sorry

end snack_combinations_equal_six_l680_680353


namespace proof_problem_l680_680828
noncomputable def prop1 (x : ℝ) : ℝ := conj x = x

noncomputable def prop2 (z : ℂ) : ℂ := z - conj z = I * (z - conj z).im

noncomputable def prop3 (m : ℤ) (i2 : I * I = -1) : ℂ := 
  zpow I m + zpow I (m + 1) + zpow I (m + 2) + zpow I (m + 3) = 0

theorem proof_problem:
  (∀ x : ℝ, prop1 x) ∧ (∀ z : ℂ, ¬ prop2 z) ∧ (∀ m : ℤ, prop3 m (by norm_num)) :=
by sorry

end proof_problem_l680_680828


namespace total_cotton_yield_l680_680646

variables {m n a b : ℕ}

theorem total_cotton_yield (m n a b : ℕ) : 
  m * a + n * b = m * a + n * b := by
  sorry

end total_cotton_yield_l680_680646


namespace tangent_product_l680_680229

theorem tangent_product (α : ℝ) (n : ℕ) :
  (∑ k in Finset.range (n-1), (Real.tan (k+1) * α * Real.tan (k+2) * α)) = 
  (Real.tan (n * α) / Real.tan α) - n :=
sorry

end tangent_product_l680_680229


namespace hats_per_yard_of_velvet_l680_680911

theorem hats_per_yard_of_velvet
  (H : ℕ)
  (velvet_for_cloak : ℕ := 3)
  (total_velvet : ℕ := 21)
  (number_of_cloaks : ℕ := 6)
  (number_of_hats : ℕ := 12)
  (yards_for_6_cloaks : ℕ := number_of_cloaks * velvet_for_cloak)
  (remaining_yards_for_hats : ℕ := total_velvet - yards_for_6_cloaks)
  (hats_per_remaining_yard : ℕ := number_of_hats / remaining_yards_for_hats)
  : H = hats_per_remaining_yard :=
  by
  sorry

end hats_per_yard_of_velvet_l680_680911


namespace smallest_x_multiple_of_53_l680_680776

theorem smallest_x_multiple_of_53 :
  ∃ (x : ℕ), (3 * x + 41) % 53 = 0 ∧ x > 0 ∧ x = 4 :=
sorry

end smallest_x_multiple_of_53_l680_680776


namespace distinct_sequences_l680_680490

-- Define the set of all letters in "CHAMPION"
def letters : List Char := ['C', 'H', 'A', 'M', 'P', 'I', 'O', 'N']

-- Define the conditions
def begins_with_M (s : List Char) : Prop := s.head = 'M'
def not_end_with_N (s : List Char) : Prop := s.last ≠ 'N'
def five_distinct_chars (s : List Char) : Prop := s.length = 5 ∧ s.nodup

-- Define the final proof problem
theorem distinct_sequences : 
  ∃ l : List (List Char), 
  (∀ s ∈ l, begins_with_M s ∧ not_end_with_N s ∧ five_distinct_chars s) ∧ l.length = 720 := by
  sorry

end distinct_sequences_l680_680490


namespace positive_whole_numbers_with_cube_root_less_than_8_l680_680860

theorem positive_whole_numbers_with_cube_root_less_than_8 :
  { n : ℕ | n > 0 ∧ n < 512 }.card = 511 :=
by
  sorry

end positive_whole_numbers_with_cube_root_less_than_8_l680_680860


namespace find_a_l680_680091

noncomputable theory
open Complex

theorem find_a (a : ℝ) (h : ((1 : ℂ) + (a : ℂ) * Complex.I) * Complex.I = (3 : ℂ) + Complex.I) : a = -3 :=
sorry

end find_a_l680_680091


namespace age_when_hired_l680_680315

/-
Conditions:
- A company retirement plan allows an employee to retire when their age plus years of employment is at least 70 (Rule of 70).
- A female employee was hired in 1990.
- She could first be eligible to retire in 2009.

Question: How old was the female employee when she was hired?
- We need to prove that her age was 51.
-/

def years_of_employment : ℕ := 2009 - 1990

theorem age_when_hired (A : ℕ) (hiring_year retirement_year : ℕ) (rule_of_70 : A + years_of_employment = 70) :
  A = 51 :=
by
  have y_def : years_of_employment = 19 := by rfl
  rw y_def at rule_of_70
  linarith

-- The age of the employee when hired is 51.

end age_when_hired_l680_680315


namespace polynomial_is_constant_prime_l680_680065

theorem polynomial_is_constant_prime (P : ℤ[X]) 
  (h1 : ∀ n : ℕ, nat.prime (P.eval (2017 * n)))
  : ∃ q : ℕ, nat.prime q ∧ ∀ x : ℤ, P.eval x = q := sorry

end polynomial_is_constant_prime_l680_680065


namespace average_goals_per_game_l680_680213

theorem average_goals_per_game
  (slices_per_pizza : ℕ := 12)
  (total_pizzas : ℕ := 6)
  (total_games : ℕ := 8)
  (total_slices : ℕ := total_pizzas * slices_per_pizza)
  (total_goals : ℕ := total_slices)
  (average_goals : ℕ := total_goals / total_games) :
  average_goals = 9 :=
by
  sorry

end average_goals_per_game_l680_680213


namespace intersection_A_B_l680_680814

open Set

def A : Set ℤ := {x | ∃ n : ℤ, x = 3 * n - 1}
def B : Set ℤ := {x | 0 < x ∧ x < 6}

theorem intersection_A_B : A ∩ B = {2, 5} := by
  sorry

end intersection_A_B_l680_680814


namespace stamp_arrangements_l680_680392

-- Define the quantities of each type of stamp
def stamps : ℕ → ℕ
| 1 := 1
| 2 := 2
| 3 := 3
| 4 := 4
| 5 := 5
| 6 := 6
| 7 := 7
| 8 := 8
| 9 := 9
| 10 := 10
| _ := 0

-- Define the function that calculates the number of different arrangements of stamps summing to 15 cents
def count_arrangements : ℕ := 
  1 + 12 + 1 + 72 + 60

theorem stamp_arrangements : count_arrangements = 146 := by
  -- Detailed proof would go here
  sorry

end stamp_arrangements_l680_680392


namespace max_dot_product_is_six_l680_680508

noncomputable def max_value_dot_product : ℝ :=
  let ellipse_eq (x y : ℝ) := (x^2 / 4) + (y^2 / 3) = 1
  let focus_F := (-1, 0)
  let dot_product (x y : ℝ) := x * (x + 1) + y^2
  let y_eq (x : ℝ) := 3 * (1 - x^2 / 4)
  let eval_dot_product (x : ℝ) := dot_product x (sqrt (y_eq x))
  if hmax : ∃ x, ∀ y, ellipse_eq x y → eval_dot_product x = 6 then 6 else 0

theorem max_dot_product_is_six : max_value_dot_product = 6 := 
  by 
    sorry

end max_dot_product_is_six_l680_680508


namespace distance_apart_after_3_hours_l680_680011

-- Definitions derived from conditions
def Ann_speed (hour : ℕ) : ℕ :=
  if hour = 1 then 6 else if hour = 2 then 8 else 4

def Glenda_speed (hour : ℕ) : ℕ :=
  if hour = 1 then 8 else if hour = 2 then 5 else 9

-- The total distance function for a given skater
def total_distance (speed : ℕ → ℕ) : ℕ :=
  speed 1 + speed 2 + speed 3

-- Ann's total distance skated
def Ann_total_distance : ℕ := total_distance Ann_speed

-- Glenda's total distance skated
def Glenda_total_distance : ℕ := total_distance Glenda_speed

-- The total distance between Ann and Glenda after 3 hours
def total_distance_apart : ℕ := Ann_total_distance + Glenda_total_distance

-- Proof statement (without the proof itself; just the goal declaration)
theorem distance_apart_after_3_hours : total_distance_apart = 40 := by
  sorry

end distance_apart_after_3_hours_l680_680011


namespace simplify_question_l680_680975

def simplify_complex_division (a b c d : ℤ) : ℂ :=
  let numerator : ℂ := ⟨a, b⟩ * ⟨c, -d⟩
  let denominator : ℂ := ⟨c, d⟩ * ⟨c, -d⟩
  numerator / denominator

theorem simplify_question :
  simplify_complex_division 6 8 3 4 = - (14 / 25 : ℂ) + (48 / 25 : ℂ) * complex.I :=
by
  sorry

end simplify_question_l680_680975


namespace a_3_is_one_sum_first_60_terms_l680_680101

noncomputable def a : ℕ → ℤ 
| 1       := 1
| (n + 1) := 2 * n - 1 - (-1)^n * a n

noncomputable def sum60 : ℤ :=
(List.range 60).sum 

theorem a_3_is_one : a 3 = 1 :=
sorry

theorem sum_first_60_terms : sum60 = 1830 :=
sorry

end a_3_is_one_sum_first_60_terms_l680_680101


namespace expression_value_is_7_l680_680871

def expression : Type := (ℕ → ℕ → ℕ) → (ℕ → ℕ → ℕ) → (ℕ → ℕ → ℕ) → ℕ

def eval_expression (a b c : expression) : ℕ :=
list.foldr ($$ \n m, eval n + m) $ (list.zip a b)

theorem expression_value_is_7 : 
  ∃ (f g h : ℕ → ℕ → ℕ),
    (f = (+) ∨ f = (-) ∨ f = (λ x y, x / y)) ∧ 
    (g = (+) ∨ g = (-) ∨ g = (λ x y, x / y)) ∧ 
    (h = (+) ∨ h = (-) ∨ h = (λ x y, x / y)) ∧ 
    f ≠ g ∧ f ≠ h ∧ g ≠ h ∧ 
    eval_expression f g h = 7 :=
begin
  sorry
end

end expression_value_is_7_l680_680871


namespace exists_even_acquaintances_l680_680958

-- Define a structure for representing mutual acquaintances
structure Person :=
(knows : Person → Prop)

def has_even_number_of_acquaintances {n : ℕ} (people : Fin n → Person) (i : Fin n) : Prop :=
  even (Finset.filter (people i).knows (Finset.univ (Fin n))).card

theorem exists_even_acquaintances :
  ∃ i j : Fin 50, i ≠ j ∧ has_even_number_of_acquaintances (fun i => Person) i ∧ has_even_number_of_acquaintances (fun i => Person) j := 
sorry

end exists_even_acquaintances_l680_680958


namespace next_fastest_time_l680_680917

theorem next_fastest_time (speed_john : ℝ) (distance : ℝ) (john_wins_by : ℝ) :
  speed_john = 15 → distance = 5 → john_wins_by = 3 → 
  let time_john := distance / speed_john in
  let time_john_min := time_john * 60 in
  let next_fastest_time := time_john_min + john_wins_by in
  next_fastest_time = 23 :=
by
  intros h1 h2 h3
  let time_john := distance / speed_john
  let time_john_min := time_john * 60
  let next_fastest_time := time_john_min + john_wins_by
  have h4 : time_john = 1 / 3 := by 
    rw [h1, h2]
    exact div_self (by norm_num : (15 : ℝ) ≠ 0)
  have h5 : time_john_min = 20 := by
    rw [h4]
    exact (one_div 3 * 60).symm
  have h6 : next_fastest_time = 23 := by
    rw [h3, h5]
    exact add_comm _ _ 
  exact h6

end next_fastest_time_l680_680917


namespace number_of_whole_numbers_with_cube_roots_less_than_8_l680_680858

theorem number_of_whole_numbers_with_cube_roots_less_than_8 :
  ∃ (n : ℕ), (∀ (x : ℕ), (1 ≤ x ∧ x < 512) → x ≤ n) ∧ n = 511 := 
sorry

end number_of_whole_numbers_with_cube_roots_less_than_8_l680_680858


namespace positive_whole_numbers_with_cube_root_less_than_8_l680_680862

theorem positive_whole_numbers_with_cube_root_less_than_8 :
  { n : ℕ | n > 0 ∧ n < 512 }.card = 511 :=
by
  sorry

end positive_whole_numbers_with_cube_root_less_than_8_l680_680862


namespace parking_ways_l680_680639

-- Definitions for conditions
variables (T : Type) [fintype T] [decidable_eq T]
variables (trains : fin 5 → T)
variables (track : fin 5 → T)
variable (A : T) -- Express train A
variable (B : T) -- Freight train B
variable (options : list (fin 5)) -- Parking options

-- Acceptable tracks for trains
def valid_parking_options_A : list (fin 5) := [0, 1, 2, 4] -- A cannot park on track 3 (index 2)
def valid_parking_options_B : list (fin 5) := [1, 2, 3, 4] -- B cannot park on track 1 (index 0)

-- Mathematical problem translated to Lean 4 statement
theorem parking_ways : 
  (∃ (f : T → fin 5) (g : fin 5 → T),
    (∀ t, t ∈ [A, B] → g (f t) = t) ∧
    (f A ∈ valid_parking_options_A) ∧ 
    (f B ∈ valid_parking_options_B) ∧ 
    (fintype.card {t // f t ∈ options}) = (fintype.card (fin 5))) → 
  sum (fun p : list (fin 5) * list (fin 5), (length p.1) * (length p.2)) = 78 :=
by sorry

end parking_ways_l680_680639


namespace terminal_side_condition_sufficient_not_necessary_l680_680124

-- Creating the context and definitions required
def is_first_or_second_quadrant (θ : ℝ) : Prop := 
(theta ≥ 0 ∧ theta < π)

def sin_positive (θ : ℝ) : Prop := (Real.sin θ > 0)

def terminal_in_first_or_second_quadrant_implies_sin_positive (θ : ℝ) : Prop :=
  is_first_or_second_quadrant θ ∧ (θ = 0 ∨ θ = π / 2)

theorem terminal_side_condition_sufficient_not_necessary (α : ℝ) 
  (h₀: terminal_in_first_or_second_quadrant_implies_sin_positive α) :
  (is_first_or_second_quadrant α → sin_positive α) ∧
  ¬ (sin_positive α → is_first_or_second_quadrant α) :=
by
  sorry

end terminal_side_condition_sufficient_not_necessary_l680_680124


namespace log_base_value_l680_680471

theorem log_base_value (a b : ℝ) (h1 : 0 < a) (h2 : a ≠ 1)
  (h3 : ∀ x y : ℝ, y = log a (x + b) → 
    ((-1, 0) = (x, y) ∨ (0, 1) = (x, y))) : 
  log b a = 1 := 
by sorry

end log_base_value_l680_680471


namespace rita_remaining_amount_l680_680970

theorem rita_remaining_amount :
  let num_dresses := 5
  let num_pants := 3
  let num_jackets := 4
  let cost_per_dress := 20
  let cost_per_pants := 12
  let cost_per_jacket := 30
  let additional_cost := 5
  let initial_amount := 400
  let total_cost := num_dresses * cost_per_dress + num_pants * cost_per_pants + num_jackets * cost_per_jacket + additional_cost
  remaining_amount = initial_amount - total_cost
  in remaining_amount = 139 :=
by
  let num_dresses := 5
  let num_pants := 3
  let num_jackets := 4
  let cost_per_dress := 20
  let cost_per_pants := 12
  let cost_per_jacket := 30
  let additional_cost := 5
  let initial_amount := 400
  let total_cost := num_dresses * cost_per_dress + num_pants * cost_per_pants + num_jackets * cost_per_jacket + additional_cost
  let remaining_amount := initial_amount - total_cost
  show remaining_amount = 139
  sorry

end rita_remaining_amount_l680_680970


namespace functional_eq_solution_l680_680193

theorem functional_eq_solution (f : ℕ → ℕ) (h : ∀ m n : ℕ, f (m + f n) = f m + n) : ∀ n, f n = n := 
by
  sorry

end functional_eq_solution_l680_680193


namespace number_of_three_digit_numbers_value_of_89th_item_l680_680083

open List

def digits : Finset ℕ := {1, 2, 3, 4, 5, 6}

def three_digit_numbers : Finset (List ℕ) :=
  Finset.univ.binder (λ _ ∈ digits.to_list.permutations.filter (λ l, l.length = 3))

/-- Prove the number of three-digit numbers using the digits 1, 2, 3, 4, 5, and 6 without repetition is 120. -/
theorem number_of_three_digit_numbers : (three_digit_numbers.card = 120) :=
sorry

/-- Prove the value of the 89th item in the ascending sequence of three-digit numbers is 526. -/
theorem value_of_89th_item : 
  (three_digit_numbers.to_list.qsort (≤) !! 88 = [5, 2, 6]) :=
sorry

end number_of_three_digit_numbers_value_of_89th_item_l680_680083


namespace figure_area_correct_l680_680938

noncomputable def area_of_figure (M : set (ℝ × ℝ)) : ℝ :=
  30 * Real.pi - 5 * Real.sqrt 3

def condition_M (M : set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ M ↔ ∃ (a b : ℝ),
    (x - a)^2 + (y - b)^2 ≤ 10 ∧
    a^2 + b^2 ≤ min (-6 * a - 2 * b) 10

theorem figure_area_correct (M : set (ℝ × ℝ)) (hM : condition_M M) :
  area_of_figure M = 30 * Real.pi - 5 * Real.sqrt 3 :=
sorry

end figure_area_correct_l680_680938


namespace problem_1_problem_2_l680_680785

noncomputable def f (x : ℝ) : ℝ := (1 / (9 * (Real.sin x)^2)) + (4 / (9 * (Real.cos x)^2))

theorem problem_1 (x : ℝ) (hx : 0 < x ∧ x < Real.pi / 2) : f x ≥ 1 := 
sorry

theorem problem_2 (x : ℝ) : x^2 + |x-2| + 1 ≥ 3 ↔ (x ≤ 0 ∨ x ≥ 1) :=
sorry

end problem_1_problem_2_l680_680785


namespace triangle_side_lengths_inequality_iff_l680_680345

theorem triangle_side_lengths_inequality_iff :
  {x : ℕ | 7 < x^2 ∧ x^2 < 17} = {3, 4} :=
by
  sorry

end triangle_side_lengths_inequality_iff_l680_680345


namespace exists_grid_line_dividing_grid_without_cutting_domino_l680_680818

-- Definition of a 6x6 grid and 18 dominoes covering the grid without overlap.
def grid := fin 6 × fin 6
def domino := fin 18 → (grid × grid)
def covers_all_squares (d : domino) := ∀ (s : grid), ∃ (i : fin 18), (s = d i.1) ∨ (s = d i.2)

theorem exists_grid_line_dividing_grid_without_cutting_domino (d : domino) (h_cover : covers_all_squares d) :
  ∃ l : grid → bool, (∀ (i : fin 18), l (d i).1 = l (d i).2) :=
begin
  sorry
end

end exists_grid_line_dividing_grid_without_cutting_domino_l680_680818


namespace miles_driven_l680_680908

theorem miles_driven (years_driving : ℕ) (miles_per_four_months : ℕ) (four_month_groups_per_year : ℕ) : 
  years_driving = 9 ∧ miles_per_four_months = 37000 ∧ four_month_groups_per_year = 3 → 
  let miles_per_year := miles_per_four_months * four_month_groups_per_year in
  let total_miles := miles_per_year * years_driving in
  total_miles = 999000 :=
begin
  intros h,
  rcases h with ⟨h1, h2, h3⟩,
  let miles_per_year := miles_per_four_months * four_month_groups_per_year,
  let total_miles := miles_per_year * years_driving,
  sorry
end

end miles_driven_l680_680908


namespace number_of_elements_in_B_l680_680112

-- Define the set A
def A : set ℤ := {0, 1, 2}

-- Define the set B
def B : set ℤ := {z | ∃ x ∈ A, ∃ y ∈ A, z = x - y}

-- The theorem stating the number of elements in set B
theorem number_of_elements_in_B : (B.to_finset.card = 5) :=
sorry

end number_of_elements_in_B_l680_680112


namespace double_sum_equality_l680_680724

theorem double_sum_equality : 
  (∑ n in (rangeFrom 3), ∑ k in finset.range (n-2) \ k -> k + 2, (k^2 : ℝ) / 3^(n+k)) = 729 / 17576 :=
sorry

end double_sum_equality_l680_680724


namespace BD_range_l680_680171

noncomputable def quadrilateral_BD (AB BC CD DA : ℕ) (BD : ℤ) :=
  AB = 7 ∧ BC = 15 ∧ CD = 7 ∧ DA = 11 ∧ (9 ≤ BD ∧ BD ≤ 17)

theorem BD_range : 
  ∀ (AB BC CD DA : ℕ) (BD : ℤ),
  quadrilateral_BD AB BC CD DA BD → 
  9 ≤ BD ∧ BD ≤ 17 :=
by
  intros AB BC CD DA BD h
  cases h
  -- We would then prove the conditions
  sorry

end BD_range_l680_680171


namespace candy_distribution_l680_680052

theorem candy_distribution :
  (∑ r in finset.range 7 \ finset.range 2, ∑ b in finset.range (9 - r) \ finset.range 1,
    nat.choose 8 r * nat.choose (8 - r) b * 2 ^ (8 - r - b)) = 4096 :=
by
  sorry

end candy_distribution_l680_680052


namespace value_of_x_l680_680296

theorem value_of_x (m n : ℝ) (z x : ℝ) (hz : z ≠ 0) (hx : x = m * (n / z) ^ 3) (hconst : 5 * (16 ^ 3) = m * (n ^ 3)) (hz_const : z = 64) : x = 5 / 64 :=
by
  -- proof omitted
  sorry

end value_of_x_l680_680296


namespace john_cuts_his_grass_to_l680_680544

theorem john_cuts_his_grass_to (growth_rate monthly_cost annual_cost cut_height : ℝ)
  (h : ℝ) : 
  growth_rate = 0.5 ∧ monthly_cost = 100 ∧ annual_cost = 300 ∧ cut_height = 4 →
  h = 2 := by
  intros conditions
  sorry

end john_cuts_his_grass_to_l680_680544


namespace total_sample_size_is_200_l680_680260

variable (A B C : ℕ) (sample_size_largest_population : ℕ)
variable (ratio1 ratio2 ratio3 : ℕ)
variable (total_population_ratio : ℕ := ratio1 + ratio2 + ratio3)

-- Given conditions
def population_ratio_condition : Prop := ratio1 = 2 ∧ ratio2 = 3 ∧ ratio3 = 5
def largest_population_sampling : Prop := sample_size_largest_population = 100

-- Proving that the total sample size is 200 given the conditions
theorem total_sample_size_is_200 (h_ratio : population_ratio_condition) (h_sampling : largest_population_sampling) : 
  (sample_size_largest_population * total_population_ratio) / ratio3 = 200 := 
by 
  sorry

end total_sample_size_is_200_l680_680260


namespace part1_part2_l680_680189

variable {n : ℕ}

def Sn (n : ℕ) (a : ℕ → ℕ) : ℕ := n * a n - n^2 + n

def a (n : ℕ) : ℕ := 2 * n + 1

noncomputable def bn (n : ℕ) : ℕ := (-1) ^ (n + 1) * (a n + a (n + 1)) / (a n * a (n + 1))

noncomputable def Tn (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), bn i

theorem part1 (H : Sn n a = Sn n (λ n, 2 * n + 1))  : 
  ∀ n, a n = 2 * n + 1 := by
  sorry

theorem part2 (H : Sn n a = Sn n (λ n, 2 * n + 1))  : 
  ∀ n, Tn n = (1 / 3) + (-1) ^ (n + 1) / (2 * n + 3) := by
  sorry

end part1_part2_l680_680189


namespace average_class_size_correct_l680_680638

def students_by_age : ℕ → ℕ
| 3 := 13
| 4 := 20
| 5 := 15
| 6 := 22
| 7 := 18
| 8 := 25
| 9 := 30
| 10 := 16
| _ := 0

def class1_students := students_by_age 3 + students_by_age 4
def class2_students := students_by_age 5
def class3_students := students_by_age 6 + students_by_age 7
def class4_students := students_by_age 8
def class5_students := students_by_age 9 + students_by_age 10

def total_students := class1_students + class2_students + class3_students + class4_students + class5_students
def number_of_classes := 5
def average_class_size := total_students / number_of_classes

theorem average_class_size_correct :
  average_class_size = 31.8 :=
by
  -- Skip the proof for now
  sorry

end average_class_size_correct_l680_680638


namespace equation_of_perpendicular_bisector_l680_680872

theorem equation_of_perpendicular_bisector 
  (A B : ℝ × ℝ) 
  (h₁ : ∃ l : (ℝ × ℝ) → Prop, l A ∧ l B)
  (h₂ : (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1)
  (h₃ : (A.1^2 / 9 + A.2^2 / 5 = 1) ∧ (B.1^2 / 9 + B.2^2 / 5 = 1) : 
  5 * (A.1 + B.1) + 9 * (A.2 + B.2) = 14 :=
sorry

end equation_of_perpendicular_bisector_l680_680872


namespace find_a_value_l680_680479

open Set

theorem find_a_value :
  ∃ a : ℤ, 
    (let A := {x : ℤ | x^2 - a * x + a^2 - 19 = 0},
         B := {x : ℤ | x^2 - 5 * x + 6 = 0},
         C := {x : ℤ | x^2 + 2 * x - 8 = 0} in
    (A ∩ B ≠ ∅) ∧ (A ∩ C = ∅)) ∧ (a = -2) :=
sorry

end find_a_value_l680_680479


namespace carol_additional_cupcakes_l680_680081

-- Define the initial number of cupcakes Carol made
def initial_cupcakes : ℕ := 30

-- Define the number of cupcakes Carol sold
def sold_cupcakes : ℕ := 9

-- Define the total number of cupcakes Carol wanted to have
def total_cupcakes : ℕ := 49

-- Calculate the number of cupcakes Carol had left after selling
def remaining_cupcakes : ℕ := initial_cupcakes - sold_cupcakes

-- The number of additional cupcakes Carol made can be defined and proved as follows:
theorem carol_additional_cupcakes : initial_cupcakes - sold_cupcakes + 28 = total_cupcakes :=
by
  -- left side: initial_cupcakes (30) - sold_cupcakes (9) + additional_cupcakes (28) = total_cupcakes (49)
  sorry

end carol_additional_cupcakes_l680_680081


namespace fraction_of_board_shaded_is_one_fourth_l680_680894

def totalArea : ℕ := 16
def shadedTopLeft : ℕ := 4
def shadedBottomRight : ℕ := 4
def fractionShaded (totalArea shadedTopLeft shadedBottomRight : ℕ) : ℚ :=
  (shadedTopLeft + shadedBottomRight) / totalArea

theorem fraction_of_board_shaded_is_one_fourth :
  fractionShaded totalArea shadedTopLeft shadedBottomRight = 1 / 4 := by
  sorry

end fraction_of_board_shaded_is_one_fourth_l680_680894


namespace three_subsets_equal_sum_l680_680066

theorem three_subsets_equal_sum (n : ℕ) (h1 : n ≡ 0 [MOD 3] ∨ n ≡ 2 [MOD 3]) (h2 : 5 ≤ n) :
  ∃ (A B C : Finset ℕ), A ∪ B ∪ C = Finset.range (n + 1) ∧
                        A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ C ∩ A = ∅ ∧
                        A.sum id = B.sum id ∧ B.sum id = C.sum id ∧ C.sum id = A.sum id :=
sorry

end three_subsets_equal_sum_l680_680066


namespace game_winning_starting_numbers_count_l680_680216

theorem game_winning_starting_numbers_count : 
  ∃ win_count : ℕ, (win_count = 6) ∧ 
                  ∀ n : ℕ, (1 ≤ n ∧ n < 10) → 
                  (n = 1 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9) ↔ 
                  ((∃ m, (2 * n ≤ m ∧ m ≤ 3 * n) ∧ m < 2007)  → 
                   (∃ k, (2 * m ≤ k ∧ k ≤ 3 * m) ∧ k ≥ 2007) = false) := 
sorry

end game_winning_starting_numbers_count_l680_680216


namespace maximum_area_of_triangle_OAB_l680_680188

noncomputable def maximum_area_triangle (a b : ℝ) : ℝ :=
  if 2 * a + b = 5 ∧ a > 0 ∧ b > 0 then (1 / 2) * a * b else 0

theorem maximum_area_of_triangle_OAB : 
  (∀ (a b : ℝ), 2 * a + b = 5 ∧ a > 0 ∧ b > 0 → (1 / 2) * a * b ≤ 25 / 16) :=
by
  sorry

end maximum_area_of_triangle_OAB_l680_680188


namespace number_of_ordered_pairs_l680_680027
open Complex

def is_real (z : ℂ) : Prop := z.im = 0

def power_i_is_real (n : ℕ) : Prop :=
  is_real (Complex.i ^ n)

theorem number_of_ordered_pairs :
  (∃ n, ∃ pairs, pairs = (counting_pairs_with_conditions) n ∧ n = 1036) :=
by 
  sorry

def counting_pairs_with_conditions : ℕ :=
  -- Add detailed definition here and count pairs based on conditions 1 and 2
  sorry

end number_of_ordered_pairs_l680_680027


namespace min_max_S_l680_680941

open Real

theorem min_max_S (a b c : ℝ) (h_a : 0 ≤ a) (h_b : 0 ≤ b) (h_c : 0 ≤ c) :
  let S := sqrt (a * b / ((b + c) * (c + a))) + sqrt (b * c / ((a + c) * (b + a))) + sqrt (c * a / ((b + c) * (b + a)))
  in 1 ≤ S ∧ S ≤ 3 / 2 :=
by
  let S := sqrt (a * b / ((b + c) * (c + a))) + sqrt (b * c / ((a + c) * (b + a))) + sqrt (c * a / ((b + c) * (b + a)))
  -- Proof steps skipped, only statement provided
  sorry

end min_max_S_l680_680941


namespace common_chord_length_l680_680484

noncomputable def dist_to_line (P : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * P.1 + b * P.2 + c) / Real.sqrt (a^2 + b^2)

theorem common_chord_length
  (x y : ℝ)
  (h1 : (x-2)^2 + (y-1)^2 = 10)
  (h2 : (x+6)^2 + (y+3)^2 = 50) :
  (dist_to_line (2, 1) 2 1 0 = Real.sqrt 5) →
  2 * Real.sqrt 5 = 2 * Real.sqrt 5 :=
by
  sorry

end common_chord_length_l680_680484


namespace side_length_of_square_l680_680648

-- Define the conditions
def stamps_count : ℕ := 300
def stamp_width : ℕ := 4
def stamp_height : ℕ := 3
def stamp_area := stamp_width * stamp_height

-- Declare the theorem
theorem side_length_of_square : (stamp_area * stamps_count = (60 : ℕ) ^ 2) :=
by
  -- Compute the area covered by the stamps
  have total_area : ℕ := stamp_area * stamps_count
  -- Verify the side length of the square computes correctly
  have side_length : ℕ := 60
  -- Apply the given conditions
  show total_area = side_length ^ 2,
  -- Rewriting total_area using the definition
  rw [total_area],
  -- Verification
  sorry

end side_length_of_square_l680_680648


namespace intersecting_lines_at_1_1_l680_680568

theorem intersecting_lines_at_1_1 (m : ℝ) :
  (∀ x y : ℝ, y = -2 * x + 3 → y = m * x + 1 → (x, y) = (1, 1)) → m = 0 :=
by
  intro h
  have h1 := h 1 1
  specialize h1 (by refl)
  cases h1 
  sorry

end intersecting_lines_at_1_1_l680_680568


namespace inverse_function_l680_680420

theorem inverse_function (x : ℝ) (hx : x > 1) : ∃ y : ℝ, x = 2^y + 1 ∧ y = Real.logb 2 (x - 1) :=
sorry

end inverse_function_l680_680420


namespace negation_of_proposition_p_l680_680447

-- Define the proposition p
def proposition_p : Prop := ∀ (n : ℕ), ¬prime (2^n - 2)

-- Statement of negation of proposition p
theorem negation_of_proposition_p : ¬proposition_p ↔ ∃ (n : ℕ), prime (2^n - 2) := by
  sorry

end negation_of_proposition_p_l680_680447


namespace study_time_in_minutes_l680_680915

theorem study_time_in_minutes :
  let day1_hours := 2
  let day2_hours := 2 * day1_hours
  let day3_hours := day2_hours - 1
  let total_hours := day1_hours + day2_hours + day3_hours
  total_hours * 60 = 540 :=
by
  let day1_hours := 2
  let day2_hours := 2 * day1_hours
  let day3_hours := day2_hours - 1
  let total_hours := day1_hours + day2_hours + day3_hours
  sorry

end study_time_in_minutes_l680_680915


namespace plane_through_points_and_perpendicular_l680_680766

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def plane_eq (A B C D : ℝ) (P : Point3D) : Prop :=
  A * P.x + B * P.y + C * P.z + D = 0

def vector_sub (P Q : Point3D) : Point3D :=
  ⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩

def cross_product (u v : Point3D) : Point3D :=
  ⟨u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x⟩

def is_perpendicular (normal1 normal2 : Point3D) : Prop :=
  normal1.x * normal2.x + normal1.y * normal2.y + normal1.z * normal2.z = 0

theorem plane_through_points_and_perpendicular
  (P1 P2 : Point3D)
  (A B C D : ℝ)
  (n_perp : Point3D)
  (normal1_eq : n_perp = ⟨2, -1, 4⟩)
  (eqn_given : plane_eq 2 (-1) 4 7 P1)
  (vec := vector_sub P1 P2)
  (n := cross_product vec n_perp)
  (eqn : plane_eq 11 (-10) (-9) (-33) P1) :
  (plane_eq 11 (-10) (-9) (-33) P2 ∧ is_perpendicular n n_perp) :=
sorry

end plane_through_points_and_perpendicular_l680_680766


namespace depletion_rate_l680_680326

theorem depletion_rate (initial_value final_value : ℝ) (years: ℕ) (r : ℝ) 
  (h1 : initial_value = 2500)
  (h2 : final_value = 2256.25)
  (h3 : years = 2)
  (h4 : final_value = initial_value * (1 - r) ^ years) :
  r = 0.05 :=
by
  sorry

end depletion_rate_l680_680326


namespace sum_series_eq_l680_680729

theorem sum_series_eq :
  (∑ n from 3 to ∞, ∑ k from 1 to n - 2, k^2 / 3^(n+k)) = 405 / 20736 :=
by
  sorry

end sum_series_eq_l680_680729


namespace find_value_of_expression_l680_680782

theorem find_value_of_expression
  (a b : ℝ)
  (h₁ : a = 4 + Real.sqrt 15)
  (h₂ : b = 4 - Real.sqrt 15)
  (h₃ : ∀ x : ℝ, (x^3 - 9 * x^2 + 9 * x = 1) → (x = a ∨ x = b ∨ x = 1))
  : (a / b) + (b / a) = 62 := sorry

end find_value_of_expression_l680_680782


namespace calculate_total_selling_price_l680_680705

noncomputable def total_selling_price (cost_price1 cost_price2 cost_price3 profit_percent1 profit_percent2 profit_percent3 : ℝ) : ℝ :=
  let sp1 := cost_price1 + (profit_percent1 / 100 * cost_price1)
  let sp2 := cost_price2 + (profit_percent2 / 100 * cost_price2)
  let sp3 := cost_price3 + (profit_percent3 / 100 * cost_price3)
  sp1 + sp2 + sp3

theorem calculate_total_selling_price :
  total_selling_price 550 750 1000 30 25 20 = 2852.5 :=
by
  -- proof omitted
  sorry

end calculate_total_selling_price_l680_680705


namespace largest_valid_3_digit_number_l680_680288

-- Define a predicate that encapsulates the conditions 
def is_geometric_sequence (a b c : ℕ) : Prop := a * b = b * b ∧ b * c = c * c

-- Define a predicate for distinct digits
def distinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Define what it means to be a valid three-digit number with the required properties
def valid_3_digit_number (n : ℕ) : Prop :=
  let d1 := n / 100 in
  let d2 := (n % 100) / 10 in
  let d3 := n % 10 in
  100 ≤ n ∧ n < 1000 ∧ d1 ≠ 0 ∧ d2 ≠ 0 ∧ d3 ≠ 0 ∧ distinct d1 d2 d3 ∧ is_geometric_sequence d1 d2 d3

-- Prove the main statement
theorem largest_valid_3_digit_number : ∃ n, valid_3_digit_number n ∧ (∀ m, valid_3_digit_number m → m ≤ n) :=
by
  use 842
  sorry

end largest_valid_3_digit_number_l680_680288


namespace no_real_roots_of_equation_l680_680382

theorem no_real_roots_of_equation : ∀ (x : ℝ), x + sqrt (x + 1) ≠ 6 :=
by
  sorry

end no_real_roots_of_equation_l680_680382


namespace min_value_z_l680_680157

variable {x y : ℝ}

def constraint1 (x y : ℝ) : Prop := x + y ≤ 3
def constraint2 (x y : ℝ) : Prop := x - y ≥ -1
def constraint3 (y : ℝ) : Prop := y ≥ 1

theorem min_value_z (x y : ℝ) 
  (h1 : constraint1 x y) 
  (h2 : constraint2 x y) 
  (h3 : constraint3 y) 
  (hx_pos : x > 0) 
  (hy_pos : y > 0) : 
  ∃ x y, x > 0 ∧ y ≥ 1 ∧ x + y ≤ 3 ∧ x - y ≥ -1 ∧ (∀ x' y', x' > 0 ∧ y' ≥ 1 ∧ x' + y' ≤ 3 ∧ x' - y' ≥ -1 → (y' / x' ≥ y / x)) ∧ y / x = 1 / 2 := 
sorry

end min_value_z_l680_680157


namespace range_of_a_exists_x_l680_680158

theorem range_of_a_exists_x (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 3, 2 * x - x ^ 2 ≥ a) ↔ a ≤ 1 := 
sorry

end range_of_a_exists_x_l680_680158


namespace smallest_n_l680_680078

noncomputable def exists_x (n : ℕ) := 
  ∃ (x : ℕ → ℝ), (∑ i in finset.range n, x i) = 1500 ∧ (∑ i in finset.range n, (x i)^6) = 1139062500

theorem smallest_n : ∃ (n : ℕ), n > 0 ∧ exists_x n ∧ ∀ (m : ℕ), m < n → ¬ exists_x m := 
by
  sorry

end smallest_n_l680_680078


namespace arithmetic_seq_num_terms_l680_680444

theorem arithmetic_seq_num_terms
  (a_n : ℕ → ℕ) -- the arithmetic sequence
  (d : ℕ) -- common difference
  (k : ℕ) -- half the number of terms
  (h1 : d = 2)
  (h2 : ∑ i in finset.range k, a_n (2 * i + 1) = 15) -- sum of odd-numbered terms
  (h3 : ∑ i in finset.range k, a_n (2 * (i+1)) = 35) -- sum of even-numbered terms
  : 2 * k = 20 :=
sorry

end arithmetic_seq_num_terms_l680_680444


namespace one_prime_p_10_14_l680_680863

theorem one_prime_p_10_14 :
  ∃! (p : ℕ), Prime p ∧ Prime (p + 10) ∧ Prime (p + 14) :=
sorry

end one_prime_p_10_14_l680_680863


namespace probability_sum_is_seven_l680_680661

/--
When two cubic dice are thrown at the same time,
the probability that the sum of the numbers obtained is 7 is \(1/6\).
-/
theorem probability_sum_is_seven :
  let num_faces := 6
  let total_outcomes := num_faces * num_faces
  let favorable_outcomes := 6
  let probability := (favorable_outcomes : ℚ) / total_outcomes
  probability = 1 / 6 := by
  let num_faces := 6
  let total_outcomes := num_faces * num_faces
  let favorable_outcomes := 6
  have probability : ℚ := favorable_outcomes / total_outcomes
  show probability = 1 / 6, from sorry

end probability_sum_is_seven_l680_680661


namespace weight_of_a_is_75_l680_680672

theorem weight_of_a_is_75 (a b c d e : ℕ) 
  (h1 : (a + b + c) / 3 = 84) 
  (h2 : (a + b + c + d) / 4 = 80) 
  (h3 : e = d + 3) 
  (h4 : (b + c + d + e) / 4 = 79) : 
  a = 75 :=
by
  -- Proof omitted
  sorry

end weight_of_a_is_75_l680_680672


namespace whole_number_between_l680_680716

theorem whole_number_between (M : ℕ) (h : 5 < M / 4 ∧ M / 4 < 5.5) : M = 21 := 
sorry

end whole_number_between_l680_680716


namespace calculate_fourth_quarter_points_l680_680367

variable (W1 W2 W3 W4 L1 : ℕ)

-- Conditions
-- 1. At the end of the first quarter, the winning team had double the points of the losing team.
def condition1 := W1 = 2 * L1 

-- 2. At the end of the second quarter, the winning team had 10 more points than it started with.
def condition2 := W2 = W1 + 10

-- 3. At the end of the third quarter, the winning team had 20 more points than the number it had in the second quarter.
def condition3 := W3 = W2 + 20

-- 4. The total points the winning team scored in the game was 80.
def condition4 := W1 + W2 + W3 + W4 = 80

-- 5. The losing team had 10 points in the first quarter.
def condition5 := L1 = 10

theorem calculate_fourth_quarter_points :
  condition1 W1 L1 ∧ condition2 W2 W1 ∧ condition3 W3 W2 ∧ condition4 W1 W2 W3 W4 ∧ condition5 L1 → W4 = 30 :=
by {
  intro h,
  have h1 := h.1,
  have h2 := h.2.1,
  have h3 := h.2.2.1,
  have h4 := h.2.2.2.1,
  have h5 := h.2.2.2.2,
  sorry
}

end calculate_fourth_quarter_points_l680_680367


namespace imaginary_part_of_complex_number_l680_680256

noncomputable def r : ℂ := 2
noncomputable def θ : ℂ := real.pi / 4
noncomputable def n : ℂ := 5
noncomputable def z : ℂ := r * (complex.cos θ + complex.sin θ * complex.I)

theorem imaginary_part_of_complex_number : 
  complex.im (z^n) = -16 * real.sqrt 2 := 
by
  sorry

end imaginary_part_of_complex_number_l680_680256


namespace triangle_inequality_l680_680902

namespace Geometry

-- Definitions of required points and lengths.
variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Definition of angles at points.
variables {angle_A : ℝ} {angle_B : ℝ} {angle_C : ℝ}
variables (AB BC CD : ℝ)

-- Conditions provided as hypothesis.
axiom angle_B_equals_2_angle_C : angle_B = 2 * angle_C
axiom point_D_on_ray_BA : Exists (λ D, D ∈ line (B, A) ∧ dist A C = dist B D)

open_locale classical

theorem triangle_inequality {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (h1 : angle_B = 2 * angle_C) (h2 : Exists (λ D, D ∈ line (B, A) ∧ dist A C = dist B D)) :
  dist A B + dist B C > dist C D :=
sorry

end Geometry

end triangle_inequality_l680_680902


namespace prove_a_sqrt2_rational_l680_680708

variable (M : Set ℝ) (hM : ∃ s : Finset ℝ, (s.card = 2003) ∧ (∀ x ∈ s, x ∈ M))
variable (h : ∀ a b ∈ M, a ≠ b → IsRational (a^2 + b * Real.sqrt 2))

theorem prove_a_sqrt2_rational (a : ℝ) (ha : a ∈ M) : IsRational (a * Real.sqrt 2) :=
  sorry

end prove_a_sqrt2_rational_l680_680708


namespace determine_right_triangle_l680_680506

theorem determine_right_triangle (a b c : ℕ) :
  (∀ c b, (c + b) * (c - b) = a^2 → c^2 = a^2 + b^2) ∧
  (∀ A B C, A + B = C → C = 90) ∧
  (a = 3^2 ∧ b = 4^2 ∧ c = 5^2 → a^2 + b^2 ≠ c^2) ∧
  (a = 5 ∧ b = 12 ∧ c = 13 → a^2 + b^2 = c^2) → 
  ( ∃ x y z : ℕ, x = a ∧ y = b ∧ z = c ∧ x^2 + y^2 ≠ z^2 )
:= by
  sorry

end determine_right_triangle_l680_680506


namespace bobby_candy_chocolate_diff_l680_680369

/-- Bobby initially had 250 pieces of candy and 175 pieces of chocolate.
He ate 38 pieces of candy, then 36 more, and then shared 12 pieces with his friends.
He also ate 16 pieces of chocolate and bought 28 more.
Prove how many more pieces of candy than chocolate Bobby had left. -/
theorem bobby_candy_chocolate_diff :
  let initial_candy := 250
  let initial_chocolate := 175
  let candy_eaten := 38 + 36 + 12
  let chocolate_eaten := 16
  let chocolate_bought := 28
  let remaining_candy := initial_candy - candy_eaten
  let remaining_chocolate := initial_chocolate - chocolate_eaten + chocolate_bought
  abs (remaining_candy - remaining_chocolate) = 23 :=
by
  sorry

end bobby_candy_chocolate_diff_l680_680369


namespace find_number_l680_680668

theorem find_number (x : ℝ) (h : x - (3/5) * x = 56) : x = 140 :=
sorry

end find_number_l680_680668


namespace sum_of_digits_p_l680_680559

def sum_of_digits (x : ℕ) : ℕ :=
  x.digits.sum

def is_valid_element (n : ℕ) : Prop :=
  sum_of_digits n = 10 ∧ 0 ≤ n ∧ n < 10 ^ 6

def T : set ℕ := {n | is_valid_element n}

def p : ℕ := set.card T

theorem sum_of_digits_p : sum_of_digits p = 27 :=
  sorry

end sum_of_digits_p_l680_680559


namespace trigonometric_identity_l680_680117

theorem trigonometric_identity
  (α : ℝ) (h1 : tan α = 3) (h2 : 0 < α ∧ α < Real.pi / 2) :
  Real.sin (2 * α) + Real.cos (Real.pi - α) = (6 - Real.sqrt 10) / 10 :=
by
  sorry

end trigonometric_identity_l680_680117


namespace bc_product_l680_680090
noncomputable theory

open real

def vec_a (x : ℝ) : ℝ × ℝ := (cos x + sqrt 3 * sin x, 1)
def vec_b (y x : ℝ) : ℝ × ℝ := (y, cos x)
def parallel (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, a = (k * b.1, k * b.2)
def area_triangle (a b c : ℝ) := sqrt 3 / 4

variables {A B C : ℝ}
variables {a b c x y : ℝ}

theorem bc_product :
  (vec_a x).snd ≠ 0 →
  parallel (vec_a x) (vec_b y x) →
  f (A/2) = 3 →
  area_triangle a b c = sqrt 3 →
  b * c = 2 :=
by
  sorry

end bc_product_l680_680090


namespace moles_KCl_formed_l680_680771

-- Given conditions
def moles_NH4Cl : ℕ := 3
def moles_KOH : ℕ := 3
def reaction_ratio : ℕ := 1 -- The ratio 1:1:1 from the balanced equation NH4Cl + KOH → KCl + NH3 + H2O

-- Theorem to prove
theorem moles_KCl_formed : (moles_NH4Cl = moles_KOH) → (moles_NH4Cl / reaction_ratio = 3) := by
  intro h
  rw h
  simp [reaction_ratio]
  sorry

end moles_KCl_formed_l680_680771


namespace area_ratio_triangle_PQR_ABC_l680_680159

noncomputable def area (A B C : ℝ×ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1 / 2) * (x1*y2 + x2*y3 + x3*y1 - x1*y3 - x2*y1 - x3*y2)

theorem area_ratio_triangle_PQR_ABC {A B C P Q R : ℝ×ℝ} 
  (h1 : dist A B + dist B C + dist C A = 1)
  (h2 : dist A P + dist P Q + dist Q B + dist B C + dist C A = 1)
  (h3 : dist P Q + dist Q R + dist R P = 1)
  (h4 : P.1 <= A.1 ∧ A.1 <= Q.1 ∧ Q.1 <= B.1) :
  area P Q R / area A B C > 2 / 9 :=
by
  sorry

end area_ratio_triangle_PQR_ABC_l680_680159


namespace angle_CAD_in_equilateral_triangle_l680_680891

open EuclideanGeometry

-- Define the equilateral triangle ABC
theorem angle_CAD_in_equilateral_triangle (A B C : Point) (H D : Point) :
  (equilateral_triangle A B C) ∧
  (is_altitude_of B H A C) ∧
  (B ∈ Line B H) ∧
  (segment_length B D = segment_length A B) →
  (angle_eq_at A C D (75)) ∨ (angle_eq_at A C D (15)) := 
by
  sorry

end angle_CAD_in_equilateral_triangle_l680_680891


namespace trig_identity_example_l680_680058

theorem trig_identity_example :
  (∀ x ∈ (Set.Icc 0 90), ∃ (cos_x sin_x : ℝ), cos_x = cos x ∧ sin_x = sin x) →
  (∃ y ∈ (Set.Icc 0 90), ∀ sin_y, cos_y, y = 50 → sin_y = sin y ∧ cos_y = cos y) →
  ∀ θ : ℝ, θ = 10 → 
  ∀ Δ : ℝ, Δ = 30 → 
  (let cos10 := cos 10
       sin10 := sin 10
       sin50 := sin 50
       cos50 := cos 50 in
   (cos10 + sqrt 3 * sin10) / (sqrt (1 - sin50 ^ 2)) = 2) :=
begin
  sorry, -- Proof is omitted
end

end trig_identity_example_l680_680058


namespace miles_driven_l680_680907

theorem miles_driven (years_driving : ℕ) (miles_per_four_months : ℕ) (four_month_groups_per_year : ℕ) : 
  years_driving = 9 ∧ miles_per_four_months = 37000 ∧ four_month_groups_per_year = 3 → 
  let miles_per_year := miles_per_four_months * four_month_groups_per_year in
  let total_miles := miles_per_year * years_driving in
  total_miles = 999000 :=
begin
  intros h,
  rcases h with ⟨h1, h2, h3⟩,
  let miles_per_year := miles_per_four_months * four_month_groups_per_year,
  let total_miles := miles_per_year * years_driving,
  sorry
end

end miles_driven_l680_680907


namespace number_of_integer_solutions_l680_680557

-- Define the given conditions and problem.
def floor (x : ℝ) : ℤ := Int.floor x

theorem number_of_integer_solutions :
  (finset.card { x : ℤ | floor (-77.66 * (x : ℝ)) = floor (-77.66) * x + 1 }) = 3 :=
sorry

end number_of_integer_solutions_l680_680557


namespace incircle_and_excircles_l680_680899

-- Definitions of the triangles and circles
variable (ABC : Type) [Triangle ABC]
variable (ω : Incircle ABC) (ω1 ω2 ω3 : Excircle ABC)

-- Radii of the circles
variable (r r1 r2 r3 : ℝ)

-- Conditions
axiom incircle_radius : ω.radius = r
axiom excircle1_radius : ω1.radius = r1
axiom excircle2_radius : ω2.radius = r2
axiom excircle3_radius : ω3.radius = r3

-- The theorem to prove
theorem incircle_and_excircles 
  (ABC : Type) [Triangle ABC]
  (ω : Incircle ABC) (ω1 ω2 ω3 : Excircle ABC)
  (r r1 r2 r3 : ℝ)
  (incircle_radius : ω.radius = r)
  (excircle1_radius : ω1.radius = r1)
  (excircle2_radius : ω2.radius = r2)
  (excircle3_radius : ω3.radius = r3) :
  sqrt(r1 * r2) + sqrt(r2 * r3) + sqrt(r3 * r1) = r :=
sorry

end incircle_and_excircles_l680_680899


namespace angle_PCA_eq_twenty_l680_680609

open Real 

noncomputable def calculateAnglePCA (A B P : Type) 
  (angle_A angle_B angle_PAB angle_PBA angle_ABC : ℝ) (h_iso : angle_A = angle_B)
  (h_angle_A : angle_A = 40) (h_angle_B : angle_B = 40) 
  (h_angle_PAB : angle_PAB = 30) (h_angle_PBA : angle_PBA = 20) : ℝ :=
  (let angle_C := 180 - angle_A - angle_B in
   let angle_APB := 180 - angle_PAB - angle_PBA in
   let angle_CB := angle_B - angle_PBA in
   let angle_PCB := 180 - angle_CB - angle_APB in
   angle_C - angle_PCB)

theorem angle_PCA_eq_twenty (A B P : Type) 
  (angle_A angle_B angle_PAB angle_PBA angle_ABC : ℝ) 
  (h_iso : angle_A = angle_B)
  (h_angle_A : angle_A = 40) 
  (h_angle_B : angle_B = 40) 
  (h_angle_PAB : angle_PAB = 30) 
  (h_angle_PBA : angle_PBA = 20) : 
  calculateAnglePCA A B P angle_A angle_B angle_PAB angle_PBA angle_ABC h_iso h_angle_A h_angle_B h_angle_PAB h_angle_PBA = 20 :=
by
  sorry

end angle_PCA_eq_twenty_l680_680609


namespace volume_of_solid_l680_680268

theorem volume_of_solid :
  ∀ (x y z : ℝ),
    (x^2 + y^2 + z^2 = 10*x - 40*y + 8*z) →
    (∃ (V : ℝ), V = 12348 * Real.pi) := 
by
  intros x y z h
  use 12348 * Real.pi
  sorry

end volume_of_solid_l680_680268


namespace problem_proof_l680_680466

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + (1 / Real.sqrt (2 - x))
noncomputable def g (x : ℝ) : ℝ := x^2 + 1

def A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def B : Set ℝ := {y | y ≥ 1}
def CU_B : Set ℝ := {y | y < 1}
def U : Set ℝ := Set.univ

theorem problem_proof :
  (∀ x, x ∈ A ↔ -1 ≤ x ∧ x < 2) ∧
  (∀ y, y ∈ B ↔ y ≥ 1) ∧
  (A ∩ CU_B = {x | -1 ≤ x ∧ x < 1}) :=
by
  sorry

end problem_proof_l680_680466


namespace friend_selling_price_l680_680328

-- Definitions and conditions
def original_cost_price : ℝ := 51724.14

def loss_percentage : ℝ := 0.13
def gain_percentage : ℝ := 0.20

def selling_price_man (CP : ℝ) : ℝ := (1 - loss_percentage) * CP
def selling_price_friend (SP1 : ℝ) : ℝ := (1 + gain_percentage) * SP1

-- Prove that the friend's selling price is 54,000 given the conditions
theorem friend_selling_price :
  selling_price_friend (selling_price_man original_cost_price) = 54000 :=
by
  sorry

end friend_selling_price_l680_680328


namespace image_relative_velocity_l680_680008

-- Definitions of the constants
def f : ℝ := 0.2
def x : ℝ := 0.5
def vt : ℝ := 3

-- Lens equation
def lens_equation (f x y : ℝ) : Prop :=
  (1 / x) + (1 / y) = 1 / f

-- Image distance
noncomputable def y (f x : ℝ) : ℝ :=
  1 / (1 / f - 1 / x)

-- Derivative of y with respect to x
noncomputable def dy_dx (f x : ℝ) : ℝ :=
  (f^2) / (x - f)^2

-- Image velocity
noncomputable def vk (vt dy_dx : ℝ) : ℝ :=
  vt * dy_dx

-- Relative velocity
noncomputable def v_rel (vt vk : ℝ) : ℝ :=
  vk - vt

-- Theorem to prove the relative velocity
theorem image_relative_velocity : v_rel vt (vk vt (dy_dx f x)) = -5 / 3 := 
by
  sorry

end image_relative_velocity_l680_680008


namespace there_exists_class_with_at_least_35_students_l680_680523

theorem there_exists_class_with_at_least_35_students
  (num_classes : ℕ) (total_students : ℕ)
  (h_classes : num_classes = 33)
  (h_students : total_students = 1150) :
  ∃ (class_size : ℕ), class_size ≥ 35 ∧ class_size * num_classes ≥ total_students :=
by
  -- We introduce our assumption which is to be contradicted
  assume h : ∀ i, i < num_classes → i < 35
  sorry

end there_exists_class_with_at_least_35_students_l680_680523


namespace alcohol_percentage_new_mixture_l680_680311

theorem alcohol_percentage_new_mixture :
  ∀ (original_volume : ℚ) (percent_alcohol : ℚ) (additional_water : ℚ),
    original_volume = 15 →
    percent_alcohol = 20 →
    additional_water = 5 →
    let original_alcohol := (percent_alcohol / 100) * original_volume in
    let original_water := original_volume - original_alcohol in
    let new_water := original_water + additional_water in
    let new_volume := original_volume + additional_water in
    (original_alcohol / new_volume) * 100 = 15 := 
by 
  intro original_volume percent_alcohol additional_water
  intros h1 h2 h3
  let original_alcohol := (percent_alcohol / 100) * original_volume
  let original_water := original_volume - original_alcohol
  let new_water := original_water + additional_water
  let new_volume := original_volume + additional_water
  sorry

end alcohol_percentage_new_mixture_l680_680311


namespace number_of_valid_colorings_l680_680914

-- Define type for Color
inductive Color
| color1
| color2

open Color

-- Define the 3x3 grid as a type
def Grid := Matrix (Fin 3) (Fin 3) (Option Color)

-- Define the adjacency constraint
def adjacent (grid : Grid) (i j : Fin 3) (i' j' : Fin 3) : Prop :=
  (i ≠ i' ∨ j ≠ j') ∧ (abs (i.val - i'.val) + abs (j.val - j'.val) = 1)

def valid_coloring (grid : Grid) : Prop :=
  ∀ i j i' j', adjacent grid i j i' j' → grid i j ≠ grid i' j'

-- The main theorem stating the number of valid ways to color the grid
theorem number_of_valid_colorings : ∃ (grid1 grid2 : Grid),
  valid_coloring grid1 ∧ valid_coloring grid2 ∧
  (∀ grid, valid_coloring grid → grid = grid1 ∨ grid = grid2) :=
sorry

end number_of_valid_colorings_l680_680914


namespace measure_four_liters_impossible_l680_680540

theorem measure_four_liters_impossible (a b c : ℕ) (h1 : a = 12) (h2 : b = 9) (h3 : c = 4) :
  ¬ ∃ x y : ℕ, x * a + y * b = c := 
by
  sorry

end measure_four_liters_impossible_l680_680540


namespace trig_relation_l680_680265

theorem trig_relation : (Real.pi/4 < 1) ∧ (1 < Real.pi/2) → Real.tan 1 > Real.sin 1 ∧ Real.sin 1 > Real.cos 1 := 
by 
  intro h
  sorry

end trig_relation_l680_680265


namespace count_numbers_with_cube_root_lt_8_l680_680847

theorem count_numbers_with_cube_root_lt_8 : 
  ∀ n : ℕ, (n > 0) → (n < 8^3) → n ≤ 8^3 - 1 :=
by
  -- We need to prove that the count of such numbers is 511
  sorry

end count_numbers_with_cube_root_lt_8_l680_680847


namespace sugar_amount_l680_680178

-- Definitions based on conditions
variables (S F B C : ℝ) -- S = amount of sugar, F = amount of flour, B = amount of baking soda, C = amount of chocolate chips

-- Conditions
def ratio_sugar_flour (S F : ℝ) : Prop := S / F = 5 / 4
def ratio_flour_baking_soda (F B : ℝ) : Prop := F / B = 10 / 1
def ratio_baking_soda_chocolate_chips (B C : ℝ) : Prop := B / C = 3 / 2
def new_ratio_flour_baking_soda_chocolate_chips (F B C : ℝ) : Prop :=
  F / (B + 120) = 16 / 3 ∧ F / (C + 50) = 16 / 2

-- Prove that the current amount of sugar is 1714 pounds
theorem sugar_amount (S F B C : ℝ) (h1 : ratio_sugar_flour S F)
  (h2 : ratio_flour_baking_soda F B) (h3 : ratio_baking_soda_chocolate_chips B C)
  (h4 : new_ratio_flour_baking_soda_chocolate_chips F B C) : 
  S = 1714 :=
sorry

end sugar_amount_l680_680178


namespace problem1_problem2_l680_680962

theorem problem1 : 
  (2 : ℝ) / (sqrt 5 - sqrt 3) = sqrt 5 + sqrt 3 :=
sorry

theorem problem2 : 
  series_sum (λ n, (4 : ℝ) / (sqrt (2 * n) + sqrt (2 * n + 2))) n = 
  2 * (sqrt (2 * n + 2) - sqrt 2) :=
sorry

end problem1_problem2_l680_680962


namespace geometric_sequence_a4_range_l680_680700

theorem geometric_sequence_a4_range
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : 0 < a 1 ∧ a 1 < 1)
  (h2 : 1 < a 1 * q ∧ a 1 * q < 2)
  (h3 : 2 < a 1 * q^2 ∧ a 1 * q^2 < 3) :
  ∃ a4 : ℝ, a4 = a 1 * q^3 ∧ 2 * Real.sqrt 2 < a4 ∧ a4 < 9 := 
sorry

end geometric_sequence_a4_range_l680_680700


namespace sufficient_not_necessary_condition_l680_680500

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x^2 > 1 → 1 / x < 1) ∧ (¬(1 / x < 1 → x^2 > 1)) :=
by sorry

end sufficient_not_necessary_condition_l680_680500


namespace brady_earns_181_l680_680370

def bradyEarnings (basic_count : ℕ) (gourmet_count : ℕ) (total_cards : ℕ) : ℕ :=
  let basic_earnings := basic_count * 70
  let gourmet_earnings := gourmet_count * 90
  let total_earnings := basic_earnings + gourmet_earnings
  let total_bonus := (total_cards / 100) * 10 + ((total_cards / 100) - 1) * 5
  total_earnings + total_bonus

theorem brady_earns_181 :
  bradyEarnings 120 80 200 = 181 :=
by 
  sorry

end brady_earns_181_l680_680370


namespace basketball_league_total_games_l680_680985

theorem basketball_league_total_games : 
  let num_divisions := 3 in
  let teams_per_division := 6 in
  let games_within_division := (teams_per_division - 1) * 2 in
  let total_teams := num_divisions * teams_per_division in
  let games_outside_division := (total_teams - teams_per_division) * 2 in
  let games_per_team := games_within_division + games_outside_division in
  total_teams * games_per_team / 2 = 306 :=
by
  sorry

end basketball_league_total_games_l680_680985


namespace biscuits_per_guest_correct_l680_680547

def flour_per_batch : ℚ := 5 / 4
def biscuits_per_batch : ℕ := 9
def flour_needed : ℚ := 5
def guests : ℕ := 18

theorem biscuits_per_guest_correct :
  (flour_needed * biscuits_per_batch / flour_per_batch) / guests = 2 := by
  sorry

end biscuits_per_guest_correct_l680_680547


namespace gervais_days_l680_680434

theorem gervais_days :
  ∀ (d : ℕ), (henridist : ℕ), (gervaisdist : ℕ),
  (gervais_avg_per_day : ℕ) (henri_total_dist : ℕ) (diff : ℕ),
  (gervais_avg_per_day = 315) →
  (henri_total_dist = 1250) →
  (diff = 305) →
  (gervaisdist = 315 * d) →
  (henridist = henri_total_dist) →
  (henridist = gervaisdist + diff) →
  d = 3 :=
by
  intros d henridist gervaisdist gervais_avg_per_day henri_total_dist diff h1 h2 h3 h4 h5 h6
  sorry

end gervais_days_l680_680434


namespace miles_driven_l680_680909

-- Definitions based on the conditions
def years : ℕ := 9
def months_in_a_year : ℕ := 12
def months_in_a_period : ℕ := 4
def miles_per_period : ℕ := 37000

-- The proof statement
theorem miles_driven : years * months_in_a_year / months_in_a_period * miles_per_period = 999000 := 
sorry

end miles_driven_l680_680909


namespace correct_function_l680_680358

open Real

def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x

def is_monotonic_decreasing (f : ℝ → ℝ) := ∀ x y : ℝ, 0 < x → x < y → f y < f x

theorem correct_function :
  (is_even (λ x : ℝ, -log (abs x)) ∧ is_monotonic_decreasing (λ x : ℝ, -log (abs x))) ∧
  (∀ f : ℝ → ℝ, f ≠ (λ x : ℝ, -log (abs x)) → ¬ (is_even f ∧ is_monotonic_decreasing f))
:= by
  sorry

end correct_function_l680_680358


namespace prime_pairs_l680_680765

-- Define the predicate to check whether a number is a prime.
def is_prime (n : Nat) : Prop := Nat.Prime n

-- Define the main theorem.
theorem prime_pairs (p q : Nat) (hp : is_prime p) (hq : is_prime q) : 
  (p^3 - q^5 = (p + q)^2) → (p = 7 ∧ q = 3) :=
by
  sorry

end prime_pairs_l680_680765


namespace number_of_incorrect_statements_l680_680924

variables {Plane : Type} [Field Plane]
variables (α β γ : Plane) (m n : Plane → Prop)

-- Define the perpendicular and parallel relationships
def perp (p q : Plane) : Prop := sorry
def parallel (p q : Plane) : Prop := sorry
def proj_perp (m n : Plane → Prop) (p : Plane) : Prop := sorry

-- Define the conditions
def stmt1 := (perp α β ∧ perp β γ) → ¬ perp α γ
def stmt2 := (parallel m α ∧ parallel n β ∧ perp α β) → ¬ perp m n
def stmt3 := (parallel α β ∧ parallel γ β) → parallel α γ
def stmt4 := (proj_perp m n γ) → ¬ perp m n

-- The main theorem
theorem number_of_incorrect_statements :
  (¬ stmt1) ∧ (¬ stmt2) ∧ stmt3 ∧ (¬ stmt4) → (D) := sorry

end number_of_incorrect_statements_l680_680924


namespace extreme_points_sum_gt_l680_680472

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - a * x^2 - Real.log x

theorem extreme_points_sum_gt (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1 / 8)
    {x₁ x₂ : ℝ} (h₂ : f x₁ a = 0) (h₃ : f x₂ a = 0) (h₄ : x₁ < x₂)
    (h₅ : 0 < x₁) (h₆ : 0 < x₂) : f x₁ a + f x₂ a > 3 - 2 * Real.log 2 := sorry

end extreme_points_sum_gt_l680_680472


namespace c_sq_minus_a_sq_divisible_by_48_l680_680504

theorem c_sq_minus_a_sq_divisible_by_48
  (a b c : ℤ) (h_ac : a < c) (h_eq : a^2 + c^2 = 2 * b^2) : 48 ∣ (c^2 - a^2) := 
  sorry

end c_sq_minus_a_sq_divisible_by_48_l680_680504


namespace line_through_A_parallel_to_BC_l680_680842

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 4, y := 0 }
def B : Point := { x := 8, y := 10 }
def C : Point := { x := 0, y := 6 }

def slope (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

def line_equation (m x1 y1 : ℝ) : String :=
  let b := y1 - m * x1
  let lhs := "x - " ++ toString m ++ "y"
  let rhs := if b == 0 then "0" else toString (-b)
  lhs ++ " - " ++ rhs ++ " = 0"

theorem line_through_A_parallel_to_BC :
  line_equation (slope B C) A.x A.y = "x - 2y - 4 = 0" :=
by
  -- The proof will go here.
  sorry

end line_through_A_parallel_to_BC_l680_680842


namespace floor_sqrt_120_l680_680396

theorem floor_sqrt_120 : (Int.floor (Real.sqrt 120) = 10) :=
by
  have h1 : 10^2 = 100 := rfl
  have h2 : 11^2 = 121 := rfl
  have h3 : 100 < 120 < 121 := by simp [h1, h2]
  have h4 : 10 < Real.sqrt 120 < 11 := by
    rw [Real.sqrt_lt, Real.sqrt_lt']
    use 120; exact h3
  exact Int.floor_eq_zero_or_incr (Real.sqrt 120) 10 (by linarith)
  sorry

end floor_sqrt_120_l680_680396


namespace calculate_expression_l680_680018

theorem calculate_expression :
  -1 ^ 4 + ((-1 / 2) ^ 2 * |(-5 + 3)|) / ((-1 / 2) ^ 3) = -5 := by
  sorry

end calculate_expression_l680_680018


namespace boy_speed_l680_680314

theorem boy_speed
  (side : ℕ)
  (time_seconds : ℕ)
  (speed_km_per_hr : ℝ)
  (h1 : side = 40)
  (h2 : time_seconds = 48)
  (h3 : speed_km_per_hr = (4 * side : ℕ) / time_seconds * 3.6) :
  speed_km_per_hr ≈ 12.00 := 
by 
  sorry

end boy_speed_l680_680314


namespace domain_of_sqrt_l680_680253

noncomputable def domain (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x, x ∈ D ↔ ∃ y, f x = y

def f (x : ℝ) : ℝ := real.sqrt (x + 1)

theorem domain_of_sqrt :
  domain f { x : ℝ | x ≥ -1 } :=
by
  intro x
  simp [f, real.sqrt_eq_iff]
  split
  intro h
  use (real.sqrt (x + 1))
  exact h
  intro h
  cases h with y hy
  rw [← hy]
  exact real.sqrt_nonneg (x + 1)
  use (x + 1)
  intro hx1
  rw [hx1]
  exact real.sqrt_nonneg (x + 1)
  sorry

end domain_of_sqrt_l680_680253


namespace number_of_quadruples_l680_680028

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem number_of_quadruples : { (a, b, c, d : Nat) |
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  factorial a * factorial b * factorial c * factorial d = factorial 24 }.card = 52 := 
sorry

end number_of_quadruples_l680_680028


namespace graph_of_g_contains_1_0_and_sum_l680_680460

noncomputable def f : ℝ → ℝ := sorry

def g (x y : ℝ) : Prop := 3 * y = 2 * f (3 * x) + 4

theorem graph_of_g_contains_1_0_and_sum :
  f 3 = -2 → g 1 0 ∧ (1 + 0 = 1) :=
by
  intro h
  sorry

end graph_of_g_contains_1_0_and_sum_l680_680460


namespace largest_value_l680_680153

theorem largest_value (x : ℝ) (hx : x = 1 / 4) :
  max (max (max (max x (x^2)) ((1/2) * x)) (1 / x)) (real.sqrt x) = 1 / x :=
by
  sorry

end largest_value_l680_680153


namespace regular_pentagon_of_complex_modulus_l680_680217

open Complex

theorem regular_pentagon_of_complex_modulus (z : ℕ → ℂ) (h₀ : ∀ i, z i ≠ 0) (h₁ : ∀ i j, abs (z i) = abs (z j)) 
    (h₂ : (∑ i in Finset.range 5, z i) = 0) (h₃ : (∑ i in Finset.range 5, z i ^ 2) = 0) : 
    ∃ θ : ℂ, (∀ i, z i = θ * exp ((2 * π * I * i : ℂ) / 5)) ∧ abs θ = abs (z 0) := 
by
  sorry

end regular_pentagon_of_complex_modulus_l680_680217


namespace simplify_f_value_of_f_l680_680119

def f (α : ℝ) : ℝ :=
  (sin (α - π / 2) * cos (3 * π / 2 + α) * tan (π - α)) /
  (tan (-π + α) * sin (π + α))

-- α is in the third quadrant
axiom α_in_third_quadrant : ∀ α : ℝ, π < α ∧ α < 3 * π / 2

-- Main theorem to prove part (I)
theorem simplify_f (α : ℝ) (h1 : π < α) (h2 : α < 3 * π / 2) :
  f(α) = -cos(α) :=
  sorry

-- Given the additional condition for part (II)
axiom cos_condition : ∀ α : ℝ, cos (α - 3*π / 2) = 1/5

-- Main theorem to prove part (II)
theorem value_of_f (α : ℝ) (h1 : π < α) (h2 : α < 3 * π / 2) (h3 : cos (α - 3 * π / 2) = 1/5) :
  f(α) = 2 * sqrt 6 / 5 :=
  sorry

end simplify_f_value_of_f_l680_680119


namespace quotient_of_division_l680_680619

theorem quotient_of_division (S L Q : ℕ) (h1 : S = 270) (h2 : L - S = 1365) (h3 : L % S = 15) : Q = 6 :=
by
  sorry

end quotient_of_division_l680_680619


namespace semicircle_area_with_isosceles_right_triangle_l680_680719

noncomputable def isosceles_right_triangle_hypotenuse (a : ℝ) : ℝ :=
  Real.sqrt (a^2 + a^2)

noncomputable def semicircle_radius_from_hypotenuse (d : ℝ) : ℝ := (1/2) * d

noncomputable def semicircle_area (r : ℝ) : ℝ := (1/2) * π * r^2

theorem semicircle_area_with_isosceles_right_triangle :
  let a := 1 in
  let d := isosceles_right_triangle_hypotenuse a in
  let r := semicircle_radius_from_hypotenuse d in
  semicircle_area r = π / 4 :=
by
  sorry

end semicircle_area_with_isosceles_right_triangle_l680_680719


namespace no_such_subset_exists_l680_680047

variable {M : Set Nat}
variable {N : Set Nat}

def condition1 (M N : Set Nat) : Prop :=
  ∀ n ∈ N, n > 1 → ∃ a b ∈ M, n = a + b

def condition2 (M : Set Nat) : Prop :=
  ∀ a b c d ∈ M, 10 < a → 10 < b → 10 < c → 10 < d →
  (a + b = c + d ↔ a = c ∨ a = d)

theorem no_such_subset_exists (M N : Set Nat) :
  ¬ (condition1 M N ∧ condition2 M) :=
sorry

end no_such_subset_exists_l680_680047


namespace conics_meet_at_origin_angle_arctan_half_l680_680562

theorem conics_meet_at_origin_angle_arctan_half :
  ∀ (a : ℝ) (C : ℝ → ℝ → Prop),
    (∀ x y, C x y ↔ (2 * y + x)^2 = a * (y + x)) →
    (∃ k : ℝ, ∀ x y, C x y → x = 0 → y = -(1 / 2) * x ∧ atan (1 / 2) = k) :=
begin
  sorry
end

end conics_meet_at_origin_angle_arctan_half_l680_680562


namespace sum_equiv_l680_680732

theorem sum_equiv {f : ℕ → ℕ → ℝ} (h : ∀ (n k : ℕ), n ≥ 3 ∧ 1 ≤ k ∧ k ≤ n - 2 → f n k = (k^2) / (3^(n+k))) :
  (∑' n=3, ∑' k=1, if h : k ≤ n - 2 then f n k else 0) = 135 / 512 :=
by sorry

end sum_equiv_l680_680732


namespace count_numbers_with_cube_root_lt_8_l680_680849

theorem count_numbers_with_cube_root_lt_8 : 
  ∀ n : ℕ, (n > 0) → (n < 8^3) → n ≤ 8^3 - 1 :=
by
  -- We need to prove that the count of such numbers is 511
  sorry

end count_numbers_with_cube_root_lt_8_l680_680849


namespace probability_between_six_and_ten_l680_680274

open Finset

def spinner1 : Finset ℕ := {1, 4, 5}
def spinner2 : Finset ℕ := {2, 3, 6}

def favorable_sums : Finset ℕ := spinner1.product spinner2 
  |>.filter (λ x, let sum := x.1 + x.2 in 6 ≤ sum ∧ sum ≤ 10) 
  |>.image (λ x, x.1 + x.2)

def all_sums : Finset ℕ := spinner1.product spinner2 |>.image (λ x, x.1 + x.2)

def probability : ℚ := favorable_sums.card / all_sums.card

theorem probability_between_six_and_ten : probability = 2 / 3 := by
  sorry

end probability_between_six_and_ten_l680_680274


namespace distance_gracie_joe_l680_680487

noncomputable def distance_between_points := Real.sqrt (5^2 + (-1)^2)
noncomputable def joe_point := Complex.mk 3 (-4)
noncomputable def gracie_point := Complex.mk (-2) (-3)

theorem distance_gracie_joe : Complex.abs (joe_point - gracie_point) = distance_between_points := by 
  sorry

end distance_gracie_joe_l680_680487


namespace count_ordered_pairs_l680_680024

theorem count_ordered_pairs :
  (∃ (n : ℕ), n = 3515 ∧
    ∀ (x y : ℕ), 1 ≤ x ∧ x < y ∧ y ≤ 150 →
    (∀ (k m : ℕ), k % 4 = x % 4 ∧ m % 4 = y % 4 →
    ((x ≡ y [MOD 4] ∨ x ≡ y + 2 [MOD 4] ∨ x + 2 ≡ y [MOD 4]) →
    is_real (complex.I ^ x + complex.I ^ y)) →
    true)) := sorry

end count_ordered_pairs_l680_680024


namespace weekly_diesel_spending_l680_680950

-- Conditions
def cost_per_gallon : ℝ := 3
def fuel_used_in_two_weeks : ℝ := 24

-- Question: Prove that Mr. Alvarez spends $36 on diesel fuel each week.
theorem weekly_diesel_spending : (fuel_used_in_two_weeks / 2) * cost_per_gallon = 36 := by
  sorry

end weekly_diesel_spending_l680_680950


namespace number_of_girls_in_study_group_l680_680342

theorem number_of_girls_in_study_group : ∀ (n : ℕ), 0 < n ∧ n ≤ 6 → 
  (∏ (i : ℕ) in (Finset.range 2), 6 - i) / 2! - (∏ (j : ℕ) in (Finset.range 2), 6 - n - j) / 2! = 12 → n = 3 :=
by
  intros
  sorry

end number_of_girls_in_study_group_l680_680342


namespace concurrence_A1C1_BD_EF_l680_680552

-- Definitions of the points and the configuration
variable (A B C D : Point)
variable (AD_parallel_BC : Parallel AD BC)
variable (AD_lt_BC : AD < BC)
variable (E F : Point)
variable (E_on_AB : OnLine E AB)
variable (F_on_CD : OnLine F CD)
variable (A1 : Point)
variable (A1_on_circle_AEF : OnCircle A1 (CircleThrough A E F))
variable (A1_on_AD : OnSegment A1 AD)
variable (C1 : Point)
variable (C1_on_circle_CEF : OnCircle C1 (CircleThrough C E F))
variable (C1_on_BC : OnSegment C1 BC)

-- Theorem statement
theorem concurrence_A1C1_BD_EF : Concurrent (LineThrough A1 C1) (LineThrough B D) (LineThrough E F) := 
sorry

end concurrence_A1C1_BD_EF_l680_680552


namespace KLMN_convex_and_area_l680_680956

-- Definitions used as conditions
variables (A B C D E F K L M N : Point)
variables (AB BC CD DA DE BF CE AF : Segment)
variables (alpha : Real)

-- Midpoints definitions
def is_midpoint (P Q R : Point) : Prop := midpoint P Q = R

-- Conditions
axiom convex_quad (ABCD : Quadrilateral) : convex ABCD
axiom on_sides (E_on_AB : E ∈ segment AB) (F_on_CD : F ∈ segment CD) : True
axioms (K_is_mid_DE : is_midpoint D E K)
       (L_is_mid_BF : is_midpoint B F L)
       (M_is_mid_CE : is_midpoint C E M)
       (N_is_mid_AF : is_midpoint A F N)

-- Questions to be proven
theorem KLMN_convex_and_area : convex (quadrilateral K L M N) ∧
                               area (quadrilateral K L M N) = (1/8) * length segment AB * length segment CD * Real.sin alpha := 
begin
  sorry
end

end KLMN_convex_and_area_l680_680956


namespace annika_hiking_distance_l680_680667

theorem annika_hiking_distance :
  ∀ (hiking_rate : ℝ) (initial_distance : ℝ) (total_time : ℝ) (remaining_time : ℝ),
  hiking_rate = 12 ∧ initial_distance = 2.75 ∧ total_time = 51 ∧ remaining_time = total_time - (initial_distance * hiking_rate)
  → initial_distance + remaining_time / (2 * hiking_rate) = 3.5 :=
by
  intros hiking_rate initial_distance total_time remaining_time
  assume h
  sorry

end annika_hiking_distance_l680_680667


namespace intersection_M_N_l680_680566

-- Define the sets M and N
def M : Set ℝ := { x : ℝ | x^2 - x - 6 < 0 }
def N : Set ℝ := { x : ℝ | 1 < x }

-- State the problem in terms of Lean definitions and theorem
theorem intersection_M_N : M ∩ N = {x : ℝ | 1 < x ∧ x < 3} :=
by
  sorry

end intersection_M_N_l680_680566


namespace train_speed_proof_l680_680712

noncomputable def speed_of_train (train_length_m : ℕ) (man_speed_kmh : ℕ) 
  (pass_time_sec : ℕ) : ℕ :=
let train_length_km := (train_length_m : ℝ) / 1000 in
let pass_time_hr := (pass_time_sec : ℝ) / 3600 in
let relative_speed := train_length_km / pass_time_hr in
relative_speed.to_nat - man_speed_kmh

theorem train_speed_proof :
  speed_of_train 110 9 4 = 90 :=
by sorry

end train_speed_proof_l680_680712


namespace quadratic_touches_x_axis_l680_680154

theorem quadratic_touches_x_axis (a : ℝ) : 
  (∃ x : ℝ, 2 * x ^ 2 - 8 * x + a = 0) ∧ (∀ y : ℝ, y^2 - 4 * a = 0 → y = 0) → a = 8 := 
by 
  sorry

end quadratic_touches_x_axis_l680_680154


namespace point_in_third_quadrant_iff_l680_680458

theorem point_in_third_quadrant_iff (m : ℝ) :
  (m - 1 < 0) ∧ (2 * m - 3 < 0) ↔ m < 1 :=
begin
  sorry
end

end point_in_third_quadrant_iff_l680_680458


namespace Sn_lt_1_In_le_Sn_l680_680201

noncomputable def v : ℕ → ℕ
| 0     := 2
| (n+1) := Nat.prod (List.ofFn (λ i => v i)) + 1

def S (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => 1 / (v i : ℚ))

theorem Sn_lt_1 (n : ℕ) : S n < 1 :=
begin
  sorry
end

theorem In_le_Sn (In : ℚ) (hIn : In < 1) (hIn_sum : ∃ l : List ℕ, (∀ x ∈ l, x ∈ (List.ofFn (λ i => v i.succ))) ∧ (l.sum⁻¹ : ℚ) = In) : In ≤ S (hIn_sum.some.length + 1) :=
begin
  sorry
end

end Sn_lt_1_In_le_Sn_l680_680201


namespace find_counterfeit_coin_l680_680279

-- Define the context of the problem
variables (coins : Fin 6 → ℝ) -- six coins represented as a function from Fin 6 to their weights
          (is_counterfeit : Fin 6 → Prop) -- a predicate indicating if the coin is counterfeit
          (real_weight : ℝ) -- the unknown weight of a real coin

-- Existence assertion for the counterfeit coin
axiom exists_counterfeit : ∃ x, is_counterfeit x

-- Define the total weights of coins 1&2 and 3&4
def weight_1_2 := coins 0 + coins 1
def weight_3_4 := coins 2 + coins 3

-- Statement of the problem
theorem find_counterfeit_coin :
  (weight_1_2 = weight_3_4 → (is_counterfeit 4 ∨ is_counterfeit 5)) ∧ 
  (weight_1_2 ≠ weight_3_4 → (is_counterfeit 0 ∨ is_counterfeit 1 ∨ is_counterfeit 2 ∨ is_counterfeit 3)) :=
sorry

end find_counterfeit_coin_l680_680279


namespace summation_proof_l680_680739

open BigOperators

theorem summation_proof :
  ∑ n in finset.range (∞).filter (λ n, n ≥ 3), ∑ k in finset.range (n - 2).filter (λ k, k ≥ 1), k^2 * (3:ℝ) ^ (- (n + k)) = 5 / 72 := 
by 
  sorry

end summation_proof_l680_680739


namespace eval_expression_l680_680404

theorem eval_expression (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ x) (hz' : z ≠ -x) :
  ((x / (x + z) + z / (x - z)) / (z / (x + z) - x / (x - z)) = -1) :=
by
  sorry

end eval_expression_l680_680404


namespace euler_line_equation_l680_680572

-- Given vertices of the triangle
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (5, 0)
def C : ℝ × ℝ := (2, 4)

-- Define centroid computation
def centroid (A B C : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  ((x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3)

-- Define circumcenter computation (placeholder for a more detailed definition)
def circumcenter (A B C : ℝ × ℝ) : ℝ × ℝ :=
  -- Specific circumcenter calculation for given triangle
  (5 / 2, 5 / 4)

-- Define the proof statement
theorem euler_line_equation : 
  let G := centroid A B C
  let W := circumcenter A B C
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧ (a, b, c) = (1, 2, -5) ∧ 
  ∀ x y : ℝ, G.1 + 2 * G.2 - 5 = 0 ∧ W.1 + 2 * W.2 - 5 = 0 :=
by
  let G := centroid A B C
  let W := circumcenter A B C
  use [1, 2, -5]
  constructor
  · exact Or.inl (by norm_num)
  constructor
  · rfl
  · intro x y
    constructor
    · sorry  -- Proof that G satisfies the line equation
    · sorry  -- Proof that W satisfies the line equation

end euler_line_equation_l680_680572


namespace cosine_of_angle_between_planes_l680_680459

def n1 : ℝ × ℝ × ℝ := (3, 2, 1)
def n2 : ℝ × ℝ × ℝ := (2, 0, -1)
def cos_angle_between_planes : ℝ :=
  let dot_product := (n1.1 * n2.1 + n1.2 * n2.2 + n1.3 * n2.3)
  let magnitude_n1 := Real.sqrt (n1.1^2 + n1.2^2 + n1.3^2)
  let magnitude_n2 := Real.sqrt (n2.1^2 + n2.2^2 + n2.3^2)
  dot_product / (magnitude_n1 * magnitude_n2)

theorem cosine_of_angle_between_planes :
  cos_angle_between_planes = Real.sqrt 70 / 14 := by
  sorry

end cosine_of_angle_between_planes_l680_680459


namespace no_solution_inequality_C_l680_680293

theorem no_solution_inequality_C : ¬∃ x : ℝ, 2 * x - x^2 > 5 := by
  -- There is no need to include the other options in the Lean theorem, as the proof focuses on the condition C directly.
  sorry

end no_solution_inequality_C_l680_680293


namespace rahul_matches_l680_680304

variable (m : ℕ)

/-- Rahul's current batting average is 51, and if he scores 78 runs in today's match,
    his new batting average will become 54. Prove that the number of matches he had played
    in this season before today's match is 8. -/
theorem rahul_matches (h1 : (51 * m) / m = 51)
                      (h2 : (51 * m + 78) / (m + 1) = 54) : m = 8 := by
  sorry

end rahul_matches_l680_680304


namespace work_done_l680_680674

-- Define constants and conditions
def m : ℝ := 3.0 * 10^3 -- mass in kg
def H : ℝ := 650 * 10^3 -- height in meters
def R₃ : ℝ := 6380 * 10^3 -- radius of the Earth in meters
def g : ℝ := 10 -- acceleration due to gravity in m/s^2

-- Prove that the work done A is approximately 1.77 × 10^10 J
theorem work_done : 
  let A := m * g * R₃^2 * (1 / R₃ - 1 / (R₃ + H)) in
  abs (A - 1.77 * 10^10) < 10^8 :=
by 
  have A := m * g * R₃^2 * (1 / R₃ - 1 / (R₃ + H))
  sorry

end work_done_l680_680674


namespace quotient_of_polynomial_l680_680427

theorem quotient_of_polynomial (x : ℤ) :
  (x^6 + 8) = (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) + 9 :=
by { sorry }

end quotient_of_polynomial_l680_680427


namespace smallest_positive_period_and_monotonically_increasing_interval_find_value_of_cos_2_x0_l680_680468

noncomputable def f (x : ℝ) := 2 * real.sqrt 3 * real.cos (π / 2 - x) * real.cos x - real.sin x ^ 2 + real.cos x ^ 2

theorem smallest_positive_period_and_monotonically_increasing_interval :
  ∀ k : ℤ, (∀ x : ℝ, f x = f (x + π)) ∧ ∀ x : ℝ, x ∈ set.Icc (k * π - π / 3) (k * π + π / 6) → 
    (∀ a b : ℝ, a ∈ set.Icc (k * π - π / 3) (k * π + π / 6) → b ∈ set.Icc (k * π - π / 3) (k * π + π / 6) → a < b → f a < f b) :=
sorry

theorem find_value_of_cos_2_x0 
  (x0 : ℝ) (h1 : f x0 = 6 / 5) (h2 : x0 ∈ set.Icc (π / 4) (π / 2)) :
  real.cos (2 * x0) = (3 - 4 * real.sqrt 3) / 10 :=
sorry

end smallest_positive_period_and_monotonically_increasing_interval_find_value_of_cos_2_x0_l680_680468


namespace determinant_of_matrix_l680_680017

theorem determinant_of_matrix :
  let A := Matrix.of_list 3 3 [[2, -1, 1], [3, 2, 2], [1, -2, 1]] in
  A.det = 5 :=
by
  sorry

end determinant_of_matrix_l680_680017


namespace exists_fixed_point_subset_l680_680553

-- Definitions of set and function f with the required properties
variable {α : Type} [DecidableEq α]
variable (H : Finset α)
variable (f : Finset α → Finset α)

-- Conditions
axiom increasing_mapping (X Y : Finset α) : X ⊆ Y → f X ⊆ f Y
axiom range_in_H (X : Finset α) : f X ⊆ H

-- Statement to prove
theorem exists_fixed_point_subset : ∃ H₀ ⊆ H, f H₀ = H₀ :=
sorry

end exists_fixed_point_subset_l680_680553


namespace vet_fees_cat_result_l680_680718

-- Given conditions
def vet_fees_dog : ℕ := 15
def families_dogs : ℕ := 8
def families_cats : ℕ := 3
def vet_donation : ℕ := 53

-- Mathematical equivalency in Lean
noncomputable def vet_fees_cat (C : ℕ) : Prop :=
  (1 / 3 : ℚ) * (families_dogs * vet_fees_dog + families_cats * C) = vet_donation

-- Prove the vet fees for cats are 13 using above conditions
theorem vet_fees_cat_result : ∃ (C : ℕ), vet_fees_cat C ∧ C = 13 :=
by {
  use 13,
  sorry
}

end vet_fees_cat_result_l680_680718


namespace evaluate_log_example_l680_680757

def log_example (x y : ℝ) : ℝ := log x / log y

theorem evaluate_log_example : 
  log_example 64.sqrt 2 = 13 / 2 := by
  sorry

end evaluate_log_example_l680_680757


namespace positive_solution_exists_l680_680425

noncomputable def equation_solution : ℝ :=
let y := Real.roots (λ y, y^3 - y^2 + y) in
if y.exists_pos then y.some else 0

theorem positive_solution_exists : ∃ x > 0, equation_solution ^ 3 = x :=
by
  have y := equation_solution
  have hy := by sorry -- Proof of y satisfies y^3 - y^2 + y = 0
  have x := y ^ 3
  use x
  split
  · exact Real.exp_pos (Real.log y) -- Showing y > 0 implies exp(log y) > 0
  · exact hy -- Showing y^3 = x

end positive_solution_exists_l680_680425


namespace product_inequality_l680_680486

variable (a b : ℕ → ℕ)
variable (n : ℕ)

-- Define the conditions
def condition1 : Prop := 
  ∑ i in finset.range (n + 1), a i = ∑ i in finset.range (n + 1), b i

def condition2 : Prop :=
  (∀ i j, i ≤ j → a i ≤ a j) ∧
  (∀ i j, i ≤ j → b j ≥ b i) ∧ 
  (a n ≥ b n) ∧ (b 1 ≥ a 1)

def condition3 : Prop :=
  ∀ i j, 1 ≤ i → i < j → j ≤ n → a j - a i ≥ b j - b i

-- Define the main theorem statement
theorem product_inequality 
  (cond1 : condition1 a b n) 
  (cond2 : condition2 a b n) 
  (cond3 : condition3 a b n) : 
  (∏ i in finset.range (n + 1), a i) ≤ (∏ i in finset.range (n + 1), b i) := 
sorry

end product_inequality_l680_680486


namespace terminal_angle_rotation_cos_value_l680_680360

noncomputable def angle := 30
noncomputable def rotated_angle := 3 * 360 + angle 

axiom sin_val : ∀ α : ℝ, (Real.sin (-π / 2 - α) = -1 / 3)
axiom tan_negative : ∀ α : ℝ, (Real.tan α < 0)

theorem terminal_angle_rotation : rotated_angle = 1110 := by
  -- Proof omitted
  sorry

theorem cos_value (α : ℝ) : 
  sin_val α → tan_negative α → Real.cos (3 * π / 2 + α) = -2 * Real.sqrt 2 / 3 := by
  -- Proof omitted
  sorry

end terminal_angle_rotation_cos_value_l680_680360


namespace unique_digits_product_l680_680675

theorem unique_digits_product :
  ∃ (A B C D E F : ℕ),
  {A, B, C, D, E, F} = {1, 2, 3, 4, 5, 6} ∧
  (10 * A + B) * C = 100 * D + 10 * E + F ∧
  C = 3 :=
by sorry

end unique_digits_product_l680_680675


namespace math_problem_l680_680826

theorem math_problem
  (a b c : ℝ)
  (h1 : sqrt (2 * a - 1) = 3 ∨ sqrt (2 * a - 1) = -3)
  (h2 : real.cbrt (3 * a + b - 9) = 2)
  (h3 : c = ⌊real.sqrt 7⌋) :
  a = 5 ∧ b = 2 ∧ c = ⌊real.sqrt 7⌋ ∧ sqrt (a + 2 * b + c) = real.sqrt 11 :=
by
  sorry

end math_problem_l680_680826


namespace problem1_problem2_l680_680463

-- Define the curves C1, C2, and C3 in their parametric forms.
def C1 (t : ℝ) : ℝ × ℝ := (-4 + Real.cos t, 3 + Real.sin t)
def C2 (θ : ℝ) : ℝ × ℝ := (8 * Real.cos θ, 3 * Real.sin θ)
def C3 (t : ℝ) : ℝ × ℝ := (3 + 2 * t, -2 + t)

-- Transform C1 to its standard form (circle equation).
def C1_standard (x y : ℝ) : Prop := (x + 4)^2 + (y - 3)^2 = 1

-- Transform C2 to its standard form (ellipse equation).
def C2_standard (x y : ℝ) : Prop := (x^2) / 64 + (y^2) / 9 = 1

-- Given point P on C1 with parameter t = π/2
def P : ℝ × ℝ := C1 (Real.pi / 2)

-- Point Q on C2 with parameter θ
def Q (θ : ℝ) : ℝ × ℝ := C2 θ

-- Midpoint M of P and Q
def M (θ : ℝ) : ℝ × ℝ := ((-4 + 8 * Real.cos θ) / 2, (4 + 3 * Real.sin θ) / 2)

-- Define the distance formula from a point to a line
def distance (x y : ℝ) : ℝ := abs (x - 2 * y - 7) / Real.sqrt (1 + 4)

-- Problem statements
theorem problem1 (x y : ℝ) : (∃ t, C1 t = (x, y)) ↔ C1_standard x y := sorry

theorem problem2 (θ : ℝ) : M θ = ((-4 + 8 * Real.cos θ) / 2, (4 + 3 * Real.sin θ) / 2) ∧
  (∀ θ, distance (M θ). fst (M θ). snd ≥ distance 0 0) ∧
  (distance (-4 + 8 * Real.cos θ) / 2 (4 + 3 * Real.sin θ) / 2 = 8 * Real.sqrt 5 / 5) := sorry

end problem1_problem2_l680_680463


namespace intersection_cardinality_l680_680481

-- Declare the sets A and B as constants
constant A : Set ℕ := {1, 3, 5}
constant B : Set ℕ := {3, 4, 5}

-- State the theorem
theorem intersection_cardinality :
  |A ∩ B| = 2 :=
sorry

end intersection_cardinality_l680_680481


namespace factor_3a3_minus_6a2_plus_3a_factor_a2_minus_b2_x_minus_y_factor_16a_plus_b_sq_minus_9a_minus_b_sq_l680_680416

-- First factorization problem
theorem factor_3a3_minus_6a2_plus_3a (a : ℝ) : 
  3 * a ^ 3 - 6 * a ^ 2 + 3 * a = 3 * a * (a - 1) ^ 2 :=
by sorry

-- Second factorization problem
theorem factor_a2_minus_b2_x_minus_y (a b x y : ℝ) : 
  a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a - b) * (a + b) :=
by sorry

-- Third factorization problem
theorem factor_16a_plus_b_sq_minus_9a_minus_b_sq (a b : ℝ) : 
  16 * (a + b) ^ 2 - 9 * (a - b) ^ 2 = (a + 7 * b) * (7 * a + b) :=
by sorry

end factor_3a3_minus_6a2_plus_3a_factor_a2_minus_b2_x_minus_y_factor_16a_plus_b_sq_minus_9a_minus_b_sq_l680_680416


namespace triangle_area_ratio_l680_680583

noncomputable def vector_sum_property (OA OB OC : ℝ × ℝ × ℝ) : Prop :=
  OA + (2 : ℝ) • OB + (3 : ℝ) • OC = (0 : ℝ × ℝ × ℝ)

noncomputable def area_ratio (S_ABC S_AOC : ℝ) : Prop :=
  S_ABC / S_AOC = 3

theorem triangle_area_ratio
    (OA OB OC : ℝ × ℝ × ℝ)
    (S_ABC S_AOC : ℝ)
    (h1 : vector_sum_property OA OB OC)
    (h2 : S_ABC = 3 * S_AOC) :
  area_ratio S_ABC S_AOC :=
by
  sorry

end triangle_area_ratio_l680_680583


namespace snack_combinations_equal_six_l680_680354

theorem snack_combinations_equal_six : ∃ (comb : ℕ), comb = nat.choose 4 2 ∧ comb = 6 :=
by
  use nat.choose 4 2
  split
  . rfl
  . sorry

end snack_combinations_equal_six_l680_680354


namespace units_digit_p_plus_five_l680_680034

/--
Let \( f(x) = ax^2 + bx + c \) be a quadratic function where \( a, b, \) and \( c \) are integers and \( x \) is a real number. 
Let \( p \) be a positive even integer with a positive units digit such that \( 10 \leq p \leq 50 \).
If the units digit of \( p^3 \) minus the units digit of \( p^2 \) is equal to 0, \( f(p^2) = f(p^3) \), 
and \( f(p) = 0 \), then the units digit of \( p + 5 \) is 1.
-/
theorem units_digit_p_plus_five 
  (a b c p : ℤ)
  (h0 : 10 ≤ p ∧ p ≤ 50)
  (h1 : ∃ n, p = 2 * (n + 1))
  (h2 : units_digit (p^3) = units_digit (p^2))
  (h3 : (a * p^2 + b * p + c = 0))
  (h4 : (f : ℤ -> ℤ) = (λ x, a * x^2 + b * x + c))
  (h5 : f(p^2) = f(p^3)) :
  units_digit (p + 5) = 1 :=
sorry

end units_digit_p_plus_five_l680_680034


namespace average_marks_increase_ratio_l680_680334

theorem average_marks_increase_ratio
  (T : ℕ)  -- The correct total marks of the class
  (n : ℕ)  -- The number of pupils in the class
  (h_n : n = 16) (wrong_mark : ℕ) (correct_mark : ℕ)  -- The wrong and correct marks
  (h_wrong : wrong_mark = 73) (h_correct : correct_mark = 65) :
  (8 : ℚ) / T = (wrong_mark - correct_mark : ℚ) / n * (n / T) :=
by
  sorry

end average_marks_increase_ratio_l680_680334


namespace flux_through_sphere_l680_680767

noncomputable def vector_field (r : ℝ^3) : ℝ^3 := r / (∥r∥^3)

noncomputable def sphere_radius_R {r : ℝ^3} (R : ℝ) : Prop := ∥r∥ = R

theorem flux_through_sphere (R : ℝ) :
  ∀ (r : ℝ^3), sphere_radius_R R r → ∮ (vector_field r) • (r / ∥r∥) = 4 * Real.pi :=
sorry

end flux_through_sphere_l680_680767


namespace problems_completed_l680_680865

theorem problems_completed (p t : ℕ) (h1 : p ≥ 15) (h2 : p * t = (2 * p - 10) * (t - 1)) : p * t = 60 := sorry

end problems_completed_l680_680865


namespace side_length_S2_l680_680323

-- Definition of the problem conditions
def larger_rectangle_width : ℕ := 3500
def larger_rectangle_height : ℕ := 2350

-- Assign variables for sides
variable (s r : ℕ)

-- Conditions
def height_eq : 2 * r + s = 2350 := sorry
def width_eq : 2 * r + 3 * s = 3500 := sorry

-- The statement to prove
theorem side_length_S2 :
  s = 575 :=
by
  have h1 : 2 * r + s = 2350 := height_eq
  have h2 : 2 * r + 3 * s = 3500 := width_eq
  -- Proof will go here using the given conditions
  sorry

end side_length_S2_l680_680323


namespace laura_blocks_l680_680185

theorem laura_blocks : 
  ∀ (num_friends num_blocks_per_friend total_blocks : ℕ), 
  num_friends = 4 →
  num_blocks_per_friend = 7 →
  total_blocks = num_friends * num_blocks_per_friend →
  total_blocks = 28 :=
by 
  intros num_friends num_blocks_per_friend total_blocks h_friends h_blocks_per_friend h_total_blocks
  rw [h_friends, h_blocks_per_friend] at h_total_blocks
  exact h_total_blocks.symm

end laura_blocks_l680_680185


namespace cubic_sum_of_roots_l680_680802

theorem cubic_sum_of_roots :
  ∀ (r s : ℝ), (r + s = 5) → (r * s = 6) → (r^3 + s^3 = 35) :=
by
  intros r s h₁ h₂
  sorry

end cubic_sum_of_roots_l680_680802


namespace n_value_l680_680876

noncomputable def h : ℝ → ℝ := λ x, Real.exp x + Real.log (x + 1) - 5

theorem n_value {n : ℤ} (hn : ∃ x : ℝ, x ∈ (n:ℝ, (n+1:ℤ)) ∧ h x = 0) : n = 1 := 
sorry

end n_value_l680_680876


namespace digit_inequality_l680_680245

theorem digit_inequality : ∃ (n : ℕ), n = 9 ∧ ∀ (d : ℕ), d < 10 → (2 + d / 10 + 5 / 1000 > 2 + 5 / 1000) → d > 0 :=
by
  sorry

end digit_inequality_l680_680245


namespace original_triangle_angles_determined_l680_680048

-- Define the angles of the formed triangle
def formed_triangle_angles : Prop := 
  52 + 61 + 67 = 180

-- Define the angles of the original triangle
def original_triangle_angles (α β γ : ℝ) : Prop := 
  α + β + γ = 180

theorem original_triangle_angles_determined :
  formed_triangle_angles → 
  ∃ α β γ : ℝ, 
    original_triangle_angles α β γ ∧
    α = 76 ∧ β = 58 ∧ γ = 46 :=
by
  sorry

end original_triangle_angles_determined_l680_680048


namespace intersect_at_B_l680_680614

theorem intersect_at_B
  (O₁ O₂ A B P Q : Point)
  (h₁ : Circle O₁ ∩ Circle O₂ = {A, B})
  (h₂ : Circle (O₁, A, O₂) ∩ Circle O₁ = {P})
  (h₃ : Circle (O₁, A, O₂) ∩ Circle O₂ = {Q}) :
  ∃ B : Point, B ∈ Line O₁ Q ∧ B ∈ Line O₂ P :=
sorry

end intersect_at_B_l680_680614


namespace sequence_element_bound_l680_680550

theorem sequence_element_bound (n : ℕ) (a : Fin n → ℝ) :
  ∃ k : ℕ, ∃ (x : Fin n → ℝ), (∀ i, x i = a i) →
  (∀ I J : Finset (Fin n), I ∩ J = ∅ ∧ I ∪ J = Finset.univ → 
  let S := ∑ i in I, x i - ∑ j in J, x j in
  let y := λ i, if i ∈ I then x i + 1 else x i - 1 in
  ∀ m, m > k → y (Fin.cast m (Fin.ofNat' i)) = x (Fin.cast m (Fin.ofNat' i))) →
  ∃ i, |x i| ≥ n / 2 :=
by
  sorry

end sequence_element_bound_l680_680550


namespace smallest_C_inequality_l680_680075

theorem smallest_C_inequality (x y z : ℝ) (h : x + y + z = -1) : 
  |x^3 + y^3 + z^3 + 1| ≤ (9/10) * |x^5 + y^5 + z^5 + 1| :=
  sorry

end smallest_C_inequality_l680_680075


namespace rotate_180_maps_segments_l680_680294

theorem rotate_180_maps_segments :
  let C := (3, -2)
  let C' := (-3, 2)
  let D := (4, -5)
  let D' := (-4, 5)
  ∃ θ : ℝ, θ = 180 ∧
  ∀ (x y : ℝ), 
    let (x', y') := (x * cos θ - y * sin θ, x * sin θ + y * cos θ) in
    (x, y) = C → (x', y') = C' ∧
    (x, y) = D → (x', y') = D' :=
by
  sorry

end rotate_180_maps_segments_l680_680294


namespace derivative_at_pi_l680_680831

def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem derivative_at_pi :
  deriv f π = -1 := by
  sorry

end derivative_at_pi_l680_680831


namespace positive_whole_numbers_cube_root_less_than_eight_l680_680851

theorem positive_whole_numbers_cube_root_less_than_eight : 
  { n : ℕ | n > 0 ∧ n < 512 }.card = 511 :=
by sorry

end positive_whole_numbers_cube_root_less_than_eight_l680_680851


namespace books_left_in_library_l680_680626

theorem books_left_in_library (initial_books : ℕ) (borrowed_books : ℕ) (left_books : ℕ) 
  (h1 : initial_books = 75) (h2 : borrowed_books = 18) : left_books = 57 :=
by
  sorry

end books_left_in_library_l680_680626


namespace range_of_a_l680_680207

noncomputable def f (x a : ℝ) : ℝ := log (abs (x + 1) + a - 1)

def A (a : ℝ) : set ℝ := { x | abs (x + 1) + a - 1 > 0 }

def B : set ℝ := { x | ∃ k : ℤ, x = 2 * k }

def C_U_A (a : ℝ) : set ℝ := { x | -2 + a ≤ x ∧ x ≤ -a }

theorem range_of_a :
  ∀ a : ℝ, (a < 1) →
    (∀ x, f x a = log (abs (x + 1) + a - 1)) →
    (B ∩ C_U_A a).to_finset.card = 2 →
    -2 < a ∧ a ≤ 0 :=
by
  intros a h1 h2 h3
  sorry

end range_of_a_l680_680207


namespace minimum_cells_to_prevent_victory_l680_680218

def is_tromino_coverable (grid : Fin 5 × Fin 5 → bool) : Prop :=
  sorry -- define what it means for a grid to be coverable by L-trominoes

theorem minimum_cells_to_prevent_victory (grid : Fin 5 × Fin 5 → bool) :
  (∀ marked_cells : Fin 25 → bool, (∑ i in Finset.univ.filter marked_cells, 1) ≥ 9) ∧
  ¬ is_tromino_coverable grid :=
sorry -- the proof that 9 is the minimum number of cells Petya must mark to prevent Vasya from winning

end minimum_cells_to_prevent_victory_l680_680218


namespace num_subsets_l680_680864

theorem num_subsets (Y : Set ℕ) :
  {1, 2, 6} ⊆ Y ∧ Y ⊆ {1, 2, 3, 4, 5, 6, 7} →
  ∃ n : ℕ, n = 16 :=
by
  intro h
  use 16
  sorry

end num_subsets_l680_680864


namespace measure_of_angle_BAD_l680_680220

noncomputable def angle_BAD (α β δ : ℝ) : ℝ := 36 -- This is used to directly state the conclusion

-- The statement with the conditions and the conclusion.
theorem measure_of_angle_BAD (A B C D E : Type) 
  (triangle_ABC : is_isosceles A B C)
  (on_BC : D ∈ line_segment B C)
  (on_AC : E ∈ line_segment A C)
  (AD_eq_AE : distance A D = distance A E)
  (angle_EDC : angle E D C = 18) 
  : angle_BAD (angle B A C) (angle A C B) (angle B C A) = 36 :=
sorry

end measure_of_angle_BAD_l680_680220


namespace dimes_total_l680_680592

def initial_dimes : ℕ := 9
def added_dimes : ℕ := 7

theorem dimes_total : initial_dimes + added_dimes = 16 := by
  sorry

end dimes_total_l680_680592


namespace temperature_decrease_l680_680526

theorem temperature_decrease (T : ℝ) 
    (h1 : T * (3 / 4) = T - 21)
    (h2 : T > 0) : 
    T = 84 := 
  sorry

end temperature_decrease_l680_680526


namespace probability_of_rolling_9_twice_is_1_over_64_l680_680651

-- Define the problem conditions
def has_eight_sides (d : ℕ) : Prop := d = 8
def sum_is_nine (a b : ℕ) : Prop := a + b = 9
noncomputable def probability_sum_9_twice : ℚ :=
  let P_one_roll := (8 : ℚ) / (64 : ℚ) in
  P_one_roll * P_one_roll

-- The main theorem statement
theorem probability_of_rolling_9_twice_is_1_over_64 (d1 d2 d3 d4 : ℕ)
  (h1 : has_eight_sides d1) (h2 : has_eight_sides d2)
  (h3 : has_eight_sides d3) (h4 : has_eight_sides d4)
  (h_sum1 : sum_is_nine d1 d2) (h_sum2 : sum_is_nine d3 d4) :
  probability_sum_9_twice = 1 / 64 := by
  -- Here would go the detailed proof steps
  sorry

end probability_of_rolling_9_twice_is_1_over_64_l680_680651


namespace total_cost_correct_l680_680488

-- Define each condition separately.
def selling_price_1 : ℝ := 120
def profit_percentage_1 : ℝ := 0.25

def selling_price_2 : ℝ := 225
def profit_percentage_2 : ℝ := 0.40

def selling_price_3 : ℝ := 450
def profit_percentage_3 : ℝ := 0.20

def selling_price_4 : ℝ := 300
def profit_percentage_4 : ℝ := 0.30

def selling_price_5 : ℝ := 600
def profit_percentage_5 : ℝ := 0.15

-- Define the cost price calculation.
def cost_price (sp : ℝ) (pp : ℝ) : ℝ := sp / (1 + pp)

-- Define total cost as the sum of all cost prices of the items.
def total_cost : ℝ := 
  (cost_price selling_price_1 profit_percentage_1) + 
  (cost_price selling_price_2 profit_percentage_2) +
  (cost_price selling_price_3 profit_percentage_3) +
  (cost_price selling_price_4 profit_percentage_4) +
  (cost_price selling_price_5 profit_percentage_5)

-- The theorem to be proved.
theorem total_cost_correct : total_cost = 1384.22 :=
by
  -- Proof will be provided here.
  sorry

end total_cost_correct_l680_680488


namespace length_of_bridge_l680_680298

-- Define the conditions
def speed : ℝ := 8 -- speed in km/hr
def time_minutes : ℝ := 15 -- time in minutes
def time_hours : ℝ := time_minutes / 60 -- convert time to hours

-- Given conditions, prove the length of the bridge
theorem length_of_bridge : speed * time_hours = 2 :=
by
  -- The proof will be provided here
  sorry

end length_of_bridge_l680_680298


namespace prove_inequalities_l680_680204

-- Definitions and initial relations
def sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 1 <= n → S (n + 1) = a (n + 1) * S n

-- The proof goal based on the given conditions
theorem prove_inequalities (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℕ) (hS : sequence a S) (hk : 3 ≤ k):
  0 ≤ a (k + 1) ∧ a (k + 1) ≤ a k ∧ a k ≤ 4 / 3 :=
begin
  sorry
end

end prove_inequalities_l680_680204


namespace rita_remaining_money_l680_680967

-- Defining the conditions
def num_dresses := 5
def price_dress := 20
def num_pants := 3
def price_pant := 12
def num_jackets := 4
def price_jacket := 30
def transport_cost := 5
def initial_amount := 400

-- Calculating the total cost
def total_cost : ℕ :=
  (num_dresses * price_dress) + 
  (num_pants * price_pant) + 
  (num_jackets * price_jacket) + 
  transport_cost

-- Stating the proof problem 
theorem rita_remaining_money : initial_amount - total_cost = 139 := by
  sorry

end rita_remaining_money_l680_680967


namespace problem_l680_680875

def f (x : ℝ) : ℝ :=
  if x < 1 then Real.log x / Real.log (1/2)
  else 2^(x - 1)

theorem problem :
  f (f (1 / 16)) = 8 :=
by
  sorry

end problem_l680_680875


namespace line_passes_through_fixed_point_l680_680627

theorem line_passes_through_fixed_point : ∀ m : ℝ, m * 1 + 0 - m = 0 := by
  assume m
  sorry

end line_passes_through_fixed_point_l680_680627


namespace count_fractions_l680_680359

def is_fraction (x : ℝ) : Prop :=
∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def problem_numbers : List ℝ :=
[1.2, Real.pi / 3, 0, -22 / 7, 1.010010001, 5, 0.353535353535...]

theorem count_fractions :
  (problem_numbers.filter is_fraction).length = 4 :=
by
  sorry

end count_fractions_l680_680359


namespace Emily_finishes_first_l680_680747

theorem Emily_finishes_first (a r : ℝ) (ha : 0 < a) (hr : 0 < r) :
  let t_david := a / r
  let t_emily := (2 * a / 3) / (3 * r / 4)
  let t_frank := (a / 2) / (r / 2)
  t_emily < t_david ∧ t_emily < t_frank :=
by {
  let t_david := a / r,
  let t_emily := (2 * a / 3) / (3 * r / 4),
  let t_frank := (a / 2) / (r / 2),
  have h_david_time : t_david = a / r, by refl,
  have h_emily_time : t_emily = 8 * a / (9 * r), by { field_simp, ring },
  have h_frank_time : t_frank = a / r, by field_simp,
  rw [h_emily_time, h_david_time, h_frank_time],
  split,
  { linarith },
  { linarith },
  sorry -- Proof left as an exercise
}

end Emily_finishes_first_l680_680747


namespace curve_type_l680_680438

theorem curve_type (k : ℝ) (hk : k > 1) :
  let A := k - 1 in
  let B := k^2 - 1 in
  (1 - k) * x^2 + y^2 = k^2 - 1 →
  set.eq (set_of (λ x y, (x^2 / A) - (y^2 / B) = 1)) { p | true } :=
  sorry

end curve_type_l680_680438


namespace inequality_proof_l680_680940

theorem inequality_proof (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) :
  (a^2 - b * c) / (2 * a^2 + b * c) + (b^2 - c * a) / (2 * b^2 + c * a) + (c^2 - a * b) / (2 * c^2 + a * b) ≤ 0 :=
sorry

end inequality_proof_l680_680940


namespace find_k_of_odd_prime_l680_680107

theorem find_k_of_odd_prime (p k : ℕ) (h_prime : Prime p) (h_odd: p % 2 = 1) (h_pos_k : 0 < k) 
(h_sqrt_int : ∃ n : ℕ, 0 < n ∧ n * n = k^2 - p * k) : 
  k = (p + 1) * (p + 1) / 4 :=
begin
  sorry
end

end find_k_of_odd_prime_l680_680107


namespace find_cost_of_article_l680_680505

-- Define the given conditions and the corresponding proof statement.
theorem find_cost_of_article
  (tax_rate : ℝ) (selling_price1 : ℝ)
  (selling_price2 : ℝ) (profit_increase_rate : ℝ)
  (cost : ℝ) : tax_rate = 0.05 →
              selling_price1 = 360 →
              selling_price2 = 340 →
              profit_increase_rate = 0.05 →
              (selling_price1 / (1 + tax_rate) - cost = 1.05 * (selling_price2 / (1 + tax_rate) - cost)) →
              cost = 57.13 :=
by sorry

end find_cost_of_article_l680_680505


namespace omega_range_l680_680258

def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

def g (ω x : ℝ) : ℝ := Real.sin (2 * ω * x - Real.pi / 6)

theorem omega_range (ω : ℝ) (h : 0 < ω) :
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 4 → g ω x = 0 → ∃! y, x = y) ↔ (7/3 : ℝ) ≤ ω ∧ ω < (13/3 : ℝ) := 
sorry

end omega_range_l680_680258


namespace geometric_sequence_problem_l680_680800

theorem geometric_sequence_problem
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : q ≠ 1)
  (h_sum : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 6)
  (h_sum_squares : a 1 ^ 2 + a 2 ^ 2 + a 3 ^ 2 + a 4 ^ 2 + a 5 ^ 2 + a 6 ^ 2 + a 7 ^ 2 = 18)
  (h_geom_seq : ∀ n : ℕ, a (n + 1) = a 1 * q ^ n) :
  a 1 - a 2 + a 3 - a 4 + a 5 - a 6 + a 7 = 3 :=
by sorry

end geometric_sequence_problem_l680_680800


namespace probability_of_two_correct_deliveries_l680_680789

noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

def probability_exactly_two_correct_deliveries : ℚ :=
  let n := 4
  let k := 2
  let pairs := choose n k
  let prob_of_specific_two_correct := (1 / 4 : ℚ) * (1 / 3) * (1 / 2)
  let total_prob := pairs * prob_of_specific_two_correct
  total_prob

theorem probability_of_two_correct_deliveries :
  probability_exactly_two_correct_deliveries = 1 / 4 :=
by
  sorry

end probability_of_two_correct_deliveries_l680_680789


namespace integral_of_sqrt_circle_quadrant_l680_680406

-- Define the integral problem and its evaluation
theorem integral_of_sqrt_circle_quadrant :
  ∫ x in 2..3, sqrt (1 - (x - 3)^2) = (π / 4) :=
by
  sorry

end integral_of_sqrt_circle_quadrant_l680_680406


namespace quadratic_solution_l680_680239

theorem quadratic_solution (x : ℝ) : x^2 - 2 * x - 3 = 0 → (x = 3 ∨ x = -1) :=
by
  sorry

end quadratic_solution_l680_680239


namespace parameter_interval_l680_680787

theorem parameter_interval (a : ℝ):
  (∃ x1 x2 : ℝ, 
    (∀ x : ℝ, Real.log 2 (2 * x^2 + (2 * a + 1) * x - 2 * a) - 
                2 * Real.log 4 (x^2 + 3 * a * x + 2 * a^2) = 0) ∧
    x1 ≠ x2 ∧ 
    x1^2 + x2^2 > 4
  ) ↔ (a < -1 ∨ (3 / 5 < a ∧ a < 1)) :=
sorry

end parameter_interval_l680_680787


namespace adults_at_zoo_l680_680678

theorem adults_at_zoo (A K : ℕ) (h1 : A + K = 254) (h2 : 28 * A + 12 * K = 3864) : A = 51 :=
sorry

end adults_at_zoo_l680_680678


namespace periodic_function_implies_rational_ratio_l680_680799

noncomputable def g (i : ℕ) (a ω θ x : ℝ) : ℝ := 
  a * Real.sin (ω * x + θ)

theorem periodic_function_implies_rational_ratio 
  (a1 a2 ω1 ω2 θ1 θ2 : ℝ) (h1 : a1 * ω1 ≠ 0) (h2 : a2 * ω2 ≠ 0)
  (h3 : |ω1| ≠ |ω2|) 
  (hf_periodic : ∃ T : ℝ, ∀ x : ℝ, g 1 a1 ω1 θ1 (x + T) + g 2 a2 ω2 θ2 (x + T) = g 1 a1 ω1 θ1 x + g 2 a2 ω2 θ2 x) :
  ∃ m n : ℤ, n ≠ 0 ∧ ω1 / ω2 = m / n :=
sorry

end periodic_function_implies_rational_ratio_l680_680799


namespace carter_has_255_cards_l680_680571

-- Definition of the number of baseball cards Marcus has.
def marcus_cards : ℕ := 350

-- Definition of the number of more cards Marcus has than Carter.
def difference : ℕ := 95

-- Definition of the number of baseball cards Carter has.
def carter_cards : ℕ := marcus_cards - difference

-- Theorem stating that Carter has 255 baseball cards.
theorem carter_has_255_cards : carter_cards = 255 :=
sorry

end carter_has_255_cards_l680_680571


namespace binary_representation_of_fourteen_l680_680414

theorem binary_representation_of_fourteen :
  (14 : ℕ) = 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 0 * 2^0 :=
by
  sorry

end binary_representation_of_fourteen_l680_680414


namespace AM_bisects_BMC_l680_680095

variable (A B M C : Point) -- Points A, B, M, C
variable (AB_eq_BC : dist A B = dist B C) -- AB = BC
variable (angle_BAM_30 : angle B A M = 30) -- ∠BAM = 30°
variable (angle_ACM_150 : angle A C M = 150) -- ∠ACM = 150°
variable (ABC_convex : Convexquadvectg A B M C) -- A B M C is convex

theorem AM_bisects_BMC :
  isAngleBisector A M B C :=
by
  sorry

end AM_bisects_BMC_l680_680095


namespace construct_triangle_from_distances_l680_680036

-- Defining points and distances
variables {A B C S M O : Point}
variables (r s_0 m_0 : ℝ)

-- Assuming the given conditions
def isCircumcircle (A B C O : Point) (r : ℝ) : Prop :=
  dist A O = r ∧ dist B O = r ∧ dist C O = r

def isCentroid (S A B C : Point) : Prop :=
  dist A S = 2 / 3 * dist A ((B + C) / 2)

def isOrthocenter (M A B C : Point) : Prop :=
  ∀ L, L.isPerpendicular (A, M) (L : Point)

theorem construct_triangle_from_distances 
  (h_circ : isCircumcircle A B C O r)
  (h_centroid : dist A S = s_0)
  (h_orthocenter : dist A M = m_0) :
  ∃ (A B C O S M : Point), 
  isCircumcircle A B C O r ∧ 
  dist A S = s_0 ∧ 
  dist A M = m_0 :=
sorry

end construct_triangle_from_distances_l680_680036


namespace max_s_n_l680_680805

theorem max_s_n (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : a 1 = 7) 
  (h2 : ∀ n : ℕ, 0 < n → a (n + 1) = a n - 2) :
  ∃ n : ℕ, n = 4 ∧ S n = (finset.range n).sum (λ i, a (i+1)) ∧ ∀ k : ℕ, (finset.range k).sum (λ i, a (i+1)) ≤ S n := 
begin
  sorry
end

end max_s_n_l680_680805


namespace fuel_station_cost_l680_680885

theorem fuel_station_cost (service_cost_per_vehicle : ℝ)
                         (fuel_cost_per_liter : ℝ)
                         (num_minivans : ℕ)
                         (num_trucks : ℕ)
                         (minivan_tank_liters : ℝ)
                         (truck_percentage_bigger : ℝ) :
  service_cost_per_vehicle = 2.10 →
  fuel_cost_per_liter = 0.70 →
  num_minivans = 3 →
  num_trucks = 2 →
  minivan_tank_liters = 65 →
  truck_percentage_bigger = 1.20 →
  let truck_tank_liters := minivan_tank_liters + truck_percentage_bigger * minivan_tank_liters in
  let total_fuel_cost := (num_minivans * minivan_tank_liters * fuel_cost_per_liter)
                        + (num_trucks * truck_tank_liters * fuel_cost_per_liter) in
  let total_service_cost := (num_minivans + num_trucks) * service_cost_per_vehicle in
  total_fuel_cost + total_service_cost = 347.20 := 
by
  intros h1 h2 h3 h4 h5 h6
  let truck_tank_liters := minivan_tank_liters + truck_percentage_bigger * minivan_tank_liters
  let total_fuel_cost := (num_minivans * minivan_tank_liters * fuel_cost_per_liter)
                        + (num_trucks * truck_tank_liters * fuel_cost_per_liter)
  let total_service_cost := (num_minivans + num_trucks) * service_cost_per_vehicle
  have total_cost := total_fuel_cost + total_service_cost
  have cost_correct : total_cost = 347.20 := sorry
  exact cost_correct

end fuel_station_cost_l680_680885


namespace tangent_to_circumcircle_of_triangle_l680_680820

open Locale Classical

variables (I D A B C A1 B1 C1 : Point)
variables (hI : is_incenter I A B C)
variables (hD : on_line D B C)
variables (h_angle_AID_90 : angle I A D = 90)
variables (h_excircles_touch : 
  (excircle_opposite A touches_line I A A1) ∧
  (excircle_opposite B touches_line I B B1) ∧
  (excircle_opposite C touches_line I C C1))
variables (h_cyclic : cyclic_quadrilateral A B1 A1 C1)

theorem tangent_to_circumcircle_of_triangle :
  is_tangent (line_through A D) (circumcircle D B1 C1) :=
sorry

end tangent_to_circumcircle_of_triangle_l680_680820


namespace perfect_square_fraction_l680_680043

theorem perfect_square_fraction (n : ℤ) : 
  n < 30 ∧ ∃ k : ℤ, (n / (30 - n)) = k^2 → ∃ cnt : ℕ, cnt = 4 :=
  by
  sorry

end perfect_square_fraction_l680_680043


namespace speeds_of_persons_l680_680084

/-
Person A and Person B start from locations 6 km and 10 km away from the destination, respectively. 
The ratio of their speeds is 3:4, and Person A arrives 20 minutes earlier than Person B.
We aim to prove that the speeds of Person A and Person B are 4.5 km/h and 6 km/h, respectively.
-/
theorem speeds_of_persons
  (dA dB : ℝ) -- distances
  (h_ratio : ℕ) -- speed ratio multiplier
  (time_diff : ℝ) -- time difference in hours
  (h_A : 6 = dA) -- Person A starts 6 km away
  (h_B : 10 = dB) -- Person B starts 10 km away
  (h_speed_ratio : 3 * h_ratio = 4 * h_ratio - 1)
  (h_time_diff : 1/3 = time_diff) :
  let speed_A := 3 * (2/3) * h_ratio,
      speed_B := 4 * (2/3) * h_ratio
  in speed_A = 4.5 ∧ speed_B = 6 :=
by
  sorry

end speeds_of_persons_l680_680084


namespace sum_series_eq_l680_680730

theorem sum_series_eq :
  (∑ n from 3 to ∞, ∑ k from 1 to n - 2, k^2 / 3^(n+k)) = 405 / 20736 :=
by
  sorry

end sum_series_eq_l680_680730


namespace original_number_unique_l680_680655

theorem original_number_unique (N : ℤ) (h : (N - 31) % 87 = 0) : N = 118 :=
by
  sorry

end original_number_unique_l680_680655


namespace seq_value_at_2018_l680_680097

noncomputable def f (x : ℝ) : ℝ := sorry
def seq (a : ℕ → ℝ) : Prop :=
  a 1 = f 0 ∧ ∀ (n : ℕ), n > 0 → f (a (n + 1)) = 1 / f (-2 - a n)

theorem seq_value_at_2018 (a : ℕ → ℝ) (h_seq : seq a) : a 2018 = 4035 := 
by sorry

end seq_value_at_2018_l680_680097


namespace average_of_numbers_l680_680873

theorem average_of_numbers (x : ℝ) (h : (5 + -1 + -2 + x) / 4 = 1) : x = 2 :=
by
  sorry

end average_of_numbers_l680_680873


namespace polynomial_is_linear_l680_680441

variables {P : ℤ → ℤ} {n : ℤ}

-- Definition of the sequence based on the polynomial P and the starting point n
def sequence (P : ℤ → ℤ) (n : ℤ) : ℕ → ℤ
| 0       := n
| (k + 1) := P (sequence k)

-- Proof that for any positive integer b, there exists a term in the sequence that is a positive power of b greater than 1
theorem polynomial_is_linear
  (h1 : ∀ b : ℤ, b > 0 → ∃ k : ℕ, ∃ m : ℤ, m > 1 ∧ sequence P n k = m^b)
  (h2 : ∃ a : ℤ, P a ≠ a) :
  ∃ (a b : ℤ), P x = a * x + b :=
sorry

end polynomial_is_linear_l680_680441


namespace matrix_transformation_l680_680067

variable {α : Type*} [Field α]

def mat3x3 : Type := matrix (fin 3) (fin 3) α

def M : mat3x3 := ![![2, 0, 0], ![0, 0, 1], ![0, 1, 0]]

theorem matrix_transformation (N : mat3x3) :
  M.mul N = ![![2 * N 0 0, 2 * N 0 1, 2 * N 0 2],
              ![N 2 0, N 2 1, N 2 2],
              ![N 1 0, N 1 1, N 1 2]] :=
by
  sorry

end matrix_transformation_l680_680067


namespace simplify_expression_l680_680598

-- Defining each term in the sum
def term1  := 1 / ((1 / 3) ^ 1)
def term2  := 1 / ((1 / 3) ^ 2)
def term3  := 1 / ((1 / 3) ^ 3)

-- Sum of the terms
def terms_sum := term1 + term2 + term3

-- The simplification proof statement
theorem simplify_expression : 1 / terms_sum = 1 / 39 :=
by sorry

end simplify_expression_l680_680598


namespace interest_rate_as_percentage_l680_680912

noncomputable def car_cost : ℝ := 32000
noncomputable def down_payment : ℝ := 8000
noncomputable def loan_duration : ℕ := 48
noncomputable def monthly_payment : ℝ := 525

theorem interest_rate_as_percentage :
  let total_payments := monthly_payment * loan_duration,
      loan_amount := car_cost - down_payment,
      total_interest := total_payments - loan_amount,
      interest_rate := (total_interest / total_payments) * 100 in
  interest_rate ≈ 4.76 :=
by
  let total_payments := monthly_payment * loan_duration
  let loan_amount := car_cost - down_payment
  let total_interest := total_payments - loan_amount
  let interest_rate := (total_interest / total_payments) * 100
  have h : interest_rate ≈ 4.76 := sorry
  exact h

end interest_rate_as_percentage_l680_680912


namespace solution_set_I_range_of_a_l680_680129

-- Define the function f(x) = |x + a| - |x + 1|
def f (x a : ℝ) : ℝ := abs (x + a) - abs (x + 1)

-- Part (I)
theorem solution_set_I (a : ℝ) : 
  (f a a > 1) ↔ (a < -2/3 ∨ a > 2) := by
  sorry

-- Part (II)
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≤ 2 * a) ↔ (a ≥ 1/3) := by
  sorry

end solution_set_I_range_of_a_l680_680129


namespace pairings_without_alice_and_bob_l680_680534

theorem pairings_without_alice_and_bob (n : ℕ) (h : n = 12) : 
    ∃ k : ℕ, k = ((n * (n - 1)) / 2) - 1 ∧ k = 65 :=
by
  sorry

end pairings_without_alice_and_bob_l680_680534


namespace diagonals_bisect_each_other_l680_680264

theorem diagonals_bisect_each_other {R H S : Type} 
  (is_rectangle : (diagonals R).bisect_each_other) 
  (is_rhombus : (diagonals H).bisect_each_other) 
  (is_square : (diagonals S).bisect_each_other) :
  (diagonals R).bisect_each_other ∧ (diagonals H).bisect_each_other ∧ (diagonals S).bisect_each_other :=
by 
  sorry

end diagonals_bisect_each_other_l680_680264


namespace final_result_l680_680536

-- Definitions for the given sequence
def a (n : ℕ) : ℕ :=
  if n = 2 then 2
  else if n = 3 then 10 - a (7) -- implicitly defines that a(3) + a(7) = 10
  else sorry -- define formula according to the problem conditions

-- Hypothesis for sequence properties
axiom a_recurrence (n : ℕ) (h : n ≥ 2) : a (n + 1) = 2 * a (n) - a (n - 1)

-- Definition of b_k
def b (k n : ℕ) : ℕ := if k ≤ n ∧ n ≤ 2^k then 1 else 0

-- Sum of first k terms T_k
def T (k : ℕ) : ℕ :=
  ∑ i in Finset.range (k + 1), (2^i - i + 1)

-- The main theorem we want to prove
theorem final_result (k : ℕ) : T (k) = 2^(k + 1) - k * (k + 1) / 2 + k - 2 :=
by
  sorry

end final_result_l680_680536


namespace sum_of_squares_of_distances_l680_680284

variables (a b : ℝ)
variables (A1 B1 C1 A B C : ℝ)
variables (G : ℝ)

-- The centroid of the equilateral triangle with side a
def centroid_equilateral := G

-- The centroid of the isosceles right triangle with legs b
def centroid_isosceles := G

-- Distance relation function 
def distance_sum_squares := 
  (A1 - A)^2 + (A1 - B)^2 + (A1 - C)^2 + 
  (B1 - A)^2 + (B1 - B)^2 + (B1 - C)^2 + 
  (C1 - A)^2 + (C1 - B)^2 + (C1 - C)^2

theorem sum_of_squares_of_distances :
  centroid_equilateral = centroid_isosceles → 
  distance_sum_squares = 3 * a^2 + 4 * b^2 :=
sorry

end sum_of_squares_of_distances_l680_680284


namespace gage_skate_time_l680_680790

theorem gage_skate_time (day1_4 day5_8 min_needed avg desired_avg : ℕ) 
  (h1 : day1_4 = 80) (h2 : day5_8 = 105) (h3 : avg = 100) 
  (h4 : desired_avg = 9 * 100 - (4 * day1_4 + 4 * day5_8)) 
  : min_needed = 160 :=
by
  rw [h1, h2] at h4
  simp [desired_avg, avg] at h4
  exact h4

end gage_skate_time_l680_680790


namespace Ned_washed_shirts_l680_680951

-- Definitions based on conditions
def short_sleeve_shirts : ℕ := 9
def long_sleeve_shirts : ℕ := 21
def total_shirts : ℕ := short_sleeve_shirts + long_sleeve_shirts
def not_washed_shirts : ℕ := 1
def washed_shirts : ℕ := total_shirts - not_washed_shirts

-- Statement to prove
theorem Ned_washed_shirts : washed_shirts = 29 := by
  sorry

end Ned_washed_shirts_l680_680951


namespace find_ab_inequality_f_l680_680834

noncomputable def f (a b : ℝ) (x : ℝ) := (2 * a * Real.log x) / (x + 1) + b

theorem find_ab 
  (f_tangent : ∀ x y : ℝ, x + y - 3 = 0 → f a b 1 = 2 → deriv (f a b) 1 = -1) :
  a = -1 ∧ b = 2 :=
  sorry

theorem inequality_f 
  (h : ∀ x : ℝ, x > 0 → x ≠ 1 → (let f := f (-1) 2 in f x > (2 * Real.log x) / (x - 1))) :
  ∀ x : ℝ, x > 0 ∧ x ≠ 1 → f (-1) 2 x > (2 * Real.log x) / (x - 1) :=
  sorry

end find_ab_inequality_f_l680_680834


namespace calculation_l680_680019

theorem calculation :
  (-1:ℤ)^(2023) + (1/2:ℚ)^(-1) - (2023 - real.pi :ℝ)^0 - ((real.sqrt 5 + real.sqrt 3) * (real.sqrt 5 - real.sqrt 3)) = -2 :=
  sorry

end calculation_l680_680019


namespace num_black_squares_in_37th_row_l680_680685

-- Define the total number of squares in the n-th row
def total_squares_in_row (n : ℕ) : ℕ := 2 * n - 1

-- Define the number of black squares in the n-th row
def black_squares_in_row (n : ℕ) : ℕ := (total_squares_in_row n - 1) / 2

theorem num_black_squares_in_37th_row : black_squares_in_row 37 = 36 :=
by
  sorry

end num_black_squares_in_37th_row_l680_680685


namespace min_value_l680_680564

noncomputable def min_value_expression (n : ℕ) (x : Fin n → ℝ) : ℝ :=
  ∑ i in Finset.range n, (x i) ^ (i + 1) / (i + 1)

theorem min_value (n : ℕ) (hpos : 0 < n)
  (x : Fin n → ℝ) (hx_pos : ∀ i, 0 < x i)
  (hx_sum : ∑ i in Finset.range n, 1 / (x i) = n) :
  min_value_expression n x = ∑ i in Finset.range n, 1 / (i + 1) :=
begin
  sorry
end

end min_value_l680_680564


namespace kathleen_min_savings_l680_680546

variables (savings_june : ℕ) (savings_july : ℕ) (savings_august : ℕ)
variables (spent_school_supplies : ℕ) (spent_clothes : ℕ)
variables (remaining : ℕ) (aunt_gift_condition : ℕ) (min_saved : ℕ)

def total_savings : ℕ := savings_june + savings_july + savings_august
def total_spent : ℕ := spent_school_supplies + spent_clothes
def calculated_remaining : ℕ := total_savings - total_spent

theorem kathleen_min_savings :
  savings_june = 21 ∧
  savings_july = 46 ∧
  savings_august = 45 ∧
  spent_school_supplies = 12 ∧
  spent_clothes = 54 ∧
  remaining = 46 ∧
  aunt_gift_condition > min_saved →
  min_saved ≤ remaining :=
by
  -- Conditions from the problem
  intros hs,
  cases hs with hs1 hss,
  cases hss with hs2 hs3,
  cases hs3 with hs4 hs5,
  cases hs5 with hs6 hs7,
  cases hs7 with hs8 hra,
  cases hra with hr hr_cond,
  sorry

end kathleen_min_savings_l680_680546


namespace cosine_angle_EF_BC_l680_680645

noncomputable def vector := ℝ × ℝ × ℝ

variables (A B C E F : vector)
variables (BF BE AB AE AC AF EF BC : ℝ)
variables (angle_BF_EF angle_BC_EF : ℝ)

-- Given conditions
def condition1 : BF = 2 * BE := sorry
def condition2 : AB = 2 := sorry
def condition3 : EF = 2 := sorry
def condition4 : BC = 8 := sorry
def condition5 : CA = 4 * Real.sqrt 2 := sorry
def condition6 : (A - B) • (A - E) + (A - C) • (A - F) = 6 := sorry

-- Theorem to prove
theorem cosine_angle_EF_BC : (angle_BC_EF = Real.arccos (9 / 16)) := sorry

end cosine_angle_EF_BC_l680_680645


namespace find_positive_integer_divisible_by_24_between_7_9_and_8_l680_680063

theorem find_positive_integer_divisible_by_24_between_7_9_and_8 :
  ∃ (n : ℕ), n > 0 ∧ (24 ∣ n) ∧ (7.9 < real.cbrt n) ∧ (real.cbrt n < 8) :=
begin
  use 504,
  split,
  { exact nat.zero_lt_succ 503, },
  split,
  { use 21, norm_num, },
  split,
  { norm_num, },
  { norm_num, },
end

end find_positive_integer_divisible_by_24_between_7_9_and_8_l680_680063


namespace power_of_eight_l680_680291

theorem power_of_eight (x : ℝ) (h₁ : x = 8) : (sqrt x / real.root x 4) = x ^ (1 / 4) :=
by
  sorry

end power_of_eight_l680_680291


namespace intersection_eq_l680_680922

def A : Set ℤ := {x | x ∈ Set.Icc (-2 : ℤ) 2}
def B : Set ℝ := {y | y ≤ 1}

theorem intersection_eq : A ∩ {y | y ∈ Set.Icc (-2 : ℤ) 1} = {-2, -1, 0, 1} := by
  sorry

end intersection_eq_l680_680922


namespace checker_cannot_visit_all_cells_exactly_once_l680_680528

-- Define the structure of a chessboard and the conditions
structure Chessboard :=
  (n : ℕ) -- Number of rows
  (m : ℕ) -- Number of columns
  (start : ℕ × ℕ) -- Starting position of the checker
  (cells : List (ℕ × ℕ))

noncomputable def move_up (pos : ℕ × ℕ) : ℕ × ℕ :=
  (pos.1, pos.2 + 1)

noncomputable def move_right (pos : ℕ × ℕ) : ℕ × ℕ :=
  (pos.1 + 1, pos.2)

noncomputable def move_diagonal (pos : ℕ × ℕ) : ℕ × ℕ :=
  (pos.1 - 1, pos.2 - 1)

def valid_moves (pos : ℕ × ℕ) : List (ℕ × ℕ) :=
  [move_up pos, move_right pos, move_diagonal pos]

-- Create an instance of the chessboard 8x8 with the starting position
def chessboard_8x8 : Chessboard :=
{
  n := 8,
  m := 8,
  start := (1, 1),
  cells := List.range (8 * 8) |> List.map (λ i, ((i % 8) + 1, (i / 8) + 1))
}

-- Define the labeling pattern (cell type) as a function
def cell_type (pos : ℕ × ℕ) : ℕ :=
  match pos with
  | (x, y) =>
    match (x + y) % 3 with
    | 0 => 3
    | 1 => 1
    | _ => 2

-- Prove that it is impossible for the checker to visit all cells exactly once
theorem checker_cannot_visit_all_cells_exactly_once :
  ¬(∃ path : List (ℕ × ℕ), ∀ cell ∈ chessboard_8x8.cells, cell ∈ path ∧ 
  ∀ i < (path.length - 1), path.nth i = move_up (path.nth i) ∨ 
      path.nth i = move_right (path.nth i) ∨ 
      path.nth i = move_diagonal (path.nth i)) :=
by
  sorry

end checker_cannot_visit_all_cells_exactly_once_l680_680528


namespace math_proof_problem_l680_680918

variables (a b n p : ℕ) (x : ℝ)
variables (KI SI BC DI : ℝ)
variables (A B C I M D B' C' P Q S K E F : Type)

-- Define conditions
def acute_scalene_triangle (A B C I : Type) := True -- Placeholder for actual conditions on acute scalene triangle with incenter
def circumcenter_BIC (M : Type) := True             -- Placeholder for actual condition on being circumcenter
def perpendiculars (B' C' D I : Type) := True       -- Placeholder for perpendicular angle conditions
def intersections (P Q S K : Type) := True          -- Placeholder conditions for intersections P, Q, S, K
def diameter_incircle (E F : Type) := True

-- Given lengths
def KI_value (x : ℝ) : Prop := KI = 15 * x
def SI_value (x : ℝ) : Prop := SI = 20 * x + 15
def BC_value (x : ℝ) : Prop := BC = 20 * x ^ (5 / 2)
def DI_value (x : ℝ) : Prop := DI = 20 * x ^ (3 / 2)

-- Define x
def x_value (a b n p : ℕ) (x : ℝ) : Prop := 
  x = (a : ℝ) / (b : ℝ) * (n + real.sqrt p)

-- Main Theorem
theorem math_proof_problem :
  acute_scalene_triangle A B C I →
  circumcenter_BIC M →
  perpendiculars B' C' D I →
  intersections P Q S K →
  diameter_incircle E F →
  KI_value x →
  SI_value x →
  BC_value x →
  DI_value x →
  x_value a b n p x →
  a + b + n + p = 99 :=
begin
  sorry
end

end math_proof_problem_l680_680918


namespace max_rho_value_ratio_value_l680_680177

noncomputable def max_rho_on_curve_C : ℝ :=
  let C := λ θ : ℝ, 2 * cos θ + 2 * sin θ in
  let max_rho := 2 * Real.sqrt 2 in 
  let P := (2 * Real.sqrt 2, π/4) in
  if 0 ≤ θ ∧ θ < 2 * π then max_rho else sorry

theorem max_rho_value : ∃ P : ℝ × ℝ, P = (2 * Real.sqrt 2, π / 4) ∧ 
                       (∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * π → (2 * cos θ + 2 * sin θ) ≤ 2 * Real.sqrt 2) :=
  sorry

def ratio_MA_MB : ℝ := 
  let x := λ t : ℝ, (√2 / 2) * t in
  let y := λ t : ℝ, 1 + (√2 / 2) * t in
  let MA := λ t1 : ℝ, Real.sqrt ((x t1 - 1)^2 + (y t1 - 0)^2) in
  let MB := λ t2 : ℝ, Real.sqrt ((x t2 - 1)^2 + (y t2 - 0)^2) in
  (MA ((√6 + √2) / 2) / MB ((√6 - √2) / 2))

theorem ratio_value : ratio_MA_MB = 2 + Real.sqrt 3 := 
  sorry

end max_rho_value_ratio_value_l680_680177


namespace sum_equiv_l680_680734

theorem sum_equiv {f : ℕ → ℕ → ℝ} (h : ∀ (n k : ℕ), n ≥ 3 ∧ 1 ≤ k ∧ k ≤ n - 2 → f n k = (k^2) / (3^(n+k))) :
  (∑' n=3, ∑' k=1, if h : k ≤ n - 2 then f n k else 0) = 135 / 512 :=
by sorry

end sum_equiv_l680_680734


namespace inequality_1_inequality_2_inequality_4_l680_680436

variable {a b : ℝ}

def condition (a b : ℝ) : Prop := (1/a < 1/b) ∧ (1/b < 0)

theorem inequality_1 (ha : a < 0) (hb : b < 0) (hc : condition a b) : a + b < a * b :=
sorry

theorem inequality_2 (hc : condition a b) : |a| < |b| :=
sorry

theorem inequality_4 (hc : condition a b) : (b / a) + (a / b) > 2 :=
sorry

end inequality_1_inequality_2_inequality_4_l680_680436


namespace largest_root_vieta_l680_680421

theorem largest_root_vieta 
  (a b c : ℝ)
  (h1 : a + b + c = 6)
  (h2 : a * b + a * c + b * c = 11)
  (h3 : a * b * c = -6) : 
  max a (max b c) = 3 :=
sorry

end largest_root_vieta_l680_680421


namespace count_special_multiples_l680_680141

theorem count_special_multiples : 
  let multiples_of_35 := {n | ∃ k : ℕ, 1 ≤ n ∧ n ≤ 300 ∧ n = 35 * k},
      not_multiples_of_6_or_8 := {n | ¬ (∃ m : ℕ, n = 6 * m) ∧ ¬ (∃ p : ℕ, n = 8 * p)},
      valid_numbers := multiples_of_35 ∩ not_multiples_of_6_or_8
  in set.finite (valid_numbers) ∧ set.card valid_numbers = 6 
 := sorry

end count_special_multiples_l680_680141


namespace part1_part2_l680_680813

-- Definitions of propositions P and q
def P (t : ℝ) : Prop := (4 - t > t - 1 ∧ t - 1 > 0)
def q (a t : ℝ) : Prop := t^2 - (a+3)*t + (a+2) < 0

-- Part 1: If P is true, find the range of t.
theorem part1 (t : ℝ) (hP : P t) : 1 < t ∧ t < 5/2 :=
by sorry

-- Part 2: If P is a sufficient but not necessary condition for q, find the range of a.
theorem part2 (a : ℝ) 
  (hP_q : ∀ t, P t → q a t) 
  (hsubset : ∀ t, 1 < t ∧ t < 5/2 → q a t) 
  : a > 1/2 :=
by sorry

end part1_part2_l680_680813


namespace max_abs_z_l680_680928

noncomputable def largest_possible_value_of_abs_z : ℝ :=
  (1 + Real.sqrt 5) / 2

theorem max_abs_z (a b c z : ℂ) 
  (h1 : abs a = 1) 
  (h2 : abs b = 1) 
  (h3 : abs c = 1) 
  (h4 : Complex.arg c = Complex.arg a + Complex.arg b) 
  (h5 : a * z^2 + b * z + c = 0)
  : abs z ≤ largest_possible_value_of_abs_z := 
sorry

end max_abs_z_l680_680928


namespace probability_of_rolling_2_4_6_l680_680656

def fair_eight_sided_die : ℕ := 8
def successful_outcomes : set ℕ := {2, 4, 6}
def num_successful_outcomes : ℕ := successful_outcomes.to_finset.card

theorem probability_of_rolling_2_4_6 :
  (num_successful_outcomes : ℚ) / fair_eight_sided_die = 3 / 8 :=
by
  -- Note: The proof is omitted by using 'sorry'
  sorry

end probability_of_rolling_2_4_6_l680_680656


namespace sum_of_a_for_unique_solution_l680_680430

theorem sum_of_a_for_unique_solution (a : ℝ) (x : ℝ) :
  (∃ (a : ℝ), 3 * x ^ 2 + a * x + 6 * x + 7 = 0 ∧ (a + 6) ^ 2 - 4 * 3 * 7 = 0) →
  (-6 + 2 * Real.sqrt 21 + -6 - 2 * Real.sqrt 21 = -12) :=
by
  sorry

end sum_of_a_for_unique_solution_l680_680430


namespace num_quadratics_eq_9900_l680_680919

def X : Set ℕ := {n | ∃ (m n : ℕ), 0 ≤ m ∧ m ≤ 9 ∧ 0 ≤ n ∧ n ≤ 9 ∧ n = 2^m * 3^n}

theorem num_quadratics_eq_9900 :
  let quadratics := { (a, b, c) | a ∈ X ∧ b ∈ X ∧ c ∈ X ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (2 * b)^2 = 4 * a * c }
  in quadratics.card = 9900 :=
by
  sorry

end num_quadratics_eq_9900_l680_680919


namespace Eric_points_l680_680886

def points (E M S : ℕ) : Prop :=
  S = M + 8 ∧ M = 3 * E / 2 ∧ (E + M + S = 32)

theorem Eric_points (E M S : ℕ) (h : points E M S) : E = 6 :=
by
  cases h with
  | intro h1 (intro h2 h3) => 
    sorry

end Eric_points_l680_680886


namespace exists_disjoint_subsets_with_close_reciprocal_sums_l680_680108

theorem exists_disjoint_subsets_with_close_reciprocal_sums
  (c : Fin 14 -> ℕ) 
  (h_distinct : ∀ i j : Fin 14, i ≠ j -> c i ≠ c j):
  ∃ k (a b : Fin k → Fin 14), 
    (∀ i j : Fin k, a i ≠ a j ∧ b i ≠ b j ∧ a i ≠ b j) ∧ 
    (1 ≤ k ∧ k ≤ 7) ∧ 
    |(∑ i : Fin k, (1 : ℚ) / c (a i)) - (∑ i : Fin k, (1 : ℚ) / c (b i))| < 0.001 := 
sorry

end exists_disjoint_subsets_with_close_reciprocal_sums_l680_680108


namespace sequence_a_n_general_formula_and_value_sequence_b_n_general_formula_l680_680102

theorem sequence_a_n_general_formula_and_value (a : ℕ → ℕ) 
  (h1 : a 1 = 3) 
  (h10 : a 10 = 21) 
  (h_linear : ∃ (k b : ℕ), ∀ n, a n = k * n + b) :
  (∀ n, a n = 2 * n + 1) ∧ a 2005 = 4011 :=
by 
  sorry

theorem sequence_b_n_general_formula (a b : ℕ → ℕ)
  (h_seq_a : ∀ n, a n = 2 * n + 1) 
  (h_b_formed : ∀ n, b n = a (2 * n)) : 
  ∀ n, b n = 4 * n + 1 :=
by 
  sorry

end sequence_a_n_general_formula_and_value_sequence_b_n_general_formula_l680_680102


namespace complement_I_nat_is_empty_l680_680492

def set_I : Set ℤ := {x | x ≥ -1}

theorem complement_I_nat_is_empty :
  (∀ x : ℕ, x ∈ set_I) → ∀ x : ℕ, x ∉ (set.univ \ set_I : Set ℕ) :=
by
  intro h x hx
  exact hx (h x)

end complement_I_nat_is_empty_l680_680492


namespace remainder_when_divided_by_2000_l680_680190

def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

noncomputable def count_disjoint_subsets (S : Set ℕ) : ℕ :=
  let totalWays := 3^12
  let emptyACases := 2*2^12
  let bothEmptyCase := 1
  (totalWays - emptyACases + bothEmptyCase) / 2

theorem remainder_when_divided_by_2000 : count_disjoint_subsets S % 2000 = 1625 := by
  sorry

end remainder_when_divided_by_2000_l680_680190


namespace locus_equation_of_points_at_distance_2_from_line_l680_680073

theorem locus_equation_of_points_at_distance_2_from_line :
  {P : ℝ × ℝ | abs ((3 / 5) * P.1 - (4 / 5) * P.2 - (1 / 5)) = 2} =
    {P : ℝ × ℝ | 3 * P.1 - 4 * P.2 - 11 = 0} ∪ {P : ℝ × ℝ | 3 * P.1 - 4 * P.2 + 9 = 0} :=
by
  -- Proof goes here
  sorry

end locus_equation_of_points_at_distance_2_from_line_l680_680073


namespace number_of_solutions_l680_680130

noncomputable def f (x : ℝ) : ℝ :=
  abs (4 - 4 * abs x) - 2

theorem number_of_solutions : 
  (∀ x : ℝ, f(f(x)) = x) → (∃ n : ℕ, n = 16) :=
by
  sorry

end number_of_solutions_l680_680130


namespace total_balloons_l680_680003

theorem total_balloons (A_initial : Nat) (A_additional : Nat) (J_initial : Nat) 
  (h1 : A_initial = 3) (h2 : J_initial = 5) (h3 : A_additional = 2) : 
  A_initial + A_additional + J_initial = 10 := by
  sorry

end total_balloons_l680_680003


namespace jenny_distance_l680_680913

-- Definitions from conditions
def jenny_initial_distance : ℝ := x
def jenny_total_distance (x : ℝ) : ℝ := x + (1 / 3) * x
def mark_initial_distance : ℝ := 15
def mark_total_distance : ℝ := mark_initial_distance + 2 * mark_initial_distance
def difference : ℝ := 21

-- Theorem stating the solution
theorem jenny_distance (x : ℝ) : 
  jenny_total_distance x + difference = mark_total_distance → x = 18 :=
by
  sorry

end jenny_distance_l680_680913


namespace midpoints_perpendiculars_intersect_at_one_point_l680_680556

theorem midpoints_perpendiculars_intersect_at_one_point
  (A B C M A1 B1 C1 A2 B2 C2 : Type)
  [Midpoint A B C A1] [Midpoint B C A C1] [Midpoint C A B B1]
  [Perpendicular M B C A2] [Perpendicular M C A B2] [Perpendicular M A B C2]
  [LineSegment B2 C2 A1] [LineSegment C2 A2 B1] [LineSegment A2 B2 C1] : 
  ∃ P : Type, 
    Perpendicular A1 B2 C2 P ∧ 
    Perpendicular B1 C2 A2 P ∧ 
    Perpendicular C1 A2 B2 P := 
sorry

end midpoints_perpendiculars_intersect_at_one_point_l680_680556


namespace find_area_of_triangle_l680_680208

-- Defining the geometric setup
variables {A B C M O D : Type*}
variables (α β : ℝ)

-- Defining the conditions given in the problem
def right_triangle (A B C : Type*) : Prop :=
  ∠B = 90° ∧ ∠BAC + ∠ACB = 90°

def medial_point (B C M : Type*) : Prop :=
  M = midpoint B C

def intersect_median_angle_bisector (A M C D O : Type*) : Prop :=
  AM is the median ∧ CD is the angle bisector ∧ AM ∩ CD = O

def lengths (CO OD : ℝ) : Prop :=
  CO = 9 ∧ OD = 5

-- Theorem stating the area of triangle ABC
theorem find_area_of_triangle
  (A B C M O D : Type*)
  (h1 : right_triangle A B C)
  (h2 : medial_point B C M)
  (h3 : intersect_median_angle_bisector A M C D O)
  (h4 : lengths 9 5)
  : area_of_triangle A B C = 1323 / 20 :=
sorry

end find_area_of_triangle_l680_680208


namespace tangent_line_min_a_condition_l680_680465

noncomputable def f (x : ℝ) := 2 * Real.log x - 3 * x^2 - 11 * x

theorem tangent_line (x0 : ℝ) (hx0 : x0 = 1) :
  ∃ (m c : ℝ), (∀ x, f x = m * (x - x0) + c)
  ∧ m = -15 ∧ c = f 1 := 
begin
  sorry,
end

theorem min_a_condition (a : ℝ) (h : ∀ x : ℝ, f x ≤ (a-3) * x^2 + (2*a-13) * x + 1) :
  a ≥ 1 :=
begin
  sorry,
end

end tangent_line_min_a_condition_l680_680465


namespace curve_equation_exists_iff_a_curve_y_eq_zero_when_a_sqrt_3_curve_ellipse_when_a_gt_sqrt_3_range_of_a_when_AC_perp_AD_max_min_area_AOB_l680_680135

noncomputable def curve_equation (a : ℝ) (x y : ℝ) : Prop :=
if a < sqrt 3 then false
else if a = sqrt 3 then y = 0 ∧ -sqrt 3 ≤ x ∧ x ≤ sqrt 3
else x^2 / a^2 + y^2 / (a^2 - 3) = 1

theorem curve_equation_exists_iff_a (a : ℝ) :
  (∀ x y, curve_equation a x y → (a ≥ sqrt 3)) := sorry

theorem curve_y_eq_zero_when_a_sqrt_3 :
  (∀ x y, a = sqrt 3 → curve_equation a x y → y = 0) := sorry

theorem curve_ellipse_when_a_gt_sqrt_3 :
  (∀ x y, a > sqrt 3 → curve_equation a x y → x^2 / a^2 + y^2 / (a^2 - 3) = 1) := sorry

theorem range_of_a_when_AC_perp_AD (a : ℝ) :
  (AC_perp_AD_condition a → sqrt 3 < a ∧ a ≤ sqrt 6) := sorry

theorem max_min_area_AOB (a : ℝ) (S : ℝ) :
  a = 2 → (AO_perp_OB_condition a S → S ∈ (set.Icc (4/5) 1)) := sorry

end curve_equation_exists_iff_a_curve_y_eq_zero_when_a_sqrt_3_curve_ellipse_when_a_gt_sqrt_3_range_of_a_when_AC_perp_AD_max_min_area_AOB_l680_680135


namespace perpendicular_lines_a_value_l680_680138

theorem perpendicular_lines_a_value (a : ℝ) :
  (a * (a + 2) = -1) → a = -1 :=
by
  intro h
  sorry

end perpendicular_lines_a_value_l680_680138


namespace v_17_eq_b_l680_680384

noncomputable def v : ℕ → ℝ
| 0     := 0  -- dummy value to handle lean's zero-based indexing 
| 1     := b
| (n+2) := -2 / (v (n + 1) + 2)

theorem v_17_eq_b (b : ℝ) (hb : b > 0) : v 17 = b :=
sorry

end v_17_eq_b_l680_680384


namespace equal_areas_black_white_within_bound_l680_680695

theorem equal_areas_black_white_within_bound {board : array (array bool)} (h : board.size = 8 ∧ ∀ r, board[r].size = 8): 
  ∀ (line : list (fin 8 × fin 8)), 
  (∀ (i : ℕ), i < line.length - 1 → 
    (line.nth i).fst - (line.nth (i + 1)).fst <= 1 ∧ 
    (line.nth i).fst - (line.nth (i + 1)).fst >= -1 ∧ 
    (line.nth i).snd - (line.nth (i + 1)).snd <= 1 ∧ 
    (line.nth i).snd - (line.nth (i + 1)).snd >= -1) → 
  (line.head! = line.last!) → 
  board_bound_contains_line board line → 
  sum_of_black_areas board line = sum_of_white_areas board line :=
sorry

end equal_areas_black_white_within_bound_l680_680695


namespace hyperbola_eccentricity_l680_680251

theorem hyperbola_eccentricity 
  (a b c : ℝ) 
  (h_pos_a : a > 0) 
  (h_pos_b : b > 0)
  (h_hyperbola : y^2 / a^2 - x^2 / b^2 = 1)
  (h_distance : ∃ d, d = (2 * real.sqrt 5 / 5) * a 
                      ∧ d = b * c / real.sqrt (a^2 + b^2)) :
  c / a = 3 * (real.sqrt 5) / 5 :=
sorry

end hyperbola_eccentricity_l680_680251


namespace radius_of_isosceles_trapezoid_circle_l680_680982

noncomputable def radius_of_inscribed_circle (AB CD : ℝ) (hAB : AB = 1) (hCD : CD = 6) : ℝ :=
  let r := 3 / 7 in
  r

-- Theorem that expresses the problem
theorem radius_of_isosceles_trapezoid_circle (AB CD : ℝ) (hAB : AB = 1) (hCD : CD = 6) :
  radius_of_inscribed_circle AB CD hAB hCD = 3 / 7 :=
sorry

end radius_of_isosceles_trapezoid_circle_l680_680982


namespace range_of_a_l680_680829

noncomputable def f (a : ℝ) : ℝ → ℝ :=
λ x, if x > 1 then (1 / x - 1) else (-2 * x + a)

theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f a x₁ ≠ f a x₂) ↔ a ∈ Set.Ici 2 := by
  sorry

end range_of_a_l680_680829


namespace derivative_at_x_equals_1_l680_680618

variable (x : ℝ)
def y : ℝ := (x + 1) * (x - 1)

theorem derivative_at_x_equals_1 : deriv y 1 = 2 :=
by
  sorry

end derivative_at_x_equals_1_l680_680618


namespace more_cats_than_dogs_l680_680376

theorem more_cats_than_dogs:
  ∃ (cats_before cats_after dogs: ℕ),
    cats_before = 28 ∧
    dogs = 18 ∧
    cats_after = cats_before - 3 ∧
    cats_after - dogs = 7 :=
by
  use 28, 25, 18
  split
  case left =>
    exact rfl
  case right =>
    split
    case left =>
      exact rfl
    case right =>
      split
      case left =>
        exact rfl
      case right =>
        exact rfl

end more_cats_than_dogs_l680_680376


namespace max_three_digit_divisible_by_25_count_l680_680273

open Nat

-- Define the sequence with the given properties
def seq (n : ℕ) : ℕ := sorry -- Placeholder for the sequence definition

-- State the conditions
axiom seq_conditions : ∀ (k : ℕ), k ≤ n - 2 → seq (k + 2) = 3 * seq (k + 1) - 2 * seq k - 1

axiom seq_includes_2021 : ∃ (k : ℕ), seq k = 2021

-- Define the condition for the number being three-digit and divisible by 25
def is_three_digit_divisible_by_25 (m : ℕ) : Prop :=
  100 ≤ m ∧ m ≤ 999 ∧ m % 25 = 0

-- State the main theorem to be proved
theorem max_three_digit_divisible_by_25_count (n : ℕ) (hn : 3 ≤ n) :
  ∃! (count : ℕ), count = 36 ∧
  ∀ m, is_three_digit_divisible_by_25 m → ∃ k, seq k = m :=
sorry

end max_three_digit_divisible_by_25_count_l680_680273


namespace whole_numbers_in_interval_l680_680144

theorem whole_numbers_in_interval : 
  let a := (7 : ℝ) / 2
  let b := 3 * Real.pi
  ∃ n, n = 6 ∧ ∀ k, (⌊a⌋ + 1 <= k ∧ k < ⌊b⌋ + 1) → (k : ℤ) ∈ set.Ico (⌊a⌋ + 1) (⌊b⌋ + 1) :=
by {
  let a := (7 : ℝ) / 2,
  let b := 3 * Real.pi,
  use 6,
  split,
  -- Proof that there are 6 whole numbers
  -- Placeholder
  sorry,
  refine λ k hk, set.mem_Ico.mpr ⟨_,_⟩,
  -- Placeholder
  sorry,
  -- Placeholder
  sorry,
}

end whole_numbers_in_interval_l680_680144


namespace quadratic_solution_l680_680240

theorem quadratic_solution (x : ℝ) : x^2 - 2 * x - 3 = 0 → (x = 3 ∨ x = -1) :=
by
  sorry

end quadratic_solution_l680_680240


namespace spencer_walked_distance_l680_680184

/-- Define the distances involved -/
def total_distance := 0.8
def library_to_post_office := 0.1
def post_office_to_home := 0.4

/-- Define the distance from house to library as a variable to calculate -/
def house_to_library := total_distance - library_to_post_office - post_office_to_home

/-- The theorem states that Spencer walked 0.3 miles from his house to the library -/
theorem spencer_walked_distance : 
  house_to_library = 0.3 :=
by
  -- Proof omitted
  sorry

end spencer_walked_distance_l680_680184


namespace ball_center_distance_l680_680688

noncomputable def travel_distance (d R1 R3 L : ℝ) : ℝ :=
  let radius_ball := d / 2
  let adjusted_R1 := R1 - radius_ball
  let adjusted_R3 := R3 - radius_ball
  let distance_arc1 := π * adjusted_R1
  let distance_arc3 := π * adjusted_R3
  distance_arc1 + L + distance_arc3

theorem ball_center_distance (d R1 R3 L : ℝ) (hd : d = 6) (hR1 : R1 = 120) (hR3 : R3 = 90) (hL : L = 50) :
  travel_distance d R1 R3 L = 204 * π + 50 :=
by
  rw [hd, hR1, hR3, hL]
  simp [travel_distance]
  sorry

end ball_center_distance_l680_680688


namespace interest_percentage_is_correct_l680_680303

-- Define the conditions and the entities involved in the problem.
def purchasePrice : ℝ := 118
def downPayment : ℝ := 18
def monthlyPayment : ℝ := 10
def numberOfPayments : ℝ := 12

-- Define the relevant computations and the target statement.
theorem interest_percentage_is_correct :
  let totalPaid := downPayment + (monthlyPayment * numberOfPayments)
  let interestPaid := totalPaid - purchasePrice
  let interestPercent := (interestPaid / purchasePrice) * 100
  Real.ceil (interestPercent * 10) / 10 = 16.9 :=
by
  sorry

end interest_percentage_is_correct_l680_680303


namespace construct_tangent_circles_l680_680744

-- Definitions for the given problem conditions
variable {l : Line} -- Given line l
variable {P : Point} -- Given point P on the line l
variable {r : ℝ} -- Given radius r

-- Definition of a circle (assuming certain definitions for Line and Point)
structure Circle where
  center : Point
  radius : ℝ

-- Tangent relationship between a line and a circle at a point
def is_tangent_at (c : Circle) (l : Line) (P : Point) : Prop := 
  sorry  -- Implementation would describe tangent property

-- Positional property defining O1 and O2
def points_distance (O : Point) (P : Point) (r : ℝ) : Prop :=
  sorry  -- Implementation would describe the distance criteria

-- The main problem statement as a Lean theorem
theorem construct_tangent_circles :
  ∃ O1 O2 : Point, 
    points_distance O1 P r ∧
    points_distance O2 P r ∧
    is_tangent_at (Circle.mk O1 r) l P ∧
    is_tangent_at (Circle.mk O2 r) l P :=
  sorry

end construct_tangent_circles_l680_680744


namespace complement_U_A_l680_680477

-- Definitions of U and A based on problem conditions
def U : Set ℤ := {-1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0, 2}

-- Definition of the complement in Lean
def complement (A B : Set ℤ) : Set ℤ := {x | x ∈ A ∧ x ∉ B}

-- The main statement to be proved
theorem complement_U_A :
  complement U A = {1, 3} :=
sorry

end complement_U_A_l680_680477


namespace floor_sqrt_120_l680_680401

theorem floor_sqrt_120 :
  (∀ x : ℝ, 10^2 = 100 ∧ 11^2 = 121 ∧ 100 < 120 ∧ 120 < 121 → 
  (∃ y : ℕ, y = 10 ∧ floor (real.sqrt 120) = y)) :=
by
  assume h,
  sorry

end floor_sqrt_120_l680_680401


namespace example_one_example_two_l680_680230

/- Part 1: Perform the defined operation on the pair (45, 80) -/
theorem example_one (a b : ℕ) (h₀ : a = 45) (h₁ : b = 80) :
  let f := (λ (x y : ℕ), if x > y then (x - y, y) else (x, y - x)) in
  ((f (f (f (f (f (f (a, b)))))))) = (5, 5) :=
sorry

/- Part 2: Find the maximum possible sum of two four-digit numbers that result in 17 -/
theorem example_two (a b : ℕ) 
  (h₀ : a > 999)
  (h₁ : b > 999)
  (h₂ : a < 10000)
  (h₃ : b < 10000)
  (h₄ : (∀ (x y : ℕ), (x ≠ y → x > 0 → y > 0 → if x > y then x - y else y - x = 17) → a = 17 ∧ b = 17) :
  a + b = 19975 :=
sorry

end example_one_example_two_l680_680230


namespace birds_joined_l680_680282

def numBirdsInitially : Nat := 1
def numBirdsNow : Nat := 5

theorem birds_joined : numBirdsNow - numBirdsInitially = 4 := by
  -- proof goes here
  sorry

end birds_joined_l680_680282


namespace determine_k_l680_680743

theorem determine_k (a b c k : ℝ) (h : a + b + c = 1) (h_eq : k * (a + bc) = (a + b) * (a + c)) : k = 1 :=
sorry

end determine_k_l680_680743


namespace binary_representation_of_14_l680_680408

theorem binary_representation_of_14 : nat.binary_repr 14 = "1110" :=
sorry

end binary_representation_of_14_l680_680408


namespace number_of_students_l680_680611

theorem number_of_students (n : ℕ) (A : ℕ) 
  (h1 : A = 10 * n)
  (h2 : (A - 11 + 41) / n = 11) :
  n = 30 := 
sorry

end number_of_students_l680_680611


namespace inv_3i_minus_2inv_i_eq_neg_inv_5i_l680_680781

-- Define the imaginary unit i such that i^2 = -1
def i : ℂ := Complex.I
axiom i_square : i^2 = -1

-- Proof statement
theorem inv_3i_minus_2inv_i_eq_neg_inv_5i : (3 * i - 2 * (1 / i))⁻¹ = -i / 5 :=
by
  -- Replace these steps with the corresponding actual proofs
  sorry

end inv_3i_minus_2inv_i_eq_neg_inv_5i_l680_680781


namespace total_ladybugs_eq_11676_l680_680640

def Number_of_leaves : ℕ := 84
def Ladybugs_per_leaf : ℕ := 139

theorem total_ladybugs_eq_11676 : Number_of_leaves * Ladybugs_per_leaf = 11676 := by
  sorry

end total_ladybugs_eq_11676_l680_680640


namespace solve_for_F_l680_680209

variable (S W F : ℝ)

def condition1 (S W : ℝ) : Prop := S = W / 3
def condition2 (W F : ℝ) : Prop := W = F + 60
def condition3 (S W F : ℝ) : Prop := S + W + F = 150

theorem solve_for_F (S W F : ℝ) (h1 : condition1 S W) (h2 : condition2 W F) (h3 : condition3 S W F) : F = 52.5 :=
sorry

end solve_for_F_l680_680209


namespace smallest_angle_l680_680558

-- Conditions
variables {a b c : Vec2}  -- unit vectors in ℝ^2
variable (β : ℝ)  -- angle in degrees

-- Additional definitions and assumptions
def unit_vector (v : Vec2) : Prop := ∥v∥ = 1

def angle_between (v w : Vec2) (θ : ℝ) : Prop :=
  θ = acos ((v ⬝ w) / (∥v∥ * ∥w∥))

def scalar_triple_product (u v w : Vec2) : ℝ :=
  (u ⬝ (v × w))

-- Statement of the problem
theorem smallest_angle (h_a_unit : unit_vector a)
                      (h_b_unit : unit_vector b)
                      (h_c_unit : unit_vector c)
                      (h_ab_angle : angle_between a b β)
                      (h_ac_cross_angle : angle_between c (a × b) β)
                      (h_scalar_triple : scalar_triple_product b c a = (sqrt 3) / 4) :
  β = 30 := sorry

end smallest_angle_l680_680558


namespace more_cats_than_dogs_l680_680375

-- Define the initial conditions
def initial_cats : ℕ := 28
def initial_dogs : ℕ := 18
def cats_adopted : ℕ := 3

-- Compute the number of cats after adoption
def cats_now : ℕ := initial_cats - cats_adopted

-- Define the target statement
theorem more_cats_than_dogs : cats_now - initial_dogs = 7 := by
  unfold cats_now
  unfold initial_cats
  unfold cats_adopted
  unfold initial_dogs
  sorry

end more_cats_than_dogs_l680_680375


namespace find_f_l680_680677

theorem find_f (f : ℤ → ℤ) (h : ∀ x : ℤ, f(x - 1) = x^2) : ∀ x : ℤ, f x = (x + 1)^2 :=
by
  sorry

end find_f_l680_680677


namespace count_ordered_pairs_l680_680025

theorem count_ordered_pairs :
  (∃ (n : ℕ), n = 3515 ∧
    ∀ (x y : ℕ), 1 ≤ x ∧ x < y ∧ y ≤ 150 →
    (∀ (k m : ℕ), k % 4 = x % 4 ∧ m % 4 = y % 4 →
    ((x ≡ y [MOD 4] ∨ x ≡ y + 2 [MOD 4] ∨ x + 2 ≡ y [MOD 4]) →
    is_real (complex.I ^ x + complex.I ^ y)) →
    true)) := sorry

end count_ordered_pairs_l680_680025


namespace pentagon_position_3010_l680_680621

def rotate_72 (s : String) : String :=
match s with
| "ABCDE" => "EABCD"
| "EABCD" => "DCBAE"
| "DCBAE" => "EDABC"
| "EDABC" => "ABCDE"
| _ => s

def reflect_vertical (s : String) : String :=
match s with
| "EABCD" => "DCBAE"
| "DCBAE" => "EABCD"
| _ => s

def transform (s : String) (n : Nat) : String :=
match n % 5 with
| 0 => s
| 1 => reflect_vertical (rotate_72 s)
| 2 => rotate_72 (reflect_vertical (rotate_72 s))
| 3 => reflect_vertical (rotate_72 (reflect_vertical (rotate_72 s)))
| 4 => rotate_72 (reflect_vertical (rotate_72 (reflect_vertical (rotate_72 s))))
| _ => s

theorem pentagon_position_3010 :
  transform "ABCDE" 3010 = "ABCDE" :=
by 
  sorry

end pentagon_position_3010_l680_680621


namespace complement_U_P_l680_680793

def U := { y : ℝ | ∃ x > 0, y = log x / log 2 }
def P := { y : ℝ | ∃ x > 2, y = 1 / x }

theorem complement_U_P :
  (U \ P) = (Iic 0 ∪ Ici (1/2:ℝ)) :=
sorry

end complement_U_P_l680_680793


namespace intersection_points_l680_680424

theorem intersection_points :
  {p : ℝ × ℝ |
    (∃ x : ℝ, p = (x, 3*x^2 - 4*x + 2) ∧ p = (x, x^3 - 2*x^2 + 5*x - 1))} =
  {(1, 1), (3, 17)} :=
  sorry

end intersection_points_l680_680424


namespace city_numbering_for_optimal_road_systems_l680_680307

theorem city_numbering_for_optimal_road_systems
    (N : ℕ)
    (roads : fin N → fin N → ℕ)
    (maintenance_cost : fin N → fin N → ℕ)
    (k_optimal : Π k : fin N.succ, Prop) :
    (∃ (numbering : fin N.succ → fin N),
        ∀ k : fin N.succ, k_optimal k) :=
begin
  -- Proof omitted
  sorry
end

end city_numbering_for_optimal_road_systems_l680_680307


namespace simplify_expression_l680_680596

-- Defining each term in the sum
def term1  := 1 / ((1 / 3) ^ 1)
def term2  := 1 / ((1 / 3) ^ 2)
def term3  := 1 / ((1 / 3) ^ 3)

-- Sum of the terms
def terms_sum := term1 + term2 + term3

-- The simplification proof statement
theorem simplify_expression : 1 / terms_sum = 1 / 39 :=
by sorry

end simplify_expression_l680_680596


namespace find_b_l680_680070

theorem find_b (a b c : ℚ) :
  -- Condition from the problem, equivalence of polynomials for all x
  ((4 : ℚ) * x^2 - 2 * x + 5 / 2) * (a * x^2 + b * x + c) =
    12 * x^4 - 8 * x^3 + 15 * x^2 - 5 * x + 5 / 2 →
  -- Given we found that a = 3 from the solution
  a = 3 →
  -- We need to prove that b = -1/2
  b = -1 / 2 :=
sorry

end find_b_l680_680070


namespace smallest_prime_perimeter_l680_680337

theorem smallest_prime_perimeter :
  ∃ (p q r : ℕ), nat.prime p ∧ nat.prime q ∧ nat.prime r ∧ (p < q ∧ q < r) ∧
  (q = p + 2) ∧ (r = p + 4) ∧ nat.prime (p + q + r) ∧ (p + q + r = 23) :=
begin
  sorry
end

end smallest_prime_perimeter_l680_680337


namespace sum_of_squares_of_roots_l680_680778

theorem sum_of_squares_of_roots :
  ∀ (r₁ r₂ : ℝ), (r₁ + r₂ = 15) → (r₁ * r₂ = 6) → (r₁^2 + r₂^2 = 213) :=
by
  intros r₁ r₂ h_sum h_prod
  -- Proof goes here, but skipping it for now
  sorry

end sum_of_squares_of_roots_l680_680778


namespace smallest_n_for_f_eq_4_l680_680980

-- Define a function f that returns the number of distinct ordered pairs (a, b) such that a^2 + ab + b^2 = n
def f (n : ℕ) : ℕ := 
  fintype.card {p : ℕ × ℕ // p.fst^2 + p.fst * p.snd + p.snd^2 = n}

-- State the theorem: Find the smallest n such that f(n) = 4
theorem smallest_n_for_f_eq_4 : ∃ n : ℕ, f(n) = 4 ∧ ∀ m : ℕ, m < n → f(m) ≠ 4 :=
by
  use 19
  split
  repeat sorry

end smallest_n_for_f_eq_4_l680_680980


namespace revenue_from_full_price_tickets_l680_680338

theorem revenue_from_full_price_tickets (f h p : ℕ) 
    (h1 : f + h = 160) 
    (h2 : f * p + h * (p / 2) = 2400) 
    (h3 : h = 160 - f)
    (h4 : 2 * 2400 = 4800) :
  f * p = 800 := 
sorry

end revenue_from_full_price_tickets_l680_680338


namespace projection_matrix_l680_680046

-- Define matrix Q
def Q : Matrix (Fin 2) (Fin 2) ℚ := 
  ![![x, 1/5], ![y, 4/5]]

-- Theorem statement according to the problem description
theorem projection_matrix (x y : ℚ) : (Q * Q = Q) → (x = 1 ∧ y = 0) :=
begin
  sorry
end

end projection_matrix_l680_680046


namespace summation_proof_l680_680737

open BigOperators

theorem summation_proof :
  ∑ n in finset.range (∞).filter (λ n, n ≥ 3), ∑ k in finset.range (n - 2).filter (λ k, k ≥ 1), k^2 * (3:ℝ) ^ (- (n + k)) = 5 / 72 := 
by 
  sorry

end summation_proof_l680_680737


namespace find_integer_x_l680_680581

theorem find_integer_x (x : ℤ) :
  (2 * x > 70 → x > 35 ∧ 4 * x > 25 → x > 6) ∧
  (x < 100 → x = 6) ∧ -- Given valid inequality relations and restrictions
  ((2 * x <= 70 ∨ x >= 100) ∨ (4 * x <= 25 ∨ x <= 5)) := -- Certain contradictions inherent

  by sorry

end find_integer_x_l680_680581


namespace distinct_placements_count_l680_680151

theorem distinct_placements_count : 
  ∃ (n : ℕ), 
    (∃ (k : ℕ),
      (∀ (total_boxes : ℕ) (empty_boxes : ℕ), 
        total_boxes = 4 ∧ empty_boxes = 2 →
        ∃ (ways_to_choose_empty : ℕ),
        ways_to_choose_empty = Nat.choose total_boxes empty_boxes ∧
        (∃ (ways_to_arrange_digits : ℕ),
          ways_to_arrange_digits = (Nat.factorial 4) / (Nat.factorial (4 - 2)) ∧
          n = ways_to_choose_empty * ways_to_arrange_digits))) 
  ∧ n = 72 := 
by {
  sorry
}

end distinct_placements_count_l680_680151


namespace simplify_complex_number_l680_680454

def imaginary_unit : ℂ := complex.I

theorem simplify_complex_number (h : imaginary_unit^2 = -1) : (1 + imaginary_unit) / imaginary_unit = 1 - imaginary_unit :=
by
  -- Proof is not required according to the instructions
  sorry

end simplify_complex_number_l680_680454


namespace probability_odd_divisor_15_l680_680029

theorem probability_odd_divisor_15! :
  let n := 15 !
  let num_factors := (11 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  let odd_factors := (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  let p_odd := odd_factors / num_factors
  p_odd = 1 / 12 :=
by {
  let n : ℕ := 15 !
  let num_factors := (11 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  let odd_factors := (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  let p_odd := odd_factors / num_factors
  show p_odd = 1 / 12
  sorry
}

end probability_odd_divisor_15_l680_680029


namespace factorization_correct_l680_680006

-- Definitions of the given expressions
def optionA (x : ℝ) : Prop := (x + 3) * (x - 3) = x^2 - 9
def optionB (x : ℝ) : Prop := x^3 - 1 = x * (x^2 - 1 / x)
def optionC (x : ℝ) : Prop := x^2 - 3 * x - 4 = x * (x - 3) - 4
def optionD (x : ℝ) : Prop := x^2 - 4 * x + 4 = (x - 2)^2

-- The target theorem
theorem factorization_correct : optionD :=
by sorry

end factorization_correct_l680_680006


namespace Nell_has_210_cards_left_l680_680574

-- Define the initial amount of cards Nell had
def initial_cards : ℕ := 573

-- Define the cards given to John and Jeff
def cards_to_John : ℕ := 195
def cards_to_Jeff : ℕ := 168

-- Define the total number of cards given away
def total_given_away : ℕ := cards_to_John + cards_to_Jeff

-- Define the number of cards left
def cards_left : ℕ := initial_cards - total_given_away

theorem Nell_has_210_cards_left (initial_cards = 573) (cards_to_John = 195) (cards_to_Jeff = 168) :
  cards_left = 210 :=
by
  sorry

end Nell_has_210_cards_left_l680_680574


namespace find_angle4_l680_680794

theorem find_angle4
  (angle1 angle2 angle3 angle4 : ℝ)
  (h1 : angle1 = 70)
  (h2 : angle2 = 110)
  (h3 : angle3 = 40)
  (h4 : angle2 + angle3 + angle4 = 180) :
  angle4 = 30 := 
  sorry

end find_angle4_l680_680794


namespace sum_possible_b_lengths_l680_680167

theorem sum_possible_b_lengths (ABCD_is_convex : Quadrilateral ABCD)
  (AB_eq_20 : AB = 20) (AB_is_longest : ∀ S ∈ {A, B, C, D}, AB ≥ S.length)
  (AB_parallel_CD : AB ∥ CD) 
  (geom_progression_perims : geom_prog (perimeter ABC) (perimeter BCD) (perimeter ABCD))
  (sides_ABC : sides ABC = {20, a, b} ∧ ∠B = 120) :
  ∑ possible_lengths_of_b ≠ 20 = 80 :=
sorry

end sum_possible_b_lengths_l680_680167


namespace least_integer_in_ratio_l680_680077

theorem least_integer_in_ratio :
  ∃ (x : ℝ), (15 * x = 100) ∧ (x = 20/3) :=
begin
  use (20 / 3),
  split,
  { linarith },
  { refl },
end

end least_integer_in_ratio_l680_680077


namespace range_of_a_l680_680497

variable {x a : ℝ}

def f (x : ℝ) (a : ℝ) : ℝ := -x^2 + 2*a*x
def g (x : ℝ) (a : ℝ) : ℝ := (a + 1)^(1 - x)

theorem range_of_a (h_decreasing_f : ∀ x ∈ set.Icc 1 2, f x a < 0)
                   (h_decreasing_g : ∀ x ∈ set.Icc 1 2, g x a < 0) :
  0 < a ∧ a ≤ 1 :=
sorry

end range_of_a_l680_680497


namespace triangle_AEG_area_l680_680963

-- Given conditions
variables (A B C D E F G H : Type)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables [metric_space E] [metric_space F] [metric_space G] [metric_space H]
variables (area : ℝ)
variables (midpoint : Type → Type → Type)
variables (AG AD : ℝ)
variables (DG BH : ℝ)
variables (AB CD : ℝ)
variables (rectABCD : AB × CD = 24)

-- Definitions from conditions
def is_midpoint (M : midpoint AB CD) (A B : ℝ) := M = (A + B) / 2
def is_parallel (A B C D : ℝ) := (A = B + C ∨ A = B - C)
def segment_ratio_4 (AG AD : ℝ) := DG = 3 * AG ∧ AD = 4 * AG
def height_unchanged_from_A (A F : ℝ) := A = F
def EF_parallel_to_AD := is_parallel (F : midpoint AB CD) (A D)
def area_of_rectangle := area = AB * CD
def divides_into_4 (G A : ℝ) := AG = AD / 4 ∧ BH = CD / 4

-- Prove that the area of triangle AEG is 1 square unit given conditions
theorem triangle_AEG_area :
  divides_into_4 AG AD → is_midpoint E AB → is_midpoint F CD → EF_parallel_to_AD → area_of_rectangle AB CD →
  is_parallel A B E F → segment_ratio_4 AG AD →
  height_unchanged_from_A A (height_from A to EF) → area = 1 :=
by sorry

end triangle_AEG_area_l680_680963


namespace smallest_k_for_regular_tetrahedron_l680_680289

/-- Theorem: The smallest natural number k for which the statement 
"If a tetrahedron has k edge angles that are 60 degrees, then the tetrahedron must be regular"
is k = 7. -/
theorem smallest_k_for_regular_tetrahedron (k : ℕ) : 
  (∀ (tetra : Tetrahedron), (∃ n, n ≥ k ∧ tetra.has_n_edge_angles_eq_60 n) → tetra.is_regular) ↔ k = 7 := 
sorry

end smallest_k_for_regular_tetrahedron_l680_680289


namespace simplified_expression_eq_l680_680599

noncomputable def simplify_expression : ℚ :=
  1 / (1 / (1 / 3)^1 + 1 / (1 / 3)^2 + 1 / (1 / 3)^3)

-- We need to prove that the simplifed expression is equal to 1 / 39
theorem simplified_expression_eq : simplify_expression = 1 / 39 :=
by sorry

end simplified_expression_eq_l680_680599


namespace sum_equiv_l680_680733

theorem sum_equiv {f : ℕ → ℕ → ℝ} (h : ∀ (n k : ℕ), n ≥ 3 ∧ 1 ≤ k ∧ k ≤ n - 2 → f n k = (k^2) / (3^(n+k))) :
  (∑' n=3, ∑' k=1, if h : k ≤ n - 2 then f n k else 0) = 135 / 512 :=
by sorry

end sum_equiv_l680_680733


namespace smallest_constant_l680_680428

theorem smallest_constant (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  (a^2 + b^2 + a * b) / c^2 ≥ 3 / 4 :=
sorry

end smallest_constant_l680_680428


namespace exists_even_acquaintances_l680_680959

-- Define a structure for representing mutual acquaintances
structure Person :=
(knows : Person → Prop)

def has_even_number_of_acquaintances {n : ℕ} (people : Fin n → Person) (i : Fin n) : Prop :=
  even (Finset.filter (people i).knows (Finset.univ (Fin n))).card

theorem exists_even_acquaintances :
  ∃ i j : Fin 50, i ≠ j ∧ has_even_number_of_acquaintances (fun i => Person) i ∧ has_even_number_of_acquaintances (fun i => Person) j := 
sorry

end exists_even_acquaintances_l680_680959


namespace tangent_line_eq_l680_680120

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then log (-x) + 3 * x else - log x + 3 * x

theorem tangent_line_eq (f_odd : ∀ x : ℝ, f (-x) = - f x) :
  tangent_line_at f 1 = (2 : ℝ) * (x - 1) + (3 : ℝ) :=
by
  sorry

end tangent_line_eq_l680_680120


namespace program_output_l680_680622

def prog_loop (x y z n : ℕ) : ℕ × ℕ :=
  if z > 7000 then (n, z)
  else prog_loop (x + 2) (2 * y) (z + (x * y)) (n + 1)

theorem program_output :
  let init_x := 1
  let init_y := 1
  let init_z := 0
  let init_n := 0
  let (n, z) := prog_loop init_x init_y init_z init_n
  n = 8 ∧ z = 7682 :=
by {
  -- proof would go here
  sorry
}

end program_output_l680_680622


namespace no_5_7_8_9_three_digit_numbers_l680_680143

theorem no_5_7_8_9_three_digit_numbers : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
  ∀ d ∈ int_to_digits n, d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 8 ∧ d ≠ 9) → 
  nat.card {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∀ d ∈ int_to_digits n, d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 8 ∧ d ≠ 9} = 180 :=
by
  sorry

end no_5_7_8_9_three_digit_numbers_l680_680143


namespace intersection_is_unique_l680_680033

noncomputable def log3 := real.log

def f (x : ℝ) := 3 * log3 x
def g (x : ℝ) := log3 (5 * x)

theorem intersection_is_unique :
  ∃! x : ℝ, f x = g x := by
  sorry

end intersection_is_unique_l680_680033


namespace sum_of_solutions_l680_680290

theorem sum_of_solutions (x : ℝ) : (x^2 - 7*x + 20 = 0) -> ∑ (x : ℝ), x = 7 :=
by
  sorry

end sum_of_solutions_l680_680290


namespace new_perimeter_is_60_l680_680673

-- Define the initial conditions
def width : ℝ := 10
def original_area : ℝ := 150
def scaling_factor : ℝ := 4 / 3
def original_length := original_area / width
def new_area := original_area * scaling_factor
def new_length := new_area / width

-- Define the question and answer
def new_perimeter := 2 * (new_length + width)

theorem new_perimeter_is_60 : new_perimeter = 60 := 
by
  -- Proof would go here
  sorry

end new_perimeter_is_60_l680_680673


namespace range_of_f_l680_680748

noncomputable def f (x y z : ℝ) := ((x * y + y * z + z * x) * (x + y + z)) / ((x + y) * (y + z) * (z + x))

theorem range_of_f :
  ∃ x y z : ℝ, (0 < x ∧ 0 < y ∧ 0 < z) ∧ f x y z = r ↔ 1 ≤ r ∧ r ≤ 9 / 8 :=
sorry

end range_of_f_l680_680748


namespace sam_fish_count_l680_680953

/-- Let S be the number of fish Sam has. -/
def num_fish_sam : ℕ := sorry

/-- Joe has 8 times as many fish as Sam, which gives 8S fish. -/
def num_fish_joe (S : ℕ) : ℕ := 8 * S

/-- Harry has 4 times as many fish as Joe, hence 32S fish. -/
def num_fish_harry (S : ℕ) : ℕ := 32 * S

/-- Harry has 224 fish. -/
def harry_fish : ℕ := 224

/-- Prove that Sam has 7 fish given the conditions above. -/
theorem sam_fish_count : num_fish_harry num_fish_sam = harry_fish → num_fish_sam = 7 := by
  sorry

end sam_fish_count_l680_680953


namespace sum_of_reciprocals_of_roots_l680_680942

-- Defining the problem with conditions and required proof in Lean 4
theorem sum_of_reciprocals_of_roots 
  (a b c d : ℝ)
  (h : ∀ z : ℂ, is_root (λ z, z^4 + a*z^3 + b*z^2 + c*z + d) z → |z| = 2):
  (∑ z in {z | is_root (λ z, z^4 + a*z^3 + b*z^2 + c*z + d) z}, 1/z) = -a / 2  :=
by
  sorry

end sum_of_reciprocals_of_roots_l680_680942


namespace a_4_is_zero_l680_680132

def a_n (n : ℕ) : ℕ := n^2 - 2*n - 8

theorem a_4_is_zero : a_n 4 = 0 := 
by
  sorry

end a_4_is_zero_l680_680132


namespace probability_of_rolling_2_4_6_l680_680657

def fair_eight_sided_die : ℕ := 8
def successful_outcomes : set ℕ := {2, 4, 6}
def num_successful_outcomes : ℕ := successful_outcomes.to_finset.card

theorem probability_of_rolling_2_4_6 :
  (num_successful_outcomes : ℚ) / fair_eight_sided_die = 3 / 8 :=
by
  -- Note: The proof is omitted by using 'sorry'
  sorry

end probability_of_rolling_2_4_6_l680_680657


namespace captain_position_and_ab_next_to_each_other_l680_680954

theorem captain_position_and_ab_next_to_each_other :
  ∃ captain_positions ab_next_to_each_other total_permutations,
    captain_positions = 2 ∧ 
    ab_next_to_each_other = 2 ∧ 
    total_permutations = 24 ∧ 
    (captain_positions * ab_next_to_each_other * total_permutations = 96) :=
by {
  use (2, 2, 24),
  repeat { split },
  { refl },
  { refl },
  { refl },
  { norm_num }
}

end captain_position_and_ab_next_to_each_other_l680_680954


namespace product_is_even_l680_680931

-- Definitions are captured from conditions

noncomputable def is_permutation {α : Type*} [DecidableEq α] (l1 l2 : List α) : Prop :=
  l1.perm l2

theorem product_is_even (a : ℕ → ℕ) :
  (is_permutation (List.range 2015) (List.ofFn (λ i, a i + 2014))) →
  Even (Finset.univ.prod (λ i : Fin 2015, a i - i.val.succ)) :=
by
  sorry

end product_is_even_l680_680931


namespace conclusion_l680_680925

noncomputable def partial_sum (a : ℕ → ℝ) (n : ℕ) : ℝ := (finset.range (n+1)).sum a

variables (a : ℕ → ℝ) (d : ℝ)

-- Given an arithmetic sequence {a_n} with difference d
-- Defined as: a_0 = a_0, a_(n+1) = a_n + d
def is_arithmetic_sequence : Prop := ∀ n : ℕ, a (n + 1) = a n + d

-- S_n is the sum of the first n terms of {a_n}
def S (n : ℕ) : ℝ := partial_sum a n

-- Given conditions
axiom S_n_conditions : 
  is_arithmetic_sequence a d ∧
  S a 5 < S a 6 ∧
  S a 6 = S a 7 ∧
  S a 7 > S a 8

-- The conclusion to prove
theorem conclusion : S a 9 ≤ S a 5 :=
sorry

end conclusion_l680_680925


namespace pentagon_area_is_18_5_l680_680418

def shoelace_area (vertices : List (ℝ × ℝ)) : ℝ :=
  (1 / 2) * ( (vertices.zip (vertices.drop 1 ++ [vertices.head!])).sum
  (λ ⟨(x1, y1), (x2, y2)⟩, (x1 * y2 - y1 * x2)).abs )

theorem pentagon_area_is_18_5 :
  let vertices := [ (0, 0), (4, 0), (5, 3), (2, 5), (0, 3) ]
  shoelace_area vertices = 18.5 := by
  -- We'll skip the proof here
  sorry

end pentagon_area_is_18_5_l680_680418


namespace Taehyung_mother_age_l680_680248

theorem Taehyung_mother_age (Taehyung_young_brother_age : ℕ) (Taehyung_age_diff : ℕ) (Mother_age_diff : ℕ) (H1 : Taehyung_young_brother_age = 7) (H2 : Taehyung_age_diff = 5) (H3 : Mother_age_diff = 31) :
  ∃ (Mother_age : ℕ), Mother_age = 43 := 
by
  have Taehyung_age : ℕ := Taehyung_young_brother_age + Taehyung_age_diff
  have Mother_age := Taehyung_age + Mother_age_diff
  existsi (Mother_age)
  sorry

end Taehyung_mother_age_l680_680248


namespace probability_Cecilia_rolls_4_given_win_l680_680888

noncomputable def P_roll_Cecilia_4_given_win : ℚ :=
  let P_C1_4 := 1/6
  let P_W_C := 1/5
  let P_W_C_given_C1_4 := (4/6)^4
  let P_C1_4_and_W_C := P_C1_4 * P_W_C_given_C1_4
  let P_C1_4_given_W_C := P_C1_4_and_W_C / P_W_C
  P_C1_4_given_W_C

theorem probability_Cecilia_rolls_4_given_win :
  P_roll_Cecilia_4_given_win = 256 / 1555 :=
by 
  -- Here the proof would go, but we include sorry for now.
  sorry

end probability_Cecilia_rolls_4_given_win_l680_680888


namespace isosceles_triangle_height_l680_680362

theorem isosceles_triangle_height (l w h : ℝ) 
  (h1 : l * w = (1 / 2) * w * h) : h = 2 * l :=
by
  sorry

end isosceles_triangle_height_l680_680362


namespace find_g_l680_680560

noncomputable def f (x : ℝ) : ℝ := sorry  -- Define the nonzero polynomial f(x)
noncomputable def g (x : ℝ) : ℝ := x^2 + 3 * x  -- We need to prove that g(x) = x^2 + 3x

theorem find_g (h1 : ∀ (x : ℝ), f(g(x)) = f(x) * g(x))
  (h2 : g(2) = 10) :
  g = (λ x, x^2 + 3 * x) :=
by 
  sorry

end find_g_l680_680560


namespace cos_x1_minus_x2_l680_680453

theorem cos_x1_minus_x2 {x1 x2 : ℝ} 
  (h1 : 0 < x1) (h2 : x1 < x2) (h3 : x2 < π)
  (h4 : sin (2 * x1 - π / 3) = 4 / 5) 
  (h5 : sin (2 * x2 - π / 3) = 4 / 5) :
  cos (x1 - x2) = 3 / 5 :=
sorry

end cos_x1_minus_x2_l680_680453


namespace product_even_permutation_l680_680933

theorem product_even_permutation (a : Fin 2015 → ℕ) (h : ∀ i j, i ≠ j → a i ≠ a j)
    (range_a : {x // 2015 ≤ x ∧ x ≤ 4029}): 
    ∃ i, (a i - (i + 1)) % 2 = 0 :=
by
  sorry

end product_even_permutation_l680_680933


namespace width_percentage_increase_l680_680625

theorem width_percentage_increase (l w : ℝ) (hl_gt_zero : l > 0) (hw_gt_zero : w > 0) :
  let l' := 1.4 * l in
  let A := l * w in
  let new_desired_area := 2 * A in
  let y := (2 * l * w) / (1.4 * l * w) in
  (y - 1) * 100 = 42.857 :=
by
  sorry

end width_percentage_increase_l680_680625


namespace geometry_problem_l680_680693

noncomputable def circle_radius := √3
noncomputable def kc_length := 1
noncomputable def al_length := 4
noncomputable def angle_acb := 2 * Real.pi / 3
noncomputable def mk_length := 3
noncomputable def ab_length := 5 * Real.sqrt 7 / 2
noncomputable def area_cmn := 45 * Real.sqrt 3 / 28

theorem geometry_problem :
  let O := (0, 0) in
  let k := (O.1 + kc_length, O.2 + circle_radius) in
  let l := (O.1 + al_length, O.2 - circle_radius) in
  let m := (O.1 + 1, O.2 + 1) in -- these will be found as appropriate
  let n := (O.1 + 2, O.2 + 1) in -- these will be found as appropriate
  -- Given conditions:
  circle_radius = √3 →
  kc_length = 1 →
  al_length = 4 →
  -- Expected results:  
  ∠(ACB) = angle_acb ∧
  segment_length MK = mk_length ∧
  segment_length AB = ab_length ∧
  area_triangle CMN = area_cmn :=
by
  sorry

end geometry_problem_l680_680693


namespace line_passing_through_points_l680_680620

-- Definition of points
def point1 : ℝ × ℝ := (1, 0)
def point2 : ℝ × ℝ := (0, -2)

-- Definition of the line equation
def line_eq (x y : ℝ) : Prop := 2 * x - y - 2 = 0

-- Theorem statement
theorem line_passing_through_points : 
  line_eq point1.1 point1.2 ∧ line_eq point2.1 point2.2 :=
by
  sorry

end line_passing_through_points_l680_680620


namespace snacks_combination_count_l680_680351

def snacks := ["apple", "orange", "banana", "granola bar"]

theorem snacks_combination_count : (nat.choose 4 2) = 6 := by
  sorry

end snacks_combination_count_l680_680351


namespace range_of_first_term_l680_680634

theorem range_of_first_term 
  {a_n : ℕ → ℝ} (S_n : ℕ → ℝ)
  (h_Sn : ∀ n, S_n n = (∑ i in Finset.range n, a_n i))
  (h_lim : filter.tendsto S_n filter.at_top (𝓝 (1 / 2))) :
  ∃ a_1 : ℝ, (a_1 ∈ set.Ioo 0 (1 / 2) ∪ set.Ioo (1 / 2) 1) := 
sorry

end range_of_first_term_l680_680634


namespace locus_of_M_in_equilateral_triangle_l680_680431

theorem locus_of_M_in_equilateral_triangle (ABC : Type*) [inner_product_space ℝ ABC]
  (A B C M : ABC)
  (hABC : ∀ (X Y : ABC), dist X Y = dist (perp (same_span (A - B) (A - C)) X Y) (perp (same_span (A - B) (A - C)) X Y))
  (D E F : ABC)
  (hD : orthogonal_projection (line_span (BC : set ABC)) M = D)
  (hE : orthogonal_projection (line_span (CA : set ABC)) M = E)
  (hF : orthogonal_projection (line_span (AB : set ABC)) M = F) :
  (angle F D E = π/2) ↔
  (∃ (O : ABC) (r : ℝ), circle (O, r) ∧ arc (O, r) B C ∧ angle B M C = 5 * π / 6) :=
sorry

end locus_of_M_in_equilateral_triangle_l680_680431


namespace russian_needed_goals_equals_tunisian_scored_goals_l680_680986

-- Define the total goals required by each team
def russian_goals := 9
def tunisian_goals := 5

-- Statement: there exists a moment where the Russian remaining goals equal the Tunisian scored goals
theorem russian_needed_goals_equals_tunisian_scored_goals :
  ∃ n : ℕ, n ≤ russian_goals ∧ (russian_goals - n) = (tunisian_goals) := by
  sorry

end russian_needed_goals_equals_tunisian_scored_goals_l680_680986


namespace sum_of_squares_of_roots_of_quadratic_l680_680779

theorem sum_of_squares_of_roots_of_quadratic :
  (∀ (s₁ s₂ : ℝ), (s₁ + s₂ = 15) → (s₁ * s₂ = 6) → (s₁^2 + s₂^2 = 213)) :=
by
  intros s₁ s₂ h_sum h_prod
  sorry

end sum_of_squares_of_roots_of_quadratic_l680_680779


namespace rational_expression_nonnegative_l680_680082

noncomputable def f (x : ℝ) : ℝ := 
  (x * (1 - 8 * x) * (1 - 7 * x)) / ((2 - x) * (x^2 + 2 * x + 5))

theorem rational_expression_nonnegative :
  ∀ x : ℝ, (0 ≤ x ∧ x ≤ 1/8) → f x ≥ 0 :=
begin
  sorry
end

end rational_expression_nonnegative_l680_680082


namespace simplified_expression_eq_l680_680600

noncomputable def simplify_expression : ℚ :=
  1 / (1 / (1 / 3)^1 + 1 / (1 / 3)^2 + 1 / (1 / 3)^3)

-- We need to prove that the simplifed expression is equal to 1 / 39
theorem simplified_expression_eq : simplify_expression = 1 / 39 :=
by sorry

end simplified_expression_eq_l680_680600


namespace sum_of_squares_of_ages_eq_35_l680_680643

theorem sum_of_squares_of_ages_eq_35
  (d t h : ℕ)
  (h1 : 3 * d + 4 * t = 2 * h + 2)
  (h2 : 2 * d^2 + t^2 = 6 * h)
  (relatively_prime : Nat.gcd (Nat.gcd d t) h = 1) :
  d^2 + t^2 + h^2 = 35 := 
sorry

end sum_of_squares_of_ages_eq_35_l680_680643


namespace max_true_statements_l680_680927

theorem max_true_statements (a b : ℝ) :
  ((a < b) → (b < 0) → (a < 0) → ¬(1 / a < 1 / b)) ∧
  ((a < b) → (b < 0) → (a < 0) → ¬(a^2 < b^2)) →
  3 = 3
:=
by
  intros
  sorry

end max_true_statements_l680_680927


namespace coin_stack_compositions_l680_680895

noncomputable def thickness_penny : Real := 1.60
noncomputable def thickness_nickel : Real := 1.90
noncomputable def thickness_dime : Real := 1.40
noncomputable def thickness_quarter : Real := 1.80
noncomputable def stack_height : Real := 15.2

theorem coin_stack_compositions :
  ∃ (p n d q : Nat),
    1.60 * p + 1.90 * n = 15.2 ∧
    1.40 * d + 1.80 * q = 15.2 ∧
    p = 5 ∧ n = 3 ∧
    d = 2 ∧ q = 7 := by
  sorry

end coin_stack_compositions_l680_680895


namespace second_odd_integer_is_72_l680_680272

def consecutive_odd_integers (n : ℤ) : ℤ × ℤ × ℤ :=
  (n - 2, n, n + 2)

theorem second_odd_integer_is_72 (n : ℤ) (h : (n - 2) + (n + 2) = 144) : n = 72 :=
by {
  sorry
}

end second_odd_integer_is_72_l680_680272


namespace S_9_equals_72_l680_680206

-- Definitions
def a_n (a1 d : ℕ) (n : ℕ) : ℕ := a1 + (n - 1) * d

def S_n (a1 d n : ℕ) : ℕ := n * a1 + (n * (n - 1) // 2) * d

-- Conditions
def arithmetic_seq_cond (a1 d : ℕ) : Prop :=
  a_n a1 d 2 + a_n a1 d 4 + a_n a1 d 9 = 24

-- Proof statement
theorem S_9_equals_72 {a1 d : ℕ} (h : arithmetic_seq_cond a1 d) : S_n a1 d 9 = 72 := by
  sorry

end S_9_equals_72_l680_680206


namespace students_per_class_l680_680691

variable (c : ℕ) (s : ℕ)

def books_per_month := 6
def months_per_year := 12
def books_per_year := books_per_month * months_per_year
def total_books_read := 72

theorem students_per_class : (s * c = 1 ∧ s * books_per_year = total_books_read) → s = 1 := by
  intros h
  have h1: books_per_year = total_books_read := by
    calc
      books_per_year = books_per_month * months_per_year := rfl
      _ = 6 * 12 := rfl
      _ = 72 := rfl
  sorry

end students_per_class_l680_680691


namespace set_elements_count_l680_680549

theorem set_elements_count (X Y : Set ℝ) [Fintype X] [Fintype Y] (hX : ∀ x ∈ X, 0 ≤ x ∧ x < 1)
    (hY : ∀ y ∈ Y, 0 ≤ y ∧ y < 1) (h0 : 0 ∈ X ∧ 0 ∈ Y) (hx_y : ∀ x ∈ X, ∀ y ∈ Y, x + y ≠ 1) :
    Fintype.card ({ z : ℝ | ∃ x ∈ X, ∃ y ∈ Y, z = x + y - Real.floor (x + y) }) ≥ Fintype.card X + Fintype.card Y - 1 :=
by
  sorry

end set_elements_count_l680_680549


namespace inequality_for_positive_reals_l680_680595

noncomputable def ϕ (x : ℝ) : ℝ := (1 / Real.sqrt (2 * Real.pi)) * Real.exp (-(x^2) / 2)

noncomputable def Φ (x : ℝ) : ℝ := ∫ y in -∞..x, ϕ(y)

theorem inequality_for_positive_reals (x : ℝ) (hx : 0 < x) :
  (x / (1 + x^2)) * ϕ(x) < 1 - Φ(x) ∧ 1 - Φ(x) < ϕ(x) / x := sorry

end inequality_for_positive_reals_l680_680595


namespace equation_of_ellipse_correct_minimum_slope_of_MN_correct_l680_680106

noncomputable def equation_of_ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 0) : Prop :=
  (a^2 = 4) ∧ (b^2 = 3) ∧ (∀ x y : ℝ, (x, y) = (1, 3 / 2) → ((x ^ 2) / 4) + ((y ^ 2) / 3) = 1)

noncomputable def minimum_slope_of_MN (y0 : ℝ) (h : y0 > 0) : ℝ :=
  -2 * y0 / (3 * y0^2 + 16)

theorem equation_of_ellipse_correct : 
  ∀ (a b : ℝ) (h1 : a > b) (h2 : b > 0), 
    equation_of_ellipse a b h1 h2 :=
by
  sorry

theorem minimum_slope_of_MN_correct : 
  ∀ (y0 : ℝ) (h : y0 > 0), 
    minimum_slope_of_MN y0 h = - (Real.sqrt 3) / 12 :=
by
  sorry

end equation_of_ellipse_correct_minimum_slope_of_MN_correct_l680_680106


namespace angle_of_inclination_of_line_l680_680254

theorem angle_of_inclination_of_line (a b : ℝ) (h : b = a - 1) : 
  let θ := Real.arctan 1 in θ = π / 4 :=
by
  have θ := Real.arctan 1
  show θ = π / 4
  sorry

end angle_of_inclination_of_line_l680_680254


namespace average_gas_mileage_round_trip_l680_680319

-- necessary definitions related to the problem conditions
def total_distance_one_way := 150
def fuel_efficiency_going := 35
def fuel_efficiency_return := 30
def round_trip_distance := total_distance_one_way + total_distance_one_way

-- calculation of gasoline used for each trip and total usage
def gasoline_used_going := total_distance_one_way / fuel_efficiency_going
def gasoline_used_return := total_distance_one_way / fuel_efficiency_return
def total_gasoline_used := gasoline_used_going + gasoline_used_return

-- calculation of average gas mileage
def average_gas_mileage := round_trip_distance / total_gasoline_used

-- the final theorem to prove the average gas mileage for the round trip 
theorem average_gas_mileage_round_trip : average_gas_mileage = 32 := 
by
  sorry

end average_gas_mileage_round_trip_l680_680319


namespace cross_section_fraction_of_surface_area_l680_680332

theorem cross_section_fraction_of_surface_area (r : ℝ) :
  let circle_area := π * (r / 2) ^ 2 in
  let sphere_area := 4 * π * r ^ 2 in
  circle_area / sphere_area = 1 / 4 :=
by
  sorry

end cross_section_fraction_of_surface_area_l680_680332


namespace area_ADC_rounded_to_nearest_integer_is_1458_l680_680900

noncomputable def triangle_ABC_area_ADC : ℝ :=
let AB : ℝ := 120
let angle_ABC : ℝ := 90
let AD_is_angle_bisector := true
let BC (x : ℝ) : ℝ := x
let AC (x : ℝ) : ℝ := 3 * x - 9
let x := (6.75 + sqrt (7205.0625)) / 2
let BC_val : ℝ := BC x
let AC_val : ℝ := AC x
let DC : ℝ := AC_val / (1 + AB / AC_val) * BC_val / (AB + AC_val)
AB * DC * 1/2

theorem area_ADC_rounded_to_nearest_integer_is_1458 :
triangle_ABC_area_ADC = 1458 := by
sorry

end area_ADC_rounded_to_nearest_integer_is_1458_l680_680900


namespace pascal_no_divisible_by_prime_iff_form_l680_680554

theorem pascal_no_divisible_by_prime_iff_form (p : ℕ) (n : ℕ) 
  (hp : Nat.Prime p) :
  (∀ k ≤ n, Nat.choose n k % p ≠ 0) ↔ ∃ s q : ℕ, s ≥ 0 ∧ 0 < q ∧ q < p ∧ n = p^s * q - 1 :=
by
  sorry

end pascal_no_divisible_by_prime_iff_form_l680_680554


namespace total_ticket_cost_l680_680602

theorem total_ticket_cost :
  ∀ (A : ℝ), 
  -- Conditions
  (6 : ℝ) * (5 : ℝ) + (2 : ℝ) * A = 50 :=
by
  sorry

end total_ticket_cost_l680_680602


namespace repeating_decimal_to_fraction_l680_680385

theorem repeating_decimal_to_fraction :
  let x := 0.565656... (or equivalently written \(0.\overline{56}\))
  in x = 56 / 99 :=
sorry

end repeating_decimal_to_fraction_l680_680385


namespace circumscribed_inscribed_center_l680_680587

variables {Point : Type} [MetricSpace Point] (A B C D : Point)

-- Definitions of distances
def dist (p q : Point) : ℝ := sorry 

def is_circumscribed_center (O : Point) (A B C D : Point) : Prop := sorry
def is_inscribed_center (I : Point) (A B C D : Point) : Prop := sorry

-- Main statement
theorem circumscribed_inscribed_center (A B C D : Point) :
  (∃ T : Point, is_circumscribed_center T A B C D ∧ 
                is_inscribed_center T A B C D) ↔ 
  (dist A B = dist C D ∧ 
   dist A C = dist B D ∧ 
   dist A D = dist B C) := sorry

end circumscribed_inscribed_center_l680_680587


namespace find_fraction_of_original_flow_rate_l680_680715

noncomputable def fraction_of_original_flow_rate (f : ℚ) : Prop :=
  let original_flow_rate := 5
  let reduced_flow_rate := 2
  reduced_flow_rate = f * original_flow_rate - 1

theorem find_fraction_of_original_flow_rate : ∃ (f : ℚ), fraction_of_original_flow_rate f ∧ f = 3 / 5 :=
by
  sorry

end find_fraction_of_original_flow_rate_l680_680715


namespace fourth_power_sum_l680_680139

theorem fourth_power_sum (a b c : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a^2 + b^2 + c^2 = 3) 
  (h3 : a^3 + b^3 + c^3 = 6) : 
  a^4 + b^4 + c^4 = 4.5 :=
by
  sorry

end fourth_power_sum_l680_680139


namespace max_triangles_when_moving_l680_680641

-- Definitions for the conditions:
def equilateral_triangle := sorry -- This would be a formal definition of an equilateral triangle.
def midpoint (s: LineSegment) := sorry -- This would be a formal definition of the midpoint of a line segment.
def line_segment_midpoints_eq (t : equilateral_triangle) : Prop := sorry -- Formalize that a line segment connects midpoints of two sides.
def symmetrical_about_vertical_axis (A B : equilateral_triangle) : Prop := sorry -- Formalize the symmetry condition.

-- Definition of the movement of triangle A relative to B:
def triangle_movement (A B : equilateral_triangle) : Prop  := sorry -- Define movement conditions and the formations of smaller triangles.

-- Main theorem to prove
theorem max_triangles_when_moving (A B : equilateral_triangle)
  (eq_size : A = B)
  (mid_eq : line_segment_midpoints_eq A)
  (symm : symmetrical_about_vertical_axis A B)
  (move : triangle_movement A B) :
  ∃ n, n = 11 :=
by
  sorry

end max_triangles_when_moving_l680_680641


namespace number_of_articles_sold_at_cost_price_l680_680617

-- Let C be the cost price of one article.
-- Let S be the selling price of one article.
-- Let X be the number of articles sold at cost price.

variables (C S : ℝ) (X : ℕ)

-- Condition 1: The cost price of X articles is equal to the selling price of 32 articles.
axiom condition1 : (X : ℝ) * C = 32 * S

-- Condition 2: The profit is 25%, so the selling price S is 1.25 times the cost price C.
axiom condition2 : S = 1.25 * C

-- The theorem we need to prove
theorem number_of_articles_sold_at_cost_price : X = 40 :=
by
  -- Proof here
  sorry

end number_of_articles_sold_at_cost_price_l680_680617


namespace power_expression_equals_twelve_l680_680867

variable (a m n : ℝ)

-- Conditions
def log_a_2_eq_m : Prop := log a 2 = m
def log_a_3_eq_n : Prop := log a 3 = n

theorem power_expression_equals_twelve (h1 : log_a_2_eq_m a m) (h2 : log_a_3_eq_n a n) : a^(2 * m + n) = 12 := sorry

end power_expression_equals_twelve_l680_680867


namespace factorize_expr_l680_680760

-- Define the expression
def expr (a : ℝ) := -3 * a + 12 * a^2 - 12 * a^3

-- State the theorem
theorem factorize_expr (a : ℝ) : expr a = -3 * a * (1 - 2 * a)^2 :=
by
  sorry

end factorize_expr_l680_680760


namespace rita_remaining_amount_l680_680971

theorem rita_remaining_amount :
  let num_dresses := 5
  let num_pants := 3
  let num_jackets := 4
  let cost_per_dress := 20
  let cost_per_pants := 12
  let cost_per_jacket := 30
  let additional_cost := 5
  let initial_amount := 400
  let total_cost := num_dresses * cost_per_dress + num_pants * cost_per_pants + num_jackets * cost_per_jacket + additional_cost
  remaining_amount = initial_amount - total_cost
  in remaining_amount = 139 :=
by
  let num_dresses := 5
  let num_pants := 3
  let num_jackets := 4
  let cost_per_dress := 20
  let cost_per_pants := 12
  let cost_per_jacket := 30
  let additional_cost := 5
  let initial_amount := 400
  let total_cost := num_dresses * cost_per_dress + num_pants * cost_per_pants + num_jackets * cost_per_jacket + additional_cost
  let remaining_amount := initial_amount - total_cost
  show remaining_amount = 139
  sorry

end rita_remaining_amount_l680_680971


namespace moles_NH4NO3_needed_l680_680772

theorem moles_NH4NO3_needed
  (moles_NaNO3 : ℕ) (moles_NaOH : ℕ) (moles_NH4NO3 : ℕ) 
  (balanced_eq : NH4NO3 + NaOH → NaNO3 + NH4OH) 
  (h1 : moles_NaNO3 = 2)
  (h2 : moles_NaOH = 2)
  (h3 : balanced_eq)
  (h4 : 1 * moles_NH4NO3 + 1 * moles_NaOH = 1 * moles_NaNO3) :
  moles_NH4NO3 = 2 :=
sorry

end moles_NH4NO3_needed_l680_680772


namespace number_less_than_its_reciprocal_l680_680754

theorem number_less_than_its_reciprocal (x : ℝ) (h₁ : x = 1/2) : x < 1 / x :=
by {
    rw h₁,
    norm_num,
    sorry
}

end number_less_than_its_reciprocal_l680_680754


namespace Roger_cookie_price_l680_680788

def Art_circle_radius := 2
def Roger_square_side := 4
def Art_cookies_count := 9
def Art_cookie_price := 50

theorem Roger_cookie_price :
  let Art_cookie_area := Real.pi * (Art_circle_radius ^ 2)
  let Art_total_area := Art_cookies_count * Art_cookie_area
  let Roger_cookie_area := Roger_square_side ^ 2
  let Roger_cookies_count := Art_total_area / Roger_cookie_area
  let Art_earnings := Art_cookies_count * Art_cookie_price
  let Roger_cookie_price := Art_earnings / Roger_cookies_count
  Roger_cookie_price ≈ 64 :=
by
  sorry

end Roger_cookie_price_l680_680788


namespace china_gdp_scientific_notation_l680_680349

theorem china_gdp_scientific_notation :
  ∃ a n : ℝ, (827000 : ℝ) = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 8.27 ∧ n = 5 := 
begin
  use [8.27, 5],
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { refl, },
  { refl, },
end

end china_gdp_scientific_notation_l680_680349


namespace determine_b_l680_680192

theorem determine_b (b : ℝ) (h_pos : 0 < b) (h : (λ x, b * x ^ 3 - 1) ((λ x, b * x ^ 3 - 1) 1) = -1) : b = 1 :=
by
  sorry

end determine_b_l680_680192


namespace inverse_of_ln_function_l680_680992

theorem inverse_of_ln_function (x : ℝ) (h : x > 1) : 
    ∃ y : ℝ, y = e^(x - 1) + 1 ∧ (y = 1 + Real.log(x - 1)) :=
by
  sorry

end inverse_of_ln_function_l680_680992


namespace root_interval_l680_680203

noncomputable def f (a b x : ℝ) : ℝ := 2 * a^x - b^x

theorem root_interval (a b : ℝ) (h₀ : 0 < a) (h₁ : b ≥ 2 * a) :
  ∃ x : ℝ, 0 < x ∧ x ≤ 1 ∧ f a b x = 0 := 
sorry

end root_interval_l680_680203


namespace mutual_funds_yield_range_l680_680548

theorem mutual_funds_yield_range
  (L : ℝ) -- Annual yield (low) last year
  (H : ℝ) -- Annual yield (high) last year
  (H_minus_L : H - L = 12500)
  (new_range_improved : ∀x ∈ {1, 51, 101}, x = 12 ∨ x = 17 ∨ x = 22) :
  let L_new := L * 1.12,
      H_new := H * 1.22
  in H_new - L_new = 27750 := 
begin
  sorry
end

end mutual_funds_yield_range_l680_680548


namespace find_angle_between_planes_l680_680015

noncomputable def angle_between_planes 
  (a α k : ℝ) 
  (α_pos : 0 < α) 
  (α_lt_90 : α < π / 2) :
  ℝ :=
  let sinα := Real.sin α in
   
  arctan (k / (2 * sinα))

theorem find_angle_between_planes 
  (a : ℝ)
  (α : ℝ)
  (k : ℝ)
  (hα_pos : 0 < α)
  (hα_lt_90 : α < π/2) :
  angle_between_planes a α k hα_pos hα_lt_90 = arctan (k / (2 * Real.sin α)) :=
by
  unfold angle_between_planes
  sorry

end find_angle_between_planes_l680_680015


namespace sum_of_squares_of_roots_of_quadratic_l680_680780

theorem sum_of_squares_of_roots_of_quadratic :
  (∀ (s₁ s₂ : ℝ), (s₁ + s₂ = 15) → (s₁ * s₂ = 6) → (s₁^2 + s₂^2 = 213)) :=
by
  intros s₁ s₂ h_sum h_prod
  sorry

end sum_of_squares_of_roots_of_quadratic_l680_680780


namespace divisor_of_difference_is_62_l680_680329

-- The problem conditions as definitions
def x : Int := 859622
def y : Int := 859560
def difference : Int := x - y

-- The proof statement
theorem divisor_of_difference_is_62 (d : Int) (h₁ : d ∣ y) (h₂ : d ∣ difference) : d = 62 := by
  sorry

end divisor_of_difference_is_62_l680_680329


namespace suitable_sampling_method_l680_680356

-- Conditions given
def num_products : ℕ := 40
def num_top_quality : ℕ := 10
def num_second_quality : ℕ := 25
def num_defective : ℕ := 5
def draw_count : ℕ := 8

-- Possible sampling methods
inductive SamplingMethod
| DrawingLots : SamplingMethod
| RandomNumberTable : SamplingMethod
| Systematic : SamplingMethod
| Stratified : SamplingMethod

-- Problem statement (to be proved)
theorem suitable_sampling_method : 
  (num_products = 40) ∧ 
  (num_top_quality = 10) ∧ 
  (num_second_quality = 25) ∧ 
  (num_defective = 5) ∧ 
  (draw_count = 8) → 
  SamplingMethod.Stratified = SamplingMethod.Stratified :=
by sorry

end suitable_sampling_method_l680_680356


namespace trigonometric_det_zero_l680_680947

theorem trigonometric_det_zero (A B C : ℝ) (h : A + B + C = π) :
  det ![
    ![cos A ^ 2, tan A, 1],
    ![cos B ^ 2, tan B, 1],
    ![cos C ^ 2, tan C, 1]
  ] = 0 :=
sorry

end trigonometric_det_zero_l680_680947


namespace total_stones_l680_680974

theorem total_stones (sent_away kept total : ℕ) (h1 : sent_away = 63) (h2 : kept = 15) (h3 : total = sent_away + kept) : total = 78 :=
by
  sorry

end total_stones_l680_680974


namespace number_of_pencils_is_48_l680_680997

-- Define the conditions
variable (pens pencils : ℕ)
variable (ratio_pens_pencils : 5 * pencils = 6 * pens)
variable (diff_pens_pencils : pencils = pens + 8)

-- State the problem: Prove there are 48 pencils.
theorem number_of_pencils_is_48 (h_ratio : ratio_pens_pencils) (h_diff : diff_pens_pencils) : pencils = 48 := by
  sorry

end number_of_pencils_is_48_l680_680997


namespace probability_of_picking_letter_in_mathematics_l680_680502

-- Definitions and conditions
def total_letters : ℕ := 26
def unique_letters_in_mathematics : ℕ := 8
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

-- Theorem to be proven
theorem probability_of_picking_letter_in_mathematics :
  probability unique_letters_in_mathematics total_letters = 4 / 13 :=
by
  sorry

end probability_of_picking_letter_in_mathematics_l680_680502


namespace four_planes_divide_space_into_fifteen_parts_l680_680579

theorem four_planes_divide_space_into_fifteen_parts :
  (4^3 + 5 * 4 + 6) / 6 = 15 := 
by
  -- defining variables
  let k := 4
  let formula := (k^3 + 5 * k + 6) / 6
  -- expressing the expected outcome
  have h : formula = 15
  sorry

end four_planes_divide_space_into_fifteen_parts_l680_680579


namespace sum_of_sequence_l680_680840

-- Define the sequence based on the given conditions
def sequence (n : ℕ) : ℕ → ℚ 
| 1     := 2
| (k+1) := sequence k + 1/3

-- Define the sum of the sequence
def sum_sequence (n : ℕ) : ℚ := 
  (Finset.range n).sum (λ k, sequence (k + 1))

-- Statement of the theorem to be proven
theorem sum_of_sequence (n : ℕ) : sum_sequence n = n * (n + 11) / 6 :=
sorry

end sum_of_sequence_l680_680840


namespace not_exists_a_b_l680_680199

open Real

theorem not_exists_a_b (n : ℕ) (f : ℝ → ℝ) (x : Fin n → ℝ) :
  (∀ i, -1 ≤ x i ∧ x i ≤ 1) →
  ¬ ∃ a b : ℝ, -1 < a ∧ a < 0 ∧ 0 < b ∧ b < 1 ∧ |f a| ≥ 1 ∧ |f b| ≥ 1 ∧
  (∀ x, f x = ∏ i, (x - (x i))) :=
by
  sorry

end not_exists_a_b_l680_680199


namespace nitrogen_highest_mass_percentage_ammonia_l680_680071

noncomputable def molar_mass_N : ℝ := 14.01
noncomputable def molar_mass_H : ℝ := 1.01
noncomputable def molar_mass_NH3 : ℝ := molar_mass_N + 3 * molar_mass_H
noncomputable def mass_percentage_N : ℝ := (molar_mass_N / molar_mass_NH3) * 100

theorem nitrogen_highest_mass_percentage_ammonia :
  mass_percentage_N ≈ 82.28 :=
by
  have h1 : molar_mass_H * 3 = 3.03 := by norm_num
  have h2 : molar_mass_NH3 = 17.04 := by
    simp [molar_mass_H, molar_mass_N, molar_mass_NH3, h1]
  have h3 : mass_percentage_N = (14.01 / 17.04) * 100 := by simp [mass_percentage_N, h2]
  exact h3

end nitrogen_highest_mass_percentage_ammonia_l680_680071


namespace pyramid_area_l680_680752

def total_area_of_pyramid (base_edge : ℝ) (lateral_edge : ℝ) : ℝ :=
  let h := real.sqrt (lateral_edge ^ 2 - ((base_edge / 2) ^ 2))
  let area_of_one_triangle := (1 / 2) * base_edge * h
  4 * area_of_one_triangle

theorem pyramid_area (h : 0 < base_edge) :
  total_area_of_pyramid 8 10 = 32 * real.sqrt 21 := by
  sorry

end pyramid_area_l680_680752


namespace cosine_inequality_l680_680439

theorem cosine_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 0 < x^2 + y^2 ∧ x^2 + y^2 ≤ π) :
  1 + Real.cos (x * y) ≥ Real.cos x + Real.cos y :=
sorry

end cosine_inequality_l680_680439


namespace max_sum_cannot_split_l680_680582

theorem max_sum_cannot_split (nums : Set ℕ) (total_sum : ℕ) : 
  (∀ k ∈ nums, 4 ≤ k ∧ k ≤ 14) ∧ total_sum = 99 ∧ 
  (∀ subset : Set ℕ, subset ⊆ nums → (total_sum₀ = subset.sum ∧ 
    ¬ (∃ groups : Set (Set ℕ), 
      groups.pairwise_disjoint id ∧ 
      (∀ g ∈ groups, g.sum = total_sum₀ / groups.size)) → 
  total_sum₀ = 91)) :=
by sorry

end max_sum_cannot_split_l680_680582


namespace find_positive_integer_divisible_by_24_between_7_9_and_8_l680_680064

theorem find_positive_integer_divisible_by_24_between_7_9_and_8 :
  ∃ (n : ℕ), n > 0 ∧ (24 ∣ n) ∧ (7.9 < real.cbrt n) ∧ (real.cbrt n < 8) :=
begin
  use 504,
  split,
  { exact nat.zero_lt_succ 503, },
  split,
  { use 21, norm_num, },
  split,
  { norm_num, },
  { norm_num, },
end

end find_positive_integer_divisible_by_24_between_7_9_and_8_l680_680064


namespace prob_ineq_l680_680474

open Probability

variable {σ : ℝ} (a : ℝ)
-- Assume the random variable X follows a normal distribution with mean 3 and variance σ^2
def X := normal 3 σ

-- Given condition
axiom h1 : ∀ t : ℝ, P(X ≤ t) = normalCDF 3 σ t
axiom h2 : P(X < a) = 0.4

theorem prob_ineq : P(a ≤ X ∧ X < (6 - a)) = 0.2 :=
by
  sorry

end prob_ineq_l680_680474


namespace cube_inverse_sum_l680_680301

theorem cube_inverse_sum (m : ℝ) (h : m^2 - 8 * m + 1 = 0) : m^3 + m^(-3) = 61 := 
by 
  sorry

end cube_inverse_sum_l680_680301


namespace triangle_AXY_is_obtuse_l680_680990

variables (A B C D X Y : Type)
-- Define the conditions regarding inscribed and exscribed spheres.
-- The following definitions assume existence of geometric configurations.

-- X is the touchpoint of the inscribed sphere on face BCD.
-- Y is the touchpoint of the exscribed sphere on face BCD.
-- X and Y are different points on face BCD.

axiom inscribed_sphere_touch (X : Type) : Prop
axiom exscribed_sphere_touch (Y : Type) : Prop

axiom different_points (X Y : Type) : X ≠ Y

-- The theorem we want to prove.
theorem triangle_AXY_is_obtuse (h_inscribed : inscribed_sphere_touch X)
  (h_exscribed : exscribed_sphere_touch Y) (h_diff : different_points X Y) :
  ∃ (angle_axy : Type), angle_axy > 90 :=
sorry

end triangle_AXY_is_obtuse_l680_680990


namespace find_f_5_l680_680121

section
variables (f : ℝ → ℝ)

-- Given condition
def functional_equation (x : ℝ) : Prop := x * f x = 2 * f (1 - x) + 1

-- Prove that f(5) = 1/12 given the condition
theorem find_f_5 (h : ∀ x, functional_equation f x) : f 5 = 1 / 12 :=
sorry
end

end find_f_5_l680_680121


namespace variance_equivalence_l680_680880

def variance (s : List ℝ) : ℝ :=
  let μ := s.sum / s.length
  (s.map (λ x => (x - μ) ^ 2)).sum / s.length

theorem variance_equivalence (x : ℝ) :
  variance [1, 2, 3, 4, x] = variance [2020, 2021, 2022, 2023, 2024] ↔ x = 0 ∨ x = 5 := by
  sorry

end variance_equivalence_l680_680880


namespace sqrt_four_plus_two_inv_eq_five_halves_l680_680373

noncomputable def sqrt_four_plus_two_inv : ℚ :=
  (Real.sqrt 4) + (2⁻¹ : ℚ)

theorem sqrt_four_plus_two_inv_eq_five_halves :
  sqrt_four_plus_two_inv = 5 / 2 := 
by 
  sorry

end sqrt_four_plus_two_inv_eq_five_halves_l680_680373


namespace lawsuit_win_probability_l680_680010

theorem lawsuit_win_probability (P_L1 P_L2 P_W1 P_W2 : ℝ) (h1 : P_L2 = 0.5) 
  (h2 : P_L1 * P_L2 = P_W1 * P_W2 + 0.20 * P_W1 * P_W2)
  (h3 : P_W1 + P_L1 = 1)
  (h4 : P_W2 + P_L2 = 1) : 
  P_W1 = 1 / 2.20 :=
by
  sorry

end lawsuit_win_probability_l680_680010


namespace log_simplification_l680_680671

theorem log_simplification (x : ℝ) (h1 : 7 * x - 3 > 0) (h2 : 5 * x^3 ≠ 1) :
    (log ((7 * x - 3) ^ (1/2)) / log (5 * x^3)) / log (7 * x - 3) = 1 / 2 :=
by sorry

end log_simplification_l680_680671


namespace barbara_total_candies_l680_680721

theorem barbara_total_candies :
  let boxes1 := 9
  let candies_per_box1 := 25
  let boxes2 := 18
  let candies_per_box2 := 35
  boxes1 * candies_per_box1 + boxes2 * candies_per_box2 = 855 := 
by
  let boxes1 := 9
  let candies_per_box1 := 25
  let boxes2 := 18
  let candies_per_box2 := 35
  show boxes1 * candies_per_box1 + boxes2 * candies_per_box2 = 855
  sorry

end barbara_total_candies_l680_680721


namespace gcd_g_50_52_l680_680935

/-- Define the polynomial function g -/
def g (x : ℤ) : ℤ := x^2 - 3 * x + 2023

/-- The theorem stating the gcd of g(50) and g(52) -/
theorem gcd_g_50_52 : Int.gcd (g 50) (g 52) = 1 := by
  sorry

end gcd_g_50_52_l680_680935


namespace intersection_of_A_and_B_l680_680451

open Set

noncomputable def A := {x : ℝ | -3 ≤ x ∧ x ≤ 1}
noncomputable def B := {x : ℝ | Real.log x / Real.log 2 ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by sorry

end intersection_of_A_and_B_l680_680451


namespace range_of_m_l680_680509

theorem range_of_m (f : ℝ → ℝ) (h : ∀ x, f x = x^3 + x^2 + m * x + 1) :
  (∀ x, 0 ≤ f' x) → m ≥ 1 / 3 :=
by
  sorry

end range_of_m_l680_680509


namespace proof_problem_l680_680797

noncomputable def f (a x : ℝ) : ℝ := (a / (a^2 - 1)) * (Real.exp (Real.log a * x) - Real.exp (-Real.log a * x))

theorem proof_problem (
  a : ℝ
) (h1 : a > 1) :
  (∀ x, f a x = (a / (a^2 - 1)) * (Real.exp (Real.log a * x) - Real.exp (-Real.log a * x))) ∧
  (∀ x, f a (-x) = -f a x) ∧
  (∀ x1 x2, x1 < x2 → f a x1 < f a x2) ∧
  (∀ m, -1 < 1 - m ∧ 1 - m < m^2 - 1 ∧ m^2 - 1 < 1 → 1 < m ∧ m < Real.sqrt 2)
  :=
sorry

end proof_problem_l680_680797


namespace distance_between_A_and_B_is_43_km_l680_680883

theorem distance_between_A_and_B_is_43_km
    (travels_half_speed : ∀ t ∈ Icc (60 * 7) (60 * 8), v t = (1 / 2) * v_normal)
    (A_start_time : 6 * 60 + 50)
    (B_start_time : 6 * 60 + 50)
    (meet_24_km_from_A : distance A meet_point = 24)
    (A_depart_20_min_later_meet_midpoint : distance A midpoint = distance B midpoint)
    (B_depart_20_min_earlier_meet_20_km_from_A : distance A meet_point_20_km = 20) :
    distance A B = 43 := by
  sorry

end distance_between_A_and_B_is_43_km_l680_680883


namespace binary_representation_of_14_binary_representation_of_14_l680_680411

-- Define the problem as a proof goal
theorem binary_representation_of_14 : (14 : ℕ) = 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 0 * 2^0 :=
by sorry

-- An alternative formula to exactly represent the binary string using a conversion function can be provided:
theorem binary_representation_of_14' : nat.to_digits 2 14 = [1, 1, 1, 0] :=
by sorry

end binary_representation_of_14_binary_representation_of_14_l680_680411


namespace players_taking_all_three_subjects_l680_680013

-- Define the variables for the number of players in each category
def num_players : ℕ := 18
def num_physics : ℕ := 10
def num_biology : ℕ := 7
def num_chemistry : ℕ := 5
def num_physics_biology : ℕ := 3
def num_biology_chemistry : ℕ := 2
def num_physics_chemistry : ℕ := 1

-- Define the proposition we want to prove
theorem players_taking_all_three_subjects :
  ∃ x : ℕ, x = 2 ∧
  num_players = num_physics + num_biology + num_chemistry
                - num_physics_chemistry
                - num_physics_biology
                - num_biology_chemistry
                + x :=
by {
  sorry -- Placeholder for the proof
}

end players_taking_all_three_subjects_l680_680013


namespace num_integer_solutions_prime_l680_680432

def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m > 0 ∧ m < n → n % m ≠ 0

def integer_solutions : List ℤ := [-1, 3]

theorem num_integer_solutions_prime :
  (∀ x ∈ integer_solutions, is_prime (|15 * x^2 - 32 * x - 28|)) ∧ (integer_solutions.length = 2) :=
by
  sorry

end num_integer_solutions_prime_l680_680432


namespace C1_is_circle_when_k1_common_points_C1_C2_when_k4_l680_680527

-- Definitions for the two parts of the problem
def C1_parametric (k : ℕ) (t : ℝ) : ℝ × ℝ := (Real.cos t ^ k, Real.sin t ^ k)
def C2_cartesian (x y : ℝ) : Prop := 4 * x - 16 * y + 3 = 0

-- Translate Part (1): When k = 1
theorem C1_is_circle_when_k1 : ∀ (t : ℝ), C1_parametric 1 t = (Real.cos t, Real.sin t) :=
by
  intros t
  unfold C1_parametric
  simp
  sorry

-- Translate Part (2): When k = 4
theorem common_points_C1_C2_when_k4 : ∀ (t : ℝ), 
  let coords := C1_parametric 4 t
  coords = (1/4, 1/4) →
  C2_cartesian (coords.1) (coords.2) :=
by
  intros t coords h_coords
  unfold C1_parametric at h_coords
  have h_coords_x : Real.cos t ^ 4 = 1 / 4,
  {
    sorry -- proof that cos^4 t = 1 / 4
  }
  have h_coords_y : Real.sin t ^ 4 = 1 / 4,
  {
    sorry -- proof that sin^4 t = 1 / 4
  }
  rw [←h_coords_x, ←h_coords_y]
  change 4 * (1 / 4) - 16 * (1 / 4) + 3 = 0
  norm_num
  sorry

end C1_is_circle_when_k1_common_points_C1_C2_when_k4_l680_680527


namespace find_m_for_increasing_graph_l680_680511

theorem find_m_for_increasing_graph (m : ℝ) :
  (∀ x y : ℝ, x > 0 → y > 0 → (m + 1) * x ^ (3 - m^2) < (m + 1) * y ^ (3 - m^2) → x < y) ↔ m = -2 :=
by
  sorry

end find_m_for_increasing_graph_l680_680511


namespace label_cubes_zero_sum_l680_680092

theorem label_cubes_zero_sum (n : ℕ) (h : n ≥ 3) : 
  ∃ (label : Finₓ n × Finₓ n × Finₓ n → ℤ), 
    (∀ (x : Finₓ n), ∑ k in finset.univ.image (λ (i : Finₓ n), label (x, i, i)) = 0) ∧
    (∀ (y : Finₓ n), ∑ k in finset.univ.image (λ (i : Finₓ n), label (i, y, i)) = 0) ∧
    (∀ (z : Finₓ n), ∑ k in finset.univ.image (λ (i : Finₓ n), label (i, i, z)) = 0) ∧
    Function.Injective (label : Finₓ n × Finₓ n × Finₓ n → ℤ) :=
by
  sorry

end label_cubes_zero_sum_l680_680092


namespace average_height_24_2_l680_680340

def tree_height (tree2 tree1 tree3 tree4 tree5 : ℕ) : Prop := 
  ((tree1 = 2 * tree2 ∨ tree1 = tree2 / 2) ∧
  (tree3 = 2 * tree2 ∨ tree3 = tree2 / 2) ∧
  (tree4 = 2 * tree3 ∨ tree4 = tree3 / 2) ∧
  (tree5 = 2 * tree4 ∨ tree5 = tree4 / 2))

def valid_heights (tree1 tree2 tree3 tree4 tree5 : ℕ) : Prop := 
  (tree2 = 11 ∧
  ∃ k : ℕ, tree1 + tree2 + tree3 + tree4 + tree5 = 5 * k + 1 ∧ 
  (tree1 + tree2 + tree3 + tree4 + tree5) = 24 * 5 + 1)

theorem average_height_24_2 : ∃ tree1 tree3 tree4 tree5 : ℕ, tree_height 11 tree1 tree3 tree4 tree5 ∧ valid_heights tree1 11 tree3 tree4 tree5 :=
begin
  sorry,
end

end average_height_24_2_l680_680340


namespace sticks_left_in_yard_l680_680038

def number_of_sticks_picked_up : Nat := 14
def difference_between_picked_and_left : Nat := 10

theorem sticks_left_in_yard 
  (picked_up : Nat := number_of_sticks_picked_up)
  (difference : Nat := difference_between_picked_and_left) 
  : Nat :=
  picked_up - difference

example : sticks_left_in_yard = 4 := by 
  sorry

end sticks_left_in_yard_l680_680038


namespace johns_running_hours_l680_680051

-- Define the conditions
variable (x : ℕ) -- let x represent the number of hours at 8 mph and 6 mph
variable (total_hours : ℕ) (total_distance : ℕ)
variable (speed_8 : ℕ) (speed_6 : ℕ) (speed_5 : ℕ)
variable (distance_8 : ℕ := speed_8 * x)
variable (distance_6 : ℕ := speed_6 * x)
variable (distance_5 : ℕ := speed_5 * (total_hours - 2 * x))

-- Total hours John completes the marathon
axiom h1: total_hours = 15

-- Total distance John completes in miles
axiom h2: total_distance = 95

-- Speed factors
axiom h3: speed_8 = 8
axiom h4: speed_6 = 6
axiom h5: speed_5 = 5

-- Distance equation
axiom h6: distance_8 + distance_6 + distance_5 = total_distance

-- Prove the number of hours John ran at each speed
theorem johns_running_hours : x = 5 :=
by
  sorry

end johns_running_hours_l680_680051


namespace part_I_part_II_l680_680815

-- Part (I)
section
variable {A B : Set ℝ}
local notation "ℝ" => Real

def A : Set ℝ := { x | x ≥ 2 }
def B : Set ℝ := { x | -1 ≤ x ∧ x ≤ 5 }

theorem part_I : ((Aᶜ) ∩ B) = { x | -1 ≤ x ∧ x < 2 } := by
  sorry
end

-- Part (II)
section
variable {a : ℝ}
local notation "ℝ" => Real

def D (a : ℝ) : Set ℝ := { x | 1 - a ≤ x ∧ x ≤ 1 + a }
def B : Set ℝ := { x | -1 ≤ x ∧ x ≤ 5 }
def compB : Set ℝ := { x | x < -1 ∨ x > 5 }

theorem part_II : (∀ a, (D a ∪ compB) = compB → a ∈ Iio 0) := by
  sorry
end

end part_I_part_II_l680_680815


namespace box_dimensions_sum_l680_680707

theorem box_dimensions_sum (A B C : ℝ)
  (h1 : A * B = 18)
  (h2 : A * C = 32)
  (h3 : B * C = 50) :
  A + B + C = 57.28 := 
sorry

end box_dimensions_sum_l680_680707


namespace graph_shifted_correct_l680_680835

def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then -2 - x
  else if 0 < x ∧ x ≤ 2 then real.sqrt (4 - (x - 2) ^ 2) - 2
  else if 2 < x ∧ x ≤ 3 then 2 * (x - 2)
  else 0

def g (x : ℝ) : ℝ :=
  f(x) + 3

theorem graph_shifted_correct :
  (∀ x, -3 ≤ x ∧ x ≤ 0 → g x = 1 - x) ∧
  (∀ x, 0 < x ∧ x ≤ 2 → g x = real.sqrt (4 - (x - 2) ^ 2) + 1) ∧
  (∀ x, 2 < x ∧ x ≤ 3 → g x = 2 * x - 1) :=
by
  sorry

end graph_shifted_correct_l680_680835


namespace fraction_of_grassy_area_covered_by_flower_beds_l680_680330

theorem fraction_of_grassy_area_covered_by_flower_beds :
  let
    length := 40,
    width := 20,
    base_combined := 10,
    s := base_combined / 3,
    height := (real.sqrt 3 / 2) * s,
    area_one_triangle := 1 / 2 * s * height,
    area_flower_beds := 3 * area_one_triangle,
    area_grassy_area := length * width
  in
    area_flower_beds / area_grassy_area = (25 * real.sqrt 3) / 480 := by
{
  sorry
}

end fraction_of_grassy_area_covered_by_flower_beds_l680_680330


namespace compute_P_part_l680_680195

noncomputable def P (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem compute_P_part (a b c d : ℝ) 
  (H1 : P 1 a b c d = 1993) 
  (H2 : P 2 a b c d = 3986) 
  (H3 : P 3 a b c d = 5979) : 
  (1 / 4) * (P 11 a b c d + P (-7) a b c d) = 4693 :=
by
  sorry

end compute_P_part_l680_680195


namespace num_zeros_sin_minus_ln_l680_680261

theorem num_zeros_sin_minus_ln : ∃! x ∈ set.Ioi 0, sin x = Real.log x :=
sorry

end num_zeros_sin_minus_ln_l680_680261


namespace integer_solution_exists_for_a_equals_7_l680_680387

theorem integer_solution_exists_for_a_equals_7 :
  (∃ x : ℤ, (∀ x a : ℤ, a > 0 ∧ a = 7 → 
    ((1 + (1 / x)) * (1 + (1 / (x + 1))) * ... * (1 + (1 / (x + a))) = a - x))) := sorry

end integer_solution_exists_for_a_equals_7_l680_680387


namespace magnitude_of_b_l680_680122

noncomputable theory
open_locale real_inner_product_space

variables (a b : ℝ^3)

def angle_between (a b : ℝ^3) : ℝ := real.acos (inner a b / (∥a∥ * ∥b∥))

def a_is_unit_norm : Prop := ∥a∥ = 1
def dist_cond : Prop := ∥2 • a - b∥ = 1
def angle_cond : Prop := angle_between a b = real.pi / 6

theorem magnitude_of_b 
(h1 : a_is_unit_norm a) 
(h2 : dist_cond a b) 
(h3 : angle_cond a b) : ∥b∥ = sqrt 3 := 
sorry

end magnitude_of_b_l680_680122


namespace positive_whole_numbers_cube_root_less_than_eight_l680_680853

theorem positive_whole_numbers_cube_root_less_than_eight : 
  { n : ℕ | n > 0 ∧ n < 512 }.card = 511 :=
by sorry

end positive_whole_numbers_cube_root_less_than_eight_l680_680853


namespace num_non_similar_regular_2019_pointed_stars_l680_680142

theorem num_non_similar_regular_2019_pointed_stars : 
  let n := 2019 in 
  let star_conditions (P : ℕ → ℕ → Prop) := 
    (∀ i j, P i j → (i ≤ n ∧ j ≤ n)) ∧ 
    (∀ i j k, P i j ∧ P j k ∧ i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬ collinear i j k) ∧
    (∀ i j, P i j → ∃ k, k ≠ i ∧ k ≠ j ∧ P i k ∧ P k j) ∧ 
    (∀ i j k, P i j ∧ P k (k + 1 % n) → angle i j k = 120) ∧ 
    (∀ i j, P i j → length i j = length j (j + 1 % n)) ∧ 
    (∃ i j k, P i j ∧ P j k ∧ counterclockwise_turn i j k < 180) 
  in
  ∃ m, gcd(m, 2019) = 1 → 
       Euler_totient 2019 = 1344 / 2 := 
by
  let n := 2019
  have h1 : n = 3 * 673 := by norm_num
  have h2 : is_prime 3 := by norm_num
  have h3 : is_prime 673 := by norm_num
  have h4 : Euler_totient 3 = 2 := Euler_totient_prime 3 h2
  have h5 : Euler_totient 673 = 672 := Euler_totient_prime 673 h3
  have h6 : Euler_totient (3 * 673) = (Euler_totient 3) * (Euler_totient 673) := Euler_totient_mul_coprime h2 h3
  rw [h4, h5] at h6
  have h7 : Euler_totient 2019 = 2 * 672 := by norm_num
  have h8 : 2 * 672 = 1344 := by norm_num
  have h9 : 1344 / 2 = 672 := by norm_num
  exact h9 

sorry

end num_non_similar_regular_2019_pointed_stars_l680_680142


namespace arithmetic_sequence_general_term_l680_680104

theorem arithmetic_sequence_general_term (a : ℕ → ℤ) (a1 : a 1 = -1) (d : ℤ) (h : d = 4) :
  a = λ n, 4 * n - 5 :=
by
  sorry

end arithmetic_sequence_general_term_l680_680104


namespace time_to_cross_signal_pole_l680_680313

-- Definitions for the train's length and platform's length
def train_length : ℝ := 300
def platform_length : ℝ := 250
def time_to_cross_platform : ℝ := 33

-- Assume the speed of the train when it crosses the platform
def train_speed : ℝ := (train_length + platform_length) / time_to_cross_platform

-- Proof that the time it takes to cross the signal pole is approximately 18 seconds
theorem time_to_cross_signal_pole : (train_length / train_speed) ≈ 18 := by
  sorry

end time_to_cross_signal_pole_l680_680313


namespace ginger_water_bottle_cups_l680_680792

theorem ginger_water_bottle_cups :
  let h := 8 in -- hours worked
  let b_d := h in -- bottles drunk
  let b_p := 5 in -- additional bottles poured
  let b_t := b_d + b_p in -- total bottles used
  let c_t := 26 in -- total cups of water used
  
  b_t * 2 = c_t := -- cups per bottle
begin
  sorry
end

end ginger_water_bottle_cups_l680_680792


namespace scientific_notation_l680_680692

def significant_digits : ℝ := 4.032
def exponent : ℤ := 11
def original_number : ℝ := 403200000000

theorem scientific_notation : original_number = significant_digits * 10 ^ exponent := 
by
  sorry

end scientific_notation_l680_680692


namespace line_plane_intersection_l680_680768

theorem line_plane_intersection 
  (t : ℝ)
  (x_eq : ∀ t: ℝ, x = 5 - t)
  (y_eq : ∀ t: ℝ, y = -3 + 5 * t)
  (z_eq : ∀ t: ℝ, z = 1 + 2 * t)
  (plane_eq : 3 * x + 7 * y - 5 * z - 11 = 0)
  : x = 4 ∧ y = 2 ∧ z = 3 :=
by
  sorry

end line_plane_intersection_l680_680768


namespace action_figures_added_l680_680182

theorem action_figures_added :
  ∃ x : ℕ, (4 + x = 3 + 3) ∧ x = 2 :=
by
  exist 2
  split
  ・calc 4 + 2 = 6 : by norm_num
          3 + 3 = 6 : by norm_num
    done
  rfl

end action_figures_added_l680_680182


namespace proof_a5_eq_10_l680_680271

-- Define the arithmetic sequence conditions
variables {a : ℕ → ℝ} {d : ℝ}
axiom a2_plus_a3 : a 2 + a 3 = 5
axiom S5 : (5/2) * (a 1 + a 5) = 20

-- The proof problem statement
theorem proof_a5_eq_10 (h1 : a 2 = a 1 + d) (h2 : a 3 = a 1 + 2 * d) :
  a 5 = 10 :=
begin
  sorry
end

end proof_a5_eq_10_l680_680271


namespace find_line_equation_l680_680821

def is_parallel_to (l₁ l₂ : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (l₂.1, l₂.2) = (k * l₁.1, k * l₁.2)

structure Point where
  x : ℝ
  y : ℝ

def satisfies (line : ℝ × ℝ × ℝ) (P : Point) : Prop :=
  line.1 * P.x + line.2 * P.y + line.3 = 0

theorem find_line_equation (P : Point)
  (H1 : is_parallel_to (2, -1, 1) (2, -1, P.y - 2 * P.x)) 
  (H2 : satisfies (2, -1, 0) P):
  (2 * P.x - P.y = 0) :=
by
  -- The proof is omitted
  sorry

end find_line_equation_l680_680821


namespace alice_cookie_fills_l680_680001

theorem alice_cookie_fills :
  (∀ (a b : ℚ), a = 3 + (3/4) ∧ b = 1/3 → (a / b) = 12) :=
sorry

end alice_cookie_fills_l680_680001


namespace jake_car_washes_l680_680542

theorem jake_car_washes :
  ∀ (washes_per_bottle cost_per_bottle total_spent weekly_washes : ℕ),
  washes_per_bottle = 4 →
  cost_per_bottle = 4 →
  total_spent = 20 →
  weekly_washes = 1 →
  (total_spent / cost_per_bottle) * washes_per_bottle / weekly_washes = 20 :=
by
  intros washes_per_bottle cost_per_bottle total_spent weekly_washes
  sorry

end jake_car_washes_l680_680542


namespace proof_problem_l680_680501

variable (a b c d x : ℤ)

-- Conditions
axiom condition1 : a - b = c + d + x
axiom condition2 : a + b = c - d - 3
axiom condition3 : a - c = 3
axiom answer_eq : x = 9

-- Proof statement
theorem proof_problem : (a - b) = (c + d + 9) :=
by
  sorry

end proof_problem_l680_680501


namespace sum_first_100_terms_l680_680897

-- Define the sequence
def seq : ℕ → ℤ
| 0     := 1
| 1     := 3
| (n+2) := seq (n+1) - seq n

-- Define the function to sum the first n terms of the sequence
def sum_seq : ℕ → ℤ
| 0     := seq 0
| (n+1) := sum_seq n + seq (n+1)

-- Statement to prove
theorem sum_first_100_terms : sum_seq 99 = 5 := 
by
  sorry  -- Proof goes here

end sum_first_100_terms_l680_680897


namespace actual_average_height_is_179_76_l680_680884

-- Define the initial conditions
def number_of_boys := 50
def initial_average_height := 180
def initial_total_height := initial_average_height * number_of_boys

def incorrect_heights := [200, 155, 190, 172, 140]
def correct_heights := [170, 165, 178, 182, 150]

-- Calculate total incorrect heights and correct heights
def total_incorrect_height := incorrect_heights.sum
def total_correct_height := correct_heights.sum
def height_difference := total_incorrect_height - total_correct_height

-- Adjust the initial total height by the difference
def adjusted_total_height := initial_total_height - height_difference

-- Calculate the adjusted average height
def adjusted_average_height := adjusted_total_height / number_of_boys

-- Prove the actual average height
theorem actual_average_height_is_179_76 : adjusted_average_height = 179.76 :=
by {
  sorry -- The proof will be filled in later
}

end actual_average_height_is_179_76_l680_680884


namespace special_lines_count_l680_680325

noncomputable def count_special_lines : ℕ :=
  sorry

theorem special_lines_count :
  count_special_lines = 3 :=
by sorry

end special_lines_count_l680_680325


namespace product_of_possible_x_l680_680150

theorem product_of_possible_x : 
  (∀ x : ℚ, abs ((18 / x) + 4) = 3 → x = -18 ∨ x = -18 / 7) → 
  ((-18) * (-18 / 7) = 324 / 7) :=
by
  sorry

end product_of_possible_x_l680_680150


namespace number_of_zeros_in_Q_l680_680039

def R (k : ℕ) : ℕ := (10^k - 1) / 9

def Q : ℕ := R 30 / R 6

theorem number_of_zeros_in_Q : (Q.toString.filter (λ c, c = '0')).length = 25 :=
sorry

end number_of_zeros_in_Q_l680_680039


namespace volume_of_sphere_from_cube_surface_area_l680_680515

theorem volume_of_sphere_from_cube_surface_area (S : ℝ) (h : S = 24) : 
  ∃ V : ℝ, V = 4 * Real.sqrt 3 * Real.pi := 
sorry

end volume_of_sphere_from_cube_surface_area_l680_680515


namespace range_of_k_l680_680879

-- Definitions to use in statement
variable (k : ℝ)

-- Statement: Proving the range of k
theorem range_of_k (h : ∀ x : ℝ, k * x^2 - k * x - 1 < 0) : -4 < k ∧ k ≤ 0 :=
  sorry

end range_of_k_l680_680879


namespace line_through_point_with_equal_intercepts_l680_680255

theorem line_through_point_with_equal_intercepts
    (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h_eq : a = b) :
    (∃ (c : ℝ), ∀ x y : ℝ, (x + y = c) ∧ (1, 2) ∈ {⟨x, y⟩ : ℝ × ℝ | x + y = c}) 
    ∨ (∃ k : ℝ, ∀ x : ℝ, y : ℝ, y = k * x) :=
sorry

end line_through_point_with_equal_intercepts_l680_680255


namespace compute_P_part_l680_680196

noncomputable def P (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem compute_P_part (a b c d : ℝ) 
  (H1 : P 1 a b c d = 1993) 
  (H2 : P 2 a b c d = 3986) 
  (H3 : P 3 a b c d = 5979) : 
  (1 / 4) * (P 11 a b c d + P (-7) a b c d) = 4693 :=
by
  sorry

end compute_P_part_l680_680196


namespace find_integer_x_l680_680580

theorem find_integer_x (x : ℤ) :
  (2 * x > 70 → x > 35 ∧ 4 * x > 25 → x > 6) ∧
  (x < 100 → x = 6) ∧ -- Given valid inequality relations and restrictions
  ((2 * x <= 70 ∨ x >= 100) ∨ (4 * x <= 25 ∨ x <= 5)) := -- Certain contradictions inherent

  by sorry

end find_integer_x_l680_680580


namespace problem_statement_l680_680437

open Real

noncomputable def to_prove (x : ℝ) : Prop :=
  tan x = -2 ∧ (π / 2 < x ∧ x < π) → cos x = - (sqrt 5) / 5

theorem problem_statement (x : ℝ) : to_prove x := by
  sorry

end problem_statement_l680_680437


namespace quadratic_function_properties_l680_680804

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

theorem quadratic_function_properties
  (a b c : ℝ)
  (h_a : a ≠ 0)
  (h_symmetry : ∀ x : ℝ, f a b c x = f a b c (-x - 2))  -- representing symmetry about x = -1
  (h_value_at_1 : f a b c 1 = 1)
  (h_min_value : ∃ x : ℝ, f a b c x = 0) :
  f a b c = (λ x : ℝ, (1 / 4) * x ^ 2 + (1 / 2) * x + (1 / 4)) ∧    -- Answer to (1)
  (∃ m : ℝ, m = 9 ∧ m > 1 ∧ ∃ t : ℝ, ∀ x ∈ set.Icc (1 : ℝ) m, f a b c (x + t) ≤ x) := -- Answer to (2)
sorry

end quadratic_function_properties_l680_680804


namespace sum_series_eq_l680_680728

theorem sum_series_eq :
  (∑ n from 3 to ∞, ∑ k from 1 to n - 2, k^2 / 3^(n+k)) = 405 / 20736 :=
by
  sorry

end sum_series_eq_l680_680728


namespace find_a_l680_680480

noncomputable def M (a : ℤ) : Set ℤ := {a, 0}
noncomputable def N : Set ℤ := { x : ℤ | 2 * x^2 - 3 * x < 0 }

theorem find_a (a : ℤ) (h : (M a ∩ N).Nonempty) : a = 1 := sorry

end find_a_l680_680480


namespace maximum_value_of_a4_l680_680155

noncomputable def max_a4_of_geometric_seq (a : ℕ → ℝ) (h1 : ∀ n, 0 < a n) (h2 : (a 2 + a 4 = 4)) : ℝ :=
  let h3 := (a 2 * a 4 = a 3 ^ 2) in
  let h4 := (a 2 = 2 ∧ a 4 = 2 ∧ a 3 = 2) in
  2

theorem maximum_value_of_a4 {a : ℕ → ℝ} (h1 : ∀ n, 0 < a n) (h2 : a 2 + a 4 = 4) : (∃ b, b = max_a4_of_geometric_seq a h1 h2) :=
by
  use 2
  simpl
  sorry

end maximum_value_of_a4_l680_680155


namespace number_of_ordered_pairs_l680_680026
open Complex

def is_real (z : ℂ) : Prop := z.im = 0

def power_i_is_real (n : ℕ) : Prop :=
  is_real (Complex.i ^ n)

theorem number_of_ordered_pairs :
  (∃ n, ∃ pairs, pairs = (counting_pairs_with_conditions) n ∧ n = 1036) :=
by 
  sorry

def counting_pairs_with_conditions : ℕ :=
  -- Add detailed definition here and count pairs based on conditions 1 and 2
  sorry

end number_of_ordered_pairs_l680_680026


namespace proof_vector_eq_l680_680495

noncomputable def vector_eq : Prop :=
  ∀ (a b : ℝ × ℝ × ℝ),
  a × b = (7, -3, 2) →
  2 * a × (5 * b) = (140, -60, 40)

theorem proof_vector_eq : vector_eq :=
begin
  sorry
end

end proof_vector_eq_l680_680495


namespace rita_remaining_money_l680_680968

-- Defining the conditions
def num_dresses := 5
def price_dress := 20
def num_pants := 3
def price_pant := 12
def num_jackets := 4
def price_jacket := 30
def transport_cost := 5
def initial_amount := 400

-- Calculating the total cost
def total_cost : ℕ :=
  (num_dresses * price_dress) + 
  (num_pants * price_pant) + 
  (num_jackets * price_jacket) + 
  transport_cost

-- Stating the proof problem 
theorem rita_remaining_money : initial_amount - total_cost = 139 := by
  sorry

end rita_remaining_money_l680_680968


namespace find_original_prices_l680_680263

theorem find_original_prices
  (C_inc : 30) (C_new : 377)
  (T_inc : 20) (T_new : 720)
  (R_inc : 15) (R_new : 1150) :
  ∃ (C T R : ℝ), 
    C_new = C * (1 + C_inc / 100) ∧
    T_new = T * (1 + T_inc / 100) ∧
    R_new = R * (1 + R_inc / 100) ∧
    C = 290 ∧
    T = 600 ∧
    R = 1000 :=
begin
  sorry
end

end find_original_prices_l680_680263


namespace polynomial_value_l680_680198

def P (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem polynomial_value (a b c d : ℝ) 
  (h1 : P 1 a b c d = 1993) 
  (h2 : P 2 a b c d = 3986) 
  (h3 : P 3 a b c d = 5979) :
  (1 / 4) * (P 11 a b c d + P (-7) a b c d) = 4693 := 
by 
  sorry

end polynomial_value_l680_680198


namespace smallest_n_with_four_pairs_l680_680751

noncomputable def f (n : ℕ) : ℕ :=
  ∑ a in Finset.range (n + 1), ∑ b in Finset.range (n + 1), if (a^2 + b^2 = n) then 1 else 0

theorem smallest_n_with_four_pairs : ∃ (n : ℕ), f n = 4 ∧ ∀ m < n, f m ≠ 4 := 
begin
  use 25,
  split,
  { sorry }, -- Proof that f(25) = 4
  { -- Proof that for all m < 25, f(m) ≠ 4
    assume m hm,
    sorry
  }
end

end smallest_n_with_four_pairs_l680_680751


namespace no_labeling_for_odd_n_six_labelings_for_even_n_l680_680565

-- Define the basic setup
def S_n (n : ℕ) (labeling : Fin n → Fin n → Int) : Int :=
  let row_sum (i : Fin n) := (Fin.sum (Fin n) (λ j, labeling i j))
  let col_sum (j : Fin n) := (Fin.sum (Fin n) (λ i, labeling i j))
  Fin.sum (Fin n) row_sum + Fin.sum (Fin n) col_sum

-- Part (a)
theorem no_labeling_for_odd_n (n : ℕ) (h_n_ge_2 : 2 ≤ n) (h_n_odd : n % 2 = 1) (labeling : Fin n → Fin n → Int) : S_n n labeling ≠ 0 :=
sorry

-- Part (b)
theorem six_labelings_for_even_n (n : ℕ) (h_n_ge_2 : 2 ≤ n) (h_n_even : n % 2 = 0) : 
  ∃ (labelings : Fin 6 → (Fin n → Fin n → Int)), 
  ∀ i : Fin 6, S_n n (labelings i) = 0 :=
sorry

end no_labeling_for_odd_n_six_labelings_for_even_n_l680_680565


namespace integral_matches_value_l680_680759

noncomputable def find_k (k : ℝ) : Prop :=
  ∫ x in 0..2, (3 * x^2 + k) = 10 → k = 1

-- Here, we define the main theorem which uses the above definition
theorem integral_matches_value : ∀ k : ℝ, find_k k :=
by
  intros k
  sorry

end integral_matches_value_l680_680759


namespace polynomial_value_l680_680197

def P (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem polynomial_value (a b c d : ℝ) 
  (h1 : P 1 a b c d = 1993) 
  (h2 : P 2 a b c d = 3986) 
  (h3 : P 3 a b c d = 5979) :
  (1 / 4) * (P 11 a b c d + P (-7) a b c d) = 4693 := 
by 
  sorry

end polynomial_value_l680_680197


namespace cyclic_sum_nonneg_l680_680936

-- Definitions of the conditions
variables {r x y z : ℝ}
variables (h_r : r ∈ ℝ)
variables (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0)

-- Theorem declaration
theorem cyclic_sum_nonneg (h_r : r ∈ ℝ) (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) :
  ∑ (i : ℕ) in finset.range 3, (x^r) * ((vector.nth i 0 - vector.nth i 1) * (vector.nth i 0 - vector.nth i 2)) ≥ 0 :=
sorry

end cyclic_sum_nonneg_l680_680936


namespace expected_distance_between_andy_and_midpoint_l680_680539

def midpoint (a b : ℝ) : ℝ := (a + b) / 2
noncomputable def expected_distance_after_2010_flips
  (start : ℝ)
  (flips : ℕ)
  (n : ℕ) : ℝ :=
if n = 0
then start
else
  let d := expected_distance_after_2010_flips start flips (n - 1) in
  (1/2) * ((1 - d) + d) / 2

theorem expected_distance_between_andy_and_midpoint :
  expected_distance_after_2010_flips  (midpoint 0 1) 2010 2010 = 1/4 := sorry

end expected_distance_between_andy_and_midpoint_l680_680539


namespace polynomials_symmetric_l680_680341

noncomputable def P : ℕ → (ℝ → ℝ → ℝ → ℝ)
  | 0       => λ x y z => 1
  | (m + 1) => λ x y z => (x + z) * (y + z) * (P m x y (z + 1)) - z^2 * (P m x y z)

theorem polynomials_symmetric (m : ℕ) (x y z : ℝ) : 
  P m x y z = P m y x z ∧ P m x y z = P m x z y := 
sorry

end polynomials_symmetric_l680_680341


namespace num_ordered_pairs_xy_eq_12_l680_680388

/-- Define 2541 with its known prime factorization -/
def n : ℕ := 2541
def p : ℕ := 3
def q : ℕ := 13
def r : ℕ := 5
def e₁ : ℕ := 1
def e₂ : ℕ := 2
def e₃ : ℕ := 1

/-- Define the main theorem -/
theorem num_ordered_pairs_xy_eq_12 :
  (∀ x y : ℕ, x * y = n → x > 0 ∧ y > 0) →
  (number_of_divisors (p^e₁ * q^e₂ * r^e₃) = 12) :=
sorry

end num_ordered_pairs_xy_eq_12_l680_680388


namespace sum_of_abs_first_10_terms_l680_680827

noncomputable def sum_of_first_n_terms (n : ℕ) : ℤ := n^2 - 5 * n + 2

theorem sum_of_abs_first_10_terms : 
  let S := sum_of_first_n_terms 10
  let S3 := sum_of_first_n_terms 3
  (S - 2 * S3) = 60 := 
by
  sorry

end sum_of_abs_first_10_terms_l680_680827


namespace trigonometric_values_l680_680455

theorem trigonometric_values (x y : ℝ) 
  (h1 : Real.sin x + Real.sin y = 1 / 3) 
  (h2 : Real.cos x - Real.cos y = 1 / 5) : 
  Real.cos (x + y) = 208 / 225 ∧ Real.sin (x - y) = -15 / 17 := 
by 
  sorry

end trigonometric_values_l680_680455


namespace find_area_of_rhombus_l680_680996

variables {P : ℝ} {r1 r2 : ℝ} {d1 d2 : ℝ}

-- Condition: P = 2
def perimeter (a : ℝ) : Prop := 4 * a = P

-- Condition: The diagonals are in the ratio 3:4
def ratio_diagonals (k : ℝ) : Prop := d1 = 3 * k ∧ d2 = 4 * k

-- The diagonals bisect each other at right angles
def pythagoras_relation (a d1 d2: ℝ) : Prop := (d1 / 2)^2 + (d2 / 2)^2 = a^2

-- Area of the rhombus
def area (d1 d2 : ℝ) : ℝ := (1 / 2) * d1 * d2

-- Given the conditions, prove the area is 0.24 square meters.
theorem find_area_of_rhombus (a k : ℝ)
  (hP : perimeter a)
  (hd : ratio_diagonals k)
  (hp : pythagoras_relation a d1 d2)
  (hP_val : P = 2) :
  area d1 d2 = 0.24 :=
by
  sorry

end find_area_of_rhombus_l680_680996


namespace convert_4512_base8_to_base10_l680_680745

-- Definitions based on conditions
def base8_to_base10 (n : Nat) : Nat :=
  let d3 := 4 * 8^3
  let d2 := 5 * 8^2
  let d1 := 1 * 8^1
  let d0 := 2 * 8^0
  d3 + d2 + d1 + d0

-- The proof statement
theorem convert_4512_base8_to_base10 :
  base8_to_base10 4512 = 2378 :=
by
  -- proof goes here
  sorry

end convert_4512_base8_to_base10_l680_680745


namespace binary_representation_of_fourteen_l680_680413

theorem binary_representation_of_fourteen :
  (14 : ℕ) = 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 0 * 2^0 :=
by
  sorry

end binary_representation_of_fourteen_l680_680413


namespace moment_of_inertia_is_316_l680_680308

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def masses_and_coordinates : List (ℝ × ℝ) :=
  [(3, 4), (2, 0), (3, 4), (8, 0)]

def masses : List ℝ := [2, 6, 2, 3]

def distances_from_S : List ℝ := 
  List.map (λ p, distance 0 0 p.1 p.2) masses_and_coordinates

def moment_of_inertia : ℝ :=
  List.sum (List.zipWith (λ m d, m * d^2) masses distances_from_S)

theorem moment_of_inertia_is_316 : moment_of_inertia = 316 :=
by
  unfold moment_of_inertia
  unfold distances_from_S
  unfold masses_and_coordinates
  sorry

end moment_of_inertia_is_316_l680_680308


namespace geometric_sequence_problem_l680_680098

noncomputable def geometric_sequence_product (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a n = a 1 * (2 ^ (n - 1))

theorem geometric_sequence_problem 
  (a : ℕ → ℝ) 
  (h1 : geometric_sequence_product a)
  (h2 : a 1 ≠ 0)
  (h3 : ∏ i in finset.range 30, a (i + 1) = 2 ^ 30) :
  ∏ i in finset.range 10, a (3 * (i + 1)) = 2 ^ 20 := 
by 
  -- proof to be filled in
  sorry

end geometric_sequence_problem_l680_680098


namespace range_of_a_l680_680877

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > a → x * (x - 1) > 0) → a ≥ 1 :=
begin
  sorry
end

end range_of_a_l680_680877


namespace smallest_value_in_interval_l680_680148

open Real

noncomputable def smallest_value (x : ℝ) (h : 1 < x ∧ x < 2) : Prop :=
  1 / x^2 < x ∧
  1 / x^2 < x^2 ∧
  1 / x^2 < 2 * x^2 ∧
  1 / x^2 < 3 * x ∧
  1 / x^2 < sqrt x ∧
  1 / x^2 < 1 / x

theorem smallest_value_in_interval (x : ℝ) (h : 1 < x ∧ x < 2) : smallest_value x h :=
by
  sorry

end smallest_value_in_interval_l680_680148


namespace binomial_properties_l680_680175

-- Definitions of the binomial expansion and terms of interest
def binomial_expansion (x : ℝ) : ℝ := (Real.sqrt x - 1 / (2 * x)) ^ 6

-- The terms to be proved
theorem binomial_properties : 
  let T := binomial_expansion in
  -- 1. The constant term
  let const_term := (15 : ℝ) / 4 in
  -- 2. The sum of the coefficients of all terms
  let sum_of_coeffs := (1 : ℝ) / 64 in
  -- 3. The largest binomial coefficient is C(6, 3) = 20, at the 4th term
  let largest_coeff := 20 in
  -- 4. The sum of the coefficients of the odd terms
  let sum_of_odd_coeffs := 32 in
  -- Prove the individual statements
  (True) ∧ (True) ∧ (True) ∧ (True) :=
by {
  -- Proof is omitted
  sorry
}

end binomial_properties_l680_680175


namespace monotonic_intervals_extreme_values_in_interval_l680_680469

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 3 * x + 2 * Real.log x

theorem monotonic_intervals : (∀ x > 0, 0 < x ∧ x < 1 → f x > 0 ∨ 2 < x → f x > 0) ∧ 
  (∀ x > 0, 1 < x ∧ x < 2 → f x < 0) := 
  sorry

theorem extreme_values_in_interval : 
  f 1 = -(5 / 2) ∧ 
  f 2 = 2 * Real.log 2 - 4 ∧ 
  f 3 = 2 * Real.log 3 - 9 / 2 ∧ 
  ∀ x ∈ Set.Icc 1 3, ∃ (minval : ℝ) (maxval : ℝ), minval = f 2 ∧ maxval = f 3 :=
  sorry

end monotonic_intervals_extreme_values_in_interval_l680_680469


namespace rational_sum_of_cubic_roots_inverse_l680_680145

theorem rational_sum_of_cubic_roots_inverse 
  (p q r : ℚ) 
  (h1 : p ≠ 0) 
  (h2 : q ≠ 0) 
  (h3 : r ≠ 0) 
  (h4 : ∃ a b c : ℚ, a = (pq^2)^(1/3) ∧ b = (qr^2)^(1/3) ∧ c = (rp^2)^(1/3) ∧ a + b + c ≠ 0) 
  : ∃ s : ℚ, s = 1/((pq^2)^(1/3)) + 1/((qr^2)^(1/3)) + 1/((rp^2)^(1/3)) :=
sorry

end rational_sum_of_cubic_roots_inverse_l680_680145


namespace min_positive_sum_value_l680_680809

variable {a : ℕ → ℝ}  -- Define an arithmetic sequence {a_n}
variable {S : ℕ → ℝ}  -- Define the sum of the first n terms {S_n}

-- Condition: \frac{a_{13}}{a_{12}} < -1
def condition1 : Prop := a 13 / a 12 < -1

-- Condition: The sum of the first n terms S_n has a maximum value at some point
def has_max_sum (S : ℕ → ℝ) : Prop := ∃ N, ∀ n ≥ N, S n ≤ S N

theorem min_positive_sum_value (h1 : condition1) (h2 : has_max_sum S) :
  ∃ n, S n > 0 ∧ ∀ m, m ≠ 23 → S m ≤ 0 ∧ n = 23 :=
sorry

end min_positive_sum_value_l680_680809


namespace tony_min_moves_l680_680277

theorem tony_min_moves (n k : ℕ) : 
  (∀ stones : ℕ → ℕ, stones 0 = k ∧ (∀ m, 0 < m ≤ n → (∃ i ≤ m, stones i = 0)) → ∃ moves ≥ (n * (∑ i in finset.range k.succ, (1/i))),
  all_stones_at_n n k stones) :=
sorry

/-
Conditions:
1. ℕ represents the set of natural numbers.
2. stones : ℕ → ℕ represents the number of stones on each square.
3. stones 0 = k means that Tony starts with k stones on square 0.
4. 0 < m ≤ n means that m is a valid number of squares to move.
5. ∃ i ≤ m, stones i = 0 means that Tony can advance the stone only up to m squares if there are no stones beyond.
6. The total moves must be at least n * (∑ i in finset.range k.succ, (1/i)).
7. all_stones_at_n n k stones means all k stones must be at square n.
-/


def all_stones_at_n (n k : ℕ) (stones : ℕ → ℕ): Prop :=
  stones n = k 


end tony_min_moves_l680_680277


namespace floor_sqrt_120_l680_680395

theorem floor_sqrt_120 : (Int.floor (Real.sqrt 120) = 10) :=
by
  have h1 : 10^2 = 100 := rfl
  have h2 : 11^2 = 121 := rfl
  have h3 : 100 < 120 < 121 := by simp [h1, h2]
  have h4 : 10 < Real.sqrt 120 < 11 := by
    rw [Real.sqrt_lt, Real.sqrt_lt']
    use 120; exact h3
  exact Int.floor_eq_zero_or_incr (Real.sqrt 120) 10 (by linarith)
  sorry

end floor_sqrt_120_l680_680395


namespace arithmetic_sequence_formula_geometric_sequence_formula_absolute_arithmetic_sequence_sum_l680_680808

open Real

noncomputable def a_sequence (n : ℕ) : ℝ := 11 - 2 * n

noncomputable def b_sequence (n : ℕ) : ℝ := (1 / 2) ^ n

noncomputable def abs_a_sequence_sum (n : ℕ) : ℝ :=
  if n ≤ 5 then 10 * n - n^2 else n^2 - 10 * n + 50

theorem arithmetic_sequence_formula (a_3 a_4 a_5 S_4 : ℝ) :
  S_4 = 4 * (a_3 + 1) →
  3 * a_3 = 5 * a_4 →
  a_4 = a_sequence 4 →
  a_3 = a_sequence 3 →
  a_5 = a_sequence 5 →
  ∀ n : ℕ, a_sequence n = 11 - 2 * n := 
by sorry

theorem geometric_sequence_formula (b_1 b_2 b_3 : ℝ) :
  2 * b_1 = a_sequence 5 →
  b_1 * b_2 = b_3 →
  b_1 = b_sequence 1 →
  b_2 = b_sequence 2 →
  b_3 = b_sequence 3 →
  ∀ n : ℕ, b_sequence n = (1 / 2) ^ n :=
by sorry

theorem absolute_arithmetic_sequence_sum (a_3 a_4 a_5 S_4 b_1 b_2 b_3 : ℝ) :
  S_4 = 4 * (a_3 + 1) →
  3 * a_3 = 5 * a_4 →
  a_4 = a_sequence 4 →
  a_3 = a_sequence 3 →
  a_5 = a_sequence 5 →
  2 * b_1 = a_sequence 5 →
  b_1 * b_2 = b_3 →
  b_1 = b_sequence 1 →
  b_2 = b_sequence 2 →
  b_3 = b_sequence 3 →
  ∀ n : ℕ, abs_a_sequence_sum n = if n ≤ 5 then 10 * n - n^2 else n^2 - 10 * n + 50 :=
by sorry

end arithmetic_sequence_formula_geometric_sequence_formula_absolute_arithmetic_sequence_sum_l680_680808


namespace total_students_is_800_l680_680312

noncomputable def total_students_in_school : ℕ :=
  let G := 150 * 4 / 3 in
  let B := 400 * 3 / 2 in
  B + G

theorem total_students_is_800 
  (B G : ℕ)
  (hG: (3 / 4 : ℝ) * G = 150)
  (hB: ((2 / 3 : ℝ) * B + (3 / 4 : ℝ) * G = 550))
  (G_200 : G = 150 * 4 / 3)
  (B_600 : B = 400 * 3 / 2) 
  : total_students_in_school = 800 := sorry

end total_students_is_800_l680_680312


namespace positive_whole_numbers_with_cube_root_less_than_8_l680_680859

theorem positive_whole_numbers_with_cube_root_less_than_8 :
  { n : ℕ | n > 0 ∧ n < 512 }.card = 511 :=
by
  sorry

end positive_whole_numbers_with_cube_root_less_than_8_l680_680859


namespace max_average_speed_l680_680723

-- Define the initial conditions
def initial_odometer : ℕ := 69696
def total_time_hours : ℕ := 5
def max_speed_kmh : ℕ := 85
def final_odometer : ℕ := 70107

-- Our goal is to prove the maximum average speed
theorem max_average_speed : ((final_odometer - initial_odometer) / total_time_hours) = 82.2 := 
by sorry

end max_average_speed_l680_680723


namespace January25_is_Thursday_l680_680981

-- Definitions based on the conditions provided
def December25_day_of_week () : String := "Monday"
def January25_day_of_week () : String :=
  if (December25_day_of_week () = "Monday") then "Thursday" else "Unknown"

theorem January25_is_Thursday (h : December25_day_of_week () = "Monday") : January25_day_of_week () = "Thursday" :=
by
  sorry

end January25_is_Thursday_l680_680981


namespace positive_whole_numbers_with_cube_root_less_than_8_l680_680861

theorem positive_whole_numbers_with_cube_root_less_than_8 :
  { n : ℕ | n > 0 ∧ n < 512 }.card = 511 :=
by
  sorry

end positive_whole_numbers_with_cube_root_less_than_8_l680_680861


namespace jay_paint_time_correct_l680_680906

noncomputable def jay_paint_time : ℝ :=
  let work_rate_jay (J : ℝ) := 1 / J
  let work_rate_bong := 1 / 3
  let combined_work_rate := 1 / 1.2
  let equation := work_rate_jay J + work_rate_bong = combined_work_rate
  2

theorem jay_paint_time_correct (J : ℝ) (hb : 3 = 3) (ht : 1.2 = 1.2) : jay_paint_time = 2 :=
  sorry

end jay_paint_time_correct_l680_680906


namespace cylinder_volume_ratio_l680_680320

noncomputable def ratio_of_volumes (r h V_small V_large : ℝ) : ℝ := V_large / V_small

theorem cylinder_volume_ratio (r : ℝ) (h : ℝ) 
  (original_height : ℝ := 3 * r)
  (height_small : ℝ := r / 4)
  (height_large : ℝ := 3 * r - height_small)
  (A_small : ℝ := 2 * π * r * (r + height_small))
  (A_large : ℝ := 2 * π * r * (r + height_large))
  (V_small : ℝ := π * r^2 * height_small) 
  (V_large : ℝ := π * r^2 * height_large) :
  A_large = 3 * A_small → 
  ratio_of_volumes r height_small V_small V_large = 11 := by 
  sorry

end cylinder_volume_ratio_l680_680320


namespace ada_initial_position_l680_680236

-- Define the seats and friends
inductive Friend : Type
| Ada | Bea | Ceci | Dee | Edie | Fi | Gigi

def initial_positions : Friend → ℕ
| Friend.Ada := ? -- Ada's initial position is to be determined
| Friend.Bea := 2
| Friend.Ceci := 3
| Friend.Dee := 4
| Friend.Edie := 5
| Friend.Fi := 6
| Friend.Gigi := 7

-- Define the movements
def move_position (pos : ℕ) (displacement : ℕ) : ℕ :=
(pos + displacement - 1) % 7 + 1

-- Define the final positions after movements
def final_positions : Friend → ℕ
| Friend.Ada := 1
| Friend.Bea := move_position (initial_positions Friend.Bea) 3
| Friend.Ceci := move_position (initial_positions Friend.Ceci) (-2)
| Friend.Dee := initial_positions Friend.Edie -- Dee and Edie switch seats
| Friend.Edie := initial_positions Friend.Dee
| Friend.Fi := move_position (initial_positions Friend.Fi) 1
| Friend.Gigi := move_position (initial_positions Friend.Gigi) (-1)

theorem ada_initial_position :
  ∃ pos, initial_positions Friend.Ada = pos ∧ final_positions Friend.Ada = 1 → initial_positions Friend.Ada = 2 :=
sorry

end ada_initial_position_l680_680236


namespace minimum_value_x_plus_2y_l680_680147

theorem minimum_value_x_plus_2y (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x * y^2 = 4) : x + 2*y = 3 * real.cbrt 4 :=
by
  -- Proof goes here
  sorry

end minimum_value_x_plus_2y_l680_680147


namespace necessary_and_sufficient_condition_l680_680499

variable (m n : ℕ)
def positive_integers (m n : ℕ) := m > 0 ∧ n > 0
def at_least_one_is_1 (m n : ℕ) : Prop := m = 1 ∨ n = 1
def sum_gt_product (m n : ℕ) : Prop := m + n > m * n

theorem necessary_and_sufficient_condition (h : positive_integers m n) : 
  sum_gt_product m n ↔ at_least_one_is_1 m n :=
by sorry

end necessary_and_sufficient_condition_l680_680499


namespace smallest_t_value_temperature_l680_680955

theorem smallest_t_value_temperature :
  ∃ t : ℝ, t ≥ 0 ∧ (-t^2 + 14 * t + 40 = 94) ∧ (∀ t' : ℝ, t' ≥ 0 ∧ (-t'^2 + 14 * t' + 40 = 94) → t ≤ t') :=
begin
  sorry
end

end smallest_t_value_temperature_l680_680955


namespace find_alpha_l680_680113

theorem find_alpha (α : ℝ) (h1 : sin α = -1/2) (h2 : α ∈ (π/2, 3 * π / 2)) : α = 7 * π / 6 :=
sorry

end find_alpha_l680_680113


namespace complex_number_property_l680_680615

-- Define complex numbers and their basic operations
def z : ℂ := complex.of_real (sqrt 2) / (1 - complex.I)  -- Using I for the imaginary unit

theorem complex_number_property :
  z * (1 - complex.I) = complex.abs (1 - complex.I) :=
by
  -- z is given by the problem constraints
  let z := (sqrt 2) / (1 - complex.I) in
  -- Expected simplification
  have h1 : 1 + (1 : ℂ) / complex.I = 1 - complex.I := by
  {
    -- Compute the value
    sorry
  },
  have h2 : complex.abs (1 - complex.I) = sqrt 2 := by
  {
    -- Compute the absolute value
    sorry
  },
  -- Show multiplication holds
  calc
  -- the core operation provided by the theorem
  (sqrt 2) / (1 - complex.I) * (1 - complex.I) = sqrt 2 : by
  {
    sorry
  }

end complex_number_property_l680_680615


namespace number_of_required_permutations_l680_680074

-- Define what it means for two elements to be adjacent
def adjacent {α : Type*} (a b : α) (l : List α) : Prop :=
  ∃ (l₁ l₂ : List α), l = l₁ ++ [a, b] ++ l₂ ∨ l = l₁ ++ [b, a] ++ l₂

-- Define the set of elements
def elements := ['A', 'B', 'C', 'D', 'E']

-- Calculate the total number of permutations of the set
def total_permutations := elements.permutations.length

-- Define the property for the required permutations
def required_permutations :=
  (elements.permutations.filter (λ l, ¬ adjacent 'A' 'B' l ∧ ¬ adjacent 'C' 'D' l)).length

theorem number_of_required_permutations :
  required_permutations = 48 :=
by
  sorry

end number_of_required_permutations_l680_680074


namespace quadratic_eq_real_roots_m_ge_neg1_quadratic_eq_real_roots_cond_l680_680803

theorem quadratic_eq_real_roots_m_ge_neg1 (m : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 + 2*(m+1)*x1 + m^2 - 1 = 0 ∧ x2^2 + 2*(m+1)*x2 + m^2 - 1 = 0) →
  m ≥ -1 :=
sorry

theorem quadratic_eq_real_roots_cond (m : ℝ) (x1 x2 : ℝ) :
  x1^2 + 2*(m+1)*x1 + m^2 - 1 = 0 ∧ x2^2 + 2*(m+1)*x2 + m^2 - 1 = 0 ∧
  (x1 - x2)^2 = 16 - x1 * x2 →
  m = 1 :=
sorry

end quadratic_eq_real_roots_m_ge_neg1_quadratic_eq_real_roots_cond_l680_680803


namespace system_solution_l680_680665

theorem system_solution (x y : ℝ) (h1 : x + y = 1) (h2 : x - y = 3) : x = 2 ∧ y = -1 :=
by
  sorry

end system_solution_l680_680665


namespace locus_of_M_l680_680110

-- Defining the given fixed point P
structure Point where
  x : ℝ
  y : ℝ

def P := Point.mk (-1) 0

-- Defining the line l passing through P and intersecting with parabola at two distinct points
def parabola (x y : ℕ) [ring x] [ring y] : Prop :=
  y^2 = x

-- Define the locus problem based on given conditions
theorem locus_of_M :
  ∃ (M : Point), 
    (∀ k : ℝ, -1/2 < k ∧ k < 1/2 ∧ k ≠ 0 → 
      let x1 := some_x1 -- implicitly defined from quadratic roots
      let x2 := some_x2 -- implicitly defined from quadratic roots
      let y1 := some_y1 -- implicitly defined from line equation
      let y2 := some_y2 -- implicitly defined from line equation
      x1 + x2 = (1 - 2 * k^2) / (k^2) ∧ x1 * x2 = 1 ∧ 
      M.x = 1/(2 * k^2) - 1 ∧ M.y = 1/(2 * k) ) → 
    (M.y)^2 = 1/2 * M.x + 1/2 ∧ M.x > 1 :=
sorry

end locus_of_M_l680_680110


namespace cos_terms_rational_l680_680994

theorem cos_terms_rational 
  (x : ℝ) 
  (S : ℝ) 
  (C : ℝ)
  (hS : S = sin (64 * x) + sin (65 * x))
  (hC : C = cos (64 * x) + cos (65 * x))
  (hS_rational : S ∈ ℚ) 
  (hC_rational : C ∈ ℚ) 
: (cos (64 * x) ∈ ℚ) ∧ (cos (65 * x) ∈ ℚ) := 
sorry

end cos_terms_rational_l680_680994


namespace problem_1_problem_2_problem_3_problem_4_l680_680833

def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x ^ 2
  else 2 * x

theorem problem_1 : f 2 = 4 := sorry

theorem problem_2 : f (1 / 2) = 1 / 4 := sorry

theorem problem_3 : f (f (-1)) = 1 := sorry

theorem problem_4 (a : ℝ) (h : f a = 3) : a = 1 ∨ a = real.sqrt 3 := sorry

end problem_1_problem_2_problem_3_problem_4_l680_680833


namespace find_number_exists_l680_680642

theorem find_number_exists (n : ℤ) : (50 < n ∧ n < 70) ∧
    (n % 5 = 3) ∧
    (n % 7 = 2) ∧
    (n % 8 = 2) → n = 58 := 
sorry

end find_number_exists_l680_680642


namespace ella_time_difference_l680_680394

variable (mile_per_day : Nat → ℕ) (speed : Nat → ℕ) 
variable (days : Fin 4)

def ella_time (miles speed : ℕ) : ℕ := miles / speed

def total_time : ℕ := 
  ella_time (mile_per_day 0) (speed 0) + ella_time (mile_per_day 1) (speed 1) + 
  ella_time (mile_per_day 2) (speed 2) + ella_time (mile_per_day 3) (speed 3)

noncomputable def hypothetical_time : ℕ :=
  ella_time (mile_per_day 0 * 4) 5

theorem ella_time_difference :
  mile_per_day 0 = 3 → mile_per_day 1 = 3 → mile_per_day 2 = 3 → mile_per_day 3 = 3 →
  speed 0 = 6 → speed 1 = 4 → speed 2 = 5 → speed 3 = 3 →
  ((total_time - hypothetical_time) * 60 = 27) :=
by
  sorry

end ella_time_difference_l680_680394


namespace determine_duralumin_cubes_l680_680604

-- Define the conditions and the goal
def metal_cubes (n : ℕ) : Prop :=
  ∃ method : finitary_rel (20) (2), ∀ cubes, ∃ d : set cubes, n ≤ 11 ∧
  ∀ cube ∈ d, is_duralumin cube

theorem determine_duralumin_cubes : metal_cubes 11 :=
sorry

end determine_duralumin_cubes_l680_680604


namespace distinct_triangles_in_3x3_grid_l680_680846

theorem distinct_triangles_in_3x3_grid : 
  let num_points := 9 
  let total_combinations := Nat.choose num_points 3 
  let degenerate_cases := 8
  total_combinations - degenerate_cases = 76 := 
by
  sorry

end distinct_triangles_in_3x3_grid_l680_680846


namespace eulerian_path_impossible_doubly_covered_edges_possible_l680_680905

-- Definition of vertices and their degrees
def graph_vertices : Type := {A B C D E F G H I}

def vertex_degree : graph_vertices → ℕ
| A := 2
| B := 3
| C := 2
| D := 3
| E := 6
| F := 3
| G := 2
| H := 3
| I := 2

-- Part (a) statement: Proving the impossibility of an Eulerian path
theorem eulerian_path_impossible : 
  ¬ (∃ p : List graph_vertices, (∀ e ∈ p, is_edge e) ∧ (p.head = p.last ∨ p.tail.head = p.tail.last)) := sorry

-- Part (b) statement: Proving the possibility of drawing each line segment exactly twice
theorem doubly_covered_edges_possible : 
  (∃ p : List graph_vertices, (∀ e ∈ p, is_edge e) ∧ 
    (list.count p = list.size edges * 2)) := sorry

end eulerian_path_impossible_doubly_covered_edges_possible_l680_680905


namespace problem1_problem2_problem3_l680_680843

open Set
open Finset

-- Define the universal set
def U : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define sets A, B, and C
def A : Finset ℕ := {x | x^2 - 3*x + 2 = 0}
def B : Finset ℕ := {x | 1 ≤ x ∧ x ≤ 5 ∧ x ∈ ℕ}
def C : Finset ℕ := {x | 2 < x ∧ x < 9 ∧ x ∈ ℕ}

-- Define complements with respect to U
def complement_U (S : Finset ℕ) : Finset ℕ := U \ S

-- Proof problem 1
theorem problem1 : A ∩ B = {1, 2} := by
  sorry

-- Proof problem 2
theorem problem2 : A ∪ (B ∩ C) = {1, 2, 3, 4, 5} := by
  sorry

-- Proof problem 3
theorem problem3 : complement_U B ∪ complement_U C = {1, 2, 6, 7, 8} := by
  sorry

end problem1_problem2_problem3_l680_680843


namespace number_of_basic_events_prob_of_one_black_prob_of_one_blue_l680_680165
-- Import the mathlib library to use combinatorial and probability functions.

open Finset

-- Definitions of labeled pens
def black_pens : Finset ℕ := {1, 2, 3}  -- {A, B, C}
def blue_pens  : Finset ℕ := {4, 5}      -- {d, e}
def red_pen    : Finset ℕ := {6}         -- {x}
def all_pens   : Finset ℕ := black_pens ∪ blue_pens ∪ red_pen

-- Three pens are drawn randomly
def selection_event : Finset (Finset ℕ) := (all_pens.powerset.filter (λ s => s.card = 3))

noncomputable def probability (event : Finset (Finset ℕ)) : ℚ :=
  (event.card : ℚ) / (selection_event.card : ℚ)

-- Prove the number of basic events
theorem number_of_basic_events : selection_event.card = 20 := by
  sorry

-- Prove the probability of selecting exactly one black pen
def exactly_one_black_pen : Finset (Finset ℕ) :=
  selection_event.filter (λ s => (s ∩ black_pens).card = 1)

theorem prob_of_one_black : probability exactly_one_black_pen = 9 / 20 := by
  sorry

-- Prove the probability of selecting at least one blue pen
def at_least_one_blue_pen : Finset (Finset ℕ) :=
  selection_event.filter (λ s => (s ∩ blue_pens).nonempty)

theorem prob_of_one_blue : probability at_least_one_blue_pen = 4 / 5 := by
  sorry

end number_of_basic_events_prob_of_one_black_prob_of_one_blue_l680_680165


namespace monotonically_increasing_intervals_center_of_symmetry_cos_value_l680_680934

def f (x : ℝ) : ℝ :=
  2 * (Real.sin x) * (Real.cos x) - 2 * (Real.cos (x + Real.pi / 4))^2

theorem monotonically_increasing_intervals :
  ∃ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi - Real.pi / 4) (k * Real.pi + Real.pi / 4), f' x > 0 :=
sorry

theorem center_of_symmetry :
  ∃ k : ℤ, f (k * Real.pi / 2) = -1 :=
sorry

theorem cos_value (x : ℝ) (h1 : 0 < x ∧ x < Real.pi / 2) (h2 : f (x + Real.pi / 6) = 3/5) :
  Real.cos (2 * x) = (4 * Real.sqrt 3 - 3) / 10 :=
sorry

end monotonically_increasing_intervals_center_of_symmetry_cos_value_l680_680934


namespace probability_of_rolling_2_4_6_on_8_sided_die_l680_680659

theorem probability_of_rolling_2_4_6_on_8_sided_die : 
  ∀ (ω : Fin 8), 
  (1 / 8) * (ite (ω = 1 ∨ ω = 3 ∨ ω = 5) 1 0) = 3 / 8 := 
by 
  sorry

end probability_of_rolling_2_4_6_on_8_sided_die_l680_680659


namespace average_goals_per_game_l680_680212

theorem average_goals_per_game
  (number_of_pizzas : ℕ)
  (slices_per_pizza : ℕ)
  (number_of_games : ℕ)
  (h1 : number_of_pizzas = 6)
  (h2 : slices_per_pizza = 12)
  (h3 : number_of_games = 8) : 
  (number_of_pizzas * slices_per_pizza) / number_of_games = 9 :=
by
  sorry

end average_goals_per_game_l680_680212


namespace cubic_polynomial_representation_exists_l680_680988

noncomputable def polynomial := 8 * x^3 + 8 * x^2 + 80 * x + 800

theorem cubic_polynomial_representation_exists :
  ∃ a b c : ℝ, polynomial = a * (x + b)^3 + c ∧ a + b + c = 808.333 := sorry

end cubic_polynomial_representation_exists_l680_680988


namespace product_of_squares_greater_half_l680_680943

theorem product_of_squares_greater_half (n : ℕ) 
  (p : Fin n → ℕ) 
  (hp : ∀ i j : Fin n, i ≠ j → p i ≠ p j) 
  (hq : ∀ i : Fin n, p i > 1) 
  (hr : ∀ i : Fin n, ∃ k : ℕ, p i = k^2) 
  : 
  ∏ i in Finset.univ, (1 - 1 / (p i)^2 : ℝ) > 1 / 2 := 
sorry

end product_of_squares_greater_half_l680_680943


namespace magnitude_of_z_l680_680567

-- Define the complex numbers and the condition.
def z := complex
def w : complex := 1 + 2 * complex.i
def v : complex := 3 - 4 * complex.i

-- State the given condition.
axiom condition : w * z = v

-- State the theorem that needs to be proven.
theorem magnitude_of_z (hz : w * z = v) : complex.abs z = real.sqrt 5 := 
sorry

end magnitude_of_z_l680_680567


namespace double_sum_equality_l680_680726

theorem double_sum_equality : 
  (∑ n in (rangeFrom 3), ∑ k in finset.range (n-2) \ k -> k + 2, (k^2 : ℝ) / 3^(n+k)) = 729 / 17576 :=
sorry

end double_sum_equality_l680_680726


namespace CD_length_l680_680364

variables {D A B C : Type} [InnerProductSpace ℝ D A B C]

-- Given conditions
def volume_tetrahedron : ℝ := 1 / 6
def angle_ACB : ℝ := 45
def sum_AD_BC_AC : ℝ := 3
def AD_BC_AC_sqrt2 := ∀ AD BC AC : ℝ, AD + BC + AC / sqrt 2 = sum_AD_BC_AC

-- Convert the given angle to radians for trigonometric calculations
def degToRad (d : ℝ) : ℝ := d * (Real.pi / 180)

theorem CD_length (AD BC AC : ℝ) (h₁ : 1 / 6 = volume_tetrahedron)
  (h₂ : ∀ x y, ∠ x A y = degToRad angle_ACB)
  (h₃ : AD_BC_AC_sqrt2 AD BC AC)
  : CD = sqrt 3 :=
sorry

end CD_length_l680_680364


namespace allocation_schemes_l680_680322

theorem allocation_schemes (students classes : ℕ) (h_students : students = 5) (h_classes : classes = 3) :
  let count := 90 in
  ∃ (allocations : ℕ), allocations = count := 
by
  sorry

end allocation_schemes_l680_680322


namespace incorrect_judgment_l680_680663

theorem incorrect_judgment : ¬ ((¬ (p ∧ q)) → (¬ p ∧ ¬ q)) :=
by {
  assume h,
  have h1 : ¬ (p ∧ q) → (¬ p ∨ ¬ q), from by apply Classical.not_and_distrib,
  have h2 : ¬ (¬ p ∨ ¬ q) → (p ∧ q), from by apply Classical.demorgan,
  have h3 : p ∧ q, from h2 (Classical.not_not_intro (h1 h)),
  exact h h3,
}

end incorrect_judgment_l680_680663


namespace limit_of_no_marmalade_permutations_l680_680520

-- Definitions matching the problem conditions
def is_no_marmalade_permutation (σ : List ℕ) (n : ℕ) : Prop :=
  ∀ i ∈ σ, i ≠ List.index_of i σ ∧ 
  ∀ j, j > i → ¬ (σ.nthLe i sorry = j ∧ σ.nthLe j sorry = i)

def f (n : ℕ) : ℕ :=
  (List.permutations (List.range n)).count (λ σ, is_no_marmalade_permutation σ n)

-- Statement of the problem
theorem limit_of_no_marmalade_permutations :
  tendsto (λ (n : ℕ), (f n : ℝ) / n.factorial) at_top (𝓝 (Real.exp (-3 / 2))) :=
sorry

end limit_of_no_marmalade_permutations_l680_680520


namespace parabola_focus_distance_3_l680_680442

-- Definitions and conditions
def parabola (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1
def distance_to_line (P : ℝ × ℝ) (a : ℝ) : ℝ := abs (P.1 - a)
def focus : ℝ × ℝ := (1, 0)

-- The theorem
theorem parabola_focus_distance_3 (P : ℝ × ℝ) (hP_on_parabola : parabola P) 
  (hP_distance_to_xn3 : distance_to_line P (-3) = 5) :
  sqrt ((P.1 - focus.1)^2 + (P.2 - focus.2)^2) = 3 :=
by
  sorry

end parabola_focus_distance_3_l680_680442


namespace segment_length_l680_680838

-- Define the parabola and line
def parabola (x : ℝ) : ℝ := (1 / 4) * x ^ 2
def line (x : ℝ) : ℝ := (1 / 2) * x + 1

-- The focus of the parabola
def focus : ℝ × ℝ := (0, 1)

-- Prove the length of the segment cut off by the parabola and the line
theorem segment_length (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (hx1 : parabola x1 = line x1)
  (hx2 : parabola x2 = line x2)
  (hy1 : y1 = parabola x1)
  (hy2 : y2 = parabola x2)
  (h_focus: focus = (0, 1))
  (h_slope : (line 1 - line 0) / (1 - 0) = (1/2)) :
  |(y1 + y2 + 2) - (0 + 0)| = 5 :=
by
  sorry

end segment_length_l680_680838


namespace find_distances_l680_680483

noncomputable def OB_eq (a α β : ℝ) : ℝ :=
  a * Real.sqrt (- (Real.cot α) * Real.tan (α + β))

noncomputable def OC_eq (a α β : ℝ) : ℝ :=
  a * Real.sqrt (- (Real.cot β) * Real.tan (α + β))

theorem find_distances 
  (a α β : ℝ) 
  (OM ON OP : ℝ → Prop)
  (OA_eq : ∃ A, OM A ∧ dist A 0 = a)
  (Angle_AB_eq : ∃ B C, ON B ∧ OP C ∧ ∠ABC = α ∧ ∠ACB = β) :
  (∃ OB OC : ℝ, OB = OB_eq a α β ∧ OC = OC_eq a α β) :=
by
  sorry

end find_distances_l680_680483


namespace solve_quadratic_eq_l680_680244

theorem solve_quadratic_eq {x : ℝ} :
  (x = 3 ∨ x = -1) ↔ x^2 - 2 * x - 3 = 0 :=
by
  sorry

end solve_quadratic_eq_l680_680244


namespace find_tenth_term_l680_680173

/- Define the general term formula -/
def a (a1 d : ℤ) (n : ℤ) : ℤ := a1 + (n - 1) * d

/- Define the sum of the first n terms formula -/
def S (a1 d : ℤ) (n : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem find_tenth_term
  (a1 d : ℤ)
  (h1 : a a1 d 2 + a a1 d 5 = 19)
  (h2 : S a1 d 5 = 40) :
  a a1 d 10 = 29 := by
  /- Sorry used to skip the proof steps. -/
  sorry

end find_tenth_term_l680_680173


namespace f_not_periodic_l680_680541

noncomputable def f (x : ℝ) : ℝ := x + Real.cos x

theorem f_not_periodic : ¬ ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f(x + T) = f x :=
by
  sorry

end f_not_periodic_l680_680541


namespace floor_sqrt_120_l680_680400

theorem floor_sqrt_120 : (Real.floor (Real.sqrt 120)) = 10 :=
by
  have a := 10
  have b := 11
  have h1 : a < b := by norm_num
  have h2 : a^2 < 120 := by norm_num
  have h3 : 120 < b^2 := by norm_num
  sorry

end floor_sqrt_120_l680_680400


namespace unique_difference_of_cubes_l680_680231

theorem unique_difference_of_cubes (x y : ℕ) (h1 : x > y) (h2 : x^3 - y^3 = 19) :
  (x = 3 ∧ y = 2) :=
by
  have h3 : x - y = 1, sorry
  have h4 : x^2 + x * y + y^2 = 19, sorry
  sorry

end unique_difference_of_cubes_l680_680231


namespace bounded_area_l680_680371

-- Define the parametric equations and the line condition
def parametric_x (t : ℝ) : ℝ := 4 * real.sqrt 2 * (real.cos t) ^ 3
def parametric_y (t : ℝ) : ℝ := real.sqrt 2 * (real.sin t) ^ 3

-- Define the bounded x condition
def x_condition (x : ℝ) : Prop := x = 2 ∧ 2 ≤ x

-- Define the statement of area calculation
theorem bounded_area :
  (∫ t in -π/4..π/4, parametric_y t * (derivative parametric_x t)) = 3 * π / 4 :=
sorry

end bounded_area_l680_680371


namespace alice_cookie_fills_l680_680002

theorem alice_cookie_fills :
  (∀ (a b : ℚ), a = 3 + (3/4) ∧ b = 1/3 → (a / b) = 12) :=
sorry

end alice_cookie_fills_l680_680002


namespace seq_an_geom_bn_formula_sum_cn_formula_l680_680100

section subproblem1

theorem seq_an_geom :
  ∀ (a : ℕ → ℝ), (a 1 = 1/2) ∧ (∀ n ≥ 2, a (n - 1) + 1 = 2 * a n) → 
  (∀ n, n ≥ 1 → (a n - 1) / (a (n - 1) - 1) = 1/2) ∧ a = (λ n, 1 - (1 / 2) ^ n) :=
sorry

end subproblem1

section subproblem2

theorem bn_formula :
  ∀ (b : ℕ → ℝ), (∀ n, 2^n * b n = n * 2^n) → 
  b = (λ n, (n + 1) / 2) :=
sorry

end subproblem2

section subproblem3

theorem sum_cn_formula 
  (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ) (T : ℕ → ℝ) :
  (∀ n, a n = 1 - (1 / 2) ^ n) →
  (∀ n, b n = (n + 1) / 2) →
  (∀ n, c n = -2 * a n * b n + (n + 1)) →
  (∀ n, T n = 2 - (n + 3) * (1/2)^n) :=
sorry

end subproblem3

end seq_an_geom_bn_formula_sum_cn_formula_l680_680100


namespace maximize_y_l680_680093

noncomputable def y (x : ℝ) : ℝ :=
  - tan (x + 2 * Real.pi / 3) - tan (x + Real.pi / 6) + cos (x + Real.pi / 6)

theorem maximize_y :
  ∃ x ∈ Set.Icc (-Real.pi / 3) (-Real.pi / 12), y x = (11 / 6) * Real.sqrt 3 := 
sorry

end maximize_y_l680_680093


namespace area_triangle_PQT_l680_680202

noncomputable def sqrt2_sqrt4 : ℝ := real.sqrt(2 + real.sqrt(4))

structure Square (α : Type*) :=
  (P : α)
  (Q : α)
  (R : α)
  (S : α)
  (side_length : ℝ)

structure EquilateralTriangle (α : Type*) :=
  (P : α)
  (Q : α)
  (R : α)
  (side_length : ℝ)

theorem area_triangle_PQT : 
  ∃ T : ℝ × ℝ, 
  (let P := (0, 0 : ℝ),
       Q := (sqrt2_sqrt4, 0 : ℝ),
       R := (sqrt2_sqrt4, sqrt2_sqrt4),
       S := (0, sqrt2_sqrt4) in 
  ∃ (PQRS : Square (ℝ × ℝ)) 
  (PQRS = Square.mk P Q R S sqrt2_sqrt4)
  (PTR : EquilateralTriangle (ℝ × ℝ))
  (PTR = EquilateralTriangle.mk P T R sqrt2_sqrt4)
  (area : ℝ),
  0 < sqrt2_sqrt4 →
  PR.slope ≠ QS.slope →
  line_intersects (P, Q) (T, QS) →
  area = real.sqrt 3) :=
begin
  sorry
end

end area_triangle_PQT_l680_680202


namespace probability_of_rolling_2_4_6_on_8_sided_die_l680_680658

theorem probability_of_rolling_2_4_6_on_8_sided_die : 
  ∀ (ω : Fin 8), 
  (1 / 8) * (ite (ω = 1 ∨ ω = 3 ∨ ω = 5) 1 0) = 3 / 8 := 
by 
  sorry

end probability_of_rolling_2_4_6_on_8_sided_die_l680_680658


namespace problem1_proof_l680_680020

def problem1_expr : ℚ := (5/6) + (-3/4) - |(1/4 : ℚ)| - (-1/6)

theorem problem1_proof : problem1_expr = 0 := by
  have h_abs : |(0.25 : ℚ)| = 1/4 := by sorry
  rw [h_abs]
  sorry

end problem1_proof_l680_680020


namespace number_total_11_l680_680280

theorem number_total_11 (N : ℕ) (S : ℝ)
  (h1 : S = 10.7 * N)
  (h2 : (6 : ℝ) * 10.5 = 63)
  (h3 : (6 : ℝ) * 11.4 = 68.4)
  (h4 : 13.7 = 13.700000000000017)
  (h5 : S = 63 + 68.4 - 13.7) : 
  N = 11 := 
sorry

end number_total_11_l680_680280


namespace identify_person_l680_680179

variable (Person : Type) (Tweedledum Tralyalya : Person)
variable (has_black_card : Person → Prop)
variable (statement_true : Person → Prop)
variable (statement_made_by : Person)

-- Condition: The statement made: "Either I am Tweedledum, or I have a card of a black suit in my pocket."
def statement (p : Person) : Prop := p = Tweedledum ∨ has_black_card p

-- Condition: Anyone with a black card making a true statement is not possible.
axiom black_card_truth_contradiction : ∀ p : Person, has_black_card p → ¬ statement_true p

theorem identify_person :
statement_made_by = Tralyalya ∧ ¬ has_black_card statement_made_by :=
by
  sorry

end identify_person_l680_680179


namespace probability_linda_picks_letter_in_mathematics_l680_680152

def english_alphabet : Finset Char := "ABCDEFGHIJKLMNOPQRSTUVWXYZ".toList.toFinset

def word_mathematics : Finset Char := "MATHEMATICS".toList.toFinset

theorem probability_linda_picks_letter_in_mathematics : 
  (word_mathematics.card : ℚ) / (english_alphabet.card : ℚ) = 4 / 13 := by sorry

end probability_linda_picks_letter_in_mathematics_l680_680152


namespace number_of_wheels_l680_680889

theorem number_of_wheels (V : ℕ) (W_2 : ℕ) (n : ℕ) 
  (hV : V = 16) 
  (h_eq : 2 * W_2 + 16 * n = 66) : 
  n = 4 := 
by 
  sorry

end number_of_wheels_l680_680889


namespace sum_g_from_3_l680_680783

noncomputable def g (n : ℕ) : ℝ := ∑' k in finset.Ico 3 (n+3), 1 / (k^n)

theorem sum_g_from_3 : ∑' n in finset.Ici 3, g n = 1 / 3 := 
by
  -- You would include the actual proof here.
  sorry

end sum_g_from_3_l680_680783


namespace runner_speed_ratio_l680_680336

theorem runner_speed_ratio
  (total_distance : ℝ)
  (distance_first_half : ℝ)
  (distance_second_half : ℝ)
  (time_second_half : ℝ)
  (time_difference : ℝ)
  (h1 : total_distance = 40)
  (h2 : distance_first_half = total_distance / 2)
  (h3 : distance_second_half = total_distance / 2)
  (h4 : time_second_half = 16)
  (h5 : time_difference = 8) :
  (distance_second_half / time_second_half) / (distance_first_half / (time_second_half - time_difference)) = 1 / 2 :=
begin
  sorry
end

end runner_speed_ratio_l680_680336


namespace circles_radii_in_isosceles_right_triangle_l680_680133

noncomputable def isosceles_right_triangle_circles_radii (b : ℝ) :=
let varrho1 := (b*(Real.sqrt 2 - 1))/2,
    varrho2 := (b*(Real.sqrt 2 - 1))/2,
    varrho3 := b * (Real.sqrt 2 - 1/2 - Real.sqrt(2 - Real.sqrt 2)) in
(varrho1, varrho2, varrho3)

theorem circles_radii_in_isosceles_right_triangle (b : ℝ) :
  isosceles_right_triangle_circles_radii b =
  ((b * (Real.sqrt 2 - 1))/2,
   (b * (Real.sqrt 2 - 1))/2,
   b * (Real.sqrt 2 - 1/2 - Real.sqrt(2 - Real.sqrt 2))) :=
sorry

end circles_radii_in_isosceles_right_triangle_l680_680133


namespace area_under_cosine_curve_l680_680250

noncomputable def area_enclosed_by_curve :=
  - ∫ x in 0 .. (Real.pi / 2), -Real.cos x + ∫ x in (Real.pi / 2) .. (3 * Real.pi / 2), -Real.cos x

theorem area_under_cosine_curve :
  area_enclosed_by_curve = 3 :=
sorry

end area_under_cosine_curve_l680_680250


namespace remainder_8_pow_2023_mod_5_l680_680652

theorem remainder_8_pow_2023_mod_5 :
  8 ^ 2023 % 5 = 2 :=
by
  sorry

end remainder_8_pow_2023_mod_5_l680_680652


namespace new_mean_after_modification_l680_680916

theorem new_mean_after_modification :
  ∀ (numbers : List ℕ), 
  numbers.length = 15 →
  (numbers.sum / 15 = 40) →
  let new_numbers := (numbers.take 9).map (λ x, x + 10) ++ (numbers.drop 9).map (λ x, x - 5) in
  (new_numbers.sum / 15 = 44) :=
by
  intros numbers hlen havg
  let new_numbers := (numbers.take 9).map (λ x, x + 10) ++ (numbers.drop 9).map (λ x, x - 5)
  sorry

end new_mean_after_modification_l680_680916


namespace fraction_spent_by_Rica_is_one_fifth_l680_680232

-- Define the conditions
variable (totalPrizeMoney : ℝ) (fractionReceived : ℝ) (amountLeft : ℝ)
variable (h1 : totalPrizeMoney = 1000) (h2 : fractionReceived = 3 / 8) (h3 : amountLeft = 300)

-- Define Rica's original prize money
noncomputable def RicaOriginalPrizeMoney (totalPrizeMoney fractionReceived : ℝ) : ℝ :=
  fractionReceived * totalPrizeMoney

-- Define amount spent by Rica
noncomputable def AmountSpent (originalPrizeMoney amountLeft : ℝ) : ℝ :=
  originalPrizeMoney - amountLeft

-- Define the fraction of prize money spent by Rica
noncomputable def FractionSpent (amountSpent originalPrizeMoney : ℝ) : ℝ :=
  amountSpent / originalPrizeMoney

-- Main theorem to prove
theorem fraction_spent_by_Rica_is_one_fifth :
  let totalPrizeMoney := 1000
  let fractionReceived := 3 / 8
  let amountLeft := 300
  let RicaOriginalPrizeMoney := fractionReceived * totalPrizeMoney
  let AmountSpent := RicaOriginalPrizeMoney - amountLeft
  let FractionSpent := AmountSpent / RicaOriginalPrizeMoney
  FractionSpent = 1 / 5 :=
by {
  -- Proof details are omitted as per instructions
  sorry
}

end fraction_spent_by_Rica_is_one_fifth_l680_680232


namespace zinc_sulfate_produced_l680_680773

-- Definitions of the conditions
def sulfuric_acid_moles := 2
def zinc_moles := 2

-- Balanced chemical equation for reference
axiom balanced_reaction :
  (∀ a b c d : ℕ, a * Zn + b * H2SO4 = c * ZnSO4 + d * H2 → a = b ∧ b = c ∧ c = 1 ∧ d = 1)

-- Let's state the theorem
theorem zinc_sulfate_produced :
  (Zn_moles = 2) ∧ (H2SO4_moles = 2) →
  ZnSO4_moles = 2 :=
by {
  sorry
}

end zinc_sulfate_produced_l680_680773


namespace probability_of_no_adjacent_stands_is_correct_l680_680054

noncomputable def number_of_arrangements : ℕ → ℕ
| 2 := 3
| 3 := 4
| (n+1) := number_of_arrangements n + number_of_arrangements (n-1)

def total_outcomes : ℕ := 2^8

def favorable_outcomes : ℕ := number_of_arrangements 8

def probability := (favorable_outcomes : ℚ) / total_outcomes

theorem probability_of_no_adjacent_stands_is_correct :
  probability = 47 / 256 :=
by sorry

end probability_of_no_adjacent_stands_is_correct_l680_680054


namespace number_of_men_first_group_l680_680690

theorem number_of_men_first_group :
  (∃ M : ℕ, 30 * 3 * (M : ℚ) * (84 / 30) / 3 = 112 / 6) → ∃ M : ℕ, M = 20 := 
by
  sorry

end number_of_men_first_group_l680_680690


namespace problem_solution_l680_680115

noncomputable def alpha : ℝ := (3 + Real.sqrt 13) / 2
noncomputable def beta  : ℝ := (3 - Real.sqrt 13) / 2

theorem problem_solution : 7 * alpha ^ 4 + 10 * beta ^ 3 = 1093 :=
by
  -- Prove roots relation
  have hr1 : alpha * alpha - 3 * alpha - 1 = 0 := by sorry
  have hr2 : beta * beta - 3 * beta - 1 = 0 := by sorry
  -- Proceed to prove the required expression
  sorry

end problem_solution_l680_680115


namespace symmetric_concurrent_or_parallel_l680_680226

theorem symmetric_concurrent_or_parallel 
  (A B C P : Point) -- defining points
  (l_A l_B l_C : Line)
  (d_A d_B d_C : Line)
  (l_A' l_B' l_C' : Line)
  (hA : l_A.Contains A) 
  (hB : l_B.Contains B)
  (hC : l_C.Contains C)
  (h_common : l_A.IsConcurrentWith l_B l_C P)
  (dA_bisect : IsAngleBisector d_A A B C)
  (dB_bisect : IsAngleBisector d_B B A C)
  (dC_bisect : IsAngleBisector d_C C A B)
  (h_sym_lA : IsSymmetricTo l_A l_A' d_A)
  (h_sym_lB : IsSymmetricTo l_B l_B' d_B)
  (h_sym_lC : IsSymmetricTo l_C l_C' d_C) :
  l_A'.IsConcurrentWith l_B' l_C' ∨ l_A'.IsParallelTo l_B' ∧ l_A'.IsParallelTo l_C' := sorry

end symmetric_concurrent_or_parallel_l680_680226


namespace pillar_at_P_height_l680_680007

open Real

noncomputable def height_of_pillar_at_P : ℝ :=
  let L := (0, 0, 0) : ℝ × ℝ × ℝ
  let M := (10, 0, 0) : ℝ × ℝ × ℝ
  let N := (5, 5 * sqrt 3, 0) : ℝ × ℝ × ℝ
  let S := (0, 0, 15 : ℝ) 
  let T := (10, 0, 12 : ℝ)
  let U := (5, 5 * sqrt 3, 13 : ℝ)
  let ST := (10 - 0, 0 - 0, 12 - 15 : ℝ × ℝ × ℝ)
  let SU := (5 - 0, 5 * sqrt 3 - 0, 13 - 15 : ℝ × ℝ × ℝ)
  let n := (
    (0 * (-2) - (-3) * (5 * sqrt 3)),
    - ((10 * (-2) - (-3) * 5)),
    (10 * (5 * sqrt 3) - 0 * 5) : ℝ × ℝ × ℝ
  )
  let d := (15 * sqrt 3 * 0 + 35 * 0 + 50 * sqrt 3 * 15)
  by {
    let P := (0, -10 * sqrt 3, decide (750 / 50 + 15))
    P.2.2
  }

theorem pillar_at_P_height :
  height_of_pillar_at_P = 22 :=
sorry

end pillar_at_P_height_l680_680007


namespace count_numbers_with_cube_root_lt_8_l680_680850

theorem count_numbers_with_cube_root_lt_8 : 
  ∀ n : ℕ, (n > 0) → (n < 8^3) → n ≤ 8^3 - 1 :=
by
  -- We need to prove that the count of such numbers is 511
  sorry

end count_numbers_with_cube_root_lt_8_l680_680850


namespace percent_flamingos_among_non_parrots_l680_680366

theorem percent_flamingos_among_non_parrots
  (total_birds : ℝ) (flamingos : ℝ) (parrots : ℝ) (eagles : ℝ) (owls : ℝ)
  (h_total : total_birds = 100)
  (h_flamingos : flamingos = 40)
  (h_parrots : parrots = 20)
  (h_eagles : eagles = 15)
  (h_owls : owls = 25) :
  ((flamingos / (total_birds - parrots)) * 100 = 50) :=
by sorry

end percent_flamingos_among_non_parrots_l680_680366


namespace no_extremum_iff_l680_680948

noncomputable def f (a x : ℝ) : ℝ := exp x + a * x

theorem no_extremum_iff (a : ℝ) : (∀ x : ℝ, deriv (f a) x ≠ 0) → a ≥ 0 :=
  by
  sorry

end no_extremum_iff_l680_680948


namespace binary_representation_of_14_l680_680407

theorem binary_representation_of_14 : nat.binary_repr 14 = "1110" :=
sorry

end binary_representation_of_14_l680_680407


namespace cos_beta_eq_sqrt10_over_10_l680_680114

-- Define the conditions and the statement
theorem cos_beta_eq_sqrt10_over_10 
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h_tan : Real.tan α = 2)
  (h_sin_sum : Real.sin (α + β) = Real.sqrt 2 / 2) :
  Real.cos β = Real.sqrt 10 / 10 :=
sorry

end cos_beta_eq_sqrt10_over_10_l680_680114


namespace more_cats_than_dogs_l680_680374

-- Define the initial conditions
def initial_cats : ℕ := 28
def initial_dogs : ℕ := 18
def cats_adopted : ℕ := 3

-- Compute the number of cats after adoption
def cats_now : ℕ := initial_cats - cats_adopted

-- Define the target statement
theorem more_cats_than_dogs : cats_now - initial_dogs = 7 := by
  unfold cats_now
  unfold initial_cats
  unfold cats_adopted
  unfold initial_dogs
  sorry

end more_cats_than_dogs_l680_680374


namespace coincident_circumcenters_of_triangles_l680_680180

theorem coincident_circumcenters_of_triangles
  (A B C S A1 A2 B1 B2 C1 C2 Ap App Bp Bpp Cp Cpp : Point)
  (hS_centroid : centroid (triangle A B C) = S)
  (h_segments_parallel :
    parallel (line_through S M)
             (line_through X Y) ∧ -- replace M, X, Y with specific points as necessary
  (h_equilateral : equilateral \(...) ∧ -- condition for equilateral triangles
  (h_midpoints  : midpoint A A1 = Ap ∧ midpoint A A2 = App ∧
                  midpoint B B1 = Bp ∧ midpoint B B2 = Bpp ∧
                  midpoint C C1 = Cp ∧ midpoint C C2 = Cpp)
  (h_circumcenter1 : circumcenter (triangle Ap Bp Cp) = S)
  (h_circumcenter2 : circumcenter (triangle App Bpp Cpp) = S) :
  (circumcenter (triangle Ap Bp Cp) = circumcenter (triangle App Bpp Cp)) :=
begin
  sorry
end

end coincident_circumcenters_of_triangles_l680_680180


namespace min_value_of_y_l680_680292

variable {x : ℝ}

def log_condition (x : ℝ) : Prop := (Real.log x / Real.log 2) ^ 2 - (Real.log x / Real.log 2) - 2 ≤ 0

def y (x : ℝ) := 4 ^ x - 2 ^ x + 3

theorem min_value_of_y : x ∈ {x | log_condition x} → ∃ x₀, y x₀ = 5 - Real.sqrt 2 :=
by
  sorry

end min_value_of_y_l680_680292


namespace range_of_a_l680_680839

theorem range_of_a (a : ℝ) :
  (¬ (∃ x ∈ Icc (-1 : ℝ) 1, 2 * x^2 + a * x - a^2 = 0) ∨
   ∃! x_0 : ℝ, x_0^2 + 2 * a * x_0 + 2 * a ≤ 0) →
  a ∈ Ioo (-∞: ℝ) (-2) ∪ Ioo (2) (∞) :=
by
  sorry

end range_of_a_l680_680839


namespace number_of_birds_flew_up_correct_l680_680679

def initial_number_of_birds : ℕ := 29
def final_number_of_birds : ℕ := 42
def number_of_birds_flew_up : ℕ := final_number_of_birds - initial_number_of_birds

theorem number_of_birds_flew_up_correct :
  number_of_birds_flew_up = 13 := sorry

end number_of_birds_flew_up_correct_l680_680679


namespace ratio_to_percent_l680_680630

theorem ratio_to_percent :
  (9 / 5 * 100) = 180 :=
by
  sorry

end ratio_to_percent_l680_680630


namespace angle_bisector_theorem_l680_680162

noncomputable def ratio_of_segments (x y z p q : ℝ) :=
  q / x = y / (y + x)

theorem angle_bisector_theorem (x y z p q : ℝ) (h1 : p / x = q / y)
  (h2 : p + q = z) : ratio_of_segments x y z p q :=
by
  sorry

end angle_bisector_theorem_l680_680162


namespace euler_totient_coprime_euler_totient_formula_l680_680237

variable {a b n : ℕ}
variable {p : ℕ → Prop}

def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def euler_totient (n : ℕ) : ℕ :=
  (Finset.range n).filter (λ k => Nat.gcd k n = 1).card

theorem euler_totient_coprime (a b : ℕ) (h : coprime a b) :
  euler_totient (a * b) = euler_totient a * euler_totient b := sorry

theorem euler_totient_formula (n : ℕ) (factors : List ℕ)
  (h_factors : ∀ p ∈ factors, p.Prime ∧ p ∣ n) :
  euler_totient n = n * ∏ p in factors.toFinset, (1 - 1 / p : ℚ) := sorry

end euler_totient_coprime_euler_totient_formula_l680_680237


namespace find_angle_A_l680_680160

theorem find_angle_A (a b c A : ℝ) (h1 : a^2 = b^2 + (sqrt 3) * b * c + c^2) (h2 : 0 < A ∧ A < Real.pi) : 
  A = 5 * Real.pi / 6 := 
by 
  sorry

end find_angle_A_l680_680160


namespace pyramid_circumscribed_radius_l680_680607

noncomputable def inscribed_radius : ℝ := real.sqrt 2 - 1

noncomputable def circumscribed_radius (r : ℝ) : ℝ := real.sqrt 6 + 1

theorem pyramid_circumscribed_radius :
  circumscribed_radius inscribed_radius = real.sqrt 6 + 1 := 
sorry

end pyramid_circumscribed_radius_l680_680607


namespace stone_pillar_shape_l680_680697

theorem stone_pillar_shape (r h : ℝ) (a b c : ℝ) (V : ℝ) 
  (h_radius : r = 2)
  (h_dimensions : (a = 2 ∧ b = 8 ∧ c = 12) ∨ (a = 2 ∧ b = 12 ∧ c = 8) ∨ (a = 8 ∧ b = 2 ∧ c = 12)
    ∨ (a = 8 ∧ b = 12 ∧ c = 2) ∨ (a = 12 ∧ b = 2 ∧ c = 8) ∨ (a = 12 ∧ b = 8 ∧ c = 2))
  (h_fits : (2 * r) ≤ min a (min b c))
  (h_max_height : h = 12) 
  (h_volume : V = π * r^2 * h) :
  ∃ height : ℝ, ∃ shape : string, shape = "cylinder" ∧ r = 2 ∧ height = 12 :=
by
  sorry

end stone_pillar_shape_l680_680697


namespace triangle_possible_values_l680_680993

theorem triangle_possible_values (n : ℕ) (hn : 0 < n) (A : 0 < (1 + log 2 48 + log 2 n)) (B : 0 < (1 + log 2 8 + log 2 48 - log 2 n)) (C : 0 < (1 + log 2 8 + log 2 n - log 2 48)): 
  (7 ≤ n ∧ n < 384) → ∃ k : ℕ, k = 377 := 
by
  sorry

end triangle_possible_values_l680_680993


namespace negation_of_proposition_p_l680_680448

-- Define the proposition p
def proposition_p : Prop := ∀ (n : ℕ), ¬prime (2^n - 2)

-- Statement of negation of proposition p
theorem negation_of_proposition_p : ¬proposition_p ↔ ∃ (n : ℕ), prime (2^n - 2) := by
  sorry

end negation_of_proposition_p_l680_680448


namespace october_visitors_l680_680009

def visitors_in_october (V : ℝ) (november_increase : V * 0.15) (december_increase : V * 0.15 + 15) (total_visitors : V + (1 + 0.15) * V + ((1 + 0.15) * V + 15) = 345) : Prop :=
  V = 100

-- The statement that needs to be proven
theorem october_visitors : ∃ V : ℝ, visitors_in_october V (0.15 * V) (0.15 * V + 15) (V + 1.15 * V + (1.15 * V + 15) = 345) :=
begin
  use 100,
  sorry
end

end october_visitors_l680_680009


namespace binary_representation_of_14_binary_representation_of_14_l680_680412

-- Define the problem as a proof goal
theorem binary_representation_of_14 : (14 : ℕ) = 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 0 * 2^0 :=
by sorry

-- An alternative formula to exactly represent the binary string using a conversion function can be provided:
theorem binary_representation_of_14' : nat.to_digits 2 14 = [1, 1, 1, 0] :=
by sorry

end binary_representation_of_14_binary_representation_of_14_l680_680412


namespace increase_in_area_l680_680513

theorem increase_in_area (a : ℝ) : 
  let original_radius := 3
  let new_radius := original_radius + a
  let original_area := π * original_radius ^ 2
  let new_area := π * new_radius ^ 2
  new_area - original_area = π * (3 + a) ^ 2 - 9 * π := 
by
  sorry

end increase_in_area_l680_680513


namespace solve_for_sum_l680_680591

theorem solve_for_sum (x y : ℝ) (h : x^2 + y^2 = 18 * x - 10 * y + 22) : x + y = 4 + 2 * Real.sqrt 42 :=
sorry

end solve_for_sum_l680_680591


namespace trig_expression_independence_of_phi_l680_680588

theorem trig_expression_independence_of_phi (α ϕ : ℝ) :
  4 * cos α * cos ϕ * cos (α - ϕ) + 2 * sin (α - ϕ) ^ 2 - cos (2 * ϕ) = cos (2 * α) + 2 :=
by
  sorry

end trig_expression_independence_of_phi_l680_680588


namespace Q1_Intersection_Q1_Union_Q2_l680_680806

namespace Example

def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 5}

def B (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 2}

-- Question 1: 
theorem Q1_Intersection (a : ℝ) (ha : a = -1) : 
  A ∩ B a = {x | -2 ≤ x ∧ x ≤ -1} :=
sorry

theorem Q1_Union (a : ℝ) (ha : a = -1) :
  A ∪ B a = {x | x ≤ 1 ∨ x ≥ 5} :=
sorry

-- Question 2:
theorem Q2 (a : ℝ) :
  (A ∩ B a = B a) ↔ (a ≤ -3 ∨ a > 2) :=
sorry

end Example

end Q1_Intersection_Q1_Union_Q2_l680_680806


namespace students_between_jimin_yuna_l680_680683

theorem students_between_jimin_yuna 
  (total_students : ℕ) 
  (jimin_position : ℕ) 
  (yuna_position : ℕ) 
  (h1 : total_students = 32) 
  (h2 : jimin_position = 27) 
  (h3 : yuna_position = 11) 
  : (jimin_position - yuna_position - 1) = 15 := 
by
  sorry

end students_between_jimin_yuna_l680_680683


namespace smallest_nineteen_l680_680300

theorem smallest_nineteen :
  ∃ (n : nat), 
  (n = 19 ∧ 
   ∃ (x : fin n → ℝ), 
   (∀ i, -1 < x i ∧ x i < 1) ∧ 
   (finset.univ.sum x = 0) ∧ 
   (finset.univ.sum (λ i, (x i) ^ 2) = 20)) ∧
  (∀ m : nat, (m < 19 → ¬∃ (x : fin m → ℝ), 
   (∀ i, -1 < x i ∧ x i < 1) ∧ 
   (finset.univ.sum x = 0) ∧ 
   (finset.univ.sum (λ i, (x i) ^ 2) = 20)))
:=
sorry

end smallest_nineteen_l680_680300


namespace ratio_of_percent_increase_to_decrease_l680_680297

variable (P U V : ℝ)
variable (h1 : P * U = 0.25 * P * V)
variable (h2 : P ≠ 0)

theorem ratio_of_percent_increase_to_decrease (h : U = 0.25 * V) :
  ((V - U) / U) * 100 / 75 = 4 :=
by
  sorry

end ratio_of_percent_increase_to_decrease_l680_680297


namespace integer_solutions_l680_680944

theorem integer_solutions (x y : ℤ) (h₁ : x + y ≠ 0) :
  (x^2 + y^2) / (x + y) = 10 ↔
  (x, y) ∈ {(-2, 4), (-2, 6), (0, 10), (4, -2), (4, 12), (6, -2), (6, 12), (10, 0), (10, 10), (12, 4), (12, 6)} :=
by
  sorry

end integer_solutions_l680_680944


namespace max_alpha_beta_square_l680_680866

theorem max_alpha_beta_square (k : ℝ) (α β : ℝ)
  (h1 : α^2 - (k - 2) * α + (k^2 + 3 * k + 5) = 0)
  (h2 : β^2 - (k - 2) * β + (k^2 + 3 * k + 5) = 0)
  (h3 : α ≠ β) :
  (α^2 + β^2) ≤ 18 :=
sorry

end max_alpha_beta_square_l680_680866


namespace circle_cos_intersection_l680_680318

-- Definition of the circle equation
def circle (h k r : ℝ) (x y : ℝ) : Prop := (x - h) ^ 2 + (y - k) ^ 2 = r ^ 2

-- Definition of the graph of cos x
def graph_cos (x y : ℝ) : Prop := y = Real.cos x

-- The statement to be proved: There can be more than 16 intersection points
theorem circle_cos_intersection (h k r : ℝ) : 
  (∃ x y : ℝ, circle h k r x y ∧ graph_cos x y) →
  (∃ (n : ℕ), n > 16 ∧ ∃ (x y : ℝ), circle h k r x y ∧ graph_cos x y) :=
sorry

end circle_cos_intersection_l680_680318


namespace correct_proposition_2_correct_proposition_4_l680_680485

variables {Line Plane : Type} 
variables (m l : Line) (α β : Plane)

-- Predicate definitions corresponding to the conditions
def is_subset (l : Line) (α : Plane) : Prop := sorry
def is_parallel (α β : Plane) : Prop := sorry
def is_perpendicular (x y : Prop) : Prop := sorry
def projection_within (l : Line) (α : Plane) : Prop := sorry

-- Definitions extracted from the problem conditions
axiom h1 : is_subset l β
axiom h2 : is_perpendicular l α
axiom h3 : is_perpendicular l α
axiom h4 : is_perpendicular m β
axiom h5 : is_perpendicular l m

-- Proof problems based on correct answers (2) and (4)
theorem correct_proposition_2 : is_perpendicular α β :=
by sorry

theorem correct_proposition_4 : is_perpendicular α β :=
by sorry

end correct_proposition_2_correct_proposition_4_l680_680485


namespace sum_of_edges_of_rectangular_solid_l680_680275

theorem sum_of_edges_of_rectangular_solid 
  (a r : ℝ) 
  (volume_eq : (a / r) * a * (a * r) = 343) 
  (surface_area_eq : 2 * ((a^2 / r) + (a^2 * r) + a^2) = 294) 
  (gp : a / r > 0 ∧ a > 0 ∧ a * r > 0) :
  4 * ((a / r) + a + (a * r)) = 84 :=
by
  sorry

end sum_of_edges_of_rectangular_solid_l680_680275


namespace select_officers_correct_probability_same_college_correct_l680_680166

noncomputable def total_officers : Nat := 36 + 24 + 12
noncomputable def sampling_ratio : Rat := 6 / total_officers

def num_officers_selected (college_size : Nat) : Nat := college_size * sampling_ratio

def prob_same_college_selected : Rat := 4 / 15

theorem select_officers_correct :
  num_officers_selected 36 = 3 ∧
  num_officers_selected 24 = 2 ∧
  num_officers_selected 12 = 1 := by
  sorry

theorem probability_same_college_correct :
  prob_same_college_selected = 4 / 15 := by
  sorry

end select_officers_correct_probability_same_college_correct_l680_680166


namespace linear_eq_value_abs_sum_l680_680816

theorem linear_eq_value_abs_sum (a m : ℤ)
  (h1: m^2 - 9 = 0)
  (h2: m ≠ 3)
  (h3: |a| ≤ 3) : 
  |a + m| + |a - m| = 6 :=
by
  sorry

end linear_eq_value_abs_sum_l680_680816


namespace day_after_53_days_from_monday_is_friday_l680_680286

constant days_in_week : ℕ := 7
constant start_day : ℕ := 0 -- Monday

noncomputable def day_of_week (days_from_start : ℕ) : ℕ :=
  (start_day + days_from_start) % days_in_week

theorem day_after_53_days_from_monday_is_friday :
  day_of_week 53 = 4 := 
  sorry

end day_after_53_days_from_monday_is_friday_l680_680286


namespace evaluate_expression_l680_680758

variable (x y : ℝ)

def P : ℝ := 2 * x + y
def Q : ℝ := x - 2 * y

theorem evaluate_expression :
  (P + Q) / (P - Q) - (P - Q) / (P + Q) = 8 * (x^2 - 2 * x * y - y^2) / ((x + 3 * y) * (3 * x - y)) :=
by
  sorry

end evaluate_expression_l680_680758


namespace product_pqr_roots_cos_l680_680247

noncomputable def Q (x : ℂ) (p q r : ℂ) := x^3 + p * x^2 + q * x + r

theorem product_pqr_roots_cos (p q r : ℂ) :
  (∀ x, Q x p q r = 0 ↔ x = (complex.cos (real.pi / 9)) ∨ x = (complex.cos (2 * real.pi / 9)) ∨ x = (complex.cos (4 * real.pi / 9))) →
  p * q * r = (1/576 : ℂ) :=
by
  sorry

end product_pqr_roots_cos_l680_680247


namespace probability_even_and_greater_than_14_l680_680235

-- Define the set of numbered balls
def balls : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the condition of valid pairs for the product to be even and greater than 14
def is_valid_pair (a b : ℕ) : Prop :=
  (a ∈ balls ∧ b ∈ balls ∧ (a * b) % 2 = 0 ∧ a * b > 14)

-- Define the probability of a valid pair
def probability_of_valid_pair : ℚ :=
  11 / 49

theorem probability_even_and_greater_than_14 :
  (∑ (a : ℕ) in balls.toFinset, ∑ (b : ℕ) in balls.toFinset, if is_valid_pair a b then 1 else 0) / 49 = probability_of_valid_pair := by
  sorry

end probability_even_and_greater_than_14_l680_680235


namespace first_two_digits_of_one_over_137_l680_680649

theorem first_two_digits_of_one_over_137 : 
  let x := (1 : ℚ) / 137 in
    (∃ a b : ℕ, a ≠ 0 ∧ b ≠ 0 ∧ a < 10 ∧ b < 10 ∧ 
    ∃ n : ℕ, x * 100 ^ (n + 1) = a * 10 + b) → 
    (a = 7 ∧ b = 6) :=
by
  sorry

end first_two_digits_of_one_over_137_l680_680649


namespace distance_between_intersections_of_circle_and_line_l680_680044

theorem distance_between_intersections_of_circle_and_line :
  let circle_eqn := x^2 + y^2 = 25,
  let line_eqn := x + y = 5,
  ∃ (P Q : ℝ × ℝ), 
    (P.1^2 + P.2^2 = 25) ∧ (Q.1^2 + Q.2^2 = 25) ∧ 
    (P.1 + P.2 = 5) ∧ (Q.1 + Q.2 = 5) ∧ 
    dist P Q = 5 * √2 := sorry

end distance_between_intersections_of_circle_and_line_l680_680044


namespace ratio_of_volumes_l680_680817

theorem ratio_of_volumes (m n : ℕ) (h₁ : Nat.coprime m n)
  (h₂ : Tetrahedron T) (h₃ : Cube C)
  (vol_T : volume T = s³ * √3 / 12)
  (vol_C : volume C = (16 * s³ * √6) / 243):
  (m, n) = (81, 64) ∧ m + n = 145 := by
  sorry

end ratio_of_volumes_l680_680817


namespace simplify_expression_l680_680597

-- Defining each term in the sum
def term1  := 1 / ((1 / 3) ^ 1)
def term2  := 1 / ((1 / 3) ^ 2)
def term3  := 1 / ((1 / 3) ^ 3)

-- Sum of the terms
def terms_sum := term1 + term2 + term3

-- The simplification proof statement
theorem simplify_expression : 1 / terms_sum = 1 / 39 :=
by sorry

end simplify_expression_l680_680597


namespace geo_sequence_general_formula_sum_b_abs_l680_680475

variable {a : ℕ → ℤ}

-- Given conditions:
axiom a_1 : a 1 = 5
axiom a_2 : a 2 = 5
axiom a_recurrence : ∀ n, n ≥ 2 → a (n + 1) = a n + 6 * a (n - 1)

-- Question 1: Prove that {a_{n+1} + 2a_n} is a geometric sequence with first term 15 and common ratio 3.
theorem geo_sequence (n : ℕ) (h: n ≥ 2) : 
  ∃ (r : ℤ), r = 3 ∧ a (n + 1) + 2 * a n = r * (a n + 2 * a (n - 1)) := sorry

-- Question 2: Find the general formula for the sequence {a_n}.
theorem general_formula (n : ℕ) : a n = 2 * (-2)^(n-1) + 3^n := sorry

-- Question 3: Let 3^n * b_n = n(3^n - a_n), find the sum of |b_1| + |b_2| + ... + |b_n|.
def b (n : ℕ) : ℤ := n * (3^n - a n) / 3^n

theorem sum_b_abs (n : ℕ) : 
  (Finset.range n).sum (λ i, |b (i + 1)|) = 6 - 2 * (n + 3) * (2 / 3)^n := sorry

end geo_sequence_general_formula_sum_b_abs_l680_680475


namespace perimeter_of_triangle_MN_F2_l680_680456

-- Definitions of constants:
def a := 4 -- semi-major axis length since \(2a = 8\)
def ellipse (x y : ℝ) := x^2 / 16 + y^2 / 9 = 1

-- Definitions of propositions:
def distance (p q : ℝ × ℝ) := real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
def isOnEllipse (p : ℝ × ℝ) := ellipse p.1 p.2

def focus1 : ℝ × ℝ := (-4, 0) -- Assumed locations based on the standard form
def focus2 : ℝ × ℝ := (4, 0)

variables (M N : ℝ × ℝ)
  (H1 : isOnEllipse M)
  (H2 : isOnEllipse N)
  (H3 : ∃ t : ℝ, M = (t*(-4),  t*0))  -- Line passing through F1
  (H4 : ∃ t : ℝ, N = (t*(-4),  t*0))  -- Line passing through F1

theorem perimeter_of_triangle_MN_F2 : 
  distance focus1 M + distance focus2 M +
  distance focus1 N + distance focus2 N = 16 := 
sorry

end perimeter_of_triangle_MN_F2_l680_680456


namespace calculate_fourth_quarter_points_l680_680368

variable (W1 W2 W3 W4 L1 : ℕ)

-- Conditions
-- 1. At the end of the first quarter, the winning team had double the points of the losing team.
def condition1 := W1 = 2 * L1 

-- 2. At the end of the second quarter, the winning team had 10 more points than it started with.
def condition2 := W2 = W1 + 10

-- 3. At the end of the third quarter, the winning team had 20 more points than the number it had in the second quarter.
def condition3 := W3 = W2 + 20

-- 4. The total points the winning team scored in the game was 80.
def condition4 := W1 + W2 + W3 + W4 = 80

-- 5. The losing team had 10 points in the first quarter.
def condition5 := L1 = 10

theorem calculate_fourth_quarter_points :
  condition1 W1 L1 ∧ condition2 W2 W1 ∧ condition3 W3 W2 ∧ condition4 W1 W2 W3 W4 ∧ condition5 L1 → W4 = 30 :=
by {
  intro h,
  have h1 := h.1,
  have h2 := h.2.1,
  have h3 := h.2.2.1,
  have h4 := h.2.2.2.1,
  have h5 := h.2.2.2.2,
  sorry
}

end calculate_fourth_quarter_points_l680_680368


namespace max_days_baron_l680_680080

def max_days_condition (D : ℕ → ℝ) (n : ℕ) : Prop :=
  D(n) > D(n-2) ∧ D(n) < D(n-7)

theorem max_days_baron (D : ℕ → ℝ) : 
  ∃ N, ∀ n ≥ N, ¬max_days_condition D n := 
begin
  sorry -- This is where the proof would be constructed.
end

end max_days_baron_l680_680080


namespace non_involved_count_l680_680882

-- Define the number of students and Jacob’s direct friends and their friends.
variables (students : set ℕ) (jacob : ℕ) (friendship : ℕ → ℕ → Prop)

-- Jacob is one of the students, let's assume 25 students and index them from 0 to 24.
hypothesis h_students : students = {0, 1, ..., 24}
hypothesis h_jacob : jacob ∈ students

-- Friendship relation is symmetric and reflexive.
hypothesis h_friendship_symm : ∀ {a b : ℕ}, friendship a b → friendship b a
hypothesis h_friendship_refl : ∀ {a : ℕ}, a ∈ students → friendship a a

-- Jane will include his immediate friends and those at most two friendship links away.
def involved_in_project : set ℕ := {x | friendship jacob x ∨ (∃ y, friendship jacob y ∧ friendship y x) ∨ (∃ y z, friendship jacob y ∧ friendship y z ∧ friendship z x)}

-- Non-involved classmates are those who are not in the involved_in_project set.
def non_involved : set ℕ := students \ involved_in_project

-- Number of non-involved students.
def num_non_involved := set.card non_involved

theorem non_involved_count : num_non_involved = 8 :=
by {
  sorry
}

end non_involved_count_l680_680882


namespace largest_coefficient_terms_l680_680532

open BigOperators

noncomputable def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

def general_term (n r : ℕ) (a b : ℤ) : ℤ :=
  binomial_coeff n r * a^(n-r) * b^r

def specific_term (r : ℕ) : ℤ :=
  binomial_coeff 13 r * (x^2)^(13-r) * ((1/x)^r)

theorem largest_coefficient_terms :
  specific_term 6 = specific_term 7 :=
sorry

end largest_coefficient_terms_l680_680532


namespace closest_ratio_l680_680608

noncomputable def ratio_closest_to_one (a c : ℕ) : ℚ :=
  if c = 0 then 0 else (a : ℚ) / (c : ℚ)

theorem closest_ratio :
  ∃ (a c : ℕ), 30 * a + 15 * c = 2400 ∧
              a > 0 ∧
              c > 0 ∧
              (ratio_closest_to_one a c = 27/26) :=
begin
  sorry
end

end closest_ratio_l680_680608


namespace more_cats_than_dogs_l680_680377

theorem more_cats_than_dogs:
  ∃ (cats_before cats_after dogs: ℕ),
    cats_before = 28 ∧
    dogs = 18 ∧
    cats_after = cats_before - 3 ∧
    cats_after - dogs = 7 :=
by
  use 28, 25, 18
  split
  case left =>
    exact rfl
  case right =>
    split
    case left =>
      exact rfl
    case right =>
      split
      case left =>
        exact rfl
      case right =>
        exact rfl

end more_cats_than_dogs_l680_680377


namespace rita_remaining_money_l680_680969

-- Defining the conditions
def num_dresses := 5
def price_dress := 20
def num_pants := 3
def price_pant := 12
def num_jackets := 4
def price_jacket := 30
def transport_cost := 5
def initial_amount := 400

-- Calculating the total cost
def total_cost : ℕ :=
  (num_dresses * price_dress) + 
  (num_pants * price_pant) + 
  (num_jackets * price_jacket) + 
  transport_cost

-- Stating the proof problem 
theorem rita_remaining_money : initial_amount - total_cost = 139 := by
  sorry

end rita_remaining_money_l680_680969


namespace probability_geologists_distance_greater_than_six_km_l680_680529

-- Noncomputable theory since we're dealing with probabilities.
noncomputable theory

-- Define the conditions for the problem.
def num_roads : ℕ := 8

def geologist_speed : ℝ := 4

-- Define the problem statement.
theorem probability_geologists_distance_greater_than_six_km :
  let probability := 0.375
  in probability = 24 / (8 * 8) :=
sorry

end probability_geologists_distance_greater_than_six_km_l680_680529


namespace average_goals_per_game_l680_680214

theorem average_goals_per_game
  (slices_per_pizza : ℕ := 12)
  (total_pizzas : ℕ := 6)
  (total_games : ℕ := 8)
  (total_slices : ℕ := total_pizzas * slices_per_pizza)
  (total_goals : ℕ := total_slices)
  (average_goals : ℕ := total_goals / total_games) :
  average_goals = 9 :=
by
  sorry

end average_goals_per_game_l680_680214


namespace length_of_train_is_800_2_l680_680343

-- Definitions and conditions
def train_speed_kmh : ℝ := 78
def tunnel_length_m : ℝ := 500
def crossing_time_s : ℝ := 60

-- Convert speed to m/s
def train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600

-- Expected length of the train in meters
def train_length_m : ℝ := 800.2

theorem length_of_train_is_800_2 :
  let total_distance : ℝ := train_speed_ms * crossing_time_s,
      train_length : ℝ := total_distance - tunnel_length_m
  in train_length = train_length_m :=
by
  sorry

end length_of_train_is_800_2_l680_680343


namespace find_v2_poly_l680_680285

theorem find_v2_poly (x : ℤ) (v0 v1 v2 : ℤ) 
  (h1 : x = -4)
  (h2 : v0 = 1) 
  (h3 : v1 = v0 * x)
  (h4 : v2 = v1 * x + 6) :
  v2 = 22 :=
by
  -- To be filled with proof (example problem requirement specifies proof is not needed)
  sorry

end find_v2_poly_l680_680285


namespace third_smallest_palindromic_prime_l680_680628

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem third_smallest_palindromic_prime {n : ℕ} (h_step1 : is_palindrome 101 ∧ is_prime 101) (h_step2 : is_palindrome 131 ∧ is_prime 131) (h_step3 : n = 151) :
  is_palindrome n ∧ is_prime n :=
by
  sorry

end third_smallest_palindromic_prime_l680_680628


namespace trigonometric_identity_l680_680796

theorem trigonometric_identity (α : ℝ) (h : Real.sin (π + α) = -1/3) : Real.sin (2 * α) / Real.cos α = 2 / 3 := by
  sorry

end trigonometric_identity_l680_680796


namespace relationship_a_b_l680_680087

noncomputable def a (n : ℕ) (h : 0 < n) : ℝ := (1 / n) * (∑ i in finset.range n, ((i + 1 : ℝ) / n) ^ 2)
def b : ℝ := ∫ x in 0..1, x ^ 2

theorem relationship_a_b (n : ℕ) (h : 0 < n) : a n h > b := sorry

end relationship_a_b_l680_680087


namespace complex_expression_value_l680_680722

-- Define the complex number (1 - i)
def z : ℂ := 1 - complex.I

-- Define the expression (1 - i)^2 * i
def expr : ℂ := z^2 * complex.I

-- State the theorem to prove the expression is equal to 2
theorem complex_expression_value : expr = 2 :=
by
  -- This is where the proof would go
  sorry

end complex_expression_value_l680_680722


namespace log_identity_example_l680_680381

theorem log_identity_example :
  (2 : ℝ) * log 5 10 + log 5 0.25 = 2 :=
by
  sorry

end log_identity_example_l680_680381


namespace largest_number_is_D_l680_680664

def repeating_decimal (int_part : ℕ) (nonrep_part : ℕ) (rep_part : ℕ) (rep_length : ℕ) : ℝ :=
  int_part + (nonrep_part / (10 ^ rep_length)) + (rep_part / (10 ^ rep_length)) / (1 - (1 / (10 ^ rep_length)))

def A : ℝ := 3.2571
def B : ℝ := repeating_decimal 3 0 2571 4
def C : ℝ := repeating_decimal 3 2 571 3
def D : ℝ := repeating_decimal 3 25 71 2
def E : ℝ := repeating_decimal 3 257 1 1

theorem largest_number_is_D : D > A ∧ D > B ∧ D > C ∧ D > E :=
by sorry

end largest_number_is_D_l680_680664


namespace arithmetic_mean_geometric_mean_l680_680586

theorem arithmetic_mean_geometric_mean (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  (a + b) / 2 ≥ Real.sqrt (a * b) :=
sorry

end arithmetic_mean_geometric_mean_l680_680586


namespace cost_of_jacket_is_60_l680_680021

/-- Define the constants from the problem --/
def cost_of_shirt : ℕ := 8
def cost_of_pants : ℕ := 18
def shirts_bought : ℕ := 4
def pants_bought : ℕ := 2
def jackets_bought : ℕ := 2
def carrie_paid : ℕ := 94

/-- Define the problem statement --/
theorem cost_of_jacket_is_60 (total_cost jackets_cost : ℕ) 
    (H1 : total_cost = (shirts_bought * cost_of_shirt) + (pants_bought * cost_of_pants) + jackets_cost)
    (H2 : carrie_paid = total_cost / 2)
    : jackets_cost / jackets_bought = 60 := 
sorry

end cost_of_jacket_is_60_l680_680021


namespace product_is_even_l680_680930

-- Definitions are captured from conditions

noncomputable def is_permutation {α : Type*} [DecidableEq α] (l1 l2 : List α) : Prop :=
  l1.perm l2

theorem product_is_even (a : ℕ → ℕ) :
  (is_permutation (List.range 2015) (List.ofFn (λ i, a i + 2014))) →
  Even (Finset.univ.prod (λ i : Fin 2015, a i - i.val.succ)) :=
by
  sorry

end product_is_even_l680_680930


namespace irreducible_polynomial_l680_680984

theorem irreducible_polynomial (k : ℤ) (hk : ¬(5 ∣ k)) : 
  ¬ ∃ (p q : polynomial ℤ), p.degree < 5 ∧ q.degree < 5 ∧ p * q = polynomial.C k + polynomial.X^5 - polynomial.X :=
begin
  sorry
end

end irreducible_polynomial_l680_680984


namespace probability_two_red_balls_l680_680666

theorem probability_two_red_balls (total_balls red_balls blue_balls green_balls : ℕ)
  (total_balls_eq : total_balls = 13)
  (red_balls_eq : red_balls = 5)
  (blue_balls_eq : blue_balls = 6)
  (green_balls_eq : green_balls = 2)
  (pick_two_is_78 : (total_balls.choose 2) = 78)
  (pick_two_red_is_10 : (red_balls.choose 2) = 10) :
  ( (red_balls.choose 2).to_rat / (total_balls.choose 2).to_rat = 5 / 39 ) :=
sorry

end probability_two_red_balls_l680_680666


namespace dessert_menu_count_l680_680699

def Dessert : Type := {d : String // d = "cake" ∨ d = "pie" ∨ d = "ice cream" ∨ d = "pudding"}

def valid_menu (menu : Fin 7 → Dessert) : Prop :=
  (menu 0).1 ≠ (menu 1).1 ∧
  menu 1 = ⟨"ice cream", Or.inr (Or.inr (Or.inl rfl))⟩ ∧
  (menu 1).1 ≠ (menu 2).1 ∧
  (menu 2).1 ≠ (menu 3).1 ∧
  (menu 3).1 ≠ (menu 4).1 ∧
  (menu 4).1 ≠ (menu 5).1 ∧
  menu 5 = ⟨"cake", Or.inl rfl⟩ ∧
  (menu 5).1 ≠ (menu 6).1

def total_valid_menus : Nat :=
  4 * 1 * 3 * 3 * 3 * 1 * 3

theorem dessert_menu_count : ∃ (count : Nat), count = 324 ∧ count = total_valid_menus :=
  sorry

end dessert_menu_count_l680_680699


namespace probability_no_adjacent_stands_l680_680055

-- Definitions based on the conditions
def fair_coin_flip : ℕ := 2 -- Each person can flip one of two possible outcomes (head or tail).

-- The main theorem stating the probability
theorem probability_no_adjacent_stands : 
  let total_outcomes := fair_coin_flip ^ 8 in -- Total number of possible sequences
  let favorable_outcomes := 47 in -- Number of valid sequences where no two adjacent people stand
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 47 / 256 :=
by
  sorry

end probability_no_adjacent_stands_l680_680055


namespace greatest_integer_lt_M_over_100_l680_680452

theorem greatest_integer_lt_M_over_100 :
  (let M := (∑ n in {3, 4, 5, 6, 7, 8, 9}, Nat.choose 19 n) * 2 / 19 in
  ⌊M / 100⌋ = 275) :=
by
  sorry

end greatest_integer_lt_M_over_100_l680_680452


namespace largest_three_digit_solution_l680_680650

theorem largest_three_digit_solution :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 55 * n ≡ 165 [MOD 260] ∧
    ∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ 55 * m ≡ 165 [MOD 260] → m ≤ n :=
begin
  sorry
end

end largest_three_digit_solution_l680_680650


namespace actual_average_height_of_students_l680_680305

def average_height_corrected (n : ℕ) (incorrect_avg : ℝ) (incorrect_height : ℝ) (actual_height : ℝ) : ℝ :=
  let incorrect_total := incorrect_avg * n
  let difference := incorrect_height - actual_height
  let correct_total := incorrect_total - difference
  correct_total / n

theorem actual_average_height_of_students :
  average_height_corrected 30 175 151 136 = 174.5 :=
by 
  -- This is the proof we need to skip using sorry.
  sorry

end actual_average_height_of_students_l680_680305


namespace angle_BFC_right_angle_l680_680807

theorem angle_BFC_right_angle
  (A B C P Q F B₁ C₁ : Type) -- Define points as type
  (acute_triangle : Triangle A B C)
  (altitude_BB₁ : LinearDependant A B₁)
  (altitude_CC₁ : LinearDependant A C₁)
  (ext_P : Collinear B B₁ P)
  (ext_Q : Collinear C C₁ Q)
  (angle_PAQ_90 : ∠ P A Q = 90°)
  (altitude_AF : LinearDependant A F)
  (altitude_APQ : Triangle A P Q) :
  ∠ B F C = 90° := sorry

end angle_BFC_right_angle_l680_680807


namespace part_I_part_II_l680_680844

-- Part I
theorem part_I (x : ℝ) (m : ℝ) (h : m = 2)
  (hp : |x + 1| ≤ 3)
  (hq : x^2 - 2 * x + 1 - m^2 ≤ 0) :
  x ∈ Icc (-4) (-1) ∪ Icc 2 3 :=
sorry

-- Part II
theorem part_II (x : ℝ) (m : ℝ)
  (hp : |x + 1| ≤ 3)
  (hq : x^2 - 2 * x + 1 - m^2 ≤ 0)
  (hm : m > 0) :
  (0 < m ∧ m ≤ 1) ↔ (∀ (x : ℝ), |x + 1| ≤ 3 → x^2 - 2 * x + 1 - m^2 ≤ 0) :=
sorry

end part_I_part_II_l680_680844


namespace probability_even_sum_l680_680952

theorem probability_even_sum (m n : ℕ) (hnrelprime : Nat.coprime m n) (hprob : (5/21 : ℚ) = m / n) :
  m + n = 26 :=
sorry

end probability_even_sum_l680_680952


namespace hyperbola_equation_l680_680822

-- Definitions of given conditions
def hyperbola_asymptote (x y : ℝ) : Prop := x - √3 * y = 0
def parabola_directrix (x y : ℝ) : Prop := y^2 = -4 * x
def hyperbola (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the theorem to be proved
theorem hyperbola_equation 
  (a b x y k : ℝ)
  (ha_pos : a > 0)
  (hb_pos : b > 0)
  (hasymptote : hyperbola_asymptote x y)
  (hdirectrix : parabola_directrix x y)
  (hfocus : x = 1)
  (hra : a = √3 * k)
  (hrb : b = k) :
  hyperbola x y (√3 * (1/2)) (1/2) :=
by
   sorry

end hyperbola_equation_l680_680822


namespace ball_distribution_l680_680491

theorem ball_distribution :
  (∃ f : Fin 6 → Fin 2, (∀ b : Fin 2, (∃ n : ℕ, (n ≤ 4 ∧ (f ⁻¹' {b}).card = n)) 
  ∧  (Finset.filter (λ i, f i = 0) (Finset.univ: Finset (Fin 6))).card ≤ 4 
  ∧  (Finset.filter (λ i, f i = 1) (Finset.univ: Finset (Fin 6))).card ≤ 4 
  ∧ Finset.card (Finset.range 2) = 2)) ∧ 
  (∑ i in (Finset.filter (λ i, i = 1) (Finset.Powerset (Finset.fin_range 6))), 
  2 * binomial 6 (i.card)) / 2 + 
  (∑ i in (Finset.filter (λ i, i = 0) (Finset.Powerset (Finset.fin_range 6))), 
  binomial 6 (i.card)) = 25 :=
begin
  sorry
end

end ball_distribution_l680_680491


namespace G_eq_G_l680_680194

-- Define the problem
variable (A K L M N G G' : Point)
variable (alpha : ℝ)
variable (R : Point → ℝ → Point → Point)

-- Condition: A K L and A M N are similar isosceles triangles with vertex A and angle alpha.
noncomputable def similar_triangles_AKL_AMN (A K L M N : Point) (alpha : ℝ) : Prop := sorry

-- Condition: G N K and G' L M are similar isosceles triangles with angle π - alpha at the vertex.
noncomputable def similar_triangles_GNK_GLM (G G' N K L M : Point) (alpha : ℝ) : Prop := sorry

-- Rotation transformations
noncomputable def rotation (P : Point) (theta : ℝ) (X : Point) : Point := R P theta X

-- Given transformations:
noncomputable def transform1 (G' A N L : Point) (alpha : ℝ) : Prop := 
  L = rotation G' (π - alpha) (rotation A alpha N)

noncomputable def transform2 (G A L N : Point) (alpha : ℝ) : Prop := 
  N = rotation G (π - alpha) (rotation A alpha L)

-- Main statement to prove: G = G'
theorem G_eq_G' (A K L M N G G' : Point) (alpha : ℝ) (R : Point → ℝ → Point → Point) 
  (similar1 : similar_triangles_AKL_AMN A K L M N alpha)
  (similar2 : similar_triangles_GNK_GLM G G' N K L M alpha)
  (trans1 : transform1 G' A N L alpha)
  (trans2 : transform2 G A L N alpha) :
  G = G' := 
sorry

end G_eq_G_l680_680194


namespace area_bounded_by_sec2theta_csc2theta_yx_yaxis_l680_680068

-- Define the polar forms and rotation
def sec (x : ℝ) := 1 / Real.cos x
def csc (x : ℝ) := 1 / Real.sin x

-- Hypothesize the region bounded by the given curves
theorem area_bounded_by_sec2theta_csc2theta_yx_yaxis :
  let r_sec := λ θ, sec (2 * θ)
  let r_csc := λ θ, csc (2 * θ)
  let x := λ θ, r_sec θ * Real.cos θ
  let y := λ θ, r_csc θ * Real.sin θ
  area_bounded_by r_sec r_csc (λ x, x) (λ y, y) = 0.125 :=
begin
  sorry
end

end area_bounded_by_sec2theta_csc2theta_yx_yaxis_l680_680068


namespace cos_tan_inequality_unbounded_l680_680072

theorem cos_tan_inequality_unbounded :
  ∀ (n : ℕ) (x : ℝ), (0 < n) → (tan x ≠ 0) → (cos x ^ n + tan x ^ n ≥ 1 / n) :=
by
  -- The proof is skipped using 'sorry'
  sorry

end cos_tan_inequality_unbounded_l680_680072


namespace stones_required_correct_l680_680701

/- 
Given:
- The hall measures 36 meters long and 15 meters broad.
- Each stone measures 6 decimeters by 5 decimeters.

We need to prove that the number of stones required to pave the hall is 1800.
-/
noncomputable def stones_required 
  (hall_length_m : ℕ) 
  (hall_breadth_m : ℕ) 
  (stone_length_dm : ℕ) 
  (stone_breadth_dm : ℕ) : ℕ :=
  (hall_length_m * 10) * (hall_breadth_m * 10) / (stone_length_dm * stone_breadth_dm)

theorem stones_required_correct : 
  stones_required 36 15 6 5 = 1800 :=
by 
  -- Placeholder for proof
  sorry

end stones_required_correct_l680_680701


namespace volume_ratio_proof_l680_680164

-- Definitions:
def height_ratio := 2 / 3
def volume_ratio (r : ℚ) := r^3
def small_pyramid_volume_ratio := volume_ratio height_ratio
def frustum_volume_ratio := 1 - small_pyramid_volume_ratio
def volume_ratio_small_to_frustum (v_small v_frustum : ℚ) := v_small / v_frustum

-- Lean 4 Statement:
theorem volume_ratio_proof
  (height_ratio : ℚ := 2 / 3)
  (small_pyramid_volume_ratio : ℚ := volume_ratio height_ratio)
  (frustum_volume_ratio : ℚ := 1 - small_pyramid_volume_ratio)
  (v_orig : ℚ) :
  volume_ratio_small_to_frustum (small_pyramid_volume_ratio * v_orig) (frustum_volume_ratio * v_orig) = 8 / 19 :=
by
  sorry

end volume_ratio_proof_l680_680164


namespace find_smallest_x_l680_680775

theorem find_smallest_x : ∃ (x : ℕ), (52 * x + 14) % 24 = 6 ∧ x = 4 :=
by {
  use 4,
  split,
  { norm_num },
  { refl }
}

end find_smallest_x_l680_680775


namespace cos4_minus_sin4_pi_over_8_eq_sqrt2_over_2_l680_680629

theorem cos4_minus_sin4_pi_over_8_eq_sqrt2_over_2 :
  (cos (Real.pi / 8)) ^ 4 - (sin (Real.pi / 8)) ^ 4 = (Real.sqrt 2) / 2 := 
by 
  sorry

end cos4_minus_sin4_pi_over_8_eq_sqrt2_over_2_l680_680629


namespace birds_flew_up_count_l680_680681

def initial_birds : ℕ := 29
def final_birds : ℕ := 42

theorem birds_flew_up_count : final_birds - initial_birds = 13 :=
by sorry

end birds_flew_up_count_l680_680681


namespace problem1_problem2_l680_680740

-- Proof problem 1
theorem problem1 : 
  (real.cbrt 2 * real.sqrt 3) ^ 6 + (real.sqrt (2 * real.sqrt 2)) ^ (4 / 3) - 
  4 * (16 / 49) ^ (-1 / 2) - real.root 4 2 * 8 ^ (1 / 4) - (-2017) ^ 0 = 100 :=
by
  sorry

-- Proof problem 2
theorem problem2 :
  real.log 2.5 6.25 + real.lg 0.01 + real.ln (real.sqrt real.exp 1) - 2 ^ (1 + real.log 2 3) = -11 / 2 :=
by
  sorry

end problem1_problem2_l680_680740


namespace spherical_to_rectangular_coordinates_l680_680746

theorem spherical_to_rectangular_coordinates :
  ∀ (ρ θ φ : ℝ), ρ = 10 ∧ θ = 3 * Real.pi / 4 ∧ φ = Real.pi / 6 →
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z) = (-5 * Real.sqrt 2 / 2, 5 * Real.sqrt 2 / 2, 5 * Real.sqrt 3)
  :=
by
  intros ρ θ φ h
  rcases h with ⟨hρ, hθ, hφ⟩
  simp [hρ, hθ, hφ]
  sorry

end spherical_to_rectangular_coordinates_l680_680746


namespace largest_non_related_sequences_l680_680186

-- Define the properties of binary sequences and the allowed operations
def binary_sequence (n : ℕ) := vector bool n

def flip_subsequence (s : binary_sequence 2022) (start length : ℕ) : binary_sequence 2022 :=
  let prefix := s.to_list.take start
  let block := s.to_list.slice start (start + length)
  let block_flipped := block.reverse
  let suffix := s.to_list.drop (start + length)
  ⟨prefix ++ block_flipped ++ suffix, by sorry⟩

def related (s₁ s₂ : binary_sequence 2022) : Prop :=
  ∃ m, (list.nilable (flip_subsequence^m) s₁ = s₂)

-- Statement of the problem in Lean 4
theorem largest_non_related_sequences :
  ∃ n, (∀ (A : list (binary_sequence 2022)), A.length = n →
    (∀ (i j : ℕ), i ≠ j → ¬related (A.nth_le i sorry) (A.nth_le j sorry)))
    ∧ n = 2025 :=
begin
  sorry
end

end largest_non_related_sequences_l680_680186


namespace rice_grains_difference_l680_680348

theorem rice_grains_difference :
  let grains (k : ℕ) := 2^k in
  (grains 12) - (∑ k in Finset.range 10, grains (k + 1)) = 2050 :=
by
  let grains := λ k : ℕ, 2^k
  have s : ∑ k in Finset.range 10, grains (k + 1) = 2046 := sorry
  have t : grains 12 = 4096 := sorry
  calc
    4096 - 2046 = 2050 : by norm_num

end rice_grains_difference_l680_680348


namespace sequence_properties_l680_680514

noncomputable def geometric_sequence_sum (a r n : ℕ) : ℕ := a * ((r^n - 1) / (r - 1))

theorem sequence_properties (a : ℕ) (f : ℕ → ℕ)
  (h1 : ∀ n, log 2 (f (n + 1)) = 1 + log 2 (f n))
  (h2 : (finset.range 5).sum f = 62) :
  f 10 = 1024 :=
sorry

end sequence_properties_l680_680514


namespace cos_pi_minus_2alpha_l680_680795

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.sin α = 2 / 3) : Real.cos (Real.pi - 2 * α) = -1 / 9 := 
by
  sorry

end cos_pi_minus_2alpha_l680_680795


namespace magnitude_subtraction_l680_680823

variables (m n : ℝ → ℝ → ℝ)

-- Define the magnitude of vectors m and n
axiom mag_m : real.sqrt (m 0 + m 1) = real.sqrt 3
axiom mag_n : real.sqrt (n 0 + n 1) = 2

-- Define the angle between vectors m and n
axiom angle_m_n : real.acos ((m 0 * n 0 + m 1 * n 1) / (real.sqrt (m 0 ^ 2 + m 1 ^ 2) * real.sqrt (n 0 ^ 2 + n 1 ^ 2))) = real.pi / 6

-- Prove the magnitude of m - n
theorem magnitude_subtraction : real.sqrt ((m 0 - n 0) ^ 2 + (m 1 - n 1) ^ 2) = 1 := 
begin
  sorry
end

end magnitude_subtraction_l680_680823


namespace chord_length_of_intersection_l680_680770

theorem chord_length_of_intersection 
  (A B C : ℝ) (x0 y0 r : ℝ)
  (line_eq : A * x0 + B * y0 + C = 0)
  (circle_eq : (x0 - 1)^2 + (y0 - 3)^2 = r^2) 
  (A_line : A = 4) (B_line : B = -3) (C_line : C = 0) 
  (x0_center : x0 = 1) (y0_center : y0 = 3) (r_circle : r^2 = 10) :
  2 * (Real.sqrt (r^2 - ((A * x0 + B * y0 + C) / (Real.sqrt (A^2 + B^2)))^2)) = 6 :=
by
  sorry

end chord_length_of_intersection_l680_680770


namespace trapezoid_of_centroid_collinear_l680_680696

variables {A B C D M : Type*} [affine_space ℝ A] [affine_space ℝ B]
variables [affine_space ℝ C] [affine_space ℝ D] [affine_space ℝ M]
variables 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : ∃ P, is_intersection_of_diagonals P A B C D ∧ P = M)
  (h3 : is_on_line (centroid A B M) (centroid C D M) M)

theorem trapezoid_of_centroid_collinear (A B C D M : Type*) [affine_space ℝ A] [affine_space ℝ B]
    [affine_space ℝ C] [affine_space ℝ D] [affine_space ℝ M]
    (h1 : is_convex_quadrilateral A B C D) 
    (h2 : ∃ P, is_intersection_of_diagonals P A B C D ∧ P = M)
    (h3 : is_on_line (centroid A B M) (centroid C D M) M) : is_trapezoid A B C D :=
sorry

end trapezoid_of_centroid_collinear_l680_680696


namespace yoongi_has_fewer_apples_l680_680545

-- Define the number of apples Jungkook originally has and receives more.
def jungkook_original_apples := 6
def jungkook_received_apples := 3

-- Calculate the total number of apples Jungkook has.
def jungkook_total_apples := jungkook_original_apples + jungkook_received_apples

-- Define the number of apples Yoongi has.
def yoongi_apples := 4

-- State that Yoongi has fewer apples than Jungkook.
theorem yoongi_has_fewer_apples : yoongi_apples < jungkook_total_apples := by
  sorry

end yoongi_has_fewer_apples_l680_680545


namespace spring_length_relationship_l680_680457

def spring_length (x : ℝ) : ℝ := 6 + 0.3 * x

theorem spring_length_relationship (x : ℝ) : spring_length x = 0.3 * x + 6 :=
by sorry

end spring_length_relationship_l680_680457


namespace binary_representation_of_14_binary_representation_of_14_l680_680410

-- Define the problem as a proof goal
theorem binary_representation_of_14 : (14 : ℕ) = 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 0 * 2^0 :=
by sorry

-- An alternative formula to exactly represent the binary string using a conversion function can be provided:
theorem binary_representation_of_14' : nat.to_digits 2 14 = [1, 1, 1, 0] :=
by sorry

end binary_representation_of_14_binary_representation_of_14_l680_680410


namespace constant_term_correct_max_binomial_term_correct_l680_680462

-- Define the conditions and required constants
noncomputable def binomial_expansion (x : ℝ) (n : ℕ) : ℝ := 
  (1 / x + 2 * real.cbrt x)^n

-- Define the expansion terms
noncomputable def constant_term (x : ℝ) : ℝ :=
  nat.choose 8 6 * 2^6

noncomputable def max_binomial_term (x : ℝ) : ℝ :=
  nat.choose 8 4 * 2^4 * x^(-8.0 / 3.0)

-- Lean statements for the proof problems
theorem constant_term_correct (x : ℝ) (h : binomial_expansion x 8 = 256) : constant_term x = 1792 := 
by { sorry }

theorem max_binomial_term_correct (x : ℝ) (h : binomial_expansion x 8 = 256) : max_binomial_term x = 1120 * x^(-8.0 / 3.0) := 
by { sorry }

end constant_term_correct_max_binomial_term_correct_l680_680462


namespace min_value_of_expression_l680_680429

theorem min_value_of_expression (a b c : ℝ) (hb : b > a) (ha : a > c) (hc : b ≠ 0) :
  ∃ l : ℝ, l = 5.5 ∧ l ≤ (a + b)^2 / b^2 + (b + c)^2 / b^2 + (c + a)^2 / b^2 :=
by
  sorry

end min_value_of_expression_l680_680429


namespace average_goals_per_game_l680_680211

theorem average_goals_per_game
  (number_of_pizzas : ℕ)
  (slices_per_pizza : ℕ)
  (number_of_games : ℕ)
  (h1 : number_of_pizzas = 6)
  (h2 : slices_per_pizza = 12)
  (h3 : number_of_games = 8) : 
  (number_of_pizzas * slices_per_pizza) / number_of_games = 9 :=
by
  sorry

end average_goals_per_game_l680_680211


namespace value_of_f_five_sixths_l680_680127

noncomputable def f : ℝ → ℝ 
| x => if x ≤ 0 then 3 * x else f (x - 1)

theorem value_of_f_five_sixths : f (5 / 6) = -1 / 2 :=
by sorry

end value_of_f_five_sixths_l680_680127


namespace find_b_l680_680149

-- Definitions for conditions
def is_square_of_binomial (f : ℝ → ℝ) (q : ℝ) : Prop :=
  ∀ x : ℝ, f x = (x + q) ^ 2

-- Theorem statement using conditions above
theorem find_b (p : ℝ) (b : ℝ) (h₁ : p = -10) (h₂ : ∃ q : ℝ, is_square_of_binomial (λ x, x^2 + p * x + b) q) : b = 25 :=
by 
  sorry

end find_b_l680_680149


namespace conic_sections_l680_680624

theorem conic_sections (x y : ℝ) : 
  y^4 - 16*x^4 = 8*y^2 - 4 → 
  (y^2 - 4 * x^2 = 4 ∨ y^2 + 4 * x^2 = 4) :=
sorry

end conic_sections_l680_680624


namespace inscribed_square_side_length_l680_680032

theorem inscribed_square_side_length {A B C P Q R S : Type} [linear_ordered_field A] 
  (h_triangle : ∀ {x y z : A}, (x = y) ∧ (y = z) ∧ (z = x))
  (h_angle : ∠ BAC = 60)
  (h_AB : AB = 30)
  (h_AC : AC = 30)
  (h_BC : BC = 30)
  (h_CQ : CQ = 29)
  (h_P_on_AB : P ∈ segment A B)
  (h_Q_on_BC : Q ∈ segment B C)
  (h_R_on_CA : R ∈ segment C A)
  (h_square : square PQRS)
  (h_PQ : PQ = x) : 
  x = 30 :=
by
  sorry

end inscribed_square_side_length_l680_680032


namespace price_reduction_l680_680317

variable (sale_increase : ℕ → ℕ)
variable (n p profit_goal : ℕ)

theorem price_reduction (x : ℝ) :
  n = 20 → p = 44 → sale_increase x = 20 + 5 * x → profit_goal = 1600 → 
  (p - x) * sale_increase x = profit_goal → 
  x = 4 ∨ x = 36 :=
by
  intros h_n h_p h_sale_increase h_profit_goal h_eq
  simp at h_n h_p h_sale_increase h_profit_goal h_eq
  simp [h_n, h_p, h_sale_increase, h_profit_goal] at h_eq
  sorry

end price_reduction_l680_680317


namespace buratino_sunday_letters_l680_680570

theorem buratino_sunday_letters (avg : ℕ) (n : ℕ) (wednesday thursday friday saturday : ℕ)
  (h_avg : avg = 9) (h_n : n = 7)
  (h_wed : 13 ≤ wednesday) (h_thu : 12 ≤ thursday) 
  (h_fri : 9 ≤ friday) (h_sat : 7 ≤ saturday) :
  let total_letters := n * avg in
  let sum_visible := wednesday + thursday + friday + saturday in
  total_letters - sum_visible = 0 :=
by
  let monday := 0
  let tuesday := 0
  sorry

end buratino_sunday_letters_l680_680570


namespace intersection_of_circles_l680_680306

open Classical

variables {Ω ω : Circle}
variables {A B C P Q O : Point}
variables {Γb Γc : Circle}

-- Definitions of the given conditions:
def is_inscribed (ABC : Triangle) (Ω : Circle) := 
  Ω.contains A ∧ Ω.contains B ∧ Ω.contains C

def is_circumscribed (ABC : Triangle) (ω : Circle) := 
  ∃ B2 C2, ω.tangentAt B2 ∧ ω.tangentAt C2 ∧ B2 ∈ AC ∧ C2 ∈ AB

def passes_through_center (P Q : Point) (O : Point) := 
  aligned P O Q

def constructed_on_as_diameter (B P : Point) (Γb : Circle) := 
  diameter_on_segment B P Γb

-- The theorem to be proved:
theorem intersection_of_circles 
  (ABC : Triangle)
  (Ω ω : Circle)
  (A B C P Q O : Point)
  (Γb Γc : Circle)
  (h1 : is_inscribed ABC Ω)
  (h2 : is_circumscribed ABC ω)
  (h3 : passes_through_center P Q O)
  (h4 : constructed_on_as_diameter B P Γb)
  (h5 : constructed_on_as_diameter C Q Γc):
  ∃ X Y, Γb ∩ Γc = {X, Y} ∧ (Ω.contains X ∧ ω.contains Y) :=
sorry

end intersection_of_circles_l680_680306


namespace curve_trajectory_max_area_triangle_l680_680094

noncomputable def circle := {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 9}
noncomputable def line_l1 := {p : ℝ × ℝ | p.1 - (√3) * p.2 + 6 = 0}

-- Definition of moving point A on the circle, point B on the x-axis, and point N  
def is_on_circle (A : ℝ × ℝ) : Prop := A ∈ circle
def is_on_x_axis (B : ℝ × ℝ) : Prop := B.2 = 0
def AB_perpendicular_x_axis (A B : ℝ × ℝ) : Prop := B.1 = A.1 ∧ B.2 = 0
def AB_NB_relation (A B N : ℝ × ℝ) : Prop := (0, -A.2) = √3 * (A.1 - N.1, -N.2)

-- Equation of the curve
theorem curve_trajectory :
  ∃ C : set (ℝ × ℝ), ∀ p ∈ C, (p.1 ^ 2 / 9 + p.2 ^ 2 / 3 = 1) :=
sorry

-- Maximum area of the triangle
noncomputable def curve_C := {p : ℝ × ℝ | p.1 ^ 2 / 9 + p.2 ^ 2 / 3 = 1}
noncomputable def line_l (m : ℝ) := {p : ℝ × ℝ | (√3) * p.1 + p.2 + m = 0}
def O : ℝ × ℝ := (0, 0)

theorem max_area_triangle (m : ℝ) (h : m^2 < 30) :
  ∃ B D : ℝ × ℝ, B ∈ curve_C ∧ D ∈ curve_C ∧ B ∈ line_l m ∧ D ∈ line_l m ∧
  (let area := (|m| / 2) * (|2 * (B.1 - D.1)| / 10) in
   area = 3sqrt(3) / 2) :=
sorry

end curve_trajectory_max_area_triangle_l680_680094


namespace periodic_sequence_l680_680443

variable (X : ℕ → ℝ)

theorem periodic_sequence (T : ℕ) (h : ∀ k : ℕ, ∃ n ≤ T, (X (k+1), X (k+2), ..., X (k+T)) = (X (n+1), X (n+2), ..., X (n+T))) : 
  ∃ P : ℕ, ∀ n : ℕ, X (n+P) = X n :=
sorry

end periodic_sequence_l680_680443


namespace ring_toss_total_l680_680266

theorem ring_toss_total (money_per_day : ℕ) (days : ℕ) (total_money : ℕ) 
(h1 : money_per_day = 140) (h2 : days = 3) : total_money = 420 :=
by
  sorry

end ring_toss_total_l680_680266


namespace total_miles_l680_680221

-- Define the variables and equations as given in the conditions
variables (a b c d e : ℝ)
axiom h1 : a + b = 36
axiom h2 : b + c + d = 45
axiom h3 : c + d + e = 45
axiom h4 : a + c + e = 38

-- The conjecture we aim to prove
theorem total_miles : a + b + c + d + e = 83 :=
sorry

end total_miles_l680_680221


namespace sum_c_n_l680_680824

-- Definitions of the sequences based on given conditions
def a_n (n : ℕ) : ℕ := 2 * n - 1
def b_n (n : ℕ) : ℕ := 2^n
def c_n (n : ℕ) : ℕ := a_n n * b_n n

-- Sum of the first n terms of sequence {c_n}
def T_n (n : ℕ) : ℕ := ∑ k in finset.range n, c_n (k + 1)

-- Theorem stating the required sum of the first n terms of the sequence {c_n}
theorem sum_c_n (n : ℕ) : T_n n = 6 + (2 * n - 3) * 2^(n + 1) := 
  by sorry

end sum_c_n_l680_680824


namespace height_of_burj_khalifa_l680_680233

theorem height_of_burj_khalifa 
  (height_eiffel: ℕ) 
  (difference_height: ℕ) 
  (height_eiffel = 324) 
  (difference_height = 506) : 
  height_eiffel + difference_height = 830 := 
sorry

end height_of_burj_khalifa_l680_680233


namespace gcd_poly_eq_l680_680946

noncomputable def gcd (a b : ℕ) : ℕ :=
if b = 0 then a else gcd b (a % b)

theorem gcd_poly_eq (n m : ℕ) : (X^n - 1).gcd (X^m - 1) = (X^(gcd n m) - 1) :=
by
  sorry

end gcd_poly_eq_l680_680946


namespace ratio_divides_AC_l680_680537

variables {A B C P Q S T : Type} [Field A]
variables [Module A P] [Module A Q] [Module A S] [Module A T]
variables [Inhabited P] [Inhabited Q] [Inhabited S] [Inhabited T]

def divides_in_ratio (X Y Z : P) (n m : ℕ) := 
  ∃ λ p q r : P, (p = n) ∧ (q = m) ∧ (n + m = 1)

axiom divides_AB : divides_in_ratio A B P 1 3
axiom divides_BC : divides_in_ratio B C Q 2 1
axiom midpoint_AP : S = (1 / 2) • A + (1 / 2) • P
axiom intersection_QS_AC : ∃ T, ∃ λ (line_QS : P), line_QS = Q ∧ λ (line_AC : P), line_AC = A ∧ λ (intersect : P), intersect = T

theorem ratio_divides_AC : ∃ (T : P), divides_in_ratio A C T 2 1 :=
by
  sorry

end ratio_divides_AC_l680_680537


namespace intersection_point_l680_680134

theorem intersection_point (
  l : ℝ → ℝ → Prop := λ x y, 2 * x + y = 10,
  point_on_l' : (ℝ × ℝ) := (-10, 0),
  perpendicular_condition : (ℝ → ℝ → Prop) → Prop :=
    λ f, ∃ (m' : ℝ), f = λ x y, y = m' * x + 5 ∧ m' = 1/2
) : 
  ∃ x y, l x y ∧ (∃ m', λ x' y', y' = m' * x' + 5 x y) ∧ (x, y) = (2, 6) :=
by 
  sorry

end intersection_point_l680_680134


namespace marble_problem_l680_680281

theorem marble_problem 
  (G R B W : ℕ) 
  (h_total : G + R + B + W = 84) 
  (h_green : (G : ℚ) / 84 = 1 / 7) 
  (h_red_blue : (R + B : ℚ) / 84 = 0.6071428571428572) : 
  (W : ℚ) / 84 = 1 / 4 := 
by 
  sorry

end marble_problem_l680_680281


namespace probability_X_lt_0_l680_680099

noncomputable def X_distribution (σ : ℝ) (hσ : σ > 0) : MeasureTheory.ProbabilityTheory.NormalDist := 
  MeasureTheory.ProbabilityTheory.NormalDist.mk 2 σ

theorem probability_X_lt_0 (σ : ℝ) (hσ : σ > 0) (h : MeasureTheory.ProbabilityTheory.cdf (X_distribution σ hσ) 4 = 0.8) :
  MeasureTheory.ProbabilityTheory.cdf (X_distribution σ hσ) 0 = 0.2 :=
by
  sorry

end probability_X_lt_0_l680_680099


namespace arithmetic_sequence_middle_term_l680_680287

theorem arithmetic_sequence_middle_term :
  ∃ y : ℤ, 2^2, y, 2^4 is_arithmetic_sequence ∧ y = 10 :=
by
  sorry

end arithmetic_sequence_middle_term_l680_680287


namespace cars_meeting_distance_l680_680378

theorem cars_meeting_distance (v : ℝ) (h1 : ¬(v = 0)):
  let speedA := 1.2 * v in
  let speedB := v in
  let total_distance := 8 * 2 in
  let meeting_distance := total_distance / (speedA - speedB) in
  let relative_speed := speedA + speedB in
  let total_time := total_distance / relative_speed in
  let distance_between_AB := relative_speed * total_time in
  distance_between_AB = 176 :=
by
  -- Proof will use the provided conditions and mathematical relationships
  sorry

end cars_meeting_distance_l680_680378


namespace optimal_selling_price_minimize_loss_l680_680577

theorem optimal_selling_price_minimize_loss 
  (C : ℝ) (h1 : 17 * C = 720 + 5 * C) 
  (h2 : ∀ x : ℝ, x * (1 - 0.1) = 720 * 0.9)
  (h3 : ∀ y : ℝ, y * (1 + 0.05) = 648 * 1.05)
  (selling_price : ℝ)
  (optimal_selling_price : selling_price = 60) :
  selling_price = C :=
by 
  sorry

end optimal_selling_price_minimize_loss_l680_680577


namespace ava_legs_count_l680_680016

-- Conditions:
-- There are a total of 9 animals in the farm.
-- There are only chickens and buffalos in the farm.
-- There are 5 chickens in the farm.

def total_animals : Nat := 9
def num_chickens : Nat := 5
def legs_per_chicken : Nat := 2
def legs_per_buffalo : Nat := 4

-- Proof statement: Ava counted 26 legs.
theorem ava_legs_count (num_buffalos : Nat) 
  (H1 : total_animals = num_chickens + num_buffalos) : 
  num_chickens * legs_per_chicken + num_buffalos * legs_per_buffalo = 26 :=
by 
  have H2 : num_buffalos = total_animals - num_chickens := by sorry
  sorry

end ava_legs_count_l680_680016


namespace packets_in_box_l680_680022

theorem packets_in_box 
  (coffees_per_day : ℕ) 
  (packets_per_coffee : ℕ) 
  (cost_per_box : ℝ) 
  (total_cost : ℝ) 
  (days : ℕ) 
  (P : ℕ) 
  (h_coffees_per_day : coffees_per_day = 2)
  (h_packets_per_coffee : packets_per_coffee = 1)
  (h_cost_per_box : cost_per_box = 4)
  (h_total_cost : total_cost = 24)
  (h_days : days = 90)
  : P = 30 := 
by
  sorry

end packets_in_box_l680_680022


namespace fido_yard_area_reach_l680_680059

theorem fido_yard_area_reach (s r : ℝ) (h1 : r = s / (2 * Real.sqrt 2)) (h2 : ∃ (a b : ℕ), (Real.pi * Real.sqrt a) / b = Real.pi * (r ^ 2) / (2 * s^2 * Real.sqrt 2) ) :
  ∃ (a b : ℕ), a * b = 64 :=
by
  sorry

end fido_yard_area_reach_l680_680059


namespace evaluate_expression_l680_680405

theorem evaluate_expression : (3 + 5 + 7) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 5 + 7) = 9 / 20 := 
by 
  sorry

end evaluate_expression_l680_680405


namespace increasing_log_function_interval_l680_680874

theorem increasing_log_function_interval (a : ℝ) :
  (∀ x > 1, (log (x^2 + a * x) / log 2) ≥ (log ((x + ϵ)^2 + a * (x + ϵ)) / log 2)) ↔ a ≥ -1 :=
by sorry

end increasing_log_function_interval_l680_680874


namespace g_neither_even_nor_odd_l680_680904

noncomputable def g (x : ℝ) : ℝ := 3 ^ (x^2 - 3) - |x| + Real.sin x

theorem g_neither_even_nor_odd : ∀ x : ℝ, g x ≠ g (-x) ∧ g x ≠ -g (-x) := 
by
  intro x
  sorry

end g_neither_even_nor_odd_l680_680904


namespace inequality_solution_subset_l680_680825

theorem inequality_solution_subset {x a : ℝ} : (∀ x, |x| > a * x + 1 → x ≤ 0) ↔ a ≥ 1 :=
by sorry

end inequality_solution_subset_l680_680825


namespace domain_of_f_interval_of_monotonic_increase_l680_680252

noncomputable def f (x : ℝ) : ℝ := Real.log (9 - x^2)

theorem domain_of_f : {x : ℝ | 9 - x^2 > 0} = set.Ioo (-3) 3 :=
by
  sorry

theorem interval_of_monotonic_increase : {x : ℝ | ∀ y ∈ set.Icc (-3) 0, x ≤ y → f x ≤ f y} = set.Icc (-3) 0 :=
by
  sorry

end domain_of_f_interval_of_monotonic_increase_l680_680252


namespace determine_a_minus_b_l680_680464

theorem determine_a_minus_b (a b : ℤ) 
  (h1 : 2009 * a + 2013 * b = 2021) 
  (h2 : 2011 * a + 2015 * b = 2023) : 
  a - b = -5 :=
sorry

end determine_a_minus_b_l680_680464


namespace sum_series_eq_l680_680731

theorem sum_series_eq :
  (∑ n from 3 to ∞, ∑ k from 1 to n - 2, k^2 / 3^(n+k)) = 405 / 20736 :=
by
  sorry

end sum_series_eq_l680_680731


namespace summation_proof_l680_680738

open BigOperators

theorem summation_proof :
  ∑ n in finset.range (∞).filter (λ n, n ≥ 3), ∑ k in finset.range (n - 2).filter (λ k, k ≥ 1), k^2 * (3:ℝ) ^ (- (n + k)) = 5 / 72 := 
by 
  sorry

end summation_proof_l680_680738


namespace moles_of_NaHCO3_needed_l680_680423

theorem moles_of_NaHCO3_needed 
  (HC2H3O2_moles: ℕ)
  (H2O_moles: ℕ)
  (NaHCO3_HC2H3O2_molar_ratio: ℕ)
  (reaction: NaHCO3_HC2H3O2_molar_ratio = 1 ∧ H2O_moles = 3) :
  ∃ NaHCO3_moles : ℕ, NaHCO3_moles = 3 :=
by
  sorry

end moles_of_NaHCO3_needed_l680_680423


namespace floor_sqrt_120_l680_680403

theorem floor_sqrt_120 :
  (∀ x : ℝ, 10^2 = 100 ∧ 11^2 = 121 ∧ 100 < 120 ∧ 120 < 121 → 
  (∃ y : ℕ, y = 10 ∧ floor (real.sqrt 120) = y)) :=
by
  assume h,
  sorry

end floor_sqrt_120_l680_680403


namespace floor_sqrt_120_l680_680398

theorem floor_sqrt_120 : (Real.floor (Real.sqrt 120)) = 10 :=
by
  have a := 10
  have b := 11
  have h1 : a < b := by norm_num
  have h2 : a^2 < 120 := by norm_num
  have h3 : 120 < b^2 := by norm_num
  sorry

end floor_sqrt_120_l680_680398


namespace water_leakage_l680_680183

theorem water_leakage (initial_quarts : ℚ) (remaining_gallons : ℚ)
  (conversion_rate : ℚ) (expected_leakage : ℚ) :
  initial_quarts = 4 ∧ remaining_gallons = 0.33 ∧ conversion_rate = 4 ∧ 
  expected_leakage = 2.68 →
  initial_quarts - remaining_gallons * conversion_rate = expected_leakage :=
by 
  sorry

end water_leakage_l680_680183


namespace average_salary_techs_l680_680890

noncomputable def total_salary := 20000
noncomputable def average_salary_all := 750
noncomputable def num_technicians := 5
noncomputable def average_salary_non_tech := 700
noncomputable def total_workers := 20

theorem average_salary_techs :
  (20000 - (num_technicians + average_salary_non_tech * (total_workers - num_technicians))) / num_technicians = 900 := by
  sorry

end average_salary_techs_l680_680890


namespace even_times_odd_is_even_l680_680391

theorem even_times_odd_is_even {a b : ℤ} (h₁ : ∃ k, a = 2 * k) (h₂ : ∃ j, b = 2 * j + 1) : ∃ m, a * b = 2 * m :=
by
  sorry

end even_times_odd_is_even_l680_680391


namespace simplified_expression_eq_l680_680601

noncomputable def simplify_expression : ℚ :=
  1 / (1 / (1 / 3)^1 + 1 / (1 / 3)^2 + 1 / (1 / 3)^3)

-- We need to prove that the simplifed expression is equal to 1 / 39
theorem simplified_expression_eq : simplify_expression = 1 / 39 :=
by sorry

end simplified_expression_eq_l680_680601


namespace survivor_quit_probability_l680_680999

noncomputable def probability_both_quit_same_tribe (n : ℕ) (tribe_size : ℕ) (quits : ℕ) : ℚ :=
  (2 * (nat.choose tribe_size quits : ℚ) / nat.choose n quits)

theorem survivor_quit_probability :
  let n := 16
  let tribe_size := 8
  let quits := 2
  probability_both_quit_same_tribe n tribe_size quits = 7 / 15 :=
by
  sorry

end survivor_quit_probability_l680_680999


namespace triangle_area_bounded_by_lines_l680_680069

theorem triangle_area_bounded_by_lines :
  let y1 (x : ℝ) := 4 * x + 1
  let y2 (x : ℝ) := (6 + 3 * x) / 2
  ∃ a b (x1 x2 : ℝ), y1 x1 = 0 ∧ y2 x2 = 0 ∧ x1 ≠ x2 ∧
  let base := abs(x2 - x1)
  let x_intersect := 4 / 5
  let y_intersect := 21 / 5
  base = 7/4 ∧ y_intersect > 0 ∧
  let height := y_intersect
  ∃ (area : ℝ), area = (1 / 2) * base * height ∧ area = 147 / 40 := 
by
  sorry

end triangle_area_bounded_by_lines_l680_680069


namespace positive_whole_numbers_cube_root_less_than_eight_l680_680854

theorem positive_whole_numbers_cube_root_less_than_eight : 
  { n : ℕ | n > 0 ∧ n < 512 }.card = 511 :=
by sorry

end positive_whole_numbers_cube_root_less_than_eight_l680_680854


namespace man_speed_in_still_water_l680_680703

theorem man_speed_in_still_water 
  (V_u : ℕ) (V_d : ℕ) 
  (hu : V_u = 34) 
  (hd : V_d = 48) : 
  V_s = (V_u + V_d) / 2 :=
by
  sorry

end man_speed_in_still_water_l680_680703


namespace area_of_overlapped_region_l680_680163

theorem area_of_overlapped_region (w1 w2 : ℝ) (θ : ℝ) (hw1 : w1 = 4) (hw2 : w2 = 5) (hθ : θ = π / 6) : 
  let A := w1 * w2 * Real.sin θ in
  A = 40 :=
by
  simp [hw1, hw2, hθ, Real.sin_pi_div_six]
  norm_num
  sorry

end area_of_overlapped_region_l680_680163


namespace max_radius_of_cylinder_in_crate_l680_680698

/-- Given a crate with dimensions 2 feet by 8 feet by 12 feet on the inside.
    A stone pillar in the shape of a right circular cylinder must fit into the crate 
    for shipping, such that the crate rests on its smallest side area 
    (2 feet by 8 feet) and the pillar stands upright.
    Prove that the radius of the pillar with the largest volume that could still 
    fit in the crate under these conditions is 1 foot. -/
theorem max_radius_of_cylinder_in_crate :
  ∃ (r : ℝ), 
  let w := 2, l := 8, h := 12 in
  (∀ (d : ℝ), d / 2 = r → d ≤ w) ∧ r = 1 :=
sorry

end max_radius_of_cylinder_in_crate_l680_680698


namespace percentage_dogs_movies_l680_680393

-- Definitions from conditions
def total_students : ℕ := 30
def students_preferring_dogs_videogames : ℕ := total_students / 2
def students_preferring_dogs : ℕ := 18
def students_preferring_dogs_movies : ℕ := students_preferring_dogs - students_preferring_dogs_videogames

-- Theorem statement
theorem percentage_dogs_movies : (students_preferring_dogs_movies * 100 / total_students) = 10 := by
  sorry

end percentage_dogs_movies_l680_680393


namespace find_polynomial_coefficients_l680_680426

-- Define the quadratic polynomial q(x) = ax^2 + bx + c
def polynomial (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions for polynomial
axiom condition1 (a b c : ℝ) : polynomial a b c (-2) = 9
axiom condition2 (a b c : ℝ) : polynomial a b c 1 = 2
axiom condition3 (a b c : ℝ) : polynomial a b c 3 = 10

-- Conjecture for the polynomial q(x)
theorem find_polynomial_coefficients : 
  ∃ (a b c : ℝ), 
    polynomial a b c (-2) = 9 ∧
    polynomial a b c 1 = 2 ∧
    polynomial a b c 3 = 10 ∧
    a = 19 / 15 ∧
    b = -2 / 15 ∧
    c = 13 / 15 :=
by {
  -- Placeholder proof
  sorry
}

end find_polynomial_coefficients_l680_680426


namespace number_of_birds_flew_up_correct_l680_680680

def initial_number_of_birds : ℕ := 29
def final_number_of_birds : ℕ := 42
def number_of_birds_flew_up : ℕ := final_number_of_birds - initial_number_of_birds

theorem number_of_birds_flew_up_correct :
  number_of_birds_flew_up = 13 := sorry

end number_of_birds_flew_up_correct_l680_680680


namespace original_cost_of_horse_l680_680704

theorem original_cost_of_horse (x : ℝ) (h : x - x^2 / 100 = 24) : x = 40 ∨ x = 60 := 
by 
  sorry

end original_cost_of_horse_l680_680704


namespace alternating_sum_excluding_divisibles_by_3_l680_680085

theorem alternating_sum_excluding_divisibles_by_3 (n : ℕ) (h : n % 3 = 0) :
  let seq := List.range (n - 1) in
  let filtered_seq := seq.filter (λ x, x % 3 ≠ 0) in
  let alternated_seq := List.map_with_index (λ i x, if (i / 2) % 2 = 0 then x else -x) filtered_seq in
  List.sum alternated_seq = n :=
sorry

end alternating_sum_excluding_divisibles_by_3_l680_680085


namespace mod_inverse_expression_l680_680278

theorem mod_inverse_expression (a b c d e : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h_bounds : 0 < a ∧ a < 12 ∧ 0 < b ∧ b < 12 ∧ 0 < c ∧ c < 12 ∧ 0 < d ∧ d < 12 ∧ 0 < e ∧ e < 12)
  (h_invertible : Nat.gcd a 12 = 1 ∧ Nat.gcd b 12 = 1 ∧ Nat.gcd c 12 = 1 ∧ Nat.gcd d 12 = 1 ∧ Nat.gcd e 12 = 1) :
  let product := a * b * c * d * e in
  let sum := a * b * c + a * b * d + a * b * e + a * c * d + a * c * e + a * d * e + b * c * d + b * c * e + b * d * e + c * d * e in
  (product⁻¹ * sum) % 12 = 9 :=
by
  sorry

end mod_inverse_expression_l680_680278


namespace number_of_whole_numbers_with_cube_roots_less_than_8_l680_680857

theorem number_of_whole_numbers_with_cube_roots_less_than_8 :
  ∃ (n : ℕ), (∀ (x : ℕ), (1 ≤ x ∧ x < 512) → x ≤ n) ∧ n = 511 := 
sorry

end number_of_whole_numbers_with_cube_roots_less_than_8_l680_680857


namespace rita_remaining_money_l680_680964

theorem rita_remaining_money :
  let dresses_cost := 5 * 20
  let pants_cost := 3 * 12
  let jackets_cost := 4 * 30
  let transport_cost := 5
  let total_expenses := dresses_cost + pants_cost + jackets_cost + transport_cost
  let initial_money := 400
  let remaining_money := initial_money - total_expenses
  remaining_money = 139 := 
by
  sorry

end rita_remaining_money_l680_680964


namespace g50_zero_solution_count_l680_680561

noncomputable def g₀ (x : ℝ) : ℝ :=
  if x < -150 then 2 * x + 300
  else if x < 150 then -4 * x
  else 2 * x - 300

noncomputable def g : ℕ → ℝ → ℝ
| 0, x => g₀ x
| (n+1), x => |g n x| - 3

theorem g50_zero_solution_count :
  set.count {x : ℝ | g 50 x = 0} = 49 :=
sorry

end g50_zero_solution_count_l680_680561


namespace solve_quadratic_eq_l680_680243

theorem solve_quadratic_eq {x : ℝ} :
  (x = 3 ∨ x = -1) ↔ x^2 - 2 * x - 3 = 0 :=
by
  sorry

end solve_quadratic_eq_l680_680243


namespace solve_quadratic_l680_680603

theorem solve_quadratic : ∃ x : ℚ, 3 * x^2 + 11 * x - 20 = 0 ∧ x > 0 ∧ x = 4 / 3 :=
by
  sorry

end solve_quadratic_l680_680603


namespace sum_distances_even_eq_odd_sum_tangents_even_eq_odd_l680_680030

-- Definitions for problem conditions
variable {n : ℕ} (A A1 A2 : Point)
variable (vertices : List Point) (circle : Circle)

-- Conditions:
-- 1. Circle inscribed with a regular (2n + 1)-gon A1 A2 ... A(2n+1)
def regular_polygon_inscribed_in_circle (circle : Circle) (vertices : List Point) : Prop := sorry

-- 2. A is an arbitrary point on the arc A1 A(2n+1)
def A_on_arc_A1_A2n1 (A A1 A2n1 : Point) (circle : Circle) : Prop := sorry

-- The regular (2n+1)-gon's vertices list
def vertices_list (vertices : List Point) (n : ℕ) : Prop :=
  vertices.length = 2 * n + 1

-- Define the problem statements as Lean propositions
-- Part a)
theorem sum_distances_even_eq_odd
  (circle : Circle) (vertices : List Point) (A A1 A2 : Point)
  (h1 : regular_polygon_inscribed_in_circle circle vertices)
  (h2 : A_on_arc_A1_A2n1 A A1 (List.get (vertices.length - 1) vertices) circle)
  (h3 : vertices_list vertices n) :
  (∑ k in Finset.range n, dist A (vertices.get (2 * k + 2))) =
  (∑ k in Finset.range n, dist A (vertices.get (2 * k + 1))) :=
sorry

-- Part b)
theorem sum_tangents_even_eq_odd
  (circle : Circle) (vertices : List Point) (A A1 A2 : Point)
  (h1 : regular_polygon_inscribed_in_circle circle vertices)
  (h2 : A_on_arc_A1_A2n1 A A1 (List.get (vertices.length - 1) vertices) circle)
  (h3 : vertices_list vertices n) :
  (∑ k in Finset.range n, tangent_length (circle_tangent_at vertices.get (2 * k + 2)) A) =
  (∑ k in Finset.range n, tangent_length (circle_tangent_at vertices.get (2 * k + 1)) A) :=
sorry

end sum_distances_even_eq_odd_sum_tangents_even_eq_odd_l680_680030


namespace no_common_points_line_circle_l680_680521

theorem no_common_points_line_circle (m : ℝ) :
  (∃ x y : ℝ, (3 * x + 4 * y + m = 0) ∧ ((x + 1) ^ 2 + (y - 2) ^ 2 = 1)) ↔
  m ∈ (-∞:ℝ, -10) ∪ (0:ℝ, ∞:ℝ) :=
begin
  sorry
end

end no_common_points_line_circle_l680_680521


namespace correct_mark_proof_l680_680335

-- Define the conditions
def wrong_mark := 85
def increase_in_average : ℝ := 0.5
def number_of_pupils : ℕ := 104

-- Define the correct mark to be proven
noncomputable def correct_mark : ℕ := 33

-- Statement to be proven
theorem correct_mark_proof (x : ℝ) :
  (wrong_mark - x) / number_of_pupils = increase_in_average → x = correct_mark :=
by
  sorry

end correct_mark_proof_l680_680335


namespace sum_first_10_terms_b_l680_680267

theorem sum_first_10_terms_b (a b : ℕ → ℝ) (h_b : ∀ n, b n = 1 / (n^2 + 3 * n + 2)) :
  ∑ n in Finset.range 10, b n = 5 / 12 := by
sorry

end sum_first_10_terms_b_l680_680267


namespace perpendicular_lines_l680_680811

theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, 2 * x + y + 1 = 0) ∧ (∀ x y : ℝ, x + a * y + 3 = 0) ∧ (∀ A1 B1 A2 B2 : ℝ, A1 * A2 + B1 * B2 = 0) →
  a = -2 :=
by
  intros h
  sorry

end perpendicular_lines_l680_680811


namespace area_of_triangle_ABC_l680_680473

def f (x : ℝ) : ℝ := 2 * sin (π * x + π / 3)
def g (x : ℝ) : ℝ := 2 * cos (π * x + π / 3)
def intersection_points : List (ℝ × ℝ) := [(-13/12, -sqrt 2), (-1/12, sqrt 2), (11/12, -sqrt 2)]

theorem area_of_triangle_ABC :
  let A := (-13/12, -sqrt 2)
  let B := (-1/12, sqrt 2)
  let C := (11/12, -sqrt2)
  (1 / 2) * abs ((fst B - fst A) * (snd C - snd A) - (fst C - fst A) * (snd B - snd A)) = 2 * sqrt 2 :=
sorry

end area_of_triangle_ABC_l680_680473


namespace complete_square_form_l680_680662

theorem complete_square_form (x : ℝ) (a : ℝ) 
  (h : x^2 - 2 * x - 4 = 0) : (x - 1)^2 = a ↔ a = 5 :=
by
  sorry

end complete_square_form_l680_680662


namespace time_spent_on_type_A_problems_l680_680519

theorem time_spent_on_type_A_problems
  (exam_duration : ℕ := 180)
  (total_questions : ℕ := 200)
  (num_type_A : ℕ := 25)
  (type_A_multiplier : ℕ := 2) : 
  (∃ (time_on_A : ℕ), 
    let time_on_B : ℕ := 180 // 225 in
    let total_time_A : ℕ := num_type_A * type_A_multiplier * time_on_B in
    total_time_A = 40) :=
begin
  sorry
end

end time_spent_on_type_A_problems_l680_680519


namespace not_divisible_by_n_l680_680585

theorem not_divisible_by_n (n : ℕ) (h : n > 1) : ¬n ∣ 2^n - 1 :=
by
  -- proof to be filled in
  sorry

end not_divisible_by_n_l680_680585


namespace v_n_sequence_periodicity_l680_680383

theorem v_n_sequence_periodicity (b : ℝ) (hb : b > 0) : 
  let v : ℕ → ℝ := λ n, 
    match n with
    | 0       => b
    | n + 1   => -2 / (v n + 2)
  in 
    v 16 = -2 * (b + 1) / (b + 3) := 
sorry

end v_n_sequence_periodicity_l680_680383


namespace axis_of_symmetry_l680_680987

theorem axis_of_symmetry (x : ℝ) : 
  let y := cos (2 * x) - sin (2 * x)
  (∃ k : ℤ, x = k * π / 2 - π / 8) → 
  x = -π / 8 :=
by
  sorry

end axis_of_symmetry_l680_680987


namespace find_m_n_l680_680531

-- Definitions based on the problem statements
variables (OA OB OC : ℝ)
variables (AOCtan BOC : ℝ)
variables (m n : ℝ)

-- Conditions
def conditions :=
  (∥OA∥ = 2) ∧
  (∥OB∥ = 2) ∧
  (∥OC∥ = 2) ∧
  (AOCtan = 3) ∧
  (BOC = 60)

-- Theorem statement
theorem find_m_n (h : conditions OA OB OC AOCtan BOC) : 
  m = 7/6 ∧ n = 8/6 :=
sorry

end find_m_n_l680_680531


namespace min_of_fraction_sum_min_value_when_equal_l680_680929

theorem min_of_fraction_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / b + b / c + c / a + a / c) ≥ 4 :=
begin
  sorry
end

theorem min_value_when_equal (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_eq : a = b ∧ b = c) :
  (a / b + b / c + c / a + a / c) = 4 :=
begin
  sorry
end

end min_of_fraction_sum_min_value_when_equal_l680_680929


namespace prime_sum_probability_is_5_over_6_l680_680636

-- Define the sets of outcomes for the two spinners
def SpinnerA := {2, 4, 6}
def SpinnerB := {1, 3}

-- Define the set of prime numbers
def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

-- Define a function to compute the probability of obtaining a prime sum
noncomputable def prime_sum_probability : ℚ :=
  let all_sums := {sum | a ∈ SpinnerA, b ∈ SpinnerB, sum = a + b}
  let prime_sums := {sum | sum ∈ all_sums ∧ is_prime sum}
  (prime_sums.to_finset.card / all_sums.to_finset.card : ℚ)

-- Theorem statement
theorem prime_sum_probability_is_5_over_6 :
  prime_sum_probability = 5 / 6 :=
sorry

end prime_sum_probability_is_5_over_6_l680_680636


namespace double_sum_equality_l680_680727

theorem double_sum_equality : 
  (∑ n in (rangeFrom 3), ∑ k in finset.range (n-2) \ k -> k + 2, (k^2 : ℝ) / 3^(n+k)) = 729 / 17576 :=
sorry

end double_sum_equality_l680_680727


namespace solve_x_l680_680763

noncomputable def proof_problem (x : ℝ) : Prop :=
  (x ^ 2 - 7 * x + 6) / (x - 1) + (2 * x ^ 2 + 7 * x - 6) / (2 * x - 1) = 1

theorem solve_x : ∀ x : ℝ, x ≠ 1 ∧ x ≠ 1 / 2 → proof_problem x → x = 1 / 2 :=
by
  intro x h1 h2
  intro hx_eq
  sorry

end solve_x_l680_680763


namespace least_element_in_T_l680_680923

open Set

noncomputable def T : Set ℕ := {4, 6, 7, 9, 10, 11, 13}

theorem least_element_in_T : ∀ T : Set ℕ, (T ⊆ (finset.range 14).to_set) ∧ (∀ a b ∈ T, a ≠ b → ¬ a ∣ b) → ∃ x ∈ T, x = 4 :=
by
  intro T h
  have h1 := h.1
  have h2 := h.2
  sorry

end least_element_in_T_l680_680923


namespace sum_of_first_5_terms_l680_680176

-- Define the geometric sequence and its sum
def geom_sequence (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) :=
  ∀ n, S n = a 0 * (1 - q^(n + 1)) / (1 - q)

-- State the conditions as def
def conditions (a S : ℕ → ℝ) :=
  a 0 ≠ 0 ∧
  geom_sequence a 2 S ∧
  S 2 = 3 ∧ 
  S 3 = 7

-- Finally state the theorem
theorem sum_of_first_5_terms (a S : ℕ → ℝ) (h : conditions a S) : 
  S 5 = 31 :=
begin
  sorry
end

end sum_of_first_5_terms_l680_680176


namespace probability_of_no_adjacent_stands_is_correct_l680_680053

noncomputable def number_of_arrangements : ℕ → ℕ
| 2 := 3
| 3 := 4
| (n+1) := number_of_arrangements n + number_of_arrangements (n-1)

def total_outcomes : ℕ := 2^8

def favorable_outcomes : ℕ := number_of_arrangements 8

def probability := (favorable_outcomes : ℚ) / total_outcomes

theorem probability_of_no_adjacent_stands_is_correct :
  probability = 47 / 256 :=
by sorry

end probability_of_no_adjacent_stands_is_correct_l680_680053


namespace solve_quadratic_eq_l680_680242

theorem solve_quadratic_eq {x : ℝ} :
  (x = 3 ∨ x = -1) ↔ x^2 - 2 * x - 3 = 0 :=
by
  sorry

end solve_quadratic_eq_l680_680242


namespace rook_placements_5x5_l680_680870

/-- The number of ways to place five distinct rooks on a 
  5x5 chess board such that each column and row of the 
  board contains exactly one rook is 120. -/
theorem rook_placements_5x5 : 
  ∃! (f : Fin 5 → Fin 5), Function.Bijective f :=
by
  sorry

end rook_placements_5x5_l680_680870


namespace problem_statement_l680_680563

def f (n : ℕ) (p : ℕ) : ℕ :=
  (Finset.range (p - 1)).sum (λ k, (k + 1) * n ^ k)

theorem problem_statement (p m n : ℕ) (hp : prime p) (hodd : p % 2 = 1)
  (h_eq : f m p % p = f n p % p) : m % p = n % p := sorry

end problem_statement_l680_680563


namespace walking_time_l680_680543

theorem walking_time (r s : ℕ) (h₁ : r + s = 50) (h₂ : 2 * s = 30) : 2 * r = 70 :=
by
  sorry

end walking_time_l680_680543


namespace determine_y_l680_680670

open List

-- Define List I and List II
def listI (y : ℤ) : List ℤ := [y, 2, 4, 7, 10, 11]
def listII : List ℤ := [3, 3, 4, 6, 7, 10]

-- Define median function for a list of integers
def median (l : List ℤ) : ℤ :=
  let sorted := sort l
  let n := length sorted
  if even n then (nth sorted (n / 2 - 1)).getD 0 + (nth sorted (n / 2)).getD 0 / 2 else (nth sorted (n / 2)).getD 0

-- Define mode function for a list of integers
def mode (l : List ℤ) : ℤ :=
  let freqs := (l.groupBy (· = ·)).map (λ group => (group.length, head group).getD (0, 0))
  fst (argmax prod.fst freqs)

-- Define the problem statement in Lean 4
theorem determine_y (y : ℤ) : median (listI y) = median listII + mode listII → y = 9 := by
  sorry

end determine_y_l680_680670


namespace balls_in_boxes_l680_680589

theorem balls_in_boxes :
  (∃ (x1 x2 x3 : ℕ), x1 + x2 + x3 = 10 ∧ x1 ≥ 1 ∧ x2 ≥ 2 ∧ x3 ≥ 3) →
  15 = nat.choose ((10 - 1 - 2 - 3) + 3 - 1) (3 - 1) :=
by
  intros h
  sorry

end balls_in_boxes_l680_680589


namespace negated_proposition_l680_680449

open Nat

def is_prime (n : ℕ) : Prop := Prime n

def original_proposition : Prop :=
  ∀ n ∈ Nat, ¬ is_prime (2^n - 2)

theorem negated_proposition :
  (∃ n ∈ Nat, is_prime (2^n - 2)) ↔ ¬ original_proposition := by
  sorry

end negated_proposition_l680_680449


namespace area_of_hexagon_DEFGHI_l680_680057

-- Condition: Equilateral ΔABC with side length 2 and squares ABDE, BCHI, CAFG

/-- Hexagon formed by extending squares on the sides of an equilateral triangle with side length 2. -/
theorem area_of_hexagon_DEFGHI (a b c d e f g h i : ℝ) :
  let triangle_side := 2 in
  let altitude_length := triangle_side * sqrt 3 / 2 in
  let square_area := triangle_side^2 in
  let jkl_side := triangle_side + 2 * altitude_length in
  let jkl_area := (sqrt 3 / 4) * jkl_side^2 in
  let hexagon_area := jkl_area - 3 * square_area in
  hexagon_area = (sqrt 3 / 4) * (16 + 8 * sqrt 3) - 12 :=
by
  sorry

end area_of_hexagon_DEFGHI_l680_680057


namespace curve_tangent_parallel_line_l680_680126

theorem curve_tangent_parallel_line {a : ℝ} :
  (∃ (a : ℝ), let curve := fun (x : ℝ) => a * x^2 in
              let tangent_slope := (deriv curve) 1 in
              let line := 2 in
              tangent_slope = line) ↔ a = 1 :=
by
  sorry

end curve_tangent_parallel_line_l680_680126


namespace find_constants_minimum_distance_and_coordinates_range_m_l680_680470

-- Define the function f
def f (x a b : ℝ) : ℝ := (a * x) / (x + b)

-- Proving the values of a and b
theorem find_constants (a b : ℝ) (h1 : f 1 a b = 1) (h2 : f (-2) a b = 4) :
  a = 2 ∧ b = 1 :=
by sorry

-- Proving the minimum value of |AP| and coordinates of P
theorem minimum_distance_and_coordinates (P : ℝ × ℝ) (h1 : P = ⟨-ℝ.sqrt 2 - 1, 2 + ℝ.sqrt 2⟩) :
  let A : ℝ × ℝ := (1, 0)
  let x := P.1
  let y := P.2
  x < -1 → f x 2 1 = y →
  (min_val: |P - A| = 2 * ℝ.sqrt 2 + 2) :=
by sorry

-- Proving the range of m
theorem range_m (m : ℝ) :
  (∀ x, 0 < x ∧ x < 1 → f x 2 1 ≤ (2 * m * real.log x) / (x + 1)) →
  m ≤ -1 / ℝ.exp 1 :=
by sorry

end find_constants_minimum_distance_and_coordinates_range_m_l680_680470


namespace label_beads_coprime_l680_680983

-- Define the conditions of the problem
variable (n : ℕ) (odd_nat : n % 2 = 1) (n_ge_1 : n ≥ 1)
def beads_A := 14
def beads_B := 19
def sequence := {i : ℕ | n ≤ i ∧ i ≤ n + 32}

-- Statement to be proved in Lean
theorem label_beads_coprime : ∃ (label : ℕ → ℕ), 
  ∀ (x : ℕ), x ∈ sequence → 
  (∀ (a b : ℕ), a ≠ b → label a ≠ label b) ∧ 
  (∀ (a b : ℕ), (a, b) ∈ (sequence × sequence) → 
    (a = b + 1 ∨ a + 1 = b) → 
    Nat.gcd (label a) (label b) = 1) :=
by
  sorry

end label_beads_coprime_l680_680983


namespace two_b_is_16667_percent_of_a_l680_680605

theorem two_b_is_16667_percent_of_a {a b : ℝ} (h : a = 1.2 * b) : (2 * b / a) = 5 / 3 := by
  sorry

end two_b_is_16667_percent_of_a_l680_680605


namespace negated_proposition_l680_680450

open Nat

def is_prime (n : ℕ) : Prop := Prime n

def original_proposition : Prop :=
  ∀ n ∈ Nat, ¬ is_prime (2^n - 2)

theorem negated_proposition :
  (∃ n ∈ Nat, is_prime (2^n - 2)) ↔ ¬ original_proposition := by
  sorry

end negated_proposition_l680_680450


namespace mean_temperature_is_correct_l680_680249

def temperatures : List ℝ :=
  [-7, -4, -4, -5, 1, 3, 2, 4]

theorem mean_temperature_is_correct :
  (List.sum temperatures / (temperatures.length : ℝ)) = -1.25 :=
by
  sorry

end mean_temperature_is_correct_l680_680249


namespace max_diff_of_seven_distinct_primes_l680_680191

theorem max_diff_of_seven_distinct_primes :
  ∃ (a b c : ℕ), 
    prime a ∧ prime b ∧ prime c ∧ 
    prime (a + b - c) ∧ prime (a + c - b) ∧ prime (b + c - a) ∧ prime (a + b + c) ∧
    a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
    a + b ≠ a + c - b ∧ a + b ≠ b + c - a ∧ a + b ≠ a + b + c ∧
    a + c - b ≠ b + c - a ∧ a + c - b ≠ a + b + c ∧
    b + c - a ≠ a + b + c ∧
    (∀ x ∈ {a, b, c, a + b - c, a + c - b, b + c - a, a + b + c}.to_finset, prime x) ∧
    (
      let d := 2 * c in
      d = 1594
    ) :=
begin
  sorry
end

end max_diff_of_seven_distinct_primes_l680_680191


namespace oliver_workout_hours_l680_680576

variable (x : ℕ)

theorem oliver_workout_hours :
  (x + (x - 2) + 2 * x + 2 * (x - 2) = 18) → x = 4 :=
by
  sorry

end oliver_workout_hours_l680_680576


namespace expression_for_f_range_of_a_l680_680118

noncomputable def f : ℝ → ℝ
| x := if x ≥ 0 then 1 - 3^x else -1 + 3^(-x)

theorem expression_for_f (x : ℝ) :
  (x ≥ 0 → f x = 1 - 3^x) ∧ (x < 0 → f x = -1 + 3^(-x)) :=
by {
    sorry
}

theorem range_of_a {a : ℝ} :
  (∀ x : ℝ, (2 ≤ x ∧ x ≤ 8) → f ((log 2 x) ^ 2) + f (5 - a * log 2 x) ≥ 0) →
  a ≥ 6 :=
by {
    sorry
}

end expression_for_f_range_of_a_l680_680118


namespace quadrilateral_sum_of_squares_l680_680961

theorem quadrilateral_sum_of_squares
  (a b c d m n t : ℝ) : 
  a^2 + b^2 + c^2 + d^2 = m^2 + n^2 + 4 * t^2 :=
sorry

end quadrilateral_sum_of_squares_l680_680961


namespace money_remaining_l680_680949

noncomputable def cost_of_drink (p : ℝ) := p
noncomputable def cost_of_medium_pizza (p : ℝ) := 3 * p
noncomputable def cost_of_large_pizza (p : ℝ) := 4 * p
noncomputable def initial_money := 30

def total_cost (p : ℝ) := 5 * cost_of_drink p + cost_of_medium_pizza p + cost_of_large_pizza p

theorem money_remaining (p : ℝ) : initial_money - total_cost p = 30 - 12 * p :=
by
  sorry

end money_remaining_l680_680949


namespace annuity_problem_l680_680331

noncomputable def calc_annuity (A p : ℝ) (t N : ℕ) : ℝ :=
  A * (Real.exp (-p * t)) * ((Real.exp (p * N) - 1) / (Real.exp p - 1))

theorem annuity_problem :
  let p := 0.04
  let e := Real.exp 1 
  let PV_original := calc_annuity 1000 p 5 5 + calc_annuity 1200 p 10 5 + calc_annuity 1400 p 15 5
  let PV_modified := calc_annuity 1400 p 5 5 + calc_annuity 1200 p 10 5 + calc_annuity x p 15 5
  in PV_original = PV_modified :=
by
  sorry

end annuity_problem_l680_680331
