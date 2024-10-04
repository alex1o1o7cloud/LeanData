import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Perm
import Mathlib.Data.Fintype
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Factorial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Data.Set.Function
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.MeasureTheory.MeasurableSpace
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.SetTheory.Cardinal.Basic
import Mathlib.SetTheory.Cardinal.Finite
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Instances.Real

namespace average_rate_of_change_correct_l136_136859

def f (x : ℝ) : ℝ := 2 * x + 1

theorem average_rate_of_change_correct :
  (f 2 - f 1) / (2 - 1) = 2 :=
by
  sorry

end average_rate_of_change_correct_l136_136859


namespace min_value_of_a_l136_136655

noncomputable def min_a_monotone_increasing : ℝ :=
  let f : ℝ → ℝ := λ x a, a * exp x - log x in 
  let a_min : ℝ := exp (-1) in
  a_min

theorem min_value_of_a 
  {f : ℝ → ℝ} 
  {a : ℝ}
  (mono_incr : ∀ x ∈ Ioo 1 2, 0 ≤ deriv (λ x, a * exp x - log x) x) :
  a ≥ exp (-1) :=
begin
  sorry
end

end min_value_of_a_l136_136655


namespace min_value_frac_sum_l136_136352

theorem min_value_frac_sum
  (a b : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : ∃ λ, (a - 1 = λ * (-b-1)) ∧ (1 = 2 * λ)) :
  (frac_sum : ℝ) :=
  frac_sum = 8 :=
sorry

end min_value_frac_sum_l136_136352


namespace staff_price_l136_136121

-- Definitions of the conditions
def original_price (d : ℝ) := d
def discount_rate_initial : ℝ := 0.25
def staff_discount_rate : ℝ := 0.20

-- Statement of the theorem
theorem staff_price (d : ℝ) :
  let price_after_initial_discount := d * (1 - discount_rate_initial),
      final_price := price_after_initial_discount * (1 - staff_discount_rate)
  in final_price = 0.60 * d :=
by
  sorry

end staff_price_l136_136121


namespace average_price_per_book_l136_136013

theorem average_price_per_book :
  let
    price1 := 1150,
    price2 := 920,
    price3 := 630 - 0.10 * 630,
    price4 := 1080 - 0.05 * 1080,
    price5 := 350,
    total_price := price1 + price2 + price3 + price4 + price5,
    num_books := 65 + 50 + 30 + 45 + 10,
    average_price := total_price / num_books
  in average_price = 20.065 :=
by
  let price1 := 1150
  let price2 := 920
  let price3 := 630 - 0.10 * 630
  let price4 := 1080 - 0.05 * 1080
  let price5 := 350
  let total_price := price1 + price2 + price3 + price4 + price5
  let num_books := 65 + 50 + 30 + 45 + 10
  let average_price := total_price / num_books
  have h : average_price = 20.065 := sorry
  exact h

end average_price_per_book_l136_136013


namespace slices_per_pizza_l136_136911

theorem slices_per_pizza (num_pizzas num_slices : ℕ) (h1 : num_pizzas = 17) (h2 : num_slices = 68) :
  (num_slices / num_pizzas) = 4 :=
by
  sorry

end slices_per_pizza_l136_136911


namespace surface_area_DABC_l136_136809

structure Triangle (α : Type*) :=
(A B C : α)

structure Pyramid (α : Type*) :=
(A B C D : α)

def edge_length (α : Type*) [has_dist : Π x y : α, ℝ] (x y : α) : ℝ := 
  dist x y

variables {α : Type*} [metric_space α]
variables (ABC : Triangle α) (D : α)
variables (h₁ : ∀ x ∈ {ABC.A, ABC.B, ABC.C}, x ≠ D)
variables (h₂ : ∀ s ∈ ({ABC.A, ABC.B, ABC.C} : set α), ∀ t ∈ ({ABC.A, ABC.B, ABC.C} : set α), edge_length α s t = 20 ∨ edge_length α s t = 45)
variables (h₃ : ¬ ∃ (A B C ∈ {ABC.A, ABC.B, ABC.C}),  edge_length α A B = edge_length α B C ∧ edge_length α B C = edge_length α C A)

theorem surface_area_DABC (h₄ : ∀ (A B ∈ {ABC.A, ABC.B, ABC.C}), (edge_length α A B = 20 ∧ edge_length α A D = 45 ∧ edge_length α B D = 45)
  ∨ (edge_length α A B = 45 ∧ edge_length α A D = 45 ∧ edge_length α B D = 20)) :
  ∃ area, area = 40 * real.sqrt 1925 :=
begin
  sorry
end

end surface_area_DABC_l136_136809


namespace angle_C_in_triangle_l136_136716

theorem angle_C_in_triangle (A B C : ℝ) (a b c : ℝ) (k : ℝ) (h_k_pos : k > 0) :
  sin A / sin B = 7 / 8 ∧ sin A / sin C = 7 / 13 ∧ 
  sin B / sin C = 8 / 13 → C = 120 :=
by
  sorry

end angle_C_in_triangle_l136_136716


namespace area_of_ABC_l136_136616

theorem area_of_ABC (a : ℝ) (h : a > 0) : 
  let s := sqrt 2 * a in   -- side of the equilateral projection triangle
  let original_area := sqrt 6 * a^2 in  -- area we need to prove
  (equilateral : ∀ (x y: ℝ), x = y -> abs (x - y) = 0) ->
  area_of_ABC: s → original_area = sqrt 6 * a^2 :=
sorry

end area_of_ABC_l136_136616


namespace odd_count_valid_numbers_l136_136307

open Nat

def is_valid_number (n : ℕ) : Prop :=
  (n.digits 10).length = 64 ∧
  (∀ d ∈ n.digits 10, d ≠ 0) ∧
  n % 101 = 0

theorem odd_count_valid_numbers :
  ∃ count : ℕ, (count % 2 = 1) ∧ (count = (∑ n in Finset.range (10^64), if is_valid_number n then 1 else 0)) :=
sorry

end odd_count_valid_numbers_l136_136307


namespace combined_length_of_snakes_in_inches_l136_136523

theorem combined_length_of_snakes_in_inches :
  let snake1 := 2.4 * 12
  let snake2 := 16.2
  let snake3 := 10.75
  let snake4 := 50.5 / 2.54
  let snake5 := 0.8 * 100 / 2.54
  let snake6 := 120.35 / 2.54
  let snake7 := 1.35 * 36
  let total_length := snake1 + snake2 + snake3 + snake4 + snake5 + snake6 + snake7
  abs (total_length - 203.11) < 0.01 :=
by
  let snake1 := 2.4 * 12
  let snake2 := 16.2
  let snake3 := 10.75
  let snake4 := 50.5 / 2.54
  let snake5 := 0.8 * 100 / 2.54
  let snake6 := 120.35 / 2.54
  let snake7 := 1.35 * 36
  let total_length := snake1 + snake2 + snake3 + snake4 + snake5 + snake6 + snake7
  have : abs (total_length - 203.11) < 0.01 := by sorry
  exact this

end combined_length_of_snakes_in_inches_l136_136523


namespace triangle_angle_B_l136_136556

theorem triangle_angle_B (A B C H : Type) [Triangle A B C] (CH_half_AB : height C H = (1/2) * side A B) (angle_A_75 : angle A = 75) :
  angle B = 30 := by sorry

end triangle_angle_B_l136_136556


namespace convert_25_to_binary_l136_136074

-- The function to convert a decimal number to binary
def binary_conversion : ℕ → string
| 0       := "0"
| 1       := "1"
| n       := let rec aux (n : ℕ) (acc : string) : string :=
                 if n = 0 then acc
                 else aux (n / 2) (to_string (n % 2) ++ acc)
             in aux n ""

-- The theorem to prove
theorem convert_25_to_binary : binary_conversion 25 = "11001" :=
by
  -- conversion method
  sorry

end convert_25_to_binary_l136_136074


namespace cos_sin_probability_correct_l136_136509

noncomputable def cos_sin_probability : ℚ :=
  let numerator := 41
  let denominator := 6400
  (numerator : ℚ) / (denominator : ℚ)

theorem cos_sin_probability_correct (x y : ℝ) (h : cos (sin x) = cos (sin y)) (hx : -20 * π ≤ x ∧ x ≤ 20 * π) (hy : -20 * π ≤ y ∧ y ≤ 20 * π) :
  let pairs := (λ (x y : ℝ), x = y)
  (pairs (x, y) = pairs (cos_sin_probability, cos_sin_probability)) :=
sorry

end cos_sin_probability_correct_l136_136509


namespace minimum_value_of_a_l136_136670

theorem minimum_value_of_a (a : ℝ) (h : ∀ x ∈ set.Ioo 1 2, ae^x - 1/x ≥ 0) : a ≥ e⁻¹ :=
sorry

end minimum_value_of_a_l136_136670


namespace general_term_eq_2n_T_2016_value_l136_136589

noncomputable def a_n (n : ℕ) : ℕ :=
  2 * n

noncomputable def S_n (n : ℕ) : ℕ :=
  n * (n + 1)

noncomputable def T_n (n : ℕ) : ℝ :=
  ∑ i in range n, if 2 * i + 1 < n then 1 / (real.sqrt (2 * i + 1 - 1) + real.sqrt (2 * i + 1 + 1)) else 1 / (real.sqrt (i + 1 - 1) + real.sqrt (i + 1 + 1)) -- Note: This assumes a misinterpretation, corrected from the context

theorem general_term_eq_2n (n : ℕ) : 
  S_n n - (a_n n) = n^2 - n :=
sorry

theorem T_2016_value :
  T_n 2016 = 6 * real.sqrt 14 - (505 / 2018) :=
sorry

end general_term_eq_2n_T_2016_value_l136_136589


namespace gcd_1911_1183_l136_136040

theorem gcd_1911_1183 : gcd 1911 1183 = 91 :=
by sorry

end gcd_1911_1183_l136_136040


namespace compare_2_5_sqrt_6_l136_136526

theorem compare_2_5_sqrt_6 : 2.5 > Real.sqrt 6 := by
  sorry

end compare_2_5_sqrt_6_l136_136526


namespace force_on_dam_l136_136157

noncomputable def calculate_force (ρ g a b h : ℝ) :=
  ρ * g * h^2 * (b / 2 - (b - a) / 3)

theorem force_on_dam :
  let ρ := 1000
  let g := 10
  let a := 6.0
  let b := 9.6
  let h := 4.0
  calculate_force ρ g a b h = 576000 :=
by sorry

end force_on_dam_l136_136157


namespace remainder_when_101_divided_by_7_is_3_l136_136002

theorem remainder_when_101_divided_by_7_is_3
    (A : ℤ)
    (h : 9 * A + 1 = 10 * A - 100) : A % 7 = 3 := by
  -- Mathematical steps are omitted as instructed
  sorry

end remainder_when_101_divided_by_7_is_3_l136_136002


namespace length_of_FD_l136_136299

-- Define the conditions
def is_square (ABCD : ℝ) (side_length : ℝ) : Prop :=
  side_length = 8 ∧ ABCD = 4 * side_length

def point_E (x : ℝ) : Prop :=
  x = 8 / 3

def point_F (CD : ℝ) (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 8

-- State the theorem
theorem length_of_FD (side_length : ℝ) (x : ℝ) (CD ED FD : ℝ) :
  is_square 4 side_length → 
  point_E ED → 
  point_F CD x → 
  FD = 20 / 9 :=
by
  sorry

end length_of_FD_l136_136299


namespace intersection_height_correct_l136_136711

noncomputable def intersection_height 
  (height_pole_1 height_pole_2 distance : ℝ) : ℝ := 
  let slope_1 := -(height_pole_1 / distance)
  let slope_2 := height_pole_2 / distance
  let y_intercept_1 := height_pole_1
  let y_intercept_2 := 0
  let x_intersection := height_pole_1 / (slope_2 - slope_1)
  let y_intersection := slope_2 * x_intersection + y_intercept_2
  y_intersection

theorem intersection_height_correct 
  : intersection_height 30 90 150 = 22.5 := 
by sorry

end intersection_height_correct_l136_136711


namespace least_positive_integer_l136_136914

theorem least_positive_integer (n : ℕ) : 
  (530 + n) % 4 = 0 → n = 2 :=
by {
  sorry
}

end least_positive_integer_l136_136914


namespace line_passes_through_fixed_point_l136_136218

def parabola_equation (x y p : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ p > 0

def point_distance (x y p : ℝ) : Prop :=
  2 + p / 2 = 4

def intersection_property (ME NE : ℝ) : Prop :=
  ME * NE = 8

theorem line_passes_through_fixed_point (x y p ME NE : ℝ) (hx : parabola_equation x y p)
  (hdist : point_distance x y p) (hinter : intersection_property ME NE) : 
  x = 4 ∧ y = 0 :=
begin
  -- Lean proof would go here
  sorry
end

end line_passes_through_fixed_point_l136_136218


namespace megan_music_collection_l136_136361

def albums : List ℕ := [7, 10, 5, 12, 14, 6, 8, 16]
def removedAlbums : List ℕ := [List.maximum albums, List.minimum albums]
def remainingAlbums : List ℕ := (albums.filter fun x => x ≠ 5).filter fun x => x ≠ 16
def costPerSong := 1.50
def discountAlbumSongs := 7
def discountRate := 0.10

-- Calculate the total number of remaining songs and the new cost considering discounts
theorem megan_music_collection (totalSongs : ℕ) (newTotalCost : ℝ) :
  totalSongs = (remainingAlbums.foldr (· + ·) 0) ∧
  newTotalCost = (remainingAlbums.foldr (· + ·) 0) * costPerSong - (discountAlbumSongs * costPerSong * discountRate) :=
  sorry

end megan_music_collection_l136_136361


namespace correct_second_number_l136_136383

theorem correct_second_number : 
  (∀ sum_with_errors : ℝ, sum_with_errors = 402 → 
  ∀ correct_sum : ℝ, correct_sum = 403 →
  ∀ first_error_difference : ℝ, first_error_difference = 19 →
  ∀ second_error_difference : ℝ, second_error_difference = 13 →
  ∃ correct_value : ℝ, correct_value = 33) :=
by {
  intros sum_with_errors Hsum correct_sum Hcorrect_sum first_error_difference Hfirst_error second_error_difference Hsecond_error,
  use 33,
  sorry
}

end correct_second_number_l136_136383


namespace sum_floor_diff_1_to_2003_l136_136186

def floor_sum_diff (m : ℕ) : ℤ :=
  ∑ n in Finset.range (m + 1), (Int.floor (Real.sqrt n) - Int.floor (Real.cbrt n))

theorem sum_floor_diff_1_to_2003 :
  floor_sum_diff 2003 = 40842 :=
by
  sorry

end sum_floor_diff_1_to_2003_l136_136186


namespace zachary_nickels_l136_136969

-- Definitions based on conditions
variables (p n : ℕ)

-- Defining the conditions
def total_coins : Prop := p + n = 32
def total_value : Prop := p + 5 * n = 100

-- Stating the theorem
theorem zachary_nickels : total_coins p n → total_value p n → n = 17 :=
by {
    intros h_total_coins h_total_value,
    sorry
}

end zachary_nickels_l136_136969


namespace fraction_of_quarters_from_1860_to_1869_l136_136371

theorem fraction_of_quarters_from_1860_to_1869
  (total_quarters : ℕ) (quarters_from_1860s : ℕ)
  (h1 : total_quarters = 30) (h2 : quarters_from_1860s = 15) :
  (quarters_from_1860s : ℚ) / (total_quarters : ℚ) = 1 / 2 := by
  sorry

end fraction_of_quarters_from_1860_to_1869_l136_136371


namespace minimum_a_for_monotonic_increasing_on_interval_l136_136649

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp x - Real.log x

noncomputable def f_prime (a x : ℝ) : ℝ := a * Real.exp x - 1 / x

def min_value_of_a : ℝ := Real.exp (-1)

theorem minimum_a_for_monotonic_increasing_on_interval :
  (∀ x ∈ Set.Ioo 1 2, 0 ≤ f_prime a x) ↔ a ≥ min_value_of_a :=
by
  sorry

end minimum_a_for_monotonic_increasing_on_interval_l136_136649


namespace fair_coin_difference_l136_136928

noncomputable def binom_coeff (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem fair_coin_difference :
  let p := (1 : ℚ) / 2 in
  let p_1 := binom_coeff 5 4 * p^4 * (1 - p) in
  let p_2 := p^5 in
  p_1 - p_2 = 1 / 8 :=
by
  sorry

end fair_coin_difference_l136_136928


namespace baker_sold_more_pastries_l136_136153

theorem baker_sold_more_pastries {cakes_made pastries_made pastries_sold cakes_sold : ℕ}
    (h1 : cakes_made = 105)
    (h2 : pastries_made = 275)
    (h3 : pastries_sold = 214)
    (h4 : cakes_sold = 163) :
    pastries_sold - cakes_sold = 51 := by
  sorry

end baker_sold_more_pastries_l136_136153


namespace hockey_league_games_l136_136427

theorem hockey_league_games (n : ℕ) (k : ℕ) (h_n : n = 25) (h_k : k = 15) : 
  (n * (n - 1) / 2) * k = 4500 := by
  sorry

end hockey_league_games_l136_136427


namespace area_of_sectors_of_circle_l136_136443

theorem area_of_sectors_of_circle :
  (let r := 10
       sector1_angle := 45
       sector2_angle := 90
       area_of_circle := π * r^2
       sector1_area := (sector1_angle / 360) * area_of_circle
       sector2_area := (sector2_angle / 360) * area_of_circle in
   sector1_area + sector2_area = 37.5 * π) :=
by sorry

end area_of_sectors_of_circle_l136_136443


namespace constant_term_is_neg_160_l136_136192

-- Define the binomial term for the general (r+1)th term in the expansion
noncomputable def binomial_term (n r : ℕ) (a b x : ℚ) : ℚ :=
  (-1)^r * a^(n - r) * b^r * (n.choose r)

-- Define the main expression and calculate the constant term for the specified power
noncomputable def constant_term_in_binom_expansion : ℚ :=
  let a := (2 : ℚ)
  let b := (1 : ℚ)
  let r := 3
  binomial_term 6 r a b

-- Prove that the constant term is -160
theorem constant_term_is_neg_160 : constant_term_in_binom_expansion = -160 :=
by
  unfold constant_term_in_binom_expansion binomial_term
  apply congr_arg
  -- Substituting known values and simplifying expressions
  sorry

end constant_term_is_neg_160_l136_136192


namespace expected_value_linear_transform_l136_136560

-- Conditions
variable (X : Type) [MeasureSpace X]
variable [IsFiniteMeasure X]
variable (M : (X → ℝ) → ℝ)
variable (hMX : M (fun _ => 4) = 4)

-- Question (Equivalent proof problem)
theorem expected_value_linear_transform (hX : M (fun x : X => x) = 4) : M (fun x : X => 2 * x + 7) = 15 := by
  sorry

end expected_value_linear_transform_l136_136560


namespace min_marked_price_l136_136478

theorem min_marked_price 
  (x : ℝ) 
  (sets : ℝ) 
  (cost_per_set : ℝ) 
  (discount : ℝ) 
  (desired_profit : ℝ) 
  (purchase_cost : ℝ) 
  (total_revenue : ℝ) 
  (cost : ℝ)
  (h1 : sets = 40)
  (h2 : cost_per_set = 80)
  (h3 : discount = 0.9)
  (h4 : desired_profit = 4000)
  (h5 : cost = sets * cost_per_set)
  (h6 : total_revenue = sets * (discount * x))
  (h7 : total_revenue - cost ≥ desired_profit) : x ≥ 200 := by
  sorry

end min_marked_price_l136_136478


namespace second_student_weight_inconsistent_l136_136384

theorem second_student_weight_inconsistent:
  (average_weight_19_students: 15) ->
  (new_avg_weight_21_students: 14.6) ->
  (weight_first_new_student: 12) ->
  ∃ weight_second_new_student, (weight_second_new_student ≥ 14) ∧
  (285 + 12 + weight_second_new_student = 306.6) -> false :=
by
  intro average_weight_19_students new_avg_weight_21_students weight_first_new_student 
  use weight_second_new_student
  intros hw
  simp only [average_weight_19_students, new_avg_weight_21_students, weight_first_new_student] at *
  have h1 := calc
    306.6 - 285 - 12 = 9.6 : by norm_num
  have h2 := calc
    306.6 - 285 - 12 ≥ 14 : by linarith only [hw]
  linarith only [h1, h2]

end second_student_weight_inconsistent_l136_136384


namespace only_optionA_is_linear_l136_136966

-- Definitions for the given sets of equations
def optionA := (λ (x y : ℝ), (x - 2 = 0) ∧ (y = 7))
def optionB := (λ (x y z : ℝ), (6 * x + y = 1) ∧ (y + z = 7))
def optionC := (λ (x y : ℝ), (x - 3 * y = 6) ∧ (y - 2 * x * y = 0))
def optionD := (λ (x y : ℝ), ((2 / x) - 3 * y = 2) ∧ ((1 / y) + x = 4))

-- Predicate to check if a set of equations is a system of two linear equations in terms of x and y
def is_linear_system (eqns : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c d : ℝ, ∀ x y : ℝ, eqns x y → (a * x + b * y = c ∧ d * x + b * y = c)

-- Now we state our problem as a theorem in Lean
theorem only_optionA_is_linear :
  is_linear_system optionA ∧ ¬is_linear_system optionB ∧ ¬is_linear_system optionC ∧ ¬is_linear_system optionD :=
by
sorry

end only_optionA_is_linear_l136_136966


namespace one_fourth_of_eight_point_four_l136_136515

theorem one_fourth_of_eight_point_four : (8.4 / 4) = (21 / 10) :=
by
  -- The expected proof would go here
  sorry

end one_fourth_of_eight_point_four_l136_136515


namespace sector_area_15deg_radius_6cm_l136_136049

noncomputable def sector_area (r : ℝ) (theta : ℝ) : ℝ :=
  0.5 * theta * r^2

theorem sector_area_15deg_radius_6cm :
  sector_area 6 (15 * Real.pi / 180) = 3 * Real.pi / 2 := by
  sorry

end sector_area_15deg_radius_6cm_l136_136049


namespace sonny_cookie_problem_l136_136377

theorem sonny_cookie_problem 
  (total_boxes : ℕ) (boxes_sister : ℕ) (boxes_cousin : ℕ) (boxes_left : ℕ) (boxes_brother : ℕ) : 
  total_boxes = 45 → boxes_sister = 9 → boxes_cousin = 7 → boxes_left = 17 → 
  boxes_brother = total_boxes - boxes_left - boxes_sister - boxes_cousin → 
  boxes_brother = 12 :=
by
  intros h_total h_sister h_cousin h_left h_brother
  rw [h_total, h_sister, h_cousin, h_left] at h_brother
  exact h_brother

end sonny_cookie_problem_l136_136377


namespace largest_value_among_three_l136_136700

variable (a b : ℝ)

theorem largest_value_among_three (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) : 
  ∃ x, (x = log b a ∧ x > a^b ∧ x > a * b) :=
by
  -- Due to the nature of the problem, we assert that:
  -- We are asked to find x which is equal to log_b(a) and that it is greater than both a^b and ab
  -- The conditions are: 0 < a < b < 1 
  sorry

end largest_value_among_three_l136_136700


namespace pipe_fill_rate_l136_136006

variable (R_A R_B : ℝ)

theorem pipe_fill_rate :
  R_A = 1 / 32 →
  R_A + R_B = 1 / 6.4 →
  R_B / R_A = 4 :=
by
  intros hRA hSum
  have hRA_pos : R_A ≠ 0 := by linarith
  sorry

end pipe_fill_rate_l136_136006


namespace min_value_of_a_l136_136671

theorem min_value_of_a {f : ℝ → ℝ} (a : ℝ) (h : ∀ x ∈ Ioo 1 2, deriv (λ x, a * real.exp x - real.log x) x ≥ 0) : 
  a ≥ real.exp (-1) :=
sorry

end min_value_of_a_l136_136671


namespace problem_y_value_l136_136277

theorem problem_y_value :
  (y : ℝ) → y = 2 ↔ y = real.sqrt (2 + real.sqrt (2 + real.sqrt (2 + real.sqrt (2 + real.sqrt (2 + real.sqrt (2 + real.sqrt (2 + …))))))) :=
begin
  sorry
end

end problem_y_value_l136_136277


namespace min_value_of_a_l136_136676

theorem min_value_of_a {f : ℝ → ℝ} (a : ℝ) (h : ∀ x ∈ Ioo 1 2, deriv (λ x, a * real.exp x - real.log x) x ≥ 0) : 
  a ≥ real.exp (-1) :=
sorry

end min_value_of_a_l136_136676


namespace base10_sum_in_base7_l136_136957

-- Statement: Prove that the base 7 representation of the sum of 37_10 and 45_10 is 145_7.
theorem base10_sum_in_base7 : nat_to_base 7 (37 + 45) = [1, 4, 5] := by
  sorry

end base10_sum_in_base7_l136_136957


namespace complex_quad_l136_136238

-- Define the complex conjugate function
def conj (z : ℂ) : ℂ := ⟨z.re, -z.im⟩

-- Define the complex division
def cdiv (z w : ℂ) : ℂ :=
  let denom := w.re * w.re + w.im * w.im
  ⟨(z.re * w.re + z.im * w.im) / denom, (z.im * w.re - z.re * w.im) / denom⟩

-- Define the conditions
axiom h1 : ∃ z : ℂ, conj z = 1 + 3 * complex.i

-- Define the main theorem
theorem complex_quad (z : ℂ) (h : conj z = 1 + 3 * complex.i) : 
  let w := cdiv z (1 + complex.i) 
  in w.re < 0 ∧ w.im < 0 :=
sorry

end complex_quad_l136_136238


namespace min_value_of_a_l136_136633

theorem min_value_of_a (a : ℝ) : 
  (∀ x ∈ Ioo 1 2, (a * exp x - 1/x) ≥ 0) → a = exp (-1) :=
by
  sorry

end min_value_of_a_l136_136633


namespace concyclic_of_quadrilateral_l136_136806

open EuclideanGeometry

noncomputable def midpoint_of_arc (Γ : Circle) (B C : Point) : Point :=
  sorry -- Definition of A as the midpoint of the arc BC

def points_on_circle (Γ : Circle) (P : Point) : Prop :=
  -- Definition to denote P lies on circle Γ
  sorry 

def intersection_of_chords (A D B C : Point) : Point :=
  sorry -- Definition of intersection point of chords AD and BC

theorem concyclic_of_quadrilateral
  (Γ : Circle) (B C D E : Point)
  (hBC : isChord B C Γ)
  (hD_onΓ : points_on_circle Γ D)
  (hE_onΓ : points_on_circle Γ E)
  (hD_not_on_arc : ¬(isOnArc D (midpoint_of_arc Γ B C) B C))
  (hE_not_on_arc : ¬( isOnArc E (midpoint_of_arc Γ B C) B C))
  (A := midpoint_of_arc Γ B C)
  (F := intersection_of_chords A D B C)
  (G := intersection_of_chords A E B C) :
  cyclic_quad D E F G :=
sorry

end concyclic_of_quadrilateral_l136_136806


namespace fair_coin_flip_probability_difference_l136_136942

noncomputable def choose : ℕ → ℕ → ℕ
| _, 0 => 1
| 0, _ => 0
| n + 1, r + 1 => choose n r + choose n (r + 1)

theorem fair_coin_flip_probability_difference :
  let p := 1 / 2
  let p_heads_4_out_of_5 := choose 5 4 * p^4 * p
  let p_heads_5_out_of_5 := p^5
  p_heads_4_out_of_5 - p_heads_5_out_of_5 = 1 / 8 :=
by
  sorry

end fair_coin_flip_probability_difference_l136_136942


namespace non_GF_non_V_non_NF_cupcakes_l136_136494

-- Define the relevant conditions as constants and parameters
constant total_cupcakes : ℕ := 200
constant gf_percentage : ℝ := 0.40
constant v_percentage : ℝ := 0.25
constant nf_percentage : ℝ := 0.30
constant gf_v_percentage : ℝ := 0.20
constant gf_nf_percentage : ℝ := 0.15
constant nf_v_percentage : ℝ := 0.25
constant gf_nf_not_v_percentage : ℝ := 0.10

-- Define the Lean theorem using the given conditions to prove the answer
theorem non_GF_non_V_non_NF_cupcakes : 
  let gf_cupcakes := gf_percentage * total_cupcakes,
      v_cupcakes := v_percentage * total_cupcakes,
      nf_cupcakes := nf_percentage * total_cupcakes,
      gf_v_cupcakes := gf_v_percentage * gf_cupcakes,
      gf_nf_cupcakes := gf_nf_percentage * gf_cupcakes,
      nf_v_cupcakes := nf_v_percentage * nf_cupcakes,
      gf_nf_not_v_cupcakes := gf_nf_not_v_percentage * total_cupcakes,
      total_gf_v_nf := gf_cupcakes + v_cupcakes + nf_cupcakes - gf_v_cupcakes - gf_nf_cupcakes - nf_v_cupcakes + gf_nf_not_v_cupcakes,
      non_gf_non_v_non_nf := total_cupcakes - total_gf_v_nf
  in non_gf_non_v_non_nf = 33 :=
by
  sorry

end non_GF_non_V_non_NF_cupcakes_l136_136494


namespace fair_coin_flip_difference_l136_136924

theorem fair_coin_flip_difference :
  let P1 := 5 * (1 / 2)^5;
  let P2 := (1 / 2)^5;
  P1 - P2 = 9 / 32 :=
by
  sorry

end fair_coin_flip_difference_l136_136924


namespace buy_items_ways_l136_136724

theorem buy_items_ways (headphones keyboards mice keyboard_mouse_sets headphone_mouse_sets : ℕ) :
  headphones = 9 → keyboards = 5 → mice = 13 → keyboard_mouse_sets = 4 → headphone_mouse_sets = 5 →
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 :=
by
  intros h_eq k_eq m_eq kms_eq hms_eq
  have h_eq_gen : headphones = 9 := h_eq
  have k_eq_gen : keyboards = 5 := k_eq
  have m_eq_gen : mice = 13 := m_eq
  have kms_eq_gen : keyboard_mouse_sets = 4 := kms_eq
  have hms_eq_gen : headphone_mouse_sets = 5 := hms_eq
  sorry

end buy_items_ways_l136_136724


namespace num_5_digit_palindromes_eq_900_l136_136172

-- Definitions of the conditions
def is_palindrome (n : Nat) : Prop :=
  let digits := to_digits 10 n
  digits = List.reverse digits

def is_5_digit (n : Nat) : Prop :=
  n >= 10000 ∧ n < 100000

def is_valid_start (n : Nat) : Prop :=
  let digits := to_digits 10 n
  digits.head ≠ 0

def to_digits (base : Nat) (n : Nat) : List Nat :=
  if n = 0 then [0]
  else List.reverse $ List.unfoldr (λ n, if n = 0 then none else some (n % base, n / base)) n

-- Prove the number of 5-digit palindromes is 900
theorem num_5_digit_palindromes_eq_900 :
  {n : Nat | is_5_digit n ∧ is_valid_start n ∧ is_palindrome n}.card = 900 :=
  sorry

end num_5_digit_palindromes_eq_900_l136_136172


namespace apples_in_bowl_l136_136415

theorem apples_in_bowl (green_plus_red_diff red_count : ℕ) (h1 : green_plus_red_diff = 12) (h2 : red_count = 16) :
  red_count + (red_count + green_plus_red_diff) = 44 :=
by
  sorry

end apples_in_bowl_l136_136415


namespace line_canonical_form_l136_136101

theorem line_canonical_form :
  (∀ x y z : ℝ, 4 * x + y - 3 * z + 2 = 0 → 2 * x - y + z - 8 = 0 ↔
    ∃ t : ℝ, x = 1 + -2 * t ∧ y = -6 + -10 * t ∧ z = -6 * t) :=
by
  sorry

end line_canonical_form_l136_136101


namespace probability_x_y_gte_one_fourth_l136_136369

open ProbabilityTheory

noncomputable def probability_x_y_condition := sorry

theorem probability_x_y_gte_one_fourth:
  let E : Set (ℝ × ℝ) := {z : ℝ × ℝ | |z.1 - z.2| > 1 / 4} in
  P(E | probability_x_y_condition) = 1 / 2 := 
sorry

end probability_x_y_gte_one_fourth_l136_136369


namespace triangle_sin_ratio_cos_side_l136_136290

noncomputable section

variables (A B C a b c : ℝ)
variables (h1 : a + b + c = 5)
variables (h2 : Real.cos B = 1 / 4)
variables (h3 : Real.cos A - 2 * Real.cos C = (2 * c - a) / b * Real.cos B)

theorem triangle_sin_ratio_cos_side :
  (Real.sin C / Real.sin A = 2) ∧ (b = 2) :=
  sorry

end triangle_sin_ratio_cos_side_l136_136290


namespace added_water_correct_l136_136989

theorem added_water_correct (initial_fullness : ℝ) (final_fullness : ℝ) (capacity : ℝ) (initial_amount : ℝ) (final_amount : ℝ) (added_water : ℝ) :
    initial_fullness = 0.30 →
    final_fullness = 3/4 →
    capacity = 60 →
    initial_amount = initial_fullness * capacity →
    final_amount = final_fullness * capacity →
    added_water = final_amount - initial_amount →
    added_water = 27 :=
by
  intros
  -- Insert the proof here
  sorry

end added_water_correct_l136_136989


namespace select_six_numbers_l136_136580

theorem select_six_numbers (S : Finset (Fin 100)) (hS : S.card = 51) :
  ∃ T : Finset (Fin 100), T.card = 6 ∧ ∀ (a b : Fin 100), a ∈ T → b ∈ T → a ≠ b →
  ¬(a.toNat / 10 = b.toNat / 10 ∨ a.toNat % 10 = b.toNat % 10) :=
by
  sorry

end select_six_numbers_l136_136580


namespace smallest_f_n_l136_136208

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def pairwise_coprime (s : Set ℕ) : Prop :=
  ∀ a b ∈ s, a ≠ b → is_coprime a b

def f (n : ℕ) : ℕ :=
  (n + 1) / 2 + (n + 1) / 3 - (n + 1) / 6 + 1

theorem smallest_f_n (n : ℕ) (h : n ≥ 4) :
  ∀ m : ℕ, ∀ s : Finset ℕ, (s.card = f n) → (s ⊆ Finset.range (m + n)) → 
  ∃ a b c ∈ s, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ is_coprime a b ∧ is_coprime b c ∧ is_coprime a c  := 
by
  sorry

end smallest_f_n_l136_136208


namespace total_games_in_season_l136_136420

theorem total_games_in_season {n : ℕ} {k : ℕ} (h1 : n = 25) (h2 : k = 15) :
  (n * (n - 1) / 2) * k = 4500 :=
by
  sorry

end total_games_in_season_l136_136420


namespace trigonometric_series_sum_l136_136518

theorem trigonometric_series_sum :
  (∑ x in Finset.range 30, 2 * Real.cos (x + 1) * Real.cos 2 * (1 + Real.csc (x : ℝ + 1 - 1) * Real.csc (x + 1 + 2))) = 33 :=
by
  sorry

end trigonometric_series_sum_l136_136518


namespace second_player_wins_l136_136522

-- Definitions based on problem conditions
def allowed (n : ℕ) : Prop :=
  (nat.factors n).to_finset.card ≤ 20

def initial_stones : ℕ := nat.factorial 2004

-- Game outcome definition and conditions
theorem second_player_wins :
  ∀ initial number_of_stones (player1_take : ℕ) (player2_take: ℕ), 
  initial = initial_stones →
  allowed player1_take →
  allowed player2_take →
  (number_of_stones = initial - player1_take) ∧ ¬allowed (number_of_stones) →
  (number_of_stones - player2_take) % (prod (take 21 (nat.primes))).val = 0 →
  wins (second_player : bool) :=
sorry

end second_player_wins_l136_136522


namespace max_product_with_853_l136_136444

-- Define the digits to be used
def digits : List ℕ := [3, 5, 6, 8, 9]

-- Define the three-digit number 853 using the available digits
def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

-- Define the two-digit number
def two_digit_number (d e : ℕ) : ℕ := 10 * d + e

-- Define the product of the three-digit number and the two-digit number
def product (a b c d e : ℕ) : ℕ := three_digit_number a b c * two_digit_number d e

-- Define the condition that each digit is used exactly once
def used_once (a b c d e : ℕ) : Prop :=
  digits.perm [a, b, c, d, e]

-- Prove that the greatest product is achieved with the three-digit integer 853
theorem max_product_with_853 : ∃ (d e : ℕ), used_once 8 5 3 d e ∧
  ∀ (a b c d' e' : ℕ), used_once a b c d' e' → 
    three_digit_number a b c = 853 →
    product a b c d' e' ≥ product a b c d e :=
by sorry

end max_product_with_853_l136_136444


namespace minimum_value_of_a_l136_136667

theorem minimum_value_of_a (a : ℝ) (h : ∀ x ∈ set.Ioo 1 2, ae^x - 1/x ≥ 0) : a ≥ e⁻¹ :=
sorry

end minimum_value_of_a_l136_136667


namespace ice_cream_remaining_l136_136108

def total_initial_scoops : ℕ := 3 * 10
def ethan_scoops : ℕ := 1 + 1
def lucas_danny_connor_scoops : ℕ := 2 * 3
def olivia_scoops : ℕ := 1 + 1
def shannon_scoops : ℕ := 2 * olivia_scoops
def total_consumed_scoops : ℕ := ethan_scoops + lucas_danny_connor_scoops + olivia_scoops + shannon_scoops
def remaining_scoops : ℕ := total_initial_scoops - total_consumed_scoops

theorem ice_cream_remaining : remaining_scoops = 16 := by
  sorry

end ice_cream_remaining_l136_136108


namespace intersection_is_l136_136777

noncomputable section

open Real

def point := ℝ × ℝ

def M : point := (0, 0)
def P : point := (0, 5)
def Q : point := (12/5, 9/5)

def circle_through (p : point) (h k : ℝ) (r : ℝ) : Prop :=
  (p.1 - h)^2 + (p.2 - k)^2 = r^2

def A : set (ℝ × ℝ → Prop) := {c | ∃ h k r, circle_through M h k r}
def B : set (ℝ × ℝ → Prop) := {c | ∃ h k r, circle_through P h k r}
def C : set (ℝ × ℝ → Prop) := {c | ∃ h k r, circle_through Q h k r}

def intersection (A B C : set (ℝ × ℝ → Prop)) : set (ℝ × ℝ → Prop) :=
  {c | c ∈ A ∧ c ∈ B ∧ c ∈ C}

theorem intersection_is : ∀ c ∈ intersection A B C,
  ∃ h k r, ∀ x y, c (x, y) ↔ x^2 + (y - 5/2)^2 = 25/4 :=
  sorry

end intersection_is_l136_136777


namespace new_energy_vehicles_l136_136968

-- Given conditions
def conditions (a b : ℕ) : Prop :=
  3 * a + 2 * b = 95 ∧ 4 * a + 1 * b = 110

-- Given prices
def purchase_prices : Prop :=
  ∃ a b, conditions a b ∧ a = 25 ∧ b = 10

-- Total value condition for different purchasing plans
def purchase_plans (m n : ℕ) : Prop :=
  25 * m + 10 * n = 250 ∧ m > 0 ∧ n > 0

-- Number of different purchasing plans
def num_purchase_plans : Prop :=
  ∃ num_plans, num_plans = 4

-- Profit calculation for a given plan
def profit (m n : ℕ) : ℕ :=
  12 * m + 8 * n

-- Maximum profit condition
def max_profit : Prop :=
  ∃ max_profit, max_profit = 184 ∧ ∀ (m n : ℕ), purchase_plans m n → profit m n ≤ 184

-- Main theorem
theorem new_energy_vehicles : purchase_prices ∧ num_purchase_plans ∧ max_profit :=
  sorry

end new_energy_vehicles_l136_136968


namespace initial_moments_l136_136561

noncomputable def pdf (x : ℝ) : ℝ :=
if x ≤ 1 then 0 else 5 / x ^ 6

noncomputable def moment (k : ℕ) : ℝ :=
if k < 5 then 
  5 / (5 - k) 
else 
  0 

theorem initial_moments (k : ℕ) : 
  (∃ m, m = ∫ x in (1 : ℝ)..(float.infinity), x ^ k * pdf x ) → 
  k < 5 → moment k = 5 / (5 - k) 
:= 
begin
  sorry
end

end initial_moments_l136_136561


namespace largest_prime_divisor_of_base6_number_l136_136562

theorem largest_prime_divisor_of_base6_number :
  let n := 1 * 6^8 + 0 * 6^7 + 2 * 6^6 + 1 * 6^5 + 1 * 6^4 + 1 * 6^3 + 0 * 6^2 + 1 * 6^1 + 1 * 6^0 in
  n = 1785223 →
  ∃ p, prime p ∧ p ∣ 1785223 ∧ (∀ q, prime q ∧ q ∣ 1785223 → q ≤ p) ∧ p = 162293 :=
by
  sorry

end largest_prime_divisor_of_base6_number_l136_136562


namespace j_at_4_l136_136852

noncomputable def h (x : ℚ) : ℚ := 5 / (3 - x)

noncomputable def h_inv (x : ℚ) : ℚ := (3 * x - 5) / x

noncomputable def j (x : ℚ) : ℚ := (1 / h_inv x) + 7

theorem j_at_4 : j 4 = 53 / 7 :=
by
  -- Proof steps would be inserted here.
  sorry

end j_at_4_l136_136852


namespace no_pair_ab_equal_M_and_N_l136_136826

def f (x : ℝ) : ℝ := -x / (1 + abs x)

def M (a b : ℝ) := set.Icc a b

def N (a b : ℝ) := {y : ℝ | ∃ x : ℝ, x ∈ M a b ∧ y = f x}

theorem no_pair_ab_equal_M_and_N (a b : ℝ) (h : a < b) : M a b ≠ N a b :=
by
  sorry

end no_pair_ab_equal_M_and_N_l136_136826


namespace game_show_prizes_count_l136_136485

theorem game_show_prizes_count :
  let digits := [1, 2, 2, 2, 3, 3, 3]
  let D_vals := Finset.Icc 1 999
  let E_vals := Finset.Icc 1 999
  let F_vals := Finset.Icc 1 999
  ∃ (D E F : ℕ), 
    D ∈ D_vals ∧ E ∈ E_vals ∧ F ∈ F_vals ∧
    multiset.of_list digits = multiset.of_list (D.digits 10) + multiset.of_list (E.digits 10) + multiset.of_list (F.digits 10) →
    (Finset.card (Finset.filter (λ (d : ℕ × ℕ × ℕ), 
          let D := d.1 in
          let E := d.2.1 in
          let F := d.2.2 in
          D + E + F = sum digits) 
        (Finset.product (Finset.product D_vals E_vals) F_vals))) = 2100 := 
sorry

end game_show_prizes_count_l136_136485


namespace max_m_value_l136_136609

theorem max_m_value (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) : 
  (∀ a b : ℝ, a > 0 → b > 0 → ((2 / a) + (1 / b) ≥ (m / (2 * a + b)))) → m ≤ 9 :=
sorry

end max_m_value_l136_136609


namespace subcommittee_count_l136_136484

theorem subcommittee_count :
  (nat.choose 10 4) * (nat.choose 8 3) = 11760 :=
by sorry

end subcommittee_count_l136_136484


namespace find_triples_of_reciprocals_eq_one_l136_136986

theorem find_triples_of_reciprocals_eq_one :
    { (a, b, c) : ℕ × ℕ × ℕ // 0 < a ∧ 0 < b ∧ 0 < c ∧ (1 / (a:ℝ) + 1 / (b:ℝ) + 1 / (c:ℝ) = 1) }
        = { (2, 3, 6), (2, 6, 3), (3, 2, 6), (3, 6, 2), (6, 2, 3), (6, 3, 2),
            (2, 4, 4), (4, 2, 4), (4, 4, 2), (3, 3, 3) } := 
sorry

end find_triples_of_reciprocals_eq_one_l136_136986


namespace cube_root_of_sum_powers_l136_136959

noncomputable def cube_root (x : ℝ) : ℝ := x ^ (1 / 3)

theorem cube_root_of_sum_powers :
  cube_root (2^7 + 2^7 + 2^7) = 4 * cube_root 2 :=
by
  sorry

end cube_root_of_sum_powers_l136_136959


namespace angela_spent_78_l136_136150

-- Definitions
def angela_initial_money : ℕ := 90
def angela_left_money : ℕ := 12
def angela_spent_money : ℕ := angela_initial_money - angela_left_money

-- Theorem statement
theorem angela_spent_78 : angela_spent_money = 78 := by
  -- Proof would go here, but it is not required.
  sorry

end angela_spent_78_l136_136150


namespace instantaneous_velocity_at_1_l136_136990

-- Define the motion equation
def motion_eq (g t : ℝ) : ℝ := (1/2) * g * t^2

-- Define the limit condition
def limit_condition (g : ℝ) : Prop :=
  ∀ (ε > 0), ∃ (δ > 0), ∀ (Δt : ℝ), (0 < abs Δt ∧ abs Δt < δ) → abs ((motion_eq g (1 + Δt) - motion_eq g 1) / Δt - g) < ε

theorem instantaneous_velocity_at_1 (g : ℝ) (h : limit_condition g) (h_g : g = 9.8) : 
  (∃ v, v = g) := 
sorry

end instantaneous_velocity_at_1_l136_136990


namespace non_neg_integer_solutions_l136_136190

theorem non_neg_integer_solutions (a b c : ℕ) :
  (∀ x : ℕ, x^2 - 2 * a * x + b = 0 → x ≥ 0) ∧ 
  (∀ y : ℕ, y^2 - 2 * b * y + c = 0 → y ≥ 0) ∧ 
  (∀ z : ℕ, z^2 - 2 * c * z + a = 0 → z ≥ 0) → 
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ 
  (a = 0 ∧ b = 0 ∧ c = 0) :=
sorry

end non_neg_integer_solutions_l136_136190


namespace sum_ratio_l136_136805

-- Definitions and conditions
variable {a_n : ℕ → ℚ} -- Assume a_n is a sequence of rational numbers

-- Sum of first n terms of arithmetic sequence
def S (n : ℕ) : ℚ := n * (a_n 1 + a_n n) / 2

-- Given conditions
variable (h : a_n 6 / a_n 5 = 9 / 11)

-- Theorem to prove
theorem sum_ratio : S 11 / S 9 = 1 :=
by sorry

end sum_ratio_l136_136805


namespace fair_coin_flip_probability_difference_l136_136943

noncomputable def choose : ℕ → ℕ → ℕ
| _, 0 => 1
| 0, _ => 0
| n + 1, r + 1 => choose n r + choose n (r + 1)

theorem fair_coin_flip_probability_difference :
  let p := 1 / 2
  let p_heads_4_out_of_5 := choose 5 4 * p^4 * p
  let p_heads_5_out_of_5 := p^5
  p_heads_4_out_of_5 - p_heads_5_out_of_5 = 1 / 8 :=
by
  sorry

end fair_coin_flip_probability_difference_l136_136943


namespace coin_flip_difference_l136_136952

/-- The positive difference between the probability of a fair coin landing heads up
exactly 4 times out of 5 flips and the probability of a fair coin landing heads up
5 times out of 5 flips is 1/8. -/
theorem coin_flip_difference :
  (5 * (1 / 2) ^ 5) - ((1 / 2) ^ 5) = (1 / 8) :=
by
  sorry

end coin_flip_difference_l136_136952


namespace integer_coordinates_midpoint_exists_l136_136528

theorem integer_coordinates_midpoint_exists (P : Fin 5 → ℤ × ℤ) :
  ∃ i j : Fin 5, i ≠ j ∧
    ∃ x y : ℤ, (2 * x = (P i).1 + (P j).1) ∧ (2 * y = (P i).2 + (P j).2) := sorry

end integer_coordinates_midpoint_exists_l136_136528


namespace Pyarelal_loss_share_l136_136096

-- Define the conditions
variables (P : ℝ) (A : ℝ) (total_loss : ℝ)

-- Ashok's capital is 1/9 of Pyarelal's capital
axiom Ashok_capital : A = (1 / 9) * P

-- Total loss is Rs 900
axiom total_loss_val : total_loss = 900

-- Prove Pyarelal's share of the loss is Rs 810
theorem Pyarelal_loss_share : (P / (A + P)) * total_loss = 810 :=
by
  sorry

end Pyarelal_loss_share_l136_136096


namespace sequence_geom_and_sum_l136_136784

theorem sequence_geom_and_sum :
  ∀ (n : ℕ), n > 0 →
  ∃ a : ℕ → ℕ,
    (a 1 = 1) ∧
    (∀ n, n > 0 → a (n + 1) = 2 * a n - n + 2) ∧
    (∀ m : ℕ, m > 0 → ∃ r, ∀ k : ℕ, k ≥ 1 →
      (a k - (k - 1) = (a 1 - 1 + 1) * r ^ (k - 1))) ∧
    (∀ n, a n = 2 ^ (n - 1) + (n - 1)) ∧
    (∑ i in finset.range n, a (i + 1) = 2 ^ n - 1 + n * (n - 1) / 2) :=
by {
  sorry
}

end sequence_geom_and_sum_l136_136784


namespace largest_log_value_l136_136908

theorem largest_log_value :
  ∃ (x y z t : ℝ) (a b c : ℝ),
    x ≤ y ∧ y ≤ z ∧ z ≤ t ∧
    a = Real.log y / Real.log x ∧
    b = Real.log z / Real.log y ∧
    c = Real.log t / Real.log z ∧
    a = 15 ∧ b = 20 ∧ c = 21 ∧
    (∃ u v w, u = a * b ∧ v = b * c ∧ w = a * b * c ∧ w = 420) := sorry

end largest_log_value_l136_136908


namespace average_speed_correct_l136_136477

-- Conditions from the problem
constant d1 : Real := 25
constant d2 : Real := 30
constant s1 : Real := 35
constant s2 : Real := 55
constant t3 : Real := (40 : Real) / 60
constant s3 : Real := 65
constant t4 : Real := (20 : Real) / 60
constant s4 : Real := 50

-- Derived distances from times and speeds
def d3 : Real := s3 * t3
def d4 : Real := s4 * t4

-- Total distance and total time calculations
def total_distance : Real := d1 + d2 + d3 + d4
def t1 : Real := d1 / s1
def t2 : Real := d2 / s2
def total_time : Real := t1 + t2 + t3 + t4

-- Average speed calculation
def average_speed : Real := total_distance / total_time

-- Lean statement to express the proof problem
theorem average_speed_correct :
  average_speed = 51.52 := by
  sorry

end average_speed_correct_l136_136477


namespace number_added_l136_136893

theorem number_added (x y : ℝ) (h1 : x = 33) (h2 : x / 4 + y = 15) : y = 6.75 :=
by sorry

end number_added_l136_136893


namespace number_of_pairs_of_shoes_l136_136106

/-- A box contains some pairs of shoes with a total of 10 shoes.
    If two shoes are selected at random, the probability that they are matching shoes is 1/9.
    Prove that the number of pairs of shoes in the box is 5. -/
theorem number_of_pairs_of_shoes (n : ℕ) (h1 : 2 * n = 10) 
  (h2 : ((n * (n - 1)) / (10 * (10 - 1))) = 1 / 9) : n = 5 := 
sorry

end number_of_pairs_of_shoes_l136_136106


namespace interior_diagonal_exists_l136_136322

-- Defining the simple polygon P with at least four vertices
variable {P : Type} [simple_polygon P] (h1 : 4 ≤ P.num_vertices)

-- Defining what it means for P to have an interior diagonal
def has_interior_diagonal : Prop := ∃ (d : diagonal P), d.is_interior

-- The theorem statement for the proof problem
theorem interior_diagonal_exists (h1 : 4 ≤ P.num_vertices) : has_interior_diagonal P := 
sorry

end interior_diagonal_exists_l136_136322


namespace customers_left_tip_l136_136140

-- Definition of the given conditions
def initial_customers : ℕ := 29
def added_customers : ℕ := 20
def customers_didnt_tip : ℕ := 34

-- Lean 4 statement proving that the number of customers who did leave a tip (answer) equals 15
theorem customers_left_tip : (initial_customers + added_customers - customers_didnt_tip) = 15 :=
by
  sorry

end customers_left_tip_l136_136140


namespace savings_same_l136_136105

theorem savings_same (A_salary B_salary total_salary : ℝ)
  (A_spend_perc B_spend_perc : ℝ)
  (h_total : A_salary + B_salary = total_salary)
  (h_A_salary : A_salary = 4500)
  (h_A_spend_perc : A_spend_perc = 0.95)
  (h_B_spend_perc : B_spend_perc = 0.85)
  (h_total_salary : total_salary = 6000) :
  ((1 - A_spend_perc) * A_salary) = ((1 - B_spend_perc) * B_salary) :=
by
  sorry

end savings_same_l136_136105


namespace carnival_ring_toss_revenue_per_day_l136_136881

theorem carnival_ring_toss_revenue_per_day : 
  ∀ (total_money days : ℕ), total_money = 7560 ∧ days = 12 → total_money / days = 630 :=
by
  intros total_money days h,
  cases h,
  rw [h_left, h_right],
  norm_num,
  sorry

end carnival_ring_toss_revenue_per_day_l136_136881


namespace estimated_total_sales_volume_daily_purchase_quantity_l136_136862

/-- Define the sales volumes for 20 days. --/
def sales_volumes : List ℕ := [40, 42, 44, 45, 46, 48, 52, 52, 53, 54, 55, 56, 57, 58, 59, 61, 63, 64, 65, 66]

/-- Prove the estimated total sales volume for 30 days is 1620 kg, given the sales volumes for 20 days. --/
theorem estimated_total_sales_volume : 
  (sales_volumes.sum / sales_volumes.length : ℚ) * 30 = 1620 :=
by
  sorry

/-- Prove that the daily purchase quantity of apples to meet 75% of customers' needs should be 59 kg. --/
theorem daily_purchase_quantity : 
  sales_volumes.nth_le ((20 * 75 / 100) - 1) (by norm_num) = 59 :=
by
  sorry

end estimated_total_sales_volume_daily_purchase_quantity_l136_136862


namespace coin_flip_probability_difference_l136_136945

theorem coin_flip_probability_difference :
  let p1 := (nat.choose 5 4) * (1 / 2)^5
  let p2 := (1 / 2)^5
  (p1 - p2 = 1 / 8) :=
by
  let p1 := (nat.choose 5 4) * (1 / 2)^5
  let p2 := (1 / 2)^5
  sorry

end coin_flip_probability_difference_l136_136945


namespace find_length_of_base_l136_136304

noncomputable def length_of_base (M N : ℝ × ℝ) : ℝ :=
  let D := ((M.1 + N.1) / 2, 0) in
  ∥(D.1, D.2) - M∥ * 2

theorem find_length_of_base :
  let M := (2, 2)
  let N := (4, 4)
  length_of_base M N = 4 * Real.sqrt 5 :=
by
  let M : ℝ × ℝ := (2, 2)
  let N : ℝ × ℝ := (4, 4)
  show length_of_base M N = 4 * Real.sqrt 5
  sorry

end find_length_of_base_l136_136304


namespace part_a_part_b_l136_136464

-- Proof for part (a)
theorem part_a (x : ℝ) : (x - 1) * (x + 1) + 1 = x^2 := 
by 
  calc
    (x - 1) * (x + 1) + 1 = (x^2 - 1) + 1 : by ring
                       ... = x^2          : by ring

-- Proof for part (b)
theorem part_b : sqrt(1 + 2014 * sqrt(1 + 2015 * sqrt(1 + 2016 * 2018))) = 2015 := 
by
  have h1 : 2016 * 2018 = 2016 * (2017 + 1) := by ring,
  have h2 : 2016 * 2017 = (2016 - 1) * (2016 + 1) := by ring,
  have h3 : (2016 - 1) * (2016 + 1) + 2016 = 2017^2 - 1 + 2016 := by ring,
  have h4 : 1 + 2016 * 2018 = 2017^2 + 1 := by rw [←h1, h3, sq],
  have h5 : 2016^2 = (2016 * 2016) := pow_two,
  have h6 : sqrt(1 + 2015 * sqrt(1 + 2016 * 2018)) = sqrt(1 + 2015 * 2017) := by rw h4,
  have h7 : 1 + 2015 * 2017 = 2016^2 := 
    by calc 
      1 + 2015 * 2017 = 1 + (2015 * 2017) : by ring
      ... = 2016 * 2016 := by rw h5,
  have h8 : sqrt(1 + 2015 * sqrt(2017^2)) = sqrt(1 + 2015 * 2017) := by rw [sq_sqrt, h7],
  have h9 : sqrt(2016^2) = 2016 := by rw sq_sqrt,
  have h10 : sqrt(1 + 2014 * 2016) = sqrt(2015^2) :=
    by calc 
      1 + 2014 * 2016 = (2015 * 2015) : eq_symm h7,
  calc 
    sqrt(1 + 2014 * sqrt(1 + 2015 * sqrt(1 + 2016 * 2018))) = sqrt(1 + 2014 * 2016)    : by rw h8
    ... = sqrt(2015^2)           : eq_symm h10 
    ... = 2015                   : by rw sq_sqrt,

end part_a_part_b_l136_136464


namespace total_time_last_two_videos_l136_136798

theorem total_time_last_two_videos
  (first_video_length : ℕ := 2 * 60)
  (second_video_length : ℕ := 4 * 60 + 30)
  (total_time : ℕ := 510) :
  ∃ t1 t2 : ℕ, t1 ≠ t2 ∧ t1 + t2 = total_time - first_video_length - second_video_length := by
  sorry

end total_time_last_two_videos_l136_136798


namespace smallest_period_2pi_l136_136510

def smallest_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  (∀ x, f (x + T) = f x) ∧ (T > 0) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T')

theorem smallest_period_2pi :
  smallest_positive_period (λ x, abs (sin (x / 2))) (2 * Real.pi) :=
sorry

end smallest_period_2pi_l136_136510


namespace fifteen_colors_are_sufficient_l136_136987

structure Network (n : ℕ) :=
  (lines : Finset (Fin (n + 1) × Fin (n + 1)))
  (nodes : Finset (Fin (n + 1)))
  (adj_lines : ∀ (x : Fin (n + 1)), x ∈ nodes -> ∃ s : Finset (Fin (n + 1)), s.card ≤ 10 ∧ ∀ l ∈ s, (x, l) ∈ lines)

def colorable_with_15_colors (network : Network n) : Prop :=
  ∃ (coloring : Finset (Fin (n + 1) × Fin (n + 1)) → Fin 15), 
    ∀ (x y : Fin (n + 1)), (x, y) ∈ network.lines ∧ (x = y) → coloring (x, y) ≠ coloring (y, x)

theorem fifteen_colors_are_sufficient (n : ℕ) (network : Network n) : colorable_with_15_colors network := sorry

end fifteen_colors_are_sufficient_l136_136987


namespace number_of_primes_in_sequence_is_zero_l136_136461

-- Define Q as the product of all prime numbers less than or equal to 59
def Q : ℕ := Nat.factorial (59 / 2)

-- Sequence defined as Q + n for 1 ≤ n ≤ 60
def sequence (n : ℕ) : ℕ := Q + n

-- Prove that the number of primes in the sequence is 0
theorem number_of_primes_in_sequence_is_zero : 
  ∀ n, 1 ≤ n ∧ n ≤ 60 → ¬ Nat.Prime (sequence n) :=
by
  sorry

end number_of_primes_in_sequence_is_zero_l136_136461


namespace quadratic_intersects_x_axis_only_once_l136_136685

theorem quadratic_intersects_x_axis_only_once (a : ℝ) :
  (∀ x : ℝ, (a * x^2 - a * x + 3 * x + 1 = 0) → a = 1 ∨ a = 9) :=
sorry

end quadratic_intersects_x_axis_only_once_l136_136685


namespace intersect_four_points_l136_136038

theorem intersect_four_points 
  (f : ℝ → ℝ)
  (h1 : ∀ x, x ≥ 0 → f x = |cos x|)
  (h2 : ∀ l : ℝ → ℝ, l 0 = 0 → (∃ x₁ x₂ x₃ x₄, x₁ < x₂ < x₃ < x₄ ∧ ∀ x ∈ {x₁, x₂, x₃, x₄}, f x = l x))
  (θ : ℝ)
  (h3 : θ ∈ Icc (3 * Real.pi / 2) (2 * Real.pi))
  (h4 : ∀ x, x ∈ Icc (3 * Real.pi / 2) (2 * Real.pi) → f x = cos x)
  (h5 : ∀ l : ℝ → ℝ, l 0 = 0 → θ = (λ x, -1 / (tan x)) θ) :
  (1 + θ^2) * sin (2 * θ) / θ = -2 := 
by
  sorry

end intersect_four_points_l136_136038


namespace intersection_complement_l136_136258

open set

def U := univ : set ℝ
def A := {x : ℝ | x < -1 ∨ x ≥ 2}
def B := {x : ℝ | 0 ≤ x ∧ x < 4}

theorem intersection_complement (x : ℝ) :
  x ∈ A ∩ (U \ B) ↔ x ∈ ((Iio (-1)) ∪ Ici 4) := by
  sorry

end intersection_complement_l136_136258


namespace candy_remainders_l136_136063

theorem candy_remainders {original : ℕ} {talitha : ℕ} {solomon : ℕ} {maya : ℕ}
  (h_original : original = 572)
  (h_talitha : talitha = 183)
  (h_solomon : solomon = 238)
  (h_maya : maya = 127) :
  original - (talitha + solomon + maya) = 24 :=
by
  rw [h_original, h_talitha, h_solomon, h_maya]
  norm_num
  sorry

end candy_remainders_l136_136063


namespace extreme_point_range_of_a_l136_136413

-- Defining the function f(x) and its domain
def f (x : ℝ) : ℝ := (1 / 2) * x^2 - 9 * log x

-- Predicate stating that the function has an extreme point in the interval [a-1, a+1]
def has_extreme_point_in_interval (a : ℝ) : Prop :=
  ∃ x ∈ set.Icc (a - 1) (a + 1), has_deriv_at f  x 0

-- The theorem to prove the required range for a
noncomputable def range_of_a : set ℝ := set.Ioo 2 4

theorem extreme_point_range_of_a (a : ℝ) :
  has_extreme_point_in_interval a ↔ a ∈ range_of_a :=
sorry

end extreme_point_range_of_a_l136_136413


namespace coin_flip_probability_difference_l136_136946

theorem coin_flip_probability_difference :
  let p1 := (nat.choose 5 4) * (1 / 2)^5
  let p2 := (1 / 2)^5
  (p1 - p2 = 1 / 8) :=
by
  let p1 := (nat.choose 5 4) * (1 / 2)^5
  let p2 := (1 / 2)^5
  sorry

end coin_flip_probability_difference_l136_136946


namespace alpha_delta_hours_l136_136209

theorem alpha_delta_hours (A D O : ℝ) (H1 : 1 / A + 1 / D + 3 / O = 1 / (A - 4))
  (H2 : A - 4 = 15) (H3 : D - 2 = 15) (H4 : Z := O / 2) : 
  ∃ h : ℝ, h = 323 / 36 :=
by
  sorry

end alpha_delta_hours_l136_136209


namespace problem1_problem2_l136_136530

-- Define the given conditions as variables and types
variables {R r : ℝ} (R_pos : 0 < R) (r_pos : 0 < r) (R_gt_r : R > r)
variables {O P B C A : Point}

-- Assume P is on the smaller circle's circumference
axiom P_on_smaller_circle : dist O P = r

-- Assume B is on the larger circle's circumference
axiom B_on_larger_circle : dist O B = R

-- Line BP intersects the larger circle at C
axiom BP_intersects_C : some_condition -- You need to fill out the exact formalizations

-- Line through P perpendicular to BP intersects the smaller circle at A
axiom perpendicular_BP : some_condition -- You need to fill out the exact formalizations

-- Define the proof problems
theorem problem1 (h1 : P_on_smaller_circle) (h2 : B_on_larger_circle) (h3 : BP_intersects_C) (h4 : perpendicular_BP) :
  dist B C ^ 2 + dist C A ^ 2 + dist A B ^ 2 = 6 * R ^ 2 + 2 * r ^ 2 :=
sorry

theorem problem2 (h1 : P_on_smaller_circle) (h2 : B_on_larger_circle) (h3 : BP_intersects_C) (h4 : perpendicular_BP) :
  ∀ Q, midpoint A B Q → Q ∈ circle_midpoint_OP (R / 2) :=
sorry

end problem1_problem2_l136_136530


namespace alex_box_jellybeans_l136_136154

theorem alex_box_jellybeans (b_volume: ℕ) (b_jellybeans: ℕ) (scale: ℕ) : Prop :=
  ((b_volume * (scale ^ 3)) * b_jellybeans) / b_volume = 421 :=
begin
  let bert_volume := b_volume,
  let alex_volume := b_volume * (scale ^ 3),
  let ratio := alex_volume / bert_volume,
  exact (ratio * b_jellybeans) = 421,
end

end alex_box_jellybeans_l136_136154


namespace has_only_one_zero_point_l136_136622

noncomputable def f (x a : ℝ) := (x - 1) * Real.exp x + (a / 2) * x^2

theorem has_only_one_zero_point (a : ℝ) (h : -Real.exp 1 ≤ a ∧ a ≤ 0) :
  ∃! x : ℝ, f x a = 0 :=
sorry

end has_only_one_zero_point_l136_136622


namespace min_value_of_a_l136_136672

theorem min_value_of_a {f : ℝ → ℝ} (a : ℝ) (h : ∀ x ∈ Ioo 1 2, deriv (λ x, a * real.exp x - real.log x) x ≥ 0) : 
  a ≥ real.exp (-1) :=
sorry

end min_value_of_a_l136_136672


namespace find_first_month_sale_l136_136488

theorem find_first_month_sale (s_2 s_3 s_4 s_5 s_6 avg : ℝ)
  (h2 : s_2 = 6500) (h3 : s_3 = 9855) (h4 : s_4 = 7230) (h5 : s_5 = 7000) (h6 : s_6 = 11915) (h_avg : avg = 7500)
  : (∃ s_1, (s_1 + s_2 + s_3 + s_4 + s_5 + s_6) / 6 = avg ∧ s_1 = 2500) :=
by {
  have total_sales : s_2 + s_3 + s_4 + s_5 + s_6 = 42500,
  calc
    s_2 + s_3 + s_4 + s_5 + s_6 = 6500 + 9855 + 7230 + 7000 + 11915 : by rw [h2, h3, h4, h5, h6]
    ... = 42500 : sorry,
  have total_avg_sales : 6 * avg = 45000,
  calc
    6 * avg = 6 * 7500 : by rw h_avg
    ... = 45000 : sorry,
  have s_1_val : 45000 - 42500 = 2500,
  calc
    45000 - 42500 = 2500 : sorry,
  use 2500,
  split,
  calc
    (2500 + s_2 + s_3 + s_4 + s_5 + s_6) / 6 = (45000 - 42500 + s_2 + s_3 + s_4 + s_5 + s_6) / 6 : by rw s_1_val
    ... = 45000 / 6 : sorry
    ... = 7500 : by rw h_avg,
  exact rfl
}

end find_first_month_sale_l136_136488


namespace combined_weight_prism_pyramid_l136_136310

def volume_prism (h : ℝ) (b : ℝ) : ℝ :=
  b * b * h

def volume_pyramid (b : ℝ) (h : ℝ) : ℝ :=
  (b * b * h) / 3

def weight_structure (volume : ℝ) (density : ℝ) : ℝ :=
  volume * density

theorem combined_weight_prism_pyramid :
  let h_prism := 8
  let b_prism := 2
  let h_pyramid := 3
  let b_pyramid := 1.5
  let density := 2700
  volume_prism b_prism h_prism * density + volume_pyramid b_pyramid h_pyramid * density = 92475 := 
by 
  sorry

end combined_weight_prism_pyramid_l136_136310


namespace sequence_formula_l136_136319

def sequence (n : ℕ) : ℕ :=
  if n = 1 then 4 else
  if n = 2 then 12 else
  if n > 2 then
    Nat.gcd ((sequence (n - 1)) ^ 2 - 4) ((sequence (n - 2)) ^ 2 + 3 * (sequence (n - 2)))
  else 0

theorem sequence_formula (n : ℕ) : sequence n = 4 * (2 ^ n - 1) :=
  sorry

end sequence_formula_l136_136319


namespace solve_problem_for_m_n_l136_136554

theorem solve_problem_for_m_n (m n : ℕ) (h₀ : m > 0) (h₁ : n > 0) (h₂ : m * (n + m) = n * (n - m)) :
  ((∃ h : ℕ, m = (2 * h + 1) * h ∧ n = (2 * h + 1) * (h + 1)) ∨ 
   (∃ h : ℕ, h > 0 ∧ m = 2 * h * (4 * h^2 - 1) ∧ n = 2 * h * (4 * h^2 + 1))) := 
sorry

end solve_problem_for_m_n_l136_136554


namespace angle_E_measure_of_parallelogram_l136_136840

theorem angle_E_measure_of_parallelogram (EFGH : Type)
  [parallelogram EFGH]
  (angle_FGH : ℝ) (h1 : angle_FGH = 70) :
  angle_E = 110 :=
by sorry

end angle_E_measure_of_parallelogram_l136_136840


namespace customers_left_tip_l136_136139

-- Definition of the given conditions
def initial_customers : ℕ := 29
def added_customers : ℕ := 20
def customers_didnt_tip : ℕ := 34

-- Lean 4 statement proving that the number of customers who did leave a tip (answer) equals 15
theorem customers_left_tip : (initial_customers + added_customers - customers_didnt_tip) = 15 :=
by
  sorry

end customers_left_tip_l136_136139


namespace triangle_inequality_a2_lt_ab_ac_l136_136575

theorem triangle_inequality_a2_lt_ab_ac {a b c : ℝ} (h1 : a < b + c) (h2 : 0 < a) : a^2 < a * b + a * c := 
sorry

end triangle_inequality_a2_lt_ab_ac_l136_136575


namespace exists_quadratic_polynomial_distinct_remainders_l136_136177

theorem exists_quadratic_polynomial_distinct_remainders :
  ∃ (a b c : ℤ), 
    (¬ (2014 ∣ a)) ∧ 
    (∀ x y : ℤ, (1 ≤ x ∧ x ≤ 2014) ∧ (1 ≤ y ∧ y ≤ 2014) → x ≠ y → 
      (1007 * x^2 + 1008 * x + c) % 2014 ≠ (1007 * y^2 + 1008 * y + c) % 2014) :=
  sorry

end exists_quadratic_polynomial_distinct_remainders_l136_136177


namespace lilith_caps_collection_l136_136357

noncomputable def monthlyCollectionYear1 := 3
noncomputable def monthlyCollectionAfterYear1 := 5
noncomputable def christmasCaps := 40
noncomputable def yearlyCapsLost := 15
noncomputable def totalYears := 5

noncomputable def totalCapsCollectedByLilith :=
  let firstYearCaps := monthlyCollectionYear1 * 12
  let remainingYearsCaps := monthlyCollectionAfterYear1 * 12 * (totalYears - 1)
  let christmasCapsTotal := christmasCaps * totalYears
  let totalCapsBeforeLosses := firstYearCaps + remainingYearsCaps + christmasCapsTotal
  let lostCapsTotal := yearlyCapsLost * totalYears
  let totalCapsAfterLosses := totalCapsBeforeLosses - lostCapsTotal
  totalCapsAfterLosses

theorem lilith_caps_collection : totalCapsCollectedByLilith = 401 := by
  sorry

end lilith_caps_collection_l136_136357


namespace find_suitable_pairs_l136_136449

-- Definition of a suitable pair
def suitable_pair (a b : ℕ) : Prop :=
  (a + b) ∣ (a * b)

-- Definition of the 12 pairs
def pairs : List (ℕ × ℕ) :=
  [(3, 6), (4, 12), (5, 20), (6, 30), (7, 42), (8, 56), (9, 72), (10, 90), (11, 110), (12, 132), (13, 156), (14, 168)]

-- The statement to prove
theorem find_suitable_pairs :
  (∀ p ∈ pairs, suitable_pair p.1 p.2) ∧
  (pairs.map Prod.fst).Nodup ∧
  (pairs.map Prod.snd).Nodup ∧
  Nat.max (pairs.map Prod.fst ++ pairs.map Prod.snd) = 168 :=
by
  -- Proof is omitted
  sorry

end find_suitable_pairs_l136_136449


namespace shop_combinations_l136_136748

theorem shop_combinations :
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let keyboard_and_mouse_sets := 4
  let headphone_and_mouse_sets := 5
  (keyboard_and_mouse_sets * headphones) + (headphone_and_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 :=
by
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let keyboard_and_mouse_sets := 4
  let headphone_and_mouse_sets := 5
  have case1 : keyboard_and_mouse_sets * headphones = 4 * 9, by sorry
  have case2 : headphone_and_mouse_sets * keyboards = 5 * 5, by sorry
  have case3 : headphones * mice * keyboards = 9 * 13 * 5, by sorry
  show (keyboard_and_mouse_sets * headphones) + (headphone_and_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646,
  by rw [case1, case2, case3]; exact sorry

end shop_combinations_l136_136748


namespace marcia_wardrobe_cost_l136_136830

theorem marcia_wardrobe_cost :
  let skirt_price := 20
  let blouse_price := 15
  let pant_price := 30
  let num_skirts := 3
  let num_blouses := 5
  let num_pants := 2
  let pant_offer := buy_1_get_1_half
  let skirt_cost := num_skirts * skirt_price
  let blouse_cost := num_blouses * blouse_price
  let pant_full_price := pant_price
  let pant_half_price := pant_price / 2
  let pant_cost := pant_full_price + pant_half_price
  let total_cost := skirt_cost + blouse_cost + pant_cost
  total_cost = 180 :=
by
  sorry -- proof is omitted

end marcia_wardrobe_cost_l136_136830


namespace minimum_value_of_a_l136_136666

theorem minimum_value_of_a (a : ℝ) (h : ∀ x ∈ set.Ioo 1 2, ae^x - 1/x ≥ 0) : a ≥ e⁻¹ :=
sorry

end minimum_value_of_a_l136_136666


namespace area_ADE_l136_136179

-- Given conditions: ABC is an equilateral triangle with the specified area, and the trisected angle
variables {A B C D E : Type*}
variables {s : ℝ} {areaABC : ℝ} [noncomputable]

-- Definition of an equilateral triangle with area 18√3 square units
def equilateral_triangle (A B C : Type) (s : ℝ) (areaABC : ℝ) : Prop :=
  areaABC = 18 * Real.sqrt 3 ∧ s = 6 * Real.sqrt 2

-- Definition to capture the trisection of angle BAC and the segments
def trisect_angle_and_segments (D E : Type) : Prop :=
  ∠ BAC = 60 ∧ BD = DE ∧ DE = EC ∧
  BD = 2 * Real.sqrt 2 ∧ 
  DE = 2 * Real.sqrt 2 ∧ 
  EC = 2 * Real.sqrt 2

-- Problem statement: Prove that the area of triangle ADE is approximately 2.5712 square units.
theorem area_ADE (A B C D E : Type) (s : ℝ) (areaABC : ℝ) 
  [h1 : equilateral_triangle A B C s areaABC] 
  [h2 : trisect_angle_and_segments D E] :
  area_TRIANGLE ADE = 2.5712 := sorry

end area_ADE_l136_136179


namespace minimum_value_of_a_l136_136640

theorem minimum_value_of_a (a : ℝ) :
  (∀ x ∈ (Ioo 1 2), ae^x - ln x ≥ 0) ↔ a ≥ exp(-1) := sorry

end minimum_value_of_a_l136_136640


namespace real_values_x_l136_136543

theorem real_values_x (x y : ℝ) :
  (3 * y^2 + 5 * x * y + x + 7 = 0) →
  (5 * x + 6) * (5 * x - 14) ≥ 0 →
  x ≤ -6 / 5 ∨ x ≥ 14 / 5 :=
by
  sorry

end real_values_x_l136_136543


namespace limit_exists_l136_136839

open Nat

-- Define the sequence a_n = (1 + 1/n)^n
def a : ℕ → ℝ
| n => (1 + 1 / (n : ℝ)) ^ n

-- State the properties of the sequence
lemma a_monotonically_increasing : ∀ n : ℕ, a n ≤ a (n + 1) := sorry

lemma a_bounded : ∀ n : ℕ, a n < 3 := sorry

-- Conclude that the limit exists
theorem limit_exists : ∃ L : ℝ, filter.tendsto a filter.at_top (nhds L) :=
begin
  -- Use the monotone convergence theorem
  apply filter.tendsto_of_monotone,
  apply a_monotonically_increasing,
  apply filter.bdd_above_of_forall_le 3,
  assume n,
  apply a_bounded,
end

end limit_exists_l136_136839


namespace archer_spend_on_arrows_per_week_l136_136511

theorem archer_spend_on_arrows_per_week:
  ∀ (shots_per_day : ℕ) (days_per_week : ℕ) (recovery_rate : ℝ) (cost_per_arrow : ℝ) (team_pay_rate : ℝ),
  shots_per_day = 200 →
  days_per_week = 4 →
  recovery_rate = 0.20 →
  cost_per_arrow = 5.5 →
  team_pay_rate = 0.70 →
  let total_shots := shots_per_day * days_per_week in
  let recovered_arrows := recovery_rate * total_shots in
  let arrows_to_replace := total_shots - recovered_arrows in
  let total_cost := arrows_to_replace * cost_per_arrow in
  let archer_pay_rate := 1 - team_pay_rate in
  archer_pay_rate * total_cost = 1056 :=
by
  intros shots_per_day days_per_week recovery_rate cost_per_arrow team_pay_rate 
  assume h_shots_per_day h_days_per_week h_recovery_rate h_cost_per_arrow h_team_pay_rate,
  let total_shots := shots_per_day * days_per_week,
  let recovered_arrows := recovery_rate * total_shots,
  let arrows_to_replace := total_shots - recovered_arrows,
  let total_cost := arrows_to_replace * cost_per_arrow,
  let archer_pay_rate := 1 - team_pay_rate,
  have h_total_shots : total_shots = 800 := by sorry,
  have h_recovered_arrows : recovered_arrows = 160 := by sorry,
  have h_arrows_to_replace : arrows_to_replace = 640 := by sorry,
  have h_total_cost : total_cost = 3520 := by sorry,
  have h_archer_pay_rate : archer_pay_rate = 0.30 := by sorry,
  calc
    archer_pay_rate * total_cost
        = 0.30 * 3520 : by sorry
    ... = 1056 : by sorry

end archer_spend_on_arrows_per_week_l136_136511


namespace part1_part2_part3_l136_136428

variable (balls boxes : Finset ℕ)
variable (h_balls : balls.card = 5)
variable (h_boxes : boxes.card = 5)

open Finset

noncomputable def num_ways_to_put_balls_in_boxes : ℕ := 
  (boxes.card)^(balls.card)

noncomputable def num_ways_one_empty_box : ℕ := 
  (boxes.card.choose 1) * (boxes.card - 1).choose (balls.card - 1)

noncomputable def num_ways_two_empty_boxes : ℕ := 
  ((boxes.card.choose 2) * (2.choose 2) + 
   (boxes.card.choose 3) * (1.choose 1)) * boxes.card.choose 3

theorem part1 : 
  num_ways_to_put_balls_in_boxes balls boxes = 3125 := sorry

theorem part2 : 
  num_ways_one_empty_box balls boxes = 1200 := sorry

theorem part3 : 
  num_ways_two_empty_boxes balls boxes = 1500 := sorry

end part1_part2_part3_l136_136428


namespace group_count_l136_136133

theorem group_count (sample_capacity : ℕ) (frequency : ℝ) (h_sample_capacity : sample_capacity = 80) (h_frequency : frequency = 0.125) : sample_capacity * frequency = 10 := 
by
  sorry

end group_count_l136_136133


namespace ones_digit_of_powers_sum_l136_136079

theorem ones_digit_of_powers_sum : 
  (∑ k in Finset.range (2017 + 1), k ^ 2017) % 10 = 3 :=
by
  sorry

end ones_digit_of_powers_sum_l136_136079


namespace intersection_points_form_line_l136_136876

theorem intersection_points_form_line :
  ∀ (x y : ℝ), ((x * y = 12) ∧ ((x^2 / 16) + (y^2 / 36) = 1)) →
  ∃ (x1 x2 : ℝ) (y1 y2 : ℝ), (x, y) = (x1, y1) ∨ (x, y) = (x2, y2) ∧ (x2 - x1) * (y2 - y1) = x1 * y1 - x2 * y2 :=
by
  sorry

end intersection_points_form_line_l136_136876


namespace max_product_with_859_l136_136446

def digits := {3, 5, 6, 8, 9}

theorem max_product_with_859 :
  ∃ d e, {d, e} ⊆ digits ∧ d ≠ 8 ∧ e ≠ 5 ∧ d ≠ e ∧ (859 * (10 * d + e) = 86738) :=
sorry

end max_product_with_859_l136_136446


namespace constant_term_is_35_l136_136569

noncomputable def constant_term_in_binomial_expansion (n : ℕ) : ℕ :=
  (Nat.choose n 3)

theorem constant_term_is_35 :
  ∃ (n : ℕ), 
    (2 * Nat.choose n 2 = Nat.choose n 1 + Nat.choose n 3) ∧
    constant_term_in_binomial_expansion n = 35 :=
by {
  use 7,
  simp only [constant_term_in_binomial_expansion, Nat.factorial, Nat.choose, mul_eq_mul_left_iff],
  split,
  {
    norm_num,
    apply Eq.symm,
    exact Nat.choose_succ_succ_eq n 2 3 
  },
  { norm_num }
}

end constant_term_is_35_l136_136569


namespace window_width_l136_136865

theorem window_width :
  ∀ (length width height : ℕ)
    (cost_per_sqft total_cost : ℕ)
    (door_length door_width : ℕ)
    (num_windows window_height : ℕ)
    (w : ℕ),
    length = 25 ∧ width = 15 ∧ height = 12 ∧ cost_per_sqft = 9 ∧ total_cost = 8154 ∧
    door_length = 6 ∧ door_width = 3 ∧ num_windows = 3 ∧ window_height = 3 →
    2 * (length + width) * height - (door_length * door_width + num_windows * (w * window_height)) = 
    (total_cost / cost_per_sqft) →
    w = 4 := by
  intro length width height cost_per_sqft total_cost door_length door_width num_windows window_height w
  intro h1 h2
  cases h1 with h1_1 h1_rem
  cases h1_1
  cases h1_rem with h1_3
  cases h1_3
  cases h1_5
  cases h1_5_1 with h1_6
  cases h1_5_2 with h1_7
  cases h1_7 with h1_8
  cases h1_8 with h1_9
  sorry

end window_width_l136_136865


namespace goldfish_in_first_tank_l136_136378

-- Definitions of conditions
def num_fish_third_tank : Nat := 10
def num_fish_second_tank := 3 * num_fish_third_tank
def num_fish_first_tank := num_fish_second_tank / 2
def goldfish_and_beta_sum (G : Nat) : Prop := G + 8 = num_fish_first_tank

-- Theorem to prove the number of goldfish in the first fish tank
theorem goldfish_in_first_tank (G : Nat) (h : goldfish_and_beta_sum G) : G = 7 :=
by
  sorry

end goldfish_in_first_tank_l136_136378


namespace sum_of_squares_le_neg_nmM_l136_136041

noncomputable theory
open_locale classical

theorem sum_of_squares_le_neg_nmM {n : ℕ} (a : fin n → ℝ) (m M : ℝ) 
  (h1 : ∑ i, a i = 0)
  (h2 : m = finset.min' (finset.univ.image a) (by simp [nat.succ_pos']))
  (h3 : M = finset.max' (finset.univ.image a) (by simp [nat.succ_pos'])) :
  ∑ i, (a i)^2 ≤ -n * m * M :=
sorry

end sum_of_squares_le_neg_nmM_l136_136041


namespace sum_interior_angles_equal_diagonals_l136_136408

theorem sum_interior_angles_equal_diagonals (n : ℕ) (h : n = 4 ∨ n = 5) :
  (n - 2) * 180 = 360 ∨ (n - 2) * 180 = 540 :=
by sorry

end sum_interior_angles_equal_diagonals_l136_136408


namespace slips_prob_ratio_l136_136549

open Nat
open Rat

def slips := finset.range 10 + 1

def num_slips (n : nat) : nat :=
  if n ∈ finset.range 5 + 1 then 5 else if n ∈ finset.range 6 + 5 then 3 else 0

def total_ways : ℕ := nat.choose 50 4

def r : rat := (5 * nat.choose 5 4 : ℕ) / total_ways

def s : rat := 
  (finset.sum (finset.range 5 + 1) (λ c, (finset.sum (finset.range 6 + 5) (λ d, (c ≠ d) * (nat.choose 5 2 * nat.choose 3 2 : ℕ)))) : ℕ) / total_ways

theorem slips_prob_ratio : s / r = 30 :=
  sorry

end slips_prob_ratio_l136_136549


namespace infinite_geometric_series_sum_l136_136545

-- Definitions of first term and common ratio
def a : ℝ := 1 / 5
def r : ℝ := 1 / 2

-- Proposition to prove that the sum of the geometric series is 2/5
theorem infinite_geometric_series_sum : (a / (1 - r)) = (2 / 5) :=
by 
  -- Sorry is used here as a placeholder for the proof.
  sorry

end infinite_geometric_series_sum_l136_136545


namespace total_journey_cost_l136_136797

variable (rental_cost : ℝ)
variable (discount_rate : ℝ)
variable (gas_gallons : ℝ)
variable (gas_price_per_gallon : ℝ)
variable (driving_cost_per_mile : ℝ)
variable (miles_to_destination : ℝ)
variable (extra_miles : ℝ)
variable (toll_fees : ℝ)
variable (parking_cost_per_day : ℝ)
variable (parking_days : ℝ)
variable (meals_lodging_per_day : ℝ)
variable (meals_lodging_days : ℝ)

# noncomputable example:

theorem total_journey_cost :
  let rental_cost := 150
  let discount_rate := 0.15
  let gas_gallons := 8
  let gas_price_per_gallon := 3.50
  let driving_cost_per_mile := 0.50
  let miles_to_destination := 320
  let extra_miles := 50
  let toll_fees := 15
  let parking_cost_per_day := 20
  let parking_days := 3
  let meals_lodging_per_day := 70
  let meals_lodging_days := 2
  
  let discounted_rental := rental_cost * (1 - discount_rate)
  let gas_cost := gas_gallons * gas_price_per_gallon
  let total_driving_miles := miles_to_destination + extra_miles
  let driving_cost := total_driving_miles * driving_cost_per_mile
  let total_parking_cost := parking_cost_per_day * parking_days
  let total_meals_lodging_cost := meals_lodging_per_day * meals_lodging_days
  
  let total_cost := discounted_rental + gas_cost + driving_cost + toll_fees + total_parking_cost + total_meals_lodging_cost
  
  show Prop, total_cost = 555.50 := by
  sorry

end total_journey_cost_l136_136797


namespace correct_propositions_l136_136579

variables {m n : Line} {α β : Plane}

/-- Given m and n are two different lines, and α and β are two different planes,
consider the following propositions:
1. If α ∩ β = m, n ⊆ α, n ⊥ m, then α ⊥ β.
2. If m ⊥ α, m ⊥ β, then α ∥ β.
3. If m ⊥ α, n ⊥ β, m ⊥ n, then α ⊥ β.
4. If m ∥ α, n ∥ β, m ∥ n, then α ∥ β.

The correct propositions are 2 and 3.
-/
theorem correct_propositions (h1 : m ≠ n) (h2 : α ≠ β) :
  (∀ (m ⊥ α) (m ⊥ β), α ∥ β) ∧
  (∀ (m ⊥ α) (n ⊥ β) (m ⊥ n), α ⊥ β) :=
sorry

end correct_propositions_l136_136579


namespace function_passes_through_fixed_point_l136_136570

noncomputable def fixed_point := (1, 1)

theorem function_passes_through_fixed_point (a : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) : 
  ∀ x y, (y = 2 * a ^ |x - 1| - 1) ∧ (x = 1) → (x, y) = fixed_point :=
by
  intros x y h
  sorry

-- specifying that the function y = 2a^|x-1| -1 passes through the point (1, 1) when x = 1 and a > 0 and a ≠ 1

end function_passes_through_fixed_point_l136_136570


namespace ratio_eq_23_over_28_l136_136956

theorem ratio_eq_23_over_28 (a b : ℚ) (h : (12 * a - 5 * b) / (14 * a - 3 * b) = 4 / 7) : 
  a / b = 23 / 28 := 
sorry

end ratio_eq_23_over_28_l136_136956


namespace squares_cover_area_l136_136024

theorem squares_cover_area (side_length : ℕ) (h1 : side_length = 12)
  (is_congruent : ∃(ABCD EFGH : Set), congruent ABCD EFGH ∧ SqAB.side = side_length ∧ SqEF.side = side_length )
  (point_G_positioned : ∃(G : point), G ∈ SqAB ∧ G ∈ SqEF ) :
  covered_area = side_length * side_length := 
sorry

end squares_cover_area_l136_136024


namespace determinant_zero_l136_136181

noncomputable def A (α β : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![
    ![0, Real.cos α, Real.sin α],
    ![-Real.cos α, 0, Real.cos β],
    ![-Real.sin α, -Real.cos β, 0]
  ]

theorem determinant_zero (α β : ℝ) : Matrix.det (A α β) = 0 := 
  sorry

end determinant_zero_l136_136181


namespace coin_toss_probability_l136_136458

theorem coin_toss_probability :
  let p := 0.5 in
  let n := 3 in
  let k := 1 in
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k)) = 0.375 :=
by
  sorry

end coin_toss_probability_l136_136458


namespace selling_prices_maximize_profit_l136_136997

-- Definitions for the conditions
def total_items : ℕ := 200
def budget : ℤ := 5000
def cost_basketball : ℤ := 30
def cost_volleyball : ℤ := 24
def selling_price_ratio : ℚ := 3 / 2
def school_purchase_basketballs_value : ℤ := 1800
def school_purchase_volleyballs_value : ℤ := 1500
def basketballs_fewer_than_volleyballs : ℤ := 10

-- Part 1: Proof of selling prices
theorem selling_prices (x : ℚ) :
  (school_purchase_volleyballs_value / x - school_purchase_basketballs_value / (x * selling_price_ratio) = basketballs_fewer_than_volleyballs)
  → ∃ (basketball_price volleyball_price : ℚ), basketball_price = 45 ∧ volleyball_price = 30 :=
by
  sorry

-- Part 2: Proof of maximizing profit
theorem maximize_profit (a : ℕ) :
  (cost_basketball * a + cost_volleyball * (total_items - a) ≤ budget)
  → ∃ optimal_a : ℕ, (optimal_a = 33 ∧ total_items - optimal_a = 167) :=
by
  sorry

end selling_prices_maximize_profit_l136_136997


namespace cos_sum_diff_l136_136837

theorem cos_sum_diff (α β : ℝ) : 
  cos (α + β) * cos (α - β) = cos α ^ 2 - sin β ^ 2 :=
  sorry

end cos_sum_diff_l136_136837


namespace rate_of_old_machine_is_100_l136_136128

-- Define the rate of the old machine
def rate_of_old_machine (R : ℝ) (time_hours : ℝ) (total_bolts : ℝ) : Prop :=
  let rate_new_machine := 150
  in total_bolts = (R + rate_new_machine) * time_hours

-- Main theorem to prove
theorem rate_of_old_machine_is_100 :
  rate_of_old_machine 100 (84 / 60) 350 :=
by 
  -- Derived from the problem's constraints
  sorry

end rate_of_old_machine_is_100_l136_136128


namespace total_flowers_bouquets_l136_136313

-- Define the number of tulips Lana picked
def tulips : ℕ := 36

-- Define the number of roses Lana picked
def roses : ℕ := 37

-- Define the number of extra flowers Lana picked
def extra_flowers : ℕ := 3

-- Prove that the total number of flowers used by Lana for the bouquets is 76
theorem total_flowers_bouquets : (tulips + roses + extra_flowers) = 76 :=
by
  sorry

end total_flowers_bouquets_l136_136313


namespace smallest_side_of_triangle_l136_136042

theorem smallest_side_of_triangle (s : ℕ) : 
  7.5 + s > 12 ∧ 7.5 + 12 > s ∧ 12 + s > 7.5 → s = 5 :=
by sorry

end smallest_side_of_triangle_l136_136042


namespace period_of_f_l136_136918

def f (x : ℝ) : ℝ := sin x - cos x

theorem period_of_f : ∀ x : ℝ, f (x + 2 * π) = f x :=
by
  simp [f, sin_add, cos_add, sin_eq_sin_sub_cos, cos_eq_sin_add_cos]
  sorry

end period_of_f_l136_136918


namespace general_term_sum_Tn_l136_136244

variable {n : ℕ} (hn : 0 < n)
variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Given condition
def Sn (n : ℕ) : ℝ := -a n - (1 / 2)^(n-1) + 2

-- Part 1: Prove that the general term of the sequence is a_n = n / 2^n
theorem general_term (hS : ∀ n, S n = Sn n) : ∀ n, a n = n / 2^n := 
sorry

-- Part 2: Let c_n = (n + 1) / n * a_n and T_n = c_1 + c_2 + ... + c_n
def cn (n : ℕ) : ℝ := (n + 1) / n * (a n)
def Tn (n : ℕ) : ℝ := ∑ i in finset.range n, (cn (i + 1))

-- Prove that T_n = 3 - (n + 3) / 2^n
theorem sum_Tn (ha: ∀ n, a n = n / 2^n) : ∀ n, Tn n = 3 - (n + 3) / 2^n :=
sorry

end general_term_sum_Tn_l136_136244


namespace find_standard_eq_of_ellipse_find_area_of_triangle_l136_136223

section Geometry

def focus_shared (ellipse_eq : ℝ → ℝ → Prop) : Prop :=
  ∃ x y, y^2 = 4 * sqrt 2 * x ∧ ellipse_eq x y

def eccentricity (ellipse_eq : ℝ → ℝ → Prop) (e : ℝ) : Prop :=
  ∃ a b c, a > b ∧ b > 0 ∧ c = sqrt 2 ∧ e = c / a ∧ ellipse_eq = (λ x y, x^2 / a^2 + y^2 / b^2 = 1)

def standard_eq_ellipse : Prop :=
  ∃ a b, a = 2 ∧ b^2 = a^2 - (sqrt 2)^2 ∧ (b^2 = 2) ∧ (λ x y, x^2 / a^2 + y^2 / b^2 = 1)

def intersection_area (ellipse_eq : ℝ → ℝ → Prop) (P A B : ℝ × ℝ) (O : ℝ × ℝ) : Prop :=
  P = (0, 1) ∧ O = (0, 0) ∧ ∃ x1 y1 x2 y2, 
  A = (x1, y1) ∧ B = (x2, y2) ∧ 
  (-(x1) = 2 * x2) ∧ (1 - y1 = 2 * (y2 - 1)) ∧ 
  let k := (y1 - 1) / x1 in
  let area := 1 / 2 * abs (x1 + x2) * abs (x1 - x2) in
  area = 3 * sqrt 14 / 8

theorem find_standard_eq_of_ellipse :
  focus_shared ellipse_eq ∧ eccentricity ellipse_eq (√2 / 2) → 
  ∃ ellipse_eq, standard_eq_ellipse :=
sorry

theorem find_area_of_triangle :
  (∃ P A B O, intersection_area ellipse_eq P A B O) →
  ∃ area, area = 3 * sqrt 14 / 8 :=
sorry

end Geometry

end find_standard_eq_of_ellipse_find_area_of_triangle_l136_136223


namespace hunter_will_hit_fox_l136_136126

noncomputable def probability_hitting (distance: ℝ) : ℝ :=
  (100 / distance)^2 * (1/2)

def hunter_hit_probability :=
  let p1 := probability_hitting 100
  let p2 := probability_hitting 150
  let p3 := probability_hitting 200
  p1 + p2 + p3 - (p1 * p2) - (p1 * p3) - (p2 * p3) + (p1 * p2 * p3)

theorem hunter_will_hit_fox : hunter_hit_probability = (95 / 144) := by
  sorry

end hunter_will_hit_fox_l136_136126


namespace only_possible_triplet_l136_136317

noncomputable def sum_geom_series (k : ℕ) (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).sum (λ j, k^j)

noncomputable def b (k a n : ℕ) : ℕ :=
  a * (sum_geom_series k n)

theorem only_possible_triplet
  (k n1 n2 n3 a1 a2 a3 : ℕ)
  (hk : k ≥ 2)
  (hn1 : n1 > 0)
  (hn2 : n2 > 0)
  (hn3 : n3 > 0)
  (ha1 : 1 ≤ a1 ∧ a1 < k)
  (ha2 : 1 ≤ a2 ∧ a2 < k)
  (ha3 : 1 ≤ a3 ∧ a3 < k)
  (h : b k a1 n1 * b k a2 n2 = b k a3 n3) :
  (n1 = 1 ∧ n2 = 1 ∧ n3 = 3) :=
begin
  sorry
end

end only_possible_triplet_l136_136317


namespace minimum_dominant_cells_l136_136860

def is_dominant_in_row {m n : ℕ} (board : ℕ → ℕ → bool) (i : ℕ) (j : ℕ) : Prop :=
  let row := fun k => board i k
  (∑ k in finset.range (2*n + 1), ite (row k = board i j) 1 0) > n

def is_dominant_in_column {m n : ℕ} (board : ℕ → ℕ → bool) (i : ℕ) (j : ℕ) : Prop :=
  let col := fun k => board k j
  (∑ k in finset.range (2*m + 1), ite (col k = board i j) 1 0) > m

theorem minimum_dominant_cells (m n : ℕ) 
  (board : ℕ → ℕ → bool)
  (h_row_dominant : ∀ i j, is_dominant_in_row board i j)
  (h_col_dominant : ∀ i j, is_dominant_in_column board i j) :
  ∃ (cells : finset (ℕ × ℕ)), cells.card ≥ m + n - 1 ∧ 
  ∀ (i j : ℕ), (i, j) ∈ cells → is_dominant_in_row board i j ∧ is_dominant_in_column board i j := 
by
  sorry

end minimum_dominant_cells_l136_136860


namespace sqrt3_mul_sqrt6_add_sqrt8_eq_5sqrt2_frac2_over_sqrt5_minus2_eq_2sqrt5_plus4_l136_136521

theorem sqrt3_mul_sqrt6_add_sqrt8_eq_5sqrt2 : sqrt 3 * sqrt 6 + sqrt 8 = 5 * sqrt 2 :=
  sorry

theorem frac2_over_sqrt5_minus2_eq_2sqrt5_plus4 : 2 / (sqrt 5 - 2) = 2 * sqrt 5 + 4 :=
  sorry

end sqrt3_mul_sqrt6_add_sqrt8_eq_5sqrt2_frac2_over_sqrt5_minus2_eq_2sqrt5_plus4_l136_136521


namespace composite_sum_of_squares_l136_136403

theorem composite_sum_of_squares (a b : ℤ) (h_roots : ∃ x1 x2 : ℕ, (x1 + x2 : ℤ) = -a ∧ (x1 * x2 : ℤ) = b + 1) :
  ∃ m n : ℕ, a^2 + b^2 = m * n ∧ 1 < m ∧ 1 < n :=
sorry

end composite_sum_of_squares_l136_136403


namespace blue_notes_per_red_note_l136_136793

-- Given conditions
def total_red_notes : ℕ := 5 * 6
def additional_blue_notes : ℕ := 10
def total_notes : ℕ := 100
def total_blue_notes := total_notes - total_red_notes

-- Proposition that needs to be proved
theorem blue_notes_per_red_note (x : ℕ) : total_red_notes * x + additional_blue_notes = total_blue_notes → x = 2 := by
  intro h
  sorry

end blue_notes_per_red_note_l136_136793


namespace expansion_of_a_plus_b_pow_4_expansion_of_a_plus_b_pow_5_computation_of_formula_l136_136014

section
variables (a b : ℚ)

theorem expansion_of_a_plus_b_pow_4 :
  (a + b) ^ 4 = a ^ 4 + 4 * a ^ 3 * b + 6 * a ^ 2 * b ^ 2 + 4 * a * b ^ 3 + b ^ 4 :=
sorry

theorem expansion_of_a_plus_b_pow_5 :
  (a + b) ^ 5 = a ^ 5 + 5 * a ^ 4 * b + 10 * a ^ 3 * b ^ 2 + 10 * a ^ 2 * b ^ 3 + 5 * a * b ^ 4 + b ^ 5 :=
sorry

theorem computation_of_formula :
  2^4 + 4*2^3*(-1/3) + 6*2^2*(-1/3)^2 + 4*2*(-1/3)^3 + (-1/3)^4 = 625 / 81 :=
sorry
end

end expansion_of_a_plus_b_pow_4_expansion_of_a_plus_b_pow_5_computation_of_formula_l136_136014


namespace mean_temp_is_correct_median_temp_is_correct_l136_136379

def temperatures : List ℚ := [-6, -3, -3, -2, 0, 4, 5]

def mean_temperature (temps : List ℚ) : ℚ :=
  (List.sum temps) / (temps.length)

def median_temperature (temps : List ℚ) : ℚ :=
  let sorted_temps := temps.qsort (≤)
  sorted_temps.get! ((sorted_temps.length) / 2)

theorem mean_temp_is_correct : mean_temperature temperatures = -5 / 7 := by
  sorry

theorem median_temp_is_correct : median_temperature temperatures = -2 := by
  sorry

end mean_temp_is_correct_median_temp_is_correct_l136_136379


namespace intersection_union_l136_136803

def M := {0, 1, 2, 4, 5, 7} : Set ℕ
def N := {1, 4, 6, 8, 9} : Set ℕ
def P := {4, 7, 9} : Set ℕ

theorem intersection_union :
  (M ∩ N) ∪ (M ∩ P) = ({1, 4, 7} : Set ℕ) :=
by {
  sorry
}

end intersection_union_l136_136803


namespace third_square_perimeter_l136_136875

theorem third_square_perimeter (p1 p2 : ℝ) (h1 : p1 = 40) (h2 : p2 = 32) : 
  let a1 := (p1 / 4) ^ 2,
      a2 := (p2 / 4) ^ 2,
      a3 := a1 - a2,
      s3 := sqrt a3,
      p3 := 4 * s3
  in p3 = 24 :=
by
  -- Let binding introduces variables a1, a2, a3, s3, p3
  sorry

end third_square_perimeter_l136_136875


namespace parallelogram_area_l136_136382

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]

def parallel_area (a b : V) : ℝ :=
  (a ⨯ b).norm

theorem parallelogram_area
  (a b : V) 
  (h : parallel_area a b = 8) :
  parallel_area (3 • a + 4 • b) (a + 2 • b) = 16 := 
sorry

end parallelogram_area_l136_136382


namespace fair_coin_difference_l136_136929

noncomputable def binom_coeff (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem fair_coin_difference :
  let p := (1 : ℚ) / 2 in
  let p_1 := binom_coeff 5 4 * p^4 * (1 - p) in
  let p_2 := p^5 in
  p_1 - p_2 = 1 / 8 :=
by
  sorry

end fair_coin_difference_l136_136929


namespace minimum_value_of_a_l136_136627

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - Real.log x

theorem minimum_value_of_a (a : ℝ) :
  (∀ x ∈ Ioo 1 2, deriv (f a) x ≥ 0) → a ≥ Real.exp ⁻¹ :=
by
  simp [f]
  sorry

end minimum_value_of_a_l136_136627


namespace find_slope_of_line_q_l136_136359

theorem find_slope_of_line_q
  (k : ℝ)
  (h₁ : ∀ (x y : ℝ), (y = 3 * x + 5) → (y = k * x + 3) → (x = -4 ∧ y = -7))
  : k = 2.5 :=
sorry

end find_slope_of_line_q_l136_136359


namespace distinct_ordered_pairs_sum_of_reciprocals_l136_136695

theorem distinct_ordered_pairs_sum_of_reciprocals :
  { (m, n) : ℕ × ℕ // 0 < m ∧ 0 < n ∧ (1 / m : ℚ) + (1 / n) = 1 / 5 }.to_finset.card = 3 :=
sorry

end distinct_ordered_pairs_sum_of_reciprocals_l136_136695


namespace find_f_at_2_l136_136213

variable (f : ℝ → ℝ)
variable (k : ℝ)
variable (h1 : ∀ x, f x = x^3 + 3 * x * f'' 2)
variable (h2 : f' 2 = 12 + 3 * f' 2)

theorem find_f_at_2 : f 2 = -28 :=
by
  sorry

end find_f_at_2_l136_136213


namespace fraction_irreducibility_l136_136206

theorem fraction_irreducibility (n : ℤ) : Int.gcd (21 * n + 4) (14 * n + 3) = 1 := 
by 
  sorry

end fraction_irreducibility_l136_136206


namespace log_7_x_l136_136276

-- Define the log_16 4 and log_4 16
def log_16_4 := Real.logb 16 4
def log_4_16 := Real.logb 4 16

-- Define x
def x := (log_16_4 ^ log_4_16)

-- Define the proof statement
theorem log_7_x : Real.logb 7 x = -2 * Real.logb 7 2 :=
  by
    -- The proof would go here, but we skip it with sorry.
    sorry

end log_7_x_l136_136276


namespace sum_of_numbers_l136_136884

theorem sum_of_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 222) 
  (h2 : a * b + b * c + c * a = 131) : 
  a + b + c = 22 := 
by 
  sorry

end sum_of_numbers_l136_136884


namespace simplify_expression_l136_136847

theorem simplify_expression :
  ((5 * 10^7) / (2 * 10^2)) + (4 * 10^5) = 650000 := 
by
  sorry

end simplify_expression_l136_136847


namespace range_of_a_l136_136252

theorem range_of_a (a : ℝ) :
  (∀ x ∈ set.Icc (1 / Real.exp 1) (Real.exp 1), 2 * Real.log x - x^2 + a = 0) →
  a ∈ set.Ioo 1 (2 + 1 / Real.exp 2) :=
sorry

end range_of_a_l136_136252


namespace quadrilateral_tangents_rhombus_l136_136483

variable {K L M N : Point}
variable (A B C D : Point)
variable (AB : Line) (BC : Line) (CD : Line) (DA : Line)
variable (S : Circle) -- Incircle of quadrilateral ABCD
variable (S1 S2 S3 S4 : Circle) -- Incircles of respective triangles
variable (O1 O2 O3 O4 : Point) -- Centers of the incircles
variable (P Q R T : Line)

-- Conditions (definitions)
def quadrilateral_inscribed_circle (A B C D : Point) (K L M N : Point) (S : Circle) :=
  touches S DA K ∧ touches S AB L ∧ touches S BC M ∧ touches S CD N

def inscircles (AKL BLM CMN DNK: Triangle) (S1 S2 S3 S4 : Circle) :=
  incircle S1 AKL ∧ incircle S2 BLM ∧ incircle S3 CMN ∧ incircle S4 DNK

def common_tangents (S1 S2 S3 S4 : Circle) (P Q R T : Line) :=
  common_tangent P S1 S2 ∧ common_tangent Q S2 S3 ∧ 
  common_tangent R S3 S4 ∧ common_tangent T S4 S1

-- Lean statement for the proof problem
theorem quadrilateral_tangents_rhombus 
  (A B C D : Point) (K L M N : Point) 
  (AB BC CD DA : Line) (S: Circle) 
  (S1 S2 S3 S4 : Circle) (O1 O2 O3 O4 : Point)
  (P Q R T : Line) 
  (h1 : quadrilateral_inscribed_circle A B C D K L M N S)
  (h2 : inscircles ⟨∠AKL, ∠BLM, ∠CMN, ∠DNK⟩ S1 S2 S3 S4)
  (h3 : common_tangents S1 S2 S3 S4 P Q R T): 
  is_rhombus (formed_quadrilateral P Q R T) := sorry

end quadrilateral_tangents_rhombus_l136_136483


namespace tan_double_angle_l136_136607

theorem tan_double_angle (α : ℝ) (h1 : α ∈ Ioo (π / 2) π) (h2 : sin (2 * α) = - sin α) : tan (2 * α) = sqrt 3 := 
by
  sorry

end tan_double_angle_l136_136607


namespace distinct_pairs_count_l136_136697

theorem distinct_pairs_count : 
  { (m, n) : ℕ × ℕ // 0 < m ∧ 0 < n ∧ (1 / (m:ℚ) + 1 / (n:ℚ) = 1 / 5) }.to_finset.card = 3 := 
by {
  sorry,
}

end distinct_pairs_count_l136_136697


namespace fair_coin_flip_difference_l136_136922

theorem fair_coin_flip_difference :
  let P1 := 5 * (1 / 2)^5;
  let P2 := (1 / 2)^5;
  P1 - P2 = 9 / 32 :=
by
  sorry

end fair_coin_flip_difference_l136_136922


namespace sequence_properties_l136_136220

variable (a b : ℕ → ℝ)

-- Definition of the sequences as given in the conditions
def S (n : ℕ) : ℝ := n^2 + n
def a (n : ℕ) : ℝ := if n = 1 then 2 else S n - S (n - 1)
def b (n : ℕ) : ℝ := (1 / 2)^(a n) + n

-- Theorem statement to prove
theorem sequence_properties (n : ℕ) :
  (∀ n : ℕ, a n = 2 * n) ∧
  (∑ i in Finset.range n, b (i + 1) = (1 / 3) * (1 - (1 / 4)^n) + (n * (n + 1)) / 2) :=
by
  sorry

end sequence_properties_l136_136220


namespace domain_of_func_l136_136032

noncomputable def func (x : ℝ) : ℝ := 1 / (2 * x - 1)

theorem domain_of_func :
  ∀ x : ℝ, x ≠ 1 / 2 ↔ ∃ y : ℝ, y = func x := sorry

end domain_of_func_l136_136032


namespace max_value_exp_l136_136820

theorem max_value_exp (x y z : ℝ) (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) (h_constraint : x^2 + y^2 + z^2 = 1) :
  2 * x * y * Real.sqrt 8 + 7 * y * z + 5 * x * z ≤ 23.0219 :=
sorry

end max_value_exp_l136_136820


namespace minimum_value_of_a_l136_136641

theorem minimum_value_of_a (a : ℝ) :
  (∀ x ∈ (Ioo 1 2), ae^x - ln x ≥ 0) ↔ a ≥ exp(-1) := sorry

end minimum_value_of_a_l136_136641


namespace tan_subtraction_l136_136281

variable (α β : ℝ)

def tan_alpha : ℝ := 9
def tan_beta : ℝ := 5

theorem tan_subtraction :
  tan (α - β) = 2 / 23 :=
by
  have h_tan_alpha : tan α = tan_alpha := sorry
  have h_tan_beta : tan β = tan_beta := sorry
  sorry

end tan_subtraction_l136_136281


namespace num_even_digits_base_9_529_l136_136198

def base_9_representation (n : ℕ) : ℕ := 6 * 81 + 4 * 9 + 7

theorem num_even_digits_base_9_529 : 
  let digits := [6, 4, 9] in
  (digits.filter (λ x => x % 2 = 0)).length = 2 :=
begin
  have h : base_9_representation 529 = 6 * 81 + 4 * 9 + 7 := rfl,
  let digits := [6, 4, 9],
  have hd1 : (6 % 2 = 0) := rfl,
  have hd2 : (4 % 2 = 0) := rfl,
  have hd3 : (9 % 2 = 1) := rfl,
  show (digits.filter (λ x => x % 2 = 0)).length = 2,
  sorry
end

end num_even_digits_base_9_529_l136_136198


namespace area_swept_by_small_square_l136_136053

-- Define the side lengths of the squares and the path traversed
def large_square_side : ℝ := 10
def small_square_side : ℝ := 1

-- Define the proof statement
theorem area_swept_by_small_square :
  let movements := ["Along AB", "Along BD", "Along DC"] in
  large_square_side = 10 ∧ small_square_side = 1 →
  (area_swept_by_small_square movements large_square_side small_square_side = 36) :=
begin
  intros _ _,
  sorry
end

end area_swept_by_small_square_l136_136053


namespace area_of_triangle_formed_by_lines_l136_136411

theorem area_of_triangle_formed_by_lines (x y : ℝ) (h1 : y = x) (h2 : x = -5) :
  let base := 5
  let height := 5
  let area := (1 / 2 : ℝ) * base * height
  area = 12.5 := 
by
  sorry

end area_of_triangle_formed_by_lines_l136_136411


namespace value_of_reciprocal_sum_roots_l136_136822

theorem value_of_reciprocal_sum_roots :
  let p, q, r, s be roots of (x^4 + 10*x^3 + 20*x^2 + 15*x + 6 = 0) in
  let a := p * q
  let b := p * r
  let c := p * s
  let d := q * r
  let e := q * s
  let f := r * s
  (1 / a) + (1 / b) + (1 / c) + (1 / d) + (1 / e) + (1 / f) = 10 / 3 :=
by
  sorry

end value_of_reciprocal_sum_roots_l136_136822


namespace isosceles_triangle_angle_difference_l136_136031

variables {α β : ℝ}
variables (ABC : Triangle)

-- Define a function to represent an isosceles triangle
def is_isosceles_triangle (t : Triangle) : Prop :=
  (t.α = t.β ∨ t.β = t.γ ∨ t.γ = t.α)

-- Define the condition of the nine-point center lying on the circumcircle
def nine_point_center_on_circumcircle (t : Triangle) : Prop :=
  let D := midpoint t.B t.C;
  let E := midpoint t.C t.A;
  let F := midpoint t.A t.B in
  let nine_point_center := midpoint t.orthocenter t.circumcenter in
  t.circumcircle.contains nine_point_center

-- Define the angles α and β as given in the condition
noncomputable def angle_α (t : Triangle) : ℝ := t.α
noncomputable def angle_β (t : Triangle) : ℝ := t.β

-- Theorem statement
theorem isosceles_triangle_angle_difference (t : Triangle) 
  (h_isosceles : is_isosceles_triangle t)
  (h_nine_point_center : nine_point_center_on_circumcircle t)
  (h_larger_angle : t.α > t.β) :
  t.α - t.β = 0 :=
sorry

end isosceles_triangle_angle_difference_l136_136031


namespace carY_average_speed_l136_136162

-- Definitions based on the conditions
def carX_speed : ℝ := 35 -- miles per hour
def carX_travel_time_before : ℝ := 48 / 60 -- in hours
def carX_distance_after : ℝ := 245 -- miles
def carX_distance_before : ℝ := carX_speed * carX_travel_time_before -- distance before car Y started
def carX_total_distance : ℝ := carX_distance_before + carX_distance_after -- total distance traveled by car X
def carY_travel_time : ℝ := carX_distance_after / carX_speed -- in hours

-- The statement to be proved
theorem carY_average_speed :
  carY_travel_time * carX_speed = carX_distance_after / carY_travel_time :=
begin
  -- The proof is not required as per the instructions
  sorry -- Placeholder for the actual proof
end

end carY_average_speed_l136_136162


namespace new_average_after_multiplying_by_5_l136_136030

theorem new_average_after_multiplying_by_5 (numbers : Fin 35 → ℝ) (h : (∑ i, numbers i) / 35 = 25) :
  (∑ i, 5 * numbers i) / 35 = 125 :=
by
  sorry

end new_average_after_multiplying_by_5_l136_136030


namespace group_sizes_l136_136720

def ticket_price := 50
def holiday_discount := 0.2
def non_holiday_discount := 0.4

-- For group size x (> 10) on non-holiday
def non_holiday_cost (x : ℕ) (h : x > 10) : ℕ := 30 * x

-- For group size x (> 10) on holiday
def holiday_cost (x : ℕ) (h : x > 10) : ℕ := 40 * x + 100

-- Given conditions
def total_cost (a b : ℕ) : Prop := 
  holiday_cost a (by simp [a > 10, nat.succ_gt_self]) + non_holiday_cost b (by simp [b > 10, nat.succ_gt_self]) = 1840
  ∧ a + b = 50

theorem group_sizes (a b : ℕ) : total_cost a b → a = 24 ∧ b = 26 :=
sorry

end group_sizes_l136_136720


namespace minimum_a_for_monotonic_increasing_on_interval_l136_136652

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp x - Real.log x

noncomputable def f_prime (a x : ℝ) : ℝ := a * Real.exp x - 1 / x

def min_value_of_a : ℝ := Real.exp (-1)

theorem minimum_a_for_monotonic_increasing_on_interval :
  (∀ x ∈ Set.Ioo 1 2, 0 ≤ f_prime a x) ↔ a ≥ min_value_of_a :=
by
  sorry

end minimum_a_for_monotonic_increasing_on_interval_l136_136652


namespace student_history_mark_l136_136136

theorem student_history_mark
  (math_score : ℕ)
  (desired_average : ℕ)
  (third_subject_score : ℕ)
  (history_score : ℕ) :
  math_score = 74 →
  desired_average = 75 →
  third_subject_score = 70 →
  (math_score + history_score + third_subject_score) / 3 = desired_average →
  history_score = 81 :=
by
  intros h_math h_avg h_third h_equiv
  sorry

end student_history_mark_l136_136136


namespace max_min_values_interval_l136_136043

noncomputable def quadratic_function (x : ℝ) : ℝ := -x^2 + 4 * x + 5

theorem max_min_values_interval : 
  ∃ (max_val min_val : ℝ), 
  (∀ x ∈ set.Icc (1 : ℝ) 4, quadratic_function x ≤ max_val) ∧ 
  (∀ x ∈ set.Icc (1 : ℝ) 4, quadratic_function x ≥ min_val) ∧ 
  max_val = quadratic_function 2 ∧ min_val = quadratic_function 4 :=
by
  sorry

end max_min_values_interval_l136_136043


namespace min_value_of_a_l136_136678

theorem min_value_of_a {f : ℝ → ℝ} (a : ℝ) (h : ∀ x ∈ Ioo 1 2, deriv (λ x, a * real.exp x - real.log x) x ≥ 0) : 
  a ≥ real.exp (-1) :=
sorry

end min_value_of_a_l136_136678


namespace speed_conversion_l136_136532

theorem speed_conversion (speed_m_s : ℚ) (conversion_factor : ℚ) :
  speed_m_s = 8 / 26 → conversion_factor = 3.6 →
  speed_m_s * conversion_factor = 1.1077 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end speed_conversion_l136_136532


namespace total_journey_time_l136_136723

-- Definitions and conditions
def river_speed := 3.5 -- in km/hr
def distance_upstream := 72.5 -- in km
def boat_speed_still_water := 8.25 -- in km/hr

-- Proof statement
theorem total_journey_time : 
  let upstream_speed := boat_speed_still_water - river_speed,
      downstream_speed := boat_speed_still_water + river_speed,
      time_upstream := distance_upstream / upstream_speed,
      time_downstream := distance_upstream / downstream_speed
  in time_upstream + time_downstream = 21.4334 :=
by 
  sorry

end total_journey_time_l136_136723


namespace evaluate_f_g_at_3_l136_136813

def f (x : ℝ) : ℝ := x^2 + 2
def g (x : ℝ) : ℝ := 3 * x + 2

theorem evaluate_f_g_at_3 : f (g 3) = 123 := by
  sorry

end evaluate_f_g_at_3_l136_136813


namespace total_apples_count_l136_136417

-- Definitions based on conditions
def red_apples := 16
def green_apples := red_apples + 12
def total_apples := green_apples + red_apples

-- Statement to prove
theorem total_apples_count : total_apples = 44 := 
by
  sorry

end total_apples_count_l136_136417


namespace minimum_value_of_a_l136_136644

theorem minimum_value_of_a (a : ℝ) :
  (∀ x ∈ (Ioo 1 2), ae^x - ln x ≥ 0) ↔ a ≥ exp(-1) := sorry

end minimum_value_of_a_l136_136644


namespace continuous_at_4_l136_136347

def f (x : ℝ) (b : ℝ) : ℝ :=
if x ≤ 4 then 5 * x^2 + 7 else b * x + 3

theorem continuous_at_4 (b : ℝ) : (f 4 b = (5 * 4^2 + 7)) → b = 21 :=
by
  calc
    f 4 b = 5 * 4^2 + 7 : by simp [f] -- Given and specific to the condition x ≤ 4
    ... = 87 : by norm_num -- Simplifying the given expression
    ... = 4 * b + 3 : sorry -- Matches the value for continuity at x = 4
    ... = 21 : sorry -- Solving for b we get the needed b = 21

end continuous_at_4_l136_136347


namespace derivative_of_f_tangent_line_at_P_l136_136680

def f (x : ℝ) : ℝ := x^2 + x * log x

theorem derivative_of_f : deriv f = λ x, 2 * x + log x + 1 := by
  sorry

theorem tangent_line_at_P : ∀ (x : ℝ), f (1) = 1 → deriv f 1 = 3 → (f 1 = 1 → deriv f 1 = 3 → f x = 3 * x - 2) := by
  sorry

end derivative_of_f_tangent_line_at_P_l136_136680


namespace proj_b_v_l136_136336

-- Define orthogonality
def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

-- Define vector projections
def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let c := (u.1 * v.1 + u.2 * v.2) / (v.1 * v.1 + v.2 * v.2)
  (c * v.1, c * v.2)

-- Given vectors a and b and their properties
variables (a b : ℝ × ℝ) (v : ℝ × ℝ)
variable h_ab_orthogonal : orthogonal a b
variable h_proj_a : proj v a = (-4/5, -8/5)

-- The goal to prove
theorem proj_b_v :
  proj v b = (24/5, -2/5) := sorry

end proj_b_v_l136_136336


namespace shop_combinations_l136_136751

theorem shop_combinations :
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let keyboard_and_mouse_sets := 4
  let headphone_and_mouse_sets := 5
  (keyboard_and_mouse_sets * headphones) + (headphone_and_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 :=
by
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let keyboard_and_mouse_sets := 4
  let headphone_and_mouse_sets := 5
  have case1 : keyboard_and_mouse_sets * headphones = 4 * 9, by sorry
  have case2 : headphone_and_mouse_sets * keyboards = 5 * 5, by sorry
  have case3 : headphones * mice * keyboards = 9 * 13 * 5, by sorry
  show (keyboard_and_mouse_sets * headphones) + (headphone_and_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646,
  by rw [case1, case2, case3]; exact sorry

end shop_combinations_l136_136751


namespace min_value_of_a_l136_136657

noncomputable def min_a_monotone_increasing : ℝ :=
  let f : ℝ → ℝ := λ x a, a * exp x - log x in 
  let a_min : ℝ := exp (-1) in
  a_min

theorem min_value_of_a 
  {f : ℝ → ℝ} 
  {a : ℝ}
  (mono_incr : ∀ x ∈ Ioo 1 2, 0 ≤ deriv (λ x, a * exp x - log x) x) :
  a ≥ exp (-1) :=
begin
  sorry
end

end min_value_of_a_l136_136657


namespace subtract_fractions_l136_136025

theorem subtract_fractions : (18 / 42 - 3 / 8) = 3 / 56 :=
by
  sorry

end subtract_fractions_l136_136025


namespace coefficient_x2_correct_l136_136861

noncomputable def coefficient_x2_expansion : ℕ := 
  let expansion1 := (1 + x) ^ 6
  let expansion2 := (1 / (x ^ 2)) * (1 + x) ^ 6
  let coeff1 := (binom 6 2 : ℕ)  -- Coefficient of x^2 in (1 + x)^6
  let coeff2 := (binom 6 4 : ℕ)  -- Coefficient of x^4 in (1 + x)^6 multiplied by 1/x^2
  coeff1 + coeff2

theorem coefficient_x2_correct : coefficient_x2_expansion = 30 := 
by 
  sorry

end coefficient_x2_correct_l136_136861


namespace g_is_odd_g_range_m_range_l136_136253

-- Given function f(x)
def f (x : ℝ) : ℝ := (2 * 4^x) / (4^x + 4^(-x))

-- Derived function g(x)
def g (x : ℝ) : ℝ := (4^x - 4^(-x)) / (4^x + 4^(-x))

-- Proof that g(x) is odd
theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

-- Proof that the range of g(x) is (-1, 1)
theorem g_range : ∀ y, g y ∈ Set.Ioo (-1 : ℝ) 1 := by
  sorry

-- Proof that for g(m) + g(m-2) > 0, m > 1
theorem m_range (m : ℝ) (h : g m + g (m - 2) > 0) : m > 1 := by
  sorry

end g_is_odd_g_range_m_range_l136_136253


namespace total_games_in_season_l136_136422

-- Define the constants according to the conditions
def number_of_teams : ℕ := 25
def games_per_pair : ℕ := 15

-- Define the mathematical statement we want to prove
theorem total_games_in_season :
  let round_robin_games := (number_of_teams * (number_of_teams - 1)) / 2 in
  let total_games := round_robin_games * games_per_pair in
  total_games = 4500 :=
by
  sorry

end total_games_in_season_l136_136422


namespace expression_multiple_of_five_l136_136205

theorem expression_multiple_of_five (n : ℕ) (h : n ≥ 10) : 
  (∃ k : ℕ, (n + 2) * (n + 1) = 5 * k) :=
sorry

end expression_multiple_of_five_l136_136205


namespace fg_minus_gf_eq_six_l136_136027

def f (x : ℝ) : ℝ := 4 * x - 6

def g (x : ℝ) : ℝ := x / 2 + 3

theorem fg_minus_gf_eq_six (x : ℝ) : f (g x) - g (f x) = 6 := 
by
  sorry

end fg_minus_gf_eq_six_l136_136027


namespace parallel_lines_l136_136321

open EuclideanGeometry

noncomputable def orthocenter (A B C : Point) (M : Point) : Prop :=
  ∃ H1 H2 H3 : Line, 
    altitude A B C H1 ∧ 
    altitude B A C H2 ∧ 
    altitude C A B H3 ∧ 
    ∀ (P : Point), 
      intersects P H1 H2 H3 → P = M

noncomputable def altitude_intersects_circumcircle_twice (A B C : Point) (A' : Point) : Prop :=
  ∃ H : Line,
    altitude A B C H ∧
    intersects_circumcircle_twice A B C H A'

noncomputable def line_intersects_circumcircle_twice (M E A'' A''' : Point) (A B C : Point) : Prop :=
  ∃ H : Line,
    line_through M E H ∧
    intersects_circumcircle_twice A B C H A'' ∧
    intersects_circumcircle_twice A B C H A'''

theorem parallel_lines (A B C M A' A'' A''' : Point) (E : Point) :
  orthocenter A B C M →
  altitude_intersects_circumcircle_twice A B C A' →
  midpoint (B, C) E →
  line_intersects_circumcircle_twice M E A'' A''' A B C →
  parallel (line_through A A'') (line_through B C) ∨
  parallel (line_through A' A''') (line_through B C) :=
by
  sorry

end parallel_lines_l136_136321


namespace incorrect_proposition_d_l136_136246

-- Define the necessary types and properties
variable {Plane : Type} {Line : Type}

-- Define the perpendicular and parallel relations
variables (perp : Line → Line → Prop)
variables (perpendicular_to_plane : Line → Plane → Prop)
variables (parallel_between_planes : Plane → Plane → Prop)
variables (parallel_between_line_and_plane : Line → Plane → Prop)
variables (intersection_of_planes : Plane → Plane → Line)

-- Contextual conditions based on the problem statement
variables (α β : Plane) (m n l : Line)

-- Define the conditions as hypotheses
hypothesis h1 : perp m n
hypothesis h2 : perpendicular_to_plane m α
hypothesis h3 : parallel_between_line_and_plane n β
hypothesis h4 : intersection_of_planes α β = l

-- Problem Statement: Prove that α and β are not necessarily perpendicular
theorem incorrect_proposition_d : ¬ (parallel_between_planes α β ∨ α = β) := by
  -- Details of the proof are omitted
  sorry

end incorrect_proposition_d_l136_136246


namespace smallest_number_divisible_by_11_and_remainder_1_l136_136083

theorem smallest_number_divisible_by_11_and_remainder_1 {n : ℕ} :
  (n % 2 = 1) ∧ 
  (n % 3 = 1) ∧ 
  (n % 4 = 1) ∧ 
  (n % 5 = 1) ∧ 
  (n % 11 = 0) -> n = 121 :=
sorry

end smallest_number_divisible_by_11_and_remainder_1_l136_136083


namespace solve_problem_l136_136611

open Complex

noncomputable def problem_statement (a : ℝ) : Prop :=
  abs ((a : ℂ) + I) / abs I = 2
  
theorem solve_problem {a : ℝ} : problem_statement a → a = Real.sqrt 3 :=
by
  sorry

end solve_problem_l136_136611


namespace proj_of_b_eq_v_diff_proj_of_a_l136_136329

noncomputable theory

variables (a b : ℝ × ℝ) (v : ℝ × ℝ)

def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (u.1 * v.1 + u.2 * v.2) / (u.1 * u.1 + u.2 * u.2)
  (scalar * u.1, scalar * u.2)

theorem proj_of_b_eq_v_diff_proj_of_a
  (h₀ : orthogonal a b)
  (h₁ : proj a ⟨4, -2⟩ = ⟨-4/5, -8/5⟩)
  : proj b ⟨4, -2⟩ = ⟨24/5, -2/5⟩ :=
sorry

end proj_of_b_eq_v_diff_proj_of_a_l136_136329


namespace angle_is_pi_over_4_l136_136691

noncomputable def angle_between_vectors {α : Type*} [inner_product_space ℝ α] (a b : α) : ℝ :=
  real.arccos ((inner_product_space.inner 𝕜 a b) / (∥a∥ * ∥b∥))

theorem angle_is_pi_over_4 {α : Type*} [inner_product_space ℝ α] 
  (a b : α) (h₁ : ∥a∥ = 1) (h₂ : ∥b∥ = real.sqrt 2)
  (h₃ : inner_product_space.inner ℝ a (a - b) = 0) :
  angle_between_vectors a b = real.pi / 4 :=
begin
  sorry
end

end angle_is_pi_over_4_l136_136691


namespace sun_radius_in_scientific_notation_l136_136400

def radius_in_kilometers : ℕ := 696_000

def kilometers_to_meters (km : ℕ) : ℕ := km * 1_000

def scientific_notation (n : ℕ) : String :=
  let norm := n.to_float / (10^8)
  s!"{norm} × 10^8"

theorem sun_radius_in_scientific_notation :
  scientific_notation (kilometers_to_meters radius_in_kilometers) = "6.96 × 10^8" :=
by
  sorry

end sun_radius_in_scientific_notation_l136_136400


namespace num_real_z_10_l136_136886

-- Definitions
def is_30th_root_unity (z : ℂ) : Prop := z ^ 30 = 1

-- Main theorem statement
theorem num_real_z_10 (z : ℂ) (hz : is_30th_root_unity z) : 
  (finset.univ.filter (λ z : ℂ, is_30th_root_unity z ∧ (z ^ 10).re = (z ^ 10))) .card = 12 :=
sorry

end num_real_z_10_l136_136886


namespace road_trip_gasoline_needed_l136_136993

/-- Define the distances and fuel efficiencies. -/

def highway_fuel_efficiency : ℝ := 40 -- kilometers per gallon on highways
def city_fuel_efficiency : ℝ := 30 -- kilometers per gallon in the city

def first_destination_distance : ℝ := 65 -- kilometers on a highway
def second_destination_distance : ℝ := 90 -- kilometers in the city
def third_destination_distance : ℝ := 45 -- kilometers on a highway

/-- Calculate the total gallons of gasoline needed for the road trip. -/
def total_gallons : ℝ :=
  (first_destination_distance / highway_fuel_efficiency) +
  (second_destination_distance / city_fuel_efficiency) +
  (third_destination_distance / highway_fuel_efficiency)

/-- The proof statement: Prove total_gallons is equal to 5.75 given the conditions. -/
theorem road_trip_gasoline_needed : total_gallons = 5.75 :=
  by
  sorry

end road_trip_gasoline_needed_l136_136993


namespace count_numbers_with_at_most_two_digits_l136_136267

theorem count_numbers_with_at_most_two_digits (count : ℕ) :
  (∀ n, 0 < n ∧ n < 1000 →
    (∀ d1 d2 d, (d = n.digits → ∃ d1 d2, d1 ≠ d2 ∧ d.all (λ x, x = d1 ∨ x = d2)))) →
  count = 855 :=
sorry

end count_numbers_with_at_most_two_digits_l136_136267


namespace set_A_infinite_and_set_B_finite_l136_136348

-- Define set A as the set of rectangles with an area of 1
def A : set (ℝ × ℝ) := {p | p.1 * p.2 = 1}

-- Define set B as the set of equilateral triangles with an area of 1
def B : set (ℝ × ℝ × ℝ) := {p | p.1 = p.2 ∧ p.2 = p.3 ∧ (sqrt 3) / 4 * p.1^2 = 1}

-- The proof goal
theorem set_A_infinite_and_set_B_finite : (set.infinite A ∧ set.finite B) :=
by
 sorry

end set_A_infinite_and_set_B_finite_l136_136348


namespace annual_average_growth_rate_l136_136480

theorem annual_average_growth_rate (x : ℝ) (h : x > 0): 
  100 * (1 + x)^2 = 169 :=
sorry

end annual_average_growth_rate_l136_136480


namespace shop_combinations_l136_136745

theorem shop_combinations :
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let keyboard_and_mouse_sets := 4
  let headphone_and_mouse_sets := 5
  (keyboard_and_mouse_sets * headphones) + (headphone_and_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 :=
by
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let keyboard_and_mouse_sets := 4
  let headphone_and_mouse_sets := 5
  have case1 : keyboard_and_mouse_sets * headphones = 4 * 9, by sorry
  have case2 : headphone_and_mouse_sets * keyboards = 5 * 5, by sorry
  have case3 : headphones * mice * keyboards = 9 * 13 * 5, by sorry
  show (keyboard_and_mouse_sets * headphones) + (headphone_and_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646,
  by rw [case1, case2, case3]; exact sorry

end shop_combinations_l136_136745


namespace slope_angle_of_tangent_line_l136_136054

theorem slope_angle_of_tangent_line :
  (∀ x : ℝ, y x = (1/2) * x^2 + 2) →
  ∀ α : ℝ, (tan α = -1) → 
  (0 ≤ α ∧ α < π) → 
  α = 3 * π / 4 :=
sorry

end slope_angle_of_tangent_line_l136_136054


namespace fair_coin_flip_probability_difference_l136_136939

noncomputable def choose : ℕ → ℕ → ℕ
| _, 0 => 1
| 0, _ => 0
| n + 1, r + 1 => choose n r + choose n (r + 1)

theorem fair_coin_flip_probability_difference :
  let p := 1 / 2
  let p_heads_4_out_of_5 := choose 5 4 * p^4 * p
  let p_heads_5_out_of_5 := p^5
  p_heads_4_out_of_5 - p_heads_5_out_of_5 = 1 / 8 :=
by
  sorry

end fair_coin_flip_probability_difference_l136_136939


namespace power_function_value_l136_136285

theorem power_function_value :
  ∀ (f : ℝ → ℝ) (n : ℝ),
    (∀ x : ℝ, f x = x ^ n) →
    f 2 = (2 : ℝ) ^ n →
    f (1 / 2) = 4 :=
begin
  intros f n hfn h2,
  sorry
end

end power_function_value_l136_136285


namespace coin_flip_probability_difference_l136_136947

theorem coin_flip_probability_difference :
  let p1 := (nat.choose 5 4) * (1 / 2)^5
  let p2 := (1 / 2)^5
  (p1 - p2 = 1 / 8) :=
by
  let p1 := (nat.choose 5 4) * (1 / 2)^5
  let p2 := (1 / 2)^5
  sorry

end coin_flip_probability_difference_l136_136947


namespace ship_speed_l136_136005

theorem ship_speed 
  (D : ℝ)
  (h1 : (D/2) - 200 = D/3)
  (S := (D / 2) / 20):
  S = 30 :=
by
  -- proof here
  sorry

end ship_speed_l136_136005


namespace find_S13_l136_136243

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Conditions 
def arithmetic_seq_sum (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
∀ n, S n = (n / 2) * (a 1 + a n)

axiom condition (h1 : a 3 + a 5 + 2 * a 10 = 4)

-- The proof goal
theorem find_S13 (h1 : a 3 + a 5 + 2 * a 10 = 4) : S 13 = 13 :=
sorry

end find_S13_l136_136243


namespace trajectory_of_N_l136_136000

noncomputable def ellipse (a b : ℝ) : set (ℝ × ℝ) :=
  {p | let x := p.fst, y := p.snd in (x^2 / a^2 + y^2 / b^2 = 1)}

def foci1 (a b : ℝ) : ℝ × ℝ :=
  (-sqrt (a^2 - b^2), 0)

def foci2 (a b : ℝ) : ℝ × ℝ :=
  (sqrt (a^2 - b^2), 0)

theorem trajectory_of_N (a b : ℝ) (M : ℝ × ℝ) (hM : M ∈ ellipse a b) :
  let F1 := foci1 a b,
      F2 := foci2 a b,
      N := some_point -- Define as described in the conditions.
  in 
  (N.1^2 + N.2^2 = a^2) :=
by 
  sorry

end trajectory_of_N_l136_136000


namespace solve_for_b_l136_136026

def f (x : ℝ) : ℝ := x / 7 + 4
def g (x : ℝ) : ℝ := 5 - x

theorem solve_for_b : ∃ b : ℝ, f (g b) = 6 ∧ b = -9 := sorry

end solve_for_b_l136_136026


namespace ellipse_equation_find_slope_l136_136591

theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_vs : b = 1) (h_fl : 2*sqrt(3) = 2*sqrt(a^2 - b^2)) : 
  ∀ x y : ℝ, (x / a)^2 + (y / b)^2 = 1 ↔ x^2 / 4 + y^2 = 1 :=
by sorry

theorem find_slope (k : ℝ) (P : ℝ × ℝ) (hP : P = (-2, 1)) (h_condition : ∃ B C : ℝ × ℝ, B ≠ C ∧ ∀ t : ℝ, (k*t + B.2)*(k*t + C.2) = -k^2*4*t^2 - 8k*t - 4k^2 - 2 - B.1 + 2 + C.1 + 4f - B.1 / (1-f) - C.1 / (1-f) = 2 ) :
  k = -4 :=
by sorry

end ellipse_equation_find_slope_l136_136591


namespace lilith_caps_collection_l136_136355

theorem lilith_caps_collection :
  let first_year_caps := 3 * 12
  let subsequent_years_caps := 5 * 12 * 4
  let christmas_caps := 40 * 5
  let lost_caps := 15 * 5
  let total_caps := first_year_caps + subsequent_years_caps + christmas_caps - lost_caps
  total_caps = 401 := by
  sorry

end lilith_caps_collection_l136_136355


namespace probability_all_captains_selected_l136_136066

theorem probability_all_captains_selected :
  let teams := [6, 9, 10],
  let captains := 3,
  (1 / 3 : ℚ) * ((6 / (6 * 5 * 4)) + (6 / (9 * 8 * 7)) + (6 / (10 * 9 * 8))) = 177 / 12600 :=
by
  sorry

end probability_all_captains_selected_l136_136066


namespace store_purchase_ways_l136_136740

theorem store_purchase_ways :
  ∃ (headphones sets_hm sets_km keyboards mice : ℕ), 
  headphones = 9 ∧
  mice = 13 ∧
  keyboards = 5 ∧
  sets_km = 4 ∧
  sets_hm = 5 ∧
  (sets_km * headphones + sets_hm * keyboards + headphones * mice * keyboards = 646) :=
by
  use 9, 5, 4, 5, 13
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  sorry

end store_purchase_ways_l136_136740


namespace part1_part2_l136_136256

-- Part 1: When k = 1, prove that f(x) >= g(x) - x^2 / 2
theorem part1 (x : ℝ) : (exp x) ≥ (x + 1) :=
by
sorry

-- Part 2: If f(x) ≥ g(x), find the range of values for k
theorem part2 (k : ℝ) : (∀ x, (exp x) ≥ (k / 2 * x^2 + x + 1)) → k ∈ Iic 0 :=
by
sorry

end part1_part2_l136_136256


namespace basketball_court_width_l136_136397

variable (width length : ℕ)

-- Given conditions
axiom h1 : length = width + 14
axiom h2 : 2 * length + 2 * width = 96

-- Prove the width is 17 meters
theorem basketball_court_width : width = 17 :=
by {
  sorry
}

end basketball_court_width_l136_136397


namespace cube_inequality_of_greater_l136_136599

variable (a b : ℝ)

theorem cube_inequality_of_greater (h : a > b) : a^3 > b^3 :=
sorry

end cube_inequality_of_greater_l136_136599


namespace coin_flip_difference_l136_136953

/-- The positive difference between the probability of a fair coin landing heads up
exactly 4 times out of 5 flips and the probability of a fair coin landing heads up
5 times out of 5 flips is 1/8. -/
theorem coin_flip_difference :
  (5 * (1 / 2) ^ 5) - ((1 / 2) ^ 5) = (1 / 8) :=
by
  sorry

end coin_flip_difference_l136_136953


namespace find_length_of_brick_l136_136113

noncomputable def length_of_brick (x : ℝ) : Prop :=
  let wall_volume := 800 * 600 * 22.5 in
  let brick_volume := x * 11.25 * 6 in
  ((6400 * brick_volume) = wall_volume) → (x = 25)

theorem find_length_of_brick (x : ℝ) (hx: length_of_brick x) : x = 25 := by
  sorry

end find_length_of_brick_l136_136113


namespace molecular_weight_of_3_moles_l136_136452

namespace AscorbicAcid

def molecular_form : List (String × ℕ) := [("C", 6), ("H", 8), ("O", 6)]

def atomic_weight : String → ℝ
| "C" => 12.01
| "H" => 1.008
| "O" => 16.00
| _ => 0

noncomputable def molecular_weight (molecular_form : List (String × ℕ)) : ℝ :=
molecular_form.foldr (λ (x : (String × ℕ)) acc => acc + (x.snd * atomic_weight x.fst)) 0

noncomputable def weight_of_3_moles (mw : ℝ) : ℝ := mw * 3

theorem molecular_weight_of_3_moles :
  weight_of_3_moles (molecular_weight molecular_form) = 528.372 :=
by
  sorry

end AscorbicAcid

end molecular_weight_of_3_moles_l136_136452


namespace cube_strictly_increasing_l136_136602

theorem cube_strictly_increasing (a b : ℝ) (h : a > b) : a^3 > b^3 :=
sorry

end cube_strictly_increasing_l136_136602


namespace range_of_a_l136_136036

noncomputable def f (a : ℝ) : ℝ → ℝ
| x => if x < 1 then a^x else (a - 3) * x + 4 * a

theorem range_of_a (a : ℝ) (h : ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) : 0 < a ∧ a ≤ 3 / 4 :=
sorry

end range_of_a_l136_136036


namespace count_numbers_divisible_by_12_not_20_l136_136148

theorem count_numbers_divisible_by_12_not_20 : 
  let N := 2017
  let a := Nat.floor (N / 12)
  let b := Nat.floor (N / 60)
  a - b = 135 := by
    -- Definitions used
    let N := 2017
    let a := Nat.floor (N / 12)
    let b := Nat.floor (N / 60)
    -- The desired statement
    show a - b = 135
    sorry

end count_numbers_divisible_by_12_not_20_l136_136148


namespace simplify_expression_and_evaluate_l136_136374

theorem simplify_expression_and_evaluate (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) (h3 : a ≠ 3) :
  ( ( (a + 3) / (a^2 - 4) - a / (a^2 - a - 6) ) / ( (2a - 9) / (5a - 10) ) ) = 5 / (a^2 - a - 6) ∧
  (( ( (a + 3) / (a^2 - 4) - a / (a^2 - a - 6) ) / ( (2a - 9) / (5a - 10) ) ) = 5 / 14) :=
sorry

end simplify_expression_and_evaluate_l136_136374


namespace sum_of_oldest_and_youngest_l136_136406

section

variables (A B C D : ℕ)

theorem sum_of_oldest_and_youngest
  (h1 : A = 32)
  (h2 : A + B = 3 * (C + D))
  (h3 : C = D + 3)
  (h4 : A + B + C + D = 100) :
  let oldest := max A (max B (max C D)),
      youngest := min A (min B (min C D))
  in oldest + youngest = 54 :=
by
  sorry

end

end sum_of_oldest_and_youngest_l136_136406


namespace min_value_of_a_l136_136675

theorem min_value_of_a {f : ℝ → ℝ} (a : ℝ) (h : ∀ x ∈ Ioo 1 2, deriv (λ x, a * real.exp x - real.log x) x ≥ 0) : 
  a ≥ real.exp (-1) :=
sorry

end min_value_of_a_l136_136675


namespace symmetric_circle_equation_l136_136540

noncomputable def equation_of_symmetric_circle (C₁ : ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, C₁ x y ↔ x^2 + y^2 - 4 * x - 8 * y + 19 = 0

theorem symmetric_circle_equation :
  ∀ (C₁ : ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop),
  equation_of_symmetric_circle C₁ l →
  (∀ x y, l x y ↔ x + 2 * y - 5 = 0) →
  ∃ C₂ : ℝ → ℝ → Prop, (∀ x y, C₂ x y ↔ x^2 + y^2 = 1) :=
by
  intros C₁ l hC₁ hₗ
  sorry

end symmetric_circle_equation_l136_136540


namespace find_digit_in_subtraction_problem_l136_136302

theorem find_digit_in_subtraction_problem :
  ∃ (digit : ℕ), ∀ (a b c d : ℕ), (a = 4) ∧ (d = 7) ∧ (b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → (a * 100 + b * 10 + d) - 189 = 268 → b = 5 :=
by
  sorry

end find_digit_in_subtraction_problem_l136_136302


namespace seahawks_field_goals_l136_136065

-- Defining the conditions as hypotheses
def final_score_seahawks : ℕ := 37
def points_per_touchdown : ℕ := 7
def points_per_fieldgoal : ℕ := 3
def touchdowns_seahawks : ℕ := 4

-- Stating the goal to prove
theorem seahawks_field_goals : 
  (final_score_seahawks - touchdowns_seahawks * points_per_touchdown) / points_per_fieldgoal = 3 := 
by 
  sorry

end seahawks_field_goals_l136_136065


namespace optionB_correct_l136_136816

variables {L P : Type} [LinearOrder L] [LinearOrder P] 

-- Two different lines
variables (m n : L)
-- Two different planes
variables (α β : P)

-- Definitions of geometric relations
noncomputable def parallel (l : L) (p : P) := sorry
noncomputable def perpendicular (l : L) (p : P) := sorry

-- Conditions provided for option B
axiom condition1 : perpendicular m α
axiom condition2 : perpendicular n β
axiom condition3 : perpendicular α β

-- The statement to be proven
theorem optionB_correct : perpendicular m n :=
by sorry

end optionB_correct_l136_136816


namespace perp_tangents_l136_136240

theorem perp_tangents (a b : ℝ) (h : a + b = 5) (tangent_perp : ∀ x y : ℝ, x = 1 ∧ y = 1) :
  a / b = 1 / 3 :=
sorry

end perp_tangents_l136_136240


namespace volume_region_revolving_around_y_eq_x_l136_136823

-- Define the region R bounded by y = x and y = x^2
def region (x : ℝ) : Prop := y = x ∧ y = x^2

-- Define the integral to compute the volume of the region R when revolved around y = x
noncomputable def volumeOfRevolution : ℝ :=
  (Real.pi / 2) * ∫ x in 0..1, (x^2 - 2*x^3 + x^4) / 2

-- The final theorem to prove the volume is π/30
theorem volume_region_revolving_around_y_eq_x :
  volumeOfRevolution = Real.pi / 30 := 
sorry

end volume_region_revolving_around_y_eq_x_l136_136823


namespace heavy_tailed_permutations_count_l136_136495

open Finset

-- Define the set of numbers and the condition defining heavy-tailed permutations
def is_heavy_tailed (perm : List ℕ) : Prop :=
  perm.length = 5 ∧
  {1, 2, 3, 5, 6}.subset perm.to_finset ∧
  (perm.headI + perm.tail.headI < perm.tail.tail.tailI.headI + perm.tail.tail.tailI.tail.headI)

-- Count the number of heavy-tailed permutations
def count_heavy_tailed_permutations : ℕ :=
  (univ : Finset (List ℕ)).filter is_heavy_tailed).card

-- Statement to prove in Lean 4
theorem heavy_tailed_permutations_count :
  count_heavy_tailed_permutations = 40 :=
by
  sorry

end heavy_tailed_permutations_count_l136_136495


namespace sufficient_condition_l136_136271

theorem sufficient_condition (a b : ℝ) (h : a > b ∧ b > 0) : a + a^2 > b + b^2 :=
by
  sorry

end sufficient_condition_l136_136271


namespace initial_interval_for_root_bisection_l136_136448

noncomputable def f (x : ℝ) : ℝ := x^3 + 5

theorem initial_interval_for_root_bisection : f (-2) * f (1) < 0 :=
by {
  -- Given conditions
  have h1 : f (-2) = (-2)^3 + 5, by rfl,
  have h2 : f (-2) = -3, from calc
    f (-2) = (-2)^3 + 5 : rfl
         ... = -8 + 5 : by norm_num
         ... = -3 : by norm_num,
  have h3 : f (1) = (1)^3 + 5, by rfl,
  have h4 : f (1) = 6, from calc
    f (1) = (1)^3 + 5 : rfl
        ... = 1 + 5 : by norm_num
        ... = 6 : by norm_num,
  -- Satisfies bisection method condition
  show f (-2) * f (1) < 0, from calc
    f (-2) * f (1) = -3 * 6 : by { rw [h2, h4] }
                 ... = -18 : by norm_num
                 ... < 0 : by norm_num
}

end initial_interval_for_root_bisection_l136_136448


namespace total_cost_is_53_l136_136017

-- Defining the costs and quantities as constants
def sandwich_cost : ℕ := 4
def soda_cost : ℕ := 3
def num_sandwiches : ℕ := 7
def num_sodas : ℕ := 10
def discount : ℕ := 5

-- Get the cost of sandwiches purchased
def cost_of_sandwiches : ℕ := num_sandwiches * sandwich_cost

-- Get the cost of sodas purchased
def cost_of_sodas : ℕ := num_sodas * soda_cost

-- Calculate the total cost before discount
def total_cost_before_discount : ℕ := cost_of_sandwiches + cost_of_sodas

-- Calculate the total cost after discount
def total_cost_after_discount : ℕ := total_cost_before_discount - discount

-- The theorem stating that the total cost is 53 dollars
theorem total_cost_is_53 : total_cost_after_discount = 53 :=
by
  sorry

end total_cost_is_53_l136_136017


namespace oranges_count_indeterminate_l136_136061

variable (bananas_per_group : ℕ)
variable (groups_of_bananas : ℕ)
variable (total_bananas : ℕ)
variable (groups_of_oranges : ℕ)

-- Conditions
axiom h1 : total_bananas = 290
axiom h2 : groups_of_bananas = 2
axiom h3 : bananas_per_group = 145
axiom h4 : groups_of_oranges = 93

theorem oranges_count_indeterminate : ¬ ∃ oranges_count : ℕ, oranges_count = _ :=
by sorry

end oranges_count_indeterminate_l136_136061


namespace monotonic_decreasing_intervals_range_of_f_in_interval_transformation_description_l136_136248

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x + Real.cos x)^2 + 2 * Real.cos x^2

theorem monotonic_decreasing_intervals :
  ∀ k : ℤ, ∀ x : ℝ, x ∈ set.Icc (Real.pi / 8 + k * Real.pi) (5 * Real.pi / 8 + k * Real.pi) → 
  ( f (x + h) ≤ f x ) sorry

theorem range_of_f_in_interval :
  ∀ x : ℝ, x ∈ set.Icc 0 (Real.pi / 2) → 
  f x ∈ set.Icc 1 (Real.sqrt 2 + 2) sorry

theorem transformation_description :
  ∀ x : ℝ, 
  y = Real.sqrt 2 * Real.sin x →
  f (x / 2 - Real.pi / 8) + 2 = y
  sorry

end monotonic_decreasing_intervals_range_of_f_in_interval_transformation_description_l136_136248


namespace correct_statement_l136_136967

theorem correct_statement (x : ℝ) : 
  (∃ y : ℝ, y ≠ 0 ∧ y * x = 1 → x = 1 ∨ x = -1 ∨ x = 0) → false ∧
  (∃ y : ℝ, -y = y → y = 0 ∨ y = 1) → false ∧
  (abs x = x → x ≥ 0) → (x ^ 2 = 1 → x = 1 ∨ x = -1) :=
by
  sorry

end correct_statement_l136_136967


namespace promoting_cashback_beneficial_for_bank_cashback_in_rubles_preferable_l136_136974

-- Definitions of conditions
def attracts_new_clients (bank_promotes_cashback : Prop) : Prop :=
  bank_promotes_cashback → ∃ (new_clients : Prop), new_clients

def promotes_partnerships (bank_promotes_cashback : Prop) : Prop :=
  bank_promotes_cashback → ∃ (partnerships : Prop), partnerships

def enhances_competitiveness (bank_promotes_cashback : Prop) : Prop :=
  bank_promotes_cashback → ∃ (competitiveness : Prop), competitiveness

def liquidity_advantage (cashback_rubles : Prop) : Prop :=
  cashback_rubles → ∃ (liquidity : Prop), liquidity

def no_expiry_concerns (cashback_rubles : Prop) : Prop :=
  cashback_rubles → ∃ (no_expiry : Prop), no_expiry

def no_partner_limitations (cashback_rubles : Prop) : Prop :=
  cashback_rubles → ∃ (partner_limitations : Prop), ¬partner_limitations

-- Lean statements for the proof problems
theorem promoting_cashback_beneficial_for_bank (bank_promotes_cashback : Prop) :
  attracts_new_clients bank_promotes_cashback ∧
  promotes_partnerships bank_promotes_cashback ∧ 
  enhances_competitiveness bank_promotes_cashback →
  bank_promotes_cashback := 
sorry

theorem cashback_in_rubles_preferable (cashback_rubles : Prop) :
  liquidity_advantage cashback_rubles ∧
  no_expiry_concerns cashback_rubles ∧
  no_partner_limitations cashback_rubles →
  cashback_rubles :=
sorry

end promoting_cashback_beneficial_for_bank_cashback_in_rubles_preferable_l136_136974


namespace binom_coeff_divisibility_l136_136553

theorem binom_coeff_divisibility (m n : ℕ) (h1 : 1 < n) :
  (∀ k : ℕ, 1 ≤ k ∧ k < m → n ∣ Nat.choose m k) ↔
  ∃ (p : ℕ) (u : ℕ), p.prime ∧ m = p ^ u ∧ n = p :=
by
  sorry

end binom_coeff_divisibility_l136_136553


namespace num_ways_to_buy_three_items_l136_136770

-- Defining the conditions based on the problem statement
def num_headphones : ℕ := 9
def num_mice : ℕ := 13
def num_keyboards : ℕ := 5
def num_kb_mouse_sets : ℕ := 4
def num_hp_mouse_sets : ℕ := 5

-- Defining the theorem statement
theorem num_ways_to_buy_three_items :
  (num_kb_mouse_sets * num_headphones) + 
  (num_hp_mouse_sets * num_keyboards) + 
  (num_headphones * num_mice * num_keyboards) = 646 := 
  by
  sorry

end num_ways_to_buy_three_items_l136_136770


namespace right_angled_triangle_solution_l136_136301

noncomputable def right_angled_triangle_hypotenuse_division 
  (A B C C₁ C₂ : ℝ) (c : ℝ) : Prop :=
  ∃ (C₁ C₂ : ℝ),
  (hypotenuse := c) ∧
  (divided_into_three := (B = (C₁ - C) ∧ C₁ = (C₂ / 3)) ∧ (C₂ - B = C / 3)) ∧
  (CC₁² + C₁C₂² + C₂C² = (2 / 3) * c²)

theorem right_angled_triangle_solution (A B C C₁ C₂ : ℝ) (c : ℝ) :
  right_angled_triangle_hypotenuse_division A B C C₁ C₂ c :=
by sorry

end right_angled_triangle_solution_l136_136301


namespace lilith_caps_collection_l136_136356

theorem lilith_caps_collection :
  let first_year_caps := 3 * 12
  let subsequent_years_caps := 5 * 12 * 4
  let christmas_caps := 40 * 5
  let lost_caps := 15 * 5
  let total_caps := first_year_caps + subsequent_years_caps + christmas_caps - lost_caps
  total_caps = 401 := by
  sorry

end lilith_caps_collection_l136_136356


namespace expression_equivalence_l136_136541

theorem expression_equivalence (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^3 - b^3) - ((a⁻¹)^3 - (b⁻¹)^3) = (a - b) * (a^2 + a * b + b^2) + (b⁻¹ - a⁻¹) * (a⁻² + a⁻¹ * b⁻¹ + b⁻²) := 
by
  sorry

end expression_equivalence_l136_136541


namespace A_and_D_independent_l136_136429

variable (Ω : Type) [Fintype Ω] [ProbabilitySpace Ω]

namespace BallDrawing

def events (ω₁ ω₂ : Ω) : Prop :=
  (ω₁ = 1 ∧ ω₂ = 2) ∨
  (ω₁ + ω₂ = 8) ∨
  (ω₁ + ω₂ = 7)

def A (ω₁ ω₂ : Ω) : Prop := ω₁ = 1
def B (ω₁ ω₂ : Ω) : Prop := ω₂ = 2
def C (ω₁ ω₂ : Ω) : Prop := ω₁ + ω₂ = 8
def D (ω₁ ω₂ : Ω) : Prop := ω₁ + ω₂ = 7

theorem A_and_D_independent : 
  ∀ Ω [Fintype Ω] [ProbabilitySpace Ω], 
  independence (event A) (event D) :=
by sorry

end BallDrawing

end A_and_D_independent_l136_136429


namespace problem1_problem2_problem3_l136_136503

-- Problem 1
theorem problem1 (n : ℕ) (a : ℕ → ℝ) (t : ℝ) (h1 : a n + a (n + 1) = 6 * 5^n) 
                (h2 : a 1 = t) (q : ℝ) (h3 : ∀ n, a (n + 1) = q * a n) 
                (k : ℕ := 1) (p : ℝ := 5) : t = 5 := 
sorry

-- Problem 2
theorem problem2 (n : ℕ) (a : ℕ → ℝ) (t : ℝ) (p : ℝ > 0) (k : ℕ ≥ 1) 
                (h1 : a n + a (n + 1) + a (n + 2) + ... + a (n + k) = 6 * p^n) 
                (h2 : ∀ n, a (n + 1) = p * a n) : (∃ q, q = p ∧ t = if p = 1 then 6 / (k + 1) else 6 * p * (1 - p) / (1 - p^(k + 1))) := 
sorry

-- Problem 3
theorem problem3 (n : ℕ) (a : ℕ → ℝ) (t : ℝ) (p : ℝ > 0) (k : ℕ := 1) (h1 : a 1 = 1)
                (T : ℕ → ℝ) (h2 : ∀ n, T n = ∑ i in range (n+1), a (i+1) / p^i)
                (c : ℝ) : (∃ c, c = \left((1+p)/p * T n - a n / p^n - 6 * n\right)) := 
sorry

end problem1_problem2_problem3_l136_136503


namespace Casper_initial_candies_l136_136163

noncomputable def initial_candies : ℕ := sorry

theorem Casper_initial_candies (x : ℕ) :
  let remaining_after_first_day := (3 / 4 * x - 3 : ℚ) in
  let remaining_after_second_day := (3 / 20 * x - 3 / 5 - 5 : ℚ) in
  let remaining_after_third_day := (1 / 40 * x - 1 / 10 - 5 / 6 - 6 : ℚ) in
  remaining_after_third_day = 10 → x = 678 :=
begin
  intros,
  sorry
end

end Casper_initial_candies_l136_136163


namespace min_value_of_a_l136_136674

theorem min_value_of_a {f : ℝ → ℝ} (a : ℝ) (h : ∀ x ∈ Ioo 1 2, deriv (λ x, a * real.exp x - real.log x) x ≥ 0) : 
  a ≥ real.exp (-1) :=
sorry

end min_value_of_a_l136_136674


namespace maximize_S_n_l136_136149

theorem maximize_S_n (a : ℕ → ℚ) (a1_gt_0 : a 1 > 0)
  (S9_gt_0 : (∑ i in range 9, a (i + 1)) > 0)
  (S10_lt_0 : (∑ i in range 10, a (i + 1)) < 0) :
  ∃ n : ℕ, n = 5 ∧ ∀ m ≠ 5, (∑ i in range m, a (i + 1)) < (∑ i in range 5, a (i + 1)) := by
  sorry

end maximize_S_n_l136_136149


namespace complement_of_A_in_U_l136_136690

-- Define the universal set U and set A
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {3, 4, 5}

-- Define the complement of A in U
theorem complement_of_A_in_U : (U \ A) = {1, 2} :=
by
  sorry

end complement_of_A_in_U_l136_136690


namespace equilibrium_angle_l136_136135

variable (d: ℝ) -- Length of the stick and diameter of the hemisphere
variable (theta: ℝ) -- Angle that the stick makes with the vertical diameter

-- Definitions related to the setup
def a : ℝ := d / 2 -- Radius of the hemisphere

-- Equation from the equilibrium condition (derived manually)
def equilibrium_condition : Prop :=
    let cos_theta := (1 + Real.sqrt 33) / 8 in
    cos theta = cos_theta

theorem equilibrium_angle :
  equilibrium_condition d theta →
  0 < theta ∧ theta < π / 2 :=
sorry

-- Conversion to degrees for the precise angle
def theta_degrees (theta : ℝ) : ℝ :=
  theta * 180 / Real.pi

#eval theta_degrees (Real.arccos ((1 + Real.sqrt 33) / 8)) -- should be approximately 32.383

end equilibrium_angle_l136_136135


namespace tan_eq_cot_num_solutions_l136_136199

theorem tan_eq_cot_num_solutions :
  ∃ n, n = 18 ∧ ∀ θ, θ ∈ Ioc (π / 4) (7 * π / 4) → 
  (tan (7 * π * cos θ) = cot (7 * π * sin θ) → ∃ N, 1 ≤ N ∧ N = n) := 
sorry

end tan_eq_cot_num_solutions_l136_136199


namespace period_of_sin_l136_136871

theorem period_of_sin (ω : ℝ) (hω : ω > 0) :
  (∃ (T : ℝ), T > 0 ∧ (∀ x : ℝ, sin (ω * (x + T) - π / 3) = sin (ω * x - π / 3)) ∧ T = π) → ω = 2 :=
by
  sorry

end period_of_sin_l136_136871


namespace sequence_of_subsets_l136_136261

variable (X : Type) [Fintype X] (n m k : ℕ)
variable [DecidableEq X]
variable (hx : Fintype.card X = n)
variable (hp : Fintype.card { S : Finset X // S.card = m } = k)

theorem sequence_of_subsets (X : Type) [Fintype X] [DecidableEq X] (n m k : ℕ)
  (hx : Fintype.card X = n)
  (hp : Fintype.card { S : Finset X // S.card = m } = k):
  ∃ (A : Fin k → { S : Finset X // S.card = m }),
    (∀ i, A i ∈ { S : Finset X // S.card = m }) ∧
    ∀ i (hi : i < k - 1), Finset.card (A i.val ∩ A (i + 1).val) = m - 1 := by
  sorry

end sequence_of_subsets_l136_136261


namespace function_neither_even_nor_odd_l136_136146

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

def f (x : ℝ) : ℝ := x^2 - x

theorem function_neither_even_nor_odd : ¬is_even_function f ∧ ¬is_odd_function f := by
  sorry

end function_neither_even_nor_odd_l136_136146


namespace min_value_of_a_l136_136673

theorem min_value_of_a {f : ℝ → ℝ} (a : ℝ) (h : ∀ x ∈ Ioo 1 2, deriv (λ x, a * real.exp x - real.log x) x ≥ 0) : 
  a ≥ real.exp (-1) :=
sorry

end min_value_of_a_l136_136673


namespace log_identity_l136_136011

open Real

theorem log_identity (x y a b : ℝ) (hx : x > 0) (hy : y > 0) (ha : a > 1) (hb : b > 1) :
  (log a x) / (log a y) = 1 :=
sorry

end log_identity_l136_136011


namespace area_of_triangle_DEF_l136_136714

noncomputable def area_of_triangle_heron (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_of_triangle_DEF :
  let DE := 25
  let EF := 25
  let DF := 36
  area_of_triangle_heron DE EF DF = 312 := by
    sorry

end area_of_triangle_DEF_l136_136714


namespace cross_section_area_l136_136499

theorem cross_section_area (a : ℝ) : 
  let base_diag := a * (√5 : ℝ)
  let height := a * (√3 : ℝ)
  let cross_section_base := base_diag / 2
  let cross_section_height := height / 2
  ∃ (area : ℝ), 
    area = 1/2 * cross_section_base * cross_section_height ∧ 
    area = (a^2 * (√51 : ℝ)) / 8 :=
begin
  sorry
end

end cross_section_area_l136_136499


namespace weeks_to_meet_goal_l136_136144

def hourly_rate : ℕ := 6
def hours_monday : ℕ := 2
def hours_tuesday : ℕ := 3
def hours_wednesday : ℕ := 4
def hours_thursday : ℕ := 2
def hours_friday : ℕ := 3
def helmet_cost : ℕ := 340
def gloves_cost : ℕ := 45
def initial_savings : ℕ := 40
def misc_expenses : ℕ := 20

theorem weeks_to_meet_goal : 
  let total_needed := helmet_cost + gloves_cost + misc_expenses
  let total_deficit := total_needed - initial_savings
  let total_weekly_hours := hours_monday + hours_tuesday + hours_wednesday + hours_thursday + hours_friday
  let weekly_earnings := total_weekly_hours * hourly_rate
  let weeks_required := Nat.ceil (total_deficit / weekly_earnings)
  weeks_required = 5 := sorry

end weeks_to_meet_goal_l136_136144


namespace distribution_centers_count_l136_136116

theorem distribution_centers_count (n : ℕ) (h : n = 5) : n + (n * (n - 1)) / 2 = 15 :=
by
  subst h -- replace n with 5
  show 5 + (5 * (5 - 1)) / 2 = 15
  have : (5 * 4) / 2 = 10 := by norm_num
  show 5 + 10 = 15
  norm_num

end distribution_centers_count_l136_136116


namespace num_ways_to_buy_three_items_l136_136766

-- Defining the conditions based on the problem statement
def num_headphones : ℕ := 9
def num_mice : ℕ := 13
def num_keyboards : ℕ := 5
def num_kb_mouse_sets : ℕ := 4
def num_hp_mouse_sets : ℕ := 5

-- Defining the theorem statement
theorem num_ways_to_buy_three_items :
  (num_kb_mouse_sets * num_headphones) + 
  (num_hp_mouse_sets * num_keyboards) + 
  (num_headphones * num_mice * num_keyboards) = 646 := 
  by
  sorry

end num_ways_to_buy_three_items_l136_136766


namespace negation_of_proposition_l136_136394

variable (x : ℝ)

theorem negation_of_proposition (h : ∃ x : ℝ, x^2 + x - 1 < 0) : ¬ (∀ x : ℝ, x^2 + x - 1 ≥ 0) :=
sorry

end negation_of_proposition_l136_136394


namespace q_investment_correct_l136_136972

-- Define the conditions
def profit_ratio := (4, 6)
def p_investment := 60000
def expected_q_investment := 90000

-- Define the theorem statement
theorem q_investment_correct (p_investment: ℕ) (q_investment: ℕ) (profit_ratio : ℕ × ℕ)
  (h_ratio: profit_ratio = (4, 6)) (hp_investment: p_investment = 60000) :
  q_investment = 90000 := by
  sorry

end q_investment_correct_l136_136972


namespace min_AP_plus_BP_l136_136340

noncomputable def A : ℝ × ℝ := (1, 0)
noncomputable def B : ℝ × ℝ := (7, 8)
noncomputable def parabola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in y^2 = 4 * x + 4

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  
theorem min_AP_plus_BP : ∀ (P : ℝ × ℝ), parabola P → distance A P + distance B P ≥ 8 :=
sorry

end min_AP_plus_BP_l136_136340


namespace ones_digit_sum_powers_l136_136080

theorem ones_digit_sum_powers (n : ℕ) :
  (∑ k in finset.range (153 + 1), k^153) % 10 = 1 := by
  sorry

end ones_digit_sum_powers_l136_136080


namespace line_circle_intersection_l136_136983

theorem line_circle_intersection (b : ℝ) : (|b| ≤  sqrt 2) ↔ ∃ (x y : ℝ), (y = x + b ∧ x^2 + y^2 = 1) := 
sorry

end line_circle_intersection_l136_136983


namespace same_side_of_line_l136_136242

theorem same_side_of_line (a : ℝ) :
    let point1 := (3, -1)
    let point2 := (-4, -3)
    let line_eq (x y : ℝ) := 3 * x - 2 * y + a
    (line_eq point1.1 point1.2) * (line_eq point2.1 point2.2) > 0 ↔
        (a < -11 ∨ a > 6) := sorry

end same_side_of_line_l136_136242


namespace minimum_dot_product_l136_136775

-- Define point coordinates
structure Point where
  x : ℝ
  y : ℝ

-- Define points A, B, C, D according to the given problem statement
def A : Point := ⟨0, 0⟩
def B : Point := ⟨1, 0⟩
def C : Point := ⟨1, 2⟩
def D : Point := ⟨0, 2⟩

-- Define the condition for points E and F on the sides BC and CD respectively.
def isOnBC (E : Point) : Prop := E.x = 1 ∧ 0 ≤ E.y ∧ E.y ≤ 2
def isOnCD (F : Point) : Prop := F.y = 2 ∧ 0 ≤ F.x ∧ F.x ≤ 1

-- Define the distance constraint for |EF| = 1
def distEF (E F : Point) : Prop :=
  (F.x - E.x)^2 + (F.y - E.y)^2 = 1

-- Define the dot product between vectors AE and AF
def dotProductAEAF (E F : Point) : ℝ :=
  2 * E.y + F.x

-- Main theorem to prove the minimum dot product value
theorem minimum_dot_product (E F : Point) (hE : isOnBC E) (hF : isOnCD F) (hDistEF : distEF E F) :
  dotProductAEAF E F = 5 - Real.sqrt 5 :=
  sorry

end minimum_dot_product_l136_136775


namespace cristina_baked_croissants_l136_136533

theorem cristina_baked_croissants : 
  (∃ n : ℕ, n = 7 * 2) → (∀ guests croissants_per_guest : ℕ, guests = 7 → croissants_per_guest = 2 → guests * croissants_per_guest = 14) :=
by
  intros h guests croissants_per_guest h_guests h_croissants_per_guest
  rw [h_guests, h_croissants_per_guest]
  exact h
  sorry

end cristina_baked_croissants_l136_136533


namespace store_purchase_ways_l136_136738

theorem store_purchase_ways :
  ∃ (headphones sets_hm sets_km keyboards mice : ℕ), 
  headphones = 9 ∧
  mice = 13 ∧
  keyboards = 5 ∧
  sets_km = 4 ∧
  sets_hm = 5 ∧
  (sets_km * headphones + sets_hm * keyboards + headphones * mice * keyboards = 646) :=
by
  use 9, 5, 4, 5, 13
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  sorry

end store_purchase_ways_l136_136738


namespace min_area_rectangle_l136_136152

theorem min_area_rectangle (l w : ℝ) 
  (hl : 3.5 ≤ l ∧ l ≤ 4.5) 
  (hw : 5.5 ≤ w ∧ w ≤ 6.5) 
  (constraint : l ≥ 2 * w) : 
  l * w = 60.5 := 
sorry

end min_area_rectangle_l136_136152


namespace positive_difference_between_probabilities_is_one_eighth_l136_136937

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem positive_difference_between_probabilities_is_one_eighth :
  let p : ℚ := 1 / 2
    n : ℕ := 5
    prob_4_heads := binomial_probability n 4 p
    prob_5_heads := binomial_probability n 5 p
  in |prob_4_heads - prob_5_heads| = 1 / 8 :=
by
  let p : ℚ := 1 / 2
  let n : ℕ := 5
  let prob_4_heads := binomial_probability n 4 p
  let prob_5_heads := binomial_probability n 5 p
  sorry

end positive_difference_between_probabilities_is_one_eighth_l136_136937


namespace proj_of_b_eq_v_diff_proj_of_a_l136_136328

noncomputable theory

variables (a b : ℝ × ℝ) (v : ℝ × ℝ)

def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (u.1 * v.1 + u.2 * v.2) / (u.1 * u.1 + u.2 * u.2)
  (scalar * u.1, scalar * u.2)

theorem proj_of_b_eq_v_diff_proj_of_a
  (h₀ : orthogonal a b)
  (h₁ : proj a ⟨4, -2⟩ = ⟨-4/5, -8/5⟩)
  : proj b ⟨4, -2⟩ = ⟨24/5, -2/5⟩ :=
sorry

end proj_of_b_eq_v_diff_proj_of_a_l136_136328


namespace positive_difference_between_probabilities_is_one_eighth_l136_136935

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem positive_difference_between_probabilities_is_one_eighth :
  let p : ℚ := 1 / 2
    n : ℕ := 5
    prob_4_heads := binomial_probability n 4 p
    prob_5_heads := binomial_probability n 5 p
  in |prob_4_heads - prob_5_heads| = 1 / 8 :=
by
  let p : ℚ := 1 / 2
  let n : ℕ := 5
  let prob_4_heads := binomial_probability n 4 p
  let prob_5_heads := binomial_probability n 5 p
  sorry

end positive_difference_between_probabilities_is_one_eighth_l136_136935


namespace total_ways_to_buy_l136_136761

-- Definitions:
def h : ℕ := 9  -- number of headphones
def m : ℕ := 13 -- number of mice
def k : ℕ := 5  -- number of keyboards
def s_km : ℕ := 4 -- number of sets of keyboard and mouse
def s_hm : ℕ := 5 -- number of sets of headphones and mouse

-- Theorem stating the number of ways to buy three items is 646.
theorem total_ways_to_buy : (s_km * h) + (s_hm * k) + (h * m * k) = 646 := by
  -- Placeholder proof using sorry.
  sorry

end total_ways_to_buy_l136_136761


namespace unique_pairs_of_union_l136_136288

theorem unique_pairs_of_union (A B : Set) (f : Set → ℕ) (h_union : (A ∪ B).size = 2) : 
    {p : ℕ × ℕ | p = (f A, f B) ∧ ∃ A B : Set, (A ∪ B).size = 2}.size = 6 := by
  sorry

end unique_pairs_of_union_l136_136288


namespace largest_non_expressible_integer_l136_136170

open BigOperators

noncomputable def A (n : ℕ) (k : ℕ) : ℤ := 2^n - 2^k
def An (n : ℕ) : Set ℤ := {m | ∃ k, k < n ∧ m = A n k}

theorem largest_non_expressible_integer (n : ℕ) (hn : n ≥ 2) : ∃ m, m = 2^n * (n - 2) + 1 ∧ (m ∉ {sum | ∃ S, S ⊆ An n ∧ sum = S.sum (fun x => x)}) := sorry

end largest_non_expressible_integer_l136_136170


namespace Ms_Hatcher_total_students_l136_136001

noncomputable def number_of_students (third_graders fourth_graders fifth_graders sixth_graders : ℕ) : ℕ :=
  third_graders + fourth_graders + fifth_graders + sixth_graders

theorem Ms_Hatcher_total_students (third_graders fourth_graders fifth_graders sixth_graders : ℕ) 
  (h1 : third_graders = 20)
  (h2 : fourth_graders = 2 * third_graders) 
  (h3 : fifth_graders = third_graders / 2) 
  (h4 : sixth_graders = 3 * (third_graders + fourth_graders) / 4) : 
  number_of_students third_graders fourth_graders fifth_graders sixth_graders = 115 :=
by
  sorry

end Ms_Hatcher_total_students_l136_136001


namespace five_spheres_tangency_l136_136790

variable (S : Fin 5 → Sphere ℝ)
variable (tangent_plane : ∀ (i : Fin 5), Plane ℝ)

theorem five_spheres_tangency :
  (∀ i : Fin 5, tangent_plane i ∈ tangent_planes_through_center (S i)) →
  (∀ i j : Fin 5, i ≠ j → S j ∈ tangent_plane i) →
  ∃ (sphere_setup : Fin 5 → Sphere ℝ), 
  (∀ i : Fin 5, ∃ (plane : Plane ℝ), plane ∈ tangent_planes_through_center (sphere_setup i) 
    ∧ (∀ j : Fin 5, i ≠ j → plane ∈ tangent_planes_through_center (sphere_setup j))) :=
sorry

end five_spheres_tangency_l136_136790


namespace distance_from_origin_l136_136721

theorem distance_from_origin (x y : ℝ) : x = 3 → y = -4 → real.sqrt (x^2 + y^2) = 5 :=
by
  intros hx hy
  rw [hx, hy]
  exact sorry

end distance_from_origin_l136_136721


namespace min_value_of_a_l136_136677

theorem min_value_of_a {f : ℝ → ℝ} (a : ℝ) (h : ∀ x ∈ Ioo 1 2, deriv (λ x, a * real.exp x - real.log x) x ≥ 0) : 
  a ≥ real.exp (-1) :=
sorry

end min_value_of_a_l136_136677


namespace sum_of_four_digit_numbers_equal_to_cube_of_sum_of_digits_l136_136455

noncomputable def valid_four_digit_cubes := {4913, 5832}

theorem sum_of_four_digit_numbers_equal_to_cube_of_sum_of_digits :
  ∑ x in valid_four_digit_cubes, x = 10745 :=
by
  sorry

end sum_of_four_digit_numbers_equal_to_cube_of_sum_of_digits_l136_136455


namespace coin_flip_difference_l136_136950

/-- The positive difference between the probability of a fair coin landing heads up
exactly 4 times out of 5 flips and the probability of a fair coin landing heads up
5 times out of 5 flips is 1/8. -/
theorem coin_flip_difference :
  (5 * (1 / 2) ^ 5) - ((1 / 2) ^ 5) = (1 / 8) :=
by
  sorry

end coin_flip_difference_l136_136950


namespace remainder_division_l136_136963

-- Define the polynomial f(x) = x^51 + 51
def f (x : ℤ) : ℤ := x^51 + 51

-- State the theorem to be proven
theorem remainder_division : f (-1) = 50 :=
by
  -- proof goes here
  sorry

end remainder_division_l136_136963


namespace sum_of_x_coordinates_above_line_l136_136683

def points : List (ℕ × ℕ) := [(4, 15), (7, 25), (13, 42), (19, 57), (21, 65)]

def above_line (p : ℕ × ℕ) : Prop := p.2 > 3 * p.1 + 5

theorem sum_of_x_coordinates_above_line :
  (points.filter above_line).sum (λ p, p.1) = 0 :=
by
  sorry

end sum_of_x_coordinates_above_line_l136_136683


namespace purchase_combinations_correct_l136_136758

theorem purchase_combinations_correct : 
  let headphones : ℕ := 9
  let mice : ℕ := 13
  let keyboards : ℕ := 5
  let sets_keyboard_mouse : ℕ := 4
  let sets_headphones_mouse : ℕ := 5
in 
  (sets_keyboard_mouse * headphones) + 
  (sets_headphones_mouse * keyboards) + 
  (headphones * mice * keyboards) = 646 := 
by
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let sets_keyboard_mouse := 4
  let sets_headphones_mouse := 5
  sorry

end purchase_combinations_correct_l136_136758


namespace minimum_value_of_a_l136_136645

theorem minimum_value_of_a (a : ℝ) :
  (∀ x ∈ (Ioo 1 2), ae^x - ln x ≥ 0) ↔ a ≥ exp(-1) := sorry

end minimum_value_of_a_l136_136645


namespace fifth_largest_divisor_of_645120000_is_40320000_l136_136870

theorem fifth_largest_divisor_of_645120000_is_40320000 :
  let n := 645_120_000
  40320_000 is_dvd_of_nth_largest n 5 := 
  sorry

end fifth_largest_divisor_of_645120000_is_40320000_l136_136870


namespace bananas_count_l136_136887

theorem bananas_count
    (total_fruit : ℕ)
    (apples_ratio : ℕ)
    (persimmons_ratio : ℕ)
    (apples_and_persimmons : apples_ratio * bananas + persimmons_ratio * bananas = total_fruit)
    (apples_ratio_val : apples_ratio = 4)
    (persimmons_ratio_val : persimmons_ratio = 3)
    (total_fruit_value : total_fruit = 210) :
    bananas = 30 :=
by
  sorry

end bananas_count_l136_136887


namespace parameter_relationship_l136_136081

theorem parameter_relationship (a b c : ℝ) (h : c > 0) :
  (∀ x : ℝ, ∃ infinite_solutions : Prop, (sqrt (x + a * sqrt x + b) = c)) → 
  (a = -2 * c ∧ b = c^2) := 
by
  sorry

end parameter_relationship_l136_136081


namespace simplify_expression_l136_136020

theorem simplify_expression : 4 * (18 / 5) * (25 / -72) = -5 := by
  sorry

end simplify_expression_l136_136020


namespace reporter_wrong_l136_136988

theorem reporter_wrong : 
  ∃ (P : Fin 20 → ℕ), 
    (∀ i : Fin 20, ∃ W D : ℕ, W = D ∧ P i = 3 * D) ∧ 
    (∑ i, P i) ≠ 380 :=
by sorry

end reporter_wrong_l136_136988


namespace parabola_focus_coordinates_parabola_distance_to_directrix_l136_136385

-- Define constants and variables
def parabola_equation (x y : ℝ) : Prop := y^2 = 4 * x

noncomputable def focus_coordinates : ℝ × ℝ := (1, 0)

noncomputable def point : ℝ × ℝ := (4, 4)

noncomputable def directrix : ℝ := -1

noncomputable def distance_to_directrix : ℝ := 5

-- Proof statements
theorem parabola_focus_coordinates (x y : ℝ) (h : parabola_equation x y) : 
  focus_coordinates = (1, 0) :=
sorry

theorem parabola_distance_to_directrix (p : ℝ × ℝ) (d : ℝ) (h : p = point) (h_line : d = directrix) : 
  distance_to_directrix = 5 :=
  by
    -- Define and use the distance between point and vertical line formula
    sorry

end parabola_focus_coordinates_parabola_distance_to_directrix_l136_136385


namespace ellipse_equation_find_slope_l136_136593

theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_vs : b = 1) (h_fl : 2*sqrt(3) = 2*sqrt(a^2 - b^2)) : 
  ∀ x y : ℝ, (x / a)^2 + (y / b)^2 = 1 ↔ x^2 / 4 + y^2 = 1 :=
by sorry

theorem find_slope (k : ℝ) (P : ℝ × ℝ) (hP : P = (-2, 1)) (h_condition : ∃ B C : ℝ × ℝ, B ≠ C ∧ ∀ t : ℝ, (k*t + B.2)*(k*t + C.2) = -k^2*4*t^2 - 8k*t - 4k^2 - 2 - B.1 + 2 + C.1 + 4f - B.1 / (1-f) - C.1 / (1-f) = 2 ) :
  k = -4 :=
by sorry

end ellipse_equation_find_slope_l136_136593


namespace total_sales_value_l136_136312

-- Define the conditions from the problem
def base_salary_per_hour : ℝ := 7.50
def worked_hours : ℝ := 160
def commission_rate : ℝ := 0.16
def insurance_amount : ℝ := 260
def necessities_budget_rate : ℝ := 0.95

-- Define a function to calculate the basic salary for the month
def basic_salary (hourly_rate : ℝ) (hours : ℝ) : ℝ :=
  hourly_rate * hours

-- Define a function to calculate the commission based on sales
def commission (rate : ℝ) (sales : ℝ) : ℝ :=
  rate * sales

-- Define a function to calculate total earnings
def total_earnings (basic : ℝ) (comm : ℝ) : ℝ :=
  basic + comm

-- Define a function to calculate the remainder for insurance
def insurance_amount_from_earnings (earnings : ℝ) : ℝ :=
  earnings * (1 - necessities_budget_rate)

-- Main theorem to prove
theorem total_sales_value : 
  ∃ V : ℝ,
    let S := basic_salary base_salary_per_hour worked_hours in
    let E := S + commission commission_rate V in
    E * 0.05 = insurance_amount ∧ E = 5200 ∧ V = 25000 :=
  by
    let S := basic_salary base_salary_per_hour worked_hours
    let V : ℝ := 25000
    let E := S + commission commission_rate V
    have h: E * 0.05 = insurance_amount := sorry -- proof to be filled
    have h2: E = 5200 := sorry -- proof to be filled
    exact ⟨V, h, h2, rfl⟩

end total_sales_value_l136_136312


namespace range_of_a_l136_136610

variable (f : ℝ → ℝ) (a : ℝ)

def conditions : Prop :=
  (∀ x, f (-x) = f x) ∧                 -- Even function
  (∀ x, f (x + 3 *n) = f x) ∧           -- Periodic function with period 3
  (f 1 < 1) ∧ 
  (f 5 = (2 * a - 3) / (a + 1))

theorem range_of_a (hf : conditions f a) : -1 < a ∧ a < 4 :=
sorry

end range_of_a_l136_136610


namespace pythagorean_conditions_l136_136722

theorem pythagorean_conditions 
  (m n : ℕ)
  (coprime : Nat.coprime m n)
  (h_mn_val : m > n)
  (h_multiple_4 : (2 * m * n) % 4 = 0 ∨ (m * m - n * n) % 4 = 0)
  (h_multiple_3 : (2 * m * n) % 3 = 0 ∨ (m * m - n * n) % 3 = 0)
  (h_multiple_5 : ((m * m + n * n) % 5 = 0 ∨ (m * m - n * n) % 5 = 0)) :
  ∃ (a b c : ℕ), a^2 + b^2 = c^2 ∧ ((a % 4 = 0) ∨ (b % 4 = 0)) ∧ ((a % 3 = 0) ∨ (b % 3 = 0)) ∧ ((a % 5 = 0) ∨ (b % 5 = 0) ∨ (c % 5 = 0)) :=
by
  sorry -- to be completed with an appropriate proof

end pythagorean_conditions_l136_136722


namespace purchase_combinations_correct_l136_136754

theorem purchase_combinations_correct : 
  let headphones : ℕ := 9
  let mice : ℕ := 13
  let keyboards : ℕ := 5
  let sets_keyboard_mouse : ℕ := 4
  let sets_headphones_mouse : ℕ := 5
in 
  (sets_keyboard_mouse * headphones) + 
  (sets_headphones_mouse * keyboards) + 
  (headphones * mice * keyboards) = 646 := 
by
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let sets_keyboard_mouse := 4
  let sets_headphones_mouse := 5
  sorry

end purchase_combinations_correct_l136_136754


namespace purchase_combinations_correct_l136_136755

theorem purchase_combinations_correct : 
  let headphones : ℕ := 9
  let mice : ℕ := 13
  let keyboards : ℕ := 5
  let sets_keyboard_mouse : ℕ := 4
  let sets_headphones_mouse : ℕ := 5
in 
  (sets_keyboard_mouse * headphones) + 
  (sets_headphones_mouse * keyboards) + 
  (headphones * mice * keyboards) = 646 := 
by
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let sets_keyboard_mouse := 4
  let sets_headphones_mouse := 5
  sorry

end purchase_combinations_correct_l136_136755


namespace area_of_curve_l136_136516

noncomputable def polar_curve (φ : Real) : Real :=
  (1 / 2) + Real.sin φ

noncomputable def area_enclosed_by_polar_curve : Real :=
  2 * ((1 / 2) * ∫ (φ : Real) in (-Real.pi / 2)..(Real.pi / 2), (polar_curve φ) ^ 2)

theorem area_of_curve : area_enclosed_by_polar_curve = (3 * Real.pi) / 4 :=
by
  sorry

end area_of_curve_l136_136516


namespace profit_range_l136_136505

noncomputable def price (t : ℕ) := 1 / 4 * t + 30
noncomputable def sales_volume (t : ℕ) := 120 - 2 * t
noncomputable def daily_profit (n : ℝ) (t : ℕ) := 
  let p := price t
  let y := sales_volume t
  (p - 20 - n) * y

-- We need to prove that for n within the bounds, the profit increases for the first 28 days.
theorem profit_range (n : ℝ) : 
  ∀ t : ℕ, 1 ≤ t ∧ t ≤ 48 → (daily_profit n t).deriv t > 0 → (8.75 < n ∧ n ≤ 9.25) :=
sorry

end profit_range_l136_136505


namespace find_b_l136_136250

def piecewise_function (x : ℝ) : ℝ :=
if x ≤ -1 then x + 2 else
if -1 < x ∧ x < 2 then x^2 else
if x ≥ 2 then 2 * x else 0 -- the else case is added to handle all possible values of x

theorem find_b {b : ℝ} (h : piecewise_function b = 1/2) :
b = -3/2 ∨ b = sqrt 2 / 2 ∨ b = -sqrt 2 / 2 :=
by sorry

end find_b_l136_136250


namespace sum_of_areas_l136_136844

-- Given definitions and conditions
def regular_hexagon (side_length : ℝ) : Prop := true
def square (side_length : ℝ) : Prop := true
def lies_outside (A B : Prop) : Prop := true
def area (shape : Prop) : ℝ := sorry

-- Let's define our shapes and the conditions in the problem
def NOSAME : Prop := regular_hexagon 1
def UDON : Prop := square 1
def UDON_lies_outside_NOSAME : Prop := lies_outside UDON NOSAME

-- Define the quadrilaterals SAND and SEND
def SAND : Prop := true -- Placeholder for quadrilateral SAND
def SEND : Prop := true -- Placeholder for quadrilateral SEND

-- Define their areas
def area_SAND : ℝ := area SAND
def area_SEND : ℝ := area SEND

-- The final theorem statement
theorem sum_of_areas : 
  NOSAME ∧ UDON ∧ UDON_lies_outside_NOSAME →
  (area_SAND + area_SEND) = (3 / 2 + (3 * Real.sqrt 3) / 2) := 
by 
  intro h,
  sorry

end sum_of_areas_l136_136844


namespace xiaotong_phys_ed_score_l136_136112

def max_score := 100
def weight_extracurricular := 0.30
def weight_finalexam := 0.70
def score_extracurricular := 90
def score_finalexam := 80

theorem xiaotong_phys_ed_score : 
  score_extracurricular * weight_extracurricular + score_finalexam * weight_finalexam = 83 :=
by 
  sorry

end xiaotong_phys_ed_score_l136_136112


namespace minimum_a_for_monotonic_increasing_on_interval_l136_136647

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp x - Real.log x

noncomputable def f_prime (a x : ℝ) : ℝ := a * Real.exp x - 1 / x

def min_value_of_a : ℝ := Real.exp (-1)

theorem minimum_a_for_monotonic_increasing_on_interval :
  (∀ x ∈ Set.Ioo 1 2, 0 ≤ f_prime a x) ↔ a ≥ min_value_of_a :=
by
  sorry

end minimum_a_for_monotonic_increasing_on_interval_l136_136647


namespace probability_same_color_white_l136_136062

/--
Given a box with 6 white balls and 5 black balls, if 3 balls are drawn such that all drawn balls have the same color,
prove that the probability that these balls are white is 2/3.
-/
theorem probability_same_color_white :
  (∃ (n_white n_black drawn_white drawn_black total_same_color : ℕ),
    n_white = 6 ∧ n_black = 5 ∧
    drawn_white = Nat.choose n_white 3 ∧ drawn_black = Nat.choose n_black 3 ∧
    total_same_color = drawn_white + drawn_black ∧
    (drawn_white:ℚ) / total_same_color = 2 / 3) :=
sorry

end probability_same_color_white_l136_136062


namespace number_of_ways_to_buy_three_items_l136_136733

def headphones : ℕ := 9
def mice : ℕ := 13
def keyboards : ℕ := 5
def keyboard_mouse_sets : ℕ := 4
def headphone_mouse_sets : ℕ := 5

theorem number_of_ways_to_buy_three_items : 
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 := 
by 
  sorry

end number_of_ways_to_buy_three_items_l136_136733


namespace selling_prices_maximize_profit_l136_136996

-- Definitions for the conditions
def total_items : ℕ := 200
def budget : ℤ := 5000
def cost_basketball : ℤ := 30
def cost_volleyball : ℤ := 24
def selling_price_ratio : ℚ := 3 / 2
def school_purchase_basketballs_value : ℤ := 1800
def school_purchase_volleyballs_value : ℤ := 1500
def basketballs_fewer_than_volleyballs : ℤ := 10

-- Part 1: Proof of selling prices
theorem selling_prices (x : ℚ) :
  (school_purchase_volleyballs_value / x - school_purchase_basketballs_value / (x * selling_price_ratio) = basketballs_fewer_than_volleyballs)
  → ∃ (basketball_price volleyball_price : ℚ), basketball_price = 45 ∧ volleyball_price = 30 :=
by
  sorry

-- Part 2: Proof of maximizing profit
theorem maximize_profit (a : ℕ) :
  (cost_basketball * a + cost_volleyball * (total_items - a) ≤ budget)
  → ∃ optimal_a : ℕ, (optimal_a = 33 ∧ total_items - optimal_a = 167) :=
by
  sorry

end selling_prices_maximize_profit_l136_136996


namespace coin_flip_probability_difference_l136_136944

theorem coin_flip_probability_difference :
  let p1 := (nat.choose 5 4) * (1 / 2)^5
  let p2 := (1 / 2)^5
  (p1 - p2 = 1 / 8) :=
by
  let p1 := (nat.choose 5 4) * (1 / 2)^5
  let p2 := (1 / 2)^5
  sorry

end coin_flip_probability_difference_l136_136944


namespace buy_items_ways_l136_136727

theorem buy_items_ways (headphones keyboards mice keyboard_mouse_sets headphone_mouse_sets : ℕ) :
  headphones = 9 → keyboards = 5 → mice = 13 → keyboard_mouse_sets = 4 → headphone_mouse_sets = 5 →
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 :=
by
  intros h_eq k_eq m_eq kms_eq hms_eq
  have h_eq_gen : headphones = 9 := h_eq
  have k_eq_gen : keyboards = 5 := k_eq
  have m_eq_gen : mice = 13 := m_eq
  have kms_eq_gen : keyboard_mouse_sets = 4 := kms_eq
  have hms_eq_gen : headphone_mouse_sets = 5 := hms_eq
  sorry

end buy_items_ways_l136_136727


namespace initial_money_l136_136706

theorem initial_money (M : ℝ) 
  (h1 : let apples_cost := M / 2 in apples_cost)
  (h2 : let bag_cost := 300 in bag_cost)
  (h3 : let remaining_after_fruit := M - M / 2 - 300 in remaining_after_fruit)
  (h4 : let mackerel_cost := 1 / 2 * (M / 2 - 300) in mackerel_cost)
  (h5 : let fish_shop_cost := 400 in fish_shop_cost)
  (h6 : let remaining_after_fish := (M / 2 - 300) - 1 / 2 * (M / 2 - 300) - 400 in remaining_after_fish)
  (h7 : let total_remaining := M / 4 - 550 in total_remaining = 0) :
  M = 2200 :=
sorry

end initial_money_l136_136706


namespace length_of_side_AD_l136_136398

theorem length_of_side_AD 
  (AB BC CD AD : ℝ)
  (h1 : AB = 41)
  (h2 : BC = AB - 18)
  (h3 : CD = BC - 6)
  (h4 : AB + BC + CD + AD = 100) : 
  AD = 19 := 
begin
  sorry
end

end length_of_side_AD_l136_136398


namespace coprime_an_2n_plus_1_l136_136800

def recurrence (a : ℕ → ℤ) : Prop :=
  a 1 = a_1 ∧ ∀ n : ℕ, a (n + 1) = a n ^ 2 - a n - 1

theorem coprime_an_2n_plus_1 (a : ℕ → ℤ) (a_1 : ℤ) (h : recurrence a) (n : ℕ) :
  Int.gcd (a (n + 1)) (2 * n + 1) = 1 :=
by
  sorry

end coprime_an_2n_plus_1_l136_136800


namespace derivative_at_zero_l136_136621

-- Define the function f(x)
def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)

-- State the theorem with the given conditions and expected result
theorem derivative_at_zero :
  (deriv f 0 = -120) :=
by
  sorry

end derivative_at_zero_l136_136621


namespace total_ways_to_buy_l136_136764

-- Definitions:
def h : ℕ := 9  -- number of headphones
def m : ℕ := 13 -- number of mice
def k : ℕ := 5  -- number of keyboards
def s_km : ℕ := 4 -- number of sets of keyboard and mouse
def s_hm : ℕ := 5 -- number of sets of headphones and mouse

-- Theorem stating the number of ways to buy three items is 646.
theorem total_ways_to_buy : (s_km * h) + (s_hm * k) + (h * m * k) = 646 := by
  -- Placeholder proof using sorry.
  sorry

end total_ways_to_buy_l136_136764


namespace sum_selected_elements_l136_136099

theorem sum_selected_elements {n : ℕ} (a : ℕ → ℕ → ℕ)
  (x : ℕ → ℕ) (r s : ℕ → ℕ)
  (hsums : ∀ i, 1 ≤ x i ∧ x i ≤ n ∧
                1 ≤ r i ∧ r i ≤ n ∧
                1 ≤ s i ∧ s i ≤ n ∧
                x i = (r i - 1) * n + s i ∧
                (bijective r) ∧ (bijective s)) :
  ∑ i in range n, x i = (n * (n^2 + 1)) / 2 := 
sorry

end sum_selected_elements_l136_136099


namespace central_angle_error_bound_l136_136048

constant α: ℝ -- α is a real number representing the angle in degrees

-- Given conditions
constant is_central_angle_of_18_sided_polygon (α: ℝ): Prop 
constant central_angle_approx (α: ℝ): Prop := 
  |α - 20| < (21 / 3600)

-- Theorem stating the problem
theorem central_angle_error_bound 
  (h1: is_central_angle_of_18_sided_polygon α)
  : central_angle_approx α := 
sorry

end central_angle_error_bound_l136_136048


namespace tan_subtraction_l136_136280

variable (α β : ℝ)

def tan_alpha : ℝ := 9
def tan_beta : ℝ := 5

theorem tan_subtraction :
  tan (α - β) = 2 / 23 :=
by
  have h_tan_alpha : tan α = tan_alpha := sorry
  have h_tan_beta : tan β = tan_beta := sorry
  sorry

end tan_subtraction_l136_136280


namespace find_three_digit_number_l136_136187

/-- 
  Define the three-digit number abc and show that for some digit d in the range of 1 to 9,
  the conditions are satisfied.
-/
theorem find_three_digit_number
  (a b c d : ℕ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : 1 ≤ d ∧ d ≤ 9)
  (h_abc : 100 * a + 10 * b + c = 627)
  (h_bcd : 100 * b + 10 * c + d = 627 * a)
  (h_1a4d : 1040 + 100 * a + d = 627 * a)
  : 100 * a + 10 * b + c = 627 := 
sorry

end find_three_digit_number_l136_136187


namespace inequality_problem_l136_136345

theorem inequality_problem (n : ℕ) (x : ℕ → ℝ) 
  (h1 : ∀ (i : ℕ), 1 ≤ i → i ≤ n → x i ≠ 0) 
  (h2 : ∀ k, 1 ≤ k → k ≤ n → ∑ k in Finset.range (n+1), x k = 0) : 
  sqrt ((∑ k in Finset.range n, (x k^4 + k^2) / (x k^2))^2 - n^2 * (n+1)^2) 
  ≥ ∑ k in Finset.range n, (x k^4 - k^2) / x k^2 :=
by
  sorry

end inequality_problem_l136_136345


namespace principal_argument_e3iπ_l136_136180

noncomputable def principalArgument (z : Complex) : ℝ :=
  if h : z ≠ 0 then (Complex.arg z).toReal else 0

theorem principal_argument_e3iπ : principalArgument (Complex.exp (3 * Complex.I * Real.pi)) = Real.pi :=
by
  sorry

end principal_argument_e3iπ_l136_136180


namespace right_triangle_length_QR_l136_136853

variable (Q_p R_p : ℝ × ℝ) (Q_origin : Q_p = (0, 0)) (PQ_len : ((fst R_p - 0) ^ 2 + (snd R_p - 0) ^ 2) ^ (1/2) = 15) (cos_Q : (0.5 = (15 / QR_len)))
-- Given the preceding conditions
def QR_len : ℝ := dist (0,0) R_p

theorem right_triangle_length_QR
    (h1 : (Q_p = (0, 0)) )
    (h2 : sqrt ((fst R_p - 0) ^ 2 + (snd R_p - 0) ^ 2) = 15)
    (h3 : 0.5 = (15 / QR_len)) :
    QR_len = 30 :=
by
  sorry

end right_triangle_length_QR_l136_136853


namespace count_numbers_with_at_most_two_different_digits_l136_136266

theorem count_numbers_with_at_most_two_different_digits : 
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  number_of_such_numbers := 
    let count_one_digit := 9 in -- 1 to 9
    let count_two_distinct_non_zero := 36 * 8 in -- \(\binom{9}{2}\) * (valid arrangements)
    let count_including_zero := 9 * 4 in -- 9 (choices for non-zero digit) * (valid arrangements with 0)
    count_one_digit + count_two_distinct_non_zero + count_including_zero
  in number_of_such_numbers = 351 :=
by
  sorry

end count_numbers_with_at_most_two_different_digits_l136_136266


namespace sin_alpha_equals_3_sqrt_10_div_10_l136_136613

theorem sin_alpha_equals_3_sqrt_10_div_10
  (α : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : tan (π - α) + 3 = 0) :
  sin α = 3 * sqrt 10 / 10 :=
by
  sorry

end sin_alpha_equals_3_sqrt_10_div_10_l136_136613


namespace nim_maximum_product_l136_136780

def nim_max_product (x y : ℕ) : ℕ :=
43 * 99 * x * y

theorem nim_maximum_product :
  ∃ x y : ℕ, (43 ≠ 0) ∧ (99 ≠ 0) ∧ (x ≠ 0) ∧ (y ≠ 0) ∧
  (43 + 99 + x + y = 0) ∧ (nim_max_product x y = 7704) :=
sorry

end nim_maximum_product_l136_136780


namespace toothpicks_for_8_steps_l136_136514

theorem toothpicks_for_8_steps : 
  ∀ (t4 t5 t6 t7 t8 : ℕ),
  t4 = 28 →
  t5 = t4 + 13 →
  t6 = t5 + 17 →
  t7 = t6 + 22 →
  t8 = t7 + 28 →
  t8 - t4 = 80 := 
by
  intros,
  sorry

end toothpicks_for_8_steps_l136_136514


namespace find_z_l136_136617

variable {x y z w : ℝ}

theorem find_z (h : (1/x) + (1/y) = (1/z) + w) : z = (x * y) / (x + y - w * x * y) :=
by sorry

end find_z_l136_136617


namespace max_product_with_853_l136_136445

-- Define the digits to be used
def digits : List ℕ := [3, 5, 6, 8, 9]

-- Define the three-digit number 853 using the available digits
def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

-- Define the two-digit number
def two_digit_number (d e : ℕ) : ℕ := 10 * d + e

-- Define the product of the three-digit number and the two-digit number
def product (a b c d e : ℕ) : ℕ := three_digit_number a b c * two_digit_number d e

-- Define the condition that each digit is used exactly once
def used_once (a b c d e : ℕ) : Prop :=
  digits.perm [a, b, c, d, e]

-- Prove that the greatest product is achieved with the three-digit integer 853
theorem max_product_with_853 : ∃ (d e : ℕ), used_once 8 5 3 d e ∧
  ∀ (a b c d' e' : ℕ), used_once a b c d' e' → 
    three_digit_number a b c = 853 →
    product a b c d' e' ≥ product a b c d e :=
by sorry

end max_product_with_853_l136_136445


namespace BC_parallel_MN_l136_136305

-- Define the points and angles
variables {A B C P M N : Type}
variable [metric_space P]
variables (A B C M N : P)
variable (P : P)
variables (h1 : ∠BAC = 45)
variables (h2 : ∠APB = 45)
variables (h3 : ∠APC = 45)
variables (hM : is_projection P A B M)
variables (hN : is_projection P A C N)

-- Prove that BC is parallel to MN
theorem BC_parallel_MN :
  ∥BC, MN := sorry

end BC_parallel_MN_l136_136305


namespace duration_of_period_l136_136850

-- Define the number of pies baked per day for apple and cherry pies
def apple_pies_per_day := 12
def cherry_pies_per_day := 12

-- Define the number of days per week Steve bakes each type of pie
def apple_days_per_week := 3
def cherry_days_per_week := 2

-- Define the number of pies baked per week
def apple_pies_per_week := apple_days_per_week * apple_pies_per_day
def cherry_pies_per_week := cherry_days_per_week * cherry_pies_per_day

-- Define the duration of the period in weeks, and the condition given in the problem
def weeks := ℕ
def condition (w : weeks) := (apple_pies_per_week * w) = (cherry_pies_per_week * w + 12)

-- Define the theorem that we need to prove
theorem duration_of_period : ∃ (w : weeks), condition w ∧ w = 1 :=
by
  existsi 1
  constructor
  · unfold condition
    norm_num
  sorry

end duration_of_period_l136_136850


namespace largest_proper_fraction_smallest_improper_fraction_smallest_mixed_number_l136_136077

-- Definitions based on conditions
def isProperFraction (n d : ℕ) : Prop := n < d
def isImproperFraction (n d : ℕ) : Prop := n ≥ d
def isMixedNumber (w n d : ℕ) : Prop := w > 0 ∧ isProperFraction n d

-- Fractional part is 1/9, meaning all fractions considered have part = 1/9
def fractionalPart := 1 / 9

-- Lean 4 statements to verify the correct answers
theorem largest_proper_fraction : ∃ n d : ℕ, fractionalPart = n / d ∧ isProperFraction n d ∧ (n, d) = (8, 9) := sorry

theorem smallest_improper_fraction : ∃ n d : ℕ, fractionalPart = n / d ∧ isImproperFraction n d ∧ (n, d) = (9, 9) := sorry

theorem smallest_mixed_number : ∃ w n d : ℕ, fractionalPart = n / d ∧ isMixedNumber w n d ∧ ((w, n, d) = (1, 1, 9) ∨ (w, n, d) = (10, 9)) := sorry

end largest_proper_fraction_smallest_improper_fraction_smallest_mixed_number_l136_136077


namespace correct_angle_calculation_l136_136588

theorem correct_angle_calculation
  (α β : ℝ) 
  (hα : 0 < α ∧ α < 90) 
  (hβ : 90 < β ∧ β < 180) 
  : 22.5 < (1/4) * (α + β) ∧ (1/4) * (α + β) < 67.5 :=
begin
  sorry
end

end correct_angle_calculation_l136_136588


namespace num_5_digit_palindromes_eq_900_l136_136173

-- Definitions of the conditions
def is_palindrome (n : Nat) : Prop :=
  let digits := to_digits 10 n
  digits = List.reverse digits

def is_5_digit (n : Nat) : Prop :=
  n >= 10000 ∧ n < 100000

def is_valid_start (n : Nat) : Prop :=
  let digits := to_digits 10 n
  digits.head ≠ 0

def to_digits (base : Nat) (n : Nat) : List Nat :=
  if n = 0 then [0]
  else List.reverse $ List.unfoldr (λ n, if n = 0 then none else some (n % base, n / base)) n

-- Prove the number of 5-digit palindromes is 900
theorem num_5_digit_palindromes_eq_900 :
  {n : Nat | is_5_digit n ∧ is_valid_start n ∧ is_palindrome n}.card = 900 :=
  sorry

end num_5_digit_palindromes_eq_900_l136_136173


namespace twenty_fifth_digit_after_decimal_point_of_sum_is_zero_l136_136450

noncomputable def x : ℚ := 1 / 8
noncomputable def y : ℚ := 1 / 11
noncomputable def x_decimal_representation : ℚ := 125 / 1000  -- Equivalent to 0.125
noncomputable def y_decimal_representation : ℚ := 9 / 100 + 9 / 10000  -- Equivalent to 0.090909...

theorem twenty_fifth_digit_after_decimal_point_of_sum_is_zero :
  ∀ (n : ℕ), (n > 0) → digit_after_decimal (x + y) 25 = 0 :=
by
  sorry

end twenty_fifth_digit_after_decimal_point_of_sum_is_zero_l136_136450


namespace minimum_value_of_a_l136_136625

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - Real.log x

theorem minimum_value_of_a (a : ℝ) :
  (∀ x ∈ Ioo 1 2, deriv (f a) x ≥ 0) → a ≥ Real.exp ⁻¹ :=
by
  simp [f]
  sorry

end minimum_value_of_a_l136_136625


namespace permutations_condition_l136_136147

theorem permutations_condition (n : ℕ) : 
  let a : ℕ → ℕ := 
    λ n, if n = 0 then 1 else finset.sum (finset.range n) (λ i, a i)
  in a n = 2^(n-1) :=
by
  sorry

end permutations_condition_l136_136147


namespace percentage_increase_l136_136713

theorem percentage_increase (x : ℝ) (hx : x = 123.2) : ((x - 88) / 88) * 100 = 40 := by
  rw [hx]
  have h1 : ((123.2 - 88) / 88) = 0.4 := by norm_num
  rw [h1]
  norm_num
  -- sorry

end percentage_increase_l136_136713


namespace problem_solution_l136_136802

theorem problem_solution (A B : ℝ) (h : ∀ x, x ≠ 3 → (A / (x - 3)) + B * (x + 2) = (-4 * x^2 + 14 * x + 38) / (x - 3)) : 
  A + B = 46 :=
sorry

end problem_solution_l136_136802


namespace modulus_of_z_l136_136825

noncomputable def z : ℂ := sorry

theorem modulus_of_z (hz: z^2 = 3 + 4 * complex.I) : complex.abs z = real.sqrt 5 :=
by
  -- Assuming necessary definitions from the conditions
  sorry

end modulus_of_z_l136_136825


namespace find_pairs_l136_136552

theorem find_pairs (m n: ℕ) (h: m > 0 ∧ n > 0 ∧ m + n - (3 * m * n) / (m + n) = 2011 / 3) : (m = 1144 ∧ n = 377) ∨ (m = 377 ∧ n = 1144) :=
by sorry

end find_pairs_l136_136552


namespace concyclic_iff_l136_136346

-- Definitions of points and intersections
variables {A B C D E F : Type*}

-- Given conditions
axiom quadrilateral (ABCD : Type) : Prop
axiom intersection_AD_BC (E : Type) : Prop
axiom intersection_AB_DC (F : Type) : Prop

-- Statement to prove
theorem concyclic_iff (A B C D E F : Type) :
  quadrilateral ABCD → intersection_AD_BC E → intersection_AB_DC F →
  (concyclic A B C D ↔ (AE * ED + FA * FB = EF^2)) :=
sorry

end concyclic_iff_l136_136346


namespace numeral_of_place_face_value_difference_l136_136864

theorem numeral_of_place_face_value_difference (P F : ℕ) (H : P - F = 63) (Hface : F = 7) : P = 70 :=
sorry

end numeral_of_place_face_value_difference_l136_136864


namespace sum_of_two_digit_factors_of_8060_l136_136176

theorem sum_of_two_digit_factors_of_8060 : ∃ (a b : ℕ), (10 ≤ a ∧ a < 100) ∧ (10 ≤ b ∧ b < 100) ∧ (a * b = 8060) ∧ (a + b = 127) :=
by sorry

end sum_of_two_digit_factors_of_8060_l136_136176


namespace minimum_time_required_l136_136890

-- Definition of the problem
def information_spread (n : ℕ) : ℕ :=
  2^(n + 1) - 1

-- Theorem statement
theorem minimum_time_required (n : ℕ) (h : information_spread n ≥ 1_000_000) : n ≥ 19 :=
sorry

end minimum_time_required_l136_136890


namespace no_finite_set_of_vectors_with_properties_l136_136367

variable (R : Type*) [Ring R]
variable (V : Type*) [AddCommGroup V] [Module R V]

theorem no_finite_set_of_vectors_with_properties 
  (N : ℕ) (hN : N > 3)
  (S : Finset V) (hS1 : 2 * N < S.card)
  (h1 : ∀ (s : Finset V), s.card = N → ∃ (t : Finset V), t ⊆ S ∧ t.card = N - 1 ∧ (s ∪ t).sum = 0)
  (h2 : ∀ (s : Finset V), s.card = N → ∃ (t : Finset V), t ⊆ S ∧ t.card = N ∧ (s ∪ t).sum = 0) :
  False :=
sorry

end no_finite_set_of_vectors_with_properties_l136_136367


namespace set_intersection_l136_136689

def M : Set ℤ := {x | x * (x - 3) ≤ 0}
def N : Set ℝ := {x | 0 < x ∧ x < Real.exp 1}

theorem set_intersection (M : Set ℤ) (N : Set ℝ) :
  M ∩ N = {1, 2} :=
sorry

end set_intersection_l136_136689


namespace find_principal_l136_136465

theorem find_principal
  (A : ℝ) (r : ℝ) (t : ℝ) (P : ℝ)
  (hA : A = 896)
  (hr : r = 0.05)
  (ht : t = 12 / 5) :
  P = 800 ↔ A = P * (1 + r * t) :=
by {
  sorry
}

end find_principal_l136_136465


namespace expected_difference_coffee_tea_l136_136155

-- Define the total number of days in a leap year
def leap_year_days : ℕ := 366

-- Define the probabilities for coffee and tea based on die rolls
def prob_coffee : ℚ := 4 / 7
def prob_tea : ℚ := 3 / 7

-- Define the expected number of days drinking coffee and tea
def expected_days_coffee : ℚ := prob_coffee * leap_year_days
def expected_days_tea : ℚ := prob_tea * leap_year_days

-- Statement: Proving the expected difference
theorem expected_difference_coffee_tea : (expected_days_coffee - expected_days_tea).natAbs = 52 :=
by
  sorry

end expected_difference_coffee_tea_l136_136155


namespace proj_b_v_is_correct_l136_136332

open Real

noncomputable def a : Vector ℝ := sorry
noncomputable def b : Vector ℝ := sorry

axiom orthogonal_a_b : a ⬝ b = 0

noncomputable def v : Vector ℝ := Vec2 4 (-2)
noncomputable def proj_a_v : Vector ℝ := Vec2 (-4/5) (-8/5)

axiom proj_a_v_property : proj a v = proj_a_v

theorem proj_b_v_is_correct : proj b v = Vec2 (24/5) (-2/5) :=
sorry

end proj_b_v_is_correct_l136_136332


namespace average_gas_mileage_round_trip_l136_136125

noncomputable def average_gas_mileage
  (d1 d2 : ℕ) (m1 m2 : ℕ) : ℚ :=
  let total_distance := d1 + d2
  let total_fuel := (d1 / m1) + (d2 / m2)
  total_distance / total_fuel

theorem average_gas_mileage_round_trip :
  average_gas_mileage 150 180 25 15 = 18.3 := by
  sorry

end average_gas_mileage_round_trip_l136_136125


namespace cube_inequality_of_greater_l136_136600

variable (a b : ℝ)

theorem cube_inequality_of_greater (h : a > b) : a^3 > b^3 :=
sorry

end cube_inequality_of_greater_l136_136600


namespace min_value_of_a_l136_136661

noncomputable def min_a_monotone_increasing : ℝ :=
  let f : ℝ → ℝ := λ x a, a * exp x - log x in 
  let a_min : ℝ := exp (-1) in
  a_min

theorem min_value_of_a 
  {f : ℝ → ℝ} 
  {a : ℝ}
  (mono_incr : ∀ x ∈ Ioo 1 2, 0 ≤ deriv (λ x, a * exp x - log x) x) :
  a ≥ exp (-1) :=
begin
  sorry
end

end min_value_of_a_l136_136661


namespace cuberoot_sum_eq_l136_136961

theorem cuberoot_sum_eq (a : ℝ) (h : a = 2^7 + 2^7 + 2^7) : 
  real.cbrt a = 4 * real.cbrt 6 :=
by {
  sorry
}

end cuberoot_sum_eq_l136_136961


namespace total_ways_to_buy_l136_136760

-- Definitions:
def h : ℕ := 9  -- number of headphones
def m : ℕ := 13 -- number of mice
def k : ℕ := 5  -- number of keyboards
def s_km : ℕ := 4 -- number of sets of keyboard and mouse
def s_hm : ℕ := 5 -- number of sets of headphones and mouse

-- Theorem stating the number of ways to buy three items is 646.
theorem total_ways_to_buy : (s_km * h) + (s_hm * k) + (h * m * k) = 646 := by
  -- Placeholder proof using sorry.
  sorry

end total_ways_to_buy_l136_136760


namespace N_transform_l136_136807

variable {R : Type} [Field R]
variable {n : Type} [Fintype n] [DecidableEq n]
variable {m : Type} [Fintype m] [DecidableEq m]

variable (N : Matrix m n R)
variable (x y : Vector n R)

axiom N_x : N.mul_vec x = ![3, -2]
axiom N_y : N.mul_vec y = ![-1, 4]

theorem N_transform :
  N.mul_vec (3 • x - 4 • y) = ![13, -22] :=
by
  sorry

end N_transform_l136_136807


namespace rectangle_area_problem_l136_136130

/--
Given a rectangle with dimensions \(3x - 4\) and \(4x + 6\),
show that the area of the rectangle equals \(12x^2 + 2x - 24\) if and only if \(x \in \left(\frac{4}{3}, \infty\right)\).
-/
theorem rectangle_area_problem 
  (x : ℝ) 
  (h1 : 3 * x - 4 > 0)
  (h2 : 4 * x + 6 > 0) :
  (3 * x - 4) * (4 * x + 6) = 12 * x^2 + 2 * x - 24 ↔ x > 4 / 3 :=
sorry

end rectangle_area_problem_l136_136130


namespace tan_subtraction_formula_l136_136278

noncomputable def tan_sub (alpha beta : ℝ) : ℝ := (tan alpha - tan beta) / (1 + tan alpha * tan beta)

theorem tan_subtraction_formula (alpha beta : ℝ) (h1 : tan alpha = 9) (h2 : tan beta = 5) : tan_sub alpha beta = 2 / 23 :=
by
  simp [tan_sub, h1, h2]
  sorry

end tan_subtraction_formula_l136_136278


namespace mutual_independence_of_A_and_D_l136_136432

noncomputable theory

variables (Ω : Type) [ProbabilitySpace Ω]
-- Definition of events A, B, C, D as sets over Ω
def event_A : Event Ω := {ω | some_condition_for_A}
def event_B : Event Ω := {ω | some_condition_for_B}
def event_C : Event Ω := {ω | some_condition_for_C}
def event_D : Event Ω := {ω | some_condition_for_D}

-- Given probabilities
axiom P_A : P(event_A Ω) = 1 / 6
axiom P_B : P(event_B Ω) = 1 / 6
axiom P_C : P(event_C Ω) = 5 / 36
axiom P_D : P(event_D Ω) = 1 / 6

-- Independence definition
def are_independent (X Y : Event Ω) : Prop :=
  P(X ∩ Y) = P(X) * P(Y)

-- The problem statement: proving A and D are independent
theorem mutual_independence_of_A_and_D : are_independent Ω (event_A Ω) (event_D Ω) :=
sorry

end mutual_independence_of_A_and_D_l136_136432


namespace range_of_x_l136_136234

noncomputable def f : ℝ → ℝ := sorry    -- defined as an odd function which is monotonically decreasing on (-∞, 0]
noncomputable def g (x : ℝ) : ℝ := Real.logBase 2 (x + 3) -- g(x) = log_2(x + 3)

variable (x : ℝ)

lemma f_odd (x : ℝ) : f (-x) = -f x := sorry     -- f is odd
lemma f_decreasing (x y : ℝ) (h : x ≤ y ∧ y ≤ 0) : f x ≥ f y := sorry  -- f is decreasing on (-∞, 0]
lemma f_at_1 : f 1 = -1 := sorry

lemma g_domain (x : ℝ) : x > -3 → g x = logBase 2 (x + 3) := sorry  -- domain of g
lemma g_increasing (x y : ℝ) (h : x ≤ y ∧ y > -3) : g x ≤ g y := sorry  -- g is increasing on (-3, +∞)

theorem range_of_x : {x : ℝ | f x ≥ g x} = set.Icc (-3 : ℝ) (-1 : ℝ) :=
by
  sorry

end range_of_x_l136_136234


namespace largest_possible_val_of_sum_of_digits_l136_136701

-- Definitions of conditions
variables {a b c : ℕ} (h_digits : a < 10 ∧ b < 10 ∧ c < 10)
variables {z : ℕ} (h_z_range : 0 < z ∧ z ≤ 15)
variables (h_decimal : (1000 * (z : ℝ)⁻¹ : ℝ) = (10 * a + b) * 10 + c)

-- Proof goal (statement only, no proof required)
theorem largest_possible_val_of_sum_of_digits :
  ∃ a b c, (a < 10 ∧ b < 10 ∧ c < 10) ∧ 
           (0 < z ∧ z ≤ 15) ∧ 
           (1000 * (z : ℝ)⁻¹ : ℝ = (10 * a + b) * 10 + c) ∧ 
           (∀ a' b' c', (a' < 10 ∧ b' < 10 ∧ c' < 10) → 
                        (0 < z ∧ z ≤ 15) → 
                        (1000 * (z : ℝ)⁻¹ : ℝ = (10 * a' + b') * 10 + c') → 
                        (a + b + c ≥ a' + b' + c')).
sorry

end largest_possible_val_of_sum_of_digits_l136_136701


namespace number_of_ways_to_buy_three_items_l136_136732

def headphones : ℕ := 9
def mice : ℕ := 13
def keyboards : ℕ := 5
def keyboard_mouse_sets : ℕ := 4
def headphone_mouse_sets : ℕ := 5

theorem number_of_ways_to_buy_three_items : 
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 := 
by 
  sorry

end number_of_ways_to_buy_three_items_l136_136732


namespace positive_difference_between_probabilities_is_one_eighth_l136_136936

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem positive_difference_between_probabilities_is_one_eighth :
  let p : ℚ := 1 / 2
    n : ℕ := 5
    prob_4_heads := binomial_probability n 4 p
    prob_5_heads := binomial_probability n 5 p
  in |prob_4_heads - prob_5_heads| = 1 / 8 :=
by
  let p : ℚ := 1 / 2
  let n : ℕ := 5
  let prob_4_heads := binomial_probability n 4 p
  let prob_5_heads := binomial_probability n 5 p
  sorry

end positive_difference_between_probabilities_is_one_eighth_l136_136936


namespace max_value_of_sum_real_numbers_l136_136287

-- This indicates we will not do a computation for maximum but rather validate the given statement.
noncomputable def maxValueOfSum (x y : ℝ) (h : x - 3 * real.sqrt (x + 1) = 3 * real.sqrt (y + 2) - y) : ℝ :=
  x + y
 
theorem max_value_of_sum_real_numbers :
  ∀ x y : ℝ, (x - 3 * real.sqrt (x + 1) = 3 * real.sqrt (y + 2) - y) → maxValueOfSum x y = 9 + 3 * real.sqrt 15 :=
  by sorry

end max_value_of_sum_real_numbers_l136_136287


namespace total_league_games_l136_136380

/-- The Great Eighteen Soccer League is divided into three divisions, each containing six teams. 
 Each team plays each of the other teams in its own division three times and every team in the other divisions twice.
 We want to prove that the total number of league games scheduled is 351.
-/
theorem total_league_games :
  (Σ (d : Fin 3), Σ (i : Fin 6), Σ (j : Fin 6), (i ≠ j ∧ ((∀ k : Fin 6, i*k + j*k + 1) = 3) ∨ ((∀ l : Fin 3, i*l + j*l + 1) = 2))) = 351 := 
sorry

end total_league_games_l136_136380


namespace polyhedron_volume_correct_l136_136151

variables {Point : Type} [EuclideanSpace Point]

structure Rectangle (P : Type) :=
(A B C D : P)
(width length : ℝ)
(is_rectangle : ∀ (AB AD : line_segment P), AB.length = width ∧ AD.length = length ∧
  (∃ (M : P), M = midpoint A C ∧ ∀ (X : P), (X = B ∨ X = D) → distance A X = distance C X))

noncomputable def polyhedron_volume
  (A B C D A' B' C' D' : Point)
  (width_base length_base width_top length_top height : ℝ)
  (base : Rectangle Point) (top : Rectangle Point)
  (parallel : ∀ {X Y Z W : Point}, Rectangle.is_rectangle base X Y Z W ∧ Rectangle.is_rectangle top X Y Z W)
  (AB_eq : distance A B = width_base)
  (AD_eq : distance A D = length_base)
  (A'B'_eq : distance A' B' = width_top)
  (A'D'_eq : distance A' D' = length_top)
  (height_eq : distance A A' = height ∧ height = sqrt 3) : ℝ :=
  let volume_prism := width_base * length_base * height in
  let volume_pyramid := (1 / 3) * width_base * length_base * height in
  volume_prism + 4 * volume_pyramid

theorem polyhedron_volume_correct :
  ∀ (A B C D A' B' C' D' : Point)
  (width_base length_base width_top length_top height : ℝ)
  (base : Rectangle Point) (top : Rectangle Point)
  (parallel : ∀ {X Y Z W : Point}, Rectangle.is_rectangle base X Y Z W ∧ Rectangle.is_rectangle top X Y Z W)
  (AB_eq : distance A B = width_base)
  (AD_eq : distance A D = length_base)
  (A'B'_eq : distance A' B' = width_top)
  (A'D'_eq : distance A' D' = length_top)
  (height_eq : distance A A' = height ∧ height = sqrt 3),
  polyhedron_volume A B C D A' B' C' D' width_base length_base width_top length_top height base top parallel AB_eq AD_eq A'B'_eq A'D'_eq height_eq
  = 121 * (sqrt 3) / 3 := by
  sorry

end polyhedron_volume_correct_l136_136151


namespace quadratic_equation_in_one_variable_l136_136459

-- Definitions for each condition
def equation_A (x : ℝ) : Prop := x^2 = -1
def equation_B (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def equation_C (x : ℝ) : Prop := 2 * (x + 1)^2 = (Real.sqrt 2 * x - 1)^2
def equation_D (x : ℝ) : Prop := x + 1 / x = 1

-- Main theorem statement
theorem quadratic_equation_in_one_variable (x : ℝ) :
  equation_A x ∧ ¬(∃ a b c, equation_B a b c x ∧ a ≠ 0) ∧ ¬equation_C x ∧ ¬equation_D x :=
  sorry

end quadratic_equation_in_one_variable_l136_136459


namespace find_non_intersecting_segments_l136_136341

variables {α : Type} [linear_order α]

structure Point :=
(x : α)
(y : α)

def is_red (p : Point) : Prop := sorry
def is_blue (p : Point) : Prop := sorry

def A (n : ℕ) : set (Point α) := sorry

def no_three_collinear (A : set (Point α)) : Prop :=
∀ p₁ p₂ p₃ ∈ A, linear_independent ℝ ![p₁, p₂, p₃]

theorem find_non_intersecting_segments (n : ℕ) (A : set (Point α))
  (hA : A.card = 2 * n)
  (h_red : (∃ red_set : set (Point α), red_set.card = n ∧ ∀ p ∈ red_set, is_red p) )
  (h_blue : (∃ blue_set : set (Point α), blue_set.card = n ∧ ∀ p ∈ blue_set, is_blue p) )
  (h_no_collinear : no_three_collinear A) :
  ∃ segments : set (Point α × Point α),
    segments.card = n ∧
    (∀ (seg ∈ segments), is_red seg.1 ∧ is_blue seg.2) ∧
    pairwise (λ s t, ¬intersect s t) segments := sorry

end find_non_intersecting_segments_l136_136341


namespace ratio_of_areas_of_quadrilaterals_l136_136824

section

variables {R R1 : ℝ} (hR1_gt_R : R1 > R)
variables (A B C D A1 B1 C1 D1 : ℝ → ℝ)
variables (K_ABC : K = circle_centered_at_origin R)
variables (K1_A1B1C1D1 : K1 = circle_centered_at_origin R1)

noncomputable def area_inscribed_quadrilateral (f : ℝ → ℝ) : ℝ := sorry

theorem ratio_of_areas_of_quadrilaterals
  (h1 : ∀ θ, A = f (cos θ * R) ∧ B = f (sin θ * R) ∧ C = f (-cos θ * R) ∧ D = f (-sin θ * R))
  (h2 : ∀ θ, A1 = f (cos θ * R1) ∧ B1 = f (sin θ * R1) ∧ C1 = f (-cos θ * R1) ∧ D1 = f (-sin θ * R1))
  : (area_inscribed_quadrilateral A1 B1 C1 D1) / (area_inscribed_quadrilateral A B C D) = R1^2 / R^2 :=
sorry

end

end ratio_of_areas_of_quadrilaterals_l136_136824


namespace max_value_M_l136_136563

theorem max_value_M : 
  ∃ t : ℝ, (t = (3 / (4 ^ (1 / 3)))) ∧ 
    (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → 
      a^3 + b^3 + c^3 - 3 * a * b * c ≥ t * (a * b^2 + b * c^2 + c * a^2 - 3 * a * b * c)) :=
sorry

end max_value_M_l136_136563


namespace rahim_average_price_l136_136012

/-- 
Rahim bought 40 books for Rs. 600 from one shop and 20 books for Rs. 240 from another.
What is the average price he paid per book?
-/
def books1 : ℕ := 40
def cost1 : ℕ := 600
def books2 : ℕ := 20
def cost2 : ℕ := 240
def totalBooks : ℕ := books1 + books2
def totalCost : ℕ := cost1 + cost2
def averagePricePerBook : ℕ := totalCost / totalBooks

theorem rahim_average_price :
  averagePricePerBook = 14 :=
by
  sorry

end rahim_average_price_l136_136012


namespace value_of_ff5_l136_136037

noncomputable def f : ℝ → ℝ := sorry

axiom condition1 : ∀ x : ℝ, f(x + 2) = -f(x)
axiom condition2 : f(1) = -5

theorem value_of_ff5 : f(f(5)) = 5 :=
by sorry

end value_of_ff5_l136_136037


namespace david_marks_in_physics_l136_136171

theorem david_marks_in_physics 
  (english_marks mathematics_marks chemistry_marks biology_marks : ℕ)
  (num_subjects : ℕ)
  (average_marks : ℕ)
  (h1 : english_marks = 81)
  (h2 : mathematics_marks = 65)
  (h3 : chemistry_marks = 67)
  (h4 : biology_marks = 85)
  (h5 : num_subjects = 5)
  (h6 : average_marks = 76) :
  ∃ physics_marks : ℕ, physics_marks = 82 :=
by
  sorry

end david_marks_in_physics_l136_136171


namespace circle_tangent_standard_eq_l136_136597

noncomputable def circle_standard_eq (x y : ℝ) (r : ℝ) : String :=
  "(" ++ x.toString ++ " - (3 / 5))^2 + (" ++ y.toString ++ " - (4 / 5))^2 = " ++ r.toString

theorem circle_tangent_standard_eq (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 = 1 → x^2 + y^2 - 6 * x - 8 * y + m = 0) →
  ∃ M : ℝ × ℝ, 
    (M = (3 / 5, 4 / 5) ∨ M = (-3 / 5, -4 / 5)) ∧
    (circle_standard_eq M.1 M.2 16 = "(x - (3 / 5))^2 + (y - (4 / 5))^2 = 16" ∨
     circle_standard_eq M.1 M.2 16 = "(x + (3 / 5))^2 + (y + (4 / 5))^2 = 16") :=
begin
  -- Proof goes here
  sorry
end

end circle_tangent_standard_eq_l136_136597


namespace domino_chain_color_arrangement_l136_136888

-- Definition of dominoes and sets
inductive Color
| white
| blue
| red

structure Domino :=
(value_left : Nat)
(value_right : Nat)
(color : Color)

def isValidChain (dominos : List Domino) : Prop :=
  ∀ i, i < dominos.length - 1 → 
    dominos[i].value_right = dominos[i + 1].value_left ∧
    dominos[i].color ≠ dominos[i + 1].color

-- Problem statement in Lean 4
theorem domino_chain_color_arrangement
  (W B R : List Domino) (chain : List Domino)
  (hW : ∀ d ∈ W, d.color = Color.white)
  (hB : ∀ d ∈ B, d.color = Color.blue)
  (hR : ∀ d ∈ R, d.color = Color.red)
  (hDominoRule : ∀ d, d ∈ W ∨ d ∈ B ∨ d ∈ R → 
      ∃ l r, d = ⟨l, r, d.color⟩ ∧ l < r)
  (hChain : chain = W ++ B ++ R) :
  isValidChain chain :=
sorry

end domino_chain_color_arrangement_l136_136888


namespace number_of_valid_arrangements_l136_136059

def valid_arrangements : Finset (Finset.Perm (Fin 5) (Fin 5)) :=
  (Finset.Perm (Fin 5)).filter (λ p,
    p 2 = 2 ∧
    ¬((p 0 = 1 ∧ p 1 = 2) ∨ (p 1 = 2 ∧ p 2 = 1) ∨
      (p 2 = 3 ∧ p 3 = 2) ∨ (p 3 = 2 ∧ p 4 = 3) ∨
      (p 4 = 3 ∧ p 3 = 4) ∨ (p 3 = 4 ∧ p 4 = 3)))

theorem number_of_valid_arrangements : valid_arrangements.card = 16 := by
  sorry

end number_of_valid_arrangements_l136_136059


namespace monotonically_increasing_or_decreasing_intervals_f_plus_g_geq_zero_inequality_for_x_n_l136_136255

-- Defining the function f and its derivative g and respective properties.
def f (x : ℝ) : ℝ := Real.exp x * Real.cos x
noncomputable def g (x : ℝ) : ℝ := Real.exp x * (Real.cos x - Real.sin x)

theorem monotonically_increasing_or_decreasing_intervals :
  ∀ (k : ℤ),
    (∀ x ∈ Icc ((2 * k : ℝ) * Real.pi - 3 * Real.pi / 4) ((2 * k : ℝ) * Real.pi + Real.pi / 4), 0 < g x) ∧
    (∀ x ∈ Icc ((2 * k : ℝ) * Real.pi + Real.pi / 4) ((2 * k : ℝ) * Real.pi + 5 * Real.pi / 4), g x < 0) := sorry

theorem f_plus_g_geq_zero (x : ℝ) (h : x ∈ Icc (Real.pi / 4) (Real.pi / 2)) :
  f x + g (Real.pi / 2 - x) ≥ 0 := sorry

theorem inequality_for_x_n (n : ℕ) (x₀ : ℝ) (h₀ : ∀ n, f (2 * n * Real.pi + x₀) = 1) :
  let xₙ := Classical.choose (ExistsUnique.exists (ExistsUnique.intro (λ x, f x - 1 = 0) sorry sorry))
  in  2 * n * Real.pi + Real.pi / 2 - xₙ < Real.exp (-2 * n * Real.pi) / (Real.sin x₀ - Real.cos x₀) := sorry

end monotonically_increasing_or_decreasing_intervals_f_plus_g_geq_zero_inequality_for_x_n_l136_136255


namespace cuberoot_sum_eq_l136_136960

theorem cuberoot_sum_eq (a : ℝ) (h : a = 2^7 + 2^7 + 2^7) : 
  real.cbrt a = 4 * real.cbrt 6 :=
by {
  sorry
}

end cuberoot_sum_eq_l136_136960


namespace bisector_equation_l136_136138

noncomputable def a : ℝ := 3 -- a is given as 3 in the problem statement

def P := (-3 : ℝ, 4 : ℝ)
def Q := (-12 : ℝ, -10 : ℝ)
def R := (4 : ℝ, -2 : ℝ)

theorem bisector_equation (c : ℝ) (calculated_value_a_plus_c : ℝ) : 
  (3 * P.1 + P.2 + c = 0) →
  (a + c = calculated_value_a_plus_c) := by
  sorry

end bisector_equation_l136_136138


namespace geese_minimum_swaps_l136_136058

theorem geese_minimum_swaps : 
  let initial_even_geese := (list.range' 2 20 2).reverse;
      initial_odd_geese := list.range' 1 20 2;
      initial_config := initial_even_geese ++ initial_odd_geese;
      ordered_config := list.range 1 21;
      num_inversions := initial_even_geese.length * initial_odd_geese.length
  in 
  list.swaps_to_sort initial_config ordered_config = 55 :=
by
  sorry

end geese_minimum_swaps_l136_136058


namespace complex_problem_l136_136211

open Complex

theorem complex_problem (a b : ℝ) (h : (2 + b * Complex.I) / (1 - Complex.I) = a * Complex.I) : a + b = 4 := by
  sorry

end complex_problem_l136_136211


namespace A_and_D_independent_l136_136430

variable (Ω : Type) [Fintype Ω] [ProbabilitySpace Ω]

namespace BallDrawing

def events (ω₁ ω₂ : Ω) : Prop :=
  (ω₁ = 1 ∧ ω₂ = 2) ∨
  (ω₁ + ω₂ = 8) ∨
  (ω₁ + ω₂ = 7)

def A (ω₁ ω₂ : Ω) : Prop := ω₁ = 1
def B (ω₁ ω₂ : Ω) : Prop := ω₂ = 2
def C (ω₁ ω₂ : Ω) : Prop := ω₁ + ω₂ = 8
def D (ω₁ ω₂ : Ω) : Prop := ω₁ + ω₂ = 7

theorem A_and_D_independent : 
  ∀ Ω [Fintype Ω] [ProbabilitySpace Ω], 
  independence (event A) (event D) :=
by sorry

end BallDrawing

end A_and_D_independent_l136_136430


namespace inequality_solution_set_l136_136883

theorem inequality_solution_set (x : ℝ) : (x-1)/(x+2) > 1 → x < -2 := sorry

end inequality_solution_set_l136_136883


namespace range_of_a_l136_136688

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (x ∈ {x : ℝ | x ≥ 3 ∨ x ≤ -1} ∩ {x : ℝ | x ≤ a} ↔ x ∈ {x : ℝ | x ≤ a})) ↔ a ≤ -1 :=
by sorry

end range_of_a_l136_136688


namespace matrix_power_l136_136325

variable (A : Matrix (Fin 2) (Fin 2) ℤ)

theorem matrix_power
  (h : A •! (Vector.vec' ![5, -2]) = Vector.vec' ![-15, 6]) :
  A ^ 5 •! (Vector.vec' ![5, -2]) = Vector.vec' ![-1215, 486] := 
by
  sorry

end matrix_power_l136_136325


namespace complex_power_identity_l136_136517

theorem complex_power_identity (i : ℂ) (h : i^2 = -1) : (1 + i)^4 = -4 :=
by
  sorry

end complex_power_identity_l136_136517


namespace total_games_played_l136_136064

theorem total_games_played (months : ℕ) (games_per_month : ℕ) (h1 : months = 17) (h2 : games_per_month = 19) : 
  months * games_per_month = 323 :=
by
  sorry

end total_games_played_l136_136064


namespace arithmetic_sequence_n_equals_8_l136_136590

theorem arithmetic_sequence_n_equals_8
  (a : ℕ → ℝ) 
  (h_arith : ∀ n m, a (n + 1) - a n = a (m + 1) - a m) 
  (h2 : a 2 + a 5 = 18)
  (h3 : a 3 * a 4 = 32)
  (h_n : ∃ n, a n = 128) :
  ∃ n, a n = 128 ∧ n = 8 := 
sorry

end arithmetic_sequence_n_equals_8_l136_136590


namespace max_value_trig_sum_l136_136197

theorem max_value_trig_sum :
  ∀ (θ₁ θ₂ θ₃ θ₄ θ₅ θ₆ : ℝ),
  (cos θ₁ * sin θ₂ + cos θ₂ * sin θ₃ + cos θ₃ * sin θ₄ + cos θ₄ * sin θ₅ + cos θ₅ * sin θ₆ + cos θ₆ * sin θ₁) ≤ 3 :=
sorry

end max_value_trig_sum_l136_136197


namespace seven_digit_number_count_l136_136907

theorem seven_digit_number_count :
  let digits := {1, 2, 3, 4, 5, 6, 7}
  let valid_num_arrangements (num_list: List ℕ) : Prop :=
    3 ∈ num_list.tail ∧
    (∀ i j, num_list.nth i = some 1 ∧ num_list.nth j = some 2 → i < j) ∧
    (num_list.length = 7) ∧
    (num_list.to_set ⊆ digits ∧ digits ⊆ num_list.to_set)
  -- using the digits 1 to 7, digit 3 is the last and digit 1 is left of digit 2
  ∃ (num_list : List ℕ), valid_num_arrangements num_list ∧ num_list.last = some 3 →
  -- the total valid arrangements
  num_list.to_set.size = 360 := sorry

end seven_digit_number_count_l136_136907


namespace locus_of_vertex_C_l136_136898

noncomputable def is_equilateral (A B C : Point) : Prop :=
dist A B = dist B C ∧ dist B C = dist C A

noncomputable def locus_of_C (A : Point) (e : Line) : Locus :=
let f := rotate_line e A (π / 3) in
let f' := rotate_line e A (- (π / 3)) in
f ∪ f'

theorem locus_of_vertex_C (A B C : Point) (e : Line) :
fixed A → 
moves_along B e → 
is_equilateral A B C → 
locus_of_C A e = rotate_line e A (π / 3) ∪ rotate_line e A (- (π / 3)) :=
by sorry

end locus_of_vertex_C_l136_136898


namespace count_neither_3_nor_4_l136_136269

def is_multiple_of_3_or_4 (n : Nat) : Bool := (n % 3 = 0) ∨ (n % 4 = 0)

def three_digit_numbers := List.range' 100 900 -- Generates a list from 100 to 999 (inclusive)

def count_multiples_of_3_or_4 : Nat := three_digit_numbers.filter is_multiple_of_3_or_4 |>.length

def count_total := 900 -- Since three-digit numbers range from 100 to 999

theorem count_neither_3_nor_4 : count_total - count_multiples_of_3_or_4 = 450 := by
  sorry

end count_neither_3_nor_4_l136_136269


namespace tan_of_alpha_l136_136245

theorem tan_of_alpha (x y : ℝ) (h : x^2 + y^2 = 1) (hx : x = -4/5) (hy : y = 3/5) :
  Real.tan (arctan (y / x)) = -3 / 4 := by
  sorry

end tan_of_alpha_l136_136245


namespace tanA_tanB_eq_thirteen_div_four_l136_136047

-- Define the triangle and its properties
variables {A B C : Type}
variables (a b c : ℝ)  -- sides BC, AC, AB
variables (HF HC : ℝ)  -- segments of altitude CF
variables (tanA tanB : ℝ)

-- Given conditions
def orthocenter_divide_altitude (HF HC : ℝ) : Prop :=
  HF = 8 ∧ HC = 18

-- The result we want to prove
theorem tanA_tanB_eq_thirteen_div_four (h : orthocenter_divide_altitude HF HC) : 
  tanA * tanB = 13 / 4 :=
  sorry

end tanA_tanB_eq_thirteen_div_four_l136_136047


namespace annual_average_growth_rate_l136_136479

theorem annual_average_growth_rate (x : ℝ) (h : x > 0): 
  100 * (1 + x)^2 = 169 :=
sorry

end annual_average_growth_rate_l136_136479


namespace fair_coin_flip_probability_difference_l136_136938

noncomputable def choose : ℕ → ℕ → ℕ
| _, 0 => 1
| 0, _ => 0
| n + 1, r + 1 => choose n r + choose n (r + 1)

theorem fair_coin_flip_probability_difference :
  let p := 1 / 2
  let p_heads_4_out_of_5 := choose 5 4 * p^4 * p
  let p_heads_5_out_of_5 := p^5
  p_heads_4_out_of_5 - p_heads_5_out_of_5 = 1 / 8 :=
by
  sorry

end fair_coin_flip_probability_difference_l136_136938


namespace range_of_m_l136_136702

variable {f : ℝ → ℝ}

def is_decreasing (f : ℝ → ℝ) := ∀ x y, x < y → f x > f y

theorem range_of_m (hf_dec : is_decreasing f) (hf_odd : ∀ x, f (-x) = -f x) 
  (h : ∀ m, f (m - 1) + f (2 * m - 1) > 0) : ∀ m, m < 2 / 3 :=
by
  sorry

end range_of_m_l136_136702


namespace sum_of_constants_eq_17_l136_136889

theorem sum_of_constants_eq_17
  (x y : ℝ)
  (a b c d : ℕ)
  (ha : a = 6)
  (hb : b = 2)
  (hc : c = 3)
  (hd : d = 3)
  (h1 : x + y = 4)
  (h2 : 3 * x * y = 4)
  (h3 : x = (a + b * Real.sqrt c) / d ∨ x = (a - b * Real.sqrt c) / d) :
  a + b + c + d = 17 :=
sorry

end sum_of_constants_eq_17_l136_136889


namespace fair_coin_flip_difference_l136_136921

theorem fair_coin_flip_difference :
  let P1 := 5 * (1 / 2)^5;
  let P2 := (1 / 2)^5;
  P1 - P2 = 9 / 32 :=
by
  sorry

end fair_coin_flip_difference_l136_136921


namespace positive_difference_between_probabilities_is_one_eighth_l136_136932

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem positive_difference_between_probabilities_is_one_eighth :
  let p : ℚ := 1 / 2
    n : ℕ := 5
    prob_4_heads := binomial_probability n 4 p
    prob_5_heads := binomial_probability n 5 p
  in |prob_4_heads - prob_5_heads| = 1 / 8 :=
by
  let p : ℚ := 1 / 2
  let n : ℕ := 5
  let prob_4_heads := binomial_probability n 4 p
  let prob_5_heads := binomial_probability n 5 p
  sorry

end positive_difference_between_probabilities_is_one_eighth_l136_136932


namespace total_length_of_horizontal_lines_l136_136004

theorem total_length_of_horizontal_lines (a b : ℕ) (h₁ : a = 6) (h₂ : b = 10) :
  ∃ length : ℕ, length = 27 :=
by
  let triangle_length : ℕ := 27
  use triangle_length
  sorry

end total_length_of_horizontal_lines_l136_136004


namespace number_of_squares_l136_136016

theorem number_of_squares (k : ℕ) (S : fin (k+1) × fin (k+1)) : 
  ∑ n in finset.range (k+1), (k + 1 - n) * (k + 1 - n) = k * (k + 1) * (k + 2) / 12 :=
by
  sorry

end number_of_squares_l136_136016


namespace maximize_profit_of_store_l136_136998

-- Define the basic conditions
variable (basketball_cost volleyball_cost : ℕ)
variable (total_items budget : ℕ)
variable (price_increase_factor : ℚ)
variable (school_basketball_cost school_volleyball_difference school_volleyball_cost : ℕ)
variable (reduced_basketball_price_diff reduced_volleyball_price_diff : ℚ)

-- Definitions from conditions
def store_plans := total_items = 200
def budget_constraint := budget ≤ 5000
def cost_price_basketball := basketball_cost = 30
def cost_price_volleyball := volleyball_cost = 24
def selling_price_relation := price_increase_factor = 1.5
def school_basketball_cost_condition := school_basketball_cost = 1800
def school_volleyball_difference_condition := school_volleyball_difference = 10
def school_volleyball_cost_condition := school_volleyball_cost = 1500

-- First Problem: Finding selling price
def volleyball_selling_price (x : ℚ) := 
  school_volleyball_cost / x - school_basketball_cost / (price_increase_factor * x) = school_volleyball_difference

-- Second Problem: Maximizing profit
def profit (a : ℚ) (basketball_selling_price volleyball_selling_price : ℚ) :=
  let reduced_basketball_price := basketball_selling_price - reduced_basketball_price_diff
  let reduced_volleyball_price := volleyball_selling_price - reduced_volleyball_price_diff
  let profit_from_basketball := (reduced_basketball_price - basketball_cost) * a
  let profit_from_volleyball := (reduced_volleyball_price - volleyball_cost) * (200 - a)
  profit_from_basketball + profit_from_volleyball

def budget_constraint_for_max_profit (a : ℚ) := 
  let total_basketball_cost := basketball_cost * a
  let total_volleyball_cost := volleyball_cost * (200 - a)
  total_basketball_cost + total_volleyball_cost ≤ budget

theorem maximize_profit_of_store :
  ∃ volleyball_selling_price basketball_selling_price (a : ℚ),
    volleyball_selling_price volleyball_selling_price ∧ 
    budget_constraint_for_max_profit a ∧
    a = 33 ∧ (200 - a) = 167 := 
sorry

end maximize_profit_of_store_l136_136998


namespace original_proposition_converse_proposition_false_inverse_proposition_false_contrapositive_proposition_true_l136_136469

-- Definition of the quadrilateral being a rhombus
def is_rhombus (quad : Type) : Prop := 
-- A quadrilateral is a rhombus if and only if all its sides are equal in length
sorry

-- Definition of the diagonals of quadrilateral being perpendicular
def diagonals_are_perpendicular (quad : Type) : Prop := 
-- The diagonals of a quadrilateral are perpendicular
sorry

-- Original proposition: If a quadrilateral is a rhombus, then its diagonals are perpendicular to each other
theorem original_proposition (quad : Type) : is_rhombus quad → diagonals_are_perpendicular quad :=
sorry

-- Converse proposition: If the diagonals of a quadrilateral are perpendicular to each other, then it is a rhombus, which is False
theorem converse_proposition_false (quad : Type) : diagonals_are_perpendicular quad → ¬ is_rhombus quad :=
sorry

-- Inverse proposition: If a quadrilateral is not a rhombus, then its diagonals are not perpendicular, which is False
theorem inverse_proposition_false (quad : Type) : ¬ is_rhombus quad → ¬ diagonals_are_perpendicular quad :=
sorry

-- Contrapositive proposition: If the diagonals of a quadrilateral are not perpendicular, then it is not a rhombus, which is True
theorem contrapositive_proposition_true (quad : Type) : ¬ diagonals_are_perpendicular quad → ¬ is_rhombus quad :=
sorry

end original_proposition_converse_proposition_false_inverse_proposition_false_contrapositive_proposition_true_l136_136469


namespace triangle_vector_relation_l136_136787

open_locale real_inner_product_space

variables {V : Type*} [inner_product_space ℝ V]

def on_side (A B C D : V) : Prop := 
  ∃ (t : ℝ), (0 ≤ t) ∧ (t ≤ 1) ∧ (D = t • B + (1-t) • C)

theorem triangle_vector_relation
  (A B C D : V)
  (hD_on_side : on_side B C D)
  (hCD_2DB : D - C = 2 • (B - D))
  (hAD : ∃ r s : ℝ, A - D = r • (A - B) + s • (A - C)) :
  ∃ r s : ℝ, r + s = 1 :=
  sorry

end triangle_vector_relation_l136_136787


namespace school_selection_l136_136994

theorem school_selection :
  (nat.choose 5 3) * nat.factorial 3 = 60 := by 
  sorry

end school_selection_l136_136994


namespace S_2023_value_l136_136606

-- Define the sequence {a_n} and its sum S_n
def a_n (n : ℕ) : ℚ := 
  if n = 1 then
    ∑ k in finset.range 2023, (b (k + 1)) / (2 ^ (k + 1))
  else 
    S (n - 1) * S n

def S (n : ℕ) : ℚ := 
  ∑ i in finset.range n, a_n (i + 1)

-- Polynomial expansion given
def polynomial (x : ℚ) : ℚ := 
  finset.sum (finset.range (2023 + 1)) (λ k => b k * x ^ k)

-- Given polynomial condition:
lemma poly_condition : polynomial (-2) = (1 - 2 * (-2)) ^ 2023 := 
  sorry 

-- Main proof statement
theorem S_2023_value : S 2023 = -1 / 2023 :=
by 
  have b0_eq : b 0 = 1 := sorry
  have a1_eq : a_n 1 = -1 := 
    by rw [a_n]; exact b0_eq
  sorry

end S_2023_value_l136_136606


namespace parallel_lines_iff_l136_136681

theorem parallel_lines_iff (a : ℝ) : 
  (∀ x y : ℝ, (ax + 3 * y + 1 = 0) ∧ (x + (a - 2) * y + a = 0)) ↔ (a = 3) := 
begin
  sorry
end

end parallel_lines_iff_l136_136681


namespace pythagorean_triple_solution_l136_136867

theorem pythagorean_triple_solution
  (x y z a b : ℕ)
  (h1 : x^2 + y^2 = z^2)
  (h2 : Nat.gcd x y = 1)
  (h3 : 2 ∣ y)
  (h4 : a > b)
  (h5 : b > 0)
  (h6 : (Nat.gcd a b = 1))
  (h7 : ((a % 2 = 1 ∧ b % 2 = 0) ∨ (a % 2 = 0 ∧ b % 2 = 1))) 
  : (x = a^2 - b^2 ∧ y = 2 * a * b ∧ z = a^2 + b^2) := 
sorry

end pythagorean_triple_solution_l136_136867


namespace total_number_of_configurations_is_correct_l136_136185

open Matrix

theorem total_number_of_configurations_is_correct :
  let grid : Matrix (Fin 3) (Fin 3) ℕ := λ _ _, 0 in -- Placeholder, the actual grid will be filled accordingly
  let integers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9} in
  -- condition: Each integer should appear exactly once in the grid
  ∀ g : Matrix (Fin 3) (Fin 3) ℕ, 
  (∀ i j : Fin 3, g i j ∈ integers) ∧
  (∀ n ∈ integers, ∃! (i, j) : (Fin 3) × (Fin 3), g i j = n) ∧
  -- condition: Sum of each row is odd
  (∀ i : Fin 3, (Σ j, g i j) % 2 = 1) ∧
  -- condition: Sum of each column is odd
  (∀ j : Fin 3, (Σ i, g i j) % 2 = 1) →
  -- Prove the number of satisfying configurations is 25920
  ∃ num_configs : ℕ, num_configs = 25920 := 
sorry

end total_number_of_configurations_is_correct_l136_136185


namespace total_apples_count_l136_136418

-- Definitions based on conditions
def red_apples := 16
def green_apples := red_apples + 12
def total_apples := green_apples + red_apples

-- Statement to prove
theorem total_apples_count : total_apples = 44 := 
by
  sorry

end total_apples_count_l136_136418


namespace num_ways_to_buy_three_items_l136_136769

-- Defining the conditions based on the problem statement
def num_headphones : ℕ := 9
def num_mice : ℕ := 13
def num_keyboards : ℕ := 5
def num_kb_mouse_sets : ℕ := 4
def num_hp_mouse_sets : ℕ := 5

-- Defining the theorem statement
theorem num_ways_to_buy_three_items :
  (num_kb_mouse_sets * num_headphones) + 
  (num_hp_mouse_sets * num_keyboards) + 
  (num_headphones * num_mice * num_keyboards) = 646 := 
  by
  sorry

end num_ways_to_buy_three_items_l136_136769


namespace minimum_value_of_a_l136_136663

theorem minimum_value_of_a (a : ℝ) (h : ∀ x ∈ set.Ioo 1 2, ae^x - 1/x ≥ 0) : a ≥ e⁻¹ :=
sorry

end minimum_value_of_a_l136_136663


namespace remainder_when_divided_by_30_l136_136457

theorem remainder_when_divided_by_30 (n k R m : ℤ) (h1 : 0 ≤ R ∧ R < 30) (h2 : 2 * n % 15 = 2) (h3 : n = 30 * k + R) : R = 1 := by
  sorry

end remainder_when_divided_by_30_l136_136457


namespace min_value_of_a_l136_136662

noncomputable def min_a_monotone_increasing : ℝ :=
  let f : ℝ → ℝ := λ x a, a * exp x - log x in 
  let a_min : ℝ := exp (-1) in
  a_min

theorem min_value_of_a 
  {f : ℝ → ℝ} 
  {a : ℝ}
  (mono_incr : ∀ x ∈ Ioo 1 2, 0 ≤ deriv (λ x, a * exp x - log x) x) :
  a ≥ exp (-1) :=
begin
  sorry
end

end min_value_of_a_l136_136662


namespace trapezoid_dc_length_l136_136184

-- Define the variables and the problem structure
variables (A B C D : Type) 
          [normed_add_comm_group A] [normed_add_comm_group B]
          [metric_space A] [metric_space B] [normed_space ℝ A] 
          [normed_space ℝ B] 

-- Conditions defining the trapezoid
def trapezoid (A B C D : Type) : Prop :=
  parallel A B C D ∧ length_segment A B = 7 ∧ length_segment B C = 6 ∧
  angle B C D = 60 ∧ angle C D A = 45

-- Main theorem to prove the length of DC
theorem trapezoid_dc_length {A B C D : Type} [normed_add_comm_group A] [normed_add_comm_group B] 
  [metric_space A] [metric_space B] [normed_space ℝ A] [normed_space ℝ B] 
  (h : trapezoid A B C D):
  length_segment C D = 10 + 3 * real.sqrt 3 :=
begin
  sorry
end

end trapezoid_dc_length_l136_136184


namespace find_efg_correct_l136_136878

noncomputable def find_efg (M : ℕ) : ℕ :=
  let efgh := M % 10000
  let e := efgh / 1000
  let efg := efgh / 10
  if (M^2 % 10000 = efgh) ∧ (e ≠ 0) ∧ ((M % 32 = 0 ∧ (M - 1) % 125 = 0) ∨ (M % 125 = 0 ∧ (M - 1) % 32 = 0))
  then efg
  else 0
  
theorem find_efg_correct {M : ℕ} (h_conditions: (M^2 % 10000 = M % 10000) ∧ (M % 32 = 0 ∧ (M - 1) % 125 = 0 ∨ M % 125 = 0 ∧ (M-1) % 32 = 0) ∧ ((M % 10000 / 1000) ≠ 0)) :
  find_efg M = 362 :=
by
  sorry

end find_efg_correct_l136_136878


namespace greatest_integer_x_l136_136076

theorem greatest_integer_x (x : ℤ) (h : (5 : ℚ) / 8 > (x : ℚ) / 17) : x ≤ 10 :=
by
  sorry

end greatest_integer_x_l136_136076


namespace geometric_progression_19th_term_is_18th_power_l136_136486

theorem geometric_progression_19th_term_is_18th_power 
  (a : ℕ) (m n : ℕ) (progression : ∀ k, k ∈ (Finset.range 37).map (λ k, a * (m / n)^k))
  (coprime_first_last : Nat.coprime a (a * (m / n)^36)) 
  : ∃ k : ℕ, (a * (m / n)^18) = k^18 :=
by
  sorry

end geometric_progression_19th_term_is_18th_power_l136_136486


namespace sqrt_three_squared_l136_136456

theorem sqrt_three_squared : (Real.sqrt 3) ^ 2 = 3 := by
  sorry

end sqrt_three_squared_l136_136456


namespace find_b_l136_136389

-- Define the function
def f (x: ℝ) (b: ℝ) : ℝ := 2^x + b

-- Define the inverse function point condition
def inverse_point_condition (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∃ y, f y = p.1 ∧ y = p.2

-- State the problem
theorem find_b (b : ℝ) : inverse_point_condition (λ x, f x b) (5, 2) → b = 1 :=
by
  -- Proof is skipped
  sorry

end find_b_l136_136389


namespace minimum_a_for_monotonic_increasing_on_interval_l136_136651

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp x - Real.log x

noncomputable def f_prime (a x : ℝ) : ℝ := a * Real.exp x - 1 / x

def min_value_of_a : ℝ := Real.exp (-1)

theorem minimum_a_for_monotonic_increasing_on_interval :
  (∀ x ∈ Set.Ioo 1 2, 0 ≤ f_prime a x) ↔ a ≥ min_value_of_a :=
by
  sorry

end minimum_a_for_monotonic_increasing_on_interval_l136_136651


namespace pseudometric_d_pseudometric_rho_d_zero_not_imply_eq_rho_zero_not_imply_eq_l136_136845

-- Define the distance measures
def d (A B : Set α) [MeasurableSpace α] (P : Measure α) := P (A.symmDiff B)

def rho (A B : Set α) [MeasurableSpace α] (P : Measure α) := 
  if P (A ∪ B) ≠ 0 then P (A.symmDiff B) / P (A ∪ B) else 0

-- Define the pseudometric properties
theorem pseudometric_d (A B C : Set α) [MeasurableSpace α] (P : Measure α) :
  d A B P ≤ d A C P + d B C P :=
sorry

theorem pseudometric_rho (A B C : Set α) [MeasurableSpace α] (P : Measure α) 
  (hAB : P (A ∪ B) ≠ 0) (hAC : P (A ∪ C) ≠ 0) (hBC : P (B ∪ C) ≠ 0) :
  rho A B P ≤ rho A C P + rho B C P :=
sorry

-- Show that d(A, B) = 0 does not imply A = B
theorem d_zero_not_imply_eq (A B : Set α) [MeasurableSpace α] (P : Measure α) :
  d A B P = 0 → ¬ (A ≠ B) :=
sorry

-- Show that rho(A, B) = 0 does not imply A = B
theorem rho_zero_not_imply_eq (A B : Set α) [MeasurableSpace α] (P : Measure α) (hAB : P (A ∪ B) ≠ 0) :
  rho A B P = 0 → ¬ (A ≠ B) :=
sorry

end pseudometric_d_pseudometric_rho_d_zero_not_imply_eq_rho_zero_not_imply_eq_l136_136845


namespace b_is_arith_and_geom_seq_l136_136222

-- Define the initial conditions of the arithmetic sequence {a_n}
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

-- Define the sum of the first n terms S_n
def sum_first_n_terms (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  n * (a 0 + a (n - 1)) / 2

-- Given conditions in the problem
variables (a : ℕ → ℕ) (S_21 : ℕ)
hypothesis (h_arith_seq : is_arithmetic_sequence a)
hypothesis (h_S21_eq_42 : sum_first_n_terms a 21 = 42)

-- Define b_n based on the formula provided in the problem
def b (n : ℕ) : ℕ := 2 ^ (a 11 ^ 2 - a 9 - a 13)

-- Final proof problem
theorem b_is_arith_and_geom_seq : 
  is_arithmetic_sequence a →
  sum_first_n_terms a 21 = 42 →
  ∀ n m : ℕ, b n = b m :=
by
  intros h_arith_seq h_S21_eq_42
  -- Placeholder for the proof steps
  sorry

end b_is_arith_and_geom_seq_l136_136222


namespace minimum_value_of_a_l136_136643

theorem minimum_value_of_a (a : ℝ) :
  (∀ x ∈ (Ioo 1 2), ae^x - ln x ≥ 0) ↔ a ≥ exp(-1) := sorry

end minimum_value_of_a_l136_136643


namespace period_sin_sub_cos_l136_136916

theorem period_sin_sub_cos : ∀ x : ℝ, sin (x + 2 * π) - cos (x + 2 * π) = sin x - cos x :=
by
  intro x
  calc
    sin (x + 2 * π) - cos (x + 2 * π) = sin x - cos (x + 2 * π)    : by rw [sin_add, cos_add, cos_two_pi, sin_two_pi, add_zero, zero_add, sub_zero]
    ...                               = sin x - cos x               : by rw [cos_two_pi]
    ...                               = sin x - cos x               : by sorry  -- Proof of the equivalence

end period_sin_sub_cos_l136_136916


namespace parabola_fixed_point_l136_136682

theorem parabola_fixed_point (C : Type _) [Field C] (p : C) (P Q T : C × C)
    (h1 : y^2 = 2*p*x)
    (hp : p > 0)
    (h2 : T = (P.1, -P.2))
    (k : C) (h3 : k = 1)
    (h_PQ : dist P Q = 16)
    (h_F : F = (p / 2, 0))
    (h_l : line = { (x, y) | y = x - p / 2 }) :
    let C_eqn := y^2 = 8x in
    let fixed_point := (-2, 0) in
    line T Q passing fixed_point :=
by
  sorry

end parabola_fixed_point_l136_136682


namespace proj_b_v_is_correct_l136_136333

open Real

noncomputable def a : Vector ℝ := sorry
noncomputable def b : Vector ℝ := sorry

axiom orthogonal_a_b : a ⬝ b = 0

noncomputable def v : Vector ℝ := Vec2 4 (-2)
noncomputable def proj_a_v : Vector ℝ := Vec2 (-4/5) (-8/5)

axiom proj_a_v_property : proj a v = proj_a_v

theorem proj_b_v_is_correct : proj b v = Vec2 (24/5) (-2/5) :=
sorry

end proj_b_v_is_correct_l136_136333


namespace reflection_across_x_axis_l136_136782

variables {x1 x2 y1 y2 : ℝ}

theorem reflection_across_x_axis (hP : (x1, y1))
                                (hQ : (x2, y2))
                                (hReflectP : (x1, -y1) = (x1, -y1))
                                (hReflectQ : (x2, -y2) = (x2, -y2))
                                (hT : (x1, -y1) = (x1, -y1))
                                (hU : (x2, -y2) = (x2, -y2)) :
  (x1, -y1) = (x1, -y1) ∧ (x2, -y2) = (x2, -y2) → 
  ((x1, y1) = (x1, -y1) ∧ (x2, y2) = (x2, -y2)) :=
by 
  intros h1 h2 h3 h4 h5 h6 
  exact sorry

end reflection_across_x_axis_l136_136782


namespace min_value_of_a_l136_136632

theorem min_value_of_a (a : ℝ) : 
  (∀ x ∈ Ioo 1 2, (a * exp x - 1/x) ≥ 0) → a = exp (-1) :=
by
  sorry

end min_value_of_a_l136_136632


namespace g_of_3_eq_seven_over_two_l136_136215

theorem g_of_3_eq_seven_over_two :
  ∀ f g : ℝ → ℝ,
  (∀ x, f x = (2 * x + 3) / (x - 1)) →
  (∀ x, g x = (x + 4) / (x - 1)) →
  g 3 = 7 / 2 :=
by
  sorry

end g_of_3_eq_seven_over_two_l136_136215


namespace range_m_condition1_range_m_condition2_l136_136225

def p (m : ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-1 : ℝ) 0, Real.logBase 2 (x + 2) < 2 * m

def q (m : ℝ) : Prop :=
  let discriminant := 4 - 4 * m^2
  discriminant > 0

theorem range_m_condition1 (m : ℝ) :
  (¬ p m ∧ q m) → -1 < m ∧ m ≤ 1 / 2 :=
by
  sorry

theorem range_m_condition2 (m : ℝ) :
  ((p m ∨ q m) ∧ ¬ (p m ∧ q m)) → (m ∈ Set.Icc (-1 : ℝ) (1 / 2) ∨ m ∈ Set.Ici 1) :=
by
  sorry

end range_m_condition1_range_m_condition2_l136_136225


namespace construct_tangent_point_l136_136906

-- Definitions for the problem
structure Circle where
  center : Point \<open>
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

def is_tangent_length (P : Point) (C : Circle) (l : ℝ) : Prop :=
  let distance := (P.center - C.center).norm
  distance = sqrt (C.radius * C.radius + l * l)

theorem construct_tangent_point (O1 O2 : Point) (r1 r2 l1 l2 : ℝ) :
  ∃ P : Point, is_tangent_length P (Circle.mk O1 r1) l1 ∧ is_tangent_length P (Circle.mk O2 r2) l2 :=
  sorry

end construct_tangent_point_l136_136906


namespace case_a_perimeter_two_ab_case_b_min_perimeter_case_c_min_perimeter_l136_136202

-- Case (a)
theorem case_a_perimeter_two_ab (A B : Point) (l : Line) 
-- Conditions on A, B and l
(h_A_ne_B : A ≠ B)
(h_A_opposite_B_on_l : IsOppositeSide A B l)
: perimeter (triangle A B (projection_point A B l)) = 2 * (distance A B) :=
sorry

-- Case (b)
theorem case_b_min_perimeter (A : Point) (l m : Line) 
-- Conditions on A, l, and m
(h_A_ne_line_l : not (A ∈ l))
(h_A_ne_line_m : not (A ∈ m))
: ∃ B C, B ∈ l ∧ C ∈ m ∧ perimeter (triangle A B C) = degenerate_case_perimeter :=
sorry

-- Case (c)
theorem case_c_min_perimeter (l m n : Line) 
(h_intersect : Intersect l m n)
: ∃ A B C, A ∈ l ∧ B ∈ m ∧ C ∈ n ∧ perimeter (triangle A B C) = degenerate_case_perimeter :=
sorry

end case_a_perimeter_two_ab_case_b_min_perimeter_case_c_min_perimeter_l136_136202


namespace quadrilateral_area_l136_136296

/-- Given a quadrilateral ABCD with the following properties:
  - the angle ∠BCD is right (i.e., 90 degrees),
  - the side lengths are AB=15, BC=4, CD=3, and AD=14,
  - a perpendicular is dropped from point D to line AB, hitting at point E,
we aim to prove that the area of quadrilateral ABCD is equal to (15 / 2) * sqrt 189.75 + 6. --/
theorem quadrilateral_area (ABCD : Type) [quadrilateral ABCD]
  (AB CD : ℝ) (BC : ℝ) (AD : ℝ) (angle_BCD : ℝ)
  (h1 : angle_BCD = 90)
  (h2 : AB = 15) (h3 : BC = 4) (h4 : CD = 3) (h5 : AD = 14) :
  area ABCD = (15 / 2) * sqrt 189.75 + 6 :=
sorry

end quadrilateral_area_l136_136296


namespace maximize_profit_of_store_l136_136999

-- Define the basic conditions
variable (basketball_cost volleyball_cost : ℕ)
variable (total_items budget : ℕ)
variable (price_increase_factor : ℚ)
variable (school_basketball_cost school_volleyball_difference school_volleyball_cost : ℕ)
variable (reduced_basketball_price_diff reduced_volleyball_price_diff : ℚ)

-- Definitions from conditions
def store_plans := total_items = 200
def budget_constraint := budget ≤ 5000
def cost_price_basketball := basketball_cost = 30
def cost_price_volleyball := volleyball_cost = 24
def selling_price_relation := price_increase_factor = 1.5
def school_basketball_cost_condition := school_basketball_cost = 1800
def school_volleyball_difference_condition := school_volleyball_difference = 10
def school_volleyball_cost_condition := school_volleyball_cost = 1500

-- First Problem: Finding selling price
def volleyball_selling_price (x : ℚ) := 
  school_volleyball_cost / x - school_basketball_cost / (price_increase_factor * x) = school_volleyball_difference

-- Second Problem: Maximizing profit
def profit (a : ℚ) (basketball_selling_price volleyball_selling_price : ℚ) :=
  let reduced_basketball_price := basketball_selling_price - reduced_basketball_price_diff
  let reduced_volleyball_price := volleyball_selling_price - reduced_volleyball_price_diff
  let profit_from_basketball := (reduced_basketball_price - basketball_cost) * a
  let profit_from_volleyball := (reduced_volleyball_price - volleyball_cost) * (200 - a)
  profit_from_basketball + profit_from_volleyball

def budget_constraint_for_max_profit (a : ℚ) := 
  let total_basketball_cost := basketball_cost * a
  let total_volleyball_cost := volleyball_cost * (200 - a)
  total_basketball_cost + total_volleyball_cost ≤ budget

theorem maximize_profit_of_store :
  ∃ volleyball_selling_price basketball_selling_price (a : ℚ),
    volleyball_selling_price volleyball_selling_price ∧ 
    budget_constraint_for_max_profit a ∧
    a = 33 ∧ (200 - a) = 167 := 
sorry

end maximize_profit_of_store_l136_136999


namespace pascal_triangle_41st_number_42nd_row_l136_136912

open Nat

theorem pascal_triangle_41st_number_42nd_row :
  Nat.choose 42 40 = 861 := by
  sorry

end pascal_triangle_41st_number_42nd_row_l136_136912


namespace modular_inverse_11_mod_101_l136_136915

theorem modular_inverse_11_mod_101 : ∃ (x : ℤ), 0 ≤ x ∧ x ≤ 100 ∧ (11 * x) % 101 = 1 :=
by
  -- Define the given condition
  have h_gcd : Int.gcd 11 101 = 1 := by norm_num
  -- Define the correct answer as modulo equality
  use 46
  split
  -- Proof for 0 ≤ 46
  linarith
  split
  -- Proof for 46 ≤ 100
  linarith
  -- Proof for (11 * 46) % 101 = 1
  calc
    (11 * 46) % 101 = 506 % 101 := by ring
               ... = 1        := by norm_num

end modular_inverse_11_mod_101_l136_136915


namespace area_of_inscribed_rectangle_l136_136843

theorem area_of_inscribed_rectangle (h_triangle_altitude : 12 > 0)
  (h_segment_XZ : 15 > 0)
  (h_PQ_eq_one_third_PS : ∀ PQ PS : ℚ, PS = 3 * PQ) :
  ∃ PQ PS : ℚ, 
    (YM = 12) ∧
    (XZ = 15) ∧
    (PQ = (15 / 8 : ℚ)) ∧
    (PS = 3 * PQ) ∧ 
    ((PQ * PS) = (675 / 64 : ℚ)) :=
by
  -- Proof would go here.
  sorry

end area_of_inscribed_rectangle_l136_136843


namespace initial_distance_between_A_and_B_l136_136473

theorem initial_distance_between_A_and_B
  (start_time : ℕ)        -- time in hours, 1 pm
  (meet_time : ℕ)         -- time in hours, 3 pm
  (speed_A : ℕ)           -- speed of A in km/hr
  (speed_B : ℕ)           -- speed of B in km/hr
  (time_walked : ℕ)       -- time walked in hours
  (distance_A : ℕ)        -- distance covered by A in km
  (distance_B : ℕ)        -- distance covered by B in km
  (initial_distance : ℕ)  -- initial distance between A and B

  (h1 : start_time = 1)
  (h2 : meet_time = 3)
  (h3 : speed_A = 5)
  (h4 : speed_B = 7)
  (h5 : time_walked = meet_time - start_time)
  (h6 : distance_A = speed_A * time_walked)
  (h7 : distance_B = speed_B * time_walked)
  (h8 : initial_distance = distance_A + distance_B) :

  initial_distance = 24 :=
by
  sorry

end initial_distance_between_A_and_B_l136_136473


namespace length_of_secant_segment_l136_136584

/-
Given a circle with radius R, a point A is chosen at a distance of 2R from the center O of the circle. From this point, a tangent and a secant are drawn. The secant is equidistant from the center O and the point of tangency B. Prove that the length of the segment of the secant that lies inside the circle is 2R * sqrt(10 / 13).
-/
theorem length_of_secant_segment (R : ℝ) :
  ∃ CG : ℝ, CG = 2 * R * sqrt (10 / 13) :=
sorry

end length_of_secant_segment_l136_136584


namespace polynomial_identity_l136_136231

theorem polynomial_identity (a b c : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a^2 + b^2 + c^2 = 15) 
  (h3 : a^3 + b^3 + c^3 = 47) : 
  (a^2 + ab + b^2) * (b^2 + bc + c^2) * (c^2 + ca + a^2) = 625 := 
by 
  sorry

end polynomial_identity_l136_136231


namespace avg_age_of_new_men_l136_136857

theorem avg_age_of_new_men (A : ℕ) :
  (15 * (A + 2) - 15 * A = (21 + 23) + 30) →
  ((21 + 23) + 30 = 74) →
  (74 / 2 = 37) :=
by
  intros h1 h2
  rw h2
  norm_num
  sorry

end avg_age_of_new_men_l136_136857


namespace part1_part2_l136_136620

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x - 3) - 1 / real.sqrt (7 - x)

def A : set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : set ℝ := {x | a < x ∧ x < 2 * a + 1}

theorem part1 : 
  (A ∪ B = {x | 2 < x ∧ x < 10}) ∧ 
  ((B \ A) = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)}) :=
by sorry

theorem part2 (a : ℝ) : (B ∪ C a = B) → (a ≤ -1 ∨ 2 ≤ a ∧ a ≤ 9 / 2) :=
by sorry

end part1_part2_l136_136620


namespace equation_of_parallel_line_l136_136034

theorem equation_of_parallel_line (x y : ℝ) :
  (∀ b : ℝ, 2 * x + y + b = 0 → x = -1 → y = 2 → b = 0) →
  (∀ x y b: ℝ, 2 * x + y + b = 0 → x = -1 → y = 2 → 2 * x + y = 0) :=
by
  sorry

end equation_of_parallel_line_l136_136034


namespace negation_universal_proposition_l136_136395

theorem negation_universal_proposition {x : ℝ} : 
  (¬ ∀ x : ℝ, x^2 ≥ 2) ↔ (∃ x : ℝ, x^2 < 2) := 
sorry

end negation_universal_proposition_l136_136395


namespace count_numbers_with_zero_l136_136698

theorem count_numbers_with_zero :
  {n : ℕ | n > 0 ∧ n ≤ 1000 ∧ (∃ d ∈ string.to_list (nat.to_string n), d = '0')}.to_finset.card = 181 :=
by
  sorry

end count_numbers_with_zero_l136_136698


namespace range_of_m_l136_136708

theorem range_of_m (m : ℝ) :
  (∀ x : ℕ, (x = 1 ∨ x = 2 ∨ x = 3) → (3 * x - m ≤ 0)) ↔ 9 ≤ m ∧ m < 12 :=
by
  sorry

end range_of_m_l136_136708


namespace ellipse_eq_find_k_l136_136596

open Real

-- Part I: Prove the equation of the ellipse
theorem ellipse_eq (a b : ℝ) (h_a : a = 2) (h_b : b = 1) :
  ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 4 + y^2 = 1 :=
by
  sorry

-- Part II: Prove the value of k
theorem find_k (P : ℝ × ℝ) (hP : P = (-2, 1)) (MN_length : ℝ) (hMN : MN_length = 2) 
  (k : ℝ) (a b : ℝ) (h_a : a = 2) (h_b : b = 1) :
  ∃ k : ℝ, k = -4 :=
by
  sorry

end ellipse_eq_find_k_l136_136596


namespace standard_eq_circle_eq_line_l_l136_136233

-- Define points A and B
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (0, 4)

-- Define the line that the center of the circle lies on
def line_eq (x y : ℝ) := 2 * x - y - 3 = 0

-- Define point T
def T : ℝ × ℝ := (-1, 0)

-- Define the equation of dot product constraint
def dot_product_const (C P Q : ℝ × ℝ) : Prop :=
  let ⟨xC, yC⟩ := C
  let ⟨xP, yP⟩ := P
  let ⟨xQ, yQ⟩ := Q
  ((xP - xC) * (xQ - xC) + (yP - yC) * (yQ - yC)) = -5

-- Prove the standard equation of the circle C
theorem standard_eq_circle :
  (∃ (C : ℝ × ℝ), line_eq C.1 C.2 ∧
  dist C A = dist C B ∧
  (C.1 - 3)^2 + (C.2 - 3)^2 = 10) :=
sorry

-- Prove the equation of line l
theorem eq_line_l :
  (∃ (m b : ℝ), line_eq T.1 (m * T.1 + b) ∧
   (∀ x y : ℝ, (line_eq x y → dist T (x, y) = (sqrt 10) / 2)) ∧
   (x - 3*y + 1 = 0 ∨ 13*x - 9*y + 13 = 0)) :=
sorry

end standard_eq_circle_eq_line_l_l136_136233


namespace no_rain_5_days_probability_l136_136399

open ProbabilityTheory

-- Define the probability of rain on any given day
def prob_rain : ℝ := 2 / 3

-- Define the probability of no rain on any given day
def prob_no_rain : ℝ := 1 - prob_rain

-- Define the probability that it will not rain at all for five consecutive days
noncomputable def prob_no_rain_5_days : ℝ := prob_no_rain ^ 5

-- Statement to prove the final probability
theorem no_rain_5_days_probability : prob_no_rain_5_days = 1 / 243 :=
by
  -- This is where the proof would go
  sorry

end no_rain_5_days_probability_l136_136399


namespace number_of_gooseberries_l136_136892

theorem number_of_gooseberries 
  (total_fruits : ℕ)
  (raspberries_fraction : ℚ)
  (blackberries_fraction : ℚ)
  (blueberries_fraction : ℚ)
  (strawberries_fraction : ℚ)
  (cherries_fraction : ℚ) :
  total_fruits = 240 →
  raspberries_fraction = 1/8 →
  blackberries_fraction = 3/16 →
  blueberries_fraction = 7/30 →
  strawberries_fraction = 5/24 →
  cherries_fraction = 1/30 →
  let raspberries := raspberries_fraction * total_fruits in
  let blackberries := blackberries_fraction * total_fruits in
  let blueberries := blueberries_fraction * total_fruits in
  let strawberries := strawberries_fraction * total_fruits in
  let cherries := cherries_fraction * total_fruits in
  let total_other_fruits := raspberries + blackberries + blueberries + strawberries + cherries in
  let gooseberries := total_fruits - total_other_fruits in
  gooseberries = 51 :=
by {
  intros,
  -- Defining the number of each type of fruit
  let raspberries := raspberries_fraction * total_fruits,
  let blackberries := blackberries_fraction * total_fruits,
  let blueberries := blueberries_fraction * total_fruits,
  let strawberries := strawberries_fraction * total_fruits,
  let cherries := cherries_fraction * total_fruits,
  
  -- Calculating the total of other fruits
  let total_other_fruits := raspberries + blackberries + blueberries + strawberries + cherries,
  
  -- Calculating the number of gooseberries
  let gooseberries := total_fruits - total_other_fruits,

  -- Claim that the number of gooseberries is 51
  exact sorry
}

end number_of_gooseberries_l136_136892


namespace proj_of_b_eq_v_diff_proj_of_a_l136_136326

noncomputable theory

variables (a b : ℝ × ℝ) (v : ℝ × ℝ)

def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (u.1 * v.1 + u.2 * v.2) / (u.1 * u.1 + u.2 * u.2)
  (scalar * u.1, scalar * u.2)

theorem proj_of_b_eq_v_diff_proj_of_a
  (h₀ : orthogonal a b)
  (h₁ : proj a ⟨4, -2⟩ = ⟨-4/5, -8/5⟩)
  : proj b ⟨4, -2⟩ = ⟨24/5, -2/5⟩ :=
sorry

end proj_of_b_eq_v_diff_proj_of_a_l136_136326


namespace number_of_trips_l136_136067

theorem number_of_trips (bags_per_trip : ℕ) (weight_per_bag : ℕ) (total_weight : ℕ)
  (h1 : bags_per_trip = 10)
  (h2 : weight_per_bag = 50)
  (h3 : total_weight = 10000) : 
  total_weight / (bags_per_trip * weight_per_bag) = 20 :=
by
  sorry

end number_of_trips_l136_136067


namespace projection_problem_l136_136808

noncomputable def vector_proj (w v : ℝ × ℝ) : ℝ × ℝ := sorry -- assume this definition

variables (v w : ℝ × ℝ)

-- Given condition
axiom proj_v : vector_proj w v = ⟨4, 3⟩

-- Proof Statement
theorem projection_problem :
  vector_proj w (7 • v + 2 • w) = ⟨28, 21⟩ + 2 • w :=
sorry

end projection_problem_l136_136808


namespace average_speed_of_trip_l136_136992

theorem average_speed_of_trip 
  (d1 d2 : ℝ) (s1 s2 : ℝ)
  (h1 : d1 = 60) (h2 : s1 = 30)
  (h3 : d2 = 65) (h4 : s2 = 65) :
  (d1 + d2) / (d1 / s1 + d2 / s2) = 125 / 3 := 
by {
  have t1 : d1 / s1 = 2, by linarith [h1, h2],
  have t2 : d2 / s2 = 1, by linarith [h3, h4],
  have total_distance : d1 + d2 = 125, by linarith [h1, h3],
  have total_time : d1 / s1 + d2 / s2 = 3, by linarith [t1, t2],
  rw [total_distance, total_time],
  norm_num
}

end average_speed_of_trip_l136_136992


namespace possible_values_of_c_l136_136283

theorem possible_values_of_c (a b c : ℕ) (n : ℕ) (h₀ : a ≠ 0) (h₁ : n = 729 * a + 81 * b + 36 + c) (h₂ : ∃ k, n = k^3) :
  c = 1 ∨ c = 8 :=
sorry

end possible_values_of_c_l136_136283


namespace eccentricity_of_ellipse_l136_136614

variables {a b c x y x0 y0 : ℝ}
variables (h1 : a > b) (h2 : b > 0)

def ellipse_eq : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def vertex_A1 : Prop := x0 = -a ∧ y0 = 0
def line_eq : Prop := b * x + a * y = 0
def circle_eq : Prop := (x + a)^2 + y^2 = a^2
def symmetric_point' : Prop := b * ((x0 - a) / 2) + a * (y0 / 2) = 0 ∧ (y0 / (x0 + a)) = (a / b)
def symmetric_point_coords : Prop := x0 = -((a * c^2) / (a^2 + b^2)) ∧ y0 = (2 * a^2 * b) / (a^2 + b^2)
def circle_substitution : Prop := ((a - (a * c^2) / (a^2 + b^2))^2) + (((2 * a^2 * b) / (a^2 + b^2))^2) = a^2
def eccentricity_eq : Prop := e = (sqrt 2) / 2

theorem eccentricity_of_ellipse (h_ellipse : ellipse_eq) (h_vertex : vertex_A1) (h_symm_point : symmetric_point')
  (h_coords : symmetric_point_coords) (h_circle : circle_substitution) : eccentricity_eq :=
sorry

end eccentricity_of_ellipse_l136_136614


namespace ellipse_eq_find_k_l136_136595

open Real

-- Part I: Prove the equation of the ellipse
theorem ellipse_eq (a b : ℝ) (h_a : a = 2) (h_b : b = 1) :
  ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 4 + y^2 = 1 :=
by
  sorry

-- Part II: Prove the value of k
theorem find_k (P : ℝ × ℝ) (hP : P = (-2, 1)) (MN_length : ℝ) (hMN : MN_length = 2) 
  (k : ℝ) (a b : ℝ) (h_a : a = 2) (h_b : b = 1) :
  ∃ k : ℝ, k = -4 :=
by
  sorry

end ellipse_eq_find_k_l136_136595


namespace radius_of_triple_volume_sphere_l136_136401

def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

def radius_of_sphere_with_triple_volume (r : ℝ) : ℝ := (³√(3 * volume_of_sphere r / ((4 / 3) * Real.pi)))

theorem radius_of_triple_volume_sphere (r : ℝ) (a b : ℕ) (h_b_cubefree : ¬ ∃ (c : ℕ), b = c^3) 
  (h_radius_eq : radius_of_sphere_with_triple_volume r = a * (³√ b)) : a + b = 8 :=
by
  have r := 5
  have vol := volume_of_sphere r
  have t_vol := 3 * vol
  have rad := radius_of_sphere_with_triple_volume r
  have rad_correct : rad = a * (³√ b) := sorry
  have a := 5
  have b := 3
  show a + b = 8, sorry

end radius_of_triple_volume_sphere_l136_136401


namespace ratio_of_areas_l136_136440

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ :=
1 / 2 * a * b

theorem ratio_of_areas (a b c x y z : ℝ)
  (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) 
  (h4 : x = 9) (h5 : y = 12) (h6 : z = 15)
  (h7 : a^2 + b^2 = c^2) (h8 : x^2 + y^2 = z^2) :
  (area_of_right_triangle a b) / (area_of_right_triangle x y) = 4 / 9 :=
sorry

end ratio_of_areas_l136_136440


namespace selling_price_correct_l136_136502

-- Cost price of the apple
def cost_price : ℝ := 21

-- Fraction of cost price he sells it for
def selling_price_fraction : ℝ := 5 / 6

-- Selling price should be 5/6 of the cost price
def selling_price : ℝ := selling_price_fraction * cost_price

theorem selling_price_correct : selling_price = 17.50 :=
by
  sorry

end selling_price_correct_l136_136502


namespace triangle_angle_A_l136_136717

noncomputable def law_of_sines (a b c A B C : ℝ) := a / sin A = b / sin B ∧ b / sin B = c / sin C

theorem triangle_angle_A (a b c A B C : ℝ)
  (h : law_of_sines a b c A B C)
  (h_condition : a / sin B + b / sin A = 2 * c)
  (angle_range: ∀ (x : ℝ), (0 < x ∧ x < π) → (0 < sin x ∧ sin x ≤ 1))  : A = π / 4 :=
by
  sorry

end triangle_angle_A_l136_136717


namespace cow_calf_ratio_l136_136492

theorem cow_calf_ratio (cost_cow cost_calf : ℕ) (h_cow : cost_cow = 880) (h_calf : cost_calf = 110) :
  cost_cow / cost_calf = 8 :=
by {
  sorry
}

end cow_calf_ratio_l136_136492


namespace minimum_value_of_a_l136_136664

theorem minimum_value_of_a (a : ℝ) (h : ∀ x ∈ set.Ioo 1 2, ae^x - 1/x ≥ 0) : a ≥ e⁻¹ :=
sorry

end minimum_value_of_a_l136_136664


namespace min_omega_period_eq_3_l136_136230

theorem min_omega_period_eq_3 (ω : Real) (hω : ω > 0) :
  ∃ (ω_min : Real), (∀ ω' : Real, (ω' > 0) → (∀ x : Real, sin (ω' * x + π / 3) - 1 = sin (ω' * (x - 2 * π / 3) + π / 3) - 1) → ω' ≥ 3) ∧ ω_min = 3 :=
begin
  use 3,
  split,
  { intros ω' hω' hx,
    have : 2 * π / 3 = 2 * π / ω',
    sorry
  },
  refl
end

end min_omega_period_eq_3_l136_136230


namespace cube_root_of_sum_powers_l136_136958

noncomputable def cube_root (x : ℝ) : ℝ := x ^ (1 / 3)

theorem cube_root_of_sum_powers :
  cube_root (2^7 + 2^7 + 2^7) = 4 * cube_root 2 :=
by
  sorry

end cube_root_of_sum_powers_l136_136958


namespace team_count_l136_136050

theorem team_count :
  let girls := 4
  let boys := 7
  let choose (n k : ℕ) := nat.choose n k
  choose girls 3 * choose boys 2 = 84 := 
by 
  sorry -- proof goes here

end team_count_l136_136050


namespace shop_combinations_l136_136746

theorem shop_combinations :
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let keyboard_and_mouse_sets := 4
  let headphone_and_mouse_sets := 5
  (keyboard_and_mouse_sets * headphones) + (headphone_and_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 :=
by
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let keyboard_and_mouse_sets := 4
  let headphone_and_mouse_sets := 5
  have case1 : keyboard_and_mouse_sets * headphones = 4 * 9, by sorry
  have case2 : headphone_and_mouse_sets * keyboards = 5 * 5, by sorry
  have case3 : headphones * mice * keyboards = 9 * 13 * 5, by sorry
  show (keyboard_and_mouse_sets * headphones) + (headphone_and_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646,
  by rw [case1, case2, case3]; exact sorry

end shop_combinations_l136_136746


namespace ratio_rounded_to_nearest_tenth_l136_136365

theorem ratio_rounded_to_nearest_tenth : (Real.ceil (11 / 15 * 10) / 10 = 0.7) :=
by sorry

end ratio_rounded_to_nearest_tenth_l136_136365


namespace minimum_value_of_a_l136_136629

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - Real.log x

theorem minimum_value_of_a (a : ℝ) :
  (∀ x ∈ Ioo 1 2, deriv (f a) x ≥ 0) → a ≥ Real.exp ⁻¹ :=
by
  simp [f]
  sorry

end minimum_value_of_a_l136_136629


namespace find_n_l136_136273

theorem find_n (n : ℕ) (a b : ℕ) (h1 : n ≥ 3) 
  (h2 : (x + 2)^n = x^n + ... + a * x^3 + b * x^2 + c * x + 2^n) 
  (h3 : a / b = 3 / 2) : n = 11 :=
begin
  sorry
end

end find_n_l136_136273


namespace distance_from_circumcenter_to_centroid_right_triangle_l136_136615

noncomputable def distance_from_circumcenter_to_centroid (A B C O G : Point) (a b c : ℝ)
  (h1 : a = 5)
  (h2 : b = 12)
  (h3 : c = 13)
  (triangle_ABC : is_right_triangle A B C a b c)
  (circumcenter_O : is_circumcenter O A B C)
  (centroid_G : is_centroid G A B C)
  : ℝ :=
  distance O G

theorem distance_from_circumcenter_to_centroid_right_triangle (A B C O G : Point)
  (h1a : 5 = dist A B)
  (h1b : 12 = dist B C)
  (h1c : 13 = dist A C)
  (triangle_ABC : is_right_triangle A B C (dist A B) (dist B C) (dist A C))
  (circumcenter_O : is_circumcenter O A B C)
  (centroid_G : is_centroid G A B C) :
  distance_from_circumcenter_to_centroid A B C O G (dist A B) (dist B C) (dist A C) h1a h1b h1c triangle_ABC circumcenter_O centroid_G = 13/6 :=
by
  sorry

end distance_from_circumcenter_to_centroid_right_triangle_l136_136615


namespace line_passes_fixed_point_l136_136247

-- Define the properties of the ellipse
def ellipse_eq (a b : ℝ) : ℝ → ℝ → Prop :=
  λ x y, (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the eccentricity and relationship for a given ellipse
def eccentricity_eq (a c : ℝ) : ℝ :=
  c / a

-- Define the foci positions based on c
def focus1 (c : ℝ) : ℝ × ℝ :=
  (-c, 0)

def focus2 (c : ℝ) : ℝ × ℝ :=
  (c, 0)

-- Main theorem statement
theorem line_passes_fixed_point :
  ∀ (a b : ℝ),
  a > b → b > 0 →
  eccentricity_eq a 1 = (Real.sqrt 2) / 2 →
  ellipse_eq a b 2 (Real.sqrt 3) →
  (∀ (k m : ℝ) (x1 y1 x2 y2 : ℝ),
    (ellipse_eq a b x1 y1) →
    (ellipse_eq a b x2 y2) →
    y1 = k * x1 + m →
    y2 = k * x2 + m →
    ((k * x2 + m - k * (y1 - m) / k) / (x2 - 1)) +
    ((k * x1 + m - k * (y2 - m) / k) / (x1 - 1)) = 0 →
    m = -2 * k) →
  ∃ (p : ℝ × ℝ), p = (2, 0) := 
sorry

end line_passes_fixed_point_l136_136247


namespace range_m_l136_136879

noncomputable def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, f = λ x, a * x^2 + b * x + c

theorem range_m (f : ℝ → ℝ) 
  (h_quad : quadratic_function f) 
  (h_sym : ∀ x, f (4 + x) = f (-x))
  (h_f2 : f 2 = 1) 
  (h_f0 : f 0 = 3) 
  (h_minmax : ∀ x ∈ (set.Icc 0 m), f x ≥ 1 ∧ f x ≤ 3) :
  m ∈ set.Icc 2 4 :=
sorry

end range_m_l136_136879


namespace evaluate_g_at_neg2_l136_136084

def g (x : ℝ) : ℝ := x^3 - 3 * x^2 + 4

theorem evaluate_g_at_neg2 : g (-2) = -16 := by
  sorry

end evaluate_g_at_neg2_l136_136084


namespace coin_flip_difference_l136_136954

/-- The positive difference between the probability of a fair coin landing heads up
exactly 4 times out of 5 flips and the probability of a fair coin landing heads up
5 times out of 5 flips is 1/8. -/
theorem coin_flip_difference :
  (5 * (1 / 2) ^ 5) - ((1 / 2) ^ 5) = (1 / 8) :=
by
  sorry

end coin_flip_difference_l136_136954


namespace boat_downstream_time_l136_136474

def time_for_boat_downstream (V_b : ℕ) (V_s : ℕ) (time_up : ℝ) : ℝ :=
  let V_down := V_b + V_s
  let V_up := V_b - V_s
  let D := V_up * time_up
  D / V_down

theorem boat_downstream_time :
  time_for_boat_downstream 15 3 1.5 = 1 :=
by
  sorry

end boat_downstream_time_l136_136474


namespace total_games_in_season_l136_136424

-- Define the constants according to the conditions
def number_of_teams : ℕ := 25
def games_per_pair : ℕ := 15

-- Define the mathematical statement we want to prove
theorem total_games_in_season :
  let round_robin_games := (number_of_teams * (number_of_teams - 1)) / 2 in
  let total_games := round_robin_games * games_per_pair in
  total_games = 4500 :=
by
  sorry

end total_games_in_season_l136_136424


namespace symmetric_difference_A_B_l136_136207

def M_minus_N (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}
def M_xor_N (M N : Set ℝ) : Set ℝ := M_minus_N M N ∪ M_minus_N N M

def A : Set ℝ := {t | ∃ x, t = x^2 - 3 * x}
def B : Set ℝ := {x | ∃ y, y = real.log (-x)}

theorem symmetric_difference_A_B :
  M_xor_N A B = {x | x < -9 / 4 ∨ x ≥ 0} := sorry

end symmetric_difference_A_B_l136_136207


namespace functional_equation_solution_l136_136978

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), (∀ x y : ℝ, (f x + y) * (f (x - y) + 1) = f (f (x * f (x + 1)) - y * f (y - 1))) → (∀ x : ℝ, f x = x) :=
by
  intros f h x
  -- Proof would go here
  sorry

end functional_equation_solution_l136_136978


namespace estimated_fish_in_pond_l136_136774

theorem estimated_fish_in_pond :
  ∀ (number_marked_first_catch total_second_catch number_marked_second_catch : ℕ),
    number_marked_first_catch = 100 →
    total_second_catch = 108 →
    number_marked_second_catch = 9 →
    ∃ est_total_fish : ℕ, (number_marked_second_catch / total_second_catch : ℝ) = (number_marked_first_catch / est_total_fish : ℝ) ∧ est_total_fish = 1200 := 
by
  intros number_marked_first_catch total_second_catch number_marked_second_catch
  sorry

end estimated_fish_in_pond_l136_136774


namespace avg_of_multiples_of_4_is_even_l136_136965

theorem avg_of_multiples_of_4_is_even (m n : ℤ) (hm : m % 4 = 0) (hn : n % 4 = 0) :
  (m + n) / 2 % 2 = 0 := sorry

end avg_of_multiples_of_4_is_even_l136_136965


namespace solution_set_of_inequality_l136_136405

theorem solution_set_of_inequality :
  {x : ℝ | (x-2)*(3-x) > 0} = {x : ℝ | 2 < x ∧ x < 3} :=
sorry

end solution_set_of_inequality_l136_136405


namespace product_of_all_possible_values_l136_136848

theorem product_of_all_possible_values (x : ℝ) (h : 2 * |x + 3| - 4 = 2) :
  ∃ (a b : ℝ), (x = a ∨ x = b) ∧ a * b = 0 :=
by
  sorry

end product_of_all_possible_values_l136_136848


namespace buy_items_ways_l136_136730

theorem buy_items_ways (headphones keyboards mice keyboard_mouse_sets headphone_mouse_sets : ℕ) :
  headphones = 9 → keyboards = 5 → mice = 13 → keyboard_mouse_sets = 4 → headphone_mouse_sets = 5 →
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 :=
by
  intros h_eq k_eq m_eq kms_eq hms_eq
  have h_eq_gen : headphones = 9 := h_eq
  have k_eq_gen : keyboards = 5 := k_eq
  have m_eq_gen : mice = 13 := m_eq
  have kms_eq_gen : keyboard_mouse_sets = 4 := kms_eq
  have hms_eq_gen : headphone_mouse_sets = 5 := hms_eq
  sorry

end buy_items_ways_l136_136730


namespace trig_identity_l136_136984

theorem trig_identity (x : ℝ) : 4 * sin (5 * x) * cos (5 * x) * (cos (x) ^ 4 - sin (x) ^ 4) = sin (4 * x) := 
sorry

end trig_identity_l136_136984


namespace part1_sqrt_transform_part2_general_transform_l136_136363

-- Part (1) Proof to demonstrate the transformation for 4 + 4/15
theorem part1_sqrt_transform : 
  sqrt (4 + 4 / 15) = 8 * sqrt 15 / 15 :=
by sorry

-- Part (2) General proof for any integer n >= 2
theorem part2_general_transform (n : ℕ) (hn : 2 ≤ n) : 
  sqrt (n + n / (n^2 - 1)) = n * sqrt (n / (n^2 - 1)) :=
by sorry

end part1_sqrt_transform_part2_general_transform_l136_136363


namespace a2_value_l136_136577

open BigOperators
open Nat

def f (x : ℚ) : ℚ := ∑ i in (range 10).map (λ n, n + 1), (1 + x) ^ i

theorem a2_value : let a := (range 11).map (λ k, k * (k - 1) / 2)
                  in a.sum = 165 :=
by sorry

end a2_value_l136_136577


namespace count_numbers_with_at_most_two_digits_l136_136268

theorem count_numbers_with_at_most_two_digits (count : ℕ) :
  (∀ n, 0 < n ∧ n < 1000 →
    (∀ d1 d2 d, (d = n.digits → ∃ d1 d2, d1 ≠ d2 ∧ d.all (λ x, x = d1 ∨ x = d2)))) →
  count = 855 :=
sorry

end count_numbers_with_at_most_two_digits_l136_136268


namespace fair_coin_difference_l136_136926

noncomputable def binom_coeff (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem fair_coin_difference :
  let p := (1 : ℚ) / 2 in
  let p_1 := binom_coeff 5 4 * p^4 * (1 - p) in
  let p_2 := p^5 in
  p_1 - p_2 = 1 / 8 :=
by
  sorry

end fair_coin_difference_l136_136926


namespace max_area_triangle_ABC_l136_136779

-- Define given points and distances
variables {P A B C : Type} [EuclideanGeometry P A B C] (PA PB PC BC : ℝ)
variables (angle_ABC_right : ∀ (angle ABC : ℝ), ABC = π/2 → angle ABC = π/2)

-- Define the conditions
def conditions : Prop :=
  (PA = 3) ∧ (PB = 4) ∧ (PC = 5) ∧ (BC = 6) ∧ (angle_ABC_right = true)

-- Define the theorem to prove the maximum area of triangle ABC
theorem max_area_triangle_ABC : conditions → (maximum_area_t ABC = 9) :=
sorry

end max_area_triangle_ABC_l136_136779


namespace count_distinct_sums_l136_136161

def special_fraction (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = 20

def possible_sums : set ℚ :=
  { s | ∃ (a1 a2 a3 b1 b2 b3 : ℕ), 
      special_fraction a1 b1 ∧
      special_fraction a2 b2 ∧
      special_fraction a3 b3 ∧
      s = (a1 / b1 : ℚ) + (a2 / b2) + (a3 / b3) }

theorem count_distinct_sums : (possible_sums.filter (λ x, ∃ n : ℤ, x = n)).card = 6 :=
sorry

end count_distinct_sums_l136_136161


namespace at_least_one_no_real_roots_l136_136815

open Real

-- Definitions of quadratic functions with leading coefficients 1
def quadratic (a b c : ℝ) : ℝ → ℝ := λ x => (a * x^2) + (b * x) + c

-- Conditions
variable (b c d e : ℝ)
def f : ℝ → ℝ := quadratic 1 b c
def g : ℝ → ℝ := quadratic 1 d e

-- Assumption that f(g(x)) = 0 has no real roots
axiom f_g_no_real_roots : ∀ x, f(g(x)) ≠ 0

-- Assumption that g(f(x)) = 0 has no real roots
axiom g_f_no_real_roots : ∀ x, g(f(x)) ≠ 0

-- Theorem to prove
theorem at_least_one_no_real_roots : 
  (∀ x, f(f(x)) ≠ 0) ∨ (∀ x, g(g(x)) ≠ 0) := sorry

end at_least_one_no_real_roots_l136_136815


namespace correct_option_proof_l136_136090

-- Define the expression interpretations based on the options.
def optionA : Prop := "-3^4" = (-3)^4
def optionB : Prop := "-3^4" = -(3^4)
def optionC : Prop := "-3^4 represents the negative of the product of four 3s."
def optionD : Prop := "-3^4" = (-3) * (-3) * (-3) * (-3)

-- Define the correct interpretation.
def correct_interpretation : Prop := -(3^4) = -81

-- The theorem to be proven.
theorem correct_option_proof : optionC ↔ correct_interpretation := by
  sorry

end correct_option_proof_l136_136090


namespace min_value_of_a_l136_136659

noncomputable def min_a_monotone_increasing : ℝ :=
  let f : ℝ → ℝ := λ x a, a * exp x - log x in 
  let a_min : ℝ := exp (-1) in
  a_min

theorem min_value_of_a 
  {f : ℝ → ℝ} 
  {a : ℝ}
  (mono_incr : ∀ x ∈ Ioo 1 2, 0 ≤ deriv (λ x, a * exp x - log x) x) :
  a ≥ exp (-1) :=
begin
  sorry
end

end min_value_of_a_l136_136659


namespace chess_champion_probability_l136_136841

theorem chess_champion_probability :
  let P_R := 0.6
  let P_S := 0.3
  let P_D := 0.1
  let P := 0.06 + 0.126 + 0.024 + 0.021 + 0.03 + 0.072 + 0.01
  1000 * P = 343 :=
by 
  let P_R := 0.6
  let P_S := 0.3
  let P_D := 0.1
  let P := 0.06 + 0.126 + 0.024 + 0.021 + 0.03 + 0.072 + 0.01
  show 1000 * P = 343
  sorry

end chess_champion_probability_l136_136841


namespace olivia_spent_amount_l136_136891

noncomputable def initial_amount : ℕ := 100
noncomputable def collected_amount : ℕ := 148
noncomputable def final_amount : ℕ := 159

theorem olivia_spent_amount :
  initial_amount + collected_amount - final_amount = 89 :=
by
  sorry

end olivia_spent_amount_l136_136891


namespace minimum_value_of_a_l136_136669

theorem minimum_value_of_a (a : ℝ) (h : ∀ x ∈ set.Ioo 1 2, ae^x - 1/x ≥ 0) : a ≥ e⁻¹ :=
sorry

end minimum_value_of_a_l136_136669


namespace min_value_of_a_l136_136637

theorem min_value_of_a (a : ℝ) : 
  (∀ x ∈ Ioo 1 2, (a * exp x - 1/x) ≥ 0) → a = exp (-1) :=
by
  sorry

end min_value_of_a_l136_136637


namespace shortest_broken_line_on_surface_l136_136866

structure Tetrahedron (A B C D : Type _) :=
(perpendicular_opposite_edges : ∀ (u v : Type _), u ≠ v → (u ∈ {A, B, C, D}) → (v ∈ {A, B, C, D}) → (⊥ u v))
(face_BCD_acute_angled : acute_angle B C D)
(perpendicular_from_A_foot : ∃ T, perpendicular B C D A T)

theorem shortest_broken_line_on_surface
  (A B C D : Type _)
  (H : Tetrahedron A B C D) :
  ∀ T, T ∈ set_of_foot_of_perpendicular A B C D →
  ∃ face, (face \in {ABC, ABD, ACD}) ∧ 
          (is_shortest_broken_line A T face) :=
sorry

end shortest_broken_line_on_surface_l136_136866


namespace ellipse_polar_form_sum_of_reciprocals_l136_136776

noncomputable def ellipse_parametric_to_polar (theta : ℝ) : ℝ :=
  2 / sqrt(1 + 3 * sin(theta)^2)

theorem ellipse_polar_form (theta : ℝ) :
  ∃ (rho : ℝ), rho = ellipse_parametric_to_polar theta :=
by
  sorry

-- Defining the polar coordinates of points A and B based on given angle alpha
def rho1 (alpha : ℝ) := 2 / sqrt (1 + 3 * sin alpha ^ 2)
def rho2 (alpha : ℝ) := 2 / sqrt (1 + 3 * cos alpha ^ 2)

theorem sum_of_reciprocals (alpha : ℝ) :
  1 / (rho1 alpha) ^ 2 + 1 / (rho2 alpha) ^ 2 = 5 / 4 :=
by
  sorry

end ellipse_polar_form_sum_of_reciprocals_l136_136776


namespace distance_covered_l136_136475

noncomputable def velocity (t : ℝ) : ℝ := t^2 - 1

theorem distance_covered :
  ∫ t in 0..4, velocity t = 52 / 3 :=
by
  sorry

end distance_covered_l136_136475


namespace range_of_a_l136_136272

-- Definitions for the problem conditions
def exists_solution (a : ℝ) : Prop := ∃ x : ℝ, x + a * x + a = 0

-- The main proof theorem
theorem range_of_a (a : ℝ) : exists_solution a → a ∈ set.Iic 0 ∪ set.Ici 4 :=
by
  intro h
  sorry

end range_of_a_l136_136272


namespace sum_interior_angles_equal_diagonals_l136_136407

theorem sum_interior_angles_equal_diagonals (n : ℕ) (h : n = 4 ∨ n = 5) :
  (n - 2) * 180 = 360 ∨ (n - 2) * 180 = 540 :=
by sorry

end sum_interior_angles_equal_diagonals_l136_136407


namespace proposition_correctness_l136_136239

noncomputable def f : ℝ → ℝ := sorry

lemma even_function (x : ℝ) : f x = f (-x) := sorry

lemma periodic_function (x : ℝ) : f (x + 6) = f x + f 3 := sorry

lemma f_neg4 : f (-4) = -2 := sorry

lemma strictly_increasing (x₁ x₂ : ℝ) (h₁ : 0 ≤ x₁) (h₂ : x₁ ≤ 3) (h₃ : 0 ≤ x₂) (h₄ : x₂ ≤ 3) (h_ne : x₁ ≠ x₂) :
  (f x₁ - f x₂) / (x₁ - x₂) > 0 := sorry

theorem proposition_correctness :
  (f 2008 = -2) ∧
  (∀ x, f (x + 6) = f (-x))  ∧
  (∀ x ∈ Icc (-9 : ℝ) (-6), ∀ x' ∈ Icc (-9 : ℝ) (-6), x < x' → f x > f x') ∧
  (∃ x₁ x₂ x₃ x₄ ∈ Icc (-9 : ℝ) 9, f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0) := sorry

end proposition_correctness_l136_136239


namespace total_amount_paid_l136_136466

variable grapes_quantity : ℕ := 8
variable grapes_rate : ℕ := 80
variable mangoes_quantity : ℕ := 9
variable mangoes_rate : ℕ := 55

theorem total_amount_paid :
  (grapes_quantity * grapes_rate) + (mangoes_quantity * mangoes_rate) = 1135 := 
  by
    sorry

end total_amount_paid_l136_136466


namespace equal_distances_l136_136773

open EuclideanGeometry

variables {A B C D E F G M N : Point}

def acute_triangle (A B C : Point) : Prop :=
  let α := ∠ A B C
  let β := ∠ B C A
  let γ := ∠ C A B
  α < π / 2 ∧ β < π / 2 ∧ γ < π / 2

def tangents_to_circumcircle (circ : Circle) (P Q R : Point) (T U : Point) : Prop :=
  T = tangent_point circ P Q ∧ U = tangent_point circ Q R
  ∧ P = circ.center ∧ Q = circ.center ∧ R = circ.center

def equal_segments (P Q R S : Point) : Prop :=
  dist P Q = dist R S

def intersection_points (A B C D E : Point) (F G M N : Point) : Prop :=
  F = intersection (extension A B) (extension D E) ∧
  G = intersection (extension A C) (extension D E) ∧
  M = intersection (line C F) (line B D) ∧
  N = intersection (line C E) (line B G)

theorem equal_distances {A B C D E F G M N : Point}
  (ht : acute_triangle A B C)
  (hAC60 : ¬∠ A B C = 60)
  (hTang : tangents_to_circumcircle (circumcircle A B C) B C D E)
  (hEqSeg : equal_segments B D C E)
  (hIntPts : intersection_points A B C D E F G M N) :
  dist A M = dist A N := sorry

end equal_distances_l136_136773


namespace probability_of_passing_through_correct_l136_136098

def probability_of_passing_through (n k : ℕ) : ℚ :=
(2 * k * n - 2 * k^2 + 2 * k - 1) / n^2

theorem probability_of_passing_through_correct (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) :
  probability_of_passing_through n k = (2 * k * n - 2 * k^2 + 2 * k - 1) / n^2 := 
by
  sorry

end probability_of_passing_through_correct_l136_136098


namespace log_base_one_half_decreasing_l136_136863

noncomputable def f (x : ℝ) : ℝ := logBase (1 / 2) x

theorem log_base_one_half_decreasing : ∀ ⦃x y : ℝ⦄, 0 < x → x < y → f y < f x :=
by {
  intros x y x_pos x_lt_y,
  -- Proof omitted.
  sorry
}

end log_base_one_half_decreasing_l136_136863


namespace lilith_caps_collection_l136_136358

noncomputable def monthlyCollectionYear1 := 3
noncomputable def monthlyCollectionAfterYear1 := 5
noncomputable def christmasCaps := 40
noncomputable def yearlyCapsLost := 15
noncomputable def totalYears := 5

noncomputable def totalCapsCollectedByLilith :=
  let firstYearCaps := monthlyCollectionYear1 * 12
  let remainingYearsCaps := monthlyCollectionAfterYear1 * 12 * (totalYears - 1)
  let christmasCapsTotal := christmasCaps * totalYears
  let totalCapsBeforeLosses := firstYearCaps + remainingYearsCaps + christmasCapsTotal
  let lostCapsTotal := yearlyCapsLost * totalYears
  let totalCapsAfterLosses := totalCapsBeforeLosses - lostCapsTotal
  totalCapsAfterLosses

theorem lilith_caps_collection : totalCapsCollectedByLilith = 401 := by
  sorry

end lilith_caps_collection_l136_136358


namespace find_third_number_l136_136856

theorem find_third_number 
  (h1 : (14 + 32 + x) / 3 = (21 + 47 + 22) / 3 + 3) : x = 53 := by
  sorry

end find_third_number_l136_136856


namespace proj_of_b_eq_v_diff_proj_of_a_l136_136327

noncomputable theory

variables (a b : ℝ × ℝ) (v : ℝ × ℝ)

def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (u.1 * v.1 + u.2 * v.2) / (u.1 * u.1 + u.2 * u.2)
  (scalar * u.1, scalar * u.2)

theorem proj_of_b_eq_v_diff_proj_of_a
  (h₀ : orthogonal a b)
  (h₁ : proj a ⟨4, -2⟩ = ⟨-4/5, -8/5⟩)
  : proj b ⟨4, -2⟩ = ⟨24/5, -2/5⟩ :=
sorry

end proj_of_b_eq_v_diff_proj_of_a_l136_136327


namespace average_gpa_of_whole_class_l136_136097

-- Define the conditions
variables (n : ℕ)
def num_students_in_group1 := n / 3
def num_students_in_group2 := 2 * n / 3

def gpa_group1 := 15
def gpa_group2 := 18

-- Lean statement for the proof problem
theorem average_gpa_of_whole_class (hn_pos : 0 < n):
  ((num_students_in_group1 * gpa_group1) + (num_students_in_group2 * gpa_group2)) / n = 17 :=
sorry

end average_gpa_of_whole_class_l136_136097


namespace four_lines_circles_single_point_l136_136102

theorem four_lines_circles_single_point 
  (l₁ l₂ l₃ l₄ : Line)
  (h_non_concurrent : ∀ i j k : { i // i ≠ j ∧ j ≠ k ∧ i ≠ k }, meet_at_point l₁ l₂ l₃ l₄ = ∅)
  (h_no_parallel : ∀ i j : { i // i ≠ j }, ¬ parallel l₁ l₂ l₃ l₄)
  : ∃ P : Point, (is_on_circumcircle P (circumscribing_circle l₁ l₂ l₃)) ∧ 
                 (is_on_circumcircle P (circumscribing_circle l₂ l₃ l₄)) ∧ 
                 (is_on_circumcircle P (circumscribing_circle l₁ l₃ l₄)) ∧ 
                 (is_on_circumcircle P (circumscribing_circle l₁ l₂ l₄)) :=
sorry

end four_lines_circles_single_point_l136_136102


namespace store_purchase_ways_l136_136744

theorem store_purchase_ways :
  ∃ (headphones sets_hm sets_km keyboards mice : ℕ), 
  headphones = 9 ∧
  mice = 13 ∧
  keyboards = 5 ∧
  sets_km = 4 ∧
  sets_hm = 5 ∧
  (sets_km * headphones + sets_hm * keyboards + headphones * mice * keyboards = 646) :=
by
  use 9, 5, 4, 5, 13
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  sorry

end store_purchase_ways_l136_136744


namespace mother_is_33_l136_136854

noncomputable def mother's_age_now : ℕ :=
  let D := 7 in
  let years_ago := 10 in 
  let sum_today := 74 in 
  let sum_years_ago := 47 in
  26 + D

theorem mother_is_33 :
  let D := 7 in
  let years_ago := 10 in 
  let sum_today := 74 in 
  let sum_years_ago := 47 in
  let F := sum_today - 3 * D - 26
  in (D + (26 + D) + (sum_today - 3 * D - 26) = sum_today) ∧ 
     ((D - years_ago) + ((26 + D) - years_ago) + ((sum_today - 3 * D - 26) - years_ago) = sum_years_ago) ∧ 
     (26 + D = 33) :=
by
  sorry

end mother_is_33_l136_136854


namespace fair_coin_flip_difference_l136_136920

theorem fair_coin_flip_difference :
  let P1 := 5 * (1 / 2)^5;
  let P2 := (1 / 2)^5;
  P1 - P2 = 9 / 32 :=
by
  sorry

end fair_coin_flip_difference_l136_136920


namespace approximate_sum_l136_136519

def sequence_fraction (k : ℕ) : ℝ := (k : ℝ) / (k + 5 : ℝ)

def sum_sequence (n : ℕ) : ℝ := ∑ k in Finset.range n, sequence_fraction (k + 1)

theorem approximate_sum : abs (sum_sequence 10 - 4.924) < 0.001 :=
by
  sorry

end approximate_sum_l136_136519


namespace max_moves_on_15x15_board_l136_136498

/-- A piece is placed in the lower left-corner cell of a 15 × 15 board.
    It can move to cells that are adjacent to the sides or the corners of its current cell.
    It must also alternate between horizontal and diagonal moves (the first move must be diagonal).
    Prove that the maximum number of moves it can make without stepping on the same cell twice is 196. -/
theorem max_moves_on_15x15_board : 
  ∃ f : ℕ → (ℤ × ℤ), 
    (∀ n, f n.1 = 0 ∧ f n.2 = 0) ∧ 
    (∀ n, (n > 0 → (f (n + 1)).fst = (f n).fst + 1 ∨ (f (n + 1)).snd = (f n).snd + 1)) ∧
    (∀ n, (n > 0 → (f (n + 1)).fst ≠ (f n).fst + 1 ∨ (f (n + 1)).snd ≠ (f n).snd + 1 → (f (n + 1)).fst ≠ (f n).fst - 1 ∨ (f (n + 1)).snd ≠ (f n).snd + 1)) ∧ 
    set.inj_on f (finset.range 197)  :=
begin
  sorry
end

end max_moves_on_15x15_board_l136_136498


namespace coin_flip_probability_difference_l136_136949

theorem coin_flip_probability_difference :
  let p1 := (nat.choose 5 4) * (1 / 2)^5
  let p2 := (1 / 2)^5
  (p1 - p2 = 1 / 8) :=
by
  let p1 := (nat.choose 5 4) * (1 / 2)^5
  let p2 := (1 / 2)^5
  sorry

end coin_flip_probability_difference_l136_136949


namespace complex_number_quadrant_l136_136237

-- Definitions and assumptions based strictly on the problem conditions
variable (z : ℂ)
variable (h : (1 + complex.I) * conj z = 3 + complex.I)

-- The theorem to be proved
theorem complex_number_quadrant (h : (1 + complex.I) * conj z = 3 + complex.I) : 
  0 < z.re ∧ 0 < z.im := sorry

end complex_number_quadrant_l136_136237


namespace find_x_l136_136501

noncomputable def log_base_2 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def log_base_3 (x : ℝ) : ℝ := Real.log x / Real.log 3
noncomputable def log_base_5 (x : ℝ) : ℝ := Real.log x / Real.log 5
noncomputable def log_base_4 (x : ℝ) : ℝ := Real.log x / Real.log 4

noncomputable def right_triangular_prism_area (x : ℝ) : ℝ := 
  let a := log_base_2 x
  let b := log_base_3 x
  let c := log_base_5 x
  (1/2) * a * b + 3 * c * (a + b + c)

noncomputable def right_triangular_prism_volume (x : ℝ) : ℝ := 
  let a := log_base_2 x
  let b := log_base_3 x
  let c := log_base_5 x
  (1/2) * a * b * c

noncomputable def rectangular_prism_area (x : ℝ) : ℝ := 
  let a := log_base_2 x
  let b := log_base_4 x
  2 * (a * b + a * a + b * a)

noncomputable def rectangular_prism_volume (x : ℝ) : ℝ := 
  let a := log_base_2 x
  let b := log_base_4 x
  a * b * a

theorem find_x (x : ℝ) (h : right_triangular_prism_area x + rectangular_prism_area x = right_triangular_prism_volume x + rectangular_prism_volume x) :
  x = 1152 := by
sorry

end find_x_l136_136501


namespace twenty_fifth_entry_of_n_sequence_l136_136204

def r_11 (n : ℕ) : ℕ := n % 11

theorem twenty_fifth_entry_of_n_sequence :
  ∃ (n : ℕ), (n % 11 = 40 % 11) ∧ 
  (r_11 (7 * (fib 25 - 1)) <= 5) :=
begin
  sorry
end

end twenty_fifth_entry_of_n_sequence_l136_136204


namespace find_positive_x_l136_136564

variable {c d : ℂ} (x : ℝ)

theorem find_positive_x
    (h1 : |c| = 3)
    (h2 : |d| = 4)
    (h3 : c * d = x - 3 * Complex.I) :
    x = 3 * Real.sqrt 15 := by
  sorry

end find_positive_x_l136_136564


namespace buy_items_ways_l136_136725

theorem buy_items_ways (headphones keyboards mice keyboard_mouse_sets headphone_mouse_sets : ℕ) :
  headphones = 9 → keyboards = 5 → mice = 13 → keyboard_mouse_sets = 4 → headphone_mouse_sets = 5 →
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 :=
by
  intros h_eq k_eq m_eq kms_eq hms_eq
  have h_eq_gen : headphones = 9 := h_eq
  have k_eq_gen : keyboards = 5 := k_eq
  have m_eq_gen : mice = 13 := m_eq
  have kms_eq_gen : keyboard_mouse_sets = 4 := kms_eq
  have hms_eq_gen : headphone_mouse_sets = 5 := hms_eq
  sorry

end buy_items_ways_l136_136725


namespace shop_combinations_l136_136750

theorem shop_combinations :
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let keyboard_and_mouse_sets := 4
  let headphone_and_mouse_sets := 5
  (keyboard_and_mouse_sets * headphones) + (headphone_and_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 :=
by
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let keyboard_and_mouse_sets := 4
  let headphone_and_mouse_sets := 5
  have case1 : keyboard_and_mouse_sets * headphones = 4 * 9, by sorry
  have case2 : headphone_and_mouse_sets * keyboards = 5 * 5, by sorry
  have case3 : headphones * mice * keyboards = 9 * 13 * 5, by sorry
  show (keyboard_and_mouse_sets * headphones) + (headphone_and_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646,
  by rw [case1, case2, case3]; exact sorry

end shop_combinations_l136_136750


namespace history_homework_time_l136_136008

def total_time := 180
def math_homework := 45
def english_homework := 30
def science_homework := 50
def special_project := 30

theorem history_homework_time : total_time - (math_homework + english_homework + science_homework + special_project) = 25 := by
  sorry

end history_homework_time_l136_136008


namespace intersection_in_second_quadrant_l136_136286

-- Define the first line
def line1 (x : ℝ) : ℝ := 2 * x + 4

-- Define the second line with parameter m
def line2 (x m : ℝ) : ℝ := -2 * x + m

-- The intersection point x-coordinate in terms of m
def intersection_x (m : ℝ) : ℝ := (m - 4) / 4

-- The intersection point y-coordinate in terms of m
def intersection_y (m : ℝ) : ℝ := (m + 4) / 2

-- Lean theorem statement
theorem intersection_in_second_quadrant (m : ℝ) :
  (intersection_x m < 0 ∧ intersection_y m > 0) ↔ -4 < m ∧ m < 4 :=
sorry

end intersection_in_second_quadrant_l136_136286


namespace divide_students_into_classes_l136_136487

theorem divide_students_into_classes
  (N : ℕ)
  (scores : Fin (3 * N) → ℝ)
  (h_scores_range : ∀ i, 60 ≤ scores i ∧ scores i ≤ 100)
  (h_duplicate_scores : ∀ x ∈ set.range scores, (∃ i j, i ≠ j ∧ scores i = x ∧ scores j = x))
  (average_score : 1 / (3 : ℝ) = ((finset.univ.sum (λ i, scores i)) / (3 * N)) / 82.4) :
  ∃ (class1 class2 class3 : Finset (Fin (3 * N))),
  class1.card = N ∧ class2.card = N ∧ class3.card = N ∧
  class1.disjoint class2 ∧ class2.disjoint class3 ∧ class1.disjoint class3 ∧
  (82.4 = (class1.sum scores) / N) ∧
  (82.4 = (class2.sum scores) / N) ∧
  (82.4 = (class3.sum scores) / N) := 
sorry

end divide_students_into_classes_l136_136487


namespace min_correct_answers_l136_136295

theorem min_correct_answers (x : ℕ) : 
  (∃ x, 0 ≤ x ∧ x ≤ 20 ∧ 5 * x - (20 - x) ≥ 88) :=
sorry

end min_correct_answers_l136_136295


namespace coprime_nth_le_sigma_equality_holds_iff_prime_power_l136_136232

-- Define the Euler's totient function φ and the sum of divisors function σ
noncomputable def euler_totient (n : ℕ) : ℕ := 
  (finset.range n).filter (nat.coprime n).card

noncomputable def sigma (n : ℕ) : ℕ :=
  finset.sum (nat.divisors n) id

-- We need to state the main theorems

-- Theorem 1: Prove that the n-th smallest positive integer that is coprime with n is at least σ(n)
theorem coprime_nth_le_sigma (n : ℕ) (hn : 2 ≤ n) (k : ℕ) :
  k ≤ sigma n → k ≤ nth_coprime n k :=
sorry

-- Theorem 2: Determine for which positive integers n equality holds
theorem equality_holds_iff_prime_power (n : ℕ) (hn : 2 ≤ n) :
  (sigma n = n) ↔ (∃ p e : ℕ, nat.prime p ∧ n = p ^ e) :=
sorry

end coprime_nth_le_sigma_equality_holds_iff_prime_power_l136_136232


namespace parabola_vertex_y_value_l136_136381

theorem parabola_vertex_y_value (m : ℝ) (t : ℝ) (h_eq : t = 33) 
    (h_parabola : ∀ x, y = 3 * x^2 + 6 * real.sqrt m * x + 36 ∧ 
     y_vertex : t = 36 - 3 * m) : m = 1 :=
begin
  sorry,
end

end parabola_vertex_y_value_l136_136381


namespace slope_of_tangent_at_point_l136_136882

theorem slope_of_tangent_at_point (x : ℝ) (y : ℝ) (h_curve : y = x^3)
    (h_slope : 3*x^2 = 3) : (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) := 
sorry

end slope_of_tangent_at_point_l136_136882


namespace range_of_a_l136_136260

def proposition_P (a : ℝ) := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

def proposition_Q (a : ℝ) := 5 - 2*a > 1

theorem range_of_a :
  (∃! (p : Prop), (p = proposition_P a ∨ p = proposition_Q a) ∧ p) →
  a ∈ Set.Iic (-2) :=
by
  sorry

end range_of_a_l136_136260


namespace cylinder_radius_inscribed_in_cone_l136_136132

theorem cylinder_radius_inscribed_in_cone (d_c : ℝ) (h_c : ℝ) (h_cyl_ratio : ℝ) : 
  d_c = 12 ∧ h_c = 15 ∧ h_cyl_ratio = 3 → 
  ∃ r : ℝ, r = 30 / 11 :=
begin
  intros h,
  cases h with d_c_eq h1,
  cases h1 with h_c_eq h_cyl_ratio_eq,
  use 30 / 11,
  split,
  { exact d_c_eq },
  split,
  { exact h_c_eq },
  { exact h_cyl_ratio_eq },
  sorry  -- Here is where the proof would go.
end

end cylinder_radius_inscribed_in_cone_l136_136132


namespace min_value_expr_l136_136821

theorem min_value_expr (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ k : ℝ, k = 6 ∧ (∃ a b c : ℝ,
                  0 < a ∧
                  0 < b ∧
                  0 < c ∧
                  (k = (a^2 + b^2) / c + (a^2 + c^2) / b + (b^2 + c^2) / a)) :=
sorry

end min_value_expr_l136_136821


namespace promoting_cashback_beneficial_for_bank_cashback_in_rubles_preferable_l136_136973

-- Definitions of conditions
def attracts_new_clients (bank_promotes_cashback : Prop) : Prop :=
  bank_promotes_cashback → ∃ (new_clients : Prop), new_clients

def promotes_partnerships (bank_promotes_cashback : Prop) : Prop :=
  bank_promotes_cashback → ∃ (partnerships : Prop), partnerships

def enhances_competitiveness (bank_promotes_cashback : Prop) : Prop :=
  bank_promotes_cashback → ∃ (competitiveness : Prop), competitiveness

def liquidity_advantage (cashback_rubles : Prop) : Prop :=
  cashback_rubles → ∃ (liquidity : Prop), liquidity

def no_expiry_concerns (cashback_rubles : Prop) : Prop :=
  cashback_rubles → ∃ (no_expiry : Prop), no_expiry

def no_partner_limitations (cashback_rubles : Prop) : Prop :=
  cashback_rubles → ∃ (partner_limitations : Prop), ¬partner_limitations

-- Lean statements for the proof problems
theorem promoting_cashback_beneficial_for_bank (bank_promotes_cashback : Prop) :
  attracts_new_clients bank_promotes_cashback ∧
  promotes_partnerships bank_promotes_cashback ∧ 
  enhances_competitiveness bank_promotes_cashback →
  bank_promotes_cashback := 
sorry

theorem cashback_in_rubles_preferable (cashback_rubles : Prop) :
  liquidity_advantage cashback_rubles ∧
  no_expiry_concerns cashback_rubles ∧
  no_partner_limitations cashback_rubles →
  cashback_rubles :=
sorry

end promoting_cashback_beneficial_for_bank_cashback_in_rubles_preferable_l136_136973


namespace fair_coin_flip_difference_l136_136925

theorem fair_coin_flip_difference :
  let P1 := 5 * (1 / 2)^5;
  let P2 := (1 / 2)^5;
  P1 - P2 = 9 / 32 :=
by
  sorry

end fair_coin_flip_difference_l136_136925


namespace solve_eqn_l136_136021

theorem solve_eqn (x : ℂ) : (x - 4)^6 + (x - 6)^6 = 32 → 
  (x = 5 + complex.I * real.sqrt(3) ∨ x = 5 - complex.I * real.sqrt(3)) :=
by
  -- proof is not needed
  sorry

end solve_eqn_l136_136021


namespace minimum_a_for_monotonic_increasing_on_interval_l136_136648

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp x - Real.log x

noncomputable def f_prime (a x : ℝ) : ℝ := a * Real.exp x - 1 / x

def min_value_of_a : ℝ := Real.exp (-1)

theorem minimum_a_for_monotonic_increasing_on_interval :
  (∀ x ∈ Set.Ioo 1 2, 0 ≤ f_prime a x) ↔ a ≥ min_value_of_a :=
by
  sorry

end minimum_a_for_monotonic_increasing_on_interval_l136_136648


namespace angle_equality_in_triangle_l136_136788

theorem angle_equality_in_triangle
  (K O I M S U : Type)
  [triangle : Triangle K O I]
  (KM_MI : dist K M = dist M I)
  (SI_SO : dist S I = dist S O)
  (MU_parallel_KI : parallel (line M U) (line K I)) :
  ∠ K O I = ∠ M I U :=
sorry

end angle_equality_in_triangle_l136_136788


namespace shirts_per_kid_l136_136792

-- Define given conditions
def n_buttons : Nat := 63
def buttons_per_shirt : Nat := 7
def n_kids : Nat := 3

-- The proof goal
theorem shirts_per_kid : (n_buttons / buttons_per_shirt) / n_kids = 3 := by
  sorry

end shirts_per_kid_l136_136792


namespace determinant_inequality_l136_136368

theorem determinant_inequality (x : ℝ) (h : 2 * x - (3 - x) > 0) : 3 * x - 3 > 0 := 
by
  sorry

end determinant_inequality_l136_136368


namespace number_of_students_in_class_l136_136874

theorem number_of_students_in_class :
  ∃ a : ℤ, 100 ≤ a ∧ a ≤ 200 ∧ a % 4 = 1 ∧ a % 3 = 2 ∧ a % 7 = 3 ∧ a = 101 := 
sorry

end number_of_students_in_class_l136_136874


namespace fraction_subtraction_property_l136_136980

variable (a b c d : ℚ)

theorem fraction_subtraction_property :
  (a / b - c / d) = ((a - c) / (b + d)) → (a / c) = (b / d) ^ 2 := 
by
  sorry

end fraction_subtraction_property_l136_136980


namespace total_games_in_season_l136_136423

-- Define the constants according to the conditions
def number_of_teams : ℕ := 25
def games_per_pair : ℕ := 15

-- Define the mathematical statement we want to prove
theorem total_games_in_season :
  let round_robin_games := (number_of_teams * (number_of_teams - 1)) / 2 in
  let total_games := round_robin_games * games_per_pair in
  total_games = 4500 :=
by
  sorry

end total_games_in_season_l136_136423


namespace final_exam_mean_score_l136_136203

theorem final_exam_mean_score (μ σ : ℝ) 
  (h1 : 55 = μ - 1.5 * σ)
  (h2 : 75 = μ - 2 * σ)
  (h3 : 85 = μ + 1.5 * σ)
  (h4 : 100 = μ + 3.5 * σ) :
  μ = 115 :=
by
  sorry

end final_exam_mean_score_l136_136203


namespace fff_two_eq_two_l136_136619

-- Define the piecewise function f
def f (x : ℝ) : ℝ :=
  if x < 2 then 2 * Real.exp (x - 1) else Real.logBase 3 (x^2 - 1)

-- State the theorem to prove that f(f(2)) = 2
theorem fff_two_eq_two : f (f 2) = 2 := by
  sorry

end fff_two_eq_two_l136_136619


namespace measure_of_angle_A_l136_136291

theorem measure_of_angle_A (a b c : ℝ) (A B C : ℝ) (hABC : A + B + C = π) (hSides : 0 < a ∧ 0 < b ∧ 0 < c)
  (hOppSides : a / sin A = b / sin B ∧ b / sin B = c / sin C)
  (hEquation : 1 + tan A / tan B = 2 * c / b) : A = π / 2 :=
sorry

end measure_of_angle_A_l136_136291


namespace distinct_pairs_count_l136_136696

theorem distinct_pairs_count : 
  { (m, n) : ℕ × ℕ // 0 < m ∧ 0 < n ∧ (1 / (m:ℚ) + 1 / (n:ℚ) = 1 / 5) }.to_finset.card = 3 := 
by {
  sorry,
}

end distinct_pairs_count_l136_136696


namespace intersection_eq_l136_136604

variable {α : Type*}

def M : set (ℤ × ℤ) := {p | p.1 + p.2 = 2}
def N : set (ℤ × ℤ) := {p | p.1 - p.2 = 4}

theorem intersection_eq :
  M ∩ N = {(3, -1)} :=
by 
  sorry

end intersection_eq_l136_136604


namespace probability_same_group_l136_136433

def total_students := 800
def lunch_groups := 4
def group_size := total_students / lunch_groups
def friends := ["Dan", "Eve", "Frank", "Grace"]

theorem probability_same_group :
  ∀ (students : ℕ) (groups : ℕ) (D E F G : string),
  students = 800 → groups = 4 → (students / groups = group_size) →
  D ∈ friends → E ∈ friends → F ∈ friends → G ∈ friends → 
  (D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ E ≠ F ∧ E ≠ G ∧ F ≠ G) →
  let prob := (1 / groups) * (1 / groups) * (1 / groups) in
  prob = (1 / 64) :=
by {
  intros students groups D E F G h1 h2 h3 h4 h5 h6,
  sorry
}

end probability_same_group_l136_136433


namespace exists_perm_diff_div_factorial_l136_136344

open Nat

theorem exists_perm_diff_div_factorial (n : Nat) (h_odd : Odd n) (h_gt : n > 1)
  (c : Fin n → Int) :
  ∃ (a b : List (Fin n.+1)) (ha : a ≠ b), factorial n ∣ (List.sum (List.map (λ i, c i * ↑(a[i])) (List.finRange n)) -
                                                      List.sum (List.map (λ i, c i * ↑(b[i])) (List.finRange n))) := 
sorry

end exists_perm_diff_div_factorial_l136_136344


namespace star_rearrangement_possible_l136_136885

-- Define the setting in Lean
def star := ℕ
def Line (s : set star) := { l : set star // l.card = 5 }

-- Define the problem condition: 21 stars forming 11 lines with 5 stars each
def problem_condition (stars : set star) (lines : set (Line stars)) : Prop :=
  stars.card = 21 ∧ ∀ l ∈ lines, l.val.card = 5 ∧ lines.card = 11

-- The statement we need to prove
theorem star_rearrangement_possible : 
  ∃ stars lines, problem_condition stars lines :=
by
  sorry

end star_rearrangement_possible_l136_136885


namespace min_value_of_a_l136_136660

noncomputable def min_a_monotone_increasing : ℝ :=
  let f : ℝ → ℝ := λ x a, a * exp x - log x in 
  let a_min : ℝ := exp (-1) in
  a_min

theorem min_value_of_a 
  {f : ℝ → ℝ} 
  {a : ℝ}
  (mono_incr : ∀ x ∈ Ioo 1 2, 0 ≤ deriv (λ x, a * exp x - log x) x) :
  a ≥ exp (-1) :=
begin
  sorry
end

end min_value_of_a_l136_136660


namespace area_of_triangle_6_8_10_f_of_4_eq_52_sum_of_squares_series_area_of_square_PQRS_l136_136308

-- G2.1
theorem area_of_triangle_6_8_10 : 
  ∃ (a b c : ℕ), a = 6 ∧ b = 8 ∧ c = 10 ∧ ∃ (area : ℕ), area = 24 :=
by sorry

-- G2.2
theorem f_of_4_eq_52 (f : ℚ → ℚ) (x : ℚ) 
  (h₀ : ∀ x, f (x + 1/x) = x^3 + 1/x^3) : 
  f 4 = 52 :=
by sorry

-- G2.3
theorem sum_of_squares_series : 
  ∑ k in range 1001, (2 * (2002 - k)) * 1 = 2005003 :=
by sorry

-- G2.4
theorem area_of_square_PQRS (P Q R S T U : ℝ) 
  (h₀ : square PQRS) 
  (h₁ : isosceles T U) 
  (h₂ : angle T P U = π / 6) 
  (h₃ : area_triangle P T U = 1) : 
  area_square PQRS = 3 :=
by sorry

end area_of_triangle_6_8_10_f_of_4_eq_52_sum_of_squares_series_area_of_square_PQRS_l136_136308


namespace increasing_interval_l136_136872

-- Given function definition
def quad_func (x : ℝ) : ℝ := -x^2 + 1

-- Property to be proven: The function is increasing on the interval (-∞, 0]
theorem increasing_interval : ∀ x y : ℝ, x ≤ 0 → y ≤ 0 → x < y → quad_func x < quad_func y := by
  sorry

end increasing_interval_l136_136872


namespace annual_growth_rate_l136_136482

theorem annual_growth_rate (u_2021 u_2023 : ℝ) (x : ℝ) : 
    u_2021 = 1 ∧ u_2023 = 1.69 ∧ x > 0 → (u_2023 / u_2021) = (1 + x)^2 → x * 100 = 30 :=
by
  intros h1 h2
  sorry

end annual_growth_rate_l136_136482


namespace max_additive_triplets_l136_136981

theorem max_additive_triplets (S : Finset ℕ) (h : S.card = 20) :
  ∃ T : Finset (Finset ℕ), (∀ t ∈ T, ∃ a b c ∈ S, a + b = c ∧ t = {a, b, c} ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ T.card = 90 := 
sorry

end max_additive_triplets_l136_136981


namespace find_number_l136_136086

theorem find_number (x : ℤ) (h : 4 * x = 28) : x = 7 :=
sorry

end find_number_l136_136086


namespace lucy_total_fish_l136_136829

theorem lucy_total_fish (current_fish : ℕ) (additional_fish : ℕ) (desired_fish : ℕ) 
  (h1 : current_fish = 212) (h2 : additional_fish = 68) : desired_fish = current_fish + additional_fish :=
by
  simp [h1, h2]
  exact rfl

end lucy_total_fish_l136_136829


namespace distinct_natural_numbers_l136_136306

theorem distinct_natural_numbers (n : ℕ) (h : n = 100) : 
  ∃ (nums : Fin n → ℕ), 
    (∀ i j, i ≠ j → nums i ≠ nums j) ∧
    (∀ (a b c d e : Fin n), 
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
     c ≠ d ∧ c ≠ e ∧ 
     d ≠ e →
      (nums a) * (nums b) * (nums c) * (nums d) * (nums e) % ((nums a) + (nums b) + (nums c) + (nums d) + (nums e)) = 0) :=
by
  sorry

end distinct_natural_numbers_l136_136306


namespace min_value_of_a_l136_136658

noncomputable def min_a_monotone_increasing : ℝ :=
  let f : ℝ → ℝ := λ x a, a * exp x - log x in 
  let a_min : ℝ := exp (-1) in
  a_min

theorem min_value_of_a 
  {f : ℝ → ℝ} 
  {a : ℝ}
  (mono_incr : ∀ x ∈ Ioo 1 2, 0 ≤ deriv (λ x, a * exp x - log x) x) :
  a ≥ exp (-1) :=
begin
  sorry
end

end min_value_of_a_l136_136658


namespace rabbits_count_l136_136292

-- Definition of the context and assumptions
def cages_problem (x : ℕ) := 
  let chickens := x + 53 in
  let total_legs := 4 * x + 2 * chickens in
  total_legs = 250

-- Theorem statement
theorem rabbits_count : ∃ x : ℕ, cages_problem x ∧ x = 24 :=
by {
  -- Include assumption and statement here
  sorry
}

end rabbits_count_l136_136292


namespace minimum_value_of_a_l136_136630

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - Real.log x

theorem minimum_value_of_a (a : ℝ) :
  (∀ x ∈ Ioo 1 2, deriv (f a) x ≥ 0) → a ≥ Real.exp ⁻¹ :=
by
  simp [f]
  sorry

end minimum_value_of_a_l136_136630


namespace purchase_combinations_correct_l136_136757

theorem purchase_combinations_correct : 
  let headphones : ℕ := 9
  let mice : ℕ := 13
  let keyboards : ℕ := 5
  let sets_keyboard_mouse : ℕ := 4
  let sets_headphones_mouse : ℕ := 5
in 
  (sets_keyboard_mouse * headphones) + 
  (sets_headphones_mouse * keyboards) + 
  (headphones * mice * keyboards) = 646 := 
by
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let sets_keyboard_mouse := 4
  let sets_headphones_mouse := 5
  sorry

end purchase_combinations_correct_l136_136757


namespace fair_coin_flip_probability_difference_l136_136941

noncomputable def choose : ℕ → ℕ → ℕ
| _, 0 => 1
| 0, _ => 0
| n + 1, r + 1 => choose n r + choose n (r + 1)

theorem fair_coin_flip_probability_difference :
  let p := 1 / 2
  let p_heads_4_out_of_5 := choose 5 4 * p^4 * p
  let p_heads_5_out_of_5 := p^5
  p_heads_4_out_of_5 - p_heads_5_out_of_5 = 1 / 8 :=
by
  sorry

end fair_coin_flip_probability_difference_l136_136941


namespace first_even_number_of_8_sum_424_l136_136143

theorem first_even_number_of_8_sum_424 (x : ℕ) (h : x + (x + 2) + (x + 4) + (x + 6) + 
                   (x + 8) + (x + 10) + (x + 12) + (x + 14) = 424) : x = 46 :=
by sorry

end first_even_number_of_8_sum_424_l136_136143


namespace purchase_combinations_correct_l136_136753

theorem purchase_combinations_correct : 
  let headphones : ℕ := 9
  let mice : ℕ := 13
  let keyboards : ℕ := 5
  let sets_keyboard_mouse : ℕ := 4
  let sets_headphones_mouse : ℕ := 5
in 
  (sets_keyboard_mouse * headphones) + 
  (sets_headphones_mouse * keyboards) + 
  (headphones * mice * keyboards) = 646 := 
by
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let sets_keyboard_mouse := 4
  let sets_headphones_mouse := 5
  sorry

end purchase_combinations_correct_l136_136753


namespace sum_of_interior_angles_of_special_regular_polygon_l136_136409

theorem sum_of_interior_angles_of_special_regular_polygon (n : ℕ) (h1 : n = 4 ∨ n = 5) :
  ((n - 2) * 180 = 360 ∨ (n - 2) * 180 = 540) :=
by sorry

end sum_of_interior_angles_of_special_regular_polygon_l136_136409


namespace max_volumes_on_fedor_shelf_l136_136364

theorem max_volumes_on_fedor_shelf 
  (S s1 s2 n : ℕ) 
  (h1 : S + s1 ≥ (n - 2) / 2) 
  (h2 : S + s2 < (n - 2) / 3) 
  : n = 12 := 
sorry

end max_volumes_on_fedor_shelf_l136_136364


namespace good_family_corresponds_to_graph_l136_136073

def is_mangool (A1 A2 A3 : set ℕ) : Prop :=
  ∃ π : fin 3 → fin 3, bijective π ∧ ¬ (A2 ⊆ A1) ∧ ¬ (A3 ⊆ A1 ∪ A2)

def good_family (X : set ℕ) (As : finset (set ℕ)) : Prop :=
  ∀ A1 A2 A3 ∈ As, is_mangool X A1 A2

theorem good_family_corresponds_to_graph (G : Type) [simple_graph G] :
  ∃ (X : set ℕ) (As : finset (set ℕ)), 
    good_family X As ∧ 
    (∀ (A1 A2 : set ℕ), (X ∉ A1 ∪ A2) ↔ ¬ G.adj A1 A2) :=
sorry

end good_family_corresponds_to_graph_l136_136073


namespace tournament_problem_l136_136470

theorem tournament_problem :
  ∃ n, (∀ arrangement : matrix (fin 18) (fin 18) bool,
    (∀ i j : fin 18, i != j -> ((arrangement i j) = (arrangement j i))),
    (∀ i : fin 18, ∑ j, (arrangement i j) = 17),
    (∀ i j : fin 18, i != j -> (arrangement i j ∈ {true, false})),
    (∀ i j : fin 18, i != j -> (arrangement i j = true ↔ arrangement j i = true)),
    (∀ i j k : fin 18, i != j -> j != k -> k != i -> (arrangement i j = arrangement j k = arrangement k i)),
    (∀ rounds : matrix (fin 17) (fin 9) (fin 18), 
      (∀ r : fin 17, (∀ g : fin 9, arrangement (rounds r g 0) (rounds r g 1) = true))
      → ∃ (A B C D : fin 18), 
         (arrangement A B = arrangement B C = arrangement C D = arrangement D A = true)
          ∨ (arrangement B A = arrangement C B = arrangement D C = arrangement A D = true))
  ) ∧ n = 7 := 
sorry

end tournament_problem_l136_136470


namespace proj_b_v_l136_136334

-- Define orthogonality
def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

-- Define vector projections
def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let c := (u.1 * v.1 + u.2 * v.2) / (v.1 * v.1 + v.2 * v.2)
  (c * v.1, c * v.2)

-- Given vectors a and b and their properties
variables (a b : ℝ × ℝ) (v : ℝ × ℝ)
variable h_ab_orthogonal : orthogonal a b
variable h_proj_a : proj v a = (-4/5, -8/5)

-- The goal to prove
theorem proj_b_v :
  proj v b = (24/5, -2/5) := sorry

end proj_b_v_l136_136334


namespace volume_of_cut_cone_l136_136122

theorem volume_of_cut_cone (V_frustum : ℝ) (A_bottom : ℝ) (A_top : ℝ) (V_cut_cone : ℝ) :
  V_frustum = 52 ∧ A_bottom = 9 * A_top → V_cut_cone = 54 :=
by
  sorry

end volume_of_cut_cone_l136_136122


namespace rect_park_ratio_l136_136107

theorem rect_park_ratio (x y : ℝ) (h1 : x < y) (h2 : x + y - real.sqrt (x^2 + y^2) = (1 / 3) * y) : 
  x / y = 5 / 12 := 
sorry

end rect_park_ratio_l136_136107


namespace range_g_l136_136709

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + x + 1
noncomputable def g (a x : ℝ) : ℝ := x^2 + a * x + 1

theorem range_g (a : ℝ) (h : Set.range (λ x => f a x) = Set.univ) : Set.range (λ x => g a x) = { y : ℝ | 1 ≤ y } := by
  sorry

end range_g_l136_136709


namespace surface_area_DABC_l136_136810

structure Triangle (α : Type*) :=
(A B C : α)

structure Pyramid (α : Type*) :=
(A B C D : α)

def edge_length (α : Type*) [has_dist : Π x y : α, ℝ] (x y : α) : ℝ := 
  dist x y

variables {α : Type*} [metric_space α]
variables (ABC : Triangle α) (D : α)
variables (h₁ : ∀ x ∈ {ABC.A, ABC.B, ABC.C}, x ≠ D)
variables (h₂ : ∀ s ∈ ({ABC.A, ABC.B, ABC.C} : set α), ∀ t ∈ ({ABC.A, ABC.B, ABC.C} : set α), edge_length α s t = 20 ∨ edge_length α s t = 45)
variables (h₃ : ¬ ∃ (A B C ∈ {ABC.A, ABC.B, ABC.C}),  edge_length α A B = edge_length α B C ∧ edge_length α B C = edge_length α C A)

theorem surface_area_DABC (h₄ : ∀ (A B ∈ {ABC.A, ABC.B, ABC.C}), (edge_length α A B = 20 ∧ edge_length α A D = 45 ∧ edge_length α B D = 45)
  ∨ (edge_length α A B = 45 ∧ edge_length α A D = 45 ∧ edge_length α B D = 20)) :
  ∃ area, area = 40 * real.sqrt 1925 :=
begin
  sorry
end

end surface_area_DABC_l136_136810


namespace hockey_league_games_l136_136426

theorem hockey_league_games (n : ℕ) (k : ℕ) (h_n : n = 25) (h_k : k = 15) : 
  (n * (n - 1) / 2) * k = 4500 := by
  sorry

end hockey_league_games_l136_136426


namespace initial_money_l136_136496

variable (X : ℝ)

theorem initial_money (h1 : 2 / 3 * X = after_clothes)
                    (h2 : 4 / 5 * after_clothes = after_food)
                    (h3 : 17 / 20 * after_food = after_utilities)
                    (h4 : 7 / 8 * after_utilities = after_entertainment)
                    (h5 : 4 / 5 * after_entertainment = after_travel)
                    (h6 : after_travel = 3500) :
    X = 11029.41 :=
by
    have hX : X = 3500 * (375 / 119), from sorry
    rw [hX]

end initial_money_l136_136496


namespace find_α_l136_136612

noncomputable def α_is_acute (α : ℝ) : Prop := 0 < α ∧ α < π / 2

def vector_a (α : ℝ) : ℝ × ℝ := (3 / 4, Real.sin α)
def vector_b (α : ℝ) : ℝ × ℝ := (Real.cos α, 1 / 3)

def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem find_α (α : ℝ) (h1 : α_is_acute α) (h2 : vectors_parallel (vector_a α) (vector_b α)) :
  α = π / 12 ∨ α = 5 * π / 12 := sorry

end find_α_l136_136612


namespace painting_squares_conditions_l136_136472

theorem painting_squares_conditions :
  let grid := fin 2 × fin 2;
  let colors := {green, red};
  let valid_painting (g : grid → colors) := 
    ∀ (i j : fin 2), 
      (g (i, j) = green → 
        (∀ k : fin 2, (k > i → g (k, j) ≠ red)) ∧ 
        (∀ k : fin 2, (k > j → g (i, k) ≠ red)));
  {g : grid → colors // valid_painting g}.card = 6 :=
by sorry

end painting_squares_conditions_l136_136472


namespace infinite_prod_value_l136_136520

theorem infinite_prod_value :
  (∏ (n : ℕ) in finset.range (2^1000), (3 ^ ( (n + 1) / 2^nat.succ n))) = 9 :=
by sorry

end infinite_prod_value_l136_136520


namespace buy_items_ways_l136_136728

theorem buy_items_ways (headphones keyboards mice keyboard_mouse_sets headphone_mouse_sets : ℕ) :
  headphones = 9 → keyboards = 5 → mice = 13 → keyboard_mouse_sets = 4 → headphone_mouse_sets = 5 →
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 :=
by
  intros h_eq k_eq m_eq kms_eq hms_eq
  have h_eq_gen : headphones = 9 := h_eq
  have k_eq_gen : keyboards = 5 := k_eq
  have m_eq_gen : mice = 13 := m_eq
  have kms_eq_gen : keyboard_mouse_sets = 4 := kms_eq
  have hms_eq_gen : headphone_mouse_sets = 5 := hms_eq
  sorry

end buy_items_ways_l136_136728


namespace sphere_tangency_relation_l136_136069

theorem sphere_tangency_relation (r R : ℝ) 
  (h1 : ∀ A B : point, distance A B = 2*r) 
  (h2 : ∀ O₁ O₂ O₃ : point, is_equilateral O₁ O₂ O₃ ∧ ∀ i ≠ j, distance (O i) (O j) = 2*R)
  (h3 : ∀ A B Oᵢ : point, is_tangent A Oᵢ ∧ is_tangent B Oᵢ ∧ distance A Oᵢ = r + R ∧ distance B Oᵢ = r + R) :
  R = 6*r := 
sorry

end sphere_tangency_relation_l136_136069


namespace max_min_distance_MQ_max_min_slope_k_l136_136224

-- Definitions of the given conditions
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 14 * y + 45 = 0

def point_Q : ℝ × ℝ := (-2, 3)

-- Mathematical proof statements
theorem max_min_distance_MQ :
  ∀ (M : ℝ × ℝ), circle_C (M.1) (M.2) → (|M.1 + 2| + |M.2 - 3|) = 6 * real.sqrt 2 ∨ (|M.1 + 2| + |M.2 - 3|) = 2 * real.sqrt 2 :=
sorry

theorem max_min_slope_k :
  ∀ (m n : ℝ), circle_C m n → (real.sqrt 3 + 2) >= (n - 3)/(m + 2) ∧ (n - 3)/(m + 2) >= (2 - real.sqrt 3) :=
sorry

end max_min_distance_MQ_max_min_slope_k_l136_136224


namespace find_value_l136_136703

variable (x y : ℚ)
hypothesis (hx : x = 3/4)
hypothesis (hy : y = 4/3)

theorem find_value : (1 / 2) * x^6 * y^7 = 2 / 3 := by
  -- The proof would go here
  sorry

end find_value_l136_136703


namespace valid_colorings_l136_136537

-- Define the coloring function and the condition
variable (f : ℕ → ℕ) -- f assigns a color (0, 1, or 2) to each natural number
variable (a b c : ℕ)
-- Colors are represented by 0, 1, or 2
variable (colors : Fin 3)

-- Define the condition to be checked
def valid_coloring : Prop :=
  ∀ a b c, 2000 * (a + b) = c → (f a = f b ∧ f b = f c) ∨ (f a ≠ f b ∧ f b ≠ f c ∧ f c ≠ f a)

-- Now define the two possible valid ways of coloring
def all_same_color : Prop :=
  ∃ color, ∀ n, f n = color

def every_third_different : Prop :=
  (∀ k : ℕ, f (3 * k) = 0 ∧ f (3 * k + 1) = 1 ∧ f (3 * k + 2) = 2)

-- Prove that these are the only two valid ways
theorem valid_colorings :
  valid_coloring f →
  all_same_color f ∨ every_third_different f :=
sorry

end valid_colorings_l136_136537


namespace find_a_b_l136_136350

theorem find_a_b (f : ℝ → ℝ) (a b : ℝ)
  (h1 : f = λ x, (a * x^3) / 3 - b * x^2 + a^2 * x - 1 / 3)
  (h2 : f 1 = 0)
  (h3 : deriv f 1 = 0):
  a + b = -7 / 9 :=
by
  sorry

end find_a_b_l136_136350


namespace rectangle_ratio_l136_136894

/-- Conditions:
1. There are three identical squares and two rectangles forming a large square.
2. Each rectangle shares one side with a square and another side with the edge of the large square.
3. The side length of each square is 1 unit.
4. The total side length of the large square is 5 units.
Question:
What is the ratio of the length to the width of one of the rectangles? --/

theorem rectangle_ratio (sq_len : ℝ) (large_sq_len : ℝ) (side_ratio : ℝ) :
  sq_len = 1 ∧ large_sq_len = 5 ∧ 
  (∀ (rect_len rect_wid : ℝ), 3 * sq_len + 2 * rect_len = large_sq_len ∧ side_ratio = rect_len / rect_wid) →
  side_ratio = 1 / 2 :=
by
  sorry

end rectangle_ratio_l136_136894


namespace num_ways_to_buy_three_items_l136_136768

-- Defining the conditions based on the problem statement
def num_headphones : ℕ := 9
def num_mice : ℕ := 13
def num_keyboards : ℕ := 5
def num_kb_mouse_sets : ℕ := 4
def num_hp_mouse_sets : ℕ := 5

-- Defining the theorem statement
theorem num_ways_to_buy_three_items :
  (num_kb_mouse_sets * num_headphones) + 
  (num_hp_mouse_sets * num_keyboards) + 
  (num_headphones * num_mice * num_keyboards) = 646 := 
  by
  sorry

end num_ways_to_buy_three_items_l136_136768


namespace ticket_identification_operations_l136_136434

theorem ticket_identification_operations (students : ℕ) (tickets : Finset ℕ) (h_students_le_tickets : students ≤ 30)
  (h_ticket_range : ∀ t ∈ tickets, 1 ≤ t ∧ t ≤ 30) (h_distinct_tickets : ∀ t1 t2 ∈ tickets, t1 ≠ t2 → t1 ≠ t2) :
  ∃ (n : ℕ), n = 5 := by
  sorry

end ticket_identification_operations_l136_136434


namespace cost_expressions_store_comparison_l136_136995

theorem cost_expressions (x : ℝ) (hx : x > 40) :
  let storeA_cost := 100 * 40 + 25 * (x - 40)
  let storeB_cost := 0.9 * 100 * 40 + 0.9 * 25 * x
  storeA_cost = 25 * x + 3000 ∧ storeB_cost = 22.5 * x + 3600 :=
by {
  have storeA_cost := 100 * 40 + 25 * (x - 40),
  have storeB_cost := 0.9 * 100 * 40 + 0.9 * 25 * x,
  split,
  calc
    storeA_cost = 100 * 40 + 25 * (x - 40) : rfl
    ... = 4000 + 25 * x - 1000 : by ring
    ... = 25 * x + 3000 : by ring,
  calc 
    storeB_cost = 0.9 * 100 * 40 + 0.9 * 25 * x : rfl
    ... = 3600 + 22.5 * x : by norm_num,
}

theorem store_comparison (x : ℝ) (hx : x = 100) :
  let storeA_cost := 25 * x + 3000
  let storeB_cost := 22.5 * x + 3600
  storeA_cost < storeB_cost :=
by {
  have storeA_cost := 25 * x + 3000,
  have storeB_cost := 22.5 * x + 3600,
  rw hx at *,
  rw ←calc
    25 * 100 + 3000 = 5500 : by ring
    ... < 22.5 * 100 + 3600 : by norm_num,
  exact ⟨by norm_num⟩,
}

end cost_expressions_store_comparison_l136_136995


namespace coin_flip_difference_l136_136951

/-- The positive difference between the probability of a fair coin landing heads up
exactly 4 times out of 5 flips and the probability of a fair coin landing heads up
5 times out of 5 flips is 1/8. -/
theorem coin_flip_difference :
  (5 * (1 / 2) ^ 5) - ((1 / 2) ^ 5) = (1 / 8) :=
by
  sorry

end coin_flip_difference_l136_136951


namespace investment_calculation_l136_136437

noncomputable def calculate_investment_amount (A : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  A / (1 + r / n) ^ (n * t)

theorem investment_calculation :
  let A := 80000
  let r := 0.07
  let n := 12
  let t := 7
  let P := calculate_investment_amount A r n t
  abs (P - 46962) < 1 :=
by
  sorry

end investment_calculation_l136_136437


namespace distribution_centers_l136_136118

theorem distribution_centers (n : ℕ) (h : n = 5) : 
  (n + (nat.choose n 2) = 15) :=
by
  rw h
  -- simplifying n and the binomial coefficient for n = 5

  sorry

end distribution_centers_l136_136118


namespace range_of_a_for_root_l136_136679

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - 2 * x + a

theorem range_of_a_for_root (a : ℝ) : (∃ x, f x a = 0) ↔ a ≤ 2 * Real.log 2 - 2 := sorry

end range_of_a_for_root_l136_136679


namespace num_ways_to_buy_three_items_l136_136771

-- Defining the conditions based on the problem statement
def num_headphones : ℕ := 9
def num_mice : ℕ := 13
def num_keyboards : ℕ := 5
def num_kb_mouse_sets : ℕ := 4
def num_hp_mouse_sets : ℕ := 5

-- Defining the theorem statement
theorem num_ways_to_buy_three_items :
  (num_kb_mouse_sets * num_headphones) + 
  (num_hp_mouse_sets * num_keyboards) + 
  (num_headphones * num_mice * num_keyboards) = 646 := 
  by
  sorry

end num_ways_to_buy_three_items_l136_136771


namespace find_function_l136_136551

theorem find_function (f : ℤ → ℤ) 
  (h1 : ∀ p : ℤ, prime p → f p > 0)
  (h2 : ∀ p x : ℤ, prime p → p ∣ (f x + f p) ^ (f p) - x) : 
  ∀ x, f x = x :=
begin
  -- Proof goes here
  sorry
end

end find_function_l136_136551


namespace annual_growth_rate_l136_136481

theorem annual_growth_rate (u_2021 u_2023 : ℝ) (x : ℝ) : 
    u_2021 = 1 ∧ u_2023 = 1.69 ∧ x > 0 → (u_2023 / u_2021) = (1 + x)^2 → x * 100 = 30 :=
by
  intros h1 h2
  sorry

end annual_growth_rate_l136_136481


namespace smaller_angle_at_3_20_l136_136913

def minute_hand_angle (minutes : ℕ) : ℝ := minutes * 6

def hour_hand_angle (hours minutes : ℕ) : ℝ :=
  (hours % 12) * 30 + (minutes / 2)

def clock_angle (hours minutes : ℕ) : ℝ :=
  let minute_angle := minute_hand_angle minutes
  let hour_angle := hour_hand_angle hours minutes
  abs (minute_angle - hour_angle)

theorem smaller_angle_at_3_20 : clock_angle 3 20 = 20 :=
by
  unfold clock_angle minute_hand_angle hour_hand_angle
  sorry

end smaller_angle_at_3_20_l136_136913


namespace simplify_trig_expression_l136_136846

variable {x : ℝ}

theorem simplify_trig_expression 
  (h1: cos x ≠ 0)
  (h2: 1 + tan x ≠ 0) :
  (cos x / (1 + tan x)) + ((1 + tan x) / cos x) = cos x + (1 / cos x) :=
  by
  sorry

end simplify_trig_expression_l136_136846


namespace minimum_value_of_a_l136_136665

theorem minimum_value_of_a (a : ℝ) (h : ∀ x ∈ set.Ioo 1 2, ae^x - 1/x ≥ 0) : a ≥ e⁻¹ :=
sorry

end minimum_value_of_a_l136_136665


namespace quotient_is_9_l136_136833

theorem quotient_is_9 :
  ∀ (quotient : ℕ), (100 = 11 * quotient + 1) → quotient = 9 :=
by
  intro quotient
  intro h
  have h' : 100 - 1 = 11 * quotient := by
    rw [h]
    linarith
  have h'' : 99 = 11 * quotient := by
    exact h'
  have quotient_eq_9 : quotient = 99 / 11 := by
    linarith
  have : 99 / 11 = 9 := by
    norm_num
  rw [this] at quotient_eq_9
  exact quotient_eq_9

end quotient_is_9_l136_136833


namespace tangent_product_20_40_60_80_l136_136019

theorem tangent_product_20_40_60_80 :
  Real.tan (20 * Real.pi / 180) * Real.tan (40 * Real.pi / 180) * Real.tan (60 * Real.pi / 180) * Real.tan (80 * Real.pi / 180) = 3 :=
by
  sorry

end tangent_product_20_40_60_80_l136_136019


namespace ice_cream_ratio_l136_136003

-- Definitions based on the conditions
def oli_scoops : ℕ := 4
def victoria_scoops : ℕ := oli_scoops + 4

-- Statement to prove the ratio
theorem ice_cream_ratio :
  victoria_scoops / oli_scoops = 2 :=
by
  -- The exact proof strategy here is omitted with 'sorry'
  sorry

end ice_cream_ratio_l136_136003


namespace fred_earned_from_car_wash_l136_136210

def weekly_allowance : ℕ := 16
def spent_on_movies : ℕ := weekly_allowance / 2
def amount_after_movies : ℕ := weekly_allowance - spent_on_movies
def final_amount : ℕ := 14
def earned_from_car_wash : ℕ := final_amount - amount_after_movies

theorem fred_earned_from_car_wash : earned_from_car_wash = 6 := by
  sorry

end fred_earned_from_car_wash_l136_136210


namespace difference_length_breadth_l136_136029

theorem difference_length_breadth (B L A : ℕ) (h1 : B = 11) (h2 : A = 21 * B) (h3 : A = L * B) :
  L - B = 10 :=
by
  sorry

end difference_length_breadth_l136_136029


namespace blue_balls_count_l136_136991

def num_purple : Nat := 7
def num_yellow : Nat := 11
def min_tries : Nat := 19

theorem blue_balls_count (num_blue: Nat): num_blue = 1 :=
by
  have worst_case_picks := num_purple + num_yellow
  have h := min_tries
  sorry

end blue_balls_count_l136_136991


namespace find_a_values_l136_136387

theorem find_a_values (a x₁ x₂ : ℝ) (h1 : x^2 + a * x - 2 = 0)
                      (h2 : x₁ ≠ x₂)
                      (h3 : x₁^3 + 22 / x₂ = x₂^3 + 22 / x₁) :
                      a = 3 ∨ a = -3 :=
by
  sorry

end find_a_values_l136_136387


namespace count_numbers_with_at_most_two_different_digits_l136_136265

theorem count_numbers_with_at_most_two_different_digits : 
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  number_of_such_numbers := 
    let count_one_digit := 9 in -- 1 to 9
    let count_two_distinct_non_zero := 36 * 8 in -- \(\binom{9}{2}\) * (valid arrangements)
    let count_including_zero := 9 * 4 in -- 9 (choices for non-zero digit) * (valid arrangements with 0)
    count_one_digit + count_two_distinct_non_zero + count_including_zero
  in number_of_such_numbers = 351 :=
by
  sorry

end count_numbers_with_at_most_two_different_digits_l136_136265


namespace correct_response_percentage_l136_136039

def number_of_students : List ℕ := [300, 1100, 100, 600, 400]
def total_students : ℕ := number_of_students.sum
def correct_response_students : ℕ := number_of_students.maximum.getD 0

theorem correct_response_percentage :
  (correct_response_students * 100 / total_students) = 44 := by
  sorry

end correct_response_percentage_l136_136039


namespace part1_angle_A_part2_min_value_l136_136303

noncomputable def Triangle (A B C : ℝ) (a b c R : ℝ) :=
  (2 * R - a = (a * (b^2 + c^2 - a^2)) / (a^2 + c^2 - b^2)) ∧
  (A ≠ π / 2)

theorem part1_angle_A {A B C a b c R : ℝ} (hT: Triangle A B C a b c R) (hB: B = π / 6) :
  A = π / 6 := sorry

theorem part2_min_value {A B C a b c R : ℝ} (hT: Triangle A B C a b c R) :
  ∃ m : ℝ, (m = inf (λ x, (2 * x^2 - c^2) / b^2)) ∧ m = 4 * sqrt 2 - 7 := sorry

end part1_angle_A_part2_min_value_l136_136303


namespace part1_q_value_part1_an_formula_part2_c_sum_l136_136603

open Nat

-- Definitions and conditions
def a : ℕ+ → ℝ := λ n, if n = 1 then 1 else sorry
def b : ℕ+ → ℝ := λ n, if n = 1 then 1 else sorry
def c : ℕ+ → ℝ := λ n, if n = 1 then 1 else sorry

axiom a1 : a 1 = 1
axiom b1 : b 1 = 1
axiom c1 : c 1 = 1

axiom cn_eq_anp1_sub_an : ∀ n : ℕ+, c n = a (n + 1) - a n
axiom cn1_eq_bn_div_bn2_cn : ∀ n : ℕ+, c (n + 1) = (b n / b (n + 2)) * c n

-- Part (Ⅰ): Geometric sequence
axiom geom_seq : ∀ n : ℕ+, b (n + 1) = q^(n-1)
axiom geom_initial : b 1 + b 2 = 6 * b 3

noncomputable def q : ℝ := sorry

theorem part1_q_value : q = 1 / 2 := sorry
theorem part1_an_formula : ∀ n : ℕ+, a n = (4^(n-1) + 2) / 3 := sorry

-- Part (Ⅱ): Arithmetic sequence
axiom arith_seq : ∀ n : ℕ+, b (n + 1) = b 1 + (n-1) * d
axiom d_pos : d > 0

theorem part2_c_sum : ∀ n : ℕ+, c 1 + c 2 + ⋯ + c n < 1 + 1 / d := sorry

end part1_q_value_part1_an_formula_part2_c_sum_l136_136603


namespace area_luns_ratio_l136_136471

theorem area_luns_ratio :
  ∃ (X Y A B V : Point) (k : ℝ),
    (angle A V B = 75) ∧
    (dist A V = real.sqrt 2) ∧
    (dist B V = real.sqrt 3) ∧
    (is_luns_with_vertices A B V X Y) ∧
    (L_is_maximal_area_luns A B V k) →
  k / ((1 + real.sqrt 3) ^ 2) = (25 * real.pi - 18 + 6 * real.sqrt 3) / 144 :=
sorry

end area_luns_ratio_l136_136471


namespace number_of_possible_medians_l136_136804

-- Define the set S and its properties
def S : Set ℕ := {1, 3, 5, 7, 10, 12, 15}

-- Define the function that calculates the median of a set of 11 distinct integers
def median_of_set (T : Set ℕ) (hT : T.card = 11) : ℕ :=
(T.to_finset.sort (· ≤ ·)).nth 5

-- Define the main theorem statement
theorem number_of_possible_medians :
  ∃ (T : Set ℕ), T.card = 11 ∧ S ⊆ T ∧ ∃ (medians : Set ℕ), medians.card = 6 ∧ ∀ m ∈ medians, m = median_of_set T (by sorry) := sorry

end number_of_possible_medians_l136_136804


namespace region_R_perimeter_l136_136527

noncomputable def P0 := (0, 0)
noncomputable def P1 := (0, 4)
noncomputable def P2 := (4, 0)
noncomputable def P3 := (-2, -2)
noncomputable def P4 := (3, 3)
noncomputable def P5 := (5, 5)

noncomputable def distance (p1 p2 : prod ℝ ℝ) := Euclidean.dist p1 p2

def is_closer_to_P0 (p : prod ℝ ℝ) : Prop :=
  ∀ i ∈ [P1, P2, P3, P4, P5], distance p P0 < distance p i

def region_R := { p : prod ℝ ℝ | is_closer_to_P0 p }

noncomputable def perimeter_R : ℝ :=
  let vertices := [(1, 2), (2, 1), (2, -2), (-2, 2)]
  list.sum (list.map (λ i j, distance i j) (vertices ++ [vertices.head]))

theorem region_R_perimeter :
  perimeter_R = 10 + 5 * (Real.sqrt 2) := 
sorry

end region_R_perimeter_l136_136527


namespace number_of_ways_to_buy_three_items_l136_136731

def headphones : ℕ := 9
def mice : ℕ := 13
def keyboards : ℕ := 5
def keyboard_mouse_sets : ℕ := 4
def headphone_mouse_sets : ℕ := 5

theorem number_of_ways_to_buy_three_items : 
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 := 
by 
  sorry

end number_of_ways_to_buy_three_items_l136_136731


namespace cosine_sum_identity_l136_136236

theorem cosine_sum_identity
  {A B C P Q R : Point}
  {AP CR BQ : ℝ}
  (ABC_triangle : Triangle A B C)
  (inscribed_quadrilateral : InscribedQuadrilateral A B C D)
  (extended_meet_R : ExtendedMeet (Ext BA CD) R)
  (extended_meet_P : ExtendedMeet (Ext AD BC) P)
  (intersection_Q : Intersection (Ext AC BD) Q)
  (angle_A : ℝ := angle ABC_triangle ∠ A)
  (angle_B : ℝ := angle ABC_triangle ∠ B)
  (angle_C : ℝ := angle ABC_triangle ∠ C) :
  (cos angle_A) / AP + (cos angle_C) / CR = (cos angle_B) / BQ := sorry

end cosine_sum_identity_l136_136236


namespace tank_leak_time_l136_136507

theorem tank_leak_time
  (fill_time_without_leak : ℝ)
  (fill_time_with_leak : ℝ)
  (R : ℝ := 1 / fill_time_without_leak)
  (L : ℝ := R - 1 / fill_time_with_leak) :
  (1 / L) = 30 :=
by
  have fill_time_without_leak_pos : 0 < fill_time_without_leak := by linarith
  have fill_time_with_leak_pos : 0 < fill_time_with_leak := by linarith
  sorry

end tank_leak_time_l136_136507


namespace ellipse_equation_find_slope_l136_136592

theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_vs : b = 1) (h_fl : 2*sqrt(3) = 2*sqrt(a^2 - b^2)) : 
  ∀ x y : ℝ, (x / a)^2 + (y / b)^2 = 1 ↔ x^2 / 4 + y^2 = 1 :=
by sorry

theorem find_slope (k : ℝ) (P : ℝ × ℝ) (hP : P = (-2, 1)) (h_condition : ∃ B C : ℝ × ℝ, B ≠ C ∧ ∀ t : ℝ, (k*t + B.2)*(k*t + C.2) = -k^2*4*t^2 - 8k*t - 4k^2 - 2 - B.1 + 2 + C.1 + 4f - B.1 / (1-f) - C.1 / (1-f) = 2 ) :
  k = -4 :=
by sorry

end ellipse_equation_find_slope_l136_136592


namespace average_speed_excl_stoppages_l136_136548

-- Define the average speed of the bus including stoppages
def speed_incl_stoppages : ℝ := 40

-- Define the time the bus stops per hour in minutes
def stoppage_time_per_hour : ℝ := 12

-- Prove that the average speed of the bus excluding stoppages is 50 km/hr.
theorem average_speed_excl_stoppages (speed_incl_stoppages = 40) (stoppage_time_per_hour = 12) : 
    let t_moving_per_hour := 60 - stoppage_time_per_hour in
    let ratio_moving_time := t_moving_per_hour / 60 in
    let speed_excl_stoppages := speed_incl_stoppages / ratio_moving_time in
    speed_excl_stoppages = 50 :=
by sorry

end average_speed_excl_stoppages_l136_136548


namespace school_fitness_event_participants_l136_136785

theorem school_fitness_event_participants :
  let p0 := 500 -- initial number of participants in 2000
  let r1 := 0.3 -- increase rate in 2001
  let r2 := 0.4 -- increase rate in 2002
  let r3 := 0.5 -- increase rate in 2003
  let p1 := p0 * (1 + r1) -- participants in 2001
  let p2 := p1 * (1 + r2) -- participants in 2002
  let p3 := p2 * (1 + r3) -- participants in 2003
  p3 = 1365 -- prove that number of participants in 2003 is 1365
:= sorry

end school_fitness_event_participants_l136_136785


namespace javier_meatballs_smallest_count_l136_136794

theorem javier_meatballs_smallest_count :
  ∃ (n : ℕ), n * 15 = m * 10 ∧ n * 15 = p * 20 ∧ n = 4 :=
by {
  use 4,
  split,
  {
    calc 4 * 15 = 60 : by norm_num
       ... = 6 * 10 : by norm_num,
  },
  {
    calc 4 * 15 = 60 : by norm_num
       ... = 3 * 20 : by norm_num,
  }
}

end javier_meatballs_smallest_count_l136_136794


namespace buy_items_ways_l136_136726

theorem buy_items_ways (headphones keyboards mice keyboard_mouse_sets headphone_mouse_sets : ℕ) :
  headphones = 9 → keyboards = 5 → mice = 13 → keyboard_mouse_sets = 4 → headphone_mouse_sets = 5 →
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 :=
by
  intros h_eq k_eq m_eq kms_eq hms_eq
  have h_eq_gen : headphones = 9 := h_eq
  have k_eq_gen : keyboards = 5 := k_eq
  have m_eq_gen : mice = 13 := m_eq
  have kms_eq_gen : keyboard_mouse_sets = 4 := kms_eq
  have hms_eq_gen : headphone_mouse_sets = 5 := hms_eq
  sorry

end buy_items_ways_l136_136726


namespace num_integers_between_cubes_l136_136263

/-- Proof that the number of integers between (9.5)^3 and (9.7)^3 is 55. -/
theorem num_integers_between_cubes : 
  let a := 9.5
  let b := 9.7
  let n := ⌊(b)^3⌋ - ⌈(a)^3⌉ + 1 in
  n = 55 := by
let a := 9.5
let b := 9.7
let a_cubed := a ^ 3
let b_cubed := b ^ 3
let lower_bound := Int.ceil (a_cubed)
let upper_bound := Int.floor (b_cubed)
let n := upper_bound - lower_bound + 1
have h1 : a_cubed = 857.375 := by
  calc
    a ^ 3 = (9 + 0.5) ^ 3 : by rw [add_right_comm 9 0.5]
        ... = 9^3 + 3*9^2*0.5 + 3*9*0.5^2 + 0.5^3 :
          by rw [←add_assoc, ←add_assoc, ←add_assoc, ←add_assoc, rfl,
                 rfl, rfl, rfl, rfl]
have h2 : b_cubed = 912.673 := by
  calc
    b ^ 3 = (9 + 0.7) ^ 3 : by rw [add_right_comm 9 0.7]
        ... = 9^3 + 3*9^2*0.7 + 3*9*0.7^2 + 0.7^3 :
          by rw [←add_assoc, ←add_assoc, ←add_assoc, ←add_assoc, rfl,
                 rfl, rfl, rfl, rfl]
have h3 : lower_bound = 858 := by
  calc
    ⌊a_cubed⌋ = 858 : by rw [←Int.ceil, h1]
have h4 : upper_bound = 912 := by
  calc
    ⌊b_cubed⌋ = 912 : by rw [←Int.floor, h2]
have h5 : n = 55 := by
  calc
    n = 912 - 858 + 1 : by rw [←upper_bound, ←lower_bound]
      ... = 55 : by norm_num
exact h5

end num_integers_between_cubes_l136_136263


namespace distinct_integer_lengths_l136_136015

-- Define the vertices of the triangle and their properties
structure right_triangle (A B C : Type) [metric_space A] [metric_space B] [metric_space C] :=
  (hypotenuse : Type)
  (leg1_length : ℝ)
  (leg2_length : ℝ)
  (hypotenuse_length : ℝ)
  (is_right_triangle : leg1_length ^ 2 + leg2_length ^ 2 = hypotenuse_length ^ 2)

-- Define the specific triangle DEF with given leg lengths
def triangle_DEF : right_triangle ℝ :=
{ hypotenuse := ℝ,
  leg1_length := 15,
  leg2_length := 20,
  hypotenuse_length := 25,
  is_right_triangle := by norm_num }

-- The theorem states that the number of integer lengths from E to points on DF is 9.
theorem distinct_integer_lengths : 
  ∀ t : right_triangle ℝ, t.leg1_length = 15 → t.leg2_length = 20 → t.hypotenuse_length = 25 → 9 =
  set.card (set.filter (λ d : ℝ, ∃ x : ℝ, 0 ≤ x ∧ x ≤ t.hypotenuse_length ∧ d∈ set.range (λ y, y)) {12, 13, 14, 15, 16, 17, 18, 19, 20}) :=
sorry

end distinct_integer_lengths_l136_136015


namespace problem1_problem2_l136_136985

-- Proof Problem (1)
theorem problem1 (x : ℝ) (hx : 0 < x ∧ x < 1) : 
  let f := fun x => 2^x in
  let g := fun x => 4^x in
  g (g x) > g (f x) ∧ g (f x) > f (g x) :=
sorry

-- Proof Problem (2)
theorem problem2 (x : ℝ) (hx : x ∈ (-∞, 0] ∪ [1, 2]) : 
  ∃ y, y = 4^x - 3 * 2^x + 3 ∧ 1 ≤ y ∧ y ≤ 7 :=
sorry

end problem1_problem2_l136_136985


namespace proj_v_eq_v_l136_136692

namespace VectorProjection

def vector_v : ℝ × ℝ := (8, -12)
def vector_w : ℝ × ℝ := (-4, 6)

def proj_w_v (v w : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (v.1 * w.1 + v.2 * w.2) / (w.1 * w.1 + w.2 * w.2)
  (scalar * w.1, scalar * w.2)

theorem proj_v_eq_v : proj_w_v vector_v vector_w = vector_v := by
  sorry

end VectorProjection

end proj_v_eq_v_l136_136692


namespace cos_five_pi_over_six_l136_136566

theorem cos_five_pi_over_six :
  Real.cos (5 * Real.pi / 6) = -(Real.sqrt 3 / 2) :=
sorry

end cos_five_pi_over_six_l136_136566


namespace sum_of_valid_n_l136_136201

theorem sum_of_valid_n :
  ∃ (n : ℤ), (n^2 - 17*n + 72 = (m : ℤ)^2) ∧ (24 % n = 0) → n = 8 := 
begin
  sorry
end

end sum_of_valid_n_l136_136201


namespace angle_ABH_eq_angle_CBO_l136_136342

open Real EuclideanGeometry

-- Assume this triangle ABC
variables {A B C H O : Point}

-- H is the orthocenter, and O is the circumcenter of triangle ABC
variables (h₁ : is_orthocenter H A B C) (h₂ : is_circumcenter O A B C)

-- We need to prove that the angle ∠ABH is equal to the angle ∠CBO
theorem angle_ABH_eq_angle_CBO : ∠ A B H = ∠ C B O :=
sorry

end angle_ABH_eq_angle_CBO_l136_136342


namespace radius_greater_than_diameter_of_base_l136_136834

noncomputable def volume_frustum (R r h : ℝ) : ℝ :=
  (h * Real.pi / 3) * (R^2 + R * r + r^2)

theorem radius_greater_than_diameter_of_base
  (R r h : ℝ) (h_pos : 0 < h) (R_pos : 0 < R) (r_pos : 0 < r)
  (volume_relation : volume_frustum R r h / 2 = volume_frustum ((R + r) / 2) r (h / 2)) :
  R > 2 * r :=
begin
  sorry -- Proof will be filled here
end

end radius_greater_than_diameter_of_base_l136_136834


namespace purchase_combinations_correct_l136_136752

theorem purchase_combinations_correct : 
  let headphones : ℕ := 9
  let mice : ℕ := 13
  let keyboards : ℕ := 5
  let sets_keyboard_mouse : ℕ := 4
  let sets_headphones_mouse : ℕ := 5
in 
  (sets_keyboard_mouse * headphones) + 
  (sets_headphones_mouse * keyboards) + 
  (headphones * mice * keyboards) = 646 := 
by
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let sets_keyboard_mouse := 4
  let sets_headphones_mouse := 5
  sorry

end purchase_combinations_correct_l136_136752


namespace proj_b_v_is_correct_l136_136330

open Real

noncomputable def a : Vector ℝ := sorry
noncomputable def b : Vector ℝ := sorry

axiom orthogonal_a_b : a ⬝ b = 0

noncomputable def v : Vector ℝ := Vec2 4 (-2)
noncomputable def proj_a_v : Vector ℝ := Vec2 (-4/5) (-8/5)

axiom proj_a_v_property : proj a v = proj_a_v

theorem proj_b_v_is_correct : proj b v = Vec2 (24/5) (-2/5) :=
sorry

end proj_b_v_is_correct_l136_136330


namespace ramu_profit_percent_l136_136842

variables (cost_car cost_repair selling_price : ℝ)

def total_cost (cost_car cost_repair : ℝ) : ℝ := cost_car + cost_repair
def profit (selling_price total_cost : ℝ) : ℝ := selling_price - total_cost
def profit_percent (profit total_cost : ℝ) : ℝ := (profit / total_cost) * 100

theorem ramu_profit_percent :
  cost_car = 42000 →
  cost_repair = 13000 →
  selling_price = 64900 →
  profit_percent (profit selling_price (total_cost cost_car cost_repair)) (total_cost cost_car cost_repair) = 18 :=
begin
  sorry
end

end ramu_profit_percent_l136_136842


namespace min_value_of_a_l136_136635

theorem min_value_of_a (a : ℝ) : 
  (∀ x ∈ Ioo 1 2, (a * exp x - 1/x) ≥ 0) → a = exp (-1) :=
by
  sorry

end min_value_of_a_l136_136635


namespace equations_of_other_sides_l136_136388

theorem equations_of_other_sides (x y: ℝ) (D: ℝ × ℝ) 
  (H1: ∀ x y, x + y + 1 = 0)
  (H2: ∀ x y, 3 * x - 4 = 0)
  (H3: D = (3,3)) :
  (∀ x y, x + y - 13 = 0) ∧ (∀ x y, 3 * x - y - 16 = 0) :=
by
  sorry

end equations_of_other_sides_l136_136388


namespace pyramid_frustum_volume_fraction_l136_136134

theorem pyramid_frustum_volume_fraction 
  (base_edge original_height : ℝ)
  (base_edge = 24) 
  (original_height = 18) : 
  let smaller_height := original_height / 3
      smaller_base_edge := base_edge / 3
      original_volume := (1 / 3) * (base_edge ^ 2) * original_height
      smaller_volume := (1 / 3) * (smaller_base_edge ^ 2) * smaller_height
      frustum_volume := original_volume - smaller_volume
  in frustum_volume / original_volume = 23 / 24 :=
by sorry

end pyramid_frustum_volume_fraction_l136_136134


namespace total_cups_l136_136402

theorem total_cups {butter baking_soda flour sugar : ℕ} 
  (ratio : butter = 1 ∧ baking_soda = 2 ∧ flour = 5 ∧ sugar = 3) 
  (flour_cups : flour = 15) : 
  (butter * (flour_cups / flour) + baking_soda * (flour_cups / flour) + flour_cups + sugar * (flour_cups / flour) = 33) :=
sorry

end total_cups_l136_136402


namespace pythagorean_triple_example_l136_136089

theorem pythagorean_triple_example : ∃ (a b c : ℕ), a = 5 ∧ b = 12 ∧ c = 13 ∧ a^2 + b^2 = c^2 :=
by
  use 5, 12, 13
  simp
  sorry

end pythagorean_triple_example_l136_136089


namespace exist_three_arcs_with_sum_diff_le_one_l136_136982

theorem exist_three_arcs_with_sum_diff_le_one (x : List ℝ) 
  (hx : ∀ i, 0 < x.nth i 0 ∧ x.nth i 0 ≤ 1) : 
  ∃ a b c, a ≤ b ∧ b ≤ c ∧ a + b + c = x.sum ∧ c - a ≤ 1 :=
begin
  sorry
end

end exist_three_arcs_with_sum_diff_le_one_l136_136982


namespace swimming_pool_volume_correct_l136_136137

def shallow_side_volume := 9 * 12 * 1
def deep_side_volume := 15 * 18 * 4
def island_volume := 3 * 6 * 1
def pool_volume := shallow_side_volume + deep_side_volume - island_volume

theorem swimming_pool_volume_correct : pool_volume = 1170 :=
by
  unfold shallow_side_volume deep_side_volume island_volume pool_volume
  simp
  sorry

end swimming_pool_volume_correct_l136_136137


namespace distinct_seatings_l136_136375

theorem distinct_seatings : 
  ∃ n : ℕ, (n = 288000) ∧ 
  (∀ (men wives : Fin 6 → ℕ),
  ∃ (f : (Fin 12) → ℕ), 
  (∀ i, f (i + 1) % 12 ≠ f i) ∧
  (∀ i, f i % 2 = 0) ∧
  (∀ j, f (2 * j) = men j ∧ f (2 * j + 1) = wives j)) :=
by
  sorry

end distinct_seatings_l136_136375


namespace line_in_slope_intercept_form_l136_136491

theorem line_in_slope_intercept_form :
  ∃ m b : ℚ, (∀ x y : ℚ, 
    (begin
      let vec := λ x y, -3 * (x - 3) + -7 * (y - 14),
      ∃ vec, vec = 0 -> y = m * x + b
    end)) ∧ (m = -3 / 7) ∧ (b = 107 / 7) :=
sorry

end line_in_slope_intercept_form_l136_136491


namespace correct_ordering_of_powers_l136_136451

theorem correct_ordering_of_powers :
  (6 ^ 8) < (3 ^ 15) ∧ (3 ^ 15) < (8 ^ 10) :=
by
  -- Define the expressions for each power
  let a := (8 : ℕ) ^ 10
  let b := (3 : ℕ) ^ 15
  let c := (6 : ℕ) ^ 8
  
  -- To utilize the values directly in inequalities
  have h1 : (c < b) := sorry -- Proof that 6^8 < 3^15
  have h2 : (b < a) := sorry -- Proof that 3^15 < 8^10

  exact ⟨h1, h2⟩ -- Conjunction of h1 and h2 to show 6^8 < 3^15 < 8^10

end correct_ordering_of_powers_l136_136451


namespace inequality_solution_set_l136_136404

theorem inequality_solution_set :
  { x : ℝ | (x + 1) * (2 - x) ≤ 0 } = set.Icc (-1 : ℝ) 2 :=
by
  sorry

end inequality_solution_set_l136_136404


namespace circle_equation_l136_136033

/-- 
  Prove that the equation of a circle with center (2, -1) and tangent to the line x - y + 1 = 0 
  is (x - 2)^2 + (y + 1)^2 = 8.
--/
theorem circle_equation :
  ∃ r: ℝ, 
    ∀ x y: ℝ,
      (x - 2) ^ 2 + (y + 1) ^ 2 = r ^ 2 ∧ 
      r = (|2 + 1 + 1| / real.sqrt (1 ^ 2 + (-1) ^ 2)) → 
      (x - 2) ^ 2 + (y + 1) ^ 2 = 8 
  :=
  by
  -- sorry to skip the actual proof
  sorry

end circle_equation_l136_136033


namespace solve_system_of_equations_l136_136970

-- Given conditions
variables {a b c k x y z : ℝ}
variables (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
variables (eq1 : a * x + b * y + c * z = k)
variables (eq2 : a^2 * x + b^2 * y + c^2 * z = k^2)
variables (eq3 : a^3 * x + b^3 * y + c^3 * z = k^3)

-- Statement to be proved
theorem solve_system_of_equations :
  x = k * (k - c) * (k - b) / (a * (a - c) * (a - b)) ∧
  y = k * (k - c) * (k - a) / (b * (b - c) * (b - a)) ∧
  z = k * (k - a) * (k - b) / (c * (c - a) * (c - b)) :=
sorry

end solve_system_of_equations_l136_136970


namespace slope_range_l136_136391

-- Define the conditions
def hyperbola : Set (ℝ × ℝ) := { p | p.1^2 - p.2^2 = 1 }
def leftFocus : ℝ × ℝ := (-sqrt 2, 0)
def lowerLeftBranch (p : ℝ × ℝ) : Prop := p ∈ hyperbola ∧ p.2 < 0 ∧ p ≠ (-1, 0)

-- Define the problem
theorem slope_range (P : ℝ × ℝ) (h : lowerLeftBranch P) :
  ∃ m, (m = ((P.2 - leftFocus.2) / (P.1 - leftFocus.1))) ∧ (m ∈ (-∞, 0) ∨ m ∈ (1, +∞)) :=
sorry

end slope_range_l136_136391


namespace part1_part2_l136_136827

open Real

def f (x : ℝ) : ℝ := |2 * x - 1| - |x + 3 / 2|

theorem part1 (x : ℝ) : f(x) < 0 ↔ -1 / 6 < x ∧ x < 5 / 2 := sorry

theorem part2 (x₀ m : ℝ) (h : f(x₀) + 3 * m ^ 2 < 5 * m) : -1 / 3 < m ∧ m < 2 := sorry

end part1_part2_l136_136827


namespace shaded_region_area_l136_136441

def line1 (x : ℝ) : ℝ := -x + 5
def line2 (x : ℝ) : ℝ := - (2 / 3) * x + 6

theorem shaded_region_area :
  let f := λ x => line2 x - line1 x
  (∫ x in 0..5, f x) = 55/6 :=
by
  sorry

end shaded_region_area_l136_136441


namespace longer_train_length_l136_136070

-- Definitions based on conditions
def speed_train_1_kmh := 42
def speed_train_2_kmh := 30
def shorter_train_length_m := 160
def time_clear_s := 23.998

-- Converting speeds from km/h to m/s
def speed_train_1_ms := speed_train_1_kmh * 1000 / 3600
def speed_train_2_ms := speed_train_2_kmh * 1000 / 3600

-- Relative speed in m/s
def relative_speed_ms := speed_train_1_ms + speed_train_2_ms

-- Total distance covered in the given time (23.998 seconds)
def total_distance_m := relative_speed_ms * time_clear_s

-- The length of the longer train
def longer_train_length_m := total_distance_m - shorter_train_length_m

-- The proof problem statement
theorem longer_train_length :
  longer_train_length_m = 319.96 := sorry

end longer_train_length_l136_136070


namespace number_of_ways_to_buy_three_items_l136_136737

def headphones : ℕ := 9
def mice : ℕ := 13
def keyboards : ℕ := 5
def keyboard_mouse_sets : ℕ := 4
def headphone_mouse_sets : ℕ := 5

theorem number_of_ways_to_buy_three_items : 
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 := 
by 
  sorry

end number_of_ways_to_buy_three_items_l136_136737


namespace percentage_of_360_eq_108_l136_136962

theorem percentage_of_360_eq_108 : ∃ x : ℝ, (108.0 = (x / 100) * 360) → x = 30 :=
by
  intro h
  sorry

end percentage_of_360_eq_108_l136_136962


namespace jenna_discount_l136_136309

def normal_price : ℝ := 50
def tickets_from_website : ℝ := 2 * normal_price
def scalper_initial_price_per_ticket : ℝ := 2.4 * normal_price
def scalper_total_initial : ℝ := 2 * scalper_initial_price_per_ticket
def friend_discounted_ticket : ℝ := 0.6 * normal_price
def total_price_five_tickets : ℝ := tickets_from_website + scalper_total_initial + friend_discounted_ticket
def amount_paid_by_friends : ℝ := 360

theorem jenna_discount : 
    total_price_five_tickets - amount_paid_by_friends = 10 :=
by
  -- The proof would go here, but we leave it as sorry for now.
  sorry

end jenna_discount_l136_136309


namespace count_quadratic_polynomials_l136_136323

noncomputable def P (x : ℝ) : ℝ := (x - 1) * (x - 3) * (x - 5)

theorem count_quadratic_polynomials :
  ∃ Qs : Finset (Polynomial ℝ), Qs.card = 22 ∧ 
  ∀ Q ∈ Qs, ∃ R : Polynomial ℝ, R.degree = 3 ∧ Polynomial.eval₂ ring_hom.id Q = P * R :=
sorry

end count_quadratic_polynomials_l136_136323


namespace book_cost_l136_136068

variable (x : ℕ) -- x is the number of type A books
variable (h : x ≤ 100) -- x cannot exceed 100 because there are only 100 books

theorem book_cost (h_total : x + (100 - x) = 100)
    (h_A_price : ∀ x, 10 * x)
    (h_B_price : ∀ y, 8 * y) :
  8 * (100 - x) = 8 * (100 - x) :=
by 
suffices h_cost : 8 * (100 - x) = 8 * (100 - x), from sorry,
sorry

end book_cost_l136_136068


namespace true_propositions_l136_136145

theorem true_propositions :
  (∀ (E : Type) (P : E → Prop), (∀ x : E, P x) → true) ∧
  (¬ (∀ b : ℝ, b^2 = 9 → b = 3)) ∧
  (∀ (E : Type) (A B : set E), A ∩ B = ∅ → true) ∧
  (∀ (T₁ T₂ : Type) (a₁ a₂ : T₁ → T₂ → Prop), (∀ x y, a₁ x y ↔ a₂ x y) → true) :=
sorry

end true_propositions_l136_136145


namespace infinite_geometric_series_sum_l136_136544

-- Definitions of first term and common ratio
def a : ℝ := 1 / 5
def r : ℝ := 1 / 2

-- Proposition to prove that the sum of the geometric series is 2/5
theorem infinite_geometric_series_sum : (a / (1 - r)) = (2 / 5) :=
by 
  -- Sorry is used here as a placeholder for the proof.
  sorry

end infinite_geometric_series_sum_l136_136544


namespace nth_row_full_of_ones_l136_136370

noncomputable def is_full_of_ones (row : List Nat) : Prop :=
  ∀ x ∈ row, x = 1

theorem nth_row_full_of_ones (n : Nat) : 
  is_full_of_ones (pascal_triangle_row (2^n - 1)) :=
begin
  sorry
end

end nth_row_full_of_ones_l136_136370


namespace positive_difference_between_probabilities_is_one_eighth_l136_136933

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem positive_difference_between_probabilities_is_one_eighth :
  let p : ℚ := 1 / 2
    n : ℕ := 5
    prob_4_heads := binomial_probability n 4 p
    prob_5_heads := binomial_probability n 5 p
  in |prob_4_heads - prob_5_heads| = 1 / 8 :=
by
  let p : ℚ := 1 / 2
  let n : ℕ := 5
  let prob_4_heads := binomial_probability n 4 p
  let prob_5_heads := binomial_probability n 5 p
  sorry

end positive_difference_between_probabilities_is_one_eighth_l136_136933


namespace tangent_line_at_zero_l136_136194

-- Define the function f
def f (x : ℝ) : ℝ := Real.sin x + Real.exp x + 2

-- Define the point P(0, f(0))
def P : ℝ × ℝ := (0, f 0)

-- Define the equation of the tangent line
def tangent_line (k b x : ℝ) : ℝ := k * x + b

-- The slope at x = 0 is given by the derivative f' at x = 0
def derivative_at_zero : ℝ := (Real.cos 0) + (Real.exp 0)

-- Formal proof statement in Lean
theorem tangent_line_at_zero (x : ℝ) : tangent_line (derivative_at_zero) (f 0) x = 2 * x + 3 :=
by
  sorry

end tangent_line_at_zero_l136_136194


namespace num_ways_to_buy_three_items_l136_136772

-- Defining the conditions based on the problem statement
def num_headphones : ℕ := 9
def num_mice : ℕ := 13
def num_keyboards : ℕ := 5
def num_kb_mouse_sets : ℕ := 4
def num_hp_mouse_sets : ℕ := 5

-- Defining the theorem statement
theorem num_ways_to_buy_three_items :
  (num_kb_mouse_sets * num_headphones) + 
  (num_hp_mouse_sets * num_keyboards) + 
  (num_headphones * num_mice * num_keyboards) = 646 := 
  by
  sorry

end num_ways_to_buy_three_items_l136_136772


namespace main_theorem_l136_136052

-- Define the sequences {u_n} and {v_n}
def u : ℕ → ℤ
| 0       := 1
| 1       := 1
| (n + 2) := 2 * u (n + 1) - 3 * u n

def v : ℕ → ℤ
| 0       := a
| 1       := b
| 2       := c
| (n + 3) := v (n + 2) - 3 * v (n + 1) + 27 * v n

noncomputable def exists_N : Prop :=
∃ N : ℕ, ∀ n : ℕ, n > N → u n ∣ v n

theorem main_theorem (a b c : ℤ) (h : exists_N) : 3 * a = 2 * b + c := 
sorry

end main_theorem_l136_136052


namespace train_pass_time_l136_136094

-- Definitions based on the conditions
def train_length : ℕ := 360   -- Length of the train in meters
def platform_length : ℕ := 190 -- Length of the platform in meters
def speed_kmh : ℕ := 45       -- Speed of the train in km/h
def speed_ms : ℚ := speed_kmh * (1000 / 3600) -- Speed of the train in m/s

-- Total distance to be covered
def total_distance : ℕ := train_length + platform_length 

-- Time taken to pass the platform
def time_to_pass_platform : ℚ := total_distance / speed_ms

-- Proof that the time taken is 44 seconds
theorem train_pass_time : time_to_pass_platform = 44 := 
by 
  -- this is where the detailed proof would go
  sorry  

end train_pass_time_l136_136094


namespace customers_tipped_count_l136_136141

variable (initial_customers : ℕ)
variable (added_customers : ℕ)
variable (customers_no_tip : ℕ)

def total_customers (initial_customers added_customers : ℕ) : ℕ :=
  initial_customers + added_customers

theorem customers_tipped_count 
  (h_init : initial_customers = 29)
  (h_added : added_customers = 20)
  (h_no_tip : customers_no_tip = 34) :
  (total_customers initial_customers added_customers - customers_no_tip) = 15 :=
by
  sorry

end customers_tipped_count_l136_136141


namespace solve_problem_l136_136165

def problem_statement : Prop :=
  ⌊ (2011^3 : ℝ) / (2009 * 2010) - (2009^3 : ℝ) / (2010 * 2011) ⌋ = 8

theorem solve_problem : problem_statement := 
  by sorry

end solve_problem_l136_136165


namespace hockey_league_games_l136_136425

theorem hockey_league_games (n : ℕ) (k : ℕ) (h_n : n = 25) (h_k : k = 15) : 
  (n * (n - 1) / 2) * k = 4500 := by
  sorry

end hockey_league_games_l136_136425


namespace distance_between_homes_correct_l136_136360

noncomputable def distance_between_homes (maxwell_speed : ℝ) (brad_speed : ℝ) (maxwell_distance_to_meeting : ℝ) : ℝ :=
  let meeting_time := maxwell_distance_to_meeting / maxwell_speed
  let brad_distance_to_meeting := brad_speed * meeting_time
  maxwell_distance_to_meeting + brad_distance_to_meeting

theorem distance_between_homes_correct :
  distance_between_homes 4 6 20 = 50 :=
by
  unfold distance_between_homes
  norm_num
  sorry

end distance_between_homes_correct_l136_136360


namespace line_eq_circle_eq_l136_136598

section
  variable (A B : ℝ × ℝ)
  variable (A_eq : A = (4, 6))
  variable (B_eq : B = (-2, 4))

  theorem line_eq : ∃ (a b c : ℝ), (a, b, c) = (1, -3, 14) ∧ ∀ x y, (y - 6) = ((4 - 6) / (-2 - 4)) * (x - 4) → a * x + b * y + c = 0 :=
  sorry

  theorem circle_eq : ∃ (h k r : ℝ), (h, k, r) = (1, 5, 10) ∧ ∀ x y, (x - 1)^2 + (y - 5)^2 = 10 :=
  sorry
end

end line_eq_circle_eq_l136_136598


namespace midpoint_of_segment_distance_of_segment_l136_136158

noncomputable def midpoint (x1 y1 x2 y2 : ℝ) : ℝ × ℝ :=
  ((x1 + x2) / 2, (y1 + y2) / 2)

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem midpoint_of_segment (x1 y1 x2 y2 : ℝ):
  midpoint 9 (-8) (-5) 6 = (2, -1) := by
  sorry

theorem distance_of_segment (x1 y1 x2 y2 : ℝ):
  distance 9 (-8) (-5) 6 = 14 * real.sqrt 2 := by
  sorry

end midpoint_of_segment_distance_of_segment_l136_136158


namespace max_product_with_859_l136_136447

def digits := {3, 5, 6, 8, 9}

theorem max_product_with_859 :
  ∃ d e, {d, e} ⊆ digits ∧ d ≠ 8 ∧ e ≠ 5 ∧ d ≠ e ∧ (859 * (10 * d + e) = 86738) :=
sorry

end max_product_with_859_l136_136447


namespace distance_between_points_A_B_is_sqrt_109_l136_136801

noncomputable def distance_AB (r1 r2 θ1 θ2 : ℝ) : ℝ :=
  let angle := θ1 - θ2
  Math.sqrt (r1 ^ 2 + r2 ^ 2 - 2 * r1 * r2 * Real.cos angle)

theorem distance_between_points_A_B_is_sqrt_109 (θ1 θ2 : ℝ) (h : θ1 - θ2 = π / 3) :
  distance_AB 5 12 θ1 θ2 = Real.sqrt 109 :=
by
  -- Define the values
  let r1 := 5
  let r2 := 12
  have angle : θ1 - θ2 = π / 3 := h
  -- Calculate the cosine of the angle
  have cos_angle := Real.cos (π / 3)
  -- Calculate the expected distance squared
  let ab_squared := r1^2 + r2^2 - 2 * r1 * r2 * cos_angle
  -- ab_squared should be 109
  have ab_squared_is_109 : ab_squared = 109 := by sorry
  -- Distance AB is the sqrt of ab_squared
  show distance_AB 5 12 θ1 θ2 = Real.sqrt 109
  rw [distance_AB, ab_squared_is_109]
  simp
  sorry

end distance_between_points_A_B_is_sqrt_109_l136_136801


namespace range_of_sum_of_roots_l136_136542

/-- 
  Let x1 and x2 be real numbers such that
  x1 * 2^x1 = 1 and
  x2 * log2 x2 = 1.
  Then the sum x1 + x2 lies in the interval (2, +∞).
-/
theorem range_of_sum_of_roots (x1 x2 : ℝ)
  (hx1 : x1 * 2^x1 = 1)
  (hx2 : x2 * real.log x2 / real.log 2 = 1) :
  2 < x1 + x2 := 
sorry

end range_of_sum_of_roots_l136_136542


namespace five_digit_palindromes_count_l136_136175

theorem five_digit_palindromes_count : 
  let a_choices := 9 in
  let b_choices := 10 in
  let c_choices := 10 in
  a_choices * b_choices * c_choices = 900 :=
by
  sorry

end five_digit_palindromes_count_l136_136175


namespace average_DE_l136_136567

theorem average_DE 
  (a b c d e : ℝ) 
  (avg_all : (a + b + c + d + e) / 5 = 80) 
  (avg_abc : (a + b + c) / 3 = 78) : 
  (d + e) / 2 = 83 := 
sorry

end average_DE_l136_136567


namespace max_students_l136_136436

theorem max_students : 
  ∃ x : ℕ, x < 100 ∧ x % 9 = 4 ∧ x % 7 = 3 ∧ ∀ y : ℕ, (y < 100 ∧ y % 9 = 4 ∧ y % 7 = 3) → y ≤ x := 
by
  sorry

end max_students_l136_136436


namespace sum_b_15_l136_136343

noncomputable def b (n : ℕ) : ℕ :=
  if n = 1 then 1 else
  if n = 2 then 2 else
  if n = 3 then 1 else
  let p := b (n - 1) in
  let q := b (n - 2) * b (n - 3) in
  if (9 * p ^ 2 - 4 * q ≥ 0) && (q ≠ 0) then 4 else 0

theorem sum_b_15 : (∑ n in Finset.range 15, b (n+1)) = 55 :=
sorry

end sum_b_15_l136_136343


namespace grassy_plot_width_l136_136131

noncomputable def gravel_cost (L w p : ℝ) : ℝ :=
  0.80 * ((L + 2 * p) * (w + 2 * p) - L * w)

theorem grassy_plot_width
  (L : ℝ) 
  (p : ℝ) 
  (cost : ℝ) 
  (hL : L = 110) 
  (hp : p = 2.5) 
  (hcost : cost = 680) :
  ∃ w : ℝ, gravel_cost L w p = cost ∧ w = 97.5 :=
by
  sorry

end grassy_plot_width_l136_136131


namespace time_passed_since_midnight_l136_136085

theorem time_passed_since_midnight (h : ℝ) :
  h = (12 - h) + (2/5) * h → h = 7.5 :=
by
  sorry

end time_passed_since_midnight_l136_136085


namespace min_value_of_a_l136_136656

noncomputable def min_a_monotone_increasing : ℝ :=
  let f : ℝ → ℝ := λ x a, a * exp x - log x in 
  let a_min : ℝ := exp (-1) in
  a_min

theorem min_value_of_a 
  {f : ℝ → ℝ} 
  {a : ℝ}
  (mono_incr : ∀ x ∈ Ioo 1 2, 0 ≤ deriv (λ x, a * exp x - log x) x) :
  a ≥ exp (-1) :=
begin
  sorry
end

end min_value_of_a_l136_136656


namespace range_of_a_for_f1_ratio_d_over_t_range_of_b_l136_136578

noncomputable def f1 (x : ℝ) := 3^|(x - 1)|
noncomputable def f2 (x : ℝ) (a : ℝ) := a * 3^|(x - 2)|
noncomputable def f (x : ℝ) (a : ℝ) :=
  if f1 x ≤ f2 x a then f1 x else f2 x a

-- (1) Prove the range of a for f(x) = f1(x) for all x
theorem range_of_a_for_f1 (a : ℝ) (h : ∀ x, f x a = f1 x) : 3 ≤ a :=
  sorry

-- (2) Prove the ratio d/t for different ranges of a
theorem ratio_d_over_t (t : ℝ) (a : ℝ) (h_pos : t > 0) (h_eq : f 0 a = f t a) : d / t = 1 / 2 :=
  sorry

-- (3) Given a = 2, find the range of b
noncomputable def g (x : ℝ) (b : ℝ) := x^2 - 2*b*x + 3
theorem range_of_b (b : ℝ) (h : ∀ m : ℝ, ∃ n ∈ set.Icc 1 2, ∀ x : ℝ, f x 2 ≥ g n b) : sqrt 2 ≤ b :=
  sorry

end range_of_a_for_f1_ratio_d_over_t_range_of_b_l136_136578


namespace minimum_value_of_a_l136_136646

theorem minimum_value_of_a (a : ℝ) :
  (∀ x ∈ (Ioo 1 2), ae^x - ln x ≥ 0) ↔ a ≥ exp(-1) := sorry

end minimum_value_of_a_l136_136646


namespace product_of_intersection_coordinates_l136_136200

theorem product_of_intersection_coordinates :
  (∀ x y : ℝ,
    (x^2 - 4*x + y^2 - 6*y + 13 = 0) ∧ (x^2 - 6*x + y^2 - 6*y + 20 = 0) →
    ∃ a1 a2 b1 b2 : ℝ,
      a1 = 7/2 ∧ a2 = 7/2 ∧
      (b1 = 3 + sqrt(7) / 2 ∨ b1 = 3 - sqrt(7) / 2) ∧
      (b2 = 3 + sqrt(7) / 2 ∨ b2 = 3 - sqrt(7) / 2) ∧
      (a1 * b1 * a2 * b2 = 1421 / 16)) := sorry

end product_of_intersection_coordinates_l136_136200


namespace smallest_expression_at_7_l136_136547

theorem smallest_expression_at_7 :
  let x := 7 in
  min (3 * x / x^2) (min (5 / (x + 2)) (min (sqrt (49 / x)) (min (x^2 / 14) ((x + 3) / 9)))) = 3 / 7 :=
by
  let x := 7
  sorry

end smallest_expression_at_7_l136_136547


namespace race_runners_l136_136294

theorem race_runners (n : ℕ) (h1 : 5 * 8 + (n - 5) * 10 = 70) : n = 8 :=
sorry

end race_runners_l136_136294


namespace solve_system_of_inequalities_l136_136022

theorem solve_system_of_inequalities:
  (∀ x : ℝ, (x > 1) → (x < 2) → 
  ((x-1) * real.log 2 + real.log (2^(x+1) + 1) < real.log (7 * 2^x + 12)) ∧ 
  (real.log (x+2) / real.log x > 2)) := sorry

end solve_system_of_inequalities_l136_136022


namespace count_five_digit_odd_digits_divisible_by_5_l136_136693

def isFiveDigit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999
def hasOnlyOddDigits (n : ℕ) : Prop := 
  let digits := [n / 10000 % 10, n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  ∀ d ∈ digits, d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9
def isDivisibleBy5 (n : ℕ) : Prop := n % 5 = 0

theorem count_five_digit_odd_digits_divisible_by_5 :
  {n : ℕ | isFiveDigit n ∧ hasOnlyOddDigits n ∧ isDivisibleBy5 n}.card = 625 :=
by
  sorry

end count_five_digit_odd_digits_divisible_by_5_l136_136693


namespace xy_square_sum_l136_136282

variable (x y : ℝ)

theorem xy_square_sum : (y + 6 = (x - 3)^2) →
                        (x + 6 = (y - 3)^2) →
                        (x ≠ y) →
                        x^2 + y^2 = 43 :=
by
  intros h₁ h₂ h₃
  sorry

end xy_square_sum_l136_136282


namespace distinct_ordered_pairs_sum_of_reciprocals_l136_136694

theorem distinct_ordered_pairs_sum_of_reciprocals :
  { (m, n) : ℕ × ℕ // 0 < m ∧ 0 < n ∧ (1 / m : ℚ) + (1 / n) = 1 / 5 }.to_finset.card = 3 :=
sorry

end distinct_ordered_pairs_sum_of_reciprocals_l136_136694


namespace no_integer_roots_l136_136555

theorem no_integer_roots : ∀ x : ℤ, x^3 - 4 * x^2 - 11 * x + 20 ≠ 0 := 
by
  sorry

end no_integer_roots_l136_136555


namespace distribution_centers_count_l136_136117

theorem distribution_centers_count (n : ℕ) (h : n = 5) : n + (n * (n - 1)) / 2 = 15 :=
by
  subst h -- replace n with 5
  show 5 + (5 * (5 - 1)) / 2 = 15
  have : (5 * 4) / 2 = 10 := by norm_num
  show 5 + 10 = 15
  norm_num

end distribution_centers_count_l136_136117


namespace find_complete_sets_l136_136353

noncomputable def is_complete_set (A : Set ℝ) : Prop :=
  A.nonempty ∧ ∀ a b : ℝ, a + b ∈ A → a * b ∈ A

theorem find_complete_sets (A : Set ℝ) :
  is_complete_set A → A = Set.univ :=
begin
  sorry
end

end find_complete_sets_l136_136353


namespace plate_acceleration_l136_136072

theorem plate_acceleration (R r : ℝ) (m : ℝ) (α : ℝ) (g : ℝ) (hR : R = 1.25) (hr : r = 0.75) (hm : m = 100) (hα : α = Real.arccos 0.92) (hg : g = 10) : 
  let a := g * Real.sin(α / 2) in
  a = 2 :=
by
  -- Declaration of given data
  have hR : R = 1.25 := hR,
  have hr : r = 0.75 := hr,
  have hm : m = 100 := hm,
  have hα : α = Real.arccos 0.92 := hα,
  have hg : g = 10 := hg,
  
  -- Calculate acceleration
  
  sorry

end plate_acceleration_l136_136072


namespace hotel_bill_amount_l136_136104

-- Definition of the variables used in the conditions
def each_paid : ℝ := 124.11
def friends : ℕ := 9

-- The Lean 4 theorem statement
theorem hotel_bill_amount :
  friends * each_paid = 1116.99 := sorry

end hotel_bill_amount_l136_136104


namespace shortest_distance_A_to_C_l136_136007

-- Defining the circle parameters
def circle_radius : ℝ := 5
def circle_circumference : ℝ := 2 * Real.pi * circle_radius
def arc_length_AB : ℝ := circle_circumference / 2

-- Defining the points and their positions as per conditions
axiom point_A_on_circle : Prop
axiom point_B_on_circle : Prop
axiom arc_length_A_to_B : arc_length_AB = circle_circumference / 2

-- Defining point C on the diameter
axiom point_C_on_diameter_through_A : Prop
axiom distance_BC : ℝ := 5

-- The theorem to state that the shortest distance from A to C is 10 units
theorem shortest_distance_A_to_C : (dist A C) = 10 := by
  sorry

end shortest_distance_A_to_C_l136_136007


namespace pairs_of_natural_numbers_l136_136538

theorem pairs_of_natural_numbers (a b : ℕ) (h₁ : b ∣ a + 1) (h₂ : a ∣ b + 1) :
    (a = 1 ∧ b = 1) ∨ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 3) ∨ (a = 2 ∧ b = 1) ∨ (a = 3 ∧ b = 2) :=
by {
  sorry
}

end pairs_of_natural_numbers_l136_136538


namespace congruence_equivalence_l136_136010

theorem congruence_equivalence (m n a b : ℤ) (h_coprime : Int.gcd m n = 1) :
  a ≡ b [ZMOD m * n] ↔ (a ≡ b [ZMOD m] ∧ a ≡ b [ZMOD n]) :=
sorry

end congruence_equivalence_l136_136010


namespace min_value_ineq_l136_136608

theorem min_value_ineq (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 3) : 
  (1 / a) + (4 / b) ≥ 3 :=
sorry

end min_value_ineq_l136_136608


namespace population_equal_in_18_years_l136_136909

theorem population_equal_in_18_years : 
  ∀ (n : ℕ), 78000 - 1200 * n = 42000 + 800 * n → n = 18 :=
by
  assume n h,
  sorry

end population_equal_in_18_years_l136_136909


namespace total_ways_to_buy_l136_136762

-- Definitions:
def h : ℕ := 9  -- number of headphones
def m : ℕ := 13 -- number of mice
def k : ℕ := 5  -- number of keyboards
def s_km : ℕ := 4 -- number of sets of keyboard and mouse
def s_hm : ℕ := 5 -- number of sets of headphones and mouse

-- Theorem stating the number of ways to buy three items is 646.
theorem total_ways_to_buy : (s_km * h) + (s_hm * k) + (h * m * k) = 646 := by
  -- Placeholder proof using sorry.
  sorry

end total_ways_to_buy_l136_136762


namespace total_games_in_season_l136_136419

theorem total_games_in_season {n : ℕ} {k : ℕ} (h1 : n = 25) (h2 : k = 15) :
  (n * (n - 1) / 2) * k = 4500 :=
by
  sorry

end total_games_in_season_l136_136419


namespace function_symmetry_extremum_l136_136284

noncomputable def f (x θ : ℝ) : ℝ := 3 * Real.cos (Real.pi * x + θ)

theorem function_symmetry_extremum {θ : ℝ} (H : ∀ x : ℝ, f x θ = f (2 - x) θ) : 
  f 1 θ = 3 ∨ f 1 θ = -3 :=
by
  sorry

end function_symmetry_extremum_l136_136284


namespace incorrect_koala_l136_136504

-- Definitions based on the problem statement
def total_koalas := 2021

def koalas (n : ℕ) : Type
:= {i // 1 ≤ i ∧ i ≤ total_koalas}

def koala_color := ℕ → ℕ -- where colors are represented with ℕ values: red, white, or blue; assume 0, 1, and 2 respectively

variable (colors : koala_color)

-- Given conditions
axiom koalas_cycle_3_colors : ∀ n : ℕ, colors n ≠ colors (n + 1) ∧ colors n ≠ colors (n + 2) ∧ colors (n + 1) ≠ colors (n + 2)

axiom sheilas_guess :
  colors 2 = 1 ∧ -- Koala 2 is white
  colors 20 = 2 ∧ -- Koala 20 is blue
  colors 202 = 0 ∧ -- Koala 202 is red
  colors 1002 = 2 ∧ -- Koala 1002 is blue
  colors 2021 = 1 -- Koala 2021 is white

axiom only_one_wrong : ∃! i, (colors 2 = 1 ∨ colors 20 = 2 ∨ colors 202 = 0 ∨ colors 1002 = 2 ∨ colors 2021 = 1)

-- Proof statement
theorem incorrect_koala : ∃ i, i = 20 ∧ ¬(colors i = 2):
  sorry

end incorrect_koala_l136_136504


namespace range_slope_midpoint_l136_136235

theorem range_slope_midpoint {P Q M : Type} (hP : P ∈ { (x, y) | x + 2 * y - 1 = 0})
    (hQ : Q ∈ { (x, y) | x + 2 * y + 3 = 0})
    (hM_mid : ∃ (x₀ y₀ : ℝ), M = (x₀, y₀) ∧ 
              (P = (p1, p2) → Q = (q1, q2) → p1 + q1 = 2 * x₀ ∧ p2 + q2 = 2 * y₀))
    (ineq : ∃ (x₀ y₀ : ℝ), M = (x₀, y₀) ∧ y₀ > 2 * x₀ + 1) :
    ∃ k, k = y₀ / x₀ ∧ k ∈ Ioo (-1/2 : ℝ) (1/3 : ℝ) :=
sorry

end range_slope_midpoint_l136_136235


namespace triangle_area_l136_136191

def distance (p1 : ℝ × ℝ × ℝ) (p2 : ℝ × ℝ × ℝ) : ℝ := 
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

def herons_formula (a b c : ℝ) : ℝ :=
  let s := semi_perimeter a b c in
  real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area :
  let A := (1, 8, 11)
  let B := (0, 6, 7)
  let C := (-3, 10, 7)
  let AB := distance A B
  let BC := distance B C
  let AC := distance A C
  let area := herons_formula AB BC AC
  area = <computed_exact_area> :=
by
  sorry

end triangle_area_l136_136191


namespace problem_a_satisfied_problem_b_satisfied_l136_136376

noncomputable def solve_equation_one (a : Real) : Prop :=
  ∃ x : Real, x ^ 4 + x ^ 3 - 3 * a ^ 2 * x ^ 2 - 2 * a ^ 2 * x + 2 * a ^ 4 = 0 ∧
  (x = sqrt 2 * a ∨ x = -sqrt 2 * a ∨ x = (-1 + sqrt (1 + 4 * a ^ 2)) / 2 ∨ x = (-1 - sqrt (1 + 4 * a ^ 2)) / 2)

noncomputable def solve_equation_two (a : Real) : Prop :=
  ∃ x : Real, x ^ 3 - 3 * x = a ^ 3 + a⁻³ ∧
  ((a ≠ 0 ∧ a ≠ 1 ∧ a ≠ -1 ∧ x = a + a⁻¹) ∨
  (a = 1 ∧ (x = 2 ∨ x = -1)) ∨
  (a = -1 ∧ (x = -2 ∨ x = 1)))
  
theorem problem_a_satisfied (a : Real) : solve_equation_one a := sorry

theorem problem_b_satisfied (a : Real) : solve_equation_two a := sorry

end problem_a_satisfied_problem_b_satisfied_l136_136376


namespace prove_ratio_l136_136583

-- Definitions based on the given problem conditions
variables {A B C D P E F : ℝ} -- Variables representing lengths

-- Given conditions
variable hCircleThroughVerticesOfSquare : 
  circle_through_vertices_of_square A B C D

variable hPOnSmallerArcAD : 
  point_on_smaller_arc P A D

variable hEIntersectionPCAD : 
  intersection_of_lines E P C A D

variable hFIntersectionPDAB : 
  intersection_of_lines F P D A B

-- The theorem to prove
theorem prove_ratio (hCircleThroughVerticesOfSquare : circle_through_vertices_of_square A B C D)
                    (hPOnSmallerArcAD : point_on_smaller_arc P A D)
                    (hEIntersectionPCAD : intersection_of_lines E P C A D)
                    (hFIntersectionPDAB : intersection_of_lines F P D A B) :
  (2 / AE) = (1 / AB) + (1 / AF) :=
sorry

end prove_ratio_l136_136583


namespace rotated_shifted_line_equation_l136_136297

theorem rotated_shifted_line_equation :
  ∀ x y : ℝ, (y = x → y = -x + 1) :=
begin
  assume x y,
  intro hyp,
  sorry
end

end rotated_shifted_line_equation_l136_136297


namespace num_ways_to_buy_three_items_l136_136767

-- Defining the conditions based on the problem statement
def num_headphones : ℕ := 9
def num_mice : ℕ := 13
def num_keyboards : ℕ := 5
def num_kb_mouse_sets : ℕ := 4
def num_hp_mouse_sets : ℕ := 5

-- Defining the theorem statement
theorem num_ways_to_buy_three_items :
  (num_kb_mouse_sets * num_headphones) + 
  (num_hp_mouse_sets * num_keyboards) + 
  (num_headphones * num_mice * num_keyboards) = 646 := 
  by
  sorry

end num_ways_to_buy_three_items_l136_136767


namespace buratino_field_of_miracles_l136_136156

theorem buratino_field_of_miracles (gold silver : ℝ) (g b : ℕ) (h : g + b = 7)
  (good_weather_days_ineqs : (1.3 ^ g) * (0.7 ^ b) < 1 ∧ (1.2 ^ g) * (0.8 ^ b) > 1) : g = 4 :=
sorry

end buratino_field_of_miracles_l136_136156


namespace probability_three_fair_coins_l136_136438

noncomputable def probability_one_head_two_tails (n : ℕ) : ℚ :=
  if n = 3 then 3 / 8 else 0

theorem probability_three_fair_coins :
  probability_one_head_two_tails 3 = 3 / 8 :=
by
  sorry

end probability_three_fair_coins_l136_136438


namespace johns_average_speed_l136_136795

def start_time := 8 * 60 + 15  -- 8:15 a.m. in minutes
def end_time := 14 * 60 + 45   -- 2:45 p.m. in minutes
def break_start := 12 * 60     -- 12:00 p.m. in minutes
def break_duration := 30       -- 30 minutes
def total_distance := 240      -- Total distance in miles

def total_driving_time : ℕ := 
  (break_start - start_time) + (end_time - (break_start + break_duration))

def average_speed (distance : ℕ) (time : ℕ) : ℕ :=
  distance / (time / 60)  -- converting time from minutes to hours

theorem johns_average_speed :
  average_speed total_distance total_driving_time = 40 :=
by
  sorry

end johns_average_speed_l136_136795


namespace average_book_width_l136_136311

def book_widths : List ℝ := [7, 0.75, 1.5, 3, 0.5, 12, 4.25]

def total_sum (widths : List ℝ) : ℝ := widths.sum

def num_books (widths : List ℝ) : ℕ := widths.length

def average_width (widths : List ℝ) : ℝ :=
  total_sum widths / num_books widths

theorem average_book_width :
  average_width book_widths = 4.071428571428571 do
begin
  sorry
end

end average_book_width_l136_136311


namespace total_sections_after_admissions_l136_136046

theorem total_sections_after_admissions (S : ℕ) (h1 : (S * 24 + 24 = (S + 3) * 21)) :
  (S + 3) = 16 :=
  sorry

end total_sections_after_admissions_l136_136046


namespace avg_one_fourth_class_l136_136390

variable (N : ℕ) (A : ℕ)
variable (h1 : ((N : ℝ) * 80) = (N / 4) * A + (3 * N / 4) * 76)

theorem avg_one_fourth_class : A = 92 :=
by
  sorry

end avg_one_fourth_class_l136_136390


namespace pyramid_surface_area_l136_136812

def isosceles_triangle_area (a b : ℝ) : ℝ := 
  let h := sqrt (b ^ 2 - (a / 2) ^ 2)
  in (1 / 2) * a * h

def tetrahedron_surface_area (a b : ℝ) : ℝ :=
  4 * isosceles_triangle_area a b

theorem pyramid_surface_area {a b : ℝ} 
  (DABC_edge_length : ∀ {e : ℝ}, e ∈ {a, b} → e = 20 ∨ e = 45)
  (not_equilateral : ∀ Δ : Type, ¬(a = 20 ∧ b = 20 ∧ a = b)) :
  tetrahedron_surface_area 20 45 = 40 * sqrt (1925) := 
  by
  sorry

end pyramid_surface_area_l136_136812


namespace measure_of_angle_F_l136_136902

theorem measure_of_angle_F (x : ℝ) (h1 : ∠D = ∠E) (h2 : ∠F = x + 40) (h3 : 2 * x + (x + 40) = 180) : 
  ∠F = 86.67 :=
by 
  sorry

end measure_of_angle_F_l136_136902


namespace length_AB_of_parallelogram_l136_136057

theorem length_AB_of_parallelogram
  (AD BC : ℝ) (AB CD : ℝ) 
  (h1 : AD = 5) 
  (h2 : BC = 5) 
  (h3 : AB = CD)
  (h4 : AD + BC + AB + CD = 14) : 
  AB = 2 :=
by
  sorry

end length_AB_of_parallelogram_l136_136057


namespace min_value_expression_l136_136228

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a + 1 / b) * (b + 4 / a) ≥ 9 :=
by
  sorry

end min_value_expression_l136_136228


namespace _l136_136712

-- Define what it means for sides of a quadrilateral to be equal and parallel
def equal_and_parallel (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] (a b c d : ℝ) :=
  a = c ∧ is_parallel b d

-- Define a quadrilateral
structure Quadrilateral (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)
  (side4 : ℝ)

-- Prove the incorrect definition assertion
noncomputable def incorrect_definition (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] (q : Quadrilateral A B C D) : Prop :=
  ∃ (a b c d : ℝ), equal_and_parallel A B C D a b q.side3 q.side4 ∧ ¬ (is_parallel q.side1 q.side2)

noncomputable def main_theorem : Prop :=
  ∃ (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] 
  (q : Quadrilateral A B C D), incorrect_definition A B C D q

end _l136_136712


namespace coin_flip_difference_l136_136955

/-- The positive difference between the probability of a fair coin landing heads up
exactly 4 times out of 5 flips and the probability of a fair coin landing heads up
5 times out of 5 flips is 1/8. -/
theorem coin_flip_difference :
  (5 * (1 / 2) ^ 5) - ((1 / 2) ^ 5) = (1 / 8) :=
by
  sorry

end coin_flip_difference_l136_136955


namespace product_invariant_l136_136442

theorem product_invariant (n : ℕ) (a b : ℝ) (h_a : a = 1) (h_b : b = 2) 
  (arith_mean : ℝ := (a + b) / 2) (harm_mean : ℝ := 2 * a * b / (a + b)) :
  ∀ n, a * b = 2 :=
begin
  intros n,
  cases n with m,
  { simp [h_a, h_b], },
  { have h1: a * b = 2 := by simp [h_a, h_b],
    induction m with k ih,
    { simp only [arith_mean, harm_mean, h_a, h_b, mul_assoc, mul_inv_cancel, mul_comm, eq_self_iff_true],
      linarith, },
    { specialize ih,
      have h2: arith_mean * harm_mean = 2 := by
        simp [arith_mean, harm_mean, mul_comm, ←mul_assoc, mul_inv_cancel, eq_self_iff_true],
      exact h2 }
  }
end

end product_invariant_l136_136442


namespace point_concyclic_l136_136783

open Set Classical

noncomputable section

variables {α : Type*} [euclidean_geometry α]

-- Definitions for the geometric points and properties
variable {A B C D E F G K : α}

-- Conditions
variable (h_inscribed : inscribed_quadrilateral A B C D)
variable (h_intersect1 : ∃ E, line A B ∩ line D C = {E})
variable (h_intersect2 : ∃ F, line A D ∩ line B C = {F})
variable (h_midpoint : midpoint E F G)
variable (h_intersect3 : ∃ K, point_on_circle_intersect A G)

-- Goal
theorem point_concyclic 
  (h_inscribed : inscribed_quadrilateral A B C D)
  (h_intersect1 : ∃ E, line A B ∩ line D C = {E})
  (h_intersect2 : ∃ F, line A D ∩ line B C = {F})
  (h_midpoint : midpoint E F G)
  (h_intersect3 : ∃ K, point_on_circle_intersect A G):
  concyclic {C, K, F, E} :=
sorry

end point_concyclic_l136_136783


namespace transform_graph_l136_136896

theorem transform_graph :
  ∀ x : ℝ, (λ x, sqrt 2 * sin x) x = (λ x, sqrt 2 * cos (2 * (x / 2 - π / 8))) (2 * (x / 2 - π / 8)) :=
by
  intro x
  sorry

end transform_graph_l136_136896


namespace price_of_second_set_of_knives_l136_136796

def john_visits_houses_per_day : ℕ := 50
def percent_buying_per_day : ℝ := 0.20
def price_first_set : ℝ := 50
def weekly_sales : ℝ := 5000
def work_days_per_week : ℕ := 5

theorem price_of_second_set_of_knives
  (john_visits_houses_per_day : ℕ)
  (percent_buying_per_day : ℝ)
  (price_first_set : ℝ)
  (weekly_sales : ℝ)
  (work_days_per_week : ℕ) :
  0 < percent_buying_per_day ∧ percent_buying_per_day ≤ 1 ∧
  weekly_sales = 5000 ∧ 
  work_days_per_week = 5 ∧
  john_visits_houses_per_day = 50 ∧
  price_first_set = 50 → 
  (∃ price_second_set : ℝ, price_second_set = 150) :=
  sorry

end price_of_second_set_of_knives_l136_136796


namespace simplify_expression_l136_136373

theorem simplify_expression (x : ℝ) : 
  (3*x - 4)*(2*x + 9) - (x + 6)*(3*x + 2) = 3*x^2 - x - 48 :=
by
  sorry

end simplify_expression_l136_136373


namespace value_of_f_pi_over_12_max_min_values_of_f_on_interval_l136_136254

open Real

noncomputable def f (x : ℝ) : ℝ :=
  (sin x)^2 + sqrt 3 * sin x * sin (x + π / 2)

theorem value_of_f_pi_over_12 :
  f (π / 12) = 1 / 2 := sorry

theorem max_min_values_of_f_on_interval :
  ∀ x ∈ Icc (0 : ℝ) (π / 2), 
  1 / 2 ≤ f x ∧ f x ≤ 3 / 2 :=
begin
  intro x,
  intro hx,
  split,
  { sorry }, -- Proof for minimum value
  { sorry }  -- Proof for maximum value
end

end value_of_f_pi_over_12_max_min_values_of_f_on_interval_l136_136254


namespace imaginary_part_of_z_l136_136869

open Complex

-- Define the complex number z
def z : ℂ := -2 * I * (-1 + sqrt 3 * I)

-- The theorem to prove
theorem imaginary_part_of_z : z.im = 2 :=
by
  -- Proof would go here
  sorry

end imaginary_part_of_z_l136_136869


namespace find_radii_l136_136836

-- Define points and distances
variables (A B C D E : ℝ)
variables (AB BC CD DE : ℝ)
variables (r R : ℝ)

-- Define conditions
def condition1 : AB = 4 := by sorry
def condition2 : BC = 2 := by sorry
def condition3 : DE = 2 := by sorry
def condition4 : CD = 3 := by sorry

-- Define the radius calculations
def radius_omega : ℝ := (9 * real.sqrt 3) / (2 * real.sqrt 17)
def radius_Omega : ℝ := (8 * (real.sqrt 3)) / (real.sqrt 17)

-- The theorem to prove
theorem find_radii :
  ∀ (A B C D E : ℝ), 
    AB = 4 → BC = 2 → CD = 3 → DE = 2 →
    (r = radius_omega) ∧ (R = radius_Omega) :=
by sorry

end find_radii_l136_136836


namespace max_value_of_ten_numbers_l136_136855

theorem max_value_of_ten_numbers (a : Fin 10 → ℕ) 
  (distinct : Function.Injective a) 
  (hmean : (∑ i, a i) = 150) :
  ∃ i, a i = 105 :=
by 
  sorry

end max_value_of_ten_numbers_l136_136855


namespace problem_statement_l136_136781

-- Definitions for the rational function and its properties
def rational_function : ℚ[X] := (X^3 + 2 * X^2 + X) / (X^3 - 2 * X^2 - X + 2)

-- Number of holes, vertical asymptotes, horizontal asymptotes, and oblique asymptotes
def a := 1  -- number of holes
def b := 2  -- number of vertical asymptotes
def c := 0  -- number of horizontal asymptotes
def d := 1  -- number of oblique asymptotes

-- The sum in question
theorem problem_statement : a + 2 * b + 3 * c + 4 * d = 9 :=
by
  -- Replace the implicit values with the computed numbers
  let a := 1
  let b := 2
  let c := 0
  let d := 1
  -- Prove the final expression
  sorry

end problem_statement_l136_136781


namespace smallest_special_number_l136_136078

def all_digits_different (n : ℕ) : Prop :=
  let d := to_digits 10 n in
  d.nodup

def has_digit (n digit : ℕ) : Prop :=
  digit ∈ to_digits 10 n

def divisible_by_digits (n : ℕ) : Prop :=
  ∀ d ∈ to_digits 10 n, n % d = 0

theorem smallest_special_number : ∃ n : ℕ, 
  1000 ≤ n ∧ n < 10000 ∧
  all_digits_different n ∧ 
  has_digit n 5 ∧ 
  divisible_by_digits n ∧ 
  n = 5124 := 
sorry

end smallest_special_number_l136_136078


namespace no_carry_pairs_count_l136_136568

/-- 
  Define a function to check if there is no carrying when adding two consecutive integers x and (x+1).
--/
def no_carrying (x : ℕ) : Prop :=
  let d := x % 10,
      c := (x / 10) % 10,
      b := (x / 100) % 10,
      a := (x / 1000) % 10 in
  d < 9 ∧ c < 9 ∧ b < 9 ∧ a < 9

theorem no_carry_pairs_count : 
  ∃ n : ℕ, n = 156 ∧ ∀ (x : ℕ), 1000 ≤ x ∧ x < 2000 →
  no_carrying x ↔ ∃ m : ℕ, m < 2000 ∧ x + 1 = m := 
begin
  let n := 156,
  use n,
  split,
  { exact rfl },
  { intros x hx,
    split,
    { intro h,
      have h1 : ∃ m : ℕ, x + 1 = m ∧ m < 2000,
        from ⟨x + 1, rfl, nat.add_lt_mul_iff_lt (nat.pred_le (nat.succ x))  
  
        sorry },
    { intro h,
      cases h with m hm,
      unfold no_carrying,
      split,
      { sorry },
      { sorry },
      { sorry },
      { sorry } } }
end

end no_carry_pairs_count_l136_136568


namespace monica_tiling_tiles_count_l136_136362

theorem monica_tiling_tiles_count :
  let length := 24
      width := 18
      border_tile_length := 2
      inner_tile_length := 1
      border_tile_count := 2 * ((length - 2 * border_tile_length) / border_tile_length) + 2 * ((width - 2 * border_tile_length) / border_tile_length) - 4
      inner_tile_area := (width - 2 * border_tile_length) * (length - 2 * border_tile_length)
      inner_tile_count := inner_tile_area / inner_tile_length^2
  in border_tile_count + inner_tile_count = 310 :=
by
  let length := 24
  let width := 18
  let border_tile_length := 2
  let inner_tile_length := 1
  let border_tiles_along_length := (length - 2 * border_tile_length) / border_tile_length
  let border_tiles_along_width := (width - 2 * border_tile_length) / border_tile_length
  let border_tile_count := 2 * border_tiles_along_length + 2 * border_tiles_along_width - 4
  let inner_tile_area := (width - 2 * border_tile_length) * (length - 2 * border_tile_length)
  let inner_tile_count := inner_tile_area / inner_tile_length^2
  calc
    border_tile_count + inner_tile_count
    = 2 * ((length - 2 * border_tile_length) / border_tile_length) + 2 * ((width - 2 * border_tile_length) / border_tile_length) - 4 + inner_tile_area / inner_tile_length^2 : by sorry
    ... = 310 : by sorry

end monica_tiling_tiles_count_l136_136362


namespace store_purchase_ways_l136_136743

theorem store_purchase_ways :
  ∃ (headphones sets_hm sets_km keyboards mice : ℕ), 
  headphones = 9 ∧
  mice = 13 ∧
  keyboards = 5 ∧
  sets_km = 4 ∧
  sets_hm = 5 ∧
  (sets_km * headphones + sets_hm * keyboards + headphones * mice * keyboards = 646) :=
by
  use 9, 5, 4, 5, 13
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  sorry

end store_purchase_ways_l136_136743


namespace constant_term_in_binomial_expansion_l136_136298

noncomputable def sum_binom_coeffs (n : ℕ) : ℕ := 2^n

theorem constant_term_in_binomial_expansion :
  ∀ (n : ℕ), sum_binom_coeffs n = 256 → 
            8 = n ∧ 
            ∃ r : ℕ, 
            (binomial 8 r) * (-2)^r * x^((8 - 4*r) / 3) == 112 := by
  sorry

end constant_term_in_binomial_expansion_l136_136298


namespace symmetry_property_l136_136460

-- Definitions of the functions we are considering
def f_A (x : ℝ) : ℝ := x^2
def f_B (x : ℝ) : ℝ := abs (x^2)
def f_C (x : ℝ) : ℝ := 2 / x
def f_D (x : ℝ) : ℝ := abs (2 / x)

-- Test symmetry about the y-axis
def symmetric_y (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Test symmetry about the origin
def symmetric_origin (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- The theorem we want to prove
theorem symmetry_property :
  symmetric_y f_C ∧ symmetric_origin f_C ∧
  ¬(symmetric_y f_A ∧ symmetric_origin f_A) ∧
  ¬(symmetric_y f_B ∧ symmetric_origin f_B) ∧
  ¬(symmetric_y f_D ∧ symmetric_origin f_D) :=
by
  -- Proof skipped
  sorry

end symmetry_property_l136_136460


namespace promote_cashback_beneficial_cashback_rubles_preferable_l136_136976

-- For the benefit of banks promoting cashback services
theorem promote_cashback_beneficial :
  ∀ (bank : Type) (services : Type), 
  (∃ (customers : set bank) (businesses : set services), 
    (∀ b ∈ businesses, ∃ d ∈ customers, promotes_cashback d b) ∧
    (∀ c ∈ customers, prefers_cashback c b)) :=
sorry

-- For the preference of customers on cashback in rubles
theorem cashback_rubles_preferable :
  ∀ (customer : Type) (cashback : Type),
  (∀ r ∈ cashback, rubles r → prefers_customer r cashback) :=
sorry

end promote_cashback_beneficial_cashback_rubles_preferable_l136_136976


namespace equal_areas_of_hyperbola_and_chords_l136_136799

theorem equal_areas_of_hyperbola_and_chords
  (A B P : ℝ × ℝ)
  (hA : A.1 * A.2 = 1) 
  (hB : B.1 * B.2 = 1) 
  (hP : P.1 * P.2 = 1)
  (h_cond : A.1 > P.1 ∧ P.1 > B.1)
  (h_max_area : ∀ Q, Q.1 > B.1 ∧ Q.1 < A.1 → 
    1/2 * abs (A.1 * (B.2 - P.2) + B.1 * (P.2 - A.2) + P.1 * (A.2 - B.2)) 
    ≥ 1/2 * abs (A.1 * (B.2 - Q.2) + B.1 * (Q.2 - A.2) + Q.1 * (A.2 - B.2))):
  Δ (area_hyperbola_chord A P) = Δ (area_hyperbola_chord P B) :=
sorry

end equal_areas_of_hyperbola_and_chords_l136_136799


namespace circle_center_sum_correct_l136_136539

-- Define the center and radius calculation problem
def circle_center_sum := 5 + 3 + 3 * Real.sqrt 6

theorem circle_center_sum_correct :
  ∀ x y : ℝ, (x - 5)^2 + (y - 3)^2 = 54 → circle_center_sum = 8 + 3 * Real.sqrt 6 := by
  intros x y h
  sorry

end circle_center_sum_correct_l136_136539


namespace sally_initial_peaches_l136_136372

def initial_peaches (picked: ℕ) (total_now: ℕ) : ℕ :=
  total_now - picked

theorem sally_initial_peaches (picked: ℕ) (total_now: ℕ) (initial: ℕ) : initial = initial_peaches picked total_now :=
by
  have picked := 55
  have total_now := 68
  have initial := 13
  exact rfl

end sally_initial_peaches_l136_136372


namespace continuity_of_f_at_3_l136_136315

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
  if x ≤ 3 then 3*x^2 + 2*x - 4 else b*x + 7

theorem continuity_of_f_at_3 (b : ℝ) : 
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 3) < δ → abs (f x b - f 3 b) < ε) ↔ b = 22 / 3 :=
by
  sorry

end continuity_of_f_at_3_l136_136315


namespace proj_b_v_is_correct_l136_136331

open Real

noncomputable def a : Vector ℝ := sorry
noncomputable def b : Vector ℝ := sorry

axiom orthogonal_a_b : a ⬝ b = 0

noncomputable def v : Vector ℝ := Vec2 4 (-2)
noncomputable def proj_a_v : Vector ℝ := Vec2 (-4/5) (-8/5)

axiom proj_a_v_property : proj a v = proj_a_v

theorem proj_b_v_is_correct : proj b v = Vec2 (24/5) (-2/5) :=
sorry

end proj_b_v_is_correct_l136_136331


namespace x_y_z_sum_l136_136270

theorem x_y_z_sum :
  ∃ (x y z : ℕ), (16 / 3)^x * (27 / 25)^y * (5 / 4)^z = 256 ∧ x + y + z = 6 :=
by
  -- Proof can be completed here
  sorry

end x_y_z_sum_l136_136270


namespace total_work_done_l136_136910

theorem total_work_done 
  (D m g l_0 : ℝ) : 
  let n := 10 in
  ( ∑ i in finset.range (n + 1), 
        (1/2) * D * ((m * g / D * i) ^ 2)
    + m * g * ((i - 1) * l_0 + (m * g / D) * ((i - 1) * i / 2))
  ) = 1165 :=
by
  sorry

end total_work_done_l136_136910


namespace smallest_n_sqrt_diff_l136_136082

theorem smallest_n_sqrt_diff : ∃ n : ℕ, (sqrt n - sqrt (n - 1) < 0.02) ∧ (∀ m : ℕ, (m < n) → (sqrt m - sqrt (m - 1) ≥ 0.02)) ∧ n = 626 :=
by
  sorry

end smallest_n_sqrt_diff_l136_136082


namespace store_purchase_ways_l136_136742

theorem store_purchase_ways :
  ∃ (headphones sets_hm sets_km keyboards mice : ℕ), 
  headphones = 9 ∧
  mice = 13 ∧
  keyboards = 5 ∧
  sets_km = 4 ∧
  sets_hm = 5 ∧
  (sets_km * headphones + sets_hm * keyboards + headphones * mice * keyboards = 646) :=
by
  use 9, 5, 4, 5, 13
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  sorry

end store_purchase_ways_l136_136742


namespace minimum_a_for_monotonic_increasing_on_interval_l136_136650

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp x - Real.log x

noncomputable def f_prime (a x : ℝ) : ℝ := a * Real.exp x - 1 / x

def min_value_of_a : ℝ := Real.exp (-1)

theorem minimum_a_for_monotonic_increasing_on_interval :
  (∀ x ∈ Set.Ioo 1 2, 0 ≤ f_prime a x) ↔ a ≥ min_value_of_a :=
by
  sorry

end minimum_a_for_monotonic_increasing_on_interval_l136_136650


namespace open_interval_rationals_l136_136838

-- Defining the problem conditions and final proof statement.
theorem open_interval_rationals (n : ℕ) (h : n > 0) :
  ∀ (I : set ℝ), (exists a : ℝ, I = set.Ioo a (a + 1/n)) →
    (∀ r : ℚ, (∃ p q : ℤ, q > 0 ∧ q ≤ n ∧ r = p / q) → r ∈ I → r ∈ I) →
    ∃ N, N ≤ nat.floor ((n + 1) / 2) :=
begin
  -- Given the conditions and definitions, the proof will follow.
  sorry -- Proof to be provided.
end

end open_interval_rationals_l136_136838


namespace unique_solution_a_eq_4_l136_136710

theorem unique_solution_a_eq_4 (a : ℝ) (h : ∀ x1 x2 : ℝ, (a * x1^2 + a * x1 + 1 = 0 ∧ a * x2^2 + a * x2 + 1 = 0) → x1 = x2) : a = 4 :=
sorry

end unique_solution_a_eq_4_l136_136710


namespace sum_of_real_roots_eq_sqrt3_l136_136565

theorem sum_of_real_roots_eq_sqrt3 :
  let f : ℝ → ℝ := λ x, x^4 - 6 * x - 3 
  in (∑ x in (Finset.filter is_real_root (f.roots)), x) = real.sqrt 3 := 
sorry

end sum_of_real_roots_eq_sqrt3_l136_136565


namespace fair_coin_difference_l136_136931

noncomputable def binom_coeff (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem fair_coin_difference :
  let p := (1 : ℚ) / 2 in
  let p_1 := binom_coeff 5 4 * p^4 * (1 - p) in
  let p_2 := p^5 in
  p_1 - p_2 = 1 / 8 :=
by
  sorry

end fair_coin_difference_l136_136931


namespace length_of_segment_CC_l136_136899

open Real

def point (x y : ℝ) := (x, y)

def reflect_over_y_axis (P : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := P
  (-x, y)

def distance (P Q : ℝ × ℝ) : ℝ :=
  let (x1, y1) := P
  let (x2, y2) := Q
  sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem length_of_segment_CC' : 
  let C := point (-3) 2
  let C' := reflect_over_y_axis C
  distance C C' = 6 :=
by
  sorry

end length_of_segment_CC_l136_136899


namespace exists_valid_A_with_largest_B_l136_136396

def is_digit (x : ℕ) : Prop := 1 ≤ x ∧ x ≤ 9

def valid_A (A : ℕ) : Prop :=
  ∃ a b : ℕ, is_digit a ∧ is_digit b ∧ A = 700000 + a * 10000 + 6000 + 300 + 10 + b

def B_divisible_by_121 (A : ℕ) : Prop :=
  ∃ a b : ℕ, is_digit a ∧ is_digit b ∧ A = 700000 + a * 10000 + 6000 + 300 + 10 + b ∧
  let B := (a + b + 17) * 111111 in
  B % 121 = 0

theorem exists_valid_A_with_largest_B :
  ∃ (n : ℕ), n = 7 ∧
  ∃ A : ℕ, valid_A A ∧ B_divisible_by_121 A ∧
  ∀ A' : ℕ, valid_A A' ∧ B_divisible_by_121 A' → A' ≤ 796317 :=
sorry

end exists_valid_A_with_largest_B_l136_136396


namespace cube_strictly_increasing_l136_136601

theorem cube_strictly_increasing (a b : ℝ) (h : a > b) : a^3 > b^3 :=
sorry

end cube_strictly_increasing_l136_136601


namespace triangle_value_a_l136_136707

theorem triangle_value_a (a : ℕ) (h1: a + 2 > 6) (h2: a + 6 > 2) (h3: 2 + 6 > a) : a = 7 :=
sorry

end triangle_value_a_l136_136707


namespace g_at_5_l136_136338

def g (x : ℝ) : ℝ := 3*x^4 - 20*x^3 + 35*x^2 - 40*x + 24

theorem g_at_5 : g 5 = 74 := 
by {
  sorry
}

end g_at_5_l136_136338


namespace period_of_f_l136_136919

def f (x : ℝ) : ℝ := sin x - cos x

theorem period_of_f : ∀ x : ℝ, f (x + 2 * π) = f x :=
by
  simp [f, sin_add, cos_add, sin_eq_sin_sub_cos, cos_eq_sin_add_cos]
  sorry

end period_of_f_l136_136919


namespace minimum_value_of_a_l136_136624

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - Real.log x

theorem minimum_value_of_a (a : ℝ) :
  (∀ x ∈ Ioo 1 2, deriv (f a) x ≥ 0) → a ≥ Real.exp ⁻¹ :=
by
  simp [f]
  sorry

end minimum_value_of_a_l136_136624


namespace minimum_value_of_a_l136_136628

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - Real.log x

theorem minimum_value_of_a (a : ℝ) :
  (∀ x ∈ Ioo 1 2, deriv (f a) x ≥ 0) → a ≥ Real.exp ⁻¹ :=
by
  simp [f]
  sorry

end minimum_value_of_a_l136_136628


namespace triangle_evaluation_l136_136534

def triangle (P Q : ℝ) : ℝ := (P + Q) / 3

theorem triangle_evaluation : triangle 3 (triangle 6 9) = 8 / 3 :=
by
  sorry

end triangle_evaluation_l136_136534


namespace scaling_factor_is_2_l136_136120

-- Define the volumes of the original and scaled cubes
def V1 : ℕ := 343
def V2 : ℕ := 2744

-- Assume s1 cubed equals V1 and s2 cubed equals V2
def s1 : ℕ := 7  -- because 7^3 = 343
def s2 : ℕ := 14 -- because 14^3 = 2744

-- Scaling factor between the cubes
def scaling_factor : ℕ := s2 / s1 

-- The theorem stating the scaling factor is 2 given the volumes
theorem scaling_factor_is_2 (h1 : s1 ^ 3 = V1) (h2 : s2 ^ 3 = V2) : scaling_factor = 2 := by
  sorry

end scaling_factor_is_2_l136_136120


namespace positive_difference_between_probabilities_is_one_eighth_l136_136934

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem positive_difference_between_probabilities_is_one_eighth :
  let p : ℚ := 1 / 2
    n : ℕ := 5
    prob_4_heads := binomial_probability n 4 p
    prob_5_heads := binomial_probability n 5 p
  in |prob_4_heads - prob_5_heads| = 1 / 8 :=
by
  let p : ℚ := 1 / 2
  let n : ℕ := 5
  let prob_4_heads := binomial_probability n 4 p
  let prob_5_heads := binomial_probability n 5 p
  sorry

end positive_difference_between_probabilities_is_one_eighth_l136_136934


namespace trig_identity_30deg_l136_136164

theorem trig_identity_30deg :
  let t30 := Real.tan (Real.pi / 6)
  let s30 := Real.sin (Real.pi / 6)
  let c30 := Real.cos (Real.pi / 6)
  t30 = (Real.sqrt 3) / 3 ∧ s30 = 1 / 2 ∧ c30 = (Real.sqrt 3) / 2 →
  t30 + 4 * s30 + 2 * c30 = (2 * (Real.sqrt 3) + 3) / 3 := 
by
  intros
  sorry

end trig_identity_30deg_l136_136164


namespace solution_set_of_inequality_l136_136056

theorem solution_set_of_inequality (x : ℝ) :
  (abs x * (x - 1) ≥ 0) ↔ (x ≥ 1 ∨ x = 0) := 
by
  sorry

end solution_set_of_inequality_l136_136056


namespace maximum_value_expression_l136_136819

noncomputable def max_expr_value (x θ : ℝ) (h1 : 0 < x) (h2 : 0 ≤ θ) (h3 : θ ≤ π) : ℝ :=
  2 * Real.sqrt 2 - 2

theorem maximum_value_expression :
  ∀ (x θ : ℝ), 0 < x → 0 ≤ θ → θ ≤ π →
  ( ∃ (M : ℝ), (∀ (x' θ' : ℝ), 0 < x' → 0 ≤ θ' → θ' ≤ π → 
    (x' ^ 2 + 2 - Real.sqrt (x' ^ 4 + 4 * (Real.sin θ') ^ 2)) / x' ≤ M)
     ∧ M = 2 * Real.sqrt 2 - 2) :=
by 
  intros x θ h1 h2 h3
  use max_expr_value x θ h1 h2 h3
  split
  · sorry  -- Prove that the expression (x^2 + 2 - sqrt(x^4 + 4*sin^2 θ)) / x has an upper bound M
  · refl   -- Prove M = 2 * sqrt 2 - 2

end maximum_value_expression_l136_136819


namespace triangle_min_value_l136_136715

theorem triangle_min_value (A B C M N E F G : Type) 
  (hM : M ∈ line_segment A B) 
  (hN : N ∈ line_segment A C) 
  (hAM : vector AM = (1/2) * vector AB) 
  (hAN : vector AN = (1/3) * vector AC) 
  (hE : E ∈ line_segment B C) 
  (hF : F ∈ line_segment B C) 
  (hG : G ∈ line_segment B C) 
  (hVec : vector AE + vector AF + vector AG = x * vector AM + y * vector AN) : 
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (2 / x + 3 / y) = (4 / 3) :=
begin
  sorry,
end

end triangle_min_value_l136_136715


namespace all_points_equal_l136_136178

-- Define the problem conditions and variables
variable (P : Type) -- points in the plane
variable [MetricSpace P] -- the plane is a metric space
variable (f : P → ℝ) -- assignment of numbers to points
variable (incenter : P → P → P → P) -- calculates incenter of a nondegenerate triangle

-- Condition: the value at the incenter of a triangle is the arithmetic mean of the values at the vertices
axiom incenter_mean_property : ∀ (A B C : P), 
  (A ≠ B) → (B ≠ C) → (A ≠ C) →
  f (incenter A B C) = (f A + f B + f C) / 3

-- The theorem to be proved
theorem all_points_equal : ∀ x y : P, f x = f y :=
by
  sorry

end all_points_equal_l136_136178


namespace complex_matrix_determinant_l136_136536

theorem complex_matrix_determinant (z : ℂ) (h : (z * (1 : ℂ) - (1 + complex.I) * (1 - complex.I) = complex.I)) : 
  z = 2 + complex.I :=
by
  sorry

end complex_matrix_determinant_l136_136536


namespace tank_fill_time_with_leak_l136_136835

def fill_with_leak_time (fill_time : ℕ) (leak_time : ℕ) : ℕ :=
  let fill_rate := 1 / (fill_time : ℚ)
  let leak_rate := 1 / (leak_time : ℚ)
  let combined_rate := fill_rate - leak_rate
  if combined_rate ≠ 0 then (1 / combined_rate).toNat else 0

theorem tank_fill_time_with_leak :
  fill_with_leak_time 12 30 = 20 := 
by
  sorry

end tank_fill_time_with_leak_l136_136835


namespace total_ways_to_buy_l136_136763

-- Definitions:
def h : ℕ := 9  -- number of headphones
def m : ℕ := 13 -- number of mice
def k : ℕ := 5  -- number of keyboards
def s_km : ℕ := 4 -- number of sets of keyboard and mouse
def s_hm : ℕ := 5 -- number of sets of headphones and mouse

-- Theorem stating the number of ways to buy three items is 646.
theorem total_ways_to_buy : (s_km * h) + (s_hm * k) + (h * m * k) = 646 := by
  -- Placeholder proof using sorry.
  sorry

end total_ways_to_buy_l136_136763


namespace tan_subtraction_formula_l136_136279

noncomputable def tan_sub (alpha beta : ℝ) : ℝ := (tan alpha - tan beta) / (1 + tan alpha * tan beta)

theorem tan_subtraction_formula (alpha beta : ℝ) (h1 : tan alpha = 9) (h2 : tan beta = 5) : tan_sub alpha beta = 2 / 23 :=
by
  simp [tan_sub, h1, h2]
  sorry

end tan_subtraction_formula_l136_136279


namespace find_a10_l136_136221

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

axiom sum_first_n_terms : ∀ n m, S n + S m = S (n + m)
axiom a1_eq : a 1 = 1
axiom Sn_definition : ∀ n, a (n + 1) = S (n + 1) - S n

theorem find_a10 : a 10 = 1 :=
by
  have S10 := sum_first_n_terms 1 9
  have S1_eq_a1 : S 1 = a 1 := by sorry
  have a9_eq_S9_S8 : a 9 = S 9 - S 8 := by sorry
  rw [S1_eq_a1, a1_eq] at S10
  sorry

end find_a10_l136_136221


namespace mutual_independence_of_A_and_D_l136_136431

noncomputable theory

variables (Ω : Type) [ProbabilitySpace Ω]
-- Definition of events A, B, C, D as sets over Ω
def event_A : Event Ω := {ω | some_condition_for_A}
def event_B : Event Ω := {ω | some_condition_for_B}
def event_C : Event Ω := {ω | some_condition_for_C}
def event_D : Event Ω := {ω | some_condition_for_D}

-- Given probabilities
axiom P_A : P(event_A Ω) = 1 / 6
axiom P_B : P(event_B Ω) = 1 / 6
axiom P_C : P(event_C Ω) = 5 / 36
axiom P_D : P(event_D Ω) = 1 / 6

-- Independence definition
def are_independent (X Y : Event Ω) : Prop :=
  P(X ∩ Y) = P(X) * P(Y)

-- The problem statement: proving A and D are independent
theorem mutual_independence_of_A_and_D : are_independent Ω (event_A Ω) (event_D Ω) :=
sorry

end mutual_independence_of_A_and_D_l136_136431


namespace divisor_of_p_l136_136817

-- Define the necessary variables and assumptions
variables (p q r s : ℕ)

-- State the conditions
def conditions := gcd p q = 28 ∧ gcd q r = 45 ∧ gcd r s = 63 ∧ 80 < gcd s p ∧ gcd s p < 120 

-- State the proposition to prove: 11 divides p
theorem divisor_of_p (h : conditions p q r s) : 11 ∣ p := 
sorry

end divisor_of_p_l136_136817


namespace shop_combinations_l136_136747

theorem shop_combinations :
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let keyboard_and_mouse_sets := 4
  let headphone_and_mouse_sets := 5
  (keyboard_and_mouse_sets * headphones) + (headphone_and_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 :=
by
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let keyboard_and_mouse_sets := 4
  let headphone_and_mouse_sets := 5
  have case1 : keyboard_and_mouse_sets * headphones = 4 * 9, by sorry
  have case2 : headphone_and_mouse_sets * keyboards = 5 * 5, by sorry
  have case3 : headphones * mice * keyboards = 9 * 13 * 5, by sorry
  show (keyboard_and_mouse_sets * headphones) + (headphone_and_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646,
  by rw [case1, case2, case3]; exact sorry

end shop_combinations_l136_136747


namespace abs_sum_condition_l136_136704

theorem abs_sum_condition (a b : ℝ) (h1 : |a| = 7) (h2 : |b| = 3) (h3 : a * b > 0) : a + b = 10 ∨ a + b = -10 :=
by { sorry }

end abs_sum_condition_l136_136704


namespace expand_polynomial_l136_136183

theorem expand_polynomial :
  (∀ (x : ℝ), (x^10 - 4 * x^3 + 2 * x^(-1) - 9) * (-3 * x^5) = -3 * x^(15) + 12 * x^8 + 27 * x^5 - 6 * x^4) :=
by
  sorry

end expand_polynomial_l136_136183


namespace total_games_in_season_l136_136421

theorem total_games_in_season {n : ℕ} {k : ℕ} (h1 : n = 25) (h2 : k = 15) :
  (n * (n - 1) / 2) * k = 4500 :=
by
  sorry

end total_games_in_season_l136_136421


namespace min_value_of_a_l136_136631

theorem min_value_of_a (a : ℝ) : 
  (∀ x ∈ Ioo 1 2, (a * exp x - 1/x) ≥ 0) → a = exp (-1) :=
by
  sorry

end min_value_of_a_l136_136631


namespace ratio_michelle_katie_l136_136897

-- Define the variables based on the problem conditions
variables (T M K : ℕ)

-- Conditions provided in the problem
def tracy_drives_more : Prop := T = 2 * M + 20
def michelle_drives : Prop := M = 294
def total_distance : Prop := T + M + K = 1000

-- The statement that needs to be proven (as given in the question)
theorem ratio_michelle_katie (T M K : ℕ) 
  (h1 : tracy_drives_more T M K)
  (h2 : michelle_drives M)
  (h3 : total_distance T M K) :
  M / K = 3 / 1 := 
sorry

end ratio_michelle_katie_l136_136897


namespace remainder_5x_div_9_l136_136467

theorem remainder_5x_div_9 {x : ℕ} (h : x % 9 = 5) : (5 * x) % 9 = 7 :=
sorry

end remainder_5x_div_9_l136_136467


namespace limit_f_at_negative_three_limit_phi_at_six_l136_136196

noncomputable def f (x : ℝ) : ℝ := x^3 - 5 * x^2 + 2 * x + 4

noncomputable def phi (t : ℝ) : ℝ := t * real.sqrt (t^2 - 20) - real.log10 (t + real.sqrt (t^2 - 20))

theorem limit_f_at_negative_three : real.limit (λ x : ℝ, f x) (-3) = -74 := sorry

theorem limit_phi_at_six : real.limit (λ t : ℝ, phi t) 6 = 23 := sorry

end limit_f_at_negative_three_limit_phi_at_six_l136_136196


namespace measure_of_angle_F_l136_136900

theorem measure_of_angle_F {D E F : ℝ}
  (isosceles : D = E)
  (angle_F_condition : F = D + 40)
  (sum_of_angles : D + E + F = 180) :
  F = 260 / 3 :=
by
  sorry

end measure_of_angle_F_l136_136900


namespace circleII_area_l136_136524

noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r^2

theorem circleII_area (r₁ : ℝ) (h₁ : area_of_circle r₁ = 9) (h₂ : r₂ = 3 * 2 * r₁) : 
  area_of_circle r₂ = 324 :=
by
  sorry

end circleII_area_l136_136524


namespace picky_elephants_prefer_round_l136_136979

-- Definitions based on conditions
def Elephants := ℕ  -- number of elephants
def Hippos := ℕ  -- number of hippos
def RoundWatermelons := ℕ  -- number of round watermelons
def CubicWatermelons := ℕ  -- number of cubic watermelons

def group1_elephants : Elephants := 5
def group1_hippos : Hippos := 7
def group1_round_watermelons : RoundWatermelons := 11
def group1_cubic_watermelons : CubicWatermelons := 20

def group2_elephants : Elephants := 8
def group2_hippos : Hippos := 4
def group2_round_watermelons : RoundWatermelons := 20
def group2_cubic_watermelons : CubicWatermelons := 8

def total_watermelons (E : Elephants) (H : Hippos) : ℕ :=
  E * 2 -- each elephant eats 2 watermelons
  + H * 3 -- each hippo eats 3 watermelons

-- Theorem statement
theorem picky_elephants_prefer_round :
  ∀ (E H : Elephants),
  (5 * 2 + 7 * 3 = 31) ∧ (8 * 2 + 4 * 3 = 28) →
  (∃ e : Elephants -> ℕ, ∃ h : Hippos -> ℕ, 
    (e 0 = 2) ∧ (h 0 = 3) ∧ 
    (e 1  + round_watermelons = total_watermelons (e 0) (h 0)) ∧
    (e 1 = round_watermelons) ∧ 
    (h 1 = round_watermelons + cubic_watermelons) ∧
    (total_watermelons 5 7 = 31) ∧
    (total_watermelons 8 4 = 28)) := sorry

end picky_elephants_prefer_round_l136_136979


namespace probability_same_color_is_89_over_169_l136_136719

def total_balls : ℕ := 13
def blue_balls : ℕ := 8
def yellow_balls : ℕ := 5

def prob_blue_twice : ℚ := (blue_balls / total_balls) * (blue_balls / total_balls)
def prob_yellow_twice : ℚ := (yellow_balls / total_balls) * (yellow_balls / total_balls)
def prob_same_color : ℚ := prob_blue_twice + prob_yellow_twice

theorem probability_same_color_is_89_over_169 :
  prob_same_color = 89 / 169 :=
by
  unfold prob_same_color prob_blue_twice prob_yellow_twice blue_balls yellow_balls total_balls
  norm_num
  sorry

end probability_same_color_is_89_over_169_l136_136719


namespace find_p_l136_136880

noncomputable theory

-- Definitions based on the conditions
def binomial_distribution (n : ℕ) (p : ℝ) (x : ℕ) : ℝ := 
  (n.choose x) * (p ^ x) * ((1 - p) ^ (n - x))

variables (n : ℕ) (p : ℝ)

-- Assumptions given in the problem
axiom E_xi : (∑ x in finset.range (n + 1), x * binomial_distribution n p x) = 300
axiom D_xi : (∑ x in finset.range (n + 1), (x - 300) ^ 2 * binomial_distribution n p x) = 200

-- The theorem stating the question and answer equivalence
theorem find_p : p = (1 : ℝ) / 3 :=
sorry

end find_p_l136_136880


namespace intersection_S_T_l136_136257

def S : Set ℝ := { x | (x - 2) * (x - 3) >= 0 }
def T : Set ℝ := { x | x > 0 }

theorem intersection_S_T :
  S ∩ T = { x | (0 < x ∧ x <= 2) ∨ (x >= 3) } := by
  sorry

end intersection_S_T_l136_136257


namespace jogger_ahead_distance_l136_136127

variable (jogger_speed_kph : ℝ) (train_speed_kph : ℝ) (train_length_m : ℝ) (passing_time_s : ℝ)

def jogger_distance_ahead :=
  jogger_speed_kph = 9 ∧
  train_speed_kph = 45 ∧
  train_length_m = 200 ∧
  passing_time_s = 40 →
  let relative_speed_mps := (train_speed_kph - jogger_speed_kph) * (5 / 18) in
  relative_speed_mps * passing_time_s = train_length_m + 200

theorem jogger_ahead_distance 
    (jogger_speed_kph : ℝ)
    (train_speed_kph : ℝ)
    (train_length_m : ℝ)
    (passing_time_s : ℝ)
    (h : jogger_distance_ahead jogger_speed_kph train_speed_kph train_length_m passing_time_s):
    400 - train_length_m = 200 := 
  by
  sorry

end jogger_ahead_distance_l136_136127


namespace total_ways_to_buy_l136_136759

-- Definitions:
def h : ℕ := 9  -- number of headphones
def m : ℕ := 13 -- number of mice
def k : ℕ := 5  -- number of keyboards
def s_km : ℕ := 4 -- number of sets of keyboard and mouse
def s_hm : ℕ := 5 -- number of sets of headphones and mouse

-- Theorem stating the number of ways to buy three items is 646.
theorem total_ways_to_buy : (s_km * h) + (s_hm * k) + (h * m * k) = 646 := by
  -- Placeholder proof using sorry.
  sorry

end total_ways_to_buy_l136_136759


namespace distribution_centers_l136_136119

theorem distribution_centers (n : ℕ) (h : n = 5) : 
  (n + (nat.choose n 2) = 15) :=
by
  rw h
  -- simplifying n and the binomial coefficient for n = 5

  sorry

end distribution_centers_l136_136119


namespace correct_answer_l136_136573

theorem correct_answer (m n : ℤ) (h : 3 * m * n + 3 * m = n + 2) : 3 * m + n = -2 := 
by
  sorry

end correct_answer_l136_136573


namespace geometric_seq_fourth_term_l136_136124

theorem geometric_seq_fourth_term (a r : ℕ) (h1 : a = 3) (h3 : a * r ^ 2 = 75) : a * r ^ 3 = 375 :=
by
  rw [h1] at *
  obtain ⟨rfl : r = 5⟩ := (Nat.sqrt_eq_rfl_iff (Nat.le_add_left a 72)).1 (Nat.eq_of_mul_eq_mul_left (Nat.one_le_of_lt 3 (75/3)) h3)
  sorry

end geometric_seq_fourth_term_l136_136124


namespace glass_sphere_wall_thickness_l136_136489

/-- Mathematically equivalent proof problem statement:
Given a hollow glass sphere with outer diameter 16 cm such that 3/8 of its surface remains dry,
and specific gravity of glass s = 2.523. The wall thickness of the sphere is equal to 0.8 cm. -/
theorem glass_sphere_wall_thickness 
  (outer_diameter : ℝ) (dry_surface_fraction : ℝ) (specific_gravity : ℝ) (required_thickness : ℝ) 
  (uniform_thickness : outer_diameter = 16)
  (dry_surface : dry_surface_fraction = 3 / 8)
  (s : specific_gravity = 2.523) :
  required_thickness = 0.8 :=
by
  sorry

end glass_sphere_wall_thickness_l136_136489


namespace guarantee_game_termination_l136_136100

-- Define the 2-adic valuation (nu_2)
def nu_2 (x : ℕ) : ℕ :=
  (Nat.find_greatest (fun k => 2^k ∣ x) (x + 1))

-- The game setup
def game_will_terminate (a : ℕ) (N : ℕ) (S : Fin N → ℕ) : Prop :=
  ∀ (Alice Bob: ℕ → ℕ),
  -- Alice's move: increment by a
  (∀ i, Alice i = S i + a) ∧
  -- Bob's move: divide by 2 if even
  (∀ i, (S i % 2 = 0 → Bob i = S i / 2) ∧ (S i % 2 ≠ 0 → Bob i = S i)) →
  -- The game always terminates
  ∃ k, ∀ i, (S i % 2^k = 0) → (S i = 1)

theorem guarantee_game_termination (a : ℕ) (N : ℕ) (S : Fin N → ℕ) :
  (∀ i, nu_2 (S i) < nu_2 a) →
  game_will_terminate a N S := sorry

end guarantee_game_termination_l136_136100


namespace quadrilateral_area_l136_136581

variable (a : ℕ) (h_pos : 0 < a)

theorem quadrilateral_area (a_pos : 0 < a) :
  let eq1 := (x + a * y) ^ 2 = 4 * a ^ 2
  let eq2 := (a * x - y) ^ 2 = a ^ 2
  set lines := [
    {p : ℝ × ℝ // eq1 ∧ eq2},
    {p : ℝ × ℝ // eq1 ∧ eq2},
    {p : ℝ × ℝ // eq1 ∧ eq2},
    {p : ℝ × ℝ // eq1 ∧ eq2}
  ]
  area_of_lines lines = 8 * a ^ 2 / (a ^ 2 + 1) :=
sorry

end quadrilateral_area_l136_136581


namespace centroid_equilateral_l136_136587

variables {A B C D E F D1 D2 D3 E1 E2 E3 F1 F2 F3 P Q R M1 M2 M3 : Type*}
variables [AddCommGroup A] [Module ℝ A]
variables [AddCommGroup B] [Module ℝ B]
variables [AddCommGroup C] [Module ℝ C]
variables [AddCommGroup D] [Module ℝ D]
variables [AddCommGroup E] [Module ℝ E]
variables [AddCommGroup F] [Module ℕ F]

/- Define the points and the triangle properties -/
def is_midpoint (X Y Z : Type*) [AddCommGroup X] [Module ℝ X] := ∃ M : X, 
  M = (Y + Z) / 2

def is_centroid (X Y Z : Type*) [AddCommGroup X] [Module ℝ X] := ∃ G : X, 
  G = (X + Y + Z) / 3

def is_equilateral (X Y Z : Type*) [AddCommGroup X] [Module ℝ X] := 
  ∃ (a b c : X), a + ω b + ω² c = 0

-- Now state the main theorem
theorem centroid_equilateral 
  (hDEF : is_equilateral D E F)
  (hPD : is_midpoint P D1 D2) (hPD3 : is_midpoint D3 D1 D2)
  (hQE : is_midpoint Q E1 E2) (hQE3 : is_midpoint E3 E1 E2)
  (hRF : is_midpoint R F1 F2) (hRF3 : is_midpoint F3 F1 F2)
  (hM1 : is_centroid P D D3) (hM2 : is_centroid Q E E3) (hM3 : is_centroid R F F3) :
  is_equilateral M1 M2 M3 := sorry

end centroid_equilateral_l136_136587


namespace train_length_l136_136506

theorem train_length (speed_kmph : ℕ) (time_sec : ℕ) (length : ℕ) :
  speed_kmph = 63 ∧ time_sec = 8 → length = 140 :=
by {
  sorry,
}

end train_length_l136_136506


namespace measure_of_angle_F_l136_136901

theorem measure_of_angle_F {D E F : ℝ}
  (isosceles : D = E)
  (angle_F_condition : F = D + 40)
  (sum_of_angles : D + E + F = 180) :
  F = 260 / 3 :=
by
  sorry

end measure_of_angle_F_l136_136901


namespace angle_ACB_l136_136531

noncomputable def A : EuclideanSpace ℝ := ⟨0, 1, 0⟩
noncomputable def B : EuclideanSpace ℝ := ⟨0, -√3 / 2, 1 / 2⟩
noncomputable def C : EuclideanSpace ℝ := ⟨0, 0, 0⟩

noncomputable def CA : EuclideanSpace ℝ := A - C
noncomputable def CB : EuclideanSpace ℝ := B - C

noncomputable def dot_product (v1 v2 : EuclideanSpace ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

noncomputable def cos_theta : ℝ := dot_product CA CB / (∥CA∥ * ∥CB∥)

theorem angle_ACB : Real.arccos cos_theta = 150 :=
by sorry

end angle_ACB_l136_136531


namespace arithmetic_problem_l136_136160

theorem arithmetic_problem : 
  let part1 := (20 / 100) * 120
  let part2 := (25 / 100) * 250
  let part3 := (15 / 100) * 80
  let sum := part1 + part2 + part3
  let subtract := (10 / 100) * 600
  sum - subtract = 38.5 := by
  sorry

end arithmetic_problem_l136_136160


namespace all_possible_values_of_k_l136_136687

def is_partition_possible (k : ℕ) : Prop :=
  ∃ (A B : Finset ℕ), (A ∪ B = Finset.range (k + 1)) ∧ (A ∩ B = ∅) ∧ (A.sum id = 2 * B.sum id)

theorem all_possible_values_of_k (k : ℕ) : 
  is_partition_possible k → ∃ m : ℕ, k = 3 * m ∨ k = 3 * m - 1 :=
by
  intro h
  sorry

end all_possible_values_of_k_l136_136687


namespace exists_sequence_n_eq_25_exists_sequence_n_gt_1000_l136_136977

/- Definitions for structures and conditions -/

def cond (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ k : ℕ, k < n → odd (finset.sum (finset.range (n - k)) (λ i, a i * a (i + k + 1)))

/- Theorem statements -/

theorem exists_sequence_n_eq_25 :
  ∃ (a : ℕ → ℕ), (∀ i, a i = 0 ∨ a i = 1) ∧ cond a 25 :=
sorry

theorem exists_sequence_n_gt_1000 :
  ∃ n > 1000, ∃ (a : ℕ → ℕ), (∀ i, a i = 0 ∨ a i = 1) ∧ cond a n :=
sorry


end exists_sequence_n_eq_25_exists_sequence_n_gt_1000_l136_136977


namespace shopkeeper_profit_percentage_l136_136093

-- Definitions based on the conditions
def cost_price_per_kg : ℝ := 1
def weight_used : ℝ := 800 / 1000

-- Calculate cost for the amount the customer gets
def actual_cost : ℝ := cost_price_per_kg * weight_used

-- The selling price for the weight used (customer is charged for 1 kg)
def selling_price : ℝ := cost_price_per_kg

-- Calculate profit
def profit : ℝ := selling_price - actual_cost

-- Calculate profit percentage
def profit_percentage : ℝ := (profit / actual_cost) * 100

-- Theorem stating the profit percentage is 25%
theorem shopkeeper_profit_percentage : profit_percentage = 25 := by
  sorry

end shopkeeper_profit_percentage_l136_136093


namespace minimum_a_for_monotonic_increasing_on_interval_l136_136653

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp x - Real.log x

noncomputable def f_prime (a x : ℝ) : ℝ := a * Real.exp x - 1 / x

def min_value_of_a : ℝ := Real.exp (-1)

theorem minimum_a_for_monotonic_increasing_on_interval :
  (∀ x ∈ Set.Ioo 1 2, 0 ≤ f_prime a x) ↔ a ≥ min_value_of_a :=
by
  sorry

end minimum_a_for_monotonic_increasing_on_interval_l136_136653


namespace minimum_value_of_a_l136_136623

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - Real.log x

theorem minimum_value_of_a (a : ℝ) :
  (∀ x ∈ Ioo 1 2, deriv (f a) x ≥ 0) → a ≥ Real.exp ⁻¹ :=
by
  simp [f]
  sorry

end minimum_value_of_a_l136_136623


namespace apples_in_bowl_l136_136416

theorem apples_in_bowl (green_plus_red_diff red_count : ℕ) (h1 : green_plus_red_diff = 12) (h2 : red_count = 16) :
  red_count + (red_count + green_plus_red_diff) = 44 :=
by
  sorry

end apples_in_bowl_l136_136416


namespace no_function_satisfies_condition_l136_136188

theorem no_function_satisfies_condition : 
  ¬ (∃ f : ℕ → ℕ, ∀ x : ℕ, f(2 * f(x)) = x + 1998) :=
by
  sorry

end no_function_satisfies_condition_l136_136188


namespace radar_coverage_ring_area_l136_136571

theorem radar_coverage_ring_area (r : ℝ) (n : ℕ) (w : ℝ) (h : n = 5) (hr : r = 25) (hw : w = 14) :
  ring_area r n w = 672 * Real.pi / Real.tan (Real.pi / 5) :=
sorry

noncomputable def ring_area (r : ℝ) (n : ℕ) (w : ℝ) : ℝ :=
let a := r / Math.tan (Real.pi / n) in
Real.pi * (a + w)^2 - Real.pi * (a - w)^2

end radar_coverage_ring_area_l136_136571


namespace inequality_holds_for_all_reals_l136_136546

theorem inequality_holds_for_all_reals (x : ℝ) : 
  7 / 20 + |3 * x - 2 / 5| ≥ 1 / 4 :=
sorry

end inequality_holds_for_all_reals_l136_136546


namespace one_fourths_in_seven_halves_l136_136264

theorem one_fourths_in_seven_halves : (7 / 2) / (1 / 4) = 14 := by
  sorry

end one_fourths_in_seven_halves_l136_136264


namespace find_multiplier_l136_136705

variable {a b : ℝ} 

theorem find_multiplier (h1 : 3 * a = x * b) (h2 : a ≠ 0 ∧ b ≠ 0) (h3 : a / 4 = b / 3) : x = 4 :=
by
  sorry

end find_multiplier_l136_136705


namespace inequality_proof_l136_136366

theorem inequality_proof (n : ℕ) (h : n > 1) : 
  4^n / (n + 1) < (nat.factorial (2 * n)) / (nat.factorial n)^2 :=
  sorry

end inequality_proof_l136_136366


namespace no_perfect_squares_before_2019_l136_136324

noncomputable def sequence (c : ℕ) : ℕ → ℕ
| 0     := c
| (n+1) := ⌊sequence n + real.sqrt (sequence n)⌋.to_nat

theorem no_perfect_squares_before_2019 
  (c n : ℕ) 
  (h : ∃ m, sequence c m = 2019) 
  (a : ℕ → ℕ := sequence c) : 
  (∀ m < n, m ≠ m * m) ∧ (∃ k, ∀ i ≥ k, ∃ j, a i = j * j) :=
begin
  sorry
end

end no_perfect_squares_before_2019_l136_136324


namespace circumcircle_tangent_and_KH_passes_through_tangent_point_l136_136168

open EuclideanGeometry

variables {A B C D E F H P T L K : Point}
variables {EF AB AC : Line}
variables {ω : Circle}

-- Define the conditions
def triangle_ABC (A B C D E F H P T L K : Point) (EF AB AC : Line) (ω : Circle) : Prop :=
  is_triangle A B C ∧
  altitude A D ∧ 
  altitude B E ∧ 
  altitude C F ∧ 
  orthocenter H A B C ∧
  perpendicular H EF P ∧
  on_line P EF ∧
  on_line T AB ∧
  on_line L AC ∧
  on_side K BC ∧
  coincides_at_distance BD KC ∧
  circle_passing_through_two_points ω H P ∧
  tangent_to_line ω AH

-- Given
axiom conditions : triangle_ABC A B C D E F H P T L K EF AB AC ω

-- To prove that the circumcircle of triangle ATL and 
-- ω are tangent and KH passes through the tangency point
theorem circumcircle_tangent_and_KH_passes_through_tangent_point :
  tangent (circumcircle A T L) ω ∧
  ∃ Q, tangency_point Q (circumcircle A T L) ω ∧ lies_on Q K H :=
sorry

end circumcircle_tangent_and_KH_passes_through_tangent_point_l136_136168


namespace number_of_ways_to_buy_three_items_l136_136734

def headphones : ℕ := 9
def mice : ℕ := 13
def keyboards : ℕ := 5
def keyboard_mouse_sets : ℕ := 4
def headphone_mouse_sets : ℕ := 5

theorem number_of_ways_to_buy_three_items : 
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 := 
by 
  sorry

end number_of_ways_to_buy_three_items_l136_136734


namespace find_a_l136_136349

def f (x: ℝ) : ℝ :=
  if x < 1 then -x else (x - 1) * (x - 1)

theorem find_a (a : ℝ) (h : f a = 1) : a = -1 ∨ a = 2 :=
sorry

end find_a_l136_136349


namespace negation_proposition_false_l136_136393

variable {R : Type} [LinearOrderedField R]

theorem negation_proposition_false (x y : R) :
  ¬ (x > 2 ∧ y > 3 → x + y > 5) = false := by
sorry

end negation_proposition_false_l136_136393


namespace print_time_l136_136129

theorem print_time (P R: ℕ) (hR : R = 24) (hP : P = 360) (T : ℕ) : T = P / R → T = 15 := by
  intros h
  rw [hR, hP] at h
  exact h

end print_time_l136_136129


namespace initial_volume_is_440_l136_136115

-- Conditions as definitions
def initial_percentage_sugar : ℝ := 100 - 88 - 8
def added_sugar : ℝ := 3.2
def added_water : ℝ := 10
def added_kola : ℝ := 6.8
def final_percentage_sugar : ℝ := 4.521739130434784 / 100

-- Function to calculate the initial volume of the solution
noncomputable def initial_volume_solution (V : ℝ) : Prop :=
  let total_initial_volume := V
  let new_volume := V + added_sugar + added_water + added_kola
  let initial_sugar_volume := (initial_percentage_sugar / 100) * V
  let final_sugar_volume := initial_sugar_volume + added_sugar
  final_sugar_volume / new_volume = final_percentage_sugar

-- Declaration of the theorem
theorem initial_volume_is_440 : 
  ∃ V : ℝ, initial_volume_solution V ∧ V = 440 := 
by 
  sorry

end initial_volume_is_440_l136_136115


namespace general_term_correct_sum_formula_correct_l136_136051

-- Define the sequence {a_n}
def a : ℕ → ℤ
| 0     => 1
| n + 1 => 3 * a n + 1

-- Define the formula for the general term
noncomputable def a_formula (n : ℕ) : ℤ := 2^n - 1

-- Define the sum of the first n terms of the sequence
noncomputable def S (n : ℕ) : ℤ := ∑ i in Finset.range n, a i

-- Define the formula for the sum of the first n terms
noncomputable def S_formula (n : ℕ) : ℤ := 2^(n + 1) - 2 - n

-- Theorem to prove the general term formula
theorem general_term_correct : ∀ (n : ℕ), a n = a_formula n :=
by
  sorry

-- Theorem to prove the sum of the first n terms formula
theorem sum_formula_correct : ∀ (n : ℕ), S n = S_formula n :=
by
  sorry

end general_term_correct_sum_formula_correct_l136_136051


namespace angle_DAB_eq_45_l136_136851

-- Define the geometrical configurations and properties
variables {A B C D E : Type} [EuclideanGeometry A B C D E]

-- The conditions given in the problem
axiom parallelogram_ABCD : Parallelogram A B C D
axiom point_E_on_BC : OnLine E B C
axiom isosceles_DEC : IsoscelesTriangle D E C
axiom isosceles_BED : IsoscelesTriangle B E D
axiom isosceles_BAD : IsoscelesTriangle B A D

-- The mathematical statement to be proved
theorem angle_DAB_eq_45 (h1 : Parallelogram A B C D) 
                            (h2 : OnLine E B C) 
                            (h3 : IsoscelesTriangle D E C) 
                            (h4 : IsoscelesTriangle B E D)
                            (h5 : IsoscelesTriangle B A D) :
                            Angle_DEG (angle D A B) := 45 :=
sorry

end angle_DAB_eq_45_l136_136851


namespace total_feet_l136_136493

theorem total_feet (H C F : ℕ) (h1 : H + C = 48) (h2 : H = 28) :
  F = 2 * H + 4 * C → F = 136 :=
by
  -- substitute H = 28 and perform the calculations
  sorry

end total_feet_l136_136493


namespace five_digit_palindromes_count_l136_136174

theorem five_digit_palindromes_count : 
  let a_choices := 9 in
  let b_choices := 10 in
  let c_choices := 10 in
  a_choices * b_choices * c_choices = 900 :=
by
  sorry

end five_digit_palindromes_count_l136_136174


namespace function_interval_length_condition_l136_136189

theorem function_interval_length_condition (f : ℝ → ℝ) (H : ∀ a b : ℝ, a < b → (f '' set.Icc a b).interval_length = b - a) :
  ∃ c : ℝ, (∀ x : ℝ, f x = x + c) ∨ (∀ x : ℝ, f x = -x + c) :=
  by sorry

end function_interval_length_condition_l136_136189


namespace team_total_games_l136_136971

theorem team_total_games
  (win_rate_first_30 : ℝ)
  (first_30_games : ℕ)
  (win_rate_remaining : ℝ)
  (total_win_rate : ℝ)
  (total_games : ℕ)
  (wins_first_30 : ℕ)
  (remaining_games : ℕ)
  (wins_remaining : ℕ)
  (total_wins : ℕ) :
  win_rate_first_30 = 0.4 →
  first_30_games = 30 →
  win_rate_remaining = 0.8 →
  total_win_rate = 0.5 →
  wins_first_30 = (win_rate_first_30 * first_30_games).toNat →
  total_games = first_30_games + remaining_games →
  wins_remaining = (win_rate_remaining * remaining_games).toNat →
  total_wins = wins_first_30 + wins_remaining →
  total_wins = (total_win_rate * total_games).toNat →
  total_games = 40 :=
by
  intro win_rate_first_30_is_0_4
  intro first_30_games_is_30
  intro win_rate_remaining_is_0_8
  intro total_win_rate_is_0_5
  intro wins_first_30_is_12
  intro total_games_is_30_plus_remaining_games
  intro wins_remaining_is_0_8_times_remaining_games
  intro total_wins_is_sum_of_wins
  intro total_wins_is_0_5_times_total_games
  -- Proof omitted
  sorry

end team_total_games_l136_136971


namespace sara_quarters_l136_136018

theorem sara_quarters (initial_quarters : ℕ) (additional_quarters : ℕ) (total_quarters : ℕ) 
    (h1 : initial_quarters = 21) 
    (h2 : additional_quarters = 49) 
    (h3 : total_quarters = initial_quarters + additional_quarters) : 
    total_quarters = 70 :=
sorry

end sara_quarters_l136_136018


namespace smallest_b_for_range_l136_136463

theorem smallest_b_for_range :
  ∃ (b : ℤ), (b < 51) ∧ (∀ a, 29 < a ∧ a < 41 → 0.4 = ((40 : ℤ) / b - (30 : ℤ) / 50)) ∧ b = 40 :=
sorry

end smallest_b_for_range_l136_136463


namespace find_digits_until_2014_l136_136091

def odd_numbers_sequence := [1, 3, 5, 7, 9, 11, 13, 15, 17, 19] ++ List.range' 21 999 (2) ++ List.range' 1001 4201 (2)

def digit_count_sequence_up_to_substring (substring : String) (sequence : List ℕ) : ℕ :=
  let digits_sequence := String.join (sequence.map toString)
  match digits_sequence.indexOf? substring with
  | none => digits_sequence.length + substring.length -- Include substring length if not found
  | some idx => digits_sequence.take idx ++ substring).length

theorem find_digits_until_2014 : 
  digit_count_sequence_up_to_substring "2014" odd_numbers_sequence = 7850 := by
  sorry

end find_digits_until_2014_l136_136091


namespace avg_speed_3x_km_l136_136095

-- Definitions based on the conditions
def distance1 (x : ℕ) : ℕ := x
def speed1 : ℕ := 90
def distance2 (x : ℕ) : ℕ := 2 * x
def speed2 : ℕ := 20

-- The total distance covered
def total_distance (x : ℕ) : ℕ := distance1 x + distance2 x

-- The time taken for each part of the journey
def time1 (x : ℕ) : ℚ := distance1 x / speed1
def time2 (x : ℕ) : ℚ := distance2 x / speed2

-- The total time taken
def total_time (x : ℕ) : ℚ := time1 x + time2 x

-- The average speed
def average_speed (x : ℕ) : ℚ := total_distance x / total_time x

-- The theorem we want to prove
theorem avg_speed_3x_km (x : ℕ) : average_speed x = 27 := by
  sorry

end avg_speed_3x_km_l136_136095


namespace inequalities_count_three_l136_136831

theorem inequalities_count_three
  (x y a b : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (h1 : x^2 < a^2) (h2 : y^3 < b^3) :
  (x^2 + y^2 < a^2 + b^2) ∧ ¬(x^2 - y^2 < a^2 - b^2) ∧ (x^2 * y^3 < a^2 * b^3) ∧ (x^2 / y^3 < a^2 / b^3) := 
sorry

end inequalities_count_three_l136_136831


namespace sum_of_interior_angles_of_special_regular_polygon_l136_136410

theorem sum_of_interior_angles_of_special_regular_polygon (n : ℕ) (h1 : n = 4 ∨ n = 5) :
  ((n - 2) * 180 = 360 ∨ (n - 2) * 180 = 540) :=
by sorry

end sum_of_interior_angles_of_special_regular_polygon_l136_136410


namespace alpha_has_winning_strategy_l136_136832

def initial_piles : (ℕ × ℕ × ℕ) := (100, 101, 102)

def valid_moves (current: ℕ × ℕ × ℕ) (prev_pile: Option ℕ) : List (ℕ × ℕ × ℕ) :=
  match prev_pile with
  | none => [(current.1 - 1, current.2, current.3), (current.1, current.2 - 1, current.3), (current.1, current.2, current.3 - 1)]
  | some 0 => [(current.1, current.2 - 1, current.3), (current.1, current.2, current.3 - 1)]
  | some 1 => [(current.1 - 1, current.2, current.3), (current.1, current.2, current.3 - 1)]
  | some 2 => [(current.1 - 1, current.2, current.3), (current.1, current.2 - 1, current.3)]
  | _ => []

def next_turn_win (current: ℕ × ℕ × ℕ) (prev_pile: Option ℕ) : Bool :=
  next_move_wins current (valid_moves current prev_pile)

def next_move_wins (current: ℕ × ℕ × ℕ) (next_moves: List (ℕ × ℕ × ℕ)) : Bool :=
  next_moves.all (λ next_move => next_turn_win next_move none = false)

theorem alpha_has_winning_strategy : next_turn_win initial_piles none := 
  sorry

end alpha_has_winning_strategy_l136_136832


namespace range_of_m_l136_136241

theorem range_of_m {m : ℝ} (h1 : m^2 - 1 < 0) (h2 : m > 0) : 0 < m ∧ m < 1 :=
sorry

end range_of_m_l136_136241


namespace eccentricity_of_ellipse_l136_136226

-- Define the problem statement
theorem eccentricity_of_ellipse
  (a b : ℝ) (F1 F2 : ℝ × ℝ) (P Q : ℝ × ℝ)
  (h_ellipse : 0 < b ∧ b < a)
  (h_foci : ∀ x y, (x,y) = F1 ∨ (x,y) = F2 → (x^2 / a^2 + y^2 / b^2 = 1))
  (h_PQ_foci : (P ≠ Q) ∧ (Q = F2 ∨ ∃ k, Q = k • P))
  (h_angle : ∃ θ, θ = 45 ∧ Π θ', θ' = θ → angle Q F1 P = θ')
  (h_lengths : |dist Q P| = sqrt 2 * |dist P F1|)
  : eccentricity (a, b) = sqrt 2 - 1 := by
  -- Proof goes here
  sorry

end eccentricity_of_ellipse_l136_136226


namespace A_finishes_remaining_work_in_6_days_l136_136092

-- Definitions for conditions
def A_workdays : ℕ := 18
def B_workdays : ℕ := 15
def B_worked_days : ℕ := 10

-- Proof problem statement
theorem A_finishes_remaining_work_in_6_days (A_workdays B_workdays B_worked_days : ℕ) :
  let rate_A := 1 / A_workdays
  let rate_B := 1 / B_workdays
  let work_done_by_B := B_worked_days * rate_B
  let remaining_work := 1 - work_done_by_B
  let days_A_needs := remaining_work / rate_A
  days_A_needs = 6 :=
by
  sorry

end A_finishes_remaining_work_in_6_days_l136_136092


namespace exist_same_color_sequence_l136_136219
-- Import the entirety of the necessary library

-- Define the conditions and the theorem statement
theorem exist_same_color_sequence
  (n : ℕ) (h_n : n ≥ 2)
  (color : ℕ → Prop)
  (h_color : ∀ x, x ≤ n * (n^2 - 2*n + 3) / 2 → color x ∨ color x) :
  ∃ a : ℕ → ℕ, (strict_monotone a) ∧
    (∀ i, i < n → color (a i)) ∧
    (∀ i, 0 < i ∧ i < n → a (i + 1) - a i ≤ a i - a (i - 1)) :=
by
  sorry

end exist_same_color_sequence_l136_136219


namespace coin_flip_probability_difference_l136_136948

theorem coin_flip_probability_difference :
  let p1 := (nat.choose 5 4) * (1 / 2)^5
  let p2 := (1 / 2)^5
  (p1 - p2 = 1 / 8) :=
by
  let p1 := (nat.choose 5 4) * (1 / 2)^5
  let p2 := (1 / 2)^5
  sorry

end coin_flip_probability_difference_l136_136948


namespace tangent_line_at_e_l136_136035

noncomputable def f (x : ℝ) : ℝ := x / log x

noncomputable def f_prime (x : ℝ) : ℝ := (log x - 1) / (log x) ^ 2

theorem tangent_line_at_e :
  let e := real.exp 1 in
  let tangent_line := λ (x : ℝ), e in
  tangent_line = λ x, e := by
  sorry

end tangent_line_at_e_l136_136035


namespace problem_statement_l136_136582

theorem problem_statement (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x^2 + y^2 ≤ 1) :
  | x^2 + 2*x*y - y^2 | ≤ √2 :=
sorry

end problem_statement_l136_136582


namespace pyramid_surface_area_l136_136811

def isosceles_triangle_area (a b : ℝ) : ℝ := 
  let h := sqrt (b ^ 2 - (a / 2) ^ 2)
  in (1 / 2) * a * h

def tetrahedron_surface_area (a b : ℝ) : ℝ :=
  4 * isosceles_triangle_area a b

theorem pyramid_surface_area {a b : ℝ} 
  (DABC_edge_length : ∀ {e : ℝ}, e ∈ {a, b} → e = 20 ∨ e = 45)
  (not_equilateral : ∀ Δ : Type, ¬(a = 20 ∧ b = 20 ∧ a = b)) :
  tetrahedron_surface_area 20 45 = 40 * sqrt (1925) := 
  by
  sorry

end pyramid_surface_area_l136_136811


namespace min_steps_for_humpty_l136_136699

theorem min_steps_for_humpty (x y : ℕ) (H : 47 * x - 37 * y = 1) : x + y = 59 :=
  sorry

end min_steps_for_humpty_l136_136699


namespace average_score_of_graduating_students_is_125_l136_136314

noncomputable def average_score_graduating : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ → ℝ
| (total_students, avg_score_total, non_grad_extra_ratio_num, non_grad_extra_ratio_den, avg_score_grad_ratio_num, avg_score_grad_ratio_den) :=
  let graduating_students := total_students * non_grad_extra_ratio_den / (non_grad_extra_ratio_num + non_grad_extra_ratio_den) in
  let non_graduating_students := total_students - graduating_students in
  let avg_score_non_grad := (avg_score_total * total_students - graduating_students * avg_score_grad_ratio_num * avg_score_non_grad_ratio_den / avg_score_grad_ratio_den) / non_graduating_students in
  avg_score_grad_ratio_num * avg_score_non_grad / avg_score_grad_ratio_den

theorem average_score_of_graduating_students_is_125 :
  average_score_graduating (100, 100, 3, 2, 3, 2) = 125 :=
sorry

end average_score_of_graduating_students_is_125_l136_136314


namespace part_1_intervals_monotonically_decreasing_part_2_triangle_solution_l136_136259

noncomputable def vec_m (x : ℝ) : ℝ × ℝ := (Real.sin x, -1)
noncomputable def vec_n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, -1 / 2)
noncomputable def f (x : ℝ) : ℝ := (vec_m x + vec_n x) • vec_n x

theorem part_1_intervals_monotonically_decreasing {x : ℝ} (k : ℤ) :
  f x < f (x + 1) :=
sorry

theorem part_2_triangle_solution {A : ℝ} {b : ℝ} {S : ℝ} (hA : A = π / 3) (hb : b = 2) (hS : S = 2 * Real.sqrt 3) :
  let a := 2 * Real.sqrt 3,
  let c := 4,
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A ∧
  (1 / 2) * b * c * Real.sin A = S :=
sorry

end part_1_intervals_monotonically_decreasing_part_2_triangle_solution_l136_136259


namespace period_sin_sub_cos_l136_136917

theorem period_sin_sub_cos : ∀ x : ℝ, sin (x + 2 * π) - cos (x + 2 * π) = sin x - cos x :=
by
  intro x
  calc
    sin (x + 2 * π) - cos (x + 2 * π) = sin x - cos (x + 2 * π)    : by rw [sin_add, cos_add, cos_two_pi, sin_two_pi, add_zero, zero_add, sub_zero]
    ...                               = sin x - cos x               : by rw [cos_two_pi]
    ...                               = sin x - cos x               : by sorry  -- Proof of the equivalence

end period_sin_sub_cos_l136_136917


namespace a_range_is_correct_l136_136229

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * a * x^2 - (x - 3) * Exp.exp x + 1
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 6 * a * x - x * Exp.exp x + 2 * Exp.exp x

def a_in_range (a : ℝ) : Prop :=
  ∃ x y ∈ Ioo 0 2, x ≠ y ∧ f_derivative a x = 0 ∧ f_derivative a y = 0

theorem a_range_is_correct (a : ℝ) : a_in_range a ↔ (a > e / 3 ∧ a < e^2 / 6) :=
sorry

end a_range_is_correct_l136_136229


namespace problem1_period_monotonic_problem2_range_fA_l136_136214

-- Definitions for f(x)
def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * (Real.sqrt 3) * (Real.sin x) * (Real.cos x)

-- Problem 1: Finding the smallest positive period and the monotonically increasing interval
theorem problem1_period_monotonic (x : ℝ) : ∃ T : ℝ, T = π ∧
  ∀ k : ℤ, -π/3 + k * π ≤ x ∧ x ≤ π / 6 + k * π → f' x > 0 :=
sorry

-- Definition of conditions in triangle ABC
variables (a b c : ℝ) (A B C : ℝ)

-- Problem 2: Finding the range of f(A) based on given conditions
theorem problem2_range_fA (A B C : ℝ) (h1 : (a + 2 * c) * Real.cos B = -b * Real.cos A)
  (h2 : 0 < A ∧ A < π / 3) :
  2 < f A ∧ f A ≤ 3 :=
sorry

end problem1_period_monotonic_problem2_range_fA_l136_136214


namespace compounding_interest_rate_interest_earned_l136_136558

noncomputable def continuous_compounding_interest_rate
  (A P : ℝ) (t : ℝ) : ℝ :=
  (1 / t) * Real.log (A / P)

theorem compounding_interest_rate
  (A : ℝ)
  (P : ℝ)
  (t : ℝ)
  (hA : A = 45000)
  (hP : P = 30000)
  (ht : t = 3.5) :
  continuous_compounding_interest_rate A P t ≈ 0.11584717 :=
by
  rw [continuous_compounding_interest_rate, hA, hP, ht]
  simp
  sorry


theorem interest_earned
  (A P : ℝ)
  (hA : A = 45000)
  (hP : P = 30000) :
  A - P = 15000 :=
by
  rw [hA, hP]
  norm_num

end compounding_interest_rate_interest_earned_l136_136558


namespace solution_set_for_inequality_l136_136055

theorem solution_set_for_inequality :
  {x : ℝ | -x^2 + 2 * x + 3 ≥ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} :=
sorry

end solution_set_for_inequality_l136_136055


namespace christmas_distribution_l136_136023

theorem christmas_distribution :
  ∃ (n x : ℕ), 
    (240 + 120 + 1 = 361) ∧
    (n * x = 361) ∧
    (n = 19) ∧
    (x = 19) ∧
    ∃ (a b : ℕ), (a + b = 19) ∧ (a * 5 + b * 6 = 100) :=
by
  sorry

end christmas_distribution_l136_136023


namespace area_of_triangle_ABC_l136_136289

theorem area_of_triangle_ABC
  (A B C H M : Type)
  (hABC : ∀ (A B C : Type), ∃ (H : Type), ∃ (M : Type), ∠C = 90 ∧ altitude CH ∧ median CM ∧ bisects_CM_C ∧ area_ΔCHM K)
  (hK : ∃ (H : Type), ∃ (M : Type), ∀ (A B C : Type), ∃ (K : ℝ), K = area_ΔCHM)
  (bisects_CM_C : ∀ (C M : Type), ∠MCH = 45 ∧ ∠MCB = 45)
  (altitude CH : ∀ (C H : Type), perpendicular CH to AB)
  (median CM : ∀ (C M : Type), midpoint M of AB)
  (area_ΔCHM : ℝ)
  (h_area_ΔCHM_eq_K : area_ΔCHM = K)
  : area_ΔABC = 2 * K :=
sorry

end area_of_triangle_ABC_l136_136289


namespace cube_inequality_contradiction_l136_136009

theorem cube_inequality_contradiction
  (x y : ℝ) (h : x > y) : x^3 > y^3 :=
begin
  have : ¬ (x^3 < y^3 ∨ x^3 = y^3), -- Assuming the negation
  have : x^3 ≠ y^3, -- Assuming no equality between cubes
  have : x^3 ≥ y^3, -- From initial comparison property
  sorry -- proof goes here
end

end cube_inequality_contradiction_l136_136009


namespace minimum_value_of_a_l136_136639

theorem minimum_value_of_a (a : ℝ) :
  (∀ x ∈ (Ioo 1 2), ae^x - ln x ≥ 0) ↔ a ≥ exp(-1) := sorry

end minimum_value_of_a_l136_136639


namespace distance_is_correct_line_not_perpendicular_l136_136559

-- Given conditions

constant a b c d : ℝ
noncomputable def Plane1 (x y z : ℝ) : Prop := 3 * x + 2 * y - z = 7
noncomputable def Plane2 (x y z : ℝ) : Prop := 3 * x + 2 * y - z = 2
noncomputable def Line (t : ℝ) (x y z : ℝ) : Prop := (x = t ∧ y = t ∧ z = t)

def distance_between_planes : ℝ :=
  7 / (√(3^2 + 2^2 + (-1)^2))

def direction_vector_line : ℝ × ℝ × ℝ := (1, 1, 1)
def normal_vector_planes : ℝ × ℝ × ℝ := (3, 2, -1)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem distance_is_correct :
  distance_between_planes = (7 * √14) / 14 := sorry

theorem line_not_perpendicular :
  dot_product direction_vector_line normal_vector_planes ≠ 0 := sorry

end distance_is_correct_line_not_perpendicular_l136_136559


namespace complex_conj_modulus_l136_136227

noncomputable def complex_expr (a : ℝ) : ℂ :=
  (a + complex.I) / (1 + complex.I)

theorem complex_conj_modulus (a : ℝ) 
  (h1 : ∀ z : ℂ, (z = complex_expr a) → z.re = z.im) :
  a = 0 ∧ |conj (complex_expr 0)| = real.sqrt 2 / 2 :=
by
  sorry

end complex_conj_modulus_l136_136227


namespace ellipse_eq_find_k_l136_136594

open Real

-- Part I: Prove the equation of the ellipse
theorem ellipse_eq (a b : ℝ) (h_a : a = 2) (h_b : b = 1) :
  ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 4 + y^2 = 1 :=
by
  sorry

-- Part II: Prove the value of k
theorem find_k (P : ℝ × ℝ) (hP : P = (-2, 1)) (MN_length : ℝ) (hMN : MN_length = 2) 
  (k : ℝ) (a b : ℝ) (h_a : a = 2) (h_b : b = 1) :
  ∃ k : ℝ, k = -4 :=
by
  sorry

end ellipse_eq_find_k_l136_136594


namespace incircle_inequality_l136_136318

variable {α : Type*} [LinearOrderField α]

-- Definitions based on the conditions in the problem
variables (A B C D E F P : α)
variables (BC CA AB AD AP : α)
variables (touchpoints : α) (intersection : α)

-- Assumptions based on the problem setting
variables (h1 : D = touchpoints)
variables (h2 : E = touchpoints)
variables (h3 : F = touchpoints)
variables (h4 : AD = intersection)
variables (h5 : EF = intersection)

-- The theorem based on the transformed problem
theorem incircle_inequality (h6 : \(\frac{AP}{AD}\) ≥ 1 - \(\frac{BC}{AB + CA}\))
  : \(\frac{AP}{AD} \ge 1 - \frac{BC}{AB + CA}\) := 
sorry

end incircle_inequality_l136_136318


namespace ratio_of_areas_l136_136454

-- Define the side length of small squares and derived areas
def side_length_small_square : ℕ := 5
def area_small_square : ℕ := side_length_small_square * side_length_small_square

-- Define the area of the shaded triangle
def area_shaded_triangle : ℝ := (3 * area_small_square + (area_small_square / 2 : ℝ))

-- Define the side length of the large square and its area
def side_length_large_square : ℕ := 25
def area_large_square : ℕ := side_length_large_square * side_length_large_square

-- Prove the ratio of the area of the shaded triangle to the area of the large square
theorem ratio_of_areas : area_shaded_triangle / (area_large_square : ℝ) = (7 / 50 : ℝ) :=
by
  -- Proof will be filled in here
  sorry

end ratio_of_areas_l136_136454


namespace trapezium_other_side_length_l136_136904

theorem trapezium_other_side_length 
  (side1 : ℝ) (perpendicular_distance : ℝ) (area : ℝ) (side1_val : side1 = 5) 
  (perpendicular_distance_val : perpendicular_distance = 6) (area_val : area = 27) : 
  ∃ other_side : ℝ, other_side = 4 :=
by
  sorry

end trapezium_other_side_length_l136_136904


namespace area_of_triangle_ABC_l136_136103

theorem area_of_triangle_ABC (AB AC : ℝ) (angle_A : ℝ) : 
  AB = real.sqrt 3 → AC = 1 → angle_A = real.pi / 6 → 
  let a := AB * AC * real.sin angle_A / 2 in a = real.sqrt 3 / 4 :=
by sorry

end area_of_triangle_ABC_l136_136103


namespace tiffany_ate_pies_l136_136182

theorem tiffany_ate_pies (baking_days : ℕ) (pies_per_day : ℕ) (wc_per_pie : ℕ) 
                         (remaining_wc : ℕ) (total_pies : ℕ) (total_wc : ℕ) :
  baking_days = 11 → pies_per_day = 3 → wc_per_pie = 2 → remaining_wc = 58 →
  total_pies = pies_per_day * baking_days → total_wc = total_pies * wc_per_pie →
  (total_wc - remaining_wc) / wc_per_pie = 4 :=
by 
  intros h1 h2 h3 h4 h5 h6
  sorry

end tiffany_ate_pies_l136_136182


namespace ice_cream_remaining_l136_136109

def total_initial_scoops : ℕ := 3 * 10
def ethan_scoops : ℕ := 1 + 1
def lucas_danny_connor_scoops : ℕ := 2 * 3
def olivia_scoops : ℕ := 1 + 1
def shannon_scoops : ℕ := 2 * olivia_scoops
def total_consumed_scoops : ℕ := ethan_scoops + lucas_danny_connor_scoops + olivia_scoops + shannon_scoops
def remaining_scoops : ℕ := total_initial_scoops - total_consumed_scoops

theorem ice_cream_remaining : remaining_scoops = 16 := by
  sorry

end ice_cream_remaining_l136_136109


namespace base8_subtraction_correct_l136_136550

theorem base8_subtraction_correct : (453 - 326 : ℕ) = 125 :=
by sorry

end base8_subtraction_correct_l136_136550


namespace coprime_permutations_count_l136_136905

def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def adjacent_coprime (l : List ℕ) : Prop :=
  ∀ i, i < l.length - 1 → coprime (l.nthLe i sorry) (l.nthLe (i + 1) sorry)

def valid_permutations (l : List ℕ) : List (List ℕ) :=
  l.permutations.filter adjacent_coprime

theorem coprime_permutations_count :
  let l := [2, 3, 4, 5, 6, 7] in
  (valid_permutations l).length = 72 := 
by
  sorry

end coprime_permutations_count_l136_136905


namespace remaining_scoops_l136_136110

-- Define the initial scoops for each flavor
def initial_chocolate : Nat := 10
def initial_strawberry : Nat := 10
def initial_vanilla : Nat := 10

-- Define the scoops requested by each person
def ethan_chocolate : Nat := 1
def ethan_vanilla : Nat := 1

def lucas_danny_connor_chocolate : Nat := 3 * 2

def olivia_strawberry : Nat := 1
def olivia_vanilla : Nat := 1

def shannon_strawberry : Nat := 2
def shannon_vanilla : Nat := 2

-- Calculate and prove the remaining scoops for each flavor
theorem remaining_scoops :
  let remaining_chocolate := initial_chocolate - (ethan_chocolate + lucas_danny_connor_chocolate)
  let remaining_strawberry := initial_strawberry - (olivia_strawberry + shannon_strawberry)
  let remaining_vanilla := initial_vanilla - (ethan_vanilla + olivia_vanilla + shannon_vanilla)
  remaining_chocolate + remaining_strawberry + remaining_vanilla = 16 :=
by
  let remaining_chocolate := initial_chocolate - (ethan_chocolate + lucas_danny_connor_chocolate)
  let remaining_strawberry := initial_strawberry - (olivia_strawberry + shannon_strawberry)
  let remaining_vanilla := initial_vanilla - (ethan_vanilla + olivia_vanilla + shannon_vanilla)
  calc
    remaining_chocolate + remaining_strawberry + remaining_vanilla
    = (10 - (1 + 6)) + (10 - (1 + 2)) + (10 - (1 + 1 + 2)) : by rfl
    ... = 3 + 7 + 6 : by rfl
    ... = 16 : by rfl

end remaining_scoops_l136_136110


namespace average_of_remaining_8_numbers_l136_136858

theorem average_of_remaining_8_numbers (S : ℕ) (h1 : S / 10 = 85) :
    (S - 72 - 78) / 8 = 87.5 :=
sorry

end average_of_remaining_8_numbers_l136_136858


namespace part1_arithmetic_sequence_part2_max_min_terms_part3_bounds_l136_136249

-- Define the sequence {a_n}
def a (n : ℕ) : ℝ :=
  if n = 1 then 3/5
  else 2 - 1 / (a (n - 1))

-- Define the sequence {b_n}
def b (n : ℕ) : ℝ := 1 / (a n - 1)

-- Part 1: Prove {b_n} is an arithmetic sequence
theorem part1_arithmetic_sequence (n : ℕ) (h₀ : n ≥ 2) : b n - b (n - 1) = 1 :=
sorry

-- Part 2: Prove existence of max and min term in {a_n} and identify them
theorem part2_max_min_terms (n : ℕ) : 
  (∃ m, a m = a 3 ∧ a m ≤ a k) ∧ (∃ M, a M = a 4 ∧ a M ≥ a k)
  (k : ℕ) : 
sorry

-- Part 3: Prove 1 < a_{n+1} < a_n < 2
theorem part3_bounds (n : ℕ) (h₀ : 1 < a 1 ∧ a 1 < 2) : 1 < a (n+1) ∧ a (n+1) < a n ∧ a n < 2 :=
sorry

end part1_arithmetic_sequence_part2_max_min_terms_part3_bounds_l136_136249


namespace gather_checkers_in_n_minus_1_moves_l136_136789

noncomputable def gather_checkers (n : ℕ) : ℕ :=
  n - 1

theorem gather_checkers_in_n_minus_1_moves (n : ℕ) (Hpos : 1 ≤ n) :
  ∃ m, m = gather_checkers n ∧ 
  (∀ (initial_checker_positions : fin n → bool), gather_checkers_in_moves n initial_checker_positions m) :=
begin
  -- provided initial_checker_positions has a checker in each position (all are true)
  sorry
end

-- Definition to check valid gathering of checkers within moves
def gather_checkers_in_moves (n : ℕ) (initial_checker_positions : fin n → bool) (m : ℕ) :=
  -- check condition to gather all checkers in (n-1) moves
  sorry 

end gather_checkers_in_n_minus_1_moves_l136_136789


namespace complex_number_quadrant_l136_136778

theorem complex_number_quadrant :
  let z := (⟨ 2, 4 ⟩ : ℂ) / ⟨ 1, 1 ⟩ in
  let coord := (z.re, z.im) in
  coord.1 > 0 ∧ coord.2 > 0 := by
sorry

end complex_number_quadrant_l136_136778


namespace velocity_of_M_l136_136167

open Real

-- Conditions
def ω : ℝ := 10
def length_OA : ℝ := 0.9 -- in meters
def length_AB : ℝ := 0.9 -- in meters
def length_AM : ℝ := length_AB / 2

-- Coordinates of points A and M
def A_x (t : ℝ) : ℝ := length_OA * cos (ω * t)
def A_y (t : ℝ) : ℝ := length_OA * sin (ω * t)
def M_x (t : ℝ) : ℝ := A_x t
def M_y (t : ℝ) : ℝ := A_y t / 2

theorem velocity_of_M {t : ℝ} : sqrt ((-900 * sin(10 * t))^2 + (450 * cos(10 * t))^2) = 450 := by
  sorry

end velocity_of_M_l136_136167


namespace initial_tickets_l136_136435

theorem initial_tickets 
  (jude_tickets : ℕ := 16)
  (andrea_tickets : ℕ := 2 * jude_tickets)
  (sandra_tickets : ℕ := (jude_tickets / 2) + 4)
  (tickets_left : ℕ := 40) :
  let total_tickets_sold := jude_tickets + andrea_tickets + sandra_tickets in
  let initial_tickets := total_tickets_sold + tickets_left in
  initial_tickets = 100 := 
by
  sorry

end initial_tickets_l136_136435


namespace count_of_satisfying_f_mod_10_eq_0_l136_136814

def f (x : ℤ) : ℤ := x^3 + 2 * x^2 + 3 * x + 2

def S : finset ℤ := finset.range 31

theorem count_of_satisfying_f_mod_10_eq_0 :
  (S.filter (λ s, f s % 10 = 0)).card = 12 := by
  sorry

end count_of_satisfying_f_mod_10_eq_0_l136_136814


namespace store_purchase_ways_l136_136739

theorem store_purchase_ways :
  ∃ (headphones sets_hm sets_km keyboards mice : ℕ), 
  headphones = 9 ∧
  mice = 13 ∧
  keyboards = 5 ∧
  sets_km = 4 ∧
  sets_hm = 5 ∧
  (sets_km * headphones + sets_hm * keyboards + headphones * mice * keyboards = 646) :=
by
  use 9, 5, 4, 5, 13
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  sorry

end store_purchase_ways_l136_136739


namespace measure_of_angle_F_l136_136903

theorem measure_of_angle_F (x : ℝ) (h1 : ∠D = ∠E) (h2 : ∠F = x + 40) (h3 : 2 * x + (x + 40) = 180) : 
  ∠F = 86.67 :=
by 
  sorry

end measure_of_angle_F_l136_136903


namespace f_f_15_div_2_eq_sqrt_3_div_2_l136_136320

def f (x : ℝ) : ℝ :=
  if x >= 0 then sin (π * x)
  else cos (π * x / 2 + π / 3)

theorem f_f_15_div_2_eq_sqrt_3_div_2 : f (f (15 / 2)) = sqrt 3 / 2 := by
  sorry

end f_f_15_div_2_eq_sqrt_3_div_2_l136_136320


namespace problem_I1_3_problem_I1_4_l136_136300

-- Definition and theorem for Problem I1.3
def angle_sum_of_triangle (x y z : ℝ) : Prop := x + y + z = 180
def opposite_angles_of_cyclic_quad (a b : ℝ) : Prop := a + b = 180

theorem problem_I1_3 (p q a : ℝ) (h1: opposite_angles_of_cyclic_quad (180 - p) q) (h2: angle_sum_of_triangle (180 - p) (180 - q) a) (ha: a = 25) :
  p + q = 205 :=
by { sorry }

-- Definition and theorem for Problem I1.4
def sum_series (z : ℕ) : ℕ :=
(1 + 2 - 3 - 4 + 5 + 6 - 7 - 8 ... + z) -- This needs better definition in Lean syntax

theorem problem_I1_4 (z : ℕ) (hz: z = 205) :
  sum_series z = 1 :=
by { sorry }

end problem_I1_3_problem_I1_4_l136_136300


namespace positive_integers_are_N_star_l136_136088

def Q := { x : ℚ | true } -- The set of rational numbers
def N := { x : ℕ | true } -- The set of natural numbers
def N_star := { x : ℕ | x > 0 } -- The set of positive integers
def Z := { x : ℤ | true } -- The set of integers

theorem positive_integers_are_N_star : 
  ∀ x : ℕ, (x ∈ N_star) ↔ (x > 0) := 
sorry

end positive_integers_are_N_star_l136_136088


namespace circles_externally_tangent_l136_136877

namespace CircleTangency

-- Define the first circle with its equation
def circle1 := {x y : ℝ // x^2 + y^2 - 4 * x + 2 * y + 1 = 0}

-- Define the second circle with its equation
def circle2 := {x y : ℝ // x^2 + y^2 + 4 * x - 4 * y - 1 = 0}

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (2, -1)
def radius1 : ℝ := 2

def center2 : ℝ × ℝ := (-2, 2)
def radius2 : ℝ := 3

-- Define the distance formula between the two centers
def distance : ℝ := Real.sqrt ((-2 - 2)^2 + (2 - (-1))^2)

-- Theorem to prove the positional relationship
theorem circles_externally_tangent :
  distance = radius1 + radius2 :=
begin
  -- The proof would go here
  sorry
end

end CircleTangency

end circles_externally_tangent_l136_136877


namespace triangle_sine_value_l136_136718

-- Define the triangle sides and angles
variables {a b c A B C : ℝ}

-- Main theorem stating the proof problem
theorem triangle_sine_value (h : a^2 = b^2 + c^2 - bc) :
  (a * Real.sin B) / b = Real.sqrt 3 / 2 := sorry

end triangle_sine_value_l136_136718


namespace find_FC_l136_136574

-- Define the geometric elements
variables {D C B A E F : ℝ} (AD DC CB AB ED FC : ℝ)

-- Conditions
axiom h1 : DC = 6
axiom h2 : CB = 9
axiom h3 : AB = (1 / 5) * AD
axiom h4 : ED = (2 / 3) * AD

-- Theorem to prove
theorem find_FC (h_AD : AD = 18.75)
                (h_BD : (4/5) * AD = 15)
                (h_ED_def : ED = 12.5)
                (h_AB_def : AB = 3.75)
                (h_CA_def : CA = 12.75)
                (h_similarity : FC = (ED * CA) / AD) : 
                FC = 8.54 := by
  -- Introduction of variables and axiom dependencies
  obtain ⟨AD, _⟩ := h_AD,
  obtain ⟨BD, _⟩ := h_BD,
  obtain ⟨ED, _⟩ := h_ED_def,
  obtain ⟨AB, _⟩ := h_AB_def,
  obtain ⟨CA, _⟩ := h_CA_def,
  obtain ⟨FC, _⟩ := h_similarity,
  
  -- Placeholder for the proof
  sorry

end find_FC_l136_136574


namespace find_m_and_max_value_l136_136251

theorem find_m_and_max_value (m : ℝ) 
  (f : ℝ → ℝ := λ x, -x^3 + 3 * x^2 + 9 * x + m) 
  (h_max : ∀ x ∈ set.Icc (-2 : ℝ) (2 : ℝ), f x ≤ 20) 
  (h_at_2 : f 2 = 20) : 
  m = -2 ∧ f 2 = 20 := 
by 
  sorry

end find_m_and_max_value_l136_136251


namespace max_distinct_circular_highways_l136_136572

/--
Given four cities that are not on the same circle, prove that the greatest number of geographically distinct circular highways equidistant from all four cities is 7.
-/
theorem max_distinct_circular_highways (cities : Set Point) (h₁ : ∃ p₁ p₂ p₃ p₄ ∈ cities, ¬ collinear ({p₁, p₂, p₃, p₄} : Set Point) 4) :
  ∃ n, n = 7 ∧ distinct_circular_highways(cities, n) :=
sorry

end max_distinct_circular_highways_l136_136572


namespace tuna_cost_is_correct_l136_136513

noncomputable def cost_of_tuna (T : ℝ) : Prop :=
  let cost_tuna := 5 * T
  let cost_water := 4 * 1.5
  let total_cost_tuna_water := 56 - 40
  cost_tuna + cost_water = total_cost_tuna_water

theorem tuna_cost_is_correct : cost_of_tuna 2 :=
by
  unfold cost_of_tuna
  norm_num
  sorry

end tuna_cost_is_correct_l136_136513


namespace plate_acceleration_l136_136071

theorem plate_acceleration (R r : ℝ) (m : ℝ) (α : ℝ) (g : ℝ) (hR : R = 1.25) (hr : r = 0.75) (hm : m = 100) (hα : α = Real.arccos 0.92) (hg : g = 10) : 
  let a := g * Real.sin(α / 2) in
  a = 2 :=
by
  -- Declaration of given data
  have hR : R = 1.25 := hR,
  have hr : r = 0.75 := hr,
  have hm : m = 100 := hm,
  have hα : α = Real.arccos 0.92 := hα,
  have hg : g = 10 := hg,
  
  -- Calculate acceleration
  
  sorry

end plate_acceleration_l136_136071


namespace promote_cashback_beneficial_cashback_rubles_preferable_l136_136975

-- For the benefit of banks promoting cashback services
theorem promote_cashback_beneficial :
  ∀ (bank : Type) (services : Type), 
  (∃ (customers : set bank) (businesses : set services), 
    (∀ b ∈ businesses, ∃ d ∈ customers, promotes_cashback d b) ∧
    (∀ c ∈ customers, prefers_cashback c b)) :=
sorry

-- For the preference of customers on cashback in rubles
theorem cashback_rubles_preferable :
  ∀ (customer : Type) (cashback : Type),
  (∀ r ∈ cashback, rubles r → prefers_customer r cashback) :=
sorry

end promote_cashback_beneficial_cashback_rubles_preferable_l136_136975


namespace definite_integral_eq_l136_136166

open Real
open Set

theorem definite_integral_eq :
  (∫ x in -5 / 3 .. 1, (cbrt (3 * x + 5) + 2) / (1 + cbrt (3 * x + 5))) = (8 / 3 + log 3) :=
by
  sorry

end definite_integral_eq_l136_136166


namespace min_value_of_a_l136_136634

theorem min_value_of_a (a : ℝ) : 
  (∀ x ∈ Ioo 1 2, (a * exp x - 1/x) ≥ 0) → a = exp (-1) :=
by
  sorry

end min_value_of_a_l136_136634


namespace proj_b_v_l136_136337

-- Define orthogonality
def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

-- Define vector projections
def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let c := (u.1 * v.1 + u.2 * v.2) / (v.1 * v.1 + v.2 * v.2)
  (c * v.1, c * v.2)

-- Given vectors a and b and their properties
variables (a b : ℝ × ℝ) (v : ℝ × ℝ)
variable h_ab_orthogonal : orthogonal a b
variable h_proj_a : proj v a = (-4/5, -8/5)

-- The goal to prove
theorem proj_b_v :
  proj v b = (24/5, -2/5) := sorry

end proj_b_v_l136_136337


namespace monotonic_decreasing_interval_l136_136044

noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.log x

theorem monotonic_decreasing_interval :
  ∀ x, 0 < x ∧ x < 1 / 2 → ∀ (y > 0), (y ≤ x) → f'(y) < 0 ∧ ∀(z ≥ 1 / 2), f'(z) > 0 
  :=
begin
  sorry -- Proof will be constructed here
end

end monotonic_decreasing_interval_l136_136044


namespace wheat_ear_frequencies_l136_136500

theorem wheat_ear_frequencies :
  ∀ (n : ℕ) (W1A W2B : ℚ),
  n = 200 → W1A = 0.125 → W2B = 0.05 →
  let m1 := W1A * n in
  let m2 := W2B * n in
  m1 = 25 ∧ m2 = 10 := 
by
  intros n W1A W2B hn hW1A hW2B
  rw [hn, hW1A, hW2B]
  have m1 : ℚ := W1A * n
  have m2 : ℚ := W2B * n
  simp [m1, m2]
  split
  norm_num
  norm_num
  sorry

end wheat_ear_frequencies_l136_136500


namespace bucket_weight_l136_136476

theorem bucket_weight (x y p q : ℝ) 
  (h1 : x + (3 / 4) * y = p) 
  (h2 : x + (1 / 3) * y = q) :
  x + (5 / 6) * y = (6 * p - q) / 5 :=
sorry

end bucket_weight_l136_136476


namespace compare_probabilities_l136_136490

noncomputable def box_bad_coin_prob_method_one : ℝ := 1 - (0.99 ^ 10)
noncomputable def box_bad_coin_prob_method_two : ℝ := 1 - ((49 / 50) ^ 5)

theorem compare_probabilities : box_bad_coin_prob_method_one < box_bad_coin_prob_method_two := by
  sorry

end compare_probabilities_l136_136490


namespace length_of_one_pencil_l136_136791

theorem length_of_one_pencil (l : ℕ) (h1 : 2 * l = 24) : l = 12 :=
by {
  sorry
}

end length_of_one_pencil_l136_136791


namespace cannot_determine_parallelogram_l136_136964

-- Define the conditions
variable {A B C D O : Type} -- Define points as types/variables
variable [Geometry A B C D O] -- Geometry context in which A, B, C, D, O exist

-- Conditions for quadrilateral ABCD
def condition_A (h1: Parallel AB CD) (h2: Parallel AD BC) : Parallelogram ABCD := sorry
def condition_B (h1: Eq AB CD) (h2: Eq AD BC) : Parallelogram ABCD := sorry
def condition_C (h1: Eq OA OC) (h2: Eq OB OD) : Parallelogram ABCD := sorry
def condition_D (h1: Parallel AB CD) (h2: Eq AD BC) : ¬Parallelogram ABCD := sorry

-- Main theorem: Option D does not determine that quadrilateral is a parallelogram.
theorem cannot_determine_parallelogram (h1: Parallel AB CD) (h2: Eq AD BC) : ¬Parallelogram ABCD :=
  condition_D h1 h2

end cannot_determine_parallelogram_l136_136964


namespace purchase_combinations_correct_l136_136756

theorem purchase_combinations_correct : 
  let headphones : ℕ := 9
  let mice : ℕ := 13
  let keyboards : ℕ := 5
  let sets_keyboard_mouse : ℕ := 4
  let sets_headphones_mouse : ℕ := 5
in 
  (sets_keyboard_mouse * headphones) + 
  (sets_headphones_mouse * keyboards) + 
  (headphones * mice * keyboards) = 646 := 
by
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let sets_keyboard_mouse := 4
  let sets_headphones_mouse := 5
  sorry

end purchase_combinations_correct_l136_136756


namespace square_diagonal_sum_of_squares_l136_136586

theorem square_diagonal_sum_of_squares
  (A B C D E F H : Point) 
  (h_square: Square A B C D)
  (h_perpendicular1: Perpendicular (LineThrough A E) (LineThrough B C))
  (h_perpendicular2: Perpendicular (LineThrough A F) (LineThrough C D))
  (h_orthocenter: Orthocenter H (Triangle A E F)) :
  dist A C ^ 2 = dist A H ^ 2 + dist E F ^ 2 :=
sorry

end square_diagonal_sum_of_squares_l136_136586


namespace determine_BH_value_l136_136439
open Classical

-- Definition of the conditions
variables (ABC : Triangle)
  (AB BC CA : ℝ) (G H I : ABC.Point)
  (AG AH : ℝ) (p q r s : ℕ)

-- Constants fulfilling given conditions
def given_conditions :=
  AB = 3 ∧ BC = 7 ∧ CA = 8 ∧
  AB < AG ∧ AG < AH ∧
  GI = 1 ∧ HI = 6 ∧ I ≠ C

-- The proof problem statement
theorem determine_BH_value (h : given_conditions) :
  p + q + r + s = 204 :=
sorry

end determine_BH_value_l136_136439


namespace reinforcement_size_l136_136123

theorem reinforcement_size (R : ℕ) : 
  2000 * 39 = (2000 + R) * 20 → R = 1900 :=
by
  intro h
  sorry

end reinforcement_size_l136_136123


namespace find_value_of_p_find_line_equation_l136_136217

open Real

variables {p : ℝ} (hp : p > 0)
variables {x1 y1 x2 y2 : ℝ}
variables {A B M : ℝ × ℝ} (O : ℝ × ℝ)
variables {l : ℝ × ℝ → Prop}

-- Line passing through point M intersects the parabola at points A and B
def line_intersects_parabola (M : ℝ × ℝ) (l : ℝ × ℝ → Prop) (A B : ℝ × ℝ) : Prop :=
  M = (p / 2, 0) ∧ l M ∧ l A ∧ l B ∧ (∃ x p > 0, par : ℝ × ℝ → Prop, par (x, y) → y^2 = 2 * p * x)

-- Dot product condition
def dot_product_condition (O : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  let (x1, y1) := A in
  let (x2, y2) := B in
  x1 * x2 + y1 * y2 = -3

-- Variables for the second part of the problem
variables {l_eq : ℝ × ℝ → Prop} (x y m : ℝ)

-- Equation of line when |AM| + 4|BM| is minimized
def minimization_condition (A B M : ℝ × ℝ) : Prop :=
  let (x1, _) := A in
  let (x2, _) := B in
  |((x1, 0) - M)| + 4 * |((x2, 0) - M)| = 9 ∧ line_eq (λ x y, 4 * x + (sqrt 2) * y - 4 = 0 ∨ 4 * x - (sqrt 2) * y - 4 = 0)

theorem find_value_of_p (A B M : ℝ × ℝ) (O : ℝ × ℝ) (l : ℝ × ℝ → Prop) :
  line_intersects_parabola M l A B → dot_product_condition O A B → p = 2 :=
sorry

theorem find_line_equation (A B M : ℝ × ℝ) (O : ℝ × ℝ) (l : ℝ × ℝ → Prop) :
  line_intersects_parabola M l A B → dot_product_condition O A B → minimization_condition A B M → 
  line_eq (λ x y, 4 * x + (sqrt 2) * y - 4 = 0 ∨ 4 * x - (sqrt 2) * y - 4 = 0) :=
sorry

end find_value_of_p_find_line_equation_l136_136217


namespace fair_coin_flip_probability_difference_l136_136940

noncomputable def choose : ℕ → ℕ → ℕ
| _, 0 => 1
| 0, _ => 0
| n + 1, r + 1 => choose n r + choose n (r + 1)

theorem fair_coin_flip_probability_difference :
  let p := 1 / 2
  let p_heads_4_out_of_5 := choose 5 4 * p^4 * p
  let p_heads_5_out_of_5 := p^5
  p_heads_4_out_of_5 - p_heads_5_out_of_5 = 1 / 8 :=
by
  sorry

end fair_coin_flip_probability_difference_l136_136940


namespace proper_subsets_A_l136_136873

def A : Set ℤ := {x | |x| < 3}

theorem proper_subsets_A (A : Set ℤ) (hA : A = {x | |x| < 3}) : 
  Fintype.card (Set.Subset A) = 2 ^ Fintype.card A - 1 := by
  sorry

#eval proper_subsets_A

end proper_subsets_A_l136_136873


namespace total_ways_to_buy_l136_136765

-- Definitions:
def h : ℕ := 9  -- number of headphones
def m : ℕ := 13 -- number of mice
def k : ℕ := 5  -- number of keyboards
def s_km : ℕ := 4 -- number of sets of keyboard and mouse
def s_hm : ℕ := 5 -- number of sets of headphones and mouse

-- Theorem stating the number of ways to buy three items is 646.
theorem total_ways_to_buy : (s_km * h) + (s_hm * k) + (h * m * k) = 646 := by
  -- Placeholder proof using sorry.
  sorry

end total_ways_to_buy_l136_136765


namespace relationship_between_p_and_q_l136_136818

variables {x y : ℝ}

def p (x y : ℝ) := (x^2 + y^2) * (x - y)
def q (x y : ℝ) := (x^2 - y^2) * (x + y)

theorem relationship_between_p_and_q (h1 : x < y) (h2 : y < 0) : p x y > q x y := 
  by sorry

end relationship_between_p_and_q_l136_136818


namespace proj_b_v_l136_136335

-- Define orthogonality
def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

-- Define vector projections
def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let c := (u.1 * v.1 + u.2 * v.2) / (v.1 * v.1 + v.2 * v.2)
  (c * v.1, c * v.2)

-- Given vectors a and b and their properties
variables (a b : ℝ × ℝ) (v : ℝ × ℝ)
variable h_ab_orthogonal : orthogonal a b
variable h_proj_a : proj v a = (-4/5, -8/5)

-- The goal to prove
theorem proj_b_v :
  proj v b = (24/5, -2/5) := sorry

end proj_b_v_l136_136335


namespace circle_equation_tangent_to_two_lines_l136_136216

noncomputable def tangent_circle_equation : Prop :=
  ∃ (a r : ℝ), 
    (∀ x y : ℝ, (x - a)^2 + (y - 1)^2 = r^2)
    ∧ (∀ x y : ℝ, abs ((2 * x - y + 4)) / sqrt 5 = r)
    ∧ (∀ x y : ℝ, abs ((2 * x - y - 6)) / sqrt 5 = r)
    ∧ a = 1 ∧ r = sqrt 5

theorem circle_equation_tangent_to_two_lines : tangent_circle_equation :=
  sorry

end circle_equation_tangent_to_two_lines_l136_136216


namespace point_P_in_second_quadrant_l136_136605

-- Definitions for the three angles of an acute triangle
variable (A B C : ℝ)
-- Definitions for representing the point P
variable P : ℝ × ℝ

-- Conditions provided by the problem: A and B are angles of an acute triangle
def acute_triangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < π / 2 ∧
  0 < B ∧ B < π / 2 ∧
  0 < C ∧ C < π / 2 ∧
  A + B + C = π

-- Definitions of the coordinates of the point P based on A and B
def point_P (A B : ℝ) : ℝ × ℝ :=
  (Real.cos B - Real.sin A, Real.sin B - Real.cos A)

-- The theorem to prove: Point P lies in the second quadrant
theorem point_P_in_second_quadrant
  (h : acute_triangle A B C) : 
  let P := point_P A B in
  (P.1 < 0) ∧ (P.2 > 0) := 
sorry

end point_P_in_second_quadrant_l136_136605


namespace number_of_solutions_g100_eq_0_l136_136339

def g0 (x : ℝ) : ℝ :=
if x < -200 then x + 400
else if x < 200 then -x
else x - 400

def gn : ℕ → ℝ → ℝ 
| 0, x := g0 x
| (n + 1), x := abs (gn n x) - 2

theorem number_of_solutions_g100_eq_0 : 
  (∃ (s : finset ℝ), (∀ x ∈ s, gn 100 x = 0) ∧ s.card = 3) :=
sorry

end number_of_solutions_g100_eq_0_l136_136339


namespace area_enclosed_by_curve_and_tangent_l136_136557

-- Define the curve function
def curve (x : ℝ) := x^3

-- Define the tangent line function derived at x = 3
def tangent_line (x : ℝ) := 27 * x - 54

-- Statement to prove
theorem area_enclosed_by_curve_and_tangent : ∫ x in 0..3, (curve x - tangent_line x) = 27 / 4 :=
by 
  -- formulation of integral and area calculation
  sorry

end area_enclosed_by_curve_and_tangent_l136_136557


namespace multiply_negatives_l136_136159

theorem multiply_negatives : (- (1 / 2)) * (- 2) = 1 :=
by
  sorry

end multiply_negatives_l136_136159


namespace canoe_rental_cost_l136_136075

theorem canoe_rental_cost (C : ℕ) (K : ℕ) :
  18 * K + C * (K + 5) = 405 → 
  3 * K = 2 * (K + 5) → 
  C = 15 :=
by
  intros revenue_eq ratio_eq
  sorry

end canoe_rental_cost_l136_136075


namespace Q_zero_roots_count_l136_136529
-- Lean 4 Statement for the proof problem

noncomputable def Q (x : ℝ) : ℝ :=
  Real.cos x + 2 * Real.sin x - Real.cos (2 * x) - 2 * Real.sin (2 * x) + Real.cos (3 * x) + 2 * Real.sin (3 * x)

theorem Q_zero_roots_count :
  ∃ (x1 x2 : ℝ), 0 ≤ x1 ∧ x1 < 2 * Real.pi ∧ 0 ≤ x2 ∧ x2 < 2 * Real.pi ∧ x1 ≠ x2 ∧ Q x1 = 0 ∧ Q x2 = 0 ∧
  ∀ x ∈ Ico 0 (2 * Real.pi), Q x = 0 → x = x1 ∨ x = x2 :=
sorry

end Q_zero_roots_count_l136_136529


namespace buy_items_ways_l136_136729

theorem buy_items_ways (headphones keyboards mice keyboard_mouse_sets headphone_mouse_sets : ℕ) :
  headphones = 9 → keyboards = 5 → mice = 13 → keyboard_mouse_sets = 4 → headphone_mouse_sets = 5 →
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 :=
by
  intros h_eq k_eq m_eq kms_eq hms_eq
  have h_eq_gen : headphones = 9 := h_eq
  have k_eq_gen : keyboards = 5 := k_eq
  have m_eq_gen : mice = 13 := m_eq
  have kms_eq_gen : keyboard_mouse_sets = 4 := kms_eq
  have hms_eq_gen : headphone_mouse_sets = 5 := hms_eq
  sorry

end buy_items_ways_l136_136729


namespace daisy_germination_rate_theorem_l136_136262

-- Define the conditions of the problem
variables (daisySeeds sunflowerSeeds : ℕ) (sunflowerGermination flowerProduction finalFlowerPlants : ℝ)
def conditions : Prop :=
  daisySeeds = 25 ∧ sunflowerSeeds = 25 ∧ sunflowerGermination = 0.80 ∧ flowerProduction = 0.80 ∧ finalFlowerPlants = 28

-- Define the statement that the germination rate of the daisy seeds is 60%
def germination_rate_of_daisy_seeds : Prop :=
  ∃ (daisyGerminationRate : ℝ), (conditions daisySeeds sunflowerSeeds sunflowerGermination flowerProduction finalFlowerPlants) →
  daisyGerminationRate = 0.60

-- The proof is omitted - note this is just the statement
theorem daisy_germination_rate_theorem : germination_rate_of_daisy_seeds 25 25 0.80 0.80 28 :=
sorry

end daisy_germination_rate_theorem_l136_136262


namespace det_is_18_l136_136412

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![4, 1],
    ![2, 5]]

theorem det_is_18 : det A = 18 := by
  sorry

end det_is_18_l136_136412


namespace absentees_exactly_three_truthful_l136_136392

def statements (n : Nat) :=
  (n > 1, n > 2, n > 3, n > 4, n < 4, n < 3)

theorem absentees_exactly_three_truthful :
  ∃ n : Nat, ∃ truths : Fin 6 → Bool, 
    (∑ i, if truths i then 1 else 0) = 3 ∧
    (truths 0 = (statements n).1) ∧
    (truths 1 = (statements n).2) ∧
    (truths 2 = (statements n).3) ∧
    (truths 3 = (statements n).4) ∧
    (truths 4 = (statements n).5) ∧
    (truths 5 = (statements n).6) ∧
    n = 2 ∨ n = 3 ∨ n = 4 :=
begin
  sorry
end

end absentees_exactly_three_truthful_l136_136392


namespace value_of_g_at_3_l136_136274

theorem value_of_g_at_3 (g : ℕ → ℕ) (h : ∀ x, g (x + 2) = 2 * x + 3) : g 3 = 5 := by
  sorry

end value_of_g_at_3_l136_136274


namespace boys_difference_twice_girls_l136_136045

theorem boys_difference_twice_girls :
  ∀ (total_students girls boys : ℕ),
  total_students = 68 →
  girls = 28 →
  boys = total_students - girls →
  2 * girls - boys = 16 :=
by
  intros total_students girls boys h1 h2 h3
  sorry

end boys_difference_twice_girls_l136_136045


namespace problem1_problem2_problem3_problem4_l136_136686

/-
A set M is a "perpendicular point set" if for any point (x₁, y₁) ∈ M, 
there exists (x₂, y₂) ∈ M such that x₁x₂ + y₁y₂ = 0.
-/

/-- Define when a set is a "perpendicular point set". -/
def is_perpendicular_point_set (f : ℝ → ℝ) : Prop :=
  ∀ (x₁ : ℝ), ∃ (x₂ : ℝ), x₁ * x₂ + f x₁ * f x₂ = 0

/-- Proportion 1: y = 1 / x² is a "perpendicular point set". -/
theorem problem1 : is_perpendicular_point_set (λ x, 1 / (x ^ 2)) :=
sorry

/-- Proportion 2: y = sin x + 1 is a "perpendicular point set". -/
theorem problem2 : is_perpendicular_point_set (λ x, Real.sin x + 1) :=
sorry

/-- Proportion 3: y = 2 ^ x - 2 is a "perpendicular point set". -/
theorem problem3 : is_perpendicular_point_set (λ x, 2 ^ x - 2) :=
sorry

/-- Proportion 4: y = log₂ x is not a "perpendicular point set". -/
theorem problem4 : ¬ is_perpendicular_point_set (λ x, Real.log x / Real.log 2) :=
sorry

end problem1_problem2_problem3_problem4_l136_136686


namespace f_of_exp_x_f_of_5_eq_ln_5_l136_136212

noncomputable 
def f : ℝ → ℝ := sorry

theorem f_of_exp_x (x : ℝ) : f (real.exp x) = x := sorry

theorem f_of_5_eq_ln_5 : f 5 = real.log 5 :=
by
  have h := f_of_exp_x (real.log 5)
  rw real.exp_log (by norm_num) at h
  exact h

end f_of_exp_x_f_of_5_eq_ln_5_l136_136212


namespace remaining_scoops_l136_136111

-- Define the initial scoops for each flavor
def initial_chocolate : Nat := 10
def initial_strawberry : Nat := 10
def initial_vanilla : Nat := 10

-- Define the scoops requested by each person
def ethan_chocolate : Nat := 1
def ethan_vanilla : Nat := 1

def lucas_danny_connor_chocolate : Nat := 3 * 2

def olivia_strawberry : Nat := 1
def olivia_vanilla : Nat := 1

def shannon_strawberry : Nat := 2
def shannon_vanilla : Nat := 2

-- Calculate and prove the remaining scoops for each flavor
theorem remaining_scoops :
  let remaining_chocolate := initial_chocolate - (ethan_chocolate + lucas_danny_connor_chocolate)
  let remaining_strawberry := initial_strawberry - (olivia_strawberry + shannon_strawberry)
  let remaining_vanilla := initial_vanilla - (ethan_vanilla + olivia_vanilla + shannon_vanilla)
  remaining_chocolate + remaining_strawberry + remaining_vanilla = 16 :=
by
  let remaining_chocolate := initial_chocolate - (ethan_chocolate + lucas_danny_connor_chocolate)
  let remaining_strawberry := initial_strawberry - (olivia_strawberry + shannon_strawberry)
  let remaining_vanilla := initial_vanilla - (ethan_vanilla + olivia_vanilla + shannon_vanilla)
  calc
    remaining_chocolate + remaining_strawberry + remaining_vanilla
    = (10 - (1 + 6)) + (10 - (1 + 2)) + (10 - (1 + 1 + 2)) : by rfl
    ... = 3 + 7 + 6 : by rfl
    ... = 16 : by rfl

end remaining_scoops_l136_136111


namespace problem_statement_l136_136786

-- Let ABC be a triangle
variables {A B C D E F : Type} 

-- Conditions
def is_midpoint (D A B : Type) [AffineSpace ℜ A] := (A + B) / 2 = D
def is_perpendicular (U V W : Type) [InnerProductSpace ℜ U] := ⟪V, W⟫ = 0

-- Given Conditions
variables [inhabited A] [inhabited B] [inhabited C]
variables [inhabited D] [inhabited E] [inhabited F]
variables [inhabited (AffineSpace ℜ A)] [inhabited (InnerProductSpace ℜ A)]

-- Questions and assertions
theorem problem_statement (h1 : is_midpoint D A B) (h2 : is_perpendicular E D C) (h3 : is_perpendicular F D A B) : 
  (DE = DF ∨ DE < DF ∨ DE > DF) := sorry

end problem_statement_l136_136786


namespace propositions_true_l136_136618

theorem propositions_true (a b : ℝ) (f : ℝ → ℝ)
  (h1 : ∀ a b : ℝ, ∃ x : ℝ, f(x) = Math.ln(x) ∧ x ≠ Math.real.log(x^2 + b*x + c))
  (h2 : ∀ x : ℝ, ¬(f(x + 2) = f(2 - x)))
  (h3 : ∃! x : ℝ, Math.ln(x) + x = 4)
  (h4 : ∀ x y : ℝ, ∃ a : ℝ, (a^2*x^2 + (a + 2)*y^2 + 2*a*x + a = 0 → a = -1)) :
  (h1) ∧ ¬(h2) ∧ (h3) ∧ (h4) :=
by
  sorry

end propositions_true_l136_136618


namespace find_first_type_cookies_l136_136525

section CookiesProof

variable (x : ℕ)

-- Conditions
def box_first_type_cookies : ℕ := x
def box_second_type_cookies : ℕ := 20
def box_third_type_cookies : ℕ := 16
def boxes_first_type_sold : ℕ := 50
def boxes_second_type_sold : ℕ := 80
def boxes_third_type_sold : ℕ := 70
def total_cookies_sold : ℕ := 3320

-- Theorem to prove
theorem find_first_type_cookies 
  (h1 : 50 * x + 80 * box_second_type_cookies + 70 * box_third_type_cookies = total_cookies_sold) :
  x = 12 := by
    sorry

end CookiesProof

end find_first_type_cookies_l136_136525


namespace number_of_ways_to_buy_three_items_l136_136735

def headphones : ℕ := 9
def mice : ℕ := 13
def keyboards : ℕ := 5
def keyboard_mouse_sets : ℕ := 4
def headphone_mouse_sets : ℕ := 5

theorem number_of_ways_to_buy_three_items : 
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 := 
by 
  sorry

end number_of_ways_to_buy_three_items_l136_136735


namespace reflect_y_axis_reflect_x_axis_reflect_origin_reflect_line_y_eq_x_reflect_line_y_eq_neg_x_l136_136585

-- Define a point in ℝ^2
structure Point : Type :=
(x : ℝ)
(y : ℝ)

-- Conditions and expected results for the reflections
theorem reflect_y_axis (P : Point) : Point :=
  { x := -P.x, y := P.y }

theorem reflect_x_axis (P : Point) : Point :=
  { x := P.x, y := -P.y }

theorem reflect_origin (P : Point) : Point :=
  { x := -P.x, y := -P.y }

theorem reflect_line_y_eq_x (P : Point) : Point :=
  { x := P.y, y := P.x }

theorem reflect_line_y_eq_neg_x (P : Point) : Point :=
  { x := -P.y, y := -P.x }

#eval reflect_y_axis ⟨1, 2⟩  -- Should output ⟨-1, 2⟩
#eval reflect_x_axis ⟨1, 2⟩  -- Should output ⟨1, -2⟩
#eval reflect_origin ⟨1, 2⟩  -- Should output ⟨-1, -2⟩
#eval reflect_line_y_eq_x ⟨1, 2⟩  -- Should output ⟨2, 1⟩
#eval reflect_line_y_eq_neg_x ⟨1, 2⟩  -- Should output ⟨-2, -1⟩

end reflect_y_axis_reflect_x_axis_reflect_origin_reflect_line_y_eq_x_reflect_line_y_eq_neg_x_l136_136585


namespace digit_in_decimal_expansion_of_x2_l136_136462

noncomputable def x_1 : ℝ := (1/2) - (Real.sqrt 2 / 4)
noncomputable def x_2 : ℝ := (1/2) + (Real.sqrt 2 / 4)

theorem digit_in_decimal_expansion_of_x2 :
  (∀ n : ℕ, ((x_1.toDecimalString 10 (n + 1))[(2047 - n).toNat % 10]) = ('6')) →
  ((x_2.toDecimalString 10 1994[1993].toNat % 10) = '3') :=
sorry

end digit_in_decimal_expansion_of_x2_l136_136462


namespace min_value_of_a_l136_136636

theorem min_value_of_a (a : ℝ) : 
  (∀ x ∈ Ioo 1 2, (a * exp x - 1/x) ≥ 0) → a = exp (-1) :=
by
  sorry

end min_value_of_a_l136_136636


namespace minimum_value_of_a_l136_136668

theorem minimum_value_of_a (a : ℝ) (h : ∀ x ∈ set.Ioo 1 2, ae^x - 1/x ≥ 0) : a ≥ e⁻¹ :=
sorry

end minimum_value_of_a_l136_136668


namespace fair_coin_difference_l136_136927

noncomputable def binom_coeff (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem fair_coin_difference :
  let p := (1 : ℚ) / 2 in
  let p_1 := binom_coeff 5 4 * p^4 * (1 - p) in
  let p_2 := p^5 in
  p_1 - p_2 = 1 / 8 :=
by
  sorry

end fair_coin_difference_l136_136927


namespace barking_dogs_count_l136_136114

def number_of_dogs (barks_per_min: ℕ) (total_barks: ℕ) (minutes: ℕ) : ℕ :=
  total_barks / (minutes * barks_per_min)

theorem barking_dogs_count :
  ∀ (barks_per_dog_per_min: ℕ) (total_barks: ℕ) (minutes: ℕ),
  barks_per_dog_per_min = 30 →
  total_barks = 600 →
  minutes = 10 →
  number_of_dogs barks_per_dog_per_min total_barks minutes = 2 :=
by
  intros
  rw [‹barks_per_dog_per_min = 30›, ‹total_barks = 600›, ‹minutes = 10›]
  exact sorry

end barking_dogs_count_l136_136114


namespace incorrect_differentiations_l136_136087

noncomputable def optionA_derivative : ℝ → ℝ := λ x, (1 + 3 / (x^2))
noncomputable def optionA_correct : ℝ → ℝ := λ x, (1 - 3 / (x^2))

noncomputable def optionB_derivative : ℝ → ℝ := λ x, 3 * (x + 3)^2
noncomputable def optionB_correct : ℝ → ℝ := λ x, 3 * (x + 3)^2

noncomputable def optionC_derivative : ℝ → ℝ := λ x, 3 * Real.log x
noncomputable def optionC_correct : ℝ → ℝ := λ x, (3^x) * Real.log 3

noncomputable def optionD_derivative : ℝ → ℝ := λ x, -2 * x * Real.sin x
noncomputable def optionD_correct : ℝ → ℝ := λ x, 2 * x * Real.cos x - x^2 * Real.sin x

theorem incorrect_differentiations :
  (optionA_derivative ≠ optionA_correct) ∧
  (optionB_derivative = optionB_correct) ∧
  (optionC_derivative ≠ optionC_correct) ∧
  (optionD_derivative ≠ optionD_correct) :=
by {
  -- Proof omitted
  sorry
}

end incorrect_differentiations_l136_136087


namespace shop_combinations_l136_136749

theorem shop_combinations :
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let keyboard_and_mouse_sets := 4
  let headphone_and_mouse_sets := 5
  (keyboard_and_mouse_sets * headphones) + (headphone_and_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 :=
by
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let keyboard_and_mouse_sets := 4
  let headphone_and_mouse_sets := 5
  have case1 : keyboard_and_mouse_sets * headphones = 4 * 9, by sorry
  have case2 : headphone_and_mouse_sets * keyboards = 5 * 5, by sorry
  have case3 : headphones * mice * keyboards = 9 * 13 * 5, by sorry
  show (keyboard_and_mouse_sets * headphones) + (headphone_and_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646,
  by rw [case1, case2, case3]; exact sorry

end shop_combinations_l136_136749


namespace complex_power_identity_l136_136275

theorem complex_power_identity (w : ℂ) (h : w + w⁻¹ = 2) : w^(2022 : ℕ) + (w⁻¹)^(2022 : ℕ) = 2 := by
  sorry

end complex_power_identity_l136_136275


namespace minimize_material_used_l136_136895

theorem minimize_material_used (r h : ℝ) (V : ℝ) (S : ℝ) 
  (volume_formula : π * r^2 * h = V) (volume_given : V = 27 * π) :
  ∃ r, r = 3 :=
by
  sorry

end minimize_material_used_l136_136895


namespace oak_trees_remaining_l136_136060

theorem oak_trees_remaining (initial_trees cut_trees remaining_trees : ℕ)
  (h_initial : initial_trees = 9)
  (h_cut : cut_trees = 2)
  (h_remaining : remaining_trees = initial_trees - cut_trees) :
  remaining_trees = 7 := by
  rw [h_initial, h_cut, h_remaining]
  norm_num

end oak_trees_remaining_l136_136060


namespace knave_of_hearts_steals_tarts_l136_136028

theorem knave_of_hearts_steals_tarts :
  (∃ (T : ℝ), (T / 8 - 3 / 8 - 1 / 2) = 1) → T = 15 :=
by
  intros T
  let T_after_hearts := T - (T / 2 + 1 / 2)
  let T_after_diamonds := T_after_hearts - (T_after_hearts / 2 + 1 / 2)
  let T_after_clubs := T_after_diamonds - (T_after_diamonds / 2 + 1 / 2)
  have h : T_after_clubs = 1, by sorry
  sorry

end knave_of_hearts_steals_tarts_l136_136028


namespace dot_product_of_a_and_b_is_correct_l136_136351

-- Define vectors a and b
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (2, -1)

-- Define dot product for ℝ × ℝ vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Theorem statement (proof can be omitted with sorry)
theorem dot_product_of_a_and_b_is_correct : dot_product a b = -4 :=
by
  -- proof goes here, omitted for now
  sorry

end dot_product_of_a_and_b_is_correct_l136_136351


namespace customers_tipped_count_l136_136142

variable (initial_customers : ℕ)
variable (added_customers : ℕ)
variable (customers_no_tip : ℕ)

def total_customers (initial_customers added_customers : ℕ) : ℕ :=
  initial_customers + added_customers

theorem customers_tipped_count 
  (h_init : initial_customers = 29)
  (h_added : added_customers = 20)
  (h_no_tip : customers_no_tip = 34) :
  (total_customers initial_customers added_customers - customers_no_tip) = 15 :=
by
  sorry

end customers_tipped_count_l136_136142


namespace fair_coin_flip_difference_l136_136923

theorem fair_coin_flip_difference :
  let P1 := 5 * (1 / 2)^5;
  let P2 := (1 / 2)^5;
  P1 - P2 = 9 / 32 :=
by
  sorry

end fair_coin_flip_difference_l136_136923


namespace fermat_point_sum_l136_136354

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

theorem fermat_point_sum :
  let D := (0,0)
  let E := (8,0)
  let F := (5,7)
  let P := (3,3)
  distance P D + distance P E + distance P F = 3 * real.sqrt 2 + real.sqrt 34 + 2 * real.sqrt 5 → 
  (3 + 2 = 5) :=
by
  intro h
  sorry

end fermat_point_sum_l136_136354


namespace additional_points_condition_l136_136468

theorem additional_points_condition {n k : ℕ} (h1 : ∀ (A : Fin n → Prop), convex_polygon A)
  (h2 : ∀ (P : Fin k → Prop), marked_inside_polygon P) 
  (h3 : ∀ (p1 p2 p3 : Prop), 
  (p1 ∈ A ∨ p1 ∈ P) ∧ 
  (p2 ∈ A ∨ p2 ∈ P) ∧ 
  (p3 ∈ A ∨ p3 ∈ P) ∧ 
  ¬ collinear p1 p2 p3) 
  (h4 : ∀ (p1 p2 p3 : Prop), 
  (p1 ∈ A ∨ p1 ∈ P) ∧ 
  (p2 ∈ A ∨ p2 ∈ P) ∧ 
  (p3 ∈ A ∨ p3 ∈ P) → 
  is_isosceles_triangle p1 p2 p3) : 
  k ≤ 1 :=
by sorry

end additional_points_condition_l136_136468


namespace igor_number_is_five_l136_136508

-- Define the initial lineup
def initial_lineup : List Nat := [9, 11, 10, 6, 8, 5, 4, 1]

-- Define the condition function that checks if a player should run based on neighbors
def should_run (player : Nat) (left_neighbor right_neighbor : Option Nat) : Bool :=
  (left_neighbor.map (λ n => player < n) == some true) || (right_neighbor.map (λ n => player < n) == some true)

-- Define the process of removing players
def remove_players (lineup : List Nat) : List Nat :=
  lineup.filter_with_index (λ i player =>
    should_run player
      (lineup.nth (i - 1))
      (lineup.nth (i + 1))
  ).reverse ++ lineup.filter_with_index (λ i player =>
    ¬ should_run player
      (lineup.nth (i - 1))
      (lineup.nth (i + 1))
  )

-- Define the process to determine the final player numbers given the initial lineup and ending with 3 players
def final_three_players (lineup : List Nat) : List Nat :=
  (iterate remove_players (lineup, (λ x => x.length != 3))) lineup

-- Prove that Igor's number is 5 given that removing him leaves 3 players
theorem igor_number_is_five (lineup : List Nat) (igor_number : Nat) (remaining_after_igor_leaves : List Nat) :
  final_three_players lineup = remaining_after_igor_leaves ++ [igor_number] →
  igor_number = 5 := by
  sorry

end igor_number_is_five_l136_136508


namespace perpendicular_and_intersection_l136_136169

variables (x y : ℚ)

def line1 := 4 * y - 3 * x = 15
def line4 := 3 * y + 4 * x = 15

theorem perpendicular_and_intersection :
  (4 * y - 3 * x = 15) ∧ (3 * y + 4 * x = 15) →
  let m1 := (3 : ℚ) / 4
  let m4 := -(4 : ℚ) / 3
  m1 * m4 = -1 ∧
  ∃ x y : ℚ, 4*y - 3*x = 15 ∧ 3*y + 4*x = 15 ∧ x = 15/32 ∧ y = 35/8 :=
by
  sorry

end perpendicular_and_intersection_l136_136169


namespace exists_h_l136_136868

noncomputable def F (x : ℝ) : ℝ := x^2 + 12 / x^2
noncomputable def G (x : ℝ) : ℝ := Real.sin (Real.pi * x^2)
noncomputable def H (x : ℝ) : ℝ := 1

theorem exists_h (h : ℝ → ℝ) (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 10) :
  |h x - x| < 1 / 3 :=
sorry

end exists_h_l136_136868


namespace hyperbola_asymptotes_l136_136193

theorem hyperbola_asymptotes (a b t : ℝ) (h_a : a > 0) (h_b : b > 0) (h_t : t ≠ 0)
    (is_asymptote : ∀ x y, (x, y) ≠ (0, 0) → y = b / a * x ∨ y = -b / a * x) 
    (intersect : ∀ x y, x - 3 * y + t = 0)
    (equal_dist : ∀ A B M : Prod ℝ ℝ, |M - A| = |M - B| → M = (t, 0)) :
    (a = 2 * b) → ∀ x, y = ± (1 / 2) * x :=
by sorry

end hyperbola_asymptotes_l136_136193


namespace domain_of_f_l136_136386

def f (x : ℝ) : ℝ := (3 * x ^ 2) / Real.sqrt(1 - x) + Real.log(-3 * x ^ 2 + 5 * x + 2)

theorem domain_of_f : 
  {x : ℝ | 1 - x > 0 ∧ -3 * x ^ 2 + 5 * x + 2 > 0} = 
  {x : ℝ | -1 / 3 < x ∧ x < 1} :=
sorry

end domain_of_f_l136_136386


namespace total_sick_animals_l136_136414

theorem total_sick_animals 
  (num_chickens num_piglets num_goats num_cows num_sheep num_horses : ℕ)
  (sick_frac_chickens sick_frac_piglets sick_frac_goats sick_frac_cows sick_frac_sheep sick_frac_horses : ℚ)
  (total_sick : ℕ) 
  (h1 : num_chickens = 100)
  (h2 : num_piglets = 150)
  (h3 : num_goats = 120)
  (h4 : num_cows = 60)
  (h5 : num_sheep = 75)
  (h6 : num_horses = 30)
  (h7 : sick_frac_chickens = 2/5)
  (h8 : sick_frac_piglets = 5/6)
  (h9 : sick_frac_goats = 1/2)
  (h10 : sick_frac_cows = 3/5)
  (h11 : sick_frac_sheep = 2/3)
  (h12 : sick_frac_horses = 9/10)
  (h13 : total_sick = (sick_frac_chickens * num_chickens).natAbs + 
                    (sick_frac_piglets * num_piglets).natAbs + 
                    (sick_frac_goats * num_goats).natAbs + 
                    (sick_frac_cows * num_cows).natAbs + 
                    (sick_frac_sheep * num_sheep).natAbs + 
                    (sick_frac_horses * num_horses).natAbs) :
  total_sick = 338 := 
by 
  sorry

end total_sick_animals_l136_136414


namespace range_of_t_l136_136535

variable (f : ℝ → ℝ) (t : ℝ)

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem range_of_t {f : ℝ → ℝ} {t : ℝ} 
  (Hodd : is_odd f) 
  (Hperiodic : ∀ x, f (x + 5 / 2) = -1 / f x) 
  (Hf1 : f 1 ≥ 1) 
  (Hf2014 : f 2014 = (t + 3) / (t - 3)) : 
  0 ≤ t ∧ t < 3 := by
  sorry

end range_of_t_l136_136535


namespace min_value_of_a_l136_136638

theorem min_value_of_a (a : ℝ) : 
  (∀ x ∈ Ioo 1 2, (a * exp x - 1/x) ≥ 0) → a = exp (-1) :=
by
  sorry

end min_value_of_a_l136_136638


namespace decimal_fraction_error_l136_136849

theorem decimal_fraction_error (A B C D E : ℕ) (hA : A < 100) 
    (h10B : 10 * B = A + C) (h10C : 10 * C = 6 * A + D) (h10D : 10 * D = 7 * A + E) 
    (hBCDE_lt_A : B < A ∧ C < A ∧ D < A ∧ E < A) : 
    false :=
sorry

end decimal_fraction_error_l136_136849


namespace fair_coin_difference_l136_136930

noncomputable def binom_coeff (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem fair_coin_difference :
  let p := (1 : ℚ) / 2 in
  let p_1 := binom_coeff 5 4 * p^4 * (1 - p) in
  let p_2 := p^5 in
  p_1 - p_2 = 1 / 8 :=
by
  sorry

end fair_coin_difference_l136_136930


namespace minimum_value_of_a_l136_136642

theorem minimum_value_of_a (a : ℝ) :
  (∀ x ∈ (Ioo 1 2), ae^x - ln x ≥ 0) ↔ a ≥ exp(-1) := sorry

end minimum_value_of_a_l136_136642


namespace oscillation_f_g_max_72_l136_136828

-- Define the given ranges for the functions f and g
def f_range := set.Icc (-8 : ℝ) 4
def g_range := set.Icc (-2 : ℝ) 6

-- The goal is to prove the maximum oscillation of f(x) * g(x) given their ranges
theorem oscillation_f_g_max_72 (f g : ℝ → ℝ) 
  (hf : ∀ x, f(x) ∈ f_range) 
  (hg : ∀ x, g(x) ∈ g_range) : 
  (let product_range := { h | ∃ (x y : ℝ), h = f(x) * g(y) ∧ f(x) ∈ f_range ∧ g(y) ∈ g_range } in
  let min_value := Inf product_range in
  let max_value := Sup product_range in
  max_value - min_value = 72) :=
sorry

end oscillation_f_g_max_72_l136_136828


namespace minimum_value_of_a_l136_136626

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - Real.log x

theorem minimum_value_of_a (a : ℝ) :
  (∀ x ∈ Ioo 1 2, deriv (f a) x ≥ 0) → a ≥ Real.exp ⁻¹ :=
by
  simp [f]
  sorry

end minimum_value_of_a_l136_136626


namespace store_purchase_ways_l136_136741

theorem store_purchase_ways :
  ∃ (headphones sets_hm sets_km keyboards mice : ℕ), 
  headphones = 9 ∧
  mice = 13 ∧
  keyboards = 5 ∧
  sets_km = 4 ∧
  sets_hm = 5 ∧
  (sets_km * headphones + sets_hm * keyboards + headphones * mice * keyboards = 646) :=
by
  use 9, 5, 4, 5, 13
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  sorry

end store_purchase_ways_l136_136741


namespace functional_relationship_and_max_profit_l136_136512

noncomputable def y_of_x (x : ℤ) : ℤ := -2 * x + 320
def profit (x : ℤ) : ℤ := (x - 100) * (y_of_x x)

theorem functional_relationship_and_max_profit :
  (∀ x : ℤ, 100 ≤ x ∧ x ≤ 160 →
    y_of_x 120 = 80 ∧ y_of_x 140 = 40) ∧
  (∃ x : ℤ, 100 ≤ x ∧ x ≤ 160 ∧ 
    x = 130 ∧ profit 130 = 1800) :=
by
  sorry

end functional_relationship_and_max_profit_l136_136512


namespace largest_k_partition_l136_136195

theorem largest_k_partition (k : ℕ) :
  (∀ k, k ≤ 3 → ∀ {n : ℕ} (h_n : n ≥ 15), True) ∧ 
  (∀ k, (k > 3 → ∃ (A : ℕ → Prop) i, (i ≤ k ∧ (∃ x y, (x ∈ A i ∧ y ∈ A i ∧ x ≠ y ∧ x + y = n))) → False)) :=
begin
  sorry
end

end largest_k_partition_l136_136195


namespace minimum_a_for_monotonic_increasing_on_interval_l136_136654

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp x - Real.log x

noncomputable def f_prime (a x : ℝ) : ℝ := a * Real.exp x - 1 / x

def min_value_of_a : ℝ := Real.exp (-1)

theorem minimum_a_for_monotonic_increasing_on_interval :
  (∀ x ∈ Set.Ioo 1 2, 0 ≤ f_prime a x) ↔ a ≥ min_value_of_a :=
by
  sorry

end minimum_a_for_monotonic_increasing_on_interval_l136_136654


namespace exists_hamiltonian_path_l136_136293

theorem exists_hamiltonian_path (n : ℕ) (cities : Fin n → Type) (roads : ∀ (i j : Fin n), cities i → cities j → Prop) 
(road_one_direction : ∀ i j (c1 : cities i) (c2 : cities j), roads i j c1 c2 → ¬ roads j i c2 c1) :
∃ start : Fin n, ∃ path : Fin n → Fin n, ∀ i j : Fin n, i ≠ j → path i ≠ path j :=
sorry

end exists_hamiltonian_path_l136_136293


namespace number_of_ways_to_buy_three_items_l136_136736

def headphones : ℕ := 9
def mice : ℕ := 13
def keyboards : ℕ := 5
def keyboard_mouse_sets : ℕ := 4
def headphone_mouse_sets : ℕ := 5

theorem number_of_ways_to_buy_three_items : 
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 := 
by 
  sorry

end number_of_ways_to_buy_three_items_l136_136736


namespace ratio_of_NH3_to_HNO3_for_2_moles_l136_136453

def balanced_chemical_equation : Prop :=
  ∀ (NH3 HNO3 NH4NO3 : ℕ), NH3 + HNO3 = NH4NO3

theorem ratio_of_NH3_to_HNO3_for_2_moles (NH3 HNO3 NH4NO3 : ℕ)
  (h_equation : balanced_chemical_equation) :
  NH3 = 1 ∧ HNO3 = 1 → (NH3 * 2 : ℕ) / (HNO3 * 2 : ℕ) = 1 := 
by
  intros h_ratio
  let needed_NH3 := NH3 * 2
  let needed_HNO3 := HNO3 * 2
  have h1 : needed_NH3 = 2 := by rw [h_ratio.1, mul_one]
  have h2 : needed_HNO3 = 2 := by rw [h_ratio.2, mul_one]
  rw [h1, h2]
  simp
  sorry

end ratio_of_NH3_to_HNO3_for_2_moles_l136_136453


namespace distance_from_wall_end_l136_136497

theorem distance_from_wall_end
  (wall_width : ℝ) (picture_width : ℝ) (centered : Bool) :
  wall_width = 25 → picture_width = 3 → centered = true → 
  let distance := (wall_width - picture_width) / 2 in
  distance = 11 :=
by
  intros h1 h2 h3
  rw [h1, h2]
  let distance := (25 - 3) / 2
  show distance = 11
  sorry

end distance_from_wall_end_l136_136497


namespace matrix_cubic_l136_136316

noncomputable def matrix_entries (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x, y, z], ![y, z, x], ![z, x, y]]

theorem matrix_cubic (x y z : ℝ) (N : Matrix (Fin 3) (Fin 3) ℝ)
    (hN : N = matrix_entries x y z)
    (hn : N ^ 2 = 2 • (1 : Matrix (Fin 3) (Fin 3) ℝ))
    (hxyz : x * y * z = -2) :
  x^3 + y^3 + z^3 = -6 + 2 * Real.sqrt 2 ∨ x^3 + y^3 + z^3 = -6 - 2 * Real.sqrt 2 :=
by
  sorry

end matrix_cubic_l136_136316


namespace range_of_a_l136_136684

theorem range_of_a (a : ℝ) :
  ¬ ∃ x : ℝ, 2 * x^2 + (a - 1) * x + 1 / 2 ≤ 0 → -1 < a ∧ a < 3 :=
by
  intro h
  sorry

end range_of_a_l136_136684


namespace derivative_of_f_l136_136576

noncomputable def f (x : ℝ) : ℝ := sin x * (cos x + 1)

theorem derivative_of_f (x : ℝ) : (deriv f x) = cos (2 * x) + cos x :=
by
  sorry

end derivative_of_f_l136_136576
